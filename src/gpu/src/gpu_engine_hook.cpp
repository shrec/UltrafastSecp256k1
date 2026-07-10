/* ============================================================================
 * Engine GPU column-verify provider (self-installing)
 * ============================================================================
 * Bridges secp256k1's engine-owned GpuColumnsVerifyHook (declared in
 * secp256k1/batch_verify.hpp) to a real GPU backend. The engine target
 * (fastsecp256k1) carries NO gpu:: dependency and must never link back to
 * secp256k1_gpu_host -- gpu_host already PRIVATE-links the engine, so a reverse
 * link would be a static-library cycle. Instead this provider lives in the
 * GPU-host layer and SELF-INSTALLS at static-init time, so the unified engine
 * surface (secp256k1::ecdsa_batch_verify_opaque_columns /
 * schnorr_batch_verify_bip340_columns -- the single libbitcoin-direct verify
 * call) transparently accelerates on the GPU whenever this translation unit is
 * linked (i.e. whenever the GPU host is built), and falls back to the CPU column
 * path otherwise. No compile-time macro gates it: enablement is the inspectable
 * build fact "is this provider TU linked".
 *
 * Hook return contract (see secp256k1/batch_verify.hpp):
 *    1 -> handled; out_results fully written; ALL rows valid.
 *    0 -> handled; out_results fully written; >=1 row invalid.
 *   -1 -> declined (no GPU / operational backend error) -> engine CPU fallback.
 * An operational backend error is ALWAYS a decline (-1), never invalid rows
 * (fatal-not-invalid). Each backend method owns memory-bounded chunking, so the
 * full count is safe to pass straight through.
 * ============================================================================ */

#include "secp256k1/batch_verify.hpp"   /* GpuColumnsVerifyHook + installer */
#include "gpu_backend.hpp"              /* GpuBackend, GpuError, backend_ids,
                                          is_available, create_backend (registry).
                                          NOT ufsecp_gpu.h: the libbitcoin-direct
                                          path must carry no C ABI dependency so the
                                          no-CABI build links (acceptance criterion:
                                          "libbitcoin direct entrypoint must not call
                                          the C ABI functions"). */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

/* Link-retention anchor. This TU's only payload is the file-scope
 * EngineGpuColumnsInstaller static initializer below, which has no
 * externally-referenced symbol — so a normal static link of secp256k1_gpu_host
 * discards the whole object and the self-installing hook never runs. Consumers
 * force retention by referencing this exported anchor:
 *   - the ufsecp C ABI (src/cpu/src/ufsecp_gpu_impl.cpp) references it, dragging
 *     this object into libufsecp so the ufsecp/audit path installs the hook;
 *   - the libbitcoin-direct path references no gpu_host symbol, so its executables
 *     force this object in with a linker `--undefined=secp256k1_gpu_columns_provider_anchor`
 *     at their own link (compat/libbitcoin_direct/CMakeLists.txt). Targeted -u
 *     retention is preferred over a blanket WHOLE_ARCHIVE: it pulls only this
 *     provider object, not every backend TU. (WHOLE_ARCHIVE historically also broke
 *     the ZK-less link by dragging in gpu_backend_fallback.o and its undefined
 *     secp256k1::zk:: reference; that TU is now #if SECP256K1_HAS_ZK-gated and
 *     omitted from a ZK-off build entirely — LBTC-GPU-DIRECT-ZK-BLOCKER.)
 * Enablement stays the inspectable build fact "is this provider object linked". */
extern "C" int secp256k1_gpu_columns_provider_anchor = 1;

namespace {

std::mutex g_engine_gpu_backend_mtx;

secp256k1::gpu::GpuBackend* engine_gpu_backend() {  /* call under g_engine_gpu_backend_mtx */
    static std::unique_ptr<secp256k1::gpu::GpuBackend> backend;   /* reused across calls */
    static bool probed = false;
    if (!probed) {
        probed = true;
        uint32_t ids[8] = {};
        const uint32_t n = secp256k1::gpu::backend_ids(ids, 8);
        for (uint32_t i = 0; i < n; ++i) {
            if (!secp256k1::gpu::is_available(ids[i])) continue;
            auto b = secp256k1::gpu::create_backend(ids[i]);
            if (b && b->init(0) == secp256k1::gpu::GpuError::Ok && b->is_ready()) {
                backend = std::move(b);
                break;
            }
        }
    }
    return backend.get();
}

int engine_gpu_columns_hook(int kind, const std::uint8_t* digests32,
        const std::uint8_t* keys, const std::uint8_t* sigs64,
        std::size_t count, std::uint8_t* out_results) noexcept {
    try {
        std::lock_guard<std::mutex> lk(g_engine_gpu_backend_mtx);
        secp256k1::gpu::GpuBackend* b = engine_gpu_backend();
        if (b == nullptr) return -1;  /* no GPU -> CPU fallback */
        const secp256k1::gpu::GpuError e = (kind == 0)
            ? b->ecdsa_verify_lbtc_columns(digests32, keys, sigs64, count, out_results)
            : b->schnorr_verify_lbtc_columns(digests32, keys, sigs64, count, out_results);
        if (e != secp256k1::gpu::GpuError::Ok) return -1;  /* operational error -> CPU fallback (never invalid rows) */
        for (std::size_t i = 0; i < count; ++i) if (out_results[i] == 0) return 0;
        return 1;
    } catch (...) {
        return -1;  /* fall back to CPU; never fabricate a verdict */
    }
}

/* Self-install at load time. When this TU is linked (GPU host built) the engine
 * column entrypoints acquire a non-null hook automatically; a caller may still
 * override with secp256k1::install_gpu_columns_verify_hook(). */
struct EngineGpuColumnsInstaller {
    EngineGpuColumnsInstaller() noexcept {
        secp256k1::install_gpu_columns_verify_hook(&engine_gpu_columns_hook);
    }
};
EngineGpuColumnsInstaller g_engine_gpu_columns_installer;

}  // namespace

/* ============================================================================
 * libbitcoin public-data batch ops — GPU offload trampolines (self-installing)
 * ============================================================================
 * Sibling of the column-verify provider above, for the seven header-only
 * ufsecp::lbtc batch ops (xonly/pubkey/taproot-commitment validate +
 * tagged_hash/tagged_hash_var/hash256/hash256_var). The header CPU surface
 * (<ufsecp/libbitcoin.hpp>) consults engine-owned atomic fn-ptr hooks
 * (<ufsecp/lbtc_gpu_ops.hpp>); these trampolines wire those hooks to the
 * matching EXISTING GpuBackend virtual and translate GpuError -> {handled /
 * -1 decline}. An operational backend error (no device, non-Ok GpuError,
 * exception) is ALWAYS a decline (-1) so the header's deterministic CPU
 * fallback covers the batch — never an all-zero / consensus-invalid buffer.
 *
 * Compiled ONLY in the direct-GPU profile (SECP256K1_LBTC_GPU_OPS, set on
 * secp256k1_gpu_host by compat/libbitcoin_direct/CMakeLists.txt); other GPU
 * builds leave the guard off, so this block imposes no include-path dependency.
 * Reuses engine_gpu_backend() and g_engine_gpu_backend_mtx from the unnamed
 * namespace above (same TU). Retained by the SAME existing -u
 * secp256k1_gpu_columns_provider_anchor — no new anchor. The four HASH
 * trampolines pre-check a length cap before dispatch: tagged_hash,
 * tagged_hash_var, and hash256 enforce the hard on-chip buffer caps of the
 * CUDA overrides (msg_len<=256, stride<=256, input_len<=320) and decline
 * out-of-cap lengths so the CPU covers ALL lengths (no cap divergence).
 * hash256_var's CUDA/OpenCL/Metal kernels stream 64-byte SHA-256 blocks
 * directly from device/global memory instead of copying a full row into a
 * fixed on-chip buffer, so it has no such device-correctness limit; its
 * stride<=kMaxHash256VarStride check below is a policy bound mirroring the
 * ABI layer's cap (src/cpu/src/ufsecp_gpu_impl.cpp), applied here too because
 * this libbitcoin-direct path never goes through that C ABI wrapper.
 * ============================================================================ */
#if defined(SECP256K1_LBTC_GPU_OPS)

#include <ufsecp/lbtc_gpu_ops.hpp>

namespace {

int engine_lbtc_xonly_hook(const std::uint8_t* keys32, std::size_t count,
                           std::uint8_t* out_results) noexcept {
    try {
        std::lock_guard<std::mutex> lk(g_engine_gpu_backend_mtx);
        secp256k1::gpu::GpuBackend* b = engine_gpu_backend();
        if (b == nullptr) return -1;  /* no GPU -> CPU fallback */
        if (b->xonly_validate(keys32, count, out_results) != secp256k1::gpu::GpuError::Ok)
            return -1;                /* operational error -> decline (never invalid rows) */
        for (std::size_t i = 0; i < count; ++i) if (out_results[i] == 0) return 0;
        return 1;
    } catch (...) {
        return -1;
    }
}

int engine_lbtc_pubkey_hook(const std::uint8_t* pubkeys33, std::size_t count,
                            std::uint8_t* out_results) noexcept {
    try {
        std::lock_guard<std::mutex> lk(g_engine_gpu_backend_mtx);
        secp256k1::gpu::GpuBackend* b = engine_gpu_backend();
        if (b == nullptr) return -1;
        if (b->pubkey_validate(pubkeys33, count, out_results) != secp256k1::gpu::GpuError::Ok)
            return -1;
        for (std::size_t i = 0; i < count; ++i) if (out_results[i] == 0) return 0;
        return 1;
    } catch (...) {
        return -1;
    }
}

int engine_lbtc_commit_hook(const std::uint8_t* internal_x32, const std::uint8_t* tweak32,
                            const std::uint8_t* tweaked_x32, const std::uint8_t* parity,
                            std::size_t count, std::uint8_t* out_results) noexcept {
    try {
        std::lock_guard<std::mutex> lk(g_engine_gpu_backend_mtx);
        secp256k1::gpu::GpuBackend* b = engine_gpu_backend();
        if (b == nullptr) return -1;
        if (b->commitment_verify(internal_x32, tweak32, tweaked_x32, parity, count, out_results)
                != secp256k1::gpu::GpuError::Ok)
            return -1;
        for (std::size_t i = 0; i < count; ++i) if (out_results[i] == 0) return 0;
        return 1;
    } catch (...) {
        return -1;
    }
}

int engine_lbtc_tagged_hash_hook(const std::uint8_t* tag_hash32, const std::uint8_t* msgs,
                                 std::size_t msg_len, std::size_t count,
                                 std::uint8_t* out32) noexcept {
    try {
        if (msg_len > 256) return -1;  /* device cap -> CPU covers all lengths */
        std::lock_guard<std::mutex> lk(g_engine_gpu_backend_mtx);
        secp256k1::gpu::GpuBackend* b = engine_gpu_backend();
        if (b == nullptr) return -1;
        if (b->tagged_hash(tag_hash32, msgs, msg_len, count, out32) != secp256k1::gpu::GpuError::Ok)
            return -1;
        return 0;  /* handled: every out32 row written */
    } catch (...) {
        return -1;
    }
}

int engine_lbtc_tagged_hash_var_hook(const std::uint8_t* tag_hash32, const std::uint8_t* msgs,
                                     const std::uint32_t* msg_lens, std::size_t stride,
                                     std::size_t count, std::uint8_t* out32) noexcept {
    try {
        if (stride > 256) return -1;  /* device cap -> CPU covers all lengths */
        std::lock_guard<std::mutex> lk(g_engine_gpu_backend_mtx);
        secp256k1::gpu::GpuBackend* b = engine_gpu_backend();
        if (b == nullptr) return -1;
        if (b->tagged_hash_var(tag_hash32, msgs, msg_lens, stride, count, out32)
                != secp256k1::gpu::GpuError::Ok)
            return -1;
        return 0;
    } catch (...) {
        return -1;
    }
}

int engine_lbtc_hash256_hook(const std::uint8_t* inputs, std::size_t input_len,
                             std::size_t count, std::uint8_t* out32) noexcept {
    try {
        if (input_len > 320) return -1;  /* device cap -> CPU covers all lengths */
        std::lock_guard<std::mutex> lk(g_engine_gpu_backend_mtx);
        secp256k1::gpu::GpuBackend* b = engine_gpu_backend();
        if (b == nullptr) return -1;
        if (b->hash256(inputs, input_len, count, out32) != secp256k1::gpu::GpuError::Ok)
            return -1;
        return 0;
    } catch (...) {
        return -1;
    }
}

/* Hard upper bound on hash256_var's per-row stride for this libbitcoin-direct
 * path, mirroring kMaxHash256VarStride in src/cpu/src/ufsecp_gpu_impl.cpp (4
 * MiB, Bitcoin's block-weight-derived tx size ceiling). Not a device-buffer
 * correctness limit -- the CUDA/OpenCL/Metal hash256_var kernels stream
 * 64-byte blocks straight from device/global memory -- but this path bypasses
 * the C ABI wrapper entirely, so the same policy cap is enforced here too. */
int engine_lbtc_hash256_var_hook(const std::uint8_t* inputs, const std::uint32_t* input_lens,
                                 std::size_t stride, std::size_t count,
                                 std::uint8_t* out32) noexcept {
    try {
        constexpr std::size_t kMaxHash256VarStride = std::size_t{4} * 1024 * 1024;
        if (stride > kMaxHash256VarStride) return -1;  /* policy cap -> CPU fallback */
        std::lock_guard<std::mutex> lk(g_engine_gpu_backend_mtx);
        secp256k1::gpu::GpuBackend* b = engine_gpu_backend();
        if (b == nullptr) return -1;
        if (b->hash256_var(inputs, input_lens, stride, count, out32) != secp256k1::gpu::GpuError::Ok)
            return -1;
        return 0;
    } catch (...) {
        return -1;
    }
}

int engine_lbtc_merkle_pair_hook(const std::uint8_t* left32, const std::uint8_t* right32,
                                 std::size_t count, std::uint8_t* out32) noexcept {
    try {
        // Fixed 64-byte combined input — no device-cap check needed (unlike
        // hash256's input_len<=320 or tagged_hash's msg_len<=256). The kernel
        // always processes exactly 64 bytes per row via lbtc_sha256.
        std::lock_guard<std::mutex> lk(g_engine_gpu_backend_mtx);
        secp256k1::gpu::GpuBackend* b = engine_gpu_backend();
        if (b == nullptr) return -1;
        if (b->merkle_pair_hash(left32, right32, count, out32) != secp256k1::gpu::GpuError::Ok)
            return -1;
        return 0;  /* handled: every out32 row written */
    } catch (...) {
        return -1;
    }
}

int engine_lbtc_sighash_hook(const std::uint8_t* descriptor, std::size_t descriptor_len,
                              const std::uint8_t* const* field_data,
                              const std::uint32_t* field_lengths,
                              const std::uint32_t* const* field_var_lens,
                              std::size_t count, std::uint8_t* out32) noexcept {
    try {
        // Sighash preimages may span multiple megabytes (scriptCode, annex,
        // raw_literal); the CUDA kernel streams 64-byte blocks directly from
        // device global memory, so no per-op device-cap check is needed here.
        // The hook contract is: 0 = handled, -1 = decline -> CPU fallback.
        std::lock_guard<std::mutex> lk(g_engine_gpu_backend_mtx);
        secp256k1::gpu::GpuBackend* b = engine_gpu_backend();
        if (b == nullptr) return -1;  /* no GPU -> CPU fallback */
        if (b->sighash_descriptor_hash(descriptor, descriptor_len,
                                       field_data, field_lengths,
                                       field_var_lens, count, out32)
                != secp256k1::gpu::GpuError::Ok)
            return -1;  /* operational error -> decline (never invalid rows) */
        return 0;  /* handled: every out32 row written */
    } catch (...) {
        return -1;
    }
}

/* Benchmark/evidence-only telemetry trampoline (see lbtc_gpu_ops.hpp
 * GpuTelemetry doc comment). Reuses engine_gpu_backend() -- the SAME cached
 * probe used by every op hook above, under the SAME mutex -- so this reports
 * exactly the backend/device that op hooks above actually dispatch to,
 * queried through already-existing GpuBackend::backend_id() /
 * backend_name() / device_info() virtuals only. No new backend method, no
 * gpu_backend.hpp / *_cuda.cu / *_opencl.cpp / *_metal.mm edit.
 * driver_version is intentionally not sourced here: DeviceInfo carries no
 * driver field, so callers must treat it as unavailable rather than
 * fabricate one (see docs/BENCHMARK_POLICY.md). Never called from any
 * production/hot-path code -- only benchmark harnesses opt in. */
bool engine_lbtc_gpu_telemetry(ufsecp::lbtc::gpu_hook::GpuTelemetry* out) noexcept {
    if (out == nullptr) return false;
    *out = ufsecp::lbtc::gpu_hook::GpuTelemetry{};
    try {
        std::lock_guard<std::mutex> lk(g_engine_gpu_backend_mtx);
        secp256k1::gpu::GpuBackend* b = engine_gpu_backend();
        if (b == nullptr) return false;  /* no GPU -> telemetry unavailable, never fabricated */

        out->backend_id = b->backend_id();
        if (const char* name = b->backend_name()) {
            std::size_t i = 0;
            for (; i + 1 < sizeof(out->backend_name) && name[i] != '\0'; ++i)
                out->backend_name[i] = name[i];
            out->backend_name[i] = '\0';
        }

        /* engine_gpu_backend() always init()s device_index 0 (see above), so
         * device_info(0, ...) queries exactly the bound device. */
        secp256k1::gpu::DeviceInfo di{};
        if (b->device_info(0, di) == secp256k1::gpu::GpuError::Ok) {
            std::size_t i = 0;
            for (; i + 1 < sizeof(out->device_name) && di.name[i] != '\0'; ++i)
                out->device_name[i] = di.name[i];
            out->device_name[i] = '\0';
            out->device_index = di.device_index;
        }

        out->available = true;
        return true;
    } catch (...) {
        *out = ufsecp::lbtc::gpu_hook::GpuTelemetry{};
        return false;
    }
}

/* Decline-diagnostics trampoline (see lbtc_gpu_ops.hpp GpuLastError doc
 * comment). Reuses engine_gpu_backend() -- the SAME cached probe and mutex
 * used by every op hook above -- so this reports the shared backend's most
 * recently recorded operational error via the ALREADY-EXISTING
 * GpuBackend::last_error() / last_error_msg() virtuals. No new backend
 * method, no gpu_backend.hpp / *_cuda.cu / *_opencl.cpp / *_metal.mm edit.
 * Never called from any production/hot-path code -- only benchmark
 * harnesses opt in, exactly like the telemetry trampoline above. */
bool engine_lbtc_gpu_last_error(ufsecp::lbtc::gpu_hook::GpuLastError* out) noexcept {
    if (out == nullptr) return false;
    *out = ufsecp::lbtc::gpu_hook::GpuLastError{};
    try {
        std::lock_guard<std::mutex> lk(g_engine_gpu_backend_mtx);
        secp256k1::gpu::GpuBackend* b = engine_gpu_backend();
        if (b == nullptr) return false;  /* no GPU -> diagnostics unavailable, never fabricated */

        out->code = static_cast<int>(b->last_error());
        if (const char* msg = b->last_error_msg()) {
            std::size_t i = 0;
            for (; i + 1 < sizeof(out->message) && msg[i] != '\0'; ++i)
                out->message[i] = msg[i];
            out->message[i] = '\0';
        }
        out->available = true;
        return true;
    } catch (...) {
        *out = ufsecp::lbtc::gpu_hook::GpuLastError{};
        return false;
    }
}

/* Self-install at load time. Runs when this TU is retained (the direct-GPU
 * profile forces it via the shared -u secp256k1_gpu_columns_provider_anchor). */
struct EngineLbtcOpsInstaller {
    EngineLbtcOpsInstaller() noexcept {
        ufsecp::lbtc::gpu_hook::install_lbtc_xonly_hook(&engine_lbtc_xonly_hook);
        ufsecp::lbtc::gpu_hook::install_lbtc_pubkey_hook(&engine_lbtc_pubkey_hook);
        ufsecp::lbtc::gpu_hook::install_lbtc_commit_hook(&engine_lbtc_commit_hook);
        ufsecp::lbtc::gpu_hook::install_lbtc_tagged_hash_hook(&engine_lbtc_tagged_hash_hook);
        ufsecp::lbtc::gpu_hook::install_lbtc_tagged_hash_var_hook(&engine_lbtc_tagged_hash_var_hook);
        ufsecp::lbtc::gpu_hook::install_lbtc_hash256_hook(&engine_lbtc_hash256_hook);
        ufsecp::lbtc::gpu_hook::install_lbtc_hash256_var_hook(&engine_lbtc_hash256_var_hook);
        ufsecp::lbtc::gpu_hook::install_lbtc_merkle_pair_hook(&engine_lbtc_merkle_pair_hook);
        ufsecp::lbtc::gpu_hook::install_lbtc_sighash_hook(&engine_lbtc_sighash_hook);
        ufsecp::lbtc::gpu_hook::install_lbtc_gpu_telemetry_hook(&engine_lbtc_gpu_telemetry);
        ufsecp::lbtc::gpu_hook::install_lbtc_gpu_last_error_hook(&engine_lbtc_gpu_last_error);
    }
};
EngineLbtcOpsInstaller g_engine_lbtc_ops_installer;

}  // namespace

#endif  // SECP256K1_LBTC_GPU_OPS
