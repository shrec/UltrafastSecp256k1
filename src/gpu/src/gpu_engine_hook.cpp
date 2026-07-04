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
