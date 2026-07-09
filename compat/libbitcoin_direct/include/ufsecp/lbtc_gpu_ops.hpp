// ============================================================================
// ufsecp/lbtc_gpu_ops.hpp — engine-owned GPU-offload hook contract for the
// libbitcoin public-data batch ops (xonly/pubkey/taproot-commitment validate
// + tagged_hash / tagged_hash_var / hash256 / hash256_var / merkle_pair_hash).
// ============================================================================
// This is the ONE shared contract between the header-only CPU surface
// (<ufsecp/libbitcoin.hpp>) and the GPU-host provider (src/gpu/src/
// gpu_engine_hook.cpp). It intentionally depends only on <atomic>, <cstdint>
// and <cstddef> — NOT gpu_backend.hpp, NOT any engine header, NOT any gpu::
// symbol — so it can be included from the header-only libbitcoin surface
// without dragging in the GPU host or creating an include cycle.
//
// Design (mirrors the ecdsa/schnorr column hook, relocated to the header-only
// surface because src/cpu/src/batch_verify.cpp is not a writable target here):
//   * The fn-ptr storage lives here as C++17 `inline std::atomic<...>` variables
//     (single shared external-linkage definition, no .cpp needed).
//   * <ufsecp/libbitcoin.hpp> consults each hook (acquire load). When the hook
//     is null (CPU-only build — the GPU host is not linked) OR returns a decline
//     (-1), the header runs its deterministic CPU fallback.
//   * gpu_engine_hook.cpp (compiled only under SECP256K1_LBTC_GPU_OPS, i.e. the
//     direct-GPU profile) installs one trampoline per op, each calling the
//     matching EXISTING GpuBackend virtual and translating GpuError ->
//     {handled / -1 decline}. It self-installs via a file-scope static
//     initializer retained by the same -u secp256k1_gpu_columns_provider_anchor.
//
// Hook return contract:
//   validate ops (xonly / pubkey / commitment):
//     >= 0 -> handled: out_results fully written (1 valid / 0 invalid per row).
//       (a trampoline may return 1 = all-valid, 0 = >=1 invalid; the header
//        recomputes the AND from out_results either way).
//       -1 -> decline (no GPU / non-Ok GpuError / exception) -> CPU fallback.
//   hash ops (tagged_hash / tagged_hash_var / hash256 / hash256_var):
//       0 -> handled: every out32 row written with the correct hash.
//      -1 -> decline -> CPU fallback recomputes every row.
//   An operational backend error is ALWAYS a decline (-1), NEVER an all-zero /
//   consensus-invalid result buffer (fatal-not-invalid).
//
// These hooks operate on PUBLIC on-chain data (x-only / compressed pubkeys,
// taproot commitment tuples, tagged-hash messages, hash256/merkle preimages). No
// secret key, nonce, signing share, or ECDH scalar is ever routed through this
// path — variable-time on both the GPU and CPU sides is the correct choice.
// ============================================================================
#ifndef UFSECP_LBTC_GPU_OPS_HPP
#define UFSECP_LBTC_GPU_OPS_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace ufsecp::lbtc::gpu_hook {

// -- Hook fn-ptr typedefs (arg layouts match the GpuBackend virtuals) --------
// Deliberately NOT declared noexcept: a noexcept trampoline converts implicitly
// to these plain pointer types, keeping the installer flexible.

using xonly_validate_fn = int (*)(const std::uint8_t* keys32, std::size_t count,
                                  std::uint8_t* out_results);

using pubkey_validate_fn = int (*)(const std::uint8_t* pubkeys33, std::size_t count,
                                   std::uint8_t* out_results);

using commitment_verify_fn = int (*)(const std::uint8_t* internal_x32,
                                     const std::uint8_t* tweak32,
                                     const std::uint8_t* tweaked_x32,
                                     const std::uint8_t* parity, std::size_t count,
                                     std::uint8_t* out_results);

using tagged_hash_fn = int (*)(const std::uint8_t* tag_hash32, const std::uint8_t* msgs,
                               std::size_t msg_len, std::size_t count,
                               std::uint8_t* out32);

using tagged_hash_var_fn = int (*)(const std::uint8_t* tag_hash32, const std::uint8_t* msgs,
                                   const std::uint32_t* msg_lens, std::size_t stride,
                                   std::size_t count, std::uint8_t* out32);

using hash256_fn = int (*)(const std::uint8_t* inputs, std::size_t input_len,
                           std::size_t count, std::uint8_t* out32);

using hash256_var_fn = int (*)(const std::uint8_t* inputs, const std::uint32_t* input_lens,
                               std::size_t stride, std::size_t count, std::uint8_t* out32);

using merkle_pair_hash_fn = int (*)(const std::uint8_t* left32,
                                    const std::uint8_t* right32,
                                    std::size_t count,
                                    std::uint8_t* out32);

// -- Shared fn-ptr storage (C++17 inline vars: one definition across all TUs) -
inline std::atomic<xonly_validate_fn>    g_lbtc_xonly_hook{nullptr};
inline std::atomic<pubkey_validate_fn>   g_lbtc_pubkey_hook{nullptr};
inline std::atomic<commitment_verify_fn> g_lbtc_commit_hook{nullptr};
inline std::atomic<tagged_hash_fn>       g_lbtc_tagged_hash_hook{nullptr};
inline std::atomic<tagged_hash_var_fn>   g_lbtc_tagged_hash_var_hook{nullptr};
inline std::atomic<hash256_fn>           g_lbtc_hash256_hook{nullptr};
inline std::atomic<hash256_var_fn>       g_lbtc_hash256_var_hook{nullptr};
inline std::atomic<merkle_pair_hash_fn>  g_lbtc_merkle_pair_hook{nullptr};

// -- Installers (thread-safe store, return the previous value) ---------------
inline xonly_validate_fn install_lbtc_xonly_hook(xonly_validate_fn fn) noexcept {
    return g_lbtc_xonly_hook.exchange(fn, std::memory_order_release);
}
inline pubkey_validate_fn install_lbtc_pubkey_hook(pubkey_validate_fn fn) noexcept {
    return g_lbtc_pubkey_hook.exchange(fn, std::memory_order_release);
}
inline commitment_verify_fn install_lbtc_commit_hook(commitment_verify_fn fn) noexcept {
    return g_lbtc_commit_hook.exchange(fn, std::memory_order_release);
}
inline tagged_hash_fn install_lbtc_tagged_hash_hook(tagged_hash_fn fn) noexcept {
    return g_lbtc_tagged_hash_hook.exchange(fn, std::memory_order_release);
}
inline tagged_hash_var_fn install_lbtc_tagged_hash_var_hook(tagged_hash_var_fn fn) noexcept {
    return g_lbtc_tagged_hash_var_hook.exchange(fn, std::memory_order_release);
}
inline hash256_fn install_lbtc_hash256_hook(hash256_fn fn) noexcept {
    return g_lbtc_hash256_hook.exchange(fn, std::memory_order_release);
}
inline hash256_var_fn install_lbtc_hash256_var_hook(hash256_var_fn fn) noexcept {
    return g_lbtc_hash256_var_hook.exchange(fn, std::memory_order_release);
}

inline merkle_pair_hash_fn install_lbtc_merkle_pair_hook(merkle_pair_hash_fn fn) noexcept {
    return g_lbtc_merkle_pair_hook.exchange(fn, std::memory_order_release);
}

// ----------------------------------------------------------------------------
// GPU telemetry (benchmark/evidence-gathering only -- NOT on any hot path)
// ----------------------------------------------------------------------------
// Minimal backend-identification snapshot. Benchmark harnesses
// (bench_workloads.cpp / bench_public_ops.cpp) query this on demand, purely to
// honestly attribute a hook-active ("production") row to the real
// backend/device that served it, instead of the historical hardcoded
// backend="cpu"/device="n/a". Populated straight from the ALREADY-EXISTING
// secp256k1::gpu::GpuBackend::backend_id() / backend_name() / device_info()
// virtuals (gpu_backend.hpp) -- this header adds no new backend method and
// requires zero edits to gpu_backend.hpp or any *_cuda.cu / *_opencl.cpp /
// *_metal.mm backend file.
//
// driver_version is deliberately NOT part of this struct:
// secp256k1::gpu::DeviceInfo carries no driver field, and adding one is out
// of this change's writable scope. Callers MUST NOT fabricate a driver
// string from this struct -- report driver_version as unavailable (null) in
// any evidence artifact that reads it (see docs/BENCHMARK_POLICY.md).
//
// No production code path (the header-only <ufsecp/libbitcoin.hpp> CPU
// surface) ever calls this hook -- it exists solely for benchmark/evidence
// callers that opt in explicitly, exactly like the per-op hooks above.
struct GpuTelemetry {
    bool          available    = false;  // true iff a backend was probed, init()-ed, and is_ready()
    std::uint32_t backend_id   = 0;      // GpuBackend::backend_id() of the bound backend (1=CUDA,2=OpenCL,3=Metal)
    char          backend_name[32]  = {};  // GpuBackend::backend_name(), NUL-terminated, truncated if longer
    char          device_name[128]  = {};  // DeviceInfo::name of the bound device, NUL-terminated
    std::uint32_t device_index = 0;      // DeviceInfo::device_index of the bound device
};

// Fills *out and returns true only when a real GPU backend is linked,
// initialized, and ready. Returns false (with *out left default/available=false)
// when out is null, no GPU backend is linked, or none could be initialized --
// never fabricates a value on failure.
using gpu_telemetry_fn = bool (*)(GpuTelemetry* out);

inline std::atomic<gpu_telemetry_fn> g_lbtc_gpu_telemetry_hook{nullptr};

inline gpu_telemetry_fn install_lbtc_gpu_telemetry_hook(gpu_telemetry_fn fn) noexcept {
    return g_lbtc_gpu_telemetry_hook.exchange(fn, std::memory_order_release);
}

// ----------------------------------------------------------------------------
// GPU decline diagnostics (benchmark/evidence-gathering only -- NOT on any hot path)
// ----------------------------------------------------------------------------
// Bounded, best-effort explanation of why a hook-active benchmark row declined
// (see bench_workloads.cpp / bench_public_ops.cpp "GPU hook declined" /
// "did not independently handle" sites). Reuses the ALREADY-EXISTING
// GpuBackend::last_error() / last_error_msg() virtuals (gpu_backend.hpp)
// through the SAME cached engine_gpu_backend() probe every op hook above
// dispatches through -- this header adds no new backend method and requires
// zero edits to gpu_backend.hpp or any *_cuda.cu / *_opencl.cpp / *_metal.mm
// backend file.
//
// `message` is diagnostic prose only (may embed an OpenCL/CUDA build log or
// "no GPU backend"); callers MUST NOT parse it for control flow, MUST bound
// how much of it they print/store, and MUST NOT let it change backend/
// evidence_class in any JSON artifact -- companion diagnostic text only, per
// CLAUDE.md's honest-evidence policy. Because the underlying backend keeps a
// single last-error slot shared by every op, this message reflects whichever
// operational failure was recorded MOST RECENTLY on the shared backend
// instance -- for a single-process benchmark run this is normally the
// FIRST extended-kernel failure encountered (later calls typically observe a
// backend-level "previously failed" state rather than re-deriving their own
// per-kernel error), so treat it as "why the shared GPU backend is declining
// right now", not necessarily "why THIS specific op declined".
struct GpuLastError {
    bool          available   = false;  // true iff a backend was probed and queried (false: no backend linked/ready)
    int           code        = 0;      // secp256k1::gpu::GpuError as int (0 == Ok, i.e. no recorded error)
    char          message[256] = {};    // GpuBackend::last_error_msg(), NUL-terminated, truncated if longer
};

// Fills *out and returns true only when a backend was linked and queried
// (regardless of whether that backend currently holds an error). Returns
// false (with *out left default) when out is null or no GPU backend is
// linked -- never fabricates a value.
using gpu_last_error_fn = bool (*)(GpuLastError* out);

inline std::atomic<gpu_last_error_fn> g_lbtc_gpu_last_error_hook{nullptr};

inline gpu_last_error_fn install_lbtc_gpu_last_error_hook(gpu_last_error_fn fn) noexcept {
    return g_lbtc_gpu_last_error_hook.exchange(fn, std::memory_order_release);
}

}  // namespace ufsecp::lbtc::gpu_hook

#endif  // UFSECP_LBTC_GPU_OPS_HPP
