// ============================================================================
// ufsecp/lbtc_gpu_ops.hpp — engine-owned GPU-offload hook contract for the
// seven libbitcoin public-data batch ops (xonly/pubkey/taproot-commitment
// validate + tagged_hash / tagged_hash_var / hash256 / hash256_var).
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
// All seven ops operate on PUBLIC on-chain data (x-only / compressed pubkeys,
// taproot commitment tuples, tagged-hash messages, hash256 preimages). No
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

}  // namespace ufsecp::lbtc::gpu_hook

#endif  // UFSECP_LBTC_GPU_OPS_HPP
