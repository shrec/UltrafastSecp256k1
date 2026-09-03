// sp_scan_batch_impl.hpp — shared scan_batch core for BIP-352 BTC and LTC-SP.
//
// Both protocols use the same pipeline:
//   1. A_sum_i = Σ input_pubkeys per tx
//   2. S_i    = scan_privkey · A_sum_i               (KPlan + batch field-inv)
//   3. S_comp = batch_to_compressed(S_i)
//   4. t_k    = SHA256(tag || tag || S_comp || k_be) (per-output, k is u32 big-endian)
//   5. C_k    = spend_pubkey + t_k · G               (batch_scalar_mul_generator)
//   6. match  = (x-only(C_k) == output_pubkey[k])    (batch x-only extraction)
//
// The only protocol-specific element is the SHA256 domain-separation tag:
//   BIP-352 BTC:  "BIP0352/SharedSecret"   (20 chars)
//   LTC-SP:       "LTCSP/SharedSecret"     (18 chars)
//
// Both feed into a precomputed midstate (sp_tag_midstate) so per-output work
// is a single SHA256 block compression — see sp_scanner.cpp and ltc/ltc_sp.cpp
// for the call sites.

#pragma once

#include "secp256k1/ct/point.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/precompute.hpp"
#include "secp256k1/detail/secure_erase.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <vector>

namespace secp256k1::detail {

// Compute SHA256 midstate after processing tag||tag (64 bytes, one block).
// Caller typically wraps this in a `static const auto` lambda so the midstate
// is computed once per process.
inline std::array<std::uint32_t, 8>
sp_tag_midstate(std::string_view tag) {
    auto t = SHA256::hash(reinterpret_cast<const std::uint8_t*>(tag.data()),
                          tag.size());
    std::uint8_t blk[64];
    std::memcpy(blk,      t.data(), 32);
    std::memcpy(blk + 32, t.data(), 32);
    std::array<std::uint32_t, 8> st = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };
    sha256_compress_dispatch(blk, st.data());
    return st;
}

// Shared scan_batch pipeline.
//
// BatchMatchT must be brace-initialisable from
// {std::uint32_t tx, std::uint32_t k, fast::Scalar privkey} — both
// SilentPaymentScanner::BatchMatch and LtcSpScanner::BatchMatch satisfy this.
template <typename BatchMatchT>
std::vector<BatchMatchT>
sp_scan_batch_impl(const fast::Scalar& scan_privkey,
                   const fast::Point&  spend_pubkey,
                   const fast::Scalar& spend_privkey,
                   const std::array<std::uint32_t, 8>& tag_midstate,
                   const std::vector<std::vector<fast::Point>>& input_pubkeys_per_tx,
                   const std::vector<std::vector<std::array<std::uint8_t, 32>>>& outputs_per_tx)
{
    using fast::Point;
    using fast::Scalar;

    std::vector<BatchMatchT> results;
    const std::size_t N = input_pubkeys_per_tx.size();
    if (N == 0) return results;

    // Thread-local scratch buffers — no heap allocation after first call per
    // thread. Resize-in-place only when N exceeds the previous high-water mark.
    // Per-template-instantiation: BTC and LTC scanners each get their own
    // buffer set, which is fine (small constant memory cost per thread).
    static thread_local std::vector<Point>                       tl_a_sums;
    static thread_local std::vector<Point>                       tl_shared;
    static thread_local std::vector<std::array<std::uint8_t,33>> tl_S_comps;
    static thread_local std::vector<std::uint64_t>               tl_out_map;
    static thread_local std::vector<Scalar>                      tl_t_scalars;
    static thread_local std::vector<Point>                       tl_candidates;
    static thread_local std::vector<std::array<std::uint8_t,32>> tl_x_bytes;

    tl_a_sums.assign(N, Point::infinity());
    for (std::size_t i = 0; i < N; ++i)
        for (const auto& P : input_pubkeys_per_tx[i])
            tl_a_sums[i].add_inplace(P);

    // Stage 1: S_i = scan_privkey × A_sum_i (KPlan + batch field-inv).
    fast::KPlan plan = fast::KPlan::from_scalar(scan_privkey);
    tl_shared.resize(N);
    Point::batch_scalar_mul_fixed_k(plan, tl_a_sums.data(), N, tl_shared.data());

    tl_S_comps.resize(N);
    Point::batch_to_compressed(tl_shared.data(), N, tl_S_comps.data());

    // Pass 2a: compute all t_k scalars via raw SHA256 block compression
    // (no SHA256 context object — direct midstate + 1 block per output).
    tl_out_map.clear();
    tl_t_scalars.clear();

    for (std::uint32_t tx = 0; tx < static_cast<std::uint32_t>(N); ++tx) {
        if (tl_shared[tx].is_infinity()) continue;
        const auto& S_comp = tl_S_comps[tx];

        // Build block1 template: S_comp(33) || k_be(4) || 0x80 || zero-pad ||
        // length-bits-be(64) = exactly 64 bytes. Length = (32+32+33+4)*8 = 808
        // bits = 0x0328. Per-k loop only rewrites bytes 33..36 (k_be).
        std::uint8_t blk[64];
        std::memcpy(blk, S_comp.data(), 33);
        blk[37] = 0x80;
        std::memset(blk + 38, 0, 24);
        blk[62] = 0x03;
        blk[63] = 0x28;

        const auto& outs = outputs_per_tx[tx];
        for (std::uint32_t k = 0; k < static_cast<std::uint32_t>(outs.size()); ++k) {
            blk[33] = std::uint8_t(k >> 24);
            blk[34] = std::uint8_t(k >> 16);
            blk[35] = std::uint8_t(k >>  8);
            blk[36] = std::uint8_t(k);

            std::array<std::uint32_t, 8> h;
            std::memcpy(h.data(), tag_midstate.data(), 32);
            sha256_compress_dispatch(blk, h.data());

            std::array<std::uint8_t, 32> t_bytes;
            for (int b = 0; b < 8; ++b) {
                t_bytes[b*4+0] = std::uint8_t(h[b] >> 24);
                t_bytes[b*4+1] = std::uint8_t(h[b] >> 16);
                t_bytes[b*4+2] = std::uint8_t(h[b] >>  8);
                t_bytes[b*4+3] = std::uint8_t(h[b]);
            }
            tl_out_map.push_back((static_cast<std::uint64_t>(tx) << 32) | k);
            tl_t_scalars.push_back(Scalar::from_bytes(t_bytes));
        }
    }

    if (tl_t_scalars.empty()) return results;
    const std::size_t M = tl_t_scalars.size();

    // Pass 2b: batch t_k·G — one precomputed-table scan over all outputs.
    std::vector<Point> out_jac(M);
    fast::batch_scalar_mul_generator(tl_t_scalars.data(), out_jac.data(), M);

    // Pass 2c: spend_pubkey + t_k·G for all M outputs.
    tl_candidates.resize(M);
    for (std::size_t i = 0; i < M; ++i)
        tl_candidates[i] = spend_pubkey.add(out_jac[i]);

    // Pass 2d: batch x-only extraction — one field-inv (H-trick).
    tl_x_bytes.resize(M);
    Point::batch_x_only_bytes(tl_candidates.data(), M, tl_x_bytes.data());

    // Pass 2e: compare x-coordinates.
    for (std::size_t i = 0; i < M; ++i) {
        std::uint32_t tx = static_cast<std::uint32_t>(tl_out_map[i] >> 32);
        std::uint32_t k  = static_cast<std::uint32_t>(tl_out_map[i] & 0xffffffff);
        if (tl_x_bytes[i] == outputs_per_tx[tx][k])
            results.push_back(BatchMatchT{tx, k, spend_privkey + tl_t_scalars[i]});
    }
    return results;
}

} // namespace secp256k1::detail
