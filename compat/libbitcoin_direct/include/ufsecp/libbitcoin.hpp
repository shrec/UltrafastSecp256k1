// ============================================================================
// ufsecp/libbitcoin.hpp — DIRECT C++ integration entry point for libbitcoin.
// ============================================================================
// libbitcoin is a static C++20 build and is making UltrafastSecp256k1 its
// default engine. It does NOT need the libsecp256k1 C-ABI shim, the ufsecp C
// ABI, the C bridge, or any FFI/NuGet layer — those are pure marshalling
// overhead and abstraction for a C++ consumer. This header is the ONE
// integration surface: inline C++ that hands libbitcoin's exact byte layouts
// straight to the engine (secp256k1::*), zero-copy, fully inline-able.
//
// Byte layouts (match libbitcoin / libsecp256k1 exactly):
//   pubkey  : 33-byte compressed (0x02/0x03 || X big-endian)
//   hash    : 32-byte message hash
//   ecdsa   : 64-byte secp256k1_ecdsa_signature == raw scalar limbs, little-
//             endian (r limbs || s limbs). On LE x86 this is byte-identical to
//             libbitcoin's ec_signature (which aliases secp256k1_ecdsa_signature).
//   schnorr : 64-byte BIP-340 signature (R.x big-endian || s big-endian)
//   xonly   : 32-byte x-only public key
//
// Verify paths use variable-time arithmetic (all inputs public) — correct and
// fastest. No secret material is handled here.
// ============================================================================
#ifndef UFSECP_LIBBITCOIN_DIRECT_HPP
#define UFSECP_LIBBITCOIN_DIRECT_HPP

#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/field.hpp"
#include "secp256k1/field_52.hpp"
#include "secp256k1/batch_verify.hpp"
#include "secp256k1/detail/batch_pool.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace ufsecp::lbtc {

namespace detail {

// Lean compressed-pubkey decompress: recover the curve point from 33 bytes
// WITHOUT building per-pubkey GLV verify tables (the batch's own
// dual_scalar_mul builds its GLV) and WITHOUT the shim's pubkey cache. This is
// the libsecp-equivalent cost path: parse X, y = sqrt(x^3+7), pick parity.
[[nodiscard]] inline bool decompress(const std::uint8_t pub33[33],
                                     secp256k1::fast::Point& out) noexcept {
    using secp256k1::fast::FieldElement;
    using secp256k1::fast::FieldElement52;
    using secp256k1::fast::Point;
    if (pub33[0] != 0x02 && pub33[0] != 0x03) return false;
    FieldElement x;
    if (!FieldElement::parse_bytes_strict(pub33 + 1, x)) return false;
    const FieldElement52 x52 = FieldElement52::from_fe(x);
    static const std::uint64_t k7[4] = {7u, 0u, 0u, 0u};
    const FieldElement52 y2  = x52.square() * x52 + FieldElement52::from_4x64_limbs(k7);
    const FieldElement52 y52 = y2.sqrt();
    if (!(y52.square() == y2)) return false;   // X not on curve
    FieldElement y = y52.to_fe();
    if (((y.limbs()[0] & 1u) != 0u) != (pub33[0] == 0x03)) y = FieldElement::zero() - y;
    out = Point::from_affine(x, y);
    return !out.is_infinity();
}

// secp256k1_ecdsa_signature internal layout == ufsecp opaque == raw scalar
// limbs, little-endian. Read one 32-byte half into a Scalar (no reduction; the
// caller's data is canonical, and ecdsa_batch_verify re-checks low-S/range).
[[nodiscard]] inline secp256k1::fast::Scalar opaque_scalar(const std::uint8_t* p) noexcept {
    auto rd = [](const std::uint8_t* q) noexcept {
        std::uint64_t v = 0;
        for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(q[i]) << (i * 8);
        return v;
    };
    return secp256k1::fast::Scalar::from_limbs({rd(p), rd(p + 8), rd(p + 16), rd(p + 24)});
}

[[nodiscard]] inline bool ecdsa_entry(const std::uint8_t* hash32,
                                      const std::uint8_t* pub33,
                                      const std::uint8_t* sig64,
                                      secp256k1::ECDSABatchEntry& out) noexcept {
    std::memcpy(out.msg_hash.data(), hash32, 32);
    if (!decompress(pub33, out.public_key)) return false;
    out.signature = secp256k1::ECDSASignature{opaque_scalar(sig64), opaque_scalar(sig64 + 32)};
    return true;
}

} // namespace detail

// ─── ECDSA single verify ────────────────────────────────────────────────────
// pub33 compressed, hash32, sig64 == secp256k1_ecdsa_signature (opaque LE limbs).
// Returns true iff the signature verifies. Variable-time (public data).
[[nodiscard]] inline bool ecdsa_verify(const std::uint8_t pub33[33],
                                       const std::uint8_t hash32[32],
                                       const std::uint8_t sig64[64]) noexcept {
    secp256k1::fast::Point P;
    if (!detail::decompress(pub33, P)) return false;
    const secp256k1::ECDSASignature sig{detail::opaque_scalar(sig64),
                                        detail::opaque_scalar(sig64 + 32)};
    std::array<std::uint8_t, 32> h;
    std::memcpy(h.data(), hash32, 32);
    return secp256k1::ecdsa_verify(h, P, sig);
}

// ─── ECDSA batch verify ─────────────────────────────────────────────────────
// rows: `count` records of `stride` bytes, each laid out [hash32 | pub33 | sig64]
// (libbitcoin's ec_signature == opaque LE). max_threads: 0=auto, 1=serial, N=cap.
// Returns true iff ALL valid. If out_results != nullptr, writes 1/0 per row
// (fail-closed: any unparsable/invalid row → 0). The all-valid fast path runs the
// engine's multithreaded batch; only a mixed/invalid batch pays the per-row locate.
[[nodiscard]] inline bool ecdsa_verify_batch(const std::uint8_t* rows, std::size_t stride,
                                             std::size_t count, std::uint8_t* out_results,
                                             std::size_t max_threads) {
    if (count == 0) return true;
    std::vector<secp256k1::ECDSABatchEntry> e(count);
    bool parse_ok = true;
    for (std::size_t i = 0; i < count; ++i) {
        const std::uint8_t* row = rows + i * stride;
        if (!detail::ecdsa_entry(row, row + 32, row + 65, e[i])) {
            parse_ok = false;
            if (out_results) out_results[i] = 0u;
        }
    }
    if (parse_ok && secp256k1::ecdsa_batch_verify_mt(e.data(), count, max_threads)) {
        if (out_results) std::memset(out_results, 1, count);
        return true;
    }
    // Mixed/invalid: locate per row (only after the fast path reported a failure).
    bool all = true;
    for (std::size_t i = 0; i < count; ++i) {
        const std::uint8_t* row = rows + i * stride;
        secp256k1::ECDSABatchEntry one{};
        const bool ok = detail::ecdsa_entry(row, row + 32, row + 65, one) &&
                        secp256k1::ecdsa_batch_verify(&one, 1);
        if (out_results) out_results[i] = ok ? 1u : 0u;
        all = all && ok;
    }
    return all;
}

// ─── ECDSA batch verify, COLUMNS (Structure-of-Arrays) ──────────────────────
// libbitcoin's ecdsa::batch holds parallel spans: digests[count][32],
// points[count][33] (compressed), signatures[count][64] (opaque LE limbs).
// max_threads: 0=auto, 1=serial, N=cap. Returns true iff ALL valid; per-row
// results written if out_results != nullptr (fail-closed).
[[nodiscard]] inline bool ecdsa_verify_columns(const std::uint8_t* digests32,
                                               const std::uint8_t* points33,
                                               const std::uint8_t* sigs64,
                                               std::size_t count,
                                               std::uint8_t* out_results,
                                               std::size_t max_threads) {
    if (count == 0) return true;
    // Fused parallel parse+verify: each work-steal chunk decompresses ITS OWN
    // pubkeys and batch-verifies, so the per-pubkey sqrt decompress is NOT a
    // serial prelude (which on a 64-core box would dominate by Amdahl). Mirrors
    // the engine's opaque-rows MT path (persistent pool, warm thread_locals).
    auto& pool = secp256k1::detail::batch_worker_pool();
    const unsigned hw = pool.size();
    const unsigned want = (max_threads == 0) ? hw
        : static_cast<unsigned>(std::min<std::size_t>(max_threads, hw));
    const std::size_t by_work = std::max<std::size_t>(1, count / 128);  // kMinRowsPerThread
    const unsigned nt = static_cast<unsigned>(std::min<std::size_t>(want, by_work));
    const std::size_t steal = (nt <= 1) ? count
        : std::clamp<std::size_t>(count / (static_cast<std::size_t>(nt) * 4), 64, 4096);

    std::atomic<bool> parse_bad{false};
    const bool all = pool.run(count, steal, nt, [&](std::size_t s, std::size_t e) -> bool {
        static thread_local std::vector<secp256k1::ECDSABatchEntry> local;
        local.clear();
        local.resize(e - s);
        for (std::size_t i = s; i < e; ++i) {
            secp256k1::ECDSABatchEntry& en = local[i - s];
            std::memcpy(en.msg_hash.data(), digests32 + i * 32, 32);
            if (!detail::decompress(points33 + i * 33, en.public_key)) {
                parse_bad.store(true, std::memory_order_relaxed);
                return false;
            }
            en.signature = secp256k1::ECDSASignature{
                detail::opaque_scalar(sigs64 + i * 64),
                detail::opaque_scalar(sigs64 + i * 64 + 32)};
        }
        return secp256k1::ecdsa_batch_verify(local.data(), e - s);
    });
    if (all && !parse_bad.load(std::memory_order_relaxed)) {
        if (out_results) std::memset(out_results, 1, count);
        return true;
    }
    // Per-row locate (serial; runs only after the fast path reported a failure).
    bool ok_all = true;
    for (std::size_t i = 0; i < count; ++i) {
        secp256k1::ECDSABatchEntry one{};
        std::memcpy(one.msg_hash.data(), digests32 + i * 32, 32);
        bool ok = detail::decompress(points33 + i * 33, one.public_key);
        if (ok) {
            one.signature = secp256k1::ECDSASignature{
                detail::opaque_scalar(sigs64 + i * 64),
                detail::opaque_scalar(sigs64 + i * 64 + 32)};
            ok = secp256k1::ecdsa_batch_verify(&one, 1);
        }
        if (out_results) out_results[i] = ok ? 1u : 0u;
        ok_all = ok_all && ok;
    }
    return ok_all;
}

// ─── Schnorr (BIP-340) single verify ────────────────────────────────────────
// xonly32 x-only pubkey, msg32, sig64 BIP-340 (R.x big-endian || s big-endian).
[[nodiscard]] inline bool schnorr_verify(const std::uint8_t xonly32[32],
                                         const std::uint8_t msg32[32],
                                         const std::uint8_t sig64[64]) noexcept {
    secp256k1::SchnorrSignature sig;
    if (!secp256k1::SchnorrSignature::parse_strict(sig64, sig)) return false;  // BIP-340 strict
    return secp256k1::schnorr_verify(xonly32, msg32, sig);
}

// ─── Schnorr batch verify ───────────────────────────────────────────────────
// rows: [msg32 | xonly32 | sig64(BIP-340)] @ stride (libbitcoin schnorr::batch layout).
// max_threads: 0=auto, 1=serial, N=cap. Returns true iff ALL valid; per-row results
// written if out_results != nullptr (fail-closed).
[[nodiscard]] inline bool schnorr_verify_batch(const std::uint8_t* rows, std::size_t stride,
                                               std::size_t count, std::uint8_t* out_results,
                                               std::size_t max_threads) {
    if (count == 0) return true;
    std::vector<secp256k1::SchnorrBatchEntry> e(count);
    bool parse_ok = true;
    for (std::size_t i = 0; i < count; ++i) {
        const std::uint8_t* row = rows + i * stride;
        std::memcpy(e[i].message.data(),  row,      32);
        std::memcpy(e[i].pubkey_x.data(), row + 32, 32);
        if (!secp256k1::SchnorrSignature::parse_strict(row + 64, e[i].signature)) {
            parse_ok = false;
            if (out_results) out_results[i] = 0u;
        }
    }
    if (parse_ok && secp256k1::schnorr_batch_verify_mt(e.data(), count, max_threads)) {
        if (out_results) std::memset(out_results, 1, count);
        return true;
    }
    bool all = true;
    for (std::size_t i = 0; i < count; ++i) {
        const std::uint8_t* row = rows + i * stride;
        secp256k1::SchnorrSignature sig;
        const bool ok = secp256k1::SchnorrSignature::parse_strict(row + 64, sig) &&
                        secp256k1::schnorr_verify(row + 32 /*xonly*/, row /*msg*/, sig);
        if (out_results) out_results[i] = ok ? 1u : 0u;
        all = all && ok;
    }
    return all;
}

// ─── Schnorr batch verify, COLUMNS (Structure-of-Arrays) ────────────────────
// libbitcoin's schnorr::batch holds parallel spans: digests[count][32],
// points[count][32] (x-only), signatures[count][64] (BIP-340).
[[nodiscard]] inline bool schnorr_verify_columns(const std::uint8_t* digests32,
                                                 const std::uint8_t* xonly32,
                                                 const std::uint8_t* sigs64,
                                                 std::size_t count,
                                                 std::uint8_t* out_results,
                                                 std::size_t max_threads) {
    if (count == 0) return true;
    auto& pool = secp256k1::detail::batch_worker_pool();
    const unsigned hw = pool.size();
    const unsigned want = (max_threads == 0) ? hw
        : static_cast<unsigned>(std::min<std::size_t>(max_threads, hw));
    const std::size_t by_work = std::max<std::size_t>(1, count / 128);
    const unsigned nt = static_cast<unsigned>(std::min<std::size_t>(want, by_work));
    const std::size_t steal = (nt <= 1) ? count
        : std::clamp<std::size_t>(count / (static_cast<std::size_t>(nt) * 4), 64, 4096);

    std::atomic<bool> parse_bad{false};
    const bool all = pool.run(count, steal, nt, [&](std::size_t s, std::size_t e) -> bool {
        static thread_local std::vector<secp256k1::SchnorrBatchEntry> local;
        local.clear();
        local.resize(e - s);
        for (std::size_t i = s; i < e; ++i) {
            secp256k1::SchnorrBatchEntry& en = local[i - s];
            std::memcpy(en.message.data(),  digests32 + i * 32, 32);
            std::memcpy(en.pubkey_x.data(), xonly32   + i * 32, 32);
            if (!secp256k1::SchnorrSignature::parse_strict(sigs64 + i * 64, en.signature)) {
                parse_bad.store(true, std::memory_order_relaxed);
                return false;
            }
        }
        return secp256k1::schnorr_batch_verify(local.data(), e - s);
    });
    if (all && !parse_bad.load(std::memory_order_relaxed)) {
        if (out_results) std::memset(out_results, 1, count);
        return true;
    }
    bool ok_all = true;
    for (std::size_t i = 0; i < count; ++i) {
        secp256k1::SchnorrSignature sig;
        const bool ok = secp256k1::SchnorrSignature::parse_strict(sigs64 + i * 64, sig) &&
                        secp256k1::schnorr_verify(xonly32 + i * 32, digests32 + i * 32, sig);
        if (out_results) out_results[i] = ok ? 1u : 0u;
        ok_all = ok_all && ok;
    }
    return ok_all;
}

} // namespace ufsecp::lbtc

#endif // UFSECP_LIBBITCOIN_DIRECT_HPP
