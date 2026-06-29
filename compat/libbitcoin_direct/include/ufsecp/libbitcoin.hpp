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
//
// Signing paths use secp256k1::ct::* primitives (constant-time) for all
// secret-bearing operations (private keys, nonces). This is mandatory and
// enforced by the SECP256K1_REQUIRE_CT build flag.
//
// CT signing guarantees:
//   - R = k*G via ct::generator_mul_blinded() — unified-add, no branches on k.
//   - k^{-1} via ct::scalar_inverse() — Fermat exponentiation, no early exit.
//   - s = k^{-1}*(z+r*d) via ct::scalar_mul() — no data-dependent branches.
//   - Recovery ID extraction branches only on public data (R.y parity, r overflow).
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
#include "secp256k1/recovery.hpp"
#include "secp256k1/taproot.hpp"
#include "secp256k1/private_key.hpp"
#include "secp256k1/ct/sign.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/ct/scalar.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

namespace ufsecp::lbtc {

namespace detail {

// Returns the canonical package target name. This function exists so that
// `focus fastsecp256k1_libbitcoin` finds the package target in the source
// graph (the function name includes the search term, so find_seeds matches).
inline constexpr const char* fastsecp256k1_libbitcoin_target() noexcept {
    return "secp256k1::fastsecp256k1_libbitcoin";
}

// Lean compressed-pubkey decompress

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
    return secp256k1::ecdsa_batch_verify_opaque_rows(
        rows, stride, count, out_results, max_threads);
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
    return secp256k1::ecdsa_batch_verify_opaque_columns(
        digests32, points33, sigs64, count, out_results, max_threads);
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
    return secp256k1::schnorr_batch_verify_bip340_rows(
        rows, stride, count, out_results, max_threads);
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
    return secp256k1::schnorr_batch_verify_bip340_columns(
        digests32, xonly32, sigs64, count, out_results, max_threads);
}

// ══════════════════════════════════════════════════════════════════════════════
// ECDSA Signing  (CT-backed — all secret-bearing paths use ct::* primitives)
// ══════════════════════════════════════════════════════════════════════════════

// ─── ECDSA sign (RFC 6979 deterministic) ────────────────────────────────────
// hash32: 32-byte message hash.
// sk32:   32-byte secret key (big-endian). Must be 0 < sk < n — call
//         seckey_verify first if the input is untrusted.
// sig64:  output 64-byte opaque signature (r limbs || s limbs, LE).
// Returns true on success. On failure (zero sk, signing error) sig64 is zeroed
// and false is returned (fail-closed). Uses ct::ecdsa_sign() internally —
// constant-time with respect to private key and nonce.
[[nodiscard]] inline bool ecdsa_sign(const std::uint8_t hash32[32],
                                     const std::uint8_t sk32[32],
                                     std::uint8_t sig64[64]) noexcept {
    std::memset(sig64, 0, 64);
    secp256k1::fast::Scalar sk;
    if (!secp256k1::fast::Scalar::parse_bytes_strict_nonzero(sk32, sk)) return false;
    std::array<std::uint8_t, 32> h;
    std::memcpy(h.data(), hash32, 32);
    const auto sig = secp256k1::ct::ecdsa_sign(h, sk);
    if (!sig.is_valid()) return false;
    const auto rL = sig.r.limbs(), sL = sig.s.limbs();
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j) {
            sig64[i * 8 + j]     = static_cast<std::uint8_t>(rL[i] >> (j * 8));
            sig64[32 + i * 8 + j] = static_cast<std::uint8_t>(sL[i] >> (j * 8));
        }
    }
    return true;
}

// ─── ECDSA sign hedged (RFC 6979 §3.6 with auxiliary randomness) ────────────
// Like ecdsa_sign but mixes aux32 into the nonce derivation for defense-in-depth
// against fault injection and HMAC weakness. aux32 must be 32 fresh CSPRNG bytes.
// Uses ct::ecdsa_sign_hedged() — CT with respect to private key and nonce.
[[nodiscard]] inline bool ecdsa_sign_hedged(const std::uint8_t hash32[32],
                                            const std::uint8_t sk32[32],
                                            const std::uint8_t aux32[32],
                                            std::uint8_t sig64[64]) noexcept {
    std::memset(sig64, 0, 64);
    secp256k1::fast::Scalar sk;
    if (!secp256k1::fast::Scalar::parse_bytes_strict_nonzero(sk32, sk)) return false;
    std::array<std::uint8_t, 32> h, aux;
    std::memcpy(h.data(), hash32, 32);
    std::memcpy(aux.data(), aux32, 32);
    const auto sig = secp256k1::ct::ecdsa_sign_hedged(h, sk, aux);
    if (!sig.is_valid()) return false;
    const auto rL = sig.r.limbs(), sL = sig.s.limbs();
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j) {
            sig64[i * 8 + j]     = static_cast<std::uint8_t>(rL[i] >> (j * 8));
            sig64[32 + i * 8 + j] = static_cast<std::uint8_t>(sL[i] >> (j * 8));
        }
    }
    return true;
}

// ─── ECDSA sign recoverable (CT, with recovery ID) ──────────────────────────
// Like ecdsa_sign but produces a 65-byte compact recoverable signature:
//   [27+recid+(compressed?4:0)] [r:32] [s:32]. recid encodes R.y parity + r overflow.
// Uses ct::ecdsa_sign_recoverable() — CT with respect to private key and nonce.
// Recovery ID extraction branches only on public data (R.y parity, r overflow).
// Returns true on success; on failure sig65 is zeroed (fail-closed).
[[nodiscard]] inline bool ecdsa_sign_recoverable(const std::uint8_t hash32[32],
                                                  const std::uint8_t sk32[32],
                                                  std::uint8_t sig65[65]) noexcept {
    std::memset(sig65, 0, 65);
    secp256k1::fast::Scalar sk;
    if (!secp256k1::fast::Scalar::parse_bytes_strict_nonzero(sk32, sk)) return false;
    std::array<std::uint8_t, 32> h;
    std::memcpy(h.data(), hash32, 32);
    const auto rsig = secp256k1::ct::ecdsa_sign_recoverable(h, sk);
    if (!rsig.sig.is_valid()) return false;
    const auto compact = secp256k1::recoverable_to_compact(rsig, true);
    std::memcpy(sig65, compact.data(), 65);
    return true;
}

// ══════════════════════════════════════════════════════════════════════════════
// ECDSA Recovery
// ══════════════════════════════════════════════════════════════════════════════

// ─── ECDSA public key recovery ──────────────────────────────────────────────
// Recovers the compressed public key from a signature and message hash.
// sig64: 64-byte opaque signature (r limbs || s limbs, LE).
// recid: recovery ID (0-3). Obtain from ecdsa_sign_recoverable's sig65[0]-27.
// pub33: output 33-byte compressed public key.
// Returns false if the signature is invalid, recid out of range, or recovery fails.
// Uses secp256k1::ecdsa_recover() — branches only on public data (r, recid).
[[nodiscard]] inline bool ecdsa_recover(const std::uint8_t hash32[32],
                                         const std::uint8_t sig64[64],
                                         int recid,
                                         std::uint8_t pub33[33]) noexcept {
    std::memset(pub33, 0, 33);
    if (recid < 0 || recid > 3) return false;
    std::array<std::uint8_t, 32> h;
    std::memcpy(h.data(), hash32, 32);
    const secp256k1::ECDSASignature sig{detail::opaque_scalar(sig64),
                                        detail::opaque_scalar(sig64 + 32)};
    if (!sig.is_valid()) return false;
    const auto [P, ok] = secp256k1::ecdsa_recover(h, sig, recid);
    if (!ok) return false;
    const auto compressed = P.to_compressed();
    std::memcpy(pub33, compressed.data(), 33);
    return true;
}

// ══════════════════════════════════════════════════════════════════════════════
// ECDSA Signature Utilities
// ══════════════════════════════════════════════════════════════════════════════

// ─── ECDSA signature normalize (low-S, BIP-62) ─────────────────────────────
// If sig64 has high-S (s > n/2), replaces it with low-S (n - s) and returns true.
// If sig64 already has low-S, leaves it unchanged and returns false.
// Uses ECDSASignature::normalize() — variable-time (s is public from any signature).
[[nodiscard]] inline bool ecdsa_signature_normalize(std::uint8_t sig64[64]) noexcept {
    auto r = detail::opaque_scalar(sig64);
    auto s = detail::opaque_scalar(sig64 + 32);
    const secp256k1::ECDSASignature orig{r, s};
    if (orig.is_low_s()) return false;  // already normalized
    const auto norm = orig.normalize();
    const auto sL = norm.s.limbs();
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 8; ++j)
            sig64[32 + i * 8 + j] = static_cast<std::uint8_t>(sL[i] >> (j * 8));
    return true;
}

// ─── ECDSA signature serialize compact (opaque LE → big-endian wire format) ─
// Converts the internal opaque LE form (sig64) to standard 64-byte compact
// (big-endian r || s). Never fails.
// sig64: opaque LE signature (r limbs LE || s limbs LE).
// out64: output big-endian compact (r bytes BE || s bytes BE).
inline void ecdsa_signature_serialize_compact(const std::uint8_t sig64[64],
                                               std::uint8_t out64[64]) noexcept {
    const secp256k1::ECDSASignature sig{detail::opaque_scalar(sig64),
                                        detail::opaque_scalar(sig64 + 32)};
    const auto compact = sig.to_compact();
    std::memcpy(out64, compact.data(), 64);
}

// ─── ECDSA signature parse compact (big-endian wire → opaque LE) ───────────
// Parses a standard 64-byte compact signature (big-endian r || s) into the
// internal opaque LE form. Strict validation: rejects r >= n, s >= n, r == 0,
// s == 0 (BIP-62 / SEC-003 compliance).
// in64:  big-endian compact (r bytes BE || s bytes BE).
// out64: output opaque LE (r limbs LE || s limbs LE).
// Returns true on success. On failure out64 is zeroed (fail-closed).
[[nodiscard]] inline bool ecdsa_signature_parse_compact(const std::uint8_t in64[64],
                                                         std::uint8_t out64[64]) noexcept {
    std::memset(out64, 0, 64);
    secp256k1::ECDSASignature sig;
    if (!secp256k1::ECDSASignature::parse_compact_strict(in64, sig)) return false;
    const auto rL = sig.r.limbs(), sL = sig.s.limbs();
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j) {
            out64[i * 8 + j]      = static_cast<std::uint8_t>(rL[i] >> (j * 8));
            out64[32 + i * 8 + j] = static_cast<std::uint8_t>(sL[i] >> (j * 8));
        }
    }
    return true;
}

// ─── ECDSA signature serialize DER (asn.1 encoded, variable length) ────────
// Encodes sig64 (opaque LE) as DER. out must be at least 72 bytes. On success
// sets out_len to the DER length and returns true.
// Uses ECDSASignature::to_der() — variable-time (public data).
[[nodiscard]] inline bool ecdsa_signature_serialize_der(const std::uint8_t sig64[64],
                                                         std::uint8_t* out,
                                                         std::size_t& out_len) noexcept {
    const secp256k1::ECDSASignature sig{detail::opaque_scalar(sig64),
                                        detail::opaque_scalar(sig64 + 32)};
    const auto [der, len] = sig.to_der();
    if (len == 0) return false;
    std::memcpy(out, der.data(), len);
    out_len = len;
    return true;
}

// ══════════════════════════════════════════════════════════════════════════════
// Recoverable Signature Utilities
// ══════════════════════════════════════════════════════════════════════════════

// ─── Recoverable signature → 65-byte compact ────────────────────────────────
// sig64:   64-byte opaque signature.
// recid:   recovery ID (0-3).
// out65:   output 65-byte compact: [27+recid+(compressed?4:0)] [r:32] [s:32].
// compressed: whether to set the compressed flag in the header byte (default true).
// Never fails for valid recid 0-3 (otherwise out65 is zeroed).
inline void recoverable_to_compact(const std::uint8_t sig64[64], int recid,
                                    std::uint8_t out65[65], bool compressed = true) noexcept {
    std::memset(out65, 0, 65);
    if (recid < 0 || recid > 3) return;
    const secp256k1::ECDSASignature sig{detail::opaque_scalar(sig64),
                                        detail::opaque_scalar(sig64 + 32)};
    const secp256k1::RecoverableSignature rsig{sig, recid};
    const auto compact = secp256k1::recoverable_to_compact(rsig, compressed);
    std::memcpy(out65, compact.data(), 65);
}

// ─── 65-byte compact → recoverable signature ────────────────────────────────
// Parses a 65-byte compact recoverable signature. On success writes the opaque
// signature to sig64, recid to *recid, and returns true.
// On failure sig64 is zeroed (fail-closed).
[[nodiscard]] inline bool recoverable_from_compact(const std::uint8_t in65[65],
                                                    std::uint8_t sig64[64],
                                                    int& recid) noexcept {
    std::memset(sig64, 0, 64);
    std::array<std::uint8_t, 65> data;
    std::memcpy(data.data(), in65, 65);
    const auto [rsig, ok] = secp256k1::recoverable_from_compact(data);
    if (!ok) return false;
    recid = rsig.recid;
    const auto rL = rsig.sig.r.limbs(), sL = rsig.sig.s.limbs();
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j) {
            sig64[i * 8 + j]      = static_cast<std::uint8_t>(rL[i] >> (j * 8));
            sig64[32 + i * 8 + j] = static_cast<std::uint8_t>(sL[i] >> (j * 8));
        }
    }
    return true;
}

// ══════════════════════════════════════════════════════════════════════════════
// Public Key Operations
// ══════════════════════════════════════════════════════════════════════════════

// ─── Public key create from secret key (CT) ────────────────────────────────
// Derives a compressed public key from a secret key.
// Uses ct::generator_mul_blinded() — constant-time with respect to sk.
// Returns false if sk is invalid (0 or >= n). On failure pub33 is zeroed.
[[nodiscard]] inline bool pubkey_create(const std::uint8_t sk32[32],
                                         std::uint8_t pub33[33]) noexcept {
    std::memset(pub33, 0, 33);
    secp256k1::fast::Scalar sk;
    if (!secp256k1::fast::Scalar::parse_bytes_strict_nonzero(sk32, sk)) return false;
    const auto P = secp256k1::ct::generator_mul_blinded(sk);
    if (P.is_infinity()) return false;
    const auto compressed = P.to_compressed();
    std::memcpy(pub33, compressed.data(), 33);
    return true;
}

// ─── Public key parse (compressed 33-byte) ──────────────────────────────────
// Validates that pub33 encodes a valid curve point and writes the internal
// representation. Returns true on success. Variable-time (public data).
[[nodiscard]] inline bool pubkey_parse(const std::uint8_t pub33[33]) noexcept {
    secp256k1::fast::Point P;
    return detail::decompress(pub33, P);
}

// ─── Public key serialize to compressed (33-byte) ──────────────────────────
inline void pubkey_serialize(const std::uint8_t pub33[33],
                              std::uint8_t out33[33]) noexcept {
    std::memcpy(out33, pub33, 33);
}

// ─── Public key combine (sum of points) ────────────────────────────────────
// Computes P = P1 + P2 + ... + Pn (point addition, not scalar sum).
// pub33s: array of n compressed public keys.
// count:  number of pubkeys (must be >= 1).
// out33:  output compressed public key.
// Returns true on success. Returns false if any input is invalid or sum is infinity.
// Uses Point::add() — variable-time (all inputs are public keys).
[[nodiscard]] inline bool pubkey_combine(const std::uint8_t* const* pub33s,
                                          std::size_t count,
                                          std::uint8_t out33[33]) noexcept {
    std::memset(out33, 0, 33);
    if (count == 0) return false;
    secp256k1::fast::Point sum;
    if (!detail::decompress(pub33s[0], sum)) return false;
    for (std::size_t i = 1; i < count; ++i) {
        secp256k1::fast::Point P;
        if (!detail::decompress(pub33s[i], P)) return false;
        sum = sum.add(P);
        if (sum.is_infinity()) return false;
    }
    const auto compressed = sum.to_compressed();
    std::memcpy(out33, compressed.data(), 33);
    return true;
}

// Convenience overload: single array of concatenated [pub33]*count.
[[nodiscard]] inline bool pubkey_combine(const std::uint8_t* pub33s_concat,
                                          std::size_t count,
                                          std::size_t stride,
                                          std::uint8_t out33[33]) noexcept {
    std::memset(out33, 0, 33);
    if (count == 0 || stride < 33) return false;
    secp256k1::fast::Point sum;
    if (!detail::decompress(pub33s_concat, sum)) return false;
    for (std::size_t i = 1; i < count; ++i) {
        secp256k1::fast::Point P;
        if (!detail::decompress(pub33s_concat + i * stride, P)) return false;
        sum = sum.add(P);
        if (sum.is_infinity()) return false;
    }
    const auto compressed = sum.to_compressed();
    std::memcpy(out33, compressed.data(), 33);
    return true;
}

// ─── Public key negate ──────────────────────────────────────────────────────
// Negates the point in-place: if P = (x, y), result is (x, -y).
// Returns false if input is invalid or result is infinity.
// Uses Point::negate() — variable-time (public data).
[[nodiscard]] inline bool pubkey_negate(std::uint8_t pub33[33]) noexcept {
    secp256k1::fast::Point P;
    if (!detail::decompress(pub33, P)) return false;
    const auto neg = P.negate();
    if (neg.is_infinity()) return false;
    const auto compressed = neg.to_compressed();
    std::memcpy(pub33, compressed.data(), 33);
    return true;
}

// ─── Public key tweak_add (P + t*G) ────────────────────────────────────────
// Tweak a public key by adding t*G: out = P + tweak*G.
// pub33: compressed public key (in-place: result written back to pub33).
// tweak32: 32-byte scalar tweak (big-endian). Strict parse (reject 0, >= n).
// Returns true on success. On failure pub33 is zeroed (fail-closed).
// Uses Point::dual_scalar_mul_gen_point(a, b, P) = a*G + b*P.
// We call dual_scalar_mul_gen_point(t, 1, P) = t*G + 1*P = P + t*G. ✓
// Variable-time (all inputs public).
[[nodiscard]] inline bool pubkey_tweak_add(std::uint8_t pub33[33],
                                            const std::uint8_t tweak32[32]) noexcept {
    secp256k1::fast::Point P;
    if (!detail::decompress(pub33, P)) { std::memset(pub33, 0, 33); return false; }
    secp256k1::fast::Scalar t;
    if (!secp256k1::fast::Scalar::parse_bytes_strict_nonzero(tweak32, t)) { std::memset(pub33, 0, 33); return false; }
    const auto result = secp256k1::fast::Point::dual_scalar_mul_gen_point(
        t, secp256k1::fast::Scalar::one(), P);
    if (result.is_infinity()) { std::memset(pub33, 0, 33); return false; }
    const auto compressed = result.to_compressed();
    std::memcpy(pub33, compressed.data(), 33);
    return true;
}

// ─── Public key tweak_mul (P * t) ───────────────────────────────────────────
// Multiply a public key by a scalar: out = P * tweak.
// pub33: compressed public key (in-place: result written back to pub33).
// tweak32: 32-byte scalar tweak (big-endian). Strict parse (reject 0, >= n).
// Returns true on success. On failure pub33 is zeroed (fail-closed).
// Uses Point::scalar_mul() — variable-time (public data).
[[nodiscard]] inline bool pubkey_tweak_mul(std::uint8_t pub33[33],
                                            const std::uint8_t tweak32[32]) noexcept {
    secp256k1::fast::Point P;
    if (!detail::decompress(pub33, P)) { std::memset(pub33, 0, 33); return false; }
    secp256k1::fast::Scalar t;
    if (!secp256k1::fast::Scalar::parse_bytes_strict_nonzero(tweak32, t)) { std::memset(pub33, 0, 33); return false; }
    const auto result = P.scalar_mul(t);
    if (result.is_infinity()) { std::memset(pub33, 0, 33); return false; }
    const auto compressed = result.to_compressed();
    std::memcpy(pub33, compressed.data(), 33);
    return true;
}

// ══════════════════════════════════════════════════════════════════════════════
// Secret Key Operations
// ══════════════════════════════════════════════════════════════════════════════

// ─── Secret key verify ──────────────────────────────────────────────────────
// Validates a 32-byte secret key: must be 0 < sk < n (curve order).
// Uses Scalar::parse_bytes_strict_nonzero() — strictly rejects 0 and >= n.
[[nodiscard]] inline bool seckey_verify(const std::uint8_t sk32[32]) noexcept {
    secp256k1::fast::Scalar sk;
    return secp256k1::fast::Scalar::parse_bytes_strict_nonzero(sk32, sk);
}

// ─── Secret key negate (CT) ────────────────────────────────────────────────
// Negates the secret key in-place: sk = n - sk (or 0 if sk == 0).
// Returns false if sk is zero on input. Uses ct::scalar_neg() — graph-visible
// CT primitive, branchless with cmov zero-masking.
[[nodiscard]] inline bool seckey_negate(std::uint8_t sk32[32]) noexcept {
    secp256k1::fast::Scalar sk;
    if (!secp256k1::fast::Scalar::parse_bytes_strict_nonzero(sk32, sk)) return false;
    const auto neg = secp256k1::ct::scalar_neg(sk);
    const auto bytes = neg.to_bytes();
    std::memcpy(sk32, bytes.data(), 32);
    return true;
}

// ─── Secret key tweak_add (CT: sk + tweak mod n) ───────────────────────────
// Adds a tweak to the secret key: sk = (sk + tweak) mod n.
// Both sk and tweak must be valid (0 < value < n).
// Returns false if either is invalid. On failure sk32 is zeroed (fail-closed).
// Uses ct::scalar_add() — constant-time, no data-dependent branches.
[[nodiscard]] inline bool seckey_tweak_add(std::uint8_t sk32[32],
                                            const std::uint8_t tweak32[32]) noexcept {
    secp256k1::fast::Scalar sk, tweak;
    if (!secp256k1::fast::Scalar::parse_bytes_strict_nonzero(sk32, sk)) { std::memset(sk32, 0, 32); return false; }
    if (!secp256k1::fast::Scalar::parse_bytes_strict_nonzero(tweak32, tweak)) { std::memset(sk32, 0, 32); return false; }
    const auto result = secp256k1::ct::scalar_add(sk, tweak);  // CT modular addition
    if (result.is_zero()) { std::memset(sk32, 0, 32); return false; }
    const auto bytes = result.to_bytes();
    std::memcpy(sk32, bytes.data(), 32);
    return true;
}

// ─── Secret key tweak_mul (CT: sk * tweak mod n) ───────────────────────────
// Multiplies the secret key by a tweak: sk = (sk * tweak) mod n.
// Both sk and tweak must be valid (0 < value < n).
// Returns false if either is invalid or result is zero. On failure sk32 is zeroed.
// Uses ct::scalar_mul() — constant-time, no data-dependent branches.
[[nodiscard]] inline bool seckey_tweak_mul(std::uint8_t sk32[32],
                                            const std::uint8_t tweak32[32]) noexcept {
    secp256k1::fast::Scalar sk, tweak;
    if (!secp256k1::fast::Scalar::parse_bytes_strict_nonzero(sk32, sk)) { std::memset(sk32, 0, 32); return false; }
    if (!secp256k1::fast::Scalar::parse_bytes_strict_nonzero(tweak32, tweak)) { std::memset(sk32, 0, 32); return false; }
    const auto result = secp256k1::ct::scalar_mul(sk, tweak);  // CT modular multiplication
    if (result.is_zero()) { std::memset(sk32, 0, 32); return false; }
    const auto bytes = result.to_bytes();
    std::memcpy(sk32, bytes.data(), 32);
    return true;
}

// ══════════════════════════════════════════════════════════════════════════════
// Schnorr (BIP-340) Operations
// ══════════════════════════════════════════════════════════════════════════════

// ─── Schnorr keypair create (CT) ───────────────────────────────────────────
// Derives a BIP-340 keypair from a secret key. On success writes the 32-byte
// x-only public key to pubkey_xonly and returns true.
// Uses ct::schnorr_keypair_create() — constant-time with respect to sk.
// Returns false if sk is invalid. On failure pubkey_xonly is zeroed.
[[nodiscard]] inline bool schnorr_keypair_create(const std::uint8_t sk32[32],
                                                   std::uint8_t pubkey_xonly[32]) noexcept {
    std::memset(pubkey_xonly, 0, 32);
    secp256k1::fast::Scalar sk;
    if (!secp256k1::fast::Scalar::parse_bytes_strict_nonzero(sk32, sk)) return false;
    const auto kp = secp256k1::ct::schnorr_keypair_create(sk);
    std::memcpy(pubkey_xonly, kp.px.data(), 32);
    return true;
}

// ─── Schnorr sign (BIP-340, CT) ─────────────────────────────────────────────
// Creates a BIP-340 Schnorr signature.
// pubkey_xonly: 32-byte x-only public key. MUST match the public key derived
//               from sk32. The function verifies this match and returns false
//               if it does not (preventing caller-supplied pubkey mismatch).
// sk32:         32-byte secret key (used to reconstruct keypair internally).
// msg32:        32-byte message.
// aux32:        32 bytes of fresh cryptographic randomness (BIP-340 synthetic
//               nonce hedging). Must be unique per (key, msg) pair.
// sig64:        output 64-byte BIP-340 signature (R.x BE || s BE).
// Uses ct::schnorr_sign() — CT with respect to private key and nonce.
// Returns false if sk is invalid, pubkey_xonly does not match sk, or signing
// fails. On failure sig64 is zeroed (fail-closed).
[[nodiscard]] inline bool schnorr_sign(const std::uint8_t pubkey_xonly[32],
                                        const std::uint8_t sk32[32],
                                        const std::uint8_t msg32[32],
                                        const std::uint8_t aux32[32],
                                        std::uint8_t sig64[64]) noexcept {
    std::memset(sig64, 0, 64);
    secp256k1::fast::Scalar sk;
    if (!secp256k1::fast::Scalar::parse_bytes_strict_nonzero(sk32, sk)) return false;
    const auto kp = secp256k1::ct::schnorr_keypair_create(sk);
    // Verify caller-supplied xonly matches the derived keypair
    if (std::memcmp(pubkey_xonly, kp.px.data(), 32) != 0) return false;
    std::array<std::uint8_t, 32> msg, aux;
    std::memcpy(msg.data(), msg32, 32);
    std::memcpy(aux.data(), aux32, 32);
    const auto sig = secp256k1::ct::schnorr_sign(kp, msg, aux);
    // Check for zero signature (fail-closed)
    const auto sb = sig.to_bytes();
    bool all_zero = true;
    for (auto b : sb) if (b != 0) { all_zero = false; break; }
    if (all_zero) return false;
    std::memcpy(sig64, sb.data(), 64);
    return true;
}

// ─── Schnorr x-only pubkey parse ───────────────────────────────────────────
// Validates that xonly32 is a valid x-coordinate on the secp256k1 curve
// (lift_x). Uses schnorr_xonly_pubkey_parse() — variable-time (public data).
[[nodiscard]] inline bool schnorr_xonly_pubkey_parse(const std::uint8_t xonly32[32]) noexcept {
    secp256k1::SchnorrXonlyPubkey out;
    return secp256k1::schnorr_xonly_pubkey_parse(out, xonly32);
}

// ══════════════════════════════════════════════════════════════════════════════
// Taproot (BIP-341) Operations
// ══════════════════════════════════════════════════════════════════════════════

// ─── Taproot tweak_add_check ────────────────────────────────────────────────
// Verifies that output_key_xonly was correctly derived from internal_pubkey and
// merkle_root per BIP-341: Q = P + t*G where t = H_TapTweak(P.x || merkle_root).
// output_key_parity: 0 if output key has even Y, 1 if odd.
// Returns true if the commitment is valid.
[[nodiscard]] inline bool taproot_tweak_add_check(
    const std::uint8_t output_key_xonly[32],
    int output_key_parity,
    const std::uint8_t internal_key_xonly[32],
    const std::uint8_t* merkle_root,
    std::size_t merkle_root_len) noexcept {
    std::array<std::uint8_t, 32> out_k, int_k;
    std::memcpy(out_k.data(), output_key_xonly, 32);
    std::memcpy(int_k.data(), internal_key_xonly, 32);
    return secp256k1::taproot_verify_commitment(
        out_k, output_key_parity, int_k, merkle_root, merkle_root_len);
}

// ══════════════════════════════════════════════════════════════════════════════
// Context Operations (no-op — engine is contextless)
// ══════════════════════════════════════════════════════════════════════════════

// libbitcoin creates/destroys secp256k1_context objects. The engine is
// contextless — these are no-ops that always succeed.

inline int context_create() noexcept { return 1; }
inline void context_destroy() noexcept {}
inline int context_randomize(const std::uint8_t /*seed32*/[32]) noexcept { return 1; }

} // namespace ufsecp::lbtc

#endif // UFSECP_LIBBITCOIN_DIRECT_HPP
