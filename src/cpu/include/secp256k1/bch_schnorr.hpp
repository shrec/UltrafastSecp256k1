#ifndef SECP256K1_BCH_SCHNORR_HPP
#define SECP256K1_BCH_SCHNORR_HPP
#pragma once

// ============================================================================
// Bitcoin Cash 2019 Schnorr -- verification and batch verification
// ============================================================================
// This is NOT BIP-340. The authoritative construction is the 2019-05-15 BCH
// upgrade specification, https://upgradespecs.bitcoincashnode.org/2019-05-15-schnorr/
// (OP_CHECKSIG / OP_CHECKDATASIG with a 64-byte signature).
//
//   sig = r[32] || s[32]          big-endian
//   fail if r >= p or s >= n      (this engine also rejects s == 0, see below)
//   fail if P is the point at infinity
//   e   = SHA256(r || compressed33(P) || msg32) mod n
//   R   = s*G - e*P
//   fail if R = O, if R.y is not a quadratic residue, or if x(R) != r
//
// s == 0: the reference verifier rejects only s >= n. A signature with s == 0
// verifies only if r == x(-e(r)*P), which needs a discrete log, so the extra
// rejection is unobservable on any signature anyone can produce. Recorded in
// docs/SHIM_KNOWN_DIVERGENCES.md.
//
// Every input is public, so every function here is VARIABLE-TIME by design
// (CLAUDE.md CT-VERIFY). Applying ct:: primitives here would be a performance
// bug with no security benefit. Signing lives in the BCHN shim
// (compat/libsecp256k1_bchn_shim/src/shim_schnorr_bch.cpp) and is CT.
//
// Batch verification (bch_schnorr_batch_verify) checks
//     sum_i a_i * (s_i*G - e_i*P_i - R_i) == O
// with a_0 = 1 and 128-bit odd a_i bound to every input and to fresh CSPRNG
// bytes, where R_i is the lift of r_i whose y IS a quadratic residue. Its
// verdict equals the AND of bch_schnorr_verify over the entries (exactly when
// every entry is valid; with an invalid entry the batch rejects with
// probability >= 1 - 2^-127). See src/cpu/src/bch_schnorr.cpp for the proof.
// ============================================================================

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"

namespace secp256k1 {

// One signature to verify, everything still in wire form. The pubkey is the
// 33-byte SEC1 compressed encoding: prefix 0x02/0x03, x < p, x on the curve.
struct BchSchnorrBatchEntry {
    std::array<std::uint8_t, 33> pubkey;     // SEC1 compressed public key
    std::array<std::uint8_t, 32> message;    // 32-byte message hash
    std::array<std::uint8_t, 64> signature;  // r[32] || s[32]
};

// Same, with the public key already parsed by the caller. `pubkey33` MUST be
// the compressed encoding of `point` (it is what e hashes); the verifier does
// not re-derive one from the other. Used by the BCHN shim, whose
// secp256k1_pubkey carries affine (x, y) validated at parse time.
struct BchSchnorrBatchPointEntry {
    fast::Point point;                       // public key, affine
    std::array<std::uint8_t, 33> pubkey33;   // compressed33(point)
    std::array<std::uint8_t, 32> message;
    std::array<std::uint8_t, 64> signature;
};

// -- Single verification -----------------------------------------------------
// One dual multiplication R = s*G - e*P, then NO field inversion and NO square
// root: the residue test is a Jacobi symbol on Y*Z (Jacobian R = (X, Y, Z),
// y = Y/Z^3, and Y/Z^3 = (Y*Z) / Z^4 with Z^4 a square), and the x test is the
// projective r*Z^2 == X.
//
// Returns false on any malformed input (bad prefix, x >= p, off-curve key,
// r >= p, s >= n, s == 0).
bool bch_schnorr_verify(const std::uint8_t sig64[64],
                        const std::uint8_t msg32[32],
                        const std::uint8_t pubkey33[33]) noexcept;

// Pre-parsed key: `P` is the public key, `pubkey33` its compressed encoding.
bool bch_schnorr_verify(const std::uint8_t sig64[64],
                        const std::uint8_t msg32[32],
                        const fast::Point& P,
                        const std::uint8_t pubkey33[33]) noexcept;

// -- Batch verification ------------------------------------------------------
// Returns true iff every entry verifies. n == 0 -> true; n == 1 -> exactly
// bch_schnorr_verify. `seed32` (may be NULL) is caller randomness mixed into
// the weight seed IN ADDITION to the batch binding digest and 32 CSPRNG
// bytes; it can never replace the CSPRNG bytes, so a fixed or hostile seed
// cannot make the weights predictable.
bool bch_schnorr_batch_verify(const BchSchnorrBatchEntry* entries, std::size_t n,
                              const std::uint8_t* seed32 = nullptr);
bool bch_schnorr_batch_verify(const std::vector<BchSchnorrBatchEntry>& entries,
                              const std::uint8_t* seed32 = nullptr);
bool bch_schnorr_batch_verify(const BchSchnorrBatchPointEntry* entries, std::size_t n,
                              const std::uint8_t* seed32 = nullptr);

// Per-item verdicts. out_results[i] = 1 if entry i verifies, else 0 (required
// when n > 0). Runs the batch first; only when it rejects does it fall back to
// verifying every entry individually. An all-valid batch is never rejected, so
// out_results[i] == bch_schnorr_verify(entry i) for every i except with the
// batch's soundness error (<= 2^-127 per call, only when two or more entries
// are invalid). Returns true iff all entries are valid.
bool bch_schnorr_batch_verify_identify(const BchSchnorrBatchEntry* entries,
                                       std::size_t n,
                                       std::uint8_t* out_results,
                                       const std::uint8_t* seed32 = nullptr);
bool bch_schnorr_batch_verify_identify(const BchSchnorrBatchPointEntry* entries,
                                       std::size_t n,
                                       std::uint8_t* out_results,
                                       const std::uint8_t* seed32 = nullptr);

// SEC1 compressed public key parse used by every function above: prefix
// 0x02/0x03, x < p, x^3 + 7 a square. Returns false (out untouched) otherwise.
bool bch_schnorr_pubkey_parse(const std::uint8_t pubkey33[33],
                              fast::Point& out) noexcept;

// -- Test views of the weight derivation --------------------------------------
// The forgery audit module calls these so it checks the engine's OWN seed and
// weight functions rather than a model of them (a model is what let the
// seedless-weight regression of ca0dde78 pass audit/test_batch_randomness.cpp).
// Not for production use.
namespace detail {

// SHA256 binding digest over every entry (and seed32 when non-NULL), before
// the CSPRNG bytes are mixed in. Deterministic.
std::array<std::uint8_t, 32> bch_schnorr_batch_digest_for_test(
    const BchSchnorrBatchEntry* entries, std::size_t n, const std::uint8_t* seed32);

// The full per-call weight seed exactly as the batch verifier derives it:
// the digest above XOR 32 fresh CSPRNG bytes. Differs on every call.
std::array<std::uint8_t, 32> bch_schnorr_batch_seed_for_test(
    const BchSchnorrBatchEntry* entries, std::size_t n, const std::uint8_t* seed32);

// Weight a_index under `seed`: a_0 = 1; a_i = low 128 bits of
// SHA256(seed || i_le32) with bit 0 set.
fast::Scalar bch_schnorr_batch_weight_for_test(
    const std::array<std::uint8_t, 32>& seed, std::uint32_t index);

} // namespace detail

} // namespace secp256k1

#endif // SECP256K1_BCH_SCHNORR_HPP
