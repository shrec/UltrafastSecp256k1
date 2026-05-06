#ifndef SECP256K1_SCHNORR_HPP
#define SECP256K1_SCHNORR_HPP
#pragma once

// ============================================================================
// Schnorr Signatures (BIP-340) for secp256k1
// ============================================================================
// Implements BIP-340 Schnorr signatures:
//   - X-only public keys (32 bytes)
//   - 64-byte signatures (R.x || s)
//   - Tagged hashing per BIP-340 spec
//
// Reference: https://github.com/bitcoin/bips/blob/master/bip-0340.mediawiki
//
// SECP256K1_SIGNING_IS_CT: All signing functions in this header (schnorr_sign,
// schnorr_sign_verified) route R = k'*G through ct::generator_mul_blinded().
// They are constant-time with respect to the private key and nonce.
// Sensitive stack data (d, k, t, nonce_input) is erased before return.
//
// For the canonical CT-validated ABI, prefer ufsecp_schnorr_sign() (C ABI) or
// secp256k1::ct::schnorr_sign() (explicit CT C++ namespace).
// ============================================================================

#include <array>
#include <cstdint>
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"

namespace secp256k1 {

// -- Schnorr Signature --------------------------------------------------------

struct SchnorrSignature {
    std::array<std::uint8_t, 32> r;  // R.x (x-coordinate of nonce point)
    fast::Scalar s;                   // scalar s

    // 64-byte compact encoding: r [32 bytes] concatenated with s [32 bytes]
    std::array<std::uint8_t, 64> to_bytes() const;
    // Parse 64 bytes into SchnorrSignature. s is range-checked: if s >= n,
    // the returned signature has s = Scalar::zero() (always fails verify).
    // BIP-340 requires strict rejection of s >= n at verification.
    // For callers that need explicit parse failure, use parse_strict().
    static SchnorrSignature from_bytes(const std::array<std::uint8_t, 64>& data);
    static SchnorrSignature from_bytes(const std::uint8_t* data64);

    // BIP-340 strict parsing: rejects if r >= p or s >= n or s == 0.
    // Returns false for non-canonical encodings (BIP-340 compliance).
    static bool parse_strict(const std::uint8_t* data64, SchnorrSignature& out) noexcept;
    static bool parse_strict(const std::array<std::uint8_t, 64>& data, SchnorrSignature& out) noexcept;
};

// -- Pre-computed Schnorr Keypair ----------------------------------------------
// Equivalent to libsecp256k1's secp256k1_keypair: pre-computes pubkey x-bytes
// and adjusts private key for even-Y, saving 1 gen_mul + 1 inverse per sign.

struct SchnorrKeypair {
    fast::Scalar d;                         // signing key (negated for even Y)
    std::array<std::uint8_t, 32> px;        // x-coordinate bytes of pubkey
};

// Create a pre-computed keypair (call once, then reuse for multiple signs).
SchnorrKeypair schnorr_keypair_create(const fast::Scalar& private_key);

// -- BIP-340 Operations -------------------------------------------------------

// Sign using pre-computed keypair (fast: only 1 gen_mul per sign).
//
// aux_rand: MUST be 32 bytes of fresh cryptographic randomness (e.g. from
//   OS CSPRNG). Per BIP-340: aux_rand provides synthetic nonce hedging --
//   the signing nonce k is derived as H(d XOR H(aux_rand) || P || m).
//   All-zeros aux_rand makes the nonce fully deterministic (no entropy
//   hedging), which is safe against nonce reuse but not against fault
//   injection or HMAC-state compromise.
//   WARNING: Never reuse aux_rand across different messages with the same
//   key -- while BIP-340 nonces remain safe, unique randomness per sign
//   maximizes defense-in-depth.
//
// NOTE: CT — routes R = k'*G through ct::generator_mul_blinded().
// Constant-time with respect to the private key and nonce.
// Sensitive stack data (d_bytes, t, k_prime, k, nonce_input) is erased
// before return. For the canonical CT-validated ABI, prefer
// ufsecp_schnorr_sign() or secp256k1::ct::schnorr_sign().
[[nodiscard]] SchnorrSignature schnorr_sign(const SchnorrKeypair& kp,
                              const std::array<std::uint8_t, 32>& msg,
                              const std::array<std::uint8_t, 32>& aux_rand);

// Sign + verify (FIPS 186-4 fault attack countermeasure).
// Verifies the produced Schnorr signature before returning it.
// NOTE: CT — routes through ct::generator_mul_blinded (same as schnorr_sign above).
[[nodiscard]] SchnorrSignature schnorr_sign_verified(const SchnorrKeypair& kp,
                                       const std::array<std::uint8_t, 32>& msg,
                                       const std::array<std::uint8_t, 32>& aux_rand);

// Sign from raw private key (convenience: creates keypair internally).
// See above for aux_rand entropy requirements.
// NOTE: CT — same CT guarantees as the keypair overload above.
[[nodiscard]] SchnorrSignature schnorr_sign(const fast::Scalar& private_key,
                              const std::array<std::uint8_t, 32>& msg,
                              const std::array<std::uint8_t, 32>& aux_rand);

// Raw key sign + verify (fault attack countermeasure).
// NOTE: CT — same CT guarantees as the keypair overload above.
[[nodiscard]] SchnorrSignature schnorr_sign_verified(const fast::Scalar& private_key,
                                       const std::array<std::uint8_t, 32>& msg,
                                       const std::array<std::uint8_t, 32>& aux_rand);

// Verify a BIP-340 Schnorr signature.
// pubkey_x: 32-byte x-only public key
// msg: 32-byte message
// sig: 64-byte signature
[[nodiscard]] bool schnorr_verify(const std::uint8_t* pubkey_x32,
                    const std::uint8_t* msg32,
                    const SchnorrSignature& sig) noexcept;

// Array convenience wrappers
[[nodiscard]] bool schnorr_verify(const std::array<std::uint8_t, 32>& pubkey_x,
                    const std::array<std::uint8_t, 32>& msg,
                    const SchnorrSignature& sig) noexcept;

[[nodiscard]] bool schnorr_verify(const std::array<std::uint8_t, 32>& pubkey_x,
                    const std::uint8_t* msg32,
                    const SchnorrSignature& sig) noexcept;

// -- Pre-cached X-only Public Key ---------------------------------------------
// Caches the full Point (avoiding sqrt per verify), similar to libsecp's
// secp256k1_xonly_pubkey which internally stores the cached (x,y) point.

struct SchnorrXonlyPubkey {
    fast::Point point;
    std::array<std::uint8_t, 32> x_bytes;

#if defined(SECP256K1_FAST_52BIT) && !defined(SECP256K1_USE_4X64_POINT_OPS)
    // Cached GLV tables built once by schnorr_xonly_pubkey_parse().
    // tbl_P:       odd multiples [1P, 3P, ..., 15P] (pseudo-affine, shared Z, w=5→8 entries)
    // tbl_phi_base: phi(P) multiples with canonical y (flip applied per-verify)
    // Z_shared:    implicit Z common to all table entries
    // Eliminates ~1,954 ns of build_glv52_table_zr + derive_phi52_table
    // on every schnorr_verify(SchnorrXonlyPubkey, ...) call.
    std::array<fast::AffinePoint52, 8> tbl_P{};
    std::array<fast::AffinePoint52, 8> tbl_phi_base{}; // phi(P) with no flip
    fast::FieldElement52 Z_shared{};
    bool tables_valid = false;
#endif
};

// Parse an x-only pubkey (call once; lift_x + sqrt done here).
// Returns false if the x-coordinate is not on the curve.
bool schnorr_xonly_pubkey_parse(SchnorrXonlyPubkey& out,
                                const std::uint8_t* pubkey_x32);
bool schnorr_xonly_pubkey_parse(SchnorrXonlyPubkey& out,
                                const std::array<std::uint8_t, 32>& pubkey_x);

// Create from keypair (no sqrt needed -- point already known).
SchnorrXonlyPubkey schnorr_xonly_from_keypair(const SchnorrKeypair& kp);

// Verify using pre-cached pubkey (fast: skips lift_x sqrt).
[[nodiscard]] bool schnorr_verify(const SchnorrXonlyPubkey& pubkey,
                    const std::array<std::uint8_t, 32>& msg,
                    const SchnorrSignature& sig) noexcept;

// Raw-pointer msg overload for pre-cached pubkey.
[[nodiscard]] bool schnorr_verify(const SchnorrXonlyPubkey& pubkey,
                    const std::uint8_t* msg32,
                    const SchnorrSignature& sig) noexcept;

// -- Tagged Hashing (BIP-340) -------------------------------------------------

// H_tag: SHA256 of (SHA256(tag) concatenated twice with msg)
std::array<std::uint8_t, 32> tagged_hash(const char* tag,
                                          const void* data, std::size_t len);

// X-only public key from private key (BIP-340: negate if Y is odd)
std::array<std::uint8_t, 32> schnorr_pubkey(const fast::Scalar& private_key);

} // namespace secp256k1

#endif // SECP256K1_SCHNORR_HPP
