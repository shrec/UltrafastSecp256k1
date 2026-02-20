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
// ============================================================================

#include <array>
#include <cstdint>
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"

namespace secp256k1 {

// ── Schnorr Signature ────────────────────────────────────────────────────────

struct SchnorrSignature {
    std::array<std::uint8_t, 32> r;  // R.x (x-coordinate of nonce point)
    fast::Scalar s;                   // scalar s

    // 64-byte compact encoding: r (32) || s (32)
    std::array<std::uint8_t, 64> to_bytes() const;
    static SchnorrSignature from_bytes(const std::array<std::uint8_t, 64>& data);
};

// ── Pre-computed Schnorr Keypair ──────────────────────────────────────────────
// Equivalent to libsecp256k1's secp256k1_keypair: pre-computes pubkey x-bytes
// and adjusts private key for even-Y, saving 1 gen_mul + 1 inverse per sign.

struct SchnorrKeypair {
    fast::Scalar d;                         // signing key (negated for even Y)
    std::array<std::uint8_t, 32> px;        // x-coordinate bytes of pubkey
};

// Create a pre-computed keypair (call once, then reuse for multiple signs).
SchnorrKeypair schnorr_keypair_create(const fast::Scalar& private_key);

// ── BIP-340 Operations ───────────────────────────────────────────────────────

// Sign using pre-computed keypair (fast: only 1 gen_mul per sign).
SchnorrSignature schnorr_sign(const SchnorrKeypair& kp,
                              const std::array<std::uint8_t, 32>& msg,
                              const std::array<std::uint8_t, 32>& aux_rand);

// Sign from raw private key (convenience: creates keypair internally).
SchnorrSignature schnorr_sign(const fast::Scalar& private_key,
                              const std::array<std::uint8_t, 32>& msg,
                              const std::array<std::uint8_t, 32>& aux_rand);

// Verify a BIP-340 Schnorr signature.
// pubkey_x: 32-byte x-only public key
// msg: 32-byte message
// sig: 64-byte signature
bool schnorr_verify(const std::array<std::uint8_t, 32>& pubkey_x,
                    const std::array<std::uint8_t, 32>& msg,
                    const SchnorrSignature& sig);

// ── Pre-cached X-only Public Key ─────────────────────────────────────────────
// Caches the full Point (avoiding sqrt per verify), similar to libsecp's
// secp256k1_xonly_pubkey which internally stores the cached (x,y) point.

struct SchnorrXonlyPubkey {
    fast::Point point;
    std::array<std::uint8_t, 32> x_bytes;
};

// Parse an x-only pubkey (call once; lift_x + sqrt done here).
// Returns false if the x-coordinate is not on the curve.
bool schnorr_xonly_pubkey_parse(SchnorrXonlyPubkey& out,
                                const std::array<std::uint8_t, 32>& pubkey_x);

// Create from keypair (no sqrt needed — point already known).
SchnorrXonlyPubkey schnorr_xonly_from_keypair(const SchnorrKeypair& kp);

// Verify using pre-cached pubkey (fast: skips lift_x sqrt).
bool schnorr_verify(const SchnorrXonlyPubkey& pubkey,
                    const std::array<std::uint8_t, 32>& msg,
                    const SchnorrSignature& sig);

// ── Tagged Hashing (BIP-340) ─────────────────────────────────────────────────

// H_tag(msg) = SHA256(SHA256(tag) || SHA256(tag) || msg)
std::array<std::uint8_t, 32> tagged_hash(const char* tag,
                                          const void* data, std::size_t len);

// X-only public key from private key (BIP-340: negate if Y is odd)
std::array<std::uint8_t, 32> schnorr_pubkey(const fast::Scalar& private_key);

} // namespace secp256k1

#endif // SECP256K1_SCHNORR_HPP
