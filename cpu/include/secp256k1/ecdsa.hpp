#ifndef SECP256K1_ECDSA_HPP
#define SECP256K1_ECDSA_HPP
#pragma once

// ============================================================================
// ECDSA Sign / Verify for secp256k1
// ============================================================================
// Standard ECDSA (RFC 6979 deterministic nonce recommended for production).
// This implementation provides:
//   - sign(message_hash, private_key) -> (r, s)
//   - verify(message_hash, public_key, r, s) -> bool
//   - Signature normalization (low-S, BIP-62)
//
// WARNING: The nonce k MUST be cryptographically random or deterministic
// (RFC 6979). Reusing k leaks the private key. This library provides a
// deterministic nonce function (RFC 6979) for safety.
// ============================================================================

#include <array>
#include <cstdint>
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"

namespace secp256k1 {

// -- Signature ----------------------------------------------------------------

struct ECDSASignature {
    fast::Scalar r;
    fast::Scalar s;

    // DER encoding (variable length, max 72 bytes)
    // Returns {encoded_bytes, length}
    std::pair<std::array<std::uint8_t, 72>, std::size_t> to_der() const;

    // Compact 64-byte encoding: r (32 bytes) || s (32 bytes)
    std::array<std::uint8_t, 64> to_compact() const;

    // Decode from compact 64-byte encoding
    static ECDSASignature from_compact(const std::uint8_t* data64);
    static ECDSASignature from_compact(const std::array<std::uint8_t, 64>& data);

    // Normalize to low-S form (BIP-62): if s > n/2, replace with n - s
    ECDSASignature normalize() const;

    // Check if signature has low-S
    bool is_low_s() const;
};

// -- ECDSA Operations ---------------------------------------------------------

// Sign a 32-byte message hash with a private key.
// Uses RFC 6979 deterministic nonce generation.
// Returns normalized (low-S) signature.
// Returns {zero, zero} signature on failure (zero key, etc.)
ECDSASignature ecdsa_sign(const std::array<std::uint8_t, 32>& msg_hash,
                          const fast::Scalar& private_key);

// Verify an ECDSA signature against a public key and message hash.
// Accepts both low-S and high-S signatures.
// Raw-pointer overload: avoids 32B array copy when caller has a raw pointer.
bool ecdsa_verify(const std::uint8_t* msg_hash32,
                  const fast::Point& public_key,
                  const ECDSASignature& sig);

// Array overload: thin wrapper.
bool ecdsa_verify(const std::array<std::uint8_t, 32>& msg_hash,
                  const fast::Point& public_key,
                  const ECDSASignature& sig);

// -- RFC 6979 Deterministic Nonce ---------------------------------------------

// Generate deterministic nonce k per RFC 6979.
// Inputs: private key (32 bytes), message hash (32 bytes).
// Output: scalar k suitable for ECDSA signing.
fast::Scalar rfc6979_nonce(const fast::Scalar& private_key,
                           const std::array<std::uint8_t, 32>& msg_hash);

} // namespace secp256k1

#endif // SECP256K1_ECDSA_HPP
