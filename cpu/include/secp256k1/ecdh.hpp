#ifndef SECP256K1_ECDH_HPP
#define SECP256K1_ECDH_HPP

// ============================================================================
// ECDH (Elliptic Curve Diffie-Hellman) Key Exchange — secp256k1
// ============================================================================
// Implements ECDH shared secret derivation per SEC 1 v2 §3.3.1.
//
// Shared secret = SHA-256(x-coordinate of sk * PK)
//
// Usage:
//   Scalar sk = ...; // private key
//   Point pk = ...; // other party's public key
//   auto secret = ecdh_compute(sk, pk); // 32-byte shared secret
//
// Raw variant returns just the x-coordinate without hashing:
//   auto raw = ecdh_compute_raw(sk, pk); // 32-byte x-coordinate
// ============================================================================

#include <array>
#include <cstdint>
#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"

namespace secp256k1 {

using fast::Scalar;
using fast::Point;

// ── ECDH Shared Secret (hashed) ──────────────────────────────────────────────
// Computes shared_secret = SHA-256(compressed_point)
// where compressed_point = serialize(sk * PK) as 33-byte compressed encoding.
// Returns 32-byte hash. Returns all-zeros on failure (infinity point).
std::array<std::uint8_t, 32> ecdh_compute(
    const Scalar& private_key,
    const Point& public_key);

// ── ECDH Shared Secret (x-only) ─────────────────────────────────────────────
// Returns SHA-256(x-coordinate of sk * PK) — 32 bytes.
// This is the most common variant (used by libsecp256k1 default).
// Returns all-zeros on failure.
std::array<std::uint8_t, 32> ecdh_compute_xonly(
    const Scalar& private_key,
    const Point& public_key);

// ── ECDH Raw x-coordinate ───────────────────────────────────────────────────
// Returns raw 32-byte x-coordinate of sk * PK (no hashing).
// Useful when caller wants to apply their own KDF.
// Returns all-zeros on failure.
std::array<std::uint8_t, 32> ecdh_compute_raw(
    const Scalar& private_key,
    const Point& public_key);

} // namespace secp256k1

#endif // SECP256K1_ECDH_HPP
