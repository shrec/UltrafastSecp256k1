#ifndef SECP256K1_RECOVERY_HPP
#define SECP256K1_RECOVERY_HPP

// ============================================================================
// ECDSA Public Key Recovery — secp256k1
// ============================================================================
// Given an ECDSA signature (r, s) and the message hash, recover the public key
// that was used to create the signature. Up to 4 candidates exist; the
// recovery ID (recid = 0..3) selects which one.
//
// BIP-137 / Ethereum use recovery IDs (v) to enable address derivation from
// transaction signatures without transmitting the public key.
//
// Usage:
//   // Sign with recovery
//   auto [sig, recid] = ecdsa_sign_recoverable(msg_hash, private_key);
//
//   // Recover public key
//   auto [pk, ok] = ecdsa_recover(msg_hash, sig, recid);
//   if (ok) { /* pk is the recovered public key */ }
// ============================================================================

#include <array>
#include <cstdint>
#include <utility>
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"

namespace secp256k1 {

using fast::Scalar;
using fast::Point;

// ── Recoverable Signature ────────────────────────────────────────────────────

struct RecoverableSignature {
    ECDSASignature sig;
    int recid;  // 0-3: recovery ID
};

// ── Sign with Recovery ID ────────────────────────────────────────────────────
// Like ecdsa_sign() but also returns the recovery ID needed for recovery.
// recid encodes:
//   bit 0: parity of R.y (0 = even, 1 = odd)
//   bit 1: whether R.x overflowed the curve order (almost never; r = R.x mod n
//           and R.x could be >= n but < p, happens with probability ~2^-128)
RecoverableSignature ecdsa_sign_recoverable(
    const std::array<std::uint8_t, 32>& msg_hash,
    const Scalar& private_key);

// ── Public Key Recovery ──────────────────────────────────────────────────────
// Recovers the public key from a signature and message hash.
// Returns {Point, bool} where bool indicates success.
// Fails if:
//   - r or s is zero
//   - R point is not on the curve
//   - Recovered point is infinity
std::pair<Point, bool> ecdsa_recover(
    const std::array<std::uint8_t, 32>& msg_hash,
    const ECDSASignature& sig,
    int recid);

// ── Compact Recovery Serialization ───────────────────────────────────────────
// 65-byte format: [recid_byte] [r: 32 bytes] [s: 32 bytes]
// recid_byte = 27 + recid + (compressed ? 4 : 0)
std::array<std::uint8_t, 65> recoverable_to_compact(
    const RecoverableSignature& rsig,
    bool compressed = true);

// Parse 65-byte compact recoverable signature
// Returns {RecoverableSignature, bool} where bool indicates success
std::pair<RecoverableSignature, bool> recoverable_from_compact(
    const std::array<std::uint8_t, 65>& data);

} // namespace secp256k1

#endif // SECP256K1_RECOVERY_HPP
