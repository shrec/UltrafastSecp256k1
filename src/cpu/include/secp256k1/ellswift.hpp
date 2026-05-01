#ifndef SECP256K1_ELLSWIFT_HPP
#define SECP256K1_ELLSWIFT_HPP
#pragma once

// ============================================================================
// ElligatorSwift encoding for secp256k1 (BIP-324)
// ============================================================================
// Implements the ElligatorSwift encoding as specified in BIP-324, which uses
// a variant of the Elligator Squared technique to encode secp256k1 public keys
// as uniformly random-looking 64-byte strings.
//
// The encoding is:
//   ellswift_encode(pubkey) -> 64 bytes (u || t), indistinguishable from random
//   ellswift_decode(64 bytes) -> pubkey (x-only)
//
// For BIP-324 ECDH:
//   ellswift_xdh(our_ell64, their_ell64, our_privkey, initiator) -> 32-byte secret
//
// Reference: BIP-324, https://github.com/bitcoin/bips/blob/master/bip-0324.mediawiki
// ============================================================================

#include <array>
#include <cstdint>
#include <cstddef>
#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/field.hpp"

namespace secp256k1 {

using fast::Scalar;
using fast::Point;
using fast::FieldElement;

// -- ElligatorSwift encoding/decoding -----------------------------------------

// Decode a 64-byte ElligatorSwift encoding to an x-coordinate.
// Returns the x-coordinate of the encoded point.
FieldElement ellswift_decode(const std::uint8_t encoding[64]) noexcept;

// Create a 64-byte ElligatorSwift encoding from a private key.
// Generates a uniformly random-looking 64 bytes that encodes privkey * G.
// Uses OS CSPRNG for the randomness needed by the encoding.
std::array<std::uint8_t, 64> ellswift_create(const Scalar& privkey);

// Auxrnd variant: mixes auxrnd32 into the encoding RNG (matches libsecp256k1
// secp256k1_ellswift_create auxrnd32 semantics). When auxrnd32 is NULL,
// falls back to pure CSPRNG (identical to the single-argument overload).
// Callers SHOULD pass a fresh 32-byte random value so that the same private
// key produces a different encoding on each call — required by BIP-324 to
// prevent key-identity leakage across connections.
std::array<std::uint8_t, 64> ellswift_create(const Scalar& privkey,
                                              const std::uint8_t* auxrnd32);

// Fast variant: uses the precomputed fixed-base table (non-CT generator mul).
// Suitable for ephemeral keys (BIP-324 session keys) where CT is not required.
std::array<std::uint8_t, 64> ellswift_create_fast(const Scalar& privkey);

// Fast XDH variant: uses non-CT variable-base scalar_mul for the ECDH step.
// Suitable for BIP-324 ephemeral session keys where CT is not required.
std::array<std::uint8_t, 32> ellswift_xdh_fast(
    const std::uint8_t ell_a64[64],
    const std::uint8_t ell_b64[64],
    const Scalar& our_privkey,
    bool initiating) noexcept;

// Encode an existing x-coordinate as a 64-byte ElligatorSwift encoding.
// rnd32: 32 bytes of randomness that determine which of the many valid
//        encodings is selected. Must not be a deterministic function of x.
// Used by secp256k1_ellswift_encode (encode an existing pubkey).
std::array<std::uint8_t, 64> ellswift_encode_x(const FieldElement& x,
                                                const std::uint8_t rnd32[32]);

// -- ElligatorSwift ECDH (BIP-324) --------------------------------------------

// Perform x-only ECDH using ElligatorSwift-encoded public keys.
// Both ell_a64 and ell_b64 are 64-byte ElligatorSwift encodings.
// our_privkey is our secret key.
// initiating: true if we are the connection initiator (determines key order).
// Returns 32-byte shared secret (SHA256-based).
std::array<std::uint8_t, 32> ellswift_xdh(
    const std::uint8_t ell_a64[64],
    const std::uint8_t ell_b64[64],
    const Scalar& our_privkey,
    bool initiating) noexcept;

} // namespace secp256k1

#endif // SECP256K1_ELLSWIFT_HPP
