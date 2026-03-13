#ifndef SECP256K1_ECIES_HPP
#define SECP256K1_ECIES_HPP
#pragma once

// ============================================================================
// ECIES -- Elliptic Curve Integrated Encryption Scheme
// ============================================================================
// Implements ECIES as used by MetaMask / eth-ecies / bitcore-ecies:
//   encrypt(recipient_pubkey, plaintext) -> ciphertext envelope
//   decrypt(privkey, ciphertext_envelope) -> plaintext
//
// Envelope format (variable length):
//   [33 bytes ephemeral pubkey] [16 bytes IV] [N bytes AES-256-CBC ciphertext]
//   [32 bytes HMAC-SHA256 tag]
//
// Also provides a simpler AES-256-GCM variant:
//   [33 bytes ephemeral pubkey] [12 bytes nonce] [N bytes ciphertext + 16B tag]
//
// Key derivation: SHA-512(ECDH_raw_x) -> first 32 bytes = AES key,
//                                        last 32 bytes  = HMAC key (CBC mode)
// ============================================================================

#include <array>
#include <cstdint>
#include <cstddef>
#include <vector>
#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"

namespace secp256k1 {

// ECIES encrypt: returns envelope bytes
// Returns empty vector on failure
std::vector<std::uint8_t>
ecies_encrypt(const fast::Point& recipient_pubkey,
              const std::uint8_t* plaintext, std::size_t plaintext_len);

// ECIES decrypt: returns plaintext bytes
// Returns empty vector on failure (bad key, tampered ciphertext, etc.)
std::vector<std::uint8_t>
ecies_decrypt(const fast::Scalar& privkey,
              const std::uint8_t* envelope, std::size_t envelope_len);

} // namespace secp256k1

#endif // SECP256K1_ECIES_HPP
