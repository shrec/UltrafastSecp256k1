#ifndef SECP256K1_CHACHA20_POLY1305_HPP
#define SECP256K1_CHACHA20_POLY1305_HPP
#pragma once

// ============================================================================
// ChaCha20-Poly1305 AEAD (RFC 8439)
// ============================================================================
// Used by BIP-324 for authenticated encryption of P2P transport packets.
// Zero external dependencies — built entirely on integer arithmetic.
//
// ChaCha20:  256-bit key, 96-bit nonce, 32-bit counter stream cipher
// Poly1305:  One-time authenticator producing 128-bit tags
// AEAD:      Combines both per RFC 8439 Section 2.8
// ============================================================================

#include <array>
#include <cstdint>
#include <cstddef>

namespace secp256k1 {

// -- ChaCha20 -----------------------------------------------------------------

// Encrypt/decrypt in-place (XOR with keystream). counter starts at given value.
void chacha20_crypt(const std::uint8_t key[32],
                    const std::uint8_t nonce[12],
                    std::uint32_t counter,
                    std::uint8_t* data, std::size_t len) noexcept;

// Generate raw keystream block (64 bytes) for a specific counter value.
void chacha20_block(const std::uint8_t key[32],
                    const std::uint8_t nonce[12],
                    std::uint32_t counter,
                    std::uint8_t out[64]) noexcept;

// -- Poly1305 -----------------------------------------------------------------

// One-shot Poly1305 MAC: tag = Poly1305(key, data)
// key must be 32 bytes (r || s) as produced by ChaCha20 block 0.
std::array<std::uint8_t, 16> poly1305_mac(
    const std::uint8_t key[32],
    const std::uint8_t* data, std::size_t len) noexcept;

// -- ChaCha20-Poly1305 AEAD (RFC 8439) ----------------------------------------

// Encrypt plaintext and produce authentication tag.
// out must have space for plaintext_len bytes.
// tag receives the 16-byte Poly1305 authentication tag.
void aead_chacha20_poly1305_encrypt(
    const std::uint8_t key[32],
    const std::uint8_t nonce[12],
    const std::uint8_t* aad, std::size_t aad_len,
    const std::uint8_t* plaintext, std::size_t plaintext_len,
    std::uint8_t* out,
    std::uint8_t tag[16]) noexcept;

// Decrypt ciphertext and verify authentication tag.
// Returns true on success (tag valid), false on failure (out is zeroed).
bool aead_chacha20_poly1305_decrypt(
    const std::uint8_t key[32],
    const std::uint8_t nonce[12],
    const std::uint8_t* aad, std::size_t aad_len,
    const std::uint8_t* ciphertext, std::size_t ciphertext_len,
    const std::uint8_t tag[16],
    std::uint8_t* out) noexcept;

} // namespace secp256k1

#endif // SECP256K1_CHACHA20_POLY1305_HPP
