#ifndef SECP256K1_BIP39_HPP
#define SECP256K1_BIP39_HPP
#pragma once

// ============================================================================
// BIP-39: Mnemonic Code for Generating Deterministic Keys
// ============================================================================
// Implements BIP-39 mnemonic seed phrase generation and validation:
//   - Entropy (128-256 bits) -> mnemonic word sequence (12-24 words)
//   - Mnemonic + passphrase -> 512-bit seed (PBKDF2-HMAC-SHA512, 2048 rounds)
//   - Mnemonic validation (checksum, word lookup)
//
// The seed output is compatible with BIP-32 master key derivation:
//   auto [mnemonic, ok1] = bip39_generate(16);          // 12 words
//   auto [seed, ok2]     = bip39_mnemonic_to_seed(mnemonic, "");
//   auto [master, ok3]   = bip32_master_key(seed.data(), 64);
//
// Reference: https://github.com/bitcoin/bips/blob/master/bip-0039.mediawiki
// ============================================================================

#include <array>
#include <cstdint>
#include <cstddef>
#include <string>
#include <utility>

namespace secp256k1 {

// -- Mnemonic Generation ------------------------------------------------------

// Generate a BIP-39 mnemonic from entropy.
// entropy_bytes: 16 (12 words), 20 (15), 24 (18), 28 (21), 32 (24)
// Returns {mnemonic_string, success}
// Uses OS CSPRNG for entropy if entropy_in == nullptr.
std::pair<std::string, bool>
bip39_generate(std::size_t entropy_bytes,
               const std::uint8_t* entropy_in = nullptr);

// -- Mnemonic Validation ------------------------------------------------------

// Validate a BIP-39 mnemonic: word count, word membership, checksum.
// Returns true if the mnemonic is valid.
bool bip39_validate(const std::string& mnemonic);

// -- Seed Derivation ----------------------------------------------------------

// Derive 512-bit seed from mnemonic + passphrase.
// Uses PBKDF2-HMAC-SHA512 with 2048 iterations.
// salt = "mnemonic" + passphrase (UTF-8)
// Returns {64-byte seed, success}
std::pair<std::array<std::uint8_t, 64>, bool>
bip39_mnemonic_to_seed(const std::string& mnemonic,
                       const std::string& passphrase = "");

// -- Mnemonic <-> Entropy Roundtrip -------------------------------------------

// Decode mnemonic back to entropy bytes.
// Returns {entropy_bytes, entropy_length, success}
struct Bip39Entropy {
    std::array<std::uint8_t, 32> data{};  // max 256-bit entropy
    std::size_t length = 0;               // actual byte count (16-32)
};
std::pair<Bip39Entropy, bool>
bip39_mnemonic_to_entropy(const std::string& mnemonic);

// -- PBKDF2-HMAC-SHA512 (exposed for testing) --------------------------------

void pbkdf2_hmac_sha512(const std::uint8_t* password, std::size_t password_len,
                         const std::uint8_t* salt, std::size_t salt_len,
                         std::uint32_t iterations,
                         std::uint8_t* output, std::size_t output_len);

// -- Wordlist Access ----------------------------------------------------------

// Get the English BIP-39 wordlist (2048 words, sorted).
const char* const* bip39_wordlist_english();

} // namespace secp256k1

#endif // SECP256K1_BIP39_HPP
