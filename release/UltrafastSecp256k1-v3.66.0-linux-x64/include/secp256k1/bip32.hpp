#ifndef SECP256K1_BIP32_HPP
#define SECP256K1_BIP32_HPP
#pragma once

// ============================================================================
// BIP-32: Hierarchical Deterministic Key Derivation for secp256k1
// ============================================================================
// Implements BIP-32 (HD wallets):
//   - Extended key (xprv / xpub) derivation
//   - Normal child derivation (public derivable)
//   - Hardened child derivation (private only)
//   - Path parsing ("m/44'/0'/0'/0/0")
//
// Reference: https://github.com/bitcoin/bips/blob/master/bip-0032.mediawiki
// ============================================================================

#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"

namespace secp256k1 {

// -- Extended Key -------------------------------------------------------------

// BIP-32 extended key (private or public)
struct ExtendedKey {
    std::array<std::uint8_t, 32> key;         // Private key or compressed X
    std::array<std::uint8_t, 32> chain_code;  // Chain code for derivation
    std::uint8_t depth;                        // 0 for master, increments per level
    std::uint32_t child_number;                // Which child this is
    std::array<std::uint8_t, 4> parent_fingerprint; // First 4 bytes of HASH160(parent pubkey)
    bool is_private;                           // true = xprv, false = xpub
    std::uint8_t pub_prefix = 0;               // 0x02 or 0x03 when is_private==false

    // Derive a child key at index.
    // Hardened key: index >= 0x80000000 (or use derive_hardened())
    // Normal: index < 0x80000000 (or use derive_normal())
    //
    // Returns {valid_child, success}
    std::pair<ExtendedKey, bool> derive_child(std::uint32_t index) const;

    // Convenience wrappers
    std::pair<ExtendedKey, bool> derive_normal(std::uint32_t index) const {
        return derive_child(index);
    }
    std::pair<ExtendedKey, bool> derive_hardened(std::uint32_t index) const {
        return derive_child(index | 0x80000000u);
    }

    // Get the public key point from this extended key
    fast::Point public_key() const;

    // Get the private key scalar (only valid if is_private)
    fast::Scalar private_key() const;

    // Convert to the public extended key (strips private key)
    ExtendedKey to_public() const;

    // Serialize to 78 bytes (BIP-32 standard)
    std::array<std::uint8_t, 78> serialize() const;

    // Key fingerprint: first 4 bytes of HASH160(compressed pubkey)
    std::array<std::uint8_t, 4> fingerprint() const;
};

// -- Master Key Generation ----------------------------------------------------

// Generate master key from seed bytes (BIP-32).
// Seed should be 16-64 bytes (BIP-39 uses 64).
// Returns {ExtendedKey, success}
std::pair<ExtendedKey, bool> bip32_master_key(const std::uint8_t* seed,
                                               std::size_t seed_len);

// -- Path Derivation ----------------------------------------------------------

// Derive key from path string.
// Path format: "m/44'/0'/0'/0/0" (apostrophe = hardened)
// Returns {ExtendedKey, success}
std::pair<ExtendedKey, bool> bip32_derive_path(const ExtendedKey& master,
                                                const std::string& path);

// -- HMAC-SHA512 (needed for BIP-32) ------------------------------------------
// Exposed for testing. Computes HMAC-SHA512(key, data).

std::array<std::uint8_t, 64> hmac_sha512(const std::uint8_t* key, std::size_t key_len,
                                          const std::uint8_t* data, std::size_t data_len);

} // namespace secp256k1

#endif // SECP256K1_BIP32_HPP
