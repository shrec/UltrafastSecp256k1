#ifndef SECP256K1_COINS_COIN_HD_HPP
#define SECP256K1_COINS_COIN_HD_HPP
#pragma once

// ============================================================================
// Coin HD -- BIP-44 Hierarchical Deterministic Derivation per Coin
// ============================================================================
// Wraps BIP-32 derivation with coin-aware path construction:
//   m / purpose' / coin_type' / account' / change / address_index
//   (where ' denotes hardened derivation)
//
// Standard purposes:
//   44' -- P2PKH (BIP-44)
//   49' -- P2SH-P2WPKH (BIP-49)
//   84' -- P2WPKH Native SegWit (BIP-84)
//   86' -- P2TR Taproot (BIP-86)
//
// Usage:
//   auto master = bip32_master_key(seed, 64);
//   auto key = coin_derive_key(master.first, Bitcoin, 0, false, 0);
//   auto addr = coin_address(key.first.public_key(), Bitcoin);
//
//   // Ethereum at m/44'/60'/0'/0/0
//   auto eth_key = coin_derive_key(master.first, Ethereum, 0, false, 0);
//   auto eth_addr = coin_address(eth_key.first.public_key(), Ethereum);
// ============================================================================

#include <string>
#include <cstdint>
#include <utility>
#include "secp256k1/bip32.hpp"
#include "secp256k1/coins/coin_params.hpp"

namespace secp256k1::coins {

// -- BIP-44 Purposes ----------------------------------------------------------

enum class DerivationPurpose : std::uint32_t {
    BIP44 = 44,   // P2PKH (legacy)
    BIP49 = 49,   // P2SH-P2WPKH (nested SegWit)
    BIP84 = 84,   // P2WPKH (native SegWit)
    BIP86 = 86,   // P2TR (Taproot)
};

// -- Path Construction --------------------------------------------------------

// Build BIP-44 derivation path string.
// Example: coin_derive_path(Bitcoin, 0, false, 0) -> "m/44'/0'/0'/0/0"
// Example: coin_derive_path(Ethereum, 0, false, 0, BIP44) -> "m/44'/60'/0'/0/0"
std::string coin_derive_path(const CoinParams& coin,
                             std::uint32_t account = 0,
                             bool change = false,
                             std::uint32_t address_index = 0,
                             DerivationPurpose purpose = DerivationPurpose::BIP44);

// Select best derivation purpose for a coin.
// Bitcoin -> BIP84 (native SegWit), Taproot -> BIP86, Legacy -> BIP44
DerivationPurpose best_purpose(const CoinParams& coin);

// -- Key Derivation -----------------------------------------------------------

// Derive a coin-specific child key from master key.
// Automatically selects the best purpose for the coin.
// Returns {ExtendedKey, success}
std::pair<ExtendedKey, bool>
coin_derive_key(const ExtendedKey& master,
                const CoinParams& coin,
                std::uint32_t account = 0,
                bool change = false,
                std::uint32_t address_index = 0);

// Derive with explicit purpose override.
std::pair<ExtendedKey, bool>
coin_derive_key_with_purpose(const ExtendedKey& master,
                             const CoinParams& coin,
                             DerivationPurpose purpose,
                             std::uint32_t account = 0,
                             bool change = false,
                             std::uint32_t address_index = 0);

// -- Convenience: Seed -> Address ----------------------------------------------

// Full pipeline: seed bytes -> BIP-44 derived address string.
// Returns {address, success}
std::pair<std::string, bool>
coin_address_from_seed(const std::uint8_t* seed, std::size_t seed_len,
                       const CoinParams& coin,
                       std::uint32_t account = 0,
                       std::uint32_t address_index = 0);

} // namespace secp256k1::coins

#endif // SECP256K1_COINS_COIN_HD_HPP
