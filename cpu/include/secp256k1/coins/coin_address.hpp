#ifndef SECP256K1_COINS_COIN_ADDRESS_HPP
#define SECP256K1_COINS_COIN_ADDRESS_HPP
#pragma once

// ============================================================================
// Coin Address -- Unified address generation for all secp256k1 coins
// ============================================================================
// Thin wrappers over existing address.hpp functions, parameterized by
// CoinParams. Automatically selects the correct encoding, version bytes,
// and hash algorithm for each coin.
//
// Usage:
//   #include "secp256k1/coins/coin_address.hpp"
//   using namespace secp256k1::coins;
//
//   auto addr = coin_address(pubkey, Bitcoin);           // P2WPKH bc1q...
//   auto addr = coin_address(pubkey, Litecoin);          // ltc1q...
//   auto addr = coin_address(pubkey, Dogecoin);          // D...
//   auto addr = coin_address(pubkey, Ethereum);          // 0x... (EIP-55)
//   auto wif  = coin_wif_encode(privkey, Bitcoin);       // 5... or K/L...
//   auto p2pk = coin_address_p2pkh(pubkey, Dogecoin);    // explicit P2PKH
// ============================================================================

#include <string>
#include <cstdint>
#include "secp256k1/coins/coin_params.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/address.hpp"
#include "secp256k1/context.hpp"

namespace secp256k1::coins {

// -- Default Address (best format for each coin) ------------------------------

// Generate the default/preferred address format for a coin.
// - Bitcoin/Litecoin/DigiByte: Bech32 (P2WPKH)
// - Dogecoin/Dash: Base58Check (P2PKH)
// - Ethereum/BSC/Polygon: EIP-55 hex address
std::string coin_address(const fast::Point& pubkey,
                         const CoinParams& coin,
                         bool testnet = false);

// -- Explicit Address Types ---------------------------------------------------

// P2PKH address with coin-specific version byte
std::string coin_address_p2pkh(const fast::Point& pubkey,
                               const CoinParams& coin,
                               bool testnet = false);

// P2WPKH (SegWit v0) address with coin-specific Bech32 HRP
// Returns empty string if coin doesn't support SegWit
std::string coin_address_p2wpkh(const fast::Point& pubkey,
                                const CoinParams& coin,
                                bool testnet = false);

// P2TR (Taproot) address with coin-specific Bech32m HRP
// Returns empty string if coin doesn't support Taproot
std::string coin_address_p2tr(const fast::Point& internal_key,
                              const CoinParams& coin,
                              bool testnet = false);

// P2SH-P2WPKH (nested/wrapped SegWit) address with coin-specific version byte
// Returns empty string if coin doesn't support SegWit
std::string coin_address_p2sh_p2wpkh(const fast::Point& pubkey,
                                     const CoinParams& coin,
                                     bool testnet = false);

// P2SH address from a 20-byte script hash with coin-specific version byte
std::string coin_address_p2sh(const std::array<std::uint8_t, 20>& script_hash,
                              const CoinParams& coin,
                              bool testnet = false);

// CashAddr address (Bitcoin Cash) with coin-specific prefix
// Returns empty string if coin doesn't use CASHADDR encoding
std::string coin_address_cashaddr(const fast::Point& pubkey,
                                  const CoinParams& coin,
                                  bool testnet = false);

// -- WIF (Wallet Import Format) -----------------------------------------------

// Encode private key as WIF with coin-specific prefix
std::string coin_wif_encode(const fast::Scalar& private_key,
                            const CoinParams& coin,
                            bool compressed = true,
                            bool testnet = false);

// -- Full Key Generation ------------------------------------------------------

// Result of full key generation
struct CoinKeyPair {
    fast::Scalar private_key;
    fast::Point  public_key;
    std::string  address;         // Default format for coin
    std::string  wif;             // WIF-encoded private key (empty for EVM coins)
};

// Generate address from private key in one call (uses default format)
// ctx: optional custom generator context (nullptr = standard secp256k1)
CoinKeyPair coin_derive(const fast::Scalar& private_key,
                        const CoinParams& coin,
                        bool testnet = false,
                        const CurveContext* ctx = nullptr);

} // namespace secp256k1::coins

#endif // SECP256K1_COINS_COIN_ADDRESS_HPP
