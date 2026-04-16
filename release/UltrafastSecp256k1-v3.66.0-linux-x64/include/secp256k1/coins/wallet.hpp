#ifndef SECP256K1_COINS_WALLET_HPP
#define SECP256K1_COINS_WALLET_HPP
#pragma once

// ============================================================================
// Unified Wallet API -- One interface for all secp256k1 chains
// ============================================================================
//
// Usage:
//
//   using namespace secp256k1::coins;
//
//   // Create wallet for any chain
//   auto btc = wallet::create_random(Bitcoin);
//   auto eth = wallet::create_random(Ethereum);
//   auto trx = wallet::create_random(Tron);
//
//   // Same API regardless of chain
//   auto addr = wallet::get_address(Bitcoin, btc);
//   auto sig  = wallet::sign_message(Bitcoin, btc, msg, msg_len);
//   bool ok   = wallet::verify_message(Bitcoin, btc.pub, msg, msg_len, sig);
//
// Core design:
//   - No chain-specific knowledge leaks to the caller
//   - CoinParams descriptor drives all chain-specific behavior
//   - Thin facade over existing battle-tested implementations
//   - Zero heap allocation in hot paths
//
// ============================================================================

#include <array>
#include <cstdint>
#include <cstddef>
#include <string>
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/recovery.hpp"
#include "secp256k1/coins/coin_params.hpp"

namespace secp256k1::coins::wallet {

// -- Key Types ----------------------------------------------------------------

struct WalletKey {
    fast::Scalar priv;           // 32-byte private scalar
    fast::Point  pub;            // Compressed or uncompressed public key
};

// -- Signature Result (chain-agnostic) ----------------------------------------

struct MessageSignature {
    std::array<std::uint8_t, 32> r;
    std::array<std::uint8_t, 32> s;
    int recid;                   // Recovery ID (0-3)
    std::uint64_t v;             // EIP-155 v value (EVM) or 27+recid (Bitcoin)

    // Convenience: 65-byte compact form [r:32][s:32][v:1]
    std::array<std::uint8_t, 65> to_rsv() const;
};

// -- Key Management -----------------------------------------------------------

// Create a wallet key from raw 32-byte private key
// Returns (key, success). Fails if privkey is zero or >= curve order.
std::pair<WalletKey, bool> from_private_key(const std::uint8_t* priv32);

// Generate address string for a given coin
std::string get_address(const CoinParams& coin, const WalletKey& key,
                        bool testnet = false);

// Generate P2PKH (legacy) address for a coin
std::string get_address_p2pkh(const CoinParams& coin, const WalletKey& key,
                              bool testnet = false);

// Generate P2WPKH (native SegWit) address for a coin
// Returns empty string if coin doesn't support SegWit
std::string get_address_p2wpkh(const CoinParams& coin, const WalletKey& key,
                               bool testnet = false);

// Generate P2SH-P2WPKH (nested SegWit, "3...") address for a coin
// Returns empty string if coin doesn't support SegWit
std::string get_address_p2sh_p2wpkh(const CoinParams& coin, const WalletKey& key,
                                    bool testnet = false);

// Generate P2TR (Taproot) address for a coin
// Returns empty string if coin doesn't support Taproot
std::string get_address_p2tr(const CoinParams& coin, const WalletKey& key,
                             bool testnet = false);

// Generate CashAddr address for a coin (Bitcoin Cash)
// Returns empty string if coin doesn't use CASHADDR encoding
std::string get_address_cashaddr(const CoinParams& coin, const WalletKey& key,
                                 bool testnet = false);

// Export private key in chain-appropriate format:
//   Bitcoin-family: WIF (Base58Check)
//   EVM-family:     0x-prefixed hex
//   Tron:           raw hex (no 0x)
std::string export_private_key(const CoinParams& coin, const WalletKey& key,
                               bool testnet = false);

// Export public key as hex string (compressed for Bitcoin, uncompressed for EVM)
std::string export_public_key_hex(const CoinParams& coin, const WalletKey& key);

// -- Signing ------------------------------------------------------------------

// Sign a message using chain-appropriate format:
//   Bitcoin-family: "\x18Bitcoin Signed Message:\n" + varint(len) + msg -> dSHA256
//   EVM-family:     "\x19Ethereum Signed Message:\n" + decimal(len) + msg -> Keccak
//   Generic:        raw SHA-256(msg) signing
MessageSignature sign_message(const CoinParams& coin, const WalletKey& key,
                              const std::uint8_t* msg, std::size_t msg_len);

// Sign a raw 32-byte hash (no message prefix, no hashing)
MessageSignature sign_hash(const CoinParams& coin, const WalletKey& key,
                           const std::uint8_t* hash32);

// -- Verification -------------------------------------------------------------

// Verify a signed message against a public key
bool verify_message(const CoinParams& coin, const fast::Point& pubkey,
                    const std::uint8_t* msg, std::size_t msg_len,
                    const MessageSignature& sig);

// -- Recovery -----------------------------------------------------------------

// Recover public key from signed message + signature
std::pair<fast::Point, bool>
recover_signer(const CoinParams& coin,
               const std::uint8_t* msg, std::size_t msg_len,
               const MessageSignature& sig);

// Recover address string from signed message + signature
std::pair<std::string, bool>
recover_address(const CoinParams& coin,
                const std::uint8_t* msg, std::size_t msg_len,
                const MessageSignature& sig);

} // namespace secp256k1::coins::wallet

#endif // SECP256K1_COINS_WALLET_HPP
