#ifndef SECP256K1_COINS_COIN_PARAMS_HPP
#define SECP256K1_COINS_COIN_PARAMS_HPP
#pragma once

// ============================================================================
// Coin Parameters -- constexpr definitions for all secp256k1-based coins
// ============================================================================
// Each CoinParams holds:
//   - Network prefixes (P2PKH version byte, WIF prefix)
//   - Bech32 HRP (human-readable part)
//   - BIP-44 coin_type
//   - Address hash algorithm (HASH160 vs Keccak-256)
//   - Feature flags (supports_segwit, supports_taproot, etc.)
//   - Display name + ticker
//
// All data is constexpr -- zero runtime cost, no heap allocation.
// ============================================================================

#include <cstdint>
#include <array>
#include <iterator>

namespace secp256k1::coins {

// -- Address Hash Algorithm ---------------------------------------------------

enum class AddressHash : std::uint8_t {
    HASH160,     // RIPEMD160(SHA256(pubkey)) -- Bitcoin, Litecoin, etc.
    KECCAK256,   // Keccak-256(pubkey[1:]) -- Ethereum, BNB, etc.
    BLAKE2B_160, // BLAKE2b-160 -- reserved for future use
};

// -- Address Encoding ---------------------------------------------------------

enum class AddressEncoding : std::uint8_t {
    BASE58CHECK, // Base58Check (P2PKH, P2SH)
    BECH32,      // Bech32 / Bech32m (SegWit)
    EIP55,       // Ethereum EIP-55 mixed-case checksum
    CASHADDR,    // CashAddr (Bitcoin Cash)
    TRON_BASE58, // Tron: Keccak-256(pubkey) + 0x41 prefix + Base58Check
};

// -- Feature Flags ------------------------------------------------------------

struct CoinFeatures {
    bool supports_segwit   : 1;  // SegWit (P2WPKH, Bech32)
    bool supports_taproot  : 1;  // Taproot (P2TR, Bech32m)
    bool supports_p2sh     : 1;  // P2SH (pay-to-script-hash)
    bool compressed_only   : 1;  // Only compressed pubkeys allowed
    bool uses_evm          : 1;  // EVM-compatible (Ethereum-like)
    bool uses_schnorr      : 1;  // Native Schnorr support (BIP-340)
};

// -- Coin Parameters ----------------------------------------------------------

struct CoinParams {
    // Display
    const char* name;           // "Bitcoin", "Litecoin", etc.
    const char* ticker;         // "BTC", "LTC", etc.
    
    // Network prefixes
    std::uint8_t  p2pkh_version;      // P2PKH address version byte
    std::uint8_t  p2pkh_version_test; // P2PKH testnet version byte
    std::uint8_t  p2sh_version;       // P2SH version byte
    std::uint8_t  wif_prefix;         // WIF private key prefix
    std::uint8_t  wif_prefix_test;    // WIF testnet prefix
    
    // Bech32
    const char*   bech32_hrp;         // "bc", "ltc", "tb" or nullptr
    const char*   bech32_hrp_test;    // Testnet HRP or nullptr
    
    // BIP-44
    std::uint32_t coin_type;          // BIP-44 coin_type index
    
    // Addressing
    AddressHash    hash_algo;         // Which hash for address derivation
    AddressEncoding default_encoding; // Primary address format
    
    // Extended key versions (BIP-32)
    std::uint32_t xprv_version;       // xprv serialization magic (4 bytes)
    std::uint32_t xpub_version;       // xpub serialization magic (4 bytes)
    
    // EVM chain ID (for EIP-155 signing; 0 = not applicable)
    std::uint64_t chain_id;
    
    // Features
    CoinFeatures  features;
};

// ============================================================================
// Predefined Coin Configurations (25+ coins)
// ============================================================================
// All secp256k1-based cryptocurrencies

// -- Bitcoin (BTC) ------------------------------------------------------------
inline constexpr CoinParams Bitcoin = {
    .name               = "Bitcoin",
    .ticker             = "BTC",
    .p2pkh_version      = 0x00,
    .p2pkh_version_test = 0x6F,
    .p2sh_version       = 0x05,
    .wif_prefix         = 0x80,
    .wif_prefix_test    = 0xEF,
    .bech32_hrp         = "bc",
    .bech32_hrp_test    = "tb",
    .coin_type          = 0,
    .hash_algo          = AddressHash::HASH160,
    .default_encoding   = AddressEncoding::BECH32,
    .xprv_version       = 0x0488ADE4,
    .xpub_version       = 0x0488B21E,
    .chain_id           = 0,
    .features           = {true, true, true, true, false, true},
};

// -- Litecoin (LTC) ----------------------------------------------------------
inline constexpr CoinParams Litecoin = {
    .name               = "Litecoin",
    .ticker             = "LTC",
    .p2pkh_version      = 0x30,
    .p2pkh_version_test = 0x6F,
    .p2sh_version       = 0x32,
    .wif_prefix         = 0xB0,
    .wif_prefix_test    = 0xEF,
    .bech32_hrp         = "ltc",
    .bech32_hrp_test    = "tltc",
    .coin_type          = 2,
    .hash_algo          = AddressHash::HASH160,
    .default_encoding   = AddressEncoding::BECH32,
    .xprv_version       = 0x0488ADE4,
    .xpub_version       = 0x0488B21E,
    .chain_id           = 0,
    .features           = {true, false, true, true, false, false},
};

// -- Dogecoin (DOGE) ----------------------------------------------------------
inline constexpr CoinParams Dogecoin = {
    .name               = "Dogecoin",
    .ticker             = "DOGE",
    .p2pkh_version      = 0x1E,
    .p2pkh_version_test = 0x71,
    .p2sh_version       = 0x16,
    .wif_prefix         = 0x9E,
    .wif_prefix_test    = 0xF1,
    .bech32_hrp         = nullptr,
    .bech32_hrp_test    = nullptr,
    .coin_type          = 3,
    .hash_algo          = AddressHash::HASH160,
    .default_encoding   = AddressEncoding::BASE58CHECK,
    .xprv_version       = 0x02FAC398,
    .xpub_version       = 0x02FACAFD,
    .chain_id           = 0,
    .features           = {false, false, true, false, false, false},
};

// -- Dash (DASH) --------------------------------------------------------------
inline constexpr CoinParams Dash = {
    .name               = "Dash",
    .ticker             = "DASH",
    .p2pkh_version      = 0x4C,
    .p2pkh_version_test = 0x8C,
    .p2sh_version       = 0x10,
    .wif_prefix         = 0xCC,
    .wif_prefix_test    = 0xEF,
    .bech32_hrp         = nullptr,
    .bech32_hrp_test    = nullptr,
    .coin_type          = 5,
    .hash_algo          = AddressHash::HASH160,
    .default_encoding   = AddressEncoding::BASE58CHECK,
    .xprv_version       = 0x0488ADE4,
    .xpub_version       = 0x0488B21E,
    .chain_id           = 0,
    .features           = {false, false, true, true, false, false},
};

// -- Ethereum (ETH) -----------------------------------------------------------
inline constexpr CoinParams Ethereum = {
    .name               = "Ethereum",
    .ticker             = "ETH",
    .p2pkh_version      = 0x00,  // Not used (EVM addresses)
    .p2pkh_version_test = 0x00,
    .p2sh_version       = 0x00,
    .wif_prefix         = 0x00,  // Not used (raw hex private keys)
    .wif_prefix_test    = 0x00,
    .bech32_hrp         = nullptr,
    .bech32_hrp_test    = nullptr,
    .coin_type          = 60,
    .hash_algo          = AddressHash::KECCAK256,
    .default_encoding   = AddressEncoding::EIP55,
    .xprv_version       = 0x0488ADE4,
    .xpub_version       = 0x0488B21E,
    .chain_id           = 1,
    .features           = {false, false, false, true, true, false},
};

// -- Bitcoin Cash (BCH) -------------------------------------------------------
inline constexpr CoinParams BitcoinCash = {
    .name               = "Bitcoin Cash",
    .ticker             = "BCH",
    .p2pkh_version      = 0x00,
    .p2pkh_version_test = 0x6F,
    .p2sh_version       = 0x05,
    .wif_prefix         = 0x80,
    .wif_prefix_test    = 0xEF,
    .bech32_hrp         = nullptr,
    .bech32_hrp_test    = nullptr,
    .coin_type          = 145,
    .hash_algo          = AddressHash::HASH160,
    .default_encoding   = AddressEncoding::CASHADDR,
    .xprv_version       = 0x0488ADE4,
    .xpub_version       = 0x0488B21E,
    .chain_id           = 0,
    .features           = {false, false, true, true, false, true},
};

// -- Bitcoin SV (BSV) ---------------------------------------------------------
inline constexpr CoinParams BitcoinSV = {
    .name               = "Bitcoin SV",
    .ticker             = "BSV",
    .p2pkh_version      = 0x00,
    .p2pkh_version_test = 0x6F,
    .p2sh_version       = 0x05,
    .wif_prefix         = 0x80,
    .wif_prefix_test    = 0xEF,
    .bech32_hrp         = nullptr,
    .bech32_hrp_test    = nullptr,
    .coin_type          = 236,
    .hash_algo          = AddressHash::HASH160,
    .default_encoding   = AddressEncoding::BASE58CHECK,
    .xprv_version       = 0x0488ADE4,
    .xpub_version       = 0x0488B21E,
    .chain_id           = 0,
    .features           = {false, false, true, false, false, false},
};

// -- Zcash (ZEC) --------------------------------------------------------------
inline constexpr CoinParams Zcash = {
    .name               = "Zcash",
    .ticker             = "ZEC",
    .p2pkh_version      = 0x1C, // t-addr prefix first byte (0x1CB8 two-byte)
    .p2pkh_version_test = 0x1D,
    .p2sh_version       = 0x1C, // 0x1CBD
    .wif_prefix         = 0x80,
    .wif_prefix_test    = 0xEF,
    .bech32_hrp         = nullptr,
    .bech32_hrp_test    = nullptr,
    .coin_type          = 133,
    .hash_algo          = AddressHash::HASH160,
    .default_encoding   = AddressEncoding::BASE58CHECK,
    .xprv_version       = 0x0488ADE4,
    .xpub_version       = 0x0488B21E,
    .chain_id           = 0,
    .features           = {false, false, true, true, false, false},
};

// -- DigiByte (DGB) -----------------------------------------------------------
inline constexpr CoinParams DigiByte = {
    .name               = "DigiByte",
    .ticker             = "DGB",
    .p2pkh_version      = 0x1E,
    .p2pkh_version_test = 0x7E,
    .p2sh_version       = 0x3F,
    .wif_prefix         = 0x80,
    .wif_prefix_test    = 0xFE,
    .bech32_hrp         = "dgb",
    .bech32_hrp_test    = "dgbt",
    .coin_type          = 20,
    .hash_algo          = AddressHash::HASH160,
    .default_encoding   = AddressEncoding::BECH32,
    .xprv_version       = 0x0488ADE4,
    .xpub_version       = 0x0488B21E,
    .chain_id           = 0,
    .features           = {true, false, true, true, false, false},
};

// -- Namecoin (NMC) -----------------------------------------------------------
inline constexpr CoinParams Namecoin = {
    .name               = "Namecoin",
    .ticker             = "NMC",
    .p2pkh_version      = 0x34,
    .p2pkh_version_test = 0x6F,
    .p2sh_version       = 0x0D,
    .wif_prefix         = 0xB4,
    .wif_prefix_test    = 0xEF,
    .bech32_hrp         = nullptr,
    .bech32_hrp_test    = nullptr,
    .coin_type          = 7,
    .hash_algo          = AddressHash::HASH160,
    .default_encoding   = AddressEncoding::BASE58CHECK,
    .xprv_version       = 0x0488ADE4,
    .xpub_version       = 0x0488B21E,
    .chain_id           = 0,
    .features           = {false, false, true, true, false, false},
};

// -- Peercoin (PPC) -----------------------------------------------------------
inline constexpr CoinParams Peercoin = {
    .name               = "Peercoin",
    .ticker             = "PPC",
    .p2pkh_version      = 0x37,
    .p2pkh_version_test = 0x6F,
    .p2sh_version       = 0x75,
    .wif_prefix         = 0xB7,
    .wif_prefix_test    = 0xEF,
    .bech32_hrp         = nullptr,
    .bech32_hrp_test    = nullptr,
    .coin_type          = 6,
    .hash_algo          = AddressHash::HASH160,
    .default_encoding   = AddressEncoding::BASE58CHECK,
    .xprv_version       = 0x0488ADE4,
    .xpub_version       = 0x0488B21E,
    .chain_id           = 0,
    .features           = {false, false, true, false, false, false},
};

// -- Vertcoin (VTC) -----------------------------------------------------------
inline constexpr CoinParams Vertcoin = {
    .name               = "Vertcoin",
    .ticker             = "VTC",
    .p2pkh_version      = 0x47,
    .p2pkh_version_test = 0x4A,
    .p2sh_version       = 0x05,
    .wif_prefix         = 0x80,
    .wif_prefix_test    = 0xEF,
    .bech32_hrp         = "vtc",
    .bech32_hrp_test    = "tvtc",
    .coin_type          = 28,
    .hash_algo          = AddressHash::HASH160,
    .default_encoding   = AddressEncoding::BECH32,
    .xprv_version       = 0x0488ADE4,
    .xpub_version       = 0x0488B21E,
    .chain_id           = 0,
    .features           = {true, false, true, true, false, false},
};

// -- Viacoin (VIA) ------------------------------------------------------------
inline constexpr CoinParams Viacoin = {
    .name               = "Viacoin",
    .ticker             = "VIA",
    .p2pkh_version      = 0x47,
    .p2pkh_version_test = 0x7F,
    .p2sh_version       = 0x21,
    .wif_prefix         = 0xC7,
    .wif_prefix_test    = 0xFF,
    .bech32_hrp         = "via",
    .bech32_hrp_test    = "tvia",
    .coin_type          = 14,
    .hash_algo          = AddressHash::HASH160,
    .default_encoding   = AddressEncoding::BECH32,
    .xprv_version       = 0x0488ADE4,
    .xpub_version       = 0x0488B21E,
    .chain_id           = 0,
    .features           = {true, false, true, true, false, false},
};

// -- Groestlcoin (GRS) --------------------------------------------------------
inline constexpr CoinParams Groestlcoin = {
    .name               = "Groestlcoin",
    .ticker             = "GRS",
    .p2pkh_version      = 0x24,
    .p2pkh_version_test = 0x6F,
    .p2sh_version       = 0x05,
    .wif_prefix         = 0x80,
    .wif_prefix_test    = 0xEF,
    .bech32_hrp         = "grs",
    .bech32_hrp_test    = "tgrs",
    .coin_type          = 17,
    .hash_algo          = AddressHash::HASH160,
    .default_encoding   = AddressEncoding::BECH32,
    .xprv_version       = 0x0488ADE4,
    .xpub_version       = 0x0488B21E,
    .chain_id           = 0,
    .features           = {true, true, true, true, false, false},
};

// -- Syscoin (SYS) ------------------------------------------------------------
inline constexpr CoinParams Syscoin = {
    .name               = "Syscoin",
    .ticker             = "SYS",
    .p2pkh_version      = 0x3F,
    .p2pkh_version_test = 0x41,
    .p2sh_version       = 0x05,
    .wif_prefix         = 0x80,
    .wif_prefix_test    = 0xEF,
    .bech32_hrp         = "sys",
    .bech32_hrp_test    = "tsys",
    .coin_type          = 57,
    .hash_algo          = AddressHash::HASH160,
    .default_encoding   = AddressEncoding::BECH32,
    .xprv_version       = 0x0488ADE4,
    .xpub_version       = 0x0488B21E,
    .chain_id           = 0,
    .features           = {true, false, true, true, false, false},
};

// -- BNB Smart Chain (BNB) ----------------------------------------------------
inline constexpr CoinParams BNBSmartChain = {
    .name               = "BNB Smart Chain",
    .ticker             = "BNB",
    .p2pkh_version      = 0x00,
    .p2pkh_version_test = 0x00,
    .p2sh_version       = 0x00,
    .wif_prefix         = 0x00,
    .wif_prefix_test    = 0x00,
    .bech32_hrp         = nullptr,
    .bech32_hrp_test    = nullptr,
    .coin_type          = 60,  // Same as Ethereum for BSC
    .hash_algo          = AddressHash::KECCAK256,
    .default_encoding   = AddressEncoding::EIP55,
    .xprv_version       = 0x0488ADE4,
    .xpub_version       = 0x0488B21E,
    .chain_id           = 56,
    .features           = {false, false, false, true, true, false},
};

// -- Polygon (MATIC / POL) ----------------------------------------------------
inline constexpr CoinParams Polygon = {
    .name               = "Polygon",
    .ticker             = "POL",
    .p2pkh_version      = 0x00,
    .p2pkh_version_test = 0x00,
    .p2sh_version       = 0x00,
    .wif_prefix         = 0x00,
    .wif_prefix_test    = 0x00,
    .bech32_hrp         = nullptr,
    .bech32_hrp_test    = nullptr,
    .coin_type          = 60,
    .hash_algo          = AddressHash::KECCAK256,
    .default_encoding   = AddressEncoding::EIP55,
    .xprv_version       = 0x0488ADE4,
    .xpub_version       = 0x0488B21E,
    .chain_id           = 137,
    .features           = {false, false, false, true, true, false},
};

// -- Avalanche C-Chain (AVAX) -------------------------------------------------
inline constexpr CoinParams Avalanche = {
    .name               = "Avalanche",
    .ticker             = "AVAX",
    .p2pkh_version      = 0x00,
    .p2pkh_version_test = 0x00,
    .p2sh_version       = 0x00,
    .wif_prefix         = 0x00,
    .wif_prefix_test    = 0x00,
    .bech32_hrp         = nullptr,
    .bech32_hrp_test    = nullptr,
    .coin_type          = 60,
    .hash_algo          = AddressHash::KECCAK256,
    .default_encoding   = AddressEncoding::EIP55,
    .xprv_version       = 0x0488ADE4,
    .xpub_version       = 0x0488B21E,
    .chain_id           = 43114,
    .features           = {false, false, false, true, true, false},
};

// -- Fantom (FTM) -------------------------------------------------------------
inline constexpr CoinParams Fantom = {
    .name               = "Fantom",
    .ticker             = "FTM",
    .p2pkh_version      = 0x00,
    .p2pkh_version_test = 0x00,
    .p2sh_version       = 0x00,
    .wif_prefix         = 0x00,
    .wif_prefix_test    = 0x00,
    .bech32_hrp         = nullptr,
    .bech32_hrp_test    = nullptr,
    .coin_type          = 60,
    .hash_algo          = AddressHash::KECCAK256,
    .default_encoding   = AddressEncoding::EIP55,
    .xprv_version       = 0x0488ADE4,
    .xpub_version       = 0x0488B21E,
    .chain_id           = 250,
    .features           = {false, false, false, true, true, false},
};

// -- Arbitrum (ARB) -----------------------------------------------------------
inline constexpr CoinParams Arbitrum = {
    .name               = "Arbitrum",
    .ticker             = "ARB",
    .p2pkh_version      = 0x00,
    .p2pkh_version_test = 0x00,
    .p2sh_version       = 0x00,
    .wif_prefix         = 0x00,
    .wif_prefix_test    = 0x00,
    .bech32_hrp         = nullptr,
    .bech32_hrp_test    = nullptr,
    .coin_type          = 60,
    .hash_algo          = AddressHash::KECCAK256,
    .default_encoding   = AddressEncoding::EIP55,
    .xprv_version       = 0x0488ADE4,
    .xpub_version       = 0x0488B21E,
    .chain_id           = 42161,
    .features           = {false, false, false, true, true, false},
};

// -- Optimism (OP) ------------------------------------------------------------
inline constexpr CoinParams Optimism = {
    .name               = "Optimism",
    .ticker             = "OP",
    .p2pkh_version      = 0x00,
    .p2pkh_version_test = 0x00,
    .p2sh_version       = 0x00,
    .wif_prefix         = 0x00,
    .wif_prefix_test    = 0x00,
    .bech32_hrp         = nullptr,
    .bech32_hrp_test    = nullptr,
    .coin_type          = 60,
    .hash_algo          = AddressHash::KECCAK256,
    .default_encoding   = AddressEncoding::EIP55,
    .xprv_version       = 0x0488ADE4,
    .xpub_version       = 0x0488B21E,
    .chain_id           = 10,
    .features           = {false, false, false, true, true, false},
};

// -- Ravencoin (RVN) ----------------------------------------------------------
inline constexpr CoinParams Ravencoin = {
    .name               = "Ravencoin",
    .ticker             = "RVN",
    .p2pkh_version      = 0x3C,
    .p2pkh_version_test = 0x6F,
    .p2sh_version       = 0x7A,
    .wif_prefix         = 0x80,
    .wif_prefix_test    = 0xEF,
    .bech32_hrp         = nullptr,
    .bech32_hrp_test    = nullptr,
    .coin_type          = 175,
    .hash_algo          = AddressHash::HASH160,
    .default_encoding   = AddressEncoding::BASE58CHECK,
    .xprv_version       = 0x0488ADE4,
    .xpub_version       = 0x0488B21E,
    .chain_id           = 0,
    .features           = {false, false, true, true, false, false},
};

// -- Flux (FLUX) --------------------------------------------------------------
inline constexpr CoinParams Flux = {
    .name               = "Flux",
    .ticker             = "FLUX",
    .p2pkh_version      = 0x1C, // t1...
    .p2pkh_version_test = 0x1D,
    .p2sh_version       = 0x1C,
    .wif_prefix         = 0x80,
    .wif_prefix_test    = 0xEF,
    .bech32_hrp         = nullptr,
    .bech32_hrp_test    = nullptr,
    .coin_type          = 19167,
    .hash_algo          = AddressHash::HASH160,
    .default_encoding   = AddressEncoding::BASE58CHECK,
    .xprv_version       = 0x0488ADE4,
    .xpub_version       = 0x0488B21E,
    .chain_id           = 0,
    .features           = {false, false, true, true, false, false},
};

// -- Qtum (QTUM) --------------------------------------------------------------
inline constexpr CoinParams Qtum = {
    .name               = "Qtum",
    .ticker             = "QTUM",
    .p2pkh_version      = 0x3A,
    .p2pkh_version_test = 0x78,
    .p2sh_version       = 0x32,
    .wif_prefix         = 0x80,
    .wif_prefix_test    = 0xEF,
    .bech32_hrp         = "qc",
    .bech32_hrp_test    = "tq",
    .coin_type          = 2301,
    .hash_algo          = AddressHash::HASH160,
    .default_encoding   = AddressEncoding::BASE58CHECK,
    .xprv_version       = 0x0488ADE4,
    .xpub_version       = 0x0488B21E,
    .chain_id           = 0,
    .features           = {true, false, true, true, false, false},
};

// -- Horizen (ZEN) ------------------------------------------------------------
inline constexpr CoinParams Horizen = {
    .name               = "Horizen",
    .ticker             = "ZEN",
    .p2pkh_version      = 0x20, // zn...
    .p2pkh_version_test = 0x20,
    .p2sh_version       = 0x20,
    .wif_prefix         = 0x80,
    .wif_prefix_test    = 0xEF,
    .bech32_hrp         = nullptr,
    .bech32_hrp_test    = nullptr,
    .coin_type          = 121,
    .hash_algo          = AddressHash::HASH160,
    .default_encoding   = AddressEncoding::BASE58CHECK,
    .xprv_version       = 0x0488ADE4,
    .xpub_version       = 0x0488B21E,
    .chain_id           = 0,
    .features           = {false, false, true, true, false, false},
};

// -- Bitcoin Gold (BTG) -------------------------------------------------------
inline constexpr CoinParams BitcoinGold = {
    .name               = "Bitcoin Gold",
    .ticker             = "BTG",
    .p2pkh_version      = 0x26,
    .p2pkh_version_test = 0x6F,
    .p2sh_version       = 0x17,
    .wif_prefix         = 0x80,
    .wif_prefix_test    = 0xEF,
    .bech32_hrp         = "btg",
    .bech32_hrp_test    = "tbtg",
    .coin_type          = 156,
    .hash_algo          = AddressHash::HASH160,
    .default_encoding   = AddressEncoding::BASE58CHECK,
    .xprv_version       = 0x0488ADE4,
    .xpub_version       = 0x0488B21E,
    .chain_id           = 0,
    .features           = {true, false, true, true, false, false},
};

// -- Komodo (KMD) -------------------------------------------------------------
inline constexpr CoinParams Komodo = {
    .name               = "Komodo",
    .ticker             = "KMD",
    .p2pkh_version      = 0x3C,
    .p2pkh_version_test = 0x00,
    .p2sh_version       = 0x55,
    .wif_prefix         = 0xBC,
    .wif_prefix_test    = 0x00,
    .bech32_hrp         = nullptr,
    .bech32_hrp_test    = nullptr,
    .coin_type          = 141,
    .hash_algo          = AddressHash::HASH160,
    .default_encoding   = AddressEncoding::BASE58CHECK,
    .xprv_version       = 0x0488ADE4,
    .xpub_version       = 0x0488B21E,
    .chain_id           = 0,
    .features           = {false, false, true, true, false, false},
};

// -- Tron (TRX) ---------------------------------------------------------------
inline constexpr CoinParams Tron = {
    .name               = "Tron",
    .ticker             = "TRX",
    .p2pkh_version      = 0x41,  // Tron address prefix byte
    .p2pkh_version_test = 0xA0,
    .p2sh_version       = 0x00,
    .wif_prefix         = 0x00,  // Not used (raw hex private keys)
    .wif_prefix_test    = 0x00,
    .bech32_hrp         = nullptr,
    .bech32_hrp_test    = nullptr,
    .coin_type          = 195,
    .hash_algo          = AddressHash::KECCAK256,
    .default_encoding   = AddressEncoding::TRON_BASE58,
    .xprv_version       = 0x0488ADE4,
    .xpub_version       = 0x0488B21E,
    .chain_id           = 0,
    .features           = {false, false, false, true, false, false},
};

// ============================================================================
// Coin Registry -- Lookup by coin_type or ticker
// ============================================================================

// All predefined coins for iteration
inline constexpr const CoinParams* ALL_COINS[] = {
    &Bitcoin, &Litecoin, &Dogecoin, &Dash, &Ethereum,
    &BitcoinCash, &BitcoinSV, &Zcash, &DigiByte, &Namecoin,
    &Peercoin, &Vertcoin, &Viacoin, &Groestlcoin, &Syscoin,
    &BNBSmartChain, &Polygon, &Avalanche, &Fantom, &Arbitrum,
    &Optimism, &Ravencoin, &Flux, &Qtum, &Horizen,
    &BitcoinGold, &Komodo, &Tron,
};

inline constexpr std::size_t ALL_COINS_COUNT = std::size(ALL_COINS);

// Find coin by BIP-44 coin_type (returns nullptr if not found)
inline const CoinParams* find_by_coin_type(std::uint32_t coin_type) {
    for (const auto* coin : ALL_COINS) {
        if (coin->coin_type == coin_type) return coin;
    }
    return nullptr;
}

// Find coin by ticker string (case-sensitive, returns nullptr if not found)
inline const CoinParams* find_by_ticker(const char* ticker) {
    for (const auto* coin : ALL_COINS) {
        const char* a = coin->ticker;
        const char* b = ticker;
        bool match = true;
        while (*a && *b) {
            if (*a != *b) { match = false; break; }
            ++a; ++b;
        }
        if (match && *a == *b) return coin;
    }
    return nullptr;
}

} // namespace secp256k1::coins

#endif // SECP256K1_COINS_COIN_PARAMS_HPP
