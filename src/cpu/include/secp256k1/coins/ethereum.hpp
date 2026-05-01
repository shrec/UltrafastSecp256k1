#ifndef SECP256K1_COINS_ETHEREUM_HPP
#define SECP256K1_COINS_ETHEREUM_HPP
#pragma once

// ============================================================================
// Ethereum Address & Utilities
// ============================================================================
// Ethereum-specific functionality:
//   - EIP-55 mixed-case checksummed addresses (0x...)
//   - Address derivation from public key (Keccak-256)
//   - Raw 20-byte address extraction
//
// Compatible with all EVM chains:
//   Ethereum, BSC, Polygon, Avalanche, Fantom, Arbitrum, Optimism, etc.
//
// Address derivation:
//   1. Get uncompressed public key (65 bytes, prefix 0x04 + x + y)
//   2. Keccak-256(pubkey[1..65])  -- skip prefix byte
//   3. Take last 20 bytes of hash
//   4. Apply EIP-55 checksum (mixed-case hex encoding)
// ============================================================================

#include <array>
#include <cstdint>
#include <string>
#include "secp256k1/point.hpp"

namespace secp256k1::coins {

// -- Ethereum Address (EIP-55) ------------------------------------------------

// Generate EIP-55 checksummed Ethereum address from public key
// Returns: "0x" + 40 hex chars with mixed-case checksum
// Example: "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed"
std::string ethereum_address(const fast::Point& pubkey);

// Get raw 20-byte Ethereum address (lowercase, no "0x" prefix)
std::string ethereum_address_raw(const fast::Point& pubkey);

// Get raw 20-byte address as bytes
std::array<std::uint8_t, 20> ethereum_address_bytes(const fast::Point& pubkey);

// -- EIP-55 Checksum ----------------------------------------------------------

// Apply EIP-55 checksum to a 40-char lowercase hex address
// Input: 40 hex chars (no "0x" prefix)
// Output: 40 hex chars with mixed-case checksum
std::string eip55_checksum(const std::string& hex_addr);

// Verify EIP-55 checksum
bool eip55_verify(const std::string& addr);

} // namespace secp256k1::coins

#endif // SECP256K1_COINS_ETHEREUM_HPP
