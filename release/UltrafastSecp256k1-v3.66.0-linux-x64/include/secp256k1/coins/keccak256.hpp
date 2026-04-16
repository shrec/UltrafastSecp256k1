#ifndef SECP256K1_COINS_KECCAK256_HPP
#define SECP256K1_COINS_KECCAK256_HPP
#pragma once

// ============================================================================
// Keccak-256 Hash Function
// ============================================================================
// Standard Keccak-256 (NOT SHA3-256; Ethereum uses raw Keccak before NIST
// finalization, i.e., without the 0x06 domain separator).
//
// Used by:
//   - Ethereum address derivation (Keccak-256 of uncompressed pubkey)
//   - EVM-compatible chains (BSC, Polygon, Avalanche, etc.)
//   - Solidity keccak256()
//
// Implementation:
//   - No heap allocation
//   - Fixed 32-byte output
//   - Incremental (absorb/squeeze) API available
// ============================================================================

#include <array>
#include <cstdint>
#include <cstddef>

namespace secp256k1::coins {

// -- Keccak-256 State ---------------------------------------------------------

struct Keccak256State {
    std::uint64_t state[25];     // 1600-bit Keccak state (5x5 lanes)
    std::uint8_t  buf[136];      // Rate buffer (r = 1088 bits = 136 bytes)
    std::size_t   buf_pos;       // Current position in buffer
    
    Keccak256State();
    ~Keccak256State();
    
    // Absorb data
    void update(const std::uint8_t* data, std::size_t len);
    
    // Finalize and produce 32-byte hash
    // Uses Keccak padding (0x01), NOT SHA3 padding (0x06)
    std::array<std::uint8_t, 32> finalize();
};

// -- One-Shot API -------------------------------------------------------------

// Compute Keccak-256 hash of data
std::array<std::uint8_t, 32> keccak256(const std::uint8_t* data, std::size_t len);

} // namespace secp256k1::coins

#endif // SECP256K1_COINS_KECCAK256_HPP
