#ifndef SECP256K1_COINS_ETH_SIGNING_HPP
#define SECP256K1_COINS_ETH_SIGNING_HPP
#pragma once

// ============================================================================
// Ethereum Signing Primitives
// ============================================================================
// EIP-191: Personal message signing ("\x19Ethereum Signed Message:\n" prefix)
// EIP-155: Chain-ID replay protection (v = recid + 35 + 2*chainId)
// ecrecover: Recover Ethereum address from (v, r, s) + message hash
//
// These are the core operations wallets & dApps use:
//   - MetaMask personal_sign / eth_sign
//   - Raw transaction signing (v,r,s with chain ID)
//   - Address recovery from signatures (ecrecover precompile 0x01)
//
// All signing uses CT layer (constant-time). Recovery uses fast layer.
// ============================================================================

#include <array>
#include <cstdint>
#include <cstddef>
#include <utility>
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/recovery.hpp"

namespace secp256k1::coins {

// -- EIP-191: Personal Message Hash -------------------------------------------

// Compute EIP-191 personal message hash:
//   Keccak256("\x19Ethereum Signed Message:\n" + decimal_len(msg) + msg)
// This is what MetaMask's personal_sign / eth_sign computes before signing.
std::array<std::uint8_t, 32> eip191_hash(const std::uint8_t* msg, std::size_t msg_len);

// -- EIP-155: Chain-ID Encoding -----------------------------------------------

// Convert recovery ID (0-3) + chain ID to EIP-155 v value.
// Legacy (pre-EIP155): v = 27 + recid
// EIP-155:             v = 35 + 2*chainId + recid
inline std::uint64_t eip155_v(int recid, std::uint64_t chain_id) {
    return 35 + 2 * chain_id + static_cast<std::uint64_t>(recid);
}

// Extract recovery ID from EIP-155 v value.
// Legacy: recid = v - 27
// EIP-155: recid = (v - 35) % 2  (works for v >= 35)
inline int eip155_recid(std::uint64_t v) {
    if (v < 27) {
        return -1;  // Invalid v value
    }
    if (v <= 28) {
        return static_cast<int>(v - 27);
    }
    return static_cast<int>((v - 35) & 1);
}

// Extract chain ID from EIP-155 v value.
// Returns 0 for legacy (v=27 or v=28).
inline std::uint64_t eip155_chain_id(std::uint64_t v) {
    if (v <= 28) return 0;
    return (v - 35) / 2;
}

// -- Ethereum Sign (personal_sign) --------------------------------------------

struct EthSignature {
    std::array<std::uint8_t, 32> r;
    std::array<std::uint8_t, 32> s;
    std::uint64_t v;  // EIP-155 v value (27/28 for legacy, 35+2*chainId+recid)
};

// Sign a raw message with EIP-191 prefix (personal_sign).
// Hashes: Keccak256("\x19Ethereum Signed Message:\n" + len + msg)
// then signs with ECDSA recovery.  v = 27 + recid (legacy format).
EthSignature eth_personal_sign(const std::uint8_t* msg, std::size_t msg_len,
                               const fast::Scalar& private_key);

// Sign a pre-computed 32-byte hash with recovery.
// v = 27 + recid (legacy) or 35 + 2*chainId + recid (EIP-155).
EthSignature eth_sign_hash(const std::array<std::uint8_t, 32>& hash,
                           const fast::Scalar& private_key,
                           std::uint64_t chain_id = 0);

// -- ecrecover: Recover Address from Signature --------------------------------

// Recover Ethereum address (20 bytes) from signature + message hash.
// This is the Ethereum ecrecover precompile (address 0x01).
// Returns {address, ok}. ok=false if recovery fails.
std::pair<std::array<std::uint8_t, 20>, bool>
ecrecover(const std::array<std::uint8_t, 32>& msg_hash,
          const std::array<std::uint8_t, 32>& r,
          const std::array<std::uint8_t, 32>& s,
          std::uint64_t v);

// ecrecover from EthSignature struct
std::pair<std::array<std::uint8_t, 20>, bool>
ecrecover(const std::array<std::uint8_t, 32>& msg_hash,
          const EthSignature& sig);

// -- Verify: Check that signature was produced by address ---------------------

// Verify that a personal_sign signature was produced by the given address.
bool eth_personal_verify(const std::uint8_t* msg, std::size_t msg_len,
                         const EthSignature& sig,
                         const std::array<std::uint8_t, 20>& expected_addr);

} // namespace secp256k1::coins

#endif // SECP256K1_COINS_ETH_SIGNING_HPP
