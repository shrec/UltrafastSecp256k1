#ifndef SECP256K1_COINS_MESSAGE_SIGNING_HPP
#define SECP256K1_COINS_MESSAGE_SIGNING_HPP
#pragma once

// ============================================================================
// Unified Message Signing -- Bitcoin, Ethereum, and generic secp256k1 chains
// ============================================================================
// Provides chain-aware message signing and verification:
//
//   Bitcoin:   "\x18Bitcoin Signed Message:\n" + varint(len) + msg -> dSHA256
//   Ethereum:  "\x19Ethereum Signed Message:\n" + decimal(len) + msg -> Keccak
//   Generic:   raw 32-byte hash signing (no prefix)
//
// All signing produces recoverable signatures (r, s, recid).
// Bitcoin uses base64-encoded 65-byte format (1-byte header + r + s).
// Ethereum uses RSV (r + s + v) with EIP-155 chain ID encoding.
// ============================================================================

#include <array>
#include <cstdint>
#include <cstddef>
#include <string>
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/recovery.hpp"

namespace secp256k1::coins {

// -- Bitcoin Message Signing (BIP-137 / Electrum) -----------------------------

// Hash a message using Bitcoin signed message format:
//   SHA256(SHA256("\x18Bitcoin Signed Message:\n" + varint(msg_len) + msg))
std::array<std::uint8_t, 32> bitcoin_message_hash(const std::uint8_t* msg,
                                                   std::size_t msg_len);

// Sign a message using Bitcoin signed message format.
// Returns recoverable signature (can extract recid for base64 encoding).
RecoverableSignature bitcoin_sign_message(const std::uint8_t* msg,
                                          std::size_t msg_len,
                                          const fast::Scalar& private_key);

// Verify a Bitcoin signed message against a public key.
bool bitcoin_verify_message(const std::uint8_t* msg,
                            std::size_t msg_len,
                            const fast::Point& pubkey,
                            const ECDSASignature& sig);

// Recover the public key from a Bitcoin signed message + recoverable signature.
// Returns (pubkey, success).
std::pair<fast::Point, bool>
bitcoin_recover_message(const std::uint8_t* msg,
                        std::size_t msg_len,
                        const ECDSASignature& sig,
                        int recid);

// Encode a recoverable signature as Bitcoin signed message base64 (65 bytes).
// Header byte encodes recid + compression flag (27-34 range).
std::string bitcoin_sig_to_base64(const RecoverableSignature& rsig,
                                  bool compressed = true);

// Decode a base64 Bitcoin signed message signature.
// Returns (signature, recid, compressed_flag, success).
struct BitcoinSigDecodeResult {
    ECDSASignature sig;
    int recid;
    bool compressed;
    bool valid;
};
BitcoinSigDecodeResult bitcoin_sig_from_base64(const std::string& base64);

} // namespace secp256k1::coins

#endif // SECP256K1_COINS_MESSAGE_SIGNING_HPP
