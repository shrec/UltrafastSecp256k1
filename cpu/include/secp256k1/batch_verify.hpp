#ifndef SECP256K1_BATCH_VERIFY_HPP
#define SECP256K1_BATCH_VERIFY_HPP
#pragma once

// ============================================================================
// Batch Verification for ECDSA and Schnorr (BIP-340) on secp256k1
// ============================================================================
// Verifies N signatures in one multi-scalar multiplication, yielding
// ~2-3x speedup over N individual verifications for large batches.
//
// Method (Schnorr):
//   For signatures (R_i, s_i) on messages m_i with pubkeys P_i:
//   Pick random weights a_i, verify:
//     sum(a_i * s_i) * G  ==  sum(a_i * R_i) + sum(a_i * e_i * P_i)
//
// Method (ECDSA):
//   For signatures (r_i, s_i) on messages z_i with pubkeys Q_i:
//   Pick random weights a_i, w_i = s_i^{-1}, verify:
//     sum(a_i * z_i * w_i) * G + sum(a_i * r_i * w_i * Q_i) has x == r_i
//   (ECDSA batch is less efficient than Schnorr batch due to the structure)
//
// If batch verification fails, falls back to individual verification to
// identify the invalid signature(s).
// ============================================================================

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"

namespace secp256k1 {

// -- Schnorr Batch Verification -----------------------------------------------

struct SchnorrBatchEntry {
    std::array<std::uint8_t, 32> pubkey_x;  // X-only public key
    std::array<std::uint8_t, 32> message;   // 32-byte message
    SchnorrSignature signature;              // (r, s)
};

struct SchnorrBatchCachedEntry {
    const SchnorrXonlyPubkey* pubkey;        // Parsed x-only pubkey; must outlive the batch call
    std::array<std::uint8_t, 32> message;    // 32-byte message
    SchnorrSignature signature;              // (r, s)
};

// Verify a batch of Schnorr signatures.
// Returns true if ALL signatures are valid.
// Uses random linear combination for efficiency.
//
// Performance: ~2-3x faster than N individual schnorr_verify() calls.
bool schnorr_batch_verify(const SchnorrBatchEntry* entries, std::size_t n);
bool schnorr_batch_verify(const std::vector<SchnorrBatchEntry>& entries);
bool schnorr_batch_verify(const SchnorrBatchCachedEntry* entries, std::size_t n);
bool schnorr_batch_verify(const std::vector<SchnorrBatchCachedEntry>& entries);

// -- ECDSA Batch Verification -------------------------------------------------

struct ECDSABatchEntry {
    std::array<std::uint8_t, 32> msg_hash;  // 32-byte message hash
    fast::Point public_key;                  // Full public key point
    ECDSASignature signature;                // (r, s)
};

// Verify a batch of ECDSA signatures.
// Returns true if ALL signatures are valid.
//
// NOTE: ECDSA batch verification is less efficient than Schnorr due to
// requiring per-signature modular inversions. Speedup is ~1.5-2x.
bool ecdsa_batch_verify(const ECDSABatchEntry* entries, std::size_t n);
bool ecdsa_batch_verify(const std::vector<ECDSABatchEntry>& entries);

// -- Identify Invalid Signatures ----------------------------------------------

// After a batch fails, identify which signature(s) are invalid.
// Returns indices of invalid entries.
std::vector<std::size_t> schnorr_batch_identify_invalid(
    const SchnorrBatchEntry* entries, std::size_t n);

std::vector<std::size_t> schnorr_batch_identify_invalid(
    const SchnorrBatchCachedEntry* entries, std::size_t n);

std::vector<std::size_t> ecdsa_batch_identify_invalid(
    const ECDSABatchEntry* entries, std::size_t n);

} // namespace secp256k1

#endif // SECP256K1_BATCH_VERIFY_HPP
