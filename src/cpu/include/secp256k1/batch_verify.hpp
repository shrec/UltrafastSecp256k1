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

// Multi-threaded Schnorr batch verify — first-class engine parallelism, the
// Schnorr twin of ecdsa_batch_verify_mt. Boolean result is identical to
// schnorr_batch_verify for any thread count; BIP-340 verification is
// variable-time over public data (pubkey/msg/sig), so threads are a pure
// throughput win. max_threads == 0 => auto (hardware_concurrency); an explicit
// request is honoured, reduced only to what the hardware can run (no arbitrary
// cap); max_threads == 1 => serial. Chunked so per-thread scratch stays
// O(chunk), never O(n). Any invalid entry in any chunk makes the call false.
bool schnorr_batch_verify_mt(const SchnorrBatchEntry* entries, std::size_t n,
                             std::size_t max_threads = 0);
bool schnorr_batch_verify_mt(const std::vector<SchnorrBatchEntry>& entries,
                             std::size_t max_threads = 0);

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

// Multi-threaded ECDSA batch verify — first-class engine parallelism.
// Boolean result is identical to ecdsa_batch_verify for any thread count;
// verification is variable-time over public data, so threads are a pure
// throughput win. max_threads == 0 => auto (hardware_concurrency); an explicit
// request is honoured, reduced only to what the hardware can run (no arbitrary
// cap); max_threads == 1 => serial. Chunked so per-thread scratch stays
// O(chunk), never O(n). See batch_verify.cpp for the full contract.
bool ecdsa_batch_verify_mt(const ECDSABatchEntry* entries, std::size_t n,
                           std::size_t max_threads = 0);
bool ecdsa_batch_verify_mt(const std::vector<ECDSABatchEntry>& entries,
                           std::size_t max_threads = 0);

// -- Opaque byte-span batch verification ---------------------------------------

// Direct byte-layout APIs for C++ consumers that already store libbitcoin/libsecp
// compatible rows or column arrays. These functions parse and verify in bounded
// chunks inside the engine; callers do not need to marshal the full table into
// ECDSABatchEntry/SchnorrBatchEntry vectors.
//
// ECDSA row layout: [hash32 | compressed_pubkey33 | opaque_sig64]
// ECDSA columns: digests[count][32], pubkeys[count][33], sigs[count][64]
// opaque_sig64 is the libsecp/libbitcoin in-memory ECDSA signature layout:
// little-endian scalar limbs for r followed by s.
//
// Schnorr row layout: [msg32 | xonly_pubkey32 | bip340_sig64]
// Schnorr columns: digests[count][32], xonly[count][32], sigs[count][64]
// ECDSA opaque rows/columns are libbitcoin consensus verify paths: they accept
// mathematically valid high-S signatures. Low-S standardness enforcement remains
// on the strict ecdsa_batch_verify/libsecp-compatible surfaces.
//
// Returns true iff all rows verify. If out_results is non-null, all-valid batches
// fill it with 1; mixed, invalid, or unparsable batches write 1/0 per row after
// per-row fallback. count == 0 returns true. Invalid pointers or undersized row
// strides return false and zero out_results when provided.
bool ecdsa_batch_verify_opaque_rows(const std::uint8_t* rows, std::size_t stride,
                                    std::size_t count,
                                    std::uint8_t* out_results = nullptr,
                                    std::size_t max_threads = 0);
bool ecdsa_batch_verify_opaque_columns(const std::uint8_t* digests32,
                                       const std::uint8_t* pubkeys33,
                                       const std::uint8_t* sigs64,
                                       std::size_t count,
                                       std::uint8_t* out_results = nullptr,
                                       std::size_t max_threads = 0);
bool schnorr_batch_verify_bip340_rows(const std::uint8_t* rows, std::size_t stride,
                                      std::size_t count,
                                      std::uint8_t* out_results = nullptr,
                                      std::size_t max_threads = 0);
bool schnorr_batch_verify_bip340_columns(const std::uint8_t* digests32,
                                         const std::uint8_t* xonly32,
                                         const std::uint8_t* sigs64,
                                         std::size_t count,
                                         std::uint8_t* out_results = nullptr,
                                         std::size_t max_threads = 0);

// -- GPU column-verify accelerator hook (PUBLIC-DATA verify only) --------------
// Null by default (pure CPU). Self-installed by the GPU-host layer
// (src/gpu/src/gpu_engine_hook.cpp) via a static initializer whenever that TU is
// linked; the engine keeps NO gpu:: dependency (a reverse engine->gpu_host link
// would be a static-lib cycle) and no compile-time macro gates it. The two
// *_batch_verify_*_columns entrypoints consult it internally so the
// libbitcoin-direct caller keeps ONE API with no caller-visible CPU/GPU split and
// no recoverable GPU status.
//   kind == 0 : ECDSA opaque-LE columns  (keys = pubkeys33)
//   kind == 1 : Schnorr BIP-340 columns  (keys = xonly32)
// Return contract:
//    1  -> handled; out_results fully written; ALL rows valid.
//    0  -> handled; out_results fully written; at least one row invalid.
//   -1  -> NOT handled (GPU unavailable/unsupported/operational error); the engine
//          MUST fall back to the CPU column path. A declining hook MUST NOT write a
//          consensus-invalid all-zero buffer to signal "not handled".
// Operational backend errors are declines (-1) -> CPU fallback, never invalid rows.
// A hook that cannot complete the math and cannot decline must abort/fail hard
// itself; it must never return an incorrect verdict.
using GpuColumnsVerifyHook = int (*)(int kind,
        const std::uint8_t* digests32, const std::uint8_t* keys,
        const std::uint8_t* sigs64, std::size_t count,
        std::uint8_t* out_results) noexcept;

// Install (nullptr clears). Thread-safe. Returns the previous hook (save/restore).
GpuColumnsVerifyHook install_gpu_columns_verify_hook(GpuColumnsVerifyHook hook) noexcept;

// -- Identify Invalid Signatures ----------------------------------------------

// After a batch fails, identify which signature(s) are invalid.
// Returns indices of invalid entries.
void schnorr_batch_identify_invalid(
    const SchnorrBatchEntry* entries, std::size_t n,
    std::vector<std::size_t>& invalid_out);

std::vector<std::size_t> schnorr_batch_identify_invalid(
    const SchnorrBatchEntry* entries, std::size_t n);

void schnorr_batch_identify_invalid(
    const SchnorrBatchCachedEntry* entries, std::size_t n,
    std::vector<std::size_t>& invalid_out);

std::vector<std::size_t> schnorr_batch_identify_invalid(
    const SchnorrBatchCachedEntry* entries, std::size_t n);

void ecdsa_batch_identify_invalid(
    const ECDSABatchEntry* entries, std::size_t n,
    std::vector<std::size_t>& invalid_out);

std::vector<std::size_t> ecdsa_batch_identify_invalid(
    const ECDSABatchEntry* entries, std::size_t n);

} // namespace secp256k1

#endif // SECP256K1_BATCH_VERIFY_HPP
