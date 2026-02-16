#ifndef SECP256K1_HASH_ACCEL_HPP
#define SECP256K1_HASH_ACCEL_HPP
#pragma once

// ============================================================================
// Accelerated Hashing — SHA-256 / RIPEMD-160 / Hash160
// ============================================================================
//
// ## Three tiers of acceleration (runtime-detected):
//
//   Tier 0: SCALAR   — Portable C++ (baseline, always available)
//   Tier 1: SHA-NI   — Intel SHA Extensions (single-message HW accel, ~3-5×)
//   Tier 2: AVX2     — 4-way multi-buffer SHA-256 (interleaved, ~8-12×)
//                       + optimized RIPEMD-160 with BMI/BMI2
//   Tier 3: AVX-512  — 8-way multi-buffer SHA-256 (if available, ~16×)
//
// ## Hot-path API for search pipeline:
//
//   Compressed pubkey (33 bytes) → SHA-256 → RIPEMD-160 = Hash160 (20 bytes)
//
//   - hash160_single():       single pubkey → 20 bytes
//   - hash160_batch():        N pubkeys → N×20 bytes (multi-buffer SIMD)
//   - sha256_33():            SHA-256 of exactly 33 bytes (pubkey-optimized)
//   - ripemd160_32():         RIPEMD-160 of exactly 32 bytes (SHA output)
//
// ## Pubkey-specific optimization:
//
//   SHA-256(33 bytes) always produces exactly 1 block (33 + 1 + 22 + 8 = 64).
//   The padding is constant and can be precomputed. This eliminates all
//   branching, length encoding, and buffer management from the hot path.
//
// ============================================================================

#include <array>
#include <cstdint>
#include <cstddef>

// Architecture detection (must match hash_accel.cpp)
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #ifndef SECP256K1_X86_TARGET
        #define SECP256K1_X86_TARGET 1
    #endif
#endif

namespace secp256k1::hash {

// ── Feature Detection ────────────────────────────────────────────────────────

enum class HashTier : int {
    SCALAR  = 0,
    SHA_NI  = 1,  // Intel SHA Extensions
    AVX2    = 2,  // 4-way multi-buffer
    AVX512  = 3,  // 8-way multi-buffer
};

/// Detect best available hashing tier at runtime.
HashTier detect_hash_tier() noexcept;

/// Human-readable tier name.
const char* hash_tier_name(HashTier tier) noexcept;

/// Check individual features.
bool sha_ni_available() noexcept;
bool avx2_available() noexcept;
bool avx512_available() noexcept;

// ── Single-message SHA-256 ───────────────────────────────────────────────────

/// SHA-256 of arbitrary data (auto-selects best implementation).
std::array<std::uint8_t, 32> sha256(const void* data, std::size_t len) noexcept;

/// SHA-256 of exactly 33 bytes (compressed pubkey hot path).
/// Precomputed padding — no branches, no buffer management.
void sha256_33(const std::uint8_t* pubkey33, std::uint8_t* out32) noexcept;

/// SHA-256 of exactly 32 bytes (e.g. second hash in double-SHA256)
void sha256_32(const std::uint8_t* in32, std::uint8_t* out32) noexcept;

/// Double-SHA256: SHA256(SHA256(data)) for arbitrary data.
std::array<std::uint8_t, 32> sha256d(const void* data, std::size_t len) noexcept;

// ── Single-message RIPEMD-160 ────────────────────────────────────────────────

/// RIPEMD-160 of arbitrary data (auto-selects best implementation).
std::array<std::uint8_t, 20> ripemd160(const void* data, std::size_t len) noexcept;

/// RIPEMD-160 of exactly 32 bytes (SHA-256 output → Hash160 hot path).
/// Precomputed padding — no branches, no buffer management.
void ripemd160_32(const std::uint8_t* in32, std::uint8_t* out20) noexcept;

// ── Hash160 — RIPEMD160(SHA256(data)) ────────────────────────────────────────

/// Hash160 of arbitrary data.
std::array<std::uint8_t, 20> hash160(const void* data, std::size_t len) noexcept;

/// Hash160 of exactly 33 bytes (compressed pubkey).
/// Fused SHA256(33) + RIPEMD160(32) with minimal intermediary overhead.
void hash160_33(const std::uint8_t* pubkey33, std::uint8_t* out20) noexcept;

// ── Batch operations (multi-buffer SIMD) ─────────────────────────────────────
//
// Process multiple independent messages simultaneously using SIMD lanes.
// AVX2: 4 messages per cycle, AVX-512: 8 messages per cycle.
// Falls back to sequential scalar/SHA-NI when SIMD unavailable.
//
// All batch functions expect contiguous arrays:
//   - Input:  pubkeys[count * 33] (packed, no gaps)
//   - Output: hashes[count * output_size] (packed, no gaps)
//
// Hot-path contract: No heap allocation if scratch is pre-sized.

/// Batch SHA-256 of N×33-byte compressed pubkeys.
/// out32s: caller-allocated, at least count×32 bytes.
void sha256_33_batch(
    const std::uint8_t* pubkeys,    // count × 33 bytes (packed)
    std::uint8_t* out32s,           // count × 32 bytes output
    std::size_t count) noexcept;

/// Batch RIPEMD-160 of N×32-byte SHA-256 digests.
/// out20s: caller-allocated, at least count×20 bytes.
void ripemd160_32_batch(
    const std::uint8_t* in32s,      // count × 32 bytes
    std::uint8_t* out20s,           // count × 20 bytes output
    std::size_t count) noexcept;

/// Batch Hash160 of N×33-byte compressed pubkeys.
/// Fused pipeline: strides of 4/8 messages through SHA256→RIPEMD160.
/// out20s: caller-allocated, at least count×20 bytes.
void hash160_33_batch(
    const std::uint8_t* pubkeys,    // count × 33 bytes (packed)
    std::uint8_t* out20s,           // count × 20 bytes output
    std::size_t count) noexcept;

// ── Implementation selectors (for benchmarking / testing) ────────────────────
// These bypass auto-detection to force a specific tier.

namespace scalar {
    void sha256_compress(const std::uint8_t block[64], std::uint32_t state[8]) noexcept;
    void sha256_33(const std::uint8_t* pubkey33, std::uint8_t* out32) noexcept;
    void sha256_32(const std::uint8_t* in32, std::uint8_t* out32) noexcept;
    void ripemd160_compress(const std::uint8_t block[64], std::uint32_t state[5]) noexcept;
    void ripemd160_32(const std::uint8_t* in32, std::uint8_t* out20) noexcept;
    void hash160_33(const std::uint8_t* pubkey33, std::uint8_t* out20) noexcept;
}

#ifdef SECP256K1_X86_TARGET
namespace shani {
    void sha256_compress(const std::uint8_t block[64], std::uint32_t state[8]) noexcept;
    void sha256_33(const std::uint8_t* pubkey33, std::uint8_t* out32) noexcept;
    void sha256_32(const std::uint8_t* in32, std::uint8_t* out32) noexcept;
    void hash160_33(const std::uint8_t* pubkey33, std::uint8_t* out20) noexcept;
}
#endif

} // namespace secp256k1::hash

#endif // SECP256K1_HASH_ACCEL_HPP
