#ifndef SECP256K1_FIELD_SIMD_HPP
#define SECP256K1_FIELD_SIMD_HPP

// ============================================================================
// AVX2 / AVX-512 SIMD Field Arithmetic — secp256k1
// ============================================================================
// Batch field operations using x86 SIMD intrinsics for 4x (AVX2) or
// 8x (AVX-512) parallel field element processing.
//
// Architecture:
//   - Runtime CPUID detection: avx2_available(), avx512_available()
//   - Batch API processes N field elements in parallel
//   - Falls back to scalar when SIMD not available
//
// Performance model:
//   - AVX2 (256-bit): 4 field ops in parallel → ~3x throughput for batch work
//   - AVX-512 (512-bit): 8 field ops → ~5-6x throughput
//   - Only beneficial for batch operations (batch verify, multi-scalar mul)
//   - Single-element ops are faster with scalar code (pipeline fill overhead)
//
// Usage:
//   if (secp256k1::simd::avx2_available()) {
//       secp256k1::simd::batch_field_add_avx2(out, a, b, count);
//   }
// ============================================================================

#include <cstdint>
#include <cstddef>
#include "secp256k1/field.hpp"

// Architecture detection
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define SECP256K1_X86_TARGET 1
#endif

#ifdef SECP256K1_X86_TARGET
    #ifdef _MSC_VER
        #include <intrin.h>
    #else
        #include <cpuid.h>
    #endif
#endif

namespace secp256k1::simd {

using fast::FieldElement;

// ── Runtime Feature Detection ────────────────────────────────────────────────

// Check if AVX2 is available at runtime
inline bool avx2_available() noexcept {
#ifdef SECP256K1_X86_TARGET
    #ifdef _MSC_VER
        int info[4];
        __cpuidex(info, 7, 0);
        return (info[1] & (1 << 5)) != 0;  // EBX bit 5 = AVX2
    #elif defined(__GNUC__) || defined(__clang__)
        unsigned int eax, ebx, ecx, edx;
        if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
            return (ebx & (1 << 5)) != 0;
        }
        return false;
    #endif
#else
    return false;
#endif
}

// Check if AVX-512F is available at runtime
inline bool avx512_available() noexcept {
#ifdef SECP256K1_X86_TARGET
    #ifdef _MSC_VER
        int info[4];
        __cpuidex(info, 7, 0);
        return (info[1] & (1 << 16)) != 0;  // EBX bit 16 = AVX-512F
    #elif defined(__GNUC__) || defined(__clang__)
        unsigned int eax, ebx, ecx, edx;
        if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
            return (ebx & (1 << 16)) != 0;
        }
        return false;
    #endif
#else
    return false;
#endif
}

// ── SIMD Tier Enum ───────────────────────────────────────────────────────────

enum class SimdTier : int {
    SCALAR  = 0,  // No SIMD, scalar fallback
    AVX2    = 1,  // AVX2 (256-bit, 4-way)
    AVX512  = 2,  // AVX-512 (512-bit, 8-way)
};

// Detect best available SIMD tier
inline SimdTier detect_simd_tier() noexcept {
    if (avx512_available()) return SimdTier::AVX512;
    if (avx2_available()) return SimdTier::AVX2;
    return SimdTier::SCALAR;
}

inline const char* simd_tier_name(SimdTier tier) noexcept {
    switch (tier) {
        case SimdTier::AVX512: return "AVX-512";
        case SimdTier::AVX2:   return "AVX2";
        default:               return "Scalar";
    }
}

// ── Batch API (auto-dispatching) ─────────────────────────────────────────────
// These functions auto-detect SIMD tier and dispatch accordingly.
// All operate on arrays of FieldElements.
// count can be any value; non-aligned remainder handled by scalar fallback.

// Batch addition: out[i] = a[i] + b[i]  for i in [0, count)
void batch_field_add(FieldElement* out,
                     const FieldElement* a,
                     const FieldElement* b,
                     std::size_t count);

// Batch subtraction: out[i] = a[i] - b[i]
void batch_field_sub(FieldElement* out,
                     const FieldElement* a,
                     const FieldElement* b,
                     std::size_t count);

// Batch multiplication: out[i] = a[i] * b[i]
void batch_field_mul(FieldElement* out,
                     const FieldElement* a,
                     const FieldElement* b,
                     std::size_t count);

// Batch squaring: out[i] = a[i]²
void batch_field_sqr(FieldElement* out,
                     const FieldElement* a,
                     std::size_t count);

// ── Batch Modular Inverse (Montgomery's trick) ──────────────────────────────
// Computes count inversions using only 1 field inversion + 3(n-1) multiplications.
// Much faster than n individual inversions for batch verification.
// Scratch buffer: needs at least 'count' FieldElements of scratch space.
// If scratch is nullptr, allocates internally (non-hot-path use).
void batch_field_inv(FieldElement* out,
                     const FieldElement* a,
                     std::size_t count,
                     FieldElement* scratch = nullptr);

// ── Architecture-Specific Entry Points (for benchmarking) ────────────────────
// These are only available if compiled with appropriate flags.
// Normal code should use the auto-dispatching batch_field_* functions above.

namespace detail {

// Scalar fallback (always available)
void batch_field_add_scalar(FieldElement* out, const FieldElement* a,
                            const FieldElement* b, std::size_t count);
void batch_field_sub_scalar(FieldElement* out, const FieldElement* a,
                            const FieldElement* b, std::size_t count);
void batch_field_mul_scalar(FieldElement* out, const FieldElement* a,
                            const FieldElement* b, std::size_t count);
void batch_field_sqr_scalar(FieldElement* out, const FieldElement* a,
                            std::size_t count);

} // namespace detail

} // namespace secp256k1::simd

#endif // SECP256K1_FIELD_SIMD_HPP
