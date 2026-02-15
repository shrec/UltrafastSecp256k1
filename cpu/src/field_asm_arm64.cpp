// ============================================================================
// ARM64 (AArch64) Optimized Field Arithmetic for secp256k1
// ============================================================================
// Uses MUL/UMULH instructions for 64×64→128 multiply (single cycle each on
// Cortex-A55/A76). Fully unrolled, allocation-free, branchless.
//
// secp256k1 prime: p = 2^256 - 2^32 - 977
// Reduction constant: c = 2^32 + 977 = 0x1000003D1
//
// Architecture: ARMv8-A (aarch64)
// Tested on: RK3588 (Cortex-A55/A76), Snapdragon, Samsung Exynos
// ============================================================================

#if defined(__aarch64__) || defined(_M_ARM64)

#include <secp256k1/field.hpp>
#include <secp256k1/field_asm.hpp>
#include <secp256k1/config.hpp>
#include <cstdint>
#include <cstring>

namespace secp256k1::fast {
namespace arm64 {

// ============================================================================
// ARM64 256-bit Field Multiplication with secp256k1 Reduction
// ============================================================================
// Full 4×4 limb multiply → 512-bit → reduce mod p in one pass.
// Uses MUL/UMULH pairs (2 cycles each on Cortex-A76, 3 on A55).
// Total: 16 MUL + 16 UMULH + reduction ≈ 50-80 cycles on A76
//
// Strategy: Schoolbook 4×4 multiply into 8 limbs, then secp256k1-specific
// reduction using c = 2^32 + 977.
// ============================================================================

void field_mul_arm64(uint64_t out[4], const uint64_t a[4], const uint64_t b[4]) noexcept {
    // Phase 1: Full 256×256 → 512 bit multiplication
    // Using inline assembly to maximize register utilization and
    // exploit ARM64's ability to do MUL+UMULH in parallel pipelines.
    
    uint64_t t[8]; // 512-bit product
    
    __asm__ __volatile__(
        // ---- Column 0: t[0] ----
        // a[0]*b[0]
        "mul    x8,  %[a0], %[b0]       \n\t"   // lo(a0*b0)
        "umulh  x9,  %[a0], %[b0]       \n\t"   // hi(a0*b0)
        "str    x8,  [%[t], #0]         \n\t"   // t[0] = lo

        // ---- Column 1: t[1] ----
        // a[0]*b[1] + a[1]*b[0] + carry
        "mul    x10, %[a0], %[b1]       \n\t"
        "umulh  x11, %[a0], %[b1]       \n\t"
        "mul    x12, %[a1], %[b0]       \n\t"
        "umulh  x13, %[a1], %[b0]       \n\t"
        "adds   x9,  x9,  x10          \n\t"   // carry_in + lo(a0*b1)
        "adcs   x11, x11, xzr          \n\t"   // propagate carry
        "adds   x9,  x9,  x12          \n\t"   // + lo(a1*b0)
        "adcs   x11, x11, x13          \n\t"   // hi carry chain
        "str    x9,  [%[t], #8]         \n\t"   // t[1]
        "mov    x14, xzr               \n\t"   // overflow accumulator
        "adc    x14, x14, xzr          \n\t"

        // ---- Column 2: t[2] ----
        // a[0]*b[2] + a[1]*b[1] + a[2]*b[0] + carry
        "mul    x8,  %[a0], %[b2]       \n\t"
        "umulh  x9,  %[a0], %[b2]       \n\t"
        "mul    x10, %[a1], %[b1]       \n\t"
        "umulh  x12, %[a1], %[b1]       \n\t"
        "mul    x13, %[a2], %[b0]       \n\t"
        "umulh  x15, %[a2], %[b0]       \n\t"
        "adds   x11, x11, x8           \n\t"
        "adcs   x9,  x9,  x14          \n\t"
        "adc    x14, xzr, xzr          \n\t"
        "adds   x11, x11, x10          \n\t"
        "adcs   x9,  x9,  x12          \n\t"
        "adc    x14, x14, xzr          \n\t"
        "adds   x11, x11, x13          \n\t"
        "adcs   x9,  x9,  x15          \n\t"
        "adc    x14, x14, xzr          \n\t"
        "str    x11, [%[t], #16]        \n\t"   // t[2]

        // ---- Column 3: t[3] ----
        // a[0]*b[3] + a[1]*b[2] + a[2]*b[1] + a[3]*b[0] + carry
        "mul    x8,  %[a0], %[b3]       \n\t"
        "umulh  x10, %[a0], %[b3]       \n\t"
        "mul    x11, %[a1], %[b2]       \n\t"
        "umulh  x12, %[a1], %[b2]       \n\t"
        "mul    x13, %[a2], %[b1]       \n\t"
        "umulh  x15, %[a2], %[b1]       \n\t"
        "adds   x9,  x9,  x8           \n\t"
        "adcs   x10, x10, x14          \n\t"
        "adc    x14, xzr, xzr          \n\t"
        "adds   x9,  x9,  x11          \n\t"
        "adcs   x10, x10, x12          \n\t"
        "adc    x14, x14, xzr          \n\t"
        "adds   x9,  x9,  x13          \n\t"
        "adcs   x10, x10, x15          \n\t"
        "adc    x14, x14, xzr          \n\t"
        "mul    x8,  %[a3], %[b0]       \n\t"
        "umulh  x11, %[a3], %[b0]       \n\t"
        "adds   x9,  x9,  x8           \n\t"
        "adcs   x10, x10, x11          \n\t"
        "adc    x14, x14, xzr          \n\t"
        "str    x9,  [%[t], #24]        \n\t"   // t[3]

        // ---- Column 4: t[4] ----
        // a[1]*b[3] + a[2]*b[2] + a[3]*b[1] + carry
        "mul    x8,  %[a1], %[b3]       \n\t"
        "umulh  x9,  %[a1], %[b3]       \n\t"
        "mul    x11, %[a2], %[b2]       \n\t"
        "umulh  x12, %[a2], %[b2]       \n\t"
        "mul    x13, %[a3], %[b1]       \n\t"
        "umulh  x15, %[a3], %[b1]       \n\t"
        "adds   x10, x10, x8           \n\t"
        "adcs   x9,  x9,  x14          \n\t"
        "adc    x14, xzr, xzr          \n\t"
        "adds   x10, x10, x11          \n\t"
        "adcs   x9,  x9,  x12          \n\t"
        "adc    x14, x14, xzr          \n\t"
        "adds   x10, x10, x13          \n\t"
        "adcs   x9,  x9,  x15          \n\t"
        "adc    x14, x14, xzr          \n\t"
        "str    x10, [%[t], #32]        \n\t"   // t[4]

        // ---- Column 5: t[5] ----
        // a[2]*b[3] + a[3]*b[2] + carry
        "mul    x8,  %[a2], %[b3]       \n\t"
        "umulh  x10, %[a2], %[b3]       \n\t"
        "mul    x11, %[a3], %[b2]       \n\t"
        "umulh  x12, %[a3], %[b2]       \n\t"
        "adds   x9,  x9,  x8           \n\t"
        "adcs   x10, x10, x14          \n\t"
        "adc    x14, xzr, xzr          \n\t"
        "adds   x9,  x9,  x11          \n\t"
        "adcs   x10, x10, x12          \n\t"
        "adc    x14, x14, xzr          \n\t"
        "str    x9,  [%[t], #40]        \n\t"   // t[5]

        // ---- Column 6: t[6] ----
        // a[3]*b[3] + carry
        "mul    x8,  %[a3], %[b3]       \n\t"
        "umulh  x9,  %[a3], %[b3]       \n\t"
        "adds   x10, x10, x8           \n\t"
        "adcs   x9,  x9,  x14          \n\t"
        "str    x10, [%[t], #48]        \n\t"   // t[6]
        "str    x9,  [%[t], #56]        \n\t"   // t[7]

        : /* no output operands — results stored via pointers */
        : [a0] "r"(a[0]), [a1] "r"(a[1]), [a2] "r"(a[2]), [a3] "r"(a[3]),
          [b0] "r"(b[0]), [b1] "r"(b[1]), [b2] "r"(b[2]), [b3] "r"(b[3]),
          [t] "r"(t)
        : "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15",
          "cc", "memory"
    );

    // Phase 2: secp256k1 fast reduction
    // p = 2^256 - c where c = 0x1000003D1
    // t[0..3] + t[4..7] * c (mod 2^256), then normalize
    //
    // For each high limb t[4+i], add t[4+i] * c to position [i..].
    // c = 2^32 + 977, so t[4+i]*c = t[4+i]*(2^32 + 977)
    //   = (t[4+i] << 32) + t[4+i] * 977
    
    constexpr uint64_t C = 0x1000003D1ULL;
    
    uint64_t r0 = t[0], r1 = t[1], r2 = t[2], r3 = t[3];
    
    __asm__ __volatile__(
        // Reduce t[4]: r += t[4] * C
        "mul    x8,  %[t4], %[c]        \n\t"
        "umulh  x9,  %[t4], %[c]        \n\t"
        "adds   %[r0], %[r0], x8        \n\t"
        "adcs   %[r1], %[r1], x9        \n\t"
        "adcs   %[r2], %[r2], xzr       \n\t"
        "adcs   %[r3], %[r3], xzr       \n\t"
        "adc    x14, xzr, xzr           \n\t"

        // Reduce t[5]: r += t[5] * C << 64
        "mul    x8,  %[t5], %[c]        \n\t"
        "umulh  x9,  %[t5], %[c]        \n\t"
        "adds   %[r1], %[r1], x8        \n\t"
        "adcs   %[r2], %[r2], x9        \n\t"
        "adcs   %[r3], %[r3], xzr       \n\t"
        "adc    x14, x14, xzr           \n\t"

        // Reduce t[6]: r += t[6] * C << 128
        "mul    x8,  %[t6], %[c]        \n\t"
        "umulh  x9,  %[t6], %[c]        \n\t"
        "adds   %[r2], %[r2], x8        \n\t"
        "adcs   %[r3], %[r3], x9        \n\t"
        "adc    x14, x14, xzr           \n\t"

        // Reduce t[7]: r += t[7] * C << 192
        "mul    x8,  %[t7], %[c]        \n\t"
        "umulh  x9,  %[t7], %[c]        \n\t"
        "adds   %[r3], %[r3], x8        \n\t"
        "adc    x14, x14, x9            \n\t"

        // Second-round reduction for overflow in x14 (≤ ~0x1000003D1 max)
        "mul    x8,  x14, %[c]          \n\t"
        "umulh  x9,  x14, %[c]          \n\t"
        "adds   %[r0], %[r0], x8        \n\t"
        "adcs   %[r1], %[r1], x9        \n\t"
        "adcs   %[r2], %[r2], xzr       \n\t"
        "adcs   %[r3], %[r3], xzr       \n\t"
        "adc    x14, xzr, xzr           \n\t"

        // Third-round (extremely rare, but mathematically possible)
        "mul    x8,  x14, %[c]          \n\t"
        "adds   %[r0], %[r0], x8        \n\t"
        "adcs   %[r1], %[r1], xzr       \n\t"
        "adcs   %[r2], %[r2], xzr       \n\t"
        "adc    %[r3], %[r3], xzr       \n\t"

        : [r0] "+r"(r0), [r1] "+r"(r1), [r2] "+r"(r2), [r3] "+r"(r3)
        : [t4] "r"(t[4]), [t5] "r"(t[5]), [t6] "r"(t[6]), [t7] "r"(t[7]),
          [c] "r"(C)
        : "x8", "x9", "x14", "cc"
    );
    
    // Final normalization: if r >= p, subtract p (branchless)
    // p = {0xFFFFFFFEFFFFFC2F, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF}
    constexpr uint64_t P0 = 0xFFFFFFFEFFFFFC2FULL;
    constexpr uint64_t P1 = 0xFFFFFFFFFFFFFFFFULL;
    uint64_t s0, s1, s2, s3;
    uint64_t mask;
    
    __asm__ __volatile__(
        // Subtract p
        "subs   %[s0], %[r0], %[p0]         \n\t"
        "sbcs   %[s1], %[r1], %[p1]         \n\t"
        "sbcs   %[s2], %[r2], %[p1]         \n\t"
        "sbcs   %[s3], %[r3], %[p1]         \n\t"
        // mask = borrow ? ~0 : 0
        "csetm  %[mask], cc                 \n\t"
        // Branchless select: out = borrow ? r : s
        "bic    x8,  %[s0], %[mask]         \n\t"
        "and    x9,  %[r0], %[mask]         \n\t"
        "orr    %[s0], x8, x9               \n\t"
        "bic    x8,  %[s1], %[mask]         \n\t"
        "and    x9,  %[r1], %[mask]         \n\t"
        "orr    %[s1], x8, x9               \n\t"
        "bic    x8,  %[s2], %[mask]         \n\t"
        "and    x9,  %[r2], %[mask]         \n\t"
        "orr    %[s2], x8, x9               \n\t"
        "bic    x8,  %[s3], %[mask]         \n\t"
        "and    x9,  %[r3], %[mask]         \n\t"
        "orr    %[s3], x8, x9               \n\t"
        : [s0] "=&r"(s0), [s1] "=&r"(s1), [s2] "=&r"(s2), [s3] "=&r"(s3),
          [mask] "=&r"(mask)
        : [r0] "r"(r0), [r1] "r"(r1), [r2] "r"(r2), [r3] "r"(r3),
          [p0] "r"(P0), [p1] "r"(P1)
        : "x8", "x9", "cc"
    );
    
    out[0] = s0;
    out[1] = s1;
    out[2] = s2;
    out[3] = s3;
}

// ============================================================================
// ARM64 256-bit Field Squaring with secp256k1 Reduction
// ============================================================================
// Exploits symmetry: a[i]*a[j] appears twice for i≠j, so we compute it once
// and double. Only 10 multiplications instead of 16.
// ============================================================================

void field_sqr_arm64(uint64_t out[4], const uint64_t a[4]) noexcept {
    uint64_t t[8];
    
    __asm__ __volatile__(
        // Compute cross products (i<j), each appears 2×
        // We compute them once and will double later
        
        // a[0]*a[1]
        "mul    x8,  %[a0], %[a1]       \n\t"
        "umulh  x9,  %[a0], %[a1]       \n\t"
        
        // a[0]*a[2]
        "mul    x10, %[a0], %[a2]       \n\t"
        "umulh  x11, %[a0], %[a2]       \n\t"
        
        // a[0]*a[3]
        "mul    x12, %[a0], %[a3]       \n\t"
        "umulh  x13, %[a0], %[a3]       \n\t"
        
        // a[1]*a[2]
        "mul    x14, %[a1], %[a2]       \n\t"
        "umulh  x15, %[a1], %[a2]       \n\t"
        
        // a[1]*a[3]
        "mul    x16, %[a1], %[a3]       \n\t"
        "umulh  x17, %[a1], %[a3]       \n\t"
        
        // a[2]*a[3]
        "mul    x19, %[a2], %[a3]       \n\t"
        "umulh  x20, %[a2], %[a3]       \n\t"
        
        // Accumulate cross products into columns
        // col1 = a01_lo
        // col2 = a01_hi + a02_lo
        // col3 = a02_hi + a03_lo + a12_lo
        // col4 = a03_hi + a12_hi + a13_lo
        // col5 = a13_hi + a23_lo
        // col6 = a23_hi
        
        // col2 += a02_lo
        "adds   x9,  x9,  x10          \n\t"   // col2
        "adcs   x11, x11, x12          \n\t"   // col3 partial
        "adcs   x13, x13, xzr          \n\t"   // col4 partial
        "adc    x21, xzr, xzr          \n\t"   // overflow
        
        // col3 += a12_lo
        "adds   x11, x11, x14          \n\t"
        "adcs   x13, x13, x15          \n\t"
        "adc    x21, x21, xzr          \n\t"
        
        // col4 += a13_lo
        "adds   x13, x13, x16          \n\t"
        "adcs   x17, x17, x21          \n\t"   // col5 partial
        "adc    x21, xzr, xzr          \n\t"
        
        // col5 += a23_lo
        "adds   x17, x17, x19          \n\t"
        "adc    x20, x20, x21          \n\t"   // col6
        
        // Double everything (shift left by 1 for 2× cross products)
        "adds   x8,  x8,  x8           \n\t"   // col1 × 2
        "adcs   x9,  x9,  x9           \n\t"   // col2 × 2
        "adcs   x11, x11, x11          \n\t"   // col3 × 2
        "adcs   x13, x13, x13          \n\t"   // col4 × 2
        "adcs   x17, x17, x17          \n\t"   // col5 × 2
        "adcs   x20, x20, x20          \n\t"   // col6 × 2
        "adc    x21, xzr, xzr          \n\t"   // col7 carry
        
        // Now add diagonal products a[i]*a[i]
        // a[0]*a[0]
        "mul    x10, %[a0], %[a0]       \n\t"
        "umulh  x12, %[a0], %[a0]       \n\t"
        "str    x10, [%[t], #0]         \n\t"   // t[0] = lo(a0²)
        "adds   x8,  x8,  x12          \n\t"   // t[1] += hi(a0²)
        "str    x8,  [%[t], #8]         \n\t"   // t[1]
        
        // a[1]*a[1]  
        "mul    x10, %[a1], %[a1]       \n\t"
        "umulh  x12, %[a1], %[a1]       \n\t"
        "adcs   x9,  x9,  x10          \n\t"   // t[2] += lo(a1²)
        "adcs   x11, x11, x12          \n\t"   // t[3] += hi(a1²)
        "str    x9,  [%[t], #16]        \n\t"   // t[2]
        "str    x11, [%[t], #24]        \n\t"   // t[3]
        
        // a[2]*a[2]
        "mul    x10, %[a2], %[a2]       \n\t"
        "umulh  x12, %[a2], %[a2]       \n\t"
        "adcs   x13, x13, x10          \n\t"   // t[4] += lo(a2²)
        "adcs   x17, x17, x12          \n\t"   // t[5] += hi(a2²)
        "str    x13, [%[t], #32]        \n\t"   // t[4]
        "str    x17, [%[t], #40]        \n\t"   // t[5]
        
        // a[3]*a[3]
        "mul    x10, %[a3], %[a3]       \n\t"
        "umulh  x12, %[a3], %[a3]       \n\t"
        "adcs   x20, x20, x10          \n\t"   // t[6] += lo(a3²)
        "adc    x21, x21, x12          \n\t"   // t[7] += hi(a3²)
        "str    x20, [%[t], #48]        \n\t"   // t[6]
        "str    x21, [%[t], #56]        \n\t"   // t[7]
        
        : /* outputs via memory */
        : [a0] "r"(a[0]), [a1] "r"(a[1]), [a2] "r"(a[2]), [a3] "r"(a[3]),
          [t] "r"(t)
        : "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15",
          "x16", "x17", "x19", "x20", "x21",
          "cc", "memory"
    );
    
    // Phase 2: secp256k1 fast reduction (identical to mul)
    constexpr uint64_t C = 0x1000003D1ULL;
    
    uint64_t r0 = t[0], r1 = t[1], r2 = t[2], r3 = t[3];
    
    __asm__ __volatile__(
        "mul    x8,  %[t4], %[c]        \n\t"
        "umulh  x9,  %[t4], %[c]        \n\t"
        "adds   %[r0], %[r0], x8        \n\t"
        "adcs   %[r1], %[r1], x9        \n\t"
        "adcs   %[r2], %[r2], xzr       \n\t"
        "adcs   %[r3], %[r3], xzr       \n\t"
        "adc    x14, xzr, xzr           \n\t"

        "mul    x8,  %[t5], %[c]        \n\t"
        "umulh  x9,  %[t5], %[c]        \n\t"
        "adds   %[r1], %[r1], x8        \n\t"
        "adcs   %[r2], %[r2], x9        \n\t"
        "adcs   %[r3], %[r3], xzr       \n\t"
        "adc    x14, x14, xzr           \n\t"

        "mul    x8,  %[t6], %[c]        \n\t"
        "umulh  x9,  %[t6], %[c]        \n\t"
        "adds   %[r2], %[r2], x8        \n\t"
        "adcs   %[r3], %[r3], x9        \n\t"
        "adc    x14, x14, xzr           \n\t"

        "mul    x8,  %[t7], %[c]        \n\t"
        "umulh  x9,  %[t7], %[c]        \n\t"
        "adds   %[r3], %[r3], x8        \n\t"
        "adc    x14, x14, x9            \n\t"

        "mul    x8,  x14, %[c]          \n\t"
        "umulh  x9,  x14, %[c]          \n\t"
        "adds   %[r0], %[r0], x8        \n\t"
        "adcs   %[r1], %[r1], x9        \n\t"
        "adcs   %[r2], %[r2], xzr       \n\t"
        "adcs   %[r3], %[r3], xzr       \n\t"
        "adc    x14, xzr, xzr           \n\t"

        "mul    x8,  x14, %[c]          \n\t"
        "adds   %[r0], %[r0], x8        \n\t"
        "adcs   %[r1], %[r1], xzr       \n\t"
        "adcs   %[r2], %[r2], xzr       \n\t"
        "adc    %[r3], %[r3], xzr       \n\t"

        : [r0] "+r"(r0), [r1] "+r"(r1), [r2] "+r"(r2), [r3] "+r"(r3)
        : [t4] "r"(t[4]), [t5] "r"(t[5]), [t6] "r"(t[6]), [t7] "r"(t[7]),
          [c] "r"(C)
        : "x8", "x9", "x14", "cc"
    );
    
    // Branchless normalization
    constexpr uint64_t NP0 = 0xFFFFFFFEFFFFFC2FULL;
    constexpr uint64_t NP1 = 0xFFFFFFFFFFFFFFFFULL;
    uint64_t s0, s1, s2, s3, mask;
    __asm__ __volatile__(
        "subs   %[s0], %[r0], %[p0]         \n\t"
        "sbcs   %[s1], %[r1], %[p1]         \n\t"
        "sbcs   %[s2], %[r2], %[p1]         \n\t"
        "sbcs   %[s3], %[r3], %[p1]         \n\t"
        "csetm  %[mask], cc                 \n\t"
        "bic    x8,  %[s0], %[mask]         \n\t"
        "and    x9,  %[r0], %[mask]         \n\t"
        "orr    %[s0], x8, x9               \n\t"
        "bic    x8,  %[s1], %[mask]         \n\t"
        "and    x9,  %[r1], %[mask]         \n\t"
        "orr    %[s1], x8, x9               \n\t"
        "bic    x8,  %[s2], %[mask]         \n\t"
        "and    x9,  %[r2], %[mask]         \n\t"
        "orr    %[s2], x8, x9               \n\t"
        "bic    x8,  %[s3], %[mask]         \n\t"
        "and    x9,  %[r3], %[mask]         \n\t"
        "orr    %[s3], x8, x9               \n\t"
        : [s0] "=&r"(s0), [s1] "=&r"(s1), [s2] "=&r"(s2), [s3] "=&r"(s3),
          [mask] "=&r"(mask)
        : [r0] "r"(r0), [r1] "r"(r1), [r2] "r"(r2), [r3] "r"(r3),
          [p0] "r"(NP0), [p1] "r"(NP1)
        : "x8", "x9", "cc"
    );
    
    out[0] = s0;
    out[1] = s1;
    out[2] = s2;
    out[3] = s3;
}

// ============================================================================
// ARM64 256-bit Field Add/Sub with Branchless Normalization
// ============================================================================

void field_add_arm64(uint64_t out[4], const uint64_t a[4], const uint64_t b[4]) noexcept {
    uint64_t r0, r1, r2, r3;
    uint64_t s0, s1, s2, s3;
    uint64_t mask;
    
    // Prime p limbs — passed as register inputs (ARM64 MOV can't encode arbitrary 64-bit)
    constexpr uint64_t P0 = 0xFFFFFFFEFFFFFC2FULL;
    constexpr uint64_t P1 = 0xFFFFFFFFFFFFFFFFULL;
    
    __asm__ __volatile__(
        // a + b
        "adds   %[r0], %[a0], %[b0]     \n\t"
        "adcs   %[r1], %[a1], %[b1]     \n\t"
        "adcs   %[r2], %[a2], %[b2]     \n\t"
        "adcs   %[r3], %[a3], %[b3]     \n\t"
        "adc    x14, xzr, xzr           \n\t"   // carry into x14
        
        // Subtract p (using register inputs for prime limbs)
        "subs   %[s0], %[r0], %[p0]        \n\t"
        "sbcs   %[s1], %[r1], %[p1]        \n\t"
        "sbcs   %[s2], %[r2], %[p1]        \n\t"
        "sbcs   %[s3], %[r3], %[p1]        \n\t"
        "sbcs   x14,  x14,  xzr            \n\t"  // carry - 0 - borrow
        
        // If x14 == 0 (no borrow from carry): use subtracted
        // If x14 wrapped (borrow): use original sum
        "csetm  %[mask], cc                 \n\t"  // mask = borrow ? ~0 : 0
        
        // Branchless select
        "bic    x8,  %[s0], %[mask]         \n\t"
        "and    x9,  %[r0], %[mask]         \n\t"
        "orr    %[s0], x8, x9               \n\t"
        "bic    x8,  %[s1], %[mask]         \n\t"
        "and    x9,  %[r1], %[mask]         \n\t"
        "orr    %[s1], x8, x9               \n\t"
        "bic    x8,  %[s2], %[mask]         \n\t"
        "and    x9,  %[r2], %[mask]         \n\t"
        "orr    %[s2], x8, x9               \n\t"
        "bic    x8,  %[s3], %[mask]         \n\t"
        "and    x9,  %[r3], %[mask]         \n\t"
        "orr    %[s3], x8, x9               \n\t"
        
        : [r0] "=&r"(r0), [r1] "=&r"(r1), [r2] "=&r"(r2), [r3] "=&r"(r3),
          [s0] "=&r"(s0), [s1] "=&r"(s1), [s2] "=&r"(s2), [s3] "=&r"(s3),
          [mask] "=&r"(mask)
        : [a0] "r"(a[0]), [a1] "r"(a[1]), [a2] "r"(a[2]), [a3] "r"(a[3]),
          [b0] "r"(b[0]), [b1] "r"(b[1]), [b2] "r"(b[2]), [b3] "r"(b[3]),
          [p0] "r"(P0), [p1] "r"(P1)
        : "x8", "x9", "x14", "cc"
    );
    
    out[0] = s0;
    out[1] = s1;
    out[2] = s2;
    out[3] = s3;
}

void field_sub_arm64(uint64_t out[4], const uint64_t a[4], const uint64_t b[4]) noexcept {
    uint64_t r0, r1, r2, r3;
    uint64_t mask;
    
    // p = {0xFFFFFFFEFFFFFC2F, 0xFFFFFFFFFFFFFFFF, ...}
    constexpr uint64_t P0 = 0xFFFFFFFEFFFFFC2FULL;
    constexpr uint64_t P1 = 0xFFFFFFFFFFFFFFFFULL;
    
    __asm__ __volatile__(
        // a - b
        "subs   %[r0], %[a0], %[b0]     \n\t"
        "sbcs   %[r1], %[a1], %[b1]     \n\t"
        "sbcs   %[r2], %[a2], %[b2]     \n\t"
        "sbcs   %[r3], %[a3], %[b3]     \n\t"
        // If borrow: mask = 0xFFFFFFFFFFFFFFFF, else 0
        "csetm  %[mask], cc             \n\t"
        
        // Conditionally add p: r += p & mask
        "and    x8, %[p0], %[mask]      \n\t"
        "and    x9, %[p1], %[mask]      \n\t"
        "adds   %[r0], %[r0], x8        \n\t"
        "adcs   %[r1], %[r1], x9        \n\t"
        "adcs   %[r2], %[r2], x9        \n\t"
        "adc    %[r3], %[r3], x9        \n\t"
        
        : [r0] "=&r"(r0), [r1] "=&r"(r1), [r2] "=&r"(r2), [r3] "=&r"(r3),
          [mask] "=&r"(mask)
        : [a0] "r"(a[0]), [a1] "r"(a[1]), [a2] "r"(a[2]), [a3] "r"(a[3]),
          [b0] "r"(b[0]), [b1] "r"(b[1]), [b2] "r"(b[2]), [b3] "r"(b[3]),
          [p0] "r"(P0), [p1] "r"(P1)
        : "x8", "x9", "cc"
    );
    
    out[0] = r0;
    out[1] = r1;
    out[2] = r2;
    out[3] = r3;
}

// ============================================================================
// ARM64 Field Negate
// ============================================================================

void field_neg_arm64(uint64_t out[4], const uint64_t a[4]) noexcept {
    // -a mod p = p - a (if a != 0), 0 if a == 0
    constexpr uint64_t P0 = 0xFFFFFFFEFFFFFC2FULL;
    constexpr uint64_t P1 = 0xFFFFFFFFFFFFFFFFULL;
    
    uint64_t r0, r1, r2, r3;
    uint64_t is_zero;
    
    __asm__ __volatile__(
        // p - a
        "subs   %[r0], %[p0], %[a0]     \n\t"
        "sbcs   %[r1], %[p1], %[a1]     \n\t"
        "sbcs   %[r2], %[p1], %[a2]     \n\t"
        "sbc    %[r3], %[p1], %[a3]     \n\t"
        
        // Check if a == 0 (branchless)
        "orr    %[iz], %[a0], %[a1]     \n\t"
        "orr    x8,    %[a2], %[a3]     \n\t"
        "orr    %[iz], %[iz], x8        \n\t"
        "cmp    %[iz], #0               \n\t"
        "csel   %[r0], xzr, %[r0], eq   \n\t"
        "csel   %[r1], xzr, %[r1], eq   \n\t"
        "csel   %[r2], xzr, %[r2], eq   \n\t"
        "csel   %[r3], xzr, %[r3], eq   \n\t"
        
        : [r0] "=&r"(r0), [r1] "=&r"(r1), [r2] "=&r"(r2), [r3] "=&r"(r3),
          [iz] "=&r"(is_zero)
        : [a0] "r"(a[0]), [a1] "r"(a[1]), [a2] "r"(a[2]), [a3] "r"(a[3]),
          [p0] "r"(P0), [p1] "r"(P1)
        : "x8", "cc"
    );
    
    out[0] = r0;
    out[1] = r1;
    out[2] = r2;
    out[3] = r3;
}

} // namespace arm64

// ============================================================================
// Public API — FieldElement wrappers
// ============================================================================

FieldElement field_mul_arm64(const FieldElement& a, const FieldElement& b) {
    FieldElement::limbs_type out;
    arm64::field_mul_arm64(out.data(), a.limbs().data(), b.limbs().data());
    return FieldElement::from_limbs(out);
}

FieldElement field_square_arm64(const FieldElement& a) {
    FieldElement::limbs_type out;
    arm64::field_sqr_arm64(out.data(), a.limbs().data());
    return FieldElement::from_limbs(out);
}

FieldElement field_add_arm64(const FieldElement& a, const FieldElement& b) {
    FieldElement::limbs_type out;
    arm64::field_add_arm64(out.data(), a.limbs().data(), b.limbs().data());
    return FieldElement::from_limbs(out);
}

FieldElement field_sub_arm64(const FieldElement& a, const FieldElement& b) {
    FieldElement::limbs_type out;
    arm64::field_sub_arm64(out.data(), a.limbs().data(), b.limbs().data());
    return FieldElement::from_limbs(out);
}

FieldElement field_negate_arm64(const FieldElement& a) {
    FieldElement::limbs_type out;
    arm64::field_neg_arm64(out.data(), a.limbs().data());
    return FieldElement::from_limbs(out);
}

} // namespace secp256k1::fast

#endif // __aarch64__ || _M_ARM64
