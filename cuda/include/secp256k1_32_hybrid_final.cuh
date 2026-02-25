#pragma once

// Smart Hybrid: 32-bit multiplication (native PTX) + 64-bit reduction (proven)
// This file is included AFTER reduce_512_to_256 is defined
// Does NOT redefine FieldElement - uses existing 64-bit FieldElement

// ============================================================================
// 32-bit multiplication using proven Comba's method
// Input: 64-bit FieldElement (4x64) viewed as 32-bit (8x32)
// Output: 512-bit result for reduce_512_to_256
// ============================================================================

// Core 32-bit Comba multiplication -> raw uint32_t[16] output (no packing)
// Separated from wrapper to allow direct use with 32-bit reduction
__device__ __forceinline__ void mul_256_comba32(
    const secp256k1::cuda::FieldElement* a,
    const secp256k1::cuda::FieldElement* b, 
    uint32_t t32[16]
) {
    uint32_t a32[8], b32[8];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        a32[2*i]   = (uint32_t)(a->limbs[i]);
        a32[2*i+1] = (uint32_t)(a->limbs[i] >> 32);
        b32[2*i]   = (uint32_t)(b->limbs[i]);
        b32[2*i+1] = (uint32_t)(b->limbs[i] >> 32);
    }
    
    uint32_t r0 = 0, r1 = 0, r2 = 0;
    
    #define MUL32_ACC(ai, bj) { \
        asm volatile( \
            "mad.lo.cc.u32 %0, %3, %4, %0; \n\t" \
            "madc.hi.cc.u32 %1, %3, %4, %1; \n\t" \
            "addc.u32 %2, %2, 0; \n\t" \
            : "+r"(r0), "+r"(r1), "+r"(r2) \
            : "r"(a32[ai]), "r"(b32[bj]) \
        ); \
    }
    
    // Column 0
    MUL32_ACC(0, 0);
    t32[0] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 1
    MUL32_ACC(0, 1); MUL32_ACC(1, 0);
    t32[1] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 2
    MUL32_ACC(0, 2); MUL32_ACC(1, 1); MUL32_ACC(2, 0);
    t32[2] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 3
    MUL32_ACC(0, 3); MUL32_ACC(1, 2); MUL32_ACC(2, 1); MUL32_ACC(3, 0);
    t32[3] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 4
    MUL32_ACC(0, 4); MUL32_ACC(1, 3); MUL32_ACC(2, 2); MUL32_ACC(3, 1); MUL32_ACC(4, 0);
    t32[4] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 5
    MUL32_ACC(0, 5); MUL32_ACC(1, 4); MUL32_ACC(2, 3); MUL32_ACC(3, 2); MUL32_ACC(4, 1); MUL32_ACC(5, 0);
    t32[5] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 6
    MUL32_ACC(0, 6); MUL32_ACC(1, 5); MUL32_ACC(2, 4); MUL32_ACC(3, 3); MUL32_ACC(4, 2); MUL32_ACC(5, 1); MUL32_ACC(6, 0);
    t32[6] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 7
    MUL32_ACC(0, 7); MUL32_ACC(1, 6); MUL32_ACC(2, 5); MUL32_ACC(3, 4); MUL32_ACC(4, 3); MUL32_ACC(5, 2); MUL32_ACC(6, 1); MUL32_ACC(7, 0);
    t32[7] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 8
    MUL32_ACC(1, 7); MUL32_ACC(2, 6); MUL32_ACC(3, 5); MUL32_ACC(4, 4); MUL32_ACC(5, 3); MUL32_ACC(6, 2); MUL32_ACC(7, 1);
    t32[8] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 9
    MUL32_ACC(2, 7); MUL32_ACC(3, 6); MUL32_ACC(4, 5); MUL32_ACC(5, 4); MUL32_ACC(6, 3); MUL32_ACC(7, 2);
    t32[9] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 10
    MUL32_ACC(3, 7); MUL32_ACC(4, 6); MUL32_ACC(5, 5); MUL32_ACC(6, 4); MUL32_ACC(7, 3);
    t32[10] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 11
    MUL32_ACC(4, 7); MUL32_ACC(5, 6); MUL32_ACC(6, 5); MUL32_ACC(7, 4);
    t32[11] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 12
    MUL32_ACC(5, 7); MUL32_ACC(6, 6); MUL32_ACC(7, 5);
    t32[12] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 13
    MUL32_ACC(6, 7); MUL32_ACC(7, 6);
    t32[13] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 14
    MUL32_ACC(7, 7);
    t32[14] = r0;
    t32[15] = r1;
    
    #undef MUL32_ACC
}

// Legacy wrapper: packs 32-bit output to uint64_t[8] (for Montgomery path)
__device__ __forceinline__ void mul_256_512_hybrid(
    const secp256k1::cuda::FieldElement* a,
    const secp256k1::cuda::FieldElement* b, 
    uint64_t t[8]
) {
    uint32_t t32[16];
    mul_256_comba32(a, b, t32);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        t[i] = ((uint64_t)t32[2*i+1] << 32) | t32[2*i];
    }
}

// ============================================================================
// Optimized 32-bit squaring using Comba's method
// Exploits symmetry: a[i]*a[j] computed once, added twice (except diagonal)
// ~40% fewer multiplications than generic multiplication
// ============================================================================

// Core 32-bit Comba squaring -> raw uint32_t[16] output
__device__ __forceinline__ void sqr_256_comba32(
    const secp256k1::cuda::FieldElement* a,
    uint32_t t32[16]
) {
    uint32_t a32[8];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        a32[2*i]   = (uint32_t)(a->limbs[i]);
        a32[2*i+1] = (uint32_t)(a->limbs[i] >> 32);
    }
    
    uint32_t r0 = 0, r1 = 0, r2 = 0;
    
    // Diagonal multiplication (no doubling)
    #define SQR32_DIAG(ai) { \
        asm volatile( \
            "mad.lo.cc.u32 %0, %3, %3, %0; \n\t" \
            "madc.hi.cc.u32 %1, %3, %3, %1; \n\t" \
            "addc.u32 %2, %2, 0; \n\t" \
            : "+r"(r0), "+r"(r1), "+r"(r2) \
            : "r"(a32[ai]) \
        ); \
    }
    
    // Off-diagonal multiplication (doubled: a[i]*a[j] added twice)
    #define SQR32_MUL2(ai, aj) { \
        uint32_t lo, hi; \
        asm volatile( \
            "mul.lo.u32 %0, %2, %3; \n\t" \
            "mul.hi.u32 %1, %2, %3; \n\t" \
            : "=r"(lo), "=r"(hi) \
            : "r"(a32[ai]), "r"(a32[aj]) \
        ); \
        asm volatile( \
            "add.cc.u32 %0, %0, %3; \n\t" \
            "addc.cc.u32 %1, %1, %4; \n\t" \
            "addc.u32 %2, %2, 0; \n\t" \
            "add.cc.u32 %0, %0, %3; \n\t" \
            "addc.cc.u32 %1, %1, %4; \n\t" \
            "addc.u32 %2, %2, 0; \n\t" \
            : "+r"(r0), "+r"(r1), "+r"(r2) \
            : "r"(lo), "r"(hi) \
        ); \
    }
    
    // Column 0: a[0]*a[0]
    SQR32_DIAG(0);
    t32[0] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 1: 2*a[0]*a[1]
    SQR32_MUL2(0, 1);
    t32[1] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 2: 2*a[0]*a[2] + a[1]*a[1]
    SQR32_MUL2(0, 2);
    SQR32_DIAG(1);
    t32[2] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 3: 2*(a[0]*a[3] + a[1]*a[2])
    SQR32_MUL2(0, 3);
    SQR32_MUL2(1, 2);
    t32[3] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 4: 2*(a[0]*a[4] + a[1]*a[3]) + a[2]*a[2]
    SQR32_MUL2(0, 4);
    SQR32_MUL2(1, 3);
    SQR32_DIAG(2);
    t32[4] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 5: 2*(a[0]*a[5] + a[1]*a[4] + a[2]*a[3])
    SQR32_MUL2(0, 5);
    SQR32_MUL2(1, 4);
    SQR32_MUL2(2, 3);
    t32[5] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 6: 2*(a[0]*a[6] + a[1]*a[5] + a[2]*a[4]) + a[3]*a[3]
    SQR32_MUL2(0, 6);
    SQR32_MUL2(1, 5);
    SQR32_MUL2(2, 4);
    SQR32_DIAG(3);
    t32[6] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 7: 2*(a[0]*a[7] + a[1]*a[6] + a[2]*a[5] + a[3]*a[4])
    SQR32_MUL2(0, 7);
    SQR32_MUL2(1, 6);
    SQR32_MUL2(2, 5);
    SQR32_MUL2(3, 4);
    t32[7] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 8: 2*(a[1]*a[7] + a[2]*a[6] + a[3]*a[5]) + a[4]*a[4]
    SQR32_MUL2(1, 7);
    SQR32_MUL2(2, 6);
    SQR32_MUL2(3, 5);
    SQR32_DIAG(4);
    t32[8] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 9: 2*(a[2]*a[7] + a[3]*a[6] + a[4]*a[5])
    SQR32_MUL2(2, 7);
    SQR32_MUL2(3, 6);
    SQR32_MUL2(4, 5);
    t32[9] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 10: 2*(a[3]*a[7] + a[4]*a[6]) + a[5]*a[5]
    SQR32_MUL2(3, 7);
    SQR32_MUL2(4, 6);
    SQR32_DIAG(5);
    t32[10] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 11: 2*(a[4]*a[7] + a[5]*a[6])
    SQR32_MUL2(4, 7);
    SQR32_MUL2(5, 6);
    t32[11] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 12: 2*a[5]*a[7] + a[6]*a[6]
    SQR32_MUL2(5, 7);
    SQR32_DIAG(6);
    t32[12] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 13: 2*a[6]*a[7]
    SQR32_MUL2(6, 7);
    t32[13] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 14: a[7]*a[7]
    SQR32_DIAG(7);
    t32[14] = r0;
    t32[15] = r1;
    
    #undef SQR32_DIAG
    #undef SQR32_MUL2
}

// Legacy wrapper: packs to uint64_t[8] (for Montgomery path)
__device__ __forceinline__ void sqr_256_512_hybrid(
    const secp256k1::cuda::FieldElement* a,
    uint64_t t[8]
) {
    uint32_t t32[16];
    sqr_256_comba32(a, t32);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        t[i] = ((uint64_t)t32[2*i+1] << 32) | t32[2*i];
    }
}

// ============================================================================
// 32-bit secp256k1 reduction (consumer GPU optimized)
// On consumer NVIDIA GPUs (Turing/Ampere/Ada/Blackwell), INT64 multiply
// throughput is 1/32 of INT32. By doing the main T_hi x K_MOD multiplication
// in 32-bit, we avoid the INT64 multiply bottleneck.
// Phase 1+2: fully 32-bit (T_hi x K_MOD + add to T_lo)
// Phase 3+4: 64-bit (overflow handling + conditional subtraction -- proven code)
// ============================================================================
__device__ __forceinline__ void reduce_512_to_256_32(
    uint32_t t32[16],
    secp256k1::cuda::FieldElement* r
) {
    uint32_t t0 = t32[0], t1 = t32[1], t2 = t32[2], t3 = t32[3];
    uint32_t t4 = t32[4], t5 = t32[5], t6 = t32[6], t7 = t32[7];
    const uint32_t t8  = t32[8],  t9  = t32[9],  t10 = t32[10], t11 = t32[11];
    const uint32_t t12 = t32[12], t13 = t32[13], t14 = t32[14], t15 = t32[15];

    // ---- Phase 1: A = T_hi x 977 (32-bit scalar MAD chain -> 9 limbs) ----
    uint32_t a0, a1, a2, a3, a4, a5, a6, a7, a8;
    asm volatile(
        "mul.lo.u32 %0, %9, 977;\n\t"
        "mul.hi.u32 %1, %9, 977;\n\t"
        "mad.lo.cc.u32 %1, %10, 977, %1;\n\t"
        "madc.hi.u32 %2, %10, 977, 0;\n\t"
        "mad.lo.cc.u32 %2, %11, 977, %2;\n\t"
        "madc.hi.u32 %3, %11, 977, 0;\n\t"
        "mad.lo.cc.u32 %3, %12, 977, %3;\n\t"
        "madc.hi.u32 %4, %12, 977, 0;\n\t"
        "mad.lo.cc.u32 %4, %13, 977, %4;\n\t"
        "madc.hi.u32 %5, %13, 977, 0;\n\t"
        "mad.lo.cc.u32 %5, %14, 977, %5;\n\t"
        "madc.hi.u32 %6, %14, 977, 0;\n\t"
        "mad.lo.cc.u32 %6, %15, 977, %6;\n\t"
        "madc.hi.u32 %7, %15, 977, 0;\n\t"
        "mad.lo.cc.u32 %7, %16, 977, %7;\n\t"
        "madc.hi.u32 %8, %16, 977, 0;\n\t"
        : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3), "=r"(a4),
          "=r"(a5), "=r"(a6), "=r"(a7), "=r"(a8)
        : "r"(t8), "r"(t9), "r"(t10), "r"(t11),
          "r"(t12), "r"(t13), "r"(t14), "r"(t15)
    );

    // ---- Phase 1b: Add T_hi << 32 (shift by 1 limb = x2^32 component of K_MOD) ----
    uint32_t a9;
    asm volatile(
        "add.cc.u32 %0, %0, %9;\n\t"
        "addc.cc.u32 %1, %1, %10;\n\t"
        "addc.cc.u32 %2, %2, %11;\n\t"
        "addc.cc.u32 %3, %3, %12;\n\t"
        "addc.cc.u32 %4, %4, %13;\n\t"
        "addc.cc.u32 %5, %5, %14;\n\t"
        "addc.cc.u32 %6, %6, %15;\n\t"
        "addc.cc.u32 %7, %7, %16;\n\t"
        "addc.u32 %8, 0, 0;\n\t"
        : "+r"(a1), "+r"(a2), "+r"(a3), "+r"(a4),
          "+r"(a5), "+r"(a6), "+r"(a7), "+r"(a8), "=r"(a9)
        : "r"(t8), "r"(t9), "r"(t10), "r"(t11),
          "r"(t12), "r"(t13), "r"(t14), "r"(t15)
    );

    // ---- Phase 2: T_lo[0..7] += R[0..7] (32-bit carry chain) ----
    uint32_t carry;
    asm volatile(
        "add.cc.u32 %0, %0, %9;\n\t"
        "addc.cc.u32 %1, %1, %10;\n\t"
        "addc.cc.u32 %2, %2, %11;\n\t"
        "addc.cc.u32 %3, %3, %12;\n\t"
        "addc.cc.u32 %4, %4, %13;\n\t"
        "addc.cc.u32 %5, %5, %14;\n\t"
        "addc.cc.u32 %6, %6, %15;\n\t"
        "addc.cc.u32 %7, %7, %16;\n\t"
        "addc.u32 %8, 0, 0;\n\t"
        : "+r"(t0), "+r"(t1), "+r"(t2), "+r"(t3),
          "+r"(t4), "+r"(t5), "+r"(t6), "+r"(t7), "=r"(carry)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(a4), "r"(a5), "r"(a6), "r"(a7)
    );

    // ---- Phase 3: Overflow reduction (64-bit for proven correctness) ----
    // Pack lower 256 bits to 64-bit
    uint64_t r0 = ((uint64_t)t1 << 32) | t0;
    uint64_t r1 = ((uint64_t)t3 << 32) | t2;
    uint64_t r2 = ((uint64_t)t5 << 32) | t4;
    uint64_t r3 = ((uint64_t)t7 << 32) | t6;

    // extra = a8 + carry + a9 * 2^32 (up to ~2^33)
    uint64_t extra = (uint64_t)a8 + carry + ((uint64_t)a9 << 32);
    uint64_t ek_lo, ek_hi;
    asm volatile(
        "mul.lo.u64 %0, %2, %3;\n\t"
        "mul.hi.u64 %1, %2, %3;\n\t"
        : "=l"(ek_lo), "=l"(ek_hi)
        : "l"(extra), "l"((uint64_t)0x1000003D1ULL)
    );

    uint64_t c;
    asm volatile(
        "add.cc.u64 %0, %0, %5;\n\t"
        "addc.cc.u64 %1, %1, %6;\n\t"
        "addc.cc.u64 %2, %2, 0;\n\t"
        "addc.cc.u64 %3, %3, 0;\n\t"
        "addc.u64 %4, 0, 0;\n\t"
        : "+l"(r0), "+l"(r1), "+l"(r2), "+l"(r3), "=l"(c)
        : "l"(ek_lo), "l"(ek_hi)
    );

    if (c) {
        asm volatile(
            "add.cc.u64 %0, %0, %4;\n\t"
            "addc.cc.u64 %1, %1, 0;\n\t"
            "addc.cc.u64 %2, %2, 0;\n\t"
            "addc.u64 %3, %3, 0;\n\t"
            : "+l"(r0), "+l"(r1), "+l"(r2), "+l"(r3)
            : "l"((uint64_t)0x1000003D1ULL)
        );
    }

    // ---- Phase 4: Conditional subtraction of P ----
    uint64_t s0, s1, s2, s3, borrow;
    asm volatile(
        "sub.cc.u64 %0, %5, %9;\n\t"
        "subc.cc.u64 %1, %6, %10;\n\t"
        "subc.cc.u64 %2, %7, %11;\n\t"
        "subc.cc.u64 %3, %8, %12;\n\t"
        "subc.u64 %4, 0, 0;\n\t"
        : "=l"(s0), "=l"(s1), "=l"(s2), "=l"(s3), "=l"(borrow)
        : "l"(r0), "l"(r1), "l"(r2), "l"(r3),
          "l"(MODULUS[0]), "l"(MODULUS[1]), "l"(MODULUS[2]), "l"(MODULUS[3])
    );

    if (borrow == 0) {
        r->limbs[0] = s0; r->limbs[1] = s1; r->limbs[2] = s2; r->limbs[3] = s3;
    } else {
        r->limbs[0] = r0; r->limbs[1] = r1; r->limbs[2] = r2; r->limbs[3] = r3;
    }
}

// ============================================================================
// Hybrid field operations: 32-bit mul/sqr + 32-bit reduce (optimized)
// Consumer GPUs have INT32 multiply throughput 32x higher than INT64.
// By keeping the main reduction in 32-bit, we avoid the INT64 bottleneck.
// ============================================================================

__device__ __forceinline__ void field_mul_hybrid(
    const secp256k1::cuda::FieldElement* a,
    const secp256k1::cuda::FieldElement* b,
    secp256k1::cuda::FieldElement* r
) {
    uint32_t t32[16];
    mul_256_comba32(a, b, t32);
    reduce_512_to_256_32(t32, r);
}

__device__ __forceinline__ void field_sqr_hybrid(
    const secp256k1::cuda::FieldElement* a,
    secp256k1::cuda::FieldElement* r
) {
    uint32_t t32[16];
    sqr_256_comba32(a, t32);
    reduce_512_to_256_32(t32, r);
}

// ============================================================================
// Montgomery hybrid operations: 32-bit mul/sqr + Montgomery reduction
// These use fast 32-bit multiplication but Montgomery-specific reduction
// ============================================================================

__device__ __forceinline__ void field_mul_mont_hybrid(
    const secp256k1::cuda::FieldElement* a,
    const secp256k1::cuda::FieldElement* b,
    secp256k1::cuda::FieldElement* r
) {
    uint64_t t[8];
    mul_256_512_hybrid(a, b, t);  // Fast 32-bit PTX multiplication!
    mont_reduce_512(t, r);         // Montgomery reduction
}

__device__ __forceinline__ void field_sqr_mont_hybrid(
    const secp256k1::cuda::FieldElement* a,
    secp256k1::cuda::FieldElement* r
) {
    uint64_t t[8];
    sqr_256_512_hybrid(a, t);  // Fast optimized 32-bit squaring!
    mont_reduce_512(t, r);      // Montgomery reduction
}
