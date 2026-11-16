// =============================================================================
// UltrafastSecp256k1 Metal -- Field Arithmetic (secp256k1_field.h)
// =============================================================================
// secp256k1 field: F_p where p = 2^256 - 2^32 - 977
// 256-bit integers using 8x32-bit limbs (little-endian)
//
// Metal Shading Language (MSL) does NOT support 64-bit integer types in
// shader functions, so all arithmetic uses 32-bit operations with explicit
// carry propagation. Apple Silicon GPUs have outstanding 32-bit ALU
// throughput -- this is the natural representation.
//
// ACCELERATION STRATEGY (PTX equivalent for Metal):
//   - Comba product scanning: column-by-column accumulation like PTX MAD_ACC
//   - Fused multiply-add: ulong(a)*ulong(b)+carry -> compiler maps to MAC
//   - Full loop unrolling: no dynamic indexing overhead
//   - Branchless reduction: constant-time modular arithmetic
//   - mad24() where applicable: Apple Silicon 1-cycle 24-bit multiply-add
//
// Matching CUDA: FieldElement{uint64_t limbs[4]} <-> Metal FieldElement{uint limbs[8]}
// Memory layout is identical -- just different view granularity.
// =============================================================================

#pragma once

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Constants
// =============================================================================

// Field prime p = 2^256 - 0x1000003D1
// 8x32-bit limbs (little-endian)
constant uint SECP256K1_P[8] = {
    0xFFFFFC2Fu, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
    0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
};

// K = 2^32 + 977 = 0x1000003D1 -- reduction constant
// K split into two 32-bit limbs
constant uint K_LO = 0x3D1u;   // low 32 bits of K (= 977)
constant uint K_HI = 0x1u;     // high 32 bits of K

// =============================================================================
// Field Element Type -- 256-bit as 8x32-bit limbs
// =============================================================================

struct FieldElement {
    uint limbs[8];  // Little-endian: limbs[0] = bits [0..31]
};

// =============================================================================
// Multiply-Accumulate Macro -- Metal equivalent of PTX MAD_ACC
// =============================================================================
// Accumulates:  (c0, c1, c2) += a * b
// where (c0, c1, c2) is a 96-bit running accumulator.
//
// CUDA PTX does this with:  mad.lo.cc.u64 / madc.hi.cc.u64 / addc.u64
// On Metal, the compiler maps ulong arithmetic to the ALU's native MAC units.
// Apple Silicon's 32-bit FMA pipeline handles ulong->uint2 splits efficiently.

#define METAL_MAD_ACC(c0, c1, c2, a, b)            \
    do {                                            \
        ulong _p = ulong(a) * ulong(b);            \
        ulong _s0 = ulong(c0) + uint(_p);          \
        (c0) = uint(_s0);                           \
        ulong _s1 = ulong(c1) + uint(_p >> 32)     \
                   + uint(_s0 >> 32);               \
        (c1) = uint(_s1);                           \
        (c2) += uint(_s1 >> 32);                    \
    } while(0)

// =============================================================================
// Field Element Init / Query
// =============================================================================

inline FieldElement field_zero() {
    FieldElement r;
    r.limbs[0] = 0; r.limbs[1] = 0; r.limbs[2] = 0; r.limbs[3] = 0;
    r.limbs[4] = 0; r.limbs[5] = 0; r.limbs[6] = 0; r.limbs[7] = 0;
    return r;
}

inline FieldElement field_one() {
    FieldElement r = field_zero();
    r.limbs[0] = 1;
    return r;
}

inline bool field_is_zero(thread const FieldElement &a) {
    return (a.limbs[0] | a.limbs[1] | a.limbs[2] | a.limbs[3] |
            a.limbs[4] | a.limbs[5] | a.limbs[6] | a.limbs[7]) == 0;
}

inline bool field_eq(thread const FieldElement &a, thread const FieldElement &b) {
    uint d = 0;
    for (int i = 0; i < 8; i++) d |= (a.limbs[i] ^ b.limbs[i]);
    return d == 0;
}

// =============================================================================
// Field Addition: r = (a + b) mod p -- Branchless
// =============================================================================

inline FieldElement field_add(thread const FieldElement &a, thread const FieldElement &b) {
    FieldElement r;
    ulong carry = 0;
    for (int i = 0; i < 8; i++) {
        ulong s = ulong(a.limbs[i]) + ulong(b.limbs[i]) + carry;
        r.limbs[i] = uint(s);
        carry = s >> 32;
    }

    // Conditional subtract p (branchless)
    ulong borrow = 0;
    uint diff[8];
    for (int i = 0; i < 8; i++) {
        ulong d = ulong(r.limbs[i]) - ulong(SECP256K1_P[i]) - borrow;
        diff[i] = uint(d);
        borrow = (d >> 63); // bit 63 = sign
    }

    // Select: if (carry || !borrow) -> use diff, else keep r
    uint use_diff = uint(carry) | uint(borrow == 0);
    uint mask = -use_diff; // 0xFFFFFFFF or 0x00000000
    uint nmask = ~mask;
    for (int i = 0; i < 8; i++) {
        r.limbs[i] = (diff[i] & mask) | (r.limbs[i] & nmask);
    }
    return r;
}

// =============================================================================
// Field Subtraction: r = (a - b) mod p -- Branchless
// =============================================================================

inline FieldElement field_sub(thread const FieldElement &a, thread const FieldElement &b) {
    FieldElement r;
    ulong borrow = 0;
    for (int i = 0; i < 8; i++) {
        ulong d = ulong(a.limbs[i]) - ulong(b.limbs[i]) - borrow;
        r.limbs[i] = uint(d);
        borrow = (d >> 63);
    }
    // If borrow, add p -- branchless
    uint mask = -(uint(borrow));
    ulong carry = 0;
    for (int i = 0; i < 8; i++) {
        ulong s = ulong(r.limbs[i]) + ulong(SECP256K1_P[i] & mask) + carry;
        r.limbs[i] = uint(s);
        carry = s >> 32;
    }
    return r;
}

// =============================================================================
// Field Negation: r = p - a (mod p)
// =============================================================================

inline FieldElement field_negate(thread const FieldElement &a) {
    FieldElement r;
    ulong borrow = 0;
    for (int i = 0; i < 8; i++) {
        ulong d = ulong(SECP256K1_P[i]) - ulong(a.limbs[i]) - borrow;
        r.limbs[i] = uint(d);
        borrow = (d >> 63);
    }
    return r;
}

// =============================================================================
// 512->256 Reduction: prod[16] mod p
// p = 2^256 - K, K = 2^32 + 977
// Two-pass fold with branchless final subtraction
// =============================================================================

inline FieldElement field_reduce_512(thread const uint prod[16]) {
    // Accumulator: 9 limbs to handle potential overflow
    ulong acc[9];
    for (int i = 0; i < 8; i++) acc[i] = prod[i];
    acc[8] = 0;

    // Pass 1: Fold high[0..7] x K_LO (= 977)
    for (int i = 0; i < 8; i++) {
        ulong p = ulong(prod[8 + i]) * ulong(K_LO);
        acc[i]     += uint(p);
        acc[i + 1] += uint(p >> 32);
    }

    // Pass 2: Fold high[0..7] x K_HI (= 1) -> shift-add by 32 bits
    for (int i = 0; i < 8; i++) {
        acc[i + 1] += prod[8 + i];
    }

    // Propagate carries through accumulator
    for (int i = 0; i < 8; i++) {
        acc[i + 1] += acc[i] >> 32;
        acc[i] = uint(acc[i]);
    }

    // Second reduction: fold acc[8]·K back into limbs [0..7].
    // acc[8] can exceed 32 bits after first fold (issue #226), absorbed in at
    // most 2 folds. CONSTANT-TIME: was `while (acc[8] != 0)` — a data-dependent
    // loop whose trip count (0/1/2) leaks the secret-derived overflow during
    // signing. Now a FIXED 2 iterations: when acc[8]==0 the fold uses extra==0
    // and is a no-op, so the result is identical for any input (mirrors the
    // CUDA/OpenCL masked rare-carry fold). Verified by test_exploit_metal_field_reduce.
    for (int iter = 0; iter < 2; iter++) {
        ulong extra = acc[8];
        acc[8] = 0;
        ulong p = extra * ulong(K_LO);
        acc[0] += p & 0xFFFFFFFF;
        acc[1] += (p >> 32);
        // K_HI = 1: add extra at one-limb offset
        acc[1] += extra & 0xFFFFFFFF;
        acc[2] += extra >> 32;

        // Re-propagate carries
        for (int i = 0; i < 8; i++) {
            acc[i + 1] += acc[i] >> 32;
            acc[i] = uint(acc[i]);
        }
    }

    // Final branchless conditional subtract of p
    ulong borrow = 0;
    uint diff[8];
    for (int i = 0; i < 8; i++) {
        ulong d = ulong(uint(acc[i])) - ulong(SECP256K1_P[i]) - borrow;
        diff[i] = uint(d);
        borrow = (d >> 63);
    }

    FieldElement r;
    uint mask = -(uint(borrow == 0)); // if no borrow -> use diff
    uint nmask = ~mask;
    for (int i = 0; i < 8; i++) {
        r.limbs[i] = (diff[i] & mask) | (uint(acc[i]) & nmask);
    }
    return r;
}

// =============================================================================
// Field Multiplication: r = (a * b) mod p
// Comba Product Scanning -- Metal equivalent of CUDA's PTX mul_256_512
//
// Uses METAL_MAD_ACC macro: column-by-column accumulation with
// carry chain, matching the PTX mad.lo.cc / madc.hi.cc pattern.
// Fully unrolled -- no dynamic loop indexing.
// =============================================================================

inline FieldElement field_mul(thread const FieldElement &a, thread const FieldElement &b) {
    uint prod[16];

    uint a0 = a.limbs[0], a1 = a.limbs[1], a2 = a.limbs[2], a3 = a.limbs[3];
    uint a4 = a.limbs[4], a5 = a.limbs[5], a6 = a.limbs[6], a7 = a.limbs[7];
    uint b0 = b.limbs[0], b1 = b.limbs[1], b2 = b.limbs[2], b3 = b.limbs[3];
    uint b4 = b.limbs[4], b5 = b.limbs[5], b6 = b.limbs[6], b7 = b.limbs[7];

    uint c0 = 0, c1 = 0, c2 = 0;

    // Column 0: a0*b0
    METAL_MAD_ACC(c0, c1, c2, a0, b0);
    prod[0] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 1: a0*b1 + a1*b0
    METAL_MAD_ACC(c0, c1, c2, a0, b1);
    METAL_MAD_ACC(c0, c1, c2, a1, b0);
    prod[1] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 2: a0*b2 + a1*b1 + a2*b0
    METAL_MAD_ACC(c0, c1, c2, a0, b2);
    METAL_MAD_ACC(c0, c1, c2, a1, b1);
    METAL_MAD_ACC(c0, c1, c2, a2, b0);
    prod[2] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 3: a0*b3 + a1*b2 + a2*b1 + a3*b0
    METAL_MAD_ACC(c0, c1, c2, a0, b3);
    METAL_MAD_ACC(c0, c1, c2, a1, b2);
    METAL_MAD_ACC(c0, c1, c2, a2, b1);
    METAL_MAD_ACC(c0, c1, c2, a3, b0);
    prod[3] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 4: a0*b4 + a1*b3 + a2*b2 + a3*b1 + a4*b0
    METAL_MAD_ACC(c0, c1, c2, a0, b4);
    METAL_MAD_ACC(c0, c1, c2, a1, b3);
    METAL_MAD_ACC(c0, c1, c2, a2, b2);
    METAL_MAD_ACC(c0, c1, c2, a3, b1);
    METAL_MAD_ACC(c0, c1, c2, a4, b0);
    prod[4] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 5: a0*b5 + a1*b4 + a2*b3 + a3*b2 + a4*b1 + a5*b0
    METAL_MAD_ACC(c0, c1, c2, a0, b5);
    METAL_MAD_ACC(c0, c1, c2, a1, b4);
    METAL_MAD_ACC(c0, c1, c2, a2, b3);
    METAL_MAD_ACC(c0, c1, c2, a3, b2);
    METAL_MAD_ACC(c0, c1, c2, a4, b1);
    METAL_MAD_ACC(c0, c1, c2, a5, b0);
    prod[5] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 6: a0*b6 + a1*b5 + a2*b4 + a3*b3 + a4*b2 + a5*b1 + a6*b0
    METAL_MAD_ACC(c0, c1, c2, a0, b6);
    METAL_MAD_ACC(c0, c1, c2, a1, b5);
    METAL_MAD_ACC(c0, c1, c2, a2, b4);
    METAL_MAD_ACC(c0, c1, c2, a3, b3);
    METAL_MAD_ACC(c0, c1, c2, a4, b2);
    METAL_MAD_ACC(c0, c1, c2, a5, b1);
    METAL_MAD_ACC(c0, c1, c2, a6, b0);
    prod[6] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 7: a0*b7 + a1*b6 + a2*b5 + a3*b4 + a4*b3 + a5*b2 + a6*b1 + a7*b0
    METAL_MAD_ACC(c0, c1, c2, a0, b7);
    METAL_MAD_ACC(c0, c1, c2, a1, b6);
    METAL_MAD_ACC(c0, c1, c2, a2, b5);
    METAL_MAD_ACC(c0, c1, c2, a3, b4);
    METAL_MAD_ACC(c0, c1, c2, a4, b3);
    METAL_MAD_ACC(c0, c1, c2, a5, b2);
    METAL_MAD_ACC(c0, c1, c2, a6, b1);
    METAL_MAD_ACC(c0, c1, c2, a7, b0);
    prod[7] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 8: a1*b7 + a2*b6 + a3*b5 + a4*b4 + a5*b3 + a6*b2 + a7*b1
    METAL_MAD_ACC(c0, c1, c2, a1, b7);
    METAL_MAD_ACC(c0, c1, c2, a2, b6);
    METAL_MAD_ACC(c0, c1, c2, a3, b5);
    METAL_MAD_ACC(c0, c1, c2, a4, b4);
    METAL_MAD_ACC(c0, c1, c2, a5, b3);
    METAL_MAD_ACC(c0, c1, c2, a6, b2);
    METAL_MAD_ACC(c0, c1, c2, a7, b1);
    prod[8] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 9: a2*b7 + a3*b6 + a4*b5 + a5*b4 + a6*b3 + a7*b2
    METAL_MAD_ACC(c0, c1, c2, a2, b7);
    METAL_MAD_ACC(c0, c1, c2, a3, b6);
    METAL_MAD_ACC(c0, c1, c2, a4, b5);
    METAL_MAD_ACC(c0, c1, c2, a5, b4);
    METAL_MAD_ACC(c0, c1, c2, a6, b3);
    METAL_MAD_ACC(c0, c1, c2, a7, b2);
    prod[9] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 10: a3*b7 + a4*b6 + a5*b5 + a6*b4 + a7*b3
    METAL_MAD_ACC(c0, c1, c2, a3, b7);
    METAL_MAD_ACC(c0, c1, c2, a4, b6);
    METAL_MAD_ACC(c0, c1, c2, a5, b5);
    METAL_MAD_ACC(c0, c1, c2, a6, b4);
    METAL_MAD_ACC(c0, c1, c2, a7, b3);
    prod[10] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 11: a4*b7 + a5*b6 + a6*b5 + a7*b4
    METAL_MAD_ACC(c0, c1, c2, a4, b7);
    METAL_MAD_ACC(c0, c1, c2, a5, b6);
    METAL_MAD_ACC(c0, c1, c2, a6, b5);
    METAL_MAD_ACC(c0, c1, c2, a7, b4);
    prod[11] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 12: a5*b7 + a6*b6 + a7*b5
    METAL_MAD_ACC(c0, c1, c2, a5, b7);
    METAL_MAD_ACC(c0, c1, c2, a6, b6);
    METAL_MAD_ACC(c0, c1, c2, a7, b5);
    prod[12] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 13: a6*b7 + a7*b6
    METAL_MAD_ACC(c0, c1, c2, a6, b7);
    METAL_MAD_ACC(c0, c1, c2, a7, b6);
    prod[13] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 14: a7*b7
    METAL_MAD_ACC(c0, c1, c2, a7, b7);
    prod[14] = c0;
    prod[15] = c1;

    return field_reduce_512(prod);
}

// =============================================================================
// Field Squaring: r = a^2 mod p
// Comba Product Scanning -- exploits symmetry (off-diagonal doubled)
//
// For squaring, a[i]*a[j] == a[j]*a[i], so off-diagonal products appear
// twice. We compute them once and double, then add diagonal terms.
// Total: 36 multiplications instead of 64 (8x8 schoolbook).
// =============================================================================

inline FieldElement field_sqr(thread const FieldElement &a) {
    uint a0 = a.limbs[0], a1 = a.limbs[1], a2 = a.limbs[2], a3 = a.limbs[3];
    uint a4 = a.limbs[4], a5 = a.limbs[5], a6 = a.limbs[6], a7 = a.limbs[7];

    uint prod[16];
    uint c0 = 0, c1 = 0, c2 = 0;

    // Column 0: a0*a0 (diagonal only)
    METAL_MAD_ACC(c0, c1, c2, a0, a0);
    prod[0] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 1: 2 * a0*a1
    METAL_MAD_ACC(c0, c1, c2, a0, a1);
    METAL_MAD_ACC(c0, c1, c2, a0, a1);
    prod[1] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 2: 2 * a0*a2 + a1*a1
    METAL_MAD_ACC(c0, c1, c2, a0, a2);
    METAL_MAD_ACC(c0, c1, c2, a0, a2);
    METAL_MAD_ACC(c0, c1, c2, a1, a1);
    prod[2] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 3: 2 * (a0*a3 + a1*a2)
    METAL_MAD_ACC(c0, c1, c2, a0, a3);
    METAL_MAD_ACC(c0, c1, c2, a0, a3);
    METAL_MAD_ACC(c0, c1, c2, a1, a2);
    METAL_MAD_ACC(c0, c1, c2, a1, a2);
    prod[3] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 4: 2 * (a0*a4 + a1*a3) + a2*a2
    METAL_MAD_ACC(c0, c1, c2, a0, a4);
    METAL_MAD_ACC(c0, c1, c2, a0, a4);
    METAL_MAD_ACC(c0, c1, c2, a1, a3);
    METAL_MAD_ACC(c0, c1, c2, a1, a3);
    METAL_MAD_ACC(c0, c1, c2, a2, a2);
    prod[4] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 5: 2 * (a0*a5 + a1*a4 + a2*a3)
    METAL_MAD_ACC(c0, c1, c2, a0, a5);
    METAL_MAD_ACC(c0, c1, c2, a0, a5);
    METAL_MAD_ACC(c0, c1, c2, a1, a4);
    METAL_MAD_ACC(c0, c1, c2, a1, a4);
    METAL_MAD_ACC(c0, c1, c2, a2, a3);
    METAL_MAD_ACC(c0, c1, c2, a2, a3);
    prod[5] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 6: 2 * (a0*a6 + a1*a5 + a2*a4) + a3*a3
    METAL_MAD_ACC(c0, c1, c2, a0, a6);
    METAL_MAD_ACC(c0, c1, c2, a0, a6);
    METAL_MAD_ACC(c0, c1, c2, a1, a5);
    METAL_MAD_ACC(c0, c1, c2, a1, a5);
    METAL_MAD_ACC(c0, c1, c2, a2, a4);
    METAL_MAD_ACC(c0, c1, c2, a2, a4);
    METAL_MAD_ACC(c0, c1, c2, a3, a3);
    prod[6] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 7: 2 * (a0*a7 + a1*a6 + a2*a5 + a3*a4)
    METAL_MAD_ACC(c0, c1, c2, a0, a7);
    METAL_MAD_ACC(c0, c1, c2, a0, a7);
    METAL_MAD_ACC(c0, c1, c2, a1, a6);
    METAL_MAD_ACC(c0, c1, c2, a1, a6);
    METAL_MAD_ACC(c0, c1, c2, a2, a5);
    METAL_MAD_ACC(c0, c1, c2, a2, a5);
    METAL_MAD_ACC(c0, c1, c2, a3, a4);
    METAL_MAD_ACC(c0, c1, c2, a3, a4);
    prod[7] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 8: 2 * (a1*a7 + a2*a6 + a3*a5) + a4*a4
    METAL_MAD_ACC(c0, c1, c2, a1, a7);
    METAL_MAD_ACC(c0, c1, c2, a1, a7);
    METAL_MAD_ACC(c0, c1, c2, a2, a6);
    METAL_MAD_ACC(c0, c1, c2, a2, a6);
    METAL_MAD_ACC(c0, c1, c2, a3, a5);
    METAL_MAD_ACC(c0, c1, c2, a3, a5);
    METAL_MAD_ACC(c0, c1, c2, a4, a4);
    prod[8] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 9: 2 * (a2*a7 + a3*a6 + a4*a5)
    METAL_MAD_ACC(c0, c1, c2, a2, a7);
    METAL_MAD_ACC(c0, c1, c2, a2, a7);
    METAL_MAD_ACC(c0, c1, c2, a3, a6);
    METAL_MAD_ACC(c0, c1, c2, a3, a6);
    METAL_MAD_ACC(c0, c1, c2, a4, a5);
    METAL_MAD_ACC(c0, c1, c2, a4, a5);
    prod[9] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 10: 2 * (a3*a7 + a4*a6) + a5*a5
    METAL_MAD_ACC(c0, c1, c2, a3, a7);
    METAL_MAD_ACC(c0, c1, c2, a3, a7);
    METAL_MAD_ACC(c0, c1, c2, a4, a6);
    METAL_MAD_ACC(c0, c1, c2, a4, a6);
    METAL_MAD_ACC(c0, c1, c2, a5, a5);
    prod[10] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 11: 2 * (a4*a7 + a5*a6)
    METAL_MAD_ACC(c0, c1, c2, a4, a7);
    METAL_MAD_ACC(c0, c1, c2, a4, a7);
    METAL_MAD_ACC(c0, c1, c2, a5, a6);
    METAL_MAD_ACC(c0, c1, c2, a5, a6);
    prod[11] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 12: 2 * a5*a7 + a6*a6
    METAL_MAD_ACC(c0, c1, c2, a5, a7);
    METAL_MAD_ACC(c0, c1, c2, a5, a7);
    METAL_MAD_ACC(c0, c1, c2, a6, a6);
    prod[12] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 13: 2 * a6*a7
    METAL_MAD_ACC(c0, c1, c2, a6, a7);
    METAL_MAD_ACC(c0, c1, c2, a6, a7);
    prod[13] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 14: a7*a7
    METAL_MAD_ACC(c0, c1, c2, a7, a7);
    prod[14] = c0;
    prod[15] = c1;

    return field_reduce_512(prod);
}

// =============================================================================
// Repeated Squaring: r = a^(2^n) -- in place
// =============================================================================

inline FieldElement field_sqr_n(thread const FieldElement &a, int n) {
    FieldElement r = a;
    for (int i = 0; i < n; i++) {
        r = field_sqr(r);
    }
    return r;
}

// =============================================================================
// Field Multiplication by Small Constant -- branchless reduction
// =============================================================================

inline FieldElement field_mul_small(thread const FieldElement &a, uint small_val) {
    ulong carry = 0;
    uint tmp[9];
    for (int i = 0; i < 8; i++) {
        ulong p = ulong(a.limbs[i]) * ulong(small_val) + carry;
        tmp[i] = uint(p);
        carry = p >> 32;
    }
    tmp[8] = uint(carry);

    // Reduce: fold tmp[8] * K into low limbs (branchless -- if extra==0, adds 0)
    uint extra = tmp[8];
    ulong ek = ulong(extra) * ulong(K_LO);
    ulong s = ulong(tmp[0]) + uint(ek);
    tmp[0] = uint(s);
    s = ulong(tmp[1]) + uint(ek >> 32) + ulong(extra) + (s >> 32);
    tmp[1] = uint(s);
    carry = s >> 32;
    for (int i = 2; i < 8; i++) {
        s = ulong(tmp[i]) + carry;
        tmp[i] = uint(s);
        carry = s >> 32;
    }

    // Final branchless conditional subtract of p
    ulong borrow = 0;
    uint diff[8];
    for (int i = 0; i < 8; i++) {
        ulong d = ulong(tmp[i]) - ulong(SECP256K1_P[i]) - borrow;
        diff[i] = uint(d);
        borrow = (d >> 63);
    }

    FieldElement r;
    uint mask = -(uint(borrow == 0));
    uint nmask = ~mask;
    for (int i = 0; i < 8; i++) {
        r.limbs[i] = (diff[i] & mask) | (tmp[i] & nmask);
    }
    return r;
}

// =============================================================================
// Field Inverse: a^(p-2) mod p -- Fermat's Little Theorem
// Exact addition chain from CUDA backend (field_inv_fermat_chain_impl)
// =============================================================================

inline FieldElement field_inv(thread const FieldElement &a) {
    if (field_is_zero(a)) return field_zero();

    FieldElement x_0, x_1, x_2, x_3, x_4, x_5, t;

    // x2 = a^3 (2 ones) -> x_0
    x_0 = field_sqr(a);
    x_0 = field_mul(x_0, a);

    // x3 = a^7 (3 ones) -> x_1
    x_1 = field_sqr(x_0);
    x_1 = field_mul(x_1, a);

    // x6 = a^63 (6 ones) -> x_2
    x_2 = field_sqr(x_1);
    x_2 = field_sqr(x_2);
    x_2 = field_sqr(x_2);
    x_2 = field_mul(x_2, x_1);

    // x9 (9 ones) -> x_3
    x_3 = field_sqr(x_2);
    x_3 = field_sqr(x_3);
    x_3 = field_sqr(x_3);
    x_3 = field_mul(x_3, x_1);

    // x11 (11 ones) -> x_4
    x_4 = field_sqr(x_3);
    x_4 = field_sqr(x_4);
    x_4 = field_mul(x_4, x_0);

    // x22 (22 ones) -> x_3 (reuse)
    t = field_sqr_n(x_4, 11);
    x_3 = field_mul(t, x_4);

    // x44 (44 ones) -> x_4 (reuse)
    t = field_sqr_n(x_3, 22);
    x_4 = field_mul(t, x_3);

    // x88 (88 ones) -> x_5
    t = field_sqr_n(x_4, 44);
    x_5 = field_mul(t, x_4);

    // x176 (176 ones) -> x_5 (reuse)
    t = field_sqr_n(x_5, 88);
    x_5 = field_mul(t, x_5);

    // x220 (220 ones) -> x_5
    x_5 = field_sqr_n(x_5, 44);
    x_5 = field_mul(x_5, x_4);

    // x223 (223 ones) -> x_5
    x_5 = field_sqr(x_5);
    x_5 = field_sqr(x_5);
    x_5 = field_sqr(x_5);
    x_5 = field_mul(x_5, x_1);

    // Assemble p-2 tail
    t = field_sqr(x_5);                          // shift 1 (0 bit)
    t = field_sqr_n(t, 22);
    t = field_mul(t, x_3);                        // append 22 ones

    // 4 zero bits
    t = field_sqr(t); t = field_sqr(t); t = field_sqr(t); t = field_sqr(t);

    // Append binary 101101 (6 bits = p-2 tail)
    t = field_sqr(t); t = field_mul(t, a);        // 1
    t = field_sqr(t);                              // 0
    t = field_sqr(t); t = field_mul(t, a);        // 1
    t = field_sqr(t); t = field_mul(t, a);        // 1
    t = field_sqr(t);                              // 0
    t = field_sqr(t); t = field_mul(t, a);        // 1

    return t;
}

// =============================================================================
// Variable-time modular inverse (Bernstein-Yang SafeGCD) -- PUBLIC DATA ONLY
// =============================================================================
//   field_inv_var(a)           a^-1 mod p   (defined below)
//   scalar_inv_var(a)          a^-1 mod n   (secp256k1_extended.h, next to
//                                            scalar_inverse)
//   jacobian_to_affine_var(p)  affine form of a PUBLIC point (extended.h)
//   modinv30_var_impl(...)     the core all of them call
//
// !! VARIABLE TIME -- PUBLIC DATA ONLY !!
// The number of loop iterations depends on the VALUE being inverted. Never
// call these on a private key, a nonce, a signing share, an ECDH scalar, a
// BIP-32 / BIP-352 secret, or on the Z coordinate of a point produced by a
// secret scalar multiplication (k*G, sk*G, sk*P). Signing, ECDH, key
// generation, provers, every secp256k1_ct_*.h file and the shared affine
// helpers keep field_inv / scalar_inverse / the ct_* inverses.
// MSL has no deleted declarations usable as a fence, so the rule is enforced
// by a source scan in audit/test_regression_opencl_kernel_closure.cpp: these
// names may not appear in any secp256k1_ct_*.h file, on a preprocessor line,
// nor inside any function whose name contains sign, ecdh, keygen, derive or
// prove. A new caller needs its own argument that no secret, or value derived
// from one, reaches the inversion.
//
// Contract (same value as field_inv / scalar_inverse, bit for bit):
//   - input: any 256-bit value, reduced or not; input and output may alias
//   - output: canonical, in [0, modulus); 0 when the input is 0 mod modulus
//   - always terminates: at most MODINV30_MAX_ITERS outer iterations of at
//     most 30 inner passes each; if the cap were reached the result is 0
//     (fail closed) -- not reachable for a 256-bit input, see the cap below
//
// Algorithm: Bernstein-Yang safegcd ("Fast constant-time gcd computation and
// modular inversion", 2019), variable-time form, 30 divsteps per batch on
// 9 signed 30-bit limbs with 64-bit accumulators. Port of the OpenCL code
// (modinv30_var_impl in src/opencl/kernels/secp256k1_field.cl) and the CUDA
// code (src/cuda/include/modinv_var.cuh), themselves ports of this library's
// CPU safegcd30 code, which is adapted from bitcoin-core/secp256k1
// secp256k1_modinv32_var (src/modinv32_impl.h):
//   Copyright (c) 2020 Peter Dettman
//   Distributed under the MIT software license, see
//   https://www.opensource.org/licenses/mit-license.php.
// As in the CUDA / OpenCL code: hard cap on the outer loop, table-free 6-bit
// inner step w = f*g*(f*f-2), f and g always keep 9 limbs, the input is
// reduced first. Metal-specific: the element is 8 x 32-bit limbs, converted
// to and from the 9 x 30-bit form here; and the shaders use no signed 64-bit
// type and no ctz/clz builtin, so the 64-bit accumulators are two's-complement
// values held in a ulong (wrapping add/multiply are exact; the arithmetic
// right shift is spelled out in modinv30_asr30) and the trailing-zero count
// is a five-step branchless binary search.
// =============================================================================

// 25 * 30 = 750 divsteps. 741 divsteps are proven sufficient for any 256-bit
// input with delta starting at 1 (Bernstein-Yang, Theorem 11.2); 724 by
// sipa/safegcd-bounds. bitcoin-core/secp256k1 modinv32_var uses the same 25.
constant int MODINV30_MAX_ITERS = 25;

// int -> the same signed value as a 64-bit two's-complement word.
inline ulong modinv30_sext(int a) {
    return ulong(uint(a)) | ((a < 0) ? (ulong(0xFFFFFFFFu) << 32) : ulong(0));
}

// acc + a*b, signed 32x32 -> 64, on two's-complement words held in a ulong.
inline ulong modinv30_mul_add(ulong acc, int a, int b) {
    return acc + modinv30_sext(a) * modinv30_sext(b);
}

// Arithmetic right shift by 30 of a two's-complement 64-bit word.
inline ulong modinv30_asr30(ulong x) {
    return (x >> 30) | (((x >> 63) != 0) ? (ulong(0xFFFFFFFFu) << 34) : ulong(0));
}

// Arithmetic right shift by 30 of a 32-bit int (floor division by 2^30).
inline int modinv30_asr30_i(int v) {
    return (v < 0) ? ~int(uint(~v) >> 30) : int(uint(v) >> 30);
}

// Number of trailing zero bits of x. x != 0 (the caller sets a sentinel bit).
inline int modinv30_ctz(uint x) {
    int n = 0;
    uint s;
    s = ((x & 0xFFFFu) == 0u) ? 16u : 0u; n += int(s); x >>= s;
    s = ((x & 0xFFu) == 0u)   ? 8u  : 0u; n += int(s); x >>= s;
    s = ((x & 0xFu) == 0u)    ? 4u  : 0u; n += int(s); x >>= s;
    s = ((x & 0x3u) == 0u)    ? 2u  : 0u; n += int(s); x >>= s;
    s = ((x & 0x1u) == 0u)    ? 1u  : 0u; n += int(s);
    return n;
}

// 30 divsteps on the low words of f and g. Returns the new eta (= -delta) and
// writes the transition matrix (u, v, q, r), scaled by 2^30.
// Bounded for ANY f, g: every pass after the first removes at least one of
// the 30 steps (f is forced odd by the caller, and g is odd after the shift).
inline int modinv30_divsteps_30_var(int eta, uint f, uint g,
                                    thread int &tu, thread int &tv,
                                    thread int &tq, thread int &tr) {
    uint u = 1u, v = 0u, q = 0u, r = 1u;
    int i = 30;
    while (true) {
        // strip zero bits of g; the sentinel at bit i stops the count at i
        const int zeros = modinv30_ctz(g | (0xFFFFFFFFu << i));
        g >>= zeros; u <<= zeros; v <<= zeros; eta -= zeros; i -= zeros;
        if (i == 0) break;
        // eta < 0: (f, g) <- (g, -f), same for the matrix rows (selects)
        const bool swap = eta < 0;
        const uint f0 = f, u0 = u, v0 = v;
        eta = swap ? -eta : eta;
        f = swap ? g : f0;   g = swap ? (0u - f0) : g;
        u = swap ? q : u0;   q = swap ? (0u - u0) : q;
        v = swap ? r : v0;   r = swap ? (0u - v0) : r;
        // cancel up to 6 low bits of g with a multiple of f (f odd):
        // g + f*w == g*(f*f-1)^2 == 0 (mod 64)
        const int limit = (eta + 1) < i ? (eta + 1) : i;
        const uint m = (0xFFFFFFFFu >> (32 - limit)) & 63u;
        const uint w = ((f * g) * (f * f - 2u)) & m;
        g += f * w; q += u * w; r += v * w;
    }
    tu = int(u); tv = int(v); tq = int(q); tr = int(r);
    return eta;
}

// r8 = (a8 mod M)^-1 mod M, canonical, where M = n when is_n, else p; both
// arguments are 8 x 32-bit little-endian limbs. 0 if a8 == 0 mod M or if
// max_iters is reached. Returns the number of outer iterations used. a8 == r8
// is allowed. max_iters is a parameter only so that a test kernel can
// exercise the fail-closed path; production callers pass MODINV30_MAX_ITERS.
// VARIABLE TIME, PUBLIC DATA ONLY (see the block comment above).
inline int modinv30_var_impl(thread const uint *a8, thread uint *r8,
                             const bool is_n, const int max_iters) {
    const uint M30 = 0x3FFFFFFFu;

    // Modulus as 9 signed 30-bit limbs (value = sum ml[i] * 2^(30*i)), its
    // inverse mod 2^30, and the same modulus as 8 x 32-bit limbs.
    const int ml[9] = {
        is_n ? 0x10364141 : -0x3D1,  is_n ? 0x3F497A33 : -4,
        is_n ? 0x348A03BB : 0,       is_n ? 0x2BB739AB : 0,
        is_n ? -0x146 : 0,           0, 0, 0, 65536
    };
    const uint inv30 = is_n ? 0x2A774EC1u : 0x2DDACACFu;
    const uint m8[8] = {
        is_n ? 0xD0364141u : 0xFFFFFC2Fu, is_n ? 0xBFD25E8Cu : 0xFFFFFFFEu,
        is_n ? 0xAF48A03Bu : 0xFFFFFFFFu, is_n ? 0xBAAEDCE6u : 0xFFFFFFFFu,
        is_n ? 0xFFFFFFFEu : 0xFFFFFFFFu, 0xFFFFFFFFu,
        0xFFFFFFFFu,                      0xFFFFFFFFu
    };

    // x = a >= M ? a - M : a   (M < 2^256 < 2M, so one subtraction is enough)
    uint xs[8], ts[8];
    {
        ulong borrow = 0;
        for (int i = 0; i < 8; i++) {
            const ulong dd = ulong(a8[i]) - ulong(m8[i]) - borrow;
            ts[i] = uint(dd);
            borrow = dd >> 63;
        }
        const uint keep = 0u - uint(borrow);        // all-ones when a < M
        for (int i = 0; i < 8; i++) xs[i] = (a8[i] & keep) | (ts[i] & ~keep);
    }
    const ulong x0 = ulong(xs[0]) | (ulong(xs[1]) << 32);
    const ulong x1 = ulong(xs[2]) | (ulong(xs[3]) << 32);
    const ulong x2 = ulong(xs[4]) | (ulong(xs[5]) << 32);
    const ulong x3 = ulong(xs[6]) | (ulong(xs[7]) << 32);

    // d = 0, e = 1, f = M, g = x
    int d[9], e[9], f[9], g[9];
    for (int i = 0; i < 9; i++) { d[i] = 0; e[i] = 0; f[i] = ml[i]; }
    e[0] = 1;
    g[0] = int(uint(x0) & M30);
    g[1] = int(uint(x0 >> 30) & M30);
    g[2] = int(uint((x0 >> 60) | (x1 << 4)) & M30);
    g[3] = int(uint(x1 >> 26) & M30);
    g[4] = int(uint((x1 >> 56) | (x2 << 8)) & M30);
    g[5] = int(uint(x2 >> 22) & M30);
    g[6] = int(uint((x2 >> 52) | (x3 << 12)) & M30);
    g[7] = int(uint(x3 >> 18) & M30);
    g[8] = int(uint(x3 >> 48));

    int eta = -1;
    bool done = false;
    int it = 0;
    while (it < max_iters) {
        ++it;
        int u, v, q, r;
        // "| 1u": f is always odd, the OR only makes the inner loop bound
        // independent of that invariant
        eta = modinv30_divsteps_30_var(eta, uint(f[0]) | 1u, uint(g[0]), u, v, q, r);

        // (d, e) <- (matrix / 2^30) * (d, e) mod M
        {
            const int sd = (d[8] < 0) ? -1 : 0, se = (e[8] < 0) ? -1 : 0;
            int md = (u & sd) + (v & se);
            int me = (q & sd) + (r & se);
            ulong cd = modinv30_mul_add(modinv30_mul_add(ulong(0), u, d[0]), v, e[0]);
            ulong ce = modinv30_mul_add(modinv30_mul_add(ulong(0), q, d[0]), r, e[0]);
            md -= int((inv30 * uint(cd) + uint(md)) & M30);
            me -= int((inv30 * uint(ce) + uint(me)) & M30);
            cd = modinv30_mul_add(cd, ml[0], md);
            ce = modinv30_mul_add(ce, ml[0], me);
            cd = modinv30_asr30(cd); ce = modinv30_asr30(ce);   // low 30 bits are zero
            for (int i = 1; i < 9; i++) {
                cd = modinv30_mul_add(modinv30_mul_add(cd, u, d[i]), v, e[i]);
                ce = modinv30_mul_add(modinv30_mul_add(ce, q, d[i]), r, e[i]);
                cd = modinv30_mul_add(cd, ml[i], md);            // ml[i] == 0: no-op
                ce = modinv30_mul_add(ce, ml[i], me);
                d[i - 1] = int(uint(cd) & M30); cd = modinv30_asr30(cd);
                e[i - 1] = int(uint(ce) & M30); ce = modinv30_asr30(ce);
            }
            d[8] = int(uint(cd)); e[8] = int(uint(ce));
        }

        // (f, g) <- (matrix / 2^30) * (f, g), exact; g_nonzero = OR of the new g
        int g_nonzero;
        {
            ulong cf = modinv30_mul_add(modinv30_mul_add(ulong(0), u, f[0]), v, g[0]);
            ulong cg = modinv30_mul_add(modinv30_mul_add(ulong(0), q, f[0]), r, g[0]);
            cf = modinv30_asr30(cf); cg = modinv30_asr30(cg);   // low 30 bits are zero
            g_nonzero = 0;
            for (int i = 1; i < 9; i++) {
                cf = modinv30_mul_add(modinv30_mul_add(cf, u, f[i]), v, g[i]);
                cg = modinv30_mul_add(modinv30_mul_add(cg, q, f[i]), r, g[i]);
                f[i - 1] = int(uint(cf) & M30); cf = modinv30_asr30(cf);
                g[i - 1] = int(uint(cg) & M30); cg = modinv30_asr30(cg);
                g_nonzero |= g[i - 1];
            }
            f[8] = int(uint(cf)); g[8] = int(uint(cg));
            g_nonzero |= g[8];
        }
        if (g_nonzero == 0) { done = true; break; }
    }

    // normalize d to [0, M), negating first when f == -1
    {
        int cond_add = (d[8] < 0) ? -1 : 0;
        for (int i = 0; i < 9; i++) d[i] += ml[i] & cond_add;
        const int cond_neg = (f[8] < 0) ? -1 : 0;
        for (int i = 0; i < 9; i++) d[i] = (d[i] ^ cond_neg) - cond_neg;
        for (int i = 0; i < 8; i++) { d[i + 1] += modinv30_asr30_i(d[i]); d[i] &= int(M30); }
        cond_add = (d[8] < 0) ? -1 : 0;
        for (int i = 0; i < 9; i++) d[i] += ml[i] & cond_add;
        for (int i = 0; i < 8; i++) { d[i + 1] += modinv30_asr30_i(d[i]); d[i] &= int(M30); }
    }

    const ulong ok = ulong(0) - ulong(done ? 1u : 0u);   // cap reached -> 0 (fail closed)
    const ulong d0 = ulong(uint(d[0])), d1 = ulong(uint(d[1])), d2 = ulong(uint(d[2])),
                d3 = ulong(uint(d[3])), d4 = ulong(uint(d[4])), d5 = ulong(uint(d[5])),
                d6 = ulong(uint(d[6])), d7 = ulong(uint(d[7])), d8 = ulong(uint(d[8]));
    const ulong r0 = ok & (d0 | (d1 << 30) | (d2 << 60));
    const ulong r1 = ok & ((d2 >> 4) | (d3 << 26) | (d4 << 56));
    const ulong r2 = ok & ((d4 >> 8) | (d5 << 22) | (d6 << 52));
    const ulong r3 = ok & ((d6 >> 12) | (d7 << 18) | (d8 << 48));
    r8[0] = uint(r0); r8[1] = uint(r0 >> 32);
    r8[2] = uint(r1); r8[3] = uint(r1 >> 32);
    r8[4] = uint(r2); r8[5] = uint(r2 >> 32);
    r8[6] = uint(r3); r8[7] = uint(r3 >> 32);
    return it;
}

// a^-1 mod p. VARIABLE TIME, PUBLIC DATA ONLY (see the block comment above).
inline FieldElement field_inv_var(thread const FieldElement &a) {
    FieldElement r;
    modinv30_var_impl(a.limbs, r.limbs, false, MODINV30_MAX_ITERS);
    return r;
}
