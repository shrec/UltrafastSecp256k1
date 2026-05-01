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
    // acc[8] can exceed 32 bits after first fold (issue #226), so use
    // full ulong and loop until fully absorbed (at most 2 iterations).
    while (acc[8] != 0) {
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
