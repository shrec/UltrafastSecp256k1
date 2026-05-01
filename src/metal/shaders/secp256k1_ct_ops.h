// =============================================================================
// secp256k1_ct_ops.h -- Constant-time primitive operations for Metal
// =============================================================================
// Provides value_barrier, CT masks, conditional moves/swaps/selects.
// All operations are branchless: no secret-dependent branches or memory access.
// Metal uses 8x32 limbs -- masks are per-limb uint (0 or 0xFFFFFFFF).
// =============================================================================

#ifndef SECP256K1_CT_OPS_H
#define SECP256K1_CT_OPS_H

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// value_barrier: prevent compiler from reasoning about value.
// Metal has no inline asm; use volatile through threadgroup-invisible path.
// ---------------------------------------------------------------------------
inline uint ct_value_barrier(uint v) {
    volatile uint tmp = v;
    return tmp;
}

// ---------------------------------------------------------------------------
// Mask generators -- all produce 0 or 0xFFFFFFFF (uint)
// ---------------------------------------------------------------------------
inline uint ct_is_zero_mask(uint v) {
    v = ct_value_barrier(v);
    uint t = (v | (0u - v)) >> 31;  // 1 if nonzero
    return t - 1u;  // 0xFFFFFFFF if zero, 0 if nonzero
}

inline uint ct_is_nonzero_mask(uint v) {
    v = ct_value_barrier(v);
    uint t = (v | (0u - v)) >> 31;  // 1 if nonzero
    return 0u - t;  // 0xFFFFFFFF if nonzero, 0 if zero
}

// Check if all 8 limbs are zero (for FieldElement / Scalar256)
inline uint ct_is_zero_mask_8limb(thread const uint limbs[8]) {
    uint acc = 0;
    for (int i = 0; i < 8; ++i) acc |= limbs[i];
    return ct_is_zero_mask(acc);
}

inline uint ct_eq_mask(uint a, uint b) {
    return ct_is_zero_mask(a ^ b);
}

inline uint ct_bool_to_mask(bool b) {
    return 0u - (uint)(b);
}

// Return 0xFFFFFFFF if a < b (unsigned), else 0
inline uint ct_lt_mask(uint a, uint b) {
    uint x = a ^ b;
    uint d = a - b;
    uint borrow = (a ^ (x | (d ^ a))) >> 31;
    return 0u - borrow;
}

// ---------------------------------------------------------------------------
// Conditional move (8-limb): r = mask ? a : r
// ---------------------------------------------------------------------------
inline void ct_cmov_8limb(thread uint r[8], thread const uint a[8], uint mask) {
    for (int i = 0; i < 8; ++i)
        r[i] = (r[i] & ~mask) | (a[i] & mask);
}

// ---------------------------------------------------------------------------
// Conditional swap (8-limb): if mask, swap a[] and b[]
// ---------------------------------------------------------------------------
inline void ct_cswap_8limb(thread uint a[8], thread uint b[8], uint mask) {
    for (int i = 0; i < 8; ++i) {
        uint diff = (a[i] ^ b[i]) & mask;
        a[i] ^= diff;
        b[i] ^= diff;
    }
}

// ---------------------------------------------------------------------------
// CT select from table: always scans ALL entries
// ---------------------------------------------------------------------------
inline void ct_select_8limb(thread const uint table[][8], int table_size,
                            int index, thread uint out[8]) {
    for (int j = 0; j < 8; ++j) out[j] = 0;
    for (int i = 0; i < table_size; ++i) {
        uint mask = ct_eq_mask((uint)i, (uint)index);
        for (int j = 0; j < 8; ++j)
            out[j] |= table[i][j] & mask;
    }
}

// CT lookup with 16 limbs (AffinePoint = 2 * 8 limbs = 16 uints)
inline void ct_lookup_16limb(thread const uint table[][16], int table_size,
                             int index, thread uint out[16]) {
    for (int j = 0; j < 16; ++j) out[j] = 0;
    for (int i = 0; i < table_size; ++i) {
        uint mask = ct_eq_mask((uint)i, (uint)index);
        for (int j = 0; j < 16; ++j)
            out[j] |= table[i][j] & mask;
    }
}

// ---------------------------------------------------------------------------
// CT FieldElement operations (8x32 limbs)
// ---------------------------------------------------------------------------
inline void ct_field_cmov(thread FieldElement &r, thread const FieldElement &a, uint mask) {
    ct_cmov_8limb(r.limbs, a.limbs, mask);
}

inline void ct_field_cswap(thread FieldElement &a, thread FieldElement &b, uint mask) {
    ct_cswap_8limb(a.limbs, b.limbs, mask);
}

inline FieldElement ct_field_select(thread const FieldElement &a,
                                    thread const FieldElement &b, uint mask) {
    // mask=all-ones -> a, mask=0 -> b
    FieldElement r;
    for (int i = 0; i < 8; ++i)
        r.limbs[i] = (b.limbs[i] & ~mask) | (a.limbs[i] & mask);
    return r;
}

inline FieldElement ct_field_cneg(thread const FieldElement &a, uint mask) {
    FieldElement neg = field_negate(a);
    return ct_field_select(neg, a, mask);
}

// ---------------------------------------------------------------------------
// CT Scalar256 operations (8x32 limbs)
// ---------------------------------------------------------------------------
inline void ct_scalar_cmov(thread Scalar256 &r, thread const Scalar256 &a, uint mask) {
    ct_cmov_8limb(r.limbs, a.limbs, mask);
}

inline void ct_scalar_cswap(thread Scalar256 &a, thread Scalar256 &b, uint mask) {
    ct_cswap_8limb(a.limbs, b.limbs, mask);
}

inline Scalar256 ct_scalar_select(thread const Scalar256 &a,
                                  thread const Scalar256 &b, uint mask) {
    Scalar256 r;
    for (int i = 0; i < 8; ++i)
        r.limbs[i] = (b.limbs[i] & ~mask) | (a.limbs[i] & mask);
    return r;
}

inline Scalar256 ct_scalar_cneg(thread const Scalar256 &a, uint mask) {
    Scalar256 neg = scalar_negate(a);
    return ct_scalar_select(neg, a, mask);
}

#endif // SECP256K1_CT_OPS_H
