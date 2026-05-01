// =============================================================================
// secp256k1_ct_ops.cl -- Constant-time primitive operations for OpenCL
// =============================================================================
// Provides value_barrier, CT masks, conditional moves/swaps/selects.
// All operations are branchless: no secret-dependent branches or memory access.
// =============================================================================

#ifndef SECP256K1_CT_OPS_CL
#define SECP256K1_CT_OPS_CL

// ---------------------------------------------------------------------------
// value_barrier: prevent compiler from reasoning about value, breaking
// branch-to-cmov and dead-code optimizations that leak secrets.
// OpenCL has no inline asm; use volatile through private memory.
// ---------------------------------------------------------------------------
inline ulong ct_value_barrier(ulong v) {
    volatile ulong tmp = v;
    return tmp;
}

inline uint ct_value_barrier32(uint v) {
    volatile uint tmp = v;
    return tmp;
}

// ---------------------------------------------------------------------------
// Mask generators -- all produce 0 or 0xFFFFFFFFFFFFFFFF
// ---------------------------------------------------------------------------
inline ulong ct_is_zero_mask(ulong v) {
    // (v | -v) >> 63 is 1 when v != 0, 0 when v == 0
    v = ct_value_barrier(v);
    return ~(((v | (0UL - v)) >> 63) - 1UL) ^ 0xFFFFFFFFFFFFFFFFUL;
    // Simplified: zero -> mask = all-ones, nonzero -> mask = 0
}

inline ulong ct_is_nonzero_mask(ulong v) {
    v = ct_value_barrier(v);
    ulong t = (v | (0UL - v)) >> 63;  // 1 if nonzero
    return 0UL - t;  // 0xFFFF... if nonzero, 0 if zero
}

inline ulong ct_eq_mask(ulong a, ulong b) {
    return ct_is_zero_mask(a ^ b);
}

inline ulong ct_bool_to_mask(int b) {
    return 0UL - (ulong)(b & 1);
}

// Return all-ones if a < b (unsigned), else 0
inline ulong ct_lt_mask(ulong a, ulong b) {
    // (a - b) borrows iff a < b, bit 63 of (a ^ ((a ^ b) | ((a - b) ^ a)))
    ulong x = a ^ b;
    ulong d = a - b;
    ulong borrow = (a ^ ((x) | (d ^ a))) >> 63;
    return 0UL - borrow;
}

// ---------------------------------------------------------------------------
// Conditional move: r = cond ? a : r  (cond is 0 or all-ones mask)
// ---------------------------------------------------------------------------
inline void ct_cmov64(ulong* r, ulong a, ulong mask) {
    *r = (*r & ~mask) | (a & mask);
}

inline void ct_cmov256(ulong r[4], const ulong a[4], ulong mask) {
    for (int i = 0; i < 4; ++i)
        r[i] = (r[i] & ~mask) | (a[i] & mask);
}

// ---------------------------------------------------------------------------
// Conditional swap: if mask is all-ones, swap a[] and b[]
// ---------------------------------------------------------------------------
inline void ct_cswap256(ulong a[4], ulong b[4], ulong mask) {
    for (int i = 0; i < 4; ++i) {
        ulong diff = (a[i] ^ b[i]) & mask;
        a[i] ^= diff;
        b[i] ^= diff;
    }
}

// ---------------------------------------------------------------------------
// CT select from array: always scans ALL entries (no early exit)
// ---------------------------------------------------------------------------
inline void ct_select256(const ulong table[][4], int table_size, int index, ulong out[4]) {
    out[0] = 0; out[1] = 0; out[2] = 0; out[3] = 0;
    for (int i = 0; i < table_size; ++i) {
        ulong mask = ct_eq_mask((ulong)i, (ulong)index);
        for (int j = 0; j < 4; ++j)
            out[j] |= table[i][j] & mask;
    }
}

// CT lookup: 8-limb version (for FieldElement + Scalar pairs stored as 8 ulongs)
inline void ct_lookup_256(const ulong table[][8], int table_size, int index, ulong out[8]) {
    for (int j = 0; j < 8; ++j) out[j] = 0;
    for (int i = 0; i < table_size; ++i) {
        ulong mask = ct_eq_mask((ulong)i, (ulong)index);
        for (int j = 0; j < 8; ++j)
            out[j] |= table[i][j] & mask;
    }
}

// ---------------------------------------------------------------------------
// CT FieldElement operations (wrapping the 4-limb FieldElement struct)
// ---------------------------------------------------------------------------
inline void ct_field_cmov(FieldElement* r, const FieldElement* a, ulong mask) {
    ct_cmov256((ulong*)r->limbs, (const ulong*)a->limbs, mask);
}

inline void ct_field_cswap(FieldElement* a, FieldElement* b, ulong mask) {
    ct_cswap256((ulong*)a->limbs, (ulong*)b->limbs, mask);
}

inline void ct_field_select(const FieldElement* a, const FieldElement* b,
                            ulong mask, FieldElement* out) {
    // out = mask ? a : b
    for (int i = 0; i < 4; ++i)
        out->limbs[i] = (b->limbs[i] & ~mask) | (a->limbs[i] & mask);
}

inline void ct_field_cneg(FieldElement* r, const FieldElement* a, ulong mask) {
    FieldElement neg;
    field_neg_impl(&neg, a);
    ct_field_select(&neg, a, mask, r);
}

// ---------------------------------------------------------------------------
// CT Scalar operations (wrapping the 4-limb Scalar struct)
// ---------------------------------------------------------------------------
inline void ct_scalar_cmov(Scalar* r, const Scalar* a, ulong mask) {
    ct_cmov256((ulong*)r->limbs, (const ulong*)a->limbs, mask);
}

inline void ct_scalar_cswap(Scalar* a, Scalar* b, ulong mask) {
    ct_cswap256((ulong*)a->limbs, (ulong*)b->limbs, mask);
}

inline void ct_scalar_select(const Scalar* a, const Scalar* b,
                             ulong mask, Scalar* out) {
    for (int i = 0; i < 4; ++i)
        out->limbs[i] = (b->limbs[i] & ~mask) | (a->limbs[i] & mask);
}

inline void ct_scalar_cneg(Scalar* r, const Scalar* a, ulong mask) {
    Scalar neg;
    scalar_negate_impl(a, &neg);
    ct_scalar_select(&neg, a, mask, r);
}

#endif // SECP256K1_CT_OPS_CL
