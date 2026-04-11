// =============================================================================
// secp256k1_ct_scalar.h -- Constant-time scalar arithmetic for Metal
// =============================================================================
// 8x32-bit limbs. Branchless scalar ops mod n. Fermat-based inverse.
// Requires: secp256k1_extended.h (scalar_*), secp256k1_ct_ops.h
// =============================================================================

#ifndef SECP256K1_CT_SCALAR_H
#define SECP256K1_CT_SCALAR_H

// secp256k1 order n in 8x32 LE
constant uint CT_ORDER_N[8] = {
    0xD0364141u, 0xBFD25E8Cu, 0xAF48A03Bu, 0xBAAEDCE6u,
    0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
};

// Half order (n-1)/2 in 8x32 LE
constant uint CT_HALF_N[8] = {
    0x681B20A0u, 0xDFE92F46u, 0x57A4501Du, 0x5D576E73u,
    0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x7FFFFFFFu
};

// ---------------------------------------------------------------------------
// Branchless reduce mod n
// ---------------------------------------------------------------------------
inline void ct_reduce_order(thread Scalar256 &r) {
    uint tmp[8];
    uint borrow = ct_sub_8limb(r.limbs, CT_ORDER_N, tmp);
    uint mask = ct_bool_to_mask(borrow == 0);
    for (int i = 0; i < 8; ++i)
        r.limbs[i] = (r.limbs[i] & ~mask) | (tmp[i] & mask);
}

// ---------------------------------------------------------------------------
// CT scalar_add
// ---------------------------------------------------------------------------
inline Scalar256 ct_scalar_add(thread const Scalar256 &a, thread const Scalar256 &b) {
    Scalar256 r;
    uint carry = ct_add_8limb(a.limbs, b.limbs, r.limbs);
    uint tmp[8];
    uint borrow = ct_sub_8limb(r.limbs, CT_ORDER_N, tmp);
    uint do_sub = ct_is_nonzero_mask(carry) | ct_is_zero_mask(borrow);
    for (int i = 0; i < 8; ++i)
        r.limbs[i] = (r.limbs[i] & ~do_sub) | (tmp[i] & do_sub);
    return r;
}

// ---------------------------------------------------------------------------
// CT scalar_sub
// ---------------------------------------------------------------------------
inline Scalar256 ct_scalar_sub(thread const Scalar256 &a, thread const Scalar256 &b) {
    Scalar256 r;
    uint borrow = ct_sub_8limb(a.limbs, b.limbs, r.limbs);
    uint tmp[8];
    ct_add_8limb(r.limbs, CT_ORDER_N, tmp);
    uint mask = ct_is_nonzero_mask(borrow);
    for (int i = 0; i < 8; ++i)
        r.limbs[i] = (r.limbs[i] & ~mask) | (tmp[i] & mask);
    return r;
}

// ---------------------------------------------------------------------------
// CT scalar_neg
// ---------------------------------------------------------------------------
inline Scalar256 ct_scalar_neg(thread const Scalar256 &a) {
    uint tmp[8];
    ct_sub_8limb(CT_ORDER_N, a.limbs, tmp);
    uint acc = 0;
    for (int i = 0; i < 8; ++i) acc |= a.limbs[i];
    uint is_zero = ct_is_zero_mask(acc);
    Scalar256 r;
    for (int i = 0; i < 8; ++i)
        r.limbs[i] = tmp[i] & ~is_zero;
    return r;
}

// ---------------------------------------------------------------------------
// CT scalar_half
// ---------------------------------------------------------------------------
inline Scalar256 ct_scalar_half(thread const Scalar256 &a) {
    uint odd_mask = ct_bool_to_mask((bool)(a.limbs[0] & 1u));
    uint tmp[8];
    ct_add_8limb(a.limbs, CT_ORDER_N, tmp);
    uint src[8];
    for (int i = 0; i < 8; ++i)
        src[i] = (a.limbs[i] & ~odd_mask) | (tmp[i] & odd_mask);
    Scalar256 r;
    for (int i = 0; i < 7; ++i)
        r.limbs[i] = (src[i] >> 1) | (src[i + 1] << 31);
    r.limbs[7] = src[7] >> 1;
    return r;
}

// ---------------------------------------------------------------------------
// CT scalar_mul/sqr: wrap fast-path with value_barrier
// ---------------------------------------------------------------------------
inline Scalar256 ct_scalar_mul(thread const Scalar256 &a, thread const Scalar256 &b) {
    Scalar256 a2 = a, b2 = b;
    for (int i = 0; i < 8; ++i) {
        a2.limbs[i] = ct_value_barrier(a2.limbs[i]);
        b2.limbs[i] = ct_value_barrier(b2.limbs[i]);
    }
    return scalar_mul_mod_n(a2, b2);
}

inline Scalar256 ct_scalar_sqr(thread const Scalar256 &a) {
    return ct_scalar_mul(a, a);
}

// ---------------------------------------------------------------------------
// CT scalar_inverse: Fermat a^(n-2) mod n, fixed-trace 256 squares + cmovs
// ---------------------------------------------------------------------------
inline Scalar256 ct_scalar_inverse(thread const Scalar256 &a) {
    // n-2 in 8x32 LE
    const uint nm2[8] = {
        CT_ORDER_N[0] - 2u, CT_ORDER_N[1], CT_ORDER_N[2], CT_ORDER_N[3],
        CT_ORDER_N[4], CT_ORDER_N[5], CT_ORDER_N[6], CT_ORDER_N[7]
    };

    Scalar256 result;
    for (int i = 0; i < 8; ++i) result.limbs[i] = 0;
    result.limbs[0] = 1;
    Scalar256 base = a;

    for (int bit = 0; bit < 256; ++bit) {
        int limb_idx = bit >> 5;
        int bit_idx = bit & 31;
        uint bit_val = (nm2[limb_idx] >> bit_idx) & 1u;
        uint mask = ct_bool_to_mask(bit_val != 0);

        Scalar256 tmp = ct_scalar_mul(result, base);
        ct_scalar_cmov(result, tmp, mask);
        base = ct_scalar_sqr(base);
    }
    return result;
}

// ---------------------------------------------------------------------------
// CT scalar predicates
// ---------------------------------------------------------------------------
inline uint ct_scalar_is_zero_mask(thread const Scalar256 &a) {
    uint acc = 0;
    for (int i = 0; i < 8; ++i) acc |= a.limbs[i];
    return ct_is_zero_mask(acc);
}

inline uint ct_scalar_eq_mask(thread const Scalar256 &a, thread const Scalar256 &b) {
    uint acc = 0;
    for (int i = 0; i < 8; ++i) acc |= (a.limbs[i] ^ b.limbs[i]);
    return ct_is_zero_mask(acc);
}

// Returns 0xFFFFFFFF if s > n/2
inline uint ct_scalar_is_high_mask(thread const Scalar256 &s) {
    uint tmp[8];
    uint borrow = ct_sub_8limb(CT_HALF_N, s.limbs, tmp);
    return ct_is_nonzero_mask(borrow);
}

inline int ct_scalar_bit(thread const Scalar256 &s, int pos) {
    int limb_idx = pos >> 5;
    int bit_idx = pos & 31;
    return (int)((s.limbs[limb_idx] >> bit_idx) & 1u);
}

inline int ct_scalar_window(thread const Scalar256 &s, int pos, int width) {
    int val = 0;
    for (int i = 0; i < width; ++i)
        val |= ct_scalar_bit(s, pos + i) << i;
    return val;
}

// Low-S normalization
inline Scalar256 ct_scalar_normalize_low_s(thread const Scalar256 &s) {
    uint mask = ct_scalar_is_high_mask(s);
    Scalar256 neg = ct_scalar_neg(s);
    return ct_scalar_select(neg, s, mask);
}

// ---------------------------------------------------------------------------
// CT GLV decomposition
// ---------------------------------------------------------------------------
struct CTGLVDecompositionMetal {
    Scalar256 k1;
    Scalar256 k2;
    uint k1_neg;  // mask
    uint k2_neg;  // mask
};

inline CTGLVDecompositionMetal ct_glv_decompose(thread const Scalar256 &k) {
    CTGLVDecompositionMetal out;
    out.k1 = k;
    for (int i = 0; i < 8; ++i) out.k2.limbs[i] = 0;

    out.k1_neg = ct_scalar_is_high_mask(out.k1);
    Scalar256 neg_k1 = ct_scalar_neg(out.k1);
    ct_scalar_cmov(out.k1, neg_k1, out.k1_neg);
    out.k2_neg = 0;
    return out;
}

#endif // SECP256K1_CT_SCALAR_H
