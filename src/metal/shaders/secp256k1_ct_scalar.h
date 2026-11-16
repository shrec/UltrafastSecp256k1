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
// CT GLV decomposition (branchless)
// ---------------------------------------------------------------------------
// Line-for-line mirror of ct_glv_decompose_impl in
// src/opencl/kernels/secp256k1_ct_scalar.cl, in 8x32-bit limbs.
// k == (+/-k1) + (+/-k2)*lambda (mod n) with k1, k2 < 2^128: Babai rounding on
// the secp256k1 lattice with the same g1, g2, -b1, -b2 and lambda as the
// variable-time GLV decomposition. Fixed multiplies, masked mod-n add/sub/neg
// and mask selects only; no branch, loop bound or memory index depends on k.
// ct_generator_mul_metal and ct_scalar_mul_point scan 33 four-bit windows
// (132 bits) of k1 and k2, which is correct only because of the 2^128 bound --
// a k1 = k, k2 = 0 split silently drops the top bits of k.
struct CTGLVDecompositionMetal {
    Scalar256 k1;
    Scalar256 k2;
    uint k1_neg;  // mask
    uint k2_neg;  // mask
};

// GLV lattice constants in 8x32 LE (same values as GLV_G1, GLV_G2,
// GLV_MINUS_B1, GLV_MINUS_B2 and GLV_LAMBDA0..3 in secp256k1_extended.cl)
constant uint CT_GLV_G1_METAL[8] = {
    0x45DBB031u, 0xE893209Au, 0x71E8CA7Fu, 0x3DAA8A14u,
    0x9284EB15u, 0xE86C90E4u, 0xA7D46BCDu, 0x3086D221u
};
constant uint CT_GLV_G2_METAL[8] = {
    0x8AC47F71u, 0x1571B4AEu, 0x9DF506C6u, 0x221208ACu,
    0x0ABFE4C4u, 0x6F547FA9u, 0x010E8828u, 0xE4437ED6u
};
constant uint CT_GLV_MINUS_B1_METAL[8] = {
    0x0ABFE4C3u, 0x6F547FA9u, 0x010E8828u, 0xE4437ED6u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u
};
constant uint CT_GLV_MINUS_B2_METAL[8] = {
    0x3DB1562Cu, 0xD765CDA8u, 0x0774346Du, 0x8A280AC5u,
    0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
};
constant uint CT_GLV_LAMBDA_METAL[8] = {
    0x1B23BD72u, 0xDF02967Cu, 0x20816678u, 0x122E22EAu,
    0x8812645Au, 0xA5261C02u, 0xC05C30E0u, 0x5363AD4Cu
};

// CT (a * b) >> 384, rounded at bit 383: a fixed 8x8 schoolbook product, and
// the rounding bit is added as a carry instead of being branched on.
// For every 256-bit a the result is < 2^129, so limbs 4..7 are always 0.
inline Scalar256 ct_mul_shift_384(thread const Scalar256 &a, constant const uint *b) {
    uint prod[16];
    for (int i = 0; i < 16; ++i) prod[i] = 0u;
    for (int i = 0; i < 8; ++i) {
        ulong carry = 0;
        for (int j = 0; j < 8; ++j) {
            ulong t = (ulong)a.limbs[i] * (ulong)b[j] + (ulong)prod[i + j] + carry;
            prod[i + j] = (uint)t;
            carry = t >> 32;
        }
        prod[i + 8] = (uint)carry;
    }
    Scalar256 r;
    ulong c = (ulong)(prod[11] >> 31);
    for (int i = 0; i < 4; ++i) {
        ulong t = (ulong)prod[12 + i] + c;
        r.limbs[i] = (uint)t;
        c = t >> 32;
    }
    for (int i = 4; i < 8; ++i) r.limbs[i] = 0u;
    return r;
}

inline CTGLVDecompositionMetal ct_glv_decompose(thread const Scalar256 &k) {
    // c1 = round(k * g1 / 2^384), c2 = round(k * g2 / 2^384); both < 2^129 < n
    Scalar256 c1 = ct_mul_shift_384(k, CT_GLV_G1_METAL);
    Scalar256 c2 = ct_mul_shift_384(k, CT_GLV_G2_METAL);

    // k2 = c1*(-b1) + c2*(-b2) (mod n)
    Scalar256 minus_b1, minus_b2;
    for (int i = 0; i < 8; ++i) {
        minus_b1.limbs[i] = CT_GLV_MINUS_B1_METAL[i];
        minus_b2.limbs[i] = CT_GLV_MINUS_B2_METAL[i];
    }
    Scalar256 t1 = ct_scalar_mul(c1, minus_b1);
    Scalar256 t2 = ct_scalar_mul(c2, minus_b2);
    Scalar256 k2_mod = ct_scalar_add(t1, t2);

    // k1 = k - lambda*k2 (mod n)
    Scalar256 lambda_s;
    for (int i = 0; i < 8; ++i) lambda_s.limbs[i] = CT_GLV_LAMBDA_METAL[i];
    Scalar256 lk2 = ct_scalar_mul(lambda_s, k2_mod);
    Scalar256 k1_mod = ct_scalar_sub(k, lk2);

    // Shorter form: a residue above n/2 is the negation of a value < 2^128.
    // Mask select, both negations always computed.
    CTGLVDecompositionMetal out;
    out.k1_neg = ct_scalar_is_high_mask(k1_mod);
    out.k2_neg = ct_scalar_is_high_mask(k2_mod);
    Scalar256 neg;
    out.k1 = k1_mod;
    neg = ct_scalar_neg(k1_mod);
    ct_scalar_cmov(out.k1, neg, out.k1_neg);
    out.k2 = k2_mod;
    neg = ct_scalar_neg(k2_mod);
    ct_scalar_cmov(out.k2, neg, out.k2_neg);
    return out;
}

#endif // SECP256K1_CT_SCALAR_H
