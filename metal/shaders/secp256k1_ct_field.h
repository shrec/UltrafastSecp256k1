// =============================================================================
// secp256k1_ct_field.h -- Constant-time field arithmetic for Metal
// =============================================================================
// 8x32-bit limbs. Branchless add/sub with inline reduction.
// mul/sqr/inv wrap fast-path (data-independent instruction trace).
// Requires: secp256k1_field.h, secp256k1_ct_ops.h
// =============================================================================

#ifndef SECP256K1_CT_FIELD_H
#define SECP256K1_CT_FIELD_H

// secp256k1 field prime p in 8x32 LE
constant uint CT_FIELD_P_LIMBS[8] = {
    0xFFFFFC2Fu, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
    0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
};

// ---------------------------------------------------------------------------
// 256-bit add with carry (8x32) — thread/thread
// ---------------------------------------------------------------------------
inline uint ct_add_8limb(thread const uint a[8], thread const uint b[8], thread uint r[8]) {
    uint carry = 0;
    for (int i = 0; i < 8; ++i) {
        uint sum = a[i] + b[i] + carry;
        carry = (sum < a[i]) || (carry && sum == a[i]) ? 1u : 0u;
        r[i] = sum;
    }
    return carry;
}

// 256-bit add — thread/constant overload (for constant P)
inline uint ct_add_8limb(thread const uint a[8], constant const uint b[8], thread uint r[8]) {
    uint carry = 0;
    for (int i = 0; i < 8; ++i) {
        uint sum = a[i] + b[i] + carry;
        carry = (sum < a[i]) || (carry && sum == a[i]) ? 1u : 0u;
        r[i] = sum;
    }
    return carry;
}

// 256-bit sub with borrow (8x32) — thread/thread
inline uint ct_sub_8limb(thread const uint a[8], thread const uint b[8], thread uint r[8]) {
    uint borrow = 0;
    for (int i = 0; i < 8; ++i) {
        uint diff = a[i] - b[i] - borrow;
        borrow = (a[i] < b[i] + borrow) || (borrow && b[i] == 0xFFFFFFFFu) ? 1u : 0u;
        r[i] = diff;
    }
    return borrow;
}

// 256-bit sub — thread/constant overload (for constant P)
inline uint ct_sub_8limb(thread const uint a[8], constant const uint b[8], thread uint r[8]) {
    uint borrow = 0;
    for (int i = 0; i < 8; ++i) {
        uint diff = a[i] - b[i] - borrow;
        borrow = (a[i] < b[i] + borrow) || (borrow && b[i] == 0xFFFFFFFFu) ? 1u : 0u;
        r[i] = diff;
    }
    return borrow;
}

// 256-bit sub — constant/thread overload (for p - a)
inline uint ct_sub_8limb(constant const uint a[8], thread const uint b[8], thread uint r[8]) {
    uint borrow = 0;
    for (int i = 0; i < 8; ++i) {
        uint diff = a[i] - b[i] - borrow;
        borrow = (a[i] < b[i] + borrow) || (borrow && b[i] == 0xFFFFFFFFu) ? 1u : 0u;
        r[i] = diff;
    }
    return borrow;
}

// ---------------------------------------------------------------------------
// Branchless field reduce: if val >= p, subtract p
// ---------------------------------------------------------------------------
inline void ct_reduce_field(thread FieldElement &r) {
    uint tmp[8];
    uint borrow = ct_sub_8limb(r.limbs, CT_FIELD_P_LIMBS, tmp);
    uint mask = ct_bool_to_mask(borrow == 0);
    for (int i = 0; i < 8; ++i)
        r.limbs[i] = (r.limbs[i] & ~mask) | (tmp[i] & mask);
}

// ---------------------------------------------------------------------------
// CT field_add
// ---------------------------------------------------------------------------
inline FieldElement ct_field_add(thread const FieldElement &a, thread const FieldElement &b) {
    FieldElement r;
    uint carry = ct_add_8limb(a.limbs, b.limbs, r.limbs);
    uint tmp[8];
    uint borrow = ct_sub_8limb(r.limbs, CT_FIELD_P_LIMBS, tmp);
    uint do_sub = ct_is_nonzero_mask(carry) | ct_is_zero_mask(borrow);
    for (int i = 0; i < 8; ++i)
        r.limbs[i] = (r.limbs[i] & ~do_sub) | (tmp[i] & do_sub);
    return r;
}

// ---------------------------------------------------------------------------
// CT field_sub
// ---------------------------------------------------------------------------
inline FieldElement ct_field_sub(thread const FieldElement &a, thread const FieldElement &b) {
    FieldElement r;
    uint borrow = ct_sub_8limb(a.limbs, b.limbs, r.limbs);
    uint tmp[8];
    ct_add_8limb(r.limbs, CT_FIELD_P_LIMBS, tmp);
    uint mask = ct_is_nonzero_mask(borrow);
    for (int i = 0; i < 8; ++i)
        r.limbs[i] = (r.limbs[i] & ~mask) | (tmp[i] & mask);
    return r;
}

// ---------------------------------------------------------------------------
// CT field_neg
// ---------------------------------------------------------------------------
inline FieldElement ct_field_neg(thread const FieldElement &a) {
    uint tmp[8];
    ct_sub_8limb(CT_FIELD_P_LIMBS, a.limbs, tmp);
    uint acc = 0;
    for (int i = 0; i < 8; ++i) acc |= a.limbs[i];
    uint is_zero = ct_is_zero_mask(acc);
    FieldElement r;
    for (int i = 0; i < 8; ++i)
        r.limbs[i] = tmp[i] & ~is_zero;
    return r;
}

// ---------------------------------------------------------------------------
// CT field_mul/sqr/inv: wrap fast-path with value_barrier
// ---------------------------------------------------------------------------
inline FieldElement ct_field_mul(thread const FieldElement &a, thread const FieldElement &b) {
    FieldElement a2 = a, b2 = b;
    for (int i = 0; i < 8; ++i) {
        a2.limbs[i] = ct_value_barrier(a2.limbs[i]);
        b2.limbs[i] = ct_value_barrier(b2.limbs[i]);
    }
    return field_mul(a2, b2);
}

inline FieldElement ct_field_sqr(thread const FieldElement &a) {
    FieldElement a2 = a;
    for (int i = 0; i < 8; ++i)
        a2.limbs[i] = ct_value_barrier(a2.limbs[i]);
    return field_sqr(a2);
}

inline FieldElement ct_field_inv(thread const FieldElement &a) {
    FieldElement a2 = a;
    for (int i = 0; i < 8; ++i)
        a2.limbs[i] = ct_value_barrier(a2.limbs[i]);
    return field_inv(a2);
}

// CT field_half: branchless (a+p)/2 if odd, a/2 if even
inline FieldElement ct_field_half(thread const FieldElement &a) {
    uint odd_mask = ct_bool_to_mask((bool)(a.limbs[0] & 1u));
    uint tmp[8];
    ct_add_8limb(a.limbs, CT_FIELD_P_LIMBS, tmp);
    uint src[8];
    for (int i = 0; i < 8; ++i)
        src[i] = (a.limbs[i] & ~odd_mask) | (tmp[i] & odd_mask);
    FieldElement r;
    for (int i = 0; i < 7; ++i)
        r.limbs[i] = (src[i] >> 1) | (src[i + 1] << 31);
    r.limbs[7] = src[7] >> 1;
    return r;
}

// ---------------------------------------------------------------------------
// CT field predicates
// ---------------------------------------------------------------------------
inline uint ct_field_is_zero_mask(thread const FieldElement &a) {
    uint acc = 0;
    for (int i = 0; i < 8; ++i) acc |= a.limbs[i];
    return ct_is_zero_mask(acc);
}

inline uint ct_field_eq_mask(thread const FieldElement &a, thread const FieldElement &b) {
    uint acc = 0;
    for (int i = 0; i < 8; ++i) acc |= (a.limbs[i] ^ b.limbs[i]);
    return ct_is_zero_mask(acc);
}

inline void ct_field_normalize(thread FieldElement &r) {
    ct_reduce_field(r);
}

#endif // SECP256K1_CT_FIELD_H
