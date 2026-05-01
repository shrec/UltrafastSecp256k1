// =============================================================================
// secp256k1_ct_field.cl -- Constant-time field arithmetic for OpenCL
// =============================================================================
// Branchless add/sub with inline reduction. mul/sqr/inv wrap fast-path since
// they already have data-independent instruction traces (fixed iteration count).
// Requires: secp256k1_field.cl, secp256k1_ct_ops.cl
// =============================================================================

#ifndef SECP256K1_CT_FIELD_CL
#define SECP256K1_CT_FIELD_CL

// ---------------------------------------------------------------------------
// 256-bit add with carry (a + b -> r, returns carry)
// ---------------------------------------------------------------------------
inline ulong ct_add256(const ulong a[4], const ulong b[4], ulong r[4]) {
    ulong carry = 0;
    for (int i = 0; i < 4; ++i) {
        ulong sum = a[i] + b[i] + carry;
        carry = (sum < a[i]) || (carry && sum == a[i]) ? 1UL : 0UL;
        r[i] = sum;
    }
    return carry;
}

// 256-bit sub with borrow (a - b -> r, returns borrow)
inline ulong ct_sub256(const ulong a[4], const ulong b[4], ulong r[4]) {
    ulong borrow = 0;
    for (int i = 0; i < 4; ++i) {
        ulong diff = a[i] - b[i] - borrow;
        borrow = (a[i] < b[i] + borrow) || (borrow && b[i] == 0xFFFFFFFFFFFFFFFFUL) ? 1UL : 0UL;
        r[i] = diff;
    }
    return borrow;
}

// ---------------------------------------------------------------------------
// Branchless field reduce: if val >= p, subtract p
// p = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
// ---------------------------------------------------------------------------
#define CT_FIELD_P0 0xFFFFFFFEFFFFFC2FUL
#define CT_FIELD_P1 0xFFFFFFFFFFFFFFFFUL
#define CT_FIELD_P2 0xFFFFFFFFFFFFFFFFUL
#define CT_FIELD_P3 0xFFFFFFFFFFFFFFFFUL

inline void ct_reduce_field(FieldElement* r) {
    const ulong p[4] = { CT_FIELD_P0, CT_FIELD_P1, CT_FIELD_P2, CT_FIELD_P3 };
    ulong tmp[4];
    ulong borrow = ct_sub256(r->limbs, p, tmp);
    // If no borrow (val >= p), use tmp; else keep r
    ulong mask = ct_bool_to_mask(borrow == 0);
    for (int i = 0; i < 4; ++i)
        r->limbs[i] = (r->limbs[i] & ~mask) | (tmp[i] & mask);
}

// ---------------------------------------------------------------------------
// CT field_add: r = (a + b) mod p, branchless
// ---------------------------------------------------------------------------
inline void ct_field_add_impl(FieldElement* r, const FieldElement* a, const FieldElement* b) {
    ulong carry = ct_add256(a->limbs, b->limbs, r->limbs);
    // If carry or r >= p, subtract p
    const ulong p[4] = { CT_FIELD_P0, CT_FIELD_P1, CT_FIELD_P2, CT_FIELD_P3 };
    ulong tmp[4];
    ulong borrow = ct_sub256(r->limbs, p, tmp);
    // Use tmp if carry || !borrow
    ulong do_sub = ct_is_nonzero_mask(carry) | ct_is_zero_mask(borrow);
    for (int i = 0; i < 4; ++i)
        r->limbs[i] = (r->limbs[i] & ~do_sub) | (tmp[i] & do_sub);
}

// ---------------------------------------------------------------------------
// CT field_sub: r = (a - b) mod p, branchless
// ---------------------------------------------------------------------------
inline void ct_field_sub_impl(FieldElement* r, const FieldElement* a, const FieldElement* b) {
    ulong borrow = ct_sub256(a->limbs, b->limbs, r->limbs);
    // If borrow, add p back
    const ulong p[4] = { CT_FIELD_P0, CT_FIELD_P1, CT_FIELD_P2, CT_FIELD_P3 };
    ulong tmp[4];
    ct_add256(r->limbs, p, tmp);
    ulong mask = ct_is_nonzero_mask(borrow);
    for (int i = 0; i < 4; ++i)
        r->limbs[i] = (r->limbs[i] & ~mask) | (tmp[i] & mask);
}

// ---------------------------------------------------------------------------
// CT field_neg: r = (-a) mod p = p - a (if a != 0), else 0
// ---------------------------------------------------------------------------
inline void ct_field_neg_impl(FieldElement* r, const FieldElement* a) {
    const ulong p[4] = { CT_FIELD_P0, CT_FIELD_P1, CT_FIELD_P2, CT_FIELD_P3 };
    ulong tmp[4];
    ct_sub256(p, a->limbs, tmp);
    ulong is_zero = ct_is_zero_mask(a->limbs[0] | a->limbs[1] | a->limbs[2] | a->limbs[3]);
    for (int i = 0; i < 4; ++i)
        r->limbs[i] = tmp[i] & ~is_zero;
}

// ---------------------------------------------------------------------------
// CT field_mul/sqr/inv: wrap fast-path (already data-independent instruction count)
// The value_barrier prevents the compiler from short-circuiting
// ---------------------------------------------------------------------------
inline void ct_field_mul(FieldElement* r, const FieldElement* a, const FieldElement* b) {
    FieldElement a2 = *a, b2 = *b;
    for (int i = 0; i < 4; ++i) {
        a2.limbs[i] = ct_value_barrier(a2.limbs[i]);
        b2.limbs[i] = ct_value_barrier(b2.limbs[i]);
    }
    field_mul_impl(r, &a2, &b2);
}

inline void ct_field_sqr(FieldElement* r, const FieldElement* a) {
    FieldElement a2 = *a;
    for (int i = 0; i < 4; ++i)
        a2.limbs[i] = ct_value_barrier(a2.limbs[i]);
    field_sqr_impl(r, &a2);
}

inline void ct_field_inv(FieldElement* r, const FieldElement* a) {
    FieldElement a2 = *a;
    for (int i = 0; i < 4; ++i)
        a2.limbs[i] = ct_value_barrier(a2.limbs[i]);
    field_inv_impl(r, &a2);
}

// CT field_half: r = a/2 mod p (branchless)
inline void ct_field_half_impl(FieldElement* r, const FieldElement* a) {
    ulong odd_mask = ct_bool_to_mask(a->limbs[0] & 1);
    const ulong p[4] = { CT_FIELD_P0, CT_FIELD_P1, CT_FIELD_P2, CT_FIELD_P3 };
    ulong tmp[4];
    ct_add256(a->limbs, p, tmp);
    // If odd, use (a+p)/2; else use a/2
    ulong src[4];
    for (int i = 0; i < 4; ++i)
        src[i] = (a->limbs[i] & ~odd_mask) | (tmp[i] & odd_mask);
    // Right shift by 1
    for (int i = 0; i < 3; ++i)
        r->limbs[i] = (src[i] >> 1) | (src[i + 1] << 63);
    r->limbs[3] = src[3] >> 1;
}

// ---------------------------------------------------------------------------
// CT field predicates (constant-time, branchless)
// ---------------------------------------------------------------------------
inline ulong ct_field_is_zero(const FieldElement* a) {
    ulong acc = 0;
    for (int i = 0; i < 4; ++i) acc |= a->limbs[i];
    return ct_is_zero_mask(acc);
}

inline ulong ct_field_eq(const FieldElement* a, const FieldElement* b) {
    ulong acc = 0;
    for (int i = 0; i < 4; ++i) acc |= (a->limbs[i] ^ b->limbs[i]);
    return ct_is_zero_mask(acc);
}

inline void ct_field_normalize(FieldElement* r) {
    ct_reduce_field(r);
}

#endif // SECP256K1_CT_FIELD_CL
