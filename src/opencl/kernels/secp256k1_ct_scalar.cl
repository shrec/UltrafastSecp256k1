// =============================================================================
// secp256k1_ct_scalar.cl -- Constant-time scalar arithmetic for OpenCL
// =============================================================================
// Branchless scalar ops mod n (secp256k1 order). Fermat-based inverse.
// GLV decomposition with CT reduce/select. No secret-dependent branches.
// Requires: secp256k1_extended.cl (scalar_*_impl), secp256k1_ct_ops.cl
// =============================================================================

#ifndef SECP256K1_CT_SCALAR_CL
#define SECP256K1_CT_SCALAR_CL

// secp256k1 order n
#define CT_ORDER_N0 0xBFD25E8CD0364141UL
#define CT_ORDER_N1 0xBAAEDCE6AF48A03BUL
#define CT_ORDER_N2 0xFFFFFFFFFFFFFFFEUL
#define CT_ORDER_N3 0xFFFFFFFFFFFFFFFFUL

// Half order (n+1)/2 for low-S normalization
#define CT_HALF_N0 0xDFE92F46681B20A1UL
#define CT_HALF_N1 0x5D576E7357A4501DUL
#define CT_HALF_N2 0xFFFFFFFFFFFFFFFFUL
#define CT_HALF_N3 0x7FFFFFFFFFFFFFFFUL

// ---------------------------------------------------------------------------
// Branchless reduce mod n: if val >= n, subtract n
// ---------------------------------------------------------------------------
static inline void ct_reduce_order(Scalar* r) {
    const ulong n[4] = { CT_ORDER_N0, CT_ORDER_N1, CT_ORDER_N2, CT_ORDER_N3 };
    ulong tmp[4];
    ulong borrow = ct_sub256(r->limbs, n, tmp);
    ulong mask = ct_bool_to_mask(borrow == 0);
    for (int i = 0; i < 4; ++i)
        r->limbs[i] = (r->limbs[i] & ~mask) | (tmp[i] & mask);
}

// ---------------------------------------------------------------------------
// CT scalar_add: r = (a + b) mod n
// ---------------------------------------------------------------------------
static inline void ct_scalar_add_impl(const Scalar* a, const Scalar* b, Scalar* r) {
    ulong carry = ct_add256(a->limbs, b->limbs, r->limbs);
    const ulong n[4] = { CT_ORDER_N0, CT_ORDER_N1, CT_ORDER_N2, CT_ORDER_N3 };
    ulong tmp[4];
    ulong borrow = ct_sub256(r->limbs, n, tmp);
    ulong do_sub = ct_is_nonzero_mask(carry) | ct_is_zero_mask(borrow);
    for (int i = 0; i < 4; ++i)
        r->limbs[i] = (r->limbs[i] & ~do_sub) | (tmp[i] & do_sub);
}

// ---------------------------------------------------------------------------
// CT scalar_sub: r = (a - b) mod n
// ---------------------------------------------------------------------------
static inline void ct_scalar_sub_impl(const Scalar* a, const Scalar* b, Scalar* r) {
    ulong borrow = ct_sub256(a->limbs, b->limbs, r->limbs);
    const ulong n[4] = { CT_ORDER_N0, CT_ORDER_N1, CT_ORDER_N2, CT_ORDER_N3 };
    ulong tmp[4];
    ct_add256(r->limbs, n, tmp);
    ulong mask = ct_is_nonzero_mask(borrow);
    for (int i = 0; i < 4; ++i)
        r->limbs[i] = (r->limbs[i] & ~mask) | (tmp[i] & mask);
}

// ---------------------------------------------------------------------------
// CT scalar_neg: r = (-a) mod n = n - a if a != 0
// ---------------------------------------------------------------------------
static inline void ct_scalar_neg_impl(const Scalar* a, Scalar* r) {
    const ulong n[4] = { CT_ORDER_N0, CT_ORDER_N1, CT_ORDER_N2, CT_ORDER_N3 };
    ulong tmp[4];
    ct_sub256(n, a->limbs, tmp);
    ulong is_zero = ct_is_zero_mask(a->limbs[0] | a->limbs[1] | a->limbs[2] | a->limbs[3]);
    for (int i = 0; i < 4; ++i)
        r->limbs[i] = tmp[i] & ~is_zero;
}

// ---------------------------------------------------------------------------
// CT scalar_half: r = a/2 mod n (branchless)
// ---------------------------------------------------------------------------
static inline void ct_scalar_half_impl(const Scalar* a, Scalar* r) {
    ulong odd_mask = ct_bool_to_mask(a->limbs[0] & 1);
    const ulong n[4] = { CT_ORDER_N0, CT_ORDER_N1, CT_ORDER_N2, CT_ORDER_N3 };
    ulong tmp[4];
    ct_add256(a->limbs, n, tmp);
    ulong src[4];
    for (int i = 0; i < 4; ++i)
        src[i] = (a->limbs[i] & ~odd_mask) | (tmp[i] & odd_mask);
    for (int i = 0; i < 3; ++i)
        r->limbs[i] = (src[i] >> 1) | (src[i + 1] << 63);
    r->limbs[3] = src[3] >> 1;
}

// ---------------------------------------------------------------------------
// CT scalar_mul/sqr: wrap fast-path (Montgomery is data-independent)
// ---------------------------------------------------------------------------
static inline void ct_scalar_mul_impl(const Scalar* a, const Scalar* b, Scalar* r) {
    Scalar a2 = *a, b2 = *b;
    for (int i = 0; i < 4; ++i) {
        a2.limbs[i] = ct_value_barrier(a2.limbs[i]);
        b2.limbs[i] = ct_value_barrier(b2.limbs[i]);
    }
    scalar_mul_mod_n_impl(&a2, &b2, r);
}

static inline void ct_scalar_sqr_impl(const Scalar* a, Scalar* r) {
    ct_scalar_mul_impl(a, a, r);
}

// ---------------------------------------------------------------------------
// CT scalar_inverse: Fermat's little theorem a^(n-2) mod n
// Fixed-trace: always 256 squares + 256 CT-selected multiplies
// ---------------------------------------------------------------------------
static inline void ct_scalar_inverse_impl(const Scalar* a, Scalar* r) {
    // n-2 in 4 limbs
    const ulong nm2[4] = {
        CT_ORDER_N0 - 2,
        CT_ORDER_N1,
        CT_ORDER_N2,
        CT_ORDER_N3
    };

    Scalar result;
    result.limbs[0] = 1; result.limbs[1] = 0;
    result.limbs[2] = 0; result.limbs[3] = 0;
    Scalar base = *a;

    for (int bit = 0; bit < 256; ++bit) {
        int limb_idx = bit >> 6;
        int bit_idx = bit & 63;
        ulong bit_val = (nm2[limb_idx] >> bit_idx) & 1;
        ulong mask = ct_bool_to_mask(bit_val != 0);

        // Always compute the product, conditionally store
        Scalar tmp;
        ct_scalar_mul_impl(&result, &base, &tmp);
        ct_scalar_cmov(&result, &tmp, mask);

        // Always square the base
        ct_scalar_sqr_impl(&base, &base);
    }
    *r = result;
}

// ---------------------------------------------------------------------------
// CT scalar predicates (scan ALL limbs, no early exit)
// ---------------------------------------------------------------------------
static inline ulong ct_scalar_is_zero(const Scalar* a) {
    ulong acc = 0;
    for (int i = 0; i < 4; ++i) acc |= a->limbs[i];
    return ct_is_zero_mask(acc);
}

static inline ulong ct_scalar_eq(const Scalar* a, const Scalar* b) {
    ulong acc = 0;
    for (int i = 0; i < 4; ++i) acc |= (a->limbs[i] ^ b->limbs[i]);
    return ct_is_zero_mask(acc);
}

// CT scalar_is_high: returns mask if s > n/2
static inline ulong ct_scalar_is_high(const Scalar* s) {
    ulong n_half[4] = { CT_HALF_N0, CT_HALF_N1, CT_HALF_N2, CT_HALF_N3 };
    // Compare s > n/2: check if n_half < s
    // Subtract: n_half - s, if borrow then s > n_half
    ulong tmp[4];
    ulong borrow = ct_sub256(n_half, s->limbs, tmp);
    return ct_is_nonzero_mask(borrow);
}

static inline int ct_scalar_bit(const Scalar* s, int pos) {
    int limb_idx = pos >> 6;
    int bit_idx = pos & 63;
    return (int)((s->limbs[limb_idx] >> bit_idx) & 1);
}

static inline int ct_scalar_window(const Scalar* s, int pos, int width) {
    int val = 0;
    for (int i = 0; i < width; ++i)
        val |= ct_scalar_bit(s, pos + i) << i;
    return val;
}

// ---------------------------------------------------------------------------
// Low-S normalization: if s > n/2, s = n - s (BIP-62 / BIP-340)
// ---------------------------------------------------------------------------
static inline void ct_scalar_normalize_low_s(Scalar* s) {
    ulong mask = ct_scalar_is_high(s);
    Scalar neg;
    ct_scalar_neg_impl(s, &neg);
    ct_scalar_cmov(s, &neg, mask);
}

// ---------------------------------------------------------------------------
// CT GLV decomposition (branchless)
// ---------------------------------------------------------------------------
// k == (+/-k1) + (+/-k2)*lambda (mod n) with k1, k2 < 2^128: Babai rounding on
// the secp256k1 lattice with the constants GLV_G1, GLV_G2, GLV_MINUS_B1,
// GLV_MINUS_B2 and GLV_LAMBDA0..3 that the variable-time glv_decompose_impl in
// secp256k1_extended.cl uses. Port of the CUDA ct_glv_decompose
// (src/cuda/include/ct/ct_scalar.cuh): fixed multiplies, masked mod-n
// add/sub/neg and mask selects only; no branch, loop bound or memory index
// depends on k. ct_generator_mul_impl and ct_scalar_mul_point scan 33 four-bit
// windows (132 bits) of k1 and k2, which is correct only because of the 2^128
// bound -- a k1 = k, k2 = 0 split silently drops the top bits of k.
typedef struct {
    Scalar k1;
    Scalar k2;
    ulong k1_neg;  // mask: all-ones if k1 was negated
    ulong k2_neg;  // mask: all-ones if k2 was negated
} CTGLVDecompositionOCL;

// CT (a * b) >> 384, rounded at bit 383: a fixed 4x4 schoolbook product, and
// the rounding bit is added as a carry instead of being branched on (the
// variable-time glv_mul_shift_384_impl branches on it). b is a lattice vector.
// For every 256-bit a the result is < 2^129, so limbs 2 and 3 are always 0.
static inline void ct_mul_shift_384_impl(const ulong a[4], __constant const ulong b[4],
                                         ulong r[4]) {
    ulong prod[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < 4; ++i) {
        ulong carry = 0;
        for (int j = 0; j < 4; ++j) {
            ulong2 full = mul64_full(a[i], b[j]);
            ulong c1 = 0, c2 = 0;
            ulong s = add_with_carry(full.x, prod[i + j], 0, &c1);
            s = add_with_carry(s, carry, 0, &c2);
            prod[i + j] = s;
            carry = full.y + c1 + c2;
        }
        prod[i + 4] = carry;
    }
    ulong c = 0;
    r[0] = add_with_carry(prod[6], prod[5] >> 63, 0, &c);
    r[1] = add_with_carry(prod[7], 0, c, &c);
    r[2] = 0;
    r[3] = 0;
}

static inline void ct_glv_decompose_impl(const Scalar* k, CTGLVDecompositionOCL* out) {
    // c1 = round(k * g1 / 2^384), c2 = round(k * g2 / 2^384); both < 2^129 < n
    Scalar c1, c2;
    ct_mul_shift_384_impl(k->limbs, GLV_G1, c1.limbs);
    ct_mul_shift_384_impl(k->limbs, GLV_G2, c2.limbs);

    // k2 = c1*(-b1) + c2*(-b2) (mod n)
    Scalar minus_b1, minus_b2;
    for (int i = 0; i < 4; ++i) {
        minus_b1.limbs[i] = GLV_MINUS_B1[i];
        minus_b2.limbs[i] = GLV_MINUS_B2[i];
    }
    Scalar t1, t2, k2_mod;
    ct_scalar_mul_impl(&c1, &minus_b1, &t1);
    ct_scalar_mul_impl(&c2, &minus_b2, &t2);
    ct_scalar_add_impl(&t1, &t2, &k2_mod);

    // k1 = k - lambda*k2 (mod n)
    Scalar lambda_s;
    lambda_s.limbs[0] = GLV_LAMBDA0; lambda_s.limbs[1] = GLV_LAMBDA1;
    lambda_s.limbs[2] = GLV_LAMBDA2; lambda_s.limbs[3] = GLV_LAMBDA3;
    Scalar lk2, k1_mod;
    ct_scalar_mul_impl(&lambda_s, &k2_mod, &lk2);
    ct_scalar_sub_impl(k, &lk2, &k1_mod);

    // Shorter form: a residue above n/2 is the negation of a value < 2^128.
    // Mask select, both negations always computed.
    out->k1_neg = ct_scalar_is_high(&k1_mod);
    out->k2_neg = ct_scalar_is_high(&k2_mod);
    Scalar neg;
    out->k1 = k1_mod;
    ct_scalar_neg_impl(&k1_mod, &neg);
    ct_scalar_cmov(&out->k1, &neg, out->k1_neg);
    out->k2 = k2_mod;
    ct_scalar_neg_impl(&k2_mod, &neg);
    ct_scalar_cmov(&out->k2, &neg, out->k2_neg);
}

#endif // SECP256K1_CT_SCALAR_CL
