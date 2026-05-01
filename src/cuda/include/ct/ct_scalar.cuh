#pragma once
// ============================================================================
// Constant-Time Scalar Arithmetic -- CUDA Device
// ============================================================================
// Side-channel resistant scalar operations for secp256k1 curve order.
// Uses the same Scalar type as fast path -- 4x64-bit limbs.
//
// CT guarantees:
//   - No secret-dependent branches
//   - No secret-dependent memory access patterns
//   - Fixed instruction count for all inputs
//
// Port of: cpu/include/secp256k1/ct/scalar.hpp + cpu/src/ct_scalar.cpp
// ============================================================================

#include "ct/ct_ops.cuh"
#include "ct/ct_field.cuh"

namespace secp256k1 {
namespace cuda {
namespace ct {

// n/2 for is_high check
__device__ __constant__ static const uint64_t HALF_ORDER[4] = {
    0xDFE92F46681B20A0ULL, 0x5D576E7357A4501DULL,
    0xFFFFFFFFFFFFFFFFULL, 0x7FFFFFFFFFFFFFFFULL
};

// --- Internal: branchless 256-bit ops with ORDER ----------------------------

// CT reduce once mod n: if value >= n, subtract n. Uses cmov.
__device__ __forceinline__
void ct_reduce_order(uint64_t r[4]) {
    uint64_t tmp[4];
    uint64_t borrow = sub256(tmp, r, ORDER);
    uint64_t mask = is_zero_mask(borrow);  // all-1s if no borrow (r >= n)
    cmov256(r, tmp, mask);
}

// --- CT Scalar Arithmetic (public API) ---------------------------------------

// CT modular addition: r = (a + b) mod n
__device__ __forceinline__
void scalar_add(const Scalar* a, const Scalar* b, Scalar* r) {
    uint64_t tmp[4];
    uint64_t carry = add256(tmp, a->limbs, b->limbs);
    uint64_t reduced[4];
    uint64_t borrow = sub256(reduced, tmp, ORDER);
    // Use reduced if carry occurred OR no borrow (tmp >= n)
    uint64_t use_reduced = is_nonzero_mask(carry) | is_zero_mask(borrow);
    ct_select256(r->limbs, reduced, tmp, use_reduced);
}

// CT modular subtraction: r = (a - b) mod n
__device__ __forceinline__
void scalar_sub(const Scalar* a, const Scalar* b, Scalar* r) {
    uint64_t tmp[4];
    uint64_t borrow = sub256(tmp, a->limbs, b->limbs);
    uint64_t mask = is_nonzero_mask(borrow);
    uint64_t corrected[4];
    add256(corrected, tmp, ORDER);
    ct_select256(r->limbs, corrected, tmp, mask);
}

// CT modular negation: r = -a mod n (branchless)
// Reuses the existing fast-path implementation which is already branchless.
__device__ __forceinline__
void scalar_neg(const Scalar* a, Scalar* r) {
    uint64_t tmp[4];
    sub256(tmp, ORDER, a->limbs);
    uint64_t nz = a->limbs[0] | a->limbs[1] | a->limbs[2] | a->limbs[3];
    uint64_t mask = is_nonzero_mask(nz);
    r->limbs[0] = tmp[0] & mask;
    r->limbs[1] = tmp[1] & mask;
    r->limbs[2] = tmp[2] & mask;
    r->limbs[3] = tmp[3] & mask;
}

// CT modular halving: r = a/2 mod n
// Already branchless in fast path (mask-based parity handling).
__device__ __forceinline__
void scalar_half(const Scalar* a, Scalar* r) {
    secp256k1::cuda::scalar_half(a, r);
}

// CT modular multiplication: r = (a * b) mod n
// Schoolbook 4x4 + Barrett reduction is fixed-instruction-count.
__device__ __forceinline__
void scalar_mul(const Scalar* a, const Scalar* b, Scalar* r) {
    secp256k1::cuda::scalar_mul_mod_n(a, b, r);
}

// CT modular squaring: r = a^2 mod n
__device__ __forceinline__
void scalar_sqr(const Scalar* a, Scalar* r) {
    secp256k1::cuda::scalar_sqr_mod_n(a, r);
}

// CT modular inverse: r = a^(-1) mod n (Fermat: a^(n-2))
// Fixed add-chain: always multiply regardless of bit value.
// Slower than fast-path but constant-time.
__device__ inline void scalar_inverse(const Scalar* a, Scalar* r) {
    Scalar result;
    result.limbs[0] = 1; result.limbs[1] = 0;
    result.limbs[2] = 0; result.limbs[3] = 0;
    Scalar base = *a;

    for (int i = 255; i >= 0; --i) {
        Scalar sqrd;
        scalar_sqr(&result, &sqrd);

        // Always compute the multiply
        Scalar mulled;
        scalar_mul(&sqrd, &base, &mulled);

        // CT select: if bit is set, use mulled; else use sqrd
        int limb_idx = i / 64;
        int bit_idx = i % 64;
        uint64_t bit = (ORDER_MINUS_2[limb_idx] >> bit_idx) & 1;
        uint64_t mask = bool_to_mask(bit);

        ct_select256(result.limbs, mulled.limbs, sqrd.limbs, mask);
    }
    *r = result;
}

// --- CT Scalar Conditional Operations ----------------------------------------

// CT conditional move: if mask == all-ones, *r = a
__device__ __forceinline__
void scalar_cmov(Scalar* r, const Scalar* a, uint64_t mask) {
    cmov256(r->limbs, a->limbs, mask);
}

// CT conditional swap: when mask is all-ones, swaps a and b
__device__ __forceinline__
void scalar_cswap(Scalar* a, Scalar* b, uint64_t mask) {
    cswap256(a->limbs, b->limbs, mask);
}

// CT select: if mask == all-ones, r = a; else r = b
__device__ __forceinline__
void scalar_select(Scalar* r, const Scalar* a, const Scalar* b, uint64_t mask) {
    ct_select256(r->limbs, a->limbs, b->limbs, mask);
}

// CT conditional negate: if mask == all-ones, r = -a; else r = a
__device__ __forceinline__
void scalar_cneg(Scalar* r, const Scalar* a, uint64_t mask) {
    Scalar neg;
    scalar_neg(a, &neg);
    scalar_select(r, &neg, a, mask);
}

// --- CT Scalar Comparisons (mask-based, not bool) ----------------------------

// Returns all-ones mask if a == 0, else 0
__device__ __forceinline__
uint64_t scalar_is_zero(const Scalar* a) {
    uint64_t acc = a->limbs[0] | a->limbs[1] | a->limbs[2] | a->limbs[3];
    return is_zero_mask(acc);
}

// Returns all-ones mask if a == b, else 0
__device__ __forceinline__
uint64_t scalar_eq(const Scalar* a, const Scalar* b) {
    uint64_t diff = (a->limbs[0] ^ b->limbs[0]) |
                    (a->limbs[1] ^ b->limbs[1]) |
                    (a->limbs[2] ^ b->limbs[2]) |
                    (a->limbs[3] ^ b->limbs[3]);
    return is_zero_mask(diff);
}

// Returns all-ones mask if a > n/2, else 0 (CT, no early-exit)
__device__ __forceinline__
uint64_t scalar_is_high(const Scalar* a) {
    // Compare a > HALF_ORDER lexicographically from high to low
    // Result is the OR of all "a[i] > half[i]" while all higher limbs are equal
    uint64_t gt = 0;  // accumulated "greater than"
    uint64_t eq_so_far = ~(uint64_t)0;  // all equal so far (all-ones)
    for (int i = 3; i >= 0; --i) {
        uint64_t a_gt_h = lt_mask(HALF_ORDER[i], a->limbs[i]);  // half < a => a > half
        uint64_t a_eq_h = eq_mask(a->limbs[i], HALF_ORDER[i]);
        gt |= (a_gt_h & eq_so_far);  // this limb is greater AND all above were equal
        eq_so_far &= a_eq_h;
    }
    return gt;
}

// --- CT Bit Access -----------------------------------------------------------

// Returns bit at position 'index' (0 = LSB). CT (always same computation).
__device__ __forceinline__
uint64_t scalar_bit(const Scalar* a, int index) {
    int limb_idx = index >> 6;
    int bit_idx = index & 63;
    uint64_t result = (a->limbs[limb_idx] >> bit_idx) & 1;
    return result;
}

// Returns w-bit window at position 'pos' (0 = LSB). CT.
__device__ __forceinline__
uint64_t scalar_window(const Scalar* a, int pos, int width) {
    int limb_idx = pos >> 6;
    int bit_pos = pos & 63;
    uint64_t mask_w = ((uint64_t)1 << width) - 1;
    uint64_t result = (a->limbs[limb_idx] >> bit_pos) & mask_w;
    // Handle case where window spans two limbs
    int remaining = 64 - bit_pos;
    if (remaining < width && limb_idx < 3) {
        uint64_t extra = a->limbs[limb_idx + 1] & (((uint64_t)1 << (width - remaining)) - 1);
        result |= (extra << remaining);
    }
    return result;
}

// --- CT ECDSA Low-S Normalize ------------------------------------------------

// CT low-S normalization: if s > n/2 return n-s, else return s.
// Branchless comparison + conditional negate.
__device__ __forceinline__
void scalar_normalize_low_s(Scalar* s) {
    uint64_t high = scalar_is_high(s);
    Scalar neg;
    scalar_neg(s, &neg);
    scalar_cmov(s, &neg, high);
}

// --- CT GLV Decompose --------------------------------------------------------

struct CTGLVDecomposition {
    Scalar k1, k2;
    uint64_t k1_neg;  // all-ones mask if negated, 0 otherwise
    uint64_t k2_neg;
};

// CT GLV decomposition: k = k1 + k2*lambda (mod n)
// No branches on k value. Uses CT comparison for sign selection.
__device__ inline CTGLVDecomposition ct_glv_decompose(const Scalar* k) {
    CTGLVDecomposition result;

    // Step 1: c1 = round(k * g1 / 2^384), c2 = round(k * g2 / 2^384)
    uint64_t c1_limbs[4], c2_limbs[4];
    mul_shift_384(k->limbs, GLV_G1, c1_limbs);
    mul_shift_384(k->limbs, GLV_G2, c2_limbs);

    Scalar c1, c2;
    for (int i = 0; i < 4; i++) {
        c1.limbs[i] = c1_limbs[i];
        c2.limbs[i] = c2_limbs[i];
    }
    // CT normalize
    ct_reduce_order(c1.limbs);
    ct_reduce_order(c2.limbs);

    // Step 2: k2 = c1*(-b1) + c2*(-b2) (mod n)
    Scalar minus_b1, minus_b2;
    for (int i = 0; i < 4; i++) {
        minus_b1.limbs[i] = GLV_MINUS_B1[i];
        minus_b2.limbs[i] = GLV_MINUS_B2[i];
    }

    Scalar t1, t2, k2_mod;
    scalar_mul(&c1, &minus_b1, &t1);
    scalar_mul(&c2, &minus_b2, &t2);
    scalar_add(&t1, &t2, &k2_mod);

    // Step 3: CT pick shorter representation for k2
    Scalar k2_neg_val;
    scalar_neg(&k2_mod, &k2_neg_val);
    // CT comparison: is_high(k2_mod) means k2_mod > n/2, so negation is shorter
    uint64_t k2_is_neg = scalar_is_high(&k2_mod);
    Scalar k2_abs;
    scalar_select(&k2_abs, &k2_neg_val, &k2_mod, k2_is_neg);

    // k2_signed = k2_is_neg ? -k2_abs : k2_abs = k2_is_neg ? k2_mod : k2_mod
    // Actually: k2_signed = k2_mod always (the sign just tells us which form to use)
    Scalar k2_signed = k2_mod;

    // Step 4: k1 = k - lambda*k2_signed (mod n)
    Scalar lambda_s;
    for (int i = 0; i < 4; i++) lambda_s.limbs[i] = LAMBDA[i];
    Scalar lk2;
    scalar_mul(&lambda_s, &k2_signed, &lk2);
    Scalar k1_mod;
    scalar_sub(k, &lk2, &k1_mod);

    // Step 5: CT pick shorter representation for k1
    Scalar k1_neg_val;
    scalar_neg(&k1_mod, &k1_neg_val);
    uint64_t k1_is_neg = scalar_is_high(&k1_mod);
    Scalar k1_abs;
    scalar_select(&k1_abs, &k1_neg_val, &k1_mod, k1_is_neg);

    result.k1 = k1_abs;
    result.k2 = k2_abs;
    result.k1_neg = k1_is_neg;
    result.k2_neg = k2_is_neg;

    return result;
}

} // namespace ct
} // namespace cuda
} // namespace secp256k1
