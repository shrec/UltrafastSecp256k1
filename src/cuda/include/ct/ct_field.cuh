#pragma once
// ============================================================================
// Constant-Time Field Arithmetic -- CUDA Device
// ============================================================================
// Side-channel resistant field operations for secp256k1.
// Uses the same FieldElement type as fast path -- 4x64-bit limbs.
//
// CT guarantees:
//   - No secret-dependent branches (branchless add/sub/normalize)
//   - No secret-dependent memory access patterns
//   - Fixed instruction count for all inputs
//
// The fast::field_mul / fast::field_sqr are inherently CT (fixed Comba/PTX),
// so ct::field_mul / ct::field_sqr just wrap them with a CT normalize.
// field_inv uses Fermat chain a^(p-2) which is CT (fixed exponent).
//
// Port of: cpu/include/secp256k1/ct/field.hpp + cpu/src/ct_field.cpp
// ============================================================================

#include "ct/ct_ops.cuh"
#include "secp256k1.cuh"

namespace secp256k1 {
namespace cuda {
namespace ct {

// --- Internal: branchless 256-bit add/sub ------------------------------------

// CT 256-bit addition with carry out. Returns carry (0 or 1).
__device__ __forceinline__
uint64_t add256(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t r0, r1, r2, r3, carry;
#if SECP256K1_USE_PTX
    asm volatile(
        "add.cc.u64  %0, %5, %9;  \n\t"
        "addc.cc.u64 %1, %6, %10; \n\t"
        "addc.cc.u64 %2, %7, %11; \n\t"
        "addc.cc.u64 %3, %8, %12; \n\t"
        "addc.u64    %4, 0, 0;    \n\t"
        : "=l"(r0), "=l"(r1), "=l"(r2), "=l"(r3), "=l"(carry)
        : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]),
          "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3])
    );
#else
    unsigned __int128 sum;
    carry = 0;
    sum = (unsigned __int128)a[0] + b[0]; r0 = (uint64_t)sum; carry = (uint64_t)(sum >> 64);
    sum = (unsigned __int128)a[1] + b[1] + carry; r1 = (uint64_t)sum; carry = (uint64_t)(sum >> 64);
    sum = (unsigned __int128)a[2] + b[2] + carry; r2 = (uint64_t)sum; carry = (uint64_t)(sum >> 64);
    sum = (unsigned __int128)a[3] + b[3] + carry; r3 = (uint64_t)sum; carry = (uint64_t)(sum >> 64);
#endif
    r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
    return carry;
}

// CT 256-bit subtraction with borrow out. Returns borrow (0 or 1).
__device__ __forceinline__
uint64_t sub256(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t r0, r1, r2, r3, borrow;
#if SECP256K1_USE_PTX
    asm volatile(
        "sub.cc.u64  %0, %5, %9;  \n\t"
        "subc.cc.u64 %1, %6, %10; \n\t"
        "subc.cc.u64 %2, %7, %11; \n\t"
        "subc.cc.u64 %3, %8, %12; \n\t"
        "subc.u64    %4, 0, 0;    \n\t"
        : "=l"(r0), "=l"(r1), "=l"(r2), "=l"(r3), "=l"(borrow)
        : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]),
          "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3])
    );
    // PTX subc.u64 %4,0,0 gives 0xFFFFFFFFFFFFFFFF on borrow, 0 otherwise
    borrow &= 1;
#else
    unsigned __int128 diff;
    borrow = 0;
    diff = (unsigned __int128)a[0] - b[0]; r0 = (uint64_t)diff; borrow = (diff >> 127) & 1;
    diff = (unsigned __int128)a[1] - b[1] - borrow; r1 = (uint64_t)diff; borrow = (diff >> 127) & 1;
    diff = (unsigned __int128)a[2] - b[2] - borrow; r2 = (uint64_t)diff; borrow = (diff >> 127) & 1;
    diff = (unsigned __int128)a[3] - b[3] - borrow; r3 = (uint64_t)diff; borrow = (diff >> 127) & 1;
#endif
    r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
    return borrow;
}

// CT reduce once: if value >= p, subtract p. Uses cmov (no branch).
__device__ __forceinline__
void ct_reduce_field(uint64_t r[4]) {
    uint64_t tmp[4];
    uint64_t borrow = sub256(tmp, r, MODULUS);
    // If borrow == 0, r >= p -> use tmp (reduced). Else keep r.
    uint64_t mask = is_zero_mask(borrow);  // all-1s if no borrow (r >= p)
    cmov256(r, tmp, mask);
}

// --- CT Field Arithmetic (public API) ----------------------------------------

// CT modular addition: r = (a + b) mod p
__device__ __forceinline__
void field_add(const FieldElement* a, const FieldElement* b, FieldElement* r) {
    uint64_t tmp[4];
    uint64_t carry = add256(tmp, a->limbs, b->limbs);
    // Try to subtract p
    uint64_t reduced[4];
    uint64_t borrow = sub256(reduced, tmp, MODULUS);
    // Use reduced if: carry (overflow 256 bits) OR no borrow (tmp >= p)
    // Matches fast path logic: if (carry || borrow == 0) use reduced
    uint64_t use_reduced = is_nonzero_mask(carry) | is_zero_mask(borrow);
    ct_select256(r->limbs, reduced, tmp, use_reduced);
}

// CT modular subtraction: r = (a - b) mod p
__device__ __forceinline__
void field_sub(const FieldElement* a, const FieldElement* b, FieldElement* r) {
    uint64_t tmp[4];
    uint64_t borrow = sub256(tmp, a->limbs, b->limbs);
    // If borrow, add p back. mask = all-1s if borrow occurred.
    uint64_t mask = is_nonzero_mask(borrow);
    uint64_t corrected[4];
    add256(corrected, tmp, MODULUS);
    ct_select256(r->limbs, corrected, tmp, mask);
}

// CT modular negation: r = -a mod p
// Always computes p - a; if a == 0, result is 0 (p - 0 overflows to p, reduce)
__device__ __forceinline__
void field_neg(const FieldElement* a, FieldElement* r) {
    // p - a is always correct for a in [1, p-1]
    // For a == 0: p - 0 = p, which we reduce to 0
    uint64_t tmp[4];
    sub256(tmp, MODULUS, a->limbs);
    // Zero check: if a was 0, tmp == p, need to set to 0
    uint64_t a_nz = a->limbs[0] | a->limbs[1] | a->limbs[2] | a->limbs[3];
    uint64_t mask = is_nonzero_mask(a_nz);
    r->limbs[0] = tmp[0] & mask;
    r->limbs[1] = tmp[1] & mask;
    r->limbs[2] = tmp[2] & mask;
    r->limbs[3] = tmp[3] & mask;
}

// CT modular multiplication: r = (a * b) mod p
// The underlying mul is already fixed-instruction-count.
__device__ __forceinline__
void field_mul(const FieldElement* a, const FieldElement* b, FieldElement* r) {
    secp256k1::cuda::field_mul(a, b, r);
}

// CT modular squaring: r = a^2 mod p
__device__ __forceinline__
void field_sqr(const FieldElement* a, FieldElement* r) {
    secp256k1::cuda::field_sqr(a, r);
}

// CT modular inverse: r = a^(-1) mod p (Fermat: a^(p-2))
// Fixed add-chain: always same number of mul+sqr regardless of input.
__device__ __forceinline__
void field_inv(const FieldElement* a, FieldElement* r) {
    secp256k1::cuda::field_inv(a, r);
}

// CT modular half: r = a/2 mod p
// Already branchless in the fast path (mask-based).
__device__ __forceinline__
void field_half(const FieldElement* a, FieldElement* r) {
    secp256k1::cuda::field_half(a, r);
}

// --- CT Field Conditional Operations -----------------------------------------

// CT conditional move: if mask == all-ones, *r = a; else unchanged
__device__ __forceinline__
void field_cmov(FieldElement* r, const FieldElement* a, uint64_t mask) {
    cmov256(r->limbs, a->limbs, mask);
}

// CT conditional swap: when mask is all-ones, swaps a and b
__device__ __forceinline__
void field_cswap(FieldElement* a, FieldElement* b, uint64_t mask) {
    cswap256(a->limbs, b->limbs, mask);
}

// CT select: returns a if mask==all-ones, else b
__device__ __forceinline__
void field_select(FieldElement* r, const FieldElement* a, const FieldElement* b, uint64_t mask) {
    ct_select256(r->limbs, a->limbs, b->limbs, mask);
}

// CT conditional negate: if mask == all-ones, r = -a; else r = a
__device__ __forceinline__
void field_cneg(FieldElement* r, const FieldElement* a, uint64_t mask) {
    FieldElement neg;
    field_neg(a, &neg);
    field_select(r, &neg, a, mask);
}

// --- CT Field Comparisons (mask-based) ---------------------------------------

// Returns all-ones mask if a == 0, else 0.
__device__ __forceinline__
uint64_t field_is_zero(const FieldElement* a) {
    uint64_t acc = a->limbs[0] | a->limbs[1] | a->limbs[2] | a->limbs[3];
    return is_zero_mask(acc);
}

// Returns all-ones mask if a == b, else 0.
__device__ __forceinline__
uint64_t field_eq(const FieldElement* a, const FieldElement* b) {
    uint64_t diff = (a->limbs[0] ^ b->limbs[0]) |
                    (a->limbs[1] ^ b->limbs[1]) |
                    (a->limbs[2] ^ b->limbs[2]) |
                    (a->limbs[3] ^ b->limbs[3]);
    return is_zero_mask(diff);
}

// CT normalize: ensure value in [0, p). Branchless.
__device__ __forceinline__
void field_normalize(FieldElement* a) {
    ct_reduce_field(a->limbs);
}

} // namespace ct
} // namespace cuda
} // namespace secp256k1
