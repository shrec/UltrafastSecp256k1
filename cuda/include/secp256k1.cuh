#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "hash160.cuh"
#include "secp256k1/types.hpp"

namespace secp256k1 {
namespace cuda {

#include "ptx_math.cuh"

#if SECP256K1_CUDA_LIMBS_32
#include "secp256k1_32.cuh"
#else

// If enabled, use optimized 32-bit hybrid mul/sqr (1.10x faster!)
#ifndef SECP256K1_CUDA_USE_HYBRID_MUL
#define SECP256K1_CUDA_USE_HYBRID_MUL 1
#endif

// If enabled, field_mul/field_sqr use Montgomery multiplication.
// NOTE: Values are treated as Montgomery residues when this is enabled.
// Call field_to_mont / field_from_mont at domain boundaries.
#ifndef SECP256K1_CUDA_USE_MONTGOMERY
#define SECP256K1_CUDA_USE_MONTGOMERY 0
#endif

// Field element representation (4 x 64-bit limbs)
// Little-endian: limbs[0] is least significant
// Uses shared POD type from secp256k1/types.hpp
using FieldElement = ::secp256k1::FieldElementData;

// Scalar (256-bit integer)
// Uses shared POD type from secp256k1/types.hpp
using Scalar = ::secp256k1::ScalarData;

// Jacobian Point (X, Y, Z) where affine (x, y) = (X/Z^2, Y/Z^3)
// Backend-specific: uses bool infinity for CUDA compatibility
struct JacobianPoint {
    FieldElement x;
    FieldElement y;
    FieldElement z;
    bool infinity;
};

// Affine Point (x, y)
// Uses shared POD type from secp256k1/types.hpp
using AffinePoint = ::secp256k1::AffinePointData;

// Compile-time verification
static_assert(sizeof(FieldElement) == 32, "Must be 256 bits");

// Cross-backend layout compatibility (shared types contract)
static_assert(sizeof(FieldElement) == sizeof(::secp256k1::FieldElementData),
              "CUDA FieldElement must match shared data layout");
static_assert(sizeof(Scalar) == sizeof(::secp256k1::ScalarData),
              "CUDA Scalar must match shared data layout");
static_assert(sizeof(AffinePoint) == sizeof(::secp256k1::AffinePointData),
              "CUDA AffinePoint must match shared data layout");

// Constants
__constant__ static const uint64_t MODULUS[4] = {
    0xFFFFFFFEFFFFFC2FULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL
};

// 32-bit modulus view (same value, different representation)
__constant__ static const uint32_t MODULUS_32[8] = {
    0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

// Montgomery constants for p = 2^256 - 0x1000003D1.
// R = 2^256 mod p = 0x1000003D1
// R^2 mod p = (0x1000003D1)^2 mod p
// R^3 mod p = R^2 * R mod p

// Standard 1
__constant__ static const FieldElement FIELD_ONE = {
    {1ULL, 0ULL, 0ULL, 0ULL}
};

// R mod p = 2^256 mod p = 0x1000003D1
__constant__ static const FieldElement FIELD_R = {
    {0x1000003D1ULL, 0ULL, 0ULL, 0ULL}
};

// R^2 mod p = (2^256)^2 mod p
__constant__ static const FieldElement FIELD_R2 = {
    {0x000007A2000E90A1ULL, 0x0000000000000001ULL, 0x0000000000000000ULL, 0x0000000000000000ULL}
};

// R^3 mod p = (2^256)^3 mod p
__constant__ static const FieldElement FIELD_R3 = {
    {0x002BB1E33795F671ULL, 0x0000000100000B73ULL, 0x0000000000000000ULL, 0x0000000000000000ULL}
};

// R^(-1) mod p = (2^256)^-1 mod p
__constant__ static const FieldElement FIELD_R_INV = {
    {0xD838091D0868192AULL, 0xBCB223FEDC24A059ULL, 0x9C46C2C295F2B761ULL, 0xC9BD190515538399ULL}
};

// Helper functions for backward compatibility
__device__ __forceinline__ void field_const_one(FieldElement* r) {
    *r = FIELD_ONE;
}

// 1 in Montgomery domain: 1*R mod p = R mod p
__device__ __forceinline__ void field_const_one_mont(FieldElement* r) {
    *r = FIELD_R;
}

// K = 2^32 + 977 = 0x1000003D1
__constant__ static const uint64_t K_MOD = 0x1000003D1ULL;

// Initialize field element to 0 (works in all modes)
__device__ __forceinline__ void field_set_zero(FieldElement* r) {
    r->limbs[0] = 0; r->limbs[1] = 0; r->limbs[2] = 0; r->limbs[3] = 0;
}

// Initialize field element to 1 (domain-aware)
__device__ __forceinline__ void field_set_one(FieldElement* r) {
#if SECP256K1_CUDA_USE_MONTGOMERY
    field_const_one_mont(r);  // 1 in Montgomery domain = R mod p
#else
    field_const_one(r);       // 1 in standard domain
#endif
}

// Check if field element is zero
__device__ __forceinline__ bool field_is_zero(const FieldElement* a) {
    return (a->limbs[0] | a->limbs[1] | a->limbs[2] | a->limbs[3]) == 0;
}

// Check if two field elements are equal
__device__ __forceinline__ bool field_eq(const FieldElement* a, const FieldElement* b) {
    return (a->limbs[0] == b->limbs[0]) &&
           (a->limbs[1] == b->limbs[1]) &&
           (a->limbs[2] == b->limbs[2]) &&
           (a->limbs[3] == b->limbs[3]);
}

// Scalar order N
__constant__ static const uint64_t ORDER[4] = {
    0xBFD25E8CD0364141ULL,
    0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL,
    0xFFFFFFFFFFFFFFFFULL
};

// GLV constants
__constant__ static const uint64_t LAMBDA[4] = {
    0xDF02967C1B23BD72ULL,
    0x122E22EA20816678ULL,
    0xA5261C028812645AULL,
    0x5363AD4CC05C30E0ULL
};

__constant__ static const uint64_t BETA[4] = {
    0xC1396C28719501EEULL,
    0x9CF0497512F58995ULL,
    0x6E64479EAC3434E9ULL,
    0x7AE96A2B657C0710ULL
};

// Generator point G in affine coordinates
__constant__ static const uint64_t GENERATOR_X[4] = {
    0x59F2815B16F81798ULL,
    0x029BFCDB2DCE28D9ULL,
    0x55A06295CE870B07ULL,
    0x79BE667EF9DCBBACULL
};

__constant__ static const uint64_t GENERATOR_Y[4] = {
    0x9C47D08FFB10D4B8ULL,
    0xFD17B448A6855419ULL,
    0x5DA4FBFC0E1108A8ULL,
    0x483ADA7726A3C465ULL
};

// Hash160 kernel: RIPEMD160(SHA256(pubkey)) with no prefixes/base58
__global__ void hash160_pubkey_kernel(const uint8_t* pubkeys, int pubkey_len, uint8_t* out_hashes, int count);

// Helper functions for 128-bit arithmetic
__device__ __forceinline__ uint64_t add_cc(uint64_t a, uint64_t b, uint64_t& carry) {
    unsigned __int128 sum = (unsigned __int128)a + b + carry;
    carry = (uint64_t)(sum >> 64);
    return (uint64_t)sum;
}

// Forward decls used by Montgomery helpers.
__device__ inline void mul_256_512(const FieldElement* a, const FieldElement* b, uint64_t r[8]);
__device__ inline void sqr_256_512(const FieldElement* a, uint64_t r[8]);

// Montgomery reduction for 512-bit input t[0..7] (little-endian limbs).
// Computes t * R^{-1} mod p, where p = 2^256 - k, k = 0x1000003D1.
// We use n' = -p^{-1} mod 2^64. Since p0 = 2^64 - k, p0^{-1} = (-k)^{-1}.
// For k = 2^32 + 977, (-k)^{-1} mod 2^64 = -(0x1000003D1)^{-1}.
// Numerically, n' = 0xD838091DD2253531.
__device__ __forceinline__ void mont_reduce_512(const uint64_t t_in[8], FieldElement* r) {
    uint64_t t0 = t_in[0], t1 = t_in[1], t2 = t_in[2], t3 = t_in[3];
    uint64_t t4 = t_in[4], t5 = t_in[5], t6 = t_in[6], t7 = t_in[7];

    constexpr uint64_t N0_INV = 0xD838091DD2253531ULL;
    // K = 2^32 + 977

    uint64_t m, m_977_lo, m_977_hi, x0, x1, c_x;
    uint64_t top_carry = 0;
    uint64_t b, c;

    // Iteration 0
    m = t0 * N0_INV;
    asm volatile("mul.lo.u64 %0, %2, 977; mul.hi.u64 %1, %2, 977;" : "=l"(m_977_lo), "=l"(m_977_hi) : "l"(m));
    x0 = m_977_lo + (m << 32);
    c_x = (x0 < m_977_lo);
    x1 = m_977_hi + (m >> 32) + c_x;

    asm volatile(
        "sub.cc.u64 %0, %0, %8; \n\t"
        "subc.cc.u64 %1, %1, 0; \n\t"
        "subc.cc.u64 %2, %2, 0; \n\t"
        "subc.cc.u64 %3, %3, 0; \n\t"
        "subc.cc.u64 %4, %4, 0; \n\t"
        "subc.cc.u64 %5, %5, 0; \n\t"
        "subc.cc.u64 %6, %6, 0; \n\t"
        "subc.u64 %7, 0, 0; \n\t"
        : "+l"(t1), "+l"(t2), "+l"(t3), "+l"(t4), "+l"(t5), "+l"(t6), "+l"(t7), "=l"(b)
        : "l"(x1)
    );

    asm volatile(
        "add.cc.u64 %0, %0, %5; \n\t"
        "addc.cc.u64 %1, %1, 0; \n\t"
        "addc.cc.u64 %2, %2, 0; \n\t"
        "addc.cc.u64 %3, %3, 0; \n\t"
        "addc.u64 %4, 0, 0; \n\t"
        : "+l"(t4), "+l"(t5), "+l"(t6), "+l"(t7), "=l"(c)
        : "l"(m)
    );
    top_carry += c + b;

    // Iteration 1
    m = t1 * N0_INV;
    asm volatile("mul.lo.u64 %0, %2, 977; mul.hi.u64 %1, %2, 977;" : "=l"(m_977_lo), "=l"(m_977_hi) : "l"(m));
    x0 = m_977_lo + (m << 32);
    c_x = (x0 < m_977_lo);
    x1 = m_977_hi + (m >> 32) + c_x;

    asm volatile(
        "sub.cc.u64 %0, %0, %7; \n\t"
        "subc.cc.u64 %1, %1, 0; \n\t"
        "subc.cc.u64 %2, %2, 0; \n\t"
        "subc.cc.u64 %3, %3, 0; \n\t"
        "subc.cc.u64 %4, %4, 0; \n\t"
        "subc.cc.u64 %5, %5, 0; \n\t"
        "subc.u64 %6, 0, 0; \n\t"
        : "+l"(t2), "+l"(t3), "+l"(t4), "+l"(t5), "+l"(t6), "+l"(t7), "=l"(b)
        : "l"(x1)
    );

    asm volatile(
        "add.cc.u64 %0, %0, %4; \n\t"
        "addc.cc.u64 %1, %1, 0; \n\t"
        "addc.cc.u64 %2, %2, 0; \n\t"
        "addc.u64 %3, 0, 0; \n\t"
        : "+l"(t5), "+l"(t6), "+l"(t7), "=l"(c)
        : "l"(m)
    );
    top_carry += c + b;

    // Iteration 2
    m = t2 * N0_INV;
    asm volatile("mul.lo.u64 %0, %2, 977; mul.hi.u64 %1, %2, 977;" : "=l"(m_977_lo), "=l"(m_977_hi) : "l"(m));
    x0 = m_977_lo + (m << 32);
    c_x = (x0 < m_977_lo);
    x1 = m_977_hi + (m >> 32) + c_x;

    asm volatile(
        "sub.cc.u64 %0, %0, %6; \n\t"
        "subc.cc.u64 %1, %1, 0; \n\t"
        "subc.cc.u64 %2, %2, 0; \n\t"
        "subc.cc.u64 %3, %3, 0; \n\t"
        "subc.cc.u64 %4, %4, 0; \n\t"
        "subc.u64 %5, 0, 0; \n\t"
        : "+l"(t3), "+l"(t4), "+l"(t5), "+l"(t6), "+l"(t7), "=l"(b)
        : "l"(x1)
    );

    asm volatile(
        "add.cc.u64 %0, %0, %3; \n\t"
        "addc.cc.u64 %1, %1, 0; \n\t"
        "addc.u64 %2, 0, 0; \n\t"
        : "+l"(t6), "+l"(t7), "=l"(c)
        : "l"(m)
    );
    top_carry += c + b;

    // Iteration 3
    m = t3 * N0_INV;
    asm volatile("mul.lo.u64 %0, %2, 977; mul.hi.u64 %1, %2, 977;" : "=l"(m_977_lo), "=l"(m_977_hi) : "l"(m));
    x0 = m_977_lo + (m << 32);
    c_x = (x0 < m_977_lo);
    x1 = m_977_hi + (m >> 32) + c_x;

    asm volatile(
        "sub.cc.u64 %0, %0, %5; \n\t"
        "subc.cc.u64 %1, %1, 0; \n\t"
        "subc.cc.u64 %2, %2, 0; \n\t"
        "subc.cc.u64 %3, %3, 0; \n\t"
        "subc.u64 %4, 0, 0; \n\t"
        : "+l"(t4), "+l"(t5), "+l"(t6), "+l"(t7), "=l"(b)
        : "l"(x1)
    );

    asm volatile(
        "add.cc.u64 %0, %0, %2; \n\t"
        "addc.u64 %1, 0, 0; \n\t"
        : "+l"(t7), "=l"(c)
        : "l"(m)
    );
    top_carry += c + b;

    // Result in t4, t5, t6, t7
    // We want to compute: if (result >= P) result -= P;
    // Since P = 2^256 - K, result >= 2^256 - K  <==> result + K >= 2^256.
    // So we compute result + K. If it overflows (carry out), then result >= P.
    // Also if top_carry is set, result >= 2^256 > P, so we definitely subtract P.
    // Subtracting P is equivalent to Adding K (mod 2^256).
    // So in both cases (top_carry or carry_out), the answer is (result + K) mod 2^256.
    // Otherwise, the answer is result.

    uint64_t k0, k1, k2, k3, k_carry;
    k0 = add_cc(t4, K_MOD, k_carry);
    k1 = add_cc(t5, 0, k_carry);
    k2 = add_cc(t6, 0, k_carry);
    k3 = add_cc(t7, 0, k_carry);
    
    // If top_carry is 1, we must use k0..k3.
    // If k_carry is 1, we must use k0..k3.
    bool use_k = (top_carry != 0) || (k_carry != 0);
    
    if (use_k) {
        r->limbs[0] = k0;
        r->limbs[1] = k1;
        r->limbs[2] = k2;
        r->limbs[3] = k3;
    } else {
        r->limbs[0] = t4;
        r->limbs[1] = t5;
        r->limbs[2] = t6;
        r->limbs[3] = t7;
    }
}

// Forward declarations for Montgomery conversion functions (defined after hybrid include)
__device__ __forceinline__ void field_to_mont(const FieldElement* a, FieldElement* r);
__device__ __forceinline__ void field_from_mont(const FieldElement* a, FieldElement* r);
__device__ __forceinline__ void field_mul_mont(const FieldElement* a, const FieldElement* b, FieldElement* r);
__device__ __forceinline__ void field_sqr_mont(const FieldElement* a, FieldElement* r);

__device__ __forceinline__ uint64_t sub_cc(uint64_t a, uint64_t b, uint64_t& borrow) {
    unsigned __int128 diff = (unsigned __int128)a - b - borrow;
    borrow = (uint64_t)((diff >> 127) & 1);
    return (uint64_t)diff;
}

__device__ __forceinline__ void mul64(uint64_t a, uint64_t b, uint64_t& lo, uint64_t& hi) {
    asm volatile(
        "mul.lo.u64 %0, %2, %3; \n\t"
        "mul.hi.u64 %1, %2, %3; \n\t"
        : "=l"(lo), "=l"(hi)
        : "l"(a), "l"(b)
    );
}

// Scalar helper functions
__device__ __forceinline__ bool scalar_ge(const Scalar* a, const uint64_t* b) {
    for (int i = 3; i >= 0; --i) {
        if (a->limbs[i] > b[i]) return true;
        if (a->limbs[i] < b[i]) return false;
    }
    return true;
}

__device__ inline void scalar_add(const Scalar* a, const Scalar* b, Scalar* r) {
    uint64_t carry = 0;
    r->limbs[0] = add_cc(a->limbs[0], b->limbs[0], carry);
    r->limbs[1] = add_cc(a->limbs[1], b->limbs[1], carry);
    r->limbs[2] = add_cc(a->limbs[2], b->limbs[2], carry);
    r->limbs[3] = add_cc(a->limbs[3], b->limbs[3], carry);
    
    // Conditional subtraction of ORDER
    uint64_t borrow = 0;
    uint64_t t0 = sub_cc(r->limbs[0], ORDER[0], borrow);
    uint64_t t1 = sub_cc(r->limbs[1], ORDER[1], borrow);
    uint64_t t2 = sub_cc(r->limbs[2], ORDER[2], borrow);
    uint64_t t3 = sub_cc(r->limbs[3], ORDER[3], borrow);
    
    if (carry || borrow == 0) {
        r->limbs[0] = t0; r->limbs[1] = t1; r->limbs[2] = t2; r->limbs[3] = t3;
    }
}

__device__ inline void scalar_sub(const Scalar* a, const Scalar* b, Scalar* r) {
    uint64_t borrow = 0;
    r->limbs[0] = sub_cc(a->limbs[0], b->limbs[0], borrow);
    r->limbs[1] = sub_cc(a->limbs[1], b->limbs[1], borrow);
    r->limbs[2] = sub_cc(a->limbs[2], b->limbs[2], borrow);
    r->limbs[3] = sub_cc(a->limbs[3], b->limbs[3], borrow);
    
    if (borrow) {
        uint64_t carry = 0;
        r->limbs[0] = add_cc(r->limbs[0], ORDER[0], carry);
        r->limbs[1] = add_cc(r->limbs[1], ORDER[1], carry);
        r->limbs[2] = add_cc(r->limbs[2], ORDER[2], carry);
        // Optimization: last limb addition doesn't need to capture carry
        r->limbs[3] += ORDER[3] + carry;
    }
}

__device__ inline void scalar_add_u64(const Scalar* a, uint64_t b, Scalar* r) {
    uint64_t carry = 0;
    r->limbs[0] = add_cc(a->limbs[0], b, carry);
    r->limbs[1] = add_cc(a->limbs[1], 0, carry);
    r->limbs[2] = add_cc(a->limbs[2], 0, carry);
    r->limbs[3] = add_cc(a->limbs[3], 0, carry);
    
    // Conditional subtraction of ORDER
    uint64_t borrow = 0;
    uint64_t t0 = sub_cc(r->limbs[0], ORDER[0], borrow);
    uint64_t t1 = sub_cc(r->limbs[1], ORDER[1], borrow);
    uint64_t t2 = sub_cc(r->limbs[2], ORDER[2], borrow);
    uint64_t t3 = sub_cc(r->limbs[3], ORDER[3], borrow);
    
    if (carry || borrow == 0) {
        r->limbs[0] = t0; r->limbs[1] = t1; r->limbs[2] = t2; r->limbs[3] = t3;
    }
}

__device__ inline void scalar_sub_u64(const Scalar* a, uint64_t b, Scalar* r) {
    uint64_t borrow = 0;
    r->limbs[0] = sub_cc(a->limbs[0], b, borrow);
    r->limbs[1] = sub_cc(a->limbs[1], 0, borrow);
    r->limbs[2] = sub_cc(a->limbs[2], 0, borrow);
    r->limbs[3] = sub_cc(a->limbs[3], 0, borrow);
    
    if (borrow) {
        uint64_t carry = 0;
        r->limbs[0] = add_cc(r->limbs[0], ORDER[0], carry);
        r->limbs[1] = add_cc(r->limbs[1], ORDER[1], carry);
        r->limbs[2] = add_cc(r->limbs[2], ORDER[2], carry);
        r->limbs[3] += ORDER[3] + carry;
    }
}

__device__ inline bool scalar_is_zero(const Scalar* s) {
    return (s->limbs[0] | s->limbs[1] | s->limbs[2] | s->limbs[3]) == 0;
}

__device__ inline uint8_t scalar_bit(const Scalar* s, int index) {
    if (index >= 256) return 0;
    int limb_idx = index / 64;
    int bit_idx = index % 64;
    return (s->limbs[limb_idx] >> bit_idx) & 1;
}

// ============================================================================
// Standard 64-bit field operations (used for add/sub - faster for these!)
// ============================================================================

// Field Addition (PTX Optimized)
__device__ __forceinline__ void field_add(const FieldElement* a, const FieldElement* b, FieldElement* r) {
    uint64_t r0, r1, r2, r3, carry;
    
    asm volatile(
        "add.cc.u64 %0, %5, %9; \n\t"
        "addc.cc.u64 %1, %6, %10; \n\t"
        "addc.cc.u64 %2, %7, %11; \n\t"
        "addc.cc.u64 %3, %8, %12; \n\t"
        "addc.u64 %4, 0, 0; \n\t"
        : "=l"(r0), "=l"(r1), "=l"(r2), "=l"(r3), "=l"(carry)
        : "l"(a->limbs[0]), "l"(a->limbs[1]), "l"(a->limbs[2]), "l"(a->limbs[3]),
          "l"(b->limbs[0]), "l"(b->limbs[1]), "l"(b->limbs[2]), "l"(b->limbs[3])
    );
    
    uint64_t t0, t1, t2, t3, borrow;
    asm volatile(
        "sub.cc.u64 %0, %5, %9; \n\t"
        "subc.cc.u64 %1, %6, %10; \n\t"
        "subc.cc.u64 %2, %7, %11; \n\t"
        "subc.cc.u64 %3, %8, %12; \n\t"
        "subc.u64 %4, 0, 0; \n\t"
        : "=l"(t0), "=l"(t1), "=l"(t2), "=l"(t3), "=l"(borrow)
        : "l"(r0), "l"(r1), "l"(r2), "l"(r3),
          "l"(MODULUS[0]), "l"(MODULUS[1]), "l"(MODULUS[2]), "l"(MODULUS[3])
    );
    
    if (carry || borrow == 0) {
        r->limbs[0] = t0; r->limbs[1] = t1; r->limbs[2] = t2; r->limbs[3] = t3;
    } else {
        r->limbs[0] = r0; r->limbs[1] = r1; r->limbs[2] = r2; r->limbs[3] = r3;
    }
}

// Field Subtraction (PTX Optimized)
__device__ __forceinline__ void field_sub(const FieldElement* a, const FieldElement* b, FieldElement* r) {
    uint64_t r0, r1, r2, r3, borrow;
    
    asm volatile(
        "sub.cc.u64 %0, %5, %9; \n\t"
        "subc.cc.u64 %1, %6, %10; \n\t"
        "subc.cc.u64 %2, %7, %11; \n\t"
        "subc.cc.u64 %3, %8, %12; \n\t"
        "subc.u64 %4, 0, 0; \n\t"
        : "=l"(r0), "=l"(r1), "=l"(r2), "=l"(r3), "=l"(borrow)
        : "l"(a->limbs[0]), "l"(a->limbs[1]), "l"(a->limbs[2]), "l"(a->limbs[3]),
          "l"(b->limbs[0]), "l"(b->limbs[1]), "l"(b->limbs[2]), "l"(b->limbs[3])
    );
    
    if (borrow) {
        asm volatile(
            "add.cc.u64 %0, %4, %8; \n\t"
            "addc.cc.u64 %1, %5, %9; \n\t"
            "addc.cc.u64 %2, %6, %10; \n\t"
            "addc.u64 %3, %7, %11; \n\t"
            : "=l"(r->limbs[0]), "=l"(r->limbs[1]), "=l"(r->limbs[2]), "=l"(r->limbs[3])
            : "l"(r0), "l"(r1), "l"(r2), "l"(r3),
              "l"(MODULUS[0]), "l"(MODULUS[1]), "l"(MODULUS[2]), "l"(MODULUS[3])
        );
    } else {
        r->limbs[0] = r0; r->limbs[1] = r1; r->limbs[2] = r2; r->limbs[3] = r3;
    }
}

// Field Negation: r = -a (mod P)
__device__ inline void field_negate(const FieldElement* a, FieldElement* r) {
    // -a = P - a
    uint64_t r0, r1, r2, r3;
    
    asm volatile(
        "sub.cc.u64 %0, %4, %8; \n\t"
        "subc.cc.u64 %1, %5, %9; \n\t"
        "subc.cc.u64 %2, %6, %10; \n\t"
        "subc.u64 %3, %7, %11; \n\t"
        : "=l"(r0), "=l"(r1), "=l"(r2), "=l"(r3)
        : "l"(MODULUS[0]), "l"(MODULUS[1]), "l"(MODULUS[2]), "l"(MODULUS[3]),
          "l"(a->limbs[0]), "l"(a->limbs[1]), "l"(a->limbs[2]), "l"(a->limbs[3])
    );
    
    r->limbs[0] = r0;
    r->limbs[1] = r1;
    r->limbs[2] = r2;
    r->limbs[3] = r3;
}

// Field multiplication by small constant: r = a * small (mod P)
// Optimized for small constants (e.g., 7, 28 for secp256k1)
__device__ inline void field_mul_small(const FieldElement* a, uint32_t small, FieldElement* r) {
    // Simple approach: multiply and reduce
    // For small constants, this is faster than full field_mul
    __uint128_t carry = 0;
    uint64_t tmp[4];
    
    for (int i = 0; i < 4; i++) {
        __uint128_t prod = (__uint128_t)a->limbs[i] * small + carry;
        tmp[i] = (uint64_t)prod;
        carry = prod >> 64;
    }
    
    // Now we have a 320-bit number: tmp[0..3] + carry * 2^256
    // Reduce carry * 2^256 mod P
    // Since P = 2^256 - 0x1000003d1, we have 2^256 ≡ 0x1000003d1 (mod P)
    // So carry * 2^256 ≡ carry * 0x1000003d1
    
    uint64_t c = (uint64_t)carry;
    if (c > 0) {
        // carry * 0x1000003d1 = carry * (2^32 + 0x3d1) = carry << 32 + carry * 977
        __uint128_t reduction = (__uint128_t)c * 0x1000003d1ULL;
        
        __uint128_t sum = (__uint128_t)tmp[0] + (uint64_t)reduction;
        r->limbs[0] = (uint64_t)sum;
        
        sum = (__uint128_t)tmp[1] + (reduction >> 64) + (sum >> 64);
        r->limbs[1] = (uint64_t)sum;
        
        sum = (__uint128_t)tmp[2] + (sum >> 64);
        r->limbs[2] = (uint64_t)sum;
        
        sum = (__uint128_t)tmp[3] + (sum >> 64);
        r->limbs[3] = (uint64_t)sum;
        
        // Final carry (very rare for small constants)
        if (sum >> 64) {
            __uint128_t final_red = (sum >> 64) * 0x1000003d1ULL;
            sum = (__uint128_t)r->limbs[0] + (uint64_t)final_red;
            r->limbs[0] = (uint64_t)sum;
            r->limbs[1] += (sum >> 64);
        }
    } else {
        r->limbs[0] = tmp[0];
        r->limbs[1] = tmp[1];
        r->limbs[2] = tmp[2];
        r->limbs[3] = tmp[3];
    }
}

// Full 256x256 -> 512 multiplication
__device__ __forceinline__ void mul_256_512(const FieldElement* a, const FieldElement* b, uint64_t r[8]) {
    mul_256_512_ptx(a->limbs, b->limbs, r);
}

// Full 256 -> 512 squaring
__device__ __forceinline__ void sqr_256_512(const FieldElement* a, uint64_t r[8]) {
    sqr_256_512_ptx(a->limbs, r);
}

__device__ __forceinline__ void reduce_512_to_256(uint64_t t[8], FieldElement* r) {
    // P = 2^256 - K_MOD, where K_MOD = 2^32 + 977 = 0x1000003D1
    // T = T_hi * 2^256 + T_lo ≡ T_hi * K_MOD + T_lo (mod P)
    //
    // OPTIMIZATION: Multiply T_hi by K_MOD directly in one MAD chain,
    // instead of splitting into T_hi*977 + T_hi<<32 (two separate passes).
    // Saves ~26 instructions and 7 registers per call.
    
    uint64_t t0 = t[0], t1 = t[1], t2 = t[2], t3 = t[3];
    uint64_t t4 = t[4], t5 = t[5], t6 = t[6], t7 = t[7];
    
    // 1. Compute A = T_hi * K_MOD (5 limbs: a0..a4)
    //    Single MAD chain — replaces separate *977 + <<32 two-pass approach
    uint64_t a0, a1, a2, a3, a4;
    
    asm volatile(
        "mul.lo.u64 %0, %5, %9; \n\t"
        "mul.hi.u64 %1, %5, %9; \n\t"
        
        "mad.lo.cc.u64 %1, %6, %9, %1; \n\t"
        "madc.hi.u64 %2, %6, %9, 0; \n\t"
        
        "mad.lo.cc.u64 %2, %7, %9, %2; \n\t"
        "madc.hi.u64 %3, %7, %9, 0; \n\t"
        
        "mad.lo.cc.u64 %3, %8, %9, %3; \n\t"
        "madc.hi.u64 %4, %8, %9, 0; \n\t"
        
        : "=l"(a0), "=l"(a1), "=l"(a2), "=l"(a3), "=l"(a4)
        : "l"(t4), "l"(t5), "l"(t6), "l"(t7), "l"(K_MOD)
    );
    
    // 2. Add A[0..3] to T_lo
    uint64_t carry;
    asm volatile(
        "add.cc.u64 %0, %0, %5; \n\t"
        "addc.cc.u64 %1, %1, %6; \n\t"
        "addc.cc.u64 %2, %2, %7; \n\t"
        "addc.cc.u64 %3, %3, %8; \n\t"
        "addc.u64 %4, 0, 0; \n\t"
        : "+l"(t0), "+l"(t1), "+l"(t2), "+l"(t3), "=l"(carry)
        : "l"(a0), "l"(a1), "l"(a2), "l"(a3)
    );
    
    // 3. Reduce overflow: extra = a4 + carry (≤ 2^33 + 1)
    //    extra * K_MOD fits in 2 limbs (≤ 2^66)
    uint64_t extra = a4 + carry;
    uint64_t ek_lo, ek_hi;
    asm volatile(
        "mul.lo.u64 %0, %2, %3; \n\t"
        "mul.hi.u64 %1, %2, %3; \n\t"
        : "=l"(ek_lo), "=l"(ek_hi)
        : "l"(extra), "l"(K_MOD)
    );
    
    uint64_t c;
    asm volatile(
        "add.cc.u64 %0, %0, %5; \n\t"
        "addc.cc.u64 %1, %1, %6; \n\t"
        "addc.cc.u64 %2, %2, 0; \n\t"
        "addc.cc.u64 %3, %3, 0; \n\t"
        "addc.u64 %4, 0, 0; \n\t"
        : "+l"(t0), "+l"(t1), "+l"(t2), "+l"(t3), "=l"(c)
        : "l"(ek_lo), "l"(ek_hi)
    );
    
    // 4. Rare carry overflow (probability ≈ 2^{-190})
    if (c) {
        asm volatile(
            "add.cc.u64 %0, %0, %4; \n\t"
            "addc.cc.u64 %1, %1, 0; \n\t"
            "addc.cc.u64 %2, %2, 0; \n\t"
            "addc.u64 %3, %3, 0; \n\t"
            : "+l"(t0), "+l"(t1), "+l"(t2), "+l"(t3)
            : "l"(K_MOD)
        );
    }
    
    // 5. Conditional subtraction of P
    uint64_t r0, r1, r2, r3, borrow;
    asm volatile(
        "sub.cc.u64 %0, %5, %9; \n\t"
        "subc.cc.u64 %1, %6, %10; \n\t"
        "subc.cc.u64 %2, %7, %11; \n\t"
        "subc.cc.u64 %3, %8, %12; \n\t"
        "subc.u64 %4, 0, 0; \n\t"
        : "=l"(r0), "=l"(r1), "=l"(r2), "=l"(r3), "=l"(borrow)
        : "l"(t0), "l"(t1), "l"(t2), "l"(t3),
          "l"(MODULUS[0]), "l"(MODULUS[1]), "l"(MODULUS[2]), "l"(MODULUS[3])
    );
    
    if (borrow == 0) {
        r->limbs[0] = r0; r->limbs[1] = r1; r->limbs[2] = r2; r->limbs[3] = r3;
    } else {
        r->limbs[0] = t0; r->limbs[1] = t1; r->limbs[2] = t2; r->limbs[3] = t3;
    }
}

// ============================================================================
// HYBRID 32-bit smart operations (include AFTER reduce_512_to_256)
// Smart hybrid: proven 32-bit mul + proven 64-bit reduce
// ============================================================================
#if SECP256K1_CUDA_USE_HYBRID_MUL
#include "secp256k1_32_hybrid_final.cuh"
#endif

// Forward declarations for functions defined later
__device__ inline void field_inv(const FieldElement* a, FieldElement* r);
__device__ inline void jacobian_double(const JacobianPoint* p, JacobianPoint* r);
__device__ inline void jacobian_add_mixed(const JacobianPoint* p, const AffinePoint* q, JacobianPoint* r);

// Forward declarations for Montgomery hybrid functions (when both toggles enabled)
#if SECP256K1_CUDA_USE_HYBRID_MUL
__device__ __forceinline__ void field_mul_mont_hybrid(const FieldElement* a, const FieldElement* b, FieldElement* r);
__device__ __forceinline__ void field_sqr_mont_hybrid(const FieldElement* a, FieldElement* r);
#endif

// ============================================================================
// Montgomery operation implementations (defined here to use hybrid functions)
// ============================================================================

// Montgomery multiplication: inputs and output are Montgomery residues.
// Returns MontMul(aR, bR) = abR (mod p).
__device__ __forceinline__ void field_mul_mont(const FieldElement* a, const FieldElement* b, FieldElement* r) {
#if SECP256K1_CUDA_USE_HYBRID_MUL
    // Use fast 32-bit hybrid multiplication + Montgomery reduction!
    field_mul_mont_hybrid(a, b, r);
#else
    uint64_t t[8];
    mul_256_512(a, b, t);
    mont_reduce_512(t, r);
#endif
}

__device__ __forceinline__ void field_sqr_mont(const FieldElement* a, FieldElement* r) {
#if SECP256K1_CUDA_USE_HYBRID_MUL
    // Use fast 32-bit hybrid squaring + Montgomery reduction!
    field_sqr_mont_hybrid(a, r);
#else
    uint64_t t[8];
    sqr_256_512(a, t);
    mont_reduce_512(t, r);
#endif
}

__device__ __forceinline__ void field_to_mont(const FieldElement* a, FieldElement* r) {
    // Convert a (standard) -> aR (Montgomery): MontMul(a, R^2).
    field_mul_mont(a, &FIELD_R2, r);
}

__device__ __forceinline__ void field_from_mont(const FieldElement* a, FieldElement* r) {
    // Convert a (Montgomery residue aR) -> a (standard): MontMul(aR, 1).
    FieldElement one;
    field_const_one(&one);
    field_mul_mont(a, &one, r);
}
__device__ inline void jacobian_double(const JacobianPoint* p, JacobianPoint* r);
__device__ inline void jacobian_add_mixed(const JacobianPoint* p, const AffinePoint* q, JacobianPoint* r);

#endif // SECP256K1_CUDA_LIMBS_32

#ifndef SECP256K1_CUDA_LIMBS_32
// Field Multiplication with Reduction
// Uses smart hybrid: proven 32-bit mul + proven 64-bit reduce
__device__ __forceinline__ void field_mul(const FieldElement* a, const FieldElement* b, FieldElement* r) {
#if SECP256K1_CUDA_USE_MONTGOMERY
    field_mul_mont(a, b, r);
#elif SECP256K1_CUDA_USE_HYBRID_MUL
    // Use proven hybrid: 32-bit PTX mul + standard 64-bit reduction
    field_mul_hybrid(a, b, r);
#else
    uint64_t t[8];
    mul_256_512(a, b, t);
    reduce_512_to_256(t, r);
#endif
}

// Field Squaring - uses proven hybrid
__device__ __forceinline__ void field_sqr(const FieldElement* a, FieldElement* r) {
#if SECP256K1_CUDA_USE_MONTGOMERY
    field_sqr_mont(a, r);
#elif SECP256K1_CUDA_USE_HYBRID_MUL
    // Use proven hybrid: 32-bit PTX sqr + standard 64-bit reduction
    field_sqr_hybrid(a, r);
#else
    uint64_t t[8];
    sqr_256_512(a, t);
    reduce_512_to_256(t, r);
#endif
}

#endif // !SECP256K1_CUDA_LIMBS_32

// Point doubling - dbl-2001-b formula for a=0 curves (secp256k1)
// Optimized: all computation in local registers, write output once at end
// 3M + 4S + 7add/sub (matches OpenCL kernel throughput)
__device__ inline void jacobian_double(const JacobianPoint* p, JacobianPoint* r) {
    if (p->infinity) {
        r->infinity = true;
        return;
    }
    
    // Check if Y == 0 (use helper for limb-agnostic check)
    if (field_is_zero(&p->y)) {
        r->infinity = true;
        return;
    }

    FieldElement S, M, X3, Y3, Z3, YY, YYYY, t1;

    // YY = Y^2  [1S]
    field_sqr(&p->y, &YY);

    // S = 4*X*Y^2  [1M + 2add]
    field_mul(&p->x, &YY, &S);
    field_add(&S, &S, &S);
    field_add(&S, &S, &S);

    // M = 3*X^2  [2S + 2add]
    field_sqr(&p->x, &M);
    field_add(&M, &M, &t1);     // t1 = 2*X^2
    field_add(&M, &t1, &M);     // M = 3*X^2

    // X3 = M^2 - 2*S  [3S + 1add + 1sub]
    field_sqr(&M, &X3);
    field_add(&S, &S, &t1);     // t1 = 2*S
    field_sub(&X3, &t1, &X3);

    // YYYY = Y^4  [4S]
    field_sqr(&YY, &YYYY);

    // Y3 = M*(S - X3) - 8*Y^4  [1sub + 2M + 3add + 1sub]
    field_add(&YYYY, &YYYY, &t1);   // 2*Y^4
    field_add(&t1, &t1, &t1);       // 4*Y^4
    field_add(&t1, &t1, &t1);       // 8*Y^4
    field_sub(&S, &X3, &S);         // S - X3 (reuse S)
    field_mul(&M, &S, &Y3);         // M*(S - X3)
    field_sub(&Y3, &t1, &Y3);       // Y3 final

    // Z3 = 2*Y*Z  [3M + 1add]
    field_mul(&p->y, &p->z, &Z3);
    field_add(&Z3, &Z3, &Z3);

    // Write output once
    r->x = X3;
    r->y = Y3;
    r->z = Z3;
    r->infinity = false;
}

// Mixed addition: P (Jacobian) + Q (Affine) -> Result (Jacobian)
// All computation in local registers, single output write at end
__device__ inline void jacobian_add_mixed(const JacobianPoint* p, const AffinePoint* q, JacobianPoint* r) {
    if (p->infinity) {
        r->x = q->x;
        r->y = q->y;
        field_set_one(&r->z);
        r->infinity = false;
        return;
    }
    
    FieldElement z1z1, u2, s2, h, hh, i, j, rr, v;
    FieldElement X3, Y3, Z3, t1, t2;

    // Z1² [1S]
    field_sqr(&p->z, &z1z1);
    
    // U2 = X2*Z1² [1M]
    field_mul(&q->x, &z1z1, &u2);
    
    // S2 = Y2*Z1³ [2M, 3M]
    field_mul(&p->z, &z1z1, &t1);
    field_mul(&q->y, &t1, &s2);
    
    // H = U2 - X1
    field_sub(&u2, &p->x, &h);

    // Check if same x-coordinate (branchless zero check)
    if (field_is_zero(&h)) {
        // rr = S2 - Y1
        field_sub(&s2, &p->y, &t1);
        if (field_is_zero(&t1)) {
            jacobian_double(p, r);
            return;
        }
        r->infinity = true;
        return;
    }
    
    // HH = H² [2S]
    field_sqr(&h, &hh);
    
    // I = 4*HH
    field_add(&hh, &hh, &i);
    field_add(&i, &i, &i);
    
    // J = H*I [4M]
    field_mul(&h, &i, &j);
    
    // rr = 2*(S2 - Y1)
    field_sub(&s2, &p->y, &t1);
    field_add(&t1, &t1, &rr);
    
    // V = X1*I [5M]
    field_mul(&p->x, &i, &v);
    
    // X3 = rr² - J - 2*V [3S]
    field_sqr(&rr, &X3);
    field_sub(&X3, &j, &X3);
    field_add(&v, &v, &t1);
    field_sub(&X3, &t1, &X3);
    
    // Y3 = rr*(V - X3) - 2*Y1*J [6M, 7M]
    field_sub(&v, &X3, &t1);
    field_mul(&rr, &t1, &Y3);
    field_mul(&p->y, &j, &t2);
    field_add(&t2, &t2, &t2);
    field_sub(&Y3, &t2, &Y3);
    
    // Z3 = (Z1+H)² - Z1² - HH [4S]
    field_add(&p->z, &h, &t1);
    field_sqr(&t1, &Z3);
    field_sub(&Z3, &z1z1, &Z3);
    field_sub(&Z3, &hh, &Z3);

    // Write output once
    r->x = X3;
    r->y = Y3;
    r->z = Z3;
    r->infinity = false;
}

// Using madd-2004-hmv formula (8M + 3S) - original baseline
__device__ inline void jacobian_add_mixed_h(const JacobianPoint* p, const AffinePoint* q, JacobianPoint* r, FieldElement& h_out) {
    if (p->infinity) {
        r->x = q->x;
        r->y = q->y;
        field_set_one(&r->z);
        r->infinity = false;
        field_set_one(&h_out);
        return;
    }

    // Z1² [1S]
    FieldElement z1z1;
    field_sqr(&p->z, &z1z1);

    // U2 = X2*Z1² [1M]
    FieldElement u2;
    field_mul(&q->x, &z1z1, &u2);

    // S2 = Y2*Z1³ [2M]
    FieldElement s2, temp;
    field_mul(&p->z, &z1z1, &temp);  // Z1³
    field_mul(&q->y, &temp, &s2);

    // Check if same point
    bool x_eq = field_eq(&p->x, &u2);

    if (x_eq) {
        bool y_eq = field_eq(&p->y, &s2);
        if (y_eq) {
            jacobian_double(p, r);
            field_set_one(&h_out);
            return;
        }
        r->infinity = true;
        field_set_one(&h_out);
        return;
    }

    // H = U2 - X1
    FieldElement h;
    field_sub(&u2, &p->x, &h);

    h_out = h; // Return H directly (Z_{n+1} = Z_n * H)

    // HH = H² [1S]
    FieldElement hh;
    field_sqr(&h, &hh);

    // HHH = H³ [1M]
    FieldElement hhh;
    field_mul(&h, &hh, &hhh);

    // r = S2 - Y1
    FieldElement rr;
    field_sub(&s2, &p->y, &rr);

    // V = X1 * H² [1M]
    FieldElement v;
    field_mul(&p->x, &hh, &v);

    // X3 = r² - H³ - 2*V [1S]
    FieldElement X3, Y3, Z3, t1;
    field_add(&v, &v, &t1);
    field_sqr(&rr, &X3);
    field_sub(&X3, &hhh, &X3);
    field_sub(&X3, &t1, &X3);

    // Y3 = r*(V - X3) - Y1*H³ [2M]
    field_mul(&p->y, &hhh, &t1);
    field_sub(&v, &X3, &v);       // reuse v
    field_mul(&rr, &v, &Y3);
    field_sub(&Y3, &t1, &Y3);

    // Z3 = Z1 * H [1M]
    field_mul(&p->z, &h, &Z3);

    // Write output once
    r->x = X3;
    r->y = Y3;
    r->z = Z3;
    r->infinity = false;
}

// H2-based variant: Optimized madd-2007-bl formula (7M + 4S)
// Returns h_out = 2*H to maintain serial inversion invariant (Z3 = 2*Z1*H)
__device__ inline void jacobian_add_mixed_h2(const JacobianPoint* p, const AffinePoint* q, JacobianPoint* r, FieldElement& h_out) {
    if (p->infinity) {
        r->x = q->x;
        r->y = q->y;
        field_set_one(&r->z);
        r->infinity = false;
        field_set_one(&h_out);
        return;
    }

    // Z1Z1 = Z1² [1S]
    FieldElement z1z1;
    field_sqr(&p->z, &z1z1);

    // U2 = X2*Z1Z1 [1M]
    FieldElement u2;
    field_mul(&q->x, &z1z1, &u2);

    // S2 = Y2*Z1*Z1Z1 [2M]
    FieldElement s2, z1_cubed;
    field_mul(&p->z, &z1z1, &z1_cubed);
    field_mul(&q->y, &z1_cubed, &s2);

    // Check if same point
    bool x_eq = field_eq(&p->x, &u2);

    if (x_eq) {
        bool y_eq = field_eq(&p->y, &s2);
        if (y_eq) {
            jacobian_double(p, r);
            field_set_one(&h_out);
            return;
        }
        r->infinity = true;
        field_set_one(&h_out);
        return;
    }

    // H = U2 - X1
    FieldElement h;
    field_sub(&u2, &p->x, &h);

    // HH = H² [1S]
    FieldElement hh;
    field_sqr(&h, &hh);

    // I = 4*HH (cheap: 2 adds)
    FieldElement i_val;
    field_add(&hh, &hh, &i_val);
    field_add(&i_val, &i_val, &i_val);

    // J = H*I [1M]
    FieldElement j, temp;
    field_mul(&h, &i_val, &j);

    // r = 2*(S2-Y1) (cheap: sub + add)
    FieldElement rr;
    field_sub(&s2, &p->y, &rr);
    field_add(&rr, &rr, &rr);

    // V = X1*I [1M]
    FieldElement v;
    field_mul(&p->x, &i_val, &v);

    // X3 = r²-J-2*V [1S]
    FieldElement X3, Y3, Z3;
    field_add(&v, &v, &temp);
    field_sqr(&rr, &X3);
    field_sub(&X3, &j, &X3);
    field_sub(&X3, &temp, &X3);

    // Y3 = r*(V-X3) - 2*Y1*J [2M]
    FieldElement y1j;
    field_mul(&p->y, &j, &y1j);
    field_add(&y1j, &y1j, &y1j);
    field_sub(&v, &X3, &temp);
    field_mul(&rr, &temp, &Y3);
    field_sub(&Y3, &y1j, &Y3);

    // Z3 = (Z1+H)²-Z1Z1-HH = 2*Z1*H [1S instead of 1M!]
    field_add(&p->z, &h, &temp);
    field_sqr(&temp, &Z3);
    field_sub(&Z3, &z1z1, &Z3);
    field_sub(&Z3, &hh, &Z3);

    // Return 2*H for serial inversion: Z_n = Z_0 * ∏(2*H_i) = Z_0 * 2^N * ∏H_i
    field_add(&h, &h, &h_out);

    // Write output once
    r->x = X3;
    r->y = Y3;
    r->z = Z3;
    r->infinity = false;
}

// Z=1 specialized variant: When input has Z=1, skip Z powers (5M + 2S vs 8M + 3S)
// Use this for the FIRST step after affine initialization (saves 3 mul + 1 sqr!)
// Assumes: p->z == 1 (caller must ensure this)
__device__ inline void jacobian_add_mixed_h_z1(const JacobianPoint* p, const AffinePoint* q, JacobianPoint* r, FieldElement& h_out) {
    // When Z1 = 1:
    // Z1² = 1, Z1³ = 1
    // U2 = X2 * 1 = X2  (0 mul saved!)
    // S2 = Y2 * 1 = Y2  (2 mul saved!)
    
    // H = X2 - X1 (since U2 = X2)
    FieldElement h;
    field_sub(&q->x, &p->x, &h);

    // Check for same point (X2 == X1)
    bool h_is_zero = field_is_zero(&h);
    if (h_is_zero) {
        // Check Y: if Y2 == Y1, double; else infinity
        FieldElement y_diff;
        field_sub(&q->y, &p->y, &y_diff);
        if (field_is_zero(&y_diff)) {
            jacobian_double(p, r);
            field_set_one(&h_out);
            return;
        }
        r->infinity = true;
        field_set_one(&h_out);
        return;
    }

    h_out = h;  // Return H directly

    // HH = H² [1S]
    FieldElement hh;
    field_sqr(&h, &hh);

    // HHH = H³ [1M]
    FieldElement hhh;
    field_mul(&h, &hh, &hhh);

    // r = Y2 - Y1 (since S2 = Y2)
    FieldElement rr;
    field_sub(&q->y, &p->y, &rr);

    // V = X1 * H² [1M]
    FieldElement v;
    field_mul(&p->x, &hh, &v);

    // X3 = r² - H³ - 2*V [1S]
    FieldElement X3, Y3, t1;
    field_add(&v, &v, &t1);
    field_sqr(&rr, &X3);
    field_sub(&X3, &hhh, &X3);
    field_sub(&X3, &t1, &X3);

    // Y3 = r*(V - X3) - Y1*H³ [2M]
    field_mul(&p->y, &hhh, &t1);
    field_sub(&v, &X3, &v);       // reuse v
    field_mul(&rr, &v, &Y3);
    field_sub(&Y3, &t1, &Y3);

    // Z3 = 1 * H = H [0M saved! just copy]
    // Write output once
    r->x = X3;
    r->y = Y3;
    r->z = h;
    r->infinity = false;
}

// Constant-point variant: Optimized for adding a CONSTANT affine point
// Takes X2, Y2 directly as separate FieldElements (not via pointer/struct)
// This allows: 1) Better register allocation 2) Removal of branch checks
// Uses madd-2004-hmv formula (8M + 3S) - same as baseline
// Note: No infinity/same-point checks (constant G is never infinity, collision with Q is negligible)
__device__ inline void jacobian_add_mixed_const(
    const JacobianPoint* p,
    const FieldElement& qx,   // Constant X coordinate (pass by ref from const memory)
    const FieldElement& qy,   // Constant Y coordinate (pass by ref from const memory)
    JacobianPoint* r,
    FieldElement& h_out
) {
    // Z1² [1S]
    FieldElement z1z1;
    field_sqr(&p->z, &z1z1);

    // U2 = X2*Z1² [1M]
    FieldElement u2;
    field_mul(&qx, &z1z1, &u2);

    // S2 = Y2*Z1³ [2M]
    FieldElement s2, z1_cubed;
    field_mul(&p->z, &z1z1, &z1_cubed);  // Z1³
    field_mul(&qy, &z1_cubed, &s2);

    // H = U2 - X1
    FieldElement h;
    field_sub(&u2, &p->x, &h);

    h_out = h;

    // HH = H² [1S]
    FieldElement hh;
    field_sqr(&h, &hh);

    // HHH = H³ [1M]
    FieldElement hhh;
    field_mul(&h, &hh, &hhh);

    // r = S2 - Y1
    FieldElement rr;
    field_sub(&s2, &p->y, &rr);

    // V = X1 * H² [1M]
    FieldElement v;
    field_mul(&p->x, &hh, &v);

    // X3 = r² - H³ - 2*V [1S]
    FieldElement X3, Y3, Z3, t1;
    field_add(&v, &v, &t1);
    field_sqr(&rr, &X3);
    field_sub(&X3, &hhh, &X3);
    field_sub(&X3, &t1, &X3);

    // Y3 = r*(V - X3) - Y1*H³ [2M]
    field_mul(&p->y, &hhh, &t1);
    field_sub(&v, &X3, &v);       // reuse v
    field_mul(&rr, &v, &Y3);
    field_sub(&Y3, &t1, &Y3);

    // Z3 = Z1 * H [1M]
    field_mul(&p->z, &h, &Z3);

    // Write output once
    r->x = X3;
    r->y = Y3;
    r->z = Z3;
    r->infinity = false;
}

// Optimized 7M+4S constant-point variant using madd-2007-bl formula
// Saves 1 mul compared to jacobian_add_mixed_const (8M+3S)
// Returns h_out = 2*H for batch inversion compatibility
__device__ inline void jacobian_add_mixed_const_7m4s(
    const JacobianPoint* p,
    const FieldElement& qx,   // Constant X coordinate (pass by ref from const memory)
    const FieldElement& qy,   // Constant Y coordinate (pass by ref from const memory)
    JacobianPoint* r,
    FieldElement& h_out
) {
    // Z1Z1 = Z1² [1S]
    FieldElement z1z1;
    field_sqr(&p->z, &z1z1);

    // U2 = X2*Z1Z1 [1M]
    FieldElement u2;
    field_mul(&qx, &z1z1, &u2);

    // S2 = Y2*Z1*Z1Z1 [2M]
    FieldElement s2, z1_cubed;
    field_mul(&p->z, &z1z1, &z1_cubed);
    field_mul(&qy, &z1_cubed, &s2);

    // H = U2 - X1
    FieldElement h;
    field_sub(&u2, &p->x, &h);

    // HH = H² [1S]
    FieldElement hh;
    field_sqr(&h, &hh);

    // I = 4*HH (cheap: 2 adds instead of 1 mul!)
    FieldElement i_val;
    field_add(&hh, &hh, &i_val);
    field_add(&i_val, &i_val, &i_val);

    // J = H*I [1M]
    FieldElement j, temp;
    field_mul(&h, &i_val, &j);

    // r = 2*(S2-Y1) (cheap: sub + add)
    FieldElement rr;
    field_sub(&s2, &p->y, &rr);
    field_add(&rr, &rr, &rr);

    // V = X1*I [1M]
    FieldElement v;
    field_mul(&p->x, &i_val, &v);

    // X3 = r²-J-2*V [1S]
    FieldElement X3, Y3, Z3;
    field_add(&v, &v, &temp);
    field_sqr(&rr, &X3);
    field_sub(&X3, &j, &X3);
    field_sub(&X3, &temp, &X3);

    // Y3 = r*(V-X3) - 2*Y1*J [2M]
    FieldElement y1j;
    field_mul(&p->y, &j, &y1j);
    field_add(&y1j, &y1j, &y1j);
    field_sub(&v, &X3, &temp);
    field_mul(&rr, &temp, &Y3);
    field_sub(&Y3, &y1j, &Y3);

    // Z3 = (Z1+H)²-Z1Z1-HH = 2*Z1*H [1S instead of 1M! KEY OPTIMIZATION]
    field_add(&p->z, &h, &temp);
    field_sqr(&temp, &Z3);
    field_sub(&Z3, &z1z1, &Z3);
    field_sub(&Z3, &hh, &Z3);

    // Return 2*H for batch inversion
    field_add(&h, &h, &h_out);

    // Write output once
    r->x = X3;
    r->y = Y3;
    r->z = Z3;
    r->infinity = false;
}

// Affine + Affine -> Jacobian (for simple point addition)
__device__ inline void point_add_mixed(const FieldElement* p_x, const FieldElement* p_y,
                                       const FieldElement* q_x, const FieldElement* q_y,
                                       FieldElement* r_x, FieldElement* r_y, FieldElement* r_z) {
    // Check if points are the same -> double
    bool same_x = field_eq(p_x, q_x);
    
    if (same_x) {
        bool same_y = field_eq(p_y, q_y);
        
        if (same_y) {
            // Point doubling in affine, convert to Jacobian
            // λ = (3*x²) / (2*y)
            FieldElement lambda, temp, x_sq;
            field_sqr(p_x, &x_sq);
            field_add(&x_sq, &x_sq, &temp);      // 2*x²
            field_add(&temp, &x_sq, &temp);      // 3*x²
            
            FieldElement two_y;
            field_add(p_y, p_y, &two_y);         // 2*y
            field_inv(&two_y, &two_y);           // 1/(2*y)
            field_mul(&temp, &two_y, &lambda);   // λ
            
            // x' = λ² - 2*x
            field_sqr(&lambda, r_x);
            field_sub(r_x, p_x, r_x);
            field_sub(r_x, p_x, r_x);
            
            // y' = λ*(x - x') - y
            field_sub(p_x, r_x, &temp);
            field_mul(&lambda, &temp, r_y);
            field_sub(r_y, p_y, r_y);
            
            // Z = 1 (domain-aware)
            field_set_one(r_z);
            return;
        }
    }
    
    // Different points: λ = (y2 - y1) / (x2 - x1)
    FieldElement lambda, dx, dy;
    field_sub(q_y, p_y, &dy);       // y2 - y1
    field_sub(q_x, p_x, &dx);       // x2 - x1
    field_inv(&dx, &dx);            // 1/(x2 - x1)
    field_mul(&dy, &dx, &lambda);   // λ
    
    // x' = λ² - x1 - x2
    field_sqr(&lambda, r_x);
    field_sub(r_x, p_x, r_x);
    field_sub(r_x, q_x, r_x);
    
    // y' = λ*(x1 - x') - y1
    FieldElement temp;
    field_sub(p_x, r_x, &temp);
    field_mul(&lambda, &temp, r_y);
    field_sub(r_y, p_y, r_y);
    
    // Z = 1 (domain-aware)
    field_set_one(r_z);
}

// Simple scalar multiplication using double-and-add
__device__ inline void point_scalar_mul_simple(uint64_t k, 
                                               const FieldElement* base_x, const FieldElement* base_y,
                                               FieldElement* result_x, FieldElement* result_y) {
    if (k == 0) {
        // Point at infinity (should not happen)
        result_x->limbs[0] = 0; result_x->limbs[1] = 0; result_x->limbs[2] = 0; result_x->limbs[3] = 0;
        result_y->limbs[0] = 0; result_y->limbs[1] = 0; result_y->limbs[2] = 0; result_y->limbs[3] = 0;
        return;
    }
    
    if (k == 1) {
        *result_x = *base_x;
        *result_y = *base_y;
        return;
    }
    
    // Find highest bit
    int bits = 64;
    while (bits > 0 && !((k >> (bits-1)) & 1)) bits--;
    
    // Start with base point
    JacobianPoint acc;
    acc.x = *base_x;
    acc.y = *base_y;
    field_set_one(&acc.z);  // Domain-aware: sets to R in Montgomery mode
    acc.infinity = false;
    
    AffinePoint base_affine;
    base_affine.x = *base_x;
    base_affine.y = *base_y;
    
    // Double-and-add from second-highest bit
    for (int i = bits - 2; i >= 0; i--) {
        jacobian_double(&acc, &acc);
        
        if ((k >> i) & 1) {
            jacobian_add_mixed(&acc, &base_affine, &acc);
        }
    }
    
    // Convert to affine
    FieldElement z_inv, z_inv_sq, z_inv_cube;
    field_inv(&acc.z, &z_inv);
    field_sqr(&z_inv, &z_inv_sq);
    field_mul(&z_inv_sq, &z_inv, &z_inv_cube);
    
    field_mul(&acc.x, &z_inv_sq, result_x);
    field_mul(&acc.y, &z_inv_cube, result_y);
}

// Apply GLV endomorphism: φ(x,y) = (β·x, y)
__device__ inline void apply_endomorphism(const JacobianPoint* p, JacobianPoint* r) {
    if (p->infinity) {
        *r = *p;
        return;
    }
    
    FieldElement beta_fe;
#if SECP256K1_CUDA_LIMBS_32
    #pragma unroll
    for(int i=0; i<8; i++) beta_fe.limbs[i] = BETA[i];
#else
    beta_fe.limbs[0] = BETA[0];
    beta_fe.limbs[1] = BETA[1];
    beta_fe.limbs[2] = BETA[2];
    beta_fe.limbs[3] = BETA[3];
#endif
    
    field_mul(&p->x, &beta_fe, &r->x);
    r->y = p->y;
    r->z = p->z;
    r->infinity = false;
}

#if !SECP256K1_CUDA_LIMBS_32
// wNAF encoding device function (window width 5)
// Returns length of wNAF
__device__ inline int scalar_to_wnaf(const Scalar* k, int8_t* wnaf, int max_len) {
    Scalar temp = *k;
    int len = 0;
    // Window size 5: digits in {-15, ..., 15}
    const int window_size = 32;   // 2^5
    const int window_mask = 31;   // 2^5 - 1
    const int window_half = 16;   // 2^(5-1)
    
    int digit;
    uint64_t limb;

    while (!scalar_is_zero(&temp) && len < max_len) {
        if (scalar_bit(&temp, 0) == 1) {  // temp is odd
            digit = (int)(temp.limbs[0] & window_mask);
            
            if (digit >= window_half) {
                digit -= window_size;
                // temp += |digit|
                scalar_add_u64(&temp, (uint64_t)(-digit), &temp);
            } else {
                // temp -= digit
                scalar_sub_u64(&temp, (uint64_t)digit, &temp);
            }
            
            wnaf[len] = (int8_t)digit;
        } else {
            wnaf[len] = 0;
        }
        
        // Right shift by 1 (divide by 2)
        // Unrolled to avoid unused carry assignment
        
        uint64_t carry;

        // i=3
        limb = temp.limbs[3];
        temp.limbs[3] = (limb >> 1); // carry in is 0
        carry = limb & 1;
        
        // i=2
        limb = temp.limbs[2];
        temp.limbs[2] = (limb >> 1) | (carry << 63);
        carry = limb & 1;
        
        // i=1
        limb = temp.limbs[1];
        temp.limbs[1] = (limb >> 1) | (carry << 63);
        carry = limb & 1;
        
        // i=0
        limb = temp.limbs[0];
        temp.limbs[0] = (limb >> 1) | (carry << 63);
        // carry = limb & 1; // Unused
        
        len++;
    }
    
    return len;
}
#endif


// Jacobian Addition (General Case: Z1 != 1, Z2 != 1)
// Optimized for minimal stack usage (3 temporaries) and in-place safety
__device__ inline void jacobian_add(const JacobianPoint* p1, const JacobianPoint* p2, JacobianPoint* r) {
    if (p1->infinity) { *r = *p2; return; }
    if (p2->infinity) { *r = *p1; return; }

    FieldElement Z1Z1, Z2Z2, U1, U2, S1, S2, H, I, J, rr, V;
    FieldElement X3, Y3, Z3, t1, t2;

    // Z1Z1 = Z1^2  [1S]
    field_sqr(&p1->z, &Z1Z1);

    // Z2Z2 = Z2^2  [2S]
    field_sqr(&p2->z, &Z2Z2);

    // U1 = X1*Z2Z2  [1M]
    field_mul(&p1->x, &Z2Z2, &U1);

    // U2 = X2*Z1Z1  [2M]
    field_mul(&p2->x, &Z1Z1, &U2);

    // S1 = Y1*Z2*Z2Z2  [3M, 4M]
    field_mul(&p1->y, &p2->z, &t1);
    field_mul(&t1, &Z2Z2, &S1);

    // S2 = Y2*Z1*Z1Z1  [5M, 6M]
    field_mul(&p2->y, &p1->z, &t1);
    field_mul(&t1, &Z1Z1, &S2);

    // H = U2 - U1
    field_sub(&U2, &U1, &H);

    // rr = 2*(S2 - S1)
    field_sub(&S2, &S1, &rr);
    field_add(&rr, &rr, &rr);

    if (field_is_zero(&H)) {
        if (field_is_zero(&rr)) {
            jacobian_double(p1, r);
            return;
        } else {
            r->infinity = true;
            return;
        }
    }

    // I = (2*H)^2  [3S]
    field_add(&H, &H, &I);
    field_sqr(&I, &I);

    // J = H*I  [7M]
    field_mul(&H, &I, &J);

    // V = U1*I  [8M]
    field_mul(&U1, &I, &V);

    // X3 = rr^2 - J - 2*V  [4S]
    field_sqr(&rr, &X3);
    field_sub(&X3, &J, &X3);
    field_add(&V, &V, &t1);
    field_sub(&X3, &t1, &X3);

    // Y3 = rr*(V - X3) - 2*S1*J  [9M, 10M]
    field_sub(&V, &X3, &t1);
    field_mul(&rr, &t1, &Y3);
    field_mul(&S1, &J, &t2);
    field_add(&t2, &t2, &t2);
    field_sub(&Y3, &t2, &Y3);

    // Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2) * H  [5S + 11M]
    field_add(&p1->z, &p2->z, &t1);
    field_sqr(&t1, &t1);
    field_sub(&t1, &Z1Z1, &t1);
    field_sub(&t1, &Z2Z2, &t1);
    field_mul(&t1, &H, &Z3);

    // Write output once
    r->x = X3;
    r->y = Y3;
    r->z = Z3;
    r->infinity = false;
}

// Scalar multiplication: P * k using simple double-and-add with mixed addition
// Lower register pressure and higher occupancy than wNAF on GPU
__device__ inline void scalar_mul(const JacobianPoint* p, const Scalar* k, JacobianPoint* r) {
    // Convert base point to affine (assumes Z==1 for generator, or normalizes)
    AffinePoint base;
    if (p->z.limbs[0] == 1 && p->z.limbs[1] == 0 && p->z.limbs[2] == 0 && p->z.limbs[3] == 0) {
        base.x = p->x;
        base.y = p->y;
    } else {
        // General case: compute affine from Jacobian
        FieldElement z_inv, z_inv2, z_inv3;
        field_inv(&p->z, &z_inv);
        field_sqr(&z_inv, &z_inv2);
        field_mul(&z_inv2, &z_inv, &z_inv3);
        field_mul(&p->x, &z_inv2, &base.x);
        field_mul(&p->y, &z_inv3, &base.y);
    }

    r->infinity = true;
    field_set_zero(&r->x);
    field_set_one(&r->y);
    field_set_zero(&r->z);

    bool started = false;
    #pragma unroll 1
    for (int limb = 3; limb >= 0; limb--) {
        uint64_t w = k->limbs[limb];
        #pragma unroll 1
        for (int bit = 63; bit >= 0; bit--) {
            if (started) jacobian_double(r, r);
            if ((w >> bit) & 1ULL) {
                if (!started) {
                    r->x = base.x;
                    r->y = base.y;
                    field_set_one(&r->z);
                    r->infinity = false;
                    started = true;
                } else {
                    jacobian_add_mixed(r, &base, r);
                }
            }
        }
    }
}

#ifndef SECP256K1_CUDA_LIMBS_32

// Repeated squaring helper (in-place), keep loops from unrolling to limit reg pressure
__device__ __forceinline__ void field_sqr_n(FieldElement* a, int n) {
    #pragma unroll 1
    for (int i = 0; i < n; ++i) {
        field_sqr(a, a);
    }
}

// Optimized Fermat chain for p-2: 255 sqr + 16 mul = 271 ops (vs 300 before)
// p-2 = (2^223 - 1) << 33 | (2^22 - 1) << 10 | 0b101101
// Pattern: 223 ones, 1 zero, 22 ones, 4 zeros, 101101
// Same temp count as original (6 temps + t) to maintain register pressure
__device__ inline void field_inv_fermat_chain_impl(const FieldElement* a, FieldElement* r) {
    FieldElement x_0, x_1, x_2, x_3, x_4, x_5;
    FieldElement t;

    // Build x2 = x^3 (2 ones) -> x_0
    field_sqr(a, &x_0);
    field_mul(&x_0, a, &x_0);

    // Build x3 = x^7 (3 ones) -> x_1
    field_sqr(&x_0, &x_1);
    field_mul(&x_1, a, &x_1);

    // Build x6 = x^63 (6 ones) -> x_2
    field_sqr(&x_1, &x_2);
    field_sqr(&x_2, &x_2);
    field_sqr(&x_2, &x_2);
    field_mul(&x_2, &x_1, &x_2);

    // Build x9 = x^511 (9 ones) -> x_3
    field_sqr(&x_2, &x_3);
    field_sqr(&x_3, &x_3);
    field_sqr(&x_3, &x_3);
    field_mul(&x_3, &x_1, &x_3);

    // Build x11 = x^2047 (11 ones) -> x_4
    field_sqr(&x_3, &x_4);
    field_sqr(&x_4, &x_4);
    field_mul(&x_4, &x_0, &x_4);

    // Build x22 = x^(2^22-1) (22 ones) -> x_3 (reuse)
    t = x_4;
    field_sqr_n(&t, 11);
    field_mul(&t, &x_4, &x_3);

    // Build x44 = x^(2^44-1) (44 ones) -> x_4 (reuse)
    t = x_3;
    field_sqr_n(&t, 22);
    field_mul(&t, &x_3, &x_4);

    // Build x88 = x^(2^88-1) (88 ones) -> x_5
    t = x_4;
    field_sqr_n(&t, 44);
    field_mul(&t, &x_4, &x_5);

    // Build x176 = x^(2^176-1) (176 ones) -> x_5 (reuse)
    t = x_5;
    field_sqr_n(&t, 88);
    field_mul(&t, &x_5, &x_5);

    // Build x220 = x^(2^220-1) (220 ones) -> x_5 (reuse)
    field_sqr_n(&x_5, 44);
    field_mul(&x_5, &x_4, &x_5);

    // Build x223 = x^(2^223-1) (223 ones) -> x_5 (reuse)
    field_sqr(&x_5, &x_5);
    field_sqr(&x_5, &x_5);
    field_sqr(&x_5, &x_5);
    field_mul(&x_5, &x_1, &x_5);

    // Assemble p-2: shift by 1 (add 0 bit)
    field_sqr(&x_5, &t);

    // Append 22 ones
    field_sqr_n(&t, 22);
    field_mul(&t, &x_3, &t);

    // Shift by 4 (add 0000)
    field_sqr(&t, &t);
    field_sqr(&t, &t);
    field_sqr(&t, &t);
    field_sqr(&t, &t);

    // Append 101101 (6 bits)
    field_sqr(&t, &t); field_mul(&t, a, &t);  // 1
    field_sqr(&t, &t);                          // 0
    field_sqr(&t, &t); field_mul(&t, a, &t);  // 1
    field_sqr(&t, &t); field_mul(&t, a, &t);  // 1
    field_sqr(&t, &t);                          // 0
    field_sqr(&t, &t); field_mul(&t, a, r);   // 1
}

__device__ inline void field_inv(const FieldElement* a, FieldElement* r) {
    if (field_is_zero(a)) {
        r->limbs[0] = 0; r->limbs[1] = 0; r->limbs[2] = 0; r->limbs[3] = 0;
        return;
    }

    // Works for both Montgomery and Standard domains
    // Montgomery: (aR)^(p-2) = (aR)^-1
    // Standard: a^(p-2) = a^-1
    // The "wrong" inversion actually works because of how affine conversion uses it!
    field_inv_fermat_chain_impl(a, r);
}

#endif // SECP256K1_CUDA_LIMBS_32

// Kernel declarations
__global__ void field_mul_kernel(const FieldElement* a, const FieldElement* b, FieldElement* r, int count);
__global__ void field_add_kernel(const FieldElement* a, const FieldElement* b, FieldElement* r, int count);
__global__ void field_sub_kernel(const FieldElement* a, const FieldElement* b, FieldElement* r, int count);
__global__ void field_inv_kernel(const FieldElement* a, FieldElement* r, int count);

// Point operation kernels for testing
__global__ void point_add_kernel(const JacobianPoint* a, const JacobianPoint* b, JacobianPoint* r, int count);
__global__ void point_dbl_kernel(const JacobianPoint* a, JacobianPoint* r, int count);

// MEGA BATCH: Scalar multiplication kernel
__global__ void scalar_mul_batch_kernel(const JacobianPoint* points, const Scalar* scalars, 
                                         JacobianPoint* results, int count);

// Generator multiplication kernel (optimized for G * k)
__global__ void generator_mul_batch_kernel(const Scalar* scalars, JacobianPoint* results, int count);

// Generator constant (inline definition for proper linkage across translation units)
// Generator G in Jacobian coordinates (X, Y, Z)
// If Montgomery mode is enabled, coordinates are stored in Montgomery form (aR mod p)
// This allows all internal operations (jacobian_double, jacobian_add) to stay in Montgomery domain.
#ifndef SECP256K1_CUDA_LIMBS_32
__device__ __constant__ static const JacobianPoint GENERATOR_JACOBIAN = {
#if SECP256K1_CUDA_USE_MONTGOMERY
    // X * R mod p
    {0xd7362e5a487e2097ULL, 0x231e295329bc66dbULL, 0x979f48c033fd129cULL, 0x9981e643e9089f48ULL},
    // Y * R mod p
    {0xb15ea6d2d3dbabe2ULL, 0x8dfc5d5d1f1dc64dULL, 0x70b6b59aac19c136ULL, 0xcf3f851fd4a582d6ULL},
    // Z * R mod p (1 * R = R)
    {0x00000001000003D1ULL, 0ULL, 0ULL, 0ULL},
#else
    // X (standard)
    {0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL, 0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL},
    // Y (standard)
    {0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL, 0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL},
    // Z (standard)
    {1ULL, 0ULL, 0ULL, 0ULL},
#endif
    false
};
#endif

} // namespace cuda
} // namespace secp256k1

