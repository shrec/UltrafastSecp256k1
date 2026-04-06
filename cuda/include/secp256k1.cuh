#pragma once
#include "gpu_compat.h"
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

// Force hybrid off for HIP/ROCm -- 32-bit Comba uses PTX inline asm
#if !SECP256K1_USE_PTX
#undef SECP256K1_CUDA_USE_HYBRID_MUL
#define SECP256K1_CUDA_USE_HYBRID_MUL 0
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

// Precomputed dummy point D = G + endo(G) for GLV+CT generator mul.
// endo(G) = (beta*Gx mod p, Gy).  D is used as the starting accumulator
// (amortizes the field_inv that would otherwise be needed per-call).
// Verified: D.y^2 == D.x^3 + 7 (mod p).  Computed offline via Python.
// D.x  = 0xC994B69768832BCBFF5E9AB39AE8D1D3763BBF1E531BED98FE51DE5EE84F50FB
// D.y  = 0xB7C52588D95C3B9AA25B0403F1EEF75702E84BB7597AABE663B82F6F04EF2777
// -D.y = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8 (= G.y)
__constant__ static const uint64_t DUMMY_GLV_X[4] = {
    0xFE51DE5EE84F50FBULL, 0x763BBF1E531BED98ULL,
    0xFF5E9AB39AE8D1D3ULL, 0xC994B69768832BCBULL
};
__constant__ static const uint64_t DUMMY_GLV_Y[4] = {
    0x63B82F6F04EF2777ULL, 0x02E84BB7597AABE6ULL,
    0xA25B0403F1EEF757ULL, 0xB7C52588D95C3B9AULL
};
// -D.y == G.y (coincidental property — verified above)
__constant__ static const uint64_t DUMMY_GLV_NEG_Y[4] = {
    0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL,
    0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL
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
#if defined(__SIZEOF_INT128__)
#define SECP256K1_CUDA_HAS_NATIVE_UINT128 1
#else
#define SECP256K1_CUDA_HAS_NATIVE_UINT128 0
#endif

__device__ __forceinline__ void mul64_portable(uint64_t a, uint64_t b, uint64_t& lo, uint64_t& hi) {
#if SECP256K1_CUDA_HAS_NATIVE_UINT128
    unsigned __int128 product = static_cast<unsigned __int128>(a) * b;
    lo = static_cast<uint64_t>(product);
    hi = static_cast<uint64_t>(product >> 64);
#else
    const uint64_t a_lo = static_cast<uint32_t>(a);
    const uint64_t a_hi = a >> 32;
    const uint64_t b_lo = static_cast<uint32_t>(b);
    const uint64_t b_hi = b >> 32;
    const uint64_t p0 = a_lo * b_lo;
    const uint64_t p1 = a_lo * b_hi;
    const uint64_t p2 = a_hi * b_lo;
    const uint64_t p3 = a_hi * b_hi;
    const uint64_t middle = (p0 >> 32) + static_cast<uint32_t>(p1) + static_cast<uint32_t>(p2);

    lo = (p0 & 0xFFFFFFFFULL) | (middle << 32);
    hi = p3 + (p1 >> 32) + (p2 >> 32) + (middle >> 32);
#endif
}

__device__ __forceinline__ uint64_t add_cc(uint64_t a, uint64_t b, uint64_t& carry) {
    const uint64_t sum = a + b;
    const uint64_t carry0 = (sum < a) ? 1ULL : 0ULL;
    const uint64_t result = sum + carry;
    const uint64_t carry1 = (result < sum) ? 1ULL : 0ULL;
    carry = carry0 | carry1;
    return result;
}

__device__ __forceinline__ uint64_t sub_cc(uint64_t a, uint64_t b, uint64_t& borrow) {
    const uint64_t diff = a - b;
    const uint64_t borrow0 = (a < b) ? 1ULL : 0ULL;
    const uint64_t result = diff - borrow;
    const uint64_t borrow1 = (diff < borrow) ? 1ULL : 0ULL;
    borrow = borrow0 | borrow1;
    return result;
}

__device__ __forceinline__ uint64_t muladd64(uint64_t a, uint64_t b, uint64_t add, uint64_t& carry) {
    uint64_t lo;
    uint64_t hi;
    uint64_t step_carry = 0;
    mul64_portable(a, b, lo, hi);
    lo = add_cc(lo, add, step_carry);
    hi += step_carry;
    step_carry = 0;
    lo = add_cc(lo, carry, step_carry);
    hi += step_carry;
    carry = hi;
    return lo;
}

// Forward decls used by Montgomery helpers.
__device__ inline void mul_256_512(const FieldElement* a, const FieldElement* b, uint64_t r[8]);
__device__ inline void sqr_256_512(const FieldElement* a, uint64_t r[8]);

// Montgomery reduction for 512-bit input t[0..7] (little-endian limbs).
// Computes t * R^{-1} mod p, where p = 2^256 - k, k = 0x1000003D1.
// We use n' = -p^{-1} mod 2^64. Since p0 = 2^64 - k, p0^{-1} = (-k)^{-1}.
// For k = 2^32 + 977, (-k)^{-1} mod 2^64 = -(0x1000003D1)^{-1}.
// Numerically, n' = 0xD838091DD2253531.
#if SECP256K1_USE_PTX
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
#else
// Portable mont_reduce_512 -- __int128 fallback for HIP/ROCm
__device__ __forceinline__ void mont_reduce_512(const uint64_t t_in[8], FieldElement* r) {
    uint64_t t0 = t_in[0], t1 = t_in[1], t2 = t_in[2], t3 = t_in[3];
    uint64_t t4 = t_in[4], t5 = t_in[5], t6 = t_in[6], t7 = t_in[7];

    constexpr uint64_t N0_INV = 0xD838091DD2253531ULL;

    uint64_t m, m_977_lo, m_977_hi, x0, x1, c_x;
    uint64_t top_carry = 0;

    // Iteration 0
    m = t0 * N0_INV;
    mul64_portable(m, 977, m_977_lo, m_977_hi);
    x0 = m_977_lo + (m << 32);
    c_x = (x0 < m_977_lo);
    x1 = m_977_hi + (m >> 32) + c_x;
    {
        uint64_t bw = 0;
        t1 = sub_cc(t1, x1, bw); t2 = sub_cc(t2, 0, bw); t3 = sub_cc(t3, 0, bw);
        t4 = sub_cc(t4, 0, bw); t5 = sub_cc(t5, 0, bw); t6 = sub_cc(t6, 0, bw);
        t7 = sub_cc(t7, 0, bw);
        top_carry -= bw;
    }
    {
        uint64_t ca = 0;
        t4 = add_cc(t4, m, ca); t5 = add_cc(t5, 0, ca);
        t6 = add_cc(t6, 0, ca); t7 = add_cc(t7, 0, ca);
        top_carry += ca;
    }

    // Iteration 1
    m = t1 * N0_INV;
    mul64_portable(m, 977, m_977_lo, m_977_hi);
    x0 = m_977_lo + (m << 32);
    c_x = (x0 < m_977_lo);
    x1 = m_977_hi + (m >> 32) + c_x;
    {
        uint64_t bw = 0;
        t2 = sub_cc(t2, x1, bw); t3 = sub_cc(t3, 0, bw); t4 = sub_cc(t4, 0, bw);
        t5 = sub_cc(t5, 0, bw); t6 = sub_cc(t6, 0, bw); t7 = sub_cc(t7, 0, bw);
        top_carry -= bw;
    }
    {
        uint64_t ca = 0;
        t5 = add_cc(t5, m, ca); t6 = add_cc(t6, 0, ca); t7 = add_cc(t7, 0, ca);
        top_carry += ca;
    }

    // Iteration 2
    m = t2 * N0_INV;
    mul64_portable(m, 977, m_977_lo, m_977_hi);
    x0 = m_977_lo + (m << 32);
    c_x = (x0 < m_977_lo);
    x1 = m_977_hi + (m >> 32) + c_x;
    {
        uint64_t bw = 0;
        t3 = sub_cc(t3, x1, bw); t4 = sub_cc(t4, 0, bw);
        t5 = sub_cc(t5, 0, bw); t6 = sub_cc(t6, 0, bw); t7 = sub_cc(t7, 0, bw);
        top_carry -= bw;
    }
    {
        uint64_t ca = 0;
        t6 = add_cc(t6, m, ca); t7 = add_cc(t7, 0, ca);
        top_carry += ca;
    }

    // Iteration 3
    m = t3 * N0_INV;
    mul64_portable(m, 977, m_977_lo, m_977_hi);
    x0 = m_977_lo + (m << 32);
    c_x = (x0 < m_977_lo);
    x1 = m_977_hi + (m >> 32) + c_x;
    {
        uint64_t bw = 0;
        t4 = sub_cc(t4, x1, bw); t5 = sub_cc(t5, 0, bw);
        t6 = sub_cc(t6, 0, bw); t7 = sub_cc(t7, 0, bw);
        top_carry -= bw;
    }
    {
        uint64_t ca = 0;
        t7 = add_cc(t7, m, ca);
        top_carry += ca;
    }

    // Final reduction
    uint64_t k0, k1, k2, k3, k_carry;
    k_carry = 0;
    k0 = add_cc(t4, K_MOD, k_carry);
    k1 = add_cc(t5, 0, k_carry);
    k2 = add_cc(t6, 0, k_carry);
    k3 = add_cc(t7, 0, k_carry);

    bool use_k = (top_carry != 0) || (k_carry != 0);

    if (use_k) {
        r->limbs[0] = k0; r->limbs[1] = k1; r->limbs[2] = k2; r->limbs[3] = k3;
    } else {
        r->limbs[0] = t4; r->limbs[1] = t5; r->limbs[2] = t6; r->limbs[3] = t7;
    }
}
#endif // SECP256K1_USE_PTX (mont_reduce_512)

// Forward declarations for Montgomery conversion functions (defined after hybrid include)
__device__ __forceinline__ void field_to_mont(const FieldElement* a, FieldElement* r);
__device__ __forceinline__ void field_from_mont(const FieldElement* a, FieldElement* r);
__device__ __forceinline__ void field_mul_mont(const FieldElement* a, const FieldElement* b, FieldElement* r);
__device__ __forceinline__ void field_sqr_mont(const FieldElement* a, FieldElement* r);

#if SECP256K1_USE_PTX
__device__ __forceinline__ void mul64(uint64_t a, uint64_t b, uint64_t& lo, uint64_t& hi) {
    asm volatile(
        "mul.lo.u64 %0, %2, %3; \n\t"
        "mul.hi.u64 %1, %2, %3; \n\t"
        : "=l"(lo), "=l"(hi)
        : "l"(a), "l"(b)
    );
}
#else
// Portable wide-multiply fallback for HIP/ROCm and MSVC host parsing.
__device__ __forceinline__ void mul64(uint64_t a, uint64_t b, uint64_t& lo, uint64_t& hi) {
    mul64_portable(a, b, lo, hi);
}
#endif

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
// Extended scalar arithmetic (mod ORDER)
// ============================================================================

// Barrett constant: mu = floor(2^512 / ORDER), 5 limbs (LE)
__constant__ static const uint64_t BARRETT_MU[5] = {
    0x402DA1732FC9BEC0ULL,
    0x4551231950B75FC4ULL,
    0x0000000000000001ULL,
    0x0000000000000000ULL,
    0x0000000000000001ULL
};

// n - 2 (for Fermat inversion), LE limbs
__constant__ static const uint64_t ORDER_MINUS_2[4] = {
    0xBFD25E8CD036413FULL,
    0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL,
    0xFFFFFFFFFFFFFFFFULL
};

// Scalar negation: r = -a mod n (branchless, aliasing-safe: a == r OK)
__device__ inline void scalar_negate(const Scalar* a, Scalar* r) {
    // Read zero-mask BEFORE writing to r (a may alias r)
    uint64_t nz = a->limbs[0] | a->limbs[1] | a->limbs[2] | a->limbs[3];
    uint64_t mask = -(uint64_t)(nz != 0);
    uint64_t borrow = 0;
    r->limbs[0] = sub_cc(ORDER[0], a->limbs[0], borrow) & mask;
    r->limbs[1] = sub_cc(ORDER[1], a->limbs[1], borrow) & mask;
    r->limbs[2] = sub_cc(ORDER[2], a->limbs[2], borrow) & mask;
    r->limbs[3] = sub_cc(ORDER[3], a->limbs[3], borrow) & mask;
}

// Scalar parity check
__device__ __forceinline__ bool scalar_is_even(const Scalar* s) {
    return (s->limbs[0] & 1) == 0;
}

// Scalar equality
__device__ __forceinline__ bool scalar_eq(const Scalar* a, const Scalar* b) {
    return (a->limbs[0] == b->limbs[0]) &&
           (a->limbs[1] == b->limbs[1]) &&
           (a->limbs[2] == b->limbs[2]) &&
           (a->limbs[3] == b->limbs[3]);
}

// Scalar multiplication mod ORDER: r = a * b (mod n)
// Schoolbook 4x4 -> 8-limb product + Barrett reduction
__device__ inline void scalar_mul_mod_n(const Scalar* a, const Scalar* b, Scalar* r) {
    uint64_t prod[8] = {};

    // Schoolbook 4x4 multiplication
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            uint64_t lo, hi;
            mul64(a->limbs[i], b->limbs[j], lo, hi);
            uint64_t c1 = 0;
            uint64_t tmp = add_cc(lo, carry, c1);
            uint64_t c2 = 0;
            prod[i + j] = add_cc(prod[i + j], tmp, c2);
            carry = hi + c1 + c2;
        }
        prod[i + 4] = carry;
    }

    // Barrett reduction: q = prod[4..7], q_approx = floor(q * mu / 2^256)
    uint64_t qmu[9] = {};
    for (int i = 0; i < 4; i++) {
        uint64_t carry_mu = 0;
        for (int j = 0; j < 5; j++) {
            uint64_t lo, hi;
            mul64(prod[4 + i], BARRETT_MU[j], lo, hi);
            uint64_t c1 = 0;
            uint64_t tmp = add_cc(lo, carry_mu, c1);
            uint64_t c2 = 0;
            qmu[i + j] = add_cc(qmu[i + j], tmp, c2);
            carry_mu = hi + c1 + c2;
        }
        if (i + 5 < 9) qmu[i + 5] = carry_mu;
    }

    // q_approx = qmu[4..7]
    // Compute q_approx * ORDER (only low 5 limbs needed)
    uint64_t qn[5] = {};
    for (int i = 0; i < 4; i++) {
        uint64_t carry_qn = 0;
        for (int j = 0; j < 4; j++) {
            if (i + j >= 5) break;
            uint64_t lo, hi;
            mul64(qmu[4 + i], ORDER[j], lo, hi);
            uint64_t c1 = 0;
            uint64_t tmp = add_cc(lo, carry_qn, c1);
            uint64_t c2 = 0;
            qn[i + j] = add_cc(qn[i + j], tmp, c2);
            carry_qn = hi + c1 + c2;
        }
        if (i + 4 < 5) qn[i + 4] = carry_qn;
    }

    // r = prod[0..3] - qn[0..3]
    uint64_t borrow = 0;
    r->limbs[0] = sub_cc(prod[0], qn[0], borrow);
    r->limbs[1] = sub_cc(prod[1], qn[1], borrow);
    r->limbs[2] = sub_cc(prod[2], qn[2], borrow);
    r->limbs[3] = sub_cc(prod[3], qn[3], borrow);
    uint64_t r4 = prod[4] - qn[4] - borrow;

    // At most 2 conditional subtracts to bring into [0, ORDER)
    if (r4 > 0 || scalar_ge(r, ORDER)) {
        borrow = 0;
        r->limbs[0] = sub_cc(r->limbs[0], ORDER[0], borrow);
        r->limbs[1] = sub_cc(r->limbs[1], ORDER[1], borrow);
        r->limbs[2] = sub_cc(r->limbs[2], ORDER[2], borrow);
        r->limbs[3] = sub_cc(r->limbs[3], ORDER[3], borrow);
        r4 -= borrow;
    }
    if (r4 > 0 || scalar_ge(r, ORDER)) {
        borrow = 0;
        r->limbs[0] = sub_cc(r->limbs[0], ORDER[0], borrow);
        r->limbs[1] = sub_cc(r->limbs[1], ORDER[1], borrow);
        r->limbs[2] = sub_cc(r->limbs[2], ORDER[2], borrow);
        r->limbs[3] = sub_cc(r->limbs[3], ORDER[3], borrow);
    }
}

// Scalar squaring mod ORDER: r = a^2 (mod n)
__device__ inline void scalar_sqr_mod_n(const Scalar* a, Scalar* r) {
    scalar_mul_mod_n(a, a, r);
}

// Scalar inverse: r = a^(n-2) mod n (Fermat's little theorem)
// Square-and-multiply, MSB to LSB
// Scalar inverse via Fermat's little theorem: a^(n-2) mod n
// In-place aliasing safe (a == r OK via base copy)
__device__ inline void scalar_inverse(const Scalar* a, Scalar* r) {
    if (scalar_is_zero(a)) {
        r->limbs[0] = r->limbs[1] = r->limbs[2] = r->limbs[3] = 0;
        return;
    }

    Scalar base = *a;
    Scalar result;
    result.limbs[0] = 1; result.limbs[1] = 0;
    result.limbs[2] = 0; result.limbs[3] = 0;

    for (int i = 255; i >= 0; --i) {
        scalar_sqr_mod_n(&result, &result);

        int limb_idx = i / 64;
        int bit_idx = i % 64;
        if ((ORDER_MINUS_2[limb_idx] >> bit_idx) & 1)
            scalar_mul_mod_n(&result, &base, &result);
    }
    *r = result;
}

// GLV decomposition constants (LE 64-bit limbs, matching libsecp256k1)
// g1, g2: multipliers for c1 = round(k * g1 / 2^384), c2 = round(k * g2 / 2^384)
__constant__ static const uint64_t GLV_G1[4] = {
    0xE893209A45DBB031ULL, 0x3DAA8A1471E8CA7FULL,
    0xE86C90E49284EB15ULL, 0x3086D221A7D46BCDULL
};
__constant__ static const uint64_t GLV_G2[4] = {
    0x1571B4AE8AC47F71ULL, 0x221208AC9DF506C6ULL,
    0x6F547FA90ABFE4C4ULL, 0xE4437ED6010E8828ULL
};
// -b1, -b2 as 256-bit values (LE limbs): used in k2 = c1*(-b1) + c2*(-b2)
__constant__ static const uint64_t GLV_MINUS_B1[4] = {
    0x6F547FA90ABFE4C3ULL, 0xE4437ED6010E8828ULL, 0x0ULL, 0x0ULL
};
__constant__ static const uint64_t GLV_MINUS_B2[4] = {
    0xD765CDA83DB1562CULL, 0x8A280AC50774346DULL,
    0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
};

// Compute (a * b) >> 384 with rounding bit (for GLV decomposition)
// a, b: 256-bit values as LE uint64_t[4]
// Returns upper ~128 bits as LE uint64_t[4]
#if SECP256K1_USE_PTX
__device__ inline void mul_shift_384(const uint64_t a[4], const uint64_t b[4], uint64_t result[4]) {
    // 32-bit Comba: avoids INT64 multiply (64x throughput gain on consumer GPUs)
    uint32_t al[8], bl[8];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        al[2*i]   = (uint32_t)a[i];
        al[2*i+1] = (uint32_t)(a[i] >> 32);
        bl[2*i]   = (uint32_t)b[i];
        bl[2*i+1] = (uint32_t)(b[i] >> 32);
    }
    uint32_t c0 = 0, c1 = 0, c2 = 0;

    // 8x8 Comba: 64 mul.lo/hi.u32 pairs across 16 columns
    // GLV_MAC: multiply-accumulate one 32x32->64 product into {c0,c1,c2}
#define GLV_MAC(i, j) do { \
    uint32_t _lo, _hi; \
    asm volatile("mul.lo.u32 %0, %2, %3;\n\t" \
                 "mul.hi.u32 %1, %2, %3;\n\t" \
                 : "=r"(_lo), "=r"(_hi) : "r"(al[i]), "r"(bl[j])); \
    asm volatile("add.cc.u32 %0, %0, %3;\n\t" \
                 "addc.cc.u32 %1, %1, %4;\n\t" \
                 "addc.u32 %2, %2, 0;\n\t" \
                 : "+r"(c0), "+r"(c1), "+r"(c2) : "r"(_lo), "r"(_hi)); \
} while(0)
#define GLV_EXT(out) do { out = c0; c0 = c1; c1 = c2; c2 = 0; } while(0)

    uint32_t d = 0; // discard for columns 0-10
    /* col  0 */ GLV_MAC(0,0); GLV_EXT(d);
    /* col  1 */ GLV_MAC(0,1); GLV_MAC(1,0); GLV_EXT(d);
    /* col  2 */ GLV_MAC(0,2); GLV_MAC(1,1); GLV_MAC(2,0); GLV_EXT(d);
    /* col  3 */ GLV_MAC(0,3); GLV_MAC(1,2); GLV_MAC(2,1); GLV_MAC(3,0); GLV_EXT(d);
    /* col  4 */ GLV_MAC(0,4); GLV_MAC(1,3); GLV_MAC(2,2); GLV_MAC(3,1); GLV_MAC(4,0); GLV_EXT(d);
    /* col  5 */ GLV_MAC(0,5); GLV_MAC(1,4); GLV_MAC(2,3); GLV_MAC(3,2); GLV_MAC(4,1); GLV_MAC(5,0); GLV_EXT(d);
    /* col  6 */ GLV_MAC(0,6); GLV_MAC(1,5); GLV_MAC(2,4); GLV_MAC(3,3); GLV_MAC(4,2); GLV_MAC(5,1); GLV_MAC(6,0); GLV_EXT(d);
    /* col  7 */ GLV_MAC(0,7); GLV_MAC(1,6); GLV_MAC(2,5); GLV_MAC(3,4); GLV_MAC(4,3); GLV_MAC(5,2); GLV_MAC(6,1); GLV_MAC(7,0); GLV_EXT(d);
    /* col  8 */ GLV_MAC(1,7); GLV_MAC(2,6); GLV_MAC(3,5); GLV_MAC(4,4); GLV_MAC(5,3); GLV_MAC(6,2); GLV_MAC(7,1); GLV_EXT(d);
    /* col  9 */ GLV_MAC(2,7); GLV_MAC(3,6); GLV_MAC(4,5); GLV_MAC(5,4); GLV_MAC(6,3); GLV_MAC(7,2); GLV_EXT(d);
    /* col 10 */ GLV_MAC(3,7); GLV_MAC(4,6); GLV_MAC(5,5); GLV_MAC(6,4); GLV_MAC(7,3); GLV_EXT(d);
    (void)d;
    // Column 11: MSB is the rounding bit (bit 383)
    uint32_t p11;
    /* col 11 */ GLV_MAC(4,7); GLV_MAC(5,6); GLV_MAC(6,5); GLV_MAC(7,4); GLV_EXT(p11);
    // Columns 12-15: result bits [384..511]
    uint32_t p12, p13, p14, p15;
    /* col 12 */ GLV_MAC(5,7); GLV_MAC(6,6); GLV_MAC(7,5); GLV_EXT(p12);
    /* col 13 */ GLV_MAC(6,7); GLV_MAC(7,6); GLV_EXT(p13);
    /* col 14 */ GLV_MAC(7,7); GLV_EXT(p14);
    p15 = c0;
#undef GLV_MAC
#undef GLV_EXT

    result[0] = ((uint64_t)p13 << 32) | p12;
    result[1] = ((uint64_t)p15 << 32) | p14;
    result[2] = 0;
    result[3] = 0;
    // Rounding: bit 383 = bit 31 of p11
    if (p11 >> 31) {
        result[0]++;
        if (result[0] == 0) result[1]++;
    }
}
#else
__device__ inline void mul_shift_384(const uint64_t a[4], const uint64_t b[4], uint64_t result[4]) {
    // Portable 64-bit version for HIP/ROCm
    uint64_t prod[8] = {};
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            uint64_t lo, hi;
            mul64(a[i], b[j], lo, hi);
            uint64_t c1 = 0;
            uint64_t tmp = add_cc(lo, carry, c1);
            uint64_t c2 = 0;
            prod[i + j] = add_cc(prod[i + j], tmp, c2);
            carry = hi + c1 + c2;
        }
        prod[i + 4] = carry;
    }
    result[0] = prod[6];
    result[1] = prod[7];
    result[2] = 0;
    result[3] = 0;
    if (prod[5] >> 63) {
        result[0]++;
        if (result[0] == 0) result[1]++;
    }
}
#endif

// Bit-length of a scalar (for GLV sign selection)
__device__ inline int scalar_bitlen(const Scalar* s) {
    for (int i = 3; i >= 0; --i) {
        if (s->limbs[i] != 0) {
            // Count leading zeros
            uint64_t v = s->limbs[i];
            int bits = 0;
            uint32_t hi32 = (uint32_t)(v >> 32);
            if (hi32) {
                bits = 64 - __clz(hi32);
            } else {
                bits = 32 - __clz((uint32_t)v);
            }
            return i * 64 + bits;
        }
    }
    return 0;
}

// GLV decomposition result
struct GLVDecomposition {
    Scalar k1;
    Scalar k2;
    bool k1_neg;
    bool k2_neg;
};

// Decompose scalar k into k1, k2 such that k = k1 + k2*lambda (mod n)
// The resulting k1, k2 are roughly half the bit length (~128 bits each)
__device__ inline GLVDecomposition glv_decompose(const Scalar* k) {
    GLVDecomposition result;

    // Step 1: c1 = round(k * g1 / 2^384), c2 = round(k * g2 / 2^384)
    uint64_t c1_limbs[4], c2_limbs[4];
    mul_shift_384(k->limbs, GLV_G1, c1_limbs);
    mul_shift_384(k->limbs, GLV_G2, c2_limbs);

    Scalar c1, c2;
    for (int i = 0; i < 4; i++) { c1.limbs[i] = c1_limbs[i]; c2.limbs[i] = c2_limbs[i]; }
    // Normalize in case >= ORDER
    if (scalar_ge(&c1, ORDER)) scalar_sub(&c1, (const Scalar*)ORDER, &c1);
    if (scalar_ge(&c2, ORDER)) scalar_sub(&c2, (const Scalar*)ORDER, &c2);

    // Step 2: k2 = c1*(-b1) + c2*(-b2) (mod n)
    Scalar minus_b1, minus_b2;
    for (int i = 0; i < 4; i++) { minus_b1.limbs[i] = GLV_MINUS_B1[i]; minus_b2.limbs[i] = GLV_MINUS_B2[i]; }

    Scalar t1, t2, k2_mod;
    scalar_mul_mod_n(&c1, &minus_b1, &t1);
    scalar_mul_mod_n(&c2, &minus_b2, &t2);
    scalar_add(&t1, &t2, &k2_mod);

    // Step 3: pick shorter representation for k2
    Scalar k2_neg_val;
    scalar_negate(&k2_mod, &k2_neg_val);
    bool k2_is_neg = (scalar_bitlen(&k2_neg_val) < scalar_bitlen(&k2_mod));
    Scalar k2_abs = k2_is_neg ? k2_neg_val : k2_mod;

    // For k2_signed: if k2_is_neg, k2_signed = -k2_abs = k2_mod; else k2_signed = k2_abs = k2_mod
    // We need k2_signed for computing k1 = k - lambda*k2_signed
    Scalar k2_signed;
    if (k2_is_neg) {
        scalar_negate(&k2_abs, &k2_signed);
    } else {
        k2_signed = k2_abs;
    }

    // Step 4: k1 = k - lambda*k2_signed (mod n)
    Scalar lambda_s;
    for (int i = 0; i < 4; i++) lambda_s.limbs[i] = LAMBDA[i];
    Scalar lk2;
    scalar_mul_mod_n(&lambda_s, &k2_signed, &lk2);
    Scalar k1_mod;
    scalar_sub(k, &lk2, &k1_mod);

    // Step 5: pick shorter representation for k1
    Scalar k1_neg_val;
    scalar_negate(&k1_mod, &k1_neg_val);
    bool k1_is_neg = (scalar_bitlen(&k1_neg_val) < scalar_bitlen(&k1_mod));
    Scalar k1_abs = k1_is_neg ? k1_neg_val : k1_mod;

    result.k1 = k1_abs;
    result.k2 = k2_abs;
    result.k1_neg = k1_is_neg;
    result.k2_neg = k2_is_neg;

    return result;
}

// ============================================================================
// Standard 64-bit field operations (used for add/sub - faster for these!)
// ============================================================================

// Field Addition
#if SECP256K1_USE_PTX
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
#else
// Portable field_add for HIP/ROCm
__device__ __forceinline__ void field_add(const FieldElement* a, const FieldElement* b, FieldElement* r) {
    uint64_t carry = 0;
    uint64_t r0 = add_cc(a->limbs[0], b->limbs[0], carry);
    uint64_t r1 = add_cc(a->limbs[1], b->limbs[1], carry);
    uint64_t r2 = add_cc(a->limbs[2], b->limbs[2], carry);
    uint64_t r3 = add_cc(a->limbs[3], b->limbs[3], carry);

    uint64_t borrow = 0;
    uint64_t t0 = sub_cc(r0, MODULUS[0], borrow);
    uint64_t t1 = sub_cc(r1, MODULUS[1], borrow);
    uint64_t t2 = sub_cc(r2, MODULUS[2], borrow);
    uint64_t t3 = sub_cc(r3, MODULUS[3], borrow);

    if (carry || borrow == 0) {
        r->limbs[0] = t0; r->limbs[1] = t1; r->limbs[2] = t2; r->limbs[3] = t3;
    } else {
        r->limbs[0] = r0; r->limbs[1] = r1; r->limbs[2] = r2; r->limbs[3] = r3;
    }
}
#endif

// Field Subtraction
#if SECP256K1_USE_PTX
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
#else
// Portable field_sub for HIP/ROCm
__device__ __forceinline__ void field_sub(const FieldElement* a, const FieldElement* b, FieldElement* r) {
    uint64_t borrow = 0;
    uint64_t r0 = sub_cc(a->limbs[0], b->limbs[0], borrow);
    uint64_t r1 = sub_cc(a->limbs[1], b->limbs[1], borrow);
    uint64_t r2 = sub_cc(a->limbs[2], b->limbs[2], borrow);
    uint64_t r3 = sub_cc(a->limbs[3], b->limbs[3], borrow);

    if (borrow) {
        uint64_t carry = 0;
        r->limbs[0] = add_cc(r0, MODULUS[0], carry);
        r->limbs[1] = add_cc(r1, MODULUS[1], carry);
        r->limbs[2] = add_cc(r2, MODULUS[2], carry);
        r->limbs[3] = add_cc(r3, MODULUS[3], carry);
    } else {
        r->limbs[0] = r0; r->limbs[1] = r1; r->limbs[2] = r2; r->limbs[3] = r3;
    }
}
#endif

// Field Negation: r = -a (mod P)
#if SECP256K1_USE_PTX
__device__ inline void field_negate(const FieldElement* a, FieldElement* r) {
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
#else
// Portable field_negate for HIP/ROCm
__device__ inline void field_negate(const FieldElement* a, FieldElement* r) {
    uint64_t borrow = 0;
    r->limbs[0] = sub_cc(MODULUS[0], a->limbs[0], borrow);
    r->limbs[1] = sub_cc(MODULUS[1], a->limbs[1], borrow);
    r->limbs[2] = sub_cc(MODULUS[2], a->limbs[2], borrow);
    r->limbs[3] = sub_cc(MODULUS[3], a->limbs[3], borrow);
}
#endif

// Field multiplication by small constant: r = a * small (mod P)
// Optimized for small constants (e.g., 7, 28 for secp256k1)
__device__ inline void field_mul_small(const FieldElement* a, uint32_t small, FieldElement* r) {
    // Simple approach: multiply and reduce
    // For small constants, this is faster than full field_mul
    uint64_t carry = 0;
    uint64_t tmp[4];
    
    for (int i = 0; i < 4; i++) {
        tmp[i] = muladd64(a->limbs[i], static_cast<uint64_t>(small), 0, carry);
    }
    
    // Now we have a 320-bit number: tmp[0..3] + carry * 2^256
    // Reduce carry * 2^256 mod P
    // Since P = 2^256 - 0x1000003d1, we have 2^256 == 0x1000003d1 (mod P)
    // So carry * 2^256 == carry * 0x1000003d1
    
    uint64_t c = carry;
    if (c > 0) {
        // carry * 0x1000003d1 = carry * (2^32 + 0x3d1) = carry << 32 + carry * 977
        uint64_t reduction_hi = 0;
        const uint64_t reduction_lo = muladd64(c, 0x1000003d1ULL, 0, reduction_hi);
        uint64_t sum_carry = 0;

        r->limbs[0] = add_cc(tmp[0], reduction_lo, sum_carry);
        r->limbs[1] = add_cc(tmp[1], reduction_hi, sum_carry);
        r->limbs[2] = add_cc(tmp[2], 0, sum_carry);
        r->limbs[3] = add_cc(tmp[3], 0, sum_carry);

        if (sum_carry) {
            uint64_t final_hi = 0;
            const uint64_t final_lo = muladd64(sum_carry, 0x1000003d1ULL, 0, final_hi);
            sum_carry = 0;
            r->limbs[0] = add_cc(r->limbs[0], final_lo, sum_carry);
            r->limbs[1] = add_cc(r->limbs[1], final_hi, sum_carry);
            r->limbs[2] = add_cc(r->limbs[2], 0, sum_carry);
            r->limbs[3] = add_cc(r->limbs[3], 0, sum_carry);
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

// 512->256 reduction: T mod P where P = 2^256 - K_MOD
#if SECP256K1_USE_PTX
__device__ __forceinline__ void reduce_512_to_256(uint64_t t[8], FieldElement* r) {
    // P = 2^256 - K_MOD, where K_MOD = 2^32 + 977 = 0x1000003D1
    // T = T_hi * 2^256 + T_lo == T_hi * K_MOD + T_lo (mod P)
    //
    // OPTIMIZATION: Multiply T_hi by K_MOD directly in one MAD chain,
    // instead of splitting into T_hi*977 + T_hi<<32 (two separate passes).
    // Saves ~26 instructions and 7 registers per call.
    
    uint64_t t0 = t[0], t1 = t[1], t2 = t[2], t3 = t[3];
    uint64_t t4 = t[4], t5 = t[5], t6 = t[6], t7 = t[7];
    
    // 1. Compute A = T_hi * K_MOD (5 limbs: a0..a4)
    //    Single MAD chain -- replaces separate *977 + <<32 two-pass approach
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
    
    // 3. Reduce overflow: extra = a4 + carry (<= 2^33 + 1)
    //    extra * K_MOD fits in 2 limbs (<= 2^66)
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
    
    // 4. Rare carry overflow (probability ~= 2^{-190})
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
#else
// Portable reduce_512_to_256 for HIP/ROCm -- uses __int128 instead of PTX
__device__ __forceinline__ void reduce_512_to_256(uint64_t t[8], FieldElement* r) {
    uint64_t t0 = t[0], t1 = t[1], t2 = t[2], t3 = t[3];
    uint64_t t4 = t[4], t5 = t[5], t6 = t[6], t7 = t[7];

    // 1. Compute A = T_hi * K_MOD using __int128
    uint64_t carry_p = 0;
    const uint64_t a0 = muladd64(t4, K_MOD, 0, carry_p);
    const uint64_t a1 = muladd64(t5, K_MOD, 0, carry_p);
    const uint64_t a2 = muladd64(t6, K_MOD, 0, carry_p);
    const uint64_t a3 = muladd64(t7, K_MOD, 0, carry_p);
    const uint64_t a4 = carry_p;

    // 2. Add A[0..3] to T_lo
    uint64_t carry = 0;
    t0 = add_cc(t0, a0, carry);
    t1 = add_cc(t1, a1, carry);
    t2 = add_cc(t2, a2, carry);
    t3 = add_cc(t3, a3, carry);

    // 3. Reduce overflow: extra = a4 + carry
    uint64_t extra = a4 + carry;
    uint64_t ek_hi = 0;
    const uint64_t ek_lo = muladd64(extra, K_MOD, 0, ek_hi);

    carry = 0;
    t0 = add_cc(t0, ek_lo, carry);
    t1 = add_cc(t1, ek_hi, carry);
    t2 = add_cc(t2, 0, carry);
    t3 = add_cc(t3, 0, carry);
    uint64_t c = carry;

    // 4. Rare carry overflow
    if (c) {
        carry = 0;
        t0 = add_cc(t0, K_MOD, carry);
        t1 = add_cc(t1, 0, carry);
        t2 = add_cc(t2, 0, carry);
        t3 = add_cc(t3, 0, carry);
    }

    // 5. Conditional subtraction of P
    uint64_t borrow = 0;
    uint64_t r0 = sub_cc(t0, MODULUS[0], borrow);
    uint64_t r1 = sub_cc(t1, MODULUS[1], borrow);
    uint64_t r2 = sub_cc(t2, MODULUS[2], borrow);
    uint64_t r3 = sub_cc(t3, MODULUS[3], borrow);

    if (borrow == 0) {
        r->limbs[0] = r0; r->limbs[1] = r1; r->limbs[2] = r2; r->limbs[3] = r3;
    } else {
        r->limbs[0] = t0; r->limbs[1] = t1; r->limbs[2] = t2; r->limbs[3] = t3;
    }
}
#endif // SECP256K1_USE_PTX (reduce_512_to_256)

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

// Unchecked doubling: skips infinity and Y==0 checks.
// Precondition: p is a valid, non-infinity point with Y != 0.
__device__ inline void jacobian_double_unchecked(const JacobianPoint* p, JacobianPoint* r) {
    FieldElement S, M, X3, Y3, Z3, YY, YYYY, t1;

    field_sqr(&p->y, &YY);

    field_mul(&p->x, &YY, &S);
    field_add(&S, &S, &S);
    field_add(&S, &S, &S);

    field_sqr(&p->x, &M);
    field_add(&M, &M, &t1);
    field_add(&M, &t1, &M);

    field_sqr(&M, &X3);
    field_add(&S, &S, &t1);
    field_sub(&X3, &t1, &X3);

    field_sqr(&YY, &YYYY);

    field_add(&YYYY, &YYYY, &t1);
    field_add(&t1, &t1, &t1);
    field_add(&t1, &t1, &t1);
    field_sub(&S, &X3, &S);
    field_mul(&M, &S, &Y3);
    field_sub(&Y3, &t1, &Y3);

    field_mul(&p->y, &p->z, &Z3);
    field_add(&Z3, &Z3, &Z3);

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

    // Z1^2 [1S]
    field_sqr(&p->z, &z1z1);
    
    // U2 = X2*Z1^2 [1M]
    field_mul(&q->x, &z1z1, &u2);
    
    // S2 = Y2*Z1^3 [2M, 3M]
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
    
    // HH = H^2 [2S]
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
    
    // X3 = rr^2 - J - 2*V [3S]
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
    
    // Z3 = (Z1+H)^2 - Z1^2 - HH [4S]
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

// Unchecked mixed addition: skips p->infinity check.
// Precondition: p is a valid, non-infinity Jacobian point.
// Keeps the h==0 (P==Q) check for algebraic completeness.
__device__ inline void jacobian_add_mixed_unchecked(const JacobianPoint* p, const AffinePoint* q, JacobianPoint* r) {
    FieldElement z1z1, u2, s2, h, hh, i, j, rr, v;
    FieldElement X3, Y3, Z3, t1, t2;

    field_sqr(&p->z, &z1z1);
    field_mul(&q->x, &z1z1, &u2);

    field_mul(&p->z, &z1z1, &t1);
    field_mul(&q->y, &t1, &s2);

    field_sub(&u2, &p->x, &h);

    if (field_is_zero(&h)) {
        field_sub(&s2, &p->y, &t1);
        if (field_is_zero(&t1)) {
            jacobian_double_unchecked(p, r);
            return;
        }
        r->infinity = true;
        return;
    }

    field_sqr(&h, &hh);
    field_add(&hh, &hh, &i);
    field_add(&i, &i, &i);
    field_mul(&h, &i, &j);

    field_sub(&s2, &p->y, &t1);
    field_add(&t1, &t1, &rr);

    field_mul(&p->x, &i, &v);

    field_sqr(&rr, &X3);
    field_sub(&X3, &j, &X3);
    field_add(&v, &v, &t1);
    field_sub(&X3, &t1, &X3);

    field_sub(&v, &X3, &t1);
    field_mul(&rr, &t1, &Y3);
    field_mul(&p->y, &j, &t2);
    field_add(&t2, &t2, &t2);
    field_sub(&Y3, &t2, &Y3);

    field_add(&p->z, &h, &t1);
    field_sqr(&t1, &Z3);
    field_sub(&Z3, &z1z1, &Z3);
    field_sub(&Z3, &hh, &Z3);

    r->x = X3;
    r->y = Y3;
    r->z = Z3;
    r->infinity = false;  // must be explicit: unchecked variant doesn't guarantee this
}

// ---------------------------------------------------------------------------
// Constant-time conditional move: dst = (mask==~0ULL) ? src : dst
// mask must be 0 or ~0ULL (all-zeros or all-ones). No branch, no warp divergence.
// ---------------------------------------------------------------------------
__device__ inline void jacobian_cmov(JacobianPoint* __restrict__ dst,
                                     const JacobianPoint* __restrict__ src,
                                     uint64_t mask) {
    uint32_t mask32 = (uint32_t)(mask & 0xFFFFFFFFULL);
    dst->x.limbs[0] = (mask & src->x.limbs[0]) | (~mask & dst->x.limbs[0]);
    dst->x.limbs[1] = (mask & src->x.limbs[1]) | (~mask & dst->x.limbs[1]);
    dst->x.limbs[2] = (mask & src->x.limbs[2]) | (~mask & dst->x.limbs[2]);
    dst->x.limbs[3] = (mask & src->x.limbs[3]) | (~mask & dst->x.limbs[3]);
    dst->y.limbs[0] = (mask & src->y.limbs[0]) | (~mask & dst->y.limbs[0]);
    dst->y.limbs[1] = (mask & src->y.limbs[1]) | (~mask & dst->y.limbs[1]);
    dst->y.limbs[2] = (mask & src->y.limbs[2]) | (~mask & dst->y.limbs[2]);
    dst->y.limbs[3] = (mask & src->y.limbs[3]) | (~mask & dst->y.limbs[3]);
    dst->z.limbs[0] = (mask & src->z.limbs[0]) | (~mask & dst->z.limbs[0]);
    dst->z.limbs[1] = (mask & src->z.limbs[1]) | (~mask & dst->z.limbs[1]);
    dst->z.limbs[2] = (mask & src->z.limbs[2]) | (~mask & dst->z.limbs[2]);
    dst->z.limbs[3] = (mask & src->z.limbs[3]) | (~mask & dst->z.limbs[3]);
    dst->infinity = (bool)((mask32 & (uint32_t)src->infinity) |
                           (~mask32 & (uint32_t)dst->infinity));
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

    // Z1^2 [1S]
    FieldElement z1z1;
    field_sqr(&p->z, &z1z1);

    // U2 = X2*Z1^2 [1M]
    FieldElement u2;
    field_mul(&q->x, &z1z1, &u2);

    // S2 = Y2*Z1^3 [2M]
    FieldElement s2, temp;
    field_mul(&p->z, &z1z1, &temp);  // Z1^3
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

    // HH = H^2 [1S]
    FieldElement hh;
    field_sqr(&h, &hh);

    // HHH = H^3 [1M]
    FieldElement hhh;
    field_mul(&h, &hh, &hhh);

    // r = S2 - Y1
    FieldElement rr;
    field_sub(&s2, &p->y, &rr);

    // V = X1 * H^2 [1M]
    FieldElement v;
    field_mul(&p->x, &hh, &v);

    // X3 = r^2 - H^3 - 2*V [1S]
    FieldElement X3, Y3, Z3, t1;
    field_add(&v, &v, &t1);
    field_sqr(&rr, &X3);
    field_sub(&X3, &hhh, &X3);
    field_sub(&X3, &t1, &X3);

    // Y3 = r*(V - X3) - Y1*H^3 [2M]
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

    // Z1Z1 = Z1^2 [1S]
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

    // HH = H^2 [1S]
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

    // X3 = r^2-J-2*V [1S]
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

    // Z3 = (Z1+H)^2-Z1Z1-HH = 2*Z1*H [1S instead of 1M!]
    field_add(&p->z, &h, &temp);
    field_sqr(&temp, &Z3);
    field_sub(&Z3, &z1z1, &Z3);
    field_sub(&Z3, &hh, &Z3);

    // Return 2*H for serial inversion: Z_n = Z_0 * prod(2*H_i) = Z_0 * 2^N * prodH_i
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
    // Z1^2 = 1, Z1^3 = 1
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

    // HH = H^2 [1S]
    FieldElement hh;
    field_sqr(&h, &hh);

    // HHH = H^3 [1M]
    FieldElement hhh;
    field_mul(&h, &hh, &hhh);

    // r = Y2 - Y1 (since S2 = Y2)
    FieldElement rr;
    field_sub(&q->y, &p->y, &rr);

    // V = X1 * H^2 [1M]
    FieldElement v;
    field_mul(&p->x, &hh, &v);

    // X3 = r^2 - H^3 - 2*V [1S]
    FieldElement X3, Y3, t1;
    field_add(&v, &v, &t1);
    field_sqr(&rr, &X3);
    field_sub(&X3, &hhh, &X3);
    field_sub(&X3, &t1, &X3);

    // Y3 = r*(V - X3) - Y1*H^3 [2M]
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
    // Z1^2 [1S]
    FieldElement z1z1;
    field_sqr(&p->z, &z1z1);

    // U2 = X2*Z1^2 [1M]
    FieldElement u2;
    field_mul(&qx, &z1z1, &u2);

    // S2 = Y2*Z1^3 [2M]
    FieldElement s2, z1_cubed;
    field_mul(&p->z, &z1z1, &z1_cubed);  // Z1^3
    field_mul(&qy, &z1_cubed, &s2);

    // H = U2 - X1
    FieldElement h;
    field_sub(&u2, &p->x, &h);

    h_out = h;

    // HH = H^2 [1S]
    FieldElement hh;
    field_sqr(&h, &hh);

    // HHH = H^3 [1M]
    FieldElement hhh;
    field_mul(&h, &hh, &hhh);

    // r = S2 - Y1
    FieldElement rr;
    field_sub(&s2, &p->y, &rr);

    // V = X1 * H^2 [1M]
    FieldElement v;
    field_mul(&p->x, &hh, &v);

    // X3 = r^2 - H^3 - 2*V [1S]
    FieldElement X3, Y3, Z3, t1;
    field_add(&v, &v, &t1);
    field_sqr(&rr, &X3);
    field_sub(&X3, &hhh, &X3);
    field_sub(&X3, &t1, &X3);

    // Y3 = r*(V - X3) - Y1*H^3 [2M]
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
    // Z1Z1 = Z1^2 [1S]
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

    // HH = H^2 [1S]
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

    // X3 = r^2-J-2*V [1S]
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

    // Z3 = (Z1+H)^2-Z1Z1-HH = 2*Z1*H [1S instead of 1M! KEY OPTIMIZATION]
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
            // lambda = (3*x^2) / (2*y)
            FieldElement lambda, temp, x_sq;
            field_sqr(p_x, &x_sq);
            field_add(&x_sq, &x_sq, &temp);      // 2*x^2
            field_add(&temp, &x_sq, &temp);      // 3*x^2
            
            FieldElement two_y;
            field_add(p_y, p_y, &two_y);         // 2*y
            field_inv(&two_y, &two_y);           // 1/(2*y)
            field_mul(&temp, &two_y, &lambda);   // lambda
            
            // x' = lambda^2 - 2*x
            field_sqr(&lambda, r_x);
            field_sub(r_x, p_x, r_x);
            field_sub(r_x, p_x, r_x);
            
            // y' = lambda*(x - x') - y
            field_sub(p_x, r_x, &temp);
            field_mul(&lambda, &temp, r_y);
            field_sub(r_y, p_y, r_y);
            
            // Z = 1 (domain-aware)
            field_set_one(r_z);
            return;
        }
    }
    
    // Different points: lambda = (y2 - y1) / (x2 - x1)
    FieldElement lambda, dx, dy;
    field_sub(q_y, p_y, &dy);       // y2 - y1
    field_sub(q_x, p_x, &dx);       // x2 - x1
    field_inv(&dx, &dx);            // 1/(x2 - x1)
    field_mul(&dy, &dx, &lambda);   // lambda
    
    // x' = lambda^2 - x1 - x2
    field_sqr(&lambda, r_x);
    field_sub(r_x, p_x, r_x);
    field_sub(r_x, q_x, r_x);
    
    // y' = lambda*(x1 - x') - y1
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

// Apply GLV endomorphism: phi(x,y) = (beta*x, y)
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

// ---------------------------------------------------------------------------
// Constant-time scalar multiplication: r = k * P
// Uses "always double + always add + branchless cmov select" to eliminate
// all secret-dependent branches. Required for ECDH and any path where k is
// a private key. No warp divergence on GPU regardless of thread-specific k.
// Cost: 256 checked doubles + 256 checked mixed-adds + 256 cmovs.
// ---------------------------------------------------------------------------
__device__ inline void scalar_mul_ct(const JacobianPoint* p, const Scalar* k, JacobianPoint* r) {
    // Normalize base point P to affine (same logic as scalar_mul)
    AffinePoint base;
    if (p->z.limbs[0] == 1 && p->z.limbs[1] == 0 &&
        p->z.limbs[2] == 0 && p->z.limbs[3] == 0) {
        base.x = p->x;
        base.y = p->y;
    } else {
        FieldElement z_inv, z_inv2, z_inv3;
        field_inv(&p->z, &z_inv);
        field_sqr(&z_inv, &z_inv2);
        field_mul(&z_inv2, &z_inv, &z_inv3);
        field_mul(&p->x, &z_inv2, &base.x);
        field_mul(&p->y, &z_inv3, &base.y);
    }

    // Initialize accumulator to the point at infinity
    r->infinity = true;
    field_set_zero(&r->x);
    field_set_one(&r->y);
    field_set_zero(&r->z);

    JacobianPoint r_add;
    #pragma unroll 1
    for (int limb = 3; limb >= 0; limb--) {
        uint64_t w = k->limbs[limb];
        #pragma unroll 1
        for (int bit = 63; bit >= 0; bit--) {
            // Always double: jacobian_double handles r->infinity correctly
            jacobian_double(r, r);

            // Always compute r + P into r_add (checked, handles r->infinity)
            jacobian_add_mixed(r, &base, &r_add);

            // Branchless select: r = bit ? r_add : r  (no branch, no warp divergence)
            uint64_t mask = -(uint64_t)((w >> bit) & 1ULL);
            jacobian_cmov(r, &r_add, mask);
        }
    }
}

#if !SECP256K1_CUDA_LIMBS_32
// Forward declaration: wNAF-optimized GLV scalar mul (defined below)
__device__ inline void scalar_mul_glv_wnaf(const JacobianPoint* p, const Scalar* k, JacobianPoint* r);

// Forward declaration: simple window-1 GLV Shamir scalar mul (defined below)
__device__ inline void scalar_mul_glv_shamir(const JacobianPoint* p, const Scalar* k, JacobianPoint* r);

// GLV-accelerated scalar multiplication: r = k * P
// Splits k into k1 + k2*lambda and computes k1*P + k2*phi(P) with Shamir's trick
// Only available in 64-bit limb mode (glv_decompose requires 64-bit scalar ops)
__device__ inline void scalar_mul_glv(const JacobianPoint* p, const Scalar* k, JacobianPoint* r) {
    // Use simple window-1 Shamir's: lower register pressure, better occupancy for MSM scatter
    scalar_mul_glv_shamir(p, k, r);
}
#endif // !SECP256K1_CUDA_LIMBS_32

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

// -- Field Square Root --------------------------------------------------------
// Computes r = sqrt(a) = a^((p+1)/4) for secp256k1 where p == 3 (mod 4).
// (p+1)/4 = 0x3FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFBFFFFF0C
// Returns a valid sqrt if a is a quadratic residue; caller must verify r^2==a.
// Optimized addition chain: 255 squarings + 14 multiplications = 269 ops.
__device__ inline void field_sqrt(const FieldElement* a, FieldElement* r) {
    FieldElement x2, x3, x6, x22, x44, t;

    // x2 = a^(2^2-1) = a^3
    field_sqr(a, &x2);
    field_mul(&x2, a, &x2);

    // x3 = a^(2^3-1) = a^7
    field_sqr(&x2, &x3);
    field_mul(&x3, a, &x3);

    // x6 = a^(2^6-1)
    t = x3; field_sqr_n(&t, 3);
    field_mul(&t, &x3, &x6);

    // x9 = a^(2^9-1) [in t]
    t = x6; field_sqr_n(&t, 3);
    field_mul(&t, &x3, &t);

    // x11 = a^(2^11-1) [in t]
    field_sqr_n(&t, 2);
    field_mul(&t, &x2, &t);

    // x22 = a^(2^22-1)
    x3 = t;  // save x11
    field_sqr_n(&t, 11);
    field_mul(&t, &x3, &x22);

    // x44 = a^(2^44-1)
    t = x22; field_sqr_n(&t, 22);
    field_mul(&t, &x22, &x44);

    // x88 = a^(2^88-1) [in t]
    t = x44; field_sqr_n(&t, 44);
    field_mul(&t, &x44, &t);

    // x176 = a^(2^176-1) [in t]
    x3 = t;  // save x88
    field_sqr_n(&t, 88);
    field_mul(&t, &x3, &t);

    // x220 = a^(2^220-1) [in t]
    field_sqr_n(&t, 44);
    field_mul(&t, &x44, &t);

    // x222 = a^(2^222-1) [in t]
    field_sqr_n(&t, 2);
    field_mul(&t, &x2, &t);

    // Tail: extend 1^222 -> 1^223 0 1^22 0000 11 00
    // x223: t = t^2 * a
    field_sqr(&t, &t);
    field_mul(&t, a, &t);

    // Shift left 1: pattern 1^222 10
    field_sqr(&t, &t);

    // Shift left 22, add x22: pattern 1^222 10 1^22
    field_sqr_n(&t, 22);
    field_mul(&t, &x22, &t);

    // a^12 = x2^4 = (a^3)^4 [in x6]
    x6 = x2;
    field_sqr(&x6, &x6);
    field_sqr(&x6, &x6);

    // Shift left 8, add a^12: pattern 1^222 10 1^22 0000 11 00 = (p+1)/4
    field_sqr_n(&t, 8);
    field_mul(&t, &x6, r);
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

// Windowed generator multiplication kernel (w=4, shared-memory precomputed table)
// ~30-40% faster than plain double-and-add: 252 doublings + <=64 adds vs 256 + ~128.
__global__ void generator_mul_windowed_batch_kernel(const Scalar* scalars, JacobianPoint* results, int count);

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

// -- Precomputed Generator Table Builder --------------------------------------
// Builds table[i] = i*G for i=0..15 using Jacobian coordinates.
// Called by a single thread (threadIdx.x == 0).
// Caller MUST issue __syncthreads() after this returns.
__device__ inline void build_generator_table(JacobianPoint* table) {
    // table[0] = O (point at infinity)
    table[0].infinity = true;
    field_set_zero(&table[0].x);
    field_set_one(&table[0].y);
    field_set_zero(&table[0].z);

    // table[1] = G
    table[1] = GENERATOR_JACOBIAN;

    // table[2] = 2G
    jacobian_double_unchecked(&table[1], &table[2]);

    // table[3..15] = iG via mixed addition with G (affine, Z=1)
    AffinePoint G_aff;
    G_aff.x = GENERATOR_JACOBIAN.x;
    G_aff.y = GENERATOR_JACOBIAN.y;

    for (int i = 3; i <= 15; i++) {
        jacobian_add_mixed_unchecked(&table[i - 1], &G_aff, &table[i]);
    }
}

// -- Fixed-Window (w=4) Generator Scalar Multiplication ----------------------
// Uses precomputed table[0..15] = i*G from build_generator_table.
// Processes scalar 4 bits at a time (MSB to LSB): 64 windows.
// Cost: 252 doublings + <=64 jacobian_adds.
// Compared to plain double-and-add: saves ~50% of point additions.
__device__ inline void scalar_mul_generator_windowed(
    const JacobianPoint* table, const Scalar* k, JacobianPoint* r)
{
    r->infinity = true;
    field_set_zero(&r->x);
    field_set_one(&r->y);
    field_set_zero(&r->z);

    bool started = false;

    #pragma unroll 1
    for (int limb = 3; limb >= 0; limb--) {
        uint64_t w = k->limbs[limb];
        #pragma unroll 1
        for (int nib = 15; nib >= 0; nib--) {
            uint32_t idx = (uint32_t)((w >> (nib * 4)) & 0xFULL);

            if (started) {
                // Infinity is a valid intermediate result (P + (-P) = O).
                // jacobian_double handles infinity correctly (returns O).
                // We must NOT skip the 4 doublings on infinity, or the
                // window shift is lost and the result is wrong (bug: a single
                // P + (-P) mid-stream shifted all remaining windows by 4 bits).
                jacobian_double(r, r);
                jacobian_double(r, r);
                jacobian_double(r, r);
                jacobian_double(r, r);
            }

            if (idx != 0) {
                if (!started) {
                    *r = table[idx];
                    started = true;
                } else {
                    JacobianPoint tmp;
                    jacobian_add(r, &table[idx], &tmp);
                    *r = tmp;
                }
            }
        }
    }
}

// ============================================================================
// Optimized Scalar Multiplication -- wNAF w=4
// ============================================================================
// Windowed Non-Adjacent Form with pre-negated affine table.
// 8 precomputed odd multiples: [P, 3P, 5P, 7P, 9P, 11P, 13P, 15P]
// + their negations (negate y coordinate).
// Cost: ~252 doublings + ~64 mixed additions (vs ~256+128 for binary D&A).
// Lower addition count thanks to NAF property: no two consecutive non-zero digits.

// Compute signed-digit wNAF (w=4) representation of scalar.
// Outputs up to 257 digits in [-15,-13,...,-1,0,1,...,13,15], stored in out[].
// Returns the number of digits (length to iterate).
__device__ inline int scalar_to_wnaf4(const Scalar* k, int8_t* out) {
    // Work on mutable copy
    uint64_t limbs[5];
    limbs[0] = k->limbs[0]; limbs[1] = k->limbs[1];
    limbs[2] = k->limbs[2]; limbs[3] = k->limbs[3];
    limbs[4] = 0;

    int len = 0;
    while (limbs[0] | limbs[1] | limbs[2] | limbs[3] | limbs[4]) {
        if (limbs[0] & 1) {
            // k is odd: pick digit d in [-15..15] (odd) s.t. k-d divisible by 16
            int32_t d = (int32_t)(limbs[0] & 0x1F); // low 5 bits
            if (d >= 16) d -= 32; // convert to signed
            out[len] = (int8_t)d;
            // k -= d (subtract signed digit)
            if (d >= 0) {
                limbs[0] -= (uint64_t)d;
            } else {
                // add |d| to k
                uint64_t carry = (uint64_t)(-d);
                for (int i = 0; i < 5; i++) {
                    uint64_t limb_carry = 0;
                    limbs[i] = add_cc(limbs[i], carry, limb_carry);
                    carry = limb_carry;
                    if (!carry) break;
                }
            }
        } else {
            out[len] = 0;
        }
        // k >>= 1
        limbs[0] = (limbs[0] >> 1) | (limbs[1] << 63);
        limbs[1] = (limbs[1] >> 1) | (limbs[2] << 63);
        limbs[2] = (limbs[2] >> 1) | (limbs[3] << 63);
        limbs[3] = (limbs[3] >> 1) | (limbs[4] << 63);
        limbs[4] >>= 1;
        len++;
    }
    return len;
}

// wNAF w=4 scalar multiplication: r = k * P
// Uses 8 affine precomputed odd multiples + mixed Jacobian-affine addition.
__device__ inline void scalar_mul_wnaf(const JacobianPoint* p, const Scalar* k, JacobianPoint* r) {
    if (scalar_is_zero(k)) {
        r->infinity = true;
        field_set_zero(&r->x);
        field_set_one(&r->y);
        field_set_zero(&r->z);
        return;
    }

    // Convert base to affine
    AffinePoint base;
    if (p->z.limbs[0] == 1 && p->z.limbs[1] == 0 && p->z.limbs[2] == 0 && p->z.limbs[3] == 0) {
        base.x = p->x;
        base.y = p->y;
    } else {
        FieldElement z_inv, z_inv2, z_inv3;
        field_inv(&p->z, &z_inv);
        field_sqr(&z_inv, &z_inv2);
        field_mul(&z_inv2, &z_inv, &z_inv3);
        field_mul(&p->x, &z_inv2, &base.x);
        field_mul(&p->y, &z_inv3, &base.y);
    }

    // Precompute odd multiples: tbl[i] = (2i+1)*P for i=0..7
    // tbl[0]=P, tbl[1]=3P, tbl[2]=5P, ..., tbl[7]=15P
    AffinePoint tbl[8];
    AffinePoint neg_tbl[8]; // negated y for subtraction

    tbl[0] = base;

    // 2P in Jacobian
    JacobianPoint dbl_jac;
    {
        JacobianPoint p_jac;
        p_jac.x = base.x;
        p_jac.y = base.y;
        field_set_one(&p_jac.z);
        p_jac.infinity = false;
        jacobian_double_unchecked(&p_jac, &dbl_jac);
    }

    // Convert 2P to affine for mixed additions in table building
    AffinePoint dbl_aff;
    {
        FieldElement z_inv, z_inv2, z_inv3;
        field_inv(&dbl_jac.z, &z_inv);
        field_sqr(&z_inv, &z_inv2);
        field_mul(&z_inv2, &z_inv, &z_inv3);
        field_mul(&dbl_jac.x, &z_inv2, &dbl_aff.x);
        field_mul(&dbl_jac.y, &z_inv3, &dbl_aff.y);
    }

    // Build table: tbl[i] = tbl[i-1] + 2P (in Jacobian, then convert to affine)
    {
        JacobianPoint acc;
        acc.x = base.x;
        acc.y = base.y;
        field_set_one(&acc.z);
        acc.infinity = false;

        for (int i = 1; i < 8; i++) {
            jacobian_add_mixed_unchecked(&acc, &dbl_aff, &acc);
            // Convert acc to affine
            FieldElement zi, zi2, zi3;
            field_inv(&acc.z, &zi);
            field_sqr(&zi, &zi2);
            field_mul(&zi2, &zi, &zi3);
            field_mul(&acc.x, &zi2, &tbl[i].x);
            field_mul(&acc.y, &zi3, &tbl[i].y);
        }
    }

    // Pre-negate table
    FieldElement zero_fe;
    field_set_zero(&zero_fe);
    for (int i = 0; i < 8; i++) {
        neg_tbl[i].x = tbl[i].x;
        field_sub(&zero_fe, &tbl[i].y, &neg_tbl[i].y);
    }

    // Compute wNAF digits
    int8_t wnaf[260];
    for (int i = 0; i < 260; i++) wnaf[i] = 0;
    int wnaf_len = scalar_to_wnaf4(k, wnaf);

    // Main loop: MSB to LSB
    r->infinity = true;
    field_set_zero(&r->x);
    field_set_one(&r->y);
    field_set_zero(&r->z);

    #pragma unroll 1
    for (int i = wnaf_len - 1; i >= 0; i--) {
        if (!r->infinity) {
            jacobian_double_unchecked(r, r);
        }
        int8_t d = wnaf[i];
        if (d > 0) {
            int idx = (d - 1) / 2; // d=1->0, d=3->1, ..., d=15->7
            if (r->infinity) {
                r->x = tbl[idx].x;
                r->y = tbl[idx].y;
                field_set_one(&r->z);
                r->infinity = false;
            } else {
                jacobian_add_mixed_unchecked(r, &tbl[idx], r);
            }
        } else if (d < 0) {
            int idx = (-d - 1) / 2;
            if (r->infinity) {
                r->x = neg_tbl[idx].x;
                r->y = neg_tbl[idx].y;
                field_set_one(&r->z);
                r->infinity = false;
            } else {
                jacobian_add_mixed_unchecked(r, &neg_tbl[idx], r);
            }
        }
    }
}

// ============================================================================
// GLV + wNAF Scalar Multiplication
// ============================================================================
// wNAF Helpers for GLV Scalar Multiplication
// ============================================================================

// Extract `count` bits starting at bit position `pos` from 4x64 LE scalar limbs.
// count must be in [1, 31]. Positions beyond 255 return 0.
__device__ inline uint32_t scalar_get_bits(const Scalar* s, int pos, int count) {
    int limb_idx = pos >> 6;
    int bit_off  = pos & 63;
    uint64_t val = 0;
    if (limb_idx < 4) {
        val = s->limbs[limb_idx] >> bit_off;
        if (bit_off + count > 64 && limb_idx + 1 < 4) {
            val |= s->limbs[limb_idx + 1] << (64 - bit_off);
        }
    }
    return static_cast<uint32_t>(val) & ((1u << count) - 1);
}

// Encode scalar into windowed Non-Adjacent Form (wNAF).
// Digits are odd values in [-(2^(w-1)-1), ..., -1, 0, 1, ..., 2^(w-1)-1].
// Returns length of wNAF (position of last non-zero digit + 1).
__device__ inline int wnaf_encode(const Scalar* s, int w, int8_t* out, int max_len) {
    // Zero-fill
    for (int i = 0; i < max_len; i++) out[i] = 0;

    int carry = 0;
    int last_set = -1;
    int bit = 0;

    while (bit < max_len) {
        uint32_t b = scalar_get_bits(s, bit, 1);
        if (static_cast<int>(b) == carry) {
            bit++;
            continue;
        }

        int now = w;
        if (now > max_len - bit) now = max_len - bit;

        int word = static_cast<int>(scalar_get_bits(s, bit, now)) + carry;
        carry = word >> (w - 1);
        word -= carry << w;

        out[bit] = static_cast<int8_t>(word);
        last_set = bit;
        bit += now;
    }

    return (last_set >= 0) ? (last_set + 1) : 0;
}

// Build odd-multiple table [1P, 3P, 5P, ..., (2*table_size-1)*P] using the
// z-ratio technique from libsecp256k1 (ZERO field inversions).
// All table entries share an implied Z = globalz.
// base must be in affine coordinates.
__device__ inline void build_wnaf_table_zr(
    const AffinePoint* base,
    AffinePoint* tbl,
    int table_size,
    FieldElement* globalz)
{
    // D = 2*base (Jacobian double, base has Z=1)
    JacobianPoint P_jac;
    P_jac.x = base->x; P_jac.y = base->y;
    field_set_one(&P_jac.z); P_jac.infinity = false;

    JacobianPoint D;
    // Unchecked: P_jac.infinity = false (set above).
    jacobian_double_unchecked(&P_jac, &D);

    // C = D.z, C2 = C^2, C3 = C^3
    FieldElement C = D.z;
    FieldElement C2, C3;
    field_sqr(&C, &C2);
    field_mul(&C2, &C, &C3);

    // d_aff = D.(x,y) on the isomorphic curve (Z cancels in isomorphism)
    AffinePoint d_aff;
    d_aff.x = D.x; d_aff.y = D.y;

    // Transform base onto iso curve: (base.x * C^2, base.y * C^3) with Z=1
    JacobianPoint ai;
    field_mul(&base->x, &C2, &ai.x);
    field_mul(&base->y, &C3, &ai.y);
    field_set_one(&ai.z);
    ai.infinity = false;

    tbl[0].x = ai.x;
    tbl[0].y = ai.y;

    // z-ratios: zr[i] = H_i from jacobian_add_mixed_h (Z_{i+1} = Z_i * H_i)
    FieldElement zr[8]; // max 8 for w=5
    zr[0] = C; // iso mapping z-ratio

    // Build rest: (2i+1)*P on iso curve via mixed adds
    for (int i = 1; i < table_size; i++) {
        FieldElement h;
        jacobian_add_mixed_h(&ai, &d_aff, &ai, h);
        tbl[i].x = ai.x;
        tbl[i].y = ai.y;
        zr[i] = h;
    }

    // globalz = ai.z * C (maps from iso curve back to secp256k1)
    field_mul(&ai.z, &C, globalz);

    // Backward sweep: rescale entries so all share Z = Z_last
    FieldElement zs = zr[table_size - 1];
    for (int idx = table_size - 2; idx >= 0; --idx) {
        if (idx != table_size - 2) {
            FieldElement tmp;
            field_mul(&zs, &zr[idx + 1], &tmp);
            zs = tmp;
        }
        FieldElement zs2, zs3;
        field_sqr(&zs, &zs2);
        field_mul(&zs2, &zs, &zs3);
        FieldElement tx, ty;
        field_mul(&tbl[idx].x, &zs2, &tx);
        field_mul(&tbl[idx].y, &zs3, &ty);
        tbl[idx].x = tx;
        tbl[idx].y = ty;
    }
}

// Derive endomorphism table: phi(P) = (beta*x, +-y)
__device__ inline void derive_endo_table(
    const AffinePoint* tbl_P,
    AffinePoint* tbl_endoP,
    int table_size,
    bool negate_y)
{
    FieldElement beta_fe;
    beta_fe.limbs[0] = BETA[0]; beta_fe.limbs[1] = BETA[1];
    beta_fe.limbs[2] = BETA[2]; beta_fe.limbs[3] = BETA[3];

    for (int i = 0; i < table_size; i++) {
        field_mul(&tbl_P[i].x, &beta_fe, &tbl_endoP[i].x);
        if (negate_y) {
            field_negate(&tbl_P[i].y, &tbl_endoP[i].y);
        } else {
            tbl_endoP[i].y = tbl_P[i].y;
        }
    }
}

// ============================================================================
// GLV + wNAF scalar multiplication for arbitrary point: k*P
// k*P = k1*P + k2*lambda*P where k1,k2 are ~128 bits each.
// Uses Shamir's interleaved wNAF (w=5) with precomputed odd-multiple tables
// built via z-ratio technique (zero additional field inversions).
// Cost: ~128 doublings + ~42 mixed additions (vs ~128+96 for old bit-by-bit).
// ============================================================================

__device__ inline void scalar_mul_glv_wnaf(const JacobianPoint* p, const Scalar* k, JacobianPoint* r) {
    if (scalar_is_zero(k)) {
        r->infinity = true;
        field_set_zero(&r->x);
        field_set_one(&r->y);
        field_set_zero(&r->z);
        return;
    }

    GLVDecomposition decomp = glv_decompose(k);

    // Convert base to affine (1 field inversion if Z != 1)
    AffinePoint base;
    if (p->z.limbs[0] == 1 && p->z.limbs[1] == 0 && p->z.limbs[2] == 0 && p->z.limbs[3] == 0) {
        base.x = p->x;
        base.y = p->y;
    } else {
        FieldElement z_inv, z_inv2, z_inv3;
        field_inv(&p->z, &z_inv);
        field_sqr(&z_inv, &z_inv2);
        field_mul(&z_inv2, &z_inv, &z_inv3);
        field_mul(&p->x, &z_inv2, &base.x);
        field_mul(&p->y, &z_inv3, &base.y);
    }

    // Negate base if k1 was negated in GLV decomposition
    if (decomp.k1_neg) {
        field_negate(&base.y, &base.y);
    }

    // Build precomputed table [1P, 3P, 5P, ..., 15P] using z-ratio (0 inversions)
    constexpr int WNAF_W = 5;
    constexpr int TABLE_SIZE = (1 << (WNAF_W - 2)); // 8

    AffinePoint tbl_P[TABLE_SIZE];
    FieldElement globalz;
    build_wnaf_table_zr(&base, tbl_P, TABLE_SIZE, &globalz);

    // Derive endomorphism table: phi(P) = (beta*x, +-y)
    AffinePoint tbl_phiP[TABLE_SIZE];
    bool flip_phi = (decomp.k1_neg != decomp.k2_neg);
    derive_endo_table(tbl_P, tbl_phiP, TABLE_SIZE, flip_phi);

    // wNAF encode both ~128-bit half-scalars
    constexpr int WNAF_MAXLEN = 130;
    int8_t wnaf1[WNAF_MAXLEN], wnaf2[WNAF_MAXLEN];
    wnaf_encode(&decomp.k1, WNAF_W, wnaf1, WNAF_MAXLEN);
    wnaf_encode(&decomp.k2, WNAF_W, wnaf2, WNAF_MAXLEN);
    // Fixed loop bound — avoids warp divergence from variable max_len
    // wnaf_encode zero-fills beyond encoding, so extra iterations are no-ops
    constexpr int LOOP_LEN = WNAF_MAXLEN;

    // Shamir's 2-stream wNAF loop: single doubling chain, dual table lookups
    r->infinity = true;
    field_set_zero(&r->x);
    field_set_one(&r->y);
    field_set_zero(&r->z);

    #pragma unroll 1
    for (int i = LOOP_LEN - 1; i >= 0; --i) {
        if (!r->infinity) {
            // Unchecked: r->infinity is false (checked above).
            jacobian_double_unchecked(r, r);
        }

        // Apply k1 wNAF digit
        int8_t d1 = wnaf1[i];
        if (d1 != 0) {
            int idx = ((d1 > 0) ? d1 : -d1);
            idx = (idx - 1) >> 1;
            AffinePoint pt = tbl_P[idx];
            if (d1 < 0) field_negate(&pt.y, &pt.y);

            if (r->infinity) {
                r->x = pt.x; r->y = pt.y;
                field_set_one(&r->z); r->infinity = false;
            } else {
                // Unchecked: r->infinity is false (else branch above).
                jacobian_add_mixed_unchecked(r, &pt, r);
            }
        }

        // Apply k2 wNAF digit
        int8_t d2 = wnaf2[i];
        if (d2 != 0) {
            int idx = ((d2 > 0) ? d2 : -d2);
            idx = (idx - 1) >> 1;
            AffinePoint pt = tbl_phiP[idx];
            if (d2 < 0) field_negate(&pt.y, &pt.y);

            if (r->infinity) {
                r->x = pt.x; r->y = pt.y;
                field_set_one(&r->z); r->infinity = false;
            } else {
                // Unchecked: r->infinity is false (else branch above).
                jacobian_add_mixed_unchecked(r, &pt, r);
            }
        }
    }

    // Apply globalz correction (z-ratio table has implied Z=globalz)
    if (!r->infinity) {
        FieldElement tmp;
        field_mul(&r->z, &globalz, &tmp);
        r->z = tmp;
    }
}

// ============================================================================
// GLV + Shamir window-1 scalar multiplication: r = k*P
// Splits k into k1 + k2*lambda (~128-bit each), then does Shamir's interleaved
// binary double-and-add with only two precomputed affine points: P and phi(P).
// Cost: ~129 doublings + ~96 mixed adds  (vs 255 + 128 for plain binary)
// Register pressure: ~100 regs/thread    (vs ~370 for wNAF-5 GLV approach)
// Better occupancy at large n; no large precomp table or z-ratio overhead.
// ============================================================================
__device__ inline void scalar_mul_glv_shamir(const JacobianPoint* p, const Scalar* k, JacobianPoint* r) {
    if (scalar_is_zero(k)) {
        r->infinity = true;
        field_set_zero(&r->x);
        field_set_one(&r->y);
        field_set_zero(&r->z);
        return;
    }

    GLVDecomposition decomp = glv_decompose(k);

    // Convert base point to affine (fast path if Z=1)
    AffinePoint base;
    if (p->z.limbs[0] == 1 && p->z.limbs[1] == 0 && p->z.limbs[2] == 0 && p->z.limbs[3] == 0) {
        base.x = p->x;
        base.y = p->y;
    } else {
        FieldElement z_inv, z_inv2, z_inv3;
        field_inv(&p->z, &z_inv);
        field_sqr(&z_inv, &z_inv2);
        field_mul(&z_inv2, &z_inv, &z_inv3);
        field_mul(&p->x, &z_inv2, &base.x);
        field_mul(&p->y, &z_inv3, &base.y);
    }

    // Apply k1 sign to base
    if (decomp.k1_neg) {
        field_negate(&base.y, &base.y);
    }

    // Endomorphism: phi(P) = (beta*x, y), apply k2 sign separately
    AffinePoint phi_base;
    FieldElement beta_fe;
    beta_fe.limbs[0] = BETA[0]; beta_fe.limbs[1] = BETA[1];
    beta_fe.limbs[2] = BETA[2]; beta_fe.limbs[3] = BETA[3];
    field_mul(&base.x, &beta_fe, &phi_base.x);
    // k2_neg XOR k1_neg: phi_base.y = base.y flipped if signs differ
    if (decomp.k1_neg == decomp.k2_neg) {
        phi_base.y = base.y;
    } else {
        field_negate(&base.y, &phi_base.y);
    }

    // Shamir's interleaved binary scan over 130 bits (k1, k2 up to ~129 bits each)
    r->infinity = true;
    field_set_zero(&r->x);
    field_set_one(&r->y);
    field_set_zero(&r->z);

    // Scan: limb=2 (top 2 bits only), then limb=1, limb=0
    // We use #pragma unroll 1 to cap register usage
    #pragma unroll 1
    for (int limb = 2; limb >= 0; limb--) {
        int start_bit = (limb == 2) ? 1 : 63; // only bits [1:0] of limb 2
        uint64_t w1 = decomp.k1.limbs[limb];
        uint64_t w2 = decomp.k2.limbs[limb];
        for (int bit = start_bit; bit >= 0; bit--) {
            if (!r->infinity) {
                jacobian_double_unchecked(r, r);
            }
            int b1 = (int)((w1 >> bit) & 1ULL);
            int b2 = (int)((w2 >> bit) & 1ULL);
            if (b1 | b2) {
                if (r->infinity) {
                    if (b1 && b2) {
                        r->x = base.x; r->y = base.y;
                        field_set_one(&r->z); r->infinity = false;
                        jacobian_add_mixed_unchecked(r, &phi_base, r);
                    } else if (b1) {
                        r->x = base.x; r->y = base.y;
                        field_set_one(&r->z); r->infinity = false;
                    } else {
                        r->x = phi_base.x; r->y = phi_base.y;
                        field_set_one(&r->z); r->infinity = false;
                    }
                } else {
                    if (b1) jacobian_add_mixed_unchecked(r, &base, r);
                    if (b2) jacobian_add_mixed_unchecked(r, &phi_base, r);
                }
            }
        }
    }
}

// ============================================================================
// Shamir's Double-Mul: R = a*P + b*Q (single interleaved pass)
// ============================================================================
// Used by ECDSA verify (u1*G + u2*Q) and Schnorr verify (s*G + (-e)*P).
// Single doubling chain instead of two separate scalar_mul calls.
// When one point is the generator, uses GLV decomposition for both.
// Cost: ~max(bitlen(a), bitlen(b)) doublings + additions (vs 2*256 separate).

__device__ inline void shamir_double_mul(
    const JacobianPoint* P, const Scalar* a,
    const JacobianPoint* Q, const Scalar* b,
    JacobianPoint* r)
{
    // Handle degenerate cases
    if (scalar_is_zero(a) && scalar_is_zero(b)) {
        r->infinity = true;
        field_set_zero(&r->x);
        field_set_one(&r->y);
        field_set_zero(&r->z);
        return;
    }
    if (scalar_is_zero(a)) {
        scalar_mul_glv_wnaf(Q, b, r);
        return;
    }
    if (scalar_is_zero(b)) {
        scalar_mul_glv_wnaf(P, a, r);
        return;
    }

    // Convert both points to affine
    AffinePoint aff_P, aff_Q;
    if (P->z.limbs[0] == 1 && P->z.limbs[1] == 0 && P->z.limbs[2] == 0 && P->z.limbs[3] == 0) {
        aff_P.x = P->x; aff_P.y = P->y;
    } else {
        FieldElement zi, zi2, zi3;
        field_inv(&P->z, &zi);
        field_sqr(&zi, &zi2);
        field_mul(&zi2, &zi, &zi3);
        field_mul(&P->x, &zi2, &aff_P.x);
        field_mul(&P->y, &zi3, &aff_P.y);
    }
    if (Q->z.limbs[0] == 1 && Q->z.limbs[1] == 0 && Q->z.limbs[2] == 0 && Q->z.limbs[3] == 0) {
        aff_Q.x = Q->x; aff_Q.y = Q->y;
    } else {
        FieldElement zi, zi2, zi3;
        field_inv(&Q->z, &zi);
        field_sqr(&zi, &zi2);
        field_mul(&zi2, &zi, &zi3);
        field_mul(&Q->x, &zi2, &aff_Q.x);
        field_mul(&Q->y, &zi3, &aff_Q.y);
    }

    // Precompute P+Q for Shamir's trick (4 combos: 00, 01=Q, 10=P, 11=P+Q)
    AffinePoint aff_PQ; // P+Q
    {
        JacobianPoint jp, jpq;
        jp.x = aff_P.x; jp.y = aff_P.y;
        field_set_one(&jp.z); jp.infinity = false;
        jacobian_add_mixed_unchecked(&jp, &aff_Q, &jpq);
        if (jpq.infinity) {
            // P = -Q, degenerate
            r->infinity = true;
            field_set_zero(&r->x);
            field_set_one(&r->y);
            field_set_zero(&r->z);
            return;
        }
        FieldElement zi, zi2, zi3;
        field_inv(&jpq.z, &zi);
        field_sqr(&zi, &zi2);
        field_mul(&zi2, &zi, &zi3);
        field_mul(&jpq.x, &zi2, &aff_PQ.x);
        field_mul(&jpq.y, &zi3, &aff_PQ.y);
    }

    // Store in array for branchless lookup: index = (bit_a << 1) | bit_b
    // table[0] = unused(identity), table[1] = Q, table[2] = P, table[3] = P+Q
    AffinePoint table[4];
    table[1] = aff_Q;
    table[2] = aff_P;
    table[3] = aff_PQ;

    int len_a = scalar_bitlen(a);
    int len_b = scalar_bitlen(b);
    int max_len = (len_a > len_b) ? len_a : len_b;

    r->infinity = true;
    field_set_zero(&r->x);
    field_set_one(&r->y);
    field_set_zero(&r->z);

    #pragma unroll 1
    for (int i = max_len - 1; i >= 0; --i) {
        if (!r->infinity) {
            jacobian_double_unchecked(r, r);
        }

        int ba = scalar_bit(a, i);
        int bb = scalar_bit(b, i);
        int idx = (ba << 1) | bb;

        if (idx != 0) {
            if (r->infinity) {
                r->x = table[idx].x;
                r->y = table[idx].y;
                field_set_one(&r->z);
                r->infinity = false;
            } else {
                jacobian_add_mixed_unchecked(r, &table[idx], r);
            }
        }
    }
}

// ============================================================================
// Shamir's Double-Mul with GLV: R = a*P + b*Q (4-way interleaving)
// ============================================================================
// Decomposes both scalars via GLV: a = a1 + a2*lambda, b = b1 + b2*lambda
// Then computes a1*P + a2*endo(P) + b1*Q + b2*endo(Q) in a single pass.
// All 15 non-identity combos of the 4 base points are precomputed into a
// 16-entry lookup table (batch inversion via Montgomery's trick: 1 field_inv).
// Main loop: ~128 doublings + ~120 mixed additions (single lookup per position).
// This is the optimal path for ECDSA verify (u1*G + u2*Q) and Schnorr verify.

__device__ inline void shamir_double_mul_glv(
    const JacobianPoint* P, const Scalar* a,
    const JacobianPoint* Q, const Scalar* b,
    JacobianPoint* r)
{
    // Handle degenerate cases
    if (scalar_is_zero(a) && scalar_is_zero(b)) {
        r->infinity = true;
        field_set_zero(&r->x);
        field_set_one(&r->y);
        field_set_zero(&r->z);
        return;
    }
    if (scalar_is_zero(a)) {
        scalar_mul_glv_wnaf(Q, b, r);
        return;
    }
    if (scalar_is_zero(b)) {
        scalar_mul_glv_wnaf(P, a, r);
        return;
    }

    // GLV decompose both scalars
    GLVDecomposition da = glv_decompose(a);
    GLVDecomposition db = glv_decompose(b);

    // Convert P to affine
    AffinePoint aff_P;
    if (P->z.limbs[0] == 1 && P->z.limbs[1] == 0 && P->z.limbs[2] == 0 && P->z.limbs[3] == 0) {
        aff_P.x = P->x; aff_P.y = P->y;
    } else {
        FieldElement zi, zi2, zi3;
        field_inv(&P->z, &zi);
        field_sqr(&zi, &zi2);
        field_mul(&zi2, &zi, &zi3);
        field_mul(&P->x, &zi2, &aff_P.x);
        field_mul(&P->y, &zi3, &aff_P.y);
    }

    // Convert Q to affine
    AffinePoint aff_Q;
    if (Q->z.limbs[0] == 1 && Q->z.limbs[1] == 0 && Q->z.limbs[2] == 0 && Q->z.limbs[3] == 0) {
        aff_Q.x = Q->x; aff_Q.y = Q->y;
    } else {
        FieldElement zi, zi2, zi3;
        field_inv(&Q->z, &zi);
        field_sqr(&zi, &zi2);
        field_mul(&zi2, &zi, &zi3);
        field_mul(&Q->x, &zi2, &aff_Q.x);
        field_mul(&Q->y, &zi3, &aff_Q.y);
    }

    // Build 4 base points: P1, P2=endo(P), Q1, Q2=endo(Q) with sign adjustments
    AffinePoint pts[4];

    FieldElement beta_fe;
    beta_fe.limbs[0] = BETA[0]; beta_fe.limbs[1] = BETA[1];
    beta_fe.limbs[2] = BETA[2]; beta_fe.limbs[3] = BETA[3];

    pts[0] = aff_P;
    if (da.k1_neg) field_negate(&pts[0].y, &pts[0].y);

    field_mul(&aff_P.x, &beta_fe, &pts[1].x);
    pts[1].y = aff_P.y;
    if (da.k2_neg) field_negate(&pts[1].y, &pts[1].y);

    pts[2] = aff_Q;
    if (db.k1_neg) field_negate(&pts[2].y, &pts[2].y);

    field_mul(&aff_Q.x, &beta_fe, &pts[3].x);
    pts[3].y = aff_Q.y;
    if (db.k2_neg) field_negate(&pts[3].y, &pts[3].y);

    // Precompute all 15 non-identity combos into table[1..15]
    // Index encoding: bit0=P1(da.k1), bit1=P2(da.k2), bit2=Q1(db.k1), bit3=Q2(db.k2)
    AffinePoint table[16];

    // Singles (already affine)
    table[1] = pts[0];   // P1
    table[2] = pts[1];   // P2
    table[4] = pts[2];   // Q1
    table[8] = pts[3];   // Q2

    // Build 11 multi-point combos in Jacobian, then batch-convert to affine
    JacobianPoint jc[11];

    // Helper: create Jacobian from affine with Z=1
    #define MADD_PAIR(dst, a_idx, b_idx) { \
        JacobianPoint _j; _j.x = pts[a_idx].x; _j.y = pts[a_idx].y; \
        field_set_one(&_j.z); _j.infinity = false; \
        jacobian_add_mixed(&_j, &pts[b_idx], &dst); }

    // 6 pairs
    MADD_PAIR(jc[0], 0, 1)   // P1+P2       -> table[3]
    MADD_PAIR(jc[1], 0, 2)   // P1+Q1       -> table[5]
    MADD_PAIR(jc[2], 1, 2)   // P2+Q1       -> table[6]
    MADD_PAIR(jc[3], 0, 3)   // P1+Q2       -> table[9]
    MADD_PAIR(jc[4], 1, 3)   // P2+Q2       -> table[10]
    MADD_PAIR(jc[5], 2, 3)   // Q1+Q2       -> table[12]

    #undef MADD_PAIR

    // 4 triples (Jacobian pair + affine single)
    jacobian_add_mixed(&jc[0], &pts[2], &jc[6]);   // P1+P2+Q1    -> table[7]
    jacobian_add_mixed(&jc[0], &pts[3], &jc[7]);   // P1+P2+Q2    -> table[11]
    jacobian_add_mixed(&jc[1], &pts[3], &jc[8]);   // P1+Q1+Q2    -> table[13]
    jacobian_add_mixed(&jc[2], &pts[3], &jc[9]);   // P2+Q1+Q2    -> table[14]

    // 1 quad (Jacobian triple + affine single)
    jacobian_add_mixed(&jc[6], &pts[3], &jc[10]);  // P1+P2+Q1+Q2 -> table[15]

    // Check for degenerate combos (e.g., P=Q makes some pairs infinity).
    // Extremely rare in practice (would need pubkey == generator).
    {
        bool has_degen = false;
        for (int i = 0; i < 11; i++) {
            if (jc[i].infinity) { has_degen = true; break; }
        }
        if (has_degen) {
            // Fallback: 4-point bit-by-bit accumulation (original approach)
            int len1 = scalar_bitlen(&da.k1);
            int len2 = scalar_bitlen(&da.k2);
            int len3 = scalar_bitlen(&db.k1);
            int len4 = scalar_bitlen(&db.k2);
            int max_len = len1;
            if (len2 > max_len) max_len = len2;
            if (len3 > max_len) max_len = len3;
            if (len4 > max_len) max_len = len4;

            r->infinity = true;
            field_set_zero(&r->x);
            field_set_one(&r->y);
            field_set_zero(&r->z);

            const Scalar* scalars[4] = { &da.k1, &da.k2, &db.k1, &db.k2 };
            #pragma unroll 1
            for (int i = max_len - 1; i >= 0; --i) {
                // Unchecked: r->infinity is false (checked above).
                if (!r->infinity) jacobian_double_unchecked(r, r);
                for (int j = 0; j < 4; j++) {
                    if (scalar_bit(scalars[j], i)) {
                        if (r->infinity) {
                            r->x = pts[j].x; r->y = pts[j].y;
                            field_set_one(&r->z); r->infinity = false;
                        } else {
                            // Unchecked: r->infinity is false (else branch above).
                            jacobian_add_mixed_unchecked(r, &pts[j], r);
                        }
                    }
                }
            }
            return;
        }
    }

    // Batch inversion: Montgomery's trick for 11 Z values -> 1 field_inv
    {
        FieldElement prefix[11];
        prefix[0] = jc[0].z;
        for (int i = 1; i < 11; i++) {
            field_mul(&prefix[i - 1], &jc[i].z, &prefix[i]);
        }

        FieldElement inv_prod;
        field_inv(&prefix[10], &inv_prod);

        // Backward sweep: recover individual Z^-1
        FieldElement z_inv[11];
        for (int i = 10; i > 0; --i) {
            field_mul(&inv_prod, &prefix[i - 1], &z_inv[i]);
            FieldElement tmp;
            field_mul(&inv_prod, &jc[i].z, &tmp);
            inv_prod = tmp;
        }
        z_inv[0] = inv_prod;

        // Convert each Jacobian -> affine and store in table
        constexpr int tbl_map[11] = {3, 5, 6, 9, 10, 12, 7, 11, 13, 14, 15};
        for (int i = 0; i < 11; i++) {
            FieldElement zi2, zi3;
            field_sqr(&z_inv[i], &zi2);
            field_mul(&zi2, &z_inv[i], &zi3);
            field_mul(&jc[i].x, &zi2, &table[tbl_map[i]].x);
            field_mul(&jc[i].y, &zi3, &table[tbl_map[i]].y);
        }
    }

    // Main loop: single doubling chain with 16-entry table lookup
    int len1 = scalar_bitlen(&da.k1);
    int len2 = scalar_bitlen(&da.k2);
    int len3 = scalar_bitlen(&db.k1);
    int len4 = scalar_bitlen(&db.k2);
    int max_len = len1;
    if (len2 > max_len) max_len = len2;
    if (len3 > max_len) max_len = len3;
    if (len4 > max_len) max_len = len4;

    r->infinity = true;
    field_set_zero(&r->x);
    field_set_one(&r->y);
    field_set_zero(&r->z);

    #pragma unroll 1
    for (int i = max_len - 1; i >= 0; --i) {
        if (!r->infinity) {
            // Unchecked: r->infinity is false (checked above).
            jacobian_double_unchecked(r, r);
        }

        int idx = scalar_bit(&da.k1, i)
                | (scalar_bit(&da.k2, i) << 1)
                | (scalar_bit(&db.k1, i) << 2)
                | (scalar_bit(&db.k2, i) << 3);

        if (idx != 0) {
            if (r->infinity) {
                r->x = table[idx].x;
                r->y = table[idx].y;
                field_set_one(&r->z);
                r->infinity = false;
            } else {
                // Unchecked: r->infinity is false (else branch above).
                jacobian_add_mixed_unchecked(r, &table[idx], r);
            }
        }
    }
}

// ============================================================================
// Precomputed Generator Tables in __constant__ Memory
// ============================================================================

// -- w=8 table: 256 affine points [0..255]*G, 16 KB constant memory ----------
// Used by scalar_mul_generator_w8 (32 doublings + <=32 additions).
#include "gen_table_w8.cuh"

// -- w=4 table: 16 affine points [0..15]*G (legacy, kept for compatibility) ---
// 16 affine points: table[i] = i*G for i=0..15 (table[0] is unused/identity)
// Precomputed offline and stored as constants.
// These are the standard secp256k1 generator multiples.

__device__ __constant__ static const AffinePoint GENERATOR_TABLE_AFFINE[16] = {
    // [0] = O (identity, unused -- handled by branch)
    {{{0, 0, 0, 0}}, {{0, 0, 0, 0}}},
    // [1] = G
    {{{0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL, 0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL}},
     {{0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL, 0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL}}},
    // [2] = 2G
    {{{0xABAC09B95C709EE5ULL, 0x5C778E4B8CEF3CA7ULL, 0x3045406E95C07CD8ULL, 0xC6047F9441ED7D6DULL}},
     {{0x236431A950CFE52AULL, 0xF7F632653266D0E1ULL, 0xA3C58419466CEAEEULL, 0x1AE168FEA63DC339ULL}}},
    // [3] = 3G
    {{{0x8601F113BCE036F9ULL, 0xB531C845836F99B0ULL, 0x49344F85F89D5229ULL, 0xF9308A019258C310ULL}},
     {{0x6CB9FD7584B8E672ULL, 0x6500A99934C2231BULL, 0x0FE337E62A37F356ULL, 0x388F7B0F632DE814ULL}}},
    // [4] = 4G
    {{{0x74FA94ABE8C4CD13ULL, 0xCC6C13900EE07584ULL, 0x581E4904930B1404ULL, 0xE493DBF1C10D80F3ULL}},
     {{0xCFE97BDC47739922ULL, 0xD967AE33BFBDFE40ULL, 0x5642E2098EA51448ULL, 0x51ED993EA0D455B7ULL}}},
    // [5] = 5G
    {{{0xCBA8D569B240EFE4ULL, 0xE88B84BDDC619AB7ULL, 0x55B4A7250A5C5128ULL, 0x2F8BDE4D1A072093ULL}},
     {{0xDCA87D3AA6AC62D6ULL, 0xF788271BAB0D6840ULL, 0xD4DBA9DDA6C9C426ULL, 0xD8AC222636E5E3D6ULL}}},
    // [6] = 6G
    {{{0x2F057A1460297556ULL, 0x82F6472F8568A18BULL, 0x20453A14355235D3ULL, 0xFFF97BD5755EEEA4ULL}},
     {{0x3C870C36B075F297ULL, 0xDE80F0F6518FE4A0ULL, 0xF3BE96017F45C560ULL, 0xAE12777AACFBB620ULL}}},
    // [7] = 7G
    {{{0xE92BDDEDCAC4F9BCULL, 0x3D419B7E0330E39CULL, 0xA398F365F2EA7A0EULL, 0x5CBDF0646E5DB4EAULL}},
     {{0xA5082628087264DAULL, 0xA813D0B813FDE7B5ULL, 0xA3178D6D861A54DBULL, 0x6AEBCA40BA255960ULL}}},
    // [8] = 8G
    {{{0x67784EF3E10A2A01ULL, 0x0A1BDD05E5AF888AULL, 0xAFF3843FB70F3C2FULL, 0x2F01E5E15CCA351DULL}},
     {{0xB5DA2CB76CBDE904ULL, 0xC2E213D6BA5B7617ULL, 0x293D082A132D13B4ULL, 0x5C4DA8A741539949ULL}}},
    // [9] = 9G
    {{{0xC35F110DFC27CCBEULL, 0xE09796974C57E714ULL, 0x09AD178A9F559ABDULL, 0xACD484E2F0C7F653ULL}},
     {{0x05CC262AC64F9C37ULL, 0xADD888A4375F8E0FULL, 0x64380971763B61E9ULL, 0xCC338921B0A7D9FDULL}}},
    // [10] = 10G
    {{{0x52A68E2A47E247C7ULL, 0x3442D49B1943C2B7ULL, 0x35477C7B1AE6AE5DULL, 0xA0434D9E47F3C862ULL}},
     {{0x3CBEE53B037368D7ULL, 0x6F794C2ED877A159ULL, 0xA3B6C7E693A24C69ULL, 0x893ABA425419BC27ULL}}},
    // [11] = 11G
    {{{0xBBEC17895DA008CBULL, 0x5649980BE5C17891ULL, 0x5EF4246B70C65AACULL, 0x774AE7F858A9411EULL}},
     {{0x301D74C9C953C61BULL, 0x372DB1E2DFF9D6A8ULL, 0x0243DD56D7B7B365ULL, 0xD984A032EB6B5E19ULL}}},
    // [12] = 12G
    {{{0xC5B0F47070AFE85AULL, 0x687CF4419620095BULL, 0x15C38F004D734633ULL, 0xD01115D548E7561BULL}},
     {{0x6B051B13F4062327ULL, 0x79238C5DD9A86D52ULL, 0xA8B64537E17BD815ULL, 0xA9F34FFDC815E0D7ULL}}},
    // [13] = 13G
    {{{0xDEEDDF8F19405AA8ULL, 0xB075FBC6610E58CDULL, 0xC7D1D205C3748651ULL, 0xF28773C2D975288BULL}},
     {{0x29B5CB52DB03ED81ULL, 0x3A1A06DA521FA91FULL, 0x758212EB65CDAF47ULL, 0x0AB0902E8D880A89ULL}}},
    // [14] = 14G
    {{{0xE49B241A60E823E4ULL, 0x26AA7B63678949E6ULL, 0xFD64E67F07D38E32ULL, 0x499FDF9E895E719CULL}},
     {{0xC65F40D403A13F5BULL, 0x464279C27A3F95BCULL, 0x90F044E4A7B3D464ULL, 0xCAC2F6C4B54E8551ULL}}},
    // [15] = 15G
    {{{0x44ADBCF8E27E080EULL, 0x31E5946F3C85F79EULL, 0x5A465AE3095FF411ULL, 0xD7924D4F7D43EA96ULL}},
     {{0xC504DC9FF6A26B58ULL, 0xEA40AF2BD896D3A5ULL, 0x83842EC228CC6DEFULL, 0x581E2872A86C72A6ULL}}},
};

// -- Optimized Generator Scalar Multiplication with constant table ------------
// Uses GENERATOR_TABLE_AFFINE in __constant__ memory (no build_generator_table needed).
// Fixed-window w=4: 252 doublings + <=64 mixed additions.
// Saves shared-memory allocation and __syncthreads() compared to runtime table.
//
// NOTE: For signing paths prefer scalar_mul_generator_w8 (w=8, 32 windows, ~198 ns/op).
// This function (w=4, 64 windows, ~220 ns/op) is retained for audit and benchmark
// comparisons that need the original reference implementation.
__device__ inline void scalar_mul_generator_const(const Scalar* k, JacobianPoint* r) {
    r->infinity = true;
    field_set_zero(&r->x);
    field_set_one(&r->y);
    field_set_zero(&r->z);

    bool started = false;

    #pragma unroll 1
    for (int limb = 3; limb >= 0; limb--) {
        uint64_t w = k->limbs[limb];
        #pragma unroll 1
        for (int nib = 15; nib >= 0; nib--) {
            uint32_t idx = (uint32_t)((w >> (nib * 4)) & 0xFULL);

            if (started) {
                // Unchecked: started=true guarantees r->infinity=false.
                jacobian_double_unchecked(r, r);
                jacobian_double_unchecked(r, r);
                jacobian_double_unchecked(r, r);
                jacobian_double_unchecked(r, r);
            }

            if (idx != 0) {
                if (!started) {
                    r->x = GENERATOR_TABLE_AFFINE[idx].x;
                    r->y = GENERATOR_TABLE_AFFINE[idx].y;
                    field_set_one(&r->z);
                    r->infinity = false;
                    started = true;
                } else {
                    // Unchecked: started=true guarantees r->infinity=false.
                    jacobian_add_mixed_unchecked(r, &GENERATOR_TABLE_AFFINE[idx], r);
                }
            }
        }
    }
}

// -- Optimized Generator Scalar Multiplication with w=8 constant table --------
// Uses GENERATOR_TABLE_W8 in __constant__ memory (256 entries, 16 KB).
// Fixed-window w=8: 248 doublings + <=32 mixed additions.
// ~1.8x fewer additions than w=4 (32 vs 64 windows).
__device__ inline void scalar_mul_generator_w8(const Scalar* k, JacobianPoint* r) {
    r->infinity = true;
    field_set_zero(&r->x);
    field_set_one(&r->y);
    field_set_zero(&r->z);

    bool started = false;

    // Process scalar 8 bits at a time (32 windows of 8 bits)
    #pragma unroll 1
    for (int limb = 3; limb >= 0; limb--) {
        uint64_t w = k->limbs[limb];
        #pragma unroll 1
        for (int byte_idx = 7; byte_idx >= 0; byte_idx--) {
            uint32_t idx = (uint32_t)((w >> (byte_idx * 8)) & 0xFFULL);

            if (started) {
                // Unchecked: started=true guarantees r->infinity=false.
                jacobian_double_unchecked(r, r);
                jacobian_double_unchecked(r, r);
                jacobian_double_unchecked(r, r);
                jacobian_double_unchecked(r, r);
                jacobian_double_unchecked(r, r);
                jacobian_double_unchecked(r, r);
                jacobian_double_unchecked(r, r);
                jacobian_double_unchecked(r, r);
            }

            if (idx != 0) {
                if (!started) {
                    r->x = GENERATOR_TABLE_W8[idx].x;
                    r->y = GENERATOR_TABLE_W8[idx].y;
                    field_set_one(&r->z);
                    r->infinity = false;
                    started = true;
                } else {
                    // Unchecked: started=true guarantees r->infinity=false.
                    jacobian_add_mixed_unchecked(r, &GENERATOR_TABLE_W8[idx], r);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Constant-time generator scalar multiplication: r = k * G
// Fixed-window w=8, branchless, no warp divergence.
// Required for signing (ECDSA nonce k*G, Schnorr nonce k*G, key generation).
//
// Technique — "dummy-start + unchecked" (lower cost than checked-add version):
//   1. Initialize r = G (not infinity), so the accumulator is NEVER at infinity
//      throughout the main loop. This lets us use _unchecked variants everywhere.
//   2. Use jacobian_double_unchecked + jacobian_add_mixed_unchecked in the loop:
//      saves the if(infinity) check on every double and add (32 checks eliminated).
//   3. safe_idx = idx | (idx==0): always read a valid table entry (no branch).
//   4. Branchless cmov selects r_add when idx!=0, r otherwise (CUDA SELP insn).
//   5. After the loop r == (k+1)*G, so subtract G once (negate y, unchecked add).
//      The subtracted G is window 0 of the table (= 1*G exactly).
//
// Cost: 32 unchecked doubles×8 + 32 unchecked adds + 32 cmovs + 1 final sub
//       vs previous: 32 checked doubles×8 + 32 checked adds + 32 cmovs
//       Savings: ~8% by eliminating 256+32 = 288 infinity checks.
// ---------------------------------------------------------------------------
__device__ inline void scalar_mul_generator_ct(const Scalar* k, JacobianPoint* r) {
    // Start at G (Z=1, not at infinity) — eliminates all infinity checks in loop.
    r->x.limbs[0] = GENERATOR_X[0]; r->x.limbs[1] = GENERATOR_X[1];
    r->x.limbs[2] = GENERATOR_X[2]; r->x.limbs[3] = GENERATOR_X[3];
    r->y.limbs[0] = GENERATOR_Y[0]; r->y.limbs[1] = GENERATOR_Y[1];
    r->y.limbs[2] = GENERATOR_Y[2]; r->y.limbs[3] = GENERATOR_Y[3];
    field_set_one(&r->z);
    r->infinity = false;

    JacobianPoint r_add;
    #pragma unroll 1
    for (int limb = 3; limb >= 0; limb--) {
        uint64_t w = k->limbs[limb];
        #pragma unroll 1
        for (int byte_idx = 7; byte_idx >= 0; byte_idx--) {
            uint32_t idx = (uint32_t)((w >> (byte_idx * 8)) & 0xFFULL);

            // Always double 8x — unchecked: r is guaranteed non-infinity.
            jacobian_double_unchecked(r, r);
            jacobian_double_unchecked(r, r);
            jacobian_double_unchecked(r, r);
            jacobian_double_unchecked(r, r);
            jacobian_double_unchecked(r, r);
            jacobian_double_unchecked(r, r);
            jacobian_double_unchecked(r, r);
            jacobian_double_unchecked(r, r);

            // safe_idx: map 0 -> 1 so we always read a valid table entry.
            uint32_t safe_idx = idx | (uint32_t)(idx == 0);

            // Always compute the addition — unchecked: r is guaranteed non-infinity.
            jacobian_add_mixed_unchecked(r, &GENERATOR_TABLE_W8[safe_idx], &r_add);

            // Branchless select: r = (idx != 0) ? r_add : r
            uint64_t mask = -(uint64_t)(idx != 0);
            jacobian_cmov(r, &r_add, mask);
        }
    }

    // r == (k+1)*G — subtract the dummy G we started with.
    // Negate G.y and do one unchecked mixed add: r = r + (-G) = r - G = k*G.
    AffinePoint neg_G;
    neg_G.x.limbs[0] = GENERATOR_X[0]; neg_G.x.limbs[1] = GENERATOR_X[1];
    neg_G.x.limbs[2] = GENERATOR_X[2]; neg_G.x.limbs[3] = GENERATOR_X[3];
    FieldElement G_y;
    G_y.limbs[0] = GENERATOR_Y[0]; G_y.limbs[1] = GENERATOR_Y[1];
    G_y.limbs[2] = GENERATOR_Y[2]; G_y.limbs[3] = GENERATOR_Y[3];
    field_negate(&G_y, &neg_G.y);
    jacobian_add_mixed_unchecked(r, &neg_G, r);
}

// ============================================================================
// GLV + CT generator multiplication  (constant-time, ~35-40% faster)
// ---------------------------------------------------------------------------
// Decomposes k = k1 + k2*λ (GLV) where |k1|,|k2| ≤ 2^128, then processes
// both halves in a shared 16-step W8 doubling chain:
//   16 iterations × (8 unchecked doublings + 2 CT mixed adds)
//   = 128 doublings + 32 mixed additions (vs 256 + 32 in scalar_mul_generator_ct)
//
// GLV sign flags (k1_neg, k2_neg) are applied via CT bitmasks — no branches
// on secret data in the main loop.
//
// Dummy-start technique: start accumulator at D = G+endo(G) (precomputed in
// __constant__ memory — DUMMY_GLV_X/Y). Subtract D at end → net result k*G.
// No field_inv in hot path — fully amortized. Accumulator is never at infinity.
//
// Scalar blinding (DPA defense): projective coordinates are randomized before
// the main loop using GPU clock + thread-ID entropy. Each invocation maps
// (X:Y:Z) → (r²X : r³Y : rZ), same mathematical point but different values
// → prevents power/cache correlation across repeated invocations.
// Cost: 1 field_to_mont + 1 field_sqr + 2 field_mul = 4 field ops.
// ============================================================================
__device__ inline void scalar_mul_generator_ct_glv(const Scalar* k, JacobianPoint* r) {
    // GLV decompose: k = k1 + k2*lambda, |k1|, |k2| <= 2^128
    GLVDecomposition decomp = glv_decompose(k);

    // β for endomorphism: endo(P) = (β·x, y)
    FieldElement beta_fe;
    beta_fe.limbs[0] = BETA[0]; beta_fe.limbs[1] = BETA[1];
    beta_fe.limbs[2] = BETA[2]; beta_fe.limbs[3] = BETA[3];

    // -- Dummy D = G + endo(G) loaded from __constant__ memory (no field_inv) --
    // Precomputed offline: D.x = DUMMY_GLV_X, D.y = DUMMY_GLV_Y,
    // -D.y = DUMMY_GLV_NEG_Y (= G.y, verified property of this curve point).
    AffinePoint dummy_affine, neg_dummy_affine;
    for (int i = 0; i < 4; i++) {
        dummy_affine.x.limbs[i]     = DUMMY_GLV_X[i];
        dummy_affine.y.limbs[i]     = DUMMY_GLV_Y[i];
        neg_dummy_affine.x.limbs[i] = DUMMY_GLV_X[i];
        neg_dummy_affine.y.limbs[i] = DUMMY_GLV_NEG_Y[i];
    }

    // Start accumulator at D (Z=1, guaranteed non-infinity).
    r->x = dummy_affine.x;
    r->y = dummy_affine.y;
    field_set_one(&r->z);
    r->infinity = false;

    // -- Scalar blinding (DPA defense): randomize projective representation --
    // (r²·X : r³·Y : r) represents the same geometric point as (X:Y:1).
    // r is derived from GPU clock + thread/block IDs (not cryptographically
    // strong, but sufficient to prevent power-analysis correlation across runs).
    {
        uint64_t entropy = ((uint64_t)clock64())
                         ^ ((uint64_t)(threadIdx.x + 1))
                         ^ ((uint64_t)(blockIdx.x + 1) << 32);
        entropy |= 1ULL;  // ensure non-zero
        FieldElement rr_raw;
        rr_raw.limbs[0] = entropy;
        rr_raw.limbs[1] = rr_raw.limbs[2] = rr_raw.limbs[3] = 0;
        FieldElement rr, rr2, rr3, tmp;
        field_to_mont(&rr_raw, &rr);       // convert to Montgomery domain
        field_sqr(&rr, &rr2);              // rr^2
        field_mul(&rr2, &rr, &rr3);        // rr^3
        field_mul(&r->x, &rr2, &tmp); r->x = tmp;   // X ← r²·X
        field_mul(&r->y, &rr3, &tmp); r->y = tmp;   // Y ← r³·Y
        r->z = rr;                         // Z ← r
    }

    // CT sign masks: all-1s if negated, all-0s if positive.
    uint64_t k1_sign = -(uint64_t)decomp.k1_neg;
    uint64_t k2_sign = -(uint64_t)decomp.k2_neg;

    JacobianPoint r_add;

    // -- Main loop: 16 iterations (2 limbs × 8 bytes), MSB first --
    // k1 and k2 are ~128-bit scalars; only limbs[0..1] are significant.
    #pragma unroll 1
    for (int limb = 1; limb >= 0; limb--) {
        uint64_t w1 = decomp.k1.limbs[limb];
        uint64_t w2 = decomp.k2.limbs[limb];
        #pragma unroll 1
        for (int byte_idx = 7; byte_idx >= 0; byte_idx--) {
            uint32_t idx1 = (uint32_t)((w1 >> (byte_idx * 8)) & 0xFFULL);
            uint32_t idx2 = (uint32_t)((w2 >> (byte_idx * 8)) & 0xFFULL);

            // 8 unchecked doublings — accumulator is never at infinity.
            jacobian_double_unchecked(r, r);
            jacobian_double_unchecked(r, r);
            jacobian_double_unchecked(r, r);
            jacobian_double_unchecked(r, r);
            jacobian_double_unchecked(r, r);
            jacobian_double_unchecked(r, r);
            jacobian_double_unchecked(r, r);
            jacobian_double_unchecked(r, r);

            // k1: add idx1*G from table with CT sign adjustment.
            {
                uint32_t safe_idx1 = idx1 | (uint32_t)(idx1 == 0);
                AffinePoint p1;
                p1.x = GENERATOR_TABLE_W8[safe_idx1].x;
                FieldElement raw_y1 = GENERATOR_TABLE_W8[safe_idx1].y;
                FieldElement neg_y1;
                field_negate(&raw_y1, &neg_y1);
                for (int i = 0; i < 4; i++) {
                    p1.y.limbs[i] = (k1_sign & neg_y1.limbs[i])
                                  | (~k1_sign & raw_y1.limbs[i]);
                }
                jacobian_add_mixed_unchecked(r, &p1, &r_add);
                uint64_t mask1 = -(uint64_t)(idx1 != 0);
                jacobian_cmov(r, &r_add, mask1);
            }

            // k2: add idx2*endo(G) with CT sign adjustment.
            // endo table is derived on-the-fly: x' = β·x, y' = y (or -y).
            {
                uint32_t safe_idx2 = idx2 | (uint32_t)(idx2 == 0);
                AffinePoint p2;
                field_mul(&GENERATOR_TABLE_W8[safe_idx2].x, &beta_fe, &p2.x);
                FieldElement raw_y2 = GENERATOR_TABLE_W8[safe_idx2].y;
                FieldElement neg_y2;
                field_negate(&raw_y2, &neg_y2);
                for (int i = 0; i < 4; i++) {
                    p2.y.limbs[i] = (k2_sign & neg_y2.limbs[i])
                                  | (~k2_sign & raw_y2.limbs[i]);
                }
                jacobian_add_mixed_unchecked(r, &p2, &r_add);
                uint64_t mask2 = -(uint64_t)(idx2 != 0);
                jacobian_cmov(r, &r_add, mask2);
            }
        }
    }

    // r = (k1+1)*G + (k2+1)*endo(G) — subtract dummy to recover k*G.
    jacobian_add_mixed_unchecked(r, &neg_dummy_affine, r);
}

// -- Ultra-fast Generator Multiplication via 16x65536 LUT --------------------
// Precomputed table: 16 windows x 65536 affine points in global memory (64 MB).
// For window i: table[i][j] = j * 2^(16*i) * G  (j = 0..65535)
// k*G = sum of 16 table lookups = 15 mixed additions, ZERO doublings.
// Table must be built once at init via build_generator_lut kernels.
// Reference: https://bitcointalk.org/index.php?topic=5396293.0
#define GEN_LUT_WINDOWS    16
#define GEN_LUT_WINDOW_BITS 16
#define GEN_LUT_ENTRIES    65536

__device__ inline void scalar_mul_generator_lut(
    const Scalar* k,
    const AffinePoint* __restrict__ lut,
    JacobianPoint* r)
{
    r->infinity = true;

    #pragma unroll 1
    for (int win = 0; win < GEN_LUT_WINDOWS; win++) {
        uint32_t idx = (uint32_t)((k->limbs[win >> 2] >> ((win & 3) << 4)) & 0xFFFF);

        if (idx != 0) {
            const AffinePoint* pt = &lut[(uint32_t)win * GEN_LUT_ENTRIES + idx];
            if (r->infinity) {
                r->x = pt->x;
                r->y = pt->y;
                field_set_one(&r->z);
                r->infinity = false;
            } else {
                // Unchecked: r->infinity is guaranteed false after the first window.
                // Mirrors the same optimization applied to OpenCL in fc378bdc.
                jacobian_add_mixed_unchecked(r, pt, r);
            }
        }
    }

    if (r->infinity) {
        field_set_zero(&r->x);
        field_set_one(&r->y);
        field_set_zero(&r->z);
    }
}

// ============================================================================
// Byte <-> Scalar/Field conversion (big-endian bytes <-> LE uint64_t limbs)
// ============================================================================

// Convert 32 big-endian bytes to a Scalar (reduced mod n).
__device__ inline void scalar_from_bytes(const uint8_t bytes[32], Scalar* r) {
    for (int i = 0; i < 4; i++) {
        uint64_t limb = 0;
        int base = (3 - i) * 8;
        for (int j = 0; j < 8; j++) {
            limb = (limb << 8) | bytes[base + j];
        }
        r->limbs[i] = limb;
    }
    uint64_t borrow = 0;
    uint64_t tmp[4];
    for (int i = 0; i < 4; i++) {
        tmp[i] = sub_cc(r->limbs[i], ORDER[i], borrow);
    }
    uint64_t mask = -(uint64_t)(borrow == 0);
    for (int i = 0; i < 4; i++) {
        r->limbs[i] = (tmp[i] & mask) | (r->limbs[i] & ~mask);
    }
}

// Convert Scalar to 32 big-endian bytes.
__device__ inline void scalar_to_bytes(const Scalar* s, uint8_t bytes[32]) {
    for (int i = 0; i < 4; i++) {
        uint64_t limb = s->limbs[3 - i];
        for (int j = 0; j < 8; j++) {
            bytes[i * 8 + j] = (uint8_t)(limb >> (56 - j * 8));
        }
    }
}

// Reduce a FieldElement to canonical form [0, p) using three branchless
// conditional subtractions.  Handles magnitude <= 4.  A single subtraction
// is insufficient when intermediate field computations produce values >= 2p
// (issue #225).
__device__ inline void field_normalize(const FieldElement* fe, uint64_t cur[4]) {
    constexpr uint64_t P[4] = {
        0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
    };
    for (int i = 0; i < 4; i++) cur[i] = fe->limbs[i];
    for (int pass = 0; pass < 3; pass++) {
        uint64_t tmp[4];
        uint64_t borrow = 0;
        for (int i = 0; i < 4; i++) {
            uint64_t a = cur[i], b = P[i] + borrow;
            uint64_t underflow = (borrow && b == 0) ? 1ULL : 0ULL;
            tmp[i] = a - b;
            borrow = (a < b || underflow) ? 1ULL : 0ULL;
        }
        uint64_t mask = (borrow == 0) ? ~0ULL : 0ULL;
        for (int i = 0; i < 4; i++)
            cur[i] = (tmp[i] & mask) | (cur[i] & ~mask);
    }
}

// Convert FieldElement to 32 big-endian bytes (normalizes mod p first).
__device__ inline void field_to_bytes(const FieldElement* fe, uint8_t bytes[32]) {
    uint64_t cur[4];
    field_normalize(fe, cur);
    for (int i = 0; i < 4; i++) {
        uint64_t limb = cur[3 - i];
        for (int j = 0; j < 8; j++) {
            bytes[i * 8 + j] = (uint8_t)(limb >> (56 - j * 8));
        }
    }
}

// Return whether a field element is odd after normalization mod p.
__device__ __forceinline__ bool field_is_odd(const FieldElement* fe) {
    uint64_t cur[4];
    field_normalize(fe, cur);
    return (cur[0] & 1ULL) != 0;
}

#endif

// ============================================================================
// Missing operations parity with CPU library
// ============================================================================

// -- Field: from_bytes (32 BE bytes -> FieldElement) -------------------------
// Inverse of field_to_bytes. Does NOT reduce mod p (caller ensures valid).
__device__ inline void field_from_bytes(const uint8_t bytes[32], FieldElement* r) {
    for (int i = 0; i < 4; i++) {
        uint64_t limb = 0;
        int base = (3 - i) * 8;
        for (int j = 0; j < 8; j++)
            limb = (limb << 8) | bytes[base + j];
        r->limbs[i] = limb;
    }
}

// -- Field: from_bytes strict (reject >= p) ----------------------------------
__device__ inline bool field_from_bytes_strict(const uint8_t bytes[32], FieldElement* r) {
    field_from_bytes(bytes, r);
    // Check r < p by trying r - p; if no borrow, r >= p => invalid
    uint64_t borrow = 0;
    for (int i = 0; i < 4; i++) {
        (void)sub_cc(r->limbs[i], MODULUS[i], borrow);
    }
    return borrow != 0;  // valid iff r < p (borrow occurred)
}

// -- Field: from_uint64 ------------------------------------------------------
__device__ __forceinline__ void field_from_uint64(uint64_t v, FieldElement* r) {
    r->limbs[0] = v;
    r->limbs[1] = 0;
    r->limbs[2] = 0;
    r->limbs[3] = 0;
}

// -- Field: half (a/2 mod p) -------------------------------------------------
// If a is even: r = a/2. If odd: r = (a + p) / 2.
__device__ inline void field_half(const FieldElement* a, FieldElement* r) {
    uint64_t odd = a->limbs[0] & 1;
    // Conditionally add p
    uint64_t carry = 0;
    uint64_t tmp[4];
    uint64_t mask = -(uint64_t)odd;  // all-1s if odd, all-0s if even
    for (int i = 0; i < 4; i++) {
        tmp[i] = add_cc(a->limbs[i], (MODULUS[i] & mask), carry);
    }
    // Right-shift by 1
    r->limbs[0] = (tmp[0] >> 1) | (tmp[1] << 63);
    r->limbs[1] = (tmp[1] >> 1) | (tmp[2] << 63);
    r->limbs[2] = (tmp[2] >> 1) | (tmp[3] << 63);
    r->limbs[3] = (tmp[3] >> 1) | ((uint64_t)carry << 63);
}

// -- Scalar: from_uint64 -----------------------------------------------------
__device__ __forceinline__ void scalar_from_uint64(uint64_t v, Scalar* r) {
    r->limbs[0] = v;
    r->limbs[1] = 0;
    r->limbs[2] = 0;
    r->limbs[3] = 0;
}

// -- Scalar: is_high (s > n/2) -----------------------------------------------
// Requires HALF_ORDER from ecdsa.cuh; use scalar_ge if standalone.
// Re-implemented inline to avoid header dependency.
__device__ __forceinline__ bool scalar_is_high(const Scalar* s) {
    // n/2 = 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A0
    static const uint64_t HALF_N[4] = {
        0xDFE92F46681B20A0ULL, 0x5D576E7357A4501DULL,
        0xFFFFFFFFFFFFFFFFULL, 0x7FFFFFFFFFFFFFFFULL
    };
    for (int i = 3; i >= 0; i--) {
        if (s->limbs[i] > HALF_N[i]) return true;
        if (s->limbs[i] < HALF_N[i]) return false;
    }
    return false; // equal to n/2 is not high
}

// -- Scalar: parse_bytes_strict (reject >= n) --------------------------------
__device__ inline bool scalar_from_bytes_strict(const uint8_t bytes[32], Scalar* r) {
    scalar_from_bytes(bytes, r);
    // Check original value was < n by reparsing without reduction
    Scalar raw;
    for (int i = 0; i < 4; i++) {
        uint64_t limb = 0;
        int base = (3 - i) * 8;
        for (int j = 0; j < 8; j++)
            limb = (limb << 8) | bytes[base + j];
        raw.limbs[i] = limb;
    }
    // If raw != r, then reduction happened => original >= n
    return scalar_eq(&raw, r);
}

// -- Scalar: parse_bytes_strict_nonzero (reject >= n or == 0) ----------------
__device__ inline bool scalar_from_bytes_strict_nonzero(const uint8_t bytes[32], Scalar* r) {
    if (!scalar_from_bytes_strict(bytes, r)) return false;
    return !scalar_is_zero(r);
}

// -- Scalar: half (a/2 mod n) ------------------------------------------------
__device__ inline void scalar_half(const Scalar* a, Scalar* r) {
    uint64_t odd = a->limbs[0] & 1;
    uint64_t carry = 0;
    uint64_t tmp[4];
    uint64_t mask = -(uint64_t)odd;
    for (int i = 0; i < 4; i++) {
        tmp[i] = add_cc(a->limbs[i], (ORDER[i] & mask), carry);
    }
    r->limbs[0] = (tmp[0] >> 1) | (tmp[1] << 63);
    r->limbs[1] = (tmp[1] >> 1) | (tmp[2] << 63);
    r->limbs[2] = (tmp[2] >> 1) | (tmp[3] << 63);
    r->limbs[3] = (tmp[3] >> 1) | ((uint64_t)carry << 63);
}

// -- Point: jacobian_to_affine -----------------------------------------------
// Convert Jacobian (X, Y, Z) to affine (x, y) where x=X/Z^2, y=Y/Z^3.
// If p is infinity, sets out to (0,0) and returns false.
__device__ inline bool jacobian_to_affine(const JacobianPoint* p,
                                          FieldElement* out_x, FieldElement* out_y) {
    if (p->infinity) {
        field_set_zero(out_x);
        field_set_zero(out_y);
        return false;
    }
    FieldElement z_inv, z_inv2, z_inv3;
    field_inv(&p->z, &z_inv);
    field_sqr(&z_inv, &z_inv2);
    field_mul(&z_inv, &z_inv2, &z_inv3);
    field_mul(&p->x, &z_inv2, out_x);
    field_mul(&p->y, &z_inv3, out_y);
    return true;
}

// -- Point: negate -----------------------------------------------------------
__device__ inline void point_negate(const JacobianPoint* p, JacobianPoint* r) {
    r->x = p->x;
    field_negate(&p->y, &r->y);
    r->z = p->z;
    r->infinity = p->infinity;
}

// -- Point: subtract (P - Q = P + (-Q)) -------------------------------------
__device__ inline void point_sub(const JacobianPoint* p, const JacobianPoint* q,
                                 JacobianPoint* r) {
    JacobianPoint neg_q;
    point_negate(q, &neg_q);
    jacobian_add(p, &neg_q, r);
}

// -- Point: has_even_y -------------------------------------------------------
// Returns true if the affine Y coordinate is even.
// Requires a field inversion to convert from Jacobian.
__device__ inline bool point_has_even_y(const JacobianPoint* p) {
    if (p->infinity) return false;
    FieldElement ax, ay;
    jacobian_to_affine(p, &ax, &ay);
    return !field_is_odd(&ay);
}

// -- Point: to_compressed (33 bytes: 0x02/0x03 || x) ------------------------
__device__ inline bool point_to_compressed(const JacobianPoint* p, uint8_t out[33]) {
    if (p->infinity) return false;
    FieldElement ax, ay;
    jacobian_to_affine(p, &ax, &ay);
    out[0] = field_is_odd(&ay) ? 0x03 : 0x02;
    field_to_bytes(&ax, out + 1);
    return true;
}

// -- Point: from_compressed (33 bytes -> JacobianPoint) ----------------------
// Uses lift_x + parity selection. Returns false if x not on curve.
__device__ inline bool point_from_compressed(const uint8_t data[33], JacobianPoint* p) {
    uint8_t prefix = data[0];
    if (prefix != 0x02 && prefix != 0x03) return false;
    bool want_odd_y = (prefix == 0x03);

    // Parse x
    FieldElement x;
    field_from_bytes(data + 1, &x);

    // y^2 = x^3 + 7
    FieldElement x2, x3, y2, y;
    field_sqr(&x, &x2);
    field_mul(&x2, &x, &x3);
    FieldElement seven;
    field_from_uint64(7, &seven);
    field_add(&x3, &seven, &y2);

    field_sqrt(&y2, &y);

    // Verify sqrt is correct
    FieldElement y_check;
    field_sqr(&y, &y_check);
    uint8_t yc_bytes[32], y2_bytes[32];
    field_to_bytes(&y_check, yc_bytes);
    field_to_bytes(&y2, y2_bytes);
    for (int i = 0; i < 32; i++) {
        if (yc_bytes[i] != y2_bytes[i]) return false;
    }

    // Adjust parity
    uint8_t y_bytes[32];
    field_to_bytes(&y, y_bytes);
    bool y_is_odd = (y_bytes[31] & 1) != 0;
    if (y_is_odd != want_odd_y) {
        FieldElement zero;
        field_set_zero(&zero);
        field_sub(&zero, &y, &y);
    }

    p->x = x;
    p->y = y;
    field_set_one(&p->z);
    p->infinity = false;
    return true;
}

// -- Point: to_uncompressed (65 bytes: 0x04 || x || y) ----------------------
__device__ inline bool point_to_uncompressed(const JacobianPoint* p, uint8_t out[65]) {
    if (p->infinity) return false;
    FieldElement ax, ay;
    jacobian_to_affine(p, &ax, &ay);
    out[0] = 0x04;
    field_to_bytes(&ax, out + 1);
    field_to_bytes(&ay, out + 33);
    return true;
}

// -- Point: from_uncompressed (65 bytes -> JacobianPoint) --------------------
__device__ inline bool point_from_uncompressed(const uint8_t data[65], JacobianPoint* p) {
    if (data[0] != 0x04) return false;

    FieldElement x, y;
    field_from_bytes(data + 1, &x);
    field_from_bytes(data + 33, &y);

    // Verify on curve: y^2 == x^3 + 7
    FieldElement y2, x2, x3, rhs, seven;
    field_sqr(&y, &y2);
    field_sqr(&x, &x2);
    field_mul(&x2, &x, &x3);
    field_from_uint64(7, &seven);
    field_add(&x3, &seven, &rhs);

    uint8_t y2_bytes[32], rhs_bytes[32];
    field_to_bytes(&y2, y2_bytes);
    field_to_bytes(&rhs, rhs_bytes);
    for (int i = 0; i < 32; i++) {
        if (y2_bytes[i] != rhs_bytes[i]) return false;
    }

    p->x = x;
    p->y = y;
    field_set_one(&p->z);
    p->infinity = false;
    return true;
}

// -- Point: x_only_bytes (32 bytes, BIP-340 format) --------------------------
__device__ inline bool point_x_only_bytes(const JacobianPoint* p, uint8_t out[32]) {
    if (p->infinity) return false;
    FieldElement ax, ay;
    jacobian_to_affine(p, &ax, &ay);
    field_to_bytes(&ax, out);
    return true;
}

// -- Point: x_bytes_and_parity -----------------------------------------------
// Returns x as 32 bytes and whether y is odd.
__device__ inline bool point_x_bytes_and_parity(const JacobianPoint* p,
                                                 uint8_t x_out[32], bool* y_is_odd) {
    if (p->infinity) return false;
    FieldElement ax, ay;
    jacobian_to_affine(p, &ax, &ay);
    field_to_bytes(&ax, x_out);
    *y_is_odd = field_is_odd(&ay);
    return true;
}

// ============================================================================
// FROST Partial Signature Verification (device helper + batch kernel)
// ============================================================================
//
// Each thread verifies one partial signature:
//   R_i = D_i + rho_i * E_i
//   lhs = z_i * G
//   rhs = R_i + lambda_i_e * Y_i
//   result = (lhs == rhs)
//
// Device helper — called from the __global__ batch kernel in secp256k1.cu.
__device__ inline uint8_t frost_verify_partial_device(
    const uint8_t* __restrict__ z_i_bytes,
    const uint8_t* __restrict__ D_i_bytes,
    const uint8_t* __restrict__ E_i_bytes,
    const uint8_t* __restrict__ Y_i_bytes,
    const uint8_t* __restrict__ rho_i_bytes,
    const uint8_t* __restrict__ lambda_i_e_bytes,
    uint8_t negate_R_flag,
    uint8_t negate_key_flag)
{
    // Parse scalars
    Scalar z_i, rho_i, lambda_i_e;
    scalar_from_bytes(z_i_bytes,        &z_i);
    scalar_from_bytes(rho_i_bytes,      &rho_i);
    scalar_from_bytes(lambda_i_e_bytes, &lambda_i_e);

    // Decompress input points
    JacobianPoint D_pt, E_pt, Y_pt;
    bool ok  = point_from_compressed(D_i_bytes, &D_pt);
    ok      &= point_from_compressed(E_i_bytes, &E_pt);
    ok      &= point_from_compressed(Y_i_bytes, &Y_pt);
    if (!ok) return 0;

    // R_i = D_i + rho_i * E_i
    JacobianPoint rho_E;
    scalar_mul_glv(&E_pt, &rho_i, &rho_E);
    JacobianPoint R_i;
    jacobian_add(&D_pt, &rho_E, &R_i);

    // Optionally negate R_i (even-y convention)
    if (negate_R_flag) {
        field_negate(&R_i.y, &R_i.y);
    }

    // lhs = z_i * G
    JacobianPoint lhs;
    scalar_mul(&GENERATOR_JACOBIAN, &z_i, &lhs);

    // Optionally negate Y_i
    if (negate_key_flag) {
        field_negate(&Y_pt.y, &Y_pt.y);
    }

    // rhs = R_i + lambda_i_e * Y_i
    JacobianPoint lambda_Y;
    scalar_mul_glv(&Y_pt, &lambda_i_e, &lambda_Y);
    JacobianPoint rhs;
    jacobian_add(&R_i, &lambda_Y, &rhs);

    // Compare lhs == rhs via affine x + y-parity
    uint8_t lhs_xb[32], rhs_xb[32];
    bool lhs_odd, rhs_odd;
    if (!point_x_bytes_and_parity(&lhs, lhs_xb, &lhs_odd)) return 0;
    if (!point_x_bytes_and_parity(&rhs, rhs_xb, &rhs_odd)) return 0;

    if (lhs_odd != rhs_odd) return 0;
    for (int b = 0; b < 32; ++b)
        if (lhs_xb[b] != rhs_xb[b]) return 0;
    return 1;
}

// ============================================================================
// Batch Jacobian -> Compressed (device helper)
// ============================================================================
// Converts one JacobianPoint to 33-byte SEC1 compressed form at out33.
// The __global__ batch kernel lives in secp256k1.cu.
__device__ inline void jacobian_to_compressed_device(
    const JacobianPoint* __restrict__ p,
    uint8_t* __restrict__ out33)
{
    if (p->infinity) {
        for (int b = 0; b < 33; ++b) out33[b] = 0;
        return;
    }
    FieldElement ax, ay;
    jacobian_to_affine(p, &ax, &ay);
    out33[0] = field_is_odd(&ay) ? 0x03 : 0x02;
    field_to_bytes(&ax, out33 + 1);
}

} // namespace cuda
} // namespace secp256k1

