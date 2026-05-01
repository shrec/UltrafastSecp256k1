#pragma once
#include <cuda_runtime.h>
#include <cstdint>


// 32-bit implementation of FieldElement (8x32-bit)
struct FieldElement {
    uint32_t limbs[8];
};

struct Scalar {
    uint32_t limbs[8];
};

struct JacobianPoint {
    FieldElement x;
    FieldElement y;
    FieldElement z;
    bool infinity;
};

struct AffinePoint {
    FieldElement x;
    FieldElement y;
};

// P = 2^256 - 2^32 - 977
__constant__ static const uint32_t MODULUS[8] = {
    0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

// N (Order)
__constant__ static const uint32_t ORDER[8] = {
    0xD0364141, 0xBFD25E8C, 0xAF48A03B, 0xBAAEDCE6,
    0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

// Endomorphism BETA (Cube root of unity)
__constant__ static const uint32_t BETA[8] = {
    0x719501EE, 0xC1396C28,
    0x12F58995, 0x9CF04975,
    0xAC3434E9, 0x6E64479E,
    0x657C0710, 0x7AE96A2B
};

// GLV Lambda
__constant__ static const uint32_t LAMBDA[8] = {
    0x1B23BD72, 0xDF02967C, 0x20816678, 0x122E22EA, 
    0x8812645A, 0xA5261C02, 0xC05C30E0, 0x5363AD4C
};

// Generator Points (Standard Form)
__constant__ static const uint32_t GENERATOR_X[8] = {
    0x16F81798, 0x59F2815B, 0x2DCE28D9, 0x029BFCDB, 
    0xCE870B07, 0x55A06295, 0xF9DCBBAC, 0x79BE667E
};

__constant__ static const uint32_t GENERATOR_Y[8] = {
    0xFB10D4B8, 0x9C47D08F, 0xA6855419, 0xFD17B448, 
    0x0E1108A8, 0x5DA4FBFC, 0x26A3C465, 0x483ADA77
};

// Generator Point (Jacobian, Standard Form)
__constant__ const JacobianPoint GENERATOR_JACOBIAN = {
    { {0x16F81798, 0x59F2815B, 0x2DCE28D9, 0x029BFCDB, 0xCE870B07, 0x55A06295, 0xF9DCBBAC, 0x79BE667E} },
    { {0xFB10D4B8, 0x9C47D08F, 0xA6855419, 0xFD17B448, 0x0E1108A8, 0x5DA4FBFC, 0x26A3C465, 0x483ADA77} },
    { {1, 0, 0, 0, 0, 0, 0, 0} },
    false
};

// R = 2^256 mod P
__constant__ static const uint32_t FIELD_R[8] = {
    0x000003D1, 0x00000001, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000
};

// R^2
__constant__ static const uint32_t FIELD_R2[8] = {
    0x000E90A1, 0x000007A2, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000
};

// One (Standard)
__constant__ static const uint32_t FIELD_ONE[8] = {
    1, 0, 0, 0, 0, 0, 0, 0
};

// Inverse of -P mod 2^32
static constexpr uint32_t FIELD_M_INV = 0xD2253531;

// Wrappers removed (moved to end)
// Helper: add_cc
__device__ __forceinline__ uint32_t add_cc(uint32_t a, uint32_t b, uint32_t& carry) {
    uint32_t r;
    asm volatile("add.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    asm volatile("addc.u32 %0, 0, 0;" : "=r"(carry));
    return r;
}

__device__ __forceinline__ uint32_t addc_cc(uint32_t a, uint32_t b, uint32_t& carry) {
    uint32_t r;
    asm volatile("addc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    asm volatile("addc.u32 %0, 0, 0;" : "=r"(carry));
    return r;
}

__device__ __forceinline__ void field_set_zero(FieldElement* r) {
    #pragma unroll
    for(int i=0; i<8; i++) r->limbs[i] = 0;
}

__device__ __forceinline__ void field_set_one(FieldElement* r) {
#if SECP256K1_CUDA_USE_MONTGOMERY
    #pragma unroll
    for(int i=0; i<8; i++) r->limbs[i] = FIELD_R[i];
#else
    #pragma unroll
    for(int i=0; i<8; i++) r->limbs[i] = FIELD_ONE[i];
#endif
}

__device__ __forceinline__ bool field_is_zero(const FieldElement* a) {
    uint32_t acc = 0;
    #pragma unroll
    for(int i=0; i<8; i++) acc |= a->limbs[i];
    return acc == 0;
}

__device__ __forceinline__ bool field_eq(const FieldElement* a, const FieldElement* b) {
    #pragma unroll
    for(int i=0; i<8; i++) {
        if (a->limbs[i] != b->limbs[i]) return false;
    }
    return true;
}

__device__ __forceinline__ void field_add(const FieldElement* a, const FieldElement* b, FieldElement* r) {
    uint32_t t[8];
    uint32_t carry = 0;
    
    asm volatile("add.cc.u32 %0, %1, %2;" : "=r"(t[0]) : "r"(a->limbs[0]), "r"(b->limbs[0]));
    asm volatile("addc.cc.u32 %0, %1, %2;" : "=r"(t[1]) : "r"(a->limbs[1]), "r"(b->limbs[1]));
    asm volatile("addc.cc.u32 %0, %1, %2;" : "=r"(t[2]) : "r"(a->limbs[2]), "r"(b->limbs[2]));
    asm volatile("addc.cc.u32 %0, %1, %2;" : "=r"(t[3]) : "r"(a->limbs[3]), "r"(b->limbs[3]));
    asm volatile("addc.cc.u32 %0, %1, %2;" : "=r"(t[4]) : "r"(a->limbs[4]), "r"(b->limbs[4]));
    asm volatile("addc.cc.u32 %0, %1, %2;" : "=r"(t[5]) : "r"(a->limbs[5]), "r"(b->limbs[5]));
    asm volatile("addc.cc.u32 %0, %1, %2;" : "=r"(t[6]) : "r"(a->limbs[6]), "r"(b->limbs[6]));
    asm volatile("addc.cc.u32 %0, %1, %2;" : "=r"(t[7]) : "r"(a->limbs[7]), "r"(b->limbs[7]));
    asm volatile("addc.u32 %0, 0, 0;" : "=r"(carry));

    uint32_t s[8];
    uint32_t borrow;
    
    asm volatile("sub.cc.u32 %0, %1, %2;" : "=r"(s[0]) : "r"(t[0]), "r"(MODULUS[0]));
    asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(s[1]) : "r"(t[1]), "r"(MODULUS[1]));
    asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(s[2]) : "r"(t[2]), "r"(MODULUS[2]));
    asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(s[3]) : "r"(t[3]), "r"(MODULUS[3]));
    asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(s[4]) : "r"(t[4]), "r"(MODULUS[4]));
    asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(s[5]) : "r"(t[5]), "r"(MODULUS[5]));
    asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(s[6]) : "r"(t[6]), "r"(MODULUS[6]));
    asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(s[7]) : "r"(t[7]), "r"(MODULUS[7]));
    asm volatile("subc.u32 %0, 0, 0;" : "=r"(borrow));

    bool use_sub = (carry != 0) || (borrow == 0); // If carry out, definitely >= P. If no borrow, t >= P.
    
    #pragma unroll
    for(int i=0; i<8; i++) r->limbs[i] = use_sub ? s[i] : t[i];
}

__device__ __forceinline__ void field_sub(const FieldElement* a, const FieldElement* b, FieldElement* r) {
    uint32_t t[8];
    uint32_t borrow = 0;

    asm volatile("sub.cc.u32 %0, %1, %2;" : "=r"(t[0]) : "r"(a->limbs[0]), "r"(b->limbs[0]));
    asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(t[1]) : "r"(a->limbs[1]), "r"(b->limbs[1]));
    asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(t[2]) : "r"(a->limbs[2]), "r"(b->limbs[2]));
    asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(t[3]) : "r"(a->limbs[3]), "r"(b->limbs[3]));
    asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(t[4]) : "r"(a->limbs[4]), "r"(b->limbs[4]));
    asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(t[5]) : "r"(a->limbs[5]), "r"(b->limbs[5]));
    asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(t[6]) : "r"(a->limbs[6]), "r"(b->limbs[6]));
    asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(t[7]) : "r"(a->limbs[7]), "r"(b->limbs[7]));
    asm volatile("subc.u32 %0, 0, 0;" : "=r"(borrow));

    if (borrow) {
        uint32_t c = 0;
        asm volatile("add.cc.u32 %0, %0, %1;" : "+r"(t[0]) : "r"(MODULUS[0]));
        asm volatile("addc.cc.u32 %0, %0, %1;" : "+r"(t[1]) : "r"(MODULUS[1]));
        asm volatile("addc.cc.u32 %0, %0, %1;" : "+r"(t[2]) : "r"(MODULUS[2]));
        asm volatile("addc.cc.u32 %0, %0, %1;" : "+r"(t[3]) : "r"(MODULUS[3]));
        asm volatile("addc.cc.u32 %0, %0, %1;" : "+r"(t[4]) : "r"(MODULUS[4]));
        asm volatile("addc.cc.u32 %0, %0, %1;" : "+r"(t[5]) : "r"(MODULUS[5]));
        asm volatile("addc.cc.u32 %0, %0, %1;" : "+r"(t[6]) : "r"(MODULUS[6]));
        asm volatile("addc.u32 %0, %0, %1;" : "+r"(t[7]) : "r"(MODULUS[7]));
    }
    #pragma unroll
    for(int i=0; i<8; i++) r->limbs[i] = t[i];
}

__device__ __forceinline__ void field_negate(const FieldElement* a, FieldElement* r) {
    // P - a
    uint32_t borrow = 0;
    asm volatile("sub.cc.u32 %0, %1, %2;" : "=r"(r->limbs[0]) : "r"(MODULUS[0]), "r"(a->limbs[0]));
    asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(r->limbs[1]) : "r"(MODULUS[1]), "r"(a->limbs[1]));
    asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(r->limbs[2]) : "r"(MODULUS[2]), "r"(a->limbs[2]));
    asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(r->limbs[3]) : "r"(MODULUS[3]), "r"(a->limbs[3]));
    asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(r->limbs[4]) : "r"(MODULUS[4]), "r"(a->limbs[4]));
    asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(r->limbs[5]) : "r"(MODULUS[5]), "r"(a->limbs[5]));
    asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(r->limbs[6]) : "r"(MODULUS[6]), "r"(a->limbs[6]));
    asm volatile("subc.u32 %0, %1, %2;"    : "=r"(r->limbs[7]) : "r"(MODULUS[7]), "r"(a->limbs[7]));
}

__device__ __forceinline__ void mul_256_512(const uint32_t* a, const uint32_t* b, uint32_t* r) {
    uint32_t r0 = 0, r1 = 0, r2 = 0;

    #define MUL32_ACC(ai, bj) { \
        asm volatile( \
            "mad.lo.cc.u32 %0, %3, %4, %0; \n\t" \
            "madc.hi.cc.u32 %1, %3, %4, %1; \n\t" \
            "addc.u32 %2, %2, 0; \n\t" \
            : "+r"(r0), "+r"(r1), "+r"(r2) \
            : "r"(a[ai]), "r"(b[bj]) \
        ); \
    }

    // Col 0
    MUL32_ACC(0, 0);
    r[0] = r0; r0 = r1; r1 = r2; r2 = 0;

    // Col 1
    MUL32_ACC(0, 1); MUL32_ACC(1, 0);
    r[1] = r0; r0 = r1; r1 = r2; r2 = 0;

    // Col 2
    MUL32_ACC(0, 2); MUL32_ACC(1, 1); MUL32_ACC(2, 0);
    r[2] = r0; r0 = r1; r1 = r2; r2 = 0;

    // Col 3
    MUL32_ACC(0, 3); MUL32_ACC(1, 2); MUL32_ACC(2, 1); MUL32_ACC(3, 0);
    r[3] = r0; r0 = r1; r1 = r2; r2 = 0;

    // Col 4
    MUL32_ACC(0, 4); MUL32_ACC(1, 3); MUL32_ACC(2, 2); MUL32_ACC(3, 1); MUL32_ACC(4, 0);
    r[4] = r0; r0 = r1; r1 = r2; r2 = 0;

    // Col 5
    MUL32_ACC(0, 5); MUL32_ACC(1, 4); MUL32_ACC(2, 3); MUL32_ACC(3, 2); MUL32_ACC(4, 1); MUL32_ACC(5, 0);
    r[5] = r0; r0 = r1; r1 = r2; r2 = 0;

    // Col 6
    MUL32_ACC(0, 6); MUL32_ACC(1, 5); MUL32_ACC(2, 4); MUL32_ACC(3, 3); MUL32_ACC(4, 2); MUL32_ACC(5, 1); MUL32_ACC(6, 0);
    r[6] = r0; r0 = r1; r1 = r2; r2 = 0;

    // Col 7
    MUL32_ACC(0, 7); MUL32_ACC(1, 6); MUL32_ACC(2, 5); MUL32_ACC(3, 4); MUL32_ACC(4, 3); MUL32_ACC(5, 2); MUL32_ACC(6, 1); MUL32_ACC(7, 0);
    r[7] = r0; r0 = r1; r1 = r2; r2 = 0;

    // Col 8
    MUL32_ACC(1, 7); MUL32_ACC(2, 6); MUL32_ACC(3, 5); MUL32_ACC(4, 4); MUL32_ACC(5, 3); MUL32_ACC(6, 2); MUL32_ACC(7, 1);
    r[8] = r0; r0 = r1; r1 = r2; r2 = 0;

    // Col 9
    MUL32_ACC(2, 7); MUL32_ACC(3, 6); MUL32_ACC(4, 5); MUL32_ACC(5, 4); MUL32_ACC(6, 3); MUL32_ACC(7, 2);
    r[9] = r0; r0 = r1; r1 = r2; r2 = 0;

    // Col 10
    MUL32_ACC(3, 7); MUL32_ACC(4, 6); MUL32_ACC(5, 5); MUL32_ACC(6, 4); MUL32_ACC(7, 3);
    r[10] = r0; r0 = r1; r1 = r2; r2 = 0;

    // Col 11
    MUL32_ACC(4, 7); MUL32_ACC(5, 6); MUL32_ACC(6, 5); MUL32_ACC(7, 4);
    r[11] = r0; r0 = r1; r1 = r2; r2 = 0;

    // Col 12
    MUL32_ACC(5, 7); MUL32_ACC(6, 6); MUL32_ACC(7, 5);
    r[12] = r0; r0 = r1; r1 = r2; r2 = 0;

    // Col 13
    MUL32_ACC(6, 7); MUL32_ACC(7, 6);
    r[13] = r0; r0 = r1; r1 = r2; r2 = 0;

    // Col 14
    MUL32_ACC(7, 7);
    r[14] = r0; // r0 = r1 (automatically carries)
    
    r[15] = r1; // Top carry

    #undef MUL32_ACC
}

__device__ __forceinline__ void mont_reduce_512(uint32_t* r) {
    // Fully unrolled Montgomery reduction using native 32-bit PTX
    // Each iteration: m = r[i] * INV, then r[i:i+7] += m * MODULUS[0:7]
    
    #define MONT_ITER(i) { \
        uint32_t m = r[i] * FIELD_M_INV; \
        uint32_t lo, hi, c0 = 0, c1; \
        \
        asm("mul.lo.u32 %0, %1, %2;" : "=r"(lo) : "r"(m), "r"(MODULUS[0])); \
        asm("mul.hi.u32 %0, %1, %2;" : "=r"(hi) : "r"(m), "r"(MODULUS[0])); \
        asm("add.cc.u32 %0, %0, %1;" : "+r"(r[i+0]) : "r"(lo)); \
        asm("addc.u32 %0, 0, 0;" : "=r"(c1)); \
        c0 = hi + c1; \
        \
        asm("mul.lo.u32 %0, %1, %2;" : "=r"(lo) : "r"(m), "r"(MODULUS[1])); \
        asm("mul.hi.u32 %0, %1, %2;" : "=r"(hi) : "r"(m), "r"(MODULUS[1])); \
        asm("add.cc.u32 %0, %1, %2;" : "=r"(lo) : "r"(lo), "r"(c0)); \
        asm("addc.u32 %0, %1, 0;" : "=r"(c0) : "r"(hi)); \
        asm("add.cc.u32 %0, %0, %1;" : "+r"(r[i+1]) : "r"(lo)); \
        asm("addc.u32 %0, %0, 0;" : "+r"(c0)); \
        \
        asm("mul.lo.u32 %0, %1, %2;" : "=r"(lo) : "r"(m), "r"(MODULUS[2])); \
        asm("mul.hi.u32 %0, %1, %2;" : "=r"(hi) : "r"(m), "r"(MODULUS[2])); \
        asm("add.cc.u32 %0, %1, %2;" : "=r"(lo) : "r"(lo), "r"(c0)); \
        asm("addc.u32 %0, %1, 0;" : "=r"(c0) : "r"(hi)); \
        asm("add.cc.u32 %0, %0, %1;" : "+r"(r[i+2]) : "r"(lo)); \
        asm("addc.u32 %0, %0, 0;" : "+r"(c0)); \
        \
        asm("mul.lo.u32 %0, %1, %2;" : "=r"(lo) : "r"(m), "r"(MODULUS[3])); \
        asm("mul.hi.u32 %0, %1, %2;" : "=r"(hi) : "r"(m), "r"(MODULUS[3])); \
        asm("add.cc.u32 %0, %1, %2;" : "=r"(lo) : "r"(lo), "r"(c0)); \
        asm("addc.u32 %0, %1, 0;" : "=r"(c0) : "r"(hi)); \
        asm("add.cc.u32 %0, %0, %1;" : "+r"(r[i+3]) : "r"(lo)); \
        asm("addc.u32 %0, %0, 0;" : "+r"(c0)); \
        \
        asm("mul.lo.u32 %0, %1, %2;" : "=r"(lo) : "r"(m), "r"(MODULUS[4])); \
        asm("mul.hi.u32 %0, %1, %2;" : "=r"(hi) : "r"(m), "r"(MODULUS[4])); \
        asm("add.cc.u32 %0, %1, %2;" : "=r"(lo) : "r"(lo), "r"(c0)); \
        asm("addc.u32 %0, %1, 0;" : "=r"(c0) : "r"(hi)); \
        asm("add.cc.u32 %0, %0, %1;" : "+r"(r[i+4]) : "r"(lo)); \
        asm("addc.u32 %0, %0, 0;" : "+r"(c0)); \
        \
        asm("mul.lo.u32 %0, %1, %2;" : "=r"(lo) : "r"(m), "r"(MODULUS[5])); \
        asm("mul.hi.u32 %0, %1, %2;" : "=r"(hi) : "r"(m), "r"(MODULUS[5])); \
        asm("add.cc.u32 %0, %1, %2;" : "=r"(lo) : "r"(lo), "r"(c0)); \
        asm("addc.u32 %0, %1, 0;" : "=r"(c0) : "r"(hi)); \
        asm("add.cc.u32 %0, %0, %1;" : "+r"(r[i+5]) : "r"(lo)); \
        asm("addc.u32 %0, %0, 0;" : "+r"(c0)); \
        \
        asm("mul.lo.u32 %0, %1, %2;" : "=r"(lo) : "r"(m), "r"(MODULUS[6])); \
        asm("mul.hi.u32 %0, %1, %2;" : "=r"(hi) : "r"(m), "r"(MODULUS[6])); \
        asm("add.cc.u32 %0, %1, %2;" : "=r"(lo) : "r"(lo), "r"(c0)); \
        asm("addc.u32 %0, %1, 0;" : "=r"(c0) : "r"(hi)); \
        asm("add.cc.u32 %0, %0, %1;" : "+r"(r[i+6]) : "r"(lo)); \
        asm("addc.u32 %0, %0, 0;" : "+r"(c0)); \
        \
        asm("mul.lo.u32 %0, %1, %2;" : "=r"(lo) : "r"(m), "r"(MODULUS[7])); \
        asm("mul.hi.u32 %0, %1, %2;" : "=r"(hi) : "r"(m), "r"(MODULUS[7])); \
        asm("add.cc.u32 %0, %1, %2;" : "=r"(lo) : "r"(lo), "r"(c0)); \
        asm("addc.u32 %0, %1, 0;" : "=r"(c0) : "r"(hi)); \
        asm("add.cc.u32 %0, %0, %1;" : "+r"(r[i+7]) : "r"(lo)); \
        asm("addc.u32 %0, %0, 0;" : "+r"(c0)); \
        \
        if (c0) { \
            uint32_t j = i + 8; \
            while (j < 16 && c0) { \
                asm("add.cc.u32 %0, %0, %1;" : "+r"(r[j]) : "r"(c0)); \
                asm("addc.u32 %0, 0, 0;" : "=r"(c0)); \
                j++; \
            } \
        } \
    }
    
    MONT_ITER(0)
    MONT_ITER(1)
    MONT_ITER(2)
    MONT_ITER(3)
    MONT_ITER(4)
    MONT_ITER(5)
    MONT_ITER(6)
    MONT_ITER(7)
    
    #undef MONT_ITER
}

__device__ __forceinline__ void field_reduce_std(uint32_t* wide, FieldElement* r) {
    // Reduction formula: 2^256 == 2^32 + 977 (mod P)
    // For high limb h at position 8+i:
    //   h * 2^(256+32i) == h * (2^32 + 977) * 2^(32i)
    //                   = h*977 at position i + h at position i+1
    
    // Multi-pass reduction: Keep reducing until high limbs are zero
    for(int pass = 0; pass < 4; pass++) {  // 4 passes for safety
        bool any_high = false;
        
        // Check if any high limbs are non-zero
        #pragma unroll
        for(int i = 0; i < 8; i++) {
            if (wide[8 + i] != 0) {
                any_high = true;
                break;
            }
        }
        
        if (!any_high) break;  // All done
        
        // Process each high limb
        for(int i = 0; i < 8; i++) {
            uint32_t h = wide[8 + i];
            if (h == 0) continue;
            
            // Zero out this high limb
            wide[8 + i] = 0;
            
            // Compute h * 977
            uint32_t lo977, hi977;
            asm("mul.lo.u32 %0, %1, 977;" : "=r"(lo977) : "r"(h));
            asm("mul.hi.u32 %0, %1, 977;" : "=r"(hi977) : "r"(h));
            
            // Add h*977 at position i with full carry propagation
            asm("add.cc.u32 %0, %0, %1;" : "+r"(wide[i]) : "r"(lo977));
            
            // At position i+1, add hi977 and h, then propagate carries
            // We need to handle the carry from previous add AND add hi977+h
            if (i + 1 < 16) {
                asm("addc.cc.u32 %0, %0, %1;" : "+r"(wide[i+1]) : "r"(hi977));
                asm("addc.cc.u32 %0, %0, %1;" : "+r"(wide[i+1]) : "r"(h));
                
                // Propagate remaining carries through all limbs
                for(int j = i + 2; j < 16; j++) {
                    asm("addc.cc.u32 %0, %0, 0;" : "+r"(wide[j]));
                }
                uint32_t final_carry;
                asm("addc.u32 %0, 0, 0;" : "=r"(final_carry));
                
                // If final carry beyond wide[15], we have overflow
                // This means we need another reduction pass
                if (final_carry) {
                    // Add to wide[8] to trigger another pass
                    wide[8] += final_carry;
                }
            }
        }
    }
    
    // Copy low 256 bits to result
    uint32_t acc[8];
    #pragma unroll
    for(int i = 0; i < 8; i++) acc[i] = wide[i];
    
    // Final conditional subtraction (up to 2 times to ensure result < P)
    #pragma unroll
    for(int k = 0; k < 2; k++) {
        uint32_t s[8];
        uint32_t borrow;
        asm volatile(
            "sub.cc.u32 %0, %9, %17; \n\t"
            "subc.cc.u32 %1, %10, %18; \n\t"
            "subc.cc.u32 %2, %11, %19; \n\t"
            "subc.cc.u32 %3, %12, %20; \n\t"
            "subc.cc.u32 %4, %13, %21; \n\t"
            "subc.cc.u32 %5, %14, %22; \n\t"
            "subc.cc.u32 %6, %15, %23; \n\t"
            "subc.cc.u32 %7, %16, %24; \n\t"
            "subc.u32 %8, 0, 0; \n\t"
            : "=r"(s[0]), "=r"(s[1]), "=r"(s[2]), "=r"(s[3]), 
              "=r"(s[4]), "=r"(s[5]), "=r"(s[6]), "=r"(s[7]), 
              "=r"(borrow)
            : "r"(acc[0]), "r"(acc[1]), "r"(acc[2]), "r"(acc[3]), 
              "r"(acc[4]), "r"(acc[5]), "r"(acc[6]), "r"(acc[7]),
              "r"(MODULUS[0]), "r"(MODULUS[1]), "r"(MODULUS[2]), "r"(MODULUS[3]),
              "r"(MODULUS[4]), "r"(MODULUS[5]), "r"(MODULUS[6]), "r"(MODULUS[7])
        );

        // If no borrow, acc >= P, so use subtracted value
        if (borrow == 0) {
            #pragma unroll
            for(int i = 0; i < 8; i++) acc[i] = s[i];
        }
    }

    // Copy final result
    #pragma unroll
    for(int i = 0; i < 8; i++) r->limbs[i] = acc[i];
}

__device__ __forceinline__ void field_mul_mont(const FieldElement* a, const FieldElement* b, FieldElement* r) {
    uint32_t wide[16];
    mul_256_512(a->limbs, b->limbs, wide);
    mont_reduce_512(wide);
    #pragma unroll
    for(int i=0; i<8; i++) r->limbs[i] = wide[i+8]; // Result in upper half
}

__device__ __forceinline__ void field_sqr_mont(const FieldElement* a, FieldElement* r) {
    field_mul_mont(a, a, r);
}

__device__ __forceinline__ void field_const_one(FieldElement* r) {
    #pragma unroll
    for(int i=0; i<8; i++) r->limbs[i] = FIELD_ONE[i];
}

__device__ __forceinline__ void field_const_one_mont(FieldElement* r) {
    #pragma unroll
    for(int i=0; i<8; i++) r->limbs[i] = FIELD_R[i];
}

// --- Dispatch Wrappers (must be before field_inv) ---
__device__ __forceinline__ void field_mul(const FieldElement* a, const FieldElement* b, FieldElement* r) {
#if SECP256K1_CUDA_USE_MONTGOMERY
    field_mul_mont(a, b, r);
#else
    uint32_t wide[16];
    mul_256_512(a->limbs, b->limbs, wide);
    field_reduce_std(wide, r);
#endif
}

__device__ __forceinline__ void field_sqr(const FieldElement* a, FieldElement* r) {
#if SECP256K1_CUDA_USE_MONTGOMERY
    field_sqr_mont(a, r);
#else
    field_mul(a, a, r);
#endif
}

__device__ __forceinline__ void field_to_mont(const FieldElement* a, FieldElement* r) {
#if SECP256K1_CUDA_USE_MONTGOMERY
    // a * R^2 * R^-1 = a * R
    FieldElement r2;
    #pragma unroll
    for(int i=0; i<8; i++) r2.limbs[i] = FIELD_R2[i];
    field_mul_mont(a, &r2, r);
#else
    *r = *a;
#endif
}

__device__ __forceinline__ void field_from_mont(const FieldElement* a, FieldElement* r) {
#if SECP256K1_CUDA_USE_MONTGOMERY
    // a * 1 * R^-1 = a * R^-1
    FieldElement one;
    #pragma unroll
    for(int i=0; i<8; i++) one.limbs[i] = FIELD_ONE[i];
    field_mul_mont(a, &one, r);
#else
    *r = *a;
#endif
}

// Field Inverse (Fermat) - using correct addition chain from 64-bit version
__device__ __forceinline__ void field_inv(const FieldElement* a, FieldElement* r) {
    if(field_is_zero(a)) { field_set_zero(r); return; }

    FieldElement x_0, x_1, x_2, x_3, x_4, x_5;
    FieldElement t;

    // 1. x2 = x^2 * x (2 ones) -> x_0
    field_sqr(a, &x_0);
    field_mul(&x_0, a, &x_0);

    // 2. x3 = x2^2 * x (3 ones) -> x_1
    field_sqr(&x_0, &x_1);
    field_mul(&x_1, a, &x_1);

    // 3. x6 = x3^(2^3) * x3 (6 ones) -> x_2
    field_sqr(&x_1, &x_2);
    field_sqr(&x_2, &x_2);
    field_sqr(&x_2, &x_2);
    field_mul(&x_2, &x_1, &x_2);

    // 4. x12 = x6^(2^6) * x6 (12 ones) -> x_3
    field_sqr(&x_2, &x_3);
    for (int i = 0; i < 5; i++) field_sqr(&x_3, &x_3);
    field_mul(&x_3, &x_2, &x_3);

    // 5. x24 = x12^(2^12) * x12 (24 ones) -> x_3 (reuse)
    t = x_3;
    for (int i = 0; i < 12; i++) field_sqr(&t, &t);
    field_mul(&t, &x_3, &x_3);

    // 6. x48 = x24^(2^24) * x24 (48 ones) -> x_4
    t = x_3;
    for (int i = 0; i < 24; i++) field_sqr(&t, &t);
    field_mul(&t, &x_3, &x_4);

    // 7. x96 = x48^(2^48) * x48 (96 ones) -> x_4 (reuse)
    t = x_4;
    for (int i = 0; i < 48; i++) field_sqr(&t, &t);
    field_mul(&t, &x_4, &x_4);

    // 8. x192 = x96^(2^96) * x96 (192 ones) -> x_4 (reuse)
    t = x_4;
    for (int i = 0; i < 96; i++) field_sqr(&t, &t);
    field_mul(&t, &x_4, &x_4);

    // 9. x7 = x6^2 * x (7 ones) -> x_5
    field_sqr(&x_2, &x_5);
    field_mul(&x_5, a, &x_5);

    // 10. x31 = x24^(2^7) * x7 (31 ones) -> x_5 (reuse)
    t = x_3;
    for (int i = 0; i < 7; i++) field_sqr(&t, &t);
    field_mul(&t, &x_5, &x_5);

    // 11. x223 = x192^(2^31) * x31 (223 ones) -> x_5 (reuse)
    t = x_4;
    for (int i = 0; i < 31; i++) field_sqr(&t, &t);
    field_mul(&t, &x_5, &x_5);

    // 12. x5 = x3^(2^2) * x2 (5 ones) -> x_0 (reuse)
    t = x_1;
    field_sqr(&t, &t);
    field_sqr(&t, &t);
    field_mul(&t, &x_0, &x_0);

    // 13. x11 = x6^(2^5) * x5 (11 ones) -> x_1 (reuse)
    t = x_2;
    for (int i = 0; i < 5; i++) field_sqr(&t, &t);
    field_mul(&t, &x_0, &x_1);

    // 14. x22 = x11^(2^11) * x11 (22 ones) -> x_1 (reuse)
    t = x_1;
    for (int i = 0; i < 11; i++) field_sqr(&t, &t);
    field_mul(&t, &x_1, &x_1);

    // 15. t = x223^2 (bit 32 is 0)
    field_sqr(&x_5, &t);

    // 16. t = t^(2^22) * x22 (append 22 ones)
    for (int i = 0; i < 22; i++) field_sqr(&t, &t);
    field_mul(&t, &x_1, &t);

    // 17. t = t^(2^4) (bits 9,8,7,6 are 0)
    field_sqr(&t, &t);
    field_sqr(&t, &t);
    field_sqr(&t, &t);
    field_sqr(&t, &t);

    // 18. Process remaining 6 bits: 101101
    field_sqr(&t, &t);
    field_mul(&t, a, &t);
    field_sqr(&t, &t);
    field_sqr(&t, &t);
    field_mul(&t, a, &t);
    field_sqr(&t, &t);
    field_mul(&t, a, &t);
    field_sqr(&t, &t);
    field_sqr(&t, &t);
    field_mul(&t, a, r);
}

// --- Scalar Implementation ---

__device__ __forceinline__ bool scalar_ge(const Scalar* a, const uint32_t* b) {
    #pragma unroll
    for (int i = 7; i >= 0; i--) {
        if (a->limbs[i] > b[i]) return true;
        if (a->limbs[i] < b[i]) return false;
    }
    return true; 
}

__device__ __forceinline__ void scalar_add(const Scalar* a, const Scalar* b, Scalar* r) {
    uint32_t t[8];
    uint32_t carry = 0;
    
    asm volatile("add.cc.u32 %0, %1, %2;" : "=r"(t[0]) : "r"(a->limbs[0]), "r"(b->limbs[0]));
    asm volatile("addc.cc.u32 %0, %1, %2;" : "=r"(t[1]) : "r"(a->limbs[1]), "r"(b->limbs[1]));
    asm volatile("addc.cc.u32 %0, %1, %2;" : "=r"(t[2]) : "r"(a->limbs[2]), "r"(b->limbs[2]));
    asm volatile("addc.cc.u32 %0, %1, %2;" : "=r"(t[3]) : "r"(a->limbs[3]), "r"(b->limbs[3]));
    asm volatile("addc.cc.u32 %0, %1, %2;" : "=r"(t[4]) : "r"(a->limbs[4]), "r"(b->limbs[4]));
    asm volatile("addc.cc.u32 %0, %1, %2;" : "=r"(t[5]) : "r"(a->limbs[5]), "r"(b->limbs[5]));
    asm volatile("addc.cc.u32 %0, %1, %2;" : "=r"(t[6]) : "r"(a->limbs[6]), "r"(b->limbs[6]));
    asm volatile("addc.cc.u32 %0, %1, %2;" : "=r"(t[7]) : "r"(a->limbs[7]), "r"(b->limbs[7]));
    asm volatile("addc.u32 %0, 0, 0;" : "=r"(carry));

    uint32_t s[8];
    uint32_t borrow;
    
    asm volatile("sub.cc.u32 %0, %1, %2;" : "=r"(s[0]) : "r"(t[0]), "r"(ORDER[0]));
    asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(s[1]) : "r"(t[1]), "r"(ORDER[1]));
    asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(s[2]) : "r"(t[2]), "r"(ORDER[2]));
    asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(s[3]) : "r"(t[3]), "r"(ORDER[3]));
    asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(s[4]) : "r"(t[4]), "r"(ORDER[4]));
    asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(s[5]) : "r"(t[5]), "r"(ORDER[5]));
    asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(s[6]) : "r"(t[6]), "r"(ORDER[6]));
    asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(s[7]) : "r"(t[7]), "r"(ORDER[7]));
    asm volatile("subc.u32 %0, 0, 0;" : "=r"(borrow));

    bool use_sub = (carry != 0) || (borrow == 0);
    
    #pragma unroll
    for(int i=0; i<8; i++) r->limbs[i] = use_sub ? s[i] : t[i];
}

__device__ __forceinline__ void scalar_sub(const Scalar* a, const Scalar* b, Scalar* r) {
    uint32_t t[8];
    uint32_t borrow = 0;

    asm volatile("sub.cc.u32 %0, %1, %2;" : "=r"(t[0]) : "r"(a->limbs[0]), "r"(b->limbs[0]));
    asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(t[1]) : "r"(a->limbs[1]), "r"(b->limbs[1]));
    asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(t[2]) : "r"(a->limbs[2]), "r"(b->limbs[2]));
    asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(t[3]) : "r"(a->limbs[3]), "r"(b->limbs[3]));
    asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(t[4]) : "r"(a->limbs[4]), "r"(b->limbs[4]));
    asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(t[5]) : "r"(a->limbs[5]), "r"(b->limbs[5]));
    asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(t[6]) : "r"(a->limbs[6]), "r"(b->limbs[6]));
    asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(t[7]) : "r"(a->limbs[7]), "r"(b->limbs[7]));
    asm volatile("subc.u32 %0, 0, 0;" : "=r"(borrow));

    if (borrow) {
        asm volatile("add.cc.u32 %0, %0, %1;" : "+r"(t[0]) : "r"(ORDER[0]));
        asm volatile("addc.cc.u32 %0, %0, %1;" : "+r"(t[1]) : "r"(ORDER[1]));
        asm volatile("addc.cc.u32 %0, %0, %1;" : "+r"(t[2]) : "r"(ORDER[2]));
        asm volatile("addc.cc.u32 %0, %0, %1;" : "+r"(t[3]) : "r"(ORDER[3]));
        asm volatile("addc.cc.u32 %0, %0, %1;" : "+r"(t[4]) : "r"(ORDER[4]));
        asm volatile("addc.cc.u32 %0, %0, %1;" : "+r"(t[5]) : "r"(ORDER[5]));
        asm volatile("addc.cc.u32 %0, %0, %1;" : "+r"(t[6]) : "r"(ORDER[6]));
        asm volatile("addc.u32 %0, %0, %1;" : "+r"(t[7]) : "r"(ORDER[7]));
    }
    #pragma unroll
    for(int i=0; i<8; i++) r->limbs[i] = t[i];
}

__device__ __forceinline__ void scalar_add_u64(const Scalar* a, uint64_t b, Scalar* r) {
    Scalar tmp_b;
    tmp_b.limbs[0] = (uint32_t)b;
    tmp_b.limbs[1] = (uint32_t)(b >> 32);
    #pragma unroll
    for(int i=2; i<8; i++) tmp_b.limbs[i] = 0;
    
    scalar_add(a, &tmp_b, r);
}

__device__ __forceinline__ void scalar_sub_u64(const Scalar* a, uint64_t b, Scalar* r) {
    Scalar tmp_b;
    tmp_b.limbs[0] = (uint32_t)b;
    tmp_b.limbs[1] = (uint32_t)(b >> 32);
    #pragma unroll
    for(int i=2; i<8; i++) tmp_b.limbs[i] = 0;
    
    scalar_sub(a, &tmp_b, r);
}

__device__ __forceinline__ bool scalar_is_zero(const Scalar* s) {
    uint32_t acc = 0;
    #pragma unroll
    for(int i=0; i<8; i++) acc |= s->limbs[i];
    return acc == 0;
}

__device__ __forceinline__ int scalar_bit(const Scalar* s, int index) {
    return (s->limbs[index >> 5] >> (index & 0x1F)) & 1;
}

// wNAF encoding for 32-bit
__device__ inline int scalar_to_wnaf(const Scalar* k, int8_t* wnaf, int max_len) {
    Scalar temp = *k;
    int len = 0;
    const int window_size = 32;
    const int window_mask = 31;
    const int window_half = 16;
    
    while (!scalar_is_zero(&temp) && len < max_len) {
        if (scalar_bit(&temp, 0) == 1) {
            int digit = (int)(temp.limbs[0] & window_mask);
            if (digit >= window_half) {
                digit -= window_size;
                scalar_add_u64(&temp, (uint64_t)(-digit), &temp);
            } else {
                scalar_sub_u64(&temp, (uint64_t)digit, &temp);
            }
            wnaf[len] = (int8_t)digit;
        } else {
            wnaf[len] = 0;
        }

        // Right shift by 1
        uint32_t carry = 0;
        #pragma unroll
        for(int i=7; i>=0; i--) {
            uint32_t next_carry = temp.limbs[i] & 1;
            temp.limbs[i] = (temp.limbs[i] >> 1) | (carry << 31);
            carry = next_carry;
        }
        len++;
    }
    return len;
}

// ============================================================================
// Extended scalar arithmetic (mod ORDER) - 32-bit backend
// ============================================================================

// Barrett constant: mu = floor(2^512 / ORDER), 10 x 32-bit limbs (LE)
__constant__ static const uint32_t BARRETT_MU_32[10] = {
    0x2FC9BEC0, 0x402DA173, 0x50B75FC4, 0x45512319,
    0x00000001, 0x00000000, 0x00000000, 0x00000000,
    0x00000001, 0x00000000
};

// n - 2 (for Fermat inversion), 8 x 32-bit LE limbs
__constant__ static const uint32_t ORDER_MINUS_2[8] = {
    0xD036413F, 0xBFD25E8C, 0xAF48A03B, 0xBAAEDCE6,
    0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

// Scalar negation: r = -a mod n (branchless)
__device__ inline void scalar_negate(const Scalar* a, Scalar* r) {
    uint32_t borrow = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int64_t d = (int64_t)ORDER[i] - (int64_t)a->limbs[i] - (int64_t)borrow;
        r->limbs[i] = (uint32_t)d;
        borrow = (d < 0) ? 1u : 0u;
    }
    // If a == 0, result must be 0 (branchless mask)
    uint32_t nz = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) nz |= a->limbs[i];
    uint32_t mask = -(uint32_t)(nz != 0);
    #pragma unroll
    for (int i = 0; i < 8; i++) r->limbs[i] &= mask;
}

// Scalar parity check
__device__ __forceinline__ bool scalar_is_even(const Scalar* s) {
    return (s->limbs[0] & 1) == 0;
}

// Scalar equality
__device__ __forceinline__ bool scalar_eq(const Scalar* a, const Scalar* b) {
    uint32_t diff = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) diff |= (a->limbs[i] ^ b->limbs[i]);
    return diff == 0;
}

// Scalar multiplication mod ORDER: r = a * b (mod n)
// Schoolbook 8x8 -> 16-limb product + Barrett reduction
__device__ inline void scalar_mul_mod_n(const Scalar* a, const Scalar* b, Scalar* r) {
    uint32_t prod[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) prod[i] = 0;

    // Schoolbook 8x8 multiplication
    for (int i = 0; i < 8; i++) {
        uint32_t carry = 0;
        for (int j = 0; j < 8; j++) {
            uint64_t p = (uint64_t)a->limbs[i] * b->limbs[j] + prod[i + j] + carry;
            prod[i + j] = (uint32_t)p;
            carry = (uint32_t)(p >> 32);
        }
        prod[i + 8] = carry;
    }

    // Barrett reduction
    // q = prod[8..15] (high 256 bits)
    // q * BARRETT_MU_32 -> take limbs [8..15] as q_approx
    uint32_t qmu[18];
    #pragma unroll
    for (int i = 0; i < 18; i++) qmu[i] = 0;

    for (int i = 0; i < 8; i++) {
        uint32_t carry = 0;
        for (int j = 0; j < 10; j++) {
            uint64_t p = (uint64_t)prod[8 + i] * BARRETT_MU_32[j] + qmu[i + j] + carry;
            qmu[i + j] = (uint32_t)p;
            carry = (uint32_t)(p >> 32);
        }
        qmu[i + 10] = carry;
    }

    // q_approx = qmu[8..15]
    // q_approx * ORDER -> low 9 limbs
    uint32_t qn[9];
    #pragma unroll
    for (int i = 0; i < 9; i++) qn[i] = 0;

    for (int i = 0; i < 8; i++) {
        uint32_t carry = 0;
        for (int j = 0; j < 8; j++) {
            if (i + j >= 9) break;
            uint64_t p = (uint64_t)qmu[8 + i] * ORDER[j] + qn[i + j] + carry;
            qn[i + j] = (uint32_t)p;
            carry = (uint32_t)(p >> 32);
        }
        if (i + 8 < 9) qn[i + 8] = carry;
    }

    // r = prod[0..7] - qn[0..7]
    uint32_t borrow = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int64_t d = (int64_t)prod[i] - qn[i] - borrow;
        r->limbs[i] = (uint32_t)d;
        borrow = (d < 0) ? 1u : 0u;
    }
    uint32_t r8 = prod[8] - qn[8] - borrow;

    // At most 2 conditional subtracts to bring into [0, ORDER)
    if (r8 > 0 || scalar_ge(r, ORDER)) {
        borrow = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int64_t d = (int64_t)r->limbs[i] - ORDER[i] - borrow;
            r->limbs[i] = (uint32_t)d;
            borrow = (d < 0) ? 1u : 0u;
        }
        r8 -= borrow;
    }
    if (r8 > 0 || scalar_ge(r, ORDER)) {
        borrow = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int64_t d = (int64_t)r->limbs[i] - ORDER[i] - borrow;
            r->limbs[i] = (uint32_t)d;
            borrow = (d < 0) ? 1u : 0u;
        }
    }
}

// Scalar squaring mod ORDER: r = a^2 (mod n)
__device__ inline void scalar_sqr_mod_n(const Scalar* a, Scalar* r) {
    scalar_mul_mod_n(a, a, r);
}

// Scalar inverse: r = a^(n-2) mod n (Fermat's little theorem)
// Square-and-multiply, MSB to LSB
__device__ inline void scalar_inverse(const Scalar* a, Scalar* r) {
    if (scalar_is_zero(a)) {
        #pragma unroll
        for (int i = 0; i < 8; i++) r->limbs[i] = 0;
        return;
    }

    Scalar result;
    result.limbs[0] = 1;
    #pragma unroll
    for (int i = 1; i < 8; i++) result.limbs[i] = 0;
    Scalar base = *a;

    for (int i = 255; i >= 0; --i) {
        Scalar tmp;
        scalar_sqr_mod_n(&result, &tmp);
        result = tmp;

        int limb_idx = i / 32;
        int bit_idx = i % 32;
        if ((ORDER_MINUS_2[limb_idx] >> bit_idx) & 1) {
            scalar_mul_mod_n(&result, &base, &tmp);
            result = tmp;
        }
    }
    *r = result;
}

// Bit-length of a scalar (for GLV sign selection)
__device__ inline int scalar_bitlen(const Scalar* s) {
    for (int i = 7; i >= 0; --i) {
        if (s->limbs[i] != 0) {
            return i * 32 + (32 - __clz(s->limbs[i]));
        }
    }
    return 0;
}

// --- Late-bound Wrappers ---
// field_mul / field_sqr moved up for access by field_inv


__device__ __forceinline__ bool field_is_even(const FieldElement* a) {
    return (a->limbs[0] & 1) == 0;
}

__device__ __forceinline__ bool field_is_one(const FieldElement* a) {
#if SECP256K1_CUDA_USE_MONTGOMERY
    #pragma unroll
    for(int i=0; i<8; i++) if(a->limbs[i] != FIELD_R[i]) return false;
    return true;
#else
    if(a->limbs[0] != 1) return false;
    #pragma unroll
    for(int i=1; i<8; i++) if(a->limbs[i] != 0) return false;
    return true;
#endif
}


