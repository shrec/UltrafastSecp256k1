#pragma once

#include <cstdint>

// If SECP256K1_USE_PTX not defined by gpu_compat.h, default to PTX (CUDA build)
#ifndef SECP256K1_USE_PTX
#define SECP256K1_USE_PTX 1
#endif

// Macro for Accumulating Multiply-Add (c0, c1, c2) += a * b
// c0, c1, c2 are the running 192-bit accumulator (low, mid, high).
#if SECP256K1_USE_PTX
// Uses PTX mad.lo.cc / madc.hi.cc for optimal instruction chaining.
#define PTX_MAD_ACC(c0, c1, c2, a, b) \
    asm volatile( \
        "mad.lo.cc.u64 %0, %3, %4, %0; \n\t" \
        "madc.hi.cc.u64 %1, %3, %4, %1; \n\t" \
        "addc.u64 %2, %2, 0; \n\t" \
        : "+l"(c0), "+l"(c1), "+l"(c2) \
        : "l"(a), "l"(b) \
    );
#else
// Portable __int128 fallback for HIP/ROCm
#define PTX_MAD_ACC(c0, c1, c2, a, b) \
    do { \
        unsigned __int128 _mad_prod = (unsigned __int128)(uint64_t)(a) * (uint64_t)(b); \
        unsigned __int128 _mad_s0 = (unsigned __int128)(c0) + (uint64_t)_mad_prod; \
        (c0) = (uint64_t)_mad_s0; \
        unsigned __int128 _mad_s1 = (unsigned __int128)(c1) + (uint64_t)(_mad_prod >> 64) + (_mad_s0 >> 64); \
        (c1) = (uint64_t)_mad_s1; \
        (c2) += (uint64_t)(_mad_s1 >> 64); \
    } while(0)
#endif

// Optimized 256x256 -> 512 multiplication using Product Scanning (Comba)
// with PTX mad.lo.cc / madc.hi.cc chain.
__device__ __forceinline__ void mul_256_512_ptx(const uint64_t* a, const uint64_t* b, uint64_t* r) {
    uint64_t a0 = a[0];
    uint64_t a1 = a[1];
    uint64_t a2 = a[2];
    uint64_t a3 = a[3];

    uint64_t b0 = b[0];
    uint64_t b1 = b[1];
    uint64_t b2 = b[2];
    uint64_t b3 = b[3];

    uint64_t c0 = 0, c1 = 0, c2 = 0;

    // Col 0
    PTX_MAD_ACC(c0, c1, c2, a0, b0);
    r[0] = c0;
    c0 = c1; c1 = c2; c2 = 0;

    // Col 1
    PTX_MAD_ACC(c0, c1, c2, a0, b1);
    PTX_MAD_ACC(c0, c1, c2, a1, b0);
    r[1] = c0;
    c0 = c1; c1 = c2; c2 = 0;

    // Col 2
    PTX_MAD_ACC(c0, c1, c2, a0, b2);
    PTX_MAD_ACC(c0, c1, c2, a1, b1);
    PTX_MAD_ACC(c0, c1, c2, a2, b0);
    r[2] = c0;
    c0 = c1; c1 = c2; c2 = 0;

    // Col 3
    PTX_MAD_ACC(c0, c1, c2, a0, b3);
    PTX_MAD_ACC(c0, c1, c2, a1, b2);
    PTX_MAD_ACC(c0, c1, c2, a2, b1);
    PTX_MAD_ACC(c0, c1, c2, a3, b0);
    r[3] = c0;
    c0 = c1; c1 = c2; c2 = 0;

    // Col 4
    PTX_MAD_ACC(c0, c1, c2, a1, b3);
    PTX_MAD_ACC(c0, c1, c2, a2, b2);
    PTX_MAD_ACC(c0, c1, c2, a3, b1);
    r[4] = c0;
    c0 = c1; c1 = c2; c2 = 0;

    // Col 5
    PTX_MAD_ACC(c0, c1, c2, a2, b3);
    PTX_MAD_ACC(c0, c1, c2, a3, b2);
    r[5] = c0;
    c0 = c1; c1 = c2; c2 = 0;

    // Col 6
    PTX_MAD_ACC(c0, c1, c2, a3, b3);
    r[6] = c0;
    c0 = c1; c1 = c2; c2 = 0;

    // Col 7
    r[7] = c0;
}

// Optimized 256 -> 512 squaring using Product Scanning
__device__ __forceinline__ void sqr_256_512_ptx(const uint64_t* a, uint64_t* r) {
    uint64_t a0 = a[0];
    uint64_t a1 = a[1];
    uint64_t a2 = a[2];
    uint64_t a3 = a[3];

    uint64_t c0 = 0, c1 = 0, c2 = 0;

    // Col 0: a0*a0
    PTX_MAD_ACC(c0, c1, c2, a0, a0);
    r[0] = c0;
    c0 = c1; c1 = c2; c2 = 0;

    // Col 1: 2 * a0*a1
    PTX_MAD_ACC(c0, c1, c2, a0, a1);
    PTX_MAD_ACC(c0, c1, c2, a0, a1);
    r[1] = c0;
    c0 = c1; c1 = c2; c2 = 0;

    // Col 2: 2 * a0*a2 + a1*a1
    PTX_MAD_ACC(c0, c1, c2, a0, a2);
    PTX_MAD_ACC(c0, c1, c2, a0, a2);
    PTX_MAD_ACC(c0, c1, c2, a1, a1);
    r[2] = c0;
    c0 = c1; c1 = c2; c2 = 0;

    // Col 3: 2 * (a0*a3 + a1*a2)
    PTX_MAD_ACC(c0, c1, c2, a0, a3);
    PTX_MAD_ACC(c0, c1, c2, a0, a3);
    PTX_MAD_ACC(c0, c1, c2, a1, a2);
    PTX_MAD_ACC(c0, c1, c2, a1, a2);
    r[3] = c0;
    c0 = c1; c1 = c2; c2 = 0;

    // Col 4: 2 * a1*a3 + a2*a2
    PTX_MAD_ACC(c0, c1, c2, a1, a3);
    PTX_MAD_ACC(c0, c1, c2, a1, a3);
    PTX_MAD_ACC(c0, c1, c2, a2, a2);
    r[4] = c0;
    c0 = c1; c1 = c2; c2 = 0;

    // Col 5: 2 * a2*a3
    PTX_MAD_ACC(c0, c1, c2, a2, a3);
    PTX_MAD_ACC(c0, c1, c2, a2, a3);
    r[5] = c0;
    c0 = c1; c1 = c2; c2 = 0;

    // Col 6: a3*a3
    PTX_MAD_ACC(c0, c1, c2, a3, a3);
    r[6] = c0;
    c0 = c1; c1 = c2; c2 = 0;

    // Col 7
    r[7] = c0;
}
