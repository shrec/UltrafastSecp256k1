#pragma once
// ============================================================================
// Constant-Time Primitives -- CUDA Device
// ============================================================================
// GPU-side building blocks for side-channel resistant code.
// Every function has a data-independent execution trace:
//   - No secret-dependent branches
//   - No secret-dependent memory access patterns
//   - Fixed instruction count regardless of input
//
// NOTE: GPU side-channel attack surfaces differ from CPU:
//   - No branch predictor state to leak (warp-level SIMT)
//   - No cache-timing attacks (unified L2, no per-core L1d)
//   - Power/EM analysis requires physical access to GPU
//   - Warp divergence IS observable (different warps = different timing)
//
// This CT layer enables:
//   1. Research into GPU side-channel feasibility
//   2. Defense-in-depth for GPU signing workloads
//   3. Parity with CPU CT layer for comparative analysis
//
// Port of: cpu/include/secp256k1/ct/ops.hpp
// ============================================================================

#include <cstdint>

namespace secp256k1 {
namespace cuda {
namespace ct {

// --- Compiler barrier --------------------------------------------------------
// Prevents compiler from optimizing away branchless patterns.
// Uses PTX asm volatile to create optimization barrier on GPU.

__device__ __forceinline__ void value_barrier(uint64_t& v) {
    asm volatile("" : "+l"(v));
}

__device__ __forceinline__ void value_barrier(uint32_t& v) {
    asm volatile("" : "+r"(v));
}

// --- Mask generation ---------------------------------------------------------

// Returns 0xFFFFFFFFFFFFFFFF if v == 0, else 0x0000000000000000
__device__ __forceinline__ uint64_t is_zero_mask(uint64_t v) {
    uint64_t z = v;
    value_barrier(z);
    // (v | -v) >> 63: 0 if v==0, 1 if v!=0
    uint64_t nz = (z | (uint64_t)(-(int64_t)z)) >> 63;
    value_barrier(nz);
    // nz==0 → we want all-ones; nz==1 → we want 0
    return (nz - 1);  // 0-1=0xFFF... (zero case), 1-1=0 (nonzero case)
}

// Returns 0xFFFFFFFFFFFFFFFF if v != 0, else 0x0000000000000000
__device__ __forceinline__ uint64_t is_nonzero_mask(uint64_t v) {
    return ~is_zero_mask(v);
}

// Returns 0xFFFFFFFFFFFFFFFF if a == b, else 0x0000000000000000
__device__ __forceinline__ uint64_t eq_mask(uint64_t a, uint64_t b) {
    return is_zero_mask(a ^ b);
}

// Convert bool/flag to mask: 0 -> 0, nonzero -> 0xFFFF...
__device__ __forceinline__ uint64_t bool_to_mask(uint64_t flag) {
    uint64_t f = flag;
    value_barrier(f);
    return -(uint64_t)(f != 0);
}

// Unsigned less-than: returns all-ones if a < b, else 0
__device__ __forceinline__ uint64_t lt_mask(uint64_t a, uint64_t b) {
    // a < b iff (a - b) borrows, i.e., high bit of (a - b) when a < b
    // For unsigned: a < b iff MSB of (a ^ ((a ^ b) | ((a - b) ^ a))) is set
    // Simpler: use the borrow from subtraction
    uint64_t diff = a - b;
    // Borrow occurs when a < b: borrow = (a < b) ? 1 : 0
    // borrow = ((~a & b) | ((~(a ^ b)) & diff)) >> 63
    uint64_t borrow = ((~a & b) | (~(a ^ b) & diff)) >> 63;
    value_barrier(borrow);
    return -borrow;  // 0xFFF... if a < b, else 0
}

// --- Conditional operations --------------------------------------------------

// CT conditional move (64-bit): if mask is all-1s, *dst = src; else unchanged
__device__ __forceinline__ void cmov64(uint64_t* dst, uint64_t src, uint64_t mask) {
    uint64_t m = mask;
    value_barrier(m);
    *dst ^= ((*dst ^ src) & m);
}

// CT conditional move (256-bit / 4 limbs)
__device__ __forceinline__ void cmov256(uint64_t dst[4], const uint64_t src[4], uint64_t mask) {
    uint64_t m = mask;
    value_barrier(m);
    dst[0] ^= ((dst[0] ^ src[0]) & m);
    dst[1] ^= ((dst[1] ^ src[1]) & m);
    dst[2] ^= ((dst[2] ^ src[2]) & m);
    dst[3] ^= ((dst[3] ^ src[3]) & m);
}

// CT conditional swap (256-bit): if mask is all-1s, swap a and b
__device__ __forceinline__ void cswap256(uint64_t a[4], uint64_t b[4], uint64_t mask) {
    uint64_t m = mask;
    value_barrier(m);
    for (int i = 0; i < 4; i++) {
        uint64_t x = (a[i] ^ b[i]) & m;
        a[i] ^= x;
        b[i] ^= x;
    }
}

// CT select: returns a if mask == all-ones, else b
__device__ __forceinline__ uint64_t ct_select(uint64_t a, uint64_t b, uint64_t mask) {
    uint64_t m = mask;
    value_barrier(m);
    return (a & m) | (b & ~m);
}

// CT 256-bit select: copies a if mask, else b, into dst
__device__ __forceinline__ void ct_select256(uint64_t dst[4],
                                              const uint64_t a[4],
                                              const uint64_t b[4],
                                              uint64_t mask) {
    uint64_t m = mask;
    value_barrier(m);
    dst[0] = (a[0] & m) | (b[0] & ~m);
    dst[1] = (a[1] & m) | (b[1] & ~m);
    dst[2] = (a[2] & m) | (b[2] & ~m);
    dst[3] = (a[3] & m) | (b[3] & ~m);
}

// CT table lookup: scans ALL entries, returns entry at `index`.
// Always reads every entry (no secret-dependent memory pattern).
// table: array of 4-limb (256-bit) entries, `count` entries total.
__device__ inline void ct_lookup_256(const uint64_t table[][4],
                                     int count,
                                     int index,
                                     uint64_t out[4]) {
    out[0] = 0; out[1] = 0; out[2] = 0; out[3] = 0;
    for (int i = 0; i < count; i++) {
        uint64_t m = eq_mask((uint64_t)i, (uint64_t)index);
        out[0] |= (table[i][0] & m);
        out[1] |= (table[i][1] & m);
        out[2] |= (table[i][2] & m);
        out[3] |= (table[i][3] & m);
    }
}

} // namespace ct
} // namespace cuda
} // namespace secp256k1
