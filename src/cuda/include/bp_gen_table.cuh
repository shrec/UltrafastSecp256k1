#pragma once
// ============================================================================
// Bulletproof Generator Precomputed Tables -- Fixed-Base Scalar Multiplication
// ============================================================================
//
// Three table strategies for the 128 constant Bulletproof generators (G[64]+H[64]):
//
// 1. Simple fixed-window (w=4, w=8): table[gen][0..2^w-1] = j*P
//    Still does 256 doublings. Only reduces additions.
//    w=4: 128 KB, w=8: 2 MB. Speedup: ~8-9%.
//
// 2. Positional LUT w=4 (ZERO DOUBLINGS):
//    table[gen][window][digit] = digit * 2^(4*window) * P
//    128 gens x 64 windows x 16 entries x 64B = 8 MB global memory
//    Runtime: 63 mixed additions, 0 doublings per scalar_mul
//    Expected speedup: 4-5x per scalar_mul (288 ops -> 63 ops)
//
// ============================================================================

#include "secp256k1.cuh"

#if !SECP256K1_CUDA_LIMBS_32

namespace secp256k1 {
namespace cuda {

// --- Table dimensions --------------------------------------------------------
static constexpr int BP_NUM_GENS       = 128;  // 64 G + 64 H

static constexpr int BP_TABLE_W4_BITS  = 4;
static constexpr int BP_TABLE_W4_SIZE  = 16;   // 1 << 4
// Total w=4: 128 * 16 * 64 = 131,072 bytes = 128 KB

static constexpr int BP_TABLE_W8_BITS  = 8;
static constexpr int BP_TABLE_W8_SIZE  = 256;  // 1 << 8
// Total w=8: 128 * 256 * 64 = 2,097,152 bytes = 2 MB

// --- Global device tables (allocated once, filled by init kernels) -----------
__device__ AffinePoint g_bp_gen_table_w4[BP_NUM_GENS * BP_TABLE_W4_SIZE];
__device__ AffinePoint g_bp_gen_table_w8[BP_NUM_GENS * BP_TABLE_W8_SIZE];
__device__ bool g_bp_table_w4_ready = false;
__device__ bool g_bp_table_w8_ready = false;

// --- Table init kernels (launch after bulletproof_init_kernel) ----------------
// Each thread computes one generator's precomputed table.
// Launch with <<<(128+T-1)/T, T>>> where T=128 fills one block.

__global__ void bp_gen_table_init_w4_kernel() {
    int g = threadIdx.x + blockIdx.x * blockDim.x;
    if (g >= BP_NUM_GENS) return;

    // Index: 0..63 = G_i, 64..127 = H_i
    extern __device__ AffinePoint g_bulletproof_G[64];
    extern __device__ AffinePoint g_bulletproof_H[64];

    AffinePoint* table = &g_bp_gen_table_w4[g * BP_TABLE_W4_SIZE];
    const AffinePoint& base = (g < 64) ? g_bulletproof_G[g] : g_bulletproof_H[g - 64];

    // table[0] = identity placeholder (never accessed; idx=0 means skip)
    field_set_zero(&table[0].x);
    field_set_zero(&table[0].y);

    // table[1] = 1*P
    table[1] = base;

    // table[j] = j*P for j = 2..15
    JacobianPoint acc;
    acc.x = base.x; acc.y = base.y; acc.z = FIELD_ONE; acc.infinity = false;

    for (int j = 2; j < BP_TABLE_W4_SIZE; j++) {
        jacobian_add_mixed(&acc, &base, &acc);

        // Convert accumulator to affine for table storage
        FieldElement zi, zi2, zi3;
        field_inv(&acc.z, &zi);
        field_sqr(&zi, &zi2);
        field_mul(&zi2, &zi, &zi3);
        field_mul(&acc.x, &zi2, &table[j].x);
        field_mul(&acc.y, &zi3, &table[j].y);
    }

    if (g == 0) g_bp_table_w4_ready = true;
}

__global__ void bp_gen_table_init_w8_kernel() {
    int g = threadIdx.x + blockIdx.x * blockDim.x;
    if (g >= BP_NUM_GENS) return;

    extern __device__ AffinePoint g_bulletproof_G[64];
    extern __device__ AffinePoint g_bulletproof_H[64];

    AffinePoint* table = &g_bp_gen_table_w8[g * BP_TABLE_W8_SIZE];
    const AffinePoint& base = (g < 64) ? g_bulletproof_G[g] : g_bulletproof_H[g - 64];

    field_set_zero(&table[0].x);
    field_set_zero(&table[0].y);
    table[1] = base;

    JacobianPoint acc;
    acc.x = base.x; acc.y = base.y; acc.z = FIELD_ONE; acc.infinity = false;

    for (int j = 2; j < BP_TABLE_W8_SIZE; j++) {
        jacobian_add_mixed(&acc, &base, &acc);

        FieldElement zi, zi2, zi3;
        field_inv(&acc.z, &zi);
        field_sqr(&zi, &zi2);
        field_mul(&zi2, &zi, &zi3);
        field_mul(&acc.x, &zi2, &table[j].x);
        field_mul(&acc.y, &zi3, &table[j].y);
    }

    if (g == 0) g_bp_table_w8_ready = true;
}

// --- Fixed-window scalar multiplication: w=4 ---------------------------------
// Identical pattern to scalar_mul_generator_const but reads from caller-provided table.
// 64 windows of 4 bits = 256 doublings + up to 64 mixed additions.
// Eliminates the affine conversion overhead in scalar_mul().

__device__ inline void scalar_mul_bp_fixed_w4(
    const AffinePoint* __restrict__ table,  // 16 entries for this generator
    const Scalar* k,
    JacobianPoint* r)
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
                jacobian_double(r, r);
                jacobian_double(r, r);
                jacobian_double(r, r);
                jacobian_double(r, r);
            }

            if (idx != 0) {
                if (!started) {
                    r->x = table[idx].x;
                    r->y = table[idx].y;
                    field_set_one(&r->z);
                    r->infinity = false;
                    started = true;
                } else {
                    jacobian_add_mixed(r, &table[idx], r);
                }
            }
        }
    }
}

// --- Fixed-window scalar multiplication: w=8 ---------------------------------
// 32 windows of 8 bits = 256 doublings + up to 32 mixed additions.
// ~2x fewer additions than w=4, but 16x larger table per generator.

__device__ inline void scalar_mul_bp_fixed_w8(
    const AffinePoint* __restrict__ table,  // 256 entries for this generator
    const Scalar* k,
    JacobianPoint* r)
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
        for (int byte_idx = 7; byte_idx >= 0; byte_idx--) {
            uint32_t idx = (uint32_t)((w >> (byte_idx * 8)) & 0xFFULL);

            if (started) {
                jacobian_double(r, r);
                jacobian_double(r, r);
                jacobian_double(r, r);
                jacobian_double(r, r);
                jacobian_double(r, r);
                jacobian_double(r, r);
                jacobian_double(r, r);
                jacobian_double(r, r);
            }

            if (idx != 0) {
                if (!started) {
                    r->x = table[idx].x;
                    r->y = table[idx].y;
                    field_set_one(&r->z);
                    r->infinity = false;
                    started = true;
                } else {
                    jacobian_add_mixed(r, &table[idx], r);
                }
            }
        }
    }
}

// =============================================================================
// 3. Positional LUT w=4 -- ZERO DOUBLINGS
// =============================================================================
// For each generator P, precompute:
//   lut[window_j][digit_d] = d * 2^(4*j) * P      (j=0..63, d=0..15)
// Runtime: k*P = sum_{j=0..63} lut[j][ nibble_j(k) ]
//   = up to 63 mixed additions, ZERO doublings
//
// Memory: 128 gens x 64 windows x 16 entries x 64 bytes = 8,388,608 bytes = 8 MB
// Layout: g_bp_lut4[gen * (64*16) + window * 16 + digit]

static constexpr int BP_LUT4_WINDOWS    = 64;   // 256 bits / 4 bits
static constexpr int BP_LUT4_WIN_SIZE   = 16;   // 2^4
static constexpr int BP_LUT4_GEN_STRIDE = BP_LUT4_WINDOWS * BP_LUT4_WIN_SIZE;  // 1024 entries/gen
// Total entries: 128 * 1024 = 131,072 AffinePoints = 8 MB

__device__ AffinePoint g_bp_lut4[BP_NUM_GENS * BP_LUT4_GEN_STRIDE];
__device__ bool g_bp_lut4_ready = false;

// --- H generator (Pedersen) positional LUT4 -- 64 KB -------------------------
// Same structure as BP generator LUT4 but for the single fixed H point.
// Used by verify P1c lanes 0, 3, 9 which compute k*H.
__device__ AffinePoint g_bp_h_lut4[BP_LUT4_GEN_STRIDE];
__device__ bool g_bp_h_lut4_ready = false;

// --- G generator (secp256k1 base point) positional LUT4 -- 64 KB -------------
// Used by verify P1c lanes 1, 8 which compute k*G.
__device__ AffinePoint g_bp_g_lut4[BP_LUT4_GEN_STRIDE];
__device__ bool g_bp_g_lut4_ready = false;

// Init kernel: each thread builds the full LUT for one generator.
// For window j, base = 2^(4*j) * P.
// table[j][d] = d * base  for d = 0..15
// Then advance base *= 2^4 (4 doublings) for next window.
// Launch: <<<1, 128>>>

__global__ void bp_lut4_init_kernel() {
    int g = threadIdx.x + blockIdx.x * blockDim.x;
    if (g >= BP_NUM_GENS) return;

    extern __device__ AffinePoint g_bulletproof_G[64];
    extern __device__ AffinePoint g_bulletproof_H[64];

    AffinePoint* lut = &g_bp_lut4[g * BP_LUT4_GEN_STRIDE];
    const AffinePoint& gen = (g < 64) ? g_bulletproof_G[g] : g_bulletproof_H[g - 64];

    // window_base starts as P, then shifts by 2^4 each window
    JacobianPoint window_base;
    window_base.x = gen.x; window_base.y = gen.y;
    window_base.z = FIELD_ONE; window_base.infinity = false;

    for (int win = 0; win < BP_LUT4_WINDOWS; win++) {
        AffinePoint* win_table = &lut[win * BP_LUT4_WIN_SIZE];

        // Convert current window_base to affine
        AffinePoint base_affine;
        {
            FieldElement zi, zi2, zi3;
            field_inv(&window_base.z, &zi);
            field_sqr(&zi, &zi2);
            field_mul(&zi2, &zi, &zi3);
            field_mul(&window_base.x, &zi2, &base_affine.x);
            field_mul(&window_base.y, &zi3, &base_affine.y);
        }

        // win_table[0] = identity placeholder (digit 0 means skip)
        field_set_zero(&win_table[0].x);
        field_set_zero(&win_table[0].y);

        // win_table[1] = 1 * window_base
        win_table[1] = base_affine;

        // win_table[d] = d * window_base for d = 2..15
        JacobianPoint acc;
        acc.x = base_affine.x; acc.y = base_affine.y;
        acc.z = FIELD_ONE; acc.infinity = false;

        for (int d = 2; d < BP_LUT4_WIN_SIZE; d++) {
            jacobian_add_mixed(&acc, &base_affine, &acc);
            FieldElement zi, zi2, zi3;
            field_inv(&acc.z, &zi);
            field_sqr(&zi, &zi2);
            field_mul(&zi2, &zi, &zi3);
            field_mul(&acc.x, &zi2, &win_table[d].x);
            field_mul(&acc.y, &zi3, &win_table[d].y);
        }

        // Advance window_base by 2^4 = 4 doublings
        jacobian_double(&window_base, &window_base);
        jacobian_double(&window_base, &window_base);
        jacobian_double(&window_base, &window_base);
        jacobian_double(&window_base, &window_base);
    }

    if (g == 0) g_bp_lut4_ready = true;
}

// --- H generator LUT4 init kernel -------------------------------------------
// Builds positional LUT4 for the Pedersen H generator (64 KB).
// Launch: <<<1, 1>>> (single generator, single thread)

__global__ void bp_h_lut4_init_kernel(const AffinePoint* H_gen) {
    if (threadIdx.x != 0) return;

    AffinePoint* lut = g_bp_h_lut4;

    JacobianPoint window_base;
    window_base.x = H_gen->x; window_base.y = H_gen->y;
    window_base.z = FIELD_ONE; window_base.infinity = false;

    for (int win = 0; win < BP_LUT4_WINDOWS; win++) {
        AffinePoint* win_table = &lut[win * BP_LUT4_WIN_SIZE];

        AffinePoint base_affine;
        {
            FieldElement zi, zi2, zi3;
            field_inv(&window_base.z, &zi);
            field_sqr(&zi, &zi2);
            field_mul(&zi2, &zi, &zi3);
            field_mul(&window_base.x, &zi2, &base_affine.x);
            field_mul(&window_base.y, &zi3, &base_affine.y);
        }

        field_set_zero(&win_table[0].x);
        field_set_zero(&win_table[0].y);
        win_table[1] = base_affine;

        JacobianPoint acc;
        acc.x = base_affine.x; acc.y = base_affine.y;
        acc.z = FIELD_ONE; acc.infinity = false;

        for (int d = 2; d < BP_LUT4_WIN_SIZE; d++) {
            jacobian_add_mixed(&acc, &base_affine, &acc);
            FieldElement zi, zi2, zi3;
            field_inv(&acc.z, &zi);
            field_sqr(&zi, &zi2);
            field_mul(&zi2, &zi, &zi3);
            field_mul(&acc.x, &zi2, &win_table[d].x);
            field_mul(&acc.y, &zi3, &win_table[d].y);
        }

        jacobian_double(&window_base, &window_base);
        jacobian_double(&window_base, &window_base);
        jacobian_double(&window_base, &window_base);
        jacobian_double(&window_base, &window_base);
    }

    g_bp_h_lut4_ready = true;
}

// --- G generator LUT4 init kernel -------------------------------------------
// Builds positional LUT4 for the secp256k1 generator G (64 KB).
// Uses GENERATOR_TABLE_AFFINE[1] as the base point.
// Launch: <<<1, 1>>>

__global__ void bp_g_lut4_init_kernel() {
    if (threadIdx.x != 0) return;

    const AffinePoint& G_gen = GENERATOR_TABLE_AFFINE[1];
    AffinePoint* lut = g_bp_g_lut4;

    JacobianPoint window_base;
    window_base.x = G_gen.x; window_base.y = G_gen.y;
    window_base.z = FIELD_ONE; window_base.infinity = false;

    for (int win = 0; win < BP_LUT4_WINDOWS; win++) {
        AffinePoint* win_table = &lut[win * BP_LUT4_WIN_SIZE];

        AffinePoint base_affine;
        {
            FieldElement zi, zi2, zi3;
            field_inv(&window_base.z, &zi);
            field_sqr(&zi, &zi2);
            field_mul(&zi2, &zi, &zi3);
            field_mul(&window_base.x, &zi2, &base_affine.x);
            field_mul(&window_base.y, &zi3, &base_affine.y);
        }

        field_set_zero(&win_table[0].x);
        field_set_zero(&win_table[0].y);
        win_table[1] = base_affine;

        JacobianPoint acc;
        acc.x = base_affine.x; acc.y = base_affine.y;
        acc.z = FIELD_ONE; acc.infinity = false;

        for (int d = 2; d < BP_LUT4_WIN_SIZE; d++) {
            jacobian_add_mixed(&acc, &base_affine, &acc);
            FieldElement zi, zi2, zi3;
            field_inv(&acc.z, &zi);
            field_sqr(&zi, &zi2);
            field_mul(&zi2, &zi, &zi3);
            field_mul(&acc.x, &zi2, &win_table[d].x);
            field_mul(&acc.y, &zi3, &win_table[d].y);
        }

        jacobian_double(&window_base, &window_base);
        jacobian_double(&window_base, &window_base);
        jacobian_double(&window_base, &window_base);
        jacobian_double(&window_base, &window_base);
    }

    g_bp_g_lut4_ready = true;
}

// --- Positional LUT scalar multiplication: ZERO doublings --------------------
// k * P = sum_{j=0..63} lut[j][ nibble_j(k) ]
// Up to 63 mixed additions (skip zero nibbles). No doublings at runtime.

__device__ inline void scalar_mul_bp_lut4(
    const AffinePoint* __restrict__ lut,  // 64*16 = 1024 entries for this generator
    const Scalar* k,
    JacobianPoint* r)
{
    r->infinity = true;

    #pragma unroll 1
    for (int win = 0; win < BP_LUT4_WINDOWS; win++) {
        // Extract 4-bit nibble for this window
        // win 0 = bits [3:0] of limbs[0], win 15 = bits [63:60] of limbs[0],
        // win 16 = bits [3:0] of limbs[1], etc.
        int limb_idx = win >> 4;       // win / 16
        int nib_idx  = win & 15;       // win % 16
        uint32_t digit = (uint32_t)((k->limbs[limb_idx] >> (nib_idx * 4)) & 0xFULL);

        if (digit != 0) {
            const AffinePoint* pt = &lut[win * BP_LUT4_WIN_SIZE + digit];
            if (r->infinity) {
                r->x = pt->x;
                r->y = pt->y;
                field_set_one(&r->z);
                r->infinity = false;
            } else {
                jacobian_add_mixed(r, pt, r);
            }
        }
    }

    // Handle all-zero scalar (shouldn't happen in Bulletproof, but be safe)
    if (r->infinity) {
        field_set_zero(&r->x);
        field_set_one(&r->y);
        field_set_zero(&r->z);
    }
}

} // namespace cuda
} // namespace secp256k1

#endif // !SECP256K1_CUDA_LIMBS_32
