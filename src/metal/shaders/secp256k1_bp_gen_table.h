// =============================================================================
// Bulletproof Generator Precomputed Tables -- Fixed-Base Scalar Multiplication
// =============================================================================
//
// Metal port of CUDA bp_gen_table.cuh.
//
// Three table strategies for the 128 constant Bulletproof generators (G[64]+H[64]):
//
// 1. Simple fixed-window w=4: table[gen][0..15] = j*P  (128 KB)
// 2. Positional LUT w=4 (ZERO DOUBLINGS): 128x64x16 entries (8 MB)
//
// Requires: secp256k1_zk.h (gets full field/point/scalar chain)
// =============================================================================

#pragma once

#include "secp256k1_zk.h"

// --- Table dimensions --------------------------------------------------------
constant int BP_NUM_GENS       = 128;   // 64 G + 64 H
constant int BP_TABLE_W4_SIZE  = 16;    // 1 << 4
constant int BP_LUT4_WINDOWS   = 64;    // 256 bits / 4 bits
constant int BP_LUT4_WIN_SIZE  = 16;    // 2^4
constant int BP_LUT4_GEN_STRIDE = 1024; // 64 * 16

// =============================================================================
// 1. Simple Fixed-Window w=4 Init Kernel
// =============================================================================
// Each thread builds one generator's 16-entry table.
// Launch with threads_per_grid = 128.

kernel void bp_gen_table_init_w4_kernel(
    device const AffinePoint *bp_G    [[buffer(0)]],   // 64 G_i generators
    device const AffinePoint *bp_H    [[buffer(1)]],   // 64 H_i generators
    device AffinePoint *bp_table_w4   [[buffer(2)]],   // output: 128*16 AffinePoints
    uint tid [[thread_position_in_grid]])
{
    if (tid >= (uint)BP_NUM_GENS) return;

    device AffinePoint* table = &bp_table_w4[tid * BP_TABLE_W4_SIZE];

    AffinePoint base;
    if (tid < 64) {
        base = bp_G[tid];
    } else {
        base = bp_H[tid - 64];
    }

    // table[0] = identity placeholder (digit 0 means skip)
    table[0].x = field_zero();
    table[0].y = field_zero();

    // table[1] = 1*P
    table[1] = base;

    // table[j] = j*P for j = 2..15
    JacobianPoint acc;
    acc.x = base.x; acc.y = base.y;
    acc.z = field_one(); acc.infinity = 0;

    for (int j = 2; j < BP_TABLE_W4_SIZE; j++) {
        acc = jacobian_add_mixed(acc, base);
        AffinePoint aff = jacobian_to_affine(acc);
        table[j] = aff;
    }
}

// =============================================================================
// 2. Fixed-Window w=4 Scalar Multiplication
// =============================================================================

inline JacobianPoint scalar_mul_bp_fixed_w4(
    device const AffinePoint* table,  // 16 entries for this generator
    thread const Scalar256 &k)
{
    JacobianPoint r = point_at_infinity();
    bool started = false;

    for (int limb = 7; limb >= 0; limb--) {
        uint w = k.limbs[limb];
        for (int nib = 7; nib >= 0; nib--) {
            uint idx = (w >> (nib * 4)) & 0xFu;

            if (started) {
                r = jacobian_double(r);
                r = jacobian_double(r);
                r = jacobian_double(r);
                r = jacobian_double(r);
            }

            if (idx != 0) {
                AffinePoint pt = table[idx];
                if (!started) {
                    r.x = pt.x; r.y = pt.y;
                    r.z = field_one();
                    r.infinity = 0;
                    started = true;
                } else {
                    r = jacobian_add_mixed(r, pt);
                }
            }
        }
    }

    return r;
}

// =============================================================================
// 3. Positional LUT w=4 Init Kernel (ZERO DOUBLINGS at runtime)
// =============================================================================
// Each thread builds the full positional LUT for one generator.
// Launch with threads_per_grid = 128.

kernel void bp_lut4_init_kernel(
    device const AffinePoint *bp_G  [[buffer(0)]],   // 64 G_i generators
    device const AffinePoint *bp_H  [[buffer(1)]],   // 64 H_i generators
    device AffinePoint *bp_lut4     [[buffer(2)]],   // output: 128*64*16 AffinePoints (8 MB)
    uint tid [[thread_position_in_grid]])
{
    if (tid >= (uint)BP_NUM_GENS) return;

    device AffinePoint* lut = &bp_lut4[tid * BP_LUT4_GEN_STRIDE];

    AffinePoint gen;
    if (tid < 64) {
        gen = bp_G[tid];
    } else {
        gen = bp_H[tid - 64];
    }

    JacobianPoint window_base;
    window_base.x = gen.x; window_base.y = gen.y;
    window_base.z = field_one(); window_base.infinity = 0;

    for (int win = 0; win < BP_LUT4_WINDOWS; win++) {
        device AffinePoint* win_table = &lut[win * BP_LUT4_WIN_SIZE];

        AffinePoint base_affine = jacobian_to_affine(window_base);

        // win_table[0] = identity placeholder
        win_table[0].x = field_zero();
        win_table[0].y = field_zero();

        // win_table[1] = 1 * window_base
        win_table[1] = base_affine;

        // win_table[d] = d * window_base for d = 2..15
        JacobianPoint acc;
        acc.x = base_affine.x; acc.y = base_affine.y;
        acc.z = field_one(); acc.infinity = 0;

        for (int d = 2; d < BP_LUT4_WIN_SIZE; d++) {
            acc = jacobian_add_mixed(acc, base_affine);
            win_table[d] = jacobian_to_affine(acc);
        }

        // Advance window_base by 2^4 = 4 doublings
        window_base = jacobian_double(window_base);
        window_base = jacobian_double(window_base);
        window_base = jacobian_double(window_base);
        window_base = jacobian_double(window_base);
    }
}

// =============================================================================
// 4. Positional LUT Scalar Multiplication: ZERO doublings
// =============================================================================

inline JacobianPoint scalar_mul_bp_lut4(
    device const AffinePoint* lut,  // 64*16 = 1024 entries for this generator
    thread const Scalar256 &k)
{
    JacobianPoint r = point_at_infinity();

    for (int win = 0; win < BP_LUT4_WINDOWS; win++) {
        // Extract 4-bit nibble: Metal Scalar256 has 8x32-bit limbs
        // win 0 = bits [3:0] of limbs[0], win 7 = bits [31:28] of limbs[0],
        // win 8 = bits [3:0] of limbs[1], etc.
        int limb_idx = win >> 3;       // win / 8
        int nib_idx  = win & 7;        // win % 8
        uint digit = (k.limbs[limb_idx] >> (nib_idx * 4)) & 0xFu;

        if (digit != 0) {
            AffinePoint pt = lut[win * BP_LUT4_WIN_SIZE + digit];
            if (r.infinity != 0) {
                r.x = pt.x; r.y = pt.y;
                r.z = field_one();
                r.infinity = 0;
            } else {
                r = jacobian_add_mixed(r, pt);
            }
        }
    }

    return r;
}

// =============================================================================
// 5. Single-Generator Positional LUT Init (H or G)
// =============================================================================
// Builds positional LUT4 for a single fixed point.
// 64 windows x 16 entries = 1024 AffinePoints = 64 KB
// Launch with threads_per_grid = 1.

kernel void bp_single_lut4_init_kernel(
    device const AffinePoint *gen_point [[buffer(0)]],  // single generator point
    device AffinePoint *lut4           [[buffer(1)]],   // output: 64*16 AffinePoints
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0) return;

    AffinePoint gen = gen_point[0];

    JacobianPoint window_base;
    window_base.x = gen.x; window_base.y = gen.y;
    window_base.z = field_one(); window_base.infinity = 0;

    for (int win = 0; win < BP_LUT4_WINDOWS; win++) {
        device AffinePoint* win_table = &lut4[win * BP_LUT4_WIN_SIZE];

        AffinePoint base_affine = jacobian_to_affine(window_base);

        win_table[0].x = field_zero();
        win_table[0].y = field_zero();
        win_table[1] = base_affine;

        JacobianPoint acc;
        acc.x = base_affine.x; acc.y = base_affine.y;
        acc.z = field_one(); acc.infinity = 0;

        for (int d = 2; d < BP_LUT4_WIN_SIZE; d++) {
            acc = jacobian_add_mixed(acc, base_affine);
            win_table[d] = jacobian_to_affine(acc);
        }

        window_base = jacobian_double(window_base);
        window_base = jacobian_double(window_base);
        window_base = jacobian_double(window_base);
        window_base = jacobian_double(window_base);
    }
}
