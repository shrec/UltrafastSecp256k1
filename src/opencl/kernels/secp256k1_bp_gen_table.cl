// =============================================================================
// Bulletproof Generator Precomputed Tables -- Fixed-Base Scalar Multiplication
// =============================================================================
//
// Three table strategies for the 128 constant Bulletproof generators (G[64]+H[64]):
//
// 1. Simple fixed-window w=4: table[gen][0..15] = j*P
//    128 gens x 16 entries x 32B(AffinePoint) = 128 KB (4x64 limbs)
//    64 windows x 4 doublings + up to 64 mixed additions = ~320 ops
//
// 2. Positional LUT w=4 (ZERO DOUBLINGS):
//    table[gen][window][digit] = digit * 2^(4*window) * P
//    128 gens x 64 windows x 16 entries = 8 MB
//    Runtime: up to 63 mixed additions, 0 doublings
//
// Include via: #include "secp256k1_zk.cl" (gets full field/point/scalar chain)
// =============================================================================

#include "secp256k1_zk.cl"

// --- Table dimensions --------------------------------------------------------
#define BP_NUM_GENS       128   // 64 G + 64 H
#define BP_TABLE_W4_BITS  4
#define BP_TABLE_W4_SIZE  16    // 1 << 4
// Total w=4: 128 * 16 * sizeof(AffinePoint) = 128 KB

#define BP_LUT4_WINDOWS   64    // 256 bits / 4 bits
#define BP_LUT4_WIN_SIZE  16    // 2^4
#define BP_LUT4_GEN_STRIDE (BP_LUT4_WINDOWS * BP_LUT4_WIN_SIZE)  // 1024 entries/gen

// =============================================================================
// 1. Simple Fixed-Window w=4 Init Kernel
// =============================================================================
// Each work-item builds one generator's 16-entry table.
// Launch with global_size = 128, local_size = 128 (or smaller).
// Reads from bp_G[64] + bp_H[64] already filled by bulletproof_init_kernel.

__kernel void bp_gen_table_init_w4_kernel(
    __global const AffinePoint* bp_G,     // 64 G_i generators (from bulletproof_init_kernel)
    __global const AffinePoint* bp_H,     // 64 H_i generators
    __global AffinePoint* bp_table_w4)    // output: 128*16 = 2048 AffinePoints
{
    uint g = get_global_id(0);
    if (g >= BP_NUM_GENS) return;

    __global AffinePoint* table = &bp_table_w4[g * BP_TABLE_W4_SIZE];

    // Select G_i or H_i
    AffinePoint base;
    if (g < 64) {
        base = bp_G[g];
    } else {
        base = bp_H[g - 64];
    }

    // table[0] = identity placeholder (digit 0 means skip)
    table[0].x.limbs[0] = 0; table[0].x.limbs[1] = 0;
    table[0].x.limbs[2] = 0; table[0].x.limbs[3] = 0;
    table[0].y.limbs[0] = 0; table[0].y.limbs[1] = 0;
    table[0].y.limbs[2] = 0; table[0].y.limbs[3] = 0;

    // table[1] = 1*P
    table[1] = base;

    // table[j] = j*P for j = 2..15
    JacobianPoint acc;
    acc.x = base.x; acc.y = base.y;
    acc.z.limbs[0] = 1; acc.z.limbs[1] = 0;
    acc.z.limbs[2] = 0; acc.z.limbs[3] = 0;
    acc.infinity = 0;

    for (int j = 2; j < BP_TABLE_W4_SIZE; j++) {
        point_add_mixed_impl(&acc, &acc, &base);

        // Jacobian -> Affine for table storage
        FieldElement zi, zi2, zi3;
        field_inv_impl(&zi, &acc.z);
        field_sqr_impl(&zi2, &zi);
        field_mul_impl(&zi3, &zi2, &zi);
        field_mul_impl(&table[j].x, &acc.x, &zi2);
        field_mul_impl(&table[j].y, &acc.y, &zi3);
    }
}

// =============================================================================
// 2. Fixed-Window w=4 Scalar Multiplication
// =============================================================================
// 64 windows of 4 bits = 256 doublings + up to 64 mixed additions.

inline void scalar_mul_bp_fixed_w4_impl(
    const __global AffinePoint* table,  // 16 entries for this generator
    const Scalar* k,
    JacobianPoint* r)
{
    point_set_infinity(r);
    int started = 0;

    for (int limb = 3; limb >= 0; limb--) {
        ulong w = k->limbs[limb];
        for (int nib = 15; nib >= 0; nib--) {
            uint idx = (uint)((w >> (nib * 4)) & 0xFUL);

            if (started) {
                point_double_impl(r, r);
                point_double_impl(r, r);
                point_double_impl(r, r);
                point_double_impl(r, r);
            }

            if (idx != 0) {
                if (!started) {
                    AffinePoint pt = table[idx];
                    r->x = pt.x;
                    r->y = pt.y;
                    r->z.limbs[0] = 1; r->z.limbs[1] = 0;
                    r->z.limbs[2] = 0; r->z.limbs[3] = 0;
                    r->infinity = 0;
                    started = 1;
                } else {
                    AffinePoint pt = table[idx];
                    point_add_mixed_impl(r, r, &pt);
                }
            }
        }
    }
}

// =============================================================================
// 3. Positional LUT w=4 Init Kernel (ZERO DOUBLINGS at runtime)
// =============================================================================
// Each work-item builds the full positional LUT for one generator.
// For window j, base = 2^(4*j) * P.
// table[j][d] = d * base  for d = 0..15
// Launch with global_size = 128.

__kernel void bp_lut4_init_kernel(
    __global const AffinePoint* bp_G,     // 64 G_i generators
    __global const AffinePoint* bp_H,     // 64 H_i generators
    __global AffinePoint* bp_lut4)        // output: 128*64*16 = 131072 AffinePoints (8 MB)
{
    uint g = get_global_id(0);
    if (g >= BP_NUM_GENS) return;

    __global AffinePoint* lut = &bp_lut4[g * BP_LUT4_GEN_STRIDE];

    AffinePoint gen;
    if (g < 64) {
        gen = bp_G[g];
    } else {
        gen = bp_H[g - 64];
    }

    // window_base starts as P, then shifts by 2^4 each window
    JacobianPoint window_base;
    window_base.x = gen.x; window_base.y = gen.y;
    window_base.z.limbs[0] = 1; window_base.z.limbs[1] = 0;
    window_base.z.limbs[2] = 0; window_base.z.limbs[3] = 0;
    window_base.infinity = 0;

    for (int win = 0; win < BP_LUT4_WINDOWS; win++) {
        __global AffinePoint* win_table = &lut[win * BP_LUT4_WIN_SIZE];

        // Convert current window_base to affine
        AffinePoint base_affine;
        {
            FieldElement zi, zi2, zi3;
            field_inv_impl(&zi, &window_base.z);
            field_sqr_impl(&zi2, &zi);
            field_mul_impl(&zi3, &zi2, &zi);
            field_mul_impl(&base_affine.x, &window_base.x, &zi2);
            field_mul_impl(&base_affine.y, &window_base.y, &zi3);
        }

        // win_table[0] = identity placeholder (digit 0 means skip)
        win_table[0].x.limbs[0] = 0; win_table[0].x.limbs[1] = 0;
        win_table[0].x.limbs[2] = 0; win_table[0].x.limbs[3] = 0;
        win_table[0].y.limbs[0] = 0; win_table[0].y.limbs[1] = 0;
        win_table[0].y.limbs[2] = 0; win_table[0].y.limbs[3] = 0;

        // win_table[1] = 1 * window_base
        win_table[1] = base_affine;

        // win_table[d] = d * window_base for d = 2..15
        JacobianPoint acc;
        acc.x = base_affine.x; acc.y = base_affine.y;
        acc.z.limbs[0] = 1; acc.z.limbs[1] = 0;
        acc.z.limbs[2] = 0; acc.z.limbs[3] = 0;
        acc.infinity = 0;

        for (int d = 2; d < BP_LUT4_WIN_SIZE; d++) {
            point_add_mixed_impl(&acc, &acc, &base_affine);
            FieldElement zi, zi2, zi3;
            field_inv_impl(&zi, &acc.z);
            field_sqr_impl(&zi2, &zi);
            field_mul_impl(&zi3, &zi2, &zi);
            field_mul_impl(&win_table[d].x, &acc.x, &zi2);
            field_mul_impl(&win_table[d].y, &acc.y, &zi3);
        }

        // Advance window_base by 2^4 = 4 doublings
        point_double_impl(&window_base, &window_base);
        point_double_impl(&window_base, &window_base);
        point_double_impl(&window_base, &window_base);
        point_double_impl(&window_base, &window_base);
    }
}

// =============================================================================
// 4. Positional LUT Scalar Multiplication: ZERO doublings
// =============================================================================
// k * P = sum_{j=0..63} lut[j][ nibble_j(k) ]
// Up to 63 mixed additions (skip zero nibbles). No doublings at runtime.

inline void scalar_mul_bp_lut4_impl(
    const __global AffinePoint* lut,  // 64*16 = 1024 entries for this generator
    const Scalar* k,
    JacobianPoint* r)
{
    r->infinity = 1;
    r->x.limbs[0] = 0; r->x.limbs[1] = 0; r->x.limbs[2] = 0; r->x.limbs[3] = 0;
    r->y.limbs[0] = 1; r->y.limbs[1] = 0; r->y.limbs[2] = 0; r->y.limbs[3] = 0;
    r->z.limbs[0] = 0; r->z.limbs[1] = 0; r->z.limbs[2] = 0; r->z.limbs[3] = 0;

    for (int win = 0; win < BP_LUT4_WINDOWS; win++) {
        // Extract 4-bit nibble for this window
        // win 0 = bits [3:0] of limbs[0], win 15 = bits [63:60] of limbs[0], etc.
        int limb_idx = win >> 4;       // win / 16
        int nib_idx  = win & 15;       // win % 16
        uint digit = (uint)((k->limbs[limb_idx] >> (nib_idx * 4)) & 0xFUL);

        if (digit != 0) {
            AffinePoint pt = lut[win * BP_LUT4_WIN_SIZE + digit];
            if (r->infinity) {
                r->x = pt.x;
                r->y = pt.y;
                r->z.limbs[0] = 1; r->z.limbs[1] = 0;
                r->z.limbs[2] = 0; r->z.limbs[3] = 0;
                r->infinity = 0;
            } else {
                point_add_mixed_impl(r, r, &pt);
            }
        }
    }

    // Handle all-zero scalar
    if (r->infinity) {
        point_set_infinity(r);
    }
}

// =============================================================================
// 5. Single-Generator Positional LUT Init (H or G)
// =============================================================================
// Builds positional LUT4 for a single fixed point.
// 64 windows x 16 entries = 1024 AffinePoints = 64 KB
// Launch with global_size = 1.

__kernel void bp_single_lut4_init_kernel(
    __global const AffinePoint* gen_point, // single generator point
    __global AffinePoint* lut4)            // output: 64*16 = 1024 AffinePoints
{
    if (get_global_id(0) != 0) return;

    AffinePoint gen = gen_point[0];

    JacobianPoint window_base;
    window_base.x = gen.x; window_base.y = gen.y;
    window_base.z.limbs[0] = 1; window_base.z.limbs[1] = 0;
    window_base.z.limbs[2] = 0; window_base.z.limbs[3] = 0;
    window_base.infinity = 0;

    for (int win = 0; win < BP_LUT4_WINDOWS; win++) {
        __global AffinePoint* win_table = &lut4[win * BP_LUT4_WIN_SIZE];

        AffinePoint base_affine;
        {
            FieldElement zi, zi2, zi3;
            field_inv_impl(&zi, &window_base.z);
            field_sqr_impl(&zi2, &zi);
            field_mul_impl(&zi3, &zi2, &zi);
            field_mul_impl(&base_affine.x, &window_base.x, &zi2);
            field_mul_impl(&base_affine.y, &window_base.y, &zi3);
        }

        win_table[0].x.limbs[0] = 0; win_table[0].x.limbs[1] = 0;
        win_table[0].x.limbs[2] = 0; win_table[0].x.limbs[3] = 0;
        win_table[0].y.limbs[0] = 0; win_table[0].y.limbs[1] = 0;
        win_table[0].y.limbs[2] = 0; win_table[0].y.limbs[3] = 0;

        win_table[1] = base_affine;

        JacobianPoint acc;
        acc.x = base_affine.x; acc.y = base_affine.y;
        acc.z.limbs[0] = 1; acc.z.limbs[1] = 0;
        acc.z.limbs[2] = 0; acc.z.limbs[3] = 0;
        acc.infinity = 0;

        for (int d = 2; d < BP_LUT4_WIN_SIZE; d++) {
            point_add_mixed_impl(&acc, &acc, &base_affine);
            FieldElement zi, zi2, zi3;
            field_inv_impl(&zi, &acc.z);
            field_sqr_impl(&zi2, &zi);
            field_mul_impl(&zi3, &zi2, &zi);
            field_mul_impl(&win_table[d].x, &acc.x, &zi2);
            field_mul_impl(&win_table[d].y, &acc.y, &zi3);
        }

        point_double_impl(&window_base, &window_base);
        point_double_impl(&window_base, &window_base);
        point_double_impl(&window_base, &window_base);
        point_double_impl(&window_base, &window_base);
    }
}
