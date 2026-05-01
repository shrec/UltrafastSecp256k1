// =============================================================================
// UltrafastSecp256k1 OpenCL — FROST Partial Signature Verification
// =============================================================================
//
// FROST-secp256k1 partial signature verification batch kernel.
// Each thread verifies one partial signature:
//
//   R_i = D_i + rho_i * E_i
//   lhs = z_i * G
//   rhs = R_i + lambda_i_e * Y_i
//   result[i] = (lhs == rhs) ? 1 : 0
//
// Depends on: secp256k1_extended.cl (which includes secp256k1_point.cl)
// Uses 4×64-bit (ulong) limbs — matching the OpenCL convention.
// =============================================================================

#include "secp256k1_extended.cl"

// -----------------------------------------------------------------------------
// Decompress a 33-byte SEC1 compressed point (prefix || x_bytes[32]).
// Returns 1 on success, 0 on invalid prefix or non-square y^2.
// The result has z = 1 (affine Jacobian representation).
// -----------------------------------------------------------------------------
inline int frost_decompress_sec1(const __global uchar* sec1_33,
                                  JacobianPoint* out)
{
    uchar prefix = sec1_33[0];
    if (prefix != 0x02 && prefix != 0x03) return 0;
    int parity = (prefix == 0x03) ? 1 : 0;

    FieldElement x;
    for (int i = 0; i < 4; i++) {
        ulong limb = 0;
        int base = (3 - i) * 8;
        for (int j = 0; j < 8; j++)
            limb = (limb << 8) | (ulong)sec1_33[1 + base + j];
        x.limbs[i] = limb;
    }
    return lift_x_field_impl(&x, parity, out);
}

// -----------------------------------------------------------------------------
// Convert a JacobianPoint to affine x bytes (big-endian) and y-parity.
// Returns 0 if the point is at infinity, 1 on success.
// Uses field_inv_impl to normalize Z if Z != 1.
// -----------------------------------------------------------------------------
inline int frost_jac_to_affine_bytes(const JacobianPoint* p,
                                      uchar x_out[32],
                                      int* y_odd_out)
{
    if (p->infinity) return 0;

    FieldElement x_aff, y_aff;

    /* Fast-path: Z == 1 (affine already — common after decompress + operations) */
    int z_is_one = (p->z.limbs[0] == 1 &&
                    p->z.limbs[1] == 0 &&
                    p->z.limbs[2] == 0 &&
                    p->z.limbs[3] == 0);
    if (z_is_one) {
        x_aff = p->x;
        y_aff = p->y;
    } else {
        FieldElement z_inv, z_inv2, z_inv3;
        field_inv_impl(&z_inv, &p->z);
        field_sqr_impl(&z_inv2, &z_inv);
        field_mul_impl(&z_inv3, &z_inv2, &z_inv);
        field_mul_impl(&x_aff, &p->x, &z_inv2);
        field_mul_impl(&y_aff, &p->y, &z_inv3);
    }

    field_to_bytes_impl(&x_aff, x_out);

    uchar y_bytes[32];
    field_to_bytes_impl(&y_aff, y_bytes);
    *y_odd_out = (int)(y_bytes[31] & 1u);
    return 1;
}

// =============================================================================
// FROST partial signature verification kernel.
// One thread per item in the batch.
// =============================================================================
__kernel void frost_verify_partial(
    __global const uchar *z_i32,        // count × 32  — partial sig scalar z_i
    __global const uchar *D_i33,        // count × 33  — hiding nonce commitment
    __global const uchar *E_i33,        // count × 33  — binding nonce commitment
    __global const uchar *Y_i33,        // count × 33  — verification share pubkey
    __global const uchar *rho_i32,      // count × 32  — binding factor
    __global const uchar *lambda_ie32,  // count × 32  — lambda_i * e
    __global const uchar *negate_R,     // count × 1   — 1 = negate R_i
    __global const uchar *negate_key,   // count × 1   — 1 = negate Y_i
    __global int          *results,     // count × 1   — output (1=valid, 0=invalid)
    uint count)
{
    int tid = (int)get_global_id(0);
    if ((uint)tid >= count) return;

    /* ---- Parse scalars -------------------------------------------------- */
    Scalar z_i, rho_i, lambda_ie;
    scalar_from_bytes_impl(z_i32       + tid * 32, &z_i);
    scalar_from_bytes_impl(rho_i32     + tid * 32, &rho_i);
    scalar_from_bytes_impl(lambda_ie32 + tid * 32, &lambda_ie);

    /* ---- Decompress points ---------------------------------------------- */
    JacobianPoint D_jac, E_jac, Y_jac;
    if (!frost_decompress_sec1(D_i33 + tid * 33, &D_jac)) { results[tid] = 0; return; }
    if (!frost_decompress_sec1(E_i33 + tid * 33, &E_jac)) { results[tid] = 0; return; }
    if (!frost_decompress_sec1(Y_i33 + tid * 33, &Y_jac)) { results[tid] = 0; return; }

    /* Extract AffinePoint from Jacobian (Z=1 after decompress). */
    AffinePoint E_aff, Y_aff;
    E_aff.x = E_jac.x; E_aff.y = E_jac.y;
    Y_aff.x = Y_jac.x; Y_aff.y = Y_jac.y;

    /* ---- R_i = D_i + rho_i * E_i --------------------------------------- */
    JacobianPoint rho_E;
    scalar_mul_glv_impl(&rho_E, &rho_i, &E_aff);

    JacobianPoint R_i;
    point_add_impl(&R_i, &D_jac, &rho_E);

    /* Optionally negate R_i (even-y convention) */
    if (negate_R[tid]) {
        field_negate_impl(&R_i.y, &R_i.y);
    }

    /* ---- lhs = z_i * G ------------------------------------------------- */
    JacobianPoint lhs;
    scalar_mul_generator_impl(&lhs, &z_i);

    /* ---- Optionally negate Y_i ----------------------------------------- */
    if (negate_key[tid]) {
        field_negate_impl(&Y_aff.y, &Y_aff.y);
    }

    /* ---- rhs = R_i + lambda_i_e * Y_i ---------------------------------- */
    JacobianPoint lambda_Y;
    scalar_mul_glv_impl(&lambda_Y, &lambda_ie, &Y_aff);

    JacobianPoint rhs;
    point_add_impl(&rhs, &R_i, &lambda_Y);

    /* ---- Compare lhs == rhs via affine x + y-parity -------------------- */
    uchar lhs_x[32], rhs_x[32];
    int lhs_y_odd = 0, rhs_y_odd = 0;

    if (!frost_jac_to_affine_bytes(&lhs, lhs_x, &lhs_y_odd)) { results[tid] = 0; return; }
    if (!frost_jac_to_affine_bytes(&rhs, rhs_x, &rhs_y_odd)) { results[tid] = 0; return; }

    if (lhs_y_odd != rhs_y_odd) { results[tid] = 0; return; }

    int match = 1;
    for (int b = 0; b < 32; b++)
        if (lhs_x[b] != rhs_x[b]) { match = 0; break; }

    results[tid] = match;
}
