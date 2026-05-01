// =============================================================================
// secp256k1_pedersen.cl -- Pedersen Commitments for OpenCL
// =============================================================================
// C = v*H + r*G
// Provides:
//   - pedersen_commit_impl: single commitment
//   - pedersen_commit_batch_kernel: batch kernel
//   - pedersen_verify_sum_kernel: homomorphic sum verification
//   - lift_x_even_impl: find point with given x and even y
// =============================================================================

#ifndef SECP256K1_PEDERSEN_CL
#define SECP256K1_PEDERSEN_CL

// -- lift_x to curve point with even y ----------------------------------------
inline int lift_x_even_impl(const FieldElement* x, AffinePoint* out) {
    FieldElement x2, x3, y2, y;
    field_sqr_impl(&x2, x);
    field_mul_impl(&x3, &x2, x);

    FieldElement seven;
    for (int i = 0; i < 4; ++i) seven.limbs[i] = 0;
    seven.limbs[0] = 7;
    field_add_impl(&y2, &x3, &seven);

    // y = sqrt(y^2) = y2^((p+1)/4) -- field_sqrt via field_pow
    // p+1)/4 exponent: using repeated squaring
    // For secp256k1: p = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    // (p+1)/4 = 3FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFBFFFFF0C
    field_sqrt_impl(&y, &y2);

    // Verify
    FieldElement check;
    field_sqr_impl(&check, &y);
    uchar check_b[32], y2_b[32];
    for (int i = 0; i < 4; ++i) {
        ulong v1 = check.limbs[3-i], v2 = y2.limbs[3-i];
        for (int j = 0; j < 8; ++j) {
            check_b[i*8+j] = (uchar)(v1 >> (56-j*8));
            y2_b[i*8+j]    = (uchar)(v2 >> (56-j*8));
        }
    }
    for (int i = 0; i < 32; ++i)
        if (check_b[i] != y2_b[i]) return 0;

    // Ensure even y
    if (y.limbs[0] & 1) {
        FieldElement neg_y;
        field_neg_impl(&neg_y, &y);
        y = neg_y;
    }

    out->x = *x;
    out->y = y;
    return 1;
}

// -- Hash to point (try-and-increment) ----------------------------------------
inline int hash_to_point_increment_impl(const FieldElement* x_in, AffinePoint* out) {
    FieldElement x = *x_in;
    FieldElement one;
    for (int i = 0; i < 4; ++i) one.limbs[i] = 0;
    one.limbs[0] = 1;
    for (int attempt = 0; attempt < 256; ++attempt) {
        if (lift_x_even_impl(&x, out)) return 1;
        FieldElement tmp;
        field_add_impl(&tmp, &x, &one);
        x = tmp;
    }
    return 0;
}

// -- Single commitment: C = v*H + r*G ----------------------------------------
inline void pedersen_commit_impl(const Scalar* value,
                                  const Scalar* blinding,
                                  const AffinePoint* H,
                                  JacobianPoint* out)
{
    JacobianPoint vH, rG;

    JacobianPoint H_jac;
    H_jac.x = H->x; H_jac.y = H->y;
    H_jac.z.limbs[0] = 1; H_jac.z.limbs[1] = 0;
    H_jac.z.limbs[2] = 0; H_jac.z.limbs[3] = 0;
    H_jac.infinity = 0;

    scalar_mul_impl(&vH, &H_jac, value);
    scalar_mul_impl(&rG, &GENERATOR_POINT, blinding);

    point_add_impl(out, &vH, &rG);
}

// -- Batch kernel -------------------------------------------------------------
__kernel void pedersen_commit_batch_kernel(
    __global const Scalar* values,
    __global const Scalar* blindings,
    __global const AffinePoint* H_gen,
    __global AffinePoint* commitments_out,
    uint count)
{
    uint idx = get_global_id(0);
    if (idx >= count) return;

    Scalar v = values[idx];
    Scalar b = blindings[idx];
    AffinePoint H = H_gen[0];
    JacobianPoint result;
    pedersen_commit_impl(&v, &b, &H, &result);

    FieldElement z_inv, z_inv2, z_inv3;
    field_inv_impl(&z_inv, &result.z);
    field_sqr_impl(&z_inv2, &z_inv);
    field_mul_impl(&z_inv3, &z_inv2, &z_inv);

    field_mul_impl(&commitments_out[idx].x, &result.x, &z_inv2);
    field_mul_impl(&commitments_out[idx].y, &result.y, &z_inv3);
}

// -- Verify sum: sum(pos) - sum(neg) == O -------------------------------------
__kernel void pedersen_verify_sum_kernel(
    __global const AffinePoint* pos,
    uint n_pos,
    __global const AffinePoint* neg,
    uint n_neg,
    __global uint* result)
{
    if (get_global_id(0) != 0) return;

    JacobianPoint sum;
    sum.infinity = 1;
    sum.x.limbs[0] = 0; sum.x.limbs[1] = 0;
    sum.x.limbs[2] = 0; sum.x.limbs[3] = 0;
    sum.y.limbs[0] = 0; sum.y.limbs[1] = 0;
    sum.y.limbs[2] = 0; sum.y.limbs[3] = 0;
    sum.z.limbs[0] = 1; sum.z.limbs[1] = 0;
    sum.z.limbs[2] = 0; sum.z.limbs[3] = 0;

    for (uint i = 0; i < n_pos; ++i) {
        AffinePoint ap = pos[i];
        JacobianPoint jp;
        jp.x = ap.x; jp.y = ap.y;
        jp.z.limbs[0] = 1; jp.z.limbs[1] = 0;
        jp.z.limbs[2] = 0; jp.z.limbs[3] = 0;
        jp.infinity = 0;
        JacobianPoint tmp;
        point_add_impl(&tmp, &sum, &jp);
        sum = tmp;
    }

    for (uint i = 0; i < n_neg; ++i) {
        AffinePoint ap = neg[i];
        FieldElement neg_y;
        field_neg_impl(&neg_y, &ap.y);
        JacobianPoint jp;
        jp.x = ap.x; jp.y = neg_y;
        jp.z.limbs[0] = 1; jp.z.limbs[1] = 0;
        jp.z.limbs[2] = 0; jp.z.limbs[3] = 0;
        jp.infinity = 0;
        JacobianPoint tmp;
        point_add_impl(&tmp, &sum, &jp);
        sum = tmp;
    }

    // Check infinity
    int is_inf = sum.infinity;
    if (!is_inf) {
        is_inf = 1;
        for (int j = 0; j < 4; ++j)
            if (sum.z.limbs[j] != 0) is_inf = 0;
    }
    result[0] = is_inf ? 1 : 0;
}

#endif // SECP256K1_PEDERSEN_CL
