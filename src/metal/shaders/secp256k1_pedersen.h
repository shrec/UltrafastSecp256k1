// =============================================================================
// secp256k1_pedersen.h -- Pedersen Commitments for Metal
// =============================================================================
// C = v*H + r*G
// Provides:
//   - pedersen_commit_metal: single commitment
//   - lift_x_even_metal: find point with given x and even y
//   - pedersen_verify_sum_metal: homomorphic sum verification
// =============================================================================

#ifndef SECP256K1_PEDERSEN_H
#define SECP256K1_PEDERSEN_H

// -- lift_x to curve point with even y (8x32 limbs) --------------------------
inline bool lift_x_even_metal(thread const FieldElement& x, thread AffinePoint& out) {
    FieldElement x2 = field_sqr(x);
    FieldElement x3 = field_mul(x2, x);

    FieldElement seven = field_zero();
    seven.limbs[0] = 7;
    FieldElement y2 = field_add(x3, seven);

    FieldElement y = field_sqrt(y2);

    // Verify
    FieldElement check = field_sqr(y);
    // Compare via normalized bytes
    uchar cb[32], yb[32];
    for (int i = 0; i < 8; ++i) {
        uint v1 = check.limbs[7-i], v2 = y2.limbs[7-i];
        cb[i*4+0] = (uchar)(v1>>24); cb[i*4+1] = (uchar)(v1>>16);
        cb[i*4+2] = (uchar)(v1>>8);  cb[i*4+3] = (uchar)(v1);
        yb[i*4+0] = (uchar)(v2>>24); yb[i*4+1] = (uchar)(v2>>16);
        yb[i*4+2] = (uchar)(v2>>8);  yb[i*4+3] = (uchar)(v2);
    }
    for (int i = 0; i < 32; ++i)
        if (cb[i] != yb[i]) return false;

    // Even y
    if (y.limbs[0] & 1)
        y = field_negate(y);

    out.x = x;
    out.y = y;
    return true;
}

// -- Hash to point (try-and-increment) ----------------------------------------
inline bool hash_to_point_increment_metal(thread const FieldElement& x_in,
                                           thread AffinePoint& out) {
    FieldElement x = x_in;
    FieldElement one = field_zero();
    one.limbs[0] = 1;
    for (int attempt = 0; attempt < 256; ++attempt) {
        if (lift_x_even_metal(x, out)) return true;
        x = field_add(x, one);
    }
    return false;
}

// -- Single Pedersen commitment: C = v*H + r*G --------------------------------
inline JacobianPoint pedersen_commit_metal(thread const Scalar256& value,
                                            thread const Scalar256& blinding,
                                            thread const AffinePoint& H) {
    JacobianPoint H_jac;
    H_jac.x = H.x; H_jac.y = H.y;
    H_jac.z = field_one(); H_jac.infinity = 0;

    JacobianPoint vH = scalar_mul(H_jac, value);
    JacobianPoint rG = scalar_mul_generator(blinding);

    return point_add(vH, rG);
}

// -- Verify sum: sum(pos) - sum(neg) == O -------------------------------------
inline bool pedersen_verify_sum_metal(const device AffinePoint* pos, uint n_pos,
                                      const device AffinePoint* neg, uint n_neg) {
    JacobianPoint sum;
    sum.infinity = 1;
    sum.x = field_zero(); sum.y = field_zero();
    sum.z = field_one();

    for (uint i = 0; i < n_pos; ++i) {
        JacobianPoint jp;
        jp.x = pos[i].x; jp.y = pos[i].y;
        jp.z = field_one(); jp.infinity = 0;
        sum = point_add(sum, jp);
    }

    for (uint i = 0; i < n_neg; ++i) {
        JacobianPoint jp;
        jp.x = neg[i].x; jp.y = field_negate(neg[i].y);
        jp.z = field_one(); jp.infinity = 0;
        sum = point_add(sum, jp);
    }

    if (sum.infinity) return true;
    bool z_is_zero = true;
    for (int j = 0; j < 8; ++j)
        if (sum.z.limbs[j] != 0) z_is_zero = false;
    return z_is_zero;
}

#endif // SECP256K1_PEDERSEN_H
