// =============================================================================
// Affine Point Addition for secp256k1 (Metal Shader Functions)
// =============================================================================
// Pure affine-coordinate arithmetic: no Z coordinate, no projective overhead.
//
// When both points are in affine form (Z=1), the addition formula is:
//   λ  = (Q.y - P.y) / (Q.x - P.x)     [= rr * H^{-1}]
//   X3 = λ² - P.x - Q.x                 [1S + 2 subs]
//   Y3 = λ·(P.x - X3) - P.y             [1M + 1 sub]
//
// Cost per addition: 1M (λ=rr*h_inv) + 1S (λ²) + 1M (λ*(Px-X3)) = 2M + 1S
// With batch inversion: 1M + 1S per slot (the inversion is amortized).
//
// Comparison vs Jacobian mixed add (8M + 3S): ~3.5× fewer operations per add.
// =============================================================================

#pragma once
#include "secp256k1_field.h"
#include "secp256k1_point.h"

// =============================================================================
// affine_add: P + Q → R, all affine (2M + 1S total)
// Caller must ensure P.x ≠ Q.x (no doubling, no identity).
// For batch pipelines where all points are distinct by construction.
// =============================================================================
inline AffinePoint affine_add(thread const FieldElement &px, thread const FieldElement &py,
                               thread const FieldElement &qx, thread const FieldElement &qy) {
    FieldElement h   = field_sub(qx, px);       // H = Q.x - P.x
    FieldElement rr  = field_sub(qy, py);       // rr = Q.y - P.y
    FieldElement t   = field_inv(h);             // t = H^{-1} (expensive)
    FieldElement lam = field_mul(rr, t);         // λ = rr / H            [1M]

    AffinePoint r;
    r.x = field_sqr(lam);                       // X3 = λ²               [1S]
    r.x = field_sub(r.x, px);                   // X3 -= P.x
    r.x = field_sub(r.x, qx);                   // X3 -= Q.x

    r.y = field_sub(px, r.x);                   // t = P.x - X3
    r.y = field_mul(lam, r.y);                   // Y3 = λ·(P.x - X3)    [1M]
    r.y = field_sub(r.y, py);                    // Y3 -= P.y

    return r;
}

// =============================================================================
// affine_add_x_only: P + Q → X3 only (1M + 1S with pre-inverted H)
// Returns only the X coordinate — for search pipelines where Y is not needed.
//   h_inv: precomputed (Q.x - P.x)^{-1} from batch inversion
// =============================================================================
inline FieldElement affine_add_x_only(thread const FieldElement &px, thread const FieldElement &py,
                                       thread const FieldElement &qx, thread const FieldElement &qy,
                                       thread const FieldElement &h_inv) {
    FieldElement rr  = field_sub(qy, py);        // rr = Q.y - P.y
    FieldElement lam = field_mul(rr, h_inv);     // λ = rr * H^{-1}      [1M]

    FieldElement rx = field_sqr(lam);            // X3 = λ²              [1S]
    rx = field_sub(rx, px);                      // X3 -= P.x
    rx = field_sub(rx, qx);                      // X3 -= Q.x
    return rx;
}

// =============================================================================
// affine_add_lambda: P + Q → (X3, Y3) with pre-inverted H (2M + 1S)
// Full addition with precomputed H^{-1} from batch inversion.
// =============================================================================
inline AffinePoint affine_add_lambda(thread const FieldElement &px, thread const FieldElement &py,
                                      thread const FieldElement &qx, thread const FieldElement &qy,
                                      thread const FieldElement &h_inv) {
    FieldElement rr  = field_sub(qy, py);        // rr = Q.y - P.y
    FieldElement lam = field_mul(rr, h_inv);     // λ = rr * H^{-1}      [1M]

    AffinePoint r;
    r.x = field_sqr(lam);                       // X3 = λ²               [1S]
    r.x = field_sub(r.x, px);                   // X3 -= P.x
    r.x = field_sub(r.x, qx);                   // X3 -= Q.x

    r.y = field_sub(px, r.x);                   // t = P.x - X3
    r.y = field_mul(lam, r.y);                   // Y3 = λ·(P.x - X3)    [1M]
    r.y = field_sub(r.y, py);                    // Y3 -= P.y

    return r;
}

// =============================================================================
// affine_compute_h: compute H = Q.x - P.x for batch inversion
// =============================================================================
inline FieldElement affine_compute_h(thread const FieldElement &px,
                                      thread const FieldElement &qx) {
    return field_sub(qx, px);
}

// =============================================================================
// Batch Inversion (Montgomery's trick) — in-place on thread-local arrays
// =============================================================================
// Input:  h[0..n-1] = H values
// Output: h[0..n-1] = H^{-1} values
// Temp:   prefix[0..n-1] = scratch buffer (same size as h)
//
// Cost: 3(n-1) multiplications + 1 field_inv ≈ 3n + 300 M-eq
// =============================================================================
inline void affine_batch_inv_serial(thread FieldElement* h,
                                     thread FieldElement* prefix,
                                     int n) {
    // Forward pass: prefix[i] = h[0] * h[1] * ... * h[i]
    prefix[0] = h[0];
    for (int i = 1; i < n; ++i) {
        prefix[i] = field_mul(prefix[i - 1], h[i]);
    }

    // Single inversion of the full product
    FieldElement inv = field_inv(prefix[n - 1]);

    // Backward pass: peel off each h[i]
    for (int i = n - 1; i >= 1; --i) {
        FieldElement h_inv = field_mul(inv, prefix[i - 1]);
        inv = field_mul(inv, h[i]);
        h[i] = h_inv;
    }
    h[0] = inv;
}

// =============================================================================
// Jacobian → Affine conversion (single point)
// =============================================================================
inline AffinePoint jacobian_to_affine_convert(thread const JacobianPoint &p) {
    FieldElement z_inv  = field_inv(p.z);
    FieldElement z_inv2 = field_sqr(z_inv);
    FieldElement z_inv3 = field_mul(z_inv, z_inv2);

    AffinePoint r;
    r.x = field_mul(p.x, z_inv2);
    r.y = field_mul(p.y, z_inv3);
    return r;
}

// =============================================================================
// Batch Jacobian → Affine (Montgomery's trick on Z values)
// =============================================================================
inline void batch_jacobian_to_affine_serial(thread FieldElement* x,       // [n] J.X → affine x
                                             thread FieldElement* y,       // [n] J.Y → affine y
                                             thread FieldElement* z,       // [n] J.Z → scratch
                                             thread FieldElement* prefix,  // [n] scratch
                                             int n) {
    // Batch invert Z values
    prefix[0] = z[0];
    for (int i = 1; i < n; ++i) {
        prefix[i] = field_mul(prefix[i - 1], z[i]);
    }

    FieldElement inv = field_inv(prefix[n - 1]);

    // Backward: recover z[i]^{-1} and convert each point
    for (int i = n - 1; i >= 1; --i) {
        FieldElement z_inv  = field_mul(inv, prefix[i - 1]);
        inv = field_mul(inv, z[i]);

        FieldElement z_inv2 = field_sqr(z_inv);
        FieldElement z_inv3 = field_mul(z_inv, z_inv2);
        x[i] = field_mul(x[i], z_inv2);
        y[i] = field_mul(y[i], z_inv3);
    }
    // Point 0
    FieldElement z_inv2 = field_sqr(inv);
    FieldElement z_inv3 = field_mul(inv, z_inv2);
    x[0] = field_mul(x[0], z_inv2);
    y[0] = field_mul(y[0], z_inv3);
}
