// =============================================================================
// Affine Point Addition for secp256k1 (OpenCL Device Functions)
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

#ifndef SECP256K1_AFFINE_CL
#define SECP256K1_AFFINE_CL

// Requires secp256k1_field.cl and secp256k1_point.cl to be included first

// ---------------------------------------------------------------------------
// affine_add_impl: P + Q → R, all affine (2M + 1S total)
// Caller must ensure P.x ≠ Q.x (no doubling, no identity).
// For batch pipelines where all points are distinct by construction.
// ---------------------------------------------------------------------------
inline void affine_add_impl(AffinePoint* r,
                             const FieldElement* px, const FieldElement* py,
                             const FieldElement* qx, const FieldElement* qy) {
    FieldElement h, rr, t, lam;

    field_sub_impl(&h, qx, px);       // H = Q.x - P.x
    field_sub_impl(&rr, qy, py);      // rr = Q.y - P.y
    field_inv_impl(&t, &h);           // t = H^{-1} (expensive)
    field_mul_impl(&lam, &rr, &t);    // λ = rr / H                     [1M]

    field_sqr_impl(&r->x, &lam);     // X3 = λ²                        [1S]
    field_sub_impl(&r->x, &r->x, px); // X3 -= P.x
    field_sub_impl(&r->x, &r->x, qx); // X3 -= Q.x

    field_sub_impl(&r->y, px, &r->x); // t = P.x - X3
    field_mul_impl(&r->y, &lam, &r->y); // Y3 = λ·(P.x - X3)           [1M]
    field_sub_impl(&r->y, &r->y, py); // Y3 -= P.y
}

// ---------------------------------------------------------------------------
// affine_add_x_only_impl: P + Q → X3 only (1M + 1S with pre-inverted H)
// Returns only the X coordinate — for search pipelines where Y is not needed.
//   h_inv: precomputed (Q.x - P.x)^{-1} from batch inversion
// ---------------------------------------------------------------------------
inline void affine_add_x_only_impl(FieldElement* rx,
                                    const FieldElement* px, const FieldElement* py,
                                    const FieldElement* qx, const FieldElement* qy,
                                    const FieldElement* h_inv) {
    FieldElement rr, lam;

    field_sub_impl(&rr, qy, py);          // rr = Q.y - P.y
    field_mul_impl(&lam, &rr, h_inv);     // λ = rr * H^{-1}            [1M]

    field_sqr_impl(rx, &lam);             // X3 = λ²                    [1S]
    field_sub_impl(rx, rx, px);           // X3 -= P.x
    field_sub_impl(rx, rx, qx);           // X3 -= Q.x
}

// ---------------------------------------------------------------------------
// affine_add_lambda_impl: P + Q → (X3, Y3) with pre-inverted H (2M + 1S)
// Full addition with precomputed H^{-1} from batch inversion.
// ---------------------------------------------------------------------------
inline void affine_add_lambda_impl(AffinePoint* r,
                                    const FieldElement* px, const FieldElement* py,
                                    const FieldElement* qx, const FieldElement* qy,
                                    const FieldElement* h_inv) {
    FieldElement rr, lam;

    field_sub_impl(&rr, qy, py);          // rr = Q.y - P.y
    field_mul_impl(&lam, &rr, h_inv);     // λ = rr * H^{-1}            [1M]

    field_sqr_impl(&r->x, &lam);         // X3 = λ²                    [1S]
    field_sub_impl(&r->x, &r->x, px);    // X3 -= P.x
    field_sub_impl(&r->x, &r->x, qx);    // X3 -= Q.x

    field_sub_impl(&r->y, px, &r->x);    // t = P.x - X3
    field_mul_impl(&r->y, &lam, &r->y);   // Y3 = λ·(P.x - X3)         [1M]
    field_sub_impl(&r->y, &r->y, py);     // Y3 -= P.y
}

// ---------------------------------------------------------------------------
// affine_compute_h_impl: compute H = Q.x - P.x for batch inversion
// ---------------------------------------------------------------------------
inline void affine_compute_h_impl(FieldElement* h,
                                   const FieldElement* px,
                                   const FieldElement* qx) {
    field_sub_impl(h, qx, px);
}

// ---------------------------------------------------------------------------
// Batch Inversion (Montgomery's trick) — in-place
// ---------------------------------------------------------------------------
// Input:  h[0..n-1] = H values
// Output: h[0..n-1] = H^{-1} values
// Temp:   prefix[0..n-1] = scratch buffer (same size as h)
//
// Cost: 3(n-1) multiplications + 1 field_inv ≈ 3n + 300 M-eq
// ---------------------------------------------------------------------------
inline void affine_batch_inv_serial_impl(FieldElement* h,
                                          FieldElement* prefix,
                                          int n) {
    // Forward pass: prefix[i] = h[0] * h[1] * ... * h[i]
    prefix[0] = h[0];
    for (int i = 1; i < n; ++i) {
        field_mul_impl(&prefix[i], &prefix[i - 1], &h[i]);
    }

    // Single inversion of the full product
    FieldElement inv;
    field_inv_impl(&inv, &prefix[n - 1]);

    // Backward pass: peel off each h[i]
    for (int i = n - 1; i >= 1; --i) {
        FieldElement h_inv;
        field_mul_impl(&h_inv, &inv, &prefix[i - 1]);
        field_mul_impl(&inv, &inv, &h[i]);
        h[i] = h_inv;
    }
    h[0] = inv;
}

// ---------------------------------------------------------------------------
// Jacobian → Affine conversion (single point)
// ---------------------------------------------------------------------------
inline void jacobian_to_affine_convert_impl(AffinePoint* r,
                                             const FieldElement* x,
                                             const FieldElement* y,
                                             const FieldElement* z) {
    FieldElement z_inv, z_inv2, z_inv3;
    field_inv_impl(&z_inv, z);
    field_sqr_impl(&z_inv2, &z_inv);
    field_mul_impl(&z_inv3, &z_inv, &z_inv2);

    field_mul_impl(&r->x, x, &z_inv2);
    field_mul_impl(&r->y, y, &z_inv3);
}

// ---------------------------------------------------------------------------
// Batch Jacobian → Affine (Montgomery's trick on Z values)
// ---------------------------------------------------------------------------
inline void batch_jacobian_to_affine_serial_impl(FieldElement* x,       // [n] J.X → affine x
                                                  FieldElement* y,       // [n] J.Y → affine y
                                                  FieldElement* z,       // [n] J.Z → scratch
                                                  FieldElement* prefix,  // [n] scratch
                                                  int n) {
    // Batch invert Z values
    prefix[0] = z[0];
    for (int i = 1; i < n; ++i) {
        field_mul_impl(&prefix[i], &prefix[i - 1], &z[i]);
    }

    FieldElement inv;
    field_inv_impl(&inv, &prefix[n - 1]);

    // Backward: recover z[i]^{-1} and convert each point
    for (int i = n - 1; i >= 1; --i) {
        FieldElement z_inv;
        field_mul_impl(&z_inv, &inv, &prefix[i - 1]);
        field_mul_impl(&inv, &inv, &z[i]);

        FieldElement z_inv2, z_inv3;
        field_sqr_impl(&z_inv2, &z_inv);
        field_mul_impl(&z_inv3, &z_inv, &z_inv2);
        field_mul_impl(&x[i], &x[i], &z_inv2);
        field_mul_impl(&y[i], &y[i], &z_inv3);
    }
    // Point 0
    FieldElement z_inv2, z_inv3;
    field_sqr_impl(&z_inv2, &inv);
    field_mul_impl(&z_inv3, &inv, &z_inv2);
    field_mul_impl(&x[0], &x[0], &z_inv2);
    field_mul_impl(&y[0], &y[0], &z_inv3);
}

// =============================================================================
// Benchmark / dispatch kernels for affine operations
// =============================================================================

// Full affine add (2M + 1S + per-element inv)
__kernel void affine_add(
    __global const FieldElement* px, __global const FieldElement* py,
    __global const FieldElement* qx, __global const FieldElement* qy,
    __global FieldElement* rx, __global FieldElement* ry,
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    FieldElement lpx = px[gid], lpy = py[gid];
    FieldElement lqx = qx[gid], lqy = qy[gid];
    AffinePoint r;
    affine_add_impl(&r, &lpx, &lpy, &lqx, &lqy);
    rx[gid] = r.x;
    ry[gid] = r.y;
}

// Affine add with pre-inverted H — full X,Y (2M + 1S)
__kernel void affine_add_lambda(
    __global const FieldElement* px, __global const FieldElement* py,
    __global const FieldElement* qx, __global const FieldElement* qy,
    __global const FieldElement* h_inv,
    __global FieldElement* rx, __global FieldElement* ry,
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    FieldElement lpx = px[gid], lpy = py[gid];
    FieldElement lqx = qx[gid], lqy = qy[gid];
    FieldElement lhinv = h_inv[gid];
    AffinePoint r;
    affine_add_lambda_impl(&r, &lpx, &lpy, &lqx, &lqy, &lhinv);
    rx[gid] = r.x;
    ry[gid] = r.y;
}

// Affine add X-only with pre-inverted H (1M + 1S)
__kernel void affine_add_x_only(
    __global const FieldElement* px, __global const FieldElement* py,
    __global const FieldElement* qx, __global const FieldElement* qy,
    __global const FieldElement* h_inv,
    __global FieldElement* rx,
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    FieldElement lpx = px[gid], lpy = py[gid];
    FieldElement lqx = qx[gid], lqy = qy[gid];
    FieldElement lhinv = h_inv[gid];
    FieldElement lrx;
    affine_add_x_only_impl(&lrx, &lpx, &lpy, &lqx, &lqy, &lhinv);
    rx[gid] = lrx;
}

// Jacobian → Affine conversion (per-element)
__kernel void jacobian_to_affine(
    __global const FieldElement* jx,
    __global const FieldElement* jy,
    __global const FieldElement* jz,
    __global FieldElement* ax, __global FieldElement* ay,
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    FieldElement lx = jx[gid], ly = jy[gid], lz = jz[gid];
    AffinePoint r;
    jacobian_to_affine_convert_impl(&r, &lx, &ly, &lz);
    ax[gid] = r.x;
    ay[gid] = r.y;
}

#endif // SECP256K1_AFFINE_CL
