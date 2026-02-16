// =============================================================================
// Affine Point Addition for secp256k1 (CUDA Device Functions)
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
#include "secp256k1.cuh"

namespace secp256k1{
namespace cuda {

// ---------------------------------------------------------------------------
// affine_add: P + Q → R, all affine (2M + 1S total)
// Caller must ensure P.x ≠ Q.x (no doubling, no identity).
// For batch pipelines where all points are distinct by construction.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void affine_add(
    const FieldElement* __restrict__ px, const FieldElement* __restrict__ py,
    const FieldElement* __restrict__ qx, const FieldElement* __restrict__ qy,
    FieldElement* __restrict__ rx, FieldElement* __restrict__ ry
) {
    FieldElement h, rr, lambda, t;

    field_sub(qx, px, &h);       // H = Q.x - P.x
    field_sub(qy, py, &rr);      // rr = Q.y - P.y
    field_inv(&h, &t);           // t = H^{-1} (expensive — use batch version below)
    field_mul(&rr, &t, &lambda); // λ = rr / H                          [1M]

    field_sqr(&lambda, rx);      // X3 = λ²                             [1S]
    field_sub(rx, px, rx);       // X3 -= P.x
    field_sub(rx, qx, rx);      // X3 -= Q.x

    field_sub(px, rx, ry);      // t = P.x - X3
    field_mul(&lambda, ry, ry);  // Y3 = λ·(P.x - X3)                  [1M]
    field_sub(ry, py, ry);      // Y3 -= P.y
}

// ---------------------------------------------------------------------------
// affine_add_x_only: P + Q → X3 only (1M + 1S with pre-inverted H)
// Returns only the X coordinate — for search pipelines where Y is not needed.
//   h_inv: precomputed (Q.x - P.x)^{-1} from batch inversion
// ---------------------------------------------------------------------------
__device__ __forceinline__ void affine_add_x_only(
    const FieldElement* __restrict__ px, const FieldElement* __restrict__ py,
    const FieldElement* __restrict__ qx, const FieldElement* __restrict__ qy,
    const FieldElement* __restrict__ h_inv,
    FieldElement* __restrict__ rx
) {
    FieldElement rr, lambda;

    field_sub(qy, py, &rr);          // rr = Q.y - P.y
    field_mul(&rr, h_inv, &lambda);   // λ = rr * H^{-1}                [1M]

    field_sqr(&lambda, rx);          // X3 = λ²                         [1S]
    field_sub(rx, px, rx);           // X3 -= P.x
    field_sub(rx, qx, rx);          // X3 -= Q.x
}

// ---------------------------------------------------------------------------
// affine_add_lambda: P + Q → (X3, Y3) with pre-inverted H (2M + 1S)
// Full addition with precomputed H^{-1} from batch inversion.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void affine_add_lambda(
    const FieldElement* __restrict__ px, const FieldElement* __restrict__ py,
    const FieldElement* __restrict__ qx, const FieldElement* __restrict__ qy,
    const FieldElement* __restrict__ h_inv,
    FieldElement* __restrict__ rx, FieldElement* __restrict__ ry
) {
    FieldElement rr, lambda;

    field_sub(qy, py, &rr);          // rr = Q.y - P.y
    field_mul(&rr, h_inv, &lambda);   // λ = rr * H^{-1}                [1M]

    field_sqr(&lambda, rx);          // X3 = λ²                         [1S]
    field_sub(rx, px, rx);           // X3 -= P.x
    field_sub(rx, qx, rx);          // X3 -= Q.x

    field_sub(px, rx, ry);          // t = P.x - X3
    field_mul(&lambda, ry, ry);      // Y3 = λ·(P.x - X3)              [1M]
    field_sub(ry, py, ry);          // Y3 -= P.y
}

// ---------------------------------------------------------------------------
// affine_compute_h: compute H = Q.x - P.x for batch inversion
// Just a subtraction — essentially free.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void affine_compute_h(
    const FieldElement* __restrict__ px,
    const FieldElement* __restrict__ qx,
    FieldElement* __restrict__ h
) {
    field_sub(qx, px, h);
}

// ---------------------------------------------------------------------------
// Batch Inversion (Montgomery's trick) — in-place
// ---------------------------------------------------------------------------
// Input:  h[0..n-1] = H values
// Output: h[0..n-1] = H^{-1} values
// Temp:   prefix[0..n-1] = scratch buffer (same size as h)
//
// Cost: 3(n-1) multiplications + 1 field_inv ≈ 3n + 300 M-eq
//
// This is a device function for use WITHIN a single thread.
// For a kernel version, build prefix products per-thread over strided data.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void affine_batch_inv_serial(
    FieldElement* __restrict__ h,
    FieldElement* __restrict__ prefix,
    int n
) {
    // Forward pass: prefix[i] = h[0] * h[1] * ... * h[i]
    prefix[0] = h[0];
    for (int i = 1; i < n; ++i) {
        field_mul(&prefix[i - 1], &h[i], &prefix[i]);
    }

    // Single inversion of the full product
    FieldElement inv;
    field_inv(&prefix[n - 1], &inv);

    // Backward pass: peel off each h[i]
    for (int i = n - 1; i >= 1; --i) {
        // h[i]^{-1} = inv * prefix[i-1]
        FieldElement h_inv;
        field_mul(&inv, &prefix[i - 1], &h_inv);
        // Update inv: inv = inv * h[i] (original value)
        field_mul(&inv, &h[i], &inv);
        h[i] = h_inv;
    }
    h[0] = inv;  // Last remaining inverse
}

// ---------------------------------------------------------------------------
// Jacobian → Affine conversion (single point, in-place on x/y)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void jacobian_to_affine(
    FieldElement* __restrict__ x,
    FieldElement* __restrict__ y,
    const FieldElement* __restrict__ z
) {
    FieldElement z_inv, z_inv2, z_inv3;
    field_inv(z, &z_inv);
    field_sqr(&z_inv, &z_inv2);
    field_mul(&z_inv, &z_inv2, &z_inv3);

    FieldElement ax, ay;
    field_mul(x, &z_inv2, &ax);
    field_mul(y, &z_inv3, &ay);
    *x = ax;
    *y = ay;
}

// ---------------------------------------------------------------------------
// Batch Jacobian → Affine (batch of Z values → Z^{-2}, Z^{-3})
// Uses Montgomery's trick on the Z values themselves
// ---------------------------------------------------------------------------
__device__ __forceinline__ void batch_jacobian_to_affine_serial(
    FieldElement* __restrict__ x,    // [n] Jacobian X → affine x
    FieldElement* __restrict__ y,    // [n] Jacobian Y → affine y
    FieldElement* __restrict__ z,    // [n] Jacobian Z → scratch (destroyed)
    FieldElement* __restrict__ prefix, // [n] scratch
    int n
) {
    // Batch invert Z values
    // Forward prefix product of Z
    prefix[0] = z[0];
    for (int i = 1; i < n; ++i) {
        field_mul(&prefix[i - 1], &z[i], &prefix[i]);
    }

    FieldElement inv;
    field_inv(&prefix[n - 1], &inv);

    // Backward: recover z[i]^{-1} and convert each point
    for (int i = n - 1; i >= 1; --i) {
        FieldElement z_inv;
        field_mul(&inv, &prefix[i - 1], &z_inv);
        field_mul(&inv, &z[i], &inv);

        // Convert point i
        FieldElement z_inv2, z_inv3;
        field_sqr(&z_inv, &z_inv2);
        field_mul(&z_inv, &z_inv2, &z_inv3);
        field_mul(&x[i], &z_inv2, &x[i]);
        field_mul(&y[i], &z_inv3, &y[i]);
    }
    // Point 0
    FieldElement z_inv2, z_inv3;
    field_sqr(&inv, &z_inv2);
    field_mul(&inv, &z_inv2, &z_inv3);
    field_mul(&x[0], &z_inv2, &x[0]);
    field_mul(&y[0], &z_inv3, &y[0]);
}

} // namespace cuda
} // namespace secp256k1
