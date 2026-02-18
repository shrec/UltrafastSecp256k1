#pragma once
// ============================================================================
// Multi-Scalar Multiplication (MSM) — CUDA device implementation
// ============================================================================
// Device-callable MSM using Pippenger bucket method:
//   R = s₁·P₁ + s₂·P₂ + ... + sₙ·Pₙ
//
// Two variants:
//   1. msm_naive:     O(256n) — simple sequential scalar_mul + add
//   2. msm_pippenger: O(n/c + 2^c) per window — bucket method
//
// For GPU-parallel MSM across many threads, use the batch kernel.
//
// 64-bit limb mode only.
// ============================================================================

#include "secp256k1.cuh"

#if !SECP256K1_CUDA_LIMBS_32

namespace secp256k1 {
namespace cuda {

// ── Naive MSM (small n) ──────────────────────────────────────────────────────
// Simple sum of individual scalar multiplications.
// Best for n <= ~4.

__device__ inline void msm_naive(
    const Scalar* scalars,
    const JacobianPoint* points,
    int n,
    JacobianPoint* result)
{
    result->infinity = true;
    field_set_zero(&result->x);
    field_set_zero(&result->y);
    field_set_one(&result->z);

    for (int i = 0; i < n; i++) {
        if (scalar_is_zero(&scalars[i])) continue;
        JacobianPoint tmp;
        scalar_mul(&points[i], &scalars[i], &tmp);
        if (result->infinity) {
            *result = tmp;
        } else {
            JacobianPoint sum;
            jacobian_add(result, &tmp, &sum);
            *result = sum;
        }
    }
}

// ── Scalar digit extraction ─────────────────────────────────────────────────
// Extract c-bit window from scalar at position `window_idx` (from LSB).

__device__ inline unsigned scalar_get_window(
    const Scalar* s,
    int window_idx,
    int window_bits)
{
    int bit_offset = window_idx * window_bits;
    int limb_idx = bit_offset / 64;
    int bit_idx = bit_offset % 64;

    if (limb_idx >= 4) return 0;

    unsigned val = (unsigned)((s->limbs[limb_idx] >> bit_idx) & ((1u << window_bits) - 1));

    // Handle cross-limb boundary
    int bits_from_first = 64 - bit_idx;
    if (bits_from_first < window_bits && limb_idx + 1 < 4) {
        int remaining = window_bits - bits_from_first;
        val |= (unsigned)(s->limbs[limb_idx + 1] & ((1ULL << remaining) - 1)) << bits_from_first;
    }

    return val;
}

// ── Pippenger MSM ────────────────────────────────────────────────────────────
// Bucket method: optimal for n > ~8.
//
// Parameters:
//   scalars  - array of n scalars
//   points   - array of n points (Jacobian)
//   n        - count (max supported: limited by stack/registers)
//   result   - output point
//   buckets  - caller-provided scratch space for (2^c) JacobianPoints
//   c        - window width in bits (typically 4-8)
//
// The caller must provide a bucket array of size (1 << c).

__device__ inline void msm_pippenger_with_buckets(
    const Scalar* scalars,
    const JacobianPoint* points,
    int n,
    JacobianPoint* result,
    JacobianPoint* buckets,
    int c)
{
    int num_buckets = 1 << c;
    int num_windows = (256 + c - 1) / c;

    result->infinity = true;
    field_set_zero(&result->x);
    field_set_zero(&result->y);
    field_set_one(&result->z);

    // Process windows from MSB to LSB
    for (int w = num_windows - 1; w >= 0; w--) {
        // Double result c times (shift left by c bits)
        if (!result->infinity) {
            for (int d = 0; d < c; d++) {
                JacobianPoint doubled;
                jacobian_double(result, &doubled);
                *result = doubled;
            }
        }

        // Clear buckets
        for (int b = 0; b < num_buckets; b++) {
            buckets[b].infinity = true;
            field_set_zero(&buckets[b].x);
            field_set_zero(&buckets[b].y);
            field_set_one(&buckets[b].z);
        }

        // Scatter: place each point into its bucket
        for (int i = 0; i < n; i++) {
            unsigned digit = scalar_get_window(&scalars[i], w, c);
            if (digit == 0) continue;

            if (buckets[digit].infinity) {
                buckets[digit] = points[i];
            } else {
                JacobianPoint sum;
                jacobian_add(&buckets[digit], &points[i], &sum);
                buckets[digit] = sum;
            }
        }

        // Aggregate buckets: Σ = Σ_{b=1}^{num_buckets-1} b · bucket[b]
        // Efficient bottom-up: running_sum accumulates, partial_sum sums running
        JacobianPoint running_sum, partial_sum;
        running_sum.infinity = true;
        field_set_zero(&running_sum.x);
        field_set_zero(&running_sum.y);
        field_set_one(&running_sum.z);
        partial_sum.infinity = true;
        field_set_zero(&partial_sum.x);
        field_set_zero(&partial_sum.y);
        field_set_one(&partial_sum.z);

        for (int b = num_buckets - 1; b >= 1; b--) {
            if (!buckets[b].infinity) {
                if (running_sum.infinity) {
                    running_sum = buckets[b];
                } else {
                    JacobianPoint sum;
                    jacobian_add(&running_sum, &buckets[b], &sum);
                    running_sum = sum;
                }
            }
            if (!running_sum.infinity) {
                if (partial_sum.infinity) {
                    partial_sum = running_sum;
                } else {
                    JacobianPoint sum;
                    jacobian_add(&partial_sum, &running_sum, &sum);
                    partial_sum = sum;
                }
            }
        }

        // Add window contribution to result
        if (!partial_sum.infinity) {
            if (result->infinity) {
                *result = partial_sum;
            } else {
                JacobianPoint sum;
                jacobian_add(result, &partial_sum, &sum);
                *result = sum;
            }
        }
    }
}

// ── Optimal window width ─────────────────────────────────────────────────────
// Returns best c for n points. Minimizes total ops ≈ ceil(256/c)*(n + 2^c).

__device__ inline int msm_optimal_window(int n) {
    if (n <= 1) return 1;
    if (n <= 4) return 2;
    if (n <= 16) return 3;
    if (n <= 64) return 4;
    if (n <= 256) return 5;
    if (n <= 1024) return 6;
    if (n <= 4096) return 7;
    return 8;
}

// ── Convenience MSM with stack-allocated buckets ─────────────────────────────
// For small n, uses stack buckets with c=4 (16 buckets = ~2KB).
// For larger n, caller should provide external bucket storage.

__device__ inline void msm_small(
    const Scalar* scalars,
    const JacobianPoint* points,
    int n,
    JacobianPoint* result)
{
    if (n <= 0) {
        result->infinity = true;
        field_set_zero(&result->x);
        field_set_zero(&result->y);
        field_set_one(&result->z);
        return;
    }
    if (n <= 2) {
        msm_naive(scalars, points, n, result);
        return;
    }

    // Stack-allocated bucket array for c=4 (16 buckets)
    JacobianPoint buckets[16];
    msm_pippenger_with_buckets(scalars, points, n, result, buckets, 4);
}

// ── Batch MSM kernel ─────────────────────────────────────────────────────────
// Each thread computes one scalar*point pair; results are then summed.
// This kernel just does the embarrassingly parallel part.

__global__ void msm_scatter_kernel(
    const Scalar* scalars,
    const JacobianPoint* points,
    JacobianPoint* partial_results,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    scalar_mul(&points[idx], &scalars[idx], &partial_results[idx]);
}

} // namespace cuda
} // namespace secp256k1

#endif // !SECP256K1_CUDA_LIMBS_32
