// =============================================================================
// secp256k1_msm.cl -- Multi-Scalar Multiplication for OpenCL
// =============================================================================
// R = s_1*P_1 + s_2*P_2 + ... + s_n*P_n
// Two variants:
//   1. msm_naive_impl:     O(256n) simple sequential
//   2. msm_pippenger_impl: O(n/c + 2^c) bucket method
// =============================================================================

#ifndef SECP256K1_MSM_CL
#define SECP256K1_MSM_CL

// -- Naive MSM (small n) ------------------------------------------------------
inline void msm_naive_impl(const Scalar* scalars,
                            const JacobianPoint* points,
                            int n,
                            JacobianPoint* result)
{
    result->infinity = 1;
    for (int i = 0; i < 4; ++i) {
        result->x.limbs[i] = 0;
        result->y.limbs[i] = 0;
    }
    result->z.limbs[0] = 1;
    result->z.limbs[1] = 0;
    result->z.limbs[2] = 0;
    result->z.limbs[3] = 0;

    for (int i = 0; i < n; ++i) {
        int is_zero = 1;
        for (int j = 0; j < 4; ++j)
            if (scalars[i].limbs[j] != 0) is_zero = 0;
        if (is_zero) continue;

        JacobianPoint tmp;
        scalar_mul_impl(&tmp, &points[i], &scalars[i]);
        if (result->infinity) {
            *result = tmp;
        } else {
            JacobianPoint sum;
            point_add_impl(&sum, result, &tmp);
            *result = sum;
        }
    }
}

// -- Scalar get_window --------------------------------------------------------
inline uint scalar_get_window_impl(const Scalar* s, int window_idx, int window_bits) {
    int bit_offset = window_idx * window_bits;
    int limb_idx = bit_offset / 64;
    int bit_idx = bit_offset % 64;
    if (limb_idx >= 4) return 0;

    uint val = (uint)((s->limbs[limb_idx] >> bit_idx) & ((1u << window_bits) - 1));
    int bits_from_first = 64 - bit_idx;
    if (bits_from_first < window_bits && limb_idx + 1 < 4) {
        int remaining = window_bits - bits_from_first;
        val |= (uint)(s->limbs[limb_idx + 1] & ((1UL << remaining) - 1)) << bits_from_first;
    }
    return val;
}

// -- Pippenger MSM with caller-provided buckets ------------------------------
inline void msm_pippenger_impl(const Scalar* scalars,
                                const JacobianPoint* points,
                                int n,
                                JacobianPoint* result,
                                JacobianPoint* buckets,
                                int c)
{
    int num_buckets = 1 << c;
    int num_windows = (256 + c - 1) / c;

    result->infinity = 1;
    for (int i = 0; i < 4; ++i) {
        result->x.limbs[i] = 0; result->y.limbs[i] = 0;
    }
    result->z.limbs[0] = 1; result->z.limbs[1] = 0;
    result->z.limbs[2] = 0; result->z.limbs[3] = 0;

    for (int w = num_windows - 1; w >= 0; --w) {
        if (!result->infinity) {
            for (int d = 0; d < c; ++d) {
                JacobianPoint doubled;
                point_double_impl(&doubled, result);
                *result = doubled;
            }
        }

        // Clear buckets
        for (int b = 0; b < num_buckets; ++b) {
            buckets[b].infinity = 1;
            for (int j = 0; j < 4; ++j) {
                buckets[b].x.limbs[j] = 0;
                buckets[b].y.limbs[j] = 0;
            }
            buckets[b].z.limbs[0] = 1; buckets[b].z.limbs[1] = 0;
            buckets[b].z.limbs[2] = 0; buckets[b].z.limbs[3] = 0;
        }

        // Scatter
        for (int i = 0; i < n; ++i) {
            uint digit = scalar_get_window_impl(&scalars[i], w, c);
            if (digit == 0) continue;
            if (buckets[digit].infinity) {
                buckets[digit] = points[i];
            } else {
                JacobianPoint sum;
                point_add_impl(&sum, &buckets[digit], &points[i]);
                buckets[digit] = sum;
            }
        }

        // Aggregate
        JacobianPoint running_sum, partial_sum;
        running_sum.infinity = 1;
        for (int j = 0; j < 4; ++j) { running_sum.x.limbs[j] = 0; running_sum.y.limbs[j] = 0; }
        running_sum.z.limbs[0] = 1; running_sum.z.limbs[1] = 0;
        running_sum.z.limbs[2] = 0; running_sum.z.limbs[3] = 0;
        partial_sum = running_sum;

        for (int b = num_buckets - 1; b >= 1; --b) {
            if (!buckets[b].infinity) {
                if (running_sum.infinity) {
                    running_sum = buckets[b];
                } else {
                    JacobianPoint sum;
                    point_add_impl(&sum, &running_sum, &buckets[b]);
                    running_sum = sum;
                }
            }
            if (!running_sum.infinity) {
                if (partial_sum.infinity) {
                    partial_sum = running_sum;
                } else {
                    JacobianPoint sum;
                    point_add_impl(&sum, &partial_sum, &running_sum);
                    partial_sum = sum;
                }
            }
        }

        if (!partial_sum.infinity) {
            if (result->infinity) {
                *result = partial_sum;
            } else {
                JacobianPoint sum;
                point_add_impl(&sum, result, &partial_sum);
                *result = sum;
            }
        }
    }
}

// -- Optimal window width -----------------------------------------------------
inline int msm_optimal_window_impl(int n) {
    if (n <= 1)    return 1;
    if (n <= 4)    return 2;
    if (n <= 16)   return 3;
    if (n <= 64)   return 4;
    if (n <= 256)  return 5;
    if (n <= 1024) return 6;
    if (n <= 4096) return 7;
    return 8;
}

// -- Small MSM with stack buckets (c=4, 16 buckets) ---------------------------
inline void msm_small_impl(const Scalar* scalars,
                             const JacobianPoint* points,
                             int n,
                             JacobianPoint* result)
{
    if (n <= 0) {
        result->infinity = 1;
        for (int i = 0; i < 4; ++i) {
            result->x.limbs[i] = 0; result->y.limbs[i] = 0;
        }
        result->z.limbs[0] = 1; result->z.limbs[1] = 0;
        result->z.limbs[2] = 0; result->z.limbs[3] = 0;
        return;
    }
    if (n <= 2) { msm_naive_impl(scalars, points, n, result); return; }

    JacobianPoint buckets[16];
    msm_pippenger_impl(scalars, points, n, result, buckets, 4);
}

// -- Batch scatter kernel (each thread does one scalar*point) -----------------
__kernel void msm_scatter_kernel(
    __global const Scalar* scalars,
    __global const JacobianPoint* points,
    __global JacobianPoint* partial_results,
    uint n)
{
    uint idx = get_global_id(0);
    if (idx >= n) return;

    Scalar s = scalars[idx];
    JacobianPoint p = points[idx];
    JacobianPoint r;
    scalar_mul_impl(&r, &p, &s);
    partial_results[idx] = r;
}

#endif // SECP256K1_MSM_CL
