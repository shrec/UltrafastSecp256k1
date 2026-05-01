// =============================================================================
// secp256k1_msm.h -- Multi-Scalar Multiplication for Metal
// =============================================================================
// R = s_1*P_1 + s_2*P_2 + ... + s_n*P_n
// Two variants:
//   1. msm_naive:     O(256n) simple sequential
//   2. msm_pippenger: O(n/c + 2^c) bucket method
// =============================================================================

#ifndef SECP256K1_MSM_H
#define SECP256K1_MSM_H

// -- Naive MSM ----------------------------------------------------------------
inline JacobianPoint msm_naive_metal(thread const Scalar256* scalars,
                                     thread const JacobianPoint* points,
                                     int n) {
    JacobianPoint result;
    result.infinity = 1;
    result.x = field_zero();
    result.y = field_zero();
    result.z = field_one();

    for (int i = 0; i < n; ++i) {
        bool is_zero = true;
        for (int j = 0; j < 8; ++j)
            if (scalars[i].limbs[j] != 0) is_zero = false;
        if (is_zero) continue;

        JacobianPoint tmp = scalar_mul(points[i], scalars[i]);
        if (result.infinity) {
            result = tmp;
        } else {
            result = point_add(result, tmp);
        }
    }
    return result;
}

// -- Scalar window extraction -------------------------------------------------
inline uint scalar_get_window_metal(thread const Scalar256& s, int window_idx, int window_bits) {
    int bit_offset = window_idx * window_bits;
    int limb_idx = bit_offset / 32;
    int bit_idx = bit_offset % 32;
    if (limb_idx >= 8) return 0;

    uint val = (s.limbs[limb_idx] >> bit_idx) & ((1u << window_bits) - 1);
    int bits_from_first = 32 - bit_idx;
    if (bits_from_first < window_bits && limb_idx + 1 < 8) {
        int remaining = window_bits - bits_from_first;
        val |= (s.limbs[limb_idx + 1] & ((1u << remaining) - 1)) << bits_from_first;
    }
    return val;
}

// -- Pippenger MSM with caller-provided buckets ------------------------------
inline JacobianPoint msm_pippenger_metal(
    thread const Scalar256* scalars,
    thread const JacobianPoint* points,
    int n,
    thread JacobianPoint* buckets,
    int c)
{
    int num_buckets = 1 << c;
    int num_windows = (256 + c - 1) / c;

    JacobianPoint result;
    result.infinity = 1;
    result.x = field_zero(); result.y = field_zero();
    result.z = field_one();

    for (int w = num_windows - 1; w >= 0; --w) {
        if (!result.infinity) {
            for (int d = 0; d < c; ++d)
                result = point_double(result);
        }

        // Clear buckets
        for (int b = 0; b < num_buckets; ++b) {
            buckets[b].infinity = 1;
            buckets[b].x = field_zero(); buckets[b].y = field_zero();
            buckets[b].z = field_one();
        }

        // Scatter
        for (int i = 0; i < n; ++i) {
            uint digit = scalar_get_window_metal(scalars[i], w, c);
            if (digit == 0) continue;
            if (buckets[digit].infinity) {
                buckets[digit] = points[i];
            } else {
                buckets[digit] = point_add(buckets[digit], points[i]);
            }
        }

        // Aggregate
        JacobianPoint running_sum, partial_sum;
        running_sum.infinity = 1;
        running_sum.x = field_zero(); running_sum.y = field_zero();
        running_sum.z = field_one();
        partial_sum = running_sum;

        for (int b = num_buckets - 1; b >= 1; --b) {
            if (!buckets[b].infinity) {
                if (running_sum.infinity) running_sum = buckets[b];
                else running_sum = point_add(running_sum, buckets[b]);
            }
            if (!running_sum.infinity) {
                if (partial_sum.infinity) partial_sum = running_sum;
                else partial_sum = point_add(partial_sum, running_sum);
            }
        }

        if (!partial_sum.infinity) {
            if (result.infinity) result = partial_sum;
            else result = point_add(result, partial_sum);
        }
    }
    return result;
}

// -- Optimal window width -----------------------------------------------------
inline int msm_optimal_window_metal(int n) {
    if (n <= 1)    return 1;
    if (n <= 4)    return 2;
    if (n <= 16)   return 3;
    if (n <= 64)   return 4;
    if (n <= 256)  return 5;
    if (n <= 1024) return 6;
    if (n <= 4096) return 7;
    return 8;
}

// -- Small MSM with stack buckets (c=4) ---------------------------------------
inline JacobianPoint msm_small_metal(thread const Scalar256* scalars,
                                      thread const JacobianPoint* points,
                                      int n) {
    JacobianPoint result;
    result.infinity = 1;
    result.x = field_zero(); result.y = field_zero();
    result.z = field_one();

    if (n <= 0) return result;
    if (n <= 2) return msm_naive_metal(scalars, points, n);

    JacobianPoint buckets[16];
    return msm_pippenger_metal(scalars, points, n, buckets, 4);
}

#endif // SECP256K1_MSM_H
