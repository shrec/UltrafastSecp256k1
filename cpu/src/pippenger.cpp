// ============================================================================
// Pippenger Bucket Method -- Multi-Scalar Multiplication
// ============================================================================
// Reference: Bernstein et al. "Faster batch forgery identification" (2012)
//
// Bucket method for computing sum(s_i * P_i):
//   For each window of c bits, scatter points into 2^c buckets by digit,
//   aggregate buckets bottom-up (running sum trick), then combine windows.

#include "secp256k1/pippenger.hpp"
#include "secp256k1/multiscalar.hpp"
#include <algorithm>
#include <cstring>

namespace secp256k1 {

using fast::Scalar;
using fast::Point;

// -- Optimal Window Width -----------------------------------------------------
// Minimizes total cost ~= floor(256/c) * (n + 2^c)
// Standard heuristic: c ~= max(1, floor(log2(n)))

unsigned pippenger_optimal_window(std::size_t n) {
    if (n <= 1)    return 1;
    if (n <= 4)    return 2;
    if (n <= 8)    return 3;
    if (n <= 16)   return 4;
    if (n <= 32)   return 5;
    if (n <= 64)   return 6;
    if (n <= 128)  return 7;
    if (n <= 256)  return 8;
    if (n <= 512)  return 9;
    if (n <= 1024) return 10;
    if (n <= 4096) return 12;
    if (n <= 65536) return 14;
    return 16;
}

// -- Extract c-bit digit at position `bit_offset` from scalar -----------------
// Extracts bits [bit_offset, bit_offset+width) from the scalar.
// Returns unsigned digit in [0, 2^width).
static inline uint32_t extract_digit(const Scalar& s, unsigned bit_offset, unsigned width) {
    uint32_t digit = 0;
    for (unsigned b = 0; b < width; ++b) {
        unsigned const pos = bit_offset + b;
        if (pos < 256) {
            digit |= static_cast<uint32_t>(s.bit(pos)) << b;
        }
    }
    return digit;
}

// -- Pippenger Core -----------------------------------------------------------
Point pippenger_msm(const Scalar* scalars,
                    const Point* points,
                    std::size_t n) {
    // Trivial cases
    if (n == 0) return Point::infinity();
    if (n == 1) return points[0].scalar_mul(scalars[0]);

    // For small n, fall back to Strauss (lower constant factor)
    if (n <= 64) {
        return multi_scalar_mul(scalars, points, n);
    }

    unsigned const c = pippenger_optimal_window(n);
    std::size_t const num_buckets = static_cast<std::size_t>(1) << c; // 2^c
    unsigned const num_windows = (256 + c - 1) / c;                   // ceil(256/c)

    // Pre-allocate bucket array (reused per window)
    std::vector<Point> buckets(num_buckets);

    // Result accumulator
    Point result = Point::infinity();

    // Process windows from MSB to LSB
    for (int w = static_cast<int>(num_windows) - 1; w >= 0; --w) {
        unsigned const bit_offset = static_cast<unsigned>(w) * c;

        // If not the first window, shift result left by c bits
        if (w < static_cast<int>(num_windows) - 1) {
            for (unsigned shift = 0; shift < c; ++shift) {
                result.dbl_inplace();
            }
        }

        // Clear buckets
        for (std::size_t b = 0; b < num_buckets; ++b) {
            buckets[b] = Point::infinity();
        }

        // -- Scatter: distribute points into buckets --
        for (std::size_t i = 0; i < n; ++i) {
            uint32_t const digit = extract_digit(scalars[i], bit_offset, c);
            if (digit == 0) continue;  // bucket[0] is unused (identity)
            buckets[digit] = buckets[digit].add(points[i]);
        }

        // -- Aggregate buckets (running-sum trick) --
        // Computes sum_{b=1}^{2^c-1} b * bucket[b] efficiently:
        //   running_sum starts at bucket[2^c-1]
        //   partial_sum accumulates running_sum at each step
        //   This gives: partial_sum = 1*bucket[1] + 2*bucket[2] + ... = Sum b*bucket[b]
        Point running_sum = Point::infinity();
        Point partial_sum = Point::infinity();

        for (std::size_t b = num_buckets - 1; b >= 1; --b) {
            running_sum = running_sum.add(buckets[b]);
            partial_sum = partial_sum.add(running_sum);
        }

        // Combine this window's contribution
        result = result.add(partial_sum);
    }

    return result;
}

// -- Signed-digit Pippenger (halved bucket count) -----------------------------
// Uses signed digits [-2^(c-1), ..., -1, 0, 1, ..., 2^(c-1)]
// This halves the number of buckets (2^(c-1) instead of 2^c) at the cost
// of a carry propagation pass. Very effective for large n.
//
// Not yet enabled by default -- the unsigned version above is simpler and
// already very fast. This is provided for future optimization.

// -- Vector convenience -------------------------------------------------------
Point pippenger_msm(const std::vector<Scalar>& scalars,
                    const std::vector<Point>& points) {
    std::size_t const n = std::min(scalars.size(), points.size());
    if (n == 0) return Point::infinity();
    return pippenger_msm(scalars.data(), points.data(), n);
}

// -- Unified MSM (auto-select) ------------------------------------------------
// Strauss <= 128 points, Pippenger > 128
Point msm(const Scalar* scalars,
          const Point* points,
          std::size_t n) {
    if (n <= 128) {
        return multi_scalar_mul(scalars, points, n);
    }
    return pippenger_msm(scalars, points, n);
}

Point msm(const std::vector<Scalar>& scalars,
          const std::vector<Point>& points) {
    std::size_t const n = std::min(scalars.size(), points.size());
    if (n == 0) return Point::infinity();
    return msm(scalars.data(), points.data(), n);
}

} // namespace secp256k1
