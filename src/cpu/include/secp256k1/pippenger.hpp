#ifndef SECP256K1_PIPPENGER_HPP
#define SECP256K1_PIPPENGER_HPP
#pragma once

// ============================================================================
// Pippenger Bucket Method for Multi-Scalar Multiplication
// ============================================================================
//
// Computes: R = s_1*P_1 + s_2*P_2 + ... + s_n*P_n
//
// Algorithm (bucket method, a.k.a. Pippenger):
//   1. Choose window width c from an empirically tuned CPU heuristic.
//   2. Represent each scalar s in base-2^c digits.
//   3. For each digit position j (from MSB to LSB):
//      a. Scatter: place P into bucket[digit_j(s)] for all i.
//      b. Aggregate: sum buckets as Sum = Sum_{b=1}^{2^c-1} b * bucket[b].
//         Computed bottom-up: running_sum += bucket[b], partial_sum += running_sum.
//      c. Combine: R = R*2^c + Sum
//
// Complexity: O(n/c + 2^c + 256*dbl) vs Strauss O(256 + n*2^(w-1))
// Current CPU crossover: Pippenger wins around n ~= 48.
//
// This implementation:
//   - Pre-allocates all buckets in a single flat array (no heap per iteration)
//   - Uses predecoded digits and bucket reuse on the optimized CPU path
//   - Falls back to Strauss for small n
//
// Reference: Bernstein, Doumen, Lange, Oosterwijk (2012),
//            "Faster batch forgery identification"
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <vector>
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"

namespace secp256k1 {

// -- Pippenger Multi-Scalar Multiplication ------------------------------------
// Computes: R = sum( scalars[i] * points[i] ) for i in [0, n).
// Uses bucket method (Pippenger) which is asymptotically optimal.
//
// Parameters:
//   scalars  - array of n scalars
//   points   - array of n points
//   n        - number of scalar-point pairs
//
// Performance: O(n/c + 2^c) per window, with c chosen from measured bands.
//   n=256:   ~4x faster than Strauss
//   n=1024:  ~8x faster than Strauss
//   n=4096:  ~12x faster than Strauss

fast::Point pippenger_msm(const fast::Scalar* scalars,
                          const fast::Point* points,
                          std::size_t n);

// Convenience: vector version
fast::Point pippenger_msm(const std::vector<fast::Scalar>& scalars,
                          const std::vector<fast::Point>& points);

// -- Optimal Window Width -----------------------------------------------------
// Returns the optimal bucket window width c for n points.
// Uses measured CPU bands, not just the textbook floor(log2(n)) heuristic.
unsigned pippenger_optimal_window(std::size_t n);

// -- Unified MSM (auto-selects best algorithm) --------------------------------
// Automatically picks Strauss for very small MSMs and Pippenger from n >= 48.
fast::Point msm(const fast::Scalar* scalars,
                const fast::Point* points,
                std::size_t n);

fast::Point msm(const std::vector<fast::Scalar>& scalars,
                const std::vector<fast::Point>& points);

} // namespace secp256k1

#endif // SECP256K1_PIPPENGER_HPP
