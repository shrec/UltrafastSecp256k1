#ifndef SECP256K1_MULTISCALAR_HPP
#define SECP256K1_MULTISCALAR_HPP
#pragma once

// ============================================================================
// Multi-Scalar Multiplication for secp256k1
// ============================================================================
// Computes: R = s1*P1 + s2*P2 + ... + sN*PN
//
// Algorithms:
//   - Strauss (interleaved wNAF): optimal for N <= ~128
//   - Naive (sequential): fallback / reference
//
// Also provides Shamir's trick (2-point special case, used by ECDSA verify).
// ============================================================================

#include <cstddef>
#include <cstdint>
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"

namespace secp256k1 {

// -- Shamir's Trick (2-point) -------------------------------------------------
// Computes: R = a*P + b*Q  in a single pass (joint double-and-add).
// ~1.5x faster than two separate scalar_mul + add.
fast::Point shamir_trick(const fast::Scalar& a, const fast::Point& P,
                         const fast::Scalar& b, const fast::Point& Q);

// -- Multi-Scalar Multiplication (Strauss) ------------------------------------
// Computes: R = sum( scalars[i] * points[i] ) for i in [0, n).
// Uses interleaved wNAF (Strauss' algorithm) for efficiency.
//
// Parameters:
//   scalars  - array of n scalars
//   points   - array of n points (affine or Jacobian)
//   n        - number of scalar-point pairs
//
// Returns the resulting point (may be infinity if the sum cancels).
//
// Performance: O(256 + n * 2^(w-1)) instead of O(256 * n) for naive.
fast::Point multi_scalar_mul(const fast::Scalar* scalars,
                             const fast::Point* points,
                             std::size_t n);

// Convenience: vector version
fast::Point multi_scalar_mul(const std::vector<fast::Scalar>& scalars,
                             const std::vector<fast::Point>& points);

// -- Strauss Window Width Selection -------------------------------------------
// Returns optimal wNAF window width for n points.
// w=4 for n<32, w=5 for n<128, w=6 for n>=128.
unsigned strauss_optimal_window(std::size_t n);

} // namespace secp256k1

#endif // SECP256K1_MULTISCALAR_HPP
