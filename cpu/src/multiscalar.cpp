// ============================================================================
// Multi-Scalar Multiplication: Strauss / Shamir's trick
// ============================================================================

#include "secp256k1/multiscalar.hpp"
#include <algorithm>
#include <cstring>

namespace secp256k1 {

using fast::Scalar;
using fast::Point;

// -- Window Width Selection ---------------------------------------------------

unsigned strauss_optimal_window(std::size_t n) {
    if (n <= 1)   return 4;
    if (n <= 8)   return 4;
    if (n <= 32)  return 4;
    if (n <= 128) return 5;
    return 6;
}

// -- Shamir's Trick (2-point) -------------------------------------------------
// R = a*P + b*Q
// When one point is the generator, its scalar_mul automatically uses the
// precomputed fixed-base comb method (~7us), which is faster than wNAF.

Point shamir_trick(const Scalar& a, const Point& P,
                   const Scalar& b, const Point& Q) {
    if (a.is_zero() && b.is_zero()) return Point::infinity();
    if (a.is_zero()) return Q.scalar_mul(b);
    if (b.is_zero()) return P.scalar_mul(a);

    // Each scalar_mul checks is_generator_ internally:
    //   - Generator: uses precomputed comb tables (~7us)
    //   - Generic:   uses GLV + 5x52 Shamir (~25us)
    // Total: ~32us for a*G + b*Q (faster than 4-stream wNAF because
    // the precomputed generator tables are wider/deeper than wNAF w=5).
    auto aP = P.scalar_mul(a);
    aP.add_inplace(Q.scalar_mul(b));
    return aP;
}

// -- Strauss Multi-Scalar Multiplication --------------------------------------
// Interleaved wNAF: pre-compute odd multiples of each point, then scan
// all wNAFs simultaneously from MSB to LSB.

Point multi_scalar_mul(const Scalar* scalars,
                       const Point* points,
                       std::size_t n) {
    if (n == 0) return Point::infinity();
    if (n == 1) return points[0].scalar_mul(scalars[0]);
    if (n == 2) return shamir_trick(scalars[0], points[0], scalars[1], points[1]);

    unsigned const w = strauss_optimal_window(n);
    std::size_t const table_size = static_cast<std::size_t>(1) << (w - 1); // 2^(w-1) entries per point

    // Step 1: Compute wNAF for each scalar
    std::vector<std::vector<int8_t>> wnafs(n);
    std::size_t max_len = 0;
    for (std::size_t i = 0; i < n; ++i) {
        wnafs[i] = scalars[i].to_wnaf(w);
        if (wnafs[i].size() > max_len) {
            max_len = wnafs[i].size();
        }
    }

    // Step 2: Pre-compute odd multiples: table[i][j] = (2j+1) * points[i]
    // table[i][0] = P, table[i][1] = 3P, table[i][2] = 5P, ...
    std::vector<std::vector<Point>> tables(n);
    for (std::size_t i = 0; i < n; ++i) {
        tables[i].resize(table_size);
        tables[i][0] = points[i];
        if (table_size > 1) {
            Point const P2 = points[i].dbl();
            for (std::size_t j = 1; j < table_size; ++j) {
                tables[i][j] = tables[i][j - 1].add(P2);
            }
        }
    }

    // Step 3: Interleaved scan from MSB to LSB
    Point R = Point::infinity();

    for (std::size_t bit = max_len; bit-- > 0; ) {
        R.dbl_inplace();

        for (std::size_t i = 0; i < n; ++i) {
            if (bit >= wnafs[i].size()) continue;
            int8_t const digit = wnafs[i][bit];
            if (digit == 0) continue;

            std::size_t idx = 0;
            if (digit > 0) {
                idx = static_cast<std::size_t>((digit - 1) / 2);
                R.add_inplace(tables[i][idx]);
            } else {
                idx = static_cast<std::size_t>((-digit - 1) / 2);
                Point neg_pt = tables[i][idx];
                neg_pt.negate_inplace();
                R.add_inplace(neg_pt);
            }
        }
    }

    return R;
}

// Convenience: vector version
Point multi_scalar_mul(const std::vector<Scalar>& scalars,
                       const std::vector<Point>& points) {
    std::size_t const n = std::min(scalars.size(), points.size());
    if (n == 0) return Point::infinity();
    return multi_scalar_mul(scalars.data(), points.data(), n);
}

} // namespace secp256k1
