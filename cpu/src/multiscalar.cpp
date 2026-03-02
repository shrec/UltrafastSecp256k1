// ============================================================================
// Multi-Scalar Multiplication: Strauss / Shamir's trick
// ============================================================================
// GLV note: GLV-decomposition was evaluated for Strauss MSM but found
// counterproductive: it doubles point count (2N) while halving scan length
// (~130 vs ~256).  The extra precompute+per-step cost outweighs the saved
// doublings for N >= 4.  Individual scalar_mul already uses GLV internally.
//
// Effective-affine: Precomp tables are batch-converted to affine using
// Montgomery's trick (1 field inverse + O(n) muls).  The scan loop then
// uses mixed additions (7M+4S, ~170ns) instead of full Jacobian additions
// (12M+5S, ~275ns), a ~38% reduction per addition.

#include "secp256k1/multiscalar.hpp"
#include <algorithm>
#include <cstring>

#if defined(SECP256K1_FAST_52BIT)
#include "secp256k1/field_52.hpp"
#endif

namespace secp256k1 {

using fast::Scalar;
using fast::Point;

// -- Window Width Selection ---------------------------------------------------
// With effective-affine (batch precomp -> affine, mixed additions in scan),
// the cost per scan-add drops from ~275ns (Jacobian) to ~170ns (mixed).
// This shifts the precomp-vs-scan trade-off: w=4 (8 entries/point) is
// optimal for all practical point counts up to the Strauss/Pippenger
// crossover (~128).  The per-point cost at each window size:
//   w=3: precomp=1125 + affine=312 + scan=10880 = 12317
//   w=4: precomp=2625 + affine=624 + scan= 8704 = 11953  <-- optimal
//   w=5: precomp=5625 + affine=1248 + scan= 7253 = 14126

unsigned strauss_optimal_window(std::size_t n) {
    (void)n;
    return 4;
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

// -- Strauss Multi-Scalar Multiplication (Effective-Affine) -------------------
// Interleaved wNAF: pre-compute odd multiples of each point, batch convert
// to affine via Montgomery's trick, then scan all wNAFs simultaneously from
// MSB to LSB using mixed additions.

Point multi_scalar_mul(const Scalar* scalars,
                       const Point* points,
                       std::size_t n) {
    if (n == 0) return Point::infinity();
    if (n == 1) return points[0].scalar_mul(scalars[0]);
    if (n == 2) return shamir_trick(scalars[0], points[0], scalars[1], points[1]);

    unsigned const w = strauss_optimal_window(n);
    std::size_t const table_size = static_cast<std::size_t>(1) << (w - 1);

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

#if defined(SECP256K1_FAST_52BIT)
    // Step 3: Batch convert ALL precomp entries to affine via Montgomery's trick.
    // Enables mixed additions (7M+4S) in the scan loop instead of
    // full Jacobian additions (12M+5S), saving ~38% per addition.
    using FE52 = fast::FieldElement52;

    std::size_t const total_entries = n * table_size;
    std::vector<FE52> aff_x(total_entries);
    std::vector<FE52> aff_y(total_entries);

    {
        // Collect Z coords from all precomp entries
        std::vector<FE52> z_vals(total_entries);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < table_size; ++j) {
                z_vals[i * table_size + j] = tables[i][j].Z52();
            }
        }

        // Montgomery batch inversion: prefix[k] = z[0] * z[1] * ... * z[k]
        std::vector<FE52> prefix(total_entries);
        prefix[0] = z_vals[0];
        for (std::size_t k = 1; k < total_entries; ++k) {
            prefix[k] = prefix[k - 1] * z_vals[k];
        }

        // Single field inversion (~940ns vs ~275ns * total_entries saves)
        FE52 inv = prefix[total_entries - 1].inverse();

        // Back-propagate inversions and compute affine coordinates
        for (std::size_t k = total_entries; k-- > 0; ) {
            FE52 const z_inv = (k > 0) ? prefix[k - 1] * inv : inv;
            if (k > 0) inv *= z_vals[k];

            // x_aff = X * z_inv^2,  y_aff = Y * z_inv^3
            FE52 const z2 = z_inv.square();
            FE52 const z3 = z2 * z_inv;

            std::size_t const pi = k / table_size;
            std::size_t const pj = k % table_size;
            aff_x[k] = tables[pi][pj].X52() * z2;
            aff_y[k] = tables[pi][pj].Y52() * z3;
        }
    }

    // Step 4: Interleaved scan with mixed additions (effective-affine)
    Point R = Point::infinity();

    for (std::size_t bit = max_len; bit-- > 0; ) {
        R.dbl_inplace();

        for (std::size_t i = 0; i < n; ++i) {
            if (bit >= wnafs[i].size()) continue;
            int8_t const digit = wnafs[i][bit];
            if (digit == 0) continue;

            std::size_t const idx = static_cast<std::size_t>(
                (digit > 0 ? digit - 1 : -digit - 1) / 2
            );
            std::size_t const flat = i * table_size + idx;

            if (digit > 0) {
                R.add_mixed52_inplace(aff_x[flat], aff_y[flat]);
            } else {
                // Negate Y for subtraction: -(x, y) = (x, -y)
                FE52 neg_y = aff_y[flat];
                neg_y.negate_assign(1);
                R.add_mixed52_inplace(aff_x[flat], neg_y);
            }
        }
    }

    return R;

#else
    // Fallback: original Jacobian-add scan (no FE52 batch inversion available)
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
#endif
}

// Convenience: vector version
Point multi_scalar_mul(const std::vector<Scalar>& scalars,
                       const std::vector<Point>& points) {
    std::size_t const n = std::min(scalars.size(), points.size());
    if (n == 0) return Point::infinity();
    return multi_scalar_mul(scalars.data(), points.data(), n);
}

} // namespace secp256k1
