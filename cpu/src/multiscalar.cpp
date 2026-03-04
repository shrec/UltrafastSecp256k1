// ============================================================================
// Multi-Scalar Multiplication: GLV Strauss with Effective-Affine
// ============================================================================
// GLV decomposition halves the scan length (~130 vs ~257 positions) by
// splitting each 256-bit scalar into two ~128-bit halves.  Combined with
// effective-affine table construction (7M+4S per add vs 12M+5S) and
// batch-inverted affine tables for mixed additions in the scan loop.
//
// Effective-affine: Precomp tables are batch-converted to affine using
// Montgomery's trick (1 field inverse + O(n) muls).  The scan loop then
// uses mixed additions (7M+4S, ~170ns) instead of full Jacobian additions
// (12M+5S, ~275ns), a ~38% reduction per addition.

#include "secp256k1/multiscalar.hpp"
#include "secp256k1/glv.hpp"
#include "secp256k1/precompute.hpp"
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

// -- Strauss Multi-Scalar Multiplication (GLV + Effective-Affine) -------------
// GLV decomposes each 256-bit scalar into two ~128-bit halves, halving the
// doubling chain from ~257 to ~130 positions.  Per-point tables are built
// using effective-affine (iso curve + batch inversion) and phi(P) tables
// are derived via beta multiplication (no extra precompute).

Point multi_scalar_mul(const Scalar* scalars,
                       const Point* points,
                       std::size_t n) {
    if (n == 0) return Point::infinity();
    if (n == 1) return points[0].scalar_mul(scalars[0]);
    if (n == 2) return shamir_trick(scalars[0], points[0], scalars[1], points[1]);

    unsigned const w = strauss_optimal_window(n);
    std::size_t const table_size = static_cast<std::size_t>(1) << (w - 1);

#if defined(SECP256K1_FAST_52BIT)
    using FE52 = fast::FieldElement52;

    // -- Step 1: GLV decompose all scalars --------------------------------
    // Each scalar_i -> (k1_i, k2_i, neg1_i, neg2_i)
    // scalar_i * P_i = k1_i * Q_i + k2_i * phi(Q_i)
    //   where Q_i = neg1_i ? -P_i : P_i
    //   and phi(Q_i).y is negated if neg1_i XOR neg2_i

    struct GLVInfo {
        bool neg1, neg2;
    };
    std::vector<GLVInfo> glv_info(n);

    // wNAF for k1 streams at [0..n-1], k2 streams at [n..2n-1]
    std::vector<std::vector<int32_t>> wnaf_bufs(2 * n);
    std::vector<std::size_t> wnaf_lens(2 * n, 0);
    std::size_t max_len = 0;

    for (std::size_t i = 0; i < n; ++i) {
        auto decomp = fast::glv_decompose(scalars[i]);
        glv_info[i] = {decomp.k1_neg, decomp.k2_neg};

        // Allocate and compute wNAF for both half-scalars
        wnaf_bufs[i].resize(260, 0);
        wnaf_bufs[n + i].resize(260, 0);
        compute_wnaf_into(decomp.k1, w,
                          wnaf_bufs[i].data(), 260, wnaf_lens[i]);
        compute_wnaf_into(decomp.k2, w,
                          wnaf_bufs[n + i].data(), 260, wnaf_lens[n + i]);

        // Trim trailing zeros -- half-scalars are ~128 bits
        while (wnaf_lens[i] > 0 && wnaf_bufs[i][wnaf_lens[i] - 1] == 0)
            --wnaf_lens[i];
        while (wnaf_lens[n + i] > 0 && wnaf_bufs[n + i][wnaf_lens[n + i] - 1] == 0)
            --wnaf_lens[n + i];

        max_len = std::max(max_len,
                           std::max(wnaf_lens[i], wnaf_lens[n + i]));
    }

    // -- Step 2: Build precomp tables per point + batch invert to affine ----
    // Table[i][j] = (2j+1) * Q_i where Q_i = neg1_i ? -P_i : P_i
    // After batch inversion, tables are stored as affine FE52 (x, y).

    std::size_t const total_entries = n * table_size;
    std::vector<FE52> tbl_P_x(total_entries), tbl_P_y(total_entries);
    std::vector<FE52> tbl_phiP_x(total_entries), tbl_phiP_y(total_entries);

    {
        // Build tables using Point-level operations (handles all edge cases)
        std::vector<Point> base_pts(n);
        for (std::size_t i = 0; i < n; ++i) {
            base_pts[i] = glv_info[i].neg1 ? points[i].negate() : points[i];
        }

        // Build odd-multiple tables: [1Q, 3Q, 5Q, ..., (2T-1)Q]
        std::vector<std::vector<Point>> tables(n);
        for (std::size_t i = 0; i < n; ++i) {
            tables[i].resize(table_size);
            tables[i][0] = base_pts[i];
            if (table_size > 1) {
                Point const P2 = base_pts[i].dbl();
                for (std::size_t j = 1; j < table_size; ++j) {
                    tables[i][j] = tables[i][j - 1].add(P2);
                }
            }
        }

        // Batch-invert all Z values via Montgomery's trick
        std::vector<FE52> z_vals(total_entries);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < table_size; ++j) {
                z_vals[i * table_size + j] = tables[i][j].Z52();
            }
        }

        std::vector<FE52> prefix(total_entries);
        prefix[0] = z_vals[0];
        for (std::size_t k = 1; k < total_entries; ++k) {
            prefix[k] = prefix[k - 1] * z_vals[k];
        }

        // Guard: degenerate case
        if (prefix[total_entries - 1].normalizes_to_zero()) {
            Point result = Point::infinity();
            for (std::size_t i = 0; i < n; ++i) {
                result.add_inplace(points[i].scalar_mul(scalars[i]));
            }
            return result;
        }

        FE52 inv = prefix[total_entries - 1].inverse();
        for (std::size_t k = total_entries; k-- > 0; ) {
            FE52 const z_inv = (k > 0) ? prefix[k - 1] * inv : inv;
            if (k > 0) inv *= z_vals[k];

            FE52 const z2 = z_inv.square();
            FE52 const z3 = z2 * z_inv;

            std::size_t const pi = k / table_size;
            std::size_t const pj = k % table_size;
            tbl_P_x[k] = tables[pi][pj].X52() * z2;
            tbl_P_y[k] = tables[pi][pj].Y52() * z3;
        }
    }

    // -- Step 3: Derive phi(P) tables via beta multiplication -------------
    static const FE52 beta52 = FE52::from_fe(
        fast::FieldElement::from_bytes(fast::glv_constants::BETA));

    for (std::size_t i = 0; i < n; ++i) {
        bool const flip_phi = (glv_info[i].neg1 != glv_info[i].neg2);
        std::size_t const base = i * table_size;
        for (std::size_t j = 0; j < table_size; ++j) {
            tbl_phiP_x[base + j] = tbl_P_x[base + j] * beta52;
            if (flip_phi) {
                tbl_phiP_y[base + j] = tbl_P_y[base + j].negate(1);
                tbl_phiP_y[base + j].normalize_weak();
            } else {
                tbl_phiP_y[base + j] = tbl_P_y[base + j];
            }
        }
    }

    // -- Step 4: Scan ~130 positions with 2n streams ----------------------
    Point R = Point::infinity();

    for (std::size_t bit = max_len; bit-- > 0; ) {
        R.dbl_inplace();

        for (std::size_t i = 0; i < n; ++i) {
            std::size_t const base = i * table_size;

            // k1 stream: lookup from tbl_P
            if (bit < wnaf_lens[i]) {
                int32_t const digit = wnaf_bufs[i][bit];
                if (digit != 0) {
                    std::size_t const idx = static_cast<std::size_t>(
                        (digit > 0 ? digit - 1 : -digit - 1) / 2
                    );
                    FE52 lx = tbl_P_x[base + idx];
                    FE52 ly = tbl_P_y[base + idx];
                    if (digit < 0) {
                        ly.negate_assign(1);
                    }
                    R.add_mixed52_inplace(lx, ly);
                }
            }

            // k2 stream: lookup from tbl_phiP
            if (bit < wnaf_lens[n + i]) {
                int32_t const digit = wnaf_bufs[n + i][bit];
                if (digit != 0) {
                    std::size_t const idx = static_cast<std::size_t>(
                        (digit > 0 ? digit - 1 : -digit - 1) / 2
                    );
                    FE52 lx = tbl_phiP_x[base + idx];
                    FE52 ly = tbl_phiP_y[base + idx];
                    if (digit < 0) {
                        ly.negate_assign(1);
                    }
                    R.add_mixed52_inplace(lx, ly);
                }
            }
        }
    }

    return R;

#else
    // Non-FE52 fallback: original Strauss without GLV

    // Compute wNAF for each scalar
    std::vector<std::vector<int8_t>> wnafs(n);
    std::size_t max_len = 0;
    for (std::size_t i = 0; i < n; ++i) {
        wnafs[i] = scalars[i].to_wnaf(w);
        if (wnafs[i].size() > max_len) {
            max_len = wnafs[i].size();
        }
    }

    // Pre-compute odd multiples: table[i][j] = (2j+1) * points[i]
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
