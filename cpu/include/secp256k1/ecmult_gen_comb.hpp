#ifndef SECP256K1_ECMULT_GEN_COMB_HPP
#define SECP256K1_ECMULT_GEN_COMB_HPP
#pragma once

// ============================================================================
// ecmult_gen: Lim-Lee Comb Method for Generator Multiplication
// ============================================================================
//
// Computes k·G using the comb method (Lim & Lee, 1994).
// This is the algorithm used by bitcoin-core/secp256k1's ecmult_gen.
//
// Key advantage over windowed methods:
//   - Table is precomputed ONCE for G (at init time or from cache)
//   - No runtime GLV decomposition (saves ~14μs overhead per mul)
//   - Table fits in L1/L2 cache for reasonable tooth counts
//   - Exactly 256/teeth doublings + 256/teeth additions (fixed cost)
//
// Algorithm:
//   Given teeth t and spacing d = ceil(256/t):
//   1. Pre-compute table[i][j] = (2^(j*d)) · G for i=0..d-1, j=0..t-1
//      Actually stored per-comb: table[comb][entry] where entry is t-bit index
//   2. For each bit position b from d-1 down to 0:
//      a. R = 2·R (double)
//      b. For each comb c, look up bits at positions b, b+d, b+2d, ...
//         forming a t-bit index → add table[c][index] to R
//
// Table size: d * 2^t affine points
//   teeth=15, d=17: 17 * 2^15 = 557K points ≈ 35 MB
//   teeth=11, d=24: 24 * 2^11 = 49K points  ≈ 3 MB (L2-friendly)
//   teeth= 6, d=43: 43 * 2^6  = 2752 points ≈ 176 KB (L1-friendly)
//
// Cost: d doublings + d*combs additions = d*(1 + combs) point ops
//   teeth=15: 17 dbl + 17 add  = 34 ops  (fastest, but big table)
//   teeth=11: 24 dbl + 24 add  = 48 ops  (good balance)
//   teeth= 6: 43 dbl + 43 add  = 86 ops  (tiny table, cache-optimal)
//
// USAGE:
//   secp256k1::fast::CombGenContext ctx;
//   ctx.init();  // or ctx.init(teeth)
//
//   Point R = ctx.mul(scalar);  // fast generator multiplication
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/field.hpp"

namespace secp256k1::fast {

// ── Stored affine point (compact, cache-friendly) ────────────────────────────
struct CombAffinePoint {
    FieldElement x;
    FieldElement y;
    bool infinity = true;
};

// ── Comb Generator Context ───────────────────────────────────────────────────
class CombGenContext {
public:
    CombGenContext() = default;
    ~CombGenContext() = default;

    // Non-copyable (large table)
    CombGenContext(const CombGenContext&) = delete;
    CombGenContext& operator=(const CombGenContext&) = delete;
    CombGenContext(CombGenContext&&) = default;
    CombGenContext& operator=(CombGenContext&&) = default;

    // Initialize with given number of teeth (comb width).
    // teeth=15 → libsecp256k1 default (fastest, 35MB table)
    // teeth=11 → good balance (3MB, fits L2)
    // teeth=6  → compact (176KB, fits L1)
    void init(unsigned teeth = 15);

    // Is context initialized?
    bool ready() const noexcept { return teeth_ > 0; }

    // Generator multiplication: R = k·G
    // Fixed-cost: spacing_ doublings + spacing_ additions.
    Point mul(const Scalar& k) const;

    // Constant-time generator multiplication (scans all table entries)
    Point mul_ct(const Scalar& k) const;

    // Table info
    unsigned teeth() const noexcept { return teeth_; }
    unsigned spacing() const noexcept { return spacing_; }
    std::size_t table_size_bytes() const noexcept;

    // Cache: save/load precomputed table
    bool save_cache(const std::string& path) const;
    bool load_cache(const std::string& path);

private:
    unsigned teeth_ = 0;     // Number of "teeth" (comb width in bits)
    unsigned spacing_ = 0;   // = ceil(256 / teeth_)
    unsigned num_combs_ = 1; // Number of comb tables (always 1 for standard)

    // Table layout: table_[comb_idx * (1 << teeth_) + entry]
    // where entry is a teeth_-bit index formed by gathering bits at
    // positions b, b+spacing, b+2*spacing, ..., b+(teeth-1)*spacing
    std::vector<CombAffinePoint> table_;

    // Build the comb table from generator G
    void build_table();

    // Extract teeth-bit comb index for bit position b
    uint32_t extract_comb_index(const Scalar& k, unsigned b) const;
};

// ── Global comb context (singleton, like libsecp256k1's secp256k1_ecmult_gen_context) ──
void init_comb_gen(unsigned teeth = 15);
bool comb_gen_ready();
Point comb_gen_mul(const Scalar& k);
Point comb_gen_mul_ct(const Scalar& k);

} // namespace secp256k1::fast

#endif // SECP256K1_ECMULT_GEN_COMB_HPP
