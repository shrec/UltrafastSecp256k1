#ifndef SECP256K1_CT_POINT_HPP
#define SECP256K1_CT_POINT_HPP

// ============================================================================
// Constant-Time Point Arithmetic
// ============================================================================
// Side-channel resistant elliptic curve point operations for secp256k1.
// Uses secp256k1::fast::Point and FieldElement data types.
//
// Key differences from fast::Point:
//   1. COMPLETE addition formula — handles all cases (P+Q, P+P, P+O, O+Q,
//      P+(-P)) in a single codepath with NO branches on point values
//   2. CT scalar multiplication — fixed execution pattern:
//      - Always same number of doublings + additions
//      - CT table lookup (scans all entries)
//      - No early exit or conditional skip
//   3. CT table lookup for precomputed multiples
//
// The complete Jacobian addition for a=0 (secp256k1: y²=x³+7) uses
// the Renes-Costello-Bathalter (2016) formula adapted for a=0.
// Cost: 12M + 2S (vs 11M + 5S for fast incomplete)
// Advantage: No branch on P==Q, P==-Q, P==O
// ============================================================================

#include <cstdint>
#include <cstddef>
#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ct/ops.hpp"
#include "secp256k1/ct/field.hpp"
#include "secp256k1/ct/scalar.hpp"

namespace secp256k1::ct {

using Point = secp256k1::fast::Point;
using FieldElement = secp256k1::fast::FieldElement;
using Scalar = secp256k1::fast::Scalar;

// ─── CT Jacobian Point (internal representation) ─────────────────────────────
// Uses uint64_t flag instead of bool for branchless operations
struct CTJacobianPoint {
    FieldElement x;
    FieldElement y;
    FieldElement z;
    std::uint64_t infinity;  // 0 = normal, 0xFFFF...F = infinity

    static CTJacobianPoint from_point(const Point& p) noexcept;
    Point to_point() const noexcept;
    static CTJacobianPoint make_infinity() noexcept;
};

// ─── CT Affine Point (for precomputed tables) ────────────────────────────────
// Compact representation with Z=1 implied. Used in precomputed tables
// where mixed Jacobian+Affine addition saves ~4 field multiplications.
struct CTAffinePoint {
    FieldElement x;
    FieldElement y;
    std::uint64_t infinity;  // 0 = normal, 0xFFFF...F = infinity

    static CTAffinePoint make_infinity() noexcept {
        CTAffinePoint r;
        r.x = FieldElement::zero();
        r.y = FieldElement::zero();
        r.infinity = ~static_cast<std::uint64_t>(0);
        return r;
    }

    static CTAffinePoint from_point(const Point& p) noexcept {
        CTAffinePoint r;
        if (p.is_infinity()) {
            r = make_infinity();
        } else {
            r.x = p.x();
            r.y = p.y();
            r.infinity = 0;
        }
        return r;
    }
};

// ─── Complete Addition ───────────────────────────────────────────────────────
// Handles ALL cases in a single branchless codepath:
//   P + Q,  P + P (doubling),  P + O,  O + Q,  P + (-P) = O
// No secret-dependent branches. Fixed 13M + 9S cost.

CTJacobianPoint point_add_complete(const CTJacobianPoint& p,
                                   const CTJacobianPoint& q) noexcept;

// ─── Mixed Jacobian+Affine Complete Addition ─────────────────────────────────
// Same complete formula but optimized for Q in affine (Z2=1).
// Saves ~4 field multiplications vs general Jacobian+Jacobian.
// Cost: 9M + 8S (fixed, no branches)
CTJacobianPoint point_add_mixed_complete(const CTJacobianPoint& p,
                                         const CTAffinePoint& q) noexcept;

// ─── CT Point Doubling ───────────────────────────────────────────────────────
// Branchless doubling (handles identity via cmov, no branch)
CTJacobianPoint point_dbl(const CTJacobianPoint& p) noexcept;

// ─── CT Point Negation ───────────────────────────────────────────────────────
CTJacobianPoint point_neg(const CTJacobianPoint& p) noexcept;

// ─── CT Conditional Operations ───────────────────────────────────────────────
void point_cmov(CTJacobianPoint* r, const CTJacobianPoint& a,
                std::uint64_t mask) noexcept;

CTJacobianPoint point_select(const CTJacobianPoint& a, const CTJacobianPoint& b,
                             std::uint64_t mask) noexcept;

// ─── CT Table Lookup ─────────────────────────────────────────────────────────
// Always reads ALL table entries. Returns table[index].
// table_size must equal the actual table array length.
CTJacobianPoint point_table_lookup(const CTJacobianPoint* table,
                                   std::size_t table_size,
                                   std::size_t index) noexcept;

// CT affine table lookup — scans all entries, returns table[index].
CTAffinePoint affine_table_lookup(const CTAffinePoint* table,
                                  std::size_t table_size,
                                  std::size_t index) noexcept;

// CT conditional operations on affine points
void affine_cmov(CTAffinePoint* r, const CTAffinePoint& a,
                 std::uint64_t mask) noexcept;

// ─── CT Scalar Multiplication ────────────────────────────────────────────────
// The core CT operation. Fixed execution trace regardless of scalar value.
//
// Method: Fixed-window (w=4) with CT table lookup
//   - 256/4 = 64 doublings + 64 additions (always, no skip)
//   - Each addition uses CT table lookup (scans all 16 entries)
//   - Complete addition formula (no special-case branches)
//   - Handles all edge cases: k=0, k=1, k=n-1, P=O
//
// Cost: ~64 * (4 dbl + 1 complete_add + 1 CT_lookup)
// Slower than fast:: (~2-3x) but constant-time.

Point scalar_mul(const Point& p, const Scalar& k) noexcept;

// CT generator multiplication: k * G
// Optimized with precomputed table (64 × 16 affine points).
// Uses mixed Jacobian+Affine addition — NO doublings needed at runtime.
// Cost: 64 mixed_complete_add + 64 CT_lookup(16)
// Approximately 3x faster than generic scalar_mul(G, k).
Point generator_mul(const Scalar& k) noexcept;

// Initialize the precomputed generator table.
// Called automatically on first generator_mul() call.
// Can be called explicitly at startup to avoid first-call latency.
void init_generator_table() noexcept;

// ─── CT GLV Endomorphism ─────────────────────────────────────────────────────
// Apply secp256k1 endomorphism: φ(P) = (β·X, Y, Z) where β³ ≡ 1 (mod p).
// Constant-time (just one field multiplication, no branches).
CTJacobianPoint point_endomorphism(const CTJacobianPoint& p) noexcept;

// CT affine endomorphism
CTAffinePoint affine_endomorphism(const CTAffinePoint& p) noexcept;

// CT affine negation
CTAffinePoint affine_neg(const CTAffinePoint& p) noexcept;

// ─── CT GLV Decomposition ────────────────────────────────────────────────────
// Split scalar k = k1 + k2·λ (mod n) where |k1|, |k2| ≈ 128 bits.
// Fully constant-time: no branches on scalar values.
struct CTGLVDecomposition {
    Scalar k1;           // |k1| ≈ 128 bits (always positive after abs)
    Scalar k2;           // |k2| ≈ 128 bits (always positive after abs)
    std::uint64_t k1_neg;  // all-ones if k1 was negated, 0 otherwise
    std::uint64_t k2_neg;  // all-ones if k2 was negated, 0 otherwise
};

CTGLVDecomposition ct_glv_decompose(const Scalar& k) noexcept;

// ─── CT Verify (for ECDSA) ──────────────────────────────────────────────────
// Returns all-ones mask if point is on curve, else 0. CT.
std::uint64_t point_is_on_curve(const Point& p) noexcept;

// CT equality check
std::uint64_t point_eq(const Point& a, const Point& b) noexcept;

} // namespace secp256k1::ct

#endif // SECP256K1_CT_POINT_HPP
