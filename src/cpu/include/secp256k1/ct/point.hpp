#ifndef SECP256K1_CT_POINT_HPP
#define SECP256K1_CT_POINT_HPP

// ============================================================================
// Constant-Time Point Arithmetic
// ============================================================================
// Side-channel resistant elliptic curve point operations for secp256k1.
// Uses secp256k1::fast::Point and FieldElement data types.
//
// Key differences from fast::Point:
//   1. COMPLETE addition formula -- handles all cases (P+Q, P+P, P+O, O+Q,
//      P+(-P)) in a single codepath with NO branches on point values
//   2. CT scalar multiplication -- fixed execution pattern:
//      - Always same number of doublings + additions
//      - CT table lookup (scans all entries)
//      - No early exit or conditional skip
//   3. CT table lookup for precomputed multiples
//
// The complete Jacobian addition for a=0 (secp256k1: y^2=x^3+7) uses
// the Renes-Costello-Bathalter (2016) formula adapted for a=0.
// Cost: 12M + 2S (vs 11M + 5S for fast incomplete)
// Advantage: No branch on P==Q, P==-Q, P==O
// ============================================================================

#include <cstdint>
#include <cstddef>
#include "secp256k1/field.hpp"
#include "secp256k1/field_52.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ct/ops.hpp"
#include "secp256k1/ct/field.hpp"
#include "secp256k1/ct/scalar.hpp"

namespace secp256k1::ct {

using Point = secp256k1::fast::Point;
using FieldElement = secp256k1::fast::FieldElement;
using Scalar = secp256k1::fast::Scalar;

// FE type for CT internals: FieldElement52 on 5x52 platforms, FieldElement otherwise.
#if defined(SECP256K1_FAST_52BIT)
using FE52 = secp256k1::fast::FieldElement52;
#else
using FE52 = secp256k1::fast::FieldElement;
#endif

// --- CT Jacobian Point (internal representation) -----------------------------
// Uses uint64_t flag instead of bool for branchless operations
struct CTJacobianPoint {
    FE52 x;
    FE52 y;
    FE52 z;
    std::uint64_t infinity;  // 0 = normal, 0xFFFF...F = infinity

    static CTJacobianPoint from_point(const Point& p) noexcept;
    Point to_point() const noexcept;
    static CTJacobianPoint make_infinity() noexcept;
};

// --- CT Affine Point (for precomputed tables) --------------------------------
// Compact representation with Z=1 implied. Used in precomputed tables
// where mixed Jacobian+Affine addition saves ~4 field multiplications.
struct CTAffinePoint {
    FE52 x;
    FE52 y;
    std::uint64_t infinity;  // 0 = normal, 0xFFFF...F = infinity

    static CTAffinePoint make_infinity() noexcept {
        CTAffinePoint r;
#if defined(SECP256K1_FAST_52BIT)
        r.x = FE52::zero();
        r.y = FE52::zero();
#else
        r.x = FieldElement();
        r.y = FieldElement();
#endif
        r.infinity = ~static_cast<std::uint64_t>(0);
        return r;
    }

    static CTAffinePoint from_point(const Point& p) noexcept {
        CTAffinePoint r;
        if (p.is_infinity()) {
            r = make_infinity();
        } else {
#if defined(SECP256K1_FAST_52BIT)
            r.x = FE52::from_fe(p.x());
            r.y = FE52::from_fe(p.y());
#else
            r.x = p.x();
            r.y = p.y();
#endif
            r.infinity = 0;
        }
        return r;
    }
};

// --- Complete Addition -------------------------------------------------------
// Brier-Joye unified addition/doubling for general Jacobian+Jacobian.
// Handles ALL cases in a single branchless codepath:
//   P + Q,  P + P (doubling),  P + O,  O + Q,  P + (-P) = O
// No secret-dependent branches. Cost: 11M + 6S.

CTJacobianPoint point_add_complete(const CTJacobianPoint& p,
                                   const CTJacobianPoint& q) noexcept;

// --- Mixed Jacobian+Affine Complete Addition ---------------------------------
// Brier-Joye unified addition/doubling for Jacobian+Affine (Z2=1).
// Cost: 7M + 5S (fixed, no branches)
CTJacobianPoint point_add_mixed_complete(const CTJacobianPoint& p,
                                         const CTAffinePoint& q) noexcept;

// --- CT Point Doubling -------------------------------------------------------
// Libsecp-style 3M+4S+1half doubling (low magnitudes, handles identity via cmov)
CTJacobianPoint point_dbl(const CTJacobianPoint& p) noexcept;

// --- Brier-Joye Unified Mixed Addition (Jacobian + Affine, a=0) -------------
// Cost: 7M + 5S (vs 9M+8S for complete formula)
// Unified: handles both addition and doubling in one codepath.
// Precondition: b MUST NOT be infinity. a may be infinity.
// When b might be infinity, use cmov after call to skip the result.
CTJacobianPoint point_add_mixed_unified(const CTJacobianPoint& a,
                                         const CTAffinePoint& b) noexcept;

// In-place variant: writes result directly into *out (avoids 128-byte copy).
// out may alias &a (reads a first, then writes result).
void point_add_mixed_unified_into(CTJacobianPoint* out,
                                   const CTJacobianPoint& a,
                                   const CTAffinePoint& b) noexcept;

// --- CT Point Negation -------------------------------------------------------
CTJacobianPoint point_neg(const CTJacobianPoint& p) noexcept;

// --- CT Conditional Operations -----------------------------------------------
void point_cmov(CTJacobianPoint* r, const CTJacobianPoint& a,
                std::uint64_t mask) noexcept;

CTJacobianPoint point_select(const CTJacobianPoint& a, const CTJacobianPoint& b,
                             std::uint64_t mask) noexcept;

// --- CT Table Lookup ---------------------------------------------------------
// Always reads ALL table entries. Returns table[index].
// table_size must equal the actual table array length.
CTJacobianPoint point_table_lookup(const CTJacobianPoint* table,
                                   std::size_t table_size,
                                   std::size_t index) noexcept;

// CT affine table lookup -- scans all entries, returns table[index].
CTAffinePoint affine_table_lookup(const CTAffinePoint* table,
                                  std::size_t table_size,
                                  std::size_t index) noexcept;

// Signed-digit affine table lookup (Hamburg encoding).
// Table stores odd multiples [1P, 3P, ..., (2^group_size-1)P].
// n is a group_size-bit value interpreted as signed digit.
// Result is NEVER infinity.
CTAffinePoint affine_table_lookup_signed(const CTAffinePoint* table,
                                          std::size_t table_size,
                                          std::uint64_t n,
                                          unsigned group_size) noexcept;

// In-place variant: writes result directly into *out (avoids 88-byte copy).
void affine_table_lookup_signed_into(CTAffinePoint* out,
                                      const CTAffinePoint* table,
                                      std::size_t table_size,
                                      std::uint64_t n,
                                      unsigned group_size) noexcept;

// --- CT Batch Doubling -------------------------------------------------------
// In-place batch N doublings -- modifies r directly, no return copy.
void point_dbl_n_inplace(CTJacobianPoint* r, unsigned n) noexcept;

// Return-by-value wrapper (kept for backward compatibility).
CTJacobianPoint point_dbl_n(const CTJacobianPoint& p, unsigned n) noexcept;

// CT conditional operations on affine points
void affine_cmov(CTAffinePoint* r, const CTAffinePoint& a,
                 std::uint64_t mask) noexcept;

// --- CT Scalar Multiplication ------------------------------------------------
// Hamburg signed-digit comb + GLV endomorphism (GROUP_SIZE=5).
//
// Method: Transform scalar via s = (k+K)/2, GLV split -> v1, v2 (129 bits each).
// Process 26 groups of 5 bits, each yielding a guaranteed non-zero odd digit.
// Table: 16 odd multiples per curve ([1P, 3P, ..., 31P], [1lambdaP, ..., 31lambdaP]).
// Cost: 125 dbl + 52 unified_add + 52 signed_lookups(16).

Point scalar_mul(const Point& p, const Scalar& k) noexcept;

// --- CT X-Only ECDH (port of libsecp256k1 secp256k1_ecmult_const_xonly) ------
// Computes the x-coordinate of q * P where P has x-coordinate xn/xd on
// secp256k1, WITHOUT any sqrt computation.
//
// Algorithm (isomorphic curve, a=0):
//   g = xn³ + 7·xd³
//   P_eff = (g·xn, g²)  ← effective affine point on isomorphic curve
//   R = scalar_mul(P_eff, q)  ← CT Jacobian multiply (q is secret)
//   x = R.x / (R.z² · g · xd)  ← one combined field inversion
//
// Eliminates the ~3.8 µs sqrt + avoids point reconstruction overhead.
// On the 5x52 path, R is Jacobian so the Z²·g·xd correction uses ONE inverse.
// On the 4x64 fallback path, a sqrt is used (slower, for non-x86 platforms).
//
// Pass xd = FieldElement::one() when xn is already the full x-coordinate.
// Returns FieldElement::zero() on degenerate inputs (q==0, invalid point).
FieldElement ecmult_const_xonly(const FieldElement& xn, const FieldElement& xd,
                                 const Scalar& q) noexcept;

// Prebuilt GLV tables for a fixed base point P.
// Build once with build_scalar_mul_tables(); reuse across many scalar_mul calls
// with the same P. Saves ~1,954 ns per call (table build cost).
struct CTScalarMulTables {
#if defined(SECP256K1_FAST_52BIT) && !defined(SECP256K1_USE_4X64_POINT_OPS)
    static constexpr unsigned TABLE_SIZE = 16;  // GROUP_SIZE=5 → 1<<(5-1)
    CTAffinePoint pre_a    [TABLE_SIZE];         // [1P, 3P, ..., 31P] pseudo-affine
    CTAffinePoint pre_a_lam[TABLE_SIZE];         // [phi(P), ..., phi(31P)]
    FE52          global_z;                      // shared implicit Z denominator
#endif
    bool valid = false;
};

// Build GLV tables for P once. Returns valid=false if P is infinity.
CTScalarMulTables build_scalar_mul_tables(const Point& p) noexcept;

// CT scalar_mul using pre-built tables (skips the ~1,954 ns table build).
// Equivalent to scalar_mul(p, k) for the P used to build tables.
// Falls back to scalar_mul(p_fallback, k) when tables.valid==false.
Point scalar_mul_prebuilt(const CTScalarMulTables& tables,
                           const Point& p_fallback,
                           const Scalar& k) noexcept;

// BIP-324-optimized scalar_mul: incomplete mixed-add (7M+3S) instead of
// complete Brier-Joye (11M+6S).  Safe for BIP-324 ECDH: degenerate cases
// (P==Q, P==-Q) have ~2^-128 probability with random peer keys; wrong result
// causes handshake failure (no key leak).  ~25% faster than scalar_mul_prebuilt.
// MUST NOT be used for signing or operations requiring guaranteed correctness.
Point scalar_mul_prebuilt_fast(const CTScalarMulTables& tables,
                                const Point& p_fallback,
                                const Scalar& k) noexcept;

// CT generator multiplication: k * G
// Hamburg signed-digit encoding: v = (k + 2^256 - 1)/2 mod n.
// Every 4-bit window is guaranteed odd -> 8-entry table, no cmov skip.
// Cost: 64 unified_add + 64 signed_lookups(8). No doublings.
// Approximately 3x faster than generic scalar_mul(G, k).
Point generator_mul(const Scalar& k) noexcept;

// Initialize the precomputed generator table.
// Called automatically on first generator_mul() call.
// Can be called explicitly at startup to avoid first-call latency.
void init_generator_table() noexcept;

// --- CT GLV Endomorphism -----------------------------------------------------
// Apply secp256k1 endomorphism: phi(P) = (beta*X, Y, Z) where beta^3 == 1 (mod p).
// Constant-time (just one field multiplication, no branches).
CTJacobianPoint point_endomorphism(const CTJacobianPoint& p) noexcept;

// CT affine endomorphism
CTAffinePoint affine_endomorphism(const CTAffinePoint& p) noexcept;

// CT affine negation
CTAffinePoint affine_neg(const CTAffinePoint& p) noexcept;

// --- CT GLV Decomposition ----------------------------------------------------
// Split scalar k = k1 + k2*lambda (mod n) where |k1|, |k2| ~= 128 bits.
// Fully constant-time: no branches on scalar values.
struct CTGLVDecomposition {
    Scalar k1;           // |k1| ~= 128 bits (always positive after abs)
    Scalar k2;           // |k2| ~= 128 bits (always positive after abs)
    std::uint64_t k1_neg;  // all-ones if k1 was negated, 0 otherwise
    std::uint64_t k2_neg;  // all-ones if k2 was negated, 0 otherwise
};

CTGLVDecomposition ct_glv_decompose(const Scalar& k) noexcept;

// --- CT Verify (for ECDSA) --------------------------------------------------
// Returns all-ones mask if point is on curve, else 0. CT.
std::uint64_t point_is_on_curve(const Point& p) noexcept;

// CT equality check
std::uint64_t point_eq(const Point& a, const Point& b) noexcept;

// --- Context Scalar Blinding -------------------------------------------------
// Enables additive scalar blinding on the signing generator multiply path.
// When active, generator_mul_blinded(k) computes (k+r)*G - r*G = k*G, where
// r is a random scalar renewed by the caller (ufsecp_context_randomize).
// Blinding state is thread-local: each thread inherits a clean (inactive) state
// and must call set_blinding() to activate it.

// Install a new blinding factor r and its precomputed negation -r*G.
// r_G must equal r*G (computed via ct::generator_mul).  Caller is responsible.
void set_blinding(const Scalar& r, const Point& r_G) noexcept;

// Remove blinding (returns to unblinded ct::generator_mul).
void clear_blinding() noexcept;

// Blinded generator multiply for signing paths only.
// Returns k*G via (k+r)*G + (-r*G) when blinding is active; falls through to
// plain ct::generator_mul(k) otherwise.
Point generator_mul_blinded(const Scalar& k) noexcept;

} // namespace secp256k1::ct

#endif // SECP256K1_CT_POINT_HPP
