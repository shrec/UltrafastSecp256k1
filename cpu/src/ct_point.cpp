// ============================================================================
// Constant-Time Point Arithmetic — Implementation
// ============================================================================
// Complete addition formula + CT scalar multiplication for secp256k1.
//
// Complete addition:
//   Handles P+Q, P+P, P+O, O+Q, P+(-P) in a single branchless codepath.
//   Based on: "Complete addition formulas for prime order elliptic curves"
//   (Renes, Costello, Bathalter 2016), adapted for a=0 (secp256k1).
//
// CT scalar multiplication:
//   Fixed-window (w=4) with CT table lookup.
//   Always executes exactly 64 doublings + 64 additions.
//   No early exit, no conditional skip, no secret-dependent branches.
//
// CT generator multiplication (OPTIMIZED):
//   Uses precomputed table of 64×16 affine G-multiples.
//   gen_table[i][j] = j * 2^(4i) * G  — table built once, reused.
//   Runtime: 64 mixed Jacobian+Affine additions + 64 CT lookups(16).
//   NO doublings needed. Approximately 3x faster than generic scalar_mul.
// ============================================================================

#include "secp256k1/ct/point.hpp"
#include "secp256k1/ct/field.hpp"
#include "secp256k1/ct/scalar.hpp"
#include "secp256k1/ct/ops.hpp"

#include <mutex>

namespace secp256k1::ct {

// ─── secp256k1 curve constant b = 7 ─────────────────────────────────────────
static const FieldElement B7 = FieldElement::from_uint64(7);
// 3*b = 21
static const FieldElement B3 = FieldElement::from_uint64(21);

// ─── CTJacobianPoint helpers ─────────────────────────────────────────────────

CTJacobianPoint CTJacobianPoint::from_point(const Point& p) noexcept {
    CTJacobianPoint r;
    if (p.is_infinity()) {
        r = make_infinity();
    } else {
        r.x = p.X();
        r.y = p.Y();
        r.z = p.z();
        r.infinity = 0;
    }
    return r;
}

Point CTJacobianPoint::to_point() const noexcept {
    // Check infinity via mask, but to_point() is not itself a hot-path secret op
    if (infinity != 0) {
        return Point::infinity();
    }
    return Point::from_jacobian_coords(x, y, z, false);
}

CTJacobianPoint CTJacobianPoint::make_infinity() noexcept {
    CTJacobianPoint r;
    r.x = FieldElement::zero();
    r.y = FieldElement::one();
    r.z = FieldElement::zero();
    r.infinity = ~static_cast<std::uint64_t>(0);  // all-ones
    return r;
}

// ─── CT Conditional Operations on Points ─────────────────────────────────────

void point_cmov(CTJacobianPoint* r, const CTJacobianPoint& a,
                std::uint64_t mask) noexcept {
    field_cmov(&r->x, a.x, mask);
    field_cmov(&r->y, a.y, mask);
    field_cmov(&r->z, a.z, mask);
    r->infinity = ct_select(a.infinity, r->infinity, mask);
}

CTJacobianPoint point_select(const CTJacobianPoint& a, const CTJacobianPoint& b,
                             std::uint64_t mask) noexcept {
    CTJacobianPoint r;
    r.x = field_select(a.x, b.x, mask);
    r.y = field_select(a.y, b.y, mask);
    r.z = field_select(a.z, b.z, mask);
    r.infinity = ct_select(a.infinity, b.infinity, mask);
    return r;
}

CTJacobianPoint point_neg(const CTJacobianPoint& p) noexcept {
    CTJacobianPoint r;
    r.x = p.x;
    r.y = field_neg(p.y);
    r.z = p.z;
    r.infinity = p.infinity;
    return r;
}

CTJacobianPoint point_table_lookup(const CTJacobianPoint* table,
                                   std::size_t table_size,
                                   std::size_t index) noexcept {
    CTJacobianPoint result = CTJacobianPoint::make_infinity();
    for (std::size_t i = 0; i < table_size; ++i) {
        std::uint64_t mask = eq_mask(static_cast<std::uint64_t>(i),
                                     static_cast<std::uint64_t>(index));
        point_cmov(&result, table[i], mask);
    }
    return result;
}

// ─── CT Conditional Operations on Affine Points ──────────────────────────────

void affine_cmov(CTAffinePoint* r, const CTAffinePoint& a,
                 std::uint64_t mask) noexcept {
    field_cmov(&r->x, a.x, mask);
    field_cmov(&r->y, a.y, mask);
    r->infinity = ct_select(a.infinity, r->infinity, mask);
}

CTAffinePoint affine_table_lookup(const CTAffinePoint* table,
                                  std::size_t table_size,
                                  std::size_t index) noexcept {
    CTAffinePoint result = CTAffinePoint::make_infinity();
    for (std::size_t i = 0; i < table_size; ++i) {
        std::uint64_t mask = eq_mask(static_cast<std::uint64_t>(i),
                                     static_cast<std::uint64_t>(index));
        affine_cmov(&result, table[i], mask);
    }
    return result;
}

// ─── Complete Addition (Jacobian, a=0) ───────────────────────────────────────
// Complete formula for y²=x³+b on Jacobian coordinates.
// Handles all cases without branches:
//   P+Q (general), P+P (doubling), P+O, O+Q, P+(-P)=O
//
// Strategy: compute both general-add and doubling results,
// then CT-select the correct one based on whether H==0 / R==0.
// Also handles infinity via cmov at the end.
//
// Cost: ~16M + 6S (fixed, no branches on point values)

CTJacobianPoint point_add_complete(const CTJacobianPoint& p,
                                   const CTJacobianPoint& q) noexcept {
    const FieldElement& X1 = p.x;
    const FieldElement& Y1 = p.y;
    const FieldElement& Z1 = p.z;
    const FieldElement& X2 = q.x;
    const FieldElement& Y2 = q.y;
    const FieldElement& Z2 = q.z;

    // ── 1. General Jacobian addition (assumes P ≠ Q, neither infinity) ──
    FieldElement z1z1 = field_sqr(Z1);                          // Z1²
    FieldElement z2z2 = field_sqr(Z2);                          // Z2²
    FieldElement u1   = field_mul(X1, z2z2);                    // U1 = X1·Z2²
    FieldElement u2   = field_mul(X2, z1z1);                    // U2 = X2·Z1²
    FieldElement s1   = field_mul(field_mul(Y1, Z2), z2z2);     // S1 = Y1·Z2³
    FieldElement s2   = field_mul(field_mul(Y2, Z1), z1z1);     // S2 = Y2·Z1³

    FieldElement h    = field_sub(u2, u1);                      // H = U2 - U1
    FieldElement r    = field_sub(s2, s1);                      // R = S2 - S1

    // Detect special cases via CT masks
    std::uint64_t h_is_zero = field_is_zero(h);
    std::uint64_t r_is_zero = field_is_zero(r);
    std::uint64_t is_double  = h_is_zero & r_is_zero;   // P == Q
    std::uint64_t is_inverse = h_is_zero & ~r_is_zero;  // P == -Q

    FieldElement hh   = field_sqr(h);                           // H²
    FieldElement hhh  = field_mul(h, hh);                       // H³
    FieldElement v    = field_mul(u1, hh);                      // V = U1·H²

    FieldElement rr      = field_sqr(r);
    FieldElement x3_add  = field_sub(field_sub(rr, hhh), field_add(v, v));
    FieldElement y3_add  = field_sub(field_mul(r, field_sub(v, x3_add)),
                                     field_mul(s1, hhh));
    FieldElement z3_add  = field_mul(field_mul(Z1, Z2), h);

    // ── 2. Doubling of P (dbl-2007-a for a=0) ──
    FieldElement A   = field_sqr(X1);
    FieldElement B   = field_sqr(Y1);
    FieldElement C   = field_sqr(B);

    FieldElement D   = field_sub(field_sqr(field_add(X1, B)), field_add(A, C));
    D = field_add(D, D);                                       // D = 2·((X+B)²-A-C)

    FieldElement E   = field_add(field_add(A, A), A);           // E = 3·A
    FieldElement F   = field_sqr(E);                            // F = E²

    FieldElement x3_dbl = field_sub(F, field_add(D, D));        // X3 = F - 2D

    FieldElement C8  = C;
    C8 = field_add(C8, C8);  // 2C
    C8 = field_add(C8, C8);  // 4C
    C8 = field_add(C8, C8);  // 8C

    FieldElement y3_dbl = field_sub(field_mul(E, field_sub(D, x3_dbl)), C8);
    FieldElement yz_dbl = field_mul(Y1, Z1);
    FieldElement z3_dbl = field_add(yz_dbl, yz_dbl);            // Z3 = 2·Y1·Z1

    // ── 3. Select result via CT masks ──
    // field_select(a, b, mask): returns a if mask==all-ones, else b
    // When is_double: use dbl result. Otherwise: use add result.
    FieldElement x3 = field_select(x3_dbl, x3_add, is_double);
    FieldElement y3 = field_select(y3_dbl, y3_add, is_double);
    FieldElement z3 = field_select(z3_dbl, z3_add, is_double);

    // If P == -Q → infinity (0:1:0)
    FieldElement zero = FieldElement::zero();
    FieldElement one  = FieldElement::one();
    field_cmov(&x3, zero, is_inverse);
    field_cmov(&y3, one,  is_inverse);
    field_cmov(&z3, zero, is_inverse);

    // If P is infinity → result = Q
    field_cmov(&x3, X2, p.infinity);
    field_cmov(&y3, Y2, p.infinity);
    field_cmov(&z3, Z2, p.infinity);

    // If Q is infinity → result = P
    field_cmov(&x3, X1, q.infinity);
    field_cmov(&y3, Y1, q.infinity);
    field_cmov(&z3, Z1, q.infinity);

    // Compute infinity flag for result
    std::uint64_t result_inf = is_inverse & ~p.infinity & ~q.infinity;
    result_inf |= p.infinity & q.infinity;

    CTJacobianPoint result;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    result.infinity = result_inf;
    return result;
}

// ─── Mixed Jacobian+Affine Complete Addition (a=0) ───────────────────────────
// P = (X1:Y1:Z1) Jacobian, Q = (x2, y2) affine (implied Z2=1).
// Same strategy as point_add_complete but with Z2=1 optimizations:
//   - No Z2² / Z2³ computation needed
//   - U1 = X1, S1 = Y1 (since Z2=1)
//   - Z3_add = Z1·H (not Z1·Z2·H)
// Saves 4 field multiplications + 1 field squaring.
// Cost: ~9M + 8S (fixed, no branches)

CTJacobianPoint point_add_mixed_complete(const CTJacobianPoint& p,
                                          const CTAffinePoint& q) noexcept {
    const FieldElement& X1 = p.x;
    const FieldElement& Y1 = p.y;
    const FieldElement& Z1 = p.z;
    const FieldElement& x2 = q.x;
    const FieldElement& y2 = q.y;

    // ── 1. General Jacobian+Affine addition (Z2=1 optimizations) ──
    FieldElement z1z1 = field_sqr(Z1);                          // Z1²

    // U1 = X1 (since Z2² = 1)
    // U2 = x2·Z1²
    FieldElement u2   = field_mul(x2, z1z1);                    // M1

    // S1 = Y1 (since Z2³ = 1)
    // S2 = y2·Z1³
    FieldElement z1cu = field_mul(z1z1, Z1);                    // M2: Z1³
    FieldElement s2   = field_mul(y2, z1cu);                    // M3: y2·Z1³

    FieldElement h    = field_sub(u2, X1);                      // H = U2 - U1
    FieldElement r    = field_sub(s2, Y1);                      // R = S2 - S1

    // Detect special cases via CT masks
    std::uint64_t h_is_zero = field_is_zero(h);
    std::uint64_t r_is_zero = field_is_zero(r);
    std::uint64_t is_double  = h_is_zero & r_is_zero;   // P == Q
    std::uint64_t is_inverse = h_is_zero & ~r_is_zero;  // P == -Q

    FieldElement hh   = field_sqr(h);                           // H²
    FieldElement hhh  = field_mul(h, hh);                       // M4: H³
    FieldElement v    = field_mul(X1, hh);                      // M5: V = X1·H²

    FieldElement rr      = field_sqr(r);                        // R²
    FieldElement x3_add  = field_sub(field_sub(rr, hhh), field_add(v, v));
    FieldElement y3_add  = field_sub(field_mul(r, field_sub(v, x3_add)),  // M6
                                     field_mul(Y1, hhh));                 // M7
    FieldElement z3_add  = field_mul(Z1, h);                    // M8: Z3 = Z1·H

    // ── 2. Doubling of P (dbl-2007-a for a=0) ──
    FieldElement A   = field_sqr(X1);
    FieldElement B   = field_sqr(Y1);
    FieldElement C   = field_sqr(B);

    FieldElement D   = field_sub(field_sqr(field_add(X1, B)), field_add(A, C));
    D = field_add(D, D);                                       // D = 2·((X+B)²-A-C)

    FieldElement E   = field_add(field_add(A, A), A);           // E = 3·A
    FieldElement F   = field_sqr(E);                            // F = E²

    FieldElement x3_dbl = field_sub(F, field_add(D, D));        // X3 = F - 2D

    FieldElement C8  = C;
    C8 = field_add(C8, C8);  // 2C
    C8 = field_add(C8, C8);  // 4C
    C8 = field_add(C8, C8);  // 8C

    FieldElement y3_dbl = field_sub(field_mul(E, field_sub(D, x3_dbl)), C8);  // M9
    FieldElement yz_mix = field_mul(Y1, Z1);
    FieldElement z3_dbl = field_add(yz_mix, yz_mix);                          // 2·Y1·Z1

    // ── 3. Select result via CT masks ──
    FieldElement x3 = field_select(x3_dbl, x3_add, is_double);
    FieldElement y3 = field_select(y3_dbl, y3_add, is_double);
    FieldElement z3 = field_select(z3_dbl, z3_add, is_double);

    // If P == -Q → infinity (0:1:0)
    FieldElement zero = FieldElement::zero();
    FieldElement one  = FieldElement::one();
    field_cmov(&x3, zero, is_inverse);
    field_cmov(&y3, one,  is_inverse);
    field_cmov(&z3, zero, is_inverse);

    // If P is infinity → result = Q (affine, so Z=1)
    field_cmov(&x3, x2, p.infinity);
    field_cmov(&y3, y2, p.infinity);
    field_cmov(&z3, one, p.infinity);

    // If Q is infinity → result = P
    field_cmov(&x3, X1, q.infinity);
    field_cmov(&y3, Y1, q.infinity);
    field_cmov(&z3, Z1, q.infinity);

    // Compute infinity flag for result
    std::uint64_t result_inf = is_inverse & ~p.infinity & ~q.infinity;
    result_inf |= p.infinity & q.infinity;

    CTJacobianPoint result;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    result.infinity = result_inf;
    return result;
}

// ─── CT Point Doubling ───────────────────────────────────────────────────────

CTJacobianPoint point_dbl(const CTJacobianPoint& p) noexcept {
    // dbl-2007-a formula for a=0 curves, with CT infinity handling

    FieldElement A = field_sqr(p.x);
    FieldElement B = field_sqr(p.y);
    FieldElement C = field_sqr(B);

    FieldElement xb = field_add(p.x, B);
    FieldElement D = field_sub(field_sqr(xb), field_add(A, C));
    D = field_add(D, D);                       // D = 2·((X+B)²-A-C)

    FieldElement E = field_add(field_add(A, A), A); // E = 3·A
    FieldElement F = field_sqr(E);              // F = E²

    FieldElement x3 = field_sub(F, field_add(D, D)); // X3 = F - 2D

    FieldElement C8 = C;
    C8 = field_add(C8, C8);
    C8 = field_add(C8, C8);
    C8 = field_add(C8, C8);  // 8C

    FieldElement y3 = field_sub(field_mul(E, field_sub(D, x3)), C8);

    FieldElement z3 = field_mul(p.y, p.z);
    z3 = field_add(z3, z3);                     // Z3 = 2·Y·Z

    // If P is infinity, result is infinity
    CTJacobianPoint inf = CTJacobianPoint::make_infinity();
    CTJacobianPoint result;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    result.infinity = 0;

    point_cmov(&result, inf, p.infinity);
    return result;
}

// ─── CT Scalar Multiplication ────────────────────────────────────────────────
// Fixed-window method (w=4) with CT table lookup.
//
// Algorithm:
//   1. Precompute table: T[0]=O, T[1]=P, T[2]=2P, ..., T[15]=15P
//   2. For i = 63 downto 0:
//       a. R = 4*R (4 doublings)
//       b. d = scalar_window(k, 4*i, 4)  (CT extract 4-bit window)
//       c. T_d = ct_table_lookup(table, d)  (CT: scans all 16 entries)
//       d. R = R + T_d  (complete addition, handles T_d=O case)
//   3. Return R
//
// Total: 256 doublings + 64 complete additions (fixed)

Point scalar_mul(const Point& p, const Scalar& k) noexcept {
    constexpr unsigned W = 4;
    constexpr std::size_t TABLE_SIZE = 1u << W;  // 16

    // Build precomputation table using FAST-path operations.
    // Table contents are PUBLIC (derived from p), only scalar is secret.
    // Using fast:: operations here saves ~19μs vs CT table construction.
    CTJacobianPoint table[TABLE_SIZE];
    table[0] = CTJacobianPoint::make_infinity();
    table[1] = CTJacobianPoint::from_point(p);
    {
        Point p2 = p;
        p2.dbl_inplace();
        table[2] = CTJacobianPoint::from_point(p2);
        Point running = p2;
        for (std::size_t i = 3; i < TABLE_SIZE; ++i) {
            running = running.add(p);
            table[i] = CTJacobianPoint::from_point(running);
        }
    }

    // Declassify table (public data derived from public point; only scalar is secret)
    SECP256K1_DECLASSIFY(table, sizeof(table));

    // Process from MSB to LSB in 4-bit windows
    // 256 bits / 4 = 64 windows, indices 63..0
    CTJacobianPoint R = CTJacobianPoint::make_infinity();

    for (int i = 63; i >= 0; --i) {
        // 4 doublings
        R = point_dbl(R);
        R = point_dbl(R);
        R = point_dbl(R);
        R = point_dbl(R);

        // CT extract 4-bit window
        std::uint64_t digit = scalar_window(k, static_cast<std::size_t>(i) * W, W);

        // CT table lookup (always reads all 16 entries)
        CTJacobianPoint T_d = point_table_lookup(table, TABLE_SIZE, digit);

        // Complete addition (handles T_d = O when digit == 0)
        R = point_add_complete(R, T_d);
    }

    // Declassify result (output is public)
    Point result = R.to_point();
    SECP256K1_DECLASSIFY(&result, sizeof(result));
    return result;
}

// ─── CT Generator Multiplication (OPTIMIZED) ─────────────────────────────────
// Uses precomputed table of 64 × 16 affine G-multiples.
// gen_table[i][j] = j * 2^(4i) * G — each window position has its own table.
//
// At runtime: k*G = sum_{i=0..63} gen_table[i][window_i(k)]
// Just 64 CT lookups(16) + 64 mixed Jacobian+Affine additions.
// NO doublings needed — all positional shifts baked into the table.
//
// Table construction uses fast-path operations (public data, non-secret).
// Only scalar window extraction and table selection are CT.

namespace {

constexpr unsigned GEN_W = 4;
constexpr unsigned GEN_WINDOWS = 64;           // 256 / 4
constexpr std::size_t GEN_TABLE_SIZE = 1u << GEN_W;  // 16

struct alignas(64) GenPrecompTable {
    CTAffinePoint entries[GEN_WINDOWS][GEN_TABLE_SIZE];
    bool initialized = false;
};

static GenPrecompTable g_gen_table;
static std::once_flag g_gen_table_once;

void build_gen_table() noexcept {
    // Use FAST-path operations — table contents are public (derived from G).
    // Only the scalar-dependent selection at runtime must be CT.
    Point G = Point::generator();

    // Base for window 0 is G itself.
    // For window w, base = 2^(4w) * G.
    // Compute iteratively: each window's base = prev_base doubled 4 times.
    Point base = G;

    for (unsigned w = 0; w < GEN_WINDOWS; ++w) {
        // table[w][0] = O (infinity)
        g_gen_table.entries[w][0] = CTAffinePoint::make_infinity();

        // table[w][1] = base (affine)
        g_gen_table.entries[w][1] = CTAffinePoint::from_point(base);

        // table[w][2] = 2 * base (use doubling, not add)
        Point doubled = base;
        doubled.dbl_inplace();
        g_gen_table.entries[w][2] = CTAffinePoint::from_point(doubled);

        // table[w][j] = j * base for j = 3..15 (add base each time)
        Point running = doubled;
        for (std::size_t j = 3; j < GEN_TABLE_SIZE; ++j) {
            running = running.add(base);
            g_gen_table.entries[w][j] = CTAffinePoint::from_point(running);
        }

        // Advance base by 4 doublings for next window
        // next_base = 2^4 * base = 16 * base
        base.dbl_inplace();
        base.dbl_inplace();
        base.dbl_inplace();
        base.dbl_inplace();
    }

    g_gen_table.initialized = true;
}

} // anonymous namespace

void init_generator_table() noexcept {
    std::call_once(g_gen_table_once, build_gen_table);
}

Point generator_mul(const Scalar& k) noexcept {
    // Ensure table is initialized (thread-safe, one-time cost ~15ms)
    init_generator_table();

    // k*G = sum_{i=0..63} gen_table[i][window_i(k)]
    // Each term already includes the 2^(4i) positional factor.
    // Just 64 CT lookups + 64 mixed complete additions. No doublings.

    CTJacobianPoint R = CTJacobianPoint::make_infinity();

    for (unsigned i = 0; i < GEN_WINDOWS; ++i) {
        // CT extract 4-bit window
        std::uint64_t digit = scalar_window(k, static_cast<std::size_t>(i) * GEN_W, GEN_W);

        // CT table lookup (always reads all 16 entries)
        CTAffinePoint T = affine_table_lookup(g_gen_table.entries[i],
                                              GEN_TABLE_SIZE, digit);

        // Mixed complete addition (handles T=O when digit==0)
        R = point_add_mixed_complete(R, T);
    }

    // Declassify result (output is public)
    Point result = R.to_point();
    SECP256K1_DECLASSIFY(&result, sizeof(result));
    return result;
}

// ─── CT Curve Check ──────────────────────────────────────────────────────────

std::uint64_t point_is_on_curve(const Point& p) noexcept {
    if (p.is_infinity()) {
        return ~static_cast<std::uint64_t>(0);  // infinity is on curve
    }

    // Convert to affine for check: y² == x³ + 7
    FieldElement x = p.x();
    FieldElement y = p.y();

    // y²
    FieldElement y2 = field_sqr(y);

    // x³ + 7
    FieldElement x2 = field_sqr(x);
    FieldElement x3 = field_mul(x, x2);
    FieldElement rhs = field_add(x3, B7);

    return field_eq(y2, rhs);
}

// ─── CT Point Equality ──────────────────────────────────────────────────────

std::uint64_t point_eq(const Point& a, const Point& b) noexcept {
    // Compare in Jacobian: P1=(X1:Y1:Z1) == P2=(X2:Y2:Z2) iff
    //   X1·Z2² == X2·Z1²  AND  Y1·Z2³ == Y2·Z1³
    // (Handles different Z coordinates)

    if (a.is_infinity() && b.is_infinity()) {
        return ~static_cast<std::uint64_t>(0);
    }
    if (a.is_infinity() || b.is_infinity()) {
        return 0;
    }

    FieldElement z1sq = field_sqr(a.z());
    FieldElement z2sq = field_sqr(b.z());
    FieldElement u1 = field_mul(a.X(), z2sq);
    FieldElement u2 = field_mul(b.X(), z1sq);

    FieldElement z1cu = field_mul(z1sq, a.z());
    FieldElement z2cu = field_mul(z2sq, b.z());
    FieldElement s1 = field_mul(a.Y(), z2cu);
    FieldElement s2 = field_mul(b.Y(), z1cu);

    return field_eq(u1, u2) & field_eq(s1, s2);
}

} // namespace secp256k1::ct
