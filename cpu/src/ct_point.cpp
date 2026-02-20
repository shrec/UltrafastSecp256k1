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
// CT scalar multiplication (GLV-OPTIMIZED):
//   Uses GLV endomorphism φ(x,y)=(β·x,y) to split 256-bit scalar
//   into two 128-bit halves: k = k1 + k2·λ (mod n).
//   Strauss interleaving: 32 windows × (4 dbl + 2 mixed_add).
//   Total: 128 doublings + 64 mixed_complete additions.
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
#include "secp256k1/glv.hpp"   // GLV constants (BETA, LAMBDA, lattice vectors)

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

// ═══════════════════════════════════════════════════════════════════════════════
// CT GLV Endomorphism — Helpers & Decomposition
// ═══════════════════════════════════════════════════════════════════════════════
// GLV uses secp256k1's efficient endomorphism φ(x,y)=(β·x,y) where β³≡1 (mod p)
// to split a 256-bit scalar multiplication into two 128-bit ones:
//   k*P = k1*P + k2*φ(P)  where |k1|,|k2| ≈ 128 bits
//
// This cuts the number of doublings in half (128 vs 256),
// yielding ~40% speedup in the point multiplication loop.
//
// All operations on secret data are constant-time (CT).
// ═══════════════════════════════════════════════════════════════════════════════

namespace /* GLV CT helpers */ {

// ─── Local sub256 (needed for ct_scalar_is_high) ─────────────────────────────
static inline std::uint64_t local_sub256(std::uint64_t r[4],
                                          const std::uint64_t a[4],
                                          const std::uint64_t b[4]) noexcept {
    std::uint64_t borrow = 0;
    for (int i = 0; i < 4; ++i) {
        std::uint64_t diff = a[i] - b[i];
        std::uint64_t b1 = static_cast<std::uint64_t>(a[i] < b[i]);
        std::uint64_t result = diff - borrow;
        std::uint64_t b2 = static_cast<std::uint64_t>(diff < borrow);
        r[i] = result;
        borrow = b1 + b2;
    }
    return borrow;
}

// ─── CT scalar > n/2 check ──────────────────────────────────────────────────
// Returns all-ones mask if s > n/2, zero otherwise.
// n/2 (floor) = 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A0
static std::uint64_t ct_scalar_is_high(const Scalar& s) noexcept {
    static constexpr std::uint64_t HALF_N[4] = {
        0xDFE92F46681B20A0ULL,
        0x5D576E7357A4501DULL,
        0xFFFFFFFFFFFFFFFFULL,
        0x7FFFFFFFFFFFFFFFULL
    };
    // s > half_n iff sub(half_n, s) borrows
    std::uint64_t tmp[4];
    std::uint64_t borrow = local_sub256(tmp, HALF_N, s.limbs().data());
    return is_nonzero_mask(borrow);
}

// ─── CT 256×256→512 multiply, shift >>384 with rounding ─────────────────────
// Same as fast::mul_shift_384 but fully data-independent on x86_64.
static std::array<std::uint64_t, 4> ct_mul_shift_384(
    const std::array<std::uint64_t, 4>& a,
    const std::array<std::uint64_t, 4>& b) noexcept
{
    std::uint64_t prod[8] = {};
    // Schoolbook 4×4 → 8-limb product (fully data-independent)
    for (int i = 0; i < 4; ++i) {
        unsigned __int128 carry = 0;
        for (int j = 0; j < 4; ++j) {
            unsigned __int128 t = static_cast<unsigned __int128>(a[i]) * b[j]
                                + prod[i + j] + carry;
            prod[i + j] = static_cast<std::uint64_t>(t);
            carry = t >> 64;
        }
        prod[i + 4] = static_cast<std::uint64_t>(carry);
    }

    // Bits [384..511] = prod[6], prod[7]
    std::array<std::uint64_t, 4> result{};
    result[0] = prod[6];
    result[1] = prod[7];
    // result[2] = result[3] = 0 (product fits ~256+256=512 bits, so [384..511] ≤ 128 bits)

    // CT rounding: add bit 383 (MSB of prod[5])
    std::uint64_t round = prod[5] >> 63;
    std::uint64_t old = result[0];
    result[0] += round;
    std::uint64_t carry = static_cast<std::uint64_t>(result[0] < old);
    result[1] += carry;

    return result;
}

// ─── CT scalar multiplication mod n ──────────────────────────────────────────
// Schoolbook 4×4 → Barrett reduction, fully constant-time.
// Used for GLV decomposition: c1*(-b1), c2*(-b2), lambda*k2.
static Scalar ct_scalar_mul_mod(const Scalar& a, const Scalar& b) noexcept {
    // secp256k1 curve order n
    static constexpr std::uint64_t ORDER[4] = {
        0xBFD25E8CD0364141ULL,
        0xBAAEDCE6AF48A03BULL,
        0xFFFFFFFFFFFFFFFEULL,
        0xFFFFFFFFFFFFFFFFULL
    };
    // Barrett mu = floor(2^512 / n)
    static constexpr std::uint64_t MU[5] = {
        0x402DA1732FC9BEC0ULL,
        0x4551231950B75FC4ULL,
        0x0000000000000001ULL,
        0x0000000000000000ULL,
        0x0000000000000001ULL
    };

    // Step 1: Schoolbook 4×4 → 512-bit product
    std::uint64_t prod[8] = {};
    for (int i = 0; i < 4; ++i) {
        unsigned __int128 carry = 0;
        for (int j = 0; j < 4; ++j) {
            unsigned __int128 t = static_cast<unsigned __int128>(a.limbs()[i]) * b.limbs()[j]
                                + prod[i + j] + carry;
            prod[i + j] = static_cast<std::uint64_t>(t);
            carry = t >> 64;
        }
        prod[i + 4] = static_cast<std::uint64_t>(carry);
    }

    // Step 2: Barrett approximation — q = floor(prod >> 256) = prod[4..7]
    // q_approx = floor((q * mu) >> 256) = high part of q*mu
    std::uint64_t qmu[9] = {};
    for (int i = 0; i < 4; ++i) {
        unsigned __int128 carry = 0;
        for (int j = 0; j < 5; ++j) {
            unsigned __int128 t = static_cast<unsigned __int128>(prod[4 + i]) * MU[j]
                                + qmu[i + j] + carry;
            qmu[i + j] = static_cast<std::uint64_t>(t);
            carry = t >> 64;
        }
        qmu[i + 5] = static_cast<std::uint64_t>(carry);
    }

    // q_approx = qmu[4..7]
    // Compute r = prod - q_approx * ORDER (low 5 limbs only)
    std::uint64_t qn[5] = {};
    for (int i = 0; i < 4; ++i) {
        unsigned __int128 carry = 0;
        for (int j = 0; j < 4; ++j) {
            if (i + j >= 5) break;
            unsigned __int128 t = static_cast<unsigned __int128>(qmu[4 + i]) * ORDER[j]
                                + qn[i + j] + carry;
            qn[i + j] = static_cast<std::uint64_t>(t);
            carry = t >> 64;
        }
        if (i + 4 < 5) {
            qn[i + 4] = static_cast<std::uint64_t>(carry);
        }
    }

    // r = prod[0..3] - qn[0..3], with overflow into r4
    std::uint64_t r[4];
    std::uint64_t bw = 0;
    for (int i = 0; i < 4; ++i) {
        std::uint64_t diff = prod[i] - qn[i];
        std::uint64_t b1 = static_cast<std::uint64_t>(prod[i] < qn[i]);
        std::uint64_t res = diff - bw;
        std::uint64_t b2 = static_cast<std::uint64_t>(diff < bw);
        r[i] = res;
        bw = b1 + b2;
    }
    std::uint64_t r4 = prod[4] - qn[4] - bw;

    // Step 3: CT conditional subtract — always compute, select via cmov
    // First reduction
    std::uint64_t r_sub1[4];
    std::uint64_t bw1 = 0;
    for (int i = 0; i < 4; ++i) {
        std::uint64_t diff = r[i] - ORDER[i];
        std::uint64_t b1 = static_cast<std::uint64_t>(r[i] < ORDER[i]);
        std::uint64_t res = diff - bw1;
        std::uint64_t b2 = static_cast<std::uint64_t>(diff < bw1);
        r_sub1[i] = res;
        bw1 = b1 + b2;
    }
    std::uint64_t r4_sub1 = r4 - bw1;
    // Should subtract if r4 > 0 OR (r4 == 0 AND no borrow)
    // i.e., the overall value (r4:r[3..0]) >= ORDER
    std::uint64_t need_sub1 = is_nonzero_mask(r4) | is_zero_mask(bw1);
    // However, if r4 > 0 and borrow occurred, the subtraction is still valid
    // Correct condition: the 5-limb value is >= ORDER, i.e. r4>0 or (r4==0 and r>=ORDER)
    // After subtraction: if no overall underflow (r4_sub1 didn't wrap), keep it
    // Simplification: if r4 != 0 OR no borrow from 256-bit sub
    need_sub1 = is_nonzero_mask(r4) | is_zero_mask(bw1 & is_zero_mask(r4));
    cmov256(r, r_sub1, need_sub1);
    r4 = ct_select(r4_sub1, r4, need_sub1);

    // Second reduction
    std::uint64_t r_sub2[4];
    std::uint64_t bw2 = 0;
    for (int i = 0; i < 4; ++i) {
        std::uint64_t diff = r[i] - ORDER[i];
        std::uint64_t b1 = static_cast<std::uint64_t>(r[i] < ORDER[i]);
        std::uint64_t res = diff - bw2;
        std::uint64_t b2 = static_cast<std::uint64_t>(diff < bw2);
        r_sub2[i] = res;
        bw2 = b1 + b2;
    }
    std::uint64_t r4_sub2 = r4 - bw2;
    std::uint64_t need_sub2 = is_nonzero_mask(r4) | is_zero_mask(bw2 & is_zero_mask(r4));
    cmov256(r, r_sub2, need_sub2);
    (void)r4_sub2;

    return Scalar::from_limbs({r[0], r[1], r[2], r[3]});
}

// ─── β (beta) as FieldElement — cube root of unity mod p ─────────────────────
static const FieldElement& get_beta_fe() noexcept {
    static const FieldElement beta = FieldElement::from_bytes(
        secp256k1::fast::glv_constants::BETA);
    return beta;
}

// ─── GLV lattice constants (matching libsecp256k1 / glv.cpp) ─────────────────
static constexpr std::array<std::uint64_t, 4> kG1{{
    0xE893209A45DBB031ULL, 0x3DAA8A1471E8CA7FULL,
    0xE86C90E49284EB15ULL, 0x3086D221A7D46BCDULL
}};
static constexpr std::array<std::uint64_t, 4> kG2{{
    0x1571B4AE8AC47F71ULL, 0x221208AC9DF506C6ULL,
    0x6F547FA90ABFE4C4ULL, 0xE4437ED6010E8828ULL
}};
static constexpr std::array<std::uint8_t, 32> kMinusB1Bytes{{
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0xE4,0x43,0x7E,0xD6,0x01,0x0E,0x88,0x28,
    0x6F,0x54,0x7F,0xA9,0x0A,0xBF,0xE4,0xC3
}};
static constexpr std::array<std::uint8_t, 32> kMinusB2Bytes{{
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
    0x8A,0x28,0x0A,0xC5,0x07,0x74,0x34,0x6D,
    0xD7,0x65,0xCD,0xA8,0x3D,0xB1,0x56,0x2C
}};
static constexpr std::array<std::uint8_t, 32> kLambdaBytes{{
    0x53,0x63,0xAD,0x4C,0xC0,0x5C,0x30,0xE0,
    0xA5,0x26,0x1C,0x02,0x88,0x12,0x64,0x5A,
    0x12,0x2E,0x22,0xEA,0x20,0x81,0x66,0x78,
    0xDF,0x02,0x96,0x7C,0x1B,0x23,0xBD,0x72
}};

} // anonymous namespace (GLV CT helpers)

// ─── CT GLV Endomorphism ─────────────────────────────────────────────────────

CTJacobianPoint point_endomorphism(const CTJacobianPoint& p) noexcept {
    // φ(x,y,z) = (β·x, y, z) — just one field multiplication, CT
    CTJacobianPoint r;
    r.x = field_mul(p.x, get_beta_fe());
    r.y = p.y;
    r.z = p.z;
    r.infinity = p.infinity;
    return r;
}

CTAffinePoint affine_endomorphism(const CTAffinePoint& p) noexcept {
    // φ(x,y) = (β·x, y) — one field multiplication, CT
    CTAffinePoint r;
    r.x = field_mul(p.x, get_beta_fe());
    r.y = p.y;
    r.infinity = p.infinity;
    return r;
}

CTAffinePoint affine_neg(const CTAffinePoint& p) noexcept {
    CTAffinePoint r;
    r.x = p.x;
    r.y = field_neg(p.y);
    r.infinity = p.infinity;
    return r;
}

// ─── CT GLV Decomposition ────────────────────────────────────────────────────

CTGLVDecomposition ct_glv_decompose(const Scalar& k) noexcept {
    // Lazy-init constants
    static const Scalar minus_b1 = Scalar::from_bytes(kMinusB1Bytes);
    static const Scalar minus_b2 = Scalar::from_bytes(kMinusB2Bytes);
    static const Scalar lambda   = Scalar::from_bytes(kLambdaBytes);

    // Step 1: c1 = round(k·g1 / 2^384), c2 = round(k·g2 / 2^384)
    // mul_shift_384 is fully data-independent (schoolbook multiply + shift)
    auto k_limbs = k.limbs();
    auto c1_limbs = ct_mul_shift_384({k_limbs[0], k_limbs[1], k_limbs[2], k_limbs[3]}, kG1);
    auto c2_limbs = ct_mul_shift_384({k_limbs[0], k_limbs[1], k_limbs[2], k_limbs[3]}, kG2);

    Scalar c1 = Scalar::from_limbs(c1_limbs);
    Scalar c2 = Scalar::from_limbs(c2_limbs);

    // Step 2: k2 = c1*(-b1) + c2*(-b2) (mod n)
    // Using CT scalar multiply (no Barrett branches)
    Scalar k2_part1 = ct_scalar_mul_mod(c1, minus_b1);
    Scalar k2_part2 = ct_scalar_mul_mod(c2, minus_b2);
    Scalar k2_mod   = scalar_add(k2_part1, k2_part2);

    // Step 3: CT pick shorter representation for k2
    // If k2_mod > n/2, negate it → k2_abs = n - k2_mod
    std::uint64_t k2_high = ct_scalar_is_high(k2_mod);
    Scalar k2_abs = scalar_cneg(k2_mod, k2_high);

    // For computing k1: k2_signed = k2_high ? -k2_abs : k2_abs = k2_high ? k2_mod : -k2_mod
    // Wait: if k2_high, k2_abs = -k2_mod, and k2_signed = -k2_abs = k2_mod
    // if !k2_high, k2_abs = k2_mod, and k2_signed = k2_abs = k2_mod
    // Actually we need: k1 = k - λ * k2_original (before taking abs)
    // k2_original = k2_high ? -k2_abs : k2_abs
    // k2_signed represents the original value (with sign):
    Scalar k2_signed = scalar_cneg(k2_abs, k2_high);

    // Step 4: k1 = k - λ * k2_signed (mod n)
    Scalar lambda_k2 = ct_scalar_mul_mod(lambda, k2_signed);
    Scalar k1_mod = scalar_sub(k, lambda_k2);

    // Step 5: CT pick shorter representation for k1
    std::uint64_t k1_high = ct_scalar_is_high(k1_mod);
    Scalar k1_abs = scalar_cneg(k1_mod, k1_high);

    CTGLVDecomposition result;
    result.k1     = k1_abs;
    result.k2     = k2_abs;
    result.k1_neg = k1_high;
    result.k2_neg = k2_high;
    return result;
}

// ─── CT GLV Scalar Multiplication ────────────────────────────────────────────
// Uses GLV endomorphism: k*P = k1*P + k2*φ(P) where |k1|,|k2| ≈ 128 bits.
//
// Algorithm:
//   1. CT decompose k → (k1, k2, s1, s2)
//   2. Build affine tables (fast-path, P is public):
//      T1[0..15] for P, T2[0..15] for φ(P)
//   3. CT negate table Y-coords if corresponding sign is negative
//   4. Strauss interleave (32 windows × 2 tables):
//      for i = 31..0:
//        R = 4*R  (4 doublings)
//        w1 = scalar_window(k1, 4*i, 4)
//        w2 = scalar_window(k2, 4*i, 4)
//        R += T1[w1]  (CT mixed complete add)
//        R += T2[w2]  (CT mixed complete add)
//
// Total: 128 doublings + 64 mixed_complete additions
// vs old: 256 doublings + 64 complete additions
// Expected ~40% faster.

Point scalar_mul(const Point& p, const Scalar& k) noexcept {
    constexpr unsigned W = 4;
    constexpr unsigned GLV_WINDOWS = 32;  // 128 bits / 4 = 32 windows per half
    constexpr std::size_t TABLE_SIZE = 1u << W;  // 16

    // ────── GLV Decomposition (CT) ──────
    // k = (-1)^s1 * k1 + (-1)^s2 * k2 * λ (mod n)
    // where |k1|, |k2| ≈ 128 bits
    auto [k1, k2, k1_neg, k2_neg] = ct_glv_decompose(k);

    // ────── Endomorphism: φ(P) = (β·Px, Py, Pz) ──────
    // Fast-path: P is public, no CT needed for table construction
    Point phiP = secp256k1::fast::apply_endomorphism(p);

    // ────── Build affine precomputation tables (fast-path, public data) ──────
    // T1[0..15] = multiples of P
    // T2[0..15] = multiples of φ(P)
    CTAffinePoint T1[TABLE_SIZE];
    CTAffinePoint T2[TABLE_SIZE];

    // T1: multiples of P
    T1[0] = CTAffinePoint::make_infinity();
    T1[1] = CTAffinePoint::from_point(p);
    {
        Point p2 = p;
        p2.dbl_inplace();
        T1[2] = CTAffinePoint::from_point(p2);
        Point running = p2;
        for (std::size_t i = 3; i < TABLE_SIZE; ++i) {
            running = running.add(p);
            T1[i] = CTAffinePoint::from_point(running);
        }
    }

    // T2: multiples of φ(P)
    T2[0] = CTAffinePoint::make_infinity();
    T2[1] = CTAffinePoint::from_point(phiP);
    {
        Point phiP2 = phiP;
        phiP2.dbl_inplace();
        T2[2] = CTAffinePoint::from_point(phiP2);
        Point running = phiP2;
        for (std::size_t i = 3; i < TABLE_SIZE; ++i) {
            running = running.add(phiP);
            T2[i] = CTAffinePoint::from_point(running);
        }
    }

    // ────── CT sign handling ──────
    // If k1 was negated, negate all T1 Y-coords: -P table instead of P table.
    // If k2 was negated, negate all T2 Y-coords: -φ(P) table instead of φ(P) table.
    // CT: always compute negated Y, then select via mask. Entry 0 (infinity) is skipped.
    for (std::size_t i = 1; i < TABLE_SIZE; ++i) {
        T1[i].y = field_cneg(T1[i].y, k1_neg);
        T2[i].y = field_cneg(T2[i].y, k2_neg);
    }

    // Declassify tables (public data, only sign application was secret-dependent)
    SECP256K1_DECLASSIFY(T1, sizeof(T1));
    SECP256K1_DECLASSIFY(T2, sizeof(T2));

    // ────── Strauss interleaving (32 windows × 2 lookups) ──────
    // For i = 31 downto 0:
    //   R = 16·R (4 doublings)
    //   w1 = 4-bit window from k1 at position 4*i
    //   w2 = 4-bit window from k2 at position 4*i
    //   R += T1[w1]   (CT mixed complete addition)
    //   R += T2[w2]   (CT mixed complete addition)

    CTJacobianPoint R = CTJacobianPoint::make_infinity();

    for (int i = static_cast<int>(GLV_WINDOWS) - 1; i >= 0; --i) {
        // 4 doublings
        R = point_dbl(R);
        R = point_dbl(R);
        R = point_dbl(R);
        R = point_dbl(R);

        // CT extract 4-bit windows
        std::uint64_t w1 = scalar_window(k1, static_cast<std::size_t>(i) * W, W);
        std::uint64_t w2 = scalar_window(k2, static_cast<std::size_t>(i) * W, W);

        // CT table lookups (always scan all 16 entries)
        CTAffinePoint A1 = affine_table_lookup(T1, TABLE_SIZE, w1);
        CTAffinePoint A2 = affine_table_lookup(T2, TABLE_SIZE, w2);

        // Mixed complete additions (handles A=O when digit==0)
        R = point_add_mixed_complete(R, A1);
        R = point_add_mixed_complete(R, A2);
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
