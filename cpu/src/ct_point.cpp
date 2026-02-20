// ============================================================================
// Constant-Time Point Arithmetic — Implementation (5×52 Optimized)
// ============================================================================
// Complete addition formula + CT scalar multiplication for secp256k1.
//
// INTERNAL REPRESENTATION: FieldElement52 (5×52-bit limbs with lazy reduction)
// This eliminates the function-call overhead of ct::field_* wrappers and
// leverages the ~3.6× faster force-inlined 5×52 multiply/square kernels.
//
// Conversion to/from 4×64 FieldElement occurs only at API boundaries
// (from_point / to_point). All internal arithmetic stays in 5×52.
//
// Magnitude tracking: mul/sqr produce magnitude 1 output.
// Lazy adds accumulate magnitude. Max magnitude in point ops ≈ 30,
// well within the 4096 headroom (12 bits per limb).
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
//
// CT generator multiplication (OPTIMIZED):
//   Uses precomputed table of 64×16 affine G-multiples.
//   Runtime: 64 mixed Jacobian+Affine additions + 64 CT lookups(16).
//   NO doublings needed.
// ============================================================================

#include "secp256k1/ct/point.hpp"
#include "secp256k1/ct/field.hpp"
#include "secp256k1/ct/scalar.hpp"
#include "secp256k1/ct/ops.hpp"
#include "secp256k1/field_52.hpp"
#include "secp256k1/glv.hpp"

#include <mutex>

namespace secp256k1::ct {

// ─── Type aliases ────────────────────────────────────────────────────────────
using FE52 = secp256k1::fast::FieldElement52;

// ─── secp256k1 curve constant b = 7 (4×64 for API-boundary functions) ────────
static const FieldElement B7 = FieldElement::from_uint64(7);

// ─── 5×52 CT Helper Functions ────────────────────────────────────────────────
// All operations on FieldElement52 are inherently constant-time:
//   - mul/sqr: fixed __int128 multiply chain → MULX (data-independent)
//   - add: 5 plain uint64_t additions (no branches)
//   - negate: (m+1)*p - a (no branches)
//   - normalize: branchless mask-based conditional subtract
// Value barriers are applied in mask generation (is_zero_mask etc.)

namespace /* FE52 CT helpers */ {

// CT conditional move for 5×52 field element
inline void fe52_cmov(FE52* dst, const FE52& src, std::uint64_t mask) noexcept {
    dst->n[0] ^= (dst->n[0] ^ src.n[0]) & mask;
    dst->n[1] ^= (dst->n[1] ^ src.n[1]) & mask;
    dst->n[2] ^= (dst->n[2] ^ src.n[2]) & mask;
    dst->n[3] ^= (dst->n[3] ^ src.n[3]) & mask;
    dst->n[4] ^= (dst->n[4] ^ src.n[4]) & mask;
}

// CT select: returns a if mask==all-ones, else b
inline FE52 fe52_select(const FE52& a, const FE52& b, std::uint64_t mask) noexcept {
    FE52 r;
    r.n[0] = (a.n[0] & mask) | (b.n[0] & ~mask);
    r.n[1] = (a.n[1] & mask) | (b.n[1] & ~mask);
    r.n[2] = (a.n[2] & mask) | (b.n[2] & ~mask);
    r.n[3] = (a.n[3] & mask) | (b.n[3] & ~mask);
    r.n[4] = (a.n[4] & mask) | (b.n[4] & ~mask);
    return r;
}

// CT zero check: normalizes then checks. Returns all-ones if zero.
inline std::uint64_t fe52_is_zero(const FE52& a) noexcept {
    FE52 tmp = a;
    tmp.normalize();
    std::uint64_t z = tmp.n[0] | tmp.n[1] | tmp.n[2] | tmp.n[3] | tmp.n[4];
    return is_zero_mask(z);
}

// CT equality: normalizes both, then compares. Returns all-ones if equal.
inline std::uint64_t fe52_eq(const FE52& a, const FE52& b) noexcept {
    FE52 ta = a; ta.normalize();
    FE52 tb = b; tb.normalize();
    std::uint64_t diff = (ta.n[0] ^ tb.n[0]) | (ta.n[1] ^ tb.n[1]) |
                         (ta.n[2] ^ tb.n[2]) | (ta.n[3] ^ tb.n[3]) |
                         (ta.n[4] ^ tb.n[4]);
    return is_zero_mask(diff);
}

// CT conditional negate: if mask==all-ones, return -a; else a.
// 'mag' is the current magnitude of a.
inline FE52 fe52_cneg(const FE52& a, std::uint64_t mask, unsigned mag) noexcept {
    FE52 neg = a.negate(mag);
    return fe52_select(neg, a, mask);
}

// ─── FE52 Batch Inversion (Montgomery's trick) ──────────────────────────────
// Computes z_inv[i] = z[i]^{-1} mod p for all i in [0, n).
// Cost: 1 inversion + 3(n-1) multiplications (vs n inversions).
// All z[i] MUST be non-zero (caller ensures by excluding infinity entries).
inline void fe52_batch_inverse(FE52* z_inv, const FE52* z, std::size_t n) noexcept {
    if (n == 0) return;
    if (n == 1) {
        z_inv[0] = FE52::from_fe(z[0].to_fe().inverse());
        return;
    }

    // Forward accumulation: use z_inv[] as scratch for partial products
    //   z_inv[i] = z[0] * z[1] * ... * z[i]
    z_inv[0] = z[0];
    for (std::size_t i = 1; i < n; ++i) {
        z_inv[i] = z_inv[i - 1] * z[i];   // M=1 after mul
    }

    // Single inversion of the accumulated product (via 4×64 path)
    FE52 acc = FE52::from_fe(z_inv[n - 1].to_fe().inverse());

    // Backward propagation: extract individual inverses
    for (std::size_t i = n - 1; i > 0; --i) {
        z_inv[i] = z_inv[i - 1] * acc;   // z_inv[i] = (z[0]..z[i-1]) * (z[0]..z[i])^{-1} = z[i]^{-1}
        acc = acc * z[i];                 // acc = (z[0]..z[i-1])^{-1}
    }
    z_inv[0] = acc;
}

} // anonymous namespace (FE52 CT helpers)


// ─── CTJacobianPoint helpers ─────────────────────────────────────────────────

CTJacobianPoint CTJacobianPoint::from_point(const Point& p) noexcept {
    CTJacobianPoint r;
    if (p.is_infinity()) {
        r = make_infinity();
    } else {
        r.x = FE52::from_fe(p.X());
        r.y = FE52::from_fe(p.Y());
        r.z = FE52::from_fe(p.z());
        r.infinity = 0;
    }
    return r;
}

Point CTJacobianPoint::to_point() const noexcept {
    if (infinity != 0) {
        return Point::infinity();
    }
    return Point::from_jacobian_coords(x.to_fe(), y.to_fe(), z.to_fe(), false);
}

CTJacobianPoint CTJacobianPoint::make_infinity() noexcept {
    CTJacobianPoint r;
    r.x = FE52::zero();
    r.y = FE52::one();
    r.z = FE52::zero();
    r.infinity = ~static_cast<std::uint64_t>(0);
    return r;
}

// ─── CT Conditional Operations on Points ─────────────────────────────────────

void point_cmov(CTJacobianPoint* r, const CTJacobianPoint& a,
                std::uint64_t mask) noexcept {
    fe52_cmov(&r->x, a.x, mask);
    fe52_cmov(&r->y, a.y, mask);
    fe52_cmov(&r->z, a.z, mask);
    r->infinity = ct_select(a.infinity, r->infinity, mask);
}

CTJacobianPoint point_select(const CTJacobianPoint& a, const CTJacobianPoint& b,
                             std::uint64_t mask) noexcept {
    CTJacobianPoint r;
    r.x = fe52_select(a.x, b.x, mask);
    r.y = fe52_select(a.y, b.y, mask);
    r.z = fe52_select(a.z, b.z, mask);
    r.infinity = ct_select(a.infinity, b.infinity, mask);
    return r;
}

CTJacobianPoint point_neg(const CTJacobianPoint& p) noexcept {
    CTJacobianPoint r;
    r.x = p.x;
    // normalize_weak before negate to ensure magnitude 1
    FE52 yn = p.y; yn.normalize_weak();
    r.y = yn.negate(1);  // -(Y) mod p; magnitude 2
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
    fe52_cmov(&r->x, a.x, mask);
    fe52_cmov(&r->y, a.y, mask);
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

// ─── Signed-Digit Affine Table Lookup (Hamburg encoding) ─────────────────────
// Table stores odd multiples: [1·P, 3·P, 5·P, ..., (2^group_size - 1)·P]
// (table_size = 2^(group_size-1) entries)
//
// n is a group_size-bit value. Its bits are interpreted as signs of powers:
//   value = sum((2*bit[i]-1) * 2^i, i=0..group_size-1)
// This always yields an ODD number in [-(2^group_size-1), (2^group_size-1)].
//
// The top bit selects sign (0 = negative), remaining bits encode the index.
// Result is NEVER infinity — the signed-digit transform guarantees this.

CTAffinePoint affine_table_lookup_signed(const CTAffinePoint* table,
                                          std::size_t table_size,
                                          std::uint64_t n,
                                          unsigned group_size) noexcept {
    // Top bit 0 → negative (we XOR lower bits to get correct index)
    std::uint64_t negative = ((n >> (group_size - 1)) ^ 1u) & 1u;
    std::uint64_t neg_mask = static_cast<std::uint64_t>(-negative);  // all-ones if neg

    // Compute index into table:
    // If positive (top bit=1): index = lower bits
    // If negative (top bit=0): index = ~(lower bits) & mask
    unsigned index_bits = group_size - 1;
    std::uint64_t index = (static_cast<std::uint64_t>(-negative) ^ n) &
                          ((1ULL << index_bits) - 1u);

    // CT scan of table
    CTAffinePoint result = table[0];
    for (std::size_t m = 1; m < table_size; ++m) {
        std::uint64_t mask = eq_mask(static_cast<std::uint64_t>(m),
                                     static_cast<std::uint64_t>(index));
        affine_cmov(&result, table[m], mask);
    }

    // Conditional Y-negate (sign handling)
    FE52 neg_y = result.y;
    neg_y.normalize_weak();
    neg_y = neg_y.negate(1);
    fe52_cmov(&result.y, neg_y, neg_mask);

    result.infinity = 0;  // signed digits are NEVER zero
    return result;
}

// ─── Complete Addition (Jacobian, a=0, 5×52) ────────────────────────────────
// Complete formula for y²=x³+b on Jacobian coordinates.
// Handles all cases without branches:
//   P+Q (general), P+P (doubling), P+O, O+Q, P+(-P)=O
//
// Strategy: compute both general-add and doubling results,
// then CT-select the correct one based on whether H==0 / R==0.
// Also handles infinity via cmov at the end.

CTJacobianPoint point_add_complete(const CTJacobianPoint& p,
                                   const CTJacobianPoint& q) noexcept {
    // Normalize inputs to guarantee magnitude 1 (previous ops may leave M>1)
    FE52 X1 = p.x; X1.normalize_weak();
    FE52 Y1 = p.y; Y1.normalize_weak();
    FE52 Z1 = p.z; Z1.normalize_weak();
    FE52 X2 = q.x; X2.normalize_weak();
    FE52 Y2 = q.y; Y2.normalize_weak();
    FE52 Z2 = q.z; Z2.normalize_weak();

    // ── 1. General Jacobian addition ──
    FE52 z1z1 = Z1.square();                           // M=1
    FE52 z2z2 = Z2.square();                           // M=1
    FE52 u1   = X1 * z2z2;                             // M=1
    FE52 u2   = X2 * z1z1;                             // M=1
    FE52 s1   = (Y1 * Z2) * z2z2;                      // M=1
    FE52 s2   = (Y2 * Z1) * z1z1;                      // M=1

    FE52 h    = u2 + u1.negate(1);                      // H = U2 - U1; M=3
    FE52 r    = s2 + s1.negate(1);                      // R = S2 - S1; M=3

    // Detect special cases via CT masks
    std::uint64_t h_is_zero = fe52_is_zero(h);
    std::uint64_t r_is_zero = fe52_is_zero(r);
    std::uint64_t is_double  = h_is_zero & r_is_zero;
    std::uint64_t is_inverse = h_is_zero & ~r_is_zero;

    FE52 hh   = h.square();                            // M=1
    FE52 hhh  = h * hh;                                // M=1
    FE52 v    = u1 * hh;                               // M=1

    FE52 rr       = r.square();                         // M=1
    FE52 vv       = v + v;                              // M=2
    FE52 x3_add   = rr + hhh.negate(1) + vv.negate(2); // M=1+2+3=6
    FE52 vx3      = v + x3_add.negate(6);              // M=1+7=8
    FE52 y3_add   = (r * vx3) + (s1 * hhh).negate(1);  // M=1+2=3
    FE52 z3_add   = (Z1 * Z2) * h;                     // M=1

    // ── 2. Doubling of P (dbl-2007-a for a=0) ──
    FE52 A   = X1.square();                            // M=1
    FE52 B   = Y1.square();                            // M=1
    FE52 C   = B.square();                             // M=1

    FE52 xb  = X1 + B;                                // M=2
    FE52 AC  = A + C;                                  // M=2
    FE52 D   = xb.square() + AC.negate(2);             // M=1+3=4
    D = D + D;                                         // M=8

    FE52 E   = (A + A) + A;                            // M=3
    FE52 F   = E.square();                             // M=1

    FE52 DD  = D + D;                                  // M=16
    FE52 x3_dbl = F + DD.negate(16);                   // M=18

    FE52 C8  = C + C;                                  // M=2
    C8 = C8 + C8;                                      // M=4
    C8 = C8 + C8;                                      // M=8

    FE52 Dx  = D + x3_dbl.negate(18);                  // M=8+19=27
    FE52 y3_dbl = (E * Dx) + C8.negate(8);             // M=1+9=10

    FE52 yz_dbl = Y1 * Z1;
    FE52 z3_dbl = yz_dbl + yz_dbl;                     // M=2

    // ── 3. Select result via CT masks ──
    FE52 x3 = fe52_select(x3_dbl, x3_add, is_double);
    FE52 y3 = fe52_select(y3_dbl, y3_add, is_double);
    FE52 z3 = fe52_select(z3_dbl, z3_add, is_double);

    // If P == -Q → infinity (0:1:0)
    FE52 zero52 = FE52::zero();
    FE52 one52  = FE52::one();
    fe52_cmov(&x3, zero52, is_inverse);
    fe52_cmov(&y3, one52,  is_inverse);
    fe52_cmov(&z3, zero52, is_inverse);

    // If P is infinity → result = Q
    fe52_cmov(&x3, X2, p.infinity);
    fe52_cmov(&y3, Y2, p.infinity);
    fe52_cmov(&z3, Z2, p.infinity);

    // If Q is infinity → result = P
    fe52_cmov(&x3, X1, q.infinity);
    fe52_cmov(&y3, Y1, q.infinity);
    fe52_cmov(&z3, Z1, q.infinity);

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

// ─── Mixed Jacobian+Affine Complete Addition (a=0, 5×52) ─────────────────────
// P = (X1:Y1:Z1) Jacobian, Q = (x2, y2) affine (implied Z2=1).
// Z2=1 optimizations: no Z2²/Z2³, U1=X1, S1=Y1, Z3_add=Z1·H.

CTJacobianPoint point_add_mixed_complete(const CTJacobianPoint& p,
                                          const CTAffinePoint& q) noexcept {
    // Normalize inputs to guarantee magnitude 1
    FE52 X1 = p.x; X1.normalize_weak();
    FE52 Y1 = p.y; Y1.normalize_weak();
    FE52 Z1 = p.z; Z1.normalize_weak();
    FE52 x2 = q.x; x2.normalize_weak();
    FE52 y2 = q.y; y2.normalize_weak();

    // ── 1. General Jacobian+Affine addition (Z2=1) ──
    FE52 z1z1 = Z1.square();                          // M=1
    FE52 u2   = x2 * z1z1;                            // M=1
    FE52 z1cu = z1z1 * Z1;                             // M=1
    FE52 s2   = y2 * z1cu;                             // M=1

    FE52 h    = u2 + X1.negate(1);                     // H = U2 - X1; M=3
    FE52 r    = s2 + Y1.negate(1);                     // R = S2 - Y1; M=3

    // Detect special cases via CT masks
    std::uint64_t h_is_zero = fe52_is_zero(h);
    std::uint64_t r_is_zero = fe52_is_zero(r);
    std::uint64_t is_double  = h_is_zero & r_is_zero;
    std::uint64_t is_inverse = h_is_zero & ~r_is_zero;

    FE52 hh   = h.square();                           // M=1
    FE52 hhh  = h * hh;                               // M=1
    FE52 v    = X1 * hh;                              // M=1

    FE52 rr       = r.square();                        // M=1
    FE52 vv       = v + v;                             // M=2
    FE52 x3_add   = rr + hhh.negate(1) + vv.negate(2); // M=6
    FE52 vx3      = v + x3_add.negate(6);             // M=8
    FE52 y3_add   = (r * vx3) + (Y1 * hhh).negate(1); // M=3
    FE52 z3_add   = Z1 * h;                           // M=1

    // ── 2. Doubling of P (dbl-2007-a for a=0) ──
    FE52 A   = X1.square();                           // M=1
    FE52 B   = Y1.square();                           // M=1
    FE52 C   = B.square();                            // M=1

    FE52 xb  = X1 + B;                               // M=2
    FE52 AC  = A + C;                                 // M=2
    FE52 D   = xb.square() + AC.negate(2);            // M=4
    D = D + D;                                        // M=8

    FE52 E   = (A + A) + A;                           // M=3
    FE52 F   = E.square();                            // M=1

    FE52 DD  = D + D;                                 // M=16
    FE52 x3_dbl = F + DD.negate(16);                  // M=18

    FE52 C8  = C + C;                                 // M=2
    C8 = C8 + C8;                                     // M=4
    C8 = C8 + C8;                                     // M=8

    FE52 Dx  = D + x3_dbl.negate(18);                 // M=27
    FE52 y3_dbl = (E * Dx) + C8.negate(8);            // M=10
    FE52 yz_mix = Y1 * Z1;
    FE52 z3_dbl = yz_mix + yz_mix;                    // M=2

    // ── 3. Select result via CT masks ──
    FE52 x3 = fe52_select(x3_dbl, x3_add, is_double);
    FE52 y3 = fe52_select(y3_dbl, y3_add, is_double);
    FE52 z3 = fe52_select(z3_dbl, z3_add, is_double);

    // If P == -Q → infinity (0:1:0)
    FE52 zero52 = FE52::zero();
    FE52 one52  = FE52::one();
    fe52_cmov(&x3, zero52, is_inverse);
    fe52_cmov(&y3, one52,  is_inverse);
    fe52_cmov(&z3, zero52, is_inverse);

    // If P is infinity → result = Q (affine, so Z=1)
    fe52_cmov(&x3, x2, p.infinity);
    fe52_cmov(&y3, y2, p.infinity);
    fe52_cmov(&z3, one52, p.infinity);

    // If Q is infinity → result = P
    fe52_cmov(&x3, X1, q.infinity);
    fe52_cmov(&y3, Y1, q.infinity);
    fe52_cmov(&z3, Z1, q.infinity);

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

// ─── CT Point Doubling (5×52) ────────────────────────────────────────────────

CTJacobianPoint point_dbl(const CTJacobianPoint& p) noexcept {
    // Normalize inputs to guarantee magnitude 1
    FE52 X1 = p.x; X1.normalize_weak();
    FE52 Y1 = p.y; Y1.normalize_weak();
    FE52 Z1 = p.z; Z1.normalize_weak();

    FE52 A = X1.square();           // M=1
    FE52 B = Y1.square();           // M=1
    FE52 C = B.square();            // M=1

    FE52 xb = X1 + B;              // M=2
    FE52 AC = A + C;               // M=2
    FE52 D = xb.square() + AC.negate(2); // M=4
    D = D + D;                     // M=8

    FE52 E = (A + A) + A;          // M=3
    FE52 F = E.square();           // M=1

    FE52 DD = D + D;               // M=16
    FE52 x3 = F + DD.negate(16);   // M=18

    FE52 C8 = C + C;               // M=2
    C8 = C8 + C8;                  // M=4
    C8 = C8 + C8;                  // M=8

    FE52 Dx = D + x3.negate(18);   // M=27
    FE52 y3 = (E * Dx) + C8.negate(8); // M=10

    FE52 z3 = Y1 * Z1;            // M=1
    z3 = z3 + z3;                  // M=2

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

// ─── Batch N Doublings (5×52, no infinity) ───────────────────────────────────
// Performs n consecutive doublings in-place, normalizing only at the start.
// The doubling formula (dbl-2007-a for a=0) has self-stabilizing magnitudes:
// all negate() parameters depend on intermediate values (always M=1 from
// mul/sqr), NOT on input magnitude. So consecutive doublings are safe without
// per-step normalize_weak.
//
// Precondition: p is NOT infinity (caller ensures).
// Output: 2^n · p

namespace {

CTJacobianPoint point_dbl_n(const CTJacobianPoint& p, unsigned n) noexcept {
    FE52 X = p.x; X.normalize_weak();
    FE52 Y = p.y; Y.normalize_weak();
    FE52 Z = p.z; Z.normalize_weak();

    for (unsigned i = 0; i < n; ++i) {
        FE52 A = X.square();
        FE52 B = Y.square();
        FE52 C = B.square();

        FE52 xb = X + B;
        FE52 AC = A + C;
        FE52 D = xb.square() + AC.negate(2);
        D = D + D;

        FE52 E = (A + A) + A;
        FE52 F = E.square();

        FE52 DD = D + D;
        FE52 x_new = F + DD.negate(16);

        FE52 C8 = C + C; C8 = C8 + C8; C8 = C8 + C8;

        FE52 Dx = D + x_new.negate(18);
        FE52 y_new = (E * Dx) + C8.negate(8);

        FE52 yz = Y * Z;
        FE52 z_new = yz + yz;

        X = x_new;
        Y = y_new;
        Z = z_new;
    }

    CTJacobianPoint result;
    result.x = X;
    result.y = Y;
    result.z = Z;
    result.infinity = 0;
    return result;
}

} // anonymous namespace

// ─── Brier-Joye Unified Mixed Addition (Jacobian + Affine, a=0) ─────────────
// Cost: 7M + 5S (vs 9M+8S for complete formula) — ~40% cheaper
//
// Unified: handles both addition (a ≠ ±b) and doubling (a == b) in one path.
// Degenerates only when y1 == -y2 (M = 0), handled via alternate lambda cmov.
// Detects a = -b (result = infinity) via Z3 == 0.
//
// Precondition: b MUST NOT be infinity.
//   a may be infinity (handled via cmov at end → result = b).
//
// Based on: E. Brier, M. Joye "Weierstrass Elliptic Curves and Side-Channel
// Attacks" (PKC 2002). Formula from bitcoin-core/secp256k1 group_impl.h.

CTJacobianPoint point_add_mixed_unified(const CTJacobianPoint& a,
                                         const CTAffinePoint& b) noexcept {
    // Normalize a's coords to M=1 (b is already M=1 from table/precomp)
    FE52 X1 = a.x; X1.normalize_weak();
    FE52 Y1 = a.y; Y1.normalize_weak();
    FE52 Z1 = a.z; Z1.normalize_weak();

    // ── Shared intermediates ──
    FE52 zz = Z1.square();                          // Z1²       [1S]
    FE52 u1 = X1;                                   // U1 = X1 (Z2=1)
    FE52 u2 = b.x * zz;                             // U2 = x2·Z1²  [1M]
    FE52 s1 = Y1;                                   // S1 = Y1 (Z2=1)
    FE52 s2 = (b.y * zz) * Z1;                      // S2 = y2·Z1³   [2M]

    FE52 t_val = u1 + u2;                           // T = U1+U2; M=2
    FE52 m_val = s1 + s2;                            // M = S1+S2; M=2

    // R = T² - U1·U2
    FE52 rr = t_val.square();                        // T²; M=1      [1S]
    FE52 neg_u2 = u2.negate(1);                      // -U2; M=2
    FE52 tt = u1 * neg_u2;                           // -U1·U2; M=1  [1M]
    rr = rr + tt;                                    // R = T²-U1U2; M=2

    // ── Degenerate case: M≈0 (y1=-y2) ──
    // When M=0, lambda = R/M is undefined. Use alternate:
    //   lambda_alt = (y1-y2)/(x1-x2) = 2·S1/(U1-U2)
    std::uint64_t degen = fe52_is_zero(m_val);

    FE52 rr_alt = s1 + s1;                          // 2·S1; M=2
    FE52 m_alt  = u1 + neg_u2;                      // U1-U2; M=1+2=3

    // Select: if NOT degenerate, use main (rr, m_val)
    fe52_cmov(&rr_alt, rr, ~degen);
    fe52_cmov(&m_alt, m_val, ~degen);

    // ── Compute result ──
    FE52 n = m_alt.square();                         // Malt²; M=1   [1S]
    FE52 neg_t = t_val.negate(2);                    // -T; M=3
    FE52 q = neg_t * n;                              // Q=-T·Malt²; M=1 [1M]

    // n = Malt⁴ (normal) or M³·Malt=0 (degenerate, since M=0)
    // Key: either M==Malt (not degen) or M==0 (degen)
    // So: M³·Malt = degenerate ? 0 : Malt⁴
    // Compute Malt⁴ by squaring, cmov to M (≈0) when degenerate
    n = n.square();                                  // Malt⁴; M=1   [1S]
    fe52_cmov(&n, m_val, degen);                     // if degen: n=M≈0

    // X3 = Ralt² + Q
    FE52 x3 = rr_alt.square() + q;                  // M=1+1=2     [1S]

    // Z3 = Malt · Z1
    FE52 z3 = m_alt * Z1;                           // M=1         [1M]

    // Y3 = -(Ralt·(2·X3+Q) + n) / 2
    FE52 x3_2 = x3 + x3;                            // 2·X3; M=4
    FE52 tq = x3_2 + q;                             // 2·X3+Q; M=4+1=5
    FE52 y3_pre = (rr_alt * tq) + n;                // M=1+2=3     [1M=7th]
    FE52 y3 = y3_pre.negate(3).half();               // Y3; half normalizes

    // ── Handle a=infinity: replace result with (b.x, b.y, 1) ──
    FE52 one52 = FE52::one();
    fe52_cmov(&x3, b.x, a.infinity);
    fe52_cmov(&y3, b.y, a.infinity);
    fe52_cmov(&z3, one52, a.infinity);

    // Infinity detection: Z3==0 means a=-b (result should be infinity)
    std::uint64_t result_inf = fe52_is_zero(z3);

    CTJacobianPoint result;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    result.infinity = result_inf;
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CT GLV Endomorphism — Helpers & Decomposition
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
static std::uint64_t ct_scalar_is_high(const Scalar& s) noexcept {
    static constexpr std::uint64_t HALF_N[4] = {
        0xDFE92F46681B20A0ULL,
        0x5D576E7357A4501DULL,
        0xFFFFFFFFFFFFFFFFULL,
        0x7FFFFFFFFFFFFFFFULL
    };
    std::uint64_t tmp[4];
    std::uint64_t borrow = local_sub256(tmp, HALF_N, s.limbs().data());
    return is_nonzero_mask(borrow);
}

// ─── CT 256×256→512 multiply, shift >>384 with rounding ─────────────────────
static std::array<std::uint64_t, 4> ct_mul_shift_384(
    const std::array<std::uint64_t, 4>& a,
    const std::array<std::uint64_t, 4>& b) noexcept
{
    std::uint64_t prod[8] = {};
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

    std::array<std::uint64_t, 4> result{};
    result[0] = prod[6];
    result[1] = prod[7];

    std::uint64_t round = prod[5] >> 63;
    std::uint64_t old = result[0];
    result[0] += round;
    std::uint64_t carry = static_cast<std::uint64_t>(result[0] < old);
    result[1] += carry;

    return result;
}

// ─── CT scalar multiplication mod n ──────────────────────────────────────────
static Scalar ct_scalar_mul_mod(const Scalar& a, const Scalar& b) noexcept {
    static constexpr std::uint64_t ORDER[4] = {
        0xBFD25E8CD0364141ULL,
        0xBAAEDCE6AF48A03BULL,
        0xFFFFFFFFFFFFFFFEULL,
        0xFFFFFFFFFFFFFFFFULL
    };
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

    // Step 2: Barrett approximation
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

    // Step 3: CT conditional subtract
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
    std::uint64_t need_sub1 = is_nonzero_mask(r4) | is_zero_mask(bw1 & is_zero_mask(r4));
    cmov256(r, r_sub1, need_sub1);
    r4 = ct_select(r4_sub1, r4, need_sub1);

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

// ─── β (beta) as FE52 — cube root of unity mod p ────────────────────────────
static const FE52& get_beta_fe52() noexcept {
    static const FE52 beta = FE52::from_fe(
        FieldElement::from_bytes(secp256k1::fast::glv_constants::BETA));
    return beta;
}

// ─── GLV lattice constants ───────────────────────────────────────────────────
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

// ─── CT GLV Endomorphism (5×52) ──────────────────────────────────────────────

CTJacobianPoint point_endomorphism(const CTJacobianPoint& p) noexcept {
    CTJacobianPoint r;
    r.x = p.x * get_beta_fe52();
    r.y = p.y;
    r.z = p.z;
    r.infinity = p.infinity;
    return r;
}

CTAffinePoint affine_endomorphism(const CTAffinePoint& p) noexcept {
    CTAffinePoint r;
    r.x = p.x * get_beta_fe52();
    r.y = p.y;
    r.infinity = p.infinity;
    return r;
}

CTAffinePoint affine_neg(const CTAffinePoint& p) noexcept {
    CTAffinePoint r;
    r.x = p.x;
    FE52 yn = p.y; yn.normalize_weak();
    r.y = yn.negate(1);
    r.infinity = p.infinity;
    return r;
}

// ─── CT GLV Decomposition ────────────────────────────────────────────────────

CTGLVDecomposition ct_glv_decompose(const Scalar& k) noexcept {
    static const Scalar minus_b1 = Scalar::from_bytes(kMinusB1Bytes);
    static const Scalar minus_b2 = Scalar::from_bytes(kMinusB2Bytes);
    static const Scalar lambda   = Scalar::from_bytes(kLambdaBytes);

    auto k_limbs = k.limbs();
    auto c1_limbs = ct_mul_shift_384({k_limbs[0], k_limbs[1], k_limbs[2], k_limbs[3]}, kG1);
    auto c2_limbs = ct_mul_shift_384({k_limbs[0], k_limbs[1], k_limbs[2], k_limbs[3]}, kG2);

    Scalar c1 = Scalar::from_limbs(c1_limbs);
    Scalar c2 = Scalar::from_limbs(c2_limbs);

    Scalar k2_part1 = ct_scalar_mul_mod(c1, minus_b1);
    Scalar k2_part2 = ct_scalar_mul_mod(c2, minus_b2);
    Scalar k2_mod   = scalar_add(k2_part1, k2_part2);

    std::uint64_t k2_high = ct_scalar_is_high(k2_mod);
    Scalar k2_abs = scalar_cneg(k2_mod, k2_high);

    Scalar k2_signed = scalar_cneg(k2_abs, k2_high);

    Scalar lambda_k2 = ct_scalar_mul_mod(lambda, k2_signed);
    Scalar k1_mod = scalar_sub(k, lambda_k2);

    std::uint64_t k1_high = ct_scalar_is_high(k1_mod);
    Scalar k1_abs = scalar_cneg(k1_mod, k1_high);

    CTGLVDecomposition result;
    result.k1     = k1_abs;
    result.k2     = k2_abs;
    result.k1_neg = k1_high;
    result.k2_neg = k2_high;
    return result;
}

// ─── CT GLV Scalar Multiplication — Hamburg Signed-Digit (GROUP_SIZE=5) ──────
// Combines Hamburg's signed-digit comb (ePrint 2012/309 §3.3) with GLV.
//
// Algorithm:
//   1. s = (k + K) / 2 mod n, where K = (2^BITS-2^129-1)*(1+lambda) mod n
//   2. Split s into (s1, s2) via GLV
//   3. v1 = s1 + 2^128,  v2 = s2 + 2^128  (non-negative, fits in 129 bits)
//   4. Process GROUPS groups of GROUP_SIZE bits from v1, v2 (high to low)
//   5. Each group's bits encode a signed odd digit → table lookup
//
// GROUP_SIZE=5: TABLE_SIZE=16 odd multiples, GROUPS=26, BITS=130.
// Cost: 125 doublings + 52 additions (ALL real, no digit=0 waste).
// Table lookup: 26 groups × 2 tables × 15 cmov = 780 cmov iterations.

Point scalar_mul(const Point& p, const Scalar& k) noexcept {
    constexpr unsigned GROUP_SIZE = 5;
    constexpr unsigned TABLE_SIZE = 1u << (GROUP_SIZE - 1);  // 16 odd multiples
    constexpr unsigned GROUPS = 26;   // ceil(129/5)
    // BITS = GROUPS * GROUP_SIZE = 130 (used for K derivation)

    // K = (2^BITS - 2^129 - 1) * (1 + lambda) mod n
    // For GROUP_SIZE=5, BITS=130: K = (2^130 - 2^129 - 1)*(1+lambda) = (2^129 - 1)*(1+lambda)
    // Precomputed constant (same as libsecp256k1 ECMULT_CONST_BITS=130):
    static const Scalar K_CONST = Scalar::from_limbs({
        0xb5c2c1dcde9798d9ULL, 0x589ae84826ba29e4ULL,
        0xc2bdd6bf7c118d6bULL, 0xa4e88a7dcb13034eULL
    });

    // Offset to make s1, s2 non-negative: 2^128
    static const Scalar S_OFFSET = Scalar::from_limbs({0, 0, 1, 0});

    // ── 1. Compute v1, v2 via Hamburg transform + GLV split ──────────────
    Scalar s = scalar_add(k, K_CONST);
    s = scalar_half(s);

    // GLV split: s = s1 + s2*lambda  (|s1|, |s2| < 2^128)
    auto [k1_abs, k2_abs, k1_neg, k2_neg] = ct_glv_decompose(s);

    // We need the SIGNED values for the Hamburg transform
    Scalar s1 = scalar_cneg(k1_abs, k1_neg);
    Scalar s2 = scalar_cneg(k2_abs, k2_neg);

    // v1 = s1 + 2^128,  v2 = s2 + 2^128
    Scalar v1 = scalar_add(s1, S_OFFSET);
    Scalar v2 = scalar_add(s2, S_OFFSET);

    // ── 2. Build odd-multiples table via batch inversion ─────────────────
    // Table stores [1·P, 3·P, 5·P, ..., 31·P] in affine
    CTAffinePoint pre_a[TABLE_SIZE];
    CTAffinePoint pre_a_lam[TABLE_SIZE];

    // Compute odd multiples in Jacobian: 1P, 3P, 5P, ..., 31P
    Point jac[TABLE_SIZE];
    jac[0] = p;  // 1·P
    Point p2 = p;
    p2.dbl_inplace();  // 2·P
    for (std::size_t i = 1; i < TABLE_SIZE; ++i) {
        jac[i] = jac[i - 1].add(p2);  // (2i-1)P + 2P = (2i+1)P
    }

    // Batch-invert Z coordinates
    FE52 zs[TABLE_SIZE], z_invs[TABLE_SIZE];
    for (std::size_t i = 0; i < TABLE_SIZE; ++i) {
        zs[i] = jac[i].Z52();
    }
    fe52_batch_inverse(z_invs, zs, TABLE_SIZE);

    // Convert to affine
    const FE52& beta = get_beta_fe52();
    for (std::size_t i = 0; i < TABLE_SIZE; ++i) {
        FE52 zinv2 = z_invs[i].square();
        FE52 zinv3 = zinv2 * z_invs[i];
        pre_a[i].x = jac[i].X52() * zinv2;
        pre_a[i].y = jac[i].Y52() * zinv3;
        pre_a[i].infinity = 0;
        // Lambda table: φ(x,y) = (β·x, y)
        pre_a_lam[i].x = pre_a[i].x * beta;
        pre_a_lam[i].y = pre_a[i].y;
        pre_a_lam[i].infinity = 0;
    }

    // NOTE: No table Y-negation needed.
    // Hamburg signed-digit encoding bakes the sign into v1/v2 (via the offset
    // and signed GLV decomposition). The lookup_signed function handles
    // conditional Y-negate at runtime based on each window's top bit.

    SECP256K1_DECLASSIFY(pre_a, sizeof(pre_a));
    SECP256K1_DECLASSIFY(pre_a_lam, sizeof(pre_a_lam));

    // ── 3. Main loop: process GROUPS groups of GROUP_SIZE bits ───────────
    CTJacobianPoint R = CTJacobianPoint::make_infinity();

    for (int group = static_cast<int>(GROUPS) - 1; group >= 0; --group) {
        std::uint64_t bits1 = scalar_window(v1, static_cast<std::size_t>(group) * GROUP_SIZE, GROUP_SIZE);
        std::uint64_t bits2 = scalar_window(v2, static_cast<std::size_t>(group) * GROUP_SIZE, GROUP_SIZE);

        CTAffinePoint t;

        // Lookup from pre_a table using signed-digit encoding
        t = affine_table_lookup_signed(pre_a, TABLE_SIZE, bits1, GROUP_SIZE);

        if (group == static_cast<int>(GROUPS) - 1) {
            // First iteration: set R directly (Jacobian from affine)
            R.x = t.x;
            R.y = t.y;
            R.z = FE52::one();
            R.infinity = 0;
        } else {
            // Shift result up by GROUP_SIZE doublings.
            // Uses batched doubling: normalize once, no per-step normalize_weak.
            // R is never infinity here (first group already set it).
            R = point_dbl_n(R, GROUP_SIZE);

            // Add first table entry (guaranteed non-infinity)
            R = point_add_mixed_unified(R, t);
        }

        // Lookup from lambda table and add (guaranteed non-infinity)
        t = affine_table_lookup_signed(pre_a_lam, TABLE_SIZE, bits2, GROUP_SIZE);
        R = point_add_mixed_unified(R, t);
    }

    Point result = R.to_point();
    SECP256K1_DECLASSIFY(&result, sizeof(result));
    return result;
}

// ─── CT Generator Multiplication (5×52) ──────────────────────────────────────
// Uses Hamburg signed-digit encoding (Mike Hamburg, "Fast and compact
// elliptic-curve cryptography", IACR ePrint 2012/309, Section 3.3).
//
// Key insight: interpret scalar bits as signs (+1/-1) instead of (1/0).
// Given l-bit value v, define C_l(v, A) = sum((2*v[i]-1)*2^i*A, i=0..l-1).
// Then C_l(v, A) = (2*v + 1 - 2^l)*A.
// So to compute k*G: set v = (k + 2^256 - 1)/2 mod n, then C_256(v, G) = k*G.
//
// Every GROUP_SIZE-bit window of v, interpreted as a signed digit, is ODD:
// it's always in {±1, ±3, ..., ±(2^GROUP_SIZE - 1)}.
// → Table stores only odd multiples (half the entries!)
// → No digit-0 case → no cmov skip → every addition contributes.
//
// Table: GEN_WINDOWS × GEN_SIGNED_TABLE_SIZE affine points.
// Lookup: GEN_SIGNED_TABLE_SIZE-1 cmov iterations (vs 2×GEN_SIGNED_TABLE_SIZE-1 before).

namespace {

constexpr unsigned GEN_W = 4;
constexpr unsigned GEN_WINDOWS = 64;   // 256 / GEN_W
constexpr std::size_t GEN_SIGNED_TABLE_SIZE = 1u << (GEN_W - 1);  // 8 odd multiples

// Hamburg constant: K_gen = (2^256 - 1) mod n
// 2^256 mod n = 2^256 - n = {0x402DA1732FC9BEBF, 0x4551231950B75FC4, 1, 0}
// K_gen = (2^256 - 1) mod n = (2^256 mod n) - 1
static constexpr std::uint64_t K_GEN[4] = {
    0x402DA1732FC9BEBEULL,
    0x4551231950B75FC4ULL,
    0x0000000000000001ULL,
    0x0000000000000000ULL
};

struct alignas(64) GenPrecompTable {
    CTAffinePoint entries[GEN_WINDOWS][GEN_SIGNED_TABLE_SIZE];
    bool initialized = false;
};

static GenPrecompTable g_gen_table;
static std::once_flag g_gen_table_once;

void build_gen_table() noexcept {
    Point G = Point::generator();
    Point base = G;

    for (unsigned w = 0; w < GEN_WINDOWS; ++w) {
        // Build ODD multiples: [1·base, 3·base, 5·base, ..., 15·base]
        // Using: (2i+1)·base for i=0..7
        Point jac[GEN_SIGNED_TABLE_SIZE];
        jac[0] = base;  // 1·base

        Point doubled = base;
        doubled.dbl_inplace();  // 2·base

        for (std::size_t j = 1; j < GEN_SIGNED_TABLE_SIZE; ++j) {
            jac[j] = jac[j - 1].add(doubled);  // (2j-1)·base + 2·base = (2j+1)·base
        }

        // Batch-invert Z coordinates
        FE52 zs[GEN_SIGNED_TABLE_SIZE], z_invs[GEN_SIGNED_TABLE_SIZE];
        for (std::size_t j = 0; j < GEN_SIGNED_TABLE_SIZE; ++j) {
            zs[j] = jac[j].Z52();
        }
        fe52_batch_inverse(z_invs, zs, GEN_SIGNED_TABLE_SIZE);

        // Convert to affine
        for (std::size_t j = 0; j < GEN_SIGNED_TABLE_SIZE; ++j) {
            FE52 zinv2 = z_invs[j].square();
            FE52 zinv3 = zinv2 * z_invs[j];
            g_gen_table.entries[w][j].x = jac[j].X52() * zinv2;
            g_gen_table.entries[w][j].y = jac[j].Y52() * zinv3;
            g_gen_table.entries[w][j].infinity = 0;
        }

        // Advance base by 2^GEN_W for next window
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
    init_generator_table();

    // ── Hamburg scalar transform ──────────────────────────────────────────
    // v = (k + K_gen) / 2 mod n
    // This ensures that interpreting v's bits as signs (+1/-1) gives k*G.
    static const Scalar K_gen_scalar = Scalar::from_limbs(
        {K_GEN[0], K_GEN[1], K_GEN[2], K_GEN[3]});

    Scalar s = scalar_add(k, K_gen_scalar);
    Scalar v = scalar_half(s);

    CTJacobianPoint R;

    // ── Main loop: sum signed-digit lookups (no doublings) ───────────────
    // First window: set R directly (avoids unified add on infinity)
    {
        std::uint64_t digit0 = scalar_window(v, 0, GEN_W);
        CTAffinePoint T0 = affine_table_lookup_signed(
            g_gen_table.entries[0], GEN_SIGNED_TABLE_SIZE, digit0, GEN_W);
        R.x = T0.x;
        R.y = T0.y;
        R.z = FE52::one();
        R.infinity = 0;
    }

    // Remaining windows: unified add (T is guaranteed non-infinity)
    for (unsigned i = 1; i < GEN_WINDOWS; ++i) {
        std::uint64_t digit = scalar_window(v, static_cast<std::size_t>(i) * GEN_W, GEN_W);

        CTAffinePoint T = affine_table_lookup_signed(
            g_gen_table.entries[i], GEN_SIGNED_TABLE_SIZE, digit, GEN_W);

        R = point_add_mixed_unified(R, T);
    }

    Point result = R.to_point();
    SECP256K1_DECLASSIFY(&result, sizeof(result));
    return result;
}

// ─── CT Curve Check (uses 4×64 FieldElement at API boundary) ─────────────────

std::uint64_t point_is_on_curve(const Point& p) noexcept {
    if (p.is_infinity()) {
        return ~static_cast<std::uint64_t>(0);
    }

    FieldElement x = p.x();
    FieldElement y = p.y();

    FieldElement y2 = field_sqr(y);
    FieldElement x2 = field_sqr(x);
    FieldElement x3 = field_mul(x, x2);
    FieldElement rhs = field_add(x3, B7);

    return field_eq(y2, rhs);
}

// ─── CT Point Equality (uses 4×64 at API boundary) ──────────────────────────

std::uint64_t point_eq(const Point& a, const Point& b) noexcept {
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
