// ============================================================================
// Constant-Time Point Arithmetic -- Implementation (5x52 Optimized)
// ============================================================================
// Complete addition formula + CT scalar multiplication for secp256k1.
//
// INTERNAL REPRESENTATION: FieldElement52 (5x52-bit limbs with lazy reduction)
// This eliminates the function-call overhead of ct::field_* wrappers and
// leverages the ~3.6x faster force-inlined 5x52 multiply/square kernels.
//
// Conversion to/from 4x64 FieldElement occurs only at API boundaries
// (from_point / to_point). All internal arithmetic stays in 5x52.
//
// Magnitude tracking: mul/sqr produce magnitude 1 output.
// Lazy adds accumulate magnitude. Max magnitude in point ops ~= 30,
// well within the 4096 headroom (12 bits per limb).
//
// Complete addition:
//   Handles P+Q, P+P, P+O, O+Q, P+(-P) in a single branchless codepath.
//   Based on: "Complete addition formulas for prime order elliptic curves"
//   (Renes, Costello, Bathalter 2016), adapted for a=0 (secp256k1).
//
// CT scalar multiplication (GLV-OPTIMIZED):
//   Uses GLV endomorphism phi(x,y)=(beta*x,y) to split 256-bit scalar
//   into two 128-bit halves: k = k1 + k2*lambda (mod n).
//   Strauss interleaving: 32 windows x (4 dbl + 2 mixed_add).
//   Total: 128 doublings + 64 mixed_complete additions.
//
// CT generator multiplication (comb):
//   COMB_BLOCKS=11 blocks x 2^(COMB_TEETH-1)=32 entries = 352 affine points,
//   with COMB_SPACING=4, so COMB_BITS = 10*6*4 + 4*4 = 256 exactly.
//   Runtime: 44 CT lookups (32 entries each) + 44 mixed additions + 3 doublings.
//   The doublings are between spacing steps, not per block.
// ============================================================================

#include "secp256k1/config.hpp"  // SECP256K1_FAST_52BIT, SECP256K1_INLINE
#include "secp256k1/ct/point.hpp"
#include "secp256k1/detail/secure_erase.hpp"
#include "secp256k1/ct/field.hpp"
#include "secp256k1/ct/scalar.hpp"
#include "secp256k1/ct/ops.hpp"
#include "secp256k1/field_52.hpp"
#include "secp256k1/glv.hpp"

#include <mutex>

// AVX2 vectorized CT table lookup (x86-64 with -march=native)
#if defined(__x86_64__) && defined(__AVX2__)
#include <immintrin.h>
#define SECP256K1_CT_AVX2 1
#endif

namespace secp256k1::ct {

// --- secp256k1 curve constant b = 7 (4x64 for API-boundary functions) --------
// ESP32/bare-metal: avoid runtime global ctor (triggers static-init mutex crash
// before FreeRTOS scheduler starts). Use inline helper instead.
#if defined(SECP256K1_PLATFORM_ESP32) || defined(ESP_PLATFORM)
static inline FieldElement get_b7() { return FieldElement::from_uint64(7); }
#define B7 get_b7()
#else
static const FieldElement B7 = FieldElement::from_uint64(7);
#endif

// ============================================================================
// 5x52 Optimized Path (requires __int128 / SECP256K1_FAST_52BIT)
// ============================================================================
#if defined(SECP256K1_FAST_52BIT)
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif

// --- Type aliases ------------------------------------------------------------
using FE52 = secp256k1::fast::FieldElement52;

// --- 5x52 CT Helper Functions ------------------------------------------------
// All operations on FieldElement52 are inherently constant-time:
//   - mul/sqr: fixed __int128 multiply chain -> MULX (data-independent)
//   - add: 5 plain uint64_t additions (no branches)
//   - negate: (m+1)*p - a (no branches)
//   - normalize: branchless mask-based conditional subtract
// Value barriers are applied in mask generation (is_zero_mask etc.)

namespace /* FE52 CT helpers */ {

// CT conditional move for 5x52 field element
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


// Cheaper CT zero check: normalize_weak + overflow reduce + dual representation
// check. Avoids the full branchless conditional-subtract of p.
// Matches libsecp256k1's secp256k1_fe_normalizes_to_zero() approach.
// Returns all-ones if the value normalizes to zero, else 0.
inline std::uint64_t fe52_normalizes_to_zero(const FE52& a) noexcept {
    constexpr std::uint64_t M52 = 0x000FFFFFFFFFFFFFULL;
    constexpr std::uint64_t M48 = 0x0000FFFFFFFFFFFFULL;

    std::uint64_t t0 = a.n[0], t1 = a.n[1], t2 = a.n[2], t3 = a.n[3], t4 = a.n[4];

    // First pass: carry propagation (normalize_weak)
    t1 += (t0 >> 52); t0 &= M52;
    t2 += (t1 >> 52); t1 &= M52;
    t3 += (t2 >> 52); t2 &= M52;
    t4 += (t3 >> 52); t3 &= M52;

    // Overflow reduction: top bits of t4 wrap around via field modulus
    std::uint64_t const x = t4 >> 48;
    t4 &= M48;
    t0 += x * 0x1000003D1ULL;

    // Second carry propagation
    t1 += (t0 >> 52); t0 &= M52;
    t2 += (t1 >> 52); t1 &= M52;
    t3 += (t2 >> 52); t2 &= M52;
    t4 += (t3 >> 52); t3 &= M52;

    // After two passes, zero is represented as either [0,0,0,0,0] or p.
    // Check both representations constant-time.
    // p in 5x52: {0xFFFFEFFFFFC2F, 0xFFFFFFFFFFFFF, 0xFFFFFFFFFFFFF, 0xFFFFFFFFFFFFF, 0xFFFFFFFFFFFF}
    std::uint64_t const z0 = t0 | t1 | t2 | t3 | t4;
    std::uint64_t const z1 = (t0 ^ 0x000FFFFEFFFFFC2FULL) | (t1 ^ M52) |
                       (t2 ^ M52) | (t3 ^ M52) | (t4 ^ M48);
    return is_zero_mask(z0) | is_zero_mask(z1);
}

// --- Variable-time Jacobian + Affine Addition (for table build) --------------
// NOT constant-time: used only for table precomputation where the POINT (not
// scalar) is public. Standard formula: 3S + 8M per addition vs 5S + 12M for
// Jac+Jac. Caller ensures no degenerate cases (P != +/-Q, P != inf, Q != inf).
//
// Stores the Jacobian result in (rx, ry, rz). FE52-native throughout.
// Output magnitudes: x M=7, y M=3, z M=1 (self-stabilizing -- safe to chain
// without normalize_weak). bx, by must be M=1 (from mul/sqr).
// Also outputs the Z-ratio (h) for global-Z normalization.
// The Z-ratio satisfies: result.z = a.z * zr_out.
struct JacFE52 { FE52 x, y, z; };

inline JacFE52 jac_add_ge_var_zr(const JacFE52& a,
                                  const FE52& bx, const FE52& by,
                                  FE52* zr_out) noexcept {
    FE52 const z1sq = a.z.square();
    FE52 const u2   = bx * z1sq;
    FE52 const z1cu = z1sq * a.z;
    FE52 const s2   = by * z1cu;

    FE52 const h  = u2 + a.x.negate(7);
    FE52 const r  = s2 + a.y.negate(3);

    FE52 const h2  = h.square();
    FE52 const h3  = h * h2;
    FE52 const u1h2 = a.x * h2;

    FE52 const r2  = r.square();
    FE52 const x3  = r2 + h3.negate(1) + u1h2.negate(1) + u1h2.negate(1);

    FE52 const diff = u1h2 + x3.negate(7);
    FE52 const y3  = (r * diff) + (a.y * h3).negate(1);

    FE52 const z3  = a.z * h;

    *zr_out = h;  // Z-ratio: result.z = a.z * h
    return {x3, y3, z3};
}

#if defined(REPSEARCH_COZ_TABLE) && REPSEARCH_COZ_TABLE == 1
// --- co-Z table construction (EXPERIMENT ONLY) -------------------------------
// experiments/representation_search. Not compiled in a default build; selected
// by -DREPSEARCH_COZ_TABLE=1. Same transformation already applied to
// build_glv52_table_zr in point.cpp, here on the ct:: track's table.
//
// jac_add_ge_var_zr above spends four of its eleven heavy operations bridging
// two different Z values -- z1sq, bx*z1sq, z1cu, by*z1cu. If both operands
// already share a Z, that bridge does not exist: 5M+2S instead of 8M+3S.
//
// Verified before it was written: the Python model in
// experiments/representation_search/repsearch/tablebuild.py builds the table
// both ways and checks they denote the same points against repeated affine
// addition, over table sizes 2..32 and 8 curve points.

// DBLU: double a Jacobian point AND hand back P re-expressed on the new Z.
// The co-Z copy of P is free: D = 4*X*Y^2 and 8*Y^4 are exactly X and Y scaled
// by (2Y)^2 and (2Y)^3, and the doubling computes both already. Only the new Z
// picks up a factor Z, so Z_new = 2*Y*Z. 2M+5S.
inline void jac52_dblu_ct(const FE52& x, const FE52& y, const FE52& zin,
                          FE52& out_dx, FE52& out_dy,
                          FE52& out_px, FE52& out_py,
                          FE52& out_z) noexcept {
    FE52 const A = x.square();
    FE52 const B = y.square();
    FE52 const C = B.square();

    FE52 t = x + B;
    t = t.square();
    t.add_assign(A.negate(1));
    t.add_assign(C.negate(1));
    FE52 D = t; D.add_assign(t);            // D = 4XB = X on the new Z

    FE52 E = A; E.mul_int_assign(3);
    FE52 const F = E.square();

    FE52 twoD = D; twoD.add_assign(D);
    FE52 X2 = F; X2.add_assign(twoD.negate(20));

    FE52 dmx = D; dmx.add_assign(X2.negate(22));
    FE52 Y2 = E * dmx;
    FE52 eightC = C; eightC.mul_int_assign(8);   // 8Y^4 = Y on the new Z
    Y2.add_assign(eightC.negate(8));

    FE52 Z = y; Z.add_assign(y);
    Z.mul_assign(zin);                       // Z_new = 2*Y*Z

    // One normalize per table build, so every magnitude downstream starts clean.
    X2.normalize_weak(); Y2.normalize_weak();
    D.normalize_weak();  eightC.normalize_weak(); Z.normalize_weak();

    out_dx = X2; out_dy = Y2;      // 2P
    out_px = D;  out_py = eightC;  // P, on the same Z
    out_z  = Z;
}

// ZADDU: (P, Q) sharing Z -> (P+Q, P) both on the new Z, plus the Z-ratio.
//
// OPERAND ORDER IS LOAD-BEARING: this returns its FIRST operand on the new Z.
// The caller's accumulator is replaced by the sum anyway, so the operand that
// must survive is the constant 2P -- pass it FIRST. The other order still
// produces a correct table and costs 3M+1S per step to rescale, which no
// correctness test would catch.
//
// x1/y1 are always magnitude 1 here: from the normalized DBLU on the first
// iteration, and from B = x1*A (a multiply output) on every one after, so
// negate(1) is exact rather than merely safe.
inline bool jac52_zaddu_ct(FE52& x1, FE52& y1,
                           const FE52& x2, const FE52& y2,
                           FE52& z,
                           FE52& sum_x, FE52& sum_y, FE52& zr_out) noexcept {
    FE52 dx = x2; dx.add_assign(x1.negate(1));
    if (dx.normalizes_to_zero_var()) return false;   // X1 == X2: ZADDU undefined
    FE52 dy = y2; dy.add_assign(y1.negate(1));

    FE52 const A = dx.square();
    FE52 const B = x1 * A;                 // X1 on the new Z
    FE52 const C = x2 * A;
    FE52 const D = dy.square();

    FE52 x3 = D;
    x3.add_assign(B.negate(1));
    x3.add_assign(C.negate(1));            // X3 = D - B - C, mag 5

    FE52 cb = C; cb.add_assign(B.negate(1));
    FE52 const y1cb = y1 * cb;             // Y1 on the new Z

    FE52 bx3 = B; bx3.add_assign(x3.negate(5));
    FE52 y3 = dy * bx3;
    y3.add_assign(y1cb.negate(1));         // Y3 = dy(B - X3) - Y1cb, mag 3

    zr_out = dx;                           // Z_new / Z_old
    z.mul_assign(dx);

    sum_x = x3; sum_y = y3;
    x1 = B; y1 = y1cb;                     // P on the new Z
    return true;
}
#endif  // REPSEARCH_COZ_TABLE

// --- FE52 Field Inversion ---------------------------------------------------
// Delegates to FieldElement52::inverse() which uses the optimal Fermat
// addition chain (253S + 14M).  When SECP256K1_HYBRID_4X64_ACTIVE is
// defined, the class method runs the entire chain in 4x64 MULX/ADCX/ADOX
// assembly with only one FE52<->4x64 boundary conversion at each end.
// Net saving: ~538 ns per call (6 ns boundary vs ~2 ns/op x 269 ops).

inline FE52 fe52_inverse(const FE52& a) noexcept {
    return a.inverse();
}

// --- FE52 Batch Inversion (Montgomery's trick) ------------------------------
// Computes z_inv[i] = z[i]^{-1} mod p for all i in [0, n).
// Cost: 1 inversion + 3(n-1) multiplications (vs n inversions).
//
// PRECONDITION: All z[i] MUST be non-zero. This is a CT function —
// adding variable-time zero-checks would break constant-time guarantees.
// Callers ensure this by excluding infinity entries before calling.
inline void fe52_batch_inverse(FE52* z_inv, const FE52* z, std::size_t n) noexcept {
    if (n == 0) return;
    if (n == 1) {
        z_inv[0] = fe52_inverse(z[0]);
        return;
    }

    // Forward accumulation: use z_inv[] as scratch for partial products
    //   z_inv[i] = z[0] * z[1] * ... * z[i]
    z_inv[0] = z[0];
    for (std::size_t i = 1; i < n; ++i) {
        z_inv[i] = z_inv[i - 1] * z[i];   // M=1 after mul
    }

    // Single inversion of the accumulated product -- native FE52 (no 4x64 detour)
    FE52 acc = fe52_inverse(z_inv[n - 1]);

    // Backward propagation: extract individual inverses
    for (std::size_t i = n - 1; i > 0; --i) {
        z_inv[i] = z_inv[i - 1] * acc;   // z_inv[i] = (z[0]..z[i-1]) * (z[0]..z[i])^{-1} = z[i]^{-1}
        acc = acc * z[i];                 // acc = (z[0]..z[i-1])^{-1}
    }
    z_inv[0] = acc;
}

} // anonymous namespace (FE52 CT helpers)


// --- CTJacobianPoint helpers -------------------------------------------------

CTJacobianPoint CTJacobianPoint::from_point(const Point& p) noexcept {
    CTJacobianPoint r;
    if (p.is_infinity()) {
        r = make_infinity();
    } else {
        // Direct FE52 access: avoids FE52->FE(4x64)->FE52 double conversion
        r.x = p.X52();
        r.y = p.Y52();
        r.z = p.Z52();
        r.infinity = 0;
    }
    return r;
}

Point CTJacobianPoint::to_point() const noexcept {
    if (infinity != 0) {
        return Point::infinity();
    }
    // Defensive: z==0 implies infinity even if flag wasn't set (e.g. 0*G)
    if (z.is_zero()) {
        return Point::infinity();
    }
    // Direct FE52 construction: avoids 3x to_fe + Point ctor 3x from_fe round-trip
    return Point::from_jacobian52(x, y, z, false);
}

CTJacobianPoint CTJacobianPoint::make_infinity() noexcept {
    CTJacobianPoint r;
    r.x = FE52::zero();
    r.y = FE52::one();
    r.z = FE52::zero();
    r.infinity = ~static_cast<std::uint64_t>(0);
    return r;
}

// --- CT Conditional Operations on Points -------------------------------------

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
        std::uint64_t const mask = eq_mask(static_cast<std::uint64_t>(i),
                                     static_cast<std::uint64_t>(index));
        point_cmov(&result, table[i], mask);
    }
    return result;
}

// --- CT Conditional Operations on Affine Points ------------------------------

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
        std::uint64_t const mask = eq_mask(static_cast<std::uint64_t>(i),
                                     static_cast<std::uint64_t>(index));
        affine_cmov(&result, table[i], mask);
    }
    return result;
}

// --- Signed-Digit Affine Table Lookup (Hamburg encoding) ---------------------
// Table stores odd multiples: [1*P, 3*P, 5*P, ..., (2^group_size - 1)*P]
// (table_size = 2^(group_size-1) entries)
//
// n is a group_size-bit value. Its bits are interpreted as signs of powers:
//   value = sum((2*bit[i]-1) * 2^i, i=0..group_size-1)
// This always yields an ODD number in [-(2^group_size-1), (2^group_size-1)].
//
// The top bit selects sign (0 = negative), remaining bits encode the index.
// Result is NEVER infinity -- the signed-digit transform guarantees this.

CTAffinePoint affine_table_lookup_signed(const CTAffinePoint* table,
                                          std::size_t table_size,
                                          std::uint64_t n,
                                          unsigned group_size) noexcept {
    CTAffinePoint result;
    affine_table_lookup_signed_into(&result, table, table_size, n, group_size);
    return result;
}

// --- In-Place Signed-Digit Affine Table Lookup -------------------------------
// Writes directly into *out -- zero return copies.
// AVX2 path: vectorized 256-bit cmov using VPAND/VPXOR over full CTAffinePoint
// (x.n[5] + y.n[5] = 80 bytes = 2.5 x ymm256 registers).
// Scalar fallback: original 5-limb XOR/AND per field element.

#if SECP256K1_CT_AVX2

// --- Always-inline AVX2 table lookup core ------------------------------------
// NORMALIZE_Y: if false, skip normalize_weak before Y-negate (safe when table
// entries are known magnitude-1, i.e. produced by multiplication).  Saves ~5ns.
template<bool NORMALIZE_Y>
__attribute__((always_inline)) inline
void table_lookup_core(CTAffinePoint* out,
                        const CTAffinePoint* table,
                        std::size_t table_size,
                        std::uint64_t n,
                        unsigned group_size) noexcept {
    std::uint64_t const negative = ((n >> (group_size - 1)) ^ 1u) & 1u;
    std::uint64_t const neg_mask = 0ULL - negative;

    unsigned const index_bits = group_size - 1;
    std::uint64_t const index = ((0ULL - negative) ^ n) &
                          ((1ULL << index_bits) - 1u);

    // Load first entry (80 bytes = x.n[5] + y.n[5]) into 3 ymm registers
    // Layout: [x.n[0..3]] [x.n[4], y.n[0..2]] [y.n[3..4], padding]
    // Aliasing note: __m256i has __attribute__(__may_alias__) in GCC/Clang immintrin.h,
    // so reinterpret_cast<__m256i*> from char* is defined behavior — no strict-aliasing UB.
    const auto* base0 = reinterpret_cast<const char*>(&table[0].x.n[0]);
    __m256i r0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base0));           // x.n[0..3] (32 bytes)
    __m256i r1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base0 + 32));      // x.n[4], y.n[0..2]
    // Last 16 bytes: y.n[3], y.n[4]
    // Use 128-bit load for the remaining 16 bytes to avoid reading past struct
    __m128i const r2lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(base0 + 64));
    __m256i r2 = _mm256_castsi128_si256(r2lo);

    // CT linear scan: cmov each entry using vectorized AND/XOR
    for (std::size_t m = 1; m < table_size; ++m) {
        std::uint64_t const eq = eq_mask(static_cast<std::uint64_t>(m),
                                   static_cast<std::uint64_t>(index));
        __m256i const vmask = _mm256_set1_epi64x(static_cast<long long>(eq));
        __m128i const vmask128 = _mm_set1_epi64x(static_cast<long long>(eq));

        const auto* base_m = reinterpret_cast<const char*>(&table[m].x.n[0]);
        __m256i const s0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base_m));
        __m256i const s1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base_m + 32));
        __m128i const s2lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(base_m + 64));
        __m256i const s2 = _mm256_castsi128_si256(s2lo);

        // r = (r ^ s) & mask ^ r  ==  mask ? s : r
        r0 = _mm256_xor_si256(r0, _mm256_and_si256(_mm256_xor_si256(r0, s0), vmask));
        r1 = _mm256_xor_si256(r1, _mm256_and_si256(_mm256_xor_si256(r1, s1), vmask));
        // For the last partial register, use 128-bit ops
        __m128i r2_128 = _mm256_castsi256_si128(r2);
        __m128i const s2_128 = _mm256_castsi256_si128(s2);
        r2_128 = _mm_xor_si128(r2_128, _mm_and_si128(_mm_xor_si128(r2_128, s2_128), vmask128));
        r2 = _mm256_castsi128_si256(r2_128);
    }

    // Store back into out (80 bytes)
    auto* dst_base = reinterpret_cast<char*>(&out->x.n[0]);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_base), r0);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_base + 32), r1);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst_base + 64), _mm256_castsi256_si128(r2));

    // Conditional Y-negate (sign handling) -- scalar, runs once
    FE52 neg_y = out->y;
    if constexpr (NORMALIZE_Y) {
        neg_y.normalize_weak();
    }
    neg_y = neg_y.negate(1);
    fe52_cmov(&out->y, neg_y, neg_mask);

    out->infinity = 0;
}

// Public wrapper (conservative: always normalizes Y before negate).
void affine_table_lookup_signed_into(CTAffinePoint* out,
                                      const CTAffinePoint* table,
                                      std::size_t table_size,
                                      std::uint64_t n,
                                      unsigned group_size) noexcept {
    table_lookup_core<true>(out, table, table_size, n, group_size);
}

#else // Scalar fallback

// --- Always-inline scalar table lookup core (non-AVX2 path) ------------------
template<bool NORMALIZE_Y>
__attribute__((always_inline)) inline
void table_lookup_core(CTAffinePoint* out,
                        const CTAffinePoint* table,
                        std::size_t table_size,
                        std::uint64_t n,
                        unsigned group_size) noexcept {
    std::uint64_t negative = ((n >> (group_size - 1)) ^ 1u) & 1u;
    std::uint64_t neg_mask = 0ULL - negative;

    unsigned index_bits = group_size - 1;
    std::uint64_t index = ((0ULL - negative) ^ n) &
                          ((1ULL << index_bits) - 1u);

    // CT scan: write first entry, then cmov over it
    out->x = table[0].x;
    out->y = table[0].y;
    for (std::size_t m = 1; m < table_size; ++m) {
        std::uint64_t mask = eq_mask(static_cast<std::uint64_t>(m),
                                     static_cast<std::uint64_t>(index));
        fe52_cmov(&out->x, table[m].x, mask);
        fe52_cmov(&out->y, table[m].y, mask);
    }

    // Conditional Y-negate (sign handling)
    FE52 neg_y = out->y;
    if constexpr (NORMALIZE_Y) {
        neg_y.normalize_weak();
    }
    neg_y = neg_y.negate(1);
    fe52_cmov(&out->y, neg_y, neg_mask);

    out->infinity = 0;
}

// Public wrapper
void affine_table_lookup_signed_into(CTAffinePoint* out,
                                      const CTAffinePoint* table,
                                      std::size_t table_size,
                                      std::uint64_t n,
                                      unsigned group_size) noexcept {
    table_lookup_core<true>(out, table, table_size, n, group_size);
}

#endif // SECP256K1_CT_AVX2

// --- Complete Addition (Jacobian, a=0, 5x52) --------------------------------
// Brier-Joye unified addition/doubling formula for general J+J.
// Handles all cases in a single branchless codepath:
//   P+Q (general), P+P (doubling), P+O, O+Q, P+(-P)=O
// Cost: 11M + 6S (vs previous 13M+9S dual-path approach).
// Reference: Eric Brier, Marc Joye -- "Weierstrass Elliptic Curves and
//   Side-Channel Attacks" (PKC 2002), adapted for Z1,Z2 != 1.

CTJacobianPoint point_add_complete(const CTJacobianPoint& p,
                                   const CTJacobianPoint& q) noexcept {
    // Normalize inputs to guarantee magnitude 1
    FE52 X1 = p.x; X1.normalize_weak();
    FE52 Y1 = p.y; Y1.normalize_weak();
    FE52 Z1 = p.z; Z1.normalize_weak();
    FE52 X2 = q.x; X2.normalize_weak();
    FE52 Y2 = q.y; Y2.normalize_weak();
    FE52 Z2 = q.z; Z2.normalize_weak();

    // -- Projective coordinates --
    FE52 const z1z1 = Z1.square();                           // Z1^2          [sqr #1]
    FE52 const z2z2 = Z2.square();                           // Z2^2          [sqr #2]
    FE52 const u1   = X1 * z2z2;                             // U1 = X1*Z2^2  [mul #1]
    FE52 const u2   = X2 * z1z1;                             // U2 = X2*Z1^2  [mul #2]
    FE52 const s1   = (Y1 * z2z2) * Z2;                      // S1 = Y1*Z2^3  [mul #3,#4]
    FE52 const s2   = (Y2 * z1z1) * Z1;                      // S2 = Y2*Z1^3  [mul #5,#6]
    FE52 const z    = Z1 * Z2;                               // Z  = Z1*Z2   [mul #7]

    // -- Unified formula: T = U1+U2, M = S1+S2 --
    FE52 const t_val  = u1 + u2;                             // T = U1+U2            (M=2)
    FE52 const m_val  = s1 + s2;                             // M = S1+S2            (M=2)

    // R = T^2 - U1*U2
    FE52 rr     = t_val.square();                       // T^2                   [sqr #3]
    FE52 const neg_u2 = u2.negate(1);                         // -U2                  (M=2)
    FE52 const tt     = u1 * neg_u2;                          // -U1*U2               [mul #8]
    rr = rr + tt;                                       // R = T^2-U1*U2         (M=2)

    // -- Degenerate case: M normalizes to zero ==> y1 = -y2 --
    std::uint64_t const degen     = fe52_normalizes_to_zero(m_val);
    std::uint64_t const not_degen = ~degen;

    // Alternate lambda = (S1-S2)/(U1-U2) for the degenerate case
    FE52 rr_alt = s1 + s1;                             // 2*S1                 (M=2)
    FE52 m_alt  = u1 + neg_u2;                         // U1-U2                (M=3)

    fe52_cmov(&rr_alt, rr,    not_degen);              // rr_alt = R if !degen
    fe52_cmov(&m_alt,  m_val, not_degen);              // m_alt  = M if !degen

    // -- Compute result coordinates --
    FE52 nn    = m_alt.square();                        // Malt^2                [sqr #4]
    FE52 q_val = t_val.negate(2);                       // -T                   (M=3)
    q_val = q_val * nn;                                 // Q = -T*Malt^2         [mul #9]

    // M^3*Malt: equals Malt^4 when M==Malt (non-degen), or ~0 when M==0 (degen)
    nn = nn.square();                                   // Malt^4                [sqr #5]
    fe52_cmov(&nn, m_val, degen);                       // M^3*Malt

    FE52 x3 = rr_alt.square() + q_val;                 // X3 = Ralt^2+Q         [sqr #6]
    FE52 z3 = z * m_alt;                                // Z3 = Z*Malt          [mul #10]

    FE52 const tq  = (x3 + x3) + q_val;                      // 2*X3+Q               (M=5)
    FE52 const y3p = (rr_alt * tq) + nn;                      // Ralt*(2X3+Q)+M^3Malt  [mul #11]
    FE52 y3  = y3p.negate(5).half();                    // Y3 = -(...)/2

    // -- Infinity handling --
    std::uint64_t const z3_zero = fe52_normalizes_to_zero(z3);

    // If P is inf -> result = Q
    fe52_cmov(&x3, X2, p.infinity);
    fe52_cmov(&y3, Y2, p.infinity);
    fe52_cmov(&z3, Z2, p.infinity);

    // If Q is inf -> result = P
    fe52_cmov(&x3, X1, q.infinity);
    fe52_cmov(&y3, Y1, q.infinity);
    fe52_cmov(&z3, Z1, q.infinity);

    // Infinity flag: z3==0 (P==-Q) when neither input inf; or both inf.
    std::uint64_t result_inf = z3_zero & ~p.infinity & ~q.infinity;
    result_inf |= p.infinity & q.infinity;

    CTJacobianPoint result;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    result.infinity = result_inf;
    return result;
}

// --- Mixed Jacobian+Affine Complete Addition (a=0, 5x52) ---------------------
// Brier-Joye unified addition/doubling (J + Affine) with full infinity handling.
// Cost: 7M + 5S. Handles all cases: P+Q, P+P, P+inf, inf+Q, P+(-P)=inf, inf+inf.

CTJacobianPoint point_add_mixed_complete(const CTJacobianPoint& p,
                                          const CTAffinePoint& q) noexcept {
    // X1, Y1, Z1 come from CTJacobianPoint which may carry high-magnitude
    // FE52 values (e.g. accumulated from repeated point additions without
    // intermediate normalization).  normalize_weak() reduces each limb to
    // its canonical bit width (magnitude 1) so that t_val.negate(m) and
    // subsequent multiplications stay within their magnitude contracts.
    FE52 X1 = p.x; X1.normalize_weak();
    FE52 Y1 = p.y; Y1.normalize_weak();
    FE52 const Z1 = p.z;

    // -- Brier-Joye unified (Z2=1): 7M + 5S --
    FE52 const zz      = Z1.square();                        // Z1^2          [sqr #1]
    FE52 const u1      = X1;                                 // U1 = X1 (magnitude 1)
    FE52 const u2      = q.x * zz;                           // U2 = x2*Z1^2  [mul #1]
    FE52 const s1      = Y1;                                 // S1 = Y1 (magnitude 1)
    FE52 const s2      = (q.y * zz) * Z1;                    // S2 = y2*Z1^3  [mul #2,#3]

    FE52 const t_val   = u1 + u2;                            // T = U1+U2
    FE52 const m_val   = s1 + s2;                            // M = S1+S2

    FE52 rr      = t_val.square();                     // T^2           [sqr #2]
    FE52 const neg_u2  = u2.negate(1);
    FE52 const tt      = u1 * neg_u2;                        // -U1*U2       [mul #4]
    rr = rr + tt;                                      // R = T^2-U1*U2

    std::uint64_t const degen     = fe52_normalizes_to_zero(m_val);
    std::uint64_t const not_degen = ~degen;

    FE52 rr_alt  = s1 + s1;                            // 2*S1
    FE52 m_alt   = u1 + neg_u2;                        // U1-U2

    fe52_cmov(&rr_alt, rr,    not_degen);
    fe52_cmov(&m_alt,  m_val, not_degen);

    FE52 nn      = m_alt.square();                     // Malt^2        [sqr #3]
    FE52 q_val   = t_val.negate(4);
    q_val = q_val * nn;                                // Q=-T*Malt^2   [mul #5]

    nn = nn.square();                                  // Malt^4        [sqr #4]
    fe52_cmov(&nn, m_val, degen);

    FE52 x3      = rr_alt.square() + q_val;            // X3=Ralt^2+Q   [sqr #5]
    FE52 z3      = m_alt * Z1;                          // Z3=Malt*Z1   [mul #6]

    FE52 const tq      = (x3 + x3) + q_val;                  // 2*X3+Q
    FE52 const y3p     = (rr_alt * tq) + nn;                 //               [mul #7]
    FE52 y3      = y3p.negate(5).half();                // Y3 = -(...)/2

    // -- Infinity handling --
    std::uint64_t const z3_zero = fe52_normalizes_to_zero(z3);

    FE52 const one52 = FE52::one();

    // If P is inf -> result = Q (affine, Z=1)
    fe52_cmov(&x3, q.x, p.infinity);
    fe52_cmov(&y3, q.y, p.infinity);
    fe52_cmov(&z3, one52, p.infinity);

    // If Q is inf -> result = P
    fe52_cmov(&x3, X1, q.infinity);
    fe52_cmov(&y3, Y1, q.infinity);
    fe52_cmov(&z3, Z1, q.infinity);

    // Infinity flag
    std::uint64_t result_inf = z3_zero & ~p.infinity & ~q.infinity;
    result_inf |= p.infinity & q.infinity;

    CTJacobianPoint result;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    result.infinity = result_inf;
    return result;
}

// --- CT Point Doubling (5x52) ------------------------------------------------
// Libsecp-style 3M+4S+1half doubling (low magnitudes, X M<=3, Y M<=3, Z M=1).
// Formula:
//   L = (3/2) * X^2
//   S = Y^2
//   T = -X*S
//   X3 = L^2 + 2*T
//   Y3 = -(L*(X3 + T) + S^2)
//   Z3 = Y*Z

CTJacobianPoint point_dbl(const CTJacobianPoint& p) noexcept {
    FE52 const X = p.x;
    FE52 const Y = p.y;
    FE52 const Z = p.z;

    FE52 s = Y.square();                   // S = Y^2        [1S]  M=1
    FE52 l = X.square();                   // L = X^2        [1S]  M=1
    l = (l + l) + l;                       // L = 3*X^2            M=3
    l = l.half();                          // L = (3/2)*X^2        M=2
    FE52 t = s.negate(1) * X;             // T = -X*S      [1M]  M=1
    FE52 const z3 = Z * Y;                       // Z3 = Y*Z      [1M]  M=1
    FE52 x3 = l.square();                  // X3 = L^2       [1S]  M=1
    x3 = x3 + t;                           //                     M=2
    x3 = x3 + t;                           // X3 = L^2+2T         M=3
    s = s.square();                        // S' = S^2        [1S]  M=1
    t = t + x3;                             // T' = X3+T          M=4
    FE52 y3 = (t * l) + s;                 // Y3 = L*T'+S'  [1M]  M=2
    y3 = y3.negate(2);                     // Y3 = -(...)         M=3

    // If P is infinity, result is infinity
    CTJacobianPoint const inf = CTJacobianPoint::make_infinity();
    CTJacobianPoint result;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    result.infinity = 0;

    point_cmov(&result, inf, p.infinity);
    return result;
}

// --- Batch N Doublings (5x52, no infinity) -----------------------------------
// Performs n consecutive doublings in-place, normalizing only at the start.
// The doubling formula (dbl-2007-a for a=0) has self-stabilizing magnitudes:
// all negate() parameters depend on intermediate values (always M=1 from
// mul/sqr), NOT on input magnitude. So consecutive doublings are safe without
// per-step normalize_weak.
//
// Precondition: p is NOT infinity (caller ensures).
// Output: 2^n * p

// Always-inline core: called from scalar_mul hot loop AND public wrapper.
// Magnitude contract (NO normalize_weak calls):
//   Input from unified add: X M<=2, Y M=1, Z M=1
//   Input from prev dbl:    X M<=18, Y M<=10, Z M<=2
//   All negate() params already handle these bounds.
//   After each dbl: X M=18, Y M=10, Z M=2 (fixed point, independent of input).
//
// Precondition: r is NOT infinity.
__attribute__((always_inline)) inline
void point_dbl_n_core(CTJacobianPoint* r, unsigned n) noexcept {
    FE52 X = r->x;   // no normalize_weak -- magnitudes tracked
    FE52 Y = r->y;
    FE52 Z = r->z;

    for (unsigned i = 0; i < n; ++i) {
        FE52 const A = X.square();                   // A = X^2         M=1
        FE52 const B = Y.square();                   // B = Y^2         M=1
        FE52 C = B.square();                   // C = Y^4         M=1

        // D = 2*((X+B)^2 - A - C) -- reuse X as scratch (X dead after A)
        X.add_assign(B);                       // X+B            M<=19
        X.square_inplace();                    // (X+B)^2         M=1
        FE52 AC = A;
        AC.add_assign(C);                      // A+C            M=2
        AC.negate_assign(2);                   // -(A+C)         M=3
        X.add_assign(AC);                      // D_half         M=4
        X.add_assign(X);                       // D              M=8
        // X is now D

        // E = 3A (A still alive, last use)
        FE52 E = A;
        E.add_assign(E);                       // 2A             M=2
        E.add_assign(A);                       // 3A             M=3

        FE52 F = E.square();                   // F = E^2         M=1

        // x_new = F - 2D; need D preserved for Dx
        FE52 D_save = X;                       // save D         M=8
        X.add_assign(X);                       // 2D             M=16
        X.negate_assign(16);                   // -2D            M=17
        F.add_assign(X);                       // x_new = F-2D   M=18
        // F is now x_new

        // 8C chain (in-place on C)
        C.add_assign(C);                       // 2C             M=2
        C.add_assign(C);                       // 4C             M=4
        C.add_assign(C);                       // 8C             M=8
        C.negate_assign(8);                    // -8C            M=9

        // Dx = D - x_new
        FE52 neg_xnew = F;
        neg_xnew.negate_assign(18);            // -x_new         M=19
        D_save.add_assign(neg_xnew);           // Dx             M=27

        // Z_new = 2*Y*Z (reads old Y -- compute before overwrite)
        FE52 const yz = Y * Z;                       // Y*Z            M=1

        // Y_new = E*Dx - 8C
        E.mul_assign(D_save);                  // E*Dx           M=1
        E.add_assign(C);                       // E*Dx - 8C      M=10

        X = F;                                 // x_new          M=18
        Y = E;                                 // y_new          M=10
        Z = yz;
        Z.add_assign(Z);                       // 2*Y*Z          M=2
    }

    r->x = X;
    r->y = Y;
    r->z = Z;
}

// Public wrapper (out-of-line for external callers).
void point_dbl_n_inplace(CTJacobianPoint* r, unsigned n) noexcept {
    point_dbl_n_core(r, n);
}

// Backward-compatible wrapper (returns by value)
CTJacobianPoint point_dbl_n(const CTJacobianPoint& p, unsigned n) noexcept {
    CTJacobianPoint result = p;
    result.infinity = 0;
    point_dbl_n_inplace(&result, n);
    return result;
}

// --- Brier-Joye Unified Mixed Addition (Jacobian + Affine, a=0) -------------
// Cost: 7M + 5S (vs 9M+8S for complete formula) -- ~40% cheaper
//
// Unified: handles both addition (a != +/-b) and doubling (a == b) in one path.
// Degenerates only when y1 == -y2 (M = 0), handled via alternate lambda cmov.
// Detects a = -b (result = infinity) via Z3 == 0.
//
// Precondition: b MUST NOT be infinity.
//   a may be infinity (handled via cmov at end -> result = b).
//
// Based on: E. Brier, M. Joye "Weierstrass Elliptic Curves and Side-Channel
// Attacks" (PKC 2002). Formula from bitcoin-core/secp256k1 group_impl.h.

CTJacobianPoint point_add_mixed_unified(const CTJacobianPoint& a,
                                         const CTAffinePoint& b) noexcept {
    CTJacobianPoint result;
    point_add_mixed_unified_into(&result, a, b);
    return result;
}

// --- Always-Inline Brier-Joye Unified Mixed Addition (template core) --------
// Template parameter CHECK_INFINITY controls whether infinity handling is done.
// In scalar_mul's main loop, both a and b are known non-infinity, so
// CHECK_INFINITY=false saves 3 fe52_cmov + 1 fe52_is_zero per call.
//
// Cost: 7M + 5S. Precondition: b MUST NOT be infinity.

// -- Incomplete CT mixed add: Jacobian + Affine → Jacobian (7M + 3S) ---------
// PRECONDITION (Hamburg guarantee): a ≠ O, b ≠ O, a ≠ ±b.
// No degenerate case detection — 3.5× fewer squarings than unified_add_core.
// Used in the Hamburg-coded main loop where degeneracy is provably impossible.
__attribute__((always_inline)) inline
void add_affine_fast_ct(CTJacobianPoint* out,
                         const CTJacobianPoint& a,
                         const CTAffinePoint& b) noexcept {
    FE52 const& X1 = a.x;
    FE52 const& Y1 = a.y;
    FE52 const& Z1 = a.z;

    FE52 const Z1Z1 = Z1.square();          // Z1^2         [1S]
    FE52 const U2   = b.x * Z1Z1;           // X2*Z1^2      [1M]
    FE52       S2   = b.y * Z1Z1;
    S2.mul_assign(Z1);                       // Y2*Z1^3      [2M]

    FE52 H = U2;
    H.add_assign(X1.negate(18));            // H = U2-X1  (X1 mag<=18 after point_dbl_n_core)
    FE52 R = S2;
    R.add_assign(Y1.negate(10));            // R = S2-Y1  (Y1 mag<=10 after point_dbl_n_core)

    FE52 const Z3   = H * Z1;               // Z3 = H*Z1    [1M]
    FE52 const HH   = H.square();           // H^2          [1S]
    FE52 const HHH  = HH * H;               // H^3          [1M]
    FE52 const U1HH = X1 * HH;              // X1*H^2       [1M]

    FE52 const RR   = R.square();           // R^2          [1S]
    FE52 X3 = RR;
    X3.add_assign(HHH.negate(1));           // R^2-H^3
    X3.add_assign(U1HH.negate(1));
    X3.add_assign(U1HH.negate(1));         // R^2-H^3-2*X1*H^2 = X3  (mag=7)

    FE52 Y3 = U1HH;
    Y3.add_assign(X3.negate(7));            // X1*H^2 - X3  (X3 mag=7)
    Y3.mul_assign(R);                       // R*(X1*H^2 - X3) [1M]
    Y3.add_assign((Y1 * HHH).negate(1));   // - Y1*H^3       [1M]

    out->x = X3;
    out->y = Y3;
    out->z = Z3;
    out->infinity = 0;
}

// -- Incomplete CT mixed add specialised for Z1 == 1 -------------------------
// add_affine_fast_ct with Z1 = 1 substituted.  That turns five of its field
// operations into identities: Z1Z1 = 1, U2 = b.x, both halves of S2 = b.y,
// and Z3 = H.  Used for the comb's first mixed addition, where the accumulator
// was loaded straight out of an affine table entry one statement earlier.
// The peel is keyed on the block index, a public loop counter, never on the
// scalar, so the CT profile is unchanged.
//
// MAGNITUDES -- the part that must not be got wrong:
//  * U2 := b.x is magnitude 1, exactly what the multiply it replaces emitted,
//    so H is still 20 and the negate(18) literal must NOT be lowered.
//  * S2 := b.y is the one magnitude that changes.  comb_lookup may hand back
//    the conditionally negated y, which is magnitude 2 where a multiply would
//    have emitted magnitude 1, so R becomes 13 instead of 12.  R is only ever
//    squared or multiplied after this, and the lazy-add headroom is ~4096, so
//    13 is far inside the contract.
//  * Z3 := H would hand the next iteration a magnitude-20 Z, so it is
//    normalize_weak'd back to magnitude 1 for less than the cost of a multiply.
//  * X3 (7) and Y3 (3) are unchanged.
__attribute__((always_inline)) inline
void add_affine_fast_ct_z1(CTJacobianPoint* out,
                            const CTJacobianPoint& a,
                            const CTAffinePoint& b) noexcept {
    FE52 const& X1 = a.x;
    FE52 const& Y1 = a.y;

    FE52 H = b.x;                            // U2 = b.x*Z1^2 with Z1 == 1
    H.add_assign(X1.negate(18));            // H = U2-X1  (X1 mag<=18)
    FE52 R = b.y;                            // S2 = b.y*Z1^3 with Z1 == 1
    R.add_assign(Y1.negate(10));            // R = S2-Y1  (mag 13, see above)

    FE52 Z3 = H;                             // Z3 = H*Z1 with Z1 == 1
    Z3.normalize_weak();                     // back to magnitude 1

    FE52 const HH   = H.square();           // H^2          [1S]
    FE52 const HHH  = HH * H;               // H^3          [1M]
    FE52 const U1HH = X1 * HH;              // X1*H^2       [1M]

    FE52 const RR   = R.square();           // R^2          [1S]
    FE52 X3 = RR;
    X3.add_assign(HHH.negate(1));           // R^2-H^3
    X3.add_assign(U1HH.negate(1));
    X3.add_assign(U1HH.negate(1));         // R^2-H^3-2*X1*H^2 = X3  (mag=7)

    FE52 Y3 = U1HH;
    Y3.add_assign(X3.negate(7));            // X1*H^2 - X3  (X3 mag=7)
    Y3.mul_assign(R);                       // R*(X1*H^2 - X3) [1M]
    Y3.add_assign((Y1 * HHH).negate(1));   // - Y1*H^3       [1M]

    out->x = X3;
    out->y = Y3;
    out->z = Z3;
    out->infinity = 0;
}

// HAMBURG=true: skip degenerate case detection (provably impossible with
// the Hamburg scalar encoding). Keeps the same formula structure as the
// general case for optimal ILP, but removes fe52_normalizes_to_zero()
// and the conditional-select overhead (~16-20 ns per call × 52 = ~1 µs).
template<bool CHECK_INFINITY, bool HAMBURG = false>
__attribute__((always_inline)) inline
void unified_add_core(CTJacobianPoint* out,
                       const CTJacobianPoint& a,
                       const CTAffinePoint& b) noexcept {
    // Read a's coords into locals (handles out==&a aliasing, register-friendly).
    // NO normalize_weak -- magnitude bounds tracked explicitly:
    //   After dbl_n:      X M<=18, Y M<=10, Z M<=2
    //   After unified add: X M<=2,  Y M=1,   Z M=1
    // Worst case: M_x=18, M_y=10, M_z=2. All negate() params adjusted.
    FE52 const X1 = a.x;
    FE52 const Y1 = a.y;
    FE52 const Z1 = a.z;
    [[maybe_unused]] std::uint64_t const a_inf = a.infinity;
    (void)a_inf; // CodeQL: suppress unused-local-variable (kept for ABI symmetry)

    // -- Shared intermediates --
    FE52 const zz = Z1.square();                          // Z1^2       [1S]  M=1
    FE52 const u1 = X1;                                   // U1 = X1   M<=18
    FE52 u2 = b.x * zz;                             // U2 = x2*Z1^2  M=1  [1M]
    FE52 const s1 = Y1;                                   // S1 = Y1   M<=10
    FE52 const s2 = (b.y * zz) * Z1;                      // S2 = y2*Z1^3  M=1  [2M]

    FE52 t_val = u1;
    t_val.add_assign(u2);                            // T = U1+U2 M<=19
    FE52 m_val = s1;
    m_val.add_assign(s2);                            // M = S1+S2 M<=11

    // R = T^2 - U1*U2
    FE52 rr = t_val.square();                        // T^2  M=1      [1S]
    u2.negate_assign(1);                             // -U2 M=2 (u2 reused as neg_u2)
    FE52 const tt = u1 * u2;                              // -U1*U2 M=1  [1M]
    rr.add_assign(tt);                               // R   M=2

    // -- Degenerate case: M~=0 (y1=-y2) --
    // HAMBURG: degen is provably 0 — skip normalizes_to_zero + cmovs (~16 ns).
    [[maybe_unused]] std::uint64_t degen = 0;
    FE52 rr_alt;
    FE52 m_alt;

    if constexpr (!HAMBURG) {
        degen = fe52_normalizes_to_zero(m_val);
        rr_alt = s1;
        rr_alt.add_assign(s1);                       // 2*S1   M<=20
        m_alt = u1;
        m_alt.add_assign(u2);                        // U1-U2 (u2 is negated)  M<=20
        fe52_cmov(&rr_alt, rr, ~degen);
        fe52_cmov(&m_alt, m_val, ~degen);
    } else {
        // Hamburg guarantee: regular path always taken — use rr/m_val directly.
        rr_alt = rr;
        m_alt  = m_val;
    }

    // -- Compute result into locals (no pointer reads after partial writes) --
    FE52 n = m_alt.square();                         // Malt^2 M=1   [1S]
    t_val.negate_assign(19);                         // -T    M=20  (t_val reused)
    FE52 const q = t_val * n;                              // Q     M=1   [1M]

    n = n.square();                                  // Malt^4 M=1   [1S]
    if constexpr (!HAMBURG) {
        fe52_cmov(&n, m_val, degen);                 // if degen: n=M~=0  M<=11
    }

    FE52 x3 = rr_alt.square();
    x3.add_assign(q);                                // X3    M=2   [1S]
    // NOLINTNEXTLINE(misc-const-correctness) -- modified by fe52_cmov in CHECK_INFINITY path
    FE52 z3 = m_alt * Z1;                                  // Z3    M=1   [1M]

    FE52 tq = x3;
    tq.add_assign(x3);                               // 2*X3  M=4
    tq.add_assign(q);                                // 2X3+Q M=5
    rr_alt.mul_assign(tq);                           // Ralt*(2X3+Q) M=1  [1M=7th]
    rr_alt.add_assign(n);                            // +M^3Malt      M<=12
    rr_alt.negate_assign(12);                        // -(...)        M=13
    // NOLINTNEXTLINE(misc-const-correctness) -- modified by fe52_cmov in CHECK_INFINITY path
    FE52 y3 = rr_alt.half();                               // Y3    M=1 (half normalizes)

    if constexpr (CHECK_INFINITY) {
        // Handle a=infinity: replace result with (b.x, b.y, 1)
        FE52 const one52 = FE52::one();
        fe52_cmov(&x3, b.x, a_inf);
        fe52_cmov(&y3, b.y, a_inf);
        fe52_cmov(&z3, one52, a_inf);
    }

    // Single write to output at the very end
    out->x = x3;
    out->y = y3;
    out->z = z3;

    if constexpr (CHECK_INFINITY) {
        out->infinity = fe52_normalizes_to_zero(z3);
    } else {
        out->infinity = 0;
    }
}

// Public wrapper (out-of-line for external callers, preserves all safety checks).
void point_add_mixed_unified_into(CTJacobianPoint* out,
                                   const CTJacobianPoint& a,
                                   const CTAffinePoint& b) noexcept {
    unified_add_core<true>(out, a, b);
}

// ===============================================================================
// CT GLV Endomorphism -- Helpers & Decomposition
// ===============================================================================

namespace /* GLV CT helpers */ {

// --- Local sub256 (needed for ct_scalar_is_high) -----------------------------
static inline std::uint64_t local_sub256(std::uint64_t r[4],
                                          const std::uint64_t a[4],
                                          const std::uint64_t b[4]) noexcept {
    std::uint64_t borrow = 0;
    for (int i = 0; i < 4; ++i) {
        std::uint64_t const diff = a[i] - b[i];
        auto const b1 = static_cast<std::uint64_t>(a[i] < b[i]);
        std::uint64_t const result = diff - borrow;
        auto const b2 = static_cast<std::uint64_t>(diff < borrow);
        r[i] = result;
        borrow = b1 + b2;
    }
    return borrow;
}

// --- CT scalar > n/2 check --------------------------------------------------
static std::uint64_t ct_scalar_is_high(const Scalar& s) noexcept {
    static constexpr std::uint64_t HALF_N[4] = {
        0xDFE92F46681B20A0ULL,
        0x5D576E7357A4501DULL,
        0xFFFFFFFFFFFFFFFFULL,
        0x7FFFFFFFFFFFFFFFULL
    };
    std::uint64_t tmp[4];
    std::uint64_t const borrow = local_sub256(tmp, HALF_N, s.limbs().data());
    return is_nonzero_mask(borrow);
}

// --- CT 256x256->512 multiply, shift >>384 with rounding ---------------------
static std::array<std::uint64_t, 4> ct_mul_shift_384(
    const std::array<std::uint64_t, 4>& a,
    const std::array<std::uint64_t, 4>& b) noexcept
{
    std::uint64_t prod[8] = {};
    for (std::size_t i = 0; i < 4; ++i) {
        unsigned __int128 carry = 0;
        for (std::size_t j = 0; j < 4; ++j) {
            unsigned __int128 const t = static_cast<unsigned __int128>(a[i]) * b[j]
                                + prod[i + j] + carry;
            prod[i + j] = static_cast<std::uint64_t>(t);
            carry = t >> 64;
        }
        prod[i + 4] = static_cast<std::uint64_t>(carry);
    }

    std::array<std::uint64_t, 4> result{};
    result[0] = prod[6];
    result[1] = prod[7];

    std::uint64_t const round = prod[5] >> 63;
    std::uint64_t const old = result[0];
    result[0] += round;
    auto const carry = static_cast<std::uint64_t>(result[0] < old);
    result[1] += carry;

    return result;
}

// --- CT 128x128 -> 256 multiply ----------------------------------------------
// Every product in ct_glv_decompose pairs a <=128-bit rounding quotient with a
// <=128-bit lattice constant, so a 2x2 schoolbook is exact.  The operand WIDTH
// is a property of the fixed constants and of the rounding step -- c1 < 2^126
// and c2 < 2^128 hold for EVERY k, not just for the k in hand -- so the narrow
// form is width-invariant: fixed trip counts, no branch, no data-dependent
// index, nothing observable that depends on the secret scalar.
static inline void ct_mul_2x2(std::uint64_t r[4],
                               const std::uint64_t a[2],
                               const std::uint64_t b[2]) noexcept {
    r[0] = 0; r[1] = 0; r[2] = 0; r[3] = 0;
    for (std::size_t i = 0; i < 2; ++i) {
        unsigned __int128 carry = 0;
        for (std::size_t j = 0; j < 2; ++j) {
            unsigned __int128 const t = static_cast<unsigned __int128>(a[i]) * b[j]
                                + r[i + j] + carry;
            r[i + j] = static_cast<std::uint64_t>(t);
            carry = t >> 64;
        }
        r[i + 2] = static_cast<std::uint64_t>(carry);
    }
}

// --- beta (beta) as FE52 -- cube root of unity mod p ----------------------------
static const FE52& get_beta_fe52() noexcept {
    static const FE52 beta = FE52::from_fe(
        FieldElement::from_bytes(secp256k1::fast::glv_constants::BETA));
    return beta;
}

// --- GLV lattice constants ---------------------------------------------------
static constexpr std::array<std::uint64_t, 4> kG1{{
    0xE893209A45DBB031ULL, 0x3DAA8A1471E8CA7FULL,
    0xE86C90E49284EB15ULL, 0x3086D221A7D46BCDULL
}};
static constexpr std::array<std::uint64_t, 4> kG2{{
    0x1571B4AE8AC47F71ULL, 0x221208AC9DF506C6ULL,
    0x6F547FA90ABFE4C4ULL, 0xE4437ED6010E8828ULL
}};

} // anonymous namespace (GLV CT helpers)

// --- CT GLV Endomorphism (5x52) ----------------------------------------------

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

// --- CT GLV Decomposition (full scalar_mul mod n, matches libsecp256k1) ------

CTGLVDecomposition ct_glv_decompose(const Scalar& k) noexcept {
    // Raw GLV lattice basis, low 128 bits only.  libsecp256k1 publishes the
    // DERIVED constant set (-b1, -b2, lambda); -b2 = n - b2 is a full 256-bit
    // number even though b2 itself is 126 bits, which forces a 256x256 multiply
    // plus a wide mod-n reduction for every c2 term, and then a third 256x256
    // multiply by lambda to recover k1.  Working from the raw basis instead keeps
    // all four products inside 2x2:
    //   -b1  = 0xE4437ED6010E8828_6F547FA90ABFE4C3   (128 bits)
    //    b2  = 0x3086D221A7D46BCD_E86C90E49284EB15   (126 bits), and a1 == b2
    //    a2  = 2^128 + kA2Lo, kA2Lo = 125 bits
    // The lattice relations a1 + b1*lambda == 0 (mod n) and a2 + b2*lambda == 0
    // (mod n) hold exactly, so
    //   lambda*k2 = lambda*(-c1*b1 - c2*b2) = c1*a1 + c2*a2   (mod n)
    // and k1 needs no lambda multiply at all -- the 2^128 half of a2 is a limb
    // placement, not a multiply.
    static constexpr std::uint64_t kMinusB1[2] = {
        0x6F547FA90ABFE4C3ULL, 0xE4437ED6010E8828ULL
    };
    static constexpr std::uint64_t kB2[2] = {          // b2, and a1 == b2
        0xE86C90E49284EB15ULL, 0x3086D221A7D46BCDULL
    };
    static constexpr std::uint64_t kA2Lo[2] = {        // a2 - 2^128
        0x57C1108D9D44CFD8ULL, 0x14CA50F7A8E2F3F6ULL
    };

    auto k_limbs = k.limbs();
    auto c1_limbs = ct_mul_shift_384({k_limbs[0], k_limbs[1], k_limbs[2], k_limbs[3]}, kG1);
    auto c2_limbs = ct_mul_shift_384({k_limbs[0], k_limbs[1], k_limbs[2], k_limbs[3]}, kG2);

    // Width bounds, valid for EVERY 4-limb k (c1 and c2 are monotonic in k, so
    // evaluating at the largest 4-limb input k = 2^256-1 bounds them all):
    //   c1 <= 2^126 and c2 <= 2^128, hence c1_limbs[2..3] and c2_limbs[2..3] are
    //   always zero and reading only the low two limbs cannot depend on k.
    //   c1*(-b1) < 2^254, c2*b2 < 2^254, c1*a1 < 2^252, c2*kA2Lo < 2^253,
    //   c1*a1 + c2*kA2Lo < 2^253, and c2*2^128 < n - 2^253.
    // Every product is therefore already reduced -- no mod-n reduction anywhere.
    std::uint64_t mb1c1[4], b2c2[4], a1c1[4], a2c2[4];
    ct_mul_2x2(mb1c1, c1_limbs.data(), kMinusB1);
    ct_mul_2x2(b2c2,  c2_limbs.data(), kB2);
    ct_mul_2x2(a1c1,  c1_limbs.data(), kB2);
    ct_mul_2x2(a2c2,  c2_limbs.data(), kA2Lo);

    // k2 = c1*(-b1) - c2*b2 mod n.  scalar_sub is the branchless mod-n
    // subtract (borrow mask + cmov256), so the wrap is a cmov, never a branch.
    Scalar const k2_mod = scalar_sub(
        Scalar::from_limbs({mb1c1[0], mb1c1[1], mb1c1[2], mb1c1[3]}),
        Scalar::from_limbs({b2c2[0], b2c2[1], b2c2[2], b2c2[3]}));

    std::uint64_t const k2_high = ct_scalar_is_high(k2_mod);
    Scalar const k2_abs = scalar_cneg(k2_mod, k2_high);

    // k1 = k - lambda*k2 = k - c1*a1 - c2*kA2Lo - c2*2^128 mod n.
    Scalar const lambda_k2 = scalar_add(
        Scalar::from_limbs({a1c1[0], a1c1[1], a1c1[2], a1c1[3]}),
        Scalar::from_limbs({a2c2[0], a2c2[1], a2c2[2], a2c2[3]}));
    Scalar const c2_shifted = Scalar::from_limbs({0, 0, c2_limbs[0], c2_limbs[1]});
    Scalar const k1_mod = scalar_sub(scalar_sub(k, lambda_k2), c2_shifted);

    std::uint64_t const k1_high = ct_scalar_is_high(k1_mod);
    Scalar const k1_abs = scalar_cneg(k1_mod, k1_high);

    CTGLVDecomposition result;
    result.k1     = k1_abs;
    result.k2     = k2_abs;
    result.k1_neg = k1_high;
    result.k2_neg = k2_high;
    return result;
}

// --- CT GLV Scalar Multiplication -- Hamburg Signed-Digit (GROUP_SIZE=5) ------
// Combines Hamburg's signed-digit comb (ePrint 2012/309 S.3.3) with GLV.
//
// Algorithm:
//   1. s = (k + K) / 2 mod n, where K = (2^BITS-2^129-1)*(1+lambda) mod n
//   2. Split s into (s1, s2) via GLV
//   3. v1 = s1 + 2^128,  v2 = s2 + 2^128  (non-negative, fits in 129 bits)
//   4. Process GROUPS groups of GROUP_SIZE bits from v1, v2 (high to low)
//   5. Each group's bits encode a signed odd digit -> table lookup
//
// GROUP_SIZE=5: TABLE_SIZE=16 odd multiples, GROUPS=26, BITS=130.
// Cost: 125 doublings + 52 additions (ALL real, no digit=0 waste).
// Table lookup: 26 groups x 2 tables x 15 cmov = 780 cmov iterations.

// --- Internal Jacobian-output CT scalar multiply (FE52 path) -----------------
// Same algorithm as scalar_mul() but returns the raw Jacobian CTJacobianPoint
// with global_z already applied (NOT yet normalized via Z-inversion).
// Used by ecmult_const_xonly() to combine Z^-1 with the g-correction into a
// single field inversion (libsecp secp256k1_ecmult_const_xonly technique).
//
// Returns CTJacobianPoint with infinity flag set if k==0 or p==infinity.
// Output: {R.x, R.y, R.z} in Jacobian form; affine x = R.x * R.z^{-2}.
static CTJacobianPoint scalar_mul_jac(const Point& p, const Scalar& k) noexcept;

// --- CT GLV make_v helper ----------------------------------------------------
// CT-negates or CT-increments the GLV half-scalar based on its sign bit.
// Extracted from the four scalar_mul_* functions to avoid copy-paste divergence.
SECP256K1_INLINE static Scalar ct_glv_make_v(const Scalar& k_abs, std::uint64_t k_neg) noexcept {
    auto L = k_abs.limbs();
    std::uint64_t const neg_mask = -static_cast<std::uint64_t>(k_neg != 0);
    std::uint64_t const pos_mask = ~neg_mask;
    std::uint64_t nL[4], borrow = 0, diff;
    diff = 0 - L[0]; borrow = (L[0] != 0) ? 1 : 0; nL[0] = diff;
    diff = 0 - L[1] - borrow; borrow = (L[1] | borrow) ? 1 : 0; nL[1] = diff;
    diff = 1 - L[2] - borrow; borrow = (L[2] + borrow > 1) ? 1 : 0; nL[2] = diff;
    nL[3] = 0 - borrow;
    std::uint64_t pL[4];
    pL[0] = L[0]; pL[1] = L[1];
    std::uint64_t const sum = L[2] + 1;
    pL[2] = sum; pL[3] = L[3] + static_cast<std::uint64_t>(sum < L[2]);
    L[0] = (neg_mask & nL[0]) | (pos_mask & pL[0]);
    L[1] = (neg_mask & nL[1]) | (pos_mask & pL[1]);
    L[2] = (neg_mask & nL[2]) | (pos_mask & pL[2]);
    L[3] = (neg_mask & nL[3]) | (pos_mask & pL[3]);
    return Scalar::from_limbs(L);
}

// --- scalar_mul_jac_fe52_z1 --------------------------------------------------
// Same as scalar_mul_jac but takes raw FE52 affine coordinates (Z=1 assumed).
// BYPASSES the SECP_ASSERT_ON_CURVE check — safe for effective-affine points
// on secp256k1-isomorphic curves (a=0 → same group law, different b).
// Used by ecmult_const_xonly where P_eff = (g·xn, g²) is NOT on secp256k1.
static CTJacobianPoint scalar_mul_jac_fe52_z1(const FE52& px, const FE52& py,
                                               const Scalar& k) noexcept {
    constexpr unsigned GROUP_SIZE = 5;
    constexpr unsigned TABLE_SIZE = 1u << (GROUP_SIZE - 1);
    constexpr unsigned GROUPS = 26;

    static const Scalar K_CONST = Scalar::from_limbs({
        0xb5c2c1dcde9798d9ULL, 0x589ae84826ba29e4ULL,
        0xc2bdd6bf7c118d6bULL, 0xa4e88a7dcb13034eULL
    });

    Scalar s = scalar_add(k, K_CONST);
    s = scalar_half(s);
    auto [k1_abs, k2_abs, k1_neg, k2_neg] = ct_glv_decompose(s);

    Scalar const v1 = ct_glv_make_v(k1_abs, k1_neg);
    Scalar const v2 = ct_glv_make_v(k2_abs, k2_neg);

    CTAffinePoint pre_a[TABLE_SIZE];
    CTAffinePoint pre_a_lam[TABLE_SIZE];

    // Double the input point using CTJacobianPoint path (no on-curve assert).
    // point_dbl_n_core operates on CTJacobianPoint — safe for isomorphic-curve points.
    JacFE52 iso[TABLE_SIZE];
    FE52 zr[TABLE_SIZE];
    FE52 global_z;

#if defined(REPSEARCH_COZ_TABLE) && REPSEARCH_COZ_TABLE == 1
    // ---- co-Z chain (EXPERIMENT ONLY) -----------------------------------
    // Fourth and last site. This one enters with Z == 1, so DBLU's Z_new
    // reduces to 2*Y and the multiply by the incoming Z is by one -- kept
    // generic rather than specialised, because the saving is a single multiply
    // once per call and a second code path is not worth it.
    {
        FE52 dX, dY, aX, aY, chainZ;
        jac52_dblu_ct(px, py, FE52::one(), dX, dY, aX, aY, chainZ);

        iso[0] = { aX, aY, chainZ };
        zr[0] = FE52::one();

        for (std::size_t i = 1; i < TABLE_SIZE; ++i) {
            FE52 sx, sy;
            if (!jac52_zaddu_ct(dX, dY, aX, aY, chainZ, sx, sy, zr[i])) {
                // Degenerate X1 == X2 is impossible for a chain of odd multiples
                // of a valid point; return infinity rather than a wrong point.
                return CTJacobianPoint::make_infinity();
            }
            aX = sx;
            aY = sy;
            iso[i] = { aX, aY, chainZ };
        }
        global_z = chainZ;
    }
#else
    CTJacobianPoint p2_ct{px, py, FE52::one(), 0};
    point_dbl_n_core(&p2_ct, 1);   // 2P in Jacobian (bypasses SECP_ASSERT_ON_CURVE)
    FE52 const C  = p2_ct.z;
    FE52 const C2 = C.square();
    FE52 const C3 = C2 * C;
    FE52 const d_x = p2_ct.x;
    FE52 const d_y = p2_ct.y;

    iso[0] = { px * C2, py * C3, FE52::one() };   // P in iso coords (Z_P=1)

    for (std::size_t i = 1; i < TABLE_SIZE; ++i)
        iso[i] = jac_add_ge_var_zr(iso[i - 1], d_x, d_y, &zr[i]);

    global_z = iso[TABLE_SIZE - 1].z * C;
#endif  // REPSEARCH_COZ_TABLE
    const FE52& beta = get_beta_fe52();

    pre_a[TABLE_SIZE - 1].x = iso[TABLE_SIZE - 1].x;
    pre_a[TABLE_SIZE - 1].y = iso[TABLE_SIZE - 1].y;
    pre_a[TABLE_SIZE - 1].y.normalize_weak();
    pre_a[TABLE_SIZE - 1].infinity = 0;
    pre_a_lam[TABLE_SIZE - 1].x = pre_a[TABLE_SIZE - 1].x * beta;
    pre_a_lam[TABLE_SIZE - 1].y = pre_a[TABLE_SIZE - 1].y;
    pre_a_lam[TABLE_SIZE - 1].infinity = 0;

    FE52 zs_acc = zr[TABLE_SIZE - 1];
    for (int i = static_cast<int>(TABLE_SIZE) - 2; i >= 0; --i) {
        FE52 const zs2 = zs_acc.square();
        FE52 const zs3 = zs2 * zs_acc;
        pre_a[i].x = iso[i].x * zs2;
        pre_a[i].y = iso[i].y * zs3;
        pre_a[i].infinity = 0;
        pre_a_lam[i].x = pre_a[i].x * beta;
        pre_a_lam[i].y = pre_a[i].y;
        pre_a_lam[i].infinity = 0;
        if (i > 0) zs_acc = zs_acc * zr[i];
    }

    SECP256K1_DECLASSIFY(pre_a, sizeof(pre_a));
    SECP256K1_DECLASSIFY(pre_a_lam, sizeof(pre_a_lam));

    // HAMBURG=true: K_CONST guarantees m_val (S1+S2) ≠ 0 for all intermediate
    // additions in this isomorphic-curve path (~18 ns × 52 calls ≈ 936 ns saved).
    // Dedicated to ellswift_xdh — scalar_mul_jac (CT signing) keeps HAMBURG=false.
    CTJacobianPoint R;
    CTAffinePoint t;

    {
        int const group = static_cast<int>(GROUPS) - 1;
        std::uint64_t const bits1 = scalar_window(v1, static_cast<std::size_t>(group) * GROUP_SIZE, GROUP_SIZE);
        std::uint64_t const bits2 = scalar_window(v2, static_cast<std::size_t>(group) * GROUP_SIZE, GROUP_SIZE);
        table_lookup_core<false>(&t, pre_a, TABLE_SIZE, bits1, GROUP_SIZE);
        R.x = t.x; R.y = t.y; R.z = FE52::one(); R.infinity = 0;
        table_lookup_core<false>(&t, pre_a_lam, TABLE_SIZE, bits2, GROUP_SIZE);
        unified_add_core<true, true>(&R, R, t);   // HAMBURG: K_CONST → no degenerate
    }

    #ifdef __clang__
    #pragma clang loop unroll(disable)
    #endif
    for (int group = static_cast<int>(GROUPS) - 2; group >= 0; --group) {
        std::uint64_t const bits1 = scalar_window(v1, static_cast<std::size_t>(group) * GROUP_SIZE, GROUP_SIZE);
        std::uint64_t const bits2 = scalar_window(v2, static_cast<std::size_t>(group) * GROUP_SIZE, GROUP_SIZE);
        point_dbl_n_core(&R, GROUP_SIZE);
        table_lookup_core<false>(&t, pre_a, TABLE_SIZE, bits1, GROUP_SIZE);
        unified_add_core<false, true>(&R, R, t);  // HAMBURG: K_CONST → no degenerate
        table_lookup_core<false>(&t, pre_a_lam, TABLE_SIZE, bits2, GROUP_SIZE);
        unified_add_core<false, true>(&R, R, t);  // HAMBURG: K_CONST → no degenerate
    }

    R.z.mul_assign(global_z);
    return R;
}

static CTJacobianPoint scalar_mul_jac(const Point& p, const Scalar& k) noexcept {
    constexpr unsigned GROUP_SIZE = 5;
    constexpr unsigned TABLE_SIZE = 1u << (GROUP_SIZE - 1);
    constexpr unsigned GROUPS = 26;

    static const Scalar K_CONST = Scalar::from_limbs({
        0xb5c2c1dcde9798d9ULL, 0x589ae84826ba29e4ULL,
        0xc2bdd6bf7c118d6bULL, 0xa4e88a7dcb13034eULL
    });

    if (p.is_infinity()) return CTJacobianPoint::make_infinity();

    Scalar s = scalar_add(k, K_CONST);
    s = scalar_half(s);
    auto [k1_abs, k2_abs, k1_neg, k2_neg] = ct_glv_decompose(s);

    Scalar const v1 = ct_glv_make_v(k1_abs, k1_neg);
    Scalar const v2 = ct_glv_make_v(k2_abs, k2_neg);

    CTAffinePoint pre_a[TABLE_SIZE];
    CTAffinePoint pre_a_lam[TABLE_SIZE];

    JacFE52 iso[TABLE_SIZE];
    FE52 zr[TABLE_SIZE];
    FE52 global_z;

#if defined(REPSEARCH_COZ_TABLE) && REPSEARCH_COZ_TABLE == 1
    // ---- co-Z chain (EXPERIMENT ONLY) -----------------------------------
    // Third of the four sites. Same shared-global-Z contract and the same
    // backward sweep below; the accumulator and the constant 2P are kept on one
    // Z, so the four heavy operations jac_add_ge_var_zr spends bridging two Z
    // values never happen, and the iso-curve mapping is not needed at all.
    // 177 -> 112 heavy field ops on the table build, -36.7%.
    {
        FE52 dX, dY, aX, aY, chainZ;
        jac52_dblu_ct(p.X52(), p.Y52(), p.Z52(), dX, dY, aX, aY, chainZ);

        iso[0] = { aX, aY, chainZ };
        zr[0] = FE52::one();   // never read by the sweep

        for (std::size_t i = 1; i < TABLE_SIZE; ++i) {
            FE52 sx, sy;
            // (d, acc): d FIRST, so ZADDU hands it back on the new Z for free.
            if (!jac52_zaddu_ct(dX, dY, aX, aY, chainZ, sx, sy, zr[i])) {
                // X1 == X2 -- ZADDU is undefined. Cannot happen for a chain of
                // odd multiples of a valid point, but fail closed rather than
                // return a wrong point: this function has no validity channel,
                // so use the same infinity return its own guard above uses.
                return CTJacobianPoint::make_infinity();
            }
            aX = sx;
            aY = sy;
            iso[i] = { aX, aY, chainZ };
        }
        global_z = chainZ;   // no isomorphism to undo
    }
#else
    Point p2 = p;
    p2.dbl_inplace();
    FE52 const C  = p2.Z52();
    FE52 const C2 = C.square();
    FE52 const C3 = C2 * C;
    FE52 const d_x = p2.X52();
    FE52 const d_y = p2.Y52();

    iso[0] = { p.X52() * C2, p.Y52() * C3, p.Z52() };

    for (std::size_t i = 1; i < TABLE_SIZE; ++i)
        iso[i] = jac_add_ge_var_zr(iso[i - 1], d_x, d_y, &zr[i]);

    global_z = iso[TABLE_SIZE - 1].z * C;
#endif  // REPSEARCH_COZ_TABLE
    const FE52& beta = get_beta_fe52();

    pre_a[TABLE_SIZE - 1].x = iso[TABLE_SIZE - 1].x;
    pre_a[TABLE_SIZE - 1].y = iso[TABLE_SIZE - 1].y;
    pre_a[TABLE_SIZE - 1].y.normalize_weak();
    pre_a[TABLE_SIZE - 1].infinity = 0;
    pre_a_lam[TABLE_SIZE - 1].x = pre_a[TABLE_SIZE - 1].x * beta;
    pre_a_lam[TABLE_SIZE - 1].y = pre_a[TABLE_SIZE - 1].y;
    pre_a_lam[TABLE_SIZE - 1].infinity = 0;

    FE52 zs_acc = zr[TABLE_SIZE - 1];
    for (int i = static_cast<int>(TABLE_SIZE) - 2; i >= 0; --i) {
        FE52 const zs2 = zs_acc.square();
        FE52 const zs3 = zs2 * zs_acc;
        pre_a[i].x = iso[i].x * zs2;
        pre_a[i].y = iso[i].y * zs3;
        pre_a[i].infinity = 0;
        pre_a_lam[i].x = pre_a[i].x * beta;
        pre_a_lam[i].y = pre_a[i].y;
        pre_a_lam[i].infinity = 0;
        if (i > 0) zs_acc = zs_acc * zr[i];
    }

    SECP256K1_DECLASSIFY(pre_a, sizeof(pre_a));
    SECP256K1_DECLASSIFY(pre_a_lam, sizeof(pre_a_lam));

    CTJacobianPoint R;
    CTAffinePoint t;

    {
        int const group = static_cast<int>(GROUPS) - 1;
        std::uint64_t const bits1 = scalar_window(v1, static_cast<std::size_t>(group) * GROUP_SIZE, GROUP_SIZE);
        std::uint64_t const bits2 = scalar_window(v2, static_cast<std::size_t>(group) * GROUP_SIZE, GROUP_SIZE);
        table_lookup_core<false>(&t, pre_a, TABLE_SIZE, bits1, GROUP_SIZE);
        R.x = t.x; R.y = t.y; R.z = FE52::one(); R.infinity = 0;
        table_lookup_core<false>(&t, pre_a_lam, TABLE_SIZE, bits2, GROUP_SIZE);
        unified_add_core<true>(&R, R, t);
    }

    #ifdef __clang__
    #pragma clang loop unroll(disable)
    #endif
    for (int group = static_cast<int>(GROUPS) - 2; group >= 0; --group) {
        std::uint64_t const bits1 = scalar_window(v1, static_cast<std::size_t>(group) * GROUP_SIZE, GROUP_SIZE);
        std::uint64_t const bits2 = scalar_window(v2, static_cast<std::size_t>(group) * GROUP_SIZE, GROUP_SIZE);
        point_dbl_n_core(&R, GROUP_SIZE);
        table_lookup_core<false>(&t, pre_a, TABLE_SIZE, bits1, GROUP_SIZE);
        unified_add_core<false>(&R, R, t);
        table_lookup_core<false>(&t, pre_a_lam, TABLE_SIZE, bits2, GROUP_SIZE);
        unified_add_core<false>(&R, R, t);
    }

    R.z.mul_assign(global_z);
    return R;
}

Point scalar_mul(const Point& p, const Scalar& k) noexcept {
    // Delegate to the Jacobian-output variant, then normalize.
    CTJacobianPoint R = scalar_mul_jac(p, k);
    Point result = R.to_point();
    SECP256K1_DECLASSIFY(&result, sizeof(result));
    return result;
}

// --- CT X-Only ECDH: ecmult_const_xonly (port of libsecp's xonly variant) ----
//
// Computes the x-coordinate of q * P where P has x-coordinate xn/xd on secp256k1,
// WITHOUT any sqrt computation. Port of libsecp256k1 secp256k1_ecmult_const_xonly.
//
// Algorithm:
//   1. g = xn³ + 7*xd³                        (field arithmetic only)
//   2. P_eff = (g*xn, g²)                      (effective affine, on isomorphic curve)
//   3. R = scalar_mul_jac(P_eff, q)             (CT Jacobian multiply)
//   4. x = R.x / (R.z² * g * xd)              (one field inversion)
//
// Why this works (secp256k1 has a=0):
//   - P_eff lies on Y² = X³ + g⁶·7 (isomorphic to secp256k1 via the u=g twist)
//   - Since a=0, all doubling/addition formulas are twist-invariant
//   - The Jacobian scalar multiply computes q*P_eff on this isomorphic curve
//   - The back-mapping: x_secp = x_eff / (g * xd) after Jacobian normalization
//
// Security: q is treated as secret (CT path). P_eff is public (derived from
// the peer's ELL64 encoding), so the table build is variable-time OK.
//
// xd = FE52::one() when xn is already the full x-coordinate (not a fraction).
FieldElement ecmult_const_xonly(const FieldElement& xn_fe, const FieldElement& xd_fe,
                                 const Scalar& q) noexcept {
    FE52 xn = FE52::from_fe(xn_fe);
    FE52 xd = FE52::from_fe(xd_fe);

    // 1. g = xn³ + 7*xd³
    FE52 const xn2 = xn.square();
    FE52 const xn3 = xn2 * xn;
    FE52 const xd2 = xd.square();
    FE52 xd3       = xd2 * xd;
    xd3.mul_int_assign(7);          // 7 * xd³
    FE52 const g   = xn3 + xd3;    // g = xn³ + 7·xd³

    // 2. P_eff = (g·xn, g²) — affine point on the isomorphic curve
    FE52 const px52 = g * xn;
    FE52 const py52 = g.square();

    // 3. CT scalar multiply on effective affine point (bypasses on-curve assert).
    //    P_eff = (px52, py52) is on the isomorphic curve, NOT secp256k1.
    //    scalar_mul_jac_fe52_z1 uses jac52_double_z1_to directly, avoiding
    //    SECP_ASSERT_ON_CURVE which would fire for non-secp256k1 points.
    CTJacobianPoint R = scalar_mul_jac_fe52_z1(px52, py52, q);

    if (R.infinity != 0) return FieldElement::zero();

    // 4. x = R.x / (R.z² * g * xd) — single combined inversion
    //    Avoids two separate inversions (vs. normalizing R then dividing by g*xd).
    //    Combined denominator: denom = R.z² * g * xd
    FE52 const rz2   = R.z.square();
    FE52       denom = rz2 * g;
    denom = denom * xd;             // R.z² * g * xd
#if defined(REPSEARCH_CT_SAFEGCD_INV) && REPSEARCH_CT_SAFEGCD_INV == 1
    // EXPERIMENT ONLY (experiments/representation_search); default build uses
    // the line in the #else branch, unchanged.
    //
    // Both of these are CONSTANT-TIME. FieldElement52::inverse() is the Fermat
    // chain: 255 squarings + 15 multiplies, a fixed operation sequence.
    // ct::field_inv is the ct:: track's Bernstein-Yang SafeGCD: 10 x 59 = 590
    // BRANCHLESS divsteps, a port of libsecp256k1's secp256k1_modinv64 (the CT
    // variant, not the _var one). Same timing guarantee, different algorithm.
    //
    // MEASURED on this machine (i5-14400F, GCC 14.2.0, -O3 -march=native, 64
    // random elements, 11 passes, minimum taken, P-core pinned; all paths
    // verified to return true inverses first):
    //     FieldElement52::inverse()   Fermat, CT      3847.5 ns
    //     ct::field_inv()             SafeGCD, CT     1555.0 ns   2.47x
    //     FE52::inverse_safegcd()     VARIABLE-TIME   1189.4 ns   (NOT usable
    //                                                 here -- q is secret)
    // The 1555 ns figure INCLUDES the FE52 <-> 4x64 conversions below.
    //
    // The Fermat chain is therefore not the price of constant time; it is
    // simply the slower of the two constant-time inverses this repository
    // already contains. See issue #398 and KB FE52-INVERSE-CT-DOMINATED.
    //
    // This inverse is NOT amortised: one per ECDH, so the full ~2.3 us lands.
    FE52 const denom_inv = FE52::from_fe(ct::field_inv(denom.to_fe()));
#else
    FE52 const denom_inv = denom.inverse();
#endif
    FE52 const x52   = R.x * denom_inv;

    return x52.to_fe();
}

// --- CT Prebuilt Tables API --------------------------------------------------

CTScalarMulTables build_scalar_mul_tables(const Point& p) noexcept {
    CTScalarMulTables out;
    out.valid = false;
#if defined(SECP256K1_FAST_52BIT) && !defined(SECP256K1_USE_4X64_POINT_OPS)
    if (p.is_infinity()) return out;

    constexpr unsigned GROUP_SIZE = 5;
    constexpr unsigned TABLE_SIZE = 1u << (GROUP_SIZE - 1);

    JacFE52 iso[TABLE_SIZE];
    FE52 zr[TABLE_SIZE];

#if defined(REPSEARCH_COZ_TABLE) && REPSEARCH_COZ_TABLE == 1
    // ---- co-Z chain (EXPERIMENT ONLY) -----------------------------------
    // Same table, same shared-global-Z contract, same backward sweep below.
    // The accumulator and the constant 2P are kept on ONE Z, so the four heavy
    // operations jac_add_ge_var_zr spends bridging two Z values never happen,
    // and the iso-curve mapping (C, C^2, C^3 plus two multiplies) is not needed
    // at all -- the chain works on the curve directly.
    FE52 dX, dY, aX, aY, chainZ;
    jac52_dblu_ct(p.X52(), p.Y52(), p.Z52(), dX, dY, aX, aY, chainZ);

    iso[0] = { aX, aY, chainZ };
    zr[0] = FE52::one();   // never read by the sweep

    for (std::size_t i = 1; i < TABLE_SIZE; ++i) {
        FE52 sx, sy;
        // (d, acc): d FIRST, so ZADDU hands it back on the new Z for free.
        if (!jac52_zaddu_ct(dX, dY, aX, aY, chainZ, sx, sy, zr[i])) return out;
        aX = sx;
        aY = sy;
        iso[i] = { aX, aY, chainZ };
    }

    // No isomorphism to undo: the shared Z is the Z the chain ended on.
    out.global_z = chainZ;
#else
    Point p2 = p;
    p2.dbl_inplace();
    FE52 const C  = p2.Z52();
    FE52 const C2 = C.square();
    FE52 const C3 = C2 * C;
    FE52 const d_x = p2.X52();
    FE52 const d_y = p2.Y52();

    iso[0] = { p.X52() * C2, p.Y52() * C3, p.Z52() };
    for (std::size_t i = 1; i < TABLE_SIZE; ++i)
        iso[i] = jac_add_ge_var_zr(iso[i - 1], d_x, d_y, &zr[i]);

    out.global_z = iso[TABLE_SIZE - 1].z * C;
#endif  // REPSEARCH_COZ_TABLE
    const FE52& beta = get_beta_fe52();

    out.pre_a[TABLE_SIZE - 1].x = iso[TABLE_SIZE - 1].x;
    out.pre_a[TABLE_SIZE - 1].y = iso[TABLE_SIZE - 1].y;
    out.pre_a[TABLE_SIZE - 1].y.normalize_weak();
    out.pre_a[TABLE_SIZE - 1].infinity = 0;
    out.pre_a_lam[TABLE_SIZE - 1].x = out.pre_a[TABLE_SIZE - 1].x * beta;
    out.pre_a_lam[TABLE_SIZE - 1].y = out.pre_a[TABLE_SIZE - 1].y;
    out.pre_a_lam[TABLE_SIZE - 1].infinity = 0;

    FE52 zs_acc = zr[TABLE_SIZE - 1];
    for (int i = static_cast<int>(TABLE_SIZE) - 2; i >= 0; --i) {
        FE52 const zs2 = zs_acc.square();
        FE52 const zs3 = zs2 * zs_acc;
        out.pre_a[i].x = iso[i].x * zs2;
        out.pre_a[i].y = iso[i].y * zs3;
        out.pre_a[i].infinity = 0;
        out.pre_a_lam[i].x = out.pre_a[i].x * beta;
        out.pre_a_lam[i].y = out.pre_a[i].y;
        out.pre_a_lam[i].infinity = 0;
        if (i > 0) zs_acc = zs_acc * zr[i];
    }
    out.valid = true;
#endif
    return out;
}

Point scalar_mul_prebuilt(const CTScalarMulTables& tables,
                           const Point& p_fallback,
                           const Scalar& k) noexcept {
#if defined(SECP256K1_FAST_52BIT) && !defined(SECP256K1_USE_4X64_POINT_OPS)
    if (!tables.valid) return scalar_mul(p_fallback, k);

    constexpr unsigned GROUP_SIZE = 5;
    constexpr unsigned TABLE_SIZE = CTScalarMulTables::TABLE_SIZE;
    constexpr unsigned GROUPS = 26;

    static const Scalar K_CONST = Scalar::from_limbs({
        0xb5c2c1dcde9798d9ULL, 0x589ae84826ba29e4ULL,
        0xc2bdd6bf7c118d6bULL, 0xa4e88a7dcb13034eULL
    });

    Scalar s = scalar_add(k, K_CONST);
    s = scalar_half(s);
    auto [k1_abs, k2_abs, k1_neg, k2_neg] = ct_glv_decompose(s);

    Scalar const v1 = ct_glv_make_v(k1_abs, k1_neg);
    Scalar const v2 = ct_glv_make_v(k2_abs, k2_neg);

    CTJacobianPoint R;
    CTAffinePoint t;
    {
        int const group = static_cast<int>(GROUPS) - 1;
        std::uint64_t const bits1 = scalar_window(v1, static_cast<std::size_t>(group)*GROUP_SIZE, GROUP_SIZE);
        std::uint64_t const bits2 = scalar_window(v2, static_cast<std::size_t>(group)*GROUP_SIZE, GROUP_SIZE);
        table_lookup_core<false>(&t, tables.pre_a, TABLE_SIZE, bits1, GROUP_SIZE);
        R.x = t.x; R.y = t.y; R.z = FE52::one(); R.infinity = 0;
        table_lookup_core<false>(&t, tables.pre_a_lam, TABLE_SIZE, bits2, GROUP_SIZE);
        unified_add_core<true>(&R, R, t);
    }
    #ifdef __clang__
    #pragma clang loop unroll(disable)
    #endif
    for (int group = static_cast<int>(GROUPS) - 2; group >= 0; --group) {
        std::uint64_t const bits1 = scalar_window(v1, static_cast<std::size_t>(group)*GROUP_SIZE, GROUP_SIZE);
        std::uint64_t const bits2 = scalar_window(v2, static_cast<std::size_t>(group)*GROUP_SIZE, GROUP_SIZE);
        point_dbl_n_core(&R, GROUP_SIZE);
        table_lookup_core<false>(&t, tables.pre_a, TABLE_SIZE, bits1, GROUP_SIZE);
        unified_add_core<false>(&R, R, t);
        table_lookup_core<false>(&t, tables.pre_a_lam, TABLE_SIZE, bits2, GROUP_SIZE);
        unified_add_core<false>(&R, R, t);
    }

    R.z.mul_assign(tables.global_z);
    Point result = R.to_point();
    SECP256K1_DECLASSIFY(&result, sizeof(result));
    return result;
#else
    (void)tables;
    return scalar_mul(p_fallback, k);
#endif
}

Point scalar_mul_prebuilt_fast(const CTScalarMulTables& tables,
                                const Point& p_fallback,
                                const Scalar& k) noexcept {
#if defined(SECP256K1_FAST_52BIT) && !defined(SECP256K1_USE_4X64_POINT_OPS)
    if (!tables.valid) return scalar_mul(p_fallback, k);

    constexpr unsigned GROUP_SIZE = 5;
    constexpr unsigned TABLE_SIZE = CTScalarMulTables::TABLE_SIZE;
    constexpr unsigned GROUPS = 26;

    static const Scalar K_CONST = Scalar::from_limbs({
        0xb5c2c1dcde9798d9ULL, 0x589ae84826ba29e4ULL,
        0xc2bdd6bf7c118d6bULL, 0xa4e88a7dcb13034eULL
    });

    Scalar s = scalar_add(k, K_CONST);
    s = scalar_half(s);
    auto [k1_abs, k2_abs, k1_neg, k2_neg] = ct_glv_decompose(s);
    Scalar const v1 = ct_glv_make_v(k1_abs, k1_neg);
    Scalar const v2 = ct_glv_make_v(k2_abs, k2_neg);

    CTJacobianPoint R;
    CTAffinePoint t;
    {
        int const group = static_cast<int>(GROUPS) - 1;
        std::uint64_t const bits1 = scalar_window(v1, static_cast<std::size_t>(group)*GROUP_SIZE, GROUP_SIZE);
        std::uint64_t const bits2 = scalar_window(v2, static_cast<std::size_t>(group)*GROUP_SIZE, GROUP_SIZE);
        table_lookup_core<false>(&t, tables.pre_a, TABLE_SIZE, bits1, GROUP_SIZE);
        R.x = t.x; R.y = t.y; R.z = FE52::one(); R.infinity = 0;
        table_lookup_core<false>(&t, tables.pre_a_lam, TABLE_SIZE, bits2, GROUP_SIZE);
        unified_add_core<true>(&R, R, t);  // first group: keep CHECK_INFINITY for safety
    }
    #ifdef __clang__
    #pragma clang loop unroll(disable)
    #endif
    for (int group = static_cast<int>(GROUPS) - 2; group >= 0; --group) {
        std::uint64_t const bits1 = scalar_window(v1, static_cast<std::size_t>(group)*GROUP_SIZE, GROUP_SIZE);
        std::uint64_t const bits2 = scalar_window(v2, static_cast<std::size_t>(group)*GROUP_SIZE, GROUP_SIZE);
        point_dbl_n_core(&R, GROUP_SIZE);
        // Incomplete 7M+3S mixed-add: safe for BIP-324 ECDH (random peer key).
        // Degenerate case probability ~2^-128; wrong result → handshake fail, not key leak.
        table_lookup_core<false>(&t, tables.pre_a, TABLE_SIZE, bits1, GROUP_SIZE);
        add_affine_fast_ct(&R, R, t);
        table_lookup_core<false>(&t, tables.pre_a_lam, TABLE_SIZE, bits2, GROUP_SIZE);
        add_affine_fast_ct(&R, R, t);
    }

    R.z.mul_assign(tables.global_z);
    Point result = R.to_point();
    SECP256K1_DECLASSIFY(&result, sizeof(result));
    return result;
#else
    (void)tables;
    return scalar_mul(p_fallback, k);
#endif
}

// --- CT Generator Multiplication (5x52) -- Comb Method -----------------------
// Uses signed-digit multi-comb (adapted from bitcoin-core/secp256k1).
//
// Parameters: COMB_TEETH=6, COMB_BLOCKS=11, COMB_SPACING=4, with the LAST
// block short: blocks 0..9 carry 6 teeth, block 10 carries 4.  That makes
// COMB_BITS = 10*6*4 + 4*4 = 256 exactly, so every bit of v is covered once
// and no bit beyond 255 is ever addressed.
//
// Hamburg encoding: v = (k + K_gen) / 2 mod n, bits of v become {+1,-1} signs.
// Outer loop over COMB_SPACING (4 iterations with 3 doublings between them).
// Inner loop over the 10 six-tooth blocks plus the short tail block.
//
// For block b at spacing offset s, comb digit gathers bits at positions:
//   tooth t -> (b * COMB_TEETH + t) * COMB_SPACING + s
// The short block keeps that same base index (b * COMB_TEETH), so its four
// teeth land on 240..255 and the block set tiles 0..255 exactly.
// Each digit -> teeth-bit unsigned -> signed table lookup (2^(teeth-1) entries).
//
// The identity sum_{i<256} 2^i = 2^256 - 1 = K_gen is what makes the comb
// exact: sum (2*bit_i - 1)*2^i*G = (2v - (2^256-1))*G = k*G.  A comb wider
// than 256 bits would leave a residue and need a correction point added at the
// end; at exactly 256 bits there is nothing left to correct.
//
// Runtime: 43 additions + 44 lookups + 3 doublings, no correction.
// Table: 10 blocks x 32 entries + 1 block x 8 entries = 328 affine points.

namespace {

constexpr unsigned COMB_TEETH   = 6;
constexpr unsigned COMB_BLOCKS  = 11;    // 11 blocks with spacing 4
constexpr unsigned COMB_SPACING = 4;
// The last block is short so the comb spans exactly 256 bits.
constexpr unsigned COMB_FULL_BLOCKS = COMB_BLOCKS - 1;   // 10 six-tooth blocks
constexpr unsigned COMB_TAIL_TEETH  = 4;                 // block 10 has 4 teeth
constexpr unsigned COMB_BITS =
    COMB_FULL_BLOCKS * COMB_TEETH * COMB_SPACING + COMB_TAIL_TEETH * COMB_SPACING;
static_assert(COMB_BITS == 256,
              "comb must tile exactly 256 bits: K_gen = 2^256-1 assumes it");
constexpr std::size_t COMB_TABLE_SIZE      = 1u << (COMB_TEETH - 1);       // 32
constexpr std::size_t COMB_TAIL_TABLE_SIZE = 1u << (COMB_TAIL_TEETH - 1);  // 8

// Hamburg constant: K_gen = (2^256 - 1) mod n
static constexpr std::uint64_t K_GEN[4] = {
    0x402DA1732FC9BEBEULL,
    0x4551231950B75FC4ULL,
    0x0000000000000001ULL,
    0x0000000000000000ULL
};

struct alignas(64) CombGenTable {
    CTAffinePoint entries[COMB_FULL_BLOCKS][COMB_TABLE_SIZE];
    CTAffinePoint tail[COMB_TAIL_TABLE_SIZE];  // short last block
    // Note: no default member initializer — BSS guarantees zero (= false) at startup.
    // Default member init would generate a global constructor on MCUs.
    bool initialized;
};

static CombGenTable g_comb_table;
static std::once_flag g_comb_table_once;

// -- Extract a comb digit from scattered bit positions ------------------------
// For block b at spacing offset comb_off: gather bit at position
//   (b * COMB_TEETH + tooth) * COMB_SPACING + comb_off
// for each of the block's teeth.  `teeth` is COMB_TEETH for blocks 0..9 and
// COMB_TAIL_TEETH for the short last block — a property of the block index,
// which is a public loop counter.
inline std::uint64_t extract_comb_digit(const Scalar& v,
                                         unsigned block,
                                         unsigned comb_off,
                                         unsigned teeth) noexcept {
    // The teeth of one (block, comb_off) digit sit at base, base+4, ...,
    // spanning (teeth-1)*4 + 1 contiguous bit positions across at most two
    // limbs, so a single window read replaces `teeth` scalar_bit calls.
    // scalar_bit rescans all four limbs behind a mask because it assumes the
    // index may be secret; here block, comb_off and teeth are all loop-derived,
    // so base is public.  Only the bits returned are secret and they are still
    // combined branchlessly.  The highest position read is 243 + 12 = 255, so
    // the comb never addresses a bit outside the scalar.
    unsigned const span = (teeth - 1) * COMB_SPACING + 1;
    std::size_t const base = static_cast<std::size_t>(block) * COMB_TEETH
                             * COMB_SPACING + comb_off;
    std::uint64_t const window = scalar_window(v, base, span);

    std::uint64_t digit = 0;
    for (unsigned tooth = 0; tooth < teeth; ++tooth) {
        digit |= ((window >> (tooth * COMB_SPACING)) & 1) << tooth;
    }
    return digit;
}

// -- CT lookup from a signed comb table ---------------------------------------
// A TEETH-tooth block stores 2^(TEETH-1) entries, those for unsigned values
// with the top bit set.  For values with the top bit clear: look up the
// complement index and negate Y.
//
// Identity: table_u[d] = -table_u[(2^TEETH - 1) - d]  (complementary signs)
// table_s[idx] = table_u[idx | 2^(TEETH-1)]
//
// top bit 1: idx = d & (SIZE-1),        point =  table_s[idx]
// top bit 0: idx = (SIZE-1) - (d&SIZE-1), point = -table_s[idx]
//
// TEETH is a template parameter because it is fixed per block index (a public
// loop counter), so the scan length is the same for every scalar: the timing
// pattern is 32,32,...,32,8 per outer iteration regardless of the secret.
//
// AVX2 path: same XOR/AND blending technique as table_lookup_core.
// The scan READS 80 bytes per entry (x.n[5] + y.n[5] = 10 x 8) but STRIDES 88,
// which is sizeof(CTAffinePoint) including the trailing `infinity` word the comb
// never uses. Both numbers are correct and they are not the same number; the
// table-size comments elsewhere in this file use the 88 B stride.
// Measured: padding the stride to 120 and 152 B moved ct::generator_mul by
// +0.10%, so the scan is compute-bound, not traffic-bound, and shrinking the
// entry would buy nothing (knowledge base CT-COMB-NOT-MEMORY-BOUND).
// Layout fits in 2.5 ymm256 registers: r0 (x.n[0..3]), r1 (x.n[4], y.n[0..2]),
// r2_128 (y.n[3..4], 16 bytes).
template <unsigned TEETH>
static inline
void comb_lookup(CTAffinePoint* out,
                  const CTAffinePoint* table,
                  std::uint64_t digit) noexcept {
    constexpr std::size_t TABLE_SIZE = std::size_t{1} << (TEETH - 1);
    std::uint64_t const top = (digit >> (TEETH - 1)) & 1;
    std::uint64_t const needs_negate = top ^ 1;  // negate if top bit is 0
    auto const neg_mask = static_cast<std::uint64_t>(
                      -static_cast<std::int64_t>(needs_negate));

    std::uint64_t const d_lo = digit & (TABLE_SIZE - 1);

    // index: top=1 -> d_lo; top=0 -> (TABLE_SIZE-1 - d_lo)
    std::uint64_t const idx_pos = d_lo;
    std::uint64_t const idx_neg = (TABLE_SIZE - 1) - d_lo;
    // Branchless select
    std::uint64_t const index = idx_pos ^ ((idx_pos ^ idx_neg) & neg_mask);

#if SECP256K1_CT_AVX2
    // AVX2 vectorized CT linear scan over all TABLE_SIZE entries.
    // Mirrors table_lookup_core: load entry 0 into accumulators, then
    // for m=1..TABLE_SIZE-1: eq_mask(m, index), broadcast to ymm, XOR/AND blend.
    // Aliasing: __m256i has __may_alias__ in GCC/Clang immintrin.h -- safe.
    {
        const auto* base0 = reinterpret_cast<const char*>(&table[0].x.n[0]);
        __m256i r0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base0));       // x.n[0..3]
        __m256i r1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base0 + 32));  // x.n[4], y.n[0..2]
        __m128i r2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(base0 + 64));     // y.n[3..4]

        for (std::size_t m = 1; m < TABLE_SIZE; ++m) {
            std::uint64_t const eq = eq_mask(static_cast<std::uint64_t>(m), index);
            __m256i const vmask   = _mm256_set1_epi64x(static_cast<long long>(eq));
            __m128i const vmask128 = _mm_set1_epi64x(static_cast<long long>(eq));

            const auto* base_m = reinterpret_cast<const char*>(&table[m].x.n[0]);
            __m256i const s0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base_m));
            __m256i const s1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base_m + 32));
            __m128i const s2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(base_m + 64));

            // CT blend: r = (r ^ s) & mask ^ r  ≡  mask ? s : r
            r0 = _mm256_xor_si256(r0, _mm256_and_si256(_mm256_xor_si256(r0, s0), vmask));
            r1 = _mm256_xor_si256(r1, _mm256_and_si256(_mm256_xor_si256(r1, s1), vmask));
            r2 = _mm_xor_si128(r2, _mm_and_si128(_mm_xor_si128(r2, s2), vmask128));
        }

        // Store result (80 bytes)
        auto* dst = reinterpret_cast<char*>(&out->x.n[0]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), r0);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 32), r1);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 64), r2);
    }
#else
    // CT linear scan over all TABLE_SIZE entries (scalar fallback)
    out->x = table[0].x;
    out->y = table[0].y;
    for (std::size_t m = 1; m < TABLE_SIZE; ++m) {
        std::uint64_t const mask = eq_mask(static_cast<std::uint64_t>(m), index);
        fe52_cmov(&out->x, table[m].x, mask);
        fe52_cmov(&out->y, table[m].y, mask);
    }
#endif // SECP256K1_CT_AVX2

    // Conditional Y-negate (scalar -- runs once per lookup)
    FE52 neg_y = out->y;
    neg_y = neg_y.negate(1);
    fe52_cmov(&out->y, neg_y, neg_mask);

    out->infinity = 0;
}

// -- Build one block's signed table -------------------------------------------
// entry[idx] = P_{TEETH-1} + sum_{j < TEETH-1} (2*bit_j(idx) - 1) * P_j,
// where P_t = bases[base_off + t].  Shared by the six-tooth blocks and the
// four-tooth tail so both geometries come from one piece of code.
template <unsigned TEETH>
void build_comb_block(CTAffinePoint* dst, const Point* bases,
                       unsigned base_off) noexcept {
    constexpr std::size_t TABLE_SIZE = std::size_t{1} << (TEETH - 1);

    // Convert teeth base points to affine for efficient construction
    FE52 zs[TEETH], z_invs[TEETH];
    for (unsigned t = 0; t < TEETH; ++t) {
        zs[t] = bases[base_off + t].Z52();
    }
    fe52_batch_inverse(z_invs, zs, TEETH);

    FE52 aff_x[TEETH], aff_y[TEETH];
    for (unsigned t = 0; t < TEETH; ++t) {
        FE52 const zi2 = z_invs[t].square();
        FE52 const zi3 = zi2 * z_invs[t];
        aff_x[t] = bases[base_off + t].X52() * zi2;
        aff_y[t] = bases[base_off + t].Y52() * zi3;
        aff_x[t].normalize();
        aff_y[t].normalize();
    }

    Point entries_jac[TABLE_SIZE];

    for (std::size_t idx = 0; idx < TABLE_SIZE; ++idx) {
        // Start with +P_{TEETH-1} (always positive in the signed table)
        Point entry = Point::from_affine52(aff_x[TEETH - 1], aff_y[TEETH - 1]);

        for (unsigned j = 0; j < TEETH - 1; ++j) {
            FE52 const py = ((idx >> j) & 1) ? aff_y[j] : aff_y[j].negate(1);
            Point const pj = Point::from_affine52(aff_x[j], py);
            entry = entry.add(pj);
        }
        entries_jac[idx] = entry;
    }

    // Batch-invert Z coordinates and store as affine
    FE52 ent_zs[TABLE_SIZE], ent_zis[TABLE_SIZE];
    for (std::size_t idx = 0; idx < TABLE_SIZE; ++idx) {
        ent_zs[idx] = entries_jac[idx].Z52();
    }
    fe52_batch_inverse(ent_zis, ent_zs, TABLE_SIZE);

    for (std::size_t idx = 0; idx < TABLE_SIZE; ++idx) {
        FE52 const zi2 = ent_zis[idx].square();
        FE52 const zi3 = zi2 * ent_zis[idx];
        dst[idx].x = entries_jac[idx].X52() * zi2;
        dst[idx].y = entries_jac[idx].Y52() * zi3;
        dst[idx].x.normalize();
        dst[idx].y.normalize();
        dst[idx].infinity = 0;
    }
}

// -- Build comb table ---------------------------------------------------------
void build_comb_table() noexcept {
    Point const G = Point::generator();

    // Step 1: Compute base points for all (block, tooth) pairs.
    // For block b, tooth t: base = 2^((b*COMB_TEETH + t) * COMB_SPACING) * G
    //                             = 2^((b*6 + t)*4) * G
    // The tail block continues the same index run, so the bases are
    // 10*6 + 4 = 64 consecutive powers and cover exactly bits 0..255.
    constexpr unsigned NUM_BASES =
        COMB_FULL_BLOCKS * COMB_TEETH + COMB_TAIL_TEETH;  // 64
    static_assert(NUM_BASES * COMB_SPACING == COMB_BITS, "base run must tile COMB_BITS");
    Point bases[NUM_BASES];
    bases[0] = G;
    for (unsigned i = 1; i < NUM_BASES; ++i) {
        bases[i] = bases[i - 1];
        for (unsigned d = 0; d < COMB_SPACING; ++d) {
            bases[i].dbl_inplace();
        }
    }

    // Step 2: six-tooth blocks, then the four-tooth tail.
    for (unsigned b = 0; b < COMB_FULL_BLOCKS; ++b) {
        build_comb_block<COMB_TEETH>(g_comb_table.entries[b], bases,
                                     b * COMB_TEETH);
    }
    build_comb_block<COMB_TAIL_TEETH>(g_comb_table.tail, bases,
                                      COMB_FULL_BLOCKS * COMB_TEETH);

    g_comb_table.initialized = true;
}

} // anonymous namespace

void init_generator_table() noexcept {
    std::call_once(g_comb_table_once, build_comb_table);
}

Point generator_mul(const Scalar& k) noexcept {
    init_generator_table();

    // -- Hamburg scalar transform ------------------------------------------
    // v = (k + K_gen) / 2 mod n
    static const Scalar K_gen_scalar = Scalar::from_limbs(
        {K_GEN[0], K_GEN[1], K_GEN[2], K_GEN[3]});

    Scalar const s = scalar_add(k, K_gen_scalar);
    Scalar const v = scalar_half(s);

    CTJacobianPoint R;
    CTAffinePoint T;

    // -- Main loop: outer COMB_SPACING x inner blocks ---------------------
    // Outer loop iterates over spacing offsets from COMB_SPACING-1 down to 0.
    // Inner loop iterates over the 10 six-tooth blocks plus the short tail.
    // Between outer iterations, double the accumulator once.
    unsigned comb_off = COMB_SPACING - 1;

    // First outer iteration: first block initializes R directly
    {
        std::uint64_t const digit = extract_comb_digit(v, 0, comb_off, COMB_TEETH);
        comb_lookup<COMB_TEETH>(&T, g_comb_table.entries[0], digit);
        R.x = T.x;
        R.y = T.y;
        R.z = FE52::one();
        R.infinity = 0;
    }

    // Remaining blocks of the first outer iteration.
    // Safety: all T values come from the precomputed comb table (fixed G
    // multiples, never infinity). For a uniformly random scalar the probability
    // that R equals a table entry (which would cause the incomplete formula to
    // misbehave) is ~2^-128 — same reasoning as scalar_mul_prebuilt_fast.
    // Using the incomplete mixed-add instead of the complete unified formula
    // trades a smaller operation count for a formula that is only valid because
    // of the argument above. Counting from the code rather than from memory:
    // add_affine_fast_ct is 7M+5S and point_add_mixed_complete is 7M+5S as well
    // (both are labelled that way in bench_unified), so the saving is in the
    // unified formula's extra selects, not in five multiplications. There are
    // 44 additions per generator_mul, not ~43.
    // Block 1 still sees Z1 == 1 (R was loaded straight from the table above),
    // so it uses the Z1==1 specialisation.  The peel is on the block index,
    // which is a public loop counter.
    {
        std::uint64_t const digit = extract_comb_digit(v, 1, comb_off, COMB_TEETH);
        comb_lookup<COMB_TEETH>(&T, g_comb_table.entries[1], digit);
        add_affine_fast_ct_z1(&R, R, T);
    }
    #ifdef __clang__
    #pragma clang loop unroll(disable)
    #endif
    for (unsigned b = 2; b < COMB_FULL_BLOCKS; ++b) {
        std::uint64_t const digit = extract_comb_digit(v, b, comb_off, COMB_TEETH);
        comb_lookup<COMB_TEETH>(&T, g_comb_table.entries[b], digit);
        add_affine_fast_ct(&R, R, T);
    }
    {
        std::uint64_t const digit =
            extract_comb_digit(v, COMB_FULL_BLOCKS, comb_off, COMB_TAIL_TEETH);
        comb_lookup<COMB_TAIL_TEETH>(&T, g_comb_table.tail, digit);
        add_affine_fast_ct(&R, R, T);
    }

    // Remaining outer iterations with doubling
    while (comb_off-- > 0) {
        point_dbl_n_core(&R, 1);

        #ifdef __clang__
        #pragma clang loop unroll(disable)
        #endif
        for (unsigned b = 0; b < COMB_FULL_BLOCKS; ++b) {
            std::uint64_t const digit = extract_comb_digit(v, b, comb_off, COMB_TEETH);
            comb_lookup<COMB_TEETH>(&T, g_comb_table.entries[b], digit);
            add_affine_fast_ct(&R, R, T);
        }
        std::uint64_t const digit =
            extract_comb_digit(v, COMB_FULL_BLOCKS, comb_off, COMB_TAIL_TEETH);
        comb_lookup<COMB_TAIL_TEETH>(&T, g_comb_table.tail, digit);
        add_affine_fast_ct(&R, R, T);
    }

    Point result = R.to_point();
    SECP256K1_DECLASSIFY(&result, sizeof(result));
    return result;
}

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#else // !SECP256K1_FAST_52BIT -- Full CT for 4x64 FieldElement

// ===============================================================================
// 4x64 Constant-Time Path (MSVC / 32-bit / non-__int128 platforms)
// ===============================================================================
// Full CT implementation using ct::field_* on FieldElement (4x64 limbs).
// All arithmetic is always fully reduced -- no magnitude tracking needed.
// Supports: ESP32-S3, MSVC x86-64, ARM32, RISC-V 32-bit.
//
// Translation from FE52 (5x52 lazy reduction):
//   a * b           -> field_mul(a, b)
//   a.square()      -> field_sqr(a)
//   a + b           -> field_add(a, b)
//   a - b           -> field_sub(a, b)
//   a.negate(m)     -> field_neg(a)   (magnitude param ignored)
//   a.half()        -> field_half(a)
//   normalize_weak  -> no-op          (always normalized)
//   fe52_cmov       -> field_cmov
//   fe52_normalizes_to_zero -> field_is_zero

#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

// FE52 = FieldElement on this path (typedef from ct/point.hpp)

namespace /* 4x64 CT helpers */ {

// --- Portable 64x64->128 multiply (no __int128) ------------------------------

struct U128 { std::uint64_t lo, hi; };

inline U128 mul64(std::uint64_t u, std::uint64_t v) noexcept {
#if defined(_MSC_VER) && defined(_M_X64)
    std::uint64_t hi;
    std::uint64_t lo = _umul128(u, v, &hi);
    return {lo, hi};
#else
    // Knuth Algorithm M: 4 x 32-bit multiplies
    std::uint64_t u0 = u & 0xFFFFFFFFULL, u1 = u >> 32;
    std::uint64_t v0 = v & 0xFFFFFFFFULL, v1 = v >> 32;
    std::uint64_t t  = u0 * v0;
    std::uint64_t w0 = t & 0xFFFFFFFFULL;
    std::uint64_t k  = t >> 32;
    t = u1 * v0 + k;
    std::uint64_t w1 = t & 0xFFFFFFFFULL;
    std::uint64_t w2 = t >> 32;
    t = u0 * v1 + w1;
    return { ((t & 0xFFFFFFFFULL) << 32) | w0,
             u1 * v1 + w2 + (t >> 32) };
#endif
}

// --- 4x4 -> 8 limb schoolbook multiply ---------------------------------------

inline void mul_4x4(std::uint64_t d[8],
                    const std::uint64_t a[4],
                    const std::uint64_t b[4]) noexcept {
    for (int i = 0; i < 8; ++i) d[i] = 0;
    for (int i = 0; i < 4; ++i) {
        std::uint64_t carry = 0;
        for (int j = 0; j < 4; ++j) {
            U128 m = mul64(a[i], b[j]);
            std::uint64_t sum = d[i + j] + m.lo;
            std::uint64_t c1 = static_cast<std::uint64_t>(sum < d[i + j]);
            sum += carry;
            std::uint64_t c2 = static_cast<std::uint64_t>(sum < carry);
            d[i + j] = sum;
            carry = m.hi + c1 + c2;
        }
        d[i + 4] = carry;
    }
}

// --- Variable-time Jacobian + Affine addition (for public table build) -------
// NOT constant-time: used only for precomputation where POINT is public.
// Standard formula: 3S + 8M. Caller ensures no degenerate cases.

struct Jac4x64 { FE52 x, y, z; };

inline Jac4x64 jac_add_ge_var_zr(const Jac4x64& a,
                                  const FE52& bx, const FE52& by,
                                  FE52* zr_out) noexcept {
    FE52 z1sq = field_sqr(a.z);
    FE52 u2   = field_mul(bx, z1sq);
    FE52 z1cu = field_mul(z1sq, a.z);
    FE52 s2   = field_mul(by, z1cu);
    FE52 h    = field_sub(u2, a.x);
    FE52 r    = field_sub(s2, a.y);
    FE52 h2   = field_sqr(h);
    FE52 h3   = field_mul(h, h2);
    FE52 u1h2 = field_mul(a.x, h2);
    FE52 x3   = field_sub(field_sub(field_sqr(r), h3),
                           field_add(u1h2, u1h2));
    FE52 y3   = field_sub(field_mul(r, field_sub(u1h2, x3)),
                           field_mul(a.y, h3));
    FE52 z3   = field_mul(a.z, h);
    *zr_out = h;
    return {x3, y3, z3};
}

// --- Montgomery batch inversion (CT, 4×64 helpers) ---------------------------
// PRECONDITION: All z[i] MUST be non-zero. CT function — no zero-checks.

inline void fe_batch_inverse(FE52* inv, const FE52* z, std::size_t n) noexcept {
    if (n == 0) return;
    if (n == 1) { inv[0] = field_inv(z[0]); return; }
    inv[0] = z[0];
    for (std::size_t i = 1; i < n; ++i)
        inv[i] = field_mul(inv[i - 1], z[i]);
    FE52 acc = field_inv(inv[n - 1]);
    for (std::size_t i = n - 1; i > 0; --i) {
        inv[i] = field_mul(inv[i - 1], acc);
        acc = field_mul(acc, z[i]);
    }
    inv[0] = acc;
}

} // anonymous namespace (4x64 CT helpers)

// --- CTJacobianPoint helpers -------------------------------------------------

CTJacobianPoint CTJacobianPoint::from_point(const Point& p) noexcept {
    CTJacobianPoint r;
    r.x = p.X();
    r.y = p.Y();
    r.z = p.z();
    r.infinity = p.is_infinity() ? ~std::uint64_t(0) : 0;
    return r;
}

Point CTJacobianPoint::to_point() const noexcept {
    if (infinity != 0) return Point::infinity();
    if (field_is_zero(z)) return Point::infinity();
    return Point::from_jacobian_coords(x, y, z, false);
}

CTJacobianPoint CTJacobianPoint::make_infinity() noexcept {
    CTJacobianPoint r;
    r.x = FieldElement();
    r.y = FieldElement::one();
    r.z = FieldElement();
    r.infinity = ~std::uint64_t(0);
    return r;
}

// --- CT Conditional Operations on Points -------------------------------------

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

// --- CT Conditional Operations on Affine Points ------------------------------

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

// --- Signed-Digit Affine Table Lookup (Hamburg encoding, 4x64) ---------------

CTAffinePoint affine_table_lookup_signed(const CTAffinePoint* table,
                                          std::size_t table_size,
                                          std::uint64_t n,
                                          unsigned group_size) noexcept {
    CTAffinePoint r;
    affine_table_lookup_signed_into(&r, table, table_size, n, group_size);
    return r;
}

// --- Always-inline scalar table lookup core (4x64) ---------------------------
// NORMALIZE_Y exists for API compat with FE52 path; ignored (always normalized).

template<bool NORMALIZE_Y>
SECP256K1_INLINE
void table_lookup_core(CTAffinePoint* out,
                        const CTAffinePoint* table,
                        std::size_t table_size,
                        std::uint64_t n,
                        unsigned group_size) noexcept {
    std::uint64_t negative = ((n >> (group_size - 1)) ^ 1u) & 1u;
    std::uint64_t neg_mask = 0ULL - negative;

    unsigned index_bits = group_size - 1;
    std::uint64_t index = ((0ULL - negative) ^ n) &
                          ((1ULL << index_bits) - 1u);

    out->x = table[0].x;
    out->y = table[0].y;
    for (std::size_t m = 1; m < table_size; ++m) {
        std::uint64_t mask = eq_mask(static_cast<std::uint64_t>(m),
                                     static_cast<std::uint64_t>(index));
        field_cmov(&out->x, table[m].x, mask);
        field_cmov(&out->y, table[m].y, mask);
    }

    FE52 neg_y = field_neg(out->y);
    field_cmov(&out->y, neg_y, neg_mask);
    out->infinity = 0;
}

void affine_table_lookup_signed_into(CTAffinePoint* out,
                                      const CTAffinePoint* table,
                                      std::size_t table_size,
                                      std::uint64_t n,
                                      unsigned group_size) noexcept {
    table_lookup_core<true>(out, table, table_size, n, group_size);
}

// --- Complete Addition (Jacobian, a=0, 4x64) --------------------------------
// Brier-Joye unified addition/doubling. Handles all cases branchlessly.
// Cost: 11M + 6S.

CTJacobianPoint point_add_complete(const CTJacobianPoint& p,
                                   const CTJacobianPoint& q) noexcept {
    FE52 X1 = p.x, Y1 = p.y, Z1 = p.z;
    FE52 X2 = q.x, Y2 = q.y, Z2 = q.z;

    FE52 z1z1 = field_sqr(Z1);                                  // [sqr 1]
    FE52 z2z2 = field_sqr(Z2);                                  // [sqr 2]
    FE52 u1   = field_mul(X1, z2z2);                             // [mul 1]
    FE52 u2   = field_mul(X2, z1z1);                             // [mul 2]
    FE52 s1   = field_mul(field_mul(Y1, z2z2), Z2);             // [mul 3,4]
    FE52 s2   = field_mul(field_mul(Y2, z1z1), Z1);             // [mul 5,6]
    FE52 z     = field_mul(Z1, Z2);                              // [mul 7]

    FE52 t_val  = field_add(u1, u2);
    FE52 m_val  = field_add(s1, s2);

    FE52 rr = field_add(field_sqr(t_val),                        // [sqr 3]
                        field_mul(u1, field_neg(u2)));            // [mul 8]

    std::uint64_t degen     = field_is_zero(m_val);
    std::uint64_t not_degen = ~degen;

    FE52 rr_alt = field_add(s1, s1);
    FE52 m_alt  = field_sub(u1, u2);

    field_cmov(&rr_alt, rr,    not_degen);
    field_cmov(&m_alt,  m_val, not_degen);

    FE52 nn    = field_sqr(m_alt);                               // [sqr 4]
    FE52 q_val = field_mul(field_neg(t_val), nn);                // [mul 9]

    nn = field_sqr(nn);                                          // [sqr 5]
    field_cmov(&nn, m_val, degen);

    FE52 x3 = field_add(field_sqr(rr_alt), q_val);              // [sqr 6]
    FE52 z3 = field_mul(z, m_alt);                               // [mul 10]

    FE52 tq  = field_add(field_add(x3, x3), q_val);
    FE52 y3p = field_add(field_mul(rr_alt, tq), nn);            // [mul 11]
    FE52 y3  = field_half(field_neg(y3p));

    // Infinity handling
    std::uint64_t z3_zero = field_is_zero(z3);

    field_cmov(&x3, X2, p.infinity);
    field_cmov(&y3, Y2, p.infinity);
    field_cmov(&z3, Z2, p.infinity);

    field_cmov(&x3, X1, q.infinity);
    field_cmov(&y3, Y1, q.infinity);
    field_cmov(&z3, Z1, q.infinity);

    std::uint64_t result_inf = z3_zero & ~p.infinity & ~q.infinity;
    result_inf |= p.infinity & q.infinity;

    CTJacobianPoint result;
    result.x = x3;  result.y = y3;  result.z = z3;
    result.infinity = result_inf;
    return result;
}

// --- Mixed Jacobian+Affine Complete Addition (a=0, 4x64) ---------------------
// Brier-Joye 7M + 5S. Handles all cases branchlessly.

CTJacobianPoint point_add_mixed_complete(const CTJacobianPoint& p,
                                          const CTAffinePoint& q) noexcept {
    FE52 X1 = p.x; X1.normalize_weak();
    FE52 Y1 = p.y; Y1.normalize_weak();
    FE52 Z1 = p.z;

    FE52 zz     = field_sqr(Z1);                                // [sqr 1]
    FE52 u1     = X1;
    FE52 u2     = field_mul(q.x, zz);                           // [mul 1]
    FE52 s1     = Y1;
    FE52 s2     = field_mul(field_mul(q.y, zz), Z1);            // [mul 2,3]

    FE52 t_val  = field_add(u1, u2);
    FE52 m_val  = field_add(s1, s2);

    FE52 rr     = field_add(field_sqr(t_val),                    // [sqr 2]
                            field_mul(u1, field_neg(u2)));        // [mul 4]

    std::uint64_t degen     = field_is_zero(m_val);
    std::uint64_t not_degen = ~degen;

    FE52 rr_alt = field_add(s1, s1);
    FE52 m_alt  = field_sub(u1, u2);

    field_cmov(&rr_alt, rr,    not_degen);
    field_cmov(&m_alt,  m_val, not_degen);

    FE52 nn     = field_sqr(m_alt);                              // [sqr 3]
    FE52 q_val  = field_mul(field_neg(t_val), nn);               // [mul 5]

    nn = field_sqr(nn);                                          // [sqr 4]
    field_cmov(&nn, m_val, degen);

    FE52 x3     = field_add(field_sqr(rr_alt), q_val);          // [sqr 5]
    FE52 z3     = field_mul(m_alt, Z1);                          // [mul 6]

    FE52 tq     = field_add(field_add(x3, x3), q_val);
    FE52 y3p    = field_add(field_mul(rr_alt, tq), nn);         // [mul 7]
    FE52 y3     = field_half(field_neg(y3p));

    // Infinity handling
    std::uint64_t z3_zero = field_is_zero(z3);
    FE52 one_fe = FieldElement::one();

    field_cmov(&x3, q.x,   p.infinity);
    field_cmov(&y3, q.y,   p.infinity);
    field_cmov(&z3, one_fe, p.infinity);

    field_cmov(&x3, X1, q.infinity);
    field_cmov(&y3, Y1, q.infinity);
    field_cmov(&z3, Z1, q.infinity);

    std::uint64_t result_inf = z3_zero & ~p.infinity & ~q.infinity;
    result_inf |= p.infinity & q.infinity;

    CTJacobianPoint result;
    result.x = x3;  result.y = y3;  result.z = z3;
    result.infinity = result_inf;
    return result;
}

// --- CT Point Doubling (4x64) ------------------------------------------------
// 3M + 4S + 1half. Formula: L=(3/2)X^2, S=Y^2, T=-XS, X3=L^2+2T,
// Y3=-(L(X3+T)+S^2), Z3=YZ.

CTJacobianPoint point_dbl(const CTJacobianPoint& p) noexcept {
    FE52 X = p.x, Y = p.y, Z = p.z;

    FE52 s  = field_sqr(Y);                                     // [sqr 1]
    FE52 l  = field_sqr(X);                                     // [sqr 2]
    l = field_half(field_add(field_add(l, l), l));               // (3/2)X^2
    FE52 t  = field_mul(field_neg(s), X);                        // [mul 1]
    FE52 z3 = field_mul(Z, Y);                                   // [mul 2]
    FE52 x3 = field_add(field_add(field_sqr(l), t), t);         // [sqr 3]
    s = field_sqr(s);                                            // [sqr 4]
    FE52 y3 = field_neg(
        field_add(field_mul(field_add(t, x3), l), s));           // [mul 3]

    CTJacobianPoint inf = CTJacobianPoint::make_infinity();
    CTJacobianPoint result;
    result.x = x3;  result.y = y3;  result.z = z3;
    result.infinity = 0;
    point_cmov(&result, inf, p.infinity);
    return result;
}

// --- Incomplete Mixed Jacobian+Affine Addition (4x64, no degenerate check) ---
// Standard 7M+3S Jacobian+Affine formula. SAFE ONLY when table entries are
// fixed G multiples (generator comb). For random user-supplied points use
// unified_add_core (complete formula). See add_affine_fast_ct for 52-bit twin.
SECP256K1_INLINE
void add_affine_fast_ct_4x64(CTJacobianPoint* out,
                               const CTJacobianPoint& a,
                               const CTAffinePoint& b) noexcept {
    FE52 const Z1Z1 = field_sqr(a.z);                    // [1S]
    FE52 const U2   = field_mul(b.x, Z1Z1);              // [1M]
    FE52       S2   = field_mul(b.y, Z1Z1);
    S2 = field_mul(S2, a.z);                              // [2M]

    FE52 const H    = field_sub(U2, a.x);
    FE52 const R    = field_sub(S2, a.y);

    FE52 const Z3   = field_mul(H, a.z);                 // [1M]
    FE52 const HH   = field_sqr(H);                      // [1S]
    FE52 const HHH  = field_mul(HH, H);                  // [1M]
    FE52 const U1HH = field_mul(a.x, HH);                // [1M]

    FE52 const RR   = field_sqr(R);                      // [1S]
    FE52 X3 = field_sub(field_sub(RR, HHH),
                        field_add(U1HH, U1HH));
    FE52 Y3 = field_sub(field_mul(field_sub(U1HH, X3), R),
                        field_mul(a.y, HHH));             // [2M]

    out->x = X3;
    out->y = Y3;
    out->z = Z3;
    out->infinity = 0;
}

// --- Same incomplete mixed add with Z1 == 1 (4x64) --------------------------
// Z1Z1, U2, S2 and Z3 collapse to identities exactly as in the 52-bit twin.
// 4x64 arithmetic is always fully reduced, so there is no magnitude
// bookkeeping and Z3 = H needs no normalization.
SECP256K1_INLINE
void add_affine_fast_ct_4x64_z1(CTJacobianPoint* out,
                                  const CTJacobianPoint& a,
                                  const CTAffinePoint& b) noexcept {
    FE52 const H    = field_sub(b.x, a.x);   // U2 = b.x, Z1 == 1
    FE52 const R    = field_sub(b.y, a.y);   // S2 = b.y, Z1 == 1

    FE52 const Z3   = H;                      // Z3 = H*Z1, Z1 == 1
    FE52 const HH   = field_sqr(H);          // [1S]
    FE52 const HHH  = field_mul(HH, H);      // [1M]
    FE52 const U1HH = field_mul(a.x, HH);    // [1M]

    FE52 const RR   = field_sqr(R);          // [1S]
    FE52 X3 = field_sub(field_sub(RR, HHH),
                        field_add(U1HH, U1HH));
    FE52 Y3 = field_sub(field_mul(field_sub(U1HH, X3), R),
                        field_mul(a.y, HHH)); // [2M]

    out->x = X3;
    out->y = Y3;
    out->z = Z3;
    out->infinity = 0;
}

// --- Brier-Joye Unified Mixed Addition template (4x64) ----------------------
// CHECK_INFINITY=false skips infinity handling (safe in scalar_mul inner loop).
// Cost: 7M + 5S.

template<bool CHECK_INFINITY>
SECP256K1_INLINE
void unified_add_core(CTJacobianPoint* out,
                       const CTJacobianPoint& a,
                       const CTAffinePoint& b) noexcept {
    FE52 X1 = a.x, Y1 = a.y, Z1 = a.z;
    [[maybe_unused]] std::uint64_t a_inf = a.infinity;

    FE52 zz    = field_sqr(Z1);
    FE52 u1    = X1;
    FE52 u2    = field_mul(b.x, zz);
    FE52 s1    = Y1;
    FE52 s2    = field_mul(field_mul(b.y, zz), Z1);

    FE52 t_val = field_add(u1, u2);
    FE52 m_val = field_add(s1, s2);

    FE52 rr = field_add(field_sqr(t_val),
                        field_mul(u1, field_neg(u2)));

    std::uint64_t degen = field_is_zero(m_val);

    FE52 rr_alt = field_add(s1, s1);
    FE52 m_alt  = field_sub(u1, u2);

    field_cmov(&rr_alt, rr,    ~degen);
    field_cmov(&m_alt,  m_val, ~degen);

    FE52 n_val = field_sqr(m_alt);
    FE52 q     = field_mul(field_neg(t_val), n_val);
    n_val = field_sqr(n_val);
    field_cmov(&n_val, m_val, degen);

    FE52 x3 = field_add(field_sqr(rr_alt), q);
    FE52 z3 = field_mul(m_alt, Z1);
    FE52 y3 = field_half(field_neg(
        field_add(field_mul(rr_alt, field_add(field_add(x3, x3), q)),
                  n_val)));

    if constexpr (CHECK_INFINITY) {
        FE52 one_fe = FieldElement::one();
        field_cmov(&x3, b.x, a_inf);
        field_cmov(&y3, b.y, a_inf);
        field_cmov(&z3, one_fe, a_inf);
    }

    out->x = x3;  out->y = y3;  out->z = z3;
    if constexpr (CHECK_INFINITY) {
        out->infinity = field_is_zero(z3);
    } else {
        out->infinity = 0;
    }
}

CTJacobianPoint point_add_mixed_unified(const CTJacobianPoint& a,
                                         const CTAffinePoint& b) noexcept {
    CTJacobianPoint result;
    unified_add_core<true>(&result, a, b);
    return result;
}

void point_add_mixed_unified_into(CTJacobianPoint* out,
                                   const CTJacobianPoint& a,
                                   const CTAffinePoint& b) noexcept {
    unified_add_core<true>(out, a, b);
}

// --- Batch N Doublings (4x64) ------------------------------------------------
// 3M + 4S + 1half per doubling. No magnitude tracking (always normalized).
// Precondition: r is NOT infinity.

SECP256K1_INLINE
void point_dbl_n_core(CTJacobianPoint* r, unsigned n) noexcept {
    FE52 X = r->x, Y = r->y, Z = r->z;
    for (unsigned i = 0; i < n; ++i) {
        FE52 s  = field_sqr(Y);
        FE52 l  = field_sqr(X);
        l = field_half(field_add(field_add(l, l), l));
        FE52 t  = field_mul(field_neg(s), X);
        FE52 z3 = field_mul(Z, Y);
        FE52 x3 = field_add(field_add(field_sqr(l), t), t);
        s = field_sqr(s);
        FE52 y3 = field_neg(
            field_add(field_mul(field_add(t, x3), l), s));
        X = x3;  Y = y3;  Z = z3;
    }
    r->x = X;  r->y = Y;  r->z = Z;
}

void point_dbl_n_inplace(CTJacobianPoint* r, unsigned n) noexcept {
    point_dbl_n_core(r, n);
}

CTJacobianPoint point_dbl_n(const CTJacobianPoint& p, unsigned n) noexcept {
    CTJacobianPoint result = p;
    result.infinity = 0;
    point_dbl_n_inplace(&result, n);
    return result;
}

// ===============================================================================
// CT GLV Endomorphism -- Helpers & Decomposition (4x64, portable)
// ===============================================================================

namespace /* GLV CT helpers 4x64 */ {

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

static std::uint64_t ct_scalar_is_high(const Scalar& s) noexcept {
    static constexpr std::uint64_t HALF_N[4] = {
        0xDFE92F46681B20A0ULL, 0x5D576E7357A4501DULL,
        0xFFFFFFFFFFFFFFFFULL, 0x7FFFFFFFFFFFFFFFULL
    };
    std::uint64_t tmp[4];
    std::uint64_t borrow = local_sub256(tmp, HALF_N, s.limbs().data());
    return is_nonzero_mask(borrow);
}

// --- CT 256x256->512, shift >>384 (portable) ---------------------------------

static std::array<std::uint64_t, 4> ct_mul_shift_384(
    const std::array<std::uint64_t, 4>& a,
    const std::array<std::uint64_t, 4>& b) noexcept
{
    std::uint64_t prod[8];
    mul_4x4(prod, a.data(), b.data());

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

// --- CT 128x128 -> 256 multiply (portable) -----------------------------------
// See the 5x52 copy for the width argument: c1 < 2^126 and c2 < 2^128 are
// structural properties of the rounding step, true for every k, so the narrow
// multiply is width-invariant and constant-time with respect to the scalar.
static inline void ct_mul_2x2(std::uint64_t r[4],
                               const std::uint64_t a[2],
                               const std::uint64_t b[2]) noexcept {
    r[0] = 0; r[1] = 0; r[2] = 0; r[3] = 0;
    for (int i = 0; i < 2; ++i) {
        std::uint64_t carry = 0;
        for (int j = 0; j < 2; ++j) {
            U128 const m = mul64(a[i], b[j]);
            std::uint64_t sum = r[i + j] + m.lo;
            std::uint64_t const c1 = static_cast<std::uint64_t>(sum < r[i + j]);
            sum += carry;
            std::uint64_t const c2 = static_cast<std::uint64_t>(sum < carry);
            r[i + j] = sum;
            carry = m.hi + c1 + c2;
        }
        r[i + 2] = carry;
    }
}

// --- beta (beta) as 4x64 FieldElement ------------------------------------------

static const FE52& get_beta_fe() noexcept {
    static const FE52 beta = FieldElement::from_bytes(
        secp256k1::fast::glv_constants::BETA);
    return beta;
}

// --- GLV lattice constants ---------------------------------------------------

static constexpr std::array<std::uint64_t, 4> kG1{{
    0xE893209A45DBB031ULL, 0x3DAA8A1471E8CA7FULL,
    0xE86C90E49284EB15ULL, 0x3086D221A7D46BCDULL
}};
static constexpr std::array<std::uint64_t, 4> kG2{{
    0x1571B4AE8AC47F71ULL, 0x221208AC9DF506C6ULL,
    0x6F547FA90ABFE4C4ULL, 0xE4437ED6010E8828ULL
}};

} // anonymous namespace (GLV CT helpers 4x64)

// --- CT GLV Endomorphism (4x64) ----------------------------------------------

CTJacobianPoint point_endomorphism(const CTJacobianPoint& p) noexcept {
    CTJacobianPoint r;
    r.x = field_mul(p.x, get_beta_fe());
    r.y = p.y;
    r.z = p.z;
    r.infinity = p.infinity;
    return r;
}

CTAffinePoint affine_endomorphism(const CTAffinePoint& p) noexcept {
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

// --- CT GLV Decomposition (4x64, fully CT, portable) ------------------------

CTGLVDecomposition ct_glv_decompose(const Scalar& k) noexcept {
    // Raw GLV lattice basis, low 128 bits only.  libsecp256k1 publishes the
    // DERIVED constant set (-b1, -b2, lambda); -b2 = n - b2 is a full 256-bit
    // number even though b2 itself is 126 bits, which forces a 256x256 multiply
    // plus a wide mod-n reduction for every c2 term, and then a third 256x256
    // multiply by lambda to recover k1.  Working from the raw basis instead keeps
    // all four products inside 2x2:
    //   -b1  = 0xE4437ED6010E8828_6F547FA90ABFE4C3   (128 bits)
    //    b2  = 0x3086D221A7D46BCD_E86C90E49284EB15   (126 bits), and a1 == b2
    //    a2  = 2^128 + kA2Lo, kA2Lo = 125 bits
    // The lattice relations a1 + b1*lambda == 0 (mod n) and a2 + b2*lambda == 0
    // (mod n) hold exactly, so
    //   lambda*k2 = lambda*(-c1*b1 - c2*b2) = c1*a1 + c2*a2   (mod n)
    // and k1 needs no lambda multiply at all -- the 2^128 half of a2 is a limb
    // placement, not a multiply.
    static constexpr std::uint64_t kMinusB1[2] = {
        0x6F547FA90ABFE4C3ULL, 0xE4437ED6010E8828ULL
    };
    static constexpr std::uint64_t kB2[2] = {          // b2, and a1 == b2
        0xE86C90E49284EB15ULL, 0x3086D221A7D46BCDULL
    };
    static constexpr std::uint64_t kA2Lo[2] = {        // a2 - 2^128
        0x57C1108D9D44CFD8ULL, 0x14CA50F7A8E2F3F6ULL
    };

    auto k_limbs = k.limbs();
    auto c1_limbs = ct_mul_shift_384({k_limbs[0], k_limbs[1], k_limbs[2], k_limbs[3]}, kG1);
    auto c2_limbs = ct_mul_shift_384({k_limbs[0], k_limbs[1], k_limbs[2], k_limbs[3]}, kG2);

    // Width bounds, valid for EVERY 4-limb k (c1 and c2 are monotonic in k, so
    // evaluating at the largest 4-limb input k = 2^256-1 bounds them all):
    //   c1 <= 2^126 and c2 <= 2^128, hence c1_limbs[2..3] and c2_limbs[2..3] are
    //   always zero and reading only the low two limbs cannot depend on k.
    //   c1*(-b1) < 2^254, c2*b2 < 2^254, c1*a1 < 2^252, c2*kA2Lo < 2^253,
    //   c1*a1 + c2*kA2Lo < 2^253, and c2*2^128 < n - 2^253.
    // Every product is therefore already reduced -- no mod-n reduction anywhere.
    std::uint64_t mb1c1[4], b2c2[4], a1c1[4], a2c2[4];
    ct_mul_2x2(mb1c1, c1_limbs.data(), kMinusB1);
    ct_mul_2x2(b2c2,  c2_limbs.data(), kB2);
    ct_mul_2x2(a1c1,  c1_limbs.data(), kB2);
    ct_mul_2x2(a2c2,  c2_limbs.data(), kA2Lo);

    // k2 = c1*(-b1) - c2*b2 mod n.  scalar_sub is the branchless mod-n
    // subtract (borrow mask + cmov256), so the wrap is a cmov, never a branch.
    Scalar const k2_mod = scalar_sub(
        Scalar::from_limbs({mb1c1[0], mb1c1[1], mb1c1[2], mb1c1[3]}),
        Scalar::from_limbs({b2c2[0], b2c2[1], b2c2[2], b2c2[3]}));

    std::uint64_t const k2_high = ct_scalar_is_high(k2_mod);
    Scalar const k2_abs = scalar_cneg(k2_mod, k2_high);

    // k1 = k - lambda*k2 = k - c1*a1 - c2*kA2Lo - c2*2^128 mod n.
    Scalar const lambda_k2 = scalar_add(
        Scalar::from_limbs({a1c1[0], a1c1[1], a1c1[2], a1c1[3]}),
        Scalar::from_limbs({a2c2[0], a2c2[1], a2c2[2], a2c2[3]}));
    Scalar const c2_shifted = Scalar::from_limbs({0, 0, c2_limbs[0], c2_limbs[1]});
    Scalar const k1_mod = scalar_sub(scalar_sub(k, lambda_k2), c2_shifted);

    std::uint64_t const k1_high = ct_scalar_is_high(k1_mod);
    Scalar const k1_abs = scalar_cneg(k1_mod, k1_high);

    CTGLVDecomposition result;
    result.k1     = k1_abs;
    result.k2     = k2_abs;
    result.k1_neg = k1_high;
    result.k2_neg = k2_high;
    return result;
}

// --- CT GLV Scalar Multiplication (4x64) -- Hamburg Signed-Digit --------------
// Hamburg signed-digit comb + GLV. GROUP_SIZE=5, TABLE_SIZE=16, GROUPS=26.
// Cost: 125 dbl + 52 unified_add + 52 signed_lookups(16).

Point scalar_mul(const Point& p, const Scalar& k) noexcept {
    constexpr unsigned GROUP_SIZE = 5;
    constexpr unsigned TABLE_SIZE = 1u << (GROUP_SIZE - 1);  // 16
    constexpr unsigned GROUPS = 26;

    static const Scalar K_CONST = Scalar::from_limbs({
        0xb5c2c1dcde9798d9ULL, 0x589ae84826ba29e4ULL,
        0xc2bdd6bf7c118d6bULL, 0xa4e88a7dcb13034eULL
    });
    // 1. Hamburg transform + GLV split
    Scalar s = scalar_add(k, K_CONST);
    s = scalar_half(s);

    auto [k1_abs, k2_abs, k1_neg, k2_neg] = ct_glv_decompose(s);

    // Compute v = k_abs + 2^128 (positive) or 2^128 - k_abs (negative)
    // using NON-MODULAR integer ops. Modular scalar_cneg+scalar_add wraps
    // around n when |ki| marginally exceeds 2^128.
    // CT-CRYPTO-001: branchless make_v — no secret if(k_neg){...}else{...}.
    // k_neg is the SECRET GLV sign bit (ct_scalar_is_high of the secret half-
    // scalar), and this variable-base scalar_mul is reached from ECDH/ellswift/
    // seckey_tweak_mul with a secret scalar, so the old branch was a CT-hygiene
    // gap. This is the SAME masked computation as the ct_glv_make_v helper, but
    // inlined here so it compiles on every target (the helper lives in a
    // platform-guarded block not compiled on wasm/32-bit/MSVC; this public
    // scalar_mul is compiled everywhere).
    auto make_v = [](const Scalar& k_abs, std::uint64_t k_neg) noexcept -> Scalar {
        auto L = k_abs.limbs();
        std::uint64_t const neg_mask = -static_cast<std::uint64_t>(k_neg != 0);
        std::uint64_t const pos_mask = ~neg_mask;
        // negate form: 2^128 - k_abs  (non-modular)
        std::uint64_t nL[4], borrow = 0, diff;
        diff = 0 - L[0];          borrow = (L[0] != 0) ? 1 : 0;        nL[0] = diff;
        diff = 0 - L[1] - borrow; borrow = (L[1] | borrow) ? 1 : 0;    nL[1] = diff;
        diff = 1 - L[2] - borrow; borrow = (L[2] + borrow > 1) ? 1 : 0; nL[2] = diff;
        nL[3] = 0 - borrow;
        // positive form: k_abs + 2^128
        std::uint64_t pL[4];
        pL[0] = L[0]; pL[1] = L[1];
        std::uint64_t const sum = L[2] + 1;
        pL[2] = sum; pL[3] = L[3] + static_cast<std::uint64_t>(sum < L[2]);
        // branchless select per limb
        L[0] = (neg_mask & nL[0]) | (pos_mask & pL[0]);
        L[1] = (neg_mask & nL[1]) | (pos_mask & pL[1]);
        L[2] = (neg_mask & nL[2]) | (pos_mask & pL[2]);
        L[3] = (neg_mask & nL[3]) | (pos_mask & pL[3]);
        return Scalar::from_limbs(L);
    };
    Scalar v1 = make_v(k1_abs, k1_neg);
    Scalar v2 = make_v(k2_abs, k2_neg);

    // 2. Build odd-multiples table via effective-affine + Z-ratio normalization
    CTAffinePoint pre_a[TABLE_SIZE];
    CTAffinePoint pre_a_lam[TABLE_SIZE];

    Point p2 = p;
    p2.dbl_inplace();
    FE52 C  = p2.z();
    FE52 C2 = field_sqr(C);
    FE52 C3 = field_mul(C2, C);

    FE52 d_x = p2.X();
    FE52 d_y = p2.Y();

    Jac4x64 iso[TABLE_SIZE];
    iso[0] = { field_mul(p.X(), C2), field_mul(p.Y(), C3), p.z() };

    FE52 zr[TABLE_SIZE];
    for (std::size_t i = 1; i < TABLE_SIZE; ++i) {
        iso[i] = jac_add_ge_var_zr(iso[i - 1], d_x, d_y, &zr[i]);
    }

    FE52 global_z = field_mul(iso[TABLE_SIZE - 1].z, C);

    const FE52& beta = get_beta_fe();

    pre_a[TABLE_SIZE - 1].x = iso[TABLE_SIZE - 1].x;
    pre_a[TABLE_SIZE - 1].y = iso[TABLE_SIZE - 1].y;
    pre_a[TABLE_SIZE - 1].infinity = 0;
    pre_a_lam[TABLE_SIZE - 1].x = field_mul(pre_a[TABLE_SIZE - 1].x, beta);
    pre_a_lam[TABLE_SIZE - 1].y = pre_a[TABLE_SIZE - 1].y;
    pre_a_lam[TABLE_SIZE - 1].infinity = 0;

    FE52 zs_acc = zr[TABLE_SIZE - 1];
    for (int i = static_cast<int>(TABLE_SIZE) - 2; i >= 0; --i) {
        FE52 zs2 = field_sqr(zs_acc);
        FE52 zs3 = field_mul(zs2, zs_acc);
        pre_a[i].x = field_mul(iso[i].x, zs2);
        pre_a[i].y = field_mul(iso[i].y, zs3);
        pre_a[i].infinity = 0;
        pre_a_lam[i].x = field_mul(pre_a[i].x, beta);
        pre_a_lam[i].y = pre_a[i].y;
        pre_a_lam[i].infinity = 0;
        if (i > 0) {
            zs_acc = field_mul(zs_acc, zr[i]);
        }
    }

    SECP256K1_DECLASSIFY(pre_a, sizeof(pre_a));
    SECP256K1_DECLASSIFY(pre_a_lam, sizeof(pre_a_lam));

    // 3. Main loop -- in-place, always-inline
    CTJacobianPoint R;
    CTAffinePoint t;

    {
        int group = static_cast<int>(GROUPS) - 1;
        std::uint64_t bits1 = scalar_window(v1,
            static_cast<std::size_t>(group) * GROUP_SIZE, GROUP_SIZE);
        std::uint64_t bits2 = scalar_window(v2,
            static_cast<std::size_t>(group) * GROUP_SIZE, GROUP_SIZE);

        table_lookup_core<false>(&t, pre_a, TABLE_SIZE, bits1, GROUP_SIZE);
        R.x = t.x;  R.y = t.y;  R.z = FieldElement::one();  R.infinity = 0;

        table_lookup_core<false>(&t, pre_a_lam, TABLE_SIZE, bits2, GROUP_SIZE);
        unified_add_core<true>(&R, R, t);
    }

    for (int group = static_cast<int>(GROUPS) - 2; group >= 0; --group) {
        std::uint64_t bits1 = scalar_window(v1,
            static_cast<std::size_t>(group) * GROUP_SIZE, GROUP_SIZE);
        std::uint64_t bits2 = scalar_window(v2,
            static_cast<std::size_t>(group) * GROUP_SIZE, GROUP_SIZE);

        point_dbl_n_core(&R, GROUP_SIZE);

        table_lookup_core<false>(&t, pre_a, TABLE_SIZE, bits1, GROUP_SIZE);
        unified_add_core<false>(&R, R, t);

        table_lookup_core<false>(&t, pre_a_lam, TABLE_SIZE, bits2, GROUP_SIZE);
        unified_add_core<false>(&R, R, t);
    }

    R.z = field_mul(R.z, global_z);

    Point result = R.to_point();
    SECP256K1_DECLASSIFY(&result, sizeof(result));
    return result;
}

// --- CT Generator Multiplication (4x64) -- Comb Method ------------------------
// COMB_TEETH=6, COMB_BLOCKS=11, COMB_SPACING=4, last block short (4 teeth):
// COMB_BITS = 10*6*4 + 4*4 = 256 exactly, so the comb tiles every bit of v
// once, K_gen = 2^256-1 is exactly right, and no correction point is needed.
// Table: 10 blocks x 32 entries + 1 block x 8 entries = 328 affine points.
// Runtime: 43 additions + 44 lookups + 3 doublings, no correction.

namespace {

constexpr unsigned COMB_TEETH   = 6;
constexpr unsigned COMB_BLOCKS  = 11;
constexpr unsigned COMB_SPACING = 4;
constexpr unsigned COMB_FULL_BLOCKS = COMB_BLOCKS - 1;   // 10 six-tooth blocks
constexpr unsigned COMB_TAIL_TEETH  = 4;                 // block 10 has 4 teeth
constexpr unsigned COMB_BITS =
    COMB_FULL_BLOCKS * COMB_TEETH * COMB_SPACING + COMB_TAIL_TEETH * COMB_SPACING;
static_assert(COMB_BITS == 256,
              "comb must tile exactly 256 bits: K_gen = 2^256-1 assumes it");
constexpr std::size_t COMB_TABLE_SIZE      = 1u << (COMB_TEETH - 1);       // 32
constexpr std::size_t COMB_TAIL_TABLE_SIZE = 1u << (COMB_TAIL_TEETH - 1);  // 8

static constexpr std::uint64_t K_GEN[4] = {
    0x402DA1732FC9BEBEULL, 0x4551231950B75FC4ULL,
    0x0000000000000001ULL, 0x0000000000000000ULL
};

struct alignas(64) CombGenTable {
    CTAffinePoint entries[COMB_FULL_BLOCKS][COMB_TABLE_SIZE];
    CTAffinePoint tail[COMB_TAIL_TABLE_SIZE];  // short last block
    bool initialized = false;
};

static CombGenTable g_comb_table;
// ESP32/bare-metal: std::once_flag has non-trivial ctor → .data → global ctor
// before FreeRTOS starts. Use plain volatile bool (.bss, zero-init) instead.
#if defined(ESP_PLATFORM) || defined(SECP256K1_PLATFORM_ESP32)
static volatile bool g_comb_init = false;
#else
static std::once_flag g_comb_table_once;
#endif

inline std::uint64_t extract_comb_digit(const Scalar& v,
                                         unsigned block,
                                         unsigned comb_off,
                                         unsigned teeth) noexcept {
    // One window read replaces `teeth` scalar_bit scans; see the 5x52 twin for
    // the argument.  base and teeth come from loop counters only, so they are
    // public; the bits are secret and are combined branchlessly.  The highest
    // position read is 243 + 12 = 255, inside the scalar.
    unsigned span = (teeth - 1) * COMB_SPACING + 1;
    std::size_t base = static_cast<std::size_t>(block) * COMB_TEETH
                       * COMB_SPACING + comb_off;
    std::uint64_t window = scalar_window(v, base, span);

    std::uint64_t digit = 0;
    for (unsigned tooth = 0; tooth < teeth; ++tooth) {
        digit |= ((window >> (tooth * COMB_SPACING)) & 1) << tooth;
    }
    return digit;
}

// TEETH is a template parameter because it is fixed per block index (a public
// loop counter), so the scan length is identical for every scalar.
template <unsigned TEETH>
static inline
void comb_lookup(CTAffinePoint* out,
                  const CTAffinePoint* table,
                  std::uint64_t digit) noexcept {
    constexpr std::size_t TABLE_SIZE = std::size_t{1} << (TEETH - 1);
    std::uint64_t top = (digit >> (TEETH - 1)) & 1;
    std::uint64_t needs_negate = top ^ 1;
    std::uint64_t neg_mask = static_cast<std::uint64_t>(
                      -static_cast<std::int64_t>(needs_negate));

    std::uint64_t d_lo = digit & (TABLE_SIZE - 1);
    std::uint64_t idx_pos = d_lo;
    std::uint64_t idx_neg = (TABLE_SIZE - 1) - d_lo;
    std::uint64_t index = idx_pos ^ ((idx_pos ^ idx_neg) & neg_mask);

    out->x = table[0].x;
    out->y = table[0].y;
    for (std::size_t m = 1; m < TABLE_SIZE; ++m) {
        std::uint64_t mask = eq_mask(static_cast<std::uint64_t>(m), index);
        field_cmov(&out->x, table[m].x, mask);
        field_cmov(&out->y, table[m].y, mask);
    }

    FE52 neg_y = field_neg(out->y);
    field_cmov(&out->y, neg_y, neg_mask);
    out->infinity = 0;
}

// entry[idx] = P_{TEETH-1} + sum_{j < TEETH-1} (2*bit_j(idx) - 1) * P_j,
// where P_t = bases[base_off + t].  Shared by the six-tooth blocks and the
// four-tooth tail so both geometries come from one piece of code.
template <unsigned TEETH>
void build_comb_block(CTAffinePoint* dst, const Point* bases,
                       unsigned base_off) noexcept {
    constexpr std::size_t TABLE_SIZE = std::size_t{1} << (TEETH - 1);

    FE52 zs[TEETH], z_invs[TEETH];
    for (unsigned t = 0; t < TEETH; ++t)
        zs[t] = bases[base_off + t].z();
    fe_batch_inverse(z_invs, zs, TEETH);

    FE52 aff_x[TEETH], aff_y[TEETH];
    for (unsigned t = 0; t < TEETH; ++t) {
        FE52 zi2 = field_sqr(z_invs[t]);
        FE52 zi3 = field_mul(zi2, z_invs[t]);
        aff_x[t] = field_mul(bases[base_off + t].X(), zi2);
        aff_y[t] = field_mul(bases[base_off + t].Y(), zi3);
    }

    Point entries_jac[TABLE_SIZE];
    for (std::size_t idx = 0; idx < TABLE_SIZE; ++idx) {
        Point entry = Point::from_affine(aff_x[TEETH - 1], aff_y[TEETH - 1]);
        for (unsigned j = 0; j < TEETH - 1; ++j) {
            FE52 py = ((idx >> j) & 1) ? aff_y[j] : field_neg(aff_y[j]);
            Point pj = Point::from_affine(aff_x[j], py);
            entry = entry.add(pj);
        }
        entries_jac[idx] = entry;
    }

    FE52 ent_zs[TABLE_SIZE], ent_zis[TABLE_SIZE];
    for (std::size_t idx = 0; idx < TABLE_SIZE; ++idx)
        ent_zs[idx] = entries_jac[idx].z();
    fe_batch_inverse(ent_zis, ent_zs, TABLE_SIZE);

    for (std::size_t idx = 0; idx < TABLE_SIZE; ++idx) {
        FE52 zi2 = field_sqr(ent_zis[idx]);
        FE52 zi3 = field_mul(zi2, ent_zis[idx]);
        dst[idx].x = field_mul(entries_jac[idx].X(), zi2);
        dst[idx].y = field_mul(entries_jac[idx].Y(), zi3);
        dst[idx].infinity = 0;
    }
}

void build_comb_table() noexcept {
    Point G = Point::generator();

    // base[i] = 2^(i * COMB_SPACING) * G for i = 0..63; the tail block
    // continues the same run, so the bases cover exactly bits 0..255.
    constexpr unsigned NUM_BASES =
        COMB_FULL_BLOCKS * COMB_TEETH + COMB_TAIL_TEETH;  // 64
    static_assert(NUM_BASES * COMB_SPACING == COMB_BITS, "base run must tile COMB_BITS");
    Point bases[NUM_BASES];
    bases[0] = G;
    for (unsigned i = 1; i < NUM_BASES; ++i) {
        bases[i] = bases[i - 1];
        for (unsigned d = 0; d < COMB_SPACING; ++d)
            bases[i].dbl_inplace();
    }

    for (unsigned b = 0; b < COMB_FULL_BLOCKS; ++b)
        build_comb_block<COMB_TEETH>(g_comb_table.entries[b], bases,
                                     b * COMB_TEETH);
    build_comb_block<COMB_TAIL_TEETH>(g_comb_table.tail, bases,
                                      COMB_FULL_BLOCKS * COMB_TEETH);

    g_comb_table.initialized = true;
}

} // anonymous namespace

void init_generator_table() noexcept {
#if defined(ESP_PLATFORM) || defined(SECP256K1_PLATFORM_ESP32)
    if (!g_comb_init) { build_comb_table(); g_comb_init = true; }
#else
    std::call_once(g_comb_table_once, build_comb_table);
#endif
}

Point generator_mul(const Scalar& k) noexcept {
    init_generator_table();

    static const Scalar K_gen_scalar = Scalar::from_limbs(
        {K_GEN[0], K_GEN[1], K_GEN[2], K_GEN[3]});

    Scalar s = scalar_add(k, K_gen_scalar);
    Scalar v = scalar_half(s);

    CTJacobianPoint R;
    CTAffinePoint T;

    unsigned comb_off = COMB_SPACING - 1;

    // First outer iteration: first block initializes R
    {
        std::uint64_t digit = extract_comb_digit(v, 0, comb_off, COMB_TEETH);
        comb_lookup<COMB_TEETH>(&T, g_comb_table.entries[0], digit);
        R.x = T.x;  R.y = T.y;  R.z = FieldElement::one();  R.infinity = 0;
    }

    // Safety: all T values are fixed G multiples from the precomputed comb
    // table. The incomplete formula is safe here for the same reason as in the
    // 52-bit path — probability of R equalling a table entry is ~2^-128.
    // Block 1 still sees Z1 == 1 (public block index, not scalar-derived).
    {
        std::uint64_t digit = extract_comb_digit(v, 1, comb_off, COMB_TEETH);
        comb_lookup<COMB_TEETH>(&T, g_comb_table.entries[1], digit);
        add_affine_fast_ct_4x64_z1(&R, R, T);
    }
    for (unsigned b = 2; b < COMB_FULL_BLOCKS; ++b) {
        std::uint64_t digit = extract_comb_digit(v, b, comb_off, COMB_TEETH);
        comb_lookup<COMB_TEETH>(&T, g_comb_table.entries[b], digit);
        add_affine_fast_ct_4x64(&R, R, T);
    }
    {
        std::uint64_t digit =
            extract_comb_digit(v, COMB_FULL_BLOCKS, comb_off, COMB_TAIL_TEETH);
        comb_lookup<COMB_TAIL_TEETH>(&T, g_comb_table.tail, digit);
        add_affine_fast_ct_4x64(&R, R, T);
    }

    // Remaining outer iterations with doubling
    while (comb_off-- > 0) {
        point_dbl_n_core(&R, 1);

        for (unsigned b = 0; b < COMB_FULL_BLOCKS; ++b) {
            std::uint64_t digit = extract_comb_digit(v, b, comb_off, COMB_TEETH);
            comb_lookup<COMB_TEETH>(&T, g_comb_table.entries[b], digit);
            add_affine_fast_ct_4x64(&R, R, T);
        }
        std::uint64_t digit =
            extract_comb_digit(v, COMB_FULL_BLOCKS, comb_off, COMB_TAIL_TEETH);
        comb_lookup<COMB_TAIL_TEETH>(&T, g_comb_table.tail, digit);
        add_affine_fast_ct_4x64(&R, R, T);
    }

    Point result = R.to_point();
    SECP256K1_DECLASSIFY(&result, sizeof(result));
    return result;
}

// --- ecmult_const_xonly fallback (4x64 path) ---------------------------------
// Uses sqrt since we lack the FE52 Jacobian-output optimisation.
FieldElement ecmult_const_xonly(const FieldElement& xn, const FieldElement& xd,
                                 const Scalar& q) noexcept {
    // Compute x = xn / xd  (for BIP-324 ECDH, xd == one so this is xn directly)
    FieldElement x;
    if (xd == FieldElement::one()) {
        x = xn;
    } else {
        x = field_mul(xn, field_inv(xd));
    }
    // Reconstruct affine point via sqrt
    FieldElement x2 = field_sqr(x);
    FieldElement x3 = field_mul(x, x2);
    FieldElement y2 = field_add(x3, B7);
    FieldElement y  = y2.sqrt();
    if (y.square() != y2) return FieldElement::zero();
    if (y.to_bytes()[31] & 1) y = y.negate();
    Point P = fast::Point::from_affine(x, y);
    if (P.is_infinity()) return FieldElement::zero();
    Point res = scalar_mul(P, q);
    if (res.is_infinity()) return FieldElement::zero();
    return res.x();
}

#endif // SECP256K1_FAST_52BIT

// --- CT Curve Check (uses 4x64 FieldElement at API boundary) -----------------

std::uint64_t point_is_on_curve(const Point& p) noexcept {
    if (p.is_infinity()) {
        return ~static_cast<std::uint64_t>(0);
    }

    FieldElement const x = p.x();
    FieldElement const y = p.y();

    FieldElement const y2 = field_sqr(y);
    FieldElement const x2 = field_sqr(x);
    FieldElement const x3 = field_mul(x, x2);
    FieldElement const rhs = field_add(x3, B7);

    return field_eq(y2, rhs);
}

// --- CT Point Equality (uses 4x64 at API boundary) --------------------------

std::uint64_t point_eq(const Point& a, const Point& b) noexcept {
    if (a.is_infinity() && b.is_infinity()) {
        return ~static_cast<std::uint64_t>(0);
    }
    if (a.is_infinity() || b.is_infinity()) {
        return 0;
    }

    FieldElement const z1sq = field_sqr(a.z());
    FieldElement const z2sq = field_sqr(b.z());
    FieldElement const u1 = field_mul(a.X(), z2sq);
    FieldElement const u2 = field_mul(b.X(), z1sq);

    FieldElement const z1cu = field_mul(z1sq, a.z());
    FieldElement const z2cu = field_mul(z2sq, b.z());
    FieldElement const s1 = field_mul(a.Y(), z2cu);
    FieldElement const s2 = field_mul(b.Y(), z1cu);

    return field_eq(u1, u2) & field_eq(s1, s2);
}

// --- Context Scalar Blinding -------------------------------------------------

namespace {

struct BlindingState {
    Scalar       r;               // random blinding scalar
    CTAffinePoint neg_r_G;        // affine coordinates of -r*G
    bool         active{false};
};

} // anonymous namespace

static thread_local BlindingState g_blinding;

void set_blinding(const Scalar& r, const Point& r_G) noexcept {
    BlindingState& bl = g_blinding;
    bl.r = r;
    // Store -r*G as affine point for efficient mixed addition
    CTAffinePoint aff = CTAffinePoint::from_point(r_G);
    // Negate y: -r*G = (x, -y)
    aff.y = aff.y.negate(1);
    aff.y.normalize_weak();
    bl.neg_r_G = aff;
    bl.active  = true;
}

void clear_blinding() noexcept {
    using secp256k1::detail::secure_erase;
    BlindingState& bl = g_blinding;
    secure_erase(&bl.r,       sizeof(bl.r));
    secure_erase(&bl.neg_r_G, sizeof(bl.neg_r_G));
    bl.active = false;
}

Point generator_mul_blinded(const Scalar& k) noexcept {
    const BlindingState& bl = g_blinding;
    if (SECP256K1_UNLIKELY(!bl.active)) {
        return generator_mul(k);
    }
    // (k + r)*G - r*G = k*G, but attacker observes multiplication over (k+r)
    const Scalar blinded = scalar_add(k, bl.r);
    const Point  R       = generator_mul(blinded);
    // Subtract r*G: add precomputed -r*G (one mixed complete addition)
    CTJacobianPoint R_jac    = CTJacobianPoint::from_point(R);
    CTJacobianPoint R_result = point_add_mixed_complete(R_jac, bl.neg_r_G);
    Point result = R_result.to_point();
    SECP256K1_DECLASSIFY(&result, sizeof(result));
    return result;
}

} // namespace secp256k1::ct
