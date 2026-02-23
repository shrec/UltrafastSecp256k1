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
// CT generator multiplication (OPTIMIZED):
//   Uses precomputed table of 64x16 affine G-multiples.
//   Runtime: 64 mixed Jacobian+Affine additions + 64 CT lookups(16).
//   NO doublings needed.
// ============================================================================

#include "secp256k1/config.hpp"  // SECP256K1_FAST_52BIT, SECP256K1_INLINE
#include "secp256k1/ct/point.hpp"
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
static const FieldElement B7 = FieldElement::from_uint64(7);

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
    std::uint64_t x = t4 >> 48;
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
    std::uint64_t z0 = t0 | t1 | t2 | t3 | t4;
    std::uint64_t z1 = (t0 ^ 0x000FFFFEFFFFFC2FULL) | (t1 ^ M52) |
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
    FE52 z1sq = a.z.square();
    FE52 u2   = bx * z1sq;
    FE52 z1cu = z1sq * a.z;
    FE52 s2   = by * z1cu;

    FE52 h  = u2 + a.x.negate(7);
    FE52 r  = s2 + a.y.negate(3);

    FE52 h2  = h.square();
    FE52 h3  = h * h2;
    FE52 u1h2 = a.x * h2;

    FE52 r2  = r.square();
    FE52 x3  = r2 + h3.negate(1) + u1h2.negate(1) + u1h2.negate(1);

    FE52 diff = u1h2 + x3.negate(7);
    FE52 y3  = (r * diff) + (a.y * h3).negate(1);

    FE52 z3  = a.z * h;

    *zr_out = h;  // Z-ratio: result.z = a.z * h
    return {x3, y3, z3};
}

// --- FE52 Native Field Inversion (Fermat's little theorem) ------------------
// Computes a^(-1) = a^(p-2) mod p using optimal addition chain.
// p = 2^256 - 2^32 - 977. Binary structure of p-2 has blocks of 1s with
// lengths {1, 2, 22, 223}. Uses 253 squarings + 14 multiplications.
// All operations in native FE52 -- no 4x64 conversion overhead.
//
// Adapted from bitcoin-core/secp256k1 field_impl.h (MIT license).

inline FE52 fe52_inverse(const FE52& a) noexcept {
    FE52 x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223, t1;

    // x2 = a^(2^2-1) = a^3
    x2 = a.square();              // a^2
    x2 = x2 * a;                  // a^3

    // x3 = a^(2^3-1) = a^7
    x3 = x2.square();             // a^6
    x3 = x3 * a;                  // a^7

    // x6 = a^(2^6-1)
    x6 = x3;
    for (int j = 0; j < 3; ++j) x6.square_inplace();
    x6 = x6 * x3;

    // x9 = a^(2^9-1)
    x9 = x6;
    for (int j = 0; j < 3; ++j) x9.square_inplace();
    x9 = x9 * x3;

    // x11 = a^(2^11-1)
    x11 = x9;
    for (int j = 0; j < 2; ++j) x11.square_inplace();
    x11 = x11 * x2;

    // x22 = a^(2^22-1)
    x22 = x11;
    for (int j = 0; j < 11; ++j) x22.square_inplace();
    x22 = x22 * x11;

    // x44 = a^(2^44-1)
    x44 = x22;
    for (int j = 0; j < 22; ++j) x44.square_inplace();
    x44 = x44 * x22;

    // x88 = a^(2^88-1)
    x88 = x44;
    for (int j = 0; j < 44; ++j) x88.square_inplace();
    x88 = x88 * x44;

    // x176 = a^(2^176-1)
    x176 = x88;
    for (int j = 0; j < 88; ++j) x176.square_inplace();
    x176 = x176 * x88;

    // x220 = a^(2^220-1)
    x220 = x176;
    for (int j = 0; j < 44; ++j) x220.square_inplace();
    x220 = x220 * x44;

    // x223 = a^(2^223-1)
    x223 = x220;
    for (int j = 0; j < 3; ++j) x223.square_inplace();
    x223 = x223 * x3;

    // Final assembly:  a^(p-2) using sliding window
    t1 = x223;
    for (int j = 0; j < 23; ++j) t1.square_inplace();
    t1 = t1 * x22;

    for (int j = 0; j < 5; ++j) t1.square_inplace();
    t1 = t1 * a;

    for (int j = 0; j < 3; ++j) t1.square_inplace();
    t1 = t1 * x2;

    for (int j = 0; j < 2; ++j) t1.square_inplace();
    t1 = t1 * a;

    return t1;
}

// --- FE52 Batch Inversion (Montgomery's trick) ------------------------------
// Computes z_inv[i] = z[i]^{-1} mod p for all i in [0, n).
// Cost: 1 inversion + 3(n-1) multiplications (vs n inversions).
// All z[i] MUST be non-zero (caller ensures by excluding infinity entries).
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
        std::uint64_t mask = eq_mask(static_cast<std::uint64_t>(i),
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
        std::uint64_t mask = eq_mask(static_cast<std::uint64_t>(i),
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
    std::uint64_t negative = ((n >> (group_size - 1)) ^ 1u) & 1u;
    std::uint64_t neg_mask = static_cast<std::uint64_t>(-negative);

    unsigned index_bits = group_size - 1;
    std::uint64_t index = (static_cast<std::uint64_t>(-negative) ^ n) &
                          ((1ULL << index_bits) - 1u);

    // Load first entry (80 bytes = x.n[5] + y.n[5]) into 3 ymm registers
    // Layout: [x.n[0..3]] [x.n[4], y.n[0..2]] [y.n[3..4], padding]
    const auto* base0 = reinterpret_cast<const char*>(&table[0].x.n[0]);
    __m256i r0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base0));           // x.n[0..3] (32 bytes)
    __m256i r1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base0 + 32));      // x.n[4], y.n[0..2]
    // Last 16 bytes: y.n[3], y.n[4]
    // Use 128-bit load for the remaining 16 bytes to avoid reading past struct
    __m128i r2lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(base0 + 64));
    __m256i r2 = _mm256_castsi128_si256(r2lo);

    // CT linear scan: cmov each entry using vectorized AND/XOR
    for (std::size_t m = 1; m < table_size; ++m) {
        std::uint64_t eq = eq_mask(static_cast<std::uint64_t>(m),
                                   static_cast<std::uint64_t>(index));
        __m256i vmask = _mm256_set1_epi64x(static_cast<long long>(eq));
        __m128i vmask128 = _mm_set1_epi64x(static_cast<long long>(eq));

        const auto* base_m = reinterpret_cast<const char*>(&table[m].x.n[0]);
        __m256i s0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base_m));
        __m256i s1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base_m + 32));
        __m128i s2lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(base_m + 64));
        __m256i s2 = _mm256_castsi128_si256(s2lo);

        // r = (r ^ s) & mask ^ r  ==  mask ? s : r
        r0 = _mm256_xor_si256(r0, _mm256_and_si256(_mm256_xor_si256(r0, s0), vmask));
        r1 = _mm256_xor_si256(r1, _mm256_and_si256(_mm256_xor_si256(r1, s1), vmask));
        // For the last partial register, use 128-bit ops
        __m128i r2_128 = _mm256_castsi256_si128(r2);
        __m128i s2_128 = _mm256_castsi256_si128(s2);
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
    std::uint64_t neg_mask = static_cast<std::uint64_t>(-negative);

    unsigned index_bits = group_size - 1;
    std::uint64_t index = (static_cast<std::uint64_t>(-negative) ^ n) &
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
    FE52 z1z1 = Z1.square();                           // Z1^2          [sqr #1]
    FE52 z2z2 = Z2.square();                           // Z2^2          [sqr #2]
    FE52 u1   = X1 * z2z2;                             // U1 = X1*Z2^2  [mul #1]
    FE52 u2   = X2 * z1z1;                             // U2 = X2*Z1^2  [mul #2]
    FE52 s1   = (Y1 * z2z2) * Z2;                      // S1 = Y1*Z2^3  [mul #3,#4]
    FE52 s2   = (Y2 * z1z1) * Z1;                      // S2 = Y2*Z1^3  [mul #5,#6]
    FE52 z    = Z1 * Z2;                               // Z  = Z1*Z2   [mul #7]

    // -- Unified formula: T = U1+U2, M = S1+S2 --
    FE52 t_val  = u1 + u2;                             // T = U1+U2            (M=2)
    FE52 m_val  = s1 + s2;                             // M = S1+S2            (M=2)

    // R = T^2 - U1*U2
    FE52 rr     = t_val.square();                       // T^2                   [sqr #3]
    FE52 neg_u2 = u2.negate(1);                         // -U2                  (M=2)
    FE52 tt     = u1 * neg_u2;                          // -U1*U2               [mul #8]
    rr = rr + tt;                                       // R = T^2-U1*U2         (M=2)

    // -- Degenerate case: M normalizes to zero ==> y1 = -y2 --
    std::uint64_t degen     = fe52_normalizes_to_zero(m_val);
    std::uint64_t not_degen = ~degen;

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

    FE52 tq  = (x3 + x3) + q_val;                      // 2*X3+Q               (M=5)
    FE52 y3p = (rr_alt * tq) + nn;                      // Ralt*(2X3+Q)+M^3Malt  [mul #11]
    FE52 y3  = y3p.negate(5).half();                    // Y3 = -(...)/2

    // -- Infinity handling --
    std::uint64_t z3_zero = fe52_normalizes_to_zero(z3);

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
    FE52 X1 = p.x;
    FE52 Y1 = p.y;
    FE52 Z1 = p.z;

    // -- Brier-Joye unified (Z2=1): 7M + 5S --
    FE52 zz      = Z1.square();                        // Z1^2          [sqr #1]
    FE52 u1      = X1;                                 // U1 = X1
    FE52 u2      = q.x * zz;                           // U2 = x2*Z1^2  [mul #1]
    FE52 s1      = Y1;                                 // S1 = Y1
    FE52 s2      = (q.y * zz) * Z1;                    // S2 = y2*Z1^3  [mul #2,#3]

    FE52 t_val   = u1 + u2;                            // T = U1+U2
    FE52 m_val   = s1 + s2;                            // M = S1+S2

    FE52 rr      = t_val.square();                     // T^2           [sqr #2]
    FE52 neg_u2  = u2.negate(1);
    FE52 tt      = u1 * neg_u2;                        // -U1*U2       [mul #4]
    rr = rr + tt;                                      // R = T^2-U1*U2

    std::uint64_t degen     = fe52_normalizes_to_zero(m_val);
    std::uint64_t not_degen = ~degen;

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

    FE52 tq      = (x3 + x3) + q_val;                  // 2*X3+Q
    FE52 y3p     = (rr_alt * tq) + nn;                 //               [mul #7]
    FE52 y3      = y3p.negate(5).half();                // Y3 = -(...)/2

    // -- Infinity handling --
    std::uint64_t z3_zero = fe52_normalizes_to_zero(z3);

    FE52 one52 = FE52::one();

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
    FE52 X = p.x;
    FE52 Y = p.y;
    FE52 Z = p.z;

    FE52 s = Y.square();                   // S = Y^2        [1S]  M=1
    FE52 l = X.square();                   // L = X^2        [1S]  M=1
    l = (l + l) + l;                       // L = 3*X^2            M=3
    l = l.half();                          // L = (3/2)*X^2        M=2
    FE52 t = s.negate(1) * X;             // T = -X*S      [1M]  M=1
    FE52 z3 = Z * Y;                       // Z3 = Y*Z      [1M]  M=1
    FE52 x3 = l.square();                  // X3 = L^2       [1S]  M=1
    x3 = x3 + t;                           //                     M=2
    x3 = x3 + t;                           // X3 = L^2+2T         M=3
    s = s.square();                        // S' = S^2        [1S]  M=1
    t = t + x3;                             // T' = X3+T          M=4
    FE52 y3 = (t * l) + s;                 // Y3 = L*T'+S'  [1M]  M=2
    y3 = y3.negate(2);                     // Y3 = -(...)         M=3

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
        FE52 A = X.square();                   // A = X^2         M=1
        FE52 B = Y.square();                   // B = Y^2         M=1
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
        FE52 yz = Y * Z;                       // Y*Z            M=1

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

template<bool CHECK_INFINITY>
__attribute__((always_inline)) inline
void unified_add_core(CTJacobianPoint* out,
                       const CTJacobianPoint& a,
                       const CTAffinePoint& b) noexcept {
    // Read a's coords into locals (handles out==&a aliasing, register-friendly).
    // NO normalize_weak -- magnitude bounds tracked explicitly:
    //   After dbl_n:      X M<=18, Y M<=10, Z M<=2
    //   After unified add: X M<=2,  Y M=1,   Z M=1
    // Worst case: M_x=18, M_y=10, M_z=2. All negate() params adjusted.
    FE52 X1 = a.x;
    FE52 Y1 = a.y;
    FE52 Z1 = a.z;
    [[maybe_unused]] std::uint64_t a_inf = a.infinity;
    (void)a_inf; // CodeQL: suppress unused-local-variable (kept for ABI symmetry)

    // -- Shared intermediates --
    FE52 zz = Z1.square();                          // Z1^2       [1S]  M=1
    FE52 u1 = X1;                                   // U1 = X1   M<=18
    FE52 u2 = b.x * zz;                             // U2 = x2*Z1^2  M=1  [1M]
    FE52 s1 = Y1;                                   // S1 = Y1   M<=10
    FE52 s2 = (b.y * zz) * Z1;                      // S2 = y2*Z1^3  M=1  [2M]

    FE52 t_val = u1;
    t_val.add_assign(u2);                            // T = U1+U2 M<=19
    FE52 m_val = s1;
    m_val.add_assign(s2);                            // M = S1+S2 M<=11

    // R = T^2 - U1*U2
    FE52 rr = t_val.square();                        // T^2  M=1      [1S]
    u2.negate_assign(1);                             // -U2 M=2 (u2 reused as neg_u2)
    FE52 tt = u1 * u2;                              // -U1*U2 M=1  [1M]
    rr.add_assign(tt);                               // R   M=2

    // -- Degenerate case: M~=0 (y1=-y2) --
    std::uint64_t degen = fe52_normalizes_to_zero(m_val);

    FE52 rr_alt = s1;
    rr_alt.add_assign(s1);                           // 2*S1   M<=20
    FE52 m_alt = u1;
    m_alt.add_assign(u2);                            // U1-U2 (u2 is negated)  M<=20

    fe52_cmov(&rr_alt, rr, ~degen);
    fe52_cmov(&m_alt, m_val, ~degen);

    // -- Compute result into locals (no pointer reads after partial writes) --
    FE52 n = m_alt.square();                         // Malt^2 M=1   [1S]
    t_val.negate_assign(19);                         // -T    M=20  (t_val reused)
    FE52 q = t_val * n;                              // Q     M=1   [1M]

    n = n.square();                                  // Malt^4 M=1   [1S]
    fe52_cmov(&n, m_val, degen);                     // if degen: n=M~=0  M<=11

    FE52 x3 = rr_alt.square();
    x3.add_assign(q);                                // X3    M=2   [1S]
    FE52 z3 = m_alt * Z1;                            // Z3    M=1   [1M]

    FE52 tq = x3;
    tq.add_assign(x3);                               // 2*X3  M=4
    tq.add_assign(q);                                // 2X3+Q M=5
    rr_alt.mul_assign(tq);                           // Ralt*(2X3+Q) M=1  [1M=7th]
    rr_alt.add_assign(n);                            // +M^3Malt      M<=12
    rr_alt.negate_assign(12);                        // -(...)        M=13
    FE52 y3 = rr_alt.half();                         // Y3    M=1 (half normalizes)

    if constexpr (CHECK_INFINITY) {
        // Handle a=infinity: replace result with (b.x, b.y, 1)
        FE52 one52 = FE52::one();
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
        std::uint64_t diff = a[i] - b[i];
        std::uint64_t b1 = static_cast<std::uint64_t>(a[i] < b[i]);
        std::uint64_t result = diff - borrow;
        std::uint64_t b2 = static_cast<std::uint64_t>(diff < borrow);
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
    std::uint64_t borrow = local_sub256(tmp, HALF_N, s.limbs().data());
    return is_nonzero_mask(borrow);
}

// --- CT 256x256->512 multiply, shift >>384 with rounding ---------------------
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

// --- Secp256k1 order-specific 512->256 scalar reduction ----------------------
// Reduces a 512-bit product modulo n = 2^256 - {NC0, NC1, 1, 0}.
// Three phases: 512->385->258->256 bits. Constant-time.
// Adapted from bitcoin-core/secp256k1 scalar_4x64_impl.h (MIT license).
static constexpr std::uint64_t ORDER[4] = {
    0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
};
static constexpr std::uint64_t NC0 = 0x402DA1732FC9BEBFULL; // ~ORDER[0]+1
static constexpr std::uint64_t NC1 = 0x4551231950B75FC4ULL; // ~ORDER[1]
// NC2 = 1, NC3 = 0

// --- Specialized 128x128 multiply mod n (for GLV lattice) -------------------
// Both operands at most 128-bit (2 limbs). Product <= 256 bits -> conditional
// subtract of n only (no multi-phase reduction).
static Scalar ct_mul_lo128_mod(const std::uint64_t a[2],
                                const std::uint64_t b[2]) noexcept {
    unsigned __int128 c = (unsigned __int128)a[0] * b[0];
    std::uint64_t r0 = (std::uint64_t)c; c >>= 64;

    c += (unsigned __int128)a[0] * b[1] + (unsigned __int128)a[1] * b[0];
    std::uint64_t r1 = (std::uint64_t)c; c >>= 64;

    c += (unsigned __int128)a[1] * b[1];
    std::uint64_t r2 = (std::uint64_t)c;
    std::uint64_t r3 = (std::uint64_t)(c >> 64);

    std::uint64_t r_arr[4] = {r0, r1, r2, r3};
    std::uint64_t sub[4];
    std::uint64_t borrow = local_sub256(sub, r_arr, ORDER);
    cmov256(r_arr, sub, is_zero_mask(borrow));

    return Scalar::from_limbs({r_arr[0], r_arr[1], r_arr[2], r_arr[3]});
}

// --- Specialized 256x128 multiply mod n (for lambda x k2_abs) ---------------
// a is full 256-bit, b at most 128-bit. Product <= 384 bits -> single-phase
// secp256k1-specific reduction (skip phase 1 since upper 128 bits of product
// are zero).
static Scalar ct_mul_256x_lo128_mod(const Scalar& a,
                                     const std::uint64_t b[2]) noexcept {
    const auto& al = a.limbs();

    // 4x2 schoolbook -> 6 limbs (column-accumulation, unrolled)
    unsigned __int128 c = (unsigned __int128)al[0] * b[0];
    std::uint64_t d0 = (std::uint64_t)c; c >>= 64;

    c += (unsigned __int128)al[0] * b[1] + (unsigned __int128)al[1] * b[0];
    std::uint64_t d1 = (std::uint64_t)c; c >>= 64;

    c += (unsigned __int128)al[1] * b[1] + (unsigned __int128)al[2] * b[0];
    std::uint64_t d2 = (std::uint64_t)c; c >>= 64;

    c += (unsigned __int128)al[2] * b[1] + (unsigned __int128)al[3] * b[0];
    std::uint64_t d3 = (std::uint64_t)c; c >>= 64;

    c += (unsigned __int128)al[3] * b[1];
    std::uint64_t d4 = (std::uint64_t)c;
    std::uint64_t d5 = (std::uint64_t)(c >> 64);

    // Single-phase reduction: r = d[0..3] + d[4..5] * NC
    c = (unsigned __int128)d4 * NC0 + d0;
    std::uint64_t r0 = (std::uint64_t)c; c >>= 64;

    c += (unsigned __int128)d5 * NC0 + (unsigned __int128)d4 * NC1 + d1;
    std::uint64_t r1 = (std::uint64_t)c; c >>= 64;

    c += (unsigned __int128)d5 * NC1 + d4 + d2;  // d4 * NC2=1
    std::uint64_t r2 = (std::uint64_t)c; c >>= 64;

    c += d5 + d3;  // d5 * NC2=1
    std::uint64_t r3 = (std::uint64_t)c;
    unsigned overflow = (unsigned)(c >> 64);

    std::uint64_t r_arr[4] = {r0, r1, r2, r3};
    std::uint64_t sub[4];
    std::uint64_t borrow = local_sub256(sub, r_arr, ORDER);
    cmov256(r_arr, sub, is_nonzero_mask(static_cast<std::uint64_t>(overflow))
                      | is_zero_mask(borrow));

    return Scalar::from_limbs({r_arr[0], r_arr[1], r_arr[2], r_arr[3]});
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
static constexpr std::array<std::uint8_t, 32> kLambdaBytes{{
    0x53,0x63,0xAD,0x4C,0xC0,0x5C,0x30,0xE0,
    0xA5,0x26,0x1C,0x02,0x88,0x12,0x64,0x5A,
    0x12,0x2E,0x22,0xEA,0x20,0x81,0x66,0x78,
    0xDF,0x02,0x96,0x7C,0x1B,0x23,0xBD,0x72
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

// --- CT GLV Decomposition (optimized: 128x128 + 256x128 paths) --------------

CTGLVDecomposition ct_glv_decompose(const Scalar& k) noexcept {
    static const Scalar lambda = Scalar::from_bytes(kLambdaBytes);

    // GLV lattice basis magnitudes (128-bit, positive).
    // minus_b1 = -b1 mod n ~= b1 (small, < 2^128)
    // b2_pos   = b2 (small, < 2^128), used as -(c2*b2) instead of c2*(-b2 mod n)
    static constexpr std::uint64_t minus_b1_lo[2] = {
        0x6F547FA90ABFE4C3ULL, 0xE4437ED6010E8828ULL
    };
    static constexpr std::uint64_t b2_pos_lo[2] = {
        0xE86C90E49284EB15ULL, 0x3086D221A7D46BCDULL
    };

    auto k_limbs = k.limbs();
    auto c1_limbs = ct_mul_shift_384({k_limbs[0], k_limbs[1], k_limbs[2], k_limbs[3]}, kG1);
    auto c2_limbs = ct_mul_shift_384({k_limbs[0], k_limbs[1], k_limbs[2], k_limbs[3]}, kG2);

    // c1, c2 have at most 128 bits (upper limbs zero from mul_shift_384).
    // k2 = c1*(-b1) + c2*(-b2) mod n = c1*(-b1) - c2*b2 mod n
    Scalar p1     = ct_mul_lo128_mod(c1_limbs.data(), minus_b1_lo);
    Scalar p2     = ct_mul_lo128_mod(c2_limbs.data(), b2_pos_lo);
    Scalar k2_mod = scalar_sub(p1, p2);

    std::uint64_t k2_high = ct_scalar_is_high(k2_mod);
    Scalar k2_abs = scalar_cneg(k2_mod, k2_high);

    // lambda x |k2| (256x128), then conditionally negate.
    // |k2| < 2^128 (GLV lattice guarantee), so upper 2 limbs are zero.
    Scalar lambda_k2 = scalar_cneg(
        ct_mul_256x_lo128_mod(lambda, k2_abs.limbs().data()), k2_high);
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

    // -- 1. Compute v1, v2 via Hamburg transform + GLV split --------------
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

    // -- 2. Build odd-multiples table via isomorphic curve + batch inversion --
    // Adapted from bitcoin-core/secp256k1 effective-affine technique:
    //
    // Let d = 2*P (Jacobian), C = d.Z. On the isomorphic curve
    //   Y'^2 = X'^3 + C^6*7
    // the mapping phi(X, Y, Z) = (X*C^2, Y*C^3, Z) makes d affine: (d.X, d.Y).
    // This lets us use Jac+Affine mixed adds (3S+8M) instead of Jac+Jac (5S+12M).
    // At the end, the "true" Z for each entry on secp256k1 is Z_iso * C.
    CTAffinePoint pre_a[TABLE_SIZE];
    CTAffinePoint pre_a_lam[TABLE_SIZE];

    // 2.1 -- Compute d = 2P and set up isomorphic curve
    Point p2 = p;
    p2.dbl_inplace();               // d = 2*P (Jacobian, variable-time OK: P is public)
    FE52 C  = p2.Z52();             // C = d.Z
    FE52 C2 = C.square();           // C^2
    FE52 C3 = C2 * C;               // C^3

    // d as affine on iso curve: phi(d) = (d.X, d.Y) (Z=C cancels in the isomorphism)
    FE52 d_x = p2.X52();
    FE52 d_y = p2.Y52();

    // 2.2 -- Transform P to iso: phi(P) = (P.X*C^2, P.Y*C^3, P.Z) in Jacobian on iso
    JacFE52 iso[TABLE_SIZE];
    iso[0] = { p.X52() * C2, p.Y52() * C3, p.Z52() };

    // 2.3 -- Build rest of table using mixed adds on iso curve (3S+8M each)
    //        Also track Z-ratios for backward normalization (no field inverse!).
    FE52 zr[TABLE_SIZE]; // Z-ratio: iso[i].z = iso[i-1].z * zr[i]
    for (std::size_t i = 1; i < TABLE_SIZE; ++i) {
        iso[i] = jac_add_ge_var_zr(iso[i - 1], d_x, d_y, &zr[i]);
    }

    // 2.4 -- Compute global Z for secp256k1 (iso->secp conversion)
    //        global_z = Z_last * C, where Z_last = iso[TABLE_SIZE-1].z
    //        All table entries share this implicit Z denominator.
    FE52 global_z = iso[TABLE_SIZE - 1].z * C;

    // 2.5 -- Backward normalization via Z-ratios (replaces batch inverse!)
    //
    // Principle: entry i's Z on iso curve = iso[0].z * zr[1] * ... * zr[i].
    // To normalize all entries to the same Z = Z_last, multiply entry i
    // by (zr[i+1] * ... * zr[n-1])^2 for x and ^3 for y.  Then ALL entries
    // effectively have Z = Z_last on the iso curve (or global_z on secp256k1).
    // The main loop treats them as affine -- correct up to the global Z factor.
    // Final correction: R.z *= global_z after the main loop.
    //
    // Cost: (TABLE_SIZE-2) muls (accumulate) + (TABLE_SIZE-1) x (1S + 2M)
    //   ~= 14M + 15S + 30M = 44M + 15S  (vs batch inverse: 30M + ~250S inverse)
    const FE52& beta = get_beta_fe52();

    // Last entry: already at Z_last, just store directly.
    pre_a[TABLE_SIZE - 1].x = iso[TABLE_SIZE - 1].x;
    pre_a[TABLE_SIZE - 1].y = iso[TABLE_SIZE - 1].y;
    pre_a[TABLE_SIZE - 1].y.normalize_weak();
    pre_a[TABLE_SIZE - 1].infinity = 0;
    pre_a_lam[TABLE_SIZE - 1].x = pre_a[TABLE_SIZE - 1].x * beta;
    pre_a_lam[TABLE_SIZE - 1].y = pre_a[TABLE_SIZE - 1].y;
    pre_a_lam[TABLE_SIZE - 1].infinity = 0;

    // Backward accumulation: zs = product of Z-ratios from end to current+1.
    FE52 zs_acc = zr[TABLE_SIZE - 1];
    for (int i = static_cast<int>(TABLE_SIZE) - 2; i >= 0; --i) {
        FE52 zs2 = zs_acc.square();
        FE52 zs3 = zs2 * zs_acc;
        pre_a[i].x = iso[i].x * zs2;
        pre_a[i].y = iso[i].y * zs3;
        pre_a[i].infinity = 0;
        pre_a_lam[i].x = pre_a[i].x * beta;
        pre_a_lam[i].y = pre_a[i].y;
        pre_a_lam[i].infinity = 0;
        if (i > 0) {
            zs_acc = zs_acc * zr[i];
        }
    }

    // NOTE: No table Y-negation needed.
    // Hamburg signed-digit encoding bakes the sign into v1/v2 (via the offset
    // and signed GLV decomposition). The lookup_signed function handles
    // conditional Y-negate at runtime based on each window's top bit.

    SECP256K1_DECLASSIFY(pre_a, sizeof(pre_a));
    SECP256K1_DECLASSIFY(pre_a_lam, sizeof(pre_a_lam));

    // -- 3. Main loop -- FULLY IN-PLACE, ALWAYS-INLINE hot ops ------------
    // Uses core functions (always_inline) to eliminate function-call overhead.
    // Uses CHECK_INFINITY=false since R and table entries are never infinity.
    // Uses NORMALIZE_Y=false since table entries have magnitude 1 (from mul).
    CTJacobianPoint R;
    CTAffinePoint t;  // reused across iterations

    // First group: set R directly from first lookup
    {
        int group = static_cast<int>(GROUPS) - 1;
        std::uint64_t bits1 = scalar_window(v1, static_cast<std::size_t>(group) * GROUP_SIZE, GROUP_SIZE);
        std::uint64_t bits2 = scalar_window(v2, static_cast<std::size_t>(group) * GROUP_SIZE, GROUP_SIZE);

        table_lookup_core<false>(&t, pre_a, TABLE_SIZE, bits1, GROUP_SIZE);
        R.x = t.x;
        R.y = t.y;
        R.z = FE52::one();
        R.infinity = 0;

        table_lookup_core<false>(&t, pre_a_lam, TABLE_SIZE, bits2, GROUP_SIZE);
        unified_add_core<true>(&R, R, t);   // first add: R.infinity might be relevant
    }

    // Remaining groups: in-place dbl_n + in-place adds (no infinity possible)
    // Prevent unrolling: the inlined body is large (~2.5KB per iteration);
    // unrolling 25 iterations would blow the L1 i-cache (32KB).
    #pragma clang loop unroll(disable)
    for (int group = static_cast<int>(GROUPS) - 2; group >= 0; --group) {
        std::uint64_t bits1 = scalar_window(v1, static_cast<std::size_t>(group) * GROUP_SIZE, GROUP_SIZE);
        std::uint64_t bits2 = scalar_window(v2, static_cast<std::size_t>(group) * GROUP_SIZE, GROUP_SIZE);

        // In-place batch doubling -- always-inline core
        point_dbl_n_core(&R, GROUP_SIZE);

        // In-place lookup + in-place add for pre_a -- core (no infinity check)
        table_lookup_core<false>(&t, pre_a, TABLE_SIZE, bits1, GROUP_SIZE);
        unified_add_core<false>(&R, R, t);

        // In-place lookup + in-place add for pre_a_lam -- core (no infinity check)
        table_lookup_core<false>(&t, pre_a_lam, TABLE_SIZE, bits2, GROUP_SIZE);
        unified_add_core<false>(&R, R, t);
    }

    // Apply global Z correction: undo the isomorphism.
    // The main loop computed on the iso curve with implicit Z = global_z.
    // Correct by: Z_secp256k1 = Z_loop * global_z.
    R.z = R.z * global_z;

    Point result = R.to_point();
    SECP256K1_DECLASSIFY(&result, sizeof(result));
    return result;
}

// --- CT Generator Multiplication (5x52) -- Comb Method -----------------------
// Uses comb method with signed digits (adapted from bitcoin-core/secp256k1).
//
// Parameters: COMB_TEETH=6, COMB_BLOCKS=43.
//   teeth x blocks = 258 >= 256 (2 extra bits corrected at end).
//
// Hamburg encoding: v = (k + K_gen) / 2 mod n, bits of v become {+1,-1} signs.
// Comb digit for block b: gather bit at positions {b, 43+b, 86+b, 129+b, 172+b, 215+b}
// from v -> 6-bit unsigned value -> signed table lookup (32 entries).
//
// Since 6x43=258 > 256, bits 256-257 of v are 0 (treated as sign=-1).
// Correction: add precomputed (2^256 + 2^257)*G after the main loop.
//
// Runtime: 42 additions + 43 lookups (31 cmovs each) + 1 correction add.
// vs Hamburg w=4: 63 additions + 64 lookups (7 cmovs each).
// Net: 33% fewer additions, larger lookups -> ~25% faster on ARM64.
//
// Table: 43 blocks x 32 entries = 1376 affine points ~= 108 KB.

namespace {

constexpr unsigned COMB_TEETH = 6;
constexpr unsigned COMB_BLOCKS = 43;   // ceil(256 / COMB_TEETH) = 43
constexpr std::size_t COMB_TABLE_SIZE = 1u << (COMB_TEETH - 1);  // 32

// Hamburg constant: K_gen = (2^256 - 1) mod n
static constexpr std::uint64_t K_GEN[4] = {
    0x402DA1732FC9BEBEULL,
    0x4551231950B75FC4ULL,
    0x0000000000000001ULL,
    0x0000000000000000ULL
};

struct alignas(64) CombGenTable {
    CTAffinePoint entries[COMB_BLOCKS][COMB_TABLE_SIZE];
    CTAffinePoint correction;  // (2^256 + 2^257)*G for 258-bit correction
    bool initialized = false;
};

static CombGenTable g_comb_table;
static std::once_flag g_comb_table_once;

// -- Extract 6-bit comb digit from scattered bit positions --------------------
// For block b: gather bit at position (tooth * COMB_BLOCKS + b) for each tooth.
// Bits beyond 255 are treated as 0 (v is 256 bits).
inline std::uint64_t extract_comb_digit(const Scalar& v, unsigned block) noexcept {
    std::uint64_t digit = 0;
    for (unsigned tooth = 0; tooth < COMB_TEETH; ++tooth) {
        std::size_t pos = static_cast<std::size_t>(tooth) * COMB_BLOCKS + block;
        std::uint64_t bit = (pos < 256) ? scalar_bit(v, pos) : 0;
        digit |= bit << tooth;
    }
    return digit;
}

// -- CT lookup from 32-entry signed comb table --------------------------------
// Table stores entries for unsigned values 32..63 (bit5=1).
// For values 0..31 (bit5=0): look up complement index and negate Y.
//
// Identity: table_u[d] = -table_u[63-d]  (complementary signs)
// table_s[idx] = table_u[idx | 32] = table_u[idx + 32]
//
// For din[32,63] (bit5=1): idx = d & 31, point = table_s[idx]
// For din[0,31]  (bit5=0): idx = 31-(d&31), point = -table_s[idx]
static inline
void comb_lookup(CTAffinePoint* out,
                  const CTAffinePoint* table,
                  std::uint64_t digit) noexcept {
    std::uint64_t top = (digit >> (COMB_TEETH - 1)) & 1;
    std::uint64_t needs_negate = top ^ 1;  // negate if top bit is 0
    std::uint64_t neg_mask = static_cast<std::uint64_t>(
                      -static_cast<std::int64_t>(needs_negate));

    std::uint64_t d_lo = digit & (COMB_TABLE_SIZE - 1);

    // index: top=1 -> d_lo; top=0 -> (TABLE_SIZE-1 - d_lo)
    std::uint64_t idx_pos = d_lo;
    std::uint64_t idx_neg = (COMB_TABLE_SIZE - 1) - d_lo;
    // Branchless select
    std::uint64_t index = idx_pos ^ ((idx_pos ^ idx_neg) & neg_mask);

    // CT linear scan over all 32 entries
    out->x = table[0].x;
    out->y = table[0].y;
    for (std::size_t m = 1; m < COMB_TABLE_SIZE; ++m) {
        std::uint64_t mask = eq_mask(static_cast<std::uint64_t>(m), index);
        fe52_cmov(&out->x, table[m].x, mask);
        fe52_cmov(&out->y, table[m].y, mask);
    }

    // Conditional Y-negate
    FE52 neg_y = out->y;
    neg_y = neg_y.negate(1);
    fe52_cmov(&out->y, neg_y, neg_mask);

    out->infinity = 0;
}

// -- Build comb table ---------------------------------------------------------
void build_comb_table() noexcept {
    Point G = Point::generator();

    // Step 1: Compute teeth base points: 2^(tooth * COMB_BLOCKS) * G
    //   teeth_bases[0] = G
    //   teeth_bases[1] = 2^43 * G
    //   teeth_bases[2] = 2^86 * G
    //   teeth_bases[j] = 2^(j*43) * G
    Point teeth_bases[COMB_TEETH];
    teeth_bases[0] = G;
    for (unsigned j = 1; j < COMB_TEETH; ++j) {
        teeth_bases[j] = teeth_bases[j - 1];
        for (unsigned d = 0; d < COMB_BLOCKS; ++d) {
            teeth_bases[j].dbl_inplace();
        }
    }

    // Step 2: For each block b, build signed table from 6 base points.
    // After building block b, double all bases for block b+1.
    for (unsigned b = 0; b < COMB_BLOCKS; ++b) {
        // Current state: teeth_bases[j] = 2^(j*43 + b) * G

        // Convert teeth base points to affine for efficient construction
        FE52 zs[COMB_TEETH], z_invs[COMB_TEETH];
        for (unsigned j = 0; j < COMB_TEETH; ++j) {
            zs[j] = teeth_bases[j].Z52();
        }
        fe52_batch_inverse(z_invs, zs, COMB_TEETH);

        FE52 aff_x[COMB_TEETH], aff_y[COMB_TEETH];
        for (unsigned j = 0; j < COMB_TEETH; ++j) {
            FE52 zi2 = z_invs[j].square();
            FE52 zi3 = zi2 * z_invs[j];
            aff_x[j] = teeth_bases[j].X52() * zi2;
            aff_y[j] = teeth_bases[j].Y52() * zi3;
            aff_x[j].normalize();
            aff_y[j].normalize();
        }

        // Build all 32 signed entries.
        // table_s[idx] = sum_{j=0}^{4} (2*bit_j(idx)-1)*P_j + (+1)*P_5
        // Computation: build each entry as sum of +/-P_j using Jacobian arithmetic.
        Point entries_jac[COMB_TABLE_SIZE];

        for (unsigned idx = 0; idx < COMB_TABLE_SIZE; ++idx) {
            // Start with +P_5 (always positive for tooth 5 in signed table)
            Point entry = Point::from_affine52(aff_x[COMB_TEETH - 1],
                                                aff_y[COMB_TEETH - 1]);

            for (unsigned j = 0; j < COMB_TEETH - 1; ++j) {
                FE52 py = ((idx >> j) & 1) ? aff_y[j] : aff_y[j].negate(1);
                Point pj = Point::from_affine52(aff_x[j], py);
                entry = entry.add(pj);
            }
            entries_jac[idx] = entry;
        }

        // Batch-invert Z coordinates and store as affine
        FE52 ent_zs[COMB_TABLE_SIZE], ent_zis[COMB_TABLE_SIZE];
        for (unsigned idx = 0; idx < COMB_TABLE_SIZE; ++idx) {
            ent_zs[idx] = entries_jac[idx].Z52();
        }
        fe52_batch_inverse(ent_zis, ent_zs, COMB_TABLE_SIZE);

        for (unsigned idx = 0; idx < COMB_TABLE_SIZE; ++idx) {
            FE52 zi2 = ent_zis[idx].square();
            FE52 zi3 = zi2 * ent_zis[idx];
            g_comb_table.entries[b][idx].x = entries_jac[idx].X52() * zi2;
            g_comb_table.entries[b][idx].y = entries_jac[idx].Y52() * zi3;
            g_comb_table.entries[b][idx].x.normalize();
            g_comb_table.entries[b][idx].y.normalize();
            g_comb_table.entries[b][idx].infinity = 0;
        }

        // Advance all base points by one doubling for next block
        if (b + 1 < COMB_BLOCKS) {
            for (unsigned j = 0; j < COMB_TEETH; ++j) {
                teeth_bases[j].dbl_inplace();
            }
        }
    }

    // Step 3: Precompute correction point = (2^256 + 2^257) * G
    // The comb covers 258 bits but v has only 256. Bits 256-257 are 0,
    // interpreted as sign=-1 in Hamburg encoding. This adds an unwanted
    // -(2^256 + 2^257)*G. We correct by adding (2^256 + 2^257)*G at the end.
    //
    // 2^256*G: double G 256 times.
    // 2^257*G = 2 * 2^256*G.
    // correction = 2^256*G + 2^257*G = 3 * 2^256*G
    Point p256 = G;
    for (unsigned d = 0; d < 256; ++d) {
        p256.dbl_inplace();
    }
    Point p257 = p256;
    p257.dbl_inplace();
    Point corr = p256.add(p257);
    // Store as affine
    FE52 cz_inv = fe52_inverse(corr.Z52());
    FE52 cz2 = cz_inv.square();
    FE52 cz3 = cz2 * cz_inv;
    g_comb_table.correction.x = corr.X52() * cz2;
    g_comb_table.correction.y = corr.Y52() * cz3;
    g_comb_table.correction.x.normalize();
    g_comb_table.correction.y.normalize();
    g_comb_table.correction.infinity = 0;

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

    Scalar s = scalar_add(k, K_gen_scalar);
    Scalar v = scalar_half(s);

    CTJacobianPoint R;
    CTAffinePoint T;

    // -- Main loop: 43 comb blocks ----------------------------------------
    // First block: initialize R directly from table lookup
    {
        std::uint64_t digit = extract_comb_digit(v, 0);
        comb_lookup(&T, g_comb_table.entries[0], digit);
        R.x = T.x;
        R.y = T.y;
        R.z = FE52::one();
        R.infinity = 0;
    }

    // Remaining 42 blocks: lookup + unified add
    #pragma clang loop unroll(disable)
    for (unsigned b = 1; b < COMB_BLOCKS; ++b) {
        std::uint64_t digit = extract_comb_digit(v, b);
        comb_lookup(&T, g_comb_table.entries[b], digit);
        unified_add_core<false>(&R, R, T);
    }

    // -- Correction: add (2^256 + 2^257)*G for the 2 extra comb bits -----
    unified_add_core<false>(&R, R, g_comb_table.correction);

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

// --- Montgomery batch inversion ----------------------------------------------

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
    std::uint64_t neg_mask = static_cast<std::uint64_t>(-negative);

    unsigned index_bits = group_size - 1;
    std::uint64_t index = (static_cast<std::uint64_t>(-negative) ^ n) &
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
    FE52 X1 = p.x, Y1 = p.y, Z1 = p.z;

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

static constexpr std::uint64_t ORDER[4] = {
    0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
};
static constexpr std::uint64_t NC0 = 0x402DA1732FC9BEBFULL;
static constexpr std::uint64_t NC1 = 0x4551231950B75FC4ULL;

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

// --- CT 128x128 -> mod n (portable) ------------------------------------------

static Scalar ct_mul_lo128_mod(const std::uint64_t a[2],
                                const std::uint64_t b[2]) noexcept {
    U128 p00 = mul64(a[0], b[0]);
    U128 p01 = mul64(a[0], b[1]);
    U128 p10 = mul64(a[1], b[0]);
    U128 p11 = mul64(a[1], b[1]);

    std::uint64_t r0 = p00.lo;
    std::uint64_t c = 0;
    std::uint64_t r1 = p00.hi + p01.lo;
    c = static_cast<std::uint64_t>(r1 < p00.hi);
    std::uint64_t t = r1 + p10.lo;
    c += static_cast<std::uint64_t>(t < r1);
    r1 = t;

    std::uint64_t r2 = p01.hi + p10.hi;
    std::uint64_t c2 = static_cast<std::uint64_t>(r2 < p01.hi);
    t = r2 + p11.lo;  c2 += static_cast<std::uint64_t>(t < r2);
    t += c;            c2 += static_cast<std::uint64_t>(t < c);
    r2 = t;

    std::uint64_t r3 = p11.hi + c2;

    std::uint64_t r_arr[4] = {r0, r1, r2, r3};
    std::uint64_t sub[4];
    std::uint64_t borrow = local_sub256(sub, r_arr, ORDER);
    cmov256(r_arr, sub, is_zero_mask(borrow));

    return Scalar::from_limbs({r_arr[0], r_arr[1], r_arr[2], r_arr[3]});
}

// --- CT 256x128 -> mod n (portable, secp256k1-specific reduction) ------------

static Scalar ct_mul_256x_lo128_mod(const Scalar& a,
                                     const std::uint64_t b[2]) noexcept {
    const auto& al = a.limbs();

    // 4x2 schoolbook -> 6 limbs
    std::uint64_t d[6] = {};
    for (int i = 0; i < 4; ++i) {
        std::uint64_t carry = 0;
        for (int j = 0; j < 2; ++j) {
            U128 m = mul64(al[i], b[j]);
            std::uint64_t sum = d[i + j] + m.lo;
            std::uint64_t c1 = static_cast<std::uint64_t>(sum < d[i + j]);
            sum += carry;
            std::uint64_t c2 = static_cast<std::uint64_t>(sum < carry);
            d[i + j] = sum;
            carry = m.hi + c1 + c2;
        }
        for (int k = i + 2; k < 6; ++k) {
            std::uint64_t old = d[k];
            d[k] += carry;
            carry = static_cast<std::uint64_t>(d[k] < old);
        }
    }

    // Single-phase reduction: r = d[0..3] + d[4..5] * NC
    U128 m;
    std::uint64_t sum, s_hi, c_hi;

    m = mul64(d[4], NC0);
    sum = m.lo + d[0];
    c_hi = m.hi + static_cast<std::uint64_t>(sum < m.lo);
    std::uint64_t r0 = sum;

    U128 m1 = mul64(d[5], NC0);
    U128 m2 = mul64(d[4], NC1);
    sum = m1.lo + m2.lo;
    s_hi = m1.hi + m2.hi + static_cast<std::uint64_t>(sum < m1.lo);
    sum += d[1]; s_hi += static_cast<std::uint64_t>(sum < d[1]);
    sum += c_hi; s_hi += static_cast<std::uint64_t>(sum < c_hi);
    std::uint64_t r1 = sum;
    c_hi = s_hi;

    m = mul64(d[5], NC1);
    sum = m.lo + d[4];
    s_hi = m.hi + static_cast<std::uint64_t>(sum < m.lo);
    sum += d[2]; s_hi += static_cast<std::uint64_t>(sum < d[2]);
    sum += c_hi; s_hi += static_cast<std::uint64_t>(sum < c_hi);
    std::uint64_t r2 = sum;
    c_hi = s_hi;

    sum = d[5] + d[3];
    s_hi = static_cast<std::uint64_t>(sum < d[5]);
    sum += c_hi; s_hi += static_cast<std::uint64_t>(sum < c_hi);
    std::uint64_t r3 = sum;
    unsigned overflow = static_cast<unsigned>(s_hi);

    std::uint64_t r_arr[4] = {r0, r1, r2, r3};
    std::uint64_t sub[4];
    std::uint64_t borrow = local_sub256(sub, r_arr, ORDER);
    cmov256(r_arr, sub, is_nonzero_mask(static_cast<std::uint64_t>(overflow))
                      | is_zero_mask(borrow));

    return Scalar::from_limbs({r_arr[0], r_arr[1], r_arr[2], r_arr[3]});
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
static constexpr std::array<std::uint8_t, 32> kLambdaBytes{{
    0x53,0x63,0xAD,0x4C,0xC0,0x5C,0x30,0xE0,
    0xA5,0x26,0x1C,0x02,0x88,0x12,0x64,0x5A,
    0x12,0x2E,0x22,0xEA,0x20,0x81,0x66,0x78,
    0xDF,0x02,0x96,0x7C,0x1B,0x23,0xBD,0x72
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
    static const Scalar lambda = Scalar::from_bytes(kLambdaBytes);

    static constexpr std::uint64_t minus_b1_lo[2] = {
        0x6F547FA90ABFE4C3ULL, 0xE4437ED6010E8828ULL
    };
    static constexpr std::uint64_t b2_pos_lo[2] = {
        0xE86C90E49284EB15ULL, 0x3086D221A7D46BCDULL
    };

    auto k_limbs = k.limbs();
    auto c1_limbs = ct_mul_shift_384(
        {k_limbs[0], k_limbs[1], k_limbs[2], k_limbs[3]}, kG1);
    auto c2_limbs = ct_mul_shift_384(
        {k_limbs[0], k_limbs[1], k_limbs[2], k_limbs[3]}, kG2);

    Scalar p1     = ct_mul_lo128_mod(c1_limbs.data(), minus_b1_lo);
    Scalar p2     = ct_mul_lo128_mod(c2_limbs.data(), b2_pos_lo);
    Scalar k2_mod = scalar_sub(p1, p2);

    std::uint64_t k2_high = ct_scalar_is_high(k2_mod);
    Scalar k2_abs = scalar_cneg(k2_mod, k2_high);

    Scalar lambda_k2 = scalar_cneg(
        ct_mul_256x_lo128_mod(lambda, k2_abs.limbs().data()), k2_high);
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
    static const Scalar S_OFFSET = Scalar::from_limbs({0, 0, 1, 0});

    // 1. Hamburg transform + GLV split
    Scalar s = scalar_add(k, K_CONST);
    s = scalar_half(s);

    auto [k1_abs, k2_abs, k1_neg, k2_neg] = ct_glv_decompose(s);
    Scalar s1 = scalar_cneg(k1_abs, k1_neg);
    Scalar s2 = scalar_cneg(k2_abs, k2_neg);
    Scalar v1 = scalar_add(s1, S_OFFSET);
    Scalar v2 = scalar_add(s2, S_OFFSET);

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
// COMB_TEETH=6, COMB_BLOCKS=43.  teeth x blocks = 258 >= 256.
// Table: 43 blocks x 32 entries = 1376 affine points.
// Runtime: 42 additions + 43 lookups + 1 correction.

namespace {

constexpr unsigned COMB_TEETH = 6;
constexpr unsigned COMB_BLOCKS = 43;
constexpr std::size_t COMB_TABLE_SIZE = 1u << (COMB_TEETH - 1);  // 32

static constexpr std::uint64_t K_GEN[4] = {
    0x402DA1732FC9BEBEULL, 0x4551231950B75FC4ULL,
    0x0000000000000001ULL, 0x0000000000000000ULL
};

struct alignas(64) CombGenTable {
    CTAffinePoint entries[COMB_BLOCKS][COMB_TABLE_SIZE];
    CTAffinePoint correction;
    bool initialized = false;
};

static CombGenTable g_comb_table;
static std::once_flag g_comb_table_once;

inline std::uint64_t extract_comb_digit(const Scalar& v,
                                         unsigned block) noexcept {
    std::uint64_t digit = 0;
    for (unsigned tooth = 0; tooth < COMB_TEETH; ++tooth) {
        std::size_t pos = static_cast<std::size_t>(tooth) * COMB_BLOCKS + block;
        std::uint64_t bit = (pos < 256) ? scalar_bit(v, pos) : 0;
        digit |= bit << tooth;
    }
    return digit;
}

static inline
void comb_lookup(CTAffinePoint* out,
                  const CTAffinePoint* table,
                  std::uint64_t digit) noexcept {
    std::uint64_t top = (digit >> (COMB_TEETH - 1)) & 1;
    std::uint64_t needs_negate = top ^ 1;
    std::uint64_t neg_mask = static_cast<std::uint64_t>(
                      -static_cast<std::int64_t>(needs_negate));

    std::uint64_t d_lo = digit & (COMB_TABLE_SIZE - 1);
    std::uint64_t idx_pos = d_lo;
    std::uint64_t idx_neg = (COMB_TABLE_SIZE - 1) - d_lo;
    std::uint64_t index = idx_pos ^ ((idx_pos ^ idx_neg) & neg_mask);

    out->x = table[0].x;
    out->y = table[0].y;
    for (std::size_t m = 1; m < COMB_TABLE_SIZE; ++m) {
        std::uint64_t mask = eq_mask(static_cast<std::uint64_t>(m), index);
        field_cmov(&out->x, table[m].x, mask);
        field_cmov(&out->y, table[m].y, mask);
    }

    FE52 neg_y = field_neg(out->y);
    field_cmov(&out->y, neg_y, neg_mask);
    out->infinity = 0;
}

void build_comb_table() noexcept {
    Point G = Point::generator();

    Point teeth_bases[COMB_TEETH];
    teeth_bases[0] = G;
    for (unsigned j = 1; j < COMB_TEETH; ++j) {
        teeth_bases[j] = teeth_bases[j - 1];
        for (unsigned d = 0; d < COMB_BLOCKS; ++d) {
            teeth_bases[j].dbl_inplace();
        }
    }

    for (unsigned b = 0; b < COMB_BLOCKS; ++b) {
        FE52 zs[COMB_TEETH], z_invs[COMB_TEETH];
        for (unsigned j = 0; j < COMB_TEETH; ++j)
            zs[j] = teeth_bases[j].z();
        fe_batch_inverse(z_invs, zs, COMB_TEETH);

        FE52 aff_x[COMB_TEETH], aff_y[COMB_TEETH];
        for (unsigned j = 0; j < COMB_TEETH; ++j) {
            FE52 zi2 = field_sqr(z_invs[j]);
            FE52 zi3 = field_mul(zi2, z_invs[j]);
            aff_x[j] = field_mul(teeth_bases[j].X(), zi2);
            aff_y[j] = field_mul(teeth_bases[j].Y(), zi3);
        }

        Point entries_jac[COMB_TABLE_SIZE];
        for (unsigned idx = 0; idx < COMB_TABLE_SIZE; ++idx) {
            Point entry = Point::from_affine(aff_x[COMB_TEETH - 1],
                                              aff_y[COMB_TEETH - 1]);
            for (unsigned j = 0; j < COMB_TEETH - 1; ++j) {
                FE52 py = ((idx >> j) & 1) ? aff_y[j] : field_neg(aff_y[j]);
                Point pj = Point::from_affine(aff_x[j], py);
                entry = entry.add(pj);
            }
            entries_jac[idx] = entry;
        }

        FE52 ent_zs[COMB_TABLE_SIZE], ent_zis[COMB_TABLE_SIZE];
        for (unsigned idx = 0; idx < COMB_TABLE_SIZE; ++idx)
            ent_zs[idx] = entries_jac[idx].z();
        fe_batch_inverse(ent_zis, ent_zs, COMB_TABLE_SIZE);

        for (unsigned idx = 0; idx < COMB_TABLE_SIZE; ++idx) {
            FE52 zi2 = field_sqr(ent_zis[idx]);
            FE52 zi3 = field_mul(zi2, ent_zis[idx]);
            g_comb_table.entries[b][idx].x = field_mul(
                entries_jac[idx].X(), zi2);
            g_comb_table.entries[b][idx].y = field_mul(
                entries_jac[idx].Y(), zi3);
            g_comb_table.entries[b][idx].infinity = 0;
        }

        if (b + 1 < COMB_BLOCKS) {
            for (unsigned j = 0; j < COMB_TEETH; ++j)
                teeth_bases[j].dbl_inplace();
        }
    }

    // Correction point: (2^256 + 2^257)*G = 3 * 2^256 * G
    Point p256 = G;
    for (unsigned d = 0; d < 256; ++d)
        p256.dbl_inplace();
    Point p257 = p256;
    p257.dbl_inplace();
    Point corr = p256.add(p257);
    FE52 cz_inv = field_inv(corr.z());
    FE52 cz2 = field_sqr(cz_inv);
    FE52 cz3 = field_mul(cz2, cz_inv);
    g_comb_table.correction.x = field_mul(corr.X(), cz2);
    g_comb_table.correction.y = field_mul(corr.Y(), cz3);
    g_comb_table.correction.infinity = 0;

    g_comb_table.initialized = true;
}

} // anonymous namespace

void init_generator_table() noexcept {
    std::call_once(g_comb_table_once, build_comb_table);
}

Point generator_mul(const Scalar& k) noexcept {
    init_generator_table();

    static const Scalar K_gen_scalar = Scalar::from_limbs(
        {K_GEN[0], K_GEN[1], K_GEN[2], K_GEN[3]});

    Scalar s = scalar_add(k, K_gen_scalar);
    Scalar v = scalar_half(s);

    CTJacobianPoint R;
    CTAffinePoint T;

    {
        std::uint64_t digit = extract_comb_digit(v, 0);
        comb_lookup(&T, g_comb_table.entries[0], digit);
        R.x = T.x;  R.y = T.y;  R.z = FieldElement::one();  R.infinity = 0;
    }

    for (unsigned b = 1; b < COMB_BLOCKS; ++b) {
        std::uint64_t digit = extract_comb_digit(v, b);
        comb_lookup(&T, g_comb_table.entries[b], digit);
        unified_add_core<false>(&R, R, T);
    }

    unified_add_core<false>(&R, R, g_comb_table.correction);

    Point result = R.to_point();
    SECP256K1_DECLASSIFY(&result, sizeof(result));
    return result;
}

#endif // SECP256K1_FAST_52BIT

// --- CT Curve Check (uses 4x64 FieldElement at API boundary) -----------------

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

// --- CT Point Equality (uses 4x64 at API boundary) --------------------------

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
