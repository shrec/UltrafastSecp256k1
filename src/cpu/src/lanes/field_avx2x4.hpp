// ============================================================================
// lanes/field_avx2x4.hpp -- Avx2x4: 4 field elements per ymm, radix 2^26
// ============================================================================
//
// Include ONLY from a translation unit compiled with -mavx2 (MSVC:
// /arch:AVX2), i.e. src/cpu/src/lanes/lanes_avx2.cpp. All functions have
// internal linkage (see field_common.hpp for why).
//
// PUBLIC DATA ONLY: variable time by design (verification paths). Not for
// private keys, nonces or any other secret.
//
// ---- Why radix 2^26 x 10 limbs -------------------------------------------------
//
// AVX2 has no 52-bit multiplier; its only 64-bit-result vector multiply is
// vpmuludq (_mm256_mul_epu32), 32 x 32 -> 64 per 64-bit lane. Each limb lives
// in its own 64-bit lane-vector, so a product of two limbs must fit 32 x 32.
//
//   * 10 x 26 bits: a 26 x 26 product is < 2^52, so a column of ten products
//     stays < 2^56 -- eight bits of headroom per 64-bit lane for the fold terms
//     and carries. 2^260 == 0x1000003D10 = 0x400 * 2^26 + 0x3D10 (mod p): the
//     fold of a high limb is one vpmuludq by 0x3D10 plus a shift by 10, and
//     2^256 == 0x40 * 2^26 + 0x3D1 is shifts only. 100 products per mul,
//     55 per square.
//   * 9 x 29 bits (261 bits): 81 products, but a 29 x 29 product is < 2^58 and
//     nine of them reach 2^61.2, leaving under three bits of headroom, and the
//     fold constant 2^261 mod p = 0x200007A20 needs 34 bits -- more than
//     vpmuludq's 32-bit operand -- so every fold needs an extra split.
//
// 10 x 26 is the libsecp256k1 32-bit layout; the 19 saved products of 9 x 29
// would be spent again on the awkward reduction.
//
// ---- Representation and LIMB-BOUND INVARIANT ------------------------------------
//
// Per lane v = sum n_i * 2^(26 i), i = 0..9. Weak normal form, guaranteed on
// every output of every operation (load, add, sub, negate, mul_small, mul,
// sqr, normalize, select, inverse):
//
//   n0..n8 <= 2^26 - 1,   n9 <= 2^22            (so v <= 2^256 + 2^234 - 1)
//
// vpmuludq reads only the low 32 bits of each lane, so mul/sqr need every
// limb < 2^32 (and the doubled limbs of sqr < 2^32); the invariant gives
// 2^27 at most. It is enforced BY CONSTRUCTION -- every op ends in
// r26_carry_weak() -- and checked by fe_bound_ok(), which the test kernels
// (lanes_kernels.hpp) assert on every mul/sqr input and every op result.
//
// Bounds inside the ops: mul columns < 10 * 2^52 + 2^41 < 2^56; sqr columns
// < 5 * 2^53 + 2^52 < 2^56; add < 2^27; sub/negate a + 4p - b < 2^29;
// mul_small (k < 2^32) < 2^58. r26_carry_weak accepts any limbs < 2^63.
// ============================================================================

#ifndef SECP256K1_LANES_FIELD_AVX2X4_HPP
#define SECP256K1_LANES_FIELD_AVX2X4_HPP

#if (defined(__GNUC__) || defined(__clang__)) && !defined(__AVX2__)
#error "field_avx2x4.hpp needs -mavx2 (include it only from lanes_avx2.cpp)"
#endif

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

#include "field_common.hpp"

namespace secp256k1::lanes::field {

struct Avx2x4 {
    using Vec = __m256i;
    using Mask = __m256i;   // all-ones lanes are "set"
    static constexpr unsigned kLanes = 4;
    static constexpr unsigned kLimbBits = 26;
    static constexpr unsigned kLimbCount = 10;
    static constexpr std::uint64_t kTopLimbMax = std::uint64_t{1} << 22;  // inclusive

    Vec n[10];

    static unsigned mask_bits(Mask m) noexcept {
        return static_cast<unsigned>(_mm256_movemask_pd(_mm256_castsi256_pd(m)));
    }
    static Mask mask_from_bits(unsigned bits) noexcept {
        return _mm256_set_epi64x((bits & 8u) ? -1LL : 0LL, (bits & 4u) ? -1LL : 0LL,
                                 (bits & 2u) ? -1LL : 0LL, (bits & 1u) ? -1LL : 0LL);
    }
};

namespace r26 {
constexpr std::uint64_t kM26 = 0x3FFFFFFULL;
constexpr std::uint64_t kM22 = 0x3FFFFFULL;
constexpr std::uint64_t kR0 = 0x3D10ULL;   // 2^260 mod p = 0x400 * 2^26 + 0x3D10
constexpr std::uint64_t kK0 = 0x3D1ULL;    // 2^256 mod p = 0x40  * 2^26 + 0x3D1
// 4p in radix 2^26. Every weak-normal limb is below the matching 4p limb
// (n0..n8 < 2^26 < 4p_i, n9 <= 2^22 < 4p_9 = 2^24 - 4).
constexpr std::uint64_t kP4[10] = {
    0xFFFF0BCULL,   // 4 * 0x3FFFC2F
    0xFFFFEFCULL,   // 4 * 0x3FFFFBF
    0xFFFFFFCULL, 0xFFFFFFCULL, 0xFFFFFFCULL, 0xFFFFFFCULL,
    0xFFFFFFCULL, 0xFFFFFFCULL, 0xFFFFFFCULL,   // 4 * 0x3FFFFFF
    0xFFFFFCULL,    // 4 * 0x3FFFFF
};

static inline __m256i set1(std::uint64_t x) noexcept {
    return _mm256_set1_epi64x(static_cast<long long>(x));
}

// One carry pass over limbs 0..8 (limb 9 keeps the overflow).
static inline void carry_pass(__m256i* c) noexcept {
    const __m256i m26 = set1(kM26);
    for (int i = 0; i < 9; ++i) {
        c[i + 1] = _mm256_add_epi64(c[i + 1], _mm256_srli_epi64(c[i], 26));
        c[i] = _mm256_and_si256(c[i], m26);
    }
}

// t * 0x3D1 (= 977 = 1024 - 32 - 16 + 1) with shifts; t < 2^54.
static inline __m256i mul_977(__m256i t) noexcept {
    const __m256i plus = _mm256_add_epi64(_mm256_slli_epi64(t, 10), t);
    const __m256i minus = _mm256_add_epi64(_mm256_slli_epi64(t, 5), _mm256_slli_epi64(t, 4));
    return _mm256_sub_epi64(plus, minus);
}
} // namespace r26

// Carry ten limbs (each < 2^63) back to weak normal form. The first pass
// leaves n0..n8 < 2^26 and folds t = n9 >> 22 (t < 2^41) back as
// t * (0x40 * 2^26 + 0x3D1); the second pass then carries < 2^27 out of
// limb 0, < 2^23 out of limb 1 and at most 1 out of every later limb, so n9
// ends <= 2^22 - 1 + 1 = 2^22.
static inline void r26_carry_weak(__m256i* c) noexcept {
    r26::carry_pass(c);
    const __m256i t = _mm256_srli_epi64(c[9], 22);
    c[9] = _mm256_and_si256(c[9], r26::set1(r26::kM22));
    c[0] = _mm256_add_epi64(c[0], r26::mul_977(t));
    c[1] = _mm256_add_epi64(c[1], _mm256_slli_epi64(t, 6));
    r26::carry_pass(c);
}

// Reduce 19 product columns (column k has weight 2^(26k), each < 2^56):
//   1. carry columns 10..18 into exact 26-bit limbs (column 19 = overflow,
//      < 2^20), so each is a valid 32-bit vpmuludq operand;
//   2. fold column k >= 10 down by 2^260 == 0x400 * 2^26 + 0x3D10:
//      c[k-10] += c[k] * 0x3D10, c[k-9] += c[k] << 10. Column 19 lands on
//      column 10 again (< 2^30) and is folded once more the same way;
//   3. r26_carry_weak on the low ten columns.
static inline Avx2x4 r26_reduce(__m256i* c) noexcept {
    const __m256i m26 = r26::set1(r26::kM26);
    const __m256i R0 = r26::set1(r26::kR0);
    for (int k = 10; k < 18; ++k) {
        c[k + 1] = _mm256_add_epi64(c[k + 1], _mm256_srli_epi64(c[k], 26));
        c[k] = _mm256_and_si256(c[k], m26);
    }
    const __m256i c19 = _mm256_srli_epi64(c[18], 26);
    c[18] = _mm256_and_si256(c[18], m26);
    for (int k = 10; k < 19; ++k) {
        c[k - 10] = _mm256_add_epi64(c[k - 10], _mm256_mul_epu32(c[k], R0));
        c[k - 9] = _mm256_add_epi64(c[k - 9], _mm256_slli_epi64(c[k], 10));
    }
    c[9] = _mm256_add_epi64(c[9], _mm256_mul_epu32(c19, R0));
    const __m256i c10 = _mm256_slli_epi64(c19, 10);
    c[0] = _mm256_add_epi64(c[0], _mm256_mul_epu32(c10, R0));
    c[1] = _mm256_add_epi64(c[1], _mm256_slli_epi64(c10, 10));
    r26_carry_weak(c);
    Avx2x4 r;
    for (int i = 0; i < 10; ++i) r.n[i] = c[i];
    return r;
}

// ---- Load / store -----------------------------------------------------------

// Load four elements, four uint64 limbs each (fast::FieldElement layout),
// from in[4*lane + i]. Any 256-bit value is accepted (n9 <= 2^22 - 1).
static inline void fe_load(Avx2x4& r, const std::uint64_t* in) noexcept {
    alignas(32) std::uint64_t t[10][4];
    for (unsigned l = 0; l < 4u; ++l) {
        std::uint64_t limbs[10];
        lanes_split_limbs(in + 4u * l, limbs, 26u, 10u);
        for (int i = 0; i < 10; ++i) t[i][l] = limbs[i];
    }
    for (int i = 0; i < 10; ++i) r.n[i] = _mm256_load_si256(reinterpret_cast<const __m256i*>(t[i]));
}

// Load internal-form limbs, ten per element (in[10*lane + i]). The caller
// guarantees weak normal form.
static inline void fe_load_limbs(Avx2x4& r, const std::uint64_t* in) noexcept {
    alignas(32) std::uint64_t t[10][4];
    for (unsigned l = 0; l < 4u; ++l)
        for (int i = 0; i < 10; ++i) t[i][l] = in[10u * l + static_cast<unsigned>(i)];
    for (int i = 0; i < 10; ++i) r.n[i] = _mm256_load_si256(reinterpret_cast<const __m256i*>(t[i]));
}

// Write the limbs as they are (no normalization). Exact only when every limb
// is < 2^26 and the value is < 2^256 (e.g. after fe_normalize).
static inline void fe_store_raw(std::uint64_t* out, const Avx2x4& a) noexcept {
    alignas(32) std::uint64_t t[10][4];
    for (int i = 0; i < 10; ++i) _mm256_store_si256(reinterpret_cast<__m256i*>(t[i]), a.n[i]);
    for (unsigned l = 0; l < 4u; ++l) {
        std::uint64_t limbs[10];
        for (int i = 0; i < 10; ++i) limbs[i] = t[i][l];
        lanes_join_limbs(limbs, out + 4u * l, 26u, 10u);
    }
}

// ---- Arithmetic --------------------------------------------------------------

static inline Avx2x4 fe_add(const Avx2x4& a, const Avx2x4& b) noexcept {
    __m256i c[10];
    for (int i = 0; i < 10; ++i) c[i] = _mm256_add_epi64(a.n[i], b.n[i]);
    r26_carry_weak(c);
    Avx2x4 r;
    for (int i = 0; i < 10; ++i) r.n[i] = c[i];
    return r;
}

// a - b computed as a + 4p - b (no limb can go negative, see kP4).
static inline Avx2x4 fe_sub(const Avx2x4& a, const Avx2x4& b) noexcept {
    __m256i c[10];
    for (int i = 0; i < 10; ++i)
        c[i] = _mm256_sub_epi64(_mm256_add_epi64(a.n[i], r26::set1(r26::kP4[i])), b.n[i]);
    r26_carry_weak(c);
    Avx2x4 r;
    for (int i = 0; i < 10; ++i) r.n[i] = c[i];
    return r;
}

static inline Avx2x4 fe_negate(const Avx2x4& a) noexcept {
    __m256i c[10];
    for (int i = 0; i < 10; ++i) c[i] = _mm256_sub_epi64(r26::set1(r26::kP4[i]), a.n[i]);
    r26_carry_weak(c);
    Avx2x4 r;
    for (int i = 0; i < 10; ++i) r.n[i] = c[i];
    return r;
}

// a * k for a small public constant k (the same in every lane), k < 2^32.
static inline Avx2x4 fe_mul_small(const Avx2x4& a, std::uint32_t k) noexcept {
    const __m256i kv = r26::set1(k);
    __m256i c[10];
    for (int i = 0; i < 10; ++i) c[i] = _mm256_mul_epu32(a.n[i], kv);
    r26_carry_weak(c);
    Avx2x4 r;
    for (int i = 0; i < 10; ++i) r.n[i] = c[i];
    return r;
}

// Schoolbook 10x10 into 19 columns.
static inline Avx2x4 fe_mul(const Avx2x4& a, const Avx2x4& b) noexcept {
    __m256i c[19];
    for (int k = 0; k < 19; ++k) c[k] = _mm256_setzero_si256();
    for (int i = 0; i < 10; ++i)
        for (int j = 0; j < 10; ++j)
            c[i + j] = _mm256_add_epi64(c[i + j], _mm256_mul_epu32(a.n[i], b.n[j]));
    return r26_reduce(c);
}

// Square: 45 cross products against the doubled limb (2 * n_j < 2^27 still
// fits vpmuludq's 32-bit operand) plus 10 diagonal products.
static inline Avx2x4 fe_sqr(const Avx2x4& a) noexcept {
    __m256i d[10];
    for (int j = 0; j < 10; ++j) d[j] = _mm256_add_epi64(a.n[j], a.n[j]);
    __m256i c[19];
    for (int k = 0; k < 19; ++k) c[k] = _mm256_setzero_si256();
    for (int i = 0; i < 10; ++i) {
        c[2 * i] = _mm256_add_epi64(c[2 * i], _mm256_mul_epu32(a.n[i], a.n[i]));
        for (int j = i + 1; j < 10; ++j)
            c[i + j] = _mm256_add_epi64(c[i + j], _mm256_mul_epu32(a.n[i], d[j]));
    }
    return r26_reduce(c);
}

// Canonical residue (< p), still in weak normal form. Same two steps as the
// radix-2^52 version: fold the possible 2^256 bit (t = n9 >> 22 is 0 or 1),
// then add 2^256 - p and keep that sum when it reaches 2^256.
static inline Avx2x4 fe_normalize(const Avx2x4& a) noexcept {
    const __m256i zero = _mm256_setzero_si256();
    const __m256i m22 = r26::set1(r26::kM22);
    __m256i c[10];
    for (int i = 0; i < 10; ++i) c[i] = a.n[i];
    const __m256i t = _mm256_srli_epi64(c[9], 22);
    c[9] = _mm256_and_si256(c[9], m22);
    const __m256i tmask = _mm256_sub_epi64(zero, t);   // all-ones when t == 1
    c[0] = _mm256_add_epi64(c[0], _mm256_and_si256(tmask, r26::set1(r26::kK0)));
    c[1] = _mm256_add_epi64(c[1], _mm256_and_si256(tmask, r26::set1(0x40)));
    r26::carry_pass(c);

    __m256i x[10];
    for (int i = 0; i < 10; ++i) x[i] = c[i];
    x[0] = _mm256_add_epi64(x[0], r26::set1(r26::kK0));
    x[1] = _mm256_add_epi64(x[1], r26::set1(0x40));
    r26::carry_pass(x);
    const __m256i below_p = _mm256_cmpeq_epi64(_mm256_srli_epi64(x[9], 22), zero);
    x[9] = _mm256_and_si256(x[9], m22);

    Avx2x4 r;
    for (int i = 0; i < 10; ++i) r.n[i] = _mm256_blendv_epi8(x[i], c[i], below_p);
    return r;
}

// Normalize and write four canonical elements as 4x64 limbs.
static inline void fe_store(std::uint64_t* out, const Avx2x4& a) noexcept {
    fe_store_raw(out, fe_normalize(a));
}

static inline __m256i fe_is_zero(const Avx2x4& a) noexcept {
    const Avx2x4 z = fe_normalize(a);
    __m256i acc = z.n[0];
    for (int i = 1; i < 10; ++i) acc = _mm256_or_si256(acc, z.n[i]);
    return _mm256_cmpeq_epi64(acc, _mm256_setzero_si256());
}

static inline __m256i fe_equal(const Avx2x4& a, const Avx2x4& b) noexcept {
    const Avx2x4 x = fe_normalize(a);
    const Avx2x4 y = fe_normalize(b);
    __m256i acc = _mm256_xor_si256(x.n[0], y.n[0]);
    for (int i = 1; i < 10; ++i) acc = _mm256_or_si256(acc, _mm256_xor_si256(x.n[i], y.n[i]));
    return _mm256_cmpeq_epi64(acc, _mm256_setzero_si256());
}

// Per lane: mask set -> a, clear -> b.
static inline Avx2x4 fe_select(__m256i m, const Avx2x4& a, const Avx2x4& b) noexcept {
    Avx2x4 r;
    for (int i = 0; i < 10; ++i) r.n[i] = _mm256_blendv_epi8(b.n[i], a.n[i], m);
    return r;
}

// True when every lane satisfies the weak-normal invariant
// (n0..n8 < 2^26, n9 <= 2^22); same bias trick as fe_bound_ok in field_r52.hpp.
static inline bool fe_bound_ok(const Avx2x4& a) noexcept {
    __m256i bad = _mm256_srli_epi64(a.n[0], 26);
    for (int i = 1; i < 9; ++i) bad = _mm256_or_si256(bad, _mm256_srli_epi64(a.n[i], 26));
    bad = _mm256_or_si256(bad, _mm256_srli_epi64(a.n[9], 23));
    const __m256i bias = r26::set1((std::uint64_t{1} << 63) - (std::uint64_t{1} << 22) - 1u);
    bad = _mm256_or_si256(bad, _mm256_srli_epi64(_mm256_add_epi64(a.n[9], bias), 63));
    return _mm256_testz_si256(bad, bad) != 0;
}

} // namespace secp256k1::lanes::field

#endif // SECP256K1_LANES_FIELD_AVX2X4_HPP
