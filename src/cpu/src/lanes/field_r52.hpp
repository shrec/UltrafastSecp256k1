// ============================================================================
// lanes/field_r52.hpp -- radix-2^52 lane field over vpmadd52luq / vpmadd52huq
// ============================================================================
//
// One algorithm, three ISAs. FeR52<V> is parameterised by a vector-traits type
// V that supplies the vector type, the lane count, the 52-bit multiply-add and
// the mask flavour:
//
//   VIfma512   (field_ifma512x8.hpp)    zmm, EVEX, __mmask8  -> Ifma512x8
//   VIfma256   (field_ifma256x4.hpp)    ymm, EVEX, __mmask8  -> Ifma256x4
//   VAvxIfma   (field_avx_ifma256x4.hpp) ymm, VEX, vector mask -> AvxIfma256x4
//
// Include this header only through one of those, and only from a translation
// unit compiled with that ISA's flags. All functions have internal linkage
// (static / static templates) so no ISA-specific copy can leak into a generic
// translation unit through the linker.
//
// PUBLIC DATA ONLY: variable time by design (verification paths). Not for
// private keys, nonces or any other secret.
//
// ---- Representation ---------------------------------------------------------
//
// An element is five 64-bit lanes-vectors n[0..4]; per lane the value is
//   v = n0 + n1*2^52 + n2*2^104 + n3*2^156 + n4*2^208.
//
// LIMB-BOUND INVARIANT ("weak normal form"), guaranteed on every output of
// every operation below (load, add, sub, negate, mul_small, mul, sqr,
// normalize, select, inverse):
//
//   n0..n3 <= 2^52 - 1,   n4 <= 2^48            (so v <= 2^256 + 2^208 - 1)
//
// The value need not be reduced below p; only fe_normalize / fe_store produce
// the canonical residue.
//
// Why it matters: vpmadd52{l,h}uq multiply only bits 51:0 of each operand, so
// a limb >= 2^52 would be silently truncated. Every mul/sqr input must have
// all limbs < 2^52. The invariant is enforced BY CONSTRUCTION -- every op ends
// in r52_carry_weak(), which re-establishes it -- and checked by
// fe_bound_ok(), which the test kernels (lanes_kernels.hpp) assert on every
// mul/sqr input and every op result.
//
// Bounds used inside the ops (all far below 2^64):
//   mul/sqr column sums: <= 9 terms of < 2^52, plus <= 3 fold terms  < 2^56
//   add: < 2^53;  sub/negate: a + 4p - b < 2^55;  mul_small: < 2^53
//   r52_carry_weak accepts any limbs < 2^63.
// ============================================================================

#ifndef SECP256K1_LANES_FIELD_R52_HPP
#define SECP256K1_LANES_FIELD_R52_HPP

#include <cstddef>
#include <cstdint>

#include "field_common.hpp"

namespace secp256k1::lanes::field {

template <class V>
struct FeR52 {
    using Traits = V;
    using Vec = typename V::Vec;
    using Mask = typename V::Mask;
    static constexpr unsigned kLanes = V::kLanes;
    static constexpr unsigned kLimbBits = 52;
    static constexpr unsigned kLimbCount = 5;
    static constexpr std::uint64_t kTopLimbMax = std::uint64_t{1} << 48;  // inclusive

    Vec n[5];

    // Lane mask <-> bit i set for lane i.
    static unsigned mask_bits(Mask m) noexcept { return V::mask_bits(m); }
    static Mask mask_from_bits(unsigned bits) noexcept { return V::mask_from_bits(bits); }
};

namespace r52 {
constexpr std::uint64_t kM52 = 0xFFFFFFFFFFFFFULL;
constexpr std::uint64_t kM48 = 0xFFFFFFFFFFFFULL;
constexpr std::uint64_t kK256 = 0x1000003D1ULL;    // 2^256 mod p (33 bits)
constexpr std::uint64_t kR260 = 0x1000003D10ULL;   // 2^260 mod p (37 bits, < 2^52)
// 4p in radix 2^52. Every weak-normal limb is below the matching 4p limb
// (n0..n3 < 2^52 < 4p_i, n4 <= 2^48 < 4p_4 = 2^50 - 4), so a + 4p - b never
// underflows a limb.
constexpr std::uint64_t kP4[5] = {
    0x3FFFFBFFFFF0BCULL,   // 4 * 0xFFFFEFFFFFC2F
    0x3FFFFFFFFFFFFCULL,   // 4 * 0xFFFFFFFFFFFFF
    0x3FFFFFFFFFFFFCULL,
    0x3FFFFFFFFFFFFCULL,
    0x3FFFFFFFFFFFCULL,    // 4 * 0xFFFFFFFFFFFF
};
} // namespace r52

// Carry five limbs (each < 2^63) back to weak normal form.
// Two passes: the first leaves n0..n3 < 2^52 and folds everything at or above
// 2^256 (t = n4 >> 48, t < 2^15) back in as t * (2^256 mod p) < 2^48; the
// second pass then moves at most a carry of 1 through each limb, so n4 ends
// <= 2^48 - 1 + 1 = 2^48.
template <class V>
static inline void r52_carry_weak(typename V::Vec* c) noexcept {
    using Vec = typename V::Vec;
    const Vec m52 = V::set1(r52::kM52);
    c[1] = V::add(c[1], V::template srli<52>(c[0])); c[0] = V::and_(c[0], m52);
    c[2] = V::add(c[2], V::template srli<52>(c[1])); c[1] = V::and_(c[1], m52);
    c[3] = V::add(c[3], V::template srli<52>(c[2])); c[2] = V::and_(c[2], m52);
    c[4] = V::add(c[4], V::template srli<52>(c[3])); c[3] = V::and_(c[3], m52);
    const Vec t = V::template srli<48>(c[4]);
    c[4] = V::and_(c[4], V::set1(r52::kM48));
    c[0] = V::madd52lo(c[0], t, V::set1(r52::kK256));   // t * K < 2^48: the low half is exact
    c[1] = V::add(c[1], V::template srli<52>(c[0])); c[0] = V::and_(c[0], m52);
    c[2] = V::add(c[2], V::template srli<52>(c[1])); c[1] = V::and_(c[1], m52);
    c[3] = V::add(c[3], V::template srli<52>(c[2])); c[2] = V::and_(c[2], m52);
    c[4] = V::add(c[4], V::template srli<52>(c[3])); c[3] = V::and_(c[3], m52);
}

// Reduce ten product columns c[0..9] (column k has weight 2^(52k), each < 2^57)
// to weak normal form.
//   1. Carry the high columns 5..8 into exact 52-bit limbs so they are valid
//      vpmadd52 operands. Column 9 needs no carry out: it is hi(a4 * b4) plus
//      a carry < 2^5, and the invariant n4 <= 2^48 makes a4 * b4 <= 2^96, so
//      column 9 < 2^45 and there is no column 10.
//   2. Fold column k >= 5 down by 2^260 == R (mod p), R = 0x1000003D10 < 2^52:
//      lo(c_k * R) goes to column k-5, hi(c_k * R) to column k-4. The hi part
//      of column 9 lands on column 5 (< 2^31), which is folded once more the
//      same way.
//   3. r52_carry_weak on the low five columns.
template <class V>
static inline FeR52<V> r52_reduce(typename V::Vec* c) noexcept {
    using Vec = typename V::Vec;
    const Vec m52 = V::set1(r52::kM52);
    const Vec R = V::set1(r52::kR260);
    c[6] = V::add(c[6], V::template srli<52>(c[5])); c[5] = V::and_(c[5], m52);
    c[7] = V::add(c[7], V::template srli<52>(c[6])); c[6] = V::and_(c[6], m52);
    c[8] = V::add(c[8], V::template srli<52>(c[7])); c[7] = V::and_(c[7], m52);
    c[9] = V::add(c[9], V::template srli<52>(c[8])); c[8] = V::and_(c[8], m52);

    c[0] = V::madd52lo(c[0], c[5], R); c[1] = V::madd52hi(c[1], c[5], R);
    c[1] = V::madd52lo(c[1], c[6], R); c[2] = V::madd52hi(c[2], c[6], R);
    c[2] = V::madd52lo(c[2], c[7], R); c[3] = V::madd52hi(c[3], c[7], R);
    c[3] = V::madd52lo(c[3], c[8], R); c[4] = V::madd52hi(c[4], c[8], R);
    c[4] = V::madd52lo(c[4], c[9], R);
    const Vec x5 = V::madd52hi(V::zero(), c[9], R);   // c9 * R < 2^82 -> hi < 2^30
    c[0] = V::madd52lo(c[0], x5, R); c[1] = V::madd52hi(c[1], x5, R);

    r52_carry_weak<V>(c);
    FeR52<V> r;
    for (int i = 0; i < 5; ++i) r.n[i] = c[i];
    return r;
}

// ---- Load / store -----------------------------------------------------------

// Load kLanes elements, four uint64 limbs each (fast::FieldElement layout,
// limb 0 least significant), from in[4*lane + i]. Any 256-bit value is
// accepted; the result is in weak normal form (n4 <= 2^48 - 1).
template <class V>
static inline void fe_load(FeR52<V>& r, const std::uint64_t* in) noexcept {
    alignas(64) std::uint64_t t[5][V::kLanes];
    for (unsigned l = 0; l < V::kLanes; ++l) {
        std::uint64_t limbs[5];
        lanes_split_limbs(in + 4u * l, limbs, 52u, 5u);
        for (int i = 0; i < 5; ++i) t[i][l] = limbs[i];
    }
    for (int i = 0; i < 5; ++i) r.n[i] = V::loadu(t[i]);
}

// Load internal-form limbs, five per element (in[5*lane + i]). The caller
// guarantees weak normal form.
template <class V>
static inline void fe_load_limbs(FeR52<V>& r, const std::uint64_t* in) noexcept {
    alignas(64) std::uint64_t t[5][V::kLanes];
    for (unsigned l = 0; l < V::kLanes; ++l)
        for (int i = 0; i < 5; ++i) t[i][l] = in[5u * l + static_cast<unsigned>(i)];
    for (int i = 0; i < 5; ++i) r.n[i] = V::loadu(t[i]);
}

// Write the limbs as they are (no normalization) as 4x64 per lane. Exact only
// when every limb is < 2^52 and the value is < 2^256 (e.g. after fe_normalize).
template <class V>
static inline void fe_store_raw(std::uint64_t* out, const FeR52<V>& a) noexcept {
    alignas(64) std::uint64_t t[5][V::kLanes];
    for (int i = 0; i < 5; ++i) V::storeu(t[i], a.n[i]);
    for (unsigned l = 0; l < V::kLanes; ++l) {
        std::uint64_t limbs[5];
        for (int i = 0; i < 5; ++i) limbs[i] = t[i][l];
        lanes_join_limbs(limbs, out + 4u * l, 52u, 5u);
    }
}

// ---- Arithmetic --------------------------------------------------------------

template <class V>
static inline FeR52<V> fe_add(const FeR52<V>& a, const FeR52<V>& b) noexcept {
    typename V::Vec c[5];
    for (int i = 0; i < 5; ++i) c[i] = V::add(a.n[i], b.n[i]);
    r52_carry_weak<V>(c);
    FeR52<V> r;
    for (int i = 0; i < 5; ++i) r.n[i] = c[i];
    return r;
}

// a - b computed as a + 4p - b (no limb can go negative, see kP4).
template <class V>
static inline FeR52<V> fe_sub(const FeR52<V>& a, const FeR52<V>& b) noexcept {
    typename V::Vec c[5];
    for (int i = 0; i < 5; ++i)
        c[i] = V::sub(V::add(a.n[i], V::set1(r52::kP4[i])), b.n[i]);
    r52_carry_weak<V>(c);
    FeR52<V> r;
    for (int i = 0; i < 5; ++i) r.n[i] = c[i];
    return r;
}

template <class V>
static inline FeR52<V> fe_negate(const FeR52<V>& a) noexcept {
    typename V::Vec c[5];
    for (int i = 0; i < 5; ++i) c[i] = V::sub(V::set1(r52::kP4[i]), a.n[i]);
    r52_carry_weak<V>(c);
    FeR52<V> r;
    for (int i = 0; i < 5; ++i) r.n[i] = c[i];
    return r;
}

// a * k for a small public constant k (the same in every lane).
template <class V>
static inline FeR52<V> fe_mul_small(const FeR52<V>& a, std::uint32_t k) noexcept {
    using Vec = typename V::Vec;
    const Vec kv = V::set1(k);
    Vec c[5];
    for (int i = 0; i < 5; ++i) c[i] = V::madd52lo(V::zero(), a.n[i], kv);
    for (int i = 0; i < 4; ++i) c[i + 1] = V::madd52hi(c[i + 1], a.n[i], kv);
    const Vec x5 = V::madd52hi(V::zero(), a.n[4], kv);   // n4 * k < 2^80 -> hi < 2^28
    const Vec R = V::set1(r52::kR260);
    c[0] = V::madd52lo(c[0], x5, R);
    c[1] = V::madd52hi(c[1], x5, R);
    r52_carry_weak<V>(c);
    FeR52<V> r;
    for (int i = 0; i < 5; ++i) r.n[i] = c[i];
    return r;
}

// Schoolbook 5x5: lo(a_i*b_j) into column i+j, hi(a_i*b_j) into column i+j+1.
// The inner loop walks distinct accumulators, so consecutive multiply-adds do
// not wait on each other.
template <class V>
static inline FeR52<V> fe_mul(const FeR52<V>& a, const FeR52<V>& b) noexcept {
    typename V::Vec c[10];
    for (int k = 0; k < 10; ++k) c[k] = V::zero();
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            c[i + j] = V::madd52lo(c[i + j], a.n[i], b.n[j]);
            c[i + j + 1] = V::madd52hi(c[i + j + 1], a.n[i], b.n[j]);
        }
    }
    return r52_reduce<V>(c);
}

// Square: the ten cross products are accumulated once and doubled (doubling
// an operand instead could push it past 52 bits), then the five diagonal
// products are added.
template <class V>
static inline FeR52<V> fe_sqr(const FeR52<V>& a) noexcept {
    typename V::Vec c[10];
    for (int k = 0; k < 10; ++k) c[k] = V::zero();
    for (int i = 0; i < 5; ++i) {
        for (int j = i + 1; j < 5; ++j) {
            c[i + j] = V::madd52lo(c[i + j], a.n[i], a.n[j]);
            c[i + j + 1] = V::madd52hi(c[i + j + 1], a.n[i], a.n[j]);
        }
    }
    for (int k = 0; k < 10; ++k) c[k] = V::add(c[k], c[k]);
    for (int i = 0; i < 5; ++i) {
        c[2 * i] = V::madd52lo(c[2 * i], a.n[i], a.n[i]);
        c[2 * i + 1] = V::madd52hi(c[2 * i + 1], a.n[i], a.n[i]);
    }
    return r52_reduce<V>(c);
}

// Canonical residue (< p), still in weak normal form.
//   1. Fold the possible 2^256 bit (n4 <= 2^48 means t = n4 >> 48 is 0 or 1)
//      and carry once: the value is now < 2^256.
//   2. x = v + (2^256 - p); if that reaches 2^256 then v >= p and the answer
//      is x - 2^256, otherwise v.
template <class V>
static inline FeR52<V> fe_normalize(const FeR52<V>& a) noexcept {
    using Vec = typename V::Vec;
    const Vec m52 = V::set1(r52::kM52);
    const Vec m48 = V::set1(r52::kM48);
    const Vec K = V::set1(r52::kK256);
    const Vec zero = V::zero();
    Vec c[5];
    for (int i = 0; i < 5; ++i) c[i] = a.n[i];
    const Vec t = V::template srli<48>(c[4]);
    c[4] = V::and_(c[4], m48);
    c[0] = V::add(c[0], V::and_(V::sub(zero, t), K));   // t is 0 or 1
    c[1] = V::add(c[1], V::template srli<52>(c[0])); c[0] = V::and_(c[0], m52);
    c[2] = V::add(c[2], V::template srli<52>(c[1])); c[1] = V::and_(c[1], m52);
    c[3] = V::add(c[3], V::template srli<52>(c[2])); c[2] = V::and_(c[2], m52);
    c[4] = V::add(c[4], V::template srli<52>(c[3])); c[3] = V::and_(c[3], m52);

    Vec x[5];
    x[0] = V::add(c[0], K);
    x[1] = V::add(c[1], V::template srli<52>(x[0])); x[0] = V::and_(x[0], m52);
    x[2] = V::add(c[2], V::template srli<52>(x[1])); x[1] = V::and_(x[1], m52);
    x[3] = V::add(c[3], V::template srli<52>(x[2])); x[2] = V::and_(x[2], m52);
    x[4] = V::add(c[4], V::template srli<52>(x[3])); x[3] = V::and_(x[3], m52);
    const typename V::Mask below_p = V::eq_mask(V::template srli<48>(x[4]), zero);
    x[4] = V::and_(x[4], m48);

    FeR52<V> r;
    for (int i = 0; i < 5; ++i) r.n[i] = V::blend(below_p, c[i], x[i]);
    return r;
}

// Normalize and write kLanes canonical elements as 4x64 limbs.
template <class V>
static inline void fe_store(std::uint64_t* out, const FeR52<V>& a) noexcept {
    fe_store_raw(out, fe_normalize(a));
}

template <class V>
static inline typename V::Mask fe_is_zero(const FeR52<V>& a) noexcept {
    const FeR52<V> z = fe_normalize(a);
    typename V::Vec acc = z.n[0];
    for (int i = 1; i < 5; ++i) acc = V::or_(acc, z.n[i]);
    return V::eq_mask(acc, V::zero());
}

template <class V>
static inline typename V::Mask fe_equal(const FeR52<V>& a, const FeR52<V>& b) noexcept {
    const FeR52<V> x = fe_normalize(a);
    const FeR52<V> y = fe_normalize(b);
    typename V::Vec acc = V::xor_(x.n[0], y.n[0]);
    for (int i = 1; i < 5; ++i) acc = V::or_(acc, V::xor_(x.n[i], y.n[i]));
    return V::eq_mask(acc, V::zero());
}

// Per lane: mask set -> a, clear -> b.
template <class V>
static inline FeR52<V> fe_select(typename V::Mask m, const FeR52<V>& a,
                                 const FeR52<V>& b) noexcept {
    FeR52<V> r;
    for (int i = 0; i < 5; ++i) r.n[i] = V::blend(m, a.n[i], b.n[i]);
    return r;
}

// True when every lane satisfies the weak-normal invariant
// (n0..n3 < 2^52, n4 <= 2^48). n4 > 2^48 is detected without an unsigned
// compare: for n4 < 2^49, n4 + (2^63 - 2^48 - 1) reaches bit 63 exactly when
// n4 > 2^48; n4 >= 2^49 is caught by the shift.
template <class V>
static inline bool fe_bound_ok(const FeR52<V>& a) noexcept {
    using Vec = typename V::Vec;
    Vec bad = V::template srli<52>(a.n[0]);
    for (int i = 1; i < 4; ++i) bad = V::or_(bad, V::template srli<52>(a.n[i]));
    bad = V::or_(bad, V::template srli<49>(a.n[4]));
    const Vec bias = V::set1((std::uint64_t{1} << 63) - (std::uint64_t{1} << 48) - 1u);
    bad = V::or_(bad, V::template srli<63>(V::add(a.n[4], bias)));
    return V::all_zero(bad);
}

} // namespace secp256k1::lanes::field

#endif // SECP256K1_LANES_FIELD_R52_HPP
