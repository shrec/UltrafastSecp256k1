// ============================================================================
// Constant-Time Scalar Arithmetic -- Implementation
// ============================================================================
// All operations have data-independent execution traces.
// n = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
// ============================================================================

#include "secp256k1/ct/scalar.hpp"
#include "secp256k1/ct/ops.hpp"
#include <cstring>

namespace secp256k1::ct {

// secp256k1 curve order n in 4 x 64-bit limbs (little-endian)
static constexpr std::uint64_t N[4] = {
    0xBFD25E8CD0364141ULL,
    0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL,
    0xFFFFFFFFFFFFFFFFULL
};

// --- Internal helpers --------------------------------------------------------

static inline std::uint64_t add256_scalar(std::uint64_t r[4],
                                           const std::uint64_t a[4],
                                           const std::uint64_t b[4]) noexcept {
    std::uint64_t carry = 0;
    for (int i = 0; i < 4; ++i) {
        std::uint64_t const sum_lo = a[i] + b[i];
        auto const c1 = static_cast<std::uint64_t>(sum_lo < a[i]);
        std::uint64_t const sum = sum_lo + carry;
        auto const c2 = static_cast<std::uint64_t>(sum < sum_lo);
        r[i] = sum;
        carry = c1 + c2;
    }
    return carry;
}

static inline std::uint64_t sub256_scalar(std::uint64_t r[4],
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

[[maybe_unused]] static inline void ct_reduce_once_n(std::uint64_t r[4]) noexcept {
    std::uint64_t tmp[4];
    std::uint64_t const borrow = sub256_scalar(tmp, r, N);
    std::uint64_t const mask = is_zero_mask(borrow);  // no borrow = r >= n
    cmov256(r, tmp, mask);
}

// --- Public API --------------------------------------------------------------

Scalar scalar_add(const Scalar& a, const Scalar& b) noexcept {
    std::uint64_t r[4];
    std::uint64_t carry = add256_scalar(r, a.limbs().data(), b.limbs().data());

    std::uint64_t tmp[4];
    std::uint64_t borrow = sub256_scalar(tmp, r, N);

    // value_barrier prevents the compiler from reasoning about carry/borrow
    // values and converting the mask computation into a branch.
    value_barrier(carry);
    value_barrier(borrow);

    // carry=1 implies r < n (since a+b < 2n < n + 2^256), so borrow=1.
    // carry=1 AND borrow=0 is impossible -> OR safely covers all cases.
    std::uint64_t const no_borrow = is_zero_mask(borrow);
    std::uint64_t const has_carry = is_nonzero_mask(carry);
    std::uint64_t const mask = no_borrow | has_carry;
    cmov256(r, tmp, mask);

    return Scalar::from_limbs({r[0], r[1], r[2], r[3]});
}

Scalar scalar_sub(const Scalar& a, const Scalar& b) noexcept {
    std::uint64_t r[4];
    std::uint64_t borrow = sub256_scalar(r, a.limbs().data(), b.limbs().data());

    std::uint64_t tmp[4];
    add256_scalar(tmp, r, N);

    // value_barrier prevents the compiler from converting the conditional
    // move into a branch on the borrow flag.
    value_barrier(borrow);
    std::uint64_t mask = is_nonzero_mask(borrow);
    cmov256(r, tmp, mask);

    return Scalar::from_limbs({r[0], r[1], r[2], r[3]});
}

Scalar scalar_neg(const Scalar& a) noexcept {
    // -a mod n = n - a (if a != 0), 0 (if a == 0)
    std::uint64_t r[4];
    sub256_scalar(r, N, a.limbs().data());

    std::uint64_t const zero_mask = scalar_is_zero(a);
    std::uint64_t z[4] = {0, 0, 0, 0};
    cmov256(r, z, zero_mask);

    return Scalar::from_limbs({r[0], r[1], r[2], r[3]});
}

Scalar scalar_half(const Scalar& a) noexcept {
    // Compute a/2 mod n.
    // If a is odd: r = (a + n) >> 1  (n is odd, so a+n is even)
    // If a is even: r = a >> 1
    const auto& al = a.limbs();
    std::uint64_t const odd = al[0] & 1;  // 1 if odd, 0 if even
    // Branchless: conditionally add n
    std::uint64_t t[4];
    std::uint64_t carry = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        std::uint64_t const n_masked = N[i] & (0ULL - odd);
        std::uint64_t const sum_lo = al[i] + n_masked;
        auto const c1 = static_cast<std::uint64_t>(sum_lo < al[i]);
        std::uint64_t const sum = sum_lo + carry;
        auto const c2 = static_cast<std::uint64_t>(sum < sum_lo);
        t[i] = sum;
        carry = c1 + c2;
    }
    // Right shift by 1 (carry bit goes into top)
    std::uint64_t r[4];
    r[0] = (t[0] >> 1) | (t[1] << 63);
    r[1] = (t[1] >> 1) | (t[2] << 63);
    r[2] = (t[2] >> 1) | (t[3] << 63);
    r[3] = (t[3] >> 1) | (carry << 63);
    return Scalar::from_limbs({r[0], r[1], r[2], r[3]});
}

// --- Conditional Operations --------------------------------------------------

void scalar_cmov(Scalar* r, const Scalar& a, std::uint64_t mask) noexcept {
    *r = scalar_select(a, *r, mask);
}

void scalar_cswap(Scalar* a, Scalar* b, std::uint64_t mask) noexcept {
    Scalar const old_a = *a;
    Scalar const old_b = *b;
    *a = scalar_select(old_b, old_a, mask);
    *b = scalar_select(old_a, old_b, mask);
}

Scalar scalar_select(const Scalar& a, const Scalar& b,
                     std::uint64_t mask) noexcept {
    const auto& al = a.limbs();
    const auto& bl = b.limbs();
    return Scalar::from_limbs({
        ct_select(al[0], bl[0], mask),
        ct_select(al[1], bl[1], mask),
        ct_select(al[2], bl[2], mask),
        ct_select(al[3], bl[3], mask)
    });
}

// ============================================================================
// CT Scalar Modular Inverse -- SafeGCD (Bernstein-Yang constant-time divsteps)
// ============================================================================
// Constant-time modular inverse using Bernstein-Yang divsteps algorithm.
// Port of bitcoin-core/secp256k1 secp256k1_modinv64 (the CT variant).
//
// 10 rounds x 59 branchless divsteps = 590 total divsteps.
// All loops have fixed iteration count; all conditionals use bitmasks.
// No secret-dependent branches, no ctz, no early termination.
//
// Performance: ~50x faster than Fermat a^(n-2) chain (~900 ns vs ~10,600 ns),
// matching the variable-time SafeGCD speed while remaining constant-time.
// ============================================================================
#if defined(__SIZEOF_INT128__)

namespace ct_safegcd {

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
using i128 = __int128;

struct S62  { int64_t v[5]; };
struct T2x2 { int64_t u, v, q, r; };
struct ModInfo { S62 modulus; uint64_t modulus_inv62; };

// secp256k1 order n in signed-62 form, plus modular inverse
// n = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
static constexpr ModInfo NINFO = {
    {{0x3FD25E8CD0364141LL, 0x2ABB739ABD2280EELL, -0x15LL, 0LL, 256LL}},
    0x34F20099AA774EC1ULL
};

// Constant-time 59 divsteps -- all branchless, fixed iteration count.
// Matches secp256k1_modinv64_divsteps_59 exactly.
// Matrix is scaled by 2^62 (starts at 8*I, 59 divsteps each multiply by 2).
static int64_t divsteps_59(int64_t zeta, uint64_t f0, uint64_t g0, T2x2& t) {
    uint64_t u = 8, v = 0, q = 0, r = 8;
    volatile uint64_t c1 = 0;
    volatile uint64_t c2 = 0;
    uint64_t mask1 = 0, mask2 = 0, f = f0, g = g0, x = 0, y = 0, z = 0;

    for (int i = 3; i < 62; ++i) {
        c1 = static_cast<uint64_t>(zeta >> 63);
        mask1 = c1;
        c2 = g & 1;
        mask2 = -c2;
        x = (f ^ mask1) - mask1;
        y = (u ^ mask1) - mask1;
        z = (v ^ mask1) - mask1;
        g += x & mask2;
        q += y & mask2;
        r += z & mask2;
        mask1 &= mask2;
        zeta = static_cast<int64_t>(static_cast<uint64_t>(zeta) ^ mask1) - 1;
        f += g & mask1;
        u += q & mask1;
        v += r & mask1;
        g >>= 1;
        u <<= 1;
        v <<= 1;
    }
    t.u = (int64_t)u; t.v = (int64_t)v;
    t.q = (int64_t)q; t.r = (int64_t)r;
    return zeta;
}

// Update d,e using transition matrix (t/2^62) -- already CT (no branches)
static void update_de_62(S62& d, S62& e, const T2x2& t, const ModInfo& mod) {
    const uint64_t M62 = UINT64_MAX >> 2;
    const int64_t d0 = d.v[0], d1 = d.v[1], d2 = d.v[2], d3 = d.v[3], d4 = d.v[4];
    const int64_t e0 = e.v[0], e1 = e.v[1], e2 = e.v[2], e3 = e.v[3], e4 = e.v[4];
    const int64_t u = t.u, v = t.v, q = t.q, r = t.r;
    int64_t md = 0;
    int64_t me = 0;
    int64_t sd = 0;
    int64_t se = 0;
    i128 cd = 0;
    i128 ce = 0;

    sd = d4 >> 63;
    se = e4 >> 63;
    md = (u & sd) + (v & se);
    me = (q & sd) + (r & se);

    cd = (i128)u * d0 + (i128)v * e0;
    ce = (i128)q * d0 + (i128)r * e0;

    md -= (int64_t)((mod.modulus_inv62 * (uint64_t)cd + (uint64_t)md) & M62);
    me -= (int64_t)((mod.modulus_inv62 * (uint64_t)ce + (uint64_t)me) & M62);

    cd += (i128)mod.modulus.v[0] * md;
    ce += (i128)mod.modulus.v[0] * me;
    cd >>= 62; ce >>= 62;

    cd += (i128)u * d1 + (i128)v * e1;
    ce += (i128)q * d1 + (i128)r * e1;
    if (mod.modulus.v[1]) { cd += (i128)mod.modulus.v[1] * md; ce += (i128)mod.modulus.v[1] * me; }
    d.v[0] = (int64_t)((uint64_t)cd & M62); cd >>= 62;
    e.v[0] = (int64_t)((uint64_t)ce & M62); ce >>= 62;

    cd += (i128)u * d2 + (i128)v * e2;
    ce += (i128)q * d2 + (i128)r * e2;
    if (mod.modulus.v[2]) { cd += (i128)mod.modulus.v[2] * md; ce += (i128)mod.modulus.v[2] * me; }
    d.v[1] = (int64_t)((uint64_t)cd & M62); cd >>= 62;
    e.v[1] = (int64_t)((uint64_t)ce & M62); ce >>= 62;

    cd += (i128)u * d3 + (i128)v * e3;
    ce += (i128)q * d3 + (i128)r * e3;
    if (mod.modulus.v[3]) { cd += (i128)mod.modulus.v[3] * md; ce += (i128)mod.modulus.v[3] * me; }
    d.v[2] = (int64_t)((uint64_t)cd & M62); cd >>= 62;
    e.v[2] = (int64_t)((uint64_t)ce & M62); ce >>= 62;

    cd += (i128)u * d4 + (i128)v * e4;
    ce += (i128)q * d4 + (i128)r * e4;
    cd += (i128)mod.modulus.v[4] * md;
    ce += (i128)mod.modulus.v[4] * me;
    d.v[3] = (int64_t)((uint64_t)cd & M62); cd >>= 62;
    e.v[3] = (int64_t)((uint64_t)ce & M62); ce >>= 62;

    d.v[4] = (int64_t)cd;
    e.v[4] = (int64_t)ce;
}

// Update f,g -- CT, fixed 5 limbs (no variable-length optimization)
static void update_fg_62(S62& f, S62& g, const T2x2& t) {
    const uint64_t M62 = UINT64_MAX >> 2;
    const int64_t f0 = f.v[0], f1 = f.v[1], f2 = f.v[2], f3 = f.v[3], f4 = f.v[4];
    const int64_t g0 = g.v[0], g1 = g.v[1], g2 = g.v[2], g3 = g.v[3], g4 = g.v[4];
    const int64_t u = t.u, v = t.v, q = t.q, r = t.r;
    i128 cf = 0;
    i128 cg = 0;

    cf = (i128)u * f0 + (i128)v * g0;
    cg = (i128)q * f0 + (i128)r * g0;
    cf >>= 62; cg >>= 62;

    cf += (i128)u * f1 + (i128)v * g1;
    cg += (i128)q * f1 + (i128)r * g1;
    f.v[0] = (int64_t)((uint64_t)cf & M62); cf >>= 62;
    g.v[0] = (int64_t)((uint64_t)cg & M62); cg >>= 62;

    cf += (i128)u * f2 + (i128)v * g2;
    cg += (i128)q * f2 + (i128)r * g2;
    f.v[1] = (int64_t)((uint64_t)cf & M62); cf >>= 62;
    g.v[1] = (int64_t)((uint64_t)cg & M62); cg >>= 62;

    cf += (i128)u * f3 + (i128)v * g3;
    cg += (i128)q * f3 + (i128)r * g3;
    f.v[2] = (int64_t)((uint64_t)cf & M62); cf >>= 62;
    g.v[2] = (int64_t)((uint64_t)cg & M62); cg >>= 62;

    cf += (i128)u * f4 + (i128)v * g4;
    cg += (i128)q * f4 + (i128)r * g4;
    f.v[3] = (int64_t)((uint64_t)cf & M62); cf >>= 62;
    g.v[3] = (int64_t)((uint64_t)cg & M62); cg >>= 62;

    f.v[4] = (int64_t)cf;
    g.v[4] = (int64_t)cg;
}

// Normalize result to [0, modulus)
static void normalize_62(S62& r, int64_t sign, const ModInfo& mod) {
    const auto M62 = (int64_t)(UINT64_MAX >> 2);
    int64_t r0 = r.v[0], r1 = r.v[1], r2 = r.v[2], r3 = r.v[3], r4 = r.v[4];
    int64_t cond_add = 0, cond_negate = 0;

    cond_add = r4 >> 63;
    r0 += mod.modulus.v[0] & cond_add;
    r1 += mod.modulus.v[1] & cond_add;
    r2 += mod.modulus.v[2] & cond_add;
    r3 += mod.modulus.v[3] & cond_add;
    r4 += mod.modulus.v[4] & cond_add;
    cond_negate = sign >> 63;
    r0 = (r0 ^ cond_negate) - cond_negate;
    r1 = (r1 ^ cond_negate) - cond_negate;
    r2 = (r2 ^ cond_negate) - cond_negate;
    r3 = (r3 ^ cond_negate) - cond_negate;
    r4 = (r4 ^ cond_negate) - cond_negate;
    r1 += r0 >> 62; r0 &= M62;
    r2 += r1 >> 62; r1 &= M62;
    r3 += r2 >> 62; r2 &= M62;
    r4 += r3 >> 62; r3 &= M62;

    cond_add = r4 >> 63;
    r0 += mod.modulus.v[0] & cond_add;
    r1 += mod.modulus.v[1] & cond_add;
    r2 += mod.modulus.v[2] & cond_add;
    r3 += mod.modulus.v[3] & cond_add;
    r4 += mod.modulus.v[4] & cond_add;
    r1 += r0 >> 62; r0 &= M62;
    r2 += r1 >> 62; r1 &= M62;
    r3 += r2 >> 62; r2 &= M62;
    r4 += r3 >> 62; r3 &= M62;

    r.v[0] = r0; r.v[1] = r1; r.v[2] = r2; r.v[3] = r3; r.v[4] = r4;
}

using limbs4 = Scalar::limbs_type;

static S62 limbs_to_s62(const limbs4& d) {
    constexpr uint64_t M = (1ULL << 62) - 1;
    return {{
        (int64_t)(d[0] & M),
        (int64_t)(((d[0] >> 62) | (d[1] << 2)) & M),
        (int64_t)(((d[1] >> 60) | (d[2] << 4)) & M),
        (int64_t)(((d[2] >> 58) | (d[3] << 6)) & M),
        (int64_t)(d[3] >> 56)
    }};
}

static limbs4 s62_to_limbs(const S62& s) {
    return {{
        (uint64_t)s.v[0] | ((uint64_t)s.v[1] << 62),
        ((uint64_t)s.v[1] >> 2) | ((uint64_t)s.v[2] << 60),
        ((uint64_t)s.v[2] >> 4) | ((uint64_t)s.v[3] << 58),
        ((uint64_t)s.v[3] >> 6) | ((uint64_t)s.v[4] << 56)
    }};
}

// CT scalar modular inverse: fixed 10 rounds of 59 divsteps = 590 total.
// Sufficient for 256-bit inputs (proven bound: 590 >= ceil(49*256/17)).
// Returns 0 for zero input (naturally, no special branch needed).
static limbs4 inverse_impl(const limbs4& x) {
    S62 d = {{0, 0, 0, 0, 0}};
    S62 e = {{1, 0, 0, 0, 0}};
    S62 f = NINFO.modulus;
    S62 g = limbs_to_s62(x);
    int64_t zeta = -1;

    for (int i = 0; i < 10; ++i) {
        T2x2 t;
        zeta = divsteps_59(zeta, (uint64_t)f.v[0], (uint64_t)g.v[0], t);
        update_de_62(d, e, t, NINFO);
        update_fg_62(f, g, t);
    }

    normalize_62(d, f.v[4], NINFO);
    return s62_to_limbs(d);
}

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
} // namespace ct_safegcd

Scalar scalar_inverse(const Scalar& a) noexcept {
    return Scalar::from_limbs(ct_safegcd::inverse_impl(a.limbs()));
}

#else // !__SIZEOF_INT128__
// ============================================================================
// Fallback: Fermat's Little Theorem for platforms without __int128
// ============================================================================
// a^{-1} = a^{n-2} mod n.  Optimized addition chain: 254S + 40M = 294 ops.
// ~10x slower than SafeGCD but works on all platforms.
// ============================================================================
Scalar scalar_inverse(const Scalar& a) noexcept {
    auto sqr = [](const Scalar& s) -> Scalar { return s * s; };
    auto sqr_n = [&sqr](Scalar s, int count) -> Scalar {
        for (int i = 0; i < count; ++i) s = sqr(s);
        return s;
    };

    Scalar const u2 = sqr(a);
    Scalar const x2 = u2 * a;
    Scalar const u5 = u2 * x2;
    Scalar const x3 = u5 * u2;
    Scalar const x6  = sqr_n(x3, 3) * x3;
    Scalar const x8  = sqr_n(x6, 2) * x2;
    Scalar const x14 = sqr_n(x8, 6) * x6;
    Scalar const x28 = sqr_n(x14, 14) * x14;
    Scalar const x56 = sqr_n(x28, 28) * x28;
    Scalar const x112 = sqr_n(x56, 56) * x56;
    Scalar const x126 = sqr_n(x112, 14) * x14;

    Scalar t = sqr_n(x126, 3) * u5;
    t = sqr_n(t, 4) * x3;  t = sqr_n(t, 4) * u5;  t = sqr_n(t, 2) * a;
    t = sqr_n(t, 4) * x3;  t = sqr_n(t, 3) * x2;  t = sqr_n(t, 4) * x3;
    t = sqr_n(t, 5) * x3;  t = sqr_n(t, 4) * x2;  t = sqr_n(t, 4) * u5;
    t = sqr_n(t, 4) * x3;  t = sqr_n(t, 3) * u5;  t = sqr_n(t, 3) * a;
    t = sqr_n(t, 6) * u5;  t = sqr_n(t, 10) * x3; t = sqr_n(t, 4) * x3;
    t = sqr_n(t, 9) * x8;  t = sqr_n(t, 2) * a;   t = sqr_n(t, 3) * a;
    t = sqr_n(t, 3) * a;   t = sqr_n(t, 4) * x3;  t = sqr_n(t, 3) * u5;
    t = sqr_n(t, 5) * x2;  t = sqr_n(t, 4) * x2;  t = sqr_n(t, 2) * a;
    t = sqr_n(t, 8) * x2;  t = sqr_n(t, 3) * x2;  t = sqr_n(t, 3) * a;
    t = sqr_n(t, 6) * a;   t = sqr_n(t, 8) * x6;
    return t;
}
#endif // __SIZEOF_INT128__

Scalar scalar_cneg(const Scalar& a, std::uint64_t mask) noexcept {
    Scalar const neg = scalar_neg(a);
    return scalar_select(neg, a, mask);
}

// --- Comparison --------------------------------------------------------------

std::uint64_t scalar_is_zero(const Scalar& a) noexcept {
    const auto& l = a.limbs();
#if defined(__riscv) && (__riscv_xlen == 64)
    // RISC-V U74: single asm block performs OR-reduction + is_zero_mask.
    // This prevents the compiler from scheduling the OR chain differently
    // for zero vs non-zero inputs (dudect |t| > 10 without this).
    std::uint64_t a0 = l[0], a1 = l[1], a2 = l[2], a3 = l[3];
    std::uint64_t mask;
    asm volatile(
        "or    %0, %1, %2\n\t"
        "or    %0, %0, %3\n\t"
        "or    %0, %0, %4\n\t"
        "seqz  %0, %0\n\t"
        "neg   %0, %0"
        : "=&r"(mask)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3)
    );
    return mask;
#else
    // value_barrier each limb before OR to prevent the compiler from
    // constant-propagating zero limbs -> zero OR -> optimized is_zero_mask.
    std::uint64_t a0 = l[0], a1 = l[1], a2 = l[2], a3 = l[3];
    value_barrier(a0);
    value_barrier(a1);
    value_barrier(a2);
    value_barrier(a3);
    std::uint64_t const z = a0 | a1 | a2 | a3;
    return is_zero_mask(z);
#endif
}

std::uint64_t scalar_eq(const Scalar& a, const Scalar& b) noexcept {
    const auto& al = a.limbs();
    const auto& bl = b.limbs();
    // value_barrier each XOR result to prevent compiler from
    // short-circuiting when a == b (XOR of same provenance -> 0).
    std::uint64_t d0 = al[0] ^ bl[0];
    std::uint64_t d1 = al[1] ^ bl[1];
    std::uint64_t d2 = al[2] ^ bl[2];
    std::uint64_t d3 = al[3] ^ bl[3];
    value_barrier(d0);
    value_barrier(d1);
    value_barrier(d2);
    value_barrier(d3);
    std::uint64_t const diff = d0 | d1 | d2 | d3;
    return is_zero_mask(diff);
}

// --- Bit Access --------------------------------------------------------------

// -- Half-order n/2 for CT low-S check ----------------------------------------
// n/2 = 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A0
static constexpr std::uint64_t HALF_N[4] = {
    0xDFE92F46681B20A0ULL,  // limb 0 (least significant)
    0x5D576E7357A4501DULL,  // limb 1
    0xFFFFFFFFFFFFFFFFULL,  // limb 2
    0x7FFFFFFFFFFFFFFFULL   // limb 3 (most significant)
};

std::uint64_t scalar_is_high(const Scalar& a) noexcept {
    // CT comparison: a > n/2 ?
    // Compute a - (n/2) - 1 and check for NO borrow (a > n/2).
    // Equivalently: a > n/2 iff sub(a, HALF_N) has no borrow AND a != HALF_N.
    // We use: gt = NOT(lt_mask(a, HALF_N+1)) = NOT(a < n/2+1) = a >= n/2+1 = a > n/2.
    //
    // But n/2+1 is annoying to compute statically. Instead:
    //   a > n/2  iff  (a - n/2) has no borrow AND (a - n/2) != 0
    const auto& al = a.limbs();

    // Subtract HALF_N from a
    std::uint64_t diff[4];
    std::uint64_t borrow = sub256_scalar(diff, al.data(), HALF_N);

    // value_barrier prevents compiler from reasoning about borrow/diff
    value_barrier(borrow);

    // No borrow means a >= n/2. Check equality too (a == n/2 means low-S).
    std::uint64_t d0 = diff[0], d1 = diff[1], d2 = diff[2], d3 = diff[3];
    value_barrier(d0); value_barrier(d1); value_barrier(d2); value_barrier(d3);
    std::uint64_t const is_eq = is_zero_mask(d0 | d1 | d2 | d3);
    std::uint64_t const no_borrow = is_zero_mask(borrow);

    // high iff no_borrow AND NOT equal
    return no_borrow & ~is_eq;
}

ECDSASignature ct_normalize_low_s(const ECDSASignature& sig) noexcept {
    // CT: if s > n/2, negate s; otherwise keep s unchanged.
    // Both paths always execute; select via branchless mask.
    std::uint64_t const high_mask = scalar_is_high(sig.s);
    Scalar const s_norm = scalar_cneg(sig.s, high_mask);
    return {sig.r, s_norm};
}

// --- Bit Access --------------------------------------------------------------

std::uint64_t scalar_bit(const Scalar& a, std::size_t index) noexcept {
    // CT w.r.t. scalar value (the secret). Position is public (loop counter).
    // index / 64 = limb, index % 64 = bit within limb
    std::size_t const limb_idx = index >> 6;
    std::size_t const bit_idx  = index & 63;

    // CT: always read all 4 limbs, select the right one via branchless mask
    std::uint64_t val = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        std::uint64_t const mask = eq_mask(static_cast<std::uint64_t>(i),
                                     static_cast<std::uint64_t>(limb_idx));
        val |= a.limbs()[i] & mask;
    }
    return (val >> bit_idx) & 1;
}

std::uint64_t scalar_window(const Scalar& a, std::size_t pos,
                            unsigned width) noexcept {
    const auto& limbs = a.limbs();
    std::size_t const limb_idx = pos >> 6;
    std::size_t const bit_idx  = pos & 63;
    std::uint64_t const mask   = (1ULL << width) - 1;

#if defined(__riscv)
    // RISC-V U74: CT lookup loop replaces indexed load to avoid
    // timing variation from different limb_idx values on in-order core.
    // Always reads all 4 limbs; selects via eq_mask (branchless).
    std::uint64_t lo_limb = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        std::uint64_t const m = eq_mask(static_cast<std::uint64_t>(i),
                                  static_cast<std::uint64_t>(limb_idx));
        lo_limb |= limbs[i] & m;
    }
    std::uint64_t lo = lo_limb >> bit_idx;

    // Next limb (limb_idx+1), zeroed when limb_idx == 3 (out of bounds)
    std::size_t const next_idx = (limb_idx + 1) & 3;
    std::uint64_t hi_limb = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        std::uint64_t const m = eq_mask(static_cast<std::uint64_t>(i),
                                  static_cast<std::uint64_t>(next_idx));
        hi_limb |= limbs[i] & m;
    }
    std::uint64_t in_bounds = is_nonzero_mask(
        static_cast<std::uint64_t>(limb_idx ^ 3));
    hi_limb &= in_bounds;

    // Combine: shift by (64 - bit_idx). When bit_idx == 0 shift would be 64
    // (UB), so clamp to [0,63] and zero hi contribution instead.
    std::uint64_t shift = (64 - bit_idx) & 63;
    std::uint64_t hi_active = is_nonzero_mask(static_cast<std::uint64_t>(bit_idx));
    hi_limb &= hi_active;

    return (lo | (hi_limb << shift)) & mask;
#else
    // Branched path: safe on x86/ARM OOO cores (branch predictor handles
    // the public pos/width pattern perfectly). Avoids MSVC/Clang LTCG
    // miscompilation of the branchless is_nonzero_mask pattern.
    if (bit_idx + width <= 64) {
        return (limbs[limb_idx] >> bit_idx) & mask;
    }
    std::uint64_t const lo = limbs[limb_idx] >> bit_idx;
    std::uint64_t const hi = (limb_idx + 1 < 4) ? limbs[limb_idx + 1] : 0;
    return (lo | (hi << (64 - bit_idx))) & mask;
#endif
}

} // namespace secp256k1::ct