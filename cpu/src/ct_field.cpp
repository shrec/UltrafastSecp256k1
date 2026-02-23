// ============================================================================
// Constant-Time Field Arithmetic -- Implementation
// ============================================================================
// All operations have data-independent execution traces.
// p = 2^256 - 2^32 - 977 = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
//
// PLATFORM DISPATCH (field_mul / field_sqr):
//   x86-64:  Inline FE52 5x52 multiply -- zero call overhead, best codegen.
//   ARM64:   fast::operator* -> native ASM (already fastest for ARM64).
//   Generic: fast::operator* + field_normalize (MSVC, 32-bit -- not perf-critical).
//
// PLATFORM DISPATCH (field_inv):
//   x86-64 / ARM64 (__int128): SafeGCD 10x59=590 divsteps (CT, matches libsecp).
//   Generic:                   Fermat chain a^(p-2) via field_mul/field_sqr.
// ============================================================================

#include "secp256k1/ct/field.hpp"
#include "secp256k1/ct/ops.hpp"
#include "secp256k1/config.hpp"
#include <cstring>
#include <cstdint>

// Include 5x52 inline implementations for __int128 platforms.
// The inline 5x52 C path produces identical asm to hand-written mul/sqr
// but with zero function-call overhead and superior register allocation.
#if defined(__SIZEOF_INT128__)
#include "secp256k1/field_52.hpp"
#include "secp256k1/field_52_impl.hpp"
#endif

// --- Platform dispatch -------------------------------------------------------
// x86-64 (Clang/GCC): inline FE52 5x52 multiply -- zero call overhead, best codegen.
// ARM64 (Clang/GCC): operator* delegates to native ASM via fast:: -- already optimal.
// Generic (MSVC/32-bit): operator* + field_normalize (functional, not perf-critical).

namespace secp256k1::ct {

// secp256k1 prime p in 4 x 64-bit limbs (little-endian)
static constexpr std::uint64_t P[4] = {
    0xFFFFFFFEFFFFFC2FULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL
};

// 2^256 mod p = 2^32 + 977 = 0x1000003D1
static constexpr std::uint64_t MOD_K = 0x1000003D1ULL;

// --- Internal helpers --------------------------------------------------------

// CT 256-bit addition with carry out. Returns carry (0 or 1).
static inline std::uint64_t add256(std::uint64_t r[4],
                                    const std::uint64_t a[4],
                                    const std::uint64_t b[4]) noexcept {
    std::uint64_t carry = 0;
    for (int i = 0; i < 4; ++i) {
        // r[i] = a[i] + b[i] + carry
        std::uint64_t sum_lo = a[i] + b[i];
        std::uint64_t c1 = static_cast<std::uint64_t>(sum_lo < a[i]);
        std::uint64_t sum = sum_lo + carry;
        std::uint64_t c2 = static_cast<std::uint64_t>(sum < sum_lo);
        r[i] = sum;
        carry = c1 + c2;
    }
    return carry;
}

// CT 256-bit subtraction with borrow out. Returns borrow (0 or 1).
static inline std::uint64_t sub256(std::uint64_t r[4],
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

// CT normalize: reduce to [0, p) without branches
// If value >= p, subtract p. Uses cmov.
static inline void ct_reduce_once(std::uint64_t r[4]) noexcept {
    std::uint64_t tmp[4];
    std::uint64_t borrow = sub256(tmp, r, P);
    // If borrow == 0, r >= p -> use tmp (reduced). Else keep r.
    // mask = 0xFFF...F if no borrow (r >= p), else 0
    std::uint64_t mask = is_zero_mask(borrow);
    cmov256(r, tmp, mask);
}

// --- Public API --------------------------------------------------------------

FieldElement field_normalize(const FieldElement& a) noexcept {
    std::uint64_t r[4];
    std::memcpy(r, &a.limbs()[0], 32);
    ct_reduce_once(r);
    return FieldElement::from_limbs_raw({r[0], r[1], r[2], r[3]});
}

FieldElement field_add(const FieldElement& a, const FieldElement& b) noexcept {
    std::uint64_t r[4];
    std::uint64_t carry = add256(r, a.limbs().data(), b.limbs().data());

    // If carry OR r >= p, subtract p
    // First subtract p unconditionally
    std::uint64_t tmp[4];
    std::uint64_t borrow = sub256(tmp, r, P);

    // If carry, we definitely need to subtract p (result overflowed 256 bits)
    // If no carry and no borrow, we still use tmp (r was >= p)
    // If no carry and borrow, keep r (r was < p)
    // Combined: use_tmp if (carry OR (borrow == 0))
    // mask = all-ones if we should use tmp
    std::uint64_t no_borrow = is_zero_mask(borrow);
    std::uint64_t has_carry = is_nonzero_mask(carry);
    std::uint64_t mask = no_borrow | has_carry;
    cmov256(r, tmp, mask);

    return FieldElement::from_limbs_raw({r[0], r[1], r[2], r[3]});
}

FieldElement field_sub(const FieldElement& a, const FieldElement& b) noexcept {
    std::uint64_t r[4];
    std::uint64_t borrow = sub256(r, a.limbs().data(), b.limbs().data());

    // If borrow, add p back: r += p
    std::uint64_t tmp[4];
    add256(tmp, r, P);

    // mask = all-ones if borrow occurred
    std::uint64_t mask = is_nonzero_mask(borrow);
    cmov256(r, tmp, mask);

    return FieldElement::from_limbs_raw({r[0], r[1], r[2], r[3]});
}

FieldElement field_mul(const FieldElement& a, const FieldElement& b) noexcept {
#if defined(__SIZEOF_INT128__)
    // FE52 5x52 multiply with integrated reduction -- best codegen on
    // x86-64 (BMI2), ARM64, and RISC-V (dedicated asm: fe52_mul_inner_riscv64).
    using FE52 = secp256k1::fast::FieldElement52;
    return (FE52::from_fe(a) * FE52::from_fe(b)).to_fe();
#else
    // MSVC / ESP32: 4x64 Comba with separate reduction.
    return a * b;
#endif
}

FieldElement field_sqr(const FieldElement& a) noexcept {
#if defined(__SIZEOF_INT128__)
    // FE52 5x52 square with integrated reduction -- best codegen on
    // x86-64 (BMI2), ARM64, and RISC-V (dedicated asm: fe52_sqr_inner_riscv64).
    using FE52 = secp256k1::fast::FieldElement52;
    return FE52::from_fe(a).square().to_fe();
#else
    // MSVC / ESP32: 4x64 square with separate reduction.
    return a.square();
#endif
}

FieldElement field_neg(const FieldElement& a) noexcept {
    // -a mod p = p - a (if a != 0), 0 (if a == 0)
    // CT: always compute p - a, then cmov to 0 if a was zero
    std::uint64_t r[4];
    sub256(r, P, a.limbs().data());

    std::uint64_t zero_mask = field_is_zero(a);
    // If a == 0, set r to 0
    std::uint64_t z[4] = {0, 0, 0, 0};
    cmov256(r, z, zero_mask);

    return FieldElement::from_limbs_raw({r[0], r[1], r[2], r[3]});
}

FieldElement field_half(const FieldElement& a) noexcept {
    // r = a/2 mod p. Branchless.
    // If a is odd: r = (a + p) / 2; if even: r = a / 2.
    const auto& al = a.limbs();
    std::uint64_t odd = -(al[0] & 1);  // all-ones if odd, 0 if even

    // Conditionally add p (only if odd)
    std::uint64_t t[4];
    std::uint64_t carry = 0;
    for (int i = 0; i < 4; ++i) {
        std::uint64_t addend = P[i] & odd;
        std::uint64_t sum_lo = al[i] + addend;
        std::uint64_t c1 = static_cast<std::uint64_t>(sum_lo < al[i]);
        std::uint64_t sum = sum_lo + carry;
        std::uint64_t c2 = static_cast<std::uint64_t>(sum < sum_lo);
        t[i] = sum;
        carry = c1 + c2;
    }

    // Right shift 257-bit value (t[0..3] + carry*2^256) by 1
    std::uint64_t r0 = (t[0] >> 1) | (t[1] << 63);
    std::uint64_t r1 = (t[1] >> 1) | (t[2] << 63);
    std::uint64_t r2 = (t[2] >> 1) | (t[3] << 63);
    std::uint64_t r3 = (t[3] >> 1) | (carry << 63);

    return FieldElement::from_limbs_raw({r0, r1, r2, r3});
}

// -- CT SafeGCD building blocks (extracted for register allocation) ----------
#if defined(__SIZEOF_INT128__)
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
namespace {

struct SG62   { int64_t v[5]; };
struct SGTrans { int64_t u, v, q, r; };

// secp256k1 prime in signed-62:  p = 2^256 - 2^32 - 977
static constexpr SG62 SGP = {{
    -(int64_t)0x1000003D1LL, 0, 0, 0, 256
}};
// (-p)^{-1} mod 2^62
static constexpr uint64_t P_INV62 = 0x27C7F6E22DDACACFULL;

// CT batch of 59 divsteps using conditional-add pattern (not conditional-swap).
// Result matrix is scaled by 2^62 (initial 8xidentity + 59 halvings).
// zeta = -(delta + 1/2); swap condition is zeta < 0 (i.e., delta > 0).
// Matches libsecp: 10 x 59 = 590 divsteps (proven sufficient for 256-bit).
static int64_t ct_divsteps_59(int64_t zeta, uint64_t f0, uint64_t g0,
                               SGTrans& t) noexcept {
    // Start with 8xidentity (2^3) so output is scaled by 2^62 (3 + 59 = 62).
    uint64_t u = 8, v = 0, q = 0, r = 8;
    uint64_t f = f0, g = g0;

    for (int i = 3; i < 62; ++i) {
        uint64_t c1 = (uint64_t)(zeta >> 63);    // all-ones if zeta < 0
        uint64_t c2 = -(g & 1);                   // all-ones if g odd

        // Conditionally negate (f,u,v) based on zeta sign
        uint64_t x = (f ^ c1) - c1;
        uint64_t y = (u ^ c1) - c1;
        uint64_t z = (v ^ c1) - c1;

        // Conditionally add to (g,q,r) if g is odd
        g += x & c2;
        q += y & c2;
        r += z & c2;

        // Combined mask: zeta < 0 AND g was odd
        c1 &= c2;

        // zeta update: swap -> -zeta-2, no-swap -> zeta-1
        zeta = (zeta ^ (int64_t)c1) - 1;

        // Conditionally add (g,q,r) into (f,u,v) when both conditions met
        f += g & c1;
        u += q & c1;
        v += r & c1;

        g >>= 1;
        u <<= 1;
        v <<= 1;
    }

    t.u = (int64_t)u; t.v = (int64_t)v;
    t.q = (int64_t)q; t.r = (int64_t)r;
    return zeta;
}

// Apply transition matrix to (d,e) mod p.
// Exploits secp256k1 prime structure: p.v[1..3]=0, p.v[4]=256=2^8.
static void ct_update_de(SG62& d, SG62& e, const SGTrans& t) noexcept {
    const uint64_t M62 = UINT64_MAX >> 2;
    const int64_t d0=d.v[0], d1=d.v[1], d2=d.v[2], d3=d.v[3], d4=d.v[4];
    const int64_t e0=e.v[0], e1=e.v[1], e2=e.v[2], e3=e.v[3], e4=e.v[4];
    int64_t md, me;
    __int128 cd, ce;

    md = (t.u & (d4 >> 63)) + (t.v & (e4 >> 63));
    me = (t.q & (d4 >> 63)) + (t.r & (e4 >> 63));

    cd = (__int128)t.u * d0 + (__int128)t.v * e0;
    ce = (__int128)t.q * d0 + (__int128)t.r * e0;

    md -= (int64_t)((P_INV62 * (uint64_t)cd + (uint64_t)md) & M62);
    me -= (int64_t)((P_INV62 * (uint64_t)ce + (uint64_t)me) & M62);

    cd += (__int128)SGP.v[0] * md;
    ce += (__int128)SGP.v[0] * me;
    cd >>= 62;
    ce >>= 62;

    cd += (__int128)t.u * d1 + (__int128)t.v * e1;
    ce += (__int128)t.q * d1 + (__int128)t.r * e1;
    d.v[0] = (int64_t)((uint64_t)cd & M62); cd >>= 62;
    e.v[0] = (int64_t)((uint64_t)ce & M62); ce >>= 62;

    cd += (__int128)t.u * d2 + (__int128)t.v * e2;
    ce += (__int128)t.q * d2 + (__int128)t.r * e2;
    d.v[1] = (int64_t)((uint64_t)cd & M62); cd >>= 62;
    e.v[1] = (int64_t)((uint64_t)ce & M62); ce >>= 62;

    cd += (__int128)t.u * d3 + (__int128)t.v * e3;
    ce += (__int128)t.q * d3 + (__int128)t.r * e3;
    d.v[2] = (int64_t)((uint64_t)cd & M62); cd >>= 62;
    e.v[2] = (int64_t)((uint64_t)ce & M62); ce >>= 62;

    cd += (__int128)t.u * d4 + (__int128)t.v * e4 + ((__int128)md << 8);
    ce += (__int128)t.q * d4 + (__int128)t.r * e4 + ((__int128)me << 8);
    d.v[3] = (int64_t)((uint64_t)cd & M62); cd >>= 62;
    e.v[3] = (int64_t)((uint64_t)ce & M62); ce >>= 62;

    d.v[4] = (int64_t)cd;
    e.v[4] = (int64_t)ce;
}

// Apply transition matrix to full-precision (f,g).  Always 5 limbs.
static void ct_update_fg(SG62& f, SG62& g, const SGTrans& t) noexcept {
    const int64_t M62 = (int64_t)((uint64_t)(-1) >> 2);
    __int128 cf, cg;

    cf = (__int128)t.u * f.v[0] + (__int128)t.v * g.v[0];
    cg = (__int128)t.q * f.v[0] + (__int128)t.r * g.v[0];
    cf >>= 62;
    cg >>= 62;

    cf += (__int128)t.u * f.v[1] + (__int128)t.v * g.v[1];
    cg += (__int128)t.q * f.v[1] + (__int128)t.r * g.v[1];
    f.v[0] = (int64_t)cf & M62; cf >>= 62;
    g.v[0] = (int64_t)cg & M62; cg >>= 62;

    cf += (__int128)t.u * f.v[2] + (__int128)t.v * g.v[2];
    cg += (__int128)t.q * f.v[2] + (__int128)t.r * g.v[2];
    f.v[1] = (int64_t)cf & M62; cf >>= 62;
    g.v[1] = (int64_t)cg & M62; cg >>= 62;

    cf += (__int128)t.u * f.v[3] + (__int128)t.v * g.v[3];
    cg += (__int128)t.q * f.v[3] + (__int128)t.r * g.v[3];
    f.v[2] = (int64_t)cf & M62; cf >>= 62;
    g.v[2] = (int64_t)cg & M62; cg >>= 62;

    cf += (__int128)t.u * f.v[4] + (__int128)t.v * g.v[4];
    cg += (__int128)t.q * f.v[4] + (__int128)t.r * g.v[4];
    f.v[3] = (int64_t)cf & M62; cf >>= 62;
    g.v[3] = (int64_t)cg & M62; cg >>= 62;

    f.v[4] = (int64_t)cf;
    g.v[4] = (int64_t)cg;
}

// Normalize result to [0, p): conditional add/negate/carry propagation.
static void ct_sg_normalize(SG62& r, int64_t f_sign) noexcept {
    const int64_t M62 = (int64_t)(UINT64_MAX >> 2);
    int64_t r0=r.v[0], r1=r.v[1], r2=r.v[2], r3=r.v[3], r4=r.v[4];

    int64_t ca = r4 >> 63;
    r0 += SGP.v[0] & ca;
    r4 += SGP.v[4] & ca;

    int64_t cn = f_sign >> 63;
    r0 = (r0 ^ cn) - cn;
    r1 = (r1 ^ cn) - cn;
    r2 = (r2 ^ cn) - cn;
    r3 = (r3 ^ cn) - cn;
    r4 = (r4 ^ cn) - cn;

    r1 += r0 >> 62; r0 &= M62;
    r2 += r1 >> 62; r1 &= M62;
    r3 += r2 >> 62; r2 &= M62;
    r4 += r3 >> 62; r3 &= M62;

    ca = r4 >> 63;
    r0 += SGP.v[0] & ca;
    r4 += SGP.v[4] & ca;

    r1 += r0 >> 62; r0 &= M62;
    r2 += r1 >> 62; r1 &= M62;
    r3 += r2 >> 62; r2 &= M62;
    r4 += r3 >> 62; r3 &= M62;

    r.v[0]=r0; r.v[1]=r1; r.v[2]=r2; r.v[3]=r3; r.v[4]=r4;
}

} // anonymous namespace
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#endif // __SIZEOF_INT128__

FieldElement field_inv(const FieldElement& a) noexcept {
#if defined(__SIZEOF_INT128__)
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
    // -- CT SafeGCD inverse -----------------------------------------------
    // Bernstein-Yang divstep: 10 x 59 = 590 branchless divsteps.
    // Matches bitcoin-core/secp256k1 proven bound for 256-bit modular inverse.

    // fe -> s62
    const auto& al = a.limbs();
    constexpr uint64_t M = (1ULL << 62) - 1;

    SG62 d = {{0,0,0,0,0}};
    SG62 e = {{1,0,0,0,0}};
    SG62 f = SGP;
    SG62 g = {{
        (int64_t)(al[0] & M),
        (int64_t)(((al[0] >> 62) | (al[1] << 2)) & M),
        (int64_t)(((al[1] >> 60) | (al[2] << 4)) & M),
        (int64_t)(((al[2] >> 58) | (al[3] << 6)) & M),
        (int64_t)(al[3] >> 56)
    }};
    int64_t zeta = -1;  // zeta = -(delta + 1/2); delta starts at 1/2

    // 10 x 59 = 590 divsteps (proven sufficient for 256-bit; same as libsecp)
    for (int iter = 0; iter < 10; ++iter) {
        SGTrans t;
        zeta = ct_divsteps_59(zeta, (uint64_t)f.v[0], (uint64_t)g.v[0], t);
        ct_update_de(d, e, t);
        ct_update_fg(f, g, t);
    }

    ct_sg_normalize(d, f.v[4]);

    // s62 -> fe
    return FieldElement::from_limbs({
        (uint64_t)d.v[0] | ((uint64_t)d.v[1] << 62),
        ((uint64_t)d.v[1] >> 2)  | ((uint64_t)d.v[2] << 60),
        ((uint64_t)d.v[2] >> 4)  | ((uint64_t)d.v[3] << 58),
        ((uint64_t)d.v[3] >> 6)  | ((uint64_t)d.v[4] << 56)
    });

#else
    // -- Generic 4x64 path (x86_64 ASM or fallback) ----------------------
    // Fermat's little theorem: a^(p-2) mod p
    // Using addition chain optimized for secp256k1:
    // p-2 = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
    //
    // This is a fixed sequence of squarings and multiplications.
    // Always the same number of operations regardless of input.
    //
    // Optimal addition chain:
    // 1. Compute a^2 ... a^(2^k) powers via repeated squaring
    // 2. Multiply specific powers together
    // Total: 255 squarings + 14 multiplications (fixed)

    FieldElement x = a;

    // x2 = a^(2^2 - 1) = a^3
    FieldElement x2 = field_sqr(x);
    x2 = field_mul(x2, x);

    // x3 = a^(2^3 - 1) = a^7
    FieldElement x3 = field_sqr(x2);
    x3 = field_mul(x3, x);

    // x6 = a^(2^6 - 1)
    FieldElement x6 = x3;
    for (int i = 0; i < 3; ++i) x6 = field_sqr(x6);
    x6 = field_mul(x6, x3);

    // x9 = a^(2^9 - 1)
    FieldElement x9 = x6;
    for (int i = 0; i < 3; ++i) x9 = field_sqr(x9);
    x9 = field_mul(x9, x3);

    // x11 = a^(2^11 - 1)
    FieldElement x11 = x9;
    for (int i = 0; i < 2; ++i) x11 = field_sqr(x11);
    x11 = field_mul(x11, x2);

    // x22 = a^(2^22 - 1)
    FieldElement x22 = x11;
    for (int i = 0; i < 11; ++i) x22 = field_sqr(x22);
    x22 = field_mul(x22, x11);

    // x44 = a^(2^44 - 1)
    FieldElement x44 = x22;
    for (int i = 0; i < 22; ++i) x44 = field_sqr(x44);
    x44 = field_mul(x44, x22);

    // x88 = a^(2^88 - 1)
    FieldElement x88 = x44;
    for (int i = 0; i < 44; ++i) x88 = field_sqr(x88);
    x88 = field_mul(x88, x44);

    // x176 = a^(2^176 - 1)
    FieldElement x176 = x88;
    for (int i = 0; i < 88; ++i) x176 = field_sqr(x176);
    x176 = field_mul(x176, x88);

    // x220 = a^(2^220 - 1)
    FieldElement x220 = x176;
    for (int i = 0; i < 44; ++i) x220 = field_sqr(x220);
    x220 = field_mul(x220, x44);

    // x223 = a^(2^223 - 1)
    FieldElement x223 = x220;
    for (int i = 0; i < 3; ++i) x223 = field_sqr(x223);
    x223 = field_mul(x223, x3);

    // Final: t = x223^(2^23) * x22^(2^1) * ...
    // p-2 = 2^256 - 2^32 - 977 - 2
    //      = 2^256 - 0x1000003D3
    // Exponent in binary from bit 255 down:
    //   bits 255..33: all ones (223 ones = x223 covers bits 255..33)
    //   bit 32: 0
    //   bits 31..0: FFFFFC2D = ...11111100 00101101
    //
    // After x223 (covers bits 255..33):
    //   Square 23 times -> x223 * 2^23, then multiply by x22
    //   Square 5 times -> then multiply by a
    //   Square 3 times -> then multiply by x2
    //   Square 2 times -> then multiply by a

    FieldElement t = x223;

    // Square 23 times
    for (int i = 0; i < 23; ++i) t = field_sqr(t);
    t = field_mul(t, x22);

    // Square 5 times
    for (int i = 0; i < 5; ++i) t = field_sqr(t);
    t = field_mul(t, x);

    // Square 3 times
    for (int i = 0; i < 3; ++i) t = field_sqr(t);
    t = field_mul(t, x2);

    // Square 2 times
    for (int i = 0; i < 2; ++i) t = field_sqr(t);
    t = field_mul(t, x);

    return t;
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#endif  // __SIZEOF_INT128__
}

// --- Conditional Operations --------------------------------------------------

void field_cmov(FieldElement* r, const FieldElement& a,
                std::uint64_t mask) noexcept {
    // In-place XOR-mask conditional move -- no temporary FieldElement.
    // mask is all-ones (select a) or all-zeros (keep r).
    auto& rd = r->data().limbs;
    const auto& ad = a.data().limbs;
    rd[0] ^= (rd[0] ^ ad[0]) & mask;
    rd[1] ^= (rd[1] ^ ad[1]) & mask;
    rd[2] ^= (rd[2] ^ ad[2]) & mask;
    rd[3] ^= (rd[3] ^ ad[3]) & mask;
}

void field_cswap(FieldElement* a, FieldElement* b,
                 std::uint64_t mask) noexcept {
    // Direct in-place XOR swap -- no temporaries.
    auto& ad = a->data().limbs;
    auto& bd = b->data().limbs;
    for (int i = 0; i < 4; ++i) {
        std::uint64_t diff = (ad[i] ^ bd[i]) & mask;
        ad[i] ^= diff;
        bd[i] ^= diff;
    }
}

FieldElement field_select(const FieldElement& a, const FieldElement& b,
                          std::uint64_t mask) noexcept {
    const auto& al = a.limbs();
    const auto& bl = b.limbs();
    return FieldElement::from_limbs_raw({
        ct_select(al[0], bl[0], mask),
        ct_select(al[1], bl[1], mask),
        ct_select(al[2], bl[2], mask),
        ct_select(al[3], bl[3], mask)
    });
}

FieldElement field_cneg(const FieldElement& a, std::uint64_t mask) noexcept {
    FieldElement neg = field_neg(a);
    return field_select(neg, a, mask);
}

// --- Comparison --------------------------------------------------------------

std::uint64_t field_is_zero(const FieldElement& a) noexcept {
    const auto& l = a.limbs();
    std::uint64_t z = l[0] | l[1] | l[2] | l[3];
    return is_zero_mask(z);
}

std::uint64_t field_eq(const FieldElement& a, const FieldElement& b) noexcept {
    const auto& al = a.limbs();
    const auto& bl = b.limbs();
    std::uint64_t diff = (al[0] ^ bl[0]) | (al[1] ^ bl[1]) |
                         (al[2] ^ bl[2]) | (al[3] ^ bl[3]);
    return is_zero_mask(diff);
}

} // namespace secp256k1::ct
