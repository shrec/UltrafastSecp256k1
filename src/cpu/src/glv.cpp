// GLV endomorphism implementation for secp256k1
// Correct decomposition following libsecp256k1 algorithm:
//   k = k1 + k2*lambda (mod n), where |k1|,|k2| ~= sqrtn

#include "secp256k1/glv.hpp"
#include "secp256k1/field.hpp"
#include <cstring>

namespace secp256k1::fast {

// ============================================================================
//  Internal helpers for GLV decomposition
// ============================================================================

#if defined(__SIZEOF_INT128__)
// Suppress -Wpedantic for __int128 (GCC extension, required for 64-bit Comba)
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif

// 192-bit accumulator macros for Comba multiplication (shared by all mul variants).
// Accumulator state: c0:c1:c2 (local variables, declared in each caller).
// IMPORTANT: callers must declare these locals:
//   using u128 = unsigned __int128;
//   std::uint64_t c0 = 0, c1 = 0;
//   std::uint32_t c2 = 0;
#define GLV_MULADD(i, j) do { \
    const u128 p_ = (u128)(a[i]) * (b[j]); \
    const std::uint64_t tl_ = (std::uint64_t)p_; \
    std::uint64_t th_ = (std::uint64_t)(p_ >> 64); \
    c0 += tl_; \
    th_ += (c0 < tl_) ? 1ULL : 0ULL; \
    c1 += th_; \
    c2 += (c1 < th_) ? 1U : 0U; \
} while(0)

#define GLV_EXTRACT(out) do { \
    (out) = c0; \
    c0 = c1; \
    c1 = static_cast<std::uint64_t>(c2); \
    c2 = 0; \
} while(0)

// 64-bit Comba using __int128: 4x4 = 16 multiplications (vs 8x8 = 64 at 32-bit).
// Each 64x64->128 multiply maps to MUL + MULHU on x86-64, UMULH on AArch64.
// Carry chain uses libsecp256k1-style 192-bit accumulator (c0:c1:c2).
// Result: product[0..7] as 64-bit limbs (512 bits total).
static void glv_mul_comba_64(const std::uint64_t a[4], const std::uint64_t b[4],
                             std::uint64_t r[8]) {
    using u128 = unsigned __int128;
    std::uint64_t c0 = 0, c1 = 0;
    std::uint32_t c2 = 0;

    GLV_MULADD(0, 0);
    GLV_EXTRACT(r[0]);
    GLV_MULADD(0, 1);  GLV_MULADD(1, 0);
    GLV_EXTRACT(r[1]);
    GLV_MULADD(0, 2);  GLV_MULADD(1, 1);  GLV_MULADD(2, 0);
    GLV_EXTRACT(r[2]);
    GLV_MULADD(0, 3);  GLV_MULADD(1, 2);  GLV_MULADD(2, 1);  GLV_MULADD(3, 0);
    GLV_EXTRACT(r[3]);
    GLV_MULADD(1, 3);  GLV_MULADD(2, 2);  GLV_MULADD(3, 1);
    GLV_EXTRACT(r[4]);
    GLV_MULADD(2, 3);  GLV_MULADD(3, 2);
    GLV_EXTRACT(r[5]);
    GLV_MULADD(3, 3);
    GLV_EXTRACT(r[6]);
    r[7] = c0;
}

// Template version: b[] constants known at compile time -> compiler can
// constant-fold multiplies and optimize register allocation.
template<std::uint64_t B0, std::uint64_t B1, std::uint64_t B2, std::uint64_t B3>
static std::array<std::uint64_t, 4> mul_shift_384_const(
    const std::array<std::uint64_t, 4>& a) {

    static constexpr std::uint64_t b[4] = {B0, B1, B2, B3};
    std::uint64_t prod[8];
    glv_mul_comba_64(a.data(), b, prod);

    std::array<std::uint64_t, 4> result{};
    result[0] = prod[6];
    result[1] = prod[7];

    // Rounding bit: bit 383 of 512-bit product = bit 63 of prod[5]
    if (prod[5] >> 63) {
        result[0]++;
        if (result[0] == 0) result[1]++;
    }
    return result;
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#else
// 32-bit fallback for platforms without __int128 (e.g. ESP32)
// Comba product-scanning: 8x8 -> 16 words (256x256 -> 512 bit)
static void glv_mul_comba(const std::uint32_t a[8], const std::uint32_t b[8],
                          std::uint32_t r[16]) {
    std::uint32_t c0 = 0, c1 = 0, c2 = 0;
    for (int k = 0; k < 15; k++) {
        const int lo = (k < 8) ? 0 : (k - 7);
        const int hi = (k < 8) ? k : 7;
        for (int i = lo; i <= hi; i++) {
            std::uint64_t const p = (std::uint64_t)a[i] * b[k - i];
            auto const plo = (std::uint32_t)p;
            auto const phi = (std::uint32_t)(p >> 32);
            std::uint64_t s = (std::uint64_t)c0 + plo;
            c0 = (std::uint32_t)s;
            s = (std::uint64_t)c1 + phi + (s >> 32);
            c1 = (std::uint32_t)s;
            c2 += (std::uint32_t)(s >> 32);
        }
        r[k] = c0;
        c0 = c1;
        c1 = c2;
        c2 = 0;
    }
    r[15] = c0;
}

static void limbs64_to_32(const std::uint64_t* src, std::uint32_t* dst) {
    for (int i = 0; i < 4; i++) {
        dst[static_cast<std::size_t>(i) * 2]     = (std::uint32_t)src[i];
        dst[static_cast<std::size_t>(i) * 2 + 1] = (std::uint32_t)(src[i] >> 32);
    }
}

static std::array<std::uint64_t, 4> mul_shift_384(
    const std::array<std::uint64_t, 4>& a,
    const std::array<std::uint64_t, 4>& b) {

    std::uint32_t a32[8], b32[8], prod[16];
    limbs64_to_32(a.data(), a32);
    limbs64_to_32(b.data(), b32);
    glv_mul_comba(a32, b32, prod);

    std::array<std::uint64_t, 4> result{};
    result[0] = (std::uint64_t)prod[12] | ((std::uint64_t)prod[13] << 32);
    result[1] = (std::uint64_t)prod[14] | ((std::uint64_t)prod[15] << 32);

    if (prod[11] >> 31) {
        result[0]++;
        if (result[0] == 0) result[1]++;
    }
    return result;
}

// Template wrapper for 32-bit path (calls runtime mul_shift_384)
template<std::uint64_t B0, std::uint64_t B1, std::uint64_t B2, std::uint64_t B3>
static std::array<std::uint64_t, 4> mul_shift_384_const(
    const std::array<std::uint64_t, 4>& a) {
    const std::array<std::uint64_t, 4> b{{B0, B1, B2, B3}};
    return mul_shift_384(a, b);
}
#endif

// ============================================================================
//  GLV decomposition constants (matching libsecp256k1/precompute.cpp)
// ============================================================================

// g1/g2: precomputed multipliers for c1 = round(k*g1 / 2^384), c2 = round(k*g2 / 2^384)
// (little-endian 64-bit limbs)
static constexpr std::array<std::uint64_t, 4> kG1{{
    0xE893209A45DBB031ULL, 0x3DAA8A1471E8CA7FULL,
    0xE86C90E49284EB15ULL, 0x3086D221A7D46BCDULL
}};
static constexpr std::array<std::uint64_t, 4> kG2{{
    0x1571B4AE8AC47F71ULL, 0x221208AC9DF506C6ULL,
    0x6F547FA90ABFE4C4ULL, 0xE4437ED6010E8828ULL
}};

// minus_b1 and minus_b2 as big-endian 32-byte arrays (for Scalar::from_bytes)
// Only needed on platforms without __int128 (MSVC, 32-bit) where the
// fallback Scalar-arithmetic GLV path is used.
#if !defined(__SIZEOF_INT128__)
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

// lambda (lambda) scalar as big-endian bytes
static constexpr std::array<std::uint8_t, 32> kGlvLambdaBytes{{
    0x53,0x63,0xAD,0x4C,0xC0,0x5C,0x30,0xE0,
    0xA5,0x26,0x1C,0x02,0x88,0x12,0x64,0x5A,
    0x12,0x2E,0x22,0xEA,0x20,0x81,0x66,0x78,
    0xDF,0x02,0x96,0x7C,0x1B,0x23,0xBD,0x72
}};
#endif // !__SIZEOF_INT128__

// ============================================================================
//  Fast GLV decomposition helpers (exploit known limb sizes)
// ============================================================================

// Group order n (little-endian 64-bit limbs) — used by both fast and fallback paths
static constexpr std::uint64_t kN[4] = {
    0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
};

// n/2 = floor(n/2) (little-endian). k > n/2 iff k is "high" (negated form is shorter).
// ROUND3-1: half-order compare for sign selection, avoids two scalar_bitlen calls.
static constexpr std::uint64_t kNHalf[4] = {
    0xDFE92F46681B20A0ULL, 0x5D576E7357A4501DULL,
    0xFFFFFFFFFFFFFFFFULL, 0x7FFFFFFFFFFFFFFFULL
};

// Compare 4-limb unsigned value: a >= b (little-endian)
static bool glv_ge_n(const std::uint64_t a[4], const std::uint64_t b[4]) {
    for (int i = 3; i >= 0; --i) {
        if (a[i] > b[i]) return true;
        if (a[i] < b[i]) return false;
    }
    return true; // equal
}

// Subtract b from a (4-limb). Returns borrow (0 or 1).
static std::uint64_t glv_sub4(const std::uint64_t a[4], const std::uint64_t b[4],
                               std::uint64_t r[4]) {
    std::uint64_t borrow = 0;
    for (int i = 0; i < 4; ++i) {
        std::uint64_t t = a[i] - borrow;
        borrow = (a[i] < borrow) ? 1ULL : 0ULL;
        std::uint64_t s = t - b[i];
        borrow += (t < b[i]) ? 1ULL : 0ULL;
        r[i] = s;
    }
    return borrow;
}

#if defined(__SIZEOF_INT128__)
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif

// minus_b1 as raw limbs (128-bit: only 2 limbs non-zero)
static constexpr std::uint64_t kMB1[2] = {
    0x6F547FA90ABFE4C3ULL, 0xE4437ED6010E8828ULL
};

// minus_b2 as raw limbs (256-bit). Kept only to pin kB2 below at compile time:
// the decomposition uses b2 itself, never n - b2.
static constexpr std::uint64_t kMB2[4] = {
    0xD765CDA83DB1562CULL, 0x8A280AC50774346DULL,
    0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
};

// b2 = n - minus_b2, the raw lattice basis element (126 bits, 2 limbs).
// b2 also equals the lattice element a1 exactly, so this one constant serves
// both the k2 and the k1 product below.
static constexpr std::uint64_t kB2[2] = {
    0xE86C90E49284EB15ULL, 0x3086D221A7D46BCDULL
};

// a2 - 2^128, the low part of the lattice element a2 (125 bits, 2 limbs).
// a2 = 2^128 + kA2LO, so the 2^128 term costs a limb placement, not a multiply.
static constexpr std::uint64_t kA2LO[2] = {
    0x57C1108D9D44CFD8ULL, 0x14CA50F7A8E2F3F6ULL
};

// kB2 == n - kMB2, checked limb by limb at compile time so a mistyped constant
// cannot become a silent wrong decomposition. Limb 0 borrows, limb 1 absorbs
// the borrow without generating one, limbs 2 and 3 cancel exactly.
static_assert(kN[0] < kMB2[0], "n - minus_b2: limb 0 must borrow");
static_assert(kB2[0] == kN[0] - kMB2[0], "kB2 limb 0 != (n - minus_b2) limb 0");
static_assert(kN[1] - 1ULL >= kMB2[1], "n - minus_b2: limb 1 must not borrow");
static_assert(kB2[1] == kN[1] - kMB2[1] - 1ULL, "kB2 limb 1 != (n - minus_b2) limb 1");
static_assert(kN[2] == kMB2[2] && kN[3] == kMB2[3], "n - minus_b2 must fit 2 limbs");

// Add b to a (4-limb). Returns carry out (0 or 1).
static std::uint64_t glv_add4(const std::uint64_t a[4], const std::uint64_t b[4],
                              std::uint64_t r[4]) {
    std::uint64_t carry = 0;
    for (int i = 0; i < 4; ++i) {
        std::uint64_t const s = a[i] + b[i];
        std::uint64_t const c1_ = (s < a[i]) ? 1ULL : 0ULL;
        std::uint64_t const t = s + carry;
        carry = c1_ + ((t < s) ? 1ULL : 0ULL);
        r[i] = t;
    }
    return carry;
}

// r = a - b (mod n). Requires a < n and b < n, which every caller below
// establishes from the proven operand widths. On borrow the true value is
// a - b + 2^256, so folding n back in mod 2^256 (dropping the carry out)
// yields a - b + n, which is in [0, n).
static void glv_sub_mod_n(const std::uint64_t a[4], const std::uint64_t b[4],
                          std::uint64_t r[4]) {
    if (glv_sub4(a, b, r) != 0) {
        (void)glv_add4(r, kN, r);
    }
}

// 128-bit x 128-bit -> 256-bit Comba multiply (4 macs)
static void glv_mul_2x2(const std::uint64_t a[2], const std::uint64_t b[2],
                         std::uint64_t r[4]) {
    using u128 = unsigned __int128;
    std::uint64_t c0 = 0, c1 = 0;
    std::uint32_t c2 = 0;

    GLV_MULADD(0, 0);
    GLV_EXTRACT(r[0]);
    GLV_MULADD(0, 1); GLV_MULADD(1, 0);
    GLV_EXTRACT(r[1]);
    GLV_MULADD(1, 1);
    GLV_EXTRACT(r[2]);
    r[3] = c0;
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#undef GLV_MULADD
#undef GLV_EXTRACT

#endif // __SIZEOF_INT128__

// ============================================================================
//  Public API
// ============================================================================

GLVDecomposition glv_decompose(const Scalar& k) {
    GLVDecomposition result;

    // Step 1: c1 = round(k * g1 / 2^384),  c2 = round(k * g2 / 2^384)
    auto k_limbs = k.limbs();
    const std::array<std::uint64_t, 4> k_arr{{k_limbs[0], k_limbs[1], k_limbs[2], k_limbs[3]}};
    auto c1_limbs = mul_shift_384_const<kG1[0], kG1[1], kG1[2], kG1[3]>(k_arr);
    auto c2_limbs = mul_shift_384_const<kG2[0], kG2[1], kG2[2], kG2[3]>(k_arr);

#if defined(__SIZEOF_INT128__)
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
    // ---- Fast path: raw lattice basis (a1, a2, b1, b2) on known limb sizes ----
    // c1, c2 are at most 128-bit (only limbs[0..1] non-zero from mul_shift_384).
    //
    // The derived constant minus_b2 = n - b2 is a full 256-bit value, but b2
    // itself is only 126 bits, so working with the raw basis keeps every product
    // at 128x128 = 4 macs and inside 4 limbs:
    //     k2 = c1*mb1 + c2*mb2  ==  c1*mb1 - c2*b2       (mod n)
    // and the exact lattice relations a1 + b1*lambda == 0, a2 + b2*lambda == 0
    // (mod n) give lambda*k2 == c1*a1 + c2*a2, so k1 needs no lambda multiply
    // and no dependency on k2 at all:
    //     k1 = k - c1*a1 - c2*a2                          (mod n)
    // with a1 == b2 and a2 == 2^128 + a2_lo, so the 2^128 term is limb
    // placement rather than a multiply.
    //
    // OPERAND WIDTHS, worst case over every 4-limb k (c1 and c2 are monotone in
    // k, so the maxima are attained at the largest k; checked there, including
    // an unreduced k = 2^256-1):
    //     c1 <= 126 bits, c2 <= 128 bits
    //     c1*mb1 254 bits, c2*b2 254 bits, c1*a1 + c2*a2_lo 253 bits (no carry
    //     out of limb 3), c2<<128 256 bits
    // All four are < n, i.e. already reduced. That is why no wide mod-n
    // reduction appears below and only the modular subtracts can borrow.

    // k2 = c1*mb1 - c2*b2 (mod n): two 2x2 products, both < n.
    std::uint64_t p1[4];
    std::uint64_t p2[4];
    glv_mul_2x2(c1_limbs.data(), kMB1, p1);
    glv_mul_2x2(c2_limbs.data(), kB2, p2);
    std::uint64_t k2_raw[4];
    glv_sub_mod_n(p1, p2, k2_raw);

    // k2 sign handling: pick shorter representation.
    // ROUND3-1: half-order compare on raw limbs (avoids two scalar_bitlen + one Scalar sub).
    // k2_raw > n/2 iff negated form is shorter (k2_is_neg).
    bool const k2_is_neg = !glv_ge_n(kNHalf, k2_raw); // k2_raw > kNHalf
    std::uint64_t k2_abs_raw[4];
    if (k2_is_neg) {
        glv_sub4(kN, k2_raw, k2_abs_raw); // k2_abs = n - k2_raw
    } else {
        k2_abs_raw[0] = k2_raw[0]; k2_abs_raw[1] = k2_raw[1];
        k2_abs_raw[2] = k2_raw[2]; k2_abs_raw[3] = k2_raw[3];
    }
    Scalar const k2_abs = Scalar::from_limbs(
        {k2_abs_raw[0], k2_abs_raw[1], k2_abs_raw[2], k2_abs_raw[3]});

    // k1 = k - c1*a1 - c2*a2 (mod n), with a1 == b2 and a2 == 2^128 + a2_lo.
    // This is lambda*k2 rewritten through the lattice relations, so it is an
    // exact identity and not a second approximation: it does not depend on
    // k2's reduction, sign test or negation above, and needs neither the
    // 256-bit lambda constant nor a mod-n reduction.
    std::uint64_t t1[4];
    std::uint64_t t2[4];
    glv_mul_2x2(c1_limbs.data(), kB2,   t1); // c1*a1
    glv_mul_2x2(c2_limbs.data(), kA2LO, t2); // c2*(a2 - 2^128)

    std::uint64_t t[4];
    {
        using u128 = unsigned __int128;
        u128 carry = 0;
        for (int i = 0; i < 4; ++i) {
            carry += (u128)t1[i] + t2[i];
            t[i] = (std::uint64_t)carry;
            carry >>= 64;
        }
        // Sum is 253 bits at worst, so the carry out of limb 3 is always zero.
    }

    // c2 * 2^128: c2 occupies limbs [0..1] only, so this is limb placement.
    const std::uint64_t t_hi[4] = {0, 0, c2_limbs[0], c2_limbs[1]};

    // k, t and t_hi are all < n, so two modular subtracts suffice.
    std::uint64_t k1_part[4];
    std::uint64_t k1_raw_limbs[4];
    glv_sub_mod_n(k_arr.data(), t, k1_part);
    glv_sub_mod_n(k1_part, t_hi, k1_raw_limbs);
    Scalar const k1_mod = Scalar::from_limbs(
        {k1_raw_limbs[0], k1_raw_limbs[1], k1_raw_limbs[2], k1_raw_limbs[3]});

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
#else
    // ---- Fallback (no __int128): use Scalar arithmetic ----
    Scalar const c1 = Scalar::from_limbs(c1_limbs);
    Scalar const c2 = Scalar::from_limbs(c2_limbs);

    static const Scalar minus_b1 = Scalar::from_bytes(kMinusB1Bytes);
    static const Scalar minus_b2 = Scalar::from_bytes(kMinusB2Bytes);
    static const Scalar lambda   = Scalar::from_bytes(kGlvLambdaBytes);

    Scalar const k2_mod = (c1 * minus_b1) + (c2 * minus_b2);

    // ROUND3-1: half-order compare for sign selection
    auto const& k2fl = k2_mod.limbs();
    const std::uint64_t k2_raw_fb[4] = {k2fl[0], k2fl[1], k2fl[2], k2fl[3]};
    bool const k2_is_neg = !glv_ge_n(kNHalf, k2_raw_fb);
    std::uint64_t k2_abs_raw_fb[4];
    if (k2_is_neg) {
        glv_sub4(kN, k2_raw_fb, k2_abs_raw_fb);
    } else {
        k2_abs_raw_fb[0] = k2_raw_fb[0]; k2_abs_raw_fb[1] = k2_raw_fb[1];
        k2_abs_raw_fb[2] = k2_raw_fb[2]; k2_abs_raw_fb[3] = k2_raw_fb[3];
    }
    Scalar const k2_abs = Scalar::from_limbs(
        {k2_abs_raw_fb[0], k2_abs_raw_fb[1], k2_abs_raw_fb[2], k2_abs_raw_fb[3]});

    Scalar const k1_mod = k - lambda * k2_mod;
#endif

    // k1 sign handling (common path).
    // ROUND3-1: half-order compare on raw limbs, lazy negation.
    auto const& k1l = k1_mod.limbs();
    const std::uint64_t k1_raw[4] = {k1l[0], k1l[1], k1l[2], k1l[3]};
    bool const k1_is_neg = !glv_ge_n(kNHalf, k1_raw); // k1_raw > kNHalf
    std::uint64_t k1_abs_raw[4];
    if (k1_is_neg) {
        glv_sub4(kN, k1_raw, k1_abs_raw);
    } else {
        k1_abs_raw[0] = k1_raw[0]; k1_abs_raw[1] = k1_raw[1];
        k1_abs_raw[2] = k1_raw[2]; k1_abs_raw[3] = k1_raw[3];
    }
    Scalar const k1_abs = Scalar::from_limbs(
        {k1_abs_raw[0], k1_abs_raw[1], k1_abs_raw[2], k1_abs_raw[3]});

    result.k1     = k1_abs;
    result.k2     = k2_abs;
    result.k1_neg = k1_is_neg;
    result.k2_neg = k2_is_neg;

    return result;
}

Point apply_endomorphism(const Point& P) {
    if (P.is_infinity()) {
        return P;
    }
    
    // phi(x, y) = (beta*x, y) -- beta is a cube root of unity mod p
    // beta cached as static to avoid per-call from_bytes overhead
    static const FieldElement beta = FieldElement::from_bytes(glv_constants::BETA);

    return Point::from_jacobian_coords(P.x_raw() * beta, P.y_raw(), P.z_raw(), false);
}

bool verify_endomorphism(const Point& P) {
    if (P.is_infinity()) {
        return true;
    }
    
    // phi(phi(P)) + P should equal O (point at infinity)
    // Because phi^3 = identity, so phi^2 + phi + 1 = 0
    // Therefore: phi^2(P) = -P - phi(P)
    
    Point const phi_P = apply_endomorphism(P);
    Point const phi_phi_P = apply_endomorphism(phi_P);
    
    // phi^2(P) + P should equal -phi(P)
    Point const sum = phi_phi_P.add(P);
    Point const neg_phi_P = phi_P.negate();
    
    // Compare coordinates (normalize to affine first)
    auto sum_x = sum.x().to_bytes();
    auto sum_y = sum.y().to_bytes();
    auto neg_x = neg_phi_P.x().to_bytes();
    auto neg_y = neg_phi_P.y().to_bytes();
    
    return (sum_x == neg_x) && (sum_y == neg_y);
}

} // namespace secp256k1::fast
