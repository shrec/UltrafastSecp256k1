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
// 64-bit Comba using __int128: 4x4 = 16 multiplications (vs 8x8 = 64 at 32-bit).
// Each 64x64->128 multiply maps to MUL + MULHU on x86-64, UMULH on AArch64.
// Carry chain uses libsecp256k1-style 192-bit accumulator (c0:c1:c2).
// Result: product[0..7] as 64-bit limbs (512 bits total).
static void glv_mul_comba_64(const std::uint64_t a[4], const std::uint64_t b[4],
                             std::uint64_t r[8]) {
    using u128 = unsigned __int128;
    std::uint64_t c0 = 0, c1 = 0;
    std::uint32_t c2 = 0;

    // muladd: add a[i]*b[j] into 192-bit accumulator (c2:c1:c0)
    #define GLV_MULADD(i, j) do { \
        const u128 p_ = (u128)(a[i]) * (b[j]); \
        const std::uint64_t tl_ = (std::uint64_t)p_; \
        std::uint64_t th_ = (std::uint64_t)(p_ >> 64); \
        c0 += tl_; \
        th_ += (c0 < tl_) ? 1ULL : 0ULL; \
        c1 += th_; \
        c2 += (c1 < th_) ? 1U : 0U; \
    } while(0)

    // extract: output c0 as result word, shift accumulator right by 64
    #define GLV_EXTRACT(out) do { \
        (out) = c0; \
        c0 = c1; \
        c1 = static_cast<std::uint64_t>(c2); \
        c2 = 0; \
    } while(0)

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

    #undef GLV_MULADD
    #undef GLV_EXTRACT
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

// Bit-length of a Scalar (for sign selection: pick shorter representation)
static unsigned scalar_bitlen(const Scalar& s) {
    auto& limbs = s.limbs();
    for (std::size_t i = 4; i-- > 0; ) {
        if (limbs[i] != 0) {
#if defined(_MSC_VER) && !defined(__clang__)
            unsigned long index;
            _BitScanReverse64(&index, limbs[i]);
            return static_cast<unsigned>(i * 64 + index + 1);
#else
#if defined(__XTENSA__) || defined(SECP256K1_ESP32) || defined(SECP256K1_PLATFORM_ESP32)
            // 32-bit CLZ for portability (ESP32 Xtensa has NSAU for 32-bit)
            auto const hi32 = static_cast<std::uint32_t>(limbs[i] >> 32);
            if (hi32) return static_cast<unsigned>(i * 64 + 64 - static_cast<unsigned>(__builtin_clz(hi32)));
            return static_cast<unsigned>(i * 64 + 32 - static_cast<unsigned>(__builtin_clz(static_cast<std::uint32_t>(limbs[i]))));
#else
            // x86-64/ARM64/RISC-V: single LZCNT/CLZ instruction
            // SAFETY: limbs[i] != 0 is guaranteed by the if-guard above.
            // __builtin_clzll(0) is UB per GCC docs; the guard prevents this.
            return static_cast<unsigned>(i * 64 + 64 - static_cast<unsigned>(__builtin_clzll(limbs[i])));
#endif
#endif
        }
    }
    return 0;
}

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

// ============================================================================
//  Fast GLV decomposition helpers (exploit known limb sizes)
// ============================================================================

#if defined(__SIZEOF_INT128__)
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif

// Group order n (little-endian 64-bit limbs)
static constexpr std::uint64_t kN[4] = {
    0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
};

// NC = 2^256 - n (129 bits, 3 limbs). Used for fast modular reduction.
static constexpr std::uint64_t kNC[3] = {
    0x402DA1732FC9BEBFULL, 0x4551231950B75FC4ULL, 0x0000000000000001ULL
};

// minus_b1 as raw limbs (128-bit: only 2 limbs non-zero)
static constexpr std::uint64_t kMB1[2] = {
    0x6F547FA90ABFE4C3ULL, 0xE4437ED6010E8828ULL
};

// minus_b2 as raw limbs (256-bit)
static constexpr std::uint64_t kMB2[4] = {
    0xD765CDA83DB1562CULL, 0x8A280AC50774346DULL,
    0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
};

// lambda as raw limbs (256-bit)
static constexpr std::uint64_t kLambdaLimbs[4] = {
    0xDF02967C1B23BD72ULL, 0x122E22EA20816678ULL,
    0xA5261C028812645AULL, 0x5363AD4CC05C30E0ULL
};

// 128-bit x 128-bit -> 256-bit Comba multiply (4 macs)
static void glv_mul_2x2(const std::uint64_t a[2], const std::uint64_t b[2],
                         std::uint64_t r[4]) {
    using u128 = unsigned __int128;
    std::uint64_t c0 = 0, c1 = 0;
    std::uint32_t c2 = 0;

    #define GLV_F_MA(i, j) do { \
        const u128 p_ = (u128)(a[i]) * (b[j]); \
        const std::uint64_t tl_ = (std::uint64_t)p_; \
        std::uint64_t th_ = (std::uint64_t)(p_ >> 64); \
        c0 += tl_; \
        th_ += (c0 < tl_) ? 1ULL : 0ULL; \
        c1 += th_; \
        c2 += (c1 < th_) ? 1U : 0U; \
    } while(0)

    #define GLV_F_EX(out) do { \
        (out) = c0; c0 = c1; c1 = static_cast<std::uint64_t>(c2); c2 = 0; \
    } while(0)

    GLV_F_MA(0, 0);
    GLV_F_EX(r[0]);
    GLV_F_MA(0, 1); GLV_F_MA(1, 0);
    GLV_F_EX(r[1]);
    GLV_F_MA(1, 1);
    GLV_F_EX(r[2]);
    r[3] = c0;

    #undef GLV_F_MA
    #undef GLV_F_EX
}

// 128-bit x 256-bit -> 384-bit Comba multiply (8 macs)
static void glv_mul_2x4(const std::uint64_t a[2], const std::uint64_t b[4],
                         std::uint64_t r[6]) {
    using u128 = unsigned __int128;
    std::uint64_t c0 = 0, c1 = 0;
    std::uint32_t c2 = 0;

    #define GLV_F_MA(i, j) do { \
        const u128 p_ = (u128)(a[i]) * (b[j]); \
        const std::uint64_t tl_ = (std::uint64_t)p_; \
        std::uint64_t th_ = (std::uint64_t)(p_ >> 64); \
        c0 += tl_; \
        th_ += (c0 < tl_) ? 1ULL : 0ULL; \
        c1 += th_; \
        c2 += (c1 < th_) ? 1U : 0U; \
    } while(0)

    #define GLV_F_EX(out) do { \
        (out) = c0; c0 = c1; c1 = static_cast<std::uint64_t>(c2); c2 = 0; \
    } while(0)

    // Column 0: a[0]*b[0]
    GLV_F_MA(0, 0);
    GLV_F_EX(r[0]);
    // Column 1: a[0]*b[1] + a[1]*b[0]
    GLV_F_MA(0, 1); GLV_F_MA(1, 0);
    GLV_F_EX(r[1]);
    // Column 2: a[0]*b[2] + a[1]*b[1]
    GLV_F_MA(0, 2); GLV_F_MA(1, 1);
    GLV_F_EX(r[2]);
    // Column 3: a[0]*b[3] + a[1]*b[2]
    GLV_F_MA(0, 3); GLV_F_MA(1, 2);
    GLV_F_EX(r[3]);
    // Column 4: a[1]*b[3]
    GLV_F_MA(1, 3);
    GLV_F_EX(r[4]);
    // Column 5: carry
    r[5] = c0;

    #undef GLV_F_MA
    #undef GLV_F_EX
}

// Compare 4-limb unsigned value: a >= b (little-endian)
static bool glv_ge_n(const std::uint64_t a[4], const std::uint64_t b[4]) {
    for (int i = 3; i >= 0; --i) {
        if (a[i] > b[i]) return true;
        if (a[i] < b[i]) return false;
    }
    return true; // equal
}

// Subtract n from a (4-limb). Returns borrow (0 or 1).
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

// Reduce a wide value (up to 7 limbs) modulo n.
// Uses 2^256 = NC (mod n) trick for fast reduction.
// Result written to out[4], guaranteed < n.
static void glv_reduce_mod_n(const std::uint64_t* w, int wlen,
                              std::uint64_t out[4]) {
    using u128 = unsigned __int128;

    std::uint64_t r[5] = {};

    if (wlen <= 4) {
        // Already fits in 256 bits -- just conditional subtract
        for (int i = 0; i < wlen; ++i) r[i] = w[i];
    } else {
        // Split: value = w_high * 2^256 + w_low
        // w_high = w[4..wlen-1] (up to 3 limbs for wlen<=7)
        // Compute w_high * NC (schoolbook, high_len x 3)
        int const high_len = wlen - 4;
        std::uint64_t hprod[7] = {};
        for (int i = 0; i < high_len; ++i) {
            u128 carry = 0;
            for (int j = 0; j < 3; ++j) {
                carry += (u128)w[4 + i] * kNC[j] + hprod[i + j];
                hprod[i + j] = (std::uint64_t)carry;
                carry >>= 64;
            }
            hprod[i + 3] = (std::uint64_t)carry;
        }

        // Add w_low (first 4 limbs) to hprod
        u128 carry = 0;
        for (int i = 0; i < 4; ++i) {
            carry += (u128)hprod[i] + w[i];
            r[i] = (std::uint64_t)carry;
            carry >>= 64;
        }
        // Propagate carry into upper part
        int const hprod_top = high_len + 2; // max index with data
        for (int i = 4; i <= hprod_top; ++i) {
            carry += hprod[i];
            if (i < 5) {
                r[i] = (std::uint64_t)carry;
            }
            carry >>= 64;
        }
        // If r[4] > 0, we may need one more NC round (rare)
        if (r[4] > 0) {
            // r[4] is small (at most ~4). Apply: r = r[4]*NC + r[0..3]
            std::uint64_t extra[4] = {};
            u128 ec = 0;
            for (int j = 0; j < 3; ++j) {
                ec += (u128)r[4] * kNC[j] + r[j];
                extra[j] = (std::uint64_t)ec;
                ec >>= 64;
            }
            extra[3] = r[3] + (std::uint64_t)ec;
            r[0] = extra[0]; r[1] = extra[1];
            r[2] = extra[2]; r[3] = extra[3];
            r[4] = 0;
        }
    }

    // Final: while r >= n, subtract n (at most 2 times for practical inputs)
    while (glv_ge_n(r, kN)) {
        glv_sub4(r, kN, r);
    }

    for (int i = 0; i < 4; ++i) out[i] = r[i];
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
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
    // ---- Fast path: exploit known limb sizes (3.4x fewer multiply-accumulates) ----
    // c1, c2 are at most 128-bit (only limbs[0..1] non-zero from mul_shift_384).
    // c1*mb1: 2x2 = 4 macs (vs 16 for full Scalar::operator* schoolbook)
    // c2*mb2: 2x4 = 8 macs
    // Reduction via NC-trick: ~6 macs (vs 20 for Barrett)
    // Total: ~32 macs vs ~108 for 3x full Scalar multiply+Barrett.

    // c2*mb2: 128-bit x 256-bit -> 384-bit (6 limbs)
    std::uint64_t wide[7] = {};
    glv_mul_2x4(c2_limbs.data(), kMB2, wide); // fills [0..5]

    // c1*mb1: 128-bit x 128-bit -> 256-bit (4 limbs), add to wide
    std::uint64_t p1[4];
    glv_mul_2x2(c1_limbs.data(), kMB1, p1);
    {
        using u128 = unsigned __int128;
        u128 carry = 0;
        for (int i = 0; i < 4; ++i) {
            carry += (u128)wide[i] + p1[i];
            wide[i] = (std::uint64_t)carry;
            carry >>= 64;
        }
        for (int i = 4; i < 7 && carry; ++i) {
            carry += wide[i];
            wide[i] = (std::uint64_t)carry;
            carry >>= 64;
        }
    }

    // Reduce wide (up to 7 limbs) mod n -> k2
    std::uint64_t k2_raw[4];
    glv_reduce_mod_n(wide, 7, k2_raw);
    Scalar const k2_mod = Scalar::from_limbs(
        {k2_raw[0], k2_raw[1], k2_raw[2], k2_raw[3]});

    // k2 sign handling: pick shorter representation
    Scalar const k2_neg_val = Scalar::zero() - k2_mod;
    bool const k2_is_neg = (scalar_bitlen(k2_neg_val) < scalar_bitlen(k2_mod));
    Scalar const k2_abs = k2_is_neg ? k2_neg_val : k2_mod;

    // k1 = k - lambda * k2_mod (mod n)
    // k2_signed === k2_mod (mod n) always, so use k2_mod directly.
    // Exploit: k2_abs is ~128-bit -> lambda*k2_abs is 2x4 multiply (8 macs).
    // Then adjust sign: lambda*k2_mod = k2_is_neg ? -lambda*k2_abs : lambda*k2_abs
    auto const& k2a = k2_abs.limbs();

    std::uint64_t lk2[6] = {};
    if (k2a[2] == 0 && k2a[3] == 0) {
        // Fast: 2x4 multiply (k2_abs is 128-bit, as expected by GLV)
        glv_mul_2x4(k2a.data(), kLambdaLimbs, lk2);
    } else {
        // Fallback: full 4x4 (unlikely edge case)
        std::uint64_t lk8[8];
        glv_mul_comba_64(k2a.data(), kLambdaLimbs, lk8);
        for (int i = 0; i < 6; ++i) lk2[i] = lk8[i];
    }

    std::uint64_t lk2_mod[4];
    glv_reduce_mod_n(lk2, 6, lk2_mod);
    Scalar const lambda_k2_abs = Scalar::from_limbs(
        {lk2_mod[0], lk2_mod[1], lk2_mod[2], lk2_mod[3]});

    // k1 = k - lambda*k2_mod
    //     = k2_is_neg ? (k + lambda*k2_abs) : (k - lambda*k2_abs)
    Scalar const k1_mod = k2_is_neg
        ? (k + lambda_k2_abs)
        : (k - lambda_k2_abs);

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

    Scalar const k2_neg_val = Scalar::zero() - k2_mod;
    bool const k2_is_neg = (scalar_bitlen(k2_neg_val) < scalar_bitlen(k2_mod));
    Scalar const k2_abs = k2_is_neg ? k2_neg_val : k2_mod;

    Scalar const k1_mod = k - lambda * k2_mod;
#endif

    // k1 sign handling (common path)
    Scalar const k1_neg_val = Scalar::zero() - k1_mod;
    bool const k1_is_neg = (scalar_bitlen(k1_neg_val) < scalar_bitlen(k1_mod));
    Scalar const k1_abs = k1_is_neg ? k1_neg_val : k1_mod;

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
