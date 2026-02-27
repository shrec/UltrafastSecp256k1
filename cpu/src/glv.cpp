// GLV endomorphism implementation for secp256k1
// Correct decomposition following libsecp256k1 algorithm:
//   k = k1 + k2*lambda (mod n), where |k1|,|k2| ~= sqrtn

#include "secp256k1/glv.hpp"
#include "secp256k1/field.hpp"
#include <cstring>

namespace secp256k1::fast {

// ============================================================================
//  Internal helpers for GLV decomposition (32-bit safe, no __int128)
// ============================================================================

// Comba product-scanning: 8x8 -> 16 words (256x256 -> 512 bit)
// Same algorithm as esp32_mul_comba in field.cpp, duplicated here to avoid
// cross-TU dependency (glv.cpp must be self-contained).
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

// Convert uint64_t[4] LE limbs to uint32_t[8] LE limbs
static void limbs64_to_32(const std::uint64_t* src, std::uint32_t* dst) {
    for (int i = 0; i < 4; i++) {
        dst[static_cast<std::size_t>(i) * 2]     = (std::uint32_t)src[i];
        dst[static_cast<std::size_t>(i) * 2 + 1] = (std::uint32_t)(src[i] >> 32);
    }
}

// Compute (a * b) >> 384 with rounding bit (libsecp256k1 style)
// a, b: 256-bit values as LE uint64_t[4]
// Returns upper ~128 bits as LE uint64_t[4] (top two limbs typically 0)
static std::array<std::uint64_t, 4> mul_shift_384(
    const std::array<std::uint64_t, 4>& a,
    const std::array<std::uint64_t, 4>& b) {

    std::uint32_t a32[8], b32[8], prod[16];
    limbs64_to_32(a.data(), a32);
    limbs64_to_32(b.data(), b32);
    glv_mul_comba(a32, b32, prod);

    // Bits 384..511 sit in prod[12..15] (since 384/32 = 12)
    std::array<std::uint64_t, 4> result{};
    result[0] = (std::uint64_t)prod[12] | ((std::uint64_t)prod[13] << 32);
    result[1] = (std::uint64_t)prod[14] | ((std::uint64_t)prod[15] << 32);
    // result[2] = result[3] = 0

    // Rounding bit: bit 383 = MSB of prod[11]
    if (prod[11] >> 31) {
        result[0]++;
        if (result[0] == 0) result[1]++;
    }
    return result;
}

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
            // 32-bit CLZ for portability (ESP32 Xtensa has NSAU for 32-bit)
            auto const hi32 = static_cast<std::uint32_t>(limbs[i] >> 32);
            if (hi32) return static_cast<unsigned>(i * 64 + 64 - static_cast<unsigned>(__builtin_clz(hi32)));
            return static_cast<unsigned>(i * 64 + 32 - static_cast<unsigned>(__builtin_clz(static_cast<std::uint32_t>(limbs[i]))));
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
//  Public API
// ============================================================================

GLVDecomposition glv_decompose(const Scalar& k) {
    GLVDecomposition result;

    // Step 1: c1 = round(k * g1 / 2^384),  c2 = round(k * g2 / 2^384)
    auto k_limbs = k.limbs();
    auto c1_limbs = mul_shift_384({k_limbs[0], k_limbs[1], k_limbs[2], k_limbs[3]}, kG1);
    auto c2_limbs = mul_shift_384({k_limbs[0], k_limbs[1], k_limbs[2], k_limbs[3]}, kG2);

    Scalar const c1 = Scalar::from_limbs(c1_limbs);
    Scalar const c2 = Scalar::from_limbs(c2_limbs);

    // Step 2: k2 = c1*(-b1) + c2*(-b2)  (mod n)
    // Lazy-init constants (thread-safe in C++11+)
    static const Scalar minus_b1 = Scalar::from_bytes(kMinusB1Bytes);
    static const Scalar minus_b2 = Scalar::from_bytes(kMinusB2Bytes);
    static const Scalar lambda   = Scalar::from_bytes(kGlvLambdaBytes);

    Scalar const k2_mod = (c1 * minus_b1) + (c2 * minus_b2);

    // Step 3: pick shorter representation for k2
    Scalar const k2_neg = Scalar::zero() - k2_mod;
    bool const k2_is_neg = (scalar_bitlen(k2_neg) < scalar_bitlen(k2_mod));
    Scalar const k2_abs    = k2_is_neg ? k2_neg : k2_mod;
    Scalar const k2_signed = k2_is_neg ? (Scalar::zero() - k2_abs) : k2_abs;

    // Step 4: k1 = k - lambda*k2  (mod n)
    Scalar const k1_mod = k - lambda * k2_signed;

    // Step 5: pick shorter representation for k1
    Scalar const k1_neg = Scalar::zero() - k1_mod;
    bool const k1_is_neg = (scalar_bitlen(k1_neg) < scalar_bitlen(k1_mod));
    Scalar const k1_abs = k1_is_neg ? k1_neg : k1_mod;

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
