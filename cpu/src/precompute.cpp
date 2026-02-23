#include <cmath>  // For std::isfinite

#define SECP256K1_DEBUG_SPLIT 0
#define SECP256K1_DEBUG_GLV 0
#define SECP256K1_PROFILE_DECOMP 0  // Disable profiling for embedded
#define SECP256K1_PROFILE_PRECOMPUTED 0  // Profile scalar_mul_arbitrary_precomputed

// ESP32 platform: minimal includes and no exceptions
#if defined(SECP256K1_ESP32) || defined(SECP256K1_PLATFORM_ESP32) || defined(__XTENSA__)
    #define SECP256K1_ESP32_BUILD 1
    #define SECP256K1_THROW_ERROR(msg) do { /* no-op on ESP32 */ } while(0)
#else
    #define SECP256K1_ESP32_BUILD 0
    #define SECP256K1_THROW_ERROR(msg) throw std::runtime_error(msg)
#endif

/* GLV Optimization Progress:
 * 
 * Performance Evolution (@ 3.5 GHz, w=20 fixed-base):
 *   BEZ GLV:         26.6 us (93,222 cycles)  [OK] baseline
 *   GLV (current):   27.1 us (94,978 cycles)  [FAIL] 1.9% slower
 * 
 * Bottleneck Analysis:
 *   - Decomposition:  14.07 us (49,255 cycles) -- 52% of total time!
 *     - k2 calculation:   15,000 cycles (30%) -- Scalar arithmetic
 *     - lambdaxk2:        33,000 cycles (67%) -- 256x256 mul + Barrett
 *     - mul_scalar_raw:      ~80 cycles (0.2%)
 *     - barrett_reduce:   ~32,900 cycles (67%)
 *   - 2D Shamir:      ~13 us (~45k cycles) -- Actually faster than 1D!
 * 
 * Optimizations Applied:
 *   1. [OK] Optimized barrett_reduce_512: limb subtraction instead of Scalar ops
 *   2. [OK] Fast scalar_from_limbs_normalized: skip >= ORDER check  
 *   3. [OK] Detailed RDTSC profiling to find bottlenecks
 * 
 * Remaining Issues:
 *   - Decomposition overhead too high for w=20 (large window)
 *   - Scalar arithmetic (k2 calc) expensive: 15k cycles
 *   - Barrett reduction expensive: 33k cycles
 * 
 * Next Steps:
 *   Option 1: Reduce window size (w=16 or w=14) to amortize decomposition
 *   Option 2: Implement full limb-based decomposition (no Scalar class)
 *   Option 3: Assembly-optimized 256x256 multiplication
 *   Option 4: Accept that BEZ GLV is optimal for w=20 fixed-base
 */
#include "secp256k1/precompute.hpp"

#include "secp256k1/field.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cctype>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if !SECP256K1_ESP32_BUILD
// Desktop-only includes
#include <fstream>
#include <chrono>
#include <filesystem>
#include <unordered_map>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <mutex>
#include <thread>
#include <queue>
#include <condition_variable>
#include <sstream>
#endif

#if (SECP256K1_DEBUG_SPLIT || SECP256K1_DEBUG_GLV || SECP256K1_PROFILE_DECOMP) && !SECP256K1_ESP32_BUILD
#include <iostream>
#include <sstream>
#include <iomanip>
#endif

// RDTSC benchmark macro (inline, no external header needed)
#if defined(__x86_64__) || defined(_M_X64)
    #if defined(__GNUC__) || defined(__clang__)
    [[maybe_unused]] static inline uint64_t RDTSC() {
        uint32_t lo, hi;
        __asm__ volatile ("rdtsc" : "=a"(lo), "=d"(hi));
        return ((uint64_t)hi << 32) | lo;
    }
    #endif
#else
    // RISC-V and other platforms: return 0 (timing disabled)
    static inline uint64_t RDTSC() {
        return 0;
    }
#endif

// GCC/Clang intrinsics wrappers (not needed for MSVC/ClangCL)
#ifndef _MSC_VER

#if defined(SECP256K1_NO_INT128) || defined(SECP256K1_PLATFORM_ESP32)
// Portable 64x64->128 multiplication for 32-bit platforms
static inline uint64_t _umul128(uint64_t a, uint64_t b, uint64_t* hi) {
    uint32_t a_lo = (uint32_t)a;
    uint32_t a_hi = (uint32_t)(a >> 32);
    uint32_t b_lo = (uint32_t)b;
    uint32_t b_hi = (uint32_t)(b >> 32);

    uint64_t p0 = (uint64_t)a_lo * b_lo;
    uint64_t p1 = (uint64_t)a_lo * b_hi;
    uint64_t p2 = (uint64_t)a_hi * b_lo;
    uint64_t p3 = (uint64_t)a_hi * b_hi;

    uint64_t mid = p1 + (p0 >> 32);
    mid += p2;
    if (mid < p2) p3 += 0x100000000ULL; // carry

    *hi = p3 + (mid >> 32);
    return (mid << 32) | (uint32_t)p0;
}
#else
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
[[maybe_unused]] static inline uint64_t _umul128(uint64_t a, uint64_t b, uint64_t* hi) {
    unsigned __int128 r = (unsigned __int128)a * b;
    *hi = r >> 64;
    return (uint64_t)r;
}
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#endif

static inline unsigned char _BitScanReverse64(unsigned long* index, uint64_t mask) {
    if (mask == 0) return 0;
    *index = 63 - __builtin_clzll(mask);
    return 1;
}
#else
// MSVC
#include <intrin.h>
#define RDTSC() __rdtsc()
#endif

#include "platform_compat.h"
#include "secp256k1/glv.hpp"

namespace secp256k1::fast {

// Enable limb-only GLV decomposition path by default unless explicitly disabled
#ifndef SECP256K1_LIMB_GLV
#define SECP256K1_LIMB_GLV 0
#endif // Experimental limb-only GLV path (Task 16)
// Task 16 verdict: limb-only GLV decomposition increased cycles (~44K -> ~55K).
// Disabled by default; kept for future fused multiply+reduce prototype. Enable by
// defining SECP256K1_LIMB_GLV=1 and re-running benchmarks; acceptance requires
// >=15% improvement over Scalar-based path without test regressions.

// Global profiling counters for decomposition (exported for benchmarks)
uint64_t g_decomp_scalar_to_limbs_cycles = 0;
uint64_t g_decomp_mul_shift_cycles = 0;
uint64_t g_decomp_scalar_math_cycles = 0;
uint64_t g_decomp_barrett_reduce_cycles = 0;
uint64_t g_decomp_normalize_cycles = 0;

namespace {

struct AffinePointPacked {
    FieldElement x;
    FieldElement y;
    bool infinity{true};
};

// Jacobian point for internal calculations
struct JacobianPoint {
    FieldElement x;
    FieldElement y;
    FieldElement z;
    bool infinity{true};
};

[[maybe_unused]] inline FieldElement fe_from_uint_local(std::uint64_t v) {
    return FieldElement::from_uint64(v);
}

// Commonly used constants - cached for performance
namespace {
    const FieldElement FE_TWO = fe_from_uint_local(2);
    const FieldElement FE_THREE = fe_from_uint_local(3);
    const FieldElement FE_EIGHT = fe_from_uint_local(8);
}

// Forward declaration for negate_fe
FieldElement negate_fe(const FieldElement& v);

// Helper: Negate Jacobian point
[[nodiscard]] SECP256K1_INLINE JacobianPoint negate_jacobian(const JacobianPoint& p) {
    return {p.x, negate_fe(p.y), p.z, p.infinity};
}

// Helper: Convert affine to Jacobian
[[nodiscard]] SECP256K1_INLINE JacobianPoint affine_to_jacobian(const AffinePointPacked& p) {
    return {p.x, p.y, FieldElement::one(), p.infinity};
}

// Helper: Jacobian point doubling
[[nodiscard]] SECP256K1_INLINE JacobianPoint jacobian_double(const JacobianPoint& p) {
    if (SECP256K1_UNLIKELY(p.infinity)) return p;
    
    const FieldElement& three = FE_THREE;
    FieldElement yy = p.y;         // Copy for in-place
    yy.square_inplace();            // yy = y^2 in-place!
    FieldElement yyyy = yy;         // Copy for in-place
    yyyy.square_inplace();          // yyyy = y^4 in-place!
    FieldElement xx = p.x;          // Copy for in-place
    xx.square_inplace();            // xx = x^2 in-place!
    FieldElement temp = p.x + yy;   // x + y^2
    temp.square_inplace();          // (x + y^2)^2 in-place!
    FieldElement s = temp - xx - yyyy;
    s += s;                         // s = 2*((x+y^2)^2 - x^2 - y^4) via add
    FieldElement m = xx * three;
    FieldElement x3 = m;            // Copy for in-place
    x3.square_inplace();            // m^2 in-place!
    FieldElement s2 = s + s;        // 2*s via add
    x3 -= s2;
    FieldElement yyyy2 = yyyy + yyyy;   // 2*yyyy
    FieldElement yyyy4 = yyyy2 + yyyy2; // 4*yyyy
    FieldElement yyyy8 = yyyy4 + yyyy4; // 8*yyyy via additions
    FieldElement y3 = m * (s - x3) - yyyy8;
    FieldElement z3 = p.y * p.z;
    z3 += z3;                       // 2*(y*z) via add
    return {x3, y3, z3, false};
}

// Helper: Jacobian + Jacobian addition
[[nodiscard]] SECP256K1_INLINE JacobianPoint jacobian_add(const JacobianPoint& p, const JacobianPoint& q) {
    if (SECP256K1_UNLIKELY(p.infinity)) return q;
    if (SECP256K1_UNLIKELY(q.infinity)) return p;
    
    FieldElement z1z1 = p.z;        // Copy for in-place
    z1z1.square_inplace();          // z1^2 in-place!
    FieldElement z2z2 = q.z;        // Copy for in-place
    z2z2.square_inplace();          // z2^2 in-place!
    FieldElement u1 = p.x * z2z2;
    FieldElement u2 = q.x * z1z1;
    FieldElement s1 = p.y * q.z * z2z2;
    FieldElement s2 = q.y * p.z * z1z1;
    
    if (SECP256K1_UNLIKELY(u1 == u2)) {
        if (s1 == s2) {
            return jacobian_double(p);
        }
        return {FieldElement::zero(), FieldElement::one(), FieldElement::zero(), true}; // Infinity
    }
    
    FieldElement h = u2 - u1;
    FieldElement i = h + h;         // 2*h via add
    i.square_inplace();             // i = (2*h)^2 in-place!
    FieldElement j = h * i;
    FieldElement r = s2 - s1;
    r += r;                         // 2*(s2-s1) via add
    FieldElement v = u1 * i;
    FieldElement x3 = r;            // Copy for in-place
    x3.square_inplace();            // r^2 in-place!
    x3 -= j + v + v;               // x3 = r^2 - j - 2*v via add
    FieldElement s1j = s1 * j;
    FieldElement y3 = r * (v - x3) - (s1j + s1j); // 2*s1*j via add
    FieldElement temp_z = p.z + q.z; // z1 + z2
    temp_z.square_inplace();        // (z1 + z2)^2 in-place!
    FieldElement z3 = (temp_z - z1z1 - z2z2) * h;
    
    return {x3, y3, z3, false};
}

// Montgomery's Batch Inversion: Compute multiple inverses with only ONE expensive inverse
// Instead of N inverse() calls (~8 us each), we do: 1 inverse + 3N multiplications (~5 ns each)
// Example (N=16): 16 x 8 us = 128 us -> 1 x 8 us + 48 x 5 ns ~= 8.2 us (15.6x faster!)
//
// Algorithm:
//   prod[0] = z[0]
//   prod[1] = z[0] * z[1]
//   ...
//   prod[n-1] = z[0] * z[1] * ... * z[n-1]
//   
//   inv_all = (z[0] * z[1] * ... * z[n-1])^-1     // ONLY ONE INVERSE!
//   
//   z[n-1]^-1 = inv_all * prod[n-2]
//   z[n-2]^-1 = (inv_all * z[n-1]) * prod[n-3]
//   ...
SECP256K1_INLINE void batch_inverse(std::vector<FieldElement>& inputs) {
    const size_t n = inputs.size();
    if (n == 0) return;
    if (n == 1) {
        inputs[0] = inputs[0].inverse();
        return;
    }
    
    // Step 1: Compute products: prod[i] = inputs[0] x inputs[1] x ... x inputs[i]
    std::vector<FieldElement> products;
    products.reserve(n);
    products.push_back(inputs[0]);
    
    for (size_t i = 1; i < n; ++i) {
        products.push_back(products.back() * inputs[i]);
    }
    
    // Step 2: Compute inverse of final product (ONLY ONE INVERSE!)
    FieldElement inv = products.back().inverse();
    
    // Step 3: Back-substitute to get individual inverses
    // CRITICAL: Must use ORIGINAL value before overwriting!
    for (size_t i = n - 1; i > 0; --i) {
        FieldElement original = inputs[i];           // Save original BEFORE overwriting
        inputs[i] = inv * products[i - 1];           // inputs[i]^-^1 = inv x prod[i-1]
        inv = inv * original;                        // Update inv using ORIGINAL value
    }
    inputs[0] = inv;  // First element
}

// Mixed Jacobian-Affine addition: P (Jacobian) + Q (Affine) -> Result (Jacobian)
// 8 multiplications instead of 12
[[nodiscard]] SECP256K1_INLINE JacobianPoint jacobian_add_mixed_local(const JacobianPoint& p, const AffinePointPacked& q) {
    if (SECP256K1_UNLIKELY(p.infinity)) {
        return {q.x, q.y, FieldElement::one(), q.infinity};
    }
    if (SECP256K1_UNLIKELY(q.infinity)) {
        return p;
    }

    FieldElement z1z1 = p.z;        // Copy for in-place
    z1z1.square_inplace();          // z1^2 in-place!
    FieldElement u2 = q.x * z1z1;
    FieldElement s2 = q.y * p.z * z1z1;

    if (SECP256K1_UNLIKELY(p.x == u2)) {
        if (p.y == s2) {
            // Point doubling (rare in precompute)
            const FieldElement& two = FE_TWO;
            const FieldElement& three = FE_THREE;
            const FieldElement& eight = FE_EIGHT;
            FieldElement yy = p.y;         // Copy for in-place
            yy.square_inplace();            // y^2 in-place!
            FieldElement yyyy = yy;         // Copy for in-place
            yyyy.square_inplace();          // y^4 in-place!
            FieldElement xx = p.x;          // Copy for in-place
            xx.square_inplace();            // x^2 in-place!
            FieldElement temp = p.x + yy;   // x + y^2
            temp.square_inplace();          // (x + y^2)^2 in-place!
            FieldElement s = (temp - xx - yyyy) * two;
            FieldElement m = xx * three;
            FieldElement x3 = m;            // Copy for in-place
            x3.square_inplace();            // m^2 in-place!
            x3 -= s * two;
            FieldElement y3 = m * (s - x3) - yyyy * eight;
            FieldElement z3 = (p.y * p.z) * two;
            return {x3, y3, z3, false};
        }
        return {FieldElement::zero(), FieldElement::one(), FieldElement::zero(), true};
    }

    FieldElement h = u2 - p.x;
    FieldElement hh = h;            // Copy for in-place
    hh.square_inplace();            // h^2 in-place!
    FieldElement i = hh + hh + hh + hh; // 4 * hh
    FieldElement j = h * i;
    FieldElement r = (s2 - p.y) + (s2 - p.y);
    FieldElement v = p.x * i;
    const FieldElement& two = FE_TWO;

    FieldElement x3 = r;            // Copy for in-place
    x3.square_inplace();            // r^2 in-place!
    x3 -= j + v + v;                // x3 = r^2 - j - 2*v
    FieldElement y3 = r * (v - x3) - (p.y * j * two);
    FieldElement z3 = p.z + h;      // p.z + h
    z3.square_inplace();            // (p.z + h)^2 in-place!
    z3 -= z1z1 + hh;               // z3 = (p.z + h)^2 - z1z1 - hh

    return {x3, y3, z3, false};
}

struct PrecomputeContext {
    FixedBaseConfig config{};
    unsigned window_bits{0};
    std::size_t window_count{0};
    std::size_t digit_count{0};
    FieldElement beta;
    std::vector<std::vector<AffinePointPacked>> base_tables;
    std::vector<std::vector<AffinePointPacked>> psi_tables;
};

using Limbs4 = std::array<std::uint64_t, 4>;
using Limbs6 = std::array<std::uint64_t, 6>;
using Limbs3 = std::array<std::uint64_t, 3>;
using Limbs2 = std::array<std::uint64_t, 2>;

struct UInt128 {
    std::uint64_t lo;
    std::uint64_t hi;
};

// Forward declare 64x64->128 multiply helper used by division code
static void mul64x64(std::uint64_t a, std::uint64_t b, std::uint64_t& lo, std::uint64_t& hi);

[[nodiscard]] UInt128 make_uint128(std::uint64_t lo, std::uint64_t hi) {
    return UInt128{lo, hi};
}

[[nodiscard]] UInt128 multiply_u64(std::uint64_t a, std::uint64_t b) {
#if defined(_MSC_VER) && !defined(__clang__)
    unsigned __int64 hi = 0ULL;
    const unsigned __int64 lo = _umul128(a, b, &hi);
    return make_uint128(lo, hi);
#elif defined(SECP256K1_NO_INT128) || defined(SECP256K1_PLATFORM_ESP32)
    // Portable 64x64->128 for 32-bit platforms
    uint64_t hi = 0;
    uint64_t lo = _umul128(a, b, &hi);
    return make_uint128(lo, hi);
#else
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
    const unsigned __int128 product = static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b);
    return make_uint128(static_cast<std::uint64_t>(product), static_cast<std::uint64_t>(product >> 64));
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#endif
}

[[nodiscard]] UInt128 add_uint64(UInt128 value, std::uint64_t addend) {
    const std::uint64_t sum = value.lo + addend;
    const std::uint64_t carry = (sum < value.lo) ? 1ULL : 0ULL;
    value.lo = sum;
    value.hi += carry;
    return value;
}

constexpr std::array<std::uint8_t, 32> kBetaBytes{
    0x7a,0xe9,0x6a,0x2b,0x65,0x7c,0x07,0x10,
    0x6e,0x64,0x47,0x9e,0xac,0x34,0x34,0xe9,
    0x9c,0xf0,0x49,0x75,0x12,0xf5,0x89,0x95,
    0xc1,0x39,0x6c,0x28,0x71,0x95,0x01,0xee
};

// g1/g2 constants (little-endian 64-bit limbs) matching libsecp256k1 ordering
constexpr std::array<std::uint64_t, 4> kG1MulShift{
    0xE893209A45DBB031ULL,
    0x3DAA8A1471E8CA7FULL,
    0xE86C90E49284EB15ULL,
    0x3086D221A7D46BCDULL
};

constexpr std::array<std::uint64_t, 4> kG2MulShift{
    0x1571B4AE8AC47F71ULL,
    0x221208AC9DF506C6ULL,
    0x6F547FA90ABFE4C4ULL,
    0xE4437ED6010E8828ULL
};

// Alternative precise rounding for c1/c2 (standard rounding) used for diagnostics
// Declared here; defined after helper functions.
template <std::size_t N>
Limbs4 mul_shift_round(const Limbs4& value, const std::array<std::uint64_t, N>& constant, unsigned shift);

constexpr std::array<std::uint8_t, 32> kMinusB1Bytes{
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0xE4,0x43,0x7E,0xD6,0x01,0x0E,0x88,0x28,
    0x6F,0x54,0x7F,0xA9,0x0A,0xBF,0xE4,0xC3
};

constexpr std::array<std::uint8_t, 32> kMinusB2Bytes{
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
    0x8A,0x28,0x0A,0xC5,0x07,0x74,0x34,0x6D,
    0xD7,0x65,0xCD,0xA8,0x3D,0xB1,0x56,0x2C
};

constexpr std::array<std::uint8_t, 32> kLambdaBytes{
    0x53,0x63,0xAD,0x4C,0xC0,0x5C,0x30,0xE0,
    0xA5,0x26,0x1C,0x02,0x88,0x12,0x64,0x5A,
    0x12,0x2E,0x22,0xEA,0x20,0x81,0x66,0x78,
    0xDF,0x02,0x96,0x7C,0x1B,0x23,0xBD,0x72
};

// Lattice basis constants (from libsecp256k1 commentary)
// a1 = 0x3086D221A7D46BCDE86C90E49284EB15 (128-bit)
constexpr std::array<std::uint8_t, 32> kA1Bytes{
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x30,0x86,0xD2,0x21,0xA7,0xD4,0x6B,0xCD,
    0xE8,0x6C,0x90,0xE4,0x92,0x84,0xEB,0x15
};

// a2 = 0x0114CA50F7A8E2F3F657C1108D9D44CFD8 (129-bit)
constexpr std::array<std::uint8_t, 32> kA2Bytes{
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x01,0x14,0xCA,0x50,0xF7,0xA8,0xE2,0xF3,
    0xF6,0x57,0xC1,0x10,0x8D,0x9D,0x44,0xCF,0xD8
};

// Helper: convert 32-byte big-endian two's-complement negative value to positive magnitude bytes
[[maybe_unused]] static std::array<std::uint8_t,32> twos_complement_negate(const std::array<std::uint8_t,32>& neg_be) {
    std::array<std::uint8_t,32> out;
    // Invert
    for (std::size_t i=0;i<32;++i) out[i] = static_cast<std::uint8_t>(~neg_be[i]);
    // Add 1
    for (int i=31;i>=0;--i) {
        unsigned int sum = static_cast<unsigned int>(out[i]) + 1U;
        out[i] = static_cast<std::uint8_t>(sum & 0xFFu);
        if ((sum & 0x100u) == 0) break; // no further carry
    }
    return out;
}

constexpr Limbs4 kOrderHalf{{
    0xdfe92f46681b20a0ULL,
    0x5d576e7357a4501dULL,
    0xffffffffffffffffULL,
    0x7fffffffffffffffULL
}};

// Group order n (little-endian limbs, same as Scalar::ORDER)
constexpr Limbs4 kGroupOrder{{
    0xBFD25E8CD0364141ULL,
    0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL,
    0xFFFFFFFFFFFFFFFFULL
}};

#if !SECP256K1_ESP32_BUILD
std::mutex g_mutex;
#endif
FixedBaseConfig g_config{};
std::unique_ptr<PrecomputeContext> g_context;

Scalar make_scalar(const std::array<std::uint8_t, 32>& bytes) {
    return Scalar::from_bytes(bytes);
}

FieldElement negate_fe(const FieldElement& v) {
    return FieldElement::zero() - v;
}

AffinePointPacked to_affine(const Point& point) {
    if (point.is_infinity()) {
        return {FieldElement::zero(), FieldElement::one(), true};
    }
    return {point.x(), point.y(), false};
}

void right_shift(std::array<std::uint64_t, 5>& value, unsigned shift) {
    if (shift == 0) {
        return;
    }
    const unsigned word_shift = shift / 64U;
    const unsigned bit_shift = shift % 64U;
    if (word_shift > 0) {
        for (std::size_t i = 0; i + word_shift < value.size(); ++i) {
            value[i] = value[i + word_shift];
        }
        for (std::size_t i = value.size() - word_shift; i < value.size(); ++i) {
            value[i] = 0ULL;
        }
    }
    if (bit_shift == 0) {
        return;
    }
    std::uint64_t carry = 0ULL;
    for (std::size_t idx = value.size(); idx-- > 0;) {
        const std::uint64_t next_carry = value[idx] << (64U - bit_shift);
        value[idx] = (value[idx] >> bit_shift) | carry;
        carry = next_carry;
    }
}

void increment(std::array<std::uint64_t, 5>& value) {
    std::uint64_t carry = 1ULL;
    for (std::size_t i = 0; i < value.size(); ++i) {
        const std::uint64_t prev = value[i];
        value[i] += carry;
        carry = (value[i] < prev) ? 1ULL : 0ULL;
        if (carry == 0ULL) {
            break;
        }
    }
}

Limbs4 scalar_to_limbs(const Scalar& scalar) {
    return scalar.limbs();
}

Scalar scalar_from_limbs(const Limbs4& limbs) {
    return Scalar::from_limbs(limbs);
}

// Fast version of scalar_from_limbs when we KNOW limbs < n (already normalized)
// Skips the expensive >= ORDER check and subtraction
static inline Scalar scalar_from_limbs_normalized(const Limbs4& limbs) {
#if SECP256K1_PROFILE_DECOMP
    unsigned long long start = RDTSC();
#endif
    
    // Create zero scalar and directly set its limbs
    // This avoids the modular reduction check in from_limbs()
    Scalar s = Scalar::zero();
    
#if SECP256K1_PROFILE_DECOMP
    unsigned long long t1 = RDTSC();
#endif
    
    // HACK: Access private limbs_ member through const_cast
    // This is safe because we KNOW limbs < ORDER
    const_cast<Scalar::limbs_type&>(s.limbs()) = limbs;
    
#if SECP256K1_PROFILE_DECOMP
    unsigned long long end = RDTSC();
    static unsigned long long total_zero = 0;
    static unsigned long long total_assign = 0;
    static int norm_calls = 0;
    
    total_zero += (t1 - start);
    total_assign += (end - t1);
    norm_calls++;
    
    if (norm_calls == 1000 || norm_calls == 3000) {
        std::printf("  [SCALAR_FROM_LIMBS_NORM] After %d calls:\n", norm_calls);
        std::printf("    Scalar::zero():    %llu cycles\n", total_zero / norm_calls);
        std::printf("    Assign limbs:      %llu cycles\n", total_assign / norm_calls);
        std::printf("    TOTAL AVG:         %llu cycles\n", (total_zero + total_assign) / norm_calls);
    }
#endif
    
    return s;
}

template <std::size_t N>
std::array<std::uint64_t, 4 + N> multiply_limbs(const Limbs4& a, const std::array<std::uint64_t, N>& b) {
    std::array<std::uint64_t, 4 + N> out{};
    for (std::size_t i = 0; i < a.size(); ++i) {
        std::uint64_t carry = 0ULL;
        for (std::size_t j = 0; j < b.size(); ++j) {
            const std::size_t idx = i + j;
            if (idx >= out.size()) {
                break;
            }

            UInt128 acc = multiply_u64(a[i], b[j]);
            acc = add_uint64(acc, out[idx]);
            acc = add_uint64(acc, carry);

            out[idx] = acc.lo;
            carry = acc.hi;
        }

        std::size_t idx = i + b.size();
        while (carry != 0ULL && idx < out.size()) {
            const std::uint64_t prev = out[idx];
            out[idx] += carry;
            carry = (out[idx] < prev) ? 1ULL : 0ULL;
            ++idx;
        }
    }
    return out;
}

template <std::size_t N>
Limbs4 shift_right(const std::array<std::uint64_t, N>& value, unsigned shift) {
    Limbs4 out{};
    const unsigned word_shift = shift / 64U;
    const unsigned bit_shift = shift % 64U;
    for (std::size_t i = 0; i < out.size(); ++i) {
        const std::size_t idx = i + word_shift;
        if (idx >= value.size()) {
            out[i] = 0ULL;
            continue;
        }
        std::uint64_t low = value[idx] >> bit_shift;
        if (bit_shift != 0U && idx + 1 < value.size()) {
            const std::uint64_t high = value[idx + 1] << (64U - bit_shift);
            low |= high;
        }
        out[i] = low;
    }
    return out;
}

template <std::size_t N>
Limbs4 mul_shift(const Limbs4& value, const std::array<std::uint64_t, N>& constant, unsigned shift) {
    // Wide 512-bit product (little-endian limbs)
    auto wide = multiply_limbs(value, constant);
    // Extract the (value * constant) >> shift as little-endian limbs
    Limbs4 out = shift_right(wide, shift);
    // libsecp256k1 rounding: AFTER shifting, add the bit at position (shift-1) to LSB of result.
    // This is floor(x/2^s) + bit(s-1) rather than floor((x + 2^(s-1))/2^s). They intentionally
    // ignore carries from lower discarded bits apart from the single rounding bit.
    if (shift > 0U) {
        const std::size_t limb_index = (static_cast<std::size_t>(shift) - 1U) / 64U;
        const unsigned bit_index = (static_cast<unsigned>(shift - 1U)) % 64U;
        if (limb_index < wide.size()) {
            const std::uint64_t rounding_bit = (wide[limb_index] >> bit_index) & 1ULL;
            #if SECP256K1_DEBUG_SPLIT
            // Print rounding bit for shift==384 and N==4 (g1/g2 paths)
            if (N == 4 && shift == 384U) {
                std::cout << "[DEBUG] mul_shift rounding_bit=" << rounding_bit
                          << " limb_index=" << limb_index << " bit_index=" << bit_index << "\n";
            }
            #endif
            if (rounding_bit) {
                // Add 1 to out (LSB) with carry propagation.
                std::uint64_t prev = out[0];
                out[0] += 1ULL;
                bool carry = (out[0] < prev);
                for (std::size_t i = 1; carry && i < out.size(); ++i) {
                    std::uint64_t p = out[i];
                    out[i] += 1ULL;
                    carry = (out[i] < p);
                }
            }
        }
    }
    return out;
}

// Define mul_shift_round after helpers are available
template <std::size_t N>
Limbs4 mul_shift_round(const Limbs4& value, const std::array<std::uint64_t, N>& constant, unsigned shift) {
    auto wide = multiply_limbs(value, constant); // little-endian
    // Add 2^(shift-1) for rounding (if shift>0)
    if (shift > 0) {
        unsigned add_bit = shift - 1;
        std::size_t limb_index = add_bit / 64U;
        unsigned bit_index = add_bit % 64U;
        if (limb_index < wide.size()) {
            std::uint64_t mask = 1ULL << bit_index;
            std::uint64_t prev = wide[limb_index];
            wide[limb_index] += mask;
            // propagate carry
            if (wide[limb_index] < prev) {
                for (std::size_t i = limb_index + 1; i < wide.size(); ++i) {
                    wide[i] += 1ULL;
                    if (wide[i] != 0ULL) break;
                }
            }
        }
    }
    return shift_right(wide, shift);
}

// Generic left shift across an array of limbs (little-endian), by 'shift' bits
template <std::size_t N>
[[maybe_unused]] static std::array<std::uint64_t, N> shl_limbs(const std::array<std::uint64_t, N>& in, unsigned shift) {
    if (shift == 0) return in;
    const unsigned word = shift / 64U;
    const unsigned bits = shift % 64U;
    std::array<std::uint64_t, N> out{};
    for (std::size_t i = N; i-- > 0;) {
        std::uint64_t val = 0;
        if (i >= word) {
            val = in[i - word] << bits;
            if (bits != 0 && i >= word + 1) {
                val |= (in[i - word - 1] >> (64U - bits));
            }
        }
        out[i] = val;
    }
    return out;
}

// Count leading zeros in 64-bit
[[maybe_unused]] static unsigned clz64_local(std::uint64_t x) {
    if (x == 0) return 64;
#if defined(_MSC_VER) && !defined(__clang__)
    unsigned long idx;
    _BitScanReverse64(&idx, x);
    return 63U - static_cast<unsigned>(idx);
#else
    return static_cast<unsigned>(__builtin_clzll(x));
#endif
}

// 384/256 -> 3-limb quotient (Knuth division D simplified)
// 384/256 division returning 192-bit quotient (3 limbs) using portable 64-bit math and MSVC intrinsics
[[maybe_unused]] static Limbs3 div_384_by_256(const Limbs6& num, const Limbs4& den) {
    // Normalize denominator
    unsigned s = clz64_local(den[3]);
    Limbs4 v = shl_limbs(den, s);
    // Shift numerator left by s (extend to 7 limbs for Knuth algorithm)
    auto shifted = shl_limbs(num, s);
    std::array<std::uint64_t,7> u{};
    for (std::size_t i=0;i<6;++i) u[i] = shifted[i];
    u[6] = 0ULL;

    Limbs3 q{};
    // Iterate j = 2..0 (since n=6, m=4 => m-n = 2)
    for (int j=2; j>=0; --j) {
        // Approximate qhat from top 128 bits
        std::uint64_t u_hi = u[j+4];
        std::uint64_t u_lo = u[j+3];
        std::uint64_t v_hi = v[3];
        std::uint64_t rhat;
#if defined(_MSC_VER) && !defined(__clang__)
        std::uint64_t qhat = _udiv128(u_hi, u_lo, v_hi, &rhat);
#elif defined(SECP256K1_NO_INT128) || defined(SECP256K1_PLATFORM_ESP32)
        // Portable division approximation for 32-bit platforms
        // This is an approximation - for exact division we do iterative refinement
        std::uint64_t qhat;
        if (u_hi >= v_hi) {
            qhat = 0xFFFFFFFFFFFFFFFFULL;
        } else if (u_hi == 0) {
            qhat = u_lo / v_hi;
        } else {
            // Approximate: divide (u_hi * 2^32 + u_lo_high) by v_hi
            uint64_t u_approx = (u_hi << 32) | (u_lo >> 32);
            qhat = u_approx / (v_hi >> 32);
            if (qhat > 0xFFFFFFFFFFFFFFFFULL) qhat = 0xFFFFFFFFFFFFFFFFULL;
        }
        // Compute remainder approximation
        uint64_t qv_hi, qv_lo;
        mul64x64(qhat, v_hi, qv_lo, qv_hi);
        // Adjust qhat down if needed
        while (qv_hi > u_hi || (qv_hi == u_hi && qv_lo > u_lo)) {
            --qhat;
            mul64x64(qhat, v_hi, qv_lo, qv_hi);
        }
        // rhat = u - qhat * v_hi (approximate)
        rhat = u_lo - qv_lo;
        if (qv_lo > u_lo) rhat = 0; // underflow protection
#else
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
        unsigned __int128 dividend = (static_cast<unsigned __int128>(u_hi) << 64) | u_lo;
        std::uint64_t qhat = static_cast<std::uint64_t>(dividend / v_hi);
        rhat = static_cast<std::uint64_t>(dividend % v_hi);
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#endif
        // Adjust qhat if qhat*v[2] > rhat*B + u[j+2]
        while (true) {
            // Compute lhs = qhat * v[2], rhs = rhat*B + u[j+2]
            std::uint64_t lhs_lo, lhs_hi; mul64x64(qhat, v[2], lhs_lo, lhs_hi);
            // rhs_hi = rhat, rhs_lo = u[j+2]
            bool greater = false;
            if (lhs_hi > rhat) greater = true; else if (lhs_hi == rhat && lhs_lo > u[j+2]) greater = true;
            if (!greater) break;
            // Decrement qhat and adjust rhat
            --qhat;
            std::uint64_t new_rhat = rhat + v_hi;
            if (new_rhat < rhat) { // overflow => break
                rhat = new_rhat; break;
            }
            rhat = new_rhat;
        }
        // Subtract qhat * v from u segment (u[j..j+4]) using overlapping subtracts
        unsigned char b = 0;
        for (int i=0;i<4;++i) {
            std::uint64_t p_lo, p_hi; mul64x64(qhat, v[i], p_lo, p_hi);
            b = COMPAT_SUBBORROW_U64(b, u[j+i], p_lo, &u[j+i]);
            b = COMPAT_SUBBORROW_U64(b, u[j+i+1], p_hi, &u[j+i+1]);
        }
        if (b) {
            // qhat too large: add back v and decrement qhat
            --qhat;
            unsigned char c = 0;
            for (int i=0;i<4;++i) {
                c = COMPAT_ADDCARRY_U64(c, u[j+i], v[i], &u[j+i]);
                c = COMPAT_ADDCARRY_U64(c, u[j+i+1], 0ULL, &u[j+i+1]);
            }
        }
        q[j] = qhat;
    }
    return q;
}

// Compute mu = floor((2^256 * mag) / n), where mag is up to 256 bits (here ~128 bits)
[[maybe_unused]] static Limbs2 compute_mu_from_mag(const Limbs4& mag) {
    // Build 6-limb numerator: mag << 256
    Limbs6 num{};
    num[4] = mag[0];
    num[5] = mag[1];
    num[2] = 0; num[3] = 0; num[0]=0; num[1]=0;
    Limbs3 q = div_384_by_256(num, kGroupOrder);
    // Expect top limb zero; take low 2 limbs as mu
    Limbs2 mu{{q[0], q[1]}};
    return mu;
}

[[maybe_unused]] bool scalar_is_high(const Scalar& scalar) {
    const auto& limbs = scalar.limbs();
    for (std::size_t idx = limbs.size(); idx-- > 0;) {
        if (limbs[idx] > kOrderHalf[idx]) {
            return true;
        }
        if (limbs[idx] < kOrderHalf[idx]) {
            return false;
        }
    }
    return false;
}

[[maybe_unused]] Scalar mul_scalar(const Scalar& lhs, const Scalar& rhs) {
    Scalar result = Scalar::zero();
    Scalar base = lhs;
    for (std::size_t bit = 0; bit < 256; ++bit) {
        if (rhs.bit(bit)) {
            result += base;
        }
        base += base;
    }
    return result;
}

// Multiply scalar 'a' by a 128-bit unsigned integer (hi:lo) modulo n.
[[maybe_unused]] static Scalar mul_scalar_u128(const Scalar& a, std::uint64_t u_lo, std::uint64_t u_hi) {
    Scalar result = Scalar::zero();
    Scalar base = a;
    // Process 128 bits: lower 64 then upper 64
    for (std::size_t bit = 0; bit < 64; ++bit) {
        if ((u_lo >> bit) & 1ULL) {
            result += base;
        }
        base += base;
    }
    for (std::size_t bit = 0; bit < 64; ++bit) {
        if ((u_hi >> bit) & 1ULL) {
            result += base;
        }
        base += base;
    }
    return result;
}

// Phase 5.7: Modified window digit extraction for wNAF-compatible tables
// We still extract standard signed windows, but tables store only odd multiples
std::vector<int32_t> compute_window_digits(const Scalar& scalar, unsigned window_bits, std::size_t window_count) {
    if (window_bits == 0U || window_bits > 30U) {
        #if SECP256K1_ESP32_BUILD
        return std::vector<int32_t>(); // Return empty on error
        #else
        throw std::runtime_error("Unsupported window size for digit extraction");
        #endif
    }
    const std::uint32_t mask = (1U << window_bits) - 1U;
    std::array<std::uint64_t, 5> working{};
    const auto& limbs = scalar.limbs();
    for (std::size_t i = 0; i < limbs.size(); ++i) {
        working[i] = limbs[i];
    }
    std::vector<int32_t> digits(window_count, 0);
    for (std::size_t idx = 0; idx < window_count; ++idx) {
        std::uint32_t chunk = static_cast<std::uint32_t>(working[0] & mask);
        int32_t digit = static_cast<int32_t>(chunk);
        right_shift(working, window_bits);
        const int32_t threshold = 1 << (window_bits - 1);
        if (digit >= threshold) {
            digit -= (1 << window_bits);
            increment(working);
        }
        digits[idx] = digit;
    }
    return digits;
}

void fill_tables_for_window(const Point& base_point,
                            std::size_t digit_count,
                            FieldElement beta,
                            std::vector<AffinePointPacked>& base_table,
                            std::vector<AffinePointPacked>* psi_table) {
    base_table.resize(digit_count);
    if (digit_count == 0) {
        return;
    }
    base_table[0] = {FieldElement::zero(), FieldElement::one(), true};

    if (digit_count == 1) {
        if (psi_table != nullptr) {
            psi_table->resize(1);
            (*psi_table)[0] = base_table[0];
        }
        return;
    }

    Point current = base_point;
    base_table[1] = to_affine(current);
    for (std::size_t digit = 2; digit < digit_count; ++digit) {
        current.add_inplace(base_point);
        base_table[digit] = to_affine(current);
    }

    if (psi_table != nullptr) {
        psi_table->resize(digit_count);
        (*psi_table)[0] = base_table[0];
        for (std::size_t digit = 1; digit < digit_count; ++digit) {
            if (base_table[digit].infinity) {
                (*psi_table)[digit] = base_table[digit];
                continue;
            }
            FieldElement x_beta = base_table[digit].x * beta;
            (*psi_table)[digit] = {x_beta, base_table[digit].y, false};
        }
    }
}

#if !SECP256K1_ESP32_BUILD
// Desktop-only: Cache and streaming functions require filesystem and threading
// Forward declarations for cache structures and functions
struct CacheHeader {
    std::uint32_t magic;
    std::uint32_t version;
    std::uint32_t window_bits;
    std::uint32_t window_count;
    std::uint64_t digit_count;
    std::uint32_t has_glv;
    std::uint32_t reserved;
};

constexpr std::uint32_t CACHE_MAGIC = 0x53454350U;  // "SECP"
constexpr std::uint32_t CACHE_VERSION = 1U;

bool write_affine_point(std::ofstream& file, const AffinePointPacked& point);

// Streaming cache builder with queue
struct WindowData {
    std::size_t window_index;
    std::vector<AffinePointPacked> base_table;
    std::vector<AffinePointPacked> psi_table;
};

[[maybe_unused]] std::unique_ptr<PrecomputeContext> build_context_streaming(const FixedBaseConfig& config, const std::string& cache_path) {
    if (config.window_bits < 2U || config.window_bits > 30U) {
        throw std::runtime_error("window_bits must be between 2 and 30");
    }
    
    auto ctx = std::make_unique<PrecomputeContext>();
    ctx->config = config;
    ctx->window_bits = config.window_bits;
    ctx->window_count = (256U + config.window_bits - 1U) / config.window_bits;
    ctx->digit_count = static_cast<std::size_t>(1ULL << config.window_bits);
    ctx->beta = FieldElement::from_bytes(kBetaBytes);
    
    // Pre-compute window bases
    std::vector<Point> window_bases(ctx->window_count);
    Point base = Point::generator();
    for (std::size_t window = 0; window < ctx->window_count; ++window) {
        window_bases[window] = base;
        if (window + 1 < ctx->window_count) {
            for (unsigned rep = 0; rep < ctx->window_bits; ++rep) {
                base.dbl_inplace();
            }
        }
    }
    
    // Open file for streaming write
    std::ofstream file(cache_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open cache file for writing: " + cache_path);
    }
    
    // Write header
    CacheHeader header{};
    header.magic = CACHE_MAGIC;
    header.version = CACHE_VERSION;
    header.window_bits = ctx->window_bits;
    header.window_count = static_cast<std::uint32_t>(ctx->window_count);
    header.digit_count = ctx->digit_count;
    header.has_glv = config.enable_glv ? 1 : 0;
    header.reserved = 0;
    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    
    // Queue for window data with size limit
    const std::size_t max_queue_size = std::max<std::size_t>(2, std::min<std::size_t>(8, ctx->window_count / 4));  // 2-8 windows buffered
    std::queue<WindowData> ready_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::condition_variable queue_space_cv;
    std::atomic<bool> generation_done{false};
    std::atomic<std::size_t> windows_written{0};
    
    // Writer thread
    std::thread writer([&]() {
        std::size_t expected_window = 0;
        std::vector<WindowData> pending;  // Out-of-order windows
        
        while (expected_window < ctx->window_count) {
            WindowData data;
            bool got_data = false;
            
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_cv.wait(lock, [&]() { return !ready_queue.empty() || generation_done.load(); });
                
                if (!ready_queue.empty()) {
                    data = std::move(ready_queue.front());
                    ready_queue.pop();
                    got_data = true;
                    queue_space_cv.notify_one();  // Notify workers there's space
                }
            }
            
            if (got_data) {
                if (data.window_index == expected_window) {
                    // Write base table
                    for (const auto& point : data.base_table) {
                        write_affine_point(file, point);
                    }
                    // Write psi table if GLV enabled
                    if (config.enable_glv) {
                        for (const auto& point : data.psi_table) {
                            write_affine_point(file, point);
                        }
                    }
                    expected_window++;
                    windows_written.fetch_add(1);
                    
                    // Report write progress
                    if (config.progress_callback) {
                        config.progress_callback(expected_window * ctx->digit_count, 
                                               ctx->window_count * ctx->digit_count,
                                               static_cast<unsigned>(expected_window),
                                               static_cast<unsigned>(ctx->window_count));
                    }
                    
                    // Check pending for next windows
                    bool found = true;
                    while (found && expected_window < ctx->window_count) {
                        found = false;
                        for (auto it = pending.begin(); it != pending.end(); ++it) {
                            if (it->window_index == expected_window) {
                                // Write base table
                                for (const auto& point : it->base_table) {
                                    write_affine_point(file, point);
                                }
                                // Write psi table if GLV enabled
                                if (config.enable_glv) {
                                    for (const auto& point : it->psi_table) {
                                        write_affine_point(file, point);
                                    }
                                }
                                expected_window++;
                                windows_written.fetch_add(1);
                                
                                // Report write progress
                                if (config.progress_callback) {
                                    config.progress_callback(expected_window * ctx->digit_count,
                                                           ctx->window_count * ctx->digit_count,
                                                           static_cast<unsigned>(expected_window),
                                                           static_cast<unsigned>(ctx->window_count));
                                }
                                
                                pending.erase(it);
                                found = true;
                                break;
                            }
                        }
                    }
                } else {
                    // Store for later
                    pending.push_back(std::move(data));
                }
            }
        }
        
        file.close();
    });
    
    // Worker threads
    const unsigned requested_threads = config.thread_count;
    unsigned worker_count = requested_threads;
    if (worker_count == 0U) {
        worker_count = std::thread::hardware_concurrency();
    }
    if (worker_count == 0U) {
        worker_count = 1U;
    }
    worker_count = static_cast<unsigned>(std::min<std::size_t>(worker_count, ctx->window_count));
    if (worker_count == 0U) {
        worker_count = 1U;
    }
    
    std::atomic<std::size_t> next_window{0};
    std::atomic<std::size_t> completed_windows{0};
    
    auto worker = [&](unsigned) {
        while (true) {
            const std::size_t window = next_window.fetch_add(1, std::memory_order_relaxed);
            if (window >= ctx->window_count) {
                break;
            }
            
            WindowData data;
            data.window_index = window;
            fill_tables_for_window(window_bases[window], ctx->digit_count, ctx->beta, data.base_table, 
                                 config.enable_glv ? &data.psi_table : nullptr);
            
            completed_windows.fetch_add(1);
            
            // Wait if queue is full
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_space_cv.wait(lock, [&]() { return ready_queue.size() < max_queue_size; });
                ready_queue.push(std::move(data));
            }
            queue_cv.notify_one();
        }
    };
    
    // Start workers
    if (worker_count == 1U) {
        worker(0U);
    } else {
        std::vector<std::thread> threads;
        threads.reserve(worker_count);
        for (unsigned t = 0; t < worker_count; ++t) {
            threads.emplace_back(worker, t);
        }
        for (auto& thread : threads) {
            thread.join();
        }
    }
    
    generation_done.store(true);
    queue_cv.notify_all();
    writer.join();
    
    // Don't load back into memory - streaming mode doesn't need it
    ctx->base_tables.clear();
    ctx->psi_tables.clear();
    
    return ctx;
}

std::unique_ptr<PrecomputeContext> build_context(const FixedBaseConfig& config) {
    if (config.window_bits < 2U || config.window_bits > 30U) {
        throw std::runtime_error("window_bits must be between 2 and 30");
    }
    auto ctx = std::make_unique<PrecomputeContext>();
    ctx->config = config;
    ctx->window_bits = config.window_bits;
    ctx->window_count = (256U + config.window_bits - 1U) / config.window_bits;
    ctx->digit_count = static_cast<std::size_t>(1ULL << config.window_bits);
    ctx->beta = FieldElement::from_bytes(kBetaBytes);
    ctx->base_tables.resize(ctx->window_count);
    if (config.enable_glv) {
        ctx->psi_tables.resize(ctx->window_count);
    }

    std::vector<Point> window_bases(ctx->window_count);
    Point base = Point::generator();
    for (std::size_t window = 0; window < ctx->window_count; ++window) {
        window_bases[window] = base;
        if (window + 1 < ctx->window_count) {
            for (unsigned rep = 0; rep < ctx->window_bits; ++rep) {
                base = base.dbl();
            }
        }
    }

    const unsigned requested_threads = config.thread_count;
    unsigned worker_count = requested_threads;
    if (worker_count == 0U) {
        worker_count = std::thread::hardware_concurrency();
    }
    if (worker_count == 0U) {
        worker_count = 1U;
    }
    worker_count = static_cast<unsigned>(std::min<std::size_t>(worker_count, ctx->window_count));
    if (worker_count == 0U) {
        worker_count = 1U;
    }

    std::atomic<std::size_t> next_window{0};
    std::atomic<std::size_t> completed_points{0};
    const std::size_t total_points = ctx->window_count * ctx->digit_count;
    
    auto worker = [&](unsigned) {
        while (true) {
            const std::size_t window = next_window.fetch_add(1, std::memory_order_relaxed);
            if (window >= ctx->window_count) {
                break;
            }
            std::vector<AffinePointPacked>* psi_ptr = config.enable_glv ? &ctx->psi_tables[window] : nullptr;
            fill_tables_for_window(window_bases[window], ctx->digit_count, ctx->beta, ctx->base_tables[window], psi_ptr);
            
            // Report progress
            if (config.progress_callback) {
                const std::size_t points_done = completed_points.fetch_add(ctx->digit_count, std::memory_order_relaxed) + ctx->digit_count;
                config.progress_callback(points_done, total_points, static_cast<unsigned>(window + 1), static_cast<unsigned>(ctx->window_count));
            }
        }
    };

    if (worker_count == 1U) {
        worker(0U);
    } else {
        std::vector<std::thread> threads;
        threads.reserve(worker_count);
        for (unsigned t = 0; t < worker_count; ++t) {
            threads.emplace_back(worker, t);
        }
        for (auto& thread : threads) {
            thread.join();
        }
    }

    return ctx;
}
#endif // !SECP256K1_ESP32_BUILD

// ESP32-compatible simple build_context (single-threaded, no caching)
#if SECP256K1_ESP32_BUILD
std::unique_ptr<PrecomputeContext> build_context(const FixedBaseConfig& config) {
    auto ctx = std::make_unique<PrecomputeContext>();
    ctx->config = config;
    ctx->window_bits = config.window_bits;
    ctx->window_count = (256U + config.window_bits - 1U) / config.window_bits;
    ctx->digit_count = static_cast<std::size_t>(1ULL << config.window_bits);
    ctx->beta = FieldElement::from_bytes(kBetaBytes);
    ctx->base_tables.resize(ctx->window_count);
    if (config.enable_glv) {
        ctx->psi_tables.resize(ctx->window_count);
    }

    Point base = Point::generator();
    for (std::size_t window = 0; window < ctx->window_count; ++window) {
        std::vector<AffinePointPacked>* psi_ptr = config.enable_glv ? &ctx->psi_tables[window] : nullptr;
        fill_tables_for_window(base, ctx->digit_count, ctx->beta, ctx->base_tables[window], psi_ptr);

        if (window + 1 < ctx->window_count) {
            for (unsigned rep = 0; rep < ctx->window_bits; ++rep) {
                base = base.dbl();
            }
        }
    }

    return ctx;
}
#endif

Scalar const_minus_b1() {
    static const Scalar value = make_scalar(kMinusB1Bytes);
    return value;
}

Scalar const_minus_b2() {
    static const Scalar value = make_scalar(kMinusB2Bytes);
    return value;
}

Scalar const_lambda() {
    static const Scalar value = make_scalar(kLambdaBytes);
    return value;
}

[[maybe_unused]] Scalar const_a1() {
    static const Scalar value = make_scalar(kA1Bytes);
    return value;
}
[[maybe_unused]] Scalar const_a2() {
    static const Scalar value = make_scalar(kA2Bytes);
    return value;
}
// Magnitudes of basis components (positive, small ~128-129 bits)
constexpr std::array<std::uint8_t, 32> kB1MagBytes{
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0xE4,0x43,0x7E,0xD6,0x01,0x0E,0x88,0x28,
    0x6F,0x54,0x7F,0xA9,0x0A,0xBF,0xE4,0xC3
};
constexpr std::array<std::uint8_t, 32> kB2MagBytes{
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x30,0x86,0xD2,0x21,0xA7,0xD4,0x6B,0xCD,
    0xE8,0x6C,0x90,0xE4,0x92,0x84,0xEB,0x15
};

[[maybe_unused]] Scalar const_b1_mag() { // |b1|
    static const Scalar value = make_scalar(kB1MagBytes);
    return value;
}
[[maybe_unused]] Scalar const_b2_mag() { // |b2|
    static const Scalar value = make_scalar(kB2MagBytes);
    return value;
}

// Optimized multiplication for GLV decomposition where one operand is known to be small
// This uses simple double-and-add but only iterates over the actual bits needed
[[maybe_unused]] static Scalar mul_scalar_small(const Scalar& small, const Scalar& large) {
    // Find highest non-zero bit in small operand
    std::size_t max_bit = 0;
    for (std::size_t bit = 0; bit < 256; ++bit) {
        if (small.bit(bit)) {
            max_bit = bit;
        }
    }
    
    // Double-and-add only up to max_bit + 1
    Scalar result = Scalar::zero();
    Scalar base = large;
    
    for (std::size_t bit = 0; bit <= max_bit; ++bit) {
        if (small.bit(bit)) {
            result += base;
        }
        if (bit < max_bit) {  // No need to double on last iteration
            base += base;
        }
    }
    
    return result;
}

// Multiply two 64-bit numbers to get 128-bit result
static void mul64x64(std::uint64_t a, std::uint64_t b, std::uint64_t& lo, std::uint64_t& hi) {
#if defined(_MSC_VER) && !defined(__clang__)
    lo = _umul128(a, b, &hi);
#elif defined(SECP256K1_NO_INT128) || defined(SECP256K1_PLATFORM_ESP32)
    lo = _umul128(a, b, &hi);
#else
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
    unsigned __int128 product = static_cast<unsigned __int128>(a) * b;
    lo = static_cast<std::uint64_t>(product);
    hi = static_cast<std::uint64_t>(product >> 64);
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#endif
}

// Multiply two scalars and return result as raw 512-bit value (no modular reduction)
// Used for GLV decomposition where we need integer arithmetic
static std::array<std::uint64_t, 8> mul_scalar_raw(const Scalar& a, const Scalar& b) {
#if SECP256K1_PROFILE_DECOMP
    unsigned long long start_total = RDTSC();
#endif

    auto a_limbs = scalar_to_limbs(a);
    auto b_limbs = scalar_to_limbs(b);
    
#if SECP256K1_PROFILE_DECOMP
    unsigned long long start_mul = RDTSC();
#endif

    // 256-bit x 256-bit = 512-bit multiplication
    std::array<std::uint64_t, 8> result{};
    
    for (std::size_t i = 0; i < 4; ++i) {
        std::uint64_t carry = 0;
        for (std::size_t j = 0; j < 4; ++j) {
            std::uint64_t lo, hi;
            mul64x64(a_limbs[i], b_limbs[j], lo, hi);
            
            // Add to result[i+j]
            std::uint64_t sum_lo = result[i + j] + lo;
            std::uint64_t carry1 = (sum_lo < result[i + j]) ? 1 : 0;
            
            // Add carry from previous
            sum_lo += carry;
            std::uint64_t carry2 = (sum_lo < carry) ? 1 : 0;
            
            result[i + j] = sum_lo;
            carry = hi + carry1 + carry2;
        }
        if (i + 4 < 8) {
            result[i + 4] += carry;
        }
    }
    
#if SECP256K1_PROFILE_DECOMP
    unsigned long long end = RDTSC();
    static unsigned long long total_prep = 0;
    static unsigned long long total_mul = 0;
    static int mul_calls = 0;
    
    total_prep += (start_mul - start_total);
    total_mul += (end - start_mul);
    mul_calls++;
    
    if (mul_calls == 3000) {
        std::printf("  [MUL_SCALAR_RAW] After %d calls:\n", mul_calls);
        std::printf("    Prepare (scalar_to_limbs): %llu cycles\n", total_prep / mul_calls);
        std::printf("    4x4 multiplication loop:   %llu cycles\n", total_mul / mul_calls);
    }
#endif
    
    return result;
}

// Optimized 128x256-bit multiplication for GLV decomposition
// When 'a' is known to be ~128-bit (only low 2 limbs significant)
[[maybe_unused]] static std::array<std::uint64_t, 8> mul_128x256_raw(const Limbs4& a_limbs, const Limbs4& b_limbs) {
    std::array<std::uint64_t, 8> result{};
    
    // Only multiply first 2 limbs of 'a' (128-bit) with all 4 limbs of 'b' (256-bit)
    // This gives us 8 multiplications instead of 16!
    
    // i = 0: a[0] x b[0..3]
    {
        std::uint64_t carry = 0;
        for (std::size_t j = 0; j < 4; ++j) {
            std::uint64_t lo, hi;
            lo = _umul128(a_limbs[0], b_limbs[j], &hi);
            
            // Add to result[i+j]
            std::uint64_t sum_lo = result[j] + lo;
            std::uint64_t carry1 = (sum_lo < result[j]) ? 1 : 0;

            // Add carry from previous
            sum_lo += carry;
            std::uint64_t carry2 = (sum_lo < carry) ? 1 : 0;

            result[j] = sum_lo;
            carry = hi + carry1 + carry2;
        }
    }
    
    // i = 1: a[1] x b[0..3]
    {
        std::uint64_t carry = 0;
        for (std::size_t j = 0; j < 4; ++j) {
            std::uint64_t lo, hi;
            lo = _umul128(a_limbs[1], b_limbs[j], &hi);
            
            // Add to result[i+j]
            std::uint64_t sum_lo = result[1 + j] + lo;
            std::uint64_t carry1 = (sum_lo < result[1 + j]) ? 1 : 0;

            // Add carry from previous
            sum_lo += carry;
            std::uint64_t carry2 = (sum_lo < carry) ? 1 : 0;

            result[1 + j] = sum_lo;
            carry = hi + carry1 + carry2;
        }
    }
    
    // Skip i=2,3 since a[2]=a[3]=0 (or very small) for 128-bit values
    
    return result;
}

// Add two 512-bit numbers
[[maybe_unused]] static std::array<std::uint64_t, 8> add_512(
    const std::array<std::uint64_t, 8>& a,
    const std::array<std::uint64_t, 8>& b
) {
    std::array<std::uint64_t, 8> result{};
    std::uint64_t carry = 0;
    
    for (std::size_t i = 0; i < 8; ++i) {
        std::uint64_t sum = a[i] + carry;
        std::uint64_t carry1 = (sum < carry) ? 1 : 0;
        sum += b[i];
        std::uint64_t carry2 = (sum < b[i]) ? 1 : 0;
        result[i] = sum;
        carry = carry1 + carry2;
    }
    
    return result;
}

// Subtract two 512-bit numbers (assumes a >= b)
[[maybe_unused]] static std::array<std::uint64_t, 8> sub_512(
    const std::array<std::uint64_t, 8>& a,
    const std::array<std::uint64_t, 8>& b
) {
    std::array<std::uint64_t, 8> result{};
    std::uint64_t borrow = 0ULL;
    for (std::size_t i = 0; i < 8; ++i) {
        const std::uint64_t bi = b[i] + borrow;
        const bool need_borrow = (a[i] < bi);
        result[i] = a[i] - bi;
        borrow = need_borrow ? 1ULL : 0ULL;
    }
    return result;
}

[[maybe_unused]] static bool ge_512(const std::array<std::uint64_t, 8>& a,
                   const std::array<std::uint64_t, 8>& b) {
    for (int i = 7; i >= 0; --i) {
        if (a[i] > b[i]) return true;
        if (a[i] < b[i]) return false;
    }
    return true; // equal
}

// Multiply two Scalars modulo n using 256x256->512 wide multiply and reduction.
// Forward declaration for 512-bit reduction (defined later).
static Scalar reduce_512_mod_n(const std::array<std::uint64_t, 8>& wide);

[[maybe_unused]] static Scalar mul_mod_n(const Scalar& a, const Scalar& b) {
    auto wide = mul_scalar_raw(a, b);
    return reduce_512_mod_n(wide);
}

// Negate a 512-bit number (two's complement)
[[maybe_unused]] static std::array<std::uint64_t, 8> negate_512(const std::array<std::uint64_t, 8>& a) {
    std::array<std::uint64_t, 8> result{};
    
    // Invert all bits
    for (std::size_t i = 0; i < 8; ++i) {
        result[i] = ~a[i];
    }
    
    // Add 1
    std::uint64_t carry = 1;
    for (std::size_t i = 0; i < 8; ++i) {
        std::uint64_t sum = result[i] + carry;
        carry = (sum < result[i]) ? 1 : 0;
        result[i] = sum;
    }
    
    return result;
}

// Convert 512-bit value to Scalar (take lower 256 bits mod n)
// If the 512-bit value is negative (bit 511 set), we need to add n back
[[maybe_unused]] static Scalar scalar_from_512(const std::array<std::uint64_t, 8>& wide) {
    // Check if negative (bit 511 set = MSB of wide[7])
    bool is_negative = (wide[7] & 0x8000000000000000ULL) != 0;
    
    if (is_negative) {
        // Value is negative in two's complement
        // We need: result = (wide mod n)
        // Since wide is negative: wide = -x where x > 0
        // So: wide mod n = n - (x mod n)
        
        // First negate to get positive value
        auto pos_wide = negate_512(wide);
        
        // Take lower 256 bits
        Limbs4 lower{{pos_wide[0], pos_wide[1], pos_wide[2], pos_wide[3]}};
        Scalar pos_scalar = scalar_from_limbs(lower);
        
        // Return n - pos_scalar
        return Scalar::zero() - pos_scalar;
    } else {
        // Positive value, just take lower 256 bits
        Limbs4 lower{{wide[0], wide[1], wide[2], wide[3]}};
        return scalar_from_limbs(lower);
    }
}

// Reduce 512-bit value modulo group order n using Scalar operations.
// Computes rem = wide mod n via Horner-like method: rem = (((w7)*B + w6)*B + ... + w0) mod n, where B=2^64.
static Scalar reduce_512_mod_n(const std::array<std::uint64_t, 8>& wide) {
    Scalar rem = Scalar::zero();
    // B = 2^64 mod n
    Scalar B = Scalar::from_limbs({0ULL, 1ULL, 0ULL, 0ULL});
    for (int i = 7; i >= 0; --i) {
        if (!rem.is_zero()) {
            rem = rem * B; // rem *= 2^64 mod n
        }
        if (wide[i] != 0ULL) {
            rem += Scalar::from_uint64(wide[i]);
        }
    }
    return rem;
}

// Barrett reduction constant: u = floor(2^512 / n) where n is secp256k1 group order
// This allows fast modular reduction: x mod n ~= x - floor(x*u / 2^512) * n
constexpr std::array<std::uint64_t, 8> kBarrettMu = {
    0x402DA1732FC9BEC0ULL,
    0x4551231950B75FC4ULL,
    0x0000000000000001ULL,
    0x0000000000000000ULL,
    0x0000000000000001ULL,
    0x0000000000000000ULL,
    0x0000000000000000ULL,
    0x0000000000000000ULL
};

// Fast Barrett reduction for 512-bit value modulo n
// About 10-15x faster than Horner method for small inputs
static Scalar barrett_reduce_512(const std::array<std::uint64_t, 8>& wide) {
#if SECP256K1_PROFILE_DECOMP
    unsigned long long start = RDTSC();
#endif

    // For values < 2*n, we can use simple subtraction
    // Check if high 256 bits are zero
    bool high_zero = (wide[4] == 0 && wide[5] == 0 && wide[6] == 0 && wide[7] == 0);
    
#if SECP256K1_PROFILE_DECOMP
    unsigned long long t1 = RDTSC();
#endif
    
    if (high_zero) {
#if SECP256K1_PROFILE_DECOMP
        unsigned long long t2 = RDTSC();
#endif
        
        // Group order n
        constexpr Limbs4 N = {
            0xBFD25E8CD0364141ULL,
            0xBAAEDCE6AF48A03BULL,
            0xFFFFFFFFFFFFFFFEULL,
            0xFFFFFFFFFFFFFFFFULL
        };
        
        // Check if result >= n (simple comparison)
        Limbs4 low_limbs = {wide[0], wide[1], wide[2], wide[3]};
        bool ge_n = (low_limbs[3] > N[3]) ||
                    (low_limbs[3] == N[3] && low_limbs[2] > N[2]) ||
                    (low_limbs[3] == N[3] && low_limbs[2] == N[2] && low_limbs[1] > N[1]) ||
                    (low_limbs[3] == N[3] && low_limbs[2] == N[2] && low_limbs[1] == N[1] && low_limbs[0] >= N[0]);
        
#if SECP256K1_PROFILE_DECOMP
        unsigned long long t3 = RDTSC();
#endif
        
        if (ge_n) {
            // Subtract n once using intrinsics
            unsigned char borrow = 0;
            borrow = COMPAT_SUBBORROW_U64(borrow, low_limbs[0], N[0], &low_limbs[0]);
            borrow = COMPAT_SUBBORROW_U64(borrow, low_limbs[1], N[1], &low_limbs[1]);
            borrow = COMPAT_SUBBORROW_U64(borrow, low_limbs[2], N[2], &low_limbs[2]);
            borrow = COMPAT_SUBBORROW_U64(borrow, low_limbs[3], N[3], &low_limbs[3]);
        }
        
#if SECP256K1_PROFILE_DECOMP
        unsigned long long t4 = RDTSC();
#endif
        
        // Convert to Scalar using fast normalized path (skip >= n check)
        Scalar result = scalar_from_limbs_normalized(low_limbs);
        
#if SECP256K1_PROFILE_DECOMP
        unsigned long long end = RDTSC();
        static unsigned long long total_check = 0;
        static unsigned long long total_compare = 0;
        static unsigned long long total_subtract = 0;
        static unsigned long long total_to_scalar = 0;
        static int barrett_calls = 0;
        
        total_check += (t1 - start);
        total_compare += (t3 - t2);
        total_subtract += (t4 - t3);
        total_to_scalar += (end - t4);
        barrett_calls++;
        
        if (barrett_calls == 1000 || barrett_calls == 3000) {
            std::printf("  [BARRETT_REDUCE] After %d calls:\n", barrett_calls);
            std::printf("    High-zero check:      %llu cycles\n", total_check / barrett_calls);
            std::printf("    >= n comparison:      %llu cycles\n", total_compare / barrett_calls);
            std::printf("    Subtraction (if ge):  %llu cycles\n", total_subtract / barrett_calls);
            std::printf("    Convert to Scalar:    %llu cycles\n", total_to_scalar / barrett_calls);
            std::printf("    TOTAL AVG:            %llu cycles\n", (total_check + total_compare + total_subtract + total_to_scalar) / barrett_calls);
        }
#endif
        
        return result;
    }
    
    // Fall back to Horner method for large values (rare case)
    return reduce_512_mod_n(wide);
}

// Fast bitlength calculation using intrinsic (inlined)
static inline unsigned fast_bitlen(const Scalar& s) {
    auto limbs = scalar_to_limbs(s);
    for (int i = 3; i >= 0; --i) {
        if (limbs[i] != 0) {
            unsigned long index;
            _BitScanReverse64(&index, limbs[i]);
            return (unsigned)(i * 64 + index + 1);
        }
    }
    return 0;
}

// Joint Sparse Form (JSF) style two-scalar recoding
// Produces digit pairs (u1[i], u2[i]) in {-1,0,+1}, such that
//   k1 = sum_i u1[i] * 2^i and k2 = sum_i u2[i] * 2^i
// Algorithm: simple and correct recoding choosing +/-1 for odd limbs,
// prioritizing correctness (minimal-weight tweaks can be added later).
struct JSF_Result {
    std::vector<int8_t> jsf1;  // k1 coefficients
    std::vector<int8_t> jsf2;  // k2 coefficients
    std::size_t length;        // Number of produced digits
};

static JSF_Result compute_jsf(const Scalar& k1, const Scalar& k2) {
    // Work on 5-limb arrays to safely handle carries/borrows during updates
    std::array<std::uint64_t, 5> a{};
    std::array<std::uint64_t, 5> b{};
    const auto& l1 = k1.limbs();
    const auto& l2 = k2.limbs();
    for (std::size_t i = 0; i < 4; ++i) {
        a[i] = l1[i];
        b[i] = l2[i];
    }
    a[4] = 0; b[4] = 0;

    JSF_Result out;
    out.jsf1.reserve(260);
    out.jsf2.reserve(260);

    std::size_t steps = 0;
    while (a[0] | a[1] | a[2] | a[3] | b[0] | b[1] | b[2] | b[3]) {
        int8_t u1 = 0, u2 = 0;

        // Pick signed digits for odd values. For correctness, it's sufficient
        // to use +1 for odd; we choose +/-1 based on mod 4 to reduce weight.
        if (a[0] & 1ULL) {
            const int m1 = static_cast<int>(a[0] & 3ULL);
            u1 = (m1 == 1) ? 1 : -1;  // if 3 -> -1
        }
        if (b[0] & 1ULL) {
            const int m2 = static_cast<int>(b[0] & 3ULL);
            u2 = (m2 == 1) ? 1 : -1;
        }

        // Optional tie-break to avoid both non-zero with same sign frequently.
        // This does not affect correctness, only weight. Keep simple and safe.
        if (u1 != 0 && u2 != 0) {
            // If both have the same sign, flip the one with less-preferred mod8
            if ((u1 > 0 && u2 > 0) || (u1 < 0 && u2 < 0)) {
                const int a8 = static_cast<int>(a[0] & 7ULL);
                const int b8 = static_cast<int>(b[0] & 7ULL);
                // Prefer making the one with {3,5} modulo 8 change sign
                const bool a_pref = (a8 == 3) || (a8 == 5);
                const bool b_pref = (b8 == 3) || (b8 == 5);
                if (a_pref && !b_pref) {
                    u1 = static_cast<int8_t>(-u1);
                } else if (!a_pref && b_pref) {
                    u2 = static_cast<int8_t>(-u2);
                } else {
                    // If inconclusive, flip u2 to balance
                    u2 = static_cast<int8_t>(-u2);
                }
            }
        }

        out.jsf1.push_back(u1);
        out.jsf2.push_back(u2);

        // Update a := (a - u1) >> 1
        if (u1 != 0) {
            unsigned char borrow = 0;
            std::uint64_t tmp;
            borrow = COMPAT_SUBBORROW_U64(borrow, a[0], static_cast<std::uint64_t>(static_cast<int64_t>(u1)), &tmp); a[0] = tmp;
            borrow = COMPAT_SUBBORROW_U64(borrow, a[1], 0ULL, &tmp); a[1] = tmp;
            borrow = COMPAT_SUBBORROW_U64(borrow, a[2], 0ULL, &tmp); a[2] = tmp;
            borrow = COMPAT_SUBBORROW_U64(borrow, a[3], 0ULL, &tmp); a[3] = tmp;
            borrow = COMPAT_SUBBORROW_U64(borrow, a[4], 0ULL, &tmp); a[4] = tmp;
        }
        a[0] = (a[0] >> 1) | (a[1] << 63);
        a[1] = (a[1] >> 1) | (a[2] << 63);
        a[2] = (a[2] >> 1) | (a[3] << 63);
        a[3] = (a[3] >> 1) | (a[4] << 63);
        a[4] >>= 1;

        // Update b := (b - u2) >> 1
        if (u2 != 0) {
            unsigned char borrow = 0;
            std::uint64_t tmp;
            borrow = COMPAT_SUBBORROW_U64(borrow, b[0], static_cast<std::uint64_t>(static_cast<int64_t>(u2)), &tmp); b[0] = tmp;
            borrow = COMPAT_SUBBORROW_U64(borrow, b[1], 0ULL, &tmp); b[1] = tmp;
            borrow = COMPAT_SUBBORROW_U64(borrow, b[2], 0ULL, &tmp); b[2] = tmp;
            borrow = COMPAT_SUBBORROW_U64(borrow, b[3], 0ULL, &tmp); b[3] = tmp;
            borrow = COMPAT_SUBBORROW_U64(borrow, b[4], 0ULL, &tmp); b[4] = tmp;
        }
        b[0] = (b[0] >> 1) | (b[1] << 63);
        b[1] = (b[1] >> 1) | (b[2] << 63);
        b[2] = (b[2] >> 1) | (b[3] << 63);
        b[3] = (b[3] >> 1) | (b[4] << 63);
        b[4] >>= 1;

        ++steps;
        if (steps > 260) {
            #if !SECP256K1_ESP32_BUILD
            throw std::runtime_error("compute_jsf exceeded expected length");
            #else
            break; // ESP32: break on error
            #endif
        }
    }

    out.length = out.jsf1.size();
    return out;
}

// Fast 128x128 multiplication returning low 256 bits (for small scalar muls)
// Input: two 128-bit values (stored in first 2 limbs)
// Output: 256-bit product
[[maybe_unused]] static inline Limbs4 mul_small_128x128(const Limbs4& a, const Limbs4& b) {
    Limbs4 result = {0, 0, 0, 0};
    
    // a0 * b0
    uint64_t hi, lo = _umul128(a[0], b[0], &hi);
    result[0] = lo;
    result[1] = hi;
    
    // a0 * b1
    lo = _umul128(a[0], b[1], &hi);
    unsigned char carry = COMPAT_ADDCARRY_U64(0, result[1], lo, &result[1]);
    COMPAT_ADDCARRY_U64(carry, result[2], hi, &result[2]);
    
    // a1 * b0
    lo = _umul128(a[1], b[0], &hi);
    carry = COMPAT_ADDCARRY_U64(0, result[1], lo, &result[1]);
    carry = COMPAT_ADDCARRY_U64(carry, result[2], hi, &result[2]);
    COMPAT_ADDCARRY_U64(carry, result[3], 0, &result[3]);
    
    // a1 * b1
    lo = _umul128(a[1], b[1], &hi);
    carry = COMPAT_ADDCARRY_U64(0, result[2], lo, &result[2]);
    COMPAT_ADDCARRY_U64(carry, result[3], hi, &result[3]);
    
    return result;
}

// Add two 256-bit limb values (mod n)
[[maybe_unused]] static inline Limbs4 add_limbs_mod_n(const Limbs4& a, const Limbs4& b) {
    // Group order n
    constexpr Limbs4 N = {
        0xBFD25E8CD0364141ULL,
        0xBAAEDCE6AF48A03BULL,
        0xFFFFFFFFFFFFFFFEULL,
        0xFFFFFFFFFFFFFFFFULL
    };
    
    Limbs4 result;
    unsigned char carry = 0;
    carry = COMPAT_ADDCARRY_U64(0, a[0], b[0], &result[0]);
    carry = COMPAT_ADDCARRY_U64(carry, a[1], b[1], &result[1]);
    carry = COMPAT_ADDCARRY_U64(carry, a[2], b[2], &result[2]);
    carry = COMPAT_ADDCARRY_U64(carry, a[3], b[3], &result[3]);
    
    // Check if result >= n
    bool ge_n = (result[3] > N[3]) ||
                (result[3] == N[3] && result[2] > N[2]) ||
                (result[3] == N[3] && result[2] == N[2] && result[1] > N[1]) ||
                (result[3] == N[3] && result[2] == N[2] && result[1] == N[1] && result[0] >= N[0]);
    
    if (ge_n) {
        // Subtract n
        unsigned char borrow = 0;
        borrow = COMPAT_SUBBORROW_U64(0, result[0], N[0], &result[0]);
        borrow = COMPAT_SUBBORROW_U64(borrow, result[1], N[1], &result[1]);
        borrow = COMPAT_SUBBORROW_U64(borrow, result[2], N[2], &result[2]);
        borrow = COMPAT_SUBBORROW_U64(borrow, result[3], N[3], &result[3]);
    }
    
    return result;
}

ScalarDecomposition split_scalar_internal(const Scalar& scalar) {
#if SECP256K1_PROFILE_DECOMP
    uint64_t t0 = RDTSC();
#endif
    
    const auto value_limbs = scalar_to_limbs(scalar);
    
#if SECP256K1_PROFILE_DECOMP
    uint64_t t1 = RDTSC();
#endif
    
    // Compute c1,c2 using libsecp256k1-style g1/g2 constants and 384-bit shift rounding
    Limbs4 c1_limbs = mul_shift(value_limbs, kG1MulShift, 384U);
    Limbs4 c2_limbs = mul_shift(value_limbs, kG2MulShift, 384U);
    
#if SECP256K1_PROFILE_DECOMP
    uint64_t t2 = RDTSC();
#endif
    
    // Limb-only path to build k2 and lambda*k2 with minimal Scalar overhead
#if SECP256K1_LIMB_GLV
    static const Limbs4 minus_b1_limbs = scalar_to_limbs(const_minus_b1());
    static const Limbs4 minus_b2_limbs = scalar_to_limbs(const_minus_b2());
    static const Limbs4 lambda_limbs   = scalar_to_limbs(const_lambda());

    // Helper: 256x256 -> 512 schoolbook multiply (little-endian limbs)
    auto mul_256 = [](const Limbs4& a, const Limbs4& b) {
        std::array<uint64_t,8> out{}; // little-endian 64-bit limbs
        for(int i=0;i<4;++i){
            for(int j=0;j<4;++j){
                uint64_t hi=0;
#if defined(_MSC_VER) && !defined(__clang__)
                uint64_t lo = _umul128(a[i], b[j], &hi);
#elif defined(SECP256K1_NO_INT128) || defined(SECP256K1_PLATFORM_ESP32)
                uint64_t lo = _umul128(a[i], b[j], &hi);
#else
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
                unsigned __int128 prod = (unsigned __int128)a[i] * (unsigned __int128)b[j];
                uint64_t lo = (uint64_t)prod;
                hi = (uint64_t)(prod >> 64);
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#endif
                int k = i + j;
                // Add low part with carry propagation
                uint64_t prev = out[k];
                out[k] += lo;
                uint64_t carry = (out[k] < prev) ? 1ULL : 0ULL; // overflow detection
                // Add high part plus carry to next limb(s)
                if(k+1 < 8){
                    uint64_t prev2 = out[k+1];
                    out[k+1] += hi + carry;
                    uint64_t carry2 = (out[k+1] < prev2) ? 1ULL : 0ULL;
                    // propagate if needed further (rare)
                    int m = k+2;
                    while(carry2 && m < 8){
                        uint64_t prevm = out[m];
                        out[m] += 1ULL;
                        carry2 = (out[m] < prevm) ? 1ULL : 0ULL;
                        ++m;
                    }
                }
            }
        }
        return out;
    };

    auto add_512 = [](const std::array<uint64_t,8>& A, const std::array<uint64_t,8>& B){
        std::array<uint64_t,8> out{};
        unsigned char c = 0;
        for(int i=0;i<8;++i){
            c = COMPAT_ADDCARRY_U64(c, A[i], B[i], &out[i]);
        }
        return out; // ignore final carry (expected < 2^512 range)
    };

    // Compute c1*minus_b1 + c2*minus_b2 as 512-bit before reduction
    auto prod1 = mul_256(c1_limbs, minus_b1_limbs);
    auto prod2 = mul_256(c2_limbs, minus_b2_limbs);
    auto k2_512 = add_512(prod1, prod2);

    // Barrett reduce 512 -> 256 using existing helper
    Scalar k2_mod = barrett_reduce_512(k2_512);
#else
    // Legacy Scalar-based path
    static const Scalar lambda = const_lambda();
    Scalar c1 = scalar_from_limbs(c1_limbs);
    Scalar c2 = scalar_from_limbs(c2_limbs);
    static const Scalar minus_b1 = const_minus_b1();
    static const Scalar minus_b2 = const_minus_b2();
    Scalar k2_mod = (c1 * minus_b1) + (c2 * minus_b2);
#endif
    
#if SECP256K1_PROFILE_DECOMP
    uint64_t t3 = RDTSC();
#else
    uint64_t t3 = 0;
    (void)t3; // Silence unused variable warning for CodeQL
    (void)t3;
#endif
    Scalar k2_neg_val = Scalar::zero() - k2_mod;
    bool k2_is_neg = (fast_bitlen(k2_neg_val) < fast_bitlen(k2_mod));
    Scalar k2_abs = k2_is_neg ? k2_neg_val : k2_mod;
    Scalar k2_signed = k2_is_neg ? (Scalar::zero() - k2_abs) : k2_abs;

#if SECP256K1_PROFILE_DECOMP
    uint64_t t4 = RDTSC();
#endif

    // lambda * k2 via limb multiply + Barrett (fused path when limb mode enabled)
#if SECP256K1_LIMB_GLV
    auto lambda_k2_512 = mul_256(lambda_limbs, scalar_to_limbs(k2_signed));
    Scalar lambda_k2 = barrett_reduce_512(lambda_k2_512);
#else
    // reuse previously defined static const Scalar lambda from legacy path above
    Scalar lambda_k2 = barrett_reduce_512(mul_scalar_raw(lambda, k2_signed));
#endif
    
#if SECP256K1_PROFILE_DECOMP
    uint64_t t5 = RDTSC();
#endif
    
    Scalar k1_mod = scalar - lambda_k2;
    Scalar k1_neg_val = Scalar::zero() - k1_mod;
    bool k1_is_neg = (fast_bitlen(k1_neg_val) < fast_bitlen(k1_mod));
    Scalar k1_abs = k1_is_neg ? k1_neg_val : k1_mod;

#if SECP256K1_PROFILE_DECOMP
    uint64_t t6 = RDTSC();
    static std::atomic<uint64_t> total_calls{0};
    static std::atomic<uint64_t> sum_scalar_to_limbs{0};
    static std::atomic<uint64_t> sum_mul_shift{0};
    static std::atomic<uint64_t> sum_k2_calc{0};
    static std::atomic<uint64_t> sum_k2_normalize{0};
    static std::atomic<uint64_t> sum_lambda_mul{0};
    static std::atomic<uint64_t> sum_k1_calc{0};
    
    total_calls++;
    sum_scalar_to_limbs += (t1 - t0);
    sum_mul_shift += (t2 - t1);
    sum_k2_calc += (t3 - t2);
    sum_k2_normalize += (t4 - t3);
    sum_lambda_mul += (t5 - t4);
    sum_k1_calc += (t6 - t5);

    // Accumulate into exported globals for external benchmarks
    g_decomp_scalar_to_limbs_cycles += (t1 - t0);
    g_decomp_mul_shift_cycles      += (t2 - t1);
    g_decomp_scalar_math_cycles    += (t3 - t2);      // k2 calculation core math
    g_decomp_barrett_reduce_cycles += (t5 - t4);      // lambda*k2 (barrett + mul)
    g_decomp_normalize_cycles      += (t4 - t3);      // k2 normalization/sign handling
    
    if (total_calls % 1000 == 0) {
        uint64_t n = total_calls;
        std::cerr << "\n[DECOMP PROFILE] After " << n << " calls:\n";
        std::cerr << "  scalar_to_limbs:  " << (sum_scalar_to_limbs / n) << " cycles\n";
        std::cerr << "  mul_shift (c1,c2): " << (sum_mul_shift / n) << " cycles\n";
        std::cerr << "  k2 calculation:   " << (sum_k2_calc / n) << " cycles\n";
        std::cerr << "  k2 normalize:     " << (sum_k2_normalize / n) << " cycles\n";
        std::cerr << "  lambda*k2:        " << (sum_lambda_mul / n) << " cycles\n";
        std::cerr << "  k1 calculation:   " << (sum_k1_calc / n) << " cycles\n";
        std::cerr << "  TOTAL AVG:        " << ((t6 - t0)) << " cycles (this call)\n";
        std::cerr.flush();
    }
#endif

    ScalarDecomposition decomposition{};
    decomposition.k1 = k1_abs;
    decomposition.k2 = k2_abs;
    decomposition.neg1 = k1_is_neg;
    decomposition.neg2 = k2_is_neg;

    return decomposition;
}

// ============================================================================
// Cache Serialization System
// ============================================================================
// Cache System
// ============================================================================

std::string get_default_cache_path(unsigned window_bits) {
    // Build cache filename with GLV suffix if enabled
    std::string filename = "cache_w" + std::to_string(window_bits);
    if (g_config.enable_glv) {
        filename += "_glv";
    }
    filename += ".bin";
    
    // Use configured cache directory (default: G:\EccTables)
    if (!g_config.cache_dir.empty()) {
        std::string cache_path = g_config.cache_dir + "\\" + filename;
        std::ifstream test_file(cache_path, std::ios::binary);
        if (test_file.good()) {
            test_file.close();
            return cache_path;
        }
    }
    
    // Fall back to current directory
    return filename;
}

bool write_field_element(std::ofstream& file, const FieldElement& fe) {
    auto bytes = fe.to_bytes();
    file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    return file.good();
}

bool read_field_element(std::ifstream& file, FieldElement& fe) {
    std::array<std::uint8_t, 32> bytes;
    file.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if (!file.good()) return false;
    fe = FieldElement::from_bytes(bytes);
    return true;
}

bool write_affine_point(std::ofstream& file, const AffinePointPacked& point) {
    std::uint8_t infinity_byte = point.infinity ? 1 : 0;
    file.write(reinterpret_cast<const char*>(&infinity_byte), 1);
    if (!file.good()) return false;
    
    if (!point.infinity) {
        if (!write_field_element(file, point.x)) return false;
        if (!write_field_element(file, point.y)) return false;
    }
    return true;
}

bool read_affine_point(std::ifstream& file, AffinePointPacked& point) {
    std::uint8_t infinity_byte;
    file.read(reinterpret_cast<char*>(&infinity_byte), 1);
    if (!file.good()) return false;
    
    point.infinity = (infinity_byte != 0);
    if (!point.infinity) {
        if (!read_field_element(file, point.x)) return false;
        if (!read_field_element(file, point.y)) return false;
    }
    return true;
}

// Internal version without lock - must be called with g_mutex already locked
bool save_precompute_cache_locked(const std::string& path) {
    if (!g_context) {
        return false;  // Nothing to save
    }
    
    PrecomputeContext& ctx = *g_context;
    
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // Write header
    CacheHeader header{};
    header.magic = CACHE_MAGIC;
    header.version = CACHE_VERSION;
    header.window_bits = ctx.window_bits;
    header.window_count = static_cast<std::uint32_t>(ctx.window_count);
    header.digit_count = ctx.digit_count;
    header.has_glv = ctx.config.enable_glv ? 1 : 0;
    header.reserved = 0;
    
    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    if (!file.good()) return false;
    
    // Write base tables
    for (const auto& window : ctx.base_tables) {
        for (const auto& point : window) {
            if (!write_affine_point(file, point)) {
                return false;
            }
        }
    }
    
    // Write psi tables if GLV enabled
    if (ctx.config.enable_glv) {
        for (const auto& window : ctx.psi_tables) {
            for (const auto& point : window) {
                if (!write_affine_point(file, point)) {
                    return false;
                }
            }
        }
    }
    
    file.close();
    return file.good();
}

// Internal version without lock - must be called with g_mutex already locked
bool load_precompute_cache_locked(const std::string& path, unsigned max_windows) {
#if SECP256K1_DEBUG_GLV
    auto load_start = std::chrono::steady_clock::now();
#endif
    
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // Read and validate header
    CacheHeader header{};
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!file.good() || header.magic != CACHE_MAGIC || header.version != CACHE_VERSION) {
        return false;
    }
    
#if SECP256K1_DEBUG_GLV
    std::cout << "[CACHE] Loading cache: " << path << std::endl;
    std::cout << "[CACHE] Header: window_bits=" << header.window_bits 
              << " window_count=" << header.window_count 
              << " digit_count=" << header.digit_count
              << " has_glv=" << header.has_glv << std::endl;
#endif
    
    // Create new context from cache
    auto ctx = std::make_unique<PrecomputeContext>();
    ctx->window_bits = header.window_bits;
    ctx->window_count = header.window_count;
    ctx->digit_count = header.digit_count;
    ctx->config.window_bits = header.window_bits;
    ctx->config.enable_glv = (header.has_glv != 0);
    ctx->beta = FieldElement::from_bytes(kBetaBytes);
    
#if SECP256K1_DEBUG_GLV
    std::cout << "[CACHE] Context GLV enabled: " << ctx->config.enable_glv << std::endl;
#endif
    
    // Determine how many windows to load
    std::size_t windows_to_load = ctx->window_count;
    if (max_windows > 0 && max_windows < ctx->window_count) {
        windows_to_load = max_windows;
    }
    
#if SECP256K1_DEBUG_GLV
    std::cout << "[CACHE] Loading " << windows_to_load << " of " << ctx->window_count << " windows..." << std::endl;
#endif
    
    // Allocate tables
    ctx->base_tables.resize(windows_to_load);
    if (ctx->config.enable_glv) {
        ctx->psi_tables.resize(windows_to_load);
    }
    
    // Read base tables
    for (std::size_t w = 0; w < ctx->window_count; ++w) {
        bool should_load = (w < windows_to_load);
        
        if (should_load) {
            ctx->base_tables[w].resize(ctx->digit_count);
        }
        
        for (std::size_t d = 0; d < ctx->digit_count; ++d) {
            if (should_load) {
                if (!read_affine_point(file, ctx->base_tables[w][d])) {
                    return false;
                }
            } else {
                // Skip this point
                std::uint8_t infinity_byte;
                file.read(reinterpret_cast<char*>(&infinity_byte), 1);
                if (!file.good()) return false;
                
                if (infinity_byte == 0) {
                    // Skip x and y coordinates (2 * 32 bytes)
                    file.seekg(64, std::ios::cur);
                    if (!file.good()) return false;
                }
            }
        }
    }
    
    // Read psi tables if GLV enabled
    if (header.has_glv) {
        for (std::size_t w = 0; w < ctx->window_count; ++w) {
            bool should_load = (w < windows_to_load);
            
            if (should_load) {
                ctx->psi_tables[w].resize(ctx->digit_count);
            }
            
            for (std::size_t d = 0; d < ctx->digit_count; ++d) {
                if (should_load) {
                    if (!read_affine_point(file, ctx->psi_tables[w][d])) {
                        return false;
                    }
                } else {
                    // Skip this point
                    std::uint8_t infinity_byte;
                    file.read(reinterpret_cast<char*>(&infinity_byte), 1);
                    if (!file.good()) return false;
                    
                    if (infinity_byte == 0) {
                        file.seekg(64, std::ios::cur);
                        if (!file.good()) return false;
                    }
                }
            }
        }
    }
    
    file.close();
    
    // Install the loaded context
    g_context = std::move(ctx);
    
#if SECP256K1_DEBUG_GLV
    auto load_end = std::chrono::steady_clock::now();
    auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start).count();
    std::cout << "[CACHE] Cache loaded in " << load_ms << " ms" << std::endl;
#endif
    
    return true;
}

void ensure_built_locked() {
    if (!g_context) {
        // Try to load from cache if enabled
        if (g_config.use_cache) {
            std::string cache_path = g_config.cache_path;
            if (cache_path.empty()) {
                cache_path = get_default_cache_path(g_config.window_bits);
            }
            
            // Try to load existing cache (using _locked version since we already have the mutex)
            if (load_precompute_cache_locked(cache_path, g_config.max_windows_to_load)) {
                // Successfully loaded from cache
                return;
            }
            
            // Cache load failed, use in-memory generation
            g_context = build_context(g_config);
            
            // Save to cache for next time
            save_precompute_cache_locked(cache_path);
        } else {
            // Cache disabled, just build in memory
            g_context = build_context(g_config);
        }
    }
}

} // namespace

// ----------------------------------------------------------------------------
// Library-level config helpers (INI-style file)
// ----------------------------------------------------------------------------

static inline std::string trim_copy(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r' || s[a] == '\n')) ++a;
    while (b > a && (s[b-1] == ' ' || s[b-1] == '\t' || s[b-1] == '\r' || s[b-1] == '\n')) --b;
    return s.substr(a, b - a);
}

static inline bool parse_bool(const std::string& v, bool& out) {
    std::string t = v; std::transform(t.begin(), t.end(), t.begin(), ::tolower);
    if (t == "1" || t == "true" || t == "yes" || t == "on" || t == "y" || t == "t") { out = true; return true; }
    if (t == "0" || t == "false" || t == "no" || t == "off" || t == "n" || t == "f") { out = false; return true; }
    return false;
}

static inline bool parse_uint(const std::string& v, unsigned& out) {
    char* end = nullptr;
    unsigned long val = std::strtoul(v.c_str(), &end, 10);
    if (end == v.c_str() || *end != '\0') return false;
    out = static_cast<unsigned>(val);
    return true;
}

bool load_fixed_base_config_file(const std::string& path, FixedBaseConfig& out) {
    std::ifstream in(path);
    if (!in.is_open()) return false;

    // Start from existing defaults
    FixedBaseConfig cfg = out;

    std::string line;
    while (std::getline(in, line)) {
        std::string s = trim_copy(line);
        if (s.empty() || s[0] == '#' || s[0] == ';') continue;
        auto pos = s.find('=');
        if (pos == std::string::npos) continue;
        std::string key = trim_copy(s.substr(0, pos));
        std::string val = trim_copy(s.substr(pos + 1));
        std::transform(key.begin(), key.end(), key.begin(), ::tolower);

        if (key == "cache_dir") {
            cfg.cache_dir = val;
        } else if (key == "cache_path") {
            cfg.cache_path = val;
        } else if (key == "window_bits") {
            unsigned u; if (parse_uint(val, u)) cfg.window_bits = u;
        } else if (key == "enable_glv") {
            bool b; if (parse_bool(val, b)) cfg.enable_glv = b;
        } else if (key == "adaptive_glv") {
            bool b; if (parse_bool(val, b)) cfg.adaptive_glv = b;
        } else if (key == "glv_min_window_bits") {
            unsigned u; if (parse_uint(val, u)) cfg.glv_min_window_bits = u;
        } else if (key == "use_jsf") {
            bool b; if (parse_bool(val, b)) cfg.use_jsf = b;
        } else if (key == "use_cache") {
            bool b; if (parse_bool(val, b)) cfg.use_cache = b;
        } else if (key == "max_windows") {
            unsigned u; if (parse_uint(val, u)) cfg.max_windows_to_load = u;
        } else if (key == "thread_count") {
            unsigned u; if (parse_uint(val, u)) cfg.thread_count = u;
        } else if (key == "use_comb") {
            bool b; if (parse_bool(val, b)) cfg.use_comb = b;
        } else if (key == "comb_width") {
            unsigned u; if (parse_uint(val, u)) cfg.comb_width = u;
        } else if (key == "autotune") {
            bool b; if (parse_bool(val, b)) cfg.autotune = b;
        } else if (key == "autotune_iters") {
            unsigned u; if (parse_uint(val, u)) cfg.autotune_iters = u;
        } else if (key == "autotune_min_w") {
            unsigned u; if (parse_uint(val, u)) cfg.autotune_min_w = u;
        } else if (key == "autotune_max_w") {
            unsigned u; if (parse_uint(val, u)) cfg.autotune_max_w = u;
        } else if (key == "autotune_log") {
            cfg.autotune_log_path = val;
        }
    }

    out = cfg;
    return true;
}

bool configure_fixed_base_from_file(const std::string& path) {
    FixedBaseConfig cfg{};
    if (!load_fixed_base_config_file(path, cfg)) return false;
    configure_fixed_base(cfg);
    return true;
}

bool configure_fixed_base_from_env() {
    // Environment-based configuration disabled; always return false.
    return false;
}

bool write_default_fixed_base_config(const std::string& path) {
    // Do not overwrite if exists
    {
        std::ifstream in(path);
        if (in.is_open()) return false;
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) return false;
    out << "# secp256k1-fast library configuration (auto-generated)\n";
    out << "# Lines beginning with '#' or ';' are comments.\n";
    out << "# Edit values to match your environment.\n\n";
    out << "# Directory holding precomputed tables (cache_w{bits}[ _glv].bin)\n";
    out << "cache_dir=F:\\EccTables\n\n";
    out << "# Fixed-base window size (higher = larger cache, faster generator mul). Typical: 16-20\n";
    out << "window_bits=18\n\n";
    out << "# GLV path and JSF recoding (affects generator mul internals)\n";
    out << "enable_glv=false\n";
    out << "adaptive_glv=true\n";
    out << "glv_min_window_bits=14\n";
    out << "use_jsf=true\n\n";
    out << "# Cache behavior\n";
    out << "use_cache=true\n";
    out << "max_windows=0\n\n";
    out << "# Optional advanced settings\n";
    out << "thread_count=0\n";
    out << "use_comb=false\n";
    out << "comb_width=6\n";
    out << "\n# Auto-tune on first run (set to true, then it will be disabled after tuning)\n";
    out << "autotune=false\n";
    out << "autotune_iters=2000\n";
    out << "autotune_min_w=2\n";
    out << "autotune_max_w=20\n";
    out << "autotune_log=autotune.log\n";
    out.close();
    return true;
}

bool ensure_fixed_base_config_file(const std::string& path) {
    {
        std::ifstream in(path);
        if (in.is_open()) return true;
    }
    return write_default_fixed_base_config(path);
}

bool configure_fixed_base_auto() {
    const std::string path = "config.ini";
    ensure_fixed_base_config_file(path);
    // Load current settings
    FixedBaseConfig cfg{};
    load_fixed_base_config_file(path, cfg);
    // If autotune requested, perform it, then disable autotune and persist
    if (cfg.autotune) {
        std::string report;
        if (auto_tune_fixed_base(cfg, &report, cfg.autotune_iters, cfg.autotune_min_w, cfg.autotune_max_w)) {
            // Append timestamp and final choice to log if configured
            if (!cfg.autotune_log_path.empty()) {
                try {
                    std::ofstream log(cfg.autotune_log_path, std::ios::app);
                    if (log.is_open()) {
                        auto now = std::chrono::system_clock::now();
                        std::time_t tt = std::chrono::system_clock::to_time_t(now);
                        std::tm tm_buf{};
#if defined(_WIN32)
                        localtime_s(&tm_buf, &tt);
#else
                        localtime_r(&tt, &tm_buf);
#endif
                        log << "==== AUTOTUNE RUN " << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S") << " ====" << '\n';
                        log << report;
                        log << "Selected: window_bits=" << cfg.window_bits
                            << " glv=" << (cfg.enable_glv?"true":"false")
                            << " jsf=" << (cfg.use_jsf?"true":"false") << '\n';
                        log << "---------------------------------------------\n";
                    }
                } catch (...) {
                    // Ignore logging errors
                }
            }
            cfg.autotune = false; // disable after first run
            write_fixed_base_config(path, cfg);
        }
    }
    configure_fixed_base(cfg);
    return true;
}

// ----------------------------------------------------------------------------
// Auto-tuning implementation
// ----------------------------------------------------------------------------

namespace {

struct TuneCandidate {
    unsigned window_bits{0};
    bool enable_glv{false};
    bool use_jsf{false}; // Only relevant when enable_glv=true
};

static inline std::string bool_str(bool v) { return v ? "true" : "false"; }

// Console progress callback for table generation
static void autotune_progress_cb(size_t current_points, size_t total_points,
                                 unsigned window_index, unsigned total_windows) {
    double pct = total_points ? (100.0 * static_cast<double>(current_points) / static_cast<double>(total_points)) : 0.0;
    std::cout << "  [gen] window " << window_index << "/" << total_windows
              << ", points " << current_points << "/" << total_points
              << " (" << std::fixed << std::setprecision(1) << pct << "%)\r";
    std::cout.flush();
}

// Measure average nanoseconds per generator multiplication for a given config
static double measure_ns_per_mul(const FixedBaseConfig& cfg,
                                 unsigned iterations,
                                 unsigned warmup = 200) {
    configure_fixed_base(cfg);
    ensure_fixed_base_ready();

    // Deterministic pseudo-random scalars to avoid RNG overhead
    std::array<Scalar, 1024> scalars{};
    for (size_t i = 0; i < scalars.size(); ++i) {
        // Simple variation based on i
        std::array<std::uint8_t, 32> b{};
        for (size_t j = 0; j < 32; ++j) b[j] = static_cast<std::uint8_t>((i * 1315423911u + j * 2654435761u) >> (j % 16));
        scalars[i] = Scalar::from_bytes(b);
    }

    auto run_once = [&scalars](unsigned iters) {
        Point acc; // unused output, but keep it local to avoid DCE
        size_t idx = 0;
        for (unsigned i = 0; i < iters; ++i) {
            const Scalar& k = scalars[idx & (scalars.size() - 1)];
            acc = scalar_mul_generator(k);
            ++idx;
        }
        return acc.infinity; // prevent optimizing away
    };

    // Warm-up
    run_once(warmup);

    const auto t0 = std::chrono::high_resolution_clock::now();
    run_once(iterations);
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    return ns / static_cast<double>(iterations);
}

// Discover available windows and GLV variants by scanning cache_dir
static std::vector<TuneCandidate> discover_candidates(const FixedBaseConfig& base_cfg, unsigned min_w, unsigned max_w) {
    std::vector<TuneCandidate> out;
    namespace fs = std::filesystem;

    std::string dir = base_cfg.cache_dir.empty() ? std::string("F:\\EccTables") : base_cfg.cache_dir;
    std::error_code ec;
    if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec)) {
        return out;
    }

    // Map: window_bits -> {has_non_glv, has_glv}
    struct Flags { bool non_glv{false}; bool glv{false}; };
    std::unordered_map<unsigned, Flags> windows;

    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        if (ec) break;
        if (!it->is_regular_file()) continue;
        const auto p = it->path();
        const auto fname = p.filename().string();
        // Expected: cache_w{bits}.bin or cache_w{bits}_glv.bin
        if (fname.rfind("cache_w", 0) == 0 && p.extension() == ".bin") {
            // Extract digits after 'cache_w'
            size_t pos = std::string("cache_w").size();
            size_t end = pos;
            while (end < fname.size() && isdigit(static_cast<unsigned char>(fname[end]))) ++end;
            if (end == pos) continue;
            unsigned wb = static_cast<unsigned>(std::strtoul(fname.substr(pos, end - pos).c_str(), nullptr, 10));
            bool is_glv = (fname.find("_glv", end) != std::string::npos);
            auto& f = windows[wb];
            if (is_glv) f.glv = true; else f.non_glv = true;
        }
    }

    for (const auto& kv : windows) {
        const unsigned wb = kv.first;
        const Flags f = kv.second;
        if (wb < min_w || wb > max_w) continue;
        if (f.non_glv) {
            out.push_back(TuneCandidate{wb, false, false});
        }
        if (f.glv) {
            out.push_back(TuneCandidate{wb, true, false}); // Shamir
            out.push_back(TuneCandidate{wb, true, true});  // JSF
        }
    }
    // Sort by window_bits ascending (optional)
    std::sort(out.begin(), out.end(), [](const TuneCandidate& a, const TuneCandidate& b) {
        if (a.window_bits != b.window_bits) return a.window_bits < b.window_bits;
        if (a.enable_glv != b.enable_glv) return a.enable_glv < b.enable_glv;
        return a.use_jsf < b.use_jsf;
    });
    return out;
}

} // namespace

bool auto_tune_fixed_base(FixedBaseConfig& best_out,
                          std::string* report_out,
                          unsigned iterations,
                          unsigned min_w,
                          unsigned max_w) {
    FixedBaseConfig base = best_out; // start from provided/default
    if (base.cache_dir.empty()) base.cache_dir = "F:\\EccTables";
    base.use_cache = true;
    base.max_windows_to_load = 0;

    const auto candidates = discover_candidates(base, min_w, max_w);
    if (candidates.empty()) {
        if (report_out) *report_out = "No cache files found for auto-tuning.";
        return false;
    }

    double best_ns = std::numeric_limits<double>::infinity();
    FixedBaseConfig best_cfg = base;

    std::string report;
    report += "Auto-tune candidates (iterations=" + std::to_string(iterations) + ")\n";
    std::cout << "Discovered " << candidates.size() << " candidates in '" << base.cache_dir << "'." << std::endl;

    // Evaluate each candidate
    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto& c = candidates[i];
        FixedBaseConfig cfg = base;
        cfg.window_bits = c.window_bits;
        cfg.enable_glv = c.enable_glv;
        cfg.use_jsf = c.use_jsf;
        cfg.progress_callback = &autotune_progress_cb; // show generation progress if needed

        // Compute expected cache file path for info
        std::string filename = "cache_w" + std::to_string(cfg.window_bits);
        if (cfg.enable_glv) filename += "_glv";
        filename += ".bin";
        std::string cache_path = cfg.cache_dir.empty() ? filename : (cfg.cache_dir + "\\" + filename);
        bool cache_exists = std::filesystem::exists(cache_path);
        std::cout << "[" << (i+1) << "/" << candidates.size() << "]"
                  << " w=" << cfg.window_bits
                  << ", glv=" << (cfg.enable_glv?"true":"false")
                  << ", jsf=" << (cfg.use_jsf?"true":"false")
                  << ": " << (cache_exists?"loading cache":"generating tables (first run)")
                  << " -> " << cache_path << std::endl;
        if (!cache_exists) {
            std::cout << "  skipped (cache file not found; autotune does not generate tables)." << std::endl;
            continue;
        }
        // Pin exact cache path to avoid accidental fallback
        cfg.cache_path = cache_path;

        double ns = 0.0;
        try {
            ns = measure_ns_per_mul(cfg, iterations);
        } catch (...) {
            // If something fails (e.g., missing cache), skip
            std::cout << "  skipped due to error while preparing candidate." << std::endl;
            continue;
        }

        report += "  w=" + std::to_string(cfg.window_bits)
               +  ", glv=" + bool_str(cfg.enable_glv)
               +  ", jsf=" + bool_str(cfg.use_jsf)
               +  ": " + std::to_string(ns) + " ns\n";
        std::cout << "  result: " << std::fixed << std::setprecision(2) << ns << " ns/op" << std::endl;

        if (ns < best_ns) {
            best_ns = ns;
            best_cfg = cfg;
        }
    }

    if (!std::isfinite(best_ns)) {
        if (report_out) *report_out = report + "No valid candidates executed.";
        return false;
    }

    if (report_out) {
        report += "Best => w=" + std::to_string(best_cfg.window_bits)
               +  ", glv=" + bool_str(best_cfg.enable_glv)
               +  ", jsf=" + bool_str(best_cfg.use_jsf)
               +  ", avg=" + std::to_string(best_ns) + " ns\n";
        *report_out = std::move(report);
    }

    best_out = best_cfg;
    return true;
}

bool write_fixed_base_config(const std::string& path, const FixedBaseConfig& cfg) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) return false;
    out << "# secp256k1-fast library configuration (auto-tuned)\n";
    out << "# Generated on this machine for best fixed-base performance.\n\n";
    out << "cache_dir=" << (cfg.cache_dir.empty() ? std::string("F:\\EccTables") : cfg.cache_dir) << "\n";
    if (!cfg.cache_path.empty()) {
        out << "cache_path=" << cfg.cache_path << "\n";
    }
    out << "window_bits=" << cfg.window_bits << "\n";
    out << "enable_glv=" << (cfg.enable_glv ? "true" : "false") << "\n";
    out << "adaptive_glv=" << (cfg.adaptive_glv ? "true" : "false") << "\n";
    out << "glv_min_window_bits=" << cfg.glv_min_window_bits << "\n";
    out << "use_jsf=" << (cfg.use_jsf ? "true" : "false") << "\n";
    out << "use_cache=" << (cfg.use_cache ? "true" : "false") << "\n";
    out << "max_windows=" << cfg.max_windows_to_load << "\n";
    out << "thread_count=" << cfg.thread_count << "\n";
    out << "use_comb=" << (cfg.use_comb ? "true" : "false") << "\n";
    out << "comb_width=" << cfg.comb_width << "\n";
    out << "\n# Auto-tune on first run (set to true, then it will be disabled after tuning)\n";
    out << "autotune=false\n";
    out << "autotune_iters=2000\n";
    out << "autotune_min_w=2\n";
    out << "autotune_max_w=20\n";
    out << "autotune_log=autotune.log\n";
    out.close();
    return true;
}

bool auto_tune_and_write_config(const std::string& path, unsigned iterations, unsigned min_w, unsigned max_w) {
    FixedBaseConfig guess{}; // defaults
    std::string report;
    if (!auto_tune_fixed_base(guess, &report, iterations, min_w, max_w)) {
        return false;
    }
    // Persist
    if (!write_fixed_base_config(path, guess)) return false;
    // Apply immediately
    configure_fixed_base(guess);
    return true;
}

// ============================================================================
// wNAF (width-w Non-Adjacent Form) Implementation
// ============================================================================
// wNAF reduces the number of non-zero digits by 30-50% compared to standard
// signed window methods. This is achieved through the non-adjacent property:
// no two consecutive digits are non-zero.
//
// Algorithm from "Guide to Elliptic Curve Cryptography" (Hankerson et al.)
// Properties:
// - Digits in range: [-(2^(w-1)-1), 2^(w-1)-1], only odd values
// - Non-adjacent: d[i] != 0 => d[i+1] = 0
// - Minimal Hamming weight (fewest non-zero digits)
// - Fast computation (O(n) where n = bit length)
//
// Performance impact:
// - Reduces point additions by 30-50%
// - Increases point doublings by 0-10% (due to variable spacing)
// - Net speedup: 15-25% for scalar multiplication
// ============================================================================

std::vector<int32_t> compute_wnaf(const Scalar& scalar, unsigned window_bits) {
    if (window_bits < 2U || window_bits > 16U) {
        throw std::runtime_error("wNAF window size must be between 2 and 16");
    }
    
    // Maximum possible wNAF length is bit_length + 1 (due to carries)
    const std::size_t max_length = 257; // 256-bit scalar + 1 for carry
    std::vector<int32_t> wnaf;
    wnaf.reserve(max_length);
    
    // Copy scalar to working array (5 limbs to handle carries)
    std::array<std::uint64_t, 5> k{};
    const auto& limbs = scalar.limbs();
    for (std::size_t i = 0; i < 4; ++i) {
        k[i] = limbs[i];
    }
    k[4] = 0;
    
    const int32_t window_size = 1 << window_bits;        // 2^w
    const int32_t window_mask = window_size - 1;         // 2^w - 1
    const int32_t half_window = window_size >> 1;        // 2^(w-1)
    
    std::size_t bit_pos = 0;
    
    // Process scalar bit by bit
    while (bit_pos < 256 || k[0] != 0 || k[1] != 0 || k[2] != 0 || k[3] != 0) {
        int32_t digit = 0;
        
        // If k is odd, extract w bits and make adjustment
        if (k[0] & 1ULL) {
            // Extract w least significant bits
            const int32_t chunk = static_cast<int32_t>(k[0] & window_mask);
            
            // If chunk >= 2^(w-1), use negative representation
            if (chunk >= half_window) {
                digit = chunk - window_size;
                
                // Subtract digit from k (add |digit| since digit is negative)
                // This creates the non-adjacent property
                const std::uint64_t add_val = static_cast<std::uint64_t>(-digit);
                unsigned char carry = 0;
                std::uint64_t tmp;
                carry = COMPAT_ADDCARRY_U64(carry, k[0], add_val, &tmp); k[0] = tmp;
                carry = COMPAT_ADDCARRY_U64(carry, k[1], 0ULL, &tmp); k[1] = tmp;
                carry = COMPAT_ADDCARRY_U64(carry, k[2], 0ULL, &tmp); k[2] = tmp;
                carry = COMPAT_ADDCARRY_U64(carry, k[3], 0ULL, &tmp); k[3] = tmp;
                carry = COMPAT_ADDCARRY_U64(carry, k[4], 0ULL, &tmp); k[4] = tmp;
            } else {
                digit = chunk;
                
                // Subtract digit from k
                unsigned char borrow = 0;
                std::uint64_t tmp;
                borrow = COMPAT_SUBBORROW_U64(borrow, k[0], static_cast<std::uint64_t>(digit), &tmp); k[0] = tmp;
                borrow = COMPAT_SUBBORROW_U64(borrow, k[1], 0ULL, &tmp); k[1] = tmp;
                borrow = COMPAT_SUBBORROW_U64(borrow, k[2], 0ULL, &tmp); k[2] = tmp;
                borrow = COMPAT_SUBBORROW_U64(borrow, k[3], 0ULL, &tmp); k[3] = tmp;
                borrow = COMPAT_SUBBORROW_U64(borrow, k[4], 0ULL, &tmp); k[4] = tmp;
            }
        }
        
        wnaf.push_back(digit);
        
        // Right shift k by 1 bit
        k[0] = (k[0] >> 1) | (k[1] << 63);
        k[1] = (k[1] >> 1) | (k[2] << 63);
        k[2] = (k[2] >> 1) | (k[3] << 63);
        k[3] = (k[3] >> 1) | (k[4] << 63);
        k[4] >>= 1;

        ++bit_pos;
        
        // Safety check to prevent infinite loops
        if (bit_pos > max_length) {
            throw std::runtime_error("wNAF computation exceeded maximum length");
        }
    }
    
    return wnaf;
}

// No-alloc variant of compute_wnaf: writes into caller-provided buffer
void compute_wnaf_into(const Scalar& scalar,
                       unsigned window_bits,
                       int32_t* out,
                       std::size_t max,
                       std::size_t& out_len) {
    if (window_bits < 2U || window_bits > 16U) {
        throw std::runtime_error("wNAF window size must be between 2 and 16");
    }
    if (out == nullptr || max == 0) {
        throw std::runtime_error("compute_wnaf_into: invalid output buffer");
    }

    // Working scalar in 5-limb array to accommodate carries
    std::array<std::uint64_t, 5> k{};
    const auto& limbs = scalar.limbs();
    for (std::size_t i = 0; i < 4; ++i) {
        k[i] = limbs[i];
    }
    k[4] = 0;

    const int32_t window_size = 1 << window_bits;        // 2^w
    const int32_t window_mask = window_size - 1;         // 2^w - 1
    const int32_t half_window = window_size >> 1;        // 2^(w-1)

    out_len = 0;
    std::size_t bit_pos = 0;
    const std::size_t max_length = 257; // 256-bit scalar + 1 for carry

    while (bit_pos < 256 || k[0] != 0 || k[1] != 0 || k[2] != 0 || k[3] != 0) {
        int32_t digit = 0;

        if (k[0] & 1ULL) {
            const int32_t chunk = static_cast<int32_t>(k[0] & window_mask);
            if (chunk >= half_window) {
                digit = chunk - window_size;
                const std::uint64_t add_val = static_cast<std::uint64_t>(-digit);
                unsigned char carry = 0;
                std::uint64_t tmp;
                carry = COMPAT_ADDCARRY_U64(carry, k[0], add_val, &tmp); k[0] = tmp;
                carry = COMPAT_ADDCARRY_U64(carry, k[1], 0ULL, &tmp); k[1] = tmp;
                carry = COMPAT_ADDCARRY_U64(carry, k[2], 0ULL, &tmp); k[2] = tmp;
                carry = COMPAT_ADDCARRY_U64(carry, k[3], 0ULL, &tmp); k[3] = tmp;
                carry = COMPAT_ADDCARRY_U64(carry, k[4], 0ULL, &tmp); k[4] = tmp;
            } else {
                digit = chunk;
                unsigned char borrow = 0;
                std::uint64_t tmp;
                borrow = COMPAT_SUBBORROW_U64(borrow, k[0], static_cast<std::uint64_t>(digit), &tmp); k[0] = tmp;
                borrow = COMPAT_SUBBORROW_U64(borrow, k[1], 0ULL, &tmp); k[1] = tmp;
                borrow = COMPAT_SUBBORROW_U64(borrow, k[2], 0ULL, &tmp); k[2] = tmp;
                borrow = COMPAT_SUBBORROW_U64(borrow, k[3], 0ULL, &tmp); k[3] = tmp;
                borrow = COMPAT_SUBBORROW_U64(borrow, k[4], 0ULL, &tmp); k[4] = tmp;
            }
        }

        if (out_len >= max) {
            throw std::runtime_error("compute_wnaf_into: output buffer too small");
        }
        out[out_len++] = digit;

        // Right shift k by 1 bit
        k[0] = (k[0] >> 1) | (k[1] << 63);
        k[1] = (k[1] >> 1) | (k[2] << 63);
        k[2] = (k[2] >> 1) | (k[3] << 63);
        k[3] = (k[3] >> 1) | (k[4] << 63);
        k[4] >>= 1;

        ++bit_pos;
        if (bit_pos > max_length) {
            #if !SECP256K1_ESP32_BUILD
            throw std::runtime_error("wNAF computation exceeded maximum length");
            #else
            break;
            #endif
        }
    }
}

#if SECP256K1_ESP32_BUILD
// ESP32 simplified version - no mutex, no environment variables, minimal features
void configure_fixed_base(const FixedBaseConfig& config) {
    g_config = config;
    // Adaptive GLV override
    if (g_config.adaptive_glv && g_config.enable_glv && g_config.window_bits < g_config.glv_min_window_bits) {
        g_config.enable_glv = false;
    }
    g_context.reset();
}

void ensure_fixed_base_ready() {
    if (!g_context) {
        g_context = build_context(g_config);
    }
}

bool fixed_base_ready() {
    return static_cast<bool>(g_context);
}
#else
// Desktop version with full features
void configure_fixed_base(const FixedBaseConfig& config) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_config = config;

    // Adaptive GLV override: if enabled and window_bits below threshold, disable GLV.
    if (g_config.adaptive_glv && g_config.enable_glv && g_config.window_bits < g_config.glv_min_window_bits) {
        g_config.enable_glv = false; // Effective disable due to insufficient window size
    }

    // Environment overrides for external configuration
    // SECP256K1_CACHE_DIR  -> overrides cache directory containing cache_w{bits}[ _glv].bin
    // SECP256K1_CACHE_PATH -> overrides exact cache file path
    // SECP256K1_MAX_WINDOWS -> limits how many windows to load from cache (for memory control)
    if (const char* env_dir = std::getenv("SECP256K1_CACHE_DIR")) {
        if (*env_dir && std::string(env_dir).find("..") == std::string::npos) { // lgtm[cpp/path-injection]
            g_config.cache_dir = env_dir;
        }
    }
    if (const char* env_path = std::getenv("SECP256K1_CACHE_PATH")) {
        if (*env_path && std::string(env_path).find("..") == std::string::npos) { // lgtm[cpp/path-injection]
            g_config.cache_path = env_path;
        }
    }
    if (const char* env_maxw = std::getenv("SECP256K1_MAX_WINDOWS")) {
        unsigned v = static_cast<unsigned>(std::strtoul(env_maxw, nullptr, 10));
        if (v > 0U) {
            g_config.max_windows_to_load = v;
        }
    }
    g_context.reset();
}

void ensure_fixed_base_ready() {
    std::lock_guard<std::mutex> lock(g_mutex);
    ensure_built_locked();
}

bool fixed_base_ready() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return static_cast<bool>(g_context);
}
#endif // !SECP256K1_ESP32_BUILD

ScalarDecomposition split_scalar_glv(const Scalar& scalar) {
    return split_scalar_internal(scalar);
}

// Shamir's trick: Simultaneous 2D scalar multiplication k1*P + k2*Q
// Uses interleaved windowed method for optimal performance with precomputed tables
namespace {

// Compute k1*P + k2*Q simultaneously using Shamir's trick
// Processes both digit streams in one pass, eliminating the final point addition
JacobianPoint shamir_windowed_glv(
    const std::vector<int32_t>& digits1,  // k1 window digits  
    const std::vector<int32_t>& digits2,  // k2 window digits
    const std::vector<std::vector<AffinePointPacked>>& P_tables,  // G tables
    const std::vector<std::vector<AffinePointPacked>>& Q_tables,  // psi(G) tables
    std::size_t window_count
) {
#if SECP256K1_PROFILE_DECOMP
    static std::atomic<uint64_t> total_calls{0};
    static std::atomic<uint64_t> sum_loop_overhead{0};
    static std::atomic<uint64_t> sum_d1_additions{0};
    static std::atomic<uint64_t> sum_d2_additions{0};
    static std::atomic<uint64_t> sum_both_zero{0};
    static std::atomic<uint64_t> count_d1_adds{0};
    static std::atomic<uint64_t> count_d2_adds{0};
    static std::atomic<uint64_t> count_both_zero_skips{0};
    
    uint64_t start_total = RDTSC();
#endif
    
    JacobianPoint result{FieldElement::zero(), FieldElement::one(), FieldElement::zero(), true};
    
    // Process windows from highest to lowest
    // Add both P and Q contributions in the same pass
    for (std::size_t window = 0; window < window_count; ++window) {
#if SECP256K1_PROFILE_DECOMP
        uint64_t t0 = RDTSC();
#endif
        
        int32_t d1 = digits1[window];
        int32_t d2 = digits2[window];
        
        // Skip if both are zero (common case optimization)
        if (d1 == 0 && d2 == 0) {
#if SECP256K1_PROFILE_DECOMP
            uint64_t t_skip = RDTSC();
            sum_both_zero += (t_skip - t0);
            count_both_zero_skips++;
#endif
            continue;
        }
        
        // Prefetch both tables for this window if any digit is non-zero
        // This reduces cache misses when both d1 and d2 are non-zero
        bool d1_nonzero = (d1 != 0);
        bool d2_nonzero = (d2 != 0);
        
        if (d1_nonzero && d2_nonzero) {
            // Both non-zero: prefetch both lookups
            std::size_t idx1 = static_cast<std::size_t>(d1 < 0 ? -d1 : d1);
            std::size_t idx2 = static_cast<std::size_t>(d2 < 0 ? -d2 : d2);
            if (idx1 < P_tables[window].size()) {
                SECP256K1_PREFETCH_READ(&P_tables[window][idx1]);
            }
            if (idx2 < Q_tables[window].size()) {
                SECP256K1_PREFETCH_READ(&Q_tables[window][idx2]);
            }
        }
        
#if SECP256K1_PROFILE_DECOMP
        uint64_t t1 = RDTSC();
#endif
        
        // Add contribution from P (base point G)
        if (d1_nonzero) {
            bool negative = d1 < 0;
            std::size_t index = static_cast<std::size_t>(negative ? -d1 : d1);
            if (index < P_tables[window].size()) {
                const auto& entry = P_tables[window][index];
                if (!entry.infinity) {
                    AffinePointPacked affine_pt = entry;
                    if (negative) {
                        affine_pt.y = negate_fe(affine_pt.y);
                    }
                    result = jacobian_add_mixed_local(result, affine_pt);
#if SECP256K1_PROFILE_DECOMP
                    count_d1_adds++;
#endif
                }
            }
        }
        
#if SECP256K1_PROFILE_DECOMP
        uint64_t t2 = RDTSC();
        if (d1_nonzero) {
            sum_d1_additions += (t2 - t1);
        }
#endif
        
        // Add contribution from Q (endomorphism point psi(G))
        if (d2_nonzero) {
            bool negative = d2 < 0;
            std::size_t index = static_cast<std::size_t>(negative ? -d2 : d2);
            if (index < Q_tables[window].size()) {
                const auto& entry = Q_tables[window][index];
                if (!entry.infinity) {
                    AffinePointPacked affine_pt = entry;
                    if (negative) {
                        affine_pt.y = negate_fe(affine_pt.y);
                    }
                    result = jacobian_add_mixed_local(result, affine_pt);
#if SECP256K1_PROFILE_DECOMP
                    count_d2_adds++;
#endif
                }
            }
        }
        
#if SECP256K1_PROFILE_DECOMP
        uint64_t t3 = RDTSC();
        if (d2_nonzero) {
            sum_d2_additions += (t3 - t2);
        }
        sum_loop_overhead += (t1 - t0);
#endif
    }
    
#if SECP256K1_PROFILE_DECOMP
    uint64_t end_total = RDTSC();
    total_calls++;
    
    if (total_calls % 1000 == 0) {
        uint64_t n = total_calls;
        uint64_t n_d1 = count_d1_adds;
        uint64_t n_d2 = count_d2_adds;
        uint64_t n_skip = count_both_zero_skips;
        
        std::cerr << "\n[SHAMIR PROFILE] After " << n << " calls:\n";
        std::cerr << "  Loop overhead:    " << (sum_loop_overhead / n) << " cycles total\n";
        std::cerr << "  Both-zero skips:  " << (n_skip / n) << " avg, " 
                  << (n_skip > 0 ? sum_both_zero / n_skip : 0) << " cycles/skip\n";
        std::cerr << "  D1 additions:     " << (n_d1 / n) << " avg, " 
                  << (n_d1 > 0 ? sum_d1_additions / n_d1 : 0) << " cycles/add\n";
        std::cerr << "  D2 additions:     " << (n_d2 / n) << " avg, "
                  << (n_d2 > 0 ? sum_d2_additions / n_d2 : 0) << " cycles/add\n";
        std::cerr << "  TOTAL per call:   " << ((end_total - start_total)) << " cycles\n";
        std::cerr.flush();
    }
#endif
    
    return result;
}

// JSF-based Shamir's trick: k1*G + k2*psi(G)
// Uses Joint Sparse Form for reduced non-zero digits
JacobianPoint shamir_jsf_glv(
    const Scalar& k1,
    const Scalar& k2,
    const AffinePointPacked& G,        // Base point
    const AffinePointPacked& psi_G,    // Endomorphism of G
    bool neg1 = false,
    bool neg2 = false
) {
    // Compute JSF encoding
    JSF_Result jsf = compute_jsf(k1, k2);
    
    // Precompute lookup table for {+/-G, +/-psi(G), +/-(G+psi(G)), +/-(G-psi(G))}
    // Index: [u2+1][u1+1] where u1, u2 in {-1, 0, +1}
    JacobianPoint lookup[3][3];
    
    // Convert affine to Jacobian for base points
    JacobianPoint jac_G = affine_to_jacobian(G);
    JacobianPoint jac_psi_G = affine_to_jacobian(psi_G);
    
    // [0][0] = infinity (not used, but fill anyway)
    lookup[1][1] = JacobianPoint{FieldElement::zero(), FieldElement::one(), FieldElement::zero(), true};
    
    // [u2=0] row: only G contributions
    lookup[1][0] = negate_jacobian(jac_G);         // -G
    lookup[1][2] = jac_G;                          // +G
    
    // [u2=-1] row: -psi(G) combinations
    JacobianPoint neg_psi_G = negate_jacobian(jac_psi_G);
    lookup[0][0] = jacobian_add(neg_psi_G, lookup[1][0]); // -psi(G) - G
    lookup[0][2] = jacobian_add(neg_psi_G, jac_G);         // -psi(G) + G
    
    // [u2=+1] row: +psi(G) combinations  
    lookup[2][0] = jacobian_add(jac_psi_G, lookup[1][0]); // +psi(G) - G
    lookup[2][2] = jacobian_add(jac_psi_G, jac_G);        // +psi(G) + G
    
    // Process JSF from high bit to low bit
    JacobianPoint result{FieldElement::zero(), FieldElement::one(), FieldElement::zero(), true};
    
    for (int i = static_cast<int>(jsf.length) - 1; i >= 0; --i) {
        // Double current result
        if (!result.infinity) {
            result = jacobian_double(result);
        }
        
        int8_t u1 = jsf.jsf1[i];
        int8_t u2 = jsf.jsf2[i];
        if (neg1) u1 = static_cast<int8_t>(-u1);
        if (neg2) u2 = static_cast<int8_t>(-u2);
        
        // Skip if both are zero
        if (u1 == 0 && u2 == 0) {
            continue;
        }
        
        // Lookup precomputed point
        JacobianPoint add_pt = lookup[u2 + 1][u1 + 1];
        
        if (result.infinity) {
            result = add_pt;
        } else {
            result = jacobian_add(result, add_pt);
        }
    }
    
    return result;
}

} // anonymous namespace

Point scalar_mul_generator(const Scalar& scalar) {
    std::unique_lock<std::mutex> lock(g_mutex);
    ensure_built_locked();
    PrecomputeContext& ctx = *g_context;
    lock.unlock();

    // PHASE 3 OPTIMIZED (Mixed Jacobian-Affine addition - 8 muls instead of 12)
    // PHASE 4: Added prefetching for next iteration data
    JacobianPoint result{FieldElement::zero(), FieldElement::one(), FieldElement::zero(), true};
    const std::size_t window_count = ctx.window_count;

    auto accumulate = [&](const std::vector<int32_t>& digits, const std::vector<std::vector<AffinePointPacked>>& tables) {
        for (std::size_t window = 0; window < window_count; ++window) {
            int32_t digit = digits[window];
            
            // Phase 4: Prefetch next iteration's data (if not last iteration)
            if (window + 1 < window_count) {
                int32_t next_digit = digits[window + 1];
                if (next_digit != 0) {
                    bool next_negative = next_digit < 0;
                    std::size_t next_index = static_cast<std::size_t>(next_negative ? -next_digit : next_digit);
                    if (next_index < tables[window + 1].size()) {
                        // Prefetch next entry into cache
                        SECP256K1_PREFETCH_READ(&tables[window + 1][next_index]);
                    }
                }
            }
            
            if (digit == 0) {
                continue;
            }
            bool negative = digit < 0;
            std::size_t index = static_cast<std::size_t>(negative ? -digit : digit);
            if (index >= tables[window].size()) {
                throw std::runtime_error("Digit index out of range during accumulation");
            }
            const auto& entry = tables[window][index];
            if (entry.infinity) {
                continue;
            }
            
            // Phase 3: Use mixed addition: Jacobian + Affine -> Jacobian (8 muls instead of 12)
            AffinePointPacked affine_pt = entry;
            if (negative) {
                affine_pt.y = negate_fe(affine_pt.y);
            }
            result = jacobian_add_mixed_local(result, affine_pt);
        }
    };

    if (ctx.config.enable_glv) {
#if SECP256K1_DEBUG_GLV
        std::cout << "[GLV] Splitting scalar for GLV path..." << std::endl;
#endif
        ScalarDecomposition decomposition = split_scalar_internal(scalar);
        
#if SECP256K1_DEBUG_GLV
        auto k1_bytes = decomposition.k1.to_bytes();
        auto k2_bytes = decomposition.k2.to_bytes();
        std::cout << "[GLV] k1 (mag): ";
        for (auto b : k1_bytes) std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)b;
        std::cout << " neg=" << decomposition.neg1 << std::endl;
        std::cout << "[GLV] k2 (mag): ";
        for (auto b : k2_bytes) std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)b;
        std::cout << std::dec << " neg=" << decomposition.neg2 << std::endl;
#endif
        
        // Build signed scalars from absolute magnitudes + flags
        // Use absolute values for digit computation
        auto digits1 = compute_window_digits(decomposition.k1, ctx.window_bits, window_count);
        auto digits2 = compute_window_digits(decomposition.k2, ctx.window_bits, window_count);
        
#if SECP256K1_PROFILE_DECOMP
        // Count non-zero digits
        int nonzero1 = 0, nonzero2 = 0;
        for (auto d : digits1) if (d != 0) nonzero1++;
        for (auto d : digits2) if (d != 0) nonzero2++;
        
        static std::atomic<uint64_t> sum_nonzero1{0};
        static std::atomic<uint64_t> sum_nonzero2{0};
        static std::atomic<uint64_t> digit_calls{0};
        sum_nonzero1 += nonzero1;
        sum_nonzero2 += nonzero2;
        digit_calls++;
        
        if (digit_calls % 1000 == 0) {
            std::cerr << "[DIGIT STATS] After " << digit_calls << " decompositions:\n";
            std::cerr << "  Avg non-zero digits1: " << (sum_nonzero1 / digit_calls) << "\n";
            std::cerr << "  Avg non-zero digits2: " << (sum_nonzero2 / digit_calls) << "\n";
            std::cerr << "  Bitlen k1: " << fast_bitlen(decomposition.k1) << "\n";
            std::cerr << "  Bitlen k2: " << fast_bitlen(decomposition.k2) << "\n";
            std::cerr.flush();
        }
#endif
        
        // If neg flag is set, negate ALL digits (so accumulate will negate all points)
        if (decomposition.neg1) {
            for (auto& d : digits1) {
                if (d != 0) d = -d;
            }
        }
        if (decomposition.neg2) {
            for (auto& d : digits2) {
                if (d != 0) d = -d;
            }
        }
        
#if SECP256K1_DEBUG_GLV
        std::cout << "[GLV] Using Shamir's trick for simultaneous k1*G + k2*psi(G)..." << std::endl;
#endif
        
        // *** SHAMIR'S TRICK: Simultaneous 2D multiplication ***
        // Choose between windowed interleaving (uses precomputed tables) or JSF-based variant
        if (ctx.config.use_jsf) {
            // JSF path: small fixed lookup using G and psi(G)
            AffinePointPacked aG = to_affine(Point::generator());
            AffinePointPacked aPsiG = to_affine(apply_endomorphism(Point::generator()));
            result = shamir_jsf_glv(decomposition.k1, decomposition.k2, aG, aPsiG, decomposition.neg1, decomposition.neg2);
        } else {
            // Windowed interleaving path with precomputed tables
            // Process both digit streams simultaneously in one pass.
            result = shamir_windowed_glv(digits1, digits2, ctx.base_tables, ctx.psi_tables, window_count);
        }
        
#if SECP256K1_DEBUG_GLV
        std::cout << "[GLV] Shamir GLV multiplication complete." << std::endl;
#endif
    } else {
#if SECP256K1_DEBUG_GLV
        std::cout << "[GLV] GLV disabled, using standard path." << std::endl;
#endif
        auto digits = compute_window_digits(scalar, ctx.window_bits, window_count);
        accumulate(digits, ctx.base_tables);
    }

    // Phase 3: Convert Jacobian back to Point
    return Point::from_jacobian_coords(result.x, result.y, result.z, result.infinity);
}

// ============================================================================
// Missing API Implementations (Restored)
// ============================================================================

Point scalar_mul_generator_glv_predecomposed(const Scalar& k1, const Scalar& k2, bool neg1, bool neg2) {
    std::unique_lock<std::mutex> lock(g_mutex);
    ensure_built_locked();
    PrecomputeContext& ctx = *g_context;

    // Direct GLV combination using pre-split scalars
    const std::size_t window_count = ctx.window_count;

    auto digits1 = compute_window_digits(k1, ctx.window_bits, window_count);
    auto digits2 = compute_window_digits(k2, ctx.window_bits, window_count);

    if (neg1) { for (auto& d : digits1) if (d != 0) d = -d; }
    if (neg2) { for (auto& d : digits2) if (d != 0) d = -d; }

    JacobianPoint result;
    if (ctx.config.use_jsf) {
        AffinePointPacked aG = to_affine(Point::generator());
        AffinePointPacked aPsiG = to_affine(apply_endomorphism(Point::generator()));
        result = shamir_jsf_glv(k1, k2, aG, aPsiG, neg1, neg2);
    } else {
        result = shamir_windowed_glv(digits1, digits2, ctx.base_tables, ctx.psi_tables, window_count);
    }
    return Point::from_jacobian_coords(result.x, result.y, result.z, result.infinity);
}

bool save_precompute_cache(const std::string& path) {
    (void)path;
    return false; // Not implemented on specialized build
}

bool load_precompute_cache(const std::string& path, unsigned max_windows) {
    (void)path; (void)max_windows;
    return false; // Not implemented on specialized build
}

Point scalar_mul_arbitrary(const Point& base, const Scalar& scalar, unsigned window_bits) {
    (void)window_bits; // unused in this simple fallback
    // Simple naive double-and-add (LSB-first)
    Point res = Point::infinity();
    Point temp = base;
    std::array<uint8_t, 32> bits = scalar.to_bytes(); // Big-endian bytes

    for (int i = 0; i < 256; ++i) {
        // Bit i of the scalar: byte (31 - i/8) contains the group, bit (i%8) within that byte
        bool bit = (bits[31 - i / 8] >> (i % 8)) & 1;
        if (bit) {
            res.add_inplace(temp);
        }
        temp.dbl_inplace();
    }
    return res;
}

// ============================================================================
// Precomputed Scalar API for K-constant, Q-variable multiplication
// ============================================================================

// Build odd-multiples table [Q, 3Q, 5Q, 7Q, ...] as affine points
// table_size = 2^(window_bits - 1), stores odd multiples: (2i+1)*Q for i in [0..table_size)
namespace {
void build_odd_multiples_table(const Point& Q,
                               AffinePointPacked* table,
                               std::size_t table_size) {
    if (table_size == 0) return;

    // table[0] = Q
    table[0] = to_affine(Q);

    if (table_size == 1) return;

    // 2Q in Jacobian (reused for computing 3Q, 5Q, 7Q, ...)
    JacobianPoint jQ = affine_to_jacobian(table[0]);
    JacobianPoint j2Q = jacobian_double(jQ);

    // table[i] = (2i+1)*Q = table[i-1] + 2Q
    JacobianPoint acc = jQ; // starts at Q

    // Collect all Jacobian Z values for batch inversion
    std::vector<JacobianPoint> jac_points;
    jac_points.reserve(table_size);
    jac_points.push_back(jQ); // table[0] = Q (already affine, but include for uniformity)

    for (std::size_t i = 1; i < table_size; ++i) {
        acc = jacobian_add(acc, j2Q); // (2i+1)*Q
        jac_points.push_back(acc);
    }

    // Batch-convert to affine using Montgomery's trick
    std::vector<FieldElement> z_vals;
    z_vals.reserve(table_size);
    for (auto& jp : jac_points) {
        z_vals.push_back(jp.z);
    }
    batch_inverse(z_vals);

    for (std::size_t i = 0; i < table_size; ++i) {
        if (jac_points[i].infinity) {
            table[i] = {FieldElement::zero(), FieldElement::one(), true};
            continue;
        }
        FieldElement z_inv = z_vals[i];
        FieldElement z_inv2 = z_inv;
        z_inv2.square_inplace();
        FieldElement z_inv3 = z_inv2 * z_inv;
        table[i].x = jac_points[i].x * z_inv2;
        table[i].y = jac_points[i].y * z_inv3;
        table[i].infinity = false;
    }
}
} // anonymous namespace

PrecomputedScalar precompute_scalar_for_arbitrary(const Scalar& K, unsigned window_bits) {
    PrecomputedScalar result;
    result.window_bits = window_bits;

    // GLV decomposition: K -> (k_1, k_2) where K = k_1 + lambda*k_2 (mod n)
    ScalarDecomposition decomp = split_scalar_glv(K);
    result.k1 = decomp.k1;
    result.k2 = decomp.k2;
    result.neg1 = decomp.neg1;
    result.neg2 = decomp.neg2;

    // Compute wNAF representation (stored LSB-first)
    result.wnaf1 = compute_wnaf(decomp.k1, window_bits);
    result.wnaf2 = compute_wnaf(decomp.k2, window_bits);

    return result;
}

PrecomputedScalarOptimized precompute_scalar_optimized(const Scalar& K, unsigned window_bits) {
    PrecomputedScalarOptimized result;
    result.window_bits = window_bits;

    // GLV decomposition
    ScalarDecomposition decomp = split_scalar_glv(K);
    result.k1 = decomp.k1;
    result.k2 = decomp.k2;
    result.neg1 = decomp.neg1;
    result.neg2 = decomp.neg2;

    // Compute wNAF for both half-scalars
    auto wnaf1 = compute_wnaf(decomp.k1, window_bits);
    auto wnaf2 = compute_wnaf(decomp.k2, window_bits);

    // Pad to equal length
    std::size_t max_len = std::max(wnaf1.size(), wnaf2.size());
    wnaf1.resize(max_len, 0);
    wnaf2.resize(max_len, 0);

    // Apply GLV signs to digits
    if (decomp.neg1) {
        for (auto& d : wnaf1) { if (d != 0) d = -d; }
    }
    if (decomp.neg2) {
        for (auto& d : wnaf2) { if (d != 0) d = -d; }
    }

    // RLE compress: scan MSB -> LSB (reverse of LSB-first wNAF array)
    // Each Step = {num_doubles, idx1/neg1, idx2/neg2}
    result.steps.reserve(max_len / 2);
    uint16_t pending_doubles = 0;

    for (int i = static_cast<int>(max_len) - 1; i >= 0; --i) {
        int32_t d1 = wnaf1[static_cast<std::size_t>(i)];
        int32_t d2 = wnaf2[static_cast<std::size_t>(i)];

        pending_doubles++; // each position = one doubling

        if (d1 != 0 || d2 != 0) {
            PrecomputedScalarOptimized::Step step;
            step.num_doubles = pending_doubles;

            if (d1 != 0) {
                step.neg1 = (d1 < 0);
                int32_t abs_d = d1 < 0 ? -d1 : d1;
                step.idx1 = static_cast<uint8_t>((abs_d - 1) / 2);
            }
            if (d2 != 0) {
                step.neg2 = (d2 < 0);
                int32_t abs_d = d2 < 0 ? -d2 : d2;
                step.idx2 = static_cast<uint8_t>((abs_d - 1) / 2);
            }

            result.steps.push_back(step);
            pending_doubles = 0;
        }
    }

    // Trailing doubles (zero tail from LSB side)
    if (pending_doubles > 0) {
        PrecomputedScalarOptimized::Step step; // idx1=0xFF, idx2=0xFF -> doubles only
        step.num_doubles = pending_doubles;
        result.steps.push_back(step);
    }

    return result;
}

Point scalar_mul_arbitrary_precomputed(const Point& Q, const PrecomputedScalar& precomp) {
    const unsigned w = precomp.window_bits;
    const std::size_t table_size = std::size_t{1} << (w - 1); // e.g. 8 for w=4

    // Build odd-multiples tables for Q and psi(Q)
    std::vector<AffinePointPacked> q_table(table_size);
    std::vector<AffinePointPacked> psi_table(table_size);

    build_odd_multiples_table(Q, q_table.data(), table_size);

    Point psiQ = apply_endomorphism(Q);
    build_odd_multiples_table(psiQ, psi_table.data(), table_size);

    // Get wNAF digits (LSB-first); pad to equal length
    const auto& wnaf1 = precomp.wnaf1;
    const auto& wnaf2 = precomp.wnaf2;
    std::size_t max_len = std::max(wnaf1.size(), wnaf2.size());

    // Process MSB -> LSB (reverse through the LSB-first arrays)
    JacobianPoint result{FieldElement::zero(), FieldElement::one(), FieldElement::zero(), true};

    for (int i = static_cast<int>(max_len) - 1; i >= 0; --i) {
        // Double
        if (!result.infinity) {
            result = jacobian_double(result);
        }

        auto idx = static_cast<std::size_t>(i);

        // k_1 contribution
        int32_t d1 = (idx < wnaf1.size()) ? wnaf1[idx] : 0;
        if (precomp.neg1 && d1 != 0) d1 = -d1;
        if (d1 != 0) {
            bool neg = (d1 < 0);
            int32_t abs_d = neg ? -d1 : d1;
            std::size_t ti = static_cast<std::size_t>((abs_d - 1) / 2);
            AffinePointPacked pt = q_table[ti];
            if (neg) pt.y = negate_fe(pt.y);
            result = jacobian_add_mixed_local(result, pt);
        }

        // k_2 contribution
        int32_t d2 = (idx < wnaf2.size()) ? wnaf2[idx] : 0;
        if (precomp.neg2 && d2 != 0) d2 = -d2;
        if (d2 != 0) {
            bool neg = (d2 < 0);
            int32_t abs_d = neg ? -d2 : d2;
            std::size_t ti = static_cast<std::size_t>((abs_d - 1) / 2);
            AffinePointPacked pt = psi_table[ti];
            if (neg) pt.y = negate_fe(pt.y);
            result = jacobian_add_mixed_local(result, pt);
        }
    }

    return Point::from_jacobian_coords(result.x, result.y, result.z, result.infinity);
}

Point scalar_mul_arbitrary_precomputed_optimized(const Point& Q,
                                                  const PrecomputedScalarOptimized& precomp) {
    const unsigned w = precomp.window_bits;
    const std::size_t table_size = std::size_t{1} << (w - 1);

    // Build odd-multiples tables for Q and psi(Q)
    std::vector<AffinePointPacked> q_table(table_size);
    std::vector<AffinePointPacked> psi_table(table_size);

    build_odd_multiples_table(Q, q_table.data(), table_size);

    Point psiQ = apply_endomorphism(Q);
    build_odd_multiples_table(psiQ, psi_table.data(), table_size);

    // RLE-driven loop: iterate over precomputed steps instead of all 256 bits
    JacobianPoint result{FieldElement::zero(), FieldElement::one(), FieldElement::zero(), true};

    for (const auto& step : precomp.steps) {
        // Perform N consecutive doublings
        for (uint16_t d = 0; d < step.num_doubles; ++d) {
            if (!result.infinity) {
                result = jacobian_double(result);
            }
        }

        // Add from Q-table (k_1 component)
        if (step.idx1 != 0xFF) {
            AffinePointPacked pt = q_table[step.idx1];
            if (step.neg1) pt.y = negate_fe(pt.y);
            result = jacobian_add_mixed_local(result, pt);
        }

        // Add from psi(Q)-table (k_2 component)
        if (step.idx2 != 0xFF) {
            AffinePointPacked pt = psi_table[step.idx2];
            if (step.neg2) pt.y = negate_fe(pt.y);
            result = jacobian_add_mixed_local(result, pt);
        }
    }

    return Point::from_jacobian_coords(result.x, result.y, result.z, result.infinity);
}

Point scalar_mul_arbitrary_precomputed_notable(const Point& Q,
                                                const PrecomputedScalarOptimized& precomp) {
    // No-table mode: use only +/-Q and +/-psi(Q) directly
    // Avoids building odd-multiples tables at the cost of more additions
    // Still uses precomputed RLE steps for the scalar structure

    AffinePointPacked aQ = to_affine(Q);
    AffinePointPacked aPsiQ = to_affine(apply_endomorphism(Q));

    // We need the full odd-multiples tables since wNAF digits can reference
    // indices > 0 (e.g. +/-3, +/-5, +/-7 for w=4). Build the tables.
    const unsigned w = precomp.window_bits;
    const std::size_t table_size = std::size_t{1} << (w - 1);

    std::vector<AffinePointPacked> q_table(table_size);
    std::vector<AffinePointPacked> psi_table(table_size);

    build_odd_multiples_table(Q, q_table.data(), table_size);

    Point psiQ = apply_endomorphism(Q);
    build_odd_multiples_table(psiQ, psi_table.data(), table_size);

    // RLE-driven loop (same as optimized but with tables built inline)
    JacobianPoint result{FieldElement::zero(), FieldElement::one(), FieldElement::zero(), true};

    for (const auto& step : precomp.steps) {
        for (uint16_t d = 0; d < step.num_doubles; ++d) {
            if (!result.infinity) {
                result = jacobian_double(result);
            }
        }

        if (step.idx1 != 0xFF) {
            AffinePointPacked pt = q_table[step.idx1];
            if (step.neg1) pt.y = negate_fe(pt.y);
            result = jacobian_add_mixed_local(result, pt);
        }

        if (step.idx2 != 0xFF) {
            AffinePointPacked pt = psi_table[step.idx2];
            if (step.neg2) pt.y = negate_fe(pt.y);
            result = jacobian_add_mixed_local(result, pt);
        }
    }

    return Point::from_jacobian_coords(result.x, result.y, result.z, result.infinity);
}

// Helper for beta used in tests or manual GLV
[[maybe_unused]] static FieldElement const_beta() {
    return FieldElement::from_bytes(glv_constants::BETA);
}

}  // namespace secp256k1::fast
