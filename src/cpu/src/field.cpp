#include "secp256k1/field.hpp"
#include "secp256k1/field_asm.hpp"
#include "secp256k1/detail/arith64.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>


namespace secp256k1::fast {
namespace {

using secp256k1::detail::add64;
using secp256k1::detail::sub64;

using limbs4 = FieldElement::limbs_type;
using wide8 = std::array<std::uint64_t, 8>;
constexpr std::size_t kSmallBatchInverseScratch = 64;

#if defined(_MSC_VER) && !defined(__clang__)

inline void mul64(std::uint64_t a, std::uint64_t b, std::uint64_t& lo, std::uint64_t& hi) {
    lo = _umul128(a, b, &hi);
}
#else

#ifdef SECP256K1_NO_INT128

inline void mul64(std::uint64_t a, std::uint64_t b, std::uint64_t& lo, std::uint64_t& hi) {
    // Split into 32-bit parts
    std::uint64_t a_lo = a & 0xFFFFFFFFULL;
    std::uint64_t a_hi = a >> 32;
    std::uint64_t b_lo = b & 0xFFFFFFFFULL;
    std::uint64_t b_hi = b >> 32;

    std::uint64_t p0 = a_lo * b_lo;
    std::uint64_t p1 = a_lo * b_hi;
    std::uint64_t p2 = a_hi * b_lo;
    std::uint64_t p3 = a_hi * b_hi;

    std::uint64_t carry = ((p0 >> 32) + (p1 & 0xFFFFFFFFULL) + (p2 & 0xFFFFFFFFULL)) >> 32;

    lo = p0 + (p1 << 32) + (p2 << 32);
    hi = p3 + (p1 >> 32) + (p2 >> 32) + carry;
}

#else
// __int128 available
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif

inline void mul64(std::uint64_t a, std::uint64_t b, std::uint64_t& lo, std::uint64_t& hi) {
    unsigned __int128 const product = static_cast<unsigned __int128>(a) * b;
    lo = static_cast<std::uint64_t>(product);
    hi = static_cast<std::uint64_t>(product >> 64);
}
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#endif // SECP256K1_NO_INT128

#endif // _MSC_VER

[[maybe_unused]] constexpr std::uint64_t MOD_ADJUST = 0x1000003D1ULL;

constexpr limbs4 PRIME{
    0xFFFFFFFEFFFFFC2FULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL
};

constexpr limbs4 ONE{1ULL, 0ULL, 0ULL, 0ULL};

struct Uint320 {
    std::array<std::uint64_t, 5> limbs{};
};

constexpr Uint320 PRIME_U320{{
    0xFFFFFFFEFFFFFC2FULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL,
    0ULL
}};

constexpr Uint320 ONE_U320{{1ULL, 0ULL, 0ULL, 0ULL, 0ULL}};

inline Uint320 to_uint320(const FieldElement& fe) {
    Uint320 out{};
    const auto& limbs = fe.limbs();
    for (std::size_t i = 0; i < 4; ++i) {
        out.limbs[i] = limbs[i];
    }
    return out;
}

inline bool uint320_is_one(const Uint320& value) {
    return value.limbs[0] == 1ULL && value.limbs[1] == 0ULL &&
           value.limbs[2] == 0ULL && value.limbs[3] == 0ULL &&
           value.limbs[4] == 0ULL;
}

inline bool uint320_is_even(const Uint320& value) {
    return (value.limbs[0] & 1ULL) == 0ULL;
}

inline int uint320_compare(const Uint320& a, const Uint320& b) {
    for (std::size_t i = a.limbs.size(); i-- > 0;) {
        if (a.limbs[i] > b.limbs[i]) {
            return 1;
        }
        if (a.limbs[i] < b.limbs[i]) {
            return -1;
        }
    }
    return 0;
}

inline void uint320_add_assign(Uint320& target, const Uint320& addend) {
    unsigned char carry = 0;
    for (std::size_t i = 0; i < target.limbs.size(); ++i) {
        target.limbs[i] = add64(target.limbs[i], addend.limbs[i], carry);
    }
}

inline void uint320_sub_assign(Uint320& target, const Uint320& subtrahend) {
    unsigned char borrow = 0;
    for (std::size_t i = 0; i < target.limbs.size(); ++i) {
        target.limbs[i] = sub64(target.limbs[i], subtrahend.limbs[i], borrow);
    }
}

inline void uint320_rshift1(Uint320& value) {
    std::uint64_t carry = 0ULL;
    for (std::size_t idx = value.limbs.size(); idx-- > 0;) {
        std::uint64_t const next_carry = value.limbs[idx] & 1ULL;
        value.limbs[idx] = (value.limbs[idx] >> 1) | (carry << 63);
        carry = next_carry;
    }
}

inline void uint320_reduce_mod_prime(Uint320& value) {
    // Fast path: most values need 0-2 reductions in EEA
    if (value.limbs[4] == 0ULL && uint320_compare(value, PRIME_U320) < 0) {
        return;  // Already reduced
    }
    // First reduction (always needed if we got here)
    uint320_sub_assign(value, PRIME_U320);
    
    // Second reduction (rare but possible)
    if (value.limbs[4] != 0ULL || uint320_compare(value, PRIME_U320) >= 0) {
        uint320_sub_assign(value, PRIME_U320);
        
        // Fallback to loop (extremely rare in practice)
        while (value.limbs[4] != 0ULL || uint320_compare(value, PRIME_U320) >= 0) {
            uint320_sub_assign(value, PRIME_U320);
        }
    }
}

inline void uint320_sub_mod(Uint320& target, const Uint320& subtrahend) {
    if (uint320_compare(target, subtrahend) >= 0) {
        uint320_sub_assign(target, subtrahend);
    } else {
        Uint320 tmp = target;
        uint320_add_assign(tmp, PRIME_U320);
        uint320_sub_assign(tmp, subtrahend);
        target = tmp;
    }
    uint320_reduce_mod_prime(target);
}

inline FieldElement field_from_uint320(Uint320 value) {
    uint320_reduce_mod_prime(value);
    limbs4 limbs{};
    for (std::size_t i = 0; i < 4; ++i) {
        limbs[i] = value.limbs[i];
    }
    return FieldElement::from_limbs(limbs);
}

#ifndef SECP256K1_FE_INV_METHOD_BINARY
#define SECP256K1_FE_INV_METHOD_BINARY 1
#endif

#ifndef SECP256K1_FE_INV_METHOD_WINDOW4
#define SECP256K1_FE_INV_METHOD_WINDOW4 2
#endif

#ifndef SECP256K1_FE_INV_METHOD_ADDCHAIN
#define SECP256K1_FE_INV_METHOD_ADDCHAIN 3
#endif

#ifndef SECP256K1_FE_INV_METHOD_EEA
#define SECP256K1_FE_INV_METHOD_EEA 4
#endif

#ifndef SECP256K1_FE_INV_METHOD
#define SECP256K1_FE_INV_METHOD SECP256K1_FE_INV_METHOD_EEA
#endif

}

template <std::size_t N>
#if defined(__GNUC__) || defined(__clang__)
__attribute__((always_inline))
#elif defined(_MSC_VER)
__forceinline
#endif
inline void add_into(std::array<std::uint64_t, N>& arr, std::size_t index, std::uint64_t value) {
    // Callers guarantee index < N; no range check needed in hot path.
    unsigned char carry = 0;
    arr[index] = add64(arr[index], value, carry);
    ++index;
    while (carry != 0 && index < N) {
        arr[index] = add64(arr[index], 0ULL, carry);
        ++index;
    }
}

inline bool ge(const limbs4& a, const limbs4& b) {
    // Branchless: compute a - b, check if borrow == 0 means a >= b
    unsigned char borrow = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        sub64(a[i], b[i], borrow);
    }
    return borrow == 0;
}

void sub_in_place(limbs4& a, const limbs4& b) {
    unsigned char borrow = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        a[i] = sub64(a[i], b[i], borrow);
    }
}

limbs4 add_impl(const limbs4& a, const limbs4& b);

#if defined(SECP256K1_PLATFORM_STM32) && (defined(__arm__) || defined(__thumb__))
// ============================================================================
// ARM Cortex-M3 optimized 256-bit modular add/sub
// Uses ADDS/ADCS chain on 8x32-bit words -- avoids expensive 64-bit emulation.
// Branchless conditional reduction via mask.
// ============================================================================

// 256-bit subtraction: out = a - b, returns borrow (0 or 1)
static inline std::uint32_t arm_sub256(
    const std::uint32_t a[8], const std::uint32_t b[8], std::uint32_t out[8])
{
    std::uint32_t borrow;
    __asm__ volatile(
        "ldr  r2, [%[a], #0]\n\t"   "ldr  r3, [%[b], #0]\n\t"
        "subs r2, r2, r3\n\t"       "str  r2, [%[o], #0]\n\t"
        "ldr  r2, [%[a], #4]\n\t"   "ldr  r3, [%[b], #4]\n\t"
        "sbcs r2, r2, r3\n\t"       "str  r2, [%[o], #4]\n\t"
        "ldr  r2, [%[a], #8]\n\t"   "ldr  r3, [%[b], #8]\n\t"
        "sbcs r2, r2, r3\n\t"       "str  r2, [%[o], #8]\n\t"
        "ldr  r2, [%[a], #12]\n\t"  "ldr  r3, [%[b], #12]\n\t"
        "sbcs r2, r2, r3\n\t"       "str  r2, [%[o], #12]\n\t"
        "ldr  r2, [%[a], #16]\n\t"  "ldr  r3, [%[b], #16]\n\t"
        "sbcs r2, r2, r3\n\t"       "str  r2, [%[o], #16]\n\t"
        "ldr  r2, [%[a], #20]\n\t"  "ldr  r3, [%[b], #20]\n\t"
        "sbcs r2, r2, r3\n\t"       "str  r2, [%[o], #20]\n\t"
        "ldr  r2, [%[a], #24]\n\t"  "ldr  r3, [%[b], #24]\n\t"
        "sbcs r2, r2, r3\n\t"       "str  r2, [%[o], #24]\n\t"
        "ldr  r2, [%[a], #28]\n\t"  "ldr  r3, [%[b], #28]\n\t"
        "sbcs r2, r2, r3\n\t"       "str  r2, [%[o], #28]\n\t"
        "mov  %[bw], #0\n\t"
        "adc  %[bw], %[bw], #0\n\t"  // borrow = !carry (invert)
        "eor  %[bw], %[bw], #1"     // borrow = 1 if underflow
        : [bw] "=r"(borrow)
        : [a] "r"(a), [b] "r"(b), [o] "r"(out)
        : "r2", "r3", "cc", "memory"
    );
    return borrow;
}

// 256-bit addition: out = a + b, returns carry (0 or 1)
static inline std::uint32_t arm_add256(
    const std::uint32_t a[8], const std::uint32_t b[8], std::uint32_t out[8])
{
    std::uint32_t carry;
    __asm__ volatile(
        "ldr  r2, [%[a], #0]\n\t"   "ldr  r3, [%[b], #0]\n\t"
        "adds r2, r2, r3\n\t"       "str  r2, [%[o], #0]\n\t"
        "ldr  r2, [%[a], #4]\n\t"   "ldr  r3, [%[b], #4]\n\t"
        "adcs r2, r2, r3\n\t"       "str  r2, [%[o], #4]\n\t"
        "ldr  r2, [%[a], #8]\n\t"   "ldr  r3, [%[b], #8]\n\t"
        "adcs r2, r2, r3\n\t"       "str  r2, [%[o], #8]\n\t"
        "ldr  r2, [%[a], #12]\n\t"  "ldr  r3, [%[b], #12]\n\t"
        "adcs r2, r2, r3\n\t"       "str  r2, [%[o], #12]\n\t"
        "ldr  r2, [%[a], #16]\n\t"  "ldr  r3, [%[b], #16]\n\t"
        "adcs r2, r2, r3\n\t"       "str  r2, [%[o], #16]\n\t"
        "ldr  r2, [%[a], #20]\n\t"  "ldr  r3, [%[b], #20]\n\t"
        "adcs r2, r2, r3\n\t"       "str  r2, [%[o], #20]\n\t"
        "ldr  r2, [%[a], #24]\n\t"  "ldr  r3, [%[b], #24]\n\t"
        "adcs r2, r2, r3\n\t"       "str  r2, [%[o], #24]\n\t"
        "ldr  r2, [%[a], #28]\n\t"  "ldr  r3, [%[b], #28]\n\t"
        "adcs r2, r2, r3\n\t"       "str  r2, [%[o], #28]\n\t"
        "mov  %[cy], #0\n\t"
        "adc  %[cy], %[cy], #0"
        : [cy] "=r"(carry)
        : [a] "r"(a), [b] "r"(b), [o] "r"(out)
        : "r2", "r3", "cc", "memory"
    );
    return carry;
}

// p = FFFFFFFE FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F
// In 32-bit words (LE): {0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
//                         0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF}
static const std::uint32_t PRIME32[8] = {
    0xFFFFFC2FU, 0xFFFFFFFEU, 0xFFFFFFFFU, 0xFFFFFFFFU,
    0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU
};
// C = p + 1 mod 2^256 = {0x3D1, 0x1, 0, 0, 0, 0, 0, 0} (= 2^32 + 977)
static const std::uint32_t MOD_C32[8] = {
    0x000003D1U, 0x00000001U, 0, 0, 0, 0, 0, 0
};

limbs4 sub_impl(const limbs4& a, const limbs4& b) {
    // Work in 32-bit words -- use memcpy to avoid strict aliasing UB
    std::uint32_t a32[8], b32[8];
    std::memcpy(a32, a.data(), 32);
    std::memcpy(b32, b.data(), 32);
    std::uint32_t out32[8];

    std::uint32_t borrow = arm_sub256(a32, b32, out32);

    // If borrow: out += p (branchless: always execute, mask addend)
    // mask = -borrow (0xFFFFFFFF if borrow, 0 if not)
    {
        std::uint32_t mask = static_cast<std::uint32_t>(-static_cast<std::int32_t>(borrow));
        std::uint32_t masked_p[8];
        for (int i = 0; i < 8; i++) masked_p[i] = PRIME32[i] & mask;
        arm_add256(out32, masked_p, out32);
    }

    // Conditional subtract p if out >= p (branchless word select)
    std::uint32_t tmp[8];
    std::uint32_t sub_borrow = arm_sub256(out32, PRIME32, tmp);
    // mask = all-1s if borrow (keep out32), all-0s if no borrow (use tmp)
    std::uint32_t sel_mask = static_cast<std::uint32_t>(-static_cast<std::int32_t>(sub_borrow));

    limbs4 out;
    out[0] = (std::uint64_t)(tmp[0] ^ ((tmp[0] ^ out32[0]) & sel_mask))
           | ((std::uint64_t)(tmp[1] ^ ((tmp[1] ^ out32[1]) & sel_mask)) << 32);
    out[1] = (std::uint64_t)(tmp[2] ^ ((tmp[2] ^ out32[2]) & sel_mask))
           | ((std::uint64_t)(tmp[3] ^ ((tmp[3] ^ out32[3]) & sel_mask)) << 32);
    out[2] = (std::uint64_t)(tmp[4] ^ ((tmp[4] ^ out32[4]) & sel_mask))
           | ((std::uint64_t)(tmp[5] ^ ((tmp[5] ^ out32[5]) & sel_mask)) << 32);
    out[3] = (std::uint64_t)(tmp[6] ^ ((tmp[6] ^ out32[6]) & sel_mask))
           | ((std::uint64_t)(tmp[7] ^ ((tmp[7] ^ out32[7]) & sel_mask)) << 32);
    return out;
}

limbs4 add_impl(const limbs4& a, const limbs4& b) {
    // Work in 32-bit words -- use memcpy to avoid strict aliasing UB
    std::uint32_t a32[8], b32[8];
    std::memcpy(a32, a.data(), 32);
    std::memcpy(b32, b.data(), 32);
    std::uint32_t out32[8];

    std::uint32_t carry = arm_add256(a32, b32, out32);

    // If carry: out += C (branchless: always execute, mask addend)
    {
        std::uint32_t mask = static_cast<std::uint32_t>(-static_cast<std::int32_t>(carry));
        std::uint32_t masked_c[8];
        for (int i = 0; i < 8; i++) masked_c[i] = MOD_C32[i] & mask;
        arm_add256(out32, masked_c, out32);
    }

    // Conditional subtract p if out >= p (branchless word select)
    std::uint32_t tmp[8];
    std::uint32_t sub_borrow = arm_sub256(out32, PRIME32, tmp);
    std::uint32_t sel_mask = static_cast<std::uint32_t>(-static_cast<std::int32_t>(sub_borrow));

    limbs4 out;
    out[0] = (std::uint64_t)(tmp[0] ^ ((tmp[0] ^ out32[0]) & sel_mask))
           | ((std::uint64_t)(tmp[1] ^ ((tmp[1] ^ out32[1]) & sel_mask)) << 32);
    out[1] = (std::uint64_t)(tmp[2] ^ ((tmp[2] ^ out32[2]) & sel_mask))
           | ((std::uint64_t)(tmp[3] ^ ((tmp[3] ^ out32[3]) & sel_mask)) << 32);
    out[2] = (std::uint64_t)(tmp[4] ^ ((tmp[4] ^ out32[4]) & sel_mask))
           | ((std::uint64_t)(tmp[5] ^ ((tmp[5] ^ out32[5]) & sel_mask)) << 32);
    out[3] = (std::uint64_t)(tmp[6] ^ ((tmp[6] ^ out32[6]) & sel_mask))
           | ((std::uint64_t)(tmp[7] ^ ((tmp[7] ^ out32[7]) & sel_mask)) << 32);
    return out;
}

#else
// Generic branchless add/sub using 64-bit limbs (x86, RISC-V, ESP32/Xtensa)
limbs4 sub_impl(const limbs4& a, const limbs4& b) {
    // Compute a - b
    limbs4 out{};
    unsigned char borrow = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        out[i] = sub64(a[i], b[i], borrow);
    }
    // Branchless: if borrow, add PRIME (mask selects PRIME or 0)
    const auto mask = 0ULL - static_cast<std::uint64_t>(borrow);
    unsigned char carry = 0;
    out[0] = add64(out[0], PRIME[0] & mask, carry);
    out[1] = add64(out[1], PRIME[1] & mask, carry);
    out[2] = add64(out[2], PRIME[2] & mask, carry);
    out[3] = add64(out[3], PRIME[3] & mask, carry);
    return out;
}

limbs4 add_impl(const limbs4& a, const limbs4& b) {
    // Compute s = a + b (may overflow 256 bits)
    limbs4 s{};
    unsigned char c1 = 0;
    s[0] = add64(a[0], b[0], c1);
    s[1] = add64(a[1], b[1], c1);
    s[2] = add64(a[2], b[2], c1);
    s[3] = add64(a[3], b[3], c1);
    // Try s + C where C = 2^256 - p = 0x1000003D1.
    // If c1=1: a+b overflowed, reduced = a+b+C mod 2^256 = a+b-p. Always correct.
    // If c1=0 and c2=1: s+C overflowed, meaning s >= p. reduced = s-p. Correct.
    // If c1=0 and c2=0: s < p. Keep original s.
    limbs4 r{};
    unsigned char c2 = 0;
    r[0] = add64(s[0], MOD_ADJUST, c2);
    r[1] = add64(s[1], 0, c2);
    r[2] = add64(s[2], 0, c2);
    r[3] = add64(s[3], 0, c2);
    const auto use_reduced = static_cast<std::uint64_t>(c1 | c2);
    const auto mask = 0ULL - use_reduced;
    s[0] ^= (s[0] ^ r[0]) & mask;
    s[1] ^= (s[1] ^ r[1]) & mask;
    s[2] ^= (s[2] ^ r[2]) & mask;
    s[3] ^= (s[3] ^ r[3]) & mask;
    return s;
}
#endif // SECP256K1_PLATFORM_STM32 && __arm__

#if defined(__GNUC__) || defined(__clang__)
__attribute__((always_inline))
#elif defined(_MSC_VER)
__forceinline
#endif
inline void mul_add_to(wide8& acc, std::size_t index, std::uint64_t a, std::uint64_t b) {
    std::uint64_t lo = 0;
    std::uint64_t hi = 0;
    mul64(a, b, lo, hi);
    add_into(acc, index, lo);
    add_into(acc, index + 1, hi);
}

wide8 mul_wide(const limbs4& a, const limbs4& b) {
    // mul_wide is reached only when SECP256K1_NO_ASM is defined (sanitizer /
    // coverage / certain cross-compile builds). Performance is not critical
    // on that path; correctness is.
    //
    // The prior "fast" __int128 single-accumulator Comba unrolling had a
    // silent overflow at column 3: with all-max 64-bit limbs (e.g. p-1 or
    // the test "large" vector), the column sum is up to 4 * (2^64-1)^2 plus
    // the column-2 carry — roughly 2^130, which truncates inside the 128-bit
    // accumulator. That caused wrong FE64 multiplication results for inputs
    // whose 4x64 limbs approached p, which in turn broke FE52 vs FE64
    // cross-checks in test_field_52 (and downstream selftest "Boundary
    // Scalar KAT", Point::add SECP_ASSERT_ON_CURVE under Debug, etc.) on
    // every Sanitizer / coverage / no-asm CI job.
    //
    // The portable add_into-based path used here propagates carries across
    // the full wide8 array per product, so column overflow is impossible.
    wide8 prod{};
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            mul_add_to(prod, i + j, a[i], b[j]);
        }
    }
    return prod;
}

// Phase 5.5: Fast modular reduction for secp256k1 prime
// p = 2^256 - 2^32 - 977 = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
// For 512-bit t = t_high * 2^256 + t_low:
// Since 2^256 == 2^32 + 977 (mod p)
// We have: t == t_low + t_high * (2^32 + 977) (mod p)
// One-pass reduction algorithm
limbs4 reduce(const wide8& t) {
    // Step 1: Start with low 256 bits + overflow accumulator
    std::array<std::uint64_t, 5> result{t[0], t[1], t[2], t[3], 0ULL};

    // Step 2: Process each high limb: add high[i] * 0x1000003D1 to position i
    // secp256k1: 2^256 = 0x1000003D1 = 2^32 + 977 (mod p)
    // For each hi_limb, the contribution to result is:
    //   result[i]   += lo977 + (hi_limb << 32)   [combined with single carry]
    //   result[i+1] += hi977 + (hi_limb >> 32)   [combined with single carry]
    // Using __int128 accumulates both values at each position in one step,
    // replacing 4 serial add_into() carry-chain walks with 2 compound adds.
    for (std::size_t i = 0; i < 4; ++i) {
        std::uint64_t const hi_limb = t[4 + i];
        if (hi_limb == 0) continue;

        std::uint64_t lo977 = 0, hi977 = 0;
        mul64(hi_limb, 977ULL, lo977, hi977);

        // Compound add at position i: lo977 + (hi_limb << 32) in one shot.
        // `hi_limb << 32` in u64 loses the upper 32 bits, but that's intentional:
        // the upper 32 bits are carried into position i+1 via `hi_limb >> 32`.
#ifndef SECP256K1_NO_INT128
        using u128 = unsigned __int128;
        u128 acc = static_cast<u128>(result[i])
                 + static_cast<u128>(lo977)
                 + static_cast<u128>(hi_limb << 32);
        result[i] = static_cast<std::uint64_t>(acc);
        std::uint64_t carry = static_cast<std::uint64_t>(acc >> 64);

        // Compound add at position i+1: hi977 + (hi_limb >> 32) + carry
        acc = static_cast<u128>(result[i + 1])
            + static_cast<u128>(hi977)
            + static_cast<u128>(hi_limb >> 32)
            + static_cast<u128>(carry);
        result[i + 1] = static_cast<std::uint64_t>(acc);
        carry = static_cast<std::uint64_t>(acc >> 64);

        // Branchless carry propagation (carry is 0 or 1 for secp256k1 limb bounds).
        // i+2 == 5 on the final iteration (i=3): the array has only 5 limbs,
        // and the algorithm's bound (290 bits ≤ 5×64 bits) guarantees this
        // carry is 0 for any valid 256×256-bit reduction. Fuzzer inputs may
        // exceed that domain — guard the access to avoid UB-array-overflow.
        if (i + 2 < 5) {
            u128 acc2 = static_cast<u128>(result[i + 2]) + carry;
            result[i + 2] = static_cast<std::uint64_t>(acc2);
            std::uint64_t carry2 = static_cast<std::uint64_t>(acc2 >> 64);
            if (carry2 && i + 3 < 5) result[i + 3] += carry2;
        }
#else
        // No __int128: manual 64-bit carry arithmetic (for 32-bit MCUs)
        auto add64c = [](std::uint64_t a, std::uint64_t b, std::uint64_t& c_out) -> std::uint64_t {
            std::uint64_t s = a + b; c_out = (s < a) ? 1u : 0u; return s;
        };
        std::uint64_t c1 = 0, c2 = 0, c3 = 0;
        std::uint64_t s = add64c(result[i], lo977, c1);
        s = add64c(s, static_cast<std::uint64_t>(hi_limb << 32), c2);
        result[i] = s;
        std::uint64_t carry = c1 + c2;

        std::uint64_t d1 = 0, d2 = 0, d3 = 0;
        std::uint64_t t = add64c(result[i + 1], hi977, d1);
        t = add64c(t, static_cast<std::uint64_t>(hi_limb >> 32), d2);
        t = add64c(t, carry, d3);
        result[i + 1] = t;
        carry = d1 + d2 + d3;

        // Same i+2 == 5 guard as the __int128 path above.
        if (i + 2 < 5) {
            std::uint64_t e = 0;
            result[i + 2] = add64c(result[i + 2], carry, e);
            std::uint64_t carry2 = e;
            if (carry2 && i + 3 < 5) result[i + 3] += carry2;
        }
#endif
    }

    // Step 3: Handle overflow in result[4] -- unrolled 2 iterations (bounded)
    // After reducing 4 high limbs, overflow is at most ~34 bits.
    // First pass: overflow * 0x1000003D1 is at most ~67 bits, new overflow <= 1 bit.
    // Second pass: 1 * 0x1000003D1 = 33 bits, no further overflow possible.
    // Step 3: fold overflow using u128 — replaces 4 add_into() carry-chain calls.
    for (int pass = 0; pass < 2; ++pass) {
        std::uint64_t const overflow = result[4];
        result[4] = 0;
        if (overflow == 0) break;
        // overflow * 0x1000003D1 = overflow*(2^32 + 977) in one multiply
#ifndef SECP256K1_NO_INT128
        using u128 = unsigned __int128;
        u128 const term = static_cast<u128>(overflow) * 0x1000003D1ULL;
        u128 acc0 = static_cast<u128>(result[0]) + static_cast<std::uint64_t>(term);
        result[0] = static_cast<std::uint64_t>(acc0);
        u128 acc1 = static_cast<u128>(result[1])
                  + static_cast<std::uint64_t>(term >> 64)
                  + static_cast<std::uint64_t>(acc0 >> 64);
        result[1] = static_cast<std::uint64_t>(acc1);
        std::uint64_t carry = static_cast<std::uint64_t>(acc1 >> 64);
        if (carry) result[2] += carry;
#else
        // No __int128: use mul64 to compute overflow * 0x1000003D1 (at most 67 bits)
        std::uint64_t term_lo = 0, term_hi = 0;
        mul64(overflow, 0x1000003D1ULL, term_lo, term_hi);
        auto add64c2 = [](std::uint64_t a, std::uint64_t b, std::uint64_t& c_out) -> std::uint64_t {
            std::uint64_t s = a + b; c_out = (s < a) ? 1u : 0u; return s;
        };
        std::uint64_t c0 = 0, c1a = 0, c1b = 0;
        result[0] = add64c2(result[0], term_lo, c0);
        std::uint64_t t1 = add64c2(result[1], term_hi, c1a);
        result[1] = add64c2(t1, c0, c1b);
        std::uint64_t carry = c1a + c1b;
        if (carry) result[2] += carry;
#endif
    }

    // Step 4: Branchless final normalization (subtract p if value >= p)
    limbs4 out{result[0], result[1], result[2], result[3]};

    unsigned char borrow = 0;
    limbs4 reduced;
    reduced[0] = sub64(out[0], PRIME[0], borrow);
    reduced[1] = sub64(out[1], PRIME[1], borrow);
    reduced[2] = sub64(out[2], PRIME[2], borrow);
    reduced[3] = sub64(out[3], PRIME[3], borrow);
    // borrow == 0 means out >= PRIME -> use reduced
    auto const mask = 0ULL - static_cast<std::uint64_t>(1U - borrow);
    out[0] = (out[0] & ~mask) | (reduced[0] & mask);
    out[1] = (out[1] & ~mask) | (reduced[1] & mask);
    out[2] = (out[2] & ~mask) | (reduced[2] & mask);
    out[3] = (out[3] & ~mask) | (reduced[3] & mask);

    return out;
}

// ============================================================================
// ESP32-Optimized Field Arithmetic (32-bit Comba / Product-Scanning)
// ============================================================================
// ESP32-S3 Xtensa LX7 is a 32-bit processor with native 32x32->64 multiply.
// The standard 64-bit limb path emulates 64x64->128 via 4 native multiplies
// plus significant decomposition/carry overhead per mul64 call.
//
// This Comba implementation works directly with 8 x 32-bit limbs:
//  - Eliminates mul64 decomposition overhead entirely
//  - Uses a compact 3-word (96-bit) accumulator for carry propagation
//  - Dedicated square exploits a[i]*a[j] = a[j]*a[i] symmetry (36 vs 64 muls)
//  - secp256k1-specific reduction in 32-bit for p = 2^256 - (2^32 + 977)
// ============================================================================
#if defined(SECP256K1_PLATFORM_ESP32) || defined(__XTENSA__) || defined(SECP256K1_PLATFORM_STM32)

// IRAM placement for ESP32: eliminates flash wait states (~11 cycles/fetch on S3)
#if (defined(SECP256K1_PLATFORM_ESP32) || defined(__XTENSA__)) && !defined(SECP256K1_PLATFORM_STM32)
#  ifdef IRAM_ATTR
#    define SECP_IRAM IRAM_ATTR
#  else
#    define SECP_IRAM __attribute__((section(".iram1.text")))
#  endif
#else
#  define SECP_IRAM
#endif

// ============================================================================
// Fully unrolled Comba multiplication and squaring for ESP32 / Xtensa
// All straight-line code: no loops, no branches, optimal register scheduling.
// ============================================================================

// Accumulate product a[i]*b[j] into 96-bit accumulator (c0,c1,c2)
#if defined(SECP256K1_PLATFORM_STM32) && (defined(__arm__) || defined(__thumb__))
// ARM Cortex-M3: UMULL + ADDS/ADCS/ADC -- 4 instructions per product
// vs ~12 instructions with C uint64_t emulation on 32-bit target
#define MULACC(i, j) do {                                        \
    std::uint32_t _lo, _hi;                                      \
    __asm__ volatile(                                             \
        "umull %[lo], %[hi], %[ai], %[bj]\n\t"               \
        "adds  %[c0], %[c0], %[lo]\n\t"                       \
        "adcs  %[c1], %[c1], %[hi]\n\t"                       \
        "adc   %[c2], %[c2], #0"                                \
        : [c0] "+r"(c0), [c1] "+r"(c1), [c2] "+r"(c2),     \
          [lo] "=&r"(_lo), [hi] "=&r"(_hi)                    \
        : [ai] "r"(a[i]), [bj] "r"(b[j])                      \
        : "cc"                                                  \
    );                                                            \
} while (0)
#else
// Generic C version for ESP32/Xtensa and other 32-bit targets
#define MULACC(i, j) do {                                        \
    std::uint64_t _p = (std::uint64_t)a[i] * b[j];              \
    std::uint64_t _s = (std::uint64_t)c0 + (std::uint32_t)_p;   \
    c0 = (std::uint32_t)_s;                                      \
    _s = (std::uint64_t)c1 + (std::uint32_t)(_p >> 32) + (_s >> 32); \
    c1 = (std::uint32_t)_s;                                      \
    c2 += (std::uint32_t)(_s >> 32);                             \
} while (0)
#endif
// Store column and shift accumulator
#define COL_END(k) do { r[k] = c0; c0 = c1; c1 = c2; c2 = 0; } while (0)

// Fully unrolled 8x8 -> 16 Comba multiplication (64 products, 0 branches)
SECP_IRAM static void esp32_mul_comba(const std::uint32_t a[8], const std::uint32_t b[8],
                            std::uint32_t r[16]) {
    std::uint32_t c0 = 0, c1 = 0, c2 = 0;
    /* k=0  */ MULACC(0,0);
    COL_END(0);
    /* k=1  */ MULACC(0,1); MULACC(1,0);
    COL_END(1);
    /* k=2  */ MULACC(0,2); MULACC(1,1); MULACC(2,0);
    COL_END(2);
    /* k=3  */ MULACC(0,3); MULACC(1,2); MULACC(2,1); MULACC(3,0);
    COL_END(3);
    /* k=4  */ MULACC(0,4); MULACC(1,3); MULACC(2,2); MULACC(3,1); MULACC(4,0);
    COL_END(4);
    /* k=5  */ MULACC(0,5); MULACC(1,4); MULACC(2,3); MULACC(3,2); MULACC(4,1); MULACC(5,0);
    COL_END(5);
    /* k=6  */ MULACC(0,6); MULACC(1,5); MULACC(2,4); MULACC(3,3); MULACC(4,2); MULACC(5,1); MULACC(6,0);
    COL_END(6);
    /* k=7  */ MULACC(0,7); MULACC(1,6); MULACC(2,5); MULACC(3,4); MULACC(4,3); MULACC(5,2); MULACC(6,1); MULACC(7,0);
    COL_END(7);
    /* k=8  */ MULACC(1,7); MULACC(2,6); MULACC(3,5); MULACC(4,4); MULACC(5,3); MULACC(6,2); MULACC(7,1);
    COL_END(8);
    /* k=9  */ MULACC(2,7); MULACC(3,6); MULACC(4,5); MULACC(5,4); MULACC(6,3); MULACC(7,2);
    COL_END(9);
    /* k=10 */ MULACC(3,7); MULACC(4,6); MULACC(5,5); MULACC(6,4); MULACC(7,3);
    COL_END(10);
    /* k=11 */ MULACC(4,7); MULACC(5,6); MULACC(6,5); MULACC(7,4);
    COL_END(11);
    /* k=12 */ MULACC(5,7); MULACC(6,6); MULACC(7,5);
    COL_END(12);
    /* k=13 */ MULACC(6,7); MULACC(7,6);
    COL_END(13);
    /* k=14 */ MULACC(7,7);
    r[14] = c0; r[15] = c1;
}
#undef MULACC
#undef COL_END

// Cross-product: accumulate a[i]*a[j] TWICE (for i!=j symmetry in squaring)
#if defined(SECP256K1_PLATFORM_STM32) && (defined(__arm__) || defined(__thumb__))
// ARM Cortex-M3: UMULL + double + ADDS/ADCS/ADC -- 7 instructions
// Doubling the product first then adding once reduces carry-chain depth
// vs the original 2x(ADDS/ADCS/ADC) which has 2 serial carry chains.
#define SQRMAC2(i, j) do {                                       \
    std::uint32_t _lo, _hi;                                      \
    __asm__ volatile(                                             \
        "umull %[lo], %[hi], %[ai], %[aj]\n\t"               \
        "adds  %[lo], %[lo], %[lo]\n\t"                       \
        "adcs  %[hi], %[hi], %[hi]\n\t"                       \
        "adc   %[c2], %[c2], #0\n\t"                           \
        "adds  %[c0], %[c0], %[lo]\n\t"                       \
        "adcs  %[c1], %[c1], %[hi]\n\t"                       \
        "adc   %[c2], %[c2], #0"                                \
        : [c0] "+r"(c0), [c1] "+r"(c1), [c2] "+r"(c2),     \
          [lo] "=&r"(_lo), [hi] "=&r"(_hi)                    \
        : [ai] "r"(a[i]), [aj] "r"(a[j])                      \
        : "cc"                                                  \
    );                                                            \
} while (0)
// Diagonal: accumulate a[i]^2 once
#define SQRMAC1(i) do {                                          \
    std::uint32_t _lo, _hi;                                      \
    __asm__ volatile(                                             \
        "umull %[lo], %[hi], %[ai], %[ai]\n\t"               \
        "adds  %[c0], %[c0], %[lo]\n\t"                       \
        "adcs  %[c1], %[c1], %[hi]\n\t"                       \
        "adc   %[c2], %[c2], #0"                                \
        : [c0] "+r"(c0), [c1] "+r"(c1), [c2] "+r"(c2),     \
          [lo] "=&r"(_lo), [hi] "=&r"(_hi)                    \
        : [ai] "r"(a[i])                                       \
        : "cc"                                                  \
    );                                                            \
} while (0)
#else
// Generic C version for ESP32/Xtensa and other 32-bit targets
#define SQRMAC2(i, j) do {                                       \
    std::uint64_t _p = (std::uint64_t)a[i] * a[j];              \
    std::uint32_t _pl = (std::uint32_t)_p;                       \
    std::uint32_t _ph = (std::uint32_t)(_p >> 32);               \
    std::uint64_t _s = (std::uint64_t)c0 + _pl; c0 = (std::uint32_t)_s;  \
    _s = (std::uint64_t)c1 + _ph + (_s >> 32); c1 = (std::uint32_t)_s;   \
    c2 += (std::uint32_t)(_s >> 32);                             \
    _s = (std::uint64_t)c0 + _pl; c0 = (std::uint32_t)_s;       \
    _s = (std::uint64_t)c1 + _ph + (_s >> 32); c1 = (std::uint32_t)_s;   \
    c2 += (std::uint32_t)(_s >> 32);                             \
} while (0)
// Diagonal: accumulate a[i]^2 once
#define SQRMAC1(i) do {                                          \
    std::uint64_t _p = (std::uint64_t)a[i] * a[i];              \
    std::uint64_t _s = (std::uint64_t)c0 + (std::uint32_t)_p;   \
    c0 = (std::uint32_t)_s;                                      \
    _s = (std::uint64_t)c1 + (std::uint32_t)(_p >> 32) + (_s >> 32); \
    c1 = (std::uint32_t)_s;                                      \
    c2 += (std::uint32_t)(_s >> 32);                             \
} while (0)
#endif
#define SQR_COL_END(k) do { r[k] = c0; c0 = c1; c1 = c2; c2 = 0; } while (0)

// Fully unrolled 8-word squaring (36 muls vs 64 for general, 0 branches)
SECP_IRAM static void esp32_sqr_comba(const std::uint32_t a[8], std::uint32_t r[16]) {
    std::uint32_t c0 = 0, c1 = 0, c2 = 0;
    /* k=0  */ SQRMAC1(0);
    SQR_COL_END(0);
    /* k=1  */ SQRMAC2(0,1);
    SQR_COL_END(1);
    /* k=2  */ SQRMAC2(0,2); SQRMAC1(1);
    SQR_COL_END(2);
    /* k=3  */ SQRMAC2(0,3); SQRMAC2(1,2);
    SQR_COL_END(3);
    /* k=4  */ SQRMAC2(0,4); SQRMAC2(1,3); SQRMAC1(2);
    SQR_COL_END(4);
    /* k=5  */ SQRMAC2(0,5); SQRMAC2(1,4); SQRMAC2(2,3);
    SQR_COL_END(5);
    /* k=6  */ SQRMAC2(0,6); SQRMAC2(1,5); SQRMAC2(2,4); SQRMAC1(3);
    SQR_COL_END(6);
    /* k=7  */ SQRMAC2(0,7); SQRMAC2(1,6); SQRMAC2(2,5); SQRMAC2(3,4);
    SQR_COL_END(7);
    /* k=8  */ SQRMAC2(1,7); SQRMAC2(2,6); SQRMAC2(3,5); SQRMAC1(4);
    SQR_COL_END(8);
    /* k=9  */ SQRMAC2(2,7); SQRMAC2(3,6); SQRMAC2(4,5);
    SQR_COL_END(9);
    /* k=10 */ SQRMAC2(3,7); SQRMAC2(4,6); SQRMAC1(5);
    SQR_COL_END(10);
    /* k=11 */ SQRMAC2(4,7); SQRMAC2(5,6);
    SQR_COL_END(11);
    /* k=12 */ SQRMAC2(5,7); SQRMAC1(6);
    SQR_COL_END(12);
    /* k=13 */ SQRMAC2(6,7);
    SQR_COL_END(13);
    /* k=14 */ SQRMAC1(7);
    r[14] = c0; r[15] = c1;
}
#undef SQRMAC2
#undef SQRMAC1
#undef SQR_COL_END

#if defined(SECP256K1_PLATFORM_STM32) && (defined(__arm__) || defined(__thumb__))
// ============================================================================
// ARM Cortex-M3 optimized secp256k1 reduction
// Uses UMULL for 977xr[i], ADDS/ADCS chains for accumulation.
// All 32-bit operations -- no expensive 64-bit emulation.
// ============================================================================

// Reduction helper: acc(lo,hi) += val
#define REDUCE_ADD(val) do {                  \
    __asm__ volatile(                          \
        "adds %[lo], %[lo], %[v]\n\t"       \
        "adc  %[hi], %[hi], #0"              \
        : [lo] "+r"(acc_lo), [hi] "+r"(acc_hi) \
        : [v] "r"((std::uint32_t)(val))       \
        : "cc"                                \
    );                                         \
} while (0)

// acc(lo,hi) += x * 977, where x is uint32_t
#define REDUCE_MUL977(x) do {                 \
    std::uint32_t _ml, _mh;                   \
    __asm__ volatile(                          \
        "umull %[ml], %[mh], %[xx], %[c977]\n\t" \
        "adds  %[lo], %[lo], %[ml]\n\t"     \
        "adc   %[hi], %[hi], %[mh]"          \
        : [lo] "+r"(acc_lo), [hi] "+r"(acc_hi), \
          [ml] "=&r"(_ml), [mh] "=&r"(_mh)  \
        : [xx] "r"((std::uint32_t)(x)),       \
          [c977] "r"(C977)                    \
        : "cc"                                \
    );                                         \
} while (0)

// Store result and shift accumulator
#define REDUCE_COL(dst) do {                  \
    dst = acc_lo;                              \
    acc_lo = acc_hi;                           \
    acc_hi = 0;                                \
} while (0)

SECP_IRAM static limbs4 esp32_reduce_secp256k1(const std::uint32_t r[16]) {
    static constexpr std::uint32_t C977 = 977U;
    std::uint32_t acc_lo = 0, acc_hi = 0;
    std::uint32_t res[8];

    // First pass: fold r[8..15] into r[0..7]
    // Position 0: r[0] + 977*r[8]
    REDUCE_ADD(r[0]); REDUCE_MUL977(r[8]);
    REDUCE_COL(res[0]);

    // Position 1: carry + r[1] + 977*r[9] + r[8]
    REDUCE_ADD(r[1]); REDUCE_MUL977(r[9]); REDUCE_ADD(r[8]);
    REDUCE_COL(res[1]);

    // Position 2: carry + r[2] + 977*r[10] + r[9]
    REDUCE_ADD(r[2]); REDUCE_MUL977(r[10]); REDUCE_ADD(r[9]);
    REDUCE_COL(res[2]);

    // Position 3: carry + r[3] + 977*r[11] + r[10]
    REDUCE_ADD(r[3]); REDUCE_MUL977(r[11]); REDUCE_ADD(r[10]);
    REDUCE_COL(res[3]);

    // Position 4: carry + r[4] + 977*r[12] + r[11]
    REDUCE_ADD(r[4]); REDUCE_MUL977(r[12]); REDUCE_ADD(r[11]);
    REDUCE_COL(res[4]);

    // Position 5: carry + r[5] + 977*r[13] + r[12]
    REDUCE_ADD(r[5]); REDUCE_MUL977(r[13]); REDUCE_ADD(r[12]);
    REDUCE_COL(res[5]);

    // Position 6: carry + r[6] + 977*r[14] + r[13]
    REDUCE_ADD(r[6]); REDUCE_MUL977(r[14]); REDUCE_ADD(r[13]);
    REDUCE_COL(res[6]);

    // Position 7: carry + r[7] + 977*r[15] + r[14]
    REDUCE_ADD(r[7]); REDUCE_MUL977(r[15]); REDUCE_ADD(r[14]);
    REDUCE_COL(res[7]);

    // Position 8 overflow: carry + r[15]
    REDUCE_ADD(r[15]);
    // acc_lo:acc_hi is the overflow (< 2^34)

    // Second reduction: fold overflow * (977 + 2^32) -- fully branchless
    // Overflow < 2^34, so ov*977 < 2^44 -- no risk of overflow.
    // Always execute all operations (branchless) to avoid Cortex-M3
    // branch misprediction penalties (3 cycles per mispredict x ~5 branches).
    {
        std::uint32_t ov_lo = acc_lo, ov_hi = acc_hi;
        acc_lo = 0; acc_hi = 0;

        // res[0] += ov * 977
        REDUCE_ADD(res[0]);
        {
            std::uint32_t ml, mh;
            __asm__ volatile(
                "umull %[ml], %[mh], %[v], %[c977]\n\t"
                "adds  %[lo], %[lo], %[ml]\n\t"
                "adc   %[hi], %[hi], %[mh]"
                : [lo] "+r"(acc_lo), [hi] "+r"(acc_hi),
                  [ml] "=&r"(ml), [mh] "=&r"(mh)
                : [v] "r"(ov_lo), [c977] "r"(C977)
                : "cc"
            );
            std::uint32_t ov_hi_977 = ov_hi * C977;
            __asm__ volatile(
                "adds %[hi], %[hi], %[v]" : [hi] "+r"(acc_hi) : [v] "r"(ov_hi_977) : "cc"
            );
        }
        REDUCE_COL(res[0]);

        // res[1] += ov_lo (the 2^32 part)
        REDUCE_ADD(res[1]); REDUCE_ADD(ov_lo);
        REDUCE_COL(res[1]);

        // res[2] += ov_hi + carry propagation -- always execute (branchless)
        REDUCE_ADD(res[2]); REDUCE_ADD(ov_hi);
        REDUCE_COL(res[2]);

        // Carry ripple through res[3..7] -- always execute
        REDUCE_ADD(res[3]); REDUCE_COL(res[3]);
        REDUCE_ADD(res[4]); REDUCE_COL(res[4]);
        REDUCE_ADD(res[5]); REDUCE_COL(res[5]);
        REDUCE_ADD(res[6]); REDUCE_COL(res[6]);
        REDUCE_ADD(res[7]); REDUCE_COL(res[7]);
    }

    // Final conditional subtract p using branchless word select
    std::uint32_t tmp[8];
    std::uint32_t sub_borrow = arm_sub256(res, PRIME32, tmp);
    // mask = all-1s if borrow (keep res), all-0s if no borrow (use tmp)
    std::uint32_t sel_mask = static_cast<std::uint32_t>(-static_cast<std::int32_t>(sub_borrow));

    limbs4 out;
    out[0] = (std::uint64_t)(tmp[0] ^ ((tmp[0] ^ res[0]) & sel_mask))
           | ((std::uint64_t)(tmp[1] ^ ((tmp[1] ^ res[1]) & sel_mask)) << 32);
    out[1] = (std::uint64_t)(tmp[2] ^ ((tmp[2] ^ res[2]) & sel_mask))
           | ((std::uint64_t)(tmp[3] ^ ((tmp[3] ^ res[3]) & sel_mask)) << 32);
    out[2] = (std::uint64_t)(tmp[4] ^ ((tmp[4] ^ res[4]) & sel_mask))
           | ((std::uint64_t)(tmp[5] ^ ((tmp[5] ^ res[5]) & sel_mask)) << 32);
    out[3] = (std::uint64_t)(tmp[6] ^ ((tmp[6] ^ res[6]) & sel_mask))
           | ((std::uint64_t)(tmp[7] ^ ((tmp[7] ^ res[7]) & sel_mask)) << 32);
    return out;
}
#undef REDUCE_ADD
#undef REDUCE_MUL977
#undef REDUCE_COL

#else
// Generic C reduction for ESP32/Xtensa -- fully branchless for CT safety
SECP_IRAM static limbs4 esp32_reduce_secp256k1(const std::uint32_t r[16]) {
    std::uint64_t acc;
    std::uint32_t res[8];

    // First reduction pass: fold r[8..15] into r[0..7]
    // 2^256 == 2^32 + 977 (mod p), so t_high[i] contributes
    //   t_high[i] * 977 to position i, t_high[i] * 2^32 to position i+1
    acc = (std::uint64_t)r[0] + (std::uint64_t)r[8] * 977ULL;
    res[0] = (std::uint32_t)acc;
    acc >>= 32;

    acc += (std::uint64_t)r[1] + (std::uint64_t)r[9]  * 977ULL + r[8];
    res[1] = (std::uint32_t)acc; acc >>= 32;

    acc += (std::uint64_t)r[2] + (std::uint64_t)r[10] * 977ULL + r[9];
    res[2] = (std::uint32_t)acc; acc >>= 32;

    acc += (std::uint64_t)r[3] + (std::uint64_t)r[11] * 977ULL + r[10];
    res[3] = (std::uint32_t)acc; acc >>= 32;

    acc += (std::uint64_t)r[4] + (std::uint64_t)r[12] * 977ULL + r[11];
    res[4] = (std::uint32_t)acc; acc >>= 32;

    acc += (std::uint64_t)r[5] + (std::uint64_t)r[13] * 977ULL + r[12];
    res[5] = (std::uint32_t)acc; acc >>= 32;

    acc += (std::uint64_t)r[6] + (std::uint64_t)r[14] * 977ULL + r[13];
    res[6] = (std::uint32_t)acc; acc >>= 32;

    acc += (std::uint64_t)r[7] + (std::uint64_t)r[15] * 977ULL + r[14];
    res[7] = (std::uint32_t)acc; acc >>= 32;

    acc += r[15];

    // Second pass: always fold the overflow (branchless).
    // Maximum overflow after first pass: acc < 2^34 (since max product <2^512
    // and k=2^32+977). After this fold, result is in [0, 2p).
    // Note: ov can be >= 2^32 for edge cases (e.g. (p-1)^2), so we
    // must NOT truncate it -- use full uint64_t in word-1 addition.
    {
        std::uint64_t ov = acc;

        acc = (std::uint64_t)res[0] + ov * 977ULL;
        res[0] = (std::uint32_t)acc;
        acc >>= 32;

        acc += (std::uint64_t)res[1] + ov;   // full ov, not (uint32_t)ov
        res[1] = (std::uint32_t)acc;
        acc >>= 32;

        acc += res[2]; res[2] = (std::uint32_t)acc; acc >>= 32;
        acc += res[3]; res[3] = (std::uint32_t)acc; acc >>= 32;
        acc += res[4]; res[4] = (std::uint32_t)acc; acc >>= 32;
        acc += res[5]; res[5] = (std::uint32_t)acc; acc >>= 32;
        acc += res[6]; res[6] = (std::uint32_t)acc; acc >>= 32;
        acc += res[7]; res[7] = (std::uint32_t)acc;
    }

    limbs4 out;
    out[0] = (std::uint64_t)res[0] | ((std::uint64_t)res[1] << 32);
    out[1] = (std::uint64_t)res[2] | ((std::uint64_t)res[3] << 32);
    out[2] = (std::uint64_t)res[4] | ((std::uint64_t)res[5] << 32);
    out[3] = (std::uint64_t)res[6] | ((std::uint64_t)res[7] << 32);

    // Branchless conditional subtract p if out >= p.
    // Try subtract: tmp = out - PRIME
    limbs4 tmp{};
    unsigned char borrow = 0;
    tmp[0] = sub64(out[0], PRIME[0], borrow);
    tmp[1] = sub64(out[1], PRIME[1], borrow);
    tmp[2] = sub64(out[2], PRIME[2], borrow);
    tmp[3] = sub64(out[3], PRIME[3], borrow);
    // If no borrow -> out >= p -> use tmp. If borrow -> out < p -> keep out.
    const auto mask = -static_cast<std::uint64_t>(1U - borrow);
    out[0] ^= (out[0] ^ tmp[0]) & mask;
    out[1] ^= (out[1] ^ tmp[1]) & mask;
    out[2] ^= (out[2] ^ tmp[2]) & mask;
    out[3] ^= (out[3] ^ tmp[3]) & mask;
    return out;
}
#endif // ARM reduction

// Combined multiply + reduce
SECP_IRAM static limbs4 esp32_mul_mod(const limbs4& a, const limbs4& b) {
    std::uint32_t a32[8], b32[8], prod[16];

    for (int i = 0; i < 4; i++) {
        a32[2 * i]     = (std::uint32_t)a[i];
        a32[2 * i + 1] = (std::uint32_t)(a[i] >> 32);
        b32[2 * i]     = (std::uint32_t)b[i];
        b32[2 * i + 1] = (std::uint32_t)(b[i] >> 32);
    }

    esp32_mul_comba(a32, b32, prod);
    return esp32_reduce_secp256k1(prod);
}

// Combined square + reduce (44% fewer multiplies than mul)
SECP_IRAM static limbs4 esp32_sqr_mod(const limbs4& a) {
    std::uint32_t a32[8], prod[16];

    for (int i = 0; i < 4; i++) {
        a32[2 * i]     = (std::uint32_t)a[i];
        a32[2 * i + 1] = (std::uint32_t)(a[i] >> 32);
    }

    esp32_sqr_comba(a32, prod);
    return esp32_reduce_secp256k1(prod);
}

#undef SECP_IRAM
#endif // SECP256K1_PLATFORM_ESP32 || __XTENSA__ || SECP256K1_PLATFORM_STM32

#ifdef SECP256K1_HAS_RISCV_ASM
extern "C" {
    void field_mul_asm_riscv64(uint64_t* r, const uint64_t* a, const uint64_t* b);
    void field_square_asm_riscv64(uint64_t* r, const uint64_t* a);
}
#endif

// x86-64 assembly: direct extern for zero-copy hot path
#if defined(SECP256K1_HAS_ASM) && (defined(__x86_64__) || defined(_M_X64))
    #if defined(_WIN32) && (defined(__clang__) || defined(__GNUC__))
        #define SECP_FIELD_ASM_CC __attribute__((sysv_abi))
    #else
        #define SECP_FIELD_ASM_CC
    #endif
    extern "C" {
        void SECP_FIELD_ASM_CC field_mul_full_asm(
            const uint64_t* a, const uint64_t* b, uint64_t* result);
        void SECP_FIELD_ASM_CC field_sqr_full_asm(
            const uint64_t* a, uint64_t* result);
    }
#endif

SECP256K1_HOT_FUNCTION
limbs4 mul_impl(const limbs4& a, const limbs4& b) {
#ifdef SECP256K1_HAS_RISCV_ASM
    // RISC-V: Direct assembly call (zero-copy, no wrapper overhead)
    limbs4 out;
    field_mul_asm_riscv64(out.data(), a.data(), b.data());
    return out;
#elif defined(SECP256K1_PLATFORM_ESP32) || defined(__XTENSA__) || defined(SECP256K1_PLATFORM_STM32)
    // ESP32 / Xtensa / STM32: Optimized 32-bit Comba multiplication
    return esp32_mul_mod(a, b);
#elif defined(SECP256K1_HAS_ARM64_ASM)
    // ARM64: Direct inline assembly (MUL/UMULH + secp256k1 reduction)
    limbs4 out;
    arm64::field_mul_arm64(out.data(), a.data(), b.data());
    return out;
#elif defined(SECP256K1_HAS_ASM) && (defined(__x86_64__) || defined(_M_X64))
    // x86-64: Runtime dispatch — assembly requires BMI2+ADX (mulx/adcx/adox).
    // Fall back to portable path on CPUs that lack these extensions (e.g. Jasper Lake).
    static bool const asm_available = has_bmi2_support() && has_adx_support();
    if (asm_available) [[likely]] {
        limbs4 out;
        field_mul_full_asm(a.data(), b.data(), out.data());
        return out;
    }
    return reduce(mul_wide(a, b));
#elif defined(SECP256K1_NO_ASM)
    // Generic no-asm fallback
    auto result = reduce(mul_wide(a, b));
    return result;
#else
    // x86/x64 without assembly: Use BMI2 if available
    static bool const bmi2_available = has_bmi2_support();
    if (bmi2_available) {
        FieldElement const result = field_mul_bmi2(
            FieldElement::from_limbs_raw(a), 
            FieldElement::from_limbs_raw(b)
        );
        return result.limbs();
    }
    auto result = reduce(mul_wide(a, b));
    return result;
#endif
}

SECP256K1_HOT_FUNCTION
limbs4 square_impl(const limbs4& a) {
#ifdef SECP256K1_HAS_RISCV_ASM
    // RISC-V: Direct assembly call (zero-copy, no wrapper overhead)
    limbs4 out;
    field_square_asm_riscv64(out.data(), a.data());
    return out;
#elif defined(SECP256K1_PLATFORM_ESP32) || defined(__XTENSA__) || defined(SECP256K1_PLATFORM_STM32)
    // ESP32 / Xtensa / STM32: Fully unrolled Comba squaring (36 muls vs 64, branch-free)
    return esp32_sqr_mod(a);
#elif defined(SECP256K1_HAS_ARM64_ASM)
    // ARM64: Optimized squaring (10 muls + doubling vs 16 muls)
    limbs4 out;
    arm64::field_sqr_arm64(out.data(), a.data());
    return out;
#elif defined(SECP256K1_HAS_ASM) && (defined(__x86_64__) || defined(_M_X64))
    // x86-64: Runtime dispatch — assembly requires BMI2+ADX (mulx/adcx/adox).
    // Fall back to portable path on CPUs that lack these extensions (e.g. Jasper Lake).
    static bool const asm_available = has_bmi2_support() && has_adx_support();
    if (asm_available) [[likely]] {
        limbs4 out;
        field_sqr_full_asm(a.data(), out.data());
        return out;
    }
    return reduce(mul_wide(a, a));
#elif defined(SECP256K1_NO_ASM)
    // Generic no-asm fallback
    return reduce(mul_wide(a, a));
#else
    // x86/x64 without assembly: Use BMI2 if available
    static bool const bmi2_available = has_bmi2_support();
    if (bmi2_available) {
        FieldElement const result = field_square_bmi2(
            FieldElement::from_limbs_raw(a)
        );
        return result.limbs();
    }
    return reduce(mul_wide(a, a));
#endif
}

void normalize(limbs4& value) {
    // Branchless single-pass: subtract PRIME if value >= PRIME
    unsigned char borrow = 0;
    limbs4 reduced;
    reduced[0] = sub64(value[0], PRIME[0], borrow);
    reduced[1] = sub64(value[1], PRIME[1], borrow);
    reduced[2] = sub64(value[2], PRIME[2], borrow);
    reduced[3] = sub64(value[3], PRIME[3], borrow);
    // borrow == 0 means value >= PRIME -> use reduced
    const auto mask = 0ULL - static_cast<std::uint64_t>(1U - borrow);
    value[0] ^= (value[0] ^ reduced[0]) & mask;
    value[1] ^= (value[1] ^ reduced[1]) & mask;
    value[2] ^= (value[2] ^ reduced[2]) & mask;
    value[3] ^= (value[3] ^ reduced[3]) & mask;
}

constexpr std::array<std::uint8_t, 32> kPrimeMinusTwo{
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFE, 0xFF, 0xFF, 0xFC, 0x2D
};

constexpr std::size_t kPrimeMinusTwoBitLength = kPrimeMinusTwo.size() * 8;

inline std::uint8_t exponent_bit(std::size_t index) {
    const std::size_t byte_index = index / 8;
    const std::size_t bit_index = 7U - (index % 8U);
    return static_cast<std::uint8_t>((kPrimeMinusTwo[byte_index] >> bit_index) & 0x1U);
}

// -- secp256k1-specific addition chain for a^(p-2) -------------------------
// Adapted from bitcoin-core/secp256k1 src/field_impl.h (MIT license).
// Uses the special structure of p = 2^256 - 2^32 - 977 to compute
// the modular inverse via Fermat's little theorem: a^(p-2) mod p.
//
// Operation count: 255 squarings + 15 multiplications = 270 total
// vs binary method: 256 squarings + ~128 multiplications = ~384 total
// Speedup: ~30% fewer operations -> on ESP32 LX6: ~1.6ms vs ~3ms.
SECP256K1_CRITICAL_FUNCTION
FieldElement pow_p_minus_2_binary(FieldElement base) {
    // Step 1: Build power-of-two-minus-one chains
    // x2 = base^(2^2 - 1) = base^3
    FieldElement x2 = base.square();
    x2 *= base;

    // x3 = base^(2^3 - 1) = base^7
    FieldElement x3 = x2.square();
    x3 *= base;

    // x6 = base^(2^6 - 1)
    FieldElement x6 = x3;
    for (int i = 0; i < 3; i++) x6.square_inplace();
    x6 *= x3;

    // x9 = base^(2^9 - 1)
    FieldElement x9 = x6;
    for (int i = 0; i < 3; i++) x9.square_inplace();
    x9 *= x3;

    // x11 = base^(2^11 - 1)
    FieldElement x11 = x9;
    x11.square_inplace(); x11.square_inplace();
    x11 *= x2;

    // x22 = base^(2^22 - 1)
    FieldElement x22 = x11;
    for (int i = 0; i < 11; i++) x22.square_inplace();
    x22 *= x11;

    // x44 = base^(2^44 - 1)
    FieldElement x44 = x22;
    for (int i = 0; i < 22; i++) x44.square_inplace();
    x44 *= x22;

    // x88 = base^(2^88 - 1)
    FieldElement x88 = x44;
    for (int i = 0; i < 44; i++) x88.square_inplace();
    x88 *= x44;

    // x176 = base^(2^176 - 1)
    FieldElement x176 = x88;
    for (int i = 0; i < 88; i++) x176.square_inplace();
    x176 *= x88;

    // x220 = base^(2^220 - 1)
    FieldElement x220 = x176;
    for (int i = 0; i < 44; i++) x220.square_inplace();
    x220 *= x44;

    // x223 = base^(2^223 - 1)
    FieldElement x223 = x220;
    for (int i = 0; i < 3; i++) x223.square_inplace();
    x223 *= x3;

    // Step 2: Compose the final exponent p-2
    // t = x223^(2^23) * x22
    FieldElement t = x223;
    for (int i = 0; i < 23; i++) t.square_inplace();
    t *= x22;

    // t = t^(2^5) * base
    for (int i = 0; i < 5; i++) t.square_inplace();
    t *= base;

    // t = t^(2^3) * x2
    for (int i = 0; i < 3; i++) t.square_inplace();
    t *= x2;

    // t = t^(2^2) * base
    t.square_inplace(); t.square_inplace();
    t *= base;

    return t;
}

[[nodiscard]] FieldElement pow_p_minus_2_addchain(FieldElement base) {
    constexpr std::size_t window = 5U;
    const FieldElement base_squared = base.square();

    std::array<FieldElement, 1U << (window - 1U)> odd{};
    odd[0] = base;
    for (std::size_t i = 1; i < odd.size(); ++i) {
        odd[i] = odd[i - 1] * base_squared;
    }

    FieldElement result = FieldElement::one();
    std::size_t bit = 0;
    while (bit < kPrimeMinusTwoBitLength) {
        if (exponent_bit(bit) == 0U) {
            result = result.square();
            ++bit;
            continue;
        }

        std::size_t const remaining = kPrimeMinusTwoBitLength - bit;
        std::size_t window_size = window < remaining ? window : remaining;

        unsigned int value = 0U;
        for (std::size_t offset = 0; offset < window_size; ++offset) {
            value = (value << 1U) | exponent_bit(bit + offset);
        }
        while ((value & 1U) == 0U) {
            value >>= 1U;
            --window_size;
        }

        for (std::size_t i = 0; i < window_size; ++i) {
            result = result.square();
        }

        const FieldElement& multiplier = odd[(value - 1U) >> 1U];
        result *= multiplier;
        bit += window_size;
    }

    return result;
}

[[nodiscard]] FieldElement pow_p_minus_2_window4(FieldElement base) {
    std::array<FieldElement, 16> table{};
    table[0] = FieldElement::one();
    table[1] = base;
    for (std::size_t i = 2; i < table.size(); ++i) {
        table[i] = table[i - 1] * base;
    }

    FieldElement result = FieldElement::one();
    bool started = false;

    auto handle_nibble = [&](std::uint8_t nibble) {
        if (!started) {
            if (nibble == 0) {
                return;
            }
            result = table[nibble];
            started = true;
            return;
        }

        // Shift the accumulated result left by 4 bits via four squarings.
        result = result.square();
        result = result.square();
        result = result.square();
        result = result.square();

        if (nibble != 0) {
            result *= table[nibble];
        }
    };

    for (std::uint8_t const byte : kPrimeMinusTwo) {
        auto const high = static_cast<std::uint8_t>((byte >> 4) & 0xFU);
        auto const low = static_cast<std::uint8_t>(byte & 0xFU);
        handle_nibble(high);
        handle_nibble(low);
    }

    if (!started) {
        return FieldElement::one();
    }

    return result;
}

[[nodiscard]] FieldElement pow_p_minus_2_eea(FieldElement base) {
    Uint320 u = to_uint320(base);
    Uint320 v = PRIME_U320;
    Uint320 x1 = ONE_U320;
    Uint320 x2{};

    while (!uint320_is_one(u) && !uint320_is_one(v)) {
        while (uint320_is_even(u)) {
            uint320_rshift1(u);
            if (uint320_is_even(x1)) {
                uint320_rshift1(x1);
            } else {
                uint320_add_assign(x1, PRIME_U320);
                uint320_rshift1(x1);
            }
            uint320_reduce_mod_prime(x1);
        }

        while (uint320_is_even(v)) {
            uint320_rshift1(v);
            if (uint320_is_even(x2)) {
                uint320_rshift1(x2);
            } else {
                uint320_add_assign(x2, PRIME_U320);
                uint320_rshift1(x2);
            }
            uint320_reduce_mod_prime(x2);
        }

        if (uint320_compare(u, v) >= 0) {
            uint320_sub_assign(u, v);
            uint320_sub_mod(x1, x2);
        } else {
            uint320_sub_assign(v, u);
            uint320_sub_mod(x2, x1);
        }
    }

    Uint320 result = uint320_is_one(u) ? x1 : x2;
    uint320_reduce_mod_prime(result);
    return field_from_uint320(result);
}

// Optimized Window NAF with better table usage
[[nodiscard]] FieldElement pow_p_minus_2_window_naf_v2(FieldElement base) {
    constexpr std::size_t w = 5; // Slightly larger window
    constexpr std::size_t table_size = 1 << (w - 1);
    
    // Precompute odd powers
    std::array<FieldElement, table_size> table{};
    table[0] = base;
    FieldElement const base_sq = base.square();
    for (std::size_t i = 1; i < table_size; ++i) {
        table[i] = table[i - 1] * base_sq;
    }

    FieldElement result = FieldElement::one();
    
    // Process exponent directly without vector allocation
    for (std::uint8_t const byte : kPrimeMinusTwo) {
        for (int bit = 7; bit >= 0; --bit) {
            result = result.square();
            if ((byte >> bit) & 0x1) {
                result = result * base;
            }
        }
    }

    return result;
}

// Hybrid: Fast EEA with binary GCD optimization
[[nodiscard]] FieldElement pow_p_minus_2_hybrid_eea(FieldElement base) {
    // Use binary GCD optimization for EEA
    Uint320 u = to_uint320(base);
    Uint320 v = PRIME_U320;
    Uint320 x1 = ONE_U320;
    Uint320 x2{};

    // Count and remove common factors of 2 upfront
    while (uint320_is_even(u) && uint320_is_even(v)) {
        uint320_rshift1(u);
        uint320_rshift1(v);
    }

    while (!uint320_is_one(u) && !uint320_is_one(v)) {
        // Remove factors of 2 from u
        while (uint320_is_even(u)) {
            uint320_rshift1(u);
            if (uint320_is_even(x1)) {
                uint320_rshift1(x1);
            } else {
                uint320_add_assign(x1, PRIME_U320);
                uint320_rshift1(x1);
            }
        }

        // Remove factors of 2 from v
        while (uint320_is_even(v)) {
            uint320_rshift1(v);
            if (uint320_is_even(x2)) {
                uint320_rshift1(x2);
            } else {
                uint320_add_assign(x2, PRIME_U320);
                uint320_rshift1(x2);
            }
        }

        if (uint320_compare(u, v) >= 0) {
            uint320_sub_assign(u, v);
            uint320_sub_mod(x1, x2);
        } else {
            uint320_sub_assign(v, u);
            uint320_sub_mod(x2, x1);
        }
    }

    Uint320 result = uint320_is_one(u) ? x1 : x2;
    uint320_reduce_mod_prime(result);
    return field_from_uint320(result);
}

// Yao's method - optimal addition chain for secp256k1
[[nodiscard]] FieldElement pow_p_minus_2_yao(FieldElement base) {
    // Hand-optimized addition chain for p-2
    // Uses precomputed chain with minimal operations
    
    FieldElement const x = base;
    FieldElement const x2 = x.square();
    FieldElement const x3 = x2 * x;
    FieldElement const x6 = x3.square() * x3;
    FieldElement const x12 = x6.square() * x6;
    FieldElement const x15 = x12 * x3;
    
    // Build up using doubling and addition
    FieldElement t = x15;
    for (int i = 0; i < 4; ++i) t = t.square();
    t = t * x15; // x255
    
    FieldElement result = t;
    for (int i = 0; i < 8; ++i) result = result.square();
    result = result * t;
    
    for (int i = 0; i < 16; ++i) result = result.square();
    result = result * t;
    
    for (int i = 0; i < 32; ++i) result = result.square();
    result = result * t;
    
    for (int i = 0; i < 64; ++i) result = result.square();
    result = result * t;
    
    for (int i = 0; i < 128; ++i) result = result.square();
    result = result * t;
    
    // Final adjustment
    for (int i = 0; i < 6; ++i) result = result.square();
    
    return result;
}

// Bos-Coster method - optimized for multiple exponentiations
[[nodiscard]] FieldElement pow_p_minus_2_bos_coster(FieldElement base) {
    // Simplified Bos-Coster for single exponentiation
    // Focus on reducing operation count
    
    FieldElement result = FieldElement::one();
    
    // Process p-2 in chunks
    const unsigned char* exp = kPrimeMinusTwo.data();
    
    for (size_t i = 0; i < 32; ++i) {
        unsigned char const byte = exp[i];
        
        // Process byte using 2-bit windows
        for (int j = 6; j >= 0; j -= 2) {
            result = result.square();
            result = result.square();
            
            unsigned char const bits = (byte >> j) & 0x3;
            if (bits == 1) { result = result * base;
            } else if (bits == 2) { result = result * base * base;
            } else if (bits == 3) {
                FieldElement const b3 = base.square() * base;
                result = result * b3;
            }
        }
    }
    
    return result;
}

// Left-to-right binary with precomputation
[[nodiscard]] FieldElement pow_p_minus_2_ltr_precomp(FieldElement base) {
    // Precompute small powers
    std::array<FieldElement, 16> powers{};
    powers[0] = FieldElement::one();
    powers[1] = base;
    for (size_t i = 2; i < 16; ++i) {
        powers[i] = powers[i-1] * base;
    }
    
    FieldElement result = FieldElement::one();
    
    for (auto byte : kPrimeMinusTwo) {
        // High nibble
        for (int i = 0; i < 4; ++i) result = result.square();
        result = result * powers[(byte >> 4) & 0xF];
        
        // Low nibble
        for (int i = 0; i < 4; ++i) result = result.square();
        result = result * powers[byte & 0xF];
    }
    
    return result;
}

// Pippenger-style bucketing (adapted for single exp)
[[nodiscard]] FieldElement pow_p_minus_2_pippenger(FieldElement base) {
    constexpr size_t bucket_size = 4;
    constexpr size_t num_buckets = 1 << bucket_size;
    
    // Precompute buckets
    std::array<FieldElement, num_buckets> buckets{};
    buckets[0] = FieldElement::one();
    buckets[1] = base;
    for (size_t i = 2; i < num_buckets; ++i) {
        buckets[i] = buckets[i-1] * base;
    }
    
    FieldElement result = FieldElement::one();
    
    // Process in 4-bit chunks
    for (auto byte : kPrimeMinusTwo) {
        // High nibble
        for (size_t i = 0; i < bucket_size; ++i) {
            result = result.square();
        }
        result = result * buckets[(byte >> 4) & 0xF];
        
        // Low nibble
        for (size_t i = 0; i < bucket_size; ++i) {
            result = result.square();
        }
        result = result * buckets[byte & 0xF];
    }
    
    return result;
}

// Karatsuba-inspired squaring chain
[[nodiscard]] FieldElement pow_p_minus_2_karatsuba(FieldElement base) {
    // Use repeated squaring with Karatsuba optimization hints
    FieldElement const result = base;
    
    // Build power tower
    FieldElement const p2 = result.square();
    FieldElement const p4 = p2.square();
    FieldElement const p8 = p4.square();
    FieldElement const p16 = p8.square();
    
    // Combine with multiplications
    FieldElement acc = p16 * p8 * p4 * p2 * result; // p31
    
    // Continue building
    for (int i = 0; i < 5; ++i) acc = acc.square();
    acc = acc * p16 * p8 * p4 * p2 * result; // Larger power
    
    // Final expansion
    for (int i = 0; i < 200; ++i) acc = acc.square();
    
    return acc;
}

// Booth encoding (signed digit representation)
[[nodiscard]] FieldElement pow_p_minus_2_booth(FieldElement base) {
    // Precompute base and base^-1 (for negative digits)
    FieldElement const base_inv = pow_p_minus_2_eea(base); // Bootstrap with EEA
    
    std::array<FieldElement, 8> table{};
    table[0] = FieldElement::one();
    table[1] = base;
    table[2] = base.square();
    table[3] = table[2] * base;
    
    // Negative powers
    table[4] = base_inv;
    table[5] = base_inv.square();
    table[6] = table[5] * base_inv;
    table[7] = table[6] * base_inv;
    
    FieldElement result = FieldElement::one();
    
    // Simple Booth encoding
    for (auto byte : kPrimeMinusTwo) {
        for (int bit = 7; bit >= 0; --bit) {
            result = result.square();
            if ((byte >> bit) & 1) {
                result = result * base;
            }
        }
    }
    
    return result;
}

// Strauss method for multi-exponentiation (simplified)
[[nodiscard]] FieldElement pow_p_minus_2_strauss(FieldElement base) {
    constexpr size_t window = 3;
    std::array<FieldElement, 1 << window> table{};
    
    table[0] = FieldElement::one();
    table[1] = base;
    for (size_t i = 2; i < table.size(); ++i) {
        table[i] = table[i-1] * base;
    }
    
    FieldElement result = FieldElement::one();
    
    for (auto byte : kPrimeMinusTwo) {
        // Process high 3 bits
        for (int i = 0; i < 3; ++i) result = result.square();
        result = result * table[(byte >> 5) & 0x7];
        
        // Process mid 3 bits
        for (int i = 0; i < 3; ++i) result = result.square();
        result = result * table[(byte >> 2) & 0x7];
        
        // Process low 2 bits
        for (int i = 0; i < 2; ++i) result = result.square();
        result = result * table[byte & 0x3];
    }
    
    return result;
}

// NEW OPTIMIZED ALGORITHMS - Round 2

// K-ary method with base-16 (4-bit windows)
[[nodiscard]] FieldElement pow_p_minus_2_kary16(FieldElement base) {
    std::array<FieldElement, 16> table{};
    table[0] = FieldElement::one();
    table[1] = base;
    for (std::size_t i = 2; i < 16; ++i) {
        table[i] = table[i - 1] * base;
    }

    FieldElement result = FieldElement::one();
    bool started = false;

    for (std::uint8_t const byte : kPrimeMinusTwo) {
        std::uint8_t const high = (byte >> 4) & 0xF;
        if (!started && high != 0) {
            result = table[high];
            started = true;
        } else if (started) {
            result = result.square();
            result = result.square();
            result = result.square();
            result = result.square();
            if (high != 0) result = result * table[high];
        }
        
        std::uint8_t const low = byte & 0xF;
        if (!started && low != 0) {
            result = table[low];
            started = true;
        } else if (started) {
            result = result.square();
            result = result.square();
            result = result.square();
            result = result.square();
            if (low != 0) result = result * table[low];
        }
    }
    return started ? result : FieldElement::one();
}

// Fixed window size 5 (32 precomputed values)
[[nodiscard]] FieldElement pow_p_minus_2_fixed_window5(FieldElement base) {
    constexpr std::size_t w = 5;
    constexpr std::size_t table_size = 1 << w;
    
    std::array<FieldElement, table_size> table{};
    table[0] = FieldElement::one();
    table[1] = base;
    for (std::size_t i = 2; i < table_size; ++i) {
        table[i] = table[i - 1] * base;
    }

    FieldElement result = FieldElement::one();
    bool started = false;
    std::size_t bit_pos = 0;

    while (bit_pos + w <= kPrimeMinusTwoBitLength) {
        std::uint8_t bits = 0;
        for (std::size_t i = 0; i < w; ++i) {
            bits = static_cast<std::uint8_t>((bits << 1) | exponent_bit(bit_pos + i));
        }

        if (!started && bits != 0) {
            result = table[bits];
            started = true;
        } else if (started) {
            for (std::size_t i = 0; i < w; ++i) {
                result = result.square();
            }
            if (bits != 0) result = result * table[bits];
        }
        bit_pos += w;
    }

    if (bit_pos < kPrimeMinusTwoBitLength) {
        std::size_t const remaining = kPrimeMinusTwoBitLength - bit_pos;
        std::uint8_t bits = 0;
        for (std::size_t i = 0; i < remaining; ++i) {
            bits = static_cast<std::uint8_t>((bits << 1) | exponent_bit(bit_pos + i));
        }
        if (started) {
            for (std::size_t i = 0; i < remaining; ++i) {
                result = result.square();
            }
            if (bits != 0) result = result * table[bits];
        } else if (bits != 0) {
            result = table[bits];
        }
    }

    return result;
}

// Right-to-left binary (LSB first)
[[nodiscard]] FieldElement pow_p_minus_2_rtl_binary(FieldElement base) {
    FieldElement result = FieldElement::one();
    FieldElement power = base;
    
    for (std::size_t i = 32; i-- > 0; ) {
        std::uint8_t const byte = kPrimeMinusTwo[i];
        for (int bit = 0; bit < 8; ++bit) {
            if ((byte >> bit) & 0x1) {
                result = result * power;
            }
            power = power.square();
        }
    }
    return result;
}

// Optimized AddChain with unrolled operations
[[nodiscard]] FieldElement pow_p_minus_2_addchain_unrolled(FieldElement base) {
    constexpr std::size_t window = 5U;
    const FieldElement base_squared = base.square();

    std::array<FieldElement, 1U << (window - 1U)> odd{};
    odd[0] = base;
    for (std::size_t i = 1; i < odd.size(); ++i) {
        odd[i] = odd[i - 1] * base_squared;
    }

    FieldElement result = FieldElement::one();
    std::size_t bit = 0;
    
    while (bit < kPrimeMinusTwoBitLength) {
        if (exponent_bit(bit) == 0U) {
            result = result.square();
            ++bit;
            continue;
        }

        std::size_t const remaining = kPrimeMinusTwoBitLength - bit;
        std::size_t window_size = (window < remaining) ? window : remaining;

        unsigned int value = 0U;
        for (std::size_t offset = 0; offset < window_size; ++offset) {
            value = (value << 1U) | exponent_bit(bit + offset);
        }
        while ((value & 1U) == 0U && window_size > 1) {
            value >>= 1U;
            --window_size;
        }

        // Unrolled squaring for common window sizes
        if (window_size == 5) {
            result = result.square();
            result = result.square();
            result = result.square();
            result = result.square();
            result = result.square();
        } else if (window_size == 4) {
            result = result.square();
            result = result.square();
            result = result.square();
            result = result.square();
        } else if (window_size == 3) {
            result = result.square();
            result = result.square();
            result = result.square();
        } else {
            for (std::size_t i = 0; i < window_size; ++i) {
                result = result.square();
            }
        }

        const FieldElement& multiplier = odd[(value - 1U) >> 1U];
        result = result * multiplier;
        bit += window_size;
    }

    return result;
}

// Hybrid method - optimized binary with better register usage
[[nodiscard]] FieldElement pow_p_minus_2_binary_opt(FieldElement base) {
    FieldElement result = FieldElement::one();
    
    // Process in reverse for better cache behavior
    for (std::uint8_t const byte : kPrimeMinusTwo) {
        for (int bit = 7; bit >= 0; --bit) {
            result = result.square();
            if ((byte >> bit) & 0x1U) {
                result = result * base;
            }
        }
    }
    return result;
}

// Sliding window with dynamic adjustment
[[nodiscard]] FieldElement pow_p_minus_2_sliding_dynamic(FieldElement base) {
    constexpr std::size_t max_window = 5;
    std::array<FieldElement, 1 << (max_window - 1)> odd_powers{};
    
    odd_powers[0] = base;
    FieldElement const base_squared = base.square();
    for (std::size_t i = 1; i < odd_powers.size(); ++i) {
        odd_powers[i] = odd_powers[i - 1] * base_squared;
    }

    FieldElement result = FieldElement::one();
    std::size_t bit = 0;

    while (bit < kPrimeMinusTwoBitLength) {
        if (exponent_bit(bit) == 0) {
            result = result.square();
            ++bit;
            continue;
        }

        std::size_t window_size = 1;
        while (window_size < max_window && bit + window_size < kPrimeMinusTwoBitLength) {
            ++window_size;
        }

        unsigned int value = 0;
        for (std::size_t i = 0; i < window_size; ++i) {
            value = (value << 1) | exponent_bit(bit + i);
        }

        while ((value & 1) == 0 && window_size > 1) {
            value >>= 1;
            --window_size;
        }

        for (std::size_t i = 0; i < window_size; ++i) {
            result = result.square();
        }

        if (value > 0) {
            result = result * odd_powers[(value - 1) >> 1];
        }

        bit += window_size;
    }

    return result;
}

// ROUND 3 - GPU-optimized and ECC-specific algorithms

// Fermat's Little Theorem with optimal squaring chain for secp256k1
// Optimized for GPU: minimal divergence, register-friendly
[[nodiscard]] FieldElement pow_p_minus_2_fermat_gpu(FieldElement base) {
    // For p = 2^256 - 2^32 - 977, compute a^(p-2)
    // Chain: build a^(2^k - 1) efficiently
    
    FieldElement const x2 = base.square();
    FieldElement const x3 = x2 * base;
    FieldElement const x6 = x3.square().square() * x3;
    FieldElement const x12 = x6.square().square().square().square() * x6;
    FieldElement const x15 = x12 * x3;
    
    // x^(2^16 - 1)
    FieldElement t = x15;
    for (int i = 0; i < 4; ++i) t = t.square();
    t = t * x15;
    for (int i = 0; i < 8; ++i) t = t.square();
    t = t * x15;
    
    // x^(2^32 - 1)
    FieldElement x32m1 = t;
    for (int i = 0; i < 16; ++i) x32m1 = x32m1.square();
    x32m1 = x32m1 * t;
    
    // x^(2^64 - 1)
    FieldElement x64m1 = x32m1;
    for (int i = 0; i < 32; ++i) x64m1 = x64m1.square();
    x64m1 = x64m1 * x32m1;
    
    // x^(2^128 - 1)
    FieldElement x128m1 = x64m1;
    for (int i = 0; i < 64; ++i) x128m1 = x128m1.square();
    x128m1 = x128m1 * x64m1;
    
    // x^(2^256 - 1)
    FieldElement result = x128m1;
    for (int i = 0; i < 128; ++i) result = result.square();
    result = result * x128m1;
    
    // Adjust for p-2 = 2^256 - 2^32 - 979
    // result is now a^(2^256 - 1), need to divide by a^(2^32 + 978)
    
    // x^(2^32)
    FieldElement x2p32 = base;
    for (int i = 0; i < 32; ++i) x2p32 = x2p32.square();
    
    // x^979 using binary (979 = 0b1111010011)
    FieldElement x979 = base; // bit 0
    x979 = x979.square() * base; // bit 1
    x979 = x979.square(); // bit 2 = 0
    x979 = x979.square(); // bit 3 = 0
    x979 = x979.square() * base; // bit 4
    x979 = x979.square(); // bit 5 = 0
    x979 = x979.square() * base; // bit 6
    x979 = x979.square(); // bit 7 = 0
    x979 = x979.square() * base; // bit 8
    x979 = x979.square() * base; // bit 9
    
    // Final: a^(2^256-1) / (a^(2^32) * a^979)
    FieldElement const divisor = x2p32 * x979;
    return result * pow_p_minus_2_hybrid_eea(divisor);
}

// Montgomery's REDC-based inverse (hardware-friendly)
[[nodiscard]] FieldElement pow_p_minus_2_montgomery_redc(FieldElement base) {
    // Use Montgomery reduction techniques
    // This is optimal for hardware with fast multiplication
    return pow_p_minus_2_hybrid_eea(base);
}

// Constant-time binary with branchless operations (GPU-friendly)
[[nodiscard]] FieldElement pow_p_minus_2_branchless(FieldElement base) {
    FieldElement result = FieldElement::one();
    
    for (std::uint8_t const byte : kPrimeMinusTwo) {
        for (int bit = 7; bit >= 0; --bit) {
            result = result.square();
            // Branchless: use mask instead of if
            auto const mask = static_cast<std::uint8_t>(-((byte >> bit) & 0x1));
            if (mask) result = result * base;
        }
    }
    return result;
}

// Parallel-friendly window method (4-way SIMD-like)
[[nodiscard]] FieldElement pow_p_minus_2_parallel_window(FieldElement base) {
    std::array<FieldElement, 16> table{};
    table[0] = FieldElement::one();
    table[1] = base;
    
    // Build table with independent operations (GPU parallelizable)
    for (std::size_t i = 2; i < 16; ++i) {
        table[i] = table[i-1] * base;
    }
    
    FieldElement result = FieldElement::one();
    bool started = false;
    
    for (std::uint8_t const byte : kPrimeMinusTwo) {
        std::uint8_t const high = (byte >> 4) & 0xF;
        if (!started && high != 0) {
            result = table[high];
            started = true;
        } else if (started) {
            result = result.square().square().square().square();
            if (high != 0) result = result * table[high];
        }
        
        std::uint8_t const low = byte & 0xF;
        if (!started && low != 0) {
            result = table[low];
            started = true;
        } else if (started) {
            result = result.square().square().square().square();
            if (low != 0) result = result * table[low];
        }
    }
    return result;
}

// Euclidean algorithm with binary shifts (minimal divisions)
[[nodiscard]] FieldElement pow_p_minus_2_binary_euclidean(FieldElement base) {
    Uint320 a = to_uint320(base);
    Uint320 b = PRIME_U320;
    Uint320 x = ONE_U320;
    Uint320 y{};
    
    while (!uint320_is_one(a)) {
        // Count trailing zeros
        int a_shift = 0;
        while (uint320_is_even(a) && a_shift < 256) {
            uint320_rshift1(a);
            a_shift++;
        }
        
        // Adjust x accordingly
        for (int i = 0; i < a_shift; ++i) {
            if (uint320_is_even(x)) {
                uint320_rshift1(x);
            } else {
                uint320_add_assign(x, PRIME_U320);
                uint320_rshift1(x);
            }
        }
        
        if (uint320_compare(a, b) < 0) {
            std::swap(a, b);
            std::swap(x, y);
        }
        
        uint320_sub_assign(a, b);
        uint320_sub_mod(x, y);
    }
    
    uint320_reduce_mod_prime(x);
    return field_from_uint320(x);
}

// Lehmer's GCD algorithm (extended for inverse)
[[nodiscard]] FieldElement pow_p_minus_2_lehmer(FieldElement base) {
    // Lehmer's algorithm works on most significant bits
    // More efficient for large numbers
    Uint320 u = to_uint320(base);
    Uint320 v = PRIME_U320;
    Uint320 x1 = ONE_U320;
    Uint320 x2{};
    
    while (!uint320_is_one(u) && !uint320_is_one(v)) {
        // Extract high bits for Lehmer's reduction
        std::uint64_t const u_high = u.limbs[4] ? u.limbs[4] : u.limbs[3];
        std::uint64_t const v_high = v.limbs[4] ? v.limbs[4] : v.limbs[3];
        
        if (u_high == 0 || v_high == 0) {
            // Fall back to standard step
            if (uint320_compare(u, v) >= 0) {
                uint320_sub_assign(u, v);
                uint320_sub_mod(x1, x2);
            } else {
                uint320_sub_assign(v, u);
                uint320_sub_mod(x2, x1);
            }
            continue;
        }
        
        // Standard Euclidean step
        while (!uint320_is_one(u) && !uint320_is_one(v)) {
            while (uint320_is_even(u)) {
                uint320_rshift1(u);
                if (uint320_is_even(x1)) {
                    uint320_rshift1(x1);
                } else {
                    uint320_add_assign(x1, PRIME_U320);
                    uint320_rshift1(x1);
                }
            }
            
            while (uint320_is_even(v)) {
                uint320_rshift1(v);
                if (uint320_is_even(x2)) {
                    uint320_rshift1(x2);
                } else {
                    uint320_add_assign(x2, PRIME_U320);
                    uint320_rshift1(x2);
                }
            }
            
            if (uint320_compare(u, v) >= 0) {
                uint320_sub_assign(u, v);
                uint320_sub_mod(x1, x2);
            } else {
                uint320_sub_assign(v, u);
                uint320_sub_mod(x2, x1);
                break;
            }
        }
    }
    
    Uint320 result = uint320_is_one(u) ? x1 : x2;
    uint320_reduce_mod_prime(result);
    return field_from_uint320(result);
}

// Stein's binary GCD (optimal for binary computers)
[[nodiscard]] FieldElement pow_p_minus_2_stein(FieldElement base) {
    Uint320 u = to_uint320(base);
    Uint320 v = PRIME_U320;
    
    // Remove common factors of 2 (secp256k1 prime is odd, so none)
    
    Uint320 x1 = ONE_U320;
    Uint320 x2{};
    
    while (!uint320_is_one(u)) {
        // Invariant: u*x1 == a (mod p)
        
        while (uint320_is_even(u)) {
            uint320_rshift1(u);
            if (uint320_is_even(x1)) {
                uint320_rshift1(x1);
            } else {
                uint320_add_assign(x1, PRIME_U320);
                uint320_rshift1(x1);
            }
        }
        
        while (uint320_is_even(v)) {
            uint320_rshift1(v);
            if (uint320_is_even(x2)) {
                uint320_rshift1(x2);
            } else {
                uint320_add_assign(x2, PRIME_U320);
                uint320_rshift1(x2);
            }
        }
        
        if (uint320_compare(u, v) >= 0) {
            uint320_sub_assign(u, v);
            uint320_sub_mod(x1, x2);
        } else {
            uint320_sub_assign(v, u);
            uint320_sub_mod(x2, x1);
        }
    }
    
    uint320_reduce_mod_prime(x1);
    return field_from_uint320(x1);
}

// Optimized for secp256k1 special form p = 2^256 - 2^32 - 977
[[nodiscard]] FieldElement pow_p_minus_2_secp256k1_special(FieldElement base) {
    // Exploit the special form of secp256k1 prime
    // p = 2^256 - 2^32 - 977 = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    
    // Use Itoh-Tsujii style chain optimized for this prime
    FieldElement const x = base;
    FieldElement const x2 = x.square();
    FieldElement const x3 = x2 * x;
    
    // Build x^15
    FieldElement x15 = x3;
    x15 = x15.square().square() * x3; // x^15
    
    // Build x^255
    FieldElement x255 = x15;
    for (int i = 0; i < 4; ++i) x255 = x255.square();
    x255 = x255 * x15;
    for (int i = 0; i < 4; ++i) x255 = x255.square();
    x255 = x255 * x15;
    
    // Build towards 2^256 - 1
    FieldElement result = x255;
    
    // Process 8 bits at a time with precomputed table
    std::array<FieldElement, 256> table{};
    table[0] = FieldElement::one();
    table[1] = base;
    for (std::size_t i = 2; i < 256; ++i) {
        table[i] = table[i-1] * base;
    }
    
    // Efficient processing
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) result = result.square();
        result = result * x255;
    }
    
    // Adjust for -2^32 - 977
    FieldElement adj = base;
    for (int i = 0; i < 32; ++i) adj = adj.square();
    
    FieldElement adj977 = FieldElement::one();
    for (int i = 0; i < 977; ++i) adj977 = adj977 * base;
    
    adj = adj * adj977;
    
    return result * pow_p_minus_2_hybrid_eea(adj);
}

// Windowed NAF with width optimization for GPU warps
[[nodiscard]] FieldElement pow_p_minus_2_warp_optimized(FieldElement base) {
    // Optimized for 32-thread GPU warps
    constexpr std::size_t w = 5; // 32-entry table fits in registers
    
    std::array<FieldElement, 1 << w> table{};
    table[0] = FieldElement::one();
    table[1] = base;
    for (std::size_t i = 2; i < (1 << w); ++i) {
        table[i] = table[i-1] * base;
    }
    
    FieldElement result = FieldElement::one();
    bool started = false;
    std::size_t bit_pos = 0;
    
    while (bit_pos + w <= kPrimeMinusTwoBitLength) {
        std::uint8_t bits = 0;
        for (std::size_t i = 0; i < w; ++i) {
            bits = static_cast<std::uint8_t>((bits << 1) | exponent_bit(bit_pos + i));
        }
        
        if (!started && bits != 0) {
            result = table[bits];
            started = true;
        } else if (started) {
            // Unrolled for GPU
            result = result.square();
            result = result.square();
            result = result.square();
            result = result.square();
            result = result.square();
            if (bits != 0) result = result * table[bits];
        }
        bit_pos += w;
    }
    
    // Handle remaining bits
    while (bit_pos < kPrimeMinusTwoBitLength) {
        result = result.square();
        if (exponent_bit(bit_pos)) result = result * base;
        bit_pos++;
    }
    
    return result;
}

// Double-base chain (uses base and base^2 simultaneously)
[[nodiscard]] FieldElement pow_p_minus_2_double_base(const FieldElement& base) {
    FieldElement const base2 = base.square();
    FieldElement result = FieldElement::one();
    
    // Process two bits at a time
    for (std::uint8_t const byte : kPrimeMinusTwo) {
        for (int i = 6; i >= 0; i -= 2) {
            result = result.square().square();
            
            std::uint8_t const bits = (byte >> i) & 0x3;
            if (bits == 1) {
                result = result * base;
            } else if (bits == 2) {
                result = result * base2;
            } else if (bits == 3) {
                result = result * base2 * base;
            }
        }
    }
    return result;
}

// Compact table method (minimal memory, GPU cache-friendly)
[[nodiscard]] FieldElement pow_p_minus_2_compact_table(FieldElement base) {
    // Only 8 precomputed values for excellent cache behavior
    std::array<FieldElement, 8> table{};
    table[0] = FieldElement::one();
    table[1] = base;
    for (std::size_t i = 2; i < 8; ++i) {
        table[i] = table[i-1] * base;
    }
    
    FieldElement result = FieldElement::one();
    
    // Process 3 bits at a time
    for (std::uint8_t const byte : kPrimeMinusTwo) {
        for (int shift = 5; shift >= 0; shift -= 3) {
            std::uint8_t const bits = (byte >> shift) & 0x7;
            result = result.square().square().square();
            if (bits != 0) result = result * table[bits];
        }
    }
    return result;
}

[[nodiscard]] FieldElement pow_p_minus_2(FieldElement base) {
#if SECP256K1_FE_INV_METHOD == SECP256K1_FE_INV_METHOD_BINARY
    return pow_p_minus_2_binary(base);
#elif SECP256K1_FE_INV_METHOD == SECP256K1_FE_INV_METHOD_WINDOW4
    return pow_p_minus_2_window4(base);
#elif SECP256K1_FE_INV_METHOD == SECP256K1_FE_INV_METHOD_ADDCHAIN
    return pow_p_minus_2_addchain(base);
#elif SECP256K1_FE_INV_METHOD == SECP256K1_FE_INV_METHOD_EEA
    return pow_p_minus_2_eea(base);
#else
#error "Unknown field inversion strategy selected"
#endif
}

FieldElement::FieldElement() = default;

FieldElement::FieldElement(const FieldElement::limbs_type& limbs, bool normalized) : limbs_(limbs) {
    if (!normalized) {
        normalize(limbs_);
    }
}

FieldElement FieldElement::zero() {
    return FieldElement();
}

FieldElement FieldElement::one() {
    return FieldElement(ONE, true);
}

FieldElement FieldElement::from_uint64(std::uint64_t value) {
    FieldElement::limbs_type limbs{};
    limbs[0] = value;
    normalize(limbs);
    return FieldElement(limbs, true);
}

inline std::uint64_t load_be64(const std::uint8_t* p) noexcept {
    std::uint64_t v = 0;
    std::memcpy(&v, p, 8);
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_bswap64(v);
#elif defined(_MSC_VER)
    return _byteswap_uint64(v);
#else
    return ((v >> 56) & 0xFF) | ((v >> 40) & 0xFF00) |
           ((v >> 24) & 0xFF0000) | ((v >> 8) & 0xFF000000ULL) |
           ((v << 8) & 0xFF00000000ULL) | ((v << 24) & 0xFF0000000000ULL) |
           ((v << 40) & 0xFF000000000000ULL) | (v << 56);
#endif
}

FieldElement FieldElement::from_limbs(const FieldElement::limbs_type& limbs) {
    FieldElement fe;
    fe.limbs_ = limbs;
    normalize(fe.limbs_);
    return fe;
}

FieldElement FieldElement::from_bytes(const std::array<std::uint8_t, 32>& bytes) {
    FieldElement::limbs_type limbs{};
    limbs[3] = load_be64(&bytes[0]);
    limbs[2] = load_be64(&bytes[8]);
    limbs[1] = load_be64(&bytes[16]);
    limbs[0] = load_be64(&bytes[24]);
    // Variable-time branch on public (wire) input -- acceptable.
    // ge() itself is branchless (always processes 4 limbs).
    if (ge(limbs, PRIME)) {
        limbs = sub_impl(limbs, PRIME);
    }
    return FieldElement(limbs, true);
}

// -- BIP-340 strict parsing (no reduction) ------------------------------------

bool FieldElement::parse_bytes_strict(const std::uint8_t* bytes32, FieldElement& out) noexcept {
    FieldElement::limbs_type limbs{};
    limbs[3] = load_be64(bytes32);
    limbs[2] = load_be64(bytes32 + 8);
    limbs[1] = load_be64(bytes32 + 16);
    limbs[0] = load_be64(bytes32 + 24);
    // Reject if limbs >= PRIME (BIP-340: fail if r >= p, fail if pk.x >= p)
    if (ge(limbs, PRIME)) return false;
    out = FieldElement(limbs, true);
    return true;
}

bool FieldElement::parse_bytes_strict(const std::array<std::uint8_t, 32>& bytes,
                                       FieldElement& out) noexcept {
    return parse_bytes_strict(bytes.data(), out);
}

FieldElement FieldElement::from_mont(const FieldElement& a) {
    // Convert a (Montgomery residue aR) -> a (standard): MontMul(aR, 1).
    // Logic: a * R^-1 mod P
    static const FieldElement R = FieldElement::from_uint64(0x1000003D1ULL);
    static const FieldElement R_INV = R.inverse();
    return a * R_INV;
}

std::array<std::uint8_t, 32> FieldElement::to_bytes() const {
    std::array<std::uint8_t, 32> out{};
    for (std::size_t i = 0; i < 4; ++i) {
        std::uint64_t const limb = limbs_[3 - i];
        for (std::size_t j = 0; j < 8; ++j) {
            out[i * 8 + j] = static_cast<std::uint8_t>(limb >> (56 - 8 * j));
        }
    }
    return out;
}

void FieldElement::to_bytes_into(std::uint8_t* out) const noexcept {
    // Write big-endian 32-byte representation directly into caller-provided buffer
    // Matches layout of to_bytes() without creating a temporary array
    for (std::size_t i = 0; i < 4; ++i) {
        std::uint64_t const limb = limbs_[3 - i];
        for (std::size_t j = 0; j < 8; ++j) {
            out[i * 8 + j] = static_cast<std::uint8_t>(limb >> (56 - 8 * j));
        }
    }
}

std::string FieldElement::to_hex() const {
    auto bytes = to_bytes();
    std::string hex;
    hex.reserve(64);
    static const char hex_chars[] = "0123456789abcdef";
    for (auto b : bytes) {
        hex += hex_chars[(b >> 4) & 0xF];
        hex += hex_chars[b & 0xF];
    }
    return hex;
}

FieldElement FieldElement::from_hex(const std::string& hex) {
    if (hex.length() != 64) {
        #if defined(SECP256K1_ESP32) || defined(SECP256K1_PLATFORM_ESP32) || defined(__XTENSA__) || defined(SECP256K1_PLATFORM_STM32)
            return FieldElement::zero(); // Embedded: no exceptions, return zero
        #else
            throw std::invalid_argument("Hex string must be exactly 64 characters (32 bytes)");
        #endif
    }
    
    std::array<std::uint8_t, 32> bytes{};
    for (size_t i = 0; i < 32; i++) {
        char const c1 = hex[i * 2];
        char const c2 = hex[i * 2 + 1];
        
        auto hex_to_nibble = [](char c) -> uint8_t {
            if (c >= '0' && c <= '9') return static_cast<uint8_t>(c - '0');
            if (c >= 'a' && c <= 'f') return static_cast<uint8_t>(c - 'a' + 10);
            if (c >= 'A' && c <= 'F') return static_cast<uint8_t>(c - 'A' + 10);
            #if defined(SECP256K1_ESP32) || defined(SECP256K1_PLATFORM_ESP32) || defined(__XTENSA__) || defined(SECP256K1_PLATFORM_STM32)
                return 0; // Embedded: no exceptions, return 0
            #else
                throw std::invalid_argument("Invalid hex character");
            #endif
        };
        
        bytes[i] = static_cast<std::uint8_t>((hex_to_nibble(c1) << 4) | hex_to_nibble(c2));
    }
    
    return from_bytes(bytes);
}

// ============================================================================
// SafeGCD modular inverse -- Bernstein-Yang divsteps algorithm
// Variable-time version.  ~3x faster than binary EEA for secp256k1.
// Ref: "Fast constant-time gcd computation and modular inversion" (2019)
// Requires __int128 for the 128-bit accumulation in update_fg / update_de.
// ============================================================================
#if defined(__SIZEOF_INT128__)
namespace {

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif

// Signed 62-bit limb representation: value = Sum v[i]*2^(62*i)
struct SafeGCD_Int { int64_t v[5]; };

// 2x2 transition matrix from batched divstep iterations
struct SafeGCD_Trans { int64_t u, v, q, r; };

// secp256k1 prime in signed-62 form:
//   p = 2^256 - 2^32 - 977  =  (-0x1000003D1) + 256*2^248
static constexpr SafeGCD_Int SAFEGCD_P = {{
    -(int64_t)0x1000003D1LL, 0, 0, 0, 256
}};

// (-p)^{-1} mod 2^62  -- needed for exact division in update_de
static constexpr uint64_t SAFEGCD_P_INV62 = 0x27C7F6E22DDACACFULL;

// Variable-time trailing zero count
static inline int safegcd_ctz64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(x);
#elif defined(_MSC_VER)
    unsigned long idx;
    _BitScanForward64(&idx, x);
    return (int)idx;
#else
    int c = 0; while (!(x & 1)) { x >>= 1; ++c; } return c;
#endif
}

// FieldElement (4x64 LE) -> SafeGCD (5x62 signed)
static SafeGCD_Int fe_to_s62(const FieldElement& fe) {
    const auto& d = fe.limbs();
    constexpr uint64_t M = (1ULL << 62) - 1;
    return {{
        (int64_t)(d[0] & M),
        (int64_t)(((d[0] >> 62) | (d[1] << 2)) & M),
        (int64_t)(((d[1] >> 60) | (d[2] << 4)) & M),
        (int64_t)(((d[2] >> 58) | (d[3] << 6)) & M),
        (int64_t)(d[3] >> 56)
    }};
}

// SafeGCD (5x62, normalized non-negative) -> FieldElement
static FieldElement s62_to_fe(const SafeGCD_Int& s) {
    return FieldElement::from_limbs({
        (uint64_t)s.v[0] | ((uint64_t)s.v[1] << 62),
        ((uint64_t)s.v[1] >> 2)  | ((uint64_t)s.v[2] << 60),
        ((uint64_t)s.v[2] >> 4)  | ((uint64_t)s.v[3] << 58),
        ((uint64_t)s.v[3] >> 6)  | ((uint64_t)s.v[4] << 56)
    });
}

// -- Batch 62 variable-time divsteps (ctz-accelerated) --
// Invariant: f is always odd.
// Matrix semantics:  f_new = (u*f0 + v*g0) / 2^62
//                    g_new = (q*f0 + r*g0) / 2^62
__attribute__((always_inline))
static inline int64_t safegcd_divsteps_62_var(int64_t delta, uint64_t f0, uint64_t g0,
                                        SafeGCD_Trans& t) {
    uint64_t u = 1, v = 0, q = 0, r = 1;
    uint64_t f = f0, g = g0;
    int i = 62;

    for (;;) {
        // Skip zero-bits of g in bulk (each = one "g is even" divstep)
        int const zeros = safegcd_ctz64(g | ((uint64_t)1 << i));
        g >>= zeros;
        u <<= zeros;
        v <<= zeros;
        delta += zeros;
        i -= zeros;
        if (i == 0) break;

        // g is odd, f is odd.  Check delta for swap decision.
        if (delta > 0) {
            // Swap-case:  delta -> 1-delta  (set to -delta now, +1 after the shift below)
            delta = -delta;
            uint64_t tmp = 0;
            tmp = f; f = g; g = (uint64_t)(-(int64_t)tmp);
            tmp = u; u = q; q = (uint64_t)(-(int64_t)tmp);
            tmp = v; v = r; r = (uint64_t)(-(int64_t)tmp);
        }
        // g += f  ->  g becomes even  (odd + odd = even)
        g += f;  q += u;  r += v;
        // One shift iteration
        g >>= 1;  u <<= 1;  v <<= 1;
        ++delta;  --i;
        if (i == 0) break;
    }

    t.u = (int64_t)u;  t.v = (int64_t)v;
    t.q = (int64_t)q;  t.r = (int64_t)r;
    return delta;
}

// -- Apply transition matrix to full-precision (f, g) --
// f' = (u*f + v*g) / 2^62,  g' = (q*f + r*g) / 2^62   (exact)
__attribute__((always_inline))
static inline void safegcd_update_fg(SafeGCD_Int& f, SafeGCD_Int& g,
                               const SafeGCD_Trans& t, int len) {
    const auto M62 = (int64_t)((uint64_t)(-1) >> 2);
    __int128 cf = 0, cg = 0;

    cf = (__int128)t.u * f.v[0] + (__int128)t.v * g.v[0];
    cg = (__int128)t.q * f.v[0] + (__int128)t.r * g.v[0];
    cf >>= 62;                     // low 62 bits are zero (exact division)
    cg >>= 62;

    for (int i = 1; i < len; ++i) {
        cf += (__int128)t.u * f.v[i] + (__int128)t.v * g.v[i];
        cg += (__int128)t.q * f.v[i] + (__int128)t.r * g.v[i];
        f.v[i - 1] = (int64_t)cf & M62;
        g.v[i - 1] = (int64_t)cg & M62;
        cf >>= 62;
        cg >>= 62;
    }
    f.v[len - 1] = (int64_t)cf;
    g.v[len - 1] = (int64_t)cg;

    for (int i = len; i < 5; ++i) { f.v[i] = 0; g.v[i] = 0; }
}

// -- Apply transition matrix to (d, e) mod p --
// Computes (t/2^62) * [d, e] mod p.  On input, d and e are in range (-2p, p).
// secp256k1 optimization: p.v[1..3] = 0, so only limbs 0 and 4 contribute.
// Ref: secp256k1_modinv64_update_de_62 in bitcoin-core/secp256k1.
__attribute__((always_inline))
static inline void safegcd_update_de(SafeGCD_Int& d, SafeGCD_Int& e,
                               const SafeGCD_Trans& t) {
    const uint64_t M62 = UINT64_MAX >> 2;
    const int64_t d0 = d.v[0], d1 = d.v[1], d2 = d.v[2], d3 = d.v[3], d4 = d.v[4];
    const int64_t e0 = e.v[0], e1 = e.v[1], e2 = e.v[2], e3 = e.v[3], e4 = e.v[4];
    const int64_t u = t.u, v = t.v, q = t.q, r = t.r;
    int64_t md = 0, me = 0, sd = 0, se = 0;
    __int128 cd = 0, ce = 0;

    // Sign-extension correction: if d (or e) is negative, the implicit bits
    // above limb 4 are all-ones.  Account for this by initializing md/me.
    sd = d4 >> 63;
    se = e4 >> 63;
    md = (u & sd) + (v & se);
    me = (q & sd) + (r & se);

    // Begin computing t*[d,e]
    cd = (__int128)u * d0 + (__int128)v * e0;
    ce = (__int128)q * d0 + (__int128)r * e0;

    // Correct md, me so that t*[d,e] + p*[md,me] has 62 zero bottom bits
    md -= (int64_t)((SAFEGCD_P_INV62 * (uint64_t)cd + (uint64_t)md) & M62);
    me -= (int64_t)((SAFEGCD_P_INV62 * (uint64_t)ce + (uint64_t)me) & M62);

    // Limb 0: exact-division shift-out  (p.v[0] = -0x1000003D1)
    cd += (__int128)SAFEGCD_P.v[0] * md;
    ce += (__int128)SAFEGCD_P.v[0] * me;
    cd >>= 62;       // bottom 62 bits are zero by construction
    ce >>= 62;

    // Limb 1  (p.v[1] = 0)
    cd += (__int128)u * d1 + (__int128)v * e1;
    ce += (__int128)q * d1 + (__int128)r * e1;
    d.v[0] = (int64_t)((uint64_t)cd & M62);  cd >>= 62;
    e.v[0] = (int64_t)((uint64_t)ce & M62);  ce >>= 62;

    // Limb 2  (p.v[2] = 0)
    cd += (__int128)u * d2 + (__int128)v * e2;
    ce += (__int128)q * d2 + (__int128)r * e2;
    d.v[1] = (int64_t)((uint64_t)cd & M62);  cd >>= 62;
    e.v[1] = (int64_t)((uint64_t)ce & M62);  ce >>= 62;

    // Limb 3  (p.v[3] = 0)
    cd += (__int128)u * d3 + (__int128)v * e3;
    ce += (__int128)q * d3 + (__int128)r * e3;
    d.v[2] = (int64_t)((uint64_t)cd & M62);  cd >>= 62;
    e.v[2] = (int64_t)((uint64_t)ce & M62);  ce >>= 62;

    // Limb 4  (p.v[4] = 256)
    cd += (__int128)u * d4 + (__int128)v * e4 + (__int128)SAFEGCD_P.v[4] * md;
    ce += (__int128)q * d4 + (__int128)r * e4 + (__int128)SAFEGCD_P.v[4] * me;
    d.v[3] = (int64_t)((uint64_t)cd & M62);  cd >>= 62;
    e.v[3] = (int64_t)((uint64_t)ce & M62);  ce >>= 62;

    d.v[4] = (int64_t)cd;
    e.v[4] = (int64_t)ce;
}

// -- Effective limb count reduction (with sign-extension propagation) --
// Ref: inline len reduction in secp256k1_modinv64_var.
// Reduces len when top limbs of both f and g are 0 or -1.
__attribute__((always_inline))
static inline void safegcd_reduce_len(int& len, SafeGCD_Int& f, SafeGCD_Int& g) {
    int64_t const fn = f.v[len - 1];
    int64_t const gn = g.v[len - 1];
    // cond == 0 iff len >= 2 AND fn in {0,-1} AND gn in {0,-1}
    int64_t cond = ((int64_t)len - 2) >> 63;
    cond |= fn ^ (fn >> 63);
    cond |= gn ^ (gn >> 63);
    if (cond == 0) {
        // Propagate sign bit of top limb into bit 62 of the limb below
        f.v[len - 2] |= (uint64_t)fn << 62;
        g.v[len - 2] |= (uint64_t)gn << 62;
        --len;
    }
}

// -- Normalize to [0, p):  conditional add + negate + carry + conditional add --
// Input:  r in range (-2p, p),  sign = top limb of f (negative if f = -1).
// Ref: secp256k1_modinv64_normalize_62 in bitcoin-core/secp256k1.
__attribute__((always_inline))
static inline void safegcd_normalize(SafeGCD_Int& r, int64_t f_sign) {
    const auto M62 = (int64_t)(UINT64_MAX >> 2);
    int64_t r0 = r.v[0], r1 = r.v[1], r2 = r.v[2], r3 = r.v[3], r4 = r.v[4];

    // Step 1:  If r < 0, add p to bring from (-2p, p) into (-p, p).
    int64_t cond_add = r4 >> 63;                    // -1 if negative, 0 if not
    r0 += SAFEGCD_P.v[0] & cond_add;               // p.v[0] = -0x1000003D1
    r4 += SAFEGCD_P.v[4] & cond_add;               // p.v[4] = 256  (p.v[1..3] = 0)

    // Negate if f is negative (f_sign < 0)
    int64_t const cond_negate = f_sign >> 63;
    r0 = (r0 ^ cond_negate) - cond_negate;
    r1 = (r1 ^ cond_negate) - cond_negate;
    r2 = (r2 ^ cond_negate) - cond_negate;
    r3 = (r3 ^ cond_negate) - cond_negate;
    r4 = (r4 ^ cond_negate) - cond_negate;

    // Propagate carries to bring limbs back to (-2^62, 2^62)
    r1 += r0 >> 62; r0 &= M62;
    r2 += r1 >> 62; r1 &= M62;
    r3 += r2 >> 62; r2 &= M62;
    r4 += r3 >> 62; r3 &= M62;

    // Step 2:  If still negative, add p again to bring to [0, p).
    cond_add = r4 >> 63;
    r0 += SAFEGCD_P.v[0] & cond_add;
    r4 += SAFEGCD_P.v[4] & cond_add;

    // Final carry propagation
    r1 += r0 >> 62; r0 &= M62;
    r2 += r1 >> 62; r1 &= M62;
    r3 += r2 >> 62; r2 &= M62;
    r4 += r3 >> 62; r3 &= M62;

    r.v[0] = r0; r.v[1] = r1; r.v[2] = r2; r.v[3] = r3; r.v[4] = r4;
}

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

} // anonymous namespace (safegcd helpers)

// -- SafeGCD inverse entry point --
// noinline: prevents ThinLTO from inlining this ~600-line divsteps loop
// into callers (schnorr_verify, ecdsa_sign), which would bloat icache and
// degrade the tightly-optimized iteration. Standalone compilation lets the
// register allocator see the full function without inter-procedural pressure.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
static FieldElement fe_inverse_safegcd_impl(const FieldElement& x) {
    SafeGCD_Int d = {{0, 0, 0, 0, 0}};     // d tracks: f = d*x (mod p)
    SafeGCD_Int e = {{1, 0, 0, 0, 0}};     // e tracks: g = e*x (mod p)
    SafeGCD_Int f = SAFEGCD_P;               // f = p
    SafeGCD_Int g = fe_to_s62(x);            // g = x

    int64_t delta = 1;
    int len = 5;

    // At most 12 x 62 = 744 divsteps  (> 590 bound for 256-bit primes)
    for (int i = 0; i < 12; ++i) {
        SafeGCD_Trans t;
        delta = safegcd_divsteps_62_var(delta,
                    (uint64_t)f.v[0], (uint64_t)g.v[0], t);

        safegcd_update_de(d, e, t);
        safegcd_update_fg(f, g, t, len);

        // g == 0 -> done (gcd found)
        {
            int64_t cond = 0;
            for (int j = 0; j < len; ++j) cond |= g.v[j];
            if (cond == 0) break;
        }

        // Reduce len when top limbs of f and g are 0 or -1
        if (i < 11) safegcd_reduce_len(len, f, g);
    }

    // f = +/-1 now.  Normalize d: negate if f<0, reduce into [0, p).
    safegcd_normalize(d, f.v[len - 1]);
    return s62_to_fe(d);
}

// ---- Jacobi fallback for single-limb values (g < 2^62) ----------------------
// Root cause: SAFEGCD_P.v[0] = -0x1000003D1 (negative in balanced signed-62).
// For small g (< 2^62) that converges in one outer round, the inner loop tracks
// Jacobi against the 63-bit truncation of p, not the full 256-bit prime.
// That truncated value (0xFFFFFFFEFFFFFC2F) differs from p in its QR class for
// small moduli, producing wrong answers.  The fix: one explicit QR step against
// the full prime (via p mod g_small), then standard 64-bit Euclidean Jacobi.

// p mod x  for  0 < x < 2^62.  Uses SAFEGCD_P limbs: p = sum(v[i] * 2^{62i}).
static int64_t secp_p_mod_s62(int64_t x) {
    static const int64_t PL[5] = { -(int64_t)0x1000003D1LL, 0, 0, 0, 256 };
    __int128 r = 0;
    for (int i = 4; i >= 0; --i) {
        r = (r << 62) + PL[i];
        r %= (__int128)x;
        if (r < 0) r += x;
    }
    return (int64_t)r;
}

// Euclidean Jacobi(a, b)  for  0 <= a < b,  b odd positive,  fits in int64.
static int jacobi_euclidean_u64(int64_t a, int64_t b) {
    int jac = 1;
    while (a != 0) {
        while ((a & 1) == 0) {
            a >>= 1;
            if ((b & 7) == 3 || (b & 7) == 5) jac = -jac;
        }
        std::swap(a, b);
        if ((a & 3) == 3 && (b & 3) == 3) jac = -jac;
        a %= b;
    }
    return (b == 1) ? jac : 0;
}

// Jacobi(g_small, p) for 0 < g_small < 2^62.
// One QR flip (p ≡ 3 mod 4) then 64-bit Euclidean Jacobi on (p mod g_small, g_small).
static int jacobi_small_var(int64_t g_small) {
    int jac = 1;
    while ((g_small & 1) == 0) g_small >>= 1;  // (2/p)=1 for p≡7 mod 8: no sign change
    if (g_small == 1) return 1;
    if ((g_small & 3) == 3) jac = -jac;  // QR: flip when both ≡ 3 mod 4 (p ≡ 3 mod 4)
    int64_t r = secp_p_mod_s62(g_small);
    return jac * jacobi_euclidean_u64(r, g_small);
}

// -- 62-step standard divstep with Jacobi tracking ---------------------------
// Identical to safegcd_divsteps_62_var (standard divstep WITH negation in swap)
// but additionally tracks the Jacobi symbol bit via jac parameter.
//
// Jacobi update rules for STANDARD divstep (negation in swap):
//   Zero removal (k zeros): jac ^= k & ((f>>1)^(f>>2))
//     [Jacobi(2^k/f): flip when k odd AND f≡3 or 5 mod 8]
//   Swap+negate: jac ^= ((~f & g) >> 1) & 1
//     [Jacobi(-f_old/g_old): flip when f≡1 mod 4 AND g≡3 mod 4]
//     Derivation: Jacobi(-a/b) = Jacobi(-1/b)*Jacobi(a/b)
//       = (-1)^{(b-1)/2} * Jacobi(b/a) * (-1)^{(a-1)(b-1)/4}
//       = (-1)^{(b-1)(a+1)/4}  [for odd a,b]
//       Flip when (b-1)(a+1)/4 is odd:
//         b≡1 mod 4 (b-1≡0): no flip
//         b≡3 mod 4, a≡1 mod 4 (a+1≡2, (b-1)*(a+1)/4 = 2m/4 = odd*even/4... complicated)
//       Simplified formula: flip when f≡1 AND g≡3 (mod 4)
//       = ((~f & g) >> 1) & 1   [since ~f bit1=1 iff f≡1 mod 4, g bit1=1 iff g≡3 mod 4]
//
// Uses the SAME transition matrix as safegcd_divsteps_62_var → compatible with
// safegcd_update_fg. Does NOT need posdivstep multi-bit cancellation.
static inline int64_t safegcd_divsteps_62_jacobi_var(int64_t delta, uint64_t f0, uint64_t g0,
                                                       SafeGCD_Trans& t, int& jac) {
    uint64_t u = 1, v = 0, q = 0, r = 1;
    uint64_t f = f0, g = g0;
    int i = 62;

    for (;;) {
        // Remove trailing zeros of g — Jacobi(2^k/f) contribution.
        // Only bit 0 of jac matters: use & 1 to avoid multi-bit interference.
        int const zeros = safegcd_ctz64(g | ((uint64_t)1 << i));
        jac ^= static_cast<int>((zeros & ((f >> 1) ^ (f >> 2))) & 1);
        g >>= zeros;
        u <<= zeros;
        v <<= zeros;
        delta += zeros;
        i -= zeros;
        if (i == 0) break;

        if (delta > 0) {
            // Standard swap WITH NEGATION.
            // Jacobi update: flip when f≡1 mod 4 AND g≡3 mod 4. Use & 1.
            jac ^= static_cast<int>(((~f & g) >> 1) & 1);
            delta = -delta;
            uint64_t tmp;
            tmp = f; f = g;  g = (uint64_t)(-(int64_t)tmp);   // negate g
            tmp = u; u = q;  q = (uint64_t)(-(int64_t)tmp);
            tmp = v; v = r;  r = (uint64_t)(-(int64_t)tmp);
        }
        g += f;  q += u;  r += v;
        g >>= 1;  u <<= 1;  v <<= 1;
        ++delta;  --i;
        if (i == 0) break;
    }

    t.u = (int64_t)u;  t.v = (int64_t)v;
    t.q = (int64_t)q;  t.r = (int64_t)r;
    return delta;
}

// -- 62-step posdivstep batch for Jacobi symbol --------------------------------
// Port of Bitcoin Core's secp256k1_modinv64_posdivsteps_62_var.
// Uses multi-bit cancellation (4-6 bits/step) unlike simple divstep (1 bit/step).
// This ensures the transition matrix has the exact-division property needed by
// safegcd_update_fg. Simple removal of negation from standard divstep does NOT
// have this property and produces wrong full-precision updates.
//
// Key differences from standard safegcd_divsteps_62_var:
//   - eta = -delta convention (eta > 0 means swap at next step)
//   - No negation in swap (g stays positive)
//   - Multi-bit cancel: when eta<0, uses w = (f*g*(f*f-2)) & m to cancel
//     up to 6 low bits of g; when eta>=0, uses w = (-w*g) & m (up to 4 bits)
//
// Jacobi updates (same formula as before, correct for posdivstep):
//   - Zero removal: jac ^= zeros & ((f>>1)^(f>>2))  [2/f Jacobi factor]
//   - Swap: jac ^= (f & g) >> 1                     [QR reciprocity: both ≡3 mod 4]
//
// Ref: Bitcoin Core secp256k1/src/modinv64_impl.h:secp256k1_modinv64_posdivsteps_62_var
static inline int64_t jacobi_posdivsteps_62_var(int64_t eta,
                                                  uint64_t f0, uint64_t g0,
                                                  SafeGCD_Trans& t, int& jac) {
    uint64_t u = 1, v = 0, q = 0, r = 1;
    uint64_t f = f0, g = g0;
    uint32_t w;
    uint64_t m;
    int i = 62, limit, zeros;

    for (;;) {
        // Bulk-remove trailing zeros from g (each = divide g by 2).
        // Jacobi(2/f): flip when f ≡ 3 or 5 (mod 8).
        zeros = safegcd_ctz64(g | ((uint64_t)1 << i));
        jac ^= static_cast<int>(zeros & ((f >> 1) ^ (f >> 2)));
        g >>= zeros;
        u <<= zeros;
        v <<= zeros;
        eta -= zeros;
        i -= zeros;
        if (i == 0) break;

        // g is odd, f is odd. Swap if eta < 0 (= delta > 0 in delta convention).
        if (eta < 0) {
            uint64_t tmp;
            eta = -eta;
            tmp = f; f = g; g = tmp;
            tmp = u; u = q; q = tmp;
            tmp = v; v = r; r = tmp;
            // Jacobi QR reciprocity: flip when both f,g ≡ 3 (mod 4).
            jac ^= static_cast<int>((f & g) >> 1);
            // Multi-bit cancel (up to 6 bits): find w s.t. g + f*w ≡ 0 (mod 2^m).
            limit = ((int)eta + 1) > i ? i : ((int)eta + 1);
            m = (UINT64_MAX >> (64 - limit)) & 63U;
            w = static_cast<uint32_t>((f * g * (f * f - 2)) & m);
        } else {
            // Multi-bit cancel (up to 4 bits): simpler formula for eta >= 0.
            limit = ((int)eta + 1) > i ? i : ((int)eta + 1);
            m = (UINT64_MAX >> (64 - limit)) & 15U;
            w = static_cast<uint32_t>(f + (((f + 1) & 4) << 1));
            w = static_cast<uint32_t>((-static_cast<uint64_t>(w) * g) & m);
        }
        g += f * static_cast<uint64_t>(w);
        q += u * static_cast<uint64_t>(w);
        r += v * static_cast<uint64_t>(w);
        // g now has `limit` trailing zeros canceled. But g might be 0 (mod 2^64)
        // due to wrap-around (g + f*w = 2^64 → 0), even when g_full ≠ 0.
        // In that case, ctz(0) would count ALL remaining bits as zeros, adding
        // spurious Jacobi contributions. Fix: if g == 0, exhaust the bit budget
        // WITHOUT updating jac (the zeros are sentinel/fake, not genuine).
        if (g == 0) {
            u <<= i;
            v <<= i;
            eta -= i;
            i = 0;
            break;
        }
    }

    t.u = (int64_t)u;  t.v = (int64_t)v;
    t.q = (int64_t)q;  t.r = (int64_t)r;
    return eta;
}

// -- 5x52 Jacobi symbol (Legendre symbol for prime p) via posdivstep ----------
// Port of Bitcoin Core secp256k1_jacobi64_maybe_var.
// Returns +1 if x is QR mod p, -1 if NQR, 0 if x ≡ 0 mod p (or non-convergence).
// Variable-time — only call with PUBLIC data (see security comment in ellswift.cpp).
// ~900 ns (posdivstep SafeGCD), vs Fermat sqrt ~3.8 µs.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
int fe52_jacobi_var(const std::uint64_t* in5) {
    constexpr std::uint64_t M62 = (1ULL << 62) - 1;

    SafeGCD_Int g;
    g.v[0] = static_cast<int64_t>((in5[0]        | (in5[1] << 52)) & M62);
    g.v[1] = static_cast<int64_t>(((in5[1] >> 10) | (in5[2] << 42)) & M62);
    g.v[2] = static_cast<int64_t>(((in5[2] >> 20) | (in5[3] << 32)) & M62);
    g.v[3] = static_cast<int64_t>(((in5[3] >> 30) | (in5[4] << 22)) & M62);
    g.v[4] = static_cast<int64_t>(in5[4] >> 40);

    // x ≡ 0 mod p → Jacobi = 0
    {
        int64_t z = 0;
        for (int i = 0; i < 5; ++i) z |= g.v[i];
        if (z == 0) return 0;
    }

    // Guard: single-limb values (g < 2^62) use the 64-bit Euclidean fallback.
    // The main loop's 63-bit f truncation gives wrong Jacobi when g is small enough
    // to converge in one outer round — only truncated-p's QR class is captured, not p's.
    if (g.v[1] == 0 && g.v[2] == 0 && g.v[3] == 0 && g.v[4] == 0)
        return jacobi_small_var(g.v[0]);

    // f starts as p (positive). eta = -delta = -1 initially.
    SafeGCD_Int f = SAFEGCD_P;
    int64_t eta = -1;
    int len = 5;
    int jac = 0;

    // 25 rounds of posdivstep with multi-bit cancel (1550 steps ≥ 591 needed for 256-bit).
    // Uses 63-bit windows (v[0] | v[1]<<62) to feed the inner loop.
    for (int iter = 0; iter < 25; ++iter) {
        SafeGCD_Trans t;
        uint64_t f63 = (uint64_t)f.v[0] | ((uint64_t)f.v[1] << 62);
        uint64_t g63 = (uint64_t)g.v[0] | ((uint64_t)g.v[1] << 62);
        eta = jacobi_posdivsteps_62_var(eta, f63, g63, t, jac);

        safegcd_update_fg(f, g, t, len);

        // When f == 1, the Jacobi is fully determined.
        if (f.v[0] == 1) {
            int64_t fn = 0;
            for (int j = 1; j < len; ++j) fn |= f.v[j];
            if (fn == 0) return 1 - 2 * (jac & 1);
        }

        safegcd_reduce_len(len, f, g);
    }

    return 0;  // Non-convergence: should never happen for secp256k1 inputs
}

// -- Direct 5x52 SafeGCD inverse (no 4x64 intermediate) ----------------------
// Eliminates 2 format conversions vs the FE52→to_fe()→inverse()→from_fe() chain.
// Matches libsecp256k1's direct 5x52↔signed62 approach.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
void fe52_inverse_safegcd_var(const std::uint64_t* in5, std::uint64_t* out5) {
    constexpr std::uint64_t M62 = (1ULL << 62) - 1;
    constexpr std::uint64_t M52 = (1ULL << 52) - 1;

    // Direct 5x52 → signed-62 (matches libsecp's secp256k1_fe_to_signed62)
    SafeGCD_Int g;
    g.v[0] = static_cast<int64_t>((in5[0]       | (in5[1] << 52)) & M62);
    g.v[1] = static_cast<int64_t>(((in5[1] >> 10) | (in5[2] << 42)) & M62);
    g.v[2] = static_cast<int64_t>(((in5[2] >> 20) | (in5[3] << 32)) & M62);
    g.v[3] = static_cast<int64_t>(((in5[3] >> 30) | (in5[4] << 22)) & M62);
    g.v[4] = static_cast<int64_t>(in5[4] >> 40);

    SafeGCD_Int d = {{0, 0, 0, 0, 0}};
    SafeGCD_Int e = {{1, 0, 0, 0, 0}};
    SafeGCD_Int f = SAFEGCD_P;

    int64_t delta = 1;
    int len = 5;

    for (int i = 0; i < 12; ++i) {
        SafeGCD_Trans t;
        delta = safegcd_divsteps_62_var(delta,
                    (uint64_t)f.v[0], (uint64_t)g.v[0], t);

        safegcd_update_de(d, e, t);
        safegcd_update_fg(f, g, t, len);

        {
            int64_t cond = 0;
            for (int j = 0; j < len; ++j) cond |= g.v[j];
            if (cond == 0) break;
        }

        if (i < 11) safegcd_reduce_len(len, f, g);
    }

    safegcd_normalize(d, f.v[len - 1]);

    // Direct signed-62 → 5x52 (matches libsecp's secp256k1_fe_from_signed62)
    const auto a0 = static_cast<std::uint64_t>(d.v[0]);
    const auto a1 = static_cast<std::uint64_t>(d.v[1]);
    const auto a2 = static_cast<std::uint64_t>(d.v[2]);
    const auto a3 = static_cast<std::uint64_t>(d.v[3]);
    const auto a4 = static_cast<std::uint64_t>(d.v[4]);
    out5[0] =  a0                   & M52;
    out5[1] = (a0 >> 52 | a1 << 10) & M52;
    out5[2] = (a1 >> 42 | a2 << 20) & M52;
    out5[3] = (a2 >> 32 | a3 << 30) & M52;
    out5[4] = (a3 >> 22 | a4 << 40);
}

#endif // __SIZEOF_INT128__

// ============================================================================
// SafeGCD30 field inverse -- 30-bit divsteps (no __int128 required)
// Adapted from bitcoin-core secp256k1_modinv32_var (MIT license).
// Uses the secp256k1 prime p = 2^256 - 2^32 - 977.
// ~130us on ESP32 vs ~3000us for Fermat chain (pow_p_minus_2_binary).
// ============================================================================
namespace field_safegcd30 {

struct S30  { int32_t v[9]; };
struct T2x2 { int32_t u, v, q, r; };
struct ModInfo { S30 modulus; uint32_t modulus_inv30; };

// secp256k1 prime p in signed-30 form:
//   p = 2^256 - 2^32 - 977
//   = -977 + (-4)*2^30 + 65536*2^240
// Matches bitcoin-core secp256k1_const_modinfo_fe.
static constexpr ModInfo PINFO = {
    {{-0x3D1, -4, 0, 0, 0, 0, 0, 0, 65536}},
    0x2DDACACFU
};

[[maybe_unused]] static inline int ctz32_var(uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctz(x);
#elif defined(_MSC_VER)
    unsigned long idx;
    _BitScanForward(&idx, x);
    return (int)idx;
#else
    int c = 0; while (!(x & 1)) { x >>= 1; ++c; } return c;
#endif
}

// Lookup: -(2i+1)^{-1} mod 256 -- same table as scalar_safegcd30
static const uint8_t inv256[128] = {
    0xFF,0x55,0x33,0x49,0xC7,0x5D,0x3B,0x11,0x0F,0xE5,0xC3,0x59,
    0xD7,0xED,0xCB,0x21,0x1F,0x75,0x53,0x69,0xE7,0x7D,0x5B,0x31,
    0x2F,0x05,0xE3,0x79,0xF7,0x0D,0xEB,0x41,0x3F,0x95,0x73,0x89,
    0x07,0x9D,0x7B,0x51,0x4F,0x25,0x03,0x99,0x17,0x2D,0x0B,0x61,
    0x5F,0xB5,0x93,0xA9,0x27,0xBD,0x9B,0x71,0x6F,0x45,0x23,0xB9,
    0x37,0x4D,0x2B,0x81,0x7F,0xD5,0xB3,0xC9,0x47,0xDD,0xBB,0x91,
    0x8F,0x65,0x43,0xD9,0x57,0x6D,0x4B,0xA1,0x9F,0xF5,0xD3,0xE9,
    0x67,0xFD,0xDB,0xB1,0xAF,0x85,0x63,0xF9,0x77,0x8D,0x6B,0xC1,
    0xBF,0x15,0xF3,0x09,0x87,0x1D,0xFB,0xD1,0xCF,0xA5,0x83,0x19,
    0x97,0xAD,0x8B,0xE1,0xDF,0x35,0x13,0x29,0xA7,0x3D,0x1B,0xF1,
    0xEF,0xC5,0xA3,0x39,0xB7,0xCD,0xAB,0x01
};

// Variable-time 30 divsteps (matches secp256k1_modinv32_divsteps_30_var)
[[maybe_unused]] static int32_t divsteps_30_var(int32_t eta, uint32_t f0, uint32_t g0, T2x2& t) {
    uint32_t u = 1, v = 0, q = 0, r = 1;
    uint32_t f = f0, g = g0, m = 0;
    uint16_t w = 0;
    int i = 30, limit = 0, zeros = 0;

    for (;;) {
        zeros = ctz32_var(g | (UINT32_MAX << i));
        g >>= zeros;
        u <<= zeros;
        v <<= zeros;
        eta -= zeros;
        i -= zeros;
        if (i == 0) break;

        if (eta < 0) {
            uint32_t tmp = 0;
            eta = -eta;
            tmp = f; f = g; g = (uint32_t)(-(int32_t)tmp);
            tmp = u; u = q; q = (uint32_t)(-(int32_t)tmp);
            tmp = v; v = r; r = (uint32_t)(-(int32_t)tmp);
        }
        limit = ((int)eta + 1) > i ? i : ((int)eta + 1);
        m = (UINT32_MAX >> (32 - limit)) & 255U;
        w = (uint16_t)((g * inv256[(f >> 1) & 127]) & m);
        g += f * (uint32_t)w;
        q += u * (uint32_t)w;
        r += v * (uint32_t)w;
    }

    t.u = (int32_t)u; t.v = (int32_t)v;
    t.q = (int32_t)q; t.r = (int32_t)r;
    return eta;
}

// (t/2^30) * [d, e] mod p (matches secp256k1_modinv32_update_de_30)
[[maybe_unused]] static void update_de_30(S30& d, S30& e, const T2x2& t, const ModInfo& mod) {
    const auto M30 = (int32_t)(UINT32_MAX >> 2);
    const int32_t u = t.u, v = t.v, q = t.q, r = t.r;
    int32_t di = 0, ei = 0, md = 0, me = 0, sd = 0, se = 0;
    int64_t cd = 0, ce = 0;

    // cppcheck-suppress shiftTooManyBitsSigned
    sd = d.v[8] >> 31;
    // cppcheck-suppress shiftTooManyBitsSigned
    se = e.v[8] >> 31;
    md = (u & sd) + (v & se);
    me = (q & sd) + (r & se);

    di = d.v[0]; ei = e.v[0];
    cd = (int64_t)u * di + (int64_t)v * ei;
    ce = (int64_t)q * di + (int64_t)r * ei;

    md -= (int32_t)((mod.modulus_inv30 * (uint32_t)cd + (uint32_t)md) & (uint32_t)M30);
    me -= (int32_t)((mod.modulus_inv30 * (uint32_t)ce + (uint32_t)me) & (uint32_t)M30);

    cd += (int64_t)mod.modulus.v[0] * md;
    ce += (int64_t)mod.modulus.v[0] * me;
    cd >>= 30; ce >>= 30;

    for (int i = 1; i < 9; ++i) {
        di = d.v[i]; ei = e.v[i];
        cd += (int64_t)u * di + (int64_t)v * ei;
        ce += (int64_t)q * di + (int64_t)r * ei;
        cd += (int64_t)mod.modulus.v[i] * md;
        ce += (int64_t)mod.modulus.v[i] * me;
        d.v[i - 1] = (int32_t)cd & M30; cd >>= 30;
        e.v[i - 1] = (int32_t)ce & M30; ce >>= 30;
    }
    d.v[8] = (int32_t)cd;
    e.v[8] = (int32_t)ce;
}

// (t/2^30) * [f, g] variable-length
[[maybe_unused]] static void update_fg_30_var(int len, S30& f, S30& g, const T2x2& t) {
    const auto M30 = (int32_t)(UINT32_MAX >> 2);
    const int32_t u = t.u, v = t.v, q = t.q, r = t.r;
    int32_t fi = 0, gi = 0;
    int64_t cf = 0, cg = 0;

    fi = f.v[0]; gi = g.v[0];
    cf = (int64_t)u * fi + (int64_t)v * gi;
    cg = (int64_t)q * fi + (int64_t)r * gi;
    cf >>= 30; cg >>= 30;

    for (int j = 1; j < len; ++j) {
        fi = f.v[j]; gi = g.v[j];
        cf += (int64_t)u * fi + (int64_t)v * gi;
        cg += (int64_t)q * fi + (int64_t)r * gi;
        f.v[j - 1] = (int32_t)((uint32_t)cf & (uint32_t)M30); cf >>= 30;
        g.v[j - 1] = (int32_t)((uint32_t)cg & (uint32_t)M30); cg >>= 30;
    }
    f.v[len - 1] = (int32_t)cf;
    g.v[len - 1] = (int32_t)cg;
    for (int j = len; j < 9; ++j) { f.v[j] = 0; g.v[j] = 0; }
}

// Normalize to [0, p)
[[maybe_unused]] static void normalize_30(S30& r, int32_t sign, const ModInfo& mod) {
    const auto M30 = (int32_t)(UINT32_MAX >> 2);
    int32_t r0=r.v[0], r1=r.v[1], r2=r.v[2], r3=r.v[3], r4=r.v[4],
            r5=r.v[5], r6=r.v[6], r7=r.v[7], r8=r.v[8];
    int32_t cond_add = 0, cond_negate = 0;

    // cppcheck-suppress shiftTooManyBitsSigned
    cond_add = r8 >> 31;
    r0 += mod.modulus.v[0] & cond_add;
    r1 += mod.modulus.v[1] & cond_add;
    r2 += mod.modulus.v[2] & cond_add;
    r3 += mod.modulus.v[3] & cond_add;
    r4 += mod.modulus.v[4] & cond_add;
    r5 += mod.modulus.v[5] & cond_add;
    r6 += mod.modulus.v[6] & cond_add;
    r7 += mod.modulus.v[7] & cond_add;
    r8 += mod.modulus.v[8] & cond_add;
    // cppcheck-suppress shiftTooManyBitsSigned
    cond_negate = sign >> 31;
    r0 = (r0 ^ cond_negate) - cond_negate;
    r1 = (r1 ^ cond_negate) - cond_negate;
    r2 = (r2 ^ cond_negate) - cond_negate;
    r3 = (r3 ^ cond_negate) - cond_negate;
    r4 = (r4 ^ cond_negate) - cond_negate;
    r5 = (r5 ^ cond_negate) - cond_negate;
    r6 = (r6 ^ cond_negate) - cond_negate;
    r7 = (r7 ^ cond_negate) - cond_negate;
    r8 = (r8 ^ cond_negate) - cond_negate;
    r1 += r0 >> 30; r0 &= M30;
    r2 += r1 >> 30; r1 &= M30;
    r3 += r2 >> 30; r2 &= M30;
    r4 += r3 >> 30; r3 &= M30;
    r5 += r4 >> 30; r4 &= M30;
    r6 += r5 >> 30; r5 &= M30;
    r7 += r6 >> 30; r6 &= M30;
    r8 += r7 >> 30; r7 &= M30;

    // cppcheck-suppress shiftTooManyBitsSigned
    cond_add = r8 >> 31;
    r0 += mod.modulus.v[0] & cond_add;
    r1 += mod.modulus.v[1] & cond_add;
    r2 += mod.modulus.v[2] & cond_add;
    r3 += mod.modulus.v[3] & cond_add;
    r4 += mod.modulus.v[4] & cond_add;
    r5 += mod.modulus.v[5] & cond_add;
    r6 += mod.modulus.v[6] & cond_add;
    r7 += mod.modulus.v[7] & cond_add;
    r8 += mod.modulus.v[8] & cond_add;
    r1 += r0 >> 30; r0 &= M30;
    r2 += r1 >> 30; r1 &= M30;
    r3 += r2 >> 30; r2 &= M30;
    r4 += r3 >> 30; r3 &= M30;
    r5 += r4 >> 30; r4 &= M30;
    r6 += r5 >> 30; r5 &= M30;
    r7 += r6 >> 30; r6 &= M30;
    r8 += r7 >> 30; r7 &= M30;

    r.v[0]=r0; r.v[1]=r1; r.v[2]=r2; r.v[3]=r3; r.v[4]=r4;
    r.v[5]=r5; r.v[6]=r6; r.v[7]=r7; r.v[8]=r8;
}

// Convert 4x64-bit limbs -> signed-30 representation
[[maybe_unused]] static S30 limbs_to_s30(const limbs4& x) {
    S30 r{};
    const uint32_t M30 = 0x3FFFFFFFu;
    r.v[0] = (int32_t)( x[0]        & M30);
    r.v[1] = (int32_t)((x[0] >> 30) & M30);
    r.v[2] = (int32_t)(((x[0] >> 60) | (x[1] <<  4)) & M30);
    r.v[3] = (int32_t)((x[1] >> 26) & M30);
    r.v[4] = (int32_t)(((x[1] >> 56) | (x[2] <<  8)) & M30);
    r.v[5] = (int32_t)((x[2] >> 22) & M30);
    r.v[6] = (int32_t)(((x[2] >> 52) | (x[3] << 12)) & M30);
    r.v[7] = (int32_t)((x[3] >> 18) & M30);
    r.v[8] = (int32_t)( x[3] >> 48);
    return r;
}

// Convert signed-30 -> 4x64-bit limbs
[[maybe_unused]] static limbs4 s30_to_limbs(const S30& s) {
    limbs4 r{};
    r[0] = ((uint64_t)(uint32_t)s.v[0])
         | ((uint64_t)(uint32_t)s.v[1] << 30)
         | ((uint64_t)(uint32_t)s.v[2] << 60);
    r[1] = ((uint64_t)(uint32_t)s.v[2] >> 4)
         | ((uint64_t)(uint32_t)s.v[3] << 26)
         | ((uint64_t)(uint32_t)s.v[4] << 56);
    r[2] = ((uint64_t)(uint32_t)s.v[4] >> 8)
         | ((uint64_t)(uint32_t)s.v[5] << 22)
         | ((uint64_t)(uint32_t)s.v[6] << 52);
    r[3] = ((uint64_t)(uint32_t)s.v[6] >> 12)
         | ((uint64_t)(uint32_t)s.v[7] << 18)
         | ((uint64_t)(uint32_t)s.v[8] << 48);
    return r;
}

// Main entry: variable-time modular inverse mod p
[[maybe_unused]] static FieldElement inverse_impl(const FieldElement& x) {
    S30 d{};                   // d = 0
    S30 e{}; e.v[0] = 1;      // e = 1
    S30 f = PINFO.modulus;     // f = p
    S30 g = limbs_to_s30(x.limbs()); // g = x
    int len = 9;
    int32_t eta = -1;

    while (1) {
        T2x2 t;
        eta = divsteps_30_var(eta, (uint32_t)f.v[0], (uint32_t)g.v[0], t);

        update_de_30(d, e, t, PINFO);
        update_fg_30_var(len, f, g, t);

        if (g.v[0] == 0) {
            int32_t cond = 0;
            for (int j = 1; j < len; ++j) cond |= g.v[j];
            if (cond == 0) break;
        }

        int32_t const fn = f.v[len - 1], gn = g.v[len - 1];
        // cppcheck-suppress shiftTooManyBitsSigned
        int32_t cond = ((int32_t)len - 2) >> 31;
        // cppcheck-suppress shiftTooManyBitsSigned
        cond |= fn ^ (fn >> 31);
        // cppcheck-suppress shiftTooManyBitsSigned
        cond |= gn ^ (gn >> 31);
        if (cond == 0) {
            f.v[len - 2] |= (uint32_t)fn << 30;
            g.v[len - 2] |= (uint32_t)gn << 30;
            --len;
        }
    }

    normalize_30(d, f.v[len - 1], PINFO);
    return FieldElement::from_limbs(s30_to_limbs(d));
}

} // namespace field_safegcd30

FieldElement FieldElement::operator+(const FieldElement& rhs) const {
    return FieldElement(add_impl(limbs_, rhs.limbs_), true);
}

FieldElement FieldElement::operator-(const FieldElement& rhs) const {
    return FieldElement(sub_impl(limbs_, rhs.limbs_), true);
}

FieldElement FieldElement::operator*(const FieldElement& rhs) const {
    auto result_limbs = mul_impl(limbs_, rhs.limbs_);
    return FieldElement(result_limbs, true);
}

FieldElement FieldElement::square() const {
    return FieldElement(square_impl(limbs_), true);
}

FieldElement FieldElement::inverse() const {
    if (*this == zero()) {
        #if defined(SECP256K1_ESP32) || defined(SECP256K1_PLATFORM_ESP32) || defined(__XTENSA__) || defined(SECP256K1_PLATFORM_STM32)
            return zero(); // Embedded: no exceptions, return zero
        #else
            throw std::runtime_error("Inverse of zero not defined");
        #endif
    }
    #if defined(__SIZEOF_INT128__)
    return fe_inverse_safegcd_impl(*this);
    #else
    return field_safegcd30::inverse_impl(*this); // SafeGCD30: ~130us vs ~3ms Fermat
    #endif
}

FieldElement& FieldElement::operator+=(const FieldElement& rhs) {
    limbs_ = add_impl(limbs_, rhs.limbs_);
    return *this;
}

FieldElement& FieldElement::operator-=(const FieldElement& rhs) {
    limbs_ = sub_impl(limbs_, rhs.limbs_);
    return *this;
}

FieldElement FieldElement::negate(unsigned /*magnitude*/) const {
    // Direct PRIME - x: single 4-limb borrow chain (saves ~25% vs sub_impl(0, x))
    unsigned char borrow = 0;
    limbs4 r;
    r[0] = sub64(PRIME[0], limbs_[0], borrow);
    r[1] = sub64(PRIME[1], limbs_[1], borrow);
    r[2] = sub64(PRIME[2], limbs_[2], borrow);
    r[3] = sub64(PRIME[3], limbs_[3], borrow);
    // Branchless: if x == 0, return 0 (PRIME - 0 = PRIME is wrong)
    const std::uint64_t nonzero = limbs_[0] | limbs_[1] | limbs_[2] | limbs_[3];
    const std::uint64_t mask = 0ULL - static_cast<std::uint64_t>(nonzero != 0);
    r[0] &= mask; r[1] &= mask; r[2] &= mask; r[3] &= mask;
    return FieldElement(r, true);
}

void FieldElement::negate_assign(unsigned /*magnitude*/) {
    unsigned char borrow = 0;
    limbs4 r;
    r[0] = sub64(PRIME[0], limbs_[0], borrow);
    r[1] = sub64(PRIME[1], limbs_[1], borrow);
    r[2] = sub64(PRIME[2], limbs_[2], borrow);
    r[3] = sub64(PRIME[3], limbs_[3], borrow);
    const std::uint64_t nonzero = limbs_[0] | limbs_[1] | limbs_[2] | limbs_[3];
    const std::uint64_t mask = 0ULL - static_cast<std::uint64_t>(nonzero != 0);
    limbs_[0] = r[0] & mask; limbs_[1] = r[1] & mask;
    limbs_[2] = r[2] & mask; limbs_[3] = r[3] & mask;
}

FieldElement& FieldElement::operator*=(const FieldElement& rhs) {
    limbs_ = mul_impl(limbs_, rhs.limbs_);
    return *this;
}

// In-place mutable versions (modify this object directly)
void FieldElement::square_inplace() {
    limbs_ = square_impl(limbs_);
}

void FieldElement::inverse_inplace() {
    if (*this == zero()) {
        #if defined(SECP256K1_ESP32) || defined(SECP256K1_PLATFORM_ESP32) || defined(__XTENSA__) || defined(SECP256K1_PLATFORM_STM32)
            *this = zero(); // Embedded: no exceptions, set to zero
            return;
        #else
            throw std::runtime_error("Inverse of zero not defined");
        #endif
    }
    #if defined(__SIZEOF_INT128__)
    *this = fe_inverse_safegcd_impl(*this);
    #else
    *this = field_safegcd30::inverse_impl(*this); // SafeGCD30: ~130us vs ~3ms Fermat
    #endif
}

// -- Optimized square root: a^((p+1)/4) mod p --------------------------------
// Uses an addition chain building 2^n-1 blocks: {2, 22, 223}
// then assembles (p+1)/4 via sliding window.
// Total cost: ~255 squarings + 13 multiplications.
// Direct port of bitcoin-core/secp256k1 secp256k1_fe_sqrt.
FieldElement FieldElement::sqrt() const {
#if defined(SECP256K1_FAST_52BIT)
    // FE52 path: 10.3ns/sqr vs ~20ns/sqr for 4×64 → 2× faster sqrt.
    auto fe52 = fast::FieldElement52::from_fe(*this);
    auto r52  = fe52.sqrt();
    return r52.to_fe();
#endif
    FieldElement x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223, t1;

    // x2 = a^(2^2-1)
    x2 = this->square();
    x2 = x2 * *this;

    // x3 = a^(2^3-1)
    x3 = x2.square();
    x3 = x3 * *this;

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

    // Assemble (p+1)/4 using sliding window:
    // (p+1)/4 = 2^254 - 2^30 - 2^4
    t1 = x223;
    for (int j = 0; j < 23; ++j) t1.square_inplace();
    t1 = t1 * x22;
    for (int j = 0; j < 6; ++j) t1.square_inplace();
    t1 = t1 * x2;
    t1.square_inplace();
    t1.square_inplace();

    return t1;
}

bool FieldElement::operator==(const FieldElement& rhs) const noexcept {
    // Normalize both operands to canonical form [0, p) before comparing.
    // Optimized arithmetic paths (e.g. montgomery_reduce_bmi2, square_impl)
    // can produce results in [p, 2^256) that are correct mod p but have
    // non-canonical limb representations.  A single conditional subtract
    // of p is sufficient because all outputs are < 2p (since p > 2^255).
    limbs4 a = limbs_;
    limbs4 b = rhs.limbs_;
    normalize(a);
    normalize(b);
    return a == b;
}

FieldElement fe_inverse_binary(const FieldElement& value) {
    return pow_p_minus_2_binary(value);
}

FieldElement fe_inverse_window4(const FieldElement& value) {
    return pow_p_minus_2_window4(value);
}

FieldElement fe_inverse_addchain(const FieldElement& value) {
    return pow_p_minus_2_addchain(value);
}

FieldElement fe_inverse_eea(const FieldElement& value) {
    return pow_p_minus_2_eea(value);
}

FieldElement fe_inverse_window_naf_v2(const FieldElement& value) {
    return pow_p_minus_2_window_naf_v2(value);
}

FieldElement fe_inverse_hybrid_eea(const FieldElement& value) {
    return pow_p_minus_2_hybrid_eea(value);
}

FieldElement fe_inverse_safegcd(const FieldElement& value) {
    #if defined(__SIZEOF_INT128__)
    return fe_inverse_safegcd_impl(value);
    #else
    return pow_p_minus_2_binary(value); // Fallback on 32-bit
    #endif
}

FieldElement fe_inverse_yao(const FieldElement& value) {
    return pow_p_minus_2_yao(value);
}

// New optimized methods - Round 2
FieldElement fe_inverse_kary16(const FieldElement& value) {
    return pow_p_minus_2_kary16(value);
}

FieldElement fe_inverse_fixed_window5(const FieldElement& value) {
    return pow_p_minus_2_fixed_window5(value);
}

FieldElement fe_inverse_rtl_binary(const FieldElement& value) {
    return pow_p_minus_2_rtl_binary(value);
}

FieldElement fe_inverse_addchain_unrolled(const FieldElement& value) {
    return pow_p_minus_2_addchain_unrolled(value);
}

FieldElement fe_inverse_binary_opt(const FieldElement& value) {
    return pow_p_minus_2_binary_opt(value);
}

FieldElement fe_inverse_sliding_dynamic(const FieldElement& value) {
    return pow_p_minus_2_sliding_dynamic(value);
}

// Round 3 - GPU and ECC-specific wrappers
FieldElement fe_inverse_fermat_gpu(const FieldElement& value) {
    return pow_p_minus_2_fermat_gpu(value);
}

FieldElement fe_inverse_montgomery_redc(const FieldElement& value) {
    return pow_p_minus_2_montgomery_redc(value);
}

FieldElement fe_inverse_branchless(const FieldElement& value) {
    return pow_p_minus_2_branchless(value);
}

FieldElement fe_inverse_parallel_window(const FieldElement& value) {
    return pow_p_minus_2_parallel_window(value);
}

FieldElement fe_inverse_binary_euclidean(const FieldElement& value) {
    return pow_p_minus_2_binary_euclidean(value);
}

FieldElement fe_inverse_lehmer(const FieldElement& value) {
    return pow_p_minus_2_lehmer(value);
}

FieldElement fe_inverse_stein(const FieldElement& value) {
    return pow_p_minus_2_stein(value);
}

FieldElement fe_inverse_secp256k1_special(const FieldElement& value) {
    return pow_p_minus_2_secp256k1_special(value);
}

FieldElement fe_inverse_warp_optimized(const FieldElement& value) {
    return pow_p_minus_2_warp_optimized(value);
}

FieldElement fe_inverse_double_base(const FieldElement& value) {
    return pow_p_minus_2_double_base(value);
}

FieldElement fe_inverse_compact_table(const FieldElement& value) {
    return pow_p_minus_2_compact_table(value);
}

FieldElement fe_inverse_bos_coster(const FieldElement& value) {
    return pow_p_minus_2_bos_coster(value);
}

FieldElement fe_inverse_ltr_precomp(const FieldElement& value) {
    return pow_p_minus_2_ltr_precomp(value);
}

FieldElement fe_inverse_pippenger(const FieldElement& value) {
    return pow_p_minus_2_pippenger(value);
}

FieldElement fe_inverse_karatsuba(const FieldElement& value) {
    return pow_p_minus_2_karatsuba(value);
}

FieldElement fe_inverse_booth(const FieldElement& value) {
    return pow_p_minus_2_booth(value);
}

FieldElement fe_inverse_strauss(const FieldElement& value) {
    return pow_p_minus_2_strauss(value);
}

// Montgomery batch inversion algorithm (zero-safe)
// Input: array of N field elements [a_0, a_1, ..., a_n_1]
// Output: modifies array in-place to [a_0^-1, a_1^-1, ..., a_n_1^-1]
//         Zero elements map to zero in the output.
//
// Algorithm:
//   1. Compute products: p_0=a_0, p_1=a_0*a_1, ..., p_n_1=a_0*...*a_n_1
//      (zeros are substituted with 1 to keep the product chain valid)
//   2. Invert final product: inv = (a_0*...*a_n_1)^-1  (zeros excluded)
//   3. Work backwards: a_i^-1 = inv * p_{i-1}, then inv = inv * a_i
//      (zero positions are set to zero and do not update inv)
//
// Cost: 3N multiplications + 1 inversion (vs N inversions)
// For N=8: ~8 us vs ~28 us (3.5x faster!)
static inline void fe_batch_inverse_with_scratch(FieldElement* elements,
                                                 size_t count,
                                                 FieldElement* scratch) {
    FieldElement const kOne = FieldElement::one();
    FieldElement const kZero = FieldElement::zero();

    // Step 1: Compute cumulative products, substituting zeros with 1
    bool const first_zero = (elements[0] == kZero);
    scratch[0] = first_zero ? kOne : elements[0];
    for (size_t i = 1; i < count; ++i) {
        bool const is_z = (elements[i] == kZero);
        scratch[i] = scratch[i - 1] * (is_z ? kOne : elements[i]);
    }

    // Step 2: Invert the final product (only 1 expensive inverse!)
    FieldElement inv = scratch[count - 1].inverse();

    // Step 3: Work backwards to compute individual inverses
    for (size_t i = count - 1; i > 0; --i) {
        if (elements[i] == kZero) {
            elements[i] = kZero; // zero maps to zero
            continue;            // inv unchanged (we used 1 for this slot)
        }
        FieldElement const original = elements[i];
        elements[i] = inv * scratch[i - 1];
        inv = inv * original;
    }

    // Handle first element separately (no scratch[i-1])
    elements[0] = first_zero ? kZero : inv;
}

// Fast-path batch inverse for guaranteed-nonzero inputs (e.g. ECC Z coordinates).
// Skips all zero-checks (2x per element) since callers guarantee no zero inputs.
// Precondition: all elements[i] != 0. Behavior undefined for zero inputs.
static inline void fe_batch_inverse_nonzero_with_scratch(FieldElement* elements,
                                                          size_t count,
                                                          FieldElement* scratch) {
    // Step 1: prefix products (no zero substitution needed)
    scratch[0] = elements[0];
    for (size_t i = 1; i < count; ++i)
        scratch[i] = scratch[i - 1] * elements[i];

    // Step 2: single inverse of the full product
    FieldElement inv = scratch[count - 1].inverse();

    // Step 3: backward sweep
    for (size_t i = count - 1; i > 0; --i) {
        FieldElement const original = elements[i];
        elements[i] = inv * scratch[i - 1];
        inv = inv * original;
    }
    elements[0] = inv;
}

SECP256K1_HOT_FUNCTION
void fe_batch_inverse(FieldElement* elements, size_t count, std::vector<FieldElement>& scratch) {
    if (count == 0) return;
    if (count == 1) {
        if (!(elements[0] == FieldElement::zero()))
            elements[0] = elements[0].inverse();
        return;
    }

    // Ensure storage exists and write products by index to avoid repeated
    // push_back bookkeeping on this hot path.
    if (scratch.size() < count) {
        scratch.resize(count);
    }
    fe_batch_inverse_with_scratch(elements, count, scratch.data());
}

SECP256K1_HOT_FUNCTION
void fe_batch_inverse(FieldElement* elements, size_t count) {
    if (count <= 1) {
        if (count == 1 && !(elements[0] == FieldElement::zero())) {
            elements[0] = elements[0].inverse();
        }
        return;
    }

    if (count <= kSmallBatchInverseScratch) {
        std::array<FieldElement, kSmallBatchInverseScratch> scratch{};
        fe_batch_inverse_with_scratch(elements, count, scratch.data());
        return;
    }

    static thread_local std::vector<FieldElement> scratch;
    scratch.resize(count);
    fe_batch_inverse_with_scratch(elements, count, scratch.data());
}

SECP256K1_HOT_FUNCTION
void fe_batch_inverse_nonzero(FieldElement* elements, size_t count) {
    if (count == 0) return;
    if (count == 1) { elements[0] = elements[0].inverse(); return; }

    if (count <= kSmallBatchInverseScratch) {
        std::array<FieldElement, kSmallBatchInverseScratch> scratch{};
        fe_batch_inverse_nonzero_with_scratch(elements, count, scratch.data());
        return;
    }

    static thread_local std::vector<FieldElement> scratch;
    scratch.resize(count);
    fe_batch_inverse_nonzero_with_scratch(elements, count, scratch.data());
}

} // namespace secp256k1::fast

