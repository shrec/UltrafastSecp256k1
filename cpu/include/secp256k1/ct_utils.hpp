#ifndef SECP256K1_CT_UTILS_HPP
#define SECP256K1_CT_UTILS_HPP

// ============================================================================
// Constant-Time Utilities -- High-Level API
// ============================================================================
// Provides byte-level constant-time operations for use in protocol
// implementations (ECDSA, Schnorr, ECDH, etc.).
//
// All functions have fixed execution time regardless of input values.
// No secret-dependent branches or memory access patterns.
//
// Audit Status:
//   - Value barriers: compiler optimization fence via inline asm / volatile
//   - Mask generation: arithmetic (no branches)
//   - Conditional ops: bitwise (no cmov -- explicit XOR/AND/OR)
//   - Table lookup: full scan (no early exit)
//
// Layers:
//   ct/ops.hpp     -- 64-bit primitives (value_barrier, masks, cmov, cswap)
//   ct/field.hpp   -- FieldElement CT ops (add/sub/mul/sqr/inv)
//   ct/scalar.hpp  -- Scalar CT ops (add/sub/neg/cmov/cswap)
//   ct/point.hpp   -- Point CT ops (complete addition, CT scalar_mul)
//   ct_utils.hpp   -- THIS FILE: byte-level utilities for protocols
// ============================================================================

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <array>
#include "secp256k1/ct/ops.hpp"

namespace secp256k1::ct {

// -- Byte-level Constant-Time Compare -----------------------------------------
// Returns true if a[0..len) == b[0..len). Constant-time (no early exit).
inline bool ct_equal(const void* a, const void* b, std::size_t len) noexcept {
    const auto* pa = static_cast<const std::uint8_t*>(a);
    const auto* pb = static_cast<const std::uint8_t*>(b);

    std::uint64_t diff = 0;
    // Process 8 bytes at a time
    std::size_t i = 0;
    for (; i + 8 <= len; i += 8) {
        std::uint64_t va, vb;
        std::memcpy(&va, pa + i, 8);
        std::memcpy(&vb, pb + i, 8);
        diff |= va ^ vb;
    }
    // Remaining bytes
    for (; i < len; ++i) {
        diff |= static_cast<std::uint64_t>(pa[i] ^ pb[i]);
    }

    return is_zero_mask(diff) != 0;
}

// Template version for fixed-size arrays
template<std::size_t N>
inline bool ct_equal(const std::array<std::uint8_t, N>& a,
                     const std::array<std::uint8_t, N>& b) noexcept {
    return ct_equal(a.data(), b.data(), N);
}

// -- Byte-level Constant-Time Conditional Copy --------------------------------
// if (flag) memcpy(dst, src, len);  -- constant-time
inline void ct_memcpy_if(void* dst, const void* src, std::size_t len,
                         bool flag) noexcept {
    auto mask = bool_to_mask(flag);
    auto mask8 = static_cast<std::uint8_t>(mask & 0xFF);

    auto* pd = static_cast<std::uint8_t*>(dst);
    const auto* ps = static_cast<const std::uint8_t*>(src);

    for (std::size_t i = 0; i < len; ++i) {
        pd[i] ^= (pd[i] ^ ps[i]) & mask8;
    }
}

// -- Constant-Time Conditional Swap -------------------------------------------
// if (flag) swap(a[0..len), b[0..len));  -- constant-time
inline void ct_memswap_if(void* a, void* b, std::size_t len,
                          bool flag) noexcept {
    auto mask = bool_to_mask(flag);
    auto mask8 = static_cast<std::uint8_t>(mask & 0xFF);

    auto* pa = static_cast<std::uint8_t*>(a);
    auto* pb = static_cast<std::uint8_t*>(b);

    for (std::size_t i = 0; i < len; ++i) {
        std::uint8_t diff = (pa[i] ^ pb[i]) & mask8;
        pa[i] ^= diff;
        pb[i] ^= diff;
    }
}

// -- Constant-Time Zero Check -------------------------------------------------
// Returns true if all bytes are zero. Constant-time.
inline bool ct_is_zero(const void* data, std::size_t len) noexcept {
    const auto* p = static_cast<const std::uint8_t*>(data);
    std::uint64_t acc = 0;
    for (std::size_t i = 0; i < len; ++i) {
        acc |= static_cast<std::uint64_t>(p[i]);
    }
    return is_zero_mask(acc) != 0;
}

template<std::size_t N>
inline bool ct_is_zero(const std::array<std::uint8_t, N>& data) noexcept {
    return ct_is_zero(data.data(), N);
}

// -- Constant-Time Memory Set -------------------------------------------------
// Guaranteed not to be optimized away by the compiler.
inline void ct_memzero(void* data, std::size_t len) noexcept {
    auto* p = static_cast<volatile std::uint8_t*>(data);
    for (std::size_t i = 0; i < len; ++i) {
        p[i] = 0;
    }
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "r"(data) : "memory");
#endif
}

// -- Constant-Time Byte Select ------------------------------------------------
// Returns a if flag is true, b otherwise. No branch.
inline std::uint8_t ct_select_byte(std::uint8_t a, std::uint8_t b,
                                    bool flag) noexcept {
    auto mask = bool_to_mask(flag);
    auto mask8 = static_cast<std::uint8_t>(mask & 0xFF);
    return static_cast<std::uint8_t>((a & mask8) | (b & ~mask8));
}

// -- Constant-Time Lexicographic Compare --------------------------------------
// Returns: -1 if a < b, 0 if a == b, 1 if a > b. Fully branchless.
//
// For the common 32-byte case: fully unrolled, no loops, no loop-carried
// dependencies. This prevents the compiler from inserting branches on
// the "decided" flag to short-circuit iterations.

namespace ct_compare_detail {

// Branchless unsigned compare: returns {gt, lt} where each is 0 or 1.
// value_barrier on BOTH inputs prevents Clang from inserting beq/bne
// branches before the sltu instructions (observed on RISC-V Clang 21).
inline void ct_cmp_pair(std::uint64_t wa, std::uint64_t wb,
                        std::uint64_t& gt, std::uint64_t& lt) noexcept {
    ct::value_barrier(wa);
    ct::value_barrier(wb);
#if defined(__riscv) && (__riscv_xlen == 64)
    asm volatile("sltu %0, %2, %1" : "=r"(gt) : "r"(wa), "r"(wb));
    asm volatile("sltu %0, %2, %1" : "=r"(lt) : "r"(wb), "r"(wa));
#else
    gt = static_cast<std::uint64_t>(wa > wb);
    lt = static_cast<std::uint64_t>(wa < wb);
#endif
}

// Load 8 bytes + bswap for lexicographic order
inline std::uint64_t ct_load_be(const std::uint8_t* p) noexcept {
    std::uint64_t v;
    std::memcpy(&v, p, 8);
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_bswap64(v);
#elif defined(_MSC_VER)
    return _byteswap_uint64(v);
#else
    // Generic fallback
    return ((v >> 56) & 0xFF) | ((v >> 40) & 0xFF00) |
           ((v >> 24) & 0xFF0000) | ((v >> 8) & 0xFF000000) |
           ((v << 8) & 0xFF00000000) | ((v << 24) & 0xFF0000000000) |
           ((v << 40) & 0xFF000000000000) | ((v << 56));
#endif
}

} // namespace ct_compare_detail

inline int ct_compare(const void* a, const void* b, std::size_t len) noexcept {
    const auto* pa = static_cast<const std::uint8_t*>(a);
    const auto* pb = static_cast<const std::uint8_t*>(b);

    // ---- Fast path: 32 bytes (fully unrolled, zero branches) ----
    // Algorithm: reverse-scan accumulation.
    //   Process words 3→2→1→0 (least significant first).
    //   Each differing word OVERRIDES the running result.
    //   Final result reflects the FIRST (most significant) differing word.
    //   value_barrier after every step prevents Clang from injecting
    //   beq/bne branches (observed with Clang 21 RISC-V).
    if (len == 32) {
        using namespace ct_compare_detail;

        // Load all 4 word pairs in big-endian (lexicographic order)
        std::uint64_t w0a = ct_load_be(pa +  0), w0b = ct_load_be(pb +  0);
        std::uint64_t w1a = ct_load_be(pa +  8), w1b = ct_load_be(pb +  8);
        std::uint64_t w2a = ct_load_be(pa + 16), w2b = ct_load_be(pb + 16);
        std::uint64_t w3a = ct_load_be(pa + 24), w3b = ct_load_be(pb + 24);

        std::uint64_t result = 0;

        // Word 3 (bytes 24-31, least significant)
        {
            std::uint64_t gt, lt;
            ct_cmp_pair(w3a, w3b, gt, lt);
            std::uint64_t differs = gt | lt;  // 0 or 1
            ct::value_barrier(differs);
            std::uint64_t mask = 0ULL - differs;
            result = (gt - lt) & mask;  // result was 0
        }
        ct::value_barrier(result);

        // Word 2 (bytes 16-23)
        {
            std::uint64_t gt, lt;
            ct_cmp_pair(w2a, w2b, gt, lt);
            std::uint64_t differs = gt | lt;
            ct::value_barrier(differs);
            std::uint64_t mask = 0ULL - differs;
            result = ((gt - lt) & mask) | (result & ~mask);
        }
        ct::value_barrier(result);

        // Word 1 (bytes 8-15)
        {
            std::uint64_t gt, lt;
            ct_cmp_pair(w1a, w1b, gt, lt);
            std::uint64_t differs = gt | lt;
            ct::value_barrier(differs);
            std::uint64_t mask = 0ULL - differs;
            result = ((gt - lt) & mask) | (result & ~mask);
        }
        ct::value_barrier(result);

        // Word 0 (bytes 0-7, most significant — overrides all)
        {
            std::uint64_t gt, lt;
            ct_cmp_pair(w0a, w0b, gt, lt);
            std::uint64_t differs = gt | lt;
            ct::value_barrier(differs);
            std::uint64_t mask = 0ULL - differs;
            result = ((gt - lt) & mask) | (result & ~mask);
        }

#if defined(__GNUC__) || defined(__clang__)
        asm volatile("" : "+r"(result));
#endif
        return static_cast<int>(static_cast<std::int64_t>(result));
    }

    // ---- General path: arbitrary length ----
    std::uint64_t result  = 0;
    std::uint64_t decided = 0;

    std::size_t i = 0;
    for (; i + 8 <= len; i += 8) {
        std::uint64_t wa, wb;
        std::memcpy(&wa, pa + i, 8);
        std::memcpy(&wb, pb + i, 8);

        // Convert to big-endian for lexicographic comparison
#if defined(__GNUC__) || defined(__clang__)
        wa = __builtin_bswap64(wa);
        wb = __builtin_bswap64(wb);
#elif defined(_MSC_VER)
        wa = _byteswap_uint64(wa);
        wb = _byteswap_uint64(wb);
#endif
        // Barrier inputs: prevent compiler from comparing wa/wb directly
        // (Clang 21 RISC-V inserts beq before sltu without these)
        ct::value_barrier(wa);
        ct::value_barrier(wb);
        std::uint64_t xor_val = wa ^ wb;
        // nz = 1 if words differ, 0 otherwise
#if defined(__riscv) && (__riscv_xlen == 64)
        std::uint64_t nz;
        asm volatile("snez %0, %1" : "=r"(nz) : "r"(xor_val));
#else
        std::uint64_t nz = ((xor_val | (0ULL - xor_val)) >> 63) & 1ULL;
#endif

        // Barrier on decided only: prevent compiler from short-circuiting
        ct::value_barrier(decided);

        // take = 1 only for the very first differing word
        std::uint64_t take = nz & (1ULL - decided);

        // mask = all-ones when take==1, zero when take==0
        std::uint64_t mask = 0ULL - take;

        // Branchless unsigned compare: ct_cmp_pair-style barriers on inputs
        std::uint64_t gt, lt;
        ct::value_barrier(wa);
        ct::value_barrier(wb);
#if defined(__riscv) && (__riscv_xlen == 64)
        asm volatile("sltu %0, %2, %1" : "=r"(gt) : "r"(wa), "r"(wb));
        asm volatile("sltu %0, %2, %1" : "=r"(lt) : "r"(wb), "r"(wa));
#else
        gt = static_cast<std::uint64_t>(wa > wb);
        lt = static_cast<std::uint64_t>(wa < wb);
#endif
        // diff_sign encodes: 1 = a>b, 0 = equal, -1 (0xFFFF...) = a<b
        std::uint64_t diff_sign = gt - lt;

        result  = (diff_sign & mask) | (result & ~mask);
        decided |= nz;

#if defined(__GNUC__) || defined(__clang__)
        asm volatile("" : "+r"(result), "+r"(decided));
#endif
    }

    // Remaining bytes (< 8) -- byte-by-byte fallback
    for (; i < len; ++i) {
        std::uint64_t ai = pa[i];
        std::uint64_t bi = pb[i];
        std::uint64_t diff = ai ^ bi;

#if defined(__riscv) && (__riscv_xlen == 64)
        std::uint64_t nz;
        asm volatile("snez %0, %1" : "=r"(nz) : "r"(diff));
#else
        std::uint64_t nz = ((diff | (0ULL - diff)) >> 63) & 1ULL;
#endif
        ct::value_barrier(decided);
        std::uint64_t take = nz & (1ULL - decided);
        std::uint64_t mask = 0ULL - take;

        std::uint64_t gt_b, lt_b;
        ct::value_barrier(ai);
        ct::value_barrier(bi);
#if defined(__riscv) && (__riscv_xlen == 64)
        asm volatile("sltu %0, %2, %1" : "=r"(gt_b) : "r"(ai), "r"(bi));
        asm volatile("sltu %0, %2, %1" : "=r"(lt_b) : "r"(bi), "r"(ai));
#else
        gt_b = static_cast<std::uint64_t>(ai > bi);
        lt_b = static_cast<std::uint64_t>(ai < bi);
#endif
        std::uint64_t diff_sign = gt_b - lt_b;

        result  = (diff_sign & mask) | (result & ~mask);
        decided |= nz;

#if defined(__GNUC__) || defined(__clang__)
        asm volatile("" : "+r"(result), "+r"(decided));
#endif
    }

    // Normalise to {-1, 0, 1} without branches.
    // result is 0, 1, or 0xFFFFFFFFFFFFFFFF (-1 as uint64)
    return static_cast<int>(static_cast<std::int64_t>(result));
}

} // namespace secp256k1::ct

#endif // SECP256K1_CT_UTILS_HPP
