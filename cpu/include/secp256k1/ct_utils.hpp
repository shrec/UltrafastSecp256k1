#ifndef SECP256K1_CT_UTILS_HPP
#define SECP256K1_CT_UTILS_HPP

// ============================================================================
// Constant-Time Utilities — High-Level API
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
//   - Conditional ops: bitwise (no cmov — explicit XOR/AND/OR)
//   - Table lookup: full scan (no early exit)
//
// Layers:
//   ct/ops.hpp     — 64-bit primitives (value_barrier, masks, cmov, cswap)
//   ct/field.hpp   — FieldElement CT ops (add/sub/mul/sqr/inv)
//   ct/scalar.hpp  — Scalar CT ops (add/sub/neg/cmov/cswap)
//   ct/point.hpp   — Point CT ops (complete addition, CT scalar_mul)
//   ct_utils.hpp   — THIS FILE: byte-level utilities for protocols
// ============================================================================

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <array>
#include "secp256k1/ct/ops.hpp"

namespace secp256k1::ct {

// ── Byte-level Constant-Time Compare ─────────────────────────────────────────
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

// ── Byte-level Constant-Time Conditional Copy ────────────────────────────────
// if (flag) memcpy(dst, src, len);  — constant-time
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

// ── Constant-Time Conditional Swap ───────────────────────────────────────────
// if (flag) swap(a[0..len), b[0..len));  — constant-time
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

// ── Constant-Time Zero Check ─────────────────────────────────────────────────
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

// ── Constant-Time Memory Set ─────────────────────────────────────────────────
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

// ── Constant-Time Byte Select ────────────────────────────────────────────────
// Returns a if flag is true, b otherwise. No branch.
inline std::uint8_t ct_select_byte(std::uint8_t a, std::uint8_t b,
                                    bool flag) noexcept {
    auto mask = bool_to_mask(flag);
    auto mask8 = static_cast<std::uint8_t>(mask & 0xFF);
    return static_cast<std::uint8_t>((a & mask8) | (b & ~mask8));
}

// ── Constant-Time Lexicographic Compare ──────────────────────────────────────
// Returns: -1 if a < b, 0 if a == b, 1 if a > b. Constant-time.
inline int ct_compare(const void* a, const void* b, std::size_t len) noexcept {
    const auto* pa = static_cast<const std::uint8_t*>(a);
    const auto* pb = static_cast<const std::uint8_t*>(b);

    // We accumulate the result without branching
    int result = 0;
    int decided = 0;  // Have we found a difference?

    for (std::size_t i = 0; i < len; ++i) {
        int diff = static_cast<int>(pa[i]) - static_cast<int>(pb[i]);
        // Only take the first non-zero diff
        int is_diff = (diff != 0) ? 1 : 0;
        int take = is_diff & ~decided;
        result = (take != 0) ? diff : result;
        decided |= is_diff;
    }

    // Normalize to -1, 0, 1
    return (result > 0) - (result < 0);
}

} // namespace secp256k1::ct

#endif // SECP256K1_CT_UTILS_HPP
