// ============================================================================
// u128_compat.hpp -- Portable 128-bit unsigned integer for FE52 kernels
// ============================================================================
//
// On 64-bit GCC/Clang with native __int128: `secp256k1::detail::u128_compat`
// is a type alias for `unsigned __int128` (zero overhead).
//
// On wasm32 (Emscripten/Clang) when SECP256K1_NO_INT128 is defined, OR on
// 32-bit targets without __SIZEOF_INT128__: it is a struct with explicit
// 64x64 -> 128 multiplication and carry-aware addition. This is needed
// because Emscripten's compiler-rt __multi3 emulation produces wrong
// results for the 5x52 Comba/Barrett kernels (verified empirically by the
// wasm KAT 2G/3G/scalar_mul mismatches).
//
// All operations needed by field_52_impl.hpp are supported:
//   - construction from uint64_t
//   - explicit cast (uint64_t) to extract low half
//   - operator* (uint64_t) for 64x64 -> 128 multiplication (LHS must have
//     hi == 0 — the only pattern this file uses)
//   - operator+/+= (uint64_t) and operator+/+= (u128_compat)
//   - operator>> (unsigned), operator<<= for shifts
//   - operator| (uint64_t) and operator| (u128_compat)
//   - operator& (uint64_t) returning u128_compat (mask the low part)
//
// ============================================================================
#ifndef SECP256K1_U128_COMPAT_HPP
#define SECP256K1_U128_COMPAT_HPP

#include <cstdint>

#if defined(__SIZEOF_INT128__) && !defined(SECP256K1_NO_INT128)
// Native __int128 is correct on this platform — alias directly.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif

namespace secp256k1 { namespace detail {
using u128_compat = unsigned __int128;
} }

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#else
// Portable 32-bit-safe implementation. Used on wasm32 (Emscripten emulated
// __int128 is buggy) and any other target without __int128.

namespace secp256k1 { namespace detail {

struct u128_compat {
    std::uint64_t lo;
    std::uint64_t hi;

    constexpr u128_compat() noexcept : lo(0), hi(0) {}
    constexpr u128_compat(std::uint64_t x) noexcept : lo(x), hi(0) {}
    constexpr u128_compat(std::uint64_t high, std::uint64_t low) noexcept
        : lo(low), hi(high) {}

    // Extract low 64 bits.
    explicit constexpr operator std::uint64_t() const noexcept { return lo; }

    // 64x64 -> 128 multiplication.
    // Precondition: this->hi == 0. All call sites in field_52_impl.hpp
    // multiply a u128-widened u64 by another u64, so hi is always zero.
    u128_compat operator*(std::uint64_t y) const noexcept {
        const std::uint64_t x = lo;
        const std::uint64_t x_lo = x & 0xFFFFFFFFULL;
        const std::uint64_t x_hi = x >> 32;
        const std::uint64_t y_lo = y & 0xFFFFFFFFULL;
        const std::uint64_t y_hi = y >> 32;

        const std::uint64_t p00 = x_lo * y_lo;
        const std::uint64_t p01 = x_lo * y_hi;
        const std::uint64_t p10 = x_hi * y_lo;
        const std::uint64_t p11 = x_hi * y_hi;

        const std::uint64_t mid = (p00 >> 32) + (p01 & 0xFFFFFFFFULL) + (p10 & 0xFFFFFFFFULL);
        u128_compat r;
        r.lo = (p00 & 0xFFFFFFFFULL) | (mid << 32);
        r.hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
        return r;
    }

    // Addition with carry.
    u128_compat& operator+=(const u128_compat& other) noexcept {
        const std::uint64_t prev = lo;
        lo += other.lo;
        hi += other.hi + (lo < prev ? 1ULL : 0ULL);
        return *this;
    }
    u128_compat& operator+=(std::uint64_t x) noexcept {
        const std::uint64_t prev = lo;
        lo += x;
        if (lo < prev) hi += 1ULL;
        return *this;
    }
    friend u128_compat operator+(u128_compat a, const u128_compat& b) noexcept { a += b; return a; }
    friend u128_compat operator+(u128_compat a, std::uint64_t b) noexcept { a += b; return a; }
    friend u128_compat operator+(std::uint64_t a, u128_compat b) noexcept { b += a; return b; }

    // Right shift.
    u128_compat operator>>(unsigned n) const noexcept {
        if (n == 0) return *this;
        if (n >= 128) return u128_compat{0, 0};
        if (n >= 64) return u128_compat{0, hi >> (n - 64)};
        // 0 < n < 64
        return u128_compat{hi >> n, (lo >> n) | (hi << (64 - n))};
    }
    u128_compat& operator>>=(unsigned n) noexcept { *this = *this >> n; return *this; }

    // Left shift (rarely used — only for composing high/low halves).
    u128_compat operator<<(unsigned n) const noexcept {
        if (n == 0) return *this;
        if (n >= 128) return u128_compat{0, 0};
        if (n >= 64) return u128_compat{lo << (n - 64), 0};
        // 0 < n < 64
        return u128_compat{(hi << n) | (lo >> (64 - n)), lo << n};
    }
    u128_compat& operator<<=(unsigned n) noexcept { *this = *this << n; return *this; }

    // Bitwise.
    friend u128_compat operator|(u128_compat a, const u128_compat& b) noexcept {
        return u128_compat{a.hi | b.hi, a.lo | b.lo};
    }
    friend u128_compat operator|(u128_compat a, std::uint64_t b) noexcept {
        a.lo |= b; return a;
    }
    u128_compat& operator|=(std::uint64_t b) noexcept { lo |= b; return *this; }
    u128_compat& operator|=(const u128_compat& b) noexcept { lo |= b.lo; hi |= b.hi; return *this; }

    friend u128_compat operator&(u128_compat a, std::uint64_t b) noexcept {
        // Mask: keep low 64 bits ANDed with b; clear high.
        return u128_compat{0, a.lo & b};
    }
};

} } // namespace secp256k1::detail

#endif // __SIZEOF_INT128__ && !SECP256K1_NO_INT128

#endif // SECP256K1_U128_COMPAT_HPP
