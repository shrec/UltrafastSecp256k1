// ============================================================================
// u128_compat.hpp -- Portable 128-bit unsigned integer for FE52 kernels
// ============================================================================
//
// On 64-bit GCC/Clang with native __int128: `secp256k1::detail::u128_compat`
// is a type alias for `unsigned __int128` (zero overhead).
//
// On MSVC x64 (no native unsigned __int128): it is a struct backed by
// _umul128/_addcarry_u64 intrinsics, avoiding the slow 32x32 schoolbook fallback
// in the FE52 hot path.
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

// Force-inline qualifier for the tiny pointer-accumulation micro-ops below. These
// MUST inline (they are 1-3 instructions); a real call would defeat the whole point.
// On MSVC cl the GENERAL "avoid __forceinline for static libs" guidance (config.hpp)
// is about large functions — these are micro-ops, so __forceinline is correct here.
#if defined(_MSC_VER)
  #define SECP256K1_U128_FI __forceinline
#elif defined(__GNUC__) || defined(__clang__)
  #define SECP256K1_U128_FI inline __attribute__((always_inline))
#else
  #define SECP256K1_U128_FI inline
#endif

#if defined(__SIZEOF_INT128__) && !defined(SECP256K1_NO_INT128)
// Native __int128 is correct on this platform — alias directly.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif

namespace secp256k1 { namespace detail {
using u128_compat = unsigned __int128;

// Pointer-accumulation helpers (signatures mirror the struct path so the FE52
// Comba kernels call ONE idiom on every platform — see field_52_impl.hpp). On
// native __int128 these ARE the operator forms: identical optimal MULX/ADCX codegen,
// zero behavioural change.
SECP256K1_U128_FI void u128_mul(u128_compat* r, std::uint64_t a, std::uint64_t b) noexcept { *r = (u128_compat)a * b; }
SECP256K1_U128_FI void u128_accum_mul(u128_compat* r, std::uint64_t a, std::uint64_t b) noexcept { *r += (u128_compat)a * b; }
SECP256K1_U128_FI void u128_accum_u64(u128_compat* r, std::uint64_t a) noexcept { *r += a; }
} }

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#else
// Struct implementation. Uses MSVC x64 intrinsics where available, otherwise
// falls back to a portable 32-bit-safe implementation. The fallback is used on
// wasm32 (Emscripten emulated __int128 is buggy) and any other target without
// __int128.

#if defined(_MSC_VER) && !defined(__clang__) && defined(_M_X64)
#include <intrin.h>
#define SECP256K1_U128_COMPAT_MSVC_X64 1
#endif

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
#if defined(SECP256K1_U128_COMPAT_MSVC_X64)
        unsigned __int64 hi_part = 0;
        const unsigned __int64 lo_part = _umul128(
            static_cast<unsigned __int64>(lo),
            static_cast<unsigned __int64>(y),
            &hi_part);
        return u128_compat{
            static_cast<std::uint64_t>(hi_part),
            static_cast<std::uint64_t>(lo_part)};
#else
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
#endif
    }

    // Addition with carry.
    u128_compat& operator+=(const u128_compat& other) noexcept {
#if defined(SECP256K1_U128_COMPAT_MSVC_X64)
        unsigned __int64 out_lo = 0;
        unsigned __int64 out_hi = 0;
        const unsigned char carry = _addcarry_u64(
            0,
            static_cast<unsigned __int64>(lo),
            static_cast<unsigned __int64>(other.lo),
            &out_lo);
        (void)_addcarry_u64(
            carry,
            static_cast<unsigned __int64>(hi),
            static_cast<unsigned __int64>(other.hi),
            &out_hi);
        lo = static_cast<std::uint64_t>(out_lo);
        hi = static_cast<std::uint64_t>(out_hi);
        return *this;
#else
        const std::uint64_t prev = lo;
        lo += other.lo;
        hi += other.hi + (lo < prev ? 1ULL : 0ULL);
        return *this;
#endif
    }
    u128_compat& operator+=(std::uint64_t x) noexcept {
#if defined(SECP256K1_U128_COMPAT_MSVC_X64)
        unsigned __int64 out_lo = 0;
        unsigned __int64 out_hi = 0;
        const unsigned char carry = _addcarry_u64(
            0,
            static_cast<unsigned __int64>(lo),
            static_cast<unsigned __int64>(x),
            &out_lo);
        (void)_addcarry_u64(
            carry,
            static_cast<unsigned __int64>(hi),
            0,
            &out_hi);
        lo = static_cast<std::uint64_t>(out_lo);
        hi = static_cast<std::uint64_t>(out_hi);
        return *this;
#else
        const std::uint64_t prev = lo;
        lo += x;
        if (lo < prev) hi += 1ULL;
        return *this;
#endif
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
    // In-place shift: mutate lo/hi directly (no temporary struct materialization,
    // matching libsecp's secp256k1_u128_rshift). The FE52 kernel only shifts by
    // 52/64, but this stays correct for any 0<=n<128.
    u128_compat& operator>>=(unsigned n) noexcept {
        if (n == 0) return *this;
        if (n >= 64) { lo = (n >= 128) ? 0ULL : (hi >> (n - 64)); hi = 0; }
        else { lo = (lo >> n) | (hi << (64 - n)); hi >>= n; }
        return *this;
    }

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

// ---------------------------------------------------------------------------
// Pointer-accumulation helpers — the libsecp int128_struct_impl.h:58-72 idiom.
// The struct path's WHOLE performance problem in context was operator+ returning a
// fresh {lo,hi} per term in a multi-product sum (d = p0+p1+p2+p3), which spills
// under the register pressure of an inlined point op. These mutate ONE named
// accumulator in place, so the optimizer keeps {lo,hi} in a register pair across
// the entire Comba column — exactly what libsecp does to win on MSVC cl.
// ---------------------------------------------------------------------------
SECP256K1_U128_FI std::uint64_t u128_mul64(std::uint64_t a, std::uint64_t b, std::uint64_t* hi) noexcept {
#if defined(SECP256K1_U128_COMPAT_MSVC_X64)
    return _umul128(a, b, hi);
#else
    const std::uint64_t x_lo = a & 0xFFFFFFFFULL, x_hi = a >> 32;
    const std::uint64_t y_lo = b & 0xFFFFFFFFULL, y_hi = b >> 32;
    const std::uint64_t p00 = x_lo * y_lo, p01 = x_lo * y_hi, p10 = x_hi * y_lo, p11 = x_hi * y_hi;
    const std::uint64_t mid = (p00 >> 32) + (p01 & 0xFFFFFFFFULL) + (p10 & 0xFFFFFFFFULL);
    *hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return (p00 & 0xFFFFFFFFULL) | (mid << 32);
#endif
}
// r = a*b
SECP256K1_U128_FI void u128_mul(u128_compat* r, std::uint64_t a, std::uint64_t b) noexcept {
    r->lo = u128_mul64(a, b, &r->hi);
}
// r += a*b   (libsecp carry: r->hi += hi + (r->lo < lo))
SECP256K1_U128_FI void u128_accum_mul(u128_compat* r, std::uint64_t a, std::uint64_t b) noexcept {
    std::uint64_t hi;
    const std::uint64_t lo = u128_mul64(a, b, &hi);
    r->lo += lo;
    r->hi += hi + (r->lo < lo);
}
// r += a   (64-bit addend)
SECP256K1_U128_FI void u128_accum_u64(u128_compat* r, std::uint64_t a) noexcept {
    r->lo += a;
    r->hi += (r->lo < a);
}

} } // namespace secp256k1::detail

#if defined(SECP256K1_U128_COMPAT_MSVC_X64)
#undef SECP256K1_U128_COMPAT_MSVC_X64
#endif

#endif // __SIZEOF_INT128__ && !SECP256K1_NO_INT128

#undef SECP256K1_U128_FI

#endif // SECP256K1_U128_COMPAT_HPP
