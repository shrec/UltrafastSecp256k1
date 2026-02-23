// ============================================================================
// 5x52-bit Field Element -- Implementation
// ============================================================================
// Hybrid lazy-reduction field arithmetic for secp256k1.
//
// Multiplication and squaring kernels adapted from bitcoin-core/secp256k1
// field_5x52_int128_impl.h (MIT license).
//
// Key property: p = 2^256 - 0x1000003D1
//   -> 2^256 == 0x1000003D1 (mod p)
//   -> 2^260 == 0x1000003D10 (mod p)  [since d^5 = 2^(52*5) = 2^260]
// ============================================================================

#include "secp256k1/field_52.hpp"
#include <cstring>

// Require 128-bit integer support for the mul/sqr kernels.
// __SIZEOF_INT128__ is the canonical check -- defined on 64-bit GCC/Clang,
// NOT on 32-bit targets (armv7, i686, ESP32) even though __GNUC__ is set.
#if defined(__SIZEOF_INT128__)
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
    using uint128_t = unsigned __int128;
    #define SECP256K1_HAS_UINT128 1
#else
    // 32-bit or MSVC: __int128 unavailable.
    // FieldElement52 is unavailable; this TU compiles as empty.
    // The portable FieldElement (4x64 with carry chains) is used instead.
#endif

#ifdef SECP256K1_HAS_UINT128

namespace secp256k1::fast {

using namespace fe52_constants;

// ===========================================================================
// Construction
// ===========================================================================

FieldElement52 FieldElement52::zero() noexcept {
    return FieldElement52{{0, 0, 0, 0, 0}};
}

FieldElement52 FieldElement52::one() noexcept {
    return FieldElement52{{1, 0, 0, 0, 0}};
}

// ===========================================================================
// Conversion: from_fe, to_fe -- now inline in field_52_impl.hpp
// ===========================================================================

// ===========================================================================
// Normalization (out-of-line: not in inner ECC loops)
// ===========================================================================
// fe52_normalize_weak is now inline in field_52_impl.hpp
// fe52_normalize (full) is now inline in field_52_impl.hpp

// Out-of-line stub: delegates to the inline version for backward compat
void fe52_normalize(std::uint64_t* r) noexcept {
    fe52_normalize_inline(r);
}

// normalize_weak() is now inline in field_52_impl.hpp
// normalize() is now inline in field_52_impl.hpp

// operator+, add_assign, negate, negate_assign: now inline in field_52_impl.hpp

// ===========================================================================
// All hot-path functions moved to field_52_impl.hpp (inline, zero overhead):
//   fe52_mul_inner, fe52_sqr_inner, fe52_normalize_weak
//   operator*, square, mul_assign, square_inplace
//   operator+, add_assign, negate, negate_assign
//   normalizes_to_zero, half, normalize_weak
// ===========================================================================

// ===========================================================================
// Comparison
// ===========================================================================

bool FieldElement52::is_zero() const noexcept {
    // Normalize a copy and check
    FieldElement52 tmp = *this;
    tmp.normalize();
    return (tmp.n[0] | tmp.n[1] | tmp.n[2] | tmp.n[3] | tmp.n[4]) == 0;
}

// normalizes_to_zero: now inline in field_52_impl.hpp

bool FieldElement52::operator==(const FieldElement52& rhs) const noexcept {
    FieldElement52 a = *this, b = rhs;
    a.normalize();
    b.normalize();
    return (a.n[0] == b.n[0]) & (a.n[1] == b.n[1]) & (a.n[2] == b.n[2])
         & (a.n[3] == b.n[3]) & (a.n[4] == b.n[4]);
}

bool FieldElement52::operator!=(const FieldElement52& rhs) const noexcept {
    return !(*this == rhs);
}

// half: now inline in field_52_impl.hpp

// ===========================================================================
// Inverse: a^(p-2) via Fermat addition chain (255S + 14M in native FE52)
// ===========================================================================
// p-2 = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
// Shared addition chain with sqrt up to x223; final stage differs.

FieldElement52 FieldElement52::inverse() const noexcept {
    FieldElement52 x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223, t;
    const FieldElement52& a = *this;

    x2 = a.square() * a;
    x3 = x2.square() * a;

    x6 = x3;
    for (int j = 0; j < 3; ++j) x6.square_inplace();
    x6 = x6 * x3;

    x9 = x6;
    for (int j = 0; j < 3; ++j) x9.square_inplace();
    x9 = x9 * x3;

    x11 = x9;
    for (int j = 0; j < 2; ++j) x11.square_inplace();
    x11 = x11 * x2;

    x22 = x11;
    for (int j = 0; j < 11; ++j) x22.square_inplace();
    x22 = x22 * x11;

    x44 = x22;
    for (int j = 0; j < 22; ++j) x44.square_inplace();
    x44 = x44 * x22;

    x88 = x44;
    for (int j = 0; j < 44; ++j) x88.square_inplace();
    x88 = x88 * x44;

    x176 = x88;
    for (int j = 0; j < 88; ++j) x176.square_inplace();
    x176 = x176 * x88;

    x220 = x176;
    for (int j = 0; j < 44; ++j) x220.square_inplace();
    x220 = x220 * x44;

    x223 = x220;
    for (int j = 0; j < 3; ++j) x223.square_inplace();
    x223 = x223 * x3;

    // Final: 23S*x22 * 5S*a * 3S*x2 * 2S*a
    t = x223;
    for (int j = 0; j < 23; ++j) t.square_inplace();
    t = t * x22;
    for (int j = 0; j < 5; ++j) t.square_inplace();
    t = t * a;
    for (int j = 0; j < 3; ++j) t.square_inplace();
    t = t * x2;
    for (int j = 0; j < 2; ++j) t.square_inplace();
    t = t * a;
    return t;
}

// ===========================================================================
// Square Root: a^((p+1)/4) via addition chain (253S + 13M in native FE52)
// ===========================================================================
// (p+1)/4 for secp256k1 shares the x223 chain with inverse; final stage differs.

FieldElement52 FieldElement52::sqrt() const noexcept {
    FieldElement52 x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223, t;
    const FieldElement52& a = *this;

    x2 = a.square() * a;
    x3 = x2.square() * a;

    x6 = x3;
    for (int j = 0; j < 3; ++j) x6.square_inplace();
    x6 = x6 * x3;

    x9 = x6;
    for (int j = 0; j < 3; ++j) x9.square_inplace();
    x9 = x9 * x3;

    x11 = x9;
    for (int j = 0; j < 2; ++j) x11.square_inplace();
    x11 = x11 * x2;

    x22 = x11;
    for (int j = 0; j < 11; ++j) x22.square_inplace();
    x22 = x22 * x11;

    x44 = x22;
    for (int j = 0; j < 22; ++j) x44.square_inplace();
    x44 = x44 * x22;

    x88 = x44;
    for (int j = 0; j < 44; ++j) x88.square_inplace();
    x88 = x88 * x44;

    x176 = x88;
    for (int j = 0; j < 88; ++j) x176.square_inplace();
    x176 = x176 * x88;

    x220 = x176;
    for (int j = 0; j < 44; ++j) x220.square_inplace();
    x220 = x220 * x44;

    x223 = x220;
    for (int j = 0; j < 3; ++j) x223.square_inplace();
    x223 = x223 * x3;

    // Final: 23S*x22 * 6S*x2 * 2S (no trailing mul)
    t = x223;
    for (int j = 0; j < 23; ++j) t.square_inplace();
    t = t * x22;
    for (int j = 0; j < 6; ++j) t.square_inplace();
    t = t * x2;
    t.square_inplace();
    t.square_inplace();
    return t;
}

} // namespace secp256k1::fast

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#endif // SECP256K1_HAS_UINT128
