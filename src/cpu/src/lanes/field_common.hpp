// ============================================================================
// lanes/field_common.hpp -- ISA-independent pieces shared by the lane fields
// ============================================================================
//
// Included only by the per-ISA lane headers (field_r52.hpp, field_avx2x4.hpp),
// which are included only by translation units compiled for that ISA.
//
// Everything here has internal linkage (static / static templates). That is
// deliberate: an inline function with external linkage that is emitted in an
// AVX-512 translation unit could be the copy the linker keeps for a caller in
// a generic translation unit, which would execute AVX-512 instructions on a
// CPU without them. Internal linkage gives every TU its own copy.
//
// PUBLIC DATA ONLY: variable time by design (verification paths).
// ============================================================================

#ifndef SECP256K1_LANES_FIELD_COMMON_HPP
#define SECP256K1_LANES_FIELD_COMMON_HPP

#include <cstddef>
#include <cstdint>

namespace secp256k1::lanes::field {

// Bits [pos, pos + width) of a 256-bit little-endian 4x64 value (width <= 52).
static inline std::uint64_t lanes_bits_of(const std::uint64_t* w, unsigned pos,
                                          unsigned width) noexcept {
    const unsigned idx = pos / 64u;
    const unsigned sh = pos % 64u;
    std::uint64_t v = w[idx] >> sh;
    if (sh + width > 64u && idx + 1u < 4u) v |= w[idx + 1u] << (64u - sh);
    return v & ((std::uint64_t{1} << width) - 1u);
}

// 4x64 value -> `count` limbs of `radix` bits (the top limb gets the rest).
// Any 256-bit input is accepted; the top limb is then <= 2^(256 - radix*(count-1)) - 1.
static inline void lanes_split_limbs(const std::uint64_t* w, std::uint64_t* limbs,
                                     unsigned radix, unsigned count) noexcept {
    for (unsigned i = 0; i + 1u < count; ++i) limbs[i] = lanes_bits_of(w, i * radix, radix);
    const unsigned top_pos = radix * (count - 1u);
    limbs[count - 1u] = lanes_bits_of(w, top_pos, 256u - top_pos);
}

// Canonical limbs (each < 2^radix, value < 2^256) -> 4x64.
static inline void lanes_join_limbs(const std::uint64_t* limbs, std::uint64_t* w,
                                    unsigned radix, unsigned count) noexcept {
    w[0] = w[1] = w[2] = w[3] = 0;
    for (unsigned i = 0; i < count; ++i) {
        const unsigned pos = i * radix;
        const unsigned idx = pos / 64u;
        const unsigned sh = pos % 64u;
        w[idx] |= limbs[i] << sh;
        if (sh != 0u && sh + radix > 64u && idx + 1u < 4u) w[idx + 1u] |= limbs[i] >> (64u - sh);
    }
}

// x^(2^n) by repeated squaring.
template <class F>
static inline F fe_sqr_n(F x, int n) noexcept {
    for (int i = 0; i < n; ++i) x = fe_sqr(x);
    return x;
}

// a^(p-2) = a^-1 (0 -> 0), the libsecp256k1 addition chain for p - 2:
// 255 squarings and 15 multiplications, the same for every lane (no
// data-dependent control flow, though nothing here needs constant time).
// Built from the blocks x_k = a^(2^k - 1); the final exponent is
// ((((2^223-1)*2^23 + 2^22-1)*2^5 + 1)*2^3 + 3)*2^2 + 1 = 2^256 - 2^32 - 979 = p - 2.
template <class F>
static inline F fe_inverse(const F& a) noexcept {
    const F x2 = fe_mul(fe_sqr(a), a);
    const F x3 = fe_mul(fe_sqr(x2), a);
    const F x6 = fe_mul(fe_sqr_n(x3, 3), x3);
    const F x9 = fe_mul(fe_sqr_n(x6, 3), x3);
    const F x11 = fe_mul(fe_sqr_n(x9, 2), x2);
    const F x22 = fe_mul(fe_sqr_n(x11, 11), x11);
    const F x44 = fe_mul(fe_sqr_n(x22, 22), x22);
    const F x88 = fe_mul(fe_sqr_n(x44, 44), x44);
    const F x176 = fe_mul(fe_sqr_n(x88, 88), x88);
    const F x220 = fe_mul(fe_sqr_n(x176, 44), x44);
    const F x223 = fe_mul(fe_sqr_n(x220, 3), x3);
    F t = fe_mul(fe_sqr_n(x223, 23), x22);
    t = fe_mul(fe_sqr_n(t, 5), a);
    t = fe_mul(fe_sqr_n(t, 3), x2);
    return fe_mul(fe_sqr_n(t, 2), a);
}

} // namespace secp256k1::lanes::field

#endif // SECP256K1_LANES_FIELD_COMMON_HPP
