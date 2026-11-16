// ============================================================================
// lanes/field_ifma512x8.hpp -- Ifma512x8: 8 field elements per zmm register
// ============================================================================
//
// Radix 2^52 x 5 limbs, EVEX vpmadd52luq / vpmadd52huq on zmm, __mmask8 lane
// masks. Requires AVX512F + AVX512VL + AVX512IFMA. The algorithm and the
// limb-bound invariant are documented in field_r52.hpp.
//
// Include ONLY from a translation unit compiled with -mavx512f -mavx512vl
// -mavx512ifma (MSVC: /arch:AVX512), i.e. src/cpu/src/lanes/lanes_ifma512.cpp.
//
// PUBLIC DATA ONLY: variable time (verification paths).
// ============================================================================

#ifndef SECP256K1_LANES_FIELD_IFMA512X8_HPP
#define SECP256K1_LANES_FIELD_IFMA512X8_HPP

#if (defined(__GNUC__) || defined(__clang__)) && \
    !(defined(__AVX512F__) && defined(__AVX512VL__) && defined(__AVX512IFMA__))
#error "field_ifma512x8.hpp needs -mavx512f -mavx512vl -mavx512ifma (include it only from lanes_ifma512.cpp)"
#endif

#include <immintrin.h>

#include <cstdint>

#include "field_r52.hpp"

namespace secp256k1::lanes::field {

struct VIfma512 {
    using Vec = __m512i;
    using Mask = __mmask8;
    static constexpr unsigned kLanes = 8;

    static Vec zero() noexcept { return _mm512_setzero_si512(); }
    static Vec set1(std::uint64_t x) noexcept { return _mm512_set1_epi64(static_cast<long long>(x)); }
    static Vec add(Vec a, Vec b) noexcept { return _mm512_add_epi64(a, b); }
    static Vec sub(Vec a, Vec b) noexcept { return _mm512_sub_epi64(a, b); }
    static Vec and_(Vec a, Vec b) noexcept { return _mm512_and_si512(a, b); }
    static Vec or_(Vec a, Vec b) noexcept { return _mm512_or_si512(a, b); }
    static Vec xor_(Vec a, Vec b) noexcept { return _mm512_xor_si512(a, b); }
    template <int N> static Vec srli(Vec a) noexcept { return _mm512_srli_epi64(a, N); }
    static Vec madd52lo(Vec acc, Vec a, Vec b) noexcept { return _mm512_madd52lo_epu64(acc, a, b); }
    static Vec madd52hi(Vec acc, Vec a, Vec b) noexcept { return _mm512_madd52hi_epu64(acc, a, b); }
    static Vec loadu(const std::uint64_t* p) noexcept { return _mm512_loadu_si512(p); }
    static void storeu(std::uint64_t* p, Vec v) noexcept { _mm512_storeu_si512(p, v); }
    static Mask eq_mask(Vec a, Vec b) noexcept { return _mm512_cmpeq_epi64_mask(a, b); }
    // m ? a : b per lane (_mm512_mask_blend_epi64 takes its second operand where m is set).
    static Vec blend(Mask m, Vec a, Vec b) noexcept { return _mm512_mask_blend_epi64(m, b, a); }
    static bool all_zero(Vec v) noexcept { return _mm512_test_epi64_mask(v, v) == 0; }
    static unsigned mask_bits(Mask m) noexcept { return static_cast<unsigned>(m); }
    static Mask mask_from_bits(unsigned bits) noexcept { return static_cast<Mask>(bits & 0xFFu); }
};

using Ifma512x8 = FeR52<VIfma512>;

} // namespace secp256k1::lanes::field

#endif // SECP256K1_LANES_FIELD_IFMA512X8_HPP
