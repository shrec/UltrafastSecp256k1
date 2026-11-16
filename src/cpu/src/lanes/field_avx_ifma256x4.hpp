// ============================================================================
// lanes/field_avx_ifma256x4.hpp -- AvxIfma256x4: 4 elements per ymm (VEX IFMA)
// ============================================================================
//
// Radix 2^52 x 5 limbs, VEX-encoded vpmadd52luq / vpmadd52huq (AVX-IFMA, no
// AVX-512 needed, e.g. Sierra Forest and Arrow Lake),
// vector lane masks (all-ones / all-zero per 64-bit lane). The algorithm and
// the limb-bound invariant are documented in field_r52.hpp.
//
// Built only when the compiler accepts -mavxifma and the
// _mm256_madd52lo_avx_epu64 intrinsic (checked in src/cpu/CMakeLists.txt).
// Include ONLY from a translation unit compiled with -mavx2 -mavxifma, i.e.
// src/cpu/src/lanes/lanes_avx_ifma.cpp.
//
// PUBLIC DATA ONLY: variable time (verification paths).
// ============================================================================

#ifndef SECP256K1_LANES_FIELD_AVX_IFMA256X4_HPP
#define SECP256K1_LANES_FIELD_AVX_IFMA256X4_HPP

#if (defined(__GNUC__) || defined(__clang__)) && !(defined(__AVX2__) && defined(__AVXIFMA__))
#error "field_avx_ifma256x4.hpp needs -mavx2 -mavxifma (include it only from lanes_avx_ifma.cpp)"
#endif

#include <immintrin.h>

#include <cstdint>

#include "field_r52.hpp"

namespace secp256k1::lanes::field {

struct VAvxIfma {
    using Vec = __m256i;
    using Mask = __m256i;   // all-ones lanes are "set"
    static constexpr unsigned kLanes = 4;

    static Vec zero() noexcept { return _mm256_setzero_si256(); }
    static Vec set1(std::uint64_t x) noexcept { return _mm256_set1_epi64x(static_cast<long long>(x)); }
    static Vec add(Vec a, Vec b) noexcept { return _mm256_add_epi64(a, b); }
    static Vec sub(Vec a, Vec b) noexcept { return _mm256_sub_epi64(a, b); }
    static Vec and_(Vec a, Vec b) noexcept { return _mm256_and_si256(a, b); }
    static Vec or_(Vec a, Vec b) noexcept { return _mm256_or_si256(a, b); }
    static Vec xor_(Vec a, Vec b) noexcept { return _mm256_xor_si256(a, b); }
    template <int N> static Vec srli(Vec a) noexcept { return _mm256_srli_epi64(a, N); }
    static Vec madd52lo(Vec acc, Vec a, Vec b) noexcept { return _mm256_madd52lo_avx_epu64(acc, a, b); }
    static Vec madd52hi(Vec acc, Vec a, Vec b) noexcept { return _mm256_madd52hi_avx_epu64(acc, a, b); }
    static Vec loadu(const std::uint64_t* p) noexcept {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    }
    static void storeu(std::uint64_t* p, Vec v) noexcept {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
    }
    static Mask eq_mask(Vec a, Vec b) noexcept { return _mm256_cmpeq_epi64(a, b); }
    // m ? a : b per lane (blendv takes its second operand where the mask byte's top bit is set).
    static Vec blend(Mask m, Vec a, Vec b) noexcept { return _mm256_blendv_epi8(b, a, m); }
    static bool all_zero(Vec v) noexcept { return _mm256_testz_si256(v, v) != 0; }
    static unsigned mask_bits(Mask m) noexcept {
        return static_cast<unsigned>(_mm256_movemask_pd(_mm256_castsi256_pd(m)));
    }
    static Mask mask_from_bits(unsigned bits) noexcept {
        return _mm256_set_epi64x((bits & 8u) ? -1LL : 0LL, (bits & 4u) ? -1LL : 0LL,
                                 (bits & 2u) ? -1LL : 0LL, (bits & 1u) ? -1LL : 0LL);
    }
};

using AvxIfma256x4 = FeR52<VAvxIfma>;

} // namespace secp256k1::lanes::field

#endif // SECP256K1_LANES_FIELD_AVX_IFMA256X4_HPP
