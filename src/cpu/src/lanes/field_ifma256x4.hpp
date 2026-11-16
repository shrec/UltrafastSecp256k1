// ============================================================================
// lanes/field_ifma256x4.hpp -- Ifma256x4: 4 field elements per ymm (EVEX IFMA)
// ============================================================================
//
// Radix 2^52 x 5 limbs, EVEX-encoded vpmadd52luq / vpmadd52huq on ymm,
// __mmask8 lane masks (low four bits used). Requires AVX512VL + AVX512IFMA
// (AVX512F is implied by VL). The algorithm and the limb-bound invariant are
// documented in field_r52.hpp.
//
// Why a 256-bit EVEX form exists next to Ifma512x8: on cores that split a zmm
// operation into two 256-bit halves (e.g. Zen 4) the 4-lane form may reach the
// same per-element throughput with half the register footprint. That is a
// hypothesis to measure; the dispatcher therefore never picks this backend by
// default, only through UFSECP_LANES=ifma256.
//
// Include ONLY from a translation unit compiled with -mavx512f -mavx512vl
// -mavx512ifma (MSVC: /arch:AVX512), i.e. src/cpu/src/lanes/lanes_ifma256.cpp.
//
// PUBLIC DATA ONLY: variable time (verification paths).
// ============================================================================

#ifndef SECP256K1_LANES_FIELD_IFMA256X4_HPP
#define SECP256K1_LANES_FIELD_IFMA256X4_HPP

#if (defined(__GNUC__) || defined(__clang__)) && \
    !(defined(__AVX512F__) && defined(__AVX512VL__) && defined(__AVX512IFMA__))
#error "field_ifma256x4.hpp needs -mavx512f -mavx512vl -mavx512ifma (include it only from lanes_ifma256.cpp)"
#endif

#include <immintrin.h>

#include <cstdint>

#include "field_r52.hpp"

namespace secp256k1::lanes::field {

struct VIfma256 {
    using Vec = __m256i;
    using Mask = __mmask8;
    static constexpr unsigned kLanes = 4;

    static Vec zero() noexcept { return _mm256_setzero_si256(); }
    static Vec set1(std::uint64_t x) noexcept { return _mm256_set1_epi64x(static_cast<long long>(x)); }
    static Vec add(Vec a, Vec b) noexcept { return _mm256_add_epi64(a, b); }
    static Vec sub(Vec a, Vec b) noexcept { return _mm256_sub_epi64(a, b); }
    static Vec and_(Vec a, Vec b) noexcept { return _mm256_and_si256(a, b); }
    static Vec or_(Vec a, Vec b) noexcept { return _mm256_or_si256(a, b); }
    static Vec xor_(Vec a, Vec b) noexcept { return _mm256_xor_si256(a, b); }
    template <int N> static Vec srli(Vec a) noexcept { return _mm256_srli_epi64(a, N); }
    static Vec madd52lo(Vec acc, Vec a, Vec b) noexcept { return _mm256_madd52lo_epu64(acc, a, b); }
    static Vec madd52hi(Vec acc, Vec a, Vec b) noexcept { return _mm256_madd52hi_epu64(acc, a, b); }
    static Vec loadu(const std::uint64_t* p) noexcept {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    }
    static void storeu(std::uint64_t* p, Vec v) noexcept {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
    }
    static Mask eq_mask(Vec a, Vec b) noexcept { return _mm256_cmpeq_epi64_mask(a, b); }
    // m ? a : b per lane (_mm256_mask_blend_epi64 takes its second operand where m is set).
    static Vec blend(Mask m, Vec a, Vec b) noexcept { return _mm256_mask_blend_epi64(m, b, a); }
    static bool all_zero(Vec v) noexcept { return _mm256_testz_si256(v, v) != 0; }
    static unsigned mask_bits(Mask m) noexcept { return static_cast<unsigned>(m) & 0xFu; }
    static Mask mask_from_bits(unsigned bits) noexcept { return static_cast<Mask>(bits & 0xFu); }
};

using Ifma256x4 = FeR52<VIfma256>;

} // namespace secp256k1::lanes::field

#endif // SECP256K1_LANES_FIELD_IFMA256X4_HPP
