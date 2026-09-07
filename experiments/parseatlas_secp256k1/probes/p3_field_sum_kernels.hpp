#ifndef PARSEATLAS_P3_FIELD_SUM_KERNELS_HPP
#define PARSEATLAS_P3_FIELD_SUM_KERNELS_HPP

#include "secp256k1/field.hpp"
#include "secp256k1/field_52.hpp"
#include <cstddef>

namespace pa_p3 {

using FE = secp256k1::fast::FieldElement;
using FE52 = secp256k1::fast::FieldElement52;

// Final-only contract: return (x0 + sum(rhs[0..count))) mod p, canonical
// FE64. No prefix is observable. x0 and all RHS must be fully canonical;
// resident FE52 inputs additionally have normalized 52/48-bit limb bounds.
// rhs may be null only when count == 0, otherwise it names count readable
// objects. Inputs are read-only; no restrict/noalias promise is imposed.
// These public-count experimental schedules carry no CT certification.
inline FE sum_fe64(const FE& x0, const FE* rhs, std::size_t count) {
    FE accumulator = x0;
    for (std::size_t i = 0; i < count; ++i) accumulator = accumulator + rhs[i];
    return accumulator;
}

namespace detail {

template<bool WeakFirst>
inline void normalize_nonfinal(FE52& accumulator) noexcept {
    if constexpr (WeakFirst) accumulator.normalize_weak();
    accumulator.normalize();
}

template<bool WeakFirst>
inline FE decode_final(FE52 accumulator) noexcept {
    if constexpr (WeakFirst) accumulator.normalize_weak();
    // to_fe normalizes its copy fully: no redundant final normalize() here.
    return accumulator.to_fe();
}

template<std::size_t ChunkRhs, bool WeakFirst, class ReadRhs>
inline FE sum_fe52_impl(const FE& x0, std::size_t count, ReadRhs read_rhs) {
    static_assert(ChunkRhs >= 1, "P3 chunk must contain at least one RHS");
    static_assert(ChunkRhs <= (WeakFirst ? 4095U : 4094U),
                  "P3 chunk exceeds the canonical-seed normalization bound");
    if (count == 0) return x0;
    FE52 accumulator = FE52::from_fe(x0);
    std::size_t offset = 0;
    // One canonical accumulator is a term in EVERY chunk: direct full
    // normalization allows <=4095 total terms; weak-then-full <=4096.
    // count-offset avoids overflow from forming offset+ChunkRhs up front.
    while (count - offset > ChunkRhs) {
        for (std::size_t i = 0; i < ChunkRhs; ++i)
            accumulator.add_assign(read_rhs(offset + i));
        offset += ChunkRhs;
        normalize_nonfinal<WeakFirst>(accumulator);
    }
    for (std::size_t i = offset; i < count; ++i)
        accumulator.add_assign(read_rhs(i));
    return decode_final<WeakFirst>(accumulator);
}

} // namespace detail

template<std::size_t ChunkRhs, bool WeakFirst>
inline FE sum_fe52_e2e(const FE& x0, const FE* rhs, std::size_t count) {
    // RHS packing is inside the operation. x0 is packed once, not per chunk.
    return detail::sum_fe52_impl<ChunkRhs, WeakFirst>(x0, count,
        [rhs](std::size_t i) { return FE52::from_fe(rhs[i]); });
}

template<std::size_t ChunkRhs, bool WeakFirst>
inline FE sum_fe52_resident(const FE& x0, const FE52* rhs, std::size_t count) {
    // Caller owns/amortizes RHS preprocessing. Seed packing and final canonical
    // FE64 decoding remain inside this resident-region operation.
    return detail::sum_fe52_impl<ChunkRhs, WeakFirst>(x0, count,
        [rhs](std::size_t i) { return rhs[i]; });
}

} // namespace pa_p3
#endif // PARSEATLAS_P3_FIELD_SUM_KERNELS_HPP
