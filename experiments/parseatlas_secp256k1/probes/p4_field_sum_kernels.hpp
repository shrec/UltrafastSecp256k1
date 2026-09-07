#ifndef PARSEATLAS_P4_FIELD_SUM_KERNELS_HPP
#define PARSEATLAS_P4_FIELD_SUM_KERNELS_HPP

#include "secp256k1/field.hpp"
#include <array>
#include <cstddef>
#include <cstdint>

#if !defined(__SIZEOF_INT128__)
#error "P4 native wide-column experiment requires unsigned __int128"
#endif

namespace pa_p4 {
using FE = secp256k1::fast::FieldElement;

namespace detail {
using u128 = unsigned __int128;
using Limbs = FE::limbs_type;
using Columns = std::array<u128, 4>;
template<std::size_t Lanes> using Banks = std::array<Columns, Lanes>;
inline constexpr std::uint64_t K = UINT64_C(0x1000003d1);
struct Wide {
    Limbs low{};
    std::uint64_t high = 0;
};

// The phase helpers below have arithmetic preconditions, not input validation.
// add_raw/add_small accept arbitrary four-word values and return the exact
// 257-bit sum. Their high word is a single carry bit.
inline Wide add_raw(const Limbs& a, const Limbs& b) noexcept {
    Wide result;
    u128 carry = 0;
    for (unsigned i = 0; i < 4; ++i) {
        const u128 word = static_cast<u128>(a[i]) + b[i] + carry;
        result.low[i] = static_cast<std::uint64_t>(word);
        carry = word >> 64;
    }
    result.high = static_cast<std::uint64_t>(carry);
    return result;
}

inline Wide add_small(const Limbs& a, std::uint64_t small) noexcept {
    Wide result;
    u128 carry = small;
    for (unsigned i = 0; i < 4; ++i) {
        const u128 word = static_cast<u128>(a[i]) + carry;
        result.low[i] = static_cast<std::uint64_t>(word);
        carry = word >> 64;
    }
    result.high = static_cast<std::uint64_t>(carry);
    return result;
}

inline Limbs select_by_bit(const Limbs& yes, const Limbs& no,
                           std::uint64_t bit) noexcept {
    // Precondition: bit is exactly zero or one.
    const auto mask = std::uint64_t{0} - bit;
    Limbs result;
    for (unsigned i = 0; i < 4; ++i)
        result[i] = (yes[i] & mask) | (no[i] & ~mask);
    return result;
}

inline Limbs canonicalize_below_b(const Limbs& low) noexcept {
    // B=2^256, p=B-K. Since 0<=low<B<2p, adding K overflows iff
    // low>=p. The overflowing low result is exactly low-p.
    const auto correction = add_small(low, K);
    return select_by_bit(correction.low, low, correction.high);
}

inline Limbs canonicalize_sum_of_two(const Wide& sum) noexcept {
    // Stronger precondition: sum is a+b for canonical a,b, hence sum<2p.
    // If high=1, low<B-2K, so low+K<p and cannot overflow. If high=0,
    // the add-K carry detects low>=p. The two carry bits cannot both be one.
    // Select the corrected value in either case. No generic wide second fold
    // is needed for this eager control.
    const auto correction = add_small(sum.low, K);
    return select_by_bit(correction.low, sum.low, sum.high | correction.high);
}

inline void accumulate_columns(Columns& columns, const Limbs& rhs) noexcept {
    for (unsigned i = 0; i < 4; ++i) columns[i] += rhs[i];
}

template<std::size_t Lanes>
inline Banks<Lanes> accumulate_banks(const FE* rhs, std::size_t count) noexcept {
    static_assert(Lanes == 1 || Lanes == 4, "P4 admits one or four RHS banks");
    // Precondition: count<=4095 canonical RHS objects. Every bank starts
    // at zero; this phase does NOT insert the canonical prefix seed.
    Banks<Lanes> banks{};
    std::size_t position = 0;
    // Public fixed lanes expose independent column accumulators on one core.
    // Grouping changes scheduling only: RHS i belongs to bank i%Lanes.
    while (count - position >= Lanes) {
        for (std::size_t lane = 0; lane < Lanes; ++lane)
            accumulate_columns(banks[lane], rhs[position + lane].limbs());
        position += Lanes;
    }
    for (std::size_t lane = 0; position < count; ++position, ++lane)
        accumulate_columns(banks[lane], rhs[position].limbs());
    return banks;
}

template<std::size_t Lanes>
inline Columns merge_columns(const Limbs& canonical_seed,
                             const Banks<Lanes>& banks) noexcept {
    static_assert(Lanes == 1 || Lanes == 4, "P4 admits one or four RHS banks");
    Columns merged{};
    for (unsigned i = 0; i < 4; ++i) {
        merged[i] = canonical_seed[i];
        for (std::size_t lane = 0; lane < Lanes; ++lane)
            merged[i] += banks[lane][i];
    }
    return merged;
}

inline Wide propagate_columns(const Columns& columns) noexcept {
    // Up to M=4096 canonical terms: column<=M*(2^64-1).
    // Incoming carry<=M-1, so each addition<=M*2^64-1<2^128;
    // outgoing high<=4095. No inter-limb carry happened before this phase.
    Wide result;
    u128 carry = 0;
    for (unsigned i = 0; i < 4; ++i) {
        const u128 word = columns[i] + carry;
        result.low[i] = static_cast<std::uint64_t>(word);
        carry = word >> 64;
    }
    result.high = static_cast<std::uint64_t>(carry);
    return result;
}

inline Wide fold_high(const Wide& value) noexcept {
    // Precondition: high<=4095. high*K<2^45 fits a uint64_t.
    // This is the exact integer low+high*K, with its bit256 overflow kept.
    return add_small(value.low, value.high * K);
}

inline Limbs reduce_wide(const Wide& value) noexcept {
    const auto first = fold_high(value);
    const auto second = fold_high(first);
    // If first.high=1, first.low<value.high*K; adding K is then
    // <(value.high+1)*K<B. Therefore second.high is always zero.
    return canonicalize_below_b(second.low);
}

inline Limbs normalize_columns(const Columns& columns) noexcept {
    return reduce_wide(propagate_columns(columns));
}
} // namespace detail

// Final-only canonical Fp sum: x0 is counted once and only the final result
// is observable. x0 and all RHS are canonical FE64 values; for count>0, rhs
// denotes count readable objects. count=0 permits null and returns x0.
// Inputs are read-only, x0 may alias any RHS, and no restrict promise is made.
// Public loops/fixed template schedules do not constitute CT certification.
inline FE sum_fe64_inline(const FE& x0, const FE* rhs, std::size_t count) {
    if (count == 0) return x0;
    detail::Limbs accumulator = x0.limbs();
    for (std::size_t i = 0; i < count; ++i)
        accumulator = detail::canonicalize_sum_of_two(
            detail::add_raw(accumulator, rhs[i].limbs()));
    // No FE construction or production arithmetic call occurs per RHS.
    return FE::from_limbs_raw(accumulator);
}

template<std::size_t ChunkRhs, std::size_t Lanes>
inline FE sum_fe64_wide(const FE& x0, const FE* rhs, std::size_t count) {
    static_assert(ChunkRhs >= 1 && ChunkRhs <= 4095,
                  "P4 chunk admits 1..4095 RHS plus one canonical prefix");
    static_assert(Lanes == 1 || Lanes == 4, "P4 admits one or four RHS banks");
    if (count == 0) return x0;
    detail::Limbs accumulator = x0.limbs();
    std::size_t offset = 0;
    while (offset < count) {
        const auto remaining = count - offset;
        const auto take = remaining < ChunkRhs ? remaining : ChunkRhs;
        const auto banks = detail::accumulate_banks<Lanes>(rhs + offset, take);
        // Merge the current canonical prefix once, not x0 again or once/bank.
        accumulator = detail::normalize_columns(
            detail::merge_columns<Lanes>(accumulator, banks));
        offset += take; // take<=count-offset, so this cannot overflow size_t.
    }
    return FE::from_limbs_raw(accumulator);
}
} // namespace pa_p4
#endif // PARSEATLAS_P4_FIELD_SUM_KERNELS_HPP
