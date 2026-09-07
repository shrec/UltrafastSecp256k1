#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "modn_add_control.hpp"

#if !defined(__SIZEOF_INT128__) || __SIZEOF_INT128__ != 16
#error "F2 requires GCC/Clang-compatible unsigned 128-bit integers"
#endif

// F2 research-only, final-state scalar-n reduction of ONE recurrence:
//     result = (x0 + sum(rhs[0..count))) mod n.
// The canonical final value is observable; intermediate modular states are
// not. All x0/rhs operands must be canonical in [0,n), an unchecked precondition.
// rhs must designate count readable Limbs when count>0. Pointer extent and
// lifetime are unchecked, while excessive count and a positive-count null
// pointer are rejected before reads. Inputs are never modified.
//
// Both representations retain the EXACT nonnegative integer before reduction:
//   Wide5:   sum(wide[i] * 2^(64*i), i=0..4).
//   Columns: sum(columns[i] * 2^(64*i), i=0..3).
// Write W=2^64, B=W^4=2^256, and K=count+1. The total is at most
// K*(B-1), with count<=10^9, so it is <2^286 and fits all five limbs.
// Each column is <=K*(W-1)<2^94. Normalization happens at the boundary;
// no canonical-only modular adder is applied to a noncanonical intermediate.
//
// This header requires the GCC/Clang unsigned __int128 extension. Defining
// SECP256K1_NO_INT128 changes the frozen add64 helper used by Wide5/reference
// and the reducer's optional second fold, but does NOT eliminate the new
// 128-bit columns or reducer products.
// No constant-time, novelty, universal-speedup or production-eligibility claim.
namespace pa_f2 {

using Word = pa_modn::Word;
using Limbs = pa_modn::Limbs;
using Wide5 = std::array<Word, 5>;
using UInt128 = unsigned __int128;
using Columns = std::array<UInt128, 4>;

inline constexpr std::size_t max_count = 1'000'000'000;
inline constexpr UInt128 max_column_sum =
    (static_cast<UInt128>(max_count) + 1) *
    static_cast<UInt128>(std::numeric_limits<Word>::max());

namespace detail {

// D=B-n = 0x14551231950b75fc4402da1732fc9bebf, so 0<D<2^129.
inline constexpr Limbs complement{
    0x402DA1732FC9BEBFULL,
    0x4551231950B75FC4ULL,
    0x0000000000000001ULL,
    0x0000000000000000ULL
};

inline void validate_input(const Limbs* rhs, std::size_t count) {
    // The addressability check matters on narrower size_t targets and never
    // multiplies count before checking. Buffer extent remains a precondition.
    if (count > max_count ||
        count > std::numeric_limits<std::size_t>::max() / sizeof(Limbs)) {
        throw std::length_error("F2 count exceeds the bounded input domain");
    }
    if (count != 0 && rhs == nullptr) {
        throw std::invalid_argument("F2 nonempty input requires rhs");
    }
}

} // namespace detail

// Exact five-limb sum. For every input, four frozen add64 calls propagate the
// low-256-bit carry and one high-word addition retains that carry. The high
// word cannot exceed count<=10^9, so that addition cannot wrap. No mod-n
// reductions, allocation, or worker threads occur here. count==0 encodes x0.
inline Wide5 accumulate_wide(const Limbs& x0, const Limbs* rhs,
                             std::size_t count) {
    detail::validate_input(rhs, count);
    Wide5 total{x0[0], x0[1], x0[2], x0[3], 0};
    for (std::size_t i = 0; i < count; ++i) {
        unsigned char carry = 0;
        for (std::size_t limb = 0; limb < 4; ++limb) {
            total[limb] = secp256k1::detail::add64(
                total[limb], rhs[i][limb], carry);
        }
        total[4] += static_cast<Word>(carry);
    }
    return total;
}

// Four independent limb-column sums of this same input stream. Initializing
// with x0 includes it exactly once. Each input costs four 128-bit additions;
// there is no cross-column carry propagation or mod-n reduction in the loop.
// Normal C++ optimization may reschedule/vectorize; no forced barriers here.
inline Columns accumulate_columns(const Limbs& x0, const Limbs* rhs,
                                   std::size_t count) {
    detail::validate_input(rhs, count);
    Columns total{static_cast<UInt128>(x0[0]), static_cast<UInt128>(x0[1]),
                  static_cast<UInt128>(x0[2]), static_cast<UInt128>(x0[3])};
    for (std::size_t i = 0; i < count; ++i) {
        total[0] += static_cast<UInt128>(rhs[i][0]);
        total[1] += static_cast<UInt128>(rhs[i][1]);
        total[2] += static_cast<UInt128>(rhs[i][2]);
        total[3] += static_cast<UInt128>(rhs[i][3]);
    }
    return total;
}

// Checked intermediate API: each column must be <=max_column_sum. This domain
// includes synthetic bounded column tuples, not only canonical-stream-reachable
// tuples. The returned Wide5 represents sum(columns[i]*W^i) exactly.
//
// For K=max_count+1, Ci<=K*(W-1). Inductively, incoming carry<=K-1 gives
// Ci+carry<=K*W-1<2^94, and the outgoing carry is <=K-1. Thus no 128-bit
// addition can wrap, and the final carry fits the fifth 64-bit limb. Reject all
// oversized columns BEFORE adding any carry; no silent high-bit truncation.
inline Wide5 normalize_columns(const Columns& columns) {
    for (const UInt128 column : columns) {
        if (column > max_column_sum) {
            throw std::invalid_argument("F2 column exceeds the bounded domain");
        }
    }
    Wide5 total{};
    UInt128 carry = 0;
    for (std::size_t limb = 0; limb < 4; ++limb) {
        const UInt128 current = columns[limb] + carry;
        total[limb] = static_cast<Word>(current);
        carry = current >> 64;
    }
    total[4] = static_cast<Word>(carry);
    return total;
}

// General reducer: valid for EVERY 320-bit input, not merely F2's <2^286 sum.
// Let T=L+H*B, 0<=L<B, 0<=H<W. Since B=n+D, T == L+H*D (mod n).
//
// H*D<2^193. S=L+H*D<B+2^193<2*B, so its first high carry is 0 or 1.
// If that carry is 1, R=S-B<=H*D-1, hence
//   R+D <= (H+1)*D-1 <= W*D-1 <2^193 < B.
// Therefore adding D for the first high carry CANNOT produce a second high
// carry. If the first carry is zero, S was already below B. The folded result
// is in [0,B) in both cases, so at most one conditional subtraction of n is
// needed. In particular, dropping the FIRST carry would be incorrect.
inline Limbs reduce320(const Wide5& total) noexcept {
    const Word high = total[4];

    // Exact H*D in four limbs. D has limbs [d0,d1,1,0]. Each of these
    // intermediate 128-bit products plus its incoming product carry fits:
    // d0<W, d1<W/2, and the final high+carry is <2*W.
    const UInt128 p0 = static_cast<UInt128>(high) * detail::complement[0];
    const UInt128 p1 = static_cast<UInt128>(high) * detail::complement[1] +
                       (p0 >> 64);
    const UInt128 p2 = static_cast<UInt128>(high) + (p1 >> 64);
    const Limbs product{static_cast<Word>(p0), static_cast<Word>(p1),
                        static_cast<Word>(p2), static_cast<Word>(p2 >> 64)};

    Limbs folded{};
    UInt128 carry = 0;
    for (std::size_t limb = 0; limb < 4; ++limb) {
        const UInt128 current = static_cast<UInt128>(total[limb]) +
                                product[limb] + carry;
        folded[limb] = static_cast<Word>(current);
        carry = current >> 64;
    }
    if (carry != 0) {
        // The proof above guarantees this complete second fold is <B. Its
        // final carry is exactly zero, not an omitted possible correction.
        unsigned char second_carry = 0;
        for (std::size_t limb = 0; limb < 4; ++limb) {
            folded[limb] = secp256k1::detail::add64(
                folded[limb], detail::complement[limb], second_carry);
        }
    }

    // canonical() is a full-256-bit comparison, not a canonical-input-only
    // arithmetic kernel. The subtraction is unsigned, with folded>=n here.
    if (pa_modn::canonical(folded)) {
        return folded;
    }
    Limbs reduced{};
    unsigned char borrow = 0;
    for (std::size_t limb = 0; limb < 4; ++limb) {
        reduced[limb] = secp256k1::detail::sub64(
            folded[limb], pa_modn::order[limb], borrow);
    }
    return reduced;
}

// Complete final-only jobs. No heap allocation or parallel workers. For N>0
// there is exactly one reduce320 call, including all required carry folds and
// canonicalization. N==0 returns x0 and does not inspect rhs. Canonical-input
// validation belongs outside timing; all structural/bound checks above remain
// inside the job and are part of its measured cost.
inline Limbs wide_final(const Limbs& x0, const Limbs* rhs, std::size_t count) {
    if (count == 0) {
        return x0;
    }
    return reduce320(accumulate_wide(x0, rhs, count));
}

inline Limbs columns_final(const Limbs& x0, const Limbs* rhs,
                           std::size_t count) {
    if (count == 0) {
        return x0;
    }
    return reduce320(normalize_columns(accumulate_columns(x0, rhs, count)));
}

} // namespace pa_f2
