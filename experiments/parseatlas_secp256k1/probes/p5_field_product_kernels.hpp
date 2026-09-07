#pragma once

// Experimental P5 exact products and secp256k1 field reductions.
// Inputs are little-endian radix-2^64 words, not necessarily canonical.
// All inputs are read-only; the two multiply operands may alias.
// Fixed public loop bounds are not a constant-time certification.
#include <array>
#include <cstddef>
#include <cstdint>

#if !defined(__SIZEOF_INT128__)
#error "P5 kernels require native unsigned __int128"
#endif

namespace pa_p5 {
using Limbs = std::array<std::uint64_t, 4>;
using Wide = std::array<std::uint64_t, 8>;
inline constexpr std::uint64_t K = UINT64_C(0x1000003d1);

namespace detail {
using u128 = unsigned __int128;
inline constexpr Limbs prime{
    UINT64_C(0xfffffffefffffc2f), UINT64_MAX, UINT64_MAX, UINT64_MAX};

// Exact column state. The incoming carry can occupy more than one word.
// At most four 128-bit products enter any multiply column; 192 bits suffice.
struct Accumulator192 {
    std::uint64_t low = 0;
    std::uint64_t middle = 0;
    std::uint64_t high = 0;
};

inline void add_product(Accumulator192& accumulator, u128 product) noexcept {
    const u128 lower = u128(accumulator.low) + std::uint64_t(product);
    accumulator.low = std::uint64_t(lower);
    const u128 upper = u128(accumulator.middle) + std::uint64_t(product >> 64) +
                       std::uint64_t(lower >> 64);
    accumulator.middle = std::uint64_t(upper);
    accumulator.high += std::uint64_t(upper >> 64);
}

inline std::uint64_t emit_column(Accumulator192& accumulator) noexcept {
    const auto result = accumulator.low;
    accumulator.low = accumulator.middle;
    accumulator.middle = accumulator.high;
    accumulator.high = 0;
    return result;
}

// Exact nonnegative integer low + high*2^256.
struct Fold {
    Limbs low{};
    std::uint64_t high = 0;
};
using HighProducts = std::array<u128, 4>;

inline Fold first_fold_serial(const Wide& input) noexcept {
    Fold result;
    std::uint64_t carry = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        // W=2^64: t <= W*(K+1)-1 < 2^97; carry <= K.
        const u128 t = u128(input[i + 4]) * K + input[i] + carry;
        result.low[i] = std::uint64_t(t);
        carry = std::uint64_t(t >> 64);
    }
    result.high = carry;
    return result;
}

inline HighProducts high_products(const Wide& input) noexcept {
    HighProducts products{};
    for (std::size_t i = 0; i < 4; ++i) products[i] = u128(input[i + 4]) * K;
    return products;
}

inline Fold combine_first_fold(const Wide& input, const HighProducts& products) noexcept {
    // products must equal high_products(input); exposed for phase inspection.
    Fold result;
    std::uint64_t carry = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        const u128 t = products[i] + input[i] + carry;
        result.low[i] = std::uint64_t(t);
        carry = std::uint64_t(t >> 64);
    }
    result.high = carry;
    return result;
}

inline Fold first_fold_parallel(const Wide& input) noexcept {
    // The four independent products precede the column carry chain in source.
    // This does not promise a particular compiler schedule or machine speed.
    const auto products = high_products(input);
    return combine_first_fold(input, products);
}

inline Fold add_u128(const Limbs& input, u128 term) noexcept {
    Fold result;
    u128 t = u128(input[0]) + std::uint64_t(term);
    result.low[0] = std::uint64_t(t);
    t = u128(input[1]) + std::uint64_t(term >> 64) + std::uint64_t(t >> 64);
    result.low[1] = std::uint64_t(t);
    for (std::size_t i = 2; i < 4; ++i) {
        t = u128(input[i]) + std::uint64_t(t >> 64);
        result.low[i] = std::uint64_t(t);
    }
    result.high = std::uint64_t(t >> 64);
    return result;
}

inline Fold fold_high(const Fold& input) noexcept {
    // First call: q <= K, so q*K <= K^2 < 2^65 (not necessarily 64 bits).
    // Splitting this u128 term keeps both words; no carry bit is discarded.
    return add_u128(input.low, u128(input.high) * K);
}

inline Limbs canonicalize_below_b(const Limbs& input) noexcept {
    // For every input < B=2^256 < 2p, one subtract-p/select is sufficient.
    Limbs difference{};
    std::uint64_t borrow = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        const u128 subtrahend = u128(prime[i]) + borrow;
        difference[i] = std::uint64_t(u128(input[i]) - subtrahend);
        borrow = std::uint64_t(u128(input[i]) < subtrahend);
    }
    const std::uint64_t mask = std::uint64_t(0) - (std::uint64_t(1) - borrow);
    Limbs result{};
    for (std::size_t i = 0; i < 4; ++i)
        result[i] = (difference[i] & mask) | (input[i] & ~mask);
    return result;
}

inline Limbs finish_reduction(const Fold& first) noexcept {
    const Fold second = fold_high(first);
    // second.high=e <= 1. If e=1, second.low < q*K, so its next fold
    // is < K^2+K < B and cannot overflow again. Exactly two folds, no loop.
    const Fold third = fold_high(second);
    return canonicalize_below_b(third.low);
}
} // namespace detail

inline Wide mul_row(const Limbs& a, const Limbs& b) noexcept {
    Wide result{};
    for (std::size_t i = 0; i < 4; ++i) {
        std::uint64_t carry = 0;
        for (std::size_t j = 0; j < 4; ++j) {
            // (W-1)^2 + (W-1) + (W-1) = W^2-1, so u128 is sufficient.
            const detail::u128 t = detail::u128(a[i]) * b[j] + result[i + j] + carry;
            result[i + j] = std::uint64_t(t);
            carry = std::uint64_t(t >> 64);
        }
        // Earlier rows reach at most word i+3: word i+4 is untouched.
        result[i + 4] = carry;
    }
    return result;
}

inline Wide mul_comba(const Limbs& a, const Limbs& b) noexcept {
    Wide result{};
    detail::Accumulator192 accumulator;
    for (std::size_t column = 0; column < 8; ++column) {
        for (std::size_t i = 0; i < 4; ++i) {
            if (column >= i && column - i < 4)
                detail::add_product(accumulator, detail::u128(a[i]) * b[column - i]);
        }
        result[column] = detail::emit_column(accumulator);
    }
    return result;
}

inline Wide square_comba(const Limbs& a) noexcept {
    Wide result{};
    detail::Accumulator192 accumulator;
    for (std::size_t column = 0; column < 8; ++column) {
        for (std::size_t i = 0; i < 4; ++i) {
            if (column >= i && column - i < 4 && i <= column - i) {
                const std::size_t j = column - i;
                // Four diagonals plus six cross terms = ten multiplications.
                const detail::u128 product = detail::u128(a[i]) * a[j];
                detail::add_product(accumulator, product);
                // A doubled 128-bit product needs 129 bits. Add twice into
                // the 192-bit state; do not truncate product << 1 to u128.
                if (i != j) detail::add_product(accumulator, product);
            }
        }
        result[column] = detail::emit_column(accumulator);
    }
    return result;
}

inline Limbs reduce_serial(const Wide& input) noexcept {
    return detail::finish_reduction(detail::first_fold_serial(input));
}

inline Limbs reduce_parallel(const Wide& input) noexcept {
    return detail::finish_reduction(detail::first_fold_parallel(input));
}
} // namespace pa_p5
