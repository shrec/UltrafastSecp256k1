#pragma once

#include <array>
#include <cstdint>

#include <secp256k1/detail/arith64.hpp>

// Research-only complete integer boundary, NOT an Fp/Fn modular addition.
// Limbs are least-significant first. Every compute specialization requires
// carry_in in {0,1}; callers validate at their external boundary, outside timing.
// Output satisfies A+B+carry_in = low + 2^256*carry_out, with carry_out in {0,1}.
// No buffers, pointers or aliasing contract: inputs and results are values.
namespace pa_l0_native {

using Word = std::uint64_t;

struct Input {
    std::array<Word, 4> a;
    std::array<Word, 4> b;
    Word carry_in;
};

struct Output {
    std::array<Word, 4> low;
    Word carry_out;
};

enum class Method { Original, GpMaterialized, GpRecomputed, Blocked2 };

inline bool valid_input(const Input& input) noexcept {
    return input.carry_in <= 1;
}

namespace detail {

constexpr Word word_max = ~Word{0};

inline Word raw_sum(const Input& input, unsigned limb) noexcept {
    return input.a[limb] + input.b[limb];
}

inline Word generate(const Input& input, unsigned limb) noexcept {
    return Word(raw_sum(input, limb) < input.a[limb]);
}

inline Word propagate(const Input& input, unsigned limb) noexcept {
    return Word(raw_sum(input, limb) == word_max);
}

} // namespace detail

template<Method M>
inline Output compute(const Input& input) noexcept {
    static_assert(M == Method::Original || M == Method::GpMaterialized ||
                  M == Method::GpRecomputed || M == Method::Blocked2,
                  "unsupported arithmetic method");

    if constexpr (M == Method::Original) {
        unsigned char carry = static_cast<unsigned char>(input.carry_in);
        Output output{};
        for (unsigned i = 0; i < 4; ++i) {
            output.low[i] = secp256k1::detail::add64(input.a[i], input.b[i], carry);
        }
        output.carry_out = carry;
        return output;
    } else if constexpr (M == Method::GpRecomputed) {
        // H01 recompute control: repeat source expressions instead of keeping
        // arrays of raw/g/p values. This is NOT a claim of extra executed ALU
        // operations or fewer registers: normal optimization may CSE everything
        // and emit exactly the same machine code as GpMaterialized.
        using detail::generate;
        using detail::propagate;
        using detail::raw_sum;
        const Word c0 = input.carry_in;
        const Word c1 = generate(input, 0) | (propagate(input, 0) & c0);
        const Word c2 = generate(input, 1) |
            (propagate(input, 1) & generate(input, 0)) |
            (propagate(input, 1) & propagate(input, 0) & c0);
        const Word c3 = generate(input, 2) |
            (propagate(input, 2) & generate(input, 1)) |
            (propagate(input, 2) & propagate(input, 1) & generate(input, 0)) |
            (propagate(input, 2) & propagate(input, 1) & propagate(input, 0) & c0);
        const Word c4 = generate(input, 3) |
            (propagate(input, 3) & generate(input, 2)) |
            (propagate(input, 3) & propagate(input, 2) & generate(input, 1)) |
            (propagate(input, 3) & propagate(input, 2) & propagate(input, 1) &
             generate(input, 0)) |
            (propagate(input, 3) & propagate(input, 2) & propagate(input, 1) &
             propagate(input, 0) & c0);
        return {{raw_sum(input, 0) + c0, raw_sum(input, 1) + c1,
                 raw_sum(input, 2) + c2, raw_sum(input, 3) + c3}, c4};
    } else {
        std::array<Word, 4> raw{}, g{}, p{};
        for (unsigned i = 0; i < 4; ++i) {
            raw[i] = input.a[i] + input.b[i];
            g[i] = Word(raw[i] < input.a[i]);
            p[i] = Word(raw[i] == detail::word_max);
        }

        // For B=2^64: a_i+b_i=B*g_i+raw_i. Unsigned wrapping is
        // intentional. The exact transfer is C_i(c)=g_i | (p_i & c).
        // Canonical g_i and p_i cannot both be one. Keeping raw_i is essential:
        // a carry summary alone cannot reconstruct the low result.
        const Word c0 = input.carry_in;
        const Word c1 = g[0] | (p[0] & c0);
        Word c2, c3, c4;
        if constexpr (M == Method::GpMaterialized) {
            // Known 4-limb carry-lookahead control, not a novelty claim.
            c2 = g[1] | (p[1] & g[0]) | (p[1] & p[0] & c0);
            c3 = g[2] | (p[2] & g[1]) | (p[2] & p[1] & g[0]) |
                 (p[2] & p[1] & p[0] & c0);
            c4 = g[3] | (p[3] & g[2]) | (p[3] & p[2] & g[1]) |
                 (p[3] & p[2] & p[1] & g[0]) |
                 (p[3] & p[2] & p[1] & p[0] & c0);
        } else {
            // H02 ordered 2+2 control. H after L composes as
            // (g_H | p_H*g_L, p_H*p_L); significance order is NOT swappable.
            // Each block exposes its transfer summary; internal c1/c3 and the
            // original residuals retain the complete four-limb result.
            const Word lower_g = g[1] | (p[1] & g[0]);
            const Word lower_p = p[1] & p[0];
            c2 = lower_g | (lower_p & c0);
            c3 = g[2] | (p[2] & c2);
            const Word upper_g = g[3] | (p[3] & g[2]);
            const Word upper_p = p[3] & p[2];
            c4 = upper_g | (upper_p & c2);
        }
        return {{raw[0] + c0, raw[1] + c1, raw[2] + c2, raw[3] + c3}, c4};
    }
}

} // namespace pa_l0_native
