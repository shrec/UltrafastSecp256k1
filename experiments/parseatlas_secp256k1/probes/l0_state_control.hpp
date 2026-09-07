#pragma once

#include "l0_native_kernels.hpp"

// Research-only crossed source-representation control. No production call site.
namespace pa_l0_state {
using pa_l0_native::Input;
using pa_l0_native::Method;
using pa_l0_native::Word;
enum class Form { Aggregate, ScalarLocal };
constexpr Word max_operations = 1000000000;
constexpr Word max_passes = 1000000;

inline bool valid_region(const Input& state, Word count, Word passes) noexcept {
    return pa_l0_native::valid_input(state) && count <= max_operations &&
           passes <= max_passes && (passes == 0 || count <= max_operations / passes);
}

inline Word rotate13(Word value) noexcept { return (value << 13) | (value >> 51); }

template<Method M>
inline void aggregate_step(Input& state) noexcept {
    const auto output = pa_l0_native::compute<M>(state);
    state.a = output.low;
    state.carry_in = output.carry_out;
    for (unsigned limb = 0; limb < 4; ++limb)
        state.b[limb] = rotate13(state.b[limb]) ^ state.a[limb];
}

namespace detail {
inline Word raw(Word a, Word b) noexcept { return a + b; }
inline Word generate(Word a, Word b) noexcept { return Word(raw(a, b) < a); }
inline Word propagate(Word a, Word b) noexcept { return Word(raw(a, b) == ~Word{0}); }
} // namespace detail

// Every operand and output is a scalar local after inlining into advance.
// References express updates, not a pointer/overlapping-buffer public contract.
// There is no per-step Input/Output aggregate construction in this source path.
template<Method M>
inline void scalar_step(Word& a0, Word& a1, Word& a2, Word& a3,
                        Word& b0, Word& b1, Word& b2, Word& b3, Word& carry) noexcept {
    static_assert(M == Method::Original || M == Method::GpMaterialized ||
                  M == Method::GpRecomputed || M == Method::Blocked2,
                  "unsupported arithmetic method");
    Word s0, s1, s2, s3, c4;
    if constexpr (M == Method::Original) {
        unsigned char c = static_cast<unsigned char>(carry);
        s0 = secp256k1::detail::add64(a0, b0, c);
        s1 = secp256k1::detail::add64(a1, b1, c);
        s2 = secp256k1::detail::add64(a2, b2, c);
        s3 = secp256k1::detail::add64(a3, b3, c);
        c4 = c;
    } else if constexpr (M == Method::GpRecomputed) {
        using detail::generate;
        using detail::propagate;
        using detail::raw;
        const Word c0 = carry;
        const Word c1 = generate(a0,b0) | (propagate(a0,b0) & c0);
        const Word c2 = generate(a1,b1) | (propagate(a1,b1) & generate(a0,b0)) |
            (propagate(a1,b1) & propagate(a0,b0) & c0);
        const Word c3 = generate(a2,b2) | (propagate(a2,b2) & generate(a1,b1)) |
            (propagate(a2,b2) & propagate(a1,b1) & generate(a0,b0)) |
            (propagate(a2,b2) & propagate(a1,b1) & propagate(a0,b0) & c0);
        c4 = generate(a3,b3) | (propagate(a3,b3) & generate(a2,b2)) |
            (propagate(a3,b3) & propagate(a2,b2) & generate(a1,b1)) |
            (propagate(a3,b3) & propagate(a2,b2) & propagate(a1,b1) & generate(a0,b0)) |
            (propagate(a3,b3) & propagate(a2,b2) & propagate(a1,b1) &
             propagate(a0,b0) & c0);
        s0 = raw(a0,b0) + c0;
        s1 = raw(a1,b1) + c1;
        s2 = raw(a2,b2) + c2;
        s3 = raw(a3,b3) + c3;
    } else {
        const Word r0 = a0+b0, r1 = a1+b1, r2 = a2+b2, r3 = a3+b3;
        const Word g0 = Word(r0<a0), g1 = Word(r1<a1), g2 = Word(r2<a2), g3 = Word(r3<a3);
        const Word p0 = Word(r0==~Word{0}), p1 = Word(r1==~Word{0});
        const Word p2 = Word(r2==~Word{0}), p3 = Word(r3==~Word{0});
        const Word c0 = carry;
        const Word c1 = g0 | (p0 & c0);
        Word c2, c3;
        if constexpr (M == Method::GpMaterialized) {
            c2 = g1 | (p1 & g0) | (p1 & p0 & c0);
            c3 = g2 | (p2 & g1) | (p2 & p1 & g0) | (p2 & p1 & p0 & c0);
            c4 = g3 | (p3 & g2) | (p3 & p2 & g1) | (p3 & p2 & p1 & g0) |
                 (p3 & p2 & p1 & p0 & c0);
        } else {
            const Word lower_g = g1 | (p1 & g0), lower_p = p1 & p0;
            c2 = lower_g | (lower_p & c0);
            c3 = g2 | (p2 & c2);
            const Word upper_g = g3 | (p3 & g2), upper_p = p3 & p2;
            c4 = upper_g | (upper_p & c2);
        }
        s0 = r0+c0; s1 = r1+c1; s2 = r2+c2; s3 = r3+c3;
    }
    a0=s0; a1=s1; a2=s2; a3=s3; carry=c4;
    b0=rotate13(b0)^s0; b1=rotate13(b1)^s1;
    b2=rotate13(b2)^s2; b3=rotate13(b3)^s3;
}

// Precondition: valid_region. A zero count or zero passes is the identity.
// Both forms unpack once at the region boundary and write back the full state
// once. Source-level scalarization is not a promise about compiler allocation.
template<Method M, Form F>
inline void advance(Input& state, Word count, Word passes) noexcept {
    static_assert(F == Form::Aggregate || F == Form::ScalarLocal, "unsupported form");
    if constexpr (F == Form::Aggregate) {
        Input local = state;
        for (Word pass = 0; pass < passes; ++pass)
            for (Word step = 0; step < count; ++step) aggregate_step<M>(local);
        state = local;
    } else {
        Word a0=state.a[0], a1=state.a[1], a2=state.a[2], a3=state.a[3];
        Word b0=state.b[0], b1=state.b[1], b2=state.b[2], b3=state.b[3];
        Word carry=state.carry_in;
        for (Word pass = 0; pass < passes; ++pass)
            for (Word step = 0; step < count; ++step)
                scalar_step<M>(a0,a1,a2,a3,b0,b1,b2,b3,carry);
        state.a = {{a0,a1,a2,a3}};
        state.b = {{b0,b1,b2,b3}};
        state.carry_in = carry;
    }
}
} // namespace pa_l0_state
