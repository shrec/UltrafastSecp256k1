#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "l0_native_kernels.hpp"

// Research-only full scalar-field addition modulo secp256k1's group order n.
// n is NOT the coordinate-field prime p. Limbs are least-significant first.
// Contract: a,b are canonical integers in [0,n); add returns (a+b) mod n in
// [0,n). These hot kernels do not validate inputs or normalize them first:
// callers must check canonical() at their external boundary, outside timing.
// All methods use the same production reduction and early-return scaffold;
// only the complete 256-bit addition (including its high carry) varies.
// This adopts the variable-time fast::Scalar addition scaffold. It is NOT a
// constant-time implementation or proof and must not replace ct::scalar_add
// merely because its modular answers agree.
namespace pa_modn {

using Word = std::uint64_t;
using Limbs = std::array<Word, 4>;

enum class Method { Original = 0, Blocked2 = 1, GpRecomputed = 2 };

inline constexpr Limbs order{
    0xBFD25E8CD0364141ULL,
    0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL,
    0xFFFFFFFFFFFFFFFFULL
};

namespace detail {

// Mirrors scalar.cpp::order_overflow at frozen production reference
// fef231d4e4173bd016fb2a3a1eff67087396a203. Valid for every 256-bit a.
inline bool order_overflow(const Limbs& a) noexcept {
    int yes = 0;
    int no = 0;
    no |= (a[3] < order[3]);
    no |= (a[2] < order[2]);
    yes |= (a[2] > order[2]) & ~no;
    no |= (a[1] < order[1]);
    yes |= (a[1] > order[1]) & ~no;
    yes |= (a[0] >= order[0]) & ~no;
    return yes != 0;
}

} // namespace detail

inline bool canonical(const Limbs& value) noexcept {
    return !detail::order_overflow(value);
}

template<Method M>
inline Limbs add(const Limbs& a, const Limbs& b) noexcept {
    static_assert(M == Method::Original || M == Method::Blocked2 ||
                  M == Method::GpRecomputed, "unsupported modular method");
    constexpr auto native_method = M == Method::Original
        ? pa_l0_native::Method::Original
        : (M == Method::Blocked2 ? pa_l0_native::Method::Blocked2
                                : pa_l0_native::Method::GpRecomputed);
    const auto raw = pa_l0_native::compute<native_method>({a, b, Word{0}});
    const Limbs& sum = raw.low;

    // Matches scalar.cpp::add_impl: do not discard the 257th bit, and do not
    // perform an unconditional subtraction when the production fast path
    // would return. Original delegates its four add64 calls to the unchanged
    // native reference specialization; every method shares everything below.
    if (!raw.carry_out && !detail::order_overflow(sum)) {
        return sum;
    }

    Limbs reduced{};
    unsigned char borrow = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        reduced[i] = secp256k1::detail::sub64(sum[i], order[i], borrow);
    }
    // Canonical inputs imply a+b < 2*n, so exactly one subtraction suffices.
    // If raw.carry_out is 1, this unsigned subtraction wraps back to a+b-n;
    // the final borrow is intentionally not applied as a second correction.
    return reduced;
}

template<Method M>
inline void add_assign(Limbs& a, const Limbs& b) noexcept {
    // Finish every input read before writing a, including the alias case x+=x.
    const Limbs result = add<M>(a, b);
    a = result;
}

inline std::array<std::uint8_t, 32> encode_le(const Limbs& value) noexcept {
    std::array<std::uint8_t, 32> bytes{};
    for (std::size_t i = 0; i < value.size(); ++i) {
        for (std::size_t j = 0; j < 8; ++j) {
            bytes[8 * i + j] = static_cast<std::uint8_t>(value[i] >> (8 * j));
        }
    }
    return bytes;
}

} // namespace pa_modn
