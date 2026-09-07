// Small native P5 smoke: independent radix-2^32 products, bit-serial reduction,
// and canonical compositions against the frozen corrected FE64 API.
// This is not the separate comprehensive Boost oracle or a performance test.
#include "../probes/p5_field_product_kernels.hpp"
#include "secp256k1/field.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using pa_p5::Limbs;
using pa_p5::Wide;
using u128 = unsigned __int128;
using FE = secp256k1::fast::FieldElement;
constexpr Limbs prime{
    UINT64_C(0xfffffffefffffc2f), UINT64_MAX, UINT64_MAX, UINT64_MAX};
std::uint64_t checks = 0;
std::uint64_t checksum = UINT64_C(14695981039346656037);

void require(bool condition, const char* what) {
    if (!condition) throw std::runtime_error(what);
    ++checks;
}

template<std::size_t N>
void equal(const std::array<std::uint64_t, N>& actual,
           const std::array<std::uint64_t, N>& expected, const char* what) {
    require(actual == expected, what);
    for (const auto word : actual) for (unsigned shift = 0; shift < 64; shift += 8) {
        checksum ^= (word >> shift) & 255;
        checksum *= UINT64_C(1099511628211);
    }
}

Wide reference_product(const Limbs& a, const Limbs& b) {
    std::array<std::uint32_t, 8> left{}, right{};
    std::array<std::uint32_t, 16> product{};
    for (std::size_t i = 0; i < 8; ++i) {
        left[i] = std::uint32_t(a[i / 2] >> ((i % 2) * 32));
        right[i] = std::uint32_t(b[i / 2] >> ((i % 2) * 32));
    }
    for (std::size_t i = 0; i < 8; ++i) {
        std::uint64_t carry = 0;
        for (std::size_t j = 0; j < 8; ++j) {
            const auto t = std::uint64_t(left[i]) * right[j] + product[i + j] + carry;
            product[i + j] = std::uint32_t(t);
            carry = t >> 32;
        }
        product[i + 8] = std::uint32_t(carry);
    }
    Wide result{};
    for (std::size_t i = 0; i < 8; ++i)
        result[i] = std::uint64_t(product[2 * i]) |
                    (std::uint64_t(product[2 * i + 1]) << 32);
    return result;
}

Limbs reference_reduce(const Wide& input) {
    // Long division by p, one input bit at a time; no pseudo-Mersenne fold.
    Limbs remainder{};
    for (unsigned bit = 512; bit-- > 0;) {
        auto carry = (input[bit / 64] >> (bit % 64)) & 1;
        for (std::size_t i = 0; i < 4; ++i) {
            const auto next = remainder[i] >> 63;
            remainder[i] = (remainder[i] << 1) | carry;
            carry = next;
        }
        Limbs difference{};
        std::uint64_t borrow = 0;
        for (std::size_t i = 0; i < 4; ++i) {
            const u128 subtrahend = u128(prime[i]) + borrow;
            difference[i] = std::uint64_t(u128(remainder[i]) - subtrahend);
            borrow = std::uint64_t(u128(remainder[i]) < subtrahend);
        }
        // Previous remainder < p, hence shifted value < 2p.
        if (carry != 0 || borrow == 0) remainder = difference;
    }
    return remainder;
}

Limbs canonical_input(const Limbs& input) {
    return reference_reduce({input[0], input[1], input[2], input[3], 0, 0, 0, 0});
}

void check_field(const Limbs& actual, const FE& expected, const char* what) {
    const FE field = FE::from_limbs_raw(actual);
    const auto bytes = field.to_bytes();
    FE parsed;
    require(FE::parse_bytes_strict(bytes, parsed), "noncanonical result encoding");
    require(bytes == expected.to_bytes(), "FE64 full-byte comparison mismatch");
    equal(actual, expected.limbs(), what);
}

void check_pair(const Limbs& a, const Limbs& b) {
    const auto original_a = a, original_b = b;
    const auto product = reference_product(a, b);
    equal(pa_p5::mul_row(a, b), product, "row product");
    equal(pa_p5::mul_comba(a, b), product, "Comba product");
    const auto square = reference_product(a, a);
    equal(pa_p5::square_comba(a), square, "symmetric square");
    equal(pa_p5::mul_row(a, a), square, "row same-input alias");
    equal(pa_p5::mul_comba(a, a), square, "Comba same-input alias");
    const auto reduced = reference_reduce(product);
    equal(pa_p5::reduce_serial(product), reduced, "serial product reduction");
    equal(pa_p5::reduce_parallel(product), reduced, "parallel product reduction");

    const auto ca = canonical_input(a), cb = canonical_input(b);
    const FE fa = FE::from_limbs_raw(ca), fb = FE::from_limbs_raw(cb);
    const FE expected = fa * fb;
    const auto row = pa_p5::mul_row(ca, cb), comba = pa_p5::mul_comba(ca, cb);
    check_field(pa_p5::reduce_serial(row), expected, "row/serial FE64");
    check_field(pa_p5::reduce_parallel(row), expected, "row/parallel FE64");
    check_field(pa_p5::reduce_serial(comba), expected, "Comba/serial FE64");
    check_field(pa_p5::reduce_parallel(comba), expected, "Comba/parallel FE64");
    const auto canonical_square = pa_p5::square_comba(ca);
    const FE expected_square = fa.square();
    check_field(pa_p5::reduce_serial(canonical_square), expected_square, "square/serial FE64");
    check_field(pa_p5::reduce_parallel(canonical_square), expected_square, "square/parallel FE64");
    require(a == original_a && b == original_b, "multiply input mutated");
}

void check_wide(const Wide& input) {
    const auto original = input;
    const auto expected = reference_reduce(input);
    equal(pa_p5::reduce_serial(input), expected, "arbitrary512 serial");
    equal(pa_p5::reduce_parallel(input), expected, "arbitrary512 parallel");
    const auto serial = pa_p5::detail::first_fold_serial(input);
    const auto parallel = pa_p5::detail::first_fold_parallel(input);
    equal(serial.low, parallel.low, "first-fold low disagreement");
    require(serial.high == parallel.high && serial.high <= pa_p5::K,
            "first-fold q disagreement/bound");
    const auto second = pa_p5::detail::fold_high(serial);
    require(second.high <= 1, "second-fold overflow bound");
    const auto third = pa_p5::detail::fold_high(second);
    require(third.high == 0, "third-fold must not overflow");
    require(input == original, "reduction input mutated");
}

std::uint64_t random_word(std::uint64_t& state) {
    std::uint64_t z = (state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}
} // namespace

int main() {
    try {
        constexpr Limbs zero{}, one{1, 0, 0, 0};
        constexpr Limbs maximum{UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX};
        constexpr Limbs p_minus_one{
            UINT64_C(0xfffffffefffffc2e), UINT64_MAX, UINT64_MAX, UINT64_MAX};
        const std::array<Limbs, 10> fixtures{
            zero, one, maximum, prime, p_minus_one,
            Limbs{0, 0, 0, UINT64_C(0x8000000000000000)},
            Limbs{UINT64_MAX, 0, 0, 0},
            Limbs{0, UINT64_MAX, UINT64_MAX, 0},
            Limbs{UINT64_C(0xaaaaaaaaaaaaaaaa), UINT64_C(0x5555555555555555),
                  UINT64_C(0xaaaaaaaaaaaaaaaa), UINT64_C(0x5555555555555555)},
            Limbs{pa_p5::K, 1, 0, 0}};
        for (const auto& a : fixtures) for (const auto& b : fixtures) check_pair(a, b);

        Wide all_ones{};
        all_ones.fill(UINT64_MAX);
        check_wide({});
        check_wide(all_ones);
        const auto first = pa_p5::detail::first_fold_serial(all_ones);
        require(first.high == pa_p5::K, "all512 ones must realize q=K");
        const u128 qk = u128(first.high) * pa_p5::K;
        require((qk >> 64) != 0, "all512 ones must realize qK>uint64");
        const auto second = pa_p5::detail::fold_high(first);
        require(second.high == 1, "all512 ones must realize second-fold overflow");
        const u128 maximum_residue = u128(pa_p5::K) * pa_p5::K - 1;
        equal(pa_p5::reduce_serial(all_ones),
              Limbs{std::uint64_t(maximum_residue), std::uint64_t(maximum_residue >> 64), 0, 0},
              "all512 ones exact K^2-1 residue");
        equal(pa_p5::detail::canonicalize_below_b(prime), zero, "p correction");
        equal(pa_p5::detail::canonicalize_below_b(p_minus_one), p_minus_one, "p-1 identity");
        equal(pa_p5::detail::canonicalize_below_b(maximum),
              Limbs{pa_p5::K - 1, 0, 0, 0}, "B-1 correction");

        std::uint64_t state = 20260905;
        for (unsigned i = 0; i < 128; ++i) {
            Limbs a{}, b{};
            Wide input{};
            for (auto& word : a) word = random_word(state);
            for (auto& word : b) word = random_word(state);
            for (auto& word : input) word = random_word(state);
            check_pair(a, b);
            check_wide(input);
        }
        std::cout << "P5 smoke PASS checks=" << checks << " checksum=" << std::hex
                  << checksum << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "P5 smoke FAIL: " << error.what() << '\n';
        return 1;
    }
}
