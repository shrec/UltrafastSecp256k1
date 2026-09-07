// P5 finite native-C++ correctness corpus, not a benchmark or CT proof.
// Raw product/reducer domains are larger than the canonical FE64 API domain.
#include "../probes/p5_field_product_kernels.hpp"
#include "secp256k1/field.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using boost::multiprecision::cpp_int;
using FE = secp256k1::fast::FieldElement;
using Limbs = pa_p5::Limbs;
using Wide = pa_p5::Wide;
using Bytes = std::array<std::uint8_t, 32>;
using WideBytes = std::array<std::uint8_t, 64>;
using U128 = unsigned __int128;
const cpp_int kB = cpp_int(1) << 256;
const cpp_int kB2 = cpp_int(1) << 512;
const cpp_int kK = (cpp_int(1) << 32) + 977;
const cpp_int kP = kB - kK;
constexpr std::uint64_t kSeed = UINT64_C(0x5041355f50524f44);
constexpr std::size_t kRandomRawPairs = 10000;
constexpr std::size_t kRandomCanonicalPairs = 10000;
constexpr std::size_t kRandomWide = 10000;

struct Counts {
    std::uint64_t assertions = 0, raw_pairs = 0, canonical_pairs = 0, kats = 0;
    std::uint64_t direct_wide_cases = 0, product_checks = 0, residue_checks = 0;
    std::uint64_t reference_checks = 0, preservation_checks = 0, same_object_products = 0;
    std::uint64_t first_fold_checks = 0, high_product_checks = 0;
    std::uint64_t q_equals_k_cases = 0, qk_over_u64_cases = 0;
    std::uint64_t second_fold_overflow_cases = 0, final_p_correction_cases = 0;
    std::uint64_t second_fold_checks = 0, third_fold_checks = 0;
    std::uint64_t accumulator192_checks = 0, accumulator_highword_cases = 0;
    std::uint64_t chain_cases = 0, chain_steps = 0, negative_controls = 0;
    std::uint64_t checksum = UINT64_C(14695981039346656037);
    std::set<std::pair<Limbs, Limbs>> distinct_raw_pairs;
    std::set<std::pair<Limbs, Limbs>> distinct_canonical_pairs;
    std::set<Wide> distinct_wide_inputs;
};

void require(bool condition, Counts& c, const std::string& context) {
    ++c.assertions;
    if (!condition) throw std::runtime_error(context);
}

cpp_int mod(cpp_int value) {
    value %= kP;
    if (value < 0) value += kP;
    return value;
}

template <std::size_t N>
std::array<std::uint64_t, N> encode_words(cpp_int value) {
    if (value < 0 || value >= (cpp_int(1) << (64 * N))) throw std::runtime_error("oracle word range");
    const cpp_int mask = (cpp_int(1) << 64) - 1;
    std::array<std::uint64_t, N> result{};
    for (auto& word : result) {
        word = static_cast<std::uint64_t>(value & mask);
        value >>= 64;
    }
    return result;
}

template <std::size_t N>
cpp_int decode_words(const std::array<std::uint64_t, N>& words) {
    cpp_int value = 0;
    for (std::size_t i = N; i != 0; --i) {
        value *= cpp_int(1) << 64;
        value += words[i - 1];
    }
    return value;
}

template <std::size_t N>
std::array<std::uint8_t, N> encode_bytes(cpp_int value) {
    if (value < 0 || value >= (cpp_int(1) << (8 * N))) throw std::runtime_error("oracle byte range");
    std::array<std::uint8_t, N> result{};
    for (std::size_t i = N; i != 0; --i) {
        result[i - 1] = static_cast<std::uint8_t>(value & 255);
        value >>= 8;
    }
    return result;
}

template <std::size_t N>
std::array<std::uint8_t, N * 8> serialize_words(const std::array<std::uint64_t, N>& words) {
    std::array<std::uint8_t, N * 8> result{};
    for (std::size_t limb = 0; limb < N; ++limb)
        for (std::size_t byte = 0; byte < 8; ++byte)
            result[result.size() - 1 - (limb * 8 + byte)] = static_cast<std::uint8_t>(words[limb] >> (8 * byte));
    return result;
}

cpp_int decode_u128(U128 value) {
    return (cpp_int(static_cast<std::uint64_t>(value >> 64)) << 64) + static_cast<std::uint64_t>(value);
}

cpp_int decode_fold(const pa_p5::detail::Fold& value) {
    return decode_words(value.low) + cpp_int(value.high) * kB;
}

cpp_int decode_accumulator(const pa_p5::detail::Accumulator192& value) {
    return cpp_int(value.low) + (cpp_int(value.middle) << 64) + (cpp_int(value.high) << 128);
}

template <std::size_t N>
std::string hex_bytes(const std::array<std::uint8_t, N>& bytes) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (auto byte : bytes) out << std::setw(2) << static_cast<unsigned>(byte);
    return out.str();
}

template <std::size_t N>
void mix(const std::array<std::uint8_t, N>& bytes, Counts& c) {
    for (auto byte : bytes) c.checksum = (c.checksum ^ byte) * UINT64_C(1099511628211);
}

bool wide_matches(const Wide& words, const WideBytes& bytes, const cpp_int& expected) {
    const auto ew = encode_words<8>(expected);
    const auto eb = encode_bytes<64>(expected);
    bool result = true;
    for (std::size_t i = 0; i < 8; ++i) result &= words[i] == ew[i];
    for (std::size_t i = 0; i < 64; ++i) result &= bytes[i] == eb[i];
    return result;
}

bool residue_matches(const Limbs& words, const Bytes& bytes, const cpp_int& expected) {
    const auto ew = encode_words<4>(expected);
    const auto eb = encode_bytes<32>(expected);
    bool result = expected >= 0 && expected < kP && decode_words(words) < kP;
    for (std::size_t i = 0; i < 4; ++i) result &= words[i] == ew[i];
    for (std::size_t i = 0; i < 32; ++i) result &= bytes[i] == eb[i];
    return result;
}

void check_wide(const Wide& actual, const cpp_int& expected, Counts& c, const std::string& context) {
    const auto bytes = serialize_words(actual);
    if (!wide_matches(actual, bytes, expected))
        require(false, c, context + " actual512_be=" + hex_bytes(bytes) +
                " expected512_be=" + hex_bytes(encode_bytes<64>(expected)));
    ++c.product_checks;
    mix(bytes, c);
}

void check_residue(const Limbs& actual, const cpp_int& expected, Counts& c, const std::string& context) {
    const Limbs raw = actual;
    // from_limbs_raw does not normalize: raw canonicality is checked before
    // serialized equality can conceal a noncanonical representation.
    const FE value = FE::from_limbs_raw(raw);
    const Bytes bytes = value.to_bytes();
    if (!residue_matches(raw, bytes, expected))
        require(false, c, context + " actual_raw_be=" + hex_bytes(serialize_words(raw)) +
                " actual_serialized_be=" + hex_bytes(bytes) +
                " expected_be=" + hex_bytes(encode_bytes<32>(expected)));
    require(value.limbs() == raw && actual == raw, c, context + ": serialization mutation");
    ++c.residue_checks;
    mix(bytes, c);
}

void check_reference(const FE& actual, const cpp_int& expected, Counts& c, const std::string& context) {
    const auto raw = actual.limbs();
    const auto bytes = actual.to_bytes();
    require(residue_matches(raw, bytes, expected), c, context + ": actual FE64 oracle mismatch");
    require(actual.limbs() == raw, c, context + ": actual FE64 serializer mutation");
    ++c.reference_checks;
    mix(bytes, c);
}

void check_reducers(const Wide& input, const cpp_int& expected, Counts& c,
                    const std::string& context, bool serial, bool parallel) {
    const Wide before = input;
    if (serial) check_residue(pa_p5::reduce_serial(input), expected, c, context + " serial");
    if (parallel) check_residue(pa_p5::reduce_parallel(input), expected, c, context + " parallel");
    require(input == before, c, context + ": reducer input changed");
    ++c.preservation_checks;
}

void pair_case(const cpp_int& ai, const cpp_int& bi, bool canonical, Counts& c,
               const std::string& label) {
    const Limbs a = encode_words<4>(ai), b = encode_words<4>(bi);
    const Limbs before_a = a, before_b = b;
    const std::string context = label + " a=" + hex_bytes(serialize_words(a)) + " b=" + hex_bytes(serialize_words(b));
    const cpp_int product = ai * bi, square = ai * ai;
    const Wide row = pa_p5::mul_row(a, b);
    const Wide comba = pa_p5::mul_comba(a, b);
    const Wide square_row = pa_p5::mul_row(a, a);
    const Wide square_general = pa_p5::mul_comba(a, a);
    const Wide square_symmetric = pa_p5::square_comba(a);
    check_wide(row, product, c, context + " row product");
    check_wide(comba, product, c, context + " Comba product");
    check_wide(square_row, square, c, context + " row same-object square");
    check_wide(square_general, square, c, context + " Comba same-object square");
    check_wide(square_symmetric, square, c, context + " symmetric square");
    c.same_object_products += 2;
    const cpp_int product_mod = mod(product), square_mod = mod(square);
    check_reducers(row, product_mod, c, context + " row product", true, false);
    check_reducers(comba, product_mod, c, context + " Comba product", true, true);
    check_reducers(square_row, square_mod, c, context + " row square", true, false);
    check_reducers(square_general, square_mod, c, context + " general Comba square", true, true);
    check_reducers(square_symmetric, square_mod, c, context + " symmetric square", true, true);
    if (canonical) {
        require(ai >= 0 && ai < kP && bi >= 0 && bi < kP, c, context + ": canonical reference precondition");
        const FE fa = FE::from_bytes(encode_bytes<32>(ai));
        const FE fb = FE::from_bytes(encode_bytes<32>(bi));
        const Limbs fa_before = fa.limbs(), fb_before = fb.limbs();
        check_reference(fa, ai, c, context + " canonical lhs construction");
        check_reference(fb, bi, c, context + " canonical rhs construction");
        check_reference(fa * fb, product_mod, c, context + " FE64 multiply");
        check_reference(fa.square(), square_mod, c, context + " FE64 square");
        check_reference(fa * fa, square_mod, c, context + " FE64 same-object multiply");
        require(fa.limbs() == fa_before && fb.limbs() == fb_before, c, context + ": FE64 inputs changed");
        ++c.preservation_checks;
        ++c.canonical_pairs;
        c.distinct_canonical_pairs.emplace(a, b);
    } else {
        ++c.raw_pairs;
        c.distinct_raw_pairs.emplace(a, b);
    }
    require(a == before_a && b == before_b, c, context + ": raw product inputs changed");
    ++c.preservation_checks;
}

void wide_case(const cpp_int& input_int, Counts& c, const std::string& label) {
    const Wide input = encode_words<8>(input_int);
    const Wide before = input;
    const cpp_int expected = mod(input_int);
    check_reducers(input, expected, c, label, true, true);
    const cpp_int low = input_int % kB, high = input_int / kB;
    const cpp_int first_expected = low + kK * high;
    const cpp_int q = first_expected / kB;
    require(q <= kK, c, label + ": independent first-fold q bound");
    const auto serial = pa_p5::detail::first_fold_serial(input);
    const auto products = pa_p5::detail::high_products(input);
    for (std::size_t i = 0; i < 4; ++i) {
        require(decode_u128(products[i]) == cpp_int(input[i + 4]) * kK, c, label + ": exact high*K product");
        ++c.high_product_checks;
    }
    const auto combined = pa_p5::detail::combine_first_fold(input, products);
    const auto parallel = pa_p5::detail::first_fold_parallel(input);
    require(decode_fold(serial) == first_expected && decode_fold(combined) == first_expected &&
            decode_fold(parallel) == first_expected, c, label + ": exact first-fold helpers");
    ++c.first_fold_checks;
    if (q == kK) ++c.q_equals_k_cases;
    const cpp_int qk = q * kK;
    if (qk >= (cpp_int(1) << 64)) ++c.qk_over_u64_cases;
    const cpp_int second_expected = first_expected % kB + qk;
    const U128 qk_native = static_cast<U128>(serial.high) * static_cast<std::uint64_t>(UINT64_C(0x1000003d1));
    require(decode_u128(qk_native) == qk, c, label + ": full q*K product retained");
    const auto second_via_add = pa_p5::detail::add_u128(serial.low, qk_native);
    const auto second = pa_p5::detail::fold_high(serial);
    require(decode_fold(second) == second_expected && decode_fold(second_via_add) == second_expected &&
            second.high <= 1, c, label + ": exact second fold");
    ++c.second_fold_checks;
    if (second.high != 0) ++c.second_fold_overflow_cases;
    const cpp_int third_expected = second_expected % kB + (second_expected / kB) * kK;
    const auto third = pa_p5::detail::fold_high(second);
    require(decode_fold(third) == third_expected && third.high == 0, c, label + ": exact final carry fold");
    ++c.third_fold_checks;
    if (third_expected >= kP) ++c.final_p_correction_cases;
    check_residue(pa_p5::detail::canonicalize_below_b(third.low), expected, c, label + " phase final p correction");
    require(input == before, c, label + ": phase helpers changed raw input");
    ++c.preservation_checks;
    ++c.direct_wide_cases;
    c.distinct_wide_inputs.insert(input);
}

void accumulator_cases(Counts& c) {
    const U128 product = static_cast<U128>(UINT64_MAX) * UINT64_MAX;
    const cpp_int product_expected = cpp_int(UINT64_MAX) * UINT64_MAX;
    require(decode_u128(product) == product_expected, c, "max 128-bit product construction");
    pa_p5::detail::Accumulator192 accumulator{};
    cpp_int exact = 0;
    for (unsigned count = 1; count <= 4; ++count) {
        pa_p5::detail::add_product(accumulator, product);
        exact += product_expected;
        require(decode_accumulator(accumulator) == exact, c, "192-bit accumulator after max product " + std::to_string(count));
        if (accumulator.high != 0) ++c.accumulator_highword_cases;
        auto emitting = accumulator;
        const auto emitted = pa_p5::detail::emit_column(emitting);
        require(cpp_int(emitted) == exact % (cpp_int(1) << 64) &&
                decode_accumulator(emitting) == exact / (cpp_int(1) << 64), c, "192-bit emitted column/carry");
        ++c.accumulator192_checks;
    }
    // Propagate carry across both low and middle words, still well below the
    // 192-bit storage bound. This is observation of the actual add helper.
    pa_p5::detail::Accumulator192 cascade{UINT64_MAX, UINT64_MAX, 0};
    pa_p5::detail::add_product(cascade, U128(1));
    require(decode_accumulator(cascade) == (cpp_int(1) << 128), c, "192-bit two-word carry cascade");
    ++c.accumulator192_checks;
    ++c.accumulator_highword_cases;
    const Limbs all_max{{UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX}};
    const Wide expected_max{{1, 0, 0, 0, UINT64_MAX - 1, UINT64_MAX, UINT64_MAX, UINT64_MAX}};
    require(encode_words<8>((kB - 1) * (kB - 1)) == expected_max, c, "all-max full product fixed oracle");
    check_wide(pa_p5::mul_row(all_max, all_max), decode_words(expected_max), c, "all-max row KAT");
    check_wide(pa_p5::mul_comba(all_max, all_max), decode_words(expected_max), c, "all-max 130-bit column KAT");
    check_wide(pa_p5::square_comba(all_max), decode_words(expected_max), c, "all-max 129-bit cross-term KAT");
}

struct SplitMix64 {
    std::uint64_t state;
    std::uint64_t next() {
        std::uint64_t z = (state += UINT64_C(0x9e3779b97f4a7c15));
        z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
        return z ^ (z >> 31);
    }
    template <std::size_t N> cpp_int integer() {
        cpp_int result = 0;
        for (std::size_t i = 0; i < N; ++i) result += cpp_int(next()) << (64 * i);
        return result;
    }
};

std::vector<cpp_int> raw_boundaries() {
    std::vector<cpp_int> values{cpp_int(0), cpp_int(1), cpp_int(2), kP - 2, kP - 1,
                               kP, kP + 1, kB - 2, kB - 1};
    for (unsigned bit = 0; bit < 256; ++bit) {
        const cpp_int power = cpp_int(1) << bit;
        values.push_back(power - 1); values.push_back(power); values.push_back(power + 1);
        values.push_back(kB - 1 - power);
    }
    return values;
}

void explicit_kats(Counts& c) {
    // Fixed values read from audit/test_regression_field_reduce_carry.cpp;
    // each is also independently derived by Boost, not imported as the sole oracle.
    const cpp_int a1 = kB - (cpp_int(1) << 33) - 1;
    require(mod(a1 * a1) == cpp_int(UINT64_C(0xfffff860000e8900)), c, "P1 second-fold KAT independent truth");
    pair_case(a1, a1, true, c, "P1 second-fold carry KAT");
    ++c.kats;
    const cpp_int a2 = (cpp_int(1) << 255) + 1;
    const Limbs expected2{{UINT64_C(0xfffffffefffffc2e), UINT64_MAX, UINT64_MAX, UINT64_C(0x7fffffffffffffff)}};
    require(mod(a2 * (kP - 1)) == decode_words(expected2), c, "P1 first-fold KAT independent truth");
    pair_case(a2, kP - 1, true, c, "P1 first-fold carry cascade KAT");
    ++c.kats;
    const cpp_int a3 = (cpp_int(1) << 255) - 1;
    const cpp_int expected3("0x400000000000000000000000000000000000000000000000400001e740039f64");
    require(mod(a3 * a3) == expected3, c, "older large-square KAT independent truth");
    pair_case(a3, a3, true, c, "older large-square carry KAT");
    ++c.kats;
    require(mod(kB2 - 1) == kK * kK - 1, c, "all-wide-ones fixed reducer truth");
    wide_case(kB2 - 1, c, "all wide ones: q=K, qK>64, second-fold overflow");
    require(mod((kB - 1) * kB + kK) == kK * kK, c, "q=K no-carry fixed reducer truth");
    wide_case((kB - 1) * kB + kK, c, "q=K with 65-bit qK and no second-fold overflow");
    wide_case(kP, c, "low-only p requires canonical correction");
    wide_case(kP * kB, c, "high-only p requires canonical correction");
}

void chains(Counts& c) {
    SplitMix64 random{kSeed ^ UINT64_C(0x434841494e535035)};
    for (unsigned kind = 0; kind < 2; ++kind) {
        for (unsigned chain = 0; chain < 16; ++chain) {
            cpp_int value = mod(random.integer<4>());
            std::array<Limbs, 4> states{};
            for (auto& state : states) state = encode_words<4>(value);
            FE reference = FE::from_bytes(encode_bytes<32>(value));
            for (unsigned step = 0; step < 32; ++step) {
                const cpp_int rhs_int = mod(random.integer<4>());
                const Limbs rhs = encode_words<4>(rhs_int), rhs_before = rhs;
                const FE rhs_fe = FE::from_bytes(encode_bytes<32>(rhs_int));
                const std::string at = std::string(kind == 0 ? "mul" : "square") +
                    " chain" + std::to_string(chain) + " step" + std::to_string(step);
                if (kind == 0) {
                    value = mod(value * rhs_int);
                    states[0] = pa_p5::reduce_serial(pa_p5::mul_row(states[0], rhs));
                    states[1] = pa_p5::reduce_serial(pa_p5::mul_comba(states[1], rhs));
                    states[2] = pa_p5::reduce_parallel(pa_p5::mul_comba(states[2], rhs));
                    reference = reference * rhs_fe;
                } else {
                    value = mod(value * value);
                    states[0] = pa_p5::reduce_serial(pa_p5::mul_row(states[0], states[0]));
                    states[1] = pa_p5::reduce_serial(pa_p5::mul_comba(states[1], states[1]));
                    states[2] = pa_p5::reduce_serial(pa_p5::square_comba(states[2]));
                    states[3] = pa_p5::reduce_parallel(pa_p5::square_comba(states[3]));
                    reference = reference.square();
                }
                const unsigned active = kind == 0 ? 3 : 4;
                for (unsigned i = 0; i < active; ++i)
                    check_residue(states[i], value, c, at + " route" + std::to_string(i));
                check_reference(reference, value, c, at + " actual FE64 reference");
                require(rhs == rhs_before, c, at + ": chain RHS changed");
                ++c.preservation_checks;
                ++c.chain_steps;
            }
            ++c.chain_cases;
        }
    }
}

void controls(Counts& c) {
    Bytes bytes{};
    for (std::size_t i = 0; i < 32; ++i) bytes[i] = static_cast<std::uint8_t>(i + 1);
    const cpp_int expected("0x0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20");
    const Limbs limbs{{UINT64_C(0x191a1b1c1d1e1f20), UINT64_C(0x1112131415161718),
                       UINT64_C(0x090a0b0c0d0e0f10), UINT64_C(0x0102030405060708)}};
    require(residue_matches(limbs, bytes, expected), c, "known-endian positive control");
    for (std::size_t i = 0; i < 32; ++i) {
        auto bad = bytes; bad[i] ^= UINT8_C(0x80);
        require(!residue_matches(limbs, bad, expected), c, "missed result byte corruption");
        ++c.negative_controls;
    }
    for (std::size_t i = 0; i < 4; ++i) {
        auto bad = limbs; bad[i] ^= UINT64_C(0x8000000000000000);
        require(!residue_matches(bad, bytes, expected), c, "missed result raw-limb corruption");
        ++c.negative_controls;
    }
    require(!residue_matches(encode_words<4>(kP), encode_bytes<32>(kP), kP), c, "noncanonical p accepted");
    ++c.negative_controls;
    const Wide wide = encode_words<8>(expected);
    const WideBytes wide_bytes = encode_bytes<64>(expected);
    for (std::size_t i = 0; i < 8; ++i) {
        auto bad = wide; bad[i] ^= UINT64_C(0x8000000000000000);
        require(!wide_matches(bad, wide_bytes, expected), c, "missed 512-bit limb corruption");
        ++c.negative_controls;
    }
    for (std::size_t i = 0; i < 64; ++i) {
        auto bad = wide_bytes; bad[i] ^= UINT8_C(0x80);
        require(!wide_matches(wide, bad, expected), c, "missed 512-bit byte corruption");
        ++c.negative_controls;
    }
    pa_p5::detail::Accumulator192 bad_acc{}; bad_acc.high = 1;
    require(decode_accumulator(bad_acc) == (cpp_int(1) << 128), c, "192-bit observer keeps high word");
    ++c.negative_controls;
}

Counts run() {
    Counts c;
    controls(c);
    accumulator_cases(c);
    explicit_kats(c);
    const auto boundaries = raw_boundaries();
    for (std::size_t i = 0; i < boundaries.size(); ++i) {
        const cpp_int a = boundaries[i], next = boundaries[(i + 1) % boundaries.size()];
        const std::array<cpp_int, 4> raw_partners{{cpp_int(0), kB - 1, a, next}};
        for (std::size_t j = 0; j < raw_partners.size(); ++j)
            pair_case(a, raw_partners[j], false, c, "raw boundary " + std::to_string(i) + "/" + std::to_string(j));
        const std::array<cpp_int, 3> canonical_partners{{kP - 1, mod(a), mod(next)}};
        for (std::size_t j = 0; j < canonical_partners.size(); ++j)
            pair_case(mod(a), canonical_partners[j], true, c, "canonical boundary " + std::to_string(i) + "/" + std::to_string(j));
    }
    SplitMix64 random{kSeed};
    for (std::size_t i = 0; i < kRandomRawPairs; ++i) {
        const cpp_int a = random.integer<4>(), b = random.integer<4>();
        pair_case(a, b, false, c, "arbitrary raw pair " + std::to_string(i));
    }
    for (std::size_t i = 0; i < kRandomCanonicalPairs; ++i) {
        const cpp_int a = mod(random.integer<4>()), b = mod(random.integer<4>());
        pair_case(a, b, true, c, "canonical random pair " + std::to_string(i));
    }
    std::vector<cpp_int> wide_boundaries{cpp_int(0), cpp_int(1), kP - 1, kP, kP + 1,
        kB - 1, kB, kB + 1, kP * kP - 1, kP * kP, kP * kP + 1,
        kB2 - 2, kB2 - 1, (kB - 1) * kB, (kB - 1) * kB + kK - 1,
        (kB - 1) * kB + kK, (kB - 1) * kB + kK + 1};
    const Wide alternating{{UINT64_MAX, 0, UINT64_MAX, 0, UINT64_MAX, 0, UINT64_MAX, 0}};
    const Wide alternating_inverse{{0, UINT64_MAX, 0, UINT64_MAX, 0, UINT64_MAX, 0, UINT64_MAX}};
    wide_boundaries.push_back(decode_words(alternating));
    wide_boundaries.push_back(decode_words(alternating_inverse));
    for (unsigned bit = 0; bit < 512; ++bit) {
        const cpp_int power = cpp_int(1) << bit;
        wide_boundaries.push_back(power - 1); wide_boundaries.push_back(power);
        wide_boundaries.push_back(power + 1); wide_boundaries.push_back(kB2 - 1 - power);
    }
    for (std::size_t i = 0; i < wide_boundaries.size(); ++i)
        wide_case(wide_boundaries[i], c, "arbitrary wide boundary " + std::to_string(i));
    for (std::size_t i = 0; i < kRandomWide; ++i)
        wide_case(random.integer<8>(), c, "arbitrary random wide " + std::to_string(i));
    chains(c);
    require(boundaries.size() == 1033 && c.raw_pairs == 14132 && c.canonical_pairs == 13102, c, "pair fixture counts");
    require(c.direct_wide_cases == 12071 && c.kats == 3, c, "wide/KAT fixture counts");
    require(c.chain_cases == 32 && c.chain_steps == 1024, c, "chain counts");
    require(c.q_equals_k_cases != 0 && c.qk_over_u64_cases != 0 &&
            c.second_fold_overflow_cases != 0 && c.final_p_correction_cases != 0, c, "required reducer branches exercised");
    require(c.accumulator192_checks == 5 && c.accumulator_highword_cases == 4, c, "192-bit accumulator fixture counts");
    require(c.negative_controls == 110, c, "negative control counts");
    return c;
}

std::string hex64(std::uint64_t value) {
    std::ostringstream out; out << std::hex << std::setfill('0') << std::setw(16) << value; return out.str();
}

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (const unsigned char ch : value) {
        if (ch == '"' || ch == '\\') out << '\\' << static_cast<char>(ch);
        else if (ch < 0x20) out << "\\u" << std::hex << std::setfill('0') << std::setw(4) << static_cast<unsigned>(ch) << std::dec;
        else out << static_cast<char>(ch);
    }
    return out.str();
}
} // namespace

int main(int argc, char**) {
    if (argc != 1) { std::cerr << "Usage: test_p5_field_product (JSON stdout)\n"; return 2; }
    try {
        const Counts c = run();
        std::cout << "{\"schema\":\"parseatlas_p5_field_product_correctness_v1\",\"status\":\"pass\",\"mismatches\":0,"
                  << "\"finite_corpus_only\":true,\"timing_claim\":false,\"constant_time_claim\":false,"
                  << "\"oracle\":\"Boost exact 256x256 products and arbitrary512 modulo p; corrected FE64 on canonical pairs\","
                  << "\"output_checks\":\"full 8 raw product limbs and 64 BE bytes; canonical 4 raw residue limbs and all32 BE bytes\","
                  << "\"fixture_entries_may_repeat\":true,\"distinct_counts_exclude_derived_chain_states\":true,"
                  << "\"seed_hex\":\"" << hex64(kSeed) << "\",\"raw_pair_entries\":" << c.raw_pairs
                  << ",\"distinct_raw_pairs\":" << c.distinct_raw_pairs.size()
                  << ",\"canonical_pair_entries\":" << c.canonical_pairs
                  << ",\"distinct_canonical_pairs\":" << c.distinct_canonical_pairs.size()
                  << ",\"direct_wide_entries\":" << c.direct_wide_cases
                  << ",\"distinct_direct_wide_inputs\":" << c.distinct_wide_inputs.size()
                  << ",\"named_carry_KATs\":" << c.kats
                  << ",\"product_checks\":" << c.product_checks << ",\"residue_checks\":" << c.residue_checks
                  << ",\"reference_checks\":" << c.reference_checks << ",\"preservation_checks\":" << c.preservation_checks
                  << ",\"same_object_raw_products\":" << c.same_object_products
                  << ",\"first_fold_checks\":" << c.first_fold_checks << ",\"high_product_checks\":" << c.high_product_checks
                  << ",\"q_equals_K_cases\":" << c.q_equals_k_cases << ",\"qK_above_u64_cases\":" << c.qk_over_u64_cases
                  << ",\"second_fold_checks\":" << c.second_fold_checks
                  << ",\"second_fold_overflow_cases\":" << c.second_fold_overflow_cases
                  << ",\"third_fold_checks\":" << c.third_fold_checks << ",\"final_p_correction_cases\":" << c.final_p_correction_cases
                  << ",\"accumulator192_checks\":" << c.accumulator192_checks
                  << ",\"accumulator_highword_cases\":" << c.accumulator_highword_cases
                  << ",\"chain_cases\":" << c.chain_cases << ",\"derived_chain_steps\":" << c.chain_steps
                  << ",\"negative_controls\":" << c.negative_controls << ",\"assertions\":" << c.assertions
                  << ",\"checksum_fnv1a64\":\"" << hex64(c.checksum) << "\"}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cout << "{\"schema\":\"parseatlas_p5_field_product_correctness_v1\",\"status\":\"fail\",\"error\":\""
                  << json_escape(error.what()) << "\"}\n";
        return 1;
    }
}
