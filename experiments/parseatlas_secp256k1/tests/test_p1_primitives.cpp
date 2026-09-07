// Finite native-C++ correctness corpus for the unchanged public fast primitives.
// This is not a benchmark, a proof, or a constant-time test. Link the production
// library selected by the experiment's recorded CMake configuration.
#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"

#include <boost/multiprecision/cpp_int.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using boost::multiprecision::cpp_int;
using Bytes = std::array<std::uint8_t, 32>;
using Limbs = std::array<std::uint64_t, 4>;
using Field = secp256k1::fast::FieldElement;
using Scalar = secp256k1::fast::Scalar;

constexpr std::size_t kRandomPairs = 10000;
constexpr std::size_t kRandomInverses = 2000;
constexpr std::uint64_t kFieldSeed = UINT64_C(0x5041315f4649454c);
constexpr std::uint64_t kScalarSeed = UINT64_C(0x5041315f5343414c);
const cpp_int kRadix = cpp_int(1) << 256;
const cpp_int kFieldModulus =
    kRadix - (cpp_int(1) << 32) - 977;
const cpp_int kScalarModulus(
    "0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");

struct Counts {
    std::uint64_t construction_cases = 0;
    std::uint64_t boundary_input_entries = 0;
    std::uint64_t boundary_pairs = 0;
    std::uint64_t random_pairs = 0;
    std::uint64_t square_cases = 0;
    std::uint64_t square_inplace_cases = 0;
    std::uint64_t compound_cases = 0;
    std::uint64_t self_alias_cases = 0;
    std::uint64_t boundary_inverses = 0;
    std::uint64_t random_inverses = 0;
    std::uint64_t inverse_inplace_cases = 0;
    std::uint64_t inverse_zero_cases = 0;
    std::uint64_t comparison_negative_controls = 0;
    std::uint64_t preserved_inputs = 0;
    std::uint64_t checked_outputs = 0;
    std::uint64_t assertions = 0;
    std::uint64_t checksum = UINT64_C(14695981039346656037);
};

struct SplitMix64 {
    std::uint64_t state;

    std::uint64_t next() {
        // Unsigned wrap is intentional and defines this reproducible generator.
        std::uint64_t z = (state += UINT64_C(0x9e3779b97f4a7c15));
        z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
        return z ^ (z >> 31);
    }

    cpp_int raw256() {
        cpp_int result = 0;
        for (unsigned i = 0; i < 4; ++i) {
            result += cpp_int(next()) << (64U * i);
        }
        return result;
    }
};

void require(bool condition, Counts& counts, const std::string& context) {
    ++counts.assertions;
    if (!condition) throw std::runtime_error(context);
}

cpp_int reduce(cpp_int value, const cpp_int& modulus) {
    value %= modulus;
    if (value < 0) value += modulus;
    return value;
}

Bytes encode_be(cpp_int value) {
    if (value < 0 || value >= kRadix) {
        throw std::runtime_error("oracle encode_be argument out of 256-bit range");
    }
    Bytes result{};
    for (std::size_t i = result.size(); i != 0; --i) {
        result[i - 1] = static_cast<std::uint8_t>(value & 255);
        value >>= 8;
    }
    return result;
}

Limbs encode_le(cpp_int value) {
    if (value < 0 || value >= kRadix) {
        throw std::runtime_error("oracle encode_le argument out of 256-bit range");
    }
    const cpp_int mask = (cpp_int(1) << 64) - 1;
    Limbs result{};
    for (auto& limb : result) {
        limb = static_cast<std::uint64_t>(value & mask);
        value >>= 64;
    }
    return result;
}

cpp_int decode_be(const Bytes& bytes) {
    cpp_int value = 0;
    for (const auto byte : bytes) value = value * 256 + byte;
    return value;
}

cpp_int decode_le(const Limbs& limbs) {
    cpp_int value = 0;
    for (std::size_t i = limbs.size(); i != 0; --i) {
        value = value * (cpp_int(1) << 64) + limbs[i - 1];
    }
    return value;
}

std::string hex_bytes(const Bytes& bytes) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (const auto byte : bytes) out << std::setw(2) << static_cast<unsigned>(byte);
    return out.str();
}

// Inspect every byte and every raw limb; do not use production operator==,
// whose normalization could conceal a noncanonical internal result.
bool observations_match(const Bytes& bytes, const Limbs& limbs,
                        const cpp_int& expected, const cpp_int& modulus) {
    const Bytes expected_bytes = encode_be(expected);
    const Limbs expected_limbs = encode_le(expected);
    bool matches = expected >= 0 && expected < modulus;
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        matches &= bytes[i] == expected_bytes[i];
    }
    for (std::size_t i = 0; i < limbs.size(); ++i) {
        matches &= limbs[i] == expected_limbs[i];
    }
    matches &= decode_be(bytes) < modulus;
    matches &= decode_le(limbs) < modulus;
    return matches;
}

template <typename Number>
void check(const Number& actual, const cpp_int& expected, const cpp_int& modulus,
           Counts& counts, const std::string& context) {
    // Capture raw storage before serialization; a normalizing serializer must
    // not be able to hide a noncanonical output from this test.
    const Limbs limbs = actual.limbs();
    const Bytes bytes = actual.to_bytes();
    const bool matches = observations_match(bytes, limbs, expected, modulus);
    if (!matches) {
        require(false, counts, context + ": actual_be=" + hex_bytes(bytes) +
                " actual_raw_limbs_u256=" + hex_bytes(encode_be(decode_le(limbs))) +
                " expected_be=" + hex_bytes(encode_be(expected)));
    }
    require(matches, counts, context);
    require(actual.limbs() == limbs, counts, context + ": serialization mutated limbs");
    ++counts.checked_outputs;
    for (const auto byte : bytes) {
        counts.checksum = (counts.checksum ^ byte) * UINT64_C(1099511628211);
    }
    for (const auto limb : limbs) {
        // Mix limb bytes explicitly, so the checksum is host-endian independent.
        for (unsigned shift = 0; shift < 64; shift += 8) {
            const auto byte = static_cast<std::uint8_t>(limb >> shift);
            counts.checksum = (counts.checksum ^ byte) * UINT64_C(1099511628211);
        }
    }
}

template <typename Number>
Number construct(const cpp_int& raw, const cpp_int& modulus,
                 Counts& counts, const std::string& context) {
    const Bytes bytes = encode_be(raw);
    const Limbs limbs = encode_le(raw);
    const Bytes bytes_before = bytes;
    const Limbs limbs_before = limbs;
    const Number from_bytes = Number::from_bytes(bytes);
    const Number from_limbs = Number::from_limbs(limbs);
    const cpp_int expected = reduce(raw, modulus);
    check(from_bytes, expected, modulus, counts, context + ": from_bytes");
    check(from_limbs, expected, modulus, counts, context + ": from_limbs");
    require(bytes == bytes_before, counts, context + ": byte input changed");
    require(limbs == limbs_before, counts, context + ": limb input changed");
    counts.preserved_inputs += 2;
    ++counts.construction_cases;
    return from_bytes;
}

template <typename Number>
void preserved(const Number& value, const Bytes& bytes, const Limbs& limbs,
               Counts& counts, const std::string& context) {
    require(value.limbs() == limbs, counts, context + ": raw input changed");
    require(value.to_bytes() == bytes, counts, context + ": byte input changed");
    ++counts.preserved_inputs;
}

// Independent Euclidean division, not the production SafeGCD/divsteps or a
// Fermat chain. Only nonnegative cpp_int values are shifted elsewhere; this
// signed-coefficient calculation uses ordinary arbitrary-precision arithmetic.
cpp_int inverse_oracle(const cpp_int& input, const cpp_int& modulus) {
    if (input <= 0 || input >= modulus) {
        throw std::runtime_error("oracle inverse input is not canonical nonzero");
    }
    cpp_int old_r = modulus, r = input;
    cpp_int old_t = 0, t = 1;
    while (r != 0) {
        const cpp_int quotient = old_r / r;
        const cpp_int next_r = old_r - quotient * r;
        const cpp_int next_t = old_t - quotient * t;
        old_r = r;
        r = next_r;
        old_t = t;
        t = next_t;
    }
    if (old_r != 1) throw std::runtime_error("oracle gcd is not one");
    const cpp_int inverse = reduce(old_t, modulus);
    if (reduce(input * inverse, modulus) != 1) {
        throw std::runtime_error("oracle inverse self-check failed");
    }
    return inverse;
}

template <typename Number>
void binary_pair(const cpp_int& raw_a, const cpp_int& raw_b,
                 const cpp_int& modulus, Counts& counts,
                 const std::string& case_context) {
    const cpp_int a_int = reduce(raw_a, modulus);
    const cpp_int b_int = reduce(raw_b, modulus);
    const std::string context = case_context + " a_be=" + hex_bytes(encode_be(a_int)) +
        " b_be=" + hex_bytes(encode_be(b_int));
    const Number a = construct<Number>(raw_a, modulus, counts, context + ": lhs");
    const Number b = construct<Number>(raw_b, modulus, counts, context + ": rhs");
    const Bytes a_bytes = a.to_bytes(), b_bytes = b.to_bytes();
    const Limbs a_limbs = a.limbs(), b_limbs = b.limbs();
    const cpp_int sum = reduce(a_int + b_int, modulus);
    const cpp_int difference = reduce(a_int - b_int, modulus);
    const cpp_int product = reduce(a_int * b_int, modulus);
    const cpp_int squared = reduce(a_int * a_int, modulus);

    check(a + b, sum, modulus, counts, context + ": add");
    check(a - b, difference, modulus, counts, context + ": sub");
    check(a * b, product, modulus, counts, context + ": mul");
    if constexpr (std::is_same_v<Number, Field>) {
        check(a.square(), squared, modulus, counts, context + ": square");
        Number inplace = a;
        inplace.square_inplace();
        check(inplace, squared, modulus, counts, context + ": square_inplace");
        ++counts.square_inplace_cases;
    } else {
        // Scalar has no distinct public square primitive: its baseline is a*a.
        check(a * a, squared, modulus, counts, context + ": square_via_mul");
    }
    ++counts.square_cases;

    Number compound = a;
    require(&(compound += b) == &compound, counts, context + ": += return alias");
    check(compound, sum, modulus, counts, context + ": + =");
    compound = a;
    require(&(compound -= b) == &compound, counts, context + ": -= return alias");
    check(compound, difference, modulus, counts, context + ": - =");
    compound = a;
    require(&(compound *= b) == &compound, counts, context + ": *= return alias");
    check(compound, product, modulus, counts, context + ": * =");
    counts.compound_cases += 3;

    Number self = a;
    require(&(self += self) == &self, counts, context + ": self += return alias");
    check(self, reduce(2 * a_int, modulus), modulus, counts, context + ": self add");
    self = a;
    require(&(self -= self) == &self, counts, context + ": self -= return alias");
    check(self, cpp_int(0), modulus, counts, context + ": self sub");
    self = a;
    require(&(self *= self) == &self, counts, context + ": self *= return alias");
    check(self, squared, modulus, counts, context + ": self mul");
    counts.self_alias_cases += 3;
    preserved(a, a_bytes, a_limbs, counts, context + ": lhs preservation");
    preserved(b, b_bytes, b_limbs, counts, context + ": rhs preservation");
}

template <typename Number>
void inverse_case(const cpp_int& input, const cpp_int& modulus,
                  Counts& counts, const std::string& context) {
    const Number a = construct<Number>(input, modulus, counts, context);
    const Bytes bytes = a.to_bytes();
    const Limbs limbs = a.limbs();
    const cpp_int expected = inverse_oracle(input, modulus);
    const Number inverse = a.inverse();
    check(inverse, expected, modulus, counts, context + ": inverse oracle");
    check(a * inverse, cpp_int(1), modulus, counts, context + ": inverse identity");
    if constexpr (std::is_same_v<Number, Field>) {
        Number inplace = a;
        inplace.inverse_inplace();
        check(inplace, expected, modulus, counts, context + ": inverse_inplace");
        ++counts.inverse_inplace_cases;
    }
    preserved(a, bytes, limbs, counts, context + ": inverse input preservation");
}

template <typename Number>
void zero_inverse(const cpp_int& modulus, Counts& counts) {
    const Number zero = construct<Number>(cpp_int(0), modulus, counts, "zero inverse");
    const Bytes bytes = zero.to_bytes();
    const Limbs limbs = zero.limbs();
    if constexpr (std::is_same_v<Number, Field>) {
        bool threw = false;
        try {
            (void)zero.inverse();
        } catch (const std::runtime_error&) {
            threw = true;
        }
        require(threw, counts, "native FieldElement::inverse(0) must throw runtime_error");
        ++counts.inverse_zero_cases;
        Number inplace = zero;
        threw = false;
        try {
            inplace.inverse_inplace();
        } catch (const std::runtime_error&) {
            threw = true;
        }
        require(threw, counts, "native FieldElement::inverse_inplace(0) must throw");
        check(inplace, cpp_int(0), modulus, counts, "failed inverse_inplace preserves zero");
        ++counts.inverse_zero_cases;
    } else {
        check(zero.inverse(), cpp_int(0), modulus, counts, "Scalar::inverse(0) is zero");
        ++counts.inverse_zero_cases;
    }
    preserved(zero, bytes, limbs, counts, "zero inverse input preservation");
}

template <typename Number>
void comparison_controls(const cpp_int& modulus, Counts& counts) {
    Bytes bytes{};
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<std::uint8_t>(i + 1);
    }
    const Limbs limbs{{UINT64_C(0x191a1b1c1d1e1f20),
                       UINT64_C(0x1112131415161718),
                       UINT64_C(0x090a0b0c0d0e0f10),
                       UINT64_C(0x0102030405060708)}};
    const cpp_int expected(
        "0x0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20");
    require(encode_be(expected) == bytes, counts, "oracle known BE encoding");
    require(encode_le(expected) == limbs, counts, "oracle known LE limb encoding");
    require(decode_be(bytes) == expected, counts, "oracle known BE decoding");
    require(decode_le(limbs) == expected, counts, "oracle known LE limb decoding");
    const Number value = construct<Number>(expected, modulus, counts, "known endian vector");
    require(value.to_bytes() == bytes, counts, "production known BE encoding");
    require(value.limbs() == limbs, counts, "production known LE limb encoding");
    require(observations_match(bytes, limbs, expected, modulus), counts,
            "positive comparison control");
    // Corrupt one observation channel at a time, including the highest byte
    // and highest limb; agreement in the other channel must not conceal it.
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        Bytes bad_bytes = bytes;
        bad_bytes[i] ^= UINT8_C(0x80);
        require(!observations_match(bad_bytes, limbs, expected, modulus), counts,
                "byte corruption was not rejected at " + std::to_string(i));
        ++counts.comparison_negative_controls;
    }
    for (std::size_t i = 0; i < limbs.size(); ++i) {
        Limbs bad_limbs = limbs;
        bad_limbs[i] ^= UINT64_C(0x8000000000000000);
        require(!observations_match(bytes, bad_limbs, expected, modulus), counts,
                "limb corruption was not rejected at " + std::to_string(i));
        ++counts.comparison_negative_controls;
    }
    require(!observations_match(encode_be(modulus), encode_le(modulus),
                                cpp_int(0), modulus), counts,
            "noncanonical modulus must not compare equal to zero");
    ++counts.comparison_negative_controls;
}

std::vector<cpp_int> boundary_inputs(const cpp_int& modulus) {
    // Entries deliberately retain duplicates: this is a documented generated
    // corpus, not a claim of this many distinct mathematical values.
    std::vector<cpp_int> inputs{cpp_int(0), cpp_int(1), cpp_int(2),
                              modulus - 2, modulus - 1, modulus, modulus + 1,
                              kRadix - 2, kRadix - 1};
    for (unsigned bit = 0; bit < 256; ++bit) {
        const cpp_int power = cpp_int(1) << bit;
        inputs.push_back(power - 1);
        inputs.push_back(power);
        inputs.push_back(power + 1);
        inputs.push_back(kRadix - 1 - power);
    }
    return inputs;
}

template <typename Number>
Counts run_domain(const cpp_int& modulus, const char* name, std::uint64_t seed) {
    Counts counts;
    comparison_controls<Number>(modulus, counts);
    zero_inverse<Number>(modulus, counts);
    const std::vector<cpp_int> boundary = boundary_inputs(modulus);
    counts.boundary_input_entries = boundary.size();
    for (std::size_t i = 0; i < boundary.size(); ++i) {
        const cpp_int a = reduce(boundary[i], modulus);
        const std::array<cpp_int, 6> partners{{
            cpp_int(0), cpp_int(1), modulus - 1, a,
            reduce(-a, modulus), reduce(boundary[(i + 1) % boundary.size()], modulus)}};
        // In particular, (2^(64k)-1)+1 and (2^(64k))-1 expose long carry/
        // borrow chains. The first three partner choices also exercise these
        // transitions for all intervening bit positions; a+(-a) wraps at m.
        for (std::size_t partner = 0; partner < partners.size(); ++partner) {
            const std::string context = std::string(name) + " boundary " +
                std::to_string(i) + "/" + std::to_string(partner);
            binary_pair<Number>(boundary[i], partners[partner], modulus, counts, context);
            ++counts.boundary_pairs;
        }
        if (a != 0) {
            inverse_case<Number>(a, modulus, counts,
                                 std::string(name) + " boundary inverse " + std::to_string(i));
            ++counts.boundary_inverses;
        }
    }
    SplitMix64 random{seed};
    for (std::size_t i = 0; i < kRandomPairs; ++i) {
        const cpp_int raw_a = random.raw256();
        const cpp_int raw_b = random.raw256();
        binary_pair<Number>(raw_a, raw_b, modulus, counts,
                            std::string(name) + " random pair " + std::to_string(i));
        ++counts.random_pairs;
    }
    for (std::size_t i = 0; i < kRandomInverses; ++i) {
        cpp_int a;
        do {
            a = reduce(random.raw256(), modulus);
        } while (a == 0);
        inverse_case<Number>(a, modulus, counts,
                             std::string(name) + " random inverse " + std::to_string(i));
        ++counts.random_inverses;
    }
    require(counts.boundary_input_entries == 1033, counts, "boundary corpus size");
    require(counts.boundary_pairs == 6198, counts, "boundary pair corpus size");
    require(counts.random_pairs == kRandomPairs, counts, "random pair corpus size");
    require(counts.random_inverses == kRandomInverses, counts, "random inverse corpus size");
    require(counts.comparison_negative_controls == 37, counts, "negative control corpus size");
    return counts;
}

std::string hex64(std::uint64_t value) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << value;
    return out.str();
}

void print_counts(const char* name, std::uint64_t seed, const Counts& c) {
    std::cout << '"' << name << "\":{"
              << "\"seed_hex\":\"" << hex64(seed) << "\","
              << "\"boundary_input_entries\":" << c.boundary_input_entries << ','
              << "\"boundary_entries_may_repeat\":true,"
              << "\"construction_cases\":" << c.construction_cases << ','
              << "\"boundary_pairs\":" << c.boundary_pairs << ','
              << "\"random_pairs\":" << c.random_pairs << ','
              << "\"binary_pairs\":" << c.boundary_pairs + c.random_pairs << ','
              << "\"square_cases\":" << c.square_cases << ','
              << "\"square_inplace_cases\":" << c.square_inplace_cases << ','
              << "\"compound_cases\":" << c.compound_cases << ','
              << "\"self_alias_cases\":" << c.self_alias_cases << ','
              << "\"boundary_inverses_nonzero\":" << c.boundary_inverses << ','
              << "\"random_inverses_nonzero\":" << c.random_inverses << ','
              << "\"inverse_inplace_cases_nonzero\":" << c.inverse_inplace_cases << ','
              << "\"inverse_zero_cases\":" << c.inverse_zero_cases << ','
              << "\"comparison_negative_controls\":" << c.comparison_negative_controls << ','
              << "\"preserved_inputs\":" << c.preserved_inputs << ','
              << "\"checked_outputs\":" << c.checked_outputs << ','
              << "\"assertions\":" << c.assertions << ','
              << "\"checksum_fnv1a64\":\"" << hex64(c.checksum) << "\"}";
}

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (const unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << static_cast<char>(c);
        else if (c < 0x20) {
            out << "\\u" << std::hex << std::setfill('0') << std::setw(4)
                << static_cast<unsigned>(c) << std::dec;
        } else out << static_cast<char>(c);
    }
    return out.str();
}

} // namespace

int main(int argc, char** argv) {
    std::string domain = "all";
    if (argc == 3 && std::string(argv[1]) == "--domain") domain = argv[2];
    else if (argc != 1) {
        std::cerr << "Usage: test_p1_primitives [--domain field|scalar|all] (JSON stdout)\n";
        return 2;
    }
    if (domain != "all" && domain != "field" && domain != "scalar") {
        std::cerr << "Invalid domain; expected field, scalar, or all\n";
        return 2;
    }
    try {
        Counts field, scalar;
        const bool run_field = domain == "all" || domain == "field";
        const bool run_scalar = domain == "all" || domain == "scalar";
        if (run_field) field = run_domain<Field>(kFieldModulus, "field", kFieldSeed);
        if (run_scalar) scalar = run_domain<Scalar>(kScalarModulus, "scalar", kScalarSeed);
        std::cout << "{\"schema\":\"parseatlas_p1_primitives_correctness_v1\","
                  << "\"status\":\"pass\",\"mismatches\":0,"
                  << "\"requested_domain\":\"" << domain << "\","
                  << "\"finite_corpus_only\":true,\"timing_claim\":false,"
                  << "\"constant_time_claim\":false,"
                  << "\"oracle\":\"Boost cpp_int residues and independent Euclidean inverse\","
                  << "\"field_square\":\"FieldElement::square\","
                  << "\"scalar_square\":\"Scalar::operator*(self)\","
                  << "\"field_zero_inverse_contract\":\"native runtime_error\","
                  << "\"scalar_zero_inverse_contract\":\"zero\","
                  << "\"raw_inputs\":\"256-bit, independently reduced before arithmetic comparison\","
                  << "\"output_checks\":\"all 32 BE bytes, all 4 raw LE limbs, canonical range\","
                  << "\"domains\":{";
        if (run_field) print_counts("field", kFieldSeed, field);
        if (run_field && run_scalar) std::cout << ',';
        if (run_scalar) print_counts("scalar", kScalarSeed, scalar);
        std::cout << "}}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cout << "{\"schema\":\"parseatlas_p1_primitives_correctness_v1\","
                  << "\"status\":\"fail\",\"error\":\"" << json_escape(error.what())
                  << "\"}\n";
        return 1;
    }
}
