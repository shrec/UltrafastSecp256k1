// P2 finite field-only correctness corpus. No timing or side-channel claims.
// Resident FE52 results are explicitly normalized after EVERY operation. This
// tests canonical arithmetic, not an unproved lazy-magnitude/headroom budget.
#include "secp256k1/field.hpp"
#include "secp256k1/field_52.hpp"
#include "secp256k1/ct/field.hpp"

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
#include <vector>

namespace {
using boost::multiprecision::cpp_int;
using FE64 = secp256k1::fast::FieldElement;
using FE52 = secp256k1::fast::FieldElement52;
using Bytes = std::array<std::uint8_t, 32>;
using L64 = std::array<std::uint64_t, 4>;
using L52 = std::array<std::uint64_t, 5>;

const cpp_int kB = cpp_int(1) << 256;
const cpp_int kP = kB - (cpp_int(1) << 32) - 977;
constexpr std::uint64_t kSeed = UINT64_C(0x5041325f4649454c);
constexpr std::uint64_t kM52 = UINT64_C(0xfffffffffffff);
constexpr std::uint64_t kM48 = UINT64_C(0xffffffffffff);
constexpr std::size_t kRandomPairs = 10000;
constexpr std::size_t kRandomInverses = 2000;
constexpr std::size_t kChainsPerOperation = 64;
constexpr std::size_t kChainLength = 32;

struct Counts {
    std::uint64_t assertions = 0;
    std::uint64_t constructions = 0;
    std::uint64_t boundary_entries = 0;
    std::uint64_t boundary_pairs = 0;
    std::uint64_t random_pairs = 0;
    std::uint64_t regression_pairs = 0;
    std::uint64_t binary_cases = 0;
    std::uint64_t boundary_inverses = 0;
    std::uint64_t random_inverses = 0;
    std::uint64_t inverse_cases = 0;
    std::uint64_t zero_inverse_cases = 0;
    std::uint64_t chain_cases = 0;
    std::uint64_t chain_steps = 0;
    std::uint64_t fe64_compound_cases = 0;
    std::uint64_t fe64_self_alias_cases = 0;
    std::uint64_t fe52_compound_cases = 0;
    std::uint64_t fe52_self_alias_cases = 0;
    std::uint64_t input_preservation_checks = 0;
    std::uint64_t fe64_output_checks = 0;
    std::uint64_t fe52_output_checks = 0;
    std::uint64_t bridge_raw_residue_checks = 0;
    std::uint64_t fe64_negative_controls = 0;
    std::uint64_t fe52_negative_controls = 0;
    std::uint64_t checksum = UINT64_C(14695981039346656037);
};

struct SplitMix64 {
    std::uint64_t state;
    std::uint64_t next() {
        std::uint64_t z = (state += UINT64_C(0x9e3779b97f4a7c15));
        z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
        return z ^ (z >> 31);
    }
    cpp_int raw256() {
        cpp_int value = 0;
        for (unsigned i = 0; i < 4; ++i) value += cpp_int(next()) << (64U * i);
        return value;
    }
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

Bytes bytes_be(cpp_int value) {
    if (value < 0 || value >= kB) throw std::runtime_error("oracle BE input range");
    Bytes bytes{};
    for (std::size_t i = bytes.size(); i != 0; --i) {
        bytes[i - 1] = static_cast<std::uint8_t>(value & 255);
        value >>= 8;
    }
    return bytes;
}

template <std::size_t N>
std::array<std::uint64_t, N> encode_limbs(cpp_int value, unsigned width) {
    if (value < 0 || value >= kB) throw std::runtime_error("oracle limb input range");
    const cpp_int mask = (cpp_int(1) << width) - 1;
    std::array<std::uint64_t, N> limbs{};
    for (auto& limb : limbs) {
        limb = static_cast<std::uint64_t>(value & mask);
        value >>= width;
    }
    if (value != 0) throw std::runtime_error("oracle limb truncation");
    return limbs;
}

template <std::size_t N>
cpp_int decode_limbs(const std::array<std::uint64_t, N>& limbs, unsigned width) {
    cpp_int value = 0;
    for (std::size_t i = limbs.size(); i != 0; --i) {
        value *= cpp_int(1) << width;
        value += limbs[i - 1];
    }
    return value;
}

L52 raw52(const FE52& value) {
    return {{value.n[0], value.n[1], value.n[2], value.n[3], value.n[4]}};
}

std::string hex_bytes(const Bytes& bytes) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (const auto byte : bytes) out << std::setw(2) << static_cast<unsigned>(byte);
    return out.str();
}

template <std::size_t N>
std::string hex_limbs(const std::array<std::uint64_t, N>& limbs) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << '[';
    for (std::size_t i = 0; i < N; ++i) {
        if (i != 0) out << ',';
        out << std::setw(16) << limbs[i];
    }
    out << ']';
    return out.str();
}

bool matches64(const Bytes& bytes, const L64& limbs, const cpp_int& expected) {
    const auto expected_bytes = bytes_be(expected);
    const auto expected_limbs = encode_limbs<4>(expected, 64);
    bool result = expected >= 0 && expected < kP;
    for (std::size_t i = 0; i < bytes.size(); ++i) result &= bytes[i] == expected_bytes[i];
    for (std::size_t i = 0; i < limbs.size(); ++i) result &= limbs[i] == expected_limbs[i];
    result &= decode_limbs(limbs, 64) < kP;
    return result;
}

bool canonical52_matches(const L52& limbs, const cpp_int& expected) {
    const auto expected_limbs = encode_limbs<5>(expected, 52);
    bool result = expected >= 0 && expected < kP;
    for (std::size_t i = 0; i < limbs.size(); ++i) {
        result &= limbs[i] == expected_limbs[i];
        result &= limbs[i] <= (i == 4 ? kM48 : kM52);
    }
    result &= decode_limbs(limbs, 52) < kP;
    return result;
}

bool matches52(const Bytes& bytes, const L52& limbs, const cpp_int& expected) {
    const Bytes expected_bytes = bytes_be(expected);
    bool result = canonical52_matches(limbs, expected);
    for (std::size_t i = 0; i < bytes.size(); ++i) result &= bytes[i] == expected_bytes[i];
    return result;
}

void mix(const Bytes& bytes, Counts& c) {
    for (const auto byte : bytes) c.checksum = (c.checksum ^ byte) * UINT64_C(1099511628211);
}

void check64(const FE64& actual, const cpp_int& expected, Counts& c,
             const std::string& context) {
    const L64 limbs = actual.limbs();
    const Bytes bytes = actual.to_bytes();
    if (!matches64(bytes, limbs, expected)) {
        require(false, c, context + " actual_be=" + hex_bytes(bytes) +
                " actual_raw64_le=" + hex_limbs(limbs) +
                " expected_be=" + hex_bytes(bytes_be(expected)));
    }
    require(actual.limbs() == limbs, c, context + ": serialization mutated input");
    ++c.fe64_output_checks;
    mix(bytes, c);
}

void check52(const FE52& actual, const cpp_int& expected, Counts& c,
             const std::string& context) {
    // Check actual resident storage BEFORE to_fe()/serialization. Both bridge
    // functions normalize a copy and could otherwise hide a noncanonical state.
    const L52 limbs = raw52(actual);
    require(canonical52_matches(limbs, expected), c,
            context + " actual_raw52_le=" + hex_limbs(limbs) +
            " expected_be=" + hex_bytes(bytes_be(expected)));
    Bytes serialized{}, prenormalized{};
    actual.to_bytes_into(serialized.data());
    actual.store_b32_prenorm(prenormalized.data());
    require(matches52(serialized, limbs, expected), c, context + ": full 32-byte serializer");
    require(matches52(prenormalized, limbs, expected), c, context + ": full prenormalized serializer");
    check64(actual.to_fe(), expected, c, context + ": to_fe canonical bridge");
    require(raw52(actual) == limbs, c, context + ": serialization/bridge mutated input");
    ++c.fe52_output_checks;
    mix(serialized, c);
}

FE64 check_bridge_raw(const FE52& actual, const cpp_int& expected, Counts& c,
                      const std::string& context) {
    // A bridge is allowed to hold a noncanonical intermediate. Verify its
    // mathematical residue independently, then let to_fe() perform the ONE
    // canonical normalization required at this representation boundary.
    const L52 limbs = raw52(actual);
    require(mod(decode_limbs(limbs, 52)) == expected, c,
            context + " raw_bridge_residue actual_raw52_le=" + hex_limbs(limbs) +
            " expected_be=" + hex_bytes(bytes_be(expected)));
    const FE64 result = actual.to_fe();
    check64(result, expected, c, context + ": single-normalization bridge result");
    require(raw52(actual) == limbs, c, context + ": bridge mutated raw input");
    ++c.bridge_raw_residue_checks;
    ++c.input_preservation_checks;
    return result;
}

struct Inputs {
    FE64 fe64;
    FE52 fe52;
};

Inputs construct(const cpp_int& raw, Counts& c, const std::string& context) {
    const auto bytes = bytes_be(raw);
    const auto limbs = encode_limbs<4>(raw, 64);
    const auto bytes_before = bytes;
    const auto limbs_before = limbs;
    const cpp_int expected = mod(raw);
    Inputs inputs{FE64::from_bytes(bytes), FE52::from_bytes(bytes)};
    check64(inputs.fe64, expected, c, context + ": FE64 from_bytes");
    check64(FE64::from_limbs(limbs), expected, c, context + ": FE64 from_limbs");
    check52(inputs.fe52, expected, c, context + ": FE52 from_bytes");
    check52(FE52::from_bytes(bytes.data()), expected, c, context + ": FE52 pointer from_bytes");
    check52(FE52::from_fe(inputs.fe64), expected, c, context + ": FE64-to-FE52 bridge");
    const auto canonical_limbs = inputs.fe64.limbs();
    check52(FE52::from_4x64_limbs(canonical_limbs.data()), expected, c,
            context + ": canonical direct 4x64-to-5x52 bridge");
    require(bytes == bytes_before, c, context + ": byte construction input changed");
    require(limbs == limbs_before, c, context + ": limb construction input changed");
    c.input_preservation_checks += 2;
    ++c.constructions;
    return inputs;
}

enum class Operation { Add, Sub, Mul, Square };
constexpr std::array<Operation, 4> kOperations{{
    Operation::Add, Operation::Sub, Operation::Mul, Operation::Square}};

const char* operation_name(Operation op) {
    switch (op) {
        case Operation::Add: return "add";
        case Operation::Sub: return "sub";
        case Operation::Mul: return "mul";
        case Operation::Square: return "square";
    }
    throw std::runtime_error("unknown operation");
}

cpp_int oracle(Operation op, const cpp_int& a, const cpp_int& b) {
    switch (op) {
        case Operation::Add: return mod(a + b);
        case Operation::Sub: return mod(a - b);
        case Operation::Mul: return mod(a * b);
        case Operation::Square: return mod(a * a);
    }
    throw std::runtime_error("unknown oracle operation");
}

FE64 operation64(Operation op, const FE64& a, const FE64& b) {
    switch (op) {
        case Operation::Add: return a + b;
        case Operation::Sub: return a - b;
        case Operation::Mul: return a * b;
        case Operation::Square: return a.square();
    }
    throw std::runtime_error("unknown FE64 operation");
}

FE64 operation_ct(Operation op, const FE64& a, const FE64& b) {
    switch (op) {
        case Operation::Add: return secp256k1::ct::field_add(a, b);
        case Operation::Sub: return secp256k1::ct::field_sub(a, b);
        case Operation::Mul: return secp256k1::ct::field_mul(a, b);
        case Operation::Square: return secp256k1::ct::field_sqr(a);
    }
    throw std::runtime_error("unknown CT operation");
}

FE52 operation52_raw(Operation op, const FE52& a, const FE52& b) {
    FE52 result{};
    switch (op) {
        case Operation::Add: result = a + b; break;
        case Operation::Sub: result = a + b.negate(1); break;
        case Operation::Mul: result = a * b; break;
        case Operation::Square: result = a.square(); break;
    }
    return result;
}

FE52 operation52(Operation op, const FE52& a, const FE52& b) {
    FE52 result = operation52_raw(op, a, b);
    // This normalization is part of the tested full canonical contract, even
    // for mul/square whose unnormalized outputs may only have magnitude one.
    result.normalize();
    return result;
}

void input_preserved(const Inputs& value, const L64& before64, const L52& before52,
                     Counts& c, const std::string& context) {
    require(value.fe64.limbs() == before64, c, context + ": FE64 input changed");
    require(raw52(value.fe52) == before52, c, context + ": FE52 input changed");
    c.input_preservation_checks += 2;
}

void aliases(const Inputs& a, const Inputs& b, const cpp_int& ai, const cpp_int& bi,
             Counts& c, const std::string& context) {
    FE64 value64 = a.fe64;
    require(&(value64 += b.fe64) == &value64, c, context + ": FE64 += return alias");
    check64(value64, mod(ai + bi), c, context + ": FE64 +=");
    value64 = a.fe64;
    require(&(value64 -= b.fe64) == &value64, c, context + ": FE64 -= return alias");
    check64(value64, mod(ai - bi), c, context + ": FE64 -=");
    value64 = a.fe64;
    require(&(value64 *= b.fe64) == &value64, c, context + ": FE64 *= return alias");
    check64(value64, mod(ai * bi), c, context + ": FE64 *=");
    value64 = a.fe64;
    value64.square_inplace();
    check64(value64, mod(ai * ai), c, context + ": FE64 square_inplace");
    c.fe64_compound_cases += 4;
    value64 = a.fe64;
    value64 += value64;
    check64(value64, mod(ai + ai), c, context + ": FE64 self +=");
    value64 = a.fe64;
    value64 -= value64;
    check64(value64, cpp_int(0), c, context + ": FE64 self -=");
    value64 = a.fe64;
    value64 *= value64;
    check64(value64, mod(ai * ai), c, context + ": FE64 self *=");
    c.fe64_self_alias_cases += 3;

    // Only supported public wrappers are used. In particular raw
    // fe52_mul_inner destinations are NOT allowed to alias their inputs.
    FE52 value52 = a.fe52;
    value52.add_assign(b.fe52);
    value52.normalize();
    check52(value52, mod(ai + bi), c, context + ": FE52 add_assign");
    value52 = b.fe52;
    value52.negate_assign(1);
    value52.add_assign(a.fe52);
    value52.normalize();
    check52(value52, mod(ai - bi), c, context + ": FE52 composed sub_assign");
    value52 = a.fe52;
    value52.mul_assign(b.fe52);
    value52.normalize();
    check52(value52, mod(ai * bi), c, context + ": FE52 mul_assign");
    value52 = a.fe52;
    require(&(value52 *= b.fe52) == &value52, c, context + ": FE52 *= return alias");
    value52.normalize();
    check52(value52, mod(ai * bi), c, context + ": FE52 *=");
    value52 = a.fe52;
    value52.square_inplace();
    value52.normalize();
    check52(value52, mod(ai * ai), c, context + ": FE52 square_inplace");
    c.fe52_compound_cases += 5;
    value52 = a.fe52;
    value52.add_assign(value52);
    value52.normalize();
    check52(value52, mod(ai + ai), c, context + ": FE52 self add_assign");
    value52 = a.fe52;
    value52.mul_assign(value52);
    value52.normalize();
    check52(value52, mod(ai * ai), c, context + ": FE52 self mul_assign");
    value52 = a.fe52;
    require(&(value52 *= value52) == &value52, c, context + ": FE52 self *= return alias");
    value52.normalize();
    check52(value52, mod(ai * ai), c, context + ": FE52 self *=");
    c.fe52_self_alias_cases += 3;
}

void binary_case(const cpp_int& raw_a, const cpp_int& raw_b, Counts& c,
                 const std::string& label) {
    const cpp_int ai = mod(raw_a), bi = mod(raw_b);
    const std::string context = label + " a_be=" + hex_bytes(bytes_be(ai)) +
        " b_be=" + hex_bytes(bytes_be(bi));
    const Inputs a = construct(raw_a, c, context + ": a");
    const Inputs b = construct(raw_b, c, context + ": b");
    const L64 before_a64 = a.fe64.limbs(), before_b64 = b.fe64.limbs();
    const L52 before_a52 = raw52(a.fe52), before_b52 = raw52(b.fe52);
    for (const auto op : kOperations) {
        const cpp_int expected = oracle(op, ai, bi);
        const std::string at = context + ": " + operation_name(op);
        check64(operation64(op, a.fe64, b.fe64), expected, c, at + " FE64");
        check64(operation_ct(op, a.fe64, b.fe64), expected, c, at + " CT API");
        check52(operation52(op, a.fe52, b.fe52), expected, c, at + " FE52 resident");
        const FE52 bridged = operation52_raw(op, FE52::from_fe(a.fe64), FE52::from_fe(b.fe64));
        check_bridge_raw(bridged, expected, c, at + " FE52 bridge");
    }
    aliases(a, b, ai, bi, c, context);
    input_preserved(a, before_a64, before_a52, c, context + ": a preservation");
    input_preserved(b, before_b64, before_b52, c, context + ": b preservation");
    ++c.binary_cases;
}

cpp_int inverse_oracle(const cpp_int& a) {
    if (a <= 0 || a >= kP) throw std::runtime_error("oracle inverse input range");
    cpp_int old_r = kP, r = a, old_t = 0, t = 1;
    while (r != 0) {
        const cpp_int quotient = old_r / r;
        const cpp_int next_r = old_r - quotient * r;
        const cpp_int next_t = old_t - quotient * t;
        old_r = r; r = next_r; old_t = t; t = next_t;
    }
    if (old_r != 1) throw std::runtime_error("oracle inverse gcd != 1");
    const cpp_int result = mod(old_t);
    if (mod(a * result) != 1) throw std::runtime_error("oracle inverse self-check");
    return result;
}

void inverse_case(const cpp_int& ai, Counts& c, const std::string& label) {
    const std::string context = label + " a_be=" + hex_bytes(bytes_be(ai));
    const Inputs a = construct(ai, c, context);
    const L64 before64 = a.fe64.limbs();
    const L52 before52 = raw52(a.fe52);
    const cpp_int expected = inverse_oracle(ai);
    check64(a.fe64.inverse(), expected, c, context + ": FE64 inverse");
    FE64 inplace = a.fe64;
    inplace.inverse_inplace();
    check64(inplace, expected, c, context + ": FE64 inverse_inplace");
    check64(secp256k1::ct::field_inv(a.fe64), expected, c, context + ": CT inverse API");
    FE52 resident = a.fe52.inverse_safegcd();
    resident.normalize();
    check52(resident, expected, c, context + ": FE52 resident direct SafeGCD");
    FE52 bridge = FE52::from_fe(a.fe64).inverse_safegcd();
    check_bridge_raw(bridge, expected, c, context + ": FE52 bridge direct SafeGCD");
    FE52 fixed_chain = a.fe52.inverse();
    fixed_chain.normalize();
    check52(fixed_chain, expected, c, context + ": FE52 fixed-chain inverse");
    input_preserved(a, before64, before52, c, context + ": preservation");
    ++c.inverse_cases;
}

void zero_contracts(Counts& c) {
    const Inputs zero = construct(cpp_int(0), c, "zero contracts");
    const L64 before64 = zero.fe64.limbs();
    const L52 before52 = raw52(zero.fe52);
    bool threw = false;
    try { (void)zero.fe64.inverse(); }
    catch (const std::runtime_error&) { threw = true; }
    require(threw, c, "native FE64 inverse(0) must throw runtime_error");
    ++c.zero_inverse_cases;
    FE64 inplace = zero.fe64;
    threw = false;
    try { inplace.inverse_inplace(); }
    catch (const std::runtime_error&) { threw = true; }
    require(threw, c, "native FE64 inverse_inplace(0) must throw runtime_error");
    check64(inplace, cpp_int(0), c, "FE64 failed zero inverse_inplace preserves value");
    ++c.zero_inverse_cases;
    // Fresh implementation has an explicit zero return before direct 5x52 to
    // signed62 SafeGCD. It does NOT enter the old throwing FE64 bridge.
    FE52 safe = zero.fe52.inverse_safegcd();
    safe.normalize();
    check52(safe, cpp_int(0), c, "FE52 direct SafeGCD inverse(0) returns zero");
    ++c.zero_inverse_cases;
    FE52 fixed = zero.fe52.inverse();
    fixed.normalize();
    check52(fixed, cpp_int(0), c, "FE52 fixed-chain inverse(0) returns zero");
    ++c.zero_inverse_cases;
    check64(secp256k1::ct::field_inv(zero.fe64), cpp_int(0), c,
            "CT inverse API observed zero result");
    ++c.zero_inverse_cases;
    input_preserved(zero, before64, before52, c, "zero inverse preservation");
}

void comparison_controls(Counts& c) {
    Bytes bytes{};
    for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<std::uint8_t>(i + 1);
    const L64 limbs64{{UINT64_C(0x191a1b1c1d1e1f20), UINT64_C(0x1112131415161718),
                      UINT64_C(0x090a0b0c0d0e0f10), UINT64_C(0x0102030405060708)}};
    const cpp_int expected(
        "0x0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20");
    const L52 limbs52 = encode_limbs<5>(expected, 52);
    require(bytes_be(expected) == bytes, c, "known endian BE oracle vector");
    require(encode_limbs<4>(expected, 64) == limbs64, c, "known endian LE64 oracle vector");
    require(decode_limbs(limbs52, 52) == expected, c, "LE52 oracle round trip");
    construct(expected, c, "known endian vector");
    require(matches64(bytes, limbs64, expected), c, "FE64 positive comparator control");
    require(matches52(bytes, limbs52, expected), c, "FE52 positive comparator control");
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        Bytes bad = bytes;
        bad[i] ^= UINT8_C(0x80);
        require(!matches64(bad, limbs64, expected), c, "FE64 missed byte corruption " + std::to_string(i));
        require(!matches52(bad, limbs52, expected), c, "FE52 missed byte corruption " + std::to_string(i));
        ++c.fe64_negative_controls;
        ++c.fe52_negative_controls;
    }
    for (std::size_t i = 0; i < limbs64.size(); ++i) {
        auto bad = limbs64;
        bad[i] ^= UINT64_C(0x8000000000000000);
        require(!matches64(bytes, bad, expected), c, "FE64 missed limb corruption " + std::to_string(i));
        ++c.fe64_negative_controls;
    }
    for (std::size_t i = 0; i < limbs52.size(); ++i) {
        auto bad = limbs52;
        bad[i] ^= UINT64_C(1);
        require(!matches52(bytes, bad, expected), c, "FE52 missed limb corruption " + std::to_string(i));
        ++c.fe52_negative_controls;
    }
    // Matching byte/limb encodings of p must fail specifically because p is
    // not canonical. This also tests the comparator's independent range gate.
    require(!matches64(bytes_be(kP), encode_limbs<4>(kP, 64), kP), c, "FE64 comparator accepted p");
    require(!matches52(bytes_be(kP), encode_limbs<5>(kP, 52), kP), c, "FE52 comparator accepted p");
    ++c.fe64_negative_controls;
    ++c.fe52_negative_controls;
}

std::vector<cpp_int> boundary_inputs() {
    // Duplicates are retained and reported, not counted as distinct values.
    std::vector<cpp_int> values{cpp_int(0), cpp_int(1), cpp_int(2), kP - 2, kP - 1,
                               kP, kP + 1, kB - 2, kB - 1};
    for (unsigned bit = 0; bit < 256; ++bit) {
        const cpp_int power = cpp_int(1) << bit;
        values.push_back(power - 1);
        values.push_back(power);
        values.push_back(power + 1);
        values.push_back(kB - 1 - power);
    }
    return values;
}

void old_p1_regressions(Counts& c) {
    const cpp_int a1 = decode_limbs(L64{{UINT64_C(0xfffffffdffffffff), UINT64_MAX,
                                       UINT64_MAX, UINT64_MAX}}, 64);
    const cpp_int square1 = cpp_int(UINT64_C(0xfffff860000e8900));
    require(oracle(Operation::Square, a1, cpp_int(0)) == square1, c, "P1 square KAT oracle");
    binary_case(a1, cpp_int(0), c, "P1 square carry KAT expected_square=fffff860000e8900");
    ++c.regression_pairs;
    const cpp_int a2 = decode_limbs(L64{{1, 0, 0, UINT64_C(0x8000000000000000)}}, 64);
    const cpp_int b2 = kP - 1;
    const cpp_int product2 = decode_limbs(L64{{UINT64_C(0xfffffffefffffc2e), UINT64_MAX,
                                              UINT64_MAX, UINT64_C(0x7fffffffffffffff)}}, 64);
    require(oracle(Operation::Mul, a2, b2) == product2, c, "P1 multiply carry KAT oracle");
    binary_case(a2, b2, c, "P1 multiply carry KAT");
    ++c.regression_pairs;
}

void chained_cases(Counts& c) {
    SplitMix64 random{kSeed ^ UINT64_C(0x434841494e5f5032)};
    for (const auto op : kOperations) {
        for (std::size_t chain = 0; chain < kChainsPerOperation; ++chain) {
            const std::string base = std::string("chain ") + operation_name(op) +
                "/" + std::to_string(chain);
            cpp_int expected = mod(random.raw256());
            const Inputs initial = construct(expected, c, base + ": initial");
            FE64 state64 = initial.fe64, state_ct = initial.fe64, bridge_state = initial.fe64;
            FE52 resident = initial.fe52;
            for (std::size_t step = 0; step < kChainLength; ++step) {
                const cpp_int rhs_int = mod(random.raw256());
                const Inputs rhs = construct(rhs_int, c, base + ": rhs");
                const L64 rhs64 = rhs.fe64.limbs();
                const L52 rhs52 = raw52(rhs.fe52);
                expected = oracle(op, expected, rhs_int);
                const std::string context = base + " step " + std::to_string(step);
                state64 = operation64(op, state64, rhs.fe64);
                state_ct = operation_ct(op, state_ct, rhs.fe64);
                resident = operation52(op, resident, rhs.fe52);
                const FE52 bridge_intermediate = operation52_raw(
                    op, FE52::from_fe(bridge_state), FE52::from_fe(rhs.fe64));
                bridge_state = check_bridge_raw(bridge_intermediate, expected, c,
                                                context + ": FE52 bridge");
                check64(state64, expected, c, context + ": FE64");
                check64(state_ct, expected, c, context + ": CT API");
                check52(resident, expected, c, context + ": FE52 resident");
                check64(bridge_state, expected, c, context + ": FE52 bridge");
                input_preserved(rhs, rhs64, rhs52, c, context + ": rhs preservation");
                ++c.chain_steps;
            }
            ++c.chain_cases;
        }
    }
}

Counts run() {
    Counts c;
    comparison_controls(c);
    zero_contracts(c);
    old_p1_regressions(c);
    const auto boundaries = boundary_inputs();
    c.boundary_entries = boundaries.size();
    for (std::size_t i = 0; i < boundaries.size(); ++i) {
        const cpp_int a = mod(boundaries[i]);
        const std::array<cpp_int, 6> partners{{cpp_int(0), cpp_int(1), kP - 1, a,
            mod(-a), mod(boundaries[(i + 1) % boundaries.size()])}};
        // Covers (2^(64k)-1)+1 and 2^(64k)-1 carry/borrow chains as well as
        // 52-bit limb boundaries, modulus wrapping, and all 256 bit positions.
        for (std::size_t j = 0; j < partners.size(); ++j) {
            binary_case(boundaries[i], partners[j], c,
                        "boundary " + std::to_string(i) + "/" + std::to_string(j));
            ++c.boundary_pairs;
        }
        if (a != 0) {
            inverse_case(a, c, "boundary inverse " + std::to_string(i));
            ++c.boundary_inverses;
        }
    }
    SplitMix64 random{kSeed};
    for (std::size_t i = 0; i < kRandomPairs; ++i) {
        const cpp_int a = random.raw256(), b = random.raw256();
        binary_case(a, b, c, "random pair " + std::to_string(i));
        ++c.random_pairs;
    }
    for (std::size_t i = 0; i < kRandomInverses; ++i) {
        cpp_int a;
        do { a = mod(random.raw256()); } while (a == 0);
        inverse_case(a, c, "random inverse " + std::to_string(i));
        ++c.random_inverses;
    }
    chained_cases(c);
    require(c.boundary_entries == 1033 && c.boundary_pairs == 6198, c, "boundary corpus counts");
    require(c.regression_pairs == 2 && c.binary_cases == 16200, c, "binary corpus counts");
    require(c.random_pairs == kRandomPairs && c.random_inverses == kRandomInverses, c, "random corpus counts");
    require(c.boundary_inverses == 1030 && c.inverse_cases == 3030, c, "inverse corpus counts");
    require(c.chain_cases == 256 && c.chain_steps == 8192, c, "chain corpus counts");
    require(c.fe64_negative_controls == 37 && c.fe52_negative_controls == 38, c, "negative control counts");
    return c;
}

std::string hex64(std::uint64_t value) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << value;
    return out.str();
}

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (const unsigned char ch : value) {
        if (ch == '"' || ch == '\\') out << '\\' << static_cast<char>(ch);
        else if (ch < 0x20) out << "\\u" << std::hex << std::setfill('0')
                               << std::setw(4) << static_cast<unsigned>(ch) << std::dec;
        else out << static_cast<char>(ch);
    }
    return out.str();
}

int lazy_boundary_diagnostic() {
    // Separate, opt-in range experiment. It intentionally omits normalization
    // between additions and is NOT part of P2's canonical per-operation gate.
    // Zero initialization means N terms require exactly N add_assign calls.
    const FE52 term{{kM52, 0, 0, 0, kM48}};
    const cpp_int term_int = decode_limbs(raw52(term), 52);
    if (term_int >= kP) throw std::runtime_error("lazy diagnostic term must be canonical");
    const FE64 term64 = FE64::from_bytes(bytes_be(term_int));
    constexpr std::array<std::size_t, 3> counts{{4095, 4096, 4097}};
    std::ostringstream cases;
    cases << std::boolalpha;
    bool all_match = true;
    for (std::size_t case_index = 0; case_index < counts.size(); ++case_index) {
        const std::size_t terms = counts[case_index];
        FE52 lazy = FE52::zero();
        FE64 reference = FE64::zero();
        for (std::size_t i = 0; i < terms; ++i) {
            lazy.add_assign(term);
            reference += term64;  // Unchanged FE64 reduction after every term.
        }
        const L52 accumulated_raw = raw52(lazy);
        const cpp_int mathematical_sum = cpp_int(terms) * term_int;
        const cpp_int expected = mod(mathematical_sum);
        const bool raw_exact = decode_limbs(accumulated_raw, 52) == mathematical_sum;
        const bool reference_matches = matches64(reference.to_bytes(), reference.limbs(), expected);
        FE52 full = lazy;
        full.normalize();
        FE52 weak_full = lazy;
        weak_full.normalize_weak();
        weak_full.normalize();
        // Observe actual normalized raw storage before any converting method
        // can normalize again and conceal the representation-order difference.
        const L52 full_raw = raw52(full), weak_full_raw = raw52(weak_full);
        Bytes full_bytes{}, weak_full_bytes{};
        full.to_bytes_into(full_bytes.data());
        weak_full.to_bytes_into(weak_full_bytes.data());
        const bool full_matches = matches52(full_bytes, full_raw, expected);
        const bool weak_full_matches = matches52(weak_full_bytes, weak_full_raw, expected);
        all_match &= raw_exact && reference_matches && full_matches && weak_full_matches;
        if (case_index != 0) cases << ',';
        cases << "{\"terms\":" << terms << ",\"add_assign_calls_from_zero\":" << terms
              << ",\"raw_sum_mathematically_exact\":" << raw_exact
              << ",\"raw_limbs_le_hex\":\"" << hex_limbs(accumulated_raw) << '"'
              << ",\"expected_be\":\"" << hex_bytes(bytes_be(expected)) << '"'
              << ",\"fe64_reference_matches\":" << reference_matches
              << ",\"fe64_reference_be\":\"" << hex_bytes(reference.to_bytes()) << '"'
              << ",\"direct_full_matches\":" << full_matches
              << ",\"direct_full_raw_le_hex\":\"" << hex_limbs(full_raw) << '"'
              << ",\"direct_full_be\":\"" << hex_bytes(full_bytes) << '"'
              << ",\"weak_then_full_matches\":" << weak_full_matches
              << ",\"weak_then_full_raw_le_hex\":\"" << hex_limbs(weak_full_raw) << '"'
              << ",\"weak_then_full_be\":\"" << hex_bytes(weak_full_bytes) << "\"}";
    }
    std::cout << "{\"schema\":\"parseatlas_p2_optional_lazy_boundary_v1\","
              << "\"status\":\"" << (all_match ? "pass" : "fail") << "\","
              << "\"outside_p2_canonical_contract\":true,\"timing_claim\":false,"
              << "\"term_be\":\"" << hex_bytes(bytes_be(term_int)) << "\","
              << "\"term_raw_le_hex\":\"" << hex_limbs(raw52(term)) << "\","
              << "\"semantics\":\"N identical independent terms added to zero; no intermediate FE52 normalization\","
              << "\"oracle\":\"Boost cpp_int plus unchanged FE64 canonical addition reference\","
              << "\"cases\":[" << cases.str() << "]}\n";
    return all_match ? 0 : 1;
}
} // namespace

int main(int argc, char** argv) {
    const bool lazy_only = argc == 2 && std::string(argv[1]) == "--lazy-boundary-only";
    if (argc != 1 && !lazy_only) {
        std::cerr << "Usage: test_p2_field_representations [--lazy-boundary-only] (JSON stdout)\n";
        return 2;
    }
    try {
        if (lazy_only) return lazy_boundary_diagnostic();
        const Counts c = run();
        std::cout << "{\"schema\":\"parseatlas_p2_field_correctness_v1\","
                  << "\"status\":\"pass\",\"mismatches\":0,"
                  << "\"finite_corpus_only\":true,\"timing_claim\":false,"
                  << "\"constant_time_claim\":false,\"lazy_magnitude_claim\":false,"
                  << "\"oracle\":\"Boost cpp_int residues and independent Euclidean inverse\","
                  << "\"routes\":[\"FE64 fast API\",\"FE52 canonical resident\","
                  << "\"FE52 per-operation canonical bridge\",\"existing CT field API\"],"
                  << "\"extra_inverse_route\":\"FE52 fixed addition chain\","
                  << "\"fe52_normalization\":\"resident explicit after every operation; bridge raw arithmetic then one to_fe normalization\","
                  << "\"fe52_subtraction\":\"canonical a + canonical b.negate(1), then normalize\","
                  << "\"raw_checks\":\"FE64 canonical raw 4x64; resident FE52 canonical 4x52+top48 before conversion; bridge raw residue; all 32 BE bytes\","
                  << "\"zero_inverse_contracts\":{\"FE64_native\":\"throws runtime_error\","
                  << "\"FE52_direct_safegcd\":\"zero\",\"FE52_fixed_chain\":\"zero\",\"CT_API_observed\":\"zero\"},"
                  << "\"seed_hex\":\"" << hex64(kSeed) << "\","
                  << "\"boundary_entries\":" << c.boundary_entries << ','
                  << "\"boundary_entries_may_repeat\":true,"
                  << "\"construction_cases\":" << c.constructions << ','
                  << "\"boundary_pairs\":" << c.boundary_pairs << ','
                  << "\"random_pairs\":" << c.random_pairs << ','
                  << "\"p1_regression_pairs\":" << c.regression_pairs << ','
                  << "\"binary_cases\":" << c.binary_cases << ','
                  << "\"binary_operations_per_route\":" << 4 * c.binary_cases << ','
                  << "\"boundary_inverses_nonzero\":" << c.boundary_inverses << ','
                  << "\"random_inverses_nonzero\":" << c.random_inverses << ','
                  << "\"inverse_cases_per_route_nonzero\":" << c.inverse_cases << ','
                  << "\"zero_inverse_cases\":" << c.zero_inverse_cases << ','
                  << "\"chain_cases\":" << c.chain_cases << ','
                  << "\"chain_steps_per_route\":" << c.chain_steps << ','
                  << "\"chain_operations\":[\"add\",\"sub\",\"mul\",\"square\"],"
                  << "\"fe64_compound_cases\":" << c.fe64_compound_cases << ','
                  << "\"fe64_self_alias_cases\":" << c.fe64_self_alias_cases << ','
                  << "\"fe52_compound_cases\":" << c.fe52_compound_cases << ','
                  << "\"fe52_self_alias_cases\":" << c.fe52_self_alias_cases << ','
                  << "\"input_preservation_checks\":" << c.input_preservation_checks << ','
                  << "\"fe64_output_checks\":" << c.fe64_output_checks << ','
                  << "\"fe52_output_checks\":" << c.fe52_output_checks << ','
                  << "\"bridge_raw_residue_checks\":" << c.bridge_raw_residue_checks << ','
                  << "\"fe64_negative_controls\":" << c.fe64_negative_controls << ','
                  << "\"fe52_negative_controls\":" << c.fe52_negative_controls << ','
                  << "\"assertions\":" << c.assertions << ','
                  << "\"checksum_fnv1a64\":\"" << hex64(c.checksum) << "\"}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cout << "{\"schema\":\"parseatlas_p2_field_correctness_v1\","
                  << "\"status\":\"fail\",\"error\":\"" << json_escape(error.what()) << "\"}\n";
        return 1;
    }
}
