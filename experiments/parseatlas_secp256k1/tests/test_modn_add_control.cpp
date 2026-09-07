#include "../probes/modn_add_control.hpp"
#include "secp256k1/scalar.hpp"

#include <boost/multiprecision/cpp_int.hpp>

#include <array>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdexcept>

namespace {

using pa_modn::Limbs;
using pa_modn::Method;
using pa_modn::Word;
using boost::multiprecision::cpp_int;
using secp256k1::fast::Scalar;
using Bytes = std::array<std::uint8_t, 32>;

constexpr Word seed = 20260905;
constexpr Word random_case_count = 100000;
constexpr Word maximum = ~Word{0};
const cpp_int modulus{"0xfffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141"};
const cpp_int width = cpp_int{1} << 256;

void require(bool condition, const char* message) {
    // Unlike assert(), correctness gates remain active under -DNDEBUG.
    if (!condition) throw std::runtime_error(message);
}

cpp_int to_bigint(const Limbs& value) {
    cpp_int result = 0;
    for (unsigned i = 4; i != 0; --i) {
        result <<= 64;
        result += value[i - 1];
    }
    return result;
}

Limbs to_limbs(cpp_int value) {
    require(value >= 0 && value < width, "test conversion outside 256-bit domain");
    const cpp_int mask = (cpp_int{1} << 64) - 1;
    Limbs result{};
    for (auto& limb : result) {
        limb = (value & mask).convert_to<Word>();
        value >>= 64;
    }
    return result;
}

Bytes independent_le(const Limbs& value) {
    // Independently defined byte comparison: never struct padding or a digest.
    Bytes bytes{};
    for (unsigned i = 0; i < 32; ++i) {
        bytes[i] = static_cast<std::uint8_t>(value[i / 8] >> (8 * (i % 8)));
    }
    return bytes;
}

bool same_bytes(const Limbs& left, const Limbs& right) {
    return independent_le(left) == independent_le(right);
}

Limbs oracle_add(const Limbs& a, const Limbs& b) {
    return to_limbs((to_bigint(a) + to_bigint(b)) % modulus);
}

Word next_random(Word& state) {
    state += UINT64_C(0x9e3779b97f4a7c15);
    Word z = state;
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

Limbs random_canonical(Word& state) {
    Limbs value{};
    do {
        for (auto& limb : value) limb = next_random(state);
    } while (to_bigint(value) >= modulus);
    return value;
}

struct Variant {
    Limbs (*add)(const Limbs&, const Limbs&) noexcept;
    void (*assign)(Limbs&, const Limbs&) noexcept;
};

constexpr std::array<Variant, 3> variants{{
    {pa_modn::add<Method::Original>, pa_modn::add_assign<Method::Original>},
    {pa_modn::add<Method::Blocked2>, pa_modn::add_assign<Method::Blocked2>},
    {pa_modn::add<Method::GpRecomputed>, pa_modn::add_assign<Method::GpRecomputed>}
}};

struct Counts {
    Word fixture_cases = 0;
    Word boundary_values = 0;
    Word boundary_pair_cases = 0;
    Word carry_pattern_cases = 0;
    Word carry_pattern_rejected = 0;
    Word targeted_cases = 0;
    Word no_reduction_cases = 0;
    Word reduction_no_wrap_cases = 0;
    Word reduction_wrap_cases = 0;
    Word reference_add_checks = 0;
    Word reference_assign_checks = 0;
    Word reference_selfalias_checks = 0;
    Word candidate_add_checks = 0;
    Word candidate_assign_checks = 0;
    Word candidate_selfalias_checks = 0;
    Word result_canonical_checks = 0;
    Word candidate_encoding_checks = 0;
    Word input_preservation_checks = 0;
    Word public_be_checks = 0;
    Word predicate_valid_checks = 0;
    Word predicate_invalid_checks = 0;
    Word recurrence_chains = 0;
    Word recurrence_steps = 0;
    Word reference_recurrence_checks = 0;
    Word candidate_recurrence_checks = 0;
    Word negative_comparator_checks = 0;
    Word checksum = UINT64_C(14695981039346656037);
};

void check_predicate(const Limbs& value, Counts& counts) {
    const bool expected = to_bigint(value) < modulus;
    require(pa_modn::canonical(value) == expected, "canonical predicate differs from bigint");
    if (expected) ++counts.predicate_valid_checks;
    else ++counts.predicate_invalid_checks;
}

void check_result(const Limbs& actual, const Limbs& expected, Counts& counts) {
    require(same_bytes(actual, expected), "full 32-byte result differs from oracle");
    require(to_bigint(actual) < modulus && pa_modn::canonical(actual),
            "modular result is not canonical");
    ++counts.result_canonical_checks;
    require(pa_modn::encode_le(actual) == independent_le(expected),
            "candidate encoder differs from independently encoded oracle");
    ++counts.candidate_encoding_checks;
}

void check_public_be(const Scalar& actual, const Limbs& expected, Counts& counts) {
    // scalar.cpp::write_bytes stores limbs 3,2,1,0 as BE64. Do not assume LE API.
    Bytes expected_be{};
    const auto expected_le = independent_le(expected);
    for (unsigned i = 0; i < 32; ++i) expected_be[i] = expected_le[31 - i];
    require(actual.to_bytes() == expected_be, "public big-endian bytes differ from oracle");
    ++counts.public_be_checks;
}

void digest_result(const Limbs& result, unsigned variant, Counts& counts) {
    counts.checksum ^= variant;
    counts.checksum *= UINT64_C(1099511628211);
    for (const auto byte : independent_le(result)) {
        counts.checksum ^= byte;
        counts.checksum *= UINT64_C(1099511628211);
    }
}

void check_case(const Limbs& fixture_a, const Limbs& fixture_b, Counts& counts) {
    require(to_bigint(fixture_a) < modulus && to_bigint(fixture_b) < modulus,
            "generated noncanonical arithmetic input");
    check_predicate(fixture_a, counts);
    check_predicate(fixture_b, counts);
    const cpp_int raw = to_bigint(fixture_a) + to_bigint(fixture_b);
    if (raw < modulus) ++counts.no_reduction_cases;
    else if (raw < width) ++counts.reduction_no_wrap_cases;
    else ++counts.reduction_wrap_cases;
    const Limbs expected = to_limbs(raw % modulus);
    const Limbs doubled = oracle_add(fixture_a, fixture_a);

    Scalar library_a = Scalar::from_limbs(fixture_a);
    Scalar library_b = Scalar::from_limbs(fixture_b);
    require(same_bytes(library_a.limbs(), fixture_a) &&
            same_bytes(library_b.limbs(), fixture_b), "library normalized canonical fixture");
    const Scalar reference = library_a + library_b;
    check_result(reference.limbs(), expected, counts);
    check_public_be(reference, expected, counts);
    ++counts.reference_add_checks;
    require(same_bytes(library_a.limbs(), fixture_a) &&
            same_bytes(library_b.limbs(), fixture_b), "library add mutated an input");
    counts.input_preservation_checks += 2;

    Scalar library_assigned = library_a;
    require(&(library_assigned += library_b) == &library_assigned,
            "library += returned the wrong object");
    check_result(library_assigned.limbs(), expected, counts);
    ++counts.reference_assign_checks;
    require(same_bytes(library_b.limbs(), fixture_b), "library += mutated RHS");
    ++counts.input_preservation_checks;
    Scalar library_self = library_a;
    library_self += library_self;
    check_result(library_self.limbs(), doubled, counts);
    ++counts.reference_selfalias_checks;

    for (unsigned i = 0; i < variants.size(); ++i) {
        Limbs a = fixture_a;
        Limbs b = fixture_b;
        const Limbs actual = variants[i].add(a, b);
        check_result(actual, expected, counts);
        require(same_bytes(actual, reference.limbs()), "candidate differs from real library +");
        ++counts.candidate_add_checks;
        require(same_bytes(a, fixture_a) && same_bytes(b, fixture_b),
                "candidate add mutated an input");
        counts.input_preservation_checks += 2;
        variants[i].assign(a, b);
        check_result(a, expected, counts);
        require(same_bytes(a, library_assigned.limbs()), "candidate differs from real library +=");
        ++counts.candidate_assign_checks;
        require(same_bytes(b, fixture_b), "candidate add_assign mutated RHS");
        ++counts.input_preservation_checks;
        Limbs self = fixture_a;
        variants[i].assign(self, self);
        check_result(self, doubled, counts);
        require(same_bytes(self, library_self.limbs()), "candidate selfalias differs from library");
        ++counts.candidate_selfalias_checks;
        digest_result(actual, i, counts);
    }
    ++counts.fixture_cases;
}

void check_boundaries(Counts& counts) {
    // Deduplicate this finite scalar set, then execute every ordered pair.
    // Other fixture families overlap it; aggregate counts are executions only.
    std::set<Limbs> values;
    const auto insert = [&](const cpp_int& value) {
        if (value >= 0 && value < modulus) values.insert(to_limbs(value));
    };
    for (unsigned delta = 0; delta < 5; ++delta) {
        insert(cpp_int{delta});
        insert(modulus - 1 - delta);
        insert(modulus / 2 + delta);
        insert(modulus / 2 - delta);
    }
    constexpr std::array<unsigned, 12> bits{{1, 63, 64, 65, 127, 128, 129,
                                           191, 192, 193, 254, 255}};
    for (const unsigned bit : bits) {
        const cpp_int power = cpp_int{1} << bit;
        for (int delta = -1; delta <= 1; ++delta) {
            insert(power + delta);
            insert(modulus - power + delta);
        }
    }
    counts.boundary_values = values.size();
    for (const auto& a : values) for (const auto& b : values) {
        check_case(a, b, counts);
        ++counts.boundary_pair_cases;
    }
}

void check_carry_patterns(Counts& counts) {
    // Four representatives per kill/propagate/generate limb category. Reject
    // noncanonical operands; never call an unchecked modular API outside scope.
    constexpr std::array<std::array<Word, 2>, 12> pairs{{
        {{0, 0}}, {{maximum - 1, 0}}, {{0, maximum - 1}}, {{1, 1}},
        {{maximum, 0}}, {{0, maximum}}, {{maximum - 1, 1}}, {{1, maximum - 1}},
        {{maximum, 1}}, {{1, maximum}}, {{maximum, maximum}}, {{maximum - 1, 2}}
    }};
    for (unsigned code = 0; code < 12 * 12 * 12 * 12; ++code) {
        Limbs a{};
        Limbs b{};
        unsigned remaining = code;
        for (unsigned limb = 0; limb < 4; ++limb) {
            const auto& pair = pairs[remaining % 12];
            remaining /= 12;
            a[limb] = pair[0];
            b[limb] = pair[1];
        }
        check_predicate(a, counts);
        check_predicate(b, counts);
        if (to_bigint(a) >= modulus || to_bigint(b) >= modulus) {
            ++counts.carry_pattern_rejected;
            continue;
        }
        check_case(a, b, counts);
        ++counts.carry_pattern_cases;
    }
}

void check_targeted(Counts& counts) {
    const auto pair = [&](const cpp_int& a, const cpp_int& b) {
        if (a < 0 || a >= modulus || b < 0 || b >= modulus) return;
        check_case(to_limbs(a), to_limbs(b), counts);
        check_case(to_limbs(b), to_limbs(a), counts);
        counts.targeted_cases += 2;
    };
    for (unsigned bit = 0; bit < 256; ++bit) {
        const cpp_int power = cpp_int{1} << bit;
        // Full lower-bit carry chains, and the n-1/n/n+1 reduction boundary.
        pair(power - 1, cpp_int{1});
        for (int delta = -1; delta <= 1; ++delta) pair(power, modulus - power + delta);
    }
    for (const unsigned bit : std::array<unsigned, 7>{{0, 1, 2, 63, 64, 127, 128}}) {
        const cpp_int a = modulus - (cpp_int{1} << bit);
        // Raw sum 2^256-1/2^256/2^256+1, including subtraction borrow chains.
        for (int delta = -1; delta <= 1; ++delta) pair(a, width - a + delta);
    }
}

void check_domain_gate(Counts& counts) {
    require(to_bigint(pa_modn::order) == modulus, "candidate order constant differs from secp256k1 n");
    for (int delta = -32; delta <= 32; ++delta) check_predicate(to_limbs(modulus + delta), counts);
    check_predicate(to_limbs(cpp_int{0}), counts);
    check_predicate(to_limbs(width - 1), counts);
    for (unsigned bit = 0; bit < 256; ++bit) {
        const cpp_int power = cpp_int{1} << bit;
        check_predicate(to_limbs(power), counts);
        check_predicate(to_limbs(width - power), counts);
        if (modulus + power < width) check_predicate(to_limbs(modulus + power), counts);
    }
}

void check_recurrences(Counts& counts) {
    Word random = seed ^ UINT64_C(0xd1b54a32d192ed03);
    for (unsigned chain = 0; chain < 128; ++chain) {
        Limbs expected = random_canonical(random);
        if (chain == 0) expected = {};
        if (chain == 1) expected = to_limbs(modulus - 1);
        Scalar reference = Scalar::from_limbs(expected);
        std::array<Limbs, variants.size()> states{};
        states.fill(expected);
        for (unsigned step = 0; step < 257; ++step) {
            const bool selfalias = step % 7 == 0;
            Limbs rhs = random_canonical(random);
            if (step % 4 == 0) rhs = to_limbs(modulus - 1);
            if (step % 4 == 1) rhs = Limbs{1, 0, 0, 0};
            const Limbs saved_rhs = rhs;
            expected = oracle_add(expected, selfalias ? expected : rhs);
            const Scalar library_rhs = Scalar::from_limbs(rhs);
            if (selfalias) reference += reference;
            else reference += library_rhs;
            check_result(reference.limbs(), expected, counts);
            ++counts.reference_recurrence_checks;
            require(same_bytes(library_rhs.limbs(), saved_rhs), "library recurrence changed RHS");
            ++counts.input_preservation_checks;
            for (unsigned i = 0; i < variants.size(); ++i) {
                if (selfalias) variants[i].assign(states[i], states[i]);
                else variants[i].assign(states[i], rhs);
                check_result(states[i], expected, counts);
                require(same_bytes(states[i], reference.limbs()),
                        "candidate internal recurrence state differs from library");
                require(same_bytes(rhs, saved_rhs), "candidate recurrence changed RHS");
                ++counts.input_preservation_checks;
                ++counts.candidate_recurrence_checks;
                digest_result(states[i], i, counts);
            }
            ++counts.recurrence_steps;
        }
        ++counts.recurrence_chains;
    }
}

void check_negative_comparator(Counts& counts) {
    // Every result bit must affect the full-byte comparator, including high limbs.
    const Limbs zero{};
    require(same_bytes(zero, zero), "comparator rejects identical value");
    for (unsigned bit = 0; bit < 256; ++bit) {
        Limbs corrupted{};
        corrupted[bit / 64] = Word{1} << (bit % 64);
        require(!same_bytes(zero, corrupted) && !same_bytes(corrupted, zero),
                "comparator missed a deliberate result-bit corruption");
        ++counts.negative_comparator_checks;
    }
    // This all-distinct byte fixture catches a shared endian/limb reversal bug
    // in the candidate encoder even if the arithmetic fixtures happen to repeat.
    Limbs pattern{};
    Bytes expected{};
    for (unsigned byte = 0; byte < 32; ++byte) {
        expected[byte] = static_cast<std::uint8_t>(byte + 1);
        pattern[byte / 8] |= Word{byte + 1} << (8 * (byte % 8));
    }
    require(independent_le(pattern) == expected && pa_modn::encode_le(pattern) == expected,
            "explicit byte-order fixture failed");
}

} // namespace

int main(int argc, char**) {
    if (argc != 1) {
        std::cerr << "usage: test_modn_add_control (no arguments)\n";
        return 2;
    }
    try {
        Counts c;
        check_domain_gate(c);
        check_negative_comparator(c);
        check_boundaries(c);
        check_carry_patterns(c);
        check_targeted(c);
        Word random = seed;
        for (Word i = 0; i < random_case_count; ++i) {
            const Limbs a = random_canonical(random);
            const Limbs b = random_canonical(random);
            check_case(a, b, c);
        }
        check_recurrences(c);
        require(c.no_reduction_cases && c.reduction_no_wrap_cases && c.reduction_wrap_cases,
                "one modular reduction domain was not exercised");
        require(c.fixture_cases == c.boundary_pair_cases + c.carry_pattern_cases +
                c.targeted_cases + random_case_count, "fixture counts do not reconcile");
        std::cout << "{\n"
                  << "  \"schema\": \"pa_modn_add_control_correctness_v1\",\n"
                  << "  \"status\": \"pass\",\n"
                  << "  \"seed\": " << seed << ",\n"
                  << "  \"methods\": 3,\n"
                  << "  \"random_canonical_pair_cases\": " << random_case_count << ",\n"
                  << "  \"fixture_cases\": " << c.fixture_cases << ",\n"
                  << "  \"fixture_counts_are_unique_inputs\": false,\n"
                  << "  \"boundary_values_deduplicated\": true,\n"
                  << "  \"boundary_values\": " << c.boundary_values << ",\n"
                  << "  \"boundary_pair_cases\": " << c.boundary_pair_cases << ",\n"
                  << "  \"carry_pattern_cases\": " << c.carry_pattern_cases << ",\n"
                  << "  \"carry_pattern_rejected_noncanonical\": " << c.carry_pattern_rejected << ",\n"
                  << "  \"targeted_cases\": " << c.targeted_cases << ",\n"
                  << "  \"fixture_domains\": {\"no_reduction\": " << c.no_reduction_cases
                  << ", \"reduction_no_wrap\": " << c.reduction_no_wrap_cases
                  << ", \"reduction_wrap\": " << c.reduction_wrap_cases << "},\n"
                  << "  \"reference_add_checks\": " << c.reference_add_checks << ",\n"
                  << "  \"reference_assign_checks\": " << c.reference_assign_checks << ",\n"
                  << "  \"reference_selfalias_checks\": " << c.reference_selfalias_checks << ",\n"
                  << "  \"candidate_add_checks\": " << c.candidate_add_checks << ",\n"
                  << "  \"candidate_assign_checks\": " << c.candidate_assign_checks << ",\n"
                  << "  \"candidate_selfalias_checks\": " << c.candidate_selfalias_checks << ",\n"
                  << "  \"result_canonical_checks\": " << c.result_canonical_checks << ",\n"
                  << "  \"candidate_encoding_checks\": " << c.candidate_encoding_checks << ",\n"
                  << "  \"input_preservation_checks\": " << c.input_preservation_checks << ",\n"
                  << "  \"public_be_checks\": " << c.public_be_checks << ",\n"
                  << "  \"predicate_valid_checks\": " << c.predicate_valid_checks << ",\n"
                  << "  \"predicate_invalid_checks\": " << c.predicate_invalid_checks << ",\n"
                  << "  \"recurrence_chains\": " << c.recurrence_chains << ",\n"
                  << "  \"recurrence_steps\": " << c.recurrence_steps << ",\n"
                  << "  \"reference_recurrence_checks\": " << c.reference_recurrence_checks << ",\n"
                  << "  \"candidate_recurrence_checks\": " << c.candidate_recurrence_checks << ",\n"
                  << "  \"negative_comparator_checks\": " << c.negative_comparator_checks << ",\n"
                  << "  \"oracle\": \"boost::multiprecision::cpp_int_mod_independent_n\",\n"
                  << "  \"library_reference\": \"linked_unchanged_fast::Scalar_operator+_operator+=\",\n"
                  << "  \"comparison\": \"all_32_explicit_le_bytes_and_public_be_result\",\n"
                  << "  \"mismatches\": 0,\n"
                  << "  \"result_checksum_fnv1a64\": \"" << std::hex << std::setw(16)
                  << std::setfill('0') << c.checksum << std::dec << "\",\n"
                  << "  \"negative_controls\": \"pass\",\n"
                  << "  \"timing_claim\": false,\n"
                  << "  \"constant_time_claim\": false\n}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "mod-n correctness failure: " << error.what() << '\n';
        return 1;
    }
}
