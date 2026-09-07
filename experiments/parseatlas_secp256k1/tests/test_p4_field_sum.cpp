// P4 independent finite C++ oracle corpus; not a benchmark or CT certification.
// Production reference prefixes and Boost prefixes are computed once per real
// RHS corpus. Candidate jobs always consume the requested actual N-entry prefix.
#include "../probes/p4_field_sum_kernels.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#if defined(PA_P4_INVALID_ZERO)
[[maybe_unused]] pa_p4::FE invalid_zero_probe(const pa_p4::FE& seed) {
    return pa_p4::sum_fe64_wide<0, 1>(seed, nullptr, 0);
}
#endif
#if defined(PA_P4_INVALID_CHUNK)
[[maybe_unused]] pa_p4::FE invalid_chunk_probe(const pa_p4::FE& seed) {
    return pa_p4::sum_fe64_wide<4096, 1>(seed, nullptr, 0);
}
#endif
#if defined(PA_P4_INVALID_LANES)
[[maybe_unused]] pa_p4::FE invalid_lanes_probe(const pa_p4::FE& seed) {
    return pa_p4::sum_fe64_wide<16, 2>(seed, nullptr, 0);
}
#endif

namespace {
using boost::multiprecision::cpp_int;
using FE = pa_p4::FE;
using Bytes = std::array<std::uint8_t, 32>;
using Limbs = FE::limbs_type;
using U128 = pa_p4::detail::u128;
using Columns = pa_p4::detail::Columns;
using Wide = pa_p4::detail::Wide;
const cpp_int kB = cpp_int(1) << 256;
const cpp_int kK = (cpp_int(1) << 32) + 977;
const cpp_int kP = kB - kK;
constexpr std::uint64_t kSeed = UINT64_C(0x5041345f53554d53);
constexpr std::size_t kReplayLimit = 8193;
constexpr std::array<std::size_t, 18> kSizes{{
    0, 1, 2, 3, 4, 5, 15, 16, 17, 4093, 4094, 4095,
    4096, 4097, 8190, 8191, 8192, 8193}};
static_assert(sizeof(FE) == 32 && std::is_trivially_copyable_v<FE>);
static_assert(sizeof(U128) == 16);

struct Counts {
    std::uint64_t assertions = 0, corpora = 0, fixtures = 0, large_fixtures = 0;
    std::uint64_t rhs_objects = 0, nonzero_rhs_objects = 0, input_object_checks = 0;
    std::uint64_t reference_prefix_additions = 0, reference_prefix_checks = 0;
    std::uint64_t reference_seed_additions = 0, seeded_cases = 0, alias_cases = 0;
    std::uint64_t nonzero_seed_cases = 0, zero_residue_cases = 0;
    std::uint64_t inline_calls = 0, wide_calls = 0, null_identity_calls = 0;
    std::uint64_t second_oracle_checks = 0, input_preservation_checks = 0;
    std::uint64_t phase_replays = 0, replay_chunks = 0, bank_u128_checks = 0;
    std::uint64_t merged_u128_checks = 0, propagated_highword_checks = 0;
    std::uint64_t first_fold_checks = 0, first_fold_carry_cases = 0, second_fold_checks = 0;
    std::uint64_t canonical_correction_checks = 0, canonical_subtraction_cases = 0;
    std::uint64_t output_checks = 0, negative_controls = 0;
    std::uint64_t input_checksum = UINT64_C(14695981039346656037);
    std::uint64_t output_checksum = UINT64_C(14695981039346656037);
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

Bytes encode_be(cpp_int value) {
    if (value < 0 || value >= kB) throw std::runtime_error("oracle BE range");
    Bytes result{};
    for (std::size_t i = result.size(); i != 0; --i) {
        result[i - 1] = static_cast<std::uint8_t>(value & 255);
        value >>= 8;
    }
    return result;
}

Limbs encode_limbs(cpp_int value) {
    if (value < 0 || value >= kB) throw std::runtime_error("oracle limb range");
    const cpp_int mask = (cpp_int(1) << 64) - 1;
    Limbs result{};
    for (auto& limb : result) {
        limb = static_cast<std::uint64_t>(value & mask);
        value >>= 64;
    }
    return result;
}

cpp_int decode_limbs(const Limbs& limbs) {
    cpp_int result = 0;
    for (std::size_t i = 4; i != 0; --i) {
        result *= cpp_int(1) << 64;
        result += limbs[i - 1];
    }
    return result;
}

cpp_int decode_u128(U128 value) {
    return (cpp_int(static_cast<std::uint64_t>(value >> 64)) << 64) +
        static_cast<std::uint64_t>(value);
}

cpp_int decode_wide(const Wide& value) {
    return decode_limbs(value.low) + (cpp_int(value.high) << 256);
}

std::string hex_bytes(const Bytes& bytes) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (auto byte : bytes) out << std::setw(2) << static_cast<unsigned>(byte);
    return out.str();
}

std::string hex_limbs(const Limbs& limbs) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << '[';
    for (std::size_t i = 0; i < 4; ++i) {
        if (i) out << ',';
        out << std::setw(16) << limbs[i];
    }
    out << ']';
    return out.str();
}

void mix(const Bytes& bytes, std::uint64_t& checksum) {
    for (auto byte : bytes) checksum = (checksum ^ byte) * UINT64_C(1099511628211);
}

bool matches(const Bytes& bytes, const Limbs& limbs, const cpp_int& expected) {
    bool result = expected >= 0 && expected < kP;
    const auto eb = encode_be(expected);
    const auto el = encode_limbs(expected);
    for (std::size_t i = 0; i < 32; ++i) result &= bytes[i] == eb[i];
    for (std::size_t i = 0; i < 4; ++i) result &= limbs[i] == el[i];
    result &= decode_limbs(limbs) < kP;
    return result;
}

void check_output(const FE& actual, const cpp_int& expected, Counts& c,
                  const std::string& context) {
    const Limbs raw = actual.limbs();
    const Bytes bytes = actual.to_bytes();
    if (!matches(bytes, raw, expected)) {
        require(false, c, context + " actual_be=" + hex_bytes(bytes) +
                " raw_le=" + hex_limbs(raw) + " expected_be=" + hex_bytes(encode_be(expected)));
    }
    require(actual.limbs() == raw, c, context + ": serialization mutated output");
    ++c.output_checks;
    mix(bytes, c.output_checksum);
}

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

enum class Pattern { Zero, ZeroNonzeroSeed, One, MinusOne, Witness, Limbs, Cancel, Random };
constexpr std::array<Pattern, 8> kPatterns{{Pattern::Zero, Pattern::ZeroNonzeroSeed,
    Pattern::One, Pattern::MinusOne, Pattern::Witness, Pattern::Limbs, Pattern::Cancel, Pattern::Random}};

const char* pattern_name(Pattern pattern) {
    switch (pattern) {
        case Pattern::Zero: return "zero";
        case Pattern::ZeroNonzeroSeed: return "zero_nonzero_seed";
        case Pattern::One: return "one";
        case Pattern::MinusOne: return "p_minus_one";
        case Pattern::Witness: return "P2_witness";
        case Pattern::Limbs: return "limb_boundaries";
        case Pattern::Cancel: return "seeded_cancellation";
        case Pattern::Random: return "random";
    }
    throw std::runtime_error("unknown pattern");
}

cpp_int witness_value() {
    return ((cpp_int(1) << 52) - 1) + (((cpp_int(1) << 48) - 1) << 208);
}

std::vector<cpp_int> limb_patterns() {
    std::vector<cpp_int> values{cpp_int(0), cpp_int(1), kP - 1, kP, kP + 1, kB - 1,
                               witness_value(), kB - 1 - (cpp_int(1) << 33)};
    // All single-bit positions and their neighbors include every 64-bit carry
    // boundary, as well as the old FE52 representation boundaries.
    for (unsigned bit = 0; bit < 256; ++bit) {
        const cpp_int power = cpp_int(1) << bit;
        values.push_back(power - 1);
        values.push_back(power);
        values.push_back(power + 1);
    }
    return values;
}

struct Prefix {
    std::size_t count;
    cpp_int integer_sum;
    FE reference;
};

struct Corpus {
    FE seed;
    cpp_int seed_int;
    std::vector<FE> rhs;
    std::vector<FE> before;
    std::vector<Prefix> prefixes;
};

Corpus make_corpus(Pattern pattern, Counts& c) {
    Corpus f;
    SplitMix64 random{kSeed ^ (static_cast<std::uint64_t>(pattern) << 56)};
    switch (pattern) {
        case Pattern::Zero: f.seed_int = 0; break;
        case Pattern::ZeroNonzeroSeed: f.seed_int = kP - 2; break;
        case Pattern::One: f.seed_int = 7; break;
        case Pattern::MinusOne: f.seed_int = kP - 1; break;
        case Pattern::Witness: f.seed_int = witness_value(); break;
        case Pattern::Limbs: f.seed_int = (cpp_int(1) << 192) - 1; break;
        case Pattern::Cancel: f.seed_int = 1; break;
        case Pattern::Random:
            f.seed_int = mod(random.raw256());
            if (f.seed_int == 0) f.seed_int = 1;
            break;
    }
    f.seed = FE::from_bytes(encode_be(f.seed_int));
    require(matches(f.seed.to_bytes(), f.seed.limbs(), f.seed_int), c, "seed construction");
    mix(f.seed.to_bytes(), c.input_checksum);
    std::vector<std::size_t> sizes(kSizes.begin(), kSizes.end());
    if (pattern == Pattern::Witness || pattern == Pattern::Random) sizes.push_back(65536);
    if (pattern == Pattern::Random) sizes.push_back(1048576);
    f.rhs.reserve(sizes.back());
    f.prefixes.push_back(Prefix{0, cpp_int(0), FE::zero()});
    ++c.reference_prefix_checks;
    const auto boundaries = limb_patterns();
    cpp_int exact_sum = 0;
    FE reference = FE::zero();
    std::size_t next_prefix = 1;
    for (std::size_t i = 0; i < sizes.back(); ++i) {
        cpp_int raw;
        switch (pattern) {
            case Pattern::Zero: case Pattern::ZeroNonzeroSeed: raw = 0; break;
            case Pattern::One: raw = 1; break;
            case Pattern::MinusOne: raw = kP - 1; break;
            case Pattern::Witness: raw = witness_value(); break;
            case Pattern::Limbs: raw = boundaries[i % boundaries.size()]; break;
            case Pattern::Cancel: raw = (i == 0 ? kP - 1 : (i % 2 ? cpp_int(5) : kP - 5)); break;
            case Pattern::Random: raw = random.raw256(); break;
        }
        const cpp_int value_int = mod(raw);
        const FE value = FE::from_bytes(encode_be(raw));
        require(matches(value.to_bytes(), value.limbs(), value_int), c, "RHS canonical construction");
        f.rhs.push_back(value);
        exact_sum += value_int;
        reference = reference + value;  // Actual unchanged corrected FE64 API.
        ++c.reference_prefix_additions;
        ++c.rhs_objects;
        ++c.input_object_checks;
        if (value_int != 0) ++c.nonzero_rhs_objects;
        mix(value.to_bytes(), c.input_checksum);
        if (i + 1 == sizes[next_prefix]) {
            check_output(reference, mod(exact_sum), c, "corrected FE64 prefix " + std::to_string(i + 1));
            f.prefixes.push_back(Prefix{i + 1, exact_sum, reference});
            ++c.reference_prefix_checks;
            ++next_prefix;
        }
    }
    require(next_prefix == sizes.size(), c, "all requested oracle prefixes generated");
    f.before = f.rhs;
    ++c.corpora;
    return f;
}

void preserve(const Corpus& f, const FE& seed, const Limbs& seed_before,
              Counts& c, const std::string& label) {
    // Compare the whole allocated corpus, including the unconsumed tail of a
    // short prefix. No positive-length null test or cross-type alias is used.
    const bool unchanged = f.rhs.empty() ||
        std::memcmp(f.rhs.data(), f.before.data(), f.rhs.size() * sizeof(FE)) == 0;
    require(unchanged && seed.limbs() == seed_before, c, label + ": input bytes changed");
    ++c.input_preservation_checks;
}

template <std::size_t Chunk, std::size_t Lanes>
void test_wide_route(const Corpus& f, const Prefix& prefix, const FE& seed,
                     const FE& reference, const cpp_int& expected, Counts& c,
                     const std::string& label) {
    const Limbs seed_before = seed.limbs();
    const FE* rhs = prefix.count == 0 ? nullptr : f.rhs.data();
    const FE result = pa_p4::sum_fe64_wide<Chunk, Lanes>(seed, rhs, prefix.count);
    const std::string at = label + " chunk" + std::to_string(Chunk) + " lanes" + std::to_string(Lanes);
    check_output(result, expected, c, at);
    require(result.limbs() == reference.limbs() && result.to_bytes() == reference.to_bytes(), c,
            at + ": corrected FE64 second oracle");
    ++c.second_oracle_checks;
    ++c.wide_calls;
    if (prefix.count == 0) ++c.null_identity_calls;
    preserve(f, seed, seed_before, c, at);
}

void test_seed_case(const Corpus& f, const Prefix& prefix, const FE& seed,
                    const cpp_int& seed_int, bool alias, Counts& c, const std::string& label) {
    const cpp_int expected = mod(seed_int + prefix.integer_sum);
    const FE reference = seed + prefix.reference;
    ++c.reference_seed_additions;
    check_output(reference, expected, c, label + " reference seed plus prefix");
    const Limbs seed_before = seed.limbs();
    const FE* rhs = prefix.count == 0 ? nullptr : f.rhs.data();
    const FE eager = pa_p4::sum_fe64_inline(seed, rhs, prefix.count);
    check_output(eager, expected, c, label + " inline eager");
    require(eager.limbs() == reference.limbs() && eager.to_bytes() == reference.to_bytes(), c,
            label + ": eager corrected FE64 second oracle");
    ++c.second_oracle_checks;
    ++c.inline_calls;
    if (prefix.count == 0) ++c.null_identity_calls;
    preserve(f, seed, seed_before, c, label + " eager");
    test_wide_route<1, 1>(f, prefix, seed, reference, expected, c, label);
    test_wide_route<1, 4>(f, prefix, seed, reference, expected, c, label);
    test_wide_route<16, 1>(f, prefix, seed, reference, expected, c, label);
    test_wide_route<16, 4>(f, prefix, seed, reference, expected, c, label);
    test_wide_route<4094, 1>(f, prefix, seed, reference, expected, c, label);
    test_wide_route<4094, 4>(f, prefix, seed, reference, expected, c, label);
    test_wide_route<4095, 1>(f, prefix, seed, reference, expected, c, label);
    test_wide_route<4095, 4>(f, prefix, seed, reference, expected, c, label);
    if (alias) ++c.alias_cases; else ++c.seeded_cases;
    if (seed_int != 0) ++c.nonzero_seed_cases;
    if (expected == 0) ++c.zero_residue_cases;
}

template <std::size_t Chunk, std::size_t Lanes>
void replay(const Corpus& f, const Prefix& prefix, Counts& c, const std::string& label) {
    ++c.phase_replays;
    Limbs state = f.seed.limbs();
    cpp_int prefix_int = f.seed_int;
    std::size_t offset = 0;
    while (offset < prefix.count) {
        const std::size_t remaining = prefix.count - offset;
        const std::size_t count = remaining > Chunk ? Chunk : remaining;
        const std::string at = label + " chunk" + std::to_string(Chunk) +
            " lanes" + std::to_string(Lanes) + " offset" + std::to_string(offset);
        require(decode_limbs(state) == mod(prefix_int), c, at + ": canonical prefix entry");
        const auto banks = pa_p4::detail::accumulate_banks<Lanes>(f.rhs.data() + offset, count);
        std::array<std::array<cpp_int, 4>, Lanes> expected_banks{};
        cpp_int local_exact = mod(prefix_int);
        for (std::size_t i = 0; i < count; ++i) {
            const auto& limbs = f.rhs[offset + i].limbs();
            for (std::size_t j = 0; j < 4; ++j) expected_banks[i % Lanes][j] += limbs[j];
            const cpp_int value = decode_limbs(limbs);
            local_exact += value;
            prefix_int += value;
        }
        for (std::size_t lane = 0; lane < Lanes; ++lane) {
            for (std::size_t limb = 0; limb < 4; ++limb) {
                require(decode_u128(banks[lane][limb]) == expected_banks[lane][limb], c,
                        at + ": exact raw u128 bank " + std::to_string(lane) + "/" + std::to_string(limb));
                ++c.bank_u128_checks;
            }
        }
        const Columns merged = pa_p4::detail::merge_columns<Lanes>(state, banks);
        cpp_int decoded_columns = 0;
        for (std::size_t limb = 0; limb < 4; ++limb) {
            cpp_int expected_column = state[limb];
            for (std::size_t lane = 0; lane < Lanes; ++lane) expected_column += expected_banks[lane][limb];
            require(decode_u128(merged[limb]) == expected_column && expected_column < (cpp_int(1) << 76), c,
                    at + ": merged u128 column " + std::to_string(limb));
            decoded_columns += expected_column << (64 * limb);
            ++c.merged_u128_checks;
        }
        require(decoded_columns == local_exact, c, at + ": ordinary-integer merged decode");
        const Wide propagated = pa_p4::detail::propagate_columns(merged);
        require(decode_wide(propagated) == local_exact && propagated.high <= 4095, c,
                at + ": propagated 256-bit low/high word mismatch");
        ++c.propagated_highword_checks;
        const Wide first = pa_p4::detail::fold_high(propagated);
        const cpp_int expected_first = decode_limbs(propagated.low) + cpp_int(propagated.high) * kK;
        require(decode_wide(first) == expected_first && first.high <= 1, c, at + ": first fold exactness");
        ++c.first_fold_checks;
        if (first.high != 0) ++c.first_fold_carry_cases;
        const Wide second = pa_p4::detail::fold_high(first);
        const cpp_int expected_second = decode_limbs(first.low) + cpp_int(first.high) * kK;
        require(decode_wide(second) == expected_second && second.high == 0, c, at + ": second fold exactness");
        ++c.second_fold_checks;
        if (expected_second >= kP) ++c.canonical_subtraction_cases;
        const Limbs corrected = pa_p4::detail::canonicalize_below_b(second.low);
        const cpp_int expected = mod(prefix_int);
        check_output(FE::from_limbs_raw(corrected), expected, c, at + ": one p correction");
        ++c.canonical_correction_checks;
        check_output(FE::from_limbs_raw(pa_p4::detail::reduce_wide(propagated)), expected, c,
                     at + ": actual reduce_wide helper");
        state = pa_p4::detail::normalize_columns(merged);
        check_output(FE::from_limbs_raw(state), expected, c, at + ": actual normalize_columns helper");
        offset += count;
        ++c.replay_chunks;
    }
    require(mod(prefix_int) == mod(f.seed_int + prefix.integer_sum), c, label + ": replay final oracle");
}

void run_corpus(Pattern pattern, Counts& c) {
    const Corpus f = make_corpus(pattern, c);
    const Limbs seed_snapshot = f.seed.limbs();
    for (const auto& prefix : f.prefixes) {
        const std::string label = std::string(pattern_name(pattern)) + " N=" + std::to_string(prefix.count);
        test_seed_case(f, prefix, f.seed, f.seed_int, false, c, label);
        if (prefix.count != 0 && prefix.count <= kReplayLimit) {
            const std::size_t index = prefix.count / 2;
            const FE& alias = f.rhs[index];
            test_seed_case(f, prefix, alias, decode_limbs(alias.limbs()), true, c,
                           label + " x0 aliases RHS[" + std::to_string(index) + ']');
        }
        if (prefix.count <= kReplayLimit) {
            replay<1, 1>(f, prefix, c, label);
            replay<1, 4>(f, prefix, c, label);
            replay<16, 1>(f, prefix, c, label);
            replay<16, 4>(f, prefix, c, label);
            replay<4094, 1>(f, prefix, c, label);
            replay<4094, 4>(f, prefix, c, label);
            replay<4095, 1>(f, prefix, c, label);
            replay<4095, 4>(f, prefix, c, label);
            preserve(f, f.seed, seed_snapshot, c, label + " after actual-helper replay");
        } else ++c.large_fixtures;
        ++c.fixtures;
    }
}

void first_fold_carry_witness(Counts& c) {
    // A random 256-bit corpus will almost never land within high*K of B.
    // This admissible 3-term sum is exactly 2B-1: first fold overflows once,
    // second fold returns 2K-1. Chunk1 splits it; larger chunks exercise it.
    Corpus f;
    f.seed_int = kP - 1;
    f.seed = FE::from_bytes(encode_be(f.seed_int));
    mix(f.seed.to_bytes(), c.input_checksum);
    FE reference = FE::zero();
    cpp_int integer_sum = 0;
    const std::array<cpp_int, 2> values{{kP - 1, 2 * kK + 1}};
    for (const auto& value : values) {
        const FE rhs = FE::from_bytes(encode_be(value));
        require(matches(rhs.to_bytes(), rhs.limbs(), value), c, "first-fold witness canonical input");
        f.rhs.push_back(rhs);
        integer_sum += value;
        reference = reference + rhs;
        ++c.reference_prefix_additions;
        ++c.rhs_objects;
        ++c.nonzero_rhs_objects;
        ++c.input_object_checks;
        mix(rhs.to_bytes(), c.input_checksum);
    }
    f.before = f.rhs;
    const Prefix prefix{2, integer_sum, reference};
    check_output(reference, mod(integer_sum), c, "first-fold witness FE64 prefix");
    ++c.reference_prefix_checks;
    require(f.seed_int + integer_sum == 2 * kB - 1 &&
            mod(f.seed_int + integer_sum) == 2 * kK - 1, c, "first-fold witness independent identity");
    test_seed_case(f, prefix, f.seed, f.seed_int, false, c, "first-fold carry witness");
    const Limbs seed_before = f.seed.limbs();
    const std::uint64_t prior_carries = c.first_fold_carry_cases;
    replay<1, 1>(f, prefix, c, "first-fold carry witness");
    replay<1, 4>(f, prefix, c, "first-fold carry witness");
    replay<16, 1>(f, prefix, c, "first-fold carry witness");
    replay<16, 4>(f, prefix, c, "first-fold carry witness");
    replay<4094, 1>(f, prefix, c, "first-fold carry witness");
    replay<4094, 4>(f, prefix, c, "first-fold carry witness");
    replay<4095, 1>(f, prefix, c, "first-fold carry witness");
    replay<4095, 4>(f, prefix, c, "first-fold carry witness");
    require(c.first_fold_carry_cases - prior_carries == 6, c, "first-fold witness exercised all six unsplit schedules");
    preserve(f, f.seed, seed_before, c, "first-fold carry witness after replay");
    ++c.corpora;
    ++c.fixtures;
}

void controls(Counts& c) {
    Bytes bytes{};
    for (std::size_t i = 0; i < 32; ++i) bytes[i] = static_cast<std::uint8_t>(i + 1);
    const cpp_int expected("0x0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20");
    const Limbs limbs{{UINT64_C(0x191a1b1c1d1e1f20), UINT64_C(0x1112131415161718),
                       UINT64_C(0x090a0b0c0d0e0f10), UINT64_C(0x0102030405060708)}};
    require(matches(bytes, limbs, expected), c, "known-endian positive control");
    for (std::size_t i = 0; i < 32; ++i) {
        auto bad = bytes; bad[i] ^= UINT8_C(0x80);
        require(!matches(bad, limbs, expected), c, "missed byte corruption " + std::to_string(i));
        ++c.negative_controls;
    }
    for (std::size_t i = 0; i < 4; ++i) {
        auto bad = limbs; bad[i] ^= UINT64_C(0x8000000000000000);
        require(!matches(bytes, bad, expected), c, "missed raw limb corruption " + std::to_string(i));
        ++c.negative_controls;
    }
    require(!matches(encode_be(kP), encode_limbs(kP), kP), c, "noncanonical p accepted");
    ++c.negative_controls;
    // Synthetic observation-channel controls only: these do not call kernels
    // with inadmissible accumulator magnitudes.
    for (std::size_t limb = 0; limb < 4; ++limb) {
        for (unsigned bit : {64U, 127U}) {
            Columns bad{}; bad[limb] = U128(1) << bit;
            require(decode_u128(bad[limb]) != 0, c, "missed high u128 corruption");
            ++c.negative_controls;
        }
    }
    Wide bad_high{}; bad_high.high = 1;
    require(decode_wide(bad_high) != 0, c, "missed propagated high-word corruption");
    ++c.negative_controls;
}

Counts run() {
    Counts c;
    controls(c);
    for (auto pattern : kPatterns) run_corpus(pattern, c);
    first_fold_carry_witness(c);
    require(c.corpora == 9 && c.fixtures == 148 && c.large_fixtures == 3, c, "corpus/fixture counts");
    require(c.seeded_cases == 148 && c.alias_cases == 136, c, "seed/alias counts");
    require(c.inline_calls == 284 && c.wide_calls == 2272, c, "candidate route counts");
    require(c.null_identity_calls == 72, c, "null identity count");
    require(c.phase_replays == 1160, c, "phase replay count");
    require(c.first_fold_carry_cases >= 6 && c.canonical_subtraction_cases != 0, c, "fold/canonical-correction branch coverage");
    require(c.negative_controls == 46, c, "negative control count");
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
} // namespace

int main(int argc, char**) {
    if (argc != 1) { std::cerr << "Usage: test_p4_field_sum (JSON stdout)\n"; return 2; }
    try {
        const Counts c = run();
        std::cout << "{\"schema\":\"parseatlas_p4_field_sum_correctness_v1\",\"status\":\"pass\",\"mismatches\":0,"
                  << "\"finite_corpus_only\":true,\"timing_claim\":false,\"constant_time_claim\":false,"
                  << "\"oracle\":\"Boost ordinary-integer prefix sums modulo p plus unchanged corrected FE64 prefix arithmetic\","
                  << "\"contract\":\"canonical x0 plus N canonical RHS, seed once, final-only canonical raw FE64 and all 32 BE bytes\","
                  << "\"chunks\":[1,16,4094,4095],\"lanes\":[1,4],\"inline_eager_tested\":true,"
                  << "\"corpus_reuse\":\"one real canonical corpus per pattern; oracle prefixes computed once, never replayed quadratically\","
                  << "\"phase_replay_limit_rhs\":" << kReplayLimit << ",\"large_fixture_phase_replay\":false,"
                  << "\"seed_hex\":\"" << hex64(kSeed) << "\",\"corpora\":" << c.corpora
                  << ",\"fixtures\":" << c.fixtures << ",\"large_fixtures\":" << c.large_fixtures
                  << ",\"rhs_objects\":" << c.rhs_objects << ",\"nonzero_rhs_objects\":" << c.nonzero_rhs_objects
                  << ",\"input_object_checks\":" << c.input_object_checks
                  << ",\"reference_prefix_additions\":" << c.reference_prefix_additions
                  << ",\"reference_prefix_checks\":" << c.reference_prefix_checks
                  << ",\"reference_seed_additions\":" << c.reference_seed_additions
                  << ",\"seeded_cases\":" << c.seeded_cases << ",\"alias_cases\":" << c.alias_cases
                  << ",\"nonzero_seed_cases\":" << c.nonzero_seed_cases << ",\"zero_residue_cases\":" << c.zero_residue_cases
                  << ",\"inline_calls\":" << c.inline_calls << ",\"wide_calls\":" << c.wide_calls
                  << ",\"null_identity_calls\":" << c.null_identity_calls << ",\"second_oracle_checks\":" << c.second_oracle_checks
                  << ",\"input_preservation_checks\":" << c.input_preservation_checks
                  << ",\"phase_replays\":" << c.phase_replays << ",\"replay_chunks\":" << c.replay_chunks
                  << ",\"bank_u128_checks\":" << c.bank_u128_checks << ",\"merged_u128_checks\":" << c.merged_u128_checks
                  << ",\"propagated_highword_checks\":" << c.propagated_highword_checks
                  << ",\"first_fold_checks\":" << c.first_fold_checks << ",\"first_fold_carry_cases\":" << c.first_fold_carry_cases
                  << ",\"second_fold_checks\":" << c.second_fold_checks
                  << ",\"canonical_correction_checks\":" << c.canonical_correction_checks
                  << ",\"canonical_subtraction_cases\":" << c.canonical_subtraction_cases
                  << ",\"output_checks\":" << c.output_checks << ",\"negative_controls\":" << c.negative_controls
                  << ",\"assertions\":" << c.assertions
                  << ",\"input_checksum_fnv1a64\":\"" << hex64(c.input_checksum)
                  << "\",\"output_checksum_fnv1a64\":\"" << hex64(c.output_checksum) << "\"}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cout << "{\"schema\":\"parseatlas_p4_field_sum_correctness_v1\",\"status\":\"fail\",\"error\":\""
                  << json_escape(error.what()) << "\"}\n";
        return 1;
    }
}
