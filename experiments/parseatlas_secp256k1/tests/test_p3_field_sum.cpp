// P3 finite field-sum correctness corpus. No timing, CT, or production claims.
// Full-function tests call the frozen kernels. Small-fixture checkpoint tests
// replay their actual public add_assign and exposed phase helpers separately.
#include "../probes/p3_field_sum_kernels.hpp"
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

// Compile with -fsyntax-only and ONE of these macros. Each forbidden
// specialization must fail its shared kernel static_assert, even for N=0.
#if defined(PA_P3_INVALID_FULL)
[[maybe_unused]] pa_p3::FE invalid_full_probe(const pa_p3::FE& seed) {
    return pa_p3::sum_fe52_e2e<4095, false>(seed, nullptr, 0);
}
#endif
#if defined(PA_P3_INVALID_WEAK)
[[maybe_unused]] pa_p3::FE invalid_weak_probe(const pa_p3::FE& seed) {
    return pa_p3::sum_fe52_resident<4096, true>(seed, nullptr, 0);
}
#endif
#if defined(PA_P3_INVALID_ZERO)
[[maybe_unused]] pa_p3::FE invalid_zero_probe(const pa_p3::FE& seed) {
    return pa_p3::sum_fe52_e2e<0, false>(seed, nullptr, 0);
}
#endif

namespace {
using boost::multiprecision::cpp_int;
using FE = pa_p3::FE;
using FE52 = pa_p3::FE52;
using Bytes = std::array<std::uint8_t, 32>;
using L64 = std::array<std::uint64_t, 4>;
using L52 = std::array<std::uint64_t, 5>;
const cpp_int kB = cpp_int(1) << 256;
const cpp_int kK = (cpp_int(1) << 32) + 977;
const cpp_int kP = kB - kK;
constexpr std::uint64_t kM52 = UINT64_C(0xfffffffffffff);
constexpr std::uint64_t kM48 = UINT64_C(0xffffffffffff);
constexpr std::uint64_t kSeed = UINT64_C(0x5041335f53554d53);
constexpr std::size_t kPhaseReplayLimit = 8193;
constexpr std::array<std::size_t, 18> kSizes{{
    0, 1, 2, 15, 16, 17, 255, 256, 257, 4093, 4094, 4095,
    4096, 4097, 8190, 8191, 8192, 8193}};

static_assert(sizeof(FE) == 32 && sizeof(FE52) == 40);
static_assert(std::is_trivially_copyable_v<FE> && std::is_trivially_copyable_v<FE52>);

struct Counts {
    std::uint64_t assertions = 0;
    std::uint64_t fixtures = 0;
    std::uint64_t large_fixtures = 0;
    std::uint64_t rhs_objects = 0;
    std::uint64_t nonzero_rhs_objects = 0;
    std::uint64_t identity_fixtures = 0;
    std::uint64_t seeded_cases = 0;
    std::uint64_t alias_cases = 0;
    std::uint64_t nonzero_seed_cases = 0;
    std::uint64_t zero_residue_cases = 0;
    std::uint64_t reference_calls = 0;
    std::uint64_t e2e_calls = 0;
    std::uint64_t resident_calls = 0;
    std::uint64_t null_identity_calls = 0;
    std::uint64_t output_checks = 0;
    std::uint64_t input_object_checks = 0;
    std::uint64_t input_preservation_routes = 0;
    std::uint64_t second_oracle_checks = 0;
    std::uint64_t checkpoint_replays = 0;
    std::uint64_t raw_chunk_checks = 0;
    std::uint64_t canonical_checkpoint_checks = 0;
    std::uint64_t weak_state_checks = 0;
    std::uint64_t phase_final_checks = 0;
    std::uint64_t negative_controls = 0;
    std::uint64_t checksum = UINT64_C(14695981039346656037);
    std::uint64_t input_checksum = UINT64_C(14695981039346656037);
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

template <std::size_t N>
std::array<std::uint64_t, N> encode_limbs(cpp_int value, unsigned width) {
    const cpp_int mask = (cpp_int(1) << width) - 1;
    std::array<std::uint64_t, N> result{};
    for (auto& limb : result) {
        limb = static_cast<std::uint64_t>(value & mask);
        value >>= width;
    }
    if (value != 0) throw std::runtime_error("oracle limb range");
    return result;
}

template <std::size_t N>
cpp_int decode_limbs(const std::array<std::uint64_t, N>& limbs, unsigned width) {
    cpp_int result = 0;
    for (std::size_t i = limbs.size(); i != 0; --i) {
        result *= cpp_int(1) << width;
        result += limbs[i - 1];
    }
    return result;
}

L52 raw52(const FE52& value) {
    return {{value.n[0], value.n[1], value.n[2], value.n[3], value.n[4]}};
}

std::string hex_bytes(const Bytes& bytes) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (auto byte : bytes) out << std::setw(2) << static_cast<unsigned>(byte);
    return out.str();
}

template <std::size_t N>
std::string hex_limbs(const std::array<std::uint64_t, N>& limbs) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << '[';
    for (std::size_t i = 0; i < N; ++i) {
        if (i) out << ',';
        out << std::setw(16) << limbs[i];
    }
    out << ']';
    return out.str();
}

void mix_bytes(const Bytes& bytes, std::uint64_t& checksum) {
    for (auto byte : bytes) checksum = (checksum ^ byte) * UINT64_C(1099511628211);
}

bool matches64(const Bytes& bytes, const L64& limbs, const cpp_int& expected) {
    bool result = expected >= 0 && expected < kP;
    const auto eb = encode_be(expected);
    const auto el = encode_limbs<4>(expected, 64);
    for (std::size_t i = 0; i < bytes.size(); ++i) result &= bytes[i] == eb[i];
    for (std::size_t i = 0; i < limbs.size(); ++i) result &= limbs[i] == el[i];
    result &= decode_limbs(limbs, 64) < kP;
    return result;
}

bool canonical52(const L52& limbs, const cpp_int& expected) {
    bool result = expected >= 0 && expected < kP;
    const auto el = encode_limbs<5>(expected, 52);
    for (std::size_t i = 0; i < limbs.size(); ++i) {
        result &= limbs[i] == el[i];
        result &= limbs[i] <= (i == 4 ? kM48 : kM52);
    }
    result &= decode_limbs(limbs, 52) < kP;
    return result;
}

void check_output(const FE& actual, const cpp_int& expected, Counts& c,
                  const std::string& context) {
    const L64 limbs = actual.limbs();
    const Bytes bytes = actual.to_bytes();
    if (!matches64(bytes, limbs, expected)) {
        require(false, c, context + " actual_be=" + hex_bytes(bytes) +
                " raw_le=" + hex_limbs(limbs) + " expected_be=" + hex_bytes(encode_be(expected)));
    }
    require(actual.limbs() == limbs, c, context + ": serialization changed raw output");
    ++c.output_checks;
    mix_bytes(bytes, c.checksum);
}

void check_checkpoint(const FE52& state, const cpp_int& expected, Counts& c,
                      const std::string& context) {
    const L52 limbs = raw52(state);
    require(canonical52(limbs, expected), c,
            context + " raw52_le=" + hex_limbs(limbs) +
            " expected_be=" + hex_bytes(encode_be(expected)));
    // This pre-normalized serializer cannot hide a noncanonical checkpoint.
    Bytes bytes{};
    state.store_b32_prenorm(bytes.data());
    require(bytes == encode_be(expected), c, context + ": checkpoint byte mismatch");
    require(raw52(state) == limbs, c, context + ": checkpoint serializer mutation");
    ++c.canonical_checkpoint_checks;
    mix_bytes(bytes, c.checksum);
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
    return decode_limbs(L52{{kM52, 0, 0, 0, kM48}}, 52);
}

std::vector<cpp_int> limb_patterns() {
    std::vector<cpp_int> values{cpp_int(0), cpp_int(1), kP - 1, kP, kP + 1, kB - 1,
                               witness_value(), kB - 1 - (cpp_int(1) << 33)};
    constexpr std::array<unsigned, 26> bits{{0, 1, 2, 51, 52, 53, 63, 64, 65,
        103, 104, 105, 127, 128, 129, 155, 156, 157, 191, 192, 193, 207, 208, 209, 254, 255}};
    for (auto bit : bits) {
        const cpp_int power = cpp_int(1) << bit;
        values.push_back(power - 1);
        values.push_back(power);
        values.push_back(power + 1);
    }
    return values;
}

struct Fixture {
    FE seed;
    cpp_int seed_int;
    cpp_int rhs_sum = 0;
    std::vector<FE> rhs;
    std::vector<FE52> resident;
};

Fixture make_fixture(Pattern pattern, std::size_t count, Counts& c) {
    Fixture f;
    SplitMix64 random{kSeed ^ static_cast<std::uint64_t>(count) ^
        (static_cast<std::uint64_t>(pattern) << 56)};
    switch (pattern) {
        case Pattern::Zero: f.seed_int = 0; break;
        case Pattern::ZeroNonzeroSeed: f.seed_int = kP - 2; break;
        case Pattern::One: f.seed_int = 7; break;
        case Pattern::MinusOne: f.seed_int = kP - 1; break;
        case Pattern::Witness: f.seed_int = witness_value(); break;
        case Pattern::Limbs: f.seed_int = (cpp_int(1) << 208) - 1; break;
        case Pattern::Cancel: f.seed_int = 1; break;
        case Pattern::Random:
            f.seed_int = mod(random.raw256());
            if (f.seed_int == 0) f.seed_int = 1;
            break;
    }
    f.seed = FE::from_bytes(encode_be(f.seed_int));
    require(matches64(f.seed.to_bytes(), f.seed.limbs(), f.seed_int), c, "seed construction");
    mix_bytes(f.seed.to_bytes(), c.input_checksum);
    f.rhs.reserve(count);
    f.resident.reserve(count);
    const auto boundaries = limb_patterns();
    for (std::size_t i = 0; i < count; ++i) {
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
        const cpp_int expected = mod(raw);
        f.rhs_sum += expected;  // Independent ordinary-integer sum, not FE arithmetic.
        const FE value = FE::from_bytes(encode_be(raw));
        const FE52 resident = FE52::from_fe(value);
        if (!matches64(value.to_bytes(), value.limbs(), expected) ||
            !canonical52(raw52(resident), expected)) {
            require(false, c, "input canonical construction " + std::to_string(i));
        }
        f.rhs.push_back(value);
        f.resident.push_back(resident);
        ++c.input_object_checks;
        ++c.rhs_objects;
        if (expected != 0) ++c.nonzero_rhs_objects;
        mix_bytes(value.to_bytes(), c.input_checksum);
    }
    return f;
}

template <std::size_t Chunk, bool Weak>
void replay_checkpoints(const Fixture& f, Counts& c, const std::string& label) {
    ++c.checkpoint_replays;
    const std::size_t count = f.rhs.size();
    if (count == 0) return;
    FE52 state = FE52::from_fe(f.seed);
    cpp_int prefix = f.seed_int;
    std::size_t offset = 0;
    while (offset < count) {
        const std::size_t remaining = count - offset;
        const std::size_t chunk = remaining > Chunk ? Chunk : remaining;
        const std::string at = label + " checkpoint " + std::to_string(offset + chunk);
        check_checkpoint(state, mod(prefix), c, at + " canonical entry");
        cpp_int local_exact = mod(prefix);
        for (std::size_t i = 0; i < chunk; ++i) {
            state.add_assign(f.resident[offset + i]);
            const cpp_int input = decode_limbs(f.rhs[offset + i].limbs(), 64);
            prefix += input;
            local_exact += input;
        }
        const L52 raw = raw52(state);
        require(decode_limbs(raw, 52) == local_exact, c,
                at + ": component accumulation lost ordinary-integer information");
        require(mod(local_exact) == mod(prefix), c, at + ": global prefix residue mismatch");
        ++c.raw_chunk_checks;
        if constexpr (Weak) {
            FE52 weak = state;
            weak.normalize_weak();
            const auto weak_raw = raw52(weak);
            bool widths = weak_raw[4] <= (UINT64_C(1) << 48);
            for (std::size_t i = 0; i < 4; ++i) widths &= weak_raw[i] <= kM52;
            const cpp_int weak_int = decode_limbs(weak_raw, 52);
            require(widths && weak_int < kB + cpp_int(chunk) * kK &&
                    mod(weak_int) == mod(prefix), c, at + ": proven weak-output domain");
            ++c.weak_state_checks;
        }
        offset += chunk;
        if (offset < count) {
            pa_p3::detail::normalize_nonfinal<Weak>(state);
            check_checkpoint(state, mod(prefix), c, at + " canonical nonfinal result");
        } else {
            const FE final = pa_p3::detail::decode_final<Weak>(state);
            check_output(final, mod(prefix), c, at + " final raw-to-canonical decode");
            require(raw52(state) == raw, c, at + ": by-value final decode mutated raw state");
            ++c.phase_final_checks;
        }
    }
    require(mod(prefix) == mod(f.seed_int + f.rhs_sum), c, label + ": phase replay Boost final sum");
}

void preserve_inputs(const Fixture& f, const std::vector<FE>& before64,
                     const std::vector<FE52>& before52, const FE& seed,
                     const L64& seed_before, Counts& c, const std::string& label) {
    // Read every object byte. No positive-length null access is manufactured.
    const bool same64 = f.rhs.empty() ||
        std::memcmp(f.rhs.data(), before64.data(), f.rhs.size() * sizeof(FE)) == 0;
    const bool same52 = f.resident.empty() ||
        std::memcmp(f.resident.data(), before52.data(), f.resident.size() * sizeof(FE52)) == 0;
    require(same64 && same52 && seed.limbs() == seed_before, c, label + ": input bytes changed");
    ++c.input_preservation_routes;
}

template <std::size_t Chunk, bool Weak>
void check_schedule(const Fixture& f, const FE& seed, const cpp_int& expected,
                    const FE& reference, const std::vector<FE>& before64,
                    const std::vector<FE52>& before52, Counts& c, const std::string& label) {
    const std::size_t count = f.rhs.size();
    const FE* rhs = count == 0 ? nullptr : f.rhs.data();
    const FE52* resident = count == 0 ? nullptr : f.resident.data();
    const L64 seed_before = seed.limbs();
    const std::string schedule = label + (Weak ? " weak" : " full") + std::to_string(Chunk);
    const FE e2e = pa_p3::sum_fe52_e2e<Chunk, Weak>(seed, rhs, count);
    check_output(e2e, expected, c, schedule + " E2E");
    require(e2e.limbs() == reference.limbs() && e2e.to_bytes() == reference.to_bytes(), c,
            schedule + ": E2E corrected FE64 second oracle");
    ++c.second_oracle_checks;
    ++c.e2e_calls;
    preserve_inputs(f, before64, before52, seed, seed_before, c, schedule + " E2E");
    const FE direct = pa_p3::sum_fe52_resident<Chunk, Weak>(seed, resident, count);
    check_output(direct, expected, c, schedule + " resident");
    require(direct.limbs() == reference.limbs() && direct.to_bytes() == reference.to_bytes(), c,
            schedule + ": resident corrected FE64 second oracle");
    ++c.second_oracle_checks;
    ++c.resident_calls;
    preserve_inputs(f, before64, before52, seed, seed_before, c, schedule + " resident");
    if (count == 0) c.null_identity_calls += 2;
}

void run_seed_case(const Fixture& f, const FE& seed, const cpp_int& seed_int,
                   bool alias, const std::vector<FE>& before64,
                   const std::vector<FE52>& before52, Counts& c, const std::string& label) {
    const cpp_int expected = mod(seed_int + f.rhs_sum);
    const FE* rhs = f.rhs.empty() ? nullptr : f.rhs.data();
    const L64 seed_before = seed.limbs();
    const FE reference = pa_p3::sum_fe64(seed, rhs, f.rhs.size());
    check_output(reference, expected, c, label + " corrected FE64 reference");
    preserve_inputs(f, before64, before52, seed, seed_before, c, label + " reference");
    ++c.reference_calls;
    if (f.rhs.empty()) ++c.null_identity_calls;
    check_schedule<1, false>(f, seed, expected, reference, before64, before52, c, label);
    check_schedule<16, false>(f, seed, expected, reference, before64, before52, c, label);
    check_schedule<256, false>(f, seed, expected, reference, before64, before52, c, label);
    check_schedule<4094, false>(f, seed, expected, reference, before64, before52, c, label);
    check_schedule<4095, true>(f, seed, expected, reference, before64, before52, c, label);
    if (seed_int != 0) ++c.nonzero_seed_cases;
    if (expected == 0) ++c.zero_residue_cases;
    if (alias) ++c.alias_cases; else ++c.seeded_cases;
}

void run_fixture(Pattern pattern, std::size_t count, Counts& c) {
    const std::string label = std::string(pattern_name(pattern)) + " N=" + std::to_string(count);
    const Fixture f = make_fixture(pattern, count, c);
    const std::vector<FE> before64 = f.rhs;
    const std::vector<FE52> before52 = f.resident;
    run_seed_case(f, f.seed, f.seed_int, false, before64, before52, c, label);
    if (count != 0 && count <= kPhaseReplayLimit) {
        const std::size_t index = count / 2;
        // Same-type FE64 alias only. Resident RHS remain separate canonical
        // FE52 objects; no cross-type alias is fabricated.
        const FE& alias_seed = f.rhs[index];
        run_seed_case(f, alias_seed, decode_limbs(alias_seed.limbs(), 64), true,
                      before64, before52, c, label + " seed aliases RHS[" + std::to_string(index) + ']');
    }
    if (count <= kPhaseReplayLimit) {
        replay_checkpoints<1, false>(f, c, label + " full1");
        replay_checkpoints<16, false>(f, c, label + " full16");
        replay_checkpoints<256, false>(f, c, label + " full256");
        replay_checkpoints<4094, false>(f, c, label + " full4094");
        replay_checkpoints<4095, true>(f, c, label + " weak4095");
        preserve_inputs(f, before64, before52, f.seed, f.seed.limbs(), c, label + " after phase replay");
    } else ++c.large_fixtures;
    ++c.fixtures;
    if (count == 0) ++c.identity_fixtures;
}

void negative_controls(Counts& c) {
    Bytes bytes{};
    for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<std::uint8_t>(i + 1);
    const cpp_int expected("0x0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20");
    const L64 limbs{{UINT64_C(0x191a1b1c1d1e1f20), UINT64_C(0x1112131415161718),
                     UINT64_C(0x090a0b0c0d0e0f10), UINT64_C(0x0102030405060708)}};
    require(matches64(bytes, limbs, expected), c, "known endian positive comparator control");
    for (std::size_t i = 0; i < 32; ++i) {
        auto bad = bytes;
        bad[i] ^= UINT8_C(0x80);
        require(!matches64(bad, limbs, expected), c, "missed output byte corruption " + std::to_string(i));
        ++c.negative_controls;
    }
    for (std::size_t i = 0; i < 4; ++i) {
        auto bad = limbs;
        bad[i] ^= UINT64_C(0x8000000000000000);
        require(!matches64(bytes, bad, expected), c, "missed output limb corruption " + std::to_string(i));
        ++c.negative_controls;
    }
    require(!matches64(encode_be(kP), encode_limbs<4>(kP, 64), kP), c, "raw noncanonical p accepted");
    ++c.negative_controls;
    const L52 expected52 = encode_limbs<5>(expected, 52);
    for (std::size_t i = 0; i < 5; ++i) {
        auto bad = expected52;
        bad[i] ^= UINT64_C(1);
        require(!canonical52(bad, expected), c, "missed resident limb corruption " + std::to_string(i));
        ++c.negative_controls;
    }
    require(!canonical52(encode_limbs<5>(kP, 52), kP), c, "noncanonical resident p accepted");
    ++c.negative_controls;
}

Counts run() {
    Counts c;
    negative_controls(c);
    for (auto pattern : kPatterns) for (auto count : kSizes) run_fixture(pattern, count, c);
    run_fixture(Pattern::Witness, 65536, c);
    run_fixture(Pattern::Random, 65536, c);
    run_fixture(Pattern::Random, 1048576, c);
    require(c.fixtures == 147 && c.large_fixtures == 3, c, "fixture count");
    require(c.seeded_cases == 147 && c.alias_cases == 136, c, "seed/alias case count");
    require(c.reference_calls == 283 && c.e2e_calls == 1415 && c.resident_calls == 1415, c, "route count");
    require(c.identity_fixtures == 8 && c.null_identity_calls == 88, c, "null identity count");
    require(c.checkpoint_replays == 720 && c.phase_final_checks == 680, c, "checkpoint replay count");
    require(c.negative_controls == 43, c, "negative control count");
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
    if (argc != 1) {
        std::cerr << "Usage: test_p3_field_sum (JSON stdout)\n";
        return 2;
    }
    try {
        const Counts c = run();
        std::cout << "{\"schema\":\"parseatlas_p3_field_sum_correctness_v1\",\"status\":\"pass\","
                  << "\"mismatches\":0,\"finite_corpus_only\":true,\"timing_claim\":false,\"constant_time_claim\":false,"
                  << "\"oracle\":\"independent Boost ordinary-integer full sum modulo p plus corrected FE64 reference\","
                  << "\"contract\":\"canonical x0 plus N canonical RHS; N counts RHS; final-only output\","
                  << "\"schedules\":[\"full1\",\"full16\",\"full256\",\"full4094\",\"weak4095\"],"
                  << "\"layouts_tested_per_schedule\":[\"FE64 E2E\",\"FE52 resident\"],"
                  << "\"seed_hex\":\"" << hex64(kSeed) << "\","
                  << "\"phase_replay_limit_rhs\":" << kPhaseReplayLimit << ','
                  << "\"large_fixture_checkpoint_replay\":false,"
                  << "\"input_preservation\":\"all object bytes checked after each route; no cross-type aliases\","
                  << "\"fixtures\":" << c.fixtures << ",\"large_fixtures\":" << c.large_fixtures
                  << ",\"rhs_objects\":" << c.rhs_objects << ",\"nonzero_rhs_objects\":" << c.nonzero_rhs_objects
                  << ",\"seeded_cases\":" << c.seeded_cases << ",\"alias_cases\":" << c.alias_cases
                  << ",\"nonzero_seed_cases\":" << c.nonzero_seed_cases << ",\"zero_residue_cases\":" << c.zero_residue_cases
                  << ",\"identity_fixtures\":" << c.identity_fixtures << ",\"null_identity_calls\":" << c.null_identity_calls
                  << ",\"reference_calls\":" << c.reference_calls << ",\"e2e_calls\":" << c.e2e_calls
                  << ",\"resident_calls\":" << c.resident_calls << ",\"second_oracle_checks\":" << c.second_oracle_checks
                  << ",\"output_checks\":" << c.output_checks << ",\"input_object_checks\":" << c.input_object_checks
                  << ",\"input_preservation_routes\":" << c.input_preservation_routes
                  << ",\"checkpoint_replays\":" << c.checkpoint_replays << ",\"raw_chunk_checks\":" << c.raw_chunk_checks
                  << ",\"canonical_checkpoint_checks\":" << c.canonical_checkpoint_checks
                  << ",\"weak_state_checks\":" << c.weak_state_checks << ",\"phase_final_checks\":" << c.phase_final_checks
                  << ",\"negative_controls\":" << c.negative_controls << ",\"assertions\":" << c.assertions
                  << ",\"input_checksum_fnv1a64\":\"" << hex64(c.input_checksum)
                  << "\",\"output_checksum_fnv1a64\":\"" << hex64(c.checksum) << "\"}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cout << "{\"schema\":\"parseatlas_p3_field_sum_correctness_v1\",\"status\":\"fail\",\"error\":\""
                  << json_escape(error.what()) << "\"}\n";
        return 1;
    }
}
