#include "../probes/l0_state_control.hpp"

#include <boost/multiprecision/cpp_int.hpp>

#include <array>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {

using pa_l0_state::Form;
using pa_l0_state::Input;
using pa_l0_state::Method;
using pa_l0_state::Word;
using boost::multiprecision::cpp_int;

constexpr Word seed = 20260905;
constexpr Word random_cases = 100000;
constexpr Word maximum = ~Word{0};

void require(bool condition, const char* message) {
    // This gate deliberately remains live in -DNDEBUG builds.
    if (!condition) throw std::runtime_error(message);
}

Word next_random(Word& state) {
    state += UINT64_C(0x9e3779b97f4a7c15);
    Word z = state;
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

Input random_input(Word& random) {
    Input input{};
    for (auto& limb : input.a) limb = next_random(random);
    for (auto& limb : input.b) limb = next_random(random);
    input.carry_in = next_random(random) & 1;
    return input;
}

std::array<unsigned char, 64> state_bytes(const Input& input) {
    std::array<unsigned char, 64> bytes{};
    for (unsigned limb = 0; limb < 4; ++limb) {
        for (unsigned byte = 0; byte < 8; ++byte) {
            bytes[8 * limb + byte] = static_cast<unsigned char>(input.a[limb] >> (8 * byte));
            bytes[32 + 8 * limb + byte] = static_cast<unsigned char>(input.b[limb] >> (8 * byte));
        }
    }
    return bytes;
}

bool same_state(const Input& left, const Input& right) {
    // Never compare padding, carry alone, or only a checksum.
    return left.carry_in <= 1 && right.carry_in <= 1 &&
           left.carry_in == right.carry_in && state_bytes(left) == state_bytes(right);
}

cpp_int to_bigint(const std::array<Word, 4>& limbs) {
    cpp_int value = 0;
    for (unsigned i = 4; i != 0; --i) {
        value <<= 64;
        value += limbs[i - 1];
    }
    return value;
}

Input oracle_step(const Input& input) {
    require(input.carry_in <= 1, "oracle invalid carry");
    cpp_int sum = to_bigint(input.a) + to_bigint(input.b) + input.carry_in;
    const cpp_int mask = (cpp_int{1} << 64) - 1;
    Input result{};
    for (unsigned i = 0; i < 4; ++i) {
        result.a[i] = (sum & mask).convert_to<Word>();
        sum >>= 64;
        // Independent unbounded-integer rotation, masked to the word boundary.
        const cpp_int b = input.b[i];
        const cpp_int rotated = ((b << 13) | (b >> 51)) & mask;
        result.b[i] = (rotated ^ result.a[i]).convert_to<Word>();
    }
    result.carry_in = sum.convert_to<Word>();
    require(result.carry_in <= 1, "oracle complete carry exceeded one");
    return result;
}

Input frozen_original_step(const Input& input) {
    const auto output = pa_l0_native::compute<Method::Original>(input);
    Input result{};
    result.a = output.low;
    result.carry_in = output.carry_out;
    for (unsigned i = 0; i < 4; ++i) {
        result.b[i] = ((input.b[i] << 13) | (input.b[i] >> 51)) ^ result.a[i];
    }
    return result;
}

template<Method M, Form F>
void advance_adapter(Input& input, Word count, Word passes) {
    pa_l0_state::advance<M, F>(input, count, passes);
}

template<Method M, Form F>
void step_adapter(Input& input) {
    if constexpr (F == Form::Aggregate) {
        pa_l0_state::aggregate_step<M>(input);
    } else {
        pa_l0_state::scalar_step<M>(input.a[0], input.a[1], input.a[2], input.a[3],
                                   input.b[0], input.b[1], input.b[2], input.b[3],
                                   input.carry_in);
    }
}

struct Variant {
    void (*advance)(Input&, Word, Word);
    void (*step)(Input&);
};

template<Method M, Form F>
constexpr Variant variant() {
    return {advance_adapter<M, F>, step_adapter<M, F>};
}

constexpr std::array<Variant, 8> variants = {{
    variant<Method::Original, Form::Aggregate>(),
    variant<Method::GpMaterialized, Form::Aggregate>(),
    variant<Method::GpRecomputed, Form::Aggregate>(),
    variant<Method::Blocked2, Form::Aggregate>(),
    variant<Method::Original, Form::ScalarLocal>(),
    variant<Method::GpMaterialized, Form::ScalarLocal>(),
    variant<Method::GpRecomputed, Form::ScalarLocal>(),
    variant<Method::Blocked2, Form::ScalarLocal>()
}};

struct Counts {
    Word full_width_cases = 0;
    Word reference_oracle_checks = 0;
    Word advance_state_checks = 0;
    Word direct_step_checks = 0;
    Word recurrence_chains = 0;
    Word recurrence_steps = 0;
    Word recurrence_state_checks = 0;
    Word region_cases = 0;
    Word region_state_checks = 0;
    Word region_partition_checks = 0;
    Word zero_identity_checks = 0;
    Word region_gate_checks = 0;
    Word negative_comparator_checks = 0;
    Word checksum = UINT64_C(14695981039346656037);
};

void digest_byte(Counts& counts, unsigned char byte) {
    counts.checksum ^= byte;
    counts.checksum *= UINT64_C(1099511628211);
}

void digest_state(Counts& counts, const Input& input, unsigned variant_id) {
    digest_byte(counts, static_cast<unsigned char>(variant_id));
    for (unsigned char byte : state_bytes(input)) digest_byte(counts, byte);
    digest_byte(counts, static_cast<unsigned char>(input.carry_in));
}

void check_case(const Input& input, Counts& counts) {
    require(pa_l0_state::valid_region(input, 1, 1), "generated invalid one-step region");
    const Input expected = oracle_step(input);
    const Input original = frozen_original_step(input);
    require(same_state(original, expected), "frozen original differs from bigint state");
    ++counts.reference_oracle_checks;
    for (unsigned i = 0; i < variants.size(); ++i) {
        Input actual = input;
        variants[i].advance(actual, 1, 1);
        require(same_state(actual, expected) && same_state(actual, original),
                "advance differs from original/bigint full state");
        digest_state(counts, actual, i);
        ++counts.advance_state_checks;
        Input direct = input;
        variants[i].step(direct);
        require(same_state(direct, expected) && same_state(direct, original),
                "direct step differs from original/bigint full state");
        ++counts.direct_step_checks;
    }
    ++counts.full_width_cases;
}

void check_boundaries(Counts& counts) {
    const std::array<Word, 11> values = {{
        0, 1, 2, 3, (Word{1} << 63) - 1, Word{1} << 63,
        maximum - 2, maximum - 1, maximum,
        UINT64_C(0x5555555555555555), UINT64_C(0xaaaaaaaaaaaaaaaa)
    }};
    for (Word a : values) for (Word b : values) for (Word carry = 0; carry < 2; ++carry) {
        Input input{};
        input.a.fill(a);
        input.b.fill(b);
        input.carry_in = carry;
        check_case(input, counts);
    }

    // All K/P/G limb patterns, each represented by four operand pairs.
    // Fixtures may overlap; this count is not a claim of unique inputs.
    const std::array<std::array<Word, 2>, 12> pairs = {{
        {{0, 0}}, {{maximum - 1, 0}}, {{0, maximum - 1}}, {{1, 1}},
        {{maximum, 0}}, {{0, maximum}}, {{maximum - 1, 1}}, {{1, maximum - 1}},
        {{maximum, 1}}, {{1, maximum}}, {{maximum, maximum}}, {{maximum - 1, 2}}
    }};
    for (unsigned code = 0; code < 12 * 12 * 12 * 12; ++code) {
        Input input{};
        unsigned remaining = code;
        for (unsigned i = 0; i < 4; ++i) {
            const auto& pair = pairs[remaining % 12];
            remaining /= 12;
            input.a[i] = pair[0];
            input.b[i] = pair[1];
        }
        for (Word carry = 0; carry < 2; ++carry) {
            input.carry_in = carry;
            check_case(input, counts);
        }
    }
    for (unsigned limb = 0; limb < 4; ++limb) {
        for (unsigned bit = 0; bit < 64; ++bit) {
            for (unsigned lower_run = 0; lower_run < 2; ++lower_run) {
                for (Word carry = 0; carry < 2; ++carry) {
                    Input input{};
                    input.a[limb] = Word{1} << bit;
                    input.b[limb] = maximum;
                    for (unsigned j = 0; lower_run && j < limb; ++j) input.a[j] = maximum;
                    input.carry_in = carry;
                    check_case(input, counts);
                    input.a.swap(input.b);
                    check_case(input, counts);
                }
            }
        }
    }
}

void check_recurrences(Counts& counts) {
    Word random = seed ^ UINT64_C(0xd1b54a32d192ed03);
    for (unsigned chain = 0; chain < 128; ++chain) {
        Input expected = random_input(random);
        if (chain == 0) expected = Input{};
        if (chain == 1) {
            expected.a.fill(maximum);
            expected.b.fill(maximum);
            expected.carry_in = 1;
        }
        Input original = expected;
        std::array<Input, variants.size()> states{};
        states.fill(expected);
        for (unsigned step = 0; step < 257; ++step) {
            expected = oracle_step(expected);
            original = frozen_original_step(original);
            require(same_state(original, expected), "original recurrence diverged from bigint");
            ++counts.reference_oracle_checks;
            for (unsigned i = 0; i < variants.size(); ++i) {
                variants[i].advance(states[i], 1, 1);
                require(same_state(states[i], expected) && same_state(states[i], original),
                        "candidate recurrence diverged at an internal step");
                digest_state(counts, states[i], i);
                ++counts.recurrence_state_checks;
            }
            ++counts.recurrence_steps;
        }
        ++counts.recurrence_chains;
    }
}

void check_regions(Counts& counts) {
    struct Region { Word count; Word passes; };
    constexpr std::array<Region, 12> regions = {{{0, 0}, {0, 7}, {3, 0}, {1, 1},
        {1, 17}, {17, 1}, {3, 19}, {19, 3}, {16, 16}, {7, 31}, {31, 7}, {0, 1000000}}};
    Word random = seed ^ UINT64_C(0x94d049bb133111eb);
    for (unsigned fixture = 0; fixture < 24; ++fixture) {
        const Input initial = random_input(random);
        for (const Region region : regions) {
            require(pa_l0_state::valid_region(initial, region.count, region.passes),
                    "valid generated region rejected");
            const Word operations = region.count * region.passes;
            Input expected = initial;
            Input original = initial;
            for (Word op = 0; op < operations; ++op) {
                expected = oracle_step(expected);
                original = frozen_original_step(original);
                require(same_state(original, expected), "region reference differs from bigint");
                ++counts.reference_oracle_checks;
            }
            for (unsigned i = 0; i < variants.size(); ++i) {
                Input actual = initial;
                variants[i].advance(actual, region.count, region.passes);
                require(same_state(actual, expected) && same_state(actual, original),
                        "multi-count/multi-pass final state differs from oracle");
                ++counts.region_state_checks;
                Input flattened = initial;
                variants[i].advance(flattened, operations, 1);
                require(same_state(actual, flattened), "region flattening changed final state");
                ++counts.region_partition_checks;
                if (operations == 0) {
                    require(same_state(actual, initial), "zero region is not identity");
                    ++counts.zero_identity_checks;
                } else {
                    Input partitioned = initial;
                    for (Word pass = 0; pass < region.passes; ++pass) {
                        variants[i].advance(partitioned, region.count, 1);
                    }
                    require(same_state(actual, partitioned), "per-pass partitioning changed final state");
                    ++counts.region_partition_checks;
                }
                digest_state(counts, actual, i);
            }
            ++counts.region_cases;
        }
    }
}

void check_region_gate(Counts& counts) {
    struct GateCase { Word count; Word passes; bool valid; };
    constexpr std::array<GateCase, 21> cases = {{{0, 0, true}, {0, 1000000, true},
        {1000000000, 0, true}, {1000000000, 1, true}, {1000, 1000000, true},
        {1, 1000000, true}, {999, 1000000, true}, {500000000, 2, true},
        {333333333, 3, true}, {333333334, 3, false}, {500000001, 2, false},
        {1001, 1000000, false}, {1000000000, 2, false}, {1000000001, 0, false},
        {0, 1000001, false}, {1, 1000001, false}, {maximum, 0, false},
        {0, maximum, false}, {maximum, maximum, false}, {maximum, 1, false},
        {Word{1} << 63, 2, false}}};
    Input input{};
    for (Word carry : std::array<Word, 4>{{0, 1, 2, maximum}}) {
        input.carry_in = carry;
        for (const GateCase item : cases) {
            require(pa_l0_state::valid_region(input, item.count, item.passes) ==
                    (carry <= 1 && item.valid), "region gate accepted invalid or rejected valid boundary");
            ++counts.region_gate_checks;
        }
    }
    // Never call unchecked advance with the invalid or enormous gate-only cases.
}

void check_negative_comparator(Counts& counts) {
    Word random = seed;
    const Input original = random_input(random);
    require(same_state(original, original), "comparator rejects identical valid state");
    for (unsigned limb = 0; limb < 4; ++limb) {
        for (unsigned bit : std::array<unsigned, 2>{{0, 63}}) {
            Input changed = original;
            changed.a[limb] ^= Word{1} << bit;
            require(!same_state(original, changed), "comparator missed corrupted a");
            ++counts.negative_comparator_checks;
            changed = original;
            changed.b[limb] ^= Word{1} << bit;
            require(!same_state(original, changed), "comparator missed corrupted b");
            ++counts.negative_comparator_checks;
        }
    }
    Input changed = original;
    changed.carry_in ^= 1;
    require(!same_state(original, changed), "comparator missed changed valid carry");
    ++counts.negative_comparator_checks;
    for (Word carry : std::array<Word, 2>{{2, maximum}}) {
        changed.carry_in = carry;
        require(!same_state(original, changed) && !same_state(changed, original) &&
                !same_state(changed, changed), "comparator accepted invalid carry");
        ++counts.negative_comparator_checks;
    }
}

} // namespace

int main(int argc, char**) {
    if (argc != 1) {
        std::cerr << "usage: test_l0_state_control (no arguments)\n";
        return 2;
    }
    try {
        Counts counts;
        check_region_gate(counts);
        check_negative_comparator(counts);
        check_boundaries(counts);
        const Word boundary_cases = counts.full_width_cases;
        Word random = seed;
        for (Word i = 0; i < random_cases; ++i) check_case(random_input(random), counts);
        check_recurrences(counts);
        check_regions(counts);
        std::cout << "{\n"
                  << "  \"schema\": \"pa_l0_state_control_correctness_v1\",\n"
                  << "  \"status\": \"pass\",\n"
                  << "  \"seed\": " << seed << ",\n"
                  << "  \"random_cases\": " << random_cases << ",\n"
                  << "  \"boundary_cases\": " << boundary_cases << ",\n"
                  << "  \"full_width_cases\": " << counts.full_width_cases << ",\n"
                  << "  \"methods\": 4,\n  \"forms\": 2,\n  \"variants\": 8,\n"
                  << "  \"reference_oracle_checks\": " << counts.reference_oracle_checks << ",\n"
                  << "  \"advance_state_checks\": " << counts.advance_state_checks << ",\n"
                  << "  \"direct_step_checks\": " << counts.direct_step_checks << ",\n"
                  << "  \"recurrence_chains\": " << counts.recurrence_chains << ",\n"
                  << "  \"recurrence_steps\": " << counts.recurrence_steps << ",\n"
                  << "  \"recurrence_state_checks\": " << counts.recurrence_state_checks << ",\n"
                  << "  \"region_cases\": " << counts.region_cases << ",\n"
                  << "  \"region_state_checks\": " << counts.region_state_checks << ",\n"
                  << "  \"region_partition_checks\": " << counts.region_partition_checks << ",\n"
                  << "  \"zero_identity_checks\": " << counts.zero_identity_checks << ",\n"
                  << "  \"region_gate_checks\": " << counts.region_gate_checks << ",\n"
                  << "  \"negative_comparator_checks\": " << counts.negative_comparator_checks << ",\n"
                  << "  \"oracle\": \"boost::multiprecision::cpp_int\",\n"
                  << "  \"comparison\": \"serialized_le_64_bytes_a_b_and_full_carry\",\n"
                  << "  \"mismatches\": 0,\n"
                  << "  \"state_checksum_fnv1a64\": \"" << std::hex << std::setw(16)
                  << std::setfill('0') << counts.checksum << std::dec << "\",\n"
                  << "  \"negative_controls\": \"pass\",\n"
                  << "  \"timing_claim\": false\n}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "state-control correctness failure: " << error.what() << '\n';
        return 1;
    }
}
