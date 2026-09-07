#include "../probes/l0_native_kernels.hpp"

#include <boost/multiprecision/cpp_int.hpp>

#include <array>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using pa_l0_native::Input;
using pa_l0_native::Method;
using pa_l0_native::Output;
using pa_l0_native::Word;
using boost::multiprecision::cpp_int;

constexpr Word seed = 20260905;
constexpr Word random_cases = 100000;
constexpr Word maximum = ~Word{0};

void require(bool condition, const char* message) {
    // Ordinary assert would silently remove the gate in the -DNDEBUG build.
    if (!condition) throw std::runtime_error(message);
}

Word next_random(Word& state) {
    state += UINT64_C(0x9e3779b97f4a7c15);
    Word z = state;
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

std::array<unsigned char, 32> low_bytes(const Output& output) {
    std::array<unsigned char, 32> bytes{};
    for (unsigned i = 0; i < 4; ++i) {
        for (unsigned j = 0; j < 8; ++j) {
            bytes[8 * i + j] = static_cast<unsigned char>(output.low[i] >> (8 * j));
        }
    }
    return bytes;
}

bool same_output(const Output& left, const Output& right) {
    // Compare explicit bytes and carry, never struct padding or only a digest.
    return left.carry_out <= 1 && right.carry_out <= 1 &&
           left.carry_out == right.carry_out && low_bytes(left) == low_bytes(right);
}

cpp_int to_bigint(const std::array<Word, 4>& limbs) {
    cpp_int value = 0;
    for (unsigned i = 4; i != 0; --i) {
        value <<= 64;
        value += limbs[i - 1];
    }
    return value;
}

Output oracle(const Input& input) {
    require(pa_l0_native::valid_input(input), "oracle invalid carry input");
    cpp_int sum = to_bigint(input.a) + to_bigint(input.b) + input.carry_in;
    const cpp_int mask = (cpp_int{1} << 64) - 1;
    Output output{};
    for (unsigned i = 0; i < 4; ++i) {
        output.low[i] = (sum & mask).convert_to<Word>();
        sum >>= 64;
    }
    output.carry_out = sum.convert_to<Word>();
    require(output.carry_out <= 1, "bigint carry exceeded complete boundary");
    return output;
}

struct Counts {
    Word cases = 0;
    Word checks = 0;
    Word checksum = UINT64_C(14695981039346656037);
    Word layouts = 0;
    Word tiny_inputs = 0;
    Word tiny_outputs = 0;
    Word digit_pairs = 0;
    Word encoding_pairs = 0;
    Word associative_triples = 0;
    Word lossy_nonassociative = 0;
    Word needs_high_two = 0;
};

void digest_byte(Counts& counts, unsigned char byte) {
    counts.checksum ^= byte;
    counts.checksum *= UINT64_C(1099511628211);
}

void check_case(const Input& input, Counts& counts) {
    require(pa_l0_native::valid_input(input), "generated invalid carry input");
    const Output expected = oracle(input);
    const std::array<Output, 4> results = {
        pa_l0_native::compute<Method::Original>(input),
        pa_l0_native::compute<Method::GpMaterialized>(input),
        pa_l0_native::compute<Method::GpRecomputed>(input),
        pa_l0_native::compute<Method::Blocked2>(input)
    };
    for (unsigned method = 0; method < results.size(); ++method) {
        require(same_output(results[method], expected), "C++ method differs from bigint bytes/carry");
        require(same_output(results[method], results[0]), "C++ method differs from original bytes/carry");
        digest_byte(counts, static_cast<unsigned char>(method));
        for (unsigned char byte : low_bytes(results[method])) digest_byte(counts, byte);
        digest_byte(counts, static_cast<unsigned char>(results[method].carry_out));
        ++counts.checks;
    }
    ++counts.cases;
}

void check_boundaries(Counts& counts) {
    const std::array<Word, 11> values = {
        0, 1, 2, 3, (Word{1} << 63) - 1, Word{1} << 63,
        maximum - 2, maximum - 1, maximum,
        UINT64_C(0x5555555555555555), UINT64_C(0xaaaaaaaaaaaaaaaa)
    };
    for (Word a : values) for (Word b : values) for (Word carry = 0; carry < 2; ++carry) {
        Input input{};
        input.a.fill(a);
        input.b.fill(b);
        input.carry_in = carry;
        check_case(input, counts);
    }

    // Every K/P/G pattern across four limbs, with four operand-pair
    // realizations per digit class. This prevents checking carry summaries alone.
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

    // A single bit at every position and every limb boundary, also after a
    // lower all-ones propagation run. Include swapped operands explicitly.
    for (unsigned limb = 0; limb < 4; ++limb) {
        for (unsigned bit = 0; bit < 64; ++bit) {
            for (unsigned lower_run = 0; lower_run < 2; ++lower_run) {
                for (Word carry = 0; carry < 2; ++carry) {
                    Input input{};
                    input.a[limb] = Word{1} << bit;
                    input.b[limb] = maximum;
                    if (lower_run) {
                        for (unsigned j = 0; j < limb; ++j) input.a[j] = maximum;
                    }
                    input.carry_in = carry;
                    check_case(input, counts);
                    input.a.swap(input.b);
                    check_case(input, counts);
                }
            }
        }
    }
}

struct Transfer { unsigned g; unsigned p; };

unsigned apply(Transfer transfer, unsigned carry) {
    return transfer.g | (transfer.p & carry);
}

Transfer after(Transfer high, Transfer low) {
    return {high.g | (high.p & low.g), high.p & low.p};
}

bool equivalent(Transfer left, Transfer right) {
    return apply(left, 0) == apply(right, 0) && apply(left, 1) == apply(right, 1);
}

void check_transfer_algebra(Counts& counts) {
    for (unsigned width = 1; width <= 4; ++width) {
        const unsigned base = 1U << width;
        for (unsigned a = 0; a < base; ++a) for (unsigned b = 0; b < base; ++b) {
            const unsigned raw = (a + b) % base;
            const Transfer transfer{(a + b) / base, unsigned(raw == base - 1)};
            require(!(transfer.g && transfer.p), "impossible canonical GP overlap");
            for (unsigned carry = 0; carry < 2; ++carry) {
                require(apply(transfer, carry) == (a + b + carry) / base,
                        "tiny digit transfer mismatch");
            }
            ++counts.digit_pairs;
        }
    }
    for (unsigned h = 0; h < 4; ++h) for (unsigned l = 0; l < 4; ++l) {
        const Transfer high{h >> 1, h & 1}, low{l >> 1, l & 1};
        for (unsigned carry = 0; carry < 2; ++carry) {
            require(apply(after(high, low), carry) == apply(high, apply(low, carry)),
                    "Boolean encoding composition mismatch");
        }
        ++counts.encoding_pairs;
    }
    const std::array<Transfer, 3> states = {{{0, 0}, {0, 1}, {1, 0}}};
    for (Transfer a : states) for (Transfer b : states) for (Transfer c : states) {
        require(equivalent(after(a, after(b, c)), after(after(a, b), c)),
                "carry transfer associativity mismatch");
        ++counts.associative_triples;
    }
    for (Transfer state : states) {
        require(equivalent(after(states[1], state), state) &&
                equivalent(after(state, states[1]), state), "propagate identity mismatch");
    }
    require(!equivalent(states[0], states[1]) && !equivalent(states[0], states[2]) &&
            !equivalent(states[1], states[2]), "three carry functions not distinguished");
    require(!equivalent(after(states[0], states[2]), after(states[2], states[0])),
            "reversed significance-order negative control failed");
}

void check_tiny_reblocking(Counts& counts) {
    for (unsigned width = 1; width <= 4; ++width) {
        const Word base = Word{1} << width;
        for (unsigned limbs = 1; limbs <= 4; ++limbs) {
            if (width * limbs > 6) continue;
            ++counts.layouts;
            const Word modulus = Word{1} << (width * limbs);
            for (Word a = 0; a < modulus; ++a) for (Word b = 0; b < modulus; ++b) {
                for (unsigned carry_in = 0; carry_in < 2; ++carry_in) {
                    ++counts.tiny_inputs;
                    const Word total = a + b + carry_in;
                    const Word expected_low = total % modulus;
                    const Word expected_carry = total / modulus;
                    std::array<Word, 4> raw{};
                    std::array<Transfer, 4> transfers{};
                    for (unsigned i = 0; i < limbs; ++i) {
                        const Word a_digit = (a >> (i * width)) & (base - 1);
                        const Word b_digit = (b >> (i * width)) & (base - 1);
                        const Word sum = a_digit + b_digit;
                        raw[i] = sum % base;
                        transfers[i] = {unsigned(sum / base), unsigned(raw[i] == base - 1)};
                    }
                    for (unsigned cuts = 0; cuts < (1U << (limbs - 1)); ++cuts) {
                        Word low = 0;
                        unsigned carry = carry_in;
                        unsigned begin = 0;
                        while (begin < limbs) {
                            unsigned end = begin + 1;
                            while (end < limbs && !(cuts & (1U << (end - 1)))) ++end;
                            Transfer summary{0, 1};
                            for (unsigned i = begin; i < end; ++i) {
                                summary = after(transfers[i], summary);
                            }
                            const unsigned block_out = apply(summary, carry);
                            for (unsigned i = begin; i < end; ++i) {
                                const Word digit = (raw[i] + carry) % base;
                                require(digit == ((expected_low >> (i * width)) & (base - 1)),
                                        "reblocking low digit mismatch");
                                low |= digit << (i * width);
                                carry = apply(transfers[i], carry);
                            }
                            require(carry == block_out, "ordered block summary mismatch");
                            begin = end;
                        }
                        require(low == expected_low && carry == expected_carry &&
                                low + modulus * carry == total, "reblocking full output mismatch");
                        const unsigned byte_count = (width * limbs + 7) / 8;
                        for (unsigned j = 0; j < byte_count; ++j) {
                            require(static_cast<unsigned char>(low >> (8 * j)) ==
                                    static_cast<unsigned char>(expected_low >> (8 * j)),
                                    "reblocking low bytes mismatch");
                        }
                        ++counts.tiny_outputs;
                    }
                }
            }
        }
    }
    require(counts.layouts == 10 && counts.tiny_inputs == 18248 && counts.tiny_outputs == 55528,
            "exhaustive reblocking corpus count changed");
}

void check_negative_controls(Counts& counts) {
    Input input{};
    input.a.fill(maximum);
    input.b[0] = 1;
    const Output expected = oracle(input);
    Output changed = expected;
    changed.low[2] ^= Word{1} << 47;
    require(!same_output(changed, expected), "low mutation was not detected");
    changed = expected;
    changed.carry_out ^= 1;
    require(!same_output(changed, expected), "carry mutation was not detected");
    changed = expected;
    changed.carry_out = 2;
    require(!same_output(changed, expected), "non-bit output carry was not detected");
    input.carry_in = 2;
    require(!pa_l0_native::valid_input(input), "input carry two was not rejected");
    input.carry_in = maximum;
    require(!pa_l0_native::valid_input(input), "maximum input carry was not rejected");
    // Invalid inputs are deliberately never passed to unchecked compute().
    for (Word a = 0; a < 4; ++a) for (Word b = 0; b < 4; ++b) for (Word c = 0; c < 4; ++c) {
        const Word left_sum = ((a + b) % 4) + c;
        const Word right_sum = a + ((b + c) % 4);
        require(left_sum % 4 == (a + b + c) % 4 && right_sum % 4 == (a + b + c) % 4,
                "modulo low associativity mismatch");
        if (left_sum / 4 != right_sum / 4) ++counts.lossy_nonassociative;
        if ((a + b + c) / 4 == 2) ++counts.needs_high_two;
    }
    require(counts.lossy_nonassociative == 20 && counts.needs_high_two == 4,
            "discarded high-state negative-control counts changed");
}

} // namespace

int main(int argc, char**) {
    if (argc != 1) {
        std::cerr << "usage: test_l0_native (no arguments)\n";
        return 2;
    }
    try {
        static_assert(sizeof(Word) == 8, "64-bit word required");
        static_assert(sizeof(Input) == 72 && sizeof(Output) == 40, "unexpected record layout");
        Counts counts;
        check_boundaries(counts);
        const Word boundary_cases = counts.cases;
        Word state = seed;
        for (Word iteration = 0; iteration < random_cases; ++iteration) {
            Input input{};
            for (unsigned limb = 0; limb < 4; ++limb) {
                input.a[limb] = next_random(state);
                input.b[limb] = next_random(state);
            }
            input.carry_in = next_random(state) & 1;
            check_case(input, counts);
        }
        check_transfer_algebra(counts);
        check_tiny_reblocking(counts);
        check_negative_controls(counts);
        require(counts.digit_pairs == 340 && counts.encoding_pairs == 16 &&
                counts.associative_triples == 27, "carry algebra corpus count changed");
#ifdef SECP256K1_NO_INT128
        constexpr const char* backend = "portable_carry";
#else
        constexpr const char* backend = "native_int128_carry";
#endif
        std::cout << "{\"schema\":\"pa_l0_native_correctness_v1\",\"status\":\"pass\","
                  << "\"backend\":\"" << backend << "\",\"seed\":" << seed
                  << ",\"random_cases\":" << random_cases << ",\"boundary_cases\":" << boundary_cases
                  << ",\"full_width_cases\":" << counts.cases << ",\"methods\":4"
                  << ",\"method_byte_carry_checks\":" << counts.checks
                  << ",\"oracle\":\"boost::multiprecision::cpp_int\",\"mismatches\":0"
                  << ",\"output_checksum_fnv1a64\":\"" << std::hex << std::setfill('0')
                  << std::setw(16) << counts.checksum << std::dec << "\""
                  << ",\"tiny_layouts\":" << counts.layouts
                  << ",\"tiny_input_triples_across_layouts\":" << counts.tiny_inputs
                  << ",\"tiny_complete_output_checks\":" << counts.tiny_outputs
                  << ",\"digit_pairs\":" << counts.digit_pairs
                  << ",\"gp_encoding_compositions\":" << counts.encoding_pairs
                  << ",\"associative_transfer_triples\":" << counts.associative_triples
                  << ",\"lossy_nonassociative_triples\":" << counts.lossy_nonassociative
                  << ",\"triples_requiring_high_two\":" << counts.needs_high_two
                  << ",\"negative_controls\":\"pass\",\"timing_claim\":false}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "native correctness failure: " << error.what() << '\n';
        return 1;
    }
}
