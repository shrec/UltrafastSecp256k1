// P6 finite correctness/ABI corpus. No performance or CT certification claims.
// Inline and separately compiled outlined kernels have the same canonical
// arithmetic domain. Direct ASM is admitted only after a caller capability gate.
#include "../probes/p6_field_boundary.hpp"
#include "secp256k1/field.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
using boost::multiprecision::cpp_int;
using FE = secp256k1::fast::FieldElement;
using Limbs = pa_p6::Limbs;
using Bytes = std::array<std::uint8_t, 32>;
const cpp_int kB = cpp_int(1) << 256;
const cpp_int kP = kB - (cpp_int(1) << 32) - 977;
constexpr std::uint64_t kSeed = UINT64_C(0x5041365f424f554e);
constexpr std::size_t kRandomPairs = 10000;
constexpr std::size_t kChainsPerOperation = 4;
constexpr std::size_t kChainLength = 1024;
constexpr std::uint64_t kFnvOffset = UINT64_C(14695981039346656037);

struct Counts {
    std::uint64_t assertions = 0, pair_entries = 0, named_kats = 0;
    std::uint64_t inline_outputs = 0, outlined_outputs = 0, asm_outputs = 0;
    std::uint64_t raw_abi_outputs = 0, raw_abi_canary_checks = 0;
    std::uint64_t same_object_wrapper_products = 0, same_object_abi_products = 0;
    std::uint64_t reference_checks = 0, output_checks = 0, preservation_checks = 0;
    std::uint64_t asm_outputs_skipped = 0, fail_closed_attempts = 0, fail_closed_rejections = 0;
    std::uint64_t chain_cases = 0, chain_steps = 0, negative_controls = 0;
    std::uint64_t shared_checksum = kFnvOffset, asm_checksum = kFnvOffset;
    bool asm_available = false;
    std::set<std::pair<Limbs, Limbs>> distinct_pairs;
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

Limbs encode_limbs(cpp_int value) {
    if (value < 0 || value >= kB) throw std::runtime_error("oracle 256-bit limb range");
    const cpp_int mask = (cpp_int(1) << 64) - 1;
    Limbs result{};
    for (auto& limb : result) {
        limb = static_cast<std::uint64_t>(value & mask);
        value >>= 64;
    }
    return result;
}

cpp_int decode_limbs(const Limbs& limbs) {
    cpp_int value = 0;
    for (std::size_t i = 4; i != 0; --i) {
        value *= cpp_int(1) << 64;
        value += limbs[i - 1];
    }
    return value;
}

Bytes encode_bytes(cpp_int value) {
    if (value < 0 || value >= kB) throw std::runtime_error("oracle 256-bit byte range");
    Bytes result{};
    for (std::size_t i = 32; i != 0; --i) {
        result[i - 1] = static_cast<std::uint8_t>(value & 255);
        value >>= 8;
    }
    return result;
}

std::string hex_bytes(const Bytes& bytes) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (auto byte : bytes) out << std::setw(2) << static_cast<unsigned>(byte);
    return out.str();
}

bool matches(const Limbs& limbs, const Bytes& bytes, const cpp_int& expected) {
    const auto el = encode_limbs(expected);
    const auto eb = encode_bytes(expected);
    bool result = expected >= 0 && expected < kP && decode_limbs(limbs) < kP;
    for (std::size_t i = 0; i < 4; ++i) result &= limbs[i] == el[i];
    for (std::size_t i = 0; i < 32; ++i) result &= bytes[i] == eb[i];
    return result;
}

void check_observation(const Limbs& raw, const Bytes& bytes, const cpp_int& expected,
                       Counts& c, const std::string& context, bool assembly) {
    if (!matches(raw, bytes, expected))
        require(false, c, context + " actual_raw_be=" + hex_bytes(encode_bytes(decode_limbs(raw))) +
                " actual_serialized_be=" + hex_bytes(bytes) + " expected_be=" + hex_bytes(encode_bytes(expected)));
    auto& checksum = assembly ? c.asm_checksum : c.shared_checksum;
    for (auto byte : bytes) checksum = (checksum ^ byte) * UINT64_C(1099511628211);
    ++c.output_checks;
}

void check_result(const Limbs& actual, const cpp_int& expected, Counts& c,
                  const std::string& context, bool assembly = false) {
    const Limbs raw = actual;
    const FE value = FE::from_limbs_raw(raw);
    check_observation(raw, value.to_bytes(), expected, c, context, assembly);
    require(value.limbs() == raw && actual == raw, c, context + ": serialization mutation");
}

void check_reference(const FE& actual, const cpp_int& expected, Counts& c, const std::string& context) {
    const Limbs raw = actual.limbs();
    check_observation(raw, actual.to_bytes(), expected, c, context, false);
    require(actual.limbs() == raw, c, context + ": reference serialization mutation");
    ++c.reference_checks;
}

constexpr Limbs kBefore{{UINT64_C(0x1122334455667788), UINT64_C(0x89abcdef01234567),
                         UINT64_C(0xfedcba9876543210), UINT64_C(0x8877665544332211)}};
constexpr Limbs kAfter{{UINT64_C(0x55aa55aa01234567), UINT64_C(0x123456789abcdef0),
                        UINT64_C(0x0f1e2d3c4b5a6978), UINT64_C(0xc3d2e1f001234567)}};
constexpr Limbs kUnwritten{{UINT64_C(0xfeedfacecafebeef), UINT64_C(0x13579bdf2468ace0),
                            UINT64_C(0x0123456789abcdef), UINT64_C(0xa5a5a5a55a5a5a5a)}};
struct alignas(32) Guarded {
    Limbs before = kBefore;
    Limbs words = kUnwritten;
    Limbs after = kAfter;
};
static_assert(offsetof(Guarded, words) == 32 && offsetof(Guarded, after) == 64);
static_assert(sizeof(Guarded) == 96 && std::is_trivially_copyable_v<Guarded>);

bool guards_match(const Guarded& value) {
    return value.before == kBefore && value.after == kAfter;
}

Limbs raw_abi(bool square, const Limbs& a, const Limbs& b, bool same_object,
              Counts& c, const std::string& context) {
    Guarded input_a, input_b, output;
    input_a.words = a; input_b.words = b;
    const Guarded before_a = input_a, before_b = input_b;
    // Distinct complete objects: output is never aliased with either input.
    // Only the admitted read-only a==b alias is exercised.
    if (square) pa_p6_row_square(input_a.words.data(), output.words.data());
    else pa_p6_row_mul(input_a.words.data(),
                       same_object ? input_a.words.data() : input_b.words.data(), output.words.data());
    require(guards_match(output), c, context + ": raw C ABI output canary overwritten");
    require(std::memcmp(&input_a, &before_a, sizeof(Guarded)) == 0 &&
            std::memcmp(&input_b, &before_b, sizeof(Guarded)) == 0, c,
            context + ": raw C ABI input/canary mutation");
    ++c.raw_abi_canary_checks;
    ++c.preservation_checks;
    ++c.raw_abi_outputs;
    if (!square && same_object) ++c.same_object_abi_products;
    return output.words;
}

void pair_case(const cpp_int& ai, const cpp_int& bi, Counts& c, const std::string& label) {
    require(ai >= 0 && ai < kP && bi >= 0 && bi < kP, c, label + ": canonical input precondition");
    const Limbs a = encode_limbs(ai), b = encode_limbs(bi), before_a = a, before_b = b;
    const std::string context = label + " a=" + hex_bytes(encode_bytes(ai)) + " b=" + hex_bytes(encode_bytes(bi));
    const cpp_int product = mod(ai * bi), square = mod(ai * ai);
    check_result(pa_p6::row_mul_inline(a, b), product, c, context + " inline mul");
    check_result(pa_p6::row_square_inline(a), square, c, context + " inline square");
    check_result(pa_p6::row_mul_inline(a, a), square, c, context + " inline same-object mul");
    c.inline_outputs += 3;
    check_result(pa_p6::row_mul_outlined(a, b), product, c, context + " outlined mul");
    check_result(pa_p6::row_square_outlined(a), square, c, context + " outlined square");
    check_result(pa_p6::row_mul_outlined(a, a), square, c, context + " outlined same-object mul");
    c.outlined_outputs += 3;
    c.same_object_wrapper_products += 2;
    check_result(raw_abi(false, a, b, false, c, context), product, c, context + " raw ABI mul");
    check_result(raw_abi(true, a, b, false, c, context), square, c, context + " raw ABI square");
    check_result(raw_abi(false, a, a, true, c, context), square, c, context + " raw ABI same-object mul");
    // This admission check dominates every native direct-ASM call below.
    if (c.asm_available) {
        check_result(pa_p6::asm_mul_direct(a, b), product, c, context + " direct ASM mul", true);
        check_result(pa_p6::asm_square_direct(a), square, c, context + " direct ASM square", true);
        check_result(pa_p6::asm_mul_direct(a, a), square, c, context + " direct ASM same-object mul", true);
        c.asm_outputs += 3;
        ++c.same_object_wrapper_products;
    } else c.asm_outputs_skipped += 3;
    require(a == before_a && b == before_b, c, context + ": wrapper inputs changed");
    ++c.preservation_checks;
    const FE fa = FE::from_bytes(encode_bytes(ai)), fb = FE::from_bytes(encode_bytes(bi));
    const Limbs fa_before = fa.limbs(), fb_before = fb.limbs();
    check_reference(fa, ai, c, context + " FE64 lhs construction");
    check_reference(fb, bi, c, context + " FE64 rhs construction");
    check_reference(fa * fb, product, c, context + " corrected FE64 multiply");
    check_reference(fa.square(), square, c, context + " corrected FE64 square");
    check_reference(fa * fa, square, c, context + " corrected FE64 same-object multiply");
    require(fa.limbs() == fa_before && fb.limbs() == fb_before, c, context + ": FE64 inputs changed");
    ++c.preservation_checks;
    ++c.pair_entries;
    c.distinct_pairs.emplace(a, b);
}

struct SplitMix64 {
    std::uint64_t state;
    std::uint64_t next() {
        std::uint64_t z = (state += UINT64_C(0x9e3779b97f4a7c15));
        z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
        return z ^ (z >> 31);
    }
    cpp_int integer() {
        cpp_int value = 0;
        for (unsigned i = 0; i < 4; ++i) value += cpp_int(next()) << (64U * i);
        return value;
    }
    cpp_int nonzero() {
        cpp_int value;
        do { value = mod(integer()); } while (value == 0);
        return value;
    }
};

void chains(Counts& c) {
    SplitMix64 random{kSeed ^ UINT64_C(0x434841494e535036)};
    for (unsigned kind = 0; kind < 2; ++kind) {
        for (std::size_t chain = 0; chain < kChainsPerOperation; ++chain) {
            cpp_int value = random.nonzero();
            Limbs inlined = encode_limbs(value), outlined = inlined, abi = inlined, assembly = inlined;
            FE reference = FE::from_bytes(encode_bytes(value));
            for (std::size_t step = 0; step < kChainLength; ++step) {
                const cpp_int rhs_int = random.nonzero();
                const Limbs rhs = encode_limbs(rhs_int), rhs_before = rhs;
                const FE rhs_fe = FE::from_bytes(encode_bytes(rhs_int));
                const std::string at = std::string(kind == 0 ? "mul" : "square") +
                    " chain" + std::to_string(chain) + " step" + std::to_string(step);
                if (kind == 0) {
                    value = mod(value * rhs_int);
                    inlined = pa_p6::row_mul_inline(inlined, rhs);
                    outlined = pa_p6::row_mul_outlined(outlined, rhs);
                    abi = raw_abi(false, abi, rhs, false, c, at);
                    if (c.asm_available) assembly = pa_p6::asm_mul_direct(assembly, rhs);
                    reference = reference * rhs_fe;
                } else {
                    value = mod(value * value);
                    inlined = pa_p6::row_square_inline(inlined);
                    outlined = pa_p6::row_square_outlined(outlined);
                    abi = raw_abi(true, abi, rhs, false, c, at);
                    if (c.asm_available) assembly = pa_p6::asm_square_direct(assembly);
                    reference = reference.square();
                }
                check_result(inlined, value, c, at + " inline");
                check_result(outlined, value, c, at + " outlined");
                check_result(abi, value, c, at + " raw C ABI");
                ++c.inline_outputs; ++c.outlined_outputs;
                if (c.asm_available) {
                    check_result(assembly, value, c, at + " direct ASM", true);
                    ++c.asm_outputs;
                } else ++c.asm_outputs_skipped;
                check_reference(reference, value, c, at + " corrected FE64 reference");
                require(rhs == rhs_before, c, at + ": chain RHS changed");
                ++c.preservation_checks;
                ++c.chain_steps;
            }
            ++c.chain_cases;
        }
    }
}

void fail_closed_checks(Counts& c) {
    // Compile-disabled wrappers throw; a runtime-unsupported native binary
    // must NOT call them, since its direct wrappers contain unguarded ASM.
    if constexpr (!pa_p6::native_asm_compiled) {
        require(!c.asm_available, c, "compile-disabled ASM capability unexpectedly true");
        const Limbs a{{1, 0, 0, 0}}, b{{2, 0, 0, 0}};
        bool rejected = false;
        ++c.fail_closed_attempts;
        try { (void)pa_p6::asm_mul_direct(a, b); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, c, "compile-disabled direct multiply did not fail closed");
        ++c.fail_closed_rejections;
        rejected = false;
        ++c.fail_closed_attempts;
        try { (void)pa_p6::asm_square_direct(a); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, c, "compile-disabled direct square did not fail closed");
        ++c.fail_closed_rejections;
    }
}

void controls(Counts& c) {
    Bytes bytes{};
    for (std::size_t i = 0; i < 32; ++i) bytes[i] = static_cast<std::uint8_t>(i + 1);
    const cpp_int expected("0x0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20");
    const Limbs words{{UINT64_C(0x191a1b1c1d1e1f20), UINT64_C(0x1112131415161718),
                       UINT64_C(0x090a0b0c0d0e0f10), UINT64_C(0x0102030405060708)}};
    require(matches(words, bytes, expected), c, "known-endian positive comparator control");
    for (std::size_t i = 0; i < 32; ++i) {
        auto bad = bytes; bad[i] ^= UINT8_C(0x80);
        require(!matches(words, bad, expected), c, "missed output byte corruption");
        ++c.negative_controls;
    }
    for (std::size_t i = 0; i < 4; ++i) {
        auto bad = words; bad[i] ^= UINT64_C(0x8000000000000000);
        require(!matches(bad, bytes, expected), c, "missed raw output limb corruption");
        ++c.negative_controls;
    }
    require(!matches(encode_limbs(kP), encode_bytes(kP), kP), c, "noncanonical p accepted");
    ++c.negative_controls;
    for (std::size_t i = 0; i < 4; ++i) {
        Guarded bad_before, bad_after;
        bad_before.before[i] ^= 1;
        bad_after.after[i] ^= 1;
        require(!guards_match(bad_before) && !guards_match(bad_after), c, "missed output canary corruption");
        c.negative_controls += 2;
    }
}

void kats(Counts& c) {
    const cpp_int a1 = kB - (cpp_int(1) << 33) - 1;
    require(mod(a1 * a1) == cpp_int(UINT64_C(0xfffff860000e8900)), c, "P1 second-fold KAT independent truth");
    pair_case(a1, a1, c, "P1 second-fold carry KAT"); ++c.named_kats;
    const cpp_int a2 = (cpp_int(1) << 255) + 1;
    const Limbs expected2{{UINT64_C(0xfffffffefffffc2e), UINT64_MAX, UINT64_MAX, UINT64_C(0x7fffffffffffffff)}};
    require(mod(a2 * (kP - 1)) == decode_limbs(expected2), c, "P1 first-fold KAT independent truth");
    pair_case(a2, kP - 1, c, "P1 first-fold cascade KAT"); ++c.named_kats;
    const cpp_int a3 = (cpp_int(1) << 255) - 1;
    const cpp_int expected3("0x400000000000000000000000000000000000000000000000400001e740039f64");
    require(mod(a3 * a3) == expected3, c, "older large-square KAT independent truth");
    pair_case(a3, a3, c, "older large-square carry KAT"); ++c.named_kats;
}

Counts run() {
    Counts c;
    // Runtime capability query occurs once before any possible native ASM.
    c.asm_available = pa_p6::native_asm_available();
    controls(c);
    fail_closed_checks(c);
    kats(c);
    std::vector<cpp_int> boundary{cpp_int(0), cpp_int(1), cpp_int(2), kP - 2, kP - 1,
                                 kP, kP + 1, kB - 2, kB - 1};
    for (unsigned bit = 0; bit < 256; ++bit) {
        const cpp_int power = cpp_int(1) << bit;
        boundary.push_back(power - 1); boundary.push_back(power); boundary.push_back(power + 1);
        boundary.push_back(kB - 1 - power);
    }
    for (std::size_t i = 0; i < boundary.size(); ++i) {
        const cpp_int a = mod(boundary[i]);
        const std::array<cpp_int, 3> partners{{cpp_int(1), kP - 1, mod(boundary[(i + 1) % boundary.size()])}};
        for (std::size_t j = 0; j < partners.size(); ++j)
            pair_case(a, partners[j], c, "boundary " + std::to_string(i) + "/" + std::to_string(j));
    }
    SplitMix64 random{kSeed};
    for (std::size_t i = 0; i < kRandomPairs; ++i) {
        const cpp_int a = mod(random.integer()), b = mod(random.integer());
        pair_case(a, b, c, "random pair " + std::to_string(i));
    }
    chains(c);
    require(c.pair_entries == 13102 && c.named_kats == 3, c, "pair/KAT fixture counts");
    require(c.chain_cases == 8 && c.chain_steps == 8192, c, "1024-step chain counts");
    require(c.inline_outputs == 47498 && c.outlined_outputs == 47498 && c.raw_abi_outputs == 47498,
            c, "portable route output counts");
    require(c.raw_abi_canary_checks == c.raw_abi_outputs && c.same_object_abi_products == c.pair_entries,
            c, "raw ABI canary/alias counts");
    require(c.asm_outputs + c.asm_outputs_skipped == 47498, c, "direct ASM qualification/skip counts");
    require(c.negative_controls == 45, c, "negative comparator/canary controls");
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
    if (argc != 1) { std::cerr << "Usage: test_p6_field_boundary (JSON stdout)\n"; return 2; }
    try {
        const Counts c = run();
        std::cout << std::boolalpha
                  << "{\"schema\":\"parseatlas_p6_field_boundary_correctness_v1\",\"status\":\"pass\",\"mismatches\":0,"
                  << "\"finite_corpus_only\":true,\"timing_claim\":false,\"constant_time_claim\":false,"
                  << "\"oracle\":\"Boost canonical modulo-p products/squares plus corrected FE64\","
                  << "\"raw_output_checks\":\"four canonical raw limbs before serialization and all32 BE bytes\","
                  << "\"raw_C_ABI_canaries\":\"32 bytes before/after 32-byte output; disjoint aligned input/output; all input bytes preserved\","
                  << "\"asm_compiled\":" << pa_p6::native_asm_compiled << ",\"asm_runtime_available\":" << c.asm_available
                  << ",\"asm_sanitizer_instrumentation_claim\":false,\"output_alias_tested\":false,"
                  << "\"fixture_entries_may_repeat\":true,\"distinct_pair_count_excludes_derived_chain_states\":true,"
                  << "\"seed_hex\":\"" << hex64(kSeed) << "\",\"pair_entries\":" << c.pair_entries
                  << ",\"distinct_pairs\":" << c.distinct_pairs.size() << ",\"named_carry_KATs\":" << c.named_kats
                  << ",\"inline_outputs\":" << c.inline_outputs << ",\"outlined_outputs\":" << c.outlined_outputs
                  << ",\"asm_outputs\":" << c.asm_outputs << ",\"asm_outputs_skipped\":" << c.asm_outputs_skipped
                  << ",\"compile_disabled_wrapper_attempts\":" << c.fail_closed_attempts
                  << ",\"compile_disabled_wrapper_rejections\":" << c.fail_closed_rejections
                  << ",\"raw_abi_outputs\":" << c.raw_abi_outputs << ",\"raw_abi_canary_checks\":" << c.raw_abi_canary_checks
                  << ",\"same_object_wrapper_products\":" << c.same_object_wrapper_products
                  << ",\"same_object_abi_products\":" << c.same_object_abi_products
                  << ",\"reference_checks\":" << c.reference_checks << ",\"output_checks\":" << c.output_checks
                  << ",\"preservation_checks\":" << c.preservation_checks
                  << ",\"chain_cases\":" << c.chain_cases << ",\"chain_length\":" << kChainLength
                  << ",\"derived_chain_steps\":" << c.chain_steps << ",\"negative_controls\":" << c.negative_controls
                  << ",\"assertions\":" << c.assertions
                  << ",\"shared_checksum_fnv1a64\":\"" << hex64(c.shared_checksum)
                  << "\",\"asm_checksum_fnv1a64\":\"" << hex64(c.asm_checksum) << "\"}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cout << "{\"schema\":\"parseatlas_p6_field_boundary_correctness_v1\",\"status\":\"fail\",\"error\":\""
                  << json_escape(error.what()) << "\"}\n";
        return 1;
    }
}
