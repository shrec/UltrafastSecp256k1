// Research-only original add64 composition. No field/scalar modular arithmetic.
// Timings describe this complete harness region, never isolated ADC latency.
#include "secp256k1/detail/arith64.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Word = std::uint64_t;
constexpr Word kMaxOperations = 1000000000;
constexpr Word kMaxCount = Word{1} << 24;
constexpr Word kMaxWorkingBytes = Word{256} << 20;

template <std::size_t L> struct Input {
    std::array<Word, L> a;
    std::array<Word, L> b;
    Word carry;
};
template <std::size_t L> struct Output {
    std::array<Word, L> low;
    Word carry;
};

template <std::size_t L>
inline void original_add(const std::array<Word, L>& a,
                         const std::array<Word, L>& b,
                         unsigned char& carry, std::array<Word, L>& low) {
    for (std::size_t j = 0; j < L; ++j) {
        low[j] = secp256k1::detail::add64(a[j], b[j], carry);
    }
}

// An opaque memory use between passes makes every pass's stores observable to
// the compiler. The instruction has no hardware memory-fence or cache-flush cost.
// This baseline intentionally targets the recorded GCC/Clang build, not MSVC.
inline void observe_memory(const void* p) {
    asm volatile("" : : "g"(p) : "memory");
}

Word decimal(const std::string& token) {
    if (token.empty()) throw std::invalid_argument("empty decimal");
    Word result = 0;
    for (char ch : token) {
        if (ch < '0' || ch > '9') throw std::invalid_argument("invalid decimal");
        const Word digit = static_cast<Word>(ch - '0');
        if (result > (std::numeric_limits<Word>::max() - digit) / 10)
            throw std::invalid_argument("decimal overflow");
        result = result * 10 + digit;
    }
    return result;
}

Word hex_word(const std::string& token) {
    if (token.size() != 16) throw std::invalid_argument("word must have 16 lowercase hex digits");
    Word result = 0;
    for (char ch : token) {
        unsigned digit;
        if (ch >= '0' && ch <= '9') digit = static_cast<unsigned>(ch - '0');
        else if (ch >= 'a' && ch <= 'f') digit = static_cast<unsigned>(ch - 'a' + 10);
        else throw std::invalid_argument("invalid lowercase hex word");
        result = (result << 4) | digit;
    }
    return result;
}

std::string hex(Word value) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << value;
    return out.str();
}

template <std::size_t L> void check_stream() {
    std::string line;
    std::size_t cases = 0;
    while (std::getline(std::cin, line)) {
        if (++cases > 1000000 || line.size() > 1024)
            throw std::invalid_argument("correctness input limit");
        std::istringstream input(line);
        std::vector<std::string> tokens;
        std::string token;
        while (input >> token) tokens.push_back(token);
        if (tokens.size() != 2 * L + 1)
            throw std::invalid_argument("incorrect correctness record arity");
        if (tokens[0] != "0" && tokens[0] != "1")
            throw std::invalid_argument("carry must be exactly 0 or 1");
        std::array<Word, L> a{}, b{}, low{};
        for (std::size_t j = 0; j < L; ++j) {
            a[j] = hex_word(tokens[1 + j]);
            b[j] = hex_word(tokens[1 + L + j]);
        }
        unsigned char carry = static_cast<unsigned char>(tokens[0][0] - '0');
        original_add(a, b, carry, low);
        for (Word word : low) std::cout << hex(word) << ' ';
        std::cout << static_cast<unsigned>(carry) << '\n';
    }
    if (!std::cin.eof()) throw std::invalid_argument("stdin read failure");
    if (!cases) throw std::invalid_argument("empty correctness stream");
}

// Deterministic public benchmark data, not cryptographic randomness.
Word splitmix64(Word& state) {
    Word z = (state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

template <std::size_t L> Input<L> make_input(Word& state) {
    Input<L> result{};
    for (Word& word : result.a) word = splitmix64(state);
    for (Word& word : result.b) word = splitmix64(state);
    result.carry = splitmix64(state) & 1;
    return result;
}

inline Word fold(Word checksum, Word value) {
    return (checksum ^ value) * UINT64_C(0x100000001b3);
}

template <std::size_t L>
void bulk_pass(const Input<L>* input, Output<L>* output, std::size_t count) {
    // One pinned-core baseline is intentionally sequential: this measures one
    // core's region behavior. Multicore throughput is a different experiment.
    for (std::size_t i = 0; i < count; ++i) {
        unsigned char carry = static_cast<unsigned char>(input[i].carry);
        original_add(input[i].a, input[i].b, carry, output[i].low);
        output[i].carry = carry;
    }
    observe_memory(output);
}

template <std::size_t L>
void dependent_steps(std::array<Word, L>& a, std::array<Word, L>& b,
                     unsigned char& carry, Word count) {
    // All result limbs and carry feed the next operation. One rotate and one
    // XOR per limb make b runtime-dependent as well, preventing a fixed addend
    // recurrence from being replaced by a closed-form loop. These operations,
    // loop control and any register spills are INCLUDED in measured latency.
    for (Word i = 0; i < count; ++i) {
        original_add(a, b, carry, a);
        for (std::size_t j = 0; j < L; ++j)
            b[j] = ((b[j] << 13) | (b[j] >> 51)) ^ a[j];
    }
}

struct Measurement {
    Word elapsed_ns;
    Word checksum;
    Word warmup_operations;
};

template <std::size_t L> Measurement bench_bulk(Word count, Word passes, Word seed) {
    std::vector<Input<L>> input(static_cast<std::size_t>(count));
    std::vector<Output<L>> output(static_cast<std::size_t>(count));
    for (auto& record : input) record = make_input<L>(seed);
    bulk_pass(input.data(), output.data(), input.size());
    observe_memory(input.data());
    observe_memory(output.data());
    const auto start = std::chrono::steady_clock::now();
    for (Word pass = 0; pass < passes; ++pass)
        bulk_pass(input.data(), output.data(), input.size());
    observe_memory(output.data());
    const auto stop = std::chrono::steady_clock::now();
    Word checksum = UINT64_C(0xcbf29ce484222325);
    for (const auto& result : output) {
        for (Word value : result.low) checksum = fold(checksum, value);
        checksum = fold(checksum, result.carry);
    }
    return {static_cast<Word>(std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count()),
            checksum, count};
}

template <std::size_t L> Measurement bench_latency(Word count, Word passes, Word seed) {
    auto state = make_input<L>(seed);
    unsigned char carry = static_cast<unsigned char>(state.carry);
    const Word warmup = std::min<Word>(count, 4096);
    dependent_steps(state.a, state.b, carry, warmup);
    state.carry = carry;
    observe_memory(&state);
    carry = static_cast<unsigned char>(state.carry);
    const auto start = std::chrono::steady_clock::now();
    for (Word pass = 0; pass < passes; ++pass)
        dependent_steps(state.a, state.b, carry, count);
    state.carry = carry;
    observe_memory(&state);
    const auto stop = std::chrono::steady_clock::now();
    Word checksum = UINT64_C(0xcbf29ce484222325);
    for (Word value : state.a) checksum = fold(checksum, value);
    for (Word value : state.b) checksum = fold(checksum, value);
    checksum = fold(checksum, state.carry);
    return {static_cast<Word>(std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count()),
            checksum, warmup};
}

template <std::size_t L>
void benchmark(const std::string& mode, Word count, Word passes, Word seed) {
    static_assert(sizeof(Input<L>) == (2 * L + 1) * sizeof(Word));
    static_assert(sizeof(Output<L>) == (L + 1) * sizeof(Word));
    if (mode != "latency" && mode != "bulk") throw std::invalid_argument("unknown mode");
    if (!count || count > kMaxCount || !passes || passes > 1000000 ||
        count > kMaxOperations / passes)
        throw std::invalid_argument("count/passes/operation resource limit");
    const Word record_bytes = sizeof(Input<L>) + sizeof(Output<L>);
    if (mode == "bulk" && count > kMaxWorkingBytes / record_bytes)
        throw std::invalid_argument("bulk allocation exceeds 256 MiB");
    const Word operations = count * passes;
    const Word logical_bytes = mode == "bulk" ? record_bytes : 0;
    const Word working_bytes = mode == "bulk" ? count * record_bytes : sizeof(Input<L>);
    const auto result = mode == "bulk" ? bench_bulk<L>(count, passes, seed)
                                      : bench_latency<L>(count, passes, seed);
#if defined(_MSC_VER) && !defined(__clang__)
    const char* backend = "msvc_addcarry";
#elif defined(SECP256K1_NO_INT128)
    const char* backend = "portable_carry";
#else
    const char* backend = "gcc_uint128";
#endif
#ifdef SECP256K1_NO_INT128
    const char* no_int128 = "true";
#else
    const char* no_int128 = "false";
#endif
#ifdef __SIZEOF_INT128__
    const char* int128_macro = "true";
#else
    const char* int128_macro = "false";
#endif
    std::cout << "{\"protocol\":\"parseatlas_l0_baseline_v1\","
              << "\"scope\":\"original_add64_composition\","
              << "\"mode\":\"" << mode << "\",\"limbs\":" << L
              << ",\"count\":" << count << ",\"passes\":" << passes
              << ",\"seed\":" << seed << ",\"operation_count\":" << operations
              << ",\"add64_calls\":" << operations * L
              << ",\"elapsed_ns\":" << result.elapsed_ns
              << ",\"input_record_bytes\":" << sizeof(Input<L>)
              << ",\"output_record_bytes\":" << sizeof(Output<L>)
              << ",\"working_set_bytes\":" << working_bytes
              << ",\"logical_bytes_per_operation\":" << logical_bytes
              << ",\"logical_bytes_processed\":" << logical_bytes * operations
              << ",\"checksum\":\"" << hex(result.checksum)
              << "\",\"warmup_operations\":" << result.warmup_operations
              << ",\"backend\":\"" << backend << "\",\"no_int128_defined\":" << no_int128
              << ",\"int128_macro_defined\":" << int128_macro << "}\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--check") {
            if (std::string(argv[2]) == "1") check_stream<1>();
            else if (std::string(argv[2]) == "4") check_stream<4>();
            else throw std::invalid_argument("limbs must be 1 or 4");
        } else if (argc == 7 && std::string(argv[1]) == "--bench") {
            const std::string mode = argv[3];
            const Word count = decimal(argv[4]), passes = decimal(argv[5]), seed = decimal(argv[6]);
            if (std::string(argv[2]) == "1") benchmark<1>(mode, count, passes, seed);
            else if (std::string(argv[2]) == "4") benchmark<4>(mode, count, passes, seed);
            else throw std::invalid_argument("limbs must be 1 or 4");
        } else {
            throw std::invalid_argument("usage: --check <1|4> OR --bench <1|4> <latency|bulk> <count> <passes> <seed>");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "l0_baseline: " << error.what() << '\n';
        return 2;
    }
}
