// Research-only, entirely native correctness and interleaved measurement driver.
// Complete four-limb additions, not Fp/Fn operations or isolated ADC latency.
#include "l0_native_kernels.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sched.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using pa_l0_native::Input;
using pa_l0_native::Method;
using pa_l0_native::Output;
using pa_l0_native::Word;
using Clock = std::chrono::steady_clock;
constexpr Word kMaxOperations = 1000000000;
constexpr Word kMaxPasses = 1000000;
constexpr Word kMaxWorkingBytes = Word{256} << 20;
constexpr Word kInitialChecksum = UINT64_C(0xcbf29ce484222325);
constexpr std::array<Method, 4> kMethods{{Method::Original, Method::GpMaterialized,
                                       Method::GpRecomputed, Method::Blocked2}};
static_assert(sizeof(Input) == 72 && sizeof(Output) == 40,
              "This registered AoS experiment requires 72/40 byte records");

enum class Regime { Latency, Hot, Stream };

const char* method_name(Method method) {
    switch (method) {
        case Method::Original: return "original";
        case Method::GpMaterialized: return "gp_materialized";
        case Method::GpRecomputed: return "gp_recomputed";
        case Method::Blocked2: return "blocked2";
    }
    throw std::logic_error("unknown method");
}

const char* regime_name(Regime regime) {
    switch (regime) {
        case Regime::Latency: return "conditioned_latency";
        case Regime::Hot: return "hot_bulk";
        case Regime::Stream: return "large_bulk";
    }
    throw std::logic_error("unknown regime");
}

void require(bool condition, const std::string& message) {
    // Assertions deliberately remain active in the -DNDEBUG measurement build.
    if (!condition) throw std::runtime_error(message);
}

Word decimal(const std::string& text) {
    if (text.empty()) throw std::invalid_argument("empty unsigned decimal");
    Word value = 0;
    for (char ch : text) {
        if (ch < '0' || ch > '9') throw std::invalid_argument("invalid unsigned decimal");
        const Word digit = static_cast<Word>(ch - '0');
        if (value > (std::numeric_limits<Word>::max() - digit) / 10)
            throw std::invalid_argument("unsigned decimal overflow");
        value = value * 10 + digit;
    }
    return value;
}

struct Options {
    bool smoke = false;
    unsigned cpu = 4;
    Word seed = 20260905;
    Word target_ms = 200;
};

Options parse_options(int argc, char** argv) {
    if (argc < 2 || (std::string(argv[1]) != "--measure" &&
                     std::string(argv[1]) != "--smoke"))
        throw std::invalid_argument("usage: l0_native_compare (--measure|--smoke) "
                                    "[--cpu N] [--seed N] [--target-ms 200..1000]");
    Options options;
    options.smoke = std::string(argv[1]) == "--smoke";
    bool cpu_seen = false, seed_seen = false, target_seen = false;
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc) throw std::invalid_argument("missing option value");
        const std::string key = argv[i];
        const Word value = decimal(argv[i + 1]);
        if (key == "--cpu" && !cpu_seen) {
            if (value >= CPU_SETSIZE) throw std::invalid_argument("CPU exceeds CPU_SETSIZE");
            options.cpu = static_cast<unsigned>(value);
            cpu_seen = true;
        } else if (key == "--seed" && !seed_seen) {
            options.seed = value;
            seed_seen = true;
        } else if (key == "--target-ms" && !target_seen) {
            if (value < 200 || value > 1000)
                throw std::invalid_argument("measurement target must be 200..1000 ms");
            options.target_ms = value;
            target_seen = true;
        } else {
            throw std::invalid_argument("unknown or duplicate option: " + key);
        }
    }
    if (options.smoke && target_seen)
        throw std::invalid_argument("--smoke has no performance target");
    return options;
}

std::vector<unsigned> pin_cpu(unsigned cpu) {
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0)
        throw std::runtime_error("sched_getaffinity failed");
    std::vector<unsigned> before;
    for (unsigned i = 0; i < CPU_SETSIZE; ++i)
        if (CPU_ISSET(i, &allowed)) before.push_back(i);
    if (!CPU_ISSET(cpu, &allowed))
        throw std::invalid_argument("requested CPU is outside inherited allowed affinity");
    cpu_set_t selected;
    CPU_ZERO(&selected);
    CPU_SET(cpu, &selected);
    if (sched_setaffinity(0, sizeof(selected), &selected) != 0)
        throw std::runtime_error("sched_setaffinity failed");
    if (sched_getcpu() != static_cast<int>(cpu))
        throw std::runtime_error("selected CPU was not established");
    return before;
}

Word splitmix64(Word& state) {
    Word z = (state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

Input make_input(Word& seed) {
    Input value{};
    for (Word& word : value.a) word = splitmix64(seed);
    for (Word& word : value.b) word = splitmix64(seed);
    value.carry_in = splitmix64(seed) & 1;
    return value;
}

inline Word fold(Word digest, Word word) {
    return (digest ^ word) * UINT64_C(0x100000001b3);
}

Word output_checksum(const std::vector<Output>& output) {
    Word result = kInitialChecksum;
    for (const auto& item : output) {
        for (Word word : item.low) result = fold(result, word);
        result = fold(result, item.carry_out);
    }
    return result;
}

Word state_checksum(const Input& state) {
    Word result = kInitialChecksum;
    for (Word word : state.a) result = fold(result, word);
    for (Word word : state.b) result = fold(result, word);
    return fold(result, state.carry_in);
}

bool same_output(const Output& left, const Output& right) {
    return left.low == right.low && left.carry_out == right.carry_out;
}

// GCC/Clang compiler barrier only: no volatile arithmetic or hardware fence.
// Every bulk pass's complete output array remains observable to the optimizer.
inline void observe_memory(const void* pointer) {
    asm volatile("" : : "g"(pointer) : "memory");
}

template <Method M>
inline void bulk_pass(const Input* input, Output* output, std::size_t count) {
    // Deliberately one serial worker: parallel measurements would perturb cache,
    // bandwidth and frequency. This experiment is not multicore throughput.
    for (std::size_t i = 0; i < count; ++i)
        output[i] = pa_l0_native::compute<M>(input[i]);
    observe_memory(output);
}

template <Method M>
inline void dependent_steps(Input& state, Word count) {
    for (Word step = 0; step < count; ++step) {
        const Output result = pa_l0_native::compute<M>(state);
        state.a = result.low;
        state.carry_in = result.carry_out;
        for (std::size_t limb = 0; limb < 4; ++limb)
            state.b[limb] = ((state.b[limb] << 13) | (state.b[limb] >> 51)) ^ state.a[limb];
    }
}

struct RegionClock {
    Word start_ns;
    Word stop_ns;
    Word elapsed_ns;
};

RegionClock clock_result(Clock::time_point start, Clock::time_point stop) {
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
    require(elapsed > 0, "nonpositive elapsed time");
    return {static_cast<Word>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  start.time_since_epoch()).count()),
            static_cast<Word>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  stop.time_since_epoch()).count()),
            static_cast<Word>(elapsed)};
}

// Identifiable, equal-policy region wrappers. Template dispatch happens BEFORE
// the clock; there is no function pointer or method switch per logical addition.
template <Method M>
__attribute__((noinline)) RegionClock timed_bulk(const Input* input, Output* output,
                                                std::size_t count, Word passes) {
    observe_memory(input);
    observe_memory(output);
    const auto start = Clock::now();
    for (Word pass = 0; pass < passes; ++pass) bulk_pass<M>(input, output, count);
    observe_memory(output);
    const auto stop = Clock::now();
    return clock_result(start, stop);
}

template <Method M>
__attribute__((noinline)) RegionClock timed_latency(Input& state, Word count, Word passes) {
    observe_memory(&state);
    const auto start = Clock::now();
    for (Word pass = 0; pass < passes; ++pass) dependent_steps<M>(state, count);
    observe_memory(&state);
    const auto stop = Clock::now();
    return clock_result(start, stop);
}

struct Workload {
    Regime regime;
    Word count;
    Word passes = 1;
    Word expected_checksum = 0;
    Word validation_outputs = 0;
    std::vector<Input> input;
    std::vector<Output> output;
};

template <Method M>
void validate_bulk_method(const Workload& work) {
    for (std::size_t i = 0; i < work.input.size(); ++i) {
        const Input& input = work.input[i];
        require(pa_l0_native::valid_input(input), "invalid generated carry domain");
        const auto reference = pa_l0_native::compute<Method::Original>(input);
        const auto candidate = pa_l0_native::compute<M>(input);
        if (!same_output(reference, candidate))
            throw std::runtime_error(std::string("pre-timing mismatch for ") +
                                     method_name(M) + " at input " + std::to_string(i));
    }
}

void prepare_bulk(Workload& work, Word seed) {
    require(work.count <= kMaxWorkingBytes / (sizeof(Input) + sizeof(Output)),
            "working allocation exceeds 256 MiB");
    work.input.resize(static_cast<std::size_t>(work.count));
    work.output.resize(static_cast<std::size_t>(work.count));
    for (auto& item : work.input) item = make_input(seed);
    validate_bulk_method<Method::Original>(work);
    validate_bulk_method<Method::GpMaterialized>(work);
    validate_bulk_method<Method::GpRecomputed>(work);
    validate_bulk_method<Method::Blocked2>(work);
    work.validation_outputs = work.count * kMethods.size();
    bulk_pass<Method::Original>(work.input.data(), work.output.data(), work.input.size());
    work.expected_checksum = output_checksum(work.output);
}

template <Method M>
void validate_latency_prefix(Word seed, Word count) {
    Input reference = make_input(seed);
    Input candidate = reference;
    for (Word i = 0; i < count; ++i) {
        const auto original_output = pa_l0_native::compute<Method::Original>(reference);
        const auto candidate_output = pa_l0_native::compute<M>(candidate);
        require(same_output(original_output, candidate_output),
                std::string("latency prefix mismatch for ") + method_name(M));
        dependent_steps<Method::Original>(reference, 1);
        dependent_steps<M>(candidate, 1);
        require(reference.a == candidate.a && reference.b == candidate.b &&
                    reference.carry_in == candidate.carry_in,
                "latency recurrence state mismatch");
    }
}

struct Sample {
    Method method;
    unsigned round;
    unsigned position;
    Word count;
    Word passes;
    RegionClock clock;
    Word checksum;
    int cpu_before;
    int cpu_after;
};

template <Method M>
Sample run_method(Workload& work, const Options& options, unsigned round, unsigned position) {
    require(work.count && work.passes && work.passes <= kMaxPasses &&
                work.count <= kMaxOperations / work.passes,
            "operation/pass resource limit");
    RegionClock timing{};
    Word checksum;
    const int before = sched_getcpu();
    require(before == static_cast<int>(options.cpu), "CPU changed before sample");
    if (work.regime == Regime::Latency) {
        Word seed = options.seed;
        Input state = make_input(seed);
        dependent_steps<M>(state, std::min<Word>(work.count, 4096));
        timing = timed_latency<M>(state, work.count, work.passes);
        checksum = state_checksum(state);
    } else {
        bulk_pass<M>(work.input.data(), work.output.data(), work.input.size());
        timing = timed_bulk<M>(work.input.data(), work.output.data(), work.input.size(), work.passes);
        checksum = output_checksum(work.output);
    }
    const int after = sched_getcpu();
    require(after == static_cast<int>(options.cpu), "CPU changed after sample");
    return {M, round, position, work.count, work.passes, timing, checksum, before, after};
}

Sample dispatch(Method method, Workload& work, const Options& options,
                unsigned round, unsigned position) {
    switch (method) {
        case Method::Original:
            return run_method<Method::Original>(work, options, round, position);
        case Method::GpMaterialized:
            return run_method<Method::GpMaterialized>(work, options, round, position);
        case Method::GpRecomputed:
            return run_method<Method::GpRecomputed>(work, options, round, position);
        case Method::Blocked2:
            return run_method<Method::Blocked2>(work, options, round, position);
    }
    throw std::logic_error("unknown method dispatch");
}

std::vector<Sample> calibrate(Workload& work, const Options& options) {
    std::vector<Sample> samples;
    if (options.smoke) {
        work.passes = 2;
        return samples;
    }
    const Word target = options.target_ms * 1000000;
    const Word cap = std::min<Word>(kMaxPasses, kMaxOperations / work.count);
    for (unsigned trial = 0; trial < 8; ++trial) {
        Sample sample = dispatch(Method::Original, work, options, trial, 0);
        if (work.regime != Regime::Latency)
            require(sample.checksum == work.expected_checksum, "calibration checksum mismatch");
        samples.push_back(sample);
        if (sample.clock.elapsed_ns >= target) return samples;
        const long double proposed = std::ceil(static_cast<long double>(work.passes) *
                                               target * 1.2L / sample.clock.elapsed_ns);
        const Word next = proposed >= cap ? cap : static_cast<Word>(proposed);
        require(next > work.passes, "calibration target impossible within operation/pass cap");
        work.passes = next;
    }
    throw std::runtime_error("calibration did not reach target in eight trials");
}

std::array<Method, 4> initial_order(Word seed, Regime regime) {
    seed ^= UINT64_C(0xd1b54a32d192ed03) * (1 + static_cast<Word>(regime));
    auto order = kMethods;
    for (std::size_t i = order.size() - 1; i > 0; --i)
        std::swap(order[i], order[splitmix64(seed) % (i + 1)]);
    return order;
}

struct Collected {
    Regime regime;
    Word count;
    Word passes;
    Word validation_outputs;
    Word reference_checksum;
    std::array<Method, 4> order;
    std::vector<Sample> calibration;
    std::vector<Sample> warmups;
    std::vector<Sample> samples;
};

Collected collect(Regime regime, const Options& options) {
    const Word count = options.smoke ? (regime == Regime::Latency ? 16 :
                                        regime == Regime::Hot ? 7 : 257)
                                    : (regime == Regime::Latency ? 4096 :
                                        regime == Regime::Hot ? 219 : 748983);
    Workload work{regime, count, 1, 0, 0, {}, {}};
    if (regime == Regime::Latency) {
        const Word prefix = options.smoke ? 64 : 4096;
        validate_latency_prefix<Method::Original>(options.seed, prefix);
        validate_latency_prefix<Method::GpMaterialized>(options.seed, prefix);
        validate_latency_prefix<Method::GpRecomputed>(options.seed, prefix);
        validate_latency_prefix<Method::Blocked2>(options.seed, prefix);
        work.validation_outputs = prefix * kMethods.size();
    } else {
        prepare_bulk(work, options.seed);
    }
    auto calibration = calibrate(work, options);
    if (regime == Regime::Latency) {
        Word seed = options.seed;
        Input reference = make_input(seed);
        dependent_steps<Method::Original>(reference, std::min<Word>(work.count, 4096));
        // Full untimed original replay, not an independent mathematical oracle.
        dependent_steps<Method::Original>(reference, work.count * work.passes);
        work.expected_checksum = state_checksum(reference);
        if (!calibration.empty())
            require(calibration.back().checksum == work.expected_checksum,
                    "timed/reference latency boundary mismatch");
    }
    Collected result{regime, count, work.passes, work.validation_outputs,
                     work.expected_checksum, initial_order(options.seed, regime),
                     std::move(calibration), {}, {}};
    const unsigned warmup_rounds = options.smoke ? 1 : 2;
    const unsigned sample_rounds = options.smoke ? 4 : 7;
    for (unsigned phase = 0; phase < 2; ++phase) {
        const unsigned rounds = phase == 0 ? warmup_rounds : sample_rounds;
        for (unsigned round = 0; round < rounds; ++round) {
            for (unsigned position = 0; position < result.order.size(); ++position) {
                const Method method = result.order[(position + round) % result.order.size()];
                Sample sample = dispatch(method, work, options, round, position);
                require(sample.checksum == work.expected_checksum,
                        std::string("sample checksum mismatch for ") + method_name(method));
                (phase == 0 ? result.warmups : result.samples).push_back(sample);
            }
        }
    }
    return result;
}

double quantile(const std::vector<double>& sorted, double fraction) {
    require(!sorted.empty(), "empty statistics sample");
    const double location = (sorted.size() - 1) * fraction;
    const auto low = static_cast<std::size_t>(location);
    const auto high = std::min(low + 1, sorted.size() - 1);
    return sorted[low] + (sorted[high] - sorted[low]) * (location - low);
}

struct Statistics {
    double median;
    double minimum;
    double maximum;
    double mad;
    double inclusive_iqr;
};

Statistics summarize(std::vector<double> values) {
    require(!values.empty(), "empty summary input");
    for (double value : values) require(std::isfinite(value), "nonfinite summary input");
    std::sort(values.begin(), values.end());
    const double median = quantile(values, 0.5);
    std::vector<double> deviations;
    for (double value : values) deviations.push_back(std::abs(value - median));
    std::sort(deviations.begin(), deviations.end());
    return {median, values.front(), values.back(), quantile(deviations, 0.5),
            quantile(values, 0.75) - quantile(values, 0.25)};
}

void selftest() {
    const auto odd = summarize({7, 1, 4, 3, 6, 2, 5});
    require(odd.median == 4 && odd.minimum == 1 && odd.maximum == 7 &&
                odd.mad == 2 && odd.inclusive_iqr == 3, "odd statistics fixture failed");
    const auto even = summarize({4, 1, 3, 2});
    require(even.median == 2.5 && even.mad == 1 && even.inclusive_iqr == 1.5,
            "even statistics fixture failed");
    const auto one = summarize({3});
    require(one.median == 3 && one.mad == 0 && one.inclusive_iqr == 0,
            "singleton statistics fixture failed");
    require(decimal("0") == 0 && decimal("18446744073709551615") == UINT64_MAX,
            "decimal valid fixtures failed");
    for (const char* invalid : {"", "-1", "+1", "1x", " 1", "18446744073709551616"}) {
        bool rejected = false;
        try { static_cast<void>(decimal(invalid)); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "decimal invalid fixture was accepted");
    }
}

std::string quote(const std::string& value) {
    std::ostringstream result;
    result << '"';
    for (unsigned char ch : value) {
        if (ch == '"' || ch == '\\') result << '\\' << ch;
        else if (ch < 32) result << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                                 << static_cast<unsigned>(ch) << std::dec;
        else result << ch;
    }
    result << '"';
    return result.str();
}

std::string hex(Word value) {
    std::ostringstream result;
    result << std::hex << std::setfill('0') << std::setw(16) << value;
    return result.str();
}

double ns_per_operation(const Sample& sample) {
    return static_cast<double>(sample.clock.elapsed_ns) / (sample.count * sample.passes);
}

void print_statistics(const Statistics& stats) {
    std::cout << "{\"median\":" << stats.median << ",\"min\":" << stats.minimum
              << ",\"max\":" << stats.maximum << ",\"mad\":" << stats.mad
              << ",\"inclusive_iqr\":" << stats.inclusive_iqr << '}';
}

void print_sample(const Sample& sample, Word target_ns, bool comma) {
    if (comma) std::cout << ',';
    std::cout << "{\"method\":" << quote(method_name(sample.method))
              << ",\"round\":" << sample.round << ",\"position\":" << sample.position
              << ",\"count\":" << sample.count << ",\"passes\":" << sample.passes
              << ",\"operations\":" << sample.count * sample.passes
              << ",\"start_steady_ns\":" << sample.clock.start_ns
              << ",\"stop_steady_ns\":" << sample.clock.stop_ns
              << ",\"elapsed_ns\":" << sample.clock.elapsed_ns
              << ",\"ns_per_addition\":" << ns_per_operation(sample)
              << ",\"checksum\":" << quote(hex(sample.checksum))
              << ",\"cpu_before\":" << sample.cpu_before
              << ",\"cpu_after\":" << sample.cpu_after
              << ",\"below_requested_target\":";
    if (!target_ns) std::cout << "null";
    else std::cout << (sample.clock.elapsed_ns < target_ns ? "true" : "false");
    std::cout << '}';
}

void print_samples(const std::vector<Sample>& samples, Word target_ns) {
    std::cout << '[';
    for (std::size_t i = 0; i < samples.size(); ++i) print_sample(samples[i], target_ns, i != 0);
    std::cout << ']';
}

void print_collected(const Collected& result, const Options& options) {
    const bool bulk = result.regime != Regime::Latency;
    const Word target_ns = options.smoke ? 0 : options.target_ms * 1000000;
    std::cout << "{\"regime\":" << quote(regime_name(result.regime))
              << ",\"count\":" << result.count << ",\"passes\":" << result.passes
              << ",\"operations_per_sample\":" << result.count * result.passes
              << ",\"logical_working_set_bytes\":"
              << (bulk ? result.count * (sizeof(Input) + sizeof(Output)) : sizeof(Input))
              << ",\"logical_streamed_bytes_per_addition\":"
              << (bulk ? sizeof(Input) + sizeof(Output) : 0)
              << ",\"validation_full_outputs_vs_original\":" << result.validation_outputs
              << ",\"validation_scope\":"
              << quote(bulk ? "every generated bulk input, every method, before timing"
                            : "4096-step prefix (64 smoke), every method; full original checksum replay")
              << ",\"reference_checksum\":" << quote(hex(result.reference_checksum))
              << ",\"initial_method_order\":[";
    for (std::size_t i = 0; i < result.order.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << quote(method_name(result.order[i]));
    }
    std::cout << "],\"calibration\":";
    print_samples(result.calibration, target_ns);
    std::cout << ",\"warmups\":";
    print_samples(result.warmups, target_ns);
    std::cout << ",\"samples\":";
    print_samples(result.samples, target_ns);
    std::cout << ",\"summaries\":[";
    for (std::size_t method_index = 0; method_index < kMethods.size(); ++method_index) {
        const Method method = kMethods[method_index];
        std::vector<double> elapsed, paired;
        unsigned below = 0;
        for (const Sample& sample : result.samples) {
            if (sample.method != method) continue;
            elapsed.push_back(ns_per_operation(sample));
            const auto original = std::find_if(result.samples.begin(), result.samples.end(),
                [&](const Sample& other) { return other.round == sample.round &&
                                                 other.method == Method::Original; });
            require(original != result.samples.end(), "paired original control missing");
            paired.push_back(ns_per_operation(*original) / ns_per_operation(sample));
            if (target_ns && sample.clock.elapsed_ns < target_ns) ++below;
        }
        if (method_index) std::cout << ',';
        std::cout << "{\"method\":" << quote(method_name(method))
                  << ",\"sample_count\":" << elapsed.size() << ",\"below_target_count\":" << below
                  << ",\"ns_per_addition\":";
        print_statistics(summarize(elapsed));
        std::cout << ",\"paired_original_over_method\":";
        print_statistics(summarize(paired));
        std::cout << ",\"paired_ratio_by_round\":[";
        for (std::size_t i = 0; i < paired.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << paired[i];
        }
        std::cout << "],\"logical_GB_per_s_at_median\":";
        if (bulk) std::cout << (sizeof(Input) + sizeof(Output)) / summarize(elapsed).median;
        else std::cout << "null";
        std::cout << '}';
    }
    std::cout << "]}";
}

Word unix_now_ns() {
    return static_cast<Word>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::system_clock::now().time_since_epoch()).count());
}

void print_report(const std::vector<Collected>& results, const Options& options,
                  const std::vector<unsigned>& allowed, Word started, Word finished) {
    std::cout << std::setprecision(17);
    std::cout << "{\"protocol\":\"parseatlas_l0_native_compare_v1\",\"mode\":"
              << quote(options.smoke ? "smoke_not_performance_evidence" : "measurement")
              << ",\"started_unix_ns\":" << started << ",\"finished_unix_ns\":" << finished
              << ",\"compiler_version\":" << quote(__VERSION__)
              << ",\"cplusplus\":" << __cplusplus << ",\"seed\":" << options.seed
              << ",\"cpu\":" << options.cpu << ",\"inherited_allowed_cpus\":[";
    for (std::size_t i = 0; i < allowed.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << allowed[i];
    }
#ifdef SECP256K1_NO_INT128
    constexpr bool no_int128 = true;
#else
    constexpr bool no_int128 = false;
#endif
#ifdef __SIZEOF_INT128__
    constexpr bool int128_macro = true;
#else
    constexpr bool int128_macro = false;
#endif
    std::cout << "],\"original_backend\":"
              << quote(no_int128 || !int128_macro ? "portable_carry" : "gcc_uint128")
              << ",\"no_int128_defined\":" << (no_int128 ? "true" : "false")
              << ",\"int128_macro_defined\":" << (int128_macro ? "true" : "false")
              << ",\"input_record_bytes\":" << sizeof(Input)
              << ",\"output_record_bytes\":" << sizeof(Output)
              << ",\"requested_target_ns\":" << (options.smoke ? 0 : options.target_ms * 1000000)
              << ",\"warmup_rounds\":" << (options.smoke ? 1 : 2)
              << ",\"sample_rounds\":" << (options.smoke ? 4 : 7)
              << ",\"measurement_workers\":1,\"statistics_selftest\":\"passed\","
                 "\"conditions\":["
                 "\"Complete four-limb add with carry, not field/scalar modular arithmetic\","
                 "\"All orchestration, arithmetic validation, timer and statistics are native C++\","
                 "\"Independent Boost oracle belongs to the separate native correctness executable\","
                 "\"Original calibrates the common passes; faster candidates can fall below target and are retained\","
                 "\"Seeded rotating order, all four positions covered once per four rounds; seven rounds not exactly balanced\","
                 "\"Same original control in every round; paired ratios are descriptive, not confidence intervals\","
                 "\"Fixed regime order: latency, hot, large; methods interleaved within each regime\","
                 "\"One serial pinned measurement worker prevents benchmark self-interference\","
                 "\"Affinity only changes this process; no governor, turbo, sysctl, sudo or other process changes\","
                 "\"External load, sibling activity, thermal state and frequency are not controlled\","
                 "\"Input construction, allocation, internal warmup and full output checksum are outside timer\","
                 "\"Latency includes rotate/XOR conditioning, loop control and any compiler spills\","
                 "\"Logical record bytes are not cache traffic or measured DRAM bandwidth\","
                 "\"Source-level GP variants may optimize to identical code; inspect actual measured binary\","
                 "\"No outlier trimming, automatic winner, novelty claim or constant-time proof\","
                 "\"PMU counters not measured by this executable; no Python execution\"],\"workloads\":[";
    for (std::size_t i = 0; i < results.size(); ++i) {
        if (i) std::cout << ',';
        print_collected(results[i], options);
    }
    std::cout << "]}\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        selftest();
        const auto allowed = pin_cpu(options.cpu);
        const Word started = unix_now_ns();
        std::vector<Collected> results;
        for (Regime regime : {Regime::Latency, Regime::Hot, Regime::Stream})
            results.push_back(collect(regime, options));
        const Word finished = unix_now_ns();
        print_report(results, options, allowed, started, finished);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "l0_native_compare: " << error.what() << '\n';
        return 2;
    }
}
