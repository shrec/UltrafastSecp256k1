// P4 final-only field sum: native 4x64 controls; research only, no LTO.
// Task MCP is owner-suspended; worker Source Graph tools are unavailable.
// Manager-verified exact inputs supplied this bounded new-file implementation.
#include "p3_field_sum_kernels.hpp"
#include "p4_field_sum_kernels.hpp"
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include <sched.h>
#include <unistd.h>

#if defined(__GNUC__) && !defined(__clang__)
#define PA_BOUNDARY __attribute__((noinline, noipa))
#elif defined(__clang__)
#define PA_BOUNDARY __attribute__((noinline))
#else
#error "P4 requires GCC or Clang compiler memory barriers"
#endif
#define PA_STRING_IMPL(x) #x
#define PA_STRING(x) PA_STRING_IMPL(x)

namespace {
using FE = pa_p3::FE;
using Bytes = std::array<std::uint8_t, 32>;
using Clock = std::chrono::steady_clock;
constexpr std::array<std::size_t, 6> sizes{1, 16, 256, 4096, 65536, 1048576};
constexpr std::array<const char*, 8> routes{
    "fe64_api_serial", "fe64_inline_eager", "fe64_wide16_lane1", "fe64_wide4094_lane1",
    "fe64_wide16_lane4", "fe64_wide4094_lane4", "fe52_e2e_full16", "fe52_e2e_full4094"};
constexpr std::array<unsigned, 8> chunks{1, 1, 16, 4094, 16, 4094, 16, 4094};
constexpr std::array<unsigned, 8> banks{1, 1, 1, 1, 4, 4, 0, 0};
constexpr unsigned warmups = 2, max_attempts = 12, max_extensions = 12;
constexpr std::uint64_t operation_cap = 1000000000;
constexpr std::uint64_t fnv_basis = UINT64_C(14695981039346656037);
constexpr std::uint64_t corpus_salt = UINT64_C(0x6669656c645f7033);
inline void barrier() { asm volatile("" ::: "memory"); }

struct RNG {
    std::uint64_t state;
    std::uint64_t next() {
        std::uint64_t z = (state += UINT64_C(0x9e3779b97f4a7c15));
        z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
        return z ^ (z >> 31);
    }
};
std::string quote(const std::string& text) {
    std::ostringstream out; out << '"';
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') out << '\\' << static_cast<char>(c);
        else if (c < 32) out << "\\u" << std::hex << std::setfill('0')
                            << std::setw(4) << static_cast<unsigned>(c);
        else out << static_cast<char>(c);
    }
    out << '"'; return out.str();
}
std::string hex(const Bytes& bytes) {
    std::ostringstream out; out << std::hex << std::setfill('0');
    for (auto byte : bytes) out << std::setw(2) << unsigned(byte);
    return out.str();
}
std::string hex64(std::uint64_t value) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << value;
    return out.str();
}
void hash_byte(std::uint64_t& hash, std::uint8_t byte) {
    hash ^= byte; hash *= UINT64_C(1099511628211);
}
void hash_bytes(std::uint64_t& hash, const Bytes& bytes) {
    for (auto byte : bytes) hash_byte(hash, byte);
}
std::uint64_t checksum(const Bytes& bytes) {
    auto hash = fnv_basis; hash_bytes(hash, bytes); return hash;
}
bool canonical64(const FE& value) {
    constexpr FE::limbs_type p{UINT64_C(0xfffffffefffffc2f), UINT64_MAX, UINT64_MAX, UINT64_MAX};
    for (int i = 3; i >= 0; --i)
        if (value.limbs()[static_cast<unsigned>(i)] != p[static_cast<unsigned>(i)])
            return value.limbs()[static_cast<unsigned>(i)] < p[static_cast<unsigned>(i)];
    return false;
}
FE random_canonical(RNG& rng) {
    for (;;) {
        Bytes bytes{};
        for (unsigned w = 0; w < 4; ++w) {
            const auto word = rng.next();
            for (unsigned b = 0; b < 8; ++b)
                bytes[w * 8 + b] = static_cast<std::uint8_t>(word >> (56 - 8 * b));
        }
        FE result;
        if (FE::parse_bytes_strict(bytes, result)) return result; // Zero is valid for sums.
    }
}
struct Corpus {
    FE x0;
    std::vector<FE> rhs;
    std::array<std::uint64_t, 6> prefix_hash{};
    std::uint64_t hash64 = fnv_basis;
    bool ready = false;
};
void make_corpus(Corpus& corpus, std::uint64_t seed) {
    RNG rng{seed ^ corpus_salt}; // Preserve P3 FE64 sequence, including x0 first.
    corpus.x0 = random_canonical(rng);
    if (!canonical64(corpus.x0)) throw std::runtime_error("noncanonical x0");
    hash_bytes(corpus.hash64, corpus.x0.to_bytes());
    corpus.rhs.resize(sizes.back());
    unsigned prefix = 0;
    for (std::size_t i = 0; i < corpus.rhs.size(); ++i) {
        corpus.rhs[i] = random_canonical(rng);
        if (!canonical64(corpus.rhs[i])) throw std::runtime_error("noncanonical RHS");
        hash_bytes(corpus.hash64, corpus.rhs[i].to_bytes());
        if (i + 1 == sizes[prefix]) corpus.prefix_hash[prefix++] = corpus.hash64;
    }
    corpus.ready = true;
}
struct Preservation {
    std::string phase;
    bool verified = false;
    std::size_t exact_rhs_checked = 0;
    std::uint64_t hash64 = fnv_basis;
};
Preservation check_inputs(const Corpus& corpus, std::uint64_t seed, const std::string& phase) {
    Preservation check; check.phase = phase;
    RNG rng{seed ^ corpus_salt};
    const auto x0 = random_canonical(rng);
    check.verified = canonical64(corpus.x0) && x0.limbs() == corpus.x0.limbs();
    hash_bytes(check.hash64, corpus.x0.to_bytes());
    for (std::size_t i = 0; i < corpus.rhs.size(); ++i) {
        const auto expected = random_canonical(rng);
        check.verified = canonical64(corpus.rhs[i]) &&
            expected.limbs() == corpus.rhs[i].limbs() && check.verified;
        hash_bytes(check.hash64, corpus.rhs[i].to_bytes());
        ++check.exact_rhs_checked;
    }
    check.verified = check.verified && check.hash64 == corpus.hash64;
    return check;
}

template<unsigned Route> inline FE evaluate(const Corpus& corpus, std::size_t count) {
    if constexpr (Route == 0) return pa_p3::sum_fe64(corpus.x0, corpus.rhs.data(), count);
    else if constexpr (Route == 1) return pa_p4::sum_fe64_inline(corpus.x0, corpus.rhs.data(), count);
    else if constexpr (Route == 2) return pa_p4::sum_fe64_wide<16, 1>(corpus.x0, corpus.rhs.data(), count);
    else if constexpr (Route == 3) return pa_p4::sum_fe64_wide<4094, 1>(corpus.x0, corpus.rhs.data(), count);
    else if constexpr (Route == 4) return pa_p4::sum_fe64_wide<16, 4>(corpus.x0, corpus.rhs.data(), count);
    else if constexpr (Route == 5) return pa_p4::sum_fe64_wide<4094, 4>(corpus.x0, corpus.rhs.data(), count);
    else if constexpr (Route == 6) return pa_p3::sum_fe52_e2e<16, false>(corpus.x0, corpus.rhs.data(), count);
    else return pa_p3::sum_fe52_e2e<4094, false>(corpus.x0, corpus.rhs.data(), count);
}
template<unsigned Route> PA_BOUNDARY void full_job(const Corpus& corpus, std::size_t count, Bytes& output) {
    // One core deliberately isolates carry dependencies / instruction scheduling;
    // worker threads would change this experiment's contract and measured costs.
    const FE result = evaluate<Route>(corpus, count);
    output = result.to_bytes();
    // Whole-job side effect prevents hoisting repeated identical jobs, including
    // under Clang without noipa. No barrier is inserted per RHS operand.
    asm volatile("" : : "m"(output) : "memory");
}
using Job = void (*)(const Corpus&, std::size_t, Bytes&);
using Evaluator = FE (*)(const Corpus&, std::size_t);
constexpr std::array<Job, 8> jobs{full_job<0>, full_job<1>, full_job<2>, full_job<3>, full_job<4>, full_job<5>, full_job<6>, full_job<7>};
constexpr std::array<Evaluator, 8> evaluators{evaluate<0>, evaluate<1>, evaluate<2>, evaluate<3>, evaluate<4>, evaluate<5>, evaluate<6>, evaluate<7>};
struct Cell {
    unsigned id, group, route;
    std::size_t count;
    Bytes expected{};
    bool validated = false, qualified = false;
    std::uint64_t jobs = 0;
};
void validate(std::vector<Cell>& cells, const Corpus& corpus) {
    for (unsigned group = 0; group < sizes.size(); ++group) {
        // The actual unchanged corrected FE64 API is the differential reference.
        // A separate Boost C++ test is required; this replay is not that oracle.
        const FE reference = pa_p3::sum_fe64(corpus.x0, corpus.rhs.data(), sizes[group]);
        if (!canonical64(reference)) throw std::runtime_error("noncanonical reference sum");
        const auto expected = reference.to_bytes();
        for (unsigned route = 0; route < routes.size(); ++route) {
            auto& cell = cells[group * routes.size() + route];
            cell.expected = expected;
            const FE result = evaluators[route](corpus, cell.count);
            Bytes output{}; jobs[route](corpus, cell.count, output);
            cell.validated = canonical64(result) && result.limbs() == reference.limbs() &&
                result.to_bytes() == expected && output == expected;
            if (!cell.validated) throw std::runtime_error("untimed cell mismatch: " + std::to_string(cell.id));
        }
    }
}

struct Options {
    std::uint64_t seed = 20260905;
    double min_ms = 200;
    unsigned rounds = 8;
    bool smoke = false, reverse = false, help = false;
    std::string output;
};
std::uint64_t unsigned_arg(const std::string& value, const std::string& flag) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error(flag + " requires an unsigned decimal integer");
    std::size_t used = 0; const auto result = std::stoull(value, &used);
    if (used != value.size()) throw std::runtime_error("invalid integer suffix");
    return result;
}
Options parse_options(int argc, char** argv) {
    Options options; bool explicit_ms = false, explicit_rounds = false;
    std::vector<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (std::find(seen.begin(), seen.end(), flag) != seen.end()) throw std::runtime_error("duplicate option: " + flag);
        seen.push_back(flag);
        if (flag == "--help") { options.help = true; continue; }
        if (flag == "--smoke") { options.smoke = true; continue; }
        if (flag == "--reverse-order") { options.reverse = true; continue; }
        if (flag != "--output" && flag != "--seed" && flag != "--min-ms" && flag != "--rounds")
            throw std::runtime_error("unknown option: " + flag);
        if (++i == argc) throw std::runtime_error("missing value: " + flag);
        const std::string value = argv[i];
        if (flag == "--output") options.output = value;
        else if (flag == "--seed") options.seed = unsigned_arg(value, flag);
        else if (flag == "--rounds") {
            const auto rounds = unsigned_arg(value, flag);
            if (rounds < 2 || rounds > 128 || rounds % 2) throw std::runtime_error("--rounds must be even in [2,128]");
            options.rounds = static_cast<unsigned>(rounds); explicit_rounds = true;
        } else {
            std::size_t used = 0; options.min_ms = std::stod(value, &used);
            if (used != value.size() || !std::isfinite(options.min_ms) || options.min_ms < 0.1 || options.min_ms > 1000)
                throw std::runtime_error("--min-ms must be finite in [0.1,1000]");
            explicit_ms = true;
        }
    }
    if (options.smoke) {
        if (!explicit_ms) options.min_ms = 1;
        if (!explicit_rounds) options.rounds = 2;
    }
    if (!options.help && options.output.empty()) throw std::runtime_error("--output is required");
    return options;
}
struct Power { int cpu; std::string governor, khz, no_turbo; };
std::string sysfs(const std::string& path) {
    std::ifstream input(path); std::string value;
    return input >> value ? value : "unavailable";
}
Power power() {
    const int cpu = sched_getcpu();
    const auto path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cpufreq/";
    return {cpu, sysfs(path + "scaling_governor"), sysfs(path + "scaling_cur_freq"),
            sysfs("/sys/devices/system/cpu/intel_pstate/no_turbo")};
}
std::uint64_t steady_ns(Clock::time_point time) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(time.time_since_epoch()).count());
}
struct Probe {
    std::uint64_t cumulative_jobs = 0, cumulative_operations = 0;
    std::uint64_t end_steady_ns = 0, cumulative_elapsed_ns = 0;
};
enum class Stop { calibration_sample, floor_reached, work_cap, extension_limit, invalid_clock };
const char* stop_name(Stop stop) {
    switch (stop) {
    case Stop::calibration_sample: return "calibration_sample";
    case Stop::floor_reached: return "floor_reached";
    case Stop::work_cap: return "work_cap";
    case Stop::extension_limit: return "extension_limit";
    case Stop::invalid_clock: return "invalid_clock";
    }
    return "unknown";
}
struct Record {
    std::size_t region;
    std::string phase;
    unsigned cell;
    int round;
    unsigned position, attempt;
    std::uint64_t jobs, operations, begin_ns, end_ns, elapsed_ns;
    double ns_per_sum, ns_per_rhs;
    Bytes output;
    bool verified, floor_met;
    int cpu_start, cpu_end;
    Power power_start, power_end;
    unsigned streak = 0;
    std::uint64_t initial_batch_jobs = 0;
    bool duration_qualified = false;
    unsigned extensions = 0, clock_checks = 0;
    Stop stop = Stop::calibration_sample;
    std::array<Probe, max_extensions + 1> probes{};
};
Record measure(const Cell& cell, const Corpus& corpus, const Options& options, const std::string& phase,
               int round, unsigned position, unsigned attempt, std::uint64_t initial_jobs, std::size_t region) {
    if (!cell.count || !initial_jobs || initial_jobs > operation_cap / cell.count)
        throw std::runtime_error("operation cap exceeded before region");
    const auto max_jobs = operation_cap / cell.count;
    const auto floor_ns = static_cast<std::uint64_t>(std::ceil(options.min_ms * 1e6));
    const bool extend = phase != "calibration";
    Bytes output{};
    // Fixed storage: no allocation or JSON formatting occurs inside the clock.
    // Storing/checking intermediate probes is part of this continuous region.
    std::array<Clock::time_point, max_extensions + 1> probe_times{};
    std::array<std::uint64_t, max_extensions + 1> probe_jobs{};
    unsigned probe_count = 0;
    std::uint64_t total_jobs = 0, batch_jobs = initial_jobs;
    Stop stop = Stop::calibration_sample;
    const auto start_power = power();
    const auto cpu_start = sched_getcpu();
    const Job job = jobs[cell.route];
    barrier(); const auto begin = Clock::now();
    for (;;) {
        // batch_jobs <= max_jobs-total_jobs was checked before this batch.
        for (std::uint64_t i = 0; i < batch_jobs; ++i) job(corpus, cell.count, output);
        total_jobs += batch_jobs;
        const auto probe = Clock::now();
        probe_times[probe_count] = probe;
        probe_jobs[probe_count] = total_jobs;
        ++probe_count;
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(probe - begin).count();
        if (elapsed <= 0) { stop = Stop::invalid_clock; break; }
        if (static_cast<std::uint64_t>(elapsed) >= floor_ns) { stop = Stop::floor_reached; break; }
        if (!extend) { stop = Stop::calibration_sample; break; }
        if (total_jobs == max_jobs) { stop = Stop::work_cap; break; }
        if (probe_count == max_extensions + 1) { stop = Stop::extension_limit; break; }
        // Append, never restart. The final batch may be capped; no multiplication
        // or cumulative addition can exceed the public work bound.
        batch_jobs = std::min(initial_jobs, max_jobs - total_jobs);
    }
    const auto end = probe_times[probe_count - 1]; barrier();
    const auto cpu_end = sched_getcpu(); const auto end_power = power();
    const auto signed_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    const auto elapsed = signed_elapsed > 0 ? static_cast<std::uint64_t>(signed_elapsed) : 0;
    const auto operations = total_jobs * cell.count; // total_jobs <= cap/N.
    const bool qualified = elapsed >= floor_ns && signed_elapsed > 0;
    Record record{region, phase, cell.id, round, position, attempt, total_jobs, operations,
        steady_ns(begin), steady_ns(end), elapsed, static_cast<double>(elapsed) / static_cast<double>(total_jobs),
        static_cast<double>(elapsed) / static_cast<double>(operations), output,
        output == cell.expected && signed_elapsed > 0, qualified,
        cpu_start, cpu_end, start_power, end_power};
    record.initial_batch_jobs = initial_jobs; record.duration_qualified = qualified;
    record.extensions = probe_count - 1; record.clock_checks = probe_count; record.stop = stop;
    // Convert stored cumulative probes and construct the record outside timing.
    for (unsigned i = 0; i < probe_count; ++i) {
        const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(probe_times[i] - begin).count();
        record.probes[i] = {probe_jobs[i], probe_jobs[i] * cell.count, steady_ns(probe_times[i]),
            duration > 0 ? static_cast<std::uint64_t>(duration) : 0};
    }
    return record;
}
bool calibrate(Cell& cell, const Corpus& corpus, const Options& options, unsigned position, std::vector<Record>& records) {
    std::uint64_t count_jobs = 1;
    const auto max_jobs = operation_cap / cell.count;
    unsigned streak = 0;
    for (unsigned attempt = 0; attempt < max_attempts; ++attempt) {
        auto record = measure(cell, corpus, options, "calibration", -1, position, attempt, count_jobs, records.size());
        streak = record.floor_met ? streak + 1 : 0;
        record.streak = streak; records.push_back(record);
        if (!record.verified) throw std::runtime_error("calibration output/clock mismatch");
        if (streak == 2) { cell.jobs = count_jobs; cell.qualified = true; return true; }
        if (!record.floor_met) {
            if (count_jobs == max_jobs) return false;
            const double proposed = std::ceil(1.5 * options.min_ms * 1e6 * static_cast<double>(count_jobs) /
                static_cast<double>(std::max<std::uint64_t>(1, record.elapsed_ns)));
            count_jobs = proposed >= static_cast<double>(max_jobs) ? max_jobs :
                std::max(count_jobs + 1, static_cast<std::uint64_t>(proposed));
        }
    }
    return false;
}
struct Order { std::array<unsigned, 6> groups{0, 1, 2, 3, 4, 5}; std::array<unsigned, 8> routes{0, 1, 2, 3, 4, 5, 6, 7}; };
Order base_order(std::uint64_t seed) {
    Order order; RNG rng{seed ^ UINT64_C(0x6f726465725f7034)};
    for (unsigned n = order.groups.size(); n > 1; --n) std::swap(order.groups[n - 1], order.groups[rng.next() % n]);
    for (unsigned n = order.routes.size(); n > 1; --n) std::swap(order.routes[n - 1], order.routes[rng.next() % n]);
    return order;
}
std::vector<unsigned> schedule(const Order& base, unsigned round, bool reverse) {
    auto groups = base.groups;
    std::rotate(groups.begin(), groups.begin() + (round / 2) % groups.size(), groups.end());
    if (round % 2) std::reverse(groups.begin(), groups.end());
    std::vector<unsigned> order; order.reserve(48);
    for (const auto group : groups) {
        auto variants = base.routes;
        std::rotate(variants.begin(), variants.begin() + (round / 2 + group) % variants.size(), variants.end());
        if (round % 2) std::reverse(variants.begin(), variants.end());
        for (const auto route : variants) order.push_back(group * routes.size() + route);
    }
    if (reverse) std::reverse(order.begin(), order.end());
    return order;
}
double median(std::vector<double> data) {
    std::sort(data.begin(), data.end()); const auto n = data.size();
    return n % 2 ? data[n / 2] : (data[n / 2 - 1] + data[n / 2]) / 2;
}
void write_power(std::ostream& out, const Power& value) {
    out << "{\"cpu\":" << value.cpu << ",\"governor\":" << quote(value.governor)
        << ",\"current_khz\":" << quote(value.khz) << ",\"no_turbo\":" << quote(value.no_turbo) << '}';
}
void write_macros(std::ostream& out) {
    out << '{';
#ifdef SECP256K1_HAS_ASM
    out << "\"SECP256K1_HAS_ASM\":" << quote(PA_STRING(SECP256K1_HAS_ASM));
#else
    out << "\"SECP256K1_HAS_ASM\":null";
#endif
#ifdef SECP256K1_NO_ASM
    out << ",\"SECP256K1_NO_ASM\":" << quote(PA_STRING(SECP256K1_NO_ASM));
#else
    out << ",\"SECP256K1_NO_ASM\":null";
#endif
#ifdef SECP256K1_USE_FAST_REDUCTION
    out << ",\"SECP256K1_USE_FAST_REDUCTION\":" << quote(PA_STRING(SECP256K1_USE_FAST_REDUCTION));
#else
    out << ",\"SECP256K1_USE_FAST_REDUCTION\":null";
#endif
#ifdef USE_INLINE_ASSEMBLY
    out << ",\"USE_INLINE_ASSEMBLY\":" << quote(PA_STRING(USE_INLINE_ASSEMBLY));
#else
    out << ",\"USE_INLINE_ASSEMBLY\":null";
#endif
#ifdef SECP256K1_NO_INT128
    out << ",\"SECP256K1_NO_INT128\":" << quote(PA_STRING(SECP256K1_NO_INT128));
#else
    out << ",\"SECP256K1_NO_INT128\":null";
#endif
#ifdef __SIZEOF_INT128__
    out << ",\"__SIZEOF_INT128__\":" << quote(PA_STRING(__SIZEOF_INT128__));
#else
    out << ",\"__SIZEOF_INT128__\":null";
#endif
#ifdef UFSECP_FE52_FORCE_INLINE_KERNELS
    out << ",\"UFSECP_FE52_FORCE_INLINE_KERNELS\":" << quote(PA_STRING(UFSECP_FE52_FORCE_INLINE_KERNELS));
#else
    out << ",\"UFSECP_FE52_FORCE_INLINE_KERNELS\":null";
#endif
#ifdef SECP256K1_FE52_COMPUTE
    out << ",\"SECP256K1_FE52_COMPUTE\":" << quote(PA_STRING(SECP256K1_FE52_COMPUTE));
#else
    out << ",\"SECP256K1_FE52_COMPUTE\":null";
#endif
#ifdef SECP256K1_HYBRID_4X64_ACTIVE
    out << ",\"SECP256K1_HYBRID_4X64_ACTIVE\":" << quote(PA_STRING(SECP256K1_HYBRID_4X64_ACTIVE));
#else
    out << ",\"SECP256K1_HYBRID_4X64_ACTIVE\":null";
#endif
#ifdef __BMI2__
    out << ",\"__BMI2__\":" << quote(PA_STRING(__BMI2__));
#else
    out << ",\"__BMI2__\":null";
#endif
#ifdef __ADX__
    out << ",\"__ADX__\":" << quote(PA_STRING(__ADX__));
#else
    out << ",\"__ADX__\":null";
#endif
    out << '}';
}
void summary(std::ostream& out, const std::vector<double>& values) {
    if (values.empty()) { out << "null"; return; }
    out << "{\"count\":" << values.size() << ",\"min\":" << *std::min_element(values.begin(), values.end())
        << ",\"median\":" << median(values) << ",\"max\":" << *std::max_element(values.begin(), values.end()) << '}';
}
void write_comparison(std::ostream& out, unsigned group, unsigned a, unsigned b,
                      const Options& options, const std::vector<Record>& records) {
    const unsigned aid = group * routes.size() + a, bid = group * routes.size() + b;
    out << "{\"group\":" << group << ",\"rhs_count\":" << sizes[group]
        << ",\"numerator_cell\":" << aid << ",\"denominator_cell\":" << bid
        << ",\"numerator_route\":" << quote(routes[a]) << ",\"denominator_route\":" << quote(routes[b])
        << ",\"ratios_A_over_B\":[";
    std::vector<double> values; unsigned wins = 0;
    for (unsigned round = 0; round < options.rounds; ++round) {
        const Record* ra = nullptr; const Record* rb = nullptr;
        for (const auto& r : records)
            if (r.phase == "measurement" && r.round == static_cast<int>(round) && r.verified && r.duration_qualified) {
                if (r.cell == aid) ra = &r;
                if (r.cell == bid) rb = &r;
            }
        if (!ra || !rb) continue;
        if (!values.empty()) out << ',';
        const double ratio = ra->ns_per_sum / rb->ns_per_sum;
        values.push_back(ratio); if (ratio > 1) ++wins;
        out << "{\"round\":" << round << ",\"numerator_region\":" << ra->region
            << ",\"denominator_region\":" << rb->region << ",\"ratio\":" << ratio << '}';
    }
    out << "],\"wins_denominator\":" << wins << ",\"summary_ratio\":";
    summary(out, values); out << '}';
}
std::string json(const Options& options, const Corpus& corpus, const std::vector<Cell>& cells,
                 const std::vector<Record>& records, const std::vector<Preservation>& preservation,
                 const Order& base, const Power& start_power, const Power& end_power, std::int64_t wall_start_ns,
                 const std::string& status, const std::string& failure) {
    std::ostringstream out; out << std::setprecision(17) << std::boolalpha;
    out << "{\n\"schema\":\"parseatlas.p4.field_sum_compare.v1\",\n\"status\":" << quote(status)
        << ",\n\"failure\":" << quote(failure) << ",\n\"seed\":" << options.seed
        << ",\n\"min_ms\":" << options.min_ms
        << ",\n\"floor_ns\":" << static_cast<std::uint64_t>(std::ceil(options.min_ms * 1e6))
        << ",\n\"smoke\":" << options.smoke << ",\n\"reverse_order\":" << options.reverse
        << ",\n\"warmup_rounds_requested\":" << warmups
        << ",\n\"measurement_rounds_requested\":" << options.rounds
        << ",\n\"compiler\":" << quote(__VERSION__) << ",\n\"cplusplus\":" << __cplusplus
        << ",\n\"run_start_unix_ns\":" << wall_start_ns << ",\n\"harness_macros\":";
    write_macros(out);
    out << ",\n\"power_start\":"; write_power(out, start_power);
    out << ",\n\"power_end\":"; write_power(out, end_power);
    out << ",\n\"contract\":{"
        "\"result\":\"(x0 + sum(rhs[0..N))) mod p, final canonical FE64 and 32 BE bytes; N counts RHS; x0 once; no observable prefixes\","
        "\"scope\":\"field final-only vector sum; eight FE64-input routes; no production integration or CT/upper-layer gain\","
        "\"native\":\"API eager, inline eager, four u128-column deferred schedules (chunk16/4094 and banks1/4); no representation conversion\","
        "\"fe52\":\"unchanged P3 E2E Full16/Full4094; FE64 packing at use inside timed sum; no resident FE52 array\","
        "\"allocation\":\"one FE64 RHS array; allocation and generation excluded; all routes include seed handling and final bytes\","
        "\"timing\":\"whole noinline/noipa jobs include kernel, call, seed reset, RHS loads/packing, reduction, loops and serialization; no per-RHS barriers\","
        "\"verification\":\"all48cells validated raw canonical and full bytes versus actual FE64 before timing; last identical job per region checked after timing; Boost oracle is separate\","
        "\"input_preservation\":\"exact regeneration compares x0 and all raw FE64 words after untimed validation and after regions; all32bytes enter FNV hash\","
        "\"calibration\":\"one job initially; two consecutive regions >=floor at the SAME count; retain shorts; grow ceil(1.5*target*jobs/elapsed), capped; no warmup if unqualified\","
        "\"region_duration\":\"every warmup/measured region has one continuous clock; append calibrated-size job batches (last may be capped) until floor; never restart/discard a short prefix\","
        "\"probes\":\"fixed stack storage, no per-extension allocation; cumulative timestamp/job stores, clock checks and continuation work are timed; JSON and probe conversion outside clock\","
        "\"duration_failure\":\"explicit non-success status if later-phase floor cannot be reached within caps; preserve record, cumulative probes, actual jobs and last full output\","
        "\"normalization_unit\":\"elapsed/actual total jobs for ns/sum; elapsed/(actual jobs*N) for ns/RHS; not initial calibrated jobs after an extension\","
        "\"ordering\":\"seeded six-size/eight-route permutations plus round rotation/reversal; second series reverses each complete order, not corpus\","
        "\"statistics\":\"all raw attempts retained; summaries/ratios use verified duration-qualified measurement records only; any failed later record prevents complete status\","
        "\"ratio_direction\":\"numerator A / denominator B, same size and measurement round; >1 favors B; job counts can differ\","
        "\"cpu\":\"one thread deliberately measures carry/ILP scheduling; external pinning; endpoint snapshots do not prove no migration or constant clock; no policy changes\","
        "\"backend\":\"external manifest binds kernel/driver/library/binary hashes and no-LTO commands; harness macros are not proof of library configuration\","
        "\"operation_cap_per_region\":" << operation_cap << ",\"calibration_attempt_limit\":" << max_attempts
        << ",\"extensions_per_later_region_limit\":" << max_extensions
        << ",\"max_probes_per_later_region\":" << max_extensions + 1 << "},\n\"corpus\":{\"ready\":" << corpus.ready
        << ",\"recipe\":\"P3 FE64 generator unchanged: SplitMix64 state=seed xor0x6669656c645f7033; four words encoded BE32; strict parse rejects>=p, accepts zero; x0 first, then1048576 RHS; actual prefixes\""
        << ",\"hash_recipe\":\"FNV1a64 unsigned wrap; x0 BE32 followed by every RHS BE32; same x0 initializes each prefix hash\""
        << ",\"source_recipe_sha256\":\"PENDING IMPORT\""
        << ",\"x0_be32\":" << quote(hex(corpus.x0.to_bytes()))
        << ",\"rhs_count\":" << corpus.rhs.size() << ",\"rhs64_bytes\":" << corpus.rhs.size() * sizeof(FE)
        << ",\"resident_rhs52_bytes\":0,\"seed64_bytes\":" << sizeof(FE)
        << ",\"checksum_fnv1a64\":" << quote(hex64(corpus.hash64)) << ",\"prefixes\":[";
    for (unsigned g = 0; g < sizes.size(); ++g) {
        if (g) out << ',';
        out << "{\"rhs_count\":" << sizes[g] << ",\"checksum_fnv1a64\":" << quote(hex64(corpus.prefix_hash[g])) << '}';
    }
    out << "]},\n\"input_preservation\":[";
    for (unsigned i = 0; i < preservation.size(); ++i) {
        if (i) out << ',';
        const auto& p = preservation[i];
        out << "{\"phase\":" << quote(p.phase) << ",\"verified\":" << p.verified
            << ",\"exact_rhs_checked\":" << p.exact_rhs_checked << ",\"checksum_fnv1a64\":" << quote(hex64(p.hash64)) << '}';
    }
    out << "],\n\"base_group_order\":[";
    for (unsigned i = 0; i < base.groups.size(); ++i) { if (i) out << ','; out << base.groups[i]; }
    out << "],\n\"base_route_order\":[";
    for (unsigned i = 0; i < base.routes.size(); ++i) { if (i) out << ','; out << base.routes[i]; }
    out << "],\n\"cells\":[\n";
    for (const auto& cell : cells) {
        if (cell.id) out << ",\n";
        std::vector<double> sum_values, rhs_values; unsigned short_count = 0, extensions = 0;
        for (const auto& r : records) if (r.cell == cell.id && r.phase == "measurement") {
            if (!r.duration_qualified) ++short_count;
            extensions += r.extensions;
            if (r.verified && r.duration_qualified) { sum_values.push_back(r.ns_per_sum); rhs_values.push_back(r.ns_per_rhs); }
        }
        out << "{\"id\":" << cell.id << ",\"group\":" << cell.group << ",\"route_index\":" << cell.route
            << ",\"route\":" << quote(routes[cell.route]) << ",\"rhs_count\":" << cell.count
            << ",\"chunk_rhs\":" << chunks[cell.route] << ",\"explicit_native_banks\":";
        if (banks[cell.route]) out << banks[cell.route]; else out << "null";
        out << ",\"input_layout\":\"FE64\",\"validated\":" << cell.validated << ",\"qualified\":" << cell.qualified
            << ",\"initial_batch_jobs\":" << cell.jobs << ",\"expected_final_be32\":" << quote(hex(cell.expected))
            << ",\"expected_checksum_fnv1a64\":" << quote(hex64(checksum(cell.expected)))
            << ",\"short_measurement_regions\":" << short_count << ",\"measurement_extensions\":" << extensions
            << ",\"summary_ns_per_sum\":"; summary(out, sum_values);
        out << ",\"summary_ns_per_rhs\":"; summary(out, rhs_values); out << '}';
    }
    out << "\n],\n\"regions\":[\n";
    for (const auto& r : records) {
        if (r.region) out << ",\n";
        out << "{\"region\":" << r.region << ",\"phase\":" << quote(r.phase) << ",\"cell\":" << r.cell
            << ",\"round\":" << r.round << ",\"order_position\":" << r.position << ",\"attempt\":" << r.attempt
            << ",\"initial_batch_jobs\":" << r.initial_batch_jobs << ",\"jobs\":" << r.jobs << ",\"operations\":" << r.operations
            << ",\"begin_steady_ns\":" << r.begin_ns << ",\"end_steady_ns\":" << r.end_ns << ",\"elapsed_ns\":" << r.elapsed_ns
            << ",\"ns_per_sum\":" << r.ns_per_sum << ",\"ns_per_rhs\":" << r.ns_per_rhs
            << ",\"actual_final_be32\":" << quote(hex(r.output)) << ",\"checksum_fnv1a64\":" << quote(hex64(checksum(r.output)))
            << ",\"verified\":" << r.verified << ",\"floor_met\":" << r.floor_met << ",\"duration_qualified\":" << r.duration_qualified
            << ",\"extensions\":" << r.extensions << ",\"clock_checks\":" << r.clock_checks << ",\"stop_reason\":" << quote(stop_name(r.stop))
            << ",\"cpu_start\":" << r.cpu_start << ",\"cpu_end\":" << r.cpu_end << ",\"qualification_streak\":" << r.streak
            << ",\"power_start\":"; write_power(out, r.power_start);
        out << ",\"power_end\":"; write_power(out, r.power_end); out << ",\"cumulative_probes\":[";
        for (unsigned i = 0; i < r.clock_checks; ++i) {
            if (i) out << ',';
            const auto& p = r.probes[i];
            out << "{\"probe\":" << i << ",\"cumulative_jobs\":" << p.cumulative_jobs
                << ",\"cumulative_operations\":" << p.cumulative_operations
                << ",\"end_steady_ns\":" << p.end_steady_ns << ",\"cumulative_elapsed_ns\":" << p.cumulative_elapsed_ns << '}';
        }
        out << "]}";
    }
    out << "\n],\n\"primary_comparisons\":[\n";
    bool first = true;
    for (unsigned group = 0; group < sizes.size(); ++group) for (unsigned b = 1; b < routes.size(); ++b) {
        if (!first) out << ",\n";
        first = false;
        write_comparison(out, group, 0, b, options, records);
    }
    // These 13 secondary A/B pairs are fixed by P4_FIELD_SUM_PROTOCOL.md.
    constexpr std::array<std::pair<unsigned, unsigned>, 13> secondary{{
        {0,1}, {1,2}, {1,3}, {1,4}, {1,5}, {2,4}, {3,5},
        {2,3}, {4,5}, {2,6}, {3,7}, {4,6}, {5,7}}};
    out << "\n],\n\"secondary_comparisons\":[\n"; first = true;
    for (unsigned group = 0; group < sizes.size(); ++group) for (const auto& [a,b] : secondary) {
        if (!first) out << ",\n";
        first = false;
        write_comparison(out, group, a, b, options, records);
    }
    out << "\n]\n}\n"; return out.str();
}
void write_all(int fd, const std::string& data) {
    std::size_t done = 0;
    while (done < data.size()) {
        const auto n = ::write(fd, data.data() + done, data.size() - done);
        if (n < 0) { if (errno == EINTR) continue; throw std::runtime_error(std::string("write: ") + std::strerror(errno)); }
        if (!n) throw std::runtime_error("zero-length write");
        done += static_cast<std::size_t>(n);
    }
    if (::fsync(fd) != 0) throw std::runtime_error("output fsync failed");
}
} // namespace

int main(int argc, char** argv) {
    int fd = -1;
    try {
        const auto options = parse_options(argc, argv);
        if (options.help) {
            std::cout << "Usage: p4_field_sum_compare --output NEW.json [--seed UINT64] [--min-ms 0.1..1000] [--rounds EVEN_2..128] [--reverse-order] [--smoke]\n"
                "48cells: eight FE64-input routes x six actual RHS prefixes. Default200ms, two warmups and eight measurement rounds.\n"
                "Smoke defaults1ms/two measured rounds. Output never overwritten. Pin externally; no power policy changes.\n"
                "Exit0 complete; exit2 calibration/duration unqualified; exit1 error/mismatch.\n";
            return 0;
        }
        fd = ::open(options.output.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
        if (fd < 0) throw std::runtime_error(std::string("exclusive output open: ") + std::strerror(errno));
        const auto start_power = power();
        const auto wall_start = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        Corpus corpus; std::vector<Cell> cells; cells.reserve(48);
        for (unsigned g = 0; g < sizes.size(); ++g) for (unsigned r = 0; r < routes.size(); ++r)
            cells.push_back({static_cast<unsigned>(cells.size()), g, r, sizes[g]});
        const auto base = base_order(options.seed);
        std::vector<Record> records; records.reserve(cells.size() * (max_attempts + warmups + options.rounds));
        std::vector<Preservation> preservation;
        std::string status = "complete", failure; int result = 0;
        try {
            make_corpus(corpus, options.seed); validate(cells, corpus);
            preservation.push_back(check_inputs(corpus, options.seed, "after_untimed_validation"));
            if (!preservation.back().verified) throw std::runtime_error("untimed input mutation");
            const auto order = schedule(base, 0, options.reverse);
            for (unsigned position = 0; position < order.size(); ++position) {
                auto& cell = cells[order[position]];
                if (!calibrate(cell, corpus, options, position, records)) {
                    status = "unqualified"; result = 2; failure = "calibration exhausted for cell " + std::to_string(cell.id); break;
                }
            }
            if (!result) for (unsigned round = 0; round < warmups + options.rounds && !result; ++round) {
                const bool warmup = round < warmups;
                const auto order_round = schedule(base, round, options.reverse);
                for (unsigned position = 0; position < order_round.size(); ++position) {
                    const auto& cell = cells[order_round[position]];
                    records.push_back(measure(cell, corpus, options, warmup ? "warmup" : "measurement",
                        static_cast<int>(warmup ? round : round - warmups), position, 0, cell.jobs, records.size()));
                    if (!records.back().verified) throw std::runtime_error("timed output/clock mismatch");
                    if (!records.back().duration_qualified) {
                        status = "duration_unqualified"; result = 2;
                        failure = "later-phase region failed duration: " + std::to_string(records.back().region) +
                            " stop=" + stop_name(records.back().stop);
                        break;
                    }
                }
            }
        } catch (const std::exception& error) { status = "error"; failure = error.what(); result = 1; }
        if (corpus.ready) {
            preservation.push_back(check_inputs(corpus, options.seed, "after_regions_or_failure"));
            if (!preservation.back().verified) { status = "error"; failure += "; final input mutation"; result = 1; }
        }
        const auto end_power = power();
        write_all(fd, json(options, corpus, cells, records, preservation, base, start_power, end_power, wall_start, status, failure));
        if (::close(fd) != 0) { fd = -1; throw std::runtime_error("output close failed"); } fd = -1;
        std::cout << "P4 " << status << " cells=" << cells.size() << " raw_regions=" << records.size() << " output=" << options.output << '\n';
        if (!failure.empty()) std::cerr << failure << '\n';
        return result;
    } catch (const std::exception& error) {
        if (fd >= 0) ::close(fd);
        std::cerr << "P4 error: " << error.what() << '\n'; return 1;
    }
}
