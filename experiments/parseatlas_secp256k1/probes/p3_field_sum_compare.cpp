// P3 final-only field sum; research only. Link the corrected library without LTO.
#include "p3_field_sum_kernels.hpp"
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
#include <sched.h>
#include <unistd.h>

#if defined(__GNUC__) && !defined(__clang__)
#define PA_BOUNDARY __attribute__((noinline, noipa))
#elif defined(__clang__)
#define PA_BOUNDARY __attribute__((noinline))
#else
#error "P3 requires GCC or Clang compiler memory barriers"
#endif
#define PA_STRING_IMPL(x) #x
#define PA_STRING(x) PA_STRING_IMPL(x)

namespace {
using FE = pa_p3::FE;
using FE52 = pa_p3::FE52;
using Bytes = std::array<std::uint8_t, 32>;
using Clock = std::chrono::steady_clock;
constexpr std::array<std::size_t, 6> sizes{1, 16, 256, 4096, 65536, 1048576};
constexpr std::array<const char*, 7> routes{
    "fe64_serial", "fe52_e2e_full1", "fe52_e2e_full16", "fe52_e2e_full256",
    "fe52_e2e_full4094", "fe52_e2e_weak4095", "fe52_resident_weak4095"};
constexpr std::array<unsigned, 7> chunks{1, 1, 16, 256, 4094, 4095, 4095};
constexpr unsigned warmups = 2, max_attempts = 12;
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
void hash52(std::uint64_t& hash, const FE52& value) {
    for (auto word : value.n) for (unsigned b = 0; b < 8; ++b)
        hash_byte(hash, static_cast<std::uint8_t>(word >> (56 - 8 * b)));
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
bool canonical52(const FE52& value) {
    using namespace secp256k1::fast::fe52_constants;
    constexpr std::array<std::uint64_t, 5> p{P0, P1, P2, P3, P4};
    for (unsigned i = 0; i < 4; ++i) if (value.n[i] > M52) return false;
    if (value.n[4] > M48) return false;
    for (int i = 4; i >= 0; --i)
        if (value.n[i] != p[static_cast<unsigned>(i)]) return value.n[i] < p[static_cast<unsigned>(i)];
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
    std::vector<FE52> resident;
    std::array<std::uint64_t, 6> prefix_hash{}, resident_prefix_hash{};
    std::uint64_t hash64 = fnv_basis, hash52_raw = fnv_basis, setup_ns = 0;
    bool ready = false;
};
void make_corpus(Corpus& corpus, std::uint64_t seed) {
    RNG rng{seed ^ corpus_salt};
    corpus.x0 = random_canonical(rng); // x0 first, then exactly max(N) accepted RHS.
    if (!canonical64(corpus.x0)) throw std::runtime_error("noncanonical x0");
    hash_bytes(corpus.hash64, corpus.x0.to_bytes());
    hash_bytes(corpus.hash52_raw, corpus.x0.to_bytes());
    corpus.rhs.resize(sizes.back()); corpus.resident.resize(sizes.back());
    unsigned prefix = 0;
    for (std::size_t i = 0; i < corpus.rhs.size(); ++i) {
        corpus.rhs[i] = random_canonical(rng);
        if (!canonical64(corpus.rhs[i])) throw std::runtime_error("noncanonical RHS");
        hash_bytes(corpus.hash64, corpus.rhs[i].to_bytes());
        if (i + 1 == sizes[prefix]) corpus.prefix_hash[prefix++] = corpus.hash64;
    }
    barrier(); const auto begin = Clock::now();
    for (std::size_t i = 0; i < corpus.rhs.size(); ++i)
        corpus.resident[i] = FE52::from_fe(corpus.rhs[i]);
    barrier(); const auto end = Clock::now();
    corpus.setup_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
    prefix = 0;
    for (std::size_t i = 0; i < corpus.resident.size(); ++i) {
        const auto& value = corpus.resident[i];
        if (!canonical52(value) || value.to_fe().to_bytes() != corpus.rhs[i].to_bytes())
            throw std::runtime_error("resident conversion mismatch");
        hash52(corpus.hash52_raw, value);
        if (i + 1 == sizes[prefix]) corpus.resident_prefix_hash[prefix++] = corpus.hash52_raw;
    }
    corpus.ready = true;
}
struct Preservation {
    std::string phase;
    bool verified = false;
    std::size_t exact_rhs_checked = 0;
    std::uint64_t hash64 = fnv_basis, hash52_raw = fnv_basis;
};
Preservation check_inputs(const Corpus& corpus, std::uint64_t seed, const std::string& phase) {
    Preservation check; check.phase = phase;
    RNG rng{seed ^ corpus_salt};
    const auto x0 = random_canonical(rng);
    check.verified = canonical64(corpus.x0) && x0.limbs() == corpus.x0.limbs();
    hash_bytes(check.hash64, corpus.x0.to_bytes());
    hash_bytes(check.hash52_raw, corpus.x0.to_bytes());
    for (std::size_t i = 0; i < corpus.rhs.size(); ++i) {
        const auto expected = random_canonical(rng);
        const auto resident = FE52::from_fe(expected);
        check.verified = canonical64(corpus.rhs[i]) && canonical52(corpus.resident[i]) &&
            expected.limbs() == corpus.rhs[i].limbs() &&
            std::equal(std::begin(resident.n), std::end(resident.n), std::begin(corpus.resident[i].n)) &&
            check.verified;
        hash_bytes(check.hash64, corpus.rhs[i].to_bytes());
        hash52(check.hash52_raw, corpus.resident[i]);
        ++check.exact_rhs_checked;
    }
    check.verified = check.verified && check.hash64 == corpus.hash64 && check.hash52_raw == corpus.hash52_raw;
    return check;
}

template<unsigned Route> inline FE evaluate(const Corpus& corpus, std::size_t count) {
    if constexpr (Route == 0) return pa_p3::sum_fe64(corpus.x0, corpus.rhs.data(), count);
    else if constexpr (Route == 1) return pa_p3::sum_fe52_e2e<1, false>(corpus.x0, corpus.rhs.data(), count);
    else if constexpr (Route == 2) return pa_p3::sum_fe52_e2e<16, false>(corpus.x0, corpus.rhs.data(), count);
    else if constexpr (Route == 3) return pa_p3::sum_fe52_e2e<256, false>(corpus.x0, corpus.rhs.data(), count);
    else if constexpr (Route == 4) return pa_p3::sum_fe52_e2e<4094, false>(corpus.x0, corpus.rhs.data(), count);
    else if constexpr (Route == 5) return pa_p3::sum_fe52_e2e<4095, true>(corpus.x0, corpus.rhs.data(), count);
    else return pa_p3::sum_fe52_resident<4095, true>(corpus.x0, corpus.resident.data(), count);
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
constexpr std::array<Job, 7> jobs{full_job<0>, full_job<1>, full_job<2>, full_job<3>, full_job<4>, full_job<5>, full_job<6>};
constexpr std::array<Evaluator, 7> evaluators{evaluate<0>, evaluate<1>, evaluate<2>, evaluate<3>, evaluate<4>, evaluate<5>, evaluate<6>};
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
};
Record measure(const Cell& cell, const Corpus& corpus, const Options& options, const std::string& phase,
               int round, unsigned position, unsigned attempt, std::uint64_t count_jobs, std::size_t region) {
    if (!count_jobs || count_jobs > operation_cap / cell.count) throw std::runtime_error("operation cap exceeded");
    Bytes output{};
    const auto start_power = power();
    const auto cpu_start = sched_getcpu();
    const Job job = jobs[cell.route]; // Route choice is outside the per-RHS loop.
    barrier(); const auto begin = Clock::now();
    for (std::uint64_t i = 0; i < count_jobs; ++i) job(corpus, cell.count, output);
    const auto end = Clock::now(); barrier();
    const auto cpu_end = sched_getcpu(); const auto end_power = power();
    const auto signed_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    const auto elapsed = signed_elapsed > 0 ? static_cast<std::uint64_t>(signed_elapsed) : 0;
    const auto operations = count_jobs * cell.count;
    return {region, phase, cell.id, round, position, attempt, count_jobs, operations,
        steady_ns(begin), steady_ns(end), elapsed, static_cast<double>(elapsed) / static_cast<double>(count_jobs),
        static_cast<double>(elapsed) / static_cast<double>(operations), output,
        output == cell.expected && signed_elapsed > 0, static_cast<double>(elapsed) >= options.min_ms * 1e6,
        cpu_start, cpu_end, start_power, end_power, 0};
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
struct Order { std::array<unsigned, 6> groups{0, 1, 2, 3, 4, 5}; std::array<unsigned, 7> routes{0, 1, 2, 3, 4, 5, 6}; };
Order base_order(std::uint64_t seed) {
    Order order; RNG rng{seed ^ UINT64_C(0x6f726465725f7033)};
    for (unsigned n = order.groups.size(); n > 1; --n) std::swap(order.groups[n - 1], order.groups[rng.next() % n]);
    for (unsigned n = order.routes.size(); n > 1; --n) std::swap(order.routes[n - 1], order.routes[rng.next() % n]);
    return order;
}
std::vector<unsigned> schedule(const Order& base, unsigned round, bool reverse) {
    auto groups = base.groups;
    std::rotate(groups.begin(), groups.begin() + (round / 2) % groups.size(), groups.end());
    if (round % 2) std::reverse(groups.begin(), groups.end());
    std::vector<unsigned> order; order.reserve(42);
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
std::string json(const Options& options, const Corpus& corpus, const std::vector<Cell>& cells,
                 const std::vector<Record>& records, const std::vector<Preservation>& preservation,
                 const Order& base, const Power& start_power, const Power& end_power, std::int64_t wall_start_ns,
                 const std::string& status, const std::string& failure) {
    std::ostringstream out; out << std::setprecision(17) << std::boolalpha;
    out << "{\n\"schema\":\"parseatlas.p3.field_sum_compare.v1\",\n\"status\":" << quote(status)
        << ",\n\"failure\":" << quote(failure) << ",\n\"seed\":" << options.seed
        << ",\n\"min_ms\":" << options.min_ms << ",\n\"smoke\":" << options.smoke
        << ",\n\"reverse_order\":" << options.reverse << ",\n\"warmup_rounds_requested\":" << warmups
        << ",\n\"measurement_rounds_requested\":" << options.rounds
        << ",\n\"compiler\":" << quote(__VERSION__) << ",\n\"cplusplus\":" << __cplusplus
        << ",\n\"run_start_unix_ns\":" << wall_start_ns << ",\n\"harness_macros\":";
    write_macros(out);
    out << ",\n\"power_start\":"; write_power(out, start_power);
    out << ",\n\"power_end\":"; write_power(out, end_power);
    out << ",\n\"contract\":{"
        "\"result\":\"(x0+sum(rhs[0..N))) mod p, final canonical FE64 and32BEbytes; N countsRHS; x0once; no prefix observable\","
        "\"scope\":\"field final-only vector sum; no single-add replacement, CT certification, scalar or upper-layer gain\","
        "\"e2e\":\"canonical FE64 RHS, FE52 packing at each use inside job; all seed packing and final conversion timed\","
        "\"resident\":\"canonical FE52 RHS preconversion excluded from repeated jobs; seed packing and final canonical output timed; NOT E2E\","
        "\"normalization\":\"canonical accumulator begins every chunk; final to_fe supplies full normalization once; weak routes normalize_weak before final decode\","
        "\"allocation\":\"both input arrays allocated and constructed outside timing; setup observation descriptive, not a break-even measurement\","
        "\"timing\":\"whole noinline jobs include full kernel, call, x0reset, RHS loads/packing, loop, reduction and32byte serialization; every output materialized; no per-RHS barrier\","
        "\"verification\":\"all42cells validated rawcanonical andfullbytes versus actualFE64 before timing; last identical job per region checked outside clock; independent Boost test is separate\","
        "\"input_preservation\":\"exact regeneration compares all rawFE64/FE52 words after untimed validation and after all regions, plus all-byte FNV hashes\","
        "\"calibration\":\"one job initially; two consecutive regions >=min_ms at same count; short resets streak and grows ceil(1.5*target*jobs/elapsed), capped; no warmup if unqualified\","
        "\"ordering\":\"SplitMix64 seeded size and route permutations, paired rotate/reverse schedules; reverse-order reverses each complete schedule only\","
        "\"statistics\":\"retain all records including shorts/failures; verified measurement-region means and same-round FE64/candidate ratios; jobs can differ\","
        "\"cpu\":\"single thread by deliberate dependency experiment; external pinning; endpoints do not prove no migration; power snapshots not average frequency; no policy changes\","
        "\"backend\":\"harness macros do not prove library flags; external manifest must bind source/kernel/library/binary hashes and commands; no LTO\","
        "\"operation_cap_per_region\":" << operation_cap << ",\"calibration_attempt_limit\":" << max_attempts
        << "},\n\"corpus\":{\"ready\":" << corpus.ready
        << ",\"recipe\":\"SplitMix64 state=seed xor0x6669656c645f7033; four successive words encoded BE to32bytes; strict parse rejects>=p, accepts zero; first accepted value x0, then1048576 RHS in order; each size uses actual prefix\""
        << ",\"hash_recipe\":\"FNV1a64 unsigned wrap; FE64 hash=x0BE32 then RHS BE32; resident hash=x0BE32 then each RHS five raw limbs, each word BE8; prefix hashes use same initial x0\""
        << ",\"source_recipe_sha256\":\"PENDING IMPORT\""
        << ",\"x0_be32\":" << quote(hex(corpus.x0.to_bytes()))
        << ",\"rhs_count\":" << corpus.rhs.size() << ",\"rhs64_bytes\":" << corpus.rhs.size() * sizeof(FE)
        << ",\"resident_rhs52_bytes\":" << corpus.resident.size() * sizeof(FE52)
        << ",\"seed64_bytes\":" << sizeof(FE) << ",\"checksum_fnv1a64\":" << quote(hex64(corpus.hash64))
        << ",\"resident_raw_checksum_fnv1a64\":" << quote(hex64(corpus.hash52_raw))
        << ",\"resident_setup_one_observation_ns\":" << corpus.setup_ns << ",\"prefixes\":[";
    for (unsigned g = 0; g < sizes.size(); ++g) {
        if (g) out << ',';
        out << "{\"rhs_count\":" << sizes[g] << ",\"checksum_fnv1a64\":" << quote(hex64(corpus.prefix_hash[g]))
            << ",\"resident_raw_checksum_fnv1a64\":" << quote(hex64(corpus.resident_prefix_hash[g])) << '}';
    }
    out << "]},\n\"input_preservation\":[";
    for (unsigned i = 0; i < preservation.size(); ++i) {
        if (i) out << ',';
        const auto& p = preservation[i];
        out << "{\"phase\":" << quote(p.phase) << ",\"verified\":" << p.verified
            << ",\"exact_rhs_checked\":" << p.exact_rhs_checked << ",\"checksum_fnv1a64\":" << quote(hex64(p.hash64))
            << ",\"resident_raw_checksum_fnv1a64\":" << quote(hex64(p.hash52_raw)) << '}';
    }
    out << "],\n\"base_group_order\":[";
    for (unsigned i = 0; i < base.groups.size(); ++i) { if (i) out << ','; out << base.groups[i]; }
    out << "],\n\"base_route_order\":[";
    for (unsigned i = 0; i < base.routes.size(); ++i) { if (i) out << ','; out << base.routes[i]; }
    out << "],\n\"cells\":[\n";
    for (const auto& cell : cells) {
        if (cell.id) out << ",\n";
        std::vector<double> sum_values, rhs_values; unsigned short_count = 0;
        for (const auto& record : records) if (record.cell == cell.id && record.phase == "measurement" && record.verified) {
            sum_values.push_back(record.ns_per_sum); rhs_values.push_back(record.ns_per_rhs);
            if (!record.floor_met) ++short_count;
        }
        out << "{\"id\":" << cell.id << ",\"group\":" << cell.group << ",\"route_index\":" << cell.route
            << ",\"route\":" << quote(routes[cell.route]) << ",\"rhs_count\":" << cell.count
            << ",\"chunk_rhs\":" << chunks[cell.route] << ",\"resident_setup_excluded\":" << (cell.route == 6)
            << ",\"validated\":" << cell.validated << ",\"qualified\":" << cell.qualified << ",\"jobs_per_region\":" << cell.jobs
            << ",\"expected_final_be32\":" << quote(hex(cell.expected)) << ",\"expected_checksum_fnv1a64\":" << quote(hex64(checksum(cell.expected)))
            << ",\"short_measurement_regions\":" << short_count << ",\"summary_ns_per_sum\":";
        summary(out, sum_values); out << ",\"summary_ns_per_rhs\":"; summary(out, rhs_values); out << '}';
    }
    out << "\n],\n\"regions\":[\n";
    for (const auto& r : records) {
        if (r.region) out << ",\n";
        out << "{\"region\":" << r.region << ",\"phase\":" << quote(r.phase) << ",\"cell\":" << r.cell
            << ",\"round\":" << r.round << ",\"order_position\":" << r.position << ",\"attempt\":" << r.attempt
            << ",\"jobs\":" << r.jobs << ",\"operations\":" << r.operations
            << ",\"begin_steady_ns\":" << r.begin_ns << ",\"end_steady_ns\":" << r.end_ns << ",\"elapsed_ns\":" << r.elapsed_ns
            << ",\"ns_per_sum\":" << r.ns_per_sum << ",\"ns_per_rhs\":" << r.ns_per_rhs
            << ",\"actual_final_be32\":" << quote(hex(r.output)) << ",\"checksum_fnv1a64\":" << quote(hex64(checksum(r.output)))
            << ",\"verified\":" << r.verified << ",\"floor_met\":" << r.floor_met
            << ",\"cpu_start\":" << r.cpu_start << ",\"cpu_end\":" << r.cpu_end << ",\"qualification_streak\":" << r.streak
            << ",\"power_start\":"; write_power(out, r.power_start);
        out << ",\"power_end\":"; write_power(out, r.power_end); out << '}';
    }
    out << "\n],\n\"paired_region_mean_ratios\":[\n";
    bool first = true;
    for (unsigned group = 0; group < sizes.size(); ++group) for (unsigned route = 1; route < routes.size(); ++route) {
        if (!first) out << ",\n";
        first = false;
        out << "{\"group\":" << group << ",\"baseline_cell\":" << group * routes.size()
            << ",\"candidate_cell\":" << group * routes.size() + route
            << ",\"same_rhs_setup_contract\":" << (route != 6) << ",\"ratios_baseline_over_candidate\":[";
        std::vector<double> values;
        for (unsigned round = 0; round < options.rounds; ++round) {
            const Record* baseline = nullptr; const Record* candidate = nullptr;
            for (const auto& r : records) if (r.phase == "measurement" && r.round == static_cast<int>(round) && r.verified) {
                if (r.cell == group * routes.size()) baseline = &r;
                if (r.cell == group * routes.size() + route) candidate = &r;
            }
            if (!baseline || !candidate) continue;
            if (!values.empty()) out << ',';
            const double ratio = baseline->ns_per_sum / candidate->ns_per_sum; values.push_back(ratio);
            out << "{\"round\":" << round << ",\"baseline_region\":" << baseline->region
                << ",\"candidate_region\":" << candidate->region << ",\"ratio\":" << ratio << '}';
        }
        out << "],\"summary_ratio\":"; summary(out, values); out << '}';
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
            std::cout << "Usage: p3_field_sum_compare --output NEW.json [--seed UINT64] [--min-ms 0.1..1000] [--rounds EVEN_2..128] [--reverse-order] [--smoke]\n"
                "42cells: seven routes x six actual RHS prefixes. Default200ms, two warmups and eight measurement rounds.\n"
                "Smoke defaults1ms/two measured rounds. Output never overwritten. Pin externally; no power policy changes.\n"
                "Exit0 complete; exit2 unqualified calibration; exit1 error/mismatch.\n";
            return 0;
        }
        fd = ::open(options.output.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
        if (fd < 0) throw std::runtime_error(std::string("exclusive output open: ") + std::strerror(errno));
        const auto start_power = power();
        const auto wall_start = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        Corpus corpus; std::vector<Cell> cells; cells.reserve(42);
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
            if (!result) for (unsigned round = 0; round < warmups + options.rounds; ++round) {
                const bool warmup = round < warmups;
                const auto order_round = schedule(base, round, options.reverse);
                for (unsigned position = 0; position < order_round.size(); ++position) {
                    const auto& cell = cells[order_round[position]];
                    records.push_back(measure(cell, corpus, options, warmup ? "warmup" : "measurement",
                        static_cast<int>(warmup ? round : round - warmups), position, 0, cell.jobs, records.size()));
                    if (!records.back().verified) throw std::runtime_error("timed output/clock mismatch");
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
        std::cout << "P3 " << status << " cells=" << cells.size() << " raw_regions=" << records.size() << " output=" << options.output << '\n';
        if (!failure.empty()) std::cerr << failure << '\n';
        return result;
    } catch (const std::exception& error) {
        if (fd >= 0) ::close(fd);
        std::cerr << "P3 error: " << error.what() << '\n'; return 1;
    }
}
