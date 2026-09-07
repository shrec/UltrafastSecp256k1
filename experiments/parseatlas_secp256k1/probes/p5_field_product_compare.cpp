// P5 canonical field products: scheduling controls; research only, no LTO.
// Task MCP is owner-suspended; worker Source Graph tools are unavailable.
// Manager-verified exact inputs supplied this bounded new-file implementation.
#include "p5_field_product_kernels.hpp"
#include "secp256k1/field.hpp"
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
#error "P5 requires GCC or Clang compiler memory barriers"
#endif
#define PA_STRING_IMPL(x) #x
#define PA_STRING(x) PA_STRING_IMPL(x)

namespace {
using FE = secp256k1::fast::FieldElement;
using Limbs = pa_p5::Limbs;
using Wide = pa_p5::Wide;
using States = std::array<Limbs, 4>;
using Bytes = std::array<std::uint8_t, 128>;
using Clock = std::chrono::steady_clock;
constexpr unsigned steps = 1024, corpus_count = 256, cell_count = 20;
constexpr std::array<const char*, 10> routes{
    "mul_api", "mul_row_serial", "mul_comba_serial", "mul_comba_parallel",
    "square_api", "square_row_serial", "square_comba_serial", "square_comba_parallel",
    "reduce_serial", "reduce_parallel"};
constexpr std::array<const char*, 3> operations{"multiply", "square", "raw_reduce512"};
constexpr std::array<unsigned, 6> group_start{0,4,8,12,16,18};
constexpr unsigned warmups = 2, max_attempts = 12, max_extensions = 12;
constexpr std::uint64_t operation_cap = 1000000000;
constexpr std::uint64_t fnv_basis = UINT64_C(14695981039346656037);
constexpr std::uint64_t corpus_salt = UINT64_C(0x6669656c645f7035);
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
template<std::size_t N> std::string hex(const std::array<std::uint8_t, N>& bytes) {
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
template<std::size_t N> void hash_bytes(std::uint64_t& hash, const std::array<std::uint8_t, N>& bytes) {
    for (auto byte : bytes) hash_byte(hash, byte);
}
template<std::size_t N> std::uint64_t checksum(const std::array<std::uint8_t, N>& bytes) {
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
        std::array<std::uint8_t, 32> bytes{};
        for (unsigned w = 0; w < 4; ++w) {
            const auto word = rng.next();
            for (unsigned b = 0; b < 8; ++b)
                bytes[w * 8 + b] = static_cast<std::uint8_t>(word >> (56 - 8 * b));
        }
        FE result;
        if (FE::parse_bytes_strict(bytes, result) && result.limbs() != Limbs{}) return result;
    }
}
struct Corpus {
    std::array<FE, 4> seeds{};
    std::array<FE, corpus_count> rhs{};
    std::array<Wide, corpus_count> raw{};
    std::uint64_t hash64 = fnv_basis;
    bool ready = false;
};
void hash_wide(std::uint64_t& hash, const Wide& wide) {
    for (int word = 7; word >= 0; --word)
        for (int byte = 7; byte >= 0; --byte)
            hash_byte(hash, static_cast<std::uint8_t>(wide[static_cast<unsigned>(word)] >> (8 * byte)));
}
void make_corpus(Corpus& corpus, std::uint64_t seed) {
    RNG rng{seed ^ corpus_salt};
    corpus.hash64 = fnv_basis;
    for (auto& x : corpus.seeds) { x = random_canonical(rng); hash_bytes(corpus.hash64, x.to_bytes()); }
    for (auto& x : corpus.rhs) { x = random_canonical(rng); hash_bytes(corpus.hash64, x.to_bytes()); }
    for (auto& w : corpus.raw) {
        for (auto& word : w) word = rng.next();
        hash_wide(corpus.hash64, w);
    }
    corpus.ready = true;
}
struct Preservation {
    std::string phase;
    bool verified = false;
    std::size_t exact_rhs_checked = 0, exact_seed_checked = 0, exact_raw512_checked = 0;
    std::uint64_t hash64 = fnv_basis;
};
Preservation check_inputs(const Corpus& corpus, std::uint64_t seed, const std::string& phase) {
    Corpus expected; make_corpus(expected, seed);
    Preservation check; check.phase = phase; check.verified = corpus.ready;
    for (unsigned i = 0; i < corpus.seeds.size(); ++i) {
        check.verified = canonical64(corpus.seeds[i]) &&
            corpus.seeds[i].limbs() == expected.seeds[i].limbs() && check.verified;
        hash_bytes(check.hash64, corpus.seeds[i].to_bytes()); ++check.exact_seed_checked;
    }
    for (unsigned i = 0; i < corpus_count; ++i) {
        check.verified = canonical64(corpus.rhs[i]) &&
            corpus.rhs[i].limbs() == expected.rhs[i].limbs() && check.verified;
        hash_bytes(check.hash64, corpus.rhs[i].to_bytes()); ++check.exact_rhs_checked;
    }
    for (unsigned i = 0; i < corpus_count; ++i) {
        check.verified = corpus.raw[i] == expected.raw[i] && check.verified;
        hash_wide(check.hash64, corpus.raw[i]); ++check.exact_raw512_checked;
    }
    check.verified = check.verified && check.hash64 == corpus.hash64 && check.hash64 == expected.hash64;
    return check;
}
Wide raw_input(const Limbs& x, const Corpus& corpus, unsigned step) {
    Wide w = corpus.raw[step & (corpus_count - 1)];
    for (unsigned i = 0; i < 4; ++i) w[i] ^= x[i];
    return w;
}
template<unsigned Route> inline Limbs primitive(const Limbs& x, const Corpus& corpus, unsigned step) {
    if constexpr (Route == 0) return (FE::from_limbs_raw(x) * corpus.rhs[step & (corpus_count - 1)]).limbs();
    else if constexpr (Route == 4) return FE::from_limbs_raw(x).square().limbs();
    else if constexpr (Route >= 8) {
        const auto w = raw_input(x, corpus, step);
        if constexpr (Route == 8) return pa_p5::reduce_serial(w);
        else return pa_p5::reduce_parallel(w);
    } else {
        Wide product;
        if constexpr (Route == 1) product = pa_p5::mul_row(x, corpus.rhs[step & (corpus_count - 1)].limbs());
        else if constexpr (Route == 2 || Route == 3) product = pa_p5::mul_comba(x, corpus.rhs[step & (corpus_count - 1)].limbs());
        else if constexpr (Route == 5) product = pa_p5::mul_row(x, x);
        else product = pa_p5::square_comba(x);
        if constexpr (Route == 3 || Route == 7) return pa_p5::reduce_parallel(product);
        else return pa_p5::reduce_serial(product);
    }
}
template<unsigned Route, unsigned Lanes> inline States evaluate(const Corpus& corpus) {
    States result{};
    // ILP is deliberately one-core independent state, not worker threads. The
    // step/lane nesting exposes independent operations without benchmark barriers.
    if constexpr (Route == 0 || Route == 4) {
        std::array<FE, Lanes> x;
        for (unsigned lane = 0; lane < Lanes; ++lane) x[lane] = corpus.seeds[lane];
        for (unsigned step = 0; step < steps; ++step)
            for (unsigned lane = 0; lane < Lanes; ++lane) {
                if constexpr (Route == 0) x[lane] = x[lane] * corpus.rhs[step & (corpus_count - 1)];
                else x[lane] = x[lane].square();
            }
        for (unsigned lane = 0; lane < Lanes; ++lane) result[lane] = x[lane].limbs();
    } else {
        for (unsigned lane = 0; lane < Lanes; ++lane) result[lane] = corpus.seeds[lane].limbs();
        for (unsigned step = 0; step < steps; ++step)
            for (unsigned lane = 0; lane < Lanes; ++lane)
                result[lane] = primitive<Route>(result[lane], corpus, step);
    }
    return result;
}
Bytes serialize(const States& states, unsigned lanes) {
    if (lanes != 1 && lanes != 4) throw std::runtime_error("serialization requires one or four lanes");
    Bytes output{}; // Chain1 explicitly includes 96 inactive zero output bytes.
    for (unsigned lane = 0; lane < states.size(); ++lane) {
        if (lane == lanes) break;
        const auto bytes = FE::from_limbs_raw(states[lane]).to_bytes();
        std::copy(bytes.begin(), bytes.end(), output.begin() + lane * 32);
    }
    return output;
}
template<unsigned Route, unsigned Lanes> PA_BOUNDARY void full_job(const Corpus& corpus, std::size_t, Bytes& output) {
    output = serialize(evaluate<Route, Lanes>(corpus), Lanes);
    asm volatile("" : : "m"(output) : "memory"); // No per-step barrier.
}
using Job = void (*)(const Corpus&, std::size_t, Bytes&);
using Evaluator = States (*)(const Corpus&);
using Stepper = Limbs (*)(const Limbs&, const Corpus&, unsigned);
constexpr std::array<Stepper,10> steppers{
    primitive<0>,primitive<1>,primitive<2>,primitive<3>,primitive<4>,
    primitive<5>,primitive<6>,primitive<7>,primitive<8>,primitive<9>};
constexpr std::array<Job,cell_count> jobs{
    full_job<0,1>,full_job<1,1>,full_job<2,1>,full_job<3,1>,
    full_job<0,4>,full_job<1,4>,full_job<2,4>,full_job<3,4>,
    full_job<4,1>,full_job<5,1>,full_job<6,1>,full_job<7,1>,
    full_job<4,4>,full_job<5,4>,full_job<6,4>,full_job<7,4>,
    full_job<8,1>,full_job<9,1>,full_job<8,4>,full_job<9,4>};
constexpr std::array<Evaluator,cell_count> evaluators{
    evaluate<0,1>,evaluate<1,1>,evaluate<2,1>,evaluate<3,1>,
    evaluate<0,4>,evaluate<1,4>,evaluate<2,4>,evaluate<3,4>,
    evaluate<4,1>,evaluate<5,1>,evaluate<6,1>,evaluate<7,1>,
    evaluate<4,4>,evaluate<5,4>,evaluate<6,4>,evaluate<7,4>,
    evaluate<8,1>,evaluate<9,1>,evaluate<8,4>,evaluate<9,4>};
struct Cell {
    unsigned id, group, route;
    std::size_t count;
    unsigned lanes;
    Bytes expected{};
    bool validated = false, qualified = false;
    std::uint64_t jobs = 0, trace_checks = 0, trace_hash = fnv_basis;
    std::uint64_t zero_states = 0, one_states = 0;
};
std::vector<Cell> make_cells() {
    std::vector<Cell> cells; cells.reserve(cell_count);
    for (unsigned group = 0; group < 6; ++group) {
        const unsigned lanes = group % 2 ? 4 : 1;
        const unsigned first_route = group / 2 * 4, width = group < 4 ? 4 : 2;
        for (unsigned offset = 0; offset < width; ++offset)
            cells.push_back({static_cast<unsigned>(cells.size()), group, first_route + offset, steps * lanes, lanes});
    }
    return cells;
}
Limbs reference_step(const Limbs& x, const Corpus& corpus, unsigned step, unsigned kind) {
    if (kind == 0) return (FE::from_limbs_raw(x) * corpus.rhs[step & (corpus_count - 1)]).limbs();
    if (kind == 1) return FE::from_limbs_raw(x).square().limbs();
    const auto w = raw_input(x, corpus, step);
    const Limbs low{w[0],w[1],w[2],w[3]}, high{w[4],w[5],w[6],w[7]};
    // H*2^256+L == H*K+L mod p. from_limbs canonicalizes arbitrary256
    // halves; this is an actual corrected FE64 pipeline, not either reducer.
    return (FE::from_limbs(high) * FE::from_uint64(UINT64_C(0x1000003d1)) + FE::from_limbs(low)).limbs();
}
void validate(std::vector<Cell>& cells, const Corpus& corpus) {
    for (auto& cell : cells) {
        States actual{}, expected{};
        for (unsigned lane = 0; lane < cell.lanes; ++lane)
            actual[lane] = expected[lane] = corpus.seeds[lane].limbs();
        for (unsigned step = 0; step < steps; ++step)
            for (unsigned lane = 0; lane < cell.lanes; ++lane) {
                actual[lane] = steppers[cell.route](actual[lane], corpus, step);
                expected[lane] = reference_step(expected[lane], corpus, step, cell.group / 2);
                const auto a = FE::from_limbs_raw(actual[lane]), e = FE::from_limbs_raw(expected[lane]);
                if (!canonical64(a) || !canonical64(e) || actual[lane] != expected[lane] || a.to_bytes() != e.to_bytes())
                    throw std::runtime_error("trace mismatch cell=" + std::to_string(cell.id) +
                        " step=" + std::to_string(step) + " lane=" + std::to_string(lane));
                ++cell.trace_checks; hash_bytes(cell.trace_hash, a.to_bytes());
                cell.zero_states += actual[lane] == Limbs{};
                cell.one_states += actual[lane] == Limbs{1,0,0,0};
            }
        cell.expected = serialize(expected, cell.lanes);
        const auto complete = evaluators[cell.id](corpus);
        Bytes output{}; jobs[cell.id](corpus, cell.count, output);
        cell.validated = complete == expected && serialize(complete, cell.lanes) == cell.expected && output == cell.expected;
        if (!cell.validated) throw std::runtime_error("complete job mismatch cell=" + std::to_string(cell.id));
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
    double ns_per_job, ns_per_operation;
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
    const Job job = jobs[cell.id];
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
struct Order { std::array<unsigned,6> groups{0,1,2,3,4,5}; std::array<unsigned,4> routes{0,1,2,3}; };
Order base_order(std::uint64_t seed) {
    Order order; RNG rng{seed ^ UINT64_C(0x6f726465725f7035)};
    for (unsigned n = order.groups.size(); n > 1; --n) std::swap(order.groups[n-1],order.groups[rng.next()%n]);
    for (unsigned n = order.routes.size(); n > 1; --n) std::swap(order.routes[n-1],order.routes[rng.next()%n]);
    return order;
}
std::vector<unsigned> schedule(const Order& base, unsigned round, bool reverse) {
    auto groups = base.groups;
    std::rotate(groups.begin(), groups.begin() + (round/2)%groups.size(), groups.end());
    if (round%2) std::reverse(groups.begin(),groups.end());
    std::vector<unsigned> order; order.reserve(cell_count);
    for (const auto group : groups) {
        auto variants = base.routes;
        std::rotate(variants.begin(), variants.begin() + (round/2+group)%variants.size(), variants.end());
        if (round%2) std::reverse(variants.begin(),variants.end());
        for (const auto offset : variants)
            if (offset < (group < 4 ? 4U : 2U)) order.push_back(group_start[group]+offset);
    }
    if (reverse) std::reverse(order.begin(),order.end());
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
void write_comparison(std::ostream& out, const std::vector<Cell>& cells, unsigned aid, unsigned bid,
                      const Options& options, const std::vector<Record>& records) {
    const auto& a = cells[aid]; const auto& b = cells[bid];
    out << "{\"numerator_cell\":" << aid << ",\"denominator_cell\":" << bid
        << ",\"operation\":" << quote(operations[a.group/2])
        << ",\"numerator_route\":" << quote(routes[a.route]) << ",\"denominator_route\":" << quote(routes[b.route])
        << ",\"numerator_lanes\":" << a.lanes << ",\"denominator_lanes\":" << b.lanes << ",\"ratios_A_over_B\":[";
    std::vector<double> values; unsigned wins = 0, losses = 0, ties = 0;
    for (unsigned round = 0; round < options.rounds; ++round) {
        const Record* ra = nullptr; const Record* rb = nullptr;
        for (const auto& r : records)
            if (r.phase == "measurement" && r.round == static_cast<int>(round) && r.verified && r.duration_qualified) {
                if (r.cell == aid) ra = &r;
                if (r.cell == bid) rb = &r;
            }
        if (!ra || !rb) continue;
        if (!values.empty()) out << ',';
        const double ratio = ra->ns_per_operation / rb->ns_per_operation;
        values.push_back(ratio);
        if (ratio > 1) ++wins; else if (ratio < 1) ++losses; else ++ties;
        out << "{\"round\":" << round << ",\"numerator_region\":" << ra->region
            << ",\"denominator_region\":" << rb->region << ",\"ratio\":" << ratio << '}';
    }
    out << "],\"wins_denominator\":" << wins << ",\"losses_denominator\":" << losses
        << ",\"exact_ties\":" << ties << ",\"summary_ratio\":";
    summary(out,values); out << '}';
}
std::string json(const Options& options, const Corpus& corpus, const std::vector<Cell>& cells,
                 const std::vector<Record>& records, const std::vector<Preservation>& preservation,
                 const Order& base, const Power& start_power, const Power& end_power, std::int64_t wall_start_ns,
                 const std::string& status, const std::string& failure) {
    std::ostringstream out; out << std::setprecision(17) << std::boolalpha;
    out << "{\n\"schema\":\"parseatlas.p5.field_product_compare.v1\",\n\"status\":" << quote(status)
        << ",\n\"failure\":" << quote(failure) << ",\n\"seed\":" << options.seed
        << ",\n\"min_ms\":" << options.min_ms
        << ",\n\"floor_ns\":" << static_cast<std::uint64_t>(std::ceil(options.min_ms*1e6))
        << ",\n\"smoke\":" << options.smoke << ",\n\"reverse_order\":" << options.reverse
        << ",\n\"warmup_rounds_requested\":" << warmups << ",\n\"measurement_rounds_requested\":" << options.rounds
        << ",\n\"compiler\":" << quote(__VERSION__) << ",\n\"cplusplus\":" << __cplusplus
        << ",\n\"run_start_unix_ns\":" << wall_start_ns << ",\n\"harness_macros\":";
    write_macros(out);
    out << ",\n\"power_start\":"; write_power(out,start_power);
    out << ",\n\"power_end\":"; write_power(out,end_power);
    out << ",\n\"contract\":{"
        "\"scope\":\"canonical field multiply/square and arbitrary512 reducer schedules; no production, CT, novelty or upper-layer claim\","
        "\"work\":\"1024 canonical steps per active lane; chain1 or four independent one-core lanes; no lazy sum state\","
        "\"multiply\":\"x=mul(x,rhs[step&255]); canonical nonzero seeds/RHS; actual returning FE64 API is reference\","
        "\"square\":\"x=square(x); no RHS mixing; trace reports actual zero/one states\","
        "\"raw_reduce\":\"wide.low=x XOR corpus.low; wide.high=corpus.high; reduce to canonical x; arbitrary512 domain\","
        "\"raw_reference\":\"actual corrected FE64 pipeline canonicalizes each arbitrary256 half and computes high*K+low mod p; separate Boost/kernel qualification required\","
        "\"output\":\"fixed128bytes per job: active lanes in lane order, each BE32; chain1 has96inactive zero bytes; all128 observable and hashed\","
        "\"timing\":\"seed initialization,1024steps/lane,whole job call and final raw-to-FE/BE32 serialization plus zero padding included; allocation/generation/validation excluded\","
        "\"loop\":\"noinline/noipa whole jobs plus output memory barrier; no per-step benchmark barrier; route dispatch outside job loop\","
        "\"validation\":\"untimed each-step helper trace: canonical raw limbs and all32bytes vs FE64 reference; complete compiled evaluator and job outputs must match final trace; last identical job per region checked outside clock\","
        "\"trace_limit\":\"trace helpers are separate from complete compiled loops; hashes cover every active state BE32; trace is not Boost oracle or timed instrumentation\","
        "\"preservation\":\"regenerate all4seeds,256canonical RHS,256raw512 values after prevalidation and after regions/failure; compare every raw word and fullbyte hash\","
        "\"calibration\":\"retain all attempts; two consecutive>=floor regions at same count; grow ceil(1.5*target*jobs/elapsed), capped\","
        "\"region_duration\":\"warmup/measurement append whole batches under one continuous clock until floor; no discarded prefix/restart; use actual total jobs*1024*lanes\","
        "\"probes\":\"fixed preallocated storage; intermediate clock checks/stores/continuation are timed; terminal serialization and probe conversion outside clock\","
        "\"failure\":\"floor/extension/work failures prevent complete status and retain raw records; only verified duration-qualified measurements enter ratios\","
        "\"ordering\":\"seeded6group/4route-slot permutation plus round rotation/reversal; raw groups filter unused slots; second series reverses complete order\","
        "\"ratio\":\"same-round numerator ns/operation divided by denominator ns/operation; >1 favors denominator; chain/ILP comparison includes amortized complete-job costs\","
        "\"cpu\":\"one thread deliberately isolates dependency scheduling; pin externally; endpoint snapshots do not prove no migration or constant frequency; no power policy changes\","
        "\"backend\":\"external manifest binds exact kernel/driver/library/binary/compiler flags; harness macros alone do not prove library configuration\","
        "\"operation_cap_per_region\":" << operation_cap << ",\"calibration_attempt_limit\":" << max_attempts
        << ",\"extensions_per_later_region_limit\":" << max_extensions
        << ",\"max_probes_per_later_region\":" << max_extensions+1 << "},\n\"corpus\":{\"ready\":" << corpus.ready
        << ",\"recipe\":\"SplitMix64 state=seed xor0x6669656c645f7035; canonical candidates four RNGwords encoded BE32, strict reject>=p or zero; generate4seeds then256RHS; then256rawWide each8RNGwords in little-limb order\","
        "\"hash_recipe\":\"FNV1a64 unsigned wrap; seedBE32 thenRHSBE32 then eachrawWideBE64(highlimbfirst); fullbytes not lowwords\","
        "\"source_recipe_sha256\":\"PENDING IMPORT\",\"seed_count\":4,\"rhs_count\":" << corpus_count
        << ",\"raw512_count\":" << corpus_count << ",\"seed_bytes\":" << sizeof(corpus.seeds)
        << ",\"rhs_bytes\":" << sizeof(corpus.rhs) << ",\"raw512_bytes\":" << sizeof(corpus.raw)
        << ",\"checksum_fnv1a64\":" << quote(hex64(corpus.hash64)) << ",\"seeds_be32\":[";
    for (unsigned i=0;i<corpus.seeds.size();++i) { if(i)out<<',';out<<quote(hex(corpus.seeds[i].to_bytes())); }
    out << "]},\n\"input_preservation\":[";
    for (unsigned i=0;i<preservation.size();++i) {
        if(i)out<<',';
        const auto& p=preservation[i];
        out << "{\"phase\":" << quote(p.phase) << ",\"verified\":" << p.verified
            << ",\"exact_seed_checked\":" << p.exact_seed_checked << ",\"exact_rhs_checked\":" << p.exact_rhs_checked
            << ",\"exact_raw512_checked\":" << p.exact_raw512_checked << ",\"checksum_fnv1a64\":" << quote(hex64(p.hash64)) << '}';
    }
    out << "],\n\"base_group_order\":[";
    for(unsigned i=0;i<base.groups.size();++i){if(i)out<<',';out<<base.groups[i];}
    out << "],\n\"base_route_slot_order\":[";
    for(unsigned i=0;i<base.routes.size();++i){if(i)out<<',';out<<base.routes[i];}
    out << "],\n\"cells\":[\n";
    for (const auto& cell:cells) {
        if(cell.id)out<<",\n";
        std::vector<double> per_job,per_operation;unsigned short_count=0,extensions=0;
        for(const auto& r:records) if(r.cell==cell.id && r.phase=="measurement") {
            if(!r.duration_qualified)++short_count;
            extensions+=r.extensions;
            if(r.verified&&r.duration_qualified){per_job.push_back(r.ns_per_job);per_operation.push_back(r.ns_per_operation);}
        }
        out << "{\"id\":" << cell.id << ",\"group\":" << cell.group << ",\"route_index\":" << cell.route
            << ",\"route\":" << quote(routes[cell.route]) << ",\"operation\":" << quote(operations[cell.group/2])
            << ",\"mode\":" << quote(cell.lanes==1?"chain1":"ILP4") << ",\"lanes\":" << cell.lanes
            << ",\"steps_per_lane\":" << steps << ",\"operations_per_job\":" << cell.count
            << ",\"validated\":" << cell.validated << ",\"qualified\":" << cell.qualified
            << ",\"initial_batch_jobs\":" << cell.jobs << ",\"expected_final_be128\":" << quote(hex(cell.expected))
            << ",\"expected_checksum_fnv1a64\":" << quote(hex64(checksum(cell.expected)))
            << ",\"trace_state_checks\":" << cell.trace_checks << ",\"trace_checksum_fnv1a64\":" << quote(hex64(cell.trace_hash))
            << ",\"trace_zero_states\":" << cell.zero_states << ",\"trace_one_states\":" << cell.one_states
            << ",\"short_measurement_regions\":" << short_count << ",\"measurement_extensions\":" << extensions
            << ",\"summary_ns_per_job\":";summary(out,per_job);
        out << ",\"summary_ns_per_operation\":";summary(out,per_operation);out<<'}';
    }
    out << "\n],\n\"regions\":[\n";
    for (const auto& r : records) {
        if (r.region) out << ",\n";
        out << "{\"region\":" << r.region << ",\"phase\":" << quote(r.phase) << ",\"cell\":" << r.cell
            << ",\"round\":" << r.round << ",\"order_position\":" << r.position << ",\"attempt\":" << r.attempt
            << ",\"initial_batch_jobs\":" << r.initial_batch_jobs << ",\"jobs\":" << r.jobs << ",\"operations\":" << r.operations
            << ",\"begin_steady_ns\":" << r.begin_ns << ",\"end_steady_ns\":" << r.end_ns << ",\"elapsed_ns\":" << r.elapsed_ns
            << ",\"ns_per_job\":" << r.ns_per_job << ",\"ns_per_operation\":" << r.ns_per_operation
            << ",\"actual_final_be128\":" << quote(hex(r.output)) << ",\"checksum_fnv1a64\":" << quote(hex64(checksum(r.output)))
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
    bool first=true;
    for(unsigned g=0;g<6;++g) for(unsigned offset=1;offset<(g<4?4U:2U);++offset) {
        if(!first)out<<",\n";
        first=false;
        write_comparison(out,cells,group_start[g],group_start[g]+offset,options,records);
    }
    out << "\n],\n\"secondary_comparisons\":[\n";first=true;
    for(unsigned g=0;g<4;++g) for(unsigned offset=1;offset<3;++offset) {
        if(!first)out<<",\n";
        first=false;
        write_comparison(out,cells,group_start[g]+offset,group_start[g]+offset+1,options,records);
    }
    out << "\n],\n\"mode_comparisons\":[\n";first=true;
    for(unsigned g=0;g<6;g+=2) for(unsigned offset=0;offset<(g<4?4U:2U);++offset) {
        if(!first)out<<",\n";
        first=false;
        write_comparison(out,cells,group_start[g]+offset,group_start[g+1]+offset,options,records);
    }
    out << "\n]\n}\n";return out.str();
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
            std::cout << "Usage: p5_field_product_compare --output NEW.json [--seed UINT64] [--min-ms 0.1..1000] [--rounds EVEN_2..128] [--reverse-order] [--smoke]\n"
                "20cells: mul/square four routes and raw512 reduction two routes, chain1/ILP4. Default200ms, two warmups/eight measured rounds.\n"
                "Smoke defaults1ms/two measured rounds. Output never overwritten. Pin externally; no power policy changes.\n"
                "Exit0 complete; exit2 calibration/duration unqualified; exit1 error/mismatch.\n";
            return 0;
        }
        fd = ::open(options.output.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
        if (fd < 0) throw std::runtime_error(std::string("exclusive output open: ") + std::strerror(errno));
        const auto start_power = power();
        const auto wall_start = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        Corpus corpus; auto cells = make_cells();
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
        std::cout << "P5 " << status << " cells=" << cells.size() << " raw_regions=" << records.size() << " output=" << options.output << '\n';
        if (!failure.empty()) std::cerr << failure << '\n';
        return result;
    } catch (const std::exception& error) {
        if (fd >= 0) ::close(fd);
        std::cerr << "P5 error: " << error.what() << '\n'; return 1;
    }
}
