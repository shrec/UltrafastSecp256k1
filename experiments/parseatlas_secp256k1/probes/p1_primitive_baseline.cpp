// P1: linked public field/scalar API baselines. Identify source patches and
// library hashes in the external manifest. Compile and link without LTO.
#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
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
#include <type_traits>
#include <vector>
#include <sched.h>
#include <unistd.h>

#if defined(__GNUC__) && !defined(__clang__)
#define PA_JOB __attribute__((noinline, noipa))
#elif defined(__clang__)
#define PA_JOB __attribute__((noinline))
#else
#error "This audited baseline requires GCC or Clang compiler memory barriers"
#endif
#define PA_STRING_IMPL(x) #x
#define PA_STRING(x) PA_STRING_IMPL(x)

namespace {
using Field = secp256k1::fast::FieldElement;
using Scalar = secp256k1::fast::Scalar;
using Bytes = std::array<std::uint8_t, 32>;
using Output = std::array<Bytes, 4>;
using Clock = std::chrono::steady_clock;
constexpr std::size_t corpus_size = 256;
constexpr std::uint64_t steps_per_lane = 1024;
constexpr unsigned warmup_rounds = 2;
constexpr unsigned calibration_attempt_limit = 12;
constexpr std::uint64_t operation_cap = 1000000000;
inline void memory_barrier() { asm volatile("" ::: "memory"); }

struct SplitMix64 {
    std::uint64_t state;
    std::uint64_t next() {
        std::uint64_t z = (state += UINT64_C(0x9e3779b97f4a7c15));
        z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
        return z ^ (z >> 31);
    }
};

std::string quoted(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << static_cast<char>(c);
        else if (c < 32) out << "\\u" << std::hex << std::setw(4)
                             << std::setfill('0') << static_cast<unsigned>(c);
        else out << static_cast<char>(c);
    }
    out << '"';
    return out.str();
}
std::string hex_bytes(const Bytes& bytes) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (const auto byte : bytes) out << std::setw(2) << unsigned(byte);
    return out.str();
}
std::string hex64(std::uint64_t value) {
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << value;
    return out.str();
}
void hash_bytes(std::uint64_t& hash, const Bytes& bytes) {
    for (const auto byte : bytes) {
        hash ^= byte;
        hash *= UINT64_C(1099511628211);
    }
}
std::uint64_t output_checksum(const Output& output, unsigned lanes) {
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (unsigned lane = 0; lane < lanes; ++lane) hash_bytes(hash, output[lane]);
    return hash;
}

template<class T> struct Corpus {
    std::array<T, corpus_size> rhs;
    std::array<T, 4> seeds;
    std::uint64_t checksum = UINT64_C(14695981039346656037);
};
template<class T> T nonzero_canonical(SplitMix64& rng) {
    // Rejection sampling precedes timing. No truncation to a small subdomain.
    for (;;) {
        Bytes bytes{};
        for (unsigned word = 0; word < 4; ++word) {
            const auto value = rng.next();
            for (unsigned byte = 0; byte < 8; ++byte)
                bytes[word * 8 + byte] = static_cast<std::uint8_t>(value >> (56 - 8 * byte));
        }
        T value;
        if (!T::parse_bytes_strict(bytes, value)) continue;
        const auto& limbs = value.limbs();
        if ((limbs[0] | limbs[1] | limbs[2] | limbs[3]) != 0) return value;
    }
}
template<class T> Corpus<T> make_corpus(std::uint64_t seed) {
    Corpus<T> corpus;
    SplitMix64 rng{seed};
    for (auto& value : corpus.rhs) {
        value = nonzero_canonical<T>(rng);
        hash_bytes(corpus.checksum, value.to_bytes());
    }
    for (auto& value : corpus.seeds) {
        value = nonzero_canonical<T>(rng);
        hash_bytes(corpus.checksum, value.to_bytes());
    }
    return corpus;
}

enum class Operation { add, sub, mul, square, inverse };
const char* operation_name(Operation op) {
    switch (op) {
    case Operation::add: return "add";
    case Operation::sub: return "sub";
    case Operation::mul: return "mul";
    case Operation::square: return "square";
    case Operation::inverse: return "inverse";
    }
    throw std::logic_error("invalid operation");
}
template<class T, Operation Op> inline T arithmetic(const T& value, const T& rhs) {
    if constexpr (Op == Operation::add) return value + rhs;
    else if constexpr (Op == Operation::sub) return value - rhs;
    else if constexpr (Op == Operation::mul) return value * rhs;
    else if constexpr (Op == Operation::inverse) return rhs.inverse();
    else if constexpr (std::is_same_v<T, Field>) return value.square();
    else return value * value; // Scalar has no distinct square public API.
}
template<class T, Operation Op, unsigned Lanes>
PA_JOB void full_job(const void* opaque, Output& output) {
    const auto& corpus = *static_cast<const Corpus<T>*>(opaque);
    std::array<T, Lanes> states;
    for (unsigned lane = 0; lane < Lanes; ++lane) states[lane] = corpus.seeds[lane];
    // Sequential per lane is the measured dependency contract; ILP4 interleaves
    // four lanes on one core, never four threads or an aggregate reduction.
    for (std::uint64_t step = 0; step < steps_per_lane; ++step) {
        for (unsigned lane = 0; lane < Lanes; ++lane) {
            if constexpr (Op == Operation::inverse) {
                const auto index = (states[lane].limbs()[0] + step + lane) & 255;
                states[lane] = arithmetic<T, Op>(states[lane], corpus.rhs[index]);
            } else if constexpr (Op == Operation::square) {
                states[lane] = arithmetic<T, Op>(states[lane], states[lane]);
            } else {
                const auto index = (step + lane * 67U) & 255;
                states[lane] = arithmetic<T, Op>(states[lane], corpus.rhs[index]);
            }
        }
    }
    for (unsigned lane = 0; lane < Lanes; ++lane) output[lane] = states[lane].to_bytes();
    // Every job materializes all result bytes before the next job overwrites
    // output. No per-primitive barriers enter the dependency/ILP loops.
    asm volatile("" : : "m"(output) : "memory");
}
template<class T> Output replay(const Corpus<T>& corpus, Operation op, unsigned lanes) {
    Output output{};
    // Separate lane-major replay with explicit public APIs, not the timed job.
    // This checks scheduling, NOT correctness of production arithmetic itself.
    for (unsigned lane = 0; lane < lanes; ++lane) {
        T value = corpus.seeds[lane];
        for (std::uint64_t step = 0; step < steps_per_lane; ++step) {
            const auto index = (step + lane * 67U) & 255;
            switch (op) {
            case Operation::add: value = value + corpus.rhs[index]; break;
            case Operation::sub: value = value - corpus.rhs[index]; break;
            case Operation::mul: value = value * corpus.rhs[index]; break;
            case Operation::square:
                if constexpr (std::is_same_v<T, Field>) value = value.square();
                else value = value * value;
                break;
            case Operation::inverse: {
                const auto inverse_index = (value.limbs()[0] + step + lane) & 255;
                value = corpus.rhs[inverse_index].inverse();
                break;
            }
            }
        }
        output[lane] = value.to_bytes();
    }
    return output;
}
using Job = void (*)(const void*, Output&);
struct Cell {
    unsigned id;
    std::string type;
    Operation operation;
    unsigned lanes;
    const void* corpus;
    Job job;
    Output expected{};
    std::uint64_t jobs_per_region = 0;
    bool qualified = false;
};
template<class T, Operation Op>
void append_cells(std::vector<Cell>& cells, const char* name, const Corpus<T>& corpus) {
    cells.push_back({static_cast<unsigned>(cells.size()), name, Op, 1, &corpus,
                     full_job<T, Op, 1>, replay(corpus, Op, 1)});
    cells.push_back({static_cast<unsigned>(cells.size()), name, Op, 4, &corpus,
                     full_job<T, Op, 4>, replay(corpus, Op, 4)});
}
template<class T>
void append_group(std::vector<Cell>& cells, const char* name, const Corpus<T>& corpus) {
    append_cells<T, Operation::add>(cells, name, corpus);
    append_cells<T, Operation::sub>(cells, name, corpus);
    append_cells<T, Operation::mul>(cells, name, corpus);
    append_cells<T, Operation::square>(cells, name, corpus);
    append_cells<T, Operation::inverse>(cells, name, corpus);
}

struct Options {
    std::uint64_t seed = 20260905;
    double min_ms = 200.0;
    unsigned rounds = 8;
    bool smoke = false, help = false, reverse_order = false;
    std::string output;
};
std::uint64_t parse_unsigned(const std::string& value, const char* option) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error(std::string(option) + " requires unsigned decimal integer");
    std::size_t consumed = 0;
    const auto result = std::stoull(value, &consumed, 10);
    if (consumed != value.size()) throw std::runtime_error("invalid integer suffix");
    return result;
}
Options parse_options(int argc, char** argv) {
    Options options;
    bool explicit_floor = false, explicit_rounds = false;
    std::vector<std::string> seen;
    for (int arg = 1; arg < argc; ++arg) {
        const std::string flag = argv[arg];
        if (std::find(seen.begin(), seen.end(), flag) != seen.end())
            throw std::runtime_error("duplicate option: " + flag);
        seen.push_back(flag);
        if (flag == "--help") { options.help = true; continue; }
        if (flag == "--smoke") { options.smoke = true; continue; }
        if (flag == "--reverse-order") { options.reverse_order = true; continue; }
        if (flag != "--output" && flag != "--seed" && flag != "--min-ms" && flag != "--rounds")
            throw std::runtime_error("unknown option: " + flag);
        if (++arg == argc) throw std::runtime_error("missing value: " + flag);
        const std::string value = argv[arg];
        if (flag == "--output") options.output = value;
        else if (flag == "--seed") options.seed = parse_unsigned(value, "--seed");
        else if (flag == "--rounds") {
            const auto n = parse_unsigned(value, "--rounds");
            if (n < 2 || n > 128 || n % 2 != 0)
                throw std::runtime_error("--rounds must be even and in [2, 128]");
            options.rounds = static_cast<unsigned>(n);
            explicit_rounds = true;
        } else {
            std::size_t consumed = 0;
            options.min_ms = std::stod(value, &consumed);
            if (consumed != value.size() || !std::isfinite(options.min_ms) ||
                options.min_ms < 0.1 || options.min_ms > 1000.0)
                throw std::runtime_error("--min-ms must be finite and in [0.1, 1000]");
            explicit_floor = true;
        }
    }
    if (options.smoke) {
        if (!explicit_floor) options.min_ms = 1.0;
        if (!explicit_rounds) options.rounds = 2;
    }
    if (!options.help && options.output.empty()) throw std::runtime_error("--output is required");
    return options;
}

struct PowerSnapshot {
    int cpu;
    std::string governor, current_khz, no_turbo;
};
std::string read_sysfs(const std::string& path) {
    std::ifstream input(path);
    std::string value;
    if (!(input >> value)) return "unavailable";
    return value;
}
PowerSnapshot power_snapshot() {
    PowerSnapshot snapshot;
    snapshot.cpu = sched_getcpu();
    const auto prefix = "/sys/devices/system/cpu/cpu" + std::to_string(snapshot.cpu) + "/cpufreq/";
    snapshot.governor = read_sysfs(prefix + "scaling_governor");
    snapshot.current_khz = read_sysfs(prefix + "scaling_cur_freq");
    snapshot.no_turbo = read_sysfs("/sys/devices/system/cpu/intel_pstate/no_turbo");
    return snapshot;
}

struct Record {
    std::size_t region;
    std::string phase;
    unsigned cell;
    int round;
    unsigned order_position, attempt;
    std::uint64_t jobs, operations, elapsed_ns;
    double ns_per_op;
    std::uint64_t checksum;
    bool verified, floor_met;
    int cpu_start, cpu_end;
    unsigned qualification_streak = 0;
};
Record measure(const Cell& cell, const Options& options, const std::string& phase,
               int round, unsigned position, unsigned attempt, std::uint64_t jobs,
               std::size_t region) {
    const auto ops_per_job = steps_per_lane * cell.lanes;
    if (jobs == 0 || jobs > operation_cap / ops_per_job)
        throw std::runtime_error("region operation budget exceeded");
    Output output{};
    const int cpu_start = sched_getcpu();
    memory_barrier();
    const auto begin = Clock::now();
    for (std::uint64_t job = 0; job < jobs; ++job) cell.job(cell.corpus, output);
    const auto end = Clock::now();
    memory_barrier();
    const int cpu_end = sched_getcpu();
    const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    if (duration < 0) throw std::runtime_error("negative steady-clock interval");
    const auto elapsed = static_cast<std::uint64_t>(duration);
    bool verified = true;
    for (unsigned lane = 0; lane < cell.lanes; ++lane)
        verified = verified && output[lane] == cell.expected[lane];
    const auto operations = jobs * ops_per_job;
    return {region, phase, cell.id, round, position, attempt, jobs, operations,
            elapsed, static_cast<double>(elapsed) / static_cast<double>(operations),
            output_checksum(output, cell.lanes), verified,
            static_cast<double>(elapsed) >= options.min_ms * 1000000.0,
            cpu_start, cpu_end, 0};
}
bool calibrate(Cell& cell, const Options& options, unsigned position, std::vector<Record>& records) {
    std::uint64_t jobs = 1;
    const auto max_jobs = operation_cap / (steps_per_lane * cell.lanes);
    unsigned streak = 0;
    for (unsigned attempt = 0; attempt < calibration_attempt_limit; ++attempt) {
        auto record = measure(cell, options, "calibration", -1, position, attempt, jobs, records.size());
        streak = record.floor_met ? streak + 1 : 0;
        record.qualification_streak = streak;
        records.push_back(record);
        if (!record.verified) throw std::runtime_error("calibration full-byte mismatch");
        if (streak == 2) {
            cell.jobs_per_region = jobs;
            cell.qualified = true;
            return true;
        }
        if (!record.floor_met) {
            if (jobs == max_jobs) return false;
            const double elapsed = static_cast<double>(std::max<std::uint64_t>(1, record.elapsed_ns));
            const double proposed = std::ceil(1.5 * options.min_ms * 1000000.0 *
                                               static_cast<double>(jobs) / elapsed);
            jobs = proposed >= static_cast<double>(max_jobs) ? max_jobs :
                   std::max(jobs + 1, static_cast<std::uint64_t>(proposed));
        }
    }
    return false;
}
std::vector<unsigned> base_order(std::uint64_t seed, std::size_t count) {
    std::vector<unsigned> order(count);
    for (unsigned i = 0; i < count; ++i) order[i] = i;
    SplitMix64 rng{seed ^ UINT64_C(0x6f726465725f7031)};
    for (std::size_t n = count; n > 1; --n)
        std::swap(order[n - 1], order[static_cast<std::size_t>(rng.next() % n)]);
    return order;
}
std::vector<unsigned> round_order(const std::vector<unsigned>& base, unsigned round, bool reverse) {
    auto order = base;
    const auto shift = static_cast<std::ptrdiff_t>((round / 2) % order.size());
    std::rotate(order.begin(), order.begin() + shift, order.end());
    if ((round % 2 != 0) != reverse) std::reverse(order.begin(), order.end());
    return order;
}
double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const auto n = values.size();
    return n % 2 != 0 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2;
}
template<class T> void write_corpus(std::ostream& out, const Corpus<T>& corpus) {
    out << "{\"checksum_fnv1a64\":" << quoted(hex64(corpus.checksum)) << ",\"rhs_be32\":[";
    bool first = true;
    for (const auto& value : corpus.rhs) {
        if (!first) out << ',';
        first = false;
        out << quoted(hex_bytes(value.to_bytes()));
    }
    out << "],\"seeds_be32\":[";
    first = true;
    for (const auto& value : corpus.seeds) {
        if (!first) out << ',';
        first = false;
        out << quoted(hex_bytes(value.to_bytes()));
    }
    out << "]}";
}
void write_power(std::ostream& out, const PowerSnapshot& power) {
    out << "{\"cpu\":" << power.cpu << ",\"governor\":" << quoted(power.governor)
        << ",\"current_khz\":" << quoted(power.current_khz) << ",\"no_turbo\":" << quoted(power.no_turbo) << '}';
}
void write_macros(std::ostream& out) {
    out << '{';
#ifdef SECP256K1_HAS_ASM
    out << "\"SECP256K1_HAS_ASM\":" << quoted(PA_STRING(SECP256K1_HAS_ASM));
#else
    out << "\"SECP256K1_HAS_ASM\":null";
#endif
#ifdef SECP256K1_USE_FAST_REDUCTION
    out << ",\"SECP256K1_USE_FAST_REDUCTION\":" << quoted(PA_STRING(SECP256K1_USE_FAST_REDUCTION));
#else
    out << ",\"SECP256K1_USE_FAST_REDUCTION\":null";
#endif
#ifdef USE_INLINE_ASSEMBLY
    out << ",\"USE_INLINE_ASSEMBLY\":" << quoted(PA_STRING(USE_INLINE_ASSEMBLY));
#else
    out << ",\"USE_INLINE_ASSEMBLY\":null";
#endif
#ifdef SECP256K1_FAST_52BIT
    out << ",\"SECP256K1_FAST_52BIT\":" << quoted(PA_STRING(SECP256K1_FAST_52BIT));
#else
    out << ",\"SECP256K1_FAST_52BIT\":null";
#endif
#ifdef SECP256K1_FE52_COMPUTE
    out << ",\"SECP256K1_FE52_COMPUTE\":" << quoted(PA_STRING(SECP256K1_FE52_COMPUTE));
#else
    out << ",\"SECP256K1_FE52_COMPUTE\":null";
#endif
#ifdef SECP256K1_NO_INT128
    out << ",\"SECP256K1_NO_INT128\":" << quoted(PA_STRING(SECP256K1_NO_INT128));
#else
    out << ",\"SECP256K1_NO_INT128\":null";
#endif
#ifdef SECP256K1_NO_ASM
    out << ",\"SECP256K1_NO_ASM\":" << quoted(PA_STRING(SECP256K1_NO_ASM));
#else
    out << ",\"SECP256K1_NO_ASM\":null";
#endif
#ifdef __SIZEOF_INT128__
    out << ",\"__SIZEOF_INT128__\":" << quoted(PA_STRING(__SIZEOF_INT128__));
#else
    out << ",\"__SIZEOF_INT128__\":null";
#endif
    out << '}';
}
std::string make_json(const Options& options, const Corpus<Field>& field,
                      const Corpus<Scalar>& scalar, const std::vector<Cell>& cells,
                      const std::vector<Record>& records, const std::vector<unsigned>& order,
                      const PowerSnapshot& start_power, const PowerSnapshot& end_power,
                      const std::string& status, const std::string& failure) {
    std::ostringstream out;
    out << std::setprecision(17) << std::boolalpha;
    out << "{\n\"schema\":\"parseatlas.p1.primitive_baseline.v1\",\n\"status\":" << quoted(status)
        << ",\n\"failure\":" << quoted(failure) << ",\n\"seed\":" << options.seed
        << ",\n\"min_ms\":" << options.min_ms << ",\n\"smoke\":" << options.smoke
        << ",\n\"reverse_order\":" << options.reverse_order
        << ",\n\"warmup_rounds_requested\":" << warmup_rounds
        << ",\n\"measurement_rounds_requested\":" << options.rounds
        << ",\n\"compiler\":" << quoted(__VERSION__) << ",\n\"cplusplus\":" << __cplusplus
        << ",\n\"harness_macros\":";
    write_macros(out);
    out << ",\n\"power_start\":"; write_power(out, start_power);
    out << ",\n\"power_end\":"; write_power(out, end_power);
    out << ",\n\"contract\":{"
           "\"implementation\":\"linked secp256k1::fast public returning APIs; exact source patches and library hash are in the external manifest; separate library translation units, no LTO required\","
           "\"backend_metadata_scope\":\"harness macros only; library flags and generated config require external build provenance\","
           "\"timing_unit\":\"full repeated jobs; region elapsed divided by arithmetic operations, not individual-call latency quantiles\","
           "\"timed_costs\":\"arithmetic API calls, assignment, loop/index/RHS loads, per-job seed reset, full final 32-byte serialization per lane, job call and compiler barrier\","
           "\"untimed_costs\":\"corpus generation, replay, checksum and byte checking, JSON, CPU endpoint and sysfs queries\","
           "\"job_restart\":\"every job restarts identical seeds and corpus; 1024 steps per lane\","
           "\"verification\":\"every job materializes all active output bytes; only last identical job per timed region byte-compared outside timing; lane-major actual-API replay is not an independent production arithmetic oracle\","
           "\"inverse_dependency\":\"result_dependent_lookup: index=(previous_result.limbs()[0]+step+lane)&255; invert nonzero corpus[index]; includes lookup/address dependency, avoids direct inv(inv(x)) two-cycle\","
           "\"other_dependencies\":\"add/sub/mul evolve state with rhs[(step+67*lane)&255]; square evolves state; chain1 or four interleaved independent states on one core\","
           "\"scalar_square\":\"Scalar operator*(a,a), no separate public square API\","
           "\"threading\":\"one caller thread; sequential dependency lanes are the measurement contract, not a parallel-work omission\","
           "\"cpu_observation\":\"sched_getcpu before/after each region only; equal endpoints do not prove no migration; power snapshots are endpoints, not average frequency\","
           "\"calibration\":\"one job initially; two consecutive regions meeting floor at same count; short resets streak and grows count ceil(1.5*target*jobs/elapsed), bounded by arithmetic cap and attempts; all evidence retained\","
           "\"schedule\":\"seeded Fisher-Yates base permutation; paired forward/reverse rotated orders; reverse-order reverses every schedule and calibration order, not input; not a full Latin square or statistical independence claim\","
           "\"short_regions\":\"all retained, never silently rerun or discarded; floor is qualification target not censoring rule\","
           "\"scope_limits\":\"not CT assurance, upper-layer speedup, independent kernel correctness proof, cross-backend comparison or discovery claim\","
           "\"input_generation\":\"SplitMix64; four 64-bit outputs encoded big-endian; strict canonical parse and nonzero rejection; field seed xor 0x6669656c645f7031, scalar seed xor 0x7363616c61725f31\","
           "\"steps_per_lane\":" << steps_per_lane
        << ",\"rhs_count_per_group\":" << corpus_size
        << ",\"seeds_per_group\":4,\"operation_cap_per_region\":" << operation_cap
        << ",\"calibration_attempt_limit\":" << calibration_attempt_limit << "},\n\"corpora\":{\"field\":";
    write_corpus(out, field);
    out << ",\"scalar\":"; write_corpus(out, scalar);
    out << "},\n\"base_order\":[";
    for (std::size_t i = 0; i < order.size(); ++i) { if (i) out << ','; out << order[i]; }
    out << "],\n\"cells\":[\n";
    for (std::size_t i = 0; i < cells.size(); ++i) {
        const auto& cell = cells[i];
        if (i) out << ",\n";
        out << "{\"id\":" << cell.id << ",\"type\":" << quoted(cell.type)
            << ",\"operation\":" << quoted(operation_name(cell.operation))
            << ",\"mode\":" << quoted(cell.lanes == 1 ? "chain1" : "ilp4")
            << ",\"dependency\":" << quoted(cell.operation == Operation::inverse ? "result_dependent_lookup" : "state_recurrence")
            << ",\"lanes\":" << cell.lanes << ",\"operations_per_job\":" << steps_per_lane * cell.lanes
            << ",\"qualified\":" << cell.qualified << ",\"jobs_per_region\":" << cell.jobs_per_region
            << ",\"expected_checksum_fnv1a64\":" << quoted(hex64(output_checksum(cell.expected, cell.lanes)))
            << ",\"expected_final_be32\":[";
        for (unsigned lane = 0; lane < cell.lanes; ++lane) {
            if (lane) out << ',';
            out << quoted(hex_bytes(cell.expected[lane]));
        }
        out << "],\"summary_ns_per_op\":";
        std::vector<double> measured;
        unsigned shorts = 0;
        for (const auto& record : records) {
            if (record.phase == "measurement" && record.cell == cell.id) {
                measured.push_back(record.ns_per_op);
                if (!record.floor_met) ++shorts;
            }
        }
        if (measured.empty()) out << "null";
        else out << "{\"count\":" << measured.size()
                 << ",\"min\":" << *std::min_element(measured.begin(), measured.end())
                 << ",\"median\":" << median(measured)
                 << ",\"max\":" << *std::max_element(measured.begin(), measured.end())
                 << ",\"short_regions\":" << shorts << '}';
        out << '}';
    }
    out << "\n],\n\"regions\":[\n";
    for (std::size_t i = 0; i < records.size(); ++i) {
        const auto& record = records[i];
        if (i) out << ",\n";
        out << "{\"region\":" << record.region << ",\"phase\":" << quoted(record.phase)
            << ",\"cell\":" << record.cell << ",\"round\":" << record.round
            << ",\"order_position\":" << record.order_position << ",\"attempt\":" << record.attempt
            << ",\"jobs\":" << record.jobs << ",\"operations\":" << record.operations
            << ",\"elapsed_ns\":" << record.elapsed_ns << ",\"ns_per_op\":" << record.ns_per_op
            << ",\"checksum_fnv1a64\":" << quoted(hex64(record.checksum))
            << ",\"verified\":" << record.verified << ",\"floor_met\":" << record.floor_met
            << ",\"cpu_start\":" << record.cpu_start << ",\"cpu_end\":" << record.cpu_end
            << ",\"qualification_streak\":" << record.qualification_streak << '}';
    }
    out << "\n]\n}\n";
    return out.str();
}
void write_all(int fd, const std::string& text) {
    std::size_t written = 0;
    while (written < text.size()) {
        const auto n = ::write(fd, text.data() + written, text.size() - written);
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(std::string("output write: ") + std::strerror(errno));
        }
        if (n == 0) throw std::runtime_error("output write returned zero");
        written += static_cast<std::size_t>(n);
    }
    if (::fsync(fd) != 0) throw std::runtime_error(std::string("output fsync: ") + std::strerror(errno));
}
} // namespace

int main(int argc, char** argv) {
    int fd = -1;
    try {
        const auto options = parse_options(argc, argv);
        if (options.help) {
            std::cout << "Usage: p1_primitive_baseline --output NEW.json [--seed UINT64] [--min-ms 0.1..1000] [--rounds EVEN_2..128] [--smoke] [--reverse-order]\n"
                         "Default: seed 20260905, floor 200 ms, 2 warmup + 8 measured rounds, 1024 steps/lane.\n"
                         "--smoke defaults to 1 ms and 2 measured rounds unless those flags are explicit.\n"
                         "Existing output paths are never overwritten. Exit 0 complete, 2 unqualified, 1 error/mismatch.\n"
                         "Run pinned to one core; link manifest-identified CMake static library, identical backend macros, no LTO.\n";
            return 0;
        }
        fd = ::open(options.output.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
        if (fd < 0) throw std::runtime_error(std::string("exclusive output open: ") + std::strerror(errno));
        const auto start_power = power_snapshot();
        const auto field = make_corpus<Field>(options.seed ^ UINT64_C(0x6669656c645f7031));
        const auto scalar = make_corpus<Scalar>(options.seed ^ UINT64_C(0x7363616c61725f31));
        std::vector<Cell> cells;
        cells.reserve(20);
        append_group(cells, "field_fp", field);
        append_group(cells, "scalar_n", scalar);
        const auto order = base_order(options.seed, cells.size());
        auto calibration_order = order;
        if (options.reverse_order) std::reverse(calibration_order.begin(), calibration_order.end());
        std::vector<Record> records;
        records.reserve(cells.size() * (calibration_attempt_limit + warmup_rounds + options.rounds));
        std::string status = "complete", failure;
        int exit_code = 0;
        try {
            for (unsigned position = 0; position < calibration_order.size(); ++position) {
                auto& cell = cells[calibration_order[position]];
                if (!calibrate(cell, options, position, records)) {
                    status = "unqualified";
                    failure = "calibration exhausted for cell " + std::to_string(cell.id);
                    exit_code = 2;
                    break;
                }
            }
            if (exit_code == 0) {
                for (unsigned round = 0; round < warmup_rounds + options.rounds; ++round) {
                    const bool warmup = round < warmup_rounds;
                    const auto schedule = round_order(order, round, options.reverse_order);
                    for (unsigned position = 0; position < schedule.size(); ++position) {
                        const auto& cell = cells[schedule[position]];
                        records.push_back(measure(cell, options, warmup ? "warmup" : "measurement",
                            static_cast<int>(warmup ? round : round - warmup_rounds), position, 0,
                            cell.jobs_per_region, records.size()));
                        if (!records.back().verified) throw std::runtime_error("timed region full-byte mismatch");
                    }
                }
            }
        } catch (const std::exception& error) {
            status = "error";
            failure = error.what();
            exit_code = 1;
        }
        const auto end_power = power_snapshot();
        write_all(fd, make_json(options, field, scalar, cells, records, order,
                               start_power, end_power, status, failure));
        if (::close(fd) != 0) { fd = -1; throw std::runtime_error("output close failed"); }
        fd = -1;
        std::cout << "P1 " << status << ": cells=" << cells.size() << " raw_regions=" << records.size()
                  << " output=" << options.output << '\n';
        if (!failure.empty()) std::cerr << failure << '\n';
        return exit_code;
    } catch (const std::exception& error) {
        if (fd >= 0) ::close(fd);
        std::cerr << "P1 error: " << error.what() << '\n';
        return 1;
    }
}
