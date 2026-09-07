// P2 field-only representation experiment. No production implementation changes.
// Link the manager-verified corrected library without LTO. Macro/build provenance
// belongs in the external manifest as well as this harness's metadata.
#include "secp256k1/field.hpp"
#include "secp256k1/field_52.hpp"
#include "secp256k1/ct/field.hpp"
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
#error "P2 requires GCC or Clang compiler memory barriers"
#endif
#define PA_STRING_IMPL(x) #x
#define PA_STRING(x) PA_STRING_IMPL(x)

namespace {
using FE = secp256k1::fast::FieldElement;
using FE52 = secp256k1::fast::FieldElement52;
namespace ct = secp256k1::ct;
using Bytes = std::array<std::uint8_t, 32>;
using Output = std::array<Bytes, 4>;
using Clock = std::chrono::steady_clock;
constexpr std::uint64_t steps = 1024;
constexpr unsigned rhs_count = 256, warmups = 2, max_attempts = 12;
constexpr std::uint64_t operation_cap = 1000000000;
constexpr std::uint64_t fnv_basis = UINT64_C(14695981039346656037);
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
    std::ostringstream out;
    out << '"';
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') out << '\\' << static_cast<char>(c);
        else if (c < 32) out << "\\u" << std::hex << std::setfill('0')
                             << std::setw(4) << static_cast<unsigned>(c);
        else out << static_cast<char>(c);
    }
    out << '"';
    return out.str();
}
std::string hex(const Bytes& bytes) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (auto byte : bytes) out << std::setw(2) << unsigned(byte);
    return out.str();
}
std::string hex64(std::uint64_t value) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << value;
    return out.str();
}
void hash_bytes(std::uint64_t& hash, const Bytes& bytes) {
    for (auto byte : bytes) { hash ^= byte; hash *= UINT64_C(1099511628211); }
}
std::uint64_t checksum(const Output& output, unsigned lanes) {
    auto hash = fnv_basis;
    for (unsigned lane = 0; lane < lanes; ++lane) hash_bytes(hash, output[lane]);
    return hash;
}
bool canonical64(const FE& value) {
    constexpr std::array<std::uint64_t, 4> p{
        UINT64_C(0xfffffffefffffc2f), UINT64_MAX, UINT64_MAX, UINT64_MAX};
    const auto& limbs = value.limbs();
    for (int i = 3; i >= 0; --i) {
        if (limbs[static_cast<unsigned>(i)] != p[static_cast<unsigned>(i)])
            return limbs[static_cast<unsigned>(i)] < p[static_cast<unsigned>(i)];
    }
    return false;
}
bool canonical52(const FE52& value) {
    using namespace secp256k1::fast::fe52_constants;
    constexpr std::array<std::uint64_t, 5> p{P0, P1, P2, P3, P4};
    for (unsigned i = 0; i < 4; ++i) if (value.n[i] > M52) return false;
    if (value.n[4] > M48) return false;
    for (int i = 4; i >= 0; --i) {
        if (value.n[i] != p[static_cast<unsigned>(i)])
            return value.n[i] < p[static_cast<unsigned>(i)];
    }
    return false;
}
struct Corpus {
    std::array<FE, rhs_count> rhs;
    std::array<FE, 4> seeds;
    std::array<FE52, rhs_count> resident_rhs;
    std::uint64_t hash = fnv_basis, setup_conversion_ns = 0;
};
FE random_nonzero(RNG& rng) {
    for (;;) {
        Bytes bytes{};
        for (unsigned w = 0; w < 4; ++w) {
            const auto word = rng.next();
            for (unsigned b = 0; b < 8; ++b)
                bytes[w * 8 + b] = static_cast<std::uint8_t>(word >> (56 - 8 * b));
        }
        FE value;
        if (!FE::parse_bytes_strict(bytes, value)) continue;
        const auto& a = value.limbs();
        if ((a[0] | a[1] | a[2] | a[3]) != 0) return value;
    }
}
Corpus make_corpus(std::uint64_t seed) {
    Corpus corpus;
    RNG rng{seed ^ UINT64_C(0x6669656c645f7031)};
    for (auto& value : corpus.rhs) {
        value = random_nonzero(rng);
        hash_bytes(corpus.hash, value.to_bytes());
    }
    for (auto& value : corpus.seeds) {
        value = random_nonzero(rng);
        hash_bytes(corpus.hash, value.to_bytes());
    }
    barrier();
    const auto begin = Clock::now();
    for (unsigned i = 0; i < rhs_count; ++i)
        corpus.resident_rhs[i] = FE52::from_fe(corpus.rhs[i]);
    asm volatile("" : : "m"(corpus.resident_rhs) : "memory");
    const auto end = Clock::now();
    corpus.setup_conversion_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
    for (unsigned i = 0; i < rhs_count; ++i) {
        if (!canonical52(corpus.resident_rhs[i]) ||
            corpus.resident_rhs[i].to_fe().to_bytes() != corpus.rhs[i].to_bytes())
            throw std::runtime_error("resident corpus conversion mismatch");
    }
    return corpus;
}

enum class Op { add, sub, mul, square, inverse };
enum class Route { fe64, fe52_bridge, fe52_resident, ct64 };
const char* op_name(Op op) {
    switch (op) {
    case Op::add: return "add";
    case Op::sub: return "sub";
    case Op::mul: return "mul";
    case Op::square: return "square";
    case Op::inverse: return "inverse";
    }
    throw std::logic_error("invalid op");
}
const char* route_name(Route route) {
    switch (route) {
    case Route::fe64: return "fe64_api";
    case Route::fe52_bridge: return "fe52_canonical_bridge";
    case Route::fe52_resident: return "fe52_canonical_resident";
    case Route::ct64: return "ct64_api_security_baseline";
    }
    throw std::logic_error("invalid route");
}
template<Op O> inline FE apply64(const FE& a, const FE& b) {
    if constexpr (O == Op::add) return a + b;
    else if constexpr (O == Op::sub) return a - b;
    else if constexpr (O == Op::mul) return a * b;
    else if constexpr (O == Op::square) return a.square();
    else return b.inverse();
}
template<Op O> inline FE apply_ct(const FE& a, const FE& b) {
    if constexpr (O == Op::add) return ct::field_add(a, b);
    else if constexpr (O == Op::sub) return ct::field_sub(a, b);
    else if constexpr (O == Op::mul) return ct::field_mul(a, b);
    else if constexpr (O == Op::square) return ct::field_sqr(a);
    else return ct::field_inv(b);
}
template<Op O> inline FE52 raw52(const FE52& a, const FE52& b) {
    FE52 result;
    if constexpr (O == Op::add) result = a + b;
    else if constexpr (O == Op::sub) result = a + b.negate(1);
    else if constexpr (O == Op::mul) result = a * b;
    else if constexpr (O == Op::square) result = a.square();
    else result = b.inverse_safegcd();
    return result;
}
template<Op O> inline FE52 apply52(const FE52& a, const FE52& b) {
    auto result = raw52<O>(a, b);
    // The canonical-resident contract is stronger than raw kernel magnitude 1.
    // Every output is normalized, including multiplication and squaring.
    result.normalize();
    return result;
}
template<Op O> PA_BOUNDARY FE bridge(const FE& a, const FE& b) {
    // A deliberate per-operation FE64 ABI boundary prevents the compiler from
    // retaining FE52 across calls and silently turning bridge into resident.
    // Existing FE64/CT APIs are also out-of-line calls in this no-LTO build.
    if constexpr (O == Op::inverse) {
        const auto input = FE52::from_fe(b);
        return raw52<O>(input, input).to_fe();
    } else if constexpr (O == Op::square) {
        const auto input = FE52::from_fe(a);
        return raw52<O>(input, input).to_fe();
    } else {
        return raw52<O>(FE52::from_fe(a), FE52::from_fe(b)).to_fe();
    }
}
template<Op O, Route R, unsigned Lanes>
PA_BOUNDARY void full_job(const Corpus& corpus, Output& output) {
    // One core by contract: interleave independent lanes, do not create threads.
    if constexpr (R == Route::fe52_resident) {
        std::array<FE52, Lanes> state;
        for (unsigned lane = 0; lane < Lanes; ++lane)
            state[lane] = FE52::from_fe(corpus.seeds[lane]);
        for (std::uint64_t step = 0; step < steps; ++step) {
            for (unsigned lane = 0; lane < Lanes; ++lane) {
                std::uint64_t index;
                if constexpr (O == Op::inverse) index = (state[lane].n[0] + step + lane) & 255;
                else index = (step + lane * 67U) & 255;
                state[lane] = apply52<O>(state[lane], corpus.resident_rhs[index]);
            }
        }
        for (unsigned lane = 0; lane < Lanes; ++lane)
            output[lane] = state[lane].to_fe().to_bytes();
    } else {
        std::array<FE, Lanes> state;
        for (unsigned lane = 0; lane < Lanes; ++lane) state[lane] = corpus.seeds[lane];
        for (std::uint64_t step = 0; step < steps; ++step) {
            for (unsigned lane = 0; lane < Lanes; ++lane) {
                std::uint64_t index;
                if constexpr (O == Op::inverse) index = (state[lane].limbs()[0] + step + lane) & 255;
                else index = (step + lane * 67U) & 255;
                if constexpr (R == Route::fe64)
                    state[lane] = apply64<O>(state[lane], corpus.rhs[index]);
                else if constexpr (R == Route::fe52_bridge)
                    state[lane] = bridge<O>(state[lane], corpus.rhs[index]);
                else state[lane] = apply_ct<O>(state[lane], corpus.rhs[index]);
            }
        }
        for (unsigned lane = 0; lane < Lanes; ++lane) output[lane] = state[lane].to_bytes();
    }
    // Every job materializes all active 32-byte results, no per-primitive barrier.
    asm volatile("" : : "m"(output) : "memory");
}

using Job = void (*)(const Corpus&, Output&);
struct Cell {
    unsigned id, group;
    Op op;
    Route route;
    unsigned lanes;
    Job job;
    Output expected{};
    std::uint64_t validation_steps = 0, jobs = 0;
    bool qualified = false;
};
template<Op O, unsigned L>
void append_group(std::vector<Cell>& cells) {
    const auto group = static_cast<unsigned>(cells.size() / 4);
    cells.push_back({static_cast<unsigned>(cells.size()), group, O, Route::fe64, L,
                    full_job<O, Route::fe64, L>});
    cells.push_back({static_cast<unsigned>(cells.size()), group, O, Route::fe52_bridge, L,
                    full_job<O, Route::fe52_bridge, L>});
    cells.push_back({static_cast<unsigned>(cells.size()), group, O, Route::fe52_resident, L,
                    full_job<O, Route::fe52_resident, L>});
    cells.push_back({static_cast<unsigned>(cells.size()), group, O, Route::ct64, L,
                    full_job<O, Route::ct64, L>});
}
template<Op O> void append_op(std::vector<Cell>& cells) {
    append_group<O, 1>(cells); append_group<O, 4>(cells);
}
template<Op O> FE route64(Route route, const FE& a, const FE& b) {
    if (route == Route::fe64) return apply64<O>(a, b);
    if (route == Route::fe52_bridge) return bridge<O>(a, b);
    return apply_ct<O>(a, b);
}
FE reference(Op op, const FE& a, const FE& b) {
    switch (op) {
    case Op::add: return a + b;
    case Op::sub: return a - b;
    case Op::mul: return a * b;
    case Op::square: return a.square();
    case Op::inverse: return b.inverse();
    }
    throw std::logic_error("invalid reference op");
}
FE candidate64(Op op, Route route, const FE& a, const FE& b) {
    switch (op) {
    case Op::add: return route64<Op::add>(route, a, b);
    case Op::sub: return route64<Op::sub>(route, a, b);
    case Op::mul: return route64<Op::mul>(route, a, b);
    case Op::square: return route64<Op::square>(route, a, b);
    case Op::inverse: return route64<Op::inverse>(route, a, b);
    }
    throw std::logic_error("invalid candidate op");
}
FE52 candidate52(Op op, const FE52& a, const FE52& b) {
    switch (op) {
    case Op::add: return apply52<Op::add>(a, b);
    case Op::sub: return apply52<Op::sub>(a, b);
    case Op::mul: return apply52<Op::mul>(a, b);
    case Op::square: return apply52<Op::square>(a, b);
    case Op::inverse: return apply52<Op::inverse>(a, b);
    }
    throw std::logic_error("invalid candidate52 op");
}
void validate(Cell& cell, const Corpus& corpus) {
    // Lane-major replay differs from timed step-major scheduling. All intermediate
    // outputs are checked here; timed regions check only their last identical job.
    // FE64 reference is inspected raw, never normalized to conceal invalid output.
    for (unsigned lane = 0; lane < cell.lanes; ++lane) {
        FE expected = corpus.seeds[lane], actual = expected;
        FE52 resident = FE52::from_fe(expected);
        for (std::uint64_t step = 0; step < steps; ++step) {
            const auto regular_index = (step + lane * 67U) & 255;
            const auto expected_index = cell.op == Op::inverse ?
                (expected.limbs()[0] + step + lane) & 255 : regular_index;
            const auto actual_low = cell.route == Route::fe52_resident ? resident.n[0] : actual.limbs()[0];
            const auto actual_index = cell.op == Op::inverse ?
                (actual_low + step + lane) & 255 : regular_index;
            if (actual_index != expected_index) throw std::runtime_error("trajectory index mismatch");
            expected = reference(cell.op, expected, corpus.rhs[expected_index]);
            if (!canonical64(expected)) throw std::runtime_error("noncanonical raw FE64 reference");
            if (cell.route == Route::fe52_resident) {
                resident = candidate52(cell.op, resident, corpus.resident_rhs[actual_index]);
                // Check raw resident limbs BEFORE to_fe() normalizes its copy.
                if (!canonical52(resident)) throw std::runtime_error("noncanonical resident FE52 output");
                actual = resident.to_fe();
            } else actual = candidate64(cell.op, cell.route, actual, corpus.rhs[actual_index]);
            if (!canonical64(actual) || actual.to_bytes() != expected.to_bytes())
                throw std::runtime_error("byte mismatch cell=" + std::to_string(cell.id) +
                    " lane=" + std::to_string(lane) + " step=" + std::to_string(step) +
                    " expected=" + hex(expected.to_bytes()) + " actual=" + hex(actual.to_bytes()));
            ++cell.validation_steps;
        }
        cell.expected[lane] = expected.to_bytes();
    }
    Output timed_layout{};
    cell.job(corpus, timed_layout); // Untimed scheduling check; not a clock region.
    for (unsigned lane = 0; lane < cell.lanes; ++lane)
        if (timed_layout[lane] != cell.expected[lane])
            throw std::runtime_error("untimed full-job scheduling mismatch");
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
        throw std::runtime_error(flag + " requires unsigned decimal integer");
    std::size_t used = 0;
    const auto result = std::stoull(value, &used);
    if (used != value.size()) throw std::runtime_error("invalid integer suffix");
    return result;
}
Options parse_options(int argc, char** argv) {
    Options options;
    bool explicit_ms = false, explicit_rounds = false;
    std::vector<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (std::find(seen.begin(), seen.end(), flag) != seen.end())
            throw std::runtime_error("duplicate option: " + flag);
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
            if (rounds < 2 || rounds > 128 || rounds % 2)
                throw std::runtime_error("--rounds must be even, in [2,128]");
            options.rounds = static_cast<unsigned>(rounds); explicit_rounds = true;
        } else {
            std::size_t used = 0;
            options.min_ms = std::stod(value, &used);
            if (used != value.size() || !std::isfinite(options.min_ms) ||
                options.min_ms < 0.1 || options.min_ms > 1000)
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
    const auto prefix = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cpufreq/";
    return {cpu, sysfs(prefix + "scaling_governor"), sysfs(prefix + "scaling_cur_freq"),
            sysfs("/sys/devices/system/cpu/intel_pstate/no_turbo")};
}
struct Record {
    std::size_t region;
    std::string phase;
    unsigned cell;
    int round;
    unsigned position, attempt;
    std::uint64_t jobs, operations, elapsed_ns;
    double ns_per_op;
    std::uint64_t hash;
    bool verified, floor_met;
    int cpu_start, cpu_end;
    unsigned streak = 0;
};
Record measure(const Cell& cell, const Corpus& corpus, const Options& options,
               const std::string& phase, int round, unsigned position, unsigned attempt,
               std::uint64_t jobs, std::size_t region) {
    const auto ops_per_job = steps * cell.lanes;
    if (!jobs || jobs > operation_cap / ops_per_job) throw std::runtime_error("operation cap exceeded");
    Output output{};
    const auto cpu_start = sched_getcpu();
    barrier();
    const auto begin = Clock::now();
    for (std::uint64_t job = 0; job < jobs; ++job) cell.job(corpus, output);
    const auto end = Clock::now();
    barrier();
    const auto cpu_end = sched_getcpu();
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    if (elapsed < 0) throw std::runtime_error("negative steady clock interval");
    bool verified = true;
    for (unsigned lane = 0; lane < cell.lanes; ++lane)
        verified = (output[lane] == cell.expected[lane]) && verified;
    const auto operations = jobs * ops_per_job;
    return {region, phase, cell.id, round, position, attempt, jobs, operations,
            static_cast<std::uint64_t>(elapsed), static_cast<double>(elapsed) / static_cast<double>(operations),
            checksum(output, cell.lanes), verified, static_cast<double>(elapsed) >= options.min_ms * 1e6,
            cpu_start, cpu_end, 0};
}
bool calibrate(Cell& cell, const Corpus& corpus, const Options& options,
               unsigned position, std::vector<Record>& records) {
    std::uint64_t jobs = 1;
    const auto max_jobs = operation_cap / (steps * cell.lanes);
    unsigned streak = 0;
    for (unsigned attempt = 0; attempt < max_attempts; ++attempt) {
        auto record = measure(cell, corpus, options, "calibration", -1, position, attempt, jobs, records.size());
        streak = record.floor_met ? streak + 1 : 0;
        record.streak = streak;
        records.push_back(record);
        if (!record.verified) throw std::runtime_error("calibration final-byte mismatch");
        if (streak == 2) { cell.jobs = jobs; cell.qualified = true; return true; }
        if (!record.floor_met) {
            if (jobs == max_jobs) return false;
            const double proposed = std::ceil(1.5 * options.min_ms * 1e6 * static_cast<double>(jobs) /
                static_cast<double>(std::max<std::uint64_t>(1, record.elapsed_ns)));
            jobs = proposed >= static_cast<double>(max_jobs) ? max_jobs :
                std::max(jobs + 1, static_cast<std::uint64_t>(proposed));
        }
    }
    return false;
}
std::vector<unsigned> base_order(std::uint64_t seed) {
    std::vector<unsigned> groups(10);
    for (unsigned i = 0; i < 10; ++i) groups[i] = i;
    RNG rng{seed ^ UINT64_C(0x6f726465725f7032)};
    for (unsigned n = 10; n > 1; --n) std::swap(groups[n - 1], groups[rng.next() % n]);
    return groups;
}
std::vector<unsigned> schedule(const std::vector<unsigned>& base, unsigned round, bool reverse) {
    auto groups = base;
    std::rotate(groups.begin(), groups.begin() + (round / 2) % groups.size(), groups.end());
    if (round % 2) std::reverse(groups.begin(), groups.end());
    std::vector<unsigned> order;
    order.reserve(40);
    for (const auto group : groups) {
        std::array<unsigned, 4> routes{0, 1, 2, 3};
        std::rotate(routes.begin(), routes.begin() + (round / 2 + group) % 4, routes.end());
        if (round % 2) std::reverse(routes.begin(), routes.end());
        for (const auto route : routes) order.push_back(group * 4 + route);
    }
    if (reverse) std::reverse(order.begin(), order.end());
    return order;
}
double median(std::vector<double> data) {
    std::sort(data.begin(), data.end());
    const auto n = data.size();
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
#ifdef SECP256K1_FAST_52BIT
    out << ",\"SECP256K1_FAST_52BIT\":" << quote(PA_STRING(SECP256K1_FAST_52BIT));
#else
    out << ",\"SECP256K1_FAST_52BIT\":null";
#endif
#ifdef SECP256K1_FE52_COMPUTE
    out << ",\"SECP256K1_FE52_COMPUTE\":" << quote(PA_STRING(SECP256K1_FE52_COMPUTE));
#else
    out << ",\"SECP256K1_FE52_COMPUTE\":null";
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
#ifdef SECP256K1_HYBRID_4X64_ACTIVE
    out << ",\"SECP256K1_HYBRID_4X64_ACTIVE\":" << quote(PA_STRING(SECP256K1_HYBRID_4X64_ACTIVE));
#else
    out << ",\"SECP256K1_HYBRID_4X64_ACTIVE\":null";
#endif
    out << '}';
}
std::string json(const Options& options, const Corpus& corpus, const std::vector<Cell>& cells,
                 const std::vector<Record>& records, const std::vector<unsigned>& base,
                 const Power& start_power, const Power& end_power,
                 const std::string& status, const std::string& failure) {
    std::ostringstream out;
    out << std::setprecision(17) << std::boolalpha;
    out << "{\n\"schema\":\"parseatlas.p2.field_compare.v1\",\n\"status\":" << quote(status)
        << ",\n\"failure\":" << quote(failure) << ",\n\"seed\":" << options.seed
        << ",\n\"min_ms\":" << options.min_ms << ",\n\"smoke\":" << options.smoke
        << ",\n\"reverse_order\":" << options.reverse << ",\n\"warmup_rounds_requested\":" << warmups
        << ",\n\"measurement_rounds_requested\":" << options.rounds
        << ",\n\"compiler\":" << quote(__VERSION__) << ",\n\"cplusplus\":" << __cplusplus
        << ",\n\"harness_macros\":";
    write_macros(out);
    out << ",\n\"power_start\":"; write_power(out, start_power);
    out << ",\n\"power_end\":"; write_power(out, end_power);
    out << ",\n\"contract\":{"
        "\"scope\":\"field Fp only; existing FE64, FE52 and CT APIs; no production changes or scalar/upper-layer claims\","
        "\"reference\":\"manager-verified corrected FE64 library; raw canonical limbs checked before serialization; actual-API replay is not an independent arithmetic oracle\","
        "\"backend_provenance\":\"harness macros are not proof of library flags; record external source/binary hashes and no-LTO build commands\","
        "\"bridge\":\"one noinline/noipa per-operation FE64 ABI wrapper; from_fe on each used operand, raw FE52 arithmetic, to_fe full normalization and canonical return; no redundant explicit normalize before to_fe; conversion and call costs included\","
        "\"resident\":\"preconverted 256-entry FE52 RHS table excluded from repeated regions; per-job seed conversions and final to_fe/to_bytes included; every add/sub/mul/square/inverse output explicitly normalized; no lazy schedule\","
        "\"resident_amortization\":\"resident-region contract, not per-call bridge or automatic upper-layer gain; one recorded RHS setup duration is descriptive, not a benchmark result\","
        "\"subtraction\":\"canonical FE52 a+b.negate(1), then normalize; neither input is a lazy higher-magnitude operand\","
        "\"inverse\":\"FE64 inverse versus FE52 inverse_safegcd direct 5x52/signed62 SafeGCD route; CT field_inv separately labeled; header comments do not determine implementation\","
        "\"inverse_definedness\":\"only nonzero runtime corpus inputs; FE64 zero throws while FE52 SafeGCD returns zero; zero-domain contracts are not interchangeable\","
        "\"dependency\":\"chain1 or four independent interleaved states on one core; add/sub/mul rhs[(step+67*lane)&255], square repeated state squaring\","
        "\"inverse_dependency\":\"public benchmark result-dependent lookup: (low64(previous_result)+step+lane)&255; normalized FE52 low8 equals n[0]&255; includes address/load dependency, avoids direct inverse two-cycle\","
        "\"security\":\"CT API is an existing security baseline, not a new CT proof; lookup-driven benchmark is not CT; no secret-dependent runtime dispatch and no VT/CT substitutability claim\","
        "\"timing\":\"full jobs, arithmetic calls/inline kernels, assignments, loop/index/RHS loads, per-job reset, final full32 bytes per lane, full-job call and compiler barrier included; no per-primitive barriers except deliberate bridge/API call boundaries\","
        "\"timing_unit\":\"region elapsed divided by counted arithmetic operations; region-mean ns/op, not individual operation latency quantiles\","
        "\"verification\":\"every timed job materializes all active output bytes; last identical job per region byte-checked outside timing; separate untimed replay checks every intermediate value and raw normalized resident limb bounds\","
        "\"calibration\":\"start one job; require two consecutive >=min_ms regions at same count; short resets streak and grows ceil(1.5*target*jobs/elapsed), capped by operations/attempts; unqualified exits2 before warmup\","
        "\"ordering\":\"same-process groups pair same operation and lane count; seeded group permutation, rotated paired reverse orders and rotating/reversing four routes; reverse-order reverses all schedules without changing corpus\","
        "\"statistics\":\"all calibration/warmup/measurement regions retained including shorts; paired values are same-round normalized region means, not matched job counts or independence proof\","
        "\"cpu\":\"one caller thread; endpoints only, equal sched_getcpu endpoints do not prove no migration; sysfs start/end frequency is not a time-average; policy is never changed\","
        "\"input_generation\":\"same P1 field SplitMix64 sequence: seed xor0x6669656c645f7031, four words encoded big-endian, strict parse/nonzero rejection; full corpus recorded\","
        "\"steps_per_lane\":" << steps << ",\"operation_cap_per_region\":" << operation_cap
        << ",\"calibration_attempt_limit\":" << max_attempts << "},\n\"corpus\":{"
        "\"checksum_fnv1a64\":" << quote(hex64(corpus.hash))
        << ",\"rhs64_bytes\":" << sizeof(corpus.rhs) << ",\"resident_rhs52_bytes\":" << sizeof(corpus.resident_rhs)
        << ",\"seed64_bytes\":" << sizeof(corpus.seeds)
        << ",\"resident_setup_one_observation_ns\":" << corpus.setup_conversion_ns << ",\"rhs_be32\":[";
    for (unsigned i = 0; i < rhs_count; ++i) { if (i) out << ','; out << quote(hex(corpus.rhs[i].to_bytes())); }
    out << "],\"seeds_be32\":[";
    for (unsigned i = 0; i < 4; ++i) { if (i) out << ','; out << quote(hex(corpus.seeds[i].to_bytes())); }
    out << "]},\n\"base_group_order\":[";
    for (unsigned i = 0; i < base.size(); ++i) { if (i) out << ','; out << base[i]; }
    out << "],\n\"cells\":[\n";
    for (unsigned i = 0; i < cells.size(); ++i) {
        const auto& cell = cells[i];
        if (i) out << ",\n";
        out << "{\"id\":" << cell.id << ",\"group\":" << cell.group
            << ",\"operation\":" << quote(op_name(cell.op)) << ",\"route\":" << quote(route_name(cell.route))
            << ",\"mode\":" << quote(cell.lanes == 1 ? "chain1" : "ilp4") << ",\"lanes\":" << cell.lanes
            << ",\"operations_per_job\":" << steps * cell.lanes
            << ",\"validation_intermediate_steps\":" << cell.validation_steps
            << ",\"qualified\":" << cell.qualified << ",\"jobs_per_region\":" << cell.jobs
            << ",\"expected_checksum_fnv1a64\":" << quote(hex64(checksum(cell.expected, cell.lanes)))
            << ",\"expected_final_be32\":[";
        for (unsigned lane = 0; lane < cell.lanes; ++lane) { if (lane) out << ','; out << quote(hex(cell.expected[lane])); }
        out << "],\"summary_ns_per_op\":";
        std::vector<double> values;
        unsigned short_count = 0;
        for (const auto& record : records) if (record.cell == cell.id && record.phase == "measurement") {
            values.push_back(record.ns_per_op);
            if (!record.floor_met) ++short_count;
        }
        if (values.empty()) out << "null";
        else out << "{\"count\":" << values.size() << ",\"min\":" << *std::min_element(values.begin(), values.end())
                 << ",\"median\":" << median(values) << ",\"max\":" << *std::max_element(values.begin(), values.end())
                 << ",\"short_regions\":" << short_count << '}';
        out << '}';
    }
    out << "\n],\n\"regions\":[\n";
    for (unsigned i = 0; i < records.size(); ++i) {
        const auto& r = records[i];
        if (i) out << ",\n";
        out << "{\"region\":" << r.region << ",\"phase\":" << quote(r.phase) << ",\"cell\":" << r.cell
            << ",\"round\":" << r.round << ",\"order_position\":" << r.position << ",\"attempt\":" << r.attempt
            << ",\"jobs\":" << r.jobs << ",\"operations\":" << r.operations << ",\"elapsed_ns\":" << r.elapsed_ns
            << ",\"ns_per_op\":" << r.ns_per_op << ",\"checksum_fnv1a64\":" << quote(hex64(r.hash))
            << ",\"verified\":" << r.verified << ",\"floor_met\":" << r.floor_met
            << ",\"cpu_start\":" << r.cpu_start << ",\"cpu_end\":" << r.cpu_end << ",\"qualification_streak\":" << r.streak << '}';
    }
    out << "\n],\n\"paired_region_mean_ratios\":[\n";
    bool first = true;
    for (unsigned group = 0; group < 10; ++group) for (unsigned route = 1; route < 4; ++route) {
        if (!first) out << ",\n";
        first = false;
        out << "{\"group\":" << group << ",\"baseline_cell\":" << group * 4
            << ",\"candidate_cell\":" << group * 4 + route
            << ",\"security_contract_equal_claim\":false,\"ratios_baseline_over_candidate\":[";
        bool first_ratio = true;
        for (unsigned round = 0; round < options.rounds; ++round) {
            const Record* baseline = nullptr; const Record* candidate = nullptr;
            for (const auto& record : records) if (record.phase == "measurement" && record.round == static_cast<int>(round)) {
                if (record.cell == group * 4) baseline = &record;
                if (record.cell == group * 4 + route) candidate = &record;
            }
            if (!baseline || !candidate) continue;
            if (!first_ratio) out << ',';
            first_ratio = false;
            out << "{\"round\":" << round << ",\"ratio\":" << baseline->ns_per_op / candidate->ns_per_op << '}';
        }
        out << "]}";
    }
    out << "\n]\n}\n";
    return out.str();
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
            std::cout << "Usage: p2_field_compare --output NEW.json [--seed UINT64] [--min-ms 0.1..1000] [--rounds EVEN_2..128] [--reverse-order] [--smoke]\n"
                         "Default: seed20260905, 200ms floor, two warmups + eight measured rounds, 1024steps/lane.\n"
                         "40cells: FE64 / canonical FE52 bridge / canonical FE52 resident / CT API, 5ops, chain1/ILP4.\n"
                         "--smoke defaults to 1ms and two measured rounds unless explicit. Never overwrite output.\n"
                         "Exit0 complete, exit2 unqualified calibration, exit1 mismatch/error. Pin externally; no policy changes.\n";
            return 0;
        }
        fd = ::open(options.output.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
        if (fd < 0) throw std::runtime_error(std::string("exclusive output open: ") + std::strerror(errno));
        const auto start_power = power();
        const auto corpus = make_corpus(options.seed);
        std::vector<Cell> cells;
        cells.reserve(40);
        append_op<Op::add>(cells); append_op<Op::sub>(cells); append_op<Op::mul>(cells);
        append_op<Op::square>(cells); append_op<Op::inverse>(cells);
        const auto base = base_order(options.seed);
        std::vector<Record> records;
        records.reserve(cells.size() * (max_attempts + warmups + options.rounds));
        std::string status = "complete", failure;
        int result = 0;
        try {
            for (auto& cell : cells) validate(cell, corpus);
            const auto calibration_order = schedule(base, 0, options.reverse);
            for (unsigned position = 0; position < calibration_order.size(); ++position) {
                auto& cell = cells[calibration_order[position]];
                if (!calibrate(cell, corpus, options, position, records)) {
                    status = "unqualified"; result = 2;
                    failure = "calibration exhausted for cell " + std::to_string(cell.id);
                    break;
                }
            }
            if (!result) for (unsigned round = 0; round < warmups + options.rounds; ++round) {
                const bool warmup = round < warmups;
                const auto order = schedule(base, round, options.reverse);
                for (unsigned position = 0; position < order.size(); ++position) {
                    const auto& cell = cells[order[position]];
                    records.push_back(measure(cell, corpus, options, warmup ? "warmup" : "measurement",
                        static_cast<int>(warmup ? round : round - warmups), position, 0, cell.jobs, records.size()));
                    if (!records.back().verified) throw std::runtime_error("timed final-byte mismatch");
                }
            }
        } catch (const std::exception& error) { status = "error"; failure = error.what(); result = 1; }
        const auto end_power = power();
        write_all(fd, json(options, corpus, cells, records, base, start_power, end_power, status, failure));
        if (::close(fd) != 0) { fd = -1; throw std::runtime_error("output close failed"); }
        fd = -1;
        std::cout << "P2 " << status << " cells=" << cells.size() << " raw_regions=" << records.size()
                  << " output=" << options.output << '\n';
        if (!failure.empty()) std::cerr << failure << '\n';
        return result;
    } catch (const std::exception& error) {
        if (fd >= 0) ::close(fd);
        std::cerr << "P2 error: " << error.what() << '\n';
        return 1;
    }
}
