// Synthetic harness-contract checks, NOT performance measurements.
// Include the unchanged driver so the actual measure(), main(), and JSON writer
// are tested. Linux monotonic-clock interposition supplies deterministic ticks;
// real field jobs, output checks, CPU snapshots, and input preservation still run.
#include <cstdint>
#include <ctime>
#include <cstdlib>
#include <sys/syscall.h>
#include <unistd.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

namespace synthetic_clock {
// This diagnostic is deliberately single-threaded, matching the driver. No
// system clock is changed, and clocks other than CLOCK_MONOTONIC stay real.
bool enabled = false;
std::uint64_t next_ns = 0, calls = 0, fast_calls = 0;
std::uint64_t first_step_ns = 0, later_step_ns = 0;
struct Scope {
    Scope(std::uint64_t first, std::uint64_t later, std::uint64_t switch_after) {
        next_ns = UINT64_C(1000000000); calls = 0;
        first_step_ns = first; later_step_ns = later; fast_calls = switch_after;
        enabled = true;
    }
    ~Scope() { enabled = false; }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};
} // namespace synthetic_clock

extern "C" int clock_gettime(clockid_t id, struct timespec* value) noexcept {
    if (synthetic_clock::enabled && id == CLOCK_MONOTONIC) {
        // clock_gettime requires a valid output pointer. All calls made by the
        // standard-library steady_clock satisfy that contract. The real syscall
        // path preserves the kernel's return value and errno without recursion.
        const auto ns = synthetic_clock::next_ns;
        value->tv_sec = static_cast<time_t>(ns / UINT64_C(1000000000));
        value->tv_nsec = static_cast<long>(ns % UINT64_C(1000000000));
        const auto step = synthetic_clock::calls < synthetic_clock::fast_calls
            ? synthetic_clock::first_step_ns : synthetic_clock::later_step_ns;
        synthetic_clock::next_ns += step;
        ++synthetic_clock::calls;
        return 0;
    }
    return static_cast<int>(::syscall(SYS_clock_gettime, id, value));
}

#define main p4_driver_main
#include "../probes/p4_field_sum_compare.cpp"
#undef main

namespace {
std::uint64_t assertions = 0;
void require(bool condition, const std::string& message) {
    ++assertions;
    if (!condition) throw std::runtime_error(message);
}
void check_record(const Record& record, const Cell& cell, std::uint64_t initial_jobs,
                  unsigned probes, std::uint64_t tick, Stop stop, bool qualified) {
    require(record.initial_batch_jobs == initial_jobs, "initial batch was not retained");
    require(record.jobs == initial_jobs * probes, "actual jobs lost an extension");
    require(record.operations == record.jobs * cell.count, "actual RHS work mismatch");
    require(record.clock_checks == probes && record.extensions + 1 == probes, "probe/extension count mismatch");
    require(record.begin_ns == UINT64_C(1000000000), "synthetic clock did not interpose");
    require(record.end_ns - record.begin_ns == tick * probes, "clock endpoints mismatch");
    require(record.elapsed_ns == tick * probes, "elapsed was not cumulative");
    require(record.stop == stop, "wrong stopping condition");
    require(record.duration_qualified == qualified && record.floor_met == qualified, "wrong duration qualification");
    require(record.verified == (tick > 0), "output/clock verification mismatch");
    require(record.output == cell.expected, "last complete job output mismatch");
    require(record.cpu_start >= 0 && record.cpu_end >= 0, "missing CPU endpoints");
    require(record.power_start.cpu >= 0 && record.power_end.cpu >= 0, "missing power CPU snapshots");
    require(record.ns_per_sum == static_cast<double>(record.elapsed_ns) / record.jobs, "ns/sum uses initial jobs");
    require(record.ns_per_rhs == static_cast<double>(record.elapsed_ns) / record.operations, "ns/RHS uses initial jobs");
    for (unsigned i = 0; i < probes; ++i) {
        const auto& probe = record.probes[i];
        require(probe.cumulative_jobs == initial_jobs * (i + 1), "probe lost job prefix");
        require(probe.cumulative_operations == probe.cumulative_jobs * cell.count, "probe lost RHS prefix");
        require(probe.cumulative_elapsed_ns == tick * (i + 1), "probe elapsed is not cumulative");
        require(probe.end_steady_ns == record.begin_ns + probe.cumulative_elapsed_ns, "probe endpoint mismatch");
    }
    const auto& last = record.probes[probes - 1];
    require(last.cumulative_jobs == record.jobs && last.cumulative_operations == record.operations &&
            last.end_steady_ns == record.end_ns && last.cumulative_elapsed_ns == record.elapsed_ns,
            "final probe disagrees with region");
    require(synthetic_clock::calls == probes + 1, "unexpected clock call inside exact driver path");
}

unsigned test_measure() {
    Corpus corpus;
    corpus.x0 = FE::from_uint64(3);
    corpus.rhs.assign(16, FE::from_uint64(7));
    const auto expected = FE::from_uint64(3 + 16 * 7).to_bytes();
    unsigned cases = 0;
    for (unsigned route = 0; route < routes.size(); ++route) {
        const Cell cell{route, 1, route, 16, expected};
        for (const std::string phase : {"warmup", "measurement"}) {
            Options options; options.min_ms = 3;
            {
                synthetic_clock::Scope clock(1000000, 1000000, 0);
                const auto record = measure(cell, corpus, options, phase, 0, 0, 0, 2, cases++);
                check_record(record, cell, 2, 3, 1000000, Stop::floor_reached, true);
            }
            options.min_ms = 1000; // Deliberately unreachable in thirteen 1ns probes.
            {
                synthetic_clock::Scope clock(1, 1, 0);
                const auto record = measure(cell, corpus, options, phase, 0, 0, 0, 1, cases++);
                check_record(record, cell, 1, max_extensions + 1, 1, Stop::extension_limit, false);
            }
        }
        Options options; options.min_ms = 1000;
        {
            synthetic_clock::Scope clock(1, 1, 0);
            const auto record = measure(cell, corpus, options, "calibration", -1, 0, 0, 3, cases++);
            check_record(record, cell, 3, 1, 1, Stop::calibration_sample, false);
        }
        {
            synthetic_clock::Scope clock(0, 0, 0); // Invalid time must never qualify.
            const auto record = measure(cell, corpus, options, "measurement", 0, 0, 0, 1, cases++);
            check_record(record, cell, 1, 1, 0, Stop::invalid_clock, false);
        }
        {
            synthetic_clock::Scope clock(1, 1, 0);
            bool rejected = false;
            try {
                (void)measure(cell, corpus, options, "measurement", 0, 0, 0,
                              operation_cap / cell.count + 1, cases);
            } catch (const std::runtime_error&) { rejected = true; }
            require(rejected && synthetic_clock::calls == 0, "work overflow was not rejected before clock");
        }
    }
    require(corpus.x0.limbs() == FE::from_uint64(3).limbs(), "small fixture x0 mutated");
    for (const auto& rhs : corpus.rhs)
        require(rhs.limbs() == FE::from_uint64(7).limbs(), "small fixture RHS mutated");
    return cases;
}

std::string test_main_failure() {
    char directory[] = "/tmp/parseatlas-p4-duration.XXXXXX";
    if (!::mkdtemp(directory)) throw std::runtime_error("mkdtemp failed");
    const std::string output = std::string(directory) + "/SYNTHETIC_duration_failure.json";
    std::array<std::string, 4> args{"p4_synthetic_duration_test", "--smoke", "--output", output};
    std::array<char*, 4> argv{};
    for (unsigned i = 0; i < args.size(); ++i) argv[i] = args[i].data();
    // Every calibration region receives exactly 1ms. Two regions per cell,
    // each begin+end, qualify all 48 cells at one job. Then ticks become 1ns
    // without moving time backwards. The first warmup must fail after 13 probes.
    constexpr auto calibration_regions = sizes.size() * routes.size() * 2;
    constexpr auto calibration_calls = calibration_regions * 2;
    int result = -1;
    {
        synthetic_clock::Scope clock(1000000, 1, calibration_calls);
        result = p4_driver_main(static_cast<int>(argv.size()), argv.data());
        require(synthetic_clock::calls == calibration_calls + max_extensions + 2,
                "scripted full-driver clock path changed");
    }
    require(result == 2, "failed later-phase duration did not fail closed");
    boost::property_tree::ptree data;
    boost::property_tree::read_json(output, data);
    require(data.get<std::string>("status") == "duration_unqualified", "failure JSON claims success");
    require(data.get<std::string>("failure").find("extension_limit") != std::string::npos, "failure reason not retained");
    require(data.get<std::string>("corpus.checksum_fnv1a64") == "a228d8a510693ab4", "full corpus recipe changed");
    const auto& cells = data.get_child("cells");
    require(cells.size() == sizes.size() * routes.size(), "full matrix was not retained");
    for (const auto& item : cells)
        require(item.second.get<bool>("validated") && item.second.get<bool>("qualified"), "cell not qualified before synthetic failure");
    const auto& records = data.get_child("regions");
    require(records.size() == calibration_regions + 1, "short region or calibration evidence lost");
    unsigned index = 0;
    for (const auto& item : records) {
        const auto& record = item.second;
        const bool calibration = index++ < calibration_regions;
        const auto actual_jobs = record.get<std::uint64_t>("jobs");
        require(record.get<std::string>("phase") == (calibration ? "calibration" : "warmup"), "failed run continued phases");
        require(record.get<bool>("duration_qualified") == calibration && record.get<bool>("floor_met") == calibration,
                "serialized duration status mismatch");
        require(record.get<bool>("verified"), "real full output verification failed");
        require(actual_jobs == (calibration ? 1 : max_extensions + 1), "serialized actual jobs mismatch");
        require(record.get<unsigned>("initial_batch_jobs") == 1, "serialized initial jobs mismatch");
        const auto begin = record.get<std::uint64_t>("begin_steady_ns");
        const auto end = record.get<std::uint64_t>("end_steady_ns");
        require(end - begin == record.get<std::uint64_t>("elapsed_ns"), "serialized endpoint arithmetic mismatch");
        const auto tick = calibration ? 1000000U : 1U;
        require(end - begin == tick * actual_jobs, "serialized clock script mismatch");
        const auto& probes = record.get_child("cumulative_probes");
        require(probes.size() == actual_jobs && record.get<unsigned>("clock_checks") == probes.size() &&
                record.get<unsigned>("extensions") + 1 == probes.size(), "serialized probes not retained");
        auto cell = cells.begin(); std::advance(cell, record.get<unsigned>("cell"));
        const auto count = cell->second.get<std::uint64_t>("rhs_count");
        require(record.get<std::string>("actual_final_be32") == cell->second.get<std::string>("expected_final_be32"),
                "serialized output bytes mismatch");
        require(record.get<std::uint64_t>("operations") == count * actual_jobs, "serialized actual RHS count mismatch");
        require(record.get<int>("cpu_start") >= 0 && record.get<int>("cpu_end") >= 0, "serialized CPU endpoints absent");
        unsigned probe_index = 0;
        for (const auto& probe : probes) {
            ++probe_index;
            require(probe.second.get<std::uint64_t>("cumulative_jobs") == probe_index &&
                    probe.second.get<std::uint64_t>("cumulative_operations") == count * probe_index &&
                    probe.second.get<std::uint64_t>("cumulative_elapsed_ns") == tick * probe_index &&
                    probe.second.get<std::uint64_t>("end_steady_ns") == begin + tick * probe_index,
                    "serialized cumulative prefix mismatch");
        }
    }
    const auto& preservation = data.get_child("input_preservation");
    require(preservation.size() == 2, "post-failure input preservation absent");
    for (const auto& check : preservation)
        require(check.second.get<bool>("verified") && check.second.get<std::uint64_t>("exact_rhs_checked") == sizes.back(),
                "full input preservation failed");
    return output; // Keep exclusive output as diagnostic evidence; never a benchmark input.
}
} // namespace

int main() {
    try {
        std::cout << "SYNTHETIC duration diagnostic: all clock values below are fabricated, NOT performance.\n";
        const auto cases = test_measure();
        const auto artifact = test_main_failure();
        std::cout << "PASS synthetic_duration_cases=" << cases << " assertions=" << assertions
                  << " artifact=" << artifact << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL synthetic duration contract: " << error.what() << '\n';
        return 1;
    }
}
