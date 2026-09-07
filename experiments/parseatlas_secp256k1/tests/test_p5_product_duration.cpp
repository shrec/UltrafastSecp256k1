// Synthetic harness-contract checks, NOT performance measurements.
// Include the unchanged P5 driver so its actual measure(), main(), and JSON
// writer run with deterministic CLOCK_MONOTONIC ticks. Real field jobs, complete
// output checks, CPU snapshots, and input-preservation checks still execute.
#include <cstdint>
#include <ctime>
#include <cstdlib>
#include <sys/syscall.h>
#include <unistd.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

namespace synthetic_clock {
// Deliberately single-threaded, as is the measured driver. This interposition
// changes no system clock; every clock except CLOCK_MONOTONIC remains real.
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
        // steady_clock supplies a valid output pointer. The real syscall path
        // preserves the kernel return value and errno without recursion.
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

#define main p5_driver_main
#include "../probes/p5_field_product_compare.cpp"
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
    require(record.operations == record.jobs * cell.count, "actual primitive work mismatch");
    require(record.clock_checks == probes && record.extensions + 1 == probes,
            "probe/extension count mismatch");
    require(record.begin_ns == UINT64_C(1000000000), "synthetic clock did not interpose");
    require(record.end_ns - record.begin_ns == tick * probes, "clock endpoints mismatch");
    require(record.elapsed_ns == tick * probes, "elapsed was not cumulative");
    require(record.stop == stop, "wrong stopping condition");
    require(record.duration_qualified == qualified && record.floor_met == qualified,
            "wrong duration qualification");
    require(record.verified == (tick > 0), "output/clock verification mismatch");
    require(record.output == cell.expected, "last complete full-lane output mismatch");
    require(record.cpu_start >= 0 && record.cpu_end >= 0, "missing CPU endpoints");
    require(record.power_start.cpu >= 0 && record.power_end.cpu >= 0,
            "missing power CPU snapshots");
    require(record.ns_per_job == static_cast<double>(record.elapsed_ns) / record.jobs,
            "ns/job uses initial rather than actual jobs");
    require(record.ns_per_operation == static_cast<double>(record.elapsed_ns) / record.operations,
            "ns/operation uses initial rather than actual work");
    for (unsigned i = 0; i < probes; ++i) {
        const auto& probe = record.probes[i];
        require(probe.cumulative_jobs == initial_jobs * (i + 1), "probe lost job prefix");
        require(probe.cumulative_operations == probe.cumulative_jobs * cell.count,
                "probe lost primitive-work prefix");
        require(probe.cumulative_elapsed_ns == tick * (i + 1),
                "probe elapsed is not cumulative");
        require(probe.end_steady_ns == record.begin_ns + probe.cumulative_elapsed_ns,
                "probe endpoint mismatch");
    }
    const auto& last = record.probes[probes - 1];
    require(last.cumulative_jobs == record.jobs &&
            last.cumulative_operations == record.operations &&
            last.end_steady_ns == record.end_ns &&
            last.cumulative_elapsed_ns == record.elapsed_ns,
            "final probe disagrees with region");
    require(synthetic_clock::calls == probes + 1,
            "unexpected clock call inside exact driver path");
}

unsigned test_measure(const Corpus& corpus, const std::vector<Cell>& cells) {
    unsigned cases = 0;
    require(cells.size() == 20, "P5 matrix must contain all 20 cells");
    for (const auto& cell : cells) {
        require(cell.validated, "direct measure cell was not validated");
        require((cell.lanes == 1 || cell.lanes == 4) && cell.count == 1024 * cell.lanes,
                "cell work does not count every active lane");
        for (std::size_t i = 32 * cell.lanes; i < cell.expected.size(); ++i)
            require(cell.expected[i] == 0, "inactive output lane is not zero padded");
        for (const std::string phase : {"warmup", "measurement"}) {
            Options options; options.min_ms = 3;
            {
                synthetic_clock::Scope clock(1000000, 1000000, 0);
                const auto record = measure(cell, corpus, options, phase, 0, 0, 0, 2, cases++);
                check_record(record, cell, 2, 3, 1000000, Stop::floor_reached, true);
            }
            options.min_ms = 1000; // Unreachable in thirteen synthetic 1ns probes.
            {
                synthetic_clock::Scope clock(1, 1, 0);
                const auto record = measure(cell, corpus, options, phase, 0, 0, 0, 1, cases++);
                check_record(record, cell, 1, max_extensions + 1, 1,
                             Stop::extension_limit, false);
            }
        }
        Options options; options.min_ms = 1000;
        {
            synthetic_clock::Scope clock(1, 1, 0);
            const auto record = measure(cell, corpus, options, "calibration", -1, 0, 0, 3, cases++);
            check_record(record, cell, 3, 1, 1, Stop::calibration_sample, false);
        }
        {
            synthetic_clock::Scope clock(0, 0, 0);
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
            require(rejected && synthetic_clock::calls == 0,
                    "work overflow was not rejected before clock");
        }
    }
    return cases;
}

void check_preservation(const Corpus& corpus, std::uint64_t seed, const std::string& phase) {
    const auto check = check_inputs(corpus, seed, phase);
    require(check.verified, "direct-measure corpus preservation failed");
    require(check.exact_seed_checked == 4, "not all seeds were checked");
    require(check.exact_rhs_checked == 256, "not all RHS operands were checked");
    require(check.exact_raw512_checked == 256, "not all arbitrary-512 inputs were checked");
}

void test_serialize_lane_contract() {
    for (const unsigned lanes : {0U, 2U, 5U, ~0U}) {
        bool rejected = false;
        try {
            (void)serialize(States{}, lanes);
        } catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "serialize accepted a malformed active-lane count");
    }
}

std::string expected_hex(const Cell& cell) {
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (auto byte : cell.expected) output << std::setw(2) << static_cast<unsigned>(byte);
    return output.str();
}

std::string test_main_failure(const std::vector<Cell>& expected_cells) {
    char directory[] = "/tmp/parseatlas-p5-duration.XXXXXX";
    if (!::mkdtemp(directory)) throw std::runtime_error("mkdtemp failed");
    const std::string output = std::string(directory) + "/SYNTHETIC_duration_failure.json";
    std::array<std::string, 4> args{"p5_synthetic_duration_test", "--smoke", "--output", output};
    std::array<char*, 4> argv{};
    for (unsigned i = 0; i < args.size(); ++i) argv[i] = args[i].data();
    // Each of the two calibration regions per cell has a begin and one end
    // clock call. Exactly 1ms qualifies each at one job. After all 20 cells,
    // ticks become 1ns without time reversal. Warmup then fails at 13 probes.
    const auto calibration_regions = expected_cells.size() * 2;
    const auto calibration_calls = calibration_regions * 2;
    int result = -1;
    {
        synthetic_clock::Scope clock(1000000, 1, calibration_calls);
        result = p5_driver_main(static_cast<int>(argv.size()), argv.data());
        require(synthetic_clock::calls == calibration_calls + max_extensions + 2,
                "scripted full-driver clock path changed");
    }
    require(result == 2, "failed later-phase duration did not fail closed");
    boost::property_tree::ptree data;
    boost::property_tree::read_json(output, data);
    require(data.get<std::string>("status") == "duration_unqualified", "failure JSON claims success");
    require(data.get<std::string>("failure").find("extension_limit") != std::string::npos,
            "failure reason was not retained");
    const auto& cells = data.get_child("cells");
    require(cells.size() == expected_cells.size(), "full cell matrix was not retained");
    unsigned cell_index = 0;
    for (const auto& item : cells) {
        const auto& cell = item.second;
        const auto& expected = expected_cells[cell_index++];
        require(cell.get<bool>("validated") && cell.get<bool>("qualified"),
                "cell was not qualified before synthetic failure");
        require(cell.get<std::uint64_t>("operations_per_job") == expected.count,
                "serialized cell primitive count mismatch");
        require(cell.get<unsigned>("lanes") == expected.lanes, "serialized lane count mismatch");
        require(cell.get<std::string>("expected_final_be128") == expected_hex(expected),
                "serialized complete expected output mismatch");
    }
    const auto& records = data.get_child("regions");
    require(records.size() == calibration_regions + 1,
            "short region or calibration evidence was lost");
    unsigned index = 0;
    for (const auto& item : records) {
        const auto& record = item.second;
        const bool calibration = index++ < calibration_regions;
        const auto actual_jobs = record.get<std::uint64_t>("jobs");
        require(record.get<std::string>("phase") == (calibration ? "calibration" : "warmup"),
                "failed run continued to later phases");
        require(record.get<bool>("duration_qualified") == calibration &&
                record.get<bool>("floor_met") == calibration,
                "serialized duration qualification mismatch");
        require(record.get<std::string>("stop_reason") ==
                (calibration ? "floor_reached" : "extension_limit"),
                "serialized stop reason mismatch");
        require(record.get<bool>("verified"), "real full-lane output verification failed");
        require(actual_jobs == (calibration ? 1 : max_extensions + 1),
                "serialized actual jobs mismatch");
        require(record.get<unsigned>("initial_batch_jobs") == 1,
                "serialized initial jobs mismatch");
        const auto begin = record.get<std::uint64_t>("begin_steady_ns");
        const auto end = record.get<std::uint64_t>("end_steady_ns");
        const auto elapsed = record.get<std::uint64_t>("elapsed_ns");
        require(end - begin == elapsed, "serialized endpoint arithmetic mismatch");
        const auto tick = calibration ? 1000000U : 1U;
        require(elapsed == tick * actual_jobs, "serialized clock script mismatch");
        const auto& probes = record.get_child("cumulative_probes");
        require(probes.size() == actual_jobs &&
                record.get<unsigned>("clock_checks") == probes.size() &&
                record.get<unsigned>("extensions") + 1 == probes.size(),
                "serialized probes were not retained");
        const auto id = record.get<unsigned>("cell");
        require(id < expected_cells.size(), "serialized region has invalid cell");
        const auto& cell = expected_cells[id];
        require(record.get<std::string>("actual_final_be128") == expected_hex(cell),
                "serialized output bytes mismatch");
        require(record.get<std::uint64_t>("operations") == cell.count * actual_jobs,
                "serialized actual primitive count mismatch");
        const auto ns_per_job = static_cast<double>(elapsed) / actual_jobs;
        const auto ns_per_op = static_cast<double>(elapsed) / (cell.count * actual_jobs);
        require(std::abs(record.get<double>("ns_per_job") - ns_per_job) <=
                std::max(1.0, ns_per_job) * 1e-10, "serialized ns/job mismatch");
        require(std::abs(record.get<double>("ns_per_operation") - ns_per_op) <=
                std::max(1.0, ns_per_op) * 1e-10, "serialized ns/operation mismatch");
        require(record.get<int>("cpu_start") >= 0 && record.get<int>("cpu_end") >= 0,
                "serialized CPU endpoints absent");
        unsigned probe_index = 0;
        for (const auto& probe : probes) {
            ++probe_index;
            require(probe.second.get<std::uint64_t>("cumulative_jobs") == probe_index &&
                    probe.second.get<std::uint64_t>("cumulative_operations") == cell.count * probe_index &&
                    probe.second.get<std::uint64_t>("cumulative_elapsed_ns") == tick * probe_index &&
                    probe.second.get<std::uint64_t>("end_steady_ns") == begin + tick * probe_index,
                    "serialized cumulative prefix mismatch");
        }
    }
    const auto& preservation = data.get_child("input_preservation");
    require(preservation.size() == 2, "post-failure input preservation absent");
    for (const auto& item : preservation) {
        const auto& check = item.second;
        require(check.get<bool>("verified") &&
                check.get<std::uint64_t>("exact_seed_checked") == 4 &&
                check.get<std::uint64_t>("exact_rhs_checked") == 256 &&
                check.get<std::uint64_t>("exact_raw512_checked") == 256,
                "full input preservation failed");
    }
    return output; // Keep exclusive diagnostic evidence; never a benchmark input.
}
} // namespace

int main() {
    try {
        std::cout << "SYNTHETIC duration diagnostic: all clock values below are fabricated, NOT performance.\n";
        const auto seed = Options{}.seed;
        Corpus corpus;
        make_corpus(corpus, seed);
        auto cells = make_cells();
        validate(cells, corpus);
        test_serialize_lane_contract();
        check_preservation(corpus, seed, "before_synthetic_measure");
        const auto cases = test_measure(corpus, cells);
        check_preservation(corpus, seed, "after_synthetic_measure");
        const auto artifact = test_main_failure(cells);
        std::cout << "PASS synthetic_duration_cases=" << cases << " assertions=" << assertions
                  << " artifact=" << artifact << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL synthetic duration contract: " << error.what() << '\n';
        return 1;
    }
}
