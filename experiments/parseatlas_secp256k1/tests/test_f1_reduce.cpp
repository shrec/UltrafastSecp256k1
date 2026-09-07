#include "../probes/f1_reduce.hpp"
#include "secp256k1/scalar.hpp"

#include <boost/multiprecision/cpp_int.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <dirent.h>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <pthread.h>
#include <sched.h>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <vector>

namespace {

using pa_f1::Limbs;
using pa_f1::Word;
using boost::multiprecision::cpp_int;
using secp256k1::fast::Scalar;
using Bytes = std::array<std::uint8_t, 32>;

constexpr Word seed = 20260905;
constexpr Word random_case_count = 4096;
const cpp_int modulus{"0xfffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141"};
const cpp_int width = cpp_int{1} << 256;

void require(bool condition, const char* message) {
    // Correctness gates must remain live in -DNDEBUG builds.
    if (!condition) throw std::runtime_error(message);
}

cpp_int to_bigint(const Limbs& value) {
    cpp_int result = 0;
    for (unsigned i = 4; i != 0; --i) {
        result <<= 64;
        result += value[i - 1];
    }
    return result;
}

Limbs to_limbs(cpp_int value) {
    require(value >= 0 && value < width, "test conversion outside 256-bit domain");
    const cpp_int mask = (cpp_int{1} << 64) - 1;
    Limbs result{};
    for (auto& limb : result) {
        limb = (value & mask).convert_to<Word>();
        value >>= 64;
    }
    return result;
}

Bytes independent_le(const Limbs& value) {
    Bytes result{};
    for (unsigned i = 0; i < 32; ++i) {
        result[i] = static_cast<std::uint8_t>(value[i / 8] >> (8 * (i % 8)));
    }
    return result;
}

bool same_bytes(const Limbs& a, const Limbs& b) {
    return independent_le(a) == independent_le(b);
}

Word next_random(Word& state) {
    state += UINT64_C(0x9e3779b97f4a7c15);
    Word z = state;
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

Limbs random_canonical(Word& state) {
    Limbs result{};
    do {
        for (auto& limb : result) limb = next_random(state);
    } while (to_bigint(result) >= modulus);
    return result;
}

struct Affinity {
    cpu_set_t original{};
    std::vector<int> allowed;
    std::vector<std::vector<int>> lists;
    std::size_t reserved = 0;
};

Affinity discover_affinity() {
    Affinity result;
    CPU_ZERO(&result.original);
    require(sched_getaffinity(0, sizeof(result.original), &result.original) == 0,
            "cannot query test affinity");
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &result.original)) result.allowed.push_back(cpu);
    }
    require(!result.allowed.empty(), "empty allowed CPU mask");
    // Derive worker capacity from the observed allowed set, reserving at least
    // one CPU (a quarter when available) for interactive work. A one-CPU
    // environment has no spare CPU; this limitation is explicit in the JSON.
    result.reserved = result.allowed.size() > 1
        ? std::max<std::size_t>(1, (result.allowed.size() + 3) / 4) : 0;
    const std::size_t usable = result.allowed.size() - result.reserved;
    const std::array<std::size_t, 3> sizes{{1, std::max<std::size_t>(1, usable / 2), usable}};
    for (const auto size : sizes) {
        std::vector<int> list(result.allowed.begin(), result.allowed.begin() + size);
        if (std::find(result.lists.begin(), result.lists.end(), list) == result.lists.end()) {
            result.lists.push_back(list);
        }
    }
    return result;
}

struct Counts {
    Word fixture_cases = 0;
    Word boundary_cases = 0;
    Word targeted_cases = 0;
    Word total_rhs_values = 0;
    Word empty_cases = 0;
    Word reference_replay_additions = 0;
    Word reference_final_checks = 0;
    Word serial_final_checks = 0;
    Word chunk4_final_checks = 0;
    Word tree_final_checks = 0;
    Word partition_final_checks = 0;
    Word partition_kernel_calls = 0;
    Word result_canonical_checks = 0;
    Word encoder_checks = 0;
    Word public_be_checks = 0;
    Word preserved_input_values = 0;
    Word scratch_canary_checks = 0;
    Word multicore_fixture_subset = 0;
    Word multicore_calls = 0;
    Word multicore_contract_selected_workers = 0;
    Word multicore_list_larger_than_input_calls = 0;
    Word multicore_uneven_chunk_calls = 0;
    Word main_affinity_preservation_checks = 0;
    Word affinity_probe_calls = 0;
    int affinity_probe_cpu = -1;
    int affinity_probe_status = 0;
    Word affinity_failure_calls = 0;
    Word affinity_failure_threads_before = 0;
    Word affinity_failure_threads_after = 0;
    Word checked_error_cases = 0;
    Word empty_identity_gate_checks = 0;
    Word negative_comparator_checks = 0;
    Word checksum = UINT64_C(14695981039346656037);
};

void digest_result(const Limbs& result, unsigned variant, Counts& counts) {
    counts.checksum ^= variant;
    counts.checksum *= UINT64_C(1099511628211);
    for (const auto byte : independent_le(result)) {
        counts.checksum ^= byte;
        counts.checksum *= UINT64_C(1099511628211);
    }
}

void check_result(const Limbs& actual, const Limbs& expected,
                  const Limbs& reference, Counts& counts) {
    require(same_bytes(actual, expected) && same_bytes(actual, reference),
            "complete 32-byte final result differs from bigint/library");
    require(to_bigint(actual) < modulus && pa_modn::canonical(actual),
            "reduction result is not canonical");
    ++counts.result_canonical_checks;
    require(pa_modn::encode_le(actual) == independent_le(expected),
            "candidate encoder differs from independent LE bytes");
    ++counts.encoder_checks;
}

void check_preservation(const Limbs& x0, const Limbs& saved_x0,
                        const std::vector<Limbs>& rhs,
                        const std::vector<Limbs>& saved_rhs, Counts& counts) {
    require(same_bytes(x0, saved_x0) && rhs.size() == saved_rhs.size(),
            "reduction mutated x0 or input size");
    ++counts.preserved_input_values;
    for (std::size_t i = 0; i < rhs.size(); ++i) {
        require(same_bytes(rhs[i], saved_rhs[i]), "reduction mutated an RHS value");
        ++counts.preserved_input_values;
    }
}

void check_main_affinity(const Affinity& affinity, Counts& counts) {
    cpu_set_t after;
    CPU_ZERO(&after);
    require(sched_getaffinity(0, sizeof(after), &after) == 0 &&
            CPU_EQUAL(&after, &affinity.original), "multicore reduction changed caller affinity");
    ++counts.main_affinity_preservation_checks;
}

void check_case(Limbs x0, std::vector<Limbs> rhs, bool test_multicore,
                const Affinity& affinity, Word& partition_random, Counts& counts) {
    require(to_bigint(x0) < modulus, "noncanonical x0 fixture");
    cpp_int total = to_bigint(x0);
    for (const auto& value : rhs) {
        require(to_bigint(value) < modulus, "noncanonical RHS fixture");
        total += to_bigint(value);
    }
    const Limbs expected = to_limbs(total % modulus);
    const Limbs saved_x0 = x0;
    const std::vector<Limbs> saved_rhs = rhs;
    const Limbs* const input = rhs.empty() ? nullptr : rhs.data();

    Scalar library = Scalar::from_limbs(x0);
    for (const auto& value : rhs) {
        const Scalar addend = Scalar::from_limbs(value);
        library += addend;
        ++counts.reference_replay_additions;
    }
    const Limbs reference = library.limbs();
    check_result(reference, expected, expected, counts);
    ++counts.reference_final_checks;
    Bytes expected_be{};
    const Bytes expected_le = independent_le(expected);
    for (unsigned byte = 0; byte < 32; ++byte) expected_be[byte] = expected_le[31 - byte];
    require(library.to_bytes() == expected_be, "public Scalar BE bytes differ from oracle");
    ++counts.public_be_checks;

    const Limbs serial = pa_f1::serial(x0, input, rhs.size());
    check_result(serial, expected, reference, counts);
    check_preservation(x0, saved_x0, rhs, saved_rhs, counts);
    digest_result(serial, 0, counts);
    ++counts.serial_final_checks;

    const Limbs chunked = pa_f1::chunk4_interleaved(x0, input, rhs.size());
    check_result(chunked, expected, reference, counts);
    check_preservation(x0, saved_x0, rhs, saved_rhs, counts);
    digest_result(chunked, 1, counts);
    ++counts.chunk4_final_checks;

    // Only the first N scratch limbs may be modified, even when the supplied
    // capacity is larger. Two leading and two trailing sentinels cover bounds.
    const Limbs canary{{UINT64_C(0x0123456789abcdef), UINT64_C(0xfedcba9876543210),
                        UINT64_C(0xa55aa55a5aa55aa5), UINT64_C(0x55aa33cc0ff0f00f)}};
    std::vector<Limbs> scratch(rhs.size() + 4, canary);
    const Limbs tree = pa_f1::tree(x0, input, rhs.size(), scratch.data() + 2, rhs.size() + 2);
    check_result(tree, expected, reference, counts);
    for (const std::size_t index : std::array<std::size_t, 4>{{0, 1, rhs.size() + 2, rhs.size() + 3}}) {
        require(same_bytes(scratch[index], canary), "tree wrote outside first N scratch entries");
        ++counts.scratch_canary_checks;
    }
    check_preservation(x0, saved_x0, rhs, saved_rhs, counts);
    digest_result(tree, 2, counts);
    ++counts.tree_final_checks;

    // Eight random cut positions plus endpoints exercise contiguous, sometimes
    // empty partitions. x0 enters the first fold once, never once per chunk.
    std::array<std::size_t, 10> cuts{};
    cuts.back() = rhs.size();
    for (std::size_t i = 1; i + 1 < cuts.size(); ++i) {
        cuts[i] = static_cast<std::size_t>(next_random(partition_random) % (rhs.size() + 1));
    }
    std::sort(cuts.begin(), cuts.end());
    Limbs partitioned = x0;
    for (std::size_t i = 1; i < cuts.size(); ++i) {
        const std::size_t length = cuts[i] - cuts[i - 1];
        const Limbs* const part = length ? rhs.data() + cuts[i - 1] : nullptr;
        partitioned = pa_f1::chunk4_interleaved(partitioned, part, length);
        ++counts.partition_kernel_calls;
    }
    check_result(partitioned, expected, reference, counts);
    check_preservation(x0, saved_x0, rhs, saved_rhs, counts);
    digest_result(partitioned, 3, counts);
    ++counts.partition_final_checks;

    if (test_multicore) {
        ++counts.multicore_fixture_subset;
        for (const auto& cpus : affinity.lists) {
            const std::vector<int> saved_cpus = cpus;
            const Limbs parallel = pa_f1::multicore_cold(x0, input, rhs.size(), cpus);
            check_result(parallel, expected, reference, counts);
            check_preservation(x0, saved_x0, rhs, saved_rhs, counts);
            require(cpus == saved_cpus, "multicore reduction changed CPU list");
            check_main_affinity(affinity, counts);
            ++counts.multicore_calls;
            const std::size_t workers = std::min(rhs.size(), cpus.size());
            counts.multicore_contract_selected_workers += workers;
            if (cpus.size() > rhs.size()) ++counts.multicore_list_larger_than_input_calls;
            if (workers && rhs.size() % workers) ++counts.multicore_uneven_chunk_calls;
            // The checksum intentionally excludes environment-dependent CPU
            // configurations; correctness still checks every parallel result.
        }
    }
    ++counts.fixture_cases;
    counts.total_rhs_values += rhs.size();
    if (rhs.empty()) ++counts.empty_cases;
}

void check_boundaries(const Affinity& affinity, Word& partition_random, Counts& counts) {
    constexpr std::array<std::size_t, 30> sizes{{0, 1, 2, 3, 4, 5, 7, 8, 9, 15,
        16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257,
        511, 512, 513, 1023, 1024, 1025}};
    const std::array<Limbs, 3> initial{{Limbs{}, Limbs{1, 0, 0, 0}, to_limbs(modulus - 1)}};
    for (const auto size : sizes) for (unsigned pattern = 0; pattern < 4; ++pattern) {
        std::vector<Limbs> rhs(size);
        for (std::size_t i = 0; i < size; ++i) {
            if (pattern == 1) rhs[i] = to_limbs(modulus - 1);
            if (pattern == 2) rhs[i] = i % 2 ? Limbs{1, 0, 0, 0} : to_limbs(modulus - 1);
            if (pattern == 3) {
                const unsigned bit = static_cast<unsigned>((i * 63) % 256);
                Limbs value{};
                for (unsigned limb = 0; limb < bit / 64; ++limb) value[limb] = ~Word{0};
                if (bit % 64) value[bit / 64] = (Word{1} << (bit % 64)) - 1;
                rhs[i] = value;
            }
        }
        for (unsigned initial_index = 0; initial_index < initial.size(); ++initial_index) {
            check_case(initial[initial_index], rhs, pattern == 2 && initial_index == 2,
                       affinity, partition_random, counts);
            ++counts.boundary_cases;
        }
    }
}

void check_targeted(const Affinity& affinity, Word& partition_random, Counts& counts) {
    for (unsigned bit = 0; bit < 256; ++bit) {
        const cpp_int power = cpp_int{1} << bit;
        const std::vector<Limbs> rhs{{to_limbs(power - 1), Limbs{1, 0, 0, 0},
                                      to_limbs(modulus - power)}};
        check_case(Limbs{}, rhs, bit % 64 == 0, affinity, partition_random, counts);
        check_case(to_limbs(modulus - 1), rhs, false, affinity, partition_random, counts);
        counts.targeted_cases += 2;
    }
    const cpp_int a = modulus - 1;
    for (int delta = -1; delta <= 1; ++delta) {
        // Raw sum immediately below, at, and above 2^256. The third addend
        // crosses an n boundary after that wrap/reduction.
        const std::vector<Limbs> rhs{{to_limbs(a), to_limbs(width - a + delta),
                                      to_limbs(modulus - 1)}};
        check_case(to_limbs(cpp_int{17}), rhs, true, affinity, partition_random, counts);
        ++counts.targeted_cases;
    }
}

template<class Exception, class Function>
void expect_error(Function function, Counts& counts) {
    bool caught = false;
    try {
        function();
    } catch (const Exception&) {
        caught = true;
    } catch (...) {
        throw std::runtime_error("checked API threw wrong exception type");
    }
    require(caught, "checked API accepted invalid argument");
    ++counts.checked_error_cases;
}

void check_errors_and_identity(const Affinity& affinity, Counts& counts) {
    const Limbs x0 = to_limbs(modulus - 1);
    const std::array<Limbs, 2> rhs{{Limbs{1, 0, 0, 0}, Limbs{2, 0, 0, 0}}};
    const auto saved_rhs = rhs;
    std::array<Limbs, 4> scratch{{x0, x0, x0, x0}};
    const auto saved_scratch = scratch;
    const std::vector<int> cpus{affinity.allowed.front()};
    const std::vector<int> duplicate{affinity.allowed.front(), affinity.allowed.front()};

    require(pa_f1::max_count == std::numeric_limits<std::size_t>::max() / sizeof(Limbs),
            "unexpected count extent bound");
    for (const auto count : std::array<std::size_t, 2>{{pa_f1::max_count + 1,
                                                      std::numeric_limits<std::size_t>::max()}}) {
        expect_error<std::length_error>([&] { (void)pa_f1::serial(x0, nullptr, count); }, counts);
        expect_error<std::length_error>([&] { (void)pa_f1::chunk4_interleaved(x0, nullptr, count); }, counts);
        expect_error<std::length_error>([&] { (void)pa_f1::tree(x0, nullptr, count, nullptr, 0); }, counts);
        expect_error<std::length_error>([&] { (void)pa_f1::multicore_cold(x0, nullptr, count, {}); }, counts);
    }
    expect_error<std::invalid_argument>([&] { (void)pa_f1::serial(x0, nullptr, 1); }, counts);
    expect_error<std::invalid_argument>([&] { (void)pa_f1::chunk4_interleaved(x0, nullptr, 1); }, counts);
    expect_error<std::invalid_argument>([&] { (void)pa_f1::tree(x0, nullptr, 1, scratch.data(), 4); }, counts);
    expect_error<std::invalid_argument>([&] { (void)pa_f1::multicore_cold(x0, nullptr, 1, cpus); }, counts);
    expect_error<std::invalid_argument>([&] { (void)pa_f1::tree(x0, rhs.data(), 1, nullptr, 1); }, counts);
    expect_error<std::invalid_argument>([&] { (void)pa_f1::tree(x0, rhs.data(), 2, scratch.data(), 1); }, counts);
    expect_error<std::invalid_argument>([&] { (void)pa_f1::multicore_cold(x0, rhs.data(), 1, {}); }, counts);
    expect_error<std::invalid_argument>([&] { (void)pa_f1::multicore_cold(x0, rhs.data(), 1, {-1}); }, counts);
    expect_error<std::invalid_argument>([&] { (void)pa_f1::multicore_cold(x0, rhs.data(), 1, {CPU_SETSIZE}); }, counts);
    expect_error<std::invalid_argument>([&] { (void)pa_f1::multicore_cold(x0, rhs.data(), 2, duplicate); }, counts);
    // Validate the whole requested list, including an unused tail when N=1.
    expect_error<std::invalid_argument>([&] { (void)pa_f1::multicore_cold(x0, rhs.data(), 1, duplicate); }, counts);

    const std::array<Limbs, 6> empty_results{{
        pa_f1::serial(x0, nullptr, 0),
        pa_f1::chunk4_interleaved(x0, nullptr, 0),
        pa_f1::tree(x0, nullptr, 0, nullptr, 0),
        pa_f1::tree(x0, nullptr, 0, scratch.data(), scratch.size()),
        pa_f1::multicore_cold(x0, nullptr, 0, {}),
        pa_f1::multicore_cold(x0, nullptr, 0, {-1})
    }};
    for (const auto& result : empty_results) {
        require(same_bytes(result, x0), "empty reduction did not preserve x0 exactly once");
        ++counts.empty_identity_gate_checks;
    }
    for (std::size_t i = 0; i < rhs.size(); ++i) {
        require(same_bytes(rhs[i], saved_rhs[i]), "checked error mutated RHS");
    }
    for (std::size_t i = 0; i < scratch.size(); ++i) {
        require(same_bytes(scratch[i], saved_scratch[i]), "checked error/empty call wrote scratch");
        ++counts.scratch_canary_checks;
    }
    check_main_affinity(affinity, counts);
    // Canonical-input, allocation-extent and input/scratch non-overlap are
    // documented external preconditions, not checked APIs: do not invoke UB.
}

Word current_thread_count() {
    // Exact process-local runtime diagnostic, not repository discovery. A
    // completed exception plus no remaining live threads complements review
    // of JoinAll; thread counts alone cannot prove a terminated thread joined.
    DIR* const directory = opendir("/proc/self/task");
    require(directory != nullptr, "cannot open process-local thread directory");
    Word result = 0;
    while (const dirent* entry = readdir(directory)) {
        if (entry->d_name[0] >= '0' && entry->d_name[0] <= '9') ++result;
    }
    const int closed = closedir(directory);
    require(closed == 0 && result != 0, "cannot count process-local threads");
    return result;
}

void check_affinity_failure(const Affinity& affinity, Counts& counts) {
    // Being absent from the caller's mask does not prove a CPU is unavailable:
    // Linux may permit a thread to expand its inherited mask. Probe the highest
    // representable CPU in a disposable thread before expecting a failure.
    counts.affinity_probe_cpu = CPU_SETSIZE - 1;
    std::thread probe([&] {
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(counts.affinity_probe_cpu, &mask);
        counts.affinity_probe_status = pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
    });
    probe.join();
    ++counts.affinity_probe_calls;
    check_main_affinity(affinity, counts);
    if (counts.affinity_probe_status == 0) return;

    const Limbs x0 = to_limbs(modulus - 1);
    const std::vector<Limbs> input(4097, Limbs{1, 0, 0, 0});
    const auto saved_input = input;
    const std::vector<int> cpus{affinity.allowed.front(), counts.affinity_probe_cpu};
    require(cpus.front() != cpus.back(), "affinity probe unexpectedly failed on first allowed CPU");
    counts.affinity_failure_threads_before = current_thread_count();
    bool caught = false;
    try {
        (void)pa_f1::multicore_cold(x0, input.data(), input.size(), cpus);
    } catch (const std::system_error& error) {
        require(error.code().value() == counts.affinity_probe_status &&
                error.code().category() == std::generic_category(),
                "worker affinity exception differs from observed probe failure");
        caught = true;
    }
    require(caught, "observed unavailable CPU did not propagate worker affinity failure");
    ++counts.affinity_failure_calls;
    counts.affinity_failure_threads_after = current_thread_count();
    require(counts.affinity_failure_threads_after == counts.affinity_failure_threads_before,
            "worker affinity exception left live threads behind");
    check_preservation(x0, to_limbs(modulus - 1), input, saved_input, counts);
    check_main_affinity(affinity, counts);
    // This tests worker-affinity failure after launching a valid worker. It
    // does not inject allocation exhaustion or std::thread creation failure.
}

void check_negative_comparator(Counts& counts) {
    const Limbs zero{};
    require(same_bytes(zero, zero), "comparator rejects equal value");
    for (unsigned bit = 0; bit < 256; ++bit) {
        Limbs corrupted{};
        corrupted[bit / 64] = Word{1} << (bit % 64);
        require(!same_bytes(zero, corrupted) && !same_bytes(corrupted, zero),
                "full-byte comparator missed deliberate result corruption");
        ++counts.negative_comparator_checks;
    }
    Limbs pattern{};
    Bytes expected{};
    for (unsigned byte = 0; byte < 32; ++byte) {
        expected[byte] = static_cast<std::uint8_t>(byte + 1);
        pattern[byte / 8] |= Word{byte + 1} << (8 * (byte % 8));
    }
    require(independent_le(pattern) == expected && pa_modn::encode_le(pattern) == expected,
            "explicit LE byte-order fixture failed");
}

void print_cpu_array(const std::vector<int>& cpus) {
    std::cout << '[';
    for (std::size_t i = 0; i < cpus.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << cpus[i];
    }
    std::cout << ']';
}

} // namespace

int main(int argc, char**) {
    if (argc != 1) {
        std::cerr << "usage: test_f1_reduce (no arguments)\n";
        return 2;
    }
    try {
        Counts c;
        const Affinity affinity = discover_affinity();
        check_negative_comparator(c);
        check_errors_and_identity(affinity, c);
        check_affinity_failure(affinity, c);
        Word partition_random = seed ^ UINT64_C(0xa0761d6478bd642f);
        check_boundaries(affinity, partition_random, c);
        check_targeted(affinity, partition_random, c);
        Word random = seed;
        for (Word i = 0; i < random_case_count; ++i) {
            std::size_t count = static_cast<std::size_t>(next_random(random) % 258);
            if (i % 64 == 0) count = 1024 + static_cast<std::size_t>(next_random(random) % 1024);
            Limbs x0 = random_canonical(random);
            if (i % 16 == 0) x0 = {};
            if (i % 16 == 1) x0 = to_limbs(modulus - 1);
            std::vector<Limbs> rhs(count);
            for (auto& value : rhs) value = random_canonical(random);
            check_case(x0, rhs, i % 256 == 0, affinity, partition_random, c);
        }
        require(c.fixture_cases == c.boundary_cases + c.targeted_cases + random_case_count,
                "fixture counts do not reconcile");
        require(c.reference_replay_additions == c.total_rhs_values,
                "library replay count does not reconcile");
        std::cout << "{\n"
                  << "  \"schema\": \"pa_f1_reduce_correctness_v1\",\n"
                  << "  \"status\": \"pass\",\n"
                  << "  \"seed\": " << seed << ",\n"
                  << "  \"fixture_cases\": " << c.fixture_cases << ",\n"
                  << "  \"fixture_counts_are_unique_inputs\": false,\n"
                  << "  \"boundary_cases\": " << c.boundary_cases << ",\n"
                  << "  \"targeted_cases\": " << c.targeted_cases << ",\n"
                  << "  \"random_cases\": " << random_case_count << ",\n"
                  << "  \"total_rhs_values\": " << c.total_rhs_values << ",\n"
                  << "  \"empty_cases\": " << c.empty_cases << ",\n"
                  << "  \"reference_replay_additions\": " << c.reference_replay_additions << ",\n"
                  << "  \"reference_final_checks\": " << c.reference_final_checks << ",\n"
                  << "  \"serial_final_checks\": " << c.serial_final_checks << ",\n"
                  << "  \"chunk4_final_checks\": " << c.chunk4_final_checks << ",\n"
                  << "  \"tree_final_checks\": " << c.tree_final_checks << ",\n"
                  << "  \"partition_final_checks\": " << c.partition_final_checks << ",\n"
                  << "  \"partition_kernel_calls\": " << c.partition_kernel_calls << ",\n"
                  << "  \"result_canonical_checks\": " << c.result_canonical_checks << ",\n"
                  << "  \"encoder_checks\": " << c.encoder_checks << ",\n"
                  << "  \"public_be_checks\": " << c.public_be_checks << ",\n"
                  << "  \"preserved_input_values\": " << c.preserved_input_values << ",\n"
                  << "  \"scratch_canary_checks\": " << c.scratch_canary_checks << ",\n"
                  << "  \"multicore_fixture_subset\": " << c.multicore_fixture_subset << ",\n"
                  << "  \"multicore_calls\": " << c.multicore_calls << ",\n"
                  << "  \"multicore_contract_selected_workers\": " << c.multicore_contract_selected_workers << ",\n"
                  << "  \"selected_worker_count_is_thread_instrumentation\": false,\n"
                  << "  \"multicore_list_larger_than_input_calls\": " << c.multicore_list_larger_than_input_calls << ",\n"
                  << "  \"multicore_uneven_chunk_calls\": " << c.multicore_uneven_chunk_calls << ",\n"
                  << "  \"main_affinity_preservation_checks\": " << c.main_affinity_preservation_checks << ",\n"
                  << "  \"affinity_probe_calls\": " << c.affinity_probe_calls << ",\n"
                  << "  \"affinity_probe_cpu\": " << c.affinity_probe_cpu << ",\n"
                  << "  \"affinity_probe_status\": " << c.affinity_probe_status << ",\n"
                  << "  \"affinity_failure_calls\": " << c.affinity_failure_calls << ",\n"
                  << "  \"affinity_failure_threads_before\": " << c.affinity_failure_threads_before << ",\n"
                  << "  \"affinity_failure_threads_after\": " << c.affinity_failure_threads_after << ",\n"
                  << "  \"thread_creation_failure_injected\": false,\n"
                  << "  \"observed_allowed_cpus\": ";
        print_cpu_array(affinity.allowed);
        std::cout << ",\n  \"reserved_cpu_count\": " << affinity.reserved
                  << ",\n  \"multicore_cpu_lists\": [";
        for (std::size_t i = 0; i < affinity.lists.size(); ++i) {
            if (i) std::cout << ',';
            print_cpu_array(affinity.lists[i]);
        }
        std::cout << "],\n"
                  << "  \"checked_error_cases\": " << c.checked_error_cases << ",\n"
                  << "  \"empty_identity_gate_checks\": " << c.empty_identity_gate_checks << ",\n"
                  << "  \"negative_comparator_checks\": " << c.negative_comparator_checks << ",\n"
                  << "  \"oracle\": \"boost::multiprecision::cpp_int_final_sum_mod_independent_n\",\n"
                  << "  \"reference\": \"linked_unchanged_fast::Scalar_sequential_operator+=\",\n"
                  << "  \"comparison\": \"all_32_explicit_LE_bytes_and_reference_public_BE_bytes\",\n"
                  << "  \"checksum_excludes_environment_dependent_multicore_repetitions\": true,\n"
                  << "  \"result_checksum_fnv1a64\": \"" << std::hex << std::setw(16)
                  << std::setfill('0') << c.checksum << std::dec << "\",\n"
                  << "  \"mismatches\": 0,\n"
                  << "  \"negative_controls\": \"pass\",\n"
                  << "  \"timing_claim\": false,\n"
                  << "  \"constant_time_claim\": false\n}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "F1 correctness failure: " << error.what() << '\n';
        return 1;
    }
}
