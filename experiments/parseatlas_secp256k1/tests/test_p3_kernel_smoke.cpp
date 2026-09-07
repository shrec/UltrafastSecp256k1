// Minimal scheduling smoke, not an independent arithmetic proof or benchmark.
#include "../probes/p3_field_sum_kernels.hpp"
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using pa_p3::FE;
using pa_p3::FE52;
std::uint64_t checks = 0;
std::uint64_t checksum = UINT64_C(14695981039346656037);

void check(const FE& actual, const FE& expected, const char* route,
           unsigned pattern, std::size_t count, unsigned seed) {
    const auto bytes = actual.to_bytes();
    FE parsed;
    if (!FE::parse_bytes_strict(bytes, parsed) || bytes != expected.to_bytes())
        throw std::runtime_error(std::string(route) + " mismatch pattern=" +
            std::to_string(pattern) + " N=" + std::to_string(count) +
            " seed=" + std::to_string(seed));
    for (auto byte : bytes) { checksum ^= byte; checksum *= UINT64_C(1099511628211); }
    ++checks;
}

template<std::size_t Chunk, bool Weak>
void compare(const FE& seed, const FE* rhs, const FE52* rhs52, std::size_t count,
             const FE& expected, unsigned pattern, unsigned seed_index) {
    check(pa_p3::sum_fe52_e2e<Chunk, Weak>(seed, rhs, count), expected,
          "e2e", pattern, count, seed_index);
    check(pa_p3::sum_fe52_resident<Chunk, Weak>(seed, rhs52, count), expected,
          "resident", pattern, count, seed_index);
}

void schedules(const FE& seed, const FE* rhs, const FE52* rhs52,
               std::size_t count, const FE& expected, unsigned pattern, unsigned seed_index) {
    compare<1, false>(seed, rhs, rhs52, count, expected, pattern, seed_index);
    compare<16, false>(seed, rhs, rhs52, count, expected, pattern, seed_index);
    compare<256, false>(seed, rhs, rhs52, count, expected, pattern, seed_index);
    compare<4094, false>(seed, rhs, rhs52, count, expected, pattern, seed_index);
    compare<4095, true>(seed, rhs, rhs52, count, expected, pattern, seed_index);
}
} // namespace

int main() {
    try {
        const FE zero = FE::zero(), one = FE::one();
        const FE p_minus_one = FE::from_limbs(
            {UINT64_C(0xfffffffefffffc2e), UINT64_MAX, UINT64_MAX, UINT64_MAX});
        const std::array<FE, 4> seeds{zero, one, p_minus_one, FE::from_uint64(1234567)};
        constexpr std::array<std::size_t, 16> lengths{
            0, 1, 2, 15, 16, 17, 255, 256, 257, 4093, 4094, 4095, 4096, 8191, 8192, 8193};
        std::vector<FE> rhs(lengths.back());
        std::vector<FE52> rhs52(lengths.back());
        for (unsigned pattern = 0; pattern < 5; ++pattern) {
            for (std::size_t i = 0; i < rhs.size(); ++i) {
                if (pattern == 0) rhs[i] = zero;
                else if (pattern == 1) rhs[i] = one;
                else if (pattern == 2) rhs[i] = p_minus_one;
                else if (pattern == 3) rhs[i] = i % 2 ? one : p_minus_one;
                else rhs[i] = FE::from_uint64(static_cast<std::uint64_t>(i) + 1);
                rhs52[i] = FE52::from_fe(rhs[i]);
            }
            const auto original_rhs = rhs;
            const auto original_rhs52 = rhs52;
            for (unsigned s = 0; s < seeds.size(); ++s) {
                for (const auto count : lengths) {
                    const FE* input = count ? rhs.data() : nullptr;
                    const FE52* input52 = count ? rhs52.data() : nullptr;
                    const auto expected = pa_p3::sum_fe64(seeds[s], input, count);
                    // Identity and one-term cases independently assert seed
                    // counting; the schedule comparisons use the frozen API.
                    if (count == 0) check(expected, seeds[s], "identity", pattern, count, s);
                    if (count == 1) check(expected, seeds[s] + rhs[0], "one_rhs", pattern, count, s);
                    schedules(seeds[s], input, input52, count, expected, pattern, s);
                }
            }
            for (std::size_t i = 0; i < rhs.size(); ++i) {
                if (rhs[i].limbs() != original_rhs[i].limbs())
                    throw std::runtime_error("FE64 input mutated");
                for (unsigned limb = 0; limb < 5; ++limb)
                    if (rhs52[i].n[limb] != original_rhs52[i].n[limb])
                        throw std::runtime_error("FE52 input mutated");
            }
        }
        // x0 may refer to a read-only RHS object; there is no restrict promise.
        const FE aliased_expected = pa_p3::sum_fe64(rhs[0], rhs.data(), rhs.size());
        schedules(rhs[0], rhs.data(), rhs52.data(), rhs.size(), aliased_expected, 5, 0);
        std::cout << "P3 smoke PASS checks=" << checks << " checksum=" << std::hex << checksum << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "P3 smoke FAIL: " << error.what() << '\n';
        return 1;
    }
}
