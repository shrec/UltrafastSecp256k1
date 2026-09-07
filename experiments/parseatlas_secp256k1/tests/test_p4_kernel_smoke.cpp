// Scheduling/byte smoke against the frozen FE64 API, not an independent proof.
#include "../probes/p4_field_sum_kernels.hpp"
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using pa_p4::FE;
std::uint64_t checks = 0;
std::uint64_t checksum = UINT64_C(14695981039346656037);

FE reference(const FE& seed, const FE* rhs, std::size_t count) {
    FE result = seed;
    for (std::size_t i = 0; i < count; ++i) result = result + rhs[i];
    return result;
}

void check(const FE& actual, const FE& expected, const char* route,
           unsigned pattern, std::size_t count, unsigned seed) {
    const auto bytes = actual.to_bytes();
    FE parsed;
    if (!FE::parse_bytes_strict(bytes, parsed) || actual.limbs() != expected.limbs() ||
        bytes != expected.to_bytes())
        throw std::runtime_error(std::string(route) + " mismatch pattern=" +
            std::to_string(pattern) + " N=" + std::to_string(count) +
            " seed=" + std::to_string(seed));
    for (auto byte : bytes) { checksum ^= byte; checksum *= UINT64_C(1099511628211); }
    ++checks;
}

template<std::size_t Chunk, std::size_t Lanes>
void compare(const FE& seed, const FE* rhs, std::size_t count,
             const FE& expected, unsigned pattern, unsigned seed_index) {
    check(pa_p4::sum_fe64_wide<Chunk, Lanes>(seed, rhs, count), expected,
          "wide", pattern, count, seed_index);
}

void schedules(const FE& seed, const FE* rhs, std::size_t count,
               const FE& expected, unsigned pattern, unsigned seed_index) {
    check(pa_p4::sum_fe64_inline(seed, rhs, count), expected,
          "inline", pattern, count, seed_index);
    compare<1, 1>(seed, rhs, count, expected, pattern, seed_index);
    compare<16, 1>(seed, rhs, count, expected, pattern, seed_index);
    compare<4094, 1>(seed, rhs, count, expected, pattern, seed_index);
    compare<4095, 1>(seed, rhs, count, expected, pattern, seed_index);
    compare<1, 4>(seed, rhs, count, expected, pattern, seed_index);
    compare<16, 4>(seed, rhs, count, expected, pattern, seed_index);
    compare<4094, 4>(seed, rhs, count, expected, pattern, seed_index);
    compare<4095, 4>(seed, rhs, count, expected, pattern, seed_index);
}
} // namespace

int main() {
    try {
        const FE zero = FE::zero(), one = FE::one();
        const FE p_minus_one = FE::from_limbs(
            {UINT64_C(0xfffffffefffffc2e), UINT64_MAX, UINT64_MAX, UINT64_MAX});
        const std::array<FE, 4> seeds{zero, one, p_minus_one, FE::from_uint64(1234567)};
        constexpr std::array<std::size_t, 19> lengths{
            0, 1, 2, 3, 4, 5, 15, 16, 17, 255, 256, 257,
            4093, 4094, 4095, 4096, 8191, 8192, 8193};
        std::vector<FE> rhs(lengths.back());
        for (unsigned pattern = 0; pattern < 5; ++pattern) {
            for (std::size_t i = 0; i < rhs.size(); ++i) {
                if (pattern == 0) rhs[i] = zero;
                else if (pattern == 1) rhs[i] = one;
                else if (pattern == 2) rhs[i] = p_minus_one;
                else if (pattern == 3) rhs[i] = i % 2 ? one : p_minus_one;
                else rhs[i] = FE::from_uint64(static_cast<std::uint64_t>(i) + 1);
            }
            const auto original = rhs;
            for (unsigned s = 0; s < seeds.size(); ++s) for (const auto count : lengths) {
                const FE* input = count ? rhs.data() : nullptr;
                const auto expected = reference(seeds[s], input, count);
                if (count == 0) check(expected, seeds[s], "identity", pattern, count, s);
                if (count == 1) check(expected, seeds[s] + rhs[0], "one_rhs", pattern, count, s);
                schedules(seeds[s], input, count, expected, pattern, s);
            }
            for (std::size_t i = 0; i < rhs.size(); ++i)
                if (rhs[i].limbs() != original[i].limbs())
                    throw std::runtime_error("RHS input mutated");
        }
        for (const auto index : {std::size_t{0}, rhs.size() - 1}) {
            const FE expected = reference(rhs[index], rhs.data(), rhs.size());
            schedules(rhs[index], rhs.data(), rhs.size(), expected, 5, static_cast<unsigned>(index));
        }
        std::cout << "P4 smoke PASS checks=" << checks << " checksum=" << std::hex << checksum << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "P4 smoke FAIL: " << error.what() << '\n';
        return 1;
    }
}
