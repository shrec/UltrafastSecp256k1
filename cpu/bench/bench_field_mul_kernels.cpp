#include <secp256k1/field_asm.hpp>
#include <secp256k1/field.hpp>
#include <secp256k1/selftest.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <iomanip>
#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif

using namespace secp256k1::fast;

// ASM kernel only available on MSVC x64 (removed support)
// Clang/GCC use BMI2 intrinsics exclusively
#define HAS_ASM_KERNEL 0


struct Pair {
    uint64_t a[4];
    uint64_t b[4];
};

int main() {
    // Validate arithmetic correctness before benchmarking
    std::cout << "Running arithmetic validation...\n";
    secp256k1::fast::Selftest(true);
    std::cout << "\n";
    
    const size_t elements = 1024;
    const size_t iters = 200000; // total multiplies per pass
    const int passes = 5;        // median-of-5 for stability

#if defined(_WIN32)
    // Optional: pin to CPU 0 and elevate priority for more stable timings
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4996)
#endif
    if (const char* v = std::getenv("BENCH_PIN_AFFINITY")) {
        if (v[0] == '1' || v[0] == 'T' || v[0] == 't' || v[0] == 'Y' || v[0] == 'y') {
            SetThreadAffinityMask(GetCurrentThread(), 1ULL);
            SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
        }
    }
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
#endif

    // Prepare random inputs
    std::mt19937_64 rng(123456789ULL);
    std::uniform_int_distribution<uint64_t> dist;
    std::vector<Pair> vec(elements);
    for (auto& p : vec) {
        for (int i = 0; i < 4; ++i) { p.a[i] = dist(rng); p.b[i] = dist(rng); }
    }

    // Buffers
    uint64_t wide[8];

    std::cout << "\nField mul kernels benchmark (mul kernel + same reduction)\n";
    std::cout << "CPU: BMI2=" << (has_bmi2_support() ? "yes" : "no")
              << ", ASM=" << (HAS_ASM_KERNEL ? "yes" : "no") << "\n\n";

    // Warmup
    for (std::size_t i = 0; i < 1000; ++i) {
        const Pair& p = vec[i % elements];
        detail::mul_4x4_bmi2(p.a, p.b, wide);
        detail::montgomery_reduce_bmi2(wide);
    }

    // BMI2 kernel timing (median-of-N)
    std::vector<double> bmi2_runs;
    for (int pass = 0; pass < passes; ++pass) {
        auto t0 = std::chrono::high_resolution_clock::now();
        size_t idx = 0;
        for (size_t i = 0; i < iters; ++i) {
            const Pair& p = vec[idx++ & (elements - 1)];
            detail::mul_4x4_bmi2(p.a, p.b, wide);
            detail::montgomery_reduce_bmi2(wide);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double const ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;
        bmi2_runs.push_back(ns);
    }
    std::sort(bmi2_runs.begin(), bmi2_runs.end());
    double const ns_bmi2 = bmi2_runs[passes / 2];
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "BMI2 kernel:        " << ns_bmi2 << " ns/op (min/med/max: "
              << bmi2_runs.front() << "/" << ns_bmi2 << "/" << bmi2_runs.back() << ")\n";

    // ASM kernel removed (MSVC-specific, Clang/GCC use BMI2 intrinsics exclusively)
    std::cout << "ASM kernel:         N/A (MSVC-only, removed)\n";

    return 0;
}

