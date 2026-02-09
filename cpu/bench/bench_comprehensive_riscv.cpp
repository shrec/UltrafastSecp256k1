/**
 * Cross-Platform Comprehensive Benchmark
 *
 * Measures all critical operations:
 * - Field operations (mul, square, add, sub, invert)
 * - Scalar operations
 * - Point operations (add, double, scalar_mul)
 * - Batch operations
 *
 * Reports results suitable for GitHub README
 * Detects platform/architecture automatically
 */

#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <vector>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <sstream>
#include <functional>
#include "secp256k1/fast.hpp"
#include "secp256k1/selftest.hpp"

using namespace secp256k1::fast;

// Platform detection helpers
namespace platform_detect {
    inline std::string get_arch() {
        #if defined(__riscv) && __riscv_xlen == 64
            return "RISC-V 64-bit";
        #elif defined(__x86_64__) || defined(_M_X64)
            return "x86_64";
        #elif defined(__aarch64__) || defined(_M_ARM64)
            return "ARM64";
        #elif defined(__i386__) || defined(_M_IX86)
            return "x86 32-bit";
        #else
            return "Unknown";
        #endif
    }

    inline std::string get_os() {
        #ifdef _WIN32
            return "Windows";
        #elif defined(__linux__)
            return "Linux";
        #elif defined(__APPLE__)
            return "macOS";
        #elif defined(__FreeBSD__)
            return "FreeBSD";
        #else
            return "Unknown";
        #endif
    }

    inline std::string get_compiler() {
        #if defined(__clang__)
            return std::string("Clang ") + __clang_version__;
        #elif defined(__GNUC__)
            return std::string("GCC ") + __VERSION__;
        #elif defined(_MSC_VER)
            return std::string("MSVC ") + std::to_string(_MSC_VER);
        #else
            return "Unknown";
        #endif
    }

    inline std::string get_asm_status() {
        #ifdef SECP256K1_HAS_ASM
            #if defined(__riscv)
                return "RISC-V Assembly";
            #elif defined(__x86_64__) || defined(_M_X64)
                return "x86_64 Assembly";
            #else
                return "Assembly Enabled";
            #endif
        #else
            return "Portable C++";
        #endif
    }

    inline std::string get_simd_status() {
        #ifdef SECP256K1_RISCV_USE_VECTOR
            return "RVV";
        #elif defined(__AVX2__)
            return "AVX2";
        #elif defined(__AVX__)
            return "AVX";
        #elif defined(__SSE4_2__)
            return "SSE4.2";
        #elif defined(__ARM_NEON)
            return "NEON";
        #else
            return "None";
        #endif
    }

    inline std::string get_build_type() {
        #ifdef NDEBUG
            return "Release";
        #else
            return "Debug";
        #endif
    }

    inline std::string get_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }
}

// Utility to format time
std::string format_time(double ns) {
    if (ns < 1000) {
        return std::to_string((int)ns) + " ns";
    } else if (ns < 1000000) {
        return std::to_string((int)(ns / 1000)) + " μs";
    } else {
        return std::to_string((int)(ns / 1000000)) + " ms";
    }
}

// Generate random FieldElements
std::vector<FieldElement> generate_random_fields(size_t count) {
    std::vector<FieldElement> result;
    result.reserve(count);

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist;

    for (size_t i = 0; i < count; ++i) {
        std::array<uint64_t, 4> limbs;
        for (auto& limb : limbs) {
            limb = dist(gen);
        }
        result.push_back(FieldElement::from_limbs(limbs));
    }

    return result;
}

// Generate random scalars
std::vector<Scalar> generate_random_scalars(size_t count) {
    std::vector<Scalar> result;
    result.reserve(count);

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist;

    for (size_t i = 0; i < count; ++i) {
        std::array<uint8_t, 32> bytes;
        for (size_t j = 0; j < 32; j += 8) {
            uint64_t val = dist(gen);
            std::memcpy(&bytes[j], &val, std::min(size_t(8), 32 - j));
        }
        result.push_back(Scalar::from_bytes(bytes));
    }

    return result;
}

static void warmup_benchmark(const std::vector<FieldElement>& fields,
                             const std::vector<Scalar>& scalars) {
    constexpr size_t kFieldWarmIters = 2048;
    constexpr size_t kPointWarmIters = 256;
    constexpr size_t kScalarWarmIters = 32;

    FieldElement fe = fields[0];
    for (size_t i = 0; i < kFieldWarmIters; ++i) {
        const auto& other = fields[i % fields.size()];
        fe = (fe * other).square();
        fe = fe + other;
        fe = fe - other;
    }
    for (size_t i = 0; i < 8; ++i) {
        volatile auto inv = fields[i % fields.size()].inverse();
        (void)inv;
    }

    Point G = Point::generator();
    Point P = G.scalar_mul(Scalar::from_uint64(7));
    Point Q = G.scalar_mul(Scalar::from_uint64(11));
    for (size_t i = 0; i < kPointWarmIters; ++i) {
        P = P.add(Q);
        P = P.dbl();
    }
    for (size_t i = 0; i < kScalarWarmIters; ++i) {
        P = Q.scalar_mul(scalars[i % scalars.size()]);
    }

    std::array<FieldElement, 32> batch{};
    for (size_t i = 0; i < batch.size(); ++i) {
        batch[i] = fields[i % fields.size()];
    }
    fe_batch_inverse(batch.data(), batch.size());

    volatile auto prevent_opt = fe.limbs()[0] ^ P.x_raw().limbs()[0] ^ batch[0].limbs()[0];
    (void)prevent_opt;
}

// Helper to run benchmark with warmup and median filtering
template <typename Func>
double measure_with_warmup(Func&& func, int warmup_runs, int measure_runs) {
    std::vector<double> timings;
    timings.reserve(measure_runs);

    // Warmup runs (discard result)
    for (int i = 0; i < warmup_runs; ++i) {
        func();
    }

    // Measurement runs
    for (int i = 0; i < measure_runs; ++i) {
        timings.push_back(func());
    }

    // Sort to find median
    std::sort(timings.begin(), timings.end());

    if (measure_runs % 2 == 0) {
        return (timings[measure_runs / 2 - 1] + timings[measure_runs / 2]) / 2.0;
    } else {
        return timings[measure_runs / 2];
    }
}

// ============================================================
// FIELD OPERATION BENCHMARKS
// ============================================================

double bench_field_mul(const std::vector<FieldElement>& elements, size_t iterations) {
    FieldElement result = elements[0];

    // Warmup
    for (size_t i = 0; i < 1000; ++i) {
        result = result * elements[i % elements.size()];
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < iterations; ++i) {
        size_t idx = i % elements.size();
        result = result * elements[idx];
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    volatile auto prevent_opt = result.limbs()[0];
    (void)prevent_opt;

    return static_cast<double>(duration.count()) / iterations;
}

double bench_field_square(const std::vector<FieldElement>& elements, size_t iterations) {
    FieldElement result = elements[0];

    // Warmup
    for (size_t i = 0; i < 1000; ++i) {
        result = result.square();
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < iterations; ++i) {
        result = result.square();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    volatile auto prevent_opt = result.limbs()[0];
    (void)prevent_opt;

    return static_cast<double>(duration.count()) / iterations;
}

double bench_field_add(const std::vector<FieldElement>& elements, size_t iterations) {
    FieldElement result = elements[0];

    // Warmup
    for (size_t i = 0; i < 1000; ++i) {
        result = result + elements[i % elements.size()];
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < iterations; ++i) {
        size_t idx = i % elements.size();
        result = result + elements[idx];
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    volatile auto prevent_opt = result.limbs()[0];
    (void)prevent_opt;

    return static_cast<double>(duration.count()) / iterations;
}

double bench_field_sub(const std::vector<FieldElement>& elements, size_t iterations) {
    FieldElement result = elements[0];

    // Warmup
    for (size_t i = 0; i < 1000; ++i) {
        result = result - elements[i % elements.size()];
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < iterations; ++i) {
        size_t idx = i % elements.size();
        result = result - elements[idx];
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    volatile auto prevent_opt = result.limbs()[0];
    (void)prevent_opt;

    return static_cast<double>(duration.count()) / iterations;
}

double bench_field_inverse(const std::vector<FieldElement>& elements, size_t iterations) {
    std::vector<FieldElement> results;
    results.reserve(iterations);

    // Warmup
    for (size_t i = 0; i < 10; ++i) {
        volatile auto r = elements[i % elements.size()].inverse();
        (void)r;
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < iterations; ++i) {
        results.push_back(elements[i % elements.size()].inverse());
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    volatile auto prevent_opt = results[0].limbs()[0];
    (void)prevent_opt;

    return static_cast<double>(duration.count()) / iterations;
}

// ============================================================
// POINT OPERATION BENCHMARKS
// ============================================================

double bench_point_add(size_t iterations) {
    Point G = Point::generator();
    Point P = G.scalar_mul(Scalar::from_uint64(7));
    Point Q = G.scalar_mul(Scalar::from_uint64(11));
    Point result = P;

    // Warmup
    for (size_t i = 0; i < 1000; ++i) {
        result = result.add(Q);
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < iterations; ++i) {
        result = result.add(Q);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    volatile auto prevent_opt = result.x_raw().limbs()[0];
    (void)prevent_opt;

    return static_cast<double>(duration.count()) / iterations;
}

double bench_point_double(size_t iterations) {
    Point G = Point::generator();
    Point P = G.scalar_mul(Scalar::from_uint64(7));
    Point result = P;

    // Warmup
    for (size_t i = 0; i < 1000; ++i) {
        result = result.dbl();
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < iterations; ++i) {
        result = result.dbl();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    volatile auto prevent_opt = result.x_raw().limbs()[0];
    (void)prevent_opt;

    return static_cast<double>(duration.count()) / iterations;
}

double bench_point_scalar_mul(const std::vector<Scalar>& scalars, size_t iterations) {
    Point G = Point::generator();
    Point Q = G.scalar_mul(Scalar::from_uint64(12345));
    Point result = Q;

    // Warmup
    for (size_t i = 0; i < 10; ++i) {
        result = Q.scalar_mul(scalars[i % scalars.size()]);
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < iterations; ++i) {
        result = Q.scalar_mul(scalars[i % scalars.size()]);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    volatile auto prevent_opt = result.x_raw().limbs()[0];
    (void)prevent_opt;

    return static_cast<double>(duration.count()) / iterations;
}

double bench_generator_mul(const std::vector<Scalar>& scalars, size_t iterations) {
    Point G = Point::generator();
    Point result = G;

    // Warmup
    for (size_t i = 0; i < 10; ++i) {
        result = G.scalar_mul(scalars[i % scalars.size()]);
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < iterations; ++i) {
        result = G.scalar_mul(scalars[i % scalars.size()]);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    volatile auto prevent_opt = result.x_raw().limbs()[0];
    (void)prevent_opt;

    return static_cast<double>(duration.count()) / iterations;
}

// ============================================================
// BATCH OPERATIONS
// ============================================================

double bench_batch_inversion(size_t batch_size) {
    auto fields = generate_random_fields(batch_size);

    // Warmup
    fe_batch_inverse(fields.data(), batch_size);

    auto start = std::chrono::high_resolution_clock::now();

    fe_batch_inverse(fields.data(), batch_size);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    return static_cast<double>(duration.count()) / batch_size;
}

// ============================================================
// MAIN BENCHMARK RUNNER
// ============================================================

int main()
{
    const bool selftest_ok = Selftest(true);
    if (!selftest_ok) {
        std::cerr << "Self-test failed; benchmark aborted.\n";
        return 1;
    }

    std::cout << "============================================================\n";
    std::cout << "  UltrafastSecp256k1 Comprehensive Benchmark\n";
    std::cout << "============================================================\n\n";

    // Platform info
    std::cout << "Platform Information:\n";
    std::cout << "  Architecture: " << platform_detect::get_arch() << "\n";
    std::cout << "  OS:           " << platform_detect::get_os() << "\n";
    std::cout << "  Compiler:     " << platform_detect::get_compiler() << "\n";
    std::cout << "  Build Type:   " << platform_detect::get_build_type() << "\n";
    std::cout << "  Timestamp:    " << platform_detect::get_timestamp() << "\n";
    std::cout << "\n";

    // Optimization info
    std::cout << "Optimization Configuration:\n";
    std::cout << "  Assembly:     " << platform_detect::get_asm_status() << "\n";
    std::cout << "  SIMD:         " << platform_detect::get_simd_status() << "\n";
#ifdef SECP256K1_USE_FAST_REDUCTION
    std::cout << "  Fast Modular Reduction: ENABLED\n";
#else
    std::cout << "  Fast Modular Reduction: DISABLED\n";
#endif
#ifdef SECP256K1_USE_BMI2
    std::cout << "  BMI2/ADX:     ENABLED\n";
#else
    std::cout << "  BMI2/ADX:     DISABLED\n";
#endif
    std::cout << "\n";

    // Generate test data
    const size_t field_count = 1000;
    const size_t scalar_count = 100;

    std::cout << "Generating test data...\n";
    auto fields = generate_random_fields(field_count);
    auto scalars = generate_random_scalars(scalar_count);
    std::cout << "Test data ready.\n\n";

    std::cout << "Warming up benchmark...\n";
    warmup_benchmark(fields, scalars);
    std::cout << "Warm-up done.\n\n";

    constexpr int kBenchWarmupRuns = 1;
    constexpr int kBenchMeasureRuns = 3;
    std::cout << "Benchmark runs: warmup=" << kBenchWarmupRuns
              << ", measure=" << kBenchMeasureRuns
              << " (median)\n\n";

    // Store results for logging
    struct BenchResult {
        std::string name;
        double time_ns;
    };
    std::vector<BenchResult> results;

    // ========== FIELD OPERATIONS ==========
    std::cout << "==============================================\n";
    std::cout << "  FIELD ARITHMETIC OPERATIONS\n";
    std::cout << "==============================================\n";

    {
        const size_t iterations = 100000;
        double time = measure_with_warmup([&]() {
            return bench_field_mul(fields, iterations);
        }, kBenchWarmupRuns, kBenchMeasureRuns);
        results.push_back({"Field Mul", time});
        std::cout << "Field Mul:       " << std::setw(10) << format_time(time) << "\n";
    }

    {
        const size_t iterations = 100000;
        double time = measure_with_warmup([&]() {
            return bench_field_square(fields, iterations);
        }, kBenchWarmupRuns, kBenchMeasureRuns);
        results.push_back({"Field Square", time});
        std::cout << "Field Square:    " << std::setw(10) << format_time(time) << "\n";
    }

    {
        const size_t iterations = 100000;
        double time = measure_with_warmup([&]() {
            return bench_field_add(fields, iterations);
        }, kBenchWarmupRuns, kBenchMeasureRuns);
        results.push_back({"Field Add", time});
        std::cout << "Field Add:       " << std::setw(10) << format_time(time) << "\n";
    }

    {
        const size_t iterations = 100000;
        double time = measure_with_warmup([&]() {
            return bench_field_sub(fields, iterations);
        }, kBenchWarmupRuns, kBenchMeasureRuns);
        results.push_back({"Field Sub", time});
        std::cout << "Field Sub:       " << std::setw(10) << format_time(time) << "\n";
    }

    {
        const size_t iterations = 1000;
        double time = measure_with_warmup([&]() {
            return bench_field_inverse(fields, iterations);
        }, kBenchWarmupRuns, kBenchMeasureRuns);
        results.push_back({"Field Inverse", time});
        std::cout << "Field Inverse:   " << std::setw(10) << format_time(time) << "\n";
    }

    std::cout << "\n";

    // ========== POINT OPERATIONS ==========
    std::cout << "==============================================\n";
    std::cout << "  POINT OPERATIONS\n";
    std::cout << "==============================================\n";

    {
        const size_t iterations = 10000;
        double time = measure_with_warmup([&]() {
            return bench_point_add(iterations);
        }, kBenchWarmupRuns, kBenchMeasureRuns);
        results.push_back({"Point Add", time});
        std::cout << "Point Add:       " << std::setw(10) << format_time(time) << "\n";
    }

    {
        const size_t iterations = 10000;
        double time = measure_with_warmup([&]() {
            return bench_point_double(iterations);
        }, kBenchWarmupRuns, kBenchMeasureRuns);
        results.push_back({"Point Double", time});
        std::cout << "Point Double:    " << std::setw(10) << format_time(time) << "\n";
    }

    {
        const size_t iterations = 100;
        double time = measure_with_warmup([&]() {
            return bench_point_scalar_mul(scalars, iterations);
        }, kBenchWarmupRuns, kBenchMeasureRuns);
        results.push_back({"Point Scalar Mul", time});
        std::cout << "Point Scalar Mul:" << std::setw(10) << format_time(time) << "\n";
    }

    {
        const size_t iterations = 100;
        double time = measure_with_warmup([&]() {
            return bench_generator_mul(scalars, iterations);
        }, kBenchWarmupRuns, kBenchMeasureRuns);
        results.push_back({"Generator Mul", time});
        std::cout << "Generator Mul:   " << std::setw(10) << format_time(time) << "\n";
    }

    std::cout << "\n";

    // ========== BATCH OPERATIONS ==========
    std::cout << "==============================================\n";
    std::cout << "  BATCH OPERATIONS\n";
    std::cout << "==============================================\n";

    {
        const size_t batch_size = 100;
        double time = measure_with_warmup([&]() {
            return bench_batch_inversion(batch_size);
        }, kBenchWarmupRuns, kBenchMeasureRuns);
        results.push_back({"Batch Inverse (n=100)", time});
        std::cout << "Batch Inverse (n=100): " << std::setw(10) << format_time(time) << " per element\n";
    }

    {
        const size_t batch_size = 1000;
        double time = measure_with_warmup([&]() {
            return bench_batch_inversion(batch_size);
        }, kBenchWarmupRuns, kBenchMeasureRuns);
        results.push_back({"Batch Inverse (n=1000)", time});
        std::cout << "Batch Inverse (n=1000):" << std::setw(10) << format_time(time) << " per element\n";
    }

    std::cout << "\n";
    std::cout << "==============================================\n";
    std::cout << "  Benchmark Complete\n";
    std::cout << "==============================================\n";

    // Generate summary table for README
    std::cout << "\n";
    std::cout << "==============================================\n";
    std::cout << "  Performance Summary (for README)\n";
    std::cout << "==============================================\n";
    std::cout << "Platform: " << platform_detect::get_arch() << " / " << platform_detect::get_os() << "\n";
    std::cout << "Compiler: " << platform_detect::get_compiler() << "\n";
    std::cout << "Assembly: " << platform_detect::get_asm_status() << "\n";
    std::cout << "SIMD:     " << platform_detect::get_simd_status() << "\n\n";

    std::cout << "| Operation          | Time         |\n";
    std::cout << "|--------------------|-------------:|\n";
    for (const auto& result : results) {
        std::cout << "| " << std::left << std::setw(18) << result.name
                  << " | " << std::right << std::setw(12) << format_time(result.time_ns) << " |\n";
    }
    std::cout << "\n";

    // Save results to file
    std::string arch_str = platform_detect::get_arch();
    std::string os_str = platform_detect::get_os();

    // Clean filename
    std::replace(arch_str.begin(), arch_str.end(), ' ', '-');
    std::replace(os_str.begin(), os_str.end(), ' ', '-');
    std::transform(arch_str.begin(), arch_str.end(), arch_str.begin(), ::tolower);
    std::transform(os_str.begin(), os_str.end(), os_str.begin(), ::tolower);

    // Generate filename with timestamp
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream filename;
    filename << "benchmark-" << arch_str << "-" << os_str << "-"
             << std::put_time(std::localtime(&time), "%Y%m%d-%H%M%S") << ".txt";

    std::ofstream logfile(filename.str());
    if (logfile.is_open()) {
        logfile << "=================================================================\n";
        logfile << " UltrafastSecp256k1 Benchmark Results\n";
        logfile << "=================================================================\n\n";

        logfile << "Platform Information:\n";
        logfile << "  Architecture: " << platform_detect::get_arch() << "\n";
        logfile << "  OS:           " << platform_detect::get_os() << "\n";
        logfile << "  Compiler:     " << platform_detect::get_compiler() << "\n";
        logfile << "  Build Type:   " << platform_detect::get_build_type() << "\n";
        logfile << "  Timestamp:    " << platform_detect::get_timestamp() << "\n";
        logfile << "\n";

        logfile << "Optimization Configuration:\n";
        logfile << "  Assembly:     " << platform_detect::get_asm_status() << "\n";
        logfile << "  SIMD:         " << platform_detect::get_simd_status() << "\n";
        logfile << "\n";

        logfile << "Benchmark Runs:\n";
        logfile << "  Warmup:  " << kBenchWarmupRuns << "\n";
        logfile << "  Measure: " << kBenchMeasureRuns << " (median)\n\n";

        logfile << "Benchmark Results:\n";
        logfile << "-----------------------------------------------------------------\n";
        logfile << std::left << std::setw(30) << "Operation" << std::right << std::setw(15) << "Time\n";
        logfile << "-----------------------------------------------------------------\n";

        for (const auto& result : results) {
            logfile << std::left << std::setw(30) << result.name
                   << std::right << std::setw(15) << format_time(result.time_ns) << "\n";
        }

        logfile << "=================================================================\n";
        logfile.close();

        std::cout << "\nResults saved to: " << filename.str() << "\n";
    }

    return 0;
}

