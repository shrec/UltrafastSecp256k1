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
#include "secp256k1/field_optimal.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/benchmark_harness.hpp"

using namespace secp256k1::fast;

// Unified harness: 500 warmup, 11 passes, RDTSC on x86, IQR outlier removal
static bench::Harness H(500, 11);

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
        std::tm tm_buf{};
#if defined(_WIN32)
        localtime_s(&tm_buf, &time);
#else
        localtime_r(&time, &tm_buf);
#endif
        std::stringstream ss;
        ss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }
}

// Utility to format time
std::string format_time(double ns) {
    if (ns < 1000) {
        return std::to_string((int)ns) + " ns";
    } else if (ns < 1000000) {
        return std::to_string((int)(ns / 1000)) + " us";
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

// Type alias for the optimal field element selected at compile time
using OFE = secp256k1::fast::OptimalFieldElement;

// Convert FieldElements to Optimal type for benchmarking what's actually used
std::vector<OFE> to_optimal_fields(const std::vector<FieldElement>& fields) {
    std::vector<OFE> result;
    result.reserve(fields.size());
    for (const auto& f : fields) {
        result.push_back(secp256k1::fast::to_optimal(f));
}
    return result;
}

static void warmup_benchmark(const std::vector<FieldElement>& fields,
                             const std::vector<OFE>& opt_fields,
                             const std::vector<Scalar>& scalars) {
    constexpr size_t kFieldWarmIters = 2048;
    constexpr size_t kPointWarmIters = 256;
    constexpr size_t kScalarWarmIters = 32;

    // Warm up optimal field ops (what's actually used in point operations)
    OFE ofe = opt_fields[0];
    for (size_t i = 0; i < kFieldWarmIters; ++i) {
        const auto& other = opt_fields[i % opt_fields.size()];
        ofe = (ofe * other).square();
        ofe = ofe + other;
        ofe = ofe + other.negate(1);
    }
    for (size_t i = 0; i < 8; ++i) {
        volatile auto inv = fields[i % fields.size()].inverse();
        (void)inv;
    }

    Point const G = Point::generator();
    Point P = G.scalar_mul(Scalar::from_uint64(7));
    Point const Q = G.scalar_mul(Scalar::from_uint64(11));
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

    volatile auto prevent_opt = secp256k1::fast::from_optimal(ofe).limbs()[0] ^ P.x_raw().limbs()[0] ^ batch[0].limbs()[0];
    (void)prevent_opt;
}

// ============================================================
// FIELD OPERATION BENCHMARKS (uses OptimalFieldElement -- what point ops actually use)
// ============================================================

double bench_field_mul(const std::vector<OFE>& elements, size_t iterations) {
    OFE result = elements[0];
    return H.run(static_cast<int>(iterations), [&]() {
        result = result * elements[1 % elements.size()];
        bench::DoNotOptimize(result);
    });
}

double bench_field_square(const std::vector<OFE>& elements, size_t iterations) {
    OFE result = elements[0];
    return H.run(static_cast<int>(iterations), [&]() {
        result = result.square();
        bench::DoNotOptimize(result);
    });
}

double bench_field_add(const std::vector<OFE>& elements, size_t iterations) {
    OFE result = elements[0];
    return H.run(static_cast<int>(iterations), [&]() {
        result = result + elements[1 % elements.size()];
        bench::DoNotOptimize(result);
    });
}

double bench_field_negate(const std::vector<OFE>& elements, size_t iterations) {
    OFE result = elements[0];
    return H.run(static_cast<int>(iterations), [&]() {
        result = result.negate(1);
        bench::DoNotOptimize(result);
    });
}

double bench_field_inverse(const std::vector<FieldElement>& elements, size_t iterations) {
    size_t idx = 0;
    return H.run(static_cast<int>(iterations), [&]() {
        auto r = elements[idx % elements.size()].inverse();
        bench::DoNotOptimize(r);
        ++idx;
    });
}

// ============================================================
// POINT OPERATION BENCHMARKS
// ============================================================

double bench_point_add(size_t iterations) {
    Point const G = Point::generator();
    Point const P = G.scalar_mul(Scalar::from_uint64(7));
    Point const Q_jac = G.scalar_mul(Scalar::from_uint64(11));
    Point Q = Point::from_affine(Q_jac.x(), Q_jac.y());
    Point result = P;

    return H.run(static_cast<int>(iterations), [&]() {
        result.add_inplace(Q);
        bench::DoNotOptimize(result);
    });
}

double bench_point_double(size_t iterations) {
    Point const G = Point::generator();
    Point const P = G.scalar_mul(Scalar::from_uint64(7));
    Point result = P;

    return H.run(static_cast<int>(iterations), [&]() {
        result.dbl_inplace();
        bench::DoNotOptimize(result);
    });
}

double bench_point_scalar_mul(const std::vector<Scalar>& scalars, size_t iterations) {
    Point const G = Point::generator();
    Point Q = G.scalar_mul(Scalar::from_uint64(12345));
    Point result = Q;
    size_t idx = 0;

    return H.run(static_cast<int>(iterations), [&]() {
        result = Q.scalar_mul(scalars[idx % scalars.size()]);
        bench::DoNotOptimize(result);
        ++idx;
    });
}

double bench_generator_mul(const std::vector<Scalar>& scalars, size_t iterations) {
    Point G = Point::generator();
    Point result = G;
    size_t idx = 0;

    return H.run(static_cast<int>(iterations), [&]() {
        result = G.scalar_mul(scalars[idx % scalars.size()]);
        bench::DoNotOptimize(result);
        ++idx;
    });
}

// ============================================================
// ECDSA & SCHNORR SIGNATURE BENCHMARKS
// ============================================================

double bench_ecdsa_sign(const std::vector<Scalar>& keys,
                        const std::vector<std::array<uint8_t,32>>& msgs,
                        size_t iterations) {
    size_t idx = 0;
    return H.run(static_cast<int>(iterations), [&]() {
        auto sig = secp256k1::ecdsa_sign(msgs[idx % msgs.size()], keys[idx % keys.size()]);
        bench::DoNotOptimize(sig);
        ++idx;
    });
}

double bench_ecdsa_verify(const std::vector<secp256k1::ECDSASignature>& sigs,
                          const std::vector<std::array<uint8_t,32>>& msgs,
                          const std::vector<Point>& pubkeys,
                          size_t iterations) {
    size_t idx = 0;
    return H.run(static_cast<int>(iterations), [&]() {
        bool ok = secp256k1::ecdsa_verify(msgs[idx % msgs.size()], pubkeys[idx % pubkeys.size()], sigs[idx % sigs.size()]);
        bench::DoNotOptimize(ok);
        ++idx;
    });
}

double bench_schnorr_sign(const std::vector<Scalar>& keys,
                          const std::vector<std::array<uint8_t,32>>& msgs,
                          size_t iterations) {
    std::array<uint8_t,32> aux{}; // zero aux for deterministic bench
    size_t idx = 0;
    return H.run(static_cast<int>(iterations), [&]() {
        auto sig = secp256k1::schnorr_sign(keys[idx % keys.size()], msgs[idx % msgs.size()], aux);
        bench::DoNotOptimize(sig);
        ++idx;
    });
}

double bench_schnorr_verify(const std::vector<secp256k1::SchnorrSignature>& sigs,
                            const std::vector<std::array<uint8_t,32>>& msgs,
                            const std::vector<std::array<uint8_t,32>>& xonly_pks,
                            size_t iterations) {
    size_t idx = 0;
    return H.run(static_cast<int>(iterations), [&]() {
        bool ok = secp256k1::schnorr_verify(xonly_pks[idx % xonly_pks.size()], msgs[idx % msgs.size()], sigs[idx % sigs.size()]);
        bench::DoNotOptimize(ok);
        ++idx;
    });
}

// ============================================================
// BATCH OPERATIONS
// ============================================================

double bench_batch_inversion(size_t batch_size) {
    auto fields = generate_random_fields(batch_size);

    return H.run(1, [&]() {
        fe_batch_inverse(fields.data(), batch_size);
        bench::DoNotOptimize(fields[0]);
    }) / static_cast<double>(batch_size);
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
    auto opt_fields = to_optimal_fields(fields);
    std::cout << "Test data ready.\n\n";

    std::cout << "Warming up benchmark...\n";
    warmup_benchmark(fields, opt_fields, scalars);
    bench::pin_thread_and_elevate();
    std::cout << "Warm-up done.\n\n";

    std::cout << "Benchmark Configuration:\n";
    H.print_config();
    std::cout << "\n";

    // Store results for logging
    struct BenchResult {
        std::string name;
        double time_ns;
    };
    std::vector<BenchResult> results;

    // ========== FIELD OPERATIONS ==========
    std::cout << "==============================================\n";
    std::cout << "  FIELD ARITHMETIC OPERATIONS (" << secp256k1::fast::kOptimalTierName << ")\n";
    std::cout << "==============================================\n";

    {
        const size_t iterations = 100000;
        double const time = bench_field_mul(opt_fields, iterations);
        results.push_back({"Field Mul", time});
        std::cout << "Field Mul:       " << std::setw(10) << format_time(time) << "\n";
    }

    {
        const size_t iterations = 100000;
        double const time = bench_field_square(opt_fields, iterations);
        results.push_back({"Field Square", time});
        std::cout << "Field Square:    " << std::setw(10) << format_time(time) << "\n";
    }

    {
        const size_t iterations = 100000;
        double const time = bench_field_add(opt_fields, iterations);
        results.push_back({"Field Add", time});
        std::cout << "Field Add:       " << std::setw(10) << format_time(time) << "\n";
    }

    {
        const size_t iterations = 100000;
        double const time = bench_field_negate(opt_fields, iterations);
        results.push_back({"Field Negate", time});
        std::cout << "Field Negate:    " << std::setw(10) << format_time(time) << "\n";
    }

    {
        const size_t iterations = 1000;
        double const time = bench_field_inverse(fields, iterations);
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
        double const time = bench_point_add(iterations);
        results.push_back({"Point Add", time});
        std::cout << "Point Add:       " << std::setw(10) << format_time(time) << "\n";
    }

    {
        const size_t iterations = 10000;
        double const time = bench_point_double(iterations);
        results.push_back({"Point Double", time});
        std::cout << "Point Double:    " << std::setw(10) << format_time(time) << "\n";
    }

    {
        const size_t iterations = 100;
        double const time = bench_point_scalar_mul(scalars, iterations);
        results.push_back({"Point Scalar Mul", time});
        std::cout << "Point Scalar Mul:" << std::setw(10) << format_time(time) << "\n";
    }

    {
        const size_t iterations = 100;
        double const time = bench_generator_mul(scalars, iterations);
        results.push_back({"Generator Mul", time});
        std::cout << "Generator Mul:   " << std::setw(10) << format_time(time) << "\n";
    }

    std::cout << "\n";

    // ========== ECDSA & SCHNORR ==========
    std::cout << "==============================================\n";
    std::cout << "  ECDSA & SCHNORR SIGNATURES\n";
    std::cout << "==============================================\n";

    // Pre-generate signature test data
    const size_t sig_count = 64;
    std::vector<std::array<uint8_t,32>> msg_hashes(sig_count);
    std::vector<Scalar> sign_keys(sig_count);
    std::vector<Point> sign_pubkeys(sig_count);
    std::vector<secp256k1::ECDSASignature> ecdsa_sigs(sig_count);
    std::vector<secp256k1::SchnorrSignature> schnorr_sigs(sig_count);
    std::vector<std::array<uint8_t,32>> xonly_pks(sig_count);

    {
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<uint64_t> dist;
        Point const G = Point::generator();
        std::array<uint8_t,32> const aux{};

        for (size_t i = 0; i < sig_count; ++i) {
            // random message hash
            for (size_t j = 0; j < 32; j += 8) {
                uint64_t v = dist(gen);
                std::memcpy(&msg_hashes[i][j], &v, std::min(size_t(8), 32 - j));
            }
            // random private key
            std::array<uint8_t,32> kb;
            for (size_t j = 0; j < 32; j += 8) {
                uint64_t v = dist(gen);
                std::memcpy(&kb[j], &v, std::min(size_t(8), 32 - j));
            }
            sign_keys[i] = Scalar::from_bytes(kb);
            sign_pubkeys[i] = G.scalar_mul(sign_keys[i]);
            ecdsa_sigs[i] = secp256k1::ecdsa_sign(msg_hashes[i], sign_keys[i]);
            schnorr_sigs[i] = secp256k1::schnorr_sign(sign_keys[i], msg_hashes[i], aux);
            xonly_pks[i] = secp256k1::schnorr_pubkey(sign_keys[i]);
        }
        std::cout << "Prepared " << sig_count << " key/message/signature tuples.\n";
    }

    {
        const size_t iterations = 100;
        double const time = bench_ecdsa_sign(sign_keys, msg_hashes, iterations);
        results.push_back({"ECDSA Sign", time});
        std::cout << "ECDSA Sign:      " << std::setw(10) << format_time(time) << "\n";
    }

    {
        const size_t iterations = 100;
        double const time = bench_ecdsa_verify(ecdsa_sigs, msg_hashes, sign_pubkeys, iterations);
        results.push_back({"ECDSA Verify", time});
        std::cout << "ECDSA Verify:    " << std::setw(10) << format_time(time) << "\n";
    }

    {
        const size_t iterations = 100;
        double const time = bench_schnorr_sign(sign_keys, msg_hashes, iterations);
        results.push_back({"Schnorr Sign", time});
        std::cout << "Schnorr Sign:    " << std::setw(10) << format_time(time) << "\n";
    }

    {
        const size_t iterations = 100;
        double const time = bench_schnorr_verify(schnorr_sigs, msg_hashes, xonly_pks, iterations);
        results.push_back({"Schnorr Verify", time});
        std::cout << "Schnorr Verify:  " << std::setw(10) << format_time(time) << "\n";
    }

    std::cout << "\n";

    // ========== BATCH OPERATIONS ==========
    std::cout << "==============================================\n";
    std::cout << "  BATCH OPERATIONS\n";
    std::cout << "==============================================\n";

    {
        const size_t batch_size = 100;
        double const time = bench_batch_inversion(batch_size);
        results.push_back({"Batch Inverse (n=100)", time});
        std::cout << "Batch Inverse (n=100): " << std::setw(10) << format_time(time) << " per element\n";
    }

    {
        const size_t batch_size = 1000;
        double const time = bench_batch_inversion(batch_size);
        results.push_back({"Batch Inverse (n=1000)", time});
        std::cout << "Batch Inverse (n=1000):" << std::setw(10) << format_time(time) << " per element\n";
    }

    std::cout << "\n";

    std::cout << "==============================================\n";
    std::cout << "  Benchmark Complete\n";
    std::cout << "  Field Tier: " << secp256k1::fast::kOptimalTierName << "\n";
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
    std::tm tm_buf{};
#if defined(_WIN32)
    localtime_s(&tm_buf, &time);
#else
    localtime_r(&time, &tm_buf);
#endif
    std::stringstream filename;
    filename << "benchmark-" << arch_str << "-" << os_str << "-"
             << std::put_time(&tm_buf, "%Y%m%d-%H%M%S") << ".txt";

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

        logfile << "Benchmark Configuration:\n";
        logfile << "  Timer:   " << bench::Timer::timer_name() << "\n";
        logfile << "  Warmup:  " << H.warmup_iters << "\n";
        logfile << "  Passes:  " << H.passes << " (IQR outlier removal + median)\n\n";

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

