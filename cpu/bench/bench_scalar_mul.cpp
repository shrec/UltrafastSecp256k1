/**
 * Benchmark: K*Q Scalar Multiplication with Lazy Reduction
 * 
 * Measures real-world performance of scalar multiplication with lazy reduction
 * enabled in jacobian_add_mixed (hot path for wNAF).
 */

#include <iostream>
#include <random>
#include <cstring>
#include <vector>
#include <iomanip>
#include "secp256k1/fast.hpp"
#include "secp256k1/benchmark_harness.hpp"

using namespace secp256k1::fast;

// Unified harness: 500 warmup, 11 passes, RDTSC on x86, IQR outlier removal
static bench::Harness H(500, 11);

// Generate random scalars
std::vector<Scalar> generate_scalars(size_t count) {
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

// Benchmark K*G (generator multiplication)
double bench_k_times_generator(const std::vector<Scalar>& scalars, size_t iterations) {
    Point G = Point::generator();
    Point result = G;
    size_t idx = 0;
    
    return H.run(static_cast<int>(iterations), [&]() {
        result = G.scalar_mul(scalars[idx % scalars.size()]);
        bench::DoNotOptimize(result);
        ++idx;
    });
}

// Benchmark K*Q (arbitrary point multiplication)
double bench_k_times_point(const std::vector<Scalar>& scalars, size_t iterations) {
    Point const G = Point::generator();
    Point Q = G.scalar_mul(Scalar::from_uint64(12345)); // Arbitrary point
    Point result = Q;
    size_t idx = 0;
    
    return H.run(static_cast<int>(iterations), [&]() {
        result = Q.scalar_mul(scalars[idx % scalars.size()]);
        bench::DoNotOptimize(result);
        ++idx;
    });
}

// Benchmark point addition (used in wNAF loop)
double bench_point_add(size_t iterations) {
    Point const G = Point::generator();
    Point const P = G.scalar_mul(Scalar::from_uint64(7));
    Point Q = G.scalar_mul(Scalar::from_uint64(11));
    Point result = P;
    
    return H.run(static_cast<int>(iterations), [&]() {
        result = result.add(Q);
        bench::DoNotOptimize(result);
    });
}

// Benchmark point doubling
double bench_point_double(size_t iterations) {
    Point const G = Point::generator();
    Point const P = G.scalar_mul(Scalar::from_uint64(7));
    Point result = P;
    
    return H.run(static_cast<int>(iterations), [&]() {
        result = result.dbl();
        bench::DoNotOptimize(result);
    });
}

void print_header() {
    std::cout << "+========================================================+\n";
    std::cout << "|     Scalar Multiplication Benchmark (Lazy Reduction)  |\n";
    std::cout << "+========================================================+\n\n";
}

void print_result(const std::string& name, double ns_per_op) {
    std::cout << std::left << std::setw(30) << name << ": ";
    std::cout << std::right << std::setw(8) << std::fixed << std::setprecision(2);
    std::cout << ns_per_op << " ns";
    
    // Convert to microseconds if large
    if (ns_per_op > 1000.0) {
        std::cout << " (" << std::fixed << std::setprecision(2) << (ns_per_op / 1000.0) << " us)";
    }
    std::cout << "\n";
}

int main() {
    // Validate arithmetic correctness before benchmarking
    std::cout << "Running arithmetic validation...\n";
    secp256k1::fast::Selftest(true);
    std::cout << "\n";
    
    SECP256K1_INIT();  // Run integrity check
    bench::pin_thread_and_elevate();
    
    print_header();
    
    std::cout << "Benchmark Configuration:\n";
    H.print_config();
    std::cout << "\n";
    
    constexpr size_t NUM_SCALARS = 20;
    constexpr size_t SCALAR_MUL_ITERATIONS = 1000;
    constexpr size_t POINT_OP_ITERATIONS = 100000;
    
    std::cout << "Generating " << NUM_SCALARS << " random scalars...\n";
    auto scalars = generate_scalars(NUM_SCALARS);
    
    std::cout << "\nRunning benchmarks...\n\n";
    std::cout << "=======================================================\n";
    
    // Point operations (building blocks)
    double const add_time = bench_point_add(POINT_OP_ITERATIONS);
    print_result("Point Addition", add_time);
    
    double const dbl_time = bench_point_double(POINT_OP_ITERATIONS);
    print_result("Point Doubling", dbl_time);
    
    std::cout << "\n";
    
    // Scalar multiplication (main operations)
    double const kg_time = bench_k_times_generator(scalars, SCALAR_MUL_ITERATIONS);
    print_result("K*G (Generator)", kg_time);
    
    double const kq_time = bench_k_times_point(scalars, SCALAR_MUL_ITERATIONS);
    print_result("K*Q (Arbitrary Point)", kq_time);
    
    std::cout << "=======================================================\n\n";
    
    // Analysis
    std::cout << "Analysis:\n";
    
    // Estimate operations per K*Q
    // wNAF w=4: ~64 point operations (43 additions + 21 doublings on average)
    double const estimated_ops = (43.0 * add_time + 21.0 * dbl_time);
    std::cout << "  * Estimated from point ops: " << std::fixed << std::setprecision(2);
    std::cout << (estimated_ops / 1000.0) << " us\n";
    
    std::cout << "  * Actual K*Q: " << std::fixed << std::setprecision(2);
    std::cout << (kq_time / 1000.0) << " us\n";
    
    double const overhead = ((kq_time - estimated_ops) / kq_time) * 100.0;
    std::cout << "  * Overhead (wNAF, precompute): " << std::fixed << std::setprecision(1);
    std::cout << overhead << "%\n";
    

    return 0;
}
