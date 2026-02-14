// =============================================================================
// UltrafastSecp256k1 OpenCL - Benchmark
// =============================================================================

#include "secp256k1_opencl.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <string>
#include <cstdlib>

using namespace secp256k1::opencl;

// Benchmark helper
template<typename F>
double benchmark_ns(F&& func, int iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    return static_cast<double>(duration.count()) / iterations;
}

int main(int argc, char* argv[]) {
    std::cout << "UltrafastSecp256k1 OpenCL Benchmark\n";
    std::cout << "====================================\n\n";

    int platform_id = -1;
    int device_id = 0;
    bool prefer_intel = true;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--platform" && i + 1 < argc) {
            platform_id = std::atoi(argv[++i]);
        } else if (arg == "--device" && i + 1 < argc) {
            device_id = std::atoi(argv[++i]);
        } else if (arg == "--intel") {
            platform_id = -1;
            prefer_intel = true;
        } else if (arg == "--nvidia") {
            platform_id = -1;
            prefer_intel = false;
        }
    }

    // Create context
    DeviceConfig config;
    config.verbose = true;
    config.prefer_intel = prefer_intel;
    if (platform_id >= 0) {
        config.platform_id = platform_id;
        config.device_id = device_id;
    } else {
        config.platform_id = -1;  // auto-selection based on prefer_intel
    }

    auto ctx = Context::create(config);
    if (!ctx) {
        std::cout << "Failed to create OpenCL context\n";
        return 1;
    }

    const auto& info = ctx->device_info();
    std::cout << "\nDevice: " << info.name << " (" << info.vendor << ")\n";
    std::cout << "Compute Units: " << info.compute_units << "\n";
    std::cout << "Global Memory: " << (info.global_mem_size / (1024*1024)) << " MB\n\n";

    // ==========================================================================
    // Field Arithmetic Benchmarks
    // ==========================================================================
    std::cout << "Field Arithmetic:\n";
    std::cout << "-----------------\n";

    {
        FieldElement a = field_from_u64(0x123456789ABCDEFULL);
        FieldElement b = field_from_u64(0xFEDCBA987654321ULL);
        FieldElement c;

        double ns = benchmark_ns([&]() { c = ctx->field_add(a, b); }, 1000);
        std::cout << "  Field Add:      " << std::fixed << std::setprecision(1)
                  << ns << " ns/op\n";

        ns = benchmark_ns([&]() { c = ctx->field_sub(a, b); }, 1000);
        std::cout << "  Field Sub:      " << ns << " ns/op\n";

        ns = benchmark_ns([&]() { c = ctx->field_mul(a, b); }, 1000);
        std::cout << "  Field Mul:      " << ns << " ns/op\n";

        ns = benchmark_ns([&]() { c = ctx->field_sqr(a); }, 1000);
        std::cout << "  Field Sqr:      " << ns << " ns/op\n";

        ns = benchmark_ns([&]() { c = ctx->field_inv(a); }, 100);
        std::cout << "  Field Inv:      " << ns << " ns/op\n";
    }

    // ==========================================================================
    // Point Operation Benchmarks
    // ==========================================================================
    std::cout << "\nPoint Operations:\n";
    std::cout << "-----------------\n";

    {
        Scalar k = scalar_from_u64(12345);
        JacobianPoint p = ctx->scalar_mul_generator(k);
        JacobianPoint q;

        double ns = benchmark_ns([&]() { q = ctx->point_double(p); }, 1000);
        std::cout << "  Point Double:   " << std::fixed << std::setprecision(1)
                  << ns << " ns/op\n";

        ns = benchmark_ns([&]() { q = ctx->point_add(p, p); }, 1000);
        std::cout << "  Point Add:      " << ns << " ns/op\n";

        ns = benchmark_ns([&]() { q = ctx->scalar_mul_generator(k); }, 100);
        std::cout << "  Scalar Mul:     " << ns/1000.0 << " us/op\n";
    }

    // ==========================================================================
    // Batch Operation Benchmarks
    // ==========================================================================
    std::cout << "\nBatch Operations:\n";
    std::cout << "-----------------\n";

    for (std::size_t batch_size : {256, 1024, 4096, 16384}) {
        std::vector<Scalar> scalars(batch_size);
        std::vector<JacobianPoint> results(batch_size);

        for (std::size_t i = 0; i < batch_size; ++i) {
            scalars[i] = scalar_from_u64(i + 1);
        }

        auto start = std::chrono::high_resolution_clock::now();
        ctx->batch_scalar_mul_generator(scalars.data(), results.data(), batch_size);
        ctx->sync();
        auto end = std::chrono::high_resolution_clock::now();

        double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double ops_per_sec = batch_size / (duration_ms / 1000.0);

        std::cout << "  Batch " << std::setw(5) << batch_size << ": "
                  << std::fixed << std::setprecision(2) << duration_ms << " ms ("
                  << std::setprecision(0) << ops_per_sec << " ops/s)\n";
    }

    // ==========================================================================
    // Batch Inversion Benchmark
    // ==========================================================================
    std::cout << "\nBatch Field Inversion:\n";
    std::cout << "----------------------\n";

    for (std::size_t batch_size : {256, 1024, 4096}) {
        std::vector<FieldElement> inputs(batch_size);
        std::vector<FieldElement> outputs(batch_size);

        for (std::size_t i = 0; i < batch_size; ++i) {
            inputs[i] = field_from_u64(i + 3);
        }

        auto start = std::chrono::high_resolution_clock::now();
        ctx->batch_field_inv(inputs.data(), outputs.data(), batch_size);
        ctx->sync();
        auto end = std::chrono::high_resolution_clock::now();

        double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double ops_per_sec = batch_size / (duration_ms / 1000.0);

        std::cout << "  Batch " << std::setw(5) << batch_size << ": "
                  << std::fixed << std::setprecision(2) << duration_ms << " ms ("
                  << std::setprecision(0) << ops_per_sec << " ops/s)\n";
    }

    std::cout << "\nBenchmark complete!\n";
    return 0;
}

