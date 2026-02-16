// =============================================================================
// UltrafastSecp256k1 Metal — Comprehensive GPU Benchmark
// =============================================================================
// Matches CUDA benchmark format: measures all operations with warmup,
// kernel-only timing, and throughput reporting.
//
// Operations benchmarked:
//   - Field Mul, Field Add, Field Sub, Field Sqr, Field Inv
//   - Point Add, Point Double
//   - Scalar Mul (P×k), Generator Mul (G×k)
//
// Build: cmake --build build -j && ./build/metal/metal_secp256k1_bench_full
// =============================================================================

#include "metal_runtime.h"
#include "host_helpers.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <random>

using namespace secp256k1::metal;

// =============================================================================
// Configuration
// =============================================================================

struct BenchConfig {
    int warmup_iterations  = 3;
    int measure_iterations = 5;
    int batch_size         = 1 << 20;  // 1M elements default
    bool apple7_available  = false;    // Real Apple Silicon?
};

// =============================================================================
// Result
// =============================================================================

struct BenchResult {
    std::string name;
    double time_ms;
    int batch_size;
    double throughput_mops;   // Million ops/s
    double time_per_op_ns;    // Nanoseconds per op
};

// =============================================================================
// Shader source loader (concatenate headers + kernels)
// =============================================================================

static std::string load_shader_source(const std::string& exe_dir) {
    std::vector<std::string> files = {
        exe_dir + "/shaders/secp256k1_field.h",
        exe_dir + "/shaders/secp256k1_point.h",
        exe_dir + "/shaders/secp256k1_bloom.h",
        exe_dir + "/shaders/secp256k1_kernels.metal"
    };

    std::string combined;
    for (const auto& path : files) {
        std::ifstream f(path);
        if (!f.is_open()) {
            std::cerr << "[Metal] Cannot open shader file: " << path << "\n";
            return "";
        }
        std::stringstream ss;
        ss << f.rdbuf();
        combined += ss.str();
        combined += "\n";
    }
    return combined;
}

static std::string get_exe_dir(const char* argv0) {
    std::string path(argv0);
    auto pos = path.find_last_of("/\\");
    if (pos != std::string::npos) return path.substr(0, pos);
    return ".";
}

// =============================================================================
// Random data generation (fixed seed for reproducibility)
// =============================================================================

static void fill_random_field_elements(void* buf, int count, uint32_t seed = 42) {
    uint32_t* data = static_cast<uint32_t*>(buf);
    uint32_t s = seed;
    for (int i = 0; i < count * 8; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        data[i] = s;
    }
    // Ensure < p by clearing top bit of each element's last limb
    for (int i = 0; i < count; i++) {
        data[i * 8 + 7] &= 0x7FFFFFFFu;
    }
}

static void fill_random_scalars(void* buf, int count, uint32_t seed = 123) {
    uint32_t* data = static_cast<uint32_t*>(buf);
    uint32_t s = seed;
    for (int i = 0; i < count * 8; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        data[i] = s;
    }
    // Ensure < N
    for (int i = 0; i < count; i++) {
        data[i * 8 + 7] &= 0x7FFFFFFFu;
    }
}

static void fill_random_jacobian_points(void* buf, int count, uint32_t seed = 456) {
    // JacobianPoint = FieldElement x + FieldElement y + FieldElement z + uint infinity
    // sizeof = 8*4 + 8*4 + 8*4 + 4 = 100 bytes
    // But Metal struct has padding, let's use the exact layout
    uint32_t* data = static_cast<uint32_t*>(buf);
    uint32_t s = seed;
    // JacobianPoint: x(8 limbs) + y(8 limbs) + z(8 limbs) + infinity(1 uint) = 25 uints
    for (int i = 0; i < count; i++) {
        int base = i * 25;
        // x
        for (int j = 0; j < 8; j++) {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            data[base + j] = s;
        }
        data[base + 7] &= 0x7FFFFFFFu;
        // y
        for (int j = 0; j < 8; j++) {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            data[base + 8 + j] = s;
        }
        data[base + 15] &= 0x7FFFFFFFu;
        // z = 1 (affine point — Z=1)
        for (int j = 0; j < 8; j++) {
            data[base + 16 + j] = (j == 0) ? 1 : 0;
        }
        // infinity = 0
        data[base + 24] = 0;
    }
}

// =============================================================================
// Benchmark runner helper
// =============================================================================

// 2-buffer op: r = op(a, b)  — field_mul, field_add, field_sub
static BenchResult bench_field_2buf(MetalRuntime& runtime,
                                     const std::string& kernel_name,
                                     const std::string& display_name,
                                     const BenchConfig& cfg) {
    auto pipeline = runtime.make_pipeline(kernel_name);
    if (!pipeline.valid()) {
        std::cerr << "  " << display_name << ": pipeline creation FAILED\n";
        return {display_name, 0, 0, 0, 0};
    }

    int count = cfg.batch_size;
    auto a_buf = runtime.alloc_buffer(count * 32);
    auto b_buf = runtime.alloc_buffer(count * 32);
    auto r_buf = runtime.alloc_buffer(count * 32);
    auto cnt_buf = runtime.alloc_buffer(4);

    fill_random_field_elements(a_buf.contents(), count, 42);
    fill_random_field_elements(b_buf.contents(), count, 99);
    uint32_t cnt = count;
    cnt_buf.write(&cnt, 1);

    uint32_t tg = pipeline.threadExecutionWidth();
    if (tg == 0) tg = 256;

    std::vector<MetalBuffer*> buffers = {&a_buf, &b_buf, &r_buf, &cnt_buf};

    // Warmup
    for (int i = 0; i < cfg.warmup_iterations; i++)
        runtime.dispatch_sync(pipeline, count, tg, buffers);

    // Measure
    double total_ms = 0;
    for (int i = 0; i < cfg.measure_iterations; i++) {
        runtime.dispatch_sync(pipeline, count, tg, buffers);
        total_ms += runtime.last_kernel_time_ms();
    }

    double avg_ms = total_ms / cfg.measure_iterations;
    double throughput = (double)count / avg_ms / 1000.0;
    double ns_per_op = (avg_ms * 1e6) / count;

    return {display_name, avg_ms, count, throughput, ns_per_op};
}

// 1-buffer op: r = op(a)  — field_sqr, field_inv
static BenchResult bench_field_1buf(MetalRuntime& runtime,
                                     const std::string& kernel_name,
                                     const std::string& display_name,
                                     int count,
                                     const BenchConfig& cfg) {
    auto pipeline = runtime.make_pipeline(kernel_name);
    if (!pipeline.valid()) {
        std::cerr << "  " << display_name << ": pipeline creation FAILED\n";
        return {display_name, 0, 0, 0, 0};
    }

    auto a_buf = runtime.alloc_buffer(count * 32);
    auto r_buf = runtime.alloc_buffer(count * 32);
    auto cnt_buf = runtime.alloc_buffer(4);

    fill_random_field_elements(a_buf.contents(), count, 77);
    uint32_t cnt = count;
    cnt_buf.write(&cnt, 1);

    uint32_t tg = pipeline.threadExecutionWidth();
    if (tg == 0) tg = 256;

    std::vector<MetalBuffer*> buffers = {&a_buf, &r_buf, &cnt_buf};

    // Warmup
    for (int i = 0; i < cfg.warmup_iterations; i++)
        runtime.dispatch_sync(pipeline, count, tg, buffers);

    // Measure
    double total_ms = 0;
    for (int i = 0; i < cfg.measure_iterations; i++) {
        runtime.dispatch_sync(pipeline, count, tg, buffers);
        total_ms += runtime.last_kernel_time_ms();
    }

    double avg_ms = total_ms / cfg.measure_iterations;
    double throughput = (double)count / avg_ms / 1000.0;
    double ns_per_op = (avg_ms * 1e6) / count;

    return {display_name, avg_ms, count, throughput, ns_per_op};
}

// Point-operation benchmark using existing point_add_kernel / point_double_kernel
static BenchResult bench_point_op(MetalRuntime& runtime,
                                   const std::string& kernel_name,
                                   const std::string& display_name,
                                   bool needs_two_inputs,
                                   int count,
                                   const BenchConfig& cfg) {
    auto pipeline = runtime.make_pipeline(kernel_name);
    if (!pipeline.valid()) {
        std::cerr << "  " << display_name << ": pipeline creation FAILED\n";
        return {display_name, 0, 0, 0, 0};
    }

    // JacobianPoint in Metal: x(8u) + y(8u) + z(8u) + infinity(1u) = 25 uints = 100 bytes
    const int point_size = 25 * 4;  // 100 bytes

    auto a_buf = runtime.alloc_buffer(count * point_size);
    auto r_buf = runtime.alloc_buffer(count * point_size);
    auto cnt_buf = runtime.alloc_buffer(4);

    fill_random_jacobian_points(a_buf.contents(), count, 456);

    uint32_t cnt = count;
    cnt_buf.write(&cnt, 1);

    uint32_t tg = pipeline.threadExecutionWidth();
    if (tg == 0) tg = 256;

    std::vector<MetalBuffer*> buffers;

    MetalBuffer b_buf;
    if (needs_two_inputs) {
        b_buf = runtime.alloc_buffer(count * point_size);
        fill_random_jacobian_points(b_buf.contents(), count, 789);
        buffers = {&a_buf, &b_buf, &r_buf, &cnt_buf};
    } else {
        buffers = {&a_buf, &r_buf, &cnt_buf};
    }

    // Warmup
    for (int i = 0; i < cfg.warmup_iterations; i++)
        runtime.dispatch_sync(pipeline, count, tg, buffers);

    // Measure
    double total_ms = 0;
    for (int i = 0; i < cfg.measure_iterations; i++) {
        runtime.dispatch_sync(pipeline, count, tg, buffers);
        total_ms += runtime.last_kernel_time_ms();
    }

    double avg_ms = total_ms / cfg.measure_iterations;
    double throughput = (double)count / avg_ms / 1000.0;
    double ns_per_op = (avg_ms * 1e6) / count;

    return {display_name, avg_ms, count, throughput, ns_per_op};
}

// Scalar multiplication benchmark: scalar_mul_batch or generator_mul_batch
static BenchResult bench_scalar_mul_op(MetalRuntime& runtime,
                                        const std::string& kernel_name,
                                        const std::string& display_name,
                                        bool is_generator_mul,
                                        int count,
                                        const BenchConfig& cfg) {
    auto pipeline = runtime.make_pipeline(kernel_name);
    if (!pipeline.valid()) {
        std::cerr << "  " << display_name << ": pipeline creation FAILED\n";
        return {display_name, 0, 0, 0, 0};
    }

    // AffinePoint: x(8u) + y(8u) = 16 uints = 64 bytes
    // Scalar256: 8 uints = 32 bytes
    const int affine_size = 64;
    const int scalar_size = 32;

    auto scalars_buf = runtime.alloc_buffer(count * scalar_size);
    auto results_buf = runtime.alloc_buffer(count * affine_size);
    auto cnt_buf = runtime.alloc_buffer(4);

    fill_random_scalars(scalars_buf.contents(), count, 555);

    uint32_t cnt = count;
    cnt_buf.write(&cnt, 1);

    uint32_t tg = pipeline.threadExecutionWidth();
    if (tg == 0) tg = 256;

    std::vector<MetalBuffer*> buffers;
    MetalBuffer bases_buf;

    if (!is_generator_mul) {
        // scalar_mul_batch: needs base points too
        bases_buf = runtime.alloc_buffer(count * affine_size);
        // Fill with pseudo-random affine points (just random data — not on curve, but ok for bench)
        fill_random_field_elements(bases_buf.contents(), count * 2, 333);  // x+y = 2 field elements
        buffers = {&bases_buf, &scalars_buf, &results_buf, &cnt_buf};
    } else {
        buffers = {&scalars_buf, &results_buf, &cnt_buf};
    }

    // Warmup
    for (int i = 0; i < cfg.warmup_iterations; i++)
        runtime.dispatch_sync(pipeline, count, tg, buffers);

    // Measure
    double total_ms = 0;
    for (int i = 0; i < cfg.measure_iterations; i++) {
        runtime.dispatch_sync(pipeline, count, tg, buffers);
        total_ms += runtime.last_kernel_time_ms();
    }

    double avg_ms = total_ms / cfg.measure_iterations;
    double throughput = (double)count / avg_ms / 1000.0;
    double ns_per_op = (avg_ms * 1e6) / count;

    return {display_name, avg_ms, count, throughput, ns_per_op};
}

// =============================================================================
// Output formatting (matching CUDA style)
// =============================================================================

static void print_result(const BenchResult& r) {
    std::cout << "  " << std::left << std::setw(22) << r.name << " | ";

    if (r.time_per_op_ns >= 1000000) {
        std::cout << std::right << std::setw(8) << std::fixed << std::setprecision(2)
                  << r.time_per_op_ns / 1000000 << " ms";
    } else if (r.time_per_op_ns >= 1000) {
        std::cout << std::right << std::setw(8) << std::fixed << std::setprecision(2)
                  << r.time_per_op_ns / 1000 << " us";
    } else {
        std::cout << std::right << std::setw(8) << std::fixed << std::setprecision(1)
                  << r.time_per_op_ns << " ns";
    }

    std::cout << " | " << std::right << std::setw(10) << std::fixed << std::setprecision(2)
              << r.throughput_mops << " Mops/s";
    std::cout << " | batch=" << r.batch_size << "\n";
}

static void print_summary_table(const std::vector<BenchResult>& results,
                                 const std::string& device_name) {
    std::cout << "\n============================================================\n";
    std::cout << "  Performance Summary (for README)\n";
    std::cout << "============================================================\n";
    std::cout << "GPU: " << device_name << "\n\n";

    std::cout << "| Operation              | Time/Op    | Throughput   |\n";
    std::cout << "|------------------------|------------|-------------:|\n";

    for (const auto& r : results) {
        if (r.batch_size == 0) continue;  // Skip failed benchmarks

        std::cout << "| " << std::left << std::setw(22) << r.name << " | ";

        if (r.time_per_op_ns >= 1000000) {
            std::cout << std::right << std::setw(8) << std::fixed << std::setprecision(2)
                      << r.time_per_op_ns / 1000000 << " ms";
        } else if (r.time_per_op_ns >= 1000) {
            std::cout << std::right << std::setw(8) << std::fixed << std::setprecision(2)
                      << r.time_per_op_ns / 1000 << " us";
        } else {
            std::cout << std::right << std::setw(8) << std::fixed << std::setprecision(1)
                      << r.time_per_op_ns << " ns";
        }

        std::cout << " | " << std::right << std::setw(10) << std::fixed << std::setprecision(2)
                  << r.throughput_mops << " M/s |\n";
    }
    std::cout << "\n";
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char* argv[]) {
    std::cout << "============================================================\n"
              << "  Secp256k1 Metal GPU Benchmark\n"
              << "============================================================\n\n";

    MetalRuntime runtime;
    if (!runtime.init()) {
        std::cerr << "Failed to initialize Metal runtime\n";
        return 1;
    }

    auto info = runtime.device_info();

    std::cout << "GPU Information:\n"
              << "  Device:           " << info.name << "\n"
              << "  Max buffer:       " << (info.max_buffer_length / (1024*1024)) << " MB\n"
              << "  Recommended VRAM: " << (info.recommended_working_set / (1024*1024)) << " MB\n"
              << "  Max threads/group:" << info.max_threads_per_threadgroup << "\n"
              << "  Apple7+ (M1):     " << (info.supports_family_apple7 ? "Yes" : "No") << "\n"
              << "  Apple8+ (M2):     " << (info.supports_family_apple8 ? "Yes" : "No") << "\n"
              << "  Apple9+ (M3):     " << (info.supports_family_apple9 ? "Yes" : "No") << "\n"
              << "  Unified Memory:   " << (info.unified_memory ? "Yes" : "No") << "\n"
              << "\n";

    // Load shaders
    std::string exe_dir = get_exe_dir(argv[0]);
    std::string metallib_path = exe_dir + "/secp256k1_kernels.metallib";

    bool loaded = runtime.load_library_from_path(metallib_path);
    if (!loaded) {
        std::cout << "[Metal] Compiling shaders from source...\n";
        std::string source = load_shader_source(exe_dir);
        if (source.empty()) {
            std::cerr << "[Metal] ERROR: No shader source available\n";
            return 1;
        }
        loaded = runtime.load_library_from_source(source);
        if (!loaded) {
            std::cerr << "[Metal] ERROR: Shader compilation failed\n";
            return 1;
        }
        std::cout << "[Metal] Shaders compiled successfully\n";
    }

    // Parse config
    BenchConfig cfg;
    cfg.apple7_available = info.supports_family_apple7;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--batch" && i + 1 < argc) {
            cfg.batch_size = std::atoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            cfg.warmup_iterations = std::atoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            cfg.measure_iterations = std::atoi(argv[++i]);
        }
    }

    std::cout << "Benchmark Configuration:\n"
              << "  Batch Size:    " << cfg.batch_size << "\n"
              << "  Warmup Iters:  " << cfg.warmup_iterations << "\n"
              << "  Measure Iters: " << cfg.measure_iterations << "\n"
              << "\n";

    std::vector<BenchResult> results;

    // =========================================================================
    // Field Arithmetic
    // =========================================================================
    std::cout << "=== Field Arithmetic ===\n";

    results.push_back(bench_field_2buf(runtime, "field_mul_bench", "Field Mul", cfg));
    print_result(results.back());

    results.push_back(bench_field_2buf(runtime, "field_add_bench", "Field Add", cfg));
    print_result(results.back());

    results.push_back(bench_field_2buf(runtime, "field_sub_bench", "Field Sub", cfg));
    print_result(results.back());

    results.push_back(bench_field_1buf(runtime, "field_sqr_bench", "Field Sqr", cfg.batch_size, cfg));
    print_result(results.back());

    // Inverse — use smaller batch (very expensive per op)
    {
        int inv_batch = std::min(cfg.batch_size, 1 << 16);
        results.push_back(bench_field_1buf(runtime, "field_inv_bench", "Field Inverse", inv_batch, cfg));
        print_result(results.back());
    }

    // =========================================================================
    // Point Operations (require Apple7+ for complex kernels)
    // =========================================================================
    std::cout << "\n=== Point Operations ===\n";

    if (cfg.apple7_available) {
        int point_batch = std::min(cfg.batch_size, 1 << 18);  // Max 256K

        results.push_back(bench_point_op(runtime, "point_add_kernel", "Point Add",
                                          true, point_batch, cfg));
        print_result(results.back());

        results.push_back(bench_point_op(runtime, "point_double_kernel", "Point Double",
                                          false, point_batch, cfg));
        print_result(results.back());
    } else {
        std::cout << "  SKIP: Point operations require Apple7+ GPU (M1 or later)\n";
    }

    // =========================================================================
    // Scalar Multiplication
    // =========================================================================
    std::cout << "\n=== Scalar Multiplication ===\n";

    if (cfg.apple7_available) {
        int scmul_batch = std::min(cfg.batch_size, 1 << 16);  // Max 64K

        results.push_back(bench_scalar_mul_op(runtime, "scalar_mul_batch", "Scalar Mul (P*k)",
                                               false, scmul_batch, cfg));
        print_result(results.back());

        int genmul_batch = std::min(cfg.batch_size, 1 << 17);  // Max 128K
        results.push_back(bench_scalar_mul_op(runtime, "generator_mul_batch", "Generator Mul (G*k)",
                                               true, genmul_batch, cfg));
        print_result(results.back());
    } else {
        std::cout << "  SKIP: Scalar mul requires Apple7+ GPU (M1 or later)\n";
    }

    // =========================================================================
    // Summary Table
    // =========================================================================
    print_summary_table(results, info.name);

    std::cout << "Benchmark complete.\n";
    return 0;
}
