// =============================================================================
// UltrafastSecp256k1 Metal — Test & Benchmark Application
// =============================================================================
// Tests correctness of Metal GPU kernels against known secp256k1 test vectors.
// Benchmarks field operations and point operations on Apple Silicon GPU.
//
// Build: See metal/CMakeLists.txt
// Run:   ./metal_secp256k1_test
// =============================================================================

#include "metal_runtime.h"
#include "host_helpers.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <sstream>

using namespace secp256k1::metal;

// =============================================================================
// Shader Source Loading
// =============================================================================

static std::string load_shader_source(const std::string& exe_dir) {
    // Concatenate all shader headers + kernel into one source
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
            // Try metallib path instead
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
// Test: Generator Multiplication (G × k)
// =============================================================================

static bool test_generator_mul(MetalRuntime& runtime) {
    std::cout << "\n=== Test: Generator Multiplication (G × k) ===\n";

    // scalar_mul kernel (16-entry precompute + field inversion) requires
    // real Apple Silicon GPU.  Paravirtual / non-Apple7+ devices cannot
    // compile the pipeline, so skip gracefully.
    auto info = runtime.device_info();
    if (!info.supports_family_apple7) {
        std::cout << "  SKIP: generator_mul requires Apple7+ GPU (M1 or later)\n"
                  << "  Device '" << info.name << "' does not support this kernel\n";
        return true;  // not a failure — unsupported hardware
    }

    auto pipeline = runtime.make_pipeline("generator_mul_batch");
    if (!pipeline.valid()) {
        std::cerr << "  FAIL: Could not create pipeline\n";
        return false;
    }
    
    // Test vectors: k=1 → G, k=2 → 2G, k=3 → 3G
    const int N = 3;
    auto scalars_buf = runtime.alloc_buffer(N * 32); // Scalar256 = 32 bytes
    auto results_buf = runtime.alloc_buffer(N * 64); // AffinePoint = 64 bytes
    auto count_buf = runtime.alloc_buffer(4);        // uint32_t count
    
    // Fill scalars
    HostScalar scalars[3];
    scalars[0] = HostScalar::from_uint64(1);
    scalars[1] = HostScalar::from_uint64(2);
    scalars[2] = HostScalar::from_uint64(3);
    
    scalars_buf.write(scalars, 3);
    
    uint32_t count = N;
    count_buf.write(&count, 1);
    
    // Zero results
    memset(results_buf.contents(), 0, N * 64);
    
    // Dispatch
    std::vector<MetalBuffer*> buffers = {&scalars_buf, &results_buf, &count_buf};
    runtime.dispatch_sync(pipeline, N, 1, buffers);
    
    std::cout << "  Kernel time: " << runtime.last_kernel_time_ms() << " ms\n";
    
    // Read results
    HostAffinePoint results[3];
    results_buf.read(results, 3);
    
    // Verify against known test vectors
    auto g = generator_point();
    auto g2 = two_g_point();
    auto g3 = three_g_point();
    
    bool pass = true;
    
    if (results[0].x != g.x || results[0].y != g.y) {
        std::cerr << "  FAIL: 1*G mismatch!\n";
        print_point("  Got", results[0]);
        print_point("  Expected", g);
        pass = false;
    } else {
        std::cout << "  PASS: 1*G = G ✓\n";
    }
    
    if (results[1].x != g2.x || results[1].y != g2.y) {
        std::cerr << "  FAIL: 2*G mismatch!\n";
        print_point("  Got", results[1]);
        print_point("  Expected", g2);
        pass = false;
    } else {
        std::cout << "  PASS: 2*G ✓\n";
    }
    
    if (results[2].x != g3.x || results[2].y != g3.y) {
        std::cerr << "  FAIL: 3*G mismatch!\n";
        print_point("  Got", results[2]);
        print_point("  Expected", g3);
        pass = false;
    } else {
        std::cout << "  PASS: 3*G ✓\n";
    }
    
    return pass;
}

// =============================================================================
// Test: Field Multiplication
// =============================================================================

static bool test_field_mul(MetalRuntime& runtime) {
    std::cout << "\n=== Test: Field Multiplication ===\n";
    
    auto pipeline = runtime.make_pipeline("field_mul_bench");
    if (!pipeline.valid()) {
        std::cerr << "  FAIL: Could not create pipeline\n";
        return false;
    }
    
    // Test: a * 1 = a
    const int N = 1;
    auto a_buf = runtime.alloc_buffer(N * 32);
    auto b_buf = runtime.alloc_buffer(N * 32);
    auto r_buf = runtime.alloc_buffer(N * 32);
    auto count_buf = runtime.alloc_buffer(4);
    
    // Generator X as test value
    HostFieldElement gx = HostFieldElement::from_hex(
        "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
    HostFieldElement one = HostFieldElement::from_uint64(1);
    
    a_buf.write(&gx, 1);
    b_buf.write(&one, 1);
    
    uint32_t count = N;
    count_buf.write(&count, 1);
    memset(r_buf.contents(), 0, 32);
    
    std::vector<MetalBuffer*> buffers = {&a_buf, &b_buf, &r_buf, &count_buf};
    runtime.dispatch_sync(pipeline, N, 1, buffers);
    
    HostFieldElement result;
    r_buf.read(&result, 1);
    
    if (result == gx) {
        std::cout << "  PASS: Gx * 1 = Gx ✓\n";
        return true;
    } else {
        std::cerr << "  FAIL: Gx * 1 mismatch!\n";
        print_field("  Got", result);
        print_field("  Expected", gx);
        return false;
    }
}

// =============================================================================
// Benchmark: Field Multiplication Throughput
// =============================================================================

// Quick inline benchmark (field_mul only, for --bench flag)
// For full comprehensive benchmark use metal_secp256k1_bench_full
static void bench_field_mul(MetalRuntime& runtime, int count = 1024 * 1024) {
    std::cout << "\n=== Quick Benchmark: Field Multiplication (" << count << " ops) ===\n";
    
    auto pipeline = runtime.make_pipeline("field_mul_bench");
    if (!pipeline.valid()) { std::cerr << "  Pipeline creation failed\n"; return; }
    
    auto a_buf = runtime.alloc_buffer(count * 32);
    auto b_buf = runtime.alloc_buffer(count * 32);
    auto r_buf = runtime.alloc_buffer(count * 32);
    auto count_buf = runtime.alloc_buffer(4);
    
    uint32_t* a_data = (uint32_t*)a_buf.contents();
    uint32_t* b_data = (uint32_t*)b_buf.contents();
    uint32_t seed = 0x12345678;
    for (int i = 0; i < count * 8; i++) {
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        a_data[i] = seed;
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        b_data[i] = seed;
    }
    
    uint32_t cnt = count;
    count_buf.write(&cnt, 1);
    
    uint32_t tg_size = pipeline.threadExecutionWidth();
    if (tg_size == 0) tg_size = 256;
    
    std::vector<MetalBuffer*> buffers = {&a_buf, &b_buf, &r_buf, &count_buf};
    
    // Warmup
    runtime.dispatch_sync(pipeline, count, tg_size, buffers);
    
    // Timed run
    runtime.dispatch_sync(pipeline, count, tg_size, buffers);
    
    double ms = runtime.last_kernel_time_ms();
    double mops = (double)count / ms / 1000.0;
    
    std::cout << "  Time: " << ms << " ms\n"
              << "  Throughput: " << mops << " M field_mul/s\n"
              << "  (For full benchmark run: metal_secp256k1_bench_full)\n";
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char* argv[]) {
    std::cout << "╔══════════════════════════════════════════════════════════╗\n"
              << "║  UltrafastSecp256k1 — Apple Metal GPU Backend           ║\n"
              << "║  First secp256k1 library with Metal compute support!    ║\n"
              << "╚══════════════════════════════════════════════════════════╝\n";
    
    MetalRuntime runtime;
    
    if (!runtime.init()) {
        std::cerr << "Failed to initialize Metal runtime\n";
        return 1;
    }
    
    runtime.print_device_info();
    
    // Load shaders
    std::string exe_dir = get_exe_dir(argv[0]);
    std::string metallib_path = exe_dir + "/secp256k1_kernels.metallib";
    
    // Try precompiled metallib first, then fall back to runtime compilation
    bool loaded = runtime.load_library_from_path(metallib_path);
    if (!loaded) {
        std::cout << "[Metal] Precompiled .metallib not found, compiling from source...\n";
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
    
    // Run tests
    int failures = 0;
    
    if (!test_field_mul(runtime)) failures++;
    if (!test_generator_mul(runtime)) failures++;
    
    // Run benchmarks (quick — field_mul only)
    bool bench = (argc > 1 && std::string(argv[1]) == "--bench");
    if (bench) {
        bench_field_mul(runtime);
    }
    
    // Summary
    std::cout << "\n=== Summary ===\n";
    if (failures == 0) {
        std::cout << "  All tests PASSED\n";
    } else {
        std::cout << "  " << failures << " test(s) FAILED\n";
    }
    
    if (!bench) {
        std::cout << "  (Run with --bench for quick benchmark)\n"
                  << "  (Run metal_secp256k1_bench_full for comprehensive benchmark)\n";
    }
    
    return failures > 0 ? 1 : 0;
}
