/**
 * CUDA Benchmark for Secp256k1
 *
 * Comprehensive benchmark of all CUDA operations:
 * - Field arithmetic (mul, square, add, sub, inverse)
 * - Point operations (add, double)
 * - Scalar multiplication (batch)
 * - Generator multiplication (batch)
 *
 * Build:
 *   nvcc -O3 -std=c++17 -arch=sm_89 bench_cuda.cu -o bench_cuda
 *
 * Run:
 *   ./bench_cuda
 */

#include "secp256k1.cuh"
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cstring>

using namespace secp256k1::cuda;

// ============================================================================
// Error checking macro
// ============================================================================
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

// ============================================================================
// Benchmark configuration
// ============================================================================
struct BenchConfig {
    int warmup_iterations = 3;
    int measure_iterations = 5;
    int batch_size = 1 << 20;  // 1M elements default
    int threads_per_block = 256;
};

// ============================================================================
// Timer helper
// ============================================================================
class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() {
        CUDA_CHECK(cudaEventRecord(start_));
    }

    float stop() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
};

// ============================================================================
// Random data generation
// ============================================================================
void generate_random_field_elements(FieldElement* h_data, int count) {
    std::mt19937_64 rng(42);  // Fixed seed for reproducibility
    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < 4; ++j) {
            h_data[i].limbs[j] = rng();
        }
        // Ensure < P by clearing top bits if needed
        h_data[i].limbs[3] &= 0x7FFFFFFFFFFFFFFFULL;
    }
}

void generate_random_scalars(Scalar* h_data, int count) {
    std::mt19937_64 rng(123);
    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < 4; ++j) {
            h_data[i].limbs[j] = rng();
        }
        // Ensure < N
        h_data[i].limbs[3] &= 0x7FFFFFFFFFFFFFFFULL;
    }
}

void generate_random_points(JacobianPoint* h_data, int count) {
    std::mt19937_64 rng(456);
    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < 4; ++j) {
            h_data[i].x.limbs[j] = rng();
            h_data[i].y.limbs[j] = rng();
            h_data[i].z.limbs[j] = (j == 0) ? 1 : 0;  // Z = 1 (affine)
        }
        h_data[i].x.limbs[3] &= 0x7FFFFFFFFFFFFFFFULL;
        h_data[i].y.limbs[3] &= 0x7FFFFFFFFFFFFFFFULL;
        h_data[i].infinity = false;
    }
}

// ============================================================================
// Benchmark functions
// ============================================================================

struct BenchResult {
    const char* name;
    double time_ms;
    int batch_size;
    double throughput_mops;  // Million ops/sec
    double time_per_op_ns;   // Nanoseconds per operation
};

BenchResult bench_field_mul(const BenchConfig& cfg) {
    FieldElement *d_a, *d_b, *d_r;
    std::vector<FieldElement> h_a(cfg.batch_size), h_b(cfg.batch_size);

    generate_random_field_elements(h_a.data(), cfg.batch_size);
    generate_random_field_elements(h_b.data(), cfg.batch_size);

    size_t size = cfg.batch_size * sizeof(FieldElement);
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMalloc(&d_r, size));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), size, cudaMemcpyHostToDevice));

    int blocks = (cfg.batch_size + cfg.threads_per_block - 1) / cfg.threads_per_block;

    // Warmup
    for (int i = 0; i < cfg.warmup_iterations; ++i) {
        field_mul_kernel<<<blocks, cfg.threads_per_block>>>(d_a, d_b, d_r, cfg.batch_size);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Measure
    CudaTimer timer;
    timer.start();
    for (int i = 0; i < cfg.measure_iterations; ++i) {
        field_mul_kernel<<<blocks, cfg.threads_per_block>>>(d_a, d_b, d_r, cfg.batch_size);
    }
    float total_ms = timer.stop();

    double avg_ms = total_ms / cfg.measure_iterations;
    double throughput = (cfg.batch_size / avg_ms) / 1000.0;  // Million ops/sec
    double ns_per_op = (avg_ms * 1e6) / cfg.batch_size;

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_r));

    return {"Field Mul", avg_ms, cfg.batch_size, throughput, ns_per_op};
}

BenchResult bench_field_add(const BenchConfig& cfg) {
    FieldElement *d_a, *d_b, *d_r;
    std::vector<FieldElement> h_a(cfg.batch_size), h_b(cfg.batch_size);

    generate_random_field_elements(h_a.data(), cfg.batch_size);
    generate_random_field_elements(h_b.data(), cfg.batch_size);

    size_t size = cfg.batch_size * sizeof(FieldElement);
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMalloc(&d_r, size));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), size, cudaMemcpyHostToDevice));

    int blocks = (cfg.batch_size + cfg.threads_per_block - 1) / cfg.threads_per_block;

    // Warmup
    for (int i = 0; i < cfg.warmup_iterations; ++i) {
        field_add_kernel<<<blocks, cfg.threads_per_block>>>(d_a, d_b, d_r, cfg.batch_size);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Measure
    CudaTimer timer;
    timer.start();
    for (int i = 0; i < cfg.measure_iterations; ++i) {
        field_add_kernel<<<blocks, cfg.threads_per_block>>>(d_a, d_b, d_r, cfg.batch_size);
    }
    float total_ms = timer.stop();

    double avg_ms = total_ms / cfg.measure_iterations;
    double throughput = (cfg.batch_size / avg_ms) / 1000.0;
    double ns_per_op = (avg_ms * 1e6) / cfg.batch_size;

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_r));

    return {"Field Add", avg_ms, cfg.batch_size, throughput, ns_per_op};
}

BenchResult bench_field_inv(const BenchConfig& cfg) {
    // Use smaller batch for inverse (very expensive)
    int batch = std::min(cfg.batch_size, 1 << 16);  // Max 64K

    FieldElement *d_a, *d_r;
    std::vector<FieldElement> h_a(batch);

    generate_random_field_elements(h_a.data(), batch);

    size_t size = batch * sizeof(FieldElement);
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_r, size));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice));

    int blocks = (batch + cfg.threads_per_block - 1) / cfg.threads_per_block;

    // Warmup
    for (int i = 0; i < cfg.warmup_iterations; ++i) {
        field_inv_kernel<<<blocks, cfg.threads_per_block>>>(d_a, d_r, batch);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Measure
    CudaTimer timer;
    timer.start();
    for (int i = 0; i < cfg.measure_iterations; ++i) {
        field_inv_kernel<<<blocks, cfg.threads_per_block>>>(d_a, d_r, batch);
    }
    float total_ms = timer.stop();

    double avg_ms = total_ms / cfg.measure_iterations;
    double throughput = (batch / avg_ms) / 1000.0;
    double ns_per_op = (avg_ms * 1e6) / batch;

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_r));

    return {"Field Inverse", avg_ms, batch, throughput, ns_per_op};
}

BenchResult bench_point_add(const BenchConfig& cfg) {
    int batch = std::min(cfg.batch_size, 1 << 18);  // Max 256K

    JacobianPoint *d_a, *d_b, *d_r;
    std::vector<JacobianPoint> h_a(batch), h_b(batch);

    generate_random_points(h_a.data(), batch);
    generate_random_points(h_b.data(), batch);

    size_t size = batch * sizeof(JacobianPoint);
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMalloc(&d_r, size));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), size, cudaMemcpyHostToDevice));

    int blocks = (batch + cfg.threads_per_block - 1) / cfg.threads_per_block;

    // Warmup
    for (int i = 0; i < cfg.warmup_iterations; ++i) {
        point_add_kernel<<<blocks, cfg.threads_per_block>>>(d_a, d_b, d_r, batch);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Measure
    CudaTimer timer;
    timer.start();
    for (int i = 0; i < cfg.measure_iterations; ++i) {
        point_add_kernel<<<blocks, cfg.threads_per_block>>>(d_a, d_b, d_r, batch);
    }
    float total_ms = timer.stop();

    double avg_ms = total_ms / cfg.measure_iterations;
    double throughput = (batch / avg_ms) / 1000.0;
    double ns_per_op = (avg_ms * 1e6) / batch;

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_r));

    return {"Point Add", avg_ms, batch, throughput, ns_per_op};
}

BenchResult bench_point_double(const BenchConfig& cfg) {
    int batch = std::min(cfg.batch_size, 1 << 18);

    JacobianPoint *d_a, *d_r;
    std::vector<JacobianPoint> h_a(batch);

    generate_random_points(h_a.data(), batch);

    size_t size = batch * sizeof(JacobianPoint);
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_r, size));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice));

    int blocks = (batch + cfg.threads_per_block - 1) / cfg.threads_per_block;

    // Warmup
    for (int i = 0; i < cfg.warmup_iterations; ++i) {
        point_dbl_kernel<<<blocks, cfg.threads_per_block>>>(d_a, d_r, batch);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Measure
    CudaTimer timer;
    timer.start();
    for (int i = 0; i < cfg.measure_iterations; ++i) {
        point_dbl_kernel<<<blocks, cfg.threads_per_block>>>(d_a, d_r, batch);
    }
    float total_ms = timer.stop();

    double avg_ms = total_ms / cfg.measure_iterations;
    double throughput = (batch / avg_ms) / 1000.0;
    double ns_per_op = (avg_ms * 1e6) / batch;

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_r));

    return {"Point Double", avg_ms, batch, throughput, ns_per_op};
}

BenchResult bench_scalar_mul(const BenchConfig& cfg) {
    // Scalar multiplication is very expensive - use smaller batch
    int batch = std::min(cfg.batch_size, 1 << 16);  // Max 64K

    JacobianPoint *d_points, *d_results;
    Scalar *d_scalars;
    std::vector<JacobianPoint> h_points(batch);
    std::vector<Scalar> h_scalars(batch);

    generate_random_points(h_points.data(), batch);
    generate_random_scalars(h_scalars.data(), batch);

    CUDA_CHECK(cudaMalloc(&d_points, batch * sizeof(JacobianPoint)));
    CUDA_CHECK(cudaMalloc(&d_scalars, batch * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_results, batch * sizeof(JacobianPoint)));

    CUDA_CHECK(cudaMemcpy(d_points, h_points.data(), batch * sizeof(JacobianPoint), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalars, h_scalars.data(), batch * sizeof(Scalar), cudaMemcpyHostToDevice));

    int blocks = (batch + cfg.threads_per_block - 1) / cfg.threads_per_block;

    // Warmup
    for (int i = 0; i < cfg.warmup_iterations; ++i) {
        scalar_mul_batch_kernel<<<blocks, cfg.threads_per_block>>>(d_points, d_scalars, d_results, batch);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Measure
    CudaTimer timer;
    timer.start();
    for (int i = 0; i < cfg.measure_iterations; ++i) {
        scalar_mul_batch_kernel<<<blocks, cfg.threads_per_block>>>(d_points, d_scalars, d_results, batch);
    }
    float total_ms = timer.stop();

    double avg_ms = total_ms / cfg.measure_iterations;
    double throughput = (batch / avg_ms) / 1000.0;
    double us_per_op = (avg_ms * 1000.0) / batch;

    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_scalars));
    CUDA_CHECK(cudaFree(d_results));

    return {"Scalar Mul (P*k)", avg_ms, batch, throughput, us_per_op * 1000};  // Convert to ns
}

BenchResult bench_generator_mul(const BenchConfig& cfg) {
    int batch = std::min(cfg.batch_size, 1 << 17);  // Max 128K

    JacobianPoint *d_results;
    Scalar *d_scalars;
    std::vector<Scalar> h_scalars(batch);

    generate_random_scalars(h_scalars.data(), batch);

    CUDA_CHECK(cudaMalloc(&d_scalars, batch * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_results, batch * sizeof(JacobianPoint)));

    CUDA_CHECK(cudaMemcpy(d_scalars, h_scalars.data(), batch * sizeof(Scalar), cudaMemcpyHostToDevice));

    int blocks = (batch + cfg.threads_per_block - 1) / cfg.threads_per_block;

    // Warmup
    for (int i = 0; i < cfg.warmup_iterations; ++i) {
        generator_mul_batch_kernel<<<blocks, cfg.threads_per_block>>>(d_scalars, d_results, batch);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Measure
    CudaTimer timer;
    timer.start();
    for (int i = 0; i < cfg.measure_iterations; ++i) {
        generator_mul_batch_kernel<<<blocks, cfg.threads_per_block>>>(d_scalars, d_results, batch);
    }
    float total_ms = timer.stop();

    double avg_ms = total_ms / cfg.measure_iterations;
    double throughput = (batch / avg_ms) / 1000.0;
    double us_per_op = (avg_ms * 1000.0) / batch;

    CUDA_CHECK(cudaFree(d_scalars));
    CUDA_CHECK(cudaFree(d_results));

    return {"Generator Mul (G*k)", avg_ms, batch, throughput, us_per_op * 1000};
}

// ============================================================================
// Print results
// ============================================================================
void print_device_info() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "============================================================\n";
    std::cout << "  Secp256k1 CUDA Benchmark\n";
    std::cout << "============================================================\n\n";

    std::cout << "GPU Information:\n";
    std::cout << "  Device:         " << prop.name << "\n";
    std::cout << "  Compute:        " << prop.major << "." << prop.minor << "\n";
    std::cout << "  SM Count:       " << prop.multiProcessorCount << "\n";
    std::cout << "  Clock:          " << prop.clockRate / 1000 << " MHz\n";
    std::cout << "  Memory:         " << prop.totalGlobalMem / (1024*1024) << " MB\n";
    std::cout << "  Memory Clock:   " << prop.memoryClockRate / 1000 << " MHz\n";
    std::cout << "  Memory Bus:     " << prop.memoryBusWidth << " bit\n";
    std::cout << "\n";
}

void print_result(const BenchResult& r) {
    std::cout << std::left << std::setw(22) << r.name << " | ";

    if (r.time_per_op_ns >= 1000000) {
        std::cout << std::right << std::setw(8) << std::fixed << std::setprecision(2)
                  << r.time_per_op_ns / 1000000 << " ms";
    } else if (r.time_per_op_ns >= 1000) {
        std::cout << std::right << std::setw(8) << std::fixed << std::setprecision(2)
                  << r.time_per_op_ns / 1000 << " μs";
    } else {
        std::cout << std::right << std::setw(8) << std::fixed << std::setprecision(1)
                  << r.time_per_op_ns << " ns";
    }

    std::cout << " | " << std::right << std::setw(10) << std::fixed << std::setprecision(2)
              << r.throughput_mops << " Mops/s";
    std::cout << " | batch=" << r.batch_size << "\n";
}

void print_summary_table(const std::vector<BenchResult>& results) {
    std::cout << "\n==============================================\n";
    std::cout << "  Performance Summary (for README)\n";
    std::cout << "==============================================\n";

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << "\n\n";

    std::cout << "| Operation              | Time/Op    | Throughput   |\n";
    std::cout << "|------------------------|------------|-------------:|\n";

    for (const auto& r : results) {
        std::cout << "| " << std::left << std::setw(22) << r.name << " | ";

        if (r.time_per_op_ns >= 1000000) {
            std::cout << std::right << std::setw(8) << std::fixed << std::setprecision(2)
                      << r.time_per_op_ns / 1000000 << " ms";
        } else if (r.time_per_op_ns >= 1000) {
            std::cout << std::right << std::setw(8) << std::fixed << std::setprecision(2)
                      << r.time_per_op_ns / 1000 << " μs";
        } else {
            std::cout << std::right << std::setw(8) << std::fixed << std::setprecision(1)
                      << r.time_per_op_ns << " ns";
        }

        std::cout << " | " << std::right << std::setw(10) << std::fixed << std::setprecision(2)
                  << r.throughput_mops << " M/s |\n";
    }
    std::cout << "\n";
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    print_device_info();

    BenchConfig cfg;

    // Parse command line
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--batch" && i + 1 < argc) {
            cfg.batch_size = std::atoi(argv[++i]);
        } else if (std::string(argv[i]) == "--threads" && i + 1 < argc) {
            cfg.threads_per_block = std::atoi(argv[++i]);
        }
    }

    std::cout << "Benchmark Configuration:\n";
    std::cout << "  Batch Size:     " << cfg.batch_size << "\n";
    std::cout << "  Threads/Block:  " << cfg.threads_per_block << "\n";
    std::cout << "  Warmup Iters:   " << cfg.warmup_iterations << "\n";
    std::cout << "  Measure Iters:  " << cfg.measure_iterations << "\n";
    std::cout << "\n";

    std::vector<BenchResult> results;

    std::cout << "Running benchmarks...\n\n";

    // Field operations
    std::cout << "=== Field Arithmetic ===\n";
    results.push_back(bench_field_mul(cfg));
    print_result(results.back());

    results.push_back(bench_field_add(cfg));
    print_result(results.back());

    results.push_back(bench_field_inv(cfg));
    print_result(results.back());

    // Point operations
    std::cout << "\n=== Point Operations ===\n";
    results.push_back(bench_point_add(cfg));
    print_result(results.back());

    results.push_back(bench_point_double(cfg));
    print_result(results.back());

    // Scalar multiplication
    std::cout << "\n=== Scalar Multiplication ===\n";
    results.push_back(bench_scalar_mul(cfg));
    print_result(results.back());

    results.push_back(bench_generator_mul(cfg));
    print_result(results.back());

    // Print summary table
    print_summary_table(results);

    std::cout << "Benchmark complete.\n";

    return 0;
}

