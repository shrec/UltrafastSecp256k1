/**
 * CUDA Benchmark for Secp256k1
 *
 * Comprehensive benchmark of all CUDA operations:
 * - Field arithmetic (mul, square, add, sub, inverse)
 * - Point operations (add, double)
 * - Affine point addition (2M+1S vs Jacobian 11M+5S)
 * - Batch inversion (Montgomery's trick)
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
#include "affine_add.cuh"
#include "ecdsa.cuh"
#include "schnorr.cuh"
#include "recovery.cuh"
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

void generate_random_affine_points(FieldElement* h_x, FieldElement* h_y, int count) {
    std::mt19937_64 rng(789);
    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < 4; ++j) {
            h_x[i].limbs[j] = rng();
            h_y[i].limbs[j] = rng();
        }
        h_x[i].limbs[3] &= 0x7FFFFFFFFFFFFFFFULL;
        h_y[i].limbs[3] &= 0x7FFFFFFFFFFFFFFFULL;
    }
}

// ============================================================================
// Affine benchmark wrapper kernels (__device__ -> __global__)
// ============================================================================

// Full affine add (includes per-element inversion -- 2M + 1S + inv)
__global__ void bench_affine_add_kernel(
    const FieldElement* __restrict__ px, const FieldElement* __restrict__ py,
    const FieldElement* __restrict__ qx, const FieldElement* __restrict__ qy,
    FieldElement* __restrict__ rx, FieldElement* __restrict__ ry,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        secp256k1::cuda::affine_add(&px[idx], &py[idx], &qx[idx], &qy[idx],
                                    &rx[idx], &ry[idx]);
    }
}

// Affine add with pre-inverted H -- full X,Y output (2M + 1S)
__global__ void bench_affine_add_lambda_kernel(
    const FieldElement* __restrict__ px, const FieldElement* __restrict__ py,
    const FieldElement* __restrict__ qx, const FieldElement* __restrict__ qy,
    const FieldElement* __restrict__ h_inv,
    FieldElement* __restrict__ rx, FieldElement* __restrict__ ry,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        secp256k1::cuda::affine_add_lambda(&px[idx], &py[idx], &qx[idx], &qy[idx],
                                           &h_inv[idx], &rx[idx], &ry[idx]);
    }
}

// Affine add X-only with pre-inverted H (1M + 1S)
__global__ void bench_affine_add_xonly_kernel(
    const FieldElement* __restrict__ px, const FieldElement* __restrict__ py,
    const FieldElement* __restrict__ qx, const FieldElement* __restrict__ qy,
    const FieldElement* __restrict__ h_inv,
    FieldElement* __restrict__ rx,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        secp256k1::cuda::affine_add_x_only(&px[idx], &py[idx], &qx[idx], &qy[idx],
                                           &h_inv[idx], &rx[idx]);
    }
}

// Compute H = Q.x - P.x (for pre-inversion step timing)
__global__ void bench_affine_compute_h_kernel(
    const FieldElement* __restrict__ px,
    const FieldElement* __restrict__ qx,
    FieldElement* __restrict__ h,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        secp256k1::cuda::affine_compute_h(&px[idx], &qx[idx], &h[idx]);
    }
}

// Batch inversion kernel -- one thread processes a serial batch of CHAIN_LEN elements
static constexpr int BATCH_INV_CHAIN_LEN = 64;

__global__ void bench_batch_inv_kernel(
    FieldElement* __restrict__ h,
    FieldElement* __restrict__ prefix,
    int total_count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = tid * BATCH_INV_CHAIN_LEN;
    if (offset + BATCH_INV_CHAIN_LEN <= total_count) {
        secp256k1::cuda::affine_batch_inv_serial(
            &h[offset], &prefix[offset], BATCH_INV_CHAIN_LEN);
    }
}

// Jacobian -> Affine conversion kernel
__global__ void bench_jac_to_affine_kernel(
    FieldElement* __restrict__ x,
    FieldElement* __restrict__ y,
    const FieldElement* __restrict__ z,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        secp256k1::cuda::jacobian_to_affine(&x[idx], &y[idx], &z[idx]);
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

    // Must match __launch_bounds__(128, 2) on scalar_mul_batch_kernel
    constexpr int kThreads = 128;

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

    int blocks = (batch + kThreads - 1) / kThreads;

    // Warmup
    for (int i = 0; i < cfg.warmup_iterations; ++i) {
        scalar_mul_batch_kernel<<<blocks, kThreads>>>(d_points, d_scalars, d_results, batch);
    }
    CUDA_CHECK(cudaGetLastError());  // Catch launch failures
    CUDA_CHECK(cudaDeviceSynchronize());

    // Measure
    CudaTimer timer;
    timer.start();
    for (int i = 0; i < cfg.measure_iterations; ++i) {
        scalar_mul_batch_kernel<<<blocks, kThreads>>>(d_points, d_scalars, d_results, batch);
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

    // Must match __launch_bounds__(128, 2) on generator_mul_batch_kernel
    constexpr int kThreads = 128;

    JacobianPoint *d_results;
    Scalar *d_scalars;
    std::vector<Scalar> h_scalars(batch);

    generate_random_scalars(h_scalars.data(), batch);

    CUDA_CHECK(cudaMalloc(&d_scalars, batch * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_results, batch * sizeof(JacobianPoint)));

    CUDA_CHECK(cudaMemcpy(d_scalars, h_scalars.data(), batch * sizeof(Scalar), cudaMemcpyHostToDevice));

    int blocks = (batch + kThreads - 1) / kThreads;

    // Warmup
    for (int i = 0; i < cfg.warmup_iterations; ++i) {
        generator_mul_batch_kernel<<<blocks, kThreads>>>(d_scalars, d_results, batch);
    }
    CUDA_CHECK(cudaGetLastError());  // Catch launch failures
    CUDA_CHECK(cudaDeviceSynchronize());

    // Measure
    CudaTimer timer;
    timer.start();
    for (int i = 0; i < cfg.measure_iterations; ++i) {
        generator_mul_batch_kernel<<<blocks, kThreads>>>(d_scalars, d_results, batch);
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
// Affine point addition benchmarks
// ============================================================================

BenchResult bench_affine_add(const BenchConfig& cfg) {
    int batch = std::min(cfg.batch_size, 1 << 18);  // Max 256K

    FieldElement *d_px, *d_py, *d_qx, *d_qy, *d_rx, *d_ry;
    std::vector<FieldElement> h_px(batch), h_py(batch), h_qx(batch), h_qy(batch);

    generate_random_affine_points(h_px.data(), h_py.data(), batch);
    generate_random_affine_points(h_qx.data(), h_qy.data(), batch);

    size_t size = batch * sizeof(FieldElement);
    CUDA_CHECK(cudaMalloc(&d_px, size));
    CUDA_CHECK(cudaMalloc(&d_py, size));
    CUDA_CHECK(cudaMalloc(&d_qx, size));
    CUDA_CHECK(cudaMalloc(&d_qy, size));
    CUDA_CHECK(cudaMalloc(&d_rx, size));
    CUDA_CHECK(cudaMalloc(&d_ry, size));

    CUDA_CHECK(cudaMemcpy(d_px, h_px.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, h_py.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qx, h_qx.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qy, h_qy.data(), size, cudaMemcpyHostToDevice));

    int blocks = (batch + cfg.threads_per_block - 1) / cfg.threads_per_block;

    for (int i = 0; i < cfg.warmup_iterations; ++i) {
        bench_affine_add_kernel<<<blocks, cfg.threads_per_block>>>(
            d_px, d_py, d_qx, d_qy, d_rx, d_ry, batch);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaTimer timer;
    timer.start();
    for (int i = 0; i < cfg.measure_iterations; ++i) {
        bench_affine_add_kernel<<<blocks, cfg.threads_per_block>>>(
            d_px, d_py, d_qx, d_qy, d_rx, d_ry, batch);
    }
    float total_ms = timer.stop();

    double avg_ms = total_ms / cfg.measure_iterations;
    double throughput = (batch / avg_ms) / 1000.0;
    double ns_per_op = (avg_ms * 1e6) / batch;

    CUDA_CHECK(cudaFree(d_px));
    CUDA_CHECK(cudaFree(d_py));
    CUDA_CHECK(cudaFree(d_qx));
    CUDA_CHECK(cudaFree(d_qy));
    CUDA_CHECK(cudaFree(d_rx));
    CUDA_CHECK(cudaFree(d_ry));

    return {"Affine Add (2M+1S+inv)", avg_ms, batch, throughput, ns_per_op};
}

BenchResult bench_affine_add_lambda(const BenchConfig& cfg) {
    int batch = std::min(cfg.batch_size, 1 << 18);

    FieldElement *d_px, *d_py, *d_qx, *d_qy, *d_hinv, *d_rx, *d_ry;
    std::vector<FieldElement> h_px(batch), h_py(batch), h_qx(batch), h_qy(batch), h_hinv(batch);

    generate_random_affine_points(h_px.data(), h_py.data(), batch);
    generate_random_affine_points(h_qx.data(), h_qy.data(), batch);
    // Pre-compute H^{-1} on host (random values simulate pre-inverted state)
    generate_random_field_elements(h_hinv.data(), batch);

    size_t size = batch * sizeof(FieldElement);
    CUDA_CHECK(cudaMalloc(&d_px, size));
    CUDA_CHECK(cudaMalloc(&d_py, size));
    CUDA_CHECK(cudaMalloc(&d_qx, size));
    CUDA_CHECK(cudaMalloc(&d_qy, size));
    CUDA_CHECK(cudaMalloc(&d_hinv, size));
    CUDA_CHECK(cudaMalloc(&d_rx, size));
    CUDA_CHECK(cudaMalloc(&d_ry, size));

    CUDA_CHECK(cudaMemcpy(d_px, h_px.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, h_py.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qx, h_qx.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qy, h_qy.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hinv, h_hinv.data(), size, cudaMemcpyHostToDevice));

    int blocks = (batch + cfg.threads_per_block - 1) / cfg.threads_per_block;

    for (int i = 0; i < cfg.warmup_iterations; ++i) {
        bench_affine_add_lambda_kernel<<<blocks, cfg.threads_per_block>>>(
            d_px, d_py, d_qx, d_qy, d_hinv, d_rx, d_ry, batch);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaTimer timer;
    timer.start();
    for (int i = 0; i < cfg.measure_iterations; ++i) {
        bench_affine_add_lambda_kernel<<<blocks, cfg.threads_per_block>>>(
            d_px, d_py, d_qx, d_qy, d_hinv, d_rx, d_ry, batch);
    }
    float total_ms = timer.stop();

    double avg_ms = total_ms / cfg.measure_iterations;
    double throughput = (batch / avg_ms) / 1000.0;
    double ns_per_op = (avg_ms * 1e6) / batch;

    CUDA_CHECK(cudaFree(d_px));
    CUDA_CHECK(cudaFree(d_py));
    CUDA_CHECK(cudaFree(d_qx));
    CUDA_CHECK(cudaFree(d_qy));
    CUDA_CHECK(cudaFree(d_hinv));
    CUDA_CHECK(cudaFree(d_rx));
    CUDA_CHECK(cudaFree(d_ry));

    return {"Affine Lambda (2M+1S)", avg_ms, batch, throughput, ns_per_op};
}

BenchResult bench_affine_add_xonly(const BenchConfig& cfg) {
    int batch = std::min(cfg.batch_size, 1 << 18);

    FieldElement *d_px, *d_py, *d_qx, *d_qy, *d_hinv, *d_rx;
    std::vector<FieldElement> h_px(batch), h_py(batch), h_qx(batch), h_qy(batch), h_hinv(batch);

    generate_random_affine_points(h_px.data(), h_py.data(), batch);
    generate_random_affine_points(h_qx.data(), h_qy.data(), batch);
    generate_random_field_elements(h_hinv.data(), batch);

    size_t size = batch * sizeof(FieldElement);
    CUDA_CHECK(cudaMalloc(&d_px, size));
    CUDA_CHECK(cudaMalloc(&d_py, size));
    CUDA_CHECK(cudaMalloc(&d_qx, size));
    CUDA_CHECK(cudaMalloc(&d_qy, size));
    CUDA_CHECK(cudaMalloc(&d_hinv, size));
    CUDA_CHECK(cudaMalloc(&d_rx, size));

    CUDA_CHECK(cudaMemcpy(d_px, h_px.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, h_py.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qx, h_qx.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qy, h_qy.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hinv, h_hinv.data(), size, cudaMemcpyHostToDevice));

    int blocks = (batch + cfg.threads_per_block - 1) / cfg.threads_per_block;

    for (int i = 0; i < cfg.warmup_iterations; ++i) {
        bench_affine_add_xonly_kernel<<<blocks, cfg.threads_per_block>>>(
            d_px, d_py, d_qx, d_qy, d_hinv, d_rx, batch);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaTimer timer;
    timer.start();
    for (int i = 0; i < cfg.measure_iterations; ++i) {
        bench_affine_add_xonly_kernel<<<blocks, cfg.threads_per_block>>>(
            d_px, d_py, d_qx, d_qy, d_hinv, d_rx, batch);
    }
    float total_ms = timer.stop();

    double avg_ms = total_ms / cfg.measure_iterations;
    double throughput = (batch / avg_ms) / 1000.0;
    double ns_per_op = (avg_ms * 1e6) / batch;

    CUDA_CHECK(cudaFree(d_px));
    CUDA_CHECK(cudaFree(d_py));
    CUDA_CHECK(cudaFree(d_qx));
    CUDA_CHECK(cudaFree(d_qy));
    CUDA_CHECK(cudaFree(d_hinv));
    CUDA_CHECK(cudaFree(d_rx));

    return {"Affine X-Only (1M+1S)", avg_ms, batch, throughput, ns_per_op};
}

BenchResult bench_batch_inversion(const BenchConfig& cfg) {
    // Each thread processes BATCH_INV_CHAIN_LEN elements
    int chains = std::min(cfg.batch_size / BATCH_INV_CHAIN_LEN, 1 << 14);  // Max 16K chains
    int total = chains * BATCH_INV_CHAIN_LEN;

    FieldElement *d_h, *d_prefix;
    std::vector<FieldElement> h_data(total);
    generate_random_field_elements(h_data.data(), total);

    size_t size = total * sizeof(FieldElement);
    CUDA_CHECK(cudaMalloc(&d_h, size));
    CUDA_CHECK(cudaMalloc(&d_prefix, size));

    CUDA_CHECK(cudaMemcpy(d_h, h_data.data(), size, cudaMemcpyHostToDevice));

    int blocks = (chains + cfg.threads_per_block - 1) / cfg.threads_per_block;

    for (int i = 0; i < cfg.warmup_iterations; ++i) {
        // Reload data (kernel is in-place)
        CUDA_CHECK(cudaMemcpy(d_h, h_data.data(), size, cudaMemcpyHostToDevice));
        bench_batch_inv_kernel<<<blocks, cfg.threads_per_block>>>(d_h, d_prefix, total);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaTimer timer;
    timer.start();
    for (int i = 0; i < cfg.measure_iterations; ++i) {
        CUDA_CHECK(cudaMemcpy(d_h, h_data.data(), size, cudaMemcpyHostToDevice));
        bench_batch_inv_kernel<<<blocks, cfg.threads_per_block>>>(d_h, d_prefix, total);
    }
    float total_ms = timer.stop();

    double avg_ms = total_ms / cfg.measure_iterations;
    double throughput = (total / avg_ms) / 1000.0;
    double ns_per_op = (avg_ms * 1e6) / total;

    CUDA_CHECK(cudaFree(d_h));
    CUDA_CHECK(cudaFree(d_prefix));

    return {"Batch Inv (Montgomery)", avg_ms, total, throughput, ns_per_op};
}

BenchResult bench_jacobian_to_affine(const BenchConfig& cfg) {
    int batch = std::min(cfg.batch_size, 1 << 16);  // Max 64K (has inv per element)

    FieldElement *d_x, *d_y, *d_z;
    std::vector<FieldElement> h_x(batch), h_y(batch), h_z(batch);

    std::mt19937_64 rng(999);
    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < 4; ++j) {
            h_x[i].limbs[j] = rng();
            h_y[i].limbs[j] = rng();
            h_z[i].limbs[j] = rng();
        }
        h_x[i].limbs[3] &= 0x7FFFFFFFFFFFFFFFULL;
        h_y[i].limbs[3] &= 0x7FFFFFFFFFFFFFFFULL;
        h_z[i].limbs[3] &= 0x7FFFFFFFFFFFFFFFULL;
    }

    size_t size = batch * sizeof(FieldElement);
    CUDA_CHECK(cudaMalloc(&d_x, size));
    CUDA_CHECK(cudaMalloc(&d_y, size));
    CUDA_CHECK(cudaMalloc(&d_z, size));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, h_z.data(), size, cudaMemcpyHostToDevice));

    int blocks = (batch + cfg.threads_per_block - 1) / cfg.threads_per_block;

    for (int i = 0; i < cfg.warmup_iterations; ++i) {
        // Reload (in-place modification)
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), size, cudaMemcpyHostToDevice));
        bench_jac_to_affine_kernel<<<blocks, cfg.threads_per_block>>>(d_x, d_y, d_z, batch);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaTimer timer;
    timer.start();
    for (int i = 0; i < cfg.measure_iterations; ++i) {
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), size, cudaMemcpyHostToDevice));
        bench_jac_to_affine_kernel<<<blocks, cfg.threads_per_block>>>(d_x, d_y, d_z, batch);
    }
    float total_ms = timer.stop();

    double avg_ms = total_ms / cfg.measure_iterations;
    double throughput = (batch / avg_ms) / 1000.0;
    double ns_per_op = (avg_ms * 1e6) / batch;

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));

    return {"Jac->Affine (per-pt)", avg_ms, batch, throughput, ns_per_op};
}

// ============================================================================
// Signature benchmarks (ECDSA + Schnorr) -- 64-bit limb mode only
// ============================================================================

// Forward-declare batch kernels (defined in secp256k1.cu, namespace secp256k1::cuda)
namespace secp256k1 { namespace cuda {
extern __global__ void ecdsa_sign_batch_kernel(
    const uint8_t*, const Scalar*, ECDSASignatureGPU*, bool*, int);
extern __global__ void ecdsa_verify_batch_kernel(
    const uint8_t*, const JacobianPoint*, const ECDSASignatureGPU*, bool*, int);
extern __global__ void schnorr_sign_batch_kernel(
    const Scalar*, const uint8_t*, const uint8_t*, SchnorrSignatureGPU*, bool*, int);
extern __global__ void schnorr_verify_batch_kernel(
    const uint8_t*, const uint8_t*, const SchnorrSignatureGPU*, bool*, int);
extern __global__ void ecdsa_sign_recoverable_batch_kernel(
    const uint8_t*, const Scalar*, RecoverableSignatureGPU*, bool*, int);
extern __global__ void ecdsa_recover_batch_kernel(
    const uint8_t*, const ECDSASignatureGPU*, const int*, JacobianPoint*, bool*, int);
}} // namespace secp256k1::cuda

// Helper: generate test keys and sign messages on GPU for verify benchmarks
static void prepare_ecdsa_test_data(
    int batch,
    std::vector<uint8_t>& h_msgs,
    std::vector<Scalar>& h_privkeys,
    std::vector<JacobianPoint>& h_pubkeys,
    std::vector<ECDSASignatureGPU>& h_sigs)
{
    h_msgs.resize(batch * 32);
    h_privkeys.resize(batch);
    h_pubkeys.resize(batch);
    h_sigs.resize(batch);

    std::mt19937_64 rng(0xECD5A);
    for (int i = 0; i < batch; ++i) {
        // Random message
        for (int j = 0; j < 32; j++) h_msgs[i * 32 + j] = (uint8_t)(rng() & 0xFF);
        // Random private key
        for (int j = 0; j < 4; j++) h_privkeys[i].limbs[j] = rng();
        h_privkeys[i].limbs[3] &= 0x7FFFFFFFFFFFFFFFULL;
        // Prevent zero keys
        if (h_privkeys[i].limbs[0] == 0 && h_privkeys[i].limbs[1] == 0 &&
            h_privkeys[i].limbs[2] == 0 && h_privkeys[i].limbs[3] == 0)
            h_privkeys[i].limbs[0] = 1;
    }

    // Sign all on GPU to get sigs + gen pubkeys
    uint8_t *d_msgs; Scalar *d_priv; ECDSASignatureGPU *d_sigs; bool *d_res;
    CUDA_CHECK(cudaMalloc(&d_msgs, batch * 32));
    CUDA_CHECK(cudaMalloc(&d_priv, batch * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_sigs, batch * sizeof(ECDSASignatureGPU)));
    CUDA_CHECK(cudaMalloc(&d_res, batch * sizeof(bool)));

    CUDA_CHECK(cudaMemcpy(d_msgs, h_msgs.data(), batch * 32, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_priv, h_privkeys.data(), batch * sizeof(Scalar), cudaMemcpyHostToDevice));

    int blocks = (batch + 127) / 128;
    ecdsa_sign_batch_kernel<<<blocks, 128>>>(d_msgs, d_priv, d_sigs, d_res, batch);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_sigs.data(), d_sigs, batch * sizeof(ECDSASignatureGPU), cudaMemcpyDeviceToHost));

    // Generate public keys: Q = privkey * G
    JacobianPoint *d_pubs;
    CUDA_CHECK(cudaMalloc(&d_pubs, batch * sizeof(JacobianPoint)));
    generator_mul_batch_kernel<<<blocks, 128>>>(d_priv, d_pubs, batch);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_pubkeys.data(), d_pubs, batch * sizeof(JacobianPoint), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_msgs));
    CUDA_CHECK(cudaFree(d_priv));
    CUDA_CHECK(cudaFree(d_sigs));
    CUDA_CHECK(cudaFree(d_res));
    CUDA_CHECK(cudaFree(d_pubs));
}

BenchResult bench_ecdsa_sign(const BenchConfig& cfg) {
    constexpr int kThreads = 128;
    int batch = std::min(cfg.batch_size, 1 << 14);  // Max 16K (heavy)

    std::vector<uint8_t> h_msgs(batch * 32);
    std::vector<Scalar> h_privkeys(batch);

    std::mt19937_64 rng(0xEC01);
    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < 32; j++) h_msgs[i * 32 + j] = (uint8_t)(rng() & 0xFF);
        for (int j = 0; j < 4; j++) h_privkeys[i].limbs[j] = rng();
        h_privkeys[i].limbs[3] &= 0x7FFFFFFFFFFFFFFFULL;
        if (h_privkeys[i].limbs[0] == 0 && h_privkeys[i].limbs[1] == 0 &&
            h_privkeys[i].limbs[2] == 0 && h_privkeys[i].limbs[3] == 0)
            h_privkeys[i].limbs[0] = 1;
    }

    uint8_t *d_msgs; Scalar *d_priv; ECDSASignatureGPU *d_sigs; bool *d_res;
    CUDA_CHECK(cudaMalloc(&d_msgs, batch * 32));
    CUDA_CHECK(cudaMalloc(&d_priv, batch * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_sigs, batch * sizeof(ECDSASignatureGPU)));
    CUDA_CHECK(cudaMalloc(&d_res, batch * sizeof(bool)));

    CUDA_CHECK(cudaMemcpy(d_msgs, h_msgs.data(), batch * 32, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_priv, h_privkeys.data(), batch * sizeof(Scalar), cudaMemcpyHostToDevice));

    int blocks = (batch + kThreads - 1) / kThreads;

    for (int i = 0; i < cfg.warmup_iterations; ++i) {
        ecdsa_sign_batch_kernel<<<blocks, kThreads>>>(d_msgs, d_priv, d_sigs, d_res, batch);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaTimer timer;
    timer.start();
    for (int i = 0; i < cfg.measure_iterations; ++i) {
        ecdsa_sign_batch_kernel<<<blocks, kThreads>>>(d_msgs, d_priv, d_sigs, d_res, batch);
    }
    float total_ms = timer.stop();

    double avg_ms = total_ms / cfg.measure_iterations;
    double throughput = (batch / avg_ms) / 1000.0;
    double ns_per_op = (avg_ms * 1e6) / batch;

    CUDA_CHECK(cudaFree(d_msgs));
    CUDA_CHECK(cudaFree(d_priv));
    CUDA_CHECK(cudaFree(d_sigs));
    CUDA_CHECK(cudaFree(d_res));

    return {"ECDSA Sign", avg_ms, batch, throughput, ns_per_op};
}

BenchResult bench_ecdsa_verify(const BenchConfig& cfg) {
    constexpr int kThreads = 128;
    int batch = std::min(cfg.batch_size, 1 << 14);

    std::vector<uint8_t> h_msgs;
    std::vector<Scalar> h_privkeys;
    std::vector<JacobianPoint> h_pubkeys;
    std::vector<ECDSASignatureGPU> h_sigs;
    prepare_ecdsa_test_data(batch, h_msgs, h_privkeys, h_pubkeys, h_sigs);

    uint8_t *d_msgs;
    JacobianPoint *d_pubs;
    ECDSASignatureGPU *d_sigs_d;
    bool *d_res;

    CUDA_CHECK(cudaMalloc(&d_msgs, batch * 32));
    CUDA_CHECK(cudaMalloc(&d_pubs, batch * sizeof(JacobianPoint)));
    CUDA_CHECK(cudaMalloc(&d_sigs_d, batch * sizeof(ECDSASignatureGPU)));
    CUDA_CHECK(cudaMalloc(&d_res, batch * sizeof(bool)));

    CUDA_CHECK(cudaMemcpy(d_msgs, h_msgs.data(), batch * 32, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pubs, h_pubkeys.data(), batch * sizeof(JacobianPoint), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sigs_d, h_sigs.data(), batch * sizeof(ECDSASignatureGPU), cudaMemcpyHostToDevice));

    int blocks = (batch + kThreads - 1) / kThreads;

    for (int i = 0; i < cfg.warmup_iterations; ++i) {
        ecdsa_verify_batch_kernel<<<blocks, kThreads>>>(d_msgs, d_pubs, d_sigs_d, d_res, batch);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaTimer timer;
    timer.start();
    for (int i = 0; i < cfg.measure_iterations; ++i) {
        ecdsa_verify_batch_kernel<<<blocks, kThreads>>>(d_msgs, d_pubs, d_sigs_d, d_res, batch);
    }
    float total_ms = timer.stop();

    double avg_ms = total_ms / cfg.measure_iterations;
    double throughput = (batch / avg_ms) / 1000.0;
    double ns_per_op = (avg_ms * 1e6) / batch;

    CUDA_CHECK(cudaFree(d_msgs));
    CUDA_CHECK(cudaFree(d_pubs));
    CUDA_CHECK(cudaFree(d_sigs_d));
    CUDA_CHECK(cudaFree(d_res));

    return {"ECDSA Verify", avg_ms, batch, throughput, ns_per_op};
}

BenchResult bench_schnorr_sign(const BenchConfig& cfg) {
    constexpr int kThreads = 128;
    int batch = std::min(cfg.batch_size, 1 << 14);

    std::vector<Scalar> h_privkeys(batch);
    std::vector<uint8_t> h_msgs(batch * 32);
    std::vector<uint8_t> h_aux(batch * 32);

    std::mt19937_64 rng(0xBEEF);
    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < 32; j++) h_msgs[i * 32 + j] = (uint8_t)(rng() & 0xFF);
        for (int j = 0; j < 32; j++) h_aux[i * 32 + j] = (uint8_t)(rng() & 0xFF);
        for (int j = 0; j < 4; j++) h_privkeys[i].limbs[j] = rng();
        h_privkeys[i].limbs[3] &= 0x7FFFFFFFFFFFFFFFULL;
        if (h_privkeys[i].limbs[0] == 0 && h_privkeys[i].limbs[1] == 0 &&
            h_privkeys[i].limbs[2] == 0 && h_privkeys[i].limbs[3] == 0)
            h_privkeys[i].limbs[0] = 1;
    }

    Scalar *d_priv;
    uint8_t *d_msgs, *d_aux;
    SchnorrSignatureGPU *d_sigs;
    bool *d_res;

    CUDA_CHECK(cudaMalloc(&d_priv, batch * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_msgs, batch * 32));
    CUDA_CHECK(cudaMalloc(&d_aux, batch * 32));
    CUDA_CHECK(cudaMalloc(&d_sigs, batch * sizeof(SchnorrSignatureGPU)));
    CUDA_CHECK(cudaMalloc(&d_res, batch * sizeof(bool)));

    CUDA_CHECK(cudaMemcpy(d_priv, h_privkeys.data(), batch * sizeof(Scalar), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_msgs, h_msgs.data(), batch * 32, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_aux, h_aux.data(), batch * 32, cudaMemcpyHostToDevice));

    int blocks = (batch + kThreads - 1) / kThreads;

    for (int i = 0; i < cfg.warmup_iterations; ++i) {
        schnorr_sign_batch_kernel<<<blocks, kThreads>>>(d_priv, d_msgs, d_aux, d_sigs, d_res, batch);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaTimer timer;
    timer.start();
    for (int i = 0; i < cfg.measure_iterations; ++i) {
        schnorr_sign_batch_kernel<<<blocks, kThreads>>>(d_priv, d_msgs, d_aux, d_sigs, d_res, batch);
    }
    float total_ms = timer.stop();

    double avg_ms = total_ms / cfg.measure_iterations;
    double throughput = (batch / avg_ms) / 1000.0;
    double ns_per_op = (avg_ms * 1e6) / batch;

    CUDA_CHECK(cudaFree(d_priv));
    CUDA_CHECK(cudaFree(d_msgs));
    CUDA_CHECK(cudaFree(d_aux));
    CUDA_CHECK(cudaFree(d_sigs));
    CUDA_CHECK(cudaFree(d_res));

    return {"Schnorr Sign", avg_ms, batch, throughput, ns_per_op};
}

BenchResult bench_schnorr_verify(const BenchConfig& cfg) {
    constexpr int kThreads = 128;
    int batch = std::min(cfg.batch_size, 1 << 14);

    // Prepare: sign on GPU first to get valid sigs + pubkeys
    std::vector<Scalar> h_privkeys(batch);
    std::vector<uint8_t> h_msgs(batch * 32);
    std::vector<uint8_t> h_aux(batch * 32);

    std::mt19937_64 rng(0xBEEF02);
    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < 32; j++) h_msgs[i * 32 + j] = (uint8_t)(rng() & 0xFF);
        for (int j = 0; j < 32; j++) h_aux[i * 32 + j] = (uint8_t)(rng() & 0xFF);
        for (int j = 0; j < 4; j++) h_privkeys[i].limbs[j] = rng();
        h_privkeys[i].limbs[3] &= 0x7FFFFFFFFFFFFFFFULL;
        if (h_privkeys[i].limbs[0] == 0 && h_privkeys[i].limbs[1] == 0 &&
            h_privkeys[i].limbs[2] == 0 && h_privkeys[i].limbs[3] == 0)
            h_privkeys[i].limbs[0] = 1;
    }

    // Sign on GPU
    Scalar *d_priv;
    uint8_t *d_msgs, *d_aux;
    SchnorrSignatureGPU *d_sigs;
    bool *d_res;

    CUDA_CHECK(cudaMalloc(&d_priv, batch * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_msgs, batch * 32));
    CUDA_CHECK(cudaMalloc(&d_aux, batch * 32));
    CUDA_CHECK(cudaMalloc(&d_sigs, batch * sizeof(SchnorrSignatureGPU)));
    CUDA_CHECK(cudaMalloc(&d_res, batch * sizeof(bool)));

    CUDA_CHECK(cudaMemcpy(d_priv, h_privkeys.data(), batch * sizeof(Scalar), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_msgs, h_msgs.data(), batch * 32, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_aux, h_aux.data(), batch * 32, cudaMemcpyHostToDevice));

    int blocks = (batch + kThreads - 1) / kThreads;
    schnorr_sign_batch_kernel<<<blocks, kThreads>>>(d_priv, d_msgs, d_aux, d_sigs, d_res, batch);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Generate x-only pubkeys from private keys
    JacobianPoint *d_pubs;
    CUDA_CHECK(cudaMalloc(&d_pubs, batch * sizeof(JacobianPoint)));
    generator_mul_batch_kernel<<<blocks, kThreads>>>(d_priv, d_pubs, batch);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Convert to x-only pubkeys on host
    std::vector<JacobianPoint> h_pubs(batch);
    CUDA_CHECK(cudaMemcpy(h_pubs.data(), d_pubs, batch * sizeof(JacobianPoint), cudaMemcpyDeviceToHost));

    // For Schnorr verify, we need x-only pubkeys as 32-byte big-endian
    // We'll use the Jacobian x-coordinate as-is (already normalized during sign)
    // Actually we need affine x. Let's extract from the sign output instead.
    // Simpler: use the field_to_bytes approach inline on GPU.
    // For simplicity and correctness: use a small helper kernel.
    // But the Schnorr verify bench needs only timing of verify, so we can:
    // 1. Transfer signed sigs to host
    // 2. Compute pubkey x-only bytes on host (rough approximation)
    // Actually, let's just do a sign+extract approach on device side.

    // Generate pubkey_x bytes: we need to do scalar_mul + convert to affine + field_to_bytes
    // This is complex. Simpler: use a known test vector approach.
    // For benchmark purposes, we'll use a trick: sign first, extract pubkeys via sign's P computation.
    // The cleanest way: create a small extraction kernel.

    // Actually, Schnorr verify's pubkey_x is just the x-coordinate in big-endian bytes.
    // We have Jacobian points. Let's convert to affine on host (too slow for batch).
    // Better: allocate a separate extraction kernel that does G*k -> affine -> x_bytes.

    // For now, let's do a simpler approach with a small helper kernel in this file.
    // We'll create pubkey x bytes using a GPU helper and then benchmark verify.

    // Instead of complexity, let's just create test inputs where we know the pubkey.
    // Use the BIP-340 test approach: derive pubkey from privkey on GPU.

    // Clean approach: generate x-only pubkeys using a local kernel
    struct XOnlyExtractHelper {
        // We can't define a kernel here, so let's just copy Jacobian points to host
        // and extract x-only manually. For benchmark purposes this prep time doesn't matter.
    };

    // Simple host-side x extraction (only for test data prep -- not benchmarked)
    // This is a rough approximation: the actual Jacobian->affine involves field_inv
    // which we can't call from host. So let's use a different approach:
    // Sign a known message with privkey, the sign function internally computes P.
    // The schnorr_verify takes pubkey_x as bytes -- we need the x-only pubkey.
    // Let's compute it by running scalar_mul on GPU and converting to affine.

    // Actually, let's just allocate and generate x-only pubkeys on GPU with a custom approach.
    // The simplest: create byte arrays from the sign operation.
    // schnorr_sign internally computes P = d * G and gets px as bytes.
    // We can extract px from a modified sign or just compute G*k and extract.

    // Simpler approach: use ecdsa_sign which gives us Q, then extract x bytes.
    // Or even simpler: use generator_mul + jac_to_affine + field_to_bytes all on GPU.

    // Let's do it the clean way with a dedicated extraction kernel defined in bench file:
    // No, we can't define __global__ functions inside other functions.
    // Let's just compute pubkey bytes using the jacobian_to_affine bench kernel + memcpy.

    // SIMPLEST APPROACH: extract x-only bytes from JacobianPoint on CPU
    // by using field inversions. But we don't have CPU field_inv...
    // OK, the simplest correct approach: use the bench_jac_to_affine_kernel to convert,
    // then read back and extract x bytes.

    // Step 1: Convert Jacobian to affine x,y using existing bench kernel
    FieldElement *d_x, *d_y, *d_z;
    CUDA_CHECK(cudaMalloc(&d_x, batch * sizeof(FieldElement)));
    CUDA_CHECK(cudaMalloc(&d_y, batch * sizeof(FieldElement)));
    CUDA_CHECK(cudaMalloc(&d_z, batch * sizeof(FieldElement)));

    // Extract x, y, z from JacobianPoints on host then upload
    std::vector<FieldElement> h_x(batch), h_y(batch), h_z(batch);
    for (int i = 0; i < batch; i++) {
        h_x[i] = h_pubs[i].x;
        h_y[i] = h_pubs[i].y;
        h_z[i] = h_pubs[i].z;
    }
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), batch * sizeof(FieldElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), batch * sizeof(FieldElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, h_z.data(), batch * sizeof(FieldElement), cudaMemcpyHostToDevice));

    bench_jac_to_affine_kernel<<<blocks, kThreads>>>(d_x, d_y, d_z, batch);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back affine x coordinates
    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, batch * sizeof(FieldElement), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, batch * sizeof(FieldElement), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_pubs));

    // Convert affine x to big-endian bytes for schnorr_verify
    // FieldElement limbs are LE; field_to_bytes produces BE.
    // We need to replicate field_to_bytes on host.
    std::vector<uint8_t> h_pubkey_x(batch * 32);
    for (int i = 0; i < batch; i++) {
        // Schnorr BIP-340 requires even Y; if Y is odd, negate the private key.
        // The sign function already handles this internally, but we need the
        // correct x-only pubkey that matches what Schnorr sign used.
        // field_to_bytes: BE bytes from LE limbs
        for (int limb = 3; limb >= 0; limb--) {
            uint64_t v = h_x[i].limbs[limb];
            int offset = i * 32 + (3 - limb) * 8;
            for (int b = 7; b >= 0; b--) {
                h_pubkey_x[offset + (7 - b)] = (uint8_t)(v >> (b * 8));
            }
        }
    }

    // Upload verify data
    uint8_t *d_pubkey_x;
    CUDA_CHECK(cudaMalloc(&d_pubkey_x, batch * 32));
    CUDA_CHECK(cudaMemcpy(d_pubkey_x, h_pubkey_x.data(), batch * 32, cudaMemcpyHostToDevice));

    // Verify benchmark
    bool *d_vres;
    CUDA_CHECK(cudaMalloc(&d_vres, batch * sizeof(bool)));

    for (int i = 0; i < cfg.warmup_iterations; ++i) {
        schnorr_verify_batch_kernel<<<blocks, kThreads>>>(d_pubkey_x, d_msgs, d_sigs, d_vres, batch);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaTimer timer;
    timer.start();
    for (int i = 0; i < cfg.measure_iterations; ++i) {
        schnorr_verify_batch_kernel<<<blocks, kThreads>>>(d_pubkey_x, d_msgs, d_sigs, d_vres, batch);
    }
    float total_ms = timer.stop();

    double avg_ms = total_ms / cfg.measure_iterations;
    double throughput = (batch / avg_ms) / 1000.0;
    double ns_per_op = (avg_ms * 1e6) / batch;

    CUDA_CHECK(cudaFree(d_priv));
    CUDA_CHECK(cudaFree(d_msgs));
    CUDA_CHECK(cudaFree(d_aux));
    CUDA_CHECK(cudaFree(d_sigs));
    CUDA_CHECK(cudaFree(d_res));
    CUDA_CHECK(cudaFree(d_pubkey_x));
    CUDA_CHECK(cudaFree(d_vres));

    return {"Schnorr Verify", avg_ms, batch, throughput, ns_per_op};
}

BenchResult bench_ecdsa_sign_recoverable(const BenchConfig& cfg) {
    constexpr int kThreads = 128;
    int batch = std::min(cfg.batch_size, 1 << 14);

    std::vector<uint8_t> h_msgs(batch * 32);
    std::vector<Scalar> h_privkeys(batch);

    std::mt19937_64 rng(0xBEC0);
    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < 32; j++) h_msgs[i * 32 + j] = (uint8_t)(rng() & 0xFF);
        for (int j = 0; j < 4; j++) h_privkeys[i].limbs[j] = rng();
        h_privkeys[i].limbs[3] &= 0x7FFFFFFFFFFFFFFFULL;
        if (h_privkeys[i].limbs[0] == 0 && h_privkeys[i].limbs[1] == 0 &&
            h_privkeys[i].limbs[2] == 0 && h_privkeys[i].limbs[3] == 0)
            h_privkeys[i].limbs[0] = 1;
    }

    uint8_t *d_msgs; Scalar *d_priv;
    RecoverableSignatureGPU *d_rsigs;
    bool *d_res;

    CUDA_CHECK(cudaMalloc(&d_msgs, batch * 32));
    CUDA_CHECK(cudaMalloc(&d_priv, batch * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_rsigs, batch * sizeof(RecoverableSignatureGPU)));
    CUDA_CHECK(cudaMalloc(&d_res, batch * sizeof(bool)));

    CUDA_CHECK(cudaMemcpy(d_msgs, h_msgs.data(), batch * 32, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_priv, h_privkeys.data(), batch * sizeof(Scalar), cudaMemcpyHostToDevice));

    int blocks = (batch + kThreads - 1) / kThreads;

    for (int i = 0; i < cfg.warmup_iterations; ++i) {
        ecdsa_sign_recoverable_batch_kernel<<<blocks, kThreads>>>(d_msgs, d_priv, d_rsigs, d_res, batch);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaTimer timer;
    timer.start();
    for (int i = 0; i < cfg.measure_iterations; ++i) {
        ecdsa_sign_recoverable_batch_kernel<<<blocks, kThreads>>>(d_msgs, d_priv, d_rsigs, d_res, batch);
    }
    float total_ms = timer.stop();

    double avg_ms = total_ms / cfg.measure_iterations;
    double throughput = (batch / avg_ms) / 1000.0;
    double ns_per_op = (avg_ms * 1e6) / batch;

    CUDA_CHECK(cudaFree(d_msgs));
    CUDA_CHECK(cudaFree(d_priv));
    CUDA_CHECK(cudaFree(d_rsigs));
    CUDA_CHECK(cudaFree(d_res));

    return {"ECDSA Sign+Recid", avg_ms, batch, throughput, ns_per_op};
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
                  << r.time_per_op_ns / 1000 << " us";
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

    // Affine point operations
    std::cout << "\n=== Affine Point Addition ===\n";
    results.push_back(bench_affine_add(cfg));
    print_result(results.back());

    results.push_back(bench_affine_add_lambda(cfg));
    print_result(results.back());

    results.push_back(bench_affine_add_xonly(cfg));
    print_result(results.back());

    results.push_back(bench_batch_inversion(cfg));
    print_result(results.back());

    results.push_back(bench_jacobian_to_affine(cfg));
    print_result(results.back());

    // Signature operations
    std::cout << "\n=== ECDSA Signatures ===\n";
    results.push_back(bench_ecdsa_sign(cfg));
    print_result(results.back());

    results.push_back(bench_ecdsa_verify(cfg));
    print_result(results.back());

    results.push_back(bench_ecdsa_sign_recoverable(cfg));
    print_result(results.back());

    std::cout << "\n=== Schnorr Signatures (BIP-340) ===\n";
    results.push_back(bench_schnorr_sign(cfg));
    print_result(results.back());

    results.push_back(bench_schnorr_verify(cfg));
    print_result(results.back());

    // Print summary table
    print_summary_table(results);

    std::cout << "Benchmark complete.\n";

    return 0;
}

