// ============================================================================
// GPU Unified Benchmark -- UltrafastSecp256k1
// ============================================================================
//
// Unified GPU benchmark for CUDA / ROCm (HIP).
// Mirrors the CPU bench_unified in structure and output format.
//
// Single binary that benchmarks ALL GPU operations and produces structured
// JSON + text reports identical in schema to the CPU benchmark system.
//
// Categories (matching CPU where applicable):
//   1. Field Arithmetic     (mul, sqr, add, sub, inv)
//   2. Point Operations     (add, double, Jac->Affine)
//   3. Scalar Mul           (k*G, k*P batch)
//   4. Affine Pipeline      (affine add, batch inverse)
//   5. ECDSA                (sign, verify, recoverable sign)
//   6. Schnorr/BIP-340      (sign, verify)
//   7. Throughput Summary    (derived, ops/sec)
//
// Usage:
//   gpu_bench_unified                       # run all, print table
//   gpu_bench_unified --json <file>         # JSON report
//   gpu_bench_unified --suite core|all      # subset
//   gpu_bench_unified --quick               # CI mode (small batches)
//   gpu_bench_unified --batch <N>           # custom batch size
//   gpu_bench_unified --device <id>         # GPU device
//   gpu_bench_unified --passes <N>          # measurement passes
//
// Methodology:
//   - CUDA event timing (cudaEventRecord/Synchronize)
//   - 3 warmup iterations, 5 measurement passes (--passes N)
//   - IQR outlier removal on measurement passes, median
//   - Batch-level throughput (ops/sec) + per-element cost (ns/op)
//   - Fixed RNG seed for reproducibility
//
// ============================================================================

#include "secp256k1.cuh"
#include "affine_add.cuh"
#include "batch_inversion.cuh"
#include "ecdsa.cuh"
#include "schnorr.cuh"
#include "recovery.cuh"
#include "ct/ct_ops.cuh"
#include "ct/ct_field.cuh"
#include "ct/ct_scalar.cuh"
#include "ct/ct_point.cuh"
#include "ct/ct_sign.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#endif

#if __has_include("secp256k1/version.hpp")
#include "secp256k1/version.hpp"
#endif
#ifndef SECP256K1_VERSION_STRING
#define SECP256K1_VERSION_STRING "unknown"
#endif

#ifndef GIT_HASH
#define GIT_HASH "unknown"
#endif

using namespace secp256k1::cuda;

// ============================================================================
// Error checking
// ============================================================================
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                     cudaGetErrorString(err), __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while(0)

// ============================================================================
// Benchmark configuration
// ============================================================================
struct BenchConfig {
    int warmup_iterations = 3;
    int measure_passes = 5;
    int batch_size = 1 << 20;    // 1M elements
    int threads_per_block = 256;
    bool quick = false;
    std::string suite = "all";   // "core" or "all"
    std::string json_path;
    int device_id = 0;
};

// ============================================================================
// JSON result collector (fixed-size, no heap in hot path)
// ============================================================================
struct BenchEntry {
    char section[64];
    char name[64];
    double ns_per_op;
    double throughput_mops;
    int batch_size;
};

static constexpr int MAX_BENCH_ENTRIES = 128;

struct BenchReport {
    BenchEntry entries[MAX_BENCH_ENTRIES];
    int count;

    // GPU metadata
    char gpu_name[256];
    char compute_cap[16];
    int sm_count;
    int clock_mhz;
    size_t memory_mb;
    char backend[32];
    char driver[32];
    int passes;
    int warmup;
    int batch_size;
    char compiler[64];
    char arch[32];

    void add(const char* section, const char* name,
             double ns_val, double mops, int bsz) {
        if (count >= MAX_BENCH_ENTRIES) return;
        auto& e = entries[count++];
        std::snprintf(e.section, sizeof(e.section), "%s", section);
        std::snprintf(e.name, sizeof(e.name), "%s", name);
        e.ns_per_op = ns_val;
        e.throughput_mops = mops;
        e.batch_size = bsz;
    }
};

// ============================================================================
// Timer helper with CUDA events
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
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
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
// Statistical helper -- median with IQR outlier removal
// ============================================================================
static double median_iqr(std::vector<double>& samples) {
    if (samples.empty()) return 0.0;
    std::sort(samples.begin(), samples.end());

    int n = (int)samples.size();
    if (n < 4) {
        // Not enough for IQR, just return median
        return samples[n / 2];
    }

    // IQR outlier removal
    double q1 = samples[n / 4];
    double q3 = samples[3 * n / 4];
    double iqr = q3 - q1;
    double lo = q1 - 1.5 * iqr;
    double hi = q3 + 1.5 * iqr;

    std::vector<double> filtered;
    for (double s : samples) {
        if (s >= lo && s <= hi) filtered.push_back(s);
    }
    if (filtered.empty()) return samples[n / 2];
    return filtered[filtered.size() / 2];
}

// ============================================================================
// Data generation (fixed seeds for reproducibility)
// ============================================================================
static void gen_field_elements(FieldElement* h, int count, uint64_t seed) {
    std::mt19937_64 rng(seed);
    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < 4; ++j) h[i].limbs[j] = rng();
        h[i].limbs[3] &= 0x7FFFFFFFFFFFFFFFULL;
    }
}

static void gen_scalars(Scalar* h, int count, uint64_t seed) {
    std::mt19937_64 rng(seed);
    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < 4; ++j) h[i].limbs[j] = rng();
        h[i].limbs[3] &= 0x7FFFFFFFFFFFFFFFULL;
    }
}

static void gen_jacobian_points(JacobianPoint* h, int count, uint64_t seed) {
    std::mt19937_64 rng(seed);
    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < 4; ++j) {
            h[i].x.limbs[j] = rng();
            h[i].y.limbs[j] = rng();
            h[i].z.limbs[j] = (j == 0) ? 1 : 0;
        }
        h[i].x.limbs[3] &= 0x7FFFFFFFFFFFFFFFULL;
        h[i].y.limbs[3] &= 0x7FFFFFFFFFFFFFFFULL;
        h[i].infinity = false;
    }
}

// ============================================================================
// Benchmark result
// ============================================================================
struct BenchResult {
    const char* section;
    const char* name;
    double time_ms;
    int batch_size;
    double throughput_mops;
    double ns_per_op;
};

// ============================================================================
// Benchmark harness -- run kernel N times, collect timings
// ============================================================================
template<typename KernelFunc>
static BenchResult run_bench(const char* section, const char* name,
                              const BenchConfig& cfg, KernelFunc&& kernel_fn) {
    CudaTimer timer;

    // Warmup
    for (int i = 0; i < cfg.warmup_iterations; ++i) {
        kernel_fn();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Measure
    std::vector<double> samples;
    samples.reserve(cfg.measure_passes);
    for (int i = 0; i < cfg.measure_passes; ++i) {
        timer.start();
        kernel_fn();
        float ms = timer.stop();
        samples.push_back(ms);
    }

    double median_ms = median_iqr(samples);
    double ns_per_op = (median_ms * 1e6) / cfg.batch_size;
    double mops = cfg.batch_size / (median_ms * 1e3);

    return { section, name, median_ms, cfg.batch_size, mops, ns_per_op };
}

// ============================================================================
// Kernel wrappers (reuse from bench_cuda.cu but inline here for unified binary)
// ============================================================================

__global__ void bench_field_mul_k(const FieldElement* a, const FieldElement* b,
                                   FieldElement* r, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) field_mul(&a[idx], &b[idx], &r[idx]);
}

__global__ void bench_field_add_k(const FieldElement* a, const FieldElement* b,
                                   FieldElement* r, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) field_add(&a[idx], &b[idx], &r[idx]);
}

__global__ void bench_field_sub_k(const FieldElement* a, const FieldElement* b,
                                   FieldElement* r, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) field_sub(&a[idx], &b[idx], &r[idx]);
}

__global__ void bench_field_inv_k(const FieldElement* a, FieldElement* r, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) field_inv(&a[idx], &r[idx]);
}

__global__ void bench_field_sqr_k(const FieldElement* a, FieldElement* r, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) field_mul(&a[idx], &a[idx], &r[idx]);
}

__global__ void bench_point_add_k(const JacobianPoint* a, const JacobianPoint* b,
                                   JacobianPoint* r, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) jacobian_add(&a[idx], &b[idx], &r[idx]);
}

__global__ void bench_point_dbl_k(const JacobianPoint* a, JacobianPoint* r, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) jacobian_double(&a[idx], &r[idx]);
}

__global__ void bench_scalar_mul_gen_k(const Scalar* k, JacobianPoint* r, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) scalar_mul_generator_const(&k[idx], &r[idx]);
}

__global__ void bench_affine_add_k(
    const FieldElement* px, const FieldElement* py,
    const FieldElement* qx, const FieldElement* qy,
    FieldElement* rx, FieldElement* ry, int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        secp256k1::cuda::affine_add(&px[idx], &py[idx], &qx[idx], &qy[idx],
                                    &rx[idx], &ry[idx]);
    }
}

__global__ void bench_jac_to_affine_k(
    FieldElement* x, FieldElement* y, const FieldElement* z, int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) secp256k1::cuda::jacobian_to_affine(&x[idx], &y[idx], &z[idx]);
}

#if !SECP256K1_CUDA_LIMBS_32
__global__ void bench_ecdsa_sign_k(const uint8_t* msgs, const Scalar* privkeys,
                                    ECDSASignatureGPU* sigs, bool* oks, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        oks[idx] = secp256k1::cuda::ecdsa_sign(
            &msgs[idx * 32], &privkeys[idx], &sigs[idx]);
    }
}

__global__ void bench_schnorr_sign_k(const Scalar* privkeys, const uint8_t* msgs,
                                      const uint8_t* aux_rands,
                                      SchnorrSignatureGPU* sigs, bool* oks, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        oks[idx] = secp256k1::cuda::schnorr_sign(
            &privkeys[idx], &msgs[idx * 32], &aux_rands[idx * 32], &sigs[idx]);
    }
}
#endif

// ============================================================================
// CT (Constant-Time) benchmark kernels
// ============================================================================

__global__ void bench_ct_generator_mul_k(const Scalar* k, JacobianPoint* r, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) secp256k1::cuda::ct::ct_generator_mul(&k[idx], &r[idx]);
}

__global__ void bench_ct_scalar_mul_k(const JacobianPoint* pts, const Scalar* k,
                                      JacobianPoint* r, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) secp256k1::cuda::ct::ct_scalar_mul(&pts[idx], &k[idx], &r[idx]);
}

#if !SECP256K1_CUDA_LIMBS_32
__global__ void bench_ct_ecdsa_sign_k(const uint8_t* msgs, const Scalar* privkeys,
                                      ECDSASignatureGPU* sigs, bool* oks, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        oks[idx] = secp256k1::cuda::ct::ct_ecdsa_sign(
            &msgs[idx * 32], &privkeys[idx], &sigs[idx]);
    }
}

__global__ void bench_ct_schnorr_sign_k(const Scalar* privkeys, const uint8_t* msgs,
                                         const uint8_t* aux_rands,
                                         SchnorrSignatureGPU* sigs, bool* oks, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        oks[idx] = secp256k1::cuda::ct::ct_schnorr_sign(
            &privkeys[idx], &msgs[idx * 32], &aux_rands[idx * 32], &sigs[idx]);
    }
}
#endif

// ============================================================================
// Benchmark implementations -- one per operation
// ============================================================================

static BenchResult bench_field_mul_impl(const BenchConfig& cfg) {
    int N = cfg.batch_size;
    std::vector<FieldElement> h_a(N), h_b(N);
    gen_field_elements(h_a.data(), N, 42);
    gen_field_elements(h_b.data(), N, 43);

    FieldElement *d_a, *d_b, *d_r;
    size_t sz = N * sizeof(FieldElement);
    CUDA_CHECK(cudaMalloc(&d_a, sz));
    CUDA_CHECK(cudaMalloc(&d_b, sz));
    CUDA_CHECK(cudaMalloc(&d_r, sz));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), sz, cudaMemcpyHostToDevice));

    int blocks = (N + cfg.threads_per_block - 1) / cfg.threads_per_block;
    auto result = run_bench("Field Arithmetic", "mul", cfg, [&]() {
        bench_field_mul_k<<<blocks, cfg.threads_per_block>>>(d_a, d_b, d_r, N);
    });

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_r);
    return result;
}

static BenchResult bench_field_sqr_impl(const BenchConfig& cfg) {
    int N = cfg.batch_size;
    std::vector<FieldElement> h_a(N);
    gen_field_elements(h_a.data(), N, 44);

    FieldElement *d_a, *d_r;
    size_t sz = N * sizeof(FieldElement);
    CUDA_CHECK(cudaMalloc(&d_a, sz));
    CUDA_CHECK(cudaMalloc(&d_r, sz));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), sz, cudaMemcpyHostToDevice));

    int blocks = (N + cfg.threads_per_block - 1) / cfg.threads_per_block;
    auto result = run_bench("Field Arithmetic", "sqr", cfg, [&]() {
        bench_field_sqr_k<<<blocks, cfg.threads_per_block>>>(d_a, d_r, N);
    });

    cudaFree(d_a); cudaFree(d_r);
    return result;
}

static BenchResult bench_field_add_impl(const BenchConfig& cfg) {
    int N = cfg.batch_size;
    std::vector<FieldElement> h_a(N), h_b(N);
    gen_field_elements(h_a.data(), N, 45);
    gen_field_elements(h_b.data(), N, 46);

    FieldElement *d_a, *d_b, *d_r;
    size_t sz = N * sizeof(FieldElement);
    CUDA_CHECK(cudaMalloc(&d_a, sz));
    CUDA_CHECK(cudaMalloc(&d_b, sz));
    CUDA_CHECK(cudaMalloc(&d_r, sz));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), sz, cudaMemcpyHostToDevice));

    int blocks = (N + cfg.threads_per_block - 1) / cfg.threads_per_block;
    auto result = run_bench("Field Arithmetic", "add", cfg, [&]() {
        bench_field_add_k<<<blocks, cfg.threads_per_block>>>(d_a, d_b, d_r, N);
    });

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_r);
    return result;
}

static BenchResult bench_field_sub_impl(const BenchConfig& cfg) {
    int N = cfg.batch_size;
    std::vector<FieldElement> h_a(N), h_b(N);
    gen_field_elements(h_a.data(), N, 47);
    gen_field_elements(h_b.data(), N, 48);

    FieldElement *d_a, *d_b, *d_r;
    size_t sz = N * sizeof(FieldElement);
    CUDA_CHECK(cudaMalloc(&d_a, sz));
    CUDA_CHECK(cudaMalloc(&d_b, sz));
    CUDA_CHECK(cudaMalloc(&d_r, sz));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), sz, cudaMemcpyHostToDevice));

    int blocks = (N + cfg.threads_per_block - 1) / cfg.threads_per_block;
    auto result = run_bench("Field Arithmetic", "sub", cfg, [&]() {
        bench_field_sub_k<<<blocks, cfg.threads_per_block>>>(d_a, d_b, d_r, N);
    });

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_r);
    return result;
}

static BenchResult bench_field_inv_impl(const BenchConfig& cfg) {
    // Use smaller batch for inverse (expensive operation)
    int N = std::min(cfg.batch_size, 1 << 16);
    std::vector<FieldElement> h_a(N);
    gen_field_elements(h_a.data(), N, 49);
    // Ensure non-zero
    for (int i = 0; i < N; ++i) {
        if (h_a[i].limbs[0] == 0 && h_a[i].limbs[1] == 0 &&
            h_a[i].limbs[2] == 0 && h_a[i].limbs[3] == 0)
            h_a[i].limbs[0] = 1;
    }

    FieldElement *d_a, *d_r;
    size_t sz = N * sizeof(FieldElement);
    CUDA_CHECK(cudaMalloc(&d_a, sz));
    CUDA_CHECK(cudaMalloc(&d_r, sz));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), sz, cudaMemcpyHostToDevice));

    int blocks = (N + cfg.threads_per_block - 1) / cfg.threads_per_block;
    BenchConfig inv_cfg = cfg;
    inv_cfg.batch_size = N;
    auto result = run_bench("Field Arithmetic", "inv", inv_cfg, [&]() {
        bench_field_inv_k<<<blocks, cfg.threads_per_block>>>(d_a, d_r, N);
    });

    cudaFree(d_a); cudaFree(d_r);
    return result;
}

static BenchResult bench_point_add_impl(const BenchConfig& cfg) {
    int N = std::min(cfg.batch_size, 1 << 18);
    std::vector<JacobianPoint> h_a(N), h_b(N);
    gen_jacobian_points(h_a.data(), N, 51);
    gen_jacobian_points(h_b.data(), N, 52);

    JacobianPoint *d_a, *d_b, *d_r;
    size_t sz = N * sizeof(JacobianPoint);
    CUDA_CHECK(cudaMalloc(&d_a, sz));
    CUDA_CHECK(cudaMalloc(&d_b, sz));
    CUDA_CHECK(cudaMalloc(&d_r, sz));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), sz, cudaMemcpyHostToDevice));

    int blocks = (N + cfg.threads_per_block - 1) / cfg.threads_per_block;
    BenchConfig pt_cfg = cfg;
    pt_cfg.batch_size = N;
    auto result = run_bench("Point Operations", "add (Jacobian)", pt_cfg, [&]() {
        bench_point_add_k<<<blocks, cfg.threads_per_block>>>(d_a, d_b, d_r, N);
    });

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_r);
    return result;
}

static BenchResult bench_point_dbl_impl(const BenchConfig& cfg) {
    int N = std::min(cfg.batch_size, 1 << 18);
    std::vector<JacobianPoint> h_a(N);
    gen_jacobian_points(h_a.data(), N, 53);

    JacobianPoint *d_a, *d_r;
    size_t sz = N * sizeof(JacobianPoint);
    CUDA_CHECK(cudaMalloc(&d_a, sz));
    CUDA_CHECK(cudaMalloc(&d_r, sz));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), sz, cudaMemcpyHostToDevice));

    int blocks = (N + cfg.threads_per_block - 1) / cfg.threads_per_block;
    BenchConfig pt_cfg = cfg;
    pt_cfg.batch_size = N;
    auto result = run_bench("Point Operations", "double", pt_cfg, [&]() {
        bench_point_dbl_k<<<blocks, cfg.threads_per_block>>>(d_a, d_r, N);
    });

    cudaFree(d_a); cudaFree(d_r);
    return result;
}

static BenchResult bench_jac_to_affine_impl(const BenchConfig& cfg) {
    int N = std::min(cfg.batch_size, 1 << 16);
    std::vector<FieldElement> h_x(N), h_y(N), h_z(N);
    std::mt19937_64 rng(54);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < 4; ++j) {
            h_x[i].limbs[j] = rng();
            h_y[i].limbs[j] = rng();
            h_z[i].limbs[j] = rng();
        }
        h_x[i].limbs[3] &= 0x7FFFFFFFFFFFFFFFULL;
        h_y[i].limbs[3] &= 0x7FFFFFFFFFFFFFFFULL;
        h_z[i].limbs[3] &= 0x7FFFFFFFFFFFFFFFULL;
        if (h_z[i].limbs[0] == 0 && h_z[i].limbs[1] == 0 &&
            h_z[i].limbs[2] == 0 && h_z[i].limbs[3] == 0)
            h_z[i].limbs[0] = 1;
    }

    FieldElement *d_x, *d_y, *d_z;
    size_t sz = N * sizeof(FieldElement);
    CUDA_CHECK(cudaMalloc(&d_x, sz));
    CUDA_CHECK(cudaMalloc(&d_y, sz));
    CUDA_CHECK(cudaMalloc(&d_z, sz));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, h_z.data(), sz, cudaMemcpyHostToDevice));

    int blocks = (N + cfg.threads_per_block - 1) / cfg.threads_per_block;
    BenchConfig pt_cfg = cfg;
    pt_cfg.batch_size = N;
    auto result = run_bench("Point Operations", "Jac->Affine", pt_cfg, [&]() {
        bench_jac_to_affine_k<<<blocks, cfg.threads_per_block>>>(d_x, d_y, d_z, N);
    });

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    return result;
}

static BenchResult bench_scalar_mul_gen_impl(const BenchConfig& cfg) {
    int N = std::min(cfg.batch_size, 1 << 16);
    std::vector<Scalar> h_k(N);
    gen_scalars(h_k.data(), N, 55);

    Scalar* d_k;
    JacobianPoint* d_r;
    CUDA_CHECK(cudaMalloc(&d_k, N * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_r, N * sizeof(JacobianPoint)));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), N * sizeof(Scalar), cudaMemcpyHostToDevice));

    int blocks = (N + cfg.threads_per_block - 1) / cfg.threads_per_block;
    BenchConfig sm_cfg = cfg;
    sm_cfg.batch_size = N;
    auto result = run_bench("Scalar Multiplication", "k*G (generator)", sm_cfg, [&]() {
        bench_scalar_mul_gen_k<<<blocks, cfg.threads_per_block>>>(d_k, d_r, N);
    });

    cudaFree(d_k); cudaFree(d_r);
    return result;
}

static BenchResult bench_affine_add_impl(const BenchConfig& cfg) {
    int N = cfg.batch_size;
    std::vector<FieldElement> h_px(N), h_py(N), h_qx(N), h_qy(N);
    gen_field_elements(h_px.data(), N, 56);
    gen_field_elements(h_py.data(), N, 57);
    gen_field_elements(h_qx.data(), N, 58);
    gen_field_elements(h_qy.data(), N, 59);

    FieldElement *d_px, *d_py, *d_qx, *d_qy, *d_rx, *d_ry;
    size_t sz = N * sizeof(FieldElement);
    CUDA_CHECK(cudaMalloc(&d_px, sz)); CUDA_CHECK(cudaMalloc(&d_py, sz));
    CUDA_CHECK(cudaMalloc(&d_qx, sz)); CUDA_CHECK(cudaMalloc(&d_qy, sz));
    CUDA_CHECK(cudaMalloc(&d_rx, sz)); CUDA_CHECK(cudaMalloc(&d_ry, sz));
    CUDA_CHECK(cudaMemcpy(d_px, h_px.data(), sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, h_py.data(), sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qx, h_qx.data(), sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qy, h_qy.data(), sz, cudaMemcpyHostToDevice));

    int blocks = (N + cfg.threads_per_block - 1) / cfg.threads_per_block;
    auto result = run_bench("Affine Pipeline", "affine add (2M+1S+inv)", cfg, [&]() {
        bench_affine_add_k<<<blocks, cfg.threads_per_block>>>(d_px, d_py, d_qx, d_qy, d_rx, d_ry, N);
    });

    cudaFree(d_px); cudaFree(d_py); cudaFree(d_qx); cudaFree(d_qy);
    cudaFree(d_rx); cudaFree(d_ry);
    return result;
}

#if !SECP256K1_CUDA_LIMBS_32
static BenchResult bench_ecdsa_sign_impl(const BenchConfig& cfg) {
    int N = std::min(cfg.batch_size, 1 << 14);  // Signing is expensive
    std::vector<Scalar> h_privkeys(N);
    std::vector<uint8_t> h_msgs(N * 32);
    std::mt19937_64 rng(60);

    // Generate valid private keys as Scalar
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < 4; ++j) h_privkeys[i].limbs[j] = rng();
        h_privkeys[i].limbs[3] &= 0x7FFFFFFFFFFFFFFFULL;
        if (h_privkeys[i].limbs[0] == 0 && h_privkeys[i].limbs[1] == 0 &&
            h_privkeys[i].limbs[2] == 0 && h_privkeys[i].limbs[3] == 0)
            h_privkeys[i].limbs[0] = 1;
    }
    for (int i = 0; i < N * 32; ++i)
        h_msgs[i] = static_cast<uint8_t>(rng() & 0xFF);

    Scalar* d_priv;
    uint8_t* d_msg;
    ECDSASignatureGPU* d_sig;
    bool* d_ok;
    CUDA_CHECK(cudaMalloc(&d_priv, N * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_msg, N * 32));
    CUDA_CHECK(cudaMalloc(&d_sig, N * sizeof(ECDSASignatureGPU)));
    CUDA_CHECK(cudaMalloc(&d_ok, N * sizeof(bool)));
    CUDA_CHECK(cudaMemcpy(d_priv, h_privkeys.data(), N * sizeof(Scalar), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_msg, h_msgs.data(), N * 32, cudaMemcpyHostToDevice));

    int blocks = (N + cfg.threads_per_block - 1) / cfg.threads_per_block;
    BenchConfig sign_cfg = cfg;
    sign_cfg.batch_size = N;
    auto result = run_bench("ECDSA", "sign (RFC 6979)", sign_cfg, [&]() {
        bench_ecdsa_sign_k<<<blocks, cfg.threads_per_block>>>(d_msg, d_priv, d_sig, d_ok, N);
    });

    cudaFree(d_priv); cudaFree(d_msg); cudaFree(d_sig); cudaFree(d_ok);
    return result;
}

static BenchResult bench_schnorr_sign_impl(const BenchConfig& cfg) {
    int N = std::min(cfg.batch_size, 1 << 14);
    std::vector<Scalar> h_privkeys(N);
    std::vector<uint8_t> h_msgs(N * 32), h_aux(N * 32);
    std::mt19937_64 rng(61);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < 4; ++j) h_privkeys[i].limbs[j] = rng();
        h_privkeys[i].limbs[3] &= 0x7FFFFFFFFFFFFFFFULL;
        if (h_privkeys[i].limbs[0] == 0 && h_privkeys[i].limbs[1] == 0 &&
            h_privkeys[i].limbs[2] == 0 && h_privkeys[i].limbs[3] == 0)
            h_privkeys[i].limbs[0] = 1;
    }
    for (int i = 0; i < N * 32; ++i) {
        h_msgs[i] = static_cast<uint8_t>(rng() & 0xFF);
        h_aux[i] = static_cast<uint8_t>(rng() & 0xFF);
    }

    Scalar* d_priv;
    uint8_t *d_msg, *d_aux;
    SchnorrSignatureGPU* d_sig;
    bool* d_ok;
    CUDA_CHECK(cudaMalloc(&d_priv, N * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_msg, N * 32));
    CUDA_CHECK(cudaMalloc(&d_aux, N * 32));
    CUDA_CHECK(cudaMalloc(&d_sig, N * sizeof(SchnorrSignatureGPU)));
    CUDA_CHECK(cudaMalloc(&d_ok, N * sizeof(bool)));
    CUDA_CHECK(cudaMemcpy(d_priv, h_privkeys.data(), N * sizeof(Scalar), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_msg, h_msgs.data(), N * 32, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_aux, h_aux.data(), N * 32, cudaMemcpyHostToDevice));

    int blocks = (N + cfg.threads_per_block - 1) / cfg.threads_per_block;
    BenchConfig sign_cfg = cfg;
    sign_cfg.batch_size = N;
    auto result = run_bench("Schnorr/BIP-340", "sign", sign_cfg, [&]() {
        bench_schnorr_sign_k<<<blocks, cfg.threads_per_block>>>(d_priv, d_msg, d_aux, d_sig, d_ok, N);
    });

    cudaFree(d_priv); cudaFree(d_msg); cudaFree(d_aux); cudaFree(d_sig); cudaFree(d_ok);
    return result;
}
#endif

// ============================================================================
// CT benchmark implementations
// ============================================================================

static BenchResult bench_ct_generator_mul_impl(const BenchConfig& cfg) {
    int N = std::min(cfg.batch_size, 1 << 14);  // CT k*G is expensive
    std::vector<Scalar> h_keys(N);
    std::mt19937_64 rng(70);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < 4; ++j) h_keys[i].limbs[j] = rng();
        h_keys[i].limbs[3] &= 0x7FFFFFFFFFFFFFFFULL;
        if (h_keys[i].limbs[0] == 0 && h_keys[i].limbs[1] == 0 &&
            h_keys[i].limbs[2] == 0 && h_keys[i].limbs[3] == 0)
            h_keys[i].limbs[0] = 1;
    }

    Scalar* d_keys;
    JacobianPoint* d_pts;
    CUDA_CHECK(cudaMalloc(&d_keys, N * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_pts, N * sizeof(JacobianPoint)));
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), N * sizeof(Scalar), cudaMemcpyHostToDevice));

    int blocks = (N + cfg.threads_per_block - 1) / cfg.threads_per_block;
    BenchConfig ct_cfg = cfg;
    ct_cfg.batch_size = N;
    auto result = run_bench("CT Point Ops", "ct::k*G (generator)", ct_cfg, [&]() {
        bench_ct_generator_mul_k<<<blocks, cfg.threads_per_block>>>(d_keys, d_pts, N);
    });

    cudaFree(d_keys); cudaFree(d_pts);
    return result;
}

static BenchResult bench_ct_scalar_mul_impl(const BenchConfig& cfg) {
    int N = std::min(cfg.batch_size, 1 << 13);  // CT k*P is very expensive
    std::vector<Scalar> h_keys(N);
    std::vector<JacobianPoint> h_pts(N);
    std::mt19937_64 rng(71);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < 4; ++j) h_keys[i].limbs[j] = rng();
        h_keys[i].limbs[3] &= 0x7FFFFFFFFFFFFFFFULL;
        if (h_keys[i].limbs[0] == 0 && h_keys[i].limbs[1] == 0 &&
            h_keys[i].limbs[2] == 0 && h_keys[i].limbs[3] == 0)
            h_keys[i].limbs[0] = 1;
        // Use G as the base point for all (hardcoded affine coordinates)
        h_pts[i].x.limbs[0] = 0x59F2815B16F81798ULL;
        h_pts[i].x.limbs[1] = 0x029BFCDB2DCE28D9ULL;
        h_pts[i].x.limbs[2] = 0x55A06295CE870B07ULL;
        h_pts[i].x.limbs[3] = 0x79BE667EF9DCBBACULL;
        h_pts[i].y.limbs[0] = 0x9C47D08FFB10D4B8ULL;
        h_pts[i].y.limbs[1] = 0xFD17B448A6855419ULL;
        h_pts[i].y.limbs[2] = 0x5DA4FBFC0E1108A8ULL;
        h_pts[i].y.limbs[3] = 0x483ADA7726A3C465ULL;
        h_pts[i].z.limbs[0] = 1;
        h_pts[i].z.limbs[1] = 0;
        h_pts[i].z.limbs[2] = 0;
        h_pts[i].z.limbs[3] = 0;
        h_pts[i].infinity = false;
    }

    Scalar* d_keys;
    JacobianPoint *d_pts, *d_out;
    CUDA_CHECK(cudaMalloc(&d_keys, N * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_pts, N * sizeof(JacobianPoint)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(JacobianPoint)));
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), N * sizeof(Scalar), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pts, h_pts.data(), N * sizeof(JacobianPoint), cudaMemcpyHostToDevice));

    int blocks = (N + cfg.threads_per_block - 1) / cfg.threads_per_block;
    BenchConfig ct_cfg = cfg;
    ct_cfg.batch_size = N;
    auto result = run_bench("CT Point Ops", "ct::k*P (scalar mul)", ct_cfg, [&]() {
        bench_ct_scalar_mul_k<<<blocks, cfg.threads_per_block>>>(d_pts, d_keys, d_out, N);
    });

    cudaFree(d_keys); cudaFree(d_pts); cudaFree(d_out);
    return result;
}

#if !SECP256K1_CUDA_LIMBS_32
static BenchResult bench_ct_ecdsa_sign_impl(const BenchConfig& cfg) {
    int N = std::min(cfg.batch_size, 1 << 13);  // CT signing is expensive
    std::vector<Scalar> h_privkeys(N);
    std::vector<uint8_t> h_msgs(N * 32);
    std::mt19937_64 rng(72);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < 4; ++j) h_privkeys[i].limbs[j] = rng();
        h_privkeys[i].limbs[3] &= 0x7FFFFFFFFFFFFFFFULL;
        if (h_privkeys[i].limbs[0] == 0 && h_privkeys[i].limbs[1] == 0 &&
            h_privkeys[i].limbs[2] == 0 && h_privkeys[i].limbs[3] == 0)
            h_privkeys[i].limbs[0] = 1;
    }
    for (int i = 0; i < N * 32; ++i)
        h_msgs[i] = static_cast<uint8_t>(rng() & 0xFF);

    Scalar* d_priv;
    uint8_t* d_msg;
    ECDSASignatureGPU* d_sig;
    bool* d_ok;
    CUDA_CHECK(cudaMalloc(&d_priv, N * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_msg, N * 32));
    CUDA_CHECK(cudaMalloc(&d_sig, N * sizeof(ECDSASignatureGPU)));
    CUDA_CHECK(cudaMalloc(&d_ok, N * sizeof(bool)));
    CUDA_CHECK(cudaMemcpy(d_priv, h_privkeys.data(), N * sizeof(Scalar), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_msg, h_msgs.data(), N * 32, cudaMemcpyHostToDevice));

    int blocks = (N + cfg.threads_per_block - 1) / cfg.threads_per_block;
    BenchConfig sign_cfg = cfg;
    sign_cfg.batch_size = N;
    auto result = run_bench("CT ECDSA", "ct::ecdsa_sign", sign_cfg, [&]() {
        bench_ct_ecdsa_sign_k<<<blocks, cfg.threads_per_block>>>(d_msg, d_priv, d_sig, d_ok, N);
    });

    cudaFree(d_priv); cudaFree(d_msg); cudaFree(d_sig); cudaFree(d_ok);
    return result;
}

static BenchResult bench_ct_schnorr_sign_impl(const BenchConfig& cfg) {
    int N = std::min(cfg.batch_size, 1 << 13);
    std::vector<Scalar> h_privkeys(N);
    std::vector<uint8_t> h_msgs(N * 32), h_aux(N * 32);
    std::mt19937_64 rng(73);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < 4; ++j) h_privkeys[i].limbs[j] = rng();
        h_privkeys[i].limbs[3] &= 0x7FFFFFFFFFFFFFFFULL;
        if (h_privkeys[i].limbs[0] == 0 && h_privkeys[i].limbs[1] == 0 &&
            h_privkeys[i].limbs[2] == 0 && h_privkeys[i].limbs[3] == 0)
            h_privkeys[i].limbs[0] = 1;
    }
    for (int i = 0; i < N * 32; ++i) {
        h_msgs[i] = static_cast<uint8_t>(rng() & 0xFF);
        h_aux[i] = static_cast<uint8_t>(rng() & 0xFF);
    }

    Scalar* d_priv;
    uint8_t *d_msg, *d_aux;
    SchnorrSignatureGPU* d_sig;
    bool* d_ok;
    CUDA_CHECK(cudaMalloc(&d_priv, N * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_msg, N * 32));
    CUDA_CHECK(cudaMalloc(&d_aux, N * 32));
    CUDA_CHECK(cudaMalloc(&d_sig, N * sizeof(SchnorrSignatureGPU)));
    CUDA_CHECK(cudaMalloc(&d_ok, N * sizeof(bool)));
    CUDA_CHECK(cudaMemcpy(d_priv, h_privkeys.data(), N * sizeof(Scalar), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_msg, h_msgs.data(), N * 32, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_aux, h_aux.data(), N * 32, cudaMemcpyHostToDevice));

    int blocks = (N + cfg.threads_per_block - 1) / cfg.threads_per_block;
    BenchConfig sign_cfg = cfg;
    sign_cfg.batch_size = N;
    auto result = run_bench("CT Schnorr", "ct::schnorr_sign", sign_cfg, [&]() {
        bench_ct_schnorr_sign_k<<<blocks, cfg.threads_per_block>>>(d_priv, d_msg, d_aux, d_sig, d_ok, N);
    });

    cudaFree(d_priv); cudaFree(d_msg); cudaFree(d_aux); cudaFree(d_sig); cudaFree(d_ok);
    return result;
}
#endif

// ============================================================================
// GPU device info
// ============================================================================
static void fill_gpu_info(BenchReport& rpt, int device_id) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    int clockKhz = 0;
    cudaDeviceGetAttribute(&clockKhz, cudaDevAttrClockRate, device_id);
    std::snprintf(rpt.gpu_name, sizeof(rpt.gpu_name), "%s", prop.name);
    std::snprintf(rpt.compute_cap, sizeof(rpt.compute_cap), "%d.%d", prop.major, prop.minor);
    rpt.sm_count = prop.multiProcessorCount;
    rpt.clock_mhz = clockKhz / 1000;
    rpt.memory_mb = prop.totalGlobalMem / (1024 * 1024);

    int driver_ver = 0;
    cudaDriverGetVersion(&driver_ver);
    std::snprintf(rpt.driver, sizeof(rpt.driver), "%d.%d", driver_ver / 1000, (driver_ver % 100) / 10);

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    std::snprintf(rpt.backend, sizeof(rpt.backend), "ROCm/HIP");
#else
    std::snprintf(rpt.backend, sizeof(rpt.backend), "CUDA");
#endif

#if defined(__clang__)
    std::snprintf(rpt.compiler, sizeof(rpt.compiler), "Clang %d.%d", __clang_major__, __clang_minor__);
#elif defined(__GNUC__)
    std::snprintf(rpt.compiler, sizeof(rpt.compiler), "GCC %d.%d", __GNUC__, __GNUC_MINOR__);
#elif defined(__NVCC__)
    std::snprintf(rpt.compiler, sizeof(rpt.compiler), "NVCC %d.%d", __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__);
#else
    std::snprintf(rpt.compiler, sizeof(rpt.compiler), "Unknown");
#endif

#if defined(__x86_64__) || defined(_M_X64)
    std::snprintf(rpt.arch, sizeof(rpt.arch), "x86-64");
#elif defined(__aarch64__)
    std::snprintf(rpt.arch, sizeof(rpt.arch), "ARM64");
#else
    std::snprintf(rpt.arch, sizeof(rpt.arch), "Unknown");
#endif
}

// ============================================================================
// JSON report writer
// ============================================================================
static void write_json_report(const char* path, const BenchReport& rpt) {
#ifdef _WIN32
    FILE* f = std::fopen(path, "w");
#else
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    FILE* f = (fd >= 0) ? fdopen(fd, "w") : nullptr;
#endif
    if (!f) {
        std::fprintf(stderr, "WARNING: Cannot open %s for writing\n", path);
        return;
    }

    std::fprintf(f, "{\n");
    std::fprintf(f, "  \"report_type\": \"gpu_benchmark\",\n");
    std::fprintf(f, "  \"library\": \"UltrafastSecp256k1\",\n");
    std::fprintf(f, "  \"library_version\": \"%s\",\n", SECP256K1_VERSION_STRING);
    std::fprintf(f, "  \"git_hash\": \"%s\",\n", GIT_HASH);

    // Timestamp
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    char timebuf[64];
    struct tm tm_buf{};
#ifdef _WIN32
    (void)localtime_s(&tm_buf, &t);
#else
    (void)localtime_r(&t, &tm_buf);
#endif
    (void)std::strftime(timebuf, sizeof(timebuf), "%Y-%m-%dT%H:%M:%S", &tm_buf);
    std::fprintf(f, "  \"timestamp\": \"%s\",\n", timebuf);

    // GPU metadata
    std::fprintf(f, "  \"metadata\": {\n");
    std::fprintf(f, "    \"gpu\": \"%s\",\n", rpt.gpu_name);
    std::fprintf(f, "    \"compute_capability\": \"%s\",\n", rpt.compute_cap);
    std::fprintf(f, "    \"sm_count\": %d,\n", rpt.sm_count);
    std::fprintf(f, "    \"clock_mhz\": %d,\n", rpt.clock_mhz);
    std::fprintf(f, "    \"memory_mb\": %zu,\n", rpt.memory_mb);
    std::fprintf(f, "    \"backend\": \"%s\",\n", rpt.backend);
    std::fprintf(f, "    \"driver\": \"%s\",\n", rpt.driver);
    std::fprintf(f, "    \"compiler\": \"%s\",\n", rpt.compiler);
    std::fprintf(f, "    \"arch\": \"%s\",\n", rpt.arch);
    std::fprintf(f, "    \"passes\": %d,\n", rpt.passes);
    std::fprintf(f, "    \"warmup\": %d,\n", rpt.warmup);
    std::fprintf(f, "    \"batch_size\": %d\n", rpt.batch_size);
    std::fprintf(f, "  },\n");

    // Results
    std::fprintf(f, "  \"results\": [\n");
    for (int i = 0; i < rpt.count; ++i) {
        auto& e = rpt.entries[i];
        std::fprintf(f, "    { \"section\": \"%s\", \"name\": \"%s\", "
                     "\"ns_per_op\": %.2f, \"throughput_mops\": %.2f, \"batch_size\": %d }%s\n",
                     e.section, e.name,
                     e.ns_per_op, e.throughput_mops, e.batch_size,
                     (i + 1 < rpt.count) ? "," : "");
    }
    std::fprintf(f, "  ]\n");
    std::fprintf(f, "}\n");
    std::fclose(f);
}

// ============================================================================
// Console output
// ============================================================================
static void print_result(const BenchResult& r) {
    std::printf("  %-28s | ", r.name);
    if (r.ns_per_op >= 1e6) {
        std::printf("%8.2f ms | ", r.ns_per_op / 1e6);
    } else if (r.ns_per_op >= 1e3) {
        std::printf("%8.2f us | ", r.ns_per_op / 1e3);
    } else {
        std::printf("%8.1f ns | ", r.ns_per_op);
    }
    std::printf("%10.2f M/s | batch=%d\n", r.throughput_mops, r.batch_size);
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
#ifdef _WIN32
    (void)std::setvbuf(stdout, nullptr, _IONBF, 0);
#else
    (void)std::setvbuf(stdout, nullptr, _IOLBF, 0);
#endif

    BenchConfig cfg;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
            cfg.json_path = argv[++i];
        } else if (std::strcmp(argv[i], "--suite") == 0 && i + 1 < argc) {
            cfg.suite = argv[++i];
        } else if (std::strcmp(argv[i], "--passes") == 0 && i + 1 < argc) {
            cfg.measure_passes = std::atoi(argv[++i]);
            if (cfg.measure_passes < 3) cfg.measure_passes = 3;
        } else if (std::strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            cfg.batch_size = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            cfg.threads_per_block = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            cfg.device_id = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--quick") == 0) {
            cfg.quick = true;
            cfg.warmup_iterations = 1;
            cfg.measure_passes = 3;
            cfg.batch_size = 1 << 16;
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            std::printf("Usage: gpu_bench_unified [OPTIONS]\n\n");
            std::printf("  --json <file>    JSON report output\n");
            std::printf("  --suite <name>   core | all (default: all)\n");
            std::printf("  --passes <N>     Measurement passes (default: 5, min 3)\n");
            std::printf("  --batch <N>      Batch size (default: 1048576)\n");
            std::printf("  --threads <N>    Threads per block (default: 256)\n");
            std::printf("  --device <id>    GPU device index (default: 0)\n");
            std::printf("  --quick          CI mode (small batches, 3 passes)\n");
            return 0;
        }
    }

    // Set device
    CUDA_CHECK(cudaSetDevice(cfg.device_id));

    // Print header
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.device_id));
    int clockKhz = 0;
    cudaDeviceGetAttribute(&clockKhz, cudaDevAttrClockRate, cfg.device_id);

    std::printf("================================================================\n");
    std::printf("  UltrafastSecp256k1 -- GPU Unified Benchmark\n");
    std::printf("================================================================\n\n");
    std::printf("  GPU:          %s\n", prop.name);
    std::printf("  Compute:      %d.%d\n", prop.major, prop.minor);
    std::printf("  SM Count:     %d\n", prop.multiProcessorCount);
    std::printf("  Clock:        %d MHz\n", clockKhz / 1000);
    std::printf("  Memory:       %zu MB\n", prop.totalGlobalMem / (1024*1024));
    std::printf("  Memory Bus:   %d bit\n", prop.memoryBusWidth);

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    std::printf("  Backend:      ROCm/HIP\n");
#else
    std::printf("  Backend:      CUDA\n");
#endif

    std::printf("\n  Batch Size:   %d\n", cfg.batch_size);
    std::printf("  Threads/Blk:  %d\n", cfg.threads_per_block);
    std::printf("  Warmup:       %d\n", cfg.warmup_iterations);
    std::printf("  Passes:       %d\n", cfg.measure_passes);
    std::printf("  Suite:        %s\n", cfg.suite.c_str());
    std::printf("\n");

    BenchReport rpt{};
    rpt.count = 0;
    fill_gpu_info(rpt, cfg.device_id);
    rpt.passes = cfg.measure_passes;
    rpt.warmup = cfg.warmup_iterations;
    rpt.batch_size = cfg.batch_size;

    std::vector<BenchResult> results;

    // =================================================================
    // Section 1: Field Arithmetic
    // =================================================================
    std::printf("=== Field Arithmetic ===\n");
    {
        auto r = bench_field_mul_impl(cfg); print_result(r); results.push_back(r);
        rpt.add(r.section, r.name, r.ns_per_op, r.throughput_mops, r.batch_size);
    }
    {
        auto r = bench_field_sqr_impl(cfg); print_result(r); results.push_back(r);
        rpt.add(r.section, r.name, r.ns_per_op, r.throughput_mops, r.batch_size);
    }
    {
        auto r = bench_field_add_impl(cfg); print_result(r); results.push_back(r);
        rpt.add(r.section, r.name, r.ns_per_op, r.throughput_mops, r.batch_size);
    }
    {
        auto r = bench_field_sub_impl(cfg); print_result(r); results.push_back(r);
        rpt.add(r.section, r.name, r.ns_per_op, r.throughput_mops, r.batch_size);
    }
    {
        auto r = bench_field_inv_impl(cfg); print_result(r); results.push_back(r);
        rpt.add(r.section, r.name, r.ns_per_op, r.throughput_mops, r.batch_size);
    }

    // =================================================================
    // Section 2: Point Operations
    // =================================================================
    std::printf("\n=== Point Operations ===\n");
    {
        auto r = bench_point_add_impl(cfg); print_result(r); results.push_back(r);
        rpt.add(r.section, r.name, r.ns_per_op, r.throughput_mops, r.batch_size);
    }
    {
        auto r = bench_point_dbl_impl(cfg); print_result(r); results.push_back(r);
        rpt.add(r.section, r.name, r.ns_per_op, r.throughput_mops, r.batch_size);
    }
    {
        auto r = bench_jac_to_affine_impl(cfg); print_result(r); results.push_back(r);
        rpt.add(r.section, r.name, r.ns_per_op, r.throughput_mops, r.batch_size);
    }

    // =================================================================
    // Section 3: Scalar Multiplication
    // =================================================================
    std::printf("\n=== Scalar Multiplication ===\n");
    {
        auto r = bench_scalar_mul_gen_impl(cfg); print_result(r); results.push_back(r);
        rpt.add(r.section, r.name, r.ns_per_op, r.throughput_mops, r.batch_size);
    }

    // =================================================================
    // Section 4: Affine Pipeline
    // =================================================================
    std::printf("\n=== Affine Pipeline ===\n");
    {
        auto r = bench_affine_add_impl(cfg); print_result(r); results.push_back(r);
        rpt.add(r.section, r.name, r.ns_per_op, r.throughput_mops, r.batch_size);
    }

    // =================================================================
    // Section 5/6: Signatures (ECDSA + Schnorr)
    // =================================================================
#if !SECP256K1_CUDA_LIMBS_32
    if (cfg.suite == "all") {
        std::printf("\n=== ECDSA ===\n");
        {
            auto r = bench_ecdsa_sign_impl(cfg); print_result(r); results.push_back(r);
            rpt.add(r.section, r.name, r.ns_per_op, r.throughput_mops, r.batch_size);
        }

        std::printf("\n=== Schnorr/BIP-340 ===\n");
        {
            auto r = bench_schnorr_sign_impl(cfg); print_result(r); results.push_back(r);
            rpt.add(r.section, r.name, r.ns_per_op, r.throughput_mops, r.batch_size);
        }
    }
#endif

    // =================================================================
    // Section 7: CT (Constant-Time) Operations
    // =================================================================
    std::printf("\n=== CT (Constant-Time) Point Ops ===\n");
    {
        auto r = bench_ct_generator_mul_impl(cfg); print_result(r); results.push_back(r);
        rpt.add(r.section, r.name, r.ns_per_op, r.throughput_mops, r.batch_size);
    }
    {
        auto r = bench_ct_scalar_mul_impl(cfg); print_result(r); results.push_back(r);
        rpt.add(r.section, r.name, r.ns_per_op, r.throughput_mops, r.batch_size);
    }

#if !SECP256K1_CUDA_LIMBS_32
    if (cfg.suite == "all") {
        std::printf("\n=== CT ECDSA ===\n");
        {
            auto r = bench_ct_ecdsa_sign_impl(cfg); print_result(r); results.push_back(r);
            rpt.add(r.section, r.name, r.ns_per_op, r.throughput_mops, r.batch_size);
        }

        std::printf("\n=== CT Schnorr/BIP-340 ===\n");
        {
            auto r = bench_ct_schnorr_sign_impl(cfg); print_result(r); results.push_back(r);
            rpt.add(r.section, r.name, r.ns_per_op, r.throughput_mops, r.batch_size);
        }
    }
#endif

    // =================================================================
    // Summary table
    // =================================================================
    std::printf("\n================================================================\n");
    std::printf("  GPU Performance Summary\n");
    std::printf("================================================================\n");
    std::printf("  GPU: %s (SM %d.%d, %d SMs)\n\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount);

    std::printf("  %-28s | %-10s | %-12s | %s\n", "Operation", "Time/Op", "Throughput", "Batch");
    std::printf("  %-28s-+-%-10s-+-%-12s-+-%s\n", "----------------------------", "----------",
                "------------", "--------");
    for (auto& r : results) {
        std::printf("  %-28s | ", r.name);
        if (r.ns_per_op >= 1e6) std::printf("%8.2f ms", r.ns_per_op / 1e6);
        else if (r.ns_per_op >= 1e3) std::printf("%8.2f us", r.ns_per_op / 1e3);
        else std::printf("%8.1f ns", r.ns_per_op);
        std::printf(" | %10.2f M/s | %d\n", r.throughput_mops, r.batch_size);
    }
    std::printf("================================================================\n\n");

    // JSON output
    if (!cfg.json_path.empty()) {
        write_json_report(cfg.json_path.c_str(), rpt);
        std::printf("JSON report: %s\n", cfg.json_path.c_str());
    }

    std::printf("Benchmark complete.\n");
    return 0;
}
