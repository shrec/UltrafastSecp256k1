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
//   4b. Verification        (ECDSA verify, BIP-340 verify, ecrecover, hash160;
//                            CPU-made inputs, every result checked, core suite)
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
#include "msm.cuh"

// ---- CPU library (host-side input generation and result reference) ----
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/recovery.hpp"
#include "secp256k1/address.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/ct/sign.hpp"

#include <cuda_runtime.h>
#include <array>
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
    bool valid;   // false: written as INVALID with no ns/throughput value
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
             double ns_val, double mops, int bsz, bool valid = true) {
        if (count >= MAX_BENCH_ENTRIES) return;
        auto& e = entries[count++];
        std::snprintf(e.section, sizeof(e.section), "%s", section);
        std::snprintf(e.name, sizeof(e.name), "%s", name);
        e.ns_per_op = ns_val;
        e.throughput_mops = mops;
        e.batch_size = bsz;
        e.valid = valid;
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
    bool valid = true;   // false: result check failed, no timing is reported
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

__global__ void bench_field_inv_var_k(const FieldElement* a, FieldElement* r, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) field_inv_var(&a[idx], &r[idx]);
}

__global__ void bench_scalar_inv_k(const Scalar* a, Scalar* r, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) scalar_inverse(&a[idx], &r[idx]);
}

__global__ void bench_scalar_inv_var_k(const Scalar* a, Scalar* r, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) scalar_inv_var(&a[idx], &r[idx]);
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

// Inverse entries added with modinv_var.cuh ("inv (var)", "scalar inv",
// "scalar inv (var)"). Same batch size, input generator (seed 49, non-zero,
// top bit clear so every value is below p and n) and launch shape as
// bench_field_inv_impl above, so the rows compare directly with "inv".
template <typename Elem>
static BenchResult bench_inverse_impl(const char* section, const char* name, const BenchConfig& cfg,
                                      void (*kernel)(const Elem*, Elem*, int)) {
    static_assert(sizeof(Elem) == sizeof(FieldElement), "element must be 256 bits");
    int N = std::min(cfg.batch_size, 1 << 16);
    std::vector<FieldElement> h_a(N);
    gen_field_elements(h_a.data(), N, 49);
    for (int i = 0; i < N; ++i) {
        if (h_a[i].limbs[0] == 0 && h_a[i].limbs[1] == 0 &&
            h_a[i].limbs[2] == 0 && h_a[i].limbs[3] == 0)
            h_a[i].limbs[0] = 1;
    }

    Elem *d_a, *d_r;
    size_t sz = N * sizeof(Elem);
    CUDA_CHECK(cudaMalloc(&d_a, sz));
    CUDA_CHECK(cudaMalloc(&d_r, sz));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), sz, cudaMemcpyHostToDevice));

    int blocks = (N + cfg.threads_per_block - 1) / cfg.threads_per_block;
    BenchConfig inv_cfg = cfg;
    inv_cfg.batch_size = N;
    auto result = run_bench(section, name, inv_cfg, [&]() {
        kernel<<<blocks, cfg.threads_per_block>>>(d_a, d_r, N);
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

    // Erase private keys from GPU memory before free (Guardrail #10)
    cudaMemset(d_priv, 0, N * sizeof(Scalar));
    cudaDeviceSynchronize();
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

    // Erase private keys from GPU memory before free (Guardrail #10)
    cudaMemset(d_priv, 0, N * sizeof(Scalar));
    cudaDeviceSynchronize();
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

// ============================================================================
// MSM -- multi-scalar multiplication, whole pipeline
// ============================================================================
// Times what the CUDA backend actually does for sum(s_i * P_i): a scatter pass
// (one full scalar_mul_glv per point) followed by a tree reduction.
//
// The reduction is replicated here rather than reused: the backend's copies are
// file-local __global__ functions in src/gpu/src/gpu_backend_cuda.cu and are not
// exported from any device header. Both the current scatter path and any future
// bucket path must be timed through this same reduction for the comparison to
// mean anything, so it lives with the benchmark.
#if !SECP256K1_CUDA_LIMBS_32

__global__ void bench_msm_block_reduce_k(
    const JacobianPoint* __restrict__ partials, int n, JacobianPoint* block_results)
{
    extern __shared__ JacobianPoint s_msm[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < n) {
        s_msm[tid] = partials[idx];
    } else {
        s_msm[tid].infinity = true;
        field_set_zero(&s_msm[tid].x);
        field_set_zero(&s_msm[tid].y);
        field_set_one(&s_msm[tid].z);
    }
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_msm[tid].infinity) {
                s_msm[tid] = s_msm[tid + stride];
            } else if (!s_msm[tid + stride].infinity) {
                JacobianPoint tmp;
                jacobian_add(&s_msm[tid], &s_msm[tid + stride], &tmp);
                s_msm[tid] = tmp;
            }
        }
        __syncthreads();
    }

    if (tid == 0) block_results[blockIdx.x] = s_msm[0];
}

__global__ void bench_msm_final_reduce_k(
    const JacobianPoint* partials, int n, JacobianPoint* out)
{
    JacobianPoint acc;
    acc.infinity = true;
    field_set_zero(&acc.x);
    field_set_zero(&acc.y);
    field_set_one(&acc.z);

    for (int i = 0; i < n; ++i) {
        if (partials[i].infinity) continue;
        if (acc.infinity) {
            acc = partials[i];
        } else {
            JacobianPoint tmp;
            jacobian_add(&acc, &partials[i], &tmp);
            acc = tmp;
        }
    }
    *out = acc;
}

// stages: 1 = scatter only, 2 = scatter + block reduce, 3 = full pipeline.
// Splitting them apart is what says whether the single-threaded final reduce is
// worth attacking -- GPU-PIPPENGER-NEG already ruled out replacing the scatter
// with a bucket scheme, but it says nothing about the reduction tail.
static BenchResult bench_msm_impl(const BenchConfig& cfg, int stages, const char* label) {
    // ~1GB ceiling: each point costs sizeof(JacobianPoint) twice (input +
    // partial) plus a scalar.
    int N = std::min(cfg.batch_size, 1 << 22);
    constexpr int kBlk = 256;
    int const n_blks = (N + kBlk - 1) / kBlk;

    std::vector<Scalar> h_scalars(N);
    std::vector<JacobianPoint> h_pts(N);
    gen_scalars(h_scalars.data(), N, 0x4D534DULL);

    // All points are G. Bucket occupancy depends on the scalar digits, not on
    // which point sits in the bucket, so this costs the algorithm nothing and
    // matches what the other point benchmarks in this file do.
    for (int i = 0; i < N; ++i) {
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

    Scalar* d_scalars = nullptr;
    JacobianPoint *d_pts = nullptr, *d_partials = nullptr;
    JacobianPoint *d_blk = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_scalars,  N * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_pts,      N * sizeof(JacobianPoint)));
    CUDA_CHECK(cudaMalloc(&d_partials, N * sizeof(JacobianPoint)));
    CUDA_CHECK(cudaMalloc(&d_blk,      n_blks * sizeof(JacobianPoint)));
    CUDA_CHECK(cudaMalloc(&d_out,      sizeof(JacobianPoint)));
    CUDA_CHECK(cudaMemcpy(d_scalars, h_scalars.data(), N * sizeof(Scalar),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pts, h_pts.data(), N * sizeof(JacobianPoint),
                          cudaMemcpyHostToDevice));

    int const threads = cfg.threads_per_block;
    int const blocks  = (N + threads - 1) / threads;
    size_t const smem = kBlk * sizeof(JacobianPoint);

    BenchConfig msm_cfg = cfg;
    msm_cfg.batch_size = N;
    auto result = run_bench("MSM", label, msm_cfg, [&]() {
        msm_scatter_kernel<<<blocks, threads>>>(d_scalars, d_pts, d_partials, N);
        if (stages >= 2)
            bench_msm_block_reduce_k<<<n_blks, kBlk, smem>>>(d_partials, N, d_blk);
        if (stages >= 3) {
            // Mirrors the backend's tail reduce (gpu_backend_cuda.cu msm()) so
            // the before/after stays apples-to-apples: same shape, same
            // kFinisherMax, same ping-pong buffers.
            constexpr int kFinisherMax = 32;
            int count = n_blks;
            JacobianPoint* in  = d_blk;
            JacobianPoint* out = d_partials;
            while (count > kFinisherMax) {
                int const next = (count + kBlk - 1) / kBlk;
                bench_msm_block_reduce_k<<<next, kBlk, smem>>>(in, count, out);
                count = next;
                JacobianPoint* const swap = in;
                in  = out;
                out = swap;
            }
            bench_msm_final_reduce_k<<<1, 1>>>(in, count, d_out);
        }
    });

    cudaFree(d_scalars); cudaFree(d_pts); cudaFree(d_partials);
    cudaFree(d_blk); cudaFree(d_out);
    return result;
}

#endif  // !SECP256K1_CUDA_LIMBS_32

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

    // Erase private keys from GPU memory before free (Guardrail #10)
    cudaMemset(d_priv, 0, N * sizeof(Scalar));
    cudaDeviceSynchronize();
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

    // Erase private keys from GPU memory before free (Guardrail #10)
    cudaMemset(d_priv, 0, N * sizeof(Scalar));
    cudaDeviceSynchronize();
    cudaFree(d_priv); cudaFree(d_msg); cudaFree(d_aux); cudaFree(d_sig); cudaFree(d_ok);
    return result;
}
#endif

// ============================================================================
// Verification rows -- ECDSA verify, BIP-340 verify, ecrecover, hash160
// ============================================================================
// Kernel-only timing of the kernels the C ABI backend (src/gpu/src/
// gpu_backend_cuda.cu) launches, with the backend's launch shape. Host<->device
// copies are outside the timed region; the on-device conversion the backend
// runs on every call is inside it:
//   ECDSA verify : compressed-pubkey decompress + ecdsa_verify_batch_kernel
//   BIP-340      : schnorr_verify_batch_kernel (the ABI parses sigs on the host)
//   ecrecover    : ecdsa_recover_batch_kernel + Jacobian -> 33-byte compress
//   hash160      : hash160_pubkey_kernel
// Inputs are valid data made once on the host by the CPU library (CT signing,
// low-S ECDSA). After the timed passes every result is read back and checked
// against the CPU data; any mismatch makes the row INVALID (no timing value).

static_assert(sizeof(bool) == 1, "device bool results are copied as bytes");

#if !SECP256K1_CUDA_LIMBS_32
namespace secp256k1 {
namespace cuda {
// Defined in src/cuda/src/secp256k1.cu (secp256k1_cuda_lib). The declarations
// repeat the definitions' __restrict__ qualifiers exactly, as the backend does
// (MSVC mangles __restrict into the symbol name).
extern __global__ void ecdsa_verify_batch_kernel(
    const uint8_t* __restrict__ msg_hashes,
    const JacobianPoint* __restrict__ public_keys,
    const uint8_t* __restrict__ sigs64,
    bool*          __restrict__ results,
    int count);
extern __global__ void schnorr_verify_batch_kernel(
    const uint8_t* __restrict__ pubkeys_x,
    const uint8_t* __restrict__ msgs,
    const SchnorrSignatureGPU* __restrict__ sigs,
    bool*          __restrict__ results,
    int count);
#if SECP256K1_GPU_HAS_ECRECOVER
extern __global__ void ecdsa_recover_batch_kernel(
    const uint8_t* __restrict__ msg_hashes,
    const ECDSASignatureGPU* __restrict__ sigs,
    const int*     __restrict__ recids,
    JacobianPoint* __restrict__ recovered_keys,
    bool*          __restrict__ results,
    int count);
#endif
} // namespace cuda
} // namespace secp256k1

// The backend launches its verify / recover kernels with 128 threads per block
// (the kernels are declared __launch_bounds__(128, 2)).
static constexpr int kVerifyThreads = 128;

// Same bodies as batch_compressed_to_jac_kernel / batch_jac_to_compressed_kernel
// in gpu_backend_cuda.cu, which are file-local there.
__global__ void bench_compressed_to_jac_k(const uint8_t* pubs33, JacobianPoint* out,
                                          bool* ok, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    ok[idx] = point_from_compressed(pubs33 + idx * 33, &out[idx]);
}

__global__ void bench_jac_to_compressed_k(const JacobianPoint* pts, uint8_t* out33, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    if (!point_to_compressed(&pts[idx], out33 + idx * 33)) {
        memset(out33 + idx * 33, 0, 33);
    }
}

// Big-endian 32 bytes -> GPU scalar limbs, as the backend's bytes_to_scalar.
static void be32_to_gpu_scalar(const uint8_t be[32], secp256k1::cuda::Scalar* out) {
    for (int limb = 0; limb < 4; ++limb) {
        uint64_t v = 0;
        for (int b = 0; b < 8; ++b) v = (v << 8) | be[(3 - limb) * 8 + b];
        out->limbs[limb] = v;
    }
}
#endif // !SECP256K1_CUDA_LIMBS_32

// Host-side input set shared by every verification row.
struct VerifyInputs {
    int n = 0;
    bool ok = false;
    std::vector<uint8_t> msgs;           // n x 32 message hashes
    std::vector<uint8_t> pub33;          // n x 33 SEC1 compressed pubkeys
    std::vector<uint8_t> pub65;          // n x 65 SEC1 uncompressed pubkeys
    std::vector<uint8_t> ecdsa_sig64;    // n x 64 compact r||s, low-S
    std::vector<uint8_t> xonly32;        // n x 32 BIP-340 x-only pubkeys
    std::vector<uint8_t> schnorr_sig64;  // n x 64 BIP-340 r||s
    std::vector<uint8_t> rec_sig64;      // n x 64 recoverable compact r||s
    std::vector<int>     recids;         // n recovery ids
    std::vector<uint8_t> h160_33;        // n x 20 CPU hash160(pub33)
    std::vector<uint8_t> h160_65;        // n x 20 CPU hash160(pub65)
};

// Builds n valid inputs with the CPU library (CT signing paths). Sequential:
// it runs once per process, outside every timed region, and the CPU CT tables
// initialise lazily on first use.
static bool build_verify_inputs(int n, VerifyInputs& in) {
    in = VerifyInputs{};
    if (n <= 0) return false;
    const size_t un = static_cast<size_t>(n);
    in.n = n;
    in.msgs.resize(un * 32);
    in.pub33.resize(un * 33);
    in.pub65.resize(un * 65);
    in.ecdsa_sig64.resize(un * 64);
    in.xonly32.resize(un * 32);
    in.schnorr_sig64.resize(un * 64);
    in.rec_sig64.resize(un * 64);
    in.recids.resize(un);
    in.h160_33.resize(un * 20);
    in.h160_65.resize(un * 20);

    std::mt19937_64 rng(80);
    auto fill32 = [&](std::array<uint8_t, 32>& b) {
        for (int j = 0; j < 4; ++j) {
            const uint64_t v = rng();
            for (int k = 0; k < 8; ++k) b[j * 8 + k] = static_cast<uint8_t>(v >> (8 * k));
        }
    };

    for (size_t i = 0; i < un; ++i) {
        std::array<uint8_t, 32> skb{}, msg{}, aux{};
        secp256k1::fast::Scalar sk;
        do { fill32(skb); } while (!secp256k1::fast::Scalar::parse_bytes_strict_nonzero(skb, sk));
        fill32(msg);
        fill32(aux);

        const auto P   = secp256k1::ct::generator_mul(sk);
        const auto c33 = P.to_compressed();
        const auto u65 = P.to_uncompressed();

        const auto esig = secp256k1::ct::ecdsa_sign(msg, sk);
        if (!esig.is_valid() || !esig.is_low_s()) return false;
        const auto e64 = esig.to_compact();

        const auto kp   = secp256k1::ct::schnorr_keypair_create(sk);
        const auto ssig = secp256k1::ct::schnorr_sign(kp, msg, aux);
        const auto s64  = ssig.to_bytes();

        const auto rsig = secp256k1::ct::ecdsa_sign_recoverable(msg, sk);
        if (!rsig.sig.is_valid() || rsig.recid < 0 || rsig.recid > 3) return false;
        const auto r64 = rsig.sig.to_compact();

        const auto h33 = secp256k1::hash160(c33.data(), c33.size());
        const auto h65 = secp256k1::hash160(u65.data(), u65.size());

        std::memcpy(&in.msgs[i * 32], msg.data(), 32);
        std::memcpy(&in.pub33[i * 33], c33.data(), 33);
        std::memcpy(&in.pub65[i * 65], u65.data(), 65);
        std::memcpy(&in.ecdsa_sig64[i * 64], e64.data(), 64);
        std::memcpy(&in.xonly32[i * 32], kp.px.data(), 32);
        std::memcpy(&in.schnorr_sig64[i * 64], s64.data(), 64);
        std::memcpy(&in.rec_sig64[i * 64], r64.data(), 64);
        in.recids[i] = rsig.recid;
        std::memcpy(&in.h160_33[i * 20], h33.data(), 20);
        std::memcpy(&in.h160_65[i * 20], h65.data(), 20);
        skb.fill(0);  // benchmark key; never copied to the GPU
    }
    in.ok = true;
    return true;
}

[[maybe_unused]] static BenchResult invalid_result(const char* section, const char* name, int n) {
    BenchResult r{ section, name, 0.0, n, 0.0, 0.0 };
    r.valid = false;
    return r;
}

// Launch / execution errors make the row INVALID instead of reporting a time.
[[maybe_unused]] static bool device_ok(const char* what) {
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "  %s: CUDA error: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

[[maybe_unused]] static BenchResult finish_checked(BenchResult r, int bad, int first_bad, int n) {
    if (bad > 0) {
        std::fprintf(stderr, "  %s %s: %d of %d results did not match the CPU reference "
                     "(first at index %d) -- row INVALID\n", r.section, r.name, bad, n, first_bad);
        r.valid = false;
    }
    return r;
}

#if !SECP256K1_CUDA_LIMBS_32
static BenchResult bench_ecdsa_verify_impl(const BenchConfig& cfg, const VerifyInputs& in) {
    if (!in.ok) return invalid_result("ECDSA", "verify", in.n);
    const int N = in.n;
    const size_t un = static_cast<size_t>(N);

    uint8_t *d_msg = nullptr, *d_pub33 = nullptr, *d_sig = nullptr;
    JacobianPoint* d_pub = nullptr;
    bool *d_pub_ok = nullptr, *d_res = nullptr;
    CUDA_CHECK(cudaMalloc(&d_msg, un * 32));
    CUDA_CHECK(cudaMalloc(&d_pub33, un * 33));
    CUDA_CHECK(cudaMalloc(&d_sig, un * 64));
    CUDA_CHECK(cudaMalloc(&d_pub, un * sizeof(JacobianPoint)));
    CUDA_CHECK(cudaMalloc(&d_pub_ok, un * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_res, un * sizeof(bool)));
    CUDA_CHECK(cudaMemcpy(d_msg, in.msgs.data(), un * 32, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pub33, in.pub33.data(), un * 33, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sig, in.ecdsa_sig64.data(), un * 64, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_pub_ok, 0, un * sizeof(bool)));
    CUDA_CHECK(cudaMemset(d_res, 0, un * sizeof(bool)));

    const int blocks = (N + kVerifyThreads - 1) / kVerifyThreads;
    BenchConfig v_cfg = cfg;
    v_cfg.batch_size = N;
    auto result = run_bench("ECDSA", "verify", v_cfg, [&]() {
        bench_compressed_to_jac_k<<<blocks, kVerifyThreads>>>(d_pub33, d_pub, d_pub_ok, N);
        ecdsa_verify_batch_kernel<<<blocks, kVerifyThreads>>>(d_msg, d_pub, d_sig, d_res, N);
    });

    std::vector<uint8_t> h_ok(un, 0), h_res(un, 0);
    const bool ran = device_ok("ECDSA verify");
    if (ran) {
        CUDA_CHECK(cudaMemcpy(h_ok.data(), d_pub_ok, un * sizeof(bool), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, un * sizeof(bool), cudaMemcpyDeviceToHost));
    }
    int bad = 0, first_bad = -1;
    for (int i = 0; i < N; ++i) {
        if (!ran || !h_ok[i] || !h_res[i]) {
            if (first_bad < 0) first_bad = i;
            ++bad;
        }
    }

    cudaFree(d_msg); cudaFree(d_pub33); cudaFree(d_sig);
    cudaFree(d_pub); cudaFree(d_pub_ok); cudaFree(d_res);
    return finish_checked(result, bad, first_bad, N);
}

static BenchResult bench_schnorr_verify_impl(const BenchConfig& cfg, const VerifyInputs& in) {
    if (!in.ok) return invalid_result("Schnorr/BIP-340", "verify (x-only)", in.n);
    const int N = in.n;
    const size_t un = static_cast<size_t>(N);

    // Host-side signature conversion, as the backend's bytes_to_schnorr_sig.
    std::vector<SchnorrSignatureGPU> h_sigs(un);
    for (size_t i = 0; i < un; ++i) {
        std::memcpy(h_sigs[i].r, &in.schnorr_sig64[i * 64], 32);
        be32_to_gpu_scalar(&in.schnorr_sig64[i * 64 + 32], &h_sigs[i].s);
    }

    uint8_t *d_pks = nullptr, *d_msg = nullptr;
    SchnorrSignatureGPU* d_sigs = nullptr;
    bool* d_res = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pks, un * 32));
    CUDA_CHECK(cudaMalloc(&d_msg, un * 32));
    CUDA_CHECK(cudaMalloc(&d_sigs, un * sizeof(SchnorrSignatureGPU)));
    CUDA_CHECK(cudaMalloc(&d_res, un * sizeof(bool)));
    CUDA_CHECK(cudaMemcpy(d_pks, in.xonly32.data(), un * 32, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_msg, in.msgs.data(), un * 32, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sigs, h_sigs.data(), un * sizeof(SchnorrSignatureGPU),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_res, 0, un * sizeof(bool)));

    const int blocks = (N + kVerifyThreads - 1) / kVerifyThreads;
    BenchConfig v_cfg = cfg;
    v_cfg.batch_size = N;
    auto result = run_bench("Schnorr/BIP-340", "verify (x-only)", v_cfg, [&]() {
        schnorr_verify_batch_kernel<<<blocks, kVerifyThreads>>>(d_pks, d_msg, d_sigs, d_res, N);
    });

    std::vector<uint8_t> h_res(un, 0);
    const bool ran = device_ok("BIP-340 verify");
    if (ran)
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, un * sizeof(bool), cudaMemcpyDeviceToHost));
    int bad = 0, first_bad = -1;
    for (int i = 0; i < N; ++i) {
        if (!ran || !h_res[i]) {
            if (first_bad < 0) first_bad = i;
            ++bad;
        }
    }

    cudaFree(d_pks); cudaFree(d_msg); cudaFree(d_sigs); cudaFree(d_res);
    return finish_checked(result, bad, first_bad, N);
}

#if SECP256K1_GPU_HAS_ECRECOVER
static BenchResult bench_ecrecover_impl(const BenchConfig& cfg, const VerifyInputs& in) {
    if (!in.ok) return invalid_result("ECDSA", "ecrecover", in.n);
    const int N = in.n;
    const size_t un = static_cast<size_t>(N);

    // Host-side signature conversion, as the backend's bytes_to_ecdsa_sig.
    std::vector<ECDSASignatureGPU> h_sigs(un);
    for (size_t i = 0; i < un; ++i) {
        be32_to_gpu_scalar(&in.rec_sig64[i * 64], &h_sigs[i].r);
        be32_to_gpu_scalar(&in.rec_sig64[i * 64 + 32], &h_sigs[i].s);
    }

    uint8_t *d_msg = nullptr, *d_out33 = nullptr;
    ECDSASignatureGPU* d_sigs = nullptr;
    int* d_recids = nullptr;
    JacobianPoint* d_keys = nullptr;
    bool* d_res = nullptr;
    CUDA_CHECK(cudaMalloc(&d_msg, un * 32));
    CUDA_CHECK(cudaMalloc(&d_sigs, un * sizeof(ECDSASignatureGPU)));
    CUDA_CHECK(cudaMalloc(&d_recids, un * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_keys, un * sizeof(JacobianPoint)));
    CUDA_CHECK(cudaMalloc(&d_res, un * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_out33, un * 33));
    CUDA_CHECK(cudaMemcpy(d_msg, in.msgs.data(), un * 32, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sigs, h_sigs.data(), un * sizeof(ECDSASignatureGPU),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_recids, in.recids.data(), un * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_res, 0, un * sizeof(bool)));
    CUDA_CHECK(cudaMemset(d_out33, 0, un * 33));

    const int blocks = (N + kVerifyThreads - 1) / kVerifyThreads;
    BenchConfig v_cfg = cfg;
    v_cfg.batch_size = N;
    auto result = run_bench("ECDSA", "ecrecover", v_cfg, [&]() {
        ecdsa_recover_batch_kernel<<<blocks, kVerifyThreads>>>(d_msg, d_sigs, d_recids,
                                                               d_keys, d_res, N);
        bench_jac_to_compressed_k<<<blocks, kVerifyThreads>>>(d_keys, d_out33, N);
    });

    std::vector<uint8_t> h_res(un, 0), h_out(un * 33, 0);
    const bool ran = device_ok("ecrecover");
    if (ran) {
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, un * sizeof(bool), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out33, un * 33, cudaMemcpyDeviceToHost));
    }
    int bad = 0, first_bad = -1;
    for (int i = 0; i < N; ++i) {
        const size_t o = static_cast<size_t>(i) * 33;
        if (!ran || !h_res[i] || std::memcmp(&h_out[o], &in.pub33[o], 33) != 0) {
            if (first_bad < 0) first_bad = i;
            ++bad;
        }
    }

    cudaFree(d_msg); cudaFree(d_sigs); cudaFree(d_recids);
    cudaFree(d_keys); cudaFree(d_res); cudaFree(d_out33);
    return finish_checked(result, bad, first_bad, N);
}
#endif  // SECP256K1_GPU_HAS_ECRECOVER
#endif  // !SECP256K1_CUDA_LIMBS_32

#if SECP256K1_GPU_HAS_HASH160
// hash160 is cheap per element, so the CPU-made keys are tiled up to the batch
// size (capped at 4M) and each output is checked against the CPU hash160 of
// the key it was given.
static BenchResult bench_hash160_impl(const BenchConfig& cfg, const VerifyInputs& in,
                                      int pubkey_len, const char* name) {
    const int N = std::min(cfg.batch_size, 1 << 22);
    if (!in.ok || N <= 0 || (pubkey_len != 33 && pubkey_len != 65))
        return invalid_result("Hash160", name, N);
    const size_t un = static_cast<size_t>(N);
    const size_t len = static_cast<size_t>(pubkey_len);
    const size_t K = static_cast<size_t>(in.n);
    const std::vector<uint8_t>& src = (pubkey_len == 33) ? in.pub33 : in.pub65;
    const std::vector<uint8_t>& ref = (pubkey_len == 33) ? in.h160_33 : in.h160_65;

    std::vector<uint8_t> h_in(un * len);
    for (size_t i = 0; i < un; ++i)
        std::memcpy(&h_in[i * len], &src[(i % K) * len], len);

    uint8_t *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, un * len));
    CUDA_CHECK(cudaMalloc(&d_out, un * 20));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), un * len, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, un * 20));

    // The backend launches hash160_pubkey_kernel with 256 threads per block
    // (declared __launch_bounds__(256, 4)).
    constexpr int kThreads = 256;
    const int blocks = (N + kThreads - 1) / kThreads;
    BenchConfig h_cfg = cfg;
    h_cfg.batch_size = N;
    auto result = run_bench("Hash160", name, h_cfg, [&]() {
        hash160_pubkey_kernel<<<blocks, kThreads>>>(d_in, pubkey_len, d_out, N);
    });

    std::vector<uint8_t> h_out(un * 20, 0);
    const bool ran = device_ok(name);
    if (ran)
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, un * 20, cudaMemcpyDeviceToHost));
    int bad = 0, first_bad = -1;
    for (size_t i = 0; i < un; ++i) {
        if (!ran || std::memcmp(&h_out[i * 20], &ref[(i % K) * 20], 20) != 0) {
            if (first_bad < 0) first_bad = static_cast<int>(i);
            ++bad;
        }
    }

    cudaFree(d_in); cudaFree(d_out);
    return finish_checked(result, bad, first_bad, N);
}
#endif  // SECP256K1_GPU_HAS_HASH160

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
        if (!e.valid) {
            // Result check failed: no timing value is written for this row.
            std::fprintf(f, "    { \"section\": \"%s\", \"name\": \"%s\", \"status\": \"INVALID\", "
                         "\"ns_per_op\": null, \"throughput_mops\": null, \"batch_size\": %d }%s\n",
                         e.section, e.name, e.batch_size,
                         (i + 1 < rpt.count) ? "," : "");
            continue;
        }
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
    if (!r.valid) {
        std::printf("%11s | %14s | batch=%d\n", "INVALID", "-", r.batch_size);
        return;
    }
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
    {
        auto r = bench_inverse_impl<FieldElement>("Field Arithmetic", "inv (var)", cfg, bench_field_inv_var_k);
        print_result(r); results.push_back(r);
        rpt.add(r.section, r.name, r.ns_per_op, r.throughput_mops, r.batch_size);
    }

    // =================================================================
    // Section 1b: Scalar Arithmetic (inverse mod n)
    // =================================================================
    std::printf("\n=== Scalar Arithmetic ===\n");
    {
        auto r = bench_inverse_impl<Scalar>("Scalar Arithmetic", "scalar inv", cfg, bench_scalar_inv_k);
        print_result(r); results.push_back(r);
        rpt.add(r.section, r.name, r.ns_per_op, r.throughput_mops, r.batch_size);
    }
    {
        auto r = bench_inverse_impl<Scalar>("Scalar Arithmetic", "scalar inv (var)", cfg, bench_scalar_inv_var_k);
        print_result(r); results.push_back(r);
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
    // Section 4b: Verification (ECDSA / BIP-340 verify, ecrecover, hash160)
    // =================================================================
    // Core suite. Inputs are made once on the host by the CPU library; every
    // GPU result is checked after the timed passes. A row with any mismatch is
    // reported INVALID (no timing value) and the run exits non-zero.
    std::printf("\n=== Verification ===\n");
    {
        const int vn = std::min(cfg.batch_size, 1 << 16);
        VerifyInputs vin;
        std::printf("  (signing %d inputs on the CPU ...)\n", vn);
        if (!build_verify_inputs(vn, vin))
            std::fprintf(stderr, "  ERROR: CPU input generation failed; verification rows are INVALID\n");
        auto emit = [&](const BenchResult& r) {
            print_result(r); results.push_back(r);
            rpt.add(r.section, r.name, r.ns_per_op, r.throughput_mops, r.batch_size, r.valid);
        };
#if !SECP256K1_CUDA_LIMBS_32
        emit(bench_ecdsa_verify_impl(cfg, vin));
        emit(bench_schnorr_verify_impl(cfg, vin));
#if SECP256K1_GPU_HAS_ECRECOVER
        emit(bench_ecrecover_impl(cfg, vin));
#else
        std::printf("  %-28s | skipped (GPU ECRECOVER module disabled at build time)\n", "ecrecover");
#endif
#else
        std::printf("  verify / ecrecover rows skipped (no verify kernels in the 32-bit limb build)\n");
#endif
#if SECP256K1_GPU_HAS_HASH160
        emit(bench_hash160_impl(cfg, vin, 33, "hash160 (33B pubkey)"));
        emit(bench_hash160_impl(cfg, vin, 65, "hash160 (65B pubkey)"));
#else
        std::printf("  hash160 rows skipped (GPU HASH160 module disabled at build time)\n");
#endif
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
    // =================================================================
    // Section 8: MSM (multi-scalar multiplication)
    // =================================================================
    std::printf("\n=== MSM (Multi-Scalar Multiplication) ===\n");
    {
        auto r = bench_msm_impl(cfg, 1, "msm scatter only"); print_result(r); results.push_back(r);
        rpt.add(r.section, r.name, r.ns_per_op, r.throughput_mops, r.batch_size);
    }
    {
        auto r = bench_msm_impl(cfg, 2, "msm scatter+blockreduce"); print_result(r); results.push_back(r);
        rpt.add(r.section, r.name, r.ns_per_op, r.throughput_mops, r.batch_size);
    }
    {
        auto r = bench_msm_impl(cfg, 3, "msm sum(k_i*P_i)"); print_result(r); results.push_back(r);
        rpt.add(r.section, r.name, r.ns_per_op, r.throughput_mops, r.batch_size);
    }
#endif

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
        if (!r.valid) {
            std::printf("%10s | %14s | %d\n", "INVALID", "-", r.batch_size);
            continue;
        }
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

    int invalid_rows = 0;
    for (auto& r : results) {
        if (!r.valid) ++invalid_rows;
    }
    if (invalid_rows > 0) {
        std::fprintf(stderr, "ERROR: %d benchmark row(s) INVALID: GPU results did not match "
                     "the CPU reference (no timing reported for them)\n", invalid_rows);
        return 1;
    }

    std::printf("Benchmark complete.\n");
    return 0;
}
