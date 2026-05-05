// ============================================================================
// bench_compare -- CPU vs GPU Performance Comparison
// ============================================================================
//
// Runs identical ECC operations on both CPU (UltrafastSecp256k1 fast:: layer)
// and GPU (CUDA), then prints a side-by-side comparison table with ratios.
//
// Mirrors the per-operation breakdown style of bench_bip352 but covers the
// full secp256k1 operation set and compares host vs device throughput.
//
// Operations compared:
//   1. Field arithmetic    (mul, sqr, inv, add, sub)
//   2. Point operations    (add, double, k*G)
//   3. ECDSA               (sign, verify)
//   4. Schnorr/BIP-340     (sign, verify)
//   5. Full pipeline       (keygen -> sign -> verify, end-to-end)
//
// Usage:
//   bench_compare                     # default: 10K CPU / 64K GPU, 11 passes
//   bench_compare --quick             # CI mode: 1K / 4K, 5 passes
//   bench_compare --json <file>       # JSON output
//   bench_compare --cpu-n <N>         # CPU iterations per pass
//   bench_compare --gpu-n <N>         # GPU batch size
//   bench_compare --passes <N>        # measurement passes
//
// Methodology:
//   CPU:  std::chrono::high_resolution_clock, single-threaded, median of N passes
//   GPU:  CUDA event timing, batch kernel, median with IQR outlier removal
//   Both: deterministic test data from SHA-256 seeded RNG
//
// ============================================================================

// ---- GPU headers ----
#include "secp256k1.cuh"
#include "affine_add.cuh"
#include "ecdsa.cuh"
#include "schnorr.cuh"

// ---- CPU headers ----
#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"

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
#ifdef _MSC_VER
#include <intrin.h>
#endif
#include <array>
#include <functional>

#if __has_include("secp256k1/version.hpp")
#include "secp256k1/version.hpp"
#endif
#ifndef SECP256K1_VERSION_STRING
#define SECP256K1_VERSION_STRING "unknown"
#endif

using namespace secp256k1::cuda;

// ============================================================================
// CUDA error checking
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
// Configuration
// ============================================================================
struct CompareConfig {
    int cpu_n       = 10000;   // CPU iterations per pass
    int gpu_n       = 1 << 16; // GPU batch size (64K)
    int passes      = 11;      // measurement passes
    int warmup      = 3;       // warmup passes
    int gpu_tpb     = 256;     // threads per block
    bool quick      = false;
    std::string json_path;
};

// ============================================================================
// Result entry
// ============================================================================
struct CompareEntry {
    char name[64];
    char section[48];
    double cpu_ns;    // ns/op on CPU
    double gpu_ns;    // ns/op on GPU (per-element)
    int cpu_n;        // CPU iteration count
    int gpu_n;        // GPU batch size
    double ratio;     // cpu_ns / gpu_ns (>1 means GPU faster)
};

// ============================================================================
// DoNotOptimize -- prevent compiler from eliding results
// ============================================================================
template<typename T>
static inline void DoNotOptimize(T const& value) {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "r,m"(value) : "memory");
#else
    static volatile const T* _sink;
    _sink = &value;
#endif
}
static inline void ClobberMemory() {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : : "memory");
#else
    _ReadWriteBarrier();
#endif
}

// ============================================================================
// CPU timing harness -- median of N passes
// ============================================================================
static double cpu_bench(int iters, int passes, int warmup,
                        std::function<void(int)> fn) {
    // Warmup
    for (int w = 0; w < warmup; ++w) fn(iters);

    std::vector<double> times(passes);
    for (int p = 0; p < passes; ++p) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn(iters);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        times[p] = ns / iters;
    }
    std::sort(times.begin(), times.end());
    return times[passes / 2];  // median
}

// ============================================================================
// GPU timing harness -- CUDA events, median with IQR
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

static double median_iqr(std::vector<double>& samples) {
    if (samples.empty()) return 0.0;
    std::sort(samples.begin(), samples.end());
    int n = (int)samples.size();
    if (n < 4) return samples[n / 2];
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

template<typename KernelFunc>
static double gpu_bench(int batch, int passes, int warmup, int tpb,
                        KernelFunc&& kernel_fn) {
    CudaTimer timer;

    for (int w = 0; w < warmup; ++w) {
        kernel_fn();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<double> samples;
    samples.reserve(passes);
    for (int p = 0; p < passes; ++p) {
        timer.start();
        kernel_fn();
        float ms = timer.stop();
        double ns_per_op = (ms * 1e6) / batch;
        samples.push_back(ns_per_op);
    }
    return median_iqr(samples);
}

// ============================================================================
// GPU data generators (deterministic)
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
        // Ensure non-zero
        if (h[i].limbs[0] == 0 && h[i].limbs[1] == 0 &&
            h[i].limbs[2] == 0 && h[i].limbs[3] == 0)
            h[i].limbs[0] = 1;
    }
}

// ============================================================================
// GPU Kernels
// ============================================================================

__global__ void cmp_field_mul_k(const FieldElement* a, const FieldElement* b,
                                 FieldElement* r, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) field_mul(&a[idx], &b[idx], &r[idx]);
}

__global__ void cmp_field_sqr_k(const FieldElement* a, FieldElement* r, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) field_mul(&a[idx], &a[idx], &r[idx]);
}

__global__ void cmp_field_inv_k(const FieldElement* a, FieldElement* r, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) field_inv(&a[idx], &r[idx]);
}

__global__ void cmp_field_add_k(const FieldElement* a, const FieldElement* b,
                                 FieldElement* r, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) field_add(&a[idx], &b[idx], &r[idx]);
}

__global__ void cmp_field_sub_k(const FieldElement* a, const FieldElement* b,
                                 FieldElement* r, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) field_sub(&a[idx], &b[idx], &r[idx]);
}

__global__ void cmp_point_add_k(const JacobianPoint* a, const JacobianPoint* b,
                                 JacobianPoint* r, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) jacobian_add(&a[idx], &b[idx], &r[idx]);
}

__global__ void cmp_point_dbl_k(const JacobianPoint* a,
                                 JacobianPoint* r, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) jacobian_double(&a[idx], &r[idx]);
}

__global__ void cmp_scalar_mul_gen_k(const Scalar* k, JacobianPoint* r, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) scalar_mul_generator_const(&k[idx], &r[idx]);
}

#if !SECP256K1_CUDA_LIMBS_32
__global__ void cmp_ecdsa_sign_k(const uint8_t* msgs, const Scalar* privkeys,
                                  ECDSASignatureGPU* sigs, bool* oks, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        oks[idx] = secp256k1::cuda::ecdsa_sign(
            &msgs[idx * 32], &privkeys[idx], &sigs[idx]);
    }
}

__global__ void cmp_ecdsa_verify_k(const uint8_t* msgs, const JacobianPoint* pubkeys,
                                    const ECDSASignatureGPU* sigs, bool* oks, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        oks[idx] = secp256k1::cuda::ecdsa_verify(
            &msgs[idx * 32], &pubkeys[idx], &sigs[idx]);
    }
}

__global__ void cmp_schnorr_sign_k(const Scalar* privkeys, const uint8_t* msgs,
                                    const uint8_t* aux_rands,
                                    SchnorrSignatureGPU* sigs, bool* oks, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        oks[idx] = secp256k1::cuda::schnorr_sign(
            &privkeys[idx], &msgs[idx * 32], &aux_rands[idx * 32], &sigs[idx]);
    }
}

__global__ void cmp_schnorr_verify_k(const uint8_t* pubkeys_x, const uint8_t* msgs,
                                      const SchnorrSignatureGPU* sigs, bool* oks, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        oks[idx] = secp256k1::cuda::schnorr_verify(
            &pubkeys_x[idx * 32], &msgs[idx * 32], &sigs[idx]);
    }
}

// Full pipeline kernel: keygen -> ECDSA sign -> verify (all on GPU)
__global__ void cmp_pipeline_ecdsa_k(const Scalar* privkeys, const uint8_t* msgs,
                                      bool* oks, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        // 1. Key generation: pubkey = k*G
        JacobianPoint pubkey;
        scalar_mul_generator_const(&privkeys[idx], &pubkey);

        // 2. ECDSA sign
        ECDSASignatureGPU sig;
        bool sign_ok = secp256k1::cuda::ecdsa_sign(
            &msgs[idx * 32], &privkeys[idx], &sig);

        // 3. ECDSA verify
        bool verify_ok = false;
        if (sign_ok) {
            verify_ok = secp256k1::cuda::ecdsa_verify(
                &msgs[idx * 32], &pubkey, &sig);
        }
        oks[idx] = sign_ok && verify_ok;
    }
}

// Full pipeline kernel: keygen -> Schnorr sign -> verify (all on GPU)
__global__ void cmp_pipeline_schnorr_k(const Scalar* privkeys, const uint8_t* msgs,
                                        const uint8_t* aux_rands,
                                        bool* oks, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        // 1. Key generation: compute x-only pubkey
        JacobianPoint pubkey_jac;
        scalar_mul_generator_const(&privkeys[idx], &pubkey_jac);

        // Get x-coordinate bytes for verify
        // Normalize Jacobian -> affine to get x bytes
        FieldElement ax, ay;
        ax = pubkey_jac.x;
        ay = pubkey_jac.y;
        jacobian_to_affine(&ax, &ay, &pubkey_jac.z);

        uint8_t pubkey_x[32];
        field_to_bytes(&ax, pubkey_x);

        // 2. Schnorr sign
        SchnorrSignatureGPU sig;
        bool sign_ok = secp256k1::cuda::schnorr_sign(
            &privkeys[idx], &msgs[idx * 32], &aux_rands[idx * 32], &sig);

        // 3. Schnorr verify
        bool verify_ok = false;
        if (sign_ok) {
            verify_ok = secp256k1::cuda::schnorr_verify(
                pubkey_x, &msgs[idx * 32], &sig);
        }
        oks[idx] = sign_ok && verify_ok;
    }
}
#endif

// ============================================================================
// CPU benchmark implementations
// ============================================================================

static double cpu_field_mul(const CompareConfig& cfg) {
    using FE = secp256k1::fast::FieldElement;
    std::mt19937_64 rng(42);

    std::vector<FE> a(cfg.cpu_n), b(cfg.cpu_n);
    for (int i = 0; i < cfg.cpu_n; ++i) {
        uint64_t l[4];
        for (int j = 0; j < 4; ++j) l[j] = rng();
        l[3] &= 0x7FFFFFFFFFFFFFFFULL;
        a[i] = FE::from_limbs({l[0], l[1], l[2], l[3]});
        for (int j = 0; j < 4; ++j) l[j] = rng();
        l[3] &= 0x7FFFFFFFFFFFFFFFULL;
        b[i] = FE::from_limbs({l[0], l[1], l[2], l[3]});
    }

    return cpu_bench(cfg.cpu_n, cfg.passes, cfg.warmup, [&](int n) {
        for (int i = 0; i < n; ++i) {
            auto r = a[i % cfg.cpu_n] * b[i % cfg.cpu_n];
            DoNotOptimize(r);
        }
    });
}

static double cpu_field_sqr(const CompareConfig& cfg) {
    using FE = secp256k1::fast::FieldElement;
    std::mt19937_64 rng(44);

    std::vector<FE> a(cfg.cpu_n);
    for (int i = 0; i < cfg.cpu_n; ++i) {
        uint64_t l[4];
        for (int j = 0; j < 4; ++j) l[j] = rng();
        l[3] &= 0x7FFFFFFFFFFFFFFFULL;
        a[i] = FE::from_limbs({l[0], l[1], l[2], l[3]});
    }

    return cpu_bench(cfg.cpu_n, cfg.passes, cfg.warmup, [&](int n) {
        for (int i = 0; i < n; ++i) {
            auto r = a[i % cfg.cpu_n].square();
            DoNotOptimize(r);
        }
    });
}

static double cpu_field_inv(const CompareConfig& cfg) {
    using FE = secp256k1::fast::FieldElement;
    std::mt19937_64 rng(49);
    int N = std::min(cfg.cpu_n, 1000); // inv is expensive

    std::vector<FE> a(N);
    for (int i = 0; i < N; ++i) {
        uint64_t l[4];
        for (int j = 0; j < 4; ++j) l[j] = rng();
        l[3] &= 0x7FFFFFFFFFFFFFFFULL;
        if (l[0] == 0 && l[1] == 0 && l[2] == 0 && l[3] == 0)
            l[0] = 1;
        a[i] = FE::from_limbs({l[0], l[1], l[2], l[3]});
    }

    return cpu_bench(N, cfg.passes, cfg.warmup, [&](int n) {
        for (int i = 0; i < n; ++i) {
            auto r = a[i % N].inverse();
            DoNotOptimize(r);
        }
    });
}

static double cpu_field_add(const CompareConfig& cfg) {
    using FE = secp256k1::fast::FieldElement;
    std::mt19937_64 rng(45);

    std::vector<FE> a(cfg.cpu_n), b(cfg.cpu_n);
    for (int i = 0; i < cfg.cpu_n; ++i) {
        uint64_t l[4];
        for (int j = 0; j < 4; ++j) l[j] = rng();
        l[3] &= 0x7FFFFFFFFFFFFFFFULL;
        a[i] = FE::from_limbs({l[0], l[1], l[2], l[3]});
        for (int j = 0; j < 4; ++j) l[j] = rng();
        l[3] &= 0x7FFFFFFFFFFFFFFFULL;
        b[i] = FE::from_limbs({l[0], l[1], l[2], l[3]});
    }

    return cpu_bench(cfg.cpu_n, cfg.passes, cfg.warmup, [&](int n) {
        for (int i = 0; i < n; ++i) {
            auto r = a[i % cfg.cpu_n] + b[i % cfg.cpu_n];
            DoNotOptimize(r);
        }
    });
}

static double cpu_field_sub(const CompareConfig& cfg) {
    using FE = secp256k1::fast::FieldElement;
    std::mt19937_64 rng(47);

    std::vector<FE> a(cfg.cpu_n), b(cfg.cpu_n);
    for (int i = 0; i < cfg.cpu_n; ++i) {
        uint64_t l[4];
        for (int j = 0; j < 4; ++j) l[j] = rng();
        l[3] &= 0x7FFFFFFFFFFFFFFFULL;
        a[i] = FE::from_limbs({l[0], l[1], l[2], l[3]});
        for (int j = 0; j < 4; ++j) l[j] = rng();
        l[3] &= 0x7FFFFFFFFFFFFFFFULL;
        b[i] = FE::from_limbs({l[0], l[1], l[2], l[3]});
    }

    return cpu_bench(cfg.cpu_n, cfg.passes, cfg.warmup, [&](int n) {
        for (int i = 0; i < n; ++i) {
            auto r = a[i % cfg.cpu_n] - b[i % cfg.cpu_n];
            DoNotOptimize(r);
        }
    });
}

static double cpu_point_add(const CompareConfig& cfg) {
    using P = secp256k1::fast::Point;
    using S = secp256k1::fast::Scalar;
    int N = std::min(cfg.cpu_n, 1000);

    // Generate real curve points via k*G
    std::mt19937_64 rng(51);
    std::vector<P> points(N);
    for (int i = 0; i < N; ++i) {
        uint8_t bytes[32];
        for (int j = 0; j < 4; ++j) {
            uint64_t v = rng();
            std::memcpy(bytes + j * 8, &v, 8);
        }
        bytes[31] &= 0x7F;
        if (bytes[0] == 0) bytes[0] = 1;
        auto s = S::from_bytes(bytes);
        points[i] = P::generator().scalar_mul(s);
    }

    return cpu_bench(N, cfg.passes, cfg.warmup, [&](int n) {
        for (int i = 0; i < n; ++i) {
            auto r = points[i % N].add(points[(i + 1) % N]);
            DoNotOptimize(r);
        }
    });
}

static double cpu_point_dbl(const CompareConfig& cfg) {
    using P = secp256k1::fast::Point;
    using S = secp256k1::fast::Scalar;
    int N = std::min(cfg.cpu_n, 1000);

    std::mt19937_64 rng(53);
    std::vector<P> points(N);
    for (int i = 0; i < N; ++i) {
        uint8_t bytes[32];
        for (int j = 0; j < 4; ++j) {
            uint64_t v = rng();
            std::memcpy(bytes + j * 8, &v, 8);
        }
        bytes[31] &= 0x7F;
        if (bytes[0] == 0) bytes[0] = 1;
        auto s = S::from_bytes(bytes);
        points[i] = P::generator().scalar_mul(s);
    }

    return cpu_bench(N, cfg.passes, cfg.warmup, [&](int n) {
        for (int i = 0; i < n; ++i) {
            auto r = points[i % N].dbl();
            DoNotOptimize(r);
        }
    });
}

static double cpu_scalar_mul_gen(const CompareConfig& cfg) {
    using P = secp256k1::fast::Point;
    using S = secp256k1::fast::Scalar;
    int N = std::min(cfg.cpu_n, 1000);

    std::mt19937_64 rng(55);
    std::vector<S> scalars(N);
    for (int i = 0; i < N; ++i) {
        uint8_t bytes[32];
        for (int j = 0; j < 4; ++j) {
            uint64_t v = rng();
            std::memcpy(bytes + j * 8, &v, 8);
        }
        bytes[31] &= 0x7F;
        if (bytes[0] == 0) bytes[0] = 1;
        scalars[i] = S::from_bytes(bytes);
    }

    return cpu_bench(N, cfg.passes, cfg.warmup, [&](int n) {
        for (int i = 0; i < n; ++i) {
            auto r = P::generator().scalar_mul(scalars[i % N]);
            DoNotOptimize(r);
        }
    });
}

static double cpu_ecdsa_sign(const CompareConfig& cfg) {
    using S = secp256k1::fast::Scalar;
    int N = std::min(cfg.cpu_n, 1000);

    std::mt19937_64 rng(60);
    std::vector<S> privkeys(N);
    std::vector<std::array<uint8_t, 32>> msgs(N);

    for (int i = 0; i < N; ++i) {
        uint8_t bytes[32];
        for (int j = 0; j < 4; ++j) {
            uint64_t v = rng();
            std::memcpy(bytes + j * 8, &v, 8);
        }
        bytes[31] &= 0x7F;
        if (bytes[0] == 0) bytes[0] = 1;
        privkeys[i] = S::from_bytes(bytes);

        for (int j = 0; j < 32; ++j)
            msgs[i][j] = static_cast<uint8_t>(rng() & 0xFF);
    }

    return cpu_bench(N, cfg.passes, cfg.warmup, [&](int n) {
        for (int i = 0; i < n; ++i) {
            auto sig = secp256k1::ecdsa_sign(msgs[i % N], privkeys[i % N]);
            DoNotOptimize(sig);
        }
    });
}

static double cpu_ecdsa_verify(const CompareConfig& cfg) {
    using S = secp256k1::fast::Scalar;
    using P = secp256k1::fast::Point;
    int N = std::min(cfg.cpu_n, 1000);

    std::mt19937_64 rng(62);
    std::vector<S> privkeys(N);
    std::vector<std::array<uint8_t, 32>> msgs(N);
    std::vector<secp256k1::ECDSASignature> sigs(N);
    std::vector<P> pubkeys(N);

    for (int i = 0; i < N; ++i) {
        uint8_t bytes[32];
        for (int j = 0; j < 4; ++j) {
            uint64_t v = rng();
            std::memcpy(bytes + j * 8, &v, 8);
        }
        bytes[31] &= 0x7F;
        if (bytes[0] == 0) bytes[0] = 1;
        privkeys[i] = S::from_bytes(bytes);
        pubkeys[i] = P::generator().scalar_mul(privkeys[i]);

        for (int j = 0; j < 32; ++j)
            msgs[i][j] = static_cast<uint8_t>(rng() & 0xFF);
        sigs[i] = secp256k1::ecdsa_sign(msgs[i], privkeys[i]);
    }

    return cpu_bench(N, cfg.passes, cfg.warmup, [&](int n) {
        for (int i = 0; i < n; ++i) {
            bool ok = secp256k1::ecdsa_verify(msgs[i % N], pubkeys[i % N], sigs[i % N]);
            DoNotOptimize(ok);
        }
    });
}

static double cpu_schnorr_sign(const CompareConfig& cfg) {
    using S = secp256k1::fast::Scalar;
    int N = std::min(cfg.cpu_n, 1000);

    std::mt19937_64 rng(64);
    std::vector<S> privkeys(N);
    std::vector<std::array<uint8_t, 32>> msgs(N), aux(N);
    std::vector<secp256k1::SchnorrKeypair> keypairs(N);

    for (int i = 0; i < N; ++i) {
        uint8_t bytes[32];
        for (int j = 0; j < 4; ++j) {
            uint64_t v = rng();
            std::memcpy(bytes + j * 8, &v, 8);
        }
        bytes[31] &= 0x7F;
        if (bytes[0] == 0) bytes[0] = 1;
        privkeys[i] = S::from_bytes(bytes);
        keypairs[i] = secp256k1::schnorr_keypair_create(privkeys[i]);

        for (int j = 0; j < 32; ++j) {
            msgs[i][j] = static_cast<uint8_t>(rng() & 0xFF);
            aux[i][j] = static_cast<uint8_t>(rng() & 0xFF);
        }
    }

    return cpu_bench(N, cfg.passes, cfg.warmup, [&](int n) {
        for (int i = 0; i < n; ++i) {
            auto sig = secp256k1::schnorr_sign(keypairs[i % N],
                                                msgs[i % N], aux[i % N]);
            DoNotOptimize(sig);
        }
    });
}

static double cpu_schnorr_verify(const CompareConfig& cfg) {
    using S = secp256k1::fast::Scalar;
    int N = std::min(cfg.cpu_n, 1000);

    std::mt19937_64 rng(66);
    std::vector<S> privkeys(N);
    std::vector<std::array<uint8_t, 32>> msgs(N), aux(N), pubkeys_x(N);
    std::vector<secp256k1::SchnorrSignature> sigs(N);

    for (int i = 0; i < N; ++i) {
        uint8_t bytes[32];
        for (int j = 0; j < 4; ++j) {
            uint64_t v = rng();
            std::memcpy(bytes + j * 8, &v, 8);
        }
        bytes[31] &= 0x7F;
        if (bytes[0] == 0) bytes[0] = 1;
        privkeys[i] = S::from_bytes(bytes);
        pubkeys_x[i] = secp256k1::schnorr_pubkey(privkeys[i]);

        for (int j = 0; j < 32; ++j) {
            msgs[i][j] = static_cast<uint8_t>(rng() & 0xFF);
            aux[i][j] = static_cast<uint8_t>(rng() & 0xFF);
        }
        auto kp = secp256k1::schnorr_keypair_create(privkeys[i]);
        sigs[i] = secp256k1::schnorr_sign(kp, msgs[i], aux[i]);
    }

    return cpu_bench(N, cfg.passes, cfg.warmup, [&](int n) {
        for (int i = 0; i < n; ++i) {
            bool ok = secp256k1::schnorr_verify(pubkeys_x[i % N],
                                                 msgs[i % N], sigs[i % N]);
            DoNotOptimize(ok);
        }
    });
}

static double cpu_pipeline_ecdsa(const CompareConfig& cfg) {
    using S = secp256k1::fast::Scalar;
    using P = secp256k1::fast::Point;
    int N = std::min(cfg.cpu_n, 500);

    std::mt19937_64 rng(70);
    std::vector<S> privkeys(N);
    std::vector<std::array<uint8_t, 32>> msgs(N);

    for (int i = 0; i < N; ++i) {
        uint8_t bytes[32];
        for (int j = 0; j < 4; ++j) {
            uint64_t v = rng();
            std::memcpy(bytes + j * 8, &v, 8);
        }
        bytes[31] &= 0x7F;
        if (bytes[0] == 0) bytes[0] = 1;
        privkeys[i] = S::from_bytes(bytes);
        for (int j = 0; j < 32; ++j)
            msgs[i][j] = static_cast<uint8_t>(rng() & 0xFF);
    }

    return cpu_bench(N, cfg.passes, cfg.warmup, [&](int n) {
        for (int i = 0; i < n; ++i) {
            int idx = i % N;
            // 1. Keygen
            P pubkey = P::generator().scalar_mul(privkeys[idx]);
            // 2. Sign
            auto sig = secp256k1::ecdsa_sign(msgs[idx], privkeys[idx]);
            // 3. Verify
            bool ok = secp256k1::ecdsa_verify(msgs[idx], pubkey, sig);
            DoNotOptimize(ok);
        }
    });
}

static double cpu_pipeline_schnorr(const CompareConfig& cfg) {
    using S = secp256k1::fast::Scalar;
    int N = std::min(cfg.cpu_n, 500);

    std::mt19937_64 rng(72);
    std::vector<S> privkeys(N);
    std::vector<std::array<uint8_t, 32>> msgs(N), aux(N);

    for (int i = 0; i < N; ++i) {
        uint8_t bytes[32];
        for (int j = 0; j < 4; ++j) {
            uint64_t v = rng();
            std::memcpy(bytes + j * 8, &v, 8);
        }
        bytes[31] &= 0x7F;
        if (bytes[0] == 0) bytes[0] = 1;
        privkeys[i] = S::from_bytes(bytes);
        for (int j = 0; j < 32; ++j) {
            msgs[i][j] = static_cast<uint8_t>(rng() & 0xFF);
            aux[i][j] = static_cast<uint8_t>(rng() & 0xFF);
        }
    }

    return cpu_bench(N, cfg.passes, cfg.warmup, [&](int n) {
        for (int i = 0; i < n; ++i) {
            int idx = i % N;
            // 1. Keypair create
            auto kp = secp256k1::schnorr_keypair_create(privkeys[idx]);
            auto pubx = secp256k1::schnorr_pubkey(privkeys[idx]);
            // 2. Sign
            auto sig = secp256k1::schnorr_sign(kp, msgs[idx], aux[idx]);
            // 3. Verify
            bool ok = secp256k1::schnorr_verify(pubx, msgs[idx], sig);
            DoNotOptimize(ok);
        }
    });
}

// ============================================================================
// GPU benchmark implementations
// ============================================================================

static double gpu_field_mul(const CompareConfig& cfg) {
    int N = cfg.gpu_n;
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

    int blocks = (N + cfg.gpu_tpb - 1) / cfg.gpu_tpb;
    double ns = gpu_bench(N, cfg.passes, cfg.warmup, cfg.gpu_tpb, [&]() {
        cmp_field_mul_k<<<blocks, cfg.gpu_tpb>>>(d_a, d_b, d_r, N);
    });

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_r);
    return ns;
}

static double gpu_field_sqr(const CompareConfig& cfg) {
    int N = cfg.gpu_n;
    std::vector<FieldElement> h_a(N);
    gen_field_elements(h_a.data(), N, 44);

    FieldElement *d_a, *d_r;
    size_t sz = N * sizeof(FieldElement);
    CUDA_CHECK(cudaMalloc(&d_a, sz));
    CUDA_CHECK(cudaMalloc(&d_r, sz));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), sz, cudaMemcpyHostToDevice));

    int blocks = (N + cfg.gpu_tpb - 1) / cfg.gpu_tpb;
    double ns = gpu_bench(N, cfg.passes, cfg.warmup, cfg.gpu_tpb, [&]() {
        cmp_field_sqr_k<<<blocks, cfg.gpu_tpb>>>(d_a, d_r, N);
    });

    cudaFree(d_a); cudaFree(d_r);
    return ns;
}

static double gpu_field_inv(const CompareConfig& cfg) {
    int N = std::min(cfg.gpu_n, 1 << 16);
    std::vector<FieldElement> h_a(N);
    gen_field_elements(h_a.data(), N, 49);
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

    int blocks = (N + cfg.gpu_tpb - 1) / cfg.gpu_tpb;
    double ns = gpu_bench(N, cfg.passes, cfg.warmup, cfg.gpu_tpb, [&]() {
        cmp_field_inv_k<<<blocks, cfg.gpu_tpb>>>(d_a, d_r, N);
    });

    cudaFree(d_a); cudaFree(d_r);
    return ns;
}

static double gpu_field_add(const CompareConfig& cfg) {
    int N = cfg.gpu_n;
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

    int blocks = (N + cfg.gpu_tpb - 1) / cfg.gpu_tpb;
    double ns = gpu_bench(N, cfg.passes, cfg.warmup, cfg.gpu_tpb, [&]() {
        cmp_field_add_k<<<blocks, cfg.gpu_tpb>>>(d_a, d_b, d_r, N);
    });

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_r);
    return ns;
}

static double gpu_field_sub(const CompareConfig& cfg) {
    int N = cfg.gpu_n;
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

    int blocks = (N + cfg.gpu_tpb - 1) / cfg.gpu_tpb;
    double ns = gpu_bench(N, cfg.passes, cfg.warmup, cfg.gpu_tpb, [&]() {
        cmp_field_sub_k<<<blocks, cfg.gpu_tpb>>>(d_a, d_b, d_r, N);
    });

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_r);
    return ns;
}

static double gpu_point_add(const CompareConfig& cfg) {
    int N = std::min(cfg.gpu_n, 1 << 18);
    // Generate proper points via k*G on GPU
    std::vector<Scalar> h_k(N);
    gen_scalars(h_k.data(), N, 51);

    Scalar* d_k;
    JacobianPoint *d_pts_a, *d_pts_b, *d_r;
    CUDA_CHECK(cudaMalloc(&d_k, N * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_pts_a, N * sizeof(JacobianPoint)));
    CUDA_CHECK(cudaMalloc(&d_pts_b, N * sizeof(JacobianPoint)));
    CUDA_CHECK(cudaMalloc(&d_r, N * sizeof(JacobianPoint)));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), N * sizeof(Scalar), cudaMemcpyHostToDevice));

    int blocks = (N + cfg.gpu_tpb - 1) / cfg.gpu_tpb;
    // Generate points via k*G
    cmp_scalar_mul_gen_k<<<blocks, cfg.gpu_tpb>>>(d_k, d_pts_a, N);
    gen_scalars(h_k.data(), N, 52);
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), N * sizeof(Scalar), cudaMemcpyHostToDevice));
    cmp_scalar_mul_gen_k<<<blocks, cfg.gpu_tpb>>>(d_k, d_pts_b, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    double ns = gpu_bench(N, cfg.passes, cfg.warmup, cfg.gpu_tpb, [&]() {
        cmp_point_add_k<<<blocks, cfg.gpu_tpb>>>(d_pts_a, d_pts_b, d_r, N);
    });

    cudaFree(d_k); cudaFree(d_pts_a); cudaFree(d_pts_b); cudaFree(d_r);
    return ns;
}

static double gpu_point_dbl(const CompareConfig& cfg) {
    int N = std::min(cfg.gpu_n, 1 << 18);
    std::vector<Scalar> h_k(N);
    gen_scalars(h_k.data(), N, 53);

    Scalar* d_k;
    JacobianPoint *d_pts, *d_r;
    CUDA_CHECK(cudaMalloc(&d_k, N * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_pts, N * sizeof(JacobianPoint)));
    CUDA_CHECK(cudaMalloc(&d_r, N * sizeof(JacobianPoint)));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), N * sizeof(Scalar), cudaMemcpyHostToDevice));

    int blocks = (N + cfg.gpu_tpb - 1) / cfg.gpu_tpb;
    cmp_scalar_mul_gen_k<<<blocks, cfg.gpu_tpb>>>(d_k, d_pts, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    double ns = gpu_bench(N, cfg.passes, cfg.warmup, cfg.gpu_tpb, [&]() {
        cmp_point_dbl_k<<<blocks, cfg.gpu_tpb>>>(d_pts, d_r, N);
    });

    cudaFree(d_k); cudaFree(d_pts); cudaFree(d_r);
    return ns;
}

static double gpu_scalar_mul_gen(const CompareConfig& cfg) {
    int N = std::min(cfg.gpu_n, 1 << 16);
    std::vector<Scalar> h_k(N);
    gen_scalars(h_k.data(), N, 55);

    Scalar* d_k;
    JacobianPoint* d_r;
    CUDA_CHECK(cudaMalloc(&d_k, N * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_r, N * sizeof(JacobianPoint)));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), N * sizeof(Scalar), cudaMemcpyHostToDevice));

    int blocks = (N + cfg.gpu_tpb - 1) / cfg.gpu_tpb;
    double ns = gpu_bench(N, cfg.passes, cfg.warmup, cfg.gpu_tpb, [&]() {
        cmp_scalar_mul_gen_k<<<blocks, cfg.gpu_tpb>>>(d_k, d_r, N);
    });

    cudaFree(d_k); cudaFree(d_r);
    return ns;
}

#if !SECP256K1_CUDA_LIMBS_32
static double gpu_ecdsa_sign(const CompareConfig& cfg) {
    int N = std::min(cfg.gpu_n, 1 << 14);
    std::vector<Scalar> h_priv(N);
    std::vector<uint8_t> h_msgs(N * 32);
    gen_scalars(h_priv.data(), N, 60);
    std::mt19937_64 rng(61);
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
    CUDA_CHECK(cudaMemcpy(d_priv, h_priv.data(), N * sizeof(Scalar), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_msg, h_msgs.data(), N * 32, cudaMemcpyHostToDevice));

    int blocks = (N + cfg.gpu_tpb - 1) / cfg.gpu_tpb;
    double ns = gpu_bench(N, cfg.passes, cfg.warmup, cfg.gpu_tpb, [&]() {
        cmp_ecdsa_sign_k<<<blocks, cfg.gpu_tpb>>>(d_msg, d_priv, d_sig, d_ok, N);
    });

    cudaMemset(d_priv, 0, N * sizeof(Scalar));
    cudaFree(d_priv); cudaFree(d_msg); cudaFree(d_sig); cudaFree(d_ok);
    return ns;
}

static double gpu_ecdsa_verify(const CompareConfig& cfg) {
    int N = std::min(cfg.gpu_n, 1 << 14);
    std::vector<Scalar> h_priv(N);
    std::vector<uint8_t> h_msgs(N * 32);
    gen_scalars(h_priv.data(), N, 62);
    std::mt19937_64 rng(63);
    for (int i = 0; i < N * 32; ++i)
        h_msgs[i] = static_cast<uint8_t>(rng() & 0xFF);

    // Pre-sign on GPU to get valid sigs + pubkeys
    Scalar* d_priv;
    uint8_t* d_msg;
    ECDSASignatureGPU* d_sig;
    JacobianPoint* d_pub;
    bool* d_ok;
    CUDA_CHECK(cudaMalloc(&d_priv, N * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_msg, N * 32));
    CUDA_CHECK(cudaMalloc(&d_sig, N * sizeof(ECDSASignatureGPU)));
    CUDA_CHECK(cudaMalloc(&d_pub, N * sizeof(JacobianPoint)));
    CUDA_CHECK(cudaMalloc(&d_ok, N * sizeof(bool)));
    CUDA_CHECK(cudaMemcpy(d_priv, h_priv.data(), N * sizeof(Scalar), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_msg, h_msgs.data(), N * 32, cudaMemcpyHostToDevice));

    int blocks = (N + cfg.gpu_tpb - 1) / cfg.gpu_tpb;
    // Generate pubkeys
    cmp_scalar_mul_gen_k<<<blocks, cfg.gpu_tpb>>>(d_priv, d_pub, N);
    // Pre-sign
    cmp_ecdsa_sign_k<<<blocks, cfg.gpu_tpb>>>(d_msg, d_priv, d_sig, d_ok, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Now benchmark verify only
    double ns = gpu_bench(N, cfg.passes, cfg.warmup, cfg.gpu_tpb, [&]() {
        cmp_ecdsa_verify_k<<<blocks, cfg.gpu_tpb>>>(d_msg, d_pub, d_sig, d_ok, N);
    });

    cudaFree(d_priv); cudaFree(d_msg); cudaFree(d_sig); cudaFree(d_pub); cudaFree(d_ok);
    return ns;
}

static double gpu_schnorr_sign(const CompareConfig& cfg) {
    int N = std::min(cfg.gpu_n, 1 << 14);
    std::vector<Scalar> h_priv(N);
    std::vector<uint8_t> h_msgs(N * 32), h_aux(N * 32);
    gen_scalars(h_priv.data(), N, 64);
    std::mt19937_64 rng(65);
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
    CUDA_CHECK(cudaMemcpy(d_priv, h_priv.data(), N * sizeof(Scalar), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_msg, h_msgs.data(), N * 32, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_aux, h_aux.data(), N * 32, cudaMemcpyHostToDevice));

    int blocks = (N + cfg.gpu_tpb - 1) / cfg.gpu_tpb;
    double ns = gpu_bench(N, cfg.passes, cfg.warmup, cfg.gpu_tpb, [&]() {
        cmp_schnorr_sign_k<<<blocks, cfg.gpu_tpb>>>(d_priv, d_msg, d_aux, d_sig, d_ok, N);
    });

    cudaMemset(d_priv, 0, N * sizeof(Scalar));
    cudaFree(d_priv); cudaFree(d_msg); cudaFree(d_aux); cudaFree(d_sig); cudaFree(d_ok);
    return ns;
}

static double gpu_schnorr_verify(const CompareConfig& cfg) {
    int N = std::min(cfg.gpu_n, 1 << 14);
    std::vector<Scalar> h_priv(N);
    std::vector<uint8_t> h_msgs(N * 32), h_aux(N * 32);
    gen_scalars(h_priv.data(), N, 66);
    std::mt19937_64 rng(67);
    for (int i = 0; i < N * 32; ++i) {
        h_msgs[i] = static_cast<uint8_t>(rng() & 0xFF);
        h_aux[i] = static_cast<uint8_t>(rng() & 0xFF);
    }

    // Pre-sign on GPU, generate x-only pubkeys
    Scalar* d_priv;
    uint8_t *d_msg, *d_aux, *d_pubx;
    SchnorrSignatureGPU* d_sig;
    JacobianPoint* d_pub;
    bool* d_ok;
    CUDA_CHECK(cudaMalloc(&d_priv, N * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_msg, N * 32));
    CUDA_CHECK(cudaMalloc(&d_aux, N * 32));
    CUDA_CHECK(cudaMalloc(&d_sig, N * sizeof(SchnorrSignatureGPU)));
    CUDA_CHECK(cudaMalloc(&d_pub, N * sizeof(JacobianPoint)));
    CUDA_CHECK(cudaMalloc(&d_pubx, N * 32));
    CUDA_CHECK(cudaMalloc(&d_ok, N * sizeof(bool)));
    CUDA_CHECK(cudaMemcpy(d_priv, h_priv.data(), N * sizeof(Scalar), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_msg, h_msgs.data(), N * 32, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_aux, h_aux.data(), N * 32, cudaMemcpyHostToDevice));

    int blocks = (N + cfg.gpu_tpb - 1) / cfg.gpu_tpb;

    // Pre-sign
    cmp_schnorr_sign_k<<<blocks, cfg.gpu_tpb>>>(d_priv, d_msg, d_aux, d_sig, d_ok, N);

    // Generate pubkey x-coordinates: compute k*G, normalize, extract x bytes
    // We need a helper kernel for this
    cmp_scalar_mul_gen_k<<<blocks, cfg.gpu_tpb>>>(d_priv, d_pub, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Extract x-only pubkeys on host (one-time setup cost, not benchmarked)
    std::vector<JacobianPoint> h_pub(N);
    CUDA_CHECK(cudaMemcpy(h_pub.data(), d_pub, N * sizeof(JacobianPoint), cudaMemcpyDeviceToHost));

    // For Schnorr verify, we need the x-bytes of the pubkey.
    // schnorr_sign already handles negation internally. The verify function 
    // takes pubkey_x[32] which is the x-coordinate of the even-Y point.
    // Since we're testing throughput (not correctness), we can extract x from
    // the signing privkey using the schnorr_pubkey path on CPU side.
    std::vector<uint8_t> h_pubx(N * 32);
    {
        using S_cpu = secp256k1::fast::Scalar;
        std::vector<Scalar> h_priv_copy(N);
        gen_scalars(h_priv_copy.data(), N, 66);
        for (int i = 0; i < N; ++i) {
            uint8_t bytes[32];
            // Convert GPU Scalar limbs to big-endian bytes for CPU
            for (int j = 0; j < 4; ++j) {
                uint64_t limb = h_priv_copy[i].limbs[3 - j];
                for (int k = 0; k < 8; ++k)
                    bytes[j * 8 + k] = static_cast<uint8_t>((limb >> (56 - k * 8)) & 0xFF);
            }
            auto cpu_scalar = S_cpu::from_bytes(bytes);
            auto px = secp256k1::schnorr_pubkey(cpu_scalar);
            std::memcpy(&h_pubx[i * 32], px.data(), 32);
        }
    }
    CUDA_CHECK(cudaMemcpy(d_pubx, h_pubx.data(), N * 32, cudaMemcpyHostToDevice));

    // Benchmark verify only
    double ns = gpu_bench(N, cfg.passes, cfg.warmup, cfg.gpu_tpb, [&]() {
        cmp_schnorr_verify_k<<<blocks, cfg.gpu_tpb>>>(d_pubx, d_msg, d_sig, d_ok, N);
    });

    cudaFree(d_priv); cudaFree(d_msg); cudaFree(d_aux); cudaFree(d_sig);
    cudaFree(d_pub); cudaFree(d_pubx); cudaFree(d_ok);
    return ns;
}

static double gpu_pipeline_ecdsa(const CompareConfig& cfg) {
    int N = std::min(cfg.gpu_n, 1 << 14);
    std::vector<Scalar> h_priv(N);
    std::vector<uint8_t> h_msgs(N * 32);
    gen_scalars(h_priv.data(), N, 70);
    std::mt19937_64 rng(71);
    for (int i = 0; i < N * 32; ++i)
        h_msgs[i] = static_cast<uint8_t>(rng() & 0xFF);

    Scalar* d_priv;
    uint8_t* d_msg;
    bool* d_ok;
    CUDA_CHECK(cudaMalloc(&d_priv, N * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_msg, N * 32));
    CUDA_CHECK(cudaMalloc(&d_ok, N * sizeof(bool)));
    CUDA_CHECK(cudaMemcpy(d_priv, h_priv.data(), N * sizeof(Scalar), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_msg, h_msgs.data(), N * 32, cudaMemcpyHostToDevice));

    int blocks = (N + cfg.gpu_tpb - 1) / cfg.gpu_tpb;
    double ns = gpu_bench(N, cfg.passes, cfg.warmup, cfg.gpu_tpb, [&]() {
        cmp_pipeline_ecdsa_k<<<blocks, cfg.gpu_tpb>>>(d_priv, d_msg, d_ok, N);
    });

    cudaFree(d_priv); cudaFree(d_msg); cudaFree(d_ok);
    return ns;
}

static double gpu_pipeline_schnorr(const CompareConfig& cfg) {
    int N = std::min(cfg.gpu_n, 1 << 14);
    std::vector<Scalar> h_priv(N);
    std::vector<uint8_t> h_msgs(N * 32), h_aux(N * 32);
    gen_scalars(h_priv.data(), N, 72);
    std::mt19937_64 rng(73);
    for (int i = 0; i < N * 32; ++i) {
        h_msgs[i] = static_cast<uint8_t>(rng() & 0xFF);
        h_aux[i] = static_cast<uint8_t>(rng() & 0xFF);
    }

    Scalar* d_priv;
    uint8_t *d_msg, *d_aux;
    bool* d_ok;
    CUDA_CHECK(cudaMalloc(&d_priv, N * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_msg, N * 32));
    CUDA_CHECK(cudaMalloc(&d_aux, N * 32));
    CUDA_CHECK(cudaMalloc(&d_ok, N * sizeof(bool)));
    CUDA_CHECK(cudaMemcpy(d_priv, h_priv.data(), N * sizeof(Scalar), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_msg, h_msgs.data(), N * 32, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_aux, h_aux.data(), N * 32, cudaMemcpyHostToDevice));

    int blocks = (N + cfg.gpu_tpb - 1) / cfg.gpu_tpb;
    double ns = gpu_bench(N, cfg.passes, cfg.warmup, cfg.gpu_tpb, [&]() {
        cmp_pipeline_schnorr_k<<<blocks, cfg.gpu_tpb>>>(d_priv, d_msg, d_aux, d_ok, N);
    });

    cudaFree(d_priv); cudaFree(d_msg); cudaFree(d_aux); cudaFree(d_ok);
    return ns;
}
#endif

// ============================================================================
// JSON report writer
// ============================================================================
static void write_json_report(const char* path,
                               const std::vector<CompareEntry>& entries,
                               const cudaDeviceProp& prop,
                               int clock_mhz) {
    FILE* f = std::fopen(path, "w");
    if (!f) {
        std::fprintf(stderr, "Failed to open %s for writing\n", path);
        return;
    }

    std::fprintf(f, "{\n");
    std::fprintf(f, "  \"benchmark\": \"bench_compare\",\n");
    std::fprintf(f, "  \"version\": \"%s\",\n", SECP256K1_VERSION_STRING);
    std::fprintf(f, "  \"gpu\": {\n");
    std::fprintf(f, "    \"name\": \"%s\",\n", prop.name);
    std::fprintf(f, "    \"compute_capability\": \"%d.%d\",\n", prop.major, prop.minor);
    std::fprintf(f, "    \"sm_count\": %d,\n", prop.multiProcessorCount);
    std::fprintf(f, "    \"clock_mhz\": %d,\n", clock_mhz);
    std::fprintf(f, "    \"memory_mb\": %zu\n", prop.totalGlobalMem / (1024 * 1024));
    std::fprintf(f, "  },\n");
    std::fprintf(f, "  \"results\": [\n");

    for (size_t i = 0; i < entries.size(); ++i) {
        auto& e = entries[i];
        std::fprintf(f, "    {\n");
        std::fprintf(f, "      \"section\": \"%s\",\n", e.section);
        std::fprintf(f, "      \"name\": \"%s\",\n", e.name);
        std::fprintf(f, "      \"cpu_ns\": %.2f,\n", e.cpu_ns);
        std::fprintf(f, "      \"gpu_ns\": %.2f,\n", e.gpu_ns);
        std::fprintf(f, "      \"cpu_n\": %d,\n", e.cpu_n);
        std::fprintf(f, "      \"gpu_n\": %d,\n", e.gpu_n);
        std::fprintf(f, "      \"ratio\": %.2f\n", e.ratio);
        std::fprintf(f, "    }%s\n", (i + 1 < entries.size()) ? "," : "");
    }

    std::fprintf(f, "  ]\n");
    std::fprintf(f, "}\n");
    std::fclose(f);
}

// ============================================================================
// Formatting helpers
// ============================================================================
static void format_time(double ns, char* buf, size_t buflen) {
    if (ns >= 1e6)
        std::snprintf(buf, buflen, "%8.2f ms", ns / 1e6);
    else if (ns >= 1e3)
        std::snprintf(buf, buflen, "%8.2f us", ns / 1e3);
    else
        std::snprintf(buf, buflen, "%8.1f ns", ns);
}

static void print_entry(const CompareEntry& e) {
    char cpu_str[32], gpu_str[32];
    format_time(e.cpu_ns, cpu_str, sizeof(cpu_str));
    format_time(e.gpu_ns, gpu_str, sizeof(gpu_str));

    const char* winner = (e.ratio > 1.0) ? "GPU" : "CPU";
    double factor = (e.ratio > 1.0) ? e.ratio : (1.0 / e.ratio);

    std::printf("  %-28s | %s | %s | %5.1fx %s\n",
                e.name, cpu_str, gpu_str, factor, winner);
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    CompareConfig cfg;

    // Parse CLI arguments
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--quick") == 0) {
            cfg.quick = true;
            cfg.cpu_n = 1000;
            cfg.gpu_n = 1 << 12;  // 4K
            cfg.passes = 5;
            cfg.warmup = 1;
        } else if (std::strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
            cfg.json_path = argv[++i];
        } else if (std::strcmp(argv[i], "--cpu-n") == 0 && i + 1 < argc) {
            cfg.cpu_n = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--gpu-n") == 0 && i + 1 < argc) {
            cfg.gpu_n = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--passes") == 0 && i + 1 < argc) {
            cfg.passes = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--help") == 0) {
            std::printf("Usage: bench_compare [--quick] [--json <file>] "
                        "[--cpu-n N] [--gpu-n N] [--passes N]\n");
            return 0;
        }
    }

    // GPU device info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    CUDA_CHECK(cudaSetDevice(0));
    int clockKhz = 0;
    cudaDeviceGetAttribute(&clockKhz, cudaDevAttrClockRate, 0);

    // Header
    std::printf("================================================================\n");
    std::printf("  UltrafastSecp256k1 -- CPU vs GPU Benchmark Comparison\n");
    std::printf("================================================================\n");
    std::printf("  Library:  %s\n", SECP256K1_VERSION_STRING);
    std::printf("  GPU:      %s (SM %d.%d, %d SMs, %d MHz)\n",
                prop.name, prop.major, prop.minor,
                prop.multiProcessorCount, clockKhz / 1000);
    std::printf("  VRAM:     %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
    std::printf("  CPU N:    %d ops/pass    GPU N: %d ops/batch\n", cfg.cpu_n, cfg.gpu_n);
    std::printf("  Passes:   %d (median)    Warmup: %d\n", cfg.passes, cfg.warmup);
    std::printf("================================================================\n\n");

    std::vector<CompareEntry> results;

    auto add_result = [&](const char* section, const char* name,
                           double cpu_ns, double gpu_ns, int cn, int gn) {
        CompareEntry e;
        std::snprintf(e.name, sizeof(e.name), "%s", name);
        std::snprintf(e.section, sizeof(e.section), "%s", section);
        e.cpu_ns = cpu_ns;
        e.gpu_ns = gpu_ns;
        e.cpu_n = cn;
        e.gpu_n = gn;
        e.ratio = cpu_ns / gpu_ns;
        results.push_back(e);
    };

    // ================================================================
    // Section 1: Field Arithmetic
    // ================================================================
    std::printf("=== Field Arithmetic ===\n");
    std::printf("  %-28s | %-12s | %-12s | %s\n", "Operation", "CPU", "GPU", "Winner");
    std::printf("  %-28s-+-%-12s-+-%-12s-+-%s\n", "----------------------------",
                "------------", "------------", "-----------");

    {
        double cpu = cpu_field_mul(cfg);
        double gpu = gpu_field_mul(cfg);
        add_result("Field", "mul", cpu, gpu, cfg.cpu_n, cfg.gpu_n);
        print_entry(results.back());
    }
    {
        double cpu = cpu_field_sqr(cfg);
        double gpu = gpu_field_sqr(cfg);
        add_result("Field", "sqr", cpu, gpu, cfg.cpu_n, cfg.gpu_n);
        print_entry(results.back());
    }
    {
        double cpu = cpu_field_add(cfg);
        double gpu = gpu_field_add(cfg);
        add_result("Field", "add", cpu, gpu, cfg.cpu_n, cfg.gpu_n);
        print_entry(results.back());
    }
    {
        double cpu = cpu_field_sub(cfg);
        double gpu = gpu_field_sub(cfg);
        add_result("Field", "sub", cpu, gpu, cfg.cpu_n, cfg.gpu_n);
        print_entry(results.back());
    }
    {
        double cpu = cpu_field_inv(cfg);
        double gpu = gpu_field_inv(cfg);
        add_result("Field", "inv", cpu, gpu,
                   std::min(cfg.cpu_n, 1000),
                   std::min(cfg.gpu_n, 1 << 16));
        print_entry(results.back());
    }

    // ================================================================
    // Section 2: Point Operations
    // ================================================================
    std::printf("\n=== Point Operations ===\n");
    std::printf("  %-28s | %-12s | %-12s | %s\n", "Operation", "CPU", "GPU", "Winner");
    std::printf("  %-28s-+-%-12s-+-%-12s-+-%s\n", "----------------------------",
                "------------", "------------", "-----------");

    {
        double cpu = cpu_point_add(cfg);
        double gpu = gpu_point_add(cfg);
        add_result("Point", "add (Jacobian)", cpu, gpu,
                   std::min(cfg.cpu_n, 1000),
                   std::min(cfg.gpu_n, 1 << 18));
        print_entry(results.back());
    }
    {
        double cpu = cpu_point_dbl(cfg);
        double gpu = gpu_point_dbl(cfg);
        add_result("Point", "double", cpu, gpu,
                   std::min(cfg.cpu_n, 1000),
                   std::min(cfg.gpu_n, 1 << 18));
        print_entry(results.back());
    }

    // ================================================================
    // Section 3: Scalar Multiplication
    // ================================================================
    std::printf("\n=== Scalar Multiplication ===\n");
    std::printf("  %-28s | %-12s | %-12s | %s\n", "Operation", "CPU", "GPU", "Winner");
    std::printf("  %-28s-+-%-12s-+-%-12s-+-%s\n", "----------------------------",
                "------------", "------------", "-----------");

    {
        double cpu = cpu_scalar_mul_gen(cfg);
        double gpu = gpu_scalar_mul_gen(cfg);
        add_result("ScalarMul", "k*G (generator)", cpu, gpu,
                   std::min(cfg.cpu_n, 1000),
                   std::min(cfg.gpu_n, 1 << 16));
        print_entry(results.back());
    }

#if !SECP256K1_CUDA_LIMBS_32
    // ================================================================
    // Section 4: ECDSA
    // ================================================================
    std::printf("\n=== ECDSA ===\n");
    std::printf("  %-28s | %-12s | %-12s | %s\n", "Operation", "CPU", "GPU", "Winner");
    std::printf("  %-28s-+-%-12s-+-%-12s-+-%s\n", "----------------------------",
                "------------", "------------", "-----------");

    {
        double cpu = cpu_ecdsa_sign(cfg);
        double gpu = gpu_ecdsa_sign(cfg);
        add_result("ECDSA", "sign (RFC 6979)", cpu, gpu,
                   std::min(cfg.cpu_n, 1000),
                   std::min(cfg.gpu_n, 1 << 14));
        print_entry(results.back());
    }
    {
        double cpu = cpu_ecdsa_verify(cfg);
        double gpu = gpu_ecdsa_verify(cfg);
        add_result("ECDSA", "verify", cpu, gpu,
                   std::min(cfg.cpu_n, 1000),
                   std::min(cfg.gpu_n, 1 << 14));
        print_entry(results.back());
    }

    // ================================================================
    // Section 5: Schnorr/BIP-340
    // ================================================================
    std::printf("\n=== Schnorr/BIP-340 ===\n");
    std::printf("  %-28s | %-12s | %-12s | %s\n", "Operation", "CPU", "GPU", "Winner");
    std::printf("  %-28s-+-%-12s-+-%-12s-+-%s\n", "----------------------------",
                "------------", "------------", "-----------");

    {
        double cpu = cpu_schnorr_sign(cfg);
        double gpu = gpu_schnorr_sign(cfg);
        add_result("Schnorr", "sign", cpu, gpu,
                   std::min(cfg.cpu_n, 1000),
                   std::min(cfg.gpu_n, 1 << 14));
        print_entry(results.back());
    }
    {
        double cpu = cpu_schnorr_verify(cfg);
        double gpu = gpu_schnorr_verify(cfg);
        add_result("Schnorr", "verify", cpu, gpu,
                   std::min(cfg.cpu_n, 1000),
                   std::min(cfg.gpu_n, 1 << 14));
        print_entry(results.back());
    }

    // ================================================================
    // Section 6: Full Pipeline (keygen -> sign -> verify)
    // ================================================================
    std::printf("\n=== Full Pipeline (keygen -> sign -> verify) ===\n");
    std::printf("  %-28s | %-12s | %-12s | %s\n", "Operation", "CPU", "GPU", "Winner");
    std::printf("  %-28s-+-%-12s-+-%-12s-+-%s\n", "----------------------------",
                "------------", "------------", "-----------");

    {
        double cpu = cpu_pipeline_ecdsa(cfg);
        double gpu = gpu_pipeline_ecdsa(cfg);
        add_result("Pipeline", "ECDSA (keygen+sign+verify)", cpu, gpu,
                   std::min(cfg.cpu_n, 500),
                   std::min(cfg.gpu_n, 1 << 14));
        print_entry(results.back());
    }
    {
        double cpu = cpu_pipeline_schnorr(cfg);
        double gpu = gpu_pipeline_schnorr(cfg);
        add_result("Pipeline", "Schnorr (keygen+sign+verify)", cpu, gpu,
                   std::min(cfg.cpu_n, 500),
                   std::min(cfg.gpu_n, 1 << 14));
        print_entry(results.back());
    }
#endif

    // ================================================================
    // Summary Table
    // ================================================================
    std::printf("\n================================================================\n");
    std::printf("  CPU vs GPU Comparison Summary\n");
    std::printf("================================================================\n");
    std::printf("  GPU: %s (SM %d.%d, %d SMs)\n\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount);

    std::printf("  %-28s | %-12s | %-12s | %s\n", "Operation", "CPU (ns/op)", "GPU (ns/op)", "Ratio");
    std::printf("  %-28s-+-%-12s-+-%-12s-+-%s\n", "----------------------------",
                "------------", "------------", "-----------");

    int gpu_wins = 0, cpu_wins = 0;
    for (auto& e : results) {
        char cpu_str[32], gpu_str[32];
        format_time(e.cpu_ns, cpu_str, sizeof(cpu_str));
        format_time(e.gpu_ns, gpu_str, sizeof(gpu_str));

        const char* winner = (e.ratio > 1.0) ? "GPU" : "CPU";
        double factor = (e.ratio > 1.0) ? e.ratio : (1.0 / e.ratio);

        std::printf("  %-28s | %s | %s | %5.1fx %s\n",
                    e.name, cpu_str, gpu_str, factor, winner);

        if (e.ratio > 1.0) gpu_wins++; else cpu_wins++;
    }

    std::printf("  %-28s-+-%-12s-+-%-12s-+-%s\n", "----------------------------",
                "------------", "------------", "-----------");
    std::printf("  GPU wins: %d / %d    CPU wins: %d / %d\n",
                gpu_wins, (int)results.size(), cpu_wins, (int)results.size());
    std::printf("================================================================\n\n");

    // JSON output
    if (!cfg.json_path.empty()) {
        write_json_report(cfg.json_path.c_str(), results, prop, clockKhz / 1000);
        std::printf("JSON report: %s\n", cfg.json_path.c_str());
    }

    std::printf("Benchmark complete.\n");
    return 0;
}
