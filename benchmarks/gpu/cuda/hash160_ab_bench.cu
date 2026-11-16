// ============================================================================
// CUDA hash160 A/B harness for 65-byte (uncompressed) public keys
// ============================================================================
// Times, in ONE binary and ONE run, the two 65-byte hash160 paths of
// src/cuda/include/hash160.cuh on the same device with the same inputs:
//
//   OLD : the generic, length-looping sha256() over a byte buffer, then
//         ripemd160_32(). This is what hash160_pubkey() did for 65-byte keys
//         before sha256_65() existed.
//   NEW : hash160_pubkey(key, 65, out) as shipped -- the length dispatch sends
//         65-byte keys to the registers-only two-block sha256_65().
//
// The length is passed to both kernels as a runtime argument (as the production
// hash160_pubkey_kernel does), so the compiler cannot fold the dispatch away.
// The timed work (global read of the 65-byte key, hash160, global write of 20
// bytes) is identical apart from the SHA-256.
//
// Correctness: both outputs are compared byte-for-byte for every key, and key 0
// is the secp256k1 generator, 04||Gx||Gy, whose hash160 must equal the known
// answer 91b24bf9f5288532960ac687abb035127b1d28a5. Any mismatch exits 1.
//
// Timing uses cudaEvent only (no clock-rate queries). The order of OLD / NEW
// alternates on every rep so drift and clock ramp do not favour one side. No
// CUDA device: the program prints a notice and exits 0 without measuring.
//
// Build (from libs/UltrafastSecp256k1, nvcc 13.x, RTX 5060 Ti = sm_120):
//   nvcc -O3 -std=c++17 -gencode arch=compute_120,code=sm_120 \
//        -I src/cuda/include \
//        benchmarks/gpu/cuda/hash160_ab_bench.cu -o hash160_ab_bench
// Run:  ./hash160_ab_bench [keys=1048576] [iters=20] [reps=11]
//
// It prints measured numbers for the machine it runs on and nothing else; do
// not copy them into docs, comments or commit messages. Repeat the run (the
// project's 5-run protocol) to judge run-to-run variance.
// ============================================================================

#include "hash160.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace secp256k1::cuda;

#define CUDA_CHECK(c) do { cudaError_t e_ = (c); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error '%s' at line %d\n", cudaGetErrorString(e_), __LINE__); \
    std::exit(2); } } while (0)

static constexpr int kKeyLen = 65;

// OLD: generic SHA-256 over a byte buffer + ripemd160_32.
__global__ void k_hash160_old(const uint8_t* __restrict__ keys, uint8_t* __restrict__ out,
                              uint32_t count, uint32_t len) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    uint8_t pk[kKeyLen];
    for (uint32_t i = 0; i < len; ++i) pk[i] = keys[(size_t)tid * len + i];
    uint8_t sha[32], h[20];
    sha256(pk, len, sha);
    ripemd160_32(sha, h);
    for (int i = 0; i < 20; ++i) out[(size_t)tid * 20 + i] = h[i];
}

// NEW: the production hash160_pubkey() dispatch (65 bytes -> sha256_65).
__global__ void k_hash160_new(const uint8_t* __restrict__ keys, uint8_t* __restrict__ out,
                              uint32_t count, uint32_t len) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    uint8_t pk[kKeyLen];
    for (uint32_t i = 0; i < len; ++i) pk[i] = keys[(size_t)tid * len + i];
    uint8_t h[20];
    hash160_pubkey(pk, len, h);
    for (int i = 0; i < 20; ++i) out[(size_t)tid * 20 + i] = h[i];
}

static float time_launch(bool use_new, const uint8_t* d_in, uint8_t* d_out, uint32_t count,
                         uint32_t len, int iters, cudaEvent_t e0, cudaEvent_t e1) {
    const int block = 128;
    const int grid = (int)((count + block - 1) / block);
    CUDA_CHECK(cudaEventRecord(e0));
    for (int it = 0; it < iters; ++it) {
        if (use_new) k_hash160_new<<<grid, block>>>(d_in, d_out, count, len);
        else         k_hash160_old<<<grid, block>>>(d_in, d_out, count, len);
    }
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaGetLastError());
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    return ms;
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

int main(int argc, char** argv) {
    uint32_t keys = 1u << 20;
    int iters = 20;
    int reps = 11;
    if (argc > 1) keys  = (uint32_t)std::strtoul(argv[1], nullptr, 10);
    if (argc > 2) iters = std::atoi(argv[2]);
    if (argc > 3) reps  = std::atoi(argv[3]);
    if (keys == 0 || iters <= 0 || reps <= 0) {
        std::fprintf(stderr, "keys, iters and reps must be positive\n");
        return 2;
    }

    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("hash160_ab_bench: no CUDA device available; nothing measured.\n");
        return 0;
    }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device: %s\n", prop.name);
    std::printf("keys=%u iters=%d reps=%d (median over reps; ns/hash is cudaEvent time / (keys*iters))\n",
                keys, iters, reps);

    // Inputs: key 0 is the generator 04||Gx||Gy (known answer); the rest are
    // pseudo-random 65-byte values with a 0x04 prefix, seeded deterministically.
    static const uint8_t kGx[32] = {
        0x79,0xBE,0x66,0x7E,0xF9,0xDC,0xBB,0xAC,0x55,0xA0,0x62,0x95,0xCE,0x87,0x0B,0x07,
        0x02,0x9B,0xFC,0xDB,0x2D,0xCE,0x28,0xD9,0x59,0xF2,0x81,0x5B,0x16,0xF8,0x17,0x98};
    static const uint8_t kGy[32] = {
        0x48,0x3A,0xDA,0x77,0x26,0xA3,0xC4,0x65,0x5D,0xA4,0xFB,0xFC,0x0E,0x11,0x08,0xA8,
        0xFD,0x17,0xB4,0x48,0xA6,0x85,0x54,0x19,0x9C,0x47,0xD0,0x8F,0xFB,0x10,0xD4,0xB8};
    static const uint8_t kKat[20] = {
        0x91,0xb2,0x4b,0xf9,0xf5,0x28,0x85,0x32,0x96,0x0a,
        0xc6,0x87,0xab,0xb0,0x35,0x12,0x7b,0x1d,0x28,0xa5};

    std::vector<uint8_t> h_keys((size_t)keys * kKeyLen);
    uint64_t rng = 0x243F6A8885A308D3ULL;
    for (auto& b : h_keys) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        b = (uint8_t)(rng >> 24);
    }
    for (uint32_t i = 0; i < keys; ++i) h_keys[(size_t)i * kKeyLen] = 4;
    h_keys[0] = 4;
    std::memcpy(&h_keys[1], kGx, 32);
    std::memcpy(&h_keys[33], kGy, 32);

    uint8_t *d_in = nullptr, *d_old = nullptr, *d_new = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  h_keys.size()));
    CUDA_CHECK(cudaMalloc(&d_old, (size_t)keys * 20));
    CUDA_CHECK(cudaMalloc(&d_new, (size_t)keys * 20));
    CUDA_CHECK(cudaMemcpy(d_in, h_keys.data(), h_keys.size(), cudaMemcpyHostToDevice));
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    // Correctness first: OLD and NEW must agree on every key, and NEW must hit
    // the known answer on the generator.
    time_launch(false, d_in, d_old, keys, kKeyLen, 1, e0, e1);
    time_launch(true,  d_in, d_new, keys, kKeyLen, 1, e0, e1);
    std::vector<uint8_t> h_old((size_t)keys * 20), h_new((size_t)keys * 20);
    CUDA_CHECK(cudaMemcpy(h_old.data(), d_old, h_old.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_new.data(), d_new, h_new.size(), cudaMemcpyDeviceToHost));
    bool agree = (h_old == h_new);
    const bool kat_ok = std::memcmp(h_new.data(), kKat, 20) == 0 && std::memcmp(h_old.data(), kKat, 20) == 0;
    if (!agree) {
        for (uint32_t i = 0; i < keys; ++i) {
            if (std::memcmp(&h_old[(size_t)i * 20], &h_new[(size_t)i * 20], 20) != 0) {
                std::fprintf(stderr, "MISMATCH first at key %u\n", i);
                break;
            }
        }
    }
    if (!kat_ok) std::fprintf(stderr, "known answer hash160(04||Gx||Gy) not reproduced\n");

    // Timing: alternate which side goes first on each rep.
    std::vector<double> t_old, t_new;
    for (int r = 0; r < reps; ++r) {
        float a, b;
        if (r % 2 == 0) {
            a = time_launch(false, d_in, d_old, keys, kKeyLen, iters, e0, e1);
            b = time_launch(true,  d_in, d_new, keys, kKeyLen, iters, e0, e1);
        } else {
            b = time_launch(true,  d_in, d_new, keys, kKeyLen, iters, e0, e1);
            a = time_launch(false, d_in, d_old, keys, kKeyLen, iters, e0, e1);
        }
        const double denom = (double)keys * iters;
        t_old.push_back(a * 1.0e6 / denom);   // ms -> ns
        t_new.push_back(b * 1.0e6 / denom);
    }
    const double mo = median(t_old), mn = median(t_new);
    std::printf("len=%d  old=%.4f ns/hash  new=%.4f ns/hash  old/new=%.3f  agree=%s  kat=%s\n",
                kKeyLen, mo, mn, mn > 0 ? mo / mn : 0.0, agree ? "yes" : "NO", kat_ok ? "yes" : "NO");
    std::printf("  old min..max: %.4f..%.4f   new min..max: %.4f..%.4f\n",
                *std::min_element(t_old.begin(), t_old.end()), *std::max_element(t_old.begin(), t_old.end()),
                *std::min_element(t_new.begin(), t_new.end()), *std::max_element(t_new.begin(), t_new.end()));

    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    cudaFree(d_in); cudaFree(d_old); cudaFree(d_new);

    const bool ok = agree && kat_ok;
    std::printf("%s\n", ok ? "RESULT: old and new agree on every key"
                           : "RESULT: MISMATCH between old and new");
    return ok ? 0 : 1;
}
