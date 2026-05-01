// ============================================================================
// bench_snark_witness -- CPU vs GPU ECDSA SNARK Witness Benchmark
// ============================================================================
//
// Benchmarks ufsecp_zk_ecdsa_snark_witness (CPU internal API) and
// the ecdsa_snark_witness_batch_kernel (GPU) side-by-side.
//
// Follows the same style as bench_bip352 / bench_compare.
// Compiled as part of secp256k1_cuda_lib / fastsecp256k1 builds.
//
// Usage:
//   bench_snark_witness                     # default N=65536
//   bench_snark_witness --batch 4096        # small N for quick test
//   bench_snark_witness --passes 11         # number of passes
// ============================================================================

// ---- GPU headers ----
#include "secp256k1.cuh"
#include "ecdsa.cuh"

// ---- CPU headers ----
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/zk.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>
#include <string>
#include <array>

using namespace secp256k1::cuda;
namespace cpu_fast = secp256k1::fast;
namespace cpu_zk   = secp256k1::zk;

// ---- Forward-declare the kernel (defined in secp256k1.cu, secp256k1::cuda ns) --
namespace secp256k1 { namespace cuda {
__global__ void ecdsa_snark_witness_batch_kernel(
    const uint8_t* __restrict__ msg_hashes32,
    const JacobianPoint* __restrict__ pubkeys,
    const ECDSASignatureGPU* __restrict__ sigs,
    EcdsaSnarkWitnessFlat* __restrict__ out,
    int count);
} } // namespace secp256k1::cuda

// Inline decompression kernel using the inline point_from_compressed device fn
__global__ static void decompress_points_kernel(
    const uint8_t* __restrict__ comp,
    secp256k1::cuda::JacobianPoint* __restrict__ pts,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    secp256k1::cuda::point_from_compressed(comp + idx * 33, &pts[idx]);
}

// ---- CUDA error check -------------------------------------------------------
#define CUDA_BAIL(expr)                                                        \
    do {                                                                       \
        cudaError_t _e = (expr);                                               \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error at line %d: %s\n",               \
                         __LINE__, cudaGetErrorString(_e));                    \
            return 1;                                                          \
        }                                                                      \
    } while (0)

// ---- Timing helpers ---------------------------------------------------------
static double median_d(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2 == 0) ? (v[n/2-1] + v[n/2]) / 2.0 : v[n/2];
}

// ---- Deterministic scalar from index ----------------------------------------
static cpu_fast::Scalar make_privkey(int idx) {
    // simple deterministic key: ((idx+1) * 0x0102030405060708) mod n
    // guaranteed non-zero for idx < ~2^32
    uint64_t lo = (uint64_t)(idx + 1) * 0x0102030405060708ULL;
    cpu_fast::Scalar::limbs_type lmbs = {lo, (uint64_t)(idx + 1), 0ULL, 0ULL};
    return cpu_fast::Scalar::from_limbs(lmbs);
}

// ---- main ------------------------------------------------------------------
int main(int argc, char** argv) {
    int bench_n = 65536;
    int warmup  = 3;
    int passes  = 11;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--batch" || arg == "-n") && i + 1 < argc)
            bench_n = std::atoi(argv[++i]);
        else if (arg == "--passes" && i + 1 < argc)
            passes = std::atoi(argv[++i]);
    }

    // -----------------------------------------------------------------------
    // Header
    // -----------------------------------------------------------------------
    {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, 0);
        std::printf("============================================================\n");
        std::printf("  ECDSA SNARK Witness Benchmark (eprint 2025/695)\n");
        std::printf("  CPU: secp256k1::zk::ecdsa_snark_witness  (single-thread)\n");
        std::printf("  GPU: ecdsa_snark_witness_batch_kernel     (CUDA)\n");
        std::printf("============================================================\n");
        std::printf("  GPU: %s (%d SMs, %d MHz)\n",
                    prop.name, prop.multiProcessorCount,
                    (int)(prop.clockRate / 1000));
        std::printf("  N = %d, %d passes (median)\n\n", bench_n, passes);
    }

    // -----------------------------------------------------------------------
    // Generate deterministic test data
    // -----------------------------------------------------------------------
    std::printf("Generating %d deterministic key/sig pairs...\n", bench_n);

    // MSG: all pairs share one deterministic 32-byte message
    static const std::array<uint8_t,32> MSG = {
        0x9f,0x86,0xd0,0x81,0x88,0x4c,0x7d,0x65,
        0x9a,0x2f,0xea,0xa0,0xc5,0x5a,0xd0,0x15,
        0xa3,0xbf,0x4f,0x1b,0x2b,0x0b,0x82,0x2c,
        0xd1,0x5d,0x6c,0x15,0xb0,0xf0,0x0a,0x08
    };

    // Host-side flat arrays for GPU
    std::vector<uint8_t>              h_msgs(bench_n * 32);
    std::vector<uint8_t>              h_pubs33(bench_n * 33);
    std::vector<ECDSASignatureGPU>    h_sigs_gpu(bench_n);
    // CPU-side structures for CPU benchmark
    std::vector<cpu_fast::Point>      h_pubs_cpu(bench_n);
    std::vector<cpu_fast::Scalar>     h_sig_r(bench_n);
    std::vector<cpu_fast::Scalar>     h_sig_s(bench_n);

    for (int i = 0; i < bench_n; ++i) {
        cpu_fast::Scalar priv = make_privkey(i);
        cpu_fast::Point  pub  = cpu_fast::Point::generator().scalar_mul(priv);
        h_pubs_cpu[i] = pub;

        // Sign
        auto esig = secp256k1::ecdsa_sign(MSG, priv);
        h_sig_r[i] = esig.r;
        h_sig_s[i] = esig.s;

        // Compressed pubkey for GPU
        auto comp = pub.to_compressed();
        std::memcpy(h_pubs33.data() + i * 33, comp.data(), 33);

        // Message bytes (same for all)
        std::memcpy(h_msgs.data() + i * 32, MSG.data(), 32);

        // GPU sig struct — ScalarData uses little-endian uint64_t limbs
        auto rl = esig.r.limbs();
        auto sl = esig.s.limbs();
        std::memcpy(&h_sigs_gpu[i].r, rl.data(), 32);
        std::memcpy(&h_sigs_gpu[i].s, sl.data(), 32);
    }
    std::printf("Done.\n\n");

    // -----------------------------------------------------------------------
    // CPU warmup + benchmark
    // -----------------------------------------------------------------------
    std::printf("--- CPU (secp256k1::zk::ecdsa_snark_witness, single-threaded) ---\n");

    // Warmup
    for (int w = 0; w < warmup && w < bench_n; ++w)
        (void)cpu_zk::ecdsa_snark_witness(MSG, h_pubs_cpu[w], h_sig_r[w], h_sig_s[w]);

    std::vector<double> cpu_times;
    cpu_zk::EcdsaSnarkWitness cpu_ref_w{};
    for (int p = 0; p < passes; ++p) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < bench_n; ++i) {
            auto w = cpu_zk::ecdsa_snark_witness(MSG, h_pubs_cpu[i], h_sig_r[i], h_sig_s[i]);
            if (i == 0 && p == 0) cpu_ref_w = w;  // save first result for validation
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
        cpu_times.push_back(ms);
        std::printf("  pass %2d: %8.3f ms\n", p+1, ms);
    }
    double cpu_ms    = median_d(cpu_times);
    double cpu_ns_op = cpu_ms * 1e6 / bench_n;
    double cpu_mops  = bench_n / (cpu_ms * 1e-3) / 1e6;
    std::printf("\n  CPU: %.3f ms / %d ops = %.1f ns/op  (%.4f M/s)\n\n",
                cpu_ms, bench_n, cpu_ns_op, cpu_mops);

    // -----------------------------------------------------------------------
    // Upload to GPU
    // -----------------------------------------------------------------------
    uint8_t*               d_msgs   = nullptr;
    uint8_t*               d_pubs33 = nullptr;
    JacobianPoint*         d_pubs   = nullptr;
    ECDSASignatureGPU*     d_sigs   = nullptr;
    EcdsaSnarkWitnessFlat* d_out    = nullptr;
    std::vector<EcdsaSnarkWitnessFlat> h_out(bench_n);

    CUDA_BAIL(cudaMalloc(&d_msgs,   bench_n * 32));
    CUDA_BAIL(cudaMalloc(&d_pubs33, bench_n * 33));
    CUDA_BAIL(cudaMalloc(&d_pubs,   bench_n * sizeof(JacobianPoint)));
    CUDA_BAIL(cudaMalloc(&d_sigs,   bench_n * sizeof(ECDSASignatureGPU)));
    CUDA_BAIL(cudaMalloc(&d_out,    bench_n * sizeof(EcdsaSnarkWitnessFlat)));

    CUDA_BAIL(cudaMemcpy(d_msgs,   h_msgs.data(),    bench_n * 32,                         cudaMemcpyHostToDevice));
    CUDA_BAIL(cudaMemcpy(d_pubs33, h_pubs33.data(),  bench_n * 33,                         cudaMemcpyHostToDevice));
    CUDA_BAIL(cudaMemcpy(d_sigs,   h_sigs_gpu.data(),bench_n * sizeof(ECDSASignatureGPU),  cudaMemcpyHostToDevice));

    // Decompress pubkeys once (not timed)
    int threads = 128, blocks = (bench_n + threads - 1) / threads;
    decompress_points_kernel<<<blocks, threads>>>(d_pubs33, d_pubs, bench_n);
    CUDA_BAIL(cudaDeviceSynchronize());

    // -----------------------------------------------------------------------
    // GPU warmup + benchmark
    // -----------------------------------------------------------------------
    std::printf("--- GPU (ecdsa_snark_witness_batch_kernel, CUDA) ---\n");

    cudaEvent_t ev_start, ev_stop;
    CUDA_BAIL(cudaEventCreate(&ev_start));
    CUDA_BAIL(cudaEventCreate(&ev_stop));

    // Warmup
    for (int w = 0; w < warmup; ++w) {
        secp256k1::cuda::ecdsa_snark_witness_batch_kernel<<<blocks, threads>>>(
            d_msgs, d_pubs, d_sigs, d_out, bench_n);
        CUDA_BAIL(cudaDeviceSynchronize());
    }

    std::vector<double> gpu_times;
    for (int p = 0; p < passes; ++p) {
        CUDA_BAIL(cudaEventRecord(ev_start, 0));
        secp256k1::cuda::ecdsa_snark_witness_batch_kernel<<<blocks, threads>>>(
            d_msgs, d_pubs, d_sigs, d_out, bench_n);
        CUDA_BAIL(cudaEventRecord(ev_stop, 0));
        CUDA_BAIL(cudaEventSynchronize(ev_stop));
        float ms_f = 0.0f;
        CUDA_BAIL(cudaEventElapsedTime(&ms_f, ev_start, ev_stop));
        gpu_times.push_back((double)ms_f);
        std::printf("  pass %2d: %8.3f ms\n", p+1, (double)ms_f);
    }

    // Download first result for validation
    CUDA_BAIL(cudaMemcpy(h_out.data(), d_out,
                         bench_n * sizeof(EcdsaSnarkWitnessFlat),
                         cudaMemcpyDeviceToHost));

    double gpu_ms    = median_d(gpu_times);
    double gpu_ns_op = gpu_ms * 1e6 / bench_n;
    double gpu_mops  = bench_n / (gpu_ms * 1e-3) / 1e6;
    double ratio     = cpu_ns_op / gpu_ns_op;
    std::printf("\n  GPU: %.3f ms / %d ops = %.1f ns/op  (%.4f M/s)\n\n",
                gpu_ms, bench_n, gpu_ns_op, gpu_mops);

    // -----------------------------------------------------------------------
    // Validation: GPU result[0] s_inv bytes == CPU result s_inv bytes
    // -----------------------------------------------------------------------
    bool match = (std::memcmp(h_out[0].s_inv, cpu_ref_w.bytes_s_inv.data(), 32) == 0)
              && (h_out[0].valid == (cpu_ref_w.valid ? 1 : 0));

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    std::printf("=== Summary ===\n");
    std::printf("  CPU (single-thread):    %10.1f ns/op  (%.4f M/s)\n",
                cpu_ns_op, cpu_mops);
    std::printf("  GPU (batch kernel):     %10.1f ns/op  (%.4f M/s)  (%.2fx vs CPU)\n",
                gpu_ns_op, gpu_mops, ratio);
    std::printf("  Witness size:           760 bytes/witness\n");
    std::printf("  Validation: %s\n", match ? "[OK] MATCH" : "[FAIL] MISMATCH");
    std::printf("============================================================\n");

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaFree(d_msgs); cudaFree(d_pubs33); cudaFree(d_pubs);
    cudaFree(d_sigs); cudaFree(d_out);
    return match ? 0 : 1;
}


