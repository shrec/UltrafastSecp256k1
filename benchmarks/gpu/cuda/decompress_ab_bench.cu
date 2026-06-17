// GPU Pubkey Decompress A/B Experiment
// Path A: GPU verify with JacobianPoint (current baseline)
// Path B: GPU decompress 33-byte pubkey + verify in registers
//
// Build: add to src/cuda/CMakeLists.txt, links against secp256k1_cuda_lib

#include "secp256k1.cuh"
#include "ecdsa.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

// === Forward-declare batch kernels (defined in secp256k1.cu, same namespace) ===
// These are NOT in any .cuh header, so we must declare them ourselves.
namespace secp256k1 { namespace cuda {

__global__ void ecdsa_verify_batch_kernel(
    const uint8_t* msg_hashes,
    const JacobianPoint* public_keys,
    const ECDSASignatureGPU* sigs,
    bool* results, int count);

__global__ void ecdsa_sign_batch_kernel(
    const uint8_t* msg_hashes,
    const Scalar* private_keys,
    ECDSASignatureGPU* sigs,
    bool* results, int count);

// generator_mul_batch_kernel is declared in secp256k1.cuh — no need to forward-declare

__global__ void batch_jacobian_to_compressed_kernel(
    const JacobianPoint* points, uint8_t* out33, int count);

// === GPU-side: decompress 33-byte compressed pubkey in registers ===
__device__ inline bool gpu_decompress_33(const uint8_t pk33[33],
                                          FieldElement* ox, FieldElement* oy) {
    uint8_t prefix = pk33[0];
    if (prefix != 0x02 && prefix != 0x03) return false;
    FieldElement x;
    if (!field_from_bytes_strict(pk33 + 1, &x)) return false;
    FieldElement x2, x3, y2;
    field_sqr(&x, &x2);
    field_mul(&x2, &x, &x3);
    FieldElement seven; field_from_uint64(7, &seven);
    field_add(&x3, &seven, &y2);
    FieldElement y;
    field_sqrt(&y2, &y);
    FieldElement yc; field_sqr(&y, &yc);
    if (!field_eq(&yc, &y2)) return false;
    uint8_t yb[32]; field_to_bytes(&y, yb);
    bool odd = (yb[31] & 1) != 0;
    bool want = (prefix == 0x03);
    if (odd != want) {
        FieldElement zero; field_set_zero(&zero);
        field_sub(&zero, &y, &y);
    }
    *ox = x; *oy = y;
    return true;
}

// === Path B kernel: decompress + verify in one shot ===
__global__ void ecdsa_verify_decompress_kernel(
    const uint8_t* msgs, const uint8_t* pubkeys33,
    const ECDSASignatureGPU* sigs, bool* results, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const uint8_t* pk = pubkeys33 + (size_t)i * 33;
    FieldElement px, py;
    if (!gpu_decompress_33(pk, &px, &py)) { results[i] = false; return; }
    JacobianPoint pub;
    pub.x = px; pub.y = py;
    field_set_one(&pub.z);
    pub.infinity = false;
    results[i] = ecdsa_verify(msgs + (size_t)i * 32, &pub, &sigs[i]);
}

}} // namespace secp256k1::cuda

using namespace secp256k1::cuda;

#define CUDA_CHECK(c) do { cudaError_t e = c; if (e != cudaSuccess) { \
    std::fprintf(stderr,"CUDA err %s at %d\n",cudaGetErrorString(e),__LINE__); \
    std::exit(1); } } while(0)

int main(int argc, char** argv) {
    int batch = 16384;
    int warmup = 3, measure = 7;
    if (argc > 1) batch = std::atoi(argv[1]);

    int dev; cudaGetDevice(&dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    std::printf("GPU: %s | SM=%d | %d MHz | Batch=%d\n\n",
                prop.name, prop.multiProcessorCount, prop.clockRate/1000, batch);

    // Generate test vectors
    std::mt19937_64 rng(0xDEC0DEC0);
    std::vector<uint8_t>  h_msgs(batch * 32);
    std::vector<Scalar>   h_sk(batch);
    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < 32; j++) h_msgs[i*32+j] = (uint8_t)(rng() & 0xFF);
        h_sk[i].limbs[0]=rng(); h_sk[i].limbs[1]=rng();
        h_sk[i].limbs[2]=rng(); h_sk[i].limbs[3]=rng() & 0x7FFFFFFFFFFFFFFFull;
        if (h_sk[i].limbs[0]==0&&h_sk[i].limbs[1]==0&&
            h_sk[i].limbs[2]==0&&h_sk[i].limbs[3]==0)
            h_sk[i].limbs[0]=1;
    }

    // Alloc GPU buffers
    uint8_t *d_msgs; Scalar *d_sk; ECDSASignatureGPU *d_sigs;
    bool *d_ok; JacobianPoint *d_pubs;
    CUDA_CHECK(cudaMalloc(&d_msgs, batch*32));
    CUDA_CHECK(cudaMalloc(&d_sk, batch*sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_sigs, batch*sizeof(ECDSASignatureGPU)));
    CUDA_CHECK(cudaMalloc(&d_ok, batch*sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_pubs, batch*sizeof(JacobianPoint)));
    CUDA_CHECK(cudaMemcpy(d_msgs, h_msgs.data(), batch*32, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sk, h_sk.data(), batch*sizeof(Scalar), cudaMemcpyHostToDevice));

    int blocks = (batch + 127) / 128;
    // Sign on GPU
    ecdsa_sign_batch_kernel<<<blocks,128>>>(d_msgs, d_sk, d_sigs, d_ok, batch);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    // Gen pubkeys: Q = sk * G  (declared in secp256k1.cuh)
    generator_mul_batch_kernel<<<blocks,128>>>(d_sk, d_pubs, batch);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());

    // Read back sigs + pubkeys
    std::vector<ECDSASignatureGPU> h_sigs(batch);
    std::vector<JacobianPoint> h_pubs(batch);
    CUDA_CHECK(cudaMemcpy(h_sigs.data(), d_sigs,
        batch*sizeof(ECDSASignatureGPU), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pubs.data(), d_pubs,
        batch*sizeof(JacobianPoint), cudaMemcpyDeviceToHost));

    // Convert Jacobian -> 33-byte compressed ON GPU (proper affine normalization)
    uint8_t *d_pk33;
    CUDA_CHECK(cudaMalloc(&d_pk33, batch*33));
    batch_jacobian_to_compressed_kernel<<<blocks,128>>>(d_pubs, d_pk33, batch);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());

    bool *d_res;
    CUDA_CHECK(cudaMalloc(&d_res, batch*sizeof(bool)));

    // ============ PATH A: GPU verify with JacobianPoint ============
    std::printf("============================================\n");
    std::printf("  PATH A: GPU verify (JacobianPoint input)\n");
    std::printf("  (Current: pubkeys pre-decompressed on CPU)\n");
    std::printf("============================================\n");

    std::vector<double> ga;
    for (int pass = 0; pass < warmup + measure; pass++) {
        cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        ecdsa_verify_batch_kernel<<<blocks,128>>>(d_msgs,d_pubs,d_sigs,d_res,batch);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms, s, e);
        cudaEventDestroy(s); cudaEventDestroy(e);
        if (pass >= warmup) ga.push_back(ms * 1e6);
    }
    std::sort(ga.begin(), ga.end());
    double a_ns = ga[ga.size()/2] / batch;
    std::printf("  GPU verify (Jacobian):  %8.1f ns/op  (%.2f M/s)\n",
                a_ns, 1000.0/a_ns);
    std::vector<uint8_t> ra(batch);
    CUDA_CHECK(cudaMemcpy(ra.data(), d_res, batch, cudaMemcpyDeviceToHost));
    int oa = 0; for (int i=0; i<batch; i++) if (ra[i]) oa++;
    std::printf("  Valid: %d/%d\n", oa, batch);

    // ============ PATH B: GPU decompress + verify ============
    std::printf("\n============================================\n");
    std::printf("  PATH B: GPU decompress + verify (new)\n");
    std::printf("  (33-byte in, decompress in registers)\n");
    std::printf("============================================\n");

    std::vector<double> gb;
    for (int pass = 0; pass < warmup + measure; pass++) {
        cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        ecdsa_verify_decompress_kernel<<<blocks,128>>>(d_msgs,d_pk33,d_sigs,d_res,batch);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms, s, e);
        cudaEventDestroy(s); cudaEventDestroy(e);
        if (pass >= warmup) gb.push_back(ms * 1e6);
    }
    std::sort(gb.begin(), gb.end());
    double b_ns = gb[gb.size()/2] / batch;
    std::printf("  GPU decomp+verify:      %8.1f ns/op  (%.2f M/s)\n",
                b_ns, 1000.0/b_ns);
    std::vector<uint8_t> rb(batch);
    CUDA_CHECK(cudaMemcpy(rb.data(), d_res, batch, cudaMemcpyDeviceToHost));
    int ob = 0; for (int i=0; i<batch; i++) if (rb[i]) ob++;
    std::printf("  Valid: %d/%d\n", ob, batch);

    // ============ COMPARISON ============
    double ovh = b_ns - a_ns;
    double pct = (ovh / a_ns) * 100.0;
    std::printf("\n============================================\n");
    std::printf("  COMPARISON  (RTX 5060 Ti, batch=%d)\n", batch);
    std::printf("============================================\n");
    std::printf("  Path A (Jacobian verify):   %8.1f ns/op  (%.2f M/s)\n", a_ns, 1000.0/a_ns);
    std::printf("  Path B (decompress+verify): %8.1f ns/op  (%.2f M/s)\n", b_ns, 1000.0/b_ns);
    std::printf("  GPU decompress overhead:    %8.1f ns  (%+.1f%%)\n", ovh, pct);
    std::printf("\n");
    std::printf("  DATA SIZES:\n");
    std::printf("    JacobianPoint: %zu bytes\n", sizeof(JacobianPoint));
    std::printf("    Compressed:    33 bytes  (%.1fx less PCIe data)\n",
                (double)sizeof(JacobianPoint)/33.0);
    std::printf("  CPU SIDE SAVINGS:\n");
    std::printf("    - No sqrt(x^3+7) + parity on CPU\n");
    std::printf("    - No buffer alloc for decompressed points\n");
    std::printf("    - No sequential CPU copy to JacobianPoint array\n");
    std::printf("============================================\n");

    // Cleanup
    CUDA_CHECK(cudaFree(d_msgs));
    CUDA_CHECK(cudaMemset(d_sk, 0, batch*sizeof(Scalar)));
    CUDA_CHECK(cudaFree(d_sk)); CUDA_CHECK(cudaFree(d_sigs));
    CUDA_CHECK(cudaFree(d_ok)); CUDA_CHECK(cudaFree(d_pubs));
    CUDA_CHECK(cudaFree(d_pk33)); CUDA_CHECK(cudaFree(d_res));
    return 0;
}
