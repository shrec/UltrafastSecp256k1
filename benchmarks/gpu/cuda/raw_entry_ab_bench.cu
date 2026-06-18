// GPU Raw Entry ECDSA Verify — Zero-CPU-Prep Benchmark
// Path A: pre-expanded pubkey + compact signature → GPU verify
// Path B: Raw bytes → GPU does ALL prep + verify in registers
//
// This is the ultimate "no CPU work" approach:
//   CPU only sends raw buffers: 33B pubkey + 64B sig + 32B msg → 129B/entry
//   GPU decompresses pubkey, parses signature, normalizes low-S, verifies

#include "secp256k1.cuh"
#include "ecdsa.cuh"
#include "ct/ct_scalar.cuh"  // for ct::scalar_normalize_low_s
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

// Forward-declare kernels from secp256k1.cu (same namespace)
namespace secp256k1 { namespace cuda {

__global__ void ecdsa_verify_batch_kernel(
    const uint8_t* msg_hashes, const JacobianPoint* public_keys,
    const uint8_t* sigs64, bool* results, int count);

__global__ void ecdsa_sign_batch_kernel(
    const uint8_t* msg_hashes, const Scalar* private_keys,
    ECDSASignatureGPU* sigs, bool* results, int count);

__global__ void batch_jacobian_to_compressed_kernel(
    const JacobianPoint* points, uint8_t* out33, int count);

// Opaque format is little-endian (4 × uint64 LE)
// GPU scalar_from_bytes is big-endian, so we write an LE parser
__device__ inline bool opaque_scalar_parse_strict_nonzero(
    const uint8_t* opaque32, Scalar* out)
{
    // Read 4 limbs in little-endian
    uint64_t limbs[4] = {0,0,0,0};
    for (int i = 0; i < 4; i++) {
        uint64_t v = 0;
        for (int j = 0; j < 8; j++)
            v |= ((uint64_t)opaque32[i*8 + j]) << (j * 8);
        limbs[i] = v;
    }

    // Check zero
    if ((limbs[0] | limbs[1] | limbs[2] | limbs[3]) == 0) return false;

    // Check < ORDER
    // ORDER limbs = {0xBFD25E8CD0364141, 0xBAAEDCE6AF48A03B, 0xFFFFFFFFFFFFFFFE, 0xFFFFFFFFFFFFFFFF}
    // Compare from most significant limb (index 3) down
    for (int i = 3; i >= 0; i--) {
        if (limbs[i] > ORDER[i]) return false;  // r >= n → invalid
        if (limbs[i] < ORDER[i]) break;         // r < n → OK
        // equal, continue to next limb
    }

    out->limbs[0] = limbs[0];
    out->limbs[1] = limbs[1];
    out->limbs[2] = limbs[2];
    out->limbs[3] = limbs[3];
    return true;
}

// GPU-side: decompress pubkey (33B → affine x,y)
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

// === RAW ENTRY KERNEL (Path B): decompress + parse + normalize + verify ===
__global__ void ecdsa_raw_entry_verify_kernel(
    const uint8_t* msgs,          // [count * 32]
    const uint8_t* pubkeys33,     // [count * 33]
    const uint8_t* opaque_sigs64, // [count * 64]
    bool* results, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const uint8_t* pk  = pubkeys33 + (size_t)i * 33;
    const uint8_t* sig = opaque_sigs64 + (size_t)i * 64;
    const uint8_t* msg = msgs + (size_t)i * 32;

    // 1. Decompress pubkey
    FieldElement px, py;
    if (!gpu_decompress_33(pk, &px, &py)) { results[i] = false; return; }

    // 2. Parse signature (opaque LE format → r,s scalars)
    Scalar r, s;
    if (!opaque_scalar_parse_strict_nonzero(sig, &r) ||
        !opaque_scalar_parse_strict_nonzero(sig + 32, &s)) {
        results[i] = false; return;
    }

    // 3. Low-S normalization (BIP-62)
    ct::scalar_normalize_low_s(&s);

    // 4. Build JacobianPoint + ECDSASignatureGPU + verify
    JacobianPoint pub;
    pub.x = px; pub.y = py;
    field_set_one(&pub.z);
    pub.infinity = false;

    ECDSASignatureGPU sig_gpu;
    sig_gpu.r = r;
    sig_gpu.s = s;

    results[i] = ecdsa_verify(msg, &pub, &sig_gpu);
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

    // Generate test data
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

    // Alloc GPU
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

    // Sign + gen pubkeys on GPU
    ecdsa_sign_batch_kernel<<<blocks,128>>>(d_msgs, d_sk, d_sigs, d_ok, batch);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    generator_mul_batch_kernel<<<blocks,128>>>(d_sk, d_pubs, batch);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());

    // Read back sigs + pubkeys
    std::vector<ECDSASignatureGPU> h_sigs(batch);
    std::vector<JacobianPoint> h_pubs(batch);
    CUDA_CHECK(cudaMemcpy(h_sigs.data(), d_sigs,
        batch*sizeof(ECDSASignatureGPU), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pubs.data(), d_pubs,
        batch*sizeof(JacobianPoint), cudaMemcpyDeviceToHost));

    // Convert Jacobian → 33-byte compressed on GPU
    uint8_t *d_pk33;
    CUDA_CHECK(cudaMalloc(&d_pk33, batch*33));
    batch_jacobian_to_compressed_kernel<<<blocks,128>>>(d_pubs, d_pk33, batch);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());

    // Convert ECDSASignatureGPU -> compact 64-byte BE format for the current
    // verify kernel, and opaque 64-byte LE format for the raw-entry path.
    std::vector<uint8_t> h_compact(batch * 64);
    std::vector<uint8_t> h_opaque(batch * 64);
    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < 4; j++) {
            uint64_t rv = h_sigs[i].r.limbs[j];
            uint64_t sv = h_sigs[i].s.limbs[j];
            for (int k = 0; k < 8; k++) {
                h_compact[i*64 + 31 - j*8 - k] = (uint8_t)(rv >> (k*8));
                h_compact[i*64 + 63 - j*8 - k] = (uint8_t)(sv >> (k*8));
                h_opaque[i*64 + j*8 + k] = (uint8_t)(rv >> (k*8));
                h_opaque[i*64 + 32 + j*8 + k] = (uint8_t)(sv >> (k*8));
            }
        }
    }
    uint8_t *d_compact;
    CUDA_CHECK(cudaMalloc(&d_compact, batch*64));
    CUDA_CHECK(cudaMemcpy(d_compact, h_compact.data(), batch*64, cudaMemcpyHostToDevice));
    uint8_t *d_opaque;
    CUDA_CHECK(cudaMalloc(&d_opaque, batch*64));
    CUDA_CHECK(cudaMemcpy(d_opaque, h_opaque.data(), batch*64, cudaMemcpyHostToDevice));

    bool *d_res;
    CUDA_CHECK(cudaMalloc(&d_res, batch*sizeof(bool)));

    // ============ PATH A: GPU verify (Jacobian + compact sig) ============
    std::printf("============================================\n");
    std::printf("  PATH A: GPU verify (pre-expanded pubkey + compact sig)\n");
    std::printf("  (Current verify kernel: GPU parses compact r||s)\n");
    std::printf("============================================\n");

    std::vector<double> ga;
    for (int pass = 0; pass < warmup + measure; pass++) {
        cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        ecdsa_verify_batch_kernel<<<blocks,128>>>(d_msgs,d_pubs,d_compact,d_res,batch);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms, s, e);
        cudaEventDestroy(s); cudaEventDestroy(e);
        if (pass >= warmup) ga.push_back(ms * 1e6);
    }
    std::sort(ga.begin(), ga.end());
    double a_ns = ga[ga.size()/2] / batch;
    std::printf("  GPU verify only:     %8.1f ns/op  (%.2f M/s)\n", a_ns, 1000.0/a_ns);

    std::vector<uint8_t> ra(batch);
    CUDA_CHECK(cudaMemcpy(ra.data(), d_res, batch, cudaMemcpyDeviceToHost));
    int oa = 0; for (int i=0; i<batch; i++) if (ra[i]) oa++;
    std::printf("  Valid: %d/%d\n", oa, batch);

    // ============ PATH B: GPU raw entry (decompress + parse + verify) ============
    std::printf("\n============================================\n");
    std::printf("  PATH B: GPU RAW ENTRY (zero CPU prep)\n");
    std::printf("  (GPU: decompress + parse + normalize + verify)\n");
    std::printf("============================================\n");

    std::vector<double> gb;
    for (int pass = 0; pass < warmup + measure; pass++) {
        cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        ecdsa_raw_entry_verify_kernel<<<blocks,128>>>(d_msgs,d_pk33,d_opaque,d_res,batch);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms, s, e);
        cudaEventDestroy(s); cudaEventDestroy(e);
        if (pass >= warmup) gb.push_back(ms * 1e6);
    }
    std::sort(gb.begin(), gb.end());
    double b_ns = gb[gb.size()/2] / batch;
    std::printf("  GPU raw entry:       %8.1f ns/op  (%.2f M/s)\n", b_ns, 1000.0/b_ns);

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
    std::printf("  Path A (pre-expanded + compact): %8.1f ns/op\n", a_ns);
    std::printf("  Path B (GPU raw entry):          %8.1f ns/op\n", b_ns);
    std::printf("  GPU prep overhead:               %8.1f ns  (%+.1f%%)\n", ovh, pct);
    std::printf("\n");
    std::printf("  WHAT GPU DOES PER THREAD (Path B):\n");
    std::printf("    1. Decompress pubkey   (33B -> affine x,y)\n");
    std::printf("    2. Parse r,s scalars   (64B opaque LE)\n");
    std::printf("    3. Normalize low-S     (BIP-62)\n");
    std::printf("    4. ECDSA verify        (scalar mul)\n");
    std::printf("\n");
    std::printf("  WHAT CPU DOES (Path B):\n");
    std::printf("    - NOTHING per entry. Just hands raw buffers to GPU.\n");
    std::printf("    - No decompress, no parse, no alloc, no copy.\n");
    std::printf("============================================\n");

    // Cleanup
    CUDA_CHECK(cudaFree(d_msgs));
    CUDA_CHECK(cudaMemset(d_sk, 0, batch*sizeof(Scalar)));
    CUDA_CHECK(cudaFree(d_sk)); CUDA_CHECK(cudaFree(d_sigs));
    CUDA_CHECK(cudaFree(d_ok)); CUDA_CHECK(cudaFree(d_pubs));
    CUDA_CHECK(cudaFree(d_pk33)); CUDA_CHECK(cudaFree(d_compact));
    CUDA_CHECK(cudaFree(d_opaque));
    CUDA_CHECK(cudaFree(d_res));
    return 0;
}
