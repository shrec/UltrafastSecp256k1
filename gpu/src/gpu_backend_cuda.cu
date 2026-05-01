/* ============================================================================
 * UltrafastSecp256k1 -- CUDA Backend Bridge
 * ============================================================================
 * Implements gpu::GpuBackend for CUDA/HIP.
 * Converts between flat uint8_t[] C ABI buffers and CUDA internal types,
 * manages device memory, and launches existing kernels.
 *
 * Compiled ONLY when SECP256K1_HAVE_CUDA is set (via CMake).
 * ============================================================================ */

#include "../include/gpu_backend.hpp"

#include <cstring>
#include <cstdio>
#include <vector>

/* -- Secure erase for private key zeroization ------------------------------ */
#include "secp256k1/detail/secure_erase.hpp"

/* -- CUDA runtime ---------------------------------------------------------- */
#include <cuda_runtime.h>

/* -- Existing CUDA headers (Layer 1) --------------------------------------- */
#include "secp256k1.cuh"
#include "ecdh.cuh"
#include "msm.cuh"
#include "schnorr.cuh"
#include "zk.cuh"
#include "bip324.cuh"
#include "gpu_compat.h"

/* Host helpers (gpu_cuda_host_helpers.h) no longer needed:
 * All Jacobian <-> compressed conversions now happen on-device via
 * batch_jac_to_compressed_kernel / batch_compressed_to_jac_kernel. */

/* ============================================================================
 * Internal helpers
 * ============================================================================ */

#define CUDA_TRY(expr)                                                         \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        if (_err != cudaSuccess) {                                             \
            set_error(GpuError::Launch, cudaGetErrorString(_err));              \
            return last_error();                                               \
        }                                                                      \
    } while (0)

namespace secp256k1 {

/* Kernels defined in cuda/src/secp256k1.cu (namespace secp256k1::cuda).
   Must be declared in the same namespace so the linker finds them. */
namespace cuda {
extern __global__ void ecdsa_verify_batch_kernel(
    const uint8_t* __restrict__ msg_hashes,
    const JacobianPoint* __restrict__ public_keys,
    const ECDSASignatureGPU* __restrict__ sigs,
    bool*          __restrict__ results,
    int count);

extern __global__ void schnorr_verify_batch_kernel(
    const uint8_t* __restrict__ pubkeys_x,
    const uint8_t* __restrict__ msgs,
    const SchnorrSignatureGPU* __restrict__ sigs,
    bool*          __restrict__ results,
    int count);
extern __global__ void frost_verify_partial_batch_kernel(
    const uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*,
    const uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*,
    uint8_t*, int);
extern __global__ void ecdsa_recover_batch_kernel(
    const uint8_t* __restrict__ msg_hashes,
    const ECDSASignatureGPU* __restrict__ sigs,
    const int*     __restrict__ recids,
    JacobianPoint* __restrict__ recovered_keys,
    bool*          __restrict__ results,
    int count);
extern __global__ void ecdsa_snark_witness_batch_kernel(
    const uint8_t*           __restrict__ msg_hashes,
    const JacobianPoint*     __restrict__ public_keys,
    const ECDSASignatureGPU* __restrict__ sigs,
    EcdsaSnarkWitnessFlat*   __restrict__ out,
    int count);
extern __global__ void schnorr_snark_witness_batch_kernel(
    const uint8_t*             __restrict__ msgs32,
    const uint8_t*             __restrict__ pubkeys_x32,
    const uint8_t*             __restrict__ sigs64,
    SchnorrSnarkWitnessFlat*   __restrict__ out,
    int count);
} // namespace cuda

namespace gpu {

/* ============================================================================
 * BIP-352 Silent Payment GPU scan kernel (inline, no bench dependency)
 * ============================================================================ */

namespace {

/** BIP0352/SharedSecret tagged SHA-256 midstate (precomputed):
 *  SHA256("BIP0352/SharedSecret") || SHA256("BIP0352/SharedSecret")
 *  compressed into the H[0..7] intermediate state.                  */
static __device__ __forceinline__ void bip352_tagged_hash_37(
    const uint8_t ser37[37], uint8_t hash32[32])
{
    /* Initialise from precomputed BIP0352/SharedSecret tagged midstate. */
    uint32_t h[8] = {
        0x88831537U, 0x5127079bU, 0x69c2137bU, 0xab0303e6U,
        0x98fa21faU, 0x4a888523U, 0xbd99daabU, 0xf25e5e0aU
    };
    /* Build the single 64-byte block: ser37 (37 bytes) + 0x80 + zeros + length.
     * Total bytes hashed from start = 64 (tag block) + 37 = 101.
     * Bit length = 101 * 8 = 808 = 0x0000000000000328. Low 32 bits only needed. */
    uint8_t block[64] = {};
    for (int i = 0; i < 37; ++i) block[i] = ser37[i];
    block[37] = 0x80u;
    block[62] = 0x03u;
    block[63] = 0x28u;
    secp256k1::cuda::sha256_compress(h, block);
    /* Write h[0..7] big-endian. */
    for (int i = 0; i < 8; ++i) {
        hash32[i*4  ] = (uint8_t)(h[i] >> 24);
        hash32[i*4+1] = (uint8_t)(h[i] >> 16);
        hash32[i*4+2] = (uint8_t)(h[i] >>  8);
        hash32[i*4+3] = (uint8_t)(h[i]);
    }
}

/** GPU kernel: BIP-352 Silent Payment pipeline, 1 thread per tweak point.
 *
 *  Steps per thread:
 *    1. shared    = scan_k × tweak_pts[idx]   (GLV wNAF)
 *    2. ser37     = compress(shared) ∥ [0,0,0,0]
 *    3. hash      = SHA256_tagged(ser37)
 *    4. output    = hash × G
 *    5. cand      = spend_aff + output
 *    6. prefix    = upper 64 bits of cand.x                         */
static __global__ void bip352_scan_batch_kernel(
    const secp256k1::cuda::JacobianPoint* __restrict__ tweak_pts,
    const secp256k1::cuda::Scalar*        __restrict__ scan_k,
    const secp256k1::cuda::AffinePoint*   __restrict__ spend_aff,
    uint64_t* __restrict__ prefixes,
    int n)
{
    using namespace secp256k1::cuda;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    /* 1. shared = scan_k × tweak_pts[idx] */
    JacobianPoint shared;
    scalar_mul_glv_wnaf(&tweak_pts[idx], scan_k, &shared);
    if (shared.infinity) { prefixes[idx] = 0; return; }

    /* 2. Serialize shared secret to 37 bytes: [prefix_byte][x32][0,0,0,0] */
    uint8_t comp33[33];
    point_to_compressed(&shared, comp33);
    uint8_t ser37[37];
    for (int i = 0; i < 33; ++i) ser37[i] = comp33[i];
    ser37[33] = ser37[34] = ser37[35] = ser37[36] = 0;

    /* 3. Tagged SHA-256 */
    uint8_t hash[32];
    bip352_tagged_hash_37(ser37, hash);

    /* 4. hash × G */
    Scalar hs;
    scalar_from_bytes(hash, &hs);
    JacobianPoint gen_out;
    scalar_mul_generator_const(&hs, &gen_out);

    /* 5. cand = spend_aff + gen_out */
    JacobianPoint cand;
    jacobian_add_mixed(&gen_out, &spend_aff[0], &cand);
    if (cand.infinity) { prefixes[idx] = 0; return; }

    /* 6. Extract upper 64 bits of cand.x (after Jacobian normalisation) */
    FieldElement ax, ay;
    jacobian_to_affine(&cand, &ax, &ay);
    (void)ay;
    uint8_t xb[32];
    field_to_bytes(&ax, xb);
    uint64_t pref = 0;
    for (int i = 0; i < 8; ++i) pref = (pref << 8) | xb[i];
    prefixes[idx] = pref;
}

} // anonymous namespace

/* Import CUDA types (FieldElement, Scalar, JacobianPoint, etc.) and kernels
   from the secp256k1::cuda namespace into secp256k1::gpu. */
using namespace cuda;

namespace {

struct CudaBatchScratch {
    std::vector<uint8_t> result_bytes;

    uint8_t* ensure_results(std::size_t count) {
        if (result_bytes.size() < count) {
            result_bytes.resize(count);
        }
        return result_bytes.data();
    }
};

static thread_local CudaBatchScratch g_cuda_batch_scratch;

} // namespace

/* ============================================================================
 * Thin wrapper kernels for device functions without __global__ entry points
 * ============================================================================ */

/** Batch Jacobian -> compressed pubkey (33 bytes each) on GPU.
 *  Uses point_to_compressed() which normalises via field_inv on device. */
__global__ void batch_jac_to_compressed_kernel(
    const JacobianPoint* pts, uint8_t* out33, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    if (!point_to_compressed(&pts[idx], out33 + idx * 33)) {
        /* infinity — write zero prefix */
        memset(out33 + idx * 33, 0, 33);
    }
}

/** Batch compressed pubkey (33 bytes) -> JacobianPoint on GPU.
 *  Uses point_from_compressed() which does lift_x + sqrt on device. */
__global__ void batch_compressed_to_jac_kernel(
    const uint8_t* pubs33, JacobianPoint* out, bool* ok, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    ok[idx] = point_from_compressed(pubs33 + idx * 33, &out[idx]);
}

/** MSM block reduction: each block reduces BLOCK_SZ partials → 1 result
 *  via shared-memory tree.  Shared mem = blockDim.x * sizeof(JacobianPoint).
 *  Output: one JacobianPoint per block written to block_results[blockIdx.x]. */
__global__ void msm_block_reduce_kernel(
    const JacobianPoint* __restrict__ partials,
    int n,
    JacobianPoint* block_results)
{
    extern __shared__ JacobianPoint sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < n) {
        sdata[tid] = partials[idx];
    } else {
        sdata[tid].infinity = true;
        field_set_zero(&sdata[tid].x);
        field_set_zero(&sdata[tid].y);
        field_set_one(&sdata[tid].z);
    }
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sdata[tid].infinity) {
                sdata[tid] = sdata[tid + stride];
            } else if (!sdata[tid + stride].infinity) {
                JacobianPoint tmp;
                jacobian_add(&sdata[tid], &sdata[tid + stride], &tmp);
                sdata[tid] = tmp;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_results[blockIdx.x] = sdata[0];
    }
}

/** MSM final reduction: sum a small array of per-block results and compress.
 *  Runs on a single thread; input count should be small (n/256). */
__global__ void msm_reduce_and_compress_kernel(
    const JacobianPoint* partials, int n, uint8_t* out33, bool* ok)
{
    JacobianPoint acc;
    acc.infinity = true;
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
    if (acc.infinity) {
        memset(out33, 0, 33);
        *ok = false;
    } else {
        point_to_compressed(&acc, out33);
        *ok = true;
    }
}

/** ECDH batch kernel: each thread computes SHA-256(0x02 || x) where
 *  x = x-coordinate of privkey[i] * pubkey[i]. */
__global__ void ecdh_batch_kernel(
    const Scalar* privkeys,
    const JacobianPoint* peer_pubs,
    uint8_t* out_secrets,      /* count * 32 bytes */
    bool* out_ok,
    int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    out_ok[idx] = secp256k1::cuda::ecdh_compute(
        &privkeys[idx], &peer_pubs[idx], out_secrets + idx * 32);
}

/* ============================================================================
 * CudaBackend implementation
 * ============================================================================ */

class CudaBackend final : public GpuBackend {
public:
    CudaBackend() = default;
    ~CudaBackend() override { shutdown(); }

    /* -- Backend identity -------------------------------------------------- */
    uint32_t backend_id() const override { return 1; /* CUDA */ }
    const char* backend_name() const override { return "CUDA"; }

    /* -- Device enumeration ------------------------------------------------ */
    uint32_t device_count() const override {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
        return static_cast<uint32_t>(count);
    }

    GpuError device_info(uint32_t device_index, DeviceInfo& out) const override {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || device_index >= static_cast<uint32_t>(count))
            return GpuError::Device;

        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, static_cast<int>(device_index)) != cudaSuccess)
            return GpuError::Device;

        std::memset(&out, 0, sizeof(out));
        std::memcpy(out.name, prop.name, sizeof(out.name) - 1);
        out.name[sizeof(out.name) - 1] = '\0';
        out.global_mem_bytes       = prop.totalGlobalMem;
        out.compute_units          = static_cast<uint32_t>(prop.multiProcessorCount);
#if CUDART_VERSION >= 13000
        { int _clk_khz = 0;
          cudaDeviceGetAttribute(&_clk_khz, cudaDevAttrClockRate, static_cast<int>(device_index));
          out.max_clock_mhz = static_cast<uint32_t>(_clk_khz / 1000); }
#else
        out.max_clock_mhz          = static_cast<uint32_t>(prop.clockRate / 1000);
#endif
        out.max_threads_per_block  = static_cast<uint32_t>(prop.maxThreadsPerBlock);
        out.backend_id             = 1;
        out.device_index           = device_index;
        return GpuError::Ok;
    }

    /* -- Context lifecycle ------------------------------------------------- */
    GpuError init(uint32_t device_index) override {
        if (ready_) return GpuError::Ok;
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || device_index >= static_cast<uint32_t>(count)) {
            set_error(GpuError::Device, "CUDA device not found");
            return last_error();
        }
        auto err = cudaSetDevice(static_cast<int>(device_index));
        if (err != cudaSuccess) {
            set_error(GpuError::Device, cudaGetErrorString(err));
            return last_error();
        }
        device_idx_ = device_index;
        ready_ = true;
        clear_error();
        return GpuError::Ok;
    }

    void shutdown() override {
        msm_pool_.free_all();
        ready_ = false;
    }

    bool is_ready() const override { return ready_; }

    /* -- Error tracking ---------------------------------------------------- */
    GpuError last_error() const override { return last_err_; }
    const char* last_error_msg() const override { return last_msg_; }

    /* -- First-wave ops ---------------------------------------------------- */

    GpuError generator_mul_batch(
        const uint8_t* scalars32, size_t count,
        uint8_t* out_pubkeys33) override
    {
        if (!ready_) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!scalars32 || !out_pubkeys33) return set_error(GpuError::NullArg, "NULL buffer");

        /* Allocate device memory */
        Scalar* d_scalars = nullptr;
        JacobianPoint* d_results = nullptr;
        CUDA_TRY(cudaMalloc(&d_scalars, count * sizeof(Scalar)));
        CUDA_TRY(cudaMalloc(&d_results, count * sizeof(JacobianPoint)));

        /* Convert big-endian bytes → Scalar on host, then upload */
        std::vector<Scalar> h_scalars(count);
        for (size_t i = 0; i < count; ++i) {
            bytes_to_scalar(scalars32 + i * 32, &h_scalars[i]);
        }
        CUDA_TRY(cudaMemcpy(d_scalars, h_scalars.data(),
                             count * sizeof(Scalar), cudaMemcpyHostToDevice));

        /* Launch kernel */
        int threads = 128;
        int blocks  = (static_cast<int>(count) + threads - 1) / threads;
        generator_mul_windowed_batch_kernel<<<blocks, threads>>>(
            d_scalars, d_results, static_cast<int>(count));
        CUDA_TRY(cudaGetLastError());

        /* Convert Jacobian → compressed on GPU (device-side field_inv) */
        uint8_t* d_out = nullptr;
        CUDA_TRY(cudaMalloc(&d_out, count * 33));
        batch_jac_to_compressed_kernel<<<blocks, threads>>>(
            d_results, d_out, static_cast<int>(count));
        CUDA_TRY(cudaGetLastError());
        CUDA_TRY(cudaDeviceSynchronize());

        /* Download compressed pubkeys */
        CUDA_TRY(cudaMemcpy(out_pubkeys33, d_out,
                             count * 33, cudaMemcpyDeviceToHost));

        cudaFree(d_out);
        cudaFree(d_results);
        cudaFree(d_scalars);
        clear_error();
        return GpuError::Ok;
    }

    GpuError ecdsa_verify_batch(
        const uint8_t* msg_hashes32, const uint8_t* pubkeys33,
        const uint8_t* sigs64, size_t count,
        uint8_t* out_results) override
    {
        if (!ready_) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!msg_hashes32 || !pubkeys33 || !sigs64 || !out_results)
            return set_error(GpuError::NullArg, "NULL buffer");

        /* Prepare signatures on host */
        std::vector<ECDSASignatureGPU> h_sigs(count);
        for (size_t i = 0; i < count; ++i) {
            bytes_to_ecdsa_sig(sigs64 + i * 64, &h_sigs[i]);
        }

        /* Allocate device memory */
        uint8_t*            d_msgs    = nullptr;
        uint8_t*            d_pubs33  = nullptr;
        JacobianPoint*      d_pubs    = nullptr;
        bool*               d_pub_ok  = nullptr;
        ECDSASignatureGPU*  d_sigs    = nullptr;
        bool*               d_res     = nullptr;

        CUDA_TRY(cudaMalloc(&d_msgs, count * 32));
        CUDA_TRY(cudaMalloc(&d_pubs33, count * 33));
        CUDA_TRY(cudaMalloc(&d_pubs, count * sizeof(JacobianPoint)));
        CUDA_TRY(cudaMalloc(&d_pub_ok, count * sizeof(bool)));
        CUDA_TRY(cudaMalloc(&d_sigs, count * sizeof(ECDSASignatureGPU)));
        CUDA_TRY(cudaMalloc(&d_res, count * sizeof(bool)));

        CUDA_TRY(cudaMemcpy(d_msgs, msg_hashes32, count * 32, cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_pubs33, pubkeys33, count * 33, cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_sigs, h_sigs.data(), count * sizeof(ECDSASignatureGPU), cudaMemcpyHostToDevice));

        /* Decompress pubkeys on GPU */
        int threads = 128;
        int blocks  = (static_cast<int>(count) + threads - 1) / threads;
        batch_compressed_to_jac_kernel<<<blocks, threads>>>(
            d_pubs33, d_pubs, d_pub_ok, static_cast<int>(count));
        CUDA_TRY(cudaGetLastError());

        /* Launch verify */
        ecdsa_verify_batch_kernel<<<blocks, threads>>>(
            d_msgs, d_pubs, d_sigs, d_res, static_cast<int>(count));
        CUDA_TRY(cudaGetLastError());
        CUDA_TRY(cudaDeviceSynchronize());

        /* Download results */
        uint8_t* const h_res = g_cuda_batch_scratch.ensure_results(count);
        CUDA_TRY(cudaMemcpy(h_res, d_res, count * sizeof(bool), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < count; ++i)
            out_results[i] = h_res[i] ? 1 : 0;

        cudaFree(d_res);
        cudaFree(d_sigs);
        cudaFree(d_pub_ok);
        cudaFree(d_pubs);
        cudaFree(d_pubs33);
        cudaFree(d_msgs);
        clear_error();
        return GpuError::Ok;
    }

    GpuError schnorr_verify_batch(
        const uint8_t* msg_hashes32, const uint8_t* pubkeys_x32,
        const uint8_t* sigs64, size_t count,
        uint8_t* out_results) override
    {
        if (!ready_) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!msg_hashes32 || !pubkeys_x32 || !sigs64 || !out_results)
            return set_error(GpuError::NullArg, "NULL buffer");

        /* Prepare Schnorr sigs in CUDA type */
        std::vector<SchnorrSignatureGPU> h_sigs(count);
        for (size_t i = 0; i < count; ++i) {
            bytes_to_schnorr_sig(sigs64 + i * 64, &h_sigs[i]);
        }

        /* Allocate device memory */
        uint8_t*             d_pks  = nullptr;
        uint8_t*             d_msgs = nullptr;
        SchnorrSignatureGPU* d_sigs = nullptr;
        bool*                d_res  = nullptr;

        CUDA_TRY(cudaMalloc(&d_pks, count * 32));
        CUDA_TRY(cudaMalloc(&d_msgs, count * 32));
        CUDA_TRY(cudaMalloc(&d_sigs, count * sizeof(SchnorrSignatureGPU)));
        CUDA_TRY(cudaMalloc(&d_res, count * sizeof(bool)));

        CUDA_TRY(cudaMemcpy(d_pks, pubkeys_x32, count * 32, cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_msgs, msg_hashes32, count * 32, cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_sigs, h_sigs.data(), count * sizeof(SchnorrSignatureGPU), cudaMemcpyHostToDevice));

        /* Launch */
        int threads = 128;
        int blocks  = (static_cast<int>(count) + threads - 1) / threads;
        schnorr_verify_batch_kernel<<<blocks, threads>>>(
            d_pks, d_msgs, d_sigs, d_res, static_cast<int>(count));
        CUDA_TRY(cudaGetLastError());
        CUDA_TRY(cudaDeviceSynchronize());

        /* Download results */
        uint8_t* const h_res = g_cuda_batch_scratch.ensure_results(count);
        CUDA_TRY(cudaMemcpy(h_res, d_res, count * sizeof(bool), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < count; ++i)
            out_results[i] = h_res[i] ? 1 : 0;

        cudaFree(d_res);
        cudaFree(d_sigs);
        cudaFree(d_msgs);
        cudaFree(d_pks);
        clear_error();
        return GpuError::Ok;
    }

    GpuError ecdh_batch(
        const uint8_t* privkeys32, const uint8_t* peer_pubkeys33,
        size_t count, uint8_t* out_secrets32) override
    {
        if (!ready_) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!privkeys32 || !peer_pubkeys33 || !out_secrets32)
            return set_error(GpuError::NullArg, "NULL buffer");

        /* Convert private keys on host */
        std::vector<Scalar> h_keys(count);
        for (size_t i = 0; i < count; ++i) {
            bytes_to_scalar(privkeys32 + i * 32, &h_keys[i]);
        }

        /* Allocate device memory */
        Scalar*        d_keys   = nullptr;
        uint8_t*       d_pubs33 = nullptr;
        JacobianPoint* d_pubs   = nullptr;
        bool*          d_pub_ok = nullptr;
        uint8_t*       d_out    = nullptr;
        bool*          d_ok     = nullptr;

        CUDA_TRY(cudaMalloc(&d_keys, count * sizeof(Scalar)));
        CUDA_TRY(cudaMalloc(&d_pubs33, count * 33));
        CUDA_TRY(cudaMalloc(&d_pubs, count * sizeof(JacobianPoint)));
        CUDA_TRY(cudaMalloc(&d_pub_ok, count * sizeof(bool)));
        CUDA_TRY(cudaMalloc(&d_out, count * 32));
        CUDA_TRY(cudaMalloc(&d_ok, count * sizeof(bool)));

        CUDA_TRY(cudaMemcpy(d_keys, h_keys.data(), count * sizeof(Scalar), cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_pubs33, peer_pubkeys33, count * 33, cudaMemcpyHostToDevice));

        /* Decompress pubkeys on GPU */
        int threads = 128;
        int blocks  = (static_cast<int>(count) + threads - 1) / threads;
        batch_compressed_to_jac_kernel<<<blocks, threads>>>(
            d_pubs33, d_pubs, d_pub_ok, static_cast<int>(count));
        CUDA_TRY(cudaGetLastError());

        /* Launch ECDH */
        ecdh_batch_kernel<<<blocks, threads>>>(d_keys, d_pubs, d_out, d_ok, static_cast<int>(count));
        CUDA_TRY(cudaGetLastError());
        CUDA_TRY(cudaDeviceSynchronize());

        /* Download results */
        CUDA_TRY(cudaMemcpy(out_secrets32, d_out, count * 32, cudaMemcpyDeviceToHost));

        /* Check for failures (RAII: vector cleans up even if CUDA_TRY
         * triggers an early return on the download failure path). */
        std::vector<uint8_t> h_ok_buf(count);
        CUDA_TRY(cudaMemcpy(h_ok_buf.data(), d_ok, count * sizeof(bool), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < count; ++i) {
            if (!h_ok_buf[i]) std::memset(out_secrets32 + i * 32, 0, 32);
        }

        /* Zeroize private keys on device and host before freeing */
        cudaMemset(d_keys, 0, count * sizeof(Scalar));
        secp256k1::detail::secure_erase(h_keys.data(), h_keys.size() * sizeof(Scalar));

        cudaFree(d_ok);
        cudaFree(d_out);
        cudaFree(d_pub_ok);
        cudaFree(d_pubs);
        cudaFree(d_pubs33);
        cudaFree(d_keys);
        clear_error();
        return GpuError::Ok;
    }

    GpuError hash160_pubkey_batch(
        const uint8_t* pubkeys33, size_t count,
        uint8_t* out_hash160) override
    {
        if (!ready_) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!pubkeys33 || !out_hash160)
            return set_error(GpuError::NullArg, "NULL buffer");

        uint8_t* d_pubs = nullptr;
        uint8_t* d_hash = nullptr;

        CUDA_TRY(cudaMalloc(&d_pubs, count * 33));
        CUDA_TRY(cudaMalloc(&d_hash, count * 20));
        CUDA_TRY(cudaMemcpy(d_pubs, pubkeys33, count * 33, cudaMemcpyHostToDevice));

        int threads = 256;
        int blocks  = (static_cast<int>(count) + threads - 1) / threads;
        hash160_pubkey_kernel<<<blocks, threads>>>(
            d_pubs, 33, d_hash, static_cast<int>(count));
        CUDA_TRY(cudaGetLastError());
        CUDA_TRY(cudaDeviceSynchronize());

        CUDA_TRY(cudaMemcpy(out_hash160, d_hash, count * 20, cudaMemcpyDeviceToHost));

        cudaFree(d_hash);
        cudaFree(d_pubs);
        clear_error();
        return GpuError::Ok;
    }

    GpuError msm(
        const uint8_t* scalars32, const uint8_t* points33,
        size_t n, uint8_t* out_result33) override
    {
        if (!ready_) return set_error(GpuError::Device, "context not initialised");
        if (n == 0) { clear_error(); return GpuError::Ok; }
        if (!scalars32 || !points33 || !out_result33)
            return set_error(GpuError::NullArg, "NULL buffer");

        /* Convert scalars on host (just byte reinterpretation, no field ops) */
        std::vector<Scalar> h_scalars(n);
        for (size_t i = 0; i < n; ++i) {
            bytes_to_scalar(scalars32 + i * 32, &h_scalars[i]);
        }

        /* Ensure persistent pool is large enough (avoids per-call cudaMalloc) */
        {
            cudaError_t pool_err = msm_pool_.ensure(n);
            if (pool_err != cudaSuccess)
                return set_error(GpuError::Memory, cudaGetErrorString(pool_err));
        }

        Scalar*        d_scalars   = msm_pool_.d_scalars;
        uint8_t*       d_pts33     = msm_pool_.d_pts33;
        JacobianPoint* d_points    = msm_pool_.d_points;
        bool*          d_pt_ok     = msm_pool_.d_pt_ok;
        JacobianPoint* d_partials  = msm_pool_.d_partials;
        JacobianPoint* d_blk_parts = msm_pool_.d_blk_parts;
        uint8_t*       d_out33     = msm_pool_.d_out33;
        bool*          d_ok        = msm_pool_.d_ok;

        /* Warp-parallel reduce: block size 256, shared mem per block = 256 * sizeof(JacobianPoint) */
        constexpr int kReduceBlock = 256;
        const int n_int      = static_cast<int>(n);
        const int scatter_blocks = (n_int + kReduceBlock - 1) / kReduceBlock;
        const size_t smem    = kReduceBlock * sizeof(JacobianPoint);

        CUDA_TRY(cudaMemcpy(d_scalars, h_scalars.data(), n * sizeof(Scalar), cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_pts33, points33, n * 33, cudaMemcpyHostToDevice));

        /* Decompress points on GPU */
        int threads = 128;
        int blocks  = (n_int + threads - 1) / threads;
        batch_compressed_to_jac_kernel<<<blocks, threads>>>(
            d_pts33, d_points, d_pt_ok, n_int);
        CUDA_TRY(cudaGetLastError());

        /* Phase 1: scatter — each thread computes scalars[i] * points[i] */
        msm_scatter_kernel<<<blocks, threads>>>(d_scalars, d_points, d_partials, n_int);
        CUDA_TRY(cudaGetLastError());

        /* Phase 2a: parallel block reduce (256 partials → 1 per block) */
        msm_block_reduce_kernel<<<scatter_blocks, kReduceBlock, smem>>>(
            d_partials, n_int, d_blk_parts);
        CUDA_TRY(cudaGetLastError());

        /* Phase 2b: final single-thread reduce over the small block results */
        msm_reduce_and_compress_kernel<<<1, 1>>>(d_blk_parts, scatter_blocks, d_out33, d_ok);
        CUDA_TRY(cudaGetLastError());
        CUDA_TRY(cudaDeviceSynchronize());

        /* Download compressed result */
        CUDA_TRY(cudaMemcpy(out_result33, d_out33, 33, cudaMemcpyDeviceToHost));
        bool result_ok;
        CUDA_TRY(cudaMemcpy(&result_ok, d_ok, sizeof(bool), cudaMemcpyDeviceToHost));

        /* Buffers stay alive in msm_pool_ — do NOT free them here */

        if (!result_ok) {
            std::memset(out_result33, 0, 33);
            return set_error(GpuError::Arith, "MSM result is point at infinity");
        }

        clear_error();
        return GpuError::Ok;
    }

    GpuError frost_verify_partial_batch(
        const uint8_t* z_i32, const uint8_t* D_i33, const uint8_t* E_i33,
        const uint8_t* Y_i33, const uint8_t* rho_i32, const uint8_t* lambda_ie32,
        const uint8_t* negate_R, const uint8_t* negate_key,
        size_t count, uint8_t* out_results) override
    {
        if (!ready_) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!z_i32 || !D_i33 || !E_i33 || !Y_i33 || !rho_i32 || !lambda_ie32 ||
            !negate_R || !negate_key || !out_results)
            return set_error(GpuError::NullArg, "NULL buffer");

        uint8_t *d_z = nullptr, *d_D = nullptr, *d_E = nullptr, *d_Y = nullptr;
        uint8_t *d_rho = nullptr, *d_lie = nullptr;
        uint8_t *d_nR = nullptr, *d_nK = nullptr, *d_res = nullptr;

        CUDA_TRY(cudaMalloc(&d_z,   count * 32));
        CUDA_TRY(cudaMalloc(&d_D,   count * 33));
        CUDA_TRY(cudaMalloc(&d_E,   count * 33));
        CUDA_TRY(cudaMalloc(&d_Y,   count * 33));
        CUDA_TRY(cudaMalloc(&d_rho, count * 32));
        CUDA_TRY(cudaMalloc(&d_lie, count * 32));
        CUDA_TRY(cudaMalloc(&d_nR,  count));
        CUDA_TRY(cudaMalloc(&d_nK,  count));
        CUDA_TRY(cudaMalloc(&d_res, count));

        CUDA_TRY(cudaMemcpy(d_z,   z_i32,       count * 32, cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_D,   D_i33,       count * 33, cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_E,   E_i33,       count * 33, cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_Y,   Y_i33,       count * 33, cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_rho, rho_i32,     count * 32, cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_lie, lambda_ie32, count * 32, cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_nR,  negate_R,    count,      cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_nK,  negate_key,  count,      cudaMemcpyHostToDevice));

        int threads = 128;
        int blocks  = (static_cast<int>(count) + threads - 1) / threads;
        frost_verify_partial_batch_kernel<<<blocks, threads>>>(
            d_z, d_D, d_E, d_Y, d_rho, d_lie, d_nR, d_nK,
            d_res, static_cast<int>(count));
        CUDA_TRY(cudaGetLastError());
        CUDA_TRY(cudaDeviceSynchronize());

        CUDA_TRY(cudaMemcpy(out_results, d_res, count, cudaMemcpyDeviceToHost));

        cudaFree(d_res);
        cudaFree(d_nK);  cudaFree(d_nR);
        cudaFree(d_lie); cudaFree(d_rho);
        cudaFree(d_Y);   cudaFree(d_E);   cudaFree(d_D);   cudaFree(d_z);
        clear_error();
        return GpuError::Ok;
    }

    GpuError ecrecover_batch(
        const uint8_t* msg_hashes32, const uint8_t* sigs64,
        const int* recids, size_t count,
        uint8_t* out_pubkeys33, uint8_t* out_valid) override
    {
        if (!ready_) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!msg_hashes32 || !sigs64 || !recids || !out_pubkeys33 || !out_valid)
            return set_error(GpuError::NullArg, "NULL buffer");

        /* Host-side: parse compact sigs into ECDSASignatureGPU structs */
        std::vector<ECDSASignatureGPU> h_sigs(count);
        for (size_t i = 0; i < count; ++i)
            bytes_to_ecdsa_sig(sigs64 + i * 64, &h_sigs[i]);

        /* Device allocation */
        uint8_t*            d_msgs   = nullptr;
        ECDSASignatureGPU*  d_sigs   = nullptr;
        int*                d_recids = nullptr;
        JacobianPoint*      d_keys   = nullptr;
        bool*               d_res    = nullptr;
        uint8_t*            d_out33  = nullptr;

        CUDA_TRY(cudaMalloc(&d_msgs,   count * 32));
        CUDA_TRY(cudaMalloc(&d_sigs,   count * sizeof(ECDSASignatureGPU)));
        CUDA_TRY(cudaMalloc(&d_recids, count * sizeof(int)));
        CUDA_TRY(cudaMalloc(&d_keys,   count * sizeof(JacobianPoint)));
        CUDA_TRY(cudaMalloc(&d_res,    count * sizeof(bool)));
        CUDA_TRY(cudaMalloc(&d_out33,  count * 33));

        CUDA_TRY(cudaMemcpy(d_msgs,   msg_hashes32,    count * 32,                     cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_sigs,   h_sigs.data(),   count * sizeof(ECDSASignatureGPU), cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_recids, recids,           count * sizeof(int),            cudaMemcpyHostToDevice));

        int threads = 128;
        int blocks  = (static_cast<int>(count) + threads - 1) / threads;

        /* Recover Jacobian points on GPU */
        ecdsa_recover_batch_kernel<<<blocks, threads>>>(
            d_msgs, d_sigs, d_recids, d_keys, d_res, static_cast<int>(count));
        CUDA_TRY(cudaGetLastError());

        /* Convert Jacobian → compressed 33-byte pubkeys on GPU */
        batch_jac_to_compressed_kernel<<<blocks, threads>>>(
            d_keys, d_out33, static_cast<int>(count));
        CUDA_TRY(cudaGetLastError());
        CUDA_TRY(cudaDeviceSynchronize());

        CUDA_TRY(cudaMemcpy(out_pubkeys33, d_out33, count * 33, cudaMemcpyDeviceToHost));

            std::vector<uint8_t> h_res(count);
            CUDA_TRY(cudaMemcpy(h_res.data(), d_res, count * sizeof(bool), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < count; ++i) {
            out_valid[i] = h_res[i] ? 1 : 0;
            if (!h_res[i])
                std::memset(out_pubkeys33 + i * 33, 0, 33); /* zero failed entries */
        }

        cudaFree(d_out33);
        cudaFree(d_res);   cudaFree(d_keys);
        cudaFree(d_recids); cudaFree(d_sigs);  cudaFree(d_msgs);
        clear_error();
        return GpuError::Ok;
    }

    /* -- ZK batch ops ------------------------------------------------------ */

    GpuError zk_knowledge_verify_batch(
        const uint8_t* proofs64, const uint8_t* pubkeys65,
        const uint8_t* messages32, size_t count,
        uint8_t* out_results) override
    {
        if (!ready_) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!proofs64 || !pubkeys65 || !messages32 || !out_results)
            return set_error(GpuError::NullArg, "NULL buffer");

        /* Convert proofs: 64-byte flat → KnowledgeProofGPU (rx[32] + Scalar) */
        std::vector<KnowledgeProofGPU> h_proofs(count);
        for (size_t i = 0; i < count; ++i) {
            const uint8_t* p = proofs64 + i * 64;
            std::memcpy(h_proofs[i].rx, p, 32);
            bytes_to_scalar(p + 32, &h_proofs[i].s);
        }

        /* Convert pubkeys: 65-byte uncompressed (04 || x[32] || y[32]) → AffinePoint */
        std::vector<AffinePoint> h_pubs(count);
        for (size_t i = 0; i < count; ++i) {
            const uint8_t* pk = pubkeys65 + i * 65;
            bytes_to_field(pk + 1, &h_pubs[i].x);
            bytes_to_field(pk + 33, &h_pubs[i].y);
        }

        KnowledgeProofGPU* d_proofs = nullptr;
        AffinePoint*       d_pubs   = nullptr;
        uint8_t*           d_msgs   = nullptr;
        bool*              d_res    = nullptr;

        CUDA_TRY(cudaMalloc(&d_proofs, count * sizeof(KnowledgeProofGPU)));
        CUDA_TRY(cudaMalloc(&d_pubs,   count * sizeof(AffinePoint)));
        CUDA_TRY(cudaMalloc(&d_msgs,   count * 32));
        CUDA_TRY(cudaMalloc(&d_res,    count * sizeof(bool)));

        CUDA_TRY(cudaMemcpy(d_proofs, h_proofs.data(), count * sizeof(KnowledgeProofGPU), cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_pubs,   h_pubs.data(),   count * sizeof(AffinePoint),       cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_msgs,   messages32,      count * 32,                        cudaMemcpyHostToDevice));

        int threads = 128;
        int blocks  = (static_cast<int>(count) + threads - 1) / threads;
        knowledge_verify_batch_kernel<<<blocks, threads>>>(
            d_proofs, d_pubs, d_msgs, d_res, static_cast<uint32_t>(count));
        CUDA_TRY(cudaGetLastError());
        CUDA_TRY(cudaDeviceSynchronize());

        uint8_t* const h_res = g_cuda_batch_scratch.ensure_results(count);
        CUDA_TRY(cudaMemcpy(h_res, d_res, count * sizeof(bool), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < count; ++i)
            out_results[i] = h_res[i] ? 1 : 0;

        cudaFree(d_res); cudaFree(d_msgs); cudaFree(d_pubs); cudaFree(d_proofs);
        clear_error();
        return GpuError::Ok;
    }

    GpuError zk_dleq_verify_batch(
        const uint8_t* proofs64,
        const uint8_t* G_pts65, const uint8_t* H_pts65,
        const uint8_t* P_pts65, const uint8_t* Q_pts65,
        size_t count, uint8_t* out_results) override
    {
        if (!ready_) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!proofs64 || !G_pts65 || !H_pts65 || !P_pts65 || !Q_pts65 || !out_results)
            return set_error(GpuError::NullArg, "NULL buffer");

        std::vector<DLEQProofGPU> h_proofs(count);
        std::vector<AffinePoint> h_G(count), h_H(count), h_P(count), h_Q(count);
        for (size_t i = 0; i < count; ++i) {
            const uint8_t* p = proofs64 + i * 64;
            bytes_to_scalar(p,      &h_proofs[i].e);
            bytes_to_scalar(p + 32, &h_proofs[i].s);

            bytes_to_field(G_pts65 + i * 65 + 1,  &h_G[i].x);
            bytes_to_field(G_pts65 + i * 65 + 33, &h_G[i].y);
            bytes_to_field(H_pts65 + i * 65 + 1,  &h_H[i].x);
            bytes_to_field(H_pts65 + i * 65 + 33, &h_H[i].y);
            bytes_to_field(P_pts65 + i * 65 + 1,  &h_P[i].x);
            bytes_to_field(P_pts65 + i * 65 + 33, &h_P[i].y);
            bytes_to_field(Q_pts65 + i * 65 + 1,  &h_Q[i].x);
            bytes_to_field(Q_pts65 + i * 65 + 33, &h_Q[i].y);
        }

        DLEQProofGPU* d_proofs = nullptr;
        AffinePoint*  d_G = nullptr, *d_H = nullptr, *d_P = nullptr, *d_Q = nullptr;
        bool*         d_res = nullptr;

        CUDA_TRY(cudaMalloc(&d_proofs, count * sizeof(DLEQProofGPU)));
        CUDA_TRY(cudaMalloc(&d_G,     count * sizeof(AffinePoint)));
        CUDA_TRY(cudaMalloc(&d_H,     count * sizeof(AffinePoint)));
        CUDA_TRY(cudaMalloc(&d_P,     count * sizeof(AffinePoint)));
        CUDA_TRY(cudaMalloc(&d_Q,     count * sizeof(AffinePoint)));
        CUDA_TRY(cudaMalloc(&d_res,   count * sizeof(bool)));

        CUDA_TRY(cudaMemcpy(d_proofs, h_proofs.data(), count * sizeof(DLEQProofGPU), cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_G, h_G.data(), count * sizeof(AffinePoint), cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_H, h_H.data(), count * sizeof(AffinePoint), cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_P, h_P.data(), count * sizeof(AffinePoint), cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_Q, h_Q.data(), count * sizeof(AffinePoint), cudaMemcpyHostToDevice));

        int threads = 128;
        int blocks  = (static_cast<int>(count) + threads - 1) / threads;
        dleq_verify_batch_kernel<<<blocks, threads>>>(
            d_proofs, d_G, d_H, d_P, d_Q, d_res, static_cast<uint32_t>(count));
        CUDA_TRY(cudaGetLastError());
        CUDA_TRY(cudaDeviceSynchronize());

        {
            /* RAII buffer: leaks avoided if cudaMemcpy fails and CUDA_TRY
             * returns early on the failure path. */
            std::vector<uint8_t> h_res_raw(count);
            CUDA_TRY(cudaMemcpy(h_res_raw.data(), d_res, count * sizeof(bool), cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < count; ++i)
                out_results[i] = h_res_raw[i] ? 1 : 0;
        }

        cudaFree(d_res);
        cudaFree(d_Q); cudaFree(d_P); cudaFree(d_H); cudaFree(d_G);
        cudaFree(d_proofs);
        clear_error();
        return GpuError::Ok;
    }

    GpuError bulletproof_verify_batch(
        const uint8_t* proofs324, const uint8_t* commitments65,
        const uint8_t* H_generator65, size_t count,
        uint8_t* out_results) override
    {
        if (!ready_) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!proofs324 || !commitments65 || !H_generator65 || !out_results)
            return set_error(GpuError::NullArg, "NULL buffer");

        /* Parse proofs: 4 affine points (A,S,T1,T2) + 2 scalars (tau_x, t_hat)
         * Layout per proof (324 bytes): 4 × 65-byte uncompressed points + 2 × 32-byte scalars
         * Point format: 04 || x[32] || y[32] */
        std::vector<RangeProofPolyGPU> h_proofs(count);
        for (size_t i = 0; i < count; ++i) {
            const uint8_t* p = proofs324 + i * 324;
            bytes_to_field(p + 1,   &h_proofs[i].A.x);   /* A at offset 0 */
            bytes_to_field(p + 33,  &h_proofs[i].A.y);
            bytes_to_field(p + 66,  &h_proofs[i].S.x);   /* S at offset 65 */
            bytes_to_field(p + 98,  &h_proofs[i].S.y);
            bytes_to_field(p + 131, &h_proofs[i].T1.x);  /* T1 at offset 130 */
            bytes_to_field(p + 163, &h_proofs[i].T1.y);
            bytes_to_field(p + 196, &h_proofs[i].T2.x);  /* T2 at offset 195 */
            bytes_to_field(p + 228, &h_proofs[i].T2.y);
            bytes_to_scalar(p + 260, &h_proofs[i].tau_x); /* tau_x at offset 260 */
            bytes_to_scalar(p + 292, &h_proofs[i].t_hat); /* t_hat at offset 292 */
        }

        std::vector<AffinePoint> h_commits(count);
        for (size_t i = 0; i < count; ++i) {
            const uint8_t* c = commitments65 + i * 65;
            bytes_to_field(c + 1,  &h_commits[i].x);
            bytes_to_field(c + 33, &h_commits[i].y);
        }

        AffinePoint h_gen;
        bytes_to_field(H_generator65 + 1,  &h_gen.x);
        bytes_to_field(H_generator65 + 33, &h_gen.y);

        RangeProofPolyGPU* d_proofs   = nullptr;
        AffinePoint*       d_commits  = nullptr;
        AffinePoint*       d_hgen     = nullptr;
        bool*              d_res      = nullptr;

        CUDA_TRY(cudaMalloc(&d_proofs,  count * sizeof(RangeProofPolyGPU)));
        CUDA_TRY(cudaMalloc(&d_commits, count * sizeof(AffinePoint)));
        CUDA_TRY(cudaMalloc(&d_hgen,    sizeof(AffinePoint)));
        CUDA_TRY(cudaMalloc(&d_res,     count * sizeof(bool)));

        CUDA_TRY(cudaMemcpy(d_proofs,  h_proofs.data(),  count * sizeof(RangeProofPolyGPU), cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_commits, h_commits.data(), count * sizeof(AffinePoint),       cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_hgen,    &h_gen,           sizeof(AffinePoint),                cudaMemcpyHostToDevice));

        int threads = 128;
        int blocks  = (static_cast<int>(count) + threads - 1) / threads;
        range_proof_poly_batch_kernel<<<blocks, threads>>>(
            d_proofs, d_commits, d_hgen, d_res, static_cast<uint32_t>(count));
        CUDA_TRY(cudaGetLastError());
        CUDA_TRY(cudaDeviceSynchronize());

        {
            /* RAII buffer: leaks avoided if cudaMemcpy fails and CUDA_TRY
             * returns early on the failure path. */
            std::vector<uint8_t> h_res_raw(count);
            CUDA_TRY(cudaMemcpy(h_res_raw.data(), d_res, count * sizeof(bool), cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < count; ++i)
                out_results[i] = h_res_raw[i] ? 1 : 0;
        }

        cudaFree(d_res); cudaFree(d_hgen); cudaFree(d_commits); cudaFree(d_proofs);
        clear_error();
        return GpuError::Ok;
    }

    /* -- BIP-324 batch ops ------------------------------------------------- */

    GpuError bip324_aead_encrypt_batch(
        const uint8_t* keys32, const uint8_t* nonces12,
        const uint8_t* plaintexts, const uint32_t* sizes,
        uint32_t max_payload, size_t count,
        uint8_t* wire_out) override
    {
        if (!ready_) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!keys32 || !nonces12 || !plaintexts || !sizes || !wire_out)
            return set_error(GpuError::NullArg, "NULL buffer");

        const size_t wire_stride = max_payload + 19;

        uint8_t*    d_keys = nullptr;
        uint8_t*    d_nonces = nullptr;
        uint8_t*    d_pt = nullptr;
        uint32_t*   d_sizes = nullptr;
        uint8_t*    d_wire = nullptr;

        CUDA_TRY(cudaMalloc(&d_keys,   count * 32));
        CUDA_TRY(cudaMalloc(&d_nonces, count * 12));
        CUDA_TRY(cudaMalloc(&d_pt,     count * max_payload));
        CUDA_TRY(cudaMalloc(&d_sizes,  count * sizeof(uint32_t)));
        CUDA_TRY(cudaMalloc(&d_wire,   count * wire_stride));

        CUDA_TRY(cudaMemcpy(d_keys,   keys32,     count * 32,               cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_nonces, nonces12,   count * 12,               cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_pt,     plaintexts, count * max_payload,      cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_sizes,  sizes,      count * sizeof(uint32_t), cudaMemcpyHostToDevice));

        int threads = 128;
        int blocks  = (static_cast<int>(count) + threads - 1) / threads;
        secp256k1::cuda::bip324::bip324_aead_encrypt_kernel<<<blocks, threads>>>(
            d_keys, d_nonces, d_pt, d_sizes, d_wire,
            max_payload, static_cast<int>(count));
        CUDA_TRY(cudaGetLastError());
        CUDA_TRY(cudaDeviceSynchronize());

        CUDA_TRY(cudaMemcpy(wire_out, d_wire, count * wire_stride, cudaMemcpyDeviceToHost));

        cudaFree(d_wire); cudaFree(d_sizes); cudaFree(d_pt);
        cudaFree(d_nonces); cudaFree(d_keys);
        clear_error();
        return GpuError::Ok;
    }

    GpuError bip324_aead_decrypt_batch(
        const uint8_t* keys32, const uint8_t* nonces12,
        const uint8_t* wire_in, const uint32_t* sizes,
        uint32_t max_payload, size_t count,
        uint8_t* plaintext_out, uint8_t* out_valid) override
    {
        if (!ready_) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!keys32 || !nonces12 || !wire_in || !sizes || !plaintext_out || !out_valid)
            return set_error(GpuError::NullArg, "NULL buffer");

        const size_t wire_stride = max_payload + 19;

        uint8_t*    d_keys = nullptr;
        uint8_t*    d_nonces = nullptr;
        uint8_t*    d_wire = nullptr;
        uint32_t*   d_sizes = nullptr;
        uint8_t*    d_pt = nullptr;
        uint32_t*   d_ok = nullptr;

        CUDA_TRY(cudaMalloc(&d_keys,   count * 32));
        CUDA_TRY(cudaMalloc(&d_nonces, count * 12));
        CUDA_TRY(cudaMalloc(&d_wire,   count * wire_stride));
        CUDA_TRY(cudaMalloc(&d_sizes,  count * sizeof(uint32_t)));
        CUDA_TRY(cudaMalloc(&d_pt,     count * max_payload));
        CUDA_TRY(cudaMalloc(&d_ok,     count * sizeof(uint32_t)));

        CUDA_TRY(cudaMemcpy(d_keys,   keys32,   count * 32,               cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_nonces, nonces12, count * 12,               cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_wire,   wire_in,  count * wire_stride,      cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_sizes,  sizes,    count * sizeof(uint32_t), cudaMemcpyHostToDevice));

        int threads = 128;
        int blocks  = (static_cast<int>(count) + threads - 1) / threads;
        secp256k1::cuda::bip324::bip324_aead_decrypt_kernel<<<blocks, threads>>>(
            d_keys, d_nonces, d_wire, d_sizes, d_pt, d_ok,
            max_payload, static_cast<int>(count));
        CUDA_TRY(cudaGetLastError());
        CUDA_TRY(cudaDeviceSynchronize());

        CUDA_TRY(cudaMemcpy(plaintext_out, d_pt, count * max_payload, cudaMemcpyDeviceToHost));

        {
            std::vector<uint32_t> h_ok(count);
            CUDA_TRY(cudaMemcpy(h_ok.data(), d_ok, count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < count; ++i)
                out_valid[i] = h_ok[i] ? 1 : 0;
        }

        cudaFree(d_ok); cudaFree(d_pt); cudaFree(d_sizes);
        cudaFree(d_wire); cudaFree(d_nonces); cudaFree(d_keys);
        clear_error();
        return GpuError::Ok;
    }

    GpuError snark_witness_batch(
        const uint8_t* msg_hashes32, const uint8_t* pubkeys33,
        const uint8_t* sigs64, size_t count,
        uint8_t* out_flat) override
    {
        if (!ready_) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!msg_hashes32 || !pubkeys33 || !sigs64 || !out_flat)
            return set_error(GpuError::NullArg, "NULL buffer");

        /* Prepare signatures on host */
        std::vector<ECDSASignatureGPU> h_sigs(count);
        for (size_t i = 0; i < count; ++i)
            bytes_to_ecdsa_sig(sigs64 + i * 64, &h_sigs[i]);

        /* Allocate device memory */
        uint8_t*               d_msgs    = nullptr;
        uint8_t*               d_pubs33  = nullptr;
        JacobianPoint*         d_pubs    = nullptr;
        bool*                  d_pub_ok  = nullptr;
        ECDSASignatureGPU*     d_sigs    = nullptr;
        EcdsaSnarkWitnessFlat* d_out     = nullptr;

        CUDA_TRY(cudaMalloc(&d_msgs,   count * 32));
        CUDA_TRY(cudaMalloc(&d_pubs33, count * 33));
        CUDA_TRY(cudaMalloc(&d_pubs,   count * sizeof(JacobianPoint)));
        CUDA_TRY(cudaMalloc(&d_pub_ok, count * sizeof(bool)));
        CUDA_TRY(cudaMalloc(&d_sigs,   count * sizeof(ECDSASignatureGPU)));
        CUDA_TRY(cudaMalloc(&d_out,    count * sizeof(EcdsaSnarkWitnessFlat)));

        CUDA_TRY(cudaMemcpy(d_msgs,   msg_hashes32,    count * 32,                         cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_pubs33, pubkeys33,        count * 33,                         cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_sigs,   h_sigs.data(),   count * sizeof(ECDSASignatureGPU),  cudaMemcpyHostToDevice));

        /* Decompress pubkeys on GPU */
        int threads = 128;
        int blocks  = (static_cast<int>(count) + threads - 1) / threads;
        batch_compressed_to_jac_kernel<<<blocks, threads>>>(
            d_pubs33, d_pubs, d_pub_ok, static_cast<int>(count));
        CUDA_TRY(cudaGetLastError());

        /* Launch witness computation */
        cuda::ecdsa_snark_witness_batch_kernel<<<blocks, threads>>>(
            d_msgs, d_pubs, d_sigs, d_out, static_cast<int>(count));
        CUDA_TRY(cudaGetLastError());
        CUDA_TRY(cudaDeviceSynchronize());

        /* Download flat witness records */
        CUDA_TRY(cudaMemcpy(out_flat, d_out,
                            count * sizeof(EcdsaSnarkWitnessFlat),
                            cudaMemcpyDeviceToHost));

        cudaFree(d_out);
        cudaFree(d_sigs);
        cudaFree(d_pub_ok);
        cudaFree(d_pubs);
        cudaFree(d_pubs33);
        cudaFree(d_msgs);
        clear_error();
        return GpuError::Ok;
    }

    /* -- BIP-340 Schnorr SNARK witness GPU batch (eprint 2025/695) --------- */

    GpuError schnorr_snark_witness_batch(
        const uint8_t* msgs32, const uint8_t* pubkeys_x32,
        const uint8_t* sigs64, size_t count,
        uint8_t* out_flat) override
    {
        if (!ready_) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!msgs32 || !pubkeys_x32 || !sigs64 || !out_flat)
            return set_error(GpuError::NullArg, "NULL buffer");

        uint8_t*                      d_msgs   = nullptr;
        uint8_t*                      d_pubs   = nullptr;
        uint8_t*                      d_sigs   = nullptr;
        cuda::SchnorrSnarkWitnessFlat* d_out   = nullptr;

        CUDA_TRY(cudaMalloc(&d_msgs, count * 32));
        CUDA_TRY(cudaMalloc(&d_pubs, count * 32));
        CUDA_TRY(cudaMalloc(&d_sigs, count * 64));
        CUDA_TRY(cudaMalloc(&d_out,  count * sizeof(cuda::SchnorrSnarkWitnessFlat)));

        CUDA_TRY(cudaMemcpy(d_msgs, msgs32,       count * 32, cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_pubs, pubkeys_x32,  count * 32, cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_sigs, sigs64,       count * 64, cudaMemcpyHostToDevice));

        int threads = 128;
        int blocks  = (static_cast<int>(count) + threads - 1) / threads;
        cuda::schnorr_snark_witness_batch_kernel<<<blocks, threads>>>(
            d_msgs, d_pubs, d_sigs, d_out, static_cast<int>(count));
        CUDA_TRY(cudaGetLastError());
        CUDA_TRY(cudaDeviceSynchronize());

        CUDA_TRY(cudaMemcpy(out_flat, d_out,
                            count * sizeof(cuda::SchnorrSnarkWitnessFlat),
                            cudaMemcpyDeviceToHost));

        cudaFree(d_out);
        cudaFree(d_sigs);
        cudaFree(d_pubs);
        cudaFree(d_msgs);
        clear_error();
        return GpuError::Ok;
    }

    /* -- BIP-352 Silent Payment GPU batch scan ----------------------------- */

    GpuError bip352_scan_batch(
        const uint8_t  scan_privkey32[32],
        const uint8_t  spend_pubkey33[33],
        const uint8_t* tweak_pubkeys33,
        size_t n_tweaks,
        uint64_t* prefix64_out) override
    {
        if (!ready_) return set_error(GpuError::Device, "context not initialised");
        if (n_tweaks == 0) { clear_error(); return GpuError::Ok; }
        if (!scan_privkey32 || !spend_pubkey33 || !tweak_pubkeys33 || !prefix64_out)
            return set_error(GpuError::NullArg, "NULL buffer");

        /* -- 1. Convert scan key to Scalar on CPU -- */
        cuda::Scalar h_scan_k;
        for (int limb = 0; limb < 4; ++limb) {
            uint64_t v = 0;
            int base = (3 - limb) * 8;
            for (int b = 0; b < 8; ++b) v = (v << 8) | scan_privkey32[base + b];
            h_scan_k.limbs[limb] = v;
        }

        /* -- 2. Decompress tweak pubkeys on CPU, then upload as JacobianPoint -- */
        std::vector<cuda::JacobianPoint> h_tweaks(n_tweaks);
        {
            uint8_t* d_pubs33_tmp = nullptr;
            cuda::JacobianPoint* d_pubs_tmp = nullptr;
            bool* d_ok_tmp = nullptr;
            CUDA_TRY(cudaMalloc(&d_pubs33_tmp, n_tweaks * 33));
            CUDA_TRY(cudaMalloc(&d_pubs_tmp,   n_tweaks * sizeof(cuda::JacobianPoint)));
            CUDA_TRY(cudaMalloc(&d_ok_tmp,     n_tweaks * sizeof(bool)));
            CUDA_TRY(cudaMemcpy(d_pubs33_tmp, tweak_pubkeys33, n_tweaks * 33, cudaMemcpyHostToDevice));

            int threads = 128;
            int blocks  = (static_cast<int>(n_tweaks) + threads - 1) / threads;
            batch_compressed_to_jac_kernel<<<blocks, threads>>>(
                d_pubs33_tmp, d_pubs_tmp, d_ok_tmp, static_cast<int>(n_tweaks));
            CUDA_TRY(cudaGetLastError());
            CUDA_TRY(cudaDeviceSynchronize());
            CUDA_TRY(cudaMemcpy(h_tweaks.data(), d_pubs_tmp,
                                n_tweaks * sizeof(cuda::JacobianPoint),
                                cudaMemcpyDeviceToHost));
            cudaFree(d_ok_tmp);
            cudaFree(d_pubs_tmp);
            cudaFree(d_pubs33_tmp);
        }

        /* -- 3. Decompress spend pubkey on GPU -> AffinePoint -- */
        cuda::AffinePoint h_spend_aff{};
        {
            uint8_t* d_spend33 = nullptr;
            cuda::JacobianPoint* d_spend_jac = nullptr;
            bool* d_ok2 = nullptr;
            CUDA_TRY(cudaMalloc(&d_spend33,   33));
            CUDA_TRY(cudaMalloc(&d_spend_jac, sizeof(cuda::JacobianPoint)));
            CUDA_TRY(cudaMalloc(&d_ok2,       sizeof(bool)));
            CUDA_TRY(cudaMemcpy(d_spend33, spend_pubkey33, 33, cudaMemcpyHostToDevice));
            batch_compressed_to_jac_kernel<<<1, 1>>>(d_spend33, d_spend_jac, d_ok2, 1);
            CUDA_TRY(cudaGetLastError());
            CUDA_TRY(cudaDeviceSynchronize());
            cuda::JacobianPoint h_jac{};
            CUDA_TRY(cudaMemcpy(&h_jac, d_spend_jac, sizeof(cuda::JacobianPoint), cudaMemcpyDeviceToHost));
            cudaFree(d_ok2);
            cudaFree(d_spend_jac);
            cudaFree(d_spend33);
            h_spend_aff.x = h_jac.x;
            h_spend_aff.y = h_jac.y;
        }

        /* -- 4. Allocate device buffers and upload -- */
        cuda::JacobianPoint* d_tweaks   = nullptr;
        cuda::Scalar*        d_scan_k   = nullptr;
        cuda::AffinePoint*   d_spend    = nullptr;
        uint64_t*            d_prefixes = nullptr;

        CUDA_TRY(cudaMalloc(&d_tweaks,   n_tweaks * sizeof(cuda::JacobianPoint)));
        CUDA_TRY(cudaMalloc(&d_scan_k,   sizeof(cuda::Scalar)));
        CUDA_TRY(cudaMalloc(&d_spend,    sizeof(cuda::AffinePoint)));
        CUDA_TRY(cudaMalloc(&d_prefixes, n_tweaks * sizeof(uint64_t)));

        CUDA_TRY(cudaMemcpy(d_tweaks, h_tweaks.data(),
                            n_tweaks * sizeof(cuda::JacobianPoint), cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_scan_k, &h_scan_k, sizeof(cuda::Scalar), cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_spend,  &h_spend_aff, sizeof(cuda::AffinePoint), cudaMemcpyHostToDevice));

        /* -- 5. Launch BIP-352 scan kernel -- */
        int threads = 128;
        int blocks  = (static_cast<int>(n_tweaks) + threads - 1) / threads;
        bip352_scan_batch_kernel<<<blocks, threads>>>(
            d_tweaks, d_scan_k, d_spend, d_prefixes, static_cast<int>(n_tweaks));
        CUDA_TRY(cudaGetLastError());
        CUDA_TRY(cudaDeviceSynchronize());

        /* -- 6. Download prefixes -- */
        CUDA_TRY(cudaMemcpy(prefix64_out, d_prefixes,
                            n_tweaks * sizeof(uint64_t), cudaMemcpyDeviceToHost));

        /* -- 7. Zeroize secret material before releasing device/host memory
         *       HIGH-3: scan private key must not remain in device memory
         *       after the operation completes. */
        cudaMemset(d_scan_k, 0, sizeof(cuda::Scalar));
        secp256k1::detail::secure_erase(&h_scan_k, sizeof(h_scan_k));

        cudaFree(d_prefixes);
        cudaFree(d_spend);
        cudaFree(d_scan_k);
        cudaFree(d_tweaks);
        clear_error();
        return GpuError::Ok;
    }

private:
    bool       ready_      = false;
    uint32_t   device_idx_ = 0;
    GpuError   last_err_   = GpuError::Ok;
    char       last_msg_[256] = {};

    /* -- Persistent MSM device buffer pool (avoids per-call cudaMalloc) --- */
    struct MsmPool {
        size_t         capacity    = 0;
        Scalar*        d_scalars   = nullptr;
        uint8_t*       d_pts33     = nullptr;
        JacobianPoint* d_points    = nullptr;
        bool*          d_pt_ok     = nullptr;
        JacobianPoint* d_partials  = nullptr;
        JacobianPoint* d_blk_parts = nullptr;
        uint8_t*       d_out33     = nullptr;
        bool*          d_ok        = nullptr;

        void free_all() {
            if (d_scalars)   { cudaFree(d_scalars);   d_scalars   = nullptr; }
            if (d_pts33)     { cudaFree(d_pts33);     d_pts33     = nullptr; }
            if (d_points)    { cudaFree(d_points);    d_points    = nullptr; }
            if (d_pt_ok)     { cudaFree(d_pt_ok);     d_pt_ok     = nullptr; }
            if (d_partials)  { cudaFree(d_partials);  d_partials  = nullptr; }
            if (d_blk_parts) { cudaFree(d_blk_parts); d_blk_parts = nullptr; }
            if (d_out33)     { cudaFree(d_out33);     d_out33     = nullptr; }
            if (d_ok)        { cudaFree(d_ok);        d_ok        = nullptr; }
            capacity = 0;
        }

        /* Ensure pool is at least n elements wide.
         * Grows by 2x to amortise reallocation cost. */
        cudaError_t ensure(size_t n) {
            if (n <= capacity) return cudaSuccess;
            free_all();
            /* Round up to the next power-of-two ≥ max(n, 256) for
             * block-reduce alignment and a sensible minimum pool size. */
            size_t cap = 1;
            size_t const target = n < 256 ? 256 : n;
            while (cap < target) cap <<= 1;
            constexpr int kBlk = 256;
            const size_t n_blks = (cap + kBlk - 1) / kBlk;

            cudaError_t e;
#define ALLOC(ptr, bytes) e = cudaMalloc(&(ptr), (bytes)); if (e != cudaSuccess) { free_all(); return e; }
            ALLOC(d_scalars,   cap  * sizeof(Scalar));
            ALLOC(d_pts33,     cap  * 33);
            ALLOC(d_points,    cap  * sizeof(JacobianPoint));
            ALLOC(d_pt_ok,     cap  * sizeof(bool));
            ALLOC(d_partials,  cap  * sizeof(JacobianPoint));
            ALLOC(d_blk_parts, n_blks * sizeof(JacobianPoint));
            ALLOC(d_out33,     33);
            ALLOC(d_ok,        sizeof(bool));
#undef ALLOC
            capacity = cap;
            return cudaSuccess;
        }
    } msm_pool_;

    GpuError set_error(GpuError err, const char* msg) {
        last_err_ = err;
        if (msg) {
            size_t i = 0;
            for (; i < sizeof(last_msg_) - 1 && msg[i]; ++i)
                last_msg_[i] = msg[i];
            last_msg_[i] = '\0';
        } else {
            last_msg_[0] = '\0';
        }
        return err;
    }

    void clear_error() {
        last_err_ = GpuError::Ok;
        last_msg_[0] = '\0';
    }

    /* -- Type conversion helpers (host-side) ------------------------------- */

    /** Big-endian 32 bytes → Scalar (4x uint64_t LE limbs) */
    static void bytes_to_scalar(const uint8_t be[32], Scalar* out) {
        for (int limb = 0; limb < 4; ++limb) {
            uint64_t v = 0;
            for (int b = 0; b < 8; ++b) {
                v = (v << 8) | be[(3 - limb) * 8 + b];
            }
            out->limbs[limb] = v;
        }
    }

    /** Scalar (4x uint64_t LE limbs) → big-endian 32 bytes */
    static void scalar_to_bytes(const Scalar* s, uint8_t be[32]) {
        for (int limb = 0; limb < 4; ++limb) {
            uint64_t v = s->limbs[limb];
            for (int b = 0; b < 8; ++b) {
                be[31 - limb * 8 - b] = static_cast<uint8_t>(v >> (b * 8));
            }
        }
    }

    /** Big-endian 32 bytes → FieldElement (4x uint64_t LE limbs) */
    static void bytes_to_field(const uint8_t be[32], FieldElement* out) {
        for (int limb = 0; limb < 4; ++limb) {
            uint64_t v = 0;
            for (int b = 0; b < 8; ++b) {
                v = (v << 8) | be[(3 - limb) * 8 + b];
            }
            out->limbs[limb] = v;
        }
    }

    static void field_to_bytes(const FieldElement* fe, uint8_t be[32]) {
        for (int limb = 0; limb < 4; ++limb) {
            uint64_t v = fe->limbs[limb];
            for (int b = 0; b < 8; ++b) {
                be[31 - limb * 8 - b] = static_cast<uint8_t>(v >> (b * 8));
            }
        }
    }

    /** Compact ECDSA sig (64 bytes: R[32] || S[32], big-endian) → ECDSASignatureGPU */
    static void bytes_to_ecdsa_sig(const uint8_t compact[64], ECDSASignatureGPU* out) {
        bytes_to_scalar(compact, &out->r);
        bytes_to_scalar(compact + 32, &out->s);
    }

    /** Schnorr sig (64 bytes: r[32] || s[32], big-endian) → SchnorrSignatureGPU */
    static void bytes_to_schnorr_sig(const uint8_t sig[64], SchnorrSignatureGPU* out) {
        /* r is raw bytes (x-coordinate), s is a scalar */
        std::memcpy(out->r, sig, 32);
        bytes_to_scalar(sig + 32, &out->s);
    }

    /* NOTE: compressed_to_jacobian_host and jacobian_to_compressed_host removed.
     * All Jacobian <-> compressed conversions now happen on-device via
     * batch_jac_to_compressed_kernel / batch_compressed_to_jac_kernel,
     * eliminating the host-side CPU FieldElement normalization mismatch. */
};

/* -- Factory function ------------------------------------------------------ */
std::unique_ptr<GpuBackend> create_cuda_backend() {
    return std::make_unique<CudaBackend>();
}

} // namespace gpu
} // namespace secp256k1
