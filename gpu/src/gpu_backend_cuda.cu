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
} // namespace cuda

namespace gpu {

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

/** MSM single-thread reduction: sum N Jacobian partials → 1 compressed point.
 *  Runs on a single GPU thread to avoid host-side field arithmetic. */
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

        /* Check for failures */
        bool* h_ok = new bool[count];
        CUDA_TRY(cudaMemcpy(h_ok, d_ok, count * sizeof(bool), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < count; ++i) {
            if (!h_ok[i]) std::memset(out_secrets32 + i * 32, 0, 32);
        }
        delete[] h_ok;

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

        /* Allocate device memory */
        Scalar*        d_scalars  = nullptr;
        uint8_t*       d_pts33   = nullptr;
        JacobianPoint* d_points   = nullptr;
        bool*          d_pt_ok    = nullptr;
        JacobianPoint* d_partials = nullptr;
        uint8_t*       d_out33    = nullptr;
        bool*          d_ok       = nullptr;

        CUDA_TRY(cudaMalloc(&d_scalars, n * sizeof(Scalar)));
        CUDA_TRY(cudaMalloc(&d_pts33, n * 33));
        CUDA_TRY(cudaMalloc(&d_points, n * sizeof(JacobianPoint)));
        CUDA_TRY(cudaMalloc(&d_pt_ok, n * sizeof(bool)));
        CUDA_TRY(cudaMalloc(&d_partials, n * sizeof(JacobianPoint)));
        CUDA_TRY(cudaMalloc(&d_out33, 33));
        CUDA_TRY(cudaMalloc(&d_ok, sizeof(bool)));

        CUDA_TRY(cudaMemcpy(d_scalars, h_scalars.data(), n * sizeof(Scalar), cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_pts33, points33, n * 33, cudaMemcpyHostToDevice));

        /* Decompress points on GPU */
        int threads = 128;
        int blocks  = (static_cast<int>(n) + threads - 1) / threads;
        batch_compressed_to_jac_kernel<<<blocks, threads>>>(
            d_pts33, d_points, d_pt_ok, static_cast<int>(n));
        CUDA_TRY(cudaGetLastError());

        /* Phase 1: scatter — each thread computes scalars[i] * points[i] */
        msm_scatter_kernel<<<blocks, threads>>>(d_scalars, d_points, d_partials, static_cast<int>(n));
        CUDA_TRY(cudaGetLastError());

        /* Phase 2: reduce + compress on GPU (single thread) */
        msm_reduce_and_compress_kernel<<<1, 1>>>(d_partials, static_cast<int>(n), d_out33, d_ok);
        CUDA_TRY(cudaGetLastError());
        CUDA_TRY(cudaDeviceSynchronize());

        /* Download compressed result */
        CUDA_TRY(cudaMemcpy(out_result33, d_out33, 33, cudaMemcpyDeviceToHost));
        bool result_ok;
        CUDA_TRY(cudaMemcpy(&result_ok, d_ok, sizeof(bool), cudaMemcpyDeviceToHost));

        cudaFree(d_ok);
        cudaFree(d_out33);
        cudaFree(d_partials);
        cudaFree(d_pt_ok);
        cudaFree(d_points);
        cudaFree(d_pts33);
        cudaFree(d_scalars);

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
            bool* h_res_raw = new bool[count];
            CUDA_TRY(cudaMemcpy(h_res_raw, d_res, count * sizeof(bool), cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < count; ++i)
                out_results[i] = h_res_raw[i] ? 1 : 0;
            delete[] h_res_raw;
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
            bool* h_res_raw = new bool[count];
            CUDA_TRY(cudaMemcpy(h_res_raw, d_res, count * sizeof(bool), cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < count; ++i)
                out_results[i] = h_res_raw[i] ? 1 : 0;
            delete[] h_res_raw;
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

private:
    bool       ready_      = false;
    uint32_t   device_idx_ = 0;
    GpuError   last_err_   = GpuError::Ok;
    char       last_msg_[256] = {};

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
