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

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <vector>
#include <limits>

/* -- Secure erase for private key zeroization ------------------------------ */
#include "secp256k1/detail/secure_erase.hpp"
#include "secp256k1/scalar.hpp"

/* -- CUDA runtime ---------------------------------------------------------- */
#include <cuda_runtime.h>

/* -- Existing CUDA headers (Layer 1) --------------------------------------- */
#include "secp256k1.cuh"
#include "ecdh.cuh"
#include "msm.cuh"
#include "schnorr.cuh"
#if SECP256K1_GPU_HAS_ZK
#include "zk.cuh"
#endif
#if SECP256K1_GPU_HAS_BIP324
#include "bip324.cuh"
#endif
#include "gpu_compat.h"
/* CT variable-base scalar mul for BIP-352 scan kernel (Rule 8) */
#include "ct/ct_point.cuh"

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

// Rule 10: RAII guard that zeroes a device buffer before freeing it.
// Ensures private key material is erased even when CUDA_TRY causes early return.
struct CudaKeyGuard {
    void*  ptr  = nullptr;
    size_t size = 0;
    CudaKeyGuard() = default;
    CudaKeyGuard(void* p, size_t s) : ptr(p), size(s) {}
    CudaKeyGuard(const CudaKeyGuard&) = delete;
    CudaKeyGuard& operator=(const CudaKeyGuard&) = delete;
    ~CudaKeyGuard() {
        if (ptr) {
            cudaMemset(ptr, 0, size);
            cudaFree(ptr);
        }
    }
    // Release ownership without zeroize (for non-key device buffers).
    void* release() { void* p = ptr; ptr = nullptr; return p; }
};

// Owns a non-secret device allocation for the remainder of the current scope.
// Construct immediately after cudaMalloc succeeds so CUDA_TRY early returns
// cannot leak allocations made earlier in the operation.
struct CudaBufferGuard {
    void* ptr = nullptr;
    explicit CudaBufferGuard(void* p) : ptr(p) {}
    CudaBufferGuard(const CudaBufferGuard&) = delete;
    CudaBufferGuard& operator=(const CudaBufferGuard&) = delete;
    ~CudaBufferGuard() {
        if (ptr) cudaFree(ptr);
    }
};

namespace secp256k1 {

/* Kernels defined in cuda/src/secp256k1.cu (namespace secp256k1::cuda).
   Must be declared in the same namespace so the linker finds them. */
namespace cuda {
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
extern __global__ void ecdsa_verify_collect_kernel(
    const uint8_t* __restrict__ msg_hashes,
    const JacobianPoint* __restrict__ public_keys,
    const ECDSASignatureGPU* __restrict__ sigs,
    uint8_t*       __restrict__ key_cells,
    int count);
extern __global__ void schnorr_verify_collect_kernel(
    const uint8_t* __restrict__ pubkeys_x,
    const uint8_t* __restrict__ msgs,
    const SchnorrSignatureGPU* __restrict__ sigs,
    uint8_t*       __restrict__ key_cells,
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
#if SECP256K1_GPU_HAS_BIP352

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
 *    1. shared    = scan_k × tweak_pts[idx]   (CT: Rule 8 — scan_k is secret)
 *    2. ser37     = compress(shared) ∥ [0,0,0,0]
 *    3. hash      = SHA256_tagged(ser37)       (hash output is public)
 *    4. output    = hash × G                  (hash is public — scalar_mul_generator_const OK)
 *    5. cand      = spend_aff + output         (public point arithmetic)
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

    /* 1. shared = scan_k × tweak_pts[idx]
     * Rule 8: scan_k is the BIP-352 scan private key — must use CT variable-base
     * scalar mul. tweak_pts are public input pubkey sums (z==1, decompressed).
     * Steps 3-6 use hash-derived / public scalars — scalar_mul_generator_const OK. */
    JacobianPoint shared;
    secp256k1::cuda::ct::ct_scalar_mul_varbase(&tweak_pts[idx], scan_k, &shared);
    if (shared.infinity) { prefixes[idx] = 0; return; }

    /* 2. Serialize shared secret to 37 bytes: [prefix_byte][x32][0,0,0,0] */
    uint8_t ser37[37];
    point_to_compressed(&shared, ser37);   // writes 33 bytes directly; avoids comp33 copy
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

/** GPU kernel: decompress an array of SEC1 compressed pubkeys into
 *  AffinePoint + validity flag. Used to decode the (small, O(n_spend))
 *  spend-key candidate array ONCE per bip352_scan_batch_multispend() call,
 *  independent of n_tweaks, reusing the same on-GPU decompression primitive
 *  already used/validated for tweak pubkeys (point_from_compressed). */
__global__ void bip352_decompress_points_kernel(
    const uint8_t* __restrict__ pubkeys33,
    secp256k1::cuda::AffinePoint* __restrict__ out_aff,
    uint8_t* __restrict__ out_valid,
    int n)
{
    using namespace secp256k1::cuda;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    JacobianPoint p;
    if (point_from_compressed(pubkeys33 + idx * 33, &p)) {
        out_aff[idx].x = p.x;
        out_aff[idx].y = p.y;
        out_valid[idx] = 1;
    } else {
        out_aff[idx].x = FieldElement{};
        out_aff[idx].y = FieldElement{};
        out_valid[idx] = 0;
    }
}

/** BIP-352 multi-spend-key scan kernel — 1 thread per tweak.
 *
 *  Generalizes bip352_scan_batch_kernel_compressed() (issue #335): the ECDH
 *  scalar-mul (CT, Rule 8), tagged hash, and hash×G run EXACTLY ONCE per
 *  thread/tweak (identical to the single-spend kernel); only the final mixed
 *  point addition + prefix extraction is repeated, once per spend-key
 *  candidate. spend_aff/spend_valid are precomputed once per call (not per
 *  tweak) by bip352_decompress_points_kernel above.
 *
 *  Output: prefixes[tid * n_spend + j] for spend candidate j, row-major
 *  (tweak-major) -- matches the single-spend layout exactly when n_spend==1. */
__global__ void bip352_scan_batch_multispend_kernel_compressed(
    const uint8_t* __restrict__ tweaks33,
    const secp256k1::cuda::Scalar* __restrict__ scan_k,
    const secp256k1::cuda::AffinePoint* __restrict__ spend_aff,
    const uint8_t* __restrict__ spend_valid,
    int n_spend,
    uint64_t* __restrict__ prefixes,
    int n_tweaks)
{
    using namespace secp256k1::cuda;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_tweaks) return;

    uint64_t* row = prefixes + (size_t)idx * (size_t)n_spend;

    // Decompress tweak pubkey on GPU (same primitive as single-spend kernel).
    JacobianPoint tweak;
    if (!point_from_compressed(tweaks33 + idx * 33, &tweak)) {
        for (int j = 0; j < n_spend; ++j) row[j] = 0;
        return;
    }

    // shared = scan_k × tweak (CT scalar mul) -- ONCE per tweak (Rule 8).
    JacobianPoint shared;
    ct::ct_scalar_mul_varbase(&tweak, scan_k, &shared);
    if (shared.infinity) {
        for (int j = 0; j < n_spend; ++j) row[j] = 0;
        return;
    }

    // Serialize -> tagged hash -> hash x G -- ONCE per tweak (public data).
    uint8_t ser37[37];
    point_to_compressed(&shared, ser37);
    ser37[33] = ser37[34] = ser37[35] = ser37[36] = 0;
    uint8_t hash[32];
    bip352_tagged_hash_37(ser37, hash);
    Scalar hs;
    scalar_from_bytes(hash, &hs);
    JacobianPoint gen_out;
    scalar_mul_generator_const(&hs, &gen_out);

    // Per spend-key candidate: one mixed point add + prefix extract.
    for (int j = 0; j < n_spend; ++j) {
        if (!spend_valid[j]) { row[j] = 0; continue; }
        JacobianPoint cand;
        jacobian_add_mixed(&gen_out, &spend_aff[j], &cand);
        if (cand.infinity) { row[j] = 0; continue; }
        FieldElement ax, ay;
        jacobian_to_affine(&cand, &ax, &ay);
        (void)ay;
        uint8_t xb[32];
        field_to_bytes(&ax, xb);
        uint64_t pref = 0;
        for (int b = 0; b < 8; ++b) pref = (pref << 8) | xb[b];
        row[j] = pref;
    }
}
#endif  // SECP256K1_GPU_HAS_BIP352

} // anonymous namespace

/* Import CUDA types (FieldElement, Scalar, JacobianPoint, etc.) and kernels
   from the secp256k1::cuda namespace into secp256k1::gpu. */
using namespace cuda;

namespace {

struct CudaBatchScratch {
    std::vector<uint8_t> result_bytes;
    uint8_t* lbtc_rows = nullptr;
    uint8_t* lbtc_cols = nullptr;      // packed column staging: dig | pub/xon | sig
    bool* lbtc_results = nullptr;
    std::size_t lbtc_rows_bytes = 0;
    std::size_t lbtc_cols_bytes = 0;
    std::size_t lbtc_results_count = 0;

    // Reusable sighash-descriptor staging (grow-only, freed at thread exit).
    // Column data + var-lens are placed sequentially in one pool; output and
    // small metadata tables get their own reusable allocations.
    uint8_t*  lbtc_sighash_pool        = nullptr;
    std::size_t lbtc_sighash_pool_bytes = 0;
    uint8_t*  lbtc_sighash_out         = nullptr;
    std::size_t lbtc_sighash_out_bytes  = 0;
    uint8_t*  lbtc_sighash_meta        = nullptr;  // plan + ptrs + strides + vlptrs
    std::size_t lbtc_sighash_meta_bytes = 0;

    // Track which CUDA device this scratch was allocated on.
    // A context switch/recreate must never reuse pointers from another device.
    int staging_device_idx = -1;  // -1 = uninitialised

    ~CudaBatchScratch() {
        free_lbtc_device();
    }

    uint8_t* ensure_results(std::size_t count) {
        if (result_bytes.size() < count) {
            result_bytes.resize(count);
        }
        return result_bytes.data();
    }

    void free_lbtc_device() {
        if (lbtc_sighash_meta)   { cudaFree(lbtc_sighash_meta);   lbtc_sighash_meta = nullptr;   lbtc_sighash_meta_bytes = 0; }
        if (lbtc_sighash_out)    { cudaFree(lbtc_sighash_out);    lbtc_sighash_out = nullptr;    lbtc_sighash_out_bytes = 0; }
        if (lbtc_sighash_pool)   { cudaFree(lbtc_sighash_pool);   lbtc_sighash_pool = nullptr;   lbtc_sighash_pool_bytes = 0; }
        if (lbtc_results) {
            cudaFree(lbtc_results);
            lbtc_results = nullptr;
            lbtc_results_count = 0;
        }
        if (lbtc_cols) {
            cudaFree(lbtc_cols);
            lbtc_cols = nullptr;
            lbtc_cols_bytes = 0;
        }
        if (lbtc_rows) {
            cudaFree(lbtc_rows);
            lbtc_rows = nullptr;
            lbtc_rows_bytes = 0;
        }
    }

    cudaError_t ensure_lbtc_rows(std::size_t bytes) {
        int cur_dev = -1;
        if (cudaGetDevice(&cur_dev) == cudaSuccess && cur_dev != staging_device_idx) {
            free_lbtc_device();
            staging_device_idx = cur_dev;
        }
        if (bytes <= lbtc_rows_bytes) {
            return cudaSuccess;
        }
        if (lbtc_rows) {
            cudaFree(lbtc_rows);
            lbtc_rows = nullptr;
            lbtc_rows_bytes = 0;
        }
        const cudaError_t err = cudaMalloc(&lbtc_rows, bytes);
        if (err == cudaSuccess) {
            lbtc_rows_bytes = bytes;
        }
        return err;
    }

    // Grow-only device staging for the columnar verify paths, reused across
    // successive engine calls so the hot libbitcoin verify loop performs no
    // fresh per-call cudaMalloc/cudaFree (acceptance A6). Freed at thread exit.
    cudaError_t ensure_lbtc_cols(std::size_t bytes) {
        int cur_dev = -1;
        if (cudaGetDevice(&cur_dev) == cudaSuccess && cur_dev != staging_device_idx) {
            free_lbtc_device();
            staging_device_idx = cur_dev;
        }
        if (bytes <= lbtc_cols_bytes) {
            return cudaSuccess;
        }
        if (lbtc_cols) {
            cudaFree(lbtc_cols);
            lbtc_cols = nullptr;
            lbtc_cols_bytes = 0;
        }
        const cudaError_t err = cudaMalloc(&lbtc_cols, bytes);
        if (err == cudaSuccess) {
            lbtc_cols_bytes = bytes;
        }
        return err;
    }

    cudaError_t ensure_lbtc_results(std::size_t count) {
        int cur_dev = -1;
        if (cudaGetDevice(&cur_dev) == cudaSuccess && cur_dev != staging_device_idx) {
            free_lbtc_device();
            staging_device_idx = cur_dev;
        }
        if (count <= lbtc_results_count) {
            return cudaSuccess;
        }
        if (lbtc_results) {
            cudaFree(lbtc_results);
            lbtc_results = nullptr;
            lbtc_results_count = 0;
        }
        const cudaError_t err = cudaMalloc(&lbtc_results, count * sizeof(bool));
        if (err == cudaSuccess) {
            lbtc_results_count = count;
        }
        return err;
    }

    // Grow-only sighash pool: holds column data + var-lens for all fields.
    cudaError_t ensure_lbtc_sighash_pool(std::size_t bytes) {
        // If the CUDA device changed since last allocation, free old buffers
        // so we never reuse pointers from another device/context.
        int cur_dev = -1;
        if (cudaGetDevice(&cur_dev) == cudaSuccess && cur_dev != staging_device_idx) {
            free_lbtc_device();
            staging_device_idx = cur_dev;
        }
        if (bytes <= lbtc_sighash_pool_bytes) return cudaSuccess;
        if (lbtc_sighash_pool) { cudaFree(lbtc_sighash_pool); lbtc_sighash_pool = nullptr; lbtc_sighash_pool_bytes = 0; }
        const cudaError_t err = cudaMalloc(&lbtc_sighash_pool, bytes);
        if (err == cudaSuccess) lbtc_sighash_pool_bytes = bytes;
        return err;
    }

    cudaError_t ensure_lbtc_sighash_out(std::size_t bytes) {
        int cur_dev = -1;
        if (cudaGetDevice(&cur_dev) == cudaSuccess && cur_dev != staging_device_idx) {
            free_lbtc_device();
            staging_device_idx = cur_dev;
        }
        if (bytes <= lbtc_sighash_out_bytes) return cudaSuccess;
        if (lbtc_sighash_out) { cudaFree(lbtc_sighash_out); lbtc_sighash_out = nullptr; lbtc_sighash_out_bytes = 0; }
        const cudaError_t err = cudaMalloc(&lbtc_sighash_out, bytes);
        if (err == cudaSuccess) lbtc_sighash_out_bytes = bytes;
        return err;
    }

    cudaError_t ensure_lbtc_sighash_meta(std::size_t bytes) {
        int cur_dev = -1;
        if (cudaGetDevice(&cur_dev) == cudaSuccess && cur_dev != staging_device_idx) {
            free_lbtc_device();
            staging_device_idx = cur_dev;
        }
        if (bytes <= lbtc_sighash_meta_bytes) return cudaSuccess;
        if (lbtc_sighash_meta) { cudaFree(lbtc_sighash_meta); lbtc_sighash_meta = nullptr; lbtc_sighash_meta_bytes = 0; }
        const cudaError_t err = cudaMalloc(&lbtc_sighash_meta, bytes);
        if (err == cudaSuccess) lbtc_sighash_meta_bytes = bytes;
        return err;
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

__device__ __forceinline__ uint64_t lbtc_load_le64(const uint8_t* p)
{
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i)
        v |= static_cast<uint64_t>(p[i]) << (i * 8);
    return v;
}

// The device point_from_compressed() (secp256k1.cuh) parses the x-coordinate with
// the NON-strict field_from_bytes(), which silently reduces x mod p. The CPU
// reference (decompress_compressed_pubkey) uses parse_bytes_strict and REJECTS any
// x >= p. Without this guard the GPU would accept a compressed key whose 32 x-bytes
// encode a value >= p (e.g. x_valid + p) that the CPU rejects — a differential
// false-accept. Validate x < p on the raw big-endian bytes before decompression.
__device__ __forceinline__ bool lbtc_compressed_x_lt_p(const uint8_t* pub33)
{
    FieldElement x;
    return field_from_bytes_strict(pub33 + 1, &x);
}

__device__ __forceinline__ bool lbtc_scalar_ge_order(const Scalar* s)
{
    for (int i = 3; i >= 0; --i) {
        if (s->limbs[i] > ORDER[i]) return true;
        if (s->limbs[i] < ORDER[i]) return false;
    }
    return true;
}

__device__ __forceinline__ bool lbtc_parse_opaque_scalar(
    const uint8_t* opaque, Scalar* out)
{
    out->limbs[0] = lbtc_load_le64(opaque);
    out->limbs[1] = lbtc_load_le64(opaque + 8);
    out->limbs[2] = lbtc_load_le64(opaque + 16);
    out->limbs[3] = lbtc_load_le64(opaque + 24);
    return !scalar_is_zero(out) && !lbtc_scalar_ge_order(out);
}

__device__ __forceinline__ bool lbtc_parse_opaque_signature(
    const uint8_t* opaque, ECDSASignatureGPU* sig)
{
    if (!lbtc_parse_opaque_scalar(opaque, &sig->r)) return false;
    if (!lbtc_parse_opaque_scalar(opaque + 32, &sig->s)) return false;
    // Libbitcoin consensus/direct opaque verification accepts both low-S and
    // high-S ECDSA forms. Low-S standardness belongs above this path; the device
    // parser must reject only non-scalars/zero scalars, not a valid high-S twin.
    return true;
}

__global__ void ecdsa_verify_lbtc_rows_kernel(
    const uint8_t* __restrict__ rows,
    size_t stride,
    bool* __restrict__ results,
    int count)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const uint8_t* row = rows + static_cast<size_t>(idx) * stride;
    JacobianPoint pub;
    ECDSASignatureGPU sig;
    const bool ok =
        lbtc_compressed_x_lt_p(row + 32) &&
        point_from_compressed(row + 32, &pub) &&
        lbtc_parse_opaque_signature(row + 65, &sig);
    results[idx] = ok && ecdsa_verify(row, &pub, &sig);
}

// libbitcoin ECDSA COLUMN verify kernel — sibling of the row kernel above, but
// reads three separate column spans (digests32 | pubkeys33 | sigs64) instead of
// one interleaved row. Opaque-LE signature parse + 33-byte pubkey decompression
// both happen device-side (no host compact-sig staging). 64-bit offsets.
__global__ void ecdsa_verify_lbtc_columns_kernel(
    const uint8_t* __restrict__ digests32,
    const uint8_t* __restrict__ pubkeys33,
    const uint8_t* __restrict__ sigs64,
    bool* __restrict__ results,
    int count)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    JacobianPoint pub;
    ECDSASignatureGPU sig;
    const uint8_t* pub33 = pubkeys33 + static_cast<size_t>(idx) * 33;
    const bool ok =
        lbtc_compressed_x_lt_p(pub33) &&
        point_from_compressed(pub33, &pub) &&
        lbtc_parse_opaque_signature(sigs64 + static_cast<size_t>(idx) * 64, &sig);
    results[idx] = ok && ecdsa_verify(digests32 + static_cast<size_t>(idx) * 32, &pub, &sig);
}

// Device-side BIP-340 signature parse (byte-identical to the host bytes_to_schnorr_sig
// used by schnorr_verify_batch): r is the raw 32 big-endian R.x bytes; s is a
// 32-byte big-endian scalar loaded into 4 little-endian limbs. Keeping the parse
// on device removes the host pre-conversion for the column path.
__device__ __forceinline__ void lbtc_parse_bip340_sig(
    const uint8_t* sig64, SchnorrSignatureGPU* out)
{
#pragma unroll
    for (int i = 0; i < 32; ++i) out->r[i] = sig64[i];
#pragma unroll
    for (int limb = 0; limb < 4; ++limb) {
        uint64_t v = 0;
#pragma unroll
        for (int b = 0; b < 8; ++b)
            v = (v << 8) | static_cast<uint64_t>(sig64[32 + (3 - limb) * 8 + b]);
        out->s.limbs[limb] = v;
    }
}

// libbitcoin Schnorr COLUMN verify kernel — reads digests32 | xonly32 | sigs64
// (BIP-340) column spans and parses the signature device-side.
__global__ void schnorr_verify_lbtc_columns_kernel(
    const uint8_t* __restrict__ digests32,
    const uint8_t* __restrict__ xonly32,
    const uint8_t* __restrict__ sigs64,
    bool* __restrict__ results,
    int count)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    SchnorrSignatureGPU sig;
    lbtc_parse_bip340_sig(sigs64 + static_cast<size_t>(idx) * 64, &sig);
    // BIP-340 strict: reject s == 0 and s >= n before verifying so the column
    // verdict is byte-identical to the CPU reference (SchnorrSignature::parse_strict
    // via parse_schnorr_bip340_entry) and the OpenCL/Metal backends. Without this,
    // a malleated s' ≡ s (mod n) with s' >= n maps to the same point as a valid s
    // and would be a GPU-only false-accept (signature malleability / consensus split).
    if (scalar_is_zero(&sig.s) || lbtc_scalar_ge_order(&sig.s)) {
        results[idx] = false;
        return;
    }
    results[idx] = schnorr_verify(xonly32 + static_cast<size_t>(idx) * 32,
                                  digests32 + static_cast<size_t>(idx) * 32, &sig);
}

#if SECP256K1_GPU_HAS_MSM
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
#endif  // SECP256K1_GPU_HAS_MSM

#if SECP256K1_GPU_HAS_ECDH
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
#endif  // SECP256K1_GPU_HAS_ECDH

/* ============================================================================
 * libbitcoin-bridge per-item batch kernels (all PUBLIC data -> variable-time):
 *   - lbtc_xonly_validate_kernel : x-only key on-curve check (lift_x per key)
 *   - lbtc_commitment_kernel     : BIP-341 tweak-add-check, one thread per item
 *   - lbtc_tagged_hash_kernel    : Taproot tagged hash (multi-block SHA-256;
 *                                  the engine device sha256 caps at ~119B, and a
 *                                  TapBranch preimage is 128B, so a self-contained
 *                                  multi-block SHA-256 is used here)
 * ============================================================================ */
namespace {  // file-local

__device__ __forceinline__ uint32_t lbtc_rr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

// Block-streaming SHA-256: compresses full 64-byte blocks directly from `data`
// (no whole-message local copy), then finalizes with a <=128B tail buffer sized
// for padding + length only. O(1) local memory regardless of `len` (previously
// this copied the entire message into a fixed uint8_t blk[384], which silently
// overflowed for len > ~375 bytes; every existing caller happened to stay under
// that bound, but hash256_var must support multi-megabyte rows).
__device__ void lbtc_sha256(const uint8_t* data, int len, uint8_t out[32]) {
    const uint32_t K[64] = {
      0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
      0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
      0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
      0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
      0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
      0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
      0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
      0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};
    uint32_t h[8] = {0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
                     0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};
    int off = 0;
    // Stream full 64-byte blocks directly from `data` (global or any device
    // memory — CUDA device-function pointers work uniformly, no address-space
    // overload needed unlike OpenCL/Metal). No local copy of the whole message.
    while (off + 64 <= len) {
        const uint8_t* block = data + off;
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (static_cast<uint32_t>(block[i*4])<<24)|(static_cast<uint32_t>(block[i*4+1])<<16)|
                   (static_cast<uint32_t>(block[i*4+2])<<8)|block[i*4+3];
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = lbtc_rr32(w[i-15],7)^lbtc_rr32(w[i-15],18)^(w[i-15]>>3);
            uint32_t s1 = lbtc_rr32(w[i-2],17)^lbtc_rr32(w[i-2],19)^(w[i-2]>>10);
            w[i] = w[i-16]+s0+w[i-7]+s1;
        }
        uint32_t a=h[0],b2=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1=lbtc_rr32(e,6)^lbtc_rr32(e,11)^lbtc_rr32(e,25); uint32_t ch=(e&f)^(~e&g);
            uint32_t t1=hh+S1+ch+K[i]+w[i]; uint32_t S0=lbtc_rr32(a,2)^lbtc_rr32(a,13)^lbtc_rr32(a,22);
            uint32_t mj=(a&b2)^(a&c)^(b2&c); uint32_t t2=S0+mj;
            hh=g;g=f;f=e;e=d+t1;d=c;c=b2;b2=a;a=t1+t2;
        }
        h[0]+=a;h[1]+=b2;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
        off += 64;
    }
    // Final block(s): remaining tail bytes [off,len) + 0x80 + zero pad + 8-byte
    // big-endian bit length. This is at most 2 blocks (128 bytes) REGARDLESS of
    // len, unlike the old blk[384] which was sized for the whole message.
    const int tail_len = len - off;
    uint8_t finalblk[128];
    for (int i = 0; i < 128; ++i) finalblk[i] = 0;
    for (int i = 0; i < tail_len; ++i) finalblk[i] = data[off + i];
    finalblk[tail_len] = 0x80;
    const uint64_t bit_len = static_cast<uint64_t>(len) * 8ULL;
    const int nb_final = (tail_len + 1 + 8 <= 64) ? 1 : 2;
    const int last = nb_final * 64;
    for (int i = 0; i < 8; ++i) finalblk[last - 1 - i] = static_cast<uint8_t>(bit_len >> (8 * i));
    for (int b = 0; b < nb_final; ++b) {
        const uint8_t* block = finalblk + b * 64;
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (static_cast<uint32_t>(block[i*4])<<24)|(static_cast<uint32_t>(block[i*4+1])<<16)|
                   (static_cast<uint32_t>(block[i*4+2])<<8)|block[i*4+3];
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = lbtc_rr32(w[i-15],7)^lbtc_rr32(w[i-15],18)^(w[i-15]>>3);
            uint32_t s1 = lbtc_rr32(w[i-2],17)^lbtc_rr32(w[i-2],19)^(w[i-2]>>10);
            w[i] = w[i-16]+s0+w[i-7]+s1;
        }
        uint32_t a=h[0],b2=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1=lbtc_rr32(e,6)^lbtc_rr32(e,11)^lbtc_rr32(e,25); uint32_t ch=(e&f)^(~e&g);
            uint32_t t1=hh+S1+ch+K[i]+w[i]; uint32_t S0=lbtc_rr32(a,2)^lbtc_rr32(a,13)^lbtc_rr32(a,22);
            uint32_t mj=(a&b2)^(a&c)^(b2&c); uint32_t t2=S0+mj;
            hh=g;g=f;f=e;e=d+t1;d=c;c=b2;b2=a;a=t1+t2;
        }
        h[0]+=a;h[1]+=b2;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
    }
    for (int i = 0; i < 8; ++i) {
        out[i*4]=static_cast<uint8_t>(h[i]>>24); out[i*4+1]=static_cast<uint8_t>(h[i]>>16);
        out[i*4+2]=static_cast<uint8_t>(h[i]>>8); out[i*4+3]=static_cast<uint8_t>(h[i]);
    }
}

// x < field prime p ? (big-endian). The device lift_x reduces x mod p, but a valid
// x-only key requires x < p (libsecp/ shim reject x >= p), so we gate on it explicitly.
__device__ __forceinline__ bool lbtc_x_lt_p(const uint8_t* x) {
    const uint8_t P[32] = {
        0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,
        0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xfe,0xff,0xff,0xfc,0x2f};
    for (int i = 0; i < 32; ++i) {
        if (x[i] < P[i]) return true;
        if (x[i] > P[i]) return false;
    }
    return false;  // x == p -> not < p
}

__global__ void lbtc_xonly_validate_kernel(
    const uint8_t* __restrict__ keys, int n, uint8_t* __restrict__ res) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= n) return;
    secp256k1::cuda::JacobianPoint p;
    res[i] = (lbtc_x_lt_p(keys + i*32) &&
              secp256k1::cuda::lift_x(keys + i*32, &p)) ? 1u : 0u;
}

__global__ void lbtc_commitment_kernel(
    const uint8_t* __restrict__ ix, const uint8_t* __restrict__ tw,
    const uint8_t* __restrict__ tx, const uint8_t* __restrict__ par,
    int n, uint8_t* __restrict__ res) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= n) return;
    secp256k1::cuda::JacobianPoint P;
    if (!lbtc_x_lt_p(ix + i*32) ||
        !secp256k1::cuda::lift_x(ix + i*32, &P)) { res[i] = 0; return; }   // even-y internal, x<p
    secp256k1::cuda::FieldElement ax, ay;
    secp256k1::cuda::jacobian_to_affine(&P, &ax, &ay);
    secp256k1::cuda::AffinePoint Pa; Pa.x = ax; Pa.y = ay;
    secp256k1::cuda::Scalar s; secp256k1::cuda::scalar_from_bytes(tw + i*32, &s);
    secp256k1::cuda::JacobianPoint T; secp256k1::cuda::scalar_mul_generator_const(&s, &T);
    secp256k1::cuda::JacobianPoint Q; secp256k1::cuda::jacobian_add_mixed(&T, &Pa, &Q);
    uint8_t comp[33]; secp256k1::cuda::point_to_compressed(&Q, comp);
    uint8_t ok = (comp[0] == (par[i] ? 0x03u : 0x02u)) ? 1u : 0u;          // y-parity
    for (int j = 0; j < 32; ++j) if (comp[1+j] != tx[i*32 + j]) ok = 0;    // x(Q) == tweaked_x
    res[i] = ok;
}

__global__ void lbtc_tagged_hash_kernel(
    const uint8_t* __restrict__ th, const uint8_t* __restrict__ msgs,
    int msg_len, int n, uint8_t* __restrict__ out) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= n) return;
    uint8_t buf[320];
    for (int j = 0; j < 32; ++j) { buf[j] = th[j]; buf[32+j] = th[j]; }   // SHA256(tag) twice
    for (int j = 0; j < msg_len; ++j) buf[64+j] = msgs[i*msg_len + j];
    lbtc_sha256(buf, 64 + msg_len, out + i*32);
}

__global__ void lbtc_pubkey_validate_kernel(
    const uint8_t* __restrict__ pk33, int n, uint8_t* __restrict__ res) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= n) return;
    const uint8_t* p = pk33 + i*33; const uint8_t pfx = p[0];
    bool ok = (pfx == 0x02 || pfx == 0x03) && lbtc_x_lt_p(p + 1);
    if (ok) { secp256k1::cuda::JacobianPoint J; ok = secp256k1::cuda::lift_x(p + 1, &J); }
    res[i] = ok ? 1u : 0u;
}

__global__ void lbtc_tagged_hash_var_kernel(
    const uint8_t* __restrict__ th, const uint8_t* __restrict__ msgs,
    const uint32_t* __restrict__ lens, int stride, int n, uint8_t* __restrict__ out) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= n) return;
    int L = (int)lens[i]; if (L > 256) L = 256;
    uint8_t buf[320];
    for (int j = 0; j < 32; ++j) { buf[j] = th[j]; buf[32+j] = th[j]; }
    for (int j = 0; j < L; ++j) buf[64+j] = msgs[(size_t)i*stride + j];
    lbtc_sha256(buf, 64 + L, out + i*32);
}

__global__ void lbtc_hash256_kernel(
    const uint8_t* __restrict__ inputs, int input_len, int n, uint8_t* __restrict__ out) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= n) return;
    uint8_t h1[32];
    lbtc_sha256(inputs + (size_t)i*input_len, input_len, h1);   // SHA256(input)
    lbtc_sha256(h1, 32, out + i*32);                            // SHA256(SHA256(input))
}

// Batch Merkle-tree parent hashing: out[i] = SHA256(SHA256(left32[i] || right32[i])),
// SoA inputs (left32/right32 are separate n*32-byte buffers, not interleaved).
__global__ void lbtc_merkle_pair_kernel(
    const uint8_t* __restrict__ left32, const uint8_t* __restrict__ right32,
    int n, uint8_t* __restrict__ out) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= n) return;
    uint8_t combined[64];
    for (int j = 0; j < 32; ++j) combined[j]      = left32[(size_t)i * 32 + j];
    for (int j = 0; j < 32; ++j) combined[32 + j] = right32[(size_t)i * 32 + j];
    uint8_t h1[32];
    lbtc_sha256(combined, 64, h1);
    lbtc_sha256(h1, 32, out + (size_t)i * 32);
}

// Batch HASH256 (double SHA-256) of variable-length rows: row_i = inputs[i*stride ..
// i*stride+input_lens[i]); bytes beyond input_lens[i] up to stride are ignored padding.
// Rows may span multiple megabytes (real Bitcoin transactions) -- lbtc_sha256 is
// block-streaming so this stays O(1) local memory regardless of input_lens[i].
__global__ void lbtc_hash256_var_kernel(
    const uint8_t* __restrict__ inputs, const uint32_t* __restrict__ input_lens,
    size_t stride, int n, uint8_t* __restrict__ out) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= n) return;
    uint8_t h1[32];
    lbtc_sha256(inputs + (size_t)i*stride, (int)input_lens[i], h1);
    lbtc_sha256(h1, 32, out + i*32);
}

// Batch sighash descriptor hash.  Parses the compact descriptor bytecode on the
// host, copies referenced field columns to the GPU, then each thread streams its
// row's fields directly through SHA-256 -- NO per-row preimage buffer.
//
// Device-side plan (mirrors host-side FieldPlan in libbitcoin.hpp):
struct DeviceSighashFieldPlan {
    uint16_t field_id;
    uint8_t  flags;       // bit0 = HAS_LENGTH, bit1 = ZERO_PAD
    uint8_t  pad;
    uint32_t fixed_len;   // 0 = variable-length field
};

// ── SHA-256 transform helper (device, shared by sighash streaming kernel) ──
__device__ __forceinline__ void lbtc_sha256_transform(uint32_t h[8], const uint8_t block[64])
{
    uint32_t w[64];
    for (int j = 0; j < 16; ++j) {
        const uint8_t* b = block + j * 4;
        w[j] = ((uint32_t)b[0]<<24)|((uint32_t)b[1]<<16)|
               ((uint32_t)b[2]<<8)|(uint32_t)b[3];
    }
    for (int j = 16; j < 64; ++j) {
        uint32_t s0 = lbtc_rr32(w[j-15],7)^lbtc_rr32(w[j-15],18)^(w[j-15]>>3);
        uint32_t s1 = lbtc_rr32(w[j-2],17)^lbtc_rr32(w[j-2],19)^(w[j-2]>>10);
        w[j] = w[j-16]+s0+w[j-7]+s1;
    }
    uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],fval=h[5],g=h[6],hh=h[7];
    const uint32_t K[64] = {
      0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
      0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
      0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
      0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
      0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
      0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
      0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
      0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};
    for (int j = 0; j < 64; ++j) {
        uint32_t S1=lbtc_rr32(e,6)^lbtc_rr32(e,11)^lbtc_rr32(e,25); uint32_t ch=(e&fval)^(~e&g);
        uint32_t t1=hh+S1+ch+K[j]+w[j]; uint32_t S0=lbtc_rr32(a,2)^lbtc_rr32(a,13)^lbtc_rr32(a,22);
        uint32_t mj=(a&b)^(a&c)^(b&c); uint32_t t2=S0+mj;
        hh=g;g=fval;fval=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=fval;h[6]+=g;h[7]+=hh;
}

__global__ void lbtc_sighash_descriptor_hash_kernel(
    const DeviceSighashFieldPlan* __restrict__ plan,
    int num_fields,
    const uint8_t* const* __restrict__ d_field_ptrs,
    const uint32_t* __restrict__ d_field_strides,
    const uint32_t* const* __restrict__ d_var_len_ptrs,
    int n,
    uint8_t* __restrict__ out32)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Per-thread SHA-256 state for streaming field data.
    // One inner SHA-256 context covers the concatenated field byte stream:
    //   field_0 || field_1 || ... || field_{N-1}
    // Partial blocks are carried across field boundaries; SHA-256 padding
    // (0x80 + zeroes + 64-bit big-endian length) is applied exactly ONCE
    // after the final field. This matches the HASH256 definition for
    // legacy/BIP143 sighash preimages.
    uint32_t h[8] = {0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
                     0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};
    uint64_t total_bits = 0;
    uint8_t  partial[64];
    int      partial_len = 0;

    for (int fi = 0; fi < num_fields; ++fi) {
        const DeviceSighashFieldPlan f = plan[fi];
        const uint8_t* col = d_field_ptrs[fi];
        const uint32_t stride = d_field_strides[fi];

        int field_bytes;
        const uint8_t* row_data;

        if (f.fixed_len > 0) {
            field_bytes = (int)f.fixed_len;
            row_data = col + (size_t)i * stride;
        } else {
            const uint32_t* var_lens = d_var_len_ptrs[fi];
            field_bytes = (int)var_lens[i];
            row_data = col + (size_t)i * stride;
        }

        int off = 0;

        // If we have a partial block from a previous field, fill it first.
        if (partial_len > 0) {
            int copy = 64 - partial_len;
            if (copy > field_bytes) copy = field_bytes;
            for (int j = 0; j < copy; ++j)
                partial[partial_len + j] = row_data[j];
            partial_len += copy;
            off += copy;
            if (partial_len == 64) {
                lbtc_sha256_transform(h, partial);
                total_bits += 512;
                partial_len = 0;
            }
        }

        // Process full 64-byte blocks.
        while (off + 64 <= field_bytes) {
            lbtc_sha256_transform(h, row_data + off);
            total_bits += 512;
            off += 64;
        }

        // Save remainder as new partial block.
        int rem = field_bytes - off;
        if (rem > 0) {
            for (int j = 0; j < rem; ++j)
                partial[partial_len + j] = row_data[off + j];
            partial_len += rem;
        }
    }

    // ── Finalise inner SHA-256: pad (0x80 + zeroes + 64-bit BE bit-length) ──
    total_bits += (uint64_t)partial_len * 8;

    // Build the final padded block(s) in a local buffer.
    uint8_t blk[128];
    for (int j = 0; j < 128; ++j) blk[j] = 0;
    for (int j = 0; j < partial_len; ++j) blk[j] = partial[j];
    blk[partial_len] = 0x80;

    // One or two final blocks depending on where the 8-byte length falls.
    int nb = (partial_len + 1 + 8 <= 64) ? 1 : 2;
    int last = nb * 64;
    for (int j = 0; j < 8; ++j)
        blk[last - 1 - j] = (uint8_t)(total_bits >> (8 * j));

    for (int b = 0; b < nb; ++b) {
        lbtc_sha256_transform(h, blk + b * 64);
    }

    // Write inner hash: h[0..7] as big-endian bytes.
    uint8_t inner_hash[32];
    for (int j = 0; j < 8; ++j) {
        inner_hash[j*4]   = (uint8_t)(h[j]>>24);
        inner_hash[j*4+1] = (uint8_t)(h[j]>>16);
        inner_hash[j*4+2] = (uint8_t)(h[j]>>8);
        inner_hash[j*4+3] = (uint8_t)(h[j]);
    }

    // Outer SHA-256: hash the 32-byte inner digest → HASH256.
    lbtc_sha256(inner_hash, 32, out32 + (size_t)i * 32);
}

}  // anonymous namespace (libbitcoin-bridge kernels)

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

        out = DeviceInfo{};
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
        g_cuda_batch_scratch.free_lbtc_device();
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

        Scalar*        d_scalars = nullptr;
        JacobianPoint* d_results = nullptr;
        uint8_t*       d_out     = nullptr;
        GpuError       ret       = GpuError::Ok;

        /* Host copy — needed for secure_erase on all exit paths (NF-01 fix) */
        std::vector<Scalar> h_scalars(count);
        for (size_t i = 0; i < count; ++i)
            bytes_to_scalar(scalars32 + i * 32, &h_scalars[i]);

        if (cudaMalloc(&d_scalars, count * sizeof(Scalar)) != cudaSuccess) {
            ret = set_error(GpuError::Memory, "generator_mul d_scalars"); goto gmb_cleanup; }
        if (cudaMalloc(&d_results, count * sizeof(JacobianPoint)) != cudaSuccess) {
            ret = set_error(GpuError::Memory, "generator_mul d_results"); goto gmb_cleanup; }
        if (cudaMemcpy(d_scalars, h_scalars.data(),
                       count * sizeof(Scalar), cudaMemcpyHostToDevice) != cudaSuccess) {
            ret = set_error(GpuError::Launch, "generator_mul upload"); goto gmb_cleanup; }
        {
            int threads = 128;
            int blocks  = (static_cast<int>(count) + threads - 1) / threads;
            // GPU Guardrail 8: use CT kernel — scalars are private keys (pubkey derivation).
            // ct_generator_mul_batch_kernel uses GLV comb with branchless cmov; no warp
            // divergence on scalar nibble values, eliminating the timing side-channel that
            // generator_mul_windowed_batch_kernel exposes on secret inputs.
            ct_generator_mul_batch_kernel<<<blocks, threads>>>(
                d_scalars, d_results, static_cast<int>(count));
            if (cudaGetLastError() != cudaSuccess) {
                ret = set_error(GpuError::Launch, "generator_mul kernel"); goto gmb_cleanup; }
            if (cudaMalloc(&d_out, count * 33) != cudaSuccess) {
                ret = set_error(GpuError::Memory, "generator_mul d_out"); goto gmb_cleanup; }
            batch_jac_to_compressed_kernel<<<blocks, threads>>>(
                d_results, d_out, static_cast<int>(count));
            if (cudaGetLastError() != cudaSuccess) {
                ret = set_error(GpuError::Launch, "jac_to_compressed"); goto gmb_cleanup; }
            if (cudaDeviceSynchronize() != cudaSuccess) {
                ret = set_error(GpuError::Launch, "generator_mul sync"); goto gmb_cleanup; }
            if (cudaMemcpy(out_pubkeys33, d_out, count * 33,
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
                ret = set_error(GpuError::Launch, "generator_mul download"); goto gmb_cleanup; }
            clear_error();
        }

gmb_cleanup:
        // NF-01: Rule 10 — erase private scalars from GPU VRAM on ALL exit paths
        if (d_scalars) {
            cudaMemset(d_scalars, 0, count * sizeof(Scalar));
            cudaFree(d_scalars);
        }
        if (d_results) cudaFree(d_results);
        if (d_out)     cudaFree(d_out);
        secp256k1::detail::secure_erase(h_scalars.data(),
                                        h_scalars.size() * sizeof(Scalar));
        return ret;
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

        /* Allocate device memory */
        uint8_t*            d_msgs    = nullptr;
        uint8_t*            d_pubs33  = nullptr;
        JacobianPoint*      d_pubs    = nullptr;
        bool*               d_pub_ok  = nullptr;
        uint8_t*            d_sigs64  = nullptr;
        bool*               d_res     = nullptr;

        CUDA_TRY(cudaMalloc(&d_msgs, count * 32));
        CudaBufferGuard d_msgs_guard(d_msgs);
        CUDA_TRY(cudaMalloc(&d_pubs33, count * 33));
        CudaBufferGuard d_pubs33_guard(d_pubs33);
        CUDA_TRY(cudaMalloc(&d_pubs, count * sizeof(JacobianPoint)));
        CudaBufferGuard d_pubs_guard(d_pubs);
        CUDA_TRY(cudaMalloc(&d_pub_ok, count * sizeof(bool)));
        CudaBufferGuard d_pub_ok_guard(d_pub_ok);
        CUDA_TRY(cudaMalloc(&d_sigs64, count * 64));
        CudaBufferGuard d_sigs64_guard(d_sigs64);
        CUDA_TRY(cudaMalloc(&d_res, count * sizeof(bool)));
        CudaBufferGuard d_res_guard(d_res);

        CUDA_TRY(cudaMemcpy(d_msgs, msg_hashes32, count * 32, cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_pubs33, pubkeys33, count * 33, cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_sigs64, sigs64, count * 64, cudaMemcpyHostToDevice));

        /* Decompress pubkeys on GPU */
        int threads = 128;
        int blocks  = (static_cast<int>(count) + threads - 1) / threads;
        batch_compressed_to_jac_kernel<<<blocks, threads>>>(
            d_pubs33, d_pubs, d_pub_ok, static_cast<int>(count));
        CUDA_TRY(cudaGetLastError());

        /* Launch verify */
        ecdsa_verify_batch_kernel<<<blocks, threads>>>(
            d_msgs, d_pubs, d_sigs64, d_res, static_cast<int>(count));
        CUDA_TRY(cudaGetLastError());
        CUDA_TRY(cudaDeviceSynchronize());

        /* Download results */
        uint8_t* const h_res = g_cuda_batch_scratch.ensure_results(count);
        CUDA_TRY(cudaMemcpy(h_res, d_res, count * sizeof(bool), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < count; ++i)
            out_results[i] = h_res[i] ? 1 : 0;

        clear_error();
        return GpuError::Ok;
    }

    GpuError ecdsa_verify_lbtc_rows(
        const uint8_t* rows, size_t stride, size_t count,
        uint8_t* out_results) override
    {
        if (!ready_) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!rows || !out_results)
            return set_error(GpuError::NullArg, "NULL buffer");
        if (stride < 129u)
            return set_error(GpuError::BadInput, "libbitcoin row stride < 129");

        GpuError ret = GpuError::Ok;

        const size_t row_bytes = count * stride;
        if (stride != 0 && row_bytes / stride != count) {
            return set_error(GpuError::BadInput, "libbitcoin row byte size overflow");
        }
        if (g_cuda_batch_scratch.ensure_lbtc_rows(row_bytes) != cudaSuccess) {
            return set_error(GpuError::Memory, "lbtc rows buffer alloc");
        }
        if (g_cuda_batch_scratch.ensure_lbtc_results(count) != cudaSuccess) {
            return set_error(GpuError::Memory, "lbtc result buffer alloc");
        }
        uint8_t* const d_rows = g_cuda_batch_scratch.lbtc_rows;
        bool* const d_res = g_cuda_batch_scratch.lbtc_results;

        if (cudaMemcpy(d_rows, rows, row_bytes, cudaMemcpyHostToDevice) != cudaSuccess)
            return set_error(GpuError::Launch, "lbtc rows upload");

        {
            int threads = 128;
            int blocks  = (static_cast<int>(count) + threads - 1) / threads;
            ecdsa_verify_lbtc_rows_kernel<<<blocks, threads>>>(
                d_rows, stride, d_res, static_cast<int>(count));
        }
        if (cudaGetLastError() != cudaSuccess) {
            return set_error(GpuError::Launch, "lbtc ecdsa row kernel launch");
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            return set_error(GpuError::Launch, "lbtc ecdsa row kernel sync");
        }

        {
            uint8_t* const h_res = g_cuda_batch_scratch.ensure_results(count);
            if (cudaMemcpy(h_res, d_res, count * sizeof(bool),
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
                return set_error(GpuError::Launch, "lbtc result download");
            }
            for (size_t i = 0; i < count; ++i)
                out_results[i] = h_res[i] ? 1 : 0;
        }
        clear_error();
        return ret;
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
        CudaBufferGuard d_pks_guard(d_pks);
        CUDA_TRY(cudaMalloc(&d_msgs, count * 32));
        CudaBufferGuard d_msgs_guard(d_msgs);
        CUDA_TRY(cudaMalloc(&d_sigs, count * sizeof(SchnorrSignatureGPU)));
        CudaBufferGuard d_sigs_guard(d_sigs);
        CUDA_TRY(cudaMalloc(&d_res, count * sizeof(bool)));
        CudaBufferGuard d_res_guard(d_res);

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

        clear_error();
        return GpuError::Ok;
    }

    // Engine-owned, memory-aware chunk size (rows). Bounds a chunk by half of free
    // device memory and a hard cap so a large batch cannot force a giant one-shot
    // allocation. UFSECP_GPU_COLUMNS_CHUNK is an internal/test knob only — the
    // caller never sees or manages chunking.
    static size_t lbtc_columns_chunk(size_t count, size_t per_row) {
        size_t free_b = 0, total_b = 0;
        size_t cap = (static_cast<size_t>(4) << 20);
        if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess && per_row) {
            const size_t by_mem = (free_b / 2) / per_row;
            if (by_mem < cap) cap = by_mem;
        }
        if (const char* e = std::getenv("UFSECP_GPU_COLUMNS_CHUNK")) {
            const unsigned long long v = std::strtoull(e, nullptr, 10);
            if (v > 0 && static_cast<size_t>(v) < cap) cap = static_cast<size_t>(v);
        }
        if (cap < 1) cap = 1;
        return (count < cap) ? count : cap;
    }

    GpuError ecdsa_verify_lbtc_columns(
        const uint8_t* digests32, const uint8_t* pubkeys33,
        const uint8_t* sigs64, size_t count,
        uint8_t* out_results) override
    {
        if (!ready_) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!digests32 || !pubkeys33 || !sigs64 || !out_results)
            return set_error(GpuError::NullArg, "NULL buffer");

        // Fail-closed: initialise the whole result buffer to invalid (0) up front,
        // so any early non-OK return (alloc/upload/launch/sync/download) leaves no
        // stale or partial verdict behind for a caller that inspects the buffer
        // despite the error status. On success every row is overwritten below.
        std::memset(out_results, 0, count);

        const size_t chunk = lbtc_columns_chunk(count, 32u + 33u + 64u + sizeof(bool));
        // Reusable, grow-only device + host staging (acceptance A6): no fresh
        // per-call cudaMalloc/cudaFree on the hot loop. One packed device buffer
        // holds the three column spans (dig | pub | sig) at fixed sub-offsets.
        if (g_cuda_batch_scratch.ensure_lbtc_cols(chunk * (32u + 33u + 64u)) != cudaSuccess ||
            g_cuda_batch_scratch.ensure_lbtc_results(chunk) != cudaSuccess) {
            return set_error(GpuError::Memory, "lbtc ecdsa columns device alloc");
        }
        uint8_t* const d_dig = g_cuda_batch_scratch.lbtc_cols;
        uint8_t* const d_pub = d_dig + chunk * 32;
        uint8_t* const d_sig = d_pub + chunk * 33;
        bool* const d_res = g_cuda_batch_scratch.lbtc_results;
        uint8_t* const h_res = g_cuda_batch_scratch.ensure_results(chunk);

        for (size_t off = 0; off < count; off += chunk) {
            const size_t n = (count - off < chunk) ? (count - off) : chunk;
            if (cudaMemcpy(d_dig, digests32 + off * 32, n * 32, cudaMemcpyHostToDevice) != cudaSuccess ||
                cudaMemcpy(d_pub, pubkeys33 + off * 33, n * 33, cudaMemcpyHostToDevice) != cudaSuccess ||
                cudaMemcpy(d_sig, sigs64    + off * 64, n * 64, cudaMemcpyHostToDevice) != cudaSuccess) {
                return set_error(GpuError::Launch, "lbtc ecdsa columns upload");
            }
            const int threads = 128;
            const int blocks  = (static_cast<int>(n) + threads - 1) / threads;
            ecdsa_verify_lbtc_columns_kernel<<<blocks, threads>>>(
                d_dig, d_pub, d_sig, d_res, static_cast<int>(n));
            if (cudaGetLastError() != cudaSuccess) {
                return set_error(GpuError::Launch, "lbtc ecdsa columns kernel launch");
            }
            if (cudaDeviceSynchronize() != cudaSuccess) {
                return set_error(GpuError::Launch, "lbtc ecdsa columns kernel sync");
            }
            if (cudaMemcpy(h_res, d_res, n * sizeof(bool), cudaMemcpyDeviceToHost) != cudaSuccess) {
                return set_error(GpuError::Launch, "lbtc ecdsa columns download");
            }
            for (size_t i = 0; i < n; ++i)
                out_results[off + i] = h_res[i] ? 1 : 0;
        }
        clear_error();
        return GpuError::Ok;
    }

    GpuError schnorr_verify_lbtc_columns(
        const uint8_t* digests32, const uint8_t* xonly32,
        const uint8_t* sigs64, size_t count,
        uint8_t* out_results) override
    {
        if (!ready_) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!digests32 || !xonly32 || !sigs64 || !out_results)
            return set_error(GpuError::NullArg, "NULL buffer");

        // Fail-closed: initialise the whole result buffer to invalid (0) up front,
        // so any early non-OK return leaves no stale or partial verdict behind.
        // On success every row is overwritten below.
        std::memset(out_results, 0, count);

        const size_t chunk = lbtc_columns_chunk(count, 32u + 32u + 64u + sizeof(bool));
        // Reusable, grow-only device + host staging (acceptance A6): no fresh
        // per-call cudaMalloc/cudaFree on the hot loop. One packed device buffer
        // holds the three column spans (dig | xon | sig) at fixed sub-offsets.
        if (g_cuda_batch_scratch.ensure_lbtc_cols(chunk * (32u + 32u + 64u)) != cudaSuccess ||
            g_cuda_batch_scratch.ensure_lbtc_results(chunk) != cudaSuccess) {
            return set_error(GpuError::Memory, "lbtc schnorr columns device alloc");
        }
        uint8_t* const d_dig = g_cuda_batch_scratch.lbtc_cols;
        uint8_t* const d_xon = d_dig + chunk * 32;
        uint8_t* const d_sig = d_xon + chunk * 32;
        bool* const d_res = g_cuda_batch_scratch.lbtc_results;
        uint8_t* const h_res = g_cuda_batch_scratch.ensure_results(chunk);

        for (size_t off = 0; off < count; off += chunk) {
            const size_t n = (count - off < chunk) ? (count - off) : chunk;
            if (cudaMemcpy(d_dig, digests32 + off * 32, n * 32, cudaMemcpyHostToDevice) != cudaSuccess ||
                cudaMemcpy(d_xon, xonly32   + off * 32, n * 32, cudaMemcpyHostToDevice) != cudaSuccess ||
                cudaMemcpy(d_sig, sigs64    + off * 64, n * 64, cudaMemcpyHostToDevice) != cudaSuccess) {
                return set_error(GpuError::Launch, "lbtc schnorr columns upload");
            }
            const int threads = 128;
            const int blocks  = (static_cast<int>(n) + threads - 1) / threads;
            schnorr_verify_lbtc_columns_kernel<<<blocks, threads>>>(
                d_dig, d_xon, d_sig, d_res, static_cast<int>(n));
            if (cudaGetLastError() != cudaSuccess) {
                return set_error(GpuError::Launch, "lbtc schnorr columns kernel launch");
            }
            if (cudaDeviceSynchronize() != cudaSuccess) {
                return set_error(GpuError::Launch, "lbtc schnorr columns kernel sync");
            }
            if (cudaMemcpy(h_res, d_res, n * sizeof(bool), cudaMemcpyDeviceToHost) != cudaSuccess) {
                return set_error(GpuError::Launch, "lbtc schnorr columns download");
            }
            for (size_t i = 0; i < n; ++i)
                out_results[off + i] = h_res[i] ? 1 : 0;
        }
        clear_error();
        return GpuError::Ok;
    }

    /* "Collect" variants (libbitcoin bridge). Copy of *_verify_batch above; the
     * ONLY changes: the bool* d_res result buffer becomes a 1-byte/row d_keys
     * verdict buffer that is SEEDED from the caller's key_buffer (non-zero) and
     * downloaded back into it, and the host result-scatter loop is removed (the
     * collect kernel writes 0 on valid / leaves on invalid directly). Verdict is
     * bit-identical to verify_batch (same device verify, same decompress). */
    GpuError ecdsa_verify_collect(
        const uint8_t* msg_hashes32, const uint8_t* pubkeys33,
        const uint8_t* sigs64, size_t count,
        uint8_t* key_buffer) override
    {
        if (!ready_) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!msg_hashes32 || !pubkeys33 || !sigs64 || !key_buffer)
            return set_error(GpuError::NullArg, "NULL buffer");

        std::vector<ECDSASignatureGPU> h_sigs(count);
        for (size_t i = 0; i < count; ++i)
            bytes_to_ecdsa_sig(sigs64 + i * 64, &h_sigs[i]);

        uint8_t*            d_msgs   = nullptr;
        uint8_t*            d_pubs33 = nullptr;
        JacobianPoint*      d_pubs   = nullptr;
        bool*               d_pub_ok = nullptr;
        ECDSASignatureGPU*  d_sigs   = nullptr;
        uint8_t*            d_keys   = nullptr;   /* 1 byte/row verdict channel */

        CUDA_TRY(cudaMalloc(&d_msgs, count * 32));
        CUDA_TRY(cudaMalloc(&d_pubs33, count * 33));
        CUDA_TRY(cudaMalloc(&d_pubs, count * sizeof(JacobianPoint)));
        CUDA_TRY(cudaMalloc(&d_pub_ok, count * sizeof(bool)));
        CUDA_TRY(cudaMalloc(&d_sigs, count * sizeof(ECDSASignatureGPU)));
        CUDA_TRY(cudaMalloc(&d_keys, count));

        CUDA_TRY(cudaMemcpy(d_msgs, msg_hashes32, count * 32, cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_pubs33, pubkeys33, count * 33, cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_sigs, h_sigs.data(), count * sizeof(ECDSASignatureGPU), cudaMemcpyHostToDevice));
        /* seed verdict channel with the caller's non-zero markers: a row the
         * kernel never zeroes (tail thread, or unreached after a fault) stays
         * non-zero = rejected = fail-closed. */
        CUDA_TRY(cudaMemcpy(d_keys, key_buffer, count, cudaMemcpyHostToDevice));

        int threads = 128;
        int blocks  = (static_cast<int>(count) + threads - 1) / threads;
        batch_compressed_to_jac_kernel<<<blocks, threads>>>(
            d_pubs33, d_pubs, d_pub_ok, static_cast<int>(count));
        CUDA_TRY(cudaGetLastError());

        ecdsa_verify_collect_kernel<<<blocks, threads>>>(
            d_msgs, d_pubs, d_sigs, d_keys, static_cast<int>(count));
        CUDA_TRY(cudaGetLastError());
        CUDA_TRY(cudaDeviceSynchronize());

        /* verdict already collapsed on device; no host scatter loop */
        CUDA_TRY(cudaMemcpy(key_buffer, d_keys, count, cudaMemcpyDeviceToHost));

        cudaFree(d_keys);
        cudaFree(d_sigs);
        cudaFree(d_pub_ok);
        cudaFree(d_pubs);
        cudaFree(d_pubs33);
        cudaFree(d_msgs);
        clear_error();
        return GpuError::Ok;
    }

    GpuError schnorr_verify_collect(
        const uint8_t* msg_hashes32, const uint8_t* pubkeys_x32,
        const uint8_t* sigs64, size_t count,
        uint8_t* key_buffer) override
    {
        if (!ready_) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!msg_hashes32 || !pubkeys_x32 || !sigs64 || !key_buffer)
            return set_error(GpuError::NullArg, "NULL buffer");

        std::vector<SchnorrSignatureGPU> h_sigs(count);
        for (size_t i = 0; i < count; ++i)
            bytes_to_schnorr_sig(sigs64 + i * 64, &h_sigs[i]);

        uint8_t*             d_pks  = nullptr;
        uint8_t*             d_msgs = nullptr;
        SchnorrSignatureGPU* d_sigs = nullptr;
        uint8_t*             d_keys = nullptr;

        CUDA_TRY(cudaMalloc(&d_pks, count * 32));
        CUDA_TRY(cudaMalloc(&d_msgs, count * 32));
        CUDA_TRY(cudaMalloc(&d_sigs, count * sizeof(SchnorrSignatureGPU)));
        CUDA_TRY(cudaMalloc(&d_keys, count));

        CUDA_TRY(cudaMemcpy(d_pks, pubkeys_x32, count * 32, cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_msgs, msg_hashes32, count * 32, cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_sigs, h_sigs.data(), count * sizeof(SchnorrSignatureGPU), cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_keys, key_buffer, count, cudaMemcpyHostToDevice));

        int threads = 128;
        int blocks  = (static_cast<int>(count) + threads - 1) / threads;
        schnorr_verify_collect_kernel<<<blocks, threads>>>(
            d_pks, d_msgs, d_sigs, d_keys, static_cast<int>(count));
        CUDA_TRY(cudaGetLastError());
        CUDA_TRY(cudaDeviceSynchronize());

        CUDA_TRY(cudaMemcpy(key_buffer, d_keys, count, cudaMemcpyDeviceToHost));

        cudaFree(d_keys);
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

#if SECP256K1_GPU_HAS_ECDH
        // Rule 10: h_keys erased on all exit paths via destructor + secure_erase.
        // The SecureVector wrapper zeroes on destruction — CUDA_TRY early returns
        // are safe because h_keys goes out of scope before the function returns.
        struct SecureScalarVec {
            std::vector<Scalar> v;
            explicit SecureScalarVec(size_t n) : v(n) {}
            ~SecureScalarVec() {
                secp256k1::detail::secure_erase(v.data(), v.size() * sizeof(Scalar));
            }
        } h_keys_guard(count);
        auto& h_keys = h_keys_guard.v;

        for (size_t i = 0; i < count; ++i)
            bytes_to_scalar(privkeys32 + i * 32, &h_keys[i]);

        // CudaKeyGuard zeroes + frees d_keys on all exit paths (destructor).
        Scalar* d_keys_raw = nullptr;
        CUDA_TRY(cudaMalloc(&d_keys_raw, count * sizeof(Scalar)));
        CudaKeyGuard d_keys_guard(d_keys_raw, count * sizeof(Scalar));

        uint8_t*       d_pubs33 = nullptr;
        JacobianPoint* d_pubs   = nullptr;
        bool*          d_pub_ok = nullptr;
        uint8_t*       d_out    = nullptr;
        bool*          d_ok     = nullptr;
        CUDA_TRY(cudaMalloc(&d_pubs33, count * 33));
        CUDA_TRY(cudaMalloc(&d_pubs, count * sizeof(JacobianPoint)));
        CUDA_TRY(cudaMalloc(&d_pub_ok, count * sizeof(bool)));
        CUDA_TRY(cudaMalloc(&d_out, count * 32));
        CUDA_TRY(cudaMalloc(&d_ok, count * sizeof(bool)));

        CUDA_TRY(cudaMemcpy(d_keys_raw, h_keys.data(), count * sizeof(Scalar), cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_pubs33, peer_pubkeys33, count * 33, cudaMemcpyHostToDevice));

        int threads = 128;
        int blocks  = (static_cast<int>(count) + threads - 1) / threads;
        batch_compressed_to_jac_kernel<<<blocks, threads>>>(
            d_pubs33, d_pubs, d_pub_ok, static_cast<int>(count));
        CUDA_TRY(cudaGetLastError());

        ecdh_batch_kernel<<<blocks, threads>>>(d_keys_raw, d_pubs, d_out, d_ok, static_cast<int>(count));
        CUDA_TRY(cudaGetLastError());
        CUDA_TRY(cudaDeviceSynchronize());

        CUDA_TRY(cudaMemcpy(out_secrets32, d_out, count * 32, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> h_ok_buf(count);
        CUDA_TRY(cudaMemcpy(h_ok_buf.data(), d_ok, count * sizeof(bool), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < count; ++i) {
            if (!h_ok_buf[i]) std::memset(out_secrets32 + i * 32, 0, 32);
        }

        // d_keys_guard destructor zeroes + frees d_keys_raw here.
        // h_keys_guard destructor secure_erases h_keys here.
        cudaFree(d_ok);
        cudaFree(d_out);
        cudaFree(d_pub_ok);
        cudaFree(d_pubs);
        cudaFree(d_pubs33);
        clear_error();
        return GpuError::Ok;
#else
        return set_error(GpuError::Unsupported, "GPU ECDH module disabled at build time");
#endif
    }

    GpuError hash160_pubkey_batch(
        const uint8_t* pubkeys33, size_t count,
        uint8_t* out_hash160) override
    {
        if (!ready_) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!pubkeys33 || !out_hash160)
            return set_error(GpuError::NullArg, "NULL buffer");

#if SECP256K1_GPU_HAS_HASH160
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
#else
        return set_error(GpuError::Unsupported, "GPU HASH160 module disabled at build time");
#endif
    }

    GpuError msm(
        const uint8_t* scalars32, const uint8_t* points33,
        size_t n, uint8_t* out_result33) override
    {
        if (!ready_) return set_error(GpuError::Device, "context not initialised");
        if (n == 0) { clear_error(); return GpuError::Ok; }
        if (!scalars32 || !points33 || !out_result33)
            return set_error(GpuError::NullArg, "NULL buffer");

#if SECP256K1_GPU_HAS_MSM
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
#else
        return set_error(GpuError::Unsupported, "GPU MSM module disabled at build time");
#endif
    }

    /* libbitcoin-bridge: batch x-only key validation (lift_x per key). PUBLIC. */
    GpuError xonly_validate(
        const uint8_t* keys32, size_t n, uint8_t* results) override
    {
        if (!ready_) return set_error(GpuError::Device, "context not initialised");
        if (n == 0) { clear_error(); return GpuError::Ok; }
        if (!keys32 || !results) return set_error(GpuError::NullArg, "NULL buffer");

        uint8_t* d_keys = nullptr;
        uint8_t* d_res  = nullptr;
        GpuError ret = GpuError::Ok;
        const int n_int   = static_cast<int>(n);
        const int threads = 256;
        const int blocks  = (n_int + threads - 1) / threads;

        if (cudaMalloc(&d_keys, n * 32) != cudaSuccess) {
            ret = set_error(GpuError::Memory, "xonly_validate d_keys"); goto xv_cleanup; }
        if (cudaMalloc(&d_res, n) != cudaSuccess) {
            ret = set_error(GpuError::Memory, "xonly_validate d_res"); goto xv_cleanup; }
        if (cudaMemcpy(d_keys, keys32, n * 32, cudaMemcpyHostToDevice) != cudaSuccess) {
            ret = set_error(GpuError::Launch, "xonly_validate upload"); goto xv_cleanup; }
        lbtc_xonly_validate_kernel<<<blocks, threads>>>(d_keys, n_int, d_res);
        if (cudaGetLastError() != cudaSuccess) {
            ret = set_error(GpuError::Launch, "xonly_validate kernel"); goto xv_cleanup; }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            ret = set_error(GpuError::Launch, "xonly_validate sync"); goto xv_cleanup; }
        if (cudaMemcpy(results, d_res, n, cudaMemcpyDeviceToHost) != cudaSuccess) {
            ret = set_error(GpuError::Launch, "xonly_validate download"); goto xv_cleanup; }
        clear_error();
xv_cleanup:
        if (d_keys) cudaFree(d_keys);
        if (d_res)  cudaFree(d_res);
        return ret;
    }

    /* libbitcoin-bridge: BIP-341 commitment tweak-add-check, one thread per item.
     * accept iff x(lift_x(internal)+tweak*G)==tweaked_x AND its y-parity==parity. PUBLIC. */
    GpuError commitment_verify(
        const uint8_t* internal_x32, const uint8_t* tweak32,
        const uint8_t* tweaked_x32, const uint8_t* parity,
        size_t n, uint8_t* results) override
    {
        if (!ready_) return set_error(GpuError::Device, "context not initialised");
        if (n == 0) { clear_error(); return GpuError::Ok; }
        if (!internal_x32 || !tweak32 || !tweaked_x32 || !parity || !results)
            return set_error(GpuError::NullArg, "NULL buffer");

        uint8_t *d_ix=nullptr,*d_tw=nullptr,*d_tx=nullptr,*d_par=nullptr,*d_res=nullptr;
        GpuError ret = GpuError::Ok;
        const int n_int   = static_cast<int>(n);
        const int threads = 256;
        const int blocks  = (n_int + threads - 1) / threads;

        if (cudaMalloc(&d_ix, n*32)!=cudaSuccess){ ret=set_error(GpuError::Memory,"commit d_ix"); goto cv_cleanup; }
        if (cudaMalloc(&d_tw, n*32)!=cudaSuccess){ ret=set_error(GpuError::Memory,"commit d_tw"); goto cv_cleanup; }
        if (cudaMalloc(&d_tx, n*32)!=cudaSuccess){ ret=set_error(GpuError::Memory,"commit d_tx"); goto cv_cleanup; }
        if (cudaMalloc(&d_par, n)!=cudaSuccess){ ret=set_error(GpuError::Memory,"commit d_par"); goto cv_cleanup; }
        if (cudaMalloc(&d_res, n)!=cudaSuccess){ ret=set_error(GpuError::Memory,"commit d_res"); goto cv_cleanup; }
        if (cudaMemcpy(d_ix, internal_x32, n*32, cudaMemcpyHostToDevice)!=cudaSuccess){ ret=set_error(GpuError::Launch,"commit up ix"); goto cv_cleanup; }
        if (cudaMemcpy(d_tw, tweak32, n*32, cudaMemcpyHostToDevice)!=cudaSuccess){ ret=set_error(GpuError::Launch,"commit up tw"); goto cv_cleanup; }
        if (cudaMemcpy(d_tx, tweaked_x32, n*32, cudaMemcpyHostToDevice)!=cudaSuccess){ ret=set_error(GpuError::Launch,"commit up tx"); goto cv_cleanup; }
        if (cudaMemcpy(d_par, parity, n, cudaMemcpyHostToDevice)!=cudaSuccess){ ret=set_error(GpuError::Launch,"commit up par"); goto cv_cleanup; }
        lbtc_commitment_kernel<<<blocks,threads>>>(d_ix,d_tw,d_tx,d_par,n_int,d_res);
        if (cudaGetLastError()!=cudaSuccess){ ret=set_error(GpuError::Launch,"commit kernel"); goto cv_cleanup; }
        if (cudaDeviceSynchronize()!=cudaSuccess){ ret=set_error(GpuError::Launch,"commit sync"); goto cv_cleanup; }
        if (cudaMemcpy(results, d_res, n, cudaMemcpyDeviceToHost)!=cudaSuccess){ ret=set_error(GpuError::Launch,"commit download"); goto cv_cleanup; }
        clear_error();
cv_cleanup:
        if(d_ix)cudaFree(d_ix); if(d_tw)cudaFree(d_tw); if(d_tx)cudaFree(d_tx);
        if(d_par)cudaFree(d_par); if(d_res)cudaFree(d_res);
        return ret;
    }

    /* libbitcoin-bridge: Taproot tagged hash. tag_hash32 = SHA256(tag) (host-precomputed);
     * out_i = SHA256(tag_hash||tag_hash||msg_i). PUBLIC. msg_len capped at 256. */
    GpuError tagged_hash(
        const uint8_t* tag_hash32, const uint8_t* msgs,
        size_t msg_len, size_t n, uint8_t* out32) override
    {
        if (!ready_) return set_error(GpuError::Device, "context not initialised");
        if (n == 0) { clear_error(); return GpuError::Ok; }
        if (!tag_hash32 || !msgs || !out32) return set_error(GpuError::NullArg, "NULL buffer");
        if (msg_len == 0 || msg_len > 256) return set_error(GpuError::BadInput, "msg_len out of range");

        uint8_t *d_th=nullptr,*d_msgs=nullptr,*d_out=nullptr;
        GpuError ret = GpuError::Ok;
        const int n_int   = static_cast<int>(n);
        const int ml      = static_cast<int>(msg_len);
        const int threads = 256;
        const int blocks  = (n_int + threads - 1) / threads;

        if (cudaMalloc(&d_th, 32)!=cudaSuccess){ ret=set_error(GpuError::Memory,"tagged d_th"); goto th_cleanup; }
        if (cudaMalloc(&d_msgs, n*msg_len)!=cudaSuccess){ ret=set_error(GpuError::Memory,"tagged d_msgs"); goto th_cleanup; }
        if (cudaMalloc(&d_out, n*32)!=cudaSuccess){ ret=set_error(GpuError::Memory,"tagged d_out"); goto th_cleanup; }
        if (cudaMemcpy(d_th, tag_hash32, 32, cudaMemcpyHostToDevice)!=cudaSuccess){ ret=set_error(GpuError::Launch,"tagged up th"); goto th_cleanup; }
        if (cudaMemcpy(d_msgs, msgs, n*msg_len, cudaMemcpyHostToDevice)!=cudaSuccess){ ret=set_error(GpuError::Launch,"tagged up msgs"); goto th_cleanup; }
        lbtc_tagged_hash_kernel<<<blocks,threads>>>(d_th,d_msgs,ml,n_int,d_out);
        if (cudaGetLastError()!=cudaSuccess){ ret=set_error(GpuError::Launch,"tagged kernel"); goto th_cleanup; }
        if (cudaDeviceSynchronize()!=cudaSuccess){ ret=set_error(GpuError::Launch,"tagged sync"); goto th_cleanup; }
        if (cudaMemcpy(out32, d_out, n*32, cudaMemcpyDeviceToHost)!=cudaSuccess){ ret=set_error(GpuError::Launch,"tagged download"); goto th_cleanup; }
        clear_error();
th_cleanup:
        if(d_th)cudaFree(d_th); if(d_msgs)cudaFree(d_msgs); if(d_out)cudaFree(d_out);
        return ret;
    }

    /* libbitcoin-bridge: batch full compressed-pubkey validation. PUBLIC. */
    GpuError pubkey_validate(
        const uint8_t* pubkeys33, size_t n, uint8_t* results) override
    {
        if (!ready_) return set_error(GpuError::Device, "context not initialised");
        if (n == 0) { clear_error(); return GpuError::Ok; }
        if (!pubkeys33 || !results) return set_error(GpuError::NullArg, "NULL buffer");

        uint8_t* d_pk = nullptr; uint8_t* d_res = nullptr;
        GpuError ret = GpuError::Ok;
        const int n_int = static_cast<int>(n);
        const int threads = 256;
        const int blocks  = (n_int + threads - 1) / threads;

        if (cudaMalloc(&d_pk, n*33)!=cudaSuccess){ ret=set_error(GpuError::Memory,"pkvalidate d_pk"); goto pv2_cleanup; }
        if (cudaMalloc(&d_res, n)!=cudaSuccess){ ret=set_error(GpuError::Memory,"pkvalidate d_res"); goto pv2_cleanup; }
        if (cudaMemcpy(d_pk, pubkeys33, n*33, cudaMemcpyHostToDevice)!=cudaSuccess){ ret=set_error(GpuError::Launch,"pkvalidate up"); goto pv2_cleanup; }
        lbtc_pubkey_validate_kernel<<<blocks,threads>>>(d_pk, n_int, d_res);
        if (cudaGetLastError()!=cudaSuccess){ ret=set_error(GpuError::Launch,"pkvalidate kernel"); goto pv2_cleanup; }
        if (cudaDeviceSynchronize()!=cudaSuccess){ ret=set_error(GpuError::Launch,"pkvalidate sync"); goto pv2_cleanup; }
        if (cudaMemcpy(results, d_res, n, cudaMemcpyDeviceToHost)!=cudaSuccess){ ret=set_error(GpuError::Launch,"pkvalidate download"); goto pv2_cleanup; }
        clear_error();
pv2_cleanup:
        if(d_pk)cudaFree(d_pk); if(d_res)cudaFree(d_res);
        return ret;
    }

    /* libbitcoin-bridge: Taproot tagged hash, per-item length (TapLeaf). PUBLIC. */
    GpuError tagged_hash_var(
        const uint8_t* tag_hash32, const uint8_t* msgs, const uint32_t* msg_lens,
        size_t stride, size_t n, uint8_t* out32) override
    {
        if (!ready_) return set_error(GpuError::Device, "context not initialised");
        if (n == 0) { clear_error(); return GpuError::Ok; }
        if (!tag_hash32 || !msgs || !msg_lens || !out32) return set_error(GpuError::NullArg, "NULL buffer");
        if (stride == 0 || stride > 256) return set_error(GpuError::BadInput, "stride out of range");

        uint8_t *d_th=nullptr,*d_msgs=nullptr,*d_out=nullptr; uint32_t* d_lens=nullptr;
        GpuError ret = GpuError::Ok;
        const int n_int = static_cast<int>(n);
        const int st = static_cast<int>(stride);
        const int threads = 256;
        const int blocks  = (n_int + threads - 1) / threads;

        if (cudaMalloc(&d_th, 32)!=cudaSuccess){ ret=set_error(GpuError::Memory,"tagvar d_th"); goto tv_cleanup; }
        if (cudaMalloc(&d_msgs, n*stride)!=cudaSuccess){ ret=set_error(GpuError::Memory,"tagvar d_msgs"); goto tv_cleanup; }
        if (cudaMalloc(&d_lens, n*sizeof(uint32_t))!=cudaSuccess){ ret=set_error(GpuError::Memory,"tagvar d_lens"); goto tv_cleanup; }
        if (cudaMalloc(&d_out, n*32)!=cudaSuccess){ ret=set_error(GpuError::Memory,"tagvar d_out"); goto tv_cleanup; }
        if (cudaMemcpy(d_th, tag_hash32, 32, cudaMemcpyHostToDevice)!=cudaSuccess){ ret=set_error(GpuError::Launch,"tagvar up th"); goto tv_cleanup; }
        if (cudaMemcpy(d_msgs, msgs, n*stride, cudaMemcpyHostToDevice)!=cudaSuccess){ ret=set_error(GpuError::Launch,"tagvar up msgs"); goto tv_cleanup; }
        if (cudaMemcpy(d_lens, msg_lens, n*sizeof(uint32_t), cudaMemcpyHostToDevice)!=cudaSuccess){ ret=set_error(GpuError::Launch,"tagvar up lens"); goto tv_cleanup; }
        lbtc_tagged_hash_var_kernel<<<blocks,threads>>>(d_th,d_msgs,d_lens,st,n_int,d_out);
        if (cudaGetLastError()!=cudaSuccess){ ret=set_error(GpuError::Launch,"tagvar kernel"); goto tv_cleanup; }
        if (cudaDeviceSynchronize()!=cudaSuccess){ ret=set_error(GpuError::Launch,"tagvar sync"); goto tv_cleanup; }
        if (cudaMemcpy(out32, d_out, n*32, cudaMemcpyDeviceToHost)!=cudaSuccess){ ret=set_error(GpuError::Launch,"tagvar download"); goto tv_cleanup; }
        clear_error();
tv_cleanup:
        if(d_th)cudaFree(d_th); if(d_msgs)cudaFree(d_msgs); if(d_lens)cudaFree(d_lens); if(d_out)cudaFree(d_out);
        return ret;
    }

    /* libbitcoin-bridge: batch HASH256 (double SHA-256) of fixed-length inputs. PUBLIC. */
    GpuError hash256(
        const uint8_t* inputs, size_t input_len, size_t n, uint8_t* out32) override
    {
        if (!ready_) return set_error(GpuError::Device, "context not initialised");
        if (n == 0) { clear_error(); return GpuError::Ok; }
        if (!inputs || !out32) return set_error(GpuError::NullArg, "NULL buffer");
        if (input_len == 0 || input_len > 320) return set_error(GpuError::BadInput, "input_len out of range");

        uint8_t *d_in=nullptr,*d_out=nullptr;
        GpuError ret = GpuError::Ok;
        const int n_int = static_cast<int>(n);
        const int il = static_cast<int>(input_len);
        const int threads = 256;
        const int blocks  = (n_int + threads - 1) / threads;

        if (cudaMalloc(&d_in, n*input_len)!=cudaSuccess){ ret=set_error(GpuError::Memory,"hash256 d_in"); goto h2_cleanup; }
        if (cudaMalloc(&d_out, n*32)!=cudaSuccess){ ret=set_error(GpuError::Memory,"hash256 d_out"); goto h2_cleanup; }
        if (cudaMemcpy(d_in, inputs, n*input_len, cudaMemcpyHostToDevice)!=cudaSuccess){ ret=set_error(GpuError::Launch,"hash256 up"); goto h2_cleanup; }
        lbtc_hash256_kernel<<<blocks,threads>>>(d_in, il, n_int, d_out);
        if (cudaGetLastError()!=cudaSuccess){ ret=set_error(GpuError::Launch,"hash256 kernel"); goto h2_cleanup; }
        if (cudaDeviceSynchronize()!=cudaSuccess){ ret=set_error(GpuError::Launch,"hash256 sync"); goto h2_cleanup; }
        if (cudaMemcpy(out32, d_out, n*32, cudaMemcpyDeviceToHost)!=cudaSuccess){ ret=set_error(GpuError::Launch,"hash256 download"); goto h2_cleanup; }
        clear_error();
h2_cleanup:
        if(d_in)cudaFree(d_in); if(d_out)cudaFree(d_out);
        return ret;
    }

    /* libbitcoin-bridge: batch HASH256 (double SHA-256) of variable-length rows.
     * row_i = inputs[i*stride .. i*stride+input_lens[i]); bytes beyond input_lens[i]
     * up to stride are ignored padding. out32[i*32..] = SHA256(SHA256(row_i)). PUBLIC
     * data -- no upper length cap (rows may span multi-megabyte transactions);
     * lbtc_sha256 is block-streaming so per-row device memory stays O(1). */
    GpuError hash256_var(
        const uint8_t* inputs, const uint32_t* input_lens, size_t stride, size_t n, uint8_t* out32) override
    {
        if (!ready_) return set_error(GpuError::Device, "context not initialised");
        if (n == 0) { clear_error(); return GpuError::Ok; }
        if (!inputs || !input_lens || !out32) return set_error(GpuError::NullArg, "NULL buffer");
        if (stride == 0) return set_error(GpuError::BadInput, "stride out of range");

        uint8_t *d_in=nullptr,*d_out=nullptr; uint32_t* d_lens=nullptr;
        GpuError ret = GpuError::Ok;
        const int n_int = static_cast<int>(n);
        const int threads = 256;
        const int blocks  = (n_int + threads - 1) / threads;

        if (cudaMalloc(&d_in, n*stride)!=cudaSuccess){ ret=set_error(GpuError::Memory,"hash256var d_in"); goto h2v_cleanup; }
        if (cudaMalloc(&d_lens, n*sizeof(uint32_t))!=cudaSuccess){ ret=set_error(GpuError::Memory,"hash256var d_lens"); goto h2v_cleanup; }
        if (cudaMalloc(&d_out, n*32)!=cudaSuccess){ ret=set_error(GpuError::Memory,"hash256var d_out"); goto h2v_cleanup; }
        if (cudaMemcpy(d_in, inputs, n*stride, cudaMemcpyHostToDevice)!=cudaSuccess){ ret=set_error(GpuError::Launch,"hash256var up in"); goto h2v_cleanup; }
        if (cudaMemcpy(d_lens, input_lens, n*sizeof(uint32_t), cudaMemcpyHostToDevice)!=cudaSuccess){ ret=set_error(GpuError::Launch,"hash256var up lens"); goto h2v_cleanup; }
        lbtc_hash256_var_kernel<<<blocks,threads>>>(d_in, d_lens, stride, n_int, d_out);
        if (cudaGetLastError()!=cudaSuccess){ ret=set_error(GpuError::Launch,"hash256var kernel"); goto h2v_cleanup; }
        if (cudaDeviceSynchronize()!=cudaSuccess){ ret=set_error(GpuError::Launch,"hash256var sync"); goto h2v_cleanup; }
        if (cudaMemcpy(out32, d_out, n*32, cudaMemcpyDeviceToHost)!=cudaSuccess){ ret=set_error(GpuError::Launch,"hash256var download"); goto h2v_cleanup; }
        clear_error();
h2v_cleanup:
        if(d_in)cudaFree(d_in); if(d_lens)cudaFree(d_lens); if(d_out)cudaFree(d_out);
        return ret;
    }

    /* libbitcoin-bridge: batch Merkle-tree parent hashing. PUBLIC.
     * out32[i] = SHA256(SHA256(left32[i*32..] || right32[i*32..])), SoA inputs
     * (left32/right32 are separate n*32-byte buffers, not interleaved). */
    GpuError merkle_pair_hash(
        const uint8_t* left32, const uint8_t* right32, size_t n, uint8_t* out32) override
    {
        if (!ready_) return set_error(GpuError::Device, "context not initialised");
        if (n == 0) { clear_error(); return GpuError::Ok; }
        if (!left32 || !right32 || !out32) return set_error(GpuError::NullArg, "NULL buffer");

        uint8_t *d_left=nullptr,*d_right=nullptr,*d_out=nullptr;
        GpuError ret = GpuError::Ok;
        const int n_int = static_cast<int>(n);
        const int threads = 256;
        const int blocks  = (n_int + threads - 1) / threads;

        if (cudaMalloc(&d_left, n*32)!=cudaSuccess){ ret=set_error(GpuError::Memory,"merklepair d_left"); goto mp_cleanup; }
        if (cudaMalloc(&d_right, n*32)!=cudaSuccess){ ret=set_error(GpuError::Memory,"merklepair d_right"); goto mp_cleanup; }
        if (cudaMalloc(&d_out, n*32)!=cudaSuccess){ ret=set_error(GpuError::Memory,"merklepair d_out"); goto mp_cleanup; }
        if (cudaMemcpy(d_left, left32, n*32, cudaMemcpyHostToDevice)!=cudaSuccess){ ret=set_error(GpuError::Launch,"merklepair up left"); goto mp_cleanup; }
        if (cudaMemcpy(d_right, right32, n*32, cudaMemcpyHostToDevice)!=cudaSuccess){ ret=set_error(GpuError::Launch,"merklepair up right"); goto mp_cleanup; }
        lbtc_merkle_pair_kernel<<<blocks,threads>>>(d_left, d_right, n_int, d_out);
        if (cudaGetLastError()!=cudaSuccess){ ret=set_error(GpuError::Launch,"merklepair kernel"); goto mp_cleanup; }
        if (cudaDeviceSynchronize()!=cudaSuccess){ ret=set_error(GpuError::Launch,"merklepair sync"); goto mp_cleanup; }
        if (cudaMemcpy(out32, d_out, n*32, cudaMemcpyDeviceToHost)!=cudaSuccess){ ret=set_error(GpuError::Launch,"merklepair download"); goto mp_cleanup; }
        clear_error();
mp_cleanup:
        if(d_left)cudaFree(d_left); if(d_right)cudaFree(d_right); if(d_out)cudaFree(d_out);
        return ret;
    }

    /* libbitcoin-bridge: batch sighash descriptor hash. PUBLIC-DATA ONLY.
     * Parses the compact descriptor bytecode on the host, copies referenced
     * field columns to the GPU, then each thread streams its row's fields
     * directly through SHA-256 -- NO per-row preimage buffer is materialised
     * on the CPU or GPU.
     *
     * Descriptor format (v2, accepted Phase 3a):
     *   LE field_ref pairs, terminated by 0xFF at odd-aligned position.
     *   field_ref = (field_id & 0xFF) | ((field_id>>8) << 0) in byte0,
     *               flags nibble in byte1[7:4], field_id>>8 in byte1[3:0].
     *   1..129 bytes, odd length. All field_ids with low byte 0xFF reserved. */
    GpuError sighash_descriptor_hash(
        const uint8_t* descriptor, size_t descriptor_len,
        const uint8_t* const* field_data, const uint32_t* field_lengths,
        const uint32_t* const* field_var_lens, size_t count,
        uint8_t* out32) override
    {
        if (!ready_) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!descriptor || !field_data || !field_lengths || !out32)
            return set_error(GpuError::NullArg, "NULL buffer");

        // ── Host-side descriptor parse ──────────────────────────────
        constexpr int MAX_FIELDS = 64;
        struct HostPlan { uint16_t fid; bool has_len; uint32_t fixed_len; };
        HostPlan hplan[MAX_FIELDS];
        int num_fields = 0;
        bool has_nHashType = false;

        if (descriptor_len < 1 || descriptor_len > 129) return set_error(GpuError::BadInput, "descriptor_len");
        if ((descriptor_len & 1u) == 0) return set_error(GpuError::BadInput, "descriptor_len even");
        if (descriptor[descriptor_len - 1] != 0xFF) return set_error(GpuError::BadInput, "no terminator");
        for (size_t i = 0; i < descriptor_len - 1; i += 2)
            if (descriptor[i] == 0xFF) return set_error(GpuError::BadInput, "mid-0xFF");

        const uint8_t* p = descriptor;
        const uint8_t* dend = descriptor + descriptor_len;
        while (p < dend && *p != 0xFF) {
            if (num_fields >= MAX_FIELDS) return set_error(GpuError::BadInput, "too many fields");
            if (p + 1 >= dend) return set_error(GpuError::BadInput, "truncated ref");
            uint16_t fid = (uint16_t)p[0] | ((uint16_t)(p[1] & 0x0Fu) << 8);
            uint8_t flags_nib = p[1] >> 4;
            if ((flags_nib & 0x0Cu) != 0) return set_error(GpuError::BadInput, "reserved flags");
            if ((fid & 0xFFu) == 0xFFu) return set_error(GpuError::BadInput, "reserved low-byte");
            for (int j = 0; j < num_fields; ++j)
                if (hplan[j].fid == fid) return set_error(GpuError::BadInput, "duplicate field");
            bool has_len = (flags_nib & 0x01u) != 0;
            uint32_t flen = 0;
            switch (fid) {
            case 0x00: flen=4;  break; case 0x01: flen=32; break; case 0x02: flen=32; break;
            case 0x03: flen=36; break; case 0x04:/*var*/break; case 0x05: flen=8;  break;
            case 0x06: flen=4;  break; case 0x07: flen=32; break; case 0x08: flen=4;  break;
            case 0x09: flen=4; has_nHashType=true; break;
            case 0x0A: flen=36; break; case 0x0B: flen=4;  break;
            // 0x0C..0x0F reserved for future Taproot tagged-hash mode.
            // Reject them until a separately reviewed tagged-hash mode exists
            // (coordinated with OpenCL/Metal parses in the Claude card).
            case 0x0C: case 0x0D: case 0x0E: case 0x0F:
                return set_error(GpuError::BadInput, "Taproot field_id reserved");
            case 0xF0:/*var*/break;
            default: return set_error(GpuError::BadInput, "unsupported field_id");
            }
            bool is_var = (flen == 0);
            if (is_var && !has_len) return set_error(GpuError::BadInput, "var w/o HAS_LENGTH");
            if (!is_var && has_len) return set_error(GpuError::BadInput, "fixed w/ HAS_LENGTH");
            if (!field_data[fid]) return set_error(GpuError::BadInput, "null field_data");
            if (field_lengths[fid] == 0) return set_error(GpuError::BadInput, "zero stride");
            // Fixed-field stride must be ≥ its fixed serialized length.
            if (!is_var && field_lengths[fid] < flen) return set_error(GpuError::BadInput, "stride < fixed_len");
            // Bounds: count * stride must not overflow size_t.
            if (count > SIZE_MAX / (size_t)field_lengths[fid])
                return set_error(GpuError::BadInput, "count*stride overflow");
            hplan[num_fields].fid = fid;
            hplan[num_fields].has_len = has_len;
            hplan[num_fields].fixed_len = flen;
            ++num_fields;
            p += 2;
        }
        if (p != dend - 1) return set_error(GpuError::BadInput, "bad terminator pos");
        if (!has_nHashType) return set_error(GpuError::BadInput, "missing nHashType");

        // ── Bounds: variable-length fields + per-row 4 MiB limit ────
        // Every var-len row must be ≤ its column stride.
        // Per-row actual preimage (fixed_len + var_len[i]) must be ≤ 4 MiB.
        // Uses actual per-row var_lens, not stride, so large strides with
        // short valid rows are accepted while oversized rows are rejected.
        constexpr size_t MAX_PREIMAGE = 4u * 1024u * 1024u;  // 4 MiB

        for (int fi = 0; fi < num_fields; ++fi) {
            if (hplan[fi].fixed_len != 0) continue;
            uint16_t fid = hplan[fi].fid;
            if (!field_var_lens || !field_var_lens[fid])
                return set_error(GpuError::BadInput, "null var_lens");
            uint32_t stride = field_lengths[fid];
            for (size_t r = 0; r < count; ++r) {
                if (field_var_lens[fid][r] > stride)
                    return set_error(GpuError::BadInput, "var_len > stride");
            }
        }

        // Per-row actual preimage length ≤ 4 MiB (checked-add, not stride).
        for (size_t r = 0; r < count; ++r) {
            size_t row_len = 0;
            for (int fi = 0; fi < num_fields; ++fi) {
                if (hplan[fi].fixed_len > 0) {
                    if (hplan[fi].fixed_len > MAX_PREIMAGE - row_len)
                        return set_error(GpuError::BadInput, "preimage > 4 MiB");
                    row_len += hplan[fi].fixed_len;
                } else {
                    uint32_t vl = field_var_lens[hplan[fi].fid][r];
                    if (vl > MAX_PREIMAGE - row_len)
                        return set_error(GpuError::BadInput, "preimage > 4 MiB");
                    row_len += vl;
                }
            }
            if (row_len > MAX_PREIMAGE)
                return set_error(GpuError::BadInput, "preimage > 4 MiB");
        }

        // ── Upload column data to GPU (reusable buffers) ────────────
        // Column data + var-lens go into one grow-only pool;
        // output and metadata tables each get their own reusable allocation.
        // No per-call cudaMalloc/cudaFree on the hot path — allocations
        // grow once and persist until thread exit.

        // count must fit in CUDA kernel index type (int).
        if (count > static_cast<size_t>(INT32_MAX))
            return set_error(GpuError::BadInput, "count > INT32_MAX");

        const int n_int = static_cast<int>(count);
        const int threads = 256;
        const int blocks = (n_int + threads - 1) / threads;

        // ── Checked pool-size computation ───────────────────────────
        // Pool holds: [col_0 | col_1 | ... | align | var-len_0 | var-len_1 | ...]
        // Each column is count*stride bytes (byte-aligned).
        // Each var-len array is count*uint32_t, aligned to alignof(uint32_t)=4.
        // Overflow is checked additively before accumulation.

        constexpr size_t VARLEN_ALIGN = alignof(uint32_t);  // 4
        constexpr size_t PTR_ALIGN    = alignof(uint8_t*);   // 8
        constexpr size_t PLAN_ALIGN   = alignof(DeviceSighashFieldPlan); // 8

        // Pre-check: count*stride must not overflow size_t for any field.
        for (int fi = 0; fi < num_fields; ++fi) {
            uint16_t fid = hplan[fi].fid;
            if (field_lengths[fid] > 0 && count > SIZE_MAX / field_lengths[fid])
                return set_error(GpuError::BadInput, "count*stride overflow");
        }
        // count * 32 must not overflow (output buffer).
        if (count > SIZE_MAX / 32u)
            return set_error(GpuError::BadInput, "count*32 overflow");

        // checked_add helper (returns false on overflow).
        auto checked_add = [](size_t& acc, size_t add) -> bool {
            if (add > SIZE_MAX - acc) return false;
            acc += add; return true;
        };
        auto align_up = [](size_t x, size_t a) -> size_t {
            return (x + a - 1u) & ~(a - 1u);
        };

        size_t pool_bytes = 0;
        for (int fi = 0; fi < num_fields; ++fi) {
            uint16_t fid = hplan[fi].fid;
            size_t col_bytes = (size_t)count * field_lengths[fid];
            if (!checked_add(pool_bytes, col_bytes))
                return set_error(GpuError::BadInput, "pool overflow cols");
            if (hplan[fi].fixed_len == 0) {
                // Align pool_off to uint32_t boundary before var-len array.
                size_t var_bytes = (size_t)count * sizeof(uint32_t);
                pool_bytes = align_up(pool_bytes, VARLEN_ALIGN);
                if (!checked_add(pool_bytes, var_bytes))
                    return set_error(GpuError::BadInput, "pool overflow varlens");
            }
        }

        // Compute aligned meta-buffer layout:
        //   [plan: align=PLAN_ALIGN] [field_ptrs: align=PTR_ALIGN]
        //   [strides: align=4]       [var_len_ptrs: align=PTR_ALIGN]
        size_t plan_sz       = (size_t)num_fields * sizeof(DeviceSighashFieldPlan);
        size_t fptrs_sz      = (size_t)num_fields * sizeof(uint8_t*);
        size_t strides_sz    = (size_t)num_fields * sizeof(uint32_t);
        size_t vlptrs_sz     = (size_t)num_fields * sizeof(uint32_t*);
        size_t meta_off      = 0;
        meta_off = align_up(meta_off, PLAN_ALIGN);
        size_t plan_off      = meta_off; meta_off += plan_sz;
        meta_off = align_up(meta_off, PTR_ALIGN);
        size_t fptrs_off     = meta_off; meta_off += fptrs_sz;
        meta_off = align_up(meta_off, alignof(uint32_t));
        size_t strides_off   = meta_off; meta_off += strides_sz;
        meta_off = align_up(meta_off, PTR_ALIGN);
        size_t vlptrs_off    = meta_off; meta_off += vlptrs_sz;
        size_t meta_bytes    = meta_off;

        // ── Allocate / grow reusable buffers ────────────────────────
        GpuError ret = GpuError::Ok;
        if (g_cuda_batch_scratch.ensure_lbtc_sighash_pool(pool_bytes) != cudaSuccess) {
            ret = set_error(GpuError::Memory, "sighash pool"); goto sh_decline;
        }
        if (g_cuda_batch_scratch.ensure_lbtc_sighash_out(count * 32) != cudaSuccess) {
            ret = set_error(GpuError::Memory, "sighash out"); goto sh_decline;
        }
        if (g_cuda_batch_scratch.ensure_lbtc_sighash_meta(meta_bytes) != cudaSuccess) {
            ret = set_error(GpuError::Memory, "sighash meta"); goto sh_decline;
        }

        // ── Sub-allocate + upload (scoped: no goto crosses initialisations) ──
        // nvcc forbids goto that bypasses variable initialisation, so this
        // entire block is wrapped in {}.  goto sh_decline from inside this
        // block exits the scope, destroying locals, before reaching the label.
        {
            uint8_t* pool_base = g_cuda_batch_scratch.lbtc_sighash_pool;
            size_t pool_off = 0;

            uint8_t*  d_cols[MAX_FIELDS] = {};
            uint32_t* d_varlens[MAX_FIELDS] = {};
            uint8_t*  h_field_ptrs[MAX_FIELDS] = {};
            uint32_t* h_var_len_ptrs[MAX_FIELDS] = {};
            uint32_t  h_strides[MAX_FIELDS] = {};

            for (int fi = 0; fi < num_fields; ++fi) {
                uint16_t fid = hplan[fi].fid;
                uint32_t stride = field_lengths[fid];
                h_strides[fi] = stride;

                size_t col_bytes = (size_t)count * stride;
                d_cols[fi] = pool_base + pool_off;
                pool_off += col_bytes;
                if (cudaMemcpy(d_cols[fi], field_data[fid], col_bytes,
                               cudaMemcpyHostToDevice) != cudaSuccess) {
                    ret = set_error(GpuError::Launch, "sighash up col"); goto sh_decline;
                }
                h_field_ptrs[fi] = d_cols[fi];

                if (hplan[fi].fixed_len == 0) {
                    // Align to uint32_t before var-len array.
                    pool_off = align_up(pool_off, VARLEN_ALIGN);
                    d_varlens[fi] = reinterpret_cast<uint32_t*>(pool_base + pool_off);
                    pool_off += count * sizeof(uint32_t);
                    if (cudaMemcpy(d_varlens[fi], field_var_lens[fid],
                                   count * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice) != cudaSuccess) {
                        ret = set_error(GpuError::Launch, "sighash up varlen"); goto sh_decline;
                    }
                    h_var_len_ptrs[fi] = d_varlens[fi];
                } else {
                    h_var_len_ptrs[fi] = nullptr;
                }
            }

            // Sub-allocate metadata from the reusable meta buffer (aligned).
            uint8_t* meta_base = g_cuda_batch_scratch.lbtc_sighash_meta;
            DeviceSighashFieldPlan* d_plan =
                reinterpret_cast<DeviceSighashFieldPlan*>(meta_base + plan_off);
            uint8_t**  d_field_ptrs =
                reinterpret_cast<uint8_t**>(meta_base + fptrs_off);
            uint32_t*  d_strides =
                reinterpret_cast<uint32_t*>(meta_base + strides_off);
            uint32_t** d_var_len_ptrs =
                reinterpret_cast<uint32_t**>(meta_base + vlptrs_off);

            DeviceSighashFieldPlan h_plan_arr[MAX_FIELDS];
            for (int fi = 0; fi < num_fields; ++fi) {
                h_plan_arr[fi].field_id  = hplan[fi].fid;
                h_plan_arr[fi].flags     = hplan[fi].has_len ? 1u : 0u;
                h_plan_arr[fi].pad       = 0;
                h_plan_arr[fi].fixed_len = hplan[fi].fixed_len;
            }
            if (cudaMemcpy(d_plan, h_plan_arr,
                           num_fields * sizeof(DeviceSighashFieldPlan),
                           cudaMemcpyHostToDevice) != cudaSuccess) {
                ret = set_error(GpuError::Launch, "sighash up plan"); goto sh_decline;
            }
            if (cudaMemcpy(d_field_ptrs, h_field_ptrs,
                           num_fields * sizeof(uint8_t*),
                           cudaMemcpyHostToDevice) != cudaSuccess) {
                ret = set_error(GpuError::Launch, "sighash up ptrs"); goto sh_decline;
            }
            if (cudaMemcpy(d_strides, h_strides,
                           num_fields * sizeof(uint32_t),
                           cudaMemcpyHostToDevice) != cudaSuccess) {
                ret = set_error(GpuError::Launch, "sighash up strides"); goto sh_decline;
            }
            if (cudaMemcpy(d_var_len_ptrs, h_var_len_ptrs,
                           num_fields * sizeof(uint32_t*),
                           cudaMemcpyHostToDevice) != cudaSuccess) {
                ret = set_error(GpuError::Launch, "sighash up vlptrs"); goto sh_decline;
            }

            uint8_t* d_out = g_cuda_batch_scratch.lbtc_sighash_out;

            // Launch kernel.
            lbtc_sighash_descriptor_hash_kernel<<<blocks, threads>>>(
                d_plan, num_fields, d_field_ptrs, d_strides, d_var_len_ptrs,
                n_int, d_out);
            if (cudaGetLastError() != cudaSuccess) {
                ret = set_error(GpuError::Launch, "sighash kernel"); goto sh_decline;
            }
            if (cudaDeviceSynchronize() != cudaSuccess) {
                ret = set_error(GpuError::Launch, "sighash sync"); goto sh_decline;
            }

            // Download results.
            if (cudaMemcpy(out32, d_out, count * 32, cudaMemcpyDeviceToHost) != cudaSuccess) {
                ret = set_error(GpuError::Launch, "sighash download"); goto sh_decline;
            }
            clear_error();
        }

sh_decline:
        // Reusable buffers persist across calls — do NOT free them here.
        // On error, out32 was cleared by the C ABI wrapper (fail-closed).
        return ret;
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

#if SECP256K1_GPU_HAS_FROST
        uint8_t *d_z = nullptr, *d_D = nullptr, *d_E = nullptr, *d_Y = nullptr;
        uint8_t *d_rho = nullptr, *d_lie = nullptr;
        uint8_t *d_nR = nullptr, *d_nK = nullptr, *d_res = nullptr;

        CUDA_TRY(cudaMalloc(&d_z,   count * 32));
        CudaBufferGuard d_z_guard(d_z);
        CUDA_TRY(cudaMalloc(&d_D,   count * 33));
        CudaBufferGuard d_D_guard(d_D);
        CUDA_TRY(cudaMalloc(&d_E,   count * 33));
        CudaBufferGuard d_E_guard(d_E);
        CUDA_TRY(cudaMalloc(&d_Y,   count * 33));
        CudaBufferGuard d_Y_guard(d_Y);
        CUDA_TRY(cudaMalloc(&d_rho, count * 32));
        CudaBufferGuard d_rho_guard(d_rho);
        CUDA_TRY(cudaMalloc(&d_lie, count * 32));
        CudaBufferGuard d_lie_guard(d_lie);
        CUDA_TRY(cudaMalloc(&d_nR,  count));
        CudaBufferGuard d_nR_guard(d_nR);
        CUDA_TRY(cudaMalloc(&d_nK,  count));
        CudaBufferGuard d_nK_guard(d_nK);
        CUDA_TRY(cudaMalloc(&d_res, count));
        CudaBufferGuard d_res_guard(d_res);

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

        clear_error();
        return GpuError::Ok;
#else
        return set_error(GpuError::Unsupported, "GPU FROST module disabled at build time");
#endif
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

#if SECP256K1_GPU_HAS_ECRECOVER
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
#else
        return set_error(GpuError::Unsupported, "GPU ECRECOVER module disabled at build time");
#endif
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

#if SECP256K1_GPU_HAS_ZK
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
#else
        return set_error(GpuError::Unsupported, "GPU ZK module disabled at build time");
#endif
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

#if SECP256K1_GPU_HAS_ZK
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
#else
        return set_error(GpuError::Unsupported, "GPU ZK module disabled at build time");
#endif
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
#if SECP256K1_GPU_HAS_ZK
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
#else
        return set_error(GpuError::Unsupported, "GPU ZK module disabled at build time");
#endif
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

#if SECP256K1_GPU_HAS_BIP324
        const size_t wire_stride = max_payload + 19;
        uint8_t*  d_keys   = nullptr;
        uint8_t*  d_nonces = nullptr;
        uint8_t*  d_pt     = nullptr;
        uint32_t* d_sizes  = nullptr;
        uint8_t*  d_wire   = nullptr;
        GpuError  ret      = GpuError::Ok;

        if (cudaMalloc(&d_keys,   count * 32)               != cudaSuccess) {
            ret = set_error(GpuError::Memory, "bip324e key");   goto enc_cleanup; }
        if (cudaMalloc(&d_nonces, count * 12)               != cudaSuccess) {
            ret = set_error(GpuError::Memory, "bip324e nonce"); goto enc_cleanup; }
        if (cudaMalloc(&d_pt,     count * max_payload)      != cudaSuccess) {
            ret = set_error(GpuError::Memory, "bip324e pt");    goto enc_cleanup; }
        if (cudaMalloc(&d_sizes,  count * sizeof(uint32_t)) != cudaSuccess) {
            ret = set_error(GpuError::Memory, "bip324e sizes"); goto enc_cleanup; }
        if (cudaMalloc(&d_wire,   count * wire_stride)      != cudaSuccess) {
            ret = set_error(GpuError::Memory, "bip324e wire");  goto enc_cleanup; }

        if (cudaMemcpy(d_keys,   keys32,     count * 32,               cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_nonces, nonces12,   count * 12,               cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_pt,     plaintexts, count * max_payload,      cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_sizes,  sizes,      count * sizeof(uint32_t), cudaMemcpyHostToDevice) != cudaSuccess) {
            ret = set_error(GpuError::Launch, "bip324e upload"); goto enc_cleanup; }
        {
            int threads = 128;
            int blocks  = (static_cast<int>(count) + threads - 1) / threads;
            secp256k1::cuda::bip324::bip324_aead_encrypt_kernel<<<blocks, threads>>>(
                d_keys, d_nonces, d_pt, d_sizes, d_wire,
                max_payload, static_cast<int>(count));
            if (cudaGetLastError()      != cudaSuccess ||
                cudaDeviceSynchronize() != cudaSuccess) {
                ret = set_error(GpuError::Launch, "bip324e kernel"); goto enc_cleanup; }
            if (cudaMemcpy(wire_out, d_wire, count * wire_stride,
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
                ret = set_error(GpuError::Launch, "bip324e download"); goto enc_cleanup; }
            clear_error();
        }

enc_cleanup:
        // NF-01b: Rule 10 — zero AES session keys on ALL exit paths (success + error)
        if (d_keys) { cudaMemset(d_keys, 0, count * 32); cudaFree(d_keys); }
        if (d_nonces) cudaFree(d_nonces);
        if (d_pt)     cudaFree(d_pt);
        if (d_sizes)  cudaFree(d_sizes);
        if (d_wire)   cudaFree(d_wire);
        return ret;
#else
        return set_error(GpuError::Unsupported, "GPU BIP-324 module disabled at build time");
#endif
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

#if SECP256K1_GPU_HAS_BIP324
        const size_t wire_stride = max_payload + 19;
        uint8_t*  d_keys   = nullptr;
        uint8_t*  d_nonces = nullptr;
        uint8_t*  d_wire   = nullptr;
        uint32_t* d_sizes  = nullptr;
        uint8_t*  d_pt     = nullptr;
        uint32_t* d_ok     = nullptr;
        GpuError  ret      = GpuError::Ok;

        if (cudaMalloc(&d_keys,   count * 32)               != cudaSuccess) {
            ret = set_error(GpuError::Memory, "bip324d key");   goto dec_cleanup; }
        if (cudaMalloc(&d_nonces, count * 12)               != cudaSuccess) {
            ret = set_error(GpuError::Memory, "bip324d nonce"); goto dec_cleanup; }
        if (cudaMalloc(&d_wire,   count * wire_stride)      != cudaSuccess) {
            ret = set_error(GpuError::Memory, "bip324d wire");  goto dec_cleanup; }
        if (cudaMalloc(&d_sizes,  count * sizeof(uint32_t)) != cudaSuccess) {
            ret = set_error(GpuError::Memory, "bip324d sizes"); goto dec_cleanup; }
        if (cudaMalloc(&d_pt,     count * max_payload)      != cudaSuccess) {
            ret = set_error(GpuError::Memory, "bip324d pt");    goto dec_cleanup; }
        if (cudaMalloc(&d_ok,     count * sizeof(uint32_t)) != cudaSuccess) {
            ret = set_error(GpuError::Memory, "bip324d ok");    goto dec_cleanup; }

        if (cudaMemcpy(d_keys,   keys32,   count * 32,               cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_nonces, nonces12, count * 12,               cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_wire,   wire_in,  count * wire_stride,      cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_sizes,  sizes,    count * sizeof(uint32_t), cudaMemcpyHostToDevice) != cudaSuccess) {
            ret = set_error(GpuError::Launch, "bip324d upload"); goto dec_cleanup; }
        {
            int threads = 128;
            int blocks  = (static_cast<int>(count) + threads - 1) / threads;
            secp256k1::cuda::bip324::bip324_aead_decrypt_kernel<<<blocks, threads>>>(
                d_keys, d_nonces, d_wire, d_sizes, d_pt, d_ok,
                max_payload, static_cast<int>(count));
            if (cudaGetLastError()      != cudaSuccess ||
                cudaDeviceSynchronize() != cudaSuccess) {
                ret = set_error(GpuError::Launch, "bip324d kernel"); goto dec_cleanup; }
            if (cudaMemcpy(plaintext_out, d_pt, count * max_payload,
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
                ret = set_error(GpuError::Launch, "bip324d pt dl"); goto dec_cleanup; }
            {
                std::vector<uint32_t> h_ok(count);
                if (cudaMemcpy(h_ok.data(), d_ok, count * sizeof(uint32_t),
                               cudaMemcpyDeviceToHost) != cudaSuccess) {
                    ret = set_error(GpuError::Launch, "bip324d ok dl"); goto dec_cleanup; }
                for (size_t i = 0; i < count; ++i)
                    out_valid[i] = h_ok[i] ? 1 : 0;
            }
            clear_error();
        }

dec_cleanup:
        // NF-01b: Rule 10 — zero AES session keys on ALL exit paths (success + error)
        if (d_keys) { cudaMemset(d_keys, 0, count * 32); cudaFree(d_keys); }
        if (d_ok)     cudaFree(d_ok);
        if (d_pt)     cudaFree(d_pt);
        if (d_sizes)  cudaFree(d_sizes);
        if (d_wire)   cudaFree(d_wire);
        if (d_nonces) cudaFree(d_nonces);
        return ret;
#else
        return set_error(GpuError::Unsupported, "GPU BIP-324 module disabled at build time");
#endif
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

#if SECP256K1_GPU_HAS_ZK
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
        CudaBufferGuard d_msgs_guard(d_msgs);
        CUDA_TRY(cudaMalloc(&d_pubs33, count * 33));
        CudaBufferGuard d_pubs33_guard(d_pubs33);
        CUDA_TRY(cudaMalloc(&d_pubs,   count * sizeof(JacobianPoint)));
        CudaBufferGuard d_pubs_guard(d_pubs);
        CUDA_TRY(cudaMalloc(&d_pub_ok, count * sizeof(bool)));
        CudaBufferGuard d_pub_ok_guard(d_pub_ok);
        CUDA_TRY(cudaMalloc(&d_sigs,   count * sizeof(ECDSASignatureGPU)));
        CudaBufferGuard d_sigs_guard(d_sigs);
        CUDA_TRY(cudaMalloc(&d_out,    count * sizeof(EcdsaSnarkWitnessFlat)));
        CudaBufferGuard d_out_guard(d_out);

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

        clear_error();
        return GpuError::Ok;
#else
        return set_error(GpuError::Unsupported, "GPU ZK module disabled at build time");
#endif
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

#if SECP256K1_GPU_HAS_ZK
        uint8_t*                      d_msgs   = nullptr;
        uint8_t*                      d_pubs   = nullptr;
        uint8_t*                      d_sigs   = nullptr;
        cuda::SchnorrSnarkWitnessFlat* d_out   = nullptr;

        CUDA_TRY(cudaMalloc(&d_msgs, count * 32));
        CudaBufferGuard d_msgs_guard(d_msgs);
        CUDA_TRY(cudaMalloc(&d_pubs, count * 32));
        CudaBufferGuard d_pubs_guard(d_pubs);
        CUDA_TRY(cudaMalloc(&d_sigs, count * 64));
        CudaBufferGuard d_sigs_guard(d_sigs);
        CUDA_TRY(cudaMalloc(&d_out,  count * sizeof(cuda::SchnorrSnarkWitnessFlat)));
        CudaBufferGuard d_out_guard(d_out);

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

        clear_error();
        return GpuError::Ok;
#else
        return set_error(GpuError::Unsupported, "GPU ZK module disabled at build time");
#endif
    }

    /* -- BIP-352 Silent Payment GPU batch scan (multi-spend, issue #335) --- */

    GpuError bip352_scan_batch_multispend(
        const uint8_t  scan_privkey32[32],
        const uint8_t* spend_pubkeys33,
        size_t n_spend,
        const uint8_t* tweak_pubkeys33,
        size_t n_tweaks,
        uint64_t* prefix64_out) override
    {
        if (!ready_) return set_error(GpuError::Device, "context not initialised");
        if (n_tweaks == 0 || n_spend == 0) { clear_error(); return GpuError::Ok; }
        if (!scan_privkey32 || !spend_pubkeys33 || !tweak_pubkeys33 || !prefix64_out)
            return set_error(GpuError::NullArg, "NULL buffer");

#if SECP256K1_GPU_HAS_BIP352
        // Repair (issue #335 acceptance repair): n_tweaks/n_spend/their
        // product ARE bounds-checked at the ABI layer
        // (ufsecp_gpu_bip352_scan_batch_multispend, kMaxGpuBatchN /
        // kMaxBip352Spend, src/cpu/src/ufsecp_gpu_impl.cpp) -- but this
        // method is a public GpuBackend virtual reachable by direct C++
        // callers that bypass the ABI wrapper's own checks, and the
        // static_cast<int>(n_spend)/static_cast<int>(n_tweaks) narrowing
        // casts used for kernel launch grid dimensions below are only safe
        // because those bounds hold. Re-validate locally as defense-in-depth
        // (mirrors the equivalent local check added to the OpenCL override,
        // gpu_backend_opencl.cpp, in the same repair pass). Caps duplicated
        // rather than shared because kMaxGpuBatchN/kMaxBip352Spend live in a
        // different translation unit with no common header for GPU batch
        // limits.
        static constexpr size_t kLocalMaxGpuBatchN   = size_t{1} << 26; // 64M
        static constexpr size_t kLocalMaxBip352Spend = size_t{1} << 16; // 65536
        if (n_spend > kLocalMaxBip352Spend)
            return set_error(GpuError::BadInput, "n_spend too large");
        if (n_tweaks > kLocalMaxGpuBatchN)
            return set_error(GpuError::BadInput, "n_tweaks too large");
        if (n_spend != 0 && n_tweaks > std::numeric_limits<size_t>::max() / n_spend)
            return set_error(GpuError::BadInput, "n_tweaks * n_spend overflow");
        const size_t n_rows = n_tweaks * n_spend;
        if (n_rows > kLocalMaxGpuBatchN)
            return set_error(GpuError::BadInput, "n_tweaks * n_spend too large");

        // Repair (issue #335 acceptance repair): fail-closed output guard.
        // CUDA_TRY(...) below performs a bare `return last_error();` on any
        // CUDA driver/kernel failure -- previously that left prefix64_out
        // completely untouched by this method on failure (fail-closed
        // behavior existed only via the ABI wrapper's own pre-zero +
        // to_abi_error_clear_on_fail, not from this backend method itself).
        // This RAII guard zeroes prefix64_out (n_rows -- already validated
        // above -- x 8 bytes) on ANY exit path where `success` was not
        // explicitly set true just before the final Ok return, exactly
        // mirroring how CudaKeyGuard already guarantees secret device
        // buffers are erased on every CUDA_TRY early return.
        struct FailClosedOutputGuard {
            uint64_t* out;
            size_t    n_rows;
            bool      success = false;
            ~FailClosedOutputGuard() {
                if (!success && out) std::memset(out, 0, sizeof(uint64_t) * n_rows);
            }
        } out_guard{prefix64_out, n_rows};

        /* -- 1. Convert scan key to Scalar on CPU -- */
        {
            secp256k1::fast::Scalar scan_check;
            if (!secp256k1::fast::Scalar::parse_bytes_strict_nonzero(
                    scan_privkey32, scan_check)) {
                secp256k1::detail::secure_erase(&scan_check, sizeof(scan_check));
                return set_error(GpuError::BadKey,
                                 "invalid scan key (zero or >= group order)");
            }
            secp256k1::detail::secure_erase(&scan_check, sizeof(scan_check));
        }

        // Rule 10: RAII wrapper zeroes h_scan_k on all exit paths.
        struct ScanKeyGuard {
            cuda::Scalar k{};
            ~ScanKeyGuard() { secp256k1::detail::secure_erase(&k, sizeof(k)); }
        } scan_key_guard;
        cuda::Scalar& h_scan_k = scan_key_guard.k;
        for (int limb = 0; limb < 4; ++limb) {
            uint64_t v = 0;
            int base = (3 - limb) * 8;
            for (int b = 0; b < 8; ++b) v = (v << 8) | scan_privkey32[base + b];
            h_scan_k.limbs[limb] = v;
        }

        /* -- 2. Upload 33-byte pubkeys directly (GPU decompresses in scan kernel) -- */

        cuda::Scalar* d_scan_k = nullptr;
        CUDA_TRY(cudaMalloc(&d_scan_k, sizeof(cuda::Scalar)));
        CudaKeyGuard d_scan_k_guard(d_scan_k, sizeof(cuda::Scalar));
        CUDA_TRY(cudaMemcpy(d_scan_k, &h_scan_k, sizeof(cuda::Scalar), cudaMemcpyHostToDevice));

        uint8_t* d_tweaks33 = nullptr;
        CUDA_TRY(cudaMalloc(&d_tweaks33, n_tweaks * 33));
        CudaKeyGuard d_tweaks33_guard(d_tweaks33, n_tweaks * 33);
        CUDA_TRY(cudaMemcpy(d_tweaks33, tweak_pubkeys33, n_tweaks * 33, cudaMemcpyHostToDevice));

        uint8_t* d_spend33 = nullptr;
        CUDA_TRY(cudaMalloc(&d_spend33, n_spend * 33));
        CudaKeyGuard d_spend33_guard(d_spend33, n_spend * 33);
        CUDA_TRY(cudaMemcpy(d_spend33, spend_pubkeys33, n_spend * 33, cudaMemcpyHostToDevice));

        cuda::AffinePoint* d_spend_aff = nullptr;
        CUDA_TRY(cudaMalloc(&d_spend_aff, n_spend * sizeof(cuda::AffinePoint)));
        CudaKeyGuard d_spend_aff_guard(d_spend_aff, n_spend * sizeof(cuda::AffinePoint));

        uint8_t* d_spend_valid = nullptr;
        CUDA_TRY(cudaMalloc(&d_spend_valid, n_spend * sizeof(uint8_t)));
        CudaKeyGuard d_spend_valid_guard(d_spend_valid, n_spend * sizeof(uint8_t));

        uint64_t* d_prefixes = nullptr;
        CUDA_TRY(cudaMalloc(&d_prefixes, n_rows * sizeof(uint64_t)));
        CudaKeyGuard d_prefixes_guard(d_prefixes, n_rows * sizeof(uint64_t));

        /* -- 3. Decompress the n_spend spend-key candidates ONCE, independent
         *      of n_tweaks (issue #335 Blocker 1: marginal cost per extra
         *      spend key must not redo per-tweak work). -- */
        {
            int threads = 128;
            int blocks  = (static_cast<int>(n_spend) + threads - 1) / threads;
            bip352_decompress_points_kernel<<<blocks, threads>>>(
                d_spend33, d_spend_aff, d_spend_valid, static_cast<int>(n_spend));
            CUDA_TRY(cudaGetLastError());
        }

        /* -- 4. Main scan kernel: 1 thread per tweak; ECDH/tagged-hash/hash×G
         *      computed once per thread, then n_spend mixed adds. -- */
        {
            int threads = 128;
            int blocks  = (static_cast<int>(n_tweaks) + threads - 1) / threads;
            bip352_scan_batch_multispend_kernel_compressed<<<blocks, threads>>>(
                d_tweaks33, d_scan_k, d_spend_aff, d_spend_valid,
                static_cast<int>(n_spend), d_prefixes, static_cast<int>(n_tweaks));
            CUDA_TRY(cudaGetLastError());
        }
        CUDA_TRY(cudaDeviceSynchronize());

        /* -- 5. Download prefixes -- */
        CUDA_TRY(cudaMemcpy(prefix64_out, d_prefixes,
                            n_rows * sizeof(uint64_t), cudaMemcpyDeviceToHost));

        // HIGH-3: zero device copy of scan private key before freeing device memory.
        cudaMemset(d_scan_k, 0, sizeof(cuda::Scalar));
        // HIGH-3: zero host copy of scan private key (also done by ScanKeyGuard dtor).
        secp256k1::detail::secure_erase(&h_scan_k, sizeof(h_scan_k));

        out_guard.success = true;  // prefix64_out now holds valid results -- do not zero it.
        clear_error();
        return GpuError::Ok;
#else
        return set_error(GpuError::Unsupported, "GPU BIP-352 module disabled at build time");
#endif
    }

    /* issue #335 acceptance repair: CUDA-event transfer/setup/kernel/readback
     * timing breakdown for bip352_scan_batch_multispend. Diagnostic/benchmark
     * use only (see GpuBackend::bip352_scan_batch_multispend_timed's doc
     * comment in gpu_backend.hpp -- not part of the ABI-stable C surface, not
     * subject to the GPU backend compute-parity rule). Structurally mirrors
     * bip352_scan_batch_multispend() above (same validation, same key
     * hygiene, same fail-closed guard) with cudaEvent_t timestamps inserted
     * at each phase boundary; kept as a separate method rather than
     * refactored to share a common body, to avoid touching the already
     * validated production dispatch path for a benchmark-only feature. */
    GpuError bip352_scan_batch_multispend_timed(
        const uint8_t  scan_privkey32[32],
        const uint8_t* spend_pubkeys33,
        size_t n_spend,
        const uint8_t* tweak_pubkeys33,
        size_t n_tweaks,
        uint64_t* prefix64_out,
        TimingBreakdownMs* timing_out) override
    {
        if (timing_out) *timing_out = TimingBreakdownMs{};
        if (!ready_) return set_error(GpuError::Device, "context not initialised");
        if (n_tweaks == 0 || n_spend == 0) { clear_error(); return GpuError::Ok; }
        if (!scan_privkey32 || !spend_pubkeys33 || !tweak_pubkeys33 || !prefix64_out)
            return set_error(GpuError::NullArg, "NULL buffer");

#if SECP256K1_GPU_HAS_BIP352
        static constexpr size_t kLocalMaxGpuBatchN   = size_t{1} << 26;
        static constexpr size_t kLocalMaxBip352Spend = size_t{1} << 16;
        if (n_spend > kLocalMaxBip352Spend)
            return set_error(GpuError::BadInput, "n_spend too large");
        if (n_tweaks > kLocalMaxGpuBatchN)
            return set_error(GpuError::BadInput, "n_tweaks too large");
        if (n_spend != 0 && n_tweaks > std::numeric_limits<size_t>::max() / n_spend)
            return set_error(GpuError::BadInput, "n_tweaks * n_spend overflow");
        const size_t n_rows = n_tweaks * n_spend;
        if (n_rows > kLocalMaxGpuBatchN)
            return set_error(GpuError::BadInput, "n_tweaks * n_spend too large");

        struct FailClosedOutputGuard {
            uint64_t* out;
            size_t    n_rows;
            bool      success = false;
            ~FailClosedOutputGuard() {
                if (!success && out) std::memset(out, 0, sizeof(uint64_t) * n_rows);
            }
        } out_guard{prefix64_out, n_rows};

        {
            secp256k1::fast::Scalar scan_check;
            if (!secp256k1::fast::Scalar::parse_bytes_strict_nonzero(
                    scan_privkey32, scan_check)) {
                secp256k1::detail::secure_erase(&scan_check, sizeof(scan_check));
                return set_error(GpuError::BadKey,
                                 "invalid scan key (zero or >= group order)");
            }
            secp256k1::detail::secure_erase(&scan_check, sizeof(scan_check));
        }

        struct ScanKeyGuard {
            cuda::Scalar k{};
            ~ScanKeyGuard() { secp256k1::detail::secure_erase(&k, sizeof(k)); }
        } scan_key_guard;
        cuda::Scalar& h_scan_k = scan_key_guard.k;
        for (int limb = 0; limb < 4; ++limb) {
            uint64_t v = 0;
            int base = (3 - limb) * 8;
            for (int b = 0; b < 8; ++b) v = (v << 8) | scan_privkey32[base + b];
            h_scan_k.limbs[limb] = v;
        }

        // Phase-boundary events (default stream, so recording order == actual
        // execution order): e0=start, e1=after mallocs, e2=after H2D copies,
        // e3=after decompress kernel (end of "setup"), e4=after main scan
        // kernel, e5=after D2H readback.
        struct EventGuard {
            cudaEvent_t e[6] = {};
            EventGuard() { for (auto& ev : e) cudaEventCreate(&ev); }
            ~EventGuard() { for (auto& ev : e) if (ev) cudaEventDestroy(ev); }
        } ev;
        cudaEventRecord(ev.e[0]);

        cuda::Scalar* d_scan_k = nullptr;
        CUDA_TRY(cudaMalloc(&d_scan_k, sizeof(cuda::Scalar)));
        CudaKeyGuard d_scan_k_guard(d_scan_k, sizeof(cuda::Scalar));
        uint8_t* d_tweaks33 = nullptr;
        CUDA_TRY(cudaMalloc(&d_tweaks33, n_tweaks * 33));
        CudaKeyGuard d_tweaks33_guard(d_tweaks33, n_tweaks * 33);
        uint8_t* d_spend33 = nullptr;
        CUDA_TRY(cudaMalloc(&d_spend33, n_spend * 33));
        CudaKeyGuard d_spend33_guard(d_spend33, n_spend * 33);
        cuda::AffinePoint* d_spend_aff = nullptr;
        CUDA_TRY(cudaMalloc(&d_spend_aff, n_spend * sizeof(cuda::AffinePoint)));
        CudaKeyGuard d_spend_aff_guard(d_spend_aff, n_spend * sizeof(cuda::AffinePoint));
        uint8_t* d_spend_valid = nullptr;
        CUDA_TRY(cudaMalloc(&d_spend_valid, n_spend * sizeof(uint8_t)));
        CudaKeyGuard d_spend_valid_guard(d_spend_valid, n_spend * sizeof(uint8_t));
        uint64_t* d_prefixes = nullptr;
        CUDA_TRY(cudaMalloc(&d_prefixes, n_rows * sizeof(uint64_t)));
        CudaKeyGuard d_prefixes_guard(d_prefixes, n_rows * sizeof(uint64_t));
        cudaEventRecord(ev.e[1]);

        CUDA_TRY(cudaMemcpy(d_scan_k, &h_scan_k, sizeof(cuda::Scalar), cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_tweaks33, tweak_pubkeys33, n_tweaks * 33, cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_spend33, spend_pubkeys33, n_spend * 33, cudaMemcpyHostToDevice));
        cudaEventRecord(ev.e[2]);

        {
            int threads = 128;
            int blocks  = (static_cast<int>(n_spend) + threads - 1) / threads;
            bip352_decompress_points_kernel<<<blocks, threads>>>(
                d_spend33, d_spend_aff, d_spend_valid, static_cast<int>(n_spend));
            CUDA_TRY(cudaGetLastError());
        }
        cudaEventRecord(ev.e[3]);

        {
            int threads = 128;
            int blocks  = (static_cast<int>(n_tweaks) + threads - 1) / threads;
            bip352_scan_batch_multispend_kernel_compressed<<<blocks, threads>>>(
                d_tweaks33, d_scan_k, d_spend_aff, d_spend_valid,
                static_cast<int>(n_spend), d_prefixes, static_cast<int>(n_tweaks));
            CUDA_TRY(cudaGetLastError());
        }
        CUDA_TRY(cudaDeviceSynchronize());
        cudaEventRecord(ev.e[4]);

        CUDA_TRY(cudaMemcpy(prefix64_out, d_prefixes,
                            n_rows * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        cudaEventRecord(ev.e[5]);
        cudaEventSynchronize(ev.e[5]);

        if (timing_out) {
            float ms_malloc = 0.f, ms_h2d = 0.f, ms_decompress = 0.f, ms_kernel = 0.f, ms_d2h = 0.f;
            cudaEventElapsedTime(&ms_malloc,     ev.e[0], ev.e[1]);
            cudaEventElapsedTime(&ms_h2d,        ev.e[1], ev.e[2]);
            cudaEventElapsedTime(&ms_decompress, ev.e[2], ev.e[3]);
            cudaEventElapsedTime(&ms_kernel,     ev.e[3], ev.e[4]);
            cudaEventElapsedTime(&ms_d2h,        ev.e[4], ev.e[5]);
            timing_out->setup_ms  = static_cast<double>(ms_malloc) + static_cast<double>(ms_decompress);
            timing_out->h2d_ms    = static_cast<double>(ms_h2d);
            timing_out->kernel_ms = static_cast<double>(ms_kernel);
            timing_out->d2h_ms    = static_cast<double>(ms_d2h);
        }

        cudaMemset(d_scan_k, 0, sizeof(cuda::Scalar));
        secp256k1::detail::secure_erase(&h_scan_k, sizeof(h_scan_k));

        out_guard.success = true;
        clear_error();
        return GpuError::Ok;
#else
        return set_error(GpuError::Unsupported, "GPU BIP-352 module disabled at build time");
#endif
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
