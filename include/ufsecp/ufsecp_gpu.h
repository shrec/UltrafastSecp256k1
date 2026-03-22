/* ============================================================================
 * UltrafastSecp256k1 -- GPU Acceleration C ABI
 * ============================================================================
 *
 * Backend-neutral C ABI for GPU-accelerated batch secp256k1 operations.
 *
 * ## Design principles
 *
 *   1. Opaque GPU context (`ufsecp_gpu_ctx*`) -- backend, device, queue state.
 *   2. Every function returns `ufsecp_error_t` (0 = OK).
 *   3. Backend-neutral: CUDA / OpenCL / Metal are implementation details.
 *   4. No internal GPU types leak -- all I/O is `uint8_t[]` with fixed strides.
 *   5. Thread safety: each gpu_ctx is single-thread. Create one per thread or
 *      protect externally.
 *   6. All first-wave operations are PUBLIC-DATA ONLY (verification, hashing,
 *      generator mul). ECDH is secret-bearing and documented as such.
 *
 * ## Feature maturity
 *
 *   This header defines the stable GPU API surface. Backend support
 *   per-operation varies:
 *
 *     CUDA   -- all 8 GPU ops implemented
 *     OpenCL -- 7/8 GPU ops implemented; `ecrecover_batch` currently returns
 *               UNSUPPORTED via a documented temporary stub
 *     Metal  -- 7/8 GPU ops implemented; `ecrecover_batch` currently returns
 *               UNSUPPORTED via a documented temporary stub
 *
 *   Operations that a backend does not implement return
 *   UFSECP_ERR_GPU_UNSUPPORTED (104). Callers MUST handle this gracefully.
 *   Backend coverage will expand over subsequent releases.
 *
 *   Guarantees:
 *     - Discovery + lifecycle functions work on all compiled backends.
 *     - Per-item results for batch ops are well-defined even on partial failure.
 *     - ECDH is the only secret-bearing GPU operation. All others are public-data.
 *     - ABI layout (function signatures, strides, error codes) is stable.
 *     - Backend additions do not break existing calling code.
 *
 * ## Memory
 *
 *   Caller owns all input/output buffers. Library manages device memory
 *   internally and copies results back on return.
 *
 * ## Batch layout
 *
 *   All batch inputs/outputs use flat contiguous arrays with fixed per-item
 *   strides documented in each function.
 *
 * ============================================================================ */
#ifndef UFSECP_GPU_H
#define UFSECP_GPU_H

#include "ufsecp_version.h"
#include "ufsecp_error.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * GPU-specific error codes (start at 100 to avoid conflict with CPU codes)
 * ============================================================================ */

#define UFSECP_ERR_GPU_UNAVAILABLE  100  /**< No GPU backend compiled in        */
#define UFSECP_ERR_GPU_DEVICE       101  /**< Device not found / init failed    */
#define UFSECP_ERR_GPU_LAUNCH       102  /**< Kernel launch / dispatch failed   */
#define UFSECP_ERR_GPU_MEMORY       103  /**< Device memory alloc/copy failed   */
#define UFSECP_ERR_GPU_UNSUPPORTED  104  /**< Op not supported on this backend  */
#define UFSECP_ERR_GPU_BACKEND      105  /**< Backend driver / runtime error    */
#define UFSECP_ERR_GPU_QUEUE        106  /**< Command queue / stream error      */

/* ============================================================================
 * GPU backend identifiers
 * ============================================================================ */

#define UFSECP_GPU_BACKEND_NONE     0
#define UFSECP_GPU_BACKEND_CUDA     1
#define UFSECP_GPU_BACKEND_OPENCL   2
#define UFSECP_GPU_BACKEND_METAL    3

/* ============================================================================
 * Opaque GPU context
 * ============================================================================ */

typedef struct ufsecp_gpu_ctx ufsecp_gpu_ctx;

/* ============================================================================
 * Backend & device discovery
 * ============================================================================ */

/** Return number of compiled-in GPU backends (0 if none).
 *  Fills backend_ids[] if non-NULL (caller allocates, size >= count). */
UFSECP_API uint32_t ufsecp_gpu_backend_count(uint32_t* backend_ids, uint32_t max_ids);

/** Return short name for a backend id ("CUDA", "OpenCL", "Metal", "none"). */
UFSECP_API const char* ufsecp_gpu_backend_name(uint32_t backend_id);

/** Return 1 if the backend is compiled in AND at least one device exists. */
UFSECP_API int ufsecp_gpu_is_available(uint32_t backend_id);

/** Return number of devices for the given backend (0 if unavailable). */
UFSECP_API uint32_t ufsecp_gpu_device_count(uint32_t backend_id);

/** Device info structure (filled by ufsecp_gpu_device_info). */
typedef struct {
    char     name[128];              /**< Device name (null-terminated)         */
    uint64_t global_mem_bytes;       /**< Total device memory in bytes         */
    uint32_t compute_units;          /**< Streaming multiprocessors / CUs      */
    uint32_t max_clock_mhz;         /**< Max clock speed in MHz               */
    uint32_t max_threads_per_block;  /**< Max threads per block/threadgroup    */
    uint32_t backend_id;             /**< UFSECP_GPU_BACKEND_*                 */
    uint32_t device_index;           /**< Index within backend                 */
} ufsecp_gpu_device_info_t;

/** Fill device info for (backend_id, device_index). */
UFSECP_API ufsecp_error_t ufsecp_gpu_device_info(
    uint32_t backend_id,
    uint32_t device_index,
    ufsecp_gpu_device_info_t* info_out);

/* ============================================================================
 * GPU context lifecycle
 * ============================================================================ */

/** Create a GPU context for the given backend and device.
 *  @param ctx_out   Receives the opaque context pointer.
 *  @param backend_id  UFSECP_GPU_BACKEND_CUDA / OPENCL / METAL.
 *  @param device_index  Device index within the backend (0 = default).
 *  @return UFSECP_OK on success. */
UFSECP_API ufsecp_error_t ufsecp_gpu_ctx_create(
    ufsecp_gpu_ctx** ctx_out,
    uint32_t backend_id,
    uint32_t device_index);

/** Destroy a GPU context and release all device resources. */
UFSECP_API void ufsecp_gpu_ctx_destroy(ufsecp_gpu_ctx* ctx);

/** Return the last error code from this GPU context. */
UFSECP_API ufsecp_error_t ufsecp_gpu_last_error(const ufsecp_gpu_ctx* ctx);

/** Return the last error message from this GPU context (never NULL). */
UFSECP_API const char* ufsecp_gpu_last_error_msg(const ufsecp_gpu_ctx* ctx);

/* ============================================================================
 * First-wave GPU batch operations
 * ============================================================================ */

/** Batch generator multiplication: compute k[i] * G for each scalar.
 *
 *  PUBLIC-DATA operation. Scalars are treated as public values.
 *
 *  @param ctx        GPU context.
 *  @param scalars32  Input: count * 32 bytes (big-endian scalars, contiguous).
 *  @param count      Number of scalars.
 *  @param out_pubkeys33  Output: count * 33 bytes (compressed pubkeys, contiguous).
 *  @return UFSECP_OK on success. */
UFSECP_API ufsecp_error_t ufsecp_gpu_generator_mul_batch(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* scalars32,
    size_t count,
    uint8_t* out_pubkeys33);

/** Batch ECDSA verification.
 *
 *  PUBLIC-DATA operation.
 *
 *  @param ctx           GPU context.
 *  @param msg_hashes32  Input: count * 32 bytes (message hashes, big-endian).
 *  @param pubkeys33     Input: count * 33 bytes (compressed pubkeys).
 *  @param sigs64        Input: count * 64 bytes (compact R||S signatures).
 *  @param count         Number of items.
 *  @param out_results   Output: count bytes (1 = valid, 0 = invalid per item).
 *  @return UFSECP_OK if batch processed (check out_results for per-item).
 *          GPU-specific error codes on device failure. */
UFSECP_API ufsecp_error_t ufsecp_gpu_ecdsa_verify_batch(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* msg_hashes32,
    const uint8_t* pubkeys33,
    const uint8_t* sigs64,
    size_t count,
    uint8_t* out_results);

/** Batch BIP-340 Schnorr verification.
 *
 *  PUBLIC-DATA operation.
 *
 *  @param ctx           GPU context.
 *  @param msg_hashes32  Input: count * 32 bytes (message hashes).
 *  @param pubkeys_x32   Input: count * 32 bytes (x-only public keys).
 *  @param sigs64        Input: count * 64 bytes (r||s Schnorr signatures).
 *  @param count         Number of items.
 *  @param out_results   Output: count bytes (1 = valid, 0 = invalid per item).
 *  @return UFSECP_OK if batch processed (check out_results for per-item). */
UFSECP_API ufsecp_error_t ufsecp_gpu_schnorr_verify_batch(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* msg_hashes32,
    const uint8_t* pubkeys_x32,
    const uint8_t* sigs64,
    size_t count,
    uint8_t* out_results);

/** Batch ECDH shared secret computation.
 *
 *  SECRET-BEARING operation. Private keys are uploaded to device memory.
 *  Use only when the threat model permits GPU-side secret handling.
 *
 *  @param ctx            GPU context.
 *  @param privkeys32     Input: count * 32 bytes (private keys, big-endian).
 *  @param peer_pubkeys33 Input: count * 33 bytes (compressed peer pubkeys).
 *  @param count          Number of items.
 *  @param out_secrets32  Output: count * 32 bytes (shared secrets = SHA-256(x)).
 *  @return UFSECP_OK on success. */
UFSECP_API ufsecp_error_t ufsecp_gpu_ecdh_batch(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* privkeys32,
    const uint8_t* peer_pubkeys33,
    size_t count,
    uint8_t* out_secrets32);

/** Batch Hash160 of compressed public keys: RIPEMD-160(SHA-256(pubkey33)).
 *
 *  PUBLIC-DATA operation.
 *
 *  @param ctx           GPU context.
 *  @param pubkeys33     Input: count * 33 bytes (compressed pubkeys).
 *  @param count         Number of items.
 *  @param out_hash160   Output: count * 20 bytes (hash160 digests).
 *  @return UFSECP_OK on success. */
UFSECP_API ufsecp_error_t ufsecp_gpu_hash160_pubkey_batch(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* pubkeys33,
    size_t count,
    uint8_t* out_hash160);

/** Multi-scalar multiplication: compute sum(scalars[i] * points[i]).
 *
 *  PUBLIC-DATA operation.
 *
 *  @param ctx           GPU context.
 *  @param scalars32     Input: n * 32 bytes (big-endian scalars).
 *  @param points33      Input: n * 33 bytes (compressed points).
 *  @param n             Number of (scalar, point) pairs.
 *  @param out_result33  Output: 33 bytes (compressed result point).
 *  @return UFSECP_OK on success.
 *          UFSECP_ERR_ARITH if result is point at infinity. */
UFSECP_API ufsecp_error_t ufsecp_gpu_msm(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* scalars32,
    const uint8_t* points33,
    size_t n,
    uint8_t* out_result33);

/* ============================================================================
 * GPU error string extension
 * ============================================================================ */

/** Batch FROST partial signature verification.
 *
 *  Each entry verifies: R_i = D_i + rho_i*E_i, lhs = z_i*G, rhs = R_i + lambda_ie*Y_i
 *  result[i] = (lhs == rhs).
 *  Returns UFSECP_ERR_UNSUPPORTED when backend does not implement FROST.
 *
 *  @param ctx           GPU context.
 *  @param z_i32         Input: count * 32 bytes (partial sig scalars, big-endian).
 *  @param D_i33         Input: count * 33 bytes (hiding nonce commitments, compressed).
 *  @param E_i33         Input: count * 33 bytes (binding nonce commitments, compressed).
 *  @param Y_i33         Input: count * 33 bytes (verification share pubkeys, compressed).
 *  @param rho_i32       Input: count * 32 bytes (per-signer binding factors, big-endian).
 *  @param lambda_ie32   Input: count * 32 bytes (lambda_i * e products, big-endian).
 *  @param negate_R      Input: count bytes (1 = negate R_i, 0 = keep).
 *  @param negate_key    Input: count bytes (1 = negate Y_i, 0 = keep).
 *  @param count         Number of partial signatures to verify.
 *  @param out_results   Output: count bytes (1 = valid, 0 = invalid per entry).
 *  @return UFSECP_OK if batch processed (check out_results for per-entry result).
 *          UFSECP_ERR_UNSUPPORTED if backend does not support FROST. */
UFSECP_API ufsecp_error_t ufsecp_gpu_frost_verify_partial_batch(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* z_i32,
    const uint8_t* D_i33,
    const uint8_t* E_i33,
    const uint8_t* Y_i33,
    const uint8_t* rho_i32,
    const uint8_t* lambda_ie32,
    const uint8_t* negate_R,
    const uint8_t* negate_key,
    size_t count,
    uint8_t* out_results);

/** Batch ECDSA public-key recovery on GPU.
 *  For each item recovers the compressed public key from (msg_hash, sig, recid).
 *  An entry that fails recovery writes 33 zero bytes into out_pubkeys33 and
 *  out_valid[i] = 0.
 *
 *  @param ctx           GPU context.
 *  @param msg_hashes32  Input: count * 32 bytes (32-byte message hashes).
 *  @param sigs64        Input: count * 64 bytes (compact R[32]||S[32], big-endian).
 *  @param recids        Input: count ints (recovery id 0-3 per entry).
 *  @param count         Number of entries.
 *  @param out_pubkeys33 Output: count * 33 bytes (compressed pubkeys; zeros on failure).
 *  @param out_valid     Output: count bytes (1 = recovered, 0 = failed).
 *  @return UFSECP_OK if batch processed; UFSECP_ERR_GPU_UNSUPPORTED if backend
 *          does not support this operation. */
UFSECP_API ufsecp_error_t ufsecp_gpu_ecrecover_batch(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* msg_hashes32,
    const uint8_t* sigs64,
    const int*     recids,
    size_t count,
    uint8_t* out_pubkeys33,
    uint8_t* out_valid);

/** Map GPU-specific error code to description (passes through to
 *  ufsecp_error_str for CPU error codes). */
UFSECP_API const char* ufsecp_gpu_error_str(ufsecp_error_t err);

#ifdef __cplusplus
}
#endif

#endif /* UFSECP_GPU_H */
