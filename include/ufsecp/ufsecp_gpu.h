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
 *   6. On the stable public GPU C ABI declared in this header, most operations
 *      are PUBLIC-DATA ONLY. ECDH and BIP-324 AEAD encrypt/decrypt are
 *      secret-bearing and documented as such.
 *
 * ## Feature maturity
 *
 *   This header defines the stable GPU API surface. The stable batch-op
 *   surface currently includes 13 backend-neutral operations: 8 core ops
 *   (generator_mul, ECDSA verify, Schnorr verify, ECDH, Hash160, MSM,
 *   FROST partial verify, ecrecover) plus 5 extended ZK/BIP-324 ops.
 *   Internal kernels, benchmarks, or backend test code may cover broader
 *   primitives; that does not by itself make those primitives part of the
 *   stable production GPU ABI.
 *
 *     CUDA   -- all 13 stable GPU batch ops implemented
 *     OpenCL -- all 13 stable GPU batch ops implemented
 *     Metal  -- all 13 stable GPU batch ops implemented
 *
 *   Operations that a backend does not implement return
 *   UFSECP_ERR_GPU_UNSUPPORTED (104). This is no longer expected for the
 *   stable 13-op GPU C ABI on compiled CUDA/OpenCL/Metal backends.
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
#  define UFSECP_NOEXCEPT noexcept
extern "C" {
#else
#  define UFSECP_NOEXCEPT
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
UFSECP_API uint32_t ufsecp_gpu_backend_count(uint32_t* backend_ids, uint32_t max_ids) UFSECP_NOEXCEPT;

/** Return short name for a backend id ("CUDA", "OpenCL", "Metal", "none"). */
UFSECP_API const char* ufsecp_gpu_backend_name(uint32_t backend_id);

/** Return 1 if the backend is compiled in AND at least one device exists. */
UFSECP_API int ufsecp_gpu_is_available(uint32_t backend_id) UFSECP_NOEXCEPT;

/** Return number of devices for the given backend (0 if unavailable). */
UFSECP_API uint32_t ufsecp_gpu_device_count(uint32_t backend_id) UFSECP_NOEXCEPT;

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

/** Return 1 if the GPU context is fully initialized and ready to accept batch
 *  operations, 0 otherwise. A context is ready after a successful
 *  ufsecp_gpu_ctx_create() and until ufsecp_gpu_ctx_destroy() is called.
 *  Returns 0 if ctx is NULL. */
UFSECP_API int ufsecp_gpu_is_ready(const ufsecp_gpu_ctx* ctx) UFSECP_NOEXCEPT;

/** Return the last error code from this GPU context. */
UFSECP_API ufsecp_error_t ufsecp_gpu_last_error(const ufsecp_gpu_ctx* ctx);

/** Return the last error message from this GPU context (never NULL).
 *  The returned pointer is borrowed storage owned by ctx/backend state.
 *  It remains valid until the next call that mutates the same ctx, or until
 *  ufsecp_gpu_ctx_destroy(ctx). Copy it if it must outlive the context/call. */
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

/* ============================================================================
 * ZK proof batch operations (GPU)
 * ============================================================================ */

/** Batch Schnorr knowledge-proof verification on GPU.
 *
 *  Verifies s*G == R + e*P where e = H("ZK/knowledge" || rx || P || G || msg).
 *  Each proof is 64 bytes: rx[32] || s[32] (big-endian).
 *  Each pubkey is 65 bytes: 04 || x[32] || y[32] (uncompressed affine).
 *
 *  PUBLIC-DATA operation.
 *
 *  @param ctx           GPU context.
 *  @param proofs64      Input: count * 64 bytes.
 *  @param pubkeys65     Input: count * 65 bytes (04 prefix).
 *  @param messages32    Input: count * 32 bytes.
 *  @param count         Number of proofs.
 *  @param out_results   Output: count bytes (1 = valid, 0 = invalid).
 *  @return UFSECP_OK if batch processed; UFSECP_ERR_GPU_UNSUPPORTED if
 *          backend does not support this operation. */
UFSECP_API ufsecp_error_t ufsecp_gpu_zk_knowledge_verify_batch(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* proofs64,
    const uint8_t* pubkeys65,
    const uint8_t* messages32,
    size_t count,
    uint8_t* out_results);

/** Batch DLEQ proof verification on GPU.
 *
 *  Verifies log_G(P) == log_H(Q) via Chaum–Pedersen protocol.
 *  Each proof is 64 bytes: e[32] || s[32] (big-endian).
 *  Each point is 65 bytes: 04 || x[32] || y[32] (uncompressed affine).
 *
 *  PUBLIC-DATA operation.
 *
 *  @param ctx           GPU context.
 *  @param proofs64      Input: count * 64 bytes.
 *  @param G_pts65       Input: count * 65 bytes (base point G per proof).
 *  @param H_pts65       Input: count * 65 bytes (base point H per proof).
 *  @param P_pts65       Input: count * 65 bytes (public key P per proof).
 *  @param Q_pts65       Input: count * 65 bytes (public key Q per proof).
 *  @param count         Number of proofs.
 *  @param out_results   Output: count bytes (1 = valid, 0 = invalid).
 *  @return UFSECP_OK if batch processed; UFSECP_ERR_GPU_UNSUPPORTED if
 *          backend does not support this operation. */
UFSECP_API ufsecp_error_t ufsecp_gpu_zk_dleq_verify_batch(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* proofs64,
    const uint8_t* G_pts65,
    const uint8_t* H_pts65,
    const uint8_t* P_pts65,
    const uint8_t* Q_pts65,
    size_t count,
    uint8_t* out_results);

/** Batch Bulletproof polynomial-check verification on GPU.
 *
 *  Verifies the polynomial commitment portion of a Bulletproof range proof.
 *  Each proof is 324 bytes: A[65] || S[65] || T1[65] || T2[65] || tau_x[32] || t_hat[32].
 *  Points use 65-byte uncompressed format (04 prefix).
 *
 *  PUBLIC-DATA operation.
 *
 *  @param ctx             GPU context.
 *  @param proofs324       Input: count * 324 bytes.
 *  @param commitments65   Input: count * 65 bytes (Pedersen commitments).
 *  @param H_generator65   Input: 65 bytes (Pedersen generator H).
 *  @param count           Number of proofs.
 *  @param out_results     Output: count bytes (1 = valid, 0 = invalid).
 *  @return UFSECP_OK if batch processed; UFSECP_ERR_GPU_UNSUPPORTED if
 *          backend does not support this operation. */
UFSECP_API ufsecp_error_t ufsecp_gpu_bulletproof_verify_batch(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* proofs324,
    const uint8_t* commitments65,
    const uint8_t* H_generator65,
    size_t count,
    uint8_t* out_results);

/* ============================================================================
 * BIP-324 transport batch operations (GPU)
 * ============================================================================ */

/** Batch BIP-324 AEAD encrypt on GPU.
 *
 *  Each thread processes one independent packet with its own key, nonce, and
 *  payload.  Wire format per packet: [3B length header] [ciphertext] [16B tag].
 *  Output stride per packet: max_payload + 19 bytes.
 *
 *  SECRET-BEARING operation. Keys are uploaded to device memory.
 *
 *  @param ctx           GPU context.
 *  @param keys32        Input: count * 32 bytes (ChaCha20-Poly1305 keys).
 *  @param nonces12      Input: count * 12 bytes.
 *  @param plaintexts    Input: count * max_payload bytes (contiguous).
 *  @param sizes         Input: count uint32_t (actual payload size per packet).
 *  @param max_payload   Maximum payload size (all payloads padded to this).
 *  @param count         Number of packets.
 *  @param wire_out      Output: count * (max_payload + 19) bytes.
 *  @return UFSECP_OK on success. */
UFSECP_API ufsecp_error_t ufsecp_gpu_bip324_aead_encrypt_batch(
    ufsecp_gpu_ctx* ctx,
    const uint8_t*  keys32,
    const uint8_t*  nonces12,
    const uint8_t*  plaintexts,
    const uint32_t* sizes,
    uint32_t max_payload,
    size_t count,
    uint8_t* wire_out);

/** Batch BIP-324 AEAD decrypt on GPU.
 *
 *  Verifies Poly1305 tag and decrypts.  Wire input stride: max_payload + 19.
 *
 *  SECRET-BEARING operation. Keys are uploaded to device memory.
 *
 *  @param ctx           GPU context.
 *  @param keys32        Input: count * 32 bytes.
 *  @param nonces12      Input: count * 12 bytes.
 *  @param wire_in       Input: count * (max_payload + 19) bytes.
 *  @param sizes         Input: count uint32_t (payload sizes).
 *  @param max_payload   Maximum payload size.
 *  @param count         Number of packets.
 *  @param plaintext_out Output: count * max_payload bytes.
 *  @param out_valid     Output: count bytes (1 = ok, 0 = tag mismatch).
 *  @return UFSECP_OK on success. */
UFSECP_API ufsecp_error_t ufsecp_gpu_bip324_aead_decrypt_batch(
    ufsecp_gpu_ctx* ctx,
    const uint8_t*  keys32,
    const uint8_t*  nonces12,
    const uint8_t*  wire_in,
    const uint32_t* sizes,
    uint32_t max_payload,
    size_t count,
    uint8_t*  plaintext_out,
    uint8_t*  out_valid);

/* -- ECDSA SNARK witness batch (eprint 2025/695) ------------------------------ */

/** Size in bytes of one flat witness record produced by
 *  ufsecp_gpu_zk_ecdsa_snark_witness_batch(). */
#define UFSECP_ECDSA_SNARK_WITNESS_BYTES 760

/**
 * Compute ECDSA SNARK witnesses for a batch of (message, pubkey, sig) tuples.
 *
 * @param ctx          GPU context returned by ufsecp_gpu_ctx_create().
 * @param msg_hashes32 Input: count × 32 bytes (one BE SHA-256 hash per item).
 * @param pubkeys33    Input: count × 33 bytes (compressed SEC1 public keys).
 * @param sigs64       Input: count × 64 bytes (BE compact r|s signatures).
 * @param count        Number of items to process.
 * @param out_witnesses Output: count × UFSECP_ECDSA_SNARK_WITNESS_BYTES bytes.
 *                      Each 760-byte record contains input values, intermediate
 *                      witness scalars, and 5×52-bit foreign-field limbs.
 * @return UFSECP_OK on success, or an error code on failure.
 *         Returns UFSECP_ERR_UNSUPPORTED on Metal (no MSL shader yet).
 */
UFSECP_API ufsecp_error_t ufsecp_gpu_zk_ecdsa_snark_witness_batch(
    ufsecp_gpu_ctx* ctx,
    const uint8_t*  msg_hashes32,
    const uint8_t*  pubkeys33,
    const uint8_t*  sigs64,
    size_t          count,
    uint8_t*        out_witnesses);

/* -- BIP340 Schnorr SNARK witness batch --------------------------------------- */

/** Size in bytes of one flat Schnorr witness record produced by
 *  ufsecp_gpu_zk_schnorr_snark_witness_batch(). */
#define UFSECP_SCHNORR_SNARK_WITNESS_BYTES 472

/**
 * Compute BIP-340 Schnorr SNARK witnesses for a batch of (message, xonly-pubkey, sig) tuples.
 *
 * @param ctx          GPU context returned by ufsecp_gpu_ctx_create().
 * @param msgs32       Input: count × 32 bytes (messages, per BIP-340).
 * @param pubkeys_x32  Input: count × 32 bytes (x-only public keys).
 * @param sigs64       Input: count × 64 bytes (BIP-340 sigs: R.x[32] || s[32]).
 * @param count        Number of items to process.
 * @param out_witnesses Output: count × UFSECP_SCHNORR_SNARK_WITNESS_BYTES bytes.
 *                      Each record contains public inputs, lifted point witnesses,
 *                      challenge scalar, and 5×52-bit foreign-field limbs.
 * @return UFSECP_OK on success, or an error code on failure.
 */
UFSECP_API ufsecp_error_t ufsecp_gpu_zk_schnorr_snark_witness_batch(
    ufsecp_gpu_ctx* ctx,
    const uint8_t*  msgs32,
    const uint8_t*  pubkeys_x32,
    const uint8_t*  sigs64,
    size_t          count,
    uint8_t*        out_witnesses);

/* ============================================================================
 * BIP-352 Silent Payment scanning (GPU + CPU plan utility)
 * ============================================================================ */

/** Size in bytes of the precomputed BIP-352 GPU scan key plan.
 *  Matches sizeof(BIP352ScanKeyGlv) used by the OpenCL kernel. */
#define UFSECP_BIP352_SCAN_PLAN_BYTES 264

/** Precompute a BIP-352 scan key wNAF plan for repeated GPU batch calls.
 *
 *  This is a CPU-only convenience function.  Call it once per scan key and
 *  pass the resulting 264-byte plan to @ref ufsecp_gpu_bip352_scan_batch for
 *  subsequent scanning.  The plan encodes the GLV-decomposed wNAF digits so
 *  the GPU does not need to recompute them for each batch.
 *
 *  No GPU context is required; call this before creating the GPU context if
 *  needed.
 *
 *  @param scan_privkey32  32-byte scan private key (big-endian). SECRET.
 *  @param plan264_out     Output buffer of exactly 264 bytes.
 *  @return UFSECP_OK on success, UFSECP_ERR_BAD_KEY if the key is zero or
 *          otherwise invalid.
 */
UFSECP_API ufsecp_error_t ufsecp_bip352_prepare_scan_plan(
    const uint8_t scan_privkey32[32],
    uint8_t       plan264_out[264]);

/** GPU batch BIP-352 Silent Payment scanning.
 *
 *  For each sender tweak public key in the block, computes the full BIP-352
 *  scanning pipeline and outputs the upper 64 bits of the secp256k1 x-coordinate
 *  of the candidate output point.
 *
 *  Pipeline per tweak key:
 *    1. shared   = scan_privkey × tweak_pubkey        (GLV wNAF scalar mul)
 *    2. ser37    = compress(shared) ∥ [0x00,0x00,0x00,0x00]
 *    3. hash     = SHA256_tagged("BIP0352/SharedSecret", ser37)
 *    4. output   = hash × G
 *    5. cand     = output + spend_pubkey
 *    6. prefix64 = upper 64 bits of cand.x
 *
 *  The caller compares prefix64_out[i] against the upper 64 bits of every
 *  known output's x-coordinate in the block to detect matching Silent Payment
 *  outputs.
 *
 *  SECRET-BEARING: scan_privkey32 is uploaded to device memory during this
 *  call.  Callers should zeroize it immediately after the call if required.
 *
 *  @param ctx              GPU context (OpenCL or CUDA; Metal returns
 *                          UFSECP_ERR_GPU_UNSUPPORTED).
 *  @param scan_privkey32   32-byte scan private key (big-endian). SECRET.
 *  @param spend_pubkey33   33-byte compressed secp256k1 spend public key.
 *  @param tweak_pubkeys33  Input: n_tweaks × 33 bytes, compressed pubkeys.
 *  @param n_tweaks         Number of tweak keys to process.
 *  @param prefix64_out     Output: n_tweaks × uint64_t, one per tweak key.
 *  @return UFSECP_OK on success.
 *          UFSECP_ERR_BAD_KEY   if any pubkey is invalid.
 *          UFSECP_ERR_GPU_UNSUPPORTED on Metal backend.
 */
UFSECP_API ufsecp_error_t ufsecp_gpu_bip352_scan_batch(
    ufsecp_gpu_ctx* ctx,
    const uint8_t   scan_privkey32[32],
    const uint8_t   spend_pubkey33[33],
    const uint8_t*  tweak_pubkeys33,
    size_t          n_tweaks,
    uint64_t*       prefix64_out);

#ifdef __cplusplus
}
#endif

#endif /* UFSECP_GPU_H */
