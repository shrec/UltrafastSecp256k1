/* ============================================================================
 * UltrafastSecp256k1 -- GPU Host Operations Layer (Internal)
 * ============================================================================
 * Abstract interface for GPU backends. Each backend (CUDA, OpenCL, Metal)
 * implements GpuBackend. The C ABI (ufsecp_gpu.h) dispatches through this.
 *
 * NOT part of the public API. Internal use only.
 * ============================================================================ */
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

namespace secp256k1 {
namespace gpu {

/* -- Error codes (mirrors ufsecp_error_t + GPU codes) ---------------------- */
enum class GpuError : int {
    Ok              = 0,
    NullArg         = 1,
    BadKey          = 2,
    BadPubkey       = 3,
    BadSig          = 4,
    BadInput        = 5,
    VerifyFail      = 6,
    Arith           = 7,
    Internal        = 9,
    BufTooSmall     = 10,
    Unavailable     = 100,
    Device          = 101,
    Launch          = 102,
    Memory          = 103,
    Unsupported     = 104,
    Backend         = 105,
    Queue           = 106,
};

/* -- Device info ----------------------------------------------------------- */
struct DeviceInfo {
    char     name[128]           = {};
    uint64_t global_mem_bytes    = 0;
    uint32_t compute_units       = 0;
    uint32_t max_clock_mhz       = 0;
    uint32_t max_threads_per_block = 0;
    uint32_t backend_id          = 0;
    uint32_t device_index        = 0;
};

/* -- Abstract backend interface -------------------------------------------- */
class GpuBackend {
public:
    virtual ~GpuBackend() = default;

    /* Backend identity */
    virtual uint32_t backend_id() const = 0;
    virtual const char* backend_name() const = 0;

    /* Device enumeration */
    virtual uint32_t device_count() const = 0;
    virtual GpuError device_info(uint32_t device_index, DeviceInfo& out) const = 0;

    /* Context init / teardown for selected device */
    virtual GpuError init(uint32_t device_index) = 0;
    virtual void shutdown() = 0;
    virtual bool is_ready() const = 0;

    /* Error tracking */
    virtual GpuError last_error() const = 0;
    virtual const char* last_error_msg() const = 0;

    /* First-wave batch ops */
    virtual GpuError generator_mul_batch(
        const uint8_t* scalars32, size_t count,
        uint8_t* out_pubkeys33) = 0;

    virtual GpuError ecdsa_verify_batch(
        const uint8_t* msg_hashes32, const uint8_t* pubkeys33,
        const uint8_t* sigs64, size_t count,
        uint8_t* out_results) = 0;

    virtual GpuError schnorr_verify_batch(
        const uint8_t* msg_hashes32, const uint8_t* pubkeys_x32,
        const uint8_t* sigs64, size_t count,
        uint8_t* out_results) = 0;

    virtual GpuError ecdh_batch(
        const uint8_t* privkeys32, const uint8_t* peer_pubkeys33,
        size_t count, uint8_t* out_secrets32) = 0;

    virtual GpuError hash160_pubkey_batch(
        const uint8_t* pubkeys33, size_t count,
        uint8_t* out_hash160) = 0;

    virtual GpuError msm(
        const uint8_t* scalars32, const uint8_t* points33,
        size_t n, uint8_t* out_result33) = 0;

    /** Batch FROST partial signature verification.
     *  Each item verifies: R_i = D_i + rho_i*E_i, lhs = z_i*G, rhs = R_i + lambda_ie*Y_i
     *  result[i] = (lhs == rhs). */
    virtual GpuError frost_verify_partial_batch(
        const uint8_t* z_i32,       ///< count * 32 bytes (partial sig scalars)
        const uint8_t* D_i33,       ///< count * 33 bytes (hiding nonce commitments)
        const uint8_t* E_i33,       ///< count * 33 bytes (binding nonce commitments)
        const uint8_t* Y_i33,       ///< count * 33 bytes (verification share pubkeys)
        const uint8_t* rho_i32,     ///< count * 32 bytes (binding factors)
        const uint8_t* lambda_ie32, ///< count * 32 bytes (lambda_i * e)
        const uint8_t* negate_R,    ///< count bytes (1 = negate R_i, 0 = keep)
        const uint8_t* negate_key,  ///< count bytes (1 = negate Y_i, 0 = keep)
        size_t count,
        uint8_t* out_results) = 0;  ///< count bytes (1 = valid, 0 = invalid)

    /** Batch ECDSA public-key recovery.
     *  For each item: recovers the compressed public key from (msg, sig, recid).
     *  out_pubkeys33[i] is zeroed for any entry that fails to recover.
     *  out_valid[i] = 1 if recovery succeeded, 0 if failed. */
    virtual GpuError ecrecover_batch(
        const uint8_t* msg_hashes32, ///< count * 32 bytes (message hashes)
        const uint8_t* sigs64,       ///< count * 64 bytes (compact R[32]||S[32], big-endian)
        const int*     recids,       ///< count recovery ids (0-3)
        size_t count,
        uint8_t* out_pubkeys33,      ///< count * 33 bytes (compressed pubkeys, zeros if failed)
        uint8_t* out_valid) = 0;     ///< count bytes (1 = recovered, 0 = failed)

    /* -- ZK proof batch operations ----------------------------------------- */

    /** Batch Schnorr knowledge-proof verification.
     *  Each proof is 64 bytes: rx[32] || s[32].
     *  Verifies s*G == R + e*P where e = H("ZK/knowledge" || rx || P || G || msg). */
    virtual GpuError zk_knowledge_verify_batch(
        const uint8_t* proofs64,     ///< count * 64 bytes (rx[32] || s[32])
        const uint8_t* pubkeys65,    ///< count * 65 bytes (uncompressed affine: 04 || x[32] || y[32])
        const uint8_t* messages32,   ///< count * 32 bytes
        size_t count,
        uint8_t* out_results)        ///< count bytes (1 = valid, 0 = invalid)
    {
        (void)proofs64; (void)pubkeys65; (void)messages32;
        (void)count; (void)out_results;
        return GpuError::Unsupported;
    }

    /** Batch DLEQ proof verification.
     *  Each proof is 64 bytes: e[32] || s[32].
     *  Verifies log_G(P) == log_H(Q) via Chaum–Pedersen. */
    virtual GpuError zk_dleq_verify_batch(
        const uint8_t* proofs64,     ///< count * 64 bytes (e[32] || s[32])
        const uint8_t* G_pts65,      ///< count * 65 bytes (uncompressed affine)
        const uint8_t* H_pts65,      ///< count * 65 bytes
        const uint8_t* P_pts65,      ///< count * 65 bytes
        const uint8_t* Q_pts65,      ///< count * 65 bytes
        size_t count,
        uint8_t* out_results)        ///< count bytes
    {
        (void)proofs64; (void)G_pts65; (void)H_pts65;
        (void)P_pts65; (void)Q_pts65; (void)count; (void)out_results;
        return GpuError::Unsupported;
    }

    /** Batch Bulletproof polynomial-check verification.
     *  Each proof contains A, S, T1, T2 (4 affine points) + tau_x, t_hat (2 scalars).
     *  Total per proof: 4*65 + 2*32 = 324 bytes.
     *  Each commitment is 65 bytes (uncompressed affine). */
    virtual GpuError bulletproof_verify_batch(
        const uint8_t* proofs324,        ///< count * 324 bytes
        const uint8_t* commitments65,    ///< count * 65 bytes
        const uint8_t* H_generator65,    ///< 65 bytes (Pedersen H generator)
        size_t count,
        uint8_t* out_results)            ///< count bytes
    {
        (void)proofs324; (void)commitments65; (void)H_generator65;
        (void)count; (void)out_results;
        return GpuError::Unsupported;
    }

    /* -- BIP-324 transport batch operations -------------------------------- */

    /** Batch BIP-324 AEAD encrypt.
     *  Each packet has its own key, nonce, and payload.
     *  Wire output per packet: 3-byte header + ciphertext + 16-byte tag.
     *  Stride = max_payload + 19 bytes. */
    virtual GpuError bip324_aead_encrypt_batch(
        const uint8_t*  keys32,      ///< count * 32 bytes
        const uint8_t*  nonces12,    ///< count * 12 bytes
        const uint8_t*  plaintexts,  ///< count * max_payload bytes
        const uint32_t* sizes,       ///< count payload sizes
        uint32_t max_payload,
        size_t count,
        uint8_t* wire_out)           ///< count * (max_payload + 19) bytes
    {
        (void)keys32; (void)nonces12; (void)plaintexts;
        (void)sizes; (void)max_payload; (void)count; (void)wire_out;
        return GpuError::Unsupported;
    }

    /** Batch BIP-324 AEAD decrypt.
     *  Verifies tag and decrypts.
     *  Wire input stride: max_payload + 19 bytes. */
    virtual GpuError bip324_aead_decrypt_batch(
        const uint8_t*  keys32,      ///< count * 32 bytes
        const uint8_t*  nonces12,    ///< count * 12 bytes
        const uint8_t*  wire_in,     ///< count * (max_payload + 19) bytes
        const uint32_t* sizes,       ///< count payload sizes
        uint32_t max_payload,
        size_t count,
        uint8_t*  plaintext_out,     ///< count * max_payload bytes
        uint8_t*  out_valid)         ///< count bytes (1 = ok, 0 = tag mismatch)
    {
        (void)keys32; (void)nonces12; (void)wire_in;
        (void)sizes; (void)max_payload; (void)count;
        (void)plaintext_out; (void)out_valid;
        return GpuError::Unsupported;
    }
};

/* -- Backend registry ------------------------------------------------------ */

/** Return number of compiled backends. */
uint32_t backend_count();

/** Get backend IDs. Returns count written. */
uint32_t backend_ids(uint32_t* ids, uint32_t max_ids);

/** Create a backend instance by ID. Returns nullptr if not compiled. */
std::unique_ptr<GpuBackend> create_backend(uint32_t backend_id);

/** Check if a backend is compiled and has at least one device. */
bool is_available(uint32_t backend_id);

} // namespace gpu
} // namespace secp256k1
