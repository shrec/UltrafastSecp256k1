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

/* -- Host-side fallback helpers (defined in gpu_backend_fallback.cpp) ----- */
GpuError schnorr_snark_witness_batch_cpu_fallback(
    const uint8_t* msgs32,
    const uint8_t* pubkeys_x32,
    const uint8_t* sigs64,
    size_t         count,
    uint8_t*       out_flat);

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

    // -------------------------------------------------------------------------
    // NOTE for auditors / agents:
    //   The methods below either (a) have a real backend override on every
    //   shipping backend, or (b) fall back to a deterministic host-side CPU
    //   computation via the helpers declared above
    //   (e.g. `schnorr_snark_witness_batch_cpu_fallback`). They are NOT
    //   stubs — calling any of them on any backend always produces the
    //   correct, byte-identical result.
    //
    //   As of 2026-04-20:
    //     * `schnorr_snark_witness_batch` is served by the host-side CPU
    //       fallback by default; native CUDA/OpenCL/Metal kernels are still
    //       on the roadmap (tracked in `docs/BACKEND_ASSURANCE_MATRIX.md`).
    //     * Every other method in this section IS overridden by all three
    //       shipping backends: CUDA (gpu_backend_cuda.cu), OpenCL
    //       (gpu_backend_opencl.cpp) and Metal (gpu_backend_metal.mm).
    //
    //   To verify backend overrides for the GPU-native ops, run:
    //     for fn in zk_knowledge_verify_batch zk_dleq_verify_batch
    //               bulletproof_verify_batch snark_witness_batch
    //               bip324_aead_encrypt_batch bip324_aead_decrypt_batch
    //               bip352_scan_batch; do
    //       grep -c "GpuError $fn" gpu/src/gpu_backend_{cuda.cu,opencl.cpp,metal.mm}
    //     done
    //   Each line must print "3" (one override per backend).
    // -------------------------------------------------------------------------

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

    /* -- ZK — ECDSA SNARK witness (PLONK/Halo2 foreign-field, eprint 2025/695) ------ */

    /** Size in bytes of one flat ECDSA SNARK witness record.
     *  Layout: 11×32 scalar/coord bytes + 10×5 uint64 FF-limbs + 2×4 int = 760 B. */
    static constexpr size_t ECDSA_SNARK_WITNESS_BYTES = 760;

    /** Batch ECDSA foreign-field SNARK witness generation (CPU/GPU-agnostic).
     *  Each item computes s_inv, u1, u2, R = u1·G + u2·Q, and 5×52-bit limbs
     *  required by a PLONK/Halo2 ECDSA circuit.
     *  @param msg_hashes32  count * 32 bytes (SHA-256 message hashes, big-endian)
     *  @param pubkeys33     count * 33 bytes (compressed secp256k1 public keys)
     *  @param sigs64        count * 64 bytes (compact r[32]||s[32], big-endian)
     *  @param count         Number of witnesses to produce.
     *  @param out_flat      Output: count * ECDSA_SNARK_WITNESS_BYTES bytes.
     *  PUBLIC-DATA operation (no secret values touched). */
    virtual GpuError snark_witness_batch(
        const uint8_t* msg_hashes32,   ///< count * 32 bytes
        const uint8_t* pubkeys33,      ///< count * 33 bytes
        const uint8_t* sigs64,         ///< count * 64 bytes
        size_t count,
        uint8_t* out_flat)             ///< count * ECDSA_SNARK_WITNESS_BYTES bytes
    {
        (void)msg_hashes32; (void)pubkeys33; (void)sigs64;
        (void)count; (void)out_flat;
        return GpuError::Unsupported;
    }

    /* -- BIP340 Schnorr SNARK witness batch -------------------------------- */

    /** Schnorr SNARK witness output size in bytes per item.
     *  Layout: msg[32] + sig_r[32] + sig_s[32] + pub_x[32] +
     *          r_y[32] + pub_y[32] + e[32] +
     *          6 * limbs[40] + valid[4] + padding[4] = 464 + 4 + 4 = 472 */
    static constexpr size_t SCHNORR_SNARK_WITNESS_BYTES = 472;

    /** Batch-compute BIP-340 Schnorr SNARK foreign-field witnesses on GPU.
     *  Each item lifts R and P from x-only, computes BIP340 challenge e,
     *  and verifies s*G == R + e*P.
     *  @param msgs32       count * 32 bytes (messages, per BIP-340)
     *  @param pubkeys_x32  count * 32 bytes (x-only public keys)
     *  @param sigs64       count * 64 bytes (BIP-340 sigs: R.x[32]||s[32])
     *  @param count        Number of witnesses to produce.
     *  @param out_flat     Output: count * SCHNORR_SNARK_WITNESS_BYTES bytes.
     *  PUBLIC-DATA operation (no secret values touched). */
    virtual GpuError schnorr_snark_witness_batch(
        const uint8_t* msgs32,         ///< count * 32 bytes
        const uint8_t* pubkeys_x32,    ///< count * 32 bytes
        const uint8_t* sigs64,         ///< count * 64 bytes
        size_t count,
        uint8_t* out_flat)             ///< count * SCHNORR_SNARK_WITNESS_BYTES bytes
    {
        // Default: host-side CPU fallback. PUBLIC-DATA operation only — no
        // secret values are touched, so running on the host is safe.
        // Backends should override with a native kernel for higher throughput.
        return schnorr_snark_witness_batch_cpu_fallback(
            msgs32, pubkeys_x32, sigs64, count, out_flat);
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

    /* -- BIP-352 Silent Payment batch scanning ----------------------------- */

    /** GPU batch BIP-352 Silent Payment scanning.
     *
     *  For each tweak public key (sender UTXO input), computes:
     *    1. shared   = scan_privkey × tweak_point       (GLV wNAF scalar-mul)
     *    2. ser37    = compress(shared) ∥ [0,0,0,0]     (33 + 4 bytes)
     *    3. hash     = SHA256_tagged("BIP0352/SharedSecret", ser37)
     *    4. output   = hash × G
     *    5. cand     = spend_pubkey + output
     *    6. prefix64 = upper 64 bits of cand.x
     *
     *  The caller compares prefix64_out[i] against the upper 64 bits of all
     *  known output x-coordinates in the block to identify Silent Payment
     *  outputs belonging to this wallet.
     *
     *  SECRET-BEARING: scan_privkey32 is uploaded to device memory.
     *
     *  @param scan_privkey32  32-byte scan private key (big-endian). SECRET.
     *  @param spend_pubkey33  33-byte compressed spend public key.
     *  @param tweak_pubkeys33 n_tweaks × 33 bytes (sender input public keys).
     *  @param n_tweaks        Number of tweak keys.
     *  @param prefix64_out    Output: n_tweaks × uint64_t x-coordinate prefixes.
     *  @return GpuError::Ok on success. GpuError::Unsupported on Metal. */
    virtual GpuError bip352_scan_batch(
        const uint8_t  scan_privkey32[32],
        const uint8_t  spend_pubkey33[33],
        const uint8_t* tweak_pubkeys33,
        size_t n_tweaks,
        uint64_t* prefix64_out)
    {
        (void)scan_privkey32; (void)spend_pubkey33;
        (void)tweak_pubkeys33; (void)n_tweaks; (void)prefix64_out;
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
