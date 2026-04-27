/* ============================================================================
 * UltrafastSecp256k1 -- GPU C ABI Implementation
 * ============================================================================
 * Implements the ufsecp_gpu_* functions declared in ufsecp_gpu.h.
 * Delegates all work to gpu::GpuBackend instances from gpu_backend.hpp.
 *
 * Build with: -DUFSECP_BUILDING (sets dllexport on Windows)
 * ============================================================================ */

#ifndef UFSECP_BUILDING
#define UFSECP_BUILDING
#endif

#include "ufsecp_gpu.h"
#include "../../gpu/include/gpu_backend.hpp"

#include <cstring>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <new>
#include <limits>

/* CPU headers needed for ufsecp_bip352_prepare_scan_plan */
#include "secp256k1/scalar.hpp"
#include "secp256k1/glv.hpp"
#include "secp256k1/detail/secure_erase.hpp"

using namespace secp256k1::gpu;

/* Hard upper bound on user-supplied GPU batch counts.
 * Prevents hostile callers from triggering multi-GB allocations.              */
static constexpr std::size_t kMaxGpuBatchN = std::size_t{1} << 26;  /* 64 M */

/* Macro: catch C++ exceptions at the extern "C" boundary.                    */
#define UFSECP_GPU_CATCH \
    catch (const std::bad_alloc&) { return UFSECP_ERR_GPU_MEMORY; } \
    catch (...) { return UFSECP_ERR_INTERNAL; }

/* ===========================================================================
 * Opaque GPU context definition
 * =========================================================================== */

struct ufsecp_gpu_ctx {
    std::unique_ptr<GpuBackend> backend;
    uint32_t backend_id;
    uint32_t device_index;
};

/* ===========================================================================
 * Internal helpers
 * =========================================================================== */

static inline ufsecp_error_t to_abi_error(GpuError e) {
    return static_cast<ufsecp_error_t>(static_cast<int>(e));
}

static bool has_valid_compressed_pubkeys(const uint8_t* pubkeys33, size_t count) {
    for (size_t index = 0; index < count; ++index) {
        const uint8_t prefix = pubkeys33[index * 33];
        if (prefix != 0x02 && prefix != 0x03) {
            return false;
        }
    }
    return true;
}

static bool has_valid_uncompressed_pubkeys(const uint8_t* pubkeys65, size_t count) {
    for (size_t index = 0; index < count; ++index) {
        if (pubkeys65[index * 65] != 0x04) {
            return false;
        }
    }
    return true;
}

static bool has_valid_recovery_ids(const int* recids, size_t count) {
    for (size_t index = 0; index < count; ++index) {
        if (recids[index] < 0 || recids[index] > 3) {
            return false;
        }
    }
    return true;
}

static bool has_valid_bip324_sizes(const uint32_t* sizes, size_t count, uint32_t max_payload) {
    for (size_t index = 0; index < count; ++index) {
        if (sizes[index] > max_payload) {
            return false;
        }
    }
    return true;
}

static bool has_valid_bulletproof_prefixes(const uint8_t* proofs324, size_t count) {
    for (size_t index = 0; index < count; ++index) {
        const uint8_t* proof = proofs324 + (index * 324);
        if (proof[0] != 0x04 || proof[65] != 0x04 || proof[130] != 0x04 || proof[195] != 0x04) {
            return false;
        }
    }
    return true;
}

/* ===========================================================================
 * Backend & device discovery
 * =========================================================================== */

uint32_t ufsecp_gpu_backend_count(uint32_t* backend_ids_out, uint32_t max_ids) {
    try {
    const uint32_t count = backend_count();
    if (backend_ids_out && max_ids > 0) {
        backend_ids(backend_ids_out, max_ids);
    }
    return count;
    } catch (...) { return 0; }
}

const char* ufsecp_gpu_backend_name(uint32_t bid) {
    switch (bid) {
    case 1: return "CUDA";
    case 2: return "OpenCL";
    case 3: return "Metal";
    default: return "none";
    }
}

int ufsecp_gpu_is_available(uint32_t bid) {
    try { return is_available(bid) ? 1 : 0; }
    catch (...) { return 0; }
}

uint32_t ufsecp_gpu_device_count(uint32_t bid) {
    try {
    auto b = create_backend(bid);
    if (!b) return 0;
    return b->device_count();
    } catch (...) { return 0; }
}

ufsecp_error_t ufsecp_gpu_device_info(
    uint32_t bid, uint32_t device_index,
    ufsecp_gpu_device_info_t* info_out)
{
    if (!info_out) return UFSECP_ERR_NULL_ARG;
    try {
    auto b = create_backend(bid);
    if (!b) return UFSECP_ERR_GPU_UNAVAILABLE;

    DeviceInfo di;
    auto err = b->device_info(device_index, di);
    if (err != GpuError::Ok) return to_abi_error(err);

    std::memcpy(info_out->name, di.name, sizeof(info_out->name));
    info_out->global_mem_bytes      = di.global_mem_bytes;
    info_out->compute_units         = di.compute_units;
    info_out->max_clock_mhz         = di.max_clock_mhz;
    info_out->max_threads_per_block = di.max_threads_per_block;
    info_out->backend_id            = di.backend_id;
    info_out->device_index          = di.device_index;
    return UFSECP_OK;
    } UFSECP_GPU_CATCH
}

/* ===========================================================================
 * GPU context lifecycle
 * =========================================================================== */

ufsecp_error_t ufsecp_gpu_ctx_create(
    ufsecp_gpu_ctx** ctx_out,
    uint32_t bid,
    uint32_t device_index)
{
    if (!ctx_out) return UFSECP_ERR_NULL_ARG;
    *ctx_out = nullptr;
    try {
    auto backend = create_backend(bid);
    if (!backend) return UFSECP_ERR_GPU_UNAVAILABLE;

    auto err = backend->init(device_index);
    if (err != GpuError::Ok) return to_abi_error(err);

    auto* ctx = new (std::nothrow) ufsecp_gpu_ctx;
    if (!ctx) return UFSECP_ERR_INTERNAL;

    ctx->backend      = std::move(backend);
    ctx->backend_id   = bid;
    ctx->device_index = device_index;
    *ctx_out = ctx;
    return UFSECP_OK;
    } UFSECP_GPU_CATCH
}

void ufsecp_gpu_ctx_destroy(ufsecp_gpu_ctx* ctx) {
    if (ctx) {
        ctx->backend.reset();
        delete ctx;
    }
}

int ufsecp_gpu_is_ready(const ufsecp_gpu_ctx* ctx) {
    if (!ctx || !ctx->backend) return 0;
    try { return ctx->backend->is_ready() ? 1 : 0; }
    catch (...) { return 0; }
}

ufsecp_error_t ufsecp_gpu_last_error(const ufsecp_gpu_ctx* ctx) {
    if (!ctx) return UFSECP_ERR_NULL_ARG;
    return to_abi_error(ctx->backend->last_error());
}

const char* ufsecp_gpu_last_error_msg(const ufsecp_gpu_ctx* ctx) {
    if (!ctx) return "NULL GPU context";
    return ctx->backend->last_error_msg();
}

/* ===========================================================================
 * First-wave GPU batch operations
 * =========================================================================== */

ufsecp_error_t ufsecp_gpu_generator_mul_batch(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* scalars32,
    size_t count,
    uint8_t* out_pubkeys33)
{
    if (!ctx) return UFSECP_ERR_NULL_ARG;
    if (count == 0) return UFSECP_OK;
    if (!scalars32 || !out_pubkeys33) return UFSECP_ERR_NULL_ARG;
    if (count > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    try {
    return to_abi_error(
        ctx->backend->generator_mul_batch(scalars32, count, out_pubkeys33));
    } UFSECP_GPU_CATCH
}

ufsecp_error_t ufsecp_gpu_ecdsa_verify_batch(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* msg_hashes32,
    const uint8_t* pubkeys33,
    const uint8_t* sigs64,
    size_t count,
    uint8_t* out_results)
{
    if (!ctx) return UFSECP_ERR_NULL_ARG;
    if (count == 0) return UFSECP_OK;
    if (!msg_hashes32 || !pubkeys33 || !sigs64 || !out_results) {
        return UFSECP_ERR_NULL_ARG;
    }
    if (count > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    try {
    return to_abi_error(
        ctx->backend->ecdsa_verify_batch(
            msg_hashes32, pubkeys33, sigs64, count, out_results));
    } UFSECP_GPU_CATCH
}

ufsecp_error_t ufsecp_gpu_schnorr_verify_batch(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* msg_hashes32,
    const uint8_t* pubkeys_x32,
    const uint8_t* sigs64,
    size_t count,
    uint8_t* out_results)
{
    if (!ctx) return UFSECP_ERR_NULL_ARG;
    if (count == 0) return UFSECP_OK;
    if (!msg_hashes32 || !pubkeys_x32 || !sigs64 || !out_results) {
        return UFSECP_ERR_NULL_ARG;
    }
    if (count > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    try {
    return to_abi_error(
        ctx->backend->schnorr_verify_batch(
            msg_hashes32, pubkeys_x32, sigs64, count, out_results));
    } UFSECP_GPU_CATCH
}

ufsecp_error_t ufsecp_gpu_ecdh_batch(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* privkeys32,
    const uint8_t* peer_pubkeys33,
    size_t count,
    uint8_t* out_secrets32)
{
    if (!ctx) return UFSECP_ERR_NULL_ARG;
    if (count == 0) return UFSECP_OK;
    if (!privkeys32 || !peer_pubkeys33 || !out_secrets32) {
        return UFSECP_ERR_NULL_ARG;
    }
    if (!has_valid_compressed_pubkeys(peer_pubkeys33, count)) {
        return UFSECP_ERR_BAD_PUBKEY;
    }
    if (count > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    try {
    return to_abi_error(
        ctx->backend->ecdh_batch(
            privkeys32, peer_pubkeys33, count, out_secrets32));
    } UFSECP_GPU_CATCH
}

ufsecp_error_t ufsecp_gpu_hash160_pubkey_batch(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* pubkeys33,
    size_t count,
    uint8_t* out_hash160)
{
    if (!ctx) return UFSECP_ERR_NULL_ARG;
    if (count == 0) return UFSECP_OK;
    if (!pubkeys33 || !out_hash160) return UFSECP_ERR_NULL_ARG;
    if (!has_valid_compressed_pubkeys(pubkeys33, count)) {
        return UFSECP_ERR_BAD_PUBKEY;
    }
    if (count > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    try {
    return to_abi_error(
        ctx->backend->hash160_pubkey_batch(pubkeys33, count, out_hash160));
    } UFSECP_GPU_CATCH
}

ufsecp_error_t ufsecp_gpu_msm(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* scalars32,
    const uint8_t* points33,
    size_t n,
    uint8_t* out_result33)
{
    if (!ctx) return UFSECP_ERR_NULL_ARG;
    if (n == 0) return UFSECP_OK;
    if (!scalars32 || !points33 || !out_result33) return UFSECP_ERR_NULL_ARG;
    if (n > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    try {
    return to_abi_error(
        ctx->backend->msm(scalars32, points33, n, out_result33));
    } UFSECP_GPU_CATCH
}

ufsecp_error_t ufsecp_gpu_frost_verify_partial_batch(
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
    uint8_t* out_results)
{
    if (!ctx) return UFSECP_ERR_NULL_ARG;
    if (count == 0) return UFSECP_OK;
    if (!z_i32 || !D_i33 || !E_i33 || !Y_i33 || !rho_i32 ||
        !lambda_ie32 || !negate_R || !negate_key || !out_results) {
        return UFSECP_ERR_NULL_ARG;
    }
    if (!has_valid_compressed_pubkeys(D_i33, count) ||
        !has_valid_compressed_pubkeys(E_i33, count) ||
        !has_valid_compressed_pubkeys(Y_i33, count)) {
        return UFSECP_ERR_BAD_PUBKEY;
    }
    if (count > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    try {
    return to_abi_error(
        ctx->backend->frost_verify_partial_batch(
            z_i32, D_i33, E_i33, Y_i33, rho_i32, lambda_ie32,
            negate_R, negate_key, count, out_results));
    } UFSECP_GPU_CATCH
}

ufsecp_error_t ufsecp_gpu_ecrecover_batch(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* msg_hashes32,
    const uint8_t* sigs64,
    const int*     recids,
    size_t count,
    uint8_t* out_pubkeys33,
    uint8_t* out_valid)
{
    if (!ctx) return UFSECP_ERR_NULL_ARG;
    if (count == 0) return UFSECP_OK;
    if (!msg_hashes32 || !sigs64 || !recids || !out_pubkeys33 || !out_valid) {
        return UFSECP_ERR_NULL_ARG;
    }
    if (!has_valid_recovery_ids(recids, count)) {
        return UFSECP_ERR_BAD_INPUT;
    }
    if (count > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    try {
    return to_abi_error(
        ctx->backend->ecrecover_batch(
            msg_hashes32, sigs64, recids, count, out_pubkeys33, out_valid));
    } UFSECP_GPU_CATCH
}

/* ===========================================================================
 * ZK proof batch operations
 * =========================================================================== */

ufsecp_error_t ufsecp_gpu_zk_knowledge_verify_batch(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* proofs64,
    const uint8_t* pubkeys65,
    const uint8_t* messages32,
    size_t count,
    uint8_t* out_results)
{
    if (!ctx) return UFSECP_ERR_NULL_ARG;
    if (count == 0) return UFSECP_OK;
    if (!proofs64 || !pubkeys65 || !messages32 || !out_results)
        return UFSECP_ERR_NULL_ARG;
    if (!has_valid_uncompressed_pubkeys(pubkeys65, count)) {
        return UFSECP_ERR_BAD_PUBKEY;
    }
    if (count > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    try {
    return to_abi_error(
        ctx->backend->zk_knowledge_verify_batch(
            proofs64, pubkeys65, messages32, count, out_results));
    } UFSECP_GPU_CATCH
}

ufsecp_error_t ufsecp_gpu_zk_dleq_verify_batch(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* proofs64,
    const uint8_t* G_pts65,
    const uint8_t* H_pts65,
    const uint8_t* P_pts65,
    const uint8_t* Q_pts65,
    size_t count,
    uint8_t* out_results)
{
    if (!ctx) return UFSECP_ERR_NULL_ARG;
    if (count == 0) return UFSECP_OK;
    if (!proofs64 || !G_pts65 || !H_pts65 || !P_pts65 || !Q_pts65 || !out_results)
        return UFSECP_ERR_NULL_ARG;
    if (!has_valid_uncompressed_pubkeys(G_pts65, count) ||
        !has_valid_uncompressed_pubkeys(H_pts65, count) ||
        !has_valid_uncompressed_pubkeys(P_pts65, count) ||
        !has_valid_uncompressed_pubkeys(Q_pts65, count)) {
        return UFSECP_ERR_BAD_PUBKEY;
    }
    if (count > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    try {
    return to_abi_error(
        ctx->backend->zk_dleq_verify_batch(
            proofs64, G_pts65, H_pts65, P_pts65, Q_pts65, count, out_results));
    } UFSECP_GPU_CATCH
}

ufsecp_error_t ufsecp_gpu_bulletproof_verify_batch(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* proofs324,
    const uint8_t* commitments65,
    const uint8_t* H_generator65,
    size_t count,
    uint8_t* out_results)
{
    if (!ctx) return UFSECP_ERR_NULL_ARG;
    if (count == 0) return UFSECP_OK;
    if (!proofs324 || !commitments65 || !H_generator65 || !out_results)
        return UFSECP_ERR_NULL_ARG;
    if (!has_valid_bulletproof_prefixes(proofs324, count) ||
        !has_valid_uncompressed_pubkeys(commitments65, count) ||
        !has_valid_uncompressed_pubkeys(H_generator65, 1)) {
        return UFSECP_ERR_BAD_PUBKEY;
    }
    if (count > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    try {
    return to_abi_error(
        ctx->backend->bulletproof_verify_batch(
            proofs324, commitments65, H_generator65, count, out_results));
    } UFSECP_GPU_CATCH
}

/* ===========================================================================
 * BIP-324 transport batch operations
 * =========================================================================== */

ufsecp_error_t ufsecp_gpu_bip324_aead_encrypt_batch(
    ufsecp_gpu_ctx* ctx,
    const uint8_t*  keys32,
    const uint8_t*  nonces12,
    const uint8_t*  plaintexts,
    const uint32_t* sizes,
    uint32_t max_payload,
    size_t count,
    uint8_t* wire_out)
{
    if (!ctx) return UFSECP_ERR_NULL_ARG;
    if (count == 0) return UFSECP_OK;
    if (!keys32 || !nonces12 || !plaintexts || !sizes || !wire_out)
        return UFSECP_ERR_NULL_ARG;
    if (!has_valid_bip324_sizes(sizes, count, max_payload)) {
        return UFSECP_ERR_BAD_INPUT;
    }
    if (count > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    try {
    return to_abi_error(
        ctx->backend->bip324_aead_encrypt_batch(
            keys32, nonces12, plaintexts, sizes, max_payload, count, wire_out));
    } UFSECP_GPU_CATCH
}

ufsecp_error_t ufsecp_gpu_bip324_aead_decrypt_batch(
    ufsecp_gpu_ctx* ctx,
    const uint8_t*  keys32,
    const uint8_t*  nonces12,
    const uint8_t*  wire_in,
    const uint32_t* sizes,
    uint32_t max_payload,
    size_t count,
    uint8_t*  plaintext_out,
    uint8_t*  out_valid)
{
    if (!ctx) return UFSECP_ERR_NULL_ARG;
    if (count == 0) return UFSECP_OK;
    if (!keys32 || !nonces12 || !wire_in || !sizes || !plaintext_out || !out_valid)
        return UFSECP_ERR_NULL_ARG;
    if (!has_valid_bip324_sizes(sizes, count, max_payload)) {
        return UFSECP_ERR_BAD_INPUT;
    }
    if (count > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    try {
    return to_abi_error(
        ctx->backend->bip324_aead_decrypt_batch(
            keys32, nonces12, wire_in, sizes, max_payload, count,
            plaintext_out, out_valid));
    } UFSECP_GPU_CATCH
}

/* ===========================================================================
 * ECDSA SNARK witness batch (eprint 2025/695)
 * =========================================================================== */

ufsecp_error_t ufsecp_gpu_zk_ecdsa_snark_witness_batch(
    ufsecp_gpu_ctx* ctx,
    const uint8_t*  msg_hashes32,
    const uint8_t*  pubkeys33,
    const uint8_t*  sigs64,
    size_t          count,
    uint8_t*        out_witnesses)
{
    if (count == 0) return UFSECP_OK;
    if (!ctx || !msg_hashes32 || !pubkeys33 || !sigs64 || !out_witnesses)
        return UFSECP_ERR_NULL_ARG;
    if (count > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    try {
        return to_abi_error(ctx->backend->snark_witness_batch(
            msg_hashes32, pubkeys33, sigs64, count, out_witnesses));
    } UFSECP_GPU_CATCH
}

ufsecp_error_t ufsecp_gpu_zk_schnorr_snark_witness_batch(
    ufsecp_gpu_ctx* ctx,
    const uint8_t*  msgs32,
    const uint8_t*  pubkeys_x32,
    const uint8_t*  sigs64,
    size_t          count,
    uint8_t*        out_witnesses)
{
    if (count == 0) return UFSECP_OK;
    if (!ctx || !msgs32 || !pubkeys_x32 || !sigs64 || !out_witnesses)
        return UFSECP_ERR_NULL_ARG;
    if (count > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    try {
        return to_abi_error(ctx->backend->schnorr_snark_witness_batch(
            msgs32, pubkeys_x32, sigs64, count, out_witnesses));
    } UFSECP_GPU_CATCH
}

/* ===========================================================================
 * BIP-352 scan plan precomputation (CPU utility, no GPU ctx needed)
 * =========================================================================== */

ufsecp_error_t ufsecp_bip352_prepare_scan_plan(
    const uint8_t scan_privkey32[32],
    uint8_t       plan264_out[264])
{
    if (!scan_privkey32 || !plan264_out) return UFSECP_ERR_NULL_ARG;

    using namespace secp256k1::fast;

    Scalar k = Scalar::from_bytes(scan_privkey32);
    if (k.is_zero()) {
        secp256k1::detail::secure_erase(&k, sizeof(k));
        return UFSECP_ERR_BAD_KEY;
    }

    auto decomp   = glv_decompose(k);
    secp256k1::detail::secure_erase(&k, sizeof(k));
    auto k1_bytes = decomp.k1.to_bytes();
    auto k2_bytes = decomp.k2.to_bytes();

    /* Layout (264 bytes, matches OpenCL BIP352ScanKeyGlv):
     *  [0..129]  wnaf1[130]  — wNAF digits for k1 half-scalar
     *  [130..259] wnaf2[130] — wNAF digits for k2 half-scalar
     *  [260]     k1_neg      — 1 if k1 was negative
     *  [261]     flip_phi    — 1 if phi table y should be negated
     *  [262..263] pad        — zero padding                         */
    auto* wn1     = reinterpret_cast<int8_t*>(plan264_out);
    auto* wn2     = reinterpret_cast<int8_t*>(plan264_out + 130);
    plan264_out[260] = decomp.k1_neg ? 1 : 0;
    plan264_out[261] = (decomp.k1_neg != decomp.k2_neg) ? 1 : 0;
    plan264_out[262] = 0;
    plan264_out[263] = 0;

    /* 5-bit wNAF encoding — mirrors host_compute_wnaf() used by the
     * bip352_pipeline_kernel OpenCL kernel.                         */
    auto compute_wnaf = [](const uint8_t* be32, int8_t out[130]) {
        uint64_t s[4] = {};
        for (int limb = 0; limb < 4; ++limb) {
            uint64_t v = 0;
            int base = limb * 8;
            for (int i = 0; i < 8; ++i) v = (v << 8) | be32[base + i];
            s[3 - limb] = v;
        }
        for (int i = 0; i < 130; ++i) {
            if (s[0] & 1ULL) {
                int d = (int)(s[0] & 0x1FULL);
                if (d >= 16) {
                    d -= 32;
                    uint64_t add = static_cast<uint64_t>(-d);
                    uint64_t prev = s[0]; s[0] += add;
                    if (s[0] < prev) { for (int j = 1; j < 4; ++j) if (++s[j]) break; }
                } else {
                    uint64_t prev = s[0]; s[0] -= static_cast<uint64_t>(d);
                    if (s[0] > prev) { for (int j = 1; j < 4; ++j) if (s[j]--) break; }
                }
                out[i] = static_cast<int8_t>(d);
            } else {
                out[i] = 0;
            }
            s[0] = (s[0] >> 1) | (s[1] << 63);
            s[1] = (s[1] >> 1) | (s[2] << 63);
            s[2] = (s[2] >> 1) | (s[3] << 63);
            s[3] >>= 1;
        }
    };

    compute_wnaf(k1_bytes.data(), wn1);
    compute_wnaf(k2_bytes.data(), wn2);
    secp256k1::detail::secure_erase(k1_bytes.data(), k1_bytes.size());
    secp256k1::detail::secure_erase(k2_bytes.data(), k2_bytes.size());
    secp256k1::detail::secure_erase(&decomp, sizeof(decomp));
    return UFSECP_OK;
}

/* ===========================================================================
 * BIP-352 GPU batch scan
 * =========================================================================== */

ufsecp_error_t ufsecp_gpu_bip352_scan_batch(
    ufsecp_gpu_ctx* ctx,
    const uint8_t   scan_privkey32[32],
    const uint8_t   spend_pubkey33[33],
    const uint8_t*  tweak_pubkeys33,
    size_t          n_tweaks,
    uint64_t*       prefix64_out)
{
    if (!ctx || !scan_privkey32 || !spend_pubkey33 || !tweak_pubkeys33 || !prefix64_out)
        return UFSECP_ERR_NULL_ARG;
    if (n_tweaks > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    try {
        return to_abi_error(ctx->backend->bip352_scan_batch(
            scan_privkey32, spend_pubkey33, tweak_pubkeys33, n_tweaks, prefix64_out));
    } UFSECP_GPU_CATCH
}

/* ===========================================================================
 * GPU error string
 * =========================================================================== */

const char* ufsecp_gpu_error_str(ufsecp_error_t err) {
    switch (err) {
    case UFSECP_OK:                   return "OK";
    case UFSECP_ERR_NULL_ARG:         return "NULL argument";
    case UFSECP_ERR_BAD_KEY:          return "invalid private key";
    case UFSECP_ERR_BAD_PUBKEY:       return "invalid public key";
    case UFSECP_ERR_BAD_SIG:          return "invalid signature";
    case UFSECP_ERR_BAD_INPUT:        return "malformed input";
    case UFSECP_ERR_VERIFY_FAIL:      return "verification failed";
    case UFSECP_ERR_ARITH:            return "arithmetic error";
    case UFSECP_ERR_SELFTEST:         return "self-test failed";
    case UFSECP_ERR_INTERNAL:         return "internal error";
    case UFSECP_ERR_BUF_TOO_SMALL:    return "buffer too small";
    case UFSECP_ERR_GPU_UNAVAILABLE:  return "GPU backend unavailable";
    case UFSECP_ERR_GPU_DEVICE:       return "GPU device error";
    case UFSECP_ERR_GPU_LAUNCH:       return "GPU kernel launch failed";
    case UFSECP_ERR_GPU_MEMORY:       return "GPU memory error";
    case UFSECP_ERR_GPU_UNSUPPORTED:  return "operation not supported on this GPU backend";
    case UFSECP_ERR_GPU_BACKEND:      return "GPU backend driver error";
    case UFSECP_ERR_GPU_QUEUE:        return "GPU command queue error";
    default:                          return "unknown error";
    }
}
