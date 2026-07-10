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
#include "secp256k1/config.hpp"
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

/* Retain the self-installing GPU column-verify provider
 * (src/gpu/src/gpu_engine_hook.cpp). Its only content is a file-scope static
 * initializer with no external references, so a normal static link of
 * secp256k1_gpu_host drops the object and the engine column entrypoints
 * (secp256k1::ecdsa_batch_verify_opaque_columns / schnorr_batch_verify_bip340_columns)
 * would silently stay CPU-only. This C ABI TU is always retained in libufsecp (it
 * defines the exported ufsecp_gpu_* symbols), so referencing the provider's anchor
 * here drags that object into the shared library and runs its installer at load.
 * This is a link-retention reference, NOT a call: the libbitcoin-direct path still
 * never invokes the C ABI (that path retains the provider with a targeted
 * `--undefined=secp256k1_gpu_columns_provider_anchor` linker anchor at its own link,
 * NOT a blanket WHOLE_ARCHIVE — see compat/libbitcoin_direct/CMakeLists.txt).
 *
 * The retention reference is a STRONG undefined symbol, so it is only sound in a
 * link that actually carries the provider object. Every shipping consumer does:
 *   - libufsecp (ufsecp_shared / ufsecp_static) compiles this TU ONLY when the
 *     secp256k1_gpu_host target exists and always links it — that archive defines
 *     the anchor (see include/ufsecp/CMakeLists.txt UFSECP_GPU_IMPL_SRC guard);
 *   - the unified audit runner lists gpu_engine_hook.cpp among its own sources.
 * But the non-GPU CABI audit *standalone* tests compile this TU as a raw source
 * WITHOUT the provider (no secp256k1_gpu_host, no gpu_engine_hook.cpp) — there the
 * strong reference is an undefined symbol that breaks the link on the CPU-only
 * (GitHub) CI (LBTC-GPU-SELFINSTALL-DROP / anchor-link). Those targets define
 * UFSECP_NO_GPU_COLUMNS_PROVIDER_ANCHOR to drop the retention reference: they never
 * exercise the transparent GPU column-verify dispatch, so losing the self-install
 * hook there is a no-op. A weak reference is deliberately NOT used instead — a weak
 * undefined symbol does not pull the provider archive member, so the hook would
 * silently stop installing in libufsecp. Gate: retention on by default (correct for
 * every provider-carrying link), opted out only where the provider is absent by
 * construction (audit/CMakeLists.txt audit_target_defaults). */
#ifndef UFSECP_NO_GPU_COLUMNS_PROVIDER_ANCHOR
extern "C" int secp256k1_gpu_columns_provider_anchor;
namespace {
struct GpuColumnsProviderKeeper {
    GpuColumnsProviderKeeper() noexcept {
        volatile int keep = secp256k1_gpu_columns_provider_anchor;
        (void)keep;
    }
};
GpuColumnsProviderKeeper g_gpu_columns_provider_keeper;
}  // namespace
#endif  /* UFSECP_NO_GPU_COLUMNS_PROVIDER_ANCHOR */

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

static bool checked_add_size(std::size_t a, std::size_t b, std::size_t& out) {
    if (a > std::numeric_limits<std::size_t>::max() - b) {
        return false;
    }
    out = a + b;
    return true;
}

static bool checked_mul_size(std::size_t a, std::size_t b, std::size_t& out) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        return false;
    }
    out = a * b;
    return true;
}

static bool clear_output_bytes(void* out, std::size_t count, std::size_t stride) {
    if (!out || count == 0 || stride == 0) {
        return true;
    }
    std::size_t bytes = 0;
    if (!checked_mul_size(count, stride, bytes)) {
        return false;
    }
    std::memset(out, 0, bytes);
    return true;
}

static ufsecp_error_t to_abi_error_clear_on_fail(
    GpuError err,
    void* out,
    std::size_t count,
    std::size_t stride)
{
    const ufsecp_error_t abi_err = to_abi_error(err);
    if (abi_err != UFSECP_OK) {
        (void)clear_output_bytes(out, count, stride);
    }
    return abi_err;
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

uint32_t ufsecp_gpu_backend_count(uint32_t* backend_ids_out, uint32_t max_ids) noexcept {
    const uint32_t count = backend_count();
    if (backend_ids_out && max_ids > 0) {
        backend_ids(backend_ids_out, max_ids);
    }
    return count;
}

const char* ufsecp_gpu_backend_name(uint32_t bid) {
    switch (bid) {
    case 1: return "CUDA";
    case 2: return "OpenCL";
    case 3: return "Metal";
    default: return "none";
    }
}

int ufsecp_gpu_is_available(uint32_t bid) noexcept {
    return is_available(bid) ? 1 : 0;
}

uint32_t ufsecp_gpu_device_count(uint32_t bid) noexcept {
    auto b = create_backend(bid);
    if (!b) return 0;
    return b->device_count();
}

ufsecp_error_t ufsecp_gpu_device_info(
    uint32_t bid, uint32_t device_index,
    ufsecp_gpu_device_info_t* info_out)
{
    if (SECP256K1_UNLIKELY(!info_out)) return UFSECP_ERR_NULL_ARG;
    try {
    auto b = create_backend(bid);
    if (SECP256K1_UNLIKELY(!b)) return UFSECP_ERR_GPU_UNAVAILABLE;

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
    if (SECP256K1_UNLIKELY(!ctx_out)) return UFSECP_ERR_NULL_ARG;
    *ctx_out = nullptr;
    try {
    auto backend = create_backend(bid);
    if (SECP256K1_UNLIKELY(!backend)) return UFSECP_ERR_GPU_UNAVAILABLE;

    auto err = backend->init(device_index);
    if (err != GpuError::Ok) return to_abi_error(err);

    auto* ctx = new (std::nothrow) ufsecp_gpu_ctx;
    if (SECP256K1_UNLIKELY(!ctx)) return UFSECP_ERR_INTERNAL;

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

int ufsecp_gpu_is_ready(const ufsecp_gpu_ctx* ctx) noexcept {
    if (!ctx || !ctx->backend) return 0;
    return ctx->backend->is_ready() ? 1 : 0;
}

ufsecp_error_t ufsecp_gpu_last_error(const ufsecp_gpu_ctx* ctx) {
    if (SECP256K1_UNLIKELY(!ctx)) return UFSECP_ERR_NULL_ARG;
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
    if (SECP256K1_UNLIKELY(!ctx)) return UFSECP_ERR_NULL_ARG;
    if (count == 0) return UFSECP_OK;
    if (count > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    if (!clear_output_bytes(out_pubkeys33, count, 33)) return UFSECP_ERR_BAD_INPUT;
    if (SECP256K1_UNLIKELY(!scalars32 || !out_pubkeys33)) return UFSECP_ERR_NULL_ARG;
    try {
    return to_abi_error_clear_on_fail(
        ctx->backend->generator_mul_batch(scalars32, count, out_pubkeys33),
        out_pubkeys33, count, 33);
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
    if (SECP256K1_UNLIKELY(!ctx)) return UFSECP_ERR_NULL_ARG;
    if (count == 0) return UFSECP_OK;
    if (count > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    if (!clear_output_bytes(out_results, count, 1)) return UFSECP_ERR_BAD_INPUT;
    if (SECP256K1_UNLIKELY(!msg_hashes32 || !pubkeys33 || !sigs64 || !out_results)) {
        return UFSECP_ERR_NULL_ARG;
    }
    try {
    return to_abi_error_clear_on_fail(
        ctx->backend->ecdsa_verify_batch(
            msg_hashes32, pubkeys33, sigs64, count, out_results),
        out_results, count, 1);
    } UFSECP_GPU_CATCH
}

ufsecp_error_t ufsecp_gpu_ecdsa_verify_opaque_rows(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* rows,
    size_t stride,
    size_t count,
    uint8_t* out_results)
{
    if (SECP256K1_UNLIKELY(!ctx)) return UFSECP_ERR_NULL_ARG;
    if (count == 0) return UFSECP_OK;
    if (count > kMaxGpuBatchN || stride < 129u) return UFSECP_ERR_BAD_INPUT;
    if (SECP256K1_UNLIKELY(!rows || !out_results)) {
        return UFSECP_ERR_NULL_ARG;
    }
    if (!clear_output_bytes(out_results, count, 1)) return UFSECP_ERR_BAD_INPUT;
    try {
        return to_abi_error_clear_on_fail(
            ctx->backend->ecdsa_verify_lbtc_rows(rows, stride, count, out_results),
            out_results, count, 1);
    } UFSECP_GPU_CATCH
}

ufsecp_error_t ufsecp_gpu_ecdsa_verify_lbtc_rows(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* rows,
    size_t stride,
    size_t count,
    uint8_t* out_results)
{
    if (SECP256K1_UNLIKELY(!ctx)) return UFSECP_ERR_NULL_ARG;
    if (count == 0) return UFSECP_OK;
    if (count > kMaxGpuBatchN || stride < 129u) return UFSECP_ERR_BAD_INPUT;
    if (SECP256K1_UNLIKELY(!rows || !out_results)) {
        return UFSECP_ERR_NULL_ARG;
    }
    return ufsecp_gpu_ecdsa_verify_opaque_rows(
        ctx, rows, stride, count, out_results);
}

ufsecp_error_t ufsecp_gpu_schnorr_verify_batch(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* msg_hashes32,
    const uint8_t* pubkeys_x32,
    const uint8_t* sigs64,
    size_t count,
    uint8_t* out_results)
{
    if (SECP256K1_UNLIKELY(!ctx)) return UFSECP_ERR_NULL_ARG;
    if (count == 0) return UFSECP_OK;
    if (count > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    if (!clear_output_bytes(out_results, count, 1)) return UFSECP_ERR_BAD_INPUT;
    if (SECP256K1_UNLIKELY(!msg_hashes32 || !pubkeys_x32 || !sigs64 || !out_results)) {
        return UFSECP_ERR_NULL_ARG;
    }
    try {
    return to_abi_error_clear_on_fail(
        ctx->backend->schnorr_verify_batch(
            msg_hashes32, pubkeys_x32, sigs64, count, out_results),
        out_results, count, 1);
    } UFSECP_GPU_CATCH
}

/* libbitcoin ECDSA/Schnorr column (Structure-of-Arrays) verify. Public-data.
 * Completeness mirror of the *_verify_lbtc_rows / *_verify_batch ABI wrappers.
 * NOTE: the libbitcoin direct entrypoint does NOT call these C ABI functions —
 * it calls the C++ GpuBackend methods directly (GPU is an internal accelerator,
 * no separate caller-facing GPU API/status). These exist to satisfy the C ABI
 * completeness rule for every GpuBackend virtual method. */
ufsecp_error_t ufsecp_gpu_ecdsa_verify_lbtc_columns(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* digests32,
    const uint8_t* pubkeys33,
    const uint8_t* sigs64,
    size_t count,
    uint8_t* out_results)
{
    if (SECP256K1_UNLIKELY(!ctx)) return UFSECP_ERR_NULL_ARG;
    if (count == 0) return UFSECP_OK;
    if (count > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    if (!clear_output_bytes(out_results, count, 1)) return UFSECP_ERR_BAD_INPUT;
    if (SECP256K1_UNLIKELY(!digests32 || !pubkeys33 || !sigs64 || !out_results)) {
        return UFSECP_ERR_NULL_ARG;
    }
    try {
    return to_abi_error_clear_on_fail(
        ctx->backend->ecdsa_verify_lbtc_columns(
            digests32, pubkeys33, sigs64, count, out_results),
        out_results, count, 1);
    } UFSECP_GPU_CATCH
}

ufsecp_error_t ufsecp_gpu_schnorr_verify_lbtc_columns(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* digests32,
    const uint8_t* xonly32,
    const uint8_t* sigs64,
    size_t count,
    uint8_t* out_results)
{
    if (SECP256K1_UNLIKELY(!ctx)) return UFSECP_ERR_NULL_ARG;
    if (count == 0) return UFSECP_OK;
    if (count > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    if (!clear_output_bytes(out_results, count, 1)) return UFSECP_ERR_BAD_INPUT;
    if (SECP256K1_UNLIKELY(!digests32 || !xonly32 || !sigs64 || !out_results)) {
        return UFSECP_ERR_NULL_ARG;
    }
    try {
    return to_abi_error_clear_on_fail(
        ctx->backend->schnorr_verify_lbtc_columns(
            digests32, xonly32, sigs64, count, out_results),
        out_results, count, 1);
    } UFSECP_GPU_CATCH
}

ufsecp_error_t ufsecp_gpu_ecdsa_verify_collect(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* msg_hashes32,
    const uint8_t* pubkeys33,
    const uint8_t* sigs64,
    size_t count,
    uint8_t* key_buffer)
{
    if (SECP256K1_UNLIKELY(!ctx)) return UFSECP_ERR_NULL_ARG;
    if (count == 0) return UFSECP_OK;
    if (SECP256K1_UNLIKELY(!msg_hashes32 || !pubkeys33 || !sigs64 || !key_buffer)) {
        return UFSECP_ERR_NULL_ARG;
    }
    if (count > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    try {
    return to_abi_error(
        ctx->backend->ecdsa_verify_collect(
            msg_hashes32, pubkeys33, sigs64, count, key_buffer));
    } UFSECP_GPU_CATCH
}

ufsecp_error_t ufsecp_gpu_schnorr_verify_collect(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* msg_hashes32,
    const uint8_t* pubkeys_x32,
    const uint8_t* sigs64,
    size_t count,
    uint8_t* key_buffer)
{
    if (SECP256K1_UNLIKELY(!ctx)) return UFSECP_ERR_NULL_ARG;
    if (count == 0) return UFSECP_OK;
    if (SECP256K1_UNLIKELY(!msg_hashes32 || !pubkeys_x32 || !sigs64 || !key_buffer)) {
        return UFSECP_ERR_NULL_ARG;
    }
    if (count > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    try {
    return to_abi_error(
        ctx->backend->schnorr_verify_collect(
            msg_hashes32, pubkeys_x32, sigs64, count, key_buffer));
    } UFSECP_GPU_CATCH
}

ufsecp_error_t ufsecp_gpu_ecdh_batch(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* privkeys32,
    const uint8_t* peer_pubkeys33,
    size_t count,
    uint8_t* out_secrets32)
{
    if (SECP256K1_UNLIKELY(!ctx)) return UFSECP_ERR_NULL_ARG;
    if (count == 0) return UFSECP_OK;
    if (count > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    if (!clear_output_bytes(out_secrets32, count, 32)) return UFSECP_ERR_BAD_INPUT;
    if (SECP256K1_UNLIKELY(!privkeys32 || !peer_pubkeys33 || !out_secrets32)) {
        return UFSECP_ERR_NULL_ARG;
    }
    if (!has_valid_compressed_pubkeys(peer_pubkeys33, count)) {
        return UFSECP_ERR_BAD_PUBKEY;
    }
    try {
    return to_abi_error_clear_on_fail(
        ctx->backend->ecdh_batch(
            privkeys32, peer_pubkeys33, count, out_secrets32),
        out_secrets32, count, 32);
    } UFSECP_GPU_CATCH
}

ufsecp_error_t ufsecp_gpu_hash160_pubkey_batch(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* pubkeys33,
    size_t count,
    uint8_t* out_hash160)
{
    if (SECP256K1_UNLIKELY(!ctx)) return UFSECP_ERR_NULL_ARG;
    if (count == 0) return UFSECP_OK;
    if (count > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    if (!clear_output_bytes(out_hash160, count, 20)) return UFSECP_ERR_BAD_INPUT;
    if (SECP256K1_UNLIKELY(!pubkeys33 || !out_hash160)) return UFSECP_ERR_NULL_ARG;
    if (!has_valid_compressed_pubkeys(pubkeys33, count)) {
        return UFSECP_ERR_BAD_PUBKEY;
    }
    try {
    return to_abi_error_clear_on_fail(
        ctx->backend->hash160_pubkey_batch(pubkeys33, count, out_hash160),
        out_hash160, count, 20);
    } UFSECP_GPU_CATCH
}

ufsecp_error_t ufsecp_gpu_msm(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* scalars32,
    const uint8_t* points33,
    size_t n,
    uint8_t* out_result33)
{
    if (SECP256K1_UNLIKELY(!ctx)) return UFSECP_ERR_NULL_ARG;
    if (n == 0) return UFSECP_OK;
    if (n > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    if (!clear_output_bytes(out_result33, 1, 33)) return UFSECP_ERR_BAD_INPUT;
    if (SECP256K1_UNLIKELY(!scalars32 || !points33 || !out_result33)) return UFSECP_ERR_NULL_ARG;
    try {
    return to_abi_error_clear_on_fail(
        ctx->backend->msm(scalars32, points33, n, out_result33),
        out_result33, 1, 33);
    } UFSECP_GPU_CATCH
}

ufsecp_error_t ufsecp_gpu_xonly_validate(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* keys32,
    size_t n,
    uint8_t* results)
{
    if (SECP256K1_UNLIKELY(!ctx)) return UFSECP_ERR_NULL_ARG;
    if (n == 0) return UFSECP_OK;
    if (n > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    if (!clear_output_bytes(results, n, 1)) return UFSECP_ERR_BAD_INPUT;
    if (SECP256K1_UNLIKELY(!keys32 || !results)) return UFSECP_ERR_NULL_ARG;
    try {
    return to_abi_error_clear_on_fail(
        ctx->backend->xonly_validate(keys32, n, results),
        results, n, 1);
    } UFSECP_GPU_CATCH
}

ufsecp_error_t ufsecp_gpu_commitment_verify(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* internal_x32,
    const uint8_t* tweak32,
    const uint8_t* tweaked_x32,
    const uint8_t* parity,
    size_t n,
    uint8_t* results)
{
    if (SECP256K1_UNLIKELY(!ctx)) return UFSECP_ERR_NULL_ARG;
    if (n == 0) return UFSECP_OK;
    if (n > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    if (!clear_output_bytes(results, n, 1)) return UFSECP_ERR_BAD_INPUT;
    if (SECP256K1_UNLIKELY(!internal_x32 || !tweak32 || !tweaked_x32 || !parity || !results))
        return UFSECP_ERR_NULL_ARG;
    try {
    return to_abi_error_clear_on_fail(
        ctx->backend->commitment_verify(internal_x32, tweak32, tweaked_x32, parity, n, results),
        results, n, 1);
    } UFSECP_GPU_CATCH
}

ufsecp_error_t ufsecp_gpu_tagged_hash(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* tag_hash32,
    const uint8_t* msgs,
    size_t msg_len,
    size_t n,
    uint8_t* out32)
{
    if (SECP256K1_UNLIKELY(!ctx)) return UFSECP_ERR_NULL_ARG;
    if (n == 0) return UFSECP_OK;
    if (n > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    if (msg_len == 0 || msg_len > 256) return UFSECP_ERR_BAD_INPUT;
    if (!clear_output_bytes(out32, n, 32)) return UFSECP_ERR_BAD_INPUT;
    if (SECP256K1_UNLIKELY(!tag_hash32 || !msgs || !out32)) return UFSECP_ERR_NULL_ARG;
    try {
    return to_abi_error_clear_on_fail(
        ctx->backend->tagged_hash(tag_hash32, msgs, msg_len, n, out32),
        out32, n, 32);
    } UFSECP_GPU_CATCH
}

ufsecp_error_t ufsecp_gpu_pubkey_validate(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* pubkeys33,
    size_t n,
    uint8_t* results)
{
    if (SECP256K1_UNLIKELY(!ctx)) return UFSECP_ERR_NULL_ARG;
    if (n == 0) return UFSECP_OK;
    if (n > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    if (!clear_output_bytes(results, n, 1)) return UFSECP_ERR_BAD_INPUT;
    if (SECP256K1_UNLIKELY(!pubkeys33 || !results)) return UFSECP_ERR_NULL_ARG;
    try {
    return to_abi_error_clear_on_fail(
        ctx->backend->pubkey_validate(pubkeys33, n, results),
        results, n, 1);
    } UFSECP_GPU_CATCH
}

ufsecp_error_t ufsecp_gpu_tagged_hash_var(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* tag_hash32,
    const uint8_t* msgs,
    const uint32_t* msg_lens,
    size_t stride,
    size_t n,
    uint8_t* out32)
{
    if (SECP256K1_UNLIKELY(!ctx)) return UFSECP_ERR_NULL_ARG;
    if (n == 0) return UFSECP_OK;
    if (n > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    if (stride == 0 || stride > 256) return UFSECP_ERR_BAD_INPUT;
    if (!clear_output_bytes(out32, n, 32)) return UFSECP_ERR_BAD_INPUT;
    if (SECP256K1_UNLIKELY(!tag_hash32 || !msgs || !msg_lens || !out32)) return UFSECP_ERR_NULL_ARG;
    try {
    return to_abi_error_clear_on_fail(
        ctx->backend->tagged_hash_var(tag_hash32, msgs, msg_lens, stride, n, out32),
        out32, n, 32);
    } UFSECP_GPU_CATCH
}

ufsecp_error_t ufsecp_gpu_hash256(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* inputs,
    size_t input_len,
    size_t n,
    uint8_t* out32)
{
    if (SECP256K1_UNLIKELY(!ctx)) return UFSECP_ERR_NULL_ARG;
    if (n == 0) return UFSECP_OK;
    if (n > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    if (input_len == 0 || input_len > 320) return UFSECP_ERR_BAD_INPUT;
    if (!clear_output_bytes(out32, n, 32)) return UFSECP_ERR_BAD_INPUT;
    if (SECP256K1_UNLIKELY(!inputs || !out32)) return UFSECP_ERR_NULL_ARG;
    try {
    return to_abi_error_clear_on_fail(
        ctx->backend->hash256(inputs, input_len, n, out32),
        out32, n, 32);
    } UFSECP_GPU_CATCH
}

/* Hard upper bound on hash256_var's per-row stride. Matches Bitcoin's
 * maximum block-weight-derived transaction size ceiling (~4 MiB), giving
 * generous headroom over realistic standard-relay transaction sizes
 * (~100 KB) while bounding worst-case per-call host/device work. Deliberately
 * NOT the tagged_hash_var precedent's 256-byte cap — that cap exists because
 * tagged_hash_var must fit a 64-byte tag prefix in a small on-chip scratch
 * buffer; hash256_var has no tag prefix and streams each row directly, so it
 * is not bound by that same buffer. */
static constexpr std::size_t kMaxHash256VarStride = std::size_t{4} * 1024 * 1024;  /* 4 MiB */

ufsecp_error_t ufsecp_gpu_hash256_var(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* inputs,
    const uint32_t* input_lens,
    size_t stride,
    size_t n,
    uint8_t* out32)
{
    if (SECP256K1_UNLIKELY(!ctx)) return UFSECP_ERR_NULL_ARG;
    if (n == 0) return UFSECP_OK;
    if (n > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    if (stride == 0 || stride > kMaxHash256VarStride) return UFSECP_ERR_BAD_INPUT;
    if (!clear_output_bytes(out32, n, 32)) return UFSECP_ERR_BAD_INPUT;
    if (SECP256K1_UNLIKELY(!inputs || !input_lens || !out32)) return UFSECP_ERR_NULL_ARG;
    /* Host-side per-row validation: every row's real length must be in
     * [1, stride]. Neither ufsecp_gpu_hash256 (fixed input_len) nor
     * ufsecp_gpu_tagged_hash_var (variable msg_lens) validates individual
     * per-row lengths against stride/bounds in this ABI wrapper — that gap
     * must not be repeated here, since hash256_var's row lengths are fully
     * caller-controlled and an out-of-range row would read past its slot. */
    for (size_t i = 0; i < n; ++i) {
        if (input_lens[i] == 0 || input_lens[i] > stride) return UFSECP_ERR_BAD_INPUT;
    }
    try {
    return to_abi_error_clear_on_fail(
        ctx->backend->hash256_var(inputs, input_lens, stride, n, out32),
        out32, n, 32);
    } UFSECP_GPU_CATCH
}

ufsecp_error_t ufsecp_gpu_merkle_pair_hash(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* left32,
    const uint8_t* right32,
    size_t n,
    uint8_t* out32)
{
    if (SECP256K1_UNLIKELY(!ctx)) return UFSECP_ERR_NULL_ARG;
    if (n == 0) return UFSECP_OK;
    if (n > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    if (!clear_output_bytes(out32, n, 32)) return UFSECP_ERR_BAD_INPUT;
    if (SECP256K1_UNLIKELY(!left32 || !right32 || !out32)) return UFSECP_ERR_NULL_ARG;
    /* Fixed 64-byte combined input per row — no per-op length cap check needed
     * (unlike hash256's input_len<=320 or tagged_hash's msg_len<=256). The
     * kernel always processes exactly 64 bytes per row via lbtc_sha256. */
    try {
        return to_abi_error_clear_on_fail(
            ctx->backend->merkle_pair_hash(left32, right32, n, out32),
            out32, n, 32);
    } UFSECP_GPU_CATCH
}

ufsecp_error_t ufsecp_gpu_sighash_descriptor_hash(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* descriptor,
    size_t descriptor_len,
    const uint8_t* const* field_data,
    const uint32_t* field_lengths,
    const uint32_t* const* field_var_lens,
    size_t count,
    uint8_t* out32)
{
    if (SECP256K1_UNLIKELY(!ctx)) return UFSECP_ERR_NULL_ARG;
    if (count == 0) return UFSECP_OK;
    if (count > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    if (!clear_output_bytes(out32, count, 32)) return UFSECP_ERR_BAD_INPUT;
    if (SECP256K1_UNLIKELY(!descriptor || !field_data || !field_lengths || !out32))
        return UFSECP_ERR_NULL_ARG;

    /* Descriptor pre-dispatch validation (mirrors libbitcoin.hpp v2 contract).
     * Fail-closed: reject malformed descriptors, count*stride overflow,
     * var_len > stride, and total preimage > 4 MiB before any GPU dispatch;
     * out32 is already cleared by clear_output_bytes above on error paths.
     * Taproot field IDs 0x0C..0x0F are rejected by the backend parser. */
    if (descriptor_len < 1 || descriptor_len > 129) return UFSECP_ERR_BAD_INPUT;
    if ((descriptor_len & 1u) == 0) return UFSECP_ERR_BAD_INPUT;
    if (descriptor[descriptor_len - 1] != 0xFF) return UFSECP_ERR_BAD_INPUT;
    for (size_t i = 0; i < descriptor_len - 1; i += 2) {
        if (descriptor[i] == 0xFF) return UFSECP_ERR_BAD_INPUT;
    }

    /* Bounds: count must fit in int32 for the kernel launch, and field_data
     * stride must not overflow. The backend performs deeper per-field checks
     * (count*stride overflow, var_len <= stride, total preimage <= 4 MiB). */
    if (count > static_cast<size_t>(INT32_MAX)) return UFSECP_ERR_BAD_INPUT;

    try {
        return to_abi_error_clear_on_fail(
            ctx->backend->sighash_descriptor_hash(
                descriptor, descriptor_len, field_data, field_lengths,
                field_var_lens, count, out32),
            out32, count, 32);
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
    if (SECP256K1_UNLIKELY(!ctx)) return UFSECP_ERR_NULL_ARG;
    if (count == 0) return UFSECP_OK;
    if (count > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    if (!clear_output_bytes(out_results, count, 1)) return UFSECP_ERR_BAD_INPUT;
    if (SECP256K1_UNLIKELY(!z_i32 || !D_i33 || !E_i33 || !Y_i33 || !rho_i32 ||
        !lambda_ie32 || !negate_R || !negate_key || !out_results)) {
        return UFSECP_ERR_NULL_ARG;
    }
    if (!has_valid_compressed_pubkeys(D_i33, count) ||
        !has_valid_compressed_pubkeys(E_i33, count) ||
        !has_valid_compressed_pubkeys(Y_i33, count)) {
        return UFSECP_ERR_BAD_PUBKEY;
    }
    try {
    return to_abi_error_clear_on_fail(
        ctx->backend->frost_verify_partial_batch(
            z_i32, D_i33, E_i33, Y_i33, rho_i32, lambda_ie32,
            negate_R, negate_key, count, out_results),
        out_results, count, 1);
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
    if (SECP256K1_UNLIKELY(!ctx)) return UFSECP_ERR_NULL_ARG;
    if (count == 0) return UFSECP_OK;
    if (count > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    if (!clear_output_bytes(out_pubkeys33, count, 33)) return UFSECP_ERR_BAD_INPUT;
    if (!clear_output_bytes(out_valid, count, 1)) return UFSECP_ERR_BAD_INPUT;
    if (SECP256K1_UNLIKELY(!msg_hashes32 || !sigs64 || !recids || !out_pubkeys33 || !out_valid)) {
        return UFSECP_ERR_NULL_ARG;
    }
    if (!has_valid_recovery_ids(recids, count)) {
        return UFSECP_ERR_BAD_INPUT;
    }
    try {
    const auto err = ctx->backend->ecrecover_batch(
        msg_hashes32, sigs64, recids, count, out_pubkeys33, out_valid);
    const ufsecp_error_t abi_err = to_abi_error(err);
    if (abi_err != UFSECP_OK) {
        (void)clear_output_bytes(out_pubkeys33, count, 33);
        (void)clear_output_bytes(out_valid, count, 1);
    }
    return abi_err;
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
    if (SECP256K1_UNLIKELY(!ctx)) return UFSECP_ERR_NULL_ARG;
    if (count == 0) return UFSECP_OK;
    if (count > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    if (!clear_output_bytes(out_results, count, 1)) return UFSECP_ERR_BAD_INPUT;
    if (!proofs64 || !pubkeys65 || !messages32 || !out_results)
        return UFSECP_ERR_NULL_ARG;
    if (!has_valid_uncompressed_pubkeys(pubkeys65, count)) {
        return UFSECP_ERR_BAD_PUBKEY;
    }
    try {
    return to_abi_error_clear_on_fail(
        ctx->backend->zk_knowledge_verify_batch(
            proofs64, pubkeys65, messages32, count, out_results),
        out_results, count, 1);
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
    if (SECP256K1_UNLIKELY(!ctx)) return UFSECP_ERR_NULL_ARG;
    if (count == 0) return UFSECP_OK;
    if (count > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    if (!clear_output_bytes(out_results, count, 1)) return UFSECP_ERR_BAD_INPUT;
    if (!proofs64 || !G_pts65 || !H_pts65 || !P_pts65 || !Q_pts65 || !out_results)
        return UFSECP_ERR_NULL_ARG;
    if (!has_valid_uncompressed_pubkeys(G_pts65, count) ||
        !has_valid_uncompressed_pubkeys(H_pts65, count) ||
        !has_valid_uncompressed_pubkeys(P_pts65, count) ||
        !has_valid_uncompressed_pubkeys(Q_pts65, count)) {
        return UFSECP_ERR_BAD_PUBKEY;
    }
    try {
    return to_abi_error_clear_on_fail(
        ctx->backend->zk_dleq_verify_batch(
            proofs64, G_pts65, H_pts65, P_pts65, Q_pts65, count, out_results),
        out_results, count, 1);
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
    if (SECP256K1_UNLIKELY(!ctx)) return UFSECP_ERR_NULL_ARG;
    if (count == 0) return UFSECP_OK;
    if (count > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    if (!clear_output_bytes(out_results, count, 1)) return UFSECP_ERR_BAD_INPUT;
    if (!proofs324 || !commitments65 || !H_generator65 || !out_results)
        return UFSECP_ERR_NULL_ARG;
    if (!has_valid_bulletproof_prefixes(proofs324, count) ||
        !has_valid_uncompressed_pubkeys(commitments65, count) ||
        !has_valid_uncompressed_pubkeys(H_generator65, 1)) {
        return UFSECP_ERR_BAD_PUBKEY;
    }
    try {
    return to_abi_error_clear_on_fail(
        ctx->backend->bulletproof_verify_batch(
            proofs324, commitments65, H_generator65, count, out_results),
        out_results, count, 1);
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
    if (SECP256K1_UNLIKELY(!ctx)) return UFSECP_ERR_NULL_ARG;
    if (count == 0) return UFSECP_OK;
    if (count > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    std::size_t wire_stride = 0;
    if (!checked_add_size(static_cast<std::size_t>(max_payload), 19u, wire_stride)) {
        return UFSECP_ERR_BAD_INPUT;
    }
    if (!clear_output_bytes(wire_out, count, wire_stride)) return UFSECP_ERR_BAD_INPUT;
    if (!keys32 || !nonces12 || !plaintexts || !sizes || !wire_out)
        return UFSECP_ERR_NULL_ARG;
    if (!has_valid_bip324_sizes(sizes, count, max_payload)) {
        return UFSECP_ERR_BAD_INPUT;
    }
    try {
    return to_abi_error_clear_on_fail(
        ctx->backend->bip324_aead_encrypt_batch(
            keys32, nonces12, plaintexts, sizes, max_payload, count, wire_out),
        wire_out, count, wire_stride);
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
    if (SECP256K1_UNLIKELY(!ctx)) return UFSECP_ERR_NULL_ARG;
    if (count == 0) return UFSECP_OK;
    if (count > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    if (!clear_output_bytes(plaintext_out, count, static_cast<std::size_t>(max_payload))) {
        return UFSECP_ERR_BAD_INPUT;
    }
    if (!clear_output_bytes(out_valid, count, 1)) return UFSECP_ERR_BAD_INPUT;
    if (!keys32 || !nonces12 || !wire_in || !sizes || !plaintext_out || !out_valid)
        return UFSECP_ERR_NULL_ARG;
    if (!has_valid_bip324_sizes(sizes, count, max_payload)) {
        return UFSECP_ERR_BAD_INPUT;
    }
    try {
    const auto err = ctx->backend->bip324_aead_decrypt_batch(
        keys32, nonces12, wire_in, sizes, max_payload, count,
        plaintext_out, out_valid);
    const ufsecp_error_t abi_err = to_abi_error(err);
    if (abi_err != UFSECP_OK) {
        (void)clear_output_bytes(plaintext_out, count, static_cast<std::size_t>(max_payload));
        (void)clear_output_bytes(out_valid, count, 1);
    }
    return abi_err;
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
    if (count > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    if (!clear_output_bytes(out_witnesses, count, UFSECP_ECDSA_SNARK_WITNESS_BYTES)) {
        return UFSECP_ERR_BAD_INPUT;
    }
    if (!ctx || !msg_hashes32 || !pubkeys33 || !sigs64 || !out_witnesses)
        return UFSECP_ERR_NULL_ARG;
    // Reject malformed compressed pubkeys before dispatch (prefix must be 0x02/0x03),
    // matching the input-validation done by bulletproof_verify_batch and the other GPU
    // ZK wrappers. Without this, an off-curve / wrong-prefix pubkey silently produces a
    // garbage witness instead of an error (SW-16).
    if (!has_valid_compressed_pubkeys(pubkeys33, count))
        return UFSECP_ERR_BAD_PUBKEY;
    try {
        return to_abi_error_clear_on_fail(
            ctx->backend->snark_witness_batch(
                msg_hashes32, pubkeys33, sigs64, count, out_witnesses),
            out_witnesses, count, UFSECP_ECDSA_SNARK_WITNESS_BYTES);
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
    if (count > kMaxGpuBatchN) return UFSECP_ERR_BAD_INPUT;
    if (!clear_output_bytes(out_witnesses, count, UFSECP_SCHNORR_SNARK_WITNESS_BYTES)) {
        return UFSECP_ERR_BAD_INPUT;
    }
    if (!ctx || !msgs32 || !pubkeys_x32 || !sigs64 || !out_witnesses)
        return UFSECP_ERR_NULL_ARG;
    // Reject degenerate signatures before dispatch: a BIP-340 sig is (R.x[32] || s[32]),
    // and s == 0 is never valid. This catches all-zero / uninitialized input that would
    // otherwise silently produce a garbage witness (mirrors the snark/bulletproof
    // input-validation pattern; x-only pubkey shape has no prefix to check).
    for (size_t i = 0; i < count; ++i) {
        const uint8_t* s = sigs64 + i * 64 + 32;
        bool s_zero = true;
        for (int b = 0; b < 32; ++b) { if (s[b] != 0) { s_zero = false; break; } }
        if (s_zero) return UFSECP_ERR_BAD_SIG;
    }
    try {
        return to_abi_error_clear_on_fail(
            ctx->backend->schnorr_snark_witness_batch(
                msgs32, pubkeys_x32, sigs64, count, out_witnesses),
            out_witnesses, count, UFSECP_SCHNORR_SNARK_WITNESS_BYTES);
    } UFSECP_GPU_CATCH
}

/* ===========================================================================
 * BIP-352 scan plan precomputation (CPU utility, no GPU ctx needed)
 * =========================================================================== */

ufsecp_error_t ufsecp_bip352_prepare_scan_plan(
    const uint8_t scan_privkey32[32],
    uint8_t       plan264_out[264])
{
    if (SECP256K1_UNLIKELY(!scan_privkey32 || !plan264_out)) return UFSECP_ERR_NULL_ARG;

    using namespace secp256k1::fast;

    std::memset(plan264_out, 0, 264);
    Scalar k;
    if (!Scalar::parse_bytes_strict_nonzero(scan_privkey32, k)) {
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
    if (!clear_output_bytes(prefix64_out, n_tweaks, sizeof(uint64_t))) {
        return UFSECP_ERR_BAD_INPUT;
    }
    {
        secp256k1::fast::Scalar scan_k;
        if (!secp256k1::fast::Scalar::parse_bytes_strict_nonzero(scan_privkey32, scan_k)) {
            secp256k1::detail::secure_erase(&scan_k, sizeof(scan_k));
            return UFSECP_ERR_BAD_KEY;
        }
        secp256k1::detail::secure_erase(&scan_k, sizeof(scan_k));
    }
    // Validate compressed pubkey prefixes (must be 0x02 or 0x03) for spend_pubkey
    // and every tweak pubkey. Reject invalid input before it reaches the GPU kernel.
    if (spend_pubkey33[0] != 0x02 && spend_pubkey33[0] != 0x03)
        return UFSECP_ERR_BAD_INPUT;
    for (size_t i = 0; i < n_tweaks; ++i) {
        const uint8_t prefix = tweak_pubkeys33[i * 33];
        if (prefix != 0x02 && prefix != 0x03)
            return UFSECP_ERR_BAD_INPUT;
    }
    try {
        return to_abi_error_clear_on_fail(
            ctx->backend->bip352_scan_batch(
                scan_privkey32, spend_pubkey33, tweak_pubkeys33, n_tweaks, prefix64_out),
            prefix64_out, n_tweaks, sizeof(uint64_t));
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
