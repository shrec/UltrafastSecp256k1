/* ============================================================================
 * UltrafastSecp256k1 -- Metal Backend Bridge
 * ============================================================================
 * Implements gpu::GpuBackend for Apple Metal.
 * Wraps the existing secp256k1::metal::MetalRuntime class.
 *
 * STATUS: All 7 first-wave batch operations are wired to Metal GPU kernels.
 *
 * Compiled ONLY when SECP256K1_HAVE_METAL is set (via CMake).
 * Must be compiled as Objective-C++ (.mm) on macOS.
 * ============================================================================ */

#include "../include/gpu_backend.hpp"

#include <cstring>
#include <cstdio>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <sstream>

/* -- Metal Runtime (Layer 1) ----------------------------------------------- */
#include "metal_runtime.h"

/* -- CPU FieldElement for host-side point decompression -------------------- */
#include "secp256k1/field.hpp"

/* -- CPU SHA-256 for ECDH finalization ------------------------------------- */
#include "secp256k1/sha256.hpp"

/* -- CPU Hash160 fallback -------------------------------------------------- */
#include "secp256k1/hash_accel.hpp"

namespace secp256k1 {
namespace gpu {

// =============================================================================
// Host-side struct layouts matching the Metal shader structs.
// Metal uses uint32_t limbs[8], LE limbs, BE byte encoding:
//   limbs[7] = BE bytes[0..3] (MSW), limbs[0] = BE bytes[28..31] (LSW)
// =============================================================================

struct MetalScalar256   { uint32_t limbs[8]; };
struct MetalFieldElem   { uint32_t limbs[8]; };
struct MetalAffinePoint { MetalFieldElem x, y; };

// =============================================================================
// Conversion helpers
// =============================================================================

namespace {

/** Read a file to string, returning empty string on failure. */
static std::string metal_load_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    return {std::istreambuf_iterator<char>(f), {}};
}

struct MetalBatchScratch {
    std::vector<MetalScalar256> scalars;
    std::vector<MetalAffinePoint> affine_points;
    std::vector<uint8_t> pubkeys64;

    void ensure_scalars(std::size_t count) {
        if (scalars.size() < count) scalars.resize(count);
    }

    void ensure_affine_points(std::size_t count) {
        if (affine_points.size() < count) affine_points.resize(count);
    }

    uint8_t* ensure_pubkeys64(std::size_t count) {
        std::size_t const bytes = count * 64;
        if (pubkeys64.size() < bytes) pubkeys64.resize(bytes);
        return pubkeys64.data();
    }
};

static thread_local MetalBatchScratch g_metal_batch_scratch;

/** Convert big-endian 32 bytes → MetalScalar256 (host-side, no reduction). */
static MetalScalar256 be32_to_metal_scalar(const uint8_t be[32]) {
    MetalScalar256 s;
    for (int i = 0; i < 8; i++) {
        int base = (7 - i) * 4;
        s.limbs[i] = ((uint32_t)be[base]   << 24) |
                     ((uint32_t)be[base+1]  << 16) |
                     ((uint32_t)be[base+2]  << 8)  |
                     ((uint32_t)be[base+3]);
    }
    return s;
}

/** Convert MetalFieldElem → big-endian 32 bytes
 *  (limbs[7] = MSW → bytes[0..3], limbs[0] = LSW → bytes[28..31]). */
static void metal_fe_to_be32(const MetalFieldElem& fe, uint8_t out[32]) {
    for (int i = 0; i < 8; i++) {
        /* limb 7 → bytes[0..3], limb 6 → bytes[4..7], ..., limb 0 → bytes[28..31] */
        int out_base = (7 - i) * 4;
        out[out_base]   = (uint8_t)(fe.limbs[i] >> 24);
        out[out_base+1] = (uint8_t)(fe.limbs[i] >> 16);
        out[out_base+2] = (uint8_t)(fe.limbs[i] >> 8);
        out[out_base+3] = (uint8_t)(fe.limbs[i]);
    }
}

/** Compress MetalAffinePoint → SEC1 33-byte compressed pubkey. */
static void metal_affine_to_sec1(const MetalAffinePoint& pt, uint8_t out33[33]) {
    /* y parity from LSB of limbs[0] of y (that's the byte at position y_bytes[31]) */
    uint8_t y_lsb = (uint8_t)(pt.y.limbs[0]);
    out33[0] = (y_lsb & 1) ? 0x03 : 0x02;
    metal_fe_to_be32(pt.x, out33 + 1);
}

/** Compress 64-byte uncompressed big-endian pubkey (x||y) → SEC1 33-byte. */
static void be64_to_sec1(const uint8_t in64[64], uint8_t out33[33]) {
    out33[0] = (in64[63] & 1) ? 0x03 : 0x02;
    std::memcpy(out33 + 1, in64, 32);
}

/** Decompress SEC1 33-byte pubkey to 64-byte uncompressed (x||y) big-endian,
 *  for the ecdsa_verify_batch Metal kernel which wants N×64 uncompressed. */
static bool sec1_33_to_be64(const uint8_t pub33[33], uint8_t out64[64]) {
    uint8_t prefix = pub33[0];
    if (prefix != 0x02 && prefix != 0x03) return false;

    secp256k1::fast::FieldElement fe_x;
    if (!secp256k1::fast::FieldElement::parse_bytes_strict(pub33 + 1, fe_x))
        return false;

    auto x2 = fe_x * fe_x;
    auto x3 = x2 * fe_x;
    auto y2 = x3 + secp256k1::fast::FieldElement::from_uint64(7);
    auto fe_y = y2.sqrt();

    auto yb = fe_y.to_bytes();
    if ((yb[31] & 1) != (prefix & 1))
        fe_y = fe_y.negate();

    auto xb = fe_x.to_bytes();
    std::memcpy(out64,      xb.data(), 32);
    auto yb2 = fe_y.to_bytes();
    std::memcpy(out64 + 32, yb2.data(), 32);
    return true;
}

/** Decompress SEC1 33-byte pubkey → MetalAffinePoint struct
 *  (for scalar_mul_batch which takes AffinePoint* bases). */
static bool sec1_33_to_metal_affine(const uint8_t pub33[33], MetalAffinePoint& out) {
    uint8_t prefix = pub33[0];
    if (prefix != 0x02 && prefix != 0x03) return false;

    secp256k1::fast::FieldElement fe_x;
    if (!secp256k1::fast::FieldElement::parse_bytes_strict(pub33 + 1, fe_x))
        return false;

    auto x2 = fe_x * fe_x;
    auto x3 = x2 * fe_x;
    auto y2 = x3 + secp256k1::fast::FieldElement::from_uint64(7);
    auto fe_y = y2.sqrt();

    auto yb = fe_y.to_bytes();
    if ((yb[31] & 1) != (prefix & 1))
        fe_y = fe_y.negate();

    /* Pack into MetalAffinePoint limbs */
    const auto& xl = fe_x.limbs();
    const auto& yl = fe_y.limbs();
    /* CPU limbs are 4×uint64 (LE): xl[0] is LSW, xl[3] is MSW.
       Metal limbs are 8×uint32 (LE): mt_limbs[0]=LSW, mt_limbs[7]=MSW.
       Split each uint64 into 2 uint32 (lo, hi):
         mt_limbs[2*k]   = (uint32_t)xl[k]       (low 32 bits)
         mt_limbs[2*k+1] = (uint32_t)(xl[k]>>32) (high 32 bits)                */
    for (int k = 0; k < 4; k++) {
        out.x.limbs[2*k]   = (uint32_t)xl[k];
        out.x.limbs[2*k+1] = (uint32_t)(xl[k] >> 32);
        out.y.limbs[2*k]   = (uint32_t)yl[k];
        out.y.limbs[2*k+1] = (uint32_t)(yl[k] >> 32);
    }
    return true;
}

/** Concatenate Metal shader sources into a single string for runtime
 *  compilation.  Tries a list of candidate directories. */
static std::string metal_load_combined_source(const std::vector<std::string>& shader_dirs) {
    static const char* kHeaders[] = {
        "secp256k1_field.h",
        "secp256k1_point.h",
        "secp256k1_bloom.h",
        "secp256k1_extended.h",
        nullptr
    };
    static const char* kKernels[] = {
        "secp256k1_kernels.metal",
        nullptr
    };

    for (const auto& dir : shader_dirs) {
        std::string combined;
        bool ok = true;

        for (int i = 0; kHeaders[i]; i++) {
            std::string src = metal_load_file(dir + "/" + kHeaders[i]);
            if (src.empty()) { ok = false; break; }
            combined += src; combined += "\n";
        }
        if (!ok) continue;

        for (int i = 0; kKernels[i]; i++) {
            std::string src = metal_load_file(dir + "/" + kKernels[i]);
            if (src.empty()) { ok = false; break; }
            combined += src; combined += "\n";
        }
        if (!ok) continue;

        return combined;
    }
    return {};
}

} // anonymous namespace

// =============================================================================
// MetalBackend
// =============================================================================

class MetalBackend final : public GpuBackend {
public:
    MetalBackend() = default;
    ~MetalBackend() override { shutdown(); }

    /* -- Backend identity -------------------------------------------------- */
    uint32_t backend_id() const override { return 3; /* Metal */ }
    const char* backend_name() const override { return "Metal"; }

    /* -- Device enumeration ------------------------------------------------ */
    uint32_t device_count() const override {
#if defined(__APPLE__)
        return 1;
#else
        return 0;
#endif
    }

    GpuError device_info(uint32_t device_index, DeviceInfo& out) const override {
#if defined(__APPLE__)
        if (device_index != 0)
            return GpuError::Device;

        secp256k1::metal::MetalRuntime tmp;
        if (!tmp.init(0))
            return GpuError::Device;

        auto info = tmp.device_info();
        std::memset(&out, 0, sizeof(out));
        std::snprintf(out.name, sizeof(out.name), "%s", info.name.c_str());
        out.global_mem_bytes      = info.recommended_working_set;
        out.compute_units         = 0;
        out.max_clock_mhz         = 0;
        out.max_threads_per_block = info.max_threads_per_threadgroup;
        out.backend_id            = 3;
        out.device_index          = 0;
        return GpuError::Ok;
#else
        (void)device_index; (void)out;
        return GpuError::Unavailable;
#endif
    }

    /* -- Context lifecycle ------------------------------------------------- */
    GpuError init(uint32_t device_index) override {
#if defined(__APPLE__)
        if (device_index >= device_count())
            return set_error(GpuError::Device, "Metal device index out of range");

        if (runtime_) return GpuError::Ok;

        runtime_ = std::make_unique<secp256k1::metal::MetalRuntime>();
        if (!runtime_->init(static_cast<int>(device_index))) {
            runtime_.reset();
            return set_error(GpuError::Device, "Metal device init failed");
        }
        clear_error();
        return GpuError::Ok;
#else
        (void)device_index;
        return set_error(GpuError::Unavailable, "Metal not available on this platform");
#endif
    }

    void shutdown() override {
        runtime_.reset();
        lib_ready_          = false;
        lib_init_attempted_ = false;
    }

    bool is_ready() const override { return runtime_ != nullptr; }

    /* -- Error tracking ---------------------------------------------------- */
    GpuError last_error() const override { return last_err_; }
    const char* last_error_msg() const override { return last_msg_; }

    // =========================================================================
    // Batch operations
    // =========================================================================

    GpuError generator_mul_batch(
        const uint8_t* scalars32, size_t count,
        uint8_t* out_pubkeys33) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!scalars32 || !out_pubkeys33) return set_error(GpuError::NullArg, "NULL buffer");

        auto err = ensure_library();
        if (err != GpuError::Ok) return err;

        /* Input: N × MetalScalar256 */
        auto& scratch = g_metal_batch_scratch;
        scratch.ensure_scalars(count);
        auto* const h_scalars = scratch.scalars.data();
        for (size_t i = 0; i < count; ++i)
            h_scalars[i] = be32_to_metal_scalar(scalars32 + i * 32);

        auto buf_scalars = runtime_->alloc_buffer_shared(count * sizeof(MetalScalar256));
        std::memcpy(buf_scalars.contents(), h_scalars, count * sizeof(MetalScalar256));

        /* Output: N × MetalAffinePoint */
        auto buf_results = runtime_->alloc_buffer_shared(count * sizeof(MetalAffinePoint));

        uint32_t n32 = (uint32_t)count;
        auto buf_count = runtime_->alloc_buffer_shared(sizeof(uint32_t));
        std::memcpy(buf_count.contents(), &n32, sizeof(n32));

        auto pipe = runtime_->make_pipeline("generator_mul_batch");
        runtime_->dispatch_sync(pipe, (uint32_t)count, 64u,
                                {&buf_scalars, &buf_results, &buf_count});

        const auto* aff = static_cast<const MetalAffinePoint*>(buf_results.contents());
        for (size_t i = 0; i < count; ++i)
            metal_affine_to_sec1(aff[i], out_pubkeys33 + i * 33);

        clear_error();
        return GpuError::Ok;
    }

    GpuError ecdsa_verify_batch(
        const uint8_t* msg_hashes32, const uint8_t* pubkeys33,
        const uint8_t* sigs64, size_t count,
        uint8_t* out_results) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!msg_hashes32 || !pubkeys33 || !sigs64 || !out_results)
            return set_error(GpuError::NullArg, "NULL buffer");

        auto err = ensure_library();
        if (err != GpuError::Ok) return err;

        /* Decompress SEC1 pubkeys → 64-byte uncompressed (x||y) */
        auto& scratch = g_metal_batch_scratch;
        uint8_t* const h_pubs = scratch.ensure_pubkeys64(count);
        for (size_t i = 0; i < count; ++i) {
            if (!sec1_33_to_be64(pubkeys33 + i * 33, h_pubs + i * 64))
                return set_error(GpuError::BadKey, "invalid pubkey");
        }

        auto buf_msgs = runtime_->alloc_buffer_shared(count * 32);
        std::memcpy(buf_msgs.contents(), msg_hashes32, count * 32);

        auto buf_pubs = runtime_->alloc_buffer_shared(count * 64);
        std::memcpy(buf_pubs.contents(), h_pubs, count * 64);

        auto buf_sigs = runtime_->alloc_buffer_shared(count * 64);
        std::memcpy(buf_sigs.contents(), sigs64, count * 64);

        auto buf_res = runtime_->alloc_buffer_shared(count * sizeof(uint32_t));

        uint32_t n32 = (uint32_t)count;
        auto buf_count = runtime_->alloc_buffer_shared(sizeof(uint32_t));
        std::memcpy(buf_count.contents(), &n32, sizeof(n32));

        auto pipe = runtime_->make_pipeline("ecdsa_verify_batch");
        runtime_->dispatch_sync(pipe, (uint32_t)count, 64u,
                                {&buf_msgs, &buf_pubs, &buf_sigs, &buf_res, &buf_count});

        const auto* res = static_cast<const uint32_t*>(buf_res.contents());
        for (size_t i = 0; i < count; ++i)
            out_results[i] = res[i] ? 1 : 0;

        clear_error();
        return GpuError::Ok;
    }

    GpuError schnorr_verify_batch(
        const uint8_t* msg_hashes32, const uint8_t* pubkeys_x32,
        const uint8_t* sigs64, size_t count,
        uint8_t* out_results) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!msg_hashes32 || !pubkeys_x32 || !sigs64 || !out_results)
            return set_error(GpuError::NullArg, "NULL buffer");

        auto err = ensure_library();
        if (err != GpuError::Ok) return err;

        /* schnorr_verify_batch: pubkeys_x (N×32), msgs (N×32), sigs (N×64) */
        auto buf_pks  = runtime_->alloc_buffer_shared(count * 32);
        std::memcpy(buf_pks.contents(), pubkeys_x32, count * 32);

        auto buf_msgs = runtime_->alloc_buffer_shared(count * 32);
        std::memcpy(buf_msgs.contents(), msg_hashes32, count * 32);

        auto buf_sigs = runtime_->alloc_buffer_shared(count * 64);
        std::memcpy(buf_sigs.contents(), sigs64, count * 64);

        auto buf_res = runtime_->alloc_buffer_shared(count * sizeof(uint32_t));

        uint32_t n32 = (uint32_t)count;
        auto buf_count = runtime_->alloc_buffer_shared(sizeof(uint32_t));
        std::memcpy(buf_count.contents(), &n32, sizeof(n32));

        auto pipe = runtime_->make_pipeline("schnorr_verify_batch");
        runtime_->dispatch_sync(pipe, (uint32_t)count, 64u,
                                {&buf_pks, &buf_msgs, &buf_sigs, &buf_res, &buf_count});

        const auto* res = static_cast<const uint32_t*>(buf_res.contents());
        for (size_t i = 0; i < count; ++i)
            out_results[i] = res[i] ? 1 : 0;

        clear_error();
        return GpuError::Ok;
    }

    GpuError ecdh_batch(
        const uint8_t* privkeys32, const uint8_t* peer_pubkeys33,
        size_t count, uint8_t* out_secrets32) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!privkeys32 || !peer_pubkeys33 || !out_secrets32)
            return set_error(GpuError::NullArg, "NULL buffer");

        auto err = ensure_library();
        if (err != GpuError::Ok) return err;

        /* Use scalar_mul_batch(peers, privkeys) → AffinePoint results,
           then compress and SHA256 on host to match CUDA/OpenCL semantics. */
        auto& scratch = g_metal_batch_scratch;
        scratch.ensure_affine_points(count);
        scratch.ensure_scalars(count);
        auto* const h_bases = scratch.affine_points.data();
        for (size_t i = 0; i < count; ++i) {
            if (!sec1_33_to_metal_affine(peer_pubkeys33 + i * 33, h_bases[i]))
                return set_error(GpuError::BadKey, "invalid peer pubkey");
        }

        auto* const h_scalars = scratch.scalars.data();
        for (size_t i = 0; i < count; ++i)
            h_scalars[i] = be32_to_metal_scalar(privkeys32 + i * 32);

        auto buf_bases   = runtime_->alloc_buffer_shared(count * sizeof(MetalAffinePoint));
        std::memcpy(buf_bases.contents(), h_bases, count * sizeof(MetalAffinePoint));

        auto buf_scalars = runtime_->alloc_buffer_shared(count * sizeof(MetalScalar256));
        std::memcpy(buf_scalars.contents(), h_scalars, count * sizeof(MetalScalar256));

        auto buf_results = runtime_->alloc_buffer_shared(count * sizeof(MetalAffinePoint));

        uint32_t n32 = (uint32_t)count;
        auto buf_count = runtime_->alloc_buffer_shared(sizeof(uint32_t));
        std::memcpy(buf_count.contents(), &n32, sizeof(n32));

        auto pipe = runtime_->make_pipeline("scalar_mul_batch");
        runtime_->dispatch_sync(pipe, (uint32_t)count, 64u,
                                {&buf_bases, &buf_scalars, &buf_results, &buf_count});

        const auto* aff = static_cast<const MetalAffinePoint*>(buf_results.contents());
        for (size_t i = 0; i < count; ++i) {
            uint8_t compressed[33];
            metal_affine_to_sec1(aff[i], compressed);
            auto digest = secp256k1::SHA256::hash(compressed, sizeof(compressed));
            std::memcpy(out_secrets32 + i * 32, digest.data(), 32);
        }

        clear_error();
        return GpuError::Ok;
    }

    GpuError hash160_pubkey_batch(
        const uint8_t* pubkeys33, size_t count,
        uint8_t* out_hash160) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!pubkeys33 || !out_hash160)
            return set_error(GpuError::NullArg, "NULL buffer");

        auto err = ensure_library();
        if (err != GpuError::Ok) return err;

        /* hash160_batch kernel: pubkeys (stride bytes each), hashes (N×20),
           stride (constant uint), count (constant uint). */
        auto buf_pks  = runtime_->alloc_buffer_shared(count * 33);
        std::memcpy(buf_pks.contents(), pubkeys33, count * 33);

        auto buf_hash = runtime_->alloc_buffer_shared(count * 20);

        uint32_t stride = 33u;
        auto buf_stride = runtime_->alloc_buffer_shared(sizeof(uint32_t));
        std::memcpy(buf_stride.contents(), &stride, sizeof(stride));

        uint32_t n32 = (uint32_t)count;
        auto buf_count = runtime_->alloc_buffer_shared(sizeof(uint32_t));
        std::memcpy(buf_count.contents(), &n32, sizeof(n32));

        auto pipe = runtime_->make_pipeline("hash160_batch");
        runtime_->dispatch_sync(pipe, (uint32_t)count, 64u,
                                {&buf_pks, &buf_hash, &buf_stride, &buf_count});

        std::memcpy(out_hash160,
                    buf_hash.contents(),
                    count * 20);

        clear_error();
        return GpuError::Ok;
    }

    GpuError frost_verify_partial_batch(
        const uint8_t* z_i32,
        const uint8_t* D_i33,
        const uint8_t* E_i33,
        const uint8_t* Y_i33,
        const uint8_t* rho_i32,
        const uint8_t* lambda_ie32,
        const uint8_t* negate_R,
        const uint8_t* negate_key,
        size_t count,
        uint8_t* out_results) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!z_i32 || !D_i33 || !E_i33 || !Y_i33 ||
            !rho_i32 || !lambda_ie32 || !negate_R || !negate_key || !out_results)
            return set_error(GpuError::NullArg, "NULL buffer");

        auto err = ensure_library();
        if (err != GpuError::Ok) return err;

        auto buf_z   = runtime_->alloc_buffer_shared(count * 32);
        std::memcpy(buf_z.contents(), z_i32, count * 32);

        auto buf_D   = runtime_->alloc_buffer_shared(count * 33);
        std::memcpy(buf_D.contents(), D_i33, count * 33);

        auto buf_E   = runtime_->alloc_buffer_shared(count * 33);
        std::memcpy(buf_E.contents(), E_i33, count * 33);

        auto buf_Y   = runtime_->alloc_buffer_shared(count * 33);
        std::memcpy(buf_Y.contents(), Y_i33, count * 33);

        auto buf_rho = runtime_->alloc_buffer_shared(count * 32);
        std::memcpy(buf_rho.contents(), rho_i32, count * 32);

        auto buf_lam = runtime_->alloc_buffer_shared(count * 32);
        std::memcpy(buf_lam.contents(), lambda_ie32, count * 32);

        auto buf_nR  = runtime_->alloc_buffer_shared(count * 1);
        std::memcpy(buf_nR.contents(), negate_R, count * 1);

        auto buf_nK  = runtime_->alloc_buffer_shared(count * 1);
        std::memcpy(buf_nK.contents(), negate_key, count * 1);

        auto buf_res = runtime_->alloc_buffer_shared(count * sizeof(uint32_t));

        uint32_t n32 = (uint32_t)count;
        auto buf_count = runtime_->alloc_buffer_shared(sizeof(uint32_t));
        std::memcpy(buf_count.contents(), &n32, sizeof(n32));

        auto pipe = runtime_->make_pipeline("frost_verify_partial_batch");
        runtime_->dispatch_sync(pipe, (uint32_t)count, 64u,
                                {&buf_z, &buf_D, &buf_E, &buf_Y, &buf_rho,
                                 &buf_lam, &buf_nR, &buf_nK, &buf_res, &buf_count});

        const auto* res = static_cast<const uint32_t*>(buf_res.contents());
        for (size_t i = 0; i < count; ++i)
            out_results[i] = res[i] ? 1 : 0;

        clear_error();
        return GpuError::Ok;
    }

    GpuError ecrecover_batch(
        const uint8_t* msg_hashes32, const uint8_t* sigs64,
        const int* recids, size_t count,
        uint8_t* out_pubkeys33, uint8_t* out_valid) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!msg_hashes32 || !sigs64 || !recids || !out_pubkeys33 || !out_valid)
            return set_error(GpuError::NullArg, "NULL buffer");

        auto err = ensure_library();
        if (err != GpuError::Ok) return err;

        auto buf_msgs = runtime_->alloc_buffer_shared(count * 32);
        std::memcpy(buf_msgs.contents(), msg_hashes32, count * 32);

        auto buf_sigs = runtime_->alloc_buffer_shared(count * 64);
        std::memcpy(buf_sigs.contents(), sigs64, count * 64);

        std::vector<uint32_t> h_recids(count);
        for (size_t i = 0; i < count; ++i)
            h_recids[i] = static_cast<uint32_t>(recids[i]);
        auto buf_recids = runtime_->alloc_buffer_shared(count * sizeof(uint32_t));
        std::memcpy(buf_recids.contents(), h_recids.data(), count * sizeof(uint32_t));

        auto buf_pubs = runtime_->alloc_buffer_shared(count * 64);
        auto buf_valid = runtime_->alloc_buffer_shared(count * sizeof(uint32_t));

        uint32_t n32 = (uint32_t)count;
        auto buf_count = runtime_->alloc_buffer_shared(sizeof(uint32_t));
        std::memcpy(buf_count.contents(), &n32, sizeof(n32));

        auto pipe = runtime_->make_pipeline("ecrecover_batch");
        runtime_->dispatch_sync(pipe, (uint32_t)count, 64u,
                                {&buf_msgs, &buf_sigs, &buf_recids, &buf_pubs,
                                 &buf_valid, &buf_count});

        const auto* pubs = static_cast<const uint8_t*>(buf_pubs.contents());
        const auto* valid = static_cast<const uint32_t*>(buf_valid.contents());
        for (size_t i = 0; i < count; ++i) {
            out_valid[i] = valid[i] ? 1 : 0;
            if (valid[i]) {
                be64_to_sec1(pubs + i * 64, out_pubkeys33 + i * 33);
            } else {
                std::memset(out_pubkeys33 + i * 33, 0, 33);
            }
        }

        clear_error();
        return GpuError::Ok;
    }

    GpuError msm(
        const uint8_t* scalars32, const uint8_t* points33,
        size_t n, uint8_t* out_result33) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        if (n == 0) { clear_error(); return GpuError::Ok; }
        if (!scalars32 || !points33 || !out_result33)
            return set_error(GpuError::NullArg, "NULL buffer");

        auto err = ensure_library();
        if (err != GpuError::Ok) return err;

        /* Decompress SEC1 points → MetalAffinePoint */
        std::vector<MetalAffinePoint> h_bases(n);
        for (size_t i = 0; i < n; ++i) {
            if (!sec1_33_to_metal_affine(points33 + i * 33, h_bases[i]))
                return set_error(GpuError::BadKey, "invalid MSM point");
        }

        /* Scalars */
        std::vector<MetalScalar256> h_scalars(n);
        for (size_t i = 0; i < n; ++i)
            h_scalars[i] = be32_to_metal_scalar(scalars32 + i * 32);

        auto buf_bases   = runtime_->alloc_buffer_shared(n * sizeof(MetalAffinePoint));
        std::memcpy(buf_bases.contents(), h_bases.data(), n * sizeof(MetalAffinePoint));

        auto buf_scalars = runtime_->alloc_buffer_shared(n * sizeof(MetalScalar256));
        std::memcpy(buf_scalars.contents(), h_scalars.data(), n * sizeof(MetalScalar256));

        auto buf_results = runtime_->alloc_buffer_shared(n * sizeof(MetalAffinePoint));

        uint32_t n32 = (uint32_t)n;
        auto buf_count = runtime_->alloc_buffer_shared(sizeof(uint32_t));
        std::memcpy(buf_count.contents(), &n32, sizeof(n32));

        /* GPU: scalar_mul_batch(P[i], k[i]) → AffinePoint[i] */
        auto pipe = runtime_->make_pipeline("scalar_mul_batch");
        runtime_->dispatch_sync(pipe, (uint32_t)n, 64u,
                                {&buf_bases, &buf_scalars, &buf_results, &buf_count});

        /* CPU: accumulate affine points */
        const auto* aff = static_cast<const MetalAffinePoint*>(buf_results.contents());

        bool have_acc = false;
        secp256k1::fast::FieldElement acc_x, acc_y;

        for (size_t i = 0; i < n; ++i) {
            uint8_t xb[32], yb[32];
            metal_fe_to_be32(aff[i].x, xb);
            metal_fe_to_be32(aff[i].y, yb);

            /* Build CPU FieldElement from limb rep */
            const auto& xl_mt = aff[i].x.limbs;
            const auto& yl_mt = aff[i].y.limbs;
            /* 8×uint32 → 4×uint64: xl64[k] = xl_mt[2k+1]<<32 | xl_mt[2k] */
            std::array<uint64_t,4> xl64, yl64;
            for (int k = 0; k < 4; k++) {
                xl64[k] = ((uint64_t)xl_mt[2*k+1] << 32) | (uint64_t)xl_mt[2*k];
                yl64[k] = ((uint64_t)yl_mt[2*k+1] << 32) | (uint64_t)yl_mt[2*k];
            }
            auto px = secp256k1::fast::FieldElement::from_limbs(xl64);
            auto py = secp256k1::fast::FieldElement::from_limbs(yl64);

            /* Skip point at infinity */
            auto pxb = px.to_bytes(); auto pyb = py.to_bytes();
            bool is_zero = true;
            for (int k = 0; k < 32 && is_zero; ++k)
                if (pxb[k] || pyb[k]) is_zero = false;
            if (is_zero) continue;

            if (!have_acc) {
                acc_x = px; acc_y = py;
                have_acc = true;
                continue;
            }

            auto dx = px - acc_x;
            auto dy = py - acc_y;
            auto dxb = dx.to_bytes();
            bool dx_zero = true;
            for (int k = 0; k < 32 && dx_zero; ++k) if (dxb[k]) dx_zero = false;

            if (dx_zero) {
                auto dyb = dy.to_bytes();
                bool dy_zero = true;
                for (int k = 0; k < 32 && dy_zero; ++k) if (dyb[k]) dy_zero = false;
                if (!dy_zero) { have_acc = false; continue; }
                /* Doubling */
                auto x2  = acc_x * acc_x;
                auto num = x2 + x2 + x2;
                auto den = acc_y + acc_y;
                auto lam = num * den.inverse();
                auto rx  = lam * lam - acc_x - acc_x;
                auto ry  = lam * (acc_x - rx) - acc_y;
                acc_x = rx; acc_y = ry;
            } else {
                auto lam = dy * dx.inverse();
                auto rx  = lam * lam - acc_x - px;
                auto ry  = lam * (acc_x - rx) - acc_y;
                acc_x = rx; acc_y = ry;
            }
        }

        if (!have_acc)
            return set_error(GpuError::Arith, "MSM result is point at infinity");

        auto yb = acc_y.to_bytes();
        out_result33[0] = (yb[31] & 1) ? 0x03 : 0x02;
        auto xb = acc_x.to_bytes();
        std::memcpy(out_result33 + 1, xb.data(), 32);

        clear_error();
        return GpuError::Ok;
    }

    /* -- ZK / BIP-324 batch operations (Metal via secp256k1_kernels.metal) -- */

    GpuError zk_knowledge_verify_batch(
        const uint8_t* proofs64, const uint8_t* pubkeys65,
        const uint8_t* messages32, size_t count,
        uint8_t* out_results) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!proofs64 || !pubkeys65 || !messages32 || !out_results)
            return set_error(GpuError::NullArg, "NULL buffer");

        auto err = ensure_library();
        if (err != GpuError::Ok) return err;

        /* Split proof_rx (32 B) and proof_s (32 B) from interleaved 64-byte proofs */
        std::vector<uint8_t> h_rx(count * 32), h_s(count * 32);
        for (size_t i = 0; i < count; ++i) {
            std::memcpy(h_rx.data() + i * 32, proofs64 + i * 64,      32);
            std::memcpy(h_s.data()  + i * 32, proofs64 + i * 64 + 32, 32);
        }

        /* Extract x-coordinates from 65-byte uncompressed pubkeys (04 || x32 || y32) */
        std::vector<uint8_t> h_pks(count * 32);
        for (size_t i = 0; i < count; ++i)
            std::memcpy(h_pks.data() + i * 32, pubkeys65 + i * 65 + 1, 32);

        auto buf_rx   = runtime_->alloc_buffer_shared(count * 32);
        std::memcpy(buf_rx.contents(), h_rx.data(), count * 32);
        auto buf_s    = runtime_->alloc_buffer_shared(count * 32);
        std::memcpy(buf_s.contents(), h_s.data(), count * 32);
        auto buf_pks  = runtime_->alloc_buffer_shared(count * 32);
        std::memcpy(buf_pks.contents(), h_pks.data(), count * 32);
        auto buf_msgs = runtime_->alloc_buffer_shared(count * 32);
        std::memcpy(buf_msgs.contents(), messages32, count * 32);
        auto buf_res  = runtime_->alloc_buffer_shared(count * sizeof(uint32_t));
        uint32_t n32  = (uint32_t)count;
        auto buf_n    = runtime_->alloc_buffer_shared(sizeof(uint32_t));
        std::memcpy(buf_n.contents(), &n32, sizeof(n32));

        auto pipe = runtime_->make_pipeline("zk_knowledge_verify_batch");
        runtime_->dispatch_sync(pipe, (uint32_t)count, 64u,
                                {&buf_rx, &buf_s, &buf_pks, &buf_msgs, &buf_res, &buf_n});

        const auto* res = static_cast<const uint32_t*>(buf_res.contents());
        for (size_t i = 0; i < count; ++i)
            out_results[i] = res[i] ? 1 : 0;

        clear_error();
        return GpuError::Ok;
    }

    GpuError zk_dleq_verify_batch(
        const uint8_t* proofs64,
        const uint8_t* G_pts65, const uint8_t* H_pts65,
        const uint8_t* P_pts65, const uint8_t* Q_pts65,
        size_t count, uint8_t* out_results) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!proofs64 || !G_pts65 || !H_pts65 || !P_pts65 || !Q_pts65 || !out_results)
            return set_error(GpuError::NullArg, "NULL buffer");

        auto err = ensure_library();
        if (err != GpuError::Ok) return err;

        /* Split e[32] || s[32] from proof */
        std::vector<uint8_t> h_e(count * 32), h_s(count * 32);
        for (size_t i = 0; i < count; ++i) {
            std::memcpy(h_e.data() + i * 32, proofs64 + i * 64,      32);
            std::memcpy(h_s.data() + i * 32, proofs64 + i * 64 + 32, 32);
        }

        /* Metal kernel uses hardcoded G and tag-derived H; pass P and Q as x-coords */
        /* G_pts65 and H_pts65 are intentionally unused by this kernel path. */
        (void)G_pts65; (void)H_pts65;
        std::vector<uint8_t> h_P(count * 32), h_Q(count * 32);
        for (size_t i = 0; i < count; ++i) {
            std::memcpy(h_P.data() + i * 32, P_pts65 + i * 65 + 1, 32);
            std::memcpy(h_Q.data() + i * 32, Q_pts65 + i * 65 + 1, 32);
        }

        auto buf_e   = runtime_->alloc_buffer_shared(count * 32);
        std::memcpy(buf_e.contents(), h_e.data(), count * 32);
        auto buf_s   = runtime_->alloc_buffer_shared(count * 32);
        std::memcpy(buf_s.contents(), h_s.data(), count * 32);
        auto buf_P   = runtime_->alloc_buffer_shared(count * 32);
        std::memcpy(buf_P.contents(), h_P.data(), count * 32);
        auto buf_Q   = runtime_->alloc_buffer_shared(count * 32);
        std::memcpy(buf_Q.contents(), h_Q.data(), count * 32);
        auto buf_res = runtime_->alloc_buffer_shared(count * sizeof(uint32_t));
        uint32_t n32 = (uint32_t)count;
        auto buf_n   = runtime_->alloc_buffer_shared(sizeof(uint32_t));
        std::memcpy(buf_n.contents(), &n32, sizeof(n32));

        auto pipe = runtime_->make_pipeline("zk_dleq_verify_batch");
        runtime_->dispatch_sync(pipe, (uint32_t)count, 64u,
                                {&buf_e, &buf_s, &buf_P, &buf_Q, &buf_res, &buf_n});

        const auto* res = static_cast<const uint32_t*>(buf_res.contents());
        for (size_t i = 0; i < count; ++i)
            out_results[i] = res[i] ? 1 : 0;

        clear_error();
        return GpuError::Ok;
    }

    GpuError bulletproof_verify_batch(
        const uint8_t* proofs324, const uint8_t* commitments65,
        const uint8_t* H_generator65, size_t count,
        uint8_t* out_results) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!proofs324 || !commitments65 || !H_generator65 || !out_results)
            return set_error(GpuError::NullArg, "NULL buffer");

        auto err = ensure_library();
        if (err != GpuError::Ok) return err;

        /* Convert big-endian 32 bytes → MetalFieldElem (same format as scalar). */
        auto be32_to_metal_fe = [](const uint8_t be[32]) -> MetalFieldElem {
            MetalFieldElem fe;
            for (int i = 0; i < 8; i++) {
                int base = (7 - i) * 4;
                fe.limbs[i] = ((uint32_t)be[base]   << 24) |
                              ((uint32_t)be[base+1]  << 16) |
                              ((uint32_t)be[base+2]  << 8)  |
                              ((uint32_t)be[base+3]);
            }
            return fe;
        };

        /* Parse uncompressed point (65 bytes: 04 || x[32] || y[32]) → MetalAffinePoint */
        auto parse_pt65 = [&be32_to_metal_fe](const uint8_t pt65[65]) -> MetalAffinePoint {
            return { be32_to_metal_fe(pt65 + 1), be32_to_metal_fe(pt65 + 33) };
        };

        /* Build GPU-layout RangeProofPolyGPU structs (320 bytes each):
         *   4 x MetalAffinePoint (A, S, T1, T2) + 2 x MetalScalar256 (tau_x, t_hat)
         * Wire format per proof (324 bytes): 4 x 65-byte uncompressed + 2 x 32-byte scalars */
        struct RangeProofPolyMetal {
            MetalAffinePoint A, S, T1, T2;
            MetalScalar256 tau_x, t_hat;
        };
        static_assert(sizeof(RangeProofPolyMetal) == 320, "struct layout mismatch");

        auto buf_proofs = runtime_->alloc_buffer_shared(count * sizeof(RangeProofPolyMetal));
        auto* proofs_out = static_cast<RangeProofPolyMetal*>(buf_proofs.contents());
        for (size_t i = 0; i < count; ++i) {
            const uint8_t* p = proofs324 + i * 324;
            proofs_out[i].A    = parse_pt65(p);
            proofs_out[i].S    = parse_pt65(p + 65);
            proofs_out[i].T1   = parse_pt65(p + 130);
            proofs_out[i].T2   = parse_pt65(p + 195);
            proofs_out[i].tau_x = be32_to_metal_scalar(p + 260);
            proofs_out[i].t_hat = be32_to_metal_scalar(p + 292);
        }

        auto buf_commits = runtime_->alloc_buffer_shared(count * sizeof(MetalAffinePoint));
        auto* commits_out = static_cast<MetalAffinePoint*>(buf_commits.contents());
        for (size_t i = 0; i < count; ++i)
            commits_out[i] = parse_pt65(commitments65 + i * 65);

        auto buf_hgen = runtime_->alloc_buffer_shared(sizeof(MetalAffinePoint));
        *static_cast<MetalAffinePoint*>(buf_hgen.contents()) = parse_pt65(H_generator65);

        auto buf_res = runtime_->alloc_buffer_shared(count * sizeof(uint32_t));
        uint32_t n32 = (uint32_t)count;
        auto buf_n   = runtime_->alloc_buffer_shared(sizeof(uint32_t));
        std::memcpy(buf_n.contents(), &n32, sizeof(n32));

        auto pipe = runtime_->make_pipeline("range_proof_poly_batch");
        runtime_->dispatch_sync(pipe, (uint32_t)count, 64u,
                                {&buf_proofs, &buf_commits, &buf_hgen, &buf_res, &buf_n});

        const auto* res = static_cast<const uint32_t*>(buf_res.contents());
        for (size_t i = 0; i < count; ++i)
            out_results[i] = res[i] ? 1 : 0;

        clear_error();
        return GpuError::Ok;
    }

    GpuError bip324_aead_encrypt_batch(
        const uint8_t* keys32, const uint8_t* nonces12,
        const uint8_t* plaintexts, const uint32_t* sizes,
        uint32_t max_payload, size_t count, uint8_t* wire_out) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!keys32 || !nonces12 || !plaintexts || !sizes || !wire_out)
            return set_error(GpuError::NullArg, "NULL buffer");

        auto err = ensure_library();
        if (err != GpuError::Ok) return err;

        const size_t wire_stride = (size_t)max_payload + 19u; /* BIP324_OVERHEAD = 3 hdr + 16 tag */

        auto buf_keys   = runtime_->alloc_buffer_shared(count * 32);
        std::memcpy(buf_keys.contents(), keys32, count * 32);
        auto buf_nonces = runtime_->alloc_buffer_shared(count * 12);
        std::memcpy(buf_nonces.contents(), nonces12, count * 12);
        auto buf_pt     = runtime_->alloc_buffer_shared((size_t)max_payload * count);
        std::memcpy(buf_pt.contents(), plaintexts, (size_t)max_payload * count);
        auto buf_sizes  = runtime_->alloc_buffer_shared(sizeof(uint32_t) * count);
        std::memcpy(buf_sizes.contents(), sizes, sizeof(uint32_t) * count);
        auto buf_wire   = runtime_->alloc_buffer_shared(wire_stride * count);

        auto buf_max    = runtime_->alloc_buffer_shared(sizeof(uint32_t));
        std::memcpy(buf_max.contents(), &max_payload, sizeof(max_payload));
        uint32_t n32    = (uint32_t)count;
        auto buf_n      = runtime_->alloc_buffer_shared(sizeof(uint32_t));
        std::memcpy(buf_n.contents(), &n32, sizeof(n32));

        auto pipe = runtime_->make_pipeline("kernel_bip324_aead_encrypt");
        runtime_->dispatch_sync(pipe, (uint32_t)count, 64u,
                                {&buf_keys, &buf_nonces, &buf_pt, &buf_sizes,
                                 &buf_wire, &buf_max, &buf_n});

        std::memcpy(wire_out, buf_wire.contents(), wire_stride * count);
        clear_error();
        return GpuError::Ok;
    }

    GpuError bip324_aead_decrypt_batch(
        const uint8_t* keys32, const uint8_t* nonces12,
        const uint8_t* wire_in, const uint32_t* sizes,
        uint32_t max_payload, size_t count,
        uint8_t* plaintext_out, uint8_t* out_valid) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!keys32 || !nonces12 || !wire_in || !sizes || !plaintext_out || !out_valid)
            return set_error(GpuError::NullArg, "NULL buffer");

        auto err = ensure_library();
        if (err != GpuError::Ok) return err;

        const size_t wire_stride = (size_t)max_payload + 19u;

        auto buf_keys    = runtime_->alloc_buffer_shared(count * 32);
        std::memcpy(buf_keys.contents(), keys32, count * 32);
        auto buf_nonces  = runtime_->alloc_buffer_shared(count * 12);
        std::memcpy(buf_nonces.contents(), nonces12, count * 12);
        auto buf_wire_in = runtime_->alloc_buffer_shared(wire_stride * count);
        std::memcpy(buf_wire_in.contents(), wire_in, wire_stride * count);
        auto buf_sizes   = runtime_->alloc_buffer_shared(sizeof(uint32_t) * count);
        std::memcpy(buf_sizes.contents(), sizes, sizeof(uint32_t) * count);
        auto buf_pt_out  = runtime_->alloc_buffer_shared((size_t)max_payload * count);
        auto buf_ok      = runtime_->alloc_buffer_shared(sizeof(uint32_t) * count);

        auto buf_max     = runtime_->alloc_buffer_shared(sizeof(uint32_t));
        std::memcpy(buf_max.contents(), &max_payload, sizeof(max_payload));
        uint32_t n32     = (uint32_t)count;
        auto buf_n       = runtime_->alloc_buffer_shared(sizeof(uint32_t));
        std::memcpy(buf_n.contents(), &n32, sizeof(n32));

        auto pipe = runtime_->make_pipeline("kernel_bip324_aead_decrypt");
        runtime_->dispatch_sync(pipe, (uint32_t)count, 64u,
                                {&buf_keys, &buf_nonces, &buf_wire_in, &buf_sizes,
                                 &buf_pt_out, &buf_ok, &buf_max, &buf_n});

        std::memcpy(plaintext_out, buf_pt_out.contents(), (size_t)max_payload * count);
        const auto* ok_vals = static_cast<const uint32_t*>(buf_ok.contents());
        for (size_t i = 0; i < count; ++i)
            out_valid[i] = ok_vals[i] ? 1 : 0;

        clear_error();
        return GpuError::Ok;
    }

private:
    std::unique_ptr<secp256k1::metal::MetalRuntime> runtime_;
    bool lib_init_attempted_ = false;
    bool lib_ready_          = false;
    GpuError last_err_ = GpuError::Ok;
    char     last_msg_[256] = {};

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

    /* -- Lazy library loading ---------------------------------------------- */
    GpuError ensure_library() {
        if (lib_ready_) return GpuError::Ok;
        if (lib_init_attempted_)
            return set_error(GpuError::Launch, "Metal library load previously failed");
        lib_init_attempted_ = true;

        /* Try compiled metallib paths first */
        const char* metallib_paths[] = {
            "secp256k1_kernels.metallib",
            "./secp256k1_kernels.metallib",
            "../secp256k1_kernels.metallib",
            "../../secp256k1_kernels.metallib",
            "../metal/secp256k1_kernels.metallib",
            "../../metal/secp256k1_kernels.metallib",
            "../../../metal/secp256k1_kernels.metallib",
            nullptr
        };

        for (int i = 0; metallib_paths[i]; i++) {
            if (runtime_->load_library_from_path(metallib_paths[i])) {
                lib_ready_ = true;
                clear_error();
                return GpuError::Ok;
            }
        }

        /* Fallback: compile shader source at runtime */
        const std::vector<std::string> shader_dirs = {
            "shaders",
            "../shaders",
            "../../shaders",
            "../metal/shaders",
            "../../metal/shaders",
            "../../../metal/shaders",
        };

        std::string source = metal_load_combined_source(shader_dirs);
        if (source.empty())
            return set_error(GpuError::Launch,
                             "Metal: could not find metallib or shader sources");

        if (!runtime_->load_library_from_source(source))
            return set_error(GpuError::Launch,
                             "Metal: runtime shader compilation failed");

        lib_ready_ = true;
        clear_error();
        return GpuError::Ok;
    }
};

/* -- Factory --------------------------------------------------------------- */
std::unique_ptr<GpuBackend> create_metal_backend() {
    return std::make_unique<MetalBackend>();
}

} // namespace gpu
} // namespace secp256k1
