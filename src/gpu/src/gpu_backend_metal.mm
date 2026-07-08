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

#include <cstdlib>
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
#include "secp256k1/scalar.hpp"

/* -- CPU SHA-256 for ECDH finalization ------------------------------------- */
#include "secp256k1/sha256.hpp"

/* -- CPU Hash160 fallback -------------------------------------------------- */
#include "secp256k1/hash_accel.hpp"

/* -- Secure memory erasure ------------------------------------------------- */
#include "secp256k1/detail/secure_erase.hpp"

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

struct MetalScalarEraseGuard {
    MetalScalar256* ptr = nullptr;
    std::size_t count = 0;
    ~MetalScalarEraseGuard() {
        if (ptr && count) {
            secp256k1::detail::secure_erase(ptr, count * sizeof(MetalScalar256));
        }
    }
};

struct MetalBufferEraseGuard {
    secp256k1::metal::MetalBuffer* buffer = nullptr;
    ~MetalBufferEraseGuard() {
        if (buffer && buffer->valid()) {
            secp256k1::detail::secure_erase(buffer->contents(), buffer->length());
        }
    }
};

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

/* Jacobian point on the host side, matching Metal shader's JacobianPoint layout:
 * x(8×uint32) + y(8×uint32) + z(8×uint32) + infinity(uint32) = 100 bytes. */
struct MetalJacobianPoint {
    MetalFieldElem x;
    MetalFieldElem y;
    MetalFieldElem z;
    uint32_t       infinity;
};

/* MSM persistent buffer pool — grow-only, avoids per-call alloc_buffer_shared overhead.
 * Stores input bases, scalars, partial scalar-mul results, and Jacobian block sums. */
struct MetalMsmPool {
    secp256k1::metal::MetalBuffer buf_bases;    // n × MetalAffinePoint
    secp256k1::metal::MetalBuffer buf_scalars;  // n × MetalScalar256
    secp256k1::metal::MetalBuffer buf_partials; // n × MetalAffinePoint  (scalar_mul output)
    secp256k1::metal::MetalBuffer buf_blocks;   // ceil(n/256) × MetalJacobianPoint
    size_t capacity = 0;

    void ensure(size_t n, secp256k1::metal::MetalRuntime* rt) {
        if (n <= capacity) return;
        free_all();
        const size_t n_blocks = (n + 255) / 256;
        buf_bases    = rt->alloc_buffer_shared(n        * sizeof(MetalAffinePoint));
        buf_scalars  = rt->alloc_buffer_shared(n        * sizeof(MetalScalar256));
        buf_partials = rt->alloc_buffer_shared(n        * sizeof(MetalAffinePoint));
        buf_blocks   = rt->alloc_buffer_shared(n_blocks * sizeof(MetalJacobianPoint));
        if (buf_bases.valid() && buf_scalars.valid() && buf_partials.valid() && buf_blocks.valid())
            capacity = n;
    }

    void free_all() {
        buf_bases    = secp256k1::metal::MetalBuffer{};
        buf_scalars  = secp256k1::metal::MetalBuffer{};
        buf_partials = secp256k1::metal::MetalBuffer{};
        buf_blocks   = secp256k1::metal::MetalBuffer{};
        capacity = 0;
    }
};

/** Convert a MetalJacobianPoint to CPU affine coordinates.
 *  Returns false if the point is at infinity or z == 0. */
static bool metal_jacobian_to_cpu_affine(
    const MetalJacobianPoint& jp,
    secp256k1::fast::FieldElement& out_x,
    secp256k1::fast::FieldElement& out_y)
{
    if (jp.infinity) return false;

    auto mt_fe_to_cpu = [](const MetalFieldElem& fe) -> secp256k1::fast::FieldElement {
        std::array<uint64_t, 4> l64;
        for (int k = 0; k < 4; k++)
            l64[k] = ((uint64_t)fe.limbs[2*k+1] << 32) | (uint64_t)fe.limbs[2*k];
        return secp256k1::fast::FieldElement::from_limbs(l64);
    };

    auto z = mt_fe_to_cpu(jp.z);
    auto zb = z.to_bytes();
    bool z_zero = true;
    for (int k = 0; k < 32 && z_zero; ++k) if (zb[k]) z_zero = false;
    if (z_zero) return false;

    auto z_inv  = z.inverse();
    auto z_inv2 = z_inv * z_inv;
    auto z_inv3 = z_inv2 * z_inv;
    out_x = mt_fe_to_cpu(jp.x) * z_inv2;
    out_y = mt_fe_to_cpu(jp.y) * z_inv3;
    return true;
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
        // Metal API does not expose SM/CU count or clock MHz directly.
        // Estimate compute_units from GPU family (Apple7=M1 8-CU, Apple8=M2 8-10CU,
        // Apple9=M3 10-CU baseline). Falls back to 0 for unknown/non-Apple hardware.
        uint32_t cu_estimate = 0;
        if (info.supports_family_apple9)      cu_estimate = 10; // M3 baseline (10 GPU cores)
        else if (info.supports_family_apple8) cu_estimate = 8;  // M2 baseline (8 GPU cores)
        else if (info.supports_family_apple7) cu_estimate = 8;  // M1 baseline (8 GPU cores)
        out.compute_units         = cu_estimate;
        out.max_clock_mhz         = 0; // Metal API: clock MHz unavailable
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

        /* Pass 33-byte compressed pubkeys directly — GPU decompresses via lbtc_point_from_compressed */
        auto buf_msgs = runtime_->alloc_buffer_shared(count * 32);
        std::memcpy(buf_msgs.contents(), msg_hashes32, count * 32);

        auto buf_pubs = runtime_->alloc_buffer_shared(count * 33);
        std::memcpy(buf_pubs.contents(), pubkeys33, count * 33);

        auto buf_sigs = runtime_->alloc_buffer_shared(count * 64);
        std::memcpy(buf_sigs.contents(), sigs64, count * 64);

        auto buf_res = runtime_->alloc_buffer_shared(count * sizeof(uint32_t));

        uint32_t n32 = (uint32_t)count;
        auto buf_count = runtime_->alloc_buffer_shared(sizeof(uint32_t));
        std::memcpy(buf_count.contents(), &n32, sizeof(n32));

        auto pipe = runtime_->make_pipeline("ecdsa_verify_batch_compressed");
        runtime_->dispatch_sync(pipe, (uint32_t)count, 64u,
                                {&buf_msgs, &buf_pubs, &buf_sigs, &buf_res, &buf_count});

        const auto* res = static_cast<const uint32_t*>(buf_res.contents());
        for (size_t i = 0; i < count; ++i)
            out_results[i] = res[i] ? 1 : 0;

        clear_error();
        return GpuError::Ok;
    }

    GpuError ecdsa_verify_lbtc_rows(
        const uint8_t* rows, size_t stride, size_t count,
        uint8_t* out_results) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!rows || !out_results)
            return set_error(GpuError::NullArg, "NULL buffer");
        if (stride < 129u)
            return set_error(GpuError::BadInput, "libbitcoin row stride < 129");

        auto err = ensure_library();
        if (err != GpuError::Ok) return err;

        const size_t row_bytes = count * stride;
        auto buf_rows = runtime_->alloc_buffer_shared(row_bytes);
        std::memcpy(buf_rows.contents(), rows, row_bytes);

        auto buf_stride = runtime_->alloc_buffer_shared(sizeof(uint64_t));
        const uint64_t stride64 = static_cast<uint64_t>(stride);
        std::memcpy(buf_stride.contents(), &stride64, sizeof(stride64));

        auto buf_res = runtime_->alloc_buffer_shared(count * sizeof(uint32_t));

        uint32_t n32 = (uint32_t)count;
        auto buf_count = runtime_->alloc_buffer_shared(sizeof(uint32_t));
        std::memcpy(buf_count.contents(), &n32, sizeof(n32));

        auto pipe = runtime_->make_pipeline("ecdsa_verify_lbtc_rows");
        if (!pipe.valid())
            return set_error(GpuError::Launch,
                             "Metal: ecdsa_verify_lbtc_rows kernel missing from loaded library");
        runtime_->dispatch_sync(pipe, (uint32_t)count, 64u,
                                {&buf_rows, &buf_stride, &buf_res, &buf_count});

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

    GpuError ecdsa_verify_lbtc_columns(
        const uint8_t* digests32, const uint8_t* pubkeys33,
        const uint8_t* sigs64, size_t count,
        uint8_t* out_results) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!digests32 || !pubkeys33 || !sigs64 || !out_results)
            return set_error(GpuError::NullArg, "NULL buffer");

        // Fail-closed: initialise the whole result buffer to invalid (0) up front,
        // so any early non-OK return (library/pipeline/alloc) leaves no stale or
        // partial verdict behind. On success every row is overwritten below.
        std::memset(out_results, 0, count);

        auto err = ensure_library();
        if (err != GpuError::Ok) return err;
        auto pipe = runtime_->make_pipeline("ecdsa_verify_lbtc_columns");
        if (!pipe.valid())
            return set_error(GpuError::Launch,
                             "Metal: ecdsa_verify_lbtc_columns kernel missing from loaded library");

        // Engine-owned, memory-aware chunking: hard-cap per-chunk rows so a large
        // batch cannot force a giant one-shot device allocation. Internal only —
        // the caller never sees or manages chunking (UFSECP_GPU_COLUMNS_CHUNK is a
        // private engine/test tuning knob, not a caller-facing parameter).
        size_t cap = (static_cast<size_t>(4) << 20);
        if (const char* e = std::getenv("UFSECP_GPU_COLUMNS_CHUNK")) {
            const unsigned long long v = std::strtoull(e, nullptr, 10);
            if (v > 0 && static_cast<size_t>(v) < cap) cap = static_cast<size_t>(v);
        }
        const size_t chunk = (count < cap) ? count : cap;
        for (size_t off = 0; off < count; off += chunk) {
            const size_t n = (count - off < chunk) ? (count - off) : chunk;
            auto buf_dig = runtime_->alloc_buffer_shared(n * 32);
            auto buf_pub = runtime_->alloc_buffer_shared(n * 33);
            auto buf_sig = runtime_->alloc_buffer_shared(n * 64);
            auto buf_res = runtime_->alloc_buffer_shared(n * sizeof(uint32_t));
            auto buf_count = runtime_->alloc_buffer_shared(sizeof(uint32_t));
            // Fail-closed: a nil buffer (alloc failure) has contents()==nil; a
            // memcpy into it is UB. Decline with a non-OK GpuError so the engine
            // falls back to CPU — never proceed and emit partial/all-zero rows.
            if (!buf_dig.valid() || !buf_pub.valid() || !buf_sig.valid() ||
                !buf_res.valid() || !buf_count.valid())
                return set_error(GpuError::Memory,
                                 "Metal: ecdsa_verify_lbtc_columns chunk buffer allocation failed");
            std::memcpy(buf_dig.contents(), digests32 + off * 32, n * 32);
            std::memcpy(buf_pub.contents(), pubkeys33 + off * 33, n * 33);
            std::memcpy(buf_sig.contents(), sigs64 + off * 64, n * 64);
            uint32_t n32 = (uint32_t)n;
            std::memcpy(buf_count.contents(), &n32, sizeof(n32));

            // Fatal-not-invalid guard (acceptance A5/A9). MetalRuntime::dispatch_sync
            // is void and only logs cmd_buf.error, so a command-buffer EXECUTION
            // failure (GPU watchdog timeout, device-lost, mid-run fault) leaves
            // buf_res unwritten. Reading it back as all-zero would be misinterpreted
            // as "every row invalid" — an operational failure masquerading as a
            // consensus verdict. The kernel writes only 0/1 for every row < count, so
            // seed each slot with a sentinel it never produces; any surviving sentinel
            // after the dispatch proves the kernel did not complete -> decline with a
            // non-OK GpuError. The engine hook (gpu_engine_hook.cpp) maps non-OK to a
            // CPU fallback, never to invalid rows.
            constexpr uint32_t kUnwritten = 0xFFFFFFFFu;
            auto* res_seed = static_cast<uint32_t*>(buf_res.contents());
            for (size_t i = 0; i < n; ++i) res_seed[i] = kUnwritten;

            runtime_->dispatch_sync(pipe, (uint32_t)n, 64u,
                                    {&buf_dig, &buf_pub, &buf_sig, &buf_res, &buf_count});

            const auto* res = static_cast<const uint32_t*>(buf_res.contents());
            for (size_t i = 0; i < n; ++i)
                if (res[i] == kUnwritten)
                    return set_error(GpuError::Launch,
                                     "Metal: ecdsa_verify_lbtc_columns dispatch left results "
                                     "unwritten (command-buffer failure) — declining to CPU");
            for (size_t i = 0; i < n; ++i)
                out_results[off + i] = res[i] ? 1 : 0;
        }
        clear_error();
        return GpuError::Ok;
    }

    GpuError schnorr_verify_lbtc_columns(
        const uint8_t* digests32, const uint8_t* xonly32,
        const uint8_t* sigs64, size_t count,
        uint8_t* out_results) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!digests32 || !xonly32 || !sigs64 || !out_results)
            return set_error(GpuError::NullArg, "NULL buffer");

        // Fail-closed: initialise the whole result buffer to invalid (0) up front,
        // so any early non-OK return leaves no stale or partial verdict behind.
        // On success every row is overwritten below.
        std::memset(out_results, 0, count);

        auto err = ensure_library();
        if (err != GpuError::Ok) return err;
        auto pipe = runtime_->make_pipeline("schnorr_verify_lbtc_columns");
        if (!pipe.valid())
            return set_error(GpuError::Launch,
                             "Metal: schnorr_verify_lbtc_columns kernel missing from loaded library");

        size_t cap = (static_cast<size_t>(4) << 20);
        if (const char* e = std::getenv("UFSECP_GPU_COLUMNS_CHUNK")) {
            const unsigned long long v = std::strtoull(e, nullptr, 10);
            if (v > 0 && static_cast<size_t>(v) < cap) cap = static_cast<size_t>(v);
        }
        const size_t chunk = (count < cap) ? count : cap;
        for (size_t off = 0; off < count; off += chunk) {
            const size_t n = (count - off < chunk) ? (count - off) : chunk;
            auto buf_dig = runtime_->alloc_buffer_shared(n * 32);
            auto buf_xon = runtime_->alloc_buffer_shared(n * 32);
            auto buf_sig = runtime_->alloc_buffer_shared(n * 64);
            auto buf_res = runtime_->alloc_buffer_shared(n * sizeof(uint32_t));
            auto buf_count = runtime_->alloc_buffer_shared(sizeof(uint32_t));
            // Fail-closed: a nil buffer (alloc failure) has contents()==nil; a
            // memcpy into it is UB. Decline with a non-OK GpuError so the engine
            // falls back to CPU — never proceed and emit partial/all-zero rows.
            if (!buf_dig.valid() || !buf_xon.valid() || !buf_sig.valid() ||
                !buf_res.valid() || !buf_count.valid())
                return set_error(GpuError::Memory,
                                 "Metal: schnorr_verify_lbtc_columns chunk buffer allocation failed");
            std::memcpy(buf_dig.contents(), digests32 + off * 32, n * 32);
            std::memcpy(buf_xon.contents(), xonly32 + off * 32, n * 32);
            std::memcpy(buf_sig.contents(), sigs64 + off * 64, n * 64);
            uint32_t n32 = (uint32_t)n;
            std::memcpy(buf_count.contents(), &n32, sizeof(n32));

            // Fatal-not-invalid guard (acceptance A5/A9) — see ecdsa_verify_lbtc_columns.
            // dispatch_sync swallows command-buffer execution errors; a sentinel that
            // the kernel never writes (it writes only 0/1 per row) detects a dispatch
            // that did not complete and declines to CPU instead of emitting all-invalid
            // rows for an operational GPU failure.
            constexpr uint32_t kUnwritten = 0xFFFFFFFFu;
            auto* res_seed = static_cast<uint32_t*>(buf_res.contents());
            for (size_t i = 0; i < n; ++i) res_seed[i] = kUnwritten;

            runtime_->dispatch_sync(pipe, (uint32_t)n, 64u,
                                    {&buf_dig, &buf_xon, &buf_sig, &buf_res, &buf_count});

            const auto* res = static_cast<const uint32_t*>(buf_res.contents());
            for (size_t i = 0; i < n; ++i)
                if (res[i] == kUnwritten)
                    return set_error(GpuError::Launch,
                                     "Metal: schnorr_verify_lbtc_columns dispatch left results "
                                     "unwritten (command-buffer failure) — declining to CPU");
            for (size_t i = 0; i < n; ++i)
                out_results[off + i] = res[i] ? 1 : 0;
        }
        clear_error();
        return GpuError::Ok;
    }

    /* libbitcoin-bridge COLLECT verify (ECDSA). Native Metal parity with
     * CudaBackend::ecdsa_verify_collect (gpu_backend_cuda.cu:1230). PUBLIC-DATA,
     * variable-time. The verdict is bit-for-bit identical to
     * ecdsa_verify_lbtc_columns / ecdsa_verify_batch; only the OUTPUT convention
     * differs: key_buffer is a 1-byte/row verdict channel the caller SEEDS
     * non-zero, and a VALID row is collapsed to 0 while an INVALID/rejected row is
     * LEFT seeded (the rejected id survives). Fail-closed inversions vs the
     * *_columns clone:
     *   (1) NO up-front memset — 0 == VALID here, so zeroing key_buffer would be a
     *       mass false-accept; any early non-OK return leaves the caller's non-zero
     *       seed intact (== all-rejected == fail-closed).
     *   (2) A command-buffer fault canary (buffer(5), host-seeded 0, kernel-stamped
     *       1) REPLACES the column path's 0xFFFFFFFF result sentinel: a collect row
     *       legitimately keeps its seed on invalid, so the seed cannot double as a
     *       "kernel ran" marker — the canary is the only fault detector. A surviving
     *       0 -> non-OK GpuError (CPU fallback), no copyback. */
    GpuError ecdsa_verify_collect(
        const uint8_t* msg_hashes32, const uint8_t* pubkeys33,
        const uint8_t* sigs64, size_t count,
        uint8_t* key_buffer) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!msg_hashes32 || !pubkeys33 || !sigs64 || !key_buffer)
            return set_error(GpuError::NullArg, "NULL buffer");

        // NO up-front memset: for collect 0 == VALID, so any early non-OK return
        // must leave the caller's non-zero seed (== all-rejected) untouched.
        auto err = ensure_library();
        if (err != GpuError::Ok) return err;
        auto pipe = runtime_->make_pipeline("lbtc_ecdsa_verify_collect");
        if (!pipe.valid())
            return set_error(GpuError::Launch,
                             "Metal: lbtc_ecdsa_verify_collect kernel missing from loaded library");

        size_t cap = (static_cast<size_t>(4) << 20);
        if (const char* e = std::getenv("UFSECP_GPU_COLUMNS_CHUNK")) {
            const unsigned long long v = std::strtoull(e, nullptr, 10);
            if (v > 0 && static_cast<size_t>(v) < cap) cap = static_cast<size_t>(v);
        }
        const size_t chunk = (count < cap) ? count : cap;
        for (size_t off = 0; off < count; off += chunk) {
            const size_t n = (count - off < chunk) ? (count - off) : chunk;
            auto buf_dig    = runtime_->alloc_buffer_shared(n * 32);
            auto buf_pub    = runtime_->alloc_buffer_shared(n * 33);
            auto buf_sig    = runtime_->alloc_buffer_shared(n * 64);
            auto buf_keys   = runtime_->alloc_buffer_shared(n * 1);
            auto buf_count  = runtime_->alloc_buffer_shared(sizeof(uint32_t));
            auto buf_canary = runtime_->alloc_buffer_shared(sizeof(uint32_t));
            // Fail-closed: a nil buffer (alloc failure) has contents()==nil; a
            // memcpy into it is UB. Decline with a non-OK GpuError so the engine
            // falls back to CPU — never proceed and never zero key_buffer.
            if (!buf_dig.valid() || !buf_pub.valid() || !buf_sig.valid() ||
                !buf_keys.valid() || !buf_count.valid() || !buf_canary.valid())
                return set_error(GpuError::Memory,
                                 "Metal: lbtc_ecdsa_verify_collect chunk buffer allocation failed");
            std::memcpy(buf_dig.contents(), msg_hashes32 + off * 32, n * 32);
            std::memcpy(buf_pub.contents(), pubkeys33 + off * 33, n * 33);
            std::memcpy(buf_sig.contents(), sigs64 + off * 64, n * 64);
            // SEED the verdict channel with the caller's non-zero markers: a row
            // the kernel never zeroes (invalid, or unreached after a fault) stays
            // non-zero = rejected = fail-closed.
            std::memcpy(buf_keys.contents(), key_buffer + off, n);
            uint32_t n32 = (uint32_t)n;
            std::memcpy(buf_count.contents(), &n32, sizeof(n32));
            // Fault canary seeded 0; the kernel unconditionally stamps 1.
            uint32_t canary0 = 0u;
            std::memcpy(buf_canary.contents(), &canary0, sizeof(canary0));

            runtime_->dispatch_sync(pipe, (uint32_t)n, 64u,
                                    {&buf_dig, &buf_pub, &buf_sig, &buf_keys,
                                     &buf_count, &buf_canary});

            // dispatch_sync is void and only cerr-logs a command-buffer fault; a
            // surviving canary 0 proves the kernel did not run -> decline to CPU
            // WITHOUT copyback (caller seed stays = all-rejected). Never emit
            // all-zero, never zero a rejected row.
            uint32_t ran = 0u;
            std::memcpy(&ran, buf_canary.contents(), sizeof(ran));
            if (ran == 0u)
                return set_error(GpuError::Launch,
                                 "Metal: lbtc_ecdsa_verify_collect dispatch did not run "
                                 "(command-buffer failure) — declining to CPU");
            // Verdict already collapsed on device (valid->0, invalid->seeded); copy
            // the 1-byte/row channel back VERBATIM — no 0/1 normalization.
            std::memcpy(key_buffer + off, buf_keys.contents(), n);
        }
        clear_error();
        return GpuError::Ok;
    }

    /* libbitcoin-bridge COLLECT verify (Schnorr). Native Metal parity with
     * CudaBackend::schnorr_verify_collect (gpu_backend_cuda.cu:1290). PUBLIC-DATA,
     * variable-time. Same collect convention and fail-closed inversions as
     * ecdsa_verify_collect above (no memset; seed key_buffer; canary fault
     * detector; verbatim readback). */
    GpuError schnorr_verify_collect(
        const uint8_t* msg_hashes32, const uint8_t* pubkeys_x32,
        const uint8_t* sigs64, size_t count,
        uint8_t* key_buffer) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!msg_hashes32 || !pubkeys_x32 || !sigs64 || !key_buffer)
            return set_error(GpuError::NullArg, "NULL buffer");

        // NO up-front memset: 0 == VALID for collect (see ecdsa_verify_collect).
        auto err = ensure_library();
        if (err != GpuError::Ok) return err;
        auto pipe = runtime_->make_pipeline("lbtc_schnorr_verify_collect");
        if (!pipe.valid())
            return set_error(GpuError::Launch,
                             "Metal: lbtc_schnorr_verify_collect kernel missing from loaded library");

        size_t cap = (static_cast<size_t>(4) << 20);
        if (const char* e = std::getenv("UFSECP_GPU_COLUMNS_CHUNK")) {
            const unsigned long long v = std::strtoull(e, nullptr, 10);
            if (v > 0 && static_cast<size_t>(v) < cap) cap = static_cast<size_t>(v);
        }
        const size_t chunk = (count < cap) ? count : cap;
        for (size_t off = 0; off < count; off += chunk) {
            const size_t n = (count - off < chunk) ? (count - off) : chunk;
            auto buf_dig    = runtime_->alloc_buffer_shared(n * 32);
            auto buf_xon    = runtime_->alloc_buffer_shared(n * 32);
            auto buf_sig    = runtime_->alloc_buffer_shared(n * 64);
            auto buf_keys   = runtime_->alloc_buffer_shared(n * 1);
            auto buf_count  = runtime_->alloc_buffer_shared(sizeof(uint32_t));
            auto buf_canary = runtime_->alloc_buffer_shared(sizeof(uint32_t));
            if (!buf_dig.valid() || !buf_xon.valid() || !buf_sig.valid() ||
                !buf_keys.valid() || !buf_count.valid() || !buf_canary.valid())
                return set_error(GpuError::Memory,
                                 "Metal: lbtc_schnorr_verify_collect chunk buffer allocation failed");
            std::memcpy(buf_dig.contents(), msg_hashes32 + off * 32, n * 32);
            std::memcpy(buf_xon.contents(), pubkeys_x32 + off * 32, n * 32);
            std::memcpy(buf_sig.contents(), sigs64 + off * 64, n * 64);
            // SEED verdict channel from the caller's non-zero markers.
            std::memcpy(buf_keys.contents(), key_buffer + off, n);
            uint32_t n32 = (uint32_t)n;
            std::memcpy(buf_count.contents(), &n32, sizeof(n32));
            uint32_t canary0 = 0u;
            std::memcpy(buf_canary.contents(), &canary0, sizeof(canary0));

            runtime_->dispatch_sync(pipe, (uint32_t)n, 64u,
                                    {&buf_dig, &buf_xon, &buf_sig, &buf_keys,
                                     &buf_count, &buf_canary});

            uint32_t ran = 0u;
            std::memcpy(&ran, buf_canary.contents(), sizeof(ran));
            if (ran == 0u)
                return set_error(GpuError::Launch,
                                 "Metal: lbtc_schnorr_verify_collect dispatch did not run "
                                 "(command-buffer failure) — declining to CPU");
            std::memcpy(key_buffer + off, buf_keys.contents(), n);
        }
        clear_error();
        return GpuError::Ok;
    }

    /* libbitcoin-bridge: batch x-only key validation (lift_x per key). PUBLIC.
     * Native Metal parity with CudaBackend::xonly_validate. */
    GpuError xonly_validate(
        const uint8_t* keys32, size_t n, uint8_t* results) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        if (n == 0) { clear_error(); return GpuError::Ok; }
        if (!keys32 || !results) return set_error(GpuError::NullArg, "NULL buffer");

        // Fail-closed: any early non-OK return leaves an all-invalid buffer, never
        // a stale/partial verdict. On success every row is overwritten below.
        std::memset(results, 0, n);

        auto err = ensure_library();
        if (err != GpuError::Ok) return err;
        auto pipe = runtime_->make_pipeline("lbtc_xonly_validate");
        if (!pipe.valid())
            return set_error(GpuError::Launch,
                             "Metal: lbtc_xonly_validate kernel missing from loaded library");

        auto buf_keys  = runtime_->alloc_buffer_shared(n * 32);
        auto buf_res   = runtime_->alloc_buffer_shared(n * sizeof(uint32_t));
        auto buf_count = runtime_->alloc_buffer_shared(sizeof(uint32_t));
        if (!buf_keys.valid() || !buf_res.valid() || !buf_count.valid())
            return set_error(GpuError::Memory,
                             "Metal: lbtc_xonly_validate buffer allocation failed");

        std::memcpy(buf_keys.contents(), keys32, n * 32);
        uint32_t n32 = (uint32_t)n;
        std::memcpy(buf_count.contents(), &n32, sizeof(n32));

        // Fatal-not-invalid guard: dispatch_sync is void and only logs
        // cmd_buf.error, so a command-buffer execution failure leaves buf_res
        // unwritten. The kernel writes only 0/1 per row < count, so seed each slot
        // with a sentinel it never produces; any survivor -> decline to CPU.
        constexpr uint32_t kUnwritten = 0xFFFFFFFFu;
        auto* res_seed = static_cast<uint32_t*>(buf_res.contents());
        for (size_t i = 0; i < n; ++i) res_seed[i] = kUnwritten;

        runtime_->dispatch_sync(pipe, (uint32_t)n, 64u, {&buf_keys, &buf_res, &buf_count});

        const auto* res = static_cast<const uint32_t*>(buf_res.contents());
        for (size_t i = 0; i < n; ++i)
            if (res[i] == kUnwritten)
                return set_error(GpuError::Launch,
                                 "Metal: lbtc_xonly_validate dispatch left results unwritten "
                                 "(command-buffer failure) — declining to CPU");
        for (size_t i = 0; i < n; ++i)
            results[i] = res[i] ? 1 : 0;

        clear_error();
        return GpuError::Ok;
    }

    /* libbitcoin-bridge: batch full compressed-pubkey validation. PUBLIC.
     * Native Metal parity with CudaBackend::pubkey_validate. */
    GpuError pubkey_validate(
        const uint8_t* pubkeys33, size_t n, uint8_t* results) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        if (n == 0) { clear_error(); return GpuError::Ok; }
        if (!pubkeys33 || !results) return set_error(GpuError::NullArg, "NULL buffer");

        std::memset(results, 0, n);

        auto err = ensure_library();
        if (err != GpuError::Ok) return err;
        auto pipe = runtime_->make_pipeline("lbtc_pubkey_validate");
        if (!pipe.valid())
            return set_error(GpuError::Launch,
                             "Metal: lbtc_pubkey_validate kernel missing from loaded library");

        auto buf_pk    = runtime_->alloc_buffer_shared(n * 33);
        auto buf_res   = runtime_->alloc_buffer_shared(n * sizeof(uint32_t));
        auto buf_count = runtime_->alloc_buffer_shared(sizeof(uint32_t));
        if (!buf_pk.valid() || !buf_res.valid() || !buf_count.valid())
            return set_error(GpuError::Memory,
                             "Metal: lbtc_pubkey_validate buffer allocation failed");

        std::memcpy(buf_pk.contents(), pubkeys33, n * 33);
        uint32_t n32 = (uint32_t)n;
        std::memcpy(buf_count.contents(), &n32, sizeof(n32));

        constexpr uint32_t kUnwritten = 0xFFFFFFFFu;
        auto* res_seed = static_cast<uint32_t*>(buf_res.contents());
        for (size_t i = 0; i < n; ++i) res_seed[i] = kUnwritten;

        runtime_->dispatch_sync(pipe, (uint32_t)n, 64u, {&buf_pk, &buf_res, &buf_count});

        const auto* res = static_cast<const uint32_t*>(buf_res.contents());
        for (size_t i = 0; i < n; ++i)
            if (res[i] == kUnwritten)
                return set_error(GpuError::Launch,
                                 "Metal: lbtc_pubkey_validate dispatch left results unwritten "
                                 "(command-buffer failure) — declining to CPU");
        for (size_t i = 0; i < n; ++i)
            results[i] = res[i] ? 1 : 0;

        clear_error();
        return GpuError::Ok;
    }

    /* libbitcoin-bridge: BIP-341 commitment tweak-add-check, one thread per item.
     * accept iff x(lift_x(internal)+tweak*G)==tweaked_x AND y-parity==parity.
     * PUBLIC. Native Metal parity with CudaBackend::commitment_verify. */
    GpuError commitment_verify(
        const uint8_t* internal_x32, const uint8_t* tweak32,
        const uint8_t* tweaked_x32, const uint8_t* parity,
        size_t n, uint8_t* results) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        if (n == 0) { clear_error(); return GpuError::Ok; }
        if (!internal_x32 || !tweak32 || !tweaked_x32 || !parity || !results)
            return set_error(GpuError::NullArg, "NULL buffer");

        std::memset(results, 0, n);

        auto err = ensure_library();
        if (err != GpuError::Ok) return err;
        auto pipe = runtime_->make_pipeline("lbtc_commitment_verify");
        if (!pipe.valid())
            return set_error(GpuError::Launch,
                             "Metal: lbtc_commitment_verify kernel missing from loaded library");

        auto buf_ix    = runtime_->alloc_buffer_shared(n * 32);
        auto buf_tw    = runtime_->alloc_buffer_shared(n * 32);
        auto buf_tx    = runtime_->alloc_buffer_shared(n * 32);
        auto buf_par   = runtime_->alloc_buffer_shared(n);
        auto buf_res   = runtime_->alloc_buffer_shared(n * sizeof(uint32_t));
        auto buf_count = runtime_->alloc_buffer_shared(sizeof(uint32_t));
        if (!buf_ix.valid() || !buf_tw.valid() || !buf_tx.valid() ||
            !buf_par.valid() || !buf_res.valid() || !buf_count.valid())
            return set_error(GpuError::Memory,
                             "Metal: lbtc_commitment_verify buffer allocation failed");

        std::memcpy(buf_ix.contents(), internal_x32, n * 32);
        std::memcpy(buf_tw.contents(), tweak32, n * 32);
        std::memcpy(buf_tx.contents(), tweaked_x32, n * 32);
        std::memcpy(buf_par.contents(), parity, n);
        uint32_t n32 = (uint32_t)n;
        std::memcpy(buf_count.contents(), &n32, sizeof(n32));

        constexpr uint32_t kUnwritten = 0xFFFFFFFFu;
        auto* res_seed = static_cast<uint32_t*>(buf_res.contents());
        for (size_t i = 0; i < n; ++i) res_seed[i] = kUnwritten;

        runtime_->dispatch_sync(pipe, (uint32_t)n, 64u,
                                {&buf_ix, &buf_tw, &buf_tx, &buf_par, &buf_res, &buf_count});

        const auto* res = static_cast<const uint32_t*>(buf_res.contents());
        for (size_t i = 0; i < n; ++i)
            if (res[i] == kUnwritten)
                return set_error(GpuError::Launch,
                                 "Metal: lbtc_commitment_verify dispatch left results unwritten "
                                 "(command-buffer failure) — declining to CPU");
        for (size_t i = 0; i < n; ++i)
            results[i] = res[i] ? 1 : 0;

        clear_error();
        return GpuError::Ok;
    }

    /* libbitcoin-bridge: Taproot tagged hash. tag_hash32 = SHA256(tag)
     * (host-precomputed); out_i = SHA256(tag_hash||tag_hash||msg_i). PUBLIC.
     * Native Metal parity with CudaBackend::tagged_hash. Hash op: no sentinel
     * (matches CUDA — relies on ensure_library/pipeline/alloc guards); out32 is
     * NOT pre-zeroed, matching CUDA. */
    GpuError tagged_hash(
        const uint8_t* tag_hash32, const uint8_t* msgs,
        size_t msg_len, size_t n, uint8_t* out32) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        if (n == 0) { clear_error(); return GpuError::Ok; }
        if (!tag_hash32 || !msgs || !out32) return set_error(GpuError::NullArg, "NULL buffer");
        if (msg_len == 0 || msg_len > 256) return set_error(GpuError::BadInput, "msg_len out of range");

        auto err = ensure_library();
        if (err != GpuError::Ok) return err;
        auto pipe = runtime_->make_pipeline("lbtc_tagged_hash");
        if (!pipe.valid())
            return set_error(GpuError::Launch,
                             "Metal: lbtc_tagged_hash kernel missing from loaded library");

        auto buf_th     = runtime_->alloc_buffer_shared(32);
        auto buf_msgs   = runtime_->alloc_buffer_shared(n * msg_len);
        auto buf_msglen = runtime_->alloc_buffer_shared(sizeof(uint32_t));
        auto buf_out    = runtime_->alloc_buffer_shared(n * 32);
        auto buf_count  = runtime_->alloc_buffer_shared(sizeof(uint32_t));
        if (!buf_th.valid() || !buf_msgs.valid() || !buf_msglen.valid() ||
            !buf_out.valid() || !buf_count.valid())
            return set_error(GpuError::Memory,
                             "Metal: lbtc_tagged_hash buffer allocation failed");

        std::memcpy(buf_th.contents(), tag_hash32, 32);
        std::memcpy(buf_msgs.contents(), msgs, n * msg_len);
        uint32_t ml32 = (uint32_t)msg_len;
        std::memcpy(buf_msglen.contents(), &ml32, sizeof(ml32));
        uint32_t n32 = (uint32_t)n;
        std::memcpy(buf_count.contents(), &n32, sizeof(n32));

        runtime_->dispatch_sync(pipe, (uint32_t)n, 64u,
                                {&buf_th, &buf_msgs, &buf_msglen, &buf_out, &buf_count});

        std::memcpy(out32, buf_out.contents(), n * 32);

        clear_error();
        return GpuError::Ok;
    }

    /* libbitcoin-bridge: Taproot tagged hash, per-item length (TapLeaf). PUBLIC.
     * Native Metal parity with CudaBackend::tagged_hash_var. */
    GpuError tagged_hash_var(
        const uint8_t* tag_hash32, const uint8_t* msgs, const uint32_t* msg_lens,
        size_t stride, size_t n, uint8_t* out32) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        if (n == 0) { clear_error(); return GpuError::Ok; }
        if (!tag_hash32 || !msgs || !msg_lens || !out32) return set_error(GpuError::NullArg, "NULL buffer");
        if (stride == 0 || stride > 256) return set_error(GpuError::BadInput, "stride out of range");

        auto err = ensure_library();
        if (err != GpuError::Ok) return err;
        auto pipe = runtime_->make_pipeline("lbtc_tagged_hash_var");
        if (!pipe.valid())
            return set_error(GpuError::Launch,
                             "Metal: lbtc_tagged_hash_var kernel missing from loaded library");

        auto buf_th     = runtime_->alloc_buffer_shared(32);
        auto buf_msgs   = runtime_->alloc_buffer_shared(n * stride);
        auto buf_lens   = runtime_->alloc_buffer_shared(n * sizeof(uint32_t));
        auto buf_stride = runtime_->alloc_buffer_shared(sizeof(uint32_t));
        auto buf_out    = runtime_->alloc_buffer_shared(n * 32);
        auto buf_count  = runtime_->alloc_buffer_shared(sizeof(uint32_t));
        if (!buf_th.valid() || !buf_msgs.valid() || !buf_lens.valid() ||
            !buf_stride.valid() || !buf_out.valid() || !buf_count.valid())
            return set_error(GpuError::Memory,
                             "Metal: lbtc_tagged_hash_var buffer allocation failed");

        std::memcpy(buf_th.contents(), tag_hash32, 32);
        std::memcpy(buf_msgs.contents(), msgs, n * stride);
        std::memcpy(buf_lens.contents(), msg_lens, n * sizeof(uint32_t));
        uint32_t st32 = (uint32_t)stride;
        std::memcpy(buf_stride.contents(), &st32, sizeof(st32));
        uint32_t n32 = (uint32_t)n;
        std::memcpy(buf_count.contents(), &n32, sizeof(n32));

        runtime_->dispatch_sync(pipe, (uint32_t)n, 64u,
                                {&buf_th, &buf_msgs, &buf_lens, &buf_stride, &buf_out, &buf_count});

        std::memcpy(out32, buf_out.contents(), n * 32);

        clear_error();
        return GpuError::Ok;
    }

    /* libbitcoin-bridge: batch HASH256 (double SHA-256) of fixed-length inputs.
     * PUBLIC. Native Metal parity with CudaBackend::hash256. */
    GpuError hash256(
        const uint8_t* inputs, size_t input_len, size_t n, uint8_t* out32) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        if (n == 0) { clear_error(); return GpuError::Ok; }
        if (!inputs || !out32) return set_error(GpuError::NullArg, "NULL buffer");
        if (input_len == 0 || input_len > 320) return set_error(GpuError::BadInput, "input_len out of range");

        auto err = ensure_library();
        if (err != GpuError::Ok) return err;
        auto pipe = runtime_->make_pipeline("lbtc_hash256");
        if (!pipe.valid())
            return set_error(GpuError::Launch,
                             "Metal: lbtc_hash256 kernel missing from loaded library");

        auto buf_in    = runtime_->alloc_buffer_shared(n * input_len);
        auto buf_inlen = runtime_->alloc_buffer_shared(sizeof(uint32_t));
        auto buf_out   = runtime_->alloc_buffer_shared(n * 32);
        auto buf_count = runtime_->alloc_buffer_shared(sizeof(uint32_t));
        if (!buf_in.valid() || !buf_inlen.valid() || !buf_out.valid() || !buf_count.valid())
            return set_error(GpuError::Memory,
                             "Metal: lbtc_hash256 buffer allocation failed");

        std::memcpy(buf_in.contents(), inputs, n * input_len);
        uint32_t il32 = (uint32_t)input_len;
        std::memcpy(buf_inlen.contents(), &il32, sizeof(il32));
        uint32_t n32 = (uint32_t)n;
        std::memcpy(buf_count.contents(), &n32, sizeof(n32));

        runtime_->dispatch_sync(pipe, (uint32_t)n, 64u,
                                {&buf_in, &buf_inlen, &buf_out, &buf_count});

        std::memcpy(out32, buf_out.contents(), n * 32);

        clear_error();
        return GpuError::Ok;
    }

    /* libbitcoin-bridge: batch HASH256 (double SHA-256) of variable-length
     * inputs sharing a common row stride. PUBLIC-data hashing only, no tag
     * prefix, no transaction parsing. Rows may be multi-megabyte (unlike
     * hash256's fixed <=320-byte rows) — see sha256_update_device in
     * secp256k1_kernels.metal for why this needs its own kernel rather than
     * reusing lbtc_hash256's copy-into-thread-buffer approach.
     * Dispatch structure mirrors tagged_hash_var (also uploads a per-item
     * lengths array + stride scalar); no tag_hash32 buffer here. */
    GpuError hash256_var(
        const uint8_t* inputs, const uint32_t* input_lens,
        size_t stride, size_t n, uint8_t* out32) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        if (n == 0) { clear_error(); return GpuError::Ok; }
        if (!inputs || !input_lens || !out32) return set_error(GpuError::NullArg, "NULL buffer");
        if (stride == 0) return set_error(GpuError::BadInput, "stride out of range");

        auto err = ensure_library();
        if (err != GpuError::Ok) return err;
        auto pipe = runtime_->make_pipeline("lbtc_hash256_var");
        if (!pipe.valid())
            return set_error(GpuError::Launch,
                             "Metal: lbtc_hash256_var kernel missing from loaded library");

        auto buf_in     = runtime_->alloc_buffer_shared(n * stride);
        auto buf_lens   = runtime_->alloc_buffer_shared(n * sizeof(uint32_t));
        auto buf_stride = runtime_->alloc_buffer_shared(sizeof(uint32_t));
        auto buf_out    = runtime_->alloc_buffer_shared(n * 32);
        auto buf_count  = runtime_->alloc_buffer_shared(sizeof(uint32_t));
        if (!buf_in.valid() || !buf_lens.valid() || !buf_stride.valid() ||
            !buf_out.valid() || !buf_count.valid())
            return set_error(GpuError::Memory,
                             "Metal: lbtc_hash256_var buffer allocation failed");

        std::memcpy(buf_in.contents(), inputs, n * stride);
        std::memcpy(buf_lens.contents(), input_lens, n * sizeof(uint32_t));
        uint32_t st32 = (uint32_t)stride;
        std::memcpy(buf_stride.contents(), &st32, sizeof(st32));
        uint32_t n32 = (uint32_t)n;
        std::memcpy(buf_count.contents(), &n32, sizeof(n32));

        runtime_->dispatch_sync(pipe, (uint32_t)n, 64u,
                                {&buf_in, &buf_lens, &buf_stride, &buf_out, &buf_count});

        std::memcpy(out32, buf_out.contents(), n * 32);

        clear_error();
        return GpuError::Ok;
    }

    /* libbitcoin-bridge: Merkle-tree parent hashing, SoA layout.
     * out[i] = SHA256(SHA256(left32[i] || right32[i])). PUBLIC-data hashing only.
     * Native Metal parity with CudaBackend::merkle_pair_hash. */
    GpuError merkle_pair_hash(
        const uint8_t* left32, const uint8_t* right32,
        size_t n, uint8_t* out32) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        if (n == 0) { clear_error(); return GpuError::Ok; }
        if (!left32 || !right32 || !out32) return set_error(GpuError::NullArg, "NULL buffer");

        auto err = ensure_library();
        if (err != GpuError::Ok) return err;
        auto pipe = runtime_->make_pipeline("lbtc_merkle_pair");
        if (!pipe.valid())
            return set_error(GpuError::Launch,
                             "Metal: lbtc_merkle_pair kernel missing from loaded library");

        auto buf_left  = runtime_->alloc_buffer_shared(n * 32);
        auto buf_right = runtime_->alloc_buffer_shared(n * 32);
        auto buf_out   = runtime_->alloc_buffer_shared(n * 32);
        auto buf_count = runtime_->alloc_buffer_shared(sizeof(uint32_t));
        if (!buf_left.valid() || !buf_right.valid() || !buf_out.valid() || !buf_count.valid())
            return set_error(GpuError::Memory,
                             "Metal: lbtc_merkle_pair buffer allocation failed");

        std::memcpy(buf_left.contents(), left32, n * 32);
        std::memcpy(buf_right.contents(), right32, n * 32);
        uint32_t n32 = (uint32_t)n;
        std::memcpy(buf_count.contents(), &n32, sizeof(n32));

        runtime_->dispatch_sync(pipe, (uint32_t)n, 64u,
                                {&buf_left, &buf_right, &buf_out, &buf_count});

        std::memcpy(out32, buf_out.contents(), n * 32);

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
#if !SECP256K1_GPU_HAS_ECDH
        return set_error(GpuError::Unsupported, "GPU ECDH module disabled at build time");
#endif

        auto err = ensure_library();
        if (err != GpuError::Ok) return err;

        /* Use scalar_mul_batch_compressed(peer33, privkeys) → AffinePoint results.
           GPU decompresses pubkeys + multiplies in one step.
           Then compress and SHA256 on host to match CUDA/OpenCL semantics. */
        auto& scratch = g_metal_batch_scratch;
        scratch.ensure_scalars(count);
        auto* const h_scalars = scratch.scalars.data();
        MetalScalarEraseGuard h_scalars_guard{h_scalars, count};
        for (size_t i = 0; i < count; ++i)
            h_scalars[i] = be32_to_metal_scalar(privkeys32 + i * 32);

        auto buf_pubs33  = runtime_->alloc_buffer_shared(count * 33);
        std::memcpy(buf_pubs33.contents(), peer_pubkeys33, count * 33);

        auto buf_scalars = runtime_->alloc_buffer_shared(count * sizeof(MetalScalar256));
        std::memcpy(buf_scalars.contents(), h_scalars, count * sizeof(MetalScalar256));
        MetalBufferEraseGuard buf_scalars_guard{&buf_scalars};

        auto buf_results = runtime_->alloc_buffer_shared(count * sizeof(MetalAffinePoint));

        uint32_t n32 = (uint32_t)count;
        auto buf_count = runtime_->alloc_buffer_shared(sizeof(uint32_t));
        std::memcpy(buf_count.contents(), &n32, sizeof(n32));

        auto pipe = runtime_->make_pipeline("scalar_mul_batch_compressed");
        runtime_->dispatch_sync(pipe, (uint32_t)count, 64u,
                                {&buf_pubs33, &buf_scalars, &buf_results, &buf_count});

        const auto* aff = static_cast<const MetalAffinePoint*>(buf_results.contents());
        for (size_t i = 0; i < count; ++i) {
            uint8_t compressed[33];
            metal_affine_to_sec1(aff[i], compressed);
            auto digest = secp256k1::SHA256::hash(compressed, sizeof(compressed));
            std::memcpy(out_secrets32 + i * 32, digest.data(), 32);
        }

        // SECURITY FIX (HIGH-2): Zero private key material from shared Metal buffer
        // and thread-local scratch before returning, so key material is not retained.
        secp256k1::detail::secure_erase(buf_scalars.contents(),
                                        count * sizeof(MetalScalar256));

        // SECURITY FIX (LOW-5): Zero private key scalars from host-side scratch
        secp256k1::detail::secure_erase(scratch.scalars.data(),
                                        count * sizeof(MetalScalar256));

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
#if !SECP256K1_GPU_HAS_HASH160
        return set_error(GpuError::Unsupported, "GPU HASH160 module disabled at build time");
#endif

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
#if !SECP256K1_GPU_HAS_FROST
        return set_error(GpuError::Unsupported, "GPU FROST module disabled at build time");
#endif

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
#if !SECP256K1_GPU_HAS_ECRECOVER
        return set_error(GpuError::Unsupported, "GPU ECRECOVER module disabled at build time");
#endif

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
#if !SECP256K1_GPU_HAS_MSM
        return set_error(GpuError::Unsupported, "GPU MSM module disabled at build time");
#endif

        auto err = ensure_library();
        if (err != GpuError::Ok) return err;

        /* Pass 33-byte compressed pubkeys directly — GPU decompresses via scalar_mul_batch_compressed */
        /* Scalars */
        std::vector<MetalScalar256> h_scalars(n);
        for (size_t i = 0; i < n; ++i)
            h_scalars[i] = be32_to_metal_scalar(scalars32 + i * 32);

        auto buf_pubs33  = runtime_->alloc_buffer_shared(n * 33);
        std::memcpy(buf_pubs33.contents(), points33, n * 33);

        /* Ensure persistent pool buffers (grow-only, avoids repeated alloc overhead) */
        msm_pool_.ensure(n, runtime_.get());
        if (msm_pool_.capacity < n)
            return set_error(GpuError::Device, "MSM pool allocation failed");

        const size_t n_blocks = (n + 255) / 256;

        /* Upload input data to pool buffers */
        std::memcpy(msm_pool_.buf_bases.contents(),   points33,          n * 33);
        std::memcpy(msm_pool_.buf_scalars.contents(), h_scalars.data(), n * sizeof(MetalScalar256));

        /* Small count buffer (4 bytes) — negligible to allocate */
        uint32_t n32 = (uint32_t)n;
        auto buf_count = runtime_->alloc_buffer_shared(sizeof(uint32_t));
        std::memcpy(buf_count.contents(), &n32, sizeof(n32));

        /* Pass 1: GPU scalar_mul_batch_compressed → buf_partials (n AffinePoints) */
        auto pipe_sm = runtime_->make_pipeline("scalar_mul_batch_compressed");
        runtime_->dispatch_sync(pipe_sm, (uint32_t)n, 64u,
                                {&msm_pool_.buf_bases, &msm_pool_.buf_scalars,
                                 &msm_pool_.buf_partials, &buf_count});

        /* Pass 2 (optional): GPU msm_block_sum_kernel → buf_blocks (n_blocks JacobianPoints) */
        auto pipe_bs = runtime_->make_pipeline("msm_block_sum_kernel");
        if (pipe_bs.valid()) {
            runtime_->dispatch_sync(pipe_bs, (uint32_t)n_blocks, 1u,
                                    {&msm_pool_.buf_partials, &buf_count, &msm_pool_.buf_blocks});

            /* CPU: Jacobian→Affine + accumulate (only n_blocks iterations) */
            const auto* jac_blocks =
                static_cast<const MetalJacobianPoint*>(msm_pool_.buf_blocks.contents());

            bool have_acc = false;
            secp256k1::fast::FieldElement acc_x, acc_y;

            for (size_t b = 0; b < n_blocks; ++b) {
                secp256k1::fast::FieldElement px, py;
                if (!metal_jacobian_to_cpu_affine(jac_blocks[b], px, py)) continue;
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
        } else {
            /* Fallback: read all n partials to CPU (no block reduce available) */
            const auto* aff =
                static_cast<const MetalAffinePoint*>(msm_pool_.buf_partials.contents());
            bool have_acc = false;
            secp256k1::fast::FieldElement acc_x, acc_y;

            for (size_t i = 0; i < n; ++i) {
                const auto& xl_mt = aff[i].x.limbs;
                const auto& yl_mt = aff[i].y.limbs;
                std::array<uint64_t,4> xl64, yl64;
                for (int k = 0; k < 4; k++) {
                    xl64[k] = ((uint64_t)xl_mt[2*k+1] << 32) | (uint64_t)xl_mt[2*k];
                    yl64[k] = ((uint64_t)yl_mt[2*k+1] << 32) | (uint64_t)yl_mt[2*k];
                }
                auto px = secp256k1::fast::FieldElement::from_limbs(xl64);
                auto py = secp256k1::fast::FieldElement::from_limbs(yl64);

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
        }

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
#if !SECP256K1_GPU_HAS_ZK
        return set_error(GpuError::Unsupported, "GPU ZK module disabled at build time");
#endif

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
#if !SECP256K1_GPU_HAS_ZK
        return set_error(GpuError::Unsupported, "GPU ZK module disabled at build time");
#endif

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
#if !SECP256K1_GPU_HAS_ZK
        return set_error(GpuError::Unsupported, "GPU ZK module disabled at build time");
#endif

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
#if !SECP256K1_GPU_HAS_BIP324
        return set_error(GpuError::Unsupported, "GPU BIP-324 module disabled at build time");
#endif

        auto err = ensure_library();
        if (err != GpuError::Ok) return err;

        const size_t wire_stride = (size_t)max_payload + 19u; /* BIP324_OVERHEAD = 3 hdr + 16 tag */

        auto buf_keys   = runtime_->alloc_buffer_shared(count * 32);
        std::memcpy(buf_keys.contents(), keys32, count * 32);
        MetalBufferEraseGuard buf_keys_guard{&buf_keys};
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
#if !SECP256K1_GPU_HAS_BIP324
        return set_error(GpuError::Unsupported, "GPU BIP-324 module disabled at build time");
#endif

        auto err = ensure_library();
        if (err != GpuError::Ok) return err;

        const size_t wire_stride = (size_t)max_payload + 19u;

        auto buf_keys    = runtime_->alloc_buffer_shared(count * 32);
        std::memcpy(buf_keys.contents(), keys32, count * 32);
        MetalBufferEraseGuard buf_keys_guard{&buf_keys};
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

    GpuError snark_witness_batch(
        const uint8_t* msg_hashes32, const uint8_t* pubkeys33,
        const uint8_t* sigs64, size_t count, uint8_t* witness_flat_out) override
    {
        if (!msg_hashes32 || !pubkeys33 || !sigs64 || !witness_flat_out)
            return set_error(GpuError::NullArg, "NULL pointer passed to snark_witness_batch");
        if (!count) return GpuError::Ok;
#if !SECP256K1_GPU_HAS_ZK
        return set_error(GpuError::Unsupported, "GPU ZK module disabled at build time");
#endif

        /* Pass 33-byte compressed pubkeys directly — GPU decompresses via lbtc_point_from_compressed */
        auto buf_msgs  = runtime_->alloc_buffer_shared(count * 32);
        auto buf_pubs  = runtime_->alloc_buffer_shared(count * 33);
        auto buf_sigs  = runtime_->alloc_buffer_shared(count * 64);
        auto buf_out   = runtime_->alloc_buffer_shared(count * 760);
        auto buf_count = runtime_->alloc_buffer_shared(sizeof(uint32_t));

        std::memcpy(buf_msgs.contents(),  msg_hashes32,   count * 32);
        std::memcpy(buf_pubs.contents(),  pubkeys33,      count * 33);
        std::memcpy(buf_sigs.contents(),  sigs64,         count * 64);
        uint32_t n32 = (uint32_t)count;
        std::memcpy(buf_count.contents(), &n32, sizeof(n32));

        auto pipe = runtime_->make_pipeline("ecdsa_snark_witness_batch_compressed");
        if (!pipe.valid())
            return set_error(GpuError::Launch,
                             "Metal: ecdsa_snark_witness_batch kernel missing from loaded library");
        runtime_->dispatch_sync(pipe, n32, 64u,
                                {&buf_msgs, &buf_pubs, &buf_sigs, &buf_out, &buf_count});

        std::memcpy(witness_flat_out, buf_out.contents(), count * 760);
        clear_error();
        return GpuError::Ok;
    }

    /* -- BIP-340 Schnorr SNARK witness GPU batch (eprint 2025/695) ---------- */

    GpuError schnorr_snark_witness_batch(
        const uint8_t* msgs32, const uint8_t* pubkeys_x32,
        const uint8_t* sigs64, size_t count, uint8_t* out_flat) override
    {
        if (!msgs32 || !pubkeys_x32 || !sigs64 || !out_flat)
            return set_error(GpuError::NullArg, "NULL pointer passed to schnorr_snark_witness_batch");
        if (!count) return GpuError::Ok;
#if !SECP256K1_GPU_HAS_ZK
        return set_error(GpuError::Unsupported, "GPU ZK module disabled at build time");
#endif

        auto buf_msgs  = runtime_->alloc_buffer_shared(count * 32);
        auto buf_pubs  = runtime_->alloc_buffer_shared(count * 32);
        auto buf_sigs  = runtime_->alloc_buffer_shared(count * 64);
        auto buf_out   = runtime_->alloc_buffer_shared(count * 472);
        auto buf_count = runtime_->alloc_buffer_shared(sizeof(uint32_t));

        std::memcpy(buf_msgs.contents(),  msgs32,       count * 32);
        std::memcpy(buf_pubs.contents(),  pubkeys_x32,  count * 32);
        std::memcpy(buf_sigs.contents(),  sigs64,       count * 64);
        uint32_t n32 = (uint32_t)count;
        std::memcpy(buf_count.contents(), &n32, sizeof(n32));

        auto pipe = runtime_->make_pipeline("schnorr_snark_witness_batch");
        if (!pipe.valid())
            return set_error(GpuError::Launch,
                             "Metal: schnorr_snark_witness_batch kernel missing from loaded library");
        runtime_->dispatch_sync(pipe, n32, 64u,
                                {&buf_msgs, &buf_pubs, &buf_sigs, &buf_out, &buf_count});

        std::memcpy(out_flat, buf_out.contents(), count * 472);
        clear_error();
        return GpuError::Ok;
    }

    GpuError bip352_scan_batch(
        const uint8_t* scan_privkey32, const uint8_t* spend_pubkey33,
        const uint8_t* tweak_pubkeys33, size_t n_tweaks, uint64_t* prefix64_out) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        if (!n_tweaks) { clear_error(); return GpuError::Ok; }
        if (!scan_privkey32 || !spend_pubkey33 || !tweak_pubkeys33 || !prefix64_out)
            return set_error(GpuError::NullArg, "NULL pointer passed to bip352_scan_batch");
#if !SECP256K1_GPU_HAS_BIP352
        return set_error(GpuError::Unsupported, "GPU BIP-352 module disabled at build time");
#endif

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

        /* Parse scan private key: BE 32 bytes → MetalScalar256 */
        MetalScalar256 scan_scalar = be32_to_metal_scalar(scan_privkey32);
        MetalScalarEraseGuard scan_scalar_guard{&scan_scalar, 1};

        /* Build Metal buffers — pass 33-byte pubkeys directly (GPU decompresses) */
        auto buf_tweaks = runtime_->alloc_buffer_shared(n_tweaks * 33);
        auto buf_scan   = runtime_->alloc_buffer_shared(sizeof(MetalScalar256));
        auto buf_spend  = runtime_->alloc_buffer_shared(33);
        auto buf_prefix = runtime_->alloc_buffer_shared(n_tweaks * sizeof(uint64_t));
        auto buf_count  = runtime_->alloc_buffer_shared(sizeof(uint32_t));

        std::memcpy(buf_tweaks.contents(), tweak_pubkeys33, n_tweaks * 33);
        std::memcpy(buf_scan.contents(),  &scan_scalar, sizeof(scan_scalar));
        MetalBufferEraseGuard buf_scan_guard{&buf_scan};
        std::memcpy(buf_spend.contents(), spend_pubkey33, 33);
        uint32_t n32 = (uint32_t)n_tweaks;
        std::memcpy(buf_count.contents(), &n32, sizeof(n32));

        auto pipe = runtime_->make_pipeline("bip352_scan_pipeline_compressed");
        if (!pipe.valid())
            return set_error(GpuError::Launch,
                             "Metal: bip352_scan_pipeline_compressed kernel missing from loaded library");
        runtime_->dispatch_sync(pipe, n32, 64u,
                                {&buf_tweaks, &buf_scan, &buf_spend, &buf_prefix, &buf_count});

        std::memcpy(prefix64_out, buf_prefix.contents(), n_tweaks * sizeof(uint64_t));

        // Rule 10: zero scan private key from Metal shared buffer and stack before release
        secp256k1::detail::secure_erase(buf_scan.contents(), sizeof(MetalScalar256));
        secp256k1::detail::secure_erase(&scan_scalar, sizeof(scan_scalar));

        clear_error();
        return GpuError::Ok;
    }

private:
    std::unique_ptr<secp256k1::metal::MetalRuntime> runtime_;
    bool lib_init_attempted_ = false;
    bool lib_ready_          = false;
    MetalMsmPool msm_pool_;
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
