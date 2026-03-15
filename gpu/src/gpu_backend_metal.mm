/* ============================================================================
 * UltrafastSecp256k1 -- Metal Backend Bridge (EXPERIMENTAL)
 * ============================================================================
 * Implements gpu::GpuBackend for Apple Metal.
 * Wraps the existing secp256k1::metal::MetalRuntime class.
 *
 * STATUS: Experimental / discovery-only.
 * Device discovery and lifecycle (init, device_info) work.
 * All 6 first-wave batch ops return UNSUPPORTED -- the MSL kernel → metallib
 * → pipeline dispatch path is not yet wired from CMake.
 *
 * Compiled ONLY when SECP256K1_HAVE_METAL is set (via CMake).
 * Must be compiled as Objective-C++ (.mm) on macOS.
 * ============================================================================ */

#include "../include/gpu_backend.hpp"

#include <cstring>
#include <cstdio>
#include <vector>

/* -- Metal Runtime (Layer 1) ----------------------------------------------- */
#include "metal_runtime.h"

/* -- CPU FieldElement for host-side conversions ---------------------------- */
#include "secp256k1/field.hpp"

namespace secp256k1 {
namespace gpu {

class MetalBackend final : public GpuBackend {
public:
    MetalBackend() = default;
    ~MetalBackend() override { shutdown(); }

    /* -- Backend identity -------------------------------------------------- */
    uint32_t backend_id() const override { return 3; /* Metal */ }
    const char* backend_name() const override { return "Metal"; }

    /* -- Device enumeration ------------------------------------------------ */
    uint32_t device_count() const override {
        /* On macOS, Metal always has at least 1 device (system default).
           We rely on MetalRuntime::init() to probe. */
#if defined(__APPLE__)
        return 1; /* Conservative: report system default GPU */
#else
        return 0;
#endif
    }

    GpuError device_info(uint32_t device_index, DeviceInfo& out) const override {
#if defined(__APPLE__)
        if (device_index != 0)
            return GpuError::Device;

        /* Create a temporary runtime to probe device info */
        secp256k1::metal::MetalRuntime tmp;
        if (!tmp.init(0))
            return GpuError::Device;

        auto info = tmp.device_info();
        std::memset(&out, 0, sizeof(out));
        std::snprintf(out.name, sizeof(out.name), "%s", info.name.c_str());
        out.global_mem_bytes      = info.recommended_working_set;
        out.compute_units         = 0; /* Metal API doesn't expose CU count directly */
        out.max_clock_mhz         = 0; /* Not exposed via Metal */
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
    }

    bool is_ready() const override { return runtime_ != nullptr; }

    /* -- Error tracking ---------------------------------------------------- */
    GpuError last_error() const override { return last_err_; }
    const char* last_error_msg() const override { return last_msg_; }

    /* -- First-wave ops ---------------------------------------------------- */

    GpuError generator_mul_batch(
        const uint8_t* scalars32, size_t count,
        uint8_t* out_pubkeys33) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        (void)scalars32; (void)count; (void)out_pubkeys33;
        return set_error(GpuError::Unsupported,
                         "generator_mul_batch not yet wired for Metal");
    }

    GpuError ecdsa_verify_batch(
        const uint8_t*, const uint8_t*, const uint8_t*,
        size_t, uint8_t*) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        return set_error(GpuError::Unsupported,
                         "ECDSA verify batch not yet available on Metal");
    }

    GpuError schnorr_verify_batch(
        const uint8_t*, const uint8_t*, const uint8_t*,
        size_t, uint8_t*) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        return set_error(GpuError::Unsupported,
                         "Schnorr verify batch not yet available on Metal");
    }

    GpuError ecdh_batch(
        const uint8_t*, const uint8_t*,
        size_t, uint8_t*) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        return set_error(GpuError::Unsupported,
                         "ECDH batch not yet available on Metal");
    }

    GpuError hash160_pubkey_batch(
        const uint8_t*, size_t, uint8_t*) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        return set_error(GpuError::Unsupported,
                         "Hash160 batch not yet available on Metal");
    }

    GpuError msm(
        const uint8_t*, const uint8_t*,
        size_t, uint8_t*) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        return set_error(GpuError::Unsupported,
                         "MSM not yet available on Metal");
    }

private:
    std::unique_ptr<secp256k1::metal::MetalRuntime> runtime_;
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
};

/* -- Factory --------------------------------------------------------------- */
std::unique_ptr<GpuBackend> create_metal_backend() {
    return std::make_unique<MetalBackend>();
}

} // namespace gpu
} // namespace secp256k1
