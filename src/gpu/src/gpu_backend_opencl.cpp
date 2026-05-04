/* ============================================================================
 * UltrafastSecp256k1 -- OpenCL Backend Bridge
 * ============================================================================
 * Implements gpu::GpuBackend for OpenCL.
 * Wraps the existing secp256k1::opencl::Context class.
 *
 * Supports all 8 GPU C ABI operations:
 *   - generator_mul_batch  (via batch_scalar_mul_generator + batch_jacobian_to_affine)
 *   - hash160_pubkey_batch (CPU-side SIMD hash160 -- GPU hash kernel not yet wired)
 *   - ecdh_batch           (GPU batch_scalar_mul + CPU SHA-256 finalization)
 *   - msm                  (GPU batch_scalar_mul + CPU-side affine summation)
 *   - ecdsa_verify_batch   (GPU via secp256k1_extended.cl kernel)
 *   - schnorr_verify_batch (GPU via secp256k1_extended.cl kernel)
 *   - frost_verify_partial_batch (GPU via secp256k1_frost.cl kernel)
 *   - ecrecover_batch      (GPU via secp256k1_extended.cl kernel + affine compression)
 *
 * Compiled ONLY when SECP256K1_HAVE_OPENCL is set (via CMake).
 * ============================================================================ */

#include "../include/gpu_backend.hpp"

#include <cstring>
#include <cstdio>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <algorithm>

/* -- OpenCL Context (Layer 1) ---------------------------------------------- */
#include "secp256k1_opencl.hpp"

/* -- Raw OpenCL API for extended kernel loading ----------------------------- */
#ifdef __APPLE__
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

/* -- CPU FieldElement for host-side point compression ---------------------- */
#include "secp256k1/field.hpp"

/* -- CPU GLV decomposition for BIP-352 scan plan precomputation ------------ */
#include "secp256k1/glv.hpp"
#include "secp256k1/scalar.hpp"

/* -- Secure erase for private key zeroization ------------------------------ */
#include "secp256k1/detail/secure_erase.hpp"

/* -- CPU SHA-256 for ECDH finalization ------------------------------------- */
#include "secp256k1/sha256.hpp"

/* -- CPU Hash160 for pubkey hashing ---------------------------------------- */
#include "secp256k1/hash_accel.hpp"

/* -- Helpers --------------------------------------------------------------- */
namespace {

/** Load a file to string, or empty on failure. */
std::string load_file_to_string(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    return {std::istreambuf_iterator<char>(f), {}};
}

struct OpenCLECDSASig {
    uint64_t r[4];
    uint64_t s[4];
};

struct OpenCLBatchScratch {
    std::vector<secp256k1::opencl::Scalar> scalars;
    std::vector<secp256k1::opencl::JacobianPoint> jacobian_points;
    std::vector<secp256k1::opencl::AffinePoint> affine_points;
    std::vector<OpenCLECDSASig> ecdsa_sigs;
    std::vector<int> results;

    void ensure_generator_mul(std::size_t count) {
        if (scalars.size() < count) scalars.resize(count);
        if (jacobian_points.size() < count) jacobian_points.resize(count);
        if (affine_points.size() < count) affine_points.resize(count);
    }

    void ensure_ecdsa_verify(std::size_t count) {
        if (jacobian_points.size() < count) jacobian_points.resize(count);
        if (ecdsa_sigs.size() < count) ecdsa_sigs.resize(count);
        if (results.size() < count) results.resize(count);
    }
};

static thread_local OpenCLBatchScratch g_opencl_batch_scratch;

/* MSM persistent buffer pool — grow-only, avoids per-call clCreateBuffer overhead.
 * Stores scalars, points, Jacobian partials, and block-reduce results on GPU. */
struct OclMsmPool {
    cl_mem buf_scalars  = nullptr;  // n × sizeof(Scalar)
    cl_mem buf_points   = nullptr;  // n × sizeof(AffinePoint)
    cl_mem buf_partials = nullptr;  // n × sizeof(JacobianPoint)  -- scalar_mul output
    cl_mem buf_blocks   = nullptr;  // ceil(n/256) × sizeof(JacobianPoint) -- reduce output
    size_t capacity     = 0;

    void ensure(size_t n, cl_context ctx) {
        if (n <= capacity) return;
        free_all();
        cl_int err;
        const size_t n_blocks = (n + 255) / 256;
        buf_scalars  = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  n        * sizeof(secp256k1::opencl::Scalar),        nullptr, &err);
        buf_points   = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  n        * sizeof(secp256k1::opencl::AffinePoint),   nullptr, &err);
        buf_partials = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n        * sizeof(secp256k1::opencl::JacobianPoint), nullptr, &err);
        buf_blocks   = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n_blocks * sizeof(secp256k1::opencl::JacobianPoint), nullptr, &err);
        if (buf_scalars && buf_points && buf_partials && buf_blocks) { capacity = n; return; }
        free_all();
    }

    void free_all(cl_command_queue queue = nullptr) {
        if (buf_scalars) {
            if (queue) {
                // Zero scalar buffer before release — GPU Guardrail #10
                cl_uchar zero = 0;
                cl_event ev = nullptr;
                clEnqueueFillBuffer(queue, buf_scalars, &zero, sizeof(zero),
                                    0, capacity * sizeof(secp256k1::opencl::Scalar),
                                    0, nullptr, &ev);
                if (ev) { clWaitForEvents(1, &ev); clReleaseEvent(ev); }
            }
            clReleaseMemObject(buf_scalars);
            buf_scalars = nullptr;
        }
        if (buf_points)   { clReleaseMemObject(buf_points);   buf_points   = nullptr; }
        if (buf_partials) {
            // Zero intermediate scalar-mul results before release (Guardrail #10)
            if (queue && capacity > 0) {
                cl_uchar zero = 0;
                cl_event ev = nullptr;
                clEnqueueFillBuffer(queue, buf_partials, &zero, sizeof(zero),
                                    0, capacity * sizeof(secp256k1::opencl::JacobianPoint),
                                    0, nullptr, &ev);
                if (ev) { clWaitForEvents(1, &ev); clReleaseEvent(ev); }
            }
            clReleaseMemObject(buf_partials);
            buf_partials = nullptr;
        }
        if (buf_blocks)   { clReleaseMemObject(buf_blocks);   buf_blocks   = nullptr; }
        capacity = 0;
    }
};

} // anonymous namespace

namespace secp256k1 {
namespace gpu {

class OpenCLBackend final : public GpuBackend {
public:
    OpenCLBackend() = default;
    ~OpenCLBackend() override { shutdown(); }

    /* -- Backend identity -------------------------------------------------- */
    uint32_t backend_id() const override { return 2; /* OpenCL */ }
    const char* backend_name() const override { return "OpenCL"; }

    /* -- Device enumeration ------------------------------------------------ */
    uint32_t device_count() const override {
        auto platforms = secp256k1::opencl::enumerate_devices();
        uint32_t total = 0;
        for (auto& [pname, devs] : platforms)
            total += static_cast<uint32_t>(devs.size());
        return total;
    }

    GpuError device_info(uint32_t device_index, DeviceInfo& out) const override {
        auto platforms = secp256k1::opencl::enumerate_devices();
        uint32_t idx = 0;
        for (auto& [pname, devs] : platforms) {
            for (auto& d : devs) {
                if (idx == device_index) {
                    std::memset(&out, 0, sizeof(out));
                    std::snprintf(out.name, sizeof(out.name), "%s", d.name.c_str());
                    out.global_mem_bytes      = d.global_mem_size;
                    out.compute_units         = d.compute_units;
                    out.max_clock_mhz         = d.max_clock_freq;
                    out.max_threads_per_block = static_cast<uint32_t>(d.max_work_group_size);
                    out.backend_id            = 2;
                    out.device_index          = device_index;
                    return GpuError::Ok;
                }
                ++idx;
            }
        }
        return GpuError::Device;
    }

    /* -- Context lifecycle ------------------------------------------------- */
    GpuError init(uint32_t device_index) override {
        if (ctx_) return GpuError::Ok;

        /* Map flat device_index to (platform_id, device_id) */
        auto platforms = secp256k1::opencl::enumerate_devices();
        uint32_t idx = 0;
        int plat = -1, dev = -1;
        for (int p = 0; p < static_cast<int>(platforms.size()); ++p) {
            for (int d = 0; d < static_cast<int>(platforms[p].second.size()); ++d) {
                if (idx == device_index) { plat = p; dev = d; }
                ++idx;
            }
        }
        if (plat < 0) return set_error(GpuError::Device, "OpenCL device not found");

        secp256k1::opencl::DeviceConfig cfg;
        cfg.platform_id = plat;
        cfg.device_id   = dev;
        cfg.verbose      = false;

        ctx_ = secp256k1::opencl::Context::create(cfg);
        if (!ctx_ || !ctx_->is_valid()) {
            std::string msg = ctx_ ? ctx_->last_error() : "Context creation failed";
            ctx_.reset();
            return set_error(GpuError::Device, msg.c_str());
        }

        clear_error();
        return GpuError::Ok;
    }

    void shutdown() override {
        if (ext_ecdsa_verify_)   { clReleaseKernel(ext_ecdsa_verify_);   ext_ecdsa_verify_   = nullptr; }
        if (ext_ecrecover_)      { clReleaseKernel(ext_ecrecover_);      ext_ecrecover_      = nullptr; }
        if (ext_schnorr_verify_) { clReleaseKernel(ext_schnorr_verify_); ext_schnorr_verify_ = nullptr; }
        if (ext_ecdsa_snark_)    { clReleaseKernel(ext_ecdsa_snark_);    ext_ecdsa_snark_    = nullptr; }
        if (ext_schnorr_snark_)  { clReleaseKernel(ext_schnorr_snark_);  ext_schnorr_snark_  = nullptr; }
        if (ext_program_)        { clReleaseProgram(ext_program_);       ext_program_        = nullptr; }
        ext_init_attempted_ = false;
        if (frost_kernel_)       { clReleaseKernel(frost_kernel_);       frost_kernel_       = nullptr; }
        if (frost_program_)      { clReleaseProgram(frost_program_);     frost_program_      = nullptr; }
        frost_init_attempted_ = false;
        if (zk_knowledge_verify_) { clReleaseKernel(zk_knowledge_verify_); zk_knowledge_verify_ = nullptr; }
        if (zk_dleq_verify_)      { clReleaseKernel(zk_dleq_verify_);     zk_dleq_verify_      = nullptr; }
        if (bp_poly_batch_)       { clReleaseKernel(bp_poly_batch_);       bp_poly_batch_       = nullptr; }
        if (zk_program_)          { clReleaseProgram(zk_program_);         zk_program_          = nullptr; }
        zk_init_attempted_ = false;
        if (bip324_aead_encrypt_) { clReleaseKernel(bip324_aead_encrypt_); bip324_aead_encrypt_ = nullptr; }
        if (bip324_aead_decrypt_) { clReleaseKernel(bip324_aead_decrypt_); bip324_aead_decrypt_ = nullptr; }
        if (bip324_program_)      { clReleaseProgram(bip324_program_);     bip324_program_      = nullptr; }

        if (bip352_scan_kernel_)  { clReleaseKernel(bip352_scan_kernel_);  bip352_scan_kernel_  = nullptr; }
        if (bip352_program_)      { clReleaseProgram(bip352_program_);     bip352_program_      = nullptr; }
        bip324_init_attempted_ = false;
        auto* shutdown_queue = ctx_ ? static_cast<cl_command_queue>(ctx_->native_queue()) : nullptr;
        msm_pool_.free_all(shutdown_queue);
        ctx_.reset();
    }

    bool is_ready() const override { return ctx_ && ctx_->is_valid(); }

    /* -- Error tracking ---------------------------------------------------- */
    GpuError last_error() const override { return last_err_; }
    const char* last_error_msg() const override { return last_msg_; }

    /* -- First-wave ops ---------------------------------------------------- */

    GpuError generator_mul_batch(
        const uint8_t* scalars32, size_t count,
        uint8_t* out_pubkeys33) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!scalars32 || !out_pubkeys33) return set_error(GpuError::NullArg, "NULL buffer");

        /* Convert big-endian bytes → OpenCL Scalar (4×uint64 LE limbs) */
        auto& scratch = g_opencl_batch_scratch;
        scratch.ensure_generator_mul(count);
        auto* const h_scalars = scratch.scalars.data();
        for (size_t i = 0; i < count; ++i) {
            bytes_to_scalar(scalars32 + i * 32, &h_scalars[i]);
            // Reject zero private keys (Guardrail #11)
            if (h_scalars[i].limbs[0] == 0 && h_scalars[i].limbs[1] == 0 &&
                h_scalars[i].limbs[2] == 0 && h_scalars[i].limbs[3] == 0)
                return set_error(GpuError::BadKey, "zero scalar in generator_mul_batch");
        }

        /* Run batch k*G on GPU → Jacobian results */
        auto* const h_jac = scratch.jacobian_points.data();
        ctx_->batch_scalar_mul_generator(h_scalars, h_jac, count);

        /* Convert Jacobian → Affine on GPU */
        auto* const h_aff = scratch.affine_points.data();
        ctx_->batch_jacobian_to_affine(h_jac, h_aff, count);

        /* Compress affine → 33-byte pubkeys on host */
        for (size_t i = 0; i < count; ++i) {
            affine_to_compressed(&h_aff[i], out_pubkeys33 + i * 33);
        }

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

        auto err = ensure_extended_kernels();
        if (err != GpuError::Ok) return err;

        auto* cl_ctx = static_cast<cl_context>(ctx_->native_context());
        auto* queue   = static_cast<cl_command_queue>(ctx_->native_queue());
        cl_int clerr;

        /* Prepare GPU-side buffers ----------------------------------------- */

        /* msg_hashes: 32 bytes each, passed flat */
        cl_mem d_msgs = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       32 * count, const_cast<uint8_t*>(msg_hashes32), &clerr);
        if (clerr != CL_SUCCESS) return set_error(GpuError::Memory, "msg buffer alloc");

        /* pubkeys: decompress 33-byte → full JacobianPoint host layout */
        auto& scratch = g_opencl_batch_scratch;
        scratch.ensure_ecdsa_verify(count);
        auto* const h_pubs = scratch.jacobian_points.data();
        for (size_t i = 0; i < count; ++i) {
            secp256k1::opencl::AffinePoint aff;
            if (!pubkey33_to_affine(pubkeys33 + i * 33, &aff)) {
                clReleaseMemObject(d_msgs);
                return set_error(GpuError::BadKey, "invalid pubkey");
            }
            std::memcpy(h_pubs[i].x.limbs, aff.x.limbs, 32);
            std::memcpy(h_pubs[i].y.limbs, aff.y.limbs, 32);
            std::memset(h_pubs[i].z.limbs, 0, 32);
            h_pubs[i].z.limbs[0] = 1; /* Z = 1 (affine → Jacobian) */
            h_pubs[i].infinity = 0;
        }
        cl_mem d_pubs = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(secp256k1::opencl::JacobianPoint) * count,
                                       h_pubs, &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_msgs);
            return set_error(GpuError::Memory, "pub buffer alloc");
        }

        /* sigs: 64 bytes (r[32] | s[32]) → ECDSASig (r:Scalar, s:Scalar = 64 bytes LE limbs) */
        auto* const h_sigs = scratch.ecdsa_sigs.data();
        for (size_t i = 0; i < count; ++i) {
            be32_to_le_limbs(sigs64 + i * 64,      h_sigs[i].r);
            be32_to_le_limbs(sigs64 + i * 64 + 32, h_sigs[i].s);
        }
        cl_mem d_sigs = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(OpenCLECDSASig) * count, h_sigs, &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_pubs);
            clReleaseMemObject(d_msgs);
            return set_error(GpuError::Memory, "sig buffer alloc");
        }

        /* results: int per item */
        cl_mem d_res = clCreateBuffer(cl_ctx, CL_MEM_WRITE_ONLY,
                                      sizeof(int) * count, nullptr, &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_sigs);
            clReleaseMemObject(d_pubs);
            clReleaseMemObject(d_msgs);
            return set_error(GpuError::Memory, "result buffer alloc");
        }

        cl_uint cl_count = static_cast<cl_uint>(count);
        clSetKernelArg(ext_ecdsa_verify_, 0, sizeof(cl_mem), &d_msgs);
        clSetKernelArg(ext_ecdsa_verify_, 1, sizeof(cl_mem), &d_pubs);
        clSetKernelArg(ext_ecdsa_verify_, 2, sizeof(cl_mem), &d_sigs);
        clSetKernelArg(ext_ecdsa_verify_, 3, sizeof(cl_mem), &d_res);
        clSetKernelArg(ext_ecdsa_verify_, 4, sizeof(cl_uint), &cl_count);

        size_t global = count;
        clerr = clEnqueueNDRangeKernel(queue, ext_ecdsa_verify_, 1, nullptr,
                               &global, nullptr, 0, nullptr, nullptr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_res);
            clReleaseMemObject(d_sigs);
            clReleaseMemObject(d_pubs);
            clReleaseMemObject(d_msgs);
            return set_error(GpuError::Launch, "ecdsa_verify kernel launch failed");
        }
        clFinish(queue);

        /* Read results */
        auto* const h_res = scratch.results.data();
        clEnqueueReadBuffer(queue, d_res, CL_TRUE, 0,
                            sizeof(int) * count, h_res, 0, nullptr, nullptr);

        for (size_t i = 0; i < count; ++i)
            out_results[i] = h_res[i] ? 1 : 0;

        clReleaseMemObject(d_msgs);
        clReleaseMemObject(d_pubs);
        clReleaseMemObject(d_sigs);
        clReleaseMemObject(d_res);

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

        auto err = ensure_extended_kernels();
        if (err != GpuError::Ok) return err;

        auto* cl_ctx = static_cast<cl_context>(ctx_->native_context());
        auto* queue   = static_cast<cl_command_queue>(ctx_->native_queue());
        cl_int clerr;

        /* pubkeys_x: 32 bytes each, passed flat */
        cl_mem d_pks = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      32 * count, const_cast<uint8_t*>(pubkeys_x32), &clerr);
        if (clerr != CL_SUCCESS)
            return set_error(GpuError::Memory, "schnorr pk buffer alloc");

        /* messages: 32 bytes each, passed flat */
        cl_mem d_msgs = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       32 * count, const_cast<uint8_t*>(msg_hashes32), &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_pks);
            return set_error(GpuError::Memory, "schnorr msg buffer alloc");
        }

        /* sigs: 64 bytes (r[32] | s[32]) → SchnorrSig (r:uint8_t[32], s:Scalar = 64 bytes) */
        struct SchnorrSig { uint8_t r[32]; uint64_t s[4]; };
        std::vector<SchnorrSig> h_sigs(count);
        for (size_t i = 0; i < count; ++i) {
            std::memcpy(h_sigs[i].r, sigs64 + i * 64, 32);
            be32_to_le_limbs(sigs64 + i * 64 + 32, h_sigs[i].s);
        }
        cl_mem d_sigs = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(SchnorrSig) * count, h_sigs.data(), &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_msgs);
            clReleaseMemObject(d_pks);
            return set_error(GpuError::Memory, "schnorr sig buffer alloc");
        }

        /* results: int per item */
        cl_mem d_res = clCreateBuffer(cl_ctx, CL_MEM_WRITE_ONLY,
                                      sizeof(int) * count, nullptr, &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_sigs);
            clReleaseMemObject(d_msgs);
            clReleaseMemObject(d_pks);
            return set_error(GpuError::Memory, "schnorr result buffer alloc");
        }

        cl_uint cl_count = static_cast<cl_uint>(count);
        clSetKernelArg(ext_schnorr_verify_, 0, sizeof(cl_mem), &d_pks);
        clSetKernelArg(ext_schnorr_verify_, 1, sizeof(cl_mem), &d_msgs);
        clSetKernelArg(ext_schnorr_verify_, 2, sizeof(cl_mem), &d_sigs);
        clSetKernelArg(ext_schnorr_verify_, 3, sizeof(cl_mem), &d_res);
        clSetKernelArg(ext_schnorr_verify_, 4, sizeof(cl_uint), &cl_count);

        size_t global = count;
        clerr = clEnqueueNDRangeKernel(queue, ext_schnorr_verify_, 1, nullptr,
                               &global, nullptr, 0, nullptr, nullptr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_res);
            clReleaseMemObject(d_sigs);
            clReleaseMemObject(d_msgs);
            clReleaseMemObject(d_pks);
            return set_error(GpuError::Launch, "schnorr_verify kernel launch failed");
        }
        clFinish(queue);

        /* Read results */
        std::vector<int> h_res(count);
        clEnqueueReadBuffer(queue, d_res, CL_TRUE, 0,
                            sizeof(int) * count, h_res.data(), 0, nullptr, nullptr);

        for (size_t i = 0; i < count; ++i)
            out_results[i] = h_res[i] ? 1 : 0;

        clReleaseMemObject(d_pks);
        clReleaseMemObject(d_msgs);
        clReleaseMemObject(d_sigs);
        clReleaseMemObject(d_res);

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

        /* Validate all peer pubkeys BEFORE loading any private key material.
           This ensures no early-return path leaves h_scalars populated
           without a corresponding secure_erase (Rule 10). */
        std::vector<secp256k1::opencl::AffinePoint> h_peers(count);
        for (size_t i = 0; i < count; ++i) {
            if (!pubkey33_to_affine(peer_pubkeys33 + i * 33, &h_peers[i]))
                return set_error(GpuError::BadKey, "invalid peer pubkey");
        }

        /* Load private keys only after all pubkeys are confirmed valid. */
        std::vector<secp256k1::opencl::Scalar> h_scalars(count);
        // RAII guard: erasure is guaranteed even if batch_scalar_mul throws (Guardrail #10)
        struct ScalarEraseGuard {
            std::vector<secp256k1::opencl::Scalar>& v;
            ~ScalarEraseGuard() {
                secp256k1::detail::secure_erase(v.data(),
                                                v.size() * sizeof(v[0]));
            }
        } _scalar_guard{h_scalars};

        for (size_t i = 0; i < count; ++i)
            bytes_to_scalar(privkeys32 + i * 32, &h_scalars[i]);

        /* GPU: batch scalar_mul(priv[i], peer[i]) → Jacobian */
        std::vector<secp256k1::opencl::JacobianPoint> h_jac(count);
        ctx_->batch_scalar_mul(h_scalars.data(), h_peers.data(),
                               h_jac.data(), count);
        // h_scalars erased by _scalar_guard destructor

        /* GPU: Jacobian → Affine */
        std::vector<secp256k1::opencl::AffinePoint> h_aff(count);
        ctx_->batch_jacobian_to_affine(h_jac.data(), h_aff.data(), count);

        /* CPU: SHA-256(compressed shared point) to match ufsecp_ecdh/CUDA. */
        for (size_t i = 0; i < count; ++i) {
            uint8_t compressed[33];
            affine_to_compressed(&h_aff[i], compressed);
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

        /* CPU-side SIMD-accelerated Hash160 */
        for (size_t i = 0; i < count; ++i) {
            secp256k1::hash::hash160_33(pubkeys33 + i * 33,
                                        out_hash160 + i * 20);
        }

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

        auto err = ensure_frost_kernel();
        if (err != GpuError::Ok) return err;

        auto* cl_ctx = static_cast<cl_context>(ctx_->native_context());
        auto* queue  = static_cast<cl_command_queue>(ctx_->native_queue());
        cl_int clerr;

        /* Allocate GPU buffers for all inputs -------------------------------- */
        cl_mem d_z   = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      32 * count, const_cast<uint8_t*>(z_i32), &clerr);
        if (clerr != CL_SUCCESS)
            return set_error(GpuError::Memory, "frost z buffer alloc");

        cl_mem d_D   = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      33 * count, const_cast<uint8_t*>(D_i33), &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_z);
            return set_error(GpuError::Memory, "frost D buffer alloc");
        }

        cl_mem d_E   = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      33 * count, const_cast<uint8_t*>(E_i33), &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_D);
            clReleaseMemObject(d_z);
            return set_error(GpuError::Memory, "frost E buffer alloc");
        }

        cl_mem d_Y   = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      33 * count, const_cast<uint8_t*>(Y_i33), &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_E);
            clReleaseMemObject(d_D);
            clReleaseMemObject(d_z);
            return set_error(GpuError::Memory, "frost Y buffer alloc");
        }

        cl_mem d_rho = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      32 * count, const_cast<uint8_t*>(rho_i32), &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_Y);
            clReleaseMemObject(d_E);
            clReleaseMemObject(d_D);
            clReleaseMemObject(d_z);
            return set_error(GpuError::Memory, "frost rho buffer alloc");
        }

        cl_mem d_lam = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      32 * count, const_cast<uint8_t*>(lambda_ie32), &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_rho);
            clReleaseMemObject(d_Y);
            clReleaseMemObject(d_E);
            clReleaseMemObject(d_D);
            clReleaseMemObject(d_z);
            return set_error(GpuError::Memory, "frost lambda buffer alloc");
        }

        cl_mem d_nR  = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      1 * count, const_cast<uint8_t*>(negate_R), &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_lam);
            clReleaseMemObject(d_rho);
            clReleaseMemObject(d_Y);
            clReleaseMemObject(d_E);
            clReleaseMemObject(d_D);
            clReleaseMemObject(d_z);
            return set_error(GpuError::Memory, "frost nR buffer alloc");
        }

        cl_mem d_nK  = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      1 * count, const_cast<uint8_t*>(negate_key), &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_nR);
            clReleaseMemObject(d_lam);
            clReleaseMemObject(d_rho);
            clReleaseMemObject(d_Y);
            clReleaseMemObject(d_E);
            clReleaseMemObject(d_D);
            clReleaseMemObject(d_z);
            return set_error(GpuError::Memory, "frost nK buffer alloc");
        }

        cl_mem d_res = clCreateBuffer(cl_ctx, CL_MEM_WRITE_ONLY,
                                      sizeof(int) * count, nullptr, &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_nK);
            clReleaseMemObject(d_nR);
            clReleaseMemObject(d_lam);
            clReleaseMemObject(d_rho);
            clReleaseMemObject(d_Y);
            clReleaseMemObject(d_E);
            clReleaseMemObject(d_D);
            clReleaseMemObject(d_z);
            return set_error(GpuError::Memory, "frost result buffer alloc");
        }

        cl_uint cl_count = static_cast<cl_uint>(count);
        clSetKernelArg(frost_kernel_, 0, sizeof(cl_mem),  &d_z);
        clSetKernelArg(frost_kernel_, 1, sizeof(cl_mem),  &d_D);
        clSetKernelArg(frost_kernel_, 2, sizeof(cl_mem),  &d_E);
        clSetKernelArg(frost_kernel_, 3, sizeof(cl_mem),  &d_Y);
        clSetKernelArg(frost_kernel_, 4, sizeof(cl_mem),  &d_rho);
        clSetKernelArg(frost_kernel_, 5, sizeof(cl_mem),  &d_lam);
        clSetKernelArg(frost_kernel_, 6, sizeof(cl_mem),  &d_nR);
        clSetKernelArg(frost_kernel_, 7, sizeof(cl_mem),  &d_nK);
        clSetKernelArg(frost_kernel_, 8, sizeof(cl_mem),  &d_res);
        clSetKernelArg(frost_kernel_, 9, sizeof(cl_uint), &cl_count);

        size_t global = count;
        clerr = clEnqueueNDRangeKernel(queue, frost_kernel_, 1, nullptr,
                               &global, nullptr, 0, nullptr, nullptr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_res);
            clReleaseMemObject(d_nK);
            clReleaseMemObject(d_nR);
            clReleaseMemObject(d_lam);
            clReleaseMemObject(d_rho);
            clReleaseMemObject(d_Y);
            clReleaseMemObject(d_E);
            clReleaseMemObject(d_D);
            clReleaseMemObject(d_z);
            return set_error(GpuError::Launch, "frost_verify kernel launch failed");
        }
        clFinish(queue);

        std::vector<int> h_res(count);
        clEnqueueReadBuffer(queue, d_res, CL_TRUE, 0,
                            sizeof(int) * count, h_res.data(), 0, nullptr, nullptr);

        for (size_t i = 0; i < count; ++i)
            out_results[i] = h_res[i] ? 1 : 0;

        clReleaseMemObject(d_z);
        clReleaseMemObject(d_D);
        clReleaseMemObject(d_E);
        clReleaseMemObject(d_Y);
        clReleaseMemObject(d_rho);
        clReleaseMemObject(d_lam);
        clReleaseMemObject(d_nR);
        clReleaseMemObject(d_nK);
        clReleaseMemObject(d_res);

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

        auto err = ensure_extended_kernels();
        if (err != GpuError::Ok) return err;

        auto* cl_ctx = static_cast<cl_context>(ctx_->native_context());
        auto* queue  = static_cast<cl_command_queue>(ctx_->native_queue());
        cl_int clerr;

        cl_mem d_msgs = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       32 * count, const_cast<uint8_t*>(msg_hashes32), &clerr);
        if (clerr != CL_SUCCESS) return set_error(GpuError::Memory, "msg buffer alloc");

        struct ECDSASig { uint64_t r[4]; uint64_t s[4]; };
        std::vector<ECDSASig> h_sigs(count);
        for (size_t i = 0; i < count; ++i) {
            be32_to_le_limbs(sigs64 + i * 64,      h_sigs[i].r);
            be32_to_le_limbs(sigs64 + i * 64 + 32, h_sigs[i].s);
        }
        cl_mem d_sigs = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(ECDSASig) * count, h_sigs.data(), &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_msgs);
            return set_error(GpuError::Memory, "sig buffer alloc");
        }

        cl_mem d_recids = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         sizeof(int) * count, const_cast<int*>(recids), &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_sigs);
            clReleaseMemObject(d_msgs);
            return set_error(GpuError::Memory, "recid buffer alloc");
        }

        cl_mem d_keys = clCreateBuffer(cl_ctx, CL_MEM_WRITE_ONLY,
                                       sizeof(secp256k1::opencl::JacobianPoint) * count, nullptr, &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_recids);
            clReleaseMemObject(d_sigs);
            clReleaseMemObject(d_msgs);
            return set_error(GpuError::Memory, "key buffer alloc");
        }

        cl_mem d_res = clCreateBuffer(cl_ctx, CL_MEM_WRITE_ONLY,
                                      sizeof(int) * count, nullptr, &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_keys);
            clReleaseMemObject(d_recids);
            clReleaseMemObject(d_sigs);
            clReleaseMemObject(d_msgs);
            return set_error(GpuError::Memory, "result buffer alloc");
        }

        cl_uint cl_count = static_cast<cl_uint>(count);
        clSetKernelArg(ext_ecrecover_, 0, sizeof(cl_mem), &d_msgs);
        clSetKernelArg(ext_ecrecover_, 1, sizeof(cl_mem), &d_sigs);
        clSetKernelArg(ext_ecrecover_, 2, sizeof(cl_mem), &d_recids);
        clSetKernelArg(ext_ecrecover_, 3, sizeof(cl_mem), &d_keys);
        clSetKernelArg(ext_ecrecover_, 4, sizeof(cl_mem), &d_res);
        clSetKernelArg(ext_ecrecover_, 5, sizeof(cl_uint), &cl_count);

        size_t global = count;
        clerr = clEnqueueNDRangeKernel(queue, ext_ecrecover_, 1, nullptr,
                                       &global, nullptr, 0, nullptr, nullptr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_res);
            clReleaseMemObject(d_keys);
            clReleaseMemObject(d_recids);
            clReleaseMemObject(d_sigs);
            clReleaseMemObject(d_msgs);
            return set_error(GpuError::Launch, "ecrecover kernel launch failed");
        }
        clFinish(queue);

        std::vector<secp256k1::opencl::JacobianPoint> h_jac(count);
        std::vector<int> h_res(count);
        clEnqueueReadBuffer(queue, d_keys, CL_TRUE, 0,
                            sizeof(secp256k1::opencl::JacobianPoint) * count,
                            h_jac.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, d_res, CL_TRUE, 0,
                            sizeof(int) * count, h_res.data(), 0, nullptr, nullptr);

        std::vector<secp256k1::opencl::AffinePoint> h_aff(count);
        ctx_->batch_jacobian_to_affine(h_jac.data(), h_aff.data(), count);

        for (size_t i = 0; i < count; ++i) {
            out_valid[i] = h_res[i] ? 1 : 0;
            if (h_res[i]) {
                affine_to_compressed(&h_aff[i], out_pubkeys33 + i * 33);
            } else {
                std::memset(out_pubkeys33 + i * 33, 0, 33);
            }
        }

        clReleaseMemObject(d_res);
        clReleaseMemObject(d_keys);
        clReleaseMemObject(d_recids);
        clReleaseMemObject(d_sigs);
        clReleaseMemObject(d_msgs);
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

        auto* cl_ctx       = static_cast<cl_context>(ctx_->native_context());
        auto* queue        = static_cast<cl_command_queue>(ctx_->native_queue());
        auto* k_scalar_mul = static_cast<cl_kernel>(ctx_->native_kernel("scalar_mul"));
        auto* k_blk_reduce = static_cast<cl_kernel>(ctx_->native_kernel("msm_block_reduce_kernel"));

        if (!k_scalar_mul)
            return set_error(GpuError::Launch, "scalar_mul kernel unavailable");

        /* Convert inputs */
        std::vector<secp256k1::opencl::Scalar>      h_scalars(n);
        std::vector<secp256k1::opencl::AffinePoint> h_points(n);
        for (size_t i = 0; i < n; ++i) {
            bytes_to_scalar(scalars32 + i * 32, &h_scalars[i]);
            if (!pubkey33_to_affine(points33 + i * 33, &h_points[i]))
                return set_error(GpuError::BadKey, "invalid MSM point");
        }

        /* Ensure persistent pool buffers (grow-only) */
        msm_pool_.ensure(n, cl_ctx);
        if (msm_pool_.capacity < n)
            return set_error(GpuError::Device, "MSM pool allocation failed");

        const size_t n_blocks = (n + 255) / 256;

        /* Upload scalars + points to GPU */
        clEnqueueWriteBuffer(queue, msm_pool_.buf_scalars, CL_FALSE, 0,
                             n * sizeof(secp256k1::opencl::Scalar), h_scalars.data(), 0, nullptr, nullptr);
        clEnqueueWriteBuffer(queue, msm_pool_.buf_points,  CL_FALSE, 0,
                             n * sizeof(secp256k1::opencl::AffinePoint), h_points.data(), 0, nullptr, nullptr);
        clFlush(queue);

        /* Dispatch scalar_mul: n threads */
        cl_uint cnt = static_cast<cl_uint>(n);
        clSetKernelArg(k_scalar_mul, 0, sizeof(cl_mem), &msm_pool_.buf_scalars);
        clSetKernelArg(k_scalar_mul, 1, sizeof(cl_mem), &msm_pool_.buf_points);
        clSetKernelArg(k_scalar_mul, 2, sizeof(cl_mem), &msm_pool_.buf_partials);
        clSetKernelArg(k_scalar_mul, 3, sizeof(cl_uint), &cnt);
        size_t local_sm  = 128;
        size_t global_sm = ((n + local_sm - 1) / local_sm) * local_sm;
        clEnqueueNDRangeKernel(queue, k_scalar_mul, 1, nullptr, &global_sm, &local_sm, 0, nullptr, nullptr);

        /* GPU block reduce (or fallback: copy all partials to CPU) */
        std::vector<secp256k1::opencl::JacobianPoint> h_blocks;
        if (k_blk_reduce) {
            /* n work-groups of 256, each reduces 256 partials → 1 block result */
            cl_int n_int = static_cast<cl_int>(n);
            clSetKernelArg(k_blk_reduce, 0, sizeof(cl_mem), &msm_pool_.buf_partials);
            clSetKernelArg(k_blk_reduce, 1, sizeof(cl_int), &n_int);
            clSetKernelArg(k_blk_reduce, 2, sizeof(cl_mem), &msm_pool_.buf_blocks);
            size_t local_r  = 256;
            size_t global_r = n_blocks * 256;
            clEnqueueNDRangeKernel(queue, k_blk_reduce, 1, nullptr, &global_r, &local_r, 0, nullptr, nullptr);
            h_blocks.resize(n_blocks);
            clEnqueueReadBuffer(queue, msm_pool_.buf_blocks, CL_TRUE, 0,
                                n_blocks * sizeof(secp256k1::opencl::JacobianPoint),
                                h_blocks.data(), 0, nullptr, nullptr);
        } else {
            /* Fallback: read all partials (n points) to CPU */
            h_blocks.resize(n);
            clEnqueueReadBuffer(queue, msm_pool_.buf_partials, CL_TRUE, 0,
                                n * sizeof(secp256k1::opencl::JacobianPoint),
                                h_blocks.data(), 0, nullptr, nullptr);
        }

        /* Convert small block results Jacobian → Affine (GPU, tiny batch) */
        std::vector<secp256k1::opencl::AffinePoint> h_aff(h_blocks.size());
        ctx_->batch_jacobian_to_affine(h_blocks.data(), h_aff.data(), h_blocks.size());

        /* CPU: sum affine block results (tiny count when block reduce ran) */
        bool have_acc = false;
        secp256k1::fast::FieldElement acc_x, acc_y;

        for (size_t i = 0; i < h_aff.size(); ++i) {
            std::array<uint64_t, 4> xl, yl;
            std::memcpy(xl.data(), h_aff[i].x.limbs, 32);
            std::memcpy(yl.data(), h_aff[i].y.limbs, 32);
            auto px = secp256k1::fast::FieldElement::from_limbs(xl);
            auto py = secp256k1::fast::FieldElement::from_limbs(yl);

            /* Skip point at infinity (zero x and y) */
            auto pxb = px.to_bytes();
            auto pyb = py.to_bytes();
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
            for (int k = 0; k < 32 && dx_zero; ++k)
                if (dxb[k]) dx_zero = false;

            if (dx_zero) {
                auto dyb = dy.to_bytes();
                bool dy_zero = true;
                for (int k = 0; k < 32 && dy_zero; ++k)
                    if (dyb[k]) dy_zero = false;
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

        /* Serialize result */
        auto yb = acc_y.to_bytes();
        out_result33[0] = (yb[31] & 1) ? 0x03 : 0x02;
        auto xb = acc_x.to_bytes();
        std::memcpy(out_result33 + 1, xb.data(), 32);

        clear_error();
        return GpuError::Ok;
    }

    /* -- ECDSA SNARK witness batch (eprint 2025/695) ------------------------ */

    GpuError snark_witness_batch(
        const uint8_t* msg_hashes32, const uint8_t* pubkeys33,
        const uint8_t* sigs64, size_t count,
        uint8_t* out_flat) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!msg_hashes32 || !pubkeys33 || !sigs64 || !out_flat)
            return set_error(GpuError::NullArg, "NULL buffer");

        auto err = ensure_extended_kernels();
        if (err != GpuError::Ok) return err;

        auto* cl_ctx = static_cast<cl_context>(ctx_->native_context());
        auto* queue   = static_cast<cl_command_queue>(ctx_->native_queue());
        cl_int clerr;

        /* msgs: flat 32-byte hashes */
        cl_mem d_msgs = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       32 * count, const_cast<uint8_t*>(msg_hashes32), &clerr);
        if (clerr != CL_SUCCESS) return set_error(GpuError::Memory, "msg buffer alloc");

        /* pubkeys: decompress 33-byte → JacobianPoint {x,y,z=1} */
        auto& scratch = g_opencl_batch_scratch;
        scratch.ensure_ecdsa_verify(count);
        auto* const h_pubs = scratch.jacobian_points.data();
        for (size_t i = 0; i < count; ++i) {
            secp256k1::opencl::AffinePoint aff;
            if (!pubkey33_to_affine(pubkeys33 + i * 33, &aff)) {
                clReleaseMemObject(d_msgs);
                return set_error(GpuError::BadKey, "invalid pubkey");
            }
            std::memcpy(h_pubs[i].x.limbs, aff.x.limbs, 32);
            std::memcpy(h_pubs[i].y.limbs, aff.y.limbs, 32);
            std::memset(h_pubs[i].z.limbs, 0, 32);
            h_pubs[i].z.limbs[0] = 1; /* Z = 1 */
            h_pubs[i].infinity = 0;
        }
        cl_mem d_pubs = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(secp256k1::opencl::JacobianPoint) * count,
                                       h_pubs, &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_msgs);
            return set_error(GpuError::Memory, "pub buffer alloc");
        }

        /* sigs: BE r|s → ECDSASignature {r:Scalar, s:Scalar} */
        auto* const h_sigs = scratch.ecdsa_sigs.data();
        for (size_t i = 0; i < count; ++i) {
            be32_to_le_limbs(sigs64 + i * 64,      h_sigs[i].r);
            be32_to_le_limbs(sigs64 + i * 64 + 32, h_sigs[i].s);
        }
        cl_mem d_sigs = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(OpenCLECDSASig) * count, h_sigs, &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_pubs);
            clReleaseMemObject(d_msgs);
            return set_error(GpuError::Memory, "sig buffer alloc");
        }

        /* output: count × 760-byte flat witness records */
        constexpr size_t WITNESS_BYTES = 760;
        cl_mem d_out = clCreateBuffer(cl_ctx, CL_MEM_WRITE_ONLY,
                                      WITNESS_BYTES * count, nullptr, &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_sigs);
            clReleaseMemObject(d_pubs);
            clReleaseMemObject(d_msgs);
            return set_error(GpuError::Memory, "output buffer alloc");
        }

        cl_uint cl_count = static_cast<cl_uint>(count);
        clSetKernelArg(ext_ecdsa_snark_, 0, sizeof(cl_mem),  &d_msgs);
        clSetKernelArg(ext_ecdsa_snark_, 1, sizeof(cl_mem),  &d_pubs);
        clSetKernelArg(ext_ecdsa_snark_, 2, sizeof(cl_mem),  &d_sigs);
        clSetKernelArg(ext_ecdsa_snark_, 3, sizeof(cl_mem),  &d_out);
        clSetKernelArg(ext_ecdsa_snark_, 4, sizeof(cl_uint), &cl_count);

        size_t global = count;
        clerr = clEnqueueNDRangeKernel(queue, ext_ecdsa_snark_, 1, nullptr,
                                       &global, nullptr, 0, nullptr, nullptr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_out);
            clReleaseMemObject(d_sigs);
            clReleaseMemObject(d_pubs);
            clReleaseMemObject(d_msgs);
            return set_error(GpuError::Launch, "ecdsa_snark_witness_batch kernel launch failed");
        }
        clFinish(queue);

        clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0,
                            WITNESS_BYTES * count, out_flat, 0, nullptr, nullptr);

        clReleaseMemObject(d_out);
        clReleaseMemObject(d_sigs);
        clReleaseMemObject(d_pubs);
        clReleaseMemObject(d_msgs);
        clear_error();
        return GpuError::Ok;
    }

    /* -- BIP-340 Schnorr SNARK witness GPU batch (eprint 2025/695) ---------- */

    GpuError schnorr_snark_witness_batch(
        const uint8_t* msgs32, const uint8_t* pubkeys_x32,
        const uint8_t* sigs64, size_t count,
        uint8_t* out_flat) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!msgs32 || !pubkeys_x32 || !sigs64 || !out_flat)
            return set_error(GpuError::NullArg, "NULL buffer");

        auto err = ensure_extended_kernels();
        if (err != GpuError::Ok) return err;

        auto* cl_ctx = static_cast<cl_context>(ctx_->native_context());
        auto* queue  = static_cast<cl_command_queue>(ctx_->native_queue());
        cl_int clerr;

        cl_mem d_msgs = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       32 * count, const_cast<uint8_t*>(msgs32), &clerr);
        if (clerr != CL_SUCCESS) return set_error(GpuError::Memory, "msgs buffer alloc");

        cl_mem d_pubs = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       32 * count, const_cast<uint8_t*>(pubkeys_x32), &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_msgs);
            return set_error(GpuError::Memory, "pubkeys buffer alloc");
        }

        cl_mem d_sigs = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       64 * count, const_cast<uint8_t*>(sigs64), &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_pubs);
            clReleaseMemObject(d_msgs);
            return set_error(GpuError::Memory, "sigs buffer alloc");
        }

        constexpr size_t WITNESS_BYTES = 472;
        cl_mem d_out = clCreateBuffer(cl_ctx, CL_MEM_WRITE_ONLY,
                                      WITNESS_BYTES * count, nullptr, &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_sigs);
            clReleaseMemObject(d_pubs);
            clReleaseMemObject(d_msgs);
            return set_error(GpuError::Memory, "output buffer alloc");
        }

        cl_uint cl_count = static_cast<cl_uint>(count);
        clSetKernelArg(ext_schnorr_snark_, 0, sizeof(cl_mem),  &d_msgs);
        clSetKernelArg(ext_schnorr_snark_, 1, sizeof(cl_mem),  &d_pubs);
        clSetKernelArg(ext_schnorr_snark_, 2, sizeof(cl_mem),  &d_sigs);
        clSetKernelArg(ext_schnorr_snark_, 3, sizeof(cl_mem),  &d_out);
        clSetKernelArg(ext_schnorr_snark_, 4, sizeof(cl_uint), &cl_count);

        size_t global = count;
        clerr = clEnqueueNDRangeKernel(queue, ext_schnorr_snark_, 1, nullptr,
                                       &global, nullptr, 0, nullptr, nullptr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_out);
            clReleaseMemObject(d_sigs);
            clReleaseMemObject(d_pubs);
            clReleaseMemObject(d_msgs);
            return set_error(GpuError::Launch, "schnorr_snark_witness_batch kernel launch failed");
        }
        clFinish(queue);

        clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0,
                            WITNESS_BYTES * count, out_flat, 0, nullptr, nullptr);

        clReleaseMemObject(d_out);
        clReleaseMemObject(d_sigs);
        clReleaseMemObject(d_pubs);
        clReleaseMemObject(d_msgs);
        clear_error();
        return GpuError::Ok;
    }

    /* -- ZK proof batch operations (OpenCL via secp256k1_zk.cl) ------------- */

    GpuError zk_knowledge_verify_batch(
        const uint8_t* proofs64, const uint8_t* pubkeys65,
        const uint8_t* messages32, size_t count,
        uint8_t* out_results) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!proofs64 || !pubkeys65 || !messages32 || !out_results)
            return set_error(GpuError::NullArg, "NULL buffer");

        auto err = ensure_zk_kernels();
        if (err != GpuError::Ok) return err;

        auto* cl_ctx = static_cast<cl_context>(ctx_->native_context());
        auto* queue  = static_cast<cl_command_queue>(ctx_->native_queue());
        cl_int clerr;

        /* proofs: 64 bytes → ZKKnowledgeProof { rx[32], Scalar(u64[4]) } */
        struct ZKKnowledgeProofOCL { uint8_t rx[32]; uint64_t s[4]; };
        std::vector<ZKKnowledgeProofOCL> h_proofs(count);
        for (size_t i = 0; i < count; ++i) {
            const uint8_t* p = proofs64 + i * 64;
            std::memcpy(h_proofs[i].rx, p, 32);
            be32_to_le_limbs(p + 32, h_proofs[i].s);
        }

        /* pubkeys: 65-byte uncompressed → JacobianPoint (Z=1) */
        std::vector<secp256k1::opencl::JacobianPoint> h_pubs(count);
        for (size_t i = 0; i < count; ++i) {
            secp256k1::opencl::AffinePoint aff;
            if (!pubkey65_to_affine(pubkeys65 + i * 65, &aff))
                return set_error(GpuError::BadKey, "invalid pubkey");
            affine_to_jacobian(&aff, &h_pubs[i]);
        }

        /* bases: secp256k1 generator G repeated count times */
        secp256k1::opencl::JacobianPoint G_jac = generator_jacobian();
        std::vector<secp256k1::opencl::JacobianPoint> h_bases(count, G_jac);

        cl_mem d_proofs = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         sizeof(ZKKnowledgeProofOCL) * count,
                                         h_proofs.data(), &clerr);
        if (clerr != CL_SUCCESS)
            return set_error(GpuError::Memory, "zk proof buffer alloc");

        cl_mem d_pubs = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        sizeof(secp256k1::opencl::JacobianPoint) * count,
                                        h_pubs.data(), &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_proofs);
            return set_error(GpuError::Memory, "zk pubkey buffer alloc");
        }

        cl_mem d_bases = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         sizeof(secp256k1::opencl::JacobianPoint) * count,
                                         h_bases.data(), &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_pubs); clReleaseMemObject(d_proofs);
            return set_error(GpuError::Memory, "zk bases buffer alloc");
        }

        cl_mem d_msgs = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        32 * count, const_cast<uint8_t*>(messages32), &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_bases); clReleaseMemObject(d_pubs); clReleaseMemObject(d_proofs);
            return set_error(GpuError::Memory, "zk msg buffer alloc");
        }

        cl_mem d_res = clCreateBuffer(cl_ctx, CL_MEM_WRITE_ONLY,
                                       sizeof(int) * count, nullptr, &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_msgs); clReleaseMemObject(d_bases);
            clReleaseMemObject(d_pubs); clReleaseMemObject(d_proofs);
            return set_error(GpuError::Memory, "zk result buffer alloc");
        }

        cl_uint cl_count = static_cast<cl_uint>(count);
        clSetKernelArg(zk_knowledge_verify_, 0, sizeof(cl_mem),  &d_proofs);
        clSetKernelArg(zk_knowledge_verify_, 1, sizeof(cl_mem),  &d_pubs);
        clSetKernelArg(zk_knowledge_verify_, 2, sizeof(cl_mem),  &d_bases);
        clSetKernelArg(zk_knowledge_verify_, 3, sizeof(cl_mem),  &d_msgs);
        clSetKernelArg(zk_knowledge_verify_, 4, sizeof(cl_mem),  &d_res);
        clSetKernelArg(zk_knowledge_verify_, 5, sizeof(cl_uint), &cl_count);

        size_t global = count;
        clerr = clEnqueueNDRangeKernel(queue, zk_knowledge_verify_, 1, nullptr,
                                        &global, nullptr, 0, nullptr, nullptr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_res); clReleaseMemObject(d_msgs);
            clReleaseMemObject(d_bases); clReleaseMemObject(d_pubs); clReleaseMemObject(d_proofs);
            return set_error(GpuError::Launch, "zk_knowledge_verify kernel launch failed");
        }
        clFinish(queue);

        std::vector<int> h_res(count);
        clEnqueueReadBuffer(queue, d_res, CL_TRUE, 0,
                             sizeof(int) * count, h_res.data(), 0, nullptr, nullptr);
        for (size_t i = 0; i < count; ++i)
            out_results[i] = h_res[i] ? 1 : 0;

        clReleaseMemObject(d_res); clReleaseMemObject(d_msgs);
        clReleaseMemObject(d_bases); clReleaseMemObject(d_pubs); clReleaseMemObject(d_proofs);
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

        auto err = ensure_zk_kernels();
        if (err != GpuError::Ok) return err;

        auto* cl_ctx = static_cast<cl_context>(ctx_->native_context());
        auto* queue  = static_cast<cl_command_queue>(ctx_->native_queue());
        cl_int clerr;

        /* proofs: 64 bytes → ZKDLEQProof { Scalar e(u64[4]), Scalar s(u64[4]) } */
        struct ZKDLEQProofOCL { uint64_t e[4]; uint64_t s[4]; };
        std::vector<ZKDLEQProofOCL> h_proofs(count);
        for (size_t i = 0; i < count; ++i) {
            const uint8_t* p = proofs64 + i * 64;
            be32_to_le_limbs(p,      h_proofs[i].e);
            be32_to_le_limbs(p + 32, h_proofs[i].s);
        }

        /* 4 point arrays: 65-byte uncompressed → JacobianPoint (Z=1) */
        std::vector<secp256k1::opencl::JacobianPoint> h_G(count), h_H(count), h_P(count), h_Q(count);
        for (size_t i = 0; i < count; ++i) {
            secp256k1::opencl::AffinePoint aff;
            if (!pubkey65_to_affine(G_pts65 + i * 65, &aff)) return set_error(GpuError::BadKey, "invalid G point");
            affine_to_jacobian(&aff, &h_G[i]);
        }
        for (size_t i = 0; i < count; ++i) {
            secp256k1::opencl::AffinePoint aff;
            if (!pubkey65_to_affine(H_pts65 + i * 65, &aff)) return set_error(GpuError::BadKey, "invalid H point");
            affine_to_jacobian(&aff, &h_H[i]);
        }
        for (size_t i = 0; i < count; ++i) {
            secp256k1::opencl::AffinePoint aff;
            if (!pubkey65_to_affine(P_pts65 + i * 65, &aff)) return set_error(GpuError::BadKey, "invalid P point");
            affine_to_jacobian(&aff, &h_P[i]);
        }
        for (size_t i = 0; i < count; ++i) {
            secp256k1::opencl::AffinePoint aff;
            if (!pubkey65_to_affine(Q_pts65 + i * 65, &aff)) return set_error(GpuError::BadKey, "invalid Q point");
            affine_to_jacobian(&aff, &h_Q[i]);
        }

        size_t jp_sz = sizeof(secp256k1::opencl::JacobianPoint) * count;

        cl_mem d_proofs = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          sizeof(ZKDLEQProofOCL) * count, h_proofs.data(), &clerr);
        if (clerr != CL_SUCCESS) return set_error(GpuError::Memory, "dleq proof buf");

        cl_mem d_G = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     jp_sz, h_G.data(), &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_proofs);
            return set_error(GpuError::Memory, "dleq G buf");
        }

        cl_mem d_H = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     jp_sz, h_H.data(), &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_G); clReleaseMemObject(d_proofs);
            return set_error(GpuError::Memory, "dleq H buf");
        }

        cl_mem d_P = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     jp_sz, h_P.data(), &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_H); clReleaseMemObject(d_G); clReleaseMemObject(d_proofs);
            return set_error(GpuError::Memory, "dleq P buf");
        }

        cl_mem d_Q = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     jp_sz, h_Q.data(), &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_P); clReleaseMemObject(d_H);
            clReleaseMemObject(d_G); clReleaseMemObject(d_proofs);
            return set_error(GpuError::Memory, "dleq Q buf");
        }

        cl_mem d_res = clCreateBuffer(cl_ctx, CL_MEM_WRITE_ONLY,
                                       sizeof(int) * count, nullptr, &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_Q); clReleaseMemObject(d_P);
            clReleaseMemObject(d_H); clReleaseMemObject(d_G); clReleaseMemObject(d_proofs);
            return set_error(GpuError::Memory, "dleq result buf");
        }

        cl_uint cl_count = static_cast<cl_uint>(count);
        clSetKernelArg(zk_dleq_verify_, 0, sizeof(cl_mem),  &d_proofs);
        clSetKernelArg(zk_dleq_verify_, 1, sizeof(cl_mem),  &d_G);
        clSetKernelArg(zk_dleq_verify_, 2, sizeof(cl_mem),  &d_H);
        clSetKernelArg(zk_dleq_verify_, 3, sizeof(cl_mem),  &d_P);
        clSetKernelArg(zk_dleq_verify_, 4, sizeof(cl_mem),  &d_Q);
        clSetKernelArg(zk_dleq_verify_, 5, sizeof(cl_mem),  &d_res);
        clSetKernelArg(zk_dleq_verify_, 6, sizeof(cl_uint), &cl_count);

        size_t global = count;
        clerr = clEnqueueNDRangeKernel(queue, zk_dleq_verify_, 1, nullptr,
                                        &global, nullptr, 0, nullptr, nullptr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_res); clReleaseMemObject(d_Q); clReleaseMemObject(d_P);
            clReleaseMemObject(d_H); clReleaseMemObject(d_G); clReleaseMemObject(d_proofs);
            return set_error(GpuError::Launch, "zk_dleq_verify kernel launch failed");
        }
        clFinish(queue);

        std::vector<int> h_res(count);
        clEnqueueReadBuffer(queue, d_res, CL_TRUE, 0,
                             sizeof(int) * count, h_res.data(), 0, nullptr, nullptr);
        for (size_t i = 0; i < count; ++i)
            out_results[i] = h_res[i] ? 1 : 0;

        clReleaseMemObject(d_res); clReleaseMemObject(d_Q); clReleaseMemObject(d_P);
        clReleaseMemObject(d_H); clReleaseMemObject(d_G); clReleaseMemObject(d_proofs);
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

        auto err = ensure_zk_kernels();
        if (err != GpuError::Ok) return err;

        auto* cl_ctx = static_cast<cl_context>(ctx_->native_context());
        auto* queue  = static_cast<cl_command_queue>(ctx_->native_queue());
        cl_int clerr;

        /* Parse 324-byte proofs into host-side RangeProofPolyGPU struct.
         * Wire layout per proof: 4 × 65-byte uncompressed points (A, S, T1, T2)
         *                      + 2 × 32-byte BE scalars (tau_x, t_hat) = 324 bytes.
         * GPU struct layout: 4 × AffinePoint(64B) + 2 × Scalar(32B) = 320 bytes. */
        struct RangeProofPolyOCL {
            secp256k1::opencl::AffinePoint A, S, T1, T2;
            secp256k1::opencl::Scalar tau_x, t_hat;
        };
        std::vector<RangeProofPolyOCL> h_proofs(count);
        for (size_t i = 0; i < count; ++i) {
            const uint8_t* p = proofs324 + i * 324;
            if (!pubkey65_to_affine(p,       &h_proofs[i].A))  return set_error(GpuError::BadKey, "invalid proof A");
            if (!pubkey65_to_affine(p + 65,  &h_proofs[i].S))  return set_error(GpuError::BadKey, "invalid proof S");
            if (!pubkey65_to_affine(p + 130, &h_proofs[i].T1)) return set_error(GpuError::BadKey, "invalid proof T1");
            if (!pubkey65_to_affine(p + 195, &h_proofs[i].T2)) return set_error(GpuError::BadKey, "invalid proof T2");
            bytes_to_scalar(p + 260, &h_proofs[i].tau_x);
            bytes_to_scalar(p + 292, &h_proofs[i].t_hat);
        }

        std::vector<secp256k1::opencl::AffinePoint> h_commits(count);
        for (size_t i = 0; i < count; ++i) {
            if (!pubkey65_to_affine(commitments65 + i * 65, &h_commits[i]))
                return set_error(GpuError::BadKey, "invalid commitment");
        }

        secp256k1::opencl::AffinePoint h_gen;
        if (!pubkey65_to_affine(H_generator65, &h_gen))
            return set_error(GpuError::BadKey, "invalid H generator");

        cl_mem d_proofs = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         sizeof(RangeProofPolyOCL) * count, h_proofs.data(), &clerr);
        if (clerr != CL_SUCCESS) return set_error(GpuError::Memory, "bp proof buffer");

        cl_mem d_commits = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          sizeof(secp256k1::opencl::AffinePoint) * count,
                                          h_commits.data(), &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_proofs);
            return set_error(GpuError::Memory, "bp commit buffer");
        }

        cl_mem d_hgen = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(secp256k1::opencl::AffinePoint), &h_gen, &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_commits); clReleaseMemObject(d_proofs);
            return set_error(GpuError::Memory, "bp h-gen buffer");
        }

        cl_mem d_res = clCreateBuffer(cl_ctx, CL_MEM_WRITE_ONLY,
                                      sizeof(int) * count, nullptr, &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_hgen); clReleaseMemObject(d_commits); clReleaseMemObject(d_proofs);
            return set_error(GpuError::Memory, "bp result buffer");
        }

        cl_uint cl_count = static_cast<cl_uint>(count);
        clSetKernelArg(bp_poly_batch_, 0, sizeof(cl_mem),  &d_proofs);
        clSetKernelArg(bp_poly_batch_, 1, sizeof(cl_mem),  &d_commits);
        clSetKernelArg(bp_poly_batch_, 2, sizeof(cl_mem),  &d_hgen);
        clSetKernelArg(bp_poly_batch_, 3, sizeof(cl_mem),  &d_res);
        clSetKernelArg(bp_poly_batch_, 4, sizeof(cl_uint), &cl_count);

        size_t global = count;
        clerr = clEnqueueNDRangeKernel(queue, bp_poly_batch_, 1, nullptr,
                                        &global, nullptr, 0, nullptr, nullptr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_res);   clReleaseMemObject(d_hgen);
            clReleaseMemObject(d_commits); clReleaseMemObject(d_proofs);
            return set_error(GpuError::Launch, "bp_poly_batch kernel launch failed");
        }
        clFinish(queue);

        std::vector<int> h_res(count);
        clEnqueueReadBuffer(queue, d_res, CL_TRUE, 0,
                             sizeof(int) * count, h_res.data(), 0, nullptr, nullptr);
        for (size_t i = 0; i < count; ++i)
            out_results[i] = h_res[i] ? 1 : 0;

        clReleaseMemObject(d_res);    clReleaseMemObject(d_hgen);
        clReleaseMemObject(d_commits); clReleaseMemObject(d_proofs);
        clear_error();
        return GpuError::Ok;
    }

    /* -- BIP-324 AEAD batch operations (OpenCL via secp256k1_bip324.cl) ----- */

    GpuError bip324_aead_encrypt_batch(
        const uint8_t* keys32, const uint8_t* nonces12,
        const uint8_t* plaintexts, const uint32_t* sizes,
        uint32_t max_payload, size_t count, uint8_t* wire_out) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        if (count == 0) { clear_error(); return GpuError::Ok; }
        if (!keys32 || !nonces12 || !plaintexts || !sizes || !wire_out)
            return set_error(GpuError::NullArg, "NULL buffer");

        auto err = ensure_bip324_kernels();
        if (err != GpuError::Ok) return err;

        auto* cl_ctx = static_cast<cl_context>(ctx_->native_context());
        auto* queue  = static_cast<cl_command_queue>(ctx_->native_queue());
        cl_int clerr;

        const size_t wire_stride = (size_t)max_payload + 19u; /* BIP324_OVERHEAD = 19 */

        // NF-02: Rule 10 — helper that zeros d_keys before release on ALL exit paths
        auto zero_release_keys = [&](cl_mem buf) {
            cl_uchar z = 0;
            clEnqueueFillBuffer(queue, buf, &z, 1, 0, 32 * count, 0, nullptr, nullptr);
            clFinish(queue);
            clReleaseMemObject(buf);
        };

        cl_mem d_keys = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        32 * count, const_cast<uint8_t*>(keys32), &clerr);
        if (clerr != CL_SUCCESS) return set_error(GpuError::Memory, "bip324 key buf");

        cl_mem d_nonces = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          12 * count, const_cast<uint8_t*>(nonces12), &clerr);
        if (clerr != CL_SUCCESS) {
            zero_release_keys(d_keys);
            return set_error(GpuError::Memory, "bip324 nonce buf");
        }

        cl_mem d_pt = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      (size_t)max_payload * count,
                                      const_cast<uint8_t*>(plaintexts), &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_nonces); zero_release_keys(d_keys);
            return set_error(GpuError::Memory, "bip324 plaintext buf");
        }

        cl_mem d_sizes = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         sizeof(uint32_t) * count,
                                         const_cast<uint32_t*>(sizes), &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_pt); clReleaseMemObject(d_nonces); zero_release_keys(d_keys);
            return set_error(GpuError::Memory, "bip324 sizes buf");
        }

        cl_mem d_wire = clCreateBuffer(cl_ctx, CL_MEM_WRITE_ONLY,
                                        wire_stride * count, nullptr, &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_sizes); clReleaseMemObject(d_pt);
            clReleaseMemObject(d_nonces); zero_release_keys(d_keys);
            return set_error(GpuError::Memory, "bip324 wire_out buf");
        }

        cl_uint cl_max = static_cast<cl_uint>(max_payload);
        cl_int  cl_cnt = static_cast<cl_int>(count);
        clSetKernelArg(bip324_aead_encrypt_, 0, sizeof(cl_mem),  &d_keys);
        clSetKernelArg(bip324_aead_encrypt_, 1, sizeof(cl_mem),  &d_nonces);
        clSetKernelArg(bip324_aead_encrypt_, 2, sizeof(cl_mem),  &d_pt);
        clSetKernelArg(bip324_aead_encrypt_, 3, sizeof(cl_mem),  &d_sizes);
        clSetKernelArg(bip324_aead_encrypt_, 4, sizeof(cl_mem),  &d_wire);
        clSetKernelArg(bip324_aead_encrypt_, 5, sizeof(cl_uint), &cl_max);
        clSetKernelArg(bip324_aead_encrypt_, 6, sizeof(cl_int),  &cl_cnt);

        size_t global = count;
        clerr = clEnqueueNDRangeKernel(queue, bip324_aead_encrypt_, 1, nullptr,
                                        &global, nullptr, 0, nullptr, nullptr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_wire); clReleaseMemObject(d_sizes);
            clReleaseMemObject(d_pt); clReleaseMemObject(d_nonces); zero_release_keys(d_keys);
            return set_error(GpuError::Launch, "bip324_encrypt kernel launch failed");
        }
        clFinish(queue);

        clEnqueueReadBuffer(queue, d_wire, CL_TRUE, 0,
                             wire_stride * count, wire_out, 0, nullptr, nullptr);

        // Rule 10 — zero AES session keys before release (success path)
        clReleaseMemObject(d_wire); clReleaseMemObject(d_sizes);
        clReleaseMemObject(d_pt); clReleaseMemObject(d_nonces); zero_release_keys(d_keys);
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

        auto err = ensure_bip324_kernels();
        if (err != GpuError::Ok) return err;

        auto* cl_ctx = static_cast<cl_context>(ctx_->native_context());
        auto* queue  = static_cast<cl_command_queue>(ctx_->native_queue());
        cl_int clerr;

        const size_t wire_stride = (size_t)max_payload + 19u;

        // NF-02: Rule 10 — helper that zeros d_keys before release on ALL exit paths
        auto zero_release_keys = [&](cl_mem buf) {
            cl_uchar z = 0;
            clEnqueueFillBuffer(queue, buf, &z, 1, 0, 32 * count, 0, nullptr, nullptr);
            clFinish(queue);
            clReleaseMemObject(buf);
        };

        cl_mem d_keys = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        32 * count, const_cast<uint8_t*>(keys32), &clerr);
        if (clerr != CL_SUCCESS) return set_error(GpuError::Memory, "bip324d key buf");

        cl_mem d_nonces = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          12 * count, const_cast<uint8_t*>(nonces12), &clerr);
        if (clerr != CL_SUCCESS) {
            zero_release_keys(d_keys);
            return set_error(GpuError::Memory, "bip324d nonce buf");
        }

        cl_mem d_wire_in = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           wire_stride * count,
                                           const_cast<uint8_t*>(wire_in), &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_nonces); zero_release_keys(d_keys);
            return set_error(GpuError::Memory, "bip324d wire_in buf");
        }

        cl_mem d_sizes = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         sizeof(uint32_t) * count,
                                         const_cast<uint32_t*>(sizes), &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_wire_in); clReleaseMemObject(d_nonces); zero_release_keys(d_keys);
            return set_error(GpuError::Memory, "bip324d sizes buf");
        }

        cl_mem d_pt = clCreateBuffer(cl_ctx, CL_MEM_WRITE_ONLY,
                                      (size_t)max_payload * count, nullptr, &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_sizes); clReleaseMemObject(d_wire_in);
            clReleaseMemObject(d_nonces); zero_release_keys(d_keys);
            return set_error(GpuError::Memory, "bip324d plaintext buf");
        }

        /* ok: kernel writes cl_uint, convert to uint8_t after readback */
        cl_mem d_ok = clCreateBuffer(cl_ctx, CL_MEM_WRITE_ONLY,
                                      sizeof(cl_uint) * count, nullptr, &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_pt); clReleaseMemObject(d_sizes);
            clReleaseMemObject(d_wire_in); clReleaseMemObject(d_nonces); zero_release_keys(d_keys);
            return set_error(GpuError::Memory, "bip324d ok buf");
        }

        cl_uint cl_max = static_cast<cl_uint>(max_payload);
        cl_int  cl_cnt = static_cast<cl_int>(count);
        clSetKernelArg(bip324_aead_decrypt_, 0, sizeof(cl_mem),  &d_keys);
        clSetKernelArg(bip324_aead_decrypt_, 1, sizeof(cl_mem),  &d_nonces);
        clSetKernelArg(bip324_aead_decrypt_, 2, sizeof(cl_mem),  &d_wire_in);
        clSetKernelArg(bip324_aead_decrypt_, 3, sizeof(cl_mem),  &d_sizes);
        clSetKernelArg(bip324_aead_decrypt_, 4, sizeof(cl_mem),  &d_pt);
        clSetKernelArg(bip324_aead_decrypt_, 5, sizeof(cl_mem),  &d_ok);
        clSetKernelArg(bip324_aead_decrypt_, 6, sizeof(cl_uint), &cl_max);
        clSetKernelArg(bip324_aead_decrypt_, 7, sizeof(cl_int),  &cl_cnt);

        size_t global = count;
        clerr = clEnqueueNDRangeKernel(queue, bip324_aead_decrypt_, 1, nullptr,
                                        &global, nullptr, 0, nullptr, nullptr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_ok); clReleaseMemObject(d_pt);
            clReleaseMemObject(d_sizes); clReleaseMemObject(d_wire_in);
            clReleaseMemObject(d_nonces); zero_release_keys(d_keys);
            return set_error(GpuError::Launch, "bip324_decrypt kernel launch failed");
        }
        clFinish(queue);

        clEnqueueReadBuffer(queue, d_pt, CL_TRUE, 0,
                             (size_t)max_payload * count, plaintext_out, 0, nullptr, nullptr);

        std::vector<cl_uint> h_ok(count);
        clEnqueueReadBuffer(queue, d_ok, CL_TRUE, 0,
                             sizeof(cl_uint) * count, h_ok.data(), 0, nullptr, nullptr);
        for (size_t i = 0; i < count; ++i)
            out_valid[i] = h_ok[i] ? 1 : 0;

        // Rule 10 — zero AES session keys before release (success path)
        clReleaseMemObject(d_ok); clReleaseMemObject(d_pt);
        clReleaseMemObject(d_sizes); clReleaseMemObject(d_wire_in);
        clReleaseMemObject(d_nonces); zero_release_keys(d_keys);
        clear_error();
        return GpuError::Ok;
    }

    /* -- BIP-352 Silent Payment GPU batch scan ----------------------------- */

// 5-bit wNAF for a 128-bit GLV sub-scalar (big-endian 32-byte input).
// Shared by bip352_scan_batch (replaces inline lambda) so the algorithm
// lives in exactly one place and is easier to test/audit.
// Output: wnaf[0..129], digits in range [-15..15], trailing zeros possible.
static void bip352_glv_wnaf5(const uint8_t* scalar_be32, int8_t wnaf[130]) {
    uint64_t s[4] = {};
    for (int limb = 0; limb < 4; ++limb) {
        uint64_t v = 0;
        int base = limb * 8;
        for (int i = 0; i < 8; ++i) v = (v << 8) | scalar_be32[base + i];
        s[3 - limb] = v;
    }
    for (int i = 0; i < 130; i++) {
        if (s[0] & 1ULL) {
            int d = (int)(s[0] & 0x1FULL);
            if (d >= 16) {
                d -= 32;
                uint64_t add = (uint64_t)(-d);
                uint64_t prev = s[0]; s[0] += add;
                if (s[0] < prev) { for (int j = 1; j < 4; j++) if (++s[j]) break; }
            } else {
                uint64_t prev = s[0]; s[0] -= (uint64_t)d;
                if (s[0] > prev) { for (int j = 1; j < 4; j++) if (s[j]--) break; }
            }
            wnaf[i] = (int8_t)d;
        } else {
            wnaf[i] = 0;
        }
        s[0] = (s[0] >> 1) | (s[1] << 63);
        s[1] = (s[1] >> 1) | (s[2] << 63);
        s[2] = (s[2] >> 1) | (s[3] << 63);
        s[3] >>= 1;
    }
}

    GpuError bip352_scan_batch(
        const uint8_t  scan_privkey32[32],
        const uint8_t  spend_pubkey33[33],
        const uint8_t* tweak_pubkeys33,
        size_t n_tweaks,
        uint64_t* prefix64_out) override
    {
        if (!is_ready()) return set_error(GpuError::Device, "context not initialised");
        if (n_tweaks == 0) { clear_error(); return GpuError::Ok; }
        if (!scan_privkey32 || !spend_pubkey33 || !tweak_pubkeys33 || !prefix64_out)
            return set_error(GpuError::NullArg, "NULL buffer");

        auto err = ensure_bip352_kernel();
        if (err != GpuError::Ok) return err;

        auto* cl_ctx = static_cast<cl_context>(ctx_->native_context());
        auto* queue   = static_cast<cl_command_queue>(ctx_->native_queue());
        cl_int clerr;

        /* -- 1. Compute GLV wNAF scan plan on CPU -- */
        struct Bip352ScanPlan {
            int8_t  wnaf1[130];
            int8_t  wnaf2[130];
            uint8_t k1_neg;
            uint8_t flip_phi;
            uint8_t pad[2];
        }; // 264 bytes — must match OpenCL BIP352ScanKeyGlv exactly
        static_assert(sizeof(Bip352ScanPlan) == 264, "scan plan size mismatch");

        Bip352ScanPlan plan{};
        {
            using namespace secp256k1::fast;
            Scalar k = Scalar::from_bytes(scan_privkey32);
            if (k.is_zero()) return set_error(GpuError::BadKey, "zero scan key");

            auto decomp       = glv_decompose(k);
            auto k1_bytes     = decomp.k1.to_bytes();
            auto k2_bytes     = decomp.k2.to_bytes();
            plan.k1_neg       = decomp.k1_neg ? 1 : 0;
            plan.flip_phi     = (decomp.k1_neg != decomp.k2_neg) ? 1 : 0;

            bip352_glv_wnaf5(k1_bytes.data(), plan.wnaf1);
            bip352_glv_wnaf5(k2_bytes.data(), plan.wnaf2);

            /* HIGH-4 / HIGH-3: Secure-erase GLV sub-scalars from stack before
             * leaving this scope. k and decomp hold sensitive key material. */
            secp256k1::detail::secure_erase(k1_bytes.data(), k1_bytes.size());
            secp256k1::detail::secure_erase(k2_bytes.data(), k2_bytes.size());
            secp256k1::detail::secure_erase(&decomp, sizeof(decomp));
            secp256k1::detail::secure_erase(&k, sizeof(k));
        }

        /* -- 2. Decompress spend pubkey to OclAffine on CPU -- */
        secp256k1::opencl::AffinePoint ocl_spend{};
        {
            secp256k1::opencl::AffinePoint aff;
            if (!pubkey33_to_affine(spend_pubkey33, &aff))
                return set_error(GpuError::BadKey, "invalid spend pubkey");
            ocl_spend = aff;
        }

        /* -- 3. Decompress tweak pubkeys to OclAffine on CPU -- */
        std::vector<secp256k1::opencl::AffinePoint> ocl_tweaks(n_tweaks);
        for (size_t i = 0; i < n_tweaks; ++i) {
            if (!pubkey33_to_affine(tweak_pubkeys33 + i * 33, &ocl_tweaks[i]))
                return set_error(GpuError::BadKey, "invalid tweak pubkey");
        }

        /* -- 4. Upload buffers to device -- */
        cl_mem d_plan = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(Bip352ScanPlan), &plan, &clerr);
        if (clerr != CL_SUCCESS) return set_error(GpuError::Memory, "scan plan alloc");

        cl_mem d_spend = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        sizeof(secp256k1::opencl::AffinePoint), &ocl_spend, &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_plan);
            return set_error(GpuError::Memory, "spend point alloc");
        }

        cl_mem d_tweaks = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         sizeof(secp256k1::opencl::AffinePoint) * n_tweaks,
                                         ocl_tweaks.data(), &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_spend);
            clReleaseMemObject(d_plan);
            return set_error(GpuError::Memory, "tweak buffer alloc");
        }

        cl_mem d_prefixes = clCreateBuffer(cl_ctx, CL_MEM_WRITE_ONLY,
                                           sizeof(uint64_t) * n_tweaks, nullptr, &clerr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_tweaks);
            clReleaseMemObject(d_spend);
            clReleaseMemObject(d_plan);
            return set_error(GpuError::Memory, "prefix output alloc");
        }

        /* -- 5. Set kernel args and launch -- */
        cl_uint cl_count = static_cast<cl_uint>(n_tweaks);
        clSetKernelArg(bip352_scan_kernel_, 0, sizeof(cl_mem),  &d_tweaks);
        clSetKernelArg(bip352_scan_kernel_, 1, sizeof(cl_mem),  &d_plan);
        clSetKernelArg(bip352_scan_kernel_, 2, sizeof(cl_mem),  &d_spend);
        clSetKernelArg(bip352_scan_kernel_, 3, sizeof(cl_mem),  &d_prefixes);
        clSetKernelArg(bip352_scan_kernel_, 4, sizeof(cl_uint), &cl_count);

        // Query device for preferred work-group size multiple to maximize occupancy.
        // Different GPUs have different optimal local sizes (AMD: 64, NVIDIA: 32/64,
        // Intel: 16/32). Fallback to 128 if query fails.
        size_t local = 128;
        {
            cl_device_id dev = nullptr;
            if (clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,
                                      sizeof(dev), &dev, nullptr) == CL_SUCCESS && dev) {
                size_t pref = 0;
                if (clGetKernelWorkGroupInfo(bip352_scan_kernel_, dev,
                        CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                        sizeof(pref), &pref, nullptr) == CL_SUCCESS && pref > 0) {
                    local = std::max(pref, (size_t)32);
                    local = std::min(local, (size_t)256);
                }
            }
        }
        size_t global = ((n_tweaks + local - 1) / local) * local;
        clerr = clEnqueueNDRangeKernel(queue, bip352_scan_kernel_, 1, nullptr,
                                       &global, &local, 0, nullptr, nullptr);
        if (clerr != CL_SUCCESS) {
            clReleaseMemObject(d_prefixes);
            clReleaseMemObject(d_tweaks);
            clReleaseMemObject(d_spend);
            clReleaseMemObject(d_plan);
            return set_error(GpuError::Launch, "bip352_pipeline_kernel launch failed");
        }
        clFinish(queue);

        /* -- 6. Read back prefixes -- */
        clEnqueueReadBuffer(queue, d_prefixes, CL_TRUE, 0,
                            sizeof(uint64_t) * n_tweaks, prefix64_out, 0, nullptr, nullptr);

        /* HIGH-3: Zero the wNAF scan plan buffer on the device before releasing
         * it. The plan contains derived key material (GLV sub-scalars + wNAF
         * window digits) that could otherwise persist in GPU memory. */
        {
            static const cl_uchar zero = 0;
            clEnqueueFillBuffer(queue, d_plan, &zero, sizeof(zero), 0,
                                sizeof(Bip352ScanPlan), 0, nullptr, nullptr);
            clFinish(queue);
        }

        /* Also zero the host-side plan struct. */
        secp256k1::detail::secure_erase(&plan, sizeof(plan));

        clReleaseMemObject(d_prefixes);
        clReleaseMemObject(d_tweaks);
        clReleaseMemObject(d_spend);
        clReleaseMemObject(d_plan);
        clear_error();
        return GpuError::Ok;
    }

private:
    std::unique_ptr<secp256k1::opencl::Context> ctx_;
    GpuError last_err_ = GpuError::Ok;
    char     last_msg_[256] = {};

    /* Extended kernel handles (lazy-loaded for verify ops) */
    cl_program ext_program_         = nullptr;
    cl_kernel  ext_ecdsa_verify_    = nullptr;
    cl_kernel  ext_schnorr_verify_  = nullptr;
    cl_kernel  ext_ecrecover_       = nullptr;
    cl_kernel  ext_ecdsa_snark_        = nullptr;
    cl_kernel  ext_schnorr_snark_      = nullptr;
    bool       ext_init_attempted_  = false;

    /* FROST kernel handles (lazy-loaded) */
    cl_program frost_program_       = nullptr;
    cl_kernel  frost_kernel_        = nullptr;
    bool       frost_init_attempted_ = false;

    /* ZK proof kernel handles (lazy-loaded via secp256k1_zk.cl) */
    cl_program zk_program_            = nullptr;
    cl_kernel  zk_knowledge_verify_   = nullptr;
    cl_kernel  zk_dleq_verify_        = nullptr;
    cl_kernel  bp_poly_batch_         = nullptr;  /* range_proof_poly_batch */
    bool       zk_init_attempted_     = false;

    /* BIP-324 AEAD kernel handles (lazy-loaded via secp256k1_bip324.cl) */
    cl_program bip324_program_        = nullptr;
    cl_kernel  bip324_aead_encrypt_   = nullptr;
    cl_kernel  bip324_aead_decrypt_   = nullptr;
    bool       bip324_init_attempted_ = false;

    /* BIP-352 Silent Payment scan kernel (lazy-loaded via secp256k1_bip352.cl) */
    cl_program bip352_program_        = nullptr;
    cl_kernel  bip352_scan_kernel_    = nullptr;
    bool       bip352_init_attempted_ = false;

    /* MSM persistent buffer pool (shared across msm() calls) */
    OclMsmPool msm_pool_;

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

    /* -- Big-endian 32 bytes → 4×uint64 LE limbs -------------------------- */
    static void be32_to_le_limbs(const uint8_t be[32], uint64_t out[4]) {
        for (int limb = 0; limb < 4; ++limb) {
            uint64_t v = 0;
            int base = (3 - limb) * 8;
            for (int b = 0; b < 8; ++b)
                v = (v << 8) | be[base + b];
            out[limb] = v;
        }
    }

    /* -- Lazy-load extended OpenCL program for verify kernels -------------- */
    GpuError ensure_extended_kernels() {
        if (ext_ecdsa_verify_ && ext_schnorr_verify_ && ext_ecrecover_ && ext_ecdsa_snark_ && ext_schnorr_snark_) return GpuError::Ok;
        if (ext_init_attempted_)
            return set_error(GpuError::Launch, "extended kernel init previously failed");
        ext_init_attempted_ = true;

        auto* cl_ctx = static_cast<cl_context>(ctx_->native_context());

        /* Get device from context */
        cl_device_id device = nullptr;
        clGetContextInfo(cl_ctx, CL_CONTEXT_DEVICES, sizeof(device), &device, nullptr);
        if (!device)
            return set_error(GpuError::Launch, "no OpenCL device found in context");

        /* Search for secp256k1_extended.cl */
        const char* search_paths[] = {
            "../../opencl/kernels/secp256k1_extended.cl",
            "../opencl/kernels/secp256k1_extended.cl",
            "../../../opencl/kernels/secp256k1_extended.cl",
            "opencl/kernels/secp256k1_extended.cl",
            "kernels/secp256k1_extended.cl",
            "../kernels/secp256k1_extended.cl",
        };

        std::string src;
        std::string kernel_dir;
        for (auto* p : search_paths) {
            src = load_file_to_string(p);
            if (!src.empty()) {
                std::filesystem::path fp(p);
                kernel_dir = fp.parent_path().string();
                break;
            }
        }
        if (src.empty())
            return set_error(GpuError::Launch, "secp256k1_extended.cl not found");

        /* Compile */
        const char* src_ptr = src.c_str();
        size_t src_len = src.size();
        cl_int err;
        ext_program_ = clCreateProgramWithSource(cl_ctx, 1, &src_ptr, &src_len, &err);
        if (err != CL_SUCCESS)
            return set_error(GpuError::Launch, "clCreateProgramWithSource failed");

        std::string opts = "-cl-std=CL1.2 -cl-fast-relaxed-math -cl-mad-enable";
        if (!kernel_dir.empty())
            opts += " -I " + kernel_dir;

        err = clBuildProgram(ext_program_, 1, &device, opts.c_str(), nullptr, nullptr);
        if (err != CL_SUCCESS) {
            /* Grab build log for diagnostics */
            size_t log_len = 0;
            clGetProgramBuildInfo(ext_program_, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_len);
            std::string log(log_len, '\0');
            clGetProgramBuildInfo(ext_program_, device, CL_PROGRAM_BUILD_LOG, log_len, log.data(), nullptr);
            clReleaseProgram(ext_program_);
            ext_program_ = nullptr;
            std::string msg = "extended.cl build failed: " + log;
            return set_error(GpuError::Launch, msg.c_str());
        }

        ext_ecdsa_verify_  = clCreateKernel(ext_program_, "ecdsa_verify", &err);
        if (err != CL_SUCCESS) {
            clReleaseProgram(ext_program_); ext_program_ = nullptr;
            return set_error(GpuError::Launch, "ecdsa_verify kernel not found");
        }

        ext_schnorr_verify_ = clCreateKernel(ext_program_, "schnorr_verify", &err);
        if (err != CL_SUCCESS) {
            clReleaseKernel(ext_ecdsa_verify_); ext_ecdsa_verify_ = nullptr;
            clReleaseProgram(ext_program_); ext_program_ = nullptr;
            return set_error(GpuError::Launch, "schnorr_verify kernel not found");
        }

        ext_ecrecover_ = clCreateKernel(ext_program_, "ecrecover_batch", &err);
        if (err != CL_SUCCESS) {
            clReleaseKernel(ext_schnorr_verify_); ext_schnorr_verify_ = nullptr;
            clReleaseKernel(ext_ecdsa_verify_); ext_ecdsa_verify_ = nullptr;
            clReleaseProgram(ext_program_); ext_program_ = nullptr;
            return set_error(GpuError::Launch, "ecrecover_batch kernel not found");
        }

        ext_ecdsa_snark_ = clCreateKernel(ext_program_, "ecdsa_snark_witness_batch", &err);
        if (err != CL_SUCCESS) {
            clReleaseKernel(ext_ecrecover_);      ext_ecrecover_      = nullptr;
            clReleaseKernel(ext_schnorr_verify_); ext_schnorr_verify_ = nullptr;
            clReleaseKernel(ext_ecdsa_verify_);   ext_ecdsa_verify_   = nullptr;
            clReleaseProgram(ext_program_);       ext_program_        = nullptr;
            return set_error(GpuError::Launch, "ecdsa_snark_witness_batch kernel not found");
        }

        ext_schnorr_snark_ = clCreateKernel(ext_program_, "schnorr_snark_witness_batch", &err);
        if (err != CL_SUCCESS) {
            clReleaseKernel(ext_ecdsa_snark_);    ext_ecdsa_snark_    = nullptr;
            clReleaseKernel(ext_ecrecover_);      ext_ecrecover_      = nullptr;
            clReleaseKernel(ext_schnorr_verify_); ext_schnorr_verify_ = nullptr;
            clReleaseKernel(ext_ecdsa_verify_);   ext_ecdsa_verify_   = nullptr;
            clReleaseProgram(ext_program_);       ext_program_        = nullptr;
            return set_error(GpuError::Launch, "schnorr_snark_witness_batch kernel not found");
        }

        return GpuError::Ok;
    }

    /* -- Lazy-load FROST OpenCL program ------------------------------------- */
    GpuError ensure_frost_kernel() {
        if (frost_kernel_) return GpuError::Ok;
        if (frost_init_attempted_)
            return set_error(GpuError::Launch, "FROST kernel init previously failed");
        frost_init_attempted_ = true;

        auto* cl_ctx = static_cast<cl_context>(ctx_->native_context());
        cl_device_id device = nullptr;
        clGetContextInfo(cl_ctx, CL_CONTEXT_DEVICES, sizeof(device), &device, nullptr);
        if (!device)
            return set_error(GpuError::Launch, "no OpenCL device found in context");

        const char* search_paths[] = {
            "../../opencl/kernels/secp256k1_frost.cl",
            "../opencl/kernels/secp256k1_frost.cl",
            "../../../opencl/kernels/secp256k1_frost.cl",
            "opencl/kernels/secp256k1_frost.cl",
            "kernels/secp256k1_frost.cl",
            "../kernels/secp256k1_frost.cl",
        };

        std::string src;
        std::string kernel_dir;
        for (auto* p : search_paths) {
            src = load_file_to_string(p);
            if (!src.empty()) {
                std::filesystem::path fp(p);
                kernel_dir = fp.parent_path().string();
                break;
            }
        }
        if (src.empty())
            return set_error(GpuError::Launch, "secp256k1_frost.cl not found");

        const char* src_ptr = src.c_str();
        size_t src_len = src.size();
        cl_int err;
        frost_program_ = clCreateProgramWithSource(cl_ctx, 1, &src_ptr, &src_len, &err);
        if (err != CL_SUCCESS)
            return set_error(GpuError::Launch, "frost clCreateProgramWithSource failed");

        std::string opts = "-cl-std=CL1.2 -cl-fast-relaxed-math -cl-mad-enable";
        if (!kernel_dir.empty())
            opts += " -I " + kernel_dir;

        err = clBuildProgram(frost_program_, 1, &device, opts.c_str(), nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_len = 0;
            clGetProgramBuildInfo(frost_program_, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_len);
            std::string log(log_len, '\0');
            clGetProgramBuildInfo(frost_program_, device, CL_PROGRAM_BUILD_LOG, log_len, log.data(), nullptr);
            clReleaseProgram(frost_program_);
            frost_program_ = nullptr;
            std::string msg = "frost.cl build failed: " + log;
            return set_error(GpuError::Launch, msg.c_str());
        }

        frost_kernel_ = clCreateKernel(frost_program_, "frost_verify_partial", &err);
        if (err != CL_SUCCESS) {
            clReleaseProgram(frost_program_); frost_program_ = nullptr;
            return set_error(GpuError::Launch, "frost_verify_partial kernel not found");
        }

        return GpuError::Ok;
    }

    /* -- Type conversion helpers ------------------------------------------- */

    static void bytes_to_scalar(const uint8_t be[32],
                                secp256k1::opencl::Scalar* out) {
        for (int limb = 0; limb < 4; ++limb) {
            uint64_t v = 0;
            int base = (3 - limb) * 8; /* big-endian: limb 0 → bytes[24..31] */
            for (int b = 0; b < 8; ++b) {
                v = (v << 8) | be[base + b];
            }
            out->limbs[limb] = v;
        }
    }

    static void affine_to_compressed(const secp256k1::opencl::AffinePoint* p,
                                     uint8_t out[33]) {
        /* Convert OpenCL limbs → CPU FieldElement for safe serialisation */
        std::array<uint64_t, 4> xl, yl;
        std::memcpy(xl.data(), p->x.limbs, 32);
        std::memcpy(yl.data(), p->y.limbs, 32);
        auto cx = secp256k1::fast::FieldElement::from_limbs(xl);
        auto cy = secp256k1::fast::FieldElement::from_limbs(yl);

        auto ybytes = cy.to_bytes();
        out[0] = (ybytes[31] & 1) ? 0x03 : 0x02;
        auto xbytes = cx.to_bytes();
        std::memcpy(out + 1, xbytes.data(), 32);
    }

    /** Decompress a 33-byte compressed pubkey to OpenCL AffinePoint.
     *  Returns false if prefix is invalid.                               */
    static bool pubkey33_to_affine(const uint8_t pub[33],
                                   secp256k1::opencl::AffinePoint* out) {
        uint8_t prefix = pub[0];
        if (prefix != 0x02 && prefix != 0x03) return false;

        /* x from big-endian bytes */
        secp256k1::fast::FieldElement fe_x;
        if (!secp256k1::fast::FieldElement::parse_bytes_strict(pub + 1, fe_x))
            return false;

        /* y^2 = x^3 + 7 */
        auto x2 = fe_x * fe_x;
        auto x3 = x2 * fe_x;
        auto y2 = x3 + secp256k1::fast::FieldElement::from_uint64(7);
        auto fe_y = y2.sqrt();

        /* Validate: sqrt must satisfy y² == x³+7 (not all field elements have a square root) */
        if ((fe_y * fe_y) != y2) return false;

        /* Choose correct parity */
        auto yb = fe_y.to_bytes();
        if ((yb[31] & 1) != (prefix & 1))
            fe_y = fe_y.negate();

        /* Store as LE limbs into OpenCL AffinePoint */
        const auto& xl = fe_x.limbs();
        const auto& yl = fe_y.limbs();
        std::memcpy(out->x.limbs, xl.data(), 32);
        std::memcpy(out->y.limbs, yl.data(), 32);
        return true;
    }

    /** Decompress a 65-byte uncompressed pubkey (04 || x[32] || y[32]) to OpenCL AffinePoint.
     *  Validates y² == x³+7. Returns false for invalid inputs. */
    static bool pubkey65_to_affine(const uint8_t pub65[65],
                                    secp256k1::opencl::AffinePoint* out) {
        if (pub65[0] != 0x04) return false;
        secp256k1::fast::FieldElement fe_x, fe_y;
        if (!secp256k1::fast::FieldElement::parse_bytes_strict(pub65 + 1,  fe_x)) return false;
        if (!secp256k1::fast::FieldElement::parse_bytes_strict(pub65 + 33, fe_y)) return false;
        /* Validate point is on curve: y² == x³ + 7 */
        auto x2  = fe_x * fe_x;
        auto x3  = x2  * fe_x;
        auto y2  = fe_y * fe_y;
        auto rhs = x3  + secp256k1::fast::FieldElement::from_uint64(7);
        if (y2 != rhs) return false;
        const auto& xl = fe_x.limbs();
        const auto& yl = fe_y.limbs();
        std::memcpy(out->x.limbs, xl.data(), 32);
        std::memcpy(out->y.limbs, yl.data(), 32);
        return true;
    }

    /** Lift an AffinePoint to a JacobianPoint with Z=1. */
    static void affine_to_jacobian(const secp256k1::opencl::AffinePoint* aff,
                                    secp256k1::opencl::JacobianPoint* j) {
        std::memcpy(j->x.limbs, aff->x.limbs, 32);
        std::memcpy(j->y.limbs, aff->y.limbs, 32);
        std::memset(j->z.limbs, 0, 32);
        j->z.limbs[0] = 1; /* Z = 1 (affine lift) */
        j->infinity   = 0;
    }

    /** Return the secp256k1 generator G as a JacobianPoint. */
    static secp256k1::opencl::JacobianPoint generator_jacobian() {
        static const uint8_t G33[33] = {
            0x02,
            0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC,
            0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B, 0x07,
            0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9,
            0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8, 0x17, 0x98
        };
        secp256k1::opencl::AffinePoint aff;
        pubkey33_to_affine(G33, &aff);
        secp256k1::opencl::JacobianPoint j{};
        affine_to_jacobian(&aff, &j);
        return j;
    }

    /* -- Lazy-load ZK proof OpenCL program --------------------------------- */
    GpuError ensure_zk_kernels() {
        if (zk_knowledge_verify_ && zk_dleq_verify_ && bp_poly_batch_) return GpuError::Ok;
        if (zk_init_attempted_)
            return set_error(GpuError::Launch, "ZK kernel init previously failed");
        zk_init_attempted_ = true;

        auto* cl_ctx = static_cast<cl_context>(ctx_->native_context());
        cl_device_id device = nullptr;
        clGetContextInfo(cl_ctx, CL_CONTEXT_DEVICES, sizeof(device), &device, nullptr);
        if (!device)
            return set_error(GpuError::Launch, "no OpenCL device found in context");

        const char* search_paths[] = {
            "../../opencl/kernels/secp256k1_zk.cl",
            "../opencl/kernels/secp256k1_zk.cl",
            "../../../opencl/kernels/secp256k1_zk.cl",
            "opencl/kernels/secp256k1_zk.cl",
            "kernels/secp256k1_zk.cl",
            "../kernels/secp256k1_zk.cl",
        };

        std::string src, kernel_dir;
        for (auto* p : search_paths) {
            src = load_file_to_string(p);
            if (!src.empty()) {
                std::filesystem::path fp(p);
                kernel_dir = fp.parent_path().string();
                break;
            }
        }
        if (src.empty())
            return set_error(GpuError::Launch, "secp256k1_zk.cl not found");

        const char* src_ptr = src.c_str();
        size_t src_len = src.size();
        cl_int err;
        zk_program_ = clCreateProgramWithSource(cl_ctx, 1, &src_ptr, &src_len, &err);
        if (err != CL_SUCCESS)
            return set_error(GpuError::Launch, "zk clCreateProgramWithSource failed");

        std::string opts = "-cl-std=CL1.2 -cl-fast-relaxed-math -cl-mad-enable";
        if (!kernel_dir.empty()) opts += " -I " + kernel_dir;

        err = clBuildProgram(zk_program_, 1, &device, opts.c_str(), nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_len = 0;
            clGetProgramBuildInfo(zk_program_, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_len);
            std::string log(log_len, '\0');
            clGetProgramBuildInfo(zk_program_, device, CL_PROGRAM_BUILD_LOG, log_len, log.data(), nullptr);
            clReleaseProgram(zk_program_); zk_program_ = nullptr;
            std::string msg = "zk.cl build failed: " + log;
            return set_error(GpuError::Launch, msg.c_str());
        }

        zk_knowledge_verify_ = clCreateKernel(zk_program_, "zk_knowledge_verify_batch", &err);
        if (err != CL_SUCCESS) {
            clReleaseProgram(zk_program_); zk_program_ = nullptr;
            return set_error(GpuError::Launch, "zk_knowledge_verify_batch kernel not found");
        }

        zk_dleq_verify_ = clCreateKernel(zk_program_, "zk_dleq_verify_batch", &err);
        if (err != CL_SUCCESS) {
            clReleaseKernel(zk_knowledge_verify_); zk_knowledge_verify_ = nullptr;
            clReleaseProgram(zk_program_); zk_program_ = nullptr;
            return set_error(GpuError::Launch, "zk_dleq_verify_batch kernel not found");
        }

        bp_poly_batch_ = clCreateKernel(zk_program_, "range_proof_poly_batch", &err);
        if (err != CL_SUCCESS) {
            clReleaseKernel(zk_dleq_verify_);     zk_dleq_verify_     = nullptr;
            clReleaseKernel(zk_knowledge_verify_); zk_knowledge_verify_ = nullptr;
            clReleaseProgram(zk_program_); zk_program_ = nullptr;
            return set_error(GpuError::Launch, "range_proof_poly_batch kernel not found");
        }

        return GpuError::Ok;
    }

    /* -- Lazy-load BIP-324 AEAD OpenCL program ----------------------------- */
    GpuError ensure_bip324_kernels() {
        if (bip324_aead_encrypt_ && bip324_aead_decrypt_) return GpuError::Ok;
        if (bip324_init_attempted_)
            return set_error(GpuError::Launch, "BIP-324 kernel init previously failed");
        bip324_init_attempted_ = true;

        auto* cl_ctx = static_cast<cl_context>(ctx_->native_context());
        cl_device_id device = nullptr;
        clGetContextInfo(cl_ctx, CL_CONTEXT_DEVICES, sizeof(device), &device, nullptr);
        if (!device)
            return set_error(GpuError::Launch, "no OpenCL device found in context");

        const char* search_paths[] = {
            "../../opencl/kernels/secp256k1_bip324.cl",
            "../opencl/kernels/secp256k1_bip324.cl",
            "../../../opencl/kernels/secp256k1_bip324.cl",
            "opencl/kernels/secp256k1_bip324.cl",
            "kernels/secp256k1_bip324.cl",
            "../kernels/secp256k1_bip324.cl",
        };

        std::string src, kernel_dir;
        for (auto* p : search_paths) {
            src = load_file_to_string(p);
            if (!src.empty()) {
                std::filesystem::path fp(p);
                kernel_dir = fp.parent_path().string();
                break;
            }
        }
        if (src.empty())
            return set_error(GpuError::Launch, "secp256k1_bip324.cl not found");

        const char* src_ptr = src.c_str();
        size_t src_len = src.size();
        cl_int err;
        bip324_program_ = clCreateProgramWithSource(cl_ctx, 1, &src_ptr, &src_len, &err);
        if (err != CL_SUCCESS)
            return set_error(GpuError::Launch, "bip324 clCreateProgramWithSource failed");

        std::string opts = "-cl-std=CL1.2 -cl-fast-relaxed-math -cl-mad-enable";
        if (!kernel_dir.empty()) opts += " -I " + kernel_dir;

        err = clBuildProgram(bip324_program_, 1, &device, opts.c_str(), nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_len = 0;
            clGetProgramBuildInfo(bip324_program_, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_len);
            std::string log(log_len, '\0');
            clGetProgramBuildInfo(bip324_program_, device, CL_PROGRAM_BUILD_LOG, log_len, log.data(), nullptr);
            clReleaseProgram(bip324_program_); bip324_program_ = nullptr;
            std::string msg = "bip324.cl build failed: " + log;
            return set_error(GpuError::Launch, msg.c_str());
        }

        bip324_aead_encrypt_ = clCreateKernel(bip324_program_, "kernel_bip324_aead_encrypt", &err);
        if (err != CL_SUCCESS) {
            clReleaseProgram(bip324_program_); bip324_program_ = nullptr;
            return set_error(GpuError::Launch, "kernel_bip324_aead_encrypt not found");
        }

        bip324_aead_decrypt_ = clCreateKernel(bip324_program_, "kernel_bip324_aead_decrypt", &err);
        if (err != CL_SUCCESS) {
            clReleaseKernel(bip324_aead_encrypt_); bip324_aead_encrypt_ = nullptr;
            clReleaseProgram(bip324_program_); bip324_program_ = nullptr;
            return set_error(GpuError::Launch, "kernel_bip324_aead_decrypt not found");
        }

        return GpuError::Ok;
    }

    /* -- Lazy-load BIP-352 Silent Payment scan kernel ---------------------- */
    GpuError ensure_bip352_kernel() {
        if (bip352_scan_kernel_) return GpuError::Ok;
        if (bip352_init_attempted_)
            return set_error(GpuError::Launch, "BIP-352 kernel init previously failed");
        bip352_init_attempted_ = true;

        auto* cl_ctx = static_cast<cl_context>(ctx_->native_context());
        cl_device_id device = nullptr;
        clGetContextInfo(cl_ctx, CL_CONTEXT_DEVICES, sizeof(device), &device, nullptr);
        if (!device)
            return set_error(GpuError::Launch, "no OpenCL device found in context");

        const char* search_paths[] = {
            "../../opencl/kernels/secp256k1_bip352.cl",
            "../opencl/kernels/secp256k1_bip352.cl",
            "../../../opencl/kernels/secp256k1_bip352.cl",
            "opencl/kernels/secp256k1_bip352.cl",
            "kernels/secp256k1_bip352.cl",
            "../kernels/secp256k1_bip352.cl",
        };

        /* The BIP-352 kernel includes other .cl files via #include.
         * We must expand those includes before creating the program. */
        std::string src, kernel_dir;
        for (auto* p : search_paths) {
            src = load_file_to_string(p);
            if (!src.empty()) {
                std::filesystem::path fp(p);
                kernel_dir = fp.parent_path().string();
                break;
            }
        }
        if (src.empty())
            return set_error(GpuError::Launch, "secp256k1_bip352.cl not found");

        const char* src_ptr = src.c_str();
        size_t src_len = src.size();
        cl_int err;
        bip352_program_ = clCreateProgramWithSource(cl_ctx, 1, &src_ptr, &src_len, &err);
        if (err != CL_SUCCESS)
            return set_error(GpuError::Launch, "bip352 clCreateProgramWithSource failed");

        std::string opts = "-cl-std=CL1.2 -cl-fast-relaxed-math -cl-mad-enable";
        if (!kernel_dir.empty()) opts += " -I " + kernel_dir;

        err = clBuildProgram(bip352_program_, 1, &device, opts.c_str(), nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_len = 0;
            clGetProgramBuildInfo(bip352_program_, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_len);
            std::string log(log_len, '\0');
            clGetProgramBuildInfo(bip352_program_, device, CL_PROGRAM_BUILD_LOG, log_len, log.data(), nullptr);
            clReleaseProgram(bip352_program_); bip352_program_ = nullptr;
            std::string msg = "bip352.cl build failed: " + log;
            return set_error(GpuError::Launch, msg.c_str());
        }

        bip352_scan_kernel_ = clCreateKernel(bip352_program_, "bip352_pipeline_kernel", &err);
        if (err != CL_SUCCESS) {
            clReleaseProgram(bip352_program_); bip352_program_ = nullptr;
            return set_error(GpuError::Launch, "bip352_pipeline_kernel not found");
        }

        return GpuError::Ok;
    }
};

/* -- Factory --------------------------------------------------------------- */
std::unique_ptr<GpuBackend> create_opencl_backend() {
    return std::make_unique<OpenCLBackend>();
}

} // namespace gpu
} // namespace secp256k1
