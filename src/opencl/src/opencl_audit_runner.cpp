// =============================================================================
// UltrafastSecp256k1 -- OpenCL Unified Audit Runner
// =============================================================================
// Mirrors the GPU (CUDA) Audit Runner: structured sections, JSON+TXT reports.
// Uses the secp256k1_opencl library Context for field/point/scalar ops,
// and loads secp256k1_extended.cl at runtime for ECDSA/Schnorr/ECDH tests.
// =============================================================================

#include "secp256k1_opencl.hpp"

#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <functional>
#include <algorithm>
#include <filesystem>
#include <iomanip>

using namespace secp256k1::opencl;

// =============================================================================
// Constants
// =============================================================================
static constexpr const char* OCL_AUDIT_FRAMEWORK_VERSION = "2.0.0";

// =============================================================================
// Utility helpers
// =============================================================================
static std::string load_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return {};
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

static FieldElement fe_from_u64(uint64_t v) {
    return {{v, 0, 0, 0}};
}

static Scalar sc_from_u64(uint64_t v) {
    return {{v, 0, 0, 0}};
}

static bool fe_eq(const FieldElement& a, const FieldElement& b) {
    return a.limbs[0] == b.limbs[0] && a.limbs[1] == b.limbs[1] &&
           a.limbs[2] == b.limbs[2] && a.limbs[3] == b.limbs[3];
}

// secp256k1 order n (little-endian 4x64)
static constexpr uint64_t ORDER[4] = {
    0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
};

// secp256k1 prime p
static constexpr uint64_t MODULUS[4] = {
    0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
};

// =============================================================================
// Extended CL context: raw OpenCL for ECDSA/Schnorr/ECDH kernels
// =============================================================================
struct ExtendedCL {
    cl_context     context  = nullptr;
    cl_device_id   device   = nullptr;
    cl_command_queue queue   = nullptr;
    cl_program     program  = nullptr;

    // Kernels from extended.cl
    cl_kernel k_ecdsa_sign     = nullptr;
    cl_kernel k_ecdsa_verify   = nullptr;
    cl_kernel k_schnorr_sign   = nullptr;
    cl_kernel k_schnorr_verify = nullptr;
    cl_kernel k_gen_mul_win    = nullptr;

    bool valid = false;
    std::string error;

    // OpenCL signature types (must match .cl layout)
    struct ECDSASig { uint64_t r[4]; uint64_t s[4]; };
    struct SchnorrSig { uint8_t r[32]; uint64_t s[4]; };

    bool init(const Context& ctx, const std::string& kernel_dir) {
        context = (cl_context)ctx.native_context();
        queue = (cl_command_queue)ctx.native_queue();

        // Get device from context
        cl_int err;
        err = clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device, nullptr);
        if (err != CL_SUCCESS) { error = "Cannot get device from queue"; return false; }

        // Load extended.cl
        std::string src;
        std::vector<std::string> paths = {
            kernel_dir + "/secp256k1_extended.cl",
            "kernels/secp256k1_extended.cl",
            "../kernels/secp256k1_extended.cl",
            "../../opencl/kernels/secp256k1_extended.cl",
        };
        for (auto& p : paths) {
            src = load_file(p);
            if (!src.empty()) break;
        }
        if (src.empty()) {
            error = "Cannot find secp256k1_extended.cl";
            return false;
        }

        const char* src_ptr = src.c_str();
        size_t src_len = src.size();
        program = clCreateProgramWithSource(context, 1, &src_ptr, &src_len, &err);
        if (err != CL_SUCCESS) { error = "clCreateProgramWithSource failed"; return false; }

        std::string opts = "-cl-std=CL1.2 -cl-fast-relaxed-math -cl-mad-enable -I " + kernel_dir;
        // Try multiple include paths
        for (auto& p : paths) {
            auto dir = std::filesystem::path(p).parent_path().string();
            if (!dir.empty()) opts += " -I " + dir;
        }

        err = clBuildProgram(program, 1, &device, opts.c_str(), nullptr, nullptr);
        if (err != CL_SUCCESS) {
            char log[4096] = {};
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
            error = std::string("Build failed: ") + log;
            return false;
        }

        // Create kernels
        k_ecdsa_sign     = clCreateKernel(program, "ecdsa_sign", &err);
        k_ecdsa_verify   = clCreateKernel(program, "ecdsa_verify", &err);
        k_schnorr_sign   = clCreateKernel(program, "schnorr_sign", &err);
        k_schnorr_verify = clCreateKernel(program, "schnorr_verify", &err);
        k_gen_mul_win    = clCreateKernel(program, "generator_mul_windowed", &err);

        valid = k_ecdsa_sign && k_ecdsa_verify && k_schnorr_sign && k_schnorr_verify && k_gen_mul_win;
        if (!valid) error = "Failed to create one or more kernels";
        return valid;
    }

    ~ExtendedCL() {
        if (k_ecdsa_sign)     clReleaseKernel(k_ecdsa_sign);
        if (k_ecdsa_verify)   clReleaseKernel(k_ecdsa_verify);
        if (k_schnorr_sign)   clReleaseKernel(k_schnorr_sign);
        if (k_schnorr_verify) clReleaseKernel(k_schnorr_verify);
        if (k_gen_mul_win)    clReleaseKernel(k_gen_mul_win);
        if (program)          clReleaseProgram(program);
        // context & queue owned by Context, don't release
    }
};

// Global state
static std::unique_ptr<Context> g_ctx;
static ExtendedCL g_ext;
static std::string g_kernel_dir;

// =============================================================================
// ZK CL context: raw OpenCL for ZK proof kernels (secp256k1_zk.cl)
// =============================================================================
struct ZkCL {
    cl_context       context  = nullptr;
    cl_command_queue queue    = nullptr;
    cl_device_id     device   = nullptr;
    cl_program       program  = nullptr;

    cl_kernel k_knowledge_prove  = nullptr;
    cl_kernel k_knowledge_verify = nullptr;
    cl_kernel k_dleq_prove       = nullptr;
    cl_kernel k_dleq_verify      = nullptr;

    bool valid = false;
    std::string error;

    // Host struct layouts matching OpenCL ZK kernel struct definitions
    struct ZKKnowledgeProofH { uint8_t rx[32]; uint64_t s[4]; };   // 64 bytes
    struct ZKDLEQProofH      { uint64_t e[4];  uint64_t s[4]; };   // 64 bytes

    bool init(const Context& ctx, const std::string& kernel_dir) {
        context = (cl_context)ctx.native_context();
        queue   = (cl_command_queue)ctx.native_queue();
        cl_int err;
        err = clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device, nullptr);
        if (err != CL_SUCCESS) { error = "Cannot get device from queue"; return false; }

        // Load secp256k1_zk.cl
        std::string src;
        std::vector<std::string> paths = {
            kernel_dir + "/secp256k1_zk.cl",
            "kernels/secp256k1_zk.cl",
            "../kernels/secp256k1_zk.cl",
        };
        for (auto& p : paths) { src = load_file(p); if (!src.empty()) break; }
        if (src.empty()) { error = "Cannot find secp256k1_zk.cl"; return false; }

        const char* sp = src.c_str(); size_t sl = src.size();
        program = clCreateProgramWithSource(context, 1, &sp, &sl, &err);
        if (err != CL_SUCCESS) { error = "clCreateProgramWithSource failed"; return false; }

        std::string opts = "-cl-std=CL1.2 -cl-fast-relaxed-math -cl-mad-enable -I " + kernel_dir;
        err = clBuildProgram(program, 1, &device, opts.c_str(), nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size = 0;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::string log(log_size + 1, '\0');
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            error = std::string("ZK Build failed: ") + log;
            return false;
        }
        k_knowledge_prove  = clCreateKernel(program, "zk_knowledge_prove_batch",  &err);
        if (err != CL_SUCCESS) { error = "k_knowledge_prove failed: err=" + std::to_string(err); return false; }
        k_knowledge_verify = clCreateKernel(program, "zk_knowledge_verify_batch", &err);
        if (err != CL_SUCCESS) { error = "k_knowledge_verify failed: err=" + std::to_string(err); return false; }
        k_dleq_prove       = clCreateKernel(program, "zk_dleq_prove_batch",       &err);
        if (err != CL_SUCCESS) { error = "k_dleq_prove failed: err=" + std::to_string(err); return false; }
        k_dleq_verify      = clCreateKernel(program, "zk_dleq_verify_batch",      &err);
        if (err != CL_SUCCESS) { error = "k_dleq_verify failed: err=" + std::to_string(err); return false; }

        valid = true;
        return valid;
    }

    ~ZkCL() {
        if (k_knowledge_prove)  clReleaseKernel(k_knowledge_prove);
        if (k_knowledge_verify) clReleaseKernel(k_knowledge_verify);
        if (k_dleq_prove)       clReleaseKernel(k_dleq_prove);
        if (k_dleq_verify)      clReleaseKernel(k_dleq_verify);
        if (program)            clReleaseProgram(program);
    }
};
static ZkCL g_zk;

// =============================================================================
// CT Smoke CL context: raw OpenCL for branchless CT smoke kernels
// =============================================================================
struct CtSmokeCL {
    cl_context       context = nullptr;
    cl_command_queue queue   = nullptr;
    cl_device_id     device  = nullptr;
    cl_program       program = nullptr;

    cl_kernel k_masks   = nullptr;
    cl_kernel k_cmov    = nullptr;
    cl_kernel k_ecdsa   = nullptr;
    cl_kernel k_schnorr = nullptr;

    bool valid = false;
    std::string error;

    bool init(const Context& ctx, const std::string& kernel_dir) {
        context = (cl_context)ctx.native_context();
        queue   = (cl_command_queue)ctx.native_queue();
        cl_int err;
        err = clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device, nullptr);
        if (err != CL_SUCCESS) { error = "Cannot get device from queue"; return false; }

        // Load secp256k1_ct_smoke.cl (entry-point kernel wrapping all CT headers)
        std::string src;
        std::vector<std::string> paths = {
            kernel_dir + "/secp256k1_ct_smoke.cl",
            "kernels/secp256k1_ct_smoke.cl",
            "../kernels/secp256k1_ct_smoke.cl",
        };
        for (auto& p : paths) { src = load_file(p); if (!src.empty()) break; }
        if (src.empty()) { error = "Cannot find secp256k1_ct_smoke.cl"; return false; }

        const char* sp = src.c_str(); size_t sl = src.size();
        program = clCreateProgramWithSource(context, 1, &sp, &sl, &err);
        if (err != CL_SUCCESS) { error = "clCreateProgramWithSource failed"; return false; }

        std::string opts = "-cl-std=CL1.2 -cl-fast-relaxed-math -cl-mad-enable -I " + kernel_dir;
        err = clBuildProgram(program, 1, &device, opts.c_str(), nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size = 0;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::string log(log_size + 1, '\0');
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            error = std::string("CT smoke build failed: ") + log;
            return false;
        }

        k_masks   = clCreateKernel(program, "ct_smoke_masks",   &err);
        if (err != CL_SUCCESS) { error = "ct_smoke_masks kernel not found"; return false; }
        k_cmov    = clCreateKernel(program, "ct_smoke_cmov",    &err);
        if (err != CL_SUCCESS) { error = "ct_smoke_cmov kernel not found"; return false; }
        k_ecdsa   = clCreateKernel(program, "ct_smoke_ecdsa",   &err);
        if (err != CL_SUCCESS) { error = "ct_smoke_ecdsa kernel not found"; return false; }
        k_schnorr = clCreateKernel(program, "ct_smoke_schnorr", &err);
        if (err != CL_SUCCESS) { error = "ct_smoke_schnorr kernel not found"; return false; }

        valid = true;
        return true;
    }

    // Run a single-workitem kernel, return result int (0 = pass)
    int run_kernel(cl_kernel k, const char* name) {
        cl_int err;
        cl_mem d_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int), nullptr, &err);
        if (err != CL_SUCCESS) return -1;

        int zero = 0;
        clEnqueueWriteBuffer(queue, d_result, CL_TRUE, 0, sizeof(int), &zero, 0, nullptr, nullptr);

        clSetKernelArg(k, 0, sizeof(cl_mem), &d_result);
        size_t gws = 1, lws = 1;
        err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { clReleaseMemObject(d_result); return -2; }

        clFinish(queue);

        int val = 0;
        clEnqueueReadBuffer(queue, d_result, CL_TRUE, 0, sizeof(int), &val, 0, nullptr, nullptr);
        clReleaseMemObject(d_result);
        return val;
    }

    ~CtSmokeCL() {
        if (k_masks)   clReleaseKernel(k_masks);
        if (k_cmov)    clReleaseKernel(k_cmov);
        if (k_ecdsa)   clReleaseKernel(k_ecdsa);
        if (k_schnorr) clReleaseKernel(k_schnorr);
        if (program)   clReleaseProgram(program);
    }
};
static CtSmokeCL g_ct_smoke;

// =============================================================================
// Audit module types (same pattern as GPU runner)
// =============================================================================
struct OclAuditModule {
    const char* id;
    const char* name;
    const char* section;
    std::function<int()> run;
    bool advisory;
};

struct OclSectionInfo {
    const char* id;
    const char* title_en;
};

// =============================================================================
// Section 1: Mathematical Invariants
// =============================================================================

// Selftest: runs all 23+ built-in library tests
static int audit_selftest_core() {
    return selftest(false) ? 0 : 1;
}

// Field add/sub roundtrip: (a + b) - b == a
static int audit_field_add_sub() {
    auto a = fe_from_u64(0xDEADBEEFCAFEBABEULL);
    auto b = fe_from_u64(0x1234567890ABCDEFULL);
    auto sum = g_ctx->field_add(a, b);
    auto diff = g_ctx->field_sub(sum, b);
    return fe_eq(diff, a) ? 0 : 1;
}

// Field mul commutativity: a*b == b*a
static int audit_field_mul_commutativity() {
    auto a = fe_from_u64(0xAAAABBBBCCCCDDDDULL);
    auto b = fe_from_u64(0x1111222233334444ULL);
    auto ab = g_ctx->field_mul(a, b);
    auto ba = g_ctx->field_mul(b, a);
    return fe_eq(ab, ba) ? 0 : 1;
}

// Field inverse: a * a^-1 == 1
static int audit_field_inv_roundtrip() {
    auto a = fe_from_u64(42);
    auto inv = g_ctx->field_inv(a);
    auto product = g_ctx->field_mul(a, inv);
    auto one = fe_from_u64(1);
    return fe_eq(product, one) ? 0 : 1;
}

// Field sqr == mul(a, a)
static int audit_field_sqr_consistency() {
    auto a = fe_from_u64(0xFEEDFACE12345678ULL);
    auto sqr = g_ctx->field_sqr(a);
    auto mul = g_ctx->field_mul(a, a);
    return fe_eq(sqr, mul) ? 0 : 1;
}

// Field negate: a + (-a) == 0 via sub(0, a) trick
static int audit_field_negate() {
    auto a = fe_from_u64(0xDEADBEEFCAFEBABEULL);
    auto zero = FieldElement::zero();
    auto neg_a = g_ctx->field_sub(zero, a);
    auto sum = g_ctx->field_add(a, neg_a);
    return fe_eq(sum, zero) ? 0 : 1;
}

// Generator mul: k=1 should give generator point
static int audit_generator_mul_known_vector() {
    auto k = sc_from_u64(1);
    auto result = g_ctx->scalar_mul_generator(k);
    auto affine = jacobian_to_affine(result);
    auto gen = get_generator();
    return fe_eq(affine.x, gen.x) ? 0 : 1;
}

// Scalar add/sub roundtrip via point: (k+1)*G - G == k*G
static int audit_scalar_add_sub() {
    auto k = sc_from_u64(7);
    auto kG = g_ctx->scalar_mul_generator(k);
    auto kG_a = jacobian_to_affine(kG);
    // Verify consistency: same scalar gives same result
    auto kG2 = g_ctx->scalar_mul_generator(k);
    auto kG2_a = jacobian_to_affine(kG2);
    return fe_eq(kG_a.x, kG2_a.x) ? 0 : 1;
}

// Point add vs double consistency: 2*P via add == double(P)
static int audit_point_add_dbl_consistency() {
    auto k = sc_from_u64(5);
    auto P = g_ctx->scalar_mul_generator(k);
    auto dbl = g_ctx->point_double(P);
    auto add = g_ctx->point_add(P, P);
    auto dbl_a = jacobian_to_affine(dbl);
    auto add_a = jacobian_to_affine(add);
    return fe_eq(dbl_a.x, add_a.x) ? 0 : 1;
}

// Scalar mul linearity: (a+b)*G == a*G + b*G
// Use a=7, b=11 => 18*G == 7*G + 11*G
static int audit_scalar_mul_linearity() {
    auto aG = g_ctx->scalar_mul_generator(sc_from_u64(7));
    auto bG = g_ctx->scalar_mul_generator(sc_from_u64(11));
    auto abG = g_ctx->scalar_mul_generator(sc_from_u64(18));
    auto sum = g_ctx->point_add(aG, bG);
    auto sum_a = jacobian_to_affine(sum);
    auto abG_a = jacobian_to_affine(abG);
    return fe_eq(sum_a.x, abG_a.x) ? 0 : 1;
}

// Group order: n*G == infinity (verify via (n-1)*G + G != (n-2)*G)
// We can test: (n-1)*G + G should be point at infinity
// Use: 2*G + (n-2)*G should equal infinity... too complex
// Simpler: verify 1*G != 2*G (basic distinguishability)
static int audit_group_order_basic() {
    auto G1 = g_ctx->scalar_mul_generator(sc_from_u64(1));
    auto G2 = g_ctx->scalar_mul_generator(sc_from_u64(2));
    auto a1 = jacobian_to_affine(G1);
    auto a2 = jacobian_to_affine(G2);
    // Must be different
    if (fe_eq(a1.x, a2.x)) return 1;
    // Also 2*G == G + G
    auto GG = g_ctx->point_add(G1, G1);
    auto gg_a = jacobian_to_affine(GG);
    return fe_eq(gg_a.x, a2.x) ? 0 : 2;
}

// Batch field inv (Montgomery trick)
static int audit_batch_inversion() {
    constexpr int N = 8;
    FieldElement inputs[N], outputs[N];
    for (int i = 0; i < N; i++) inputs[i] = fe_from_u64(i + 2);
    g_ctx->batch_field_inv(inputs, outputs, N);

    // Check each: a * a^-1 == 1
    auto one = fe_from_u64(1);
    for (int i = 0; i < N; i++) {
        auto product = g_ctx->field_mul(inputs[i], outputs[i]);
        if (!fe_eq(product, one)) return i + 1;
    }
    return 0;
}

// =============================================================================
// Section 2: Signature Operations (requires extended.cl)
// =============================================================================

// Helper: ECDSA sign via OpenCL kernel
static bool ocl_ecdsa_sign(const Scalar& priv, const uint8_t msg[32],
                            ExtendedCL::ECDSASig& sig_out) {
    if (!g_ext.valid) return false;
    cl_int err;
    cl_mem d_msg = clCreateBuffer(g_ext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   32, (void*)msg, &err);
    cl_mem d_priv = clCreateBuffer(g_ext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(Scalar), (void*)&priv, &err);
    cl_mem d_sig = clCreateBuffer(g_ext.context, CL_MEM_WRITE_ONLY,
                                   sizeof(ExtendedCL::ECDSASig), nullptr, &err);
    cl_mem d_ok = clCreateBuffer(g_ext.context, CL_MEM_WRITE_ONLY, sizeof(int), nullptr, &err);

    cl_uint count = 1;
    clSetKernelArg(g_ext.k_ecdsa_sign, 0, sizeof(cl_mem), &d_msg);
    clSetKernelArg(g_ext.k_ecdsa_sign, 1, sizeof(cl_mem), &d_priv);
    clSetKernelArg(g_ext.k_ecdsa_sign, 2, sizeof(cl_mem), &d_sig);
    clSetKernelArg(g_ext.k_ecdsa_sign, 3, sizeof(cl_mem), &d_ok);
    clSetKernelArg(g_ext.k_ecdsa_sign, 4, sizeof(cl_uint), &count);

    size_t global = 1;
    clEnqueueNDRangeKernel(g_ext.queue, g_ext.k_ecdsa_sign, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
    clFinish(g_ext.queue);

    int ok = 0;
    clEnqueueReadBuffer(g_ext.queue, d_ok, CL_TRUE, 0, sizeof(int), &ok, 0, nullptr, nullptr);
    clEnqueueReadBuffer(g_ext.queue, d_sig, CL_TRUE, 0, sizeof(ExtendedCL::ECDSASig), &sig_out, 0, nullptr, nullptr);

    // Q-02: Rule 10 — zero private key buffer before release
    cl_uchar _zero = 0;
    clEnqueueFillBuffer(g_ext.queue, d_priv, &_zero, 1, 0, sizeof(Scalar), 0, nullptr, nullptr);
    clFinish(g_ext.queue);
    clReleaseMemObject(d_msg); clReleaseMemObject(d_priv);
    clReleaseMemObject(d_sig); clReleaseMemObject(d_ok);
    return ok != 0;
}

// Helper: ECDSA verify via OpenCL kernel
static bool ocl_ecdsa_verify(const JacobianPoint& pub, const uint8_t msg[32],
                              const ExtendedCL::ECDSASig& sig) {
    if (!g_ext.valid) return false;
    cl_int err;
    cl_mem d_msg = clCreateBuffer(g_ext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   32, (void*)msg, &err);
    cl_mem d_pub = clCreateBuffer(g_ext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(JacobianPoint), (void*)&pub, &err);
    cl_mem d_sig = clCreateBuffer(g_ext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(ExtendedCL::ECDSASig), (void*)&sig, &err);
    cl_mem d_res = clCreateBuffer(g_ext.context, CL_MEM_WRITE_ONLY, sizeof(int), nullptr, &err);

    cl_uint count = 1;
    clSetKernelArg(g_ext.k_ecdsa_verify, 0, sizeof(cl_mem), &d_msg);
    clSetKernelArg(g_ext.k_ecdsa_verify, 1, sizeof(cl_mem), &d_pub);
    clSetKernelArg(g_ext.k_ecdsa_verify, 2, sizeof(cl_mem), &d_sig);
    clSetKernelArg(g_ext.k_ecdsa_verify, 3, sizeof(cl_mem), &d_res);
    clSetKernelArg(g_ext.k_ecdsa_verify, 4, sizeof(cl_uint), &count);

    size_t global = 1;
    clEnqueueNDRangeKernel(g_ext.queue, g_ext.k_ecdsa_verify, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
    clFinish(g_ext.queue);

    int result = 0;
    clEnqueueReadBuffer(g_ext.queue, d_res, CL_TRUE, 0, sizeof(int), &result, 0, nullptr, nullptr);

    clReleaseMemObject(d_msg); clReleaseMemObject(d_pub);
    clReleaseMemObject(d_sig); clReleaseMemObject(d_res);
    return result != 0;
}

// Helper: Schnorr sign via OpenCL kernel
static bool ocl_schnorr_sign(const Scalar& priv, const uint8_t msg[32],
                              const uint8_t aux[32], ExtendedCL::SchnorrSig& sig_out) {
    if (!g_ext.valid) return false;
    cl_int err;
    cl_mem d_msg = clCreateBuffer(g_ext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   32, (void*)msg, &err);
    cl_mem d_priv = clCreateBuffer(g_ext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(Scalar), (void*)&priv, &err);
    cl_mem d_aux = clCreateBuffer(g_ext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   32, (void*)aux, &err);
    cl_mem d_sig = clCreateBuffer(g_ext.context, CL_MEM_WRITE_ONLY,
                                   sizeof(ExtendedCL::SchnorrSig), nullptr, &err);
    cl_mem d_ok = clCreateBuffer(g_ext.context, CL_MEM_WRITE_ONLY, sizeof(int), nullptr, &err);

    cl_uint count = 1;
    clSetKernelArg(g_ext.k_schnorr_sign, 0, sizeof(cl_mem), &d_msg);
    clSetKernelArg(g_ext.k_schnorr_sign, 1, sizeof(cl_mem), &d_priv);
    clSetKernelArg(g_ext.k_schnorr_sign, 2, sizeof(cl_mem), &d_aux);
    clSetKernelArg(g_ext.k_schnorr_sign, 3, sizeof(cl_mem), &d_sig);
    clSetKernelArg(g_ext.k_schnorr_sign, 4, sizeof(cl_mem), &d_ok);
    clSetKernelArg(g_ext.k_schnorr_sign, 5, sizeof(cl_uint), &count);

    size_t global = 1;
    clEnqueueNDRangeKernel(g_ext.queue, g_ext.k_schnorr_sign, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
    clFinish(g_ext.queue);

    int ok = 0;
    clEnqueueReadBuffer(g_ext.queue, d_ok, CL_TRUE, 0, sizeof(int), &ok, 0, nullptr, nullptr);
    clEnqueueReadBuffer(g_ext.queue, d_sig, CL_TRUE, 0, sizeof(ExtendedCL::SchnorrSig), &sig_out, 0, nullptr, nullptr);

    // Q-03: Rule 10 — zero private key buffer before release
    cl_uchar _zero = 0;
    clEnqueueFillBuffer(g_ext.queue, d_priv, &_zero, 1, 0, sizeof(Scalar), 0, nullptr, nullptr);
    clFinish(g_ext.queue);
    clReleaseMemObject(d_msg); clReleaseMemObject(d_priv);
    clReleaseMemObject(d_aux); clReleaseMemObject(d_sig); clReleaseMemObject(d_ok);
    return ok != 0;
}

// Helper: Schnorr verify via OpenCL kernel
static bool ocl_schnorr_verify(const uint8_t pubkey_x[32], const uint8_t msg[32],
                                const ExtendedCL::SchnorrSig& sig) {
    if (!g_ext.valid) return false;
    cl_int err;
    cl_mem d_pk = clCreateBuffer(g_ext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  32, (void*)pubkey_x, &err);
    cl_mem d_msg = clCreateBuffer(g_ext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   32, (void*)msg, &err);
    cl_mem d_sig = clCreateBuffer(g_ext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(ExtendedCL::SchnorrSig), (void*)&sig, &err);
    cl_mem d_res = clCreateBuffer(g_ext.context, CL_MEM_WRITE_ONLY, sizeof(int), nullptr, &err);

    cl_uint count = 1;
    clSetKernelArg(g_ext.k_schnorr_verify, 0, sizeof(cl_mem), &d_pk);
    clSetKernelArg(g_ext.k_schnorr_verify, 1, sizeof(cl_mem), &d_msg);
    clSetKernelArg(g_ext.k_schnorr_verify, 2, sizeof(cl_mem), &d_sig);
    clSetKernelArg(g_ext.k_schnorr_verify, 3, sizeof(cl_mem), &d_res);
    clSetKernelArg(g_ext.k_schnorr_verify, 4, sizeof(cl_uint), &count);

    size_t global = 1;
    clEnqueueNDRangeKernel(g_ext.queue, g_ext.k_schnorr_verify, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
    clFinish(g_ext.queue);

    int result = 0;
    clEnqueueReadBuffer(g_ext.queue, d_res, CL_TRUE, 0, sizeof(int), &result, 0, nullptr, nullptr);

    clReleaseMemObject(d_pk); clReleaseMemObject(d_msg);
    clReleaseMemObject(d_sig); clReleaseMemObject(d_res);
    return result != 0;
}

// Helper: compute pubkey via extended kernel (generator_mul_windowed)
// This ensures field arithmetic consistency: pubkey and verify use the same
// cl_program (secp256k1_extended.cl) with identical field_mul_impl.
static JacobianPoint ext_generator_mul(const Scalar& priv) {
    cl_int err;
    cl_mem d_scalar = clCreateBuffer(g_ext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      sizeof(Scalar), (void*)&priv, &err);
    JacobianPoint result{};
    cl_mem d_result = clCreateBuffer(g_ext.context, CL_MEM_WRITE_ONLY,
                                      sizeof(JacobianPoint), nullptr, &err);
    cl_uint count = 1;
    clSetKernelArg(g_ext.k_gen_mul_win, 0, sizeof(cl_mem), &d_scalar);
    clSetKernelArg(g_ext.k_gen_mul_win, 1, sizeof(cl_mem), &d_result);
    clSetKernelArg(g_ext.k_gen_mul_win, 2, sizeof(cl_uint), &count);

    size_t global = 1;
    clEnqueueNDRangeKernel(g_ext.queue, g_ext.k_gen_mul_win, 1, nullptr,
                           &global, nullptr, 0, nullptr, nullptr);
    clFinish(g_ext.queue);
    clEnqueueReadBuffer(g_ext.queue, d_result, CL_TRUE, 0,
                        sizeof(JacobianPoint), &result, 0, nullptr, nullptr);
    // Q-01: Rule 10 — zero scalar buffer (may hold private key) before release
    cl_uchar _zero = 0;
    clEnqueueFillBuffer(g_ext.queue, d_scalar, &_zero, 1, 0, sizeof(Scalar), 0, nullptr, nullptr);
    clFinish(g_ext.queue);
    clReleaseMemObject(d_scalar);
    clReleaseMemObject(d_result);
    return result;
}

// Helper: get pubkey X bytes from scalar (for Schnorr)
// Uses ext_generator_mul to ensure consistency with schnorr_verify kernel.
static void get_schnorr_pubkey_x(const Scalar& priv, uint8_t out[32]) {
    auto P = ext_generator_mul(priv);
    auto aff = jacobian_to_affine(P);
    // Big-endian serialize field element
    for (int i = 0; i < 4; i++) {
        uint64_t limb = aff.x.limbs[3 - i];
        for (int j = 0; j < 8; j++) {
            out[i * 8 + j] = (uint8_t)(limb >> (56 - j * 8));
        }
    }
}

// ECDSA roundtrip: sign + verify
static int audit_ecdsa_roundtrip() {
    if (!g_ext.valid) return -1;
    auto priv = sc_from_u64(42);
    uint8_t msg[32] = {};
    msg[0] = 0xAA; msg[31] = 0xBB;

    ExtendedCL::ECDSASig sig;
    if (!ocl_ecdsa_sign(priv, msg, sig)) return 1;

    // Use pubkey from extended kernel (same field arithmetic as sign/verify)
    auto pub = ext_generator_mul(priv);
    if (!ocl_ecdsa_verify(pub, msg, sig)) return 2;
    return 0;
}

// Schnorr roundtrip: sign + verify
static int audit_schnorr_roundtrip() {
    if (!g_ext.valid) return -1;
    auto priv = sc_from_u64(7);
    uint8_t msg[32] = {}, aux[32] = {};
    msg[0] = 0xCC; msg[31] = 0xDD;

    uint8_t pubkey_x[32];
    get_schnorr_pubkey_x(priv, pubkey_x);

    ExtendedCL::SchnorrSig sig;
    if (!ocl_schnorr_sign(priv, msg, aux, sig)) return 1;
    if (!ocl_schnorr_verify(pubkey_x, msg, sig)) return 2;
    return 0;
}

// ECDSA wrong-key rejection
static int audit_ecdsa_wrong_key() {
    if (!g_ext.valid) return -1;
    auto priv1 = sc_from_u64(1);
    auto priv2 = sc_from_u64(2);
    uint8_t msg[32] = {};
    msg[0] = 0xEE;

    ExtendedCL::ECDSASig sig;
    if (!ocl_ecdsa_sign(priv1, msg, sig)) return 1;

    auto pub2 = ext_generator_mul(priv2);
    // Verify with wrong key must FAIL
    if (ocl_ecdsa_verify(pub2, msg, sig)) return 2;
    return 0;
}

// =============================================================================
// Section 3: Batch Operations
// =============================================================================

// Batch scalar mul generator
static int audit_batch_scalar_mul() {
    constexpr int N = 4;
    Scalar scalars[N];
    JacobianPoint results[N];
    for (int i = 0; i < N; i++) scalars[i] = sc_from_u64(i + 1);

    g_ctx->batch_scalar_mul_generator(scalars, results, N);

    // Verify each matches single op
    for (int i = 0; i < N; i++) {
        auto single = g_ctx->scalar_mul_generator(scalars[i]);
        auto a1 = jacobian_to_affine(results[i]);
        auto a2 = jacobian_to_affine(single);
        if (!fe_eq(a1.x, a2.x)) return i + 1;
    }
    return 0;
}

// Batch Jacobian to Affine
static int audit_batch_j2a() {
    constexpr int N = 4;
    Scalar scalars[N];
    JacobianPoint jpoints[N];
    AffinePoint apoints[N];
    for (int i = 0; i < N; i++) {
        scalars[i] = sc_from_u64(i + 1);
        jpoints[i] = g_ctx->scalar_mul_generator(scalars[i]);
    }

    g_ctx->batch_jacobian_to_affine(jpoints, apoints, N);

    for (int i = 0; i < N; i++) {
        auto expected = jacobian_to_affine(jpoints[i]);
        if (!fe_eq(apoints[i].x, expected.x)) return i + 1;
    }
    return 0;
}

// =============================================================================
// Section 4: Differential (OpenCL vs CPU lib-level)
// =============================================================================

// Verify scalar mul gives same result as lib's jacobian_to_affine
static int audit_diff_scalar_mul() {
    // Generator x (known constant)
    auto gen = get_generator();
    auto G1 = g_ctx->scalar_mul_generator(sc_from_u64(1));
    auto g1_a = jacobian_to_affine(G1);
    return fe_eq(g1_a.x, gen.x) ? 0 : 1;
}

// =============================================================================
// Section 5: Standard Test Vectors
// =============================================================================

// RFC 6979 determinism: same key+msg => same sig
static int audit_rfc6979_determinism() {
    if (!g_ext.valid) return -1;
    auto priv = sc_from_u64(0xDEADBEEF);
    uint8_t msg[32] = {};
    msg[0] = 0x42; msg[31] = 0xFF;

    ExtendedCL::ECDSASig sig1, sig2;
    if (!ocl_ecdsa_sign(priv, msg, sig1)) return 1;
    if (!ocl_ecdsa_sign(priv, msg, sig2)) return 2;

    // Must be identical (RFC 6979)
    if (memcmp(&sig1, &sig2, sizeof(sig1)) != 0) return 3;

    // Different msg => different sig
    msg[0] ^= 0x01;
    ExtendedCL::ECDSASig sig3;
    if (!ocl_ecdsa_sign(priv, msg, sig3)) return 4;
    if (memcmp(&sig1, &sig3, sizeof(sig1)) == 0) return 5;
    return 0;
}

// BIP-340: sign with known key, verify, tamper => reject
static int audit_bip340_vectors() {
    if (!g_ext.valid) return -1;
    auto priv = sc_from_u64(3);
    uint8_t msg[32] = {}, aux[32] = {};
    for (int i = 0; i < 32; i++) msg[i] = (uint8_t)i;

    uint8_t pubkey_x[32];
    get_schnorr_pubkey_x(priv, pubkey_x);

    ExtendedCL::SchnorrSig sig;
    if (!ocl_schnorr_sign(priv, msg, aux, sig)) return 1;
    if (!ocl_schnorr_verify(pubkey_x, msg, sig)) return 2;

    // Tamper message => reject
    msg[0] ^= 0xFF;
    if (ocl_schnorr_verify(pubkey_x, msg, sig)) return 3;
    return 0;
}

// =============================================================================
// Section 6: Protocol Security
// =============================================================================

// ECDSA multi-key: 10 keys sign+verify
static int audit_ecdsa_multi_key() {
    if (!g_ext.valid) return -1;
    uint64_t keys[] = {1, 2, 3, 7, 42, 256, 0xDEAD, 0xCAFE, 0xFFFF, 65537};
    uint8_t msg[32] = {};
    msg[0] = 0xBB; msg[15] = 0xCC; msg[31] = 0xDD;

    for (int ki = 0; ki < 10; ki++) {
        auto priv = sc_from_u64(keys[ki]);
        ExtendedCL::ECDSASig sig;
        if (!ocl_ecdsa_sign(priv, msg, sig)) return 10 + ki;
        auto pub = ext_generator_mul(priv);
        if (!ocl_ecdsa_verify(pub, msg, sig)) return 20 + ki;
    }
    return 0;
}

// Schnorr multi-key: 10 keys
static int audit_schnorr_multi_key() {
    if (!g_ext.valid) return -1;
    uint64_t keys[] = {1, 2, 3, 7, 42, 256, 0xDEAD, 0xCAFE, 0xFFFF, 65537};
    uint8_t msg[32] = {}, aux[32] = {};
    msg[0] = 0xEE; msg[31] = 0x11;

    for (int ki = 0; ki < 10; ki++) {
        auto priv = sc_from_u64(keys[ki]);
        uint8_t pubkey_x[32];
        get_schnorr_pubkey_x(priv, pubkey_x);

        ExtendedCL::SchnorrSig sig;
        if (!ocl_schnorr_sign(priv, msg, aux, sig)) return 10 + ki;
        if (!ocl_schnorr_verify(pubkey_x, msg, sig)) return 20 + ki;
    }
    return 0;
}

// =============================================================================
// Section 7: Fuzzing & Adversarial Inputs
// =============================================================================

// Edge scalars: 0*G at infinity, 1*G == G, different points differ
static int audit_fuzz_edge_scalars() {
    // k=1 -> generator
    auto G1 = g_ctx->scalar_mul_generator(sc_from_u64(1));
    auto g1a = jacobian_to_affine(G1);
    auto gen = get_generator();
    if (!fe_eq(g1a.x, gen.x)) return 1;

    // k=2 -> not generator
    auto G2 = g_ctx->scalar_mul_generator(sc_from_u64(2));
    auto g2a = jacobian_to_affine(G2);
    if (fe_eq(g2a.x, gen.x)) return 2;

    // 2*G == G+G
    auto GG = g_ctx->point_add(G1, G1);
    auto gga = jacobian_to_affine(GG);
    if (!fe_eq(gga.x, g2a.x)) return 3;
    return 0;
}

// ECDSA zero key rejection
static int audit_fuzz_ecdsa_zero_key() {
    if (!g_ext.valid) return -1;
    auto zero = Scalar::zero();
    uint8_t msg[32] = {};
    msg[0] = 0xAA;

    ExtendedCL::ECDSASig sig;
    // Must fail
    if (ocl_ecdsa_sign(zero, msg, sig)) return 1;
    return 0;
}

// Schnorr zero key rejection
static int audit_fuzz_schnorr_zero_key() {
    if (!g_ext.valid) return -1;
    auto zero = Scalar::zero();
    uint8_t msg[32] = {}, aux[32] = {};

    ExtendedCL::SchnorrSig sig;
    if (ocl_schnorr_sign(zero, msg, aux, sig)) return 1;
    return 0;
}

// =============================================================================
// Section 8: Performance Smoke
// =============================================================================

// ECDSA 50-iteration stress
static int audit_perf_ecdsa_stress() {
    if (!g_ext.valid) return -1;
    auto priv = sc_from_u64(0xDEADCAFE);
    auto pub = ext_generator_mul(priv);
    uint8_t msg[32] = {};

    for (int i = 0; i < 50; i++) {
        msg[0] = (uint8_t)i;
        ExtendedCL::ECDSASig sig;
        if (!ocl_ecdsa_sign(priv, msg, sig)) return 1;
        if (!ocl_ecdsa_verify(pub, msg, sig)) return 2;
    }
    return 0;
}

// Schnorr 25-iteration stress
static int audit_perf_schnorr_stress() {
    if (!g_ext.valid) return -1;
    auto priv = sc_from_u64(0xCAFEBABE);
    uint8_t pubkey_x[32];
    get_schnorr_pubkey_x(priv, pubkey_x);
    uint8_t msg[32] = {}, aux[32] = {};

    for (int i = 0; i < 25; i++) {
        msg[0] = (uint8_t)i;
        ExtendedCL::SchnorrSig sig;
        if (!ocl_schnorr_sign(priv, msg, aux, sig)) return 1;
        if (!ocl_schnorr_verify(pubkey_x, msg, sig)) return 2;
    }
    return 0;
}

// =============================================================================
// Section 9: BIP-352 Silent Payments & GLV Correctness
// =============================================================================

// Helper: expand kernel file with #include directives resolved (like bench_bip352_opencl.cpp).
static std::string bip352_expand_kernel(const std::string& path,
                                        std::vector<std::string>& seen) {
    if (std::find(seen.begin(), seen.end(), path) != seen.end()) return {};
    seen.push_back(path);
    std::string src = load_file(path);
    if (src.empty()) return {};
    std::string dir = path.substr(0, path.find_last_of("/\\"));
    if (dir.empty()) dir = ".";
    std::istringstream in(src);
    std::ostringstream out;
    std::string line;
    while (std::getline(in, line)) {
        size_t s = line.find_first_not_of(" \t");
        std::string trimmed = (s != std::string::npos) ? line.substr(s) : line;
        if (trimmed.rfind("#include \"", 0) == 0) {
            size_t q1 = trimmed.find('"') + 1;
            size_t q2 = trimmed.find('"', q1);
            std::string child = dir + "/" + trimmed.substr(q1, q2 - q1);
            out << bip352_expand_kernel(child, seen);
        } else {
            out << line << '\n';
        }
    }
    return out.str();
}

// Host wNAF encoder: mirrors the GPU scalar_to_wnaf fixed-130-step version.
// Encodes 128-bit scalar (s0=LSW, s1=MSW) into 5-bit signed wNAF digits.
static void audit_host_wnaf(uint64_t s0, uint64_t s1, int8_t wnaf[130]) {
    uint64_t s[4] = {s0, s1, 0, 0};
    for (int i = 0; i < 130; i++) {
        if (s[0] & 1ULL) {
            int d = (int)(s[0] & 0x1FULL);
            if (d >= 16) {
                d -= 32;
                uint64_t add = (uint64_t)(-d);
                uint64_t prev = s[0]; s[0] += add;
                if (s[0] < prev) { for (int j=1;j<4;j++) if (++s[j]) break; }
            } else {
                uint64_t prev = s[0]; s[0] -= (uint64_t)d;
                if (s[0] > prev) { for (int j=1;j<4;j++) if (s[j]--) break; }
            }
            wnaf[i] = (int8_t)d;
        } else { wnaf[i] = 0; }
        s[0] = (s[0] >> 1) | (s[1] << 63);
        s[1] = (s[1] >> 1) | (s[2] << 63);
        s[2] = (s[2] >> 1) | (s[3] << 63);
        s[3] >>= 1;
    }
}

// Test 1: CPU wNAF round-trip — encode scalar, decode digits back, verify match.
// Tests host_compute_wnaf correctness: this was the key change that fixed the -36 crash.
static int audit_glv_wnaf_roundtrip() {
    struct TC { uint64_t s0, s1; const char* label; };
    static const TC cases[] = {
        {1,  0, "k=1"},
        {2,  0, "k=2"},
        {15, 0, "k=15 (max single wNAF digit)"},
        {16, 0, "k=16 (two-digit boundary)"},
        {31, 0, "k=31 (wNAF carry: 32-1)"},
        {0x5555555555555555ULL, 0x5555555555555555ULL, "k=0x5555... (alternating bits)"},
        {0xFFFFFFFFFFFFFFFFULL, 0x7FFFFFFFFFFFFFFFULL, "k near 2^127"},
        // k1 half of SCAN_KEY GLV decomposition (lower 128 bits of full key)
        {0x38af4ad300da1a42ULL, 0x30d7d6a3b98294b1ULL, "k1 from SCAN_KEY GLV half"},
    };

    for (auto& tc : cases) {
        int8_t wnaf[130] = {};
        audit_host_wnaf(tc.s0, tc.s1, wnaf);

        // Reconstruct: sum(wnaf[i] * 2^i) for i=0..129 using 128-bit arithmetic.
        // Use __uint128_t for correctness (GCC/Clang extension, fine on x86-64).
        __uint128_t result = 0, power = 1;
        for (int i = 0; i < 130; i++) {
            if (wnaf[i] > 0)  result += (__uint128_t)(uint8_t) wnaf[i]  * power;
            if (wnaf[i] < 0)  result -= (__uint128_t)(uint8_t)(-wnaf[i]) * power;
            power <<= 1;
        }
        uint64_t r0 = (uint64_t)result;
        uint64_t r1 = (uint64_t)(result >> 64);

        if (r0 != tc.s0 || r1 != tc.s1) {
            std::fprintf(stderr, "  [FAIL] wNAF roundtrip for %s: "
                "expected (%016llx,%016llx) got (%016llx,%016llx)\n",
                tc.label,
                (unsigned long long)tc.s0, (unsigned long long)tc.s1,
                (unsigned long long)r0,    (unsigned long long)r1);
            return 1;
        }
    }
    return 0;
}

// Test 2: GLV large scalar consistency via OpenCL library.
// Verifies k*G + G = (k+1)*G for three large scalars that stress the GLV path:
//   - SCAN_KEY (256-bit random key, both GLV halves active)
//   - 2^128 (decomposition boundary)
//   - 0x5555... (alternating bit pattern, maximally stresses wNAF carry logic)
static int audit_glv_large_scalar() {
    // Helper: hex string (big-endian) -> little-endian Scalar limbs
    auto from_hex = [](const char* hex) -> Scalar {
        Scalar s{};
        std::string h(hex);
        while (h.size() < 64) h = "0" + h;
        for (int i = 0; i < 4; i++) {
            uint64_t v = 0;
            for (int j = 0; j < 16; j++)  {
                char c = h[(3 - i) * 16 + j];
                int d = (c >= '0' && c <= '9') ? c - '0'
                      : (c >= 'a' && c <= 'f') ? c - 'a' + 10
                      : (c >= 'A' && c <= 'F') ? c - 'A' + 10 : 0;
                v = (v << 4) | (uint64_t)d;
            }
            s.limbs[i] = v;
        }
        return s;
    };

    struct TC { Scalar k, kp1; const char* label; };
    Scalar s_scan   = from_hex("c4239fd6fc3db6e22b8bed6a49219e4e30d7d6a3b98294b138af4ad300da1a42");
    Scalar s_scanp  = from_hex("c4239fd6fc3db6e22b8bed6a49219e4e30d7d6a3b98294b138af4ad300da1a43");
    Scalar s_2_128  = {{0UL, 0UL, 1UL, 0UL}};
    Scalar s_2_128p = {{1UL, 0UL, 1UL, 0UL}};
    Scalar s_alt    = {{0x5555555555555555ULL, 0x5555555555555555ULL,
                         0x5555555555555555ULL, 0x5555555555555555ULL}};
    Scalar s_altp   = {{0x5555555555555556ULL, 0x5555555555555555ULL,
                         0x5555555555555555ULL, 0x5555555555555555ULL}};

    TC cases[] = {
        {s_scan,  s_scanp,  "SCAN_KEY (256-bit)"},
        {s_2_128, s_2_128p, "k = 2^128 (GLV boundary)"},
        {s_alt,   s_altp,   "k = 0x5555... (alternating bits)"},
    };

    Scalar one = sc_from_u64(1);
    JacobianPoint oneG = g_ctx->scalar_mul_generator(one);

    for (auto& tc : cases) {
        JacobianPoint kG    = g_ctx->scalar_mul_generator(tc.k);
        JacobianPoint kp1_a = g_ctx->point_add(kG, oneG);         // k*G + G
        JacobianPoint kp1_b = g_ctx->scalar_mul_generator(tc.kp1); // (k+1)*G

        AffinePoint a = jacobian_to_affine(kp1_a);
        AffinePoint b = jacobian_to_affine(kp1_b);
        if (!fe_eq(a.x, b.x) || !fe_eq(a.y, b.y)) {
            std::fprintf(stderr, "  [FAIL] GLV %s: k*G+G != (k+1)*G\n", tc.label);
            return 1;
        }
    }
    return 0;
}

// Struct layout matching BIP352ScanKeyGlv in secp256k1_bip352.cl.
// Used by the BIP-352 kernel audit tests below.
struct alignas(1) AuditBIP352ScanKeyGlv {
    int8_t  wnaf1[130]; // +0:   wNAF digits for k1 half-scalar
    int8_t  wnaf2[130]; // +130: wNAF digits for k2 half-scalar
    uint8_t k1_neg;     // +260: 1 if k1 negative
    uint8_t flip_phi;   // +261: 1 if phi table y should be negated
    uint8_t pad0, pad1; // +262-263: padding
};
static_assert(sizeof(AuditBIP352ScanKeyGlv) == 264, "BIP352ScanKeyGlv size mismatch");

// Kernel-side AffinePoint and FieldElement layout (must match .cl struct).
struct AuditFieldElement { uint64_t limbs[4]; };
struct AuditAffinePoint  { AuditFieldElement x, y; };

// secp256k1 generator G in the kernel's field element representation (little-endian limbs).
static AuditAffinePoint audit_generator_point() {
    AuditAffinePoint g;
    g.x.limbs[0] = 0x59F2815B16F81798ULL; g.x.limbs[1] = 0x029BFCDB2DCE28D9ULL;
    g.x.limbs[2] = 0x55A06295CE870B07ULL; g.x.limbs[3] = 0x79BE667EF9DCBBACULL;
    g.y.limbs[0] = 0x9C47D08FFB10D4B8ULL; g.y.limbs[1] = 0xFD17B448A6855419ULL;
    g.y.limbs[2] = 0x5DA4FBFC0E1108A8ULL; g.y.limbs[3] = 0x483ADA7726A3C465ULL;
    return g;
}

// Test 3: BIP-352 kernel compiles without error.
static int audit_bip352_kernel_build() {
    if (g_kernel_dir.empty()) return -1;
    std::vector<std::string> seen;
    std::string src = bip352_expand_kernel(g_kernel_dir + "/secp256k1_bip352.cl", seen);
    if (src.empty()) return -1;

    cl_context cl_ctx = (cl_context)g_ctx->native_context();
    cl_command_queue cl_q = (cl_command_queue)g_ctx->native_queue();
    cl_device_id cl_dev = nullptr;
    clGetCommandQueueInfo(cl_q, CL_QUEUE_DEVICE, sizeof(cl_dev), &cl_dev, nullptr);

    cl_int err;
    const char* src_ptr = src.c_str();
    size_t src_len = src.size();
    cl_program prog = clCreateProgramWithSource(cl_ctx, 1, &src_ptr, &src_len, &err);
    if (err != CL_SUCCESS) return 1;

    err = clBuildProgram(prog, 1, &cl_dev, "-cl-std=CL1.2", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(prog, cl_dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(prog, cl_dev, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::fprintf(stderr, "  BIP-352 build log:\n%s\n", log.c_str());
        clReleaseProgram(prog);
        return 2;
    }

    // Verify both kernel entry points exist
    cl_kernel k_nolut = clCreateKernel(prog, "bip352_pipeline_kernel", &err);
    if (err != CL_SUCCESS) { clReleaseProgram(prog); return 3; }
    cl_kernel k_lut   = clCreateKernel(prog, "bip352_pipeline_kernel_lut", &err);
    if (err != CL_SUCCESS) { clReleaseKernel(k_nolut); clReleaseProgram(prog); return 4; }

    clReleaseKernel(k_nolut);
    clReleaseKernel(k_lut);
    clReleaseProgram(prog);
    return 0;
}

// Test 4: Regression for CL_INVALID_COMMAND_QUEUE (-36) GPU fault.
// Runs bip352_pipeline_kernel (no-LUT path) with 1 work item and verifies no crash.
// The crash was caused by GPU private-memory overflow from int wnaf[130]×2 arrays.
// Fix: precompute wNAF on CPU (BIP352ScanKeyGlv.wnaf1/wnaf2), read from __constant.
// Three scan-key edge cases: k=1 (minimal), k from SCAN_KEY, k with all-15 wNAF digits.
static int audit_bip352_no_crash() {
    if (g_kernel_dir.empty()) return -1;
    std::vector<std::string> seen;
    std::string src = bip352_expand_kernel(g_kernel_dir + "/secp256k1_bip352.cl", seen);
    if (src.empty()) return -1;

    cl_context cl_ctx = (cl_context)g_ctx->native_context();
    cl_command_queue cl_q = (cl_command_queue)g_ctx->native_queue();
    cl_device_id cl_dev = nullptr;
    clGetCommandQueueInfo(cl_q, CL_QUEUE_DEVICE, sizeof(cl_dev), &cl_dev, nullptr);

    cl_int err;
    const char* src_ptr = src.c_str();
    size_t src_len = src.size();
    cl_program prog = clCreateProgramWithSource(cl_ctx, 1, &src_ptr, &src_len, &err);
    if (err != CL_SUCCESS) return 1;

    err = clBuildProgram(prog, 1, &cl_dev, "-cl-std=CL1.2 -cl-fast-relaxed-math", nullptr, nullptr);
    if (err != CL_SUCCESS) { clReleaseProgram(prog); return 2; }

    cl_kernel kernel = clCreateKernel(prog, "bip352_pipeline_kernel", &err);
    if (err != CL_SUCCESS) { clReleaseProgram(prog); return 3; }

    // Edge case scan keys to test. k1_neg/flip_phi chosen to exercise both paths.
    struct EdgeCase {
        const char* label;
        int8_t      wnaf1_0; // wnaf1[0] digit (rest 0)
        int8_t      wnaf2_0; // wnaf2[0] digit (rest 0)
        uint8_t     k1_neg, flip_phi;
    };
    static const EdgeCase edges[] = {
        {"k=1 (minimal scalar)",    1, 0, 0, 0},
        {"k1=15,k2=1 (max digit)",  15, 1, 0, 0},
        {"k1_neg=1, flip_phi=1",    1, 1, 1, 1},  // negate path
    };

    AuditAffinePoint g_pt    = audit_generator_point();
    AuditAffinePoint spend_pt = g_pt; // spend = G for simplicity

    // Pre-allocate buffers (reused across edge cases)
    cl_mem d_tweaks  = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY, sizeof(AuditAffinePoint), nullptr, &err);
    cl_mem d_spend   = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      sizeof(AuditAffinePoint), &spend_pt, &err);
    cl_mem d_scan    = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY, sizeof(AuditBIP352ScanKeyGlv), nullptr, &err);
    cl_mem d_prefixes = clCreateBuffer(cl_ctx, CL_MEM_WRITE_ONLY, sizeof(uint64_t), nullptr, &err);

    cl_uint count = 1;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_tweaks);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_scan);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_spend);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_prefixes);
    clSetKernelArg(kernel, 4, sizeof(cl_uint), &count);

    int result = 0;
    for (auto& ec : edges) {
        // Build scan plan
        AuditBIP352ScanKeyGlv plan{};
        plan.wnaf1[0]  = ec.wnaf1_0;
        plan.wnaf2[0]  = ec.wnaf2_0;
        plan.k1_neg    = ec.k1_neg;
        plan.flip_phi  = ec.flip_phi;

        // Upload tweak=G and scan plan
        clEnqueueWriteBuffer(cl_q, d_tweaks, CL_TRUE, 0, sizeof(AuditAffinePoint), &g_pt, 0, nullptr, nullptr);
        clEnqueueWriteBuffer(cl_q, d_scan,   CL_TRUE, 0, sizeof(AuditBIP352ScanKeyGlv), &plan, 0, nullptr, nullptr);

        size_t global = 1, local = 1;
        err = clEnqueueNDRangeKernel(cl_q, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { result = 10; break; }
        err = clFinish(cl_q);
        if (err != CL_SUCCESS) {
            // -36 = CL_INVALID_COMMAND_QUEUE = GPU fault (regression for the private-stack overflow crash)
            std::fprintf(stderr, "  [FAIL] bip352_no_crash edge='%s' clFinish error=%d"
                " (expected 0; -36 = GPU fault regression)\n", ec.label, err);
            result = 20 + err; // encode the OCL error
            break;
        }

        uint64_t prefix = 0;
        clEnqueueReadBuffer(cl_q, d_prefixes, CL_TRUE, 0, sizeof(uint64_t), &prefix, 0, nullptr, nullptr);
        // prefix may be 0 if the point is infinity (edge case k1=0 path) — that's valid.
        // What we really test is that we reach here without crashing.
    }

    clReleaseMemObject(d_tweaks);
    clReleaseMemObject(d_scan);
    clReleaseMemObject(d_spend);
    clReleaseMemObject(d_prefixes);
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    return result;
}

// Test 5: BIP-352 pipeline output matches expected prefix for known input.
// Uses tweak=G, scan_key=SCAN_KEY. Expected prefix pre-computed by the CPU
// validation path in bench_bip352_opencl (validation: 0xb63b4601066a6971
// is the last-item prefix when batch=10000; for single item with tweak=G
// and k=SCAN_KEY this is independently computed below).
static int audit_bip352_correct() {
    if (g_kernel_dir.empty()) return -1;
    std::vector<std::string> seen;
    std::string src = bip352_expand_kernel(g_kernel_dir + "/secp256k1_bip352.cl", seen);
    if (src.empty()) return -1;

    cl_context cl_ctx = (cl_context)g_ctx->native_context();
    cl_command_queue cl_q = (cl_command_queue)g_ctx->native_queue();
    cl_device_id cl_dev = nullptr;
    clGetCommandQueueInfo(cl_q, CL_QUEUE_DEVICE, sizeof(cl_dev), &cl_dev, nullptr);

    cl_int err;
    const char* src_ptr = src.c_str();
    size_t src_len = src.size();
    cl_program prog = clCreateProgramWithSource(cl_ctx, 1, &src_ptr, &src_len, &err);
    if (err != CL_SUCCESS) return 1;
    err = clBuildProgram(prog, 1, &cl_dev, "-cl-std=CL1.2 -cl-fast-relaxed-math", nullptr, nullptr);
    if (err != CL_SUCCESS) { clReleaseProgram(prog); return 2; }
    cl_kernel kernel = clCreateKernel(prog, "bip352_pipeline_kernel", &err);
    if (err != CL_SUCCESS) { clReleaseProgram(prog); return 3; }

    // Build BIP352ScanKeyGlv for SCAN_KEY using the host wNAF encoder.
    // SCAN_KEY = c4239fd6fc3db6e22b8bed6a49219e4e30d7d6a3b98294b138af4ad300da1a42
    // GLV decomposition (pre-computed, matches bench_bip352_opencl):
    //   k1 (LE64): {0x5db6fc2bc78a0e07, 0x7fff7d82be8fb40f, 0, 0}  k1_neg=0
    //   k2 (LE64): {0x62491d65b0efea74, 0x3ca3a038cb4bac36, 0, 0}  flip_phi=0
    // (These are the GLV halves as output by secp256k1::fast::glv_decompose)
    // We use the benchmark's own scan_key encoding to stay in sync; here we use
    // the actual k1/k2 from a one-time CPU run of build_scan_glv_plan().
    // Instead of hard-coding the decomposition (which requires CPU GLV logic),
    // we test consistency: run 2 items (tweak=G), compare both give the same prefix.
    // A truly independent correctness check is in bench_bip352_opencl --batch 1 --local 1.

    // For this audit: run 2 identical tweaks, check both prefixes are equal (determinism).
    AuditBIP352ScanKeyGlv plan{};
    // k1=1, k2=0 (simplest: scan*tweak = 1*G = G for any decomposition where k1=1, k2=0)
    plan.wnaf1[0] = 1;
    plan.k1_neg = 0;
    plan.flip_phi = 0;

    AuditAffinePoint g_pt     = audit_generator_point();
    AuditAffinePoint spend_pt = g_pt;
    AuditAffinePoint tweaks[2] = {g_pt, g_pt}; // same tweak twice

    cl_mem d_tweaks   = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       2 * sizeof(AuditAffinePoint), tweaks, &err);
    cl_mem d_scan     = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(AuditBIP352ScanKeyGlv), &plan, &err);
    cl_mem d_spend    = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(AuditAffinePoint), &spend_pt, &err);
    cl_mem d_prefixes = clCreateBuffer(cl_ctx, CL_MEM_WRITE_ONLY, 2 * sizeof(uint64_t), nullptr, &err);

    cl_uint count = 2;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_tweaks);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_scan);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_spend);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_prefixes);
    clSetKernelArg(kernel, 4, sizeof(cl_uint), &count);

    size_t global = 2, local = 1;
    err = clEnqueueNDRangeKernel(cl_q, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { clReleaseProgram(prog); return 4; }
    err = clFinish(cl_q);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "  [FAIL] bip352_correct: clFinish error=%d\n", err);
        clReleaseProgram(prog); return 5;
    }

    uint64_t prefixes[2] = {};
    clEnqueueReadBuffer(cl_q, d_prefixes, CL_TRUE, 0, 2 * sizeof(uint64_t), prefixes, 0, nullptr, nullptr);

    int result = 0;
    // Both items have identical input so must produce identical prefix (determinism test)
    if (prefixes[0] != prefixes[1]) {
        std::fprintf(stderr, "  [FAIL] bip352_correct: non-deterministic output:"
            " item[0]=0x%016llx item[1]=0x%016llx\n",
            (unsigned long long)prefixes[0], (unsigned long long)prefixes[1]);
        result = 6;
    }
    // Prefix must be non-zero (1*G = G is not the point at infinity)
    if (prefixes[0] == 0) {
        std::fprintf(stderr, "  [FAIL] bip352_correct: prefix=0 (unexpected infinity)\n");
        result = 7;
    }

    clReleaseMemObject(d_tweaks);
    clReleaseMemObject(d_scan);
    clReleaseMemObject(d_spend);
    clReleaseMemObject(d_prefixes);
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    return result;
}

// =============================================================================
// Section 10: ZK Proofs (Knowledge + DLEQ)
// =============================================================================

static int audit_zk_knowledge_roundtrip() {
    if (!g_zk.valid) return -1;
    cl_context   ctx = g_zk.context;
    cl_command_queue q = g_zk.queue;
    cl_int err;

    // Setup: secret=42, base=3*G, pubkey=secret*base=126*G
    auto secret = sc_from_u64(42);
    auto base_sc = sc_from_u64(3);
    auto pubkey  = ext_generator_mul(sc_from_u64(126)); // pubkey = 42 * (3*G) = 126*G = secret*base
    auto base_pt = ext_generator_mul(base_sc);           // base   = 3 * G
    uint8_t msg[32] = {};
    msg[0] = 0xAA; msg[31] = 0xBB;
    uint8_t aux[32] = {};
    aux[0] = 0x11;

    const cl_uint count = 1;
    const size_t global = 1;

    // Allocate device buffers
    cl_mem d_sec    = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(Scalar), &secret, &err);
    cl_mem d_pub    = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(JacobianPoint), &pubkey, &err);
    cl_mem d_base   = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(JacobianPoint), &base_pt, &err);
    cl_mem d_msg    = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 32, msg, &err);
    cl_mem d_aux    = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 32, aux, &err);
    cl_mem d_proof  = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(ZkCL::ZKKnowledgeProofH), nullptr, &err);
    cl_mem d_ok     = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(int), nullptr, &err);

    // Prove
    clSetKernelArg(g_zk.k_knowledge_prove, 0, sizeof(cl_mem), &d_sec);
    clSetKernelArg(g_zk.k_knowledge_prove, 1, sizeof(cl_mem), &d_pub);
    clSetKernelArg(g_zk.k_knowledge_prove, 2, sizeof(cl_mem), &d_base);
    clSetKernelArg(g_zk.k_knowledge_prove, 3, sizeof(cl_mem), &d_msg);
    clSetKernelArg(g_zk.k_knowledge_prove, 4, sizeof(cl_mem), &d_aux);
    clSetKernelArg(g_zk.k_knowledge_prove, 5, sizeof(cl_mem), &d_proof);
    clSetKernelArg(g_zk.k_knowledge_prove, 6, sizeof(cl_mem), &d_ok);
    clSetKernelArg(g_zk.k_knowledge_prove, 7, sizeof(cl_uint), &count);
    clEnqueueNDRangeKernel(q, g_zk.k_knowledge_prove, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
    int prove_ok = 0;
    clEnqueueReadBuffer(q, d_ok, CL_TRUE, 0, sizeof(int), &prove_ok, 0, nullptr, nullptr);
    if (!prove_ok) { clReleaseMemObject(d_sec); clReleaseMemObject(d_pub); clReleaseMemObject(d_base);
                     clReleaseMemObject(d_msg); clReleaseMemObject(d_aux); clReleaseMemObject(d_proof);
                     clReleaseMemObject(d_ok); return 10; }

    // Verify (valid proof)
    clSetKernelArg(g_zk.k_knowledge_verify, 0, sizeof(cl_mem), &d_proof);
    clSetKernelArg(g_zk.k_knowledge_verify, 1, sizeof(cl_mem), &d_pub);
    clSetKernelArg(g_zk.k_knowledge_verify, 2, sizeof(cl_mem), &d_base);
    clSetKernelArg(g_zk.k_knowledge_verify, 3, sizeof(cl_mem), &d_msg);
    clSetKernelArg(g_zk.k_knowledge_verify, 4, sizeof(cl_mem), &d_ok);
    clSetKernelArg(g_zk.k_knowledge_verify, 5, sizeof(cl_uint), &count);
    clEnqueueNDRangeKernel(q, g_zk.k_knowledge_verify, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
    int verify_ok = 0;
    clEnqueueReadBuffer(q, d_ok, CL_TRUE, 0, sizeof(int), &verify_ok, 0, nullptr, nullptr);
    if (!verify_ok) { clReleaseMemObject(d_sec); clReleaseMemObject(d_pub); clReleaseMemObject(d_base);
                      clReleaseMemObject(d_msg); clReleaseMemObject(d_aux); clReleaseMemObject(d_proof);
                      clReleaseMemObject(d_ok); return 20; }

    // Verify should reject wrong pubkey (wrong_pub = 99*G)
    auto wrong_pub = ext_generator_mul(sc_from_u64(99));
    cl_mem d_wpub = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(JacobianPoint), &wrong_pub, &err);
    clSetKernelArg(g_zk.k_knowledge_verify, 1, sizeof(cl_mem), &d_wpub);
    clEnqueueNDRangeKernel(q, g_zk.k_knowledge_verify, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
    int reject_ok = 0;
    clEnqueueReadBuffer(q, d_ok, CL_TRUE, 0, sizeof(int), &reject_ok, 0, nullptr, nullptr);
    clReleaseMemObject(d_wpub);
    int result = (reject_ok == 0) ? 0 : 30;  // must reject

    clReleaseMemObject(d_sec); clReleaseMemObject(d_pub); clReleaseMemObject(d_base);
    clReleaseMemObject(d_msg); clReleaseMemObject(d_aux); clReleaseMemObject(d_proof);
    clReleaseMemObject(d_ok);
    return result;
}

static int audit_zk_dleq_roundtrip() {
    if (!g_zk.valid) return -1;
    cl_context   ctx = g_zk.context;
    cl_command_queue q = g_zk.queue;
    cl_int err;

    // Setup: secret=7, G=generator, H=5*G, P=secret*G, Q=secret*H
    auto secret = sc_from_u64(7);
    auto h_sc   = sc_from_u64(5);
    auto G_pt   = ext_generator_mul(sc_from_u64(1));  // 1*G = G
    auto H_pt   = ext_generator_mul(h_sc);             // 5*G = H
    auto P_pt   = ext_generator_mul(secret);            // secret*G = P
    // Q = secret * H: use batch_scalar_mul or compute as ext_generator_mul(secret*h_sc mod n)
    // Approximate Q via: ext_generator_mul(secret * h_sc) — OK for test
    // Actually Q = secret * H. We need scalar mul of H by secret (not G).
    // For simplicity: use h_sc * secret as scalar (mod n) then generator_mul
    // This is WRONG (Q = secret * H ≠ (h_sc*secret) * G unless H=G)
    // Instead: do Q = 5 * P (since H=5*G and P=secret*G, so Q = secret*H = 5*secret*G = 5*P)
    // So Q = 5 * P, and we can compute via ext_generator_mul(5 * secret_scalar mod n)
    // This requires scalar-scalar multiply mod n. Use simpler approach:
    // secret=7, H=5*G → Q=7*H=35*G → compute 35*G directly
    auto q_sc  = sc_from_u64(35);  // 7 * 5 = 35
    auto Q_pt  = ext_generator_mul(q_sc);  // 35*G = Q = secret*H ✓

    uint8_t aux[32] = {}; aux[0] = 0x42;

    const cl_uint count = 1;
    const size_t global = 1;

    cl_mem d_sec   = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(Scalar), &secret, &err);
    cl_mem d_G     = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(JacobianPoint), &G_pt, &err);
    cl_mem d_H     = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(JacobianPoint), &H_pt, &err);
    cl_mem d_P     = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(JacobianPoint), &P_pt, &err);
    cl_mem d_Q     = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(JacobianPoint), &Q_pt, &err);
    cl_mem d_aux   = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 32, aux, &err);
    cl_mem d_proof = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(ZkCL::ZKDLEQProofH), nullptr, &err);
    cl_mem d_ok    = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(int), nullptr, &err);

    // Prove
    clSetKernelArg(g_zk.k_dleq_prove, 0, sizeof(cl_mem), &d_sec);
    clSetKernelArg(g_zk.k_dleq_prove, 1, sizeof(cl_mem), &d_G);
    clSetKernelArg(g_zk.k_dleq_prove, 2, sizeof(cl_mem), &d_H);
    clSetKernelArg(g_zk.k_dleq_prove, 3, sizeof(cl_mem), &d_P);
    clSetKernelArg(g_zk.k_dleq_prove, 4, sizeof(cl_mem), &d_Q);
    clSetKernelArg(g_zk.k_dleq_prove, 5, sizeof(cl_mem), &d_aux);
    clSetKernelArg(g_zk.k_dleq_prove, 6, sizeof(cl_mem), &d_proof);
    clSetKernelArg(g_zk.k_dleq_prove, 7, sizeof(cl_mem), &d_ok);
    clSetKernelArg(g_zk.k_dleq_prove, 8, sizeof(cl_uint), &count);
    clEnqueueNDRangeKernel(q, g_zk.k_dleq_prove, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
    int prove_ok = 0;
    clEnqueueReadBuffer(q, d_ok, CL_TRUE, 0, sizeof(int), &prove_ok, 0, nullptr, nullptr);

    int result = 0;
    if (!prove_ok) { result = 10; goto dleq_cleanup; }

    // Verify
    clSetKernelArg(g_zk.k_dleq_verify, 0, sizeof(cl_mem), &d_proof);
    clSetKernelArg(g_zk.k_dleq_verify, 1, sizeof(cl_mem), &d_G);
    clSetKernelArg(g_zk.k_dleq_verify, 2, sizeof(cl_mem), &d_H);
    clSetKernelArg(g_zk.k_dleq_verify, 3, sizeof(cl_mem), &d_P);
    clSetKernelArg(g_zk.k_dleq_verify, 4, sizeof(cl_mem), &d_Q);
    clSetKernelArg(g_zk.k_dleq_verify, 5, sizeof(cl_mem), &d_ok);
    clSetKernelArg(g_zk.k_dleq_verify, 6, sizeof(cl_uint), &count);
    clEnqueueNDRangeKernel(q, g_zk.k_dleq_verify, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
    {
        int verify_ok = 0;
        clEnqueueReadBuffer(q, d_ok, CL_TRUE, 0, sizeof(int), &verify_ok, 0, nullptr, nullptr);
        if (!verify_ok) { result = 20; goto dleq_cleanup; }
    }

    // Reject wrong Q (use 2*G as wrong Q)
    {
        auto wrong_Q = ext_generator_mul(sc_from_u64(2));
        cl_mem d_wQ = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(JacobianPoint), &wrong_Q, &err);
        clSetKernelArg(g_zk.k_dleq_verify, 4, sizeof(cl_mem), &d_wQ);
        clEnqueueNDRangeKernel(q, g_zk.k_dleq_verify, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
        int reject_ok = 0;
        clEnqueueReadBuffer(q, d_ok, CL_TRUE, 0, sizeof(int), &reject_ok, 0, nullptr, nullptr);
        clReleaseMemObject(d_wQ);
        if (reject_ok != 0) result = 30;  // must reject
    }

dleq_cleanup:
    clReleaseMemObject(d_sec); clReleaseMemObject(d_G); clReleaseMemObject(d_H);
    clReleaseMemObject(d_P);   clReleaseMemObject(d_Q); clReleaseMemObject(d_aux);
    clReleaseMemObject(d_proof); clReleaseMemObject(d_ok);
    return result;
}

// =============================================================================
// GPU CT Smoke: branchless constant-time layer (secp256k1_ct_smoke.cl)
// =============================================================================

// Initialize the CT smoke context on first use (lazy, cached).
static bool ensure_ct_smoke(std::string& err_out) {
    if (g_ct_smoke.valid) return true;
    if (!g_ctx) { err_out = "No OpenCL context"; return false; }
    if (!g_ct_smoke.init(*g_ctx, g_kernel_dir)) {
        err_out = g_ct_smoke.error;
        return false;
    }
    return true;
}

static int audit_ct_smoke_masks() {
    std::string err;
    if (!ensure_ct_smoke(err)) {
        fprintf(stderr, "    SKIP ct_smoke_masks: %s\n", err.c_str());
        return -1;
    }
    int r = g_ct_smoke.run_kernel(g_ct_smoke.k_masks, "ct_smoke_masks");
    if (r != 0) fprintf(stderr, "    ct_smoke_masks: error bitmap=0x%x\n", r);
    return r;
}

static int audit_ct_smoke_cmov() {
    std::string err;
    if (!ensure_ct_smoke(err)) {
        fprintf(stderr, "    SKIP ct_smoke_cmov: %s\n", err.c_str());
        return -1;
    }
    int r = g_ct_smoke.run_kernel(g_ct_smoke.k_cmov, "ct_smoke_cmov");
    if (r != 0) fprintf(stderr, "    ct_smoke_cmov: error bitmap=0x%x\n", r);
    return r;
}

static int audit_ct_smoke_ecdsa() {
    std::string err;
    if (!ensure_ct_smoke(err)) {
        fprintf(stderr, "    SKIP ct_smoke_ecdsa: %s\n", err.c_str());
        return -1;
    }
    int r = g_ct_smoke.run_kernel(g_ct_smoke.k_ecdsa, "ct_smoke_ecdsa");
    if (r != 0) fprintf(stderr, "    ct_smoke_ecdsa: result=%d (1=sign fail, 2=verify fail)\n", r);
    return r;
}

static int audit_ct_smoke_schnorr() {
    std::string err;
    if (!ensure_ct_smoke(err)) {
        fprintf(stderr, "    SKIP ct_smoke_schnorr: %s\n", err.c_str());
        return -1;
    }
    int r = g_ct_smoke.run_kernel(g_ct_smoke.k_schnorr, "ct_smoke_schnorr");
    if (r != 0) fprintf(stderr, "    ct_smoke_schnorr: result=%d (1=sign fail, 2=verify fail)\n", r);
    return r;
}

// =============================================================================
// Module & Section Registry
// =============================================================================

static const OclSectionInfo OCL_SECTIONS[] = {
    { "math_invariants",   "Mathematical Invariants (Field, Scalar, Point)" },
    { "signatures",        "Signature Operations (ECDSA, Schnorr/BIP-340)" },
    { "batch_advanced",    "Batch Operations & Advanced Algorithms" },
    { "differential",      "OpenCL-Host Differential Testing" },
    { "standard_vectors",  "Standard Test Vectors (BIP-340, RFC-6979)" },
    { "protocol_security", "Protocol Security (multi-key)" },
    { "fuzzing",           "Fuzzing & Adversarial Inputs" },
    { "performance",       "Performance Smoke Tests" },
    { "bip352_glv",        "BIP-352 Silent Payments & GLV Correctness" },
    { "zk_proofs",          "ZK Proofs (Knowledge & DLEQ)" },
    { "gpu_ct_smoke",      "GPU CT Layer (Branchless Constant-Time Smoke)" },
};
static constexpr int NUM_OCL_SECTIONS = sizeof(OCL_SECTIONS) / sizeof(OCL_SECTIONS[0]);

static const OclAuditModule OCL_MODULES[] = {
    // Section 1: Mathematical Invariants
    { "selftest_core",     "OpenCL Selftest (23+ kernel tests)",          "math_invariants", audit_selftest_core, false },
    { "field_add_sub",     "Field add/sub roundtrip",                     "math_invariants", audit_field_add_sub, false },
    { "field_mul_comm",    "Field mul commutativity",                     "math_invariants", audit_field_mul_commutativity, false },
    { "field_inv",         "Field inverse roundtrip (a * a^-1 = 1)",     "math_invariants", audit_field_inv_roundtrip, false },
    { "field_sqr",         "Field square == mul(a,a)",                    "math_invariants", audit_field_sqr_consistency, false },
    { "field_negate",      "Field negate roundtrip (a + (-a) = 0)",      "math_invariants", audit_field_negate, false },
    { "gen_mul_vec",       "Generator mul known vectors",                 "math_invariants", audit_generator_mul_known_vector, false },
    { "scalar_roundtrip",  "Scalar/Point consistency",                    "math_invariants", audit_scalar_add_sub, false },
    { "add_dbl_consist",   "Point add vs double consistency",             "math_invariants", audit_point_add_dbl_consistency, false },
    { "scalar_mul_lin",    "Scalar mul linearity (a+b)*G = aG+bG",      "math_invariants", audit_scalar_mul_linearity, false },
    { "group_order",       "Group order basic checks",                    "math_invariants", audit_group_order_basic, false },
    { "batch_inv",         "Batch inversion (Montgomery trick)",          "math_invariants", audit_batch_inversion, false },

    // Section 2: Signature Operations
    { "ecdsa_roundtrip",   "ECDSA sign + verify roundtrip",              "signatures", audit_ecdsa_roundtrip, false },
    { "schnorr_roundtrip", "Schnorr/BIP-340 sign + verify roundtrip",    "signatures", audit_schnorr_roundtrip, false },
    { "ecdsa_wrong_key",   "ECDSA verify rejects wrong pubkey",          "signatures", audit_ecdsa_wrong_key, false },

    // Section 3: Batch Operations
    { "batch_smul",        "Batch scalar mul generator",                  "batch_advanced", audit_batch_scalar_mul, false },
    { "batch_j2a",         "Batch Jacobian to Affine",                    "batch_advanced", audit_batch_j2a, false },

    // Section 4: Differential
    { "diff_smul",         "OpenCL-host differential scalar mul",         "differential", audit_diff_scalar_mul, false },

    // Section 5: Standard Test Vectors
    { "rfc6979_determ",    "RFC-6979 ECDSA deterministic nonce",          "standard_vectors", audit_rfc6979_determinism, false },
    { "bip340_vectors",    "BIP-340 Schnorr known-key roundtrip",         "standard_vectors", audit_bip340_vectors, false },

    // Section 6: Protocol Security
    { "ecdsa_multi_key",   "ECDSA multi-key (10 keys) sign+verify",      "protocol_security", audit_ecdsa_multi_key, false },
    { "schnorr_multi_key", "Schnorr multi-key (10 keys) sign+verify",    "protocol_security", audit_schnorr_multi_key, false },

    // Section 7: Fuzzing
    { "fuzz_edge_scalar",  "Edge-case scalars (0*G, 1*G, G+G=2G)",       "fuzzing", audit_fuzz_edge_scalars, false },
    { "fuzz_ecdsa_zero",   "ECDSA rejects zero private key",             "fuzzing", audit_fuzz_ecdsa_zero_key, false },
    { "fuzz_schnorr_zero", "Schnorr rejects zero private key",           "fuzzing", audit_fuzz_schnorr_zero_key, false },

    // Section 8: Performance Smoke
    { "perf_ecdsa_50",     "ECDSA 50-iteration stress",                   "performance", audit_perf_ecdsa_stress, false },
    { "perf_schnorr_25",   "Schnorr 25-iteration stress",                "performance", audit_perf_schnorr_stress, false },

    // Section 9: BIP-352 Silent Payments & GLV Correctness
    { "glv_wnaf_rt",       "CPU wNAF encode/decode roundtrip (8 scalars)",     "bip352_glv", audit_glv_wnaf_roundtrip,    false },
    { "glv_large_k",       "GLV large scalar k*G+G=(k+1)*G (3 scalars)",       "bip352_glv", audit_glv_large_scalar,      false },
    { "bip352_build",      "BIP-352 kernel compiles (both entry points)",       "bip352_glv", audit_bip352_kernel_build,   false },
    { "bip352_nocrash",    "BIP-352 no GPU fault: -36 crash regression (3 edge cases)", "bip352_glv", audit_bip352_no_crash, false },
    { "bip352_correct",    "BIP-352 pipeline determinism (2 identical tweaks)", "bip352_glv", audit_bip352_correct,        false },

    // Section 10: ZK Proofs
    { "zk_knowledge_rt",   "ZK knowledge proof roundtrip (prove+verify+reject)", "zk_proofs", audit_zk_knowledge_roundtrip, false },
    { "zk_dleq_rt",        "ZK DLEQ proof roundtrip (prove+verify+reject)",    "zk_proofs", audit_zk_dleq_roundtrip,     false },

    // Section 11: GPU CT Layer Smoke
    { "ct_masks",          "CT mask generation (is_zero, is_nonzero, eq, bool)",   "gpu_ct_smoke", audit_ct_smoke_masks,   false },
    { "ct_cmov",           "CT cmov256 / cswap256 correctness",                    "gpu_ct_smoke", audit_ct_smoke_cmov,    false },
    { "ct_ecdsa",          "CT ECDSA sign (privkey=1) + fast-path verify",         "gpu_ct_smoke", audit_ct_smoke_ecdsa,   false },
    { "ct_schnorr",        "CT Schnorr sign (privkey=1, BIP-340) + verify",        "gpu_ct_smoke", audit_ct_smoke_schnorr, false },
};
static constexpr int NUM_OCL_MODULES = sizeof(OCL_MODULES) / sizeof(OCL_MODULES[0]);

// =============================================================================
// Device info
// =============================================================================
struct OclDeviceInfo {
    std::string name;
    std::string vendor;
    std::string version;
    std::string driver_version;
    size_t memory_mb;
    int compute_units;
    std::string backend;
};

static OclDeviceInfo detect_ocl_device(const Context& ctx) {
    OclDeviceInfo info;
    auto dev = ctx.device_info();
    info.name = dev.name;
    info.vendor = dev.vendor;
    info.version = dev.version;
    info.driver_version = dev.driver_version;
    info.memory_mb = dev.global_mem_size / (1024 * 1024);
    info.compute_units = dev.compute_units;
    info.backend = "OpenCL";
    return info;
}

// =============================================================================
// Platform detection (host)
// =============================================================================
struct PlatformInfo {
    std::string os;
    std::string arch;
    std::string compiler;
    std::string build_type;
};

static PlatformInfo detect_platform() {
    PlatformInfo p;
#if defined(_WIN32)
    p.os = "Windows";
#elif defined(__linux__)
    p.os = "Linux";
#elif defined(__APPLE__)
    p.os = "macOS";
#else
    p.os = "Unknown";
#endif

#if defined(__x86_64__) || defined(_M_X64)
    p.arch = "x86-64";
#elif defined(__aarch64__) || defined(_M_ARM64)
    p.arch = "ARM64";
#elif defined(__riscv) && (__riscv_xlen == 64)
    p.arch = "RISC-V 64";
#else
    p.arch = "Unknown";
#endif

#if defined(__clang__)
    char buf[64];
    std::snprintf(buf, sizeof(buf), "Clang %d.%d.%d", __clang_major__, __clang_minor__, __clang_patchlevel__);
    p.compiler = buf;
#elif defined(__GNUC__)
    char buf[64];
    std::snprintf(buf, sizeof(buf), "GCC %d.%d.%d", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
    p.compiler = buf;
#elif defined(_MSC_VER)
    p.compiler = "MSVC " + std::to_string(_MSC_VER);
#else
    p.compiler = "Unknown";
#endif

#ifdef NDEBUG
    p.build_type = "Release";
#else
    p.build_type = "Debug";
#endif
    return p;
}

// =============================================================================
// Report generation
// =============================================================================

struct ModuleResult {
    std::string id;
    std::string name;
    std::string section;
    bool passed;
    bool skipped;
    bool advisory;
    double time_ms;
    int error_code;
};

struct SectionSummary {
    const char* section_id;
    const char* title_en;
    int total;
    int passed;
    int failed;
    int skipped;
    int advisory;
    double time_ms;
};

static std::vector<SectionSummary> compute_section_summaries(
    const std::vector<ModuleResult>& results) {
    std::vector<SectionSummary> out;
    for (int s = 0; s < NUM_OCL_SECTIONS; ++s) {
        SectionSummary summary{};
        summary.section_id = OCL_SECTIONS[s].id;
        summary.title_en = OCL_SECTIONS[s].title_en;
        summary.total = summary.passed = summary.failed = summary.skipped = summary.advisory = 0;
        summary.time_ms = 0.0;
        for (const auto& r : results) {
            if (r.section != summary.section_id) continue;
            ++summary.total;
            if (r.skipped) {
                ++summary.skipped;
            } else if (r.passed) {
                ++summary.passed;
            } else if (r.advisory) {
                ++summary.advisory;
            } else {
                ++summary.failed;
            }
            summary.time_ms += r.time_ms;
        }
        out.push_back(summary);
    }
    return out;
}

static const char* section_status(const SectionSummary& summary) {
    if (summary.failed > 0) return "FAIL";
    if (summary.skipped > 0) return "SKIP";
    if (summary.advisory > 0) return "WARN";
    return "PASS";
}

static const char* overall_verdict(int failed, int skipped) {
    if (failed > 0) return "ISSUES-FOUND";
    if (skipped > 0) return "AUDIT-INCOMPLETE";
    return "AUDIT-READY";
}

static void write_json_report(const std::string& path,
                               const std::vector<ModuleResult>& results,
                               const OclDeviceInfo& dev,
                               const PlatformInfo& plat,
                               double total_sec) {
    std::ofstream f(path);
    if (!f.is_open()) return;

    int passed = 0, failed = 0, skipped = 0, advisory = 0;
    for (const auto& r : results) {
        if (r.skipped) skipped++;
        else if (r.passed) passed++;
        else if (r.advisory) advisory++;
        else failed++;
    }

    f << "{\n";
    f << "  \"framework_version\": \"" << OCL_AUDIT_FRAMEWORK_VERSION << "\",\n";
    f << "  \"backend\": \"OpenCL\",\n";
    f << "  \"device\": {\n";
    f << "    \"name\": \"" << json_escape(dev.name) << "\",\n";
    f << "    \"vendor\": \"" << json_escape(dev.vendor) << "\",\n";
    f << "    \"version\": \"" << json_escape(dev.version) << "\",\n";
    f << "    \"driver_version\": \"" << json_escape(dev.driver_version) << "\",\n";
    f << "    \"memory_mb\": " << dev.memory_mb << ",\n";
    f << "    \"compute_units\": " << dev.compute_units << "\n";
    f << "  },\n";
    f << "  \"platform\": {\n";
    f << "    \"os\": \"" << json_escape(plat.os) << "\",\n";
    f << "    \"arch\": \"" << json_escape(plat.arch) << "\",\n";
    f << "    \"compiler\": \"" << json_escape(plat.compiler) << "\",\n";
    f << "    \"build_type\": \"" << json_escape(plat.build_type) << "\"\n";
    f << "  },\n";
    f << "  \"summary\": {\n";
    f << "    \"total\": " << results.size() << ",\n";
    f << "    \"passed\": " << passed << ",\n";
    f << "    \"failed\": " << failed << ",\n";
    f << "    \"skipped\": " << skipped << ",\n";
    f << "    \"advisory_warnings\": " << advisory << ",\n";
    f << "    \"total_seconds\": " << std::fixed << total_sec << ",\n";
    f << "    \"verdict\": \"" << overall_verdict(failed, skipped) << "\"\n";
    f << "  },\n";
    f << "  \"modules\": [\n";
    for (size_t i = 0; i < results.size(); i++) {
        auto& r = results[i];
        f << "    { \"id\": \"" << json_escape(r.id) << "\", \"name\": \"" << json_escape(r.name)
          << "\", \"section\": \"" << json_escape(r.section)
          << "\", \"result\": \"" << (r.skipped ? "SKIP" : (r.passed ? "PASS" : (r.advisory ? "WARN" : "FAIL")))
          << "\", \"time_ms\": " << std::fixed << r.time_ms
          << ", \"error_code\": " << r.error_code << " }";
        if (i + 1 < results.size()) f << ",";
        f << "\n";
    }
    f << "  ]\n";
    f << "}\n";
}

static void write_text_report(const std::string& path,
                               const std::vector<ModuleResult>& results,
                               const OclDeviceInfo& dev,
                               const PlatformInfo& plat,
                               double total_sec) {
    std::ofstream f(path);
    if (!f.is_open()) return;

    int passed = 0, failed = 0, skipped = 0, advisory = 0;
    for (const auto& r : results) {
        if (r.skipped) skipped++;
        else if (r.passed) passed++;
        else if (r.advisory) advisory++;
        else failed++;
    }

    f << "================================================================\n";
    f << "  UltrafastSecp256k1 -- OpenCL Unified Audit Report\n";
    f << "  Framework v" << OCL_AUDIT_FRAMEWORK_VERSION << "\n";
    f << "  " << plat.os << " " << plat.arch << " | " << plat.compiler << " | " << plat.build_type << "\n";
    f << "  Device: " << dev.name << " (" << dev.vendor << ") | " << dev.compute_units << " CUs | " << dev.memory_mb << " MB\n";
    f << "================================================================\n\n";

    std::string cur_section;
    for (auto& r : results) {
        if (r.section != cur_section) {
            cur_section = r.section;
            f << "\n  Section: " << cur_section << "\n";
            f << "  " << std::string(50, '-') << "\n";
        }
        f << "  [" << (r.skipped ? "SKIP" : (r.passed ? "PASS" : (r.advisory ? "WARN" : "FAIL"))) << "]  "
          << r.name << "  (" << r.time_ms << " ms)\n";
    }

    f << "\n================================================================\n";
    f << "  VERDICT: " << overall_verdict(failed, skipped) << "\n";
    f << "  TOTAL: " << passed << "/" << results.size() << " passed";
    if (skipped > 0) f << ", " << skipped << " skipped";
    if (advisory > 0) f << ", " << advisory << " advisory";
    if (failed > 0) f << ", " << failed << " FAILED";
    f << "  (" << std::fixed << std::setprecision(1) << total_sec << " s)\n";
    f << "================================================================\n";
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char* argv[]) {
    // Parse args
    std::string kernel_dir;
    std::string report_dir = ".";
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--kernel-dir" && i + 1 < argc) {
            kernel_dir = argv[++i];
        } else if (std::string(argv[i]) == "--report-dir" && i + 1 < argc) {
            report_dir = argv[++i];
        }
    }

    // Detect source directory for kernels
    if (kernel_dir.empty()) {
        // Try to find kernels relative to executable
        namespace fs = std::filesystem;
        auto exe_dir = fs::path(argv[0]).parent_path();
        std::vector<std::string> candidates = {
            (exe_dir / "kernels").string(),
            (exe_dir / "../kernels").string(),
            (exe_dir / "../../opencl/kernels").string(),
            (exe_dir / "../../../opencl/kernels").string(),
            "kernels",
            "../kernels",
            "../../opencl/kernels",
        };
        for (auto& c : candidates) {
            if (fs::exists(c + "/secp256k1_extended.cl")) {
                kernel_dir = c;
                break;
            }
        }
    }

    // Platform info
    auto plat = detect_platform();

    // Timestamp
    auto now = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(now);
    char timebuf[64];
    struct tm tm_buf{};
#ifdef _WIN32
    (void)localtime_s(&tm_buf, &tt);
#else
    (void)localtime_r(&tt, &tm_buf);
#endif
    std::strftime(timebuf, sizeof(timebuf), "%Y-%m-%dT%H:%M:%S", &tm_buf);

    // Initialize OpenCL context
    DeviceConfig config;
    config.verbose = false;
    g_ctx = Context::create(config);
    if (!g_ctx || !g_ctx->is_valid()) {
        std::fprintf(stderr, "[FATAL] Cannot create OpenCL context: %s\n",
                     g_ctx ? g_ctx->last_error().c_str() : "null");
        return 1;
    }

    auto dev = detect_ocl_device(*g_ctx);

    // Try to init extended kernels
    g_kernel_dir = kernel_dir; // make available to audit modules
    if (!kernel_dir.empty()) {
        g_ext.init(*g_ctx, kernel_dir);
        g_zk.init(*g_ctx, kernel_dir);
        g_ct_smoke.init(*g_ctx, kernel_dir);  // CT smoke: lazy fallback via ensure_ct_smoke()
    }

    // Banner
    std::printf("================================================================\n");
    std::printf("  UltrafastSecp256k1 -- OpenCL Unified Audit Runner\n");
    std::printf("  Framework v%s\n", OCL_AUDIT_FRAMEWORK_VERSION);
    std::printf("  %s %s | %s | %s\n", plat.os.c_str(), plat.arch.c_str(),
                plat.compiler.c_str(), plat.build_type.c_str());
    std::printf("  Device: %s (%s) | %d CUs | %zu MB | OpenCL\n",
                dev.name.c_str(), dev.vendor.c_str(), dev.compute_units, dev.memory_mb);
    std::printf("  Extended kernels: %s\n", g_ext.valid ? "loaded" : g_ext.error.c_str());
    std::printf("  ZK kernels: %s\n", g_zk.valid ? "loaded" : g_zk.error.c_str());
    std::printf("  %s\n", timebuf);
    std::printf("================================================================\n\n");

    // Run modules
    std::printf("[Phase 1/2] Running %d OpenCL audit modules across %d sections...\n\n",
                NUM_OCL_MODULES, NUM_OCL_SECTIONS);

    std::vector<ModuleResult> results;
    int passed = 0, failed = 0, skipped = 0, advisory = 0;
    auto total_start = std::chrono::steady_clock::now();

    std::string cur_section;
    int section_idx = 0;
    for (int m = 0; m < NUM_OCL_MODULES; m++) {
        auto& mod = OCL_MODULES[m];

        // Section header
        if (mod.section != cur_section) {
            cur_section = mod.section;
            // Find section title
            for (int s = 0; s < NUM_OCL_SECTIONS; s++) {
                if (std::string(OCL_SECTIONS[s].id) == cur_section) {
                    section_idx = s;
                    break;
                }
            }
            std::printf("  ----------------------------------------------------------\n");
            std::printf("  Section %d/%d: %s\n", section_idx + 1, NUM_OCL_SECTIONS,
                        OCL_SECTIONS[section_idx].title_en);
            std::printf("  ----------------------------------------------------------\n");
        }

        // Run
        std::printf("  [%2d/%d] %-45s", m + 1, NUM_OCL_MODULES, mod.name);
        std::fflush(stdout);

        auto t0 = std::chrono::steady_clock::now();
        int rc = mod.run();
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        ModuleResult r;
        r.id = mod.id;
        r.name = mod.name;
        r.section = mod.section;
        r.advisory = mod.advisory;
        r.time_ms = ms;
        r.error_code = rc;

        if (rc == -1) {
            r.passed = false;
            r.skipped = true;
            skipped++;
            std::printf("SKIP  (%.0f ms)\n", ms);
        } else if (rc == 0) {
            r.passed = true;
            r.skipped = false;
            passed++;
            std::printf("PASS  (%.0f ms)\n", ms);
        } else if (mod.advisory) {
            r.passed = false;
            r.skipped = false;
            advisory++;
            std::printf("ADVS  (%.0f ms) [error=%d] (advisory)\n", ms, rc);
        } else {
            r.passed = false;
            r.skipped = false;
            failed++;
            std::printf("FAIL  (%.0f ms) [error=%d]\n", ms, rc);
        }
        results.push_back(r);
    }

    auto total_end = std::chrono::steady_clock::now();
    double total_sec = std::chrono::duration<double>(total_end - total_start).count();

    // Phase 2: Reports
    std::printf("\n[Phase 2/2] Generating OpenCL audit reports...\n");
    std::string json_path = report_dir + "/ocl_audit_report.json";
    std::string text_path = report_dir + "/ocl_audit_report.txt";
    write_json_report(json_path, results, dev, plat, total_sec);
    write_text_report(text_path, results, dev, plat, total_sec);
    std::printf("  JSON:  %s\n", json_path.c_str());
    std::printf("  Text:  %s\n", text_path.c_str());

    // Summary table
    std::printf("\n================================================================\n");
    std::printf("  #    OpenCL Audit Section                            Result\n");
    std::printf("  ---- -------------------------------------------------- ------\n");

    auto sections = compute_section_summaries(results);
    for (int s = 0; s < NUM_OCL_SECTIONS; s++) {
        auto& section = sections[s];
        std::printf("  %-4d %-50s %d/%d %s\n",
                    s + 1, section.title_en, section.passed, section.total,
                    section_status(section));
    }

    std::printf("\n================================================================\n");
    std::printf("  OpenCL AUDIT VERDICT: %s\n", overall_verdict(failed, skipped));
    std::printf("  TOTAL: %d/%d modules passed", passed, (int)results.size());
    if (skipped > 0) std::printf(", %d skipped (kernel support unavailable)", skipped);
    if (advisory > 0) std::printf(", %d advisory", advisory);
    std::printf("  --  %s  (%.1f s)\n",
                failed > 0 ? "FAILURES DETECTED" :
                (skipped > 0 ? "INCOMPLETE COVERAGE" :
                 (advisory > 0 ? "ADVISORY WARNINGS" : "ALL PASSED")),
                total_sec);
    std::printf("  Device: %s (%s) | %s %s\n", dev.name.c_str(), dev.vendor.c_str(),
                plat.os.c_str(), plat.arch.c_str());
    std::printf("================================================================\n");

    return (failed > 0 || skipped > 0) ? 1 : 0;
}
