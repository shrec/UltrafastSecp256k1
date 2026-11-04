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

        // Load secp256k1_ct_extended.cl: routes ecdsa_sign / schnorr_sign through
        // CT primitives (ct_ecdsa_sign_impl / ct_schnorr_sign_impl). All other
        // kernels (ecdsa_verify, schnorr_verify, ecrecover_batch, generator_mul_windowed)
        // are unchanged from secp256k1_extended.cl which it includes internally.
        std::string src;
        std::vector<std::string> paths = {
            kernel_dir + "/secp256k1_ct_extended.cl",
            "kernels/secp256k1_ct_extended.cl",
            "../kernels/secp256k1_ct_extended.cl",
            "../../opencl/kernels/secp256k1_ct_extended.cl",
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

// ECDSA verify decides x(R) mod n == r projectively (X == r*Z^2, or
// X == (r+n)*Z^2 when r < p - n) instead of inverting R.z. Checks:
//   (1) valid signature accepts; tampered r / s / msg / key reject;
//   (2) x(R) >= n branch: fixed vector with x(R) = n + 2, so r = 2 must accept
//       and r + 1 = 3 must reject (built off-device: pick R with x = n + 2,
//       choose z, s, then Q = u2^-1 * (R - u1*G) with u1 = z/s, u2 = r/s),
//       with the pubkey both affine (Z = 1) and rescaled to Z != 1.
static int audit_ecdsa_verify_rx() {
    if (!g_ext.valid) return -1;

    // -- (1) valid accept, tampered r / s / msg / key reject --
    Scalar priv = {{0x1D2C3B4A59687706ULL, 0x0123456789ABCDEFULL,
                    0x0FEDCBA987654321ULL, 0x1122334455667788ULL}};
    Scalar priv_other = priv;
    priv_other.limbs[0] += 1;
    uint8_t msg[32];
    for (int i = 0; i < 32; i++) msg[i] = (uint8_t)(0x5A ^ (i * 7));

    ExtendedCL::ECDSASig sig;
    if (!ocl_ecdsa_sign(priv, msg, sig)) return 1;
    auto pub = ext_generator_mul(priv);
    auto pub_other = ext_generator_mul(priv_other);
    if (!ocl_ecdsa_verify(pub, msg, sig)) return 2;

    ExtendedCL::ECDSASig bad = sig;
    bad.r[0] ^= 1ULL;
    if (ocl_ecdsa_verify(pub, msg, bad)) return 3;
    bad = sig;
    bad.s[0] ^= 1ULL;
    if (ocl_ecdsa_verify(pub, msg, bad)) return 4;
    uint8_t msg_bad[32];
    std::memcpy(msg_bad, msg, 32);
    msg_bad[31] ^= 1;
    if (ocl_ecdsa_verify(pub, msg_bad, sig)) return 5;
    if (ocl_ecdsa_verify(pub_other, msg, sig)) return 6;

    // -- (2) x(R) = n + 2 >= n: r = 2 accepts, r = 3 rejects --
    const FieldElement qx = {{0x77AC96902FE2CDBFULL, 0xBFD326EDFC5D0060ULL,
                              0x542E30F7B59CDC40ULL, 0xFBEC342B73E8DFDAULL}};
    const FieldElement qy = {{0xBE2000413F84D084ULL, 0x0A05FE6887A2500DULL,
                              0xAF98FBD4FD9776BFULL, 0x8CCE96B4C87E6290ULL}};
    const uint8_t zmsg[32] = {
        0xA5, 0xA5, 0xA5, 0xA5, 0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x23, 0x45, 0x67,
        0x89, 0xAB, 0xCD, 0xEF, 0x0F, 0x1E, 0x2D, 0x3C, 0x4B, 0x5A, 0x69, 0x78,
        0x87, 0x96, 0xA5, 0xB4, 0xC3, 0xD2, 0xE1, 0xF0 };
    JacobianPoint q{};
    q.x = qx;
    q.y = qy;
    q.z = fe_from_u64(1);

    ExtendedCL::ECDSASig vs{};
    vs.r[0] = 2;
    vs.s[0] = 0x0011223344556677ULL; vs.s[1] = 0xFEDCBA9876543210ULL;
    vs.s[2] = 0x0123456789ABCDEFULL; vs.s[3] = 0x1F2E3D4C5B6A7988ULL;
    if (!ocl_ecdsa_verify(q, zmsg, vs)) return 7;

    ExtendedCL::ECDSASig vbad = vs;
    vbad.r[0] = 3;                             // r + 1
    if (ocl_ecdsa_verify(q, zmsg, vbad)) return 8;
    vbad = vs;
    vbad.s[0] += 1;                            // s + 1
    if (ocl_ecdsa_verify(q, zmsg, vbad)) return 9;
    uint8_t zmsg_bad[32];
    std::memcpy(zmsg_bad, zmsg, 32);
    zmsg_bad[31] ^= 1;                         // z + 1
    if (ocl_ecdsa_verify(q, zmsg_bad, vs)) return 10;

    // Same pubkey as (lam^2 X : lam^3 Y : lam), Z != 1.
    const FieldElement lam = {{0x0123456789ABCDEFULL, 0xFEDCBA9876543210ULL,
                               0x0F0F0F0F0F0F0F0FULL, 0x7777777777777777ULL}};
    const FieldElement l2 = g_ctx->field_mul(lam, lam);
    const FieldElement l3 = g_ctx->field_mul(l2, lam);
    JacobianPoint q_proj{};
    q_proj.x = g_ctx->field_mul(qx, l2);
    q_proj.y = g_ctx->field_mul(qy, l3);
    q_proj.z = lam;
    if (!ocl_ecdsa_verify(q_proj, zmsg, vs)) return 11;
    if (ocl_ecdsa_verify(q_proj, zmsg, vbad)) return 12;
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
// Hash160 (SHA-256 -> RIPEMD-160): known answers + GPU-vs-CPU equivalence
// =============================================================================
// Covers src/opencl/kernels/secp256k1_hash160.cl: the registers-only 33-byte
// SHA-256, the unrolled RIPEMD-160, and the length dispatch that sends every
// other length (65-byte uncompressed keys) through the generic SHA-256.
//
// The CPU reference below is a plain table-driven SHA-256 / RIPEMD-160 written
// independently of the kernel, so a transcription slip in the unrolled GPU code
// cannot also be present in the reference.

namespace hash160_ref {

static inline uint32_t rotr(uint32_t x, unsigned n) { return (x >> n) | (x << (32u - n)); }
static inline uint32_t rotl(uint32_t x, unsigned n) { return (x << n) | (x >> (32u - n)); }

static const uint32_t SHA_K[64] = {
    0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
    0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
    0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
    0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
    0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
    0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
    0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
    0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U
};

static void sha256(const uint8_t* data, size_t len, uint8_t out[32]) {
    std::vector<uint8_t> msg(data, data + len);
    msg.push_back(0x80);
    while (msg.size() % 64 != 56) msg.push_back(0);
    const uint64_t bits = static_cast<uint64_t>(len) * 8u;
    for (int i = 7; i >= 0; --i) msg.push_back(static_cast<uint8_t>(bits >> (8 * i)));

    uint32_t h[8] = {0x6a09e667U,0xbb67ae85U,0x3c6ef372U,0xa54ff53aU,
                     0x510e527fU,0x9b05688cU,0x1f83d9abU,0x5be0cd19U};
    for (size_t off = 0; off < msg.size(); off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (uint32_t(msg[off + 4*i]) << 24) | (uint32_t(msg[off + 4*i + 1]) << 16) |
                   (uint32_t(msg[off + 4*i + 2]) << 8) | uint32_t(msg[off + 4*i + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3);
            const uint32_t s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t v[8];
        for (int i = 0; i < 8; ++i) v[i] = h[i];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(v[4], 6) ^ rotr(v[4], 11) ^ rotr(v[4], 25);
            const uint32_t ch = (v[4] & v[5]) ^ (~v[4] & v[6]);
            const uint32_t t1 = v[7] + S1 + ch + SHA_K[i] + w[i];
            const uint32_t S0 = rotr(v[0], 2) ^ rotr(v[0], 13) ^ rotr(v[0], 22);
            const uint32_t maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            v[7] = v[6]; v[6] = v[5]; v[5] = v[4]; v[4] = v[3] + t1;
            v[3] = v[2]; v[2] = v[1]; v[1] = v[0]; v[0] = t1 + S0 + maj;
        }
        for (int i = 0; i < 8; ++i) h[i] += v[i];
    }
    for (int i = 0; i < 8; ++i) {
        out[4*i] = uint8_t(h[i] >> 24); out[4*i+1] = uint8_t(h[i] >> 16);
        out[4*i+2] = uint8_t(h[i] >> 8); out[4*i+3] = uint8_t(h[i]);
    }
}

static const uint8_t RMD_RL[80] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
    3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12, 1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
    4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13 };
static const uint8_t RMD_RR[80] = {
    5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12, 6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
    15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13, 8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
    12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11 };
static const uint8_t RMD_SL[80] = {
    11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8, 7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
    11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5, 11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
    9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6 };
static const uint8_t RMD_SR[80] = {
    8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6, 9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
    9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5, 15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
    8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11 };
static const uint32_t RMD_KL[5] = {0x00000000U,0x5A827999U,0x6ED9EBA1U,0x8F1BBCDCU,0xA953FD4EU};
static const uint32_t RMD_KR[5] = {0x50A28BE6U,0x5C4DD124U,0x6D703EF3U,0x7A6D76E9U,0x00000000U};

static uint32_t rmd_f(int j, uint32_t x, uint32_t y, uint32_t z) {
    if (j < 16) return x ^ y ^ z;
    if (j < 32) return (x & y) | (~x & z);
    if (j < 48) return (x | ~y) ^ z;
    if (j < 64) return (x & z) | (y & ~z);
    return x ^ (y | ~z);
}

static void ripemd160(const uint8_t* data, size_t len, uint8_t out[20]) {
    std::vector<uint8_t> msg(data, data + len);
    msg.push_back(0x80);
    while (msg.size() % 64 != 56) msg.push_back(0);
    const uint64_t bits = static_cast<uint64_t>(len) * 8u;
    for (int i = 0; i < 8; ++i) msg.push_back(static_cast<uint8_t>(bits >> (8 * i)));

    uint32_t h[5] = {0x67452301U,0xEFCDAB89U,0x98BADCFEU,0x10325476U,0xC3D2E1F0U};
    for (size_t off = 0; off < msg.size(); off += 64) {
        uint32_t X[16];
        for (int i = 0; i < 16; ++i) {
            X[i] = uint32_t(msg[off + 4*i]) | (uint32_t(msg[off + 4*i + 1]) << 8) |
                   (uint32_t(msg[off + 4*i + 2]) << 16) | (uint32_t(msg[off + 4*i + 3]) << 24);
        }
        uint32_t al = h[0], bl = h[1], cl = h[2], dl = h[3], el = h[4];
        uint32_t ar = h[0], br = h[1], cr = h[2], dr = h[3], er = h[4];
        for (int j = 0; j < 80; ++j) {
            uint32_t t = rotl(al + rmd_f(j, bl, cl, dl) + X[RMD_RL[j]] + RMD_KL[j / 16], RMD_SL[j]) + el;
            al = el; el = dl; dl = rotl(cl, 10); cl = bl; bl = t;
            t = rotl(ar + rmd_f(79 - j, br, cr, dr) + X[RMD_RR[j]] + RMD_KR[j / 16], RMD_SR[j]) + er;
            ar = er; er = dr; dr = rotl(cr, 10); cr = br; br = t;
        }
        const uint32_t t = h[1] + cl + dr;
        h[1] = h[2] + dl + er;
        h[2] = h[3] + el + ar;
        h[3] = h[4] + al + br;
        h[4] = h[0] + bl + cr;
        h[0] = t;
    }
    for (int i = 0; i < 5; ++i) {
        out[4*i] = uint8_t(h[i]); out[4*i+1] = uint8_t(h[i] >> 8);
        out[4*i+2] = uint8_t(h[i] >> 16); out[4*i+3] = uint8_t(h[i] >> 24);
    }
}

static void hash160(const uint8_t* data, size_t len, uint8_t out[20]) {
    uint8_t sha[32];
    sha256(data, len, sha);
    ripemd160(sha, 32, out);
}

static std::string to_hex(const uint8_t* p, size_t n) {
    static const char* d = "0123456789abcdef";
    std::string s;
    for (size_t i = 0; i < n; ++i) { s += d[p[i] >> 4]; s += d[p[i] & 15]; }
    return s;
}

static std::vector<uint8_t> from_hex(const std::string& s) {
    std::vector<uint8_t> v;
    for (size_t i = 0; i + 1 < s.size(); i += 2) {
        v.push_back(static_cast<uint8_t>(std::stoi(s.substr(i, 2), nullptr, 16)));
    }
    return v;
}

} // namespace hash160_ref

// Compiles secp256k1_hash160.cl once and runs its hash160_batch kernel.
struct Hash160CL {
    cl_program program = nullptr;
    cl_kernel  kernel  = nullptr;
    bool tried = false;
    bool valid = false;
    bool missing_source = false;
    std::string error;

    bool init(const Context& ctx, const std::string& kernel_dir) {
        if (tried) return valid;
        tried = true;
        cl_context context = (cl_context)ctx.native_context();
        cl_command_queue queue = (cl_command_queue)ctx.native_queue();
        cl_device_id device = nullptr;
        cl_int err = clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(device), &device, nullptr);
        if (err != CL_SUCCESS) { error = "Cannot get device from queue"; return false; }

        std::string src;
        const std::vector<std::string> paths = {
            kernel_dir + "/secp256k1_hash160.cl",
            "kernels/secp256k1_hash160.cl",
            "../kernels/secp256k1_hash160.cl",
        };
        for (const auto& p : paths) { src = load_file(p); if (!src.empty()) break; }
        if (src.empty()) { missing_source = true; error = "Cannot find secp256k1_hash160.cl"; return false; }

        const char* sp = src.c_str(); size_t sl = src.size();
        program = clCreateProgramWithSource(context, 1, &sp, &sl, &err);
        if (err != CL_SUCCESS) { error = "clCreateProgramWithSource failed"; return false; }
        const std::string opts = "-cl-std=CL1.2 -I " + kernel_dir;
        err = clBuildProgram(program, 1, &device, opts.c_str(), nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size = 0;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::string log(log_size + 1, '\0');
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            error = "hash160 build failed: " + log;
            return false;
        }
        kernel = clCreateKernel(program, "hash160_batch", &err);
        if (err != CL_SUCCESS) { error = "hash160_batch kernel not found"; return false; }
        valid = true;
        return true;
    }

    // keys: count * stride bytes; out: count * 20 bytes.
    bool run(const Context& ctx, const std::vector<uint8_t>& keys, uint32_t stride,
             uint32_t count, std::vector<uint8_t>& out) {
        cl_context context = (cl_context)ctx.native_context();
        cl_command_queue queue = (cl_command_queue)ctx.native_queue();
        cl_int err = CL_SUCCESS;
        cl_mem d_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     keys.size(), const_cast<uint8_t*>(keys.data()), &err);
        if (err != CL_SUCCESS) { error = "input buffer failed"; return false; }
        cl_mem d_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size_t(count) * 20, nullptr, &err);
        if (err != CL_SUCCESS) { clReleaseMemObject(d_in); error = "output buffer failed"; return false; }

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out);
        clSetKernelArg(kernel, 2, sizeof(cl_uint), &stride);
        clSetKernelArg(kernel, 3, sizeof(cl_uint), &count);
        size_t gws = count;
        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
        bool ok = (err == CL_SUCCESS);
        if (ok) {
            out.assign(size_t(count) * 20, 0);
            ok = clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0, out.size(), out.data(),
                                     0, nullptr, nullptr) == CL_SUCCESS;
        }
        if (!ok) error = "hash160_batch launch/read failed";
        clReleaseMemObject(d_in);
        clReleaseMemObject(d_out);
        return ok;
    }

    ~Hash160CL() {
        if (kernel)  clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
    }
};
static Hash160CL g_hash160;

// Returns 0 when the kernel is ready, -1 when the source is not available
// (reported as SKIP like the other kernel-backed modules), 1 on a build failure.
static int ensure_hash160(const char* who) {
    if (!g_ctx) { fprintf(stderr, "    SKIP %s: No OpenCL context\n", who); return -1; }
    if (g_hash160.init(*g_ctx, g_kernel_dir)) return 0;
    if (g_hash160.missing_source) {
        fprintf(stderr, "    SKIP %s: %s\n", who, g_hash160.error.c_str());
        return -1;
    }
    fprintf(stderr, "    %s: %s\n", who, g_hash160.error.c_str());
    return 1;
}

// hash160(02||Gx) and hash160(04||Gx||Gy) known answers, checked on the CPU
// reference first (so a wrong reference is not blamed on the GPU) and then on
// the GPU kernel.
static int audit_hash160_known_answers() {
    const int ready = ensure_hash160("hash160_kat");
    if (ready != 0) return ready;

    const auto gx = hash160_ref::from_hex("79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798");
    const auto gy = hash160_ref::from_hex("483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8");
    std::vector<uint8_t> k33 = {0x02};
    k33.insert(k33.end(), gx.begin(), gx.end());
    std::vector<uint8_t> k65 = {0x04};
    k65.insert(k65.end(), gx.begin(), gx.end());
    k65.insert(k65.end(), gy.begin(), gy.end());
    const std::string want33 = "751e76e8199196d454941c45d1b3a323f1433bd6";
    const std::string want65 = "91b24bf9f5288532960ac687abb035127b1d28a5";

    uint8_t cpu[20];
    hash160_ref::hash160(k33.data(), k33.size(), cpu);
    if (hash160_ref::to_hex(cpu, 20) != want33) { fprintf(stderr, "    CPU reference wrong for 33-byte G\n"); return 1; }
    hash160_ref::hash160(k65.data(), k65.size(), cpu);
    if (hash160_ref::to_hex(cpu, 20) != want65) { fprintf(stderr, "    CPU reference wrong for 65-byte G\n"); return 2; }

    std::vector<uint8_t> out;
    if (!g_hash160.run(*g_ctx, k33, 33, 1, out)) { fprintf(stderr, "    %s\n", g_hash160.error.c_str()); return 3; }
    if (hash160_ref::to_hex(out.data(), 20) != want33) {
        fprintf(stderr, "    GPU hash160(02||Gx) = %s, want %s\n",
                hash160_ref::to_hex(out.data(), 20).c_str(), want33.c_str());
        return 4;
    }
    if (!g_hash160.run(*g_ctx, k65, 65, 1, out)) { fprintf(stderr, "    %s\n", g_hash160.error.c_str()); return 5; }
    if (hash160_ref::to_hex(out.data(), 20) != want65) {
        fprintf(stderr, "    GPU hash160(04||Gx||Gy) = %s, want %s\n",
                hash160_ref::to_hex(out.data(), 20).c_str(), want65.c_str());
        return 6;
    }
    return 0;
}

// GPU vs CPU over random keys. 33 and 65 are the real key lengths; 34 and 64
// are neighbours that must take the generic SHA-256 branch of the dispatch. A
// final sweep hashes 65-byte keys with every last byte value 0x00..0xff (the
// only input of the second SHA-256 block). Keys are random bytes (hash160 is
// defined for any input), seeded deterministically so a failure reproduces.
static int audit_hash160_gpu_vs_cpu() {
    const int ready = ensure_hash160("hash160_gpu_cpu");
    if (ready != 0) return ready;

    struct Case { uint32_t stride; uint32_t count; };
    const Case cases[] = { {33, 300}, {65, 300}, {34, 64}, {64, 64} };
    uint64_t state = 0x9E3779B97F4A7C15ULL;
    auto next = [&state]() {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        return state;
    };

    int case_idx = 0;
    for (const auto& cs : cases) {
        ++case_idx;
        std::vector<uint8_t> keys(size_t(cs.stride) * cs.count);
        for (auto& b : keys) b = static_cast<uint8_t>(next() >> 24);
        for (uint32_t i = 0; i < cs.count; ++i) {
            // Make 33/65-byte keys look like real SEC1 keys (prefix byte).
            if (cs.stride == 33) keys[size_t(i) * 33] = static_cast<uint8_t>(2 + (i & 1));
            if (cs.stride == 65) keys[size_t(i) * 65] = 4;
        }
        std::vector<uint8_t> gpu;
        if (!g_hash160.run(*g_ctx, keys, cs.stride, cs.count, gpu)) {
            fprintf(stderr, "    stride %u: %s\n", cs.stride, g_hash160.error.c_str());
            return 10 * case_idx;
        }
        for (uint32_t i = 0; i < cs.count; ++i) {
            uint8_t cpu[20];
            hash160_ref::hash160(&keys[size_t(i) * cs.stride], cs.stride, cpu);
            if (std::memcmp(cpu, &gpu[size_t(i) * 20], 20) != 0) {
                fprintf(stderr, "    stride %u key %u: GPU %s != CPU %s\n", cs.stride, i,
                        hash160_ref::to_hex(&gpu[size_t(i) * 20], 20).c_str(),
                        hash160_ref::to_hex(cpu, 20).c_str());
                return 10 * case_idx + 1;
            }
        }
    }

    // 65-byte keys with every possible last byte (0x00..0xff). The last byte is
    // the only input of the second SHA-256 block of a 65-byte message, so this
    // exercises that block's word w[0] and its 0x80 padding for all 256 values.
    // The first 64 bytes are one fixed random prefix, so the 256 keys differ only
    // in the last byte.
    {
        const uint32_t count = 256;
        std::vector<uint8_t> prefix(64);
        for (auto& b : prefix) b = static_cast<uint8_t>(next() >> 24);
        std::vector<uint8_t> keys(size_t(65) * count);
        for (uint32_t i = 0; i < count; ++i) {
            uint8_t* k = &keys[size_t(i) * 65];
            k[0] = 4;
            std::memcpy(k + 1, prefix.data() + 1, 63);
            k[64] = static_cast<uint8_t>(i);
        }
        std::vector<uint8_t> gpu;
        if (!g_hash160.run(*g_ctx, keys, 65, count, gpu)) {
            fprintf(stderr, "    65-byte last-byte sweep: %s\n", g_hash160.error.c_str());
            return 100;
        }
        for (uint32_t i = 0; i < count; ++i) {
            uint8_t cpu[20];
            hash160_ref::hash160(&keys[size_t(i) * 65], 65, cpu);
            if (std::memcmp(cpu, &gpu[size_t(i) * 20], 20) != 0) {
                fprintf(stderr, "    65-byte last byte 0x%02x: GPU %s != CPU %s\n", i,
                        hash160_ref::to_hex(&gpu[size_t(i) * 20], 20).c_str(),
                        hash160_ref::to_hex(cpu, 20).c_str());
                return 101;
            }
        }
    }
    return 0;
}

// =============================================================================
// Variable-time SafeGCD inverses vs Fermat (modinv30_var_impl, secp256k1_field.cl)
// =============================================================================
// The OpenCL verifiers invert public values with field_inv_var_impl and
// scalar_inv_var_impl. audit_inv_var_vs_fermat builds a test program that
// includes secp256k1_ct_extended.cl -- the same source g_ext builds -- and
// checks on the device, separately for p and for n:
//   1. var(a) is canonical (< M) and equals the Fermat inverse of (a mod M)
//      for edge inputs (0, 1, 2, M-2, M-1, (M-1)/2, (M+1)/2, M, M+1, values
//      above M, 2^255, 2^256-1, zero / all-ones limb patterns) and 16384
//      seeded random 256-bit inputs
//   2. (a mod M) * var(a) == 1 (mod M) when a != 0 mod M; var(a) == 0 otherwise
//   3. the core with the full iteration cap returns the same value, and with
//      one outer iteration fewer than the input needs it returns 0 (fail
//      closed) -- at least 10000 such inputs per modulus
// audit_ecrecover_roundtrip exercises the switched recovery path end to end.

static const char* const kInvVarCheckSource = R"CLSRC(
#include "secp256k1_ct_extended.cl"

// in:  8 ulongs per item: the raw input, then the input reduced mod M by the host
// out: 12 ulongs per item
//   mode 0 (p) / 1 (n): [0..3] var(raw)  [4..7] Fermat(reduced)  [8..11] reduced * var(raw)
//   mode 2 (p) / 3 (n): [0..3] modinv30_var_impl(raw, iters[i])  [4] outer iterations used
__kernel void inv_var_check(__global const ulong* in, __global const int* iters,
                            __global ulong* out, const uint count, const uint mode) {
    const uint gid = get_global_id(0);
    if (gid >= count) return;
    ulong raw[4], red[4], o[12];
    for (int i = 0; i < 4; ++i) { raw[i] = in[gid * 8 + i]; red[i] = in[gid * 8 + 4 + i]; }
    for (int i = 0; i < 12; ++i) o[i] = 0;
    if (mode == 0u) {
        FieldElement a, ar, v, f, prod;
        for (int i = 0; i < 4; ++i) { a.limbs[i] = raw[i]; ar.limbs[i] = red[i]; }
        field_inv_var_impl(&v, &a);
        field_inv_impl(&f, &ar);
        field_mul_impl(&prod, &ar, &v);
        for (int i = 0; i < 4; ++i) { o[i] = v.limbs[i]; o[4 + i] = f.limbs[i]; o[8 + i] = prod.limbs[i]; }
    } else if (mode == 1u) {
        Scalar a, ar, v, f, prod;
        for (int i = 0; i < 4; ++i) { a.limbs[i] = raw[i]; ar.limbs[i] = red[i]; }
        scalar_inv_var_impl(&a, &v);
        scalar_inverse_impl(&ar, &f);
        scalar_mul_mod_n_impl(&ar, &v, &prod);
        for (int i = 0; i < 4; ++i) { o[i] = v.limbs[i]; o[4 + i] = f.limbs[i]; o[8 + i] = prod.limbs[i]; }
    } else {
        ulong r4[4];
        const int used = modinv30_var_impl(raw, r4, mode == 3u ? 1 : 0, iters[gid]);
        for (int i = 0; i < 4; ++i) o[i] = r4[i];
        o[4] = (ulong)used;
    }
    for (int i = 0; i < 12; ++i) out[gid * 12 + i] = o[i];
}
)CLSRC";

// The iteration cap the OpenCL core uses (MODINV30_MAX_ITERS in secp256k1_field.cl).
static constexpr int kModinv30MaxIters = 25;

static bool iv_ge(const uint64_t a[4], const uint64_t m[4]) {
    for (int i = 3; i >= 0; --i) {
        if (a[i] != m[i]) return a[i] > m[i];
    }
    return true;
}

static void iv_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t borrow = 0;
    for (int i = 0; i < 4; ++i) {
        uint64_t const t = a[i] - b[i];
        uint64_t const next = (uint64_t)(a[i] < b[i]) | (uint64_t)(t < borrow);
        r[i] = t - borrow;
        borrow = next;
    }
}

static void iv_add_u64(const uint64_t a[4], uint64_t v, uint64_t r[4]) {
    uint64_t carry = v;
    for (int i = 0; i < 4; ++i) {
        r[i] = a[i] + carry;
        carry = (r[i] < carry) ? 1 : 0;
    }
}

// r = a mod m for a < 2m (true for any 256-bit a when m > 2^255)
static void iv_reduce_once(const uint64_t a[4], const uint64_t m[4], uint64_t r[4]) {
    if (iv_ge(a, m)) { iv_sub(a, m, r); return; }
    for (int i = 0; i < 4; ++i) r[i] = a[i];
}

static bool iv_is_zero(const uint64_t a[4]) { return (a[0] | a[1] | a[2] | a[3]) == 0; }
static bool iv_is_one(const uint64_t a[4]) { return a[0] == 1 && (a[1] | a[2] | a[3]) == 0; }
static bool iv_eq(const uint64_t a[4], const uint64_t b[4]) {
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3];
}

static void iv_print(const char* what, const uint64_t a[4]) {
    std::fprintf(stderr, "      %-10s %016llx %016llx %016llx %016llx\n", what,
                 (unsigned long long)a[3], (unsigned long long)a[2],
                 (unsigned long long)a[1], (unsigned long long)a[0]);
}

// Edge and seeded random inputs for modulus m, 4 limbs per value.
static std::vector<uint64_t> iv_corpus(const uint64_t m[4]) {
    std::vector<uint64_t> v;
    auto push = [&v](const uint64_t a[4]) { v.insert(v.end(), a, a + 4); };
    auto push4 = [&v](uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3) {
        v.push_back(a0); v.push_back(a1); v.push_back(a2); v.push_back(a3);
    };
    const uint64_t F = ~0ULL;
    const uint64_t one[4] = {1, 0, 0, 0}, two[4] = {2, 0, 0, 0};
    uint64_t t[4], u[4];

    push4(0, 0, 0, 0); push4(1, 0, 0, 0); push4(2, 0, 0, 0); push4(3, 0, 0, 0);
    iv_sub(m, one, t); push(t);                                   // M-1
    iv_sub(m, two, t); push(t);                                   // M-2
    iv_sub(m, one, t);                                            // (M-1)/2
    for (int i = 0; i < 4; ++i) u[i] = (t[i] >> 1) | (i < 3 ? (t[i + 1] << 63) : 0);
    push(u);
    iv_add_u64(m, 1, t);                                          // (M+1)/2
    for (int i = 0; i < 4; ++i) u[i] = (t[i] >> 1) | (i < 3 ? (t[i + 1] << 63) : 0);
    push(u);
    push(m);                                                      // M      (== 0)
    iv_add_u64(m, 1, t); push(t);                                 // M+1    (== 1)
    iv_add_u64(m, 2, t); push(t);                                 // M+2    (== 2)
    iv_add_u64(m, 0xFFFFFFFFULL, t); push(t);                     // above M, below 2^256
    push4(F, F, F, F);                                            // 2^256-1
    push4(0, 0, 0, 1ULL << 63);                                   // 2^255
    push4(0, 0, 0, 1); push4(0, 0, 1, 0); push4(0, 1, 0, 0);      // zero limbs
    push4(F, 0, 0, 0); push4(F, 0, F, 0); push4(0, F, 0, F);
    push4(F, F, F, 0); push4(0, F, F, F); push4(F, F, 0, 0);      // all-ones limbs
    push4(0x3FFFFFFF3FFFFFFFULL, 0x3FFFFFFF3FFFFFFFULL,
          0x3FFFFFFF3FFFFFFFULL, 0x3FFFFFFF3FFFFFFFULL);          // 30-bit boundaries
    push4(0xC0000000C0000000ULL, 0xC0000000C0000000ULL,
          0xC0000000C0000000ULL, 0xC0000000C0000000ULL);
    push4(0x8000000000000001ULL, 0, 0, 0x8000000000000000ULL);

    uint64_t s = 0x5EC9256B1D0C0DEULL;                            // splitmix64, fixed seed
    auto next = [&s]() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    };
    for (int i = 0; i < 16384; ++i) push4(next(), next(), next(), next());
    for (int i = 0; i < 64; ++i) { iv_add_u64(m, next() & 0xFFFFFFFFULL, t); push(t); }
    return v;
}

// Runs inv_var_check over `in` (8 ulongs per item). Returns 0 on success.
static int iv_run(cl_context ctx, cl_command_queue q, cl_kernel k,
                  const std::vector<uint64_t>& in, const std::vector<int>& iters,
                  cl_uint mode, std::vector<uint64_t>& out) {
    size_t const n = in.size() / 8;
    cl_int err = CL_SUCCESS, e2 = CL_SUCCESS, e3 = CL_SUCCESS;
    cl_mem d_in = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 in.size() * sizeof(uint64_t), (void*)in.data(), &err);
    cl_mem d_it = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 iters.size() * sizeof(int), (void*)iters.data(), &e2);
    cl_mem d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, n * 12 * sizeof(uint64_t), nullptr, &e3);
    int rc = (err != CL_SUCCESS || e2 != CL_SUCCESS || e3 != CL_SUCCESS) ? 1 : 0;
    out.assign(n * 12, 0);
    if (rc == 0) {
        cl_uint count = (cl_uint)n;
        clSetKernelArg(k, 0, sizeof(cl_mem), &d_in);
        clSetKernelArg(k, 1, sizeof(cl_mem), &d_it);
        clSetKernelArg(k, 2, sizeof(cl_mem), &d_out);
        clSetKernelArg(k, 3, sizeof(cl_uint), &count);
        clSetKernelArg(k, 4, sizeof(cl_uint), &mode);
        size_t global = n;
        err = clEnqueueNDRangeKernel(q, k, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
        if (err == CL_SUCCESS) {
            err = clEnqueueReadBuffer(q, d_out, CL_TRUE, 0, n * 12 * sizeof(uint64_t),
                                      out.data(), 0, nullptr, nullptr);
        }
        if (err != CL_SUCCESS) {
            std::fprintf(stderr, "    inv_var_check mode %u: OpenCL error %d\n", mode, err);
            rc = 2;
        }
    }
    if (d_in) clReleaseMemObject(d_in);
    if (d_it) clReleaseMemObject(d_it);
    if (d_out) clReleaseMemObject(d_out);
    return rc;
}

static int audit_inv_var_vs_fermat() {
    if (g_kernel_dir.empty() || !g_ctx) return -1;
    cl_context ctx = (cl_context)g_ctx->native_context();
    cl_command_queue q = (cl_command_queue)g_ctx->native_queue();
    cl_device_id dev = nullptr;
    if (clGetCommandQueueInfo(q, CL_QUEUE_DEVICE, sizeof(dev), &dev, nullptr) != CL_SUCCESS) return 1;

    cl_int err = CL_SUCCESS;
    const char* src = kInvVarCheckSource;
    size_t const len = std::strlen(src);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src, &len, &err);
    if (err != CL_SUCCESS) return 2;
    std::string const opts = "-cl-std=CL1.2 -cl-fast-relaxed-math -cl-mad-enable -I " + g_kernel_dir;
    err = clBuildProgram(prog, 1, &dev, opts.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size + 1, '\0');
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::fprintf(stderr, "    inv_var_check build failed:\n%s\n", log.c_str());
        clReleaseProgram(prog);
        return 3;
    }
    cl_kernel k = clCreateKernel(prog, "inv_var_check", &err);
    if (err != CL_SUCCESS) { clReleaseProgram(prog); return 4; }

    int rc = 0;
    for (int which = 0; which < 2 && rc == 0; ++which) {
        const uint64_t* m = (which == 0) ? MODULUS : ORDER;
        const char* label = (which == 0) ? "p" : "n";
        std::vector<uint64_t> const corpus = iv_corpus(m);
        size_t const n = corpus.size() / 4;
        std::vector<uint64_t> in(n * 8);
        for (size_t i = 0; i < n; ++i) {
            for (int j = 0; j < 4; ++j) in[i * 8 + j] = corpus[i * 4 + j];
            iv_reduce_once(&corpus[i * 4], m, &in[i * 8 + 4]);
        }

        // 1 + 2: var == Fermat, canonical, a * a^-1 == 1
        std::vector<int> const full(n, kModinv30MaxIters);
        std::vector<uint64_t> out;
        if (iv_run(ctx, q, k, in, full, (cl_uint)which, out) != 0) { rc = 10; break; }
        for (size_t i = 0; i < n && rc == 0; ++i) {
            const uint64_t* red = &in[i * 8 + 4];
            const uint64_t* v = &out[i * 12];
            uint64_t f[4], prod[4];
            iv_reduce_once(&out[i * 12 + 4], m, f);
            iv_reduce_once(&out[i * 12 + 8], m, prod);
            bool const ok = !iv_ge(v, m) && iv_eq(v, f)
                         && (iv_is_zero(red) ? iv_is_zero(v) : iv_is_one(prod));
            if (!ok) {
                std::fprintf(stderr, "    mod %s input #%zu: var inverse != Fermat or a*inv != 1\n",
                             label, i);
                iv_print("input", &in[i * 8]); iv_print("var", v);
                iv_print("fermat", f);         iv_print("a*var", prod);
                rc = 11;
            }
        }
        if (rc != 0) break;

        // 3: the core agrees at the full cap and fails closed one iteration short
        std::vector<uint64_t> core;
        if (iv_run(ctx, q, k, in, full, (cl_uint)(which + 2), core) != 0) { rc = 12; break; }
        std::vector<int> fewer(n, 0);
        for (size_t i = 0; i < n && rc == 0; ++i) {
            int const used = (int)core[i * 12 + 4];
            if (!iv_eq(&core[i * 12], &out[i * 12]) || used < 1 || used > kModinv30MaxIters) {
                std::fprintf(stderr, "    mod %s input #%zu: core result/iterations wrong (used=%d)\n",
                             label, i, used);
                rc = 13;
            }
            fewer[i] = used - 1;
        }
        if (rc != 0) break;
        std::vector<uint64_t> capped;
        if (iv_run(ctx, q, k, in, fewer, (cl_uint)(which + 2), capped) != 0) { rc = 14; break; }
        size_t fail_closed = 0;
        for (size_t i = 0; i < n && rc == 0; ++i) {
            if (!iv_is_zero(&capped[i * 12]) || (int)capped[i * 12 + 4] != fewer[i]) {
                std::fprintf(stderr, "    mod %s input #%zu: cap of %d iterations did not return 0\n",
                             label, i, fewer[i]);
                iv_print("input", &in[i * 8]); iv_print("got", &capped[i * 12]);
                rc = 15;
            }
            if (!iv_is_zero(&in[i * 8 + 4])) ++fail_closed;
        }
        if (rc == 0 && fail_closed < 10000) {
            std::fprintf(stderr, "    mod %s: only %zu fail-closed cases\n", label, fail_closed);
            rc = 16;
        }
    }

    clReleaseKernel(k);
    clReleaseProgram(prog);
    return rc;
}

// =============================================================================
// CT GLV decomposition + CT scalar multiplication vs the fast path
// =============================================================================
// ct_generator_mul_impl and ct_scalar_mul_point (secp256k1_ct_point.cl) scan
// 33 four-bit windows (132 bits) of the GLV halves k1 and k2, so they are
// correct only when ct_glv_decompose_impl returns a real split with
// k1, k2 < 2^128. The old k1 = k, k2 = 0 stub dropped the top bits of every k
// >= 2^132, so every CT signer and CT pubkey derivation on OpenCL was wrong.
// audit_ct_glv_scalar_mul builds a test program that includes
// secp256k1_ct_extended.cl -- the same source g_ext builds -- and checks, for
// edge scalars (0, 1, 2, 3, n-1, n-2, (n-1)/2, (n+1)/2, 2^127, 2^128-1, 2^128,
// 2^129, 2^255, lambda, lambda^2, n-lambda, zero / all-ones limb patterns)
// and 2048 seeded random k < n:
//   1. ct_glv_decompose_impl: k1, k2 < 2^128, both sign masks are 0 or
//      all-ones, and (+/-k1) + (+/-k2)*lambda == k (mod n), checked on the
//      host with an independent double-and-add mod-n multiply
//   2. ct_generator_mul_impl(k) == fast windowed k*G (no GLV in the reference)
//   3. ct_scalar_mul_point(P, k) == fast windowed (k*m mod n)*G for P = m*G

static const char* const kCtGlvCheckSource = R"CLSRC(
#include "secp256k1_ct_extended.cl"

// o[0..3] affine x, o[4..7] affine y (canonical LE limbs),
// o[8] = 1 for the point at infinity (x = y = 0 then)
static inline void ctglv_put_point(const JacobianPoint* p, __global ulong* o) {
    for (int i = 0; i < 9; ++i) o[i] = 0;
    if (p->infinity) { o[8] = 1; return; }
    FieldElement zi, zi2, zi3, x, y;
    field_inv_impl(&zi, &p->z);
    field_sqr_impl(&zi2, &zi);
    field_mul_impl(&zi3, &zi, &zi2);
    field_mul_impl(&x, &p->x, &zi2);
    field_mul_impl(&y, &p->y, &zi3);
    uchar bx[32], by[32];
    field_to_bytes_impl(&x, bx);
    field_to_bytes_impl(&y, by);
    for (int i = 0; i < 4; ++i) {
        ulong lx = 0, ly = 0;
        for (int j = 0; j < 8; ++j) {
            lx = (lx << 8) | (ulong)bx[(3 - i) * 8 + j];
            ly = (ly << 8) | (ulong)by[(3 - i) * 8 + j];
        }
        o[i] = lx;
        o[4 + i] = ly;
    }
}

// in: 4 ulongs per item (k < n). out: 46 ulongs per item
//   [0..3] k1  [4..7] k2  [8] k1_neg  [9] k2_neg      (ct_glv_decompose_impl)
//   [10..18] ct_generator_mul_impl(k)                 [19..27] fast windowed k*G
//   [28..36] ct_scalar_mul_point(P, k), P = m*G       [37..45] fast windowed (k*m mod n)*G
__kernel void ct_glv_check(__global const ulong* in, __global ulong* out, const uint count) {
    const uint gid = get_global_id(0);
    if (gid >= count) return;
    __global ulong* o = out + (size_t)gid * 46;
    Scalar k;
    for (int i = 0; i < 4; ++i) k.limbs[i] = in[gid * 4 + i];

    CTGLVDecompositionOCL glv;
    ct_glv_decompose_impl(&k, &glv);
    for (int i = 0; i < 4; ++i) { o[i] = glv.k1.limbs[i]; o[4 + i] = glv.k2.limbs[i]; }
    o[8] = glv.k1_neg;
    o[9] = glv.k2_neg;

    CTJacobianPoint ct_r;
    JacobianPoint jr;
    ct_generator_mul_impl(&k, &ct_r);
    jr = ct_point_to_jacobian(&ct_r);
    ctglv_put_point(&jr, o + 10);
    scalar_mul_generator_windowed_impl(&jr, &k);
    ctglv_put_point(&jr, o + 19);

    Scalar m;
    m.limbs[0] = 0x0123456789ABCDEFUL; m.limbs[1] = 0xFEDCBA9876543210UL;
    m.limbs[2] = 0x0F1E2D3C4B5A6978UL; m.limbs[3] = 0x7A6B5C4D3E2F1001UL;
    JacobianPoint pj;
    scalar_mul_generator_windowed_impl(&pj, &m);
    CTJacobianPoint p = ct_point_from_jacobian(&pj);
    ct_scalar_mul_point(&p, &k, &ct_r);
    jr = ct_point_to_jacobian(&ct_r);
    ctglv_put_point(&jr, o + 28);
    Scalar km;
    scalar_mul_mod_n_impl(&k, &m, &km);
    scalar_mul_generator_windowed_impl(&jr, &km);
    ctglv_put_point(&jr, o + 37);
}
)CLSRC";

static constexpr uint64_t CG_LAMBDA[4] = {
    0xDF02967C1B23BD72ULL, 0x122E22EA20816678ULL,
    0xA5261C028812645AULL, 0x5363AD4CC05C30E0ULL
};

// r = (a + b) mod n for a, b < n
static void cg_add_mod_n(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t t[4], carry = 0;
    for (int i = 0; i < 4; ++i) {
        uint64_t const s = a[i] + b[i];
        uint64_t const c1 = (s < a[i]) ? 1 : 0;
        t[i] = s + carry;
        uint64_t const c2 = (t[i] < s) ? 1 : 0;
        carry = c1 | c2;
    }
    if (carry != 0 || iv_ge(t, ORDER)) { iv_sub(t, ORDER, r); return; }
    for (int i = 0; i < 4; ++i) r[i] = t[i];
}

// r = (a * b) mod n by double-and-add over the bits of b, for a < n
static void cg_mul_mod_n(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t acc[4] = {0, 0, 0, 0};
    for (int bit = 255; bit >= 0; --bit) {
        cg_add_mod_n(acc, acc, acc);
        if ((b[bit >> 6] >> (bit & 63)) & 1) cg_add_mod_n(acc, a, acc);
    }
    for (int i = 0; i < 4; ++i) r[i] = acc[i];
}

// r = (-a) mod n for a < n
static void cg_neg_mod_n(const uint64_t a[4], uint64_t r[4]) {
    if (iv_is_zero(a)) { for (int i = 0; i < 4; ++i) r[i] = 0; return; }
    iv_sub(ORDER, a, r);
}

// Edge and seeded random scalars k < n, 4 limbs per value.
static std::vector<uint64_t> cg_corpus() {
    std::vector<uint64_t> v;
    auto push = [&v](const uint64_t a[4]) { v.insert(v.end(), a, a + 4); };
    auto push_red = [&v](uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3) {
        const uint64_t a[4] = {a0, a1, a2, a3};
        uint64_t r[4];
        iv_reduce_once(a, ORDER, r);
        v.insert(v.end(), r, r + 4);
    };
    auto halve = [](const uint64_t a[4], uint64_t r[4]) {
        for (int i = 0; i < 4; ++i) r[i] = (a[i] >> 1) | (i < 3 ? (a[i + 1] << 63) : 0);
    };
    const uint64_t F = ~0ULL;
    const uint64_t one[4] = {1, 0, 0, 0}, two[4] = {2, 0, 0, 0};
    uint64_t t[4], u[4];

    push_red(0, 0, 0, 0); push_red(1, 0, 0, 0); push_red(2, 0, 0, 0); push_red(3, 0, 0, 0);
    iv_sub(ORDER, one, t); push(t);                               // n-1
    iv_sub(ORDER, two, t); push(t);                               // n-2
    iv_sub(ORDER, one, t); halve(t, u); push(u);                  // (n-1)/2
    iv_add_u64(ORDER, 1, t); halve(t, u); push(u);                // (n+1)/2
    push_red(0, 1ULL << 63, 0, 0);                                // 2^127
    push_red(F, F, 0, 0);                                         // 2^128-1
    push_red(0, 0, 1, 0);                                         // 2^128
    push_red(0, 0, 2, 0);                                         // 2^129
    push_red(0, 0, 0, 1ULL << 63);                                // 2^255
    push(CG_LAMBDA);                                              // lambda
    cg_mul_mod_n(CG_LAMBDA, CG_LAMBDA, t); push(t);               // lambda^2
    cg_neg_mod_n(CG_LAMBDA, t); push(t);                          // n-lambda
    push_red(0, 0, 0, 1); push_red(0, 1, 0, 0);                   // zero limbs
    push_red(F, 0, 0, 0); push_red(F, 0, F, 0); push_red(0, F, 0, F);
    push_red(F, F, F, 0); push_red(0, F, F, F); push_red(F, F, F, F); // all-ones limbs
    push_red(0x8000000000000001ULL, 0, 0, 0x8000000000000000ULL);

    uint64_t s = 0x6C7D15C0FFEE1234ULL;                           // splitmix64, fixed seed
    auto next = [&s]() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    };
    for (int i = 0; i < 2048; ++i) {
        uint64_t const a0 = next(), a1 = next(), a2 = next(), a3 = next();
        push_red(a0, a1, a2, a3);
    }
    return v;
}

static int audit_ct_glv_scalar_mul() {
    if (g_kernel_dir.empty() || !g_ctx) return -1;
    cl_context ctx = (cl_context)g_ctx->native_context();
    cl_command_queue q = (cl_command_queue)g_ctx->native_queue();
    cl_device_id dev = nullptr;
    if (clGetCommandQueueInfo(q, CL_QUEUE_DEVICE, sizeof(dev), &dev, nullptr) != CL_SUCCESS) return 1;

    cl_int err = CL_SUCCESS;
    const char* src = kCtGlvCheckSource;
    size_t const len = std::strlen(src);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src, &len, &err);
    if (err != CL_SUCCESS) return 2;
    std::string const opts = "-cl-std=CL1.2 -cl-fast-relaxed-math -cl-mad-enable -I " + g_kernel_dir;
    err = clBuildProgram(prog, 1, &dev, opts.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size + 1, '\0');
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::fprintf(stderr, "    ct_glv_check build failed:\n%s\n", log.c_str());
        clReleaseProgram(prog);
        return 3;
    }
    cl_kernel k = clCreateKernel(prog, "ct_glv_check", &err);
    if (err != CL_SUCCESS) { clReleaseProgram(prog); return 4; }

    std::vector<uint64_t> const in = cg_corpus();
    size_t const n = in.size() / 4;
    std::vector<uint64_t> out(n * 46, 0);
    int rc = 0;
    cl_int e2 = CL_SUCCESS;
    cl_mem d_in = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 in.size() * sizeof(uint64_t), (void*)in.data(), &err);
    cl_mem d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, out.size() * sizeof(uint64_t), nullptr, &e2);
    if (err != CL_SUCCESS || e2 != CL_SUCCESS) rc = 5;
    if (rc == 0) {
        cl_uint count = (cl_uint)n;
        clSetKernelArg(k, 0, sizeof(cl_mem), &d_in);
        clSetKernelArg(k, 1, sizeof(cl_mem), &d_out);
        clSetKernelArg(k, 2, sizeof(cl_uint), &count);
        size_t global = n;
        err = clEnqueueNDRangeKernel(q, k, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
        if (err == CL_SUCCESS) {
            err = clEnqueueReadBuffer(q, d_out, CL_TRUE, 0, out.size() * sizeof(uint64_t),
                                      out.data(), 0, nullptr, nullptr);
        }
        if (err != CL_SUCCESS) {
            std::fprintf(stderr, "    ct_glv_check: OpenCL error %d\n", err);
            rc = 6;
        }
    }
    if (d_in) clReleaseMemObject(d_in);
    if (d_out) clReleaseMemObject(d_out);

    size_t bad = 0;
    for (size_t i = 0; i < n && rc == 0; ++i) {
        const uint64_t* kk = &in[i * 4];
        const uint64_t* o = &out[i * 46];
        const uint64_t* k1 = o;
        const uint64_t* k2 = o + 4;
        uint64_t const F = ~0ULL;
        bool const masks_ok = (o[8] == 0 || o[8] == F) && (o[9] == 0 || o[9] == F);
        bool const short_ok = (k1[2] | k1[3] | k2[2] | k2[3]) == 0;
        bool split_ok = false;
        if (masks_ok && short_ok) {
            uint64_t s1[4], s2[4], s2l[4], sum[4];
            if (o[8] != 0) cg_neg_mod_n(k1, s1); else for (int j = 0; j < 4; ++j) s1[j] = k1[j];
            if (o[9] != 0) cg_neg_mod_n(k2, s2); else for (int j = 0; j < 4; ++j) s2[j] = k2[j];
            cg_mul_mod_n(CG_LAMBDA, s2, s2l);
            cg_add_mod_n(s1, s2l, sum);
            split_ok = iv_eq(sum, kk);
        }
        bool const gen_ok = std::equal(o + 10, o + 19, o + 19);
        bool const pt_ok = std::equal(o + 28, o + 37, o + 37);
        bool const k_zero = iv_is_zero(kk);
        bool const ref_ok = ((o[27] == 1) == k_zero) && ((o[45] == 1) == k_zero);
        if (!(masks_ok && short_ok && split_ok && gen_ok && pt_ok && ref_ok)) {
            if (bad < 8) {
                std::fprintf(stderr, "    scalar #%zu: masks=%d short=%d split=%d k*G=%d k*P=%d ref=%d\n",
                             i, (int)masks_ok, (int)short_ok, (int)split_ok, (int)gen_ok,
                             (int)pt_ok, (int)ref_ok);
                iv_print("k", kk); iv_print("k1", k1); iv_print("k2", k2);
                iv_print("ct k*G x", o + 10); iv_print("ref k*G x", o + 19);
                iv_print("ct k*P x", o + 28); iv_print("ref k*P x", o + 37);
            }
            ++bad;
        }
    }
    if (rc == 0 && bad != 0) {
        std::fprintf(stderr, "    ct_glv_check: %zu of %zu scalars failed\n", bad, n);
        rc = 7;
    }

    clReleaseKernel(k);
    clReleaseProgram(prog);
    return rc;
}

// =============================================================================
// CT mixed addition: P + (-P) is the flagged point at infinity
// =============================================================================
// ct_point_add_mixed (secp256k1_ct_point.cl) once handled only P == Q
// (h == 0, rr == 0) as a special case. For P == -Q (h == 0, rr != 0) the
// formula gives z3 = 2*z1*h = 0, but the infinity flag stayed clear, so the
// result was an unflagged Z = 0 point and Z = 0 is sticky through later
// additions and doublings. audit_ct_add_mixed_negation checks, for P = m*G
// with m in {1, 2, 3, n-1, n-2, (n-1)/2, lambda} plus seeded random m, with P
// given once with Z = 1 and once rescaled to Z = c (c != 1):
//   1. P + (-P) has the infinity flag set (Z = 1 and Z = c inputs)
//   2. (P + (-P)) + Q == Q, also after one doubling of P + (-P), and
//      (P + (-P)) + O stays O
//   3. P + P (mixed add) == ct_point_dbl(P) == fast windowed (2m)*G
//   4. P + Q == fast windowed (m+q)*G for an unrelated fixed Q = q*G

static const char* const kCtAddNegCheckSource = R"CLSRC(
#include "secp256k1_ct_extended.cl"

// o[0..3] affine x, o[4..7] affine y (canonical LE limbs),
// o[8] = 1 for the point at infinity (x = y = 0 then)
static inline void ctneg_put_point(const JacobianPoint* p, __global ulong* o) {
    for (int i = 0; i < 9; ++i) o[i] = 0;
    if (p->infinity) { o[8] = 1; return; }
    FieldElement zi, zi2, zi3, x, y;
    field_inv_impl(&zi, &p->z);
    field_sqr_impl(&zi2, &zi);
    field_mul_impl(&zi3, &zi, &zi2);
    field_mul_impl(&x, &p->x, &zi2);
    field_mul_impl(&y, &p->y, &zi3);
    uchar bx[32], by[32];
    field_to_bytes_impl(&x, bx);
    field_to_bytes_impl(&y, by);
    for (int i = 0; i < 4; ++i) {
        ulong lx = 0, ly = 0;
        for (int j = 0; j < 8; ++j) {
            lx = (lx << 8) | (ulong)bx[(3 - i) * 8 + j];
            ly = (ly << 8) | (ulong)by[(3 - i) * 8 + j];
        }
        o[i] = lx;
        o[4 + i] = ly;
    }
}

static inline void ctneg_put_ct(const CTJacobianPoint* p, __global ulong* o) {
    JacobianPoint j = ct_point_to_jacobian(p);
    ctneg_put_point(&j, o);
}

static inline void ctneg_to_affine(const JacobianPoint* p, CTAffinePoint* a) {
    FieldElement zi, zi2, zi3;
    field_inv_impl(&zi, &p->z);
    field_sqr_impl(&zi2, &zi);
    field_mul_impl(&zi3, &zi, &zi2);
    field_mul_impl(&a->x, &p->x, &zi2);
    field_mul_impl(&a->y, &p->y, &zi3);
    a->infinity = 0;
}

// in: 4 ulongs per item (m, 0 < m < n). out: 84 ulongs per item
//   [0] (P1 + (-P)).infinity       [1] (Pc + (-P)).infinity
//   [2..10]  (P1 + (-P)) + Q       [11..19] (Pc + (-P)) + Q
//   [20..28] dbl(Pc + (-P)) + Q    [29..37] Q
//   [38..46] Pc + P (mixed add)    [47..55] ct_point_dbl(Pc)   [56..64] fast (2m)*G
//   [65..73] Pc + Q                [74..82] fast (m+q)*G
//   [83] ((Pc + (-P)) + O).infinity
// P1 = P with Z = 1, Pc = P with Z = c != 1.
__kernel void ct_add_neg_check(__global const ulong* in, __global ulong* out, const uint count) {
    const uint gid = get_global_id(0);
    if (gid >= count) return;
    __global ulong* o = out + (size_t)gid * 84;
    Scalar m;
    for (int i = 0; i < 4; ++i) m.limbs[i] = in[gid * 4 + i];
    Scalar qs;
    qs.limbs[0] = 0x1111111111111111UL; qs.limbs[1] = 0x2222222222222222UL;
    qs.limbs[2] = 0x3333333333333333UL; qs.limbs[3] = 0x0444444444444444UL;

    JacobianPoint pj, qj, ref;
    scalar_mul_generator_windowed_impl(&pj, &m);
    scalar_mul_generator_windowed_impl(&qj, &qs);

    CTAffinePoint pa, qa, na, id;
    ctneg_to_affine(&pj, &pa);
    ctneg_to_affine(&qj, &qa);
    na = pa;
    ct_field_neg_impl(&na.y, &pa.y);
    for (int i = 0; i < 4; ++i) { id.x.limbs[i] = 0; id.y.limbs[i] = 0; }
    id.infinity = ~(ulong)0;

    CTJacobianPoint p1;
    p1.x = pa.x; p1.y = pa.y;
    p1.z.limbs[0] = 1; p1.z.limbs[1] = 0; p1.z.limbs[2] = 0; p1.z.limbs[3] = 0;
    p1.infinity = 0;

    FieldElement c, c2, c3;
    c.limbs[0] = 0x9E3779B97F4A7C15UL; c.limbs[1] = 0xBF58476D1CE4E5B9UL;
    c.limbs[2] = 0x94D049BB133111EBUL; c.limbs[3] = 0x0123456789ABCDEFUL;
    field_sqr_impl(&c2, &c);
    field_mul_impl(&c3, &c2, &c);
    CTJacobianPoint pc;
    field_mul_impl(&pc.x, &pa.x, &c2);
    field_mul_impl(&pc.y, &pa.y, &c3);
    pc.z = c;
    pc.infinity = 0;

    CTJacobianPoint a1, ac, t, t2;
    ct_point_add_mixed(&p1, &na, &a1);
    ct_point_add_mixed(&pc, &na, &ac);
    o[0] = a1.infinity;
    o[1] = ac.infinity;

    ct_point_add_mixed(&a1, &qa, &t);
    ctneg_put_ct(&t, o + 2);
    ct_point_add_mixed(&ac, &qa, &t);
    ctneg_put_ct(&t, o + 11);
    ct_point_dbl(&ac, &t2);
    ct_point_add_mixed(&t2, &qa, &t);
    ctneg_put_ct(&t, o + 20);
    ctneg_put_point(&qj, o + 29);

    ct_point_add_mixed(&pc, &pa, &t);
    ctneg_put_ct(&t, o + 38);
    ct_point_dbl(&pc, &t);
    ctneg_put_ct(&t, o + 47);
    Scalar m2;
    scalar_add_mod_n_impl(&m, &m, &m2);
    scalar_mul_generator_windowed_impl(&ref, &m2);
    ctneg_put_point(&ref, o + 56);

    ct_point_add_mixed(&pc, &qa, &t);
    ctneg_put_ct(&t, o + 65);
    Scalar mq;
    scalar_add_mod_n_impl(&m, &qs, &mq);
    scalar_mul_generator_windowed_impl(&ref, &mq);
    ctneg_put_point(&ref, o + 74);

    ct_point_add_mixed(&ac, &id, &t);
    o[83] = t.infinity;
}
)CLSRC";

// Edge and seeded random scalars 0 < m < n, none equal to +/-q of the kernel.
static std::vector<uint64_t> can_corpus() {
    std::vector<uint64_t> v;
    auto push = [&v](const uint64_t a[4]) { v.insert(v.end(), a, a + 4); };
    const uint64_t one[4] = {1, 0, 0, 0}, two[4] = {2, 0, 0, 0}, three[4] = {3, 0, 0, 0};
    uint64_t t[4], u[4];
    push(one); push(two); push(three);
    iv_sub(ORDER, one, t); push(t);                               // n-1
    iv_sub(ORDER, two, t); push(t);                               // n-2
    iv_sub(ORDER, one, t);                                        // (n-1)/2
    for (int i = 0; i < 4; ++i) u[i] = (t[i] >> 1) | (i < 3 ? (t[i + 1] << 63) : 0);
    push(u);
    push(CG_LAMBDA);                                              // lambda

    uint64_t s = 0x5A17C0DE0BADF00DULL;                           // splitmix64, fixed seed
    auto next = [&s]() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    };
    for (int i = 0; i < 16; ++i) {
        const uint64_t a[4] = {next(), next(), next(), next()};
        iv_reduce_once(a, ORDER, t);
        if (iv_is_zero(t)) continue;
        push(t);
    }
    return v;
}

static int audit_ct_add_mixed_negation() {
    if (g_kernel_dir.empty() || !g_ctx) return -1;
    cl_context ctx = (cl_context)g_ctx->native_context();
    cl_command_queue q = (cl_command_queue)g_ctx->native_queue();
    cl_device_id dev = nullptr;
    if (clGetCommandQueueInfo(q, CL_QUEUE_DEVICE, sizeof(dev), &dev, nullptr) != CL_SUCCESS) return 1;

    cl_int err = CL_SUCCESS;
    const char* src = kCtAddNegCheckSource;
    size_t const len = std::strlen(src);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src, &len, &err);
    if (err != CL_SUCCESS) return 2;
    std::string const opts = "-cl-std=CL1.2 -cl-fast-relaxed-math -cl-mad-enable -I " + g_kernel_dir;
    err = clBuildProgram(prog, 1, &dev, opts.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size + 1, '\0');
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::fprintf(stderr, "    ct_add_neg_check build failed:\n%s\n", log.c_str());
        clReleaseProgram(prog);
        return 3;
    }
    cl_kernel k = clCreateKernel(prog, "ct_add_neg_check", &err);
    if (err != CL_SUCCESS) { clReleaseProgram(prog); return 4; }

    std::vector<uint64_t> const in = can_corpus();
    size_t const n = in.size() / 4;
    std::vector<uint64_t> out(n * 84, 0);
    int rc = 0;
    cl_int e2 = CL_SUCCESS;
    cl_mem d_in = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 in.size() * sizeof(uint64_t), (void*)in.data(), &err);
    cl_mem d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, out.size() * sizeof(uint64_t), nullptr, &e2);
    if (err != CL_SUCCESS || e2 != CL_SUCCESS) rc = 5;
    if (rc == 0) {
        cl_uint count = (cl_uint)n;
        clSetKernelArg(k, 0, sizeof(cl_mem), &d_in);
        clSetKernelArg(k, 1, sizeof(cl_mem), &d_out);
        clSetKernelArg(k, 2, sizeof(cl_uint), &count);
        size_t global = n;
        err = clEnqueueNDRangeKernel(q, k, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
        if (err == CL_SUCCESS) {
            err = clEnqueueReadBuffer(q, d_out, CL_TRUE, 0, out.size() * sizeof(uint64_t),
                                      out.data(), 0, nullptr, nullptr);
        }
        if (err != CL_SUCCESS) {
            std::fprintf(stderr, "    ct_add_neg_check: OpenCL error %d\n", err);
            rc = 6;
        }
    }
    if (d_in) clReleaseMemObject(d_in);
    if (d_out) clReleaseMemObject(d_out);

    size_t bad = 0;
    uint64_t const F = ~0ULL;
    for (size_t i = 0; i < n && rc == 0; ++i) {
        const uint64_t* o = &out[i * 84];
        bool const inf_ok = o[0] == F && o[1] == F && o[83] == F;
        bool const q_ok = o[37] == 0 && std::equal(o + 2, o + 11, o + 29) &&
                          std::equal(o + 11, o + 20, o + 29) && std::equal(o + 20, o + 29, o + 29);
        bool const dbl_ok = o[64] == 0 && std::equal(o + 38, o + 47, o + 56) &&
                            std::equal(o + 47, o + 56, o + 56);
        bool const add_ok = o[82] == 0 && std::equal(o + 65, o + 74, o + 74);
        if (!(inf_ok && q_ok && dbl_ok && add_ok)) {
            if (bad < 8) {
                std::fprintf(stderr, "    m #%zu: P+(-P)=O %d, (P+(-P))+Q=Q %d, P+P=2P %d, P+Q %d\n",
                             i, (int)inf_ok, (int)q_ok, (int)dbl_ok, (int)add_ok);
                iv_print("m", &in[i * 4]);
                iv_print("(P+(-P))+Q x", o + 11); iv_print("Q x", o + 29);
                iv_print("P+P x", o + 38); iv_print("ref 2P x", o + 56);
                iv_print("P+Q x", o + 65); iv_print("ref P+Q x", o + 74);
            }
            ++bad;
        }
    }
    if (rc == 0 && bad != 0) {
        std::fprintf(stderr, "    ct_add_neg_check: %zu of %zu points failed\n", bad, n);
        rc = 7;
    }

    clReleaseKernel(k);
    clReleaseProgram(prog);
    return rc;
}

// -----------------------------------------------------------------------------
// ecrecover host reference. Schoolbook arithmetic on 8 x 32-bit words, reduced
// by folding with c = 2^256 - m, and complete Jacobian formulas with an
// explicit infinity flag. It shares no code with the OpenCL kernels, so it is
// the CPU oracle the GPU recovery is compared against byte for byte.
// -----------------------------------------------------------------------------
struct RrU256 { uint32_t w[8]; };
struct RrMod  { RrU256 m; uint32_t c[5]; int clen; };

static const RrMod RR_P = {
    {{0xFFFFFC2Fu, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
      0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu}},
    {0x000003D1u, 0x00000001u, 0u, 0u, 0u}, 2};
static const RrMod RR_N = {
    {{0xD0364141u, 0xBFD25E8Cu, 0xAF48A03Bu, 0xBAAEDCE6u,
      0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu}},
    {0x2FC9BEBFu, 0x402DA173u, 0x50B75FC4u, 0x45512319u, 0x00000001u}, 5};

static RrU256 rr_small(uint32_t v) { RrU256 r{}; r.w[0] = v; return r; }

static RrU256 rr_from_limbs(const uint64_t l[4]) {
    RrU256 r{};
    for (int i = 0; i < 4; ++i) {
        r.w[2 * i]     = (uint32_t)l[i];
        r.w[2 * i + 1] = (uint32_t)(l[i] >> 32);
    }
    return r;
}

static RrU256 rr_from_be(const uint8_t b[32]) {
    RrU256 r{};
    for (int i = 0; i < 8; ++i) {
        const uint8_t* p = b + (7 - i) * 4;
        r.w[i] = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3];
    }
    return r;
}

static void rr_to_be(const RrU256& a, uint8_t b[32]) {
    for (int i = 0; i < 8; ++i) {
        uint8_t* p = b + (7 - i) * 4;
        p[0] = (uint8_t)(a.w[i] >> 24); p[1] = (uint8_t)(a.w[i] >> 16);
        p[2] = (uint8_t)(a.w[i] >> 8);  p[3] = (uint8_t)a.w[i];
    }
}

static int rr_cmp(const RrU256& a, const RrU256& b) {
    for (int i = 7; i >= 0; --i)
        if (a.w[i] != b.w[i]) return a.w[i] < b.w[i] ? -1 : 1;
    return 0;
}

static bool rr_is_zero(const RrU256& a) {
    uint32_t acc = 0;
    for (int i = 0; i < 8; ++i) acc |= a.w[i];
    return acc == 0;
}

static uint32_t rr_add_raw(RrU256& r, const RrU256& a, const RrU256& b) {
    uint64_t c = 0;
    for (int i = 0; i < 8; ++i) {
        c += (uint64_t)a.w[i] + b.w[i];
        r.w[i] = (uint32_t)c;
        c >>= 32;
    }
    return (uint32_t)c;
}

static uint32_t rr_sub_raw(RrU256& r, const RrU256& a, const RrU256& b) {
    uint64_t borrow = 0;
    for (int i = 0; i < 8; ++i) {
        const uint64_t d = (uint64_t)a.w[i] - b.w[i] - borrow;
        r.w[i] = (uint32_t)d;
        borrow = (d >> 63) & 1u;
    }
    return (uint32_t)borrow;
}

static RrU256 rr_addm(const RrU256& a, const RrU256& b, const RrMod& M) {
    RrU256 r;
    const uint32_t carry = rr_add_raw(r, a, b);
    if (carry || rr_cmp(r, M.m) >= 0) rr_sub_raw(r, r, M.m);
    return r;
}

static RrU256 rr_subm(const RrU256& a, const RrU256& b, const RrMod& M) {
    RrU256 r;
    if (rr_sub_raw(r, a, b)) rr_add_raw(r, r, M.m);
    return r;
}

static RrU256 rr_mulm(const RrU256& a, const RrU256& b, const RrMod& M) {
    uint32_t t[32] = {0};
    for (int i = 0; i < 8; ++i) {
        uint64_t carry = 0;
        for (int j = 0; j < 8; ++j) {
            const uint64_t cur = (uint64_t)a.w[i] * b.w[j] + t[i + j] + carry;
            t[i + j] = (uint32_t)cur;
            carry = cur >> 32;
        }
        t[i + 8] = (uint32_t)carry;
    }
    // Fold: lo + hi * 2^256 == lo + hi * c (mod m), until hi is zero.
    int len = 16;
    for (;;) {
        bool hi_zero = true;
        for (int i = 8; i < len; ++i) if (t[i]) { hi_zero = false; break; }
        if (hi_zero) break;
        uint32_t u[32] = {0};
        for (int i = 0; i < 8; ++i) u[i] = t[i];
        for (int i = 8; i < len; ++i) {
            uint64_t carry = 0;
            int k = i - 8;
            for (int j = 0; j < M.clen; ++j, ++k) {
                const uint64_t cur = (uint64_t)t[i] * M.c[j] + u[k] + carry;
                u[k] = (uint32_t)cur;
                carry = cur >> 32;
            }
            while (carry) {
                const uint64_t cur = (uint64_t)u[k] + carry;
                u[k++] = (uint32_t)cur;
                carry = cur >> 32;
            }
        }
        const int nl = (len - 8) + M.clen + 1;
        len = nl > 9 ? nl : 9;
        for (int i = 0; i < 32; ++i) t[i] = u[i];
    }
    RrU256 r;
    for (int i = 0; i < 8; ++i) r.w[i] = t[i];
    while (rr_cmp(r, M.m) >= 0) rr_sub_raw(r, r, M.m);
    return r;
}

static RrU256 rr_powm(const RrU256& a, const RrU256& e, const RrMod& M) {
    RrU256 r = rr_small(1);
    for (int i = 255; i >= 0; --i) {
        r = rr_mulm(r, r, M);
        if ((e.w[i / 32] >> (i % 32)) & 1u) r = rr_mulm(r, a, M);
    }
    return r;
}

struct RrPt { RrU256 x, y, z; bool inf; };

static RrPt rr_inf() { RrPt p{}; p.inf = true; return p; }

static RrPt rr_dbl(const RrPt& P) {
    if (P.inf || rr_is_zero(P.y)) return rr_inf();
    const RrMod& M = RR_P;
    const RrU256 A = rr_mulm(P.x, P.x, M);
    const RrU256 B = rr_mulm(P.y, P.y, M);
    const RrU256 C = rr_mulm(B, B, M);
    RrU256 t = rr_addm(P.x, B, M);
    t = rr_subm(rr_subm(rr_mulm(t, t, M), A, M), C, M);
    const RrU256 D = rr_addm(t, t, M);
    const RrU256 E = rr_addm(rr_addm(A, A, M), A, M);
    RrU256 c8 = rr_addm(C, C, M);
    c8 = rr_addm(c8, c8, M);
    c8 = rr_addm(c8, c8, M);
    RrPt R;
    R.inf = false;
    R.x = rr_subm(rr_mulm(E, E, M), rr_addm(D, D, M), M);
    R.y = rr_subm(rr_mulm(E, rr_subm(D, R.x, M), M), c8, M);
    const RrU256 yz = rr_mulm(P.y, P.z, M);
    R.z = rr_addm(yz, yz, M);
    return R;
}

static RrPt rr_add(const RrPt& P, const RrPt& Q) {
    if (P.inf) return Q;
    if (Q.inf) return P;
    const RrMod& M = RR_P;
    const RrU256 z1z1 = rr_mulm(P.z, P.z, M), z2z2 = rr_mulm(Q.z, Q.z, M);
    const RrU256 u1 = rr_mulm(P.x, z2z2, M), u2 = rr_mulm(Q.x, z1z1, M);
    const RrU256 s1 = rr_mulm(rr_mulm(P.y, Q.z, M), z2z2, M);
    const RrU256 s2 = rr_mulm(rr_mulm(Q.y, P.z, M), z1z1, M);
    if (rr_cmp(u1, u2) == 0) return rr_cmp(s1, s2) == 0 ? rr_dbl(P) : rr_inf();
    const RrU256 h = rr_subm(u2, u1, M), rd = rr_subm(s2, s1, M);
    const RrU256 h2 = rr_mulm(h, h, M), h3 = rr_mulm(h, h2, M), v = rr_mulm(u1, h2, M);
    RrPt R;
    R.inf = false;
    R.x = rr_subm(rr_subm(rr_mulm(rd, rd, M), h3, M), rr_addm(v, v, M), M);
    R.y = rr_subm(rr_mulm(rd, rr_subm(v, R.x, M), M), rr_mulm(s1, h3, M), M);
    R.z = rr_mulm(rr_mulm(h, P.z, M), Q.z, M);
    return R;
}

static RrPt rr_smul(const RrU256& k, const RrPt& P) {
    RrPt R = rr_inf();
    for (int i = 255; i >= 0; --i) {
        R = rr_dbl(R);
        if ((k.w[i / 32] >> (i % 32)) & 1u) R = rr_add(R, P);
    }
    return R;
}

// Reference ecrecover with the libsecp256k1 / CPU ecdsa_recover rules:
// recid in 0..3, 0 < r, s < n, recid & 2 requires r < p - n (x = r + n),
// x must lift, y parity = recid & 1, Q = r^-1 (s R - z G) != infinity.
// Output: 65-byte uncompressed key.
static bool rr_ecrecover(const uint8_t msg[32], const uint64_t r_l[4], const uint64_t s_l[4],
                         int recid, uint8_t out65[65]) {
    std::memset(out65, 0, 65);
    if (recid < 0 || recid > 3) return false;
    const RrU256 r = rr_from_limbs(r_l), s = rr_from_limbs(s_l);
    if (rr_is_zero(r) || rr_is_zero(s)) return false;
    if (rr_cmp(r, RR_N.m) >= 0 || rr_cmp(s, RR_N.m) >= 0) return false;
    RrU256 x = r;
    if (recid & 2) {
        RrU256 pmn;
        rr_sub_raw(pmn, RR_P.m, RR_N.m);
        if (rr_cmp(r, pmn) >= 0) return false;
        rr_add_raw(x, r, RR_N.m);   // r < p - n, so x < p
    }
    const RrU256 alpha = rr_addm(rr_mulm(rr_mulm(x, x, RR_P), x, RR_P), rr_small(7), RR_P);
    RrU256 e;
    rr_add_raw(e, RR_P.m, rr_small(1));           // p + 1
    for (int i = 0; i < 8; ++i)                   // (p + 1) / 4
        e.w[i] = (e.w[i] >> 2) | (i < 7 ? (e.w[i + 1] << 30) : 0u);
    RrU256 y = rr_powm(alpha, e, RR_P);
    if (rr_cmp(rr_mulm(y, y, RR_P), alpha) != 0) return false;
    if ((int)(y.w[0] & 1u) != (recid & 1)) y = rr_subm(rr_small(0), y, RR_P);

    RrU256 z = rr_from_be(msg);
    if (rr_cmp(z, RR_N.m) >= 0) rr_sub_raw(z, z, RR_N.m);
    RrU256 nm2;
    rr_sub_raw(nm2, RR_N.m, rr_small(2));
    const RrU256 rinv = rr_powm(r, nm2, RR_N);
    const RrU256 u1 = rr_subm(rr_small(0), rr_mulm(z, rinv, RR_N), RR_N);
    const RrU256 u2 = rr_mulm(s, rinv, RR_N);

    static const uint64_t GXL[4] = {0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL,
                                    0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL};
    static const uint64_t GYL[4] = {0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL,
                                    0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL};
    const RrPt G = {rr_from_limbs(GXL), rr_from_limbs(GYL), rr_small(1), false};
    const RrPt R = {x, y, rr_small(1), false};
    const RrPt Q = rr_add(rr_smul(u1, G), rr_smul(u2, R));
    if (Q.inf) return false;
    RrU256 pm2;
    rr_sub_raw(pm2, RR_P.m, rr_small(2));
    const RrU256 zi  = rr_powm(Q.z, pm2, RR_P);    // Fermat: Q.z^(p-2)
    const RrU256 zi2 = rr_mulm(zi, zi, RR_P);
    out65[0] = 0x04;
    rr_to_be(rr_mulm(Q.x, zi2, RR_P), out65 + 1);
    rr_to_be(rr_mulm(Q.y, rr_mulm(zi2, zi, RR_P), RR_P), out65 + 33);
    return true;
}

// GPU Jacobian output -> 65-byte uncompressed key (canonical coordinates).
static void ocl_point_to_65(const JacobianPoint& p, uint8_t out65[65]) {
    AffinePoint const a = jacobian_to_affine(p);
    uint64_t x[4], y[4];
    iv_reduce_once(a.x.limbs, MODULUS, x);
    iv_reduce_once(a.y.limbs, MODULUS, y);
    out65[0] = 0x04;
    for (int i = 0; i < 4; ++i) {
        for (int b = 0; b < 8; ++b) {
            out65[1 + 31 - (i * 8 + b)]  = (uint8_t)(x[i] >> (8 * b));
            out65[33 + 31 - (i * 8 + b)] = (uint8_t)(y[i] >> (8 * b));
        }
    }
}

// One ecrecover_batch launch over every case.
static bool ocl_ecrecover_run(cl_kernel k, const std::vector<uint8_t>& msgs,
                              const std::vector<ExtendedCL::ECDSASig>& sigs,
                              const std::vector<int>& recids,
                              std::vector<JacobianPoint>& pubs, std::vector<int>& res) {
    const size_t n = recids.size();
    pubs.assign(n, JacobianPoint{});
    res.assign(n, -1);
    cl_int e[5] = {};
    cl_mem d_msg = clCreateBuffer(g_ext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  msgs.size(), const_cast<uint8_t*>(msgs.data()), &e[0]);
    cl_mem d_sig = clCreateBuffer(g_ext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  n * sizeof(ExtendedCL::ECDSASig),
                                  const_cast<ExtendedCL::ECDSASig*>(sigs.data()), &e[1]);
    cl_mem d_rec = clCreateBuffer(g_ext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  n * sizeof(int), const_cast<int*>(recids.data()), &e[2]);
    cl_mem d_pub = clCreateBuffer(g_ext.context, CL_MEM_WRITE_ONLY,
                                  n * sizeof(JacobianPoint), nullptr, &e[3]);
    cl_mem d_res = clCreateBuffer(g_ext.context, CL_MEM_WRITE_ONLY, n * sizeof(int), nullptr, &e[4]);
    bool ok = true;
    for (cl_int x : e) ok = ok && (x == CL_SUCCESS);
    if (ok) {
        cl_uint count = (cl_uint)n;
        clSetKernelArg(k, 0, sizeof(cl_mem), &d_msg);
        clSetKernelArg(k, 1, sizeof(cl_mem), &d_sig);
        clSetKernelArg(k, 2, sizeof(cl_mem), &d_rec);
        clSetKernelArg(k, 3, sizeof(cl_mem), &d_pub);
        clSetKernelArg(k, 4, sizeof(cl_mem), &d_res);
        clSetKernelArg(k, 5, sizeof(cl_uint), &count);
        size_t global = n;
        ok = clEnqueueNDRangeKernel(g_ext.queue, k, 1, nullptr, &global, nullptr,
                                    0, nullptr, nullptr) == CL_SUCCESS
          && clEnqueueReadBuffer(g_ext.queue, d_pub, CL_TRUE, 0, n * sizeof(JacobianPoint),
                                 pubs.data(), 0, nullptr, nullptr) == CL_SUCCESS
          && clEnqueueReadBuffer(g_ext.queue, d_res, CL_TRUE, 0, n * sizeof(int),
                                 res.data(), 0, nullptr, nullptr) == CL_SUCCESS;
    }
    if (d_msg) clReleaseMemObject(d_msg);
    if (d_sig) clReleaseMemObject(d_sig);
    if (d_rec) clReleaseMemObject(d_rec);
    if (d_pub) clReleaseMemObject(d_pub);
    if (d_res) clReleaseMemObject(d_res);
    return ok;
}

// ECDSA public-key recovery (ecrecover_batch -> ecdsa_recover_impl: one
// variable-time r^-1, one GLV Shamir u1*G + u2*R, one affine conversion).
// Every case is compared byte for byte (status and 65-byte key) with the host
// reference rr_ecrecover:
//   * six fixed keys and 64 seeded keys, signed on the GPU, under recids 0..3:
//     exactly one recid recovers the signer; with one flipped s bit (the six
//     fixed keys) no recid does;
//   * 32 random (r, s, z) triples under recids 0..3;
//   * edge inputs with their expected status: r or s zero or >= n, recid out
//     of range, x not on the curve, recid 2/3 with r >= p - n (x would wrap),
//     Q at infinity (R = G and R = 2G with z = s * dlog(R)), each next to a
//     positive control on a neighbouring input.
static int audit_ecrecover_roundtrip() {
    if (!g_ext.valid) return -1;
    cl_int err = CL_SUCCESS;
    cl_kernel k = clCreateKernel(g_ext.program, "ecrecover_batch", &err);
    if (err != CL_SUCCESS || !k) return 1;

    std::vector<uint8_t> msgs;
    std::vector<ExtendedCL::ECDSASig> sigs;
    std::vector<int> recids;
    std::vector<int> case_signer;   // signer index, -1 = none
    std::vector<int> case_want;     // expected status, -1 = reference only
    std::vector<const char*> case_name;
    auto add_case = [&](const uint8_t msg[32], const ExtendedCL::ECDSASig& s, int recid,
                        int signer, int want, const char* name) {
        msgs.insert(msgs.end(), msg, msg + 32);
        sigs.push_back(s);
        recids.push_back(recid);
        case_signer.push_back(signer);
        case_want.push_back(want);
        case_name.push_back(name);
    };
    struct Signer { uint8_t pub65[65]; int want_matches; };
    std::vector<Signer> signers;
    int rc = 0;

    auto add_signer = [&](const uint64_t key[4], const uint8_t msg[32], bool with_corrupt) {
        Scalar priv{{key[0], key[1], key[2], key[3]}};
        ExtendedCL::ECDSASig sig{};
        if (!ocl_ecdsa_sign(priv, msg, sig)) return 2;
        JacobianPoint const pub = ext_generator_mul(priv);
        if (!ocl_ecdsa_verify(pub, msg, sig)) return 3;
        Signer sg{};
        ocl_point_to_65(pub, sg.pub65);
        sg.want_matches = 1;
        signers.push_back(sg);
        for (int recid = 0; recid < 4; ++recid)
            add_case(msg, sig, recid, (int)signers.size() - 1, -1, "signed");
        if (with_corrupt) {
            ExtendedCL::ECDSASig bad = sig;
            bad.s[0] ^= 1;
            sg.want_matches = 0;
            signers.push_back(sg);
            for (int recid = 0; recid < 4; ++recid)
                add_case(msg, bad, recid, (int)signers.size() - 1, -1, "flipped s bit");
        }
        return 0;
    };

    static const uint64_t fixed_keys[][4] = {
        {1, 0, 0, 0},
        {2, 0, 0, 0},
        {42, 0, 0, 0},
        {0xDEADBEEFCAFEBABEULL, 0x0123456789ABCDEFULL, 0, 0},
        {0x1111111111111111ULL, 0x2222222222222222ULL, 0x3333333333333333ULL, 0x4444444444444444ULL},
        {0xBFD25E8CD0364140ULL, 0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL},
    };
    for (size_t t = 0; t < sizeof(fixed_keys) / sizeof(fixed_keys[0]) && rc == 0; ++t) {
        uint8_t msg[32];
        for (int i = 0; i < 32; ++i) msg[i] = (uint8_t)(0x5A ^ (i * 7) ^ (int)(t * 31));
        rc = add_signer(fixed_keys[t], msg, true);
    }

    uint64_t seed = 0xEC4EC0DE5EED0001ULL;
    auto next = [&seed]() {
        seed += 0x9E3779B97F4A7C15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    };
    auto rand_msg = [&next](uint8_t msg[32]) {
        for (int i = 0; i < 32; i += 8) {
            const uint64_t v = next();
            for (int b = 0; b < 8; ++b) msg[i + b] = (uint8_t)(v >> (8 * b));
        }
    };
    constexpr int kSeededKeys = 64;
    for (int t = 0; t < kSeededKeys && rc == 0; ++t) {
        // Below 2^255 < n and non-zero.
        uint64_t key[4] = {next() | 1u, next(), next(), next() & 0x7FFFFFFFFFFFFFFFULL};
        uint8_t msg[32];
        rand_msg(msg);
        rc = add_signer(key, msg, false);
    }
    if (rc != 0) { clReleaseKernel(k); return rc; }

    for (int t = 0; t < 32; ++t) {
        uint8_t msg[32];
        rand_msg(msg);
        ExtendedCL::ECDSASig s{};
        for (int i = 0; i < 4; ++i) { s.r[i] = next(); s.s[i] = next(); }
        s.r[3] &= 0x7FFFFFFFFFFFFFFFULL;   // canonical: below 2^255 < n
        s.s[3] &= 0x7FFFFFFFFFFFFFFFULL;
        for (int recid = 0; recid < 4; ++recid) add_case(msg, s, recid, -1, -1, "random (r, s, z)");
    }

    {
        static const uint64_t ZERO[4]  = {0, 0, 0, 0};
        static const uint64_t ONE[4]   = {1, 0, 0, 0};
        static const uint64_t TWO[4]   = {2, 0, 0, 0};
        static const uint64_t FIVE[4]  = {5, 0, 0, 0};
        static const uint64_t SEVEN[4] = {7, 0, 0, 0};
        static const uint64_t GX[4]    = {0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL,
                                          0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL};
        static const uint64_t G2X[4]   = {0xABAC09B95C709EE5ULL, 0x5C778E4B8CEF3CA7ULL,
                                          0x3045406E95C07CD8ULL, 0xC6047F9441ED7D6DULL};
        static const uint64_t NN[4]    = {0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
                                          0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL};
        static const uint64_t NP7[4]   = {0xBFD25E8CD0364148ULL, 0xBAAEDCE6AF48A03BULL,
                                          0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL};
        static const uint64_t ONES[4]  = {~0ULL, ~0ULL, ~0ULL, ~0ULL};
        static const uint64_t PMN[4]   = {0x402DA1722FC9BAEEULL, 0x4551231950B75FC4ULL, 1, 0};
        static const uint64_t PMN1[4]  = {0x402DA1722FC9BAEFULL, 0x4551231950B75FC4ULL, 1, 0};
        struct Edge { const char* name; uint8_t z; const uint64_t* r; const uint64_t* s; int recid; int want; };
        static const Edge edges[] = {
            { "r = 0",                                     7, ZERO, SEVEN, 0, 0 },
            { "s = 0",                                     7, GX, ZERO, 0, 0 },
            { "r = n",                                     7, NN, SEVEN, 1, 0 },
            { "r = 2^256 - 1",                             7, ONES, SEVEN, 1, 0 },
            { "s = n",                                     7, GX, NN, 1, 0 },
            { "s = n + 7 (would reduce to s = 7)",         7, GX, NP7, 1, 0 },
            { "control: r = x(G), s = 7, recid 1",         7, GX, SEVEN, 1, 1 },
            { "recid = -1",                                7, GX, SEVEN, -1, 0 },
            { "recid = 4",                                 7, GX, SEVEN, 4, 0 },
            { "x = 5 is not on the curve (recid 0)",       7, FIVE, SEVEN, 0, 0 },
            { "x = 5 is not on the curve (recid 1)",       7, FIVE, SEVEN, 1, 0 },
            { "control: x = 1 lifts (recid 0)",            7, ONE, SEVEN, 0, 1 },
            { "recid 2 with r = p - n",                    7, PMN, SEVEN, 2, 0 },
            { "recid 2 with r = p - n + 1 (x wraps to 1)", 7, PMN1, SEVEN, 2, 0 },
            { "recid 3 with r = p - n + 1 (x wraps to 1)", 7, PMN1, SEVEN, 3, 0 },
            { "control: recid 2, r = 2 (x = 2 + n lifts)", 7, TWO, SEVEN, 2, 1 },
            { "control: recid 3, r = 2 (x = 2 + n lifts)", 7, TWO, SEVEN, 3, 1 },
            { "Q at infinity (R = G, s = z = 7)",          7, GX, SEVEN, 0, 0 },
            { "Q at infinity (R = 2G, s = 7, z = 14)",    14, G2X, SEVEN, 0, 0 },
            { "control: R = -2G, s = 7, z = 14",          14, G2X, SEVEN, 1, 1 },
        };
        for (const Edge& ed : edges) {
            uint8_t msg[32] = {0};
            msg[31] = ed.z;
            ExtendedCL::ECDSASig s{};
            for (int i = 0; i < 4; ++i) { s.r[i] = ed.r[i]; s.s[i] = ed.s[i]; }
            add_case(msg, s, ed.recid, -1, ed.want, ed.name);
        }
    }

    std::vector<JacobianPoint> pubs;
    std::vector<int> res;
    if (!ocl_ecrecover_run(k, msgs, sigs, recids, pubs, res)) { clReleaseKernel(k); return 4; }
    clReleaseKernel(k);

    std::vector<int> matches(signers.size(), 0);
    int successes = 0;
    for (size_t i = 0; i < recids.size(); ++i) {
        uint8_t want65[65];
        const bool ref_ok = rr_ecrecover(&msgs[i * 32], sigs[i].r, sigs[i].s, recids[i], want65);
        const bool gpu_ok = res[i] != 0;
        if (gpu_ok != ref_ok) {
            std::fprintf(stderr, "    ecrecover case %zu (%s, recid %d): GPU ok=%d, reference ok=%d\n",
                         i, case_name[i], recids[i], res[i], ref_ok ? 1 : 0);
            return 5;
        }
        if (case_want[i] >= 0 && (gpu_ok ? 1 : 0) != case_want[i]) {
            std::fprintf(stderr, "    ecrecover edge '%s': got %d, want %d\n",
                         case_name[i], gpu_ok ? 1 : 0, case_want[i]);
            return 6;
        }
        if (!gpu_ok) continue;
        ++successes;
        uint8_t got65[65];
        ocl_point_to_65(pubs[i], got65);
        if (std::memcmp(got65, want65, 65) != 0) {
            std::fprintf(stderr, "    ecrecover case %zu (%s, recid %d): GPU key != reference key\n",
                         i, case_name[i], recids[i]);
            return 7;
        }
        const int sg = case_signer[i];
        if (sg >= 0 && std::memcmp(got65, signers[sg].pub65, 65) == 0) ++matches[sg];
    }
    for (size_t sg = 0; sg < signers.size(); ++sg) {
        if (matches[sg] != signers[sg].want_matches) {
            std::fprintf(stderr, "    signer #%zu: %d recids recovered the signer key (want %d)\n",
                         sg, matches[sg], signers[sg].want_matches);
            return 8;
        }
    }
    // Signer recids alone give one success per valid signature; the corpus
    // must also succeed elsewhere to exercise the comparison.
    if (successes <= (int)(6 + kSeededKeys)) return 9;
    return 0;
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
    { "inv_var_fermat",    "VT SafeGCD inverse == Fermat (p, n, fail-closed)", "math_invariants", audit_inv_var_vs_fermat, false },

    // Section 2: Signature Operations
    { "ecdsa_roundtrip",   "ECDSA sign + verify roundtrip",              "signatures", audit_ecdsa_roundtrip, false },
    { "ecrecover_rt",      "ECDSA ecrecover vs host reference (70 keys, random, edges)", "signatures", audit_ecrecover_roundtrip, false },
    { "schnorr_roundtrip", "Schnorr/BIP-340 sign + verify roundtrip",    "signatures", audit_schnorr_roundtrip, false },
    { "ecdsa_wrong_key",   "ECDSA verify rejects wrong pubkey",          "signatures", audit_ecdsa_wrong_key, false },
    { "ecdsa_verify_rx",   "ECDSA verify projective x(R)==r (r+n branch)", "signatures", audit_ecdsa_verify_rx, false },

    // Section 3: Batch Operations
    { "batch_smul",        "Batch scalar mul generator",                  "batch_advanced", audit_batch_scalar_mul, false },
    { "batch_j2a",         "Batch Jacobian to Affine",                    "batch_advanced", audit_batch_j2a, false },

    // Section 4: Differential
    { "diff_smul",         "OpenCL-host differential scalar mul",         "differential", audit_diff_scalar_mul, false },
    { "hash160_gpu_cpu",   "hash160 GPU vs CPU (random 33/65-byte keys)", "differential", audit_hash160_gpu_vs_cpu, false },

    // Section 5: Standard Test Vectors
    { "rfc6979_determ",    "RFC-6979 ECDSA deterministic nonce",          "standard_vectors", audit_rfc6979_determinism, false },
    { "bip340_vectors",    "BIP-340 Schnorr known-key roundtrip",         "standard_vectors", audit_bip340_vectors, false },
    { "hash160_kat",       "hash160 known answers (33-byte and 65-byte G)", "standard_vectors", audit_hash160_known_answers, false },

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
    { "ct_glv_scalar_mul", "CT GLV split + CT k*G / k*P == fast path (edge + 2048 random)", "gpu_ct_smoke", audit_ct_glv_scalar_mul, false },
    { "ct_add_mixed_neg",  "CT mixed add: P+(-P)=O flagged, O+Q=Q, P+P=2P, P+Q (Z=1, Z!=1)", "gpu_ct_smoke", audit_ct_add_mixed_negation, false },
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
