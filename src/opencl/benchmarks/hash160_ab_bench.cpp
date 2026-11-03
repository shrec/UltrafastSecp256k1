// =============================================================================
// UltrafastSecp256k1 OpenCL -- hash160 A/B harness (NF-2026-00033)
// =============================================================================
// Times the previous loop/table hash160 kernel ("old") against the production
// register-only kernel in src/opencl/kernels/secp256k1_hash160.cl ("new") in a
// single run, on the same device, with the same inputs, and checks that both
// produce identical output for every key.
//
// The old form lives ONLY in this file (OLD_HASH160_SRC below). The production
// kernel is loaded from disk, so "new" is exactly what ships.
//
// Both programs expose the same `hash160_batch(pubkeys, hashes, stride, count)`
// kernel, so the timed work (global read of the key, hash160, global write of
// 20 bytes) is identical apart from the hash itself.
//
// Usage:
//   opencl_hash160_ab_bench [--kernel-dir DIR] [--keys N] [--iters N]
//                           [--reps N] [--platform I] [--device I]
//
// It prints measured numbers for the machine it runs on and nothing else; do
// not copy them into docs, comments or commit messages. Repeat the run (the
// project's 5-run protocol) to judge run-to-run variance.
//
// Exit status: 0 when every key agreed for both lengths, 1 on any mismatch or
// OpenCL error.
// =============================================================================

#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
// Previous hash160 kernel (loop/table form), kept here as the A/B baseline.
// Byte-for-byte the code that production used before NF-2026-00033.
// -----------------------------------------------------------------------------
static const char* OLD_HASH160_SRC = R"CLC(
__constant uint SHA256_K[64] = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U
};

static inline uint hash_rotr32(uint x, uint n) { return (x >> n) | (x << (32 - n)); }
static inline uint hash_rotl32(uint x, uint n) { return (x << n) | (x >> (32 - n)); }

static inline void sha256_oneshot_impl(const uchar* data, uint len, uchar out[32]) {
    uint h0 = 0x6a09e667U;
    uint h1 = 0xbb67ae85U;
    uint h2 = 0x3c6ef372U;
    uint h3 = 0xa54ff53aU;
    uint h4 = 0x510e527fU;
    uint h5 = 0x9b05688cU;
    uint h6 = 0x1f83d9abU;
    uint h7 = 0x5be0cd19U;

    uchar block[128];
    for (int i = 0; i < 128; ++i) block[i] = 0;
    for (uint i = 0; i < len; ++i) block[i] = data[i];
    block[len] = 0x80;

    const ulong bit_len = (ulong)len * 8UL;
    const int total_blocks = ((len + 1 + 8) <= 64) ? 1 : 2;
    const int last = total_blocks * 64 - 8;
    block[last + 0] = (uchar)(bit_len >> 56);
    block[last + 1] = (uchar)(bit_len >> 48);
    block[last + 2] = (uchar)(bit_len >> 40);
    block[last + 3] = (uchar)(bit_len >> 32);
    block[last + 4] = (uchar)(bit_len >> 24);
    block[last + 5] = (uchar)(bit_len >> 16);
    block[last + 6] = (uchar)(bit_len >> 8);
    block[last + 7] = (uchar)(bit_len);

    for (int b = 0; b < total_blocks; ++b) {
        uint w[64];
        const int off = b * 64;
        for (int i = 0; i < 16; ++i) {
            const int j = off + i * 4;
            w[i] = ((uint)block[j] << 24) |
                   ((uint)block[j + 1] << 16) |
                   ((uint)block[j + 2] << 8) |
                   ((uint)block[j + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const uint s0 = hash_rotr32(w[i - 15], 7) ^ hash_rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint s1 = hash_rotr32(w[i - 2], 17) ^ hash_rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        uint a = h0, b0 = h1, c = h2, d = h3;
        uint e = h4, f = h5, g = h6, h = h7;

        for (int i = 0; i < 64; ++i) {
            const uint S1 = hash_rotr32(e, 6) ^ hash_rotr32(e, 11) ^ hash_rotr32(e, 25);
            const uint ch = (e & f) ^ (~e & g);
            const uint temp1 = h + S1 + ch + SHA256_K[i] + w[i];
            const uint S0 = hash_rotr32(a, 2) ^ hash_rotr32(a, 13) ^ hash_rotr32(a, 22);
            const uint maj = (a & b0) ^ (a & c) ^ (b0 & c);
            const uint temp2 = S0 + maj;

            h = g; g = f; f = e; e = d + temp1;
            d = c; c = b0; b0 = a; a = temp1 + temp2;
        }

        h0 += a; h1 += b0; h2 += c; h3 += d;
        h4 += e; h5 += f; h6 += g; h7 += h;
    }

    const uint hh[8] = {h0, h1, h2, h3, h4, h5, h6, h7};
    for (int i = 0; i < 8; ++i) {
        out[i * 4 + 0] = (uchar)(hh[i] >> 24);
        out[i * 4 + 1] = (uchar)(hh[i] >> 16);
        out[i * 4 + 2] = (uchar)(hh[i] >> 8);
        out[i * 4 + 3] = (uchar)(hh[i]);
    }
}

__constant uchar RIPEMD_R[80] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
    3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
    1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
    4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13
};
__constant uchar RIPEMD_R2[80] = {
    5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
    6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
    15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,
    8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
    12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11
};
__constant uchar RIPEMD_S[80] = {
    11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
    7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
    11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
    11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
    9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6
};
__constant uchar RIPEMD_S2[80] = {
    8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
    9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
    9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
    15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
    8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11
};
__constant uint RIPEMD_K[5]  = { 0x00000000U, 0x5A827999U, 0x6ED9EBA1U, 0x8F1BBCDCU, 0xA953FD4EU };
__constant uint RIPEMD_K2[5] = { 0x50A28BE6U, 0x5C4DD124U, 0x6D703EF3U, 0x7A6D76E9U, 0x00000000U };

static inline uint ripemd_f_impl(int j, uint x, uint y, uint z) {
    if (j <= 15) return x ^ y ^ z;
    if (j <= 31) return (x & y) | (~x & z);
    if (j <= 47) return (x | ~y) ^ z;
    if (j <= 63) return (x & z) | (y & ~z);
    return x ^ (y | ~z);
}

static inline void ripemd160_32_impl(const uchar data[32], uchar out[20]) {
    uchar block[64];
    for (int i = 0; i < 64; ++i) block[i] = 0;
    for (int i = 0; i < 32; ++i) block[i] = data[i];
    block[32] = 0x80;
    const ulong bit_len = 32UL * 8UL;
    block[56] = (uchar)(bit_len);
    block[57] = (uchar)(bit_len >> 8);
    block[58] = (uchar)(bit_len >> 16);
    block[59] = (uchar)(bit_len >> 24);
    block[60] = (uchar)(bit_len >> 32);
    block[61] = (uchar)(bit_len >> 40);
    block[62] = (uchar)(bit_len >> 48);
    block[63] = (uchar)(bit_len >> 56);

    uint X[16];
    for (int i = 0; i < 16; ++i) {
        const int j = i * 4;
        X[i] = (uint)block[j] |
               ((uint)block[j + 1] << 8) |
               ((uint)block[j + 2] << 16) |
               ((uint)block[j + 3] << 24);
    }

    uint h0 = 0x67452301U;
    uint h1 = 0xEFCDAB89U;
    uint h2 = 0x98BADCFEU;
    uint h3 = 0x10325476U;
    uint h4 = 0xC3D2E1F0U;

    uint a = h0, b0 = h1, c = h2, d = h3, e = h4;
    uint a2 = h0, b2 = h1, c2 = h2, d2 = h3, e2 = h4;

    for (int j = 0; j < 80; ++j) {
        const uint t = hash_rotl32(a + ripemd_f_impl(j, b0, c, d) +
                                   X[RIPEMD_R[j]] + RIPEMD_K[j / 16],
                                   RIPEMD_S[j]) + e;
        a = e; e = d; d = hash_rotl32(c, 10); c = b0; b0 = t;

        const uint t2 = hash_rotl32(a2 + ripemd_f_impl(79 - j, b2, c2, d2) +
                                    X[RIPEMD_R2[j]] + RIPEMD_K2[j / 16],
                                    RIPEMD_S2[j]) + e2;
        a2 = e2; e2 = d2; d2 = hash_rotl32(c2, 10); c2 = b2; b2 = t2;
    }

    const uint t = h1 + c + d2;
    h1 = h2 + d + e2;
    h2 = h3 + e + a2;
    h3 = h4 + a + b2;
    h4 = h0 + b0 + c2;
    h0 = t;

    const uint hh[5] = {h0, h1, h2, h3, h4};
    for (int i = 0; i < 5; ++i) {
        out[i * 4 + 0] = (uchar)(hh[i]);
        out[i * 4 + 1] = (uchar)(hh[i] >> 8);
        out[i * 4 + 2] = (uchar)(hh[i] >> 16);
        out[i * 4 + 3] = (uchar)(hh[i] >> 24);
    }
}

static inline void hash160_pubkey_impl(const uchar* pubkey, uint pubkey_len, uchar out[20]) {
    uchar sha[32];
    sha256_oneshot_impl(pubkey, pubkey_len, sha);
    ripemd160_32_impl(sha, out);
}

__kernel void hash160_batch(
    __global const uchar* pubkeys,
    __global uchar* hashes,
    const uint stride,
    const uint count
) {
    const uint tid = get_global_id(0);
    if (tid >= count) return;

    uchar pk[65];
    const uint pk_len = (stride <= 65) ? stride : 65;
    for (uint i = 0; i < pk_len; ++i) {
        pk[i] = pubkeys[tid * stride + i];
    }

    uchar h160[20];
    hash160_pubkey_impl(pk, pk_len, h160);

    for (int i = 0; i < 20; ++i) {
        hashes[tid * 20 + i] = h160[i];
    }
}
)CLC";

// -----------------------------------------------------------------------------
// Host helpers
// -----------------------------------------------------------------------------
#define CL_CHECK(expr)                                                         \
    do {                                                                       \
        cl_int err__ = (expr);                                                 \
        if (err__ != CL_SUCCESS) {                                             \
            std::fprintf(stderr, "OpenCL error %d at %s:%d: %s\n", (int)err__, \
                         __FILE__, __LINE__, #expr);                           \
            return 1;                                                          \
        }                                                                      \
    } while (0)

static std::string load_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return {};
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

struct Form {
    const char* name = "";
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
};

static bool build_form(Form& form, cl_context ctx, cl_device_id dev,
                       const std::string& src, const std::string& opts) {
    cl_int err = CL_SUCCESS;
    const char* sp = src.c_str();
    size_t sl = src.size();
    form.program = clCreateProgramWithSource(ctx, 1, &sp, &sl, &err);
    if (err != CL_SUCCESS) { std::fprintf(stderr, "%s: create program failed\n", form.name); return false; }
    err = clBuildProgram(form.program, 1, &dev, opts.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t n = 0;
        clGetProgramBuildInfo(form.program, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
        std::string log(n + 1, '\0');
        clGetProgramBuildInfo(form.program, dev, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
        std::fprintf(stderr, "%s: build failed:\n%s\n", form.name, log.c_str());
        return false;
    }
    form.kernel = clCreateKernel(form.program, "hash160_batch", &err);
    if (err != CL_SUCCESS) { std::fprintf(stderr, "%s: hash160_batch not found\n", form.name); return false; }
    return true;
}

// Launches the kernel once and returns its profiled execution time in ns
// (device-side, from the event), or a negative value on error.
static double launch_ns(cl_command_queue q, cl_kernel k, cl_mem d_in, cl_mem d_out,
                        cl_uint stride, cl_uint count) {
    clSetKernelArg(k, 0, sizeof(cl_mem), &d_in);
    clSetKernelArg(k, 1, sizeof(cl_mem), &d_out);
    clSetKernelArg(k, 2, sizeof(cl_uint), &stride);
    clSetKernelArg(k, 3, sizeof(cl_uint), &count);
    size_t gws = count;
    cl_event ev = nullptr;
    if (clEnqueueNDRangeKernel(q, k, 1, nullptr, &gws, nullptr, 0, nullptr, &ev) != CL_SUCCESS) return -1.0;
    if (clWaitForEvents(1, &ev) != CL_SUCCESS) { clReleaseEvent(ev); return -1.0; }
    cl_ulong t0 = 0, t1 = 0;
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr);
    clReleaseEvent(ev);
    return static_cast<double>(t1 - t0);
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

int main(int argc, char** argv) {
    std::string kernel_dir = "kernels";
#ifdef SECP256K1_OPENCL_KERNEL_DIR
    kernel_dir = SECP256K1_OPENCL_KERNEL_DIR;
#endif
    uint32_t keys = 1u << 20;
    int iters = 20;
    int reps = 11;
    int platform_idx = 0;
    int device_idx = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", what); std::exit(2); }
            return argv[++i];
        };
        if (a == "--kernel-dir") kernel_dir = next("--kernel-dir");
        else if (a == "--keys") keys = static_cast<uint32_t>(std::strtoul(next("--keys"), nullptr, 10));
        else if (a == "--iters") iters = std::atoi(next("--iters"));
        else if (a == "--reps") reps = std::atoi(next("--reps"));
        else if (a == "--platform") platform_idx = std::atoi(next("--platform"));
        else if (a == "--device") device_idx = std::atoi(next("--device"));
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (keys == 0 || iters <= 0 || reps <= 0) { std::fprintf(stderr, "keys, iters and reps must be positive\n"); return 2; }

    cl_uint nplat = 0;
    CL_CHECK(clGetPlatformIDs(0, nullptr, &nplat));
    if (nplat == 0 || platform_idx < 0 || static_cast<cl_uint>(platform_idx) >= nplat) {
        std::fprintf(stderr, "no such OpenCL platform\n");
        return 1;
    }
    std::vector<cl_platform_id> plats(nplat);
    CL_CHECK(clGetPlatformIDs(nplat, plats.data(), nullptr));
    cl_uint ndev = 0;
    CL_CHECK(clGetDeviceIDs(plats[platform_idx], CL_DEVICE_TYPE_ALL, 0, nullptr, &ndev));
    if (ndev == 0 || device_idx < 0 || static_cast<cl_uint>(device_idx) >= ndev) {
        std::fprintf(stderr, "no such OpenCL device\n");
        return 1;
    }
    std::vector<cl_device_id> devs(ndev);
    CL_CHECK(clGetDeviceIDs(plats[platform_idx], CL_DEVICE_TYPE_ALL, ndev, devs.data(), nullptr));
    cl_device_id dev = devs[device_idx];

    char dev_name[256] = {};
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(dev_name) - 1, dev_name, nullptr);

    cl_int err = CL_SUCCESS;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    CL_CHECK(err);
    cl_command_queue q = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);
    CL_CHECK(err);

    const std::string new_src = load_file(kernel_dir + "/secp256k1_hash160.cl");
    if (new_src.empty()) {
        std::fprintf(stderr, "cannot read %s/secp256k1_hash160.cl (use --kernel-dir)\n", kernel_dir.c_str());
        return 1;
    }
    Form old_form; old_form.name = "old";
    Form new_form; new_form.name = "new";
    const std::string opts = "-cl-std=CL1.2 -I " + kernel_dir;
    if (!build_form(old_form, ctx, dev, OLD_HASH160_SRC, opts)) return 1;
    if (!build_form(new_form, ctx, dev, new_src, opts)) return 1;

    std::printf("device: %s\n", dev_name);
    std::printf("keys=%u iters=%d reps=%d (median over reps; ns/hash is device kernel time / (keys*iters))\n",
                keys, iters, reps);

    const uint32_t strides[] = {33, 65};
    bool all_agree = true;
    uint64_t rng = 0x243F6A8885A308D3ULL;
    auto next_rand = [&rng]() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; };

    for (uint32_t stride : strides) {
        std::vector<uint8_t> host_keys(static_cast<size_t>(stride) * keys);
        for (auto& b : host_keys) b = static_cast<uint8_t>(next_rand() >> 24);
        for (uint32_t i = 0; i < keys; ++i) {
            host_keys[static_cast<size_t>(i) * stride] = (stride == 33) ? static_cast<uint8_t>(2 + (i & 1)) : 4;
        }

        cl_mem d_in = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, host_keys.size(),
                                     host_keys.data(), &err);
        CL_CHECK(err);
        cl_mem d_out_old = clCreateBuffer(ctx, CL_MEM_READ_WRITE, static_cast<size_t>(keys) * 20, nullptr, &err);
        CL_CHECK(err);
        cl_mem d_out_new = clCreateBuffer(ctx, CL_MEM_READ_WRITE, static_cast<size_t>(keys) * 20, nullptr, &err);
        CL_CHECK(err);

        // Correctness first: both forms must agree on every key.
        if (launch_ns(q, old_form.kernel, d_in, d_out_old, stride, keys) < 0 ||
            launch_ns(q, new_form.kernel, d_in, d_out_new, stride, keys) < 0) {
            std::fprintf(stderr, "launch failed (stride %u)\n", stride);
            return 1;
        }
        std::vector<uint8_t> h_old(static_cast<size_t>(keys) * 20), h_new(static_cast<size_t>(keys) * 20);
        CL_CHECK(clEnqueueReadBuffer(q, d_out_old, CL_TRUE, 0, h_old.size(), h_old.data(), 0, nullptr, nullptr));
        CL_CHECK(clEnqueueReadBuffer(q, d_out_new, CL_TRUE, 0, h_new.size(), h_new.data(), 0, nullptr, nullptr));
        const bool agree = (h_old == h_new);
        all_agree = all_agree && agree;
        if (!agree) {
            for (uint32_t i = 0; i < keys; ++i) {
                if (std::memcmp(&h_old[static_cast<size_t>(i) * 20], &h_new[static_cast<size_t>(i) * 20], 20) != 0) {
                    std::fprintf(stderr, "MISMATCH stride %u first at key %u\n", stride, i);
                    break;
                }
            }
        }

        // Timing: alternate which form goes first on each rep (ABBA-style) so
        // drift and clock ramp do not favour one form.
        std::vector<double> t_old, t_new;
        for (int r = 0; r < reps; ++r) {
            auto time_form = [&](Form& f, cl_mem out) {
                double total = 0.0;
                for (int it = 0; it < iters; ++it) {
                    const double ns = launch_ns(q, f.kernel, d_in, out, stride, keys);
                    if (ns < 0) return -1.0;
                    total += ns;
                }
                return total / (static_cast<double>(keys) * iters);
            };
            double a, b;
            if (r % 2 == 0) { a = time_form(old_form, d_out_old); b = time_form(new_form, d_out_new); t_old.push_back(a); t_new.push_back(b); }
            else            { b = time_form(new_form, d_out_new); a = time_form(old_form, d_out_old); t_old.push_back(a); t_new.push_back(b); }
            if (a < 0 || b < 0) { std::fprintf(stderr, "timing launch failed (stride %u)\n", stride); return 1; }
        }
        const double mo = median(t_old), mn = median(t_new);
        std::printf("len=%u  old=%.4f ns/hash  new=%.4f ns/hash  old/new=%.3f  agree=%s\n",
                    stride, mo, mn, mn > 0 ? mo / mn : 0.0, agree ? "yes" : "NO");
        std::printf("  old min..max: %.4f..%.4f   new min..max: %.4f..%.4f\n",
                    *std::min_element(t_old.begin(), t_old.end()), *std::max_element(t_old.begin(), t_old.end()),
                    *std::min_element(t_new.begin(), t_new.end()), *std::max_element(t_new.begin(), t_new.end()));

        clReleaseMemObject(d_in);
        clReleaseMemObject(d_out_old);
        clReleaseMemObject(d_out_new);
    }

    clReleaseKernel(old_form.kernel); clReleaseProgram(old_form.program);
    clReleaseKernel(new_form.kernel); clReleaseProgram(new_form.program);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);

    std::printf("%s\n", all_agree ? "RESULT: old and new agree on every key" : "RESULT: MISMATCH between old and new");
    return all_agree ? 0 : 1;
}
