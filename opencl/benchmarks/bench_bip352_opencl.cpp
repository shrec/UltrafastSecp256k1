#include "secp256k1_opencl.hpp"
#include "secp256k1/batch_add_affine.hpp"
#include "secp256k1/fast.hpp"
#include "secp256k1/glv.hpp"
#include "secp256k1/tagged_hash.hpp"

#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using CpuPoint = secp256k1::fast::Point;
using CpuScalar = secp256k1::fast::Scalar;
using CpuField = secp256k1::fast::FieldElement;
using OclAffine = secp256k1::opencl::AffinePoint;
using OclField = secp256k1::opencl::FieldElement;
using OclScalar = secp256k1::opencl::Scalar;

namespace {

constexpr int BENCH_N = 500000;
constexpr int BENCH_WARMUP = 3;
constexpr int BENCH_PASSES = 11;
// Extra passes before autotuning to bring GPU from idle P-state to boost clock.
// Without this, autotuning measures idle-clock performance and may pick a
// suboptimal local size (mirrors the same fix applied to the CUDA benchmark).
constexpr int BENCH_CLOCK_WARMUP = 15;
// RTX 5060 Ti (and most NVIDIA): warp=32, SM occupancy peaks at 128-256 threads.
// Previous defaults (64/32) left SMs underutilized.
constexpr int DEFAULT_LOCAL_SIZE_FUSED = 128;
constexpr int DEFAULT_LOCAL_SIZE_LUT   = 128;
constexpr std::size_t LUT_WINDOWS = 16;
constexpr std::size_t LUT_ENTRIES = 65536;

constexpr uint8_t SCAN_KEY[32] = {
    0xc4,0x23,0x9f,0xd6,0xfc,0x3d,0xb6,0xe2,
    0x2b,0x8b,0xed,0x6a,0x49,0x21,0x9e,0x4e,
    0x30,0xd7,0xd6,0xa3,0xb9,0x82,0x94,0xb1,
    0x38,0xaf,0x4a,0xd3,0x00,0xda,0x1a,0x42
};

constexpr uint8_t SPEND_PUBKEY_COMPRESSED[33] = {
    0x02,
    0xe2,0xed,0x4b,0x9c,0xe9,0x14,0x5e,0x17,
    0x21,0xf1,0x1f,0x99,0x5f,0x72,0x6e,0xf8,
    0xcf,0x50,0xfc,0x85,0x92,0x89,0xac,0x94,
    0x4b,0x2d,0xaf,0xe5,0x03,0xa3,0xc7,0x4c
};

// Must match BIP352ScanKeyGlv typedef in secp256k1_bip352.cl exactly.
struct BIP352ScanKeyGlv {
    std::int8_t  wnaf1[130]{};  // +0:   wNAF digits for k1 half-scalar
    std::int8_t  wnaf2[130]{};  // +130: wNAF digits for k2 half-scalar
    std::uint8_t k1_neg{0};     // +260: 1 if k1 negative (negate base.y)
    std::uint8_t flip_phi{0};   // +261: 1 if phi table y should be negated
    std::uint8_t pad0{0};       // +262: padding
    std::uint8_t pad1{0};       // +263: padding
}; // Total: 264 bytes

// Compute 5-bit wNAF digits for a 128-bit half-scalar.
// Mirrors the GPU's scalar_to_wnaf fixed-130-iteration version.
// scalar_bytes: big-endian 32-byte scalar (upper 128 bits should be zero for GLV halves).
static void host_compute_wnaf(const std::uint8_t* scalar_bytes, std::int8_t wnaf[130]) {
    // Convert big-endian bytes to 4 little-endian 64-bit limbs (limb[0] = LSW).
    std::uint64_t s[4] = {};
    for (int limb = 0; limb < 4; ++limb) {
        std::uint64_t v = 0;
        int base = limb * 8;
        for (int i = 0; i < 8; ++i) v = (v << 8) | scalar_bytes[base + i];
        s[3 - limb] = v;
    }
    for (int i = 0; i < 130; i++) {
        if (s[0] & 1ULL) {
            int d = (int)(s[0] & 0x1FULL);
            if (d >= 16) {
                d -= 32;
                std::uint64_t add = (std::uint64_t)(-d);
                std::uint64_t prev = s[0]; s[0] += add;
                if (s[0] < prev) { for (int j = 1; j < 4; j++) if (++s[j]) break; }
            } else {
                std::uint64_t prev = s[0]; s[0] -= (std::uint64_t)d;
                if (s[0] > prev) { for (int j = 1; j < 4; j++) if (s[j]--) break; }
            }
            wnaf[i] = (std::int8_t)d;
        } else {
            wnaf[i] = 0;
        }
        s[0] = (s[0] >> 1) | (s[1] << 63);
        s[1] = (s[1] >> 1) | (s[2] << 63);
        s[2] = (s[2] >> 1) | (s[3] << 63);
        s[3] >>= 1;
    }
}

static const uint32_t host_sha256_k[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

inline uint32_t rotr32(uint32_t a, uint32_t b) {
    return (a >> b) | (a << (32 - b));
}

void host_sha256(const uint8_t* msg, size_t len, uint8_t out[32]) {
    uint32_t h0=0x6a09e667, h1=0xbb67ae85, h2=0x3c6ef372, h3=0xa54ff53a;
    uint32_t h4=0x510e527f, h5=0x9b05688c, h6=0x1f83d9ab, h7=0x5be0cd19;

    size_t bit_len = len * 8;
    size_t padded = ((len + 9 + 63) / 64) * 64;
    std::vector<uint8_t> buf(padded, 0);
    std::memcpy(buf.data(), msg, len);
    buf[len] = 0x80;
    for (int i = 7; i >= 0; --i) buf[padded - 1 - i] = static_cast<uint8_t>(bit_len >> (i * 8));

    for (size_t off = 0; off < padded; off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; i++) {
            w[i] = (static_cast<uint32_t>(buf[off+i*4]) << 24) |
                   (static_cast<uint32_t>(buf[off+i*4+1]) << 16) |
                   (static_cast<uint32_t>(buf[off+i*4+2]) << 8) |
                   buf[off+i*4+3];
        }
        for (int i = 16; i < 64; i++) {
            uint32_t s0 = rotr32(w[i-15], 7) ^ rotr32(w[i-15], 18) ^ (w[i-15] >> 3);
            uint32_t s1 = rotr32(w[i-2], 17) ^ rotr32(w[i-2], 19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a=h0,b=h1,c=h2,d=h3,e=h4,f=h5,g=h6,hh=h7;
        for (int i = 0; i < 64; i++) {
            uint32_t S1 = rotr32(e,6)^rotr32(e,11)^rotr32(e,25);
            uint32_t ch = (e&f)^(~e&g);
            uint32_t t1 = hh+S1+ch+host_sha256_k[i]+w[i];
            uint32_t S0 = rotr32(a,2)^rotr32(a,13)^rotr32(a,22);
            uint32_t maj = (a&b)^(a&c)^(b&c);
            uint32_t t2 = S0+maj;
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h0+=a; h1+=b; h2+=c; h3+=d; h4+=e; h5+=f; h6+=g; h7+=hh;
    }

    auto store = [&](uint32_t v, int i) {
        out[i*4] = static_cast<uint8_t>(v >> 24);
        out[i*4+1] = static_cast<uint8_t>(v >> 16);
        out[i*4+2] = static_cast<uint8_t>(v >> 8);
        out[i*4+3] = static_cast<uint8_t>(v);
    };
    store(h0,0); store(h1,1); store(h2,2); store(h3,3);
    store(h4,4); store(h5,5); store(h6,6); store(h7,7);
}

CpuPoint point_from_compressed(const uint8_t* pub33) {
    if (pub33[0] != 0x02 && pub33[0] != 0x03) return CpuPoint::infinity();
    CpuField x;
    if (!CpuField::parse_bytes_strict(pub33 + 1, x)) return CpuPoint::infinity();
    auto x2 = x * x;
    auto x3 = x2 * x;
    auto y2 = x3 + CpuField::from_uint64(7);
    auto t = y2;
    auto a = t.square() * t;
    auto b = a.square() * t;
    auto c = b.square().square().square() * b;
    auto d = c.square().square().square() * b;
    auto e = d.square().square() * a;
    auto f = e;
    for (int i = 0; i < 11; ++i) f = f.square();
    f = f * e;
    auto g = f;
    for (int i = 0; i < 22; ++i) g = g.square();
    g = g * f;
    auto h = g;
    for (int i = 0; i < 44; ++i) h = h.square();
    h = h * g;
    auto j = h;
    for (int i = 0; i < 88; ++i) j = j.square();
    j = j * h;
    auto k = j;
    for (int i = 0; i < 44; ++i) k = k.square();
    k = k * g;
    auto m = k.square().square().square() * b;
    auto y = m;
    for (int i = 0; i < 23; ++i) y = y.square();
    y = y * f;
    for (int i = 0; i < 6; ++i) y = y.square();
    y = y * a;
    y = y.square().square();
    if (!(y * y == y2)) return CpuPoint::infinity();
    auto y_bytes = y.to_bytes();
    bool y_is_odd = (y_bytes[31] & 1) != 0;
    bool want_odd = (pub33[0] == 0x03);
    if (y_is_odd != want_odd) y = CpuField::from_uint64(0) - y;
    return CpuPoint::from_affine(x, y);
}

OclField bytes_to_ocl_field(const uint8_t* bytes32) {
    OclField out{};
    for (int limb = 0; limb < 4; ++limb) {
        uint64_t v = 0;
        int base = limb * 8;
        for (int i = 0; i < 8; ++i) {
            v = (v << 8) | bytes32[base + i];
        }
        out.limbs[3 - limb] = v;
    }
    return out;
}


OclAffine to_ocl_affine(const CpuPoint& p) {
    OclAffine out{};
    auto x = p.x().to_bytes();
    auto y = p.y().to_bytes();
    out.x = bytes_to_ocl_field(x.data());
    out.y = bytes_to_ocl_field(y.data());
    return out;
}

OclAffine to_ocl_affine(const secp256k1::fast::AffinePointCompact& p) {
    OclAffine out{};
    auto x = p.x.to_bytes();
    auto y = p.y.to_bytes();
    out.x = bytes_to_ocl_field(x.data());
    out.y = bytes_to_ocl_field(y.data());
    return out;
}

uint64_t extract_upper_64(const uint8_t* x_bytes) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v = (v << 8) | x_bytes[i];
    return v;
}

std::string read_text(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open: " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string dirname_of(const std::string& path) {
    auto pos = path.find_last_of("/\\");
    return pos == std::string::npos ? "." : path.substr(0, pos);
}

std::string trim_left(std::string s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.erase(s.begin());
    return s;
}

std::string expand_kernel_file(const std::string& path, std::set<std::string>& include_stack) {
    if (include_stack.count(path)) return {};
    include_stack.insert(path);
    std::istringstream in(read_text(path));
    std::ostringstream out;
    std::string dir = dirname_of(path);
    std::string line;
    while (std::getline(in, line)) {
        std::string trimmed = trim_left(line);
        if (trimmed.rfind("#include \"", 0) == 0) {
            auto start = trimmed.find('"') + 1;
            auto end = trimmed.find('"', start);
            std::string child = dir + "/" + trimmed.substr(start, end - start);
            out << expand_kernel_file(child, include_stack);
            continue;
        }
        out << line << '\n';
    }
    include_stack.erase(path);
    return out.str();
}

std::string load_bip352_kernel_source() {
    std::set<std::string> stack;
    return expand_kernel_file(std::string(SECP256K1_OPENCL_KERNEL_DIR) + "/secp256k1_bip352.cl", stack);
}

std::vector<OclAffine> build_generator_lut_host() {
    std::vector<OclAffine> lut(LUT_WINDOWS * LUT_ENTRIES);
    CpuPoint base = CpuPoint::generator();

    for (std::size_t win = 0; win < LUT_WINDOWS; ++win) {
        std::cout << "  Building LUT window " << win + 1 << "/" << LUT_WINDOWS << "...\n";
        auto base_x = base.x();
        auto base_y = base.y();
        auto table = (win == 0)
            ? secp256k1::fast::precompute_g_multiples(LUT_ENTRIES - 1)
            : secp256k1::fast::precompute_point_multiples(base_x, base_y, LUT_ENTRIES - 1);

        lut[win * LUT_ENTRIES] = OclAffine{};
        for (std::size_t i = 0; i < table.size(); ++i) {
            lut[win * LUT_ENTRIES + i + 1] = to_ocl_affine(table[i]);
        }

        for (int i = 0; i < 16; ++i) base.dbl_inplace();
    }

    return lut;
}

BIP352ScanKeyGlv build_scan_glv_plan() {
    BIP352ScanKeyGlv out{};
    auto scan_scalar = CpuScalar::from_bytes(SCAN_KEY);
    auto decomp = secp256k1::fast::glv_decompose(scan_scalar);
    auto k1 = decomp.k1.to_bytes();
    auto k2 = decomp.k2.to_bytes();
    out.k1_neg  = decomp.k1_neg ? 1 : 0;
    out.flip_phi = (decomp.k1_neg != decomp.k2_neg) ? 1 : 0;
    host_compute_wnaf(k1.data(), out.wnaf1);
    host_compute_wnaf(k2.data(), out.wnaf2);
    return out;
}

double median_iqr(std::vector<double> samples) {
    if (samples.empty()) return 0.0;
    std::sort(samples.begin(), samples.end());
    const int n = static_cast<int>(samples.size());
    if (n < 4) return samples[n / 2];
    double q1 = samples[n / 4];
    double q3 = samples[(3 * n) / 4];
    double iqr = q3 - q1;
    double lo = q1 - 1.5 * iqr;
    double hi = q3 + 1.5 * iqr;
    std::vector<double> filtered;
    filtered.reserve(samples.size());
    for (double v : samples) {
        if (v >= lo && v <= hi) filtered.push_back(v);
    }
    if (filtered.empty()) filtered = std::move(samples);
    return filtered[filtered.size() / 2];
}

void check_cl(cl_int err, const char* what) {
    if (err != CL_SUCCESS) {
        throw std::runtime_error(std::string(what) + " failed with OpenCL error " + std::to_string(err));
    }
}

// Autotune OpenCL local_size by running a few passes at candidate sizes.
// Mirrors CUDA's autotune_gpu_tpb. Returns best local size found.
static int autotune_local_size(
    const char* label,
    cl_command_queue cl_q,
    cl_kernel kernel,
    size_t count,
    size_t max_wg_size,
    std::initializer_list<int> candidates)
{
    std::printf("Autotuning %s local size...\n", label);
    int best = 0;
    double best_ns = 0.0;

    for (int ls : candidates) {
        if (ls <= 0 || static_cast<size_t>(ls) > max_wg_size) continue;

        size_t local  = static_cast<size_t>(ls);
        size_t global = ((count + local - 1) / local) * local;

        // warmup
        for (int w = 0; w < 2; ++w) {
            cl_int err2 = clEnqueueNDRangeKernel(cl_q, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);
            if (err2 != CL_SUCCESS) goto next;
        }
        clFinish(cl_q);

        {
            constexpr int SAMPLE_PASSES = 5;
            constexpr int SAMPLE_REPS   = 10;
            std::vector<double> samples;
            samples.reserve(SAMPLE_PASSES);
            for (int p = 0; p < SAMPLE_PASSES; ++p) {
                auto t0 = std::chrono::high_resolution_clock::now();
                for (int r = 0; r < SAMPLE_REPS; ++r) {
                    cl_int err2 = clEnqueueNDRangeKernel(cl_q, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);
                    if (err2 != CL_SUCCESS) goto next;
                }
                clFinish(cl_q);
                auto t1 = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                samples.push_back((ms * 1e6) / (static_cast<double>(count) * SAMPLE_REPS));
            }
            double ns = median_iqr(samples);
            std::printf("  local=%3d -> %8.1f ns/op\n", ls, ns);
            if (best == 0 || ns < best_ns) { best = ls; best_ns = ns; }
        }
        next:;
    }

    if (best == 0) best = DEFAULT_LOCAL_SIZE_FUSED;
    std::printf("  selected local=%d for %s\n\n", best, label);
    return best;
}

} // namespace

int main(int argc, char** argv) {
    bool prefer_intel = false;
    bool use_lut = false;
    int platform_id = -1;
    int device_id = 0;
    int batch_n = BENCH_N;
    int local_size = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--intel") prefer_intel = true;
        else if (arg == "--nvidia") prefer_intel = false;
        else if (arg == "--lut") use_lut = true;
        else if (arg == "--platform" && i + 1 < argc) platform_id = std::atoi(argv[++i]);
        else if (arg == "--device" && i + 1 < argc) device_id = std::atoi(argv[++i]);
        else if (arg == "--batch" && i + 1 < argc) batch_n = std::atoi(argv[++i]);
        else if (arg == "--local" && i + 1 < argc) local_size = std::atoi(argv[++i]);
    }
    if (local_size == 0) {
        local_size = use_lut ? DEFAULT_LOCAL_SIZE_LUT : DEFAULT_LOCAL_SIZE_FUSED;
    }

    secp256k1::opencl::DeviceConfig cfg;
    cfg.prefer_intel = prefer_intel;
    cfg.verbose = true;
    cfg.platform_id = platform_id;
    cfg.device_id = device_id;
    auto ctx = secp256k1::opencl::Context::create(cfg);
    if (!ctx || !ctx->is_valid()) {
        std::cerr << "Failed to create OpenCL context\n";
        return 1;
    }

    cl_context cl_ctx = static_cast<cl_context>(ctx->native_context());
    cl_command_queue cl_q = static_cast<cl_command_queue>(ctx->native_queue());
    cl_device_id cl_dev = nullptr;
    check_cl(clGetCommandQueueInfo(cl_q, CL_QUEUE_DEVICE, sizeof(cl_dev), &cl_dev, nullptr),
             "clGetCommandQueueInfo(CL_QUEUE_DEVICE)");

    std::cout << "============================================================\n";
    std::cout << "  BIP-352 Silent Payments Pipeline: CPU vs OpenCL\n";
    std::cout << "============================================================\n";
    std::cout << "  Device: " << ctx->device_info().name << " (" << ctx->device_info().vendor << ")\n";
    std::cout << "  N = " << batch_n << " tweak points, " << BENCH_PASSES << " passes (median)\n\n";
    std::cout << "  Local size = " << local_size << "\n\n";

    std::cout << "Generating " << batch_n << " deterministic tweak points...\n";
    std::vector<OclAffine> tweaks(static_cast<size_t>(batch_n));
    CpuPoint last_tweak = CpuPoint::infinity();
    uint8_t seed[32];
    const char* tag = "bench_bip352_seed";
    host_sha256(reinterpret_cast<const uint8_t*>(tag), std::strlen(tag), seed);
    for (int i = 0; i < batch_n; ++i) {
        uint8_t buf[36];
        std::memcpy(buf, seed, 32);
        buf[32] = static_cast<uint8_t>((i >> 24) & 0xff);
        buf[33] = static_cast<uint8_t>((i >> 16) & 0xff);
        buf[34] = static_cast<uint8_t>((i >> 8) & 0xff);
        buf[35] = static_cast<uint8_t>(i & 0xff);
        uint8_t scalar_bytes[32];
        host_sha256(buf, 36, scalar_bytes);
        CpuScalar s = CpuScalar::from_bytes(scalar_bytes);
        CpuPoint p = CpuPoint::generator().scalar_mul(s);
        if (i == batch_n - 1) last_tweak = p;
        tweaks[static_cast<size_t>(i)] = to_ocl_affine(p);
    }
    std::cout << "Done.\n";

    CpuPoint spend_cpu = point_from_compressed(SPEND_PUBKEY_COMPRESSED);
    if (spend_cpu.is_infinity()) {
        std::cerr << "Failed to decode spend pubkey\n";
        return 1;
    }
    OclAffine spend = to_ocl_affine(spend_cpu);

    std::cout << "Building OpenCL BIP352 pipeline kernel...\n";
    std::string source = load_bip352_kernel_source();
    const char* src_ptr = source.c_str();
    size_t src_len = source.size();
    cl_int err = CL_SUCCESS;
    cl_program program = clCreateProgramWithSource(cl_ctx, 1, &src_ptr, &src_len, &err);
    check_cl(err, "clCreateProgramWithSource");
    std::string build_options = "-cl-std=CL1.2 -cl-fast-relaxed-math -cl-mad-enable"
        " -cl-nv-opt-level=3";
    err = clBuildProgram(program, 1, &cl_dev, build_options.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(program, cl_dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(program, cl_dev, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "Build failed:\n" << log << "\n";
        return 1;
    }
    const char* kernel_name = use_lut ? "bip352_pipeline_kernel_lut" : "bip352_pipeline_kernel";
    cl_kernel kernel = clCreateKernel(program, kernel_name, &err);
    check_cl(err, kernel_name);
    std::cout << "Done.\n";

    size_t count = static_cast<size_t>(batch_n);
    size_t tweak_bytes = count * sizeof(OclAffine);
    std::vector<uint64_t> prefixes(count);
    std::vector<OclAffine> gen_lut;
    BIP352ScanKeyGlv scan_plan{};

    // Both paths now use BIP352ScanKeyGlv with precomputed wNAF digits.
    scan_plan = build_scan_glv_plan();
    cl_mem d_tweaks = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, tweak_bytes, tweaks.data(), &err);
    check_cl(err, "clCreateBuffer(d_tweaks)");
    cl_mem d_scan = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(BIP352ScanKeyGlv), &scan_plan, &err);
    check_cl(err, "clCreateBuffer(d_scan)");
    cl_mem d_spend = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(OclAffine), &spend, &err);
    check_cl(err, "clCreateBuffer(d_spend)");
    cl_mem d_prefixes = clCreateBuffer(cl_ctx, CL_MEM_WRITE_ONLY, count * sizeof(uint64_t), nullptr, &err);
    check_cl(err, "clCreateBuffer(d_prefixes)");
    cl_mem d_gen_lut = nullptr;
    if (use_lut) {
        std::cout << "Building CPU generator LUT (" << (LUT_WINDOWS * LUT_ENTRIES) << " affine points)...\n";
        gen_lut = build_generator_lut_host();
        std::cout << "Uploading generator LUT to OpenCL...\n";
        d_gen_lut = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   gen_lut.size() * sizeof(OclAffine), gen_lut.data(), &err);
        check_cl(err, "clCreateBuffer(d_gen_lut)");
    }

    cl_uint cl_count = static_cast<cl_uint>(count);
    check_cl(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_tweaks), "clSetKernelArg(0)");
    check_cl(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_scan), "clSetKernelArg(1)");
    check_cl(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_spend), "clSetKernelArg(2)");
    if (use_lut) {
        check_cl(clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_gen_lut), "clSetKernelArg(3)");
        check_cl(clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_prefixes), "clSetKernelArg(4)");
        check_cl(clSetKernelArg(kernel, 5, sizeof(cl_uint), &cl_count), "clSetKernelArg(5)");
    } else {
        check_cl(clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_prefixes), "clSetKernelArg(3)");
        check_cl(clSetKernelArg(kernel, 4, sizeof(cl_uint), &cl_count), "clSetKernelArg(4)");
    }

    if (local_size <= 0) {
        throw std::runtime_error("local size must be positive");
    }
    if (static_cast<std::size_t>(local_size) > ctx->device_info().max_work_group_size) {
        throw std::runtime_error("local size exceeds device max work group size");
    }

    // GPU clock pre-warmup: LUT build + wNAF precompute are brief and may not be
    // enough to bring the GPU from idle P-state to sustained boost clock before
    // autotuning starts.  Run BENCH_CLOCK_WARMUP passes first so the clock governor
    // has time to ramp up, then autotune from a warm state.
    {
        int default_ls = use_lut ? DEFAULT_LOCAL_SIZE_LUT : DEFAULT_LOCAL_SIZE_FUSED;
        size_t wg_global = ((static_cast<size_t>(count) + static_cast<size_t>(default_ls) - 1)
                           / static_cast<size_t>(default_ls)) * static_cast<size_t>(default_ls);
        size_t wg_local  = static_cast<size_t>(default_ls);
        printf("Pre-warming GPU clock (%d passes)...\n", BENCH_CLOCK_WARMUP);
        for (int w = 0; w < BENCH_CLOCK_WARMUP; ++w) {
            check_cl(clEnqueueNDRangeKernel(cl_q, kernel, 1, nullptr,
                &wg_global, &wg_local, 0, nullptr, nullptr), "clock warmup");
            check_cl(clFinish(cl_q), "clock warmup finish");
        }
        printf("Done.\n\n");
    }

    // Autotune: find optimal local size among candidates (mirrors CUDA autotune_gpu_tpb).
    // Only autotune when no explicit --local was given (i.e., we're still at the default).
    {
        int default_ls = use_lut ? DEFAULT_LOCAL_SIZE_LUT : DEFAULT_LOCAL_SIZE_FUSED;
        if (local_size == default_ls) {
            int tuned = autotune_local_size(
                use_lut ? "LUT kernel" : "fused kernel",
                cl_q, kernel, count,
                ctx->device_info().max_work_group_size,
                {64, 128, 256, 384});
            local_size = tuned;
        }
    }

    size_t global = ((count + static_cast<size_t>(local_size) - 1) / static_cast<size_t>(local_size)) * static_cast<size_t>(local_size);
    size_t local = static_cast<size_t>(local_size);
    std::cout << "  Running with local_size=" << local_size << "\n";

    for (int i = 0; i < BENCH_WARMUP; ++i) {
        check_cl(clEnqueueNDRangeKernel(cl_q, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                 "clEnqueueNDRangeKernel(warmup)");
    }
    check_cl(clFinish(cl_q), "clFinish(warmup)");

    std::vector<double> samples;
    samples.reserve(BENCH_PASSES);
    std::cout << "\n--- OpenCL (" << (use_lut ? "fused pipeline + LUT" : "fused pipeline") << ") ---\n";
    for (int pass = 0; pass < BENCH_PASSES; ++pass) {
        auto t0 = std::chrono::high_resolution_clock::now();
        check_cl(clEnqueueNDRangeKernel(cl_q, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                 "clEnqueueNDRangeKernel");
        check_cl(clFinish(cl_q), "clFinish");
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        samples.push_back((ms * 1e6) / static_cast<double>(count));
        std::printf("  pass %2d: %8.3f ms\n", pass + 1, ms);
    }
    double ns_per_op = median_iqr(samples);
    double ops_per_sec = 1e9 / ns_per_op;

    check_cl(clEnqueueReadBuffer(cl_q, d_prefixes, CL_TRUE, 0, count * sizeof(uint64_t), prefixes.data(), 0, nullptr, nullptr),
             "clEnqueueReadBuffer");

    CpuScalar scan_scalar = CpuScalar::from_bytes(SCAN_KEY);
    CpuPoint shared = last_tweak.scalar_mul(scan_scalar);
    auto shared_comp = shared.to_compressed();
    uint8_t shared_ser[37];
    std::memcpy(shared_ser, shared_comp.data(), 33);
    shared_ser[33] = shared_ser[34] = shared_ser[35] = shared_ser[36] = 0;
    auto tagged = secp256k1::detail::cached_tagged_hash(
        secp256k1::detail::make_tag_midstate("BIP0352/SharedSecret"),
        shared_ser,
        sizeof(shared_ser));
    CpuScalar hs = CpuScalar::from_bytes(tagged.data());
    CpuPoint out = CpuPoint::generator().scalar_mul(hs);
    CpuPoint cand = spend_cpu;
    cand.add_inplace(out);
    uint64_t cpu_validation = extract_upper_64(cand.x_only_bytes().data());
    uint64_t ocl_validation = prefixes.back();

    std::printf("\n  OpenCL%s: %.1f ns/op (%.2f M/s)\n", use_lut ? " LUT" : "", ns_per_op, ops_per_sec / 1e6);
    std::printf("  validation prefix: 0x%016llx\n", static_cast<unsigned long long>(ocl_validation));
    // CUDA reference: bench_bip352 on RTX 5060 Ti (SM 12.0, 36 SMs).
    // Measured with BENCH_CLOCK_WARMUP=15, batch=500K, CUDA_SEPARABLE_COMPILATION=OFF,
    // unchecked ops in all scalar mul inner loops, BENCH_MULTI=20 (amortised dispatch).
    // Commits: 9bfe1b21, 806b1602, + full-bench multi-launch fix.
    // GLV tpb=384: 179.1 ns/op (5.58 M/s).  LUT tpb=384: 90.6 ns/op (11.04 M/s).
    constexpr double CUDA_GLV_NS = 179.1;
    constexpr double CUDA_LUT_NS = 90.6;
    double cuda_ref = use_lut ? CUDA_LUT_NS : CUDA_GLV_NS;
    std::printf("  CUDA reference:    %.1f ns/op (%.2f M/s) [%s]\n",
                cuda_ref, 1e9 / cuda_ref / 1e6, use_lut ? "LUT" : "GLV");
    std::printf("  gap vs CUDA:       %.2fx\n", ns_per_op / cuda_ref);
    std::printf("  Validation: %s\n", cpu_validation == ocl_validation ? "[OK] MATCH" : "[FAIL] MISMATCH");

    clReleaseMemObject(d_tweaks);
    clReleaseMemObject(d_scan);
    clReleaseMemObject(d_spend);
    clReleaseMemObject(d_prefixes);
    if (d_gen_lut) clReleaseMemObject(d_gen_lut);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    return cpu_validation == ocl_validation ? 0 : 2;
}
