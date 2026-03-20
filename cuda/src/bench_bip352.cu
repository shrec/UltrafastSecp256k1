// ============================================================================
// bench_bip352 -- BIP-352 Silent Payments Scanning Pipeline: CPU vs GPU
// ============================================================================
//
// GPU port of the bench_bip352 benchmark (github.com/shrec/bench_bip352).
//
// Pipeline per row:
//   1. k*P   -- Scalar multiply tweak point by scan private key
//   2. Serialize -- Compress shared secret to 33-byte SEC1
//   3. Tagged SHA-256 -- BIP0352/SharedSecret tagged hash
//   4. k*G   -- Generator multiply by hash scalar
//   5. Point add -- spend_pubkey + output_point
//   6. Serialize + prefix -- Compress candidate, extract upper 64 bits
//   7. Prefix match -- Compare against output prefix list
//
// Compares: UltrafastSecp256k1 CPU vs CUDA GPU, per-operation breakdown
//           and full pipeline throughput.
//
// Test vectors: Identical to bench_bip352 (deterministic from SHA-256 seeds)
// ============================================================================

// ---- GPU headers ----
#include "secp256k1.cuh"
#include "ecdsa.cuh"
#include "schnorr.cuh"

// ---- CPU headers ----
#include "secp256k1/fast.hpp"
#include "secp256k1/tagged_hash.hpp"
#include "secp256k1/sha256.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <functional>

using CpuPoint  = secp256k1::fast::Point;
using CpuScalar = secp256k1::fast::Scalar;
using CpuField  = secp256k1::fast::FieldElement;
using CpuKPlan  = secp256k1::fast::KPlan;

// ============================================================================
// Configuration
// ============================================================================
static constexpr int BENCH_N       = 10000;
static constexpr int BENCH_WARMUP  = 3;
static constexpr int BENCH_PASSES  = 11;
static constexpr int DETAIL_N      = 1000;
static constexpr int GPU_TPB       = 256;

// ============================================================================
// Test vector constants (identical to bench_bip352/common.h)
// ============================================================================
static const uint8_t SCAN_KEY[32] = {
    0xc4,0x23,0x9f,0xd6,0xfc,0x3d,0xb6,0xe2,
    0x2b,0x8b,0xed,0x6a,0x49,0x21,0x9e,0x4e,
    0x30,0xd7,0xd6,0xa3,0xb9,0x82,0x94,0xb1,
    0x38,0xaf,0x4a,0xd3,0x00,0xda,0x1a,0x42
};

static const uint8_t SPEND_PUBKEY_COMPRESSED[33] = {
    0x02,
    0xe2,0xed,0x4b,0x9c,0xe9,0x14,0x5e,0x17,
    0x21,0xf1,0x1f,0x99,0x5f,0x72,0x6e,0xf8,
    0xcf,0x50,0xfc,0x85,0x92,0x89,0xac,0x94,
    0x4b,0x2d,0xaf,0xe5,0x03,0xa3,0xc7,0x4c
};

static const int64_t OUTPUT_PREFIXES[] = {
    (int64_t)0x1234567890ABCDEFLL,
    (int64_t)0xFEDCBA0987654321LL,
    (int64_t)0x0011223344556677LL
};
static constexpr int OUTPUT_COUNT = 3;

// ============================================================================
// CUDA error checking
// ============================================================================
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                     cudaGetErrorString(err), __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while(0)

// ============================================================================
// Helpers
// ============================================================================
static inline int64_t extract_upper_64(const uint8_t* x_bytes) {
    int64_t v = 0;
    for (int i = 0; i < 8; i++) v = (v << 8) | x_bytes[i];
    return v;
}

template<typename T>
static inline void DoNotOptimize(T const& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

// ============================================================================
// Host-side SHA-256 (for test vector generation only -- not timed)
// ============================================================================
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

#define ROTR32(a,b) (((a) >> (b)) | ((a) << (32-(b))))

static void host_sha256(const uint8_t* msg, size_t len, uint8_t out[32]) {
    uint32_t h0=0x6a09e667, h1=0xbb67ae85, h2=0x3c6ef372, h3=0xa54ff53a;
    uint32_t h4=0x510e527f, h5=0x9b05688c, h6=0x1f83d9ab, h7=0x5be0cd19;

    // Pad
    size_t bit_len = len * 8;
    size_t padded = ((len + 9 + 63) / 64) * 64;
    std::vector<uint8_t> buf(padded, 0);
    memcpy(buf.data(), msg, len);
    buf[len] = 0x80;
    for (int i = 7; i >= 0; --i) buf[padded - 1 - i] = (uint8_t)(bit_len >> (i * 8));

    for (size_t off = 0; off < padded; off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; i++)
            w[i] = ((uint32_t)buf[off+i*4]<<24)|((uint32_t)buf[off+i*4+1]<<16)|
                   ((uint32_t)buf[off+i*4+2]<<8)|buf[off+i*4+3];
        for (int i = 16; i < 64; i++) {
            uint32_t s0 = ROTR32(w[i-15],7)^ROTR32(w[i-15],18)^(w[i-15]>>3);
            uint32_t s1 = ROTR32(w[i-2],17)^ROTR32(w[i-2],19)^(w[i-2]>>10);
            w[i] = w[i-16]+s0+w[i-7]+s1;
        }
        uint32_t a=h0,b=h1,c=h2,d=h3,e=h4,f=h5,g=h6,hh=h7;
        for (int i = 0; i < 64; i++) {
            uint32_t S1 = ROTR32(e,6)^ROTR32(e,11)^ROTR32(e,25);
            uint32_t ch = (e&f)^(~e&g);
            uint32_t t1 = hh+S1+ch+host_sha256_k[i]+w[i];
            uint32_t S0 = ROTR32(a,2)^ROTR32(a,13)^ROTR32(a,22);
            uint32_t maj = (a&b)^(a&c)^(b&c);
            uint32_t t2 = S0+maj;
            hh=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
        }
        h0+=a;h1+=b;h2+=c;h3+=d;h4+=e;h5+=f;h6+=g;h7+=hh;
    }
    auto store = [&](uint32_t v, int i) {
        out[i*4]=(uint8_t)(v>>24);out[i*4+1]=(uint8_t)(v>>16);
        out[i*4+2]=(uint8_t)(v>>8);out[i*4+3]=(uint8_t)v;
    };
    store(h0,0);store(h1,1);store(h2,2);store(h3,3);
    store(h4,4);store(h5,5);store(h6,6);store(h7,7);
}

// ============================================================================
// CPU Point decompression (same as bench_bip352)
// ============================================================================
static CpuPoint CpuPointFromCompressed(const uint8_t* pub33) {
    if (pub33[0] != 0x02 && pub33[0] != 0x03) return CpuPoint::infinity();
    CpuField x;
    if (!CpuField::parse_bytes_strict(pub33 + 1, x)) return CpuPoint::infinity();
    auto x2 = x * x; auto x3 = x2 * x;
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

// ============================================================================
// GPU device: tagged SHA-256 for BIP-352/SharedSecret
// ============================================================================

// Pre-computed tag hash: SHA256("BIP0352/SharedSecret")
// Both copies are concatenated in the midstate prefix.
__device__ inline void bip352_tagged_sha256(
    const uint8_t* ser, int ser_len,
    uint8_t out[32])
{
    using namespace secp256k1::cuda;

    // Compute SHA256("BIP0352/SharedSecret")
    const uint8_t tag[] = "BIP0352/SharedSecret";
    uint8_t tag_hash[32];
    {
        SHA256Ctx tc; sha256_init(&tc);
        sha256_update(&tc, tag, 20);
        sha256_final(&tc, tag_hash);
    }

    // tagged_hash = SHA256(tag_hash || tag_hash || ser)
    SHA256Ctx ctx; sha256_init(&ctx);
    sha256_update(&ctx, tag_hash, 32);
    sha256_update(&ctx, tag_hash, 32);
    sha256_update(&ctx, ser, ser_len);
    sha256_final(&ctx, out);
}

// ============================================================================
// GPU Kernel: Full BIP-352 pipeline (1 thread per tweak point)
// ============================================================================
__global__ void bip352_pipeline_kernel(
    const secp256k1::cuda::JacobianPoint* __restrict__ tweak_points,
    const secp256k1::cuda::Scalar* __restrict__ scan_key,
    const secp256k1::cuda::JacobianPoint* __restrict__ spend_point,
    int64_t* __restrict__ prefixes,
    int n)
{
    using namespace secp256k1::cuda;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // 1. k*P -- scalar multiply tweak point by scan key
    JacobianPoint shared;
    scalar_mul_glv(&tweak_points[idx], scan_key, &shared);

    // 2. Serialize to compressed
    uint8_t comp[33];
    point_to_compressed(&shared, comp);

    // 3. Tagged SHA-256 (BIP0352/SharedSecret)
    uint8_t ser[37];
    for (int i = 0; i < 33; i++) ser[i] = comp[i];
    ser[33] = 0; ser[34] = 0; ser[35] = 0; ser[36] = 0;
    uint8_t hash[32];
    bip352_tagged_sha256(ser, 37, hash);

    // 4. k*G -- generator multiply by hash scalar
    Scalar hs;
    scalar_from_bytes(hash, &hs);
    JacobianPoint out;
    scalar_mul_generator_const(&hs, &out);

    // 5. Point addition -- spend + output
    JacobianPoint cand;
    jacobian_add_mixed_unchecked(&out, &BIP352_SPEND_AFFINE, &cand);

    // 6. Serialize + extract prefix
    uint8_t cc[33];
    point_to_compressed(&cand, cc);

    int64_t prefix = 0;
    for (int i = 0; i < 8; i++) prefix = (prefix << 8) | cc[1 + i];
    prefixes[idx] = prefix;
}

// ============================================================================
// GPU per-operation kernels (for detailed breakdown)
// ============================================================================
__global__ void kernel_scalar_mul(
    const secp256k1::cuda::JacobianPoint* __restrict__ pts,
    const secp256k1::cuda::Scalar* __restrict__ key,
    secp256k1::cuda::JacobianPoint* __restrict__ out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    secp256k1::cuda::scalar_mul_glv(&pts[idx], key, &out[idx]);
}

__global__ void kernel_to_compressed(
    const secp256k1::cuda::JacobianPoint* __restrict__ pts,
    uint8_t* __restrict__ out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    secp256k1::cuda::point_to_compressed(&pts[idx], out + idx * 33);
}

__global__ void kernel_tagged_sha256(
    const uint8_t* __restrict__ compressed,
    uint8_t* __restrict__ hashes, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint8_t ser[37];
    for (int i = 0; i < 33; i++) ser[i] = compressed[idx * 33 + i];
    ser[33] = 0; ser[34] = 0; ser[35] = 0; ser[36] = 0;
    bip352_tagged_sha256(ser, 37, hashes + idx * 32);
}

__global__ void kernel_generator_mul(
    const uint8_t* __restrict__ hash_bytes,
    secp256k1::cuda::JacobianPoint* __restrict__ out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    secp256k1::cuda::Scalar hs;
    secp256k1::cuda::scalar_from_bytes(hash_bytes + idx * 32, &hs);
    secp256k1::cuda::scalar_mul_generator_const(&hs, &out[idx]);
}

__global__ void kernel_point_add(
    const secp256k1::cuda::JacobianPoint* __restrict__ spend,
    const secp256k1::cuda::JacobianPoint* __restrict__ output_pts,
    secp256k1::cuda::JacobianPoint* __restrict__ cands, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    secp256k1::cuda::jacobian_add(spend, &output_pts[idx], &cands[idx]);
}

// Forward declarations -- defined at file bottom (nvcc requires file-scope __global__)
__global__ void decompress_points_kernel(
    const uint8_t* __restrict__ comp,
    secp256k1::cuda::JacobianPoint* __restrict__ pts,
    int n);
__global__ void compute_lut_base_points(
    secp256k1::cuda::AffinePoint* bases);
__global__ void gen_lut_build_kernel(
    const secp256k1::cuda::AffinePoint* __restrict__ bases,
    secp256k1::cuda::JacobianPoint* __restrict__ jac_buf,
    int n_entries);
__global__ void gen_lut_to_affine_kernel(
    const secp256k1::cuda::JacobianPoint* __restrict__ jac_buf,
    secp256k1::cuda::AffinePoint* __restrict__ aff_table,
    int total_points);
__global__ void gen_lut_build_affine_kernel(
    const secp256k1::cuda::AffinePoint* __restrict__ bases,
    secp256k1::cuda::AffinePoint* __restrict__ aff_table,
    secp256k1::cuda::FieldElement* __restrict__ h_buf,
    int n_entries);
__global__ void gen_lut_convert_zinv_kernel(
    secp256k1::cuda::AffinePoint* __restrict__ aff_table,
    const secp256k1::cuda::FieldElement* __restrict__ h_buf,
    int n_entries);

// LUT-accelerated pipeline kernel (uses 64 MB precomputed table for k*G)
__global__ void bip352_pipeline_kernel_lut(
    const secp256k1::cuda::JacobianPoint* __restrict__ tweak_points,
    const secp256k1::cuda::Scalar* __restrict__ scan_key,
    const secp256k1::cuda::JacobianPoint* __restrict__ spend_point,
    const secp256k1::cuda::AffinePoint* __restrict__ gen_lut,
    int64_t* __restrict__ prefixes,
    int n);

// LUT-accelerated k*G detail kernel
__global__ void kernel_generator_mul_lut(
    const uint8_t* __restrict__ hash_bytes,
    const secp256k1::cuda::AffinePoint* __restrict__ gen_lut,
    secp256k1::cuda::JacobianPoint* __restrict__ out, int n);

// w=8 generator mul detail kernel (for comparison)
__global__ void kernel_generator_mul_w8(
    const uint8_t* __restrict__ hash_bytes,
    secp256k1::cuda::JacobianPoint* __restrict__ out, int n);

// ============================================================================
// CUDA Timer
// ============================================================================
class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~CudaTimer() { cudaEventDestroy(start_); cudaEventDestroy(stop_); }
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    float stop() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
private:
    cudaEvent_t start_, stop_;
};

// Median with IQR outlier removal
static double median_iqr(std::vector<double>& samples) {
    if (samples.empty()) return 0.0;
    std::sort(samples.begin(), samples.end());
    int n = (int)samples.size();
    if (n < 4) return samples[n / 2];
    double q1 = samples[n / 4], q3 = samples[3 * n / 4];
    double iqr = q3 - q1;
    double lo = q1 - 1.5 * iqr, hi = q3 + 1.5 * iqr;
    std::vector<double> filtered;
    for (double s : samples)
        if (s >= lo && s <= hi) filtered.push_back(s);
    if (filtered.empty()) return samples[n / 2];
    return filtered[filtered.size() / 2];
}

// CPU benchmark harness
static double cpu_bench(int iters, int passes, int warmup,
                        std::function<void(int)> fn) {
    for (int w = 0; w < warmup; ++w) fn(iters);
    std::vector<double> times(passes);
    for (int p = 0; p < passes; ++p) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn(iters);
        auto t1 = std::chrono::high_resolution_clock::now();
        times[p] = std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;
    }
    std::sort(times.begin(), times.end());
    return times[passes / 2];
}

// GPU benchmark harness
template<typename KernelFunc>
static double gpu_bench(int batch, int passes, int warmup, KernelFunc&& kfn) {
    CudaTimer timer;
    for (int w = 0; w < warmup; ++w) { kfn(); CUDA_CHECK(cudaDeviceSynchronize()); }
    std::vector<double> samples;
    for (int p = 0; p < passes; ++p) {
        timer.start();
        kfn();
        float ms = timer.stop();
        samples.push_back((ms * 1e6) / batch);
    }
    return median_iqr(samples);
}

// ============================================================================
// Print helpers
// ============================================================================
static void print_section_header(const char* title) {
    printf("\n  %-40s %12s %12s %8s\n", title, "CPU (ns)", "GPU (ns)", "Ratio");
    printf("  %-40s %12s %12s %8s\n",
           "----------------------------------------",
           "------------", "------------", "--------");
}

static void print_row(const char* name, double cpu_ns, double gpu_ns) {
    double ratio = (gpu_ns > 0) ? cpu_ns / gpu_ns : 0.0;
    const char* winner = (ratio > 1.0) ? "GPU" : "CPU";
    printf("  %-40s %10.1f %12.1f %6.2fx %s\n",
           name, cpu_ns, gpu_ns, (ratio >= 1.0) ? ratio : 1.0 / ratio, winner);
}

// ============================================================================
// Main
// ============================================================================
int main() {
    // ---- GPU device info ----
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("============================================================\n");
    printf("  BIP-352 Silent Payments Pipeline: CPU vs GPU\n");
    printf("============================================================\n");
    printf("  GPU: %s (SM %d.%d, %d SMs, %d MHz)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, prop.clockRate / 1000);
    printf("  N = %d tweak points, %d passes (median)\n", BENCH_N, BENCH_PASSES);
    printf("  Detail breakdown: %d ops per step\n\n", DETAIL_N);

    // ================================================================
    // Phase 1: Generate test data (untimed)
    // ================================================================
    printf("Generating %d deterministic tweak points...\n", BENCH_N);

    std::vector<std::array<uint8_t, 33>> tweak_compressed(BENCH_N);
    {
        uint8_t seed[32];
        const char* tag = "bench_bip352_seed";
        host_sha256(reinterpret_cast<const uint8_t*>(tag), strlen(tag), seed);

        for (int i = 0; i < BENCH_N; i++) {
            uint8_t buf[36];
            memcpy(buf, seed, 32);
            buf[32] = (uint8_t)((i >> 24) & 0xff);
            buf[33] = (uint8_t)((i >> 16) & 0xff);
            buf[34] = (uint8_t)((i >> 8)  & 0xff);
            buf[35] = (uint8_t)( i        & 0xff);

            uint8_t scalar_bytes[32];
            host_sha256(buf, 36, scalar_bytes);

            CpuScalar s = CpuScalar::from_bytes(scalar_bytes);
            CpuPoint p = CpuPoint::generator().scalar_mul(s);
            tweak_compressed[i] = p.to_compressed();
        }
    }
    printf("Done.\n");

    // ================================================================
    // Phase 2: CPU setup (untimed)
    // ================================================================
    CpuScalar scan_scalar = CpuScalar::from_bytes(SCAN_KEY);
    CpuKPlan kplan = CpuKPlan::from_scalar(scan_scalar);
    auto tag_midstate = secp256k1::detail::make_tag_midstate("BIP0352/SharedSecret");
    CpuPoint spend_cpu = CpuPointFromCompressed(SPEND_PUBKEY_COMPRESSED);

    // Pre-parse tweak points (CPU side)
    printf("Pre-parsing tweak points (CPU)...\n");
    std::vector<CpuPoint> cpu_tweaks(BENCH_N);
    for (int i = 0; i < BENCH_N; i++)
        cpu_tweaks[i] = CpuPointFromCompressed(tweak_compressed[i].data());
    printf("Done.\n");

    // ================================================================
    // Phase 3: GPU setup -- upload data
    // ================================================================
    printf("Uploading data to GPU...\n");

    // Convert tweak points to GPU JacobianPoints
    std::vector<secp256k1::cuda::JacobianPoint> h_tweaks(BENCH_N);
    for (int i = 0; i < BENCH_N; i++) {
        // Decompress on GPU format: use scalar_from_bytes to parse, then point_from_compressed
        // Actually, parse compressed on CPU side and convert coords
        auto& ct = tweak_compressed[i];
        // Parse x coordinate
        for (int k = 0; k < 4; k++) {
            uint64_t limb = 0;
            int base = (3 - k) * 8;
            for (int j = 0; j < 8; j++) limb = (limb << 8) | ct[1 + base + j];
            h_tweaks[i].x.limbs[k] = limb;
        }
        // y^2 = x^3 + 7 -- We need to decompress properly.
        // Easiest: run a GPU kernel to decompress, or use CPU coords.
        // Use CPU point coords: extract Jacobian (x, y, z=1)
        auto comp = cpu_tweaks[i].to_compressed();
        // Actually, let's just set affine coords with z=1 from the CPU Point.
        // We need to get affine x,y from the CPU Point. The CPU Point stores Jacobian.
        // Call to_compressed and then decompress to get affine, or use internal access.
        // Simplest: decompress from the 33-byte compressed format on GPU.
        // Let's batch-decompress on GPU with a helper kernel.
        (void)comp;
    }

    // Use a GPU decompression kernel instead
    // Upload compressed pubkeys, run decompression kernel
    uint8_t* d_compressed;
    CUDA_CHECK(cudaMalloc(&d_compressed, BENCH_N * 33));
    {
        std::vector<uint8_t> flat(BENCH_N * 33);
        for (int i = 0; i < BENCH_N; i++)
            memcpy(flat.data() + i * 33, tweak_compressed[i].data(), 33);
        CUDA_CHECK(cudaMemcpy(d_compressed, flat.data(), BENCH_N * 33, cudaMemcpyHostToDevice));
    }

    // GPU decompression kernel
    secp256k1::cuda::JacobianPoint* d_tweaks;
    CUDA_CHECK(cudaMalloc(&d_tweaks, BENCH_N * sizeof(secp256k1::cuda::JacobianPoint)));

    {
        int blocks = (BENCH_N + GPU_TPB - 1) / GPU_TPB;
        decompress_points_kernel<<<blocks, GPU_TPB>>>(d_compressed, d_tweaks, BENCH_N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaFree(d_compressed));

    // Upload scan key
    secp256k1::cuda::Scalar h_scan_key;
    secp256k1::cuda::Scalar* d_scan_key;
    {
        for (int i = 0; i < 4; i++) {
            uint64_t limb = 0;
            int base = (3 - i) * 8;
            for (int j = 0; j < 8; j++) limb = (limb << 8) | SCAN_KEY[base + j];
            h_scan_key.limbs[i] = limb;
        }
    }
    CUDA_CHECK(cudaMalloc(&d_scan_key, sizeof(secp256k1::cuda::Scalar)));
    CUDA_CHECK(cudaMemcpy(d_scan_key, &h_scan_key, sizeof(secp256k1::cuda::Scalar), cudaMemcpyHostToDevice));

    // Upload spend pubkey
    secp256k1::cuda::JacobianPoint h_spend;
    secp256k1::cuda::JacobianPoint* d_spend;
    // Decompress spend pubkey on GPU
    {
        uint8_t* d_spend_comp;
        CUDA_CHECK(cudaMalloc(&d_spend_comp, 33));
        CUDA_CHECK(cudaMemcpy(d_spend_comp, SPEND_PUBKEY_COMPRESSED, 33, cudaMemcpyHostToDevice));
        secp256k1::cuda::JacobianPoint* d_spend_tmp;
        CUDA_CHECK(cudaMalloc(&d_spend_tmp, sizeof(secp256k1::cuda::JacobianPoint)));
        decompress_points_kernel<<<1, 1>>>(d_spend_comp, d_spend_tmp, 1);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_spend, d_spend_tmp, sizeof(secp256k1::cuda::JacobianPoint), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_spend_comp));
        CUDA_CHECK(cudaFree(d_spend_tmp));
    }
    CUDA_CHECK(cudaMalloc(&d_spend, sizeof(secp256k1::cuda::JacobianPoint)));
    CUDA_CHECK(cudaMemcpy(d_spend, &h_spend, sizeof(secp256k1::cuda::JacobianPoint), cudaMemcpyHostToDevice));

    // Allocate output
    int64_t* d_prefixes;
    CUDA_CHECK(cudaMalloc(&d_prefixes, BENCH_N * sizeof(int64_t)));

    printf("Done.\n");

    // ================================================================
    // Phase 3.5: Build Generator LUT (16 x 65536 = 64 MB)
    // ================================================================
    printf("Building generator LUT (16 windows x 65536 entries = 64 MB)...\n");
    CudaTimer lut_build_timer;

    secp256k1::cuda::AffinePoint* d_gen_lut = nullptr;
    {
        // Step 1: Compute 16 base points on GPU
        secp256k1::cuda::AffinePoint* d_bases;
        CUDA_CHECK(cudaMalloc(&d_bases, 16 * sizeof(secp256k1::cuda::AffinePoint)));
        compute_lut_base_points<<<1, 1>>>(d_bases);
        CUDA_CHECK(cudaDeviceSynchronize());

        const int LUT_ENTRIES = 65536;
        const int LUT_TOTAL = 16 * LUT_ENTRIES;

        // Step 2+3 fused: H-based serial build + batch-inv affine conversion
        // H buffer: 16 * 65536 * sizeof(FieldElement) = 32 MB (vs 132 MB for old Jacobian buffer)
        CUDA_CHECK(cudaMalloc(&d_gen_lut, (size_t)LUT_TOTAL * sizeof(secp256k1::cuda::AffinePoint)));
        secp256k1::cuda::FieldElement* d_h_buf;
        CUDA_CHECK(cudaMalloc(&d_h_buf, (size_t)LUT_TOTAL * sizeof(secp256k1::cuda::FieldElement)));

        lut_build_timer.start();
        gen_lut_build_affine_kernel<<<16, 1>>>(d_bases, d_gen_lut, d_h_buf, LUT_ENTRIES);
        CUDA_CHECK(cudaDeviceSynchronize());
        // Parallel affine conversion using z_inv values stored in h_buf
        int total_conv = 16 * (LUT_ENTRIES - 2);
        int conv_blk = (total_conv + 255) / 256;
        gen_lut_convert_zinv_kernel<<<conv_blk, 256>>>(d_gen_lut, d_h_buf, LUT_ENTRIES);
        float lut_ms = lut_build_timer.stop();

        printf("  LUT built in %.1f ms (%.1f MB, %d points, H-based split)\n",
               lut_ms, LUT_TOTAL * sizeof(secp256k1::cuda::AffinePoint) / 1e6, LUT_TOTAL);

        CUDA_CHECK(cudaFree(d_h_buf));
        CUDA_CHECK(cudaFree(d_bases));
    }
    printf("Done.\n\n");

    // ================================================================
    // Phase 4: Full Pipeline Benchmark -- CPU
    // ================================================================
    printf("=== Full Pipeline Benchmark ===\n");
    printf("\n--- CPU (UltrafastSecp256k1, KPlan) ---\n");

    auto cpu_pipeline = [&](int iters) {
        int64_t last_prefix = 0;
        for (int i = 0; i < iters; i++) {
            CpuPoint shared = cpu_tweaks[i % BENCH_N].scalar_mul_with_plan(kplan);
            auto comp = shared.to_compressed();
            uint8_t ser[37];
            memcpy(ser, comp.data(), 33);
            memset(ser + 33, 0, 4);
            auto hash = secp256k1::detail::cached_tagged_hash(tag_midstate, ser, 37);
            CpuScalar hs = CpuScalar::from_bytes(hash.data());
            CpuPoint out = CpuPoint::generator().scalar_mul(hs);
            CpuPoint cand = spend_cpu.add(out);
            auto cc = cand.to_compressed();
            last_prefix = extract_upper_64(cc.data() + 1);
            DoNotOptimize(last_prefix);
        }
    };

    // Warmup + measure
    std::vector<double> cpu_times(BENCH_PASSES);
    for (int w = 0; w < BENCH_WARMUP; ++w) cpu_pipeline(BENCH_N);
    int64_t cpu_validation = 0;
    for (int p = 0; p < BENCH_PASSES; ++p) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cpu_pipeline(BENCH_N);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        cpu_times[p] = ms;
        printf("  pass %2d: %8.1f ms\n", p + 1, ms);
    }
    // Run once more to get validation prefix
    {
        CpuPoint shared = cpu_tweaks[(BENCH_N-1) % BENCH_N].scalar_mul_with_plan(kplan);
        auto comp = shared.to_compressed();
        uint8_t ser[37]; memcpy(ser, comp.data(), 33); memset(ser+33,0,4);
        auto hash = secp256k1::detail::cached_tagged_hash(tag_midstate, ser, 37);
        CpuScalar hs = CpuScalar::from_bytes(hash.data());
        CpuPoint out = CpuPoint::generator().scalar_mul(hs);
        CpuPoint cand = spend_cpu.add(out);
        auto cc = cand.to_compressed();
        cpu_validation = extract_upper_64(cc.data() + 1);
    }
    std::sort(cpu_times.begin(), cpu_times.end());
    double cpu_median = cpu_times[BENCH_PASSES / 2];
    double cpu_ns_op = cpu_median * 1e6 / BENCH_N;

    printf("\n  CPU: %.1f ms / %d ops = %.1f ns/op (%.1f us/op)\n",
           cpu_median, BENCH_N, cpu_ns_op, cpu_ns_op / 1000.0);
    printf("  validation prefix: 0x%016lx\n", (unsigned long)cpu_validation);

    // ================================================================
    // Phase 5: Full Pipeline Benchmark -- GPU
    // ================================================================
    printf("\n--- GPU (CUDA, GLV) ---\n");

    CudaTimer timer;
    int blocks = (BENCH_N + GPU_TPB - 1) / GPU_TPB;

    // Warmup
    for (int w = 0; w < BENCH_WARMUP; ++w) {
        bip352_pipeline_kernel<<<blocks, GPU_TPB>>>(d_tweaks, d_scan_key, d_spend, d_prefixes, BENCH_N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<double> gpu_times(BENCH_PASSES);
    for (int p = 0; p < BENCH_PASSES; ++p) {
        timer.start();
        bip352_pipeline_kernel<<<blocks, GPU_TPB>>>(d_tweaks, d_scan_key, d_spend, d_prefixes, BENCH_N);
        float ms = timer.stop();
        gpu_times[p] = ms;
        printf("  pass %2d: %8.3f ms\n", p + 1, ms);
    }

    // Get validation prefix
    std::vector<int64_t> h_prefixes(BENCH_N);
    CUDA_CHECK(cudaMemcpy(h_prefixes.data(), d_prefixes, BENCH_N * sizeof(int64_t), cudaMemcpyDeviceToHost));
    int64_t gpu_validation = h_prefixes[BENCH_N - 1];

    std::sort(gpu_times.begin(), gpu_times.end());
    double gpu_median = gpu_times[BENCH_PASSES / 2];
    double gpu_ns_op = gpu_median * 1e6 / BENCH_N;

    printf("\n  GPU: %.3f ms / %d ops = %.1f ns/op (%.1f us/op)\n",
           gpu_median, BENCH_N, gpu_ns_op, gpu_ns_op / 1000.0);
    printf("  validation prefix: 0x%016lx\n", (unsigned long)gpu_validation);

    // ================================================================
    // Phase 5.5: Full Pipeline Benchmark -- GPU + LUT
    // ================================================================
    printf("\n--- GPU + LUT (16x64K precomputed table for k*G) ---\n");

    // Warmup
    for (int w = 0; w < BENCH_WARMUP; ++w) {
        bip352_pipeline_kernel_lut<<<blocks, GPU_TPB>>>(
            d_tweaks, d_scan_key, d_spend, d_gen_lut, d_prefixes, BENCH_N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<double> gpu_lut_times(BENCH_PASSES);
    for (int p = 0; p < BENCH_PASSES; ++p) {
        timer.start();
        bip352_pipeline_kernel_lut<<<blocks, GPU_TPB>>>(
            d_tweaks, d_scan_key, d_spend, d_gen_lut, d_prefixes, BENCH_N);
        float ms = timer.stop();
        gpu_lut_times[p] = ms;
        printf("  pass %2d: %8.3f ms\n", p + 1, ms);
    }

    // Get validation prefix from LUT pipeline
    CUDA_CHECK(cudaMemcpy(h_prefixes.data(), d_prefixes, BENCH_N * sizeof(int64_t), cudaMemcpyDeviceToHost));
    int64_t gpu_lut_validation = h_prefixes[BENCH_N - 1];

    std::sort(gpu_lut_times.begin(), gpu_lut_times.end());
    double gpu_lut_median = gpu_lut_times[BENCH_PASSES / 2];
    double gpu_lut_ns_op = gpu_lut_median * 1e6 / BENCH_N;

    printf("\n  GPU+LUT: %.3f ms / %d ops = %.1f ns/op (%.1f us/op)\n",
           gpu_lut_median, BENCH_N, gpu_lut_ns_op, gpu_lut_ns_op / 1000.0);
    printf("  validation prefix: 0x%016lx\n", (unsigned long)gpu_lut_validation);

    // ================================================================
    // Phase 6: Comparison summary
    // ================================================================
    printf("\n=== Full Pipeline Comparison ===\n");
    double pipeline_ratio = cpu_ns_op / gpu_ns_op;
    double lut_ratio = cpu_ns_op / gpu_lut_ns_op;
    double lut_vs_gpu = gpu_ns_op / gpu_lut_ns_op;
    printf("  CPU:         %10.1f ns/op\n", cpu_ns_op);
    printf("  GPU (w=4):   %10.1f ns/op  (%.2fx vs CPU)\n", gpu_ns_op, pipeline_ratio);
    printf("  GPU+LUT:     %10.1f ns/op  (%.2fx vs CPU, %.2fx vs GPU w=4)\n",
           gpu_lut_ns_op, lut_ratio, lut_vs_gpu);

    bool prefixes_match = (cpu_validation == gpu_validation) && (cpu_validation == gpu_lut_validation);
    printf("  Validation: %s\n",
           prefixes_match ? "[OK] ALL MATCH" : "[FAIL] MISMATCH");
    printf("    CPU=0x%016lx  GPU=0x%016lx  LUT=0x%016lx\n",
           (unsigned long)cpu_validation, (unsigned long)gpu_validation,
           (unsigned long)gpu_lut_validation);

    // ================================================================
    // Phase 7: Per-operation breakdown
    // ================================================================
    printf("\n=== Per-Operation Breakdown ===\n");
    printf("  (%d ops per step, %d passes, median)\n", DETAIL_N, BENCH_PASSES);

    // Allocate GPU work buffers for detail
    secp256k1::cuda::JacobianPoint* d_shared;
    uint8_t* d_comp_out;
    uint8_t* d_hashes;
    secp256k1::cuda::JacobianPoint* d_output_pts;
    secp256k1::cuda::JacobianPoint* d_candidates;

    CUDA_CHECK(cudaMalloc(&d_shared,     DETAIL_N * sizeof(secp256k1::cuda::JacobianPoint)));
    CUDA_CHECK(cudaMalloc(&d_comp_out,   DETAIL_N * 33));
    CUDA_CHECK(cudaMalloc(&d_hashes,     DETAIL_N * 32));
    CUDA_CHECK(cudaMalloc(&d_output_pts, DETAIL_N * sizeof(secp256k1::cuda::JacobianPoint)));
    CUDA_CHECK(cudaMalloc(&d_candidates, DETAIL_N * sizeof(secp256k1::cuda::JacobianPoint)));

    int dblocks = (DETAIL_N + GPU_TPB - 1) / GPU_TPB;

    // Pre-compute intermediate GPU results for isolated step timing
    kernel_scalar_mul<<<dblocks, GPU_TPB>>>(d_tweaks, d_scan_key, d_shared, DETAIL_N);
    kernel_to_compressed<<<dblocks, GPU_TPB>>>(d_shared, d_comp_out, DETAIL_N);
    kernel_tagged_sha256<<<dblocks, GPU_TPB>>>(d_comp_out, d_hashes, DETAIL_N);
    kernel_generator_mul<<<dblocks, GPU_TPB>>>(d_hashes, d_output_pts, DETAIL_N);
    kernel_point_add<<<dblocks, GPU_TPB>>>(d_spend, d_output_pts, d_candidates, DETAIL_N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // CPU pre-compute intermediates
    std::vector<CpuPoint> cpu_shared(DETAIL_N);
    std::vector<std::array<uint8_t, 33>> cpu_comp(DETAIL_N);
    std::vector<std::array<uint8_t, 32>> cpu_hashes(DETAIL_N);
    std::vector<CpuScalar> cpu_hash_scalars(DETAIL_N);
    std::vector<CpuPoint> cpu_output_pts(DETAIL_N);
    std::vector<CpuPoint> cpu_candidates(DETAIL_N);

    for (int i = 0; i < DETAIL_N; i++) {
        cpu_shared[i] = cpu_tweaks[i].scalar_mul_with_plan(kplan);
        cpu_comp[i] = cpu_shared[i].to_compressed();
        uint8_t ser[37]; memcpy(ser, cpu_comp[i].data(), 33); memset(ser+33, 0, 4);
        cpu_hashes[i] = secp256k1::detail::cached_tagged_hash(tag_midstate, ser, 37);
        cpu_hash_scalars[i] = CpuScalar::from_bytes(cpu_hashes[i].data());
        cpu_output_pts[i] = CpuPoint::generator().scalar_mul(cpu_hash_scalars[i]);
        cpu_candidates[i] = spend_cpu.add(cpu_output_pts[i]);
    }

    print_section_header("Operation");

    // Step 1: k*P
    double cpu_kP = cpu_bench(DETAIL_N, BENCH_PASSES, BENCH_WARMUP, [&](int n) {
        for (int i = 0; i < n; i++) {
            CpuPoint r = cpu_tweaks[i % DETAIL_N].scalar_mul_with_plan(kplan);
            DoNotOptimize(r);
        }
    });
    double gpu_kP = gpu_bench(DETAIL_N, BENCH_PASSES, BENCH_WARMUP, [&]() {
        kernel_scalar_mul<<<dblocks, GPU_TPB>>>(d_tweaks, d_scan_key, d_shared, DETAIL_N);
    });
    print_row("k*P (scalar_mul)", cpu_kP, gpu_kP);

    // Step 2: Serialize compressed
    double cpu_ser1 = cpu_bench(DETAIL_N, BENCH_PASSES, BENCH_WARMUP, [&](int n) {
        for (int i = 0; i < n; i++) {
            auto c = cpu_shared[i % DETAIL_N].to_compressed();
            DoNotOptimize(c);
        }
    });
    double gpu_ser1 = gpu_bench(DETAIL_N, BENCH_PASSES, BENCH_WARMUP, [&]() {
        kernel_to_compressed<<<dblocks, GPU_TPB>>>(d_shared, d_comp_out, DETAIL_N);
    });
    print_row("to_compressed (1st)", cpu_ser1, gpu_ser1);

    // Step 3: Tagged SHA-256
    double cpu_sha = cpu_bench(DETAIL_N, BENCH_PASSES, BENCH_WARMUP, [&](int n) {
        for (int i = 0; i < n; i++) {
            uint8_t ser[37]; memcpy(ser, cpu_comp[i%DETAIL_N].data(), 33); memset(ser+33,0,4);
            auto h = secp256k1::detail::cached_tagged_hash(tag_midstate, ser, 37);
            DoNotOptimize(h);
        }
    });
    double gpu_sha = gpu_bench(DETAIL_N, BENCH_PASSES, BENCH_WARMUP, [&]() {
        kernel_tagged_sha256<<<dblocks, GPU_TPB>>>(d_comp_out, d_hashes, DETAIL_N);
    });
    print_row("tagged SHA-256 (cached)", cpu_sha, gpu_sha);

    // Step 4: k*G (generator mul) -- compare w=4, w=8, and LUT
    double cpu_kG = cpu_bench(DETAIL_N, BENCH_PASSES, BENCH_WARMUP, [&](int n) {
        for (int i = 0; i < n; i++) {
            CpuPoint r = CpuPoint::generator().scalar_mul(cpu_hash_scalars[i%DETAIL_N]);
            DoNotOptimize(r);
        }
    });
    double gpu_kG = gpu_bench(DETAIL_N, BENCH_PASSES, BENCH_WARMUP, [&]() {
        kernel_generator_mul<<<dblocks, GPU_TPB>>>(d_hashes, d_output_pts, DETAIL_N);
    });
    print_row("k*G (w=4, 16-pt const)", cpu_kG, gpu_kG);

    double gpu_kG_w8 = gpu_bench(DETAIL_N, BENCH_PASSES, BENCH_WARMUP, [&]() {
        kernel_generator_mul_w8<<<dblocks, GPU_TPB>>>(d_hashes, d_output_pts, DETAIL_N);
    });
    print_row("k*G (w=8, 256-pt const)", cpu_kG, gpu_kG_w8);

    double gpu_kG_lut = gpu_bench(DETAIL_N, BENCH_PASSES, BENCH_WARMUP, [&]() {
        kernel_generator_mul_lut<<<dblocks, GPU_TPB>>>(d_hashes, d_gen_lut, d_output_pts, DETAIL_N);
    });
    print_row("k*G (LUT, 1M-pt global)", cpu_kG, gpu_kG_lut);

    // Step 5: Point addition
    double cpu_add = cpu_bench(DETAIL_N, BENCH_PASSES, BENCH_WARMUP, [&](int n) {
        for (int i = 0; i < n; i++) {
            CpuPoint r = spend_cpu.add(cpu_output_pts[i%DETAIL_N]);
            DoNotOptimize(r);
        }
    });
    double gpu_add = gpu_bench(DETAIL_N, BENCH_PASSES, BENCH_WARMUP, [&]() {
        kernel_point_add<<<dblocks, GPU_TPB>>>(d_spend, d_output_pts, d_candidates, DETAIL_N);
    });
    print_row("point_add", cpu_add, gpu_add);

    // Step 6: Serialize compressed (2nd)
    double cpu_ser2 = cpu_bench(DETAIL_N, BENCH_PASSES, BENCH_WARMUP, [&](int n) {
        for (int i = 0; i < n; i++) {
            auto c = cpu_candidates[i%DETAIL_N].to_compressed();
            DoNotOptimize(c);
        }
    });
    double gpu_ser2 = gpu_bench(DETAIL_N, BENCH_PASSES, BENCH_WARMUP, [&]() {
        kernel_to_compressed<<<dblocks, GPU_TPB>>>(d_candidates, d_comp_out, DETAIL_N);
    });
    print_row("to_compressed (2nd)", cpu_ser2, gpu_ser2);

    // ================================================================
    // Phase 8: Summary table
    // ================================================================
    double cpu_total = cpu_kP + cpu_ser1 + cpu_sha + cpu_kG + cpu_add + cpu_ser2;
    double gpu_total = gpu_kP + gpu_ser1 + gpu_sha + gpu_kG + gpu_add + gpu_ser2;
    double gpu_lut_total = gpu_kP + gpu_ser1 + gpu_sha + gpu_kG_lut + gpu_add + gpu_ser2;

    printf("\n  %-40s %10.1f %12.1f %6.2fx %s\n",
           "TOTAL (sum of steps)",
           cpu_total, gpu_total,
           (cpu_total / gpu_total >= 1.0) ? cpu_total / gpu_total : gpu_total / cpu_total,
           (cpu_total / gpu_total >= 1.0) ? "GPU" : "CPU");

    printf("\n  %-40s %10.1f %12.1f %6.2fx %s\n",
           "TOTAL w/ LUT (sum of steps)",
           cpu_total, gpu_lut_total,
           (cpu_total / gpu_lut_total >= 1.0) ? cpu_total / gpu_lut_total : gpu_lut_total / cpu_total,
           (cpu_total / gpu_lut_total >= 1.0) ? "GPU" : "CPU");

    printf("\n  %-40s %10.1f %12.1f %6.2fx %s\n",
           "Full pipeline (measured, w=4)",
           cpu_ns_op, gpu_ns_op,
           (pipeline_ratio >= 1.0) ? pipeline_ratio : 1.0 / pipeline_ratio,
           (pipeline_ratio >= 1.0) ? "GPU" : "CPU");

    printf("  %-40s %10.1f %12.1f %6.2fx %s\n",
           "Full pipeline (measured, LUT)",
           cpu_ns_op, gpu_lut_ns_op,
           (lut_ratio >= 1.0) ? lut_ratio : 1.0 / lut_ratio,
           (lut_ratio >= 1.0) ? "GPU" : "CPU");

    printf("\n  k*G speedup: w=4 -> w=8: %.2fx, w=4 -> LUT: %.2fx\n",
           gpu_kG / gpu_kG_w8, gpu_kG / gpu_kG_lut);

    // ================================================================
    // Phase 9: Percentage breakdown
    // ================================================================
    printf("\n=== Time Breakdown (percentage of full pipeline) ===\n");
    auto pct = [](double part, double total) { return (total > 0) ? 100.0 * part / total : 0.0; };
    printf("  %-30s %8s %8s\n", "Step", "CPU %%", "GPU %%");
    printf("  %-30s %8s %8s\n", "------------------------------", "--------", "--------");
    printf("  %-30s %7.1f%% %7.1f%%\n", "k*P", pct(cpu_kP, cpu_total), pct(gpu_kP, gpu_total));
    printf("  %-30s %7.1f%% %7.1f%%\n", "Serialize (1st)", pct(cpu_ser1, cpu_total), pct(gpu_ser1, gpu_total));
    printf("  %-30s %7.1f%% %7.1f%%\n", "Tagged SHA-256", pct(cpu_sha, cpu_total), pct(gpu_sha, gpu_total));
    printf("  %-30s %7.1f%% %7.1f%%\n", "k*G", pct(cpu_kG, cpu_total), pct(gpu_kG, gpu_total));
    printf("  %-30s %7.1f%% %7.1f%%\n", "Point add", pct(cpu_add, cpu_total), pct(gpu_add, gpu_total));
    printf("  %-30s %7.1f%% %7.1f%%\n", "Serialize (2nd)", pct(cpu_ser2, cpu_total), pct(gpu_ser2, gpu_total));

    printf("\n============================================================\n");
    printf("  Benchmark complete.\n");
    printf("============================================================\n");

    // Cleanup
    CUDA_CHECK(cudaFree(d_tweaks));
    CUDA_CHECK(cudaFree(d_scan_key));
    CUDA_CHECK(cudaFree(d_spend));
    CUDA_CHECK(cudaFree(d_prefixes));
    CUDA_CHECK(cudaFree(d_shared));
    CUDA_CHECK(cudaFree(d_comp_out));
    CUDA_CHECK(cudaFree(d_hashes));
    CUDA_CHECK(cudaFree(d_output_pts));
    CUDA_CHECK(cudaFree(d_candidates));
    CUDA_CHECK(cudaFree(d_gen_lut));

    return 0;
}

// ============================================================================
// Named kernel for point decompression (nvcc requires file-scope kernels)
// ============================================================================
__device__ inline void scalar_mul_fixed_scan(
{
    using namespace secp256k1::cuda;

    AffinePoint base;
    if (p->z.limbs[0] == 1 && p->z.limbs[1] == 0 && p->z.limbs[2] == 0 && p->z.limbs[3] == 0) {
        base.x = p->x;
        base.y = p->y;
    } else {
        FieldElement z_inv, z_inv2, z_inv3;
        field_inv(&p->z, &z_inv);
        field_sqr(&z_inv, &z_inv2);
        field_mul(&z_inv2, &z_inv, &z_inv3);
        field_mul(&p->x, &z_inv2, &base.x);
        field_mul(&p->y, &z_inv3, &base.y);
    }

    if (BIP352_SCANKEY_WNAF.k1_neg) {
        field_negate(&base.y, &base.y);
    }

    constexpr int TABLE_SIZE = (1 << (SCAN_WNAF_W - 2));
    AffinePoint tbl_P[TABLE_SIZE];
    FieldElement globalz;
    build_wnaf_table_zr(&base, tbl_P, TABLE_SIZE, &globalz);

    AffinePoint tbl_phiP[TABLE_SIZE];
    derive_endo_table(tbl_P, tbl_phiP, TABLE_SIZE, BIP352_SCANKEY_WNAF.flip_phi != 0);

    r->infinity = true;
    field_set_zero(&r->x);
    field_set_one(&r->y);
    field_set_zero(&r->z);

    #pragma unroll 1
    for (int i = SCAN_WNAF_MAXLEN - 1; i >= 0; --i) {
        if (!r->infinity) {
            jacobian_double_unchecked(r, r);
        }

        int8_t d1 = BIP352_SCANKEY_WNAF.wnaf1[i];
        if (d1 != 0) {
            int table_idx = ((d1 > 0) ? d1 : -d1);
            table_idx = (table_idx - 1) >> 1;
            AffinePoint pt = tbl_P[table_idx];
            if (d1 < 0) field_negate(&pt.y, &pt.y);

            if (r->infinity) {
                r->x = pt.x;
                r->y = pt.y;
                field_set_one(&r->z);
                r->infinity = false;
            } else {
                jacobian_add_mixed_unchecked(r, &pt, r);
            }
        }

        int8_t d2 = BIP352_SCANKEY_WNAF.wnaf2[i];
        if (d2 != 0) {
            int table_idx = ((d2 > 0) ? d2 : -d2);
            table_idx = (table_idx - 1) >> 1;
            AffinePoint pt = tbl_phiP[table_idx];
            if (d2 < 0) field_negate(&pt.y, &pt.y);

            if (r->infinity) {
                r->x = pt.x;
                r->y = pt.y;
                field_set_one(&r->z);
                r->infinity = false;
            } else {
                jacobian_add_mixed_unchecked(r, &pt, r);
            }
        }
    }

    if (!r->infinity) {
        FieldElement tmp;
        field_mul(&r->z, &globalz, &tmp);
        r->z = tmp;
    }
}

__device__ __forceinline__ int64_t point_prefix64(
    const secp256k1::cuda::JacobianPoint* p)
{
    using namespace secp256k1::cuda;

    if (p->infinity) return 0;

    FieldElement ax;
    if (p->z.limbs[0] == 1 && p->z.limbs[1] == 0 && p->z.limbs[2] == 0 && p->z.limbs[3] == 0) {
        ax = p->x;
    } else {
        FieldElement z_inv, z_inv2;
        field_inv(&p->z, &z_inv);
        field_sqr(&z_inv, &z_inv2);
        field_mul(&p->x, &z_inv2, &ax);
    }

    uint8_t x_bytes[32];
    field_to_bytes(&ax, x_bytes);

    int64_t prefix = 0;
    for (int i = 0; i < 8; i++) prefix = (prefix << 8) | x_bytes[i];
    return prefix;
}

__global__ void decompress_points_kernel(
    const uint8_t* __restrict__ comp,
    secp256k1::cuda::JacobianPoint* __restrict__ pts,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    secp256k1::cuda::point_from_compressed(comp + idx * 33, &pts[idx]);
}

// ============================================================================
// Generator LUT build kernels (file-scope for nvcc)
// ============================================================================

// Single-thread kernel: compute B_i = 2^(16*i) * G for i=0..15
__global__ void compute_lut_base_points(
    secp256k1::cuda::AffinePoint* bases)
{
    using namespace secp256k1::cuda;

    // bases[0] = G
    bases[0] = GENERATOR_TABLE_W8[1];

    JacobianPoint p;
    p.x = GENERATOR_TABLE_W8[1].x;
    p.y = GENERATOR_TABLE_W8[1].y;
    field_set_one(&p.z);
    p.infinity = false;

    for (int i = 1; i < 16; i++) {
        // Double 16 times: p = 2^16 * previous
        for (int d = 0; d < 16; d++)
            jacobian_double(&p, &p);

        // Convert to affine
        FieldElement z_inv, z_inv2, z_inv3;
        field_inv(&p.z, &z_inv);
        field_sqr(&z_inv, &z_inv2);
        field_mul(&z_inv2, &z_inv, &z_inv3);
        field_mul(&p.x, &z_inv2, &bases[i].x);
        field_mul(&p.y, &z_inv3, &bases[i].y);

        // Reset Jacobian to affine (z=1) for next chain
        p.x = bases[i].x;
        p.y = bases[i].y;
        field_set_one(&p.z);
    }
}

// One block per slice: build table[slice][0..65535] = sequential additions
__global__ void gen_lut_build_kernel(
    const secp256k1::cuda::AffinePoint* __restrict__ bases,
    secp256k1::cuda::JacobianPoint* __restrict__ jac_buf,
    int n_entries)
{
    using namespace secp256k1::cuda;
    int slice = blockIdx.x;
    if (slice >= 16) return;

    int offset = slice * n_entries;

    // [0] = identity
    jac_buf[offset].infinity = true;
    field_set_zero(&jac_buf[offset].x);
    field_set_one(&jac_buf[offset].y);
    field_set_zero(&jac_buf[offset].z);

    // [1] = base point
    jac_buf[offset + 1].x = bases[slice].x;
    jac_buf[offset + 1].y = bases[slice].y;
    field_set_one(&jac_buf[offset + 1].z);
    jac_buf[offset + 1].infinity = false;

    // [j] = [j-1] + base (mixed Jacobian + affine addition)
    for (int j = 2; j < n_entries; j++) {
        jacobian_add_mixed(&jac_buf[offset + j - 1], &bases[slice],
                           &jac_buf[offset + j]);
    }
}

// Massively parallel Jacobian -> Affine conversion (1 thread per point)
__global__ void gen_lut_to_affine_kernel(
    const secp256k1::cuda::JacobianPoint* __restrict__ jac_buf,
    secp256k1::cuda::AffinePoint* __restrict__ aff_table,
    int total_points)
{
    using namespace secp256k1::cuda;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_points) return;

    if (jac_buf[idx].infinity) {
        field_set_zero(&aff_table[idx].x);
        field_set_zero(&aff_table[idx].y);
        return;
    }

    FieldElement z_inv, z_inv2, z_inv3;
    field_inv(&jac_buf[idx].z, &z_inv);
    field_sqr(&z_inv, &z_inv2);
    field_mul(&z_inv2, &z_inv, &z_inv3);
    field_mul(&jac_buf[idx].x, &z_inv2, &aff_table[idx].x);
    field_mul(&jac_buf[idx].y, &z_inv3, &aff_table[idx].y);
}

// ============================================================================
// Fused LUT Build + Affine Conversion (H-based serial inversion)
// ============================================================================
// Combines build and affine conversion into one kernel:
//   - Forward pass: sequential jacobian_add_mixed_h, stores H values + Jacobian X,Y
//   - Single field_inv of final Z product (1 per slice instead of 65536)
//   - Backward sweep: reconstructs Z^{-1} for each point and converts to affine
// Eliminates the 132 MB Jacobian temp buffer (only 32 MB H buffer needed).
// Reduces field inversions from 1,048,576 to 16.
__global__ void gen_lut_build_affine_kernel(
    const secp256k1::cuda::AffinePoint* __restrict__ bases,
    secp256k1::cuda::AffinePoint* __restrict__ aff_table,
    secp256k1::cuda::FieldElement* __restrict__ h_buf,
    int n_entries)
{
    using namespace secp256k1::cuda;
    int slice = blockIdx.x;
    if (slice >= 16) return;

    int offset = slice * n_entries;
    FieldElement* h = h_buf + (size_t)slice * n_entries;

    // [0] = identity
    field_set_zero(&aff_table[offset].x);
    field_set_zero(&aff_table[offset].y);

    // [1] = base point (already affine)
    aff_table[offset + 1] = bases[slice];

    // Forward pass: build chain P[j] = P[j-1] + base
    // Store H values and Jacobian X,Y in the output buffer (temporary)
    JacobianPoint acc;
    acc.x = bases[slice].x;
    acc.y = bases[slice].y;
    field_set_one(&acc.z);
    acc.infinity = false;

    for (int j = 2; j < n_entries; j++) {
        FieldElement h_val;
        jacobian_add_mixed_h(&acc, &bases[slice], &acc, h_val);
        h[j - 2] = h_val;
        // Store Jacobian X,Y temporarily in affine output buffer
        aff_table[offset + j].x = acc.x;
        aff_table[offset + j].y = acc.y;
    }

    // Single inversion of final Z (= product of all H values * doubling-Z)
    FieldElement z_inv;
    field_inv(&acc.z, &z_inv);

    // Serial z_inv scan: overwrite h_buf with z_inv[j] at h[j-2]
    // Recurrence: Z_j = Z_{j-1} * H_j  =>  Z_{j-1}^{-1} = H_j * Z_j^{-1}
    for (int j = n_entries - 1; j >= 2; --j) {
        FieldElement h_save;
        if (j > 2) h_save = h[j - 2];  // read H_j before overwrite
        h[j - 2] = z_inv;              // store z_inv[j]
        if (j > 2) {
            FieldElement tmp;
            field_mul(&h_save, &z_inv, &tmp);
            z_inv = tmp;
        }
    }
    // h_buf now holds z_inv[j] at position h[j-2] for j=2..N-1
    // Affine conversion done in parallel by gen_lut_convert_zinv_kernel
}

// Parallel affine conversion using precomputed z_inv values from h_buf.
// Each thread converts one point from Jacobian X,Y (stored in aff_table) to affine.
__global__ void gen_lut_convert_zinv_kernel(
    secp256k1::cuda::AffinePoint* __restrict__ aff_table,
    const secp256k1::cuda::FieldElement* __restrict__ h_buf,
    int n_entries)
{
    using namespace secp256k1::cuda;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int per_slice = n_entries - 2;  // convertible points per slice (j=2..N-1)
    int total = 16 * per_slice;
    if (gid >= total) return;

    int slice = gid / per_slice;
    int j = (gid % per_slice) + 2;
    int offset = slice * n_entries;
    const FieldElement* h = h_buf + (size_t)slice * n_entries;

    FieldElement z_inv = h[j - 2];
    FieldElement z_inv2, z_inv3;
    field_sqr(&z_inv, &z_inv2);
    field_mul(&z_inv, &z_inv2, &z_inv3);

    FieldElement ax, ay;
    field_mul(&aff_table[offset + j].x, &z_inv2, &ax);
    field_mul(&aff_table[offset + j].y, &z_inv3, &ay);
    aff_table[offset + j].x = ax;
    aff_table[offset + j].y = ay;
}

// ============================================================================
// LUT-accelerated pipeline and detail kernels
// ============================================================================

// Full BIP-352 pipeline using LUT for k*G step
__global__ void bip352_pipeline_kernel_lut(
    const secp256k1::cuda::JacobianPoint* __restrict__ tweak_points,
    const secp256k1::cuda::Scalar* __restrict__ scan_key,
    const secp256k1::cuda::JacobianPoint* __restrict__ spend_point,
    const secp256k1::cuda::AffinePoint* __restrict__ gen_lut,
    int64_t* __restrict__ prefixes,
    int n)
{
    using namespace secp256k1::cuda;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // 1. k*P
    JacobianPoint shared;
    scalar_mul_glv(&tweak_points[idx], scan_key, &shared);

    // 2. Serialize
    uint8_t comp[33];
    point_to_compressed(&shared, comp);

    // 3. Tagged SHA-256
    uint8_t ser[37];
    for (int i = 0; i < 33; i++) ser[i] = comp[i];
    ser[33] = 0; ser[34] = 0; ser[35] = 0; ser[36] = 0;
    uint8_t hash[32];
    bip352_tagged_sha256(ser, 37, hash);

    // 4. k*G via LUT (15 additions, 0 doublings!)
    Scalar hs;
    scalar_from_bytes(hash, &hs);
    JacobianPoint out;
    scalar_mul_generator_lut(&hs, gen_lut, &out);

    // 5. Point add
    JacobianPoint cand;
    jacobian_add_mixed_unchecked(&out, &BIP352_SPEND_AFFINE, &cand);

    // 6. Extract X prefix directly; the full compressed candidate is not needed here.
    prefixes[idx] = point_prefix64(&cand);
}

// ============================================================================
// Precomputed-table kernels
// ============================================================================

// Pass 1: build per-tweak wNAF tables into transposed global memory.
// Layout: tables_P[slot * n + idx], tables_phiP[slot * n + idx], globalz_buf[idx].
// Transposed so that during the wNAF loop (where all threads share the same slot
// from constant-memory BIP352_SCANKEY_WNAF) reads are fully coalesced.
__global__ void precompute_tweak_tables_kernel(
    const secp256k1::cuda::JacobianPoint* __restrict__ tweak_points,
    secp256k1::cuda::AffinePoint* __restrict__ tables_P,
    secp256k1::cuda::AffinePoint* __restrict__ tables_phiP,
    secp256k1::cuda::FieldElement* __restrict__ globalz_buf,
    int n)
{
    using namespace secp256k1::cuda;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Normalise input point (should already be z=1 from decompression).
    const JacobianPoint* p = &tweak_points[idx];
    AffinePoint base;
    if (p->z.limbs[0] == 1 && p->z.limbs[1] == 0 &&
        p->z.limbs[2] == 0 && p->z.limbs[3] == 0) {
        base.x = p->x;
        base.y = p->y;
    } else {
        FieldElement z_inv, z_inv2, z_inv3;
        field_inv(&p->z, &z_inv);
        field_sqr(&z_inv, &z_inv2);
        field_mul(&z_inv2, &z_inv, &z_inv3);
        field_mul(&p->x, &z_inv2, &base.x);
        field_mul(&p->y, &z_inv3, &base.y);
    }

    if (BIP352_SCANKEY_WNAF.k1_neg) {
        field_negate(&base.y, &base.y);
    }

    constexpr int TS = (1 << (SCAN_WNAF_W - 2));
    AffinePoint tbl_P[TS];
    FieldElement globalz;
    build_wnaf_table_zr(&base, tbl_P, TS, &globalz);

    AffinePoint tbl_phiP[TS];
    derive_endo_table(tbl_P, tbl_phiP, TS, BIP352_SCANKEY_WNAF.flip_phi != 0);

    // Write transposed: tables_P[slot * n + idx]
    for (int s = 0; s < TS; ++s) {
        tables_P[   s * n + idx] = tbl_P[s];
        tables_phiP[s * n + idx] = tbl_phiP[s];
    }
    globalz_buf[idx] = globalz;
}

// wNAF scalar-mul using precomputed tables from global memory.
// All threads share the same wNAF digits (constant memory) => same slot per step
// => reads tables_P[slot * n + idx] are fully coalesced across the warp.
__device__ inline void scalar_mul_fixed_scan_pretbl(
    int idx,
    const secp256k1::cuda::AffinePoint* __restrict__ tables_P,
    const secp256k1::cuda::AffinePoint* __restrict__ tables_phiP,
    const secp256k1::cuda::FieldElement* __restrict__ globalz_buf,
    int n,
    secp256k1::cuda::JacobianPoint* r)
{
    using namespace secp256k1::cuda;

    FieldElement globalz = globalz_buf[idx];

    r->infinity = true;
    field_set_zero(&r->x);
    field_set_one(&r->y);
    field_set_zero(&r->z);

    #pragma unroll 1
    for (int i = SCAN_WNAF_MAXLEN - 1; i >= 0; --i) {
        if (!r->infinity) {
            jacobian_double_unchecked(r, r);
        }

        int8_t d1 = BIP352_SCANKEY_WNAF.wnaf1[i];
        if (d1 != 0) {
            int slot = ((d1 > 0) ? d1 : -d1);
            slot = (slot - 1) >> 1;
            // Coalesced: slot identical for all threads in warp (shared scan key).
            AffinePoint pt = tables_P[slot * n + idx];
            if (d1 < 0) field_negate(&pt.y, &pt.y);

            if (r->infinity) {
                r->x = pt.x; r->y = pt.y;
                field_set_one(&r->z); r->infinity = false;
            } else {
                jacobian_add_mixed_unchecked(r, &pt, r);
            }
        }

        int8_t d2 = BIP352_SCANKEY_WNAF.wnaf2[i];
        if (d2 != 0) {
            int slot = ((d2 > 0) ? d2 : -d2);
            slot = (slot - 1) >> 1;
            AffinePoint pt = tables_phiP[slot * n + idx];
            if (d2 < 0) field_negate(&pt.y, &pt.y);

            if (r->infinity) {
                r->x = pt.x; r->y = pt.y;
                field_set_one(&r->z); r->infinity = false;
            } else {
                jacobian_add_mixed_unchecked(r, &pt, r);
            }
        }
    }

    if (!r->infinity) {
        FieldElement tmp;
        field_mul(&r->z, &globalz, &tmp);
        r->z = tmp;
    }
}

// Pass 2: full pipeline using precomputed per-tweak tables.
__global__ void bip352_pipeline_kernel_lut_pretbl(
    const secp256k1::cuda::AffinePoint* __restrict__ tables_P,
    const secp256k1::cuda::AffinePoint* __restrict__ tables_phiP,
    const secp256k1::cuda::FieldElement* __restrict__ globalz_buf,
    const secp256k1::cuda::AffinePoint* __restrict__ gen_lut,
    int64_t* __restrict__ prefixes,
    int n)
{
    using namespace secp256k1::cuda;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // 1. k*P via precomputed tables (no local table build, no register spill for tables).
    JacobianPoint shared;
    scalar_mul_fixed_scan_pretbl(idx, tables_P, tables_phiP, globalz_buf, n, &shared);

    // 2. Serialize into tagged-hash input buffer.
    uint8_t ser[37];
    bip352_shared_secret_input(&shared, ser);

    // 3. Tagged SHA-256.
    uint8_t hash[32];
    bip352_tagged_sha256(ser, 37, hash);

    // 4. k*G via LUT.
    Scalar hs;
    scalar_from_bytes(hash, &hs);
    JacobianPoint out;
    scalar_mul_generator_lut(&hs, gen_lut, &out);

    // 5. Point add.
    JacobianPoint cand;
    jacobian_add_mixed_unchecked(&out, &BIP352_SPEND_AFFINE, &cand);

    // 6. Extract X prefix.
    prefixes[idx] = point_prefix64(&cand);
}

// k*G detail kernel using LUT
__global__ void kernel_generator_mul_lut(
    const uint8_t* __restrict__ hash_bytes,
    const secp256k1::cuda::AffinePoint* __restrict__ gen_lut,
    secp256k1::cuda::JacobianPoint* __restrict__ out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    secp256k1::cuda::Scalar hs;
    secp256k1::cuda::scalar_from_bytes(hash_bytes + idx * 32, &hs);
    secp256k1::cuda::scalar_mul_generator_lut(&hs, gen_lut, &out[idx]);
}

// k*G detail kernel using w=8 constant table
__global__ void kernel_generator_mul_w8(
    const uint8_t* __restrict__ hash_bytes,
    secp256k1::cuda::JacobianPoint* __restrict__ out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    secp256k1::cuda::Scalar hs;
    secp256k1::cuda::scalar_from_bytes(hash_bytes + idx * 32, &hs);
    secp256k1::cuda::scalar_mul_generator_w8(&hs, &out[idx]);
}
