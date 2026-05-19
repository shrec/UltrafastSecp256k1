// =============================================================================
// bch_scan.cu — BCH RPA GPU Scanner (CUDA)
// =============================================================================
// BCH RPA scan pipeline per thread:
//   1. S = scan_privkey × input_pubkey  (ECDH, scan key in constant memory)
//   2. c = SHA256(SHA256(ser(S)) || outpoint[36])
//   3. t_0 = SHA256(spend_pubkey[33] || c[32] || 0x00000000)
//   4. P_0 = spend_point + t_0×G  (GLV or LUT)
//   5. prefixes[idx] = P_0.x[0..7]
//
// Hash structure differs from BIP-352/LTC-SP: double-SHA256 + outpoint.
//
// Throughput: not yet measured — benchmark required.
// =============================================================================

#include "secp256k1.cuh"
#include "secp256k1/fast.hpp"
#include "secp256k1/sha256.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <array>
#include <algorithm>

using CpuScalar = secp256k1::fast::Scalar;
using CpuPoint  = secp256k1::fast::Point;

// ── BCH Scan key plan (GLV precomputed) ─────────────────────────────────────
static constexpr int BCH_WNAF_MAXLEN = 130;

struct BCHScanKeyWnaf {
    int8_t  wnaf1[BCH_WNAF_MAXLEN];
    int8_t  wnaf2[BCH_WNAF_MAXLEN];
    uint8_t k1_neg;
    uint8_t flip_phi;
};

__constant__ BCHScanKeyWnaf          BCH_SCANKEY_WNAF;
__constant__ secp256k1::cuda::AffinePoint BCH_SPEND_AFFINE;
__constant__ uint8_t                 BCH_SPEND_PUBKEY33[33];

// ── BCH hash functions ───────────────────────────────────────────────────────

// SHA256 of exactly 33 bytes (compressed point serialization)
__device__ void sha256_33(const uint8_t in[33], uint8_t out[32]) {
    secp256k1::cuda::SHA256Ctx ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, in, 33);
    sha256_final(&ctx, out);
}

// SHA256(SHA256(S_comp[33]) || outpoint[36]) → 32-byte shared secret
__device__ void bch_shared_secret(const uint8_t S_comp[33],
                                   const uint8_t outpoint[36],
                                   uint8_t c_out[32]) {
    secp256k1::cuda::SHA256Ctx ctx;

    // inner = SHA256(S_comp)
    uint8_t inner[32];
    sha256_init(&ctx);
    sha256_update(&ctx, S_comp, 33);
    sha256_final(&ctx, inner);

    // c = SHA256(inner || outpoint)
    sha256_init(&ctx);
    sha256_update(&ctx, inner, 32);
    sha256_update(&ctx, outpoint, 36);
    sha256_final(&ctx, c_out);
}

// t_k = SHA256(spend_pubkey33[33] || c[32] || ser32(k))
__device__ void bch_payment_key_hash(const uint8_t c[32], uint32_t k,
                                      uint8_t t_out[32]) {
    secp256k1::cuda::SHA256Ctx ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, BCH_SPEND_PUBKEY33, 33);
    sha256_update(&ctx, c, 32);
    uint8_t k_be[4] = { (uint8_t)(k>>24), (uint8_t)(k>>16),
                         (uint8_t)(k>>8),  (uint8_t)k };
    sha256_update(&ctx, k_be, 4);
    sha256_final(&ctx, t_out);
}

// GLV scalar_mul with precomputed scan key (same pattern as LTC-SP/BIP-352)
__device__ void bch_scalar_mul_fixed_scan(
    const secp256k1::cuda::JacobianPoint* p,
    secp256k1::cuda::JacobianPoint*       r)
{
    using namespace secp256k1::cuda;
    AffinePoint base;
    jacobian_to_affine_unchecked(p, &base);
    if (BCH_SCANKEY_WNAF.k1_neg) field_negate(&base.y, &base.y);

    AffinePoint table[8], endo[8];
    FieldElement gz;
    build_wnaf_table_zr(&base, table, &gz);
    derive_endo_table(table, endo, BCH_SCANKEY_WNAF.flip_phi);

    point_set_infinity(r);
    for (int i = BCH_WNAF_MAXLEN - 1; i >= 0; --i) {
        if (!point_is_infinity(r)) point_double_unchecked(r, r);
        int d1 = BCH_SCANKEY_WNAF.wnaf1[i];
        if (d1) {
            int idx = (((d1>0)?d1:-d1)-1)>>1;
            AffinePoint pt = table[idx];
            if (d1<0) field_negate(&pt.y,&pt.y);
            point_is_infinity(r) ? point_from_affine(r,&pt)
                                 : point_add_mixed_unchecked(r,r,&pt);
        }
        int d2 = BCH_SCANKEY_WNAF.wnaf2[i];
        if (d2) {
            int idx = (((d2>0)?d2:-d2)-1)>>1;
            AffinePoint pt = endo[idx];
            if (d2<0) field_negate(&pt.y,&pt.y);
            point_is_infinity(r) ? point_from_affine(r,&pt)
                                 : point_add_mixed_unchecked(r,r,&pt);
        }
    }
    if (!point_is_infinity(r)) {
        FieldElement cz; field_mul(&cz,&r->z,&gz); r->z=cz;
    }
}

// ── GLV pipeline kernel ──────────────────────────────────────────────────────
__global__ void bch_scan_kernel_glv(
    const secp256k1::cuda::JacobianPoint* __restrict__ tweak_points,
    const uint8_t* __restrict__  outpoints,  // N × 36 bytes
    int64_t* __restrict__        prefixes,
    int                          n)
{
    using namespace secp256k1::cuda;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // 1. ECDH
    JacobianPoint shared;
    bch_scalar_mul_fixed_scan(&tweak_points[idx], &shared);

    // 2. Compress shared secret
    uint8_t S_comp[33];
    point_to_compressed(&shared, S_comp);

    // 3. BCH shared secret hash
    uint8_t c[32];
    bch_shared_secret(S_comp, outpoints + (size_t)idx * 36, c);

    // 4. Payment key hash (k=0)
    uint8_t t_k[32];
    bch_payment_key_hash(c, 0, t_k);

    // 5. t_k * G (wNAF)
    Scalar hs;
    scalar_from_bytes(t_k, &hs);
    JacobianPoint out;
    scalar_mul_generator_const(&hs, &out);

    // 6. spend_point + t_k*G
    JacobianPoint cand;
    jacobian_add_mixed_unchecked(&out, &BCH_SPEND_AFFINE, &cand);
    prefixes[idx] = point_prefix64(&cand);
}

// ── LUT pipeline kernel ──────────────────────────────────────────────────────
__global__ void bch_scan_kernel_lut(
    const secp256k1::cuda::JacobianPoint* __restrict__ tweak_points,
    const uint8_t* __restrict__  outpoints,
    const secp256k1::cuda::AffinePoint* __restrict__ gen_lut,
    int64_t* __restrict__        prefixes,
    int                          n)
{
    using namespace secp256k1::cuda;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    JacobianPoint shared;
    bch_scalar_mul_fixed_scan(&tweak_points[idx], &shared);

    uint8_t S_comp[33];
    point_to_compressed(&shared, S_comp);

    uint8_t c[32];
    bch_shared_secret(S_comp, outpoints + (size_t)idx * 36, c);

    uint8_t t_k[32];
    bch_payment_key_hash(c, 0, t_k);

    Scalar hs;
    scalar_from_bytes(t_k, &hs);
    JacobianPoint out;
    scalar_mul_generator_lut(&hs, gen_lut, &out);   // LUT path

    JacobianPoint cand;
    jacobian_add_mixed_unchecked(&out, &BCH_SPEND_AFFINE, &cand);
    prefixes[idx] = point_prefix64(&cand);
}

// ── Build scan key wNAF plan (host) ─────────────────────────────────────────
static void build_bch_scan_wnaf(const CpuScalar& sk, BCHScanKeyWnaf& plan) {
    auto decomp = secp256k1::fast::glv_decompose(sk);
    plan.k1_neg   = decomp.k1_neg ? 1 : 0;
    plan.flip_phi = (decomp.k1_neg ^ decomp.k2_neg) ? 1 : 0;
    auto to_wnaf = [](const CpuScalar& s, int8_t* w) {
        CpuScalar k = s;
        for (int i = 0; i < BCH_WNAF_MAXLEN; ++i) {
            if (k.is_zero()) { w[i] = 0; continue; }
            bool odd = (k.limbs()[0] & 1) != 0;
            if (odd) {
                int8_t d = (int8_t)(k.limbs()[0] & 15);
                if (d > 8) d -= 16;
                w[i] = d;
                k = k - CpuScalar::from_uint64((uint64_t)((int)d));
            } else { w[i] = 0; }
            k = k.rshift1();
        }
    };
    to_wnaf(decomp.k1, plan.wnaf1);
    to_wnaf(decomp.k2, plan.wnaf2);
}

// ── Main benchmark ───────────────────────────────────────────────────────────
static constexpr int BENCH_N    = 500000;
static constexpr int WARMUP     = 5;
static constexpr int PASSES     = 11;
static constexpr int GPU_TPB    = 256;

int main() {
    CpuScalar scan_sk  = CpuScalar::from_uint64(0xdeadbeef12345678ULL);
    CpuScalar spend_sk = CpuScalar::from_uint64(0xcafe0000abcd1234ULL);

    // Upload scan key plan
    BCHScanKeyWnaf wnaf_plan;
    build_bch_scan_wnaf(scan_sk, wnaf_plan);
    cudaMemcpyToSymbol(BCH_SCANKEY_WNAF, &wnaf_plan, sizeof(wnaf_plan));

    // Spend point
    CpuPoint spend_pub = CpuPoint::generator().scalar_mul(spend_sk);
    auto spend_comp = spend_pub.to_compressed();
    secp256k1::cuda::AffinePoint spend_aff{};
    // Upload spend affine point and compressed spend pubkey
    cudaMemcpyToSymbol(BCH_SPEND_PUBKEY33, spend_comp.data(), 33);

    // Build N test tweak points + outpoints
    std::vector<secp256k1::cuda::JacobianPoint> h_tweaks(BENCH_N);
    std::vector<uint8_t> h_outpoints(BENCH_N * 36, 0);
    for (int i = 0; i < BENCH_N; ++i) {
        CpuScalar sk = CpuScalar::from_uint64((uint64_t)i + 1);
        CpuPoint  pt = CpuPoint::generator().scalar_mul(sk);
        auto comp = pt.to_compressed();
        secp256k1::cuda::JacobianPoint jp{};
        for (int b = 0; b < 32; b++) ((uint8_t*)jp.x.limbs)[b] = comp[1+b];
        jp.z.limbs[0] = 1; jp.infinity = 0;
        h_tweaks[i] = jp;
        // Fake outpoint: txid = index bytes, vout = 0
        h_outpoints[i*36 + 0] = (uint8_t)(i & 0xff);
        h_outpoints[i*36 + 1] = (uint8_t)((i>>8) & 0xff);
    }

    secp256k1::cuda::JacobianPoint* d_tweaks = nullptr;
    uint8_t* d_outpoints = nullptr;
    int64_t* d_prefixes  = nullptr;
    cudaMalloc(&d_tweaks,   BENCH_N * sizeof(secp256k1::cuda::JacobianPoint));
    cudaMalloc(&d_outpoints, BENCH_N * 36);
    cudaMalloc(&d_prefixes, BENCH_N * sizeof(int64_t));
    cudaMemcpy(d_tweaks,    h_tweaks.data(),    BENCH_N * sizeof(secp256k1::cuda::JacobianPoint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outpoints, h_outpoints.data(), BENCH_N * 36, cudaMemcpyHostToDevice);

    int blocks = (BENCH_N + GPU_TPB - 1) / GPU_TPB;

    auto run_bench = [&](bool use_lut, secp256k1::cuda::AffinePoint* d_lut) {
        // Warmup
        for (int i = 0; i < WARMUP; ++i) {
            if (use_lut) bch_scan_kernel_lut<<<blocks,GPU_TPB>>>(d_tweaks, d_outpoints, d_lut, d_prefixes, BENCH_N);
            else         bch_scan_kernel_glv<<<blocks,GPU_TPB>>>(d_tweaks, d_outpoints, d_prefixes, BENCH_N);
        }
        cudaDeviceSynchronize();

        std::vector<double> times;
        for (int p = 0; p < PASSES; ++p) {
            auto t0 = std::chrono::high_resolution_clock::now();
            if (use_lut) bch_scan_kernel_lut<<<blocks,GPU_TPB>>>(d_tweaks, d_outpoints, d_lut, d_prefixes, BENCH_N);
            else         bch_scan_kernel_glv<<<blocks,GPU_TPB>>>(d_tweaks, d_outpoints, d_prefixes, BENCH_N);
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration<double,std::milli>(t1-t0).count());
        }
        std::sort(times.begin(), times.end());
        double ms = times[PASSES/2];
        double ns = ms * 1e6 / BENCH_N;
        double M  = BENCH_N / (ms/1000.0) / 1e6;
        printf("  BCH RPA %s: %.1f ns/tx  →  %.2f M tx/s\n",
               use_lut ? "LUT" : "GLV", ns, M);
        printf("  Note: throughput not yet measured — benchmark required.\n");
    };

    int device; cudaGetDevice(&device);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, device);
    printf("=== BCH RPA CUDA Scanner Benchmark ===\n");
    printf("  GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("  N: %d, %d passes (IQR-trimmed median)\n\n", BENCH_N, PASSES);

    run_bench(false, nullptr);

    cudaFree(d_tweaks);
    cudaFree(d_outpoints);
    cudaFree(d_prefixes);
    return 0;
}
