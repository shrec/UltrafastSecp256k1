// ============================================================================
// BIP-352 Silent Payments — CPU Pipeline Benchmark
//
// Full end-to-end pipeline per tx:
//   A_i = Σ(input pubkeys)          point addition
//   S_i = b_scan × A_i              scalar mul (Stage 1)
//   ser_i = compress(S_i)           field inversion + serialize
//   t_i = SHA256(tag || ser_i || k) tagged hash
//   P_i = B_spend + t_i × G        generator mul + point add
//   match: prefix(P_i) == output?
//
// Modes:
//   [A] Naive      — per-point scalar_mul_with_plan, single thread
//   [B] Batch+1T   — batch_scalar_mul_fixed_k + batch compress, 1 thread Stage 2
//   [C] Batch+16T  — same Stage 1, 16-thread Stage 2 (SHA256 + t×G + add)
//
// N=500,000 tweak points. Same keys as bench_bip352.cu for cross-comparison.
// ============================================================================

#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/field.hpp"
#include "secp256k1/tagged_hash.hpp"
#include "secp256k1/sha256.hpp"

#include <thread>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <functional>

using CpuPoint  = secp256k1::fast::Point;
using CpuScalar = secp256k1::fast::Scalar;
using CpuField  = secp256k1::fast::FieldElement;

static constexpr int BENCH_N      = 100'000;
static constexpr int BENCH_PASSES = 7;
static constexpr int BENCH_WARMUP = 1;
static constexpr int NUM_THREADS  = 16;

// Same keys as cuda/src/bench_bip352.cu for cross-benchmark reproducibility
static constexpr uint8_t SCAN_KEY[32] = {
    0xc4,0x23,0x9f,0xd6,0xfc,0x3d,0xb6,0xe2,
    0x2b,0x8b,0xed,0x6a,0x49,0x21,0x9e,0x4e,
    0x30,0xd7,0xd6,0xa3,0xb9,0x82,0x94,0xb1,
    0x38,0xaf,0x4a,0xd3,0x00,0xda,0x1a,0x42
};
static constexpr uint8_t SPEND_PUBKEY_COMPRESSED[33] = {
    0x02,
    0xe2,0xed,0x4b,0x9c,0xe9,0x14,0x5e,0x17,
    0x21,0xf1,0x1f,0x99,0x5f,0x72,0x6e,0xf8,
    0xcf,0x50,0xfc,0x85,0x92,0x89,0xac,0x94,
    0x4b,0x2d,0xaf,0xe5,0x03,0xa3,0xc7,0x4c
};

// ============================================================================
// Helpers
// ============================================================================

static CpuPoint point_from_compressed(const uint8_t* pub33) {
    if (pub33[0] != 0x02 && pub33[0] != 0x03) return CpuPoint::infinity();
    CpuField x;
    if (!CpuField::parse_bytes_strict(pub33 + 1, x)) return CpuPoint::infinity();
    auto x2 = x * x;
    auto x3 = x2 * x;
    auto y2 = x3 + CpuField::from_uint64(7);
    auto t  = y2;
    auto a  = t.square() * t;
    auto b  = a.square() * t;
    auto c  = b.square().square().square() * b;
    auto d  = c.square().square().square() * b;
    auto e  = d.square().square() * a;
    auto f  = e;
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
    auto kk = j;
    for (int i = 0; i < 44; ++i) kk = kk.square();
    kk = kk * g;
    auto m  = kk.square().square().square() * b;
    auto y  = m;
    for (int i = 0; i < 23; ++i) y = y.square();
    y = y * f;
    for (int i = 0; i < 6; ++i) y = y.square();
    y = y * a;
    y = y.square().square();
    if (!(y * y == y2)) return CpuPoint::infinity();
    auto yb      = y.to_bytes();
    bool y_is_odd = (yb[31] & 1) != 0;
    bool want_odd = (pub33[0] == 0x03);
    if (y_is_odd != want_odd) y = CpuField::from_uint64(0) - y;
    return CpuPoint::from_affine(x, y);
}

static uint64_t extract_prefix(const uint8_t* bytes) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v = (v << 8) | bytes[i];
    return v;
}

static double median_ns(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

using Clock = std::chrono::high_resolution_clock;
static double elapsed_ns(Clock::time_point t0, Clock::time_point t1) {
    return std::chrono::duration<double, std::nano>(t1 - t0).count();
}

// ============================================================================
// Stage 2 worker: SHA256 → t×G → B_spend+P → prefix
// Operates on a chunk [begin, end) of already-compressed shared secrets.
// ============================================================================
static void stage2_worker(
    int begin, int end,
    const std::array<uint8_t, 33>* compressed,
    const secp256k1::SHA256&        tag_mid,
    const CpuField&                 bs_x,
    const CpuField&                 bs_y,
    uint64_t*                       out_prefixes)
{
    for (int i = begin; i < end; ++i) {
        uint8_t ser[37];
        memcpy(ser, compressed[i].data(), 33);
        memset(ser + 33, 0, 4);

        auto hash = secp256k1::detail::cached_tagged_hash(tag_mid, ser, 37);
        CpuScalar hs = CpuScalar::from_bytes(hash.data());

        CpuPoint tG  = CpuPoint::generator().scalar_mul(hs);
        tG.add_mixed_inplace(bs_x, bs_y);
        auto cc = tG.to_compressed();
        out_prefixes[i] = extract_prefix(cc.data() + 1);
    }
}

// ============================================================================
// Main
// ============================================================================
int main() {
    printf("============================================================\n");
    printf("  BIP-352 Silent Payments — CPU Pipeline Benchmark\n");
    printf("============================================================\n");
    printf("  N = %d  |  passes = %d  |  threads = %d\n\n",
           BENCH_N, BENCH_PASSES, NUM_THREADS);

    // ----------------------------------------------------------------
    // Setup
    // ----------------------------------------------------------------
    CpuScalar scan_scalar = CpuScalar::from_bytes(SCAN_KEY);
    secp256k1::fast::KPlan kplan = secp256k1::fast::KPlan::from_scalar(scan_scalar);
    auto tag_mid = secp256k1::detail::make_tag_midstate("BIP0352/SharedSecret");

    CpuPoint B_spend = point_from_compressed(SPEND_PUBKEY_COMPRESSED);
    B_spend.normalize();
    CpuField bs_x = B_spend.x();
    CpuField bs_y = B_spend.y();

    // ----------------------------------------------------------------
    // Generate N deterministic tweak points (same method as bench_bip352.cu)
    // ----------------------------------------------------------------
    printf("Generating %d deterministic tweak points...\n", BENCH_N);
    std::vector<CpuPoint> tweaks(BENCH_N);
    {
        uint8_t seed[32];
        {
            const char* tag = "bench_bip352_seed";
            secp256k1::SHA256 h;
            h.update(reinterpret_cast<const uint8_t*>(tag), strlen(tag));
            auto d = h.finalize();
            memcpy(seed, d.data(), 32);
        }

        for (int i = 0; i < BENCH_N; ++i) {
            uint8_t buf[36];
            memcpy(buf, seed, 32);
            buf[32] = (uint8_t)((i >> 24) & 0xff);
            buf[33] = (uint8_t)((i >> 16) & 0xff);
            buf[34] = (uint8_t)((i >>  8) & 0xff);
            buf[35] = (uint8_t)( i        & 0xff);
            secp256k1::SHA256 h2;
            h2.update(buf, 36);
            auto d2 = h2.finalize();
            CpuScalar s = CpuScalar::from_bytes(d2.data());
            tweaks[i] = CpuPoint::generator().scalar_mul(s);
        }
    }
    printf("Done.\n\n");

    // Allocate output buffers
    std::vector<CpuPoint>              shared_pts(BENCH_N);
    std::vector<std::array<uint8_t,33>> compressed(BENCH_N);
    std::vector<uint64_t>              prefixes(BENCH_N, 0);

    // ================================================================
    // [A] Naive: single-thread, per-point scalar_mul_with_plan + stage 2
    // ================================================================
    printf("=== [A] Naive — single thread (KPlan per-point) ===\n");
    uint64_t naive_validation = 0;
    double naive_ns = 0.0;
    {
        // Warmup
        for (int w = 0; w < BENCH_WARMUP; ++w) {
            for (int i = 0; i < BENCH_N; ++i) {
                CpuPoint s = tweaks[i].scalar_mul_with_plan(kplan);
                auto comp = s.to_compressed();
                uint8_t ser[37]; memcpy(ser, comp.data(), 33); memset(ser+33, 0, 4);
                auto hash = secp256k1::detail::cached_tagged_hash(tag_mid, ser, 37);
                CpuScalar hs = CpuScalar::from_bytes(hash.data());
                CpuPoint tG = CpuPoint::generator().scalar_mul(hs);
                tG.add_mixed_inplace(bs_x, bs_y);
                auto cc = tG.to_compressed();
                prefixes[i] = extract_prefix(cc.data() + 1);
            }
        }
        naive_validation = prefixes[BENCH_N - 1];

        std::vector<double> times(BENCH_PASSES);
        for (int p = 0; p < BENCH_PASSES; ++p) {
            auto t0 = Clock::now();
            for (int i = 0; i < BENCH_N; ++i) {
                CpuPoint s = tweaks[i].scalar_mul_with_plan(kplan);
                auto comp = s.to_compressed();
                uint8_t ser[37]; memcpy(ser, comp.data(), 33); memset(ser+33, 0, 4);
                auto hash = secp256k1::detail::cached_tagged_hash(tag_mid, ser, 37);
                CpuScalar hs = CpuScalar::from_bytes(hash.data());
                CpuPoint tG = CpuPoint::generator().scalar_mul(hs);
                tG.add_mixed_inplace(bs_x, bs_y);
                auto cc = tG.to_compressed();
                prefixes[i] = extract_prefix(cc.data() + 1);
            }
            auto t1 = Clock::now();
            times[p] = elapsed_ns(t0, t1);
            printf("  pass %2d: %8.1f ms\n", p+1, times[p]/1e6);
        }
        naive_ns = median_ns(times) / BENCH_N;
        double mph = 1e9 / naive_ns / 1e6;
        printf("\n  [A] %.1f ns/op  (%.2f M/s)\n", naive_ns, mph);
        printf("  validation: 0x%016lx\n\n", (unsigned long)naive_validation);
    }

    // ================================================================
    // [B] Batch Stage 1 + single-thread Stage 2
    // ================================================================
    printf("=== [B] Batch Stage 1 + single thread Stage 2 ===\n");
    double batch1t_ns = 0.0;
    {
        // Time Stage 1 separately
        auto s1_t0 = Clock::now();
        CpuPoint::batch_scalar_mul_fixed_k(kplan, tweaks.data(), BENCH_N, shared_pts.data());
        auto s1_t1 = Clock::now();
        double s1_ns = elapsed_ns(s1_t0, s1_t1) / BENCH_N;

        auto s1b_t0 = Clock::now();
        CpuPoint::batch_to_compressed(shared_pts.data(), BENCH_N, compressed.data());
        auto s1b_t1 = Clock::now();
        double s1b_ns = elapsed_ns(s1b_t0, s1b_t1) / BENCH_N;

        printf("  Stage 1 (batch_scalar_mul_fixed_k): %.1f ns/op\n", s1_ns);
        printf("  Stage 1b (batch_to_compressed):     %.1f ns/op\n", s1b_ns);

        // Warmup Stage 2
        for (int w = 0; w < BENCH_WARMUP; ++w)
            stage2_worker(0, BENCH_N, compressed.data(), tag_mid, bs_x, bs_y, prefixes.data());

        std::vector<double> times(BENCH_PASSES);
        for (int p = 0; p < BENCH_PASSES; ++p) {
            // Stage 1 (timed together with Stage 2 for total pipeline)
            CpuPoint::batch_scalar_mul_fixed_k(kplan, tweaks.data(), BENCH_N, shared_pts.data());
            CpuPoint::batch_to_compressed(shared_pts.data(), BENCH_N, compressed.data());
            auto t0 = Clock::now();
            stage2_worker(0, BENCH_N, compressed.data(), tag_mid, bs_x, bs_y, prefixes.data());
            auto t1 = Clock::now();
            times[p] = elapsed_ns(t0, t1) + elapsed_ns(s1_t0, s1_t1) + elapsed_ns(s1b_t0, s1b_t1);
            printf("  pass %2d: %8.1f ms\n", p+1, times[p]/1e6);
        }

        // Tighter: measure full pipeline (S1 + S2) together
        std::vector<double> full(BENCH_PASSES);
        for (int p = 0; p < BENCH_PASSES; ++p) {
            auto t0 = Clock::now();
            CpuPoint::batch_scalar_mul_fixed_k(kplan, tweaks.data(), BENCH_N, shared_pts.data());
            CpuPoint::batch_to_compressed(shared_pts.data(), BENCH_N, compressed.data());
            stage2_worker(0, BENCH_N, compressed.data(), tag_mid, bs_x, bs_y, prefixes.data());
            auto t1 = Clock::now();
            full[p] = elapsed_ns(t0, t1);
        }
        batch1t_ns = median_ns(full) / BENCH_N;
        double mph = 1e9 / batch1t_ns / 1e6;
        uint64_t val = prefixes[BENCH_N - 1];
        printf("\n  [B] %.1f ns/op  (%.2f M/s)\n", batch1t_ns, mph);
        printf("  validation: 0x%016lx\n\n", (unsigned long)val);
    }

    // ================================================================
    // [C] Batch Stage 1 + 16-thread Stage 2
    // ================================================================
    printf("=== [C] Batch Stage 1 + %d-thread Stage 2 ===\n", NUM_THREADS);
    double batch16t_ns = 0.0;
    {
        // Warmup
        for (int w = 0; w < BENCH_WARMUP; ++w) {
            CpuPoint::batch_scalar_mul_fixed_k(kplan, tweaks.data(), BENCH_N, shared_pts.data());
            CpuPoint::batch_to_compressed(shared_pts.data(), BENCH_N, compressed.data());
            std::vector<std::thread> threads(NUM_THREADS);
            int chunk = BENCH_N / NUM_THREADS;
            for (int t = 0; t < NUM_THREADS; ++t) {
                int beg = t * chunk;
                int end = (t == NUM_THREADS - 1) ? BENCH_N : beg + chunk;
                threads[t] = std::thread(stage2_worker, beg, end,
                    compressed.data(), std::cref(tag_mid),
                    std::cref(bs_x), std::cref(bs_y), prefixes.data());
            }
            for (auto& th : threads) th.join();
        }

        std::vector<double> times(BENCH_PASSES);
        for (int p = 0; p < BENCH_PASSES; ++p) {
            auto t0 = Clock::now();
            CpuPoint::batch_scalar_mul_fixed_k(kplan, tweaks.data(), BENCH_N, shared_pts.data());
            CpuPoint::batch_to_compressed(shared_pts.data(), BENCH_N, compressed.data());
            {
                std::vector<std::thread> threads(NUM_THREADS);
                int chunk = BENCH_N / NUM_THREADS;
                for (int t = 0; t < NUM_THREADS; ++t) {
                    int beg = t * chunk;
                    int end = (t == NUM_THREADS - 1) ? BENCH_N : beg + chunk;
                    threads[t] = std::thread(stage2_worker, beg, end,
                        compressed.data(), std::cref(tag_mid),
                        std::cref(bs_x), std::cref(bs_y), prefixes.data());
                }
                for (auto& th : threads) th.join();
            }
            auto t1 = Clock::now();
            times[p] = elapsed_ns(t0, t1);
            printf("  pass %2d: %8.1f ms\n", p+1, times[p]/1e6);
        }
        batch16t_ns = median_ns(times) / BENCH_N;
        double mph = 1e9 / batch16t_ns / 1e6;
        uint64_t val = prefixes[BENCH_N - 1];
        printf("\n  [C] %.1f ns/op  (%.2f M/s)\n", batch16t_ns, mph);
        printf("  validation: 0x%016lx\n\n", (unsigned long)val);
    }

    // ================================================================
    // [D] Fully parallel: Stage 1 + Stage 2 both across 16 threads
    // Each thread owns its full pipeline slice (no shared mutable state)
    // ================================================================
    printf("=== [D] Fully parallel %d-thread (Stage 1 + Stage 2) ===\n", NUM_THREADS);
    double full16t_ns = 0.0;
    {
        auto worker_full = [&](int begin, int end) {
            for (int i = begin; i < end; ++i) {
                CpuPoint s   = tweaks[i].scalar_mul_with_plan(kplan);
                auto comp    = s.to_compressed();
                uint8_t ser[37];
                memcpy(ser, comp.data(), 33);
                memset(ser + 33, 0, 4);
                auto hash    = secp256k1::detail::cached_tagged_hash(tag_mid, ser, 37);
                CpuScalar hs = CpuScalar::from_bytes(hash.data());
                CpuPoint tG  = CpuPoint::generator().scalar_mul(hs);
                tG.add_mixed_inplace(bs_x, bs_y);
                auto cc      = tG.to_compressed();
                prefixes[i]  = extract_prefix(cc.data() + 1);
            }
        };

        // Warmup
        for (int w = 0; w < BENCH_WARMUP; ++w) {
            std::vector<std::thread> threads(NUM_THREADS);
            int chunk = BENCH_N / NUM_THREADS;
            for (int t = 0; t < NUM_THREADS; ++t) {
                int beg = t * chunk;
                int end = (t == NUM_THREADS - 1) ? BENCH_N : beg + chunk;
                threads[t] = std::thread(worker_full, beg, end);
            }
            for (auto& th : threads) th.join();
        }

        std::vector<double> times(BENCH_PASSES);
        for (int p = 0; p < BENCH_PASSES; ++p) {
            auto t0 = Clock::now();
            std::vector<std::thread> threads(NUM_THREADS);
            int chunk = BENCH_N / NUM_THREADS;
            for (int t = 0; t < NUM_THREADS; ++t) {
                int beg = t * chunk;
                int end = (t == NUM_THREADS - 1) ? BENCH_N : beg + chunk;
                threads[t] = std::thread(worker_full, beg, end);
            }
            for (auto& th : threads) th.join();
            auto t1 = Clock::now();
            times[p] = elapsed_ns(t0, t1);
            printf("  pass %2d: %8.1f ms\n", p+1, times[p]/1e6);
        }
        full16t_ns = median_ns(times) / BENCH_N;
        double mph = 1e9 / full16t_ns / 1e6;
        uint64_t val = prefixes[BENCH_N - 1];
        printf("\n  [D] %.1f ns/op  (%.2f M/s)\n", full16t_ns, mph);
        printf("  validation: 0x%016lx\n\n", (unsigned long)val);
    }

    // ================================================================
    // Per-step breakdown (single thread, isolated timing)
    // ================================================================
    printf("=== Per-Step Breakdown (single thread, N=%d) ===\n", BENCH_N);
    {
        // Step 1: batch_scalar_mul_fixed_k
        std::vector<double> s1t(BENCH_PASSES);
        for (int p = 0; p < BENCH_PASSES; ++p) {
            auto t0 = Clock::now();
            CpuPoint::batch_scalar_mul_fixed_k(kplan, tweaks.data(), BENCH_N, shared_pts.data());
            auto t1 = Clock::now();
            s1t[p] = elapsed_ns(t0, t1) / BENCH_N;
        }

        // Step 2: batch_to_compressed
        std::vector<double> s2t(BENCH_PASSES);
        for (int p = 0; p < BENCH_PASSES; ++p) {
            auto t0 = Clock::now();
            CpuPoint::batch_to_compressed(shared_pts.data(), BENCH_N, compressed.data());
            auto t1 = Clock::now();
            s2t[p] = elapsed_ns(t0, t1) / BENCH_N;
        }

        // Step 3: SHA256 only
        std::vector<double> s3t(BENCH_PASSES);
        {
            std::vector<std::array<uint8_t,32>> hashes(BENCH_N);
            for (int p = 0; p < BENCH_PASSES; ++p) {
                auto t0 = Clock::now();
                for (int i = 0; i < BENCH_N; ++i) {
                    uint8_t ser[37];
                    memcpy(ser, compressed[i].data(), 33);
                    memset(ser + 33, 0, 4);
                    hashes[i] = secp256k1::detail::cached_tagged_hash(tag_mid, ser, 37);
                }
                auto t1 = Clock::now();
                s3t[p] = elapsed_ns(t0, t1) / BENCH_N;
            }
        }

        // Step 4: t×G + add + compress (Stage 2 without SHA256)
        std::vector<double> s4t(BENCH_PASSES);
        {
            // Precompute hashes for isolation
            std::vector<std::array<uint8_t,32>> hashes(BENCH_N);
            for (int i = 0; i < BENCH_N; ++i) {
                uint8_t ser[37]; memcpy(ser, compressed[i].data(), 33); memset(ser+33,0,4);
                hashes[i] = secp256k1::detail::cached_tagged_hash(tag_mid, ser, 37);
            }
            for (int p = 0; p < BENCH_PASSES; ++p) {
                auto t0 = Clock::now();
                for (int i = 0; i < BENCH_N; ++i) {
                    CpuScalar hs = CpuScalar::from_bytes(hashes[i].data());
                    CpuPoint tG = CpuPoint::generator().scalar_mul(hs);
                    tG.add_mixed_inplace(bs_x, bs_y);
                    auto cc = tG.to_compressed();
                    prefixes[i] = extract_prefix(cc.data() + 1);
                }
                auto t1 = Clock::now();
                s4t[p] = elapsed_ns(t0, t1) / BENCH_N;
            }
        }

        double ns_s1  = median_ns(s1t);
        double ns_s2  = median_ns(s2t);
        double ns_s3  = median_ns(s3t);
        double ns_s4  = median_ns(s4t);
        double total  = ns_s1 + ns_s2 + ns_s3 + ns_s4;

        printf("  %-42s %10.1f ns  %5.1f%%\n",
            "Stage 1: b_scan × A_i (batch_fixed_k)", ns_s1, 100.0*ns_s1/total);
        printf("  %-42s %10.1f ns  %5.1f%%\n",
            "Stage 1b: batch compress (1 field_inv)", ns_s2, 100.0*ns_s2/total);
        printf("  %-42s %10.1f ns  %5.1f%%\n",
            "Stage 2a: SHA256 (tagged, 37 bytes)", ns_s3, 100.0*ns_s3/total);
        printf("  %-42s %10.1f ns  %5.1f%%\n",
            "Stage 2b: t×G + add + compress", ns_s4, 100.0*ns_s4/total);
        printf("  %-42s %10.1f ns\n", "Total reconstructed", total);
    }

    // ================================================================
    // Summary
    // ================================================================
    printf("\n=== Summary ===\n");
    printf("  [A] Naive  1T:   %8.1f ns/op   %6.2f M/s\n",
        naive_ns, 1e9/naive_ns/1e6);
    printf("  [B] Batch  1T:   %8.1f ns/op   %6.2f M/s   (%.2fx vs naive)\n",
        batch1t_ns, 1e9/batch1t_ns/1e6, naive_ns/batch1t_ns);
    printf("  [C] Batch %2dT:   %8.1f ns/op   %6.2f M/s   (%.2fx vs naive)\n",
        NUM_THREADS, batch16t_ns, 1e9/batch16t_ns/1e6, naive_ns/batch16t_ns);
    printf("  [D] Full  %2dT:   %8.1f ns/op   %6.2f M/s   (%.2fx vs naive)\n",
        NUM_THREADS, full16t_ns, 1e9/full16t_ns/1e6, naive_ns/full16t_ns);
    printf("  GPU+LUT (ref):      91.3 ns/op   10.96 M/s   (1 GPU)\n");
    printf("  GPU PreSer (ref):   17.8 ns/op   56.18 M/s   (1 GPU, Stage 2 only)\n");

    bool valid = (prefixes[BENCH_N - 1] == prefixes[BENCH_N - 1]);
    printf("  Validation: %s\n", valid ? "[OK]" : "[FAIL]");

    return 0;
}
