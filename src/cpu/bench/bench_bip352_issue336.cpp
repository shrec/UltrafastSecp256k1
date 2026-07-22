// ============================================================================
// GitHub issue #336 replay driver — CPU BIP-352 scan path v3.68.0 vs v4.5.0
//
// Reproduces (as closely as this hardware/OS permits) the reporter's drive
// shape:
//   - N tweak rows (default 10,360,000 — reporter's exact count)
//   - THREADS worker threads for the per-row scan stage (default 18)
//   - configure_fixed_base(window_bits=WB, thread_count=TC) — default
//     WB=12, TC=1, matching the issue text exactly
//   - "warm second run": the whole pipeline is executed twice per process;
//     only the second run's real/user/sys is the steady-state number the
//     reporter compared. The first run is a throwaway warmup.
//
// Two pipeline modes, selected by --mode=legacy|batch:
//
//   legacy — uses ONLY APIs that are byte-identical in v3.68.0 and v4.5.0:
//     Stage 1  : Point::scalar_mul_with_plan(KPlan)      (ECDH b_scan*A_i)
//     Stage 1b : Point::batch_to_compressed              (batch-inverts,
//                serializes S_i — exists verbatim in v3.68.0)
//     Stage 2a : cached_tagged_hash (BIP0352/SharedSecret midstate)
//     Stage 2b : Point::generator().scalar_mul(hs)        (fixed-base hash*G,
//                per ROW call — dispatches to the free function
//                scalar_mul_generator() in precompute.cpp, which takes
//                g_mutex per call — this is the exact caller path this
//                issue is about)
//     Stage 2c : pack (T_i already affine from Stage 2b's eager normalize())
//     Stage 2d : batch_add_affine_x(B_spend, T_i)          (candidate
//                staging — internally batch-inverts dx values; this is
//                the "batch_add_affine_x + fe_batch_inverse" the issue
//                text refers to)
//     This is the "OLDER-API caller path" preserved exactly.
//
//   batch — same Stage 1/1b/2a, but Stage 2b uses batch_scalar_mul_generator()
//     (ONE g_mutex acquisition for the whole batch instead of N) — only
//     available from v4.5.0 onward, gated by ISSUE336_HAVE_BATCH_API.
//     This is the "new-batch-API comparison" required by the issue triage
//     (reporter's question #2).
//
// Every printed number in the "not yet measured" sense: this file only
// prints wall-clock ns/op from the internal chrono timers. real/user/sys/
// RSS/faults/ctxsw/syscalls are measured EXTERNALLY by wrapping the whole
// process with /usr/bin/time -v and strace -c (see
// benchmarks/github_issue_336/run_matrix.sh). No number in this file is a
// performance CLAIM by itself — it is raw per-process evidence input.
// ============================================================================

#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/field.hpp"
#include "secp256k1/tagged_hash.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/batch_add_affine.hpp"
#include "secp256k1/precompute.hpp"

#include <thread>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <numeric>

using CpuPoint       = secp256k1::fast::Point;
using CpuScalar      = secp256k1::fast::Scalar;
using CpuField       = secp256k1::fast::FieldElement;
using CpuAffinePoint = secp256k1::fast::AffinePointCompact;

// Identical keys to bench_bip352_cpu.cpp for cross-benchmark reproducibility
// and so [issue-336] results are directly comparable to the existing suite.
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

using Clock = std::chrono::high_resolution_clock;
static double elapsed_ns(Clock::time_point t0, Clock::time_point t1) {
    return std::chrono::duration<double, std::nano>(t1 - t0).count();
}
static double median_ns(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct Args {
    std::size_t n          = 10'360'000; // reporter's exact tweak-row count
    unsigned    threads     = 18;         // reporter's exact worker-thread count
    unsigned    window_bits = 12;         // reporter's exact configure_fixed_base
    unsigned    table_threads = 1;        // reporter's exact configure_fixed_base
    bool        enable_glv = true;        // not stated by reporter; documented assumption
    int         passes     = 7;           // CLAUDE.md protocol minimum
    int         warmup     = 1;           // internal warmup passes (in addition to the
                                           // outer "warm second run" process-level warmup)
    std::string mode       = "legacy";    // legacy | batch
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto val = [&](const char* key) -> const char* {
            std::string pfx = std::string("--") + key + "=";
            if (arg.rfind(pfx, 0) == 0) return arg.c_str() + pfx.size();
            return nullptr;
        };
        if (auto v = val("n"))            a.n = std::strtoull(v, nullptr, 10);
        else if (auto v = val("threads"))      a.threads = (unsigned)std::strtoul(v, nullptr, 10);
        else if (auto v = val("window-bits"))  a.window_bits = (unsigned)std::strtoul(v, nullptr, 10);
        else if (auto v = val("table-threads")) a.table_threads = (unsigned)std::strtoul(v, nullptr, 10);
        else if (auto v = val("glv"))           a.enable_glv = std::string(v) == "1" || std::string(v) == "true";
        else if (auto v = val("passes"))        a.passes = std::atoi(v);
        else if (auto v = val("warmup"))        a.warmup = std::atoi(v);
        else if (auto v = val("mode"))          a.mode = v;
    }
    return a;
}

// ============================================================================
// Legacy (old-API) per-row pipeline. Every function used here exists
// byte-identically in v3.68.0 and v4.5.0.
// ============================================================================
static void run_legacy_pipeline(
    const Args& args,
    const secp256k1::fast::KPlan& kplan,
    const std::vector<CpuPoint>& tweaks,
    const secp256k1::SHA256& tag_mid,
    const CpuField& bs_x, const CpuField& bs_y,
    std::vector<CpuPoint>& shared_pts,
    std::vector<std::array<uint8_t,33>>& compressed,
    std::vector<CpuAffinePoint>& t_affine,
    std::vector<CpuField>& out_x,
    std::vector<CpuField>& scratch,
    std::vector<uint64_t>& prefixes)
{
    const std::size_t n = args.n;
    const unsigned nt = args.threads;

    // Stage 1: KPlan ECDH, threaded (per-row Point::scalar_mul_with_plan)
    {
        std::vector<std::thread> pool(nt);
        std::size_t chunk = n / nt;
        for (unsigned t = 0; t < nt; ++t) {
            std::size_t beg = t * chunk;
            std::size_t end = (t == nt - 1) ? n : beg + chunk;
            pool[t] = std::thread([&, beg, end]() {
                for (std::size_t i = beg; i < end; ++i)
                    shared_pts[i] = tweaks[i].scalar_mul_with_plan(kplan);
            });
        }
        for (auto& th : pool) th.join();
    }

    // Stage 1b: batch serialize (batch-inverts under the hood) — old API
    CpuPoint::batch_to_compressed(shared_pts.data(), n, compressed.data());

    // Stage 2a: tagged SHA-256, threaded
    std::vector<std::array<uint8_t,32>> hashes(n);
    {
        std::vector<std::thread> pool(nt);
        std::size_t chunk = n / nt;
        for (unsigned t = 0; t < nt; ++t) {
            std::size_t beg = t * chunk;
            std::size_t end = (t == nt - 1) ? n : beg + chunk;
            pool[t] = std::thread([&, beg, end]() {
                for (std::size_t i = beg; i < end; ++i) {
                    uint8_t ser[37];
                    memcpy(ser, compressed[i].data(), 33);
                    memset(ser + 33, 0, 4);
                    hashes[i] = secp256k1::detail::cached_tagged_hash(tag_mid, ser, 37);
                }
            });
        }
        for (auto& th : pool) th.join();
    }

    // Stage 2b: fixed-base hash*G — PER-ROW call through Point::generator().
    // scalar_mul(), which dispatches to the free function scalar_mul_generator()
    // in precompute.cpp. THIS is the caller path the issue is about: it takes
    // g_mutex once per row (10.36M times across `nt` threads).
    {
        std::vector<std::thread> pool(nt);
        std::size_t chunk = n / nt;
        for (unsigned t = 0; t < nt; ++t) {
            std::size_t beg = t * chunk;
            std::size_t end = (t == nt - 1) ? n : beg + chunk;
            pool[t] = std::thread([&, beg, end]() {
                for (std::size_t i = beg; i < end; ++i) {
                    CpuScalar hs = CpuScalar::from_bytes(hashes[i].data());
                    CpuPoint tG  = CpuPoint::generator().scalar_mul(hs); // <-- hot path
                    t_affine[i]  = {tG.x(), tG.y()};
                }
            });
        }
        for (auto& th : pool) th.join();
    }

    // Stage 2d: candidate staging — batch_add_affine_x internally batch-inverts
    // the dx values (this is the "fe_batch_inverse for candidate staging" in
    // the issue text).
    secp256k1::fast::batch_add_affine_x(bs_x, bs_y, t_affine.data(), out_x.data(), n, scratch);

    // Stage 2e: prefix extraction
    for (std::size_t i = 0; i < n; ++i) {
        auto xb = out_x[i].to_bytes();
        prefixes[i] = extract_prefix(xb.data());
    }
}

#if defined(ISSUE336_HAVE_BATCH_API)
// ============================================================================
// New-batch-API pipeline (v4.5.0+ only). Stage 2b uses batch_scalar_mul_
// generator() — ONE g_mutex acquisition for the whole N-row batch instead
// of N acquisitions.
// ============================================================================
static void run_batch_pipeline(
    const Args& args,
    const secp256k1::fast::KPlan& kplan,
    const std::vector<CpuPoint>& tweaks,
    const secp256k1::SHA256& tag_mid,
    const CpuField& bs_x, const CpuField& bs_y,
    std::vector<CpuPoint>& shared_pts,
    std::vector<std::array<uint8_t,33>>& compressed,
    std::vector<CpuAffinePoint>& t_affine,
    std::vector<CpuField>& out_x,
    std::vector<CpuField>& scratch,
    std::vector<uint64_t>& prefixes,
    std::vector<CpuScalar>& hash_scalars,
    std::vector<CpuPoint>& t_pts)
{
    const std::size_t n = args.n;
    const unsigned nt = args.threads;

    {
        std::vector<std::thread> pool(nt);
        std::size_t chunk = n / nt;
        for (unsigned t = 0; t < nt; ++t) {
            std::size_t beg = t * chunk;
            std::size_t end = (t == nt - 1) ? n : beg + chunk;
            pool[t] = std::thread([&, beg, end]() {
                for (std::size_t i = beg; i < end; ++i)
                    shared_pts[i] = tweaks[i].scalar_mul_with_plan(kplan);
            });
        }
        for (auto& th : pool) th.join();
    }

    CpuPoint::batch_to_compressed(shared_pts.data(), n, compressed.data());

    std::vector<std::array<uint8_t,32>> hashes(n);
    {
        std::vector<std::thread> pool(nt);
        std::size_t chunk = n / nt;
        for (unsigned t = 0; t < nt; ++t) {
            std::size_t beg = t * chunk;
            std::size_t end = (t == nt - 1) ? n : beg + chunk;
            pool[t] = std::thread([&, beg, end]() {
                for (std::size_t i = beg; i < end; ++i) {
                    uint8_t ser[37];
                    memcpy(ser, compressed[i].data(), 33);
                    memset(ser + 33, 0, 4);
                    hashes[i] = secp256k1::detail::cached_tagged_hash(tag_mid, ser, 37);
                    hash_scalars[i] = CpuScalar::from_bytes(hashes[i].data());
                }
            });
        }
        for (auto& th : pool) th.join();
    }

    // Stage 2b: caller keeps its own `nt`-way threading (this build has
    // SECP256K1_ENABLE_OPENMP=OFF, the CMake default — batch_scalar_mul_
    // generator() has no internal parallelism in that configuration, so a
    // single whole-N call from one thread would serialize all compute onto
    // one core). Each thread instead calls batch_scalar_mul_generator() ONCE
    // for its own chunk: `nt` g_mutex acquisitions total instead of N in the
    // legacy per-row path — this is the realistic caller-side migration the
    // issue asks about, not a change to the caller's threading model.
    {
        std::vector<std::thread> pool(nt);
        std::size_t chunk = n / nt;
        for (unsigned t = 0; t < nt; ++t) {
            std::size_t beg = t * chunk;
            std::size_t end = (t == nt - 1) ? n : beg + chunk;
            pool[t] = std::thread([&, beg, end]() {
                secp256k1::fast::batch_scalar_mul_generator(
                    hash_scalars.data() + beg, t_pts.data() + beg, end - beg);
            });
        }
        for (auto& th : pool) th.join();
    }

    // Stage 2c: batch-normalize the lazy-Jacobian results to affine using the
    // nonzero-optimized batch inversion (fe_batch_inverse_nonzero) — the
    // second migration the issue asks about. Results here are all non-zero
    // (t_pts[i] = hash_i*G with hash_i freshly derived from a tagged hash,
    // negligible probability of hitting infinity), so the nonzero-only
    // fast path is the correct choice for this workload, not fe_batch_inverse.
    // Kept as one serial batch-inversion over all N (matches the legacy
    // path's Stage 1b/batch_add_affine_x, which are also single serial
    // batch-inversions) so the two pipelines differ ONLY in Stage 2b.
    {
        std::vector<CpuField> z_vals(n);
        for (std::size_t i = 0; i < n; ++i) z_vals[i] = t_pts[i].z();
        secp256k1::fast::fe_batch_inverse_nonzero(z_vals.data(), n);
        for (std::size_t i = 0; i < n; ++i) {
            CpuField zinv2 = z_vals[i] * z_vals[i];
            CpuField zinv3 = zinv2 * z_vals[i];
            t_affine[i] = {t_pts[i].X() * zinv2, t_pts[i].Y() * zinv3};
        }
    }

    secp256k1::fast::batch_add_affine_x(bs_x, bs_y, t_affine.data(), out_x.data(), n, scratch);

    for (std::size_t i = 0; i < n; ++i) {
        auto xb = out_x[i].to_bytes();
        prefixes[i] = extract_prefix(xb.data());
    }
}
#endif

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    printf("============================================================\n");
    printf("  BIP-352 issue-336 replay — mode=%s\n", args.mode.c_str());
    printf("============================================================\n");
    printf("  N=%zu threads=%u window_bits=%u table_threads=%u glv=%d passes=%d\n\n",
           args.n, args.threads, args.window_bits, args.table_threads,
           (int)args.enable_glv, args.passes);

    secp256k1::fast::FixedBaseConfig cfg;
    cfg.window_bits  = args.window_bits;
    cfg.enable_glv   = args.enable_glv;
    cfg.use_cache    = false;   // deterministic fresh build, no cross-version cache-file skew
    cfg.thread_count = args.table_threads;
    secp256k1::fast::configure_fixed_base(cfg);
    secp256k1::fast::ensure_fixed_base_ready();

    CpuScalar scan_scalar = CpuScalar::from_bytes(SCAN_KEY);
    secp256k1::fast::KPlan kplan = secp256k1::fast::KPlan::from_scalar(scan_scalar);
    auto tag_mid = secp256k1::detail::make_tag_midstate("BIP0352/SharedSecret");

    CpuPoint B_spend = point_from_compressed(SPEND_PUBKEY_COMPRESSED);
    B_spend.normalize();
    CpuField bs_x = B_spend.x();
    CpuField bs_y = B_spend.y();

    printf("Generating %zu deterministic tweak points...\n", args.n);
    std::vector<CpuPoint> tweaks(args.n);
    {
        uint8_t seed[32];
        const char* tag = "bench_bip352_issue336_seed";
        secp256k1::SHA256 h;
        h.update(reinterpret_cast<const uint8_t*>(tag), strlen(tag));
        auto d = h.finalize();
        memcpy(seed, d.data(), 32);

        std::vector<std::thread> pool(args.threads);
        std::size_t chunk = args.n / args.threads;
        for (unsigned t = 0; t < args.threads; ++t) {
            std::size_t beg = t * chunk;
            std::size_t end = (t == args.threads - 1) ? args.n : beg + chunk;
            pool[t] = std::thread([&, beg, end]() {
                for (std::size_t i = beg; i < end; ++i) {
                    uint8_t buf[40];
                    memcpy(buf, seed, 32);
                    buf[32] = (uint8_t)((i >> 24) & 0xff);
                    buf[33] = (uint8_t)((i >> 16) & 0xff);
                    buf[34] = (uint8_t)((i >> 8) & 0xff);
                    buf[35] = (uint8_t)(i & 0xff);
                    buf[36] = (uint8_t)((i >> 32) & 0xff);
                    buf[37] = 0; buf[38] = 0; buf[39] = 0;
                    secp256k1::SHA256 h2;
                    h2.update(buf, 40);
                    auto d2 = h2.finalize();
                    CpuScalar s = CpuScalar::from_bytes(d2.data());
                    tweaks[i] = CpuPoint::generator().scalar_mul(s);
                }
            });
        }
        for (auto& th : pool) th.join();
    }
    printf("Done.\n\n");

    std::vector<CpuPoint>               shared_pts(args.n);
    std::vector<std::array<uint8_t,33>> compressed(args.n);
    std::vector<CpuAffinePoint>         t_affine(args.n);
    std::vector<CpuField>               out_x(args.n);
    std::vector<CpuField>               scratch;
    scratch.reserve(args.n);
    std::vector<uint64_t>               prefixes(args.n, 0);

#if defined(ISSUE336_HAVE_BATCH_API)
    std::vector<CpuScalar> hash_scalars(args.n);
    std::vector<CpuPoint>  t_pts(args.n);
#endif

    auto run_once = [&]() {
        if (args.mode == "legacy") {
            run_legacy_pipeline(args, kplan, tweaks, tag_mid, bs_x, bs_y,
                                 shared_pts, compressed, t_affine, out_x, scratch, prefixes);
#if defined(ISSUE336_HAVE_BATCH_API)
        } else if (args.mode == "batch") {
            run_batch_pipeline(args, kplan, tweaks, tag_mid, bs_x, bs_y,
                                shared_pts, compressed, t_affine, out_x, scratch, prefixes,
                                hash_scalars, t_pts);
#endif
        } else {
            fprintf(stderr, "unknown --mode=%s (available: legacy%s)\n", args.mode.c_str(),
#if defined(ISSUE336_HAVE_BATCH_API)
                    ",batch"
#else
                    ""
#endif
            );
            std::exit(2);
        }
    };

    // "Warm second run": run once (cold, discarded), then treat everything
    // that follows as the reporter's "second query in the same session"
    // steady state. This matches the issue's own methodology statement.
    auto cold_t0 = Clock::now();
    run_once();
    auto cold_t1 = Clock::now();
    printf("cold run (discarded, matches reporter's first query): %.1f ms\n",
           elapsed_ns(cold_t0, cold_t1) / 1e6);

    for (int w = 0; w < args.warmup; ++w) run_once();

    std::vector<double> times(args.passes);
    for (int p = 0; p < args.passes; ++p) {
        auto t0 = Clock::now();
        run_once();
        auto t1 = Clock::now();
        times[p] = elapsed_ns(t0, t1);
        printf("  warm pass %2d: %8.1f ms\n", p + 1, times[p] / 1e6);
    }
    double ns_per_op = median_ns(times) / (double)args.n;
    double mps = 1e9 / ns_per_op / 1e6;
    printf("\n  median warm pass: %.1f ns/op  (%.2f M/s)\n", ns_per_op, mps);
    printf("  validation: 0x%016lx\n", (unsigned long)prefixes[args.n - 1]);
    return 0;
}
