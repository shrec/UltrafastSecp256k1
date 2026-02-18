// ============================================================================
// Benchmark: FieldElement (4×64) vs FieldElement52 (5×52 Lazy-Reduction)
// ============================================================================
//
// Measures individual field operations and realistic ECC-like chains
// to quantify the lazy-reduction advantage of the 5×52 representation.
//
// Key hypothesis: 5×52 wins on add-heavy chains (ECC point ops) because
// additions are 5 plain adds with NO carry propagation, while 4×64 needs
// carry propagation + conditional reduction after every add.
// ============================================================================

#include "secp256k1/field_52.hpp"
#include "secp256k1/field.hpp"
#include "secp256k1/selftest.hpp"

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <array>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif

using namespace secp256k1::fast;
using Clock = std::chrono::high_resolution_clock;

// ── Benchmark Harness ────────────────────────────────────────────────────────

static constexpr int WARMUP   = 500;
static constexpr int PASSES   = 7;       // median-of-7 for stability

template<typename Func>
double bench_ns(Func&& f, int iterations) {
    // Warmup
    for (int i = 0; i < WARMUP; ++i) f();

    std::array<double, PASSES> runs;
    for (int p = 0; p < PASSES; ++p) {
        auto t0 = Clock::now();
        for (int i = 0; i < iterations; ++i) {
            f();
        }
        auto t1 = Clock::now();
        runs[p] = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
        ) / iterations;
    }
    std::sort(runs.begin(), runs.end());
    return runs[PASSES / 2];  // median
}

// Prevent dead-code elimination
static volatile uint64_t sink;

static void escape(const void* p) {
    sink = reinterpret_cast<uintptr_t>(p);
}

// ── Test Data ────────────────────────────────────────────────────────────────

// Generator point Gx, Gy
static const uint64_t GX_LIMBS[4] = {
    0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL,
    0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL
};
static const uint64_t GY_LIMBS[4] = {
    0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL,
    0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL
};

static FieldElement make_fe(const uint64_t limbs[4]) {
    return FieldElement::from_limbs({limbs[0], limbs[1], limbs[2], limbs[3]});
}

// ── Individual Operation Benchmarks ──────────────────────────────────────────

struct BenchResult {
    const char* name;
    double ns_4x64;
    double ns_5x52;
};

static void print_header() {
    std::printf("\n");
    std::printf("=================================================================\n");
    std::printf("  FieldElement (4x64) vs FieldElement52 (5x52) Benchmark\n");
    std::printf("  Median of %d passes, %d warmup iterations\n", PASSES, WARMUP);
    std::printf("=================================================================\n\n");
    std::printf("%-36s %10s %10s %10s\n", "Operation", "4x64 (ns)", "5x52 (ns)", "Ratio");
    std::printf("%-36s %10s %10s %10s\n", "────────────────────────────────────",
                "──────────", "──────────", "──────────");
}

static void print_result(const BenchResult& r) {
    double ratio = r.ns_4x64 / r.ns_5x52;
    const char* indicator = (ratio > 1.05) ? " <-- 5x52 wins" :
                            (ratio < 0.95) ? " <-- 4x64 wins" : "";
    std::printf("%-36s %9.2f  %9.2f  %8.3fx%s\n",
                r.name, r.ns_4x64, r.ns_5x52, ratio, indicator);
}

static void print_separator(const char* section) {
    std::printf("\n--- %s ---\n", section);
}

// ═════════════════════════════════════════════════════════════════════════════
// main
// ═════════════════════════════════════════════════════════════════════════════

int main() {
    std::printf("[bench_field_52] Running arithmetic validation...\n");
    secp256k1::fast::Selftest(false);
    std::printf("[bench_field_52] Validation OK\n");

#if defined(_WIN32)
    // Pin to CPU 0 and elevate priority for stable timings
    SetThreadAffinityMask(GetCurrentThread(), 1ULL);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
#endif

    // ── Prepare test data ────────────────────────────────────────────────────
    FieldElement fe_a = make_fe(GX_LIMBS);
    FieldElement fe_b = make_fe(GY_LIMBS);

    FieldElement52 fe52_a = FieldElement52::from_fe(fe_a);
    FieldElement52 fe52_b = FieldElement52::from_fe(fe_b);

    print_header();

    // ── 1. Single Addition ───────────────────────────────────────────────────
    print_separator("Single Operations");
    {
        constexpr int ITERS = 500000;

        FieldElement r4;
        double ns4 = bench_ns([&]() {
            r4 = fe_a + fe_b;
            escape(&r4);
        }, ITERS);

        FieldElement52 r5;
        double ns5 = bench_ns([&]() {
            r5 = fe52_a + fe52_b;
            escape(&r5);
        }, ITERS);

        print_result({"Addition (single)", ns4, ns5});
    }

    // ── 2. Single Multiplication ─────────────────────────────────────────────
    {
        constexpr int ITERS = 200000;

        FieldElement r4;
        double ns4 = bench_ns([&]() {
            r4 = fe_a * fe_b;
            escape(&r4);
        }, ITERS);

        FieldElement52 r5;
        double ns5 = bench_ns([&]() {
            r5 = fe52_a * fe52_b;
            escape(&r5);
        }, ITERS);

        print_result({"Multiplication (single)", ns4, ns5});
    }

    // ── 3. Single Squaring ───────────────────────────────────────────────────
    {
        constexpr int ITERS = 200000;

        FieldElement r4;
        double ns4 = bench_ns([&]() {
            r4 = fe_a.square();
            escape(&r4);
        }, ITERS);

        FieldElement52 r5;
        double ns5 = bench_ns([&]() {
            r5 = fe52_a.square();
            escape(&r5);
        }, ITERS);

        print_result({"Squaring (single)", ns4, ns5});
    }

    // ── 4. Normalization ─────────────────────────────────────────────────────
    {
        constexpr int ITERS = 500000;

        // Create slightly un-normalized element for 5×52
        FieldElement52 acc52 = fe52_a + fe52_b;

        double ns5_weak = bench_ns([&]() {
            FieldElement52 tmp = acc52;
            tmp.normalize_weak();
            escape(&tmp);
        }, ITERS);

        double ns5_full = bench_ns([&]() {
            FieldElement52 tmp = acc52;
            tmp.normalize();
            escape(&tmp);
        }, ITERS);

        std::printf("%-36s %9s  %9.2f\n", "Normalize (weak, 5x52 only)", "N/A", ns5_weak);
        std::printf("%-36s %9s  %9.2f\n", "Normalize (full, 5x52 only)", "N/A", ns5_full);
    }

    // ── 5. Negation ──────────────────────────────────────────────────────────
    {
        constexpr int ITERS = 500000;

        FieldElement r4;
        double ns4 = bench_ns([&]() {
            r4 = FieldElement{} - fe_a;
            escape(&r4);
        }, ITERS);

        FieldElement52 r5;
        double ns5 = bench_ns([&]() {
            r5 = fe52_a.negate(1);
            escape(&r5);
        }, ITERS);

        print_result({"Negation", ns4, ns5});
    }

    // ── 6. Half ──────────────────────────────────────────────────────────────
    {
        constexpr int ITERS = 500000;

        // 4×64 has no half() – skip comparison
        double ns4 = 0.0;

        FieldElement52 r5;
        double ns5 = bench_ns([&]() {
            r5 = fe52_a.half();
            escape(&r5);
        }, ITERS);

        std::printf("%-36s %9s  %9.2f\n", "Half (5x52 only)", "N/A", ns5);
    }

    // ── 7. Conversion cost ───────────────────────────────────────────────────
    {
        constexpr int ITERS = 500000;

        FieldElement52 r52;
        double ns_to52 = bench_ns([&]() {
            r52 = FieldElement52::from_fe(fe_a);
            escape(&r52);
        }, ITERS);

        FieldElement r4;
        double ns_to4 = bench_ns([&]() {
            r4 = fe52_a.to_fe();
            escape(&r4);
        }, ITERS);

        std::printf("%-36s %9s  %9.2f\n", "Convert 4x64 -> 5x52", "N/A", ns_to52);
        std::printf("%-36s %9.2f  %9s\n", "Convert 5x52 -> 4x64", ns_to4, "N/A");
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Chained Operations (realistic ECC workload simulation)
    // ═════════════════════════════════════════════════════════════════════════

    print_separator("Addition Chains (Lazy-Reduction Core Advantage)");

    // ── 8. Chain of N additions (the whole point of lazy reduction) ──────────
    for (int chain_len : {4, 8, 16, 32, 64}) {
        constexpr int ITERS = 50000;

        // 4×64: each add does carry + conditional reduction
        double ns4 = bench_ns([&]() {
            FieldElement acc = fe_a;
            for (int i = 0; i < chain_len; ++i) {
                acc = acc + fe_b;
            }
            escape(&acc);
        }, ITERS);

        // 5×52: N plain adds, ONE normalize at end
        double ns5 = bench_ns([&]() {
            FieldElement52 acc = fe52_a;
            for (int i = 0; i < chain_len; ++i) {
                acc.add_assign(fe52_b);
            }
            acc.normalize_weak();
            escape(&acc);
        }, ITERS);

        char name[64];
        std::snprintf(name, sizeof(name), "Add chain (%d adds + norm)", chain_len);
        print_result({name, ns4, ns5});
    }

    // ── 9. Mixed add+mul chain (simulates point addition field ops) ──────────
    print_separator("Mixed Chains (ECC Point Operation Simulation)");

    // ECC point addition has roughly: 12 muls + 4 sqrs + ~7 adds + ~3 subs
    // This simulates a similar pattern
    {
        constexpr int ITERS = 50000;

        // 4×64 version
        double ns4 = bench_ns([&]() {
            FieldElement t0 = fe_a, t1 = fe_b;
            // Simulate point-add field ops
            FieldElement u1 = t0 * t1;
            FieldElement u2 = t1.square();
            FieldElement s1 = u1 + u2;
            FieldElement s2 = t0.square();
            FieldElement h  = s1 + s2 + u1;
            FieldElement r  = h * s1;
            FieldElement rx = r.square() + (FieldElement{} - h);
            FieldElement ry = rx * h + (FieldElement{} - r);
            FieldElement rz = ry.square();
            FieldElement w  = (rz + rx) * ry;
            FieldElement v  = (w + (FieldElement{} - rz)).square();
            FieldElement out = v * w;
            escape(&out);
        }, ITERS);

        // 5×52 version
        double ns5 = bench_ns([&]() {
            FieldElement52 t0 = fe52_a, t1 = fe52_b;
            FieldElement52 u1 = t0 * t1;
            FieldElement52 u2 = t1.square();
            FieldElement52 s1 = u1 + u2;          // lazy add
            FieldElement52 s2 = t0.square();
            FieldElement52 h  = s1 + s2 + u1;     // lazy chain (2 adds, no carry)
            FieldElement52 r  = h * s1;            // mul normalizes
            FieldElement52 rx = r.square() + h.negate(1);       // lazy
            FieldElement52 ry = rx * h + r.negate(1);           // lazy + mul
            FieldElement52 rz = ry.square();
            FieldElement52 w  = (rz + rx) * ry;                 // lazy + mul
            FieldElement52 v  = (w + rz.negate(1)).square();
            FieldElement52 out = v * w;
            escape(&out);
        }, ITERS);

        print_result({"Point-Add simulation (12M+4S+7A)", ns4, ns5});
    }

    // ── 10. Repeated squaring chain (scalar field inversion pattern) ─────────
    {
        constexpr int ITERS = 10000;
        constexpr int CHAIN = 256;   // ~256 squarings like in Fermat inverse

        double ns4 = bench_ns([&]() {
            FieldElement acc = fe_a;
            for (int i = 0; i < CHAIN; ++i) {
                acc.square_inplace();
            }
            escape(&acc);
        }, ITERS);

        double ns5 = bench_ns([&]() {
            FieldElement52 acc = fe52_a;
            for (int i = 0; i < CHAIN; ++i) {
                acc.square_inplace();
            }
            escape(&acc);
        }, ITERS);

        print_result({"Sqr chain (256 squarings)", ns4, ns5});
    }

    // ── 11. Alternating mul+add (mixed pipeline) ────────────────────────────
    {
        constexpr int ITERS = 20000;
        constexpr int CHAIN = 32;

        double ns4 = bench_ns([&]() {
            FieldElement acc = fe_a;
            for (int i = 0; i < CHAIN; ++i) {
                acc = acc * fe_b;
                acc = acc + fe_a;
            }
            escape(&acc);
        }, ITERS);

        double ns5 = bench_ns([&]() {
            FieldElement52 acc = fe52_a;
            for (int i = 0; i < CHAIN; ++i) {
                acc.mul_assign(fe52_b);
                acc.add_assign(fe52_a);     // lazy - no normalize needed
            }
            acc.normalize_weak();
            escape(&acc);
        }, ITERS);

        print_result({"Mul+Add alternating (32 iters)", ns4, ns5});
    }

    // ── 12. Pure multiplication chain ────────────────────────────────────────
    {
        constexpr int ITERS = 20000;
        constexpr int CHAIN = 32;

        double ns4 = bench_ns([&]() {
            FieldElement acc = fe_a;
            for (int i = 0; i < CHAIN; ++i) {
                acc *= fe_b;
            }
            escape(&acc);
        }, ITERS);

        double ns5 = bench_ns([&]() {
            FieldElement52 acc = fe52_a;
            for (int i = 0; i < CHAIN; ++i) {
                acc.mul_assign(fe52_b);
            }
            escape(&acc);
        }, ITERS);

        print_result({"Mul chain (32 muls)", ns4, ns5});
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Throughput (ops/sec)
    // ═════════════════════════════════════════════════════════════════════════

    print_separator("Throughput Summary");
    {
        constexpr int ITERS = 500000;

        // Mul throughput
        double mul4_ns = bench_ns([&]() {
            FieldElement r = fe_a * fe_b;
            escape(&r);
        }, ITERS);

        double mul5_ns = bench_ns([&]() {
            FieldElement52 r = fe52_a * fe52_b;
            escape(&r);
        }, ITERS);

        double mul4_mops = 1000.0 / mul4_ns;
        double mul5_mops = 1000.0 / mul5_ns;

        std::printf("%-36s %7.2f M/s  %7.2f M/s  %6.3fx\n",
                    "Multiplication throughput",
                    mul4_mops, mul5_mops, mul5_mops / mul4_mops);

        // Add throughput
        double add4_ns = bench_ns([&]() {
            FieldElement r = fe_a + fe_b;
            escape(&r);
        }, ITERS);

        double add5_ns = bench_ns([&]() {
            FieldElement52 r = fe52_a + fe52_b;
            escape(&r);
        }, ITERS);

        double add4_mops = 1000.0 / add4_ns;
        double add5_mops = 1000.0 / add5_ns;

        std::printf("%-36s %7.2f M/s  %7.2f M/s  %6.3fx\n",
                    "Addition throughput",
                    add4_mops, add5_mops, add5_mops / add4_mops);

        // Sqr throughput
        double sqr4_ns = bench_ns([&]() {
            FieldElement r = fe_a.square();
            escape(&r);
        }, ITERS);

        double sqr5_ns = bench_ns([&]() {
            FieldElement52 r = fe52_a.square();
            escape(&r);
        }, ITERS);

        double sqr4_mops = 1000.0 / sqr4_ns;
        double sqr5_mops = 1000.0 / sqr5_ns;

        std::printf("%-36s %7.2f M/s  %7.2f M/s  %6.3fx\n",
                    "Squaring throughput",
                    sqr4_mops, sqr5_mops, sqr5_mops / sqr4_mops);
    }

    std::printf("\n=================================================================\n");
    std::printf("  Legend: Ratio = 4x64_time / 5x52_time  (>1 = 5x52 faster)\n");
    std::printf("  5x52 advantage: add chains (lazy), fewer carries\n");
    std::printf("  4x64 advantage: fewer limbs for mul (4 vs 5), no conversion\n");
    std::printf("=================================================================\n\n");

    return 0;
}
