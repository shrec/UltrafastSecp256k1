// ============================================================================
// Benchmark: FieldElement (4x64) vs FieldElement26 (10x26 Lazy-Reduction)
// ============================================================================
//
// Measures individual field operations and realistic ECC-like chains
// to quantify the lazy-reduction advantage of the 10x26 representation.
//
// Key hypothesis: 10x26 wins on add-heavy chains because
// additions are 10 plain adds with NO carry propagation, while 4x64 needs
// carry propagation + conditional reduction after every add.
//
// Target platforms: ESP32 (Xtensa LX6/LX7), STM32 (Cortex-M3/M4),
//                   any 32-bit CPU where uint64_t multiply is expensive.
//
// On x86-64 this benchmark shows the OVERHEAD of 10x26 vs 4x64 (native),
// but on 32-bit targets the relative advantage should be significant
// because 10x26 needs only 32x32->64 multiplies (native on 32-bit).
// ============================================================================

// NOLINTBEGIN(clang-analyzer-core.StackAddressEscape) -- escape() is a benchmark idiom

#include "secp256k1/field_26.hpp"
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

// -- Benchmark Harness --------------------------------------------------------

static constexpr int WARMUP   = 500;
static constexpr int PASSES   = 7;       // median-of-7 for stability

template<typename Func>
double bench_ns(Func&& f, int iterations) {
    // Warmup
    for (int i = 0; i < WARMUP; ++i) f();

    std::array<double, PASSES> runs;
    for (std::size_t p = 0; p < PASSES; ++p) {
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

// Prevent dead-code elimination (standard benchmark anti-optimization pattern).
// Intentionally leaks stack addresses into a volatile sink to defeat DCE.
static volatile uint64_t sink;

static void escape(const void* p) {
    sink = reinterpret_cast<uintptr_t>(p);
}

// -- Test Data ----------------------------------------------------------------

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

// -- Formatting ---------------------------------------------------------------

struct BenchResult {
    const char* name;
    double ns_4x64;
    double ns_10x26;
};

static void print_header() {
    (void)std::printf("\n");
    (void)std::printf("===================================================================\n");
    (void)std::printf("  FieldElement (4x64) vs FieldElement26 (10x26) Benchmark\n");
    (void)std::printf("  Median of %d passes, %d warmup iterations\n", PASSES, WARMUP);
    (void)std::printf("===================================================================\n\n");
    (void)std::printf("%-36s %10s %10s %10s\n", "Operation", "4x64 (ns)", "10x26(ns)", "Ratio");
    (void)std::printf("%-36s %10s %10s %10s\n", "------------------------------------",
                "----------", "----------", "----------");
}

static void print_result(const BenchResult& r) {
    double const ratio = r.ns_4x64 / r.ns_10x26;
    const char* indicator = (ratio > 1.05) ? " <-- 10x26 wins" :
                            (ratio < 0.95) ? " <-- 4x64 wins" : "";
    (void)std::printf("%-36s %9.2f  %9.2f  %8.3fx%s\n",
                r.name, r.ns_4x64, r.ns_10x26, ratio, indicator);
}

static void print_separator(const char* section) {
    (void)std::printf("\n--- %s ---\n", section);
}

// ===========================================================================
// Old Reference Benchmarks (from docs/BENCHMARKS.md)
// ===========================================================================
static void print_reference_table() {
    (void)std::printf("\n");
    (void)std::printf("===================================================================\n");
    (void)std::printf("  Reference: Old Benchmark Data (docs/BENCHMARKS.md)\n");
    (void)std::printf("===================================================================\n");
    (void)std::printf("  Platform               Field Mul (ns)  Field Add (ns)\n");
    (void)std::printf("  ---------------------  --------------  --------------\n");
    (void)std::printf("  x86-64 (i7-13700K)           33              11\n");
    (void)std::printf("  ARM64  (RK3588)              85              18\n");
    (void)std::printf("  RISC-V 64 (D1)              198              34\n");
    (void)std::printf("  ESP32-S3 (LX7)            7,458             636\n");
    (void)std::printf("  ESP32-PICO (LX6)          6,993             985\n");
    (void)std::printf("  STM32F103 (CM3)          15,331           4,139\n");
    (void)std::printf("===================================================================\n\n");
}

// ===========================================================================
// main
// ===========================================================================

int main() {
    (void)std::printf("[bench_field_26] Running arithmetic validation...\n");
    secp256k1::fast::Selftest(false);
    (void)std::printf("[bench_field_26] Validation OK\n");

    // Print reference from old benchmarks
    print_reference_table();

#if defined(_WIN32)
    // Pin to CPU 0 and elevate priority for stable timings
    SetThreadAffinityMask(GetCurrentThread(), 1ULL);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
#endif

    // -- Prepare test data ----------------------------------------------------
    FieldElement fe_a = make_fe(GX_LIMBS);
    FieldElement fe_b = make_fe(GY_LIMBS);

    FieldElement26 fe26_a = FieldElement26::from_fe(fe_a);
    FieldElement26 fe26_b = FieldElement26::from_fe(fe_b);

    print_header();

    // ======================================================================
    // Section 1: Single Operations
    // ======================================================================
    print_separator("Single Operations");

    // -- 1. Single Addition -----------------------------------------------
    {
        constexpr int ITERS = 500000;

        FieldElement r4;
        double const ns4 = bench_ns([&]() {
            r4 = fe_a + fe_b;
            escape(&r4);
        }, ITERS);

        FieldElement26 r26;
        double const ns26 = bench_ns([&]() {
            r26 = fe26_a + fe26_b;
            escape(&r26);
        }, ITERS);

        print_result({"Addition (single)", ns4, ns26});
    }

    // -- 2. Single Multiplication -----------------------------------------
    {
        constexpr int ITERS = 200000;

        FieldElement r4;
        double const ns4 = bench_ns([&]() {
            r4 = fe_a * fe_b;
            escape(&r4);
        }, ITERS);

        FieldElement26 r26;
        double const ns26 = bench_ns([&]() {
            r26 = fe26_a * fe26_b;
            escape(&r26);
        }, ITERS);

        print_result({"Multiplication (single)", ns4, ns26});
    }

    // -- 3. Single Squaring -----------------------------------------------
    {
        constexpr int ITERS = 200000;

        FieldElement r4;
        double const ns4 = bench_ns([&]() {
            r4 = fe_a.square();
            escape(&r4);
        }, ITERS);

        FieldElement26 r26;
        double const ns26 = bench_ns([&]() {
            r26 = fe26_a.square();
            escape(&r26);
        }, ITERS);

        print_result({"Squaring (single)", ns4, ns26});
    }

    // -- 4. Normalization -------------------------------------------------
    {
        constexpr int ITERS = 500000;

        // Create slightly un-normalized element for 10x26
        FieldElement26 acc26 = fe26_a + fe26_b;

        double const ns26_weak = bench_ns([&]() {
            FieldElement26 tmp = acc26;
            tmp.normalize_weak();
            escape(&tmp);
        }, ITERS);

        double const ns26_full = bench_ns([&]() {
            FieldElement26 tmp = acc26;
            tmp.normalize();
            escape(&tmp);
        }, ITERS);

        (void)std::printf("%-36s %9s  %9.2f\n", "Normalize (weak, 10x26 only)", "N/A", ns26_weak);
        (void)std::printf("%-36s %9s  %9.2f\n", "Normalize (full, 10x26 only)", "N/A", ns26_full);
    }

    // -- 5. Negation ------------------------------------------------------
    {
        constexpr int ITERS = 500000;

        FieldElement r4;
        double const ns4 = bench_ns([&]() {
            r4 = FieldElement{} - fe_a;
            escape(&r4);
        }, ITERS);

        FieldElement26 r26;
        double const ns26 = bench_ns([&]() {
            r26 = fe26_a.negate(1);
            escape(&r26);
        }, ITERS);

        print_result({"Negation", ns4, ns26});
    }

    // -- 6. Half ----------------------------------------------------------
    {
        constexpr int ITERS = 500000;

        FieldElement26 r26;
        double const ns26 = bench_ns([&]() {
            r26 = fe26_a.half();
            escape(&r26);
        }, ITERS);

        (void)std::printf("%-36s %9s  %9.2f\n", "Half (10x26 only)", "N/A", ns26);
    }

    // -- 7. Conversion cost -----------------------------------------------
    {
        constexpr int ITERS = 500000;

        FieldElement26 r26;
        double const ns_to26 = bench_ns([&]() {
            r26 = FieldElement26::from_fe(fe_a);
            escape(&r26);
        }, ITERS);

        FieldElement r4;
        double const ns_to4 = bench_ns([&]() {
            r4 = fe26_a.to_fe();
            escape(&r4);
        }, ITERS);

        (void)std::printf("%-36s %9s  %9.2f\n", "Convert 4x64 -> 10x26", "N/A", ns_to26);
        (void)std::printf("%-36s %9.2f  %9s\n", "Convert 10x26 -> 4x64", ns_to4, "N/A");
    }

    // ======================================================================
    // Section 2: Addition Chains (Lazy-Reduction Core Advantage)
    // ======================================================================
    print_separator("Addition Chains (Lazy-Reduction Core Advantage)");

    for (int chain_len : {4, 8, 16, 32, 64}) {
        constexpr int ITERS = 50000;

        // 4x64: each add does carry + conditional reduction
        double const ns4 = bench_ns([&]() {
            FieldElement acc = fe_a;
            for (int i = 0; i < chain_len; ++i) {
                acc = acc + fe_b;
            }
            escape(&acc);
        }, ITERS);

        // 10x26: N plain adds, ONE normalize at end
        double const ns26 = bench_ns([&]() {
            FieldElement26 acc = fe26_a;
            for (int i = 0; i < chain_len; ++i) {
                acc.add_assign(fe26_b);
            }
            acc.normalize_weak();
            escape(&acc);
        }, ITERS);

        char name[64];
        (void)std::snprintf(name, sizeof(name), "Add chain (%d adds + norm)", chain_len);
        print_result({name, ns4, ns26});
    }

    // ======================================================================
    // Section 3: Mixed Chains (ECC Point Operation Simulation)
    // ======================================================================
    print_separator("Mixed Chains (ECC Point Operation Simulation)");

    // ECC point addition: ~12 muls + 4 sqrs + ~7 adds + ~3 subs
    {
        constexpr int ITERS = 50000;

        // 4x64 version
        double const ns4 = bench_ns([&]() {
            FieldElement const t0 = fe_a, t1 = fe_b;
            FieldElement const u1 = t0 * t1;
            FieldElement const u2 = t1.square();
            FieldElement const s1 = u1 + u2;
            FieldElement const s2 = t0.square();
            FieldElement const h  = s1 + s2 + u1;
            FieldElement const r  = h * s1;
            FieldElement const rx = r.square() + (FieldElement{} - h);
            FieldElement const ry = rx * h + (FieldElement{} - r);
            FieldElement const rz = ry.square();
            FieldElement const w  = (rz + rx) * ry;
            FieldElement const v  = (w + (FieldElement{} - rz)).square();
            FieldElement out = v * w;
            escape(&out);
        }, ITERS);

        // 10x26 version
        double const ns26 = bench_ns([&]() {
            FieldElement26 const t0 = fe26_a, t1 = fe26_b;
            FieldElement26 const u1 = t0 * t1;
            FieldElement26 const u2 = t1.square();
            FieldElement26 const s1 = u1 + u2;          // lazy add
            FieldElement26 const s2 = t0.square();
            FieldElement26 const h  = s1 + s2 + u1;     // lazy chain (2 adds, no carry)
            FieldElement26 const r  = h * s1;            // mul normalizes
            FieldElement26 const rx = r.square() + h.negate(1);       // lazy
            FieldElement26 const ry = rx * h + r.negate(1);           // lazy + mul
            FieldElement26 const rz = ry.square();
            FieldElement26 const w  = (rz + rx) * ry;                 // lazy + mul
            FieldElement26 const v  = (w + rz.negate(1)).square();
            FieldElement26 out = v * w;
            escape(&out);
        }, ITERS);

        print_result({"Point-Add simulation (12M+4S+7A)", ns4, ns26});
    }

    // -- Repeated squaring chain (scalar field inversion pattern) ---------
    {
        constexpr int ITERS = 10000;
        constexpr int CHAIN = 256;   // ~256 squarings like in Fermat inverse

        double const ns4 = bench_ns([&]() {
            FieldElement acc = fe_a;
            for (int i = 0; i < CHAIN; ++i) {
                acc.square_inplace();
            }
            escape(&acc);
        }, ITERS);

        double const ns26 = bench_ns([&]() {
            FieldElement26 acc = fe26_a;
            for (int i = 0; i < CHAIN; ++i) {
                acc.square_inplace();
            }
            escape(&acc);
        }, ITERS);

        print_result({"Sqr chain (256 squarings)", ns4, ns26});
    }

    // -- Alternating mul+add (mixed pipeline) ----------------------------
    {
        constexpr int ITERS = 20000;
        constexpr int CHAIN = 32;

        double const ns4 = bench_ns([&]() {
            FieldElement acc = fe_a;
            for (int i = 0; i < CHAIN; ++i) {
                acc = acc * fe_b;
                acc = acc + fe_a;
            }
            escape(&acc);
        }, ITERS);

        double const ns26 = bench_ns([&]() {
            FieldElement26 acc = fe26_a;
            for (int i = 0; i < CHAIN; ++i) {
                acc.mul_assign(fe26_b);
                acc.add_assign(fe26_a);     // lazy - no normalize needed
            }
            acc.normalize_weak();
            escape(&acc);
        }, ITERS);

        print_result({"Mul+Add alternating (32 iters)", ns4, ns26});
    }

    // -- Pure multiplication chain ----------------------------------------
    {
        constexpr int ITERS = 20000;
        constexpr int CHAIN = 32;

        double const ns4 = bench_ns([&]() {
            FieldElement acc = fe_a;
            for (int i = 0; i < CHAIN; ++i) {
                acc *= fe_b;
            }
            escape(&acc);
        }, ITERS);

        double const ns26 = bench_ns([&]() {
            FieldElement26 acc = fe26_a;
            for (int i = 0; i < CHAIN; ++i) {
                acc.mul_assign(fe26_b);
            }
            escape(&acc);
        }, ITERS);

        print_result({"Mul chain (32 muls)", ns4, ns26});
    }

    // ======================================================================
    // Section 4: Throughput Summary
    // ======================================================================
    print_separator("Throughput Summary");
    {
        constexpr int ITERS = 500000;

        // Mul throughput
        double const mul4_ns = bench_ns([&]() {
            FieldElement r = fe_a * fe_b;
            escape(&r);
        }, ITERS);

        double const mul26_ns = bench_ns([&]() {
            FieldElement26 r = fe26_a * fe26_b;
            escape(&r);
        }, ITERS);

        double const mul4_mops  = 1000.0 / mul4_ns;
        double const mul26_mops = 1000.0 / mul26_ns;

        (void)std::printf("%-36s %7.2f M/s  %7.2f M/s  %6.3fx\n",
                    "Multiplication throughput",
                    mul4_mops, mul26_mops, mul26_mops / mul4_mops);

        // Add throughput
        double const add4_ns = bench_ns([&]() {
            FieldElement r = fe_a + fe_b;
            escape(&r);
        }, ITERS);

        double const add26_ns = bench_ns([&]() {
            FieldElement26 r = fe26_a + fe26_b;
            escape(&r);
        }, ITERS);

        double const add4_mops  = 1000.0 / add4_ns;
        double const add26_mops = 1000.0 / add26_ns;

        (void)std::printf("%-36s %7.2f M/s  %7.2f M/s  %6.3fx\n",
                    "Addition throughput",
                    add4_mops, add26_mops, add26_mops / add4_mops);

        // Sqr throughput
        double const sqr4_ns = bench_ns([&]() {
            FieldElement r = fe_a.square();
            escape(&r);
        }, ITERS);

        double const sqr26_ns = bench_ns([&]() {
            FieldElement26 r = fe26_a.square();
            escape(&r);
        }, ITERS);

        double const sqr4_mops  = 1000.0 / sqr4_ns;
        double const sqr26_mops = 1000.0 / sqr26_ns;

        (void)std::printf("%-36s %7.2f M/s  %7.2f M/s  %6.3fx\n",
                    "Squaring throughput",
                    sqr4_mops, sqr26_mops, sqr26_mops / sqr4_mops);
    }

    // ======================================================================
    // Section 5: Platform Context
    // ======================================================================
    (void)std::printf("\n===================================================================\n");
    (void)std::printf("  Legend: Ratio = 4x64_time / 10x26_time  (>1 = 10x26 faster)\n");
    (void)std::printf("\n");
    (void)std::printf("  On x86-64:  4x64 uses native 64-bit ops -> 4x64 expected to win.\n");
    (void)std::printf("              10x26 loses here due to more limbs (10 vs 4).\n");
    (void)std::printf("\n");
    (void)std::printf("  On 32-bit:  10x26 uses only 32x32->64 multiplies (native).\n");
    (void)std::printf("              4x64 needs 64-bit math emulated as 4x calls.\n");
    (void)std::printf("              10x26 wins significantly on ESP32/STM32/ARM32.\n");
    (void)std::printf("\n");
    (void)std::printf("  Lazy-reduction advantage: add chains ALWAYS benefit regardless\n");
    (void)std::printf("  of platform, since 10x26 adds are 10 plain uint32 adds.\n");
    (void)std::printf("===================================================================\n\n");

    return 0;
}

// NOLINTEND(clang-analyzer-core.StackAddressEscape)
