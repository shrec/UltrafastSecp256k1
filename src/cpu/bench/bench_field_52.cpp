// ============================================================================
// Benchmark: FieldElement (4x64) vs FieldElement52 (5x52 Lazy-Reduction)
// ============================================================================
//
// Measures individual field operations and realistic ECC-like chains
// to quantify the lazy-reduction advantage of the 5x52 representation.
//
// Key hypothesis: 5x52 wins on add-heavy chains (ECC point ops) because
// additions are 5 plain adds with NO carry propagation, while 4x64 needs
// carry propagation + conditional reduction after every add.
// ============================================================================

#include "secp256k1/field_52.hpp"
#include "secp256k1/field.hpp"
#include "secp256k1/selftest.hpp"
#include "secp256k1/benchmark_harness.hpp"

#include <cstdio>
#include <cstdint>

using namespace secp256k1::fast;

// Unified harness: 500 warmup, 11 passes, RDTSC on x86, IQR outlier removal
static bench::Harness H(500, 11);

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

// -- Individual Operation Benchmarks ------------------------------------------

struct BenchResult {
    const char* name;
    double ns_4x64;
    double ns_5x52;
};

static void print_header() {
    (void)std::printf("\n");
    (void)std::printf("=================================================================\n");
    (void)std::printf("  FieldElement (4x64) vs FieldElement52 (5x52) Benchmark\n");
    (void)std::printf("=================================================================\n");
    H.print_config();
    (void)std::printf("\n%-36s %10s %10s %10s\n", "Operation", "4x64 (ns)", "5x52 (ns)", "Ratio");
    (void)std::printf("%-36s %10s %10s %10s\n", "------------------------------------",
                "----------", "----------", "----------");
}

static void print_result(const BenchResult& r) {
    double const ratio = r.ns_4x64 / r.ns_5x52;
    const char* indicator = (ratio > 1.05) ? " <-- 5x52 wins" :
                            (ratio < 0.95) ? " <-- 4x64 wins" : "";
    (void)std::printf("%-36s %9.2f  %9.2f  %8.3fx%s\n",
                r.name, r.ns_4x64, r.ns_5x52, ratio, indicator);
}

static void print_separator(const char* section) {
    (void)std::printf("\n--- %s ---\n", section);
}

// =============================================================================
// main
// =============================================================================

int main() {
    (void)std::printf("[bench_field_52] Running arithmetic validation...\n");
    secp256k1::fast::Selftest(false);
    (void)std::printf("[bench_field_52] Validation OK\n");

    bench::pin_thread_and_elevate();

    // -- Prepare test data ----------------------------------------------------
    FieldElement fe_a = make_fe(GX_LIMBS);
    FieldElement fe_b = make_fe(GY_LIMBS);

    FieldElement52 fe52_a = FieldElement52::from_fe(fe_a);
    FieldElement52 fe52_b = FieldElement52::from_fe(fe_b);

    print_header();

    // -- 1. Single Addition ---------------------------------------------------
    print_separator("Single Operations");
    {
        constexpr int ITERS = 500000;

        FieldElement r4;
        double const ns4 = H.run(ITERS, [&]() {
            r4 = fe_a + fe_b;
            bench::DoNotOptimize(r4);
        });

        FieldElement52 r5;
        double const ns5 = H.run(ITERS, [&]() {
            r5 = fe52_a + fe52_b;
            bench::DoNotOptimize(r5);
        });

        print_result({"Addition (single)", ns4, ns5});
    }

    // -- 2. Single Multiplication ---------------------------------------------
    {
        constexpr int ITERS = 200000;

        FieldElement r4;
        double const ns4 = H.run(ITERS, [&]() {
            r4 = fe_a * fe_b;
            bench::DoNotOptimize(r4);
        });

        FieldElement52 r5;
        double const ns5 = H.run(ITERS, [&]() {
            r5 = fe52_a * fe52_b;
            bench::DoNotOptimize(r5);
        });

        print_result({"Multiplication (single)", ns4, ns5});
    }

    // -- 3. Single Squaring ---------------------------------------------------
    {
        constexpr int ITERS = 200000;

        FieldElement r4;
        double const ns4 = H.run(ITERS, [&]() {
            r4 = fe_a.square();
            bench::DoNotOptimize(r4);
        });

        FieldElement52 r5;
        double const ns5 = H.run(ITERS, [&]() {
            r5 = fe52_a.square();
            bench::DoNotOptimize(r5);
        });

        print_result({"Squaring (single)", ns4, ns5});
    }

    // -- 4. Normalization -----------------------------------------------------
    {
        constexpr int ITERS = 500000;

        // Create slightly un-normalized element for 5x52
        FieldElement52 acc52 = fe52_a + fe52_b;

        double const ns5_weak = H.run(ITERS, [&]() {
            FieldElement52 tmp = acc52;
            tmp.normalize_weak();
            bench::DoNotOptimize(tmp);
        });

        double const ns5_full = H.run(ITERS, [&]() {
            FieldElement52 tmp = acc52;
            tmp.normalize();
            bench::DoNotOptimize(tmp);
        });

        (void)std::printf("%-36s %9s  %9.2f\n", "Normalize (weak, 5x52 only)", "N/A", ns5_weak);
        (void)std::printf("%-36s %9s  %9.2f\n", "Normalize (full, 5x52 only)", "N/A", ns5_full);
    }

    // -- 5. Negation ----------------------------------------------------------
    {
        constexpr int ITERS = 500000;

        FieldElement r4;
        double const ns4 = H.run(ITERS, [&]() {
            r4 = FieldElement{} - fe_a;
            bench::DoNotOptimize(r4);
        });

        FieldElement52 r5;
        double const ns5 = H.run(ITERS, [&]() {
            r5 = fe52_a.negate(1);
            bench::DoNotOptimize(r5);
        });

        print_result({"Negation", ns4, ns5});
    }

    // -- 6. Half --------------------------------------------------------------
    {
        constexpr int ITERS = 500000;

        FieldElement52 r5;
        double const ns5 = H.run(ITERS, [&]() {
            r5 = fe52_a.half();
            bench::DoNotOptimize(r5);
        });

        (void)std::printf("%-36s %9s  %9.2f\n", "Half (5x52 only)", "N/A", ns5);
    }

    // -- 7. Conversion cost ---------------------------------------------------
    {
        constexpr int ITERS = 500000;

        FieldElement52 r52;
        double const ns_to52 = H.run(ITERS, [&]() {
            r52 = FieldElement52::from_fe(fe_a);
            bench::DoNotOptimize(r52);
        });

        FieldElement r4;
        double const ns_to4 = H.run(ITERS, [&]() {
            r4 = fe52_a.to_fe();
            bench::DoNotOptimize(r4);
        });

        (void)std::printf("%-36s %9s  %9.2f\n", "Convert 4x64 -> 5x52", "N/A", ns_to52);
        (void)std::printf("%-36s %9.2f  %9s\n", "Convert 5x52 -> 4x64", ns_to4, "N/A");
    }

    // =========================================================================
    // Chained Operations (realistic ECC workload simulation)
    // =========================================================================

    print_separator("Addition Chains (Lazy-Reduction Core Advantage)");

    // -- 8. Chain of N additions (the whole point of lazy reduction) ----------
    for (int chain_len : {4, 8, 16, 32, 64}) {
        constexpr int ITERS = 50000;

        // 4x64: each add does carry + conditional reduction
        double const ns4 = H.run(ITERS, [&]() {
            FieldElement acc = fe_a;
            for (int i = 0; i < chain_len; ++i) {
                acc = acc + fe_b;
            }
            bench::DoNotOptimize(acc);
        });

        // 5x52: N plain adds, ONE normalize at end
        double const ns5 = H.run(ITERS, [&]() {
            FieldElement52 acc = fe52_a;
            for (int i = 0; i < chain_len; ++i) {
                acc.add_assign(fe52_b);
            }
            acc.normalize_weak();
            bench::DoNotOptimize(acc);
        });

        char name[64];
        (void)std::snprintf(name, sizeof(name), "Add chain (%d adds + norm)", chain_len);
        print_result({name, ns4, ns5});
    }

    // -- 9. Mixed add+mul chain (simulates point addition field ops) ----------
    print_separator("Mixed Chains (ECC Point Operation Simulation)");

    // ECC point addition has roughly: 12 muls + 4 sqrs + ~7 adds + ~3 subs
    // This simulates a similar pattern
    {
        constexpr int ITERS = 50000;

        // 4x64 version
        double const ns4 = H.run(ITERS, [&]() {
            FieldElement const t0 = fe_a, t1 = fe_b;
            // Simulate point-add field ops
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
            bench::DoNotOptimize(out);
        });

        // 5x52 version
        double const ns5 = H.run(ITERS, [&]() {
            FieldElement52 const t0 = fe52_a, t1 = fe52_b;
            FieldElement52 const u1 = t0 * t1;
            FieldElement52 const u2 = t1.square();
            FieldElement52 const s1 = u1 + u2;          // lazy add
            FieldElement52 const s2 = t0.square();
            FieldElement52 const h  = s1 + s2 + u1;     // lazy chain (2 adds, no carry)
            FieldElement52 const r  = h * s1;            // mul normalizes
            FieldElement52 const rx = r.square() + h.negate(1);       // lazy
            FieldElement52 const ry = rx * h + r.negate(1);           // lazy + mul
            FieldElement52 const rz = ry.square();
            FieldElement52 const w  = (rz + rx) * ry;                 // lazy + mul
            FieldElement52 const v  = (w + rz.negate(1)).square();
            FieldElement52 out = v * w;
            bench::DoNotOptimize(out);
        });

        print_result({"Point-Add simulation (12M+4S+7A)", ns4, ns5});
    }

    // -- 10. Repeated squaring chain (scalar field inversion pattern) ---------
    {
        constexpr int ITERS = 10000;
        constexpr int CHAIN = 256;   // ~256 squarings like in Fermat inverse

        double const ns4 = H.run(ITERS, [&]() {
            FieldElement acc = fe_a;
            for (int i = 0; i < CHAIN; ++i) {
                acc.square_inplace();
            }
            bench::DoNotOptimize(acc);
        });

        double const ns5 = H.run(ITERS, [&]() {
            FieldElement52 acc = fe52_a;
            for (int i = 0; i < CHAIN; ++i) {
                acc.square_inplace();
            }
            bench::DoNotOptimize(acc);
        });

        print_result({"Sqr chain (256 squarings)", ns4, ns5});
    }

    // -- 11. Alternating mul+add (mixed pipeline) ----------------------------
    {
        constexpr int ITERS = 20000;
        constexpr int CHAIN = 32;

        double const ns4 = H.run(ITERS, [&]() {
            FieldElement acc = fe_a;
            for (int i = 0; i < CHAIN; ++i) {
                acc = acc * fe_b;
                acc = acc + fe_a;
            }
            bench::DoNotOptimize(acc);
        });

        double const ns5 = H.run(ITERS, [&]() {
            FieldElement52 acc = fe52_a;
            for (int i = 0; i < CHAIN; ++i) {
                acc.mul_assign(fe52_b);
                acc.add_assign(fe52_a);     // lazy - no normalize needed
            }
            acc.normalize_weak();
            bench::DoNotOptimize(acc);
        });

        print_result({"Mul+Add alternating (32 iters)", ns4, ns5});
    }

    // -- 12. Pure multiplication chain ----------------------------------------
    {
        constexpr int ITERS = 20000;
        constexpr int CHAIN = 32;

        double const ns4 = H.run(ITERS, [&]() {
            FieldElement acc = fe_a;
            for (int i = 0; i < CHAIN; ++i) {
                acc *= fe_b;
            }
            bench::DoNotOptimize(acc);
        });

        double const ns5 = H.run(ITERS, [&]() {
            FieldElement52 acc = fe52_a;
            for (int i = 0; i < CHAIN; ++i) {
                acc.mul_assign(fe52_b);
            }
            bench::DoNotOptimize(acc);
        });

        print_result({"Mul chain (32 muls)", ns4, ns5});
    }

    // =========================================================================
    // Throughput (ops/sec)
    // =========================================================================

    print_separator("Throughput Summary");
    {
        constexpr int ITERS = 500000;

        // Mul throughput
        double const mul4_ns = H.run(ITERS, [&]() {
            FieldElement r = fe_a * fe_b;
            bench::DoNotOptimize(r);
        });

        double const mul5_ns = H.run(ITERS, [&]() {
            FieldElement52 r = fe52_a * fe52_b;
            bench::DoNotOptimize(r);
        });

        double const mul4_mops = 1000.0 / mul4_ns;
        double const mul5_mops = 1000.0 / mul5_ns;

        (void)std::printf("%-36s %7.2f M/s  %7.2f M/s  %6.3fx\n",
                    "Multiplication throughput",
                    mul4_mops, mul5_mops, mul5_mops / mul4_mops);

        // Add throughput
        double const add4_ns = H.run(ITERS, [&]() {
            FieldElement r = fe_a + fe_b;
            bench::DoNotOptimize(r);
        });

        double const add5_ns = H.run(ITERS, [&]() {
            FieldElement52 r = fe52_a + fe52_b;
            bench::DoNotOptimize(r);
        });

        double const add4_mops = 1000.0 / add4_ns;
        double const add5_mops = 1000.0 / add5_ns;

        (void)std::printf("%-36s %7.2f M/s  %7.2f M/s  %6.3fx\n",
                    "Addition throughput",
                    add4_mops, add5_mops, add5_mops / add4_mops);

        // Sqr throughput
        double const sqr4_ns = H.run(ITERS, [&]() {
            FieldElement r = fe_a.square();
            bench::DoNotOptimize(r);
        });

        double const sqr5_ns = H.run(ITERS, [&]() {
            FieldElement52 r = fe52_a.square();
            bench::DoNotOptimize(r);
        });

        double const sqr4_mops = 1000.0 / sqr4_ns;
        double const sqr5_mops = 1000.0 / sqr5_ns;

        (void)std::printf("%-36s %7.2f M/s  %7.2f M/s  %6.3fx\n",
                    "Squaring throughput",
                    sqr4_mops, sqr5_mops, sqr5_mops / sqr4_mops);
    }

    (void)std::printf("\n=================================================================\n");
    (void)std::printf("  Legend: Ratio = 4x64_time / 5x52_time  (>1 = 5x52 faster)\n");
    (void)std::printf("  5x52 advantage: add chains (lazy), fewer carries\n");
    (void)std::printf("  4x64 advantage: fewer limbs for mul (4 vs 5), no conversion\n");
    (void)std::printf("=================================================================\n\n");

    return 0;
}
