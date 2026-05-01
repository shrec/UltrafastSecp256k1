// ============================================================================
// UltrafastSecp256k1 -- Android ARM64 On-Device Test + Benchmarks
// ============================================================================
// Standalone test binary (no JNI, no Java). Push via adb and run in shell.
// Tests: selftest, point ops, scalar ops, CT ops, ECDH, timing.
// Benchmarks: field ops, point ops, scalar_mul, CT ops, ECDH
// ============================================================================

#include <cstdio>
#include <cstring>
#include <chrono>
#include <array>
#include <algorithm>

#include <secp256k1/field.hpp>
#include <secp256k1/field_26.hpp>
#include <secp256k1/field_asm.hpp>
#include <secp256k1/scalar.hpp>
#include <secp256k1/point.hpp>
#include <secp256k1/init.hpp>
#include <secp256k1/selftest.hpp>

#include <secp256k1/ct/ops.hpp>
#include <secp256k1/ct/field.hpp>
#include <secp256k1/ct/scalar.hpp>
#include <secp256k1/ct/point.hpp>

using namespace secp256k1;
using FE = fast::FieldElement;
using FE26 = fast::FieldElement26;
using SC = fast::Scalar;
using PT = fast::Point;

static void print_hex(const uint8_t* data, size_t len) {
    for (size_t i = 0; i < len; ++i) printf("%02x", data[i]);
}

// Prevent dead-code elimination
static volatile uint64_t g_sink = 0;

template<typename F>
static long long bench(const char* name, int iterations, F&& fn) {
    // Warmup
    for (int i = 0; i < std::max(iterations / 10, 2); ++i) fn();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    long long ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    long long per_op_ns = ns / iterations;
    
    printf("  %-36s %6lld ns/op  (%d ops in %lld us)\n", 
           name, per_op_ns, iterations, ns / 1000);
    return per_op_ns;
}

// ============================================================================
// Correctness Tests
// ============================================================================
static int run_tests() {
    printf("=============================================\n");
    printf("UltrafastSecp256k1 -- Android ARM64 Test\n");
    printf("=============================================\n\n");

    // 1. Self-test
    printf("[1] Self-test... ");
    fflush(stdout);
    bool ok = fast::Selftest(false);
    printf("%s\n", ok ? "PASS" : "FAIL");
    if (!ok) { printf("CRITICAL: Self-test failed!\n"); return 1; }

    // 2. Generator point
    printf("[2] Generator point G... ");
    PT g = PT::generator();
    auto g_uncomp = g.to_uncompressed();
    printf("OK (04");
    print_hex(g_uncomp.data() + 1, 8);
    printf("...)\n");

    // 3. Point doubling: 2G
    printf("[3] Point doubling 2G... ");
    PT g2 = g.dbl();
    auto g2_comp = g2.to_compressed();
    print_hex(g2_comp.data(), 33);
    printf("\n");

    // 4. Point addition: 3G = 2G + G
    printf("[4] Point addition 3G... ");
    PT g3 = g2.add(g);
    auto g3_comp = g3.to_compressed();
    print_hex(g3_comp.data(), 33);
    printf("\n");

    // 5. Scalar mul: k*G (fast)
    printf("[5] Fast scalar_mul k*G... ");
    SC k = SC::from_uint64(12345);
    auto t0 = std::chrono::high_resolution_clock::now();
    PT kg = g.scalar_mul(k);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto fast_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    auto kg_comp = kg.to_compressed();
    print_hex(kg_comp.data(), 33);
    printf(" (%lld us)\n", (long long)fast_us);

    // 6. CT scalar_mul: k*G (constant-time)
    printf("[6] CT scalar_mul k*G... ");
    t0 = std::chrono::high_resolution_clock::now();
    PT ct_kg = ct::scalar_mul(g, k);
    t1 = std::chrono::high_resolution_clock::now();
    auto ct_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    auto ct_comp = ct_kg.to_compressed();
    print_hex(ct_comp.data(), 33);
    printf(" (%lld us)\n", (long long)ct_us);

    // 7. Verify fast == CT
    printf("[7] Fast == CT result... ");
    bool match = (kg_comp == ct_comp);
    printf("%s\n", match ? "MATCH" : "MISMATCH!");
    if (!match) { printf("ERROR: fast and CT results differ!\n"); return 1; }

    // 8. CT generator_mul
    printf("[8] CT generator_mul... ");
    t0 = std::chrono::high_resolution_clock::now();
    PT ct_gen = ct::generator_mul(k);
    t1 = std::chrono::high_resolution_clock::now();
    auto gen_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    auto gen_comp = ct_gen.to_compressed();
    bool gen_match = (gen_comp == kg_comp);
    printf("%s (%lld us)\n", gen_match ? "MATCH" : "MISMATCH!", (long long)gen_us);

    // 9. ECDH test
    printf("[9] CT ECDH... ");
    SC alice_priv = SC::from_uint64(42);
    SC bob_priv = SC::from_uint64(99);
    PT alice_pub = ct::generator_mul(alice_priv);
    PT bob_pub = ct::generator_mul(bob_priv);

    t0 = std::chrono::high_resolution_clock::now();
    PT shared_a = ct::scalar_mul(bob_pub, alice_priv);
    PT shared_b = ct::scalar_mul(alice_pub, bob_priv);
    t1 = std::chrono::high_resolution_clock::now();
    auto ecdh_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    auto xa = shared_a.x().to_bytes();
    auto xb = shared_b.x().to_bytes();
    bool ecdh_ok = (xa == xb);
    printf("%s (%lld us for 2 ECDH)\n", ecdh_ok ? "PASS" : "FAIL!", (long long)ecdh_us);
    if (ecdh_ok) {
        printf("    shared secret: ");
        print_hex(xa.data(), 16);
        printf("...\n");
    }

    // 10. Scalar arithmetic
    printf("[10] Scalar arithmetic... ");
    SC a = SC::from_uint64(0xDEADBEEF);
    SC b = SC::from_uint64(0xCAFEBABE);
    SC sum = a + b;
    SC prod = a * b;
    SC diff = a - b;
    bool scalar_ok = !sum.is_zero() && !prod.is_zero();
    printf("%s\n", scalar_ok ? "PASS" : "FAIL");

    // 11. Field arithmetic verification (ARM64 assembly correctness)
    printf("[11] Field mul/sqr verify... ");
    FE fa = FE::from_uint64(0xDEADBEEF12345678ULL);
    FE fb = FE::from_uint64(0xCAFEBABE87654321ULL);
    FE fc = fa * fb;
    FE fd = fa.square();
    // Verify a * a == a^2
    FE fe = fa * fa;
    bool field_ok = (fd == fe) && !(fc == FE::zero());
    printf("%s\n", field_ok ? "PASS" : "FAIL");
    if (!field_ok) { printf("ERROR: Field arithmetic mismatch!\n"); return 1; }

    // 12. Field inverse verification
    printf("[12] Field inverse verify... ");
    FE fi = fa.inverse();
    FE should_one = fa * fi;
    bool inv_ok = (should_one == FE::one());
    printf("%s\n", inv_ok ? "PASS" : "FAIL");
    if (!inv_ok) { printf("ERROR: Field inverse failed!\n"); return 1; }

    printf("\nAll correctness tests PASSED.\n\n");
    return 0;
}

// ============================================================================
// Benchmarks
// ============================================================================
static void run_benchmarks() {
    printf("=============================================\n");
    printf("BENCHMARKS -- ARM64 (%s)\n", 
#ifdef SECP256K1_HAS_ARM64_ASM
        "ASM optimized"
#else
        "generic C++"
#endif
    );
    printf("=============================================\n\n");

    // Setup test data
    FE fa = FE::from_uint64(0xDEADBEEF12345678ULL);
    FE fb = FE::from_uint64(0xCAFEBABE87654321ULL);
    SC sk = SC::from_uint64(0x123456789ABCDEF0ULL);
    PT g = PT::generator();
    PT p2 = g.dbl();

    // ---- Field Operations ----
    printf("--- Field Operations ---\n");
    
    long long mul_ns = bench("field_mul (a*b mod p)", 100000, [&]() {
        fa = fa * fb;
        g_sink ^= fa.limbs()[0];
    });
    
    long long sqr_ns = bench("field_sqr (a^2 mod p)", 100000, [&]() {
        fa = fa.square();
        g_sink ^= fa.limbs()[0];
    });
    
    fa = FE::from_uint64(0xDEADBEEF12345678ULL);
    long long add_ns = bench("field_add (a+b mod p)", 100000, [&]() {
        fa = fa + fb;
        g_sink ^= fa.limbs()[0];
    });

    fa = FE::from_uint64(0xDEADBEEF12345678ULL);
    long long sub_ns = bench("field_sub (a-b mod p)", 100000, [&]() {
        fa = fa - fb;
        g_sink ^= fa.limbs()[0];
    });

    fa = FE::from_uint64(0xDEADBEEF12345678ULL);
    long long inv_ns = bench("field_inverse (a^(-1) mod p)", 1000, [&]() {
        fa = fa.inverse();
        g_sink ^= fa.limbs()[0];
    });

    printf("\n--- Scalar Operations ---\n");
    
    SC sa = SC::from_uint64(0xDEADBEEF12345678ULL);
    SC sb = SC::from_uint64(0xCAFEBABE87654321ULL);
    
    bench("scalar_mul (a*b mod n)", 100000, [&]() {
        sa = sa * sb;
        g_sink ^= sa.limbs()[0];
    });
    
    bench("scalar_add (a+b mod n)", 100000, [&]() {
        sa = sa + sb;
        g_sink ^= sa.limbs()[0];
    });

    printf("\n--- Point Operations ---\n");
    
    bench("point_dbl (2P)", 10000, [&]() {
        p2 = p2.dbl();
        g_sink ^= p2.x().limbs()[0];
    });
    
    p2 = g.dbl();
    PT p3 = p2;
    bench("point_add (P+Q)", 10000, [&]() {
        p3 = p3.add(g);
        g_sink ^= p3.x().limbs()[0];
    });

    printf("\n--- Scalar Multiplication (Fast) ---\n");
    
    long long fast_mul_ns = bench("scalar_mul (k*G, fast)", 500, [&]() {
        PT r = g.scalar_mul(sk);
        g_sink ^= r.x().limbs()[0];
    });

    bench("scalar_mul (k*P, fast, non-G)", 500, [&]() {
        PT r = p2.scalar_mul(sk);
        g_sink ^= r.x().limbs()[0];
    });

    printf("\n--- Scalar Multiplication (CT) ---\n");
    
    long long ct_mul_ns = bench("ct::scalar_mul (k*G)", 100, [&]() {
        PT r = ct::scalar_mul(g, sk);
        g_sink ^= r.x().limbs()[0];
    });

    bench("ct::generator_mul (k*G)", 100, [&]() {
        PT r = ct::generator_mul(sk);
        g_sink ^= r.x().limbs()[0];
    });

    bench("ct::scalar_mul (k*P, non-G)", 100, [&]() {
        PT r = ct::scalar_mul(p2, sk);
        g_sink ^= r.x().limbs()[0];
    });

    printf("\n--- ECDH ---\n");
    
    SC priv = SC::from_uint64(42);
    PT pub = ct::generator_mul(SC::from_uint64(99));
    
    bench("ECDH (full: ct::scalar_mul)", 100, [&]() {
        PT shared = ct::scalar_mul(pub, priv);
        g_sink ^= shared.x().limbs()[0];
    });

    // ---- Summary ----
    printf("\n=============================================\n");
    printf("SUMMARY -- ARM64 Performance (4x64)\n");
    printf("=============================================\n");
    printf("  field_mul:       %6lld ns/op\n", mul_ns);
    printf("  field_sqr:       %6lld ns/op\n", sqr_ns);
    printf("  field_add:       %6lld ns/op\n", add_ns);
    printf("  field_sub:       %6lld ns/op\n", sub_ns);
    printf("  field_inverse:   %6lld ns/op\n", inv_ns);
    printf("  fast scalar_mul: %6lld ns/op  (%lld us)\n", fast_mul_ns, fast_mul_ns / 1000);
    printf("  CT scalar_mul:   %6lld ns/op  (%lld us)\n", ct_mul_ns, ct_mul_ns / 1000);
    printf("  CT/Fast ratio:   %.1fx\n", (double)ct_mul_ns / fast_mul_ns);
#ifdef SECP256K1_HAS_ARM64_ASM
    printf("  Backend: ARM64 inline assembly (MUL/UMULH)\n");
#else
    printf("  Backend: Generic C++ (portable)\n");
#endif
    printf("=============================================\n");

    // ============================================================================
    // 10x26 Field Element Benchmark (Lazy-Reduction for 32-bit targets)
    // ============================================================================
    printf("\n=============================================\n");
    printf("BENCHMARK -- FieldElement26 (10x26 Lazy-Reduction)\n");
    printf("  4x64 vs 10x26 comparison on ARM64\n");
    printf("=============================================\n\n");

    FE26 fe26_a = FE26::from_fe(FE::from_uint64(0xDEADBEEF12345678ULL));
    FE26 fe26_b = FE26::from_fe(FE::from_uint64(0xCAFEBABE87654321ULL));
    FE fe_ref_a = FE::from_uint64(0xDEADBEEF12345678ULL);
    FE fe_ref_b = FE::from_uint64(0xCAFEBABE87654321ULL);

    // -- Correctness check --
    {
        FE26 prod26 = fe26_a * fe26_b;
        FE prod64 = fe_ref_a * fe_ref_b;
        FE prod26_back = prod26.to_fe();
        bool ok = (prod26_back == prod64);
        printf("  10x26 mul correctness: %s\n", ok ? "PASS" : "FAIL");
        if (!ok) { printf("    ERROR: 10x26 multiplication mismatch on device!\n"); }

        FE26 sqr26 = fe26_a.square();
        FE sqr64 = fe_ref_a.square();
        FE sqr26_back = sqr26.to_fe();
        ok = (sqr26_back == sqr64);
        printf("  10x26 sqr correctness: %s\n", ok ? "PASS" : "FAIL");
        if (!ok) { printf("    ERROR: 10x26 squaring mismatch on device!\n"); }
    }
    printf("\n");

    printf("--- 10x26 Single Operations ---\n");

    long long mul26_ns = bench("10x26 field_mul", 100000, [&]() {
        fe26_a = fe26_a * fe26_b;
        g_sink ^= fe26_a.n[0];
    });
    fe26_a = FE26::from_fe(FE::from_uint64(0xDEADBEEF12345678ULL));

    long long sqr26_ns = bench("10x26 field_sqr", 100000, [&]() {
        fe26_a = fe26_a.square();
        g_sink ^= fe26_a.n[0];
    });
    fe26_a = FE26::from_fe(FE::from_uint64(0xDEADBEEF12345678ULL));

    long long add26_ns = bench("10x26 field_add (lazy)", 100000, [&]() {
        fe26_a = fe26_a + fe26_b;
        g_sink ^= fe26_a.n[0];
    });
    fe26_a = FE26::from_fe(FE::from_uint64(0xDEADBEEF12345678ULL));

    long long neg26_ns = bench("10x26 field_negate", 100000, [&]() {
        fe26_a = fe26_a.negate(1);
        g_sink ^= fe26_a.n[0];
    });
    fe26_a = FE26::from_fe(FE::from_uint64(0xDEADBEEF12345678ULL));

    long long half26_ns = bench("10x26 field_half", 100000, [&]() {
        fe26_a = fe26_a.half();
        g_sink ^= fe26_a.n[0];
    });
    fe26_a = FE26::from_fe(FE::from_uint64(0xDEADBEEF12345678ULL));

    printf("\n--- 10x26 Lazy Add Chains ---\n");
    for (int chain : {4, 8, 16, 32, 64}) {
        char name[64];
        std::snprintf(name, sizeof(name), "10x26 add chain (%d + norm)", chain);
        bench(name, 50000, [&]() {
            FE26 acc = fe26_a;
            for (int i = 0; i < chain; ++i) acc.add_assign(fe26_b);
            acc.normalize_weak();
            g_sink ^= acc.n[0];
        });
    }

    printf("\n--- 10x26 Mixed Chain (Point-Add Sim) ---\n");
    bench("10x26 point-add sim (12M+4S+7A)", 50000, [&]() {
        FE26 t0 = fe26_a, t1 = fe26_b;
        FE26 u1 = t0 * t1;
        FE26 u2 = t1.square();
        FE26 s1 = u1 + u2;
        FE26 s2 = t0.square();
        FE26 h  = s1 + s2 + u1;
        FE26 r  = h * s1;
        FE26 rx = r.square() + h.negate(1);
        FE26 ry = rx * h + r.negate(1);
        FE26 rz = ry.square();
        FE26 w  = (rz + rx) * ry;
        FE26 v  = (w + rz.negate(1)).square();
        FE26 out = v * w;
        g_sink ^= out.n[0];
    });

    // -- 10x26 vs 4x64 Summary --
    printf("\n=============================================\n");
    printf("COMPARISON -- 4x64 vs 10x26 on ARM64\n");
    printf("=============================================\n");
    printf("  Operation       4x64 (ns)   10x26 (ns)   Ratio\n");
    printf("  --------------  ---------   ----------   -----\n");
    printf("  Multiplication  %6lld       %6lld        %.2fx\n", mul_ns, mul26_ns, (double)mul_ns / mul26_ns);
    printf("  Squaring        %6lld       %6lld        %.2fx\n", sqr_ns, sqr26_ns, (double)sqr_ns / sqr26_ns);
    printf("  Addition        %6lld       %6lld        %.2fx\n", add_ns, add26_ns, (double)add_ns / add26_ns);
    printf("  Negation        %6lld       %6lld        %.2fx\n", sub_ns, neg26_ns, (double)sub_ns / neg26_ns);
    printf("  Half            %6s       %6lld        N/A\n", "N/A", half26_ns);
    printf("\n  Ratio >1 = 10x26 faster, <1 = 4x64 faster\n");
    printf("  On ARM64 (64-bit), 4x64 should win at mul.\n");
    printf("  10x26 wins at lazy add chains.\n");
    printf("=============================================\n");
}

int main() {
    int ret = run_tests();
    if (ret != 0) return ret;
    
    run_benchmarks();
    
    return 0;
}
