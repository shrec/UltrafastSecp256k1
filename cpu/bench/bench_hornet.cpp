// ============================================================================
// bench_hornet.cpp -- Bitcoin Consensus Benchmark Suite for Hornet Node
// ============================================================================
//
// Comprehensive single-core CPU benchmark of all secp256k1 operations
// relevant to Bitcoin block validation and IBD (Initial Block Download).
//
// RDTSC on x86-64, IQR outlier removal, median of 11 passes.
// Designed for presentation to Hornet Node (hornetnode.org).
//
// Build (Clang recommended):
//   cmake -S <root> -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
//         -DCMAKE_CXX_COMPILER=clang-cl -DCMAKE_LINKER=lld-link
//   cmake --build build --target bench_hornet -j
//
// Run:
//   ./build/bench_hornet
//
// ============================================================================

#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/batch_verify.hpp"
#include "secp256k1/ct/sign.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/selftest.hpp"
#include "secp256k1/init.hpp"
#include "secp256k1/benchmark_harness.hpp"

#include <array>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <chrono>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #if defined(_MSC_VER) || defined(__clang__)
    #include <intrin.h>
  #elif defined(__GNUC__)
    #include <cpuid.h>
    #include <x86intrin.h>
    // GCC __cpuid(level,a,b,c,d) differs from MSVC __cpuid(int[4],level)
    static inline void gcc_compat_cpuid(int regs[4], int level) {
        __cpuid(level, regs[0], regs[1], regs[2], regs[3]);
    }
    #undef __cpuid
    #define __cpuid(regs, level) gcc_compat_cpuid(regs, level)
  #endif
#endif

#define BENCH_STRINGIZE_IMPL(x) #x
#define BENCH_STRINGIZE(x) BENCH_STRINGIZE_IMPL(x)

using namespace secp256k1::fast;
using namespace secp256k1;

// -- CPU identification -------------------------------------------------------

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
static void get_cpu_brand(char brand[49]) {
    int regs[4];
    __cpuid(regs, 0x80000000);
    unsigned max_ext = static_cast<unsigned>(regs[0]);
    if (max_ext < 0x80000004u) {
        snprintf(brand, 49, "(unknown CPU)");
        return;
    }
    for (unsigned i = 0; i < 3; ++i) {
        __cpuid(regs, 0x80000002u + i);
        std::memcpy(brand + i * 16, regs, 16);
    }
    brand[48] = '\0';
    // trim leading spaces
    char* p = brand;
    while (*p == ' ') ++p;
    if (p != brand) std::memmove(brand, p, 49 - (p - brand));
}
#else
static void get_cpu_brand(char brand[49]) {
    snprintf(brand, 49, "(non-x86 platform)");
}
#endif

// -- TSC frequency calibration -----------------------------------------------

static double calibrate_tsc_ghz() {
#if defined(__x86_64__) || defined(_M_X64)
    unsigned aux;
    uint64_t t0 = __rdtscp(&aux);
    auto wall0 = std::chrono::high_resolution_clock::now();

    // Spin for ~50ms to calibrate
    volatile uint64_t sink = 0;
    for (int i = 0; i < 5000000; ++i) sink += i;

    uint64_t t1 = __rdtscp(&aux);
    auto wall1 = std::chrono::high_resolution_clock::now();

    double ns = std::chrono::duration<double, std::nano>(wall1 - wall0).count();
    double cycles = static_cast<double>(t1 - t0);
    return cycles / ns; // GHz
#else
    return 0.0;
#endif
}

// -- Harness ------------------------------------------------------------------

static bench::Harness H(500, 11);

template <typename Func>
static double bench_ns(Func&& f, int iters) {
    return H.run(iters, std::forward<Func>(f));
}

// -- Helpers ------------------------------------------------------------------

static std::array<std::uint8_t, 32> make_hash(uint64_t seed) {
    std::array<std::uint8_t, 32> h{};
    for (int i = 0; i < 4; ++i) {
        uint64_t v = seed ^ (seed << 13) ^ (uint64_t(i) * 0x9e3779b97f4a7c15ULL);
        std::memcpy(&h[i * 8], &v, 8);
    }
    return h;
}

static Scalar make_scalar(uint64_t seed) {
    auto h = make_hash(seed);
    return Scalar::from_bytes(h);
}

// -- Formatting ---------------------------------------------------------------

static double g_tsc_ghz = 0.0;

static void print_line() {
    printf("+------------------------------------------+----------+----------+-----------+----------+\n");
}

static void print_header_row() {
    printf("| %-40s | %8s | %8s | %9s | %8s |\n",
           "Operation", "ns/op", "us/op", "cycles/op", "ops/sec");
}

static void print_section(const char* name) {
    print_line();
    printf("| %-40s |          |          |           |          |\n", name);
    print_line();
    print_header_row();
    print_line();
}

static void print_row(const char* name, double ns) {
    double us = ns / 1000.0;
    double cycles = ns * g_tsc_ghz;
    double ops = 1e9 / ns;

    char ops_buf[32];
    if (ops >= 1e6)       snprintf(ops_buf, sizeof(ops_buf), "%6.2f M", ops / 1e6);
    else if (ops >= 1e3)  snprintf(ops_buf, sizeof(ops_buf), "%6.1f k", ops / 1e3);
    else                  snprintf(ops_buf, sizeof(ops_buf), "%6.0f  ", ops);

    if (cycles > 0.5) {
        printf("| %-40s | %8.1f | %8.2f | %9.0f | %8s |\n",
               name, ns, us, cycles, ops_buf);
    } else {
        printf("| %-40s | %8.1f | %8.2f | %9s | %8s |\n",
               name, ns, us, "--", ops_buf);
    }
}

static void print_ratio_row(const char* name, double ratio) {
    printf("| %-40s | %7.2fx |          |           |          |\n", name, ratio);
}

// -- Main ---------------------------------------------------------------------

int main() {
    SECP256K1_INIT();
    bench::pin_thread_and_elevate();

    // -- CPU identification ---------------------------------------------------
    char cpu_brand[49] = {};
    get_cpu_brand(cpu_brand);
    g_tsc_ghz = calibrate_tsc_ghz();

    // Quick sanity check
    printf("Running integrity check... ");
    if (!secp256k1::fast::Selftest(false)) {
        printf("FAIL\n");
        return 1;
    }
    printf("OK\n");

    printf("\n");
    printf("==========================================================================================\n");
    printf("  UltrafastSecp256k1 -- Bitcoin Consensus CPU Benchmark (Single Core)\n");
    printf("  Target:   Hornet Node (hornetnode.org)\n");
    printf("==========================================================================================\n");
    printf("\n");
    printf("  CPU:       %s\n", cpu_brand);
    printf("  TSC freq:  %.3f GHz (calibrated)\n", g_tsc_ghz);
    printf("  Cores:     1 (pinned, single-threaded)\n");
    printf("  Compiler:  "
#if defined(__clang__)
        "Clang " __clang_version__
#elif defined(_MSC_VER)
        "MSVC " BENCH_STRINGIZE(_MSC_VER)
#elif defined(__GNUC__)
        "GCC " __VERSION__
#else
        "Unknown"
#endif
        "\n");
    printf("  Arch:      "
#if defined(__x86_64__) || defined(_M_X64)
        "x86-64 (64-bit, BMI2/ADX capable)"
#elif defined(__aarch64__)
        "ARM64 (AArch64)"
#elif defined(__riscv)
        "RISC-V 64"
#else
        "Unknown"
#endif
        "\n");
    printf("  Linker:    "
#if defined(__clang__) && (defined(_MSC_VER) || defined(_WIN32))
        "lld-link (LLVM LLD)"
#elif defined(_MSC_VER)
        "MSVC link.exe (LTCG)"
#elif defined(__GNUC__)
        "GNU ld / gold"
#else
        "Unknown"
#endif
        "\n");
    printf("  Library:   UltrafastSecp256k1 v3.14.0\n");
    printf("  Field:     4x64 limbs (uint64_t[4]), Montgomery reduction\n");
    printf("  Scalar:    4x64 limbs, Barrett/GLV decomposition\n");
    printf("  Point mul: GLV endomorphism + wNAF (w=5)\n");
    printf("  Dual mul:  Shamir's trick (a*G + b*P)\n");
    printf("\n");
    H.print_config();
    printf("\n");

    // -- Prepare test data ----------------------------------------------------

    constexpr int POOL = 64;

    Scalar privkeys[POOL];
    for (int i = 0; i < POOL; ++i)
        privkeys[i] = make_scalar(0xdeadbeef00ULL + i);

    Point pubkeys[POOL];
    for (int i = 0; i < POOL; ++i)
        pubkeys[i] = Point::generator().scalar_mul(privkeys[i]);

    std::array<std::uint8_t, 32> msghashes[POOL];
    for (int i = 0; i < POOL; ++i)
        msghashes[i] = make_hash(0xcafebabe00ULL + i);

    std::array<std::uint8_t, 32> aux_rands[POOL];
    for (int i = 0; i < POOL; ++i)
        aux_rands[i] = make_hash(0xfeedface00ULL + i);

    ECDSASignature ecdsa_sigs[POOL];
    for (int i = 0; i < POOL; ++i)
        ecdsa_sigs[i] = ecdsa_sign(msghashes[i], privkeys[i]);

    SchnorrKeypair schnorr_kps[POOL];
    SchnorrSignature schnorr_sigs[POOL];
    std::array<std::uint8_t, 32> schnorr_pubkeys_x[POOL];
    SchnorrXonlyPubkey schnorr_xonly[POOL];
    for (int i = 0; i < POOL; ++i) {
        schnorr_kps[i] = schnorr_keypair_create(privkeys[i]);
        schnorr_sigs[i] = schnorr_sign(schnorr_kps[i], msghashes[i], aux_rands[i]);
        schnorr_pubkeys_x[i] = schnorr_pubkey(privkeys[i]);
        schnorr_xonly_pubkey_parse(schnorr_xonly[i], schnorr_pubkeys_x[i]);
    }

    constexpr int N_SIGN     = 500;
    constexpr int N_VERIFY   = 500;
    constexpr int N_KEYGEN   = 500;
    constexpr int N_SCALAR   = 500;
    constexpr int N_FIELD    = 50000;
    constexpr int N_POINT    = 10000;
    constexpr int N_SERIAL   = 50000;
    constexpr int N_BATCH    = 20;

    int idx = 0;

    // =========================================================================
    // 1. ECDSA (RFC 6979) -- pre-Taproot Bitcoin consensus core
    // =========================================================================

    print_section("ECDSA (RFC 6979)");

    double ecdsa_sign_ns = bench_ns([&]() {
        auto sig = ecdsa_sign(msghashes[idx % POOL], privkeys[idx % POOL]);
        bench::DoNotOptimize(sig);
        ++idx;
    }, N_SIGN);
    print_row("ecdsa_sign (deterministic nonce)", ecdsa_sign_ns);

    idx = 0;
    double ecdsa_verify_ns = bench_ns([&]() {
        bool ok = ecdsa_verify(msghashes[idx % POOL], pubkeys[idx % POOL],
                               ecdsa_sigs[idx % POOL]);
        bench::DoNotOptimize(ok);
        ++idx;
    }, N_VERIFY);
    print_row("ecdsa_verify (full)", ecdsa_verify_ns);
    print_line();

    // =========================================================================
    // 2. Schnorr / BIP-340 (Taproot consensus)
    // =========================================================================

    print_section("Schnorr / BIP-340 (Taproot)");

    idx = 0;
    double schnorr_sign_ns = bench_ns([&]() {
        auto sig = schnorr_sign(schnorr_kps[idx % POOL], msghashes[idx % POOL],
                                aux_rands[idx % POOL]);
        bench::DoNotOptimize(sig);
        ++idx;
    }, N_SIGN);
    print_row("schnorr_sign (pre-computed keypair)", schnorr_sign_ns);

    idx = 0;
    double schnorr_sign_raw_ns = bench_ns([&]() {
        auto sig = schnorr_sign(privkeys[idx % POOL], msghashes[idx % POOL],
                                aux_rands[idx % POOL]);
        bench::DoNotOptimize(sig);
        ++idx;
    }, N_SIGN);
    print_row("schnorr_sign (from raw privkey)", schnorr_sign_raw_ns);

    idx = 0;
    double schnorr_verify_ns = bench_ns([&]() {
        bool ok = schnorr_verify(schnorr_pubkeys_x[idx % POOL],
                                 msghashes[idx % POOL],
                                 schnorr_sigs[idx % POOL]);
        bench::DoNotOptimize(ok);
        ++idx;
    }, N_VERIFY);
    print_row("schnorr_verify (x-only 32B pubkey)", schnorr_verify_ns);

    idx = 0;
    double schnorr_verify_cached_ns = bench_ns([&]() {
        bool ok = schnorr_verify(schnorr_xonly[idx % POOL],
                                 msghashes[idx % POOL],
                                 schnorr_sigs[idx % POOL]);
        bench::DoNotOptimize(ok);
        ++idx;
    }, N_VERIFY);
    print_row("schnorr_verify (pre-parsed pubkey)", schnorr_verify_cached_ns);
    print_line();

    // =========================================================================
    // 3. Batch Verification (block-level amortization)
    // =========================================================================

    print_section("Batch Verification (N=64)");

    double schnorr_batch_per_sig = 0;
    {
        std::vector<SchnorrBatchEntry> batch(POOL);
        for (int i = 0; i < POOL; ++i) {
            batch[i].pubkey_x = schnorr_pubkeys_x[i];
            batch[i].message  = msghashes[i];
            batch[i].signature = schnorr_sigs[i];
        }
        double total = bench_ns([&]() {
            bool ok = schnorr_batch_verify(batch);
            bench::DoNotOptimize(ok);
        }, N_BATCH);
        schnorr_batch_per_sig = total / POOL;
        char buf[80];
        snprintf(buf, sizeof(buf), "schnorr_batch_verify (per sig, N=%d)", POOL);
        print_row(buf, schnorr_batch_per_sig);
        print_ratio_row("  -> vs individual schnorr_verify", schnorr_verify_ns / schnorr_batch_per_sig);
    }

    double ecdsa_batch_per_sig = 0;
    {
        std::vector<ECDSABatchEntry> batch(POOL);
        for (int i = 0; i < POOL; ++i) {
            batch[i].msg_hash  = msghashes[i];
            batch[i].public_key = pubkeys[i];
            batch[i].signature  = ecdsa_sigs[i];
        }
        double total = bench_ns([&]() {
            bool ok = ecdsa_batch_verify(batch);
            bench::DoNotOptimize(ok);
        }, N_BATCH);
        ecdsa_batch_per_sig = total / POOL;
        char buf[80];
        snprintf(buf, sizeof(buf), "ecdsa_batch_verify (per sig, N=%d)", POOL);
        print_row(buf, ecdsa_batch_per_sig);
        print_ratio_row("  -> vs individual ecdsa_verify", ecdsa_verify_ns / ecdsa_batch_per_sig);
    }
    print_line();

    // =========================================================================
    // 4. Key Generation
    // =========================================================================

    print_section("Key Generation");

    idx = 0;
    double keygen_ns = bench_ns([&]() {
        auto pk = Point::generator().scalar_mul(privkeys[idx % POOL]);
        bench::DoNotOptimize(pk);
        ++idx;
    }, N_KEYGEN);
    print_row("pubkey_create (k*G, GLV+wNAF)", keygen_ns);

    idx = 0;
    double schnorr_keygen_ns = bench_ns([&]() {
        auto kp = schnorr_keypair_create(privkeys[idx % POOL]);
        bench::DoNotOptimize(kp);
        ++idx;
    }, N_KEYGEN);
    print_row("schnorr_keypair_create", schnorr_keygen_ns);
    print_line();

    // =========================================================================
    // 5. Scalar Multiplication & Point Arithmetic
    // =========================================================================

    print_section("Point Arithmetic (ECC core)");

    idx = 0;
    double scalar_mul_ns = bench_ns([&]() {
        auto r = pubkeys[idx % POOL].scalar_mul(privkeys[(idx + 1) % POOL]);
        bench::DoNotOptimize(r);
        ++idx;
    }, N_SCALAR);
    print_row("k*P (arbitrary point, GLV+wNAF)", scalar_mul_ns);

    idx = 0;
    double dual_mul_ns = bench_ns([&]() {
        auto r = Point::dual_scalar_mul_gen_point(
            privkeys[idx % POOL], privkeys[(idx + 1) % POOL],
            pubkeys[(idx + 2) % POOL]);
        bench::DoNotOptimize(r);
        ++idx;
    }, N_SCALAR);
    print_row("a*G + b*P (Shamir dual mul)", dual_mul_ns);

    double add_ns = bench_ns([&]() {
        auto r = pubkeys[0].add(pubkeys[1]);
        bench::DoNotOptimize(r);
    }, N_POINT);
    print_row("point_add (Jacobian mixed)", add_ns);

    double dbl_ns = bench_ns([&]() {
        auto r = pubkeys[0].dbl();
        bench::DoNotOptimize(r);
    }, N_POINT);
    print_row("point_dbl (Jacobian)", dbl_ns);
    print_line();

    // =========================================================================
    // 6. Field Arithmetic (4x64 limbs)
    // =========================================================================

    print_section("Field Arithmetic (4x64 limbs)");

    auto fe_a = FieldElement::from_hex(
        "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798");
    auto fe_b = FieldElement::from_hex(
        "483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8");

    double fmul_ns = bench_ns([&]() {
        auto r = fe_a * fe_b;
        bench::DoNotOptimize(r);
    }, N_FIELD);
    print_row("field_mul (Montgomery)", fmul_ns);

    double fsqr_ns = bench_ns([&]() {
        auto r = fe_a.square();
        bench::DoNotOptimize(r);
    }, N_FIELD);
    print_row("field_sqr (Montgomery)", fsqr_ns);

    double finv_ns = bench_ns([&]() {
        auto r = fe_a.inverse();
        bench::DoNotOptimize(r);
    }, 200);
    print_row("field_inv (Fermat, 256-bit exp)", finv_ns);

    double fadd_ns = bench_ns([&]() {
        auto r = fe_a + fe_b;
        bench::DoNotOptimize(r);
    }, N_FIELD);
    print_row("field_add (mod p)", fadd_ns);

    double fsub_ns = bench_ns([&]() {
        auto r = fe_a - fe_b;
        bench::DoNotOptimize(r);
    }, N_FIELD);
    print_row("field_sub (mod p)", fsub_ns);

    double fneg_ns = bench_ns([&]() {
        auto r = fe_a.negate();
        bench::DoNotOptimize(r);
    }, N_FIELD);
    print_row("field_negate (mod p)", fneg_ns);
    print_line();

    // =========================================================================
    // 7. Scalar Arithmetic
    // =========================================================================

    print_section("Scalar Arithmetic (4x64 limbs, mod n)");

    auto sc_a = make_scalar(0xdeadbeef01ULL);
    auto sc_b = make_scalar(0xdeadbeef02ULL);

    double smul_ns = bench_ns([&]() {
        auto r = sc_a * sc_b;
        bench::DoNotOptimize(r);
    }, N_FIELD);
    print_row("scalar_mul (mod n)", smul_ns);

    double sinv_ns = bench_ns([&]() {
        auto r = sc_a.inverse();
        bench::DoNotOptimize(r);
    }, 200);
    print_row("scalar_inv (mod n)", sinv_ns);

    double sadd_ns = bench_ns([&]() {
        auto r = sc_a + sc_b;
        bench::DoNotOptimize(r);
    }, N_FIELD);
    print_row("scalar_add (mod n)", sadd_ns);

    double sneg_ns = bench_ns([&]() {
        auto r = sc_a.negate();
        bench::DoNotOptimize(r);
    }, N_FIELD);
    print_row("scalar_negate (mod n)", sneg_ns);
    print_line();

    // =========================================================================
    // 8. Serialization
    // =========================================================================

    print_section("Serialization");

    idx = 0;
    double compress_ns = bench_ns([&]() {
        auto c = pubkeys[idx % POOL].to_compressed();
        bench::DoNotOptimize(c);
        ++idx;
    }, N_SERIAL);
    print_row("pubkey_serialize (33B compressed)", compress_ns);

    idx = 0;
    double der_encode_ns = bench_ns([&]() {
        auto d = ecdsa_sigs[idx % POOL].to_der();
        bench::DoNotOptimize(d);
        ++idx;
    }, N_SERIAL);
    print_row("ecdsa_sig_to_der (DER encode)", der_encode_ns);

    idx = 0;
    double schnorr_ser_ns = bench_ns([&]() {
        auto b = schnorr_sigs[idx % POOL].to_bytes();
        bench::DoNotOptimize(b);
        ++idx;
    }, N_SERIAL);
    print_row("schnorr_sig_to_bytes (64B)", schnorr_ser_ns);
    print_line();

    // =========================================================================
    // 9. Constant-Time Signing (side-channel resistant)
    // =========================================================================

    print_section("Constant-Time Signing (CT layer)");

    idx = 0;
    double ct_ecdsa_ns = bench_ns([&]() {
        auto sig = ct::ecdsa_sign(msghashes[idx % POOL], privkeys[idx % POOL]);
        bench::DoNotOptimize(sig);
        ++idx;
    }, N_SIGN);
    print_row("ct::ecdsa_sign", ct_ecdsa_ns);
    print_ratio_row("  -> CT overhead vs fast::ecdsa_sign", ct_ecdsa_ns / ecdsa_sign_ns);

    idx = 0;
    double ct_schnorr_ns = bench_ns([&]() {
        auto sig = ct::schnorr_sign(schnorr_kps[idx % POOL],
                                     msghashes[idx % POOL],
                                     aux_rands[idx % POOL]);
        bench::DoNotOptimize(sig);
        ++idx;
    }, N_SIGN);
    print_row("ct::schnorr_sign", ct_schnorr_ns);
    print_ratio_row("  -> CT overhead vs fast::schnorr_sign", ct_schnorr_ns / schnorr_sign_ns);
    print_line();

    // =========================================================================
    // Summary Section
    // =========================================================================

    printf("\n");
    printf("==========================================================================================\n");
    printf("  THROUGHPUT SUMMARY (1 core, pinned)\n");
    printf("==========================================================================================\n\n");

    auto print_tput = [](const char* name, double ns) {
        double ops = 1e9 / ns;
        double us = ns / 1000.0;
        if (ops >= 1e6) {
            printf("  %-42s %8.2f us  ->  %8.2f M op/s\n", name, us, ops / 1e6);
        } else if (ops >= 1e3) {
            printf("  %-42s %8.2f us  ->  %8.1f k op/s\n", name, us, ops / 1e3);
        } else {
            printf("  %-42s %8.2f us  ->  %8.0f   op/s\n", name, us, ops);
        }
    };

    printf("  --- Bitcoin Consensus Critical Path ---\n");
    print_tput("ECDSA sign (RFC 6979)",           ecdsa_sign_ns);
    print_tput("ECDSA verify",                    ecdsa_verify_ns);
    print_tput("Schnorr sign (BIP-340, keypair)", schnorr_sign_ns);
    print_tput("Schnorr verify (x-only)",         schnorr_verify_ns);
    print_tput("Schnorr verify (cached pubkey)",  schnorr_verify_cached_ns);
    printf("\n");
    printf("  --- Batch Verification (N=64) ---\n");
    print_tput("ECDSA batch (per sig)",           ecdsa_batch_per_sig);
    print_tput("Schnorr batch (per sig)",         schnorr_batch_per_sig);
    printf("\n");
    printf("  --- Key / Point Operations ---\n");
    print_tput("pubkey_create (k*G)",             keygen_ns);
    print_tput("scalar_mul (k*P)",                scalar_mul_ns);
    print_tput("dual_mul (a*G+b*P, Shamir)",      dual_mul_ns);
    print_tput("point_add",                       add_ns);
    print_tput("point_dbl",                       dbl_ns);
    printf("\n");
    printf("  --- Field / Scalar Primitives ---\n");
    print_tput("field_mul",                       fmul_ns);
    print_tput("field_sqr",                       fsqr_ns);
    print_tput("field_inv",                       finv_ns);
    print_tput("field_add",                       fadd_ns);
    print_tput("scalar_mul",                      smul_ns);
    print_tput("scalar_inv",                      sinv_ns);
    printf("\n");

    // =========================================================================
    // Block Validation Estimates
    // =========================================================================

    printf("==========================================================================================\n");
    printf("  BITCOIN BLOCK VALIDATION ESTIMATES (1 core)\n");
    printf("==========================================================================================\n\n");

    double pre_taproot_ms = 3000.0 * ecdsa_verify_ns / 1e6;
    double pre_taproot_batch_ms = 3000.0 * ecdsa_batch_per_sig / 1e6;
    double taproot_ms = (2000.0 * schnorr_verify_ns + 1000.0 * ecdsa_verify_ns) / 1e6;
    double taproot_batch_ms = (2000.0 * schnorr_batch_per_sig + 1000.0 * ecdsa_batch_per_sig) / 1e6;

    printf("  Pre-Taproot block (~3000 ECDSA verify):\n");
    printf("    Individual:    %7.1f ms\n", pre_taproot_ms);
    printf("    Batch (N=64):  %7.1f ms\n", pre_taproot_batch_ms);
    printf("\n");
    printf("  Taproot block (~2000 Schnorr + ~1000 ECDSA):\n");
    printf("    Individual:    %7.1f ms\n", taproot_ms);
    printf("    Batch (N=64):  %7.1f ms\n", taproot_batch_ms);
    printf("\n");

    // IBD estimates
    double ibd_sigs = 900000.0 * 1500.0; // ~1.35 billion
    double ibd_individual_h = ibd_sigs * ecdsa_verify_ns / 1e9 / 3600.0;
    double ibd_batch_h = ibd_sigs * ecdsa_batch_per_sig / 1e9 / 3600.0;

    printf("  Full IBD estimate (~%.2f billion sig verifies):\n", ibd_sigs / 1e9);
    printf("    Individual verify:  %6.1f hours  (%4.1f days)\n",
           ibd_individual_h, ibd_individual_h / 24.0);
    printf("    Batch verify:       %6.1f hours  (%4.1f days)\n",
           ibd_batch_h, ibd_batch_h / 24.0);
    printf("\n");

    // Multi-core projection (linear scaling assumption)
    printf("  Multi-core IBD projection (assuming linear sig-verify parallelism):\n");
    int core_counts[] = {2, 4, 8, 16};
    for (int nc : core_counts) {
        double h = ibd_individual_h / nc;
        printf("    %2d cores:  %6.1f hours  (%4.1f days)\n", nc, h, h / 24.0);
    }
    printf("\n");

    // Blocks per second
    double blocks_per_sec_pre = 1000.0 / pre_taproot_ms;
    double blocks_per_sec_tap = 1000.0 / taproot_ms;
    printf("  Blocks/sec throughput (sig verify only, 1 core):\n");
    printf("    Pre-Taproot:  %6.1f blocks/sec\n", blocks_per_sec_pre);
    printf("    Taproot:      %6.1f blocks/sec\n", blocks_per_sec_tap);
    printf("\n");

    // Transaction processing rate
    double ecdsa_per_sec = 1e9 / ecdsa_verify_ns;
    double schnorr_per_sec = 1e9 / schnorr_verify_ns;
    printf("  Transaction throughput (1-input txs, 1 core):\n");
    printf("    ECDSA txs:    %8.0f tx/sec\n", ecdsa_per_sec);
    printf("    Schnorr txs:  %8.0f tx/sec\n", schnorr_per_sec);
    printf("\n");

    printf("==========================================================================================\n");
    printf("  NOTES\n");
    printf("==========================================================================================\n\n");
    printf("  - All measurements: single-threaded, CPU pinned to core 0\n");
    printf("  - Timer: %s\n", bench::Timer::timer_name());
    printf("  - Each operation: 500 warmup + 11 passes, IQR outlier removal, median\n");
    printf("  - Pool: 64 independent key/msg/sig sets (prevents caching artifacts)\n");
    printf("  - CT layer: constant-time signing (side-channel resistant)\n");
    printf("  - FAST layer: maximum throughput (no side-channel guarantees)\n");
    printf("  - Batch verify uses Strauss multi-scalar multiplication\n");
    printf("  - ECDSA verify = Shamir dual-mul (a*G + b*P) + field inversion\n");
    printf("  - Schnorr verify = tagged hash + lift_x + dual-mul\n");
    printf("  - GLV endomorphism: 2x speedup on scalar mul via lambda splitting\n");
    printf("\n");
    printf("==========================================================================================\n");
    printf("  %s | 1 core | "
#if defined(__clang__)
        "Clang " __clang_version__
#elif defined(_MSC_VER)
        "MSVC " BENCH_STRINGIZE(_MSC_VER)
#elif defined(__GNUC__)
        "GCC " __VERSION__
#else
        "Unknown"
#endif
        " | UltrafastSecp256k1 v3.14.0\n", cpu_brand);
    printf("==========================================================================================\n\n");

    return 0;
}
