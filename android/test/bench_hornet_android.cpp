// ============================================================================
// Android ARM64 bench_hornet.cpp -- Bitcoin Consensus Benchmark Suite
// ============================================================================
//
// ARM64 (aarch64) port of the bench_hornet benchmark.
// Identical sections and output format as x86 and ESP32 versions.
// Uses clock_gettime(CLOCK_MONOTONIC) as high-resolution timer.
// Includes libsecp256k1 (bitcoin-core) apple-to-apple comparison.
//
// Build: cross-compiled via NDK, pushed to device and run in adb shell.
// ============================================================================

#include "secp256k1/field.hpp"
#include "secp256k1/field_optimal.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/batch_verify.hpp"
#include "secp256k1/ct/sign.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/selftest.hpp"

#include <array>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <ctime>
#include <sys/utsname.h>
#include <unistd.h>

using namespace secp256k1::fast;
using namespace secp256k1;

// -- Helpers ------------------------------------------------------------------

static std::array<std::uint8_t, 32> make_hash(uint64_t seed) {
    std::array<std::uint8_t, 32> h{};
    for (int i = 0; i < 4; ++i) {
        uint64_t v = seed ^ (seed << 13) ^ (uint64_t(i) * 0x9e3779b97f4a7c15ULL);
        std::memcpy(&h[static_cast<std::size_t>(i) * 8], &v, 8);
    }
    return h;
}

static Scalar make_scalar(uint64_t seed) {
    auto h = make_hash(seed);
    return Scalar::from_bytes(h);
}

// -- Timer (clock_gettime based, median-of-5) ---------------------------------

static inline double get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

template <typename Func>
static double bench_median5(Func&& f, int iters) {
    // warmup
    for (int i = 0; i < std::max(iters / 5, 3); ++i) f();

    double results[5];
    for (int r = 0; r < 5; ++r) {
        double t0 = get_time_ns();
        for (int i = 0; i < iters; ++i) {
            f();
        }
        double dt = get_time_ns() - t0;
        results[r] = dt / iters; // ns per op
    }

    // sort and take median
    std::sort(results, results + 5);
    return results[2];
}

// -- Formatting (identical to ESP32/x86 bench_hornet) -------------------------

static void print_line() {
    printf("+------------------------------------------+----------+----------+----------+\n");
}

static void print_header_row() {
    printf("| %-40s | %8s | %8s | %8s |\n",
           "Operation", "ns/op", "us/op", "ops/sec");
}

static void print_section(const char* name) {
    print_line();
    printf("| %-40s |          |          |          |\n", name);
    print_line();
    print_header_row();
    print_line();
}

static void print_row(const char* name, double ns) {
    const double us = ns / 1000.0;
    const double ops = 1e9 / ns;

    char ops_buf[32];
    if (ops >= 1e6) {
        snprintf(ops_buf, sizeof(ops_buf), "%6.2f M", ops / 1e6);
    } else if (ops >= 1e3) {
        snprintf(ops_buf, sizeof(ops_buf), "%6.1f k", ops / 1e3);
    } else {
        snprintf(ops_buf, sizeof(ops_buf), "%6.0f  ", ops);
    }

    printf("| %-40s | %8.1f | %8.2f | %8s |\n",
           name, ns, us, ops_buf);
}

static void print_ratio_row(const char* name, double ratio) {
    printf("| %-40s | %7.2fx |          |          |\n", name, ratio);
}

// -- libsecp256k1 extern ------------------------------------------------------
extern "C" void libsecp_benchmark(void);

// -- CPU info -----------------------------------------------------------------
static void get_cpu_info(char* buf, size_t bufsize) {
    FILE* f = fopen("/proc/cpuinfo", "r");
    if (!f) {
        snprintf(buf, bufsize, "Unknown ARM64 CPU");
        return;
    }
    char line[256];
    const char* model = NULL;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "Hardware", 8) == 0 || strncmp(line, "model name", 10) == 0) {
            char* colon = strchr(line, ':');
            if (colon) {
                model = colon + 2;
                // trim newline
                char* nl = strchr(colon, '\n');
                if (nl) *nl = '\0';
                snprintf(buf, bufsize, "%s", colon + 2);
                fclose(f);
                return;
            }
        }
    }
    fclose(f);
    // fallback: try to read CPU part
    snprintf(buf, bufsize, "ARM64 (aarch64)");
}

static int get_cpu_count() {
    return (int)sysconf(_SC_NPROCESSORS_ONLN);
}

// -- Main ---------------------------------------------------------------------

int main() {
    // -- Integrity check ------------------------------------------------------
    printf("Running integrity check... ");
    fflush(stdout);
    if (!Selftest(false)) {
        printf("FAIL\n");
        return 1;
    }
    printf("OK\n");

    // -- Header ---------------------------------------------------------------
    char cpu_name[128];
    get_cpu_info(cpu_name, sizeof(cpu_name));
    int ncpus = get_cpu_count();

    struct utsname uts;
    uname(&uts);

    printf("\n");
    printf("==========================================================================================\n");
    printf("  UltrafastSecp256k1 -- Bitcoin Consensus CPU Benchmark (Single Core)\n");
    printf("  Target:   Hornet Node (hornetnode.org)\n");
    printf("==========================================================================================\n");
    printf("\n");
    printf("  CPU:       %s\n", cpu_name);
    printf("  Cores:     %d (single-threaded benchmark)\n", ncpus);
    printf("  Kernel:    %s %s\n", uts.sysname, uts.release);
    printf("  Compiler:  %s %d.%d.%d\n",
#if defined(__clang__)
           "Clang", __clang_major__, __clang_minor__, __clang_patchlevel__
#elif defined(__GNUC__)
           "GCC", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__
#else
           "Unknown", 0, 0, 0
#endif
    );
    printf("  Arch:      aarch64 (64-bit, NEON, __int128)\n");
    printf("  Library:   UltrafastSecp256k1 v3.16.0\n");
    printf("  Field:     %s\n", secp256k1::fast::kOptimalTierName);
    printf("  Scalar:    4x64 limbs, Barrett/GLV decomposition\n");
    printf("  Point mul: GLV endomorphism + wNAF (w=5)\n");
    printf("  Dual mul:  Shamir's trick (a*G + b*P)\n");
    printf("\n");
    printf("  Timer:    clock_gettime(CLOCK_MONOTONIC)\n");
    printf("  Method:   median of 5 runs, per-op warmup\n");
    printf("\n");

    // -- Prepare test data ----------------------------------------------------
    constexpr int POOL = 32;

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

    // Iteration counts (ARM64 is fast -- use decent counts)
    constexpr int N_SIGN     = 50;
    constexpr int N_VERIFY   = 50;
    constexpr int N_KEYGEN   = 50;
    constexpr int N_SCALAR   = 50;
    constexpr int N_FIELD    = 5000;
    constexpr int N_POINT    = 2000;
    constexpr int N_SERIAL   = 2000;
    constexpr int N_BATCH    = 10;

    int idx = 0;

    // =========================================================================
    // 1. ECDSA (RFC 6979)
    // =========================================================================

    print_section("ECDSA (RFC 6979)");

    idx = 0;
    const double ecdsa_sign_ns = bench_median5([&]() {
        auto sig = ecdsa_sign(msghashes[idx % POOL], privkeys[idx % POOL]);
        volatile auto sink = sig.r.limbs()[0]; (void)sink;
        ++idx;
    }, N_SIGN);
    print_row("ecdsa_sign (deterministic nonce)", ecdsa_sign_ns);

    idx = 0;
    const double ecdsa_verify_ns = bench_median5([&]() {
        bool ok = ecdsa_verify(msghashes[idx % POOL], pubkeys[idx % POOL],
                               ecdsa_sigs[idx % POOL]);
        volatile bool sink = ok; (void)sink;
        ++idx;
    }, N_VERIFY);
    print_row("ecdsa_verify (full)", ecdsa_verify_ns);
    print_line();

    // =========================================================================
    // 2. Schnorr / BIP-340 (Taproot)
    // =========================================================================

    print_section("Schnorr / BIP-340 (Taproot)");

    idx = 0;
    const double schnorr_sign_ns = bench_median5([&]() {
        auto sig = schnorr_sign(schnorr_kps[idx % POOL], msghashes[idx % POOL],
                                aux_rands[idx % POOL]);
        volatile auto sink = sig.r[0]; (void)sink;
        ++idx;
    }, N_SIGN);
    print_row("schnorr_sign (pre-computed keypair)", schnorr_sign_ns);

    idx = 0;
    const double schnorr_sign_raw_ns = bench_median5([&]() {
        auto sig = schnorr_sign(privkeys[idx % POOL], msghashes[idx % POOL],
                                aux_rands[idx % POOL]);
        volatile auto sink = sig.r[0]; (void)sink;
        ++idx;
    }, N_SIGN);
    print_row("schnorr_sign (from raw privkey)", schnorr_sign_raw_ns);

    idx = 0;
    const double schnorr_verify_ns = bench_median5([&]() {
        bool ok = schnorr_verify(schnorr_pubkeys_x[idx % POOL],
                                 msghashes[idx % POOL],
                                 schnorr_sigs[idx % POOL]);
        volatile bool sink = ok; (void)sink;
        ++idx;
    }, N_VERIFY);
    print_row("schnorr_verify (x-only 32B pubkey)", schnorr_verify_ns);

    idx = 0;
    const double schnorr_verify_cached_ns = bench_median5([&]() {
        bool ok = schnorr_verify(schnorr_xonly[idx % POOL],
                                 msghashes[idx % POOL],
                                 schnorr_sigs[idx % POOL]);
        volatile bool sink = ok; (void)sink;
        ++idx;
    }, N_VERIFY);
    print_row("schnorr_verify (pre-parsed pubkey)", schnorr_verify_cached_ns);
    print_line();

    // =========================================================================
    // 3. Batch Verification (N=32)
    // =========================================================================

    print_section("Batch Verification (N=32)");

    double schnorr_batch_per_sig = 0;
    {
        std::vector<SchnorrBatchEntry> batch(POOL);
        for (int i = 0; i < POOL; ++i) {
            batch[i].pubkey_x = schnorr_pubkeys_x[i];
            batch[i].message  = msghashes[i];
            batch[i].signature = schnorr_sigs[i];
        }
        const double total = bench_median5([&]() {
            bool ok = schnorr_batch_verify(batch);
            volatile bool sink = ok; (void)sink;
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
        const double total = bench_median5([&]() {
            bool ok = ecdsa_batch_verify(batch);
            volatile bool sink = ok; (void)sink;
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
    const double keygen_ns = bench_median5([&]() {
        auto pk = Point::generator().scalar_mul(privkeys[idx % POOL]);
        volatile auto sink = pk.x().limbs()[0]; (void)sink;
        ++idx;
    }, N_KEYGEN);
    print_row("pubkey_create (k*G, GLV+wNAF)", keygen_ns);

    idx = 0;
    const double schnorr_keygen_ns = bench_median5([&]() {
        auto kp = schnorr_keypair_create(privkeys[idx % POOL]);
        volatile auto sink = kp.px[0]; (void)sink;
        ++idx;
    }, N_KEYGEN);
    print_row("schnorr_keypair_create", schnorr_keygen_ns);
    print_line();

    // =========================================================================
    // 5. Point Arithmetic (ECC core)
    // =========================================================================

    print_section("Point Arithmetic (ECC core)");

    idx = 0;
    const double scalar_mul_ns = bench_median5([&]() {
        auto r = pubkeys[idx % POOL].scalar_mul(privkeys[(idx + 1) % POOL]);
        volatile auto sink = r.x().limbs()[0]; (void)sink;
        ++idx;
    }, N_SCALAR);
    print_row("k*P (arbitrary point, GLV+wNAF)", scalar_mul_ns);

    idx = 0;
    const double dual_mul_ns = bench_median5([&]() {
        auto r = Point::dual_scalar_mul_gen_point(
            privkeys[idx % POOL], privkeys[(idx + 1) % POOL],
            pubkeys[(idx + 2) % POOL]);
        volatile auto sink = r.x().limbs()[0]; (void)sink;
        ++idx;
    }, N_SCALAR);
    print_row("a*G + b*P (Shamir dual mul)", dual_mul_ns);

    const double add_ns = bench_median5([&]() {
        auto r = pubkeys[0].add(pubkeys[1]);
        volatile auto sink = r.x().limbs()[0]; (void)sink;
    }, N_POINT);
    print_row("point_add (Jacobian mixed)", add_ns);

    const double dbl_ns = bench_median5([&]() {
        auto r = pubkeys[0].dbl();
        volatile auto sink = r.x().limbs()[0]; (void)sink;
    }, N_POINT);
    print_row("point_dbl (Jacobian)", dbl_ns);
    print_line();

    // =========================================================================
    // 6. Field Arithmetic
    // =========================================================================

    print_section("Field Arithmetic");

    auto fe_a = FieldElement::from_hex(
        "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798");
    auto fe_b = FieldElement::from_hex(
        "483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8");

    const double fmul_ns = bench_median5([&]() {
        auto r = fe_a * fe_b;
        volatile auto sink = r.limbs()[0]; (void)sink;
    }, N_FIELD);
    print_row("field_mul", fmul_ns);

    const double fsqr_ns = bench_median5([&]() {
        auto r = fe_a.square();
        volatile auto sink = r.limbs()[0]; (void)sink;
    }, N_FIELD);
    print_row("field_sqr", fsqr_ns);

    const double finv_ns = bench_median5([&]() {
        auto r = fe_a.inverse();
        volatile auto sink = r.limbs()[0]; (void)sink;
    }, 100);
    print_row("field_inv (Fermat, 256-bit exp)", finv_ns);

    const double fadd_ns = bench_median5([&]() {
        auto r = fe_a + fe_b;
        volatile auto sink = r.limbs()[0]; (void)sink;
    }, N_FIELD);
    print_row("field_add (mod p)", fadd_ns);

    const double fsub_ns = bench_median5([&]() {
        auto r = fe_a - fe_b;
        volatile auto sink = r.limbs()[0]; (void)sink;
    }, N_FIELD);
    print_row("field_sub (mod p)", fsub_ns);

    const double fneg_ns = bench_median5([&]() {
        auto r = fe_a.negate();
        volatile auto sink = r.limbs()[0]; (void)sink;
    }, N_FIELD);
    print_row("field_negate (mod p)", fneg_ns);
    print_line();

    // =========================================================================
    // 7. Scalar Arithmetic
    // =========================================================================

    print_section("Scalar Arithmetic (mod n)");

    auto sc_a = make_scalar(0xdeadbeef01ULL);
    auto sc_b = make_scalar(0xdeadbeef02ULL);

    const double smul_ns = bench_median5([&]() {
        auto r = sc_a * sc_b;
        volatile auto sink = r.limbs()[0]; (void)sink;
    }, N_FIELD);
    print_row("scalar_mul (mod n)", smul_ns);

    const double sinv_ns = bench_median5([&]() {
        auto r = sc_a.inverse();
        volatile auto sink = r.limbs()[0]; (void)sink;
    }, 100);
    print_row("scalar_inv (mod n)", sinv_ns);

    const double sadd_ns = bench_median5([&]() {
        auto r = sc_a + sc_b;
        volatile auto sink = r.limbs()[0]; (void)sink;
    }, N_FIELD);
    print_row("scalar_add (mod n)", sadd_ns);

    const double sneg_ns = bench_median5([&]() {
        auto r = sc_a.negate();
        volatile auto sink = r.limbs()[0]; (void)sink;
    }, N_FIELD);
    print_row("scalar_negate (mod n)", sneg_ns);
    print_line();

    // =========================================================================
    // 8. Serialization
    // =========================================================================

    print_section("Serialization");

    idx = 0;
    const double compress_ns = bench_median5([&]() {
        auto c = pubkeys[idx % POOL].to_compressed();
        volatile auto sink = c[0]; (void)sink;
        ++idx;
    }, N_SERIAL);
    print_row("pubkey_serialize (33B compressed)", compress_ns);

    idx = 0;
    const double der_encode_ns = bench_median5([&]() {
        auto d = ecdsa_sigs[idx % POOL].to_der();
        volatile auto sink = d.first[0]; (void)sink;
        ++idx;
    }, N_SERIAL);
    print_row("ecdsa_sig_to_der (DER encode)", der_encode_ns);

    idx = 0;
    const double schnorr_ser_ns = bench_median5([&]() {
        auto b = schnorr_sigs[idx % POOL].to_bytes();
        volatile auto sink = b[0]; (void)sink;
        ++idx;
    }, N_SERIAL);
    print_row("schnorr_sig_to_bytes (64B)", schnorr_ser_ns);
    print_line();

    // =========================================================================
    // 9. Constant-Time Signing (CT layer)
    // =========================================================================

    print_section("Constant-Time Signing (CT layer)");

    idx = 0;
    const double ct_ecdsa_ns = bench_median5([&]() {
        auto sig = ct::ecdsa_sign(msghashes[idx % POOL], privkeys[idx % POOL]);
        volatile auto sink = sig.r.limbs()[0]; (void)sink;
        ++idx;
    }, N_SIGN);
    print_row("ct::ecdsa_sign", ct_ecdsa_ns);
    print_ratio_row("  -> CT overhead vs fast::ecdsa_sign", ct_ecdsa_ns / ecdsa_sign_ns);

    idx = 0;
    const double ct_schnorr_ns = bench_median5([&]() {
        auto sig = ct::schnorr_sign(schnorr_kps[idx % POOL],
                                     msghashes[idx % POOL],
                                     aux_rands[idx % POOL]);
        volatile auto sink = sig.r[0]; (void)sink;
        ++idx;
    }, N_SIGN);
    print_row("ct::schnorr_sign", ct_schnorr_ns);
    print_ratio_row("  -> CT overhead vs fast::schnorr_sign", ct_schnorr_ns / schnorr_sign_ns);
    print_line();

    // =========================================================================
    // 10. libsecp256k1 (bitcoin-core) Apple-to-Apple Comparison
    // =========================================================================

    printf("\n");
    printf("==========================================================================================\n");
    printf("  libsecp256k1 (bitcoin-core v0.7.2) APPLE-TO-APPLE COMPARISON\n");
    printf("==========================================================================================\n\n");
    libsecp_benchmark();

    // =========================================================================
    // THROUGHPUT SUMMARY
    // =========================================================================

    printf("\n");
    printf("==========================================================================================\n");
    printf("  THROUGHPUT SUMMARY (1 core)\n");
    printf("==========================================================================================\n\n");

    auto print_tput = [](const char* name, double ns) {
        const double ops = 1e9 / ns;
        const double us = ns / 1000.0;
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
    printf("  --- Batch Verification (N=%d) ---\n", POOL);
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

    const double pre_taproot_ms = 3000.0 * ecdsa_verify_ns / 1e6;
    const double pre_taproot_batch_ms = 3000.0 * ecdsa_batch_per_sig / 1e6;
    const double taproot_ms = (2000.0 * schnorr_verify_ns + 1000.0 * ecdsa_verify_ns) / 1e6;
    const double taproot_batch_ms = (2000.0 * schnorr_batch_per_sig + 1000.0 * ecdsa_batch_per_sig) / 1e6;

    printf("  Pre-Taproot block (~3000 ECDSA verify):\n");
    printf("    Individual:    %7.1f ms\n", pre_taproot_ms);
    printf("    Batch (N=%d): %7.1f ms\n", POOL, pre_taproot_batch_ms);
    printf("\n");
    printf("  Taproot block (~2000 Schnorr + ~1000 ECDSA):\n");
    printf("    Individual:    %7.1f ms\n", taproot_ms);
    printf("    Batch (N=%d): %7.1f ms\n", POOL, taproot_batch_ms);
    printf("\n");

    const double ecdsa_per_sec = 1e9 / ecdsa_verify_ns;
    const double schnorr_per_sec = 1e9 / schnorr_verify_ns;
    printf("  Transaction throughput (1-input txs, 1 core):\n");
    printf("    ECDSA txs:    %8.0f tx/sec\n", ecdsa_per_sec);
    printf("    Schnorr txs:  %8.0f tx/sec\n", schnorr_per_sec);
    printf("\n");

    const double blocks_per_sec_pre = 1000.0 / pre_taproot_ms;
    const double blocks_per_sec_tap = 1000.0 / taproot_ms;
    printf("  Blocks/sec throughput (sig verify only, 1 core):\n");
    printf("    Pre-Taproot:  %6.2f blocks/sec\n", blocks_per_sec_pre);
    printf("    Taproot:      %6.2f blocks/sec\n", blocks_per_sec_tap);
    printf("\n");

    // =========================================================================
    // Notes
    // =========================================================================

    printf("==========================================================================================\n");
    printf("  NOTES\n");
    printf("==========================================================================================\n\n");
    printf("  - All measurements: single-threaded\n");
    printf("  - Timer: clock_gettime(CLOCK_MONOTONIC)\n");
    printf("  - Each operation: warmup + median of 5 runs\n");
    printf("  - Pool: %d independent key/msg/sig sets\n", POOL);
    printf("  - CT layer: constant-time signing (side-channel resistant)\n");
    printf("  - FAST layer: maximum throughput (no side-channel guarantees)\n");
    printf("  - Batch verify uses Strauss multi-scalar multiplication\n");
    printf("  - ECDSA verify = Shamir dual-mul (a*G + b*P) + field inversion\n");
    printf("  - Schnorr verify = tagged hash + lift_x + dual-mul\n");
    printf("  - GLV endomorphism: 2x speedup on scalar mul via lambda splitting\n");
    printf("  - libsecp256k1 comparison: same key, same hardware, same compiler\n");
    printf("\n");

    printf("==========================================================================================\n");
    printf("  %s | 1 core | %s %d.%d.%d | UltrafastSecp256k1 v3.16.0\n",
           cpu_name,
#if defined(__clang__)
           "Clang", __clang_major__, __clang_minor__, __clang_patchlevel__
#elif defined(__GNUC__)
           "GCC", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__
#else
           "Unknown", 0, 0, 0
#endif
    );
    printf("==========================================================================================\n\n");

    printf("BENCH_HORNET_COMPLETE\n");

    return 0;
}
