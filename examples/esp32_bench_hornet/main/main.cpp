// ============================================================================
// ESP32 bench_hornet.cpp -- Bitcoin Consensus Benchmark Suite for Hornet Node
// ============================================================================
//
// ESP32-S3 port of cpu/bench/bench_hornet.cpp
// Identical sections, identical output format, apple-to-apple with x86-64.
//
// Uses esp_timer (1us resolution) instead of RDTSC.
// Includes libsecp256k1 (bitcoin-core) apple-to-apple comparison section.
//
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

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_system.h"
#include "esp_chip_info.h"
#include "esp_heap_caps.h"

using namespace secp256k1::fast;
using namespace secp256k1;

// -- WDT yield ----------------------------------------------------------------
#define WDT_YIELD() vTaskDelay(pdMS_TO_TICKS(10))

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

// -- Timer (esp_timer based, median-of-3) -------------------------------------

template <typename Func>
static double bench_median3(Func&& f, int iters) {
    // warmup
    for (int i = 0; i < 3; ++i) f();
    WDT_YIELD();

    double results[3];
    for (int r = 0; r < 3; ++r) {
        int64_t t0 = esp_timer_get_time();
        for (int i = 0; i < iters; ++i) {
            f();
        }
        int64_t dt = esp_timer_get_time() - t0;
        results[r] = (double)dt * 1000.0 / iters; // ns per op
        WDT_YIELD();
    }

    // median of 3
    if (results[0] > results[1]) std::swap(results[0], results[1]);
    if (results[1] > results[2]) std::swap(results[1], results[2]);
    if (results[0] > results[1]) std::swap(results[0], results[1]);
    return results[1];
}

// -- Formatting (identical to x86 bench_hornet) -------------------------------

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

// -- Main ---------------------------------------------------------------------

extern "C" void app_main() {
    vTaskDelay(pdMS_TO_TICKS(1000));

    // -- Integrity check ------------------------------------------------------
    printf("Running integrity check... ");
    if (!Selftest(false)) {
        printf("FAIL\n");
        while (1) vTaskDelay(pdMS_TO_TICKS(10000));
        return;
    }
    printf("OK\n");

    // -- Header ---------------------------------------------------------------
    esp_chip_info_t ci;
    esp_chip_info(&ci);

    const char* chip_name = "ESP32";
#ifdef CONFIG_IDF_TARGET_ESP32S3
    chip_name = "ESP32-S3 (Xtensa LX7, dual-core)";
#elif defined(CONFIG_IDF_TARGET_ESP32)
    chip_name = "ESP32 (Xtensa LX6)";
#endif

    printf("\n");
    printf("==========================================================================================\n");
    printf("  UltrafastSecp256k1 -- Bitcoin Consensus CPU Benchmark (Single Core)\n");
    printf("  Target:   Hornet Node (hornetnode.org)\n");
    printf("==========================================================================================\n");
    printf("\n");
    printf("  CPU:       %s @ 240 MHz\n", chip_name);
    printf("  Cores:     %d (single-threaded benchmark)\n", ci.cores);
    printf("  Revision:  %d.%d\n", ci.revision / 100, ci.revision % 100);
    printf("  Free Heap: %lu bytes\n", (unsigned long)esp_get_free_heap_size());
    printf("  Compiler:  GCC %s\n", __VERSION__);
    printf("  Arch:      Xtensa LX7 (32-bit, no __int128, no SIMD)\n");
    printf("  Library:   UltrafastSecp256k1 v3.16.0\n");
    printf("  Field:     %s\n", secp256k1::fast::kOptimalTierName);
    printf("  Scalar:    10x26 limbs (uint32_t), Barrett reduction\n");
    printf("  Point mul: GLV endomorphism + wNAF (w=5)\n");
    printf("  Dual mul:  Shamir's trick (a*G + b*P)\n");
    printf("\n");
    printf("  Timer:    esp_timer (1 us resolution)\n");
    printf("  Method:   median of 3 runs, per-op warmup\n");
    printf("\n");

    // -- Prepare test data ----------------------------------------------------
    constexpr int POOL = 16; // smaller pool for ESP32 SRAM

    Scalar privkeys[POOL];
    for (int i = 0; i < POOL; ++i)
        privkeys[i] = make_scalar(0xdeadbeef00ULL + i);

    Point pubkeys[POOL];
    for (int i = 0; i < POOL; ++i)
        pubkeys[i] = Point::generator().scalar_mul(privkeys[i]);
    WDT_YIELD();

    std::array<std::uint8_t, 32> msghashes[POOL];
    for (int i = 0; i < POOL; ++i)
        msghashes[i] = make_hash(0xcafebabe00ULL + i);

    std::array<std::uint8_t, 32> aux_rands[POOL];
    for (int i = 0; i < POOL; ++i)
        aux_rands[i] = make_hash(0xfeedface00ULL + i);

    ECDSASignature ecdsa_sigs[POOL];
    for (int i = 0; i < POOL; ++i)
        ecdsa_sigs[i] = ecdsa_sign(msghashes[i], privkeys[i]);
    WDT_YIELD();

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
    WDT_YIELD();

    // Iteration counts (lower for ESP32 to avoid excessive runtime)
    constexpr int N_SIGN     = 5;
    constexpr int N_VERIFY   = 5;
    constexpr int N_KEYGEN   = 5;
    constexpr int N_SCALAR   = 5;
    constexpr int N_FIELD    = 500;
    constexpr int N_POINT    = 200;
    constexpr int N_SERIAL   = 500;
    constexpr int N_BATCH    = 3;

    int idx = 0;

    // =========================================================================
    // 1. ECDSA (RFC 6979)
    // =========================================================================

    print_section("ECDSA (RFC 6979)");

    idx = 0;
    const double ecdsa_sign_ns = bench_median3([&]() {
        auto sig = ecdsa_sign(msghashes[idx % POOL], privkeys[idx % POOL]);
        volatile auto sink = sig.r.limbs()[0]; (void)sink;
        ++idx;
    }, N_SIGN);
    print_row("ecdsa_sign (deterministic nonce)", ecdsa_sign_ns);

    idx = 0;
    const double ecdsa_verify_ns = bench_median3([&]() {
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
    const double schnorr_sign_ns = bench_median3([&]() {
        auto sig = schnorr_sign(schnorr_kps[idx % POOL], msghashes[idx % POOL],
                                aux_rands[idx % POOL]);
        volatile auto sink = sig.r[0]; (void)sink;
        ++idx;
    }, N_SIGN);
    print_row("schnorr_sign (pre-computed keypair)", schnorr_sign_ns);

    idx = 0;
    const double schnorr_sign_raw_ns = bench_median3([&]() {
        auto sig = schnorr_sign(privkeys[idx % POOL], msghashes[idx % POOL],
                                aux_rands[idx % POOL]);
        volatile auto sink = sig.r[0]; (void)sink;
        ++idx;
    }, N_SIGN);
    print_row("schnorr_sign (from raw privkey)", schnorr_sign_raw_ns);

    idx = 0;
    const double schnorr_verify_ns = bench_median3([&]() {
        bool ok = schnorr_verify(schnorr_pubkeys_x[idx % POOL],
                                 msghashes[idx % POOL],
                                 schnorr_sigs[idx % POOL]);
        volatile bool sink = ok; (void)sink;
        ++idx;
    }, N_VERIFY);
    print_row("schnorr_verify (x-only 32B pubkey)", schnorr_verify_ns);

    idx = 0;
    const double schnorr_verify_cached_ns = bench_median3([&]() {
        bool ok = schnorr_verify(schnorr_xonly[idx % POOL],
                                 msghashes[idx % POOL],
                                 schnorr_sigs[idx % POOL]);
        volatile bool sink = ok; (void)sink;
        ++idx;
    }, N_VERIFY);
    print_row("schnorr_verify (pre-parsed pubkey)", schnorr_verify_cached_ns);
    print_line();

    // =========================================================================
    // 3. Batch Verification (N=16)
    // =========================================================================

    print_section("Batch Verification (N=16)");

    double schnorr_batch_per_sig = 0;
    {
        std::vector<SchnorrBatchEntry> batch(POOL);
        for (int i = 0; i < POOL; ++i) {
            batch[i].pubkey_x = schnorr_pubkeys_x[i];
            batch[i].message  = msghashes[i];
            batch[i].signature = schnorr_sigs[i];
        }
        WDT_YIELD();
        const double total = bench_median3([&]() {
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
        WDT_YIELD();
        const double total = bench_median3([&]() {
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
    const double keygen_ns = bench_median3([&]() {
        auto pk = Point::generator().scalar_mul(privkeys[idx % POOL]);
        volatile auto sink = pk.x().limbs()[0]; (void)sink;
        ++idx;
    }, N_KEYGEN);
    print_row("pubkey_create (k*G, GLV+wNAF)", keygen_ns);

    idx = 0;
    const double schnorr_keygen_ns = bench_median3([&]() {
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
    const double scalar_mul_ns = bench_median3([&]() {
        auto r = pubkeys[idx % POOL].scalar_mul(privkeys[(idx + 1) % POOL]);
        volatile auto sink = r.x().limbs()[0]; (void)sink;
        ++idx;
    }, N_SCALAR);
    print_row("k*P (arbitrary point, GLV+wNAF)", scalar_mul_ns);

    idx = 0;
    const double dual_mul_ns = bench_median3([&]() {
        auto r = Point::dual_scalar_mul_gen_point(
            privkeys[idx % POOL], privkeys[(idx + 1) % POOL],
            pubkeys[(idx + 2) % POOL]);
        volatile auto sink = r.x().limbs()[0]; (void)sink;
        ++idx;
    }, N_SCALAR);
    print_row("a*G + b*P (Shamir dual mul)", dual_mul_ns);

    const double add_ns = bench_median3([&]() {
        auto r = pubkeys[0].add(pubkeys[1]);
        volatile auto sink = r.x().limbs()[0]; (void)sink;
    }, N_POINT);
    print_row("point_add (Jacobian mixed)", add_ns);

    const double dbl_ns = bench_median3([&]() {
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

    const double fmul_ns = bench_median3([&]() {
        auto r = fe_a * fe_b;
        volatile auto sink = r.limbs()[0]; (void)sink;
    }, N_FIELD);
    print_row("field_mul", fmul_ns);

    const double fsqr_ns = bench_median3([&]() {
        auto r = fe_a.square();
        volatile auto sink = r.limbs()[0]; (void)sink;
    }, N_FIELD);
    print_row("field_sqr", fsqr_ns);

    const double finv_ns = bench_median3([&]() {
        auto r = fe_a.inverse();
        volatile auto sink = r.limbs()[0]; (void)sink;
    }, 20);
    print_row("field_inv (Fermat, 256-bit exp)", finv_ns);

    const double fadd_ns = bench_median3([&]() {
        auto r = fe_a + fe_b;
        volatile auto sink = r.limbs()[0]; (void)sink;
    }, N_FIELD);
    print_row("field_add (mod p)", fadd_ns);

    const double fsub_ns = bench_median3([&]() {
        auto r = fe_a - fe_b;
        volatile auto sink = r.limbs()[0]; (void)sink;
    }, N_FIELD);
    print_row("field_sub (mod p)", fsub_ns);

    const double fneg_ns = bench_median3([&]() {
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

    const double smul_ns = bench_median3([&]() {
        auto r = sc_a * sc_b;
        volatile auto sink = r.limbs()[0]; (void)sink;
    }, N_FIELD);
    print_row("scalar_mul (mod n)", smul_ns);

    const double sinv_ns = bench_median3([&]() {
        auto r = sc_a.inverse();
        volatile auto sink = r.limbs()[0]; (void)sink;
    }, 20);
    print_row("scalar_inv (mod n)", sinv_ns);

    const double sadd_ns = bench_median3([&]() {
        auto r = sc_a + sc_b;
        volatile auto sink = r.limbs()[0]; (void)sink;
    }, N_FIELD);
    print_row("scalar_add (mod n)", sadd_ns);

    const double sneg_ns = bench_median3([&]() {
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
    const double compress_ns = bench_median3([&]() {
        auto c = pubkeys[idx % POOL].to_compressed();
        volatile auto sink = c[0]; (void)sink;
        ++idx;
    }, N_SERIAL);
    print_row("pubkey_serialize (33B compressed)", compress_ns);

    idx = 0;
    const double der_encode_ns = bench_median3([&]() {
        auto d = ecdsa_sigs[idx % POOL].to_der();
        volatile auto sink = d.first[0]; (void)sink;
        ++idx;
    }, N_SERIAL);
    print_row("ecdsa_sig_to_der (DER encode)", der_encode_ns);

    idx = 0;
    const double schnorr_ser_ns = bench_median3([&]() {
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
    const double ct_ecdsa_ns = bench_median3([&]() {
        auto sig = ct::ecdsa_sign(msghashes[idx % POOL], privkeys[idx % POOL]);
        volatile auto sink = sig.r.limbs()[0]; (void)sink;
        ++idx;
    }, N_SIGN);
    print_row("ct::ecdsa_sign", ct_ecdsa_ns);
    print_ratio_row("  -> CT overhead vs fast::ecdsa_sign", ct_ecdsa_ns / ecdsa_sign_ns);

    idx = 0;
    const double ct_schnorr_ns = bench_median3([&]() {
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
    WDT_YIELD();

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
    printf("  BITCOIN BLOCK VALIDATION ESTIMATES (1 core, ESP32-S3 @ 240 MHz)\n");
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

    // Transaction processing rate
    const double ecdsa_per_sec = 1e9 / ecdsa_verify_ns;
    const double schnorr_per_sec = 1e9 / schnorr_verify_ns;
    printf("  Transaction throughput (1-input txs, 1 core):\n");
    printf("    ECDSA txs:    %8.0f tx/sec\n", ecdsa_per_sec);
    printf("    Schnorr txs:  %8.0f tx/sec\n", schnorr_per_sec);
    printf("\n");

    // Blocks per second
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
    printf("  - All measurements: single-threaded, single core\n");
    printf("  - Timer: esp_timer (1 us resolution)\n");
    printf("  - Each operation: warmup + median of 3 runs\n");
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
    printf("  %s @ 240 MHz | 1 core | GCC %s | UltrafastSecp256k1 v3.16.0\n",
           chip_name, __VERSION__);
    printf("==========================================================================================\n\n");

    printf("BENCH_HORNET_COMPLETE\n");

    // idle loop
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(10000));
    }
}
