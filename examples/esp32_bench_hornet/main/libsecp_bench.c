/**
 * libsecp256k1 (bitcoin-core) benchmark wrapper for ESP32.
 * Compiles the official library as a single translation unit.
 * Returns timing results for apple-to-apple ratio computation.
 *
 * Uses median-of-3 (identical harness to Ultra benchmark).
 */

// --- libsecp256k1 configuration for ESP32 ------------------------------------
// Small precompute tables for embedded: 22KB signing + minimal verify
#define ECMULT_WINDOW_SIZE 2
#define COMB_BLOCKS        11
#define COMB_TEETH         6

// Enable modules for benchmark
#define ENABLE_MODULE_ECDH 0
#define ENABLE_MODULE_RECOVERY 0
#define ENABLE_MODULE_EXTRAKEYS 1
#define ENABLE_MODULE_SCHNORRSIG 1
#define ENABLE_MODULE_MUSIG 0
#define ENABLE_MODULE_ELLSWIFT 0

// --- Include the entire libsecp256k1 as a single compilation unit ------------
// Path relative to the ESP32 test project main/ directory
#include "../../../../../_research_repos/secp256k1/src/secp256k1.c"

// --- ESP32 benchmark API -----------------------------------------------------
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <stdio.h>
#include <string.h>

#include "libsecp_bench.h"

// Fixed test secret key (same scalar we use in our benchmark)
static const unsigned char test_seckey[32] = {
    0x47, 0x27, 0xda, 0xf2, 0x98, 0x6a, 0x98, 0x04,
    0xb1, 0x11, 0x7f, 0x82, 0x61, 0xab, 0xa6, 0x45,
    0xc3, 0x45, 0x37, 0xe4, 0x47, 0x4e, 0x19, 0xbe,
    0x58, 0x70, 0x07, 0x92, 0xd5, 0x01, 0xa5, 0x91
};

/* Median of 3 doubles */
static double median3(double a, double b, double c) {
    if (a > b) { double t = a; a = b; b = t; }
    if (b > c) { double t = b; b = c; c = t; }
    if (a > b) { double t = a; a = b; b = t; }
    return b;
}

/* Run func N times, return ns/op.  Repeat 3x, yield, return median. */
#define BENCH_MEDIAN3(out_ns, N, body)                  \
    do {                                                 \
        double _r[3];                                    \
        for (int _pass = 0; _pass < 3; ++_pass) {       \
            int64_t _t0 = esp_timer_get_time();          \
            for (int _i = 0; _i < (N); ++_i) { body; }  \
            int64_t _dt = esp_timer_get_time() - _t0;    \
            _r[_pass] = (double)_dt * 1000.0 / (N);     \
            vTaskDelay(pdMS_TO_TICKS(10));                \
        }                                                \
        (out_ns) = median3(_r[0], _r[1], _r[2]);        \
    } while (0)

#define LIBSECP_ITERS 5

void libsecp_benchmark(libsecp_results_t* out) {
    memset(out, 0, sizeof(*out));

    printf("\n");
    printf("+----------------------------------------------+------------+\n");
    printf("| libsecp256k1 (bitcoin-core v0.7.2)           |      ns/op |\n");
    printf("| ECMULT_WINDOW=%d, COMB %dx%d                   |            |\n",
           ECMULT_WINDOW_SIZE, COMB_BLOCKS, COMB_TEETH);
    printf("+----------------------------------------------+------------+\n");

    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_NONE);
    if (!ctx) {
        printf("| ERROR: context creation failed               |            |\n");
        printf("+----------------------------------------------+------------+\n");
        return;
    }

    secp256k1_pubkey pubkey;
    volatile uint64_t sink = 0;

    /* Warmup (triggers lazy table init) */
    secp256k1_ec_pubkey_create(ctx, &pubkey, test_seckey);
    sink = pubkey.data[0];
    vTaskDelay(pdMS_TO_TICKS(10));

    /* Generator * k */
    BENCH_MEDIAN3(out->generator_mul_ns, LIBSECP_ITERS, {
        secp256k1_ec_pubkey_create(ctx, &pubkey, test_seckey);
        sink ^= pubkey.data[0];
    });
    printf("| %-44s | %10.1f |\n", "generator_mul (ec_pubkey_create)", out->generator_mul_ns);

    /* ECDSA Sign */
    {
        unsigned char msg[32]; memset(msg, 0x42, 32);
        secp256k1_ecdsa_signature sig;
        secp256k1_ecdsa_sign(ctx, &sig, msg, test_seckey, NULL, NULL);
        int idx = 0;
        BENCH_MEDIAN3(out->ecdsa_sign_ns, LIBSECP_ITERS, {
            msg[0] = (unsigned char)(idx++ & 0xFF);
            secp256k1_ecdsa_sign(ctx, &sig, msg, test_seckey, NULL, NULL);
            sink ^= sig.data[0];
        });
        printf("| %-44s | %10.1f |\n", "ecdsa_sign", out->ecdsa_sign_ns);
    }

    /* ECDSA Verify */
    {
        unsigned char msg[32]; memset(msg, 0x42, 32);
        secp256k1_ecdsa_signature sig;
        secp256k1_ec_pubkey_create(ctx, &pubkey, test_seckey);
        secp256k1_ecdsa_sign(ctx, &sig, msg, test_seckey, NULL, NULL);
        secp256k1_ecdsa_verify(ctx, &sig, msg, &pubkey);
        BENCH_MEDIAN3(out->ecdsa_verify_ns, LIBSECP_ITERS, {
            volatile int ok = secp256k1_ecdsa_verify(ctx, &sig, msg, &pubkey);
            sink ^= (uint64_t)ok;
        });
        printf("| %-44s | %10.1f |\n", "ecdsa_verify", out->ecdsa_verify_ns);
    }

    /* Schnorr Keypair Create */
    {
        secp256k1_keypair keypair;
        secp256k1_keypair_create(ctx, &keypair, test_seckey);
        BENCH_MEDIAN3(out->schnorr_keypair_ns, LIBSECP_ITERS, {
            secp256k1_keypair_create(ctx, &keypair, test_seckey);
            sink ^= keypair.data[0];
        });
        printf("| %-44s | %10.1f |\n", "schnorr_keypair_create", out->schnorr_keypair_ns);
    }

    /* Schnorr Sign (BIP-340) */
    {
        secp256k1_keypair keypair;
        secp256k1_keypair_create(ctx, &keypair, test_seckey);
        unsigned char msg[32]; memset(msg, 0x42, 32);
        unsigned char sig64[64];
        unsigned char aux[32]; memset(aux, 0x11, 32);
        secp256k1_schnorrsig_sign32(ctx, sig64, msg, &keypair, aux);
        int idx = 0;
        BENCH_MEDIAN3(out->schnorr_sign_ns, LIBSECP_ITERS, {
            msg[0] = (unsigned char)((idx++ + 0x10) & 0xFF);
            secp256k1_schnorrsig_sign32(ctx, sig64, msg, &keypair, aux);
            sink ^= sig64[0];
        });
        printf("| %-44s | %10.1f |\n", "schnorr_sign (BIP-340)", out->schnorr_sign_ns);
    }

    /* Schnorr Verify (BIP-340) */
    {
        secp256k1_keypair keypair;
        secp256k1_keypair_create(ctx, &keypair, test_seckey);
        secp256k1_xonly_pubkey xonly_pk;
        secp256k1_keypair_xonly_pub(ctx, &xonly_pk, NULL, &keypair);
        unsigned char msg[32]; memset(msg, 0x42, 32);
        unsigned char sig64[64];
        unsigned char aux[32]; memset(aux, 0x11, 32);
        secp256k1_schnorrsig_sign32(ctx, sig64, msg, &keypair, aux);
        secp256k1_schnorrsig_verify(ctx, sig64, msg, 32, &xonly_pk);
        BENCH_MEDIAN3(out->schnorr_verify_ns, LIBSECP_ITERS, {
            volatile int ok = secp256k1_schnorrsig_verify(ctx, sig64, msg, 32, &xonly_pk);
            sink ^= (uint64_t)ok;
        });
        printf("| %-44s | %10.1f |\n", "schnorr_verify (BIP-340)", out->schnorr_verify_ns);
    }

    (void)sink;
    secp256k1_context_destroy(ctx);

    printf("+----------------------------------------------+------------+\n");
}
