/**
 * libsecp_bench.c -- libsecp256k1 (bitcoin-core) apple-to-apple benchmark
 *                    for Android ARM64.
 *
 * Compiles the official bitcoin-core libsecp256k1 as a single translation
 * unit and provides libsecp_benchmark() callable from bench_hornet_android.cpp.
 *
 * Uses clock_gettime(CLOCK_MONOTONIC) for timing.
 */

// --- libsecp256k1 configuration (ARM64) --------------------------------------
// Use default table sizes for fair comparison (same as bitcoin-core defaults)

// Enable modules matching our benchmark
#define ENABLE_MODULE_ECDH 0
#define ENABLE_MODULE_RECOVERY 0
#define ENABLE_MODULE_EXTRAKEYS 1
#define ENABLE_MODULE_SCHNORRSIG 1
#define ENABLE_MODULE_MUSIG 0
#define ENABLE_MODULE_ELLSWIFT 0

// --- Include the entire libsecp256k1 as a single compilation unit ------------
// secp256k1/src is in include path via CMakeLists.txt LIBSECP_SRC_DIR
#include "secp256k1.c"

// --- Platform timer ----------------------------------------------------------
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

static double get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

// Fixed test secret key (same scalar as UltrafastSecp256k1 benchmarks)
static const unsigned char test_seckey[32] = {
    0x47, 0x27, 0xda, 0xf2, 0x98, 0x6a, 0x98, 0x04,
    0xb1, 0x11, 0x7f, 0x82, 0x61, 0xab, 0xa6, 0x45,
    0xc3, 0x45, 0x37, 0xe4, 0x47, 0x4e, 0x19, 0xbe,
    0x58, 0x70, 0x07, 0x92, 0xd5, 0x01, 0xa5, 0x91
};

#define BENCH_ITERS 100
#define BENCH_WARMUP 20

void libsecp_benchmark(void) {
    printf("  Same hardware, same compiler, same test key.\n");
    printf("  Modules: ECDSA + Schnorr (BIP-340) + extrakeys\n");
    printf("  Iterations: %d (warmup: %d)\n\n", BENCH_ITERS, BENCH_WARMUP);

    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_NONE);
    if (!ctx) {
        printf("  ERROR: context creation failed\n");
        return;
    }

    secp256k1_pubkey pubkey;
    volatile uint64_t sink = 0;

    // Warmup (triggers lazy table init)
    for (int i = 0; i < BENCH_WARMUP; ++i) {
        secp256k1_ec_pubkey_create(ctx, &pubkey, test_seckey);
    }
    sink = pubkey.data[0];

    // -- Generator Multiplication (ec_pubkey_create) --
    {
        double t0 = get_time_ns();
        for (int i = 0; i < BENCH_ITERS; i++) {
            secp256k1_ec_pubkey_create(ctx, &pubkey, test_seckey);
        }
        double dt = get_time_ns() - t0;
        sink ^= pubkey.data[0];
        printf("  Generator*k:      %8.1f ns/op  (ec_pubkey_create)\n", dt / BENCH_ITERS);
    }

    // -- ECDSA Sign --
    {
        unsigned char msg[32];
        memset(msg, 0x42, 32);
        secp256k1_ecdsa_signature sig;

        for (int i = 0; i < BENCH_WARMUP; ++i) {
            msg[0] = (unsigned char)i;
            secp256k1_ecdsa_sign(ctx, &sig, msg, test_seckey, NULL, NULL);
        }

        double t0 = get_time_ns();
        for (int i = 0; i < BENCH_ITERS; i++) {
            msg[0] = (unsigned char)i;
            secp256k1_ecdsa_sign(ctx, &sig, msg, test_seckey, NULL, NULL);
        }
        double dt = get_time_ns() - t0;
        sink ^= sig.data[0];
        printf("  ECDSA Sign:       %8.1f ns/op\n", dt / BENCH_ITERS);
    }

    // -- ECDSA Verify --
    {
        unsigned char msg[32];
        memset(msg, 0x42, 32);
        secp256k1_ecdsa_signature sig;
        secp256k1_ec_pubkey_create(ctx, &pubkey, test_seckey);
        secp256k1_ecdsa_sign(ctx, &sig, msg, test_seckey, NULL, NULL);

        for (int i = 0; i < BENCH_WARMUP; ++i) {
            secp256k1_ecdsa_verify(ctx, &sig, msg, &pubkey);
        }

        double t0 = get_time_ns();
        int ok = 1;
        for (int i = 0; i < BENCH_ITERS; i++) {
            ok &= secp256k1_ecdsa_verify(ctx, &sig, msg, &pubkey);
        }
        double dt = get_time_ns() - t0;
        sink ^= (uint64_t)ok;
        printf("  ECDSA Verify:     %8.1f ns/op\n", dt / BENCH_ITERS);
    }

    // -- Schnorr Keypair Create --
    {
        secp256k1_keypair keypair;

        for (int i = 0; i < BENCH_WARMUP; ++i) {
            secp256k1_keypair_create(ctx, &keypair, test_seckey);
        }

        double t0 = get_time_ns();
        for (int i = 0; i < BENCH_ITERS; i++) {
            secp256k1_keypair_create(ctx, &keypair, test_seckey);
        }
        double dt = get_time_ns() - t0;
        sink ^= keypair.data[0];
        printf("  Schnorr Keypair:  %8.1f ns/op  (keypair_create)\n", dt / BENCH_ITERS);
    }

    // -- Schnorr Sign (BIP-340) --
    {
        secp256k1_keypair keypair;
        secp256k1_keypair_create(ctx, &keypair, test_seckey);

        unsigned char msg[32];
        memset(msg, 0x42, 32);
        unsigned char sig64[64];
        unsigned char aux[32];
        memset(aux, 0x11, 32);

        for (int i = 0; i < BENCH_WARMUP; ++i) {
            msg[0] = (unsigned char)(i + 0x10);
            secp256k1_schnorrsig_sign32(ctx, sig64, msg, &keypair, aux);
        }

        double t0 = get_time_ns();
        for (int i = 0; i < BENCH_ITERS; i++) {
            msg[0] = (unsigned char)(i + 0x10);
            secp256k1_schnorrsig_sign32(ctx, sig64, msg, &keypair, aux);
        }
        double dt = get_time_ns() - t0;
        sink ^= sig64[0];
        printf("  Schnorr Sign:     %8.1f ns/op  (BIP-340)\n", dt / BENCH_ITERS);
    }

    // -- Schnorr Verify (BIP-340) --
    {
        secp256k1_keypair keypair;
        secp256k1_keypair_create(ctx, &keypair, test_seckey);

        secp256k1_xonly_pubkey xonly_pk;
        secp256k1_keypair_xonly_pub(ctx, &xonly_pk, NULL, &keypair);

        unsigned char msg[32];
        memset(msg, 0x42, 32);
        unsigned char sig64[64];
        unsigned char aux[32];
        memset(aux, 0x11, 32);
        secp256k1_schnorrsig_sign32(ctx, sig64, msg, &keypair, aux);

        for (int i = 0; i < BENCH_WARMUP; ++i) {
            secp256k1_schnorrsig_verify(ctx, sig64, msg, 32, &xonly_pk);
        }

        double t0 = get_time_ns();
        int ok = 1;
        for (int i = 0; i < BENCH_ITERS; i++) {
            ok &= secp256k1_schnorrsig_verify(ctx, sig64, msg, 32, &xonly_pk);
        }
        double dt = get_time_ns() - t0;
        sink ^= (uint64_t)ok;
        printf("  Schnorr Verify:   %8.1f ns/op  (BIP-340)\n", dt / BENCH_ITERS);
    }

    printf("  --------------------------------------------\n");

    (void)sink;
    secp256k1_context_destroy(ctx);
}
