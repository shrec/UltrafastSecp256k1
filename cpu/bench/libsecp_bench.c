/**
 * libsecp_bench.c -- libsecp256k1 (bitcoin-core) apple-to-apple benchmark
 *
 * Compiles the official bitcoin-core libsecp256k1 as a single translation
 * unit and provides libsecp_benchmark() callable from bench_hornet.cpp.
 *
 * Uses high-resolution timer for x86-64 (QueryPerformanceCounter on Windows,
 * clock_gettime on Linux/Mac).
 *
 * Build: compiled as C (not C++), linked with bench_hornet target.
 */

// --- libsecp256k1 configuration (x86-64 desktop) ----------------------------
// Standard table sizes for comparison fairness (same as bitcoin-core defaults)
// ECMULT_WINDOW_SIZE default is 15, COMB default is 11x6
// We do NOT override -- use the library defaults for fair comparison.

// Enable modules matching our benchmark
#define ENABLE_MODULE_ECDH 0
#define ENABLE_MODULE_RECOVERY 0
#define ENABLE_MODULE_EXTRAKEYS 1
#define ENABLE_MODULE_SCHNORRSIG 1
#define ENABLE_MODULE_MUSIG 0
#define ENABLE_MODULE_ELLSWIFT 0

// --- Include the entire libsecp256k1 as a single compilation unit ------------
#include "../../../../_research_repos/secp256k1/src/secp256k1.c"

// --- Platform timer ----------------------------------------------------------
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#if defined(_WIN32) || defined(_WIN64)
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  static double get_time_us(void) {
      LARGE_INTEGER freq, cnt;
      QueryPerformanceFrequency(&freq);
      QueryPerformanceCounter(&cnt);
      return (double)cnt.QuadPart / (double)freq.QuadPart * 1e6;
  }
#else
  #include <time.h>
  static double get_time_us(void) {
      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
  }
#endif

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
    printf("\n");
    printf("==========================================================================================\n");
    printf("  APPLE-TO-APPLE: libsecp256k1 (bitcoin-core v0.7.2)\n");
    printf("==========================================================================================\n\n");
    printf("  Same hardware, same OS, same test key.\n");
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
    for (int i = 0; i < BENCH_WARMUP; i++) {
        secp256k1_ec_pubkey_create(ctx, &pubkey, test_seckey);
    }
    sink = pubkey.data[0];

    // -- Generator Multiplication (ec_pubkey_create) --
    {
        double start = get_time_us();
        for (int i = 0; i < BENCH_ITERS; i++) {
            secp256k1_ec_pubkey_create(ctx, &pubkey, test_seckey);
        }
        double elapsed = get_time_us() - start;
        sink ^= pubkey.data[0];
        printf("  Generator*k:      %8.1f ns/op  (ec_pubkey_create)\n", elapsed * 1e3 / BENCH_ITERS);
    }

    // -- ECDSA Sign --
    {
        unsigned char msg[32];
        memset(msg, 0x42, 32);
        secp256k1_ecdsa_signature sig;

        for (int i = 0; i < BENCH_WARMUP; i++) {
            msg[0] = (unsigned char)i;
            secp256k1_ecdsa_sign(ctx, &sig, msg, test_seckey, NULL, NULL);
        }

        double start = get_time_us();
        for (int i = 0; i < BENCH_ITERS; i++) {
            msg[0] = (unsigned char)(i + 0x80);
            secp256k1_ecdsa_sign(ctx, &sig, msg, test_seckey, NULL, NULL);
        }
        double elapsed = get_time_us() - start;
        sink ^= sig.data[0];
        printf("  ECDSA Sign:       %8.1f ns/op\n", elapsed * 1e3 / BENCH_ITERS);
    }

    // -- ECDSA Verify --
    {
        unsigned char msg[32];
        memset(msg, 0x42, 32);
        secp256k1_ecdsa_signature sig;
        secp256k1_ec_pubkey_create(ctx, &pubkey, test_seckey);
        secp256k1_ecdsa_sign(ctx, &sig, msg, test_seckey, NULL, NULL);

        for (int i = 0; i < BENCH_WARMUP; i++) {
            secp256k1_ecdsa_verify(ctx, &sig, msg, &pubkey);
        }

        double start = get_time_us();
        int ok = 1;
        for (int i = 0; i < BENCH_ITERS; i++) {
            ok &= secp256k1_ecdsa_verify(ctx, &sig, msg, &pubkey);
        }
        double elapsed = get_time_us() - start;
        sink ^= (uint64_t)ok;
        printf("  ECDSA Verify:     %8.1f ns/op\n", elapsed * 1e3 / BENCH_ITERS);
    }

    // -- Schnorr Keypair Create --
    {
        secp256k1_keypair keypair;

        for (int i = 0; i < BENCH_WARMUP; i++) {
            secp256k1_keypair_create(ctx, &keypair, test_seckey);
        }

        double start = get_time_us();
        for (int i = 0; i < BENCH_ITERS; i++) {
            secp256k1_keypair_create(ctx, &keypair, test_seckey);
        }
        double elapsed = get_time_us() - start;
        sink ^= keypair.data[0];
        printf("  Schnorr Keypair:  %8.1f ns/op  (keypair_create)\n", elapsed * 1e3 / BENCH_ITERS);
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

        for (int i = 0; i < BENCH_WARMUP; i++) {
            msg[0] = (unsigned char)i;
            secp256k1_schnorrsig_sign32(ctx, sig64, msg, &keypair, aux);
        }

        double start = get_time_us();
        for (int i = 0; i < BENCH_ITERS; i++) {
            msg[0] = (unsigned char)(i + 0x80);
            secp256k1_schnorrsig_sign32(ctx, sig64, msg, &keypair, aux);
        }
        double elapsed = get_time_us() - start;
        sink ^= sig64[0];
        printf("  Schnorr Sign:     %8.1f ns/op  (BIP-340)\n", elapsed * 1e3 / BENCH_ITERS);
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

        for (int i = 0; i < BENCH_WARMUP; i++) {
            secp256k1_schnorrsig_verify(ctx, sig64, msg, 32, &xonly_pk);
        }

        double start = get_time_us();
        int ok = 1;
        for (int i = 0; i < BENCH_ITERS; i++) {
            ok &= secp256k1_schnorrsig_verify(ctx, sig64, msg, 32, &xonly_pk);
        }
        double elapsed = get_time_us() - start;
        sink ^= (uint64_t)ok;
        printf("  Schnorr Verify:   %8.1f ns/op  (BIP-340)\n", elapsed * 1e3 / BENCH_ITERS);
    }

    (void)sink;
    secp256k1_context_destroy(ctx);

    printf("  --------------------------------------------\n");
}
