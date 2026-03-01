/**
 * libsecp256k1 (bitcoin-core) benchmark wrapper for ESP32.
 * Compiles the official library as a single translation unit
 * and provides a timing function for generator_mul comparison.
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
#include <stdio.h>
#include <string.h>

// Fixed test secret key (same scalar we use in our benchmark)
static const unsigned char test_seckey[32] = {
    0x47, 0x27, 0xda, 0xf2, 0x98, 0x6a, 0x98, 0x04,
    0xb1, 0x11, 0x7f, 0x82, 0x61, 0xab, 0xa6, 0x45,
    0xc3, 0x45, 0x37, 0xe4, 0x47, 0x4e, 0x19, 0xbe,
    0x58, 0x70, 0x07, 0x92, 0xd5, 0x01, 0xa5, 0x91
};

void libsecp_benchmark(void) {
    printf("\n");
    printf("==============================================\n");
    printf("  libsecp256k1 (bitcoin-core) Benchmark\n");
    printf("  Version: 0.7.2\n");
    printf("  Table:   COMB %dx%d (%dKB)\n", COMB_BLOCKS, COMB_TEETH, 22);
    printf("  Modules: ECDSA + Schnorr (BIP-340)\n");
    printf("==============================================\n");

    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_NONE);
    if (!ctx) {
        printf("  ERROR: context creation failed\n");
        return;
    }

    secp256k1_pubkey pubkey;
    volatile uint64_t sink = 0;

    // Warmup (also triggers lazy table init)
    secp256k1_ec_pubkey_create(ctx, &pubkey, test_seckey);
    sink = pubkey.data[0];

    // -- Generator Multiplication (ec_pubkey_create) --
    {
        int64_t start = esp_timer_get_time();
        for (int i = 0; i < 3; i++) {
            secp256k1_ec_pubkey_create(ctx, &pubkey, test_seckey);
        }
        int64_t elapsed = esp_timer_get_time() - start;
        sink ^= pubkey.data[0];
        printf("  Generator*k:      %5lld us/op  (ec_pubkey_create)\n", elapsed / 3);
    }

    // -- ECDSA Sign --
    {
        unsigned char msg[32];
        memset(msg, 0x42, 32);
        secp256k1_ecdsa_signature sig;

        // Warmup
        secp256k1_ecdsa_sign(ctx, &sig, msg, test_seckey, NULL, NULL);

        int64_t start = esp_timer_get_time();
        for (int i = 0; i < 3; i++) {
            msg[0] = (unsigned char)i;
            secp256k1_ecdsa_sign(ctx, &sig, msg, test_seckey, NULL, NULL);
        }
        int64_t elapsed = esp_timer_get_time() - start;
        sink ^= sig.data[0];
        printf("  ECDSA Sign:       %5lld us/op\n", elapsed / 3);
    }

    // -- ECDSA Verify --
    {
        unsigned char msg[32];
        memset(msg, 0x42, 32);
        secp256k1_ecdsa_signature sig;
        secp256k1_ec_pubkey_create(ctx, &pubkey, test_seckey);
        secp256k1_ecdsa_sign(ctx, &sig, msg, test_seckey, NULL, NULL);

        // Warmup
        secp256k1_ecdsa_verify(ctx, &sig, msg, &pubkey);

        int64_t start = esp_timer_get_time();
        int ok = 1;
        for (int i = 0; i < 3; i++) {
            ok &= secp256k1_ecdsa_verify(ctx, &sig, msg, &pubkey);
        }
        int64_t elapsed = esp_timer_get_time() - start;
        sink ^= (uint64_t)ok;
        printf("  ECDSA Verify:     %5lld us/op\n", elapsed / 3);
    }

    // -- Schnorr Keypair Create --
    {
        secp256k1_keypair keypair;

        // Warmup
        secp256k1_keypair_create(ctx, &keypair, test_seckey);

        int64_t start = esp_timer_get_time();
        for (int i = 0; i < 3; i++) {
            secp256k1_keypair_create(ctx, &keypair, test_seckey);
        }
        int64_t elapsed = esp_timer_get_time() - start;
        sink ^= keypair.data[0];
        printf("  Schnorr Keypair:  %5lld us/op  (keypair_create)\n", elapsed / 3);
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

        // Warmup
        secp256k1_schnorrsig_sign32(ctx, sig64, msg, &keypair, aux);

        int64_t start = esp_timer_get_time();
        for (int i = 0; i < 3; i++) {
            msg[0] = (unsigned char)(i + 0x10);
            secp256k1_schnorrsig_sign32(ctx, sig64, msg, &keypair, aux);
        }
        int64_t elapsed = esp_timer_get_time() - start;
        sink ^= sig64[0];
        printf("  Schnorr Sign:     %5lld us/op  (BIP-340)\n", elapsed / 3);
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

        // Warmup
        secp256k1_schnorrsig_verify(ctx, sig64, msg, 32, &xonly_pk);

        int64_t start = esp_timer_get_time();
        int ok = 1;
        for (int i = 0; i < 3; i++) {
            ok &= secp256k1_schnorrsig_verify(ctx, sig64, msg, 32, &xonly_pk);
        }
        int64_t elapsed = esp_timer_get_time() - start;
        sink ^= (uint64_t)ok;
        printf("  Schnorr Verify:   %5lld us/op  (BIP-340)\n", elapsed / 3);
    }

    (void)sink;
    secp256k1_context_destroy(ctx);

    printf("  --------------------------------------------\n");
}
