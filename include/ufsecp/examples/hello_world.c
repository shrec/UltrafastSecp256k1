/* ============================================================================
 * hello_world.c — Minimal ufsecp usage example
 * ============================================================================
 * Demonstrates:  context → keygen → ECDSA sign → verify → cleanup
 *
 * Build (gcc/clang):
 *   cc -o hello_world hello_world.c -lufsecp
 *
 * Or with CMake:
 *   target_link_libraries(hello_world PRIVATE ufsecp::ufsecp)
 * ============================================================================ */

#include <stdio.h>
#include <string.h>
#include "ufsecp/ufsecp.h"

int main(void) {
    ufsecp_ctx* ctx = NULL;
    ufsecp_error_t err;

    /* ── 1. Create context ──────────────────────────────────────────────── */
    err = ufsecp_ctx_create(&ctx);
    if (err != UFSECP_OK) {
        fprintf(stderr, "ctx_create: %s\n", ufsecp_error_str(err));
        return 1;
    }
    printf("ufsecp %s (ABI %u)\n", ufsecp_version_string(), ufsecp_abi_version());

    /* ── 2. Private key (hard-coded for demo — use real entropy!) ─────── */
    uint8_t privkey[32] = {
        0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
        0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
        0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
        0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01
    };

    /* ── 3. Derive public key ─────────────────────────────────────────── */
    uint8_t pubkey[33];
    err = ufsecp_pubkey_create(ctx, privkey, pubkey);
    if (err != UFSECP_OK) {
        fprintf(stderr, "pubkey_create: %s\n", ufsecp_error_str(err));
        ufsecp_ctx_destroy(ctx);
        return 1;
    }
    printf("pubkey: ");
    for (int i = 0; i < 33; i++) printf("%02x", pubkey[i]);
    printf("\n");

    /* ── 4. SHA-256 hash a message ────────────────────────────────────── */
    const char* msg = "hello ufsecp";
    uint8_t hash[32];
    err = ufsecp_sha256((const uint8_t*)msg, strlen(msg), hash);
    if (err != UFSECP_OK) {
        fprintf(stderr, "sha256: %s\n", ufsecp_error_str(err));
        ufsecp_ctx_destroy(ctx);
        return 1;
    }

    /* ── 5. ECDSA sign ────────────────────────────────────────────────── */
    uint8_t sig[64];
    err = ufsecp_ecdsa_sign(ctx, hash, privkey, sig);
    if (err != UFSECP_OK) {
        fprintf(stderr, "ecdsa_sign: %s\n", ufsecp_error_str(err));
        ufsecp_ctx_destroy(ctx);
        return 1;
    }
    printf("sig:    ");
    for (int i = 0; i < 64; i++) printf("%02x", sig[i]);
    printf("\n");

    /* ── 6. ECDSA verify ──────────────────────────────────────────────── */
    err = ufsecp_ecdsa_verify(ctx, hash, sig, pubkey);
    if (err == UFSECP_OK)
        printf("verify: OK\n");
    else
        printf("verify: FAIL (%s)\n", ufsecp_error_str(err));

    /* ── 7. Schnorr (BIP-340) sign + verify ───────────────────────────── */
    uint8_t xpub[32];
    err = ufsecp_pubkey_xonly(ctx, privkey, xpub);
    if (err != UFSECP_OK) {
        fprintf(stderr, "pubkey_xonly: %s\n", ufsecp_error_str(err));
        ufsecp_ctx_destroy(ctx);
        return 1;
    }

    uint8_t aux[32] = {0};  /* deterministic: all zeros */
    uint8_t schnorr_sig[64];
    err = ufsecp_schnorr_sign(ctx, hash, privkey, aux, schnorr_sig);
    if (err == UFSECP_OK) {
        err = ufsecp_schnorr_verify(ctx, hash, schnorr_sig, xpub);
        printf("schnorr: %s\n", err == UFSECP_OK ? "OK" : "FAIL");
    }

    /* ── 8. Cleanup ───────────────────────────────────────────────────── */
    ufsecp_ctx_destroy(ctx);
    return 0;
}
