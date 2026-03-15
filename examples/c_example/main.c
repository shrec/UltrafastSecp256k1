/* ============================================================================
 * UltrafastSecp256k1 -- C Example (CPU + GPU)
 * ============================================================================
 *
 * Demonstrates the stable C ABI: key operations, ECDSA, Schnorr, ECDH,
 * hashing, Bitcoin addresses, BIP-32, Taproot, and GPU batch operations.
 *
 * Build:
 *   gcc -O2 -o c_example main.c \
 *       -I../../include/ufsecp \
 *       -L../../build-linux/include/ufsecp -lufsecp \
 *       -Wl,-rpath,'$ORIGIN/../../build-linux/include/ufsecp'
 *
 * Run:
 *   LD_LIBRARY_PATH=../../build-linux/include/ufsecp ./c_example
 * ============================================================================ */

#include "ufsecp.h"
#include "ufsecp_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Helpers ─────────────────────────────────────────────────────────────── */

static void hex_print(const char* label, const uint8_t* data, size_t len)
{
    printf("  %-20s ", label);
    for (size_t i = 0; i < len; ++i) printf("%02x", data[i]);
    printf("\n");
}

static void check(ufsecp_error_t rc, const char* op)
{
    if (rc != UFSECP_OK) {
        fprintf(stderr, "[FAIL] %s: %s (code %d)\n", op, ufsecp_error_str(rc), rc);
        exit(1);
    }
}

#define OK(call, label) check((call), (label))

/* ── Test key: privkey = 1 ───────────────────────────────────────────────── */

static const uint8_t PRIVKEY[32] = {
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,1
};

/* Second key for ECDH */
static const uint8_t PRIVKEY2[32] = {
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,2
};

/* ── CPU Examples ────────────────────────────────────────────────────────── */

static void demo_cpu(ufsecp_ctx* ctx)
{
    printf("=== CPU Operations ===\n\n");

    /* -- 1. Key Generation ------------------------------------------------ */
    printf("[1] Key Generation\n");

    uint8_t pubkey33[33], pubkey65[65], xonly32[32];
    OK(ufsecp_pubkey_create(ctx, PRIVKEY, pubkey33), "pubkey_create");
    OK(ufsecp_pubkey_create_uncompressed(ctx, PRIVKEY, pubkey65), "pubkey_uncompressed");
    OK(ufsecp_pubkey_xonly(ctx, PRIVKEY, xonly32), "pubkey_xonly");

    hex_print("Private key:", PRIVKEY, 32);
    hex_print("Compressed (33B):", pubkey33, 33);
    hex_print("Uncompressed (65B):", pubkey65, 65);
    hex_print("X-only (32B):", xonly32, 32);
    printf("\n");

    /* -- 2. ECDSA (RFC 6979) ---------------------------------------------- */
    printf("[2] ECDSA Sign / Verify (RFC 6979)\n");

    uint8_t msg[32];
    OK(ufsecp_sha256((const uint8_t*)"Hello UltrafastSecp256k1!", 24, msg), "sha256");
    hex_print("Message hash:", msg, 32);

    uint8_t sig64[64];
    OK(ufsecp_ecdsa_sign(ctx, msg, PRIVKEY, sig64), "ecdsa_sign");
    hex_print("ECDSA signature:", sig64, 64);

    ufsecp_error_t vrc = ufsecp_ecdsa_verify(ctx, msg, sig64, pubkey33);
    printf("  %-20s %s\n", "Verify:", vrc == UFSECP_OK ? "VALID" : "INVALID");

    /* DER encoding */
    uint8_t der[72];
    size_t der_len = sizeof(der);
    OK(ufsecp_ecdsa_sig_to_der(ctx, sig64, der, &der_len), "sig_to_der");
    printf("  %-20s %zu bytes\n", "DER length:", der_len);

    /* Recovery */
    int recid;
    uint8_t rsig[64], recovered[33];
    OK(ufsecp_ecdsa_sign_recoverable(ctx, msg, PRIVKEY, rsig, &recid), "sign_recoverable");
    OK(ufsecp_ecdsa_recover(ctx, msg, rsig, recid, recovered), "recover");
    printf("  %-20s recid=%d, match=%s\n", "Recovery:",
           recid, memcmp(recovered, pubkey33, 33) == 0 ? "YES" : "NO");
    printf("\n");

    /* -- 3. Schnorr / BIP-340 --------------------------------------------- */
    printf("[3] Schnorr Sign / Verify (BIP-340)\n");

    uint8_t aux[32] = {0};
    uint8_t schnorr_sig[64];
    OK(ufsecp_schnorr_sign(ctx, msg, PRIVKEY, aux, schnorr_sig), "schnorr_sign");
    hex_print("Schnorr signature:", schnorr_sig, 64);

    vrc = ufsecp_schnorr_verify(ctx, msg, schnorr_sig, xonly32);
    printf("  %-20s %s\n", "Verify:", vrc == UFSECP_OK ? "VALID" : "INVALID");
    printf("\n");

    /* -- 4. ECDH ---------------------------------------------------------- */
    printf("[4] ECDH Key Agreement\n");

    uint8_t pub2[33];
    OK(ufsecp_pubkey_create(ctx, PRIVKEY2, pub2), "pubkey2");

    uint8_t secret_a[32], secret_b[32];
    OK(ufsecp_ecdh(ctx, PRIVKEY, pub2, secret_a), "ecdh_a");
    OK(ufsecp_ecdh(ctx, PRIVKEY2, pubkey33, secret_b), "ecdh_b");

    hex_print("Secret (A->B):", secret_a, 32);
    hex_print("Secret (B->A):", secret_b, 32);
    printf("  %-20s %s\n", "Match:", memcmp(secret_a, secret_b, 32) == 0 ? "YES" : "NO");
    printf("\n");

    /* -- 5. Hashing ------------------------------------------------------- */
    printf("[5] Hashing\n");

    uint8_t sha[32], h160[20];
    OK(ufsecp_sha256(pubkey33, 33, sha), "sha256_pub");
    OK(ufsecp_hash160(pubkey33, 33, h160), "hash160_pub");
    hex_print("SHA-256(pubkey):", sha, 32);
    hex_print("Hash160(pubkey):", h160, 20);
    printf("\n");

    /* -- 6. Bitcoin Addresses --------------------------------------------- */
    printf("[6] Bitcoin Addresses\n");

    char addr[128];
    size_t addr_len;

    addr_len = sizeof(addr);
    OK(ufsecp_addr_p2pkh(ctx, pubkey33, UFSECP_NET_MAINNET, addr, &addr_len), "p2pkh");
    printf("  %-20s %s\n", "P2PKH:", addr);

    addr_len = sizeof(addr);
    OK(ufsecp_addr_p2wpkh(ctx, pubkey33, UFSECP_NET_MAINNET, addr, &addr_len), "p2wpkh");
    printf("  %-20s %s\n", "P2WPKH:", addr);

    addr_len = sizeof(addr);
    OK(ufsecp_addr_p2tr(ctx, xonly32, UFSECP_NET_MAINNET, addr, &addr_len), "p2tr");
    printf("  %-20s %s\n", "P2TR:", addr);
    printf("\n");

    /* -- 7. WIF ----------------------------------------------------------- */
    printf("[7] WIF Encoding\n");

    char wif[128];
    size_t wif_len = sizeof(wif);
    OK(ufsecp_wif_encode(ctx, PRIVKEY, 1, UFSECP_NET_MAINNET, wif, &wif_len), "wif_encode");
    printf("  %-20s %s\n", "WIF:", wif);

    uint8_t wif_key[32];
    int wif_comp, wif_net;
    OK(ufsecp_wif_decode(ctx, wif, wif_key, &wif_comp, &wif_net), "wif_decode");
    printf("  %-20s match=%s\n", "Decode roundtrip:",
           memcmp(wif_key, PRIVKEY, 32) == 0 ? "YES" : "NO");
    printf("\n");

    /* -- 8. BIP-32 HD Derivation ------------------------------------------ */
    printf("[8] BIP-32 HD Key Derivation\n");

    uint8_t seed[64];
    memset(seed, 0x42, sizeof(seed));
    ufsecp_bip32_key master, child;
    OK(ufsecp_bip32_master(ctx, seed, sizeof(seed), &master), "bip32_master");

    ufsecp_bip32_key path_key;
    OK(ufsecp_bip32_derive_path(ctx, &master, "m/44'/0'/0'/0/0", &path_key), "bip32_path");

    uint8_t child_priv[32], child_pub[33];
    OK(ufsecp_bip32_privkey(ctx, &path_key, child_priv), "bip32_privkey");
    OK(ufsecp_bip32_pubkey(ctx, &path_key, child_pub), "bip32_pubkey");
    hex_print("BIP-32 child priv:", child_priv, 32);
    hex_print("BIP-32 child pub:", child_pub, 33);
    printf("\n");

    /* -- 9. Taproot ------------------------------------------------------- */
    printf("[9] Taproot (BIP-341)\n");

    uint8_t output_x[32];
    int parity;
    OK(ufsecp_taproot_output_key(ctx, xonly32, NULL, output_x, &parity), "taproot_output_key");
    hex_print("Output key:", output_x, 32);
    printf("  %-20s %d\n", "Parity:", parity);

    vrc = ufsecp_taproot_verify(ctx, output_x, parity, xonly32, NULL, 0);
    printf("  %-20s %s\n", "Verify:", vrc == UFSECP_OK ? "VALID" : "INVALID");
    printf("\n");

    /* -- 10. Pedersen Commitments ----------------------------------------- */
    printf("[10] Pedersen Commitment\n");

    uint8_t value[32] = {0};
    value[31] = 42;
    uint8_t blinding[32] = {0};
    blinding[31] = 7;
    uint8_t commit[33];
    OK(ufsecp_pedersen_commit(ctx, value, blinding, commit), "pedersen_commit");
    hex_print("Commitment:", commit, 33);

    vrc = ufsecp_pedersen_verify(ctx, commit, value, blinding);
    printf("  %-20s %s\n", "Verify:", vrc == UFSECP_OK ? "VALID" : "INVALID");
    printf("\n");
}

/* ── GPU Examples ────────────────────────────────────────────────────────── */

static void demo_gpu(void)
{
    printf("=== GPU Operations ===\n\n");

    /* -- Backend discovery ------------------------------------------------ */
    printf("[11] GPU Backend Discovery\n");

    uint32_t backend_ids[4];
    uint32_t n_backends = ufsecp_gpu_backend_count(backend_ids, 4);
    printf("  %-20s %u\n", "Backends compiled:", n_backends);

    for (uint32_t i = 0; i < n_backends; ++i) {
        uint32_t bid = backend_ids[i];
        printf("  Backend %u: %-8s available=%d devices=%u\n",
               bid, ufsecp_gpu_backend_name(bid),
               ufsecp_gpu_is_available(bid),
               ufsecp_gpu_device_count(bid));
    }

    if (n_backends == 0) {
        printf("  No GPU backends compiled -- skipping GPU demos.\n\n");
        return;
    }

    /* Find first available backend */
    uint32_t use_backend = 0;
    for (uint32_t i = 0; i < n_backends; ++i) {
        if (ufsecp_gpu_is_available(backend_ids[i])) {
            use_backend = backend_ids[i];
            break;
        }
    }
    if (!use_backend) {
        printf("  No GPU backends available at runtime -- skipping.\n\n");
        return;
    }

    /* -- Device info ------------------------------------------------------ */
    printf("\n[12] GPU Device Info\n");

    ufsecp_gpu_device_info_t info;
    OK(ufsecp_gpu_device_info(use_backend, 0, &info), "gpu_device_info");
    printf("  Device:           %s\n", info.name);
    printf("  Memory:           %lu MB\n", (unsigned long)(info.global_mem_bytes >> 20));
    printf("  Compute units:    %u\n", info.compute_units);
    printf("  Max clock:        %u MHz\n", info.max_clock_mhz);
    printf("\n");

    /* -- Create GPU context ----------------------------------------------- */
    ufsecp_gpu_ctx* gpu;
    OK(ufsecp_gpu_ctx_create(&gpu, use_backend, 0), "gpu_ctx_create");

    /* -- 13. Batch Key Generation ----------------------------------------- */
    printf("[13] GPU Batch Key Generation (4 keys)\n");

    const size_t N = 4;
    uint8_t scalars[4 * 32];
    memset(scalars, 0, sizeof(scalars));
    for (size_t i = 0; i < N; ++i) scalars[i * 32 + 31] = (uint8_t)(i + 1);

    uint8_t pubkeys[4 * 33];
    ufsecp_error_t rc = ufsecp_gpu_generator_mul_batch(gpu, scalars, N, pubkeys);
    if (rc == UFSECP_OK) {
        for (size_t i = 0; i < N; ++i) {
            char label[32];
            snprintf(label, sizeof(label), "GPU pubkey[%zu]:", i);
            hex_print(label, pubkeys + i * 33, 33);
        }
    } else {
        printf("  gpu_generator_mul_batch: %s\n", ufsecp_gpu_error_str(rc));
    }
    printf("\n");

    /* -- 14. ECDSA Batch Verify ------------------------------------------- */
    printf("[14] GPU ECDSA Batch Verify\n");

    /* Sign 4 messages on CPU, then batch-verify on GPU */
    ufsecp_ctx* cpu;
    OK(ufsecp_ctx_create(&cpu), "cpu_ctx");

    uint8_t msgs[4 * 32], sigs[4 * 64], pubs[4 * 33];
    for (size_t i = 0; i < N; ++i) {
        /* Message = SHA256(i) */
        uint8_t ibuf = (uint8_t)i;
        OK(ufsecp_sha256(&ibuf, 1, msgs + i * 32), "sha256_msg");
        /* Sign with key i+1 */
        OK(ufsecp_ecdsa_sign(cpu, msgs + i * 32, scalars + i * 32, sigs + i * 64), "sign");
        /* Pubkey */
        memcpy(pubs + i * 33, pubkeys + i * 33, 33);
    }

    uint8_t results[4];
    rc = ufsecp_gpu_ecdsa_verify_batch(gpu, msgs, pubs, sigs, N, results);
    if (rc == UFSECP_OK) {
        printf("  Results: ");
        for (size_t i = 0; i < N; ++i) printf("[%zu]=%s ", i, results[i] ? "VALID" : "INVALID");
        printf("\n");
    } else {
        printf("  gpu_ecdsa_verify_batch: %s\n", ufsecp_gpu_error_str(rc));
    }
    printf("\n");

    /* -- 15. Hash160 Batch ------------------------------------------------ */
    printf("[15] GPU Hash160 Batch\n");

    uint8_t hashes[4 * 20];
    rc = ufsecp_gpu_hash160_pubkey_batch(gpu, pubkeys, N, hashes);
    if (rc == UFSECP_OK) {
        for (size_t i = 0; i < N; ++i) {
            char label[32];
            snprintf(label, sizeof(label), "Hash160[%zu]:", i);
            hex_print(label, hashes + i * 20, 20);
        }
    } else {
        printf("  gpu_hash160_pubkey_batch: %s\n", ufsecp_gpu_error_str(rc));
    }
    printf("\n");

    /* -- 16. MSM (Multi-scalar multiplication) ---------------------------- */
    printf("[16] GPU Multi-Scalar Multiplication\n");

    uint8_t msm_result[33];
    rc = ufsecp_gpu_msm(gpu, scalars, pubkeys, N, msm_result);
    if (rc == UFSECP_OK) {
        hex_print("MSM result:", msm_result, 33);
    } else {
        printf("  gpu_msm: %s\n", ufsecp_gpu_error_str(rc));
    }
    printf("\n");

    /* -- Cleanup ---------------------------------------------------------- */
    ufsecp_gpu_ctx_destroy(gpu);
    ufsecp_ctx_destroy(cpu);
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(void)
{
    printf("UltrafastSecp256k1 -- C Example\n");
    printf("ABI version: %u\n", ufsecp_abi_version());
    printf("Library:     %s\n\n", ufsecp_version_string());

    /* Create CPU context */
    ufsecp_ctx* ctx;
    OK(ufsecp_ctx_create(&ctx), "ctx_create");

    demo_cpu(ctx);
    ufsecp_ctx_destroy(ctx);

    demo_gpu();

    printf("All examples completed successfully.\n");
    return 0;
}
