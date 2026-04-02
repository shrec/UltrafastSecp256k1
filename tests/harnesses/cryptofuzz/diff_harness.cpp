// Cryptofuzz-style differential harness: UltrafastSecp256k1 vs libsecp256k1
//
// Usage:
//   clang++ -std=c++20 -O2 \
//     -I<repo-root> \
//     -I<libsecp256k1-install>/include \
//     diff_harness.cpp \
//     -L<ufsecp-lib-dir> -lufsecp \
//     -L<libsecp256k1-install>/lib -lsecp256k1 \
//     -Wl,-rpath,<ufsecp-lib-dir> \
//     -Wl,-rpath,<libsecp256k1-install>/lib \
//     -o diff_harness
//
// libsecp256k1 headers are resolved via -I (no absolute paths here).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <random>
#include <functional>

// libsecp256k1 reference (header path supplied via -I at compile time)
extern "C" {
#include "secp256k1.h"
#include "secp256k1_schnorrsig.h"
#include "secp256k1_recovery.h"
#include "secp256k1_ecdh.h"
}

// UltrafastSecp256k1 ABI
#include "include/ufsecp/ufsecp.h"

static int g_pass = 0, g_fail = 0;

// Check that privkey is non-zero and < n (secp256k1 order)
static bool is_valid_privkey(const uint8_t k[32]) {
    static const uint8_t N[32] = {
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
        0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B,
        0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x41
    };
    bool all_zero = true;
    for (int i = 0; i < 32; ++i) if (k[i]) { all_zero = false; break; }
    if (all_zero) return false;
    for (int i = 0; i < 32; ++i) {
        if (k[i] < N[i]) return true;
        if (k[i] > N[i]) return false;
    }
    return false; // == N, invalid
}

#define CHECK_DIFF(cond, name) do { \
    if (!(cond)) { \
        fprintf(stderr, "DIVERGENCE[%s]: %s\n", name, #cond); \
        g_fail++; \
    } else { g_pass++; } \
} while(0)

// Test: pubkey_create output agrees between libraries
static void test_pubkey_create(ufsecp_ctx* ctx, secp256k1_context* ref_ctx,
                                const uint8_t privkey[32]) {
    if (!is_valid_privkey(privkey)) return;

    // Our library
    uint8_t our_pub[33] = {};
    int our_rc = ufsecp_pubkey_create(ctx, privkey, our_pub);
    if (our_rc != UFSECP_OK) return;

    // Reference
    secp256k1_pubkey ref_pub;
    int ref_rc = secp256k1_ec_pubkey_create(ref_ctx, &ref_pub, privkey);
    if (!ref_rc) return;

    uint8_t ref_ser[33] = {};
    size_t ref_len = 33;
    secp256k1_ec_pubkey_serialize(ref_ctx, ref_ser, &ref_len,
                                  &ref_pub, SECP256K1_EC_COMPRESSED);

    CHECK_DIFF(memcmp(our_pub, ref_ser, 33) == 0, "pubkey_create");
}

// Test: ECDSA sign then verify (cross-library)
static void test_ecdsa_cross(ufsecp_ctx* ctx, secp256k1_context* ref_ctx,
                              const uint8_t privkey[32],
                              const uint8_t msg32[32]) {
    if (!is_valid_privkey(privkey)) return;

    uint8_t our_pub[33] = {};
    if (ufsecp_pubkey_create(ctx, privkey, our_pub) != UFSECP_OK) return;

    // Sign with our library → compact 64-byte sig
    uint8_t our_sig64[64] = {};
    if (ufsecp_ecdsa_sign(ctx, msg32, privkey, our_sig64) != UFSECP_OK)
        return;

    // Convert compact sig to DER for libsecp256k1 verify
    uint8_t our_der[72] = {};
    size_t our_der_len = 72;
    if (ufsecp_ecdsa_sig_to_der(ctx, our_sig64, our_der, &our_der_len) != UFSECP_OK)
        return;

    // Verify with reference libsecp256k1
    secp256k1_pubkey ref_pub;
    secp256k1_ec_pubkey_parse(ref_ctx, &ref_pub, our_pub, 33);

    secp256k1_ecdsa_signature ref_sig;
    int parsed = secp256k1_ecdsa_signature_parse_der(
        ref_ctx, &ref_sig, our_der, our_der_len);
    if (!parsed) {
        fprintf(stderr, "DIVERGENCE[ecdsa_cross]: our sig not parseable by libsecp256k1\n");
        g_fail++; return;
    }
    secp256k1_ecdsa_signature_normalize(ref_ctx, &ref_sig, &ref_sig);
    int ok = secp256k1_ecdsa_verify(ref_ctx, &ref_sig, msg32, &ref_pub);
    CHECK_DIFF(ok == 1, "ecdsa_cross_verify");

    // Also: sign with reference, verify with ours
    secp256k1_ecdsa_signature rsig;
    secp256k1_ecdsa_sign(ref_ctx, &rsig, msg32, privkey, NULL, NULL);
    // Normalize to low-s (libsecp256k1 may produce high-s)
    secp256k1_ecdsa_signature_normalize(ref_ctx, &rsig, &rsig);
    unsigned char ref_der[72] = {};
    size_t ref_der_len = 72;
    secp256k1_ecdsa_signature_serialize_der(ref_ctx, ref_der, &ref_der_len, &rsig);
    // Convert reference DER to compact for ufsecp_ecdsa_verify
    uint8_t ref_compact[64] = {};
    if (ufsecp_ecdsa_sig_from_der(ctx, ref_der, ref_der_len, ref_compact) != UFSECP_OK)
        return;
    int vrc = ufsecp_ecdsa_verify(ctx, msg32, ref_compact, our_pub);
    CHECK_DIFF(vrc == UFSECP_OK, "ecdsa_cross_verify_ref_sig");
}

// Test: Schnorr BIP-340 sign then verify
static void test_schnorr_cross(ufsecp_ctx* ctx, secp256k1_context* ref_ctx,
                                const uint8_t privkey[32],
                                const uint8_t msg32[32],
                                const uint8_t aux_rand[32]) {
    if (!is_valid_privkey(privkey)) return;

    // Sign with our library (arg order: ctx, msg32, privkey, aux_rand, sig64_out)
    uint8_t our_sig64[64] = {};
    if (ufsecp_schnorr_sign(ctx, msg32, privkey, aux_rand, our_sig64) != UFSECP_OK)
        return;

    // Get x-only pubkey
    uint8_t our_pub[33] = {};
    if (ufsecp_pubkey_create(ctx, privkey, our_pub) != UFSECP_OK) return;
    uint8_t xonly[32] = {};
    memcpy(xonly, our_pub + 1, 32); // compressed → strip prefix → x-only

    // Verify with reference
    secp256k1_xonly_pubkey xpk;
    if (!secp256k1_xonly_pubkey_parse(ref_ctx, &xpk, xonly)) return;
    int ok = secp256k1_schnorrsig_verify(ref_ctx, our_sig64, msg32, 32, &xpk);
    CHECK_DIFF(ok == 1, "schnorr_cross_verify");
}

// Test: ECDH output is symmetric (k1·P2 == k2·P1)
static void test_ecdh_symmetric(ufsecp_ctx* ctx,
                                 const uint8_t k1[32], const uint8_t k2[32]) {
    if (!is_valid_privkey(k1) || !is_valid_privkey(k2)) return;

    uint8_t pub1[33] = {}, pub2[33] = {};
    if (ufsecp_pubkey_create(ctx, k1, pub1) != UFSECP_OK) return;
    if (ufsecp_pubkey_create(ctx, k2, pub2) != UFSECP_OK) return;

    // ufsecp_ecdh(ctx, privkey, pubkey33, secret32_out) — 4 args
    uint8_t shared1[32] = {}, shared2[32] = {};
    ufsecp_ecdh(ctx, k1, pub2, shared1);
    ufsecp_ecdh(ctx, k2, pub1, shared2);
    CHECK_DIFF(memcmp(shared1, shared2, 32) == 0, "ecdh_symmetric");
}

int main(int argc, char** argv) {
    const int ITERATIONS = (argc > 1) ? atoi(argv[1]) : 1000;
    fprintf(stderr, "Cryptofuzz++ differential harness — %d iterations\n", ITERATIONS);

    // ufsecp_ctx_create takes ufsecp_ctx** and returns ufsecp_error_t
    ufsecp_ctx* ctx = nullptr;
    if (ufsecp_ctx_create(&ctx) != UFSECP_OK || !ctx) {
        fprintf(stderr, "ufsecp_ctx_create failed\n"); return 1;
    }

    secp256k1_context* ref_ctx = secp256k1_context_create(
        SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    if (!ref_ctx) { fprintf(stderr, "secp256k1_context_create failed\n"); return 1; }

    std::mt19937_64 rng(0xDEADBEEFCAFEBABEULL);
    auto rand_bytes = [&](uint8_t* buf, size_t len) {
        for (size_t i = 0; i < len; i += 8) {
            uint64_t v = rng();
            memcpy(buf + i, &v, std::min((size_t)8, len - i));
        }
    };

    uint8_t privkey[32], privkey2[32], msg[32], aux[32];
    for (int i = 0; i < ITERATIONS; ++i) {
        rand_bytes(privkey, 32);
        rand_bytes(privkey2, 32);
        rand_bytes(msg, 32);
        rand_bytes(aux, 32);
        test_pubkey_create(ctx, ref_ctx, privkey);
        test_ecdsa_cross(ctx, ref_ctx, privkey, msg);
        test_schnorr_cross(ctx, ref_ctx, privkey, msg, aux);
        test_ecdh_symmetric(ctx, privkey, privkey2);
    }

    fprintf(stderr, "\nResults: %d PASS / %d FAIL\n", g_pass, g_fail);
    secp256k1_context_destroy(ref_ctx);
    ufsecp_ctx_destroy(ctx);
    return (g_fail > 0) ? 1 : 0;
}
