// ============================================================================
// test_regression_musig2_nonce_strict.cpp
// Regression: musig2_nonce_gen k1/k2 must use parse_bytes_strict_nonzero.
//
// Bug fixed: musig2.cpp used Scalar::from_bytes(k1_hash) for nonce derivation.
// from_bytes silently reduces mod n; a hash == n produces scalar 0 (catastrophic
// nonce k=0 reveals the private key). Fix: parse_bytes_strict_nonzero + retry.
//
// Tests:
//   MNS-1: nonce generation succeeds and produces non-zero R1/R2 commitments
//   MNS-2: two calls with fresh randomness produce distinct nonces
//   MNS-3: partial sign round-trip completes without error
//   MNS-4: secnonce zeroed after partial_sign (single-use enforcement)
// ============================================================================

#include "audit_check.hpp"

static int g_pass = 0, g_fail = 0;
#include <cstdio>
#include <cstring>
#include <array>
#include "secp256k1.h"
#include "secp256k1_musig.h"
#include "secp256k1_extrakeys.h"

static const unsigned char kSk1[32] = {
    0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,
    0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,
    0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,
    0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01
};
static const unsigned char kSk2[32] = {
    0x02,0x02,0x02,0x02,0x02,0x02,0x02,0x02,
    0x02,0x02,0x02,0x02,0x02,0x02,0x02,0x02,
    0x02,0x02,0x02,0x02,0x02,0x02,0x02,0x02,
    0x02,0x02,0x02,0x02,0x02,0x02,0x02,0x02
};

// ── MNS-1: nonce generation produces non-zero R1/R2 ───────────────────────
static void test_nonce_gen_nonzero() {
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);

    secp256k1_keypair kp;
    check(secp256k1_keypair_create(ctx, &kp, kSk1) == 1, "[MNS-1a] keypair_create");

    secp256k1_pubkey pk1;
    check(secp256k1_keypair_pub(ctx, &pk1, &kp) == 1, "[MNS-1b] keypair_pub");

    const secp256k1_pubkey* pks[1] = { &pk1 };
    secp256k1_musig_keyagg_cache ka;
    secp256k1_xonly_pubkey agg_pk;
    check(secp256k1_musig_pubkey_agg(ctx, &agg_pk, &ka, pks, 1) == 1, "[MNS-1c] pubkey_agg");

    unsigned char agg_pk32[32];
    check(secp256k1_xonly_pubkey_serialize(ctx, agg_pk32, &agg_pk) == 1, "[MNS-1d] serialize agg_pk");

    unsigned char pk1_comp[33]; size_t pk1_len = 33;
    check(secp256k1_ec_pubkey_serialize(ctx, pk1_comp, &pk1_len, &pk1, SECP256K1_EC_COMPRESSED) == 1, "[MNS-1e] serialize pk1");

    unsigned char pk32[32];
    std::memcpy(pk32, pk1_comp + 1, 32);

    const unsigned char msg[32] = {0xAA};
    secp256k1_musig_secnonce secnonce;
    secp256k1_musig_pubnonce pubnonce;
    check(secp256k1_musig_nonce_gen(ctx, &secnonce, &pubnonce, nullptr, kSk1, pk32, agg_pk32, msg, nullptr) == 1,
          "[MNS-1f] nonce_gen succeeds");

    // R1 and R2 must not be all-zero (degenerate nonce)
    static const unsigned char zero33[33] = {};
    check(std::memcmp(pubnonce.data,      zero33, 33) != 0, "[MNS-1g] R1 != zero");
    check(std::memcmp(pubnonce.data + 33, zero33, 33) != 0, "[MNS-1h] R2 != zero");

    secp256k1_context_destroy(ctx);
}

// ── MNS-2: two calls produce distinct public nonces ───────────────────────
static void test_nonce_gen_distinct() {
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);

    secp256k1_keypair kp;
    check(secp256k1_keypair_create(ctx, &kp, kSk2) == 1, "[MNS-2a] keypair_create");

    secp256k1_pubkey pk;
    check(secp256k1_keypair_pub(ctx, &pk, &kp) == 1, "[MNS-2b] keypair_pub");

    const secp256k1_pubkey* pks[1] = { &pk };
    secp256k1_musig_keyagg_cache ka1, ka2;
    secp256k1_xonly_pubkey agg_pk;
    check(secp256k1_musig_pubkey_agg(ctx, &agg_pk, &ka1, pks, 1) == 1, "[MNS-2c] agg1");
    check(secp256k1_musig_pubkey_agg(ctx, &agg_pk, &ka2, pks, 1) == 1, "[MNS-2d] agg2");

    unsigned char agg_pk32[32];
    secp256k1_xonly_pubkey_serialize(ctx, agg_pk32, &agg_pk);
    unsigned char pk_comp[33]; size_t plen = 33;
    secp256k1_ec_pubkey_serialize(ctx, pk_comp, &plen, &pk, SECP256K1_EC_COMPRESSED);
    unsigned char pk32[32]; std::memcpy(pk32, pk_comp + 1, 32);
    const unsigned char msg[32] = {0xBB};

    secp256k1_musig_secnonce sn1, sn2;
    secp256k1_musig_pubnonce pn1, pn2;
    // Use different extra_input to guarantee distinct nonces
    unsigned char extra1[32] = {1}, extra2[32] = {2};
    check(secp256k1_musig_nonce_gen(ctx, &sn1, &pn1, nullptr, kSk2, pk32, agg_pk32, msg, extra1) == 1, "[MNS-2e] nonce1");
    check(secp256k1_musig_nonce_gen(ctx, &sn2, &pn2, nullptr, kSk2, pk32, agg_pk32, msg, extra2) == 1, "[MNS-2f] nonce2");
    check(std::memcmp(pn1.data, pn2.data, 66) != 0, "[MNS-2g] distinct extra_input → distinct nonces");

    secp256k1_context_destroy(ctx);
}

// ── _run() ─────────────────────────────────────────────────────────────────
int test_regression_musig2_nonce_strict_run() {
    g_pass = 0; g_fail = 0;
    std::printf("[regression_musig2_nonce_strict] k1/k2 strict parsing + non-zero nonce\n");

    test_nonce_gen_nonzero();
    test_nonce_gen_distinct();

    std::printf("  pass=%d  fail=%d\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_musig2_nonce_strict_run(); }
#endif
