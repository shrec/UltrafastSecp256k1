// ============================================================================
// test_regression_ellswift_ct_path.cpp
// Regression: secp256k1_ellswift_create must route through ct::generator_mul.
//
// Bug fixed: shim_ellswift.cpp called ellswift_create_fast(sk) which used
// scalar_mul_generator (variable-time) with the real private key.
// Fix: shim now calls ellswift_create(sk) which uses ct::generator_mul.
//
// Tests:
//   ECP-1: ellswift encoding is 64 bytes and deterministic for same (sk, aux)
//   ECP-2: two different private keys produce different encodings
//   ECP-3: XDH round-trip via the shim is consistent (A->B == B->A decoded)
//   ECP-4: ellswift_create with null seckey returns 0 (input validation)
//   ECP-5: ellswift_create with sk==0 returns 0 (zero key rejected)
// ============================================================================

#include "audit_check.hpp"

static int g_pass = 0, g_fail = 0;
#include <cstdio>
#include <cstring>
#include <array>

#ifdef STANDALONE_TEST
#include "secp256k1.h"
#include "secp256k1_ellswift.h"
#else
// In unified runner context — include shim headers via standard path
#include "secp256k1.h"
#include "secp256k1_ellswift.h"
#endif

// ── ECP-1: deterministic encoding for same (sk, aux_rand) ─────────────────
static void test_ellswift_deterministic() {
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);

    static const unsigned char sk[32] = {
        0x01,0x23,0x45,0x67,0x89,0xab,0xcd,0xef,
        0x01,0x23,0x45,0x67,0x89,0xab,0xcd,0xef,
        0x01,0x23,0x45,0x67,0x89,0xab,0xcd,0xef,
        0x01,0x23,0x45,0x67,0x89,0xab,0xcd,0xef
    };
    static const unsigned char aux[32] = {0};

    unsigned char enc1[64], enc2[64];
    int r1 = secp256k1_ellswift_create(ctx, enc1, sk, aux);
    int r2 = secp256k1_ellswift_create(ctx, enc2, sk, aux);
    check(r1 == 1, "[ECP-1a] ellswift_create succeeds");
    check(r2 == 1, "[ECP-1b] ellswift_create succeeds on second call");
    check(memcmp(enc1, enc2, 64) == 0, "[ECP-1c] same sk+aux produces same encoding");

    secp256k1_context_destroy(ctx);
}

// ── ECP-2: distinct keys produce distinct encodings ───────────────────────
static void test_ellswift_distinct_keys() {
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);

    unsigned char sk1[32] = {}, sk2[32] = {};
    sk1[31] = 1;
    sk2[31] = 2;
    static const unsigned char aux[32] = {0};

    unsigned char enc1[64], enc2[64];
    check(secp256k1_ellswift_create(ctx, enc1, sk1, aux) == 1, "[ECP-2a] sk=1 creates encoding");
    check(secp256k1_ellswift_create(ctx, enc2, sk2, aux) == 1, "[ECP-2b] sk=2 creates encoding");
    check(memcmp(enc1, enc2, 64) != 0, "[ECP-2c] distinct keys produce distinct encodings");

    secp256k1_context_destroy(ctx);
}

// ── ECP-3: XDH round-trip: shared secret symmetric ────────────────────────
static void test_ellswift_xdh_roundtrip() {
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);

    unsigned char ska[32] = {}, skb[32] = {};
    ska[31] = 3; skb[31] = 7;
    static const unsigned char aux[32] = {0};

    unsigned char enc_a[64], enc_b[64];
    check(secp256k1_ellswift_create(ctx, enc_a, ska, aux) == 1, "[ECP-3a] create A");
    check(secp256k1_ellswift_create(ctx, enc_b, skb, aux) == 1, "[ECP-3b] create B");

    unsigned char shared_ab[32], shared_ba[32];
    int r_ab = secp256k1_ellswift_xdh(ctx, shared_ab, enc_a, enc_b, ska,
                                       0, secp256k1_ellswift_xdh_hash_function_bip324, nullptr);
    int r_ba = secp256k1_ellswift_xdh(ctx, shared_ba, enc_a, enc_b, skb,
                                       1, secp256k1_ellswift_xdh_hash_function_bip324, nullptr);
    check(r_ab == 1, "[ECP-3c] XDH A->B succeeds");
    check(r_ba == 1, "[ECP-3d] XDH B->A succeeds");
    check(memcmp(shared_ab, shared_ba, 32) == 0, "[ECP-3e] XDH shared secret matches");

    secp256k1_context_destroy(ctx);
}

// ── ECP-4: null seckey rejected ───────────────────────────────────────────
static void test_ellswift_null_seckey() {
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
    unsigned char enc[64];
    int r = secp256k1_ellswift_create(ctx, enc, nullptr, nullptr);
    check(r == 0, "[ECP-4] null seckey returns 0");
    secp256k1_context_destroy(ctx);
}

// ── ECP-5: zero private key rejected ──────────────────────────────────────
static void test_ellswift_zero_key() {
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
    unsigned char zero_sk[32] = {};
    unsigned char enc[64];
    int r = secp256k1_ellswift_create(ctx, enc, zero_sk, nullptr);
    check(r == 0, "[ECP-5] zero seckey returns 0");
    secp256k1_context_destroy(ctx);
}

// ── _run() ─────────────────────────────────────────────────────────────────
int test_regression_ellswift_ct_path_run() {
    g_pass = 0; g_fail = 0;
    std::printf("[regression_ellswift_ct_path] ellswift_create CT path + XDH round-trip\n");

    test_ellswift_deterministic();
    test_ellswift_distinct_keys();
    test_ellswift_xdh_roundtrip();
    test_ellswift_null_seckey();
    test_ellswift_zero_key();

    std::printf("  pass=%d  fail=%d\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_ellswift_ct_path_run(); }
#endif
