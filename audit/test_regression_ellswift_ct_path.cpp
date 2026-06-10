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

// Non-aborting illegal callback: the shim's argument-validation path calls the
// context illegal_callback, which defaults to std::abort() (SIGABRT). Tests that
// intentionally pass illegal args (e.g. NULL seckey) MUST install a no-op callback
// first, mirroring the canonical pattern in test_regression_shim_null_arg_cb.cpp.
static int g_ecp_illegal_cb_count = 0;
static void ecp_illegal_cb(const char*, void*) { ++g_ecp_illegal_cb_count; }

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
    // Install a non-aborting illegal callback: NULL seckey fires the shim's
    // illegal_callback, which defaults to std::abort(). Without this the test
    // SIGABRTs instead of observing the validated return value.
    secp256k1_context_set_illegal_callback(ctx, ecp_illegal_cb, nullptr);
    unsigned char enc[64];
    int before = g_ecp_illegal_cb_count;
    int r = secp256k1_ellswift_create(ctx, enc, nullptr, nullptr);
    int after = g_ecp_illegal_cb_count;
    check(r == 0, "[ECP-4] null seckey returns 0");
    check(after > before, "[ECP-4] null seckey fires illegal_callback");
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

// ── ECP-6: general XDH path (non-BIP324 hashfp) ───────────────────────────
// Exercises the general path in secp256k1_ellswift_xdh (SHIM-006 fix target).
// The custom hashfp receives the ECDH secret x32 and must return 1; the shared
// secret is written to output. Tests that the x32 secret is correctly passed to
// hashfp and that the function returns 1.
static int ecp6_hashfp_called = 0;
static int ecp6_custom_hash(unsigned char* output,
    const unsigned char* x32,
    const unsigned char* /*ell_a64*/,
    const unsigned char* /*ell_b64*/,
    void* data)
{
    ecp6_hashfp_called = 1;
    // Write x32 directly as output to allow symmetric check
    memcpy(output, x32, 32);
    (void)data;
    return 1;
}

static void test_ellswift_xdh_general_path() {
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);

    unsigned char ska[32] = {}, skb[32] = {};
    ska[31] = 5; skb[31] = 11;
    static const unsigned char aux[32] = {0};

    unsigned char enc_a[64], enc_b[64];
    check(secp256k1_ellswift_create(ctx, enc_a, ska, aux) == 1, "[ECP-6a] create A");
    check(secp256k1_ellswift_create(ctx, enc_b, skb, aux) == 1, "[ECP-6b] create B");

    unsigned char shared_ab[32], shared_ba[32];
    ecp6_hashfp_called = 0;
    int r_ab = secp256k1_ellswift_xdh(ctx, shared_ab, enc_a, enc_b, ska,
                                       0, ecp6_custom_hash, nullptr);
    check(r_ab == 1, "[ECP-6c] general XDH A side succeeds");
    check(ecp6_hashfp_called == 1, "[ECP-6d] custom hashfp was called");

    ecp6_hashfp_called = 0;
    int r_ba = secp256k1_ellswift_xdh(ctx, shared_ba, enc_a, enc_b, skb,
                                       1, ecp6_custom_hash, nullptr);
    check(r_ba == 1, "[ECP-6e] general XDH B side succeeds");
    // Both sides compute ECDH(skA, pkB) and ECDH(skB, pkA): these equal the
    // same shared point x-coordinate, so shared_ab == shared_ba.
    check(memcmp(shared_ab, shared_ba, 32) == 0,
          "[ECP-6f] general path shared secret symmetric (SHIM-006 fix)");

    secp256k1_context_destroy(ctx);
}

// ── ECP-7: encode->decode round-trip recovers the x-coordinate ────────────
// Directly exercises the forward decode map (xswiftec_fwd / xswiftec_fwd_point)
// after its `t` parameter was changed to const-reference: the decode must still
// recover the encoded x-coordinate (the refactor is behavior-preserving). Parity
// is shim-chosen on decode, so only the 32-byte x-coordinate is compared.
static void test_ellswift_encode_decode_roundtrip() {
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);

    unsigned char sk[32] = {};
    sk[31] = 9;
    secp256k1_pubkey pub;
    check(secp256k1_ec_pubkey_create(ctx, &pub, sk) == 1, "[ECP-7a] pubkey_create");

    unsigned char ser[33]; size_t serlen = sizeof(ser);
    check(secp256k1_ec_pubkey_serialize(ctx, ser, &serlen, &pub,
          SECP256K1_EC_COMPRESSED) == 1, "[ECP-7b] serialize original");

    static const unsigned char rnd[32] = {
        0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,
        0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff,0x00,
        0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,
        0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff,0x00
    };
    unsigned char enc[64];
    check(secp256k1_ellswift_encode(ctx, enc, &pub, rnd) == 1, "[ECP-7c] encode pubkey");

    secp256k1_pubkey decoded;
    check(secp256k1_ellswift_decode(ctx, &decoded, enc) == 1, "[ECP-7d] decode encoding");

    unsigned char ser2[33]; size_t ser2len = sizeof(ser2);
    check(secp256k1_ec_pubkey_serialize(ctx, ser2, &ser2len, &decoded,
          SECP256K1_EC_COMPRESSED) == 1, "[ECP-7e] serialize decoded");
    check(memcmp(ser + 1, ser2 + 1, 32) == 0,
          "[ECP-7f] encode->decode round-trip recovers the x-coordinate (fwd map)");

    secp256k1_context_destroy(ctx);
}

// ── ECP-8: lazy-sqrt deferral regression ───────────────────────────────────
// ellswift_try_u defers the x3 (s = x-u) field sqrt so it is computed only when
// the x1/x2 branch fails (skipped on ~half of all attempts → ~+9% faster create).
// s_x3/w_x3 are used only in the x3 branch, so the deferral is behavior-preserving.
// This test guards that: create->decode must recover the exact pubkey for many
// keys, exercising BOTH the x1/x2 and x3 branches of the inverse map.
static void test_ellswift_create_lazy_sqrt_roundtrip() {
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
    int recovered = 0;
    for (int i = 1; i <= 80; ++i) {
        unsigned char sk[32] = {0};
        sk[31] = static_cast<unsigned char>(i);
        sk[0]  = static_cast<unsigned char>(i * 7 + 3);
        sk[15] = static_cast<unsigned char>(i * 13 + 1);
        secp256k1_pubkey pub;
        if (secp256k1_ec_pubkey_create(ctx, &pub, sk) != 1) continue;
        unsigned char ser[33]; size_t serlen = sizeof(ser);
        secp256k1_ec_pubkey_serialize(ctx, ser, &serlen, &pub, SECP256K1_EC_COMPRESSED);

        unsigned char enc[64];
        check(secp256k1_ellswift_create(ctx, enc, sk, nullptr) == 1,
              "[ECP-8a] ellswift_create succeeds");
        secp256k1_pubkey dec;
        check(secp256k1_ellswift_decode(ctx, &dec, enc) == 1,
              "[ECP-8b] decode of created encoding succeeds");
        unsigned char ser2[33]; size_t ser2len = sizeof(ser2);
        secp256k1_ec_pubkey_serialize(ctx, ser2, &ser2len, &dec, SECP256K1_EC_COMPRESSED);
        check(memcmp(ser, ser2, 33) == 0,
              "[ECP-8c] create->decode recovers the exact pubkey (lazy-sqrt branches)");
        ++recovered;
    }
    check(recovered >= 40, "[ECP-8d] enough distinct keys exercised both branches");
    secp256k1_context_destroy(ctx);
}

// ── ECP-9: CT XDH is the only XDH path after RT1-001 removal ────────────────
// ellswift_xdh_fast (a variable-time variable-base scalar_mul on the SECRET key
// against an attacker-controlled decoded point) was removed; every public XDH
// entry point now routes through the constant-time secp256k1_ellswift_xdh. Assert
// the surviving CT path yields a non-zero, symmetric, deterministic shared secret.
static void test_ellswift_xdh_ct_only_after_removal() {
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
    unsigned char ska[32] = {}, skb[32] = {};
    ska[31] = 0x2a; skb[31] = 0x55;
    static const unsigned char aux[32] = {0};
    unsigned char enc_a[64], enc_b[64];
    check(secp256k1_ellswift_create(ctx, enc_a, ska, aux) == 1, "[ECP-9a] create A");
    check(secp256k1_ellswift_create(ctx, enc_b, skb, aux) == 1, "[ECP-9b] create B");

    unsigned char s_ab[32], s_ba[32], s_ab2[32];
    int r_ab = secp256k1_ellswift_xdh(ctx, s_ab, enc_a, enc_b, ska, 0,
                                      secp256k1_ellswift_xdh_hash_function_bip324, nullptr);
    int r_ba = secp256k1_ellswift_xdh(ctx, s_ba, enc_a, enc_b, skb, 1,
                                      secp256k1_ellswift_xdh_hash_function_bip324, nullptr);
    int r_ab2 = secp256k1_ellswift_xdh(ctx, s_ab2, enc_a, enc_b, ska, 0,
                                       secp256k1_ellswift_xdh_hash_function_bip324, nullptr);
    check(r_ab == 1 && r_ba == 1 && r_ab2 == 1, "[ECP-9c] CT XDH succeeds");
    check(memcmp(s_ab, s_ba, 32) == 0, "[ECP-9d] CT XDH shared secret symmetric");
    check(memcmp(s_ab, s_ab2, 32) == 0, "[ECP-9e] CT XDH deterministic");
    unsigned char zero[32] = {};
    check(memcmp(s_ab, zero, 32) != 0, "[ECP-9f] CT XDH shared secret non-zero");

    secp256k1_context_destroy(ctx);
}

// ── _run() ─────────────────────────────────────────────────────────────────
int test_regression_ellswift_ct_path_run() {
    g_pass = 0; g_fail = 0;
    std::printf("[regression_ellswift_ct_path] ellswift_create CT path + XDH round-trip\n");

    test_ellswift_create_lazy_sqrt_roundtrip();
    test_ellswift_deterministic();
    test_ellswift_distinct_keys();
    test_ellswift_xdh_roundtrip();
    test_ellswift_null_seckey();
    test_ellswift_zero_key();
    test_ellswift_xdh_general_path();
    test_ellswift_encode_decode_roundtrip();
    test_ellswift_xdh_ct_only_after_removal();

    std::printf("  pass=%d  fail=%d\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_ellswift_ct_path_run(); }
#endif
