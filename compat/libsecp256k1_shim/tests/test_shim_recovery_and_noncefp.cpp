// =============================================================================
// test_shim_recovery_and_noncefp.cpp
// Regression: recoverable signature parse compat + noncefp callback (PASS3-001/002)
// =============================================================================
//
// PASS3-002: secp256k1_ecdsa_recoverable_signature_parse_compact must accept
//            r==0 or s==0 at parse time (reject at recover time), matching
//            upstream libsecp256k1 behavior. Previously used
//            parse_bytes_strict_nonzero (rejected at parse) — divergence fixed.
//            Tests REC-1..REC-4.
//
// PASS3-001: secp256k1_ecdsa_sign, secp256k1_ecdsa_sign_recoverable, and
//            secp256k1_schnorrsig_sign_custom must fire the illegal callback
//            before returning 0 when a custom noncefp is supplied. Previously
//            returned 0 silently — divergence fixed.
//            Tests NFP-1..NFP-3.
//
// =============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>

#include "../include/secp256k1.h"
#include "../include/secp256k1_recovery.h"
#include "../include/secp256k1_schnorrsig.h"

// ---------------------------------------------------------------------------
// Test framework
// ---------------------------------------------------------------------------
static int g_pass = 0, g_fail = 0, g_illegal_called = 0;

static void counting_cb(const char* /*msg*/, void* /*data*/) { ++g_illegal_called; }

#define CHECK_EQ(a, b, msg) \
    do { \
        if ((a) == (b)) { ++g_pass; printf("  PASS: %s\n", (msg)); } \
        else { ++g_fail; printf("  FAIL: %s  (got %d, want %d)\n", (msg), (int)(a), (int)(b)); } \
    } while (0)

#define CHECK_NE(a, b, msg) \
    do { \
        if ((a) != (b)) { ++g_pass; printf("  PASS: %s\n", (msg)); } \
        else { ++g_fail; printf("  FAIL: %s  (unexpectedly equal: %d)\n", (msg), (int)(a)); } \
    } while (0)

static secp256k1_context* make_ctx() {
    secp256k1_context* ctx = secp256k1_context_create(
        SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    secp256k1_context_set_illegal_callback(ctx, counting_cb, nullptr);
    return ctx;
}

// ---------------------------------------------------------------------------
// REC-1: parse_compact(r=0, valid_s, recid=0) must return 1 (accept at parse)
// ---------------------------------------------------------------------------
static void test_rec1_zero_r_accepted_at_parse() {
    printf("  [REC-1] parse_compact(r=0, valid_s) → 1\n");
    secp256k1_context* ctx = make_ctx();

    uint8_t compact[64] = {};
    // r = 0x00..00  (bytes 0..31)
    // s = 0x00..01  (bytes 32..63) — small non-zero s
    compact[63] = 0x01;

    secp256k1_ecdsa_recoverable_signature sig;
    int rc = secp256k1_ecdsa_recoverable_signature_parse_compact(ctx, &sig, compact, 0);
    CHECK_EQ(rc, 1, "REC-1: parse_compact(r=0) returns 1");

    secp256k1_context_destroy(ctx);
}

// ---------------------------------------------------------------------------
// REC-2: parse_compact(valid_r, s=0, recid=0) must return 1 (accept at parse)
// ---------------------------------------------------------------------------
static void test_rec2_zero_s_accepted_at_parse() {
    printf("  [REC-2] parse_compact(valid_r, s=0) → 1\n");
    secp256k1_context* ctx = make_ctx();

    uint8_t compact[64] = {};
    // r = 0x00..01 (bytes 0..31)
    // s = 0x00..00 (bytes 32..63)
    compact[31] = 0x01;

    secp256k1_ecdsa_recoverable_signature sig;
    int rc = secp256k1_ecdsa_recoverable_signature_parse_compact(ctx, &sig, compact, 0);
    CHECK_EQ(rc, 1, "REC-2: parse_compact(s=0) returns 1");

    secp256k1_context_destroy(ctx);
}

// ---------------------------------------------------------------------------
// REC-3: recover on sig with r=0 must return 0 (reject at recover)
// ---------------------------------------------------------------------------
static void test_rec3_zero_r_rejected_at_recover() {
    printf("  [REC-3] recover(r=0) → 0 (rejected at recover)\n");
    secp256k1_context* ctx = make_ctx();

    uint8_t compact[64] = {};
    compact[63] = 0x01;  // r=0, s=1

    secp256k1_ecdsa_recoverable_signature sig;
    (void)secp256k1_ecdsa_recoverable_signature_parse_compact(ctx, &sig, compact, 0);

    uint8_t msg32[32] = {};
    msg32[0] = 0xAB;
    secp256k1_pubkey pubkey;
    int rc = secp256k1_ecdsa_recover(ctx, &pubkey, &sig, msg32);
    CHECK_EQ(rc, 0, "REC-3: recover(r=0) returns 0");

    secp256k1_context_destroy(ctx);
}

// ---------------------------------------------------------------------------
// REC-4: recover on sig with s=0 must return 0 (reject at recover)
// ---------------------------------------------------------------------------
static void test_rec4_zero_s_rejected_at_recover() {
    printf("  [REC-4] recover(s=0) → 0 (rejected at recover)\n");
    secp256k1_context* ctx = make_ctx();

    uint8_t compact[64] = {};
    compact[31] = 0x01;  // r=1, s=0

    secp256k1_ecdsa_recoverable_signature sig;
    (void)secp256k1_ecdsa_recoverable_signature_parse_compact(ctx, &sig, compact, 0);

    uint8_t msg32[32] = {};
    msg32[0] = 0xCD;
    secp256k1_pubkey pubkey;
    int rc = secp256k1_ecdsa_recover(ctx, &pubkey, &sig, msg32);
    CHECK_EQ(rc, 0, "REC-4: recover(s=0) returns 0");

    secp256k1_context_destroy(ctx);
}

// ---------------------------------------------------------------------------
// NFP-1: secp256k1_ecdsa_sign with custom noncefp fires illegal callback
// ---------------------------------------------------------------------------
static int dummy_noncefp(unsigned char* /*nonce32*/, const unsigned char* /*msg32*/,
                         const unsigned char* /*key32*/, const unsigned char* /*algo16*/,
                         void* /*data*/, unsigned int /*attempt*/) { return 1; }

static void test_nfp1_ecdsa_sign_custom_nonce_fires_cb() {
    printf("  [NFP-1] secp256k1_ecdsa_sign with custom noncefp → callback + return 0\n");
    secp256k1_context* ctx = make_ctx();

    uint8_t seckey[32] = {}; seckey[31] = 1;
    uint8_t msg32[32]  = {}; msg32[0]   = 0x01;
    secp256k1_ecdsa_signature sig;

    int before = g_illegal_called;
    int rc = secp256k1_ecdsa_sign(ctx, &sig, msg32, seckey, dummy_noncefp, nullptr);
    int after = g_illegal_called;

    CHECK_EQ(rc, 0,           "NFP-1: sign with custom noncefp returns 0");
    CHECK_NE(after, before,   "NFP-1: sign with custom noncefp fires illegal callback");

    secp256k1_context_destroy(ctx);
}

// ---------------------------------------------------------------------------
// NFP-2: secp256k1_ecdsa_sign_recoverable with custom noncefp fires callback
// ---------------------------------------------------------------------------
static void test_nfp2_sign_recoverable_custom_nonce_fires_cb() {
    printf("  [NFP-2] secp256k1_ecdsa_sign_recoverable with custom noncefp → callback + 0\n");
    secp256k1_context* ctx = make_ctx();

    uint8_t seckey[32] = {}; seckey[31] = 2;
    uint8_t msg32[32]  = {}; msg32[0]   = 0x02;
    secp256k1_ecdsa_recoverable_signature sig;

    int before = g_illegal_called;
    int rc = secp256k1_ecdsa_sign_recoverable(ctx, &sig, msg32, seckey, dummy_noncefp, nullptr);
    int after = g_illegal_called;

    CHECK_EQ(rc, 0,          "NFP-2: sign_recoverable with custom noncefp returns 0");
    CHECK_NE(after, before,  "NFP-2: sign_recoverable with custom noncefp fires callback");

    secp256k1_context_destroy(ctx);
}

// ---------------------------------------------------------------------------
// NFP-3: secp256k1_schnorrsig_sign_custom with unknown noncefp fires callback
// ---------------------------------------------------------------------------
static int dummy_schnorr_noncefp(unsigned char* /*nonce32*/, const unsigned char* /*msg*/,
                                  size_t /*msglen*/, const unsigned char* /*key32*/,
                                  const unsigned char* /*xonly_pk32*/,
                                  const unsigned char* /*algo*/, size_t /*algolen*/,
                                  void* /*data*/) { return 1; }

static void test_nfp3_schnorr_sign_custom_nonce_fires_cb() {
    printf("  [NFP-3] secp256k1_schnorrsig_sign_custom with custom noncefp → callback + 0\n");
    secp256k1_context* ctx = make_ctx();

    uint8_t seckey[32] = {}; seckey[31] = 3;
    secp256k1_keypair keypair;
    // P7-TEST-003: a setup failure here is a HARD failure, not a silent pass.
    // seckey=3 is a valid key, so keypair_create MUST succeed; if it ever fails,
    // the NFP-3 property (custom noncefp fires the callback) is never verified and
    // the test must FAIL rather than count a vacuous pass.
    if (!secp256k1_keypair_create(ctx, &keypair, seckey)) {
        printf("  [NFP-3] FAIL: keypair_create failed for a valid key (setup)\n");
        ++g_fail;
        secp256k1_context_destroy(ctx);
        return;
    }

    uint8_t msg32[32] = {};
    uint8_t sig64[64] = {};
    secp256k1_schnorrsig_extraparams ep = SECP256K1_SCHNORRSIG_EXTRAPARAMS_INIT;
    ep.noncefp = (secp256k1_nonce_function_hardened)dummy_schnorr_noncefp;

    int before = g_illegal_called;
    int rc = secp256k1_schnorrsig_sign_custom(ctx, sig64, msg32, 32, &keypair, &ep);
    int after = g_illegal_called;

    CHECK_EQ(rc, 0,          "NFP-3: schnorrsig_sign_custom with custom noncefp returns 0");
    CHECK_NE(after, before,  "NFP-3: schnorrsig_sign_custom with custom noncefp fires callback");

    secp256k1_context_destroy(ctx);
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
int test_shim_recovery_and_noncefp_run() {
    g_pass = 0; g_fail = 0; g_illegal_called = 0;
    printf("[test_shim_recovery_and_noncefp] PASS3-001/002: recovery parse compat + noncefp callback\n\n");

    test_rec1_zero_r_accepted_at_parse();
    test_rec2_zero_s_accepted_at_parse();
    test_rec3_zero_r_rejected_at_recover();
    test_rec4_zero_s_rejected_at_recover();
    test_nfp1_ecdsa_sign_custom_nonce_fires_cb();
    test_nfp2_sign_recoverable_custom_nonce_fires_cb();
    test_nfp3_schnorr_sign_custom_nonce_fires_cb();

    printf("\n  pass=%d  fail=%d\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() { return test_shim_recovery_and_noncefp_run(); }
#endif
