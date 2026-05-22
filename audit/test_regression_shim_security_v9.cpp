// ============================================================================
// test_regression_shim_security_v9.cpp — Shim security regressions (v9)
// ============================================================================
// Covers findings from the 2026-05-22 multi-agent review:
//
//   SHIM-NEW-012  secp256k1_ecdsa_signature_serialize_compact and
//                 secp256k1_ecdsa_signature_serialize_der previously returned 0
//                 on NULL output/sig without firing the illegal callback.
//                 Upstream libsecp256k1 fires the callback. Fixed.
//
//   SHIM-NEW-015  secp256k1_ec_seckey_verify, _negate, _tweak_add, _tweak_mul
//                 previously returned 0 on NULL seckey without firing the
//                 illegal callback. Upstream libsecp256k1 fires the callback.
//                 Fixed.
//
// advisory=true: compiled into unified_audit_runner only when secp256k1_shim
// is linked. Otherwise shim_run_stubs_unified.cpp returns ADVISORY_SKIP_CODE.
// ============================================================================

#ifndef UNIFIED_AUDIT_RUNNER
#define STANDALONE_TEST
#endif

#include "secp256k1.h"
#include "secp256k1_extrakeys.h"

#include <cstdio>
#include <cstring>
#include <cstdint>

namespace {

static int g_pass = 0, g_fail = 0;
static int g_cb_fired = 0;

#define CHECK(cond, msg) do { \
    if (cond) { ++g_pass; } \
    else { ++g_fail; std::printf("  [FAIL] %s\n", (msg)); } \
} while(0)

static void test_illegal_callback(const char* /*message*/, void* /*data*/) {
    ++g_cb_fired;
}

// ── SHIM-NEW-012: serialize_compact NULL output/sig fires illegal callback ──

static void test_serialize_compact_null_fires_callback() {
    std::printf("  [SHIM-NEW-012a] serialize_compact: NULL output64 fires illegal callback\n");

    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    secp256k1_context_set_illegal_callback(ctx, test_illegal_callback, nullptr);

    // Build a valid signature to use as input
    unsigned char sk[32] = {};
    sk[31] = 1;
    unsigned char msg[32] = {};
    msg[0] = 0xAB;
    secp256k1_ecdsa_signature sig{};
    secp256k1_ecdsa_sign(ctx, &sig, msg, sk, nullptr, nullptr);

    // NULL output64 — must fire callback
    g_cb_fired = 0;
    int rc = secp256k1_ecdsa_signature_serialize_compact(ctx, nullptr, &sig);
    CHECK(rc == 0, "SHIM-NEW-012a: NULL output64 returns 0");
    CHECK(g_cb_fired > 0, "SHIM-NEW-012a: NULL output64 fires illegal callback");

    // NULL sig — must fire callback
    unsigned char out64[64]{};
    g_cb_fired = 0;
    rc = secp256k1_ecdsa_signature_serialize_compact(ctx, out64, nullptr);
    CHECK(rc == 0, "SHIM-NEW-012a: NULL sig returns 0");
    CHECK(g_cb_fired > 0, "SHIM-NEW-012a: NULL sig fires illegal callback");

    // Valid call must still work
    g_cb_fired = 0;
    rc = secp256k1_ecdsa_signature_serialize_compact(ctx, out64, &sig);
    CHECK(rc == 1, "SHIM-NEW-012a: valid serialize_compact returns 1");
    CHECK(g_cb_fired == 0, "SHIM-NEW-012a: valid call does not fire callback");

    secp256k1_context_destroy(ctx);
}

static void test_serialize_der_null_fires_callback() {
    std::printf("  [SHIM-NEW-012b] serialize_der: NULL output fires illegal callback\n");

    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    secp256k1_context_set_illegal_callback(ctx, test_illegal_callback, nullptr);

    unsigned char sk[32] = {};
    sk[31] = 2;
    unsigned char msg[32] = {};
    msg[0] = 0xCD;
    secp256k1_ecdsa_signature sig{};
    secp256k1_ecdsa_sign(ctx, &sig, msg, sk, nullptr, nullptr);

    // NULL output
    size_t outlen = 72;
    g_cb_fired = 0;
    int rc = secp256k1_ecdsa_signature_serialize_der(ctx, nullptr, &outlen, &sig);
    CHECK(rc == 0, "SHIM-NEW-012b: NULL output returns 0");
    CHECK(g_cb_fired > 0, "SHIM-NEW-012b: NULL output fires illegal callback");

    // NULL outputlen
    unsigned char out[72]{};
    g_cb_fired = 0;
    rc = secp256k1_ecdsa_signature_serialize_der(ctx, out, nullptr, &sig);
    CHECK(rc == 0, "SHIM-NEW-012b: NULL outputlen returns 0");
    CHECK(g_cb_fired > 0, "SHIM-NEW-012b: NULL outputlen fires illegal callback");

    // NULL sig
    outlen = 72;
    g_cb_fired = 0;
    rc = secp256k1_ecdsa_signature_serialize_der(ctx, out, &outlen, nullptr);
    CHECK(rc == 0, "SHIM-NEW-012b: NULL sig returns 0");
    CHECK(g_cb_fired > 0, "SHIM-NEW-012b: NULL sig fires illegal callback");

    // Valid call must still work
    outlen = 72;
    g_cb_fired = 0;
    rc = secp256k1_ecdsa_signature_serialize_der(ctx, out, &outlen, &sig);
    CHECK(rc == 1, "SHIM-NEW-012b: valid serialize_der returns 1");
    CHECK(outlen > 0 && outlen <= 72, "SHIM-NEW-012b: valid outlen is in range");
    CHECK(g_cb_fired == 0, "SHIM-NEW-012b: valid call does not fire callback");

    secp256k1_context_destroy(ctx);
}

// ── SHIM-NEW-015: seckey_* NULL seckey fires illegal callback ──

static void test_seckey_null_fires_callback() {
    std::printf("  [SHIM-NEW-015] seckey_*: NULL seckey fires illegal callback\n");

    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    secp256k1_context_set_illegal_callback(ctx, test_illegal_callback, nullptr);

    // seckey_verify: NULL seckey
    g_cb_fired = 0;
    int rc = secp256k1_ec_seckey_verify(ctx, nullptr);
    CHECK(rc == 0, "SHIM-NEW-015a: seckey_verify NULL seckey returns 0");
    CHECK(g_cb_fired > 0, "SHIM-NEW-015a: seckey_verify NULL seckey fires callback");

    // seckey_negate: NULL seckey
    g_cb_fired = 0;
    rc = secp256k1_ec_seckey_negate(ctx, nullptr);
    CHECK(rc == 0, "SHIM-NEW-015b: seckey_negate NULL seckey returns 0");
    CHECK(g_cb_fired > 0, "SHIM-NEW-015b: seckey_negate NULL seckey fires callback");

    unsigned char tweak[32] = {};
    tweak[31] = 1;

    // seckey_tweak_add: NULL seckey
    g_cb_fired = 0;
    rc = secp256k1_ec_seckey_tweak_add(ctx, nullptr, tweak);
    CHECK(rc == 0, "SHIM-NEW-015c: seckey_tweak_add NULL seckey returns 0");
    CHECK(g_cb_fired > 0, "SHIM-NEW-015c: seckey_tweak_add NULL seckey fires callback");

    // seckey_tweak_add: NULL tweak
    unsigned char sk[32] = {};
    sk[31] = 5;
    g_cb_fired = 0;
    rc = secp256k1_ec_seckey_tweak_add(ctx, sk, nullptr);
    CHECK(rc == 0, "SHIM-NEW-015d: seckey_tweak_add NULL tweak returns 0");
    CHECK(g_cb_fired > 0, "SHIM-NEW-015d: seckey_tweak_add NULL tweak fires callback");

    // seckey_tweak_mul: NULL seckey
    g_cb_fired = 0;
    rc = secp256k1_ec_seckey_tweak_mul(ctx, nullptr, tweak);
    CHECK(rc == 0, "SHIM-NEW-015e: seckey_tweak_mul NULL seckey returns 0");
    CHECK(g_cb_fired > 0, "SHIM-NEW-015e: seckey_tweak_mul NULL seckey fires callback");

    // seckey_tweak_mul: NULL tweak
    sk[31] = 5;
    g_cb_fired = 0;
    rc = secp256k1_ec_seckey_tweak_mul(ctx, sk, nullptr);
    CHECK(rc == 0, "SHIM-NEW-015f: seckey_tweak_mul NULL tweak returns 0");
    CHECK(g_cb_fired > 0, "SHIM-NEW-015f: seckey_tweak_mul NULL tweak fires callback");

    // Confirm valid calls still work (no false-positive callback)
    sk[31] = 5;
    g_cb_fired = 0;
    rc = secp256k1_ec_seckey_verify(ctx, sk);
    CHECK(rc == 1, "SHIM-NEW-015g: valid seckey_verify returns 1");
    CHECK(g_cb_fired == 0, "SHIM-NEW-015g: valid seckey_verify does not fire callback");

    secp256k1_context_destroy(ctx);
}

} // namespace

int test_regression_shim_security_v9_run() {
    g_pass = 0; g_fail = 0;
    std::printf("[regression_shim_security_v9] SHIM-NEW-012 serialize NULL callbacks + SHIM-NEW-015 seckey NULL callbacks\n");

    test_serialize_compact_null_fires_callback();
    test_serialize_der_null_fires_callback();
    test_seckey_null_fires_callback();

    std::printf("  pass=%d  fail=%d\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_shim_security_v9_run(); }
#endif
