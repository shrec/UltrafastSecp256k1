// ============================================================================
// test_regression_shim_divergence_fixes.cpp
// ============================================================================
// Regression tests for four shim divergences fixed 2026-05-26:
//
//   SHIM-ILLCB-002: secp256k1_ec_pubkey_parse NULL args must fire illegal_cb
//   DER-STRICT:     secp256k1_ecdsa_signature_parse_der REJECTS r=0/s=0 at parse
//                   (the canonical zero encoding 02 01 00 fails the minimal-
//                   encoding rule). Documented divergence from upstream (which
//                   accepts at parse and rejects at verify) — see
//                   docs/SHIM_KNOWN_DIVERGENCES.md and shim_ecdsa.cpp:314-321.
//   SHIM-ILLCB-001: secp256k1_context_set_illegal_callback(NULL ctx) must
//                   call default_illegal_callback (aborts) rather than silently
//                   returning — tested via code-review only (cannot survive abort)
//   keypair_sec:    secp256k1_keypair_create must store BIP-340-normalized
//                   seckey (negate k when P.y is odd)
//
// Tests:
//   SDF-1: pubkey_parse(ctx, NULL, input, 33)  → callback fired + return 0
//   SDF-2: pubkey_parse(ctx, pubkey, NULL, 33) → callback fired + return 0
//   SDF-3: ecdsa_signature_parse_der with r=0, s=1 → returns 0 (rejected at parse)
//   SDF-4: context_set_illegal_callback(NULL, ...) → abort() [code-review only]
//   SDF-5: keypair_create seckey → keypair_sec returns BIP-340 normalized key
//          (i.e. the key that produces an even-Y pubkey)
//   SDF-6: keypair_create + schnorrsig_sign32 + schnorrsig_verify roundtrip
// ============================================================================

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <atomic>
#include <array>

static int g_pass = 0, g_fail = 0;
#include "audit_check.hpp"

#if __has_include("secp256k1.h") && __has_include("secp256k1_extrakeys.h")
#include "secp256k1.h"
#include "secp256k1_extrakeys.h"
#if __has_include("secp256k1_schnorrsig.h")
#include "secp256k1_schnorrsig.h"
#define HAVE_SCHNORRSIG 1
#endif

static std::atomic<int> g_cb_count{0};

static void counting_illegal_cb(const char* /*msg*/, void* /*data*/) {
    ++g_cb_count;
}

static secp256k1_context* make_ctx() {
    secp256k1_context* ctx = secp256k1_context_create(
        SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    secp256k1_context_set_illegal_callback(ctx, counting_illegal_cb, nullptr);
    return ctx;
}

// A valid secp256k1 private key that produces an ODD-Y public key.
// sk=1: P = G, G.y = 0x4…even. Use sk=2 instead for variety, but sk=1 is fine.
// For SDF-5 we need any valid key; the test checks the stored secret is the
// BIP-340-normalized version (even-Y), not necessarily the original input.
static const unsigned char g_seckey[32] = {
    0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0,
    0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,1
};

// ─── SDF-1: pubkey_parse NULL pubkey output fires illegal_callback ────────────
static void test_sdf1_pubkey_parse_null_pubkey() {
    secp256k1_context* ctx = make_ctx();
    // Build a valid compressed pubkey bytes for G (sk=1)
    secp256k1_pubkey tmp{};
    secp256k1_ec_pubkey_create(ctx, &tmp, g_seckey);
    unsigned char compressed[33]{};
    size_t len = 33;
    secp256k1_ec_pubkey_serialize(ctx, compressed, &len, &tmp, SECP256K1_EC_COMPRESSED);

    int before = g_cb_count.load();
    int rc = secp256k1_ec_pubkey_parse(ctx, nullptr, compressed, 33);
    int after = g_cb_count.load();
    CHECK(rc == 0,        "[SDF-1] pubkey_parse(NULL pubkey) must return 0");
    CHECK(after > before, "[SDF-1] pubkey_parse(NULL pubkey) must fire illegal_callback");
    secp256k1_context_destroy(ctx);
}

// ─── SDF-2: pubkey_parse NULL input fires illegal_callback ───────────────────
static void test_sdf2_pubkey_parse_null_input() {
    secp256k1_context* ctx = make_ctx();
    secp256k1_pubkey pubkey{};
    int before = g_cb_count.load();
    int rc = secp256k1_ec_pubkey_parse(ctx, &pubkey, nullptr, 33);
    int after = g_cb_count.load();
    CHECK(rc == 0,        "[SDF-2] pubkey_parse(NULL input) must return 0");
    CHECK(after > before, "[SDF-2] pubkey_parse(NULL input) must fire illegal_callback");
    secp256k1_context_destroy(ctx);
}

// ─── SDF-3: DER signature with r=0, s=1 is REJECTED at parse ─────────────────
// DOCUMENTED DIVERGENCE: upstream libsecp256k1 accepts r=0 at parse time and
// defers rejection to verify. This shim rejects it AT PARSE: the canonical zero
// encoding `02 01 00` fails parse_int's minimal-encoding rule (a lone leading
// 0x00 with len<2). See shim_ecdsa.cpp:314-321 and docs/SHIM_KNOWN_DIVERGENCES.md.
// This must stay consistent with the authoritative test_shim_der_zero_r.cpp and
// shim_test.cpp::test_der_parity, which assert the same reject-at-parse result.
// DER encoding of SEQUENCE { INTEGER 0, INTEGER 1 }:
//   30 06   SEQUENCE, 6 bytes
//   02 01 00  INTEGER, 1 byte, value 0
//   02 01 01  INTEGER, 1 byte, value 1
static void test_sdf3_der_parse_r_zero() {
    secp256k1_context* ctx = make_ctx();
    static const unsigned char der_r0[8] = {
        0x30, 0x06,
        0x02, 0x01, 0x00,   // r = 0
        0x02, 0x01, 0x01    // s = 1
    };
    secp256k1_ecdsa_signature sig{};
    int rc = secp256k1_ecdsa_signature_parse_der(ctx, &sig, der_r0, sizeof(der_r0));
    CHECK(rc == 0, "[SDF-3] parse_der with r=0 must return 0 (shim rejects zero at parse: 02 01 00 fails minimal-encoding; documented divergence)");
    secp256k1_context_destroy(ctx);
}

// ─── SDF-4: context_set_illegal_callback(NULL ctx) ───────────────────────────
// This calls default_illegal_callback which calls std::abort(). It cannot be
// tested in-process without signal handling. This is documented as a code-review
// fix; the behaviour is verified by inspection of shim_context.cpp.
// We emit a note so the test output references the fix.
static void test_sdf4_null_ctx_callback_note() {
    printf("  [SDF-4] context_set_illegal_callback(NULL ctx) → abort() "
           "[code-review verified, not in-process testable]\n");
    (void)std::fflush(stdout);
    // No in-process assertion: abort() cannot be caught without signal handling.
}

// ─── SDF-5: keypair_create stores BIP-340-normalized seckey ──────────────────
// After keypair_create, keypair_sec must return the key d such that d*G has
// even Y. If the raw input key k would produce odd-Y, the stored key is n-k.
// We verify: stored_sk * G produces a pubkey with even Y.
static void test_sdf5_keypair_sec_bip340_normalized() {
    secp256k1_context* ctx = make_ctx();

    // Try multiple keys to hit both even-Y and odd-Y cases.
    // sk=1..4 cover typical small values; sk=3 (empirically) has odd Y.
    for (unsigned char v = 1; v <= 4; ++v) {
        unsigned char raw_sk[32]{};
        raw_sk[31] = v;

        secp256k1_keypair kp{};
        int rc_create = secp256k1_keypair_create(ctx, &kp, raw_sk);
        if (rc_create != 1) continue; // skip invalid keys (shouldn't happen for v=1..4)

        // Retrieve stored seckey
        unsigned char stored_sk[32]{};
        int rc_sec = secp256k1_keypair_sec(ctx, stored_sk, &kp);
        CHECK(rc_sec == 1,
            std::string("[SDF-5] keypair_sec must succeed for sk=") + std::to_string(v));

        // Derive pubkey from the stored (normalized) seckey
        secp256k1_pubkey pub{};
        int rc_pub = secp256k1_ec_pubkey_create(ctx, &pub, stored_sk);
        CHECK(rc_pub == 1,
            std::string("[SDF-5] pubkey_create from stored_sk must succeed for sk=") + std::to_string(v));

        // Serialize compressed: prefix 0x02 = even Y, 0x03 = odd Y
        unsigned char compressed[33]{};
        size_t clen = 33;
        secp256k1_ec_pubkey_serialize(ctx, compressed, &clen, &pub, SECP256K1_EC_COMPRESSED);
        CHECK(compressed[0] == 0x02,
            std::string("[SDF-5] stored_sk must produce even-Y pubkey (BIP-340) for sk=") + std::to_string(v));
    }

    secp256k1_context_destroy(ctx);
}

// ─── SDF-6: schnorrsig_sign32 + verify roundtrip via keypair_create ──────────
#ifdef HAVE_SCHNORRSIG
static void test_sdf6_schnorr_sign_verify_roundtrip() {
    secp256k1_context* ctx = make_ctx();

    static const unsigned char msg32[32] = {
        0xde,0xad,0xbe,0xef, 0xca,0xfe,0xba,0xbe,
        0x01,0x23,0x45,0x67, 0x89,0xab,0xcd,0xef,
        0xfe,0xdc,0xba,0x98, 0x76,0x54,0x32,0x10,
        0x00,0x11,0x22,0x33, 0x44,0x55,0x66,0x77
    };

    for (unsigned char v = 1; v <= 4; ++v) {
        unsigned char raw_sk[32]{};
        raw_sk[31] = v;

        secp256k1_keypair kp{};
        if (secp256k1_keypair_create(ctx, &kp, raw_sk) != 1) continue;

        unsigned char sig64[64]{};
        int rc_sign = secp256k1_schnorrsig_sign32(ctx, sig64, msg32, &kp, nullptr);
        CHECK(rc_sign == 1,
            std::string("[SDF-6] schnorrsig_sign32 must succeed for sk=") + std::to_string(v));

        secp256k1_xonly_pubkey xpk{};
        secp256k1_keypair_xonly_pub(ctx, &xpk, nullptr, &kp);

        int rc_verify = secp256k1_schnorrsig_verify(ctx, sig64, msg32, 32, &xpk);
        CHECK(rc_verify == 1,
            std::string("[SDF-6] schnorrsig_verify must succeed for sk=") + std::to_string(v));
    }

    secp256k1_context_destroy(ctx);
}
#endif // HAVE_SCHNORRSIG

#else
// Shim not available — advisory skip
static void test_sdf_shim_not_available() {
    printf("  [SDF] shim not available — advisory skip\n");
}
#endif // __has_include

// ─── Entry point ─────────────────────────────────────────────────────────────

#ifndef UNIFIED_AUDIT_RUNNER
#define STANDALONE_TEST
int main() {
#else
int test_regression_shim_divergence_fixes_run() {
#endif
    printf("[shim_divergence_fixes] 2026-05-26: ILLCB-001/002, DER-STRICT, keypair_sec BIP-340 normalization\n");

#if __has_include("secp256k1.h") && __has_include("secp256k1_extrakeys.h")
    test_sdf1_pubkey_parse_null_pubkey();
    test_sdf2_pubkey_parse_null_input();
    test_sdf3_der_parse_r_zero();
    test_sdf4_null_ctx_callback_note();
    test_sdf5_keypair_sec_bip340_normalized();
#ifdef HAVE_SCHNORRSIG
    test_sdf6_schnorr_sign_verify_roundtrip();
#endif
#else
    test_sdf_shim_not_available();
    return ADVISORY_SKIP_CODE;
#endif

    printf("[shim_divergence_fixes] %d passed, %d failed\n", g_pass, g_fail);
    return g_fail;
}
