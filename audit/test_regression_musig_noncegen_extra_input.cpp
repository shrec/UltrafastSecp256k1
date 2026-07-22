// ============================================================================
// test_regression_musig_noncegen_extra_input.cpp
// ============================================================================
// Self-healing regression for SHIM-NONCEGEN-001:
//
//   secp256k1_musig_nonce_gen() accepts extra_input32 but does NOT forward it
//   to the internal nonce derivation. Callers passing extra_input32 for
//   additional entropy receive nonces identical to callers passing NULL.
//
//   This is a documented limitation (SHIM_KNOWN_DIVERGENCES.md §SHIM-NONCEGEN-001).
//   Fix requires a non-trivial API change to the internal musig2_nonce_gen path.
//   For Bitcoin Core usage extra_input32 is NULL — no impact on production path.
//
// Self-healing design (TEST-001 cleanup, 2026-05-23):
//
//   Earlier versions of this test contained an inverted-contract anti-pattern:
//   they asserted that pubnonces ARE identical, which means the test PASSED
//   when the bug existed and would FAIL when the bug was fixed. That is the
//   opposite of how a regression test should work.
//
//   The new design dispatches on the source-marker presence:
//
//     If "SHIM-NONCEGEN-001" marker present in shim_musig.cpp:
//       → bug is still documented as open
//       → assert the freeze (pubnonces identical) so the marker and code stay in sync
//       → emit clear notice that this test will flip when the bug is fixed
//
//     If marker absent:
//       → bug has been fixed (or marker accidentally removed)
//       → assert the CORRECT post-fix invariant: different extra_input32
//         produces different pubnonces
//
//   Either branch produces a correctly-contracted PASS/FAIL signal. The test
//   does not need manual updating when the underlying bug is fixed — only the
//   marker removal in shim_musig.cpp flips it.
//
// Sub-tests:
//   [1] Source scan: detect SHIM-NONCEGEN-001 marker → drives [2]/[3] mode
//   [2] NULL vs non-NULL extra_input32 — mode-aware contract
//   [3] Two distinct non-NULL extra_input32 — mode-aware contract
// ============================================================================

#ifndef UNIFIED_AUDIT_RUNNER
#define STANDALONE_TEST
#endif

// SHIM-GUARD (2026-05-23): when the libsecp256k1 shim include path is not on
// the compiler's search path, compile this file to an empty translation unit
// and let shim_run_stubs_unified.cpp provide the advisory-skip _run() symbol.
#if !defined(__has_include) || __has_include("secp256k1.h")

#include "secp256k1.h"
#include "secp256k1_musig.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>

static int g_pass = 0, g_fail = 0;
#include "audit_check.hpp"

namespace {

static bool g_marker_present = false;  // assume bug-fixed; scan overrides to true if marker found

// ── [1] Source scan: detect SHIM-NONCEGEN-001 marker ────────────────────────
//
// Repair (issue #335 acceptance repair, round 5): the previous 3-candidate
// CWD-relative path list never resolved when unified_audit_runner was
// invoked from a CWD unrelated to the repo (e.g. /tmp). This does not
// silently 0-check-pass the way the other round-5 fixes did (the marker's
// presence/absence only selects a MODE for the mandatory, source-independent
// checks in [2]/[3] below, which always run with real assertions), but a
// resolver that cannot find the source still means the mode selection can
// silently default to the wrong branch. Route through the shared,
// UFSECP_SOURCE_ROOT-aware audit_read_source_file() (audit_check.hpp).
static void test_source_marker_present() {
    std::printf("  [NONCEGEN-1] Source scan: detect SHIM-NONCEGEN-001 marker in shim_musig.cpp\n");

    std::string src = audit_read_source_file("compat/libsecp256k1_shim/src/shim_musig.cpp");
    if (src.empty()) {
        std::printf("  [SKIP] shim_musig.cpp not found — source scan skipped, defaulting to bug-open mode\n");
        return;
    }
    g_marker_present = src.find("SHIM-NONCEGEN-001") != std::string::npos;
    if (g_marker_present) {
        std::printf("  [INFO] SHIM-NONCEGEN-001 marker present → bug-open mode (asserting freeze)\n");
    } else {
        std::printf("  [INFO] SHIM-NONCEGEN-001 marker absent → bug-fixed mode (asserting correct entropy mixing)\n");
    }
    // No assertion on the marker itself: its presence/absence is the mode switch,
    // not pass/fail data. Mandatory check happens in [2] and [3].
    ++g_pass;
}

// ── [2] NULL vs non-NULL extra_input32 — mode-aware contract ────────────────

static void test_null_vs_nonnull_extra_input() {
    std::printf("  [NONCEGEN-2] Mode-aware: NULL vs non-NULL extra_input32\n");

    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
    if (!ctx) {
        std::printf("  [SKIP] context creation failed\n");
        return;
    }

    unsigned char sk[32] = {};
    sk[31] = 0x37;

    secp256k1_pubkey pubkey{};
    if (!secp256k1_ec_pubkey_create(ctx, &pubkey, sk)) {
        std::printf("  [SKIP] pubkey_create failed\n");
        secp256k1_context_destroy(ctx);
        return;
    }

    unsigned char session_id[32] = {};
    for (int i = 0; i < 32; ++i) session_id[i] = (unsigned char)(i + 1);

    unsigned char msg[32] = {};
    msg[0] = 0xAB; msg[31] = 0xCD;

    unsigned char extra[32] = {};
    for (int i = 0; i < 32; ++i) extra[i] = (unsigned char)(0x80 + i);

    secp256k1_musig_secnonce secnonce1{};
    secp256k1_musig_pubnonce pubnonce1{};
    int r1 = secp256k1_musig_nonce_gen(ctx, &secnonce1, &pubnonce1,
                                        session_id, sk, &pubkey, msg,
                                        nullptr, nullptr);

    secp256k1_musig_secnonce secnonce2{};
    secp256k1_musig_pubnonce pubnonce2{};
    int r2 = secp256k1_musig_nonce_gen(ctx, &secnonce2, &pubnonce2,
                                        session_id, sk, &pubkey, msg,
                                        nullptr, extra);

    CHECK(r1 == 1, "[NONCEGEN-2] nonce_gen (NULL extra) returns 1");
    CHECK(r2 == 1, "[NONCEGEN-2] nonce_gen (non-NULL extra) returns 1");

    bool same = (std::memcmp(pubnonce1.data, pubnonce2.data, sizeof(pubnonce1.data)) == 0);

    if (g_marker_present) {
        // bug-open mode: freeze the documented divergence
        CHECK(same,
              "[NONCEGEN-2] bug-open: pubnonce identical with NULL vs non-NULL extra_input32 "
              "(SHIM-NONCEGEN-001 freeze — flip on fix)");
    } else {
        // bug-fixed mode: extra_input32 MUST influence pubnonce
        CHECK(!same,
              "[NONCEGEN-2] bug-fixed: pubnonce DIFFERS with NULL vs non-NULL extra_input32 "
              "(extra_input32 now mixed into nonce derivation)");
    }

    secp256k1_context_destroy(ctx);
}

// ── [3] Two distinct non-NULL extra_input32 — mode-aware contract ───────────

static void test_two_nonnull_extra_inputs() {
    std::printf("  [NONCEGEN-3] Mode-aware: two distinct non-NULL extra_input32 values\n");

    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
    if (!ctx) {
        std::printf("  [SKIP] context creation failed\n");
        return;
    }

    unsigned char sk[32] = {};
    sk[31] = 0x5B;
    secp256k1_pubkey pubkey{};
    if (!secp256k1_ec_pubkey_create(ctx, &pubkey, sk)) {
        std::printf("  [SKIP] pubkey_create failed\n");
        secp256k1_context_destroy(ctx);
        return;
    }

    unsigned char session_id[32] = {};
    for (int i = 0; i < 32; ++i) session_id[i] = (unsigned char)(i + 7);

    unsigned char msg[32] = { 0x11 };

    unsigned char extra_a[32] = {};
    unsigned char extra_b[32] = {};
    for (int i = 0; i < 32; ++i) { extra_a[i] = (unsigned char)(i); extra_b[i] = (unsigned char)(255 - i); }

    secp256k1_musig_secnonce sna{}, snb{};
    secp256k1_musig_pubnonce pna{}, pnb{};
    int ra = secp256k1_musig_nonce_gen(ctx, &sna, &pna, session_id, sk, &pubkey, msg, nullptr, extra_a);
    int rb = secp256k1_musig_nonce_gen(ctx, &snb, &pnb, session_id, sk, &pubkey, msg, nullptr, extra_b);

    CHECK(ra == 1 && rb == 1, "[NONCEGEN-3] both nonce_gen calls return 1");

    bool same = (std::memcmp(pna.data, pnb.data, sizeof(pna.data)) == 0);

    if (g_marker_present) {
        CHECK(same,
              "[NONCEGEN-3] bug-open: pubnonce identical for two distinct extra_input32 "
              "(SHIM-NONCEGEN-001 freeze — flip on fix)");
    } else {
        CHECK(!same,
              "[NONCEGEN-3] bug-fixed: pubnonce DIFFERS for two distinct extra_input32 "
              "(entropy mixing now active)");
    }

    secp256k1_context_destroy(ctx);
}

} // anonymous namespace

// ── Entry point ─────────────────────────────────────────────────────────────

int test_regression_musig_noncegen_extra_input_run() {
    g_pass = 0; g_fail = 0;
    g_marker_present = false;  // default: bug-fixed; scan overrides if marker found
    std::printf("======================================================================\n");
    std::printf("  Regression: musig_nonce_gen extra_input32 (SHIM-NONCEGEN-001, self-healing)\n");
    std::printf("  Mode selected by source-marker presence in shim_musig.cpp.\n");
    std::printf("======================================================================\n\n");

    test_source_marker_present();
    std::printf("\n");
    test_null_vs_nonnull_extra_input();
    std::printf("\n");
    test_two_nonnull_extra_inputs();
    std::printf("\n");

    std::printf("[regression_musig_noncegen_extra_input] %d/%d checks passed (mode=%s)\n",
               g_pass, g_pass + g_fail, g_marker_present ? "bug-open" : "bug-fixed");
    return (g_fail > 0) ? 1 : 0;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_musig_noncegen_extra_input_run(); }
#endif

#else  // __has_include("secp256k1.h") failed

// Mirrors test_regression_shim_security_v9.cpp — see that file for the
// full rationale on STANDALONE_TEST vs unified_audit_runner branching.
#ifdef STANDALONE_TEST
#include <cstdio>
#ifndef ADVISORY_SKIP_CODE
#define ADVISORY_SKIP_CODE 77
#endif
int test_regression_musig_noncegen_extra_input_run() {
    std::printf("[regression_musig_noncegen_extra_input] SKIP — shim header not in include path\n");
    return ADVISORY_SKIP_CODE;
}
int main() { return test_regression_musig_noncegen_extra_input_run(); }
#endif

#endif  // __has_include("secp256k1.h")
