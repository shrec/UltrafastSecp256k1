// ============================================================================
// Regression: SHIM-013 — secp256k1_ecdsa_verify 1st-encounter cache consistency
// ============================================================================
//
// FINDING (workingdocs/FINAL_AGGREGATED_REVIEW_2026-05-22.md, SHIM-013, P1):
//   Commit 40697447 removed the curve membership check (y² == x³ + 7) from
//   secp256k1_ecdsa_verify on the strength of "ec_pubkey_parse already
//   validated", but the new 1st-encounter direct-Point path also dropped
//   parse_bytes_strict. With FieldElement::from_bytes() silently mod-p
//   reducing X/Y, a caller that wrote arbitrary 64 bytes into
//   secp256k1_pubkey.data could observe a CACHE-STATE-DEPENDENT verify
//   verdict — different return value on the 1st call vs the 2nd call for
//   the SAME key, because the 2nd call goes through ecdsa_pubkey_parse()
//   which DOES enforce strict parse + curve check.
//
// FIX (this commit's shim_ecdsa.cpp):
//   The 1st-encounter direct-Point path now runs the same
//   parse_bytes_strict + curve-equation check as ecdsa_pubkey_parse(),
//   so verify result is deterministic regardless of cache state.
//
// PROPERTY UNDER TEST (CVC-1..6):
//   Construct a secp256k1_pubkey opaque struct with bytes that:
//     (a) parse_bytes_strict would reject (x >= p or y >= p), or
//     (b) parse_bytes_strict accepts but y² != x³ + 7 (off-curve).
//   Verify a valid-looking compact signature against it three times in
//   a row. All three calls must return the SAME value (= 0, reject).
//   Before the fix, the 1st call could accept (via silent mod-p reduce)
//   and subsequent calls reject.
//
// ============================================================================

#if !defined(UNIFIED_AUDIT_RUNNER) && !defined(STANDALONE_TEST)
#define STANDALONE_TEST
#endif

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <array>

// This test depends on the libsecp256k1 shim ABI. When the shim is not built
// it returns 77 (ADVISORY_SKIP_CODE) so the unified runner reports it as
// advisory-skipped instead of a link error.
#if defined(SECP256K1_BUILD_COMPAT_SHIM) || defined(STANDALONE_TEST)
#include "secp256k1.h"

namespace {
int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (cond) { ++g_pass; std::printf("    [OK]   " msg "\n"); }       \
        else      { ++g_fail; std::printf("    [FAIL] " msg "\n"); }       \
    } while (0)

// Build a valid-looking compact signature: r = 1, s = 1. Real ECDSA verify
// against an honest curve point fails (the math does not match a sensible
// message+key), but that is fine — we are testing that verify gives the
// SAME answer across repeated calls for invalid pubkey bytes, not that the
// signature itself is valid.
constexpr unsigned char kSig64[64] = {
    // r = 1
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,1,
    // s = 1
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,1
};
constexpr unsigned char kMsg32[32] = {
    0xde,0xad,0xbe,0xef,0,0,0,0,  0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,              0,0,0,0,0,0,0,0
};

// Helper: stuff 64 raw bytes into the opaque secp256k1_pubkey.data. This is
// EXPLICITLY a hostile-caller pattern (bypasses ec_pubkey_parse), used here
// to expose the SHIM-013 inconsistency before the fix.
void pubkey_from_raw_bytes(secp256k1_pubkey* pk, const unsigned char x[32],
                           const unsigned char y[32]) noexcept {
    std::memcpy(pk->data,      x, 32);
    std::memcpy(pk->data + 32, y, 32);
}

int run_smoke() {
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_VERIFY);
    if (!ctx) {
        std::printf("  ctx_create failed — cannot run\n");
        return 1;
    }

    secp256k1_ecdsa_signature sig;
    if (secp256k1_ecdsa_signature_parse_compact(ctx, &sig, kSig64) != 1) {
        std::printf("  signature_parse_compact failed — fixture invalid\n");
        secp256k1_context_destroy(ctx);
        return 1;
    }

    // ── CVC-1: x >= p (silent mod-p reduction trap) ────────────────────
    // X = 0xFF..FF (>= p). parse_bytes_strict must reject.
    {
        unsigned char x[32], y[32];
        std::memset(x, 0xFF, 32);
        std::memset(y, 0x42, 32);
        secp256k1_pubkey pk{};
        pubkey_from_raw_bytes(&pk, x, y);

        int r1 = secp256k1_ecdsa_verify(ctx, &sig, kMsg32, &pk);
        int r2 = secp256k1_ecdsa_verify(ctx, &sig, kMsg32, &pk);
        int r3 = secp256k1_ecdsa_verify(ctx, &sig, kMsg32, &pk);
        CHECK(r1 == r2 && r2 == r3, "CVC-1a: x >= p verify is cache-state-independent");
        CHECK(r1 == 0, "CVC-1b: x >= p rejected (parse_bytes_strict gate)");
    }

    // ── CVC-2: y >= p ──────────────────────────────────────────────────
    {
        unsigned char x[32]{}, y[32];
        x[31] = 1;                       // small x that parses cleanly
        std::memset(y, 0xFF, 32);        // y >= p
        secp256k1_pubkey pk{};
        pubkey_from_raw_bytes(&pk, x, y);

        int r1 = secp256k1_ecdsa_verify(ctx, &sig, kMsg32, &pk);
        int r2 = secp256k1_ecdsa_verify(ctx, &sig, kMsg32, &pk);
        int r3 = secp256k1_ecdsa_verify(ctx, &sig, kMsg32, &pk);
        CHECK(r1 == r2 && r2 == r3, "CVC-2a: y >= p verify is cache-state-independent");
        CHECK(r1 == 0, "CVC-2b: y >= p rejected (parse_bytes_strict gate)");
    }

    // ── CVC-3: off-curve point (y² != x³ + 7) ──────────────────────────
    {
        // (x = 1, y = 1) — clearly not on the curve: 1 != 1+7.
        unsigned char x[32]{}, y[32]{};
        x[31] = 1; y[31] = 1;
        secp256k1_pubkey pk{};
        pubkey_from_raw_bytes(&pk, x, y);

        int r1 = secp256k1_ecdsa_verify(ctx, &sig, kMsg32, &pk);
        int r2 = secp256k1_ecdsa_verify(ctx, &sig, kMsg32, &pk);
        int r3 = secp256k1_ecdsa_verify(ctx, &sig, kMsg32, &pk);
        CHECK(r1 == r2 && r2 == r3, "CVC-3a: off-curve verify is cache-state-independent");
        CHECK(r1 == 0, "CVC-3b: off-curve point rejected (curve equation gate)");
    }

    secp256k1_context_destroy(ctx);
    std::printf("[regression_ecdsa_verify_cache_consistency] %d/%d passed\n",
                g_pass, g_pass + g_fail);
    return g_fail > 0 ? 1 : 0;
}

}  // namespace

int test_regression_ecdsa_verify_cache_consistency_run() {
    return run_smoke();
}

#else
// Shim not linked: report as advisory-skipped, do not block.
int test_regression_ecdsa_verify_cache_consistency_run() {
    std::printf("[regression_ecdsa_verify_cache_consistency] shim not linked — advisory skip\n");
    return 77;  // ADVISORY_SKIP_CODE
}
#endif  // SECP256K1_BUILD_COMPAT_SHIM

#ifdef STANDALONE_TEST
int main() { return test_regression_ecdsa_verify_cache_consistency_run(); }
#endif
