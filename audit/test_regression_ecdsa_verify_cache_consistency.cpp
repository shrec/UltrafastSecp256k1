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
#include <cstddef>

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

constexpr unsigned char kBlock704789Pubkey[33] = {
    0x03,0x9c,0xfc,0xfe,0x4a,0x5d,0x0e,0xfa,0xd2,0x73,0x82,
    0xe5,0xd2,0xb4,0x78,0xeb,0x39,0x8a,0x8b,0x69,0x1a,0x66,
    0xe0,0x1c,0x87,0x8b,0x60,0x0b,0x50,0x42,0xb3,0x31,0x66
};

constexpr unsigned char kBlock704789OppositePubkey[33] = {
    0x02,0x9c,0xfc,0xfe,0x4a,0x5d,0x0e,0xfa,0xd2,0x73,0x82,
    0xe5,0xd2,0xb4,0x78,0xeb,0x39,0x8a,0x8b,0x69,0x1a,0x66,
    0xe0,0x1c,0x87,0x8b,0x60,0x0b,0x50,0x42,0xb3,0x31,0x66
};

constexpr unsigned char kBlock704789Msg32[32] = {
    0x50,0x4d,0x68,0xbe,0xac,0x18,0x7d,0xd0,0xb2,0x59,0xdd,
    0xd6,0xed,0x6d,0x5d,0x63,0x48,0x15,0x0b,0x9b,0x23,0xee,
    0x6d,0xfd,0xb4,0x3e,0x87,0xf7,0x4d,0xd3,0xc5,0x47
};

constexpr unsigned char kBlock704789DerSig[70] = {
    0x30,0x44,0x02,0x20,0x43,0xf1,0x41,0x44,0x0d,0x4c,0xd2,
    0x8c,0xae,0x89,0x03,0xcb,0x59,0xce,0x70,0x67,0x63,0xff,
    0xf7,0x0d,0x74,0x6d,0x13,0xe1,0x71,0x3d,0xd6,0x20,0xc7,
    0xc7,0x34,0xb4,0x02,0x20,0x08,0xda,0xc3,0x36,0x56,0x10,
    0xad,0x1a,0x3f,0x78,0xfd,0x98,0x9e,0xa6,0x0d,0x81,0xc6,
    0xb2,0xe4,0xc8,0xec,0xae,0x3f,0x68,0xb4,0x32,0x6f,0xc3,
    0xdb,0xe1,0xf4,0x01
};

// Helper: stuff 64 raw bytes into the opaque secp256k1_pubkey.data. This is
// EXPLICITLY a hostile-caller pattern (bypasses ec_pubkey_parse), used here
// to expose the SHIM-013 inconsistency before the fix.
void pubkey_from_raw_bytes(secp256k1_pubkey* pk, const unsigned char x[32],
                           const unsigned char y[32]) noexcept {
    std::memcpy(pk->data,      x, 32);
    std::memcpy(pk->data + 32, y, 32);
}

void test_block_704789_cache_hit_tuple(secp256k1_context* ctx) {
    secp256k1_pubkey pk{};
    CHECK(secp256k1_ec_pubkey_parse(ctx, &pk, kBlock704789Pubkey,
                                    sizeof(kBlock704789Pubkey)) == 1,
          "CVC-4a: block 704789 compressed pubkey parses");

    secp256k1_ecdsa_signature parsed{};
    CHECK(secp256k1_ecdsa_signature_parse_der(ctx, &parsed, kBlock704789DerSig,
                                              sizeof(kBlock704789DerSig)) == 1,
          "CVC-4b: block 704789 DER signature parses");

    secp256k1_ecdsa_signature normalized{};
    (void)secp256k1_ecdsa_signature_normalize(ctx, &normalized, &parsed);

    int results[4]{};
    for (std::size_t i = 0; i < 4; ++i) {
        results[i] = secp256k1_ecdsa_verify(ctx, &normalized,
                                            kBlock704789Msg32, &pk);
    }

    CHECK(results[0] == 1, "CVC-4c: block 704789 tuple verifies on first encounter");
    CHECK(results[0] == results[1] && results[1] == results[2] &&
          results[2] == results[3],
          "CVC-4d: block 704789 tuple verifies consistently after ECDSA cache hits");
    CHECK(results[3] == 1, "CVC-4e: block 704789 tuple verifies after cache promotion");
}

void test_block_704789_opposite_parity_cache_poison(secp256k1_context* ctx) {
    secp256k1_pubkey target_pk{};
    secp256k1_pubkey opposite_pk{};
    CHECK(secp256k1_ec_pubkey_parse(ctx, &target_pk, kBlock704789Pubkey,
                                    sizeof(kBlock704789Pubkey)) == 1,
          "CVC-5a: target same-X odd-Y pubkey parses");
    CHECK(secp256k1_ec_pubkey_parse(ctx, &opposite_pk, kBlock704789OppositePubkey,
                                    sizeof(kBlock704789OppositePubkey)) == 1,
          "CVC-5b: opposite same-X even-Y pubkey parses");

    secp256k1_ecdsa_signature parsed{};
    CHECK(secp256k1_ecdsa_signature_parse_der(ctx, &parsed, kBlock704789DerSig,
                                              sizeof(kBlock704789DerSig)) == 1,
          "CVC-5c: block 704789 DER signature parses for poison test");

    secp256k1_ecdsa_signature normalized{};
    (void)secp256k1_ecdsa_signature_normalize(ctx, &normalized, &parsed);

    // Promote the opposite-Y key into ShimEcdsaCache. A cache keyed only by X
    // will then reuse the wrong EcdsaPublicKey for the target key.
    int poison1 = secp256k1_ecdsa_verify(ctx, &normalized,
                                         kBlock704789Msg32, &opposite_pk);
    int poison2 = secp256k1_ecdsa_verify(ctx, &normalized,
                                         kBlock704789Msg32, &opposite_pk);
    int target = secp256k1_ecdsa_verify(ctx, &normalized,
                                        kBlock704789Msg32, &target_pk);

    CHECK(poison1 == 0 && poison2 == 0,
          "CVC-5d: block 704789 signature does not verify under opposite-Y pubkey");
    CHECK(target == 1,
          "CVC-5e: same-X opposite-Y cache entry must not poison target pubkey verify");
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

    // -- CVC-4/5: libbitcoin block 704,789 release-only cache-hit regression --
    // libbitcoin's isolated unit test exercises parse -> normalize -> verify once
    // and therefore only sees the first-encounter direct Point path. Full node
    // validation can verify the same X coordinate after ShimEcdsaCache promotes
    // either parity to EcdsaPublicKey, so cover both repeated hits and same-X
    // opposite-Y cache poisoning.
    test_block_704789_opposite_parity_cache_poison(ctx);
    test_block_704789_cache_hit_tuple(ctx);

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
