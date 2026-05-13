// ============================================================================
// test_regression_adaptor_degenerate_v7.cpp
// ============================================================================
// T-09: ufsecp_ecdsa_adaptor_sign must not emit all-zero pre-sig on degenerate
// output (Rule 4: ABI wrappers must not serialize zero output as success).
// This test verifies the normal round-trip path and null-arg fail-closed behavior.
// ============================================================================

#ifndef UNIFIED_AUDIT_RUNNER
#define STANDALONE_TEST
#endif

#ifndef UFSECP_BUILDING
#define UFSECP_BUILDING
#endif
#include "ufsecp/ufsecp.h"

#include <cstdio>
#include <cstring>

namespace {

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { ++g_pass; } \
    else { ++g_fail; std::printf("  [FAIL] %s\n", (msg)); } \
} while(0)

// privkey = 1 (known generator)
static const uint8_t kPrivkey1[32] = {
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,1
};
// privkey = 2 (adaptor secret)
static const uint8_t kPrivkey2[32] = {
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,2
};
static const uint8_t kMsg[32] = {0xAA};

// ── Normal round-trip: sign / verify / adapt / extract ──────────────────────

static void test_adaptor_roundtrip() {
    std::printf("  [adaptor_roundtrip]\n");

    ufsecp_ctx* ctx = nullptr;
    CHECK(ufsecp_ctx_create(&ctx) == UFSECP_OK, "ctx_create");

    uint8_t pub33[33], adaptor_pt[33];
    CHECK(ufsecp_pubkey_create(ctx, kPrivkey1, pub33) == UFSECP_OK,
          "pubkey signer");
    CHECK(ufsecp_pubkey_create(ctx, kPrivkey2, adaptor_pt) == UFSECP_OK,
          "pubkey adaptor point");

    // Sign
    uint8_t presig[UFSECP_ECDSA_ADAPTOR_SIG_LEN] = {};
    CHECK(ufsecp_ecdsa_adaptor_sign(ctx, kPrivkey1, kMsg, adaptor_pt, presig) == UFSECP_OK,
          "adaptor_sign succeeds (T-09 normal path)");

    // R_hat (33 bytes) must not be all-zeros.
    bool rhat_nonzero = false;
    for (int i = 0; i < 33; ++i) rhat_nonzero |= (presig[i] != 0);
    CHECK(rhat_nonzero, "R_hat in pre-sig is non-zero (T-09: degenerate output check works)");

    // s_hat (bytes 33..64) must not be all-zeros.
    bool shat_nonzero = false;
    for (int i = 33; i < 65; ++i) shat_nonzero |= (presig[i] != 0);
    CHECK(shat_nonzero, "s_hat in pre-sig is non-zero");

    // Verify pre-signature.
    CHECK(ufsecp_ecdsa_adaptor_verify(ctx, presig, pub33, kMsg, adaptor_pt) == UFSECP_OK,
          "adaptor_verify accepts valid pre-sig");

    // Adapt → final sig.
    uint8_t sig64[64] = {};
    CHECK(ufsecp_ecdsa_adaptor_adapt(ctx, presig, kPrivkey2, sig64) == UFSECP_OK,
          "adaptor_adapt succeeds");

    // Extract adaptor secret.
    uint8_t extracted[32] = {};
    CHECK(ufsecp_ecdsa_adaptor_extract(ctx, presig, sig64, extracted) == UFSECP_OK,
          "adaptor_extract succeeds");

    // Extracted secret should reconstruct the adaptor point (or its negation).
    uint8_t ext_pt[33] = {}, neg_pt[33] = {};
    CHECK(ufsecp_pubkey_create(ctx, extracted, ext_pt) == UFSECP_OK,
          "pubkey from extracted secret");
    ufsecp_pubkey_negate(ctx, ext_pt, neg_pt);
    CHECK(std::memcmp(ext_pt, adaptor_pt, 33) == 0 ||
          std::memcmp(neg_pt, adaptor_pt, 33) == 0,
          "extracted secret reproduces adaptor point (or negation)");

    // NEW-TEST-004: the adapted final signature must verify against the signer's
    // pubkey. This closes the round-trip: sign → verify → adapt → verify_final.
    // Without this check, a degenerate adapt that emits an invalid sig64 would
    // not be caught by the extraction check alone.
    CHECK(ufsecp_ecdsa_verify(ctx, kMsg, sig64, pub33) == UFSECP_OK,
          "NEW-TEST-004: adapted sig64 verifies under signer pubkey (full round-trip)");

    ufsecp_ctx_destroy(ctx);
}

// ── Null-arg fail-closed ─────────────────────────────────────────────────────

static void test_adaptor_null_args() {
    std::printf("  [adaptor_null_args]\n");
    ufsecp_ctx* ctx = nullptr;
    CHECK(ufsecp_ctx_create(&ctx) == UFSECP_OK, "ctx_create");

    uint8_t adaptor_pt[33] = {0x02, 1};  // minimal valid-looking compressed pt
    uint8_t presig[UFSECP_ECDSA_ADAPTOR_SIG_LEN] = {};

    CHECK(ufsecp_ecdsa_adaptor_sign(nullptr, kPrivkey1, kMsg, adaptor_pt, presig) != UFSECP_OK,
          "null ctx rejected");
    CHECK(ufsecp_ecdsa_adaptor_sign(ctx, nullptr, kMsg, adaptor_pt, presig) != UFSECP_OK,
          "null privkey rejected");
    CHECK(ufsecp_ecdsa_adaptor_sign(ctx, kPrivkey1, nullptr, adaptor_pt, presig) != UFSECP_OK,
          "null msg rejected");
    CHECK(ufsecp_ecdsa_adaptor_sign(ctx, kPrivkey1, kMsg, nullptr, presig) != UFSECP_OK,
          "null adaptor_point rejected");
    CHECK(ufsecp_ecdsa_adaptor_sign(ctx, kPrivkey1, kMsg, adaptor_pt, nullptr) != UFSECP_OK,
          "null output rejected");

    // Zero private key rejected.
    static const uint8_t zero_key[32] = {};
    CHECK(ufsecp_ecdsa_adaptor_sign(ctx, zero_key, kMsg, adaptor_pt, presig) != UFSECP_OK,
          "zero private key rejected");

    ufsecp_ctx_destroy(ctx);
}

} // namespace

int test_regression_adaptor_degenerate_v7_run() {
    g_pass = 0; g_fail = 0;
    std::printf("[regression_adaptor_degenerate_v7] T-09 adaptor degenerate guard + round-trip\n");

    test_adaptor_roundtrip();
    test_adaptor_null_args();

    std::printf("  pass=%d  fail=%d\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_adaptor_degenerate_v7_run(); }
#endif
