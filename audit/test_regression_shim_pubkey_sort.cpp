// ============================================================================
// test_regression_shim_pubkey_sort.cpp
// Regression: secp256k1_ec_pubkey_sort crashed on every call.
//
// Bug fixed: shim_pubkey.cpp's sort comparator lambda passed nullptr as the
// context to secp256k1_ec_pubkey_serialize, triggering SHIM_REQUIRE_CTX which
// called default_illegal_callback → std::abort() on every comparison.
// Fix: pass secp256k1_context_static instead of nullptr.
//
// Tests:
//   PST-1: sort with 1 key does not crash and returns correct order
//   PST-2: sort with 2 keys produces lexicographic order
//   PST-3: sort with 3 keys is stable and correct
//   PST-4: sort with null/zero array returns without crash
// ============================================================================

#include "audit_check.hpp"
#include <cstdio>
#include <cstring>
#include "secp256k1.h"

static secp256k1_pubkey make_pubkey(secp256k1_context* ctx, unsigned char scalar_byte) {
    unsigned char sk[32] = {};
    sk[31] = scalar_byte;
    secp256k1_pubkey pk{};
    secp256k1_ec_pubkey_create(ctx, &pk, sk);
    return pk;
}

// ── PST-1: single key sort — no crash ─────────────────────────────────────
static void test_sort_single() {
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    secp256k1_pubkey pk = make_pubkey(ctx, 1);
    const secp256k1_pubkey* pks[1] = { &pk };
    // Before fix: this would abort(). After fix: completes normally.
    secp256k1_ec_pubkey_sort(ctx, pks, 1);
    check(true, "[PST-1] single-key sort completes without crash");
    secp256k1_context_destroy(ctx);
}

// ── PST-2: two keys produce deterministic sorted order ────────────────────
static void test_sort_two_keys() {
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    secp256k1_pubkey pk1 = make_pubkey(ctx, 1);
    secp256k1_pubkey pk2 = make_pubkey(ctx, 2);

    // Serialize both to get expected order
    unsigned char comp1[33] = {}, comp2[33] = {};
    size_t l = 33;
    secp256k1_ec_pubkey_serialize(ctx, comp1, &l, &pk1, SECP256K1_EC_COMPRESSED);
    secp256k1_ec_pubkey_serialize(ctx, comp2, &l, &pk2, SECP256K1_EC_COMPRESSED);
    int expected_order = memcmp(comp1, comp2, 33); // <0 means pk1 < pk2

    secp256k1_pubkey pks_arr[2] = { pk2, pk1 }; // reverse order
    const secp256k1_pubkey* pks[2] = { &pks_arr[0], &pks_arr[1] };
    secp256k1_ec_pubkey_sort(ctx, pks, 2);

    unsigned char sorted0[33] = {}, sorted1[33] = {};
    secp256k1_ec_pubkey_serialize(ctx, sorted0, &l, pks[0], SECP256K1_EC_COMPRESSED);
    secp256k1_ec_pubkey_serialize(ctx, sorted1, &l, pks[1], SECP256K1_EC_COMPRESSED);
    int after_order = memcmp(sorted0, sorted1, 33);

    check(after_order < 0 || after_order == 0, "[PST-2a] sorted[0] <= sorted[1]");
    if (expected_order < 0) {
        // pk1 < pk2: after sort, sorted[0] should be pk1
        check(memcmp(sorted0, comp1, 33) == 0, "[PST-2b] smaller key is first");
    } else {
        check(memcmp(sorted0, comp2, 33) == 0, "[PST-2c] smaller key is first (reversed)");
    }

    secp256k1_context_destroy(ctx);
}

// ── PST-3: three keys produce stable sorted order ─────────────────────────
static void test_sort_three_keys() {
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    secp256k1_pubkey pk[3];
    pk[0] = make_pubkey(ctx, 3);
    pk[1] = make_pubkey(ctx, 1);
    pk[2] = make_pubkey(ctx, 2);

    const secp256k1_pubkey* pks[3] = { &pk[0], &pk[1], &pk[2] };
    secp256k1_ec_pubkey_sort(ctx, pks, 3);

    size_t l = 33;
    unsigned char s[3][33] = {};
    for (int i = 0; i < 3; ++i)
        secp256k1_ec_pubkey_serialize(ctx, s[i], &l, pks[i], SECP256K1_EC_COMPRESSED);
    check(memcmp(s[0], s[1], 33) <= 0, "[PST-3a] s[0] <= s[1]");
    check(memcmp(s[1], s[2], 33) <= 0, "[PST-3b] s[1] <= s[2]");

    secp256k1_context_destroy(ctx);
}

// ── PST-4: n=0 returns immediately ────────────────────────────────────────
static void test_sort_zero_count() {
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    secp256k1_ec_pubkey_sort(ctx, nullptr, 0);
    check(true, "[PST-4] n=0 with null array does not crash");
    secp256k1_context_destroy(ctx);
}

int test_regression_shim_pubkey_sort_run() {
    g_pass = 0; g_fail = 0;
    std::printf("[regression_shim_pubkey_sort] pubkey_sort no longer crashes via nullptr ctx\n");
    test_sort_single();
    test_sort_two_keys();
    test_sort_three_keys();
    test_sort_zero_count();
    std::printf("  pass=%d  fail=%d\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_shim_pubkey_sort_run(); }
#endif
