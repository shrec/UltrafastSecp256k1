// ============================================================================
// test_regression_ecdsa_batch_curve_check.cpp
// Regression: CA-001 — secp256k1_ecdsa_verify_batch must reject invalid-curve
// pubkeys consistently in BOTH the small-batch (n<8) and large-batch (n>=8)
// code paths.
//
// Prior to the fix, the large-batch path had removed the y²=x³+7 membership
// check (PERF-004 comment in shim_batch_verify.cpp). The small-batch fallback
// kept the check. An adversary supplying n>=8 verifications where one pubkey
// is an invalid-curve point (y²≠x³+7) would bypass the check in the large
// path. The large path's only guard was Point::from_affine(x,y) + is_infinity()
// which does NOT reject off-curve points that aren't the point at infinity.
//
// Tests:
//   BCK-1  small-batch (n=1): valid pubkey → pass
//   BCK-2  small-batch (n=1): invalid-curve pubkey → must return 0
//   BCK-3  large-batch (n=9): all valid pubkeys → pass
//   BCK-4  large-batch (n=9): one invalid-curve pubkey (slot 4) → must return 0
//   BCK-5  large-batch (n=9): invalid-curve pubkey in first slot → must return 0
//   BCK-6  large-batch (n=9): invalid-curve pubkey in last slot → must return 0
// ============================================================================

#include <cstdio>
#include <cstring>
#include <cassert>
#include <array>
#include <vector>

static int g_pass = 0, g_fail = 0;

#include "audit_check.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/ct/sign.hpp"

using namespace secp256k1::fast;

// ── helpers ──────────────────────────────────────────────────────────────────

static Scalar make_sk(int v) {
    std::array<uint8_t,32> b{}; b[31] = static_cast<uint8_t>(v);
    Scalar s{}; Scalar::parse_bytes_strict_nonzero(b.data(), s); return s;
}

struct SlotData {
    std::array<uint8_t,32>  msg;
    std::array<uint8_t,64>  sig;
    std::array<uint8_t,64>  pubkey_xy;  // secp256k1_pubkey opaque (x||y)
};

// Build a valid slot: sign msg with privkey v, store sig + valid pubkey.
static SlotData make_valid_slot(int v) {
    SlotData d{};
    d.msg[31] = static_cast<uint8_t>(v + 1);

    Scalar sk = make_sk(v);
    // CT sign to get a valid ECDSA signature
    auto sig_pair = secp256k1::ct::ecdsa_sign(d.msg.data(), sk);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));

    auto sig_bytes = sig_pair.to_compact();
    std::memcpy(d.sig.data(), sig_bytes.data(), 64);

    // Derive public key
    Point pk = secp256k1::ct::generator_mul(make_sk(v));
    // Store as affine x||y (secp256k1_pubkey layout)
    auto aff = pk.to_affine_xy();
    std::memcpy(d.pubkey_xy.data(),      aff.first.to_bytes().data(),  32);
    std::memcpy(d.pubkey_xy.data() + 32, aff.second.to_bytes().data(), 32);
    return d;
}

// Corrupt the y-coordinate of a pubkey so that y² ≠ x³+7.
static void corrupt_pubkey_y(std::array<uint8_t,64>& xy) {
    // Flip a bit in y — almost certainly produces an off-curve point.
    xy[32] ^= 0xFF;
    xy[33] ^= 0xAA;
}

// ── shim includes (batch verify is a shim-only API) ──────────────────────────
#ifdef SECP256K1_BUILD_SHIM

#include "secp256k1.h"
#include "secp256k1_batch.h"

static secp256k1_context* make_ctx() {
    return secp256k1_context_create(SECP256K1_CONTEXT_VERIFY);
}

// Run a batch verify call. Returns the shim's return value (0 or 1).
static int run_batch(secp256k1_context* ctx,
                     const std::vector<SlotData>& slots) {
    size_t n = slots.size();
    std::vector<const unsigned char*> sig_ptrs(n), msg_ptrs(n), pk_ptrs(n);
    std::vector<secp256k1_ecdsa_signature> sigs(n);
    std::vector<secp256k1_pubkey> pubkeys(n);

    for (size_t i = 0; i < n; ++i) {
        // Fill secp256k1_ecdsa_signature (compact 64 bytes in data[])
        std::memcpy(sigs[i].data,    slots[i].sig.data(),       32);  // r
        std::memcpy(sigs[i].data+32, slots[i].sig.data() + 32,  32);  // s
        // Fill secp256k1_pubkey (x||y in data[])
        std::memcpy(pubkeys[i].data,    slots[i].pubkey_xy.data(),      32);
        std::memcpy(pubkeys[i].data+32, slots[i].pubkey_xy.data() + 32, 32);
        sig_ptrs[i] = reinterpret_cast<const unsigned char*>(&sigs[i]);
        msg_ptrs[i] = slots[i].msg.data();
        pk_ptrs[i]  = reinterpret_cast<const unsigned char*>(&pubkeys[i]);
    }

    std::vector<const secp256k1_ecdsa_signature*> sptrs(n);
    std::vector<const secp256k1_pubkey*> pptrs(n);
    for (size_t i = 0; i < n; ++i) {
        sptrs[i] = &sigs[i];
        pptrs[i] = &pubkeys[i];
    }

    return secp256k1_ecdsa_verify_batch(ctx, sptrs.data(), msg_ptrs.data(),
                                         pptrs.data(), n);
}

// ── test functions ────────────────────────────────────────────────────────────

static void test_small_batch_valid() {
    secp256k1_context* ctx = make_ctx();
    std::vector<SlotData> slots = { make_valid_slot(1) };
    int r = run_batch(ctx, slots);
    CHECK(r == 1, "BCK-1: small-batch valid pubkey must pass");
    secp256k1_context_destroy(ctx);
}

static void test_small_batch_invalid_curve() {
    secp256k1_context* ctx = make_ctx();
    std::vector<SlotData> slots = { make_valid_slot(2) };
    corrupt_pubkey_y(slots[0].pubkey_xy);  // y² ≠ x³+7
    int r = run_batch(ctx, slots);
    CHECK(!r, "BCK-2: small-batch off-curve pubkey must not verify");
    secp256k1_context_destroy(ctx);
}

static void test_large_batch_all_valid() {
    secp256k1_context* ctx = make_ctx();
    std::vector<SlotData> slots;
    for (int i = 1; i <= 9; ++i) slots.push_back(make_valid_slot(i));
    int r = run_batch(ctx, slots);
    CHECK(r == 1, "BCK-3: large-batch all valid must pass");
    secp256k1_context_destroy(ctx);
}

static void test_large_batch_invalid_middle() {
    secp256k1_context* ctx = make_ctx();
    std::vector<SlotData> slots;
    for (int i = 1; i <= 9; ++i) slots.push_back(make_valid_slot(i));
    corrupt_pubkey_y(slots[4].pubkey_xy);  // slot 4 off-curve
    int r = run_batch(ctx, slots);
    CHECK(!r, "BCK-4: large-batch with off-curve pubkey at slot 4 must not verify");
    secp256k1_context_destroy(ctx);
}

static void test_large_batch_invalid_first() {
    secp256k1_context* ctx = make_ctx();
    std::vector<SlotData> slots;
    for (int i = 1; i <= 9; ++i) slots.push_back(make_valid_slot(i));
    corrupt_pubkey_y(slots[0].pubkey_xy);
    int r = run_batch(ctx, slots);
    CHECK(!r, "BCK-5: large-batch with off-curve pubkey at slot 0 must not verify");
    secp256k1_context_destroy(ctx);
}

static void test_large_batch_invalid_last() {
    secp256k1_context* ctx = make_ctx();
    std::vector<SlotData> slots;
    for (int i = 1; i <= 9; ++i) slots.push_back(make_valid_slot(i));
    corrupt_pubkey_y(slots[8].pubkey_xy);
    int r = run_batch(ctx, slots);
    CHECK(!r, "BCK-6: large-batch with off-curve pubkey at last slot must not verify");
    secp256k1_context_destroy(ctx);
}

#endif  // SECP256K1_BUILD_SHIM

// ── entry point ───────────────────────────────────────────────────────────────

int test_regression_ecdsa_batch_curve_check_run() {
    g_pass = 0; g_fail = 0;
    printf("\n  [ecdsa-batch-curve-check] CA-001: curve membership in small+large batch\n");

#ifdef SECP256K1_BUILD_SHIM
    test_small_batch_valid();
    test_small_batch_invalid_curve();
    test_large_batch_all_valid();
    test_large_batch_invalid_middle();
    test_large_batch_invalid_first();
    test_large_batch_invalid_last();
#else
    printf("  [SKIP] shim not linked — BCK-1..6 skipped\n");
    return 77;
#endif

    printf("  [ecdsa-batch-curve-check] %d passed, %d failed\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_ecdsa_batch_curve_check_run(); }
#endif
