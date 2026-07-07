// ============================================================================
// test_shim_batch_cancel.cpp -- external cancellation token on batch verify
// ============================================================================
// Covers the trailing `const ufsecp_cancel_token* cancel` parameter added to
// every shim batch-verify function (secp256k1_{ecdsa,schnorrsig}_verify_batch
// [_mt|_results]):
//   * cancel == NULL  -> identical result to the original path (parity), for
//                        all-valid and invalid batches, all-or-nothing + per-row.
//   * immediate cancel -> returns 0 (fail-closed) and the callback is polled.
//   * mid-batch cancel -> returns 0; callback polled more than once (a chunk
//                         boundary was crossed); _results leaves unreached rows 0.
//   * throwing callback -> returns 0 (fail-closed), never 1.
//   * a NON-tripping token with a small check_interval forces chunked dispatch
//                         yet still yields the correct verdict (chunk correctness).
//   * n == 0 with a non-NULL token -> 1 (vacuously valid).
//
// Cancellation is on a verify path (public data only): variable-time, no CT
// impact. A returned 0 under cancellation is "verdict unknown, discard" — the
// caller disambiguates via its own token state.
// ============================================================================

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <array>
#include <vector>
#include <stdexcept>

#include "../include/secp256k1.h"
#include "../include/secp256k1_schnorrsig.h"
#include "../include/secp256k1_extrakeys.h"
#include "../include/secp256k1_batch.h"

static int g_fail = 0;

static void check(bool cond, const char* msg) {
    if (cond) { std::printf("PASS %s\n", msg); }
    else      { ++g_fail; std::printf("FAIL %s\n", msg); }
}

static void illegal_cb(const char* /*msg*/, void* /*data*/) { /* non-aborting */ }

static secp256k1_context* make_ctx() {
    secp256k1_context* ctx = secp256k1_context_create(
        SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    secp256k1_context_set_illegal_callback(ctx, illegal_cb, nullptr);
    return ctx;
}

// -- Cancellation callback driven by a small state struct --------------------
struct CancelState {
    int  polls        = 0;
    int  cancel_after = -1;   // < 0 => never cancel; otherwise cancel once polls > cancel_after
    bool do_throw     = false;
};

static int cancel_cb(const void* user) {
    auto* st = const_cast<CancelState*>(static_cast<const CancelState*>(user));
    ++st->polls;
    if (st->do_throw) throw std::runtime_error("cancel callback throws");
    if (st->cancel_after >= 0 && st->polls > st->cancel_after) return 1;
    return 0;
}

static ufsecp_cancel_token make_token(CancelState* st, uint32_t check_interval) {
    ufsecp_cancel_token t;
    t.is_cancelled   = cancel_cb;
    t.user           = st;
    t.check_interval = check_interval;
    return t;
}

static const size_t N = 5000;  // > engine chunk so the MT path spawns threads

int main() {
    secp256k1_context* ctx = make_ctx();

    std::vector<secp256k1_ecdsa_signature> ec_sigs(N);
    std::vector<secp256k1_pubkey>          ec_pubs(N);
    std::vector<std::array<uint8_t, 32>>   msgs(N);
    std::vector<std::array<uint8_t, 64>>   sc_sigs(N);
    std::vector<secp256k1_xonly_pubkey>    sc_pubs(N);

    for (size_t i = 0; i < N; ++i) {
        uint8_t sk[32] = {};
        sk[28] = static_cast<uint8_t>((i + 1) >> 24);
        sk[29] = static_cast<uint8_t>((i + 1) >> 16);
        sk[30] = static_cast<uint8_t>((i + 1) >> 8);
        sk[31] = static_cast<uint8_t>((i + 1) & 0xFF);

        msgs[i].fill(0);
        msgs[i][0]  = static_cast<uint8_t>(i & 0xFF);
        msgs[i][1]  = static_cast<uint8_t>((i >> 8) & 0xFF);
        msgs[i][31] = 0xA5;

        if (secp256k1_ec_pubkey_create(ctx, &ec_pubs[i], sk) != 1) { check(false, "ec_pubkey_create"); return 1; }
        if (secp256k1_ecdsa_sign(ctx, &ec_sigs[i], msgs[i].data(), sk, nullptr, nullptr) != 1) { check(false, "ecdsa_sign"); return 1; }

        secp256k1_keypair kp;
        if (secp256k1_keypair_create(ctx, &kp, sk) != 1) { check(false, "keypair_create"); return 1; }
        if (secp256k1_keypair_xonly_pub(ctx, &sc_pubs[i], nullptr, &kp) != 1) { check(false, "keypair_xonly_pub"); return 1; }
        uint8_t aux[32] = {};
        if (secp256k1_schnorrsig_sign32(ctx, sc_sigs[i].data(), msgs[i].data(), &kp, aux) != 1) { check(false, "schnorrsig_sign32"); return 1; }
    }

    std::vector<const secp256k1_ecdsa_signature*> ec_sigp(N);
    std::vector<const secp256k1_pubkey*>          ec_pubp(N);
    std::vector<const unsigned char*>             msgp(N);
    std::vector<const unsigned char*>             sc_sigp(N);
    std::vector<const secp256k1_xonly_pubkey*>    sc_pubp(N);
    for (size_t i = 0; i < N; ++i) {
        ec_sigp[i] = &ec_sigs[i];
        ec_pubp[i] = &ec_pubs[i];
        msgp[i]    = msgs[i].data();
        sc_sigp[i] = sc_sigs[i].data();
        sc_pubp[i] = &sc_pubs[i];
    }

    // -- 1) NULL token parity: same verdict as the original path ----------------
    check(secp256k1_ecdsa_verify_batch_mt(ctx, ec_sigp.data(), msgp.data(), ec_pubp.data(), N, 0, nullptr) == 1,
          "ecdsa_mt NULL token all-valid -> 1");
    check(secp256k1_schnorrsig_verify_batch_mt(ctx, sc_sigp.data(), msgp.data(), 32, sc_pubp.data(), N, 0, nullptr) == 1,
          "schnorr_mt NULL token all-valid -> 1");
    // Base functions with C++ default-arg (cancel omitted entirely).
    check(secp256k1_ecdsa_verify_batch(ctx, ec_sigp.data(), msgp.data(), ec_pubp.data(), N) == 1,
          "ecdsa_batch default-arg all-valid -> 1");

    // -- 2) Non-tripping token forces chunking, verdict still correct -----------
    {
        CancelState st; st.cancel_after = -1;               // never cancels
        ufsecp_cancel_token tok = make_token(&st, 64);      // small interval -> many chunks
        check(secp256k1_ecdsa_verify_batch_mt(ctx, ec_sigp.data(), msgp.data(), ec_pubp.data(), N, 0, &tok) == 1,
              "ecdsa_mt chunked (non-tripping) all-valid -> 1");
        check(st.polls > 1, "ecdsa_mt chunked token polled across boundaries");
        CancelState st2; st2.cancel_after = -1;
        ufsecp_cancel_token tok2 = make_token(&st2, 64);
        check(secp256k1_schnorrsig_verify_batch_mt(ctx, sc_sigp.data(), msgp.data(), 32, sc_pubp.data(), N, 0, &tok2) == 1,
              "schnorr_mt chunked (non-tripping) all-valid -> 1");
    }

    // -- 3) Immediate cancel -> 0 (fail-closed), callback polled ----------------
    {
        CancelState st; st.cancel_after = 0;                 // cancels on first poll
        ufsecp_cancel_token tok = make_token(&st, 64);
        check(secp256k1_ecdsa_verify_batch_mt(ctx, ec_sigp.data(), msgp.data(), ec_pubp.data(), N, 0, &tok) == 0,
              "ecdsa_mt immediate cancel -> 0");
        check(st.polls >= 1, "ecdsa_mt immediate cancel polled the callback");
        CancelState st2; st2.cancel_after = 0;
        ufsecp_cancel_token tok2 = make_token(&st2, 64);
        check(secp256k1_schnorrsig_verify_batch_mt(ctx, sc_sigp.data(), msgp.data(), 32, sc_pubp.data(), N, 0, &tok2) == 0,
              "schnorr_mt immediate cancel -> 0");
    }

    // -- 4) Mid-batch cancel -> 0, polled more than once ------------------------
    {
        CancelState st; st.cancel_after = 1;                 // cancels on the 2nd poll
        ufsecp_cancel_token tok = make_token(&st, 64);
        check(secp256k1_ecdsa_verify_batch_mt(ctx, ec_sigp.data(), msgp.data(), ec_pubp.data(), N, 0, &tok) == 0,
              "ecdsa_mt mid-batch cancel -> 0");
        check(st.polls >= 2, "ecdsa_mt mid-batch crossed a chunk boundary");
    }

    // -- 5) Throwing callback -> 0 (fail-closed), never 1 -----------------------
    {
        CancelState st; st.do_throw = true;
        ufsecp_cancel_token tok = make_token(&st, 64);
        check(secp256k1_ecdsa_verify_batch_mt(ctx, ec_sigp.data(), msgp.data(), ec_pubp.data(), N, 0, &tok) == 0,
              "ecdsa_mt throwing callback -> 0 (fail-closed)");
        CancelState st2; st2.do_throw = true;
        ufsecp_cancel_token tok2 = make_token(&st2, 64);
        check(secp256k1_schnorrsig_verify_batch_mt(ctx, sc_sigp.data(), msgp.data(), 32, sc_pubp.data(), N, 0, &tok2) == 0,
              "schnorr_mt throwing callback -> 0 (fail-closed)");
    }

    // -- 6) _results: non-tripping chunked token pinpoints invalid rows ---------
    const size_t bad[] = {0, 1234, 4096, N - 1};
    std::vector<std::array<uint8_t, 32>> msgs_bad = msgs;
    std::vector<const unsigned char*>    msgp_bad(N);
    for (size_t i = 0; i < N; ++i) msgp_bad[i] = msgs_bad[i].data();
    for (size_t b : bad) msgs_bad[b][0] ^= 0xFF;
    auto is_bad = [&](size_t i) { for (size_t b : bad) if (b == i) return true; return false; };
    {
        CancelState st; st.cancel_after = -1;
        ufsecp_cancel_token tok = make_token(&st, 64);
        std::vector<int> res(N, -1);
        int rc = secp256k1_ecdsa_verify_batch_results(ctx, ec_sigp.data(), msgp_bad.data(),
                                                      ec_pubp.data(), N, 0, res.data(), &tok);
        check(rc == 0, "ecdsa_results chunked returns 0 when rows invalid");
        bool ok = true;
        for (size_t i = 0; i < N; ++i) { int want = is_bad(i) ? 0 : 1; if (res[i] != want) { ok = false; break; } }
        check(ok, "ecdsa_results chunked per-row verdict correct");
    }

    // -- 7) _results: mid cancel leaves unreached rows 0 ------------------------
    {
        CancelState st; st.cancel_after = 1;                 // cancel after 1st chunk
        ufsecp_cancel_token tok = make_token(&st, 64);       // chunk = 64 rows
        std::vector<int> res(N, 7);                          // sentinel != 0/1
        int rc = secp256k1_ecdsa_verify_batch_results(ctx, ec_sigp.data(), msgp.data(),
                                                      ec_pubp.data(), N, 0, res.data(), &tok);
        check(rc == 0, "ecdsa_results mid cancel -> 0");
        // The tail (well past the first chunk) must be left at 0, not the sentinel.
        check(res[N - 1] == 0 && res[N - 100] == 0, "ecdsa_results unreached rows pre-zeroed to 0");
    }

    // -- 8) n == 0 with a non-NULL token is vacuously valid --------------------
    {
        CancelState st; st.cancel_after = 0;                 // would cancel if consulted
        ufsecp_cancel_token tok = make_token(&st, 64);
        check(secp256k1_ecdsa_verify_batch_mt(ctx, ec_sigp.data(), msgp.data(), ec_pubp.data(), 0, 0, &tok) == 1,
              "ecdsa_mt n=0 with token -> 1 (vacuous)");
    }

    secp256k1_context_destroy(ctx);
    std::printf("\n%s (%d failures)\n", g_fail == 0 ? "ALL PASSED" : "FAILURES", g_fail);
    return g_fail == 0 ? 0 : 1;
}
