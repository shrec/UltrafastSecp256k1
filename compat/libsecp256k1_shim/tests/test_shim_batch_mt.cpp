// ============================================================================
// test_shim_batch_mt.cpp -- multi-threaded + per-row batch verify (shim)
// ============================================================================
// Regression coverage for the bridge-free integration standard:
//   * secp256k1_ecdsa_verify_batch_mt / secp256k1_schnorrsig_verify_batch_mt
//     return the SAME boolean ("all valid") as single verify for every thread
//     count {0,1,2,8,64} (0=auto, 1=serial, N=cap) -- threads are a pure
//     throughput change with no effect on the result.
//   * secp256k1_ecdsa_verify_batch_results / *_schnorrsig_verify_batch_results
//     write a per-row verdict: 1 for valid rows, 0 for the injected-invalid
//     rows, and return 0 overall when any row is invalid.
//   * n == 0 is vacuously valid; small-n (< 8) parity holds.
//   * max_threads == 1 (caller runs its own pool) yields identical results.
//
// Batch size is chosen > the engine's 4096-row chunk so the MT path actually
// spawns worker threads (n_chunks > 1).
// ============================================================================

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <array>
#include <vector>

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

static const size_t N = 5000;  // > 4096 chunk => MT path spawns threads
static const size_t kThreadCounts[] = {0, 1, 2, 8, 64};

int main() {
    secp256k1_context* ctx = make_ctx();

    // -- Build N valid ECDSA + Schnorr signatures --------------------------------
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
        msgs[i][0] = static_cast<uint8_t>(i & 0xFF);
        msgs[i][1] = static_cast<uint8_t>((i >> 8) & 0xFF);
        msgs[i][31] = 0xA5;

        // ECDSA
        if (secp256k1_ec_pubkey_create(ctx, &ec_pubs[i], sk) != 1) {
            check(false, "ec_pubkey_create"); return 1;
        }
        if (secp256k1_ecdsa_sign(ctx, &ec_sigs[i], msgs[i].data(), sk, nullptr, nullptr) != 1) {
            check(false, "ecdsa_sign"); return 1;
        }

        // Schnorr
        secp256k1_keypair kp;
        if (secp256k1_keypair_create(ctx, &kp, sk) != 1) {
            check(false, "keypair_create"); return 1;
        }
        if (secp256k1_keypair_xonly_pub(ctx, &sc_pubs[i], nullptr, &kp) != 1) {
            check(false, "keypair_xonly_pub"); return 1;
        }
        uint8_t aux[32] = {};
        if (secp256k1_schnorrsig_sign32(ctx, sc_sigs[i].data(), msgs[i].data(), &kp, aux) != 1) {
            check(false, "schnorrsig_sign32"); return 1;
        }
    }

    // -- Pointer arrays for the batch API ----------------------------------------
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

    // -- 1) MT == single across thread counts (all valid) ------------------------
    for (size_t t : kThreadCounts) {
        char lbl[64];
        std::snprintf(lbl, sizeof(lbl), "ecdsa_batch_mt all-valid (threads=%zu)", t);
        check(secp256k1_ecdsa_verify_batch_mt(ctx, ec_sigp.data(), msgp.data(),
                                              ec_pubp.data(), N, t) == 1, lbl);
        std::snprintf(lbl, sizeof(lbl), "schnorr_batch_mt all-valid (threads=%zu)", t);
        check(secp256k1_schnorrsig_verify_batch_mt(ctx, sc_sigp.data(), msgp.data(), 32,
                                                   sc_pubp.data(), N, t) == 1, lbl);
    }

    // Legacy (auto) symbols agree.
    check(secp256k1_ecdsa_verify_batch(ctx, ec_sigp.data(), msgp.data(),
                                       ec_pubp.data(), N) == 1, "ecdsa_batch (legacy auto)");
    check(secp256k1_schnorrsig_verify_batch(ctx, sc_sigp.data(), msgp.data(), 32,
                                            sc_pubp.data(), N) == 1, "schnorr_batch (legacy auto)");

    // -- 2) Per-row results pinpoint injected-invalid rows -----------------------
    // Corrupt the message for a few rows: signatures still parse but verification
    // fails for exactly those rows (exercises the identify-invalid path).
    const size_t bad[] = {0, 1234, 4096, N - 1};
    std::vector<std::array<uint8_t, 32>> msgs_bad = msgs;
    std::vector<const unsigned char*>    msgp_bad(N);
    for (size_t i = 0; i < N; ++i) msgp_bad[i] = msgs_bad[i].data();
    for (size_t b : bad) { msgs_bad[b][0] ^= 0xFF; }

    auto is_bad = [&](size_t i) {
        for (size_t b : bad) if (b == i) return true;
        return false;
    };

    {
        std::vector<int> res(N, -1);
        int rc = secp256k1_ecdsa_verify_batch_results(ctx, ec_sigp.data(), msgp_bad.data(),
                                                       ec_pubp.data(), N, 0, res.data());
        check(rc == 0, "ecdsa_batch_results returns 0 when rows invalid");
        bool ok = true;
        for (size_t i = 0; i < N; ++i) {
            int want = is_bad(i) ? 0 : 1;
            if (res[i] != want) { ok = false; break; }
        }
        check(ok, "ecdsa_batch_results per-row verdict correct");
    }
    {
        std::vector<int> res(N, -1);
        int rc = secp256k1_schnorrsig_verify_batch_results(ctx, sc_sigp.data(), msgp_bad.data(), 32,
                                                            sc_pubp.data(), N, 0, res.data());
        check(rc == 0, "schnorr_batch_results returns 0 when rows invalid");
        bool ok = true;
        for (size_t i = 0; i < N; ++i) {
            int want = is_bad(i) ? 0 : 1;
            if (res[i] != want) { ok = false; break; }
        }
        check(ok, "schnorr_batch_results per-row verdict correct");
    }

    // All-or-nothing MT must also report failure on the corrupted batch.
    check(secp256k1_ecdsa_verify_batch_mt(ctx, ec_sigp.data(), msgp_bad.data(),
                                          ec_pubp.data(), N, 0) == 0,
          "ecdsa_batch_mt detects invalid batch");
    check(secp256k1_schnorrsig_verify_batch_mt(ctx, sc_sigp.data(), msgp_bad.data(), 32,
                                               sc_pubp.data(), N, 0) == 0,
          "schnorr_batch_mt detects invalid batch");

    // Per-row results all-valid: every verdict 1, return 1.
    {
        std::vector<int> res(N, -1);
        int rc = secp256k1_ecdsa_verify_batch_results(ctx, ec_sigp.data(), msgp.data(),
                                                       ec_pubp.data(), N, 1 /*serial*/, res.data());
        bool all1 = (rc == 1);
        for (size_t i = 0; i < N && all1; ++i) all1 = (res[i] == 1);
        check(all1, "ecdsa_batch_results all-valid -> all 1 (max_threads=1)");
    }

    // -- 3) n == 0 vacuously valid -----------------------------------------------
    check(secp256k1_ecdsa_verify_batch_mt(ctx, ec_sigp.data(), msgp.data(),
                                          ec_pubp.data(), 0, 0) == 1, "ecdsa_batch_mt n=0");
    {
        std::vector<int> res(1, -1);
        check(secp256k1_ecdsa_verify_batch_results(ctx, ec_sigp.data(), msgp.data(),
                                                   ec_pubp.data(), 0, 0, res.data()) == 1,
              "ecdsa_batch_results n=0");
    }

    // -- 4) small-n (< 8) parity -------------------------------------------------
    check(secp256k1_ecdsa_verify_batch_mt(ctx, ec_sigp.data(), msgp.data(),
                                          ec_pubp.data(), 4, 0) == 1, "ecdsa_batch_mt small-n valid");
    {
        std::vector<int> res(4, -1);
        int rc = secp256k1_ecdsa_verify_batch_results(ctx, ec_sigp.data(), msgp_bad.data(),
                                                       ec_pubp.data(), 4, 0, res.data());
        // index 0 was corrupted; 1..3 valid.
        check(rc == 0 && res[0] == 0 && res[1] == 1 && res[2] == 1 && res[3] == 1,
              "ecdsa_batch_results small-n pinpoints invalid");
    }

    secp256k1_context_destroy(ctx);

    std::printf("\n%s (%d failures)\n", g_fail == 0 ? "ALL PASSED" : "FAILURES", g_fail);
    return g_fail == 0 ? 0 : 1;
}
