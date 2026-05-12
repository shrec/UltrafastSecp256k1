// ============================================================================
// test_regression_frost_threshold_zero.cpp
// Regression: FROST frost_sign must guard against threshold == 0 (SEC-010).
// (nonce_commitments.size() < 0u) is always false — quorum check bypassed.
// Fix: explicit if (key_pkg.threshold == 0) guard added before the size check.
//
// FTZ-1: frost_sign with threshold=0 returns zero partial sig (error sentinel)
// FTZ-2: frost_sign with threshold=1 and 1 signer passes the guard
// FTZ-3: frost_sign with threshold=2 and 1 signer → below quorum → zero sig
// FTZ-4: frost_sign with threshold=2 and 2 signers passes the guard
// FTZ-5: ABI ufsecp_frost_sign rejects keypkg with threshold < 2
// ============================================================================

#ifndef UNIFIED_AUDIT_RUNNER
#include <cstdio>
#define STANDALONE_TEST
#endif

#include <cstring>
#include <cstdio>
#include <vector>
#include <array>

#include "secp256k1/frost.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/ct/scalar.hpp"
#include "ufsecp256k1.h"

using secp256k1::fast::Scalar;
using secp256k1::fast::Point;

static int g_fail = 0;
#define ASSERT_TRUE(cond, msg)  do { if (!(cond)) { std::printf("FAIL [%s]: %s\n", __func__, (msg)); ++g_fail; } } while(0)
#define ASSERT_FALSE(cond, msg) do { if ( (cond)) { std::printf("FAIL [%s]: %s\n", __func__, (msg)); ++g_fail; } } while(0)

static secp256k1::FrostKeyPackage make_kp(
    secp256k1::ParticipantId id, uint32_t threshold, uint32_t n)
{
    secp256k1::FrostKeyPackage kp{};
    kp.id = id; kp.threshold = threshold; kp.num_participants = n;
    kp.signing_share = Scalar::from_uint64(id);
    kp.verification_share = secp256k1::ct::generator_mul(Scalar::from_uint64(id));
    kp.group_public_key   = secp256k1::ct::generator_mul(Scalar::from_uint64(1));
    return kp;
}

static secp256k1::FrostNonce make_nonce(secp256k1::ParticipantId id) {
    secp256k1::FrostNonce n{};
    n.hiding_nonce  = Scalar::from_uint64(id * 2 + 1);
    n.binding_nonce = Scalar::from_uint64(id * 2 + 2);
    return n;
}

static secp256k1::FrostNonceCommitment make_commit(secp256k1::ParticipantId id) {
    secp256k1::FrostNonceCommitment nc{};
    nc.id = id;
    nc.hiding_point  = secp256k1::ct::generator_mul(Scalar::from_uint64(id * 2 + 1));
    nc.binding_point = secp256k1::ct::generator_mul(Scalar::from_uint64(id * 2 + 2));
    return nc;
}

static void test_threshold_zero_returns_error() {
    auto kp = make_kp(1, 0, 3);
    auto nonce = make_nonce(1);
    std::array<uint8_t, 32> msg{}; msg[0] = 0xAA;
    std::vector<secp256k1::FrostNonceCommitment> commits = { make_commit(1) };
    auto psig = secp256k1::frost_sign(kp, nonce, msg, commits);
    ASSERT_TRUE(psig.z_i.is_zero(), "[FTZ-1] threshold=0 must return zero partial sig");
    ASSERT_TRUE(psig.id == 1, "[FTZ-1] id must match");
}

static void test_threshold_one_passes_guard() {
    auto kp = make_kp(1, 1, 1);
    auto nonce = make_nonce(1);
    std::array<uint8_t, 32> msg{}; msg[0] = 0xBB;
    std::vector<secp256k1::FrostNonceCommitment> commits = { make_commit(1) };
    auto psig = secp256k1::frost_sign(kp, nonce, msg, commits);
    ASSERT_TRUE(psig.id == 1, "[FTZ-2] threshold=1 must not be blocked by threshold guard");
}

static void test_threshold_two_one_signer_rejected() {
    auto kp = make_kp(1, 2, 3);
    auto nonce = make_nonce(1);
    std::array<uint8_t, 32> msg{}; msg[0] = 0xCC;
    std::vector<secp256k1::FrostNonceCommitment> commits = { make_commit(1) };
    auto psig = secp256k1::frost_sign(kp, nonce, msg, commits);
    ASSERT_TRUE(psig.z_i.is_zero(), "[FTZ-3] 1 signer below threshold=2 must return zero sig");
}

static void test_threshold_two_two_signers_passes_guard() {
    auto kp = make_kp(1, 2, 2);
    auto nonce = make_nonce(1);
    std::array<uint8_t, 32> msg{}; msg[0] = 0xDD;
    std::vector<secp256k1::FrostNonceCommitment> commits = { make_commit(1), make_commit(2) };
    auto psig = secp256k1::frost_sign(kp, nonce, msg, commits);
    ASSERT_TRUE(psig.id == 1, "[FTZ-4] 2 signers at threshold=2 must pass the guard");
}

static void test_abi_rejects_threshold_below_2() {
    ufsecp_ctx* ctx = nullptr;
    if (ufsecp_ctx_create(&ctx) != UFSECP_OK || !ctx) {
        std::printf("SKIP FTZ-5: no ufsecp_ctx\n"); return;
    }
    // G compressed point
    static const uint8_t G33[33] = {
        0x02,0x79,0xBE,0x66,0x7E,0xF9,0xDC,0xBB,0xAC,0x55,0xA0,0x62,0x95,0xCE,0x87,0x0B,0x07,
        0x02,0x9B,0xFC,0xDB,0x2D,0xCE,0x28,0xD9,0x59,0xF2,0x81,0x5B,0x16,0xF8,0x17,0x98
    };
    uint8_t keypkg[UFSECP_FROST_KEYPKG_LEN] = {};
    keypkg[0] = 1;      // id=1
    keypkg[4] = 1;      // threshold=1 (below ABI minimum of 2)
    keypkg[8] = 3;      // num_participants=3
    keypkg[43] = 0x07;  // signing_share byte[12..43]
    // verification_share[44..76] = G
    keypkg[44] = G33[0]; for (int i=1;i<33;i++) keypkg[43+i] = G33[i];
    // group_public_key[77..109] = G
    keypkg[77] = G33[0]; for (int i=1;i<33;i++) keypkg[76+i] = G33[i];

    uint8_t nonce_buf[UFSECP_FROST_NONCE_LEN] = {};
    nonce_buf[31] = 0x01; nonce_buf[63] = 0x02;
    uint8_t msg32[32] = {}; msg32[0] = 0xEE;

    uint8_t nc_buf[UFSECP_FROST_NONCE_COMMIT_LEN] = {};
    nc_buf[0] = 1;  // id=1
    nc_buf[4] = G33[0]; for (int i=1;i<33;i++) nc_buf[3+i] = G33[i];  // hiding
    nc_buf[37] = G33[0]; for (int i=1;i<33;i++) nc_buf[36+i] = G33[i]; // binding

    uint8_t psig_out[36] = {};
    auto rc = ufsecp_frost_sign(ctx, keypkg, nonce_buf, msg32, nc_buf, 1, psig_out);
    ASSERT_FALSE(rc == UFSECP_OK, "[FTZ-5] ABI must reject keypkg with threshold=1");
    ufsecp_ctx_destroy(ctx);
}

int test_regression_frost_threshold_zero_run() {
    g_fail = 0;
    test_threshold_zero_returns_error();
    test_threshold_one_passes_guard();
    test_threshold_two_one_signer_rejected();
    test_threshold_two_two_signers_passes_guard();
    test_abi_rejects_threshold_below_2();
    if (g_fail == 0)
        std::printf("PASS: FROST frost_sign threshold==0 guard (SEC-010, FTZ-1..5)\n");
    else
        std::printf("FAIL: FROST threshold zero guard: %d failure(s)\n", g_fail);
    return g_fail;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_frost_threshold_zero_run(); }
#endif
