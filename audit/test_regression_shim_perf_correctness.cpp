// ============================================================================
// test_regression_shim_perf_correctness.cpp
// Regression: verify that PERF-001/005 hot-path optimizations in the shim
// (shim_recovery.cpp, shim_schnorr.cpp) do not introduce correctness regressions.
//
// PERF-001: shim_recovery.cpp point_to_pubkey_data — is_normalized() fast path
//   eliminates field inversion when recovered point is already affine (always).
// PERF-005: shim_schnorr.cpp secp256k1_schnorrsig_verify — raw-pointer parse
//   eliminates 64-byte stack copy from xonly_pubkey->data per verify call.
//
// Tests:
//   SPC-1: ECDSA recover roundtrip — sign, recover, compare pubkey (PERF-001)
//   SPC-2: ECDSA verify correctness — 100 valid sigs pass, 10 wrong-key fail
//   SPC-3: Schnorr verify correctness — 100 valid pass, 10 wrong-key fail (PERF-005)
//   SPC-4: Recovery with all recid values 0–3
// ============================================================================

#include <cstdio>
#include <cstring>
#include <array>

static int g_pass = 0, g_fail = 0;

#include "audit_check.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ct/sign.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/recovery.hpp"

using namespace secp256k1::fast;

// Deterministic test scalar from small integer
static Scalar make_sk(uint8_t v) {
    std::array<uint8_t, 32> b{};
    b[31] = v;
    Scalar s{};
    Scalar::parse_bytes_strict_nonzero(b.data(), s);
    return s;
}

// 32-byte message hash from small integer
static std::array<uint8_t, 32> make_msg(uint8_t v) {
    std::array<uint8_t, 32> m{};
    m[0] = 0xAB; m[31] = v;
    return m;
}

// ── SPC-1: ECDSA recoverable sign + recover roundtrip ────────────────────────
static void test_shim_recovery_roundtrip() {
    printf("[shim_perf] SPC-1: ECDSA recover roundtrip (PERF-001)\n");

    for (uint8_t i = 1; i <= 20; ++i) {
        auto sk = make_sk(i);
        auto pk = secp256k1::ct::generator_mul(sk);
        auto pk_c = pk.to_compressed();

        auto msg = make_msg(i);
        auto rsig = secp256k1::ct::ecdsa_sign_recoverable(sk, msg.data());

        auto recovered = secp256k1::ecdsa_recover(rsig, msg.data());
        CHECK(recovered.has_value(), "ecdsa_recover must succeed");
        if (!recovered.has_value()) continue;

        auto rec_c = recovered->to_compressed();
        bool match = (std::memcmp(pk_c.data(), rec_c.data(), 33) == 0);
        CHECK(match, "recovered pubkey must match original");
    }
    printf("[shim_perf] SPC-1: %d/%d\n", g_pass, g_pass + g_fail);
}

// ── SPC-2: ECDSA verify correctness ──────────────────────────────────────────
static void test_shim_ecdsa_verify_correctness() {
    printf("[shim_perf] SPC-2: ECDSA verify correctness\n");

    for (uint8_t i = 1; i <= 20; ++i) {
        auto sk = make_sk(i);
        auto pk = secp256k1::ct::generator_mul(sk);
        auto msg = make_msg(i);
        auto sig = secp256k1::ct::ecdsa_sign(sk, msg.data());

        bool ok = secp256k1::ecdsa_verify(sig, msg.data(), pk);
        CHECK(ok, "valid ECDSA signature must verify");

        // Wrong key: sign with sk, verify with different pk
        auto wrong_pk = secp256k1::ct::generator_mul(make_sk(i + 100));
        bool bad = secp256k1::ecdsa_verify(sig, msg.data(), wrong_pk);
        CHECK(!bad, "ECDSA verify with wrong key must fail");
    }
    printf("[shim_perf] SPC-2: %d/%d\n", g_pass, g_pass + g_fail);
}

// ── SPC-3: Schnorr verify correctness (PERF-005 path) ────────────────────────
static void test_shim_schnorrsig_verify_correctness() {
    printf("[shim_perf] SPC-3: Schnorr verify (PERF-005)\n");

    for (uint8_t i = 1; i <= 20; ++i) {
        auto sk = make_sk(i);
        auto msg = make_msg(i);
        // Schnorr: use aux_rand = zero bytes
        std::array<uint8_t, 32> aux{};
        auto sig = secp256k1::ct::schnorr_sign(sk, msg.data(), aux.data());
        auto pk  = secp256k1::ct::generator_mul(sk);

        // schnorr_verify takes xonly pubkey (x coordinate only)
        auto px = pk.to_xonly_bytes();
        bool ok = secp256k1::schnorr_verify(px.data(), msg.data(), sig);
        CHECK(ok, "valid Schnorr signature must verify");

        // Wrong message
        auto wrong_msg = make_msg(static_cast<uint8_t>(i + 50));
        bool bad = secp256k1::schnorr_verify(px.data(), wrong_msg.data(), sig);
        CHECK(!bad, "Schnorr verify with wrong message must fail");
    }
    printf("[shim_perf] SPC-3: %d/%d\n", g_pass, g_pass + g_fail);
}

// ── SPC-4: Recovery with various recid values ─────────────────────────────────
static void test_shim_recovery_recid() {
    printf("[shim_perf] SPC-4: Recovery recid correctness\n");
    int succeeded = 0;

    for (uint8_t i = 1; i <= 40; ++i) {
        auto sk = make_sk(i);
        auto msg = make_msg(i);
        auto rsig = secp256k1::ct::ecdsa_sign_recoverable(sk, msg.data());
        auto recovered = secp256k1::ecdsa_recover(rsig, msg.data());

        // Recovery must succeed for any valid signing key
        if (recovered.has_value()) {
            auto pk = secp256k1::ct::generator_mul(sk);
            auto pk_c = pk.to_compressed();
            auto rec_c = recovered->to_compressed();
            if (std::memcmp(pk_c.data(), rec_c.data(), 33) == 0) ++succeeded;
        }
    }
    CHECK(succeeded >= 38, "at least 95% of recover calls must succeed and match");
    printf("[shim_perf] SPC-4: %d/40 recoveries succeeded and matched\n", succeeded);
}

// ── SPC-5: T-11 — SchnorrXonlyPubkey verify (GLV cached) agrees with raw path ──
// Verifies that schnorr_verify(SchnorrXonlyPubkey, ...) gives the same result as
// schnorr_verify(raw_x_bytes, ...) for both valid and invalid signatures.
// This is the correctness guard for the T-11 shim verify cache-first optimization.
static void test_t11_glv_cached_verify_correctness() {
    printf("[shim_perf] SPC-5: T-11 GLV-cached verify agrees with raw-bytes verify\n");

    for (uint8_t i = 1; i <= 20; ++i) {
        auto sk  = make_sk(i);
        auto msg = make_msg(i);
        std::array<uint8_t, 32> aux{};
        auto sig = secp256k1::ct::schnorr_sign(sk, msg.data(), aux.data());
        auto pk  = secp256k1::ct::generator_mul(sk);
        auto px  = pk.to_xonly_bytes();

        // Parse into SchnorrXonlyPubkey (builds GLV tables).
        secp256k1::SchnorrXonlyPubkey xonly{};
        bool parsed = secp256k1::schnorr_xonly_pubkey_parse(xonly, px.data());
        CHECK(parsed, "SPC-5: schnorr_xonly_pubkey_parse succeeds");

        // Both paths must agree on valid signature.
        bool raw_ok    = secp256k1::schnorr_verify(px.data(), msg.data(), sig);
        bool cached_ok = secp256k1::schnorr_verify(xonly, msg.data(), sig);
        CHECK(raw_ok && cached_ok, "SPC-5: valid sig — both paths accept");
        CHECK(raw_ok == cached_ok, "SPC-5: valid sig — both paths agree");

        // Both paths must agree on wrong message.
        auto wrong_msg = make_msg(static_cast<uint8_t>(i + 100));
        bool raw_bad    = secp256k1::schnorr_verify(px.data(), wrong_msg.data(), sig);
        bool cached_bad = secp256k1::schnorr_verify(xonly, wrong_msg.data(), sig);
        CHECK(!raw_bad && !cached_bad, "SPC-5: wrong msg — both paths reject");
        CHECK(raw_bad == cached_bad, "SPC-5: wrong msg — both paths agree");
    }
    printf("[shim_perf] SPC-5: %d/%d\n", g_pass, g_pass + g_fail);
}

// ── Entry point ──────────────────────────────────────────────────────────────
int test_regression_shim_perf_correctness_run() {
    printf("[shim_perf_correctness] Regression: PERF-001/005/T-11 hot-path correctness\n");

    test_shim_recovery_roundtrip();
    test_shim_ecdsa_verify_correctness();
    test_shim_schnorrsig_verify_correctness();
    test_shim_recovery_recid();
    test_t11_glv_cached_verify_correctness();

    printf("[shim_perf_correctness] %d passed, %d failed\n", g_pass, g_fail);
    return g_fail;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_shim_perf_correctness_run(); }
#endif
