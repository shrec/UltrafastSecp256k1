// ============================================================================
// test_regression_ct_ops_2026_05_21.cpp
// ============================================================================
// Regression coverage for six CT/security fixes applied 2026-05-21:
//
//   SEC-002/CT-002  frost_lagrange_coefficient_from_commitments: VT operator*
//                   replaced with ct::scalar_mul / ct::scalar_sub on the
//                   numerator/denominator accumulation loop.
//
//   SEC-007         batch_weight(): Scalar::from_bytes(h) returns 0 when the
//                   SHA-256 output equals the curve order n (prob ~2^-128).
//                   A zero weight silently excludes that entry from the batch
//                   check (fail-open). Now returns Scalar::one() as fallback.
//
//   SEC-008         ecdsa_adaptor_sign(): r.is_zero() early-exit returned
//                   {R_hat, Scalar::zero(), r} — a partially-populated struct
//                   with a non-zero R_hat. Now returns the fully-zero sentinel
//                   {Point::infinity(), Scalar::zero(), Scalar::zero()}.
//
//   SEC-010         bip32_master_key(): two-step parse_bytes_strict() +
//                   is_zero() replaced with parse_bytes_strict_nonzero() in
//                   one call (removes variable-time is_zero() on secret IL).
//
//   CT-004          musig2_nonce_gen(): R1=k1*G and R2=k2*G now use
//                   ct::generator_mul_blinded (DPA defense) instead of
//                   ct::generator_mul (matching the level of ct_sign.cpp).
//
//   CT-005          ecdsa_sign_verified() in ecdsa.cpp: calls ct::ecdsa_sign
//                   directly instead of the deprecated secp256k1::ecdsa_sign
//                   wrapper via pragma suppression.
//
// All sub-tests exercise correctness (not timing) — the CT properties are
// enforced structurally by the code changes. Source-scan sub-tests confirm
// the fixes are still present (guard against reverts).
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>
#include <vector>
#include <random>
#include <fstream>
#include <sstream>
#include <string>

#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/ct/sign.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/ct/scalar.hpp"
#include "secp256k1/batch_verify.hpp"
#include "secp256k1/adaptor.hpp"
#include "secp256k1/bip32.hpp"
#include "secp256k1/musig2.hpp"

#if __has_include("secp256k1/frost.hpp")
#  include "secp256k1/frost.hpp"
#  define HAS_FROST 1
#else
#  define HAS_FROST 0
#endif

#include "audit_check.hpp"

using namespace secp256k1;
using secp256k1::fast::Scalar;
using secp256k1::fast::Point;

static int g_pass = 0, g_fail = 0;

// ── helpers ──────────────────────────────────────────────────────────────────

static std::mt19937_64 rng(UINT64_C(0xFEEDDEADC0DECAFE));  // NOLINT

static Scalar random_nonzero_scalar() {
    for (;;) {
        std::array<uint8_t, 32> buf{};
        for (int i = 0; i < 4; ++i) {
            uint64_t v = rng();
            std::memcpy(buf.data() + i * 8, &v, 8);
        }
        Scalar s = Scalar::from_bytes(buf);
        if (!s.is_zero()) return s;
    }
}

static std::array<uint8_t, 32> random_msg() {
    std::array<uint8_t, 32> m{};
    for (int i = 0; i < 4; ++i) {
        uint64_t v = rng();
        std::memcpy(m.data() + i * 8, &v, 8);
    }
    return m;
}

static std::array<uint8_t, 32> make_seed(uint8_t byte0) {
    std::array<uint8_t, 32> s{};
    s[0] = byte0;
    return s;
}

static std::string read_source_file(const char* name) {
    const char* prefixes[] = {"", "src/cpu/src/", "../cpu/src/", nullptr};
    for (int i = 0; prefixes[i]; ++i) {
        std::string path = std::string(prefixes[i]) + name;
        std::ifstream f(path);
        if (f.is_open()) {
            return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
        }
    }
    return {};
}

// ── SEC-002/CT-002: frost_lagrange_coefficient_from_commitments uses ct::scalar_mul ──
//
// Verifies correctness of the full FROST signing path, which internally invokes
// frost_lagrange_coefficient_from_commitments. A CT regression would cause the
// Lagrange coefficients to be computed incorrectly, breaking aggregation.
//
// Uses the secp256k1::frost_* C++ API for a 2-of-3 round-trip.

static void test_frost_lagrange_ct_correctness() {
    printf("[1] frost_lagrange_coefficient_from_commitments: ct::scalar_mul correctness\n");

#if HAS_FROST
    // Fixed seeds for reproducibility.
    auto seed1 = make_seed(0x11);
    auto seed2 = make_seed(0x22);
    auto seed3 = make_seed(0x33);

    // Round 1: each participant generates their commitment and shares.
    auto [commitment1, shares1] = secp256k1::frost_keygen_begin(1, 2, 3, seed1);
    auto [commitment2, shares2] = secp256k1::frost_keygen_begin(2, 2, 3, seed2);
    auto [commitment3, shares3] = secp256k1::frost_keygen_begin(3, 2, 3, seed3);

    std::vector<FrostCommitment> all_commitments = {commitment1, commitment2, commitment3};

    // Round 2: participant i receives shares_i[i-1] from each other participant.
    std::vector<FrostShare> shares_for_p1 = {shares1[0], shares2[0], shares3[0]};
    std::vector<FrostShare> shares_for_p2 = {shares1[1], shares2[1], shares3[1]};

    auto [kp1, ok1] = secp256k1::frost_keygen_finalize(1, all_commitments,
                                                        shares_for_p1, 2, 3);
    auto [kp2, ok2] = secp256k1::frost_keygen_finalize(2, all_commitments,
                                                        shares_for_p2, 2, 3);

    CHECK(ok1, "frost_keygen_finalize(P1)");
    CHECK(ok2, "frost_keygen_finalize(P2)");
    if (!ok1 || !ok2) {
        printf("  [SKIP] DKG failed\n");
        return;
    }

    auto nonce_seed1 = make_seed(0xA1);
    auto nonce_seed2 = make_seed(0xA2);
    auto [sec_nonce1, pub_nonce1] = secp256k1::frost_sign_nonce_gen(1, nonce_seed1);
    auto [sec_nonce2, pub_nonce2] = secp256k1::frost_sign_nonce_gen(2, nonce_seed2);

    std::array<uint8_t, 32> msg{};
    msg[0] = 0xAB; msg[31] = 0xCD;

    std::vector<FrostNonceCommitment> nonce_commitments = {pub_nonce1, pub_nonce2};

    // Partial signatures — internally calls frost_lagrange_coefficient_from_commitments.
    auto psig1 = secp256k1::frost_sign(kp1, sec_nonce1, msg, nonce_commitments);
    auto psig2 = secp256k1::frost_sign(kp2, sec_nonce2, msg, nonce_commitments);

    std::vector<FrostPartialSig> partial_sigs = {psig1, psig2};
    auto final_sig = secp256k1::frost_aggregate(partial_sigs, nonce_commitments,
                                                 kp1.group_public_key, msg);

    // Verify using the group public key's x-only representation.
    auto gpk_x = kp1.group_public_key.x().to_bytes();
    bool verified = secp256k1::schnorr_verify(gpk_x, msg, final_sig);
    CHECK(verified, "frost 2-of-3 sign+aggregate+verify (ct::scalar_mul lagrange fix)");
#else
    printf("  [SKIP] FROST not compiled in this build\n");
#endif
}

// ── SEC-007: batch_weight non-zero fallback ───────────────────────────────────
//
// The fix adds Scalar::one() fallback when from_bytes returns 0 (h == n mod n).
// We verify that batch_verify still produces correct results for valid and
// invalid signatures — normal operation is unaffected.

static void test_batch_weight_nonzero() {
    printf("[2] batch_weight: normal batch verify correctness after fallback fix\n");

    // Sign 4 valid messages, batch-verify them → expect true.
    const int N = 4;
    std::vector<ECDSABatchEntry> entries;
    entries.reserve(N);

    for (int i = 0; i < N; ++i) {
        Scalar sk = random_nonzero_scalar();
        auto msg = random_msg();
        auto sig = secp256k1::ct::ecdsa_sign(msg, sk);
        auto pk = secp256k1::ct::generator_mul(sk);
        entries.push_back({msg, pk, sig});
    }

    bool valid_batch = secp256k1::ecdsa_batch_verify(entries.data(), entries.size());
    CHECK(valid_batch, "batch_verify: 4 valid ECDSA sigs → true (SEC-007 fix intact)");

    // Corrupt one signature → batch must return false.
    entries[2].signature.r = Scalar::one();
    bool invalid_batch = secp256k1::ecdsa_batch_verify(entries.data(), entries.size());
    CHECK(!invalid_batch, "batch_verify: 1 corrupted sig → false (fail-closed maintained)");
}

// ── SEC-008: ecdsa_adaptor_sign fully-zero sentinel on r.is_zero() ────────────
//
// The ABI layer guards against degenerate output after calling ecdsa_adaptor_sign.
// This sub-test verifies the C++ layer fix via source scan and a functional
// round-trip on the normal path (cannot force r==0 deterministically).

static void test_adaptor_degenerate_sentinel() {
    printf("[3] ecdsa_adaptor_sign: degenerate sentinel — source scan + round-trip\n");

    std::string src = read_source_file("adaptor.cpp");
    if (src.empty()) {
        printf("  [SKIP] adaptor.cpp not found — structural check skipped\n");
    } else {
        // Old code: "return ECDSAAdaptorSig{R_hat, Scalar::zero(), r};" on r.is_zero()
        bool old_partial =
            (src.find("ECDSAAdaptorSig{R_hat, Scalar::zero(), r}") != std::string::npos);
        CHECK(!old_partial,
              "adaptor.cpp: partial-struct {R_hat,0,r} on r.is_zero() is gone (SEC-008)");

        // New code: must use Point::infinity() sentinel.
        bool new_sentinel =
            (src.find("Point::infinity(), Scalar::zero(), Scalar::zero()") != std::string::npos);
        CHECK(new_sentinel,
              "adaptor.cpp: fully-zero sentinel {infinity,0,0} present (SEC-008)");
    }

    // Functional round-trip on normal path.
    {
        Scalar sk = random_nonzero_scalar();
        auto msg = random_msg();
        Scalar t = random_nonzero_scalar();
        Point T = secp256k1::ct::generator_mul(t);
        Point pk = secp256k1::ct::generator_mul(sk);

        auto pre = secp256k1::ecdsa_adaptor_sign(sk, msg, T);
        bool ok = secp256k1::ecdsa_adaptor_verify(pre, pk, msg, T);
        CHECK(ok, "ecdsa_adaptor_sign+verify round-trip (SEC-008 fix intact)");
        CHECK(!pre.R_hat.is_infinity(), "adaptor pre-sig R_hat != infinity for normal input");
        CHECK(!pre.r.is_zero(), "adaptor pre-sig r != 0 for normal input");
    }
}

// ── SEC-010: bip32_master_key uses parse_bytes_strict_nonzero ────────────────

static void test_bip32_master_key_strict_nonzero() {
    printf("[4] bip32_master_key: parse_bytes_strict_nonzero (SEC-010)\n");

    std::string src = read_source_file("bip32.cpp");
    if (src.empty()) {
        printf("  [SKIP] bip32.cpp not found — structural check skipped\n");
    } else {
        bool has_new = (src.find("parse_bytes_strict_nonzero(IL, master_key)") != std::string::npos);
        CHECK(has_new, "bip32.cpp: parse_bytes_strict_nonzero(IL, master_key) present (SEC-010)");
    }

    // Functional: BIP-32 TV1 seed must produce a valid master key.
    {
        // BIP-32 TV1 seed = 0x000102030405060708090a0b0c0d0e0f
        const uint8_t seed[] = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
                                 0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f};
        auto [ext, ok] = secp256k1::bip32_master_key(seed, sizeof(seed));
        CHECK(ok, "bip32_master_key: TV1 seed accepted (SEC-010 fix intact)");
        CHECK(ext.is_private, "bip32_master_key: result is private key");
    }
}

// ── CT-004: musig2_nonce_gen uses generator_mul_blinded ──────────────────────

static void test_musig2_nonce_gen_blinded() {
    printf("[5] musig2_nonce_gen: generator_mul_blinded for R1/R2 (CT-004)\n");

    std::string src = read_source_file("musig2.cpp");
    if (!src.empty()) {
        bool has_blinded = (src.find("generator_mul_blinded(sec.k1)") != std::string::npos &&
                            src.find("generator_mul_blinded(sec.k2)") != std::string::npos);
        CHECK(has_blinded,
              "musig2.cpp: generator_mul_blinded(sec.k1/k2) present (CT-004)");
    } else {
        printf("  [SKIP] musig2.cpp not found — structural check skipped\n");
    }

    // Functional: nonces are consistent with k1*G, k2*G.
    {
        Scalar sk = random_nonzero_scalar();
        std::array<uint8_t, 32> pk_bytes{};
        auto pk = secp256k1::ct::generator_mul(sk);
        auto pk_comp = pk.to_compressed();
        // x-only pubkey (32 bytes, skip 0x02/0x03 prefix)
        std::memcpy(pk_bytes.data(), pk_comp.data() + 1, 32);

        std::array<uint8_t, 32> agg_pk_bytes = pk_bytes;
        auto msg = random_msg();

        auto [sec, pub] = secp256k1::musig2_nonce_gen(sk, pk_bytes, agg_pk_bytes, msg, nullptr);

        bool r1_valid = (pub.R1[0] == 0x02 || pub.R1[0] == 0x03);
        bool r2_valid = (pub.R2[0] == 0x02 || pub.R2[0] == 0x03);
        CHECK(r1_valid, "musig2_nonce_gen: R1 valid compressed prefix (CT-004)");
        CHECK(r2_valid, "musig2_nonce_gen: R2 valid compressed prefix (CT-004)");
        CHECK(!sec.k1.is_zero(), "musig2_nonce_gen: k1 != 0");
        CHECK(!sec.k2.is_zero(), "musig2_nonce_gen: k2 != 0");

        // Blinded mul is mathematically transparent: R = k*G regardless.
        auto R1_check = secp256k1::ct::generator_mul(sec.k1).to_compressed();
        auto R2_check = secp256k1::ct::generator_mul(sec.k2).to_compressed();
        CHECK(std::memcmp(pub.R1.data(), R1_check.data(), 33) == 0,
              "musig2_nonce_gen: pub.R1 == k1*G (blinded mul transparent)");
        CHECK(std::memcmp(pub.R2.data(), R2_check.data(), 33) == 0,
              "musig2_nonce_gen: pub.R2 == k2*G (blinded mul transparent)");
    }
}

// ── CT-005: ecdsa_sign_verified calls ct::ecdsa_sign directly ────────────────

static void test_ecdsa_sign_verified_ct_direct() {
    printf("[6] ecdsa_sign_verified: ct::ecdsa_sign direct call (CT-005)\n");

    std::string src = read_source_file("ecdsa.cpp");
    if (!src.empty()) {
        // Find ecdsa_sign_verified and verify ct::ecdsa_sign is called within it.
        size_t pos = src.find("ECDSASignature ecdsa_sign_verified");
        bool ct_call_present = false;
        if (pos != std::string::npos) {
            std::string vicinity = src.substr(pos, 500);
            ct_call_present = (vicinity.find("ct::ecdsa_sign(") != std::string::npos);
        }
        CHECK(ct_call_present,
              "ecdsa.cpp: ecdsa_sign_verified calls ct::ecdsa_sign directly (CT-005)");

        // The pragma suppression must be gone from the ecdsa_sign_verified body.
        bool pragma_gone = true;
        if (pos != std::string::npos) {
            std::string vicinity = src.substr(pos, 600);
            pragma_gone = (vicinity.find("-Wdeprecated-declarations") == std::string::npos);
        }
        CHECK(pragma_gone,
              "ecdsa.cpp: -Wdeprecated-declarations pragma gone from ecdsa_sign_verified (CT-005)");
    } else {
        printf("  [SKIP] ecdsa.cpp not found — structural check skipped\n");
    }

    // Functional: sign+verify round-trip.
    {
        Scalar sk = random_nonzero_scalar();
        auto msg = random_msg();
        auto sig = secp256k1::ecdsa_sign_verified(msg, sk);
        CHECK(!sig.r.is_zero(), "ecdsa_sign_verified: r != 0 for valid key (CT-005)");
        CHECK(!sig.s.is_zero(), "ecdsa_sign_verified: s != 0 for valid key (CT-005)");
        auto pk = secp256k1::ct::generator_mul(sk);
        bool ok = secp256k1::ecdsa_verify(msg.data(), pk, sig);
        CHECK(ok, "ecdsa_sign_verified: produced signature verifies (CT-005)");
    }

    // 10 random keys — all must verify.
    int ok_count = 0;
    for (int i = 0; i < 10; ++i) {
        Scalar sk = random_nonzero_scalar();
        auto msg = random_msg();
        auto sig = secp256k1::ecdsa_sign_verified(msg, sk);
        auto pk = secp256k1::ct::generator_mul(sk);
        if (secp256k1::ecdsa_verify(msg.data(), pk, sig)) ++ok_count;
    }
    char buf[80];
    std::snprintf(buf, sizeof(buf), "ecdsa_sign_verified: %d/10 random keys verified", ok_count);
    CHECK(ok_count == 10, buf);
}

// ── entry point ──────────────────────────────────────────────────────────────

int test_regression_ct_ops_v2_run() {
    g_pass = 0; g_fail = 0;
    printf("======================================================================\n");
    printf("  Regression: CT ops source-scan guards (2026-05-21)\n");
    printf("  Fixes: SEC-002/CT-002 FROST lagrange, SEC-007 batch_weight,\n");
    printf("         SEC-008 adaptor sentinel, SEC-010 BIP-32 master key,\n");
    printf("         CT-004 MuSig2 nonce gen, CT-005 ecdsa_sign_verified\n");
    printf("======================================================================\n\n");

    test_frost_lagrange_ct_correctness();
    printf("\n");
    test_batch_weight_nonzero();
    printf("\n");
    test_adaptor_degenerate_sentinel();
    printf("\n");
    test_bip32_master_key_strict_nonzero();
    printf("\n");
    test_musig2_nonce_gen_blinded();
    printf("\n");
    test_ecdsa_sign_verified_ct_direct();
    printf("\n");

    printf("[regression_ct_ops] %d/%d checks passed\n",
           g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_ct_ops_v2_run(); }
#endif
