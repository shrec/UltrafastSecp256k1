// =============================================================================
// REGRESSION TEST: CT operation regressions (SEC-002/007/008/010, CT-004/005)
// =============================================================================
//
// This test covers CT-layer correctness regressions across multiple SEC/CT findings:
//
//   CT-004  ecdsa_sign_verified() uses ct:: path internally — validate that the
//           sign+verify round-trip produces correct results.
//   CT-005  CT field operations produce same result as fast:: for public data.
//   SEC-007 ECDSA verify accepts high-S signatures (no implicit normalize).
//   SEC-008 Schnorr sign+verify round-trip and wrong-message rejection.
//   SEC-002 CT scalar operations: inverse, negate, add produce correct results.
//
// =============================================================================

#include <cstdio>
#include <cstring>
#include <array>
#include <vector>
#include <fstream>
#include <string>
#include <random>

#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ct/scalar.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/ct/sign.hpp"
#include "secp256k1/batch_verify.hpp"
#include "secp256k1/adaptor.hpp"
#include "secp256k1/bip32.hpp"
#include "secp256k1/musig2.hpp"
#include "secp256k1/detail/secure_erase.hpp"
#include "audit_check.hpp"

#if __has_include("secp256k1/frost.hpp")
#  include "secp256k1/frost.hpp"
#  define HAS_FROST 1
#else
#  define HAS_FROST 0
#endif

static int g_pass = 0;
static int g_fail = 0;

using secp256k1::fast::Scalar;
using secp256k1::fast::Point;

// Known valid private key for deterministic tests
static const uint8_t kTestKey[32] = {
    0xe8,0xf3,0x2e,0x72, 0x3d,0xec,0xf4,0x05,
    0x1a,0xef,0xac,0x8e, 0x2c,0x93,0xc9,0xc5,
    0xb2,0x14,0x31,0x38, 0x17,0xcd,0xb0,0x1a,
    0x14,0x94,0xb9,0x17, 0xc8,0x43,0x6b,0x35
};

// ---------------------------------------------------------------------------
// CT-004: ecdsa_sign_verified round-trip
// ---------------------------------------------------------------------------
static void test_ct004_ecdsa_sign_verified_roundtrip() {
    std::printf("  [CT-004] ecdsa_sign_verified round-trip\n");

    Scalar sk;
    bool ok = Scalar::parse_bytes_strict_nonzero(kTestKey, sk);
    CHECK(ok, "CT-004: test key parses OK");

    std::array<uint8_t, 32> msg_hash{};
    msg_hash[0] = 0xDE; msg_hash[1] = 0xAD;
    msg_hash[2] = 0xBE; msg_hash[3] = 0xEF;

    auto sig = secp256k1::ecdsa_sign_verified(msg_hash, sk);

    CHECK(!sig.r.is_zero(), "CT-004: sign_verified r != 0");
    CHECK(!sig.s.is_zero(), "CT-004: sign_verified s != 0");

    auto pk = secp256k1::ct::generator_mul(sk);
    bool verify_ok = secp256k1::ecdsa_verify(msg_hash.data(), pk, sig);
    CHECK(verify_ok, "CT-004: signature from ecdsa_sign_verified verifies");

    secp256k1::detail::secure_erase(&sk, sizeof(sk));
}

// ---------------------------------------------------------------------------
// CT-005: CT and fast:: generator_mul produce identical x-coordinates
// ---------------------------------------------------------------------------
static void test_ct005_ct_fast_scalar_mul_parity() {
    std::printf("  [CT-005] ct::generator_mul and fast scalar_mul agree\n");

    Scalar sk;
    Scalar::parse_bytes_strict_nonzero(kTestKey, sk);

    auto pk_ct   = secp256k1::ct::generator_mul(sk);
    auto pk_fast = Point::generator().scalar_mul(sk);

    auto x_ct   = pk_ct.x().to_bytes();
    auto x_fast = pk_fast.x().to_bytes();
    CHECK(x_ct == x_fast, "CT-005: ct::generator_mul and fast::scalar_mul produce same x-coordinate");

    secp256k1::detail::secure_erase(&sk, sizeof(sk));
}

// ---------------------------------------------------------------------------
// SEC-007: ECDSA verify accepts high-S signatures (no implicit normalize)
// ---------------------------------------------------------------------------
static void test_sec007_ecdsa_verify_accepts_high_s() {
    std::printf("  [SEC-007] ECDSA verify accepts high-S (no implicit normalize)\n");

    Scalar sk;
    Scalar::parse_bytes_strict_nonzero(kTestKey, sk);

    std::array<uint8_t, 32> msg{};
    msg[0] = 0xAB; msg[1] = 0xCD;

    auto sig = secp256k1::ecdsa_sign_verified(msg, sk);
    auto pk  = secp256k1::ct::generator_mul(sk);

    // high-S: replace s with n - s
    secp256k1::ECDSASignature high_s_sig{sig.r, secp256k1::ct::scalar_neg(sig.s)};

    bool ok_high = secp256k1::ecdsa_verify(msg.data(), pk, high_s_sig);
    CHECK(ok_high, "SEC-007: ecdsa_verify accepts high-S signature");

    bool ok_low = secp256k1::ecdsa_verify(msg.data(), pk, sig);
    CHECK(ok_low, "SEC-007: ecdsa_verify accepts low-S signature");

    secp256k1::detail::secure_erase(&sk, sizeof(sk));
}

// ---------------------------------------------------------------------------
// SEC-008: Schnorr sign + verify round-trip
// ---------------------------------------------------------------------------
static void test_sec008_schnorr_sign_verify_roundtrip() {
    std::printf("  [SEC-008] Schnorr sign + verify round-trip\n");

    Scalar sk;
    Scalar::parse_bytes_strict_nonzero(kTestKey, sk);

    std::array<uint8_t, 32> msg{};
    msg[0] = 0x11; msg[1] = 0x22; msg[2] = 0x33;

    std::array<uint8_t, 32> aux_rand{};

    auto sig = secp256k1::schnorr_sign(sk, msg, aux_rand);
    auto xpk = secp256k1::schnorr_pubkey(sk);

    bool ok = secp256k1::schnorr_verify(xpk, msg, sig);
    CHECK(ok, "SEC-008: schnorr sign+verify round-trip");

    std::array<uint8_t, 32> bad_msg = msg;
    bad_msg[0] ^= 0xFF;
    bool bad_ok = secp256k1::schnorr_verify(xpk, bad_msg, sig);
    CHECK(!bad_ok, "SEC-008: schnorr verify rejects wrong message");

    secp256k1::detail::secure_erase(&sk, sizeof(sk));
}

// ---------------------------------------------------------------------------
// SEC-002: CT scalar operations produce correct results
// ---------------------------------------------------------------------------
static void test_sec002_ct_scalar_ops_correct() {
    std::printf("  [SEC-002] CT scalar ops: inverse, negate, add\n");

    Scalar sk;
    Scalar::parse_bytes_strict_nonzero(kTestKey, sk);

    // CT inverse: a * a^{-1} == 1
    auto inv_sk  = secp256k1::ct::scalar_inverse(sk);
    auto product = secp256k1::ct::scalar_mul(sk, inv_sk);
    uint8_t one_bytes[32] = {};
    one_bytes[31] = 0x01;
    Scalar one;
    Scalar::parse_bytes_strict_nonzero(one_bytes, one);
    CHECK(product == one, "SEC-002: a * a^{-1} == 1");

    // CT negate: a + (-a) == 0
    auto neg_sk = secp256k1::ct::scalar_neg(sk);
    auto sum    = secp256k1::ct::scalar_add(sk, neg_sk);
    CHECK(sum.is_zero(), "SEC-002: a + (-a) == 0");

    // CT sub: a - b == a + (-b)
    uint8_t b_bytes[32] = {};
    b_bytes[31] = 0x07;
    Scalar b;
    Scalar::parse_bytes_strict_nonzero(b_bytes, b);
    auto diff1 = secp256k1::ct::scalar_sub(sk, b);
    auto neg_b = secp256k1::ct::scalar_neg(b);
    auto diff2 = secp256k1::ct::scalar_add(sk, neg_b);
    CHECK(diff1 == diff2, "SEC-002: scalar_sub(a,b) == scalar_add(a, scalar_neg(b))");

    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    secp256k1::detail::secure_erase(&inv_sk, sizeof(inv_sk));
    secp256k1::detail::secure_erase(&neg_sk, sizeof(neg_sk));
}

// ---------------------------------------------------------------------------
// Source-scan helpers and new sub-tests for 2026-05-21 code fixes
// ---------------------------------------------------------------------------

static std::mt19937_64 g_rng(UINT64_C(0xFEEDDEADC0DECAFE));  // NOLINT

static Scalar random_nonzero_scalar_() {
    for (;;) {
        std::array<uint8_t, 32> buf{};
        for (int i = 0; i < 4; ++i) {
            uint64_t v = g_rng();
            std::memcpy(buf.data() + i * 8, &v, 8);
        }
        Scalar s = Scalar::from_bytes(buf);
        if (!s.is_zero()) return s;
    }
}

static std::array<uint8_t, 32> random_msg_() {
    std::array<uint8_t, 32> m{};
    for (int i = 0; i < 4; ++i) {
        uint64_t v = g_rng();
        std::memcpy(m.data() + i * 8, &v, 8);
    }
    return m;
}

// Fail-closed (issue #335 acceptance repair, round 6): this file's 6
// source-scan sub-tests all read via this helper, and until this fix it was
// a CWD-relative-only bounded search ("", "src/cpu/src/", "../cpu/src/")
// identical in class to the resolver round 5 removed from
// test_regression_bip39_csprng_failclosed.cpp and friends. ci/check_audit_
// cwd_independence.py (the round-6 exhaustive dual-CWD sweep) caught this
// file specifically: from an unrelated CWD every one of the 6 callers below
// silently printed "[SKIP] <file> not found" and skipped its source-scan
// CHECK()s while the module still reported an overall PASS. Routes through
// the shared, UFSECP_SOURCE_ROOT-aware audit_read_source_file() first (see
// audit_check.hpp); the bare-name fallback is kept only for the standalone
// CTest build, which does not define UFSECP_SOURCE_ROOT.
static std::string read_src_file_(const char* name) {
    std::string src = audit_read_source_file((std::string("src/cpu/src/") + name).c_str());
    if (!src.empty()) return src;
    return audit_read_source_file(name);
}

// -- CT-005: ecdsa_sign_verified calls ct::ecdsa_sign (source scan) --
static void test_ct005_source_scan() {
    std::printf("  [CT-005] source scan: ecdsa_sign_verified calls ct::ecdsa_sign\n");
    std::string src = read_src_file_("ecdsa.cpp");
    CHECK(!src.empty(), "CT-005: ecdsa.cpp must be readable (in-tree source always exists)");
    if (src.empty()) return;
    size_t pos = src.find("ECDSASignature ecdsa_sign_verified");
    bool ct_call = false;
    bool pragma_gone = true;
    if (pos != std::string::npos) {
        std::string v = src.substr(pos, 600);
        ct_call = (v.find("ct::ecdsa_sign(") != std::string::npos);
        pragma_gone = (v.find("-Wdeprecated-declarations") == std::string::npos);
    }
    CHECK(ct_call, "CT-005: ecdsa_sign_verified calls ct::ecdsa_sign directly");
    CHECK(pragma_gone, "CT-005: pragma suppression gone from ecdsa_sign_verified");
}

// -- CT-004: musig2_nonce_gen uses generator_mul_blinded (source scan + functional) --
static void test_ct004_musig2_blinded() {
    std::printf("  [CT-004] musig2_nonce_gen: generator_mul_blinded + transparency\n");
    std::string src = read_src_file_("musig2.cpp");
    CHECK(!src.empty(), "CT-004: musig2.cpp must be readable (in-tree source always exists)");
    if (!src.empty()) {
        bool ok = (src.find("generator_mul_blinded(sec.k1)") != std::string::npos &&
                   src.find("generator_mul_blinded(sec.k2)") != std::string::npos);
        CHECK(ok, "CT-004: musig2.cpp: generator_mul_blinded(sec.k1/k2) present");
    }

    Scalar sk = random_nonzero_scalar_();
    std::array<uint8_t, 32> pk_b{};
    auto pk_comp = secp256k1::ct::generator_mul(sk).to_compressed();
    std::memcpy(pk_b.data(), pk_comp.data() + 1, 32);
    auto msg = random_msg_();
    auto [sec, pub] = secp256k1::musig2_nonce_gen(sk, pk_b, pk_b, msg, nullptr);
    auto R1_check = secp256k1::ct::generator_mul(sec.k1).to_compressed();
    auto R2_check = secp256k1::ct::generator_mul(sec.k2).to_compressed();
    CHECK(std::memcmp(pub.R1.data(), R1_check.data(), 33) == 0,
          "CT-004: pub.R1 == k1*G (blinded mul transparent)");
    CHECK(std::memcmp(pub.R2.data(), R2_check.data(), 33) == 0,
          "CT-004: pub.R2 == k2*G (blinded mul transparent)");
}

// -- SEC-007: batch_weight non-zero (functional batch_verify correctness) --
static void test_sec007_batch_weight_nonzero() {
    std::printf("  [SEC-007] batch_weight non-zero: batch_verify correctness\n");
    std::vector<secp256k1::ECDSABatchEntry> entries;
    for (int i = 0; i < 4; ++i) {
        Scalar sk = random_nonzero_scalar_();
        auto msg = random_msg_();
        auto sig = secp256k1::ct::ecdsa_sign(msg, sk);
        entries.push_back({msg, secp256k1::ct::generator_mul(sk), sig});
    }
    bool valid = secp256k1::ecdsa_batch_verify(entries.data(), entries.size());
    CHECK(valid, "SEC-007: 4 valid ECDSA sigs → batch_verify true");
    entries[2].signature.r = Scalar::one();
    bool invalid = secp256k1::ecdsa_batch_verify(entries.data(), entries.size());
    CHECK(!invalid, "SEC-007: 1 corrupted sig → batch_verify false (fail-closed)");
}

// -- SEC-008: adaptor degenerate sentinel (source scan + functional) --
static void test_sec008_adaptor_sentinel() {
    std::printf("  [SEC-008] adaptor: degenerate sentinel source scan + round-trip\n");
    std::string src = read_src_file_("adaptor.cpp");
    CHECK(!src.empty(), "SEC-008: adaptor.cpp must be readable (in-tree source always exists)");
    if (!src.empty()) {
        bool old_partial =
            (src.find("ECDSAAdaptorSig{R_hat, Scalar::zero(), r}") != std::string::npos);
        CHECK(!old_partial, "SEC-008: partial-struct {R_hat,0,r} on r.is_zero() is gone");
        // GHSA-c7q2-gv3g-rgxm: degenerate paths now return the pre-zeroed `kZero`
        // sentinel (all-infinity points + all-zero scalars) instead of any partial struct.
        bool new_sentinel = (src.find("ECDSAAdaptorSig kZero") != std::string::npos)
                         && (src.find("return kZero") != std::string::npos);
        CHECK(new_sentinel, "SEC-008: degenerate r returns the fully-zero kZero sentinel");
    }
    // Normal path round-trip
    {
        Scalar sk = random_nonzero_scalar_();
        auto msg = random_msg_();
        Scalar t = random_nonzero_scalar_();
        Point T = secp256k1::ct::generator_mul(t);
        Point pk = secp256k1::ct::generator_mul(sk);
        auto pre = secp256k1::ecdsa_adaptor_sign(sk, msg, T);
        CHECK(secp256k1::ecdsa_adaptor_verify(pre, pk, msg, T),
              "SEC-008: ecdsa_adaptor_sign+verify round-trip");
        CHECK(!pre.R_hat.is_infinity(), "SEC-008: normal path: R_hat != infinity");
        CHECK(!pre.r.is_zero(), "SEC-008: normal path: r != 0");
    }
}

// -- SEC-010: bip32_master_key uses parse_bytes_strict_nonzero (source scan + functional) --
static void test_sec010_bip32_strict_nonzero() {
    std::printf("  [SEC-010] bip32_master_key: parse_bytes_strict_nonzero\n");
    std::string src = read_src_file_("bip32.cpp");
    CHECK(!src.empty(), "SEC-010: bip32.cpp must be readable (in-tree source always exists)");
    if (!src.empty()) {
        CHECK(src.find("parse_bytes_strict_nonzero(IL, master_key)") != std::string::npos,
              "SEC-010: bip32.cpp: parse_bytes_strict_nonzero(IL, master_key) present");
    }
    // BIP-32 TV1 seed
    const uint8_t seed[] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    auto [ext, ok] = secp256k1::bip32_master_key(seed, sizeof(seed));
    CHECK(ok, "SEC-010: bip32_master_key TV1 seed accepted");
    CHECK(ext.is_private, "SEC-010: result is private key");
}

// -- SEC-002/CT-002: frost_lagrange uses ct::scalar_mul (source scan + 2-of-3) --
static void test_sec002_frost_lagrange() {
    std::printf("  [SEC-002/CT-002] frost_lagrange: ct::scalar_mul source scan + 2-of-3\n");
    std::string src = read_src_file_("frost.cpp");
    CHECK(!src.empty(), "SEC-002/CT-002: frost.cpp must be readable (in-tree source always exists)");
    if (!src.empty()) {
        CHECK(src.find("ct::scalar_mul(num, x_j)") != std::string::npos,
              "SEC-002/CT-002: frost.cpp: ct::scalar_mul(num, x_j) in lagrange loop");
        CHECK(src.find("ct::scalar_sub(x_j, x_i)") != std::string::npos,
              "SEC-002/CT-002: frost.cpp: ct::scalar_sub(x_j, x_i) in lagrange loop");
    }

#if HAS_FROST
    auto mks = [](uint8_t b) {
        std::array<uint8_t, 32> s{};
        s[0] = b;
        return s;
    };
    auto [c1, s1] = secp256k1::frost_keygen_begin(1, 2, 3, mks(0x11));
    auto [c2, s2] = secp256k1::frost_keygen_begin(2, 2, 3, mks(0x22));
    auto [c3, s3] = secp256k1::frost_keygen_begin(3, 2, 3, mks(0x33));
    std::vector<secp256k1::FrostCommitment> ac = {c1, c2, c3};
    std::vector<secp256k1::FrostShare> sp1 = {s1[0], s2[0], s3[0]};
    std::vector<secp256k1::FrostShare> sp2 = {s1[1], s2[1], s3[1]};
    auto [kp1, ok1] = secp256k1::frost_keygen_finalize(1, ac, sp1, 2, 3);
    auto [kp2, ok2] = secp256k1::frost_keygen_finalize(2, ac, sp2, 2, 3);
    CHECK(ok1 && ok2, "SEC-002: FROST 2-of-3 DKG");
    if (!ok1 || !ok2) return;
    auto [sn1, pn1] = secp256k1::frost_sign_nonce_gen(1, mks(0xA1));
    auto [sn2, pn2] = secp256k1::frost_sign_nonce_gen(2, mks(0xA2));
    std::array<uint8_t, 32> msg{}; msg[0] = 0xAB; msg[31] = 0xCD;
    std::vector<secp256k1::FrostNonceCommitment> nc = {pn1, pn2};
    auto ps1 = secp256k1::frost_sign(kp1, sn1, msg, nc);
    auto ps2 = secp256k1::frost_sign(kp2, sn2, msg, nc);
    std::vector<secp256k1::FrostPartialSig> psigs = {ps1, ps2};
    auto fs = secp256k1::frost_aggregate(psigs, nc, kp1.group_public_key, msg);
    CHECK(secp256k1::schnorr_verify(kp1.group_public_key.x().to_bytes(), msg, fs),
          "SEC-002/CT-002: FROST 2-of-3 sign+verify (ct::scalar_mul lagrange)");
#else
    std::printf("    [SKIP] FROST not compiled\n");
#endif
}

// ---------------------------------------------------------------------------
// SEC-002-EXTRACT: schnorr_adaptor_extract uses ct:: arithmetic on recovered secret
// ---------------------------------------------------------------------------
static void test_sec002_adaptor_extract_ct() {
    std::printf("  [SEC-002-EXTRACT] schnorr_adaptor_extract: ct::scalar_sub + ct::scalar_cneg source scan\n");
    std::string src = read_src_file_("adaptor.cpp");
    CHECK(!src.empty(), "SEC-002-EXTRACT: adaptor.cpp must be readable (in-tree source always exists)");
    if (src.empty()) return;

    // Verify old VT operator- is gone from schnorr_adaptor_extract
    // The old code was: Scalar t = sig.s - pre_sig.s_hat;
    // We check the function body for ct::scalar_sub usage.
    bool has_ct_sub  = (src.find("ct::scalar_sub(sig.s, pre_sig.s_hat)") != std::string::npos);
    bool has_ct_cneg = (src.find("ct::scalar_cneg(t, ct::bool_to_mask(pre_sig.needs_negation))") != std::string::npos);
    bool has_is_zero_ct = (src.find("t.is_zero_ct()") != std::string::npos);
    CHECK(has_ct_sub,  "SEC-002-EXTRACT: schnorr_adaptor_extract uses ct::scalar_sub");
    CHECK(has_ct_cneg, "SEC-002-EXTRACT: schnorr_adaptor_extract uses ct::scalar_cneg");
    CHECK(has_is_zero_ct, "SEC-002-EXTRACT: schnorr_adaptor_extract uses is_zero_ct on result");

    // Verify ecdsa_adaptor_extract uses is_zero_ct on the recovered secret t.
    // Positive assertion: the function body must contain the exact CT check
    //   `if (t.is_zero_ct()) return {Scalar::zero(), false};`
    // A substring search for `t.is_zero()` is unsafe — `hat.is_zero()` from
    // `pre_sig.s_hat.is_zero()` matches it as a suffix. Either `{t, false}`
    // or `{Scalar::zero(), false}` is acceptable (both are semantically
    // equivalent when t is zero; the latter is more explicit).
    bool ecdsa_t_is_zero_ct =
        (src.find("if (t.is_zero_ct()) return {t, false};") != std::string::npos) ||
        (src.find("if (t.is_zero_ct()) return {Scalar::zero(), false};") != std::string::npos);
    CHECK(ecdsa_t_is_zero_ct,
          "SEC-002-EXTRACT: ecdsa_adaptor_extract uses is_zero_ct on recovered t");
}

// ---------------------------------------------------------------------------
// CT-005 (extended): ecdsa_sign_verified over many random keys all verify.
// Merged here from the former test_regression_ct_ops_2026_05_21.cpp (v2) — its
// only sub-test not already covered above. The two files were otherwise
// byte-identical source-scan + functional duplicates of the same six fixes.
// ---------------------------------------------------------------------------
static void test_ecdsa_sign_verified_random_keys() {
    std::printf("  [CT-005] ecdsa_sign_verified: 10 random keys all verify\n");
    int ok_count = 0;
    for (int i = 0; i < 10; ++i) {
        Scalar sk = random_nonzero_scalar_();
        auto msg = random_msg_();
        auto sig = secp256k1::ecdsa_sign_verified(msg, sk);
        auto pk  = secp256k1::ct::generator_mul(sk);
        if (secp256k1::ecdsa_verify(msg.data(), pk, sig)) ++ok_count;
    }
    char buf[80];
    std::snprintf(buf, sizeof(buf), "CT-005: %d/10 random keys verified", ok_count);
    CHECK(ok_count == 10, buf);
}

// ---------------------------------------------------------------------------
// Entry point — single canonical CT-ops regression (was split across two
// byte-identical files; the 2026-05-21 duplicate was consolidated into this one).
// ---------------------------------------------------------------------------
int test_regression_ct_ops_run() {
    g_pass = 0;
    g_fail = 0;

    std::printf("[regression_ct_ops] CT operation regressions (SEC-002/007/008/010, CT-004/005)\n\n");

    // Original sub-tests (functional correctness)
    test_ct004_ecdsa_sign_verified_roundtrip();
    test_ct005_ct_fast_scalar_mul_parity();
    test_sec007_ecdsa_verify_accepts_high_s();
    test_sec008_schnorr_sign_verify_roundtrip();
    test_sec002_ct_scalar_ops_correct();
    std::printf("\n");

    // 2026-05-21 source-scan + extended functional tests
    test_ct005_source_scan();
    test_ct004_musig2_blinded();
    test_sec007_batch_weight_nonzero();
    test_sec008_adaptor_sentinel();
    test_sec010_bip32_strict_nonzero();
    test_sec002_frost_lagrange();

    // 2026-05-22 SEC-002-EXTRACT: adaptor extract CT fix
    test_sec002_adaptor_extract_ct();

    // Merged from the former v2 duplicate: 10-random-key sign+verify loop.
    test_ecdsa_sign_verified_random_keys();

    std::printf("\n  pass=%d  fail=%d\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_ct_ops_run(); }
#endif
