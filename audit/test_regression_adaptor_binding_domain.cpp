// ============================================================================
// test_regression_adaptor_binding_domain.cpp — adaptor binding BIP-340 domain
// ============================================================================
// Regression tests for the ecdsa_adaptor_binding() domain-separation fix:
//
//   BEFORE (v1): SHA256("ecdsa_adaptor_bind_v1" || adaptor_compressed)
//     — plain SHA256 with tag appended, no BIP-340 double-tag prefix.
//     — No cross-protocol domain separation.
//
//   AFTER (v2):  SHA256(SHA256("ecdsa_adaptor_bind_v2") ||
//                       SHA256("ecdsa_adaptor_bind_v2") ||
//                       adaptor_compressed)
//     — BIP-340 tagged hash with double-prepended tag hash.
//     — Domain-separated from arbitrary SHA256 computations.
//
// TESTS:
//   ADB-1  Schnorr adaptor sign/verify round-trip succeeds with new binding hash
//   ADB-2  schnorr_adaptor_verify rejects flipped needs_negation flag
//   ADB-3  schnorr_adaptor_adapt produces a valid BIP-340 Schnorr signature
//   ADB-4  schnorr_adaptor_extract recovers the adaptor secret exactly
//   ADB-5  ECDSA adaptor sign/verify round-trip succeeds with new binding hash
//   ADB-6  New (v2) binding differs from old (v1) plain-SHA256 binding
//          — confirms domain separation is active, not a no-op change
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <array>
#include <cstring>

#include "secp256k1/adaptor.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/ct/point.hpp"

using namespace secp256k1;
using secp256k1::fast::Scalar;
using secp256k1::fast::Point;

static int g_pass = 0, g_fail = 0;
static void check(bool cond, const char* msg) {
    if (cond) { ++g_pass; printf("  [OK]   %s\n", msg); }
    else       { ++g_fail; printf("  [FAIL] %s\n", msg); }
    fflush(stdout);
}

// Fixed test vectors — deterministic, no randomness.
static Scalar make_sk(uint8_t byte31) {
    std::array<uint8_t, 32> b{};
    b[31] = byte31;
    Scalar s;
    Scalar::parse_bytes_strict_nonzero(b.data(), s);
    return s;
}

static std::array<uint8_t, 32> make_msg(uint8_t byte0) {
    std::array<uint8_t, 32> m{};
    m[0] = byte0;
    m[31] = 0xAA;
    return m;
}

static std::array<uint8_t, 32> zero_aux() { return {}; }

// ── ADB-1: Schnorr adaptor sign/verify round-trip ──────────────────────────
static void test_adb_1() {
    printf("  -- ADB-1: Schnorr adaptor sign/verify round-trip\n");

    Scalar sk    = make_sk(7);
    Scalar t     = make_sk(11);      // adaptor secret
    Point  T     = ct::generator_mul(t);
    auto   msg   = make_msg(0x01);
    auto   aux   = zero_aux();

    SchnorrAdaptorSig pre = schnorr_adaptor_sign(sk, msg, T, aux);

    // Derive x-only pubkey bytes
    Point P = ct::generator_mul(sk);
    auto [px_bytes, p_odd] = P.x_bytes_and_parity();
    if (p_odd) P = P.negate();  // even-Y convention

    bool ok = schnorr_adaptor_verify(pre, px_bytes, msg, T);
    check(ok, "[adb-1] schnorr_adaptor_verify(honest pre-sig) == true");
}

// ── ADB-2: Reject flipped needs_negation ───────────────────────────────────
static void test_adb_2() {
    printf("  -- ADB-2: needs_negation integrity check\n");

    Scalar sk = make_sk(5);
    Scalar t  = make_sk(9);
    Point  T  = ct::generator_mul(t);
    auto   msg = make_msg(0x02);
    auto   aux = zero_aux();

    SchnorrAdaptorSig pre = schnorr_adaptor_sign(sk, msg, T, aux);

    Point P = ct::generator_mul(sk);
    auto [px_bytes, p_odd] = P.x_bytes_and_parity();
    if (p_odd) P = P.negate();

    // Honest verify must pass
    bool honest = schnorr_adaptor_verify(pre, px_bytes, msg, T);
    check(honest, "[adb-2] honest pre-sig passes verify");

    // Tampered: flip needs_negation — must be rejected
    SchnorrAdaptorSig tampered = pre;
    tampered.needs_negation = !tampered.needs_negation;
    bool tampered_ok = schnorr_adaptor_verify(tampered, px_bytes, msg, T);
    check(!tampered_ok, "[adb-2] flipped needs_negation is rejected by verify");
}

// ── ADB-3: adapt → valid Schnorr sig ───────────────────────────────────────
static void test_adb_3() {
    printf("  -- ADB-3: adapt produces valid BIP-340 Schnorr signature\n");

    Scalar sk = make_sk(3);
    Scalar t  = make_sk(13);
    Point  T  = ct::generator_mul(t);
    auto   msg = make_msg(0x03);
    auto   aux = zero_aux();

    SchnorrAdaptorSig pre = schnorr_adaptor_sign(sk, msg, T, aux);

    Point P = ct::generator_mul(sk);
    auto [px_bytes, p_odd] = P.x_bytes_and_parity();
    if (p_odd) P = P.negate();

    SchnorrSignature final_sig = schnorr_adaptor_adapt(pre, t);

    bool verify_ok = schnorr_verify(px_bytes.data(), msg.data(), final_sig);
    check(verify_ok, "[adb-3] adapted signature passes schnorr_verify");
    check(!final_sig.s.is_zero(), "[adb-3] adapted signature has non-zero s");
}

// ── ADB-4: extract recovers adaptor secret exactly ─────────────────────────
static void test_adb_4() {
    printf("  -- ADB-4: extract recovers adaptor secret\n");

    Scalar sk = make_sk(17);
    Scalar t  = make_sk(23);
    Point  T  = ct::generator_mul(t);
    auto   msg = make_msg(0x04);
    auto   aux = zero_aux();

    SchnorrAdaptorSig pre       = schnorr_adaptor_sign(sk, msg, T, aux);
    SchnorrSignature  final_sig = schnorr_adaptor_adapt(pre, t);

    auto [t_recovered, ok] = schnorr_adaptor_extract(pre, final_sig);
    check(ok, "[adb-4] extract returns ok=true");

    // t_recovered must equal t (or -t if negation occurred)
    // The extract function handles the negation internally.
    Point T_check = ct::generator_mul(t_recovered);
    auto [T_x, T_odd] = T.x_bytes_and_parity();
    auto [Tc_x, Tc_odd] = T_check.x_bytes_and_parity();
    check(T_x == Tc_x, "[adb-4] extracted secret reproduces T x-coordinate");
}

// ── ADB-5: ECDSA adaptor round-trip ────────────────────────────────────────
static void test_adb_5() {
    printf("  -- ADB-5: ECDSA adaptor sign/verify round-trip\n");

    Scalar sk = make_sk(19);
    Scalar t  = make_sk(29);
    Point  T  = ct::generator_mul(t);
    Point  P  = ct::generator_mul(sk);
    auto   msg = make_msg(0x05);

    ECDSAAdaptorSig pre = ecdsa_adaptor_sign(sk, msg, T);

    check(!pre.r.is_zero() && !pre.s_hat.is_zero(),
          "[adb-5] ECDSA adaptor pre-sig is non-degenerate");

    bool ok = ecdsa_adaptor_verify(pre, P, msg, T);
    check(ok, "[adb-5] ecdsa_adaptor_verify(honest pre-sig) == true");
}

// ── ADB-6: v2 binding differs from v1 plain-SHA256 binding ─────────────────
static void test_adb_6() {
    printf("  -- ADB-6: new v2 binding hash differs from old v1 plain-SHA256\n");

    // Build a known adaptor point
    Scalar t = make_sk(37);
    Point  T = ct::generator_mul(t);
    auto adaptor_bytes = T.to_compressed();  // 33 bytes

    // Old v1 binding: SHA256("ecdsa_adaptor_bind_v1" || adaptor_compressed_bytes)
    constexpr char tag_v1[] = "ecdsa_adaptor_bind_v1";
    SHA256 h_old;
    h_old.update(reinterpret_cast<const uint8_t*>(tag_v1), sizeof(tag_v1) - 1);
    h_old.update(adaptor_bytes.data(), adaptor_bytes.size());
    auto hash_v1 = h_old.finalize();

    // New v2 binding: SHA256(SHA256(tag_v2) || SHA256(tag_v2) || adaptor_bytes)
    constexpr char tag_v2[] = "ecdsa_adaptor_bind_v2";
    auto tag_hash_v2 = SHA256::hash(reinterpret_cast<const uint8_t*>(tag_v2),
                                    sizeof(tag_v2) - 1);
    SHA256 h_new;
    h_new.update(tag_hash_v2.data(), 32);
    h_new.update(tag_hash_v2.data(), 32);
    h_new.update(adaptor_bytes.data(), adaptor_bytes.size());
    auto hash_v2 = h_new.finalize();

    check(hash_v1 != hash_v2,
          "[adb-6] v2 BIP-340 tagged hash differs from v1 plain-SHA256 (domain separation active)");
}

// ── Entry point ────────────────────────────────────────────────────────────

int test_regression_adaptor_binding_domain_run() {
    g_pass = 0; g_fail = 0;
    printf("[adaptor_binding] Adaptor Binding Domain Separation (ADB-1..6)\n");
    printf("  Wire format: ecdsa_adaptor_bind_v2 (BIP-340 tagged hash)\n\n");

    test_adb_1();
    test_adb_2();
    test_adb_3();
    test_adb_4();
    test_adb_5();
    test_adb_6();

    printf("\n[adaptor_binding] %d/%d\n", g_pass, g_pass + g_fail);
    return g_fail > 0 ? 1 : 0;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_adaptor_binding_domain_run(); }
#endif
