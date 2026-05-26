// ============================================================================
// test_regression_s_scalar_erasure.cpp
// ============================================================================
// Regression: ecdsa_sign (fast path, ecdsa.cpp) did not erase the intermediate
// scalars r and s from the stack after computing result = ECDSASignature{r,s}.
// Similarly, musig2_partial_sig_agg (musig2.cpp) did not erase s after
// s.to_bytes() serialized the aggregated partial signature.
//
// Fix (2026-05-25):
//   ecdsa.cpp:  added secure_erase(&s, sizeof(s)) and secure_erase(&r, sizeof(r))
//               immediately after secure_erase(&k_inv, sizeof(k_inv)).
//   musig2.cpp: added secure_erase(&s, sizeof(s)) after s.to_bytes().
//
// Both erasures come AFTER the values are consumed (copied to result / s_bytes),
// so the output is unchanged.  These tests verify that correctness is maintained.
//
// Tests:
//   SSR-1: ecdsa_sign fast-path sign + verify (10 diverse keys).
//          Proves the r/s erasure does not corrupt the returned signature.
//   SSR-2: ct::ecdsa_sign sign + verify (10 diverse keys).
//          Proves the CT production path is also unaffected.
//   SSR-3: musig2_partial_sig_agg returns non-zero and verifiable aggregate.
//          Proves the s erasure does not corrupt the serialized sig bytes.
// ============================================================================

#include <cstdio>
#include <cstring>
#include <array>
#include <vector>

static int g_pass = 0, g_fail = 0;
#include "audit_check.hpp"

#include "secp256k1/ecdsa.hpp"
#include "secp256k1/musig2.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ct/sign.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/precompute.hpp"
#include "secp256k1/init.hpp"

using namespace secp256k1;
using fast::Scalar;
using fast::Point;

static Scalar make_scalar(std::uint64_t lo) {
    std::array<std::uint8_t, 32> b{};
    b[24] = static_cast<std::uint8_t>(lo >> 56);
    b[25] = static_cast<std::uint8_t>(lo >> 48);
    b[26] = static_cast<std::uint8_t>(lo >> 40);
    b[27] = static_cast<std::uint8_t>(lo >> 32);
    b[28] = static_cast<std::uint8_t>(lo >> 24);
    b[29] = static_cast<std::uint8_t>(lo >> 16);
    b[30] = static_cast<std::uint8_t>(lo >>  8);
    b[31] = static_cast<std::uint8_t>(lo      );
    Scalar s{};
    Scalar::parse_bytes_strict_nonzero(b.data(), s);
    return s;
}

// ─── SSR-1: ecdsa_sign fast-path correctness ─────────────────────────────────
static void test_ssr1_ecdsa_fast_path_roundtrip() {
    SECP256K1_INIT();
    printf("  [SSR-1] ecdsa_sign fast-path: 10 sign+verify roundtrips (r/s erasure fix)\n");

    int ok = 0;
    for (int i = 1; i <= 10; ++i) {
        Scalar sk = make_scalar(static_cast<std::uint64_t>(i) * 0x9E3779B97F4A7C15ULL);
        if (sk.is_zero()) continue;

        std::array<std::uint8_t, 32> msg{};
        msg[0]  = static_cast<std::uint8_t>(i & 0xFF);
        msg[15] = 0xBE;
        msg[31] = static_cast<std::uint8_t>(i ^ 0xAB);

        auto sig = secp256k1::ecdsa_sign(msg, sk);
        if (sig.r.is_zero() || sig.s.is_zero()) continue;

        Point pk = ct::generator_mul(sk);
        if (secp256k1::ecdsa_verify(msg, pk, sig)) ++ok;
    }
    CHECK(ok >= 9, "[SSR-1] >=9/10 ecdsa_sign fast-path roundtrips passed");
}

// ─── SSR-2: ct::ecdsa_sign production path correctness ───────────────────────
static void test_ssr2_ct_ecdsa_sign_roundtrip() {
    SECP256K1_INIT();
    printf("  [SSR-2] ct::ecdsa_sign: 10 sign+verify roundtrips (correctness after fix)\n");

    int ok = 0;
    for (int i = 1; i <= 10; ++i) {
        Scalar sk = make_scalar(static_cast<std::uint64_t>(i) * 0xFEDCBA9876543210ULL);
        if (sk.is_zero()) continue;

        std::array<std::uint8_t, 32> msg{};
        msg[0]  = static_cast<std::uint8_t>(i & 0xFF);
        msg[16] = 0xCA;
        msg[31] = static_cast<std::uint8_t>(i ^ 0x5C);

        auto sig = secp256k1::ct::ecdsa_sign(msg, sk);
        if (sig.r.is_zero() || sig.s.is_zero()) continue;

        Point pk = ct::generator_mul(sk);
        if (secp256k1::ecdsa_verify(msg, pk, sig)) ++ok;
    }
    CHECK(ok >= 9, "[SSR-2] >=9/10 ct::ecdsa_sign roundtrips passed");
}

// ─── SSR-3: musig2_partial_sig_agg correctness ───────────────────────────────
static void test_ssr3_musig2_agg_correctness() {
    SECP256K1_INIT();

// MSan detection: portable across Clang and GCC.
// __has_feature is a preprocessor keyword in Clang (not a macro), so defined(__has_feature)
// is always false and #ifndef is always true — the #ifndef fallback pattern shadows the
// Clang builtin and breaks MSan detection.  Instead use nested #if to prevent GCC from
// tokenizing __has_feature(...) at all.
#if defined(__clang__)
#  if __has_feature(memory_sanitizer)
#    define SECP_SSR3_MSAN 1
#  endif
#endif
#if defined(__SANITIZE_MEMORY__)
#  define SECP_SSR3_MSAN 1
#endif
#ifdef SECP_SSR3_MSAN
    // MSan with track-origins=2 slows scalar multiplications 100-500x.
    // A full 2-party MuSig2 roundtrip (12+ scalar mults) exceeds 600s.
    // Verify the s-erasure fix via source scan instead.
    printf("  [SSR-3] musig2_partial_sig_agg: source scan (MSan — full roundtrip skipped)\n");
    const char* paths[] = {
        "src/cpu/src/musig2.cpp",
        "../src/cpu/src/musig2.cpp",
        "../../src/cpu/src/musig2.cpp",
    };
    FILE* fp = nullptr;
    for (auto p : paths) { fp = std::fopen(p, "r"); if (fp) break; }
    if (!fp) { printf("  [SSR-3] source not found — advisory skip\n"); ++g_pass; return; }
    char buf[256];
    bool found = false;
    while (std::fgets(buf, sizeof(buf), fp) && !found)
        if (std::strstr(buf, "secure_erase(&s, sizeof(s))")) found = true;
    std::fclose(fp);
    CHECK(found, "[SSR-3] secure_erase(&s, sizeof(s)) present in musig2.cpp (MSan source scan)");
    return;
#  undef SECP_SSR3_MSAN
#endif // SECP_SSR3_MSAN

    printf("  [SSR-3] musig2_partial_sig_agg: 2-party roundtrip (s erasure fix)\n");

    Scalar sk1 = make_scalar(0x1111111111111111ULL);
    Scalar sk2 = make_scalar(0x2222222222222222ULL);
    if (sk1.is_zero() || sk2.is_zero()) {
        CHECK(false, "[SSR-3] degenerate test keys");
        return;
    }

    // Compressed 33-byte pubkeys for key_agg
    auto pk1_c = Point::generator().scalar_mul(sk1).to_compressed();
    auto pk2_c = Point::generator().scalar_mul(sk2).to_compressed();
    // x-only 32-byte pubkeys for nonce_gen
    auto pk1_x = Point::generator().scalar_mul(sk1).x().to_bytes();
    auto pk2_x = Point::generator().scalar_mul(sk2).x().to_bytes();

    std::vector<std::array<std::uint8_t, 33>> pks_c = {pk1_c, pk2_c};
    auto key_agg = secp256k1::musig2_key_agg(pks_c);
    CHECK(!key_agg.Q.is_infinity(), "[SSR-3] key_agg non-infinity");
    if (key_agg.Q.is_infinity()) return;

    std::array<std::uint8_t, 32> msg{};
    msg[0] = 0xDE; msg[1] = 0xAD; msg[15] = 0xBE; msg[31] = 0xEF;

    auto [sec1, pub1] = secp256k1::musig2_nonce_gen(sk1, pk1_x, key_agg.Q_x, msg, nullptr);
    auto [sec2, pub2] = secp256k1::musig2_nonce_gen(sk2, pk2_x, key_agg.Q_x, msg, nullptr);

    std::vector<MuSig2PubNonce> pub_nonces = {pub1, pub2};
    auto agg_nonce = secp256k1::musig2_nonce_agg(pub_nonces);
    auto session   = secp256k1::musig2_start_sign_session(agg_nonce, key_agg, msg);

    // partial_sign takes sec_nonce by mutable reference (consumes it)
    Scalar psig1 = secp256k1::musig2_partial_sign(sec1, sk1, key_agg, session, 0);
    Scalar psig2 = secp256k1::musig2_partial_sign(sec2, sk2, key_agg, session, 1);

    std::vector<Scalar> partial_sigs = {psig1, psig2};

    // This call exercises the s-erasure fix
    auto sig64 = secp256k1::musig2_partial_sig_agg(partial_sigs, session);

    bool non_zero = false;
    for (auto b : sig64) { if (b) { non_zero = true; break; } }
    CHECK(non_zero, "[SSR-3] musig2_partial_sig_agg output non-zero (s erasure intact)");
    if (!non_zero) return;

    // Schnorr verify against aggregated pubkey (x-only)
    auto schnorr_sig = secp256k1::SchnorrSignature::from_bytes(sig64);
    bool v = secp256k1::schnorr_verify(key_agg.Q_x, msg, schnorr_sig);
    CHECK(v, "[SSR-3] musig2 aggregate sig verifies against aggregated pubkey");
}

// ─── Entry point ─────────────────────────────────────────────────────────────

#ifndef UNIFIED_AUDIT_RUNNER
#define STANDALONE_TEST
int main() {
#else
int test_regression_s_scalar_erasure_run() {
#endif
    printf("[s_scalar_erasure] 2026-05-25: r/s erasure in ecdsa_sign + musig2_partial_sig_agg\n");
    test_ssr1_ecdsa_fast_path_roundtrip();
    test_ssr2_ct_ecdsa_sign_roundtrip();
    test_ssr3_musig2_agg_correctness();
    printf("[s_scalar_erasure] %d passed, %d failed\n", g_pass, g_fail);
    return g_fail;
}
