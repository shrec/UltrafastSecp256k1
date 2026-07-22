// ============================================================================
// test_regression_nonce_candidate_erase.cpp
// ============================================================================
// Regression coverage for P2-CT-001/002/003/007: nonce candidate scalars
// cand1 and cand2 were not erased after ct::scalar_select in four locations:
//
//   P2-CT-001  rfc6979_nonce (ecdsa.cpp): cand1/cand2 not erased
//   P2-CT-002  rfc6979_nonce_hedged (ecdsa.cpp): cand1/cand2 not erased
//   P2-CT-003  musig2_nonce_gen (musig2.cpp): cand1/cand2 not erased in k1
//              and k2 blocks
//   P2-CT-007  derive_scalar_from_hash (frost.cpp): cand1/cand2 not erased
//
// In each case, ct::scalar_select chose one candidate, but both candidate
// scalars — which hold nonce-derived secret material — remained on the stack
// after return as residue accessible to potential stack-inspection paths.
//
// Fix: secure_erase(&cand1, sizeof(cand1)) and secure_erase(&cand2, sizeof(cand2))
// added immediately after ct::scalar_select, before the existing erase block.
//
// Tests (correctness regression guards — structural CT property enforced by code):
//   NCER-1  200 ECDSA sign+verify round-trips: rfc6979_nonce path correctness.
//   NCER-2  Determinism: same key+msg → same signature (RFC 6979 §3.2).
//   NCER-3  Nonce uniqueness: different messages → different signatures.
//   NCER-4  Hedged nonce: 50 sign+verify round-trips, different aux_rand →
//           different sigs (hedged path correctness).
//   NCER-5  Source-scan: secure_erase(&cand1) and secure_erase(&cand2) present
//           after ct::scalar_select in each of the four source files.
// ============================================================================

#include <cstdio>
#include <cstring>
#include <array>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static int g_pass = 0, g_fail = 0;
#include "audit_check.hpp"

#include "secp256k1/ecdsa.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ct/sign.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/precompute.hpp"
#include "secp256k1/init.hpp"
#include "secp256k1/musig2.hpp"

#if __has_include("secp256k1/frost.hpp")
#  include "secp256k1/frost.hpp"
#  define HAS_FROST 1
#else
#  define HAS_FROST 0
#endif

using namespace secp256k1;
using fast::Scalar;
using fast::Point;

// ── helpers ──────────────────────────────────────────────────────────────────

static Scalar make_scalar_from_u64(uint64_t lo) {
    std::array<uint8_t, 32> b{};
    b[24] = static_cast<uint8_t>(lo >> 56);
    b[25] = static_cast<uint8_t>(lo >> 48);
    b[26] = static_cast<uint8_t>(lo >> 40);
    b[27] = static_cast<uint8_t>(lo >> 32);
    b[28] = static_cast<uint8_t>(lo >> 24);
    b[29] = static_cast<uint8_t>(lo >> 16);
    b[30] = static_cast<uint8_t>(lo >>  8);
    b[31] = static_cast<uint8_t>(lo      );
    Scalar s{};
    (void)Scalar::parse_bytes_strict_nonzero(b.data(), s);
    return s;
}

static std::array<uint8_t, 32> make_msg(uint8_t seed) {
    std::array<uint8_t, 32> m{};
    for (int i = 0; i < 32; ++i) m[i] = static_cast<uint8_t>(seed ^ (uint8_t)i);
    return m;
}

// Repair (issue #335 acceptance repair, round 5): the previous 3-candidate
// prefix list only resolved when CWD happened to be the repo root or one
// level below it. Route through the shared, UFSECP_SOURCE_ROOT-aware
// resolver (audit_check.hpp) so this scan is independent of process CWD.
static std::string read_source_file(const char* name) {
    return audit_read_source_file((std::string("src/cpu/src/") + name).c_str());
}

// ── NCER-1: rfc6979_nonce path — 200 ECDSA sign+verify round-trips ──────────

static void test_rfc6979_nonce_ecdsa_roundtrip() {
    printf("[1] rfc6979_nonce path: 200 ECDSA sign+verify round-trips (P2-CT-001)\n");
    int ok = 0;
    for (int i = 1; i <= 200; ++i) {
        auto sk = make_scalar_from_u64(static_cast<uint64_t>(i) * 0xDEADBEEFULL + 1);
        auto pk = ct::generator_mul(sk);
        auto msg = make_msg(static_cast<uint8_t>(i));
        auto sig = ct::ecdsa_sign(msg, sk);
        if (!sig.r.is_zero() && !sig.s.is_zero()) {
            if (ecdsa_verify(msg.data(), pk, sig)) ++ok;
        }
    }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "rfc6979_nonce: %d/200 sign+verify", ok);
    CHECK(ok == 200, buf);
}

// ── NCER-2: determinism — same key+msg always produces same signature ────────

static void test_rfc6979_nonce_determinism() {
    printf("[2] rfc6979_nonce determinism: same key+msg → same sig (P2-CT-001)\n");
    auto sk = make_scalar_from_u64(0xABCDEF01234567ULL);
    auto msg = make_msg(0x42);
    auto sig1 = ct::ecdsa_sign(msg, sk);
    auto sig2 = ct::ecdsa_sign(msg, sk);
    bool r_eq = (std::memcmp(sig1.r.to_bytes().data(), sig2.r.to_bytes().data(), 32) == 0);
    bool s_eq = (std::memcmp(sig1.s.to_bytes().data(), sig2.s.to_bytes().data(), 32) == 0);
    CHECK(r_eq && s_eq, "rfc6979_nonce: same key+msg produces same signature");
}

// ── NCER-3: nonce uniqueness — different messages produce different sigs ─────

static void test_rfc6979_nonce_uniqueness() {
    printf("[3] rfc6979_nonce uniqueness: different messages → different sigs (P2-CT-001)\n");
    auto sk = make_scalar_from_u64(0x1234567890ABCDEFULL);
    auto msg1 = make_msg(0x01);
    auto msg2 = make_msg(0x02);
    auto sig1 = ct::ecdsa_sign(msg1, sk);
    auto sig2 = ct::ecdsa_sign(msg2, sk);
    bool r_diff = (std::memcmp(sig1.r.to_bytes().data(), sig2.r.to_bytes().data(), 32) != 0);
    CHECK(r_diff, "rfc6979_nonce: different messages produce different nonces");
}

// ── NCER-4: hedged nonce path — 50 sign+verify round-trips ─────────────────

static void test_rfc6979_nonce_hedged_roundtrip() {
    printf("[4] rfc6979_nonce_hedged path: 50 sign+verify + distinct aux_rand (P2-CT-002)\n");
    int ok = 0;
    for (int i = 1; i <= 50; ++i) {
        auto sk = make_scalar_from_u64(static_cast<uint64_t>(i) * 0xFEEDFACEULL + 1);
        auto pk = ct::generator_mul(sk);
        auto msg = make_msg(static_cast<uint8_t>(i));
        std::array<uint8_t, 32> aux1{}, aux2{};
        aux1[0] = static_cast<uint8_t>(i);
        aux2[0] = static_cast<uint8_t>(i + 128);
        auto sig1 = ct::ecdsa_sign_hedged(msg, sk, aux1);
        auto sig2 = ct::ecdsa_sign_hedged(msg, sk, aux2);
        bool verify1 = (!sig1.r.is_zero() && ecdsa_verify(msg.data(), pk, sig1));
        bool verify2 = (!sig2.r.is_zero() && ecdsa_verify(msg.data(), pk, sig2));
        bool distinct = (std::memcmp(sig1.r.to_bytes().data(), sig2.r.to_bytes().data(), 32) != 0);
        if (verify1 && verify2 && distinct) ++ok;
    }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "rfc6979_nonce_hedged: %d/50 verify+distinct", ok);
    CHECK(ok == 50, buf);
}

// ── NCER-5: source-scan — secure_erase(&cand1/cand2) after ct::scalar_select ─

static void test_source_scan_cand_erase() {
    printf("[5] source-scan: secure_erase(&cand1/cand2) after ct::scalar_select (P2-CT-001/002/003/007)\n");

    // ecdsa.cpp: both rfc6979_nonce and rfc6979_nonce_hedged
    // Fail-closed (issue #335 acceptance repair, round 5): these files always
    // exist in-tree. A failed read means the source could not be resolved,
    // NOT that the property holds -- must never be a silent 0-checks skip.
    {
        auto src = read_source_file("ecdsa.cpp");
        CHECK(!src.empty(), "ecdsa.cpp must be readable (in-tree source always exists)");
        if (!src.empty()) {
            // Count occurrences of secure_erase(&cand1 — expect at least 2 (one per function)
            int count1 = 0, count2 = 0;
            std::string::size_type pos = 0;
            while ((pos = src.find("secure_erase(&cand1", pos)) != std::string::npos) {
                ++count1; pos += 1;
            }
            pos = 0;
            while ((pos = src.find("secure_erase(&cand2", pos)) != std::string::npos) {
                ++count2; pos += 1;
            }
            CHECK(count1 >= 2, "ecdsa.cpp: secure_erase(&cand1) appears >= 2× (P2-CT-001/002)");
            CHECK(count2 >= 2, "ecdsa.cpp: secure_erase(&cand2) appears >= 2× (P2-CT-001/002)");
        }
    }

    // musig2.cpp: k1 and k2 blocks
    {
        auto src = read_source_file("musig2.cpp");
        CHECK(!src.empty(), "musig2.cpp must be readable (in-tree source always exists)");
        if (!src.empty()) {
            int count1 = 0, count2 = 0;
            std::string::size_type pos = 0;
            while ((pos = src.find("secure_erase(&cand1", pos)) != std::string::npos) {
                ++count1; pos += 1;
            }
            pos = 0;
            while ((pos = src.find("secure_erase(&cand2", pos)) != std::string::npos) {
                ++count2; pos += 1;
            }
            CHECK(count1 >= 2, "musig2.cpp: secure_erase(&cand1) appears >= 2× (k1+k2 blocks, P2-CT-003)");
            CHECK(count2 >= 2, "musig2.cpp: secure_erase(&cand2) appears >= 2× (k1+k2 blocks, P2-CT-003)");
        }
    }

    // frost.cpp: derive_scalar_from_hash
    {
        auto src = read_source_file("frost.cpp");
        CHECK(!src.empty(), "frost.cpp must be readable (in-tree source always exists)");
        if (!src.empty()) {
            bool has_cand1 = (src.find("secure_erase(&cand1") != std::string::npos);
            bool has_cand2 = (src.find("secure_erase(&cand2") != std::string::npos);
            CHECK(has_cand1, "frost.cpp: secure_erase(&cand1) present (P2-CT-007)");
            CHECK(has_cand2, "frost.cpp: secure_erase(&cand2) present (P2-CT-007)");
        }
    }
}

// ── entry point ──────────────────────────────────────────────────────────────

int test_regression_nonce_candidate_erase_run() {
    g_pass = 0; g_fail = 0;
    printf("======================================================================\n");
    printf("  Regression: nonce candidate scalar zeroization (P2-CT-001/002/003/007)\n");
    printf("  Fixes: cand1+cand2 secure_erase after ct::scalar_select in\n");
    printf("         rfc6979_nonce, rfc6979_nonce_hedged, musig2_nonce_gen (k1/k2),\n");
    printf("         derive_scalar_from_hash\n");
    printf("======================================================================\n\n");

    test_rfc6979_nonce_ecdsa_roundtrip();
    printf("\n");
    test_rfc6979_nonce_determinism();
    printf("\n");
    test_rfc6979_nonce_uniqueness();
    printf("\n");
    test_rfc6979_nonce_hedged_roundtrip();
    printf("\n");
    test_source_scan_cand_erase();
    printf("\n");

    printf("[regression_nonce_candidate_erase] %d/%d checks passed\n",
           g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_nonce_candidate_erase_run(); }
#endif
