// ============================================================================
// test_regression_secret_scalar_residue_erase.cpp
// ============================================================================
// Regression coverage for FROST-SIGN-RESIDUE (2026-06-10): secret-derived
// intermediate Scalar products were left on the stack without secure_erase.
// Same class as T08-SCALAR-ERASE (ecdsa_sign / musig2_partial_sig_agg).
//
//   frost.cpp  frost_sign(): rho_ei = my_binding·ei and lambda_s_e =
//              lambda_i·s_i·e carry the SECRET binding nonce ei and signing
//              share s_i. The function erased d/ei/s_i but NOT these two
//              products — leaving secret-derived material as stack residue.
//              Fix: secure_erase(&rho_ei) + secure_erase(&lambda_s_e).
//
//   ct_sign.cpp / schnorr.cpp  schnorr_keypair_create(): the local d_prime is
//              a private-key copy. kp.d (the public x-only signing key) is the
//              intended output, but d_prime itself was never scrubbed.
//              Fix: secure_erase(&d_prime) before return.
//
// These are secret-erasure (stack-scrubbing) hygiene fixes, not timing fixes —
// all arithmetic is already branchless ct::. The structural property (a
// secure_erase call exists for each secret-derived local) is verified by source
// scan, mirroring the T08 / adaptor-blinded-nonce regression style. A functional
// schnorr keypair_create + sign + verify round-trip confirms the added erase of
// d_prime did not corrupt the returned keypair.
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>
#include <fstream>
#include <string>

#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/ct/scalar.hpp"

#include "audit_check.hpp"

using namespace secp256k1;
using secp256k1::fast::Scalar;
using secp256k1::fast::Point;

static int g_pass = 0, g_fail = 0;

// ── helper: CWD-independent in-tree source resolution ────────────────────────
// Repair (issue #335 acceptance repair, round 5): route through the shared,
// UFSECP_SOURCE_ROOT-aware audit_read_source_file() (audit_check.hpp) so this
// resolves identically whether invoked from the repo root or from a CWD
// unrelated to the repo (e.g. /tmp) -- the previous bounded CWD-relative
// walk-up (depth<=6, matches test_regression_adaptor_blinded_nonce's
// pre-repair helper) hard-failed correctly in the latter case but did not
// resolve the source, so real check counts diverged between the two
// invocations.
static std::string read_source_file(const char* rel_path) {
    return audit_read_source_file(rel_path);
}

// ── FROST-SIGN-RESIDUE: frost_sign erases rho_ei and lambda_s_e ──────────────
static void test_frost_sign_residue_erase_source_scan() {
    printf("[1] FROST-SIGN-RESIDUE: frost.cpp — rho_ei / lambda_s_e erased\n");

    std::string src = read_source_file("src/cpu/src/frost.cpp");
    if (src.empty()) src = read_source_file("frost.cpp");
    // Fail-closed: frost.cpp always exists in-tree. A failed read means the
    // harness ran from an unexpected cwd, NOT that the property holds.
    CHECK(!src.empty(), "frost.cpp must be readable (in-tree source always exists)");
    if (src.empty()) return;

    // The two secret-derived products must each be scrubbed.
    bool rho_erased    = (src.find("secure_erase(&rho_ei") != std::string::npos);
    bool lambda_erased = (src.find("secure_erase(&lambda_s_e") != std::string::npos);
    CHECK(rho_erased,    "frost.cpp: secure_erase(&rho_ei, ...) present (FROST-SIGN-RESIDUE)");
    CHECK(lambda_erased, "frost.cpp: secure_erase(&lambda_s_e, ...) present (FROST-SIGN-RESIDUE)");

    // Guard against regression to const-qualified (un-erasable) declarations.
    bool no_const_rho =
        (src.find("Scalar const rho_ei") == std::string::npos);
    bool no_const_lambda =
        (src.find("Scalar const lambda_s_e") == std::string::npos);
    CHECK(no_const_rho,
          "frost.cpp: rho_ei is non-const so it can be erased (FROST-SIGN-RESIDUE)");
    CHECK(no_const_lambda,
          "frost.cpp: lambda_s_e is non-const so it can be erased (FROST-SIGN-RESIDUE)");
}

// ── FROST-SIGN-RESIDUE: schnorr_keypair_create erases d_prime ────────────────
static void test_keypair_create_dprime_erase_source_scan() {
    printf("[2] FROST-SIGN-RESIDUE: ct_sign.cpp + schnorr.cpp — d_prime erased\n");

    for (const char* path : {"src/cpu/src/ct_sign.cpp", "src/cpu/src/schnorr.cpp"}) {
        std::string src = read_source_file(path);
        if (src.empty()) {
            // Try a bare-filename fallback (flat build dirs sometimes copy sources).
            const char* slash = std::strrchr(path, '/');
            src = read_source_file(slash ? slash + 1 : path);
        }
        CHECK(!src.empty(), "keypair source must be readable (in-tree source always exists)");
        if (src.empty()) continue;

        bool dprime_erased = (src.find("secure_erase(&d_prime") != std::string::npos);
        CHECK(dprime_erased,
              "schnorr_keypair_create: secure_erase(&d_prime, ...) present (FROST-SIGN-RESIDUE)");
    }
}

// ── MUSIG2 residue: musig2_partial_sign erases neg_k / neg_d / ead ────────────
// Found by the improved dev_bug_scanner secret-derived-unerased check (same class
// as the frost residue): neg_k = -k, neg_d = -d, and ead = ea*d all carry secret
// nonce/key material and must be scrubbed, not just k/d.
static void test_musig2_partial_sign_residue_erase_source_scan() {
    printf("[2b] MUSIG2-RESIDUE: musig2.cpp — neg_k / neg_d / ead erased\n");
    std::string src = read_source_file("src/cpu/src/musig2.cpp");
    if (src.empty()) src = read_source_file("musig2.cpp");
    CHECK(!src.empty(), "musig2.cpp must be readable (in-tree source always exists)");
    if (src.empty()) return;
    CHECK(src.find("secure_erase(&neg_k") != std::string::npos,
          "musig2.cpp: secure_erase(&neg_k, ...) present (secret -k residue)");
    CHECK(src.find("secure_erase(&neg_d") != std::string::npos,
          "musig2.cpp: secure_erase(&neg_d, ...) present (secret -d residue)");
    CHECK(src.find("secure_erase(&ead") != std::string::npos,
          "musig2.cpp: secure_erase(&ead, ...) present (secret ea*d residue)");
}

// ── Functional: keypair_create + sign + verify still round-trips ─────────────
// Confirms erasing d_prime did not corrupt kp.d (the returned x-only signing key).
static void test_keypair_create_functional_roundtrip() {
    printf("[3] FROST-SIGN-RESIDUE: schnorr keypair_create + sign + verify round-trip\n");

    std::array<uint8_t, 32> sk_bytes{};
    sk_bytes[31] = 0x29;  // fixed non-zero key for reproducibility
    std::array<uint8_t, 32> msg{};
    msg[0] = 0xC0; msg[31] = 0xDE;
    std::array<uint8_t, 32> aux_rand{};

    Scalar sk;
    bool sk_ok = Scalar::parse_bytes_strict_nonzero(sk_bytes.data(), sk);
    CHECK(sk_ok, "keypair round-trip: sk parse");
    if (!sk_ok) return;

    SchnorrKeypair kp = schnorr_keypair_create(sk);
    // kp.d (secret signing scalar) must be intact and non-zero after the d_prime erase.
    CHECK(!kp.d.is_zero_ct(), "keypair round-trip: kp.d intact (non-zero) after d_prime erase");

    auto sig = schnorr_sign(kp, msg, aux_rand);
    bool verify_ok = schnorr_verify(kp.px, msg, sig);
    CHECK(verify_ok, "keypair round-trip: sign+verify succeeds after d_prime erase");
}

// ── entry point ──────────────────────────────────────────────────────────────
int test_regression_secret_scalar_residue_erase_run() {
    g_pass = 0; g_fail = 0;
    printf("======================================================================\n");
    printf("  Regression: secret-derived scalar stack-residue erasure\n");
    printf("  Fix: FROST-SIGN-RESIDUE — frost_sign rho_ei/lambda_s_e +\n");
    printf("       schnorr_keypair_create d_prime secure_erase\n");
    printf("======================================================================\n\n");

    test_frost_sign_residue_erase_source_scan();
    printf("\n");
    test_keypair_create_dprime_erase_source_scan();
    printf("\n");
    test_musig2_partial_sign_residue_erase_source_scan();
    printf("\n");
    test_keypair_create_functional_roundtrip();
    printf("\n");

    printf("[regression_secret_scalar_residue_erase] %d/%d checks passed\n",
           g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_secret_scalar_residue_erase_run(); }
#endif
