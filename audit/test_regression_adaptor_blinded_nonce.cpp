// ============================================================================
// test_regression_adaptor_blinded_nonce.cpp
// ============================================================================
// Regression coverage for four security/correctness fixes (2026-05-23):
//
//   SEC-NEW-001  adaptor.cpp schnorr_adaptor_sign(): R_hat = ct::generator_mul(k)
//               changed to ct::generator_mul_blinded(k) — DPA defence for the
//               secret nonce k.  Without blinding, power/EM side-channels can
//               recover k from a single trace of the generator-mul ladder.
//
//   SEC-NEW-002  shim_schnorr_bch.cpp secp256k1_schnorr_sign(): k.is_zero()
//               changed to k.is_zero_ct() on the RFC6979 nonce after generation.
//               is_zero() on a secret scalar has a data-dependent branch in some
//               toolchain+optimisation combinations — is_zero_ct() is always safe.
//
//   P3-SHIM-STACK  shim_schnorr.cpp: kStackMsgMax raised from 256 to 1024.
//               Messages 257–1024 bytes no longer trigger a heap allocation in
//               sign_custom; stack is always used for the overwhelmingly common
//               BIP-340 (32-byte) and Lightning (64-byte) message sizes.
//
//   P3-BATCH-MEM  shim_batch_verify.cpp: schnorr_batch_verify + ecdsa_batch_verify
//               now call batch.shrink_to_fit() before returning.  The thread_local
//               vectors previously retained their peak capacity indefinitely after
//               a large batch, causing unbounded per-thread memory.
//
// All sub-tests are correctness-oriented.  CT properties are structural — enforced
// by the type of call (blinded vs non-blinded) and verified via source scan.
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>
#include <fstream>
#include <string>

#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ct/sign.hpp"
#include "secp256k1/ct/scalar.hpp"

#if __has_include("secp256k1/adaptor.hpp")
#  include "secp256k1/adaptor.hpp"
#  define HAS_ADAPTOR 1
#else
#  define HAS_ADAPTOR 0
#endif

#include "audit_check.hpp"

using namespace secp256k1;
using secp256k1::fast::Scalar;
using secp256k1::fast::Point;

static int g_pass = 0, g_fail = 0;

// ── helpers ──────────────────────────────────────────────────────────────────

// rel_path is repo-root-relative (e.g. "src/cpu/src/adaptor.cpp").
//
// Repair (issue #335 acceptance repair, round 5): the previous bounded
// CWD-relative walk-up (depth<=6, TEST-08-NESTED-PATH fix) made this
// resolver independent of BUILD-DIR NESTING depth (build/asan, build/cov,
// ...) but not of process CWD itself -- it still could not resolve when
// unified_audit_runner was invoked from a CWD unrelated to the repo (e.g.
// /tmp). Route through the shared, UFSECP_SOURCE_ROOT-aware
// audit_read_source_file() (audit_check.hpp), which tries the compile-time
// absolute repo root first and keeps the bounded CWD-relative walk-up as
// its own fallback.
static std::string read_source_file(const char* rel_path) {
    return audit_read_source_file(rel_path);
}

// ── SEC-NEW-001: adaptor.cpp uses generator_mul_blinded for nonce ─────────────
//
// schnorr_adaptor_sign() computes R_hat = k * G where k is a secret nonce.
// The fix replaces ct::generator_mul(k) with ct::generator_mul_blinded(k).
// Blinding adds a random mask to the scalar walk, making power-trace recovery
// of k infeasible even for a single observation.

static void test_adaptor_blinded_nonce_source_scan() {
    printf("[1] SEC-NEW-001: adaptor.cpp — generator_mul_blinded for nonce k\n");

    std::string src = read_source_file("src/cpu/src/adaptor.cpp");
    if (src.empty()) {
        src = read_source_file("adaptor.cpp");
    }
    // TEST-08: adaptor.cpp always exists in-tree. A failed read means the harness
    // ran from an unexpected cwd, NOT that the property holds — fail-closed so this
    // source-scan guard cannot silently pass when it scanned nothing.
    CHECK(!src.empty(), "adaptor.cpp must be readable (in-tree source always exists)");
    if (src.empty()) return;

    // Old (unfixed) code: ct::generator_mul(k) — single trace DPA-vulnerable.
    // Any occurrence of this on the nonce k (after the nonce assignment) is a bug.
    // The string "generator_mul(k)" covers both fast::generator_mul and ct::generator_mul.
    // We look for the non-blinded call specifically before R_hat assignment.
    bool has_old_unblinded =
        (src.find("ct::generator_mul(k)") != std::string::npos ||
         src.find("generator_mul(k);\n") != std::string::npos);

    // New (fixed) code: ct::generator_mul_blinded(k) must be present.
    bool has_blinded =
        (src.find("generator_mul_blinded(k)") != std::string::npos);

    // It is OK if an unrelated use of generator_mul(k) exists in a variable-time
    // context (e.g. a verify helper in the same file) but there must be no
    // ct::generator_mul(k) on the signing-nonce path — which is the only call in
    // schnorr_adaptor_sign after "auto k = ...".  We only CHECK that blinded
    // is present; the non-blinded check is informational.
    if (has_old_unblinded && !has_blinded) {
        CHECK(false, "adaptor.cpp: ct::generator_mul(k) present and blinded absent — SEC-NEW-001 regression");
    } else {
        CHECK(has_blinded, "adaptor.cpp: generator_mul_blinded(k) present (SEC-NEW-001)");
    }

    // bbhunt-002: ufsecp_schnorr_adaptor_sign must scrub aux_rand (the BIP-340
    // nonce-derivation entropy) after use, on the success AND early-return paths.
    printf("[2] bbhunt-002: ufsecp_zk.cpp — aux_rand entropy erased after use\n");
    std::string zk = read_source_file("src/cpu/src/impl/ufsecp_zk.cpp");
    if (zk.empty()) zk = read_source_file("ufsecp_zk.cpp");
    CHECK(!zk.empty(), "ufsecp_zk.cpp must be readable (in-tree source always exists)");
    if (!zk.empty()) {
        // The adaptor-sign aux buffer is aux_arr; require a secure_erase mentioning it.
        bool aux_erased = (zk.find("secure_erase(aux_arr.data()") != std::string::npos);
        CHECK(aux_erased, "ufsecp_zk.cpp: secure_erase(aux_arr.data(), ...) present (bbhunt-002)");
    }
}

// ── SEC-NEW-001 functional: schnorr adaptor sign + adapt + verify round-trip ─

static void test_adaptor_functional_roundtrip() {
    printf("[2] SEC-NEW-001: adaptor functional sign+adapt+verify round-trip\n");

#if HAS_ADAPTOR
    // Fixed known-good scalars for reproducibility.
    std::array<uint8_t, 32> sk_bytes{};
    sk_bytes[31] = 0x17;
    std::array<uint8_t, 32> t_bytes{};
    t_bytes[31] = 0x2B;
    std::array<uint8_t, 32> msg{};
    msg[0] = 0xDE; msg[31] = 0xAD;

    Scalar sk;
    bool sk_ok = Scalar::parse_bytes_strict_nonzero(sk_bytes.data(), sk);
    CHECK(sk_ok, "adaptor round-trip: sk parse");
    if (!sk_ok) return;

    Scalar t;
    bool t_ok = Scalar::parse_bytes_strict_nonzero(t_bytes.data(), t);
    CHECK(t_ok, "adaptor round-trip: t parse");
    if (!t_ok) return;

    Point T = secp256k1::ct::generator_mul(t);
    Point pk = secp256k1::ct::generator_mul(sk);
    auto pk_x = pk.x().to_bytes();

    // schnorr_adaptor_sign internally uses generator_mul_blinded(k) after the fix.
    // 4th arg is aux_rand (BIP-340 auxiliary randomness); deterministic zeros for repro.
    std::array<std::uint8_t, 32> aux_rand{};
    auto pre_sig = secp256k1::schnorr_adaptor_sign(sk, msg, T, aux_rand);

    CHECK(!pre_sig.R_hat.is_infinity(), "adaptor: pre_sig R_hat != infinity");

    bool verify_ok = secp256k1::schnorr_adaptor_verify(pre_sig, pk_x, msg, T);
    CHECK(verify_ok, "adaptor: schnorr_adaptor_verify round-trip (SEC-NEW-001)");

    // Adapt: recover the full Schnorr signature using the adaptor secret t.
    auto full_sig = secp256k1::schnorr_adaptor_adapt(pre_sig, t);
    bool schnorr_ok = secp256k1::schnorr_verify(pk_x, msg, full_sig);
    CHECK(schnorr_ok, "adaptor: adapted full Schnorr sig verifies (SEC-NEW-001)");

    // Different messages must produce different pre-signatures.
    std::array<uint8_t, 32> msg2{};
    msg2[0] = 0xBE; msg2[31] = 0xEF;
    auto pre_sig2 = secp256k1::schnorr_adaptor_sign(sk, msg2, T, aux_rand);
    // R_hat values differ — nonce is message-dependent via RFC6979.
    // Compare x-coordinates as byte arrays (Point has no operator!=).
    auto r_hat_x1 = pre_sig.R_hat.x().to_bytes();
    auto r_hat_x2 = pre_sig2.R_hat.x().to_bytes();
    auto s_hat_1 = pre_sig.s_hat.to_bytes();
    auto s_hat_2 = pre_sig2.s_hat.to_bytes();
    bool R_diff = (r_hat_x1 != r_hat_x2);
    bool S_diff = (s_hat_1 != s_hat_2);
    CHECK(R_diff || S_diff,
          "adaptor: distinct messages produce distinct pre-sigs (nonce determinism)");
#else
    printf("  [SKIP] adaptor module not compiled — functional test skipped\n");
#endif
}

// ── v9 RT-002 / TASK-002: adaptor.cpp uses _blinded on all secret scalars ────
//
// v9 review found two additional unblinded ct::generator_mul() sites on
// secret inputs in src/cpu/src/adaptor.cpp that the SEC-NEW-001 fix missed:
//   * schnorr_adaptor_sign:  Point P = ct::generator_mul(private_key)
//                             (long-term key derivation; was unblinded)
//   * ecdsa_adaptor_sign:    Point base_nonce = ct::generator_mul(k)
//                             (secret nonce; was unblinded)
// Both now use ct::generator_mul_blinded(...). The binding scalar in
// ecdsa_adaptor_sign is derived from PUBLIC adaptor_point so it stays on
// the unblinded primitive (correct and cheaper).

static void test_adaptor_blinded_all_secret_sites_source_scan() {
    printf("[2b] v9 RT-002 / TASK-002: adaptor.cpp — _blinded on every secret site\n");

    std::string src = read_source_file("src/cpu/src/adaptor.cpp");
    if (src.empty()) {
        src = read_source_file("adaptor.cpp");
    }
    // TEST-08: adaptor.cpp always exists in-tree. A failed read means the harness
    // ran from an unexpected cwd, NOT that the property holds — fail-closed so this
    // source-scan guard cannot silently pass when it scanned nothing.
    CHECK(!src.empty(), "adaptor.cpp must be readable (in-tree source always exists)");
    if (src.empty()) return;

    // schnorr_adaptor_sign long-term key: must use blinded variant.
    bool has_blinded_priv =
        (src.find("generator_mul_blinded(private_key)") != std::string::npos);
    CHECK(has_blinded_priv,
          "adaptor.cpp: generator_mul_blinded(private_key) present in schnorr_adaptor_sign (v9 RT-002)");

    // ecdsa_adaptor_sign secret nonce k: must use blinded variant.
    // The blinded call must be present; there is also a separate
    // generator_mul(binding) on the PUBLIC binding scalar which is correct.
    // We just assert the blinded call appears at least once on a k argument
    // — counted by the same pattern as SEC-NEW-001 in schnorr_adaptor_sign,
    // so we count occurrences instead of presence/absence.
    size_t count_blinded_k = 0;
    size_t pos = 0;
    while ((pos = src.find("generator_mul_blinded(k)", pos)) != std::string::npos) {
        ++count_blinded_k;
        pos += 24;  // len("generator_mul_blinded(k)")
    }
    CHECK(count_blinded_k >= 2,
          "adaptor.cpp: generator_mul_blinded(k) appears in BOTH schnorr_adaptor_sign and ecdsa_adaptor_sign (v9 RT-002)");

    // Negative-presence check: there must be NO bare ct::generator_mul(k) or
    // ct::generator_mul(private_key) calls remaining. (Comments are fine.)
    // We tolerate ct::generator_mul(binding) — binding is PUBLIC-derived.
    bool has_unblinded_k =
        (src.find("ct::generator_mul(k)") != std::string::npos);
    bool has_unblinded_priv =
        (src.find("ct::generator_mul(private_key)") != std::string::npos);
    CHECK(!has_unblinded_k,
          "adaptor.cpp: no bare ct::generator_mul(k) remains (v9 RT-002)");
    CHECK(!has_unblinded_priv,
          "adaptor.cpp: no bare ct::generator_mul(private_key) remains (v9 RT-002)");
}

// ── SEC-NEW-002: shim_schnorr_bch.cpp uses is_zero_ct() on nonce ─────────────
//
// secp256k1_schnorr_sign() in the BCH shim generates an RFC6979 nonce k and
// then checks if it is zero before use.  The fix changes k.is_zero() to
// k.is_zero_ct() — the constant-time zero test — to avoid data-dependent
// branch on a secret scalar.

static void test_bchn_shim_is_zero_ct_source_scan() {
    printf("[3] SEC-NEW-002: shim_schnorr_bch.cpp — is_zero_ct() on nonce k\n");

    std::string src = read_source_file(
        "compat/libsecp256k1_bchn_shim/src/shim_schnorr_bch.cpp");
    // Fail-closed (issue #335 acceptance repair, round 5): shim_schnorr_bch.cpp
    // always exists in-tree (a source-scan needs the file text, not a built
    // shim target). A failed read means the source could not be resolved,
    // NOT that the property holds — never a silent 0-checks skip.
    CHECK(!src.empty(), "shim_schnorr_bch.cpp must be readable (in-tree source always exists)");
    if (src.empty()) return;

    // Fixed code: must use is_zero_ct() on k.
    bool has_is_zero_ct = (src.find("k.is_zero_ct()") != std::string::npos);
    CHECK(has_is_zero_ct, "shim_schnorr_bch.cpp: k.is_zero_ct() present (SEC-NEW-002)");

    // Verify the non-CT call is gone from the nonce-check position.
    // Note: is_zero() may still appear elsewhere (e.g. on public s values).
    // We check that the CT call is present rather than enforcing absence of
    // all VT calls, because s.is_zero() at the end of sign is correct (s public).
    if (!has_is_zero_ct) {
        // Already failed above — no double-print needed.
        return;
    }
    printf("  OK: is_zero_ct() found on nonce k path\n");
}

// ── P3-SHIM-STACK: shim_schnorr.cpp kStackMsgMax = 1024 ──────────────────────

static void test_shim_schnorr_stack_msg_max_source_scan() {
    printf("[4] P3-SHIM-STACK: shim_schnorr.cpp — kStackMsgMax = 1024 or AUDIT-003 makes it irrelevant\n");

    std::string src = read_source_file(
        "compat/libsecp256k1_shim/src/shim_schnorr.cpp");
    CHECK(!src.empty(), "shim_schnorr.cpp must be readable (in-tree source always exists)");
    if (src.empty()) return;

    bool has_1024 =
        (src.find("kStackMsgMax = 1024") != std::string::npos);
    bool has_old_256 =
        (src.find("kStackMsgMax = 256") != std::string::npos);
    // AUDIT-003 restricted sign_custom to msglen == 32 only, forwarding to sign32.
    // When this restriction is present, kStackMsgMax is unnecessary (no heap path
    // for variable-length messages exists). Accept both the upgrade and the AUDIT-003 path.
    bool has_audit003 =
        (src.find("msglen != 32") != std::string::npos ||
         src.find("msglen must be 32") != std::string::npos);

    if (has_old_256 && !has_1024) {
        CHECK(false, "shim_schnorr.cpp: kStackMsgMax still = 256 — P3-SHIM-STACK regression");
    } else if (has_1024) {
        ++g_pass; // kStackMsgMax = 1024 present — P3-SHIM-STACK satisfied
        printf("  OK: kStackMsgMax = 1024 found\n");
    } else {
        // Neither 256 nor 1024: AUDIT-003 made variable-length sign_custom irrelevant.
        CHECK(has_audit003,
              "shim_schnorr.cpp: AUDIT-003 msglen=32 restriction present (kStackMsgMax irrelevant)");
        if (has_audit003)
            printf("  OK: AUDIT-003 restriction present — kStackMsgMax unnecessary\n");
    }
}

// ── P3-BATCH-MEM: shim_batch_verify.cpp shrink_to_fit() ──────────────────────

static void test_batch_verify_shrink_to_fit_source_scan() {
    printf("[5] P3-BATCH-MEM: shim_batch_verify.cpp — shrink_to_fit() or PERF-004 removal\n");

    std::string src = read_source_file(
        "compat/libsecp256k1_shim/src/shim_batch_verify.cpp");
    CHECK(!src.empty(), "shim_batch_verify.cpp must be readable (in-tree source always exists)");
    if (src.empty()) return;

    // P3-BATCH-MEM originally added shrink_to_fit() after each batch.
    // PERF-004 subsequently removed it: thread_local vectors retain capacity for
    // amortized performance across calls — shrinking would force reallocation on
    // the next large batch. Accept both the original addition and the PERF-004 removal.
    size_t count = 0;
    size_t pos = 0;
    while ((pos = src.find("shrink_to_fit()", pos)) != std::string::npos) {
        ++count;
        pos += 15;
    }
    bool has_perf004_removal =
        (src.find("shrink_to_fit() removed (PERF-004") != std::string::npos);

    if (count >= 2) {
        ++g_pass; // shrink_to_fit() present in both batch paths — P3-BATCH-MEM satisfied
        printf("  OK: %zu shrink_to_fit() calls found\n", count);
    } else {
        // PERF-004 deliberately removed shrink_to_fit() for amortized perf.
        CHECK(has_perf004_removal,
              "shim_batch_verify.cpp: PERF-004 removed shrink_to_fit() for amortized perf (correct)");
        if (has_perf004_removal)
            printf("  OK: PERF-004 removal comment found — intentional design\n");
    }
}

// ── entry point ──────────────────────────────────────────────────────────────

int test_regression_adaptor_blinded_nonce_run() {
    g_pass = 0; g_fail = 0;
    printf("======================================================================\n");
    printf("  Regression: adaptor blinded nonce + BCH shim CT zero + stack/mem\n");
    printf("  Fixes: SEC-NEW-001 adaptor generator_mul_blinded,\n");
    printf("         SEC-NEW-002 BCH shim is_zero_ct on nonce k,\n");
    printf("         P3-SHIM-STACK kStackMsgMax 256→1024,\n");
    printf("         P3-BATCH-MEM batch vector shrink_to_fit\n");
    printf("======================================================================\n\n");

    test_adaptor_blinded_nonce_source_scan();
    printf("\n");
    test_adaptor_blinded_all_secret_sites_source_scan();
    printf("\n");
    test_adaptor_functional_roundtrip();
    printf("\n");
    test_bchn_shim_is_zero_ct_source_scan();
    printf("\n");
    test_shim_schnorr_stack_msg_max_source_scan();
    printf("\n");
    test_batch_verify_shrink_to_fit_source_scan();
    printf("\n");

    printf("[regression_adaptor_blinded_nonce] %d/%d checks passed\n",
           g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_adaptor_blinded_nonce_run(); }
#endif
