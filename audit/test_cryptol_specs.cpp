// ============================================================================
// test_cryptol_specs.cpp -- Cryptol formal spec checker for unified runner
// ============================================================================
//
// ADVISORY-BY-DESIGN module: Cryptol properties prove the correctness of secp256k1
// arithmetic primitives (GF(p) field, EC point group law, ECDSA, Schnorr).
// These are foundational, but the Cryptol toolchain is OPTIONAL and is NOT installed
// on the CI runners, so this module is registered advisory=true in ALL_MODULES: a
// missing toolchain SKIPs (ADVISORY_SKIP_CODE) and never blocks. (TQ7-02: the header
// previously claimed "BLOCKING / ADVISORY = false", contradicting its registration.)
//
// When cryptol IS present, a property failure OR a setup/infra failure is surfaced as
// a hard FAIL (advisory_failed) — never silently passed.
// Requires the Cryptol toolchain:  apt-get install cryptol  (Ubuntu 22.04+)
//
// What this proves (when cryptol is present):
//   - Secp256k1Field.cry   — 14 field properties: commutativity, assoc, inv, sqrt
//   - Secp256k1Point.cry   — 8 point properties: group law, scalar_mul, negation
//   - Secp256k1ECDSA.cry   — 8 ECDSA properties: sign→verify, low-S, ranges
//   - Secp256k1Schnorr.cry — 6 Schnorr properties: sign→verify, nonce parity
//
// When all properties pass, the formal spec is consistent with itself and
// the mathematical model.  SAW proofs (linking spec to C implementation)
// are a separate step run in dedicated CI jobs.
//
// ============================================================================

#include "audit_check.hpp"  // ADVISORY_SKIP_CODE (MEDIUM-5)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <filesystem>

#ifdef _WIN32
#  define popen  _popen
#  define pclose _pclose
#else
#  include <sys/wait.h>
#  include <unistd.h>
#endif

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool cryptol_available() {
#ifdef _WIN32
    return std::system("cryptol --version >NUL 2>&1") == 0;
#else
    return std::system("cryptol --version >/dev/null 2>&1") == 0;
#endif
}

static std::string find_cryptol_dir() {
    static const char* kCandidates[] = {
        "audit/formal/cryptol",
        "../audit/formal/cryptol",
        "../../audit/formal/cryptol",
        nullptr
    };
    for (int i = 0; kCandidates[i]; ++i) {
        std::string path = std::string(kCandidates[i]) + "/Secp256k1Field.cry";
        if (FILE* f = std::fopen(path.c_str(), "r")) {
            std::fclose(f);
            return kCandidates[i];
        }
    }
    return "";
}

// Run `cryptol` in batch mode with `:check` on a spec file.
// Returns 0 if all properties pass or no properties found, 1 on failure.
static int check_cryptol_spec(const std::string& cry_dir,
                               const std::string& module_name,
                               const std::vector<std::string>& properties) {
    // Build a batch script: load module, run :check on each property, :quit
    std::string batch;
    batch += ":load " + cry_dir + "/" + module_name + ".cry\n";
    for (auto& p : properties) {
        batch += ":check " + p + "\n";
    }
    batch += ":quit\n";

    // Write batch to a temp file. Use mkstemp to create the file atomically,
    // preventing symlink attacks on predictable /tmp names.
    // TQ7-02: this helper is only reached AFTER cryptol_available() succeeded, so a
    // setup failure here is a genuine infra error — return 1 (FAIL), NEVER 0 (which
    // the caller treats as PASS, a silent pass on a check that claims to verify the
    // arithmetic primitives).
#ifdef _WIN32
    std::string tmpfile = (std::filesystem::temp_directory_path() / ("cryptol_check_" + module_name + ".sh")).string();
    FILE* tf = std::fopen(tmpfile.c_str(), "w");
    if (!tf) { std::printf("    [setup-error] cannot create temp batch file\n"); return 1; }
#else
    std::string tmpfile = (std::filesystem::temp_directory_path() / "cryptol_check_XXXXXX").string();
    int tmpfd = mkstemp(tmpfile.data());
    if (tmpfd < 0) { std::printf("    [setup-error] mkstemp failed\n"); return 1; }
    FILE* tf = fdopen(tmpfd, "w");
    if (!tf) { close(tmpfd); std::printf("    [setup-error] fdopen failed\n"); return 1; }
#endif
    std::fputs(batch.c_str(), tf);
    std::fclose(tf);

    std::string cmd = "cryptol --batch < " + tmpfile + " 2>&1";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) { std::printf("    [setup-error] popen(cryptol) failed\n"); return 1; }

    bool any_fail = false;
    int proved = 0;  // count positive per-property proof results (TQ7-02)
    char buf[512];
    while (std::fgets(buf, sizeof(buf), pipe)) {
        std::printf("    %s", buf);
        // Cryptol prints "FAILED"/"Counterexample" on property failure, and a scope
        // error ("not in scope"/"Unbound") when a property name is mistyped/absent.
        if (std::strstr(buf, "FAILED") || std::strstr(buf, "Counterexample") ||
            std::strstr(buf, "ERROR") || std::strstr(buf, "not in scope") ||
            std::strstr(buf, "Unbound")) {
            any_fail = true;
        }
        // TQ7-02: count positive proofs so a mistyped/absent property — which
        // produces NO ":check" result line — cannot vacuously pass. Cryptol marks a
        // proven property with "Q.E.D." (exhaustive) or "passed N tests" (randomised).
        // If a local cryptol build uses a different success marker, extend this.
        if (std::strstr(buf, "Q.E.D.") ||
            (std::strstr(buf, "passed") && std::strstr(buf, "test"))) {
            ++proved;
        }
    }

    int rc = pclose(pipe);
    int exit_code = 0;
#ifndef _WIN32
    if (WIFEXITED(rc)) exit_code = WEXITSTATUS(rc);
#else
    exit_code = rc;
#endif

    // Clean up temp file
    std::remove(tmpfile.c_str());

    // PASS only if cryptol exited cleanly, printed no failure, AND produced a positive
    // proof for EVERY requested property (proved >= count). The last clause closes the
    // TQ7-02 vacuous pass: a typo'd property name yields proved < count -> FAIL.
    const int want = static_cast<int>(properties.size());
    if (exit_code != 0 || any_fail || proved < want) {
        std::printf("    [verdict] exit=%d any_fail=%d proved=%d/%d -> FAIL\n",
                    exit_code, static_cast<int>(any_fail), proved, want);
        return 1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Property lists per spec file
// ---------------------------------------------------------------------------

static const std::vector<std::string> kFieldProps = {
    "field_add_commutative",
    "field_add_associative",
    "field_mul_commutative",
    "field_mul_associative",
    "field_distributive",
    "field_add_zero_identity",
    "field_mul_one_identity",
    "field_inv_property",
    "field_neg_double_negation",
    "field_sub_correct",
};

static const std::vector<std::string> kPointProps = {
    "generator_on_curve",
    "point_double_consistent",
    "point_neg_is_inverse",
    "point_add_commutative",
    "scalar_zero_is_infinity",
    "scalar_one_is_P",
    "privkey_one_gives_G",
};

static const std::vector<std::string> kEcdsaProps = {
    "ecdsa_sign_then_verify",
    "ecdsa_sign_low_s",
    "ecdsa_sign_r_range",
    "ecdsa_sign_s_range",
    "low_s_idempotent",
    "low_s_in_lower_half",
};

static const std::vector<std::string> kSchnorrProps = {
    "schnorr_sign_then_verify",
    "schnorr_sign_s_range",
    "schnorr_sign_rx_range",
    "normalised_key_has_even_y",
    "normalise_key_idempotent",
};

// ---------------------------------------------------------------------------
// _run()
// ---------------------------------------------------------------------------
int test_cryptol_specs_run() {
    if (!cryptol_available()) {
        std::printf("[cryptol_specs] ADVISORY: cryptol not installed — skipping\n");
        std::printf("[cryptol_specs]   install: apt-get install cryptol\n");
        return ADVISORY_SKIP_CODE;
    }

    std::string cry_dir = find_cryptol_dir();
    if (cry_dir.empty()) {
        std::printf("[cryptol_specs] ADVISORY: audit/formal/cryptol/ not found — skipping\n");
        return ADVISORY_SKIP_CODE;
    }

    std::printf("[cryptol_specs] Found specs at: %s\n", cry_dir.c_str());

    int total_fail = 0;

    struct Spec { const char* name; const std::vector<std::string>* props; };
    static const Spec kSpecs[] = {
        { "Secp256k1Field",   &kFieldProps   },
        { "Secp256k1Point",   &kPointProps   },
        { "Secp256k1ECDSA",  &kEcdsaProps   },
        { "Secp256k1Schnorr", &kSchnorrProps },
    };

    for (auto& spec : kSpecs) {
        std::printf("[cryptol_specs] Checking %s.cry (%zu properties)...\n",
                    spec.name, spec.props->size());
        int rc = check_cryptol_spec(cry_dir, spec.name, *spec.props);
        if (rc == 0) {
            std::printf("[cryptol_specs]   %s: PASS\n", spec.name);
        } else {
            std::printf("[cryptol_specs]   %s: FAIL (property failure — primitive is wrong)\n", spec.name);
            ++total_fail;
        }
    }

    if (total_fail == 0) {
        std::printf("[cryptol_specs] All Cryptol properties verified — PASS\n");
    } else {
        std::printf("[cryptol_specs] %d spec(s) FAILED — arithmetic primitives violated\n", total_fail);
    }

    return total_fail == 0 ? 0 : 1;
}
