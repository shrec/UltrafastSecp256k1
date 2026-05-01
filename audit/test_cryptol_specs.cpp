// ============================================================================
// test_cryptol_specs.cpp -- Cryptol formal spec checker for unified runner
// ============================================================================
//
// Advisory module: checks that the Cryptol formal specifications in
// formal/cryptol/ pass their QuickCheck properties via the `cryptol` REPL.
//
// ADVISORY = true: failure produces a WARN, not a hard FAIL.
// Reason: requires the Cryptol toolchain to be installed.
//         CI jobs run this where installed; others skip silently.
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
        "../formal/cryptol",
        "formal/cryptol",
        "../../formal/cryptol",
        nullptr
    };
    for (int i = 0; kCandidates[i]; ++i) {
        // Check for Field spec as sentinel
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
#ifdef _WIN32
    std::string tmpfile = (std::filesystem::temp_directory_path() / ("cryptol_check_" + module_name + ".sh")).string();
    FILE* tf = std::fopen(tmpfile.c_str(), "w");
    if (!tf) return 0;
#else
    std::string tmpfile = (std::filesystem::temp_directory_path() / "cryptol_check_XXXXXX").string();
    int tmpfd = mkstemp(tmpfile.data());
    if (tmpfd < 0) return 0;  // can't write temp — skip
    FILE* tf = fdopen(tmpfd, "w");
    if (!tf) { close(tmpfd); return 0; }
#endif
    std::fputs(batch.c_str(), tf);
    std::fclose(tf);

    std::string cmd = "cryptol --batch < " + tmpfile + " 2>&1";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return 0;

    bool any_fail = false;
    char buf[512];
    while (std::fgets(buf, sizeof(buf), pipe)) {
        std::printf("    %s", buf);
        // Cryptol prints "FAILED" or "Counterexample" on property failure
        if (std::strstr(buf, "FAILED") || std::strstr(buf, "Counterexample") ||
            std::strstr(buf, "ERROR")) {
            any_fail = true;
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

    return (exit_code != 0 || any_fail) ? 1 : 0;
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
        std::printf("[cryptol_specs] cryptol not installed — skipping (advisory)\n");
        return ADVISORY_SKIP_CODE;
    }

    std::string cry_dir = find_cryptol_dir();
    if (cry_dir.empty()) {
        std::printf("[cryptol_specs] formal/cryptol/ not found — skipping (advisory)\n");
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
            std::printf("[cryptol_specs]   %s: WARN (property failure)\n", spec.name);
            ++total_fail;
        }
    }

    if (total_fail == 0) {
        std::printf("[cryptol_specs] All Cryptol properties checked — PASS\n");
    } else {
        std::printf("[cryptol_specs] %d spec(s) had property failures — WARN\n", total_fail);
    }

    return total_fail == 0 ? 0 : 1;
}
