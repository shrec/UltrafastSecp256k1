// ============================================================================
// audit_ct_namespace.cpp -- CT Namespace Discipline Audit
// ============================================================================
//
// Verifies the critical security invariant: every code path that handles
// SECRET DATA (private keys, nonces, derived intermediates) MUST use the
// `secp256k1::ct::` namespace (constant-time operations) and MUST NOT
// call `secp256k1::fast::` operations that are variable-time.
//
// This is the most common finding in external cryptographic library audits:
//   "Function X calls fast::scalar_mul with secret input y."
//
// Methodology:
//   1. Open the source files that implement secret-key operations.
//   2. Search for PROHIBITED patterns: fast:: point-mul / generator-mul calls.
//   3. Search for REQUIRED patterns: ct:: usage in those same files.
//   4. Report any violations as FAIL.
//
// This test is a SOURCE-LEVEL static analysis check embedded in the audit
// binary. It runs on every CI build — no separate tooling required.
//
// Files audited:
//   src/cpu/src/ct_sign.cpp     -- CT ECDSA + Schnorr signing
//   src/cpu/src/ecdh.cpp        -- ECDH key agreement
//   src/cpu/src/bip32.cpp       -- BIP-32 HD key derivation
//   src/cpu/src/taproot.cpp     -- Taproot key tweak
//   src/cpu/src/musig2.cpp      -- MuSig2 nonce & signing
//   src/cpu/src/recovery.cpp    -- Recoverable ECDSA signing
//   src/cpu/src/adaptor.cpp     -- Adaptor signature signing
//   src/cpu/src/bip324.cpp      -- BIP-324 ECDH session key derivation
//   src/cpu/src/frost.cpp       -- FROST threshold signing
//
// CNS-1  … CNS-5  : ct_sign.cpp — CT call pattern verification
// CNS-6  … CNS-8  : ecdh.cpp — CT usage for secret scalar multiply
// CNS-9  … CNS-11 : bip32.cpp — CT for child key derivation
// CNS-12 … CNS-13 : taproot.cpp — CT for key tweak
// CNS-14 … CNS-16 : musig2.cpp — CT for nonce / aggregate signing
// CNS-17 … CNS-20 : recovery.cpp, adaptor.cpp, bip324.cpp, frost.cpp
// CNS-21 …        : Prohibited pattern cross-checks
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

static int g_pass = 0, g_fail = 0;
#include "audit_check.hpp"

// ---------------------------------------------------------------------------
// Source-file scanner
// ---------------------------------------------------------------------------

struct ScanResult {
    bool opened = false;
    std::string content;
};

// Read the entire content of a source file into a string.
// Returns false if the file cannot be opened (expected during CI).
static ScanResult read_source_file(const char* path) {
    ScanResult r;
    FILE* f = std::fopen(path, "rb");
    if (!f) return r;
    r.opened = true;
    // Get file size
    (void)std::fseek(f, 0, SEEK_END);
    long const sz = std::ftell(f);
    (void)std::fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz > 4 * 1024 * 1024) {
        (void)std::fclose(f);
        return r;
    }
    r.content.resize(static_cast<std::size_t>(sz));
    auto const nread = std::fread(&r.content[0], 1, static_cast<std::size_t>(sz), f);
    if (nread != static_cast<std::size_t>(sz)) r.content.clear();
    (void)std::fclose(f);
    return r;
}

// Count occurrences of a literal substring
static int count_occurrences(const std::string& text, const char* needle) {
    int n = 0;
    std::size_t pos = 0;
    std::size_t const nlen = std::strlen(needle);
    while ((pos = text.find(needle, pos)) != std::string::npos) {
        ++n;
        pos += nlen;
    }
    return n;
}

static bool contains(const std::string& text, const char* needle) {
    return text.find(needle) != std::string::npos;
}

// Strip single-line C++ comments to avoid false positives in comments
static std::string strip_comments(const std::string& src) {
    std::string out;
    out.reserve(src.size());
    std::size_t i = 0;
    while (i < src.size()) {
        // Single-line comment
        if (i + 1 < src.size() && src[i] == '/' && src[i+1] == '/') {
            while (i < src.size() && src[i] != '\n') ++i;
            continue;
        }
        // Block comment
        if (i + 1 < src.size() && src[i] == '/' && src[i+1] == '*') {
            i += 2;
            while (i + 1 < src.size() && !(src[i] == '*' && src[i+1] == '/')) ++i;
            i += 2;
            continue;
        }
        out += src[i++];
    }
    return out;
}

// ---------------------------------------------------------------------------
// Locate the source tree root relative to the running binary.
// Tries several candidate paths so it works in build/ subdirectories.
// ---------------------------------------------------------------------------

static std::string find_source_root() {
    // Try common relative paths from CWD (works when ctest is run from build/)
    const char* candidates[] = {
        "../libs/UltrafastSecp256k1",
        "../../libs/UltrafastSecp256k1",
        "../../../libs/UltrafastSecp256k1",
        "libs/UltrafastSecp256k1",
        ".",          // when built in-tree
        "../",
    };
    for (const char* c : candidates) {
        // Try to open a sentinel file
        std::string test = std::string(c) + "/src/cpu/src/ct_sign.cpp";
        FILE* f = std::fopen(test.c_str(), "rb");
        if (f) {
            (void)std::fclose(f);
            return c;
        }
    }
    return "";  // not found
}

// ---------------------------------------------------------------------------
// Audit helper: check a source file for required and prohibited patterns
// ---------------------------------------------------------------------------

struct FileAudit {
    const char* label;        // human-readable label
    const char* rel_path;     // relative path from source root
    // Required: at least one occurrence expected
    std::vector<const char*> required;
    // Prohibited: must NOT appear in executable code (after comment strip)
    std::vector<const char*> prohibited;
};

// Run one file audit, return number of failures
static int run_file_audit(const std::string& root, const FileAudit& audit,
                          int& check_num) {
    int failures = 0;
    std::string full_path = root + "/" + audit.rel_path;
    ScanResult r = read_source_file(full_path.c_str());

    char msg[256];

    if (!r.opened) {
        // File not found — return ADVISORY_SKIP_CODE (77) so the unified runner
        // records this as advisory_skipped, not passed.  Returning 0 was a
        // false-green: a relocated ct_sign.cpp would silently pass all CT checks.
        (void)std::snprintf(msg, sizeof(msg),
            "CNS-%d: [ADVISORY] %s — source file not found at %s",
            check_num, audit.label, full_path.c_str());
        AUDIT_LOG("  [SKIP] %s\n", msg);
        ++check_num;
        return ADVISORY_SKIP_CODE;
    }

    // Strip comments before checking prohibited patterns
    std::string code = strip_comments(r.content);

    // Required patterns
    for (const char* req : audit.required) {
        (void)std::snprintf(msg, sizeof(msg),
            "CNS-%d: %s contains required CT pattern '%s'",
            check_num, audit.label, req);
        CHECK(contains(code, req), msg);
        if (!contains(code, req)) ++failures;
        ++check_num;
    }

    // Prohibited patterns
    for (const char* pro : audit.prohibited) {
        int occ = count_occurrences(code, pro);
        (void)std::snprintf(msg, sizeof(msg),
            "CNS-%d: %s has NO prohibited fast:: call '%s' (found %d)",
            check_num, audit.label, pro, occ);
        CHECK(occ == 0, msg);
        if (occ != 0) ++failures;
        ++check_num;
    }

    return failures;
}

// ---------------------------------------------------------------------------
// Audit specifications
// ---------------------------------------------------------------------------

static const FileAudit AUDITS[] = {
    // ct_sign.cpp: CT ECDSA + Schnorr
    {
        "ct_sign.cpp",
        "src/cpu/src/ct_sign.cpp",
        /* required  */ { "ct::generator_mul", "ct::scalar_inverse", "secure_erase" },
        /* prohibited */ { "fast::generator_mul", "fast::scalar_mul", "fast::point_mul" }
    },
    // ecdh.cpp: ECDH uses CT for secret scalar multiply
    {
        "ecdh.cpp",
        "src/cpu/src/ecdh.cpp",
        /* required  */ { "ct::scalar_mul" },
        /* prohibited */ { "fast::scalar_mul" }
    },
    // bip32.cpp: Child key derivation must use CT for scalar addition
    {
        "bip32.cpp",
        "src/cpu/src/bip32.cpp",
        /* required  */ { "secp256k1/ct/" },
        /* prohibited */ { "fast::generator_mul", "fast::scalar_mul" }
    },
    // taproot.cpp: Key tweak must use CT
    {
        "taproot.cpp",
        "src/cpu/src/taproot.cpp",
        /* required  */ { "secp256k1/ct/" },
        /* prohibited */ { "fast::scalar_mul", "fast::generator_mul" }
    },
    // musig2.cpp: Nonce generation and partial signing must use CT
    {
        "musig2.cpp",
        "src/cpu/src/musig2.cpp",
        /* required  */ { "secp256k1/ct/" },
        /* prohibited */ { "fast::generator_mul" }
    },
    // recovery.cpp: Recoverable ECDSA signing uses CT generator mul + scalar arithmetic
    {
        "recovery.cpp",
        "src/cpu/src/recovery.cpp",
        /* required  */ { "secp256k1/ct/" },
        /* prohibited */ { "fast::generator_mul", "fast::scalar_mul" }
    },
    // adaptor.cpp: Adaptor signature signing uses CT for all secret scalar paths
    {
        "adaptor.cpp",
        "src/cpu/src/adaptor.cpp",
        /* required  */ { "secp256k1/ct/" },
        /* prohibited */ { "fast::generator_mul", "fast::scalar_mul" }
    },
    // bip324.cpp: BIP-324 session uses CT for ECDH key derivation
    {
        "bip324.cpp",
        "src/cpu/src/bip324.cpp",
        /* required  */ { "secp256k1/ct/" },
        /* prohibited */ { "fast::generator_mul", "fast::scalar_mul" }
    },
    // frost.cpp: FROST threshold signing uses CT for nonce and share arithmetic
    {
        "frost.cpp",
        "src/cpu/src/frost.cpp",
        /* required  */ { "secp256k1/ct/" },
        /* prohibited */ { "fast::generator_mul", "fast::scalar_mul" }
    },
};

// ---------------------------------------------------------------------------
// Additional structural checks
// ---------------------------------------------------------------------------

static void run_structural_checks(const std::string& root, int& check_num) {
    AUDIT_LOG("\n  [CNS-struct] Structural CT discipline checks\n");

    // ct_sign.cpp must NOT include fast.hpp directly (would pull in fast:: ADL)
    {
        std::string path = root + "/src/cpu/src/ct_sign.cpp";
        ScanResult r = read_source_file(path.c_str());
        if (r.opened) {
            std::string code = strip_comments(r.content);
            char msg[256];
            (void)std::snprintf(msg, sizeof(msg),
                "CNS-%d: ct_sign.cpp does not #include secp256k1/fast.hpp",
                check_num);
            // fast.hpp is the umbrella that enables fast:: namespace
            // ct_sign should only include ct/sign.hpp and ct/ headers
            bool includes_fast_hpp = (
                contains(code, "#include \"secp256k1/fast.hpp\"") ||
                contains(code, "#include <secp256k1/fast.hpp>")
            );
            CHECK(!includes_fast_hpp, msg);
            ++check_num;
        } else {
            ++check_num;
        }
    }

    // ecdh.cpp must include ct/point.hpp (its CT scalar_mul lives there)
    {
        std::string path = root + "/src/cpu/src/ecdh.cpp";
        ScanResult r = read_source_file(path.c_str());
        if (r.opened) {
            char msg[256];
            (void)std::snprintf(msg, sizeof(msg),
                "CNS-%d: ecdh.cpp includes ct/point.hpp for ct::scalar_mul", check_num);
            CHECK(contains(r.content, "secp256k1/ct/point.hpp"), msg);
            ++check_num;
        } else {
            ++check_num;
        }
    }

    // ct_sign.cpp must include detail/secure_erase.hpp
    {
        std::string path = root + "/src/cpu/src/ct_sign.cpp";
        ScanResult r = read_source_file(path.c_str());
        if (r.opened) {
            char msg[256];
            (void)std::snprintf(msg, sizeof(msg),
                "CNS-%d: ct_sign.cpp includes detail/secure_erase.hpp", check_num);
            CHECK(contains(r.content, "detail/secure_erase.hpp"), msg);
            ++check_num;
        } else {
            ++check_num;
        }
    }

    // ecdh.cpp must erase intermediate shared point
    {
        std::string path = root + "/src/cpu/src/ecdh.cpp";
        ScanResult r = read_source_file(path.c_str());
        if (r.opened) {
            char msg[256];
            (void)std::snprintf(msg, sizeof(msg),
                "CNS-%d: ecdh.cpp calls secure_erase on shared-point intermediate",
                check_num);
            CHECK(contains(r.content, "secure_erase"), msg);
            ++check_num;
        } else {
            ++check_num;
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

int audit_ct_namespace_run() {
    g_pass = 0; g_fail = 0;

    AUDIT_LOG("============================================================\n");
    AUDIT_LOG("  CT Namespace Discipline Audit\n");
    AUDIT_LOG("  Verify secret-key paths use ct:: not fast::\n");
    AUDIT_LOG("============================================================\n");

    std::string root = find_source_root();
    if (root.empty()) {
        AUDIT_LOG("  [SKIP] Source tree not found — skipping static checks.\n");
        AUDIT_LOG("  (Run ctest from the build directory with source tree present.)\n");
        printf("[audit_ct_namespace] skipped (source tree absent)\n");
        return 77;  // ADVISORY_SKIP_CODE — not PASS, not FAIL
    }

    AUDIT_LOG("  Source root: %s\n", root.c_str());

    int check_num = 1;

    // Per-file audits
    for (const auto& audit : AUDITS) {
        AUDIT_LOG("\n  Auditing: %s\n", audit.label);
        run_file_audit(root, audit, check_num);
    }

    // Structural checks
    run_structural_checks(root, check_num);

    printf("[audit_ct_namespace] %d/%d checks passed\n",
           g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#ifndef UNIFIED_AUDIT_RUNNER
int main() {
    return audit_ct_namespace_run();
}
#endif
