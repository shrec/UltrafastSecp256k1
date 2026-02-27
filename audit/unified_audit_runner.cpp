// ============================================================================
// Unified Audit Runner -- UltrafastSecp256k1
// ============================================================================
//
// Unified self-audit application. Single binary for all platforms.
// Build, run, validate all tests, save report.
//
// Single binary that runs ALL library tests and produces a structured
// JSON + text audit report. Build once, run on any platform.
//
// Usage:
//   unified_audit_runner              # run all tests, write report
//   unified_audit_runner --json-only  # suppress console, write JSON only
//   unified_audit_runner --report-dir <dir>  # write reports to <dir>
//
// Generates:
//   audit_report.json   -- machine-readable structured result
//   audit_report.txt    -- human-readable summary
// ============================================================================

#ifndef UNIFIED_AUDIT_RUNNER
#define UNIFIED_AUDIT_RUNNER  // Guard standalone main() in test modules
#endif

#include "secp256k1/selftest.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <chrono>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/utsname.h>
#endif

// Library version (injected by CMake from VERSION.txt)
#if __has_include("secp256k1/version.hpp")
#include "secp256k1/version.hpp"
#endif
#ifndef SECP256K1_VERSION_STRING
#define SECP256K1_VERSION_STRING "unknown"
#endif

// Git hash (injected at compile time via -DGIT_HASH=...)
#ifndef GIT_HASH
#define GIT_HASH "unknown"
#endif

// Audit framework version (bump when report schema changes)
static constexpr const char* AUDIT_FRAMEWORK_VERSION = "2.0.0";

using namespace secp256k1::fast;

// ============================================================================
// Forward declarations -- selftest modules (from run_selftest.cpp sources)
// ============================================================================
int test_large_scalar_multiplication_run();
int test_mul_run();
int test_arithmetic_correctness_run();
int test_ct_run();
int test_ct_equivalence_run();
int test_ecdsa_schnorr_run();
int test_multiscalar_batch_run();
int test_bip32_run();
int test_bip32_vectors_run();
int test_musig2_run();
int test_ecdh_recovery_taproot_run();
int test_simd_batch_run();
int test_v4_features_run();
int test_coins_run();
int test_batch_add_affine_run();
int test_hash_accel_run();
int run_exhaustive_tests();
int test_comprehensive_run();
int test_bip340_vectors_run();
int test_rfc6979_vectors_run();
int test_ecc_properties_run();

// ============================================================================
// Forward declarations -- additional standalone test _run() functions
// ============================================================================
int test_carry_propagation_run();
int test_fault_injection_run();
int test_fiat_crypto_vectors_run();
int test_cross_platform_kat_run();
int test_debug_invariants_run();
int test_abi_gate_run();
int test_ct_sidechannel_smoke_run();
int test_differential_run();

// ============================================================================
// Forward declarations -- MuSig2 / FROST protocol tests
// ============================================================================
int test_musig2_frost_protocol_run();
int test_musig2_frost_advanced_run();
int test_frost_kat_run();

// ============================================================================
// Forward declarations -- adversarial / fuzz tests
// ============================================================================
int test_audit_fuzz_run();
int test_fuzz_parsers_run();
int test_fuzz_address_bip32_ffi_run();

// ============================================================================
// Forward declarations -- deep audit modules
// ============================================================================
int audit_field_run();       // Section I.1: Field Fp correctness
int audit_scalar_run();      // Section I.2: Scalar Zn correctness
int audit_point_run();       // Section I.3: Point & signature correctness
int audit_ct_run();          // Section II:  CT & side-channel deep audit
int audit_integration_run(); // Section VI:  Integration & cross-protocol
int audit_security_run();    // Section V:   Security hardening
int audit_perf_run();        // Section IV:  Performance validation

// ============================================================================
// Forward declarations -- field representation tests
// ============================================================================
#ifdef __SIZEOF_INT128__
int test_field_52_main();   // 5x52 lazy-reduction (requires __uint128_t)
#endif
int test_field_26_main();   // 10x26 lazy-reduction

// ============================================================================
// Forward declarations -- diagnostics
// ============================================================================
int diag_scalar_mul_run();

// ============================================================================
// Report section IDs -- 8 audit categories
// ============================================================================
//   1. math_invariants   -- Mathematical Invariants (Fp, Zn, Group Laws)
//   2. ct_analysis       -- Constant-Time / Side-Channel Analysis
//   3. differential      -- Differential & Cross-Library Testing
//   4. standard_vectors  -- Standard Test Vectors (BIP-340, RFC-6979, BIP-32)
//   5. fuzzing           -- Fuzzing & Adversarial Attack Resilience
//   6. protocol_security -- Protocol Security (ECDSA, Schnorr, MuSig2, FROST)
//   7. memory_safety     -- ABI & Memory Safety (sanitizer, zeroization)
//   8. performance       -- Performance Validation & Regression
// ============================================================================

struct AuditModule {
    const char* id;           // short ID for JSON
    const char* name;         // human-readable name
    const char* section;      // one of 8 report sections
    int (*run)();             // returns 0=PASS, non-zero=FAIL
};

// Section display names (Georgian + English)
struct SectionInfo {
    const char* id;
    const char* title_ka;     // Georgian
    const char* title_en;     // English
};

static const SectionInfo SECTIONS[] = {
    { "math_invariants",   "\xe1\x83\x9b\xe1\x83\x90\xe1\x83\x97\xe1\x83\x94\xe1\x83\x9b\xe1\x83\x90\xe1\x83\xa2\xe1\x83\x98\xe1\x83\x99\xe1\x83\xa3\xe1\x83\xa0\xe1\x83\x98 \xe1\x83\x98\xe1\x83\x9c\xe1\x83\x95\xe1\x83\x90\xe1\x83\xa0\xe1\x83\x98\xe1\x83\x90\xe1\x83\x9c\xe1\x83\xa2\xe1\x83\x94\xe1\x83\x91\xe1\x83\x98",
                           "Mathematical Invariants (Fp, Zn, Group Laws)" },
    { "ct_analysis",       "Constant-Time \xe1\x83\x90\xe1\x83\x9c\xe1\x83\x90\xe1\x83\x9a\xe1\x83\x98\xe1\x83\x96\xe1\x83\x98",
                           "Constant-Time & Side-Channel Analysis" },
    { "differential",      "\xe1\x83\x93\xe1\x83\x98\xe1\x83\xa4\xe1\x83\x94\xe1\x83\xa0\xe1\x83\x94\xe1\x83\x9c\xe1\x83\xaa\xe1\x83\x98\xe1\x83\x90\xe1\x83\x9a\xe1\x83\xa3\xe1\x83\xa0\xe1\x83\x98 \xe1\x83\xa2\xe1\x83\x94\xe1\x83\xa1\xe1\x83\xa2\xe1\x83\x98\xe1\x83\xa0\xe1\x83\x94\xe1\x83\x91\xe1\x83\x90",
                           "Differential & Cross-Library Testing" },
    { "standard_vectors",  "\xe1\x83\xa1\xe1\x83\xa2\xe1\x83\x90\xe1\x83\x9c\xe1\x83\x93\xe1\x83\x90\xe1\x83\xa0\xe1\x83\xa2\xe1\x83\xa3\xe1\x83\x9a\xe1\x83\x98 \xe1\x83\x95\xe1\x83\x94\xe1\x83\xa5\xe1\x83\xa2\xe1\x83\x9d\xe1\x83\xa0\xe1\x83\x94\xe1\x83\x91\xe1\x83\x98",
                           "Standard Test Vectors (BIP-340, RFC-6979, BIP-32)" },
    { "fuzzing",           "\xe1\x83\xa4\xe1\x83\x90\xe1\x83\x96\xe1\x83\x98\xe1\x83\x9c\xe1\x83\x92\xe1\x83\x98 & \xe1\x83\x90\xe1\x83\x93\xe1\x83\x95\xe1\x83\x94\xe1\x83\xa0\xe1\x83\xa1\xe1\x83\x90\xe1\x83\xa0\xe1\x83\x98",
                           "Fuzzing & Adversarial Attack Resilience" },
    { "protocol_security", "\xe1\x83\x9e\xe1\x83\xa0\xe1\x83\x9d\xe1\x83\xa2\xe1\x83\x9d\xe1\x83\x99\xe1\x83\x9d\xe1\x83\x9a\xe1\x83\x94\xe1\x83\x91\xe1\x83\x98\xe1\x83\xa1 \xe1\x83\xa3\xe1\x83\xa1\xe1\x83\x90\xe1\x83\xa4\xe1\x83\xa0\xe1\x83\x97\xe1\x83\xae\xe1\x83\x9d\xe1\x83\x94\xe1\x83\x91\xe1\x83\x90",
                           "Protocol Security (ECDSA, Schnorr, MuSig2, FROST)" },
    { "memory_safety",     "ABI & \xe1\x83\x9b\xe1\x83\x94\xe1\x83\xae\xe1\x83\xa1\xe1\x83\x98\xe1\x83\x94\xe1\x83\xa0\xe1\x83\x94\xe1\x83\x91\xe1\x83\x98\xe1\x83\xa1 \xe1\x83\xa3\xe1\x83\xa1\xe1\x83\x90\xe1\x83\xa4\xe1\x83\xa0\xe1\x83\x97\xe1\x83\xae\xe1\x83\x9d\xe1\x83\x94\xe1\x83\x91\xe1\x83\x90",
                           "ABI & Memory Safety (zeroization, hardening)" },
    { "performance",       "\xe1\x83\x9e\xe1\x83\x94\xe1\x83\xa0\xe1\x83\xa4\xe1\x83\x9d\xe1\x83\xa0\xe1\x83\x9b\xe1\x83\x90\xe1\x83\x9c\xe1\x83\xa1\xe1\x83\x98\xe1\x83\xa1 \xe1\x83\x95\xe1\x83\x90\xe1\x83\x9a\xe1\x83\x98\xe1\x83\x93\xe1\x83\x90\xe1\x83\xaa\xe1\x83\x98\xe1\x83\x90",
                           "Performance Validation & Regression" },
};
static constexpr int NUM_SECTIONS = sizeof(SECTIONS) / sizeof(SECTIONS[0]);

static const AuditModule ALL_MODULES[] = {
    // ===================================================================
    // Section 1: Mathematical Invariants (Fp, Zn, Group Laws)
    // ===================================================================
    { "audit_field",       "Field Fp deep audit (add/mul/inv/sqrt/batch)", "math_invariants", audit_field_run },
    { "audit_scalar",      "Scalar Zn deep audit (mod/GLV/edge/inv)",      "math_invariants", audit_scalar_run },
    { "audit_point",       "Point ops deep audit (Jac/affine/sigs)",       "math_invariants", audit_point_run },
    { "mul",               "Field & scalar arithmetic",                    "math_invariants", test_mul_run },
    { "arith_correct",     "Arithmetic correctness",                       "math_invariants", test_arithmetic_correctness_run },
    { "scalar_mul",        "Scalar multiplication",                        "math_invariants", test_large_scalar_multiplication_run },
    { "exhaustive",        "Exhaustive algebraic verification",            "math_invariants", run_exhaustive_tests },
    { "comprehensive",     "Comprehensive 500+ suite",                     "math_invariants", test_comprehensive_run },
    { "ecc_properties",    "ECC property-based invariants",                "math_invariants", test_ecc_properties_run },
    { "batch_add",         "Affine batch addition",                        "math_invariants", test_batch_add_affine_run },
    { "carry_propagation", "Carry chain stress (limb boundary)",           "math_invariants", test_carry_propagation_run },
#ifdef __SIZEOF_INT128__
    { "field_52",          "FieldElement52 (5x52) vs 4x64",               "math_invariants", test_field_52_main },
#endif
    { "field_26",          "FieldElement26 (10x26) vs 4x64",              "math_invariants", test_field_26_main },

    // ===================================================================
    // Section 2: Constant-Time / Side-Channel Analysis
    // ===================================================================
    { "audit_ct",          "CT deep audit (masks/cmov/cswap/timing)",      "ct_analysis",    audit_ct_run },
    { "ct",                "Constant-time layer",                          "ct_analysis",    test_ct_run },
    { "ct_equivalence",    "FAST == CT equivalence",                       "ct_analysis",    test_ct_equivalence_run },
    { "ct_sidechannel",    "Side-channel dudect (smoke)",                  "ct_analysis",    test_ct_sidechannel_smoke_run },
    { "diag_scalar_mul",   "CT scalar_mul vs fast (diagnostic)",           "ct_analysis",    diag_scalar_mul_run },

    // ===================================================================
    // Section 3: Differential & Cross-Library Testing
    // ===================================================================
    { "differential",      "Differential correctness",                     "differential",   test_differential_run },
    { "fiat_crypto",       "Fiat-Crypto reference vectors",               "differential",   test_fiat_crypto_vectors_run },
    { "cross_platform_kat","Cross-platform KAT",                          "differential",   test_cross_platform_kat_run },

    // ===================================================================
    // Section 4: Standard Test Vectors (BIP-340, RFC-6979, BIP-32)
    // ===================================================================
    { "bip340_vectors",    "BIP-340 official vectors",                     "standard_vectors", test_bip340_vectors_run },
    { "bip32_vectors",     "BIP-32 official vectors TV1-5",               "standard_vectors", test_bip32_vectors_run },
    { "rfc6979_vectors",   "RFC 6979 ECDSA vectors",                      "standard_vectors", test_rfc6979_vectors_run },
    { "frost_kat",         "FROST reference KAT vectors",                 "standard_vectors", test_frost_kat_run },

    // ===================================================================
    // Section 5: Fuzzing & Adversarial Attack Resilience
    // ===================================================================
    { "audit_fuzz",        "Adversarial fuzz (malform/edge)",              "fuzzing",        test_audit_fuzz_run },
    { "fuzz_parsers",      "Parser fuzz (DER/Schnorr/Pubkey)",            "fuzzing",        test_fuzz_parsers_run },
    { "fuzz_addr_bip32",   "Address/BIP32/FFI boundary fuzz",             "fuzzing",        test_fuzz_address_bip32_ffi_run },
    { "fault_injection",   "Fault injection simulation",                   "fuzzing",        test_fault_injection_run },

    // ===================================================================
    // Section 6: Protocol Security (ECDSA, Schnorr, MuSig2, FROST)
    // ===================================================================
    { "ecdsa_schnorr",     "ECDSA + Schnorr",                             "protocol_security", test_ecdsa_schnorr_run },
    { "bip32",             "BIP-32 HD derivation",                        "protocol_security", test_bip32_run },
    { "musig2",            "MuSig2",                                       "protocol_security", test_musig2_run },
    { "ecdh_recovery",     "ECDH + recovery + taproot",                   "protocol_security", test_ecdh_recovery_taproot_run },
    { "v4_features",       "v4 (Pedersen/FROST/etc)",                     "protocol_security", test_v4_features_run },
    { "coins",             "Coins layer",                                  "protocol_security", test_coins_run },
    { "musig2_frost",      "MuSig2 + FROST protocol suite",              "protocol_security", test_musig2_frost_protocol_run },
    { "musig2_frost_adv",  "MuSig2 + FROST advanced/adversar",           "protocol_security", test_musig2_frost_advanced_run },
    { "audit_integration", "Integration (ECDH/batch/cross-proto)",        "protocol_security", audit_integration_run },

    // ===================================================================
    // Section 7: ABI & Memory Safety (zeroization, hardening)
    // ===================================================================
    { "audit_security",    "Security hardening (zero/bitflip/nonce)",      "memory_safety",  audit_security_run },
    { "debug_invariants",  "Debug invariant assertions",                   "memory_safety",  test_debug_invariants_run },
    { "abi_gate",          "ABI version gate (compile-time)",              "memory_safety",  test_abi_gate_run },

    // ===================================================================
    // Section 8: Performance Validation & Regression
    // ===================================================================
    { "hash_accel",        "Accelerated hashing",                          "performance",    test_hash_accel_run },
    { "simd_batch",        "SIMD batch operations",                        "performance",    test_simd_batch_run },
    { "multiscalar",       "Multi-scalar & batch verify",                  "performance",    test_multiscalar_batch_run },
    { "audit_perf",        "Performance smoke (sign/verify roundtrip)",    "performance",    audit_perf_run },
};

static constexpr int NUM_MODULES = sizeof(ALL_MODULES) / sizeof(ALL_MODULES[0]);

// ============================================================================
// Platform detection
// ============================================================================
struct PlatformInfo {
    std::string os;
    std::string arch;
    std::string compiler;
    std::string build_type;
    std::string timestamp;
    std::string library_version;
    std::string git_hash;
    std::string framework_version;
};

static PlatformInfo detect_platform() {
    PlatformInfo info;

    // -- OS --
#if defined(_WIN32)
    info.os = "Windows";
#elif defined(__APPLE__)
    info.os = "macOS";
#elif defined(__linux__)
    info.os = "Linux";
#elif defined(__FreeBSD__)
    info.os = "FreeBSD";
#else
    info.os = "Unknown";
#endif

    // -- Architecture --
#if defined(__x86_64__) || defined(_M_X64)
    info.arch = "x86-64";
#elif defined(__aarch64__) || defined(_M_ARM64)
    info.arch = "ARM64";
#elif defined(__riscv) && (__riscv_xlen == 64)
    info.arch = "RISC-V 64";
#elif defined(__riscv)
    info.arch = "RISC-V 32";
#elif defined(__EMSCRIPTEN__)
    info.arch = "WASM";
#elif defined(__arm__) || defined(_M_ARM)
    info.arch = "ARM32";
#else
    info.arch = "Unknown";
#endif

    // -- Compiler --
#if defined(__clang__)
    char buf[128];
    (void)std::snprintf(buf, sizeof(buf), "Clang %d.%d.%d", __clang_major__, __clang_minor__, __clang_patchlevel__);
    info.compiler = buf;
#elif defined(__GNUC__)
    char buf[128];
    (void)std::snprintf(buf, sizeof(buf), "GCC %d.%d.%d", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
    info.compiler = buf;
#elif defined(_MSC_VER)
    char buf[128];
    (void)std::snprintf(buf, sizeof(buf), "MSVC %d", _MSC_VER);
    info.compiler = buf;
#else
    info.compiler = "Unknown";
#endif

    // -- Build type --
#if defined(NDEBUG)
    info.build_type = "Release";
#else
    info.build_type = "Debug";
#endif

    // -- Timestamp --
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    char timebuf[64];
    (void)std::strftime(timebuf, sizeof(timebuf), "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
    info.timestamp = timebuf;

    // -- Version / git / framework --
    info.library_version  = SECP256K1_VERSION_STRING;
    info.git_hash         = GIT_HASH;
    info.framework_version = AUDIT_FRAMEWORK_VERSION;

    return info;
}

// ============================================================================
// JSON escaping
// ============================================================================
static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char const c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

// ============================================================================
// Module result
// ============================================================================
struct ModuleResult {
    const char* id;
    const char* name;
    const char* section;
    bool        passed;
    double      elapsed_ms;
};

// ============================================================================
// Section summary helper
// ============================================================================
struct SectionSummary {
    const char* section_id;
    const char* title_en;
    int total;
    int passed;
    int failed;
    double time_ms;
};

static std::vector<SectionSummary> compute_section_summaries(
    const std::vector<ModuleResult>& results)
{
    std::vector<SectionSummary> out;
    for (int s = 0; s < NUM_SECTIONS; ++s) {
        SectionSummary ss{};
        ss.section_id = SECTIONS[s].id;
        ss.title_en   = SECTIONS[s].title_en;
        ss.total = ss.passed = ss.failed = 0;
        ss.time_ms = 0;
        for (auto& r : results) {
            if (std::strcmp(r.section, SECTIONS[s].id) == 0) {
                ++ss.total;
                if (r.passed) ++ss.passed; else ++ss.failed;
                ss.time_ms += r.elapsed_ms;
            }
        }
        out.push_back(ss);
    }
    return out;
}

// ============================================================================
// Report writer -- JSON (structured by 8 sections)
// ============================================================================
static void write_json_report(const char* path,
                               const PlatformInfo& plat,
                               const std::vector<ModuleResult>& results,
                               bool selftest_passed,
                               double selftest_ms,
                               double total_ms) {
    FILE* f = std::fopen(path, "w");
    if (!f) {
        (void)std::fprintf(stderr, "WARNING: Cannot open %s for writing\n", path);
        return;
    }

    int total_pass = 0, total_fail = 0;
    for (auto& r : results) {
        if (r.passed) ++total_pass; else ++total_fail;
    }
    if (selftest_passed) ++total_pass; else ++total_fail;

    auto sections = compute_section_summaries(results);

    (void)std::fprintf(f, "{\n");
    (void)std::fprintf(f, "  \"report_type\": \"industrial_self_audit\",\n");
    (void)std::fprintf(f, "  \"library\": \"UltrafastSecp256k1\",\n");
    (void)std::fprintf(f, "  \"library_version\": \"%s\",\n", json_escape(plat.library_version).c_str());
    (void)std::fprintf(f, "  \"git_hash\": \"%s\",\n", json_escape(plat.git_hash).c_str());
    (void)std::fprintf(f, "  \"audit_framework_version\": \"%s\",\n", json_escape(plat.framework_version).c_str());
    (void)std::fprintf(f, "  \"timestamp\": \"%s\",\n", json_escape(plat.timestamp).c_str());
    (void)std::fprintf(f, "  \"platform\": {\n");
    (void)std::fprintf(f, "    \"os\": \"%s\",\n", json_escape(plat.os).c_str());
    (void)std::fprintf(f, "    \"arch\": \"%s\",\n", json_escape(plat.arch).c_str());
    (void)std::fprintf(f, "    \"compiler\": \"%s\",\n", json_escape(plat.compiler).c_str());
    (void)std::fprintf(f, "    \"build_type\": \"%s\"\n", json_escape(plat.build_type).c_str());
    (void)std::fprintf(f, "  },\n");
    (void)std::fprintf(f, "  \"summary\": {\n");
    (void)std::fprintf(f, "    \"total_modules\": %d,\n", (int)results.size() + 1);
    (void)std::fprintf(f, "    \"passed\": %d,\n", total_pass);
    (void)std::fprintf(f, "    \"failed\": %d,\n", total_fail);
    (void)std::fprintf(f, "    \"all_passed\": %s,\n", (total_fail == 0) ? "true" : "false");
    (void)std::fprintf(f, "    \"total_time_ms\": %.1f,\n", total_ms);
    (void)std::fprintf(f, "    \"audit_verdict\": \"%s\"\n",
                 (total_fail == 0) ? "AUDIT-READY" : "AUDIT-BLOCKED");
    (void)std::fprintf(f, "  },\n");

    // Selftest
    (void)std::fprintf(f, "  \"selftest\": {\n");
    (void)std::fprintf(f, "    \"passed\": %s,\n", selftest_passed ? "true" : "false");
    (void)std::fprintf(f, "    \"time_ms\": %.1f\n", selftest_ms);
    (void)std::fprintf(f, "  },\n");

    // Sections summary
    (void)std::fprintf(f, "  \"sections\": [\n");
    for (int s = 0; s < (int)sections.size(); ++s) {
        auto& sec = sections[s];
        (void)std::fprintf(f, "    {\n");
        (void)std::fprintf(f, "      \"id\": \"%s\",\n", sec.section_id);
        (void)std::fprintf(f, "      \"title\": \"%s\",\n", json_escape(sec.title_en).c_str());
        (void)std::fprintf(f, "      \"total\": %d,\n", sec.total);
        (void)std::fprintf(f, "      \"passed\": %d,\n", sec.passed);
        (void)std::fprintf(f, "      \"failed\": %d,\n", sec.failed);
        (void)std::fprintf(f, "      \"time_ms\": %.1f,\n", sec.time_ms);
        (void)std::fprintf(f, "      \"status\": \"%s\",\n", (sec.failed == 0) ? "PASS" : "FAIL");

        // Nested modules for this section
        (void)std::fprintf(f, "      \"modules\": [\n");
        bool first = true;
        for (auto& r : results) {
            if (std::strcmp(r.section, sec.section_id) != 0) continue;
            if (!first) (void)std::fprintf(f, ",\n");
            first = false;
            (void)std::fprintf(f, "        { \"id\": \"%s\", \"name\": \"%s\", \"passed\": %s, \"time_ms\": %.1f }",
                         r.id, json_escape(r.name).c_str(),
                         r.passed ? "true" : "false", r.elapsed_ms);
        }
        (void)std::fprintf(f, "\n      ]\n");
        (void)std::fprintf(f, "    }%s\n", (s + 1 < (int)sections.size()) ? "," : "");
    }
    (void)std::fprintf(f, "  ]\n");
    (void)std::fprintf(f, "}\n");

    (void)std::fclose(f);
}

// ============================================================================
// Report writer -- Text (structured by 8 sections)
// ============================================================================
static void write_text_report(const char* path,
                               const PlatformInfo& plat,
                               const std::vector<ModuleResult>& results,
                               bool selftest_passed,
                               double selftest_ms,
                               double total_ms) {
    FILE* f = std::fopen(path, "w");
    if (!f) {
        (void)std::fprintf(stderr, "WARNING: Cannot open %s for writing\n", path);
        return;
    }

    int total_pass = 0, total_fail = 0;
    for (auto& r : results) {
        if (r.passed) ++total_pass; else ++total_fail;
    }
    if (selftest_passed) ++total_pass; else ++total_fail;

    auto sections = compute_section_summaries(results);

    (void)std::fprintf(f, "================================================================\n");
    (void)std::fprintf(f, "  UltrafastSecp256k1 -- Industrial Self-Audit Report\n");
    (void)std::fprintf(f, "================================================================\n\n");
    (void)std::fprintf(f, "Library:    UltrafastSecp256k1 v%s\n", plat.library_version.c_str());
    (void)std::fprintf(f, "Git Hash:   %s\n", plat.git_hash.c_str());
    (void)std::fprintf(f, "Framework:  Audit Framework v%s\n", plat.framework_version.c_str());
    (void)std::fprintf(f, "Timestamp:  %s\n", plat.timestamp.c_str());
    (void)std::fprintf(f, "OS:         %s\n", plat.os.c_str());
    (void)std::fprintf(f, "Arch:       %s\n", plat.arch.c_str());
    (void)std::fprintf(f, "Compiler:   %s\n", plat.compiler.c_str());
    (void)std::fprintf(f, "Build:      %s\n", plat.build_type.c_str());
    (void)std::fprintf(f, "\n");

    // -- Library selftest ---
    (void)std::fprintf(f, "----------------------------------------------------------------\n");
    (void)std::fprintf(f, "  [0] Library Selftest (core KAT)          %s  (%.0f ms)\n",
                 selftest_passed ? "PASS" : "FAIL", selftest_ms);
    (void)std::fprintf(f, "----------------------------------------------------------------\n\n");

    // -- 8 Sections ---
    int module_idx = 1;
    for (int s = 0; s < (int)sections.size(); ++s) {
        auto& sec = sections[s];
        (void)std::fprintf(f, "================================================================\n");
        (void)std::fprintf(f, "  Section %d/8: %s\n", s + 1, sec.title_en);
        (void)std::fprintf(f, "================================================================\n");

        for (auto& r : results) {
            if (std::strcmp(r.section, sec.section_id) != 0) continue;
            (void)std::fprintf(f, "  [%2d] %-45s %s  (%.0f ms)\n",
                         module_idx++, r.name,
                         r.passed ? "PASS" : "FAIL", r.elapsed_ms);
        }

        (void)std::fprintf(f, "  -------- Section Result: %d/%d passed", sec.passed, sec.total);
        if (sec.failed > 0) (void)std::fprintf(f, " (%d FAILED)", sec.failed);
        (void)std::fprintf(f, " (%.0f ms)\n\n", sec.time_ms);
    }

    // -- Grand total ---
    std::fprintf(f, "================================================================\n");
    std::fprintf(f, "  AUDIT VERDICT: %s\n",
                 (total_fail == 0) ? "AUDIT-READY (ALL PASSED)" : "AUDIT-BLOCKED (FAILURES DETECTED)");
    std::fprintf(f, "  TOTAL: %d/%d modules passed  (%.1f s)\n",
                 total_pass, total_pass + total_fail, total_ms / 1000.0);
    std::fprintf(f, "  Platform: %s %s | %s | %s\n",
                 plat.os.c_str(), plat.arch.c_str(),
                 plat.compiler.c_str(), plat.build_type.c_str());
    std::fprintf(f, "================================================================\n");

    std::fclose(f);
}

// ============================================================================
// Resolve output directory (executable dir by default)
// ============================================================================
static std::string get_exe_dir() {
#ifdef _WIN32
    char buf[MAX_PATH] = {};
    GetModuleFileNameA(nullptr, buf, MAX_PATH);
    std::string const path(buf);
    auto pos = path.find_last_of("\\/");
    return (pos != std::string::npos) ? path.substr(0, pos) : ".";
#else
    char buf[4096] = {};
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len <= 0) return ".";
    buf[len] = '\0';
    std::string path(buf);
    auto pos = path.find_last_of('/');
    return (pos != std::string::npos) ? path.substr(0, pos) : ".";
#endif
}

// ============================================================================
// Main
// ============================================================================
static void print_usage() {
    std::printf("Usage: unified_audit_runner [OPTIONS]\n\n");
    std::printf("Options:\n");
    std::printf("  --json-only            Suppress console output; write JSON only\n");
    std::printf("  --report-dir <dir>     Write reports to <dir> (default: exe dir)\n");
    std::printf("  --section <id>         Run only modules in section <id>\n");
    std::printf("  --list-sections        Print available sections and exit\n");
    std::printf("  --help                 Show this message\n\n");
    std::printf("Sections:\n");
    for (int s = 0; s < NUM_SECTIONS; ++s) {
        std::printf("  %-20s %s\n", SECTIONS[s].id, SECTIONS[s].title_en);
    }
}

int main(int argc, char* argv[]) {
    // Parse args
    bool json_only = false;
    std::string report_dir = "";
    std::string section_filter = "";  // empty = run all
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--json-only") == 0) {
            json_only = true;
        } else if (std::strcmp(argv[i], "--report-dir") == 0 && i + 1 < argc) {
            report_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--section") == 0 && i + 1 < argc) {
            section_filter = argv[++i];
        } else if (std::strcmp(argv[i], "--list-sections") == 0) {
            for (int s = 0; s < NUM_SECTIONS; ++s) {
                std::printf("%s\n", SECTIONS[s].id);
            }
            return 0;
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage();
            return 0;
        }
    }
    if (report_dir.empty()) {
        report_dir = get_exe_dir();
    }

    // Validate section filter
    if (!section_filter.empty()) {
        bool found = false;
        for (int s = 0; s < NUM_SECTIONS; ++s) {
            if (section_filter == SECTIONS[s].id) { found = true; break; }
        }
        if (!found) {
            (void)std::fprintf(stderr, "ERROR: unknown section '%s'\n", section_filter.c_str());
            print_usage();
            return 1;
        }
    }

    auto plat = detect_platform();

    auto total_start = std::chrono::steady_clock::now();

    if (!json_only) {
        std::printf("================================================================\n");
        std::printf("  UltrafastSecp256k1 -- Unified Audit Runner\n");
        std::printf("  Library v%s  |  Git: %.8s  |  Framework v%s\n",
                    plat.library_version.c_str(), plat.git_hash.c_str(),
                    plat.framework_version.c_str());
        std::printf("  %s | %s | %s | %s\n",
                    plat.os.c_str(), plat.arch.c_str(),
                    plat.compiler.c_str(), plat.build_type.c_str());
        std::printf("  %s\n", plat.timestamp.c_str());
        if (!section_filter.empty()) {
            std::printf("  Filter: section=%s\n", section_filter.c_str());
}
        std::printf("================================================================\n\n");
    }

    // -- Phase 1: Library selftest ----------------------------------------
    if (!json_only) std::printf("[Phase 1/3] Library selftest (ci mode)...\n");
    auto st_start = std::chrono::steady_clock::now();
    bool const selftest_passed = Selftest(false, SelftestMode::ci, 0);
    auto st_end = std::chrono::steady_clock::now();
    double const selftest_ms = std::chrono::duration<double, std::milli>(st_end - st_start).count();

    if (!json_only) {
        if (selftest_passed) {
            std::printf("[Phase 1/3] Selftest PASSED (%.0f ms)\n\n", selftest_ms);
        } else {
            std::printf("[Phase 1/3] *** Selftest FAILED *** (%.0f ms)\n\n", selftest_ms);
        }
    }

    // -- Phase 2: All test modules (grouped by 8 sections) ----------------
    // Count modules to run (with filter)
    int modules_to_run = 0;
    for (int i = 0; i < NUM_MODULES; ++i) {
        if (section_filter.empty() || section_filter == ALL_MODULES[i].section) {
            ++modules_to_run;
}
    }

    if (!json_only) {
        std::printf("[Phase 2/3] Running %d test modules across %d audit sections...\n\n",
                    modules_to_run, NUM_SECTIONS);
    }

    std::vector<ModuleResult> results;
    results.reserve(NUM_MODULES);

    int modules_passed = 0;
    int modules_failed = 0;

    // Track which section we're in for console grouping
    const char* current_section = "";
    int section_num = 0;
    int run_idx = 0;

    for (int i = 0; i < NUM_MODULES; ++i) {
        auto& m = ALL_MODULES[i];

        // Apply section filter
        if (!section_filter.empty() && section_filter != m.section) {
            continue;
}

        // Print section header on transition
        if (!json_only && std::strcmp(m.section, current_section) != 0) {
            current_section = m.section;
            ++section_num;
            // Find the section title
            for (int s = 0; s < NUM_SECTIONS; ++s) {
                if (std::strcmp(SECTIONS[s].id, current_section) == 0) {
                    std::printf("  ----------------------------------------------------------\n");
                    std::printf("  Section %d/8: %s\n", section_num, SECTIONS[s].title_en);
                    std::printf("  ----------------------------------------------------------\n");
                    break;
                }
            }
        }

        ++run_idx;
        if (!json_only) {
            std::printf("  [%2d/%d] %-45s ", run_idx, modules_to_run, m.name);
            (void)std::fflush(stdout);
        }

        auto t0 = std::chrono::steady_clock::now();
        int const rc = m.run();
        auto t1 = std::chrono::steady_clock::now();
        double const ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        bool const ok = (rc == 0);
        if (ok) {
            ++modules_passed;
            if (!json_only) std::printf("PASS  (%.0f ms)\n", ms);
        } else {
            ++modules_failed;
            if (!json_only) std::printf("FAIL  (%.0f ms)\n", ms);
        }

        results.push_back({ m.id, m.name, m.section, ok, ms });
    }

    auto total_end = std::chrono::steady_clock::now();
    double const total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    // -- Phase 3: Generate reports ---------------------------------------
    if (!json_only) std::printf("\n[Phase 3/3] Generating audit reports...\n");

    std::string const json_path = report_dir + "/audit_report.json";
    std::string const text_path = report_dir + "/audit_report.txt";

    write_json_report(json_path.c_str(), plat, results, selftest_passed, selftest_ms, total_ms);
    if (!json_only) {
        write_text_report(text_path.c_str(), plat, results, selftest_passed, selftest_ms, total_ms);
    }

    if (!json_only) {
        std::printf("  JSON: %s\n", json_path.c_str());
        std::printf("  Text: %s\n", text_path.c_str());
    }

    // -- Section Summary Table -------------------------------------------
    auto sections = compute_section_summaries(results);

    if (!json_only) {
        std::printf("\n================================================================\n");
        std::printf("  %-4s %-50s %s\n", "#", "Audit Section", "Result");
        std::printf("  ---- -------------------------------------------------- ------\n");
        for (int s = 0; s < (int)sections.size(); ++s) {
            auto& sec = sections[s];
            if (sec.total == 0) continue;  // skip empty sections (filtered)
            std::printf("  %-4d %-50s %d/%d %s\n",
                        s + 1, sec.title_en, sec.passed, sec.total,
                        sec.failed == 0 ? "PASS" : "FAIL");
        }
    }

    // -- Final Summary ---------------------------------------------------
    int const total_pass = modules_passed + (selftest_passed ? 1 : 0);
    int const total_fail = modules_failed + (selftest_passed ? 0 : 1);
    int const total_count = total_pass + total_fail;

    if (!json_only) {
        std::printf("\n================================================================\n");
        std::printf("  AUDIT VERDICT: %s\n",
                    (total_fail == 0) ? "AUDIT-READY" : "AUDIT-BLOCKED");
        std::printf("  TOTAL: %d/%d modules passed", total_pass, total_count);
        if (total_fail == 0) {
            std::printf("  --  ALL PASSED");
        } else {
            std::printf("  --  %d FAILED", total_fail);
        }
        std::printf("  (%.1f s)\n", total_ms / 1000.0);
        std::printf("  Platform: %s %s | %s | %s\n",
                    plat.os.c_str(), plat.arch.c_str(),
                    plat.compiler.c_str(), plat.build_type.c_str());
        std::printf("================================================================\n");
    }

    return total_fail > 0 ? 1 : 0;
}
