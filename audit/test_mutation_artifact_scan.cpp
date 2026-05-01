// ============================================================================
// Mutation Artifact Scanner — Build-Time Source Integrity Check
// ============================================================================
// Track:  SECURITY — prevents corrupted source from shipping in releases
//
// BACKGROUND:
//   The mutation_kill_rate.py script intentionally introduces source mutations
//   and then reverts them.  If the script is killed mid-mutation (OOM, signal,
//   etc.) the corrupted source can persist undetected.  This scanner reads
//   the actual C++ source files at test-time and checks for known-bad
//   patterns that indicate an un-reverted mutation.
//
// WHAT IT CHECKS:
//   MA-1: point.cpp wNAF trim loop must use "> 0", not ">= 0"
//   MA-2: scalar.cpp divsteps mask must use "& m", not "| m"
//   MA-3: Generic bitwise-op mutation patterns in critical functions
//
// This test requires UFSECP_SOURCE_ROOT to be defined at compile time
// (set by CMake) pointing to the library source root directory.
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <fstream>
#include <sstream>
#include <regex>

static int g_pass = 0, g_fail = 0;
#include "audit_check.hpp"

#ifndef UFSECP_SOURCE_ROOT
#error "UFSECP_SOURCE_ROOT must be defined (set by CMake target_compile_definitions)"
#endif

// Read entire file into string
static std::string read_file_contents(const std::string& path) {
    std::ifstream ifs(path, std::ios::in);
    if (!ifs.is_open()) return {};
    std::ostringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

// Check that a pattern does NOT appear in the file
static void check_absent(const std::string& contents, const std::string& filename,
                          const std::regex& bad_pattern, const char* label) {
    if (contents.empty()) {
        std::printf("  [SKIP] %s: file not readable\n", label);
        return;
    }
    bool found = std::regex_search(contents, bad_pattern);
    CHECK(!found, label);
    if (found) {
        std::printf("         ^^^ DANGER: mutation artifact detected in %s\n", filename.c_str());
        std::printf("         This may indicate mutation_kill_rate.py was killed mid-mutation.\n");
        std::printf("         Fix: git checkout -- %s\n", filename.c_str());
    }
}

// Check that a pattern DOES appear in the file (expected correct code)
static void check_present(const std::string& contents, const std::string& filename,
                           const std::regex& good_pattern, const char* label) {
    if (contents.empty()) {
        std::printf("  [SKIP] %s: file not readable\n", label);
        return;
    }
    bool found = std::regex_search(contents, good_pattern);
    CHECK(found, label);
    if (!found) {
        std::printf("         ^^^ WARNING: expected code pattern missing in %s\n", filename.c_str());
    }
}

// ============================================================================
// MA-1: point.cpp wNAF trim loop integrity
// ============================================================================
// The correct code has: while (len_a_hi > 0 && wnaf_a_hi[len_a_hi - 1] == 0)
// The mutated code had: while (len_a_hi >= 0 && wnaf_a_hi[len_a_hi - 1] == 0)
//                                    ^^
// The >= 0 mutation causes wnaf_a_hi[-1] OOB read when len_a_hi reaches 0.
static void test_point_wnaf_trim() {
    std::printf("[MA-1] point.cpp wNAF trim loop integrity\n");

    std::string path = std::string(UFSECP_SOURCE_ROOT) + "/src/cpu/src/point.cpp";
    std::string contents = read_file_contents(path);

    if (contents.empty()) {
        std::printf("  [SKIP] Cannot read %s\n", path.c_str());
        return;
    }

    // BAD: len_a_hi >= 0 (or len_a_lo >= 0, len_b_lo >= 0, len_b_hi >= 0)
    // These are the wNAF trim loops — they MUST use > 0, never >= 0
    std::regex bad_trim(R"(while\s*\(\s*len_[ab]_(hi|lo)\s*>=\s*0)");
    check_absent(contents, "src/cpu/src/point.cpp", bad_trim,
                 "MA-1a: no wNAF trim loop uses >= 0 (OOB guard)");

    // GOOD: len_a_hi > 0 pattern must be present
    std::regex good_trim(R"(while\s*\(\s*len_a_hi\s*>\s*0)");
    check_present(contents, "src/cpu/src/point.cpp", good_trim,
                  "MA-1b: wNAF trim loop uses > 0 (correct pattern present)");
}

// ============================================================================
// MA-2: scalar.cpp divsteps mask integrity
// ============================================================================
// The correct code has: w = (uint32_t)((f * g * (f * f - 2)) & m);
// The mutated code had: w = (uint32_t)((f * g * (f * f - 2)) | m);
//                                                              ^
// The | m mutation breaks the modular inverse (divsteps algorithm),
// corrupting ALL ECDSA sign/verify/recover operations.
static void test_scalar_divsteps_mask() {
    std::printf("[MA-2] scalar.cpp divsteps mask integrity\n");

    std::string path = std::string(UFSECP_SOURCE_ROOT) + "/src/cpu/src/scalar.cpp";
    std::string contents = read_file_contents(path);

    if (contents.empty()) {
        std::printf("  [SKIP] Cannot read %s\n", path.c_str());
        return;
    }

    // BAD: "f * f - 2)) | m" — the mask must be AND, not OR
    std::regex bad_mask(R"(\(f\s*\*\s*f\s*-\s*2\)\)\s*\|\s*m\s*\))");
    check_absent(contents, "src/cpu/src/scalar.cpp", bad_mask,
                 "MA-2a: divsteps mask uses & not | (inverse integrity)");

    // GOOD: "f * f - 2)) & m" pattern must be present
    std::regex good_mask(R"(\(f\s*\*\s*f\s*-\s*2\)\)\s*&\s*m\s*\))");
    check_present(contents, "src/cpu/src/scalar.cpp", good_mask,
                  "MA-2b: divsteps mask uses & m (correct pattern present)");
}

// ============================================================================
// MA-3: field.cpp divsteps mask integrity (same pattern in field inverse)
// ============================================================================
// The field element inverse uses the same safegcd/divsteps algorithm.
// Same mutation risk: & m → | m.
static void test_field_divsteps_mask() {
    std::printf("[MA-3] field.cpp divsteps mask integrity\n");

    std::string path = std::string(UFSECP_SOURCE_ROOT) + "/src/cpu/src/field.cpp";
    std::string contents = read_file_contents(path);

    if (contents.empty()) {
        std::printf("  [SKIP] Cannot read %s\n", path.c_str());
        return;
    }

    // BAD: same | m pattern in field divsteps
    std::regex bad_mask(R"(\(f\s*\*\s*f\s*-\s*2\)\)\s*\|\s*m\s*\))");
    check_absent(contents, "src/cpu/src/field.cpp", bad_mask,
                 "MA-3a: field divsteps mask uses & not | (field inverse integrity)");
}

// ============================================================================
// MA-4: Generic relational mutation check in point operations
// ============================================================================
// Broader scan: no `>= 0` in array-index guard contexts within point.cpp
// These patterns should always use `> 0` to prevent OOB access.
static void test_point_generic_relational() {
    std::printf("[MA-4] point.cpp generic relational guards\n");

    std::string path = std::string(UFSECP_SOURCE_ROOT) + "/src/cpu/src/point.cpp";
    std::string contents = read_file_contents(path);

    if (contents.empty()) {
        std::printf("  [SKIP] Cannot read %s\n", path.c_str());
        return;
    }

    // BAD: len_XX >= 0 followed by array[len_XX - 1]
    // This specific pattern (variable >= 0 && array[variable - 1]) is always
    // a bug because it allows variable=0 → array[-1] = OOB.
    std::regex bad_guard(R"(len_\w+\s*>=\s*0\s*&&\s*\w+\[len_\w+\s*-\s*1\])");
    check_absent(contents, "src/cpu/src/point.cpp", bad_guard,
                 "MA-4a: no len_X >= 0 && arr[len_X-1] pattern (OOB guard)");
}

// ============================================================================
// main
// ============================================================================

int test_mutation_artifact_scan_run() {
    std::printf("============================================================\n");
    std::printf("Mutation Artifact Scanner — Source Integrity Check\n");
    std::printf("============================================================\n");
    std::printf("Source root: %s\n\n", UFSECP_SOURCE_ROOT);

    test_point_wnaf_trim();
    test_scalar_divsteps_mask();
    test_field_divsteps_mask();
    test_point_generic_relational();

    std::printf("\n============================================================\n");
    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    std::printf("============================================================\n");
    return g_fail > 0 ? 1 : 0;
}

#ifdef STANDALONE_TEST
int main() { return test_mutation_artifact_scan_run(); }
#endif
