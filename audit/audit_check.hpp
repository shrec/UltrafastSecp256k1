// ============================================================================
// audit/audit_check.hpp -- Shared sub-test harness for audit modules
// ============================================================================
//
// Centralised CHECK macro with periodic progress output.
// Designed for maximum portability: pure ASCII, newline-only (no \r),
// works on serial consoles, SSH, CTest, Docker, CI pipelines.
//
// USAGE -- every .cpp file that includes this must declare at file scope:
//
//     static int g_pass = 0, g_fail = 0;
//
// Output:
//   PASS  -> every N passes prints:  "    [####........] 4096/est OK"
//   FAIL  -> immediate:              "  [FAIL] msg (line N)"
//
// ============================================================================
#ifndef AUDIT_CHECK_HPP_
#define AUDIT_CHECK_HPP_

#include <cstdio>
#include <string>
#include <fstream>
#include <iterator>

// -- Message helper: converts const char* and std::string to const char* ----
//    Needed to support CHECK(cond, "literal") and CHECK(cond, "x"+to_string(i))
//    without -Wnon-pod-varargs errors on Clang (Android NDK, ESP-IDF).
namespace audit_detail {
inline const char* to_cstr(const char* s) noexcept { return s; }
inline const char* to_cstr(const std::string& s) noexcept { return s.c_str(); }
} // namespace audit_detail

// -- Advisory skip sentinel (MEDIUM-5 fix) -----------------------------------
// Advisory modules that cannot run due to absent infrastructure (e.g. no
// Python, no Cryptol, no OpenSSL) MUST return this code instead of 0 so that
// unified_audit_runner can classify them as advisory_skipped rather than
// advisory_failed.  The runner uses this as the primary classifier; the
// legacy elapsed_ms < 1.0 heuristic is kept as a backward-compat fallback.
// Value 77 is borrowed from autotools "skip" convention (SKIP_TEST).
#ifndef ADVISORY_SKIP_CODE
#define ADVISORY_SKIP_CODE 77
#endif

// -- How often to print progress (power of 2, fast bitmask) -----------------
#ifndef AUDIT_PROGRESS_INTERVAL
#define AUDIT_PROGRESS_INTERVAL 4096
#endif

// -- Bar geometry -----------------------------------------------------------
#define AUDIT_BAR_WIDTH 20

// -- Internal: print a simple ASCII progress bar ----------------------------
//    [########............] 8192 OK
//    Pure newline output, no \r, no ANSI escapes, serial-safe.
inline void audit_print_progress_(int total) {
    // Each tick = AUDIT_PROGRESS_INTERVAL checks
    int ticks = total / AUDIT_PROGRESS_INTERVAL;
    int filled = ticks;
    if (filled > AUDIT_BAR_WIDTH) filled = AUDIT_BAR_WIDTH;
    char bar[AUDIT_BAR_WIDTH + 1];
    for (int i = 0; i < filled; ++i)            bar[i] = '#';
    for (int i = filled; i < AUDIT_BAR_WIDTH; ++i) bar[i] = '.';
    bar[AUDIT_BAR_WIDTH] = '\0';
    (void)std::printf("    [%s] %d OK\n", bar, total);
    (void)std::fflush(stdout);
}

// -- Core assertion macro ---------------------------------------------------
#define CHECK(cond, msg) do { \
    if (cond) { \
        ++g_pass; \
        if ((g_pass & (AUDIT_PROGRESS_INTERVAL - 1)) == 0) { \
            audit_print_progress_(g_pass); \
        } \
    } else { \
        ++g_fail; \
        (void)std::printf("  [FAIL] %s (line %d)\n", ::audit_detail::to_cstr(msg), __LINE__); \
        (void)std::fflush(stdout); \
    } \
} while(0)

// -- Lowercase alias for legacy tests --------------------------------------
// Several shim regression tests (test_regression_ellswift_ct_path.cpp,
// test_regression_musig2_nonce_strict.cpp, test_regression_shim_pubkey_sort.cpp,
// test_regression_shim_per_context_blinding.cpp, test_regression_musig2_session_token.cpp)
// use lowercase `check(...)`. Provide an alias so they compile in BOTH the
// unified_audit_runner (where the shim is now linked, after the CMakeLists
// order fix that put compat/libsecp256k1_shim BEFORE audit) and as standalone
// CTest targets. Lowercase `check` matches the local `static void check(...)`
// helper used in newer tests like test_regression_musig2_abi_signer_index.cpp
// and is interchangeable with CHECK at the test-author level.
#ifndef check
#define check(cond, msg) CHECK(cond, msg)
#endif

// -- Section/progress header with immediate flush ---------------------------
#define AUDIT_LOG(...) do { \
    (void)std::printf(__VA_ARGS__); \
    (void)std::fflush(stdout); \
} while(0)

// ============================================================================
// audit_read_source_file() -- CWD-independent source-file resolution
// ============================================================================
// Repair (issue #335 acceptance repair, round 5): several source-reading
// regression modules resolved their required in-tree source (e.g.
// src/cpu/src/bip39.cpp, src/cpu/src/frost.cpp) using CWD-relative path lists
// or bounded CWD-relative walk-ups ONLY. When unified_audit_runner is invoked
// directly from a CWD unrelated to the repo (e.g. /tmp), those resolvers
// return empty -- and several call sites treated "not found" as a silent
// 0-checks-executed skip (printing a [SKIP] line but never calling CHECK()),
// so the module still reported an overall PASS with the actual regression
// coverage vacuously absent. That is a false green, not a legitimate skip.
//
// unified_audit_runner's own CMakeLists.txt target_compile_definitions()
// already bakes UFSECP_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.." -- a
// compile-time ABSOLUTE path to the repo root, fixed at build time on the
// SAME machine that later runs the binary. Resolving via this macro first
// makes source lookup independent of BOTH the process's current working
// directory AND the executable's own install/relocate location (the
// resolve_opencl_kernel() exe-relative walk-up in
// src/gpu_backend_opencl.cpp does not apply here: OpenCL kernels ship
// alongside install layouts and can genuinely move, but these are in-tree
// C++ sources read by a test binary built directly from them -- the
// compile-time root is exact, not a heuristic).
//
// The CWD-relative walk-up remains as a fallback for translation units built
// WITHOUT UFSECP_SOURCE_ROOT (standalone CTest targets that don't add the
// define) so their existing ctest WORKING_DIRECTORY-based invocation keeps
// working unchanged.
//
// Callers MUST treat an empty return as a hard failure (CHECK(!src.empty(),
// ...)), never a silent skip -- "the source could not be found" is itself
// the finding when the source is known to always exist in-tree.
inline std::string audit_read_source_file(const char* rel_path) {
#ifdef UFSECP_SOURCE_ROOT
    {
        std::string const full = std::string(UFSECP_SOURCE_ROOT) + "/" + rel_path;
        std::ifstream f(full, std::ios::in | std::ios::binary);
        if (f.is_open()) {
            return std::string(std::istreambuf_iterator<char>(f),
                                std::istreambuf_iterator<char>());
        }
    }
#endif
    std::string up;
    for (int depth = 0; depth <= 8; ++depth) {
        std::ifstream f(up + rel_path, std::ios::in | std::ios::binary);
        if (f.is_open()) {
            return std::string(std::istreambuf_iterator<char>(f),
                                std::istreambuf_iterator<char>());
        }
        up += "../";
    }
    return {};
}

#endif // AUDIT_CHECK_HPP_
