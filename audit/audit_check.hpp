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

#endif // AUDIT_CHECK_HPP_
