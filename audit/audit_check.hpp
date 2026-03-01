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
        (void)std::printf("  [FAIL] %s (line %d)\n", (msg), __LINE__); \
        (void)std::fflush(stdout); \
    } \
} while(0)

// -- Section/progress header with immediate flush ---------------------------
#define AUDIT_LOG(...) do { \
    (void)std::printf(__VA_ARGS__); \
    (void)std::fflush(stdout); \
} while(0)

#endif // AUDIT_CHECK_HPP_
