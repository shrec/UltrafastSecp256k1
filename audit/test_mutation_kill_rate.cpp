// ============================================================================
// test_mutation_kill_rate.cpp -- Mutation kill-rate check for unified runner
// ============================================================================
//
// Advisory module: runs scripts/mutation_kill_rate.py with --ctest-mode and
// checks that the mutation kill rate is above the minimum threshold.
//
// ADVISORY = true:  failure produces a WARN, not a hard FAIL.
// Reason: mutation testing requires Python 3, a build dir, and takes time.
//         It is a quality gate, not a correctness gate.
//
// When Python or the build dir is unavailable, the module passes silently
// (skip-if-not-available semantics).  CI jobs that DO have Python run it fully.
//
// Build deps: none — uses popen() to invoke the Python script.
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#ifdef _WIN32
#  include <windows.h>
#  define popen  _popen
#  define pclose _pclose
#else
#  include <unistd.h>
#endif

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Detect the build directory from the running executable's path.
// The unified audit runner lives at <build>/audit/unified_audit_runner,
// so the build root is the parent of the audit/ directory.
static std::string detect_build_dir() {
#ifdef _WIN32
    char buf[MAX_PATH] = {};
    DWORD len = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (len == 0) return "";
    std::string path(buf, len);
    // Strip filename -> <build>/audit
    auto pos = path.find_last_of("\\/");
    if (pos == std::string::npos) return "";
    std::string dir = path.substr(0, pos);
    // Strip "audit" (or "Release"/"Debug" under MSVC) -> <build>
    pos = dir.find_last_of("\\/");
    return (pos != std::string::npos) ? dir.substr(0, pos) : dir;
#else
    char buf[4096] = {};
    ssize_t const len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len <= 0) return "";
    buf[len] = '\0';
    std::string path(buf);
    // Strip filename -> <build>/audit
    auto pos = path.find_last_of('/');
    if (pos == std::string::npos) return "";
    std::string dir = path.substr(0, pos);
    // Strip "audit" -> <build>
    pos = dir.find_last_of('/');
    return (pos != std::string::npos) ? dir.substr(0, pos) : dir;
#endif
}

// Find the script relative to this executable (or fallback paths)
static std::string find_script() {
    // Try relative paths from the common build output locations
    static const char* kCandidates[] = {
        "../scripts/mutation_kill_rate.py",     // from build/audit/
        "scripts/mutation_kill_rate.py",        // from repo root
        "../../scripts/mutation_kill_rate.py",  // from deep build dir
        nullptr
    };
    for (int i = 0; kCandidates[i]; ++i) {
        if (FILE* f = std::fopen(kCandidates[i], "r")) {
            std::fclose(f);
            return kCandidates[i];
        }
    }
    return "";
}

// Check if python3 is available
static bool python3_available() {
#ifdef _WIN32
    return std::system("python3 --version >NUL 2>&1") == 0;
#else
    return std::system("python3 --version >/dev/null 2>&1") == 0;
#endif
}

// ---------------------------------------------------------------------------
// _run()
// ---------------------------------------------------------------------------
int test_mutation_kill_rate_run() {
    // Skip gracefully when Python 3 is not available
    if (!python3_available()) {
        std::printf("[mutation_kill_rate] python3 not available — skipping (advisory)\n");
        return 0;
    }

    std::string script = find_script();
    if (script.empty()) {
        std::printf("[mutation_kill_rate] script not found — skipping (advisory)\n");
        return 0;
    }

    // --ctest-mode: quick run (50 mutations, exit 0 if ≥ threshold, 1 if below)
    // --count 50: fast enough for CI (~30–60 s depending on build speed)
    // --threshold 60: slightly relaxed for CI (full run uses 75%)
    // --build-dir: auto-detected from the running executable's location so the
    //              script finds the correct build tree (build_opencl, build-audit, etc.)
    // --test-timeout 180: generous for CI runners (test_comprehensive_standalone can be slow)
    std::string build_dir = detect_build_dir();
    std::string cmd = "python3 " + script +
                      " --ctest-mode --count 50 --threshold 60 --test-timeout 180";
    if (!build_dir.empty()) {
        cmd += " --build-dir " + build_dir;
    }
    cmd += " 2>&1";

    std::printf("[mutation_kill_rate] Running: %s\n", cmd.c_str());

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        std::printf("[mutation_kill_rate] popen failed — skipping (advisory)\n");
        return 0;
    }

    // Stream output to stdout so it appears in CI logs
    char buf[256];
    while (std::fgets(buf, sizeof(buf), pipe)) {
        std::printf("  %s", buf);
    }

    int rc = pclose(pipe);
    int exit_code = 0;
#ifndef _WIN32
    if (WIFEXITED(rc)) exit_code = WEXITSTATUS(rc);
#else
    exit_code = rc;
#endif

    if (exit_code == 0) {
        std::printf("[mutation_kill_rate] PASS — kill rate above threshold\n");
    } else {
        std::printf("[mutation_kill_rate] WARN — kill rate below threshold (exit %d)\n", exit_code);
    }

    return exit_code;
}
