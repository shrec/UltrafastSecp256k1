#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace secp256k1::fast {

// -- Selftest execution modes --
// Controls which test subsets run and how many iterations stress tests perform.
//
//   smoke  -- 1-2 seconds, core KAT vectors only (suitable for app startup)
//   ci     -- 30-90 seconds, full coverage (GitHub Actions / every push)
//   stress -- 10-60 minutes, extended iterations + big sweeps (nightly / manual)
enum class SelftestMode : uint8_t {
    smoke  = 0,   // Core vectors + field/scalar identities (~1-2 s)
    ci     = 1,   // All tests including batch sweeps, bilinearity, NAF (~30-90 s)
    stress = 2    // ci + extended iterations, large random sweeps (~10-60 min)
};

// -- Structured selftest result (for bindings / programmatic use) --
struct SelftestCaseResult {
    std::string name;    // e.g. "scalar_mul_KAT_10_vectors"
    bool        passed;  // true = PASS
    std::string detail;  // empty on pass; failure description on fail
};

struct SelftestReport {
    bool                          all_passed;  // true if every case passed
    int                           total;       // number of test cases run
    int                           passed;      // number that passed
    std::string                   mode;        // "smoke", "ci", or "stress"
    uint64_t                      seed;        // PRNG seed used
    std::string                   platform;    // e.g. "x86_64 clang-17"
    std::vector<SelftestCaseResult> cases;     // per-test results

    // Render as human-readable multi-line text
    std::string to_text() const;

    // Render as JSON string
    std::string to_json() const;
};

// Run comprehensive self-tests on the library
// Returns true if all tests pass, false otherwise
// Set verbose=true to see detailed test output
bool Selftest(bool verbose);

// Run self-tests with explicit mode and deterministic PRNG seed.
// seed=0 uses default seed (deterministic but fixed).
// Prints repro bundle: commit, compiler, platform, seed, mode.
bool Selftest(bool verbose, SelftestMode mode, uint64_t seed = 0);

// Run self-tests and return a structured report (no stdout output).
// Suitable for bindings (Python, Rust, Node.js, etc.) that need
// a programmatic result rather than console output.
SelftestReport selftest_report(SelftestMode mode = SelftestMode::smoke,
                               uint64_t seed = 0);

} // namespace secp256k1::fast
