#pragma once

#include <cstdint>

namespace secp256k1::fast {

// ── Selftest execution modes ──
// Controls which test subsets run and how many iterations stress tests perform.
//
//   smoke  — 1-2 seconds, core KAT vectors only (suitable for app startup)
//   ci     — 30-90 seconds, full coverage (GitHub Actions / every push)
//   stress — 10-60 minutes, extended iterations + big sweeps (nightly / manual)
enum class SelftestMode : uint8_t {
    smoke  = 0,   // Core vectors + field/scalar identities (~1-2 s)
    ci     = 1,   // All tests including batch sweeps, bilinearity, NAF (~30-90 s)
    stress = 2    // ci + extended iterations, large random sweeps (~10-60 min)
};

// Run comprehensive self-tests on the library
// Returns true if all tests pass, false otherwise
// Set verbose=true to see detailed test output
bool Selftest(bool verbose);

// Run self-tests with explicit mode and deterministic PRNG seed.
// seed=0 uses default seed (deterministic but fixed).
// Prints repro bundle: commit, compiler, platform, seed, mode.
bool Selftest(bool verbose, SelftestMode mode, uint64_t seed = 0);

} // namespace secp256k1::fast
