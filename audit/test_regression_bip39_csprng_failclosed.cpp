// ============================================================================
// test_regression_bip39_csprng_failclosed.cpp
// ============================================================================
// ENTROPY-SOURCE-INTEGRITY regression (blind-zone #5): bip39.cpp shipped a local
// fail-OPEN csprng_fill (returned false on /dev/urandom failure / short read) that
// duplicated and weakened the single canonical fail-CLOSED detail::csprng_fill
// (std::abort on RNG failure). Key material must never be derived from a degraded
// entropy source. Fix: bip39.cpp now uses detail::csprng_fill.
//
//   [1] Source-scan: bip39.cpp defines no local csprng_fill and routes through the
//       canonical detail::csprng_fill (the structural single-source guard).
//   [2] Functional smoke: bip39_generate with CSPRNG entropy produces a valid
//       mnemonic, and a fixed entropy is deterministic + validates.
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <fstream>
#include <string>
#include <iterator>
#include <array>

#include "secp256k1/bip39.hpp"
#include "audit_check.hpp"

using namespace secp256k1;

static int g_pass = 0, g_fail = 0;

int test_regression_bip39_csprng_failclosed_run() {
    g_pass = 0; g_fail = 0;
    printf("======================================================================\n");
    printf("  Regression: BIP-39 fail-closed single-source CSPRNG (blind-zone #5)\n");
    printf("======================================================================\n\n");

    // [1] Source-scan: no local csprng_fill; routes through the canonical helper.
    // Fail-closed (issue #335 acceptance repair, round 5): bip39.cpp always exists
    // in-tree. A failed read means the source could not be resolved (CWD-dependence
    // bug or a genuinely broken build layout), NOT that the property holds -- this
    // must never be a silent 0-checks-executed skip that lets the module PASS.
    std::string src = audit_read_source_file("src/cpu/src/bip39.cpp");
    if (src.empty()) src = audit_read_source_file("bip39.cpp");
    CHECK(!src.empty(), "bip39.cpp must be readable (in-tree source always exists)");
    if (!src.empty()) {
        CHECK(src.find("bool csprng_fill(") == std::string::npos,
              "bip39.cpp must NOT define a local fail-open csprng_fill (single-source)");
        CHECK(src.find("detail::csprng_fill(") != std::string::npos,
              "bip39.cpp must use the canonical fail-closed detail::csprng_fill");
    }

    // [2] Functional smoke: CSPRNG-generated mnemonic is valid.
    {
        auto [mn, ok] = bip39_generate(16, nullptr);  // 16 bytes -> 12 words, CSPRNG path
        CHECK(ok && !mn.empty(), "bip39_generate(CSPRNG) produces a mnemonic");
        CHECK(bip39_validate(mn), "CSPRNG-generated mnemonic validates");
    }
    // Fixed entropy: deterministic + valid.
    {
        std::array<std::uint8_t, 16> ent{};
        ent[0] = 0x0F; ent[15] = 0xA5;
        auto [m1, ok1] = bip39_generate(16, ent.data());
        auto [m2, ok2] = bip39_generate(16, ent.data());
        CHECK(ok1 && ok2 && m1 == m2, "fixed entropy -> deterministic mnemonic");
        CHECK(bip39_validate(m1), "fixed-entropy mnemonic validates");
    }

    printf("\n[regression_bip39_csprng_failclosed] %d/%d checks passed\n",
           g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_bip39_csprng_failclosed_run(); }
#endif
