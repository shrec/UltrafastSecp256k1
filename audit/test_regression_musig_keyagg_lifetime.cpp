// ============================================================================
// test_regression_musig_keyagg_lifetime.cpp
// ============================================================================
// Regression for the shim MuSig2 keyagg-cache USE-AFTER-FREE (blind-zone #4):
// the same unlock-then-use-raw-pointer class as PRECOMPUTE-GCONTEXT-UAF, in
// compat/libsecp256k1_shim/src/shim_musig.cpp.
//
//   ka_get()/ka_get_by_token() looked up a session entry in a mutex-guarded map
//   `g_ka` (value type std::unique_ptr<KAEntry>) and returned `it->second.get()`
//   — a RAW pointer. The lock_guard is released at function return, so a caller
//   (partial_sign / nonce_process / pubkey_get / ...) dereferenced that raw
//   pointer AFTER the lock was gone. A concurrent ka_remove / partial_sig_agg
//   could erase the entry, freeing the unique_ptr<KAEntry> — and the secret-
//   adjacent KAEntry::ctx — while the first thread was still using it (UAF / torn
//   read of secret-adjacent state).
//
//   Fix: g_ka holds std::shared_ptr<KAEntry>, and the accessors return a
//   shared_ptr SNAPSHOT (`return it->second;`) — never a raw `it->second.get()`.
//   A caller holding the snapshot keeps the KAEntry alive for the whole op even
//   if a concurrent ka_remove erases the map entry. Mirrors the g_context fix.
//
// Source-scan (deterministic, no shim link required): assert the shared_ptr map
// + snapshot accessors are in place and the raw-`.get()` escape is gone.
// ============================================================================

#include <cstdio>
#include <fstream>
#include <string>
#include <iterator>

#include "audit_check.hpp"

static int g_pass = 0, g_fail = 0;

// Repair (issue #335 acceptance repair, round 5): shim_musig.cpp is an
// in-tree source file that always exists (a source-scan needs the file
// text, not a built shim target) -- this module is also registered
// advisory=false in ALL_MODULES (mandatory), so returning
// ADVISORY_SKIP_CODE on a resolution failure was already internally
// inconsistent with its own registration (the unified runner still counts
// a non-zero, non-{0,77} rc from a non-advisory module as a hard FAIL, so
// this was never a silent PASS -- but it also never resolved the real
// source from a CWD unrelated to the repo, unlike every sibling in-tree
// source file this same round). Route through the shared,
// UFSECP_SOURCE_ROOT-aware audit_read_source_file() (audit_check.hpp) and
// hard-fail via CHECK() instead of the advisory-skip sentinel.
static std::string read_source_file(const char* rel_path) {
    return audit_read_source_file(rel_path);
}

int test_regression_musig_keyagg_lifetime_run() {
    g_pass = 0; g_fail = 0;
    printf("======================================================================\n");
    printf("  Regression: shim MuSig2 keyagg-cache lifetime (UAF blind-zone #4)\n");
    printf("======================================================================\n\n");

    std::string src = read_source_file("compat/libsecp256k1_shim/src/shim_musig.cpp");
    if (src.empty()) src = read_source_file("shim_musig.cpp");
    CHECK(!src.empty(), "shim_musig.cpp must be readable (in-tree source always exists)");
    if (src.empty()) {
        printf("\n[regression_musig_keyagg_lifetime] %d/%d checks passed\n",
               g_pass, g_pass + g_fail);
        return 1;
    }

    // The map must hold shared_ptr so a snapshot outlives a concurrent erase.
    CHECK(src.find("std::shared_ptr<KAEntry>> g_ka") != std::string::npos,
          "g_ka must hold std::shared_ptr<KAEntry> (lifetime-safe snapshot)");
    // The unique_ptr map must be gone (the unlock-then-use-raw source).
    CHECK(src.find("std::unique_ptr<KAEntry>> g_ka") == std::string::npos,
          "g_ka must NOT be std::unique_ptr<KAEntry> (regression of the UAF class)");
    // Accessors must return a shared_ptr, not a raw KAEntry*.
    CHECK(src.find("std::shared_ptr<KAEntry> ka_get(") != std::string::npos,
          "ka_get must return std::shared_ptr<KAEntry> (snapshot, not raw)");
    CHECK(src.find("std::shared_ptr<KAEntry> ka_get_by_token(") != std::string::npos,
          "ka_get_by_token must return std::shared_ptr<KAEntry> (snapshot, not raw)");
    // The raw `it->second.get()` escape from the guarded map must be gone.
    CHECK(src.find("return it->second.get()") == std::string::npos,
          "no accessor may return a raw it->second.get() from the mutex-guarded map (UAF)");

    printf("\n[regression_musig_keyagg_lifetime] %d/%d checks passed\n",
           g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_musig_keyagg_lifetime_run(); }
#endif
