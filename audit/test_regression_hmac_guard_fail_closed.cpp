// ============================================================================
// test_regression_hmac_guard_fail_closed.cpp
// ============================================================================
// Regression: the three RFC 6979 HMAC helpers in src/cpu/src/ecdsa.cpp must
// ZERO their 32-byte output before returning on a length-precondition
// violation, not return with the caller's buffer untouched.
//
// Root cause
//   HMAC_Ctx::compute_short, ::compute_two_block and ::compute_three_block each
//   carry a length guard that exists to stop a size_t wrap in a later length
//   computation (msg_len > 55 makes `55 - msg_len` ~0ULL; msg_len < 128 makes
//   `msg_len - 128` ~0ULL). The guards did their job, but they returned with
//   `out[32]` never written -- so a caller that hit one read back whatever was
//   on its stack and could not tell that from a real HMAC.
//
//   Commit d2544fc8 added `std::memset(out, 0, 32);` to all three guards. It
//   shipped WITHOUT a test; `ci/check_security_fix_has_test.py --since d2544fc8~1`
//   names it. This module is that missing test.
//
// Why this is a source scan and not a behavioural test
//   HMAC_Ctx lives in an anonymous namespace inside ecdsa.cpp. It has no
//   header, no external linkage and no ABI entry point, so no test translation
//   unit can construct one and call the helpers. Every production caller passes
//   a compile-time-fixed length (32, 33, 97, 113, 129, 145), so the guarded
//   branch is unreachable from outside the file by construction -- which is
//   also why it is a latent hazard rather than a live bug, and why it needs
//   pinning: the next caller with a variable length is the one who finds out.
//
//   The scan is therefore the strongest available guard, and it is exact: it
//   requires the memset to be present in each of the three guard statements,
//   and it fails if any guard reverts to a bare `return;`.
//
// What is deliberately NOT claimed here
//   The in-file rationale at ecdsa.cpp says a zeroed output is "rejected by
//   parse_bytes_strict_nonzero". That holds only at the six candidate sites
//   (the compute_short(V, 32, V) calls that feed `t`). Every compute_two_block
//   and compute_three_block call writes the DRBG key K, where a zeroed output
//   is never parsed and would silently key the next HMAC with K = 0. So the
//   property this module pins is the one that actually holds everywhere:
//   a deterministic zero instead of stack residue. See docs/SECRET_LIFECYCLE.md.
// ============================================================================

#include <cstdio>
#include <cstddef>
#include <algorithm>
#include <string>

#include "audit_check.hpp"

static int g_pass = 0, g_fail = 0;

namespace {

// Returns the body text of `signature` up to the first `stop` marker after it,
// or an empty string when the signature is absent. Used to scope each guard
// assertion to its own function so a memset in a neighbour cannot satisfy it.
std::string function_body(const std::string& src,
                          const std::string& signature,
                          const std::string& stop) {
    std::size_t const begin = src.find(signature);
    if (begin == std::string::npos) return {};
    std::size_t const end = src.find(stop, begin);
    if (end == std::string::npos) return src.substr(begin);
    return src.substr(begin, end - begin);
}

// A guard is fail-closed when the same statement that returns also zeroes the
// output. Matching the whole statement -- not just "memset appears somewhere in
// this function" -- is what makes the check exact.
bool guard_zeroes_output(const std::string& body, const std::string& condition) {
    std::size_t const at = body.find(condition);
    if (at == std::string::npos) return false;
    std::size_t const eol = body.find('\n', at);
    std::string const stmt =
        body.substr(at, eol == std::string::npos ? std::string::npos : eol - at);
    return stmt.find("std::memset(out, 0, 32)") != std::string::npos &&
           stmt.find("return") != std::string::npos;
}

} // namespace

int test_regression_hmac_guard_fail_closed_run() {
    g_pass = 0; g_fail = 0;
    std::printf("======================================================================\n");
    std::printf("  Regression: RFC 6979 HMAC length guards fail closed\n");
    std::printf("======================================================================\n\n");

    std::string src = audit_read_source_file("src/cpu/src/ecdsa.cpp");
    // Normalise line endings before matching. There is no .gitattributes in
    // this repository, so whether a checkout has LF or CRLF is up to the
    // runner's core.autocrlf -- and function_body()'s "\n    }\n" terminator
    // would silently stop matching under CRLF, leaving every body empty and
    // every check below failing for a reason that is not about the code.
    src.erase(std::remove(src.begin(), src.end(), '\r'), src.end());
    CHECK(!src.empty(),
          "src/cpu/src/ecdsa.cpp resolves from any CWD (audit_read_source_file)");
    if (src.empty()) {
        std::printf("\n[regression_hmac_guard_fail_closed] %d/%d checks passed\n",
                    g_pass, g_pass + g_fail);
        return 1;
    }

    struct Guard {
        const char* label;
        const char* signature;
        const char* condition;
    };
    // Each entry: the helper, and the exact guard condition it must fail closed on.
    static const Guard guards[] = {
        { "compute_short",
          "void compute_short(",
          "if (msg_len > 55)" },
        { "compute_two_block",
          "void compute_two_block(",
          "if (msg_len <= 64 || msg_len > 119)" },
        { "compute_three_block",
          "void compute_three_block(",
          "if (msg_len < 128 || msg_len > 183)" },
    };

    for (auto const& g : guards) {
        std::string const body = function_body(src, g.signature, "\n    }\n");

        std::string msg = std::string("HMAC_Ctx::") + g.label + " is present in ecdsa.cpp";
        CHECK(!body.empty(), msg);
        if (body.empty()) continue;

        msg = std::string("HMAC_Ctx::") + g.label + " still carries its length guard \"" +
              g.condition + "\" (the size_t-wrap precondition)";
        CHECK(body.find(g.condition) != std::string::npos, msg);

        msg = std::string("HMAC_Ctx::") + g.label +
              " zeroes out[32] in the same statement it returns from -- a bare "
              "\"return;\" would leave the caller's stack bytes readable as an HMAC";
        CHECK(guard_zeroes_output(body, g.condition), msg);
    }

    // The RFC 6979 step-c midstate must be COMPUTED from the all-zero key, not
    // transcribed from a constant table: that is what keeps the process-lifetime
    // static bit-identical to the per-call computation it replaces, and what
    // makes "nothing secret is held in static storage" checkable rather than
    // asserted. See docs/SECRET_LIFECYCLE.md.
    std::string const midstate =
        function_body(src, "void init_zero_key32(", "\n    }\n");
    CHECK(!midstate.empty(), "HMAC_Ctx::init_zero_key32 is present in ecdsa.cpp");
    if (!midstate.empty()) {
        CHECK(midstate.find("static const HMAC_Ctx") != std::string::npos,
              "init_zero_key32 caches the K0 midstate in a function-local static");
        CHECK(midstate.find("c.init_key32(ZERO_KEY32)") != std::string::npos,
              "the cached midstate is produced by init_key32(ZERO_KEY32) itself, "
              "not transcribed from a table");
    }

    std::printf("\n[regression_hmac_guard_fail_closed] %d/%d checks passed\n",
                g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_hmac_guard_fail_closed_run(); }
#endif
