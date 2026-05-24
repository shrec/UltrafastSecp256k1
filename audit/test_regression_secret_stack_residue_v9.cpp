// ============================================================================
// test_regression_secret_stack_residue_v9.cpp
// ============================================================================
// Regression coverage bundle for v9 RT-006 / RT-007 / RT-014 / RT-015 / TASK-022:
//
//   RT-006 (P2): schnorr_sign(const Scalar&) + schnorr_sign_verified(const
//                Scalar&) convenience overloads in src/cpu/src/schnorr.cpp
//                must erase the local SchnorrKeypair's secret scalar `kp.d`
//                after the sub-call returns. Without this, the negated
//                signing scalar lingers in the stack frame.
//
//   RT-007 (P3): bip32::derive_child in src/cpu/src/bip32.cpp uses the
//                constant-time predicate `child_scalar.is_zero_ct()` on the
//                secret-derived child scalar (replacing the data-dependent
//                `is_zero()`). Probabilistic side-channel hardening.
//
//   RT-014 (P3): FROST derive_scalar / derive_scalar_pair in
//                src/cpu/src/frost.cpp must securely erase the local SHA256
//                state `h`, the per-tag hash `tag_hash`, and the finalized
//                `hash` array before returning. These all incorporate the
//                seed (secret material) into a stack-resident state.
//
//   RT-015 (P3): ecdsa_adaptor_sign in src/cpu/src/adaptor.cpp must erase
//                `k`, `binding`, and `R_x_bytes` before the degenerate-r
//                early return. The success-path erase block at the end is
//                unreachable when the early return fires.
//
// Verification model: source scans assert the presence of the required
// secure_erase / is_zero_ct calls in the named functions; one functional
// round-trip per affected entry confirms the fix did not regress correctness.
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>
#include <fstream>
#include <string>

#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/bip32.hpp"
#include "secp256k1/ct/sign.hpp"

#include "audit_check.hpp"

using namespace secp256k1;
using secp256k1::fast::Scalar;
using secp256k1::fast::Point;

static int g_pass = 0, g_fail = 0;

static std::string read_source_file(const char* rel_path) {
    const char* prefixes[] = {
        "", "../", "../../",
        "src/cpu/src/", "../src/cpu/src/",
        nullptr
    };
    for (int i = 0; prefixes[i]; ++i) {
        std::string path = std::string(prefixes[i]) + rel_path;
        std::ifstream f(path);
        if (f.is_open()) {
            return {std::istreambuf_iterator<char>(f),
                    std::istreambuf_iterator<char>()};
        }
    }
    return {};
}

// Helper: extract the body of a named C++ function from a source string.
// Returns the substring between the function's opening `{` and matching `}`.
static std::string extract_function_body(const std::string& src, const std::string& name) {
    size_t i = 0;
    while ((i = src.find(name, i)) != std::string::npos) {
        // Walk forward to find the next '{' on the same/following lines.
        size_t open = src.find('{', i);
        if (open == std::string::npos) return {};
        // Match braces.
        int depth = 1;
        size_t j = open + 1;
        while (j < src.size() && depth > 0) {
            char c = src[j];
            if (c == '{') ++depth;
            else if (c == '}') --depth;
            ++j;
        }
        if (depth == 0) {
            return src.substr(open, j - open);
        }
        i = open + 1;
    }
    return {};
}

// ── RT-006: schnorr_sign / schnorr_sign_verified Scalar overloads erase kp.d ──

static void test_schnorr_raw_overload_erases_kp() {
    printf("[1] RT-006 / TASK-022: schnorr.cpp — raw-key overloads erase kp.d\n");

    std::string src = read_source_file("src/cpu/src/schnorr.cpp");
    if (src.empty()) {
        src = read_source_file("schnorr.cpp");
    }
    if (src.empty()) {
        printf("  [SKIP] schnorr.cpp not found — source scan skipped\n");
        return;
    }

    // Both overloads must contain secure_erase(&kp.d, ...) after the sub-call.
    // The full call chain: schnorr_sign(Scalar) -> schnorr_keypair_create ->
    // schnorr_sign(kp, ...) -> secure_erase(&kp.d).
    bool has_erase =
        (src.find("secure_erase(&kp.d") != std::string::npos);
    CHECK(has_erase, "schnorr.cpp: secure_erase(&kp.d, ...) present (RT-006)");

    // The fix introduced the call twice (one per overload). Count to be sure
    // both overloads carry the discipline.
    size_t count = 0;
    size_t pos = 0;
    while ((pos = src.find("secure_erase(&kp.d", pos)) != std::string::npos) {
        ++count;
        pos += 18;
    }
    CHECK(count >= 2,
          "schnorr.cpp: secure_erase(&kp.d, ...) in BOTH raw-key overloads (RT-006)");
}

static void test_schnorr_raw_overload_functional() {
    printf("[1b] RT-006 / TASK-022: schnorr_sign(Scalar) round-trip works\n");

    std::array<uint8_t, 32> sk_bytes{};
    sk_bytes[31] = 0x42;
    std::array<uint8_t, 32> msg{};
    msg[0] = 0xDE; msg[31] = 0xAD;
    std::array<uint8_t, 32> aux{};

    Scalar sk;
    bool ok = Scalar::parse_bytes_strict_nonzero(sk_bytes.data(), sk);
    CHECK(ok, "RT-006: sk parse");
    if (!ok) return;

    Point pub = secp256k1::ct::generator_mul(sk);
    auto pub_x = pub.x().to_bytes();

    auto sig = secp256k1::schnorr_sign(sk, msg, aux);
    bool v = secp256k1::schnorr_verify(pub_x.data(), msg.data(), sig);
    CHECK(v, "RT-006: schnorr_sign(Scalar) round-trip verifies");

    auto sig2 = secp256k1::schnorr_sign_verified(sk, msg, aux);
    bool v2 = secp256k1::schnorr_verify(pub_x.data(), msg.data(), sig2);
    CHECK(v2, "RT-006: schnorr_sign_verified(Scalar) round-trip verifies");
}

// ── RT-007: bip32::derive_child uses is_zero_ct on the secret child scalar ────

static void test_bip32_derive_child_uses_is_zero_ct() {
    printf("[2] RT-007 / TASK-022: bip32.cpp — derive_child uses is_zero_ct\n");

    std::string src = read_source_file("src/cpu/src/bip32.cpp");
    if (src.empty()) {
        src = read_source_file("bip32.cpp");
    }
    if (src.empty()) {
        printf("  [SKIP] bip32.cpp not found — source scan skipped\n");
        return;
    }

    // The patched line: `if (child_scalar.is_zero_ct()) {`
    bool has_ct = (src.find("child_scalar.is_zero_ct()") != std::string::npos);
    CHECK(has_ct,
          "bip32.cpp: child_scalar.is_zero_ct() present in derive_child (RT-007)");

    // The non-CT call on the child scalar must be gone.
    bool has_non_ct =
        (src.find("child_scalar.is_zero()") != std::string::npos);
    CHECK(!has_non_ct,
          "bip32.cpp: no bare child_scalar.is_zero() remains (RT-007)");
}

// ── RT-014: FROST derive_scalar erases hash/tag_hash/h ────────────────────────

static void test_frost_derive_scalar_erases_sha_state() {
    printf("[3] RT-014 / TASK-022: frost.cpp — derive_scalar erases SHA state\n");

    std::string src = read_source_file("src/cpu/src/frost.cpp");
    if (src.empty()) {
        src = read_source_file("frost.cpp");
    }
    if (src.empty()) {
        printf("  [SKIP] frost.cpp not found — source scan skipped\n");
        return;
    }

    // Look for the erasure trio at the end of derive_scalar bodies.
    bool has_hash_erase     = (src.find("secure_erase(hash.data(), hash.size())") != std::string::npos);
    bool has_tag_hash_erase = (src.find("secure_erase(tag_hash.data(), tag_hash.size())") != std::string::npos);
    bool has_h_erase        = (src.find("secure_erase(&h, sizeof(h))") != std::string::npos);

    CHECK(has_hash_erase,
          "frost.cpp: secure_erase(hash) present in derive_scalar (RT-014)");
    CHECK(has_tag_hash_erase,
          "frost.cpp: secure_erase(tag_hash) present in derive_scalar (RT-014)");
    CHECK(has_h_erase,
          "frost.cpp: secure_erase(&h) present in derive_scalar (RT-014)");

    // Both helpers (derive_scalar and derive_scalar_pair) should be covered;
    // expect at least 2 occurrences of each erase pattern.
    auto count_substr = [](const std::string& s, const std::string& needle) {
        size_t n = 0, p = 0;
        while ((p = s.find(needle, p)) != std::string::npos) {
            ++n;
            p += needle.size();
        }
        return n;
    };
    CHECK(count_substr(src, "secure_erase(&h, sizeof(h))") >= 2,
          "frost.cpp: SHA state erased in BOTH derive_scalar and derive_scalar_pair (RT-014)");
}

// ── RT-015: ecdsa_adaptor_sign degenerate-r early return erases secrets ───────

static void test_ecdsa_adaptor_degenerate_r_erases() {
    printf("[4] RT-015 / TASK-022: adaptor.cpp — degenerate-r early return erases secrets\n");

    std::string src = read_source_file("src/cpu/src/adaptor.cpp");
    if (src.empty()) {
        src = read_source_file("adaptor.cpp");
    }
    if (src.empty()) {
        printf("  [SKIP] adaptor.cpp not found — source scan skipped\n");
        return;
    }

    // Restrict the scan to ecdsa_adaptor_sign's body.
    std::string body = extract_function_body(src, "ecdsa_adaptor_sign");
    if (body.empty()) {
        printf("  [SKIP] ecdsa_adaptor_sign body not located — source scan skipped\n");
        return;
    }

    // Find the degenerate-r block: `if (r.is_zero()) {` … `return ECDSAAdaptorSig{...}`.
    size_t guard = body.find("if (r.is_zero())");
    CHECK(guard != std::string::npos,
          "adaptor.cpp: ecdsa_adaptor_sign retains r.is_zero() guard (precondition for RT-015 fix)");
    if (guard == std::string::npos) return;

    size_t return_pos = body.find("return ECDSAAdaptorSig", guard);
    CHECK(return_pos != std::string::npos,
          "adaptor.cpp: ecdsa_adaptor_sign degenerate path returns ECDSAAdaptorSig sentinel");
    if (return_pos == std::string::npos) return;

    std::string guard_block = body.substr(guard, return_pos - guard);

    // The guard block must erase k, binding, and R_x_bytes before returning.
    bool erases_k       = (guard_block.find("secure_erase(&k") != std::string::npos);
    bool erases_binding = (guard_block.find("secure_erase(&binding") != std::string::npos);
    bool erases_R_x     = (guard_block.find("secure_erase(R_x_bytes") != std::string::npos);

    CHECK(erases_k,
          "adaptor.cpp: ecdsa_adaptor_sign degenerate-r block erases &k (RT-015)");
    CHECK(erases_binding,
          "adaptor.cpp: ecdsa_adaptor_sign degenerate-r block erases &binding (RT-015)");
    CHECK(erases_R_x,
          "adaptor.cpp: ecdsa_adaptor_sign degenerate-r block erases R_x_bytes (RT-015)");
}

int test_regression_secret_stack_residue_v9_run() {
    g_pass = 0; g_fail = 0;
    printf("======================================================================\n");
    printf("  Regression: secret stack residue bundle (v9 RT-006/-007/-014/-015)\n");
    printf("  Bundle: TASK-022 — schnorr raw-key erase, BIP32 is_zero_ct,\n");
    printf("          FROST SHA state erase, adaptor degenerate-r erase.\n");
    printf("======================================================================\n\n");

    test_schnorr_raw_overload_erases_kp();
    printf("\n");
    test_schnorr_raw_overload_functional();
    printf("\n");
    test_bip32_derive_child_uses_is_zero_ct();
    printf("\n");
    test_frost_derive_scalar_erases_sha_state();
    printf("\n");
    test_ecdsa_adaptor_degenerate_r_erases();
    printf("\n");

    printf("[regression_secret_stack_residue_v9] %d/%d checks passed\n",
           g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_secret_stack_residue_v9_run(); }
#endif
