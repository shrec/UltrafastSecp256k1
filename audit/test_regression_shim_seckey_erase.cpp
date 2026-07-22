// ============================================================================
// test_regression_shim_seckey_erase.cpp
// ============================================================================
// Regression coverage for the 2026-05-28 read-only-review secret-residue and
// strict-DER fixes in the libsecp256k1 compatibility shim and BIP-32:
//
//   CT-01 (P2): secp256k1_ecdsa_sign (compat/libsecp256k1_shim/src/shim_ecdsa.cpp)
//               and secp256k1_ecdsa_sign_recoverable (shim_recovery.cpp) must
//               secure_erase the parsed private-key scalar (`k` / `privkey_scalar`)
//               on every return path. The Schnorr shim already did; these did not.
//
//   SHIM-01 (P2): secp256k1_ellswift_create (shim_ellswift.cpp) must erase the
//                 secret scalar `sk` and its byte copy `kb` on success and on the
//                 strict-parse early return (BIP-324 handshake key material).
//
//   SHIM-02 (P2): secp256k1_ellswift_xdh general path — the strict-parse return
//                 plus the three general-path error returns (sqrt-fail, two
//                 is_infinity) previously skipped erasure; only the success path
//                 erased. All must erase `sk`/`kb` now (via erase_secrets()).
//
//   CT-02 (P3): ExtendedKey::derive_child (src/cpu/src/bip32.cpp) must erase the
//               secret-derived `il_scalar` (HMAC image of the parent private key
//               on hardened paths) on every return path — previously only I/IL/
//               parent_scalar/child_scalar were erased.
//
//   RT-02 (P2): secp256k1_ecdsa_signature_parse_der (shim_ecdsa.cpp) must reject
//               an inflated-SEQUENCE blob whose declared length exactly fills the
//               buffer but leaves trailing bytes *inside* the SEQUENCE after s
//               (strict-DER `if (p != end) return 0;`), matching upstream
//               secp256k1_ecdsa_sig_parse and the native C-ABI parser.
//
//   CT-04 (P2): secp256k1_keypair_xonly_tweak_add (shim_extrakeys.cpp) must
//               secure_erase the tweaked-private-key residue (sk, new_sk, new_skb)
//               on every return path — the BIP-341 key-path-spend secret. The
//               CT-01 sweep (24b29021) did not cover this function.
//
//   RT-05 (P2): shim_seckey.cpp negate/tweak_add/tweak_mul/verify must include
//               detail/secure_erase.hpp and erase the parsed key k (and the new
//               key out / tweak t / result) on every return path. The file had
//               zero secure_erase calls before this fix.
//
// Verification model (same as test_regression_secret_stack_residue_v9.cpp):
// source scans assert the presence of the required secure_erase / strict-DER
// constructs in the named functions; functional round-trips confirm the fixes
// did not regress correctness of the underlying CPU primitives the shim wraps.
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>
#include <fstream>
#include <string>

#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/ellswift.hpp"
#include "secp256k1/bip32.hpp"
#include "secp256k1/ct/sign.hpp"
#include "secp256k1/ct/point.hpp"

#include "audit_check.hpp"

using namespace secp256k1;
using secp256k1::fast::Scalar;
using secp256k1::fast::Point;

static int g_pass = 0, g_fail = 0;

// Read a repo-relative source file. Not-found is a HARD FAIL (the source
// always exists in-tree) — a silent skip would make this regression guard a
// false-green.
//
// Repair (issue #335 acceptance repair, round 5): the previous bounded
// CWD-relative-only prefix list never resolved when unified_audit_runner was
// invoked from a CWD unrelated to the repo (e.g. /tmp) — this module already
// hard-failed rather than silently passing, but its real check count still
// diverged between "run from the repo root" (49/49) and "run from an
// unrelated CWD" (13/20), which the same CWD-independence finding covers.
// Route through the shared, UFSECP_SOURCE_ROOT-aware audit_read_source_file()
// (audit_check.hpp), keeping the bounded CWD-relative walk-up only as its
// own internal fallback.
static std::string read_repo_file(const char* rel_path) {
    return audit_read_source_file(rel_path);
}

// Extract the brace-matched body of the first function whose name (including the
// trailing '(') matches `name`. Returns "" if not found.
static std::string extract_function_body(const std::string& src, const std::string& name) {
    size_t i = 0;
    while ((i = src.find(name, i)) != std::string::npos) {
        size_t open = src.find('{', i);
        if (open == std::string::npos) return {};
        int depth = 1;
        size_t j = open + 1;
        while (j < src.size() && depth > 0) {
            char c = src[j];
            if (c == '{') ++depth;
            else if (c == '}') --depth;
            ++j;
        }
        if (depth == 0) return src.substr(open, j - open);
        i = open + 1;
    }
    return {};
}

static size_t count_substr(const std::string& s, const std::string& needle) {
    size_t n = 0, p = 0;
    while ((p = s.find(needle, p)) != std::string::npos) { ++n; p += needle.size(); }
    return n;
}

// ── CT-01: shim ECDSA sign + sign_recoverable erase the parsed key scalar ─────

static void test_shim_ecdsa_sign_erases_key() {
    printf("[1] CT-01: shim_ecdsa.cpp / shim_recovery.cpp erase parsed key scalar\n");

    std::string ecdsa = read_repo_file("compat/libsecp256k1_shim/src/shim_ecdsa.cpp");
    CHECK(!ecdsa.empty(), "CT-01: shim_ecdsa.cpp readable");
    if (!ecdsa.empty()) {
        std::string body = extract_function_body(ecdsa, "secp256k1_ecdsa_sign(");
        CHECK(!body.empty(), "CT-01: secp256k1_ecdsa_sign body located");
        CHECK(body.find("secure_erase(&k") != std::string::npos,
              "CT-01: secp256k1_ecdsa_sign erases parsed scalar k");
    }

    std::string rec = read_repo_file("compat/libsecp256k1_shim/src/shim_recovery.cpp");
    CHECK(!rec.empty(), "CT-01: shim_recovery.cpp readable");
    if (!rec.empty()) {
        std::string body = extract_function_body(rec, "secp256k1_ecdsa_sign_recoverable(");
        CHECK(!body.empty(), "CT-01: secp256k1_ecdsa_sign_recoverable body located");
        // One erase per branch (compat / hedged / plain) — expect >= 2 in the body.
        CHECK(count_substr(body, "secure_erase(&privkey_scalar") >= 2,
              "CT-01: sign_recoverable erases privkey_scalar on its branches");
    }
}

// ── RT-02: strict-DER trailing-bytes-inside-SEQUENCE rejection ────────────────

static void test_shim_der_strict_consumption() {
    printf("[2] RT-02: shim_ecdsa.cpp parse_der requires exact SEQUENCE consumption\n");

    std::string ecdsa = read_repo_file("compat/libsecp256k1_shim/src/shim_ecdsa.cpp");
    CHECK(!ecdsa.empty(), "RT-02: shim_ecdsa.cpp readable");
    if (ecdsa.empty()) return;
    std::string body = extract_function_body(ecdsa, "secp256k1_ecdsa_signature_parse_der(");
    CHECK(!body.empty(), "RT-02: parse_der body located");
    // The strict-DER `p != end` rejection now also zeroes the output sig on failure
    // (PASS3-SHIM-001), so the source reads `if (p != end) { std::memset(...); return 0; }`.
    CHECK(body.find("if (p != end) {") != std::string::npos,
          "RT-02: parse_der rejects trailing bytes inside SEQUENCE (p != end)");
}

// ── SHIM-01 / SHIM-02: ellswift create + xdh erase sk/kb on all returns ───────

static void test_shim_ellswift_erases_secrets() {
    printf("[3] SHIM-01/02: shim_ellswift.cpp erases sk/kb on all return paths\n");

    std::string ell = read_repo_file("compat/libsecp256k1_shim/src/shim_ellswift.cpp");
    CHECK(!ell.empty(), "SHIM-01/02: shim_ellswift.cpp readable");
    if (ell.empty()) return;

    std::string create = extract_function_body(ell, "secp256k1_ellswift_create(");
    CHECK(!create.empty(), "SHIM-01: ellswift_create body located");
    // Success + early-return: expect erase of sk and kb at least twice each.
    CHECK(count_substr(create, "secure_erase(&sk") >= 2,
          "SHIM-01: ellswift_create erases sk on success and parse-fail");
    CHECK(count_substr(create, "secure_erase(kb.data()") >= 2,
          "SHIM-01: ellswift_create erases kb on success and parse-fail");

    std::string xdh = extract_function_body(ell, "secp256k1_ellswift_xdh(");
    CHECK(!xdh.empty(), "SHIM-02: ellswift_xdh body located");
    // General-path error returns route through erase_secrets(); the three error
    // returns (sqrt-fail, two is_infinity) must each call it.
    CHECK(xdh.find("erase_secrets") != std::string::npos,
          "SHIM-02: ellswift_xdh general path has an erase_secrets() scope helper");
    CHECK(count_substr(xdh, "erase_secrets(); return 0;") >= 3,
          "SHIM-02: all three general-path error returns erase secrets");
}

// ── CT-02: bip32 derive_child erases il_scalar ────────────────────────────────

static void test_bip32_derive_child_erases_il_scalar() {
    printf("[4] CT-02: bip32.cpp derive_child erases il_scalar on every return\n");

    std::string bip32 = read_repo_file("src/cpu/src/bip32.cpp");
    CHECK(!bip32.empty(), "CT-02: bip32.cpp readable");
    if (bip32.empty()) return;
    std::string body = extract_function_body(bip32, "::derive_child(");
    CHECK(!body.empty(), "CT-02: ExtendedKey::derive_child body located");
    // Every return path (parse-fail, is_zero_ct, depth-wrap, parent-parse-fail,
    // child-zero, infinity, success) erases il_scalar — expect >= 6 occurrences.
    CHECK(count_substr(body, "secure_erase(&il_scalar") >= 6,
          "CT-02: il_scalar erased on all derive_child return paths");
}

// ── CT-04: keypair_xonly_tweak_add erases sk/new_sk/new_skb ───────────────────

static void test_shim_keypair_xonly_tweak_add_erases() {
    printf("[8] CT-04: shim_extrakeys.cpp keypair_xonly_tweak_add erases tweaked secret\n");

    std::string ek = read_repo_file("compat/libsecp256k1_shim/src/shim_extrakeys.cpp");
    CHECK(!ek.empty(), "CT-04: shim_extrakeys.cpp readable");
    if (ek.empty()) return;
    std::string body = extract_function_body(ek, "secp256k1_keypair_xonly_tweak_add(");
    CHECK(!body.empty(), "CT-04: keypair_xonly_tweak_add body located");
    // Secrets: parsed key `sk` (erased on parse-fail, tweak-fail, zero, infinity,
    // success = >=3), the tweaked key `new_sk` (zero/infinity/success = >=2), and
    // its serialization `new_skb` (success = >=1).
    CHECK(count_substr(body, "secure_erase(&sk") >= 3,
          "CT-04: keypair_xonly_tweak_add erases sk on multiple return paths");
    CHECK(count_substr(body, "secure_erase(&new_sk") >= 2,
          "CT-04: keypair_xonly_tweak_add erases tweaked new_sk");
    CHECK(body.find("secure_erase(new_skb.data()") != std::string::npos,
          "CT-04: keypair_xonly_tweak_add erases serialized new_skb");

    // keypair_create must erase BOTH the serialized bytes and the Scalar k itself.
    std::string kc = extract_function_body(ek, "secp256k1_keypair_create(");
    CHECK(!kc.empty(), "CT-04: keypair_create body located");
    if (!kc.empty())
        CHECK(kc.find("secure_erase(&k") != std::string::npos,
              "CT-04: keypair_create erases the Scalar k (normalized private key)");
}

// ── RT-05: shim_seckey.cpp negate/tweak_add/tweak_mul/verify erase secrets ────

static void test_shim_seckey_funcs_erase() {
    printf("[9] RT-05: shim_seckey.cpp negate/tweak_add/tweak_mul/verify erase secrets\n");

    std::string sk = read_repo_file("compat/libsecp256k1_shim/src/shim_seckey.cpp");
    CHECK(!sk.empty(), "RT-05: shim_seckey.cpp readable");
    if (sk.empty()) return;
    CHECK(sk.find("secure_erase.hpp") != std::string::npos,
          "RT-05: shim_seckey.cpp includes detail/secure_erase.hpp");

    struct { const char* sig; const char* label; } fns[] = {
        { "secp256k1_ec_seckey_verify(",    "RT-05: seckey_verify erases parsed k" },
        { "secp256k1_ec_seckey_negate(",    "RT-05: seckey_negate erases k/neg/out" },
        { "secp256k1_ec_seckey_tweak_add(", "RT-05: seckey_tweak_add erases k/t/result/out" },
        { "secp256k1_ec_seckey_tweak_mul(", "RT-05: seckey_tweak_mul erases k/t/result/out" },
    };
    for (auto& f : fns) {
        std::string body = extract_function_body(sk, f.sig);
        CHECK(!body.empty(), f.label);
        if (!body.empty())
            CHECK(body.find("secure_erase(&k") != std::string::npos, f.label);
    }
}

// ── Functional: fixes did not regress the underlying CPU primitives ───────────

static void test_functional_ecdsa_roundtrip() {
    printf("[5] Functional: CT ECDSA sign/verify round-trip still works\n");
    std::array<uint8_t, 32> sk_bytes{}; sk_bytes[31] = 0x42;
    Scalar sk;
    bool ok = Scalar::parse_bytes_strict_nonzero(sk_bytes.data(), sk);
    CHECK(ok, "functional: sk parse");
    if (!ok) return;
    Point pub = secp256k1::ct::generator_mul(sk);
    std::array<uint8_t, 32> msg{}; msg[0] = 0xDE; msg[31] = 0xAD;
    auto sig = secp256k1::ct::ecdsa_sign(msg, sk);
    bool v = secp256k1::ecdsa_verify(msg, pub, sig);
    CHECK(v, "functional: ct::ecdsa_sign → ecdsa_verify round-trip");
}

static void test_functional_ellswift_xdh_roundtrip() {
    printf("[6] Functional: ellswift XDH shared secret agrees (create + xdh)\n");
    std::array<uint8_t, 32> a_bytes{}; a_bytes[31] = 0x11;
    std::array<uint8_t, 32> b_bytes{}; b_bytes[31] = 0x22;
    std::array<uint8_t, 32> aux_a{}; aux_a[0] = 0xA1;
    std::array<uint8_t, 32> aux_b{}; aux_b[0] = 0xB2;
    Scalar sk_a, sk_b;
    bool oka = Scalar::parse_bytes_strict_nonzero(a_bytes.data(), sk_a);
    bool okb = Scalar::parse_bytes_strict_nonzero(b_bytes.data(), sk_b);
    CHECK(oka && okb, "functional: ellswift sk parse");
    if (!(oka && okb)) return;
    auto ell_a = secp256k1::ellswift_create(sk_a, aux_a.data());
    auto ell_b = secp256k1::ellswift_create(sk_b, aux_b.data());
    auto secret_a = secp256k1::ellswift_xdh(ell_a.data(), ell_b.data(), sk_a, true);
    auto secret_b = secp256k1::ellswift_xdh(ell_a.data(), ell_b.data(), sk_b, false);
    CHECK(std::memcmp(secret_a.data(), secret_b.data(), 32) == 0,
          "functional: ellswift XDH initiator/responder secrets agree");
}

static void test_functional_bip32_derive() {
    printf("[7] Functional: bip32 master + hardened derive_child still works\n");
    std::array<uint8_t, 32> seed{};
    for (int i = 0; i < 32; ++i) seed[i] = static_cast<uint8_t>(i + 1);
    auto [master, ok] = secp256k1::bip32_master_key(seed.data(), seed.size());
    CHECK(ok, "functional: bip32_master_key");
    if (!ok) return;
    CHECK(master.is_private, "functional: master is private");
    // Hardened child (index 0x80000000) exercises the 0x00||privkey||index HMAC path.
    auto [child, cok] = master.derive_child(0x80000000u);
    CHECK(cok, "functional: hardened derive_child succeeds");
    if (!cok) return;
    CHECK(child.is_private, "functional: child is private");
    bool nonzero = false;
    for (uint8_t b : child.key) nonzero |= (b != 0);
    CHECK(nonzero, "functional: derived child key is non-zero");
}

static void test_functional_seckey_tweak() {
    printf("[10] Functional: CT seckey tweak/negate primitives still correct\n");
    std::array<uint8_t, 32> kb{}; kb[31] = 0x07;
    std::array<uint8_t, 32> tb{}; tb[31] = 0x05;
    Scalar k, t;
    bool ok = Scalar::parse_bytes_strict_nonzero(kb.data(), k)
           && Scalar::parse_bytes_strict_nonzero(tb.data(), t);
    CHECK(ok, "functional: seckey/tweak parse");
    if (!ok) return;
    // tweak_add: result = k + t = 12 (mod n); nonzero and matches expectation.
    auto sum = secp256k1::ct::scalar_add(k, t);
    CHECK(!sum.is_zero_ct(), "functional: ct::scalar_add(k,t) is non-zero");
    auto sumb = sum.to_bytes();
    CHECK(sumb[31] == 0x0C, "functional: 7 + 5 == 12 (CT scalar_add)");
    // negate then add original == 0 (n - k + k == 0): confirms scalar_cneg path.
    auto negk = secp256k1::ct::scalar_cneg(k, ~std::uint64_t(0));
    auto zero = secp256k1::ct::scalar_add(negk, k);
    CHECK(zero.is_zero_ct(), "functional: ct::scalar_cneg(k) + k == 0");
}

int test_regression_shim_seckey_erase_run() {
    g_pass = 0; g_fail = 0;
    printf("======================================================================\n");
    printf("  Regression: shim/bip32 secret-key erasure + strict-DER (2026-05-28)\n");
    printf("  CT-01 (ecdsa sign/recoverable), SHIM-01/02 (ellswift create/xdh),\n");
    printf("  CT-02 (bip32 il_scalar), RT-02 (parse_der exact SEQUENCE).\n");
    printf("======================================================================\n\n");

    test_shim_ecdsa_sign_erases_key();          printf("\n");
    test_shim_der_strict_consumption();         printf("\n");
    test_shim_ellswift_erases_secrets();        printf("\n");
    test_bip32_derive_child_erases_il_scalar(); printf("\n");
    test_shim_keypair_xonly_tweak_add_erases(); printf("\n");
    test_shim_seckey_funcs_erase();             printf("\n");
    test_functional_ecdsa_roundtrip();          printf("\n");
    test_functional_ellswift_xdh_roundtrip();   printf("\n");
    test_functional_bip32_derive();             printf("\n");
    test_functional_seckey_tweak();             printf("\n");

    printf("[regression_shim_seckey_erase] %d/%d checks passed\n",
           g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_shim_seckey_erase_run(); }
#endif
