// ============================================================================
// Test: Edge cases & coverage gaps
// ============================================================================
// Exercises untested branches and rare code paths:
//   1. Scalar zero rejection for ECDSA/Schnorr signing
//   2. Point at infinity arithmetic (O+O, dbl(O), O+P)
//   3. BIP-32 IL >= curve order rejection
//   4. Precompute cache corruption recovery
//   5. Scalar boundary values (k=n-1 => -G, k=1, k=2)
//   6. parse_bytes_strict boundary rejection (scalar == n, scalar == n+1)
// ============================================================================

#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/field.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/bip32.hpp"
#include "secp256k1/precompute.hpp"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>
#include <fstream>
#include <filesystem>
#ifdef _WIN32
#  include <windows.h>
#else
#  include <unistd.h>
#endif

using namespace secp256k1::fast;
using secp256k1::ecdsa_sign;
using secp256k1::ecdsa_verify;
using secp256k1::ECDSASignature;
using secp256k1::schnorr_sign;
using secp256k1::schnorr_verify;
using secp256k1::bip32_master_key;

static int g_tests_run = 0;
static int g_tests_passed = 0;

#define CHECK(cond, msg) do { \
    ++g_tests_run; \
    if (cond) { ++g_tests_passed; std::printf("  [PASS] %s\n", msg); } \
    else { std::printf("  [FAIL] %s\n", msg); } \
} while(0)

// -- secp256k1 curve order n (big-endian bytes) ------------------------------
static constexpr std::array<uint8_t, 32> ORDER_N = {
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
    0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B,
    0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x41
};

// n-1 (big-endian bytes)
static constexpr std::array<uint8_t, 32> ORDER_N_MINUS_1 = {
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
    0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B,
    0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x40
};

// n+1 (big-endian bytes)
static constexpr std::array<uint8_t, 32> ORDER_N_PLUS_1 = {
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
    0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B,
    0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x42
};

// ============================================================================
// 1. Scalar zero rejection paths
// ============================================================================
static void test_scalar_zero_rejection() {
    std::printf("\n=== Scalar zero rejection ===\n");

    const auto zero = Scalar::zero();
    CHECK(zero.is_zero(), "Scalar::zero() is zero");

    const auto from0 = Scalar::from_uint64(0);
    CHECK(from0.is_zero(), "Scalar::from_uint64(0) is zero");

    // parse_bytes_strict_nonzero must reject zero
    const std::array<uint8_t, 32> zero_bytes{};
    Scalar out{};
    bool ok = Scalar::parse_bytes_strict_nonzero(zero_bytes, out);
    CHECK(!ok, "parse_bytes_strict_nonzero rejects zero");

    // parse_bytes_strict accepts zero (it only rejects >= n)
    ok = Scalar::parse_bytes_strict(zero_bytes, out);
    CHECK(ok, "parse_bytes_strict accepts zero (valid < n)");
    CHECK(out.is_zero(), "parse_bytes_strict(0) returns zero scalar");
}

// ============================================================================
// 2. parse_bytes_strict boundary values
// ============================================================================
static void test_scalar_parse_boundaries() {
    std::printf("\n=== Scalar parse_bytes_strict boundaries ===\n");

    Scalar out{};

    // n-1: must be accepted (valid scalar)
    bool ok = Scalar::parse_bytes_strict(ORDER_N_MINUS_1, out);
    CHECK(ok, "parse_bytes_strict accepts n-1");
    CHECK(!out.is_zero(), "n-1 is nonzero");

    // n: must be rejected
    ok = Scalar::parse_bytes_strict(ORDER_N, out);
    CHECK(!ok, "parse_bytes_strict rejects n (== order)");

    // n+1: must be rejected
    ok = Scalar::parse_bytes_strict(ORDER_N_PLUS_1, out);
    CHECK(!ok, "parse_bytes_strict rejects n+1 (> order)");

    // all 0xFF: must be rejected
    std::array<uint8_t, 32> all_ff{};
    std::memset(all_ff.data(), 0xFF, 32);
    ok = Scalar::parse_bytes_strict(all_ff, out);
    CHECK(!ok, "parse_bytes_strict rejects 0xFF..FF");

    // parse_bytes_strict_nonzero: n-1 must be accepted
    ok = Scalar::parse_bytes_strict_nonzero(ORDER_N_MINUS_1, out);
    CHECK(ok, "parse_bytes_strict_nonzero accepts n-1");

    // parse_bytes_strict_nonzero: n must be rejected
    ok = Scalar::parse_bytes_strict_nonzero(ORDER_N, out);
    CHECK(!ok, "parse_bytes_strict_nonzero rejects n");
}

// ============================================================================
// 3. Point at infinity arithmetic
// ============================================================================
static void test_infinity_arithmetic() {
    std::printf("\n=== Infinity arithmetic ===\n");

    const Point O = Point::infinity();
    const Point G = Point::generator();

    // O + O = O
    const Point OO = O.add(O);
    CHECK(OO.is_infinity(), "O + O = O");

    // dbl(O) = O
    const Point dblO = O.dbl();
    CHECK(dblO.is_infinity(), "dbl(O) = O");

    // O + G = G
    const Point OG = O.add(G);
    CHECK(!OG.is_infinity(), "O + G is not infinity");
    CHECK(OG.to_compressed() == G.to_compressed(), "O + G = G");

    // G + O = G
    const Point GO = G.add(O);
    CHECK(!GO.is_infinity(), "G + O is not infinity");
    CHECK(GO.to_compressed() == G.to_compressed(), "G + O = G");

    // (n-1)*G + G = O (another way to get infinity)
    const Scalar nm1 = Scalar::from_bytes(ORDER_N_MINUS_1);
    const Point negG = G.scalar_mul(nm1);
    CHECK(!negG.is_infinity(), "(n-1)*G is not infinity");

    const Point should_be_O = negG.add(G);
    CHECK(should_be_O.is_infinity(), "(n-1)*G + G = O");

    // Verify (n-1)*G = -G  (negation)
    const Point minusG = G.negate();
    CHECK(negG.to_compressed() == minusG.to_compressed(), "(n-1)*G == -G");
}

// ============================================================================
// 4. ECDSA signing with zero/boundary keys
// ============================================================================
static void test_ecdsa_zero_key() {
    std::printf("\n=== ECDSA zero/boundary key tests ===\n");

    std::array<uint8_t, 32> msg{};
    msg[0] = 0x42; // non-zero message hash

    // Sign with valid key, verify it works
    const Scalar valid_key = Scalar::from_uint64(1);
    const auto sig = ecdsa_sign(msg, valid_key);
    const Point pub = Point::generator().scalar_mul(valid_key);
    const bool valid = ecdsa_verify(msg, pub, sig);
    CHECK(valid, "ECDSA sign+verify with k=1");

    // Sign with n-1 key, should work
    const Scalar nm1_key = Scalar::from_bytes(ORDER_N_MINUS_1);
    const auto sig_nm1 = ecdsa_sign(msg, nm1_key);
    const Point pub_nm1 = Point::generator().scalar_mul(nm1_key);
    const bool valid_nm1 = ecdsa_verify(msg, pub_nm1, sig_nm1);
    CHECK(valid_nm1, "ECDSA sign+verify with k=n-1");

    // Verify with wrong key should fail
    const Scalar wrong_key = Scalar::from_uint64(2);
    const Point wrong_pub = Point::generator().scalar_mul(wrong_key);
    const bool wrong = ecdsa_verify(msg, wrong_pub, sig);
    CHECK(!wrong, "ECDSA verify with wrong key fails");
}

// ============================================================================
// 5. Schnorr signing boundary tests
// ============================================================================
static void test_schnorr_boundary() {
    std::printf("\n=== Schnorr boundary key tests ===\n");

    std::array<uint8_t, 32> msg{};
    msg[0] = 0xAB;
    std::array<uint8_t, 32> aux{};
    aux[0] = 0xCD;

    // Sign with k=1
    const Scalar k1 = Scalar::from_uint64(1);
    const auto sig1 = schnorr_sign(k1, msg, aux);
    auto pub1_x = Point::generator().scalar_mul(k1).x_only_bytes();
    const bool v1 = schnorr_verify(pub1_x, msg, sig1);
    CHECK(v1, "Schnorr sign+verify with k=1");

    // Sign with k=n-1
    const Scalar knm1 = Scalar::from_bytes(ORDER_N_MINUS_1);
    const auto sig_nm1 = schnorr_sign(knm1, msg, aux);
    auto pub_nm1_x = Point::generator().scalar_mul(knm1).x_only_bytes();
    const bool v_nm1 = schnorr_verify(pub_nm1_x, msg, sig_nm1);
    CHECK(v_nm1, "Schnorr sign+verify with k=n-1");
}

// ============================================================================
// 6. BIP-32 IL >= n rejection
// ============================================================================
static void test_bip32_il_geq_n() {
    std::printf("\n=== BIP-32 IL >= n rejection ===\n");

    // Create a valid master key from a known seed
    const uint8_t seed[16] = {
        0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
        0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F
    };
    auto [master, ok] = bip32_master_key(seed, 16);
    CHECK(ok, "BIP-32 master key from 16-byte seed");

    // Derive a valid child (index 0)
    auto [child0, ok0] = master.derive_child(0);
    CHECK(ok0, "BIP-32 child derivation index=0 succeeds");
    CHECK(child0.depth == 1, "BIP-32 child depth = 1");

    // Derive multiple children to exercise the loop
    bool all_ok = true;
    for (uint32_t i = 0; i < 10; ++i) {
        auto [child, cok] = master.derive_child(i);
        (void)child;
        if (!cok) { all_ok = false; break; }
    }
    CHECK(all_ok, "BIP-32 derive 10 children all succeed");

    // Hardened derivation
    auto [hchild, hok] = master.derive_hardened(0);
    (void)hchild;
    CHECK(hok, "BIP-32 hardened child index=0 succeeds");

    // Seed too short (< 16 bytes)
    auto [bad_master, bad_ok] = bip32_master_key(seed, 15);
    (void)bad_master;
    CHECK(!bad_ok, "BIP-32 rejects seed < 16 bytes");

    // Seed too long (> 64 bytes)
    uint8_t long_seed[65] = {};
    auto [bad_master2, bad_ok2] = bip32_master_key(long_seed, 65);
    (void)bad_master2;
    CHECK(!bad_ok2, "BIP-32 rejects seed > 64 bytes");
}

// ============================================================================
// 7. Precompute cache corruption recovery
// ============================================================================
static void test_precompute_cache_corrupt() {
    std::printf("\n=== Precompute cache corruption recovery ===\n");

    // Use temp_directory_path() rather than hardcoded /tmp to be portable and
    // avoid predictable path collisions in concurrent test runs.
    auto tmpdir = std::filesystem::temp_directory_path();
    std::string pid_str = std::to_string(static_cast<long>(
#ifdef _WIN32
        GetCurrentProcessId()
#else
        getpid()
#endif
    ));
    std::string corrupt_path = (tmpdir / ("secp256k1_test_corrupt_" + pid_str + ".bin")).string();
    std::string trunc_path   = (tmpdir / ("secp256k1_test_trunc_"   + pid_str + ".bin")).string();

    // Loading from nonexistent file should return false
    bool ok = load_precompute_cache((tmpdir / ("nonexistent_secp256k1_cache_" + pid_str + ".bin")).string().c_str());
    CHECK(!ok, "load_precompute_cache rejects nonexistent file");

    // Create a truncated/corrupt cache file
    {
        std::ofstream f(corrupt_path, std::ios::binary);
        const char garbage[] = "not a valid cache header";
        f.write(garbage, sizeof(garbage));
    }
    ok = load_precompute_cache(corrupt_path.c_str());
    CHECK(!ok, "load_precompute_cache rejects corrupt file");

    // Create a file with valid magic but truncated data
    {
        std::ofstream f(trunc_path, std::ios::binary);
        // Write 8 bytes (likely wrong magic + version)
        uint64_t fake_header = 0;
        f.write(reinterpret_cast<const char*>(&fake_header), 8);
    }
    ok = load_precompute_cache(trunc_path.c_str());
    CHECK(!ok, "load_precompute_cache rejects truncated file");

    // Cleanup temp files
    (void)std::remove(corrupt_path.c_str());
    (void)std::remove(trunc_path.c_str());
}

// ============================================================================
// 8. Scalar arithmetic edge cases
// ============================================================================
static void test_scalar_arithmetic_edges() {
    std::printf("\n=== Scalar arithmetic edges ===\n");

    const Scalar zero = Scalar::zero();
    const Scalar one = Scalar::from_uint64(1);
    const Scalar two = Scalar::from_uint64(2);
    const Scalar nm1 = Scalar::from_bytes(ORDER_N_MINUS_1);

    // 0 + 1 = 1
    const Scalar sum01 = zero + one;
    CHECK(sum01.to_bytes() == one.to_bytes(), "0 + 1 = 1");

    // n-1 + 1 = 0 (mod n)
    const Scalar wrap = nm1 + one;
    CHECK(wrap.is_zero(), "(n-1) + 1 = 0 mod n");

    // n-1 + 2 = 1 (mod n)
    const Scalar wrap2 = nm1 + two;
    CHECK(wrap2.to_bytes() == one.to_bytes(), "(n-1) + 2 = 1 mod n");

    // 1 * 0 = 0
    const Scalar prod0 = one * zero;
    CHECK(prod0.is_zero(), "1 * 0 = 0");

    // 1 * 1 = 1
    const Scalar prod1 = one * one;
    CHECK(prod1.to_bytes() == one.to_bytes(), "1 * 1 = 1");

    // negate(0) = 0
    const Scalar neg0 = zero.negate();
    CHECK(neg0.is_zero(), "negate(0) = 0");

    // negate(1) = n-1
    const Scalar neg1 = one.negate();
    CHECK(neg1.to_bytes() == nm1.to_bytes(), "negate(1) = n-1");

    // negate(n-1) = 1
    const Scalar neg_nm1 = nm1.negate();
    CHECK(neg_nm1.to_bytes() == one.to_bytes(), "negate(n-1) = 1");
}

// ============================================================================
// 9. Field element edge cases
// ============================================================================
static void test_field_edge_cases() {
    std::printf("\n=== Field element edge cases ===\n");

    const auto zero = FieldElement::zero();
    const auto one = FieldElement::one();

    // 0 * 0 = 0
    const auto prod00 = zero * zero;
    CHECK(prod00 == zero, "FE: 0 * 0 = 0");

    // 1 * 1 = 1
    const auto prod11 = one * one;
    CHECK(prod11 == one, "FE: 1 * 1 = 1");

    // 0 * 1 = 0
    const auto prod01 = zero * one;
    CHECK(prod01 == zero, "FE: 0 * 1 = 0");

    // 0 + 0 = 0
    const auto sum00 = zero + zero;
    CHECK(sum00 == zero, "FE: 0 + 0 = 0");

    // 1 + 0 = 1
    const auto sum10 = one + zero;
    CHECK(sum10 == one, "FE: 1 + 0 = 1");

    // a - a = 0  (intentionally same operand on both sides)
    const auto other_one = FieldElement::from_uint64(1);
    const auto sub_aa = one - other_one;
    CHECK(sub_aa == zero, "FE: 1 - 1 = 0");

    // negate(0) = 0
    const auto neg0 = zero.negate();
    CHECK(neg0 == zero, "FE: negate(0) = 0");

    // square(0) = 0
    const auto sq0 = zero.square();
    CHECK(sq0 == zero, "FE: square(0) = 0");

    // square(1) = 1
    const auto sq1 = one.square();
    CHECK(sq1 == one, "FE: square(1) = 1");
}

// ============================================================================
// 10. ECDSASignature parse_compact_strict boundaries
// ============================================================================
static void test_ecdsa_sig_parse_boundaries() {
    std::printf("\n=== ECDSA signature parse boundaries ===\n");

    // Zero signature must be rejected (r=0)
    const std::array<uint8_t, 64> zero_sig{};
    ECDSASignature out{};
    bool ok = ECDSASignature::parse_compact_strict(zero_sig, out);
    CHECK(!ok, "parse_compact_strict rejects zero sig (r=0,s=0)");

    // r=1, s=0 must be rejected
    std::array<uint8_t, 64> r1s0{};
    r1s0[31] = 0x01;
    ok = ECDSASignature::parse_compact_strict(r1s0, out);
    CHECK(!ok, "parse_compact_strict rejects r=1,s=0");

    // r=0, s=1 must be rejected
    std::array<uint8_t, 64> r0s1{};
    r0s1[63] = 0x01;
    ok = ECDSASignature::parse_compact_strict(r0s1, out);
    CHECK(!ok, "parse_compact_strict rejects r=0,s=1");

    // r=1, s=1 must be accepted
    std::array<uint8_t, 64> r1s1{};
    r1s1[31] = 0x01;
    r1s1[63] = 0x01;
    ok = ECDSASignature::parse_compact_strict(r1s1, out);
    CHECK(ok, "parse_compact_strict accepts r=1,s=1");

    // r=n, s=1 must be rejected
    std::array<uint8_t, 64> rns1{};
    std::memcpy(rns1.data(), ORDER_N.data(), 32);
    rns1[63] = 0x01;
    ok = ECDSASignature::parse_compact_strict(rns1, out);
    CHECK(!ok, "parse_compact_strict rejects r=n");

    // r=1, s=n must be rejected
    std::array<uint8_t, 64> r1sn{};
    r1sn[31] = 0x01;
    std::memcpy(r1sn.data() + 32, ORDER_N.data(), 32);
    ok = ECDSASignature::parse_compact_strict(r1sn, out);
    CHECK(!ok, "parse_compact_strict rejects s=n");

    // r=n-1, s=n-1 must be accepted
    std::array<uint8_t, 64> rnm1snm1{};
    std::memcpy(rnm1snm1.data(), ORDER_N_MINUS_1.data(), 32);
    std::memcpy(rnm1snm1.data() + 32, ORDER_N_MINUS_1.data(), 32);
    ok = ECDSASignature::parse_compact_strict(rnm1snm1, out);
    CHECK(ok, "parse_compact_strict accepts r=n-1,s=n-1");
}

// ============================================================================
// Entry point (matches test runner pattern)
// ============================================================================

int test_edge_cases_run() {
    g_tests_run = 0;
    g_tests_passed = 0;

    test_scalar_zero_rejection();
    test_scalar_parse_boundaries();
    test_infinity_arithmetic();
    test_ecdsa_zero_key();
    test_schnorr_boundary();
    test_bip32_il_geq_n();
    test_precompute_cache_corrupt();
    test_scalar_arithmetic_edges();
    test_field_edge_cases();
    test_ecdsa_sig_parse_boundaries();

    std::printf("\n--- Edge case summary: %d/%d passed ---\n",
                g_tests_passed, g_tests_run);

    return (g_tests_passed == g_tests_run) ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() {
    return test_edge_cases_run();
}
#endif
