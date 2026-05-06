// ============================================================================
// Formal Constant-Time Verification Test (Valgrind/MSAN ct-verif approach)
// Track I, Task I5-1
// ============================================================================
//
// PURPOSE:
//   This test provides a FORMAL (deterministic) CT proof, complementing the
//   statistical dudect approach in test_ct_sidechannel.cpp.
//
// METHODOLOGY (ctgrind / ct-verif):
//   1. Mark secret data as "undefined" via SECP256K1_CLASSIFY()
//      (maps to VALGRIND_MAKE_MEM_UNDEFINED or __msan_allocated_memory)
//   2. Execute CT operations (ct::ecdsa_sign, ct::schnorr_sign, etc.)
//   3. Mark outputs as "defined" via SECP256K1_DECLASSIFY()
//   4. If Valgrind/MSAN reports "conditional jump depends on uninitialised
//      value" between classify and declassify, the code has a CT violation.
//
// This is the same technique used by:
//   - bitcoin-core/libsecp256k1 (valgrind_ctime_test.c via checkmem.h)
//   - Adam Langley's ctgrind
//   - BoringSSL's constant_time_test
//   - libsodium's constant-time validation
//
// RUNNING:
//   Without Valgrind: test passes (classify/declassify are no-ops)
//     ./test_ct_verif_formal_standalone
//
//   With Valgrind (actual formal check):
//     cmake -DSECP256K1_CT_VALGRIND=ON ...
//     valgrind --tool=memcheck --error-exitcode=42 ./test_ct_verif_formal_standalone
//
//   With MSAN:
//     cmake -DCMAKE_CXX_FLAGS="-fsanitize=memory" ...
//     ./test_ct_verif_formal_standalone
//
// Any CT violation will cause Valgrind/MSAN to report an error.
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>
#include <random>

#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/ct/ops.hpp"
#include "secp256k1/ct/field.hpp"
#include "secp256k1/ct/scalar.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/ct/sign.hpp"
#include "secp256k1/ct_utils.hpp"

using namespace secp256k1::fast;

// ============================================================================
// Helpers
// ============================================================================

static int g_pass = 0;
static int g_fail = 0;
static const char* g_section = "";

#include "audit_check.hpp"

// Deterministic PRNG for reproducibility
static std::mt19937_64 g_rng(0xC7'0E41F'DEAD);  // NOLINT(cert-msc32-c)

static void fill_random(uint8_t* buf, size_t len) {
    for (size_t i = 0; i < len; i += 8) {
        uint64_t v = g_rng();
        size_t chunk = (len - i < 8) ? (len - i) : 8;
        std::memcpy(buf + i, &v, chunk);
    }
}

static Scalar make_random_scalar() {
    std::array<uint8_t, 32> buf{};
    for (;;) {
        fill_random(buf.data(), 32);
        auto s = Scalar::from_bytes(buf);
        if (!s.is_zero()) return s;
    }
}

static FieldElement make_random_fe() {
    std::array<uint8_t, 32> buf{};
    fill_random(buf.data(), 32);
    return FieldElement::from_bytes(buf);
}

// ============================================================================
// Check whether CT verification is active (Valgrind or MSAN)
// ============================================================================
static bool ct_verif_active() {
#if defined(SECP256K1_CT_VALGRIND) && SECP256K1_CT_VALGRIND
    // VALGRIND_MAKE_MEM_DEFINED(NULL, 0) returns nonzero when running under memcheck
    return VALGRIND_MAKE_MEM_DEFINED(nullptr, 0) != 0;
#elif defined(__has_feature)
#if __has_feature(memory_sanitizer)
    return true;
#else
    return false;
#endif
#else
    return false;
#endif
}

// ============================================================================
// Test 1: CT Field Arithmetic
// ============================================================================
// Verify: field_mul, field_sqr, field_add, field_sub, field_inv
// with secret inputs produce no secret-dependent branches.
static void test_ct_field_ops() {
    g_section = "ct_verif_field";
    (void)printf("[1] CT field arithmetic (mul, sqr, add, sub, inv)\n");

    for (int i = 0; i < 10; ++i) {
        FieldElement a = make_random_fe();
        FieldElement b = make_random_fe();

        // Classify both inputs as secret
        SECP256K1_CLASSIFY(&a, sizeof(a));
        SECP256K1_CLASSIFY(&b, sizeof(b));

        // CT field operations (these must not branch on inputs)
        FieldElement sum  = a + b;
        FieldElement diff = a - b;
        FieldElement prod = a * b;
        FieldElement sq   = a.square();

        // Declassify outputs for verification
        SECP256K1_DECLASSIFY(&sum,  sizeof(sum));
        SECP256K1_DECLASSIFY(&diff, sizeof(diff));
        SECP256K1_DECLASSIFY(&prod, sizeof(prod));
        SECP256K1_DECLASSIFY(&sq,   sizeof(sq));
          SECP256K1_DECLASSIFY(&a,    sizeof(a));
          SECP256K1_DECLASSIFY(&b,    sizeof(b));

          CHECK((sum - b) == a && (diff + b) == a && prod == (a * b) && sq == a.square(),
              "field add/sub/mul/sqr preserve field identities after declassify");
    }

    // Field inverse with secret input
    {
        FieldElement a = make_random_fe();
        while (a == FieldElement::zero()) {
            a = make_random_fe();
        }
        SECP256K1_CLASSIFY(&a, sizeof(a));
        FieldElement inv = a.inverse();
        SECP256K1_DECLASSIFY(&inv, sizeof(inv));
        SECP256K1_DECLASSIFY(&a, sizeof(a));
        CHECK((a * inv) == FieldElement::one(), "field inverse preserves multiplicative identity");
    }
}

// ============================================================================
// Test 2: CT Scalar Arithmetic
// ============================================================================
static void test_ct_scalar_ops() {
    g_section = "ct_verif_scalar";
    (void)printf("[2] CT scalar arithmetic (mul, add, negate, inverse)\n");

    for (int i = 0; i < 10; ++i) {
        Scalar a = make_random_scalar();
        Scalar b = make_random_scalar();

        SECP256K1_CLASSIFY(&a, sizeof(a));
        SECP256K1_CLASSIFY(&b, sizeof(b));

        Scalar sum  = a + b;
        Scalar prod = a * b;
        Scalar neg  = a.negate();

        SECP256K1_DECLASSIFY(&sum,  sizeof(sum));
        SECP256K1_DECLASSIFY(&prod, sizeof(prod));
        SECP256K1_DECLASSIFY(&neg,  sizeof(neg));
          SECP256K1_DECLASSIFY(&a,    sizeof(a));
          SECP256K1_DECLASSIFY(&b,    sizeof(b));

          CHECK((sum - b) == a && prod == (a * b) && (neg + a).is_zero(),
              "scalar add/mul/negate preserve scalar identities after declassify");
    }

    // Scalar inverse with secret input
    {
        Scalar a = make_random_scalar();
        SECP256K1_CLASSIFY(&a, sizeof(a));
        Scalar inv = a.inverse();
        SECP256K1_DECLASSIFY(&inv, sizeof(inv));
        SECP256K1_DECLASSIFY(&a, sizeof(a));
        CHECK((a * inv) == Scalar::from_uint64(1), "scalar inverse preserves multiplicative identity");
    }
}

// ============================================================================
// Test 3: CT Primitive Operations (masks, cmov, cswap)
// ============================================================================
static void test_ct_primitives() {
    g_section = "ct_verif_primitives";
    (void)printf("[3] CT primitives (is_zero_mask, cmov, cswap, ct_select)\n");

    for (int i = 0; i < 20; ++i) {
        uint64_t secret = g_rng();
        SECP256K1_CLASSIFY(&secret, sizeof(secret));

        uint64_t mask = secp256k1::ct::is_zero_mask(secret);
        SECP256K1_DECLASSIFY(&mask, sizeof(mask));
          SECP256K1_DECLASSIFY(&secret, sizeof(secret));
          CHECK((secret == 0 && mask == ~static_cast<uint64_t>(0)) ||
              (secret != 0 && mask == 0),
              "is_zero_mask matches zero/non-zero semantics");
    }

    // cmov256
    {
        uint64_t original_dst[4] = {1, 2, 3, 4};
        uint64_t dst[4] = {1, 2, 3, 4};
        uint64_t src[4] = {5, 6, 7, 8};
        uint64_t flag = g_rng() & 1;
        SECP256K1_CLASSIFY(&flag, sizeof(flag));
        uint64_t mask_val = secp256k1::ct::bool_to_mask(flag != 0);
        secp256k1::ct::cmov256(dst, src, mask_val);
        SECP256K1_DECLASSIFY(dst, sizeof(dst));
        SECP256K1_DECLASSIFY(&flag, sizeof(flag));
        uint64_t expected[4];
        for (int j = 0; j < 4; ++j) {
            expected[j] = flag ? src[j] : original_dst[j];
        }
        CHECK(std::memcmp(dst, expected, sizeof(dst)) == 0, "cmov256 selects expected source words");
    }

    // cswap256
    {
        uint64_t a[4] = {10, 20, 30, 40};
        uint64_t b[4] = {50, 60, 70, 80};
        uint64_t orig_a[4] = {10, 20, 30, 40};
        uint64_t orig_b[4] = {50, 60, 70, 80};
        uint64_t flag = g_rng() & 1;
        SECP256K1_CLASSIFY(&flag, sizeof(flag));
        uint64_t mask_val = secp256k1::ct::bool_to_mask(flag != 0);
        secp256k1::ct::cswap256(a, b, mask_val);
        SECP256K1_DECLASSIFY(a, sizeof(a));
        SECP256K1_DECLASSIFY(b, sizeof(b));
        SECP256K1_DECLASSIFY(&flag, sizeof(flag));
        if (flag) {
            CHECK(std::memcmp(a, orig_b, sizeof(a)) == 0 && std::memcmp(b, orig_a, sizeof(b)) == 0,
                  "cswap256 swaps both vectors when flag is set");
        } else {
            CHECK(std::memcmp(a, orig_a, sizeof(a)) == 0 && std::memcmp(b, orig_b, sizeof(b)) == 0,
                  "cswap256 leaves both vectors unchanged when flag is clear");
        }
    }

    // ct_select
    {
        uint64_t a_val = 0x1234;
        uint64_t b_val = 0x5678;
        uint64_t flag = g_rng() & 1;
        SECP256K1_CLASSIFY(&flag, sizeof(flag));
        uint64_t mask_val = secp256k1::ct::bool_to_mask(flag != 0);
        uint64_t result = secp256k1::ct::ct_select(a_val, b_val, mask_val);
        SECP256K1_DECLASSIFY(&result, sizeof(result));
        SECP256K1_DECLASSIFY(&flag, sizeof(flag));
        CHECK(result == (flag ? a_val : b_val), "ct_select returns the expected branch value");
    }

    // ct_lookup (table scan must be constant-time)
    {
        uint64_t table[8][4] = {};
        for (int j = 0; j < 8; ++j)
            for (int k = 0; k < 4; ++k)
                table[j][k] = g_rng();

        uint64_t secret_idx = g_rng() % 8;
        SECP256K1_CLASSIFY(&secret_idx, sizeof(secret_idx));

        uint64_t out[4] = {};
        secp256k1::ct::ct_lookup_256(table, 8,
                                     static_cast<size_t>(secret_idx), out);
        SECP256K1_DECLASSIFY(out, sizeof(out));
        SECP256K1_DECLASSIFY(&secret_idx, sizeof(secret_idx));
        CHECK(std::memcmp(out, table[secret_idx], sizeof(out)) == 0,
              "ct_lookup_256 returns the selected table row");
    }
}

// ============================================================================
// Test 4: CT Generator Multiplication (signing hot path)
// ============================================================================
static void test_ct_generator_mul() {
    g_section = "ct_verif_genmul";
    (void)printf("[4] CT generator multiplication (ct::generator_mul)\n");

    for (int i = 0; i < 5; ++i) {
        Scalar k = make_random_scalar();
        SECP256K1_CLASSIFY(&k, sizeof(k));

        auto R = secp256k1::ct::generator_mul(k);

        SECP256K1_DECLASSIFY(&R, sizeof(R));
        CHECK(!R.is_infinity(), "ct::generator_mul produces valid point");
    }
}

// ============================================================================
// Test 4b: CT Arbitrary Point Scalar Multiplication (generic point hot path)
// ============================================================================
static void test_ct_point_scalar_mul() {
    g_section = "ct_verif_point";
    (void)printf("[4b] CT arbitrary-point scalar multiplication (ct::scalar_mul)\n");

    Point base = Point::generator().scalar_mul(Scalar::from_uint64(17));
    CHECK(!base.is_infinity(), "public base point initialized");

    for (int i = 0; i < 5; ++i) {
        Scalar k = make_random_scalar();
        SECP256K1_CLASSIFY(&k, sizeof(k));

        Point R = secp256k1::ct::scalar_mul(base, k);
        std::uint64_t on_curve_mask = secp256k1::ct::point_is_on_curve(R);

        SECP256K1_DECLASSIFY(&R, sizeof(R));
        SECP256K1_DECLASSIFY(&on_curve_mask, sizeof(on_curve_mask));

        CHECK(!R.is_infinity(), "ct::scalar_mul produces valid point");
        CHECK(on_curve_mask == ~static_cast<std::uint64_t>(0),
              "ct::point_is_on_curve accepts ct::scalar_mul output");
    }
}

// ============================================================================
// Test 5: CT ECDSA Sign (full signing path)
// ============================================================================
static void test_ct_ecdsa_sign() {
    g_section = "ct_verif_ecdsa";
    (void)printf("[5] CT ECDSA sign (ct::ecdsa_sign with secret key)\n");

    std::array<uint8_t, 32> msg_hash{};
    fill_random(msg_hash.data(), 32);

    for (int i = 0; i < 3; ++i) {
        Scalar privkey = make_random_scalar();
        SECP256K1_CLASSIFY(&privkey, sizeof(privkey));

        auto sig = secp256k1::ct::ecdsa_sign(msg_hash, privkey);

        SECP256K1_DECLASSIFY(&sig, sizeof(sig));
        CHECK(sig.r.to_bytes() != Scalar::zero().to_bytes(), "CT ECDSA sign produces valid sig");

        // Verify the signature (public operation, no classify needed)
        SECP256K1_DECLASSIFY(&privkey, sizeof(privkey));
        auto pubkey = Point::generator().scalar_mul(privkey);
        bool ok = secp256k1::ecdsa_verify(msg_hash.data(), pubkey, sig);
        CHECK(ok, "CT ECDSA sig verifies correctly");
    }
}

// ============================================================================
// Test 6: CT Schnorr Sign (BIP-340)
// ============================================================================
static void test_ct_schnorr_sign() {
    g_section = "ct_verif_schnorr";
    (void)printf("[6] CT Schnorr sign (ct::schnorr_sign with secret key)\n");

    std::array<uint8_t, 32> msg{};
    fill_random(msg.data(), 32);
    std::array<uint8_t, 32> aux{};
    fill_random(aux.data(), 32);

    for (int i = 0; i < 3; ++i) {
        Scalar privkey = make_random_scalar();

        // Create keypair (public operation)
        auto kp = secp256k1::schnorr_keypair_create(privkey);

        // Classify the keypair secret (contains private key material)
        SECP256K1_CLASSIFY(&kp.d, sizeof(kp.d));

        auto sig = secp256k1::ct::schnorr_sign(kp, msg, aux);

        SECP256K1_DECLASSIFY(&sig, sizeof(sig));

        // Verify (public)
        bool ok = secp256k1::schnorr_verify(kp.px, msg, sig);
        CHECK(ok, "CT Schnorr sig verifies correctly");
    }
}

// ============================================================================
// Test 7: CT ECDSA Sign Hedged (RFC 6979 + aux_rand)
// ============================================================================
static void test_ct_ecdsa_sign_hedged() {
    g_section = "ct_verif_hedged";
    (void)printf("[7] CT ECDSA hedged sign (ct::ecdsa_sign_hedged)\n");

    std::array<uint8_t, 32> msg_hash{};
    fill_random(msg_hash.data(), 32);
    std::array<uint8_t, 32> aux_rand{};
    fill_random(aux_rand.data(), 32);

    for (int i = 0; i < 3; ++i) {
        Scalar privkey = make_random_scalar();
        SECP256K1_CLASSIFY(&privkey, sizeof(privkey));

        auto sig = secp256k1::ct::ecdsa_sign_hedged(msg_hash, privkey, aux_rand);

        SECP256K1_DECLASSIFY(&sig, sizeof(sig));
        CHECK(sig.r.to_bytes() != Scalar::zero().to_bytes(),
              "CT hedged ECDSA produces valid sig");

        SECP256K1_DECLASSIFY(&privkey, sizeof(privkey));
        auto pubkey = Point::generator().scalar_mul(privkey);
        bool ok = secp256k1::ecdsa_verify(msg_hash.data(), pubkey, sig);
        CHECK(ok, "CT hedged ECDSA sig verifies correctly");
    }
}

// ============================================================================
// Test 8: CT Field -- edge case inputs (zero, one, p-1)
// ============================================================================
static void test_ct_field_edge_cases() {
    g_section = "ct_verif_edge";
    (void)printf("[8] CT field edge cases (zero, one, p-1)\n");

    // These edge values are particularly dangerous for CT violations
    // (early exit on zero, special case for one, etc.)
    FieldElement vals[] = {
        FieldElement::zero(),
        FieldElement::one(),
        FieldElement::from_hex(
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2E"),
    };

    for (auto& val : vals) {
        FieldElement other = make_random_fe();
        SECP256K1_CLASSIFY(&val, sizeof(val));
        SECP256K1_CLASSIFY(&other, sizeof(other));

        FieldElement prod = val * other;
        FieldElement sq   = val.square();

        SECP256K1_DECLASSIFY(&prod, sizeof(prod));
        SECP256K1_DECLASSIFY(&sq,   sizeof(sq));
        SECP256K1_DECLASSIFY(&val,   sizeof(val));
        SECP256K1_DECLASSIFY(&other, sizeof(other));
        CHECK(prod == (val * other) && sq == val.square(),
              "field edge-case mul/sqr preserve expected outputs");
    }
}

// ============================================================================
// Test 9: CT Scalar -- edge case inputs (1, n-1)
// ============================================================================
static void test_ct_scalar_edge_cases() {
    g_section = "ct_verif_scalar_edge";
    (void)printf("[9] CT scalar edge cases (1, n-1)\n");

    Scalar one_s = Scalar::from_uint64(1);
    Scalar nm1 = Scalar::from_bytes({
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
        0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B,
        0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x40
    });

    Scalar secrets[] = { one_s, nm1 };
    for (auto& s : secrets) {
        Scalar other = make_random_scalar();
        SECP256K1_CLASSIFY(&s, sizeof(s));
        SECP256K1_CLASSIFY(&other, sizeof(other));

        Scalar prod = s * other;
        Scalar sum  = s + other;

        SECP256K1_DECLASSIFY(&prod, sizeof(prod));
        SECP256K1_DECLASSIFY(&sum,  sizeof(sum));
        SECP256K1_DECLASSIFY(&s,     sizeof(s));
        SECP256K1_DECLASSIFY(&other, sizeof(other));
        CHECK(prod == (s * other) && sum == (s + other),
              "scalar edge-case mul/add preserve expected outputs");
    }
}

// ============================================================================
// Test 10: CT value_barrier correctness
// ============================================================================
static void test_ct_value_barrier() {
    g_section = "ct_verif_barrier";
    (void)printf("[10] CT value_barrier (optimization fence)\n");

    for (int i = 0; i < 20; ++i) {
        uint64_t val = g_rng();
        uint64_t original = val;
        SECP256K1_CLASSIFY(&val, sizeof(val));

        secp256k1::ct::value_barrier(val);

        SECP256K1_DECLASSIFY(&val, sizeof(val));
        CHECK(val == original, "value_barrier preserves value");
    }
}

// ============================================================================
// Exportable run function (for unified audit runner)
// ============================================================================
int test_ct_verif_formal_run() {
    g_pass = g_fail = 0;

    bool active = ct_verif_active();
    if (!active) {
        // Without Valgrind memcheck or MSAN, CLASSIFY/DECLASSIFY are no-ops.
        // Running the tests would only check arithmetic correctness — not CT
        // properties. Return 77 (ADVISORY_SKIP_CODE) to signal an honest skip
        // rather than a false green.
        //
        // To run the formal check:
        //   cmake -DSECP256K1_CT_VALGRIND=ON ... && ninja test_ct_verif_formal_standalone
        //   valgrind --tool=memcheck --error-exitcode=42 ./test_ct_verif_formal_standalone
        (void)printf("[ct_verif_formal] SKIPPED — not running under Valgrind/MSAN. "
                     "Formal CT verification requires instrumentation.\n");
        return 77;
    }

    (void)printf("  CT-verif backend: ACTIVE (Valgrind/MSAN -- formal checking enabled)\n");

    test_ct_field_ops();
    test_ct_scalar_ops();
    test_ct_primitives();
    test_ct_generator_mul();
    test_ct_point_scalar_mul();
    test_ct_ecdsa_sign();
    test_ct_schnorr_sign();
    test_ct_ecdsa_sign_hedged();
    test_ct_field_edge_cases();
    test_ct_scalar_edge_cases();
    test_ct_value_barrier();

    (void)printf("  [ct_verif_formal] %d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}

// ============================================================================
// Main (standalone mode)
// ============================================================================
#if defined(STANDALONE_TEST)
int main() {
    (void)printf("============================================================\n");
    (void)printf("  Formal CT Verification Test (ct-verif / ctgrind)\n");
    (void)printf("  Track I, Task I5-1\n");
    (void)printf("============================================================\n\n");

    int rc = test_ct_verif_formal_run();

    (void)printf("\n============================================================\n");
    (void)printf("  Summary: %d passed, %d failed\n", g_pass, g_fail);
    if (ct_verif_active()) {
        (void)printf("  Mode: FORMAL (Valgrind/MSAN active)\n");
    } else {
        (void)printf("  Mode: PASSIVE (no CT backend -- run under valgrind)\n");
    }
    (void)printf("============================================================\n");

    return rc;
}
#endif // STANDALONE_TEST
