// ============================================================================
// Test: Multi-Scalar Multiplication + Batch Verification
// ============================================================================

#include "secp256k1/multiscalar.hpp"
#include "secp256k1/batch_verify.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/sha256.hpp"

#include <cstdio>
#include <cstring>
#include <vector>

using namespace secp256k1;
using fast::Scalar;
using fast::Point;

static int tests_run = 0;
static int tests_passed = 0;

#define CHECK(cond, msg) do { \
    ++tests_run; \
    if (cond) { ++tests_passed; printf("  [PASS] %s\n", msg); } \
    else { printf("  [FAIL] %s\n", msg); } \
} while(0)

// -- Shamir's Trick -----------------------------------------------------------

static void test_shamir_trick() {
    printf("\n--- Shamir's Trick ---\n");

    auto G = Point::generator();
    auto a = Scalar::from_uint64(7);
    auto b = Scalar::from_uint64(13);
    auto P = G;
    auto Q = G.scalar_mul(Scalar::from_uint64(5));

    // Expected: a*P + b*Q = 7*G + 13*5G = 7G + 65G = 72G
    auto expected = G.scalar_mul(Scalar::from_uint64(72));
    auto result = shamir_trick(a, P, b, Q);

    CHECK(result.x().to_bytes() == expected.x().to_bytes(),
          "shamir_trick(7, G, 13, 5G) == 72G");

    // Edge: a=0
    auto r2 = shamir_trick(Scalar::zero(), P, b, Q);
    auto e2 = Q.scalar_mul(b);
    CHECK(r2.x().to_bytes() == e2.x().to_bytes(),
          "shamir_trick(0, P, b, Q) == b*Q");

    // Edge: b=0
    auto r3 = shamir_trick(a, P, Scalar::zero(), Q);
    auto e3 = P.scalar_mul(a);
    CHECK(r3.x().to_bytes() == e3.x().to_bytes(),
          "shamir_trick(a, P, 0, Q) == a*P");
}

// -- Multi-Scalar Multiplication ----------------------------------------------

static void test_multi_scalar_mul() {
    printf("\n--- Multi-Scalar Multiplication ---\n");

    auto G = Point::generator();

    // Test with 1 point
    {
        Scalar const s = Scalar::from_uint64(42);
        Point const p = G;
        auto result = multi_scalar_mul(&s, &p, 1);
        auto expected = G.scalar_mul(Scalar::from_uint64(42));
        CHECK(result.x().to_bytes() == expected.x().to_bytes(),
              "multi_scalar_mul: 1 point");
    }

    // Test with 3 points: 2*G + 3*2G + 5*3G = 2G + 6G + 15G = 23G
    {
        std::vector<Scalar> const scalars = {
            Scalar::from_uint64(2),
            Scalar::from_uint64(3),
            Scalar::from_uint64(5)
        };
        std::vector<Point> const points = {
            G,
            G.scalar_mul(Scalar::from_uint64(2)),
            G.scalar_mul(Scalar::from_uint64(3))
        };
        auto result = multi_scalar_mul(scalars, points);
        auto expected = G.scalar_mul(Scalar::from_uint64(23));
        CHECK(result.x().to_bytes() == expected.x().to_bytes(),
              "multi_scalar_mul: 3 points (2G+6G+15G=23G)");
    }

    // Test with 0 points
    {
        auto result = multi_scalar_mul(nullptr, nullptr, 0);
        CHECK(result.is_infinity(), "multi_scalar_mul: 0 points = infinity");
    }

    // Test: sum cancels to infinity (P + (-P) = O)
    {
        Scalar const s1 = Scalar::from_uint64(1);
        Scalar const s2 = Scalar::from_uint64(1);
        Point const P1 = G;
        Point const P2 = G.negate();
        std::vector<Scalar> const scalars = {s1, s2};
        std::vector<Point> const pts = {P1, P2};
        auto result = multi_scalar_mul(scalars, pts);
        CHECK(result.is_infinity(), "multi_scalar_mul: G + (-G) = infinity");
    }
}

// -- Schnorr Batch Verification -----------------------------------------------

static void test_schnorr_batch_verify() {
    printf("\n--- Schnorr Batch Verification ---\n");

    // Create 5 valid Schnorr signatures
    constexpr std::size_t N = 5;
    std::vector<SchnorrBatchEntry> entries(N);
    std::vector<Scalar> keys(N);

    for (std::size_t i = 0; i < N; ++i) {
        keys[i] = Scalar::from_uint64(100 + i);
        entries[i].pubkey_x = schnorr_pubkey(keys[i]);

        // Message: SHA256(i)
        uint8_t ibuf[4] = {
            static_cast<uint8_t>(i), static_cast<uint8_t>(i >> 8),
            static_cast<uint8_t>(i >> 16), static_cast<uint8_t>(i >> 24)
        };
        auto msg_h = SHA256::hash(ibuf, 4);
        entries[i].message = msg_h;

        // Sign
        std::array<uint8_t, 32> const aux{};
        entries[i].signature = schnorr_sign(keys[i], msg_h, aux);
    }

    // All valid
    bool const all_valid = schnorr_batch_verify(entries);
    CHECK(all_valid, "Schnorr batch: 5 valid signatures pass");

    // Individual verify should also pass
    bool individual_ok = true;
    for (std::size_t i = 0; i < N; ++i) {
        if (!schnorr_verify(entries[i].pubkey_x, entries[i].message,
                            entries[i].signature)) {
            individual_ok = false;
        }
    }
    CHECK(individual_ok, "Schnorr batch: individual verification agrees");

    // Corrupt one signature -- batch should fail
    auto corrupted = entries;
    corrupted[2].signature.s = corrupted[2].signature.s + Scalar::one();
    bool const should_fail = schnorr_batch_verify(corrupted);
    CHECK(!should_fail, "Schnorr batch: corrupted sig #2 detected");

    // Identify invalid
    auto invalid = schnorr_batch_identify_invalid(corrupted.data(), corrupted.size());
    CHECK(invalid.size() == 1 && invalid[0] == 2,
          "Schnorr batch identify: correctly finds sig #2");

    // Empty batch
    CHECK(schnorr_batch_verify(nullptr, 0), "Schnorr batch: empty = true");

    // Single entry
    std::vector<SchnorrBatchEntry> const single = {entries[0]};
    CHECK(schnorr_batch_verify(single), "Schnorr batch: single entry pass");
}

// -- ECDSA Batch Verification -------------------------------------------------

static void test_ecdsa_batch_verify() {
    printf("\n--- ECDSA Batch Verification ---\n");

    constexpr std::size_t N = 4;
    std::vector<ECDSABatchEntry> entries(N);
    std::vector<Scalar> keys(N);

    auto G = Point::generator();

    for (std::size_t i = 0; i < N; ++i) {
        keys[i] = Scalar::from_uint64(200 + i);
        entries[i].public_key = G.scalar_mul(keys[i]);

        // Message hash
        uint8_t ibuf[4] = {
            static_cast<uint8_t>(i + 50), 0, 0, 0
        };
        entries[i].msg_hash = SHA256::hash(ibuf, 4);

        // Sign
        entries[i].signature = ecdsa_sign(entries[i].msg_hash, keys[i]);
    }

    // All valid
    CHECK(ecdsa_batch_verify(entries), "ECDSA batch: 4 valid signatures pass");

    // Corrupt one
    auto corrupted = entries;
    corrupted[1].signature.s = corrupted[1].signature.s + Scalar::one();
    CHECK(!ecdsa_batch_verify(corrupted), "ECDSA batch: corrupted sig #1 detected");

    // Identify
    auto invalid = ecdsa_batch_identify_invalid(corrupted.data(), corrupted.size());
    CHECK(invalid.size() == 1 && invalid[0] == 1,
          "ECDSA batch identify: correctly finds sig #1");
}

// -- Main ---------------------------------------------------------------------

int test_multiscalar_batch_run() {
    printf("=== Multi-Scalar Multiplication & Batch Verification Tests ===\n");

    test_shamir_trick();
    test_multi_scalar_mul();
    test_schnorr_batch_verify();
    test_ecdsa_batch_verify();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
