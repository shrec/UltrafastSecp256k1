// Comprehensive Field Arithmetic Test
// Tests: mul, square, add, sub, normalize, reduce, edge cases, random stress
// Style: CHECK macro with pass/fail counters, Selftest(true) 36/36 upfront

#include <secp256k1/field.hpp>
#include <secp256k1/scalar.hpp>
#include <secp256k1/selftest.hpp>
#include <iostream>
#include <random>
#include <array>
#include <cstdint>

using namespace secp256k1::fast;

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "FAIL: " << (msg) << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        ++g_fail; \
    } else { ++g_pass; } \
} while(0)

// ================================================================
// 1. Field Multiplication
// ================================================================
static void test_field_mul() {
    std::cout << "-- Field Multiplication --\n";

    // Basic known products
    CHECK(FieldElement::from_uint64(2) * FieldElement::from_uint64(3) == FieldElement::from_uint64(6),
          "2*3 = 6");
    CHECK(FieldElement::from_uint64(7) * FieldElement::from_uint64(11) == FieldElement::from_uint64(77),
          "7*11 = 77");
    CHECK(FieldElement::from_uint64(0) * FieldElement::from_uint64(12345) == FieldElement::zero(),
          "0 * x = 0");
    CHECK(FieldElement::from_uint64(12345) * FieldElement::zero() == FieldElement::zero(),
          "x * 0 = 0");

    // Identity: a * 1 = a
    FieldElement const a = FieldElement::from_uint64(0xCAFEBABECAFEBABEULL);
    CHECK(a * FieldElement::one() == a, "a * 1 = a");
    CHECK(FieldElement::one() * a == a, "1 * a = a");

    // Commutativity: a * b = b * a
    FieldElement const b = FieldElement::from_uint64(0x12345678ABCDEF01ULL);
    CHECK(a * b == b * a, "a*b = b*a");

    // (p-1)^2 mod p = 1 (since p-1 == -1 mod p)
    std::array<uint64_t, 4> const pm1_limbs = {
        0xFFFFFFFEFFFFFC2EULL, 0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
    };
    FieldElement const pm1 = FieldElement::from_limbs(pm1_limbs);
    CHECK(pm1 * pm1 == FieldElement::one(), "(p-1)^2 = 1");

    // square_inplace consistency
    FieldElement a2 = a;
    a2.square_inplace();
    CHECK(a2 == a * a, "square_inplace == a*a");

    // Associativity: (a*b)*c = a*(b*c)
    FieldElement const c = FieldElement::from_uint64(0xAAAABBBBCCCCDDDDULL);
    CHECK((a * b) * c == a * (b * c), "(a*b)*c = a*(b*c)");

    // Distributivity random stress: a*(b+c) = a*b + a*c
    std::mt19937_64 rng(101);
    for (int i = 0; i < 500; i++) {
        std::array<uint64_t, 4> const la{rng(), rng(), rng(), rng()};
        std::array<uint64_t, 4> const lb{rng(), rng(), rng(), rng()};
        std::array<uint64_t, 4> const lc{rng(), rng(), rng(), rng()};
        FieldElement const fa = FieldElement::from_limbs(la);
        FieldElement const fb = FieldElement::from_limbs(lb);
        FieldElement const fc = FieldElement::from_limbs(lc);
        CHECK(fa * (fb + fc) == fa * fb + fa * fc, "distributive random");
    }
}

// ================================================================
// 2. Field Square
// ================================================================
static void test_field_square() {
    std::cout << "-- Field Square --\n";

    // 0^2 = 0
    FieldElement const z = FieldElement::zero();
    FieldElement z2 = z;
    z2.square_inplace();
    CHECK(z2 == z, "0^2 = 0");

    // 1^2 = 1
    FieldElement const o = FieldElement::one();
    FieldElement o2 = o;
    o2.square_inplace();
    CHECK(o2 == o, "1^2 = 1");

    // Random: sq(a) == a*a
    std::mt19937_64 rng(202);
    for (int i = 0; i < 500; i++) {
        std::array<uint64_t, 4> const la{rng(), rng(), rng(), rng()};
        FieldElement const fa = FieldElement::from_limbs(la);
        FieldElement sq = fa;
        sq.square_inplace();
        CHECK(sq == fa * fa, "random sq(a) == a*a");
    }
}

// ================================================================
// 3. Field Add / Sub
// ================================================================
static void test_field_add_sub() {
    std::cout << "-- Field Add / Sub --\n";

    FieldElement const z = FieldElement::zero();
    FieldElement const a = FieldElement::from_uint64(0xDEADBEEFCAFE1234ULL);

    // Identity
    CHECK(a + z == a, "a + 0 = a");
    CHECK(z + a == a, "0 + a = a");
    CHECK(a - z == a, "a - 0 = a");
    CHECK(a - a == z, "a - a = 0");

    // Commutativity: a + b = b + a
    FieldElement const b = FieldElement::from_uint64(0x12345678ABCDEF01ULL);
    CHECK(a + b == b + a, "a + b = b + a");

    // Inverse: (a - b) + b = a
    CHECK(a - b + b == a, "(a-b)+b = a");

    // Associativity: (a + b) + c = a + (b + c)
    FieldElement const c = FieldElement::from_uint64(0xFFFFFFFFFFFFFFFFULL);
    CHECK((a + b) + c == a + (b + c), "(a+b)+c = a+(b+c)");

    // Modular wrap: big + big (tests normalization)
    FieldElement const big = FieldElement::from_uint64(0xFFFFFFFFFFFFFFFFULL);
    FieldElement const sum = big + big;
    FieldElement const mul2 = big * FieldElement::from_uint64(2);
    CHECK(sum == mul2, "big+big == big*2 (modular wrap)");

    // Random stress: (a-b)+b = a
    std::mt19937_64 rng(303);
    for (int i = 0; i < 500; i++) {
        std::array<uint64_t, 4> const la{rng(), rng(), rng(), rng()};
        std::array<uint64_t, 4> const lb{rng(), rng(), rng(), rng()};
        FieldElement const fa = FieldElement::from_limbs(la);
        FieldElement const fb = FieldElement::from_limbs(lb);
        CHECK((fa - fb) + fb == fa, "random (a-b)+b = a");
    }

    // Random stress commutativity
    for (int i = 0; i < 500; i++) {
        std::array<uint64_t, 4> const la{rng(), rng(), rng(), rng()};
        std::array<uint64_t, 4> const lb{rng(), rng(), rng(), rng()};
        FieldElement const fa = FieldElement::from_limbs(la);
        FieldElement const fb = FieldElement::from_limbs(lb);
        CHECK(fa + fb == fb + fa, "random a+b = b+a");
    }
}

// ================================================================
// 4. Field Normalization edge cases
// ================================================================
static void test_field_normalize() {
    std::cout << "-- Field Normalization --\n";

    FieldElement const z = FieldElement::zero();

    // p normalizes to 0
    std::array<uint64_t, 4> const p_limbs = {
        0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
    };
    CHECK(FieldElement::from_limbs(p_limbs) == z, "p normalizes to 0");

    // p+1 normalizes to 1
    std::array<uint64_t, 4> const pp1 = {
        0xFFFFFFFEFFFFFC30ULL, 0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
    };
    CHECK(FieldElement::from_limbs(pp1) == FieldElement::one(), "p+1 = 1");

    // 2p normalizes to 0
    // 2p = 0x1FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFDFFFFF85E
    // In limbs (after mod): still 0
    [[maybe_unused]] std::array<uint64_t, 4> const twop = {
        0xFFFFFFFDFFFFF85EULL, 0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
    };
    // 2p mod p = p, which is 0; the above is 2p truncated to 256 bits = 2p - 2^256,
    // which equals p - (2^256 - p) = p - 0x1000003D1 ... actually just check via arithmetic
    FieldElement const pm1 = FieldElement::from_limbs({
        0xFFFFFFFEFFFFFC2EULL, 0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
    });
    // p-1 + 1 should be 0
    CHECK(pm1 + FieldElement::one() == z, "(p-1) + 1 = 0");

    // p-1 + p-1 = 2(p-1) mod p = p-2
    FieldElement const pm2 = FieldElement::from_limbs({
        0xFFFFFFFEFFFFFC2DULL, 0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
    });
    CHECK(pm1 + pm1 == pm2, "(p-1)+(p-1) = p-2");
}

// ================================================================
// 5. Scalar Arithmetic
// ================================================================
static void test_scalar_arithmetic() {
    std::cout << "-- Scalar Arithmetic --\n";

    Scalar const zero = Scalar::zero();
    Scalar const one  = Scalar::one();

    // Identities
    CHECK(zero + zero == zero, "0+0 = 0");
    CHECK(one + zero == one,   "1+0 = 1");
    CHECK(zero + one == one,   "0+1 = 1");
    CHECK(one - one == zero,   "1-1 = 0");
    CHECK(one * one == one,    "1*1 = 1");
    CHECK(one * zero == zero,  "1*0 = 0");
    CHECK(zero * one == zero,  "0*1 = 0");

    // Small mul
    Scalar const s7 = Scalar::from_uint64(7);
    Scalar const s3 = Scalar::from_uint64(3);
    CHECK(s7 * s3 == Scalar::from_uint64(21), "7*3 = 21");

    // Commutativity
    CHECK(s7 * s3 == s3 * s7, "7*3 = 3*7");

    // (n-1) == -1 mod n, so (n-1)^2 = 1
    // n = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    Scalar const nm1 = Scalar::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140");
    CHECK(nm1 * nm1 == one, "(n-1)^2 = 1");

    // (n-1) + 1 = 0 mod n
    CHECK(nm1 + one == zero, "(n-1)+1 = 0");

    // Distributive: a*(b+c) = a*b + a*c  (random)
    std::mt19937_64 rng(404);
    for (int i = 0; i < 1000; i++) {
        std::array<uint8_t, 32> ba{}, bb{}, bc{};
        for (std::size_t j = 0; j < 32; j++) {
            ba[j] = static_cast<uint8_t>(rng());
            bb[j] = static_cast<uint8_t>(rng());
            bc[j] = static_cast<uint8_t>(rng());
        }
        Scalar const a = Scalar::from_bytes(ba);
        Scalar const b = Scalar::from_bytes(bb);
        Scalar const c = Scalar::from_bytes(bc);
        CHECK(a * (b + c) == a * b + a * c, "distributive random");
    }

    // Associativity: (a*b)*c = a*(b*c)
    for (int i = 0; i < 200; i++) {
        std::array<uint8_t, 32> ba{}, bb{}, bc{};
        for (std::size_t j = 0; j < 32; j++) {
            ba[j] = static_cast<uint8_t>(rng());
            bb[j] = static_cast<uint8_t>(rng());
            bc[j] = static_cast<uint8_t>(rng());
        }
        Scalar const a = Scalar::from_bytes(ba);
        Scalar const b = Scalar::from_bytes(bb);
        Scalar const c = Scalar::from_bytes(bc);
        CHECK((a * b) * c == a * (b * c), "associative random");
    }

    // Additive inverse: a + (n-1)*a = (n-1+1)*a = 0*a = 0? No -- (a + (n-a)) = 0
    // Actually: a - a = 0
    for (int i = 0; i < 200; i++) {
        std::array<uint8_t, 32> ba{};
        for (std::size_t j = 0; j < 32; j++) ba[j] = static_cast<uint8_t>(rng());
        Scalar const a = Scalar::from_bytes(ba);
        CHECK((a - a) == zero, "a - a = 0 random");
        CHECK((a + zero) == a, "a + 0 = a random");
        CHECK((a * one) == a,  "a * 1 = a random");
    }
}

// ================================================================
// 6. Scalar encoding: NAF / wNAF
// ================================================================
static void test_scalar_encoding() {
    std::cout << "-- Scalar Encoding (NAF / wNAF) --\n";

    // NAF of 0 should be empty
    Scalar const zero = Scalar::zero();
    auto naf0 = zero.to_naf();
    CHECK(naf0.empty(), "NAF(0) is empty");

    // NAF of 1 should be {1}
    Scalar const one = Scalar::one();
    auto naf1 = one.to_naf();
    CHECK(naf1.size() == 1 && naf1[0] == 1, "NAF(1) = {1}");

    // NAF adjacency property: no two consecutive non-zero digits
    std::mt19937_64 rng(505);
    for (int i = 0; i < 200; i++) {
        std::array<uint8_t, 32> ba{};
        for (std::size_t j = 0; j < 32; j++) ba[j] = static_cast<uint8_t>(rng());
        Scalar const s = Scalar::from_bytes(ba);
        auto naf = s.to_naf();
        bool adjacent_ok = true;
        for (size_t k = 1; k < naf.size(); k++) {
            if (naf[k] != 0 && naf[k-1] != 0) { adjacent_ok = false; break; }
        }
        CHECK(adjacent_ok, "NAF adjacency property");
    }

    // wNAF: all non-zero digits should be odd and |d| < 2^w
    for (unsigned w = 3; w <= 6; w++) {
        for (int i = 0; i < 50; i++) {
            std::array<uint8_t, 32> ba{};
            for (std::size_t j = 0; j < 32; j++) ba[j] = static_cast<uint8_t>(rng());
            Scalar const s = Scalar::from_bytes(ba);
            auto wnaf = s.to_wnaf(w);
            bool vals_ok = true;
            int const limit = 1 << (w - 1);
            for (auto d : wnaf) {
                if (d != 0) {
                    if ((d & 1) == 0) vals_ok = false;        // must be odd
                    if (d > limit || d < -limit) vals_ok = false; // |d| < 2^(w-1)
                }
            }
            CHECK(vals_ok, "wNAF values odd & bounded");
        }
    }
}

int test_mul_run() {
    std::cout << "\n+==========================================================+\n";
    std::cout << "|  Comprehensive Field & Scalar Arithmetic Tests          |\n";
    std::cout << "+==========================================================+\n\n";

    test_field_mul();
    test_field_square();
    test_field_add_sub();
    test_field_normalize();
    test_scalar_arithmetic();
    test_scalar_encoding();

    std::cout << "\n==========================================================\n";
    std::cout << "Results: " << g_pass << " passed, " << g_fail << " failed\n";

    if (g_fail > 0) {
        std::cout << "*** FAILURES DETECTED ***\n";
        return 1;
    }
    std::cout << "All tests PASSED.\n\n";
    return 0;
}
