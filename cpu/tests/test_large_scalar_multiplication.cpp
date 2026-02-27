// Comprehensive Scalar Multiplication & Point Operations Test
// Tests: kG (known vectors), kG (fast vs generic), K*Q, point addition,
//        doubling chains, random K*Q, distributive, edge cases (infinity, n-1, etc.)
// Style: CHECK macro, pass/fail counters, Selftest(true) 36/36 upfront

#include "secp256k1/fast.hpp"
#include "secp256k1/selftest.hpp"
#include <iostream>
#include <iomanip>
#include <array>
#include <cstring>
#include <sstream>
#include <random>

using namespace secp256k1::fast;

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "FAIL: " << (msg) << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        ++g_fail; \
    } else { ++g_pass; } \
} while(0)

// Helper: compare points in affine
static bool points_equal(const Point& a, const Point& b) {
    if (a.is_infinity() && b.is_infinity()) return true;
    if (a.is_infinity() || b.is_infinity()) return false;
    return a.x() == b.x() && a.y() == b.y();
}

// Helper: field element to hex
static std::string field_to_hex(const FieldElement& f) {
    auto bytes = f.to_bytes();
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (uint8_t const b : bytes) ss << std::setw(2) << static_cast<int>(b);
    return ss.str();
}

// ================================================================
// 1. Known k*G test vectors (Bitcoin reference)
// ================================================================
struct KnownKG {
    const char* k_hex;
    const char* x_hex;
    const char* y_hex;
    const char* desc;
};

static const KnownKG KNOWN_KG[] = {
    {"0000000000000000000000000000000000000000000000000000000000000001",
     "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798",
     "483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8", "1*G"},
    {"0000000000000000000000000000000000000000000000000000000000000002",
     "c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5",
     "1ae168fea63dc339a3c58419466ceaeef7f632653266d0e1236431a950cfe52a", "2*G"},
    {"0000000000000000000000000000000000000000000000000000000000000003",
     "f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9",
     "388f7b0f632de8140fe337e62a37f3566500a99934c2231b6cb9fd7584b8e672", "3*G"},
    {"0000000000000000000000000000000000000000000000000000000000000004",
     "e493dbf1c10d80f3581e4904930b1404cc6c13900ee0758474fa94abe8c4cd13",
     "51ed993ea0d455b75642e2098ea51448d967ae33bfbdfe40cfe97bdc47739922", "4*G"},
    {"0000000000000000000000000000000000000000000000000000000000000005",
     "2f8bde4d1a07209355b4a7250a5c5128e88b84bddc619ab7cba8d569b240efe4",
     "d8ac222636e5e3d6d4dba9dda6c9c426f788271bab0d6840dca87d3aa6ac62d6", "5*G"},
    {"0000000000000000000000000000000000000000000000000000000000000006",
     "fff97bd5755eeea420453a14355235d382f6472f8568a18b2f057a1460297556",
     "ae12777aacfbb620f3be96017f45c560de80f0f6518fe4a03c870c36b075f297", "6*G"},
    {"0000000000000000000000000000000000000000000000000000000000000007",
     "5cbdf0646e5db4eaa398f365f2ea7a0e3d419b7e0330e39ce92bddedcac4f9bc",
     "6aebca40ba255960a3178d6d861a54dba813d0b813fde7b5a5082628087264da", "7*G"},
    {"0000000000000000000000000000000000000000000000000000000000000008",
     "2f01e5e15cca351daff3843fb70f3c2f0a1bdd05e5af888a67784ef3e10a2a01",
     "5c4da8a741539949293d082a132d13b4c2e213d6ba5b7617b5da2cb76cbde904", "8*G"},
    {"0000000000000000000000000000000000000000000000000000000000000009",
     "acd484e2f0c7f65309ad178a9f559abde09796974c57e714c35f110dfc27ccbe",
     "cc338921b0a7d9fd64380971763b61e9add888a4375f8e0f05cc262ac64f9c37", "9*G"},
    {"000000000000000000000000000000000000000000000000000000000000000a",
     "a0434d9e47f3c86235477c7b1ae6ae5d3442d49b1943c2b752a68e2a47e247c7",
     "893aba425419bc27a3b6c7e693a24c696f794c2ed877a1593cbee53b037368d7", "10*G"},
    {"000000000000000000000000000000000000000000000000000000000000000f",
     "d7924d4f7d43ea965a465ae3095ff41131e5946f3c85f79e44adbcf8e27e080e",
     "581e2872a86c72a683842ec228cc6defea40af2bd896d3a5c504dc9ff6a26b58", "15*G"},
    {"00000000000000000000000000000000000000000000000000000000000000ff",
     "1b38903a43f7f114ed4500b4eac7083fdefece1cf29c63528d563446f972c180",
     "4036edc931a60ae889353f77fd53de4a2708b26b6f5da72ad3394119daf408f9", "255*G"},
};

static void test_known_kg_vectors() {
    std::cout << "-- Known k*G Vectors (Bitcoin Reference) --\n";

    Point const G = Point::generator();
    constexpr int total = sizeof(KNOWN_KG) / sizeof(KNOWN_KG[0]);

    for (int i = 0; i < total; i++) {
        const auto& t = KNOWN_KG[i];
        Scalar const k = Scalar::from_hex(t.k_hex);

        Point const result = G.scalar_mul(k);
        std::string const x = field_to_hex(result.x());
        std::string const y = field_to_hex(result.y());

        CHECK(x == t.x_hex && y == t.y_hex,
              std::string(t.desc) + " known vector");
    }
}

// ================================================================
// 2. Fast (scalar_mul_generator) vs generic (G.scalar_mul) path
// ================================================================
static void test_fast_vs_generic() {
    std::cout << "-- Fast kG vs Generic kG --\n";

    Point const G = Point::generator();

    // Small multiples 1..50
    for (uint64_t k = 1; k <= 50; k++) {
        Scalar const sk = Scalar::from_uint64(k);
        Point const fast = scalar_mul_generator(sk);
        Point const slow = G.scalar_mul(sk);
        CHECK(points_equal(fast, slow),
              "kG fast=generic k=" + std::to_string(k));
    }

    // Random large scalars
    std::mt19937_64 rng(111);
    for (int i = 0; i < 100; i++) {
        std::array<uint64_t, 4> const ls{rng(), rng(), rng(), rng()};
        Scalar const k = Scalar::from_limbs(ls);
        Point const fast = scalar_mul_generator(k);
        Point const slow = G.scalar_mul(k);
        CHECK(points_equal(fast, slow), "random kG fast=generic");
    }
}

// ================================================================
// 3. Large scalar values (powers of 2)
// ================================================================
static void test_large_scalars() {
    std::cout << "-- Large Scalars (2^32, 2^64, 2^128, 2^252, 2^255-1) --\n";

    Point const G = Point::generator();
    const char* const large_hex[] = {
        "0000000000000000000000000000000000000000000000000000000100000000",
        "0000000000000000000000000000000000000000000000010000000000000000",
        "0000000000000000000000000000000100000000000000000000000000000000",
        "1000000000000000000000000000000000000000000000000000000000000000",
        "7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
    };

    for (auto hex : large_hex) {
        Scalar const k = Scalar::from_hex(hex);
        Point const result = G.scalar_mul(k);
        CHECK(!result.is_infinity(), std::string("large k=") + hex + " not infinity");

        // Cross-check: fast vs generic
        Point const fast = scalar_mul_generator(k);
        CHECK(points_equal(result, fast), std::string("large k=") + hex + " fast=generic");
    }
}

// ================================================================
// 4. Repeated addition: k*G == G+G+...+G (k times)
// ================================================================
static void test_repeated_addition() {
    std::cout << "-- Repeated Addition k*G = G+G+...+G --\n";

    Point const G = Point::generator();

    for (int k = 1; k <= 10; k++) {
        Scalar const sk = Scalar::from_uint64(static_cast<uint64_t>(k));
        Point const by_mul = G.scalar_mul(sk);

        Point by_add = G;
        for (int i = 1; i < k; i++) {
            by_add = by_add.add(G);
        }

        CHECK(points_equal(by_mul, by_add),
              std::to_string(k) + "*G = repeated add");
    }
}

// ================================================================
// 5. Doubling chain: 2^k * G via k doublings
// ================================================================
static void test_doubling_chain() {
    std::cout << "-- Doubling Chain 2^k * G --\n";

    Point const G = Point::generator();

    for (int power = 0; power <= 20; power++) {
        // Via doubling
        Point by_dbl = G;
        for (int i = 0; i < power; i++) by_dbl.dbl_inplace();

        // Via scalar_mul (only up to 2^63 fits in uint64)
        if (power < 63) {
            Scalar const sk = Scalar::from_uint64(1ULL << power);
            Point const by_mul = G.scalar_mul(sk);
            CHECK(points_equal(by_dbl, by_mul),
                  "2^" + std::to_string(power) + "*G doubling chain");
        } else {
            // Just check not infinity
            CHECK(!by_dbl.is_infinity(),
                  "2^" + std::to_string(power) + "*G not infinity");
        }
    }
}

// ================================================================
// 6. Point Addition correctness
// ================================================================
static void test_point_addition() {
    std::cout << "-- Point Addition --\n";

    Point const G = Point::generator();

    struct AddTest { uint64_t k1, k2, ksum; const char* desc; };
    AddTest const tests[] = {
        {1,1,2,"G+G=2G"}, {1,2,3,"G+2G=3G"}, {2,2,4,"2G+2G=4G"},
        {2,3,5,"2G+3G=5G"}, {3,3,6,"3G+3G=6G"}, {3,4,7,"3G+4G=7G"},
        {4,4,8,"4G+4G=8G"}, {5,5,10,"5G+5G=10G"}, {1,9,10,"G+9G=10G"},
    };

    for (const auto& t : tests) {
        Point const p1 = G.scalar_mul(Scalar::from_uint64(t.k1));
        Point const p2 = G.scalar_mul(Scalar::from_uint64(t.k2));
        Point const expected = G.scalar_mul(Scalar::from_uint64(t.ksum));
        Point const result = p1.add(p2);
        CHECK(points_equal(result, expected), t.desc);
    }

    // Commutativity: P + Q = Q + P
    Point const p3 = G.scalar_mul(Scalar::from_uint64(3));
    Point const p7 = G.scalar_mul(Scalar::from_uint64(7));
    CHECK(points_equal(p3.add(p7), p7.add(p3)), "P+Q = Q+P");

    // Associativity: (3G+5G)+7G = 3G+(5G+7G) = 15G
    Point const p5 = G.scalar_mul(Scalar::from_uint64(5));
    Point const p15 = G.scalar_mul(Scalar::from_uint64(15));
    Point const lhs = p3.add(p5).add(p7);
    Point const rhs = p3.add(p5.add(p7));
    CHECK(points_equal(lhs, p15), "(3G+5G)+7G = 15G");
    CHECK(points_equal(rhs, p15), "3G+(5G+7G) = 15G");

    // Mixed add: Jacobian + Affine
    Point const p4 = G.scalar_mul(Scalar::from_uint64(4));
    Point p3_mixed = p3;
    p3_mixed.add_mixed_inplace(G.x(), G.y());
    CHECK(points_equal(p3_mixed, p4), "3G +_mixed G = 4G");
}

// ================================================================
// 7. K*Q for arbitrary point Q (not generator)
// ================================================================
static void test_kq_arbitrary() {
    std::cout << "-- K*Q Arbitrary Point --\n";

    Point const G = Point::generator();
    Point const Q = G.scalar_mul(Scalar::from_uint64(7)); // Q = 7*G

    struct KQTest { uint64_t k, expected; const char* desc; };
    KQTest const tests[] = {
        {1,7,"1*(7G)=7G"}, {2,14,"2*(7G)=14G"}, {3,21,"3*(7G)=21G"},
        {4,28,"4*(7G)=28G"}, {5,35,"5*(7G)=35G"}, {10,70,"10*(7G)=70G"},
    };

    for (const auto& t : tests) {
        Point const result = Q.scalar_mul(Scalar::from_uint64(t.k));
        Point const expected = G.scalar_mul(Scalar::from_uint64(t.expected));
        CHECK(points_equal(result, expected), t.desc);
    }
}

// ================================================================
// 8. Random k2*(k1*G) = (k1*k2)*G
// ================================================================
static void test_kq_random() {
    std::cout << "-- Random K*Q = (k1*k2)*G --\n";

    Point const G = Point::generator();
    std::mt19937_64 rng(222);

    for (int i = 0; i < 50; i++) {
        std::array<uint8_t, 32> b1{}, b2{};
        for (std::size_t j = 0; j < 32; j++) {
            b1[j] = static_cast<uint8_t>(rng());
            b2[j] = static_cast<uint8_t>(rng());
        }
        Scalar const k1 = Scalar::from_bytes(b1);
        Scalar const k2 = Scalar::from_bytes(b2);

        Point const Q = G.scalar_mul(k1);
        Point const k2Q = Q.scalar_mul(k2);
        Scalar const k1k2 = k1 * k2;
        Point const expected = G.scalar_mul(k1k2);
        CHECK(points_equal(k2Q, expected), "k2*(k1*G) = (k1*k2)*G random");
    }
}

// ================================================================
// 9. Distributive: k*(P1+P2) = k*P1 + k*P2
// ================================================================
static void test_distributive() {
    std::cout << "-- Distributive k*(P+Q) = kP + kQ --\n";

    Point const G = Point::generator();
    Point const pt1 = G.scalar_mul(Scalar::from_uint64(2));
    Point const pt2 = G.scalar_mul(Scalar::from_uint64(5));

    for (uint64_t k = 2; k <= 20; k++) {
        Scalar const sk = Scalar::from_uint64(k);

        Point const lhs = pt1.add(pt2).scalar_mul(sk);
        Point const rhs = pt1.scalar_mul(sk).add(pt2.scalar_mul(sk));
        CHECK(points_equal(lhs, rhs), "k=" + std::to_string(k) + " dist");
    }

    // Random large k
    std::mt19937_64 rng(333);
    for (int i = 0; i < 20; i++) {
        std::array<uint8_t, 32> bk{};
        for (std::size_t j = 0; j < 32; j++) bk[j] = static_cast<uint8_t>(rng());
        Scalar const sk = Scalar::from_bytes(bk);

        Point const lhs = pt1.add(pt2).scalar_mul(sk);
        Point const rhs = pt1.scalar_mul(sk).add(pt2.scalar_mul(sk));
        CHECK(points_equal(lhs, rhs), "random k distributive");
    }
}

// ================================================================
// 10. Edge cases: n*G=infinity, (n-1)*G=-G, 0*G=infinity
// ================================================================
static void test_edge_cases() {
    std::cout << "-- Edge Cases --\n";

    Point const G = Point::generator();

    // 0*G = infinity
    Point const zero_g = G.scalar_mul(Scalar::zero());
    CHECK(zero_g.is_infinity(), "0*G = infinity");

    // 1*G = G
    Point const one_g = G.scalar_mul(Scalar::one());
    CHECK(points_equal(one_g, G), "1*G = G");

    // (n-1)*G: should be -G (same x, negated y)
    Scalar const nm1 = Scalar::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140");
    Point const nm1_g = G.scalar_mul(nm1);
    CHECK(nm1_g.x() == G.x(), "(n-1)*G.x = G.x");
    CHECK(!(nm1_g.y() == G.y()), "(n-1)*G.y != G.y (negation)");

    // n*G = infinity (n == 0 mod n -> scalar reduces to 0)
    Scalar const n_scalar = Scalar::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
    CHECK(n_scalar == Scalar::zero(), "n mod n = 0");
    Point const n_g = G.scalar_mul(n_scalar);
    CHECK(n_g.is_infinity(), "n*G = infinity");

    // P + (-P) = infinity
    Point const neg_g = nm1_g; // (n-1)*G = -G
    Point const sum = G.add(neg_g);
    CHECK(sum.is_infinity(), "G + (-G) = infinity");

    // Infinity + P = P
    Point const inf = Point::infinity();
    Point const inf_plus_g = inf.add(G);
    CHECK(points_equal(inf_plus_g, G), "inf + G = G");
}

int test_large_scalar_multiplication_run() {
    // Initialize precompute tables for scalar_mul_generator
    FixedBaseConfig config{};
    config.window_bits = 15;
    config.enable_glv = true;
    config.thread_count = 1;
    configure_fixed_base(config);

    std::cout << "\n+==========================================================+\n";
    std::cout << "|  Comprehensive Point/Scalar Multiplication Tests        |\n";
    std::cout << "+==========================================================+\n\n";

    test_known_kg_vectors();
    test_fast_vs_generic();
    test_large_scalars();
    test_repeated_addition();
    test_doubling_chain();
    test_point_addition();
    test_kq_arbitrary();
    test_kq_random();
    test_distributive();
    test_edge_cases();

    std::cout << "\n==========================================================\n";
    std::cout << "Results: " << g_pass << " passed, " << g_fail << " failed\n";

    if (g_fail > 0) {
        std::cout << "*** FAILURES DETECTED ***\n";
        return 1;
    }
    std::cout << "All tests PASSED.\n\n";
    return 0;
}
