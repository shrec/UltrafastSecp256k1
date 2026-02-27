// Comprehensive test for ALL arithmetic operations correctness
// Tests K*G, P1+P2, K*Q with different methods and scenarios

#include "secp256k1/fast.hpp"
#include "secp256k1/selftest.hpp"
#include <iostream>
#include <iomanip>
#include <array>
#include <cstring>
#include <sstream>
#include <vector>
#include <random>

using namespace secp256k1::fast;

// Helper: Convert hex to bytes
static std::array<uint8_t, 32> hex_to_bytes(const char* hex) {
    std::array<uint8_t, 32> bytes{};
    size_t len = strlen(hex);
    if (len > 64) len = 64;
    
    for (size_t i = 0; i < len; i++) {
        char c = hex[i];
        uint8_t val = 0;
        if (c >= '0' && c <= '9') val = static_cast<uint8_t>(c - '0');
        else if (c >= 'a' && c <= 'f') val = static_cast<uint8_t>(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') val = static_cast<uint8_t>(c - 'A' + 10);
        
        size_t byte_idx = (len - 1 - i) / 2;
        if ((len - 1 - i) % 2 == 0) {
            bytes[31 - byte_idx] |= val;
        } else {
            bytes[31 - byte_idx] |= (val << 4);
        }
    }
    return bytes;
}

// Helper: Field element to hex
static std::string field_to_hex(const FieldElement& f) {
    std::array<uint8_t, 32> bytes = f.to_bytes();
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (uint8_t b : bytes) {
        ss << std::setw(2) << static_cast<int>(b);
    }
    return ss.str();
}

// Helper: Points equal
static bool points_equal(const Point& p1, const Point& p2) {
    if (p1.is_infinity() && p2.is_infinity()) return true;
    if (p1.is_infinity() || p2.is_infinity()) return false;
    return field_to_hex(p1.x()) == field_to_hex(p2.x()) && 
           field_to_hex(p1.y()) == field_to_hex(p2.y());
}

// Known correct K*G results from Bitcoin reference
struct KnownKG {
    const char* k_hex;
    const char* x_hex;
    const char* y_hex;
    const char* desc;
};

const KnownKG KNOWN_KG[] = {
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
};

// ============================================================
// TEST 1: K*G Correctness with scalar_mul method
// ============================================================
static bool test_kg_scalar_mul() {
    std::cout << "\n+==========================================================+" << '\n';
    std::cout << "| TEST 1: K*G using scalar_mul() method                   |" << '\n';
    std::cout << "+==========================================================+" << '\n';
    
    Point G = Point::generator();
    int passed = 0;
    int total = sizeof(KNOWN_KG) / sizeof(KNOWN_KG[0]);
    
    for (int i = 0; i < total; i++) {
        const auto& test = KNOWN_KG[i];
        auto k_bytes = hex_to_bytes(test.k_hex);
        Scalar k = Scalar::from_bytes(k_bytes);
        
        Point result = G.scalar_mul(k);
        
        std::string x_hex = field_to_hex(result.x());
        std::string y_hex = field_to_hex(result.y());
        
        bool match = (x_hex == test.x_hex) && (y_hex == test.y_hex);
        
        std::cout << test.desc << ": " << (match ? "[OK] PASS" : "[FAIL] FAIL");
        if (!match) {
            std::cout << "\n  Expected X: " << test.x_hex;
            std::cout << "\n  Got      X: " << x_hex;
        }
        std::cout << '\n';
        
        if (match) passed++;
    }
    
    std::cout << "\nResult: " << passed << "/" << total << " tests passed" << '\n';
    return passed == total;
}

// ============================================================
// TEST 2: K*G using repeated addition (K*G = G+G+...+G)
// ============================================================
static bool test_kg_repeated_addition() {
    std::cout << "\n+==========================================================+" << '\n';
    std::cout << "| TEST 2: K*G using repeated addition (G+G+...+G)         |" << '\n';
    std::cout << "+==========================================================+" << '\n';
    
    Point G = Point::generator();
    int passed = 0;
    int total = 0;
    
    // Test k=1 to k=10
    for (int k = 1; k <= 10; k++) {
        total++;
        
        // Method 1: scalar_mul
        auto k_bytes = hex_to_bytes(KNOWN_KG[k-1].k_hex);
        Scalar k_scalar = Scalar::from_bytes(k_bytes);
        Point result_mul = G.scalar_mul(k_scalar);
        
        // Method 2: repeated addition
        Point result_add = G;
        for (int i = 1; i < k; i++) {
            result_add = result_add.add(G);
        }
        
        bool match = points_equal(result_mul, result_add);
        
        std::cout << k << "*G: scalar_mul vs repeated_add: " 
                  << (match ? "[OK] EQUAL" : "[FAIL] DIFFERENT") << '\n';
        
        if (match) passed++;
    }
    
    std::cout << "\nResult: " << passed << "/" << total << " tests passed" << '\n';
    return passed == total;
}

// ============================================================
// TEST 3: K*G using doubling (for powers of 2)
// ============================================================
static bool test_kg_doubling() {
    std::cout << "\n+==========================================================+" << '\n';
    std::cout << "| TEST 3: K*G using doubling (2*G, 4*G, 8*G, etc.)        |" << '\n';
    std::cout << "+==========================================================+" << '\n';
    
    Point G = Point::generator();
    int passed = 0;
    int total = 0;
    
    // Test powers of 2: 2^0, 2^1, 2^2, 2^3, 2^4, 2^5
    for (int power = 0; power <= 5; power++) {
        total++;
        int k = 1 << power;  // 2^power
        
        // Method 1: scalar_mul
        std::array<uint8_t, 32> k_bytes{};
        k_bytes[31] = static_cast<uint8_t>(k);
        Scalar k_scalar = Scalar::from_bytes(k_bytes);
        Point result_mul = G.scalar_mul(k_scalar);
        
        // Method 2: repeated doubling
        Point result_dbl = G;
        for (int i = 0; i < power; i++) {
            result_dbl = result_dbl.dbl();
        }
        
        bool match = points_equal(result_mul, result_dbl);
        
        std::cout << k << "*G (2^" << power << "): scalar_mul vs doubling: " 
                  << (match ? "[OK] EQUAL" : "[FAIL] DIFFERENT") << '\n';
        
        if (match) passed++;
    }
    
    std::cout << "\nResult: " << passed << "/" << total << " tests passed" << '\n';
    return passed == total;
}

// ============================================================
// TEST 4: Point Addition Correctness (P1 + P2)
// ============================================================
static bool test_point_addition() {
    std::cout << "\n+==========================================================+" << '\n';
    std::cout << "| TEST 4: Point Addition P1 + P2 Correctness              |" << '\n';
    std::cout << "+==========================================================+" << '\n';
    
    Point G = Point::generator();
    int passed = 0;
    int total = 0;
    
    // Test various additions
    struct AddTest {
        int k1, k2, k_sum;
        const char* desc;
    };
    
    AddTest tests[] = {
        {1, 1, 2, "G + G = 2*G"},
        {1, 2, 3, "G + 2*G = 3*G"},
        {2, 2, 4, "2*G + 2*G = 4*G"},
        {2, 3, 5, "2*G + 3*G = 5*G"},
        {3, 3, 6, "3*G + 3*G = 6*G"},
        {3, 4, 7, "3*G + 4*G = 7*G"},
        {4, 4, 8, "4*G + 4*G = 8*G"},
        {5, 5, 10, "5*G + 5*G = 10*G"},
    };
    
    for (const auto& test : tests) {
        total++;
        
        std::array<uint8_t, 32> k1_bytes{}, k2_bytes{}, ksum_bytes{};
        k1_bytes[31] = static_cast<uint8_t>(test.k1);
        k2_bytes[31] = static_cast<uint8_t>(test.k2);
        ksum_bytes[31] = static_cast<uint8_t>(test.k_sum);
        
        Scalar k1 = Scalar::from_bytes(k1_bytes);
        Scalar k2 = Scalar::from_bytes(k2_bytes);
        Scalar k_sum = Scalar::from_bytes(ksum_bytes);
        
        Point pt1 = G.scalar_mul(k1);
        Point pt2 = G.scalar_mul(k2);
        Point result_add = pt1.add(pt2);
        Point expected = G.scalar_mul(k_sum);
        
        bool match = points_equal(result_add, expected);
        
        std::cout << test.desc << ": " << (match ? "[OK] PASS" : "[FAIL] FAIL") << '\n';
        
        if (match) passed++;
    }
    
    std::cout << "\nResult: " << passed << "/" << total << " tests passed" << '\n';
    return passed == total;
}

// ============================================================
// TEST 5: K*Q for arbitrary point Q (not generator)
// ============================================================
static bool test_kq_arbitrary() {
    std::cout << "\n+==========================================================+" << '\n';
    std::cout << "| TEST 5: K*Q for arbitrary point Q (K*Q correctness)     |" << '\n';
    std::cout << "+==========================================================+" << '\n';
    
    Point G = Point::generator();
    int passed = 0;
    int total = 0;
    
    // Create Q = 7*G
    std::array<uint8_t, 32> seven_bytes{};
    seven_bytes[31] = 7;
    Scalar seven = Scalar::from_bytes(seven_bytes);
    Point Q = G.scalar_mul(seven);
    
    struct KQTest {
        int k;
        int expected;  // k * 7 (since Q = 7*G)
        const char* desc;
    };
    
    KQTest tests[] = {
        {1, 7, "1*(7*G) = 7*G"},
        {2, 14, "2*(7*G) = 14*G"},
        {3, 21, "3*(7*G) = 21*G"},
        {4, 28, "4*(7*G) = 28*G"},
        {5, 35, "5*(7*G) = 35*G"},
        {10, 70, "10*(7*G) = 70*G"},
    };
    
    for (const auto& test : tests) {
        total++;
        
        std::array<uint8_t, 32> k_bytes{}, expected_bytes{};
        k_bytes[31] = static_cast<uint8_t>(test.k);
        expected_bytes[31] = static_cast<uint8_t>(test.expected);
        
        Scalar k = Scalar::from_bytes(k_bytes);
        Scalar expected_scalar = Scalar::from_bytes(expected_bytes);
        
        Point result_kq = Q.scalar_mul(k);
        Point expected = G.scalar_mul(expected_scalar);
        
        bool match = points_equal(result_kq, expected);
        
        std::cout << test.desc << ": " << (match ? "[OK] PASS" : "[FAIL] FAIL") << '\n';
        
        if (match) passed++;
    }
    
    std::cout << "\nResult: " << passed << "/" << total << " tests passed" << '\n';
    return passed == total;
}

// ============================================================
// TEST 6: K*Q with random scalars
// ============================================================
static bool test_kq_random() {
    std::cout << "\n+==========================================================+" << '\n';
    std::cout << "| TEST 6: K*Q with random large scalars                   |" << '\n';
    std::cout << "+==========================================================+" << '\n';
    
    Point G = Point::generator();
    std::mt19937_64 rng(12345);
    int passed = 0;
    int total = 10;
    
    for (int i = 0; i < total; i++) {
        // Random k1 and k2
        std::array<uint8_t, 32> k1_bytes{}, k2_bytes{};
        for (std::size_t j = 0; j < 32; j++) {
            k1_bytes[j] = static_cast<uint8_t>(rng() & 0xFF);
            k2_bytes[j] = static_cast<uint8_t>(rng() & 0xFF);
        }
        
        Scalar k1 = Scalar::from_bytes(k1_bytes);
        Scalar k2 = Scalar::from_bytes(k2_bytes);
        
        // Q = k1*G
        Point Q = G.scalar_mul(k1);
        
        // k2*Q should equal (k1*k2)*G
        Point result_kq = Q.scalar_mul(k2);
        
        // Calculate k1*k2
        Scalar k1k2 = k1 * k2;
        Point expected = G.scalar_mul(k1k2);
        
        bool match = points_equal(result_kq, expected);
        
        std::cout << "Test " << (i+1) << "/10: k2*(k1*G) = (k1*k2)*G: " 
                  << (match ? "[OK] PASS" : "[FAIL] FAIL") << '\n';
        
        if (match) passed++;
    }
    
    std::cout << "\nResult: " << passed << "/" << total << " tests passed" << '\n';
    return passed == total;
}

// ============================================================
// TEST 7: Distributive property: k*(P1+P2) = k*P1 + k*P2
// ============================================================
static bool test_distributive() {
    std::cout << "\n+==========================================================+" << '\n';
    std::cout << "| TEST 7: Distributive: k*(P1+P2) = k*P1 + k*P2           |" << '\n';
    std::cout << "+==========================================================+" << '\n';
    
    Point G = Point::generator();
    int passed = 0;
    int total = 5;
    
    for (int test_num = 0; test_num < total; test_num++) {
        std::array<uint8_t, 32> k_bytes{}, k1_bytes{}, k2_bytes{};
        k_bytes[31] = static_cast<uint8_t>(3 + test_num);
        k1_bytes[31] = 2;
        k2_bytes[31] = 5;
        
        Scalar k = Scalar::from_bytes(k_bytes);
        Scalar k1 = Scalar::from_bytes(k1_bytes);
        Scalar k2 = Scalar::from_bytes(k2_bytes);
        
        Point pt1 = G.scalar_mul(k1);
        Point pt2 = G.scalar_mul(k2);
        
        // Left side: k*(pt1+pt2)
        Point pt1_plus_pt2 = pt1.add(pt2);
        Point left = pt1_plus_pt2.scalar_mul(k);
        
        // Right side: k*pt1 + k*pt2
        Point k_pt1 = pt1.scalar_mul(k);
        Point k_pt2 = pt2.scalar_mul(k);
        Point right = k_pt1.add(k_pt2);
        
        bool match = points_equal(left, right);
        
        std::cout << "k=" << (3+test_num) << ": k*(P1+P2) = k*P1 + k*P2: " 
                  << (match ? "[OK] PASS" : "[FAIL] FAIL") << '\n';
        
        if (match) passed++;
    }
    
    std::cout << "\nResult: " << passed << "/" << total << " tests passed" << '\n';
    return passed == total;
}

// ============================================================
// MAIN
// ============================================================
int test_arithmetic_correctness_run() {
    std::cout << "\n+==========================================================+" << '\n';
    std::cout << "|  COMPREHENSIVE ARITHMETIC CORRECTNESS TESTS              |" << '\n';
    std::cout << "+==========================================================+" << '\n';
    
    bool all_passed = true;
    int total_tests = 7;
    int passed_tests = 0;
    
    if (test_kg_scalar_mul()) passed_tests++;
    if (test_kg_repeated_addition()) passed_tests++;
    if (test_kg_doubling()) passed_tests++;
    if (test_point_addition()) passed_tests++;
    if (test_kq_arbitrary()) passed_tests++;
    if (test_kq_random()) passed_tests++;
    if (test_distributive()) passed_tests++;
    
    all_passed = (passed_tests == total_tests);
    
    std::cout << "\n+==========================================================+" << '\n';
    std::cout << "|                    FINAL RESULT                          |" << '\n';
    std::cout << "+==========================================================+" << '\n';
    std::cout << "|  Test Suites Passed: " << passed_tests << "/" << total_tests << "                                   |" << '\n';
    std::cout << "+==========================================================+" << '\n';
    if (all_passed) {
        std::cout << "|  [OK] ALL ARITHMETIC OPERATIONS ARE CORRECT!               |" << '\n';
        std::cout << "|                                                          |" << '\n';
        std::cout << "|  * K*G works correctly with all methods                 |" << '\n';
        std::cout << "|  * P1 + P2 always produces correct results              |" << '\n';
        std::cout << "|  * K*Q works for arbitrary points and scalars           |" << '\n';
    } else {
        std::cout << "|  [FAIL] SOME TESTS FAILED - Review results above             |" << '\n';
    }
    std::cout << "+==========================================================+" << '\n';
    
    return all_passed ? 0 : 1;
}
