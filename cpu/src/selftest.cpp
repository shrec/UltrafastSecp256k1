// SECP256K1 Library Self-Test
// Comprehensive arithmetic verification with known test vectors
// This ensures all math operations (scalar mul, point add/sub) are correct
// Supports 3 modes: smoke (fast), ci (full), stress (extended)

#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/selftest.hpp"
#if !defined(SECP256K1_PLATFORM_ESP32) && !defined(ESP_PLATFORM) && !defined(IDF_VER) && !defined(SECP256K1_PLATFORM_STM32)
#include "secp256k1/precompute.hpp"
#endif
#include "secp256k1/glv.hpp"
#include <cstring>
#include <vector>
#include <array>

// Suppress MSVC deprecation of std::getenv (safe: read-only use)
#if defined(_MSC_VER)
#pragma warning(disable: 4996)
#endif
#include <cstdlib>
#include <cstdio>

// ESP32/STM32 platform: use printf instead of iostream
#if defined(SECP256K1_PLATFORM_ESP32) || defined(ESP_PLATFORM) || defined(IDF_VER) || defined(SECP256K1_PLATFORM_STM32)
    #define SELFTEST_PRINT(...) printf(__VA_ARGS__)
#else
    #include <iostream>
    #include <fstream>
    #include <sstream>
    #include <iomanip>
    #define SELFTEST_PRINT(...) printf(__VA_ARGS__)
#endif

namespace secp256k1::fast {

// ESP32/STM32: local scalar_mul_generator that uses Point::scalar_mul
// Desktop: use precomputed tables from precompute.hpp
#if defined(SECP256K1_PLATFORM_ESP32) || defined(ESP_PLATFORM) || defined(IDF_VER) || defined(SECP256K1_PLATFORM_STM32)
static inline Point scalar_mul_generator(const Scalar& k) {
    return Point::generator().scalar_mul(k);
}
#endif

// Test vector structure
struct TestVector {
    const char* scalar_hex;
    const char* expected_x;
    const char* expected_y;
    const char* description;
};

// Known test vectors: scalar * G = expected_point
// These are from trusted reference implementation
static const TestVector TEST_VECTORS[] = {
    {
        "4727daf2986a9804b1117f8261aba645c34537e4474e19be58700792d501a591",
        "0566896db7cd8e47ceb5e4aefbcf4d46ec295a15acb089c4affa9fcdd44471ef",
        "1513fcc547db494641ee2f65926e56645ec68cceaccb278a486e68c39ee876c4",
        "Vector 1"
    },
    {
        "c77835cf72699d217c2bbe6c59811b7a599bb640f0a16b3a332ebe64f20b1afa",
        "510f6c70028903e8c0d6f7a156164b972cea569b5a29bb03ff7564dfea9e875a",
        "c02b5ff43ae3b46e281b618abb0cbdaabdd600fbd6f4b78af693dec77080ef56",
        "Vector 2"
    },
    {
        "c401899c059f1c624292fece1933c890ae4970abf56dd4d2c986a5b9d7c9aeb5",
        "8434cbaf8256a8399684ed2212afc204e2e536034612039177bba44e1ea0d1c6",
        "0c34841bd41b0d869b35cfc4be6d57f098ae4beca55dc244c762c3ca0fd56af3",
        "Vector 3"
    },
    {
        "700a25ca2ae4eb40dfa74c9eda069be7e2fc9bfceabb13953ddedd33e1f03f2c",
        "2327ee923f529e67f537a45f633c8201dbee7be0c78d0894e31855843d9fbf0a",
        "f81ad336ee0bd923ec9338dd4b5f4b98d77caba5c153a6511ab15fd2ac6a422e",
        "Vector 4"
    },
    {
        "489206bbfff1b2370619ba0e6a51b74251267e06d3abafb055464bb623d5057a",
        "3ce5eb585c77104f8b877dd5ee574bf9439213b29f027e02e667cec79cd47b9e",
        "7ea30086c7c1f617d4c21c2f6e63cd0386f47ac8a3e97861d19d5d57d7338e3b",
        "Vector 5"
    },
    {
        "0000000000000000000000000000000000000000000000000000000000000001",
        "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798",
        "483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8",
        "1*G (Generator)"
    },
    {
        "0000000000000000000000000000000000000000000000000000000000000002",
        "c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5",
        "1ae168fea63dc339a3c58419466ceaeef7f632653266d0e1236431a950cfe52a",
        "2*G"
    },
    {
        "0000000000000000000000000000000000000000000000000000000000000003",
        "f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9",
        "388f7b0f632de8140fe337e62a37f3566500a99934c2231b6cb9fd7584b8e672",
        "3*G"
    },
    {
        "000000000000000000000000000000000000000000000000000000000000000a",
        "a0434d9e47f3c86235477c7b1ae6ae5d3442d49b1943c2b752a68e2a47e247c7",
        "893aba425419bc27a3b6c7e693a24c696f794c2ed877a1593cbee53b037368d7",
        "10*G"
    },
    {
        "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364140",
        "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798",
        "b7c52588d95c3b9aa25b0403f1eef75702e84bb7597aabe663b82f6f04ef2777",
        "(n-1)*G = -G"
    }
};

// Helper: Compare hex strings (case-insensitive)
static bool hex_equal(const std::string& a, const char* b) {
    if (a.length() != strlen(b)) return false;
    for (size_t i = 0; i < a.length(); i++) {
        char ca = a[i];
        char cb = b[i];
        if (ca >= 'A' && ca <= 'F') ca += 32; // to lowercase
        if (cb >= 'A' && cb <= 'F') cb += 32;
        if (ca != cb) return false;
    }
    return true;
}

// Helper: hex -> 32-byte array
static bool hex_to_bytes32(const std::string& hex, std::array<std::uint8_t, 32>& out) {
    if (hex.size() != 64) return false;
    auto nybble = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
        if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
        return -1;
    };
    for (size_t i = 0; i < 32; ++i) {
        int hi = nybble(hex[2*i]);
        int lo = nybble(hex[2*i + 1]);
        if (hi < 0 || lo < 0) return false;
        out[i] = static_cast<std::uint8_t>((hi << 4) | lo);
    }
    return true;
}

// Test one scalar multiplication vector
static bool test_scalar_mul(const TestVector& vec, bool verbose) {
    if (verbose) {
        SELFTEST_PRINT("  Testing: %s\n", vec.description);
    }
    
    // Parse and compute k * G
    Scalar k = Scalar::from_hex(vec.scalar_hex);
    Point result = scalar_mul_generator(k);
    
    if (result.is_infinity()) {
        if (verbose) {
            SELFTEST_PRINT("    FAILED: Result is infinity!\n");
        }
        return false;
    }
    
    // Compare coordinates
    std::string result_x = result.x().to_hex();
    std::string result_y = result.y().to_hex();
    
    bool x_match = hex_equal(result_x, vec.expected_x);
    bool y_match = hex_equal(result_y, vec.expected_y);
    
    if (x_match && y_match) {
        if (verbose) {
            SELFTEST_PRINT("    PASS\n");
        }
        return true;
    } else {
        if (verbose) {
            SELFTEST_PRINT("    FAIL\n");
            if (!x_match) {
                SELFTEST_PRINT("      Expected X: %s\n", vec.expected_x);
                SELFTEST_PRINT("      Got      X: %s\n", result_x.c_str());
            }
            if (!y_match) {
                SELFTEST_PRINT("      Expected Y: %s\n", vec.expected_y);
                SELFTEST_PRINT("      Got      Y: %s\n", result_y.c_str());
            }
        }
        return false;
    }
}

// Test point addition: (k1*G) + (k2*G) = (k1+k2)*G
static bool test_addition(bool verbose) {
    if (verbose) {
        SELFTEST_PRINT("  Testing: 2*G + 3*G = 5*G\n");
    }
    
    Point pt1 = scalar_mul_generator(Scalar::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000002"));
    Point pt2 = scalar_mul_generator(Scalar::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000003"));
    Point expected = scalar_mul_generator(Scalar::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000005"));
    
    Point result = pt1.add(pt2);
    
    std::string result_x = result.x().to_hex();
    std::string result_y = result.y().to_hex();
    std::string expected_x = expected.x().to_hex();
    std::string expected_y = expected.y().to_hex();
    
    bool match = (result_x == expected_x) && (result_y == expected_y);
    
    if (verbose) {
        if (match) {
            SELFTEST_PRINT("    PASS\n");
        } else {
            SELFTEST_PRINT("    FAIL\n");
            SELFTEST_PRINT("      Expected X: %s\n", expected_x.c_str());
            SELFTEST_PRINT("      Got      X: %s\n", result_x.c_str());
            SELFTEST_PRINT("      Expected Y: %s\n", expected_y.c_str());
            SELFTEST_PRINT("      Got      Y: %s\n", result_y.c_str());
        }
    }
    
    return match;
}

// Test point subtraction: (k1*G) - (k2*G) = (k1-k2)*G
static bool test_subtraction(bool verbose) {
    if (verbose) {
        SELFTEST_PRINT("  Testing: 5*G - 2*G = 3*G\n");
    }
    
    Point pt1 = scalar_mul_generator(Scalar::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000005"));
    Point pt2 = scalar_mul_generator(Scalar::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000002"));
    Point expected = scalar_mul_generator(Scalar::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000003"));
    
    // P1 - P2 = P1 + (-P2)
    Point result = pt1.add(pt2.negate());
    
    std::string result_x = result.x().to_hex();
    std::string result_y = result.y().to_hex();
    std::string expected_x = expected.x().to_hex();
    std::string expected_y = expected.y().to_hex();
    
    bool match = (result_x == expected_x) && (result_y == expected_y);
    
    if (verbose) {
        if (match) {
            SELFTEST_PRINT("    PASS\n");
        } else {
            SELFTEST_PRINT("    FAIL\n");
            SELFTEST_PRINT("      Expected X: %s\n", expected_x.c_str());
            SELFTEST_PRINT("      Got      X: %s\n", result_x.c_str());
            SELFTEST_PRINT("      Expected Y: %s\n", expected_y.c_str());
            SELFTEST_PRINT("      Got      Y: %s\n", result_y.c_str());
        }
    }
    
    return match;
}

// Basic field arithmetic identities (deterministic sanity)
static bool test_field_arithmetic(bool verbose) {
    if (verbose) {
        SELFTEST_PRINT("\nField Arithmetic Test:\n");
    }

    bool ok = true;
    FieldElement zero = FieldElement::zero();
    FieldElement one  = FieldElement::one();
    if (!((zero + zero) == zero)) ok = false;
    if (!((one + zero) == one)) ok = false;
    if (!((one * one) == one)) ok = false;
    if (!((zero * one) == zero)) ok = false;

    FieldElement a = FieldElement::from_uint64(7);
    FieldElement b = FieldElement::from_uint64(5);
    FieldElement neg_a = FieldElement::zero() - a;
    if (!((neg_a + a) == FieldElement::zero())) ok = false;
    if (!(((a + b) - b) == a)) ok = false;
    if (!(b == FieldElement::zero() || (b.inverse() * b) == FieldElement::one())) ok = false;

    if (verbose) {
        SELFTEST_PRINT(ok ? "    PASS\n" : "    FAIL\n");
    }
    return ok;
}

// Basic scalar group identities (without inverse API)
static bool test_scalar_arithmetic(bool verbose) {
    if (verbose) {
        SELFTEST_PRINT("\nScalar Arithmetic Test:\n");
    }
    bool ok = true;
    Scalar z = Scalar::zero();
    Scalar o = Scalar::one();
    if (!((z + z) == z)) ok = false;
    if (!((o + z) == o)) ok = false;
    if (!(((o + o) - o) == o)) ok = false;
    if (verbose) {
        SELFTEST_PRINT(ok ? "    PASS\n" : "    FAIL\n");
    }
    return ok;
}

// Point group identities (O neutral, negation)
static bool test_point_identities(bool verbose) {
    if (verbose) {
        SELFTEST_PRINT("\nPoint Group Identities:\n");
    }
    bool ok = true;
    Point O = Point::infinity();
    Point G = Point::generator();
    if (!(G.add(O).x() == G.x() && G.add(O).y() == G.y())) ok = false;
    Point negG = G.negate();
    if (!G.add(negG).is_infinity()) ok = false;
    if (verbose) {
        SELFTEST_PRINT(ok ? "    PASS\n" : "    FAIL\n");
    }
    return ok;
}

// Helper: compare two points in affine coordinates
static bool points_equal(const Point& a, const Point& b) {
    if (a.is_infinity() && b.is_infinity()) return true;
    if (a.is_infinity() || b.is_infinity()) return false;
    return a.x() == b.x() && a.y() == b.y();
}

// Addition with constant expected: G + 2G = 3G (compare to known constants)
static bool test_addition_constants(bool verbose) {
    if (verbose) {
        SELFTEST_PRINT("\nPoint Addition (constants): G + 2G = 3G\n");
    }
    Point G = Point::generator();
    Point twoG = scalar_mul_generator(Scalar::from_uint64(2));
    Point sum = G.add(twoG);

    // TEST_VECTORS[7] is 3*G
    const auto& exp = TEST_VECTORS[7];
    bool ok = hex_equal(sum.x().to_hex(), exp.expected_x) && hex_equal(sum.y().to_hex(), exp.expected_y);
    if (verbose) SELFTEST_PRINT(ok ? "    PASS\n" : "    FAIL\n");
    return ok;
}

// Subtraction with constant expected: 3G - 2G = 1G
static bool test_subtraction_constants(bool verbose) {
    if (verbose) {
        SELFTEST_PRINT("\nPoint Subtraction (constants): 3G - 2G = 1G\n");
    }
    Point threeG = scalar_mul_generator(Scalar::from_uint64(3));
    Point twoG = scalar_mul_generator(Scalar::from_uint64(2));
    Point diff = threeG.add(twoG.negate());

    // TEST_VECTORS[5] is 1*G
    const auto& exp = TEST_VECTORS[5];
    bool ok = hex_equal(diff.x().to_hex(), exp.expected_x) && hex_equal(diff.y().to_hex(), exp.expected_y);
    if (verbose) SELFTEST_PRINT(ok ? "    PASS\n" : "    FAIL\n");
    return ok;
}

// Doubling with constant expected: (5G).dbl() = 10G
static bool test_doubling_constants(bool verbose) {
    if (verbose) {
        SELFTEST_PRINT("\nPoint Doubling (constants): 2*(5G) = 10G\n");
    }
    Point fiveG = scalar_mul_generator(Scalar::from_uint64(5));
    Point tenG = fiveG.dbl();

    // TEST_VECTORS[8] is 10*G
    const auto& exp = TEST_VECTORS[8];
    bool ok = hex_equal(tenG.x().to_hex(), exp.expected_x) && hex_equal(tenG.y().to_hex(), exp.expected_y);
    if (verbose) SELFTEST_PRINT(ok ? "    PASS\n" : "    FAIL\n");
    return ok;
}

// Negation with constant expected: -G equals (n-1)*G vector
static bool test_negation_constants(bool verbose) {
    if (verbose) {
        SELFTEST_PRINT("\nPoint Negation (constants): -G = (n-1)*G\n");
    }
    Point negG = Point::generator().negate();
    // TEST_VECTORS[9] is (n-1)*G = -G
    const auto& exp = TEST_VECTORS[9];
    bool ok = hex_equal(negG.x().to_hex(), exp.expected_x) && hex_equal(negG.y().to_hex(), exp.expected_y);
    if (verbose) SELFTEST_PRINT(ok ? "    PASS\n" : "    FAIL\n");
    return ok;
}

// Point (de)serialization: compressed/uncompressed encodings sanity
static bool test_point_serialization(bool verbose) {
    if (verbose) {
        SELFTEST_PRINT("\nPoint Serialization:\n");
    }
    auto check_point = [&](const Scalar& k) -> bool {
        Point P = scalar_mul_generator(k);
        auto cx = P.x().to_bytes();
        auto cy = P.y().to_bytes();
        auto comp = P.to_compressed();
        auto uncmp = P.to_uncompressed();
        std::uint8_t expected_prefix = (cy[31] & 1) ? 0x03 : 0x02;
        bool ok = true;
        if (comp[0] != expected_prefix) ok = false;
        for (size_t i = 0; i < 32; ++i) {
            if (comp[1 + i] != cx[i]) { ok = false; break; }
        }
        if (uncmp[0] != 0x04) ok = false;
        for (size_t i = 0; i < 32; ++i) {
            if (uncmp[1 + i] != cx[i]) { ok = false; break; }
        }
        for (size_t i = 0; i < 32; ++i) {
            if (uncmp[33 + i] != cy[i]) { ok = false; break; }
        }
        return ok;
    };
    bool all = true;
    all &= check_point(Scalar::from_hex("0000000000000000000000000000000000000000000000000000000000000001"));
    all &= check_point(Scalar::from_hex("0000000000000000000000000000000000000000000000000000000000000002"));
    all &= check_point(Scalar::from_hex("0000000000000000000000000000000000000000000000000000000000000003"));
    all &= check_point(Scalar::from_hex("000000000000000000000000000000000000000000000000000000000000000a"));
    if (verbose) {
        SELFTEST_PRINT(all ? "    PASS\n" : "    FAIL\n");
    }
    return all;
}

// Batch inversion vs individual inversion
static bool test_batch_inverse(bool verbose) {
    if (verbose) {
        SELFTEST_PRINT("\nBatch Inversion:\n");
    }
    FieldElement elems[4] = {
        FieldElement::from_uint64(3),
        FieldElement::from_uint64(7),
        FieldElement::from_uint64(11),
        FieldElement::from_uint64(19)
    };
    FieldElement copy[4] = { elems[0], elems[1], elems[2], elems[3] };
    fe_batch_inverse(elems, 4);
    bool ok = true;
    for (int i = 0; i < 4; ++i) {
        FieldElement inv = copy[i].inverse();
        if (!(inv == elems[i])) { ok = false; break; }
    }
    if (verbose) {
        SELFTEST_PRINT(ok ? "    PASS\n" : "    FAIL\n");
    }
    return ok;
}

// Expanded batch inversion with a larger, deterministic set
static bool test_batch_inverse_expanded(bool verbose) {
    if (verbose) {
        SELFTEST_PRINT("\nBatch Inversion (expanded 32 elems):\n");
    }
    constexpr size_t N = 32;
    FieldElement elems[N];
    FieldElement copy[N];
    // Deterministic non-zero elements: 3,5,7,...
    for (size_t i = 0; i < N; ++i) {
        std::uint64_t v = 3ULL + 2ULL * static_cast<std::uint64_t>(i);
        elems[i] = FieldElement::from_uint64(v);
        copy[i] = elems[i];
    }
    fe_batch_inverse(elems, N);
    bool ok = true;
    for (size_t i = 0; i < N; ++i) {
        FieldElement inv = copy[i].inverse();
        if (!(inv == elems[i])) { ok = false; break; }
    }
    if (verbose) {
        SELFTEST_PRINT(ok ? "    PASS\n" : "    FAIL\n");
    }
    return ok;
}

// Bilinearity checks for K*Q with non-generator points
// Tests: (Q+G)*K == Q*K + G*K, (Q-G)*K == Q*K - G*K
static bool test_bilinearity_K_times_Q(bool verbose) {
    if (verbose) {
        SELFTEST_PRINT("\nBilinearity: K*(Q+/-G) vs K*Q +/- K*G\n");
    }
    bool ok = true;
    const char* KHEX[] = {
        "0000000000000000000000000000000000000000000000000000000000000005",
        "4727daf2986a9804b1117f8261aba645c34537e4474e19be58700792d501a591",
        "c77835cf72699d217c2bbe6c59811b7a599bb640f0a16b3a332ebe64f20b1afa"
    };
    const char* QHEX[] = {
        "0000000000000000000000000000000000000000000000000000000000000011",
        "0000000000000000000000000000000000000000000000000000000000000067",
        "c401899c059f1c624292fece1933c890ae4970abf56dd4d2c986a5b9d7c9aeb5"
    };
    Point G = Point::generator();
    for (auto kh : KHEX) {
        Scalar K = Scalar::from_hex(kh);
        Point KG = scalar_mul_generator(K);
        for (auto qh : QHEX) {
            Scalar qk = Scalar::from_hex(qh);
            Point Q = scalar_mul_generator(qk); // Q = qk*G (valid arbitrary point)

            Point Lp = Q.add(G).scalar_mul(K);           // (Q+G)*K
            Point Rp = Q.scalar_mul(K).add(KG);          // Q*K + G*K
            if (!points_equal(Lp, Rp)) { ok = false; break; }

            Point Lm = Q.add(G.negate()).scalar_mul(K);  // (Q-G)*K
            Point Rm = Q.scalar_mul(K).add(KG.negate()); // Q*K - G*K
            if (!points_equal(Lm, Rm)) { ok = false; break; }
        }
        if (!ok) break;
    }
    if (verbose) SELFTEST_PRINT(ok ? "    PASS\n" : "    FAIL\n");
    return ok;
}

// Fixed-K plan consistency: scalar_mul_with_plan vs scalar_mul
static bool test_fixedK_plan(bool verbose) {
    if (verbose) {
        SELFTEST_PRINT("\nFixed-K plan: with_plan vs direct scalar_mul\n");
    }
    bool ok = true;
    const char* KHEX[] = {
        TEST_VECTORS[0].scalar_hex,
        TEST_VECTORS[1].scalar_hex,
        "00000000000000000000000000000000000000000000000000000000000000a7"
    };
    const char* QHEX[] = {
        "000000000000000000000000000000000000000000000000000000000000000d",
        "0000000000000000000000000000000000000000000000000000000000000123",
        "700a25ca2ae4eb40dfa74c9eda069be7e2fc9bfceabb13953ddedd33e1f03f2c"
    };
    for (auto kh : KHEX) {
        Scalar K = Scalar::from_hex(kh);
        KPlan plan = KPlan::from_scalar(K, 4);
        for (auto qh : QHEX) {
            Scalar qk = Scalar::from_hex(qh);
            Point Q = scalar_mul_generator(qk);
            Point A = Q.scalar_mul(K);
            Point B = Q.scalar_mul_with_plan(plan);
            if (!points_equal(A, B)) {
                if (verbose) {
                    auto aC = A.to_compressed();
                    auto bC = B.to_compressed();
                    SELFTEST_PRINT("    Mismatch!\n");
                    SELFTEST_PRINT("      K: 0x%s  (neg1=%s, neg2=%s)\n",
                        kh, plan.neg1?"1":"0", plan.neg2?"1":"0");
                    SELFTEST_PRINT("      q: 0x%s\n", qh);
                    // Print hex bytes
                    SELFTEST_PRINT("      A: ");
                    for (auto b : aC) SELFTEST_PRINT("%02x", (int)b);
                    SELFTEST_PRINT("\n");
                    SELFTEST_PRINT("      B: ");
                    for (auto b : bC) SELFTEST_PRINT("%02x", (int)b);
                    SELFTEST_PRINT("\n");
                    // Also compute explicit slow GLV sum for debugging
                    Point phiQ = apply_endomorphism(Q);
                    Point t1 = Q.scalar_mul(plan.k1);
                    Point t2 = phiQ.scalar_mul(plan.k2);
                    if (plan.neg1) t1 = t1.negate();
                    if (plan.neg2) t2 = t2.negate();
                    Point C = t1.add(t2);
                    auto cC = C.to_compressed();
                    SELFTEST_PRINT("      C(slow): ");
                    for (auto b : cC) SELFTEST_PRINT("%02x", (int)b);
                    SELFTEST_PRINT("\n");
                }
                ok = false; break;
            }
        }
        if (!ok) break;
    }
    if (verbose) SELFTEST_PRINT(ok ? "    PASS\n" : "    FAIL\n");
    return ok;
}

// Sequential Q increment property: (Q + i*G)*K = (Q*K) + i*(G*K)
static bool test_sequential_increment_property(bool verbose) {
    if (verbose) {
        SELFTEST_PRINT("\nSequential increment: (Q+i*G)*K vs (Q*K)+i*(G*K)\n");
    }
    bool ok = true;
    // Choose a fixed K and base Q
    Scalar K = Scalar::from_hex("489206bbfff1b2370619ba0e6a51b74251267e06d3abafb055464bb623d5057a");
    Scalar qk = Scalar::from_hex("0000000000000000000000000000000000000000000000000000000000000101");
    Point Q = scalar_mul_generator(qk);
    Point KG = scalar_mul_generator(K);
    // Left side incrementally via next_inplace; Right side via repeated add of KG
    Point left = Q.scalar_mul(K);
    Point right = left; // i=0
    for (int i = 1; i <= 16; ++i) {
        // Q <- Q + G
        Q.next_inplace();
        left = Q.scalar_mul(K);
        right = right.add(KG);
        if (!points_equal(left, right)) { ok = false; break; }
    }
    if (verbose) SELFTEST_PRINT(ok ? "    PASS\n" : "    FAIL\n");
    return ok;
}

// External vector file loader (optional). Format (semicolon-separated, hex lowercase or uppercase):
//  SCALARMUL;kk;expX;expY;desc
//  ADD;x1;y1;x2;y2;expX;expY;desc
//  SUB;x1;y1;x2;y2;expX;expY;desc
static bool run_external_vectors(bool verbose) {
#if defined(SECP256K1_PLATFORM_ESP32) || defined(ESP_PLATFORM) || defined(IDF_VER) || defined(SECP256K1_PLATFORM_STM32)
    // Skip external vectors on embedded - no filesystem
    (void)verbose;
    return true;
#else
#ifdef _WIN32
    char* path = nullptr;
    size_t len = 0;
    if (_dupenv_s(&path, &len, "SECP256K1_SELFTEST_VECTORS") == 0 && path != nullptr) {
        // ... use path ...
        // Note: strictly should free path, but for selftest it's fine
    } else {
        path = nullptr; // fallback
    }
    // const char* path = ...; // adapt existing logic
#else
    const char* path = std::getenv("SECP256K1_SELFTEST_VECTORS"); // lgtm[cpp/path-injection]
    // Reject paths with directory traversal
    if (path && std::string(path).find("..") != std::string::npos) return true;
#endif
    if (!path) return true; // Not provided: treat as success
    // Reject paths with directory traversal
    if (std::string(path).find("..") != std::string::npos) return true;
    std::ifstream in(path);
    if (!in) {
        if (verbose) {
            SELFTEST_PRINT("\n[Selftest] Vector file not found: %s (skipping)\n", path);
        }
        return true; // Non-fatal
    }
    if (verbose) {
        SELFTEST_PRINT("\nExternal Vector Tests (%s):\n", path);
    }
    bool all_ok = true;
    std::string line;
    size_t ln = 0;
    while (std::getline(in, line)) {
        ++ln;
        if (line.empty() || line[0] == '#') continue;
        std::vector<std::string> parts;
        std::stringstream ss(line);
        std::string item;
        while (std::getline(ss, item, ';')) parts.push_back(item);
        if (parts.empty()) continue;
        const std::string& kind = parts[0];
        auto fail_line = [&]() {
            all_ok = false;
            if (verbose) {
                SELFTEST_PRINT("    FAIL (line %zu)\n", ln);
            }
        };
        if (kind == "SCALARMUL") {
            if (parts.size() < 5) { fail_line(); continue; }
            Scalar k = Scalar::from_hex(parts[1]);
            Point r = scalar_mul_generator(k);
            std::string rx = r.x().to_hex();
            std::string ry = r.y().to_hex();
            if (!hex_equal(rx, parts[2].c_str()) || !hex_equal(ry, parts[3].c_str())) {
                fail_line();
            }
        } else if (kind == "ADD" || kind == "SUB") {
            if (parts.size() < 8) { fail_line(); continue; }
            std::array<std::uint8_t, 32> x1b{}, y1b{}, x2b{}, y2b{};
            if (!hex_to_bytes32(parts[1], x1b) || !hex_to_bytes32(parts[2], y1b) ||
                !hex_to_bytes32(parts[3], x2b) || !hex_to_bytes32(parts[4], y2b)) {
                fail_line();
                continue;
            }
            Point pt1 = Point::from_affine(FieldElement::from_bytes(x1b), FieldElement::from_bytes(y1b));
            Point pt2 = Point::from_affine(FieldElement::from_bytes(x2b), FieldElement::from_bytes(y2b));
            Point R = (kind == "ADD") ? pt1.add(pt2) : pt1.add(pt2.negate());
            std::string rx = R.x().to_hex();
            std::string ry = R.y().to_hex();
            if (!hex_equal(rx, parts[5].c_str()) || !hex_equal(ry, parts[6].c_str())) {
                fail_line();
            }
        } else {
            // Unknown entry - ignore
        }
    }
    if (verbose) {
        SELFTEST_PRINT(all_ok ? "    PASS\n" : "    FAIL\n");
    }
    return all_ok;
#endif // ESP32 platform check
}

// -- Deterministic PRNG for stress tests (no <random> dependency) --
struct SelftestRng {
    uint64_t state;
    explicit SelftestRng(uint64_t seed) : state(seed ^ 0x6a09e667f3bcc908ULL) {}
    uint64_t next() {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return state * 0x2545F4914F6CDD1DULL;
    }
};

// -- Boundary scalar KAT: limb boundaries + group order edges --
// These vectors catch carry/normalize/reduce bugs at critical bit positions.
// Expected values computed from independent Python reference implementation.
static bool test_boundary_scalar_vectors(bool verbose) {
    if (verbose) SELFTEST_PRINT("\nBoundary Scalar KAT (limb/order edges):\n");
    struct BVec { const char* k; const char* x; const char* y; const char* desc; };
    static const BVec VECS[] = {
        // Limb boundaries (uint64_t[4] representation)
        {"0000000000000000000000000000000000000000000000000000000100000000",
         "100f44da696e71672791d0a09b7bde459f1215a29b3c03bfefd7835b39a48db0",
         "cdd9e13192a00b772ec8f3300c090666b7ff4a18ff5195ac0fbd5cd62bc65a09",
         "2^32 * G"},
        {"0000000000000000000000000000000000000000000000010000000000000000",
         "3322d401243c4e2582a2147c104d6ecbf774d163db0f5e5313b7e0e742d0e6bd",
         "56e70797e9664ef5bfb019bc4ddaf9b72805f63ea2873af624f3a2e96c28b2a0",
         "2^64 * G"},
        {"0000000000000000000000000000000000000001000000000000000000000000",
         "fea74e3dbe778b1b10f238ad61686aa5c76e3db2be43057632427e2840fb27b6",
         "6e0568db9b0b13297cf674deccb6af93126b596b973f7b77701d3db7f23cb96f",
         "2^96 * G"},
        {"0000000000000000000000000000000100000000000000000000000000000000",
         "8f68b9d2f63b5f339239c1ad981f162ee88c5678723ea3351b7b444c9ec4c0da",
         "662a9f2dba063986de1d90c2b6be215dbbea2cfe95510bfdf23cbf79501fff82",
         "2^128 * G"},
        // GLV split boundary (k near 2^128)
        {"00000000000000000000000000000000ffffffffffffffffffffffffffffffff",
         "6c034fd8cc8bd548e12569b630710400e6c24a05d9d6b32f08522a241e936da8",
         "47ec36379eabcb793bfa408f7898ea619798b51289138f979b8eb3fd33d25f15",
         "(2^128 - 1) * G"},
        {"0000000000000000000000000000000100000000000000000000000000000001",
         "8b300e513eff872cdaa6d12df54a3e332f27ce937be77e3e63c5e885114cbf09",
         "1cec30677f43c0cc446f0d466b8238ea08f6a7aa9aaf716926c6ff28b3b10a39",
         "(2^128 + 1) * G"},
        // High-bit boundary (k near 2^255)
        {"8000000000000000000000000000000000000000000000000000000000000000",
         "b23790a42be63e1b251ad6c94fdef07271ec0aada31db6c3e8bd32043f8be384",
         "fc6b694919d55edbe8d50f88aa81f94517f004f4149ecb58d10a473deb19880e",
         "2^255 * G"},
        {"7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
         "370ebfed473178159fd08c3f7bc07e12301792fbd251554a80298efc666c651d",
         "ad08b75161c542e5503b777625c296b9ef85455756ba7d582bc3c00965dea4a2",
         "(2^255 - 1) * G"},
        {"8000000000000000000000000000000000000000000000000000000000000001",
         "885f71c4561e1733119c66fce72d2209771e096a8305ff8fd36a405afbcbbe10",
         "0887d9db657cf382a6fef4f8269fe0f754c3c33a0c5c87b8f4d006db0392d260",
         "(2^255 + 1) * G"},
        // Near group order
        {"fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd036413f",
         "c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5",
         "e51e970159c23cc65c3a7be6b99315110809cd9acd992f1edc9bce55af301705",
         "(n-2) * G = -2G"},
        // Wrap-around: (n+1) mod n = 1, result = G
        {"fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364142",
         "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798",
         "483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8",
         "(n+1) * G = G (wrap)"},
    };
    bool ok = true;
    for (const auto& v : VECS) {
        Scalar k = Scalar::from_hex(v.k);
        Point r = scalar_mul_generator(k);
        if (r.is_infinity() || !hex_equal(r.x().to_hex(), v.x) || !hex_equal(r.y().to_hex(), v.y)) {
            ok = false;
            if (verbose) {
                SELFTEST_PRINT("    FAIL: %s\n", v.desc);
                SELFTEST_PRINT("      Expected X: %s\n", v.x);
                SELFTEST_PRINT("      Got      X: %s\n", r.x().to_hex().c_str());
            }
            break;
        }
    }
    if (verbose) SELFTEST_PRINT(ok ? "    PASS\n" : "    FAIL\n");
    return ok;
}

// -- Field element limb boundary tests --
// Values at uint64 limb boundaries to catch carry propagation bugs
static bool test_field_limb_boundaries(bool verbose) {
    if (verbose) SELFTEST_PRINT("\nField Limb Boundaries:\n");
    bool ok = true;
    struct LimbCase {
        std::array<uint64_t, 4> limbs;
        const char* desc;
    };
    static const LimbCase CASES[] = {
        // All-ones in each limb position
        {{0xFFFFFFFFFFFFFFFFULL, 0, 0, 0}, "limb0 = max"},
        {{0, 0xFFFFFFFFFFFFFFFFULL, 0, 0}, "limb1 = max"},
        {{0, 0, 0xFFFFFFFFFFFFFFFFULL, 0}, "limb2 = max"},
        {{0, 0, 0, 0xFFFFFFFFFFFFFFFFULL}, "limb3 = max"},
        // All limbs max (normalizes to small nonzero value ~0x100003D0)
        {{0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
          0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL}, "all limbs max"},
        // High bit in each limb
        {{0x8000000000000000ULL, 0, 0, 0}, "limb0 high bit"},
        {{0, 0x8000000000000000ULL, 0, 0}, "limb1 high bit"},
        {{0, 0, 0x8000000000000000ULL, 0}, "limb2 high bit"},
        {{0, 0, 0, 0x8000000000000000ULL}, "limb3 high bit"},
        // Value 1 in each limb position
        {{1, 0, 0, 0}, "limb0 = 1"},
        {{0, 1, 0, 0}, "limb1 = 1"},
        {{0, 0, 1, 0}, "limb2 = 1"},
        {{0, 0, 0, 1}, "limb3 = 1"},
    };
    FieldElement c = FieldElement::from_uint64(0x1234567890ABCDEFULL);
    for (const auto& t : CASES) {
        FieldElement a = FieldElement::from_limbs(t.limbs);
        // a * a^(-1) = 1 (unless a normalizes to 0)
        if (!(a == FieldElement::zero())) {
            FieldElement inv = a.inverse();
            if (!(a * inv == FieldElement::one())) {
                ok = false;
                if (verbose) SELFTEST_PRINT("    FAIL: %s -- a * a^(-1) != 1\n", t.desc);
                break;
            }
        }
        // (a + c) - c = a
        if (!((a + c) - c == a)) {
            ok = false;
            if (verbose) SELFTEST_PRINT("    FAIL: %s -- (a+c)-c != a\n", t.desc);
            break;
        }
        // a * 1 = a
        if (!(a * FieldElement::one() == a)) {
            ok = false;
            if (verbose) SELFTEST_PRINT("    FAIL: %s -- a*1 != a\n", t.desc);
            break;
        }
        // a^2 = a * a
        FieldElement sq = a; sq.square_inplace();
        if (!(sq == a * a)) {
            ok = false;
            if (verbose) SELFTEST_PRINT("    FAIL: %s -- a^2 != a*a\n", t.desc);
            break;
        }
    }
    if (verbose) SELFTEST_PRINT(ok ? "    PASS\n" : "    FAIL\n");
    return ok;
}

// -- Batch inverse size sweep --
// Various sizes including warp/block boundaries to catch GPU parity bugs
static bool test_batch_inverse_sweep(bool verbose) {
    if (verbose) SELFTEST_PRINT("\nBatch Inverse Size Sweep:\n");
    static const size_t SIZES[] = {
        1, 2, 3, 7, 15, 16, 17, 31, 32, 33,
        63, 64, 65, 127, 128, 129, 255, 256, 257, 512, 1024
    };
    bool ok = true;
    for (size_t sz : SIZES) {
        std::vector<FieldElement> elems(sz);
        std::vector<FieldElement> copy(sz);
        for (size_t i = 0; i < sz; ++i) {
            uint64_t v = 3ULL + 2ULL * static_cast<uint64_t>(i);
            elems[i] = FieldElement::from_uint64(v);
            copy[i] = elems[i];
        }
        fe_batch_inverse(elems.data(), sz);
        for (size_t i = 0; i < sz; ++i) {
            if (!(copy[i].inverse() == elems[i])) {
                ok = false;
                if (verbose) SELFTEST_PRINT("    FAIL at size=%zu, idx=%zu\n", sz, i);
                break;
            }
        }
        if (!ok) break;
    }
    if (verbose) SELFTEST_PRINT(ok ? "    PASS\n" : "    FAIL\n");
    return ok;
}

// NOTE: fe_batch_inverse() requires ALL inputs to be non-zero.
// Zero inputs cause undefined cumulative product. This is the standard
// Montgomery trick contract (same as libsecp256k1). Callers must skip
// zero lanes before calling. No zero-input test is needed here;
// the contract is enforced by documentation, not runtime checks.

// -- Repro bundle: prints environment info for reproducibility --
static void print_repro_bundle(SelftestMode mode, uint64_t seed) {
    const char* mode_str = "smoke";
    if (mode == SelftestMode::ci) mode_str = "ci";
    else if (mode == SelftestMode::stress) mode_str = "stress";

    SELFTEST_PRINT("  Mode:     %s\n", mode_str);
    SELFTEST_PRINT("  Seed:     0x%016llx\n", (unsigned long long)seed);

    // Compiler
#if defined(_MSC_VER)
    SELFTEST_PRINT("  Compiler: MSVC %d\n", _MSC_VER);
#elif defined(__clang_major__)
    SELFTEST_PRINT("  Compiler: Clang %d.%d.%d\n", __clang_major__, __clang_minor__, __clang_patchlevel__);
#elif defined(__GNUC__)
    SELFTEST_PRINT("  Compiler: GCC %d.%d.%d\n", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
    SELFTEST_PRINT("  Compiler: unknown\n");
#endif

    // Platform
#if defined(_WIN64)
    SELFTEST_PRINT("  Platform: Windows x64\n");
#elif defined(_WIN32)
    SELFTEST_PRINT("  Platform: Windows x86\n");
#elif defined(__APPLE__)
    #if defined(__aarch64__)
    SELFTEST_PRINT("  Platform: macOS ARM64\n");
    #else
    SELFTEST_PRINT("  Platform: macOS x64\n");
    #endif
#elif defined(__linux__)
    #if defined(__riscv)
    SELFTEST_PRINT("  Platform: Linux RISC-V\n");
    #elif defined(__aarch64__)
    SELFTEST_PRINT("  Platform: Linux ARM64\n");
    #else
    SELFTEST_PRINT("  Platform: Linux x64\n");
    #endif
#elif defined(SECP256K1_PLATFORM_ESP32) || defined(ESP_PLATFORM)
    SELFTEST_PRINT("  Platform: ESP32\n");
#elif defined(SECP256K1_PLATFORM_STM32)
    SELFTEST_PRINT("  Platform: STM32\n");
#elif defined(__EMSCRIPTEN__)
    SELFTEST_PRINT("  Platform: WASM\n");
#else
    SELFTEST_PRINT("  Platform: unknown\n");
#endif

    // Build type
#if defined(NDEBUG)
    SELFTEST_PRINT("  Build:    Release\n");
#else
    SELFTEST_PRINT("  Build:    Debug\n");
#endif

    // Assembly
#if defined(SECP256K1_HAS_ASM) || defined(SECP256K1_HAS_RISCV_ASM) || defined(SECP256K1_HAS_ARM64_ASM)
    SELFTEST_PRINT("  ASM:      enabled\n");
#else
    SELFTEST_PRINT("  ASM:      disabled\n");
#endif

    // Sanitizers
#if defined(__SANITIZE_ADDRESS__)
    SELFTEST_PRINT("  ASan:     ON\n");
#elif defined(__has_feature)
  #if __has_feature(address_sanitizer)
    SELFTEST_PRINT("  ASan:     ON\n");
  #endif
#endif
#if defined(__SANITIZE_THREAD__)
    SELFTEST_PRINT("  TSan:     ON\n");
#elif defined(__has_feature)
  #if __has_feature(thread_sanitizer)
    SELFTEST_PRINT("  TSan:     ON\n");
  #endif
#endif
#if defined(__SANITIZE_UNDEFINED__)
    SELFTEST_PRINT("  UBSan:    ON\n");
#elif defined(__has_feature)
  #if __has_feature(undefined_behavior_sanitizer)
    SELFTEST_PRINT("  UBSan:    ON\n");
  #endif
#endif

    SELFTEST_PRINT("  Repro:    Selftest(true, SelftestMode::%s, 0x%016llx)\n",
                   mode_str, (unsigned long long)seed);
}

// -- Extended kG known vectors: 4G..9G, 15G, 255G --
static bool test_extended_kg_vectors(bool verbose) {
    if (verbose) SELFTEST_PRINT("\nExtended kG Vectors (4G-9G, 15G, 255G):\n");
    struct KGVec { const char* k; const char* x; const char* y; const char* desc; };
    static const KGVec VECS[] = {
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
        {"000000000000000000000000000000000000000000000000000000000000000f",
         "d7924d4f7d43ea965a465ae3095ff41131e5946f3c85f79e44adbcf8e27e080e",
         "581e2872a86c72a683842ec228cc6defea40af2bd896d3a5c504dc9ff6a26b58", "15*G"},
        {"00000000000000000000000000000000000000000000000000000000000000ff",
         "1b38903a43f7f114ed4500b4eac7083fdefece1cf29c63528d563446f972c180",
         "4036edc931a60ae889353f77fd53de4a2708b26b6f5da72ad3394119daf408f9", "255*G"},
    };
    bool ok = true;
    for (const auto& v : VECS) {
        Scalar k = Scalar::from_hex(v.k);
        Point r = scalar_mul_generator(k);
        if (r.is_infinity() || !hex_equal(r.x().to_hex(), v.x) || !hex_equal(r.y().to_hex(), v.y)) {
            ok = false;
            if (verbose) SELFTEST_PRINT("    FAIL: %s\n", v.desc);
            break;
        }
    }
    if (verbose) SELFTEST_PRINT(ok ? "    PASS\n" : "    FAIL\n");
    return ok;
}

// -- Fast kG vs generic kG cross-check --
static bool test_fast_vs_generic_kG(bool verbose) {
    if (verbose) SELFTEST_PRINT("\nFast kG vs Generic kG (small 1-20 + 20 random):\n");
    bool ok = true;
    Point G = Point::generator();
    Point G_aff = Point::from_affine(G.x(), G.y());
    // Small multiples
    for (uint64_t k = 1; k <= 20 && ok; ++k) {
        Scalar sk = Scalar::from_uint64(k);
        Point fast = scalar_mul_generator(sk);
        Point slow = G_aff.scalar_mul(sk);
        if (!points_equal(fast, slow)) ok = false;
    }
    // Deterministic pseudo-random large scalars
    SelftestRng rng(7001);
    for (int i = 0; i < 20 && ok; ++i) {
        std::array<uint64_t, 4> ls{rng.next(), rng.next(), rng.next(), rng.next()};
        Scalar k = Scalar::from_limbs(ls);
        Point fast = scalar_mul_generator(k);
        Point slow = G_aff.scalar_mul(k);
        if (!points_equal(fast, slow)) ok = false;
    }
    if (verbose) SELFTEST_PRINT(ok ? "    PASS\n" : "    FAIL\n");
    return ok;
}

// -- Repeated addition: k*G = G+G+...+G --
static bool test_repeated_addition_consistency(bool verbose) {
    if (verbose) SELFTEST_PRINT("\nRepeated Addition Consistency (k=2..10):\n");
    bool ok = true;
    Point G = Point::generator();
    for (int k = 2; k <= 10 && ok; ++k) {
        Scalar sk = Scalar::from_uint64(static_cast<uint64_t>(k));
        Point by_mul = scalar_mul_generator(sk);
        Point by_add = G;
        for (int i = 1; i < k; ++i) by_add = by_add.add(G);
        if (!points_equal(by_mul, by_add)) ok = false;
    }
    if (verbose) SELFTEST_PRINT(ok ? "    PASS\n" : "    FAIL\n");
    return ok;
}

// -- Field stress: normalization, commutativity, associativity, distributive --
static bool test_field_stress(bool verbose) {
    if (verbose) SELFTEST_PRINT("\nField Stress (normalization + random algebraic laws):\n");
    bool ok = true;
    // p normalizes to 0
    std::array<uint64_t, 4> p_limbs = {
        0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
    };
    if (!(FieldElement::from_limbs(p_limbs) == FieldElement::zero())) ok = false;
    // p+1 normalizes to 1
    std::array<uint64_t, 4> pp1 = {
        0xFFFFFFFEFFFFFC30ULL, 0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
    };
    if (!(FieldElement::from_limbs(pp1) == FieldElement::one())) ok = false;
    // (p-1)^2 = 1
    std::array<uint64_t, 4> pm1_limbs = {
        0xFFFFFFFEFFFFFC2EULL, 0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
    };
    FieldElement pm1 = FieldElement::from_limbs(pm1_limbs);
    if (!(pm1 * pm1 == FieldElement::one())) ok = false;
    // (p-1)+1 = 0
    if (!(pm1 + FieldElement::one() == FieldElement::zero())) ok = false;
    // Random stress: commutativity, associativity, distributive, inverse, square
    SelftestRng rng(8001);
    for (int i = 0; i < 20 && ok; ++i) {
        std::array<uint64_t, 4> la{rng.next(), rng.next(), rng.next(), rng.next()};
        std::array<uint64_t, 4> lb{rng.next(), rng.next(), rng.next(), rng.next()};
        std::array<uint64_t, 4> lc{rng.next(), rng.next(), rng.next(), rng.next()};
        FieldElement a = FieldElement::from_limbs(la);
        FieldElement b = FieldElement::from_limbs(lb);
        FieldElement c = FieldElement::from_limbs(lc);
        if (!(a * b == b * a)) ok = false;                       // commutativity
        if (!((a * b) * c == a * (b * c))) ok = false;            // associativity
        if (!(a * (b + c) == a * b + a * c)) ok = false;         // distributive
        if (!((a - b) + b == a)) ok = false;                      // inverse
        FieldElement sq = a; sq.square_inplace();
        if (!(sq == a * a)) ok = false;                           // square consistency
    }
    if (verbose) SELFTEST_PRINT(ok ? "    PASS\n" : "    FAIL\n");
    return ok;
}

// -- Scalar mul stress: (n-1)^2=1, distributive, associativity --
static bool test_scalar_stress(bool verbose) {
    if (verbose) SELFTEST_PRINT("\nScalar Stress ((n-1)^2=1 + random algebraic laws):\n");
    bool ok = true;
    Scalar one_s = Scalar::one();
    Scalar zero_s = Scalar::zero();
    // (n-1)^2 = 1
    Scalar nm1 = Scalar::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140");
    if (!(nm1 * nm1 == one_s)) ok = false;
    // (n-1)+1 = 0
    if (!(nm1 + one_s == zero_s)) ok = false;
    // Random stress: distributive, associativity, identity, inverse
    SelftestRng rng(9001);
    for (int i = 0; i < 20 && ok; ++i) {
        std::array<uint64_t, 4> la{rng.next(), rng.next(), rng.next(), rng.next()};
        std::array<uint64_t, 4> lb{rng.next(), rng.next(), rng.next(), rng.next()};
        std::array<uint64_t, 4> lc{rng.next(), rng.next(), rng.next(), rng.next()};
        Scalar a = Scalar::from_limbs(la);
        Scalar b = Scalar::from_limbs(lb);
        Scalar c = Scalar::from_limbs(lc);
        if (!(a * (b + c) == a * b + a * c)) ok = false;         // distributive
        if (!((a * b) * c == a * (b * c))) ok = false;            // associativity
        if (!((a - a) == zero_s)) ok = false;                     // a-a=0
        if (!((a * one_s) == a)) ok = false;                      // a*1=a
    }
    if (verbose) SELFTEST_PRINT(ok ? "    PASS\n" : "    FAIL\n");
    return ok;
}

// -- NAF/wNAF encoding validation --
static bool test_naf_wnaf(bool verbose) {
    if (verbose) SELFTEST_PRINT("\nNAF/wNAF Encoding Validation:\n");
    bool ok = true;
    // NAF(0) empty, NAF(1) = {1}
    auto naf0 = Scalar::zero().to_naf();
    if (!naf0.empty()) ok = false;
    auto naf1 = Scalar::one().to_naf();
    if (naf1.size() != 1 || naf1[0] != 1) ok = false;
    // NAF adjacency: no two consecutive non-zero digits
    SelftestRng rng(10001);
    for (int i = 0; i < 20 && ok; ++i) {
        std::array<uint64_t, 4> ls{rng.next(), rng.next(), rng.next(), rng.next()};
        Scalar s = Scalar::from_limbs(ls);
        auto naf = s.to_naf();
        for (size_t j = 1; j < naf.size(); ++j) {
            if (naf[j] != 0 && naf[j-1] != 0) { ok = false; break; }
        }
    }
    // wNAF (w=4): all non-zero digits must be odd and |d| < 2^(w-1) = 8
    for (int i = 0; i < 20 && ok; ++i) {
        std::array<uint64_t, 4> ls{rng.next(), rng.next(), rng.next(), rng.next()};
        Scalar s = Scalar::from_limbs(ls);
        auto wnaf = s.to_wnaf(4);
        for (auto d : wnaf) {
            if (d != 0) {
                if ((d & 1) == 0) { ok = false; break; }
                if (d >= 8 || d <= -8) { ok = false; break; }
            }
        }
    }
    if (verbose) SELFTEST_PRINT(ok ? "    PASS\n" : "    FAIL\n");
    return ok;
}

// -- Point advanced: commutativity, associativity, mixed_add, distributive, edge --
static bool test_point_advanced(bool verbose) {
    if (verbose) SELFTEST_PRINT("\nPoint Advanced (comm/assoc/mixed/dist/edge):\n");
    bool ok = true;
    Point G = Point::generator();
    Point p3 = scalar_mul_generator(Scalar::from_uint64(3));
    Point p5 = scalar_mul_generator(Scalar::from_uint64(5));
    Point p7 = scalar_mul_generator(Scalar::from_uint64(7));
    // Commutativity: 3G+7G = 7G+3G
    if (!points_equal(p3.add(p7), p7.add(p3))) ok = false;
    // Associativity: (3G+5G)+7G = 3G+(5G+7G) = 15G
    Point p15 = scalar_mul_generator(Scalar::from_uint64(15));
    if (!points_equal(p3.add(p5).add(p7), p15)) ok = false;
    if (!points_equal(p3.add(p5.add(p7)), p15)) ok = false;
    // Mixed add: 3G +_mixed G = 4G
    Point p4 = scalar_mul_generator(Scalar::from_uint64(4));
    Point p3m = p3;
    p3m.add_mixed_inplace(G.x(), G.y());
    if (!points_equal(p3m, p4)) ok = false;
    // Distributive: k*(P+Q) = kP + kQ  for k=2..6, P=2G, Q=5G
    Point P = scalar_mul_generator(Scalar::from_uint64(2));
    Point Q = scalar_mul_generator(Scalar::from_uint64(5));
    for (uint64_t k = 2; k <= 6 && ok; ++k) {
        Scalar sk = Scalar::from_uint64(k);
        Point lhs = P.add(Q).scalar_mul(sk);
        Point rhs = P.scalar_mul(sk).add(Q.scalar_mul(sk));
        if (!points_equal(lhs, rhs)) ok = false;
    }
    // K*Q: 2*(7G) = 14G, 3*(7G) = 21G
    if (!points_equal(p7.scalar_mul(Scalar::from_uint64(2)),
                      scalar_mul_generator(Scalar::from_uint64(14)))) ok = false;
    if (!points_equal(p7.scalar_mul(Scalar::from_uint64(3)),
                      scalar_mul_generator(Scalar::from_uint64(21)))) ok = false;
    // Edge: 0*G = infinity
    if (!G.scalar_mul(Scalar::zero()).is_infinity()) ok = false;
    // Edge: n*G = infinity (n mod n = 0)
    Scalar n = Scalar::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
    if (!(n == Scalar::zero())) ok = false;
    if (!G.scalar_mul(n).is_infinity()) ok = false;
    if (verbose) SELFTEST_PRINT(ok ? "    PASS\n" : "    FAIL\n");
    return ok;
}

// ---------------------------------------------------------------
// Thread-local report collector for selftest_report()
// When non-null, tally() appends case results into this report.
// ---------------------------------------------------------------
static thread_local SelftestReport* s_active_report = nullptr;

static inline void tally(int& total, int& passed,
                         const char* name, bool ok) {
    total++;
    if (ok) passed++;
    if (s_active_report) {
        s_active_report->cases.push_back({name, ok, ok ? "" : "FAIL"});
    }
}

// Platform string (compile-time)
static const char* get_platform_string() {
#if defined(_WIN64)
    return "Windows x64";
#elif defined(_WIN32)
    return "Windows x86";
#elif defined(__APPLE__) && defined(__aarch64__)
    return "macOS ARM64";
#elif defined(__APPLE__)
    return "macOS x64";
#elif defined(__linux__) && defined(__riscv)
    return "Linux RISC-V";
#elif defined(__linux__) && defined(__aarch64__)
    return "Linux ARM64";
#elif defined(__linux__)
    return "Linux x64";
#elif defined(SECP256K1_PLATFORM_ESP32) || defined(ESP_PLATFORM)
    return "ESP32";
#elif defined(SECP256K1_PLATFORM_STM32)
    return "STM32";
#elif defined(__EMSCRIPTEN__)
    return "WASM";
#else
    return "unknown";
#endif
}

// Main self-test function (legacy API -- delegates to ci mode)
bool Selftest(bool verbose) {
    return Selftest(verbose, SelftestMode::ci, 0);
}

// -- Mode-aware self-test with repro bundle --
bool Selftest(bool verbose, SelftestMode mode, uint64_t seed) {
    if (seed == 0) seed = 0x53454350324B3147ULL; // "SECP2K1G" default

    if (verbose) {
        SELFTEST_PRINT("\n==============================================\n");
        SELFTEST_PRINT("  SECP256K1 Library Self-Test\n");
        SELFTEST_PRINT("==============================================\n");
        print_repro_bundle(mode, seed);
    }
    
#if !defined(SECP256K1_PLATFORM_ESP32) && !defined(ESP_PLATFORM) && !defined(IDF_VER) && !defined(SECP256K1_PLATFORM_STM32)
    // Initialize precomputed tables (allow env overrides for quick toggles)
    // Only on desktop platforms - embedded uses simple scalar_mul
    FixedBaseConfig cfg{};
    // Environment variable overrides only on desktop platforms
    if (const char* w = std::getenv("SECP256K1_WINDOW_BITS")) {
        unsigned v = static_cast<unsigned>(std::strtoul(w, nullptr, 10));
        if (v >= 2U && v <= 30U) cfg.window_bits = v;
    }
    if (const char* g = std::getenv("SECP256K1_ENABLE_GLV")) {
        if (g[0] == '1' || g[0] == 't' || g[0] == 'T' || g[0] == 'y' || g[0] == 'Y') cfg.enable_glv = true;
    }
    if (const char* j = std::getenv("SECP256K1_USE_JSF")) {
        if (j[0] == '1' || j[0] == 't' || j[0] == 'T' || j[0] == 'y' || j[0] == 'Y') {
            cfg.use_jsf = true;
            cfg.enable_glv = true; // JSF applies to GLV path
        }
    }
    configure_fixed_base(cfg);
    ensure_fixed_base_ready();
#endif

    const bool is_smoke  = (mode == SelftestMode::smoke);
    const bool is_ci     = (mode == SelftestMode::ci) || (mode == SelftestMode::stress);
    const bool is_stress = (mode == SelftestMode::stress);
    (void)is_ci;     // used below in CI-only sections
    (void)is_stress; // used below in stress-only sections

    int passed = 0;
    int total = 0;
    
    // ===============================================================
    // TIER 1: SMOKE -- Core KAT vectors + basic identities (~1-2s)
    // Always run in every mode
    // ===============================================================
    
    // Test scalar multiplication (10 known vectors)
    if (verbose) {
        SELFTEST_PRINT("\nScalar Multiplication Tests:\n");
    }
    
    {
        int vi = 0;
        for (const auto& vec : TEST_VECTORS) {
            char vname[48];
            std::snprintf(vname, sizeof(vname), "scalar_mul_vector_%d", ++vi);
            tally(total, passed, vname, test_scalar_mul(vec, verbose));
        }
    }
    
    // Test point addition
    if (verbose) {
        SELFTEST_PRINT("\nPoint Addition Test:\n");
    }
    tally(total, passed, "point_addition", test_addition(verbose));
    
    // Test point subtraction
    if (verbose) {
        SELFTEST_PRINT("\nPoint Subtraction Test:\n");
    }
    tally(total, passed, "point_subtraction", test_subtraction(verbose));

    // Field arithmetic
    tally(total, passed, "field_arithmetic", test_field_arithmetic(verbose));

    // Scalar arithmetic (basic identities)
    tally(total, passed, "scalar_arithmetic", test_scalar_arithmetic(verbose));

    // Point group identities
    tally(total, passed, "point_identities", test_point_identities(verbose));

    // Point serialization
    tally(total, passed, "point_serialization", test_point_serialization(verbose));

    // Batch inverse (small)
    tally(total, passed, "batch_inverse", test_batch_inverse(verbose));

    // Constant-expected point ops
    tally(total, passed, "addition_constants", test_addition_constants(verbose));
    tally(total, passed, "subtraction_constants", test_subtraction_constants(verbose));
    tally(total, passed, "doubling_constants", test_doubling_constants(verbose));
    tally(total, passed, "negation_constants", test_negation_constants(verbose));

    // Boundary scalar KAT (limb/order edges)
    tally(total, passed, "boundary_scalar_vectors", test_boundary_scalar_vectors(verbose));

    // Field limb boundaries
    tally(total, passed, "field_limb_boundaries", test_field_limb_boundaries(verbose));

    // Extended kG vectors (4G-9G, 15G, 255G)
    tally(total, passed, "extended_kg_vectors", test_extended_kg_vectors(verbose));

    // Point advanced (comm/assoc/mixed/dist/edge)
    tally(total, passed, "point_advanced", test_point_advanced(verbose));

    if (is_smoke) {
        // Smoke mode ends here
        if (verbose) {
            SELFTEST_PRINT("\n==============================================\n");
            SELFTEST_PRINT("  Results: %d/%d tests passed (smoke)\n", passed, total);
            if (passed == total) {
                SELFTEST_PRINT("  [OK] ALL SMOKE TESTS PASSED\n");
            } else {
                SELFTEST_PRINT("  [FAIL] SOME SMOKE TESTS FAILED\n");
            }
            SELFTEST_PRINT("==============================================\n\n");
        }
        return (passed == total);
    }

    // ===============================================================
    // TIER 2: CI -- Full coverage (~30-90s)
    // Runs all smoke + cross-checks, stress, sweeps, bilinearity
    // ===============================================================

    // External vectors (optional, environment-driven)
    tally(total, passed, "external_vectors", run_external_vectors(verbose));

    // Doubling chain vs scalar multiples: for i=1..20, (2^i)G via dbl() equals scalar_mul
    {
        if (verbose) SELFTEST_PRINT("\nDoubling chain vs scalar multiples (2^i * G):\n");
        bool ok = true;
        Point cur = Point::generator(); // 1*G
        for (int i = 1; i <= 20; ++i) {
            cur.dbl_inplace(); // now 2^i * G
            Scalar k = Scalar::from_uint64(1ULL << i);
            Point exp = scalar_mul_generator(k);
            if (!points_equal(cur, exp)) { ok = false; break; }
        }
        if (verbose) SELFTEST_PRINT(ok ? "    PASS\n" : "    FAIL\n");
        tally(total, passed, "doubling_chain_vs_scalar", ok);
    }

    // Large scalar cross-checks (fast vs affine fallback)
    {
        if (verbose) SELFTEST_PRINT("\nLarge scalar cross-checks (fast vs affine):\n");
        bool ok = true;
        const char* L[] = {
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            "8000000000000000000000000000000000000000000000000000000000000000",
            "7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            "deadbeefcafebabef00dfeedfacefeed1234567890abcdef1122334455667788"
        };
        Point G = Point::generator();
        Point G_aff = Point::from_affine(G.x(), G.y());
        for (const char* hx : L) {
            Scalar k = Scalar::from_hex(hx);
            Point fast = scalar_mul_generator(k);
            Point ref  = G_aff.scalar_mul(k);
            if (!points_equal(fast, ref)) { ok = false; break; }
        }
        if (verbose) SELFTEST_PRINT(ok ? "    PASS\n" : "    FAIL\n");
        tally(total, passed, "large_scalar_cross_checks", ok);
    }

    // Squared scalar cases: k^2 * G
    {
        if (verbose) SELFTEST_PRINT("\nSquared scalars k^2 * G (fast vs affine):\n");
        bool ok = true;
        const char* K[] = {
            TEST_VECTORS[0].scalar_hex,
            TEST_VECTORS[1].scalar_hex,
            TEST_VECTORS[2].scalar_hex,
            TEST_VECTORS[3].scalar_hex,
            "0000000000000000000000000000000000000000000000000000000000000013",
            "0000000000000000000000000000000000000000000000000000000000000061",
            "2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e7f8091a"
        };
        Point G = Point::generator();
        Point G_aff = Point::from_affine(G.x(), G.y());
        for (const char* hx : K) {
            Scalar k = Scalar::from_hex(hx);
            Scalar k2 = k * k; // mod n
            Point fast = scalar_mul_generator(k2);
            Point ref  = G_aff.scalar_mul(k2);
            if (!points_equal(fast, ref)) { ok = false; break; }
        }
        if (verbose) SELFTEST_PRINT(ok ? "    PASS\n" : "    FAIL\n");
        tally(total, passed, "squared_scalar_cases", ok);
    }

    // Expanded batch inverse (32 elements)
    tally(total, passed, "batch_inverse_expanded", test_batch_inverse_expanded(verbose));

    // Bilinearity for K*Q with +/-G
    tally(total, passed, "bilinearity_K_times_Q", test_bilinearity_K_times_Q(verbose));

#if !defined(SECP256K1_PLATFORM_ESP32) && !defined(ESP_PLATFORM) && !defined(IDF_VER) && !defined(SECP256K1_PLATFORM_STM32)
    // Batch inverse size sweep (21 sizes)
    tally(total, passed, "batch_inverse_sweep", test_batch_inverse_sweep(verbose));
#endif

#if !defined(SECP256K1_PLATFORM_ESP32) && !defined(ESP_PLATFORM) && !defined(IDF_VER) && !defined(SECP256K1_PLATFORM_STM32)
    // Fixed-K plan equivalence (GLV-based, not available on embedded)
    tally(total, passed, "fixedK_plan", test_fixedK_plan(verbose));
#endif

    // Sequential increment property
    tally(total, passed, "sequential_increment_property", test_sequential_increment_property(verbose));

    // Fast kG vs generic kG (small 1-20 + 20 random)
    tally(total, passed, "fast_vs_generic_kG", test_fast_vs_generic_kG(verbose));

    // Repeated addition consistency (k=2..10)
    tally(total, passed, "repeated_addition_consistency", test_repeated_addition_consistency(verbose));

    // Field stress (normalization + random algebraic laws)
    tally(total, passed, "field_stress", test_field_stress(verbose));

    // Scalar stress ((n-1)^2=1 + random algebraic laws)
    tally(total, passed, "scalar_stress", test_scalar_stress(verbose));

    // NAF/wNAF encoding validation
    tally(total, passed, "naf_wnaf", test_naf_wnaf(verbose));

    // ===============================================================
    // TIER 3: STRESS -- Extended iterations (~10-60 min)
    // Large random sweeps with user-provided seed
    // ===============================================================

    if (is_stress) {
        // Stress: extended fast vs generic kG with many random scalars
        {
            if (verbose) SELFTEST_PRINT("\n[STRESS] Extended fast vs generic kG (1000 random scalars):\n");
            bool ok = true;
            Point G = Point::generator();
            Point G_aff = Point::from_affine(G.x(), G.y());
            SelftestRng rng(seed);
            for (int i = 0; i < 1000 && ok; ++i) {
                std::array<uint64_t, 4> ls{rng.next(), rng.next(), rng.next(), rng.next()};
                Scalar k = Scalar::from_limbs(ls);
                Point fast = scalar_mul_generator(k);
                Point slow = G_aff.scalar_mul(k);
                if (!points_equal(fast, slow)) {
                    ok = false;
                    if (verbose) SELFTEST_PRINT("    FAIL at i=%d\n", i);
                }
            }
            if (verbose) SELFTEST_PRINT(ok ? "    PASS\n" : "    FAIL\n");
            tally(total, passed, "stress_fast_vs_generic_kG_1000", ok);
        }

        // Stress: extended field algebraic laws (500 random triples)
        {
            if (verbose) SELFTEST_PRINT("\n[STRESS] Extended field algebraic laws (500 triples):\n");
            bool ok = true;
            SelftestRng rng(seed ^ 0xF1E1DULL);
            for (int i = 0; i < 500 && ok; ++i) {
                std::array<uint64_t, 4> la{rng.next(), rng.next(), rng.next(), rng.next()};
                std::array<uint64_t, 4> lb{rng.next(), rng.next(), rng.next(), rng.next()};
                std::array<uint64_t, 4> lc{rng.next(), rng.next(), rng.next(), rng.next()};
                FieldElement a = FieldElement::from_limbs(la);
                FieldElement b = FieldElement::from_limbs(lb);
                FieldElement c = FieldElement::from_limbs(lc);
                if (!(a * b == b * a)) ok = false;
                if (!((a * b) * c == a * (b * c))) ok = false;
                if (!(a * (b + c) == a * b + a * c)) ok = false;
                if (!((a - b) + b == a)) ok = false;
                FieldElement sq = a; sq.square_inplace();
                if (!(sq == a * a)) ok = false;
            }
            if (verbose) SELFTEST_PRINT(ok ? "    PASS\n" : "    FAIL\n");
            tally(total, passed, "stress_field_algebraic_laws_500", ok);
        }

        // Stress: extended bilinearity K*(Q+/-G) (100 random K,Q pairs)
        {
            if (verbose) SELFTEST_PRINT("\n[STRESS] Extended bilinearity K*(Q+/-G) (100 pairs):\n");
            bool ok = true;
            SelftestRng rng(seed ^ 0xB11FULL);
            Point G = Point::generator();
            for (int i = 0; i < 100 && ok; ++i) {
                std::array<uint64_t, 4> lk{rng.next(), rng.next(), rng.next(), rng.next()};
                std::array<uint64_t, 4> lq{rng.next(), rng.next(), rng.next(), rng.next()};
                Scalar K = Scalar::from_limbs(lk);
                Scalar qk = Scalar::from_limbs(lq);
                Point Q = scalar_mul_generator(qk);
                Point KG = scalar_mul_generator(K);

                Point Lp = Q.add(G).scalar_mul(K);
                Point Rp = Q.scalar_mul(K).add(KG);
                if (!points_equal(Lp, Rp)) { ok = false; break; }

                Point Lm = Q.add(G.negate()).scalar_mul(K);
                Point Rm = Q.scalar_mul(K).add(KG.negate());
                if (!points_equal(Lm, Rm)) { ok = false; break; }
            }
            if (verbose) SELFTEST_PRINT(ok ? "    PASS\n" : "    FAIL\n");
            tally(total, passed, "stress_bilinearity_100", ok);
        }

#if !defined(SECP256K1_PLATFORM_ESP32) && !defined(ESP_PLATFORM) && !defined(IDF_VER) && !defined(SECP256K1_PLATFORM_STM32)
        // Stress: batch inverse large sweep (up to 8192)
        {
            if (verbose) SELFTEST_PRINT("\n[STRESS] Batch inverse large sweep (up to 8192):\n");
            static const size_t SIZES[] = { 2048, 3072, 4096, 6144, 8192 };
            bool ok = true;
            SelftestRng rng(seed ^ 0xBA7C4ULL);
            for (size_t sz : SIZES) {
                std::vector<FieldElement> elems(sz);
                std::vector<FieldElement> copy(sz);
                for (size_t j = 0; j < sz; ++j) {
                    uint64_t v = rng.next() | 1ULL; // ensure nonzero
                    elems[j] = FieldElement::from_uint64(v);
                    copy[j] = elems[j];
                }
                fe_batch_inverse(elems.data(), sz);
                // Spot-check first/last/middle
                for (size_t idx : {(size_t)0, sz/2, sz-1}) {
                    if (!(copy[idx].inverse() == elems[idx])) {
                        ok = false;
                        if (verbose) SELFTEST_PRINT("    FAIL at size=%zu, idx=%zu\n", sz, idx);
                        break;
                    }
                }
                if (!ok) break;
            }
            if (verbose) SELFTEST_PRINT(ok ? "    PASS\n" : "    FAIL\n");
            tally(total, passed, "stress_batch_inverse_8192", ok);
        }
#endif
    }

    // Summary
    if (verbose) {
        const char* mode_label = is_stress ? "stress" : "ci";
        SELFTEST_PRINT("\n==============================================\n");
        SELFTEST_PRINT("  Results: %d/%d tests passed (%s)\n", passed, total, mode_label);
        if (passed == total) {
            SELFTEST_PRINT("  [OK] ALL TESTS PASSED\n");
        } else {
            SELFTEST_PRINT("  [FAIL] SOME TESTS FAILED\n");
        }
        SELFTEST_PRINT("==============================================\n\n");
    }
    
    return (passed == total);
}

} // namespace secp256k1::fast
