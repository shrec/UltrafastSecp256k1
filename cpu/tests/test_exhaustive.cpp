// ============================================================================
// Exhaustive Algebraic Verification Tests
// ============================================================================
// Inspired by bitcoin-core/secp256k1's exhaustive_tests.c.
//
// Since secp256k1 has a prime-order group (no small subgroups), we cannot
// enumerate all group elements. Instead, we exhaustively verify algebraic
// identities over small scalar ranges [1..N] where N = 1024.
//
// Verified properties:
//   1. Closure: k*G is on the curve for all k
//   2. Consistency: k*G + G = (k+1)*G (additive increment)
//   3. Homomorphism: a*G + b*G = (a+b)*G for all pairs (a,b)
//   4. Scalar associativity: k*(l*G) = (k*l)*G
//   5. Point addition associativity: (A + B) + C = A + (B + C)
//   6. Point addition commutativity: A + B = B + A
//   7. Doubling: 2*P = P + P
//   8. Identity: P + O = P, O + P = P
//   9. Inverse: P + (-P) = O
//  10. Order: n*G = O (curve order)
//  11. Negation: (n-1)*G = -G
//  12. CT consistency: ct::scalar_mul matches fast::scalar_mul
//  13. Pippenger MSM: sum(s_i * P_i) matches naive sum
//  14. Comb generator: comb_gen_mul(k) matches k*G
// ============================================================================

#include "secp256k1/fast.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/ct/ops.hpp"
#include "secp256k1/pippenger.hpp"
#include "secp256k1/ecmult_gen_comb.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <chrono>
#include <vector>

using FE = secp256k1::fast::FieldElement;
using SC = secp256k1::fast::Scalar;
using PT = secp256k1::fast::Point;

static int g_pass = 0;
static int g_fail = 0;

[[maybe_unused]] static std::string fe_hex_short(const FE& f) {
    auto b = f.to_bytes();
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (std::size_t i = 0; i < 8; ++i) ss << std::setw(2) << static_cast<int>(b[i]);
    ss << "...";
    return ss.str();
}

static bool pt_eq(const PT& a, const PT& b) {
    if (a.is_infinity() && b.is_infinity()) return true;
    if (a.is_infinity() || b.is_infinity()) return false;
    return a.x().to_bytes() == b.x().to_bytes() &&
           a.y().to_bytes() == b.y().to_bytes();
}

#define CHECK(cond, msg)                                        \
    do {                                                        \
        if (cond) { ++g_pass; }                                 \
        else {                                                  \
            ++g_fail;                                           \
            std::cerr << "  FAIL: " << msg << std::endl;        \
        }                                                       \
    } while (0)

// ----------------------------------------------------------------------------
// Test 1: Closure -- every k*G is on the curve
// ----------------------------------------------------------------------------
static void test_closure(unsigned N) {
    std::cout << "  [1] Closure: k*G on curve for k=1.." << N << std::endl;
    PT G = PT::generator();
    PT P = G;
    for (unsigned k = 1; k <= N; ++k) {
        CHECK(!P.is_infinity(), "k*G should not be infinity for small k");
        // Verify: y^2 = x^3 + 7 (affine check)
        FE x = P.x();
        FE y = P.y();
        FE lhs = y.square();
        FE rhs = x.square() * x + FE::from_uint64(7);
        CHECK(lhs == rhs, "k*G not on curve for k=" + std::to_string(k));
        P = P.add(G);
    }
}

// ----------------------------------------------------------------------------
// Test 2: Additive consistency -- k*G + G = (k+1)*G
// ----------------------------------------------------------------------------
static void test_additive_consistency(unsigned N) {
    std::cout << "  [2] Additive consistency: k*G + G = (k+1)*G, k=1.." << N << std::endl;
    PT G = PT::generator();
    
    // Build reference table: ref[k] = k*G via repeated addition
    std::vector<PT> ref(N + 2);
    ref[0] = PT::infinity();
    ref[1] = G;
    for (unsigned k = 2; k <= N + 1; ++k) {
        ref[k] = ref[k - 1].add(G);
    }
    
    // Verify: ref[k] + G = ref[k+1]
    for (unsigned k = 1; k <= N; ++k) {
        PT sum = ref[k].add(G);
        CHECK(pt_eq(sum, ref[k + 1]),
              "k*G + G != (k+1)*G for k=" + std::to_string(k));
    }
}

// ----------------------------------------------------------------------------
// Test 3: Homomorphism -- a*G + b*G = (a+b)*G
// ----------------------------------------------------------------------------
static void test_homomorphism(unsigned N) {
    std::cout << "  [3] Homomorphism: a*G + b*G = (a+b)*G, a,b=1.." << N << std::endl;
    PT G = PT::generator();
    
    // Build reference table
    unsigned M = 2 * N + 1;
    std::vector<PT> ref(M + 1);
    ref[0] = PT::infinity();
    ref[1] = G;
    for (unsigned k = 2; k <= M; ++k) {
        ref[k] = ref[k - 1].add(G);
    }

    // Check all pairs (a, b) with step to keep runtime manageable
    unsigned step = (N > 64) ? N / 32 : 1;
    unsigned checks = 0;
    for (unsigned a = 1; a <= N; a += step) {
        for (unsigned b = 1; b <= N; b += step) {
            PT lhs = ref[a].add(ref[b]);
            PT rhs = ref[a + b];
            CHECK(pt_eq(lhs, rhs),
                  "a*G + b*G != (a+b)*G for a=" + std::to_string(a) +
                  ", b=" + std::to_string(b));
            ++checks;
        }
    }
    std::cout << "    " << checks << " pairs verified" << std::endl;
}

// ----------------------------------------------------------------------------
// Test 4: Scalar mul consistency -- scalar_mul(k) matches iterated addition
// ----------------------------------------------------------------------------
static void test_scalar_mul_consistency(unsigned N) {
    std::cout << "  [4] Scalar mul: scalar_mul(k) vs iterated add, k=1.." << N << std::endl;
    PT G = PT::generator();
    PT iterP = G;
    
    for (unsigned k = 1; k <= N; ++k) {
        SC s = SC::from_uint64(k);
        PT mulP = G.scalar_mul(s);
        CHECK(pt_eq(mulP, iterP),
              "scalar_mul(" + std::to_string(k) + ") != iterated add");
        iterP = iterP.add(G);
    }
}

// ----------------------------------------------------------------------------
// Test 5: Scalar associativity -- k*(l*G) = (k*l)*G
// ----------------------------------------------------------------------------
static void test_scalar_associativity() {
    std::cout << "  [5] Scalar associativity: k*(l*G) = (k*l)*G" << std::endl;
    PT G = PT::generator();
    
    // Test with selected (k, l) pairs
    const uint64_t pairs[][2] = {
        {2, 3}, {3, 5}, {7, 11}, {13, 17}, {100, 200},
        {255, 255}, {1000, 999}, {12345, 6789}
    };
    
    for (auto& [k, l] : pairs) {
        SC sk = SC::from_uint64(k);
        SC sl = SC::from_uint64(l);
        SC skl = sk * sl;
        
        PT lG = G.scalar_mul(sl);
        PT klG_left = lG.scalar_mul(sk);
        PT klG_right = G.scalar_mul(skl);
        
        CHECK(pt_eq(klG_left, klG_right),
              "k*(l*G) != (k*l)*G for k=" + std::to_string(k) +
              ", l=" + std::to_string(l));
    }
}

// ----------------------------------------------------------------------------
// Test 6: Point addition -- associativity, commutativity, identity, inverse
// ----------------------------------------------------------------------------
static void test_point_addition_axioms() {
    std::cout << "  [6] Addition axioms: assoc, commut, identity, inverse" << std::endl;
    PT G = PT::generator();
    PT O = PT::infinity();
    
    // Generate several distinct points
    std::vector<PT> pts;
    PT P = G;
    for (int i = 0; i < 32; ++i) {
        pts.push_back(P);
        P = P.add(G);
    }
    
    // Commutativity: A + B = B + A
    for (std::size_t i = 0; i < 16; ++i) {
        for (std::size_t j = i + 1; j < 16; j += 3) {
            PT ab = pts[i].add(pts[j]);
            PT ba = pts[j].add(pts[i]);
            CHECK(pt_eq(ab, ba), "commutativity failed i=" + std::to_string(i) +
                                 " j=" + std::to_string(j));
        }
    }
    
    // Associativity: (A + B) + C = A + (B + C)
    for (std::size_t i = 0; i < 8; ++i) {
        for (std::size_t j = i + 1; j < 16; j += 3) {
            for (std::size_t k = j + 1; k < 24; k += 5) {
                PT ab_c = pts[i].add(pts[j]).add(pts[k]);
                PT a_bc = pts[i].add(pts[j].add(pts[k]));
                CHECK(pt_eq(ab_c, a_bc),
                      "associativity failed i=" + std::to_string(i) +
                      " j=" + std::to_string(j) + " k=" + std::to_string(k));
            }
        }
    }
    
    // Identity: P + O = P, O + P = P
    for (std::size_t i = 0; i < 8; ++i) {
        CHECK(pt_eq(pts[i].add(O), pts[i]), "P + O != P");
        CHECK(pt_eq(O.add(pts[i]), pts[i]), "O + P != P");
    }
    
    // Inverse: P + (-P) = O
    for (std::size_t i = 0; i < 8; ++i) {
        PT neg = pts[i].negate();
        PT sum = pts[i].add(neg);
        CHECK(sum.is_infinity(), "P + (-P) != O for i=" + std::to_string(i));
    }
}

// ----------------------------------------------------------------------------
// Test 7: Doubling consistency -- 2*P = P + P
// ----------------------------------------------------------------------------
static void test_doubling() {
    std::cout << "  [7] Doubling: 2*P = P + P" << std::endl;
    PT G = PT::generator();
    PT P = G;
    
    for (int i = 0; i < 64; ++i) {
        PT dbl = P.dbl();
        PT add = P.add(P);
        CHECK(pt_eq(dbl, add), "dbl != add for i=" + std::to_string(i));
        P = P.add(G);
    }
}

// ----------------------------------------------------------------------------
// Test 8: Order of the curve -- n*G = O, (n-1)*G = -G
// ----------------------------------------------------------------------------
static void test_order() {
    std::cout << "  [8] Curve order: n*G = O, (n-1)*G = -G" << std::endl;
    PT G = PT::generator();
    
    // n = group order of secp256k1
    SC n = SC::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
    PT nG = G.scalar_mul(n);
    CHECK(nG.is_infinity(), "n*G != O (must be identity)");
    
    // (n-1)*G should equal -G
    SC n_minus_1 = n - SC::one();
    PT n1G = G.scalar_mul(n_minus_1);
    PT negG = G.negate();
    CHECK(pt_eq(n1G, negG), "(n-1)*G != -G");
    
    // 0*G = O
    SC zero = SC::zero();
    PT zeroG = G.scalar_mul(zero);
    CHECK(zeroG.is_infinity(), "0*G != O");
    
    // 1*G = G
    PT oneG = G.scalar_mul(SC::one());
    CHECK(pt_eq(oneG, G), "1*G != G");
}

// ----------------------------------------------------------------------------
// Test 9: Scalar arithmetic exhaustive -- a + b, a * b, a - b for small values
// ----------------------------------------------------------------------------
static void test_scalar_arithmetic(unsigned N) {
    std::cout << "  [9] Scalar arithmetic exhaustive, N=" << N << std::endl;
    
    unsigned step = (N > 64) ? N / 32 : 1;
    unsigned checks = 0;
    
    for (unsigned a = 0; a <= N; a += step) {
        for (unsigned b = 0; b <= N; b += step) {
            SC sa = SC::from_uint64(a);
            SC sb = SC::from_uint64(b);
            
            // Addition
            SC sum = sa + sb;
            SC expected_sum = SC::from_uint64(a + b);
            CHECK(sum == expected_sum,
                  "scalar add failed: " + std::to_string(a) + "+" + std::to_string(b));
            
            // Multiplication
            SC prod = sa * sb;
            SC expected_prod = SC::from_uint64(static_cast<uint64_t>(a) * b);
            CHECK(prod == expected_prod,
                  "scalar mul failed: " + std::to_string(a) + "*" + std::to_string(b));
            
            // Subtraction (for a >= b)
            if (a >= b) {
                SC diff = sa - sb;
                SC expected_diff = SC::from_uint64(a - b);
                CHECK(diff == expected_diff,
                      "scalar sub failed: " + std::to_string(a) + "-" + std::to_string(b));
            }
            
            ++checks;
        }
    }
    std::cout << "    " << checks << " pairs verified" << std::endl;
}

// ----------------------------------------------------------------------------
// Test 10: CT scalar_mul matches fast scalar_mul
// ----------------------------------------------------------------------------
static void test_ct_consistency(unsigned N) {
    std::cout << "  [10] CT consistency: ct::scalar_mul vs fast::scalar_mul, k=1.." << N << std::endl;
    PT G = PT::generator();
    
    for (unsigned k = 1; k <= N; ++k) {
        SC s = SC::from_uint64(k);
        PT fast_result = G.scalar_mul(s);
        PT ct_result = secp256k1::ct::scalar_mul(G, s);
        CHECK(pt_eq(fast_result, ct_result),
              "CT mismatch at k=" + std::to_string(k));
    }
}

// ----------------------------------------------------------------------------
// Test 11: Negation properties
// ----------------------------------------------------------------------------
static void test_negation() {
    std::cout << "  [11] Negation properties" << std::endl;
    PT G = PT::generator();
    
    // -(-P) = P
    PT P = G;
    for (int i = 0; i < 16; ++i) {
        PT neg = P.negate();
        PT neg_neg = neg.negate();
        CHECK(pt_eq(P, neg_neg), "-(-P) != P at i=" + std::to_string(i));
        P = P.add(G);
    }
    
    // -(O) = O
    PT O = PT::infinity();
    CHECK(O.negate().is_infinity(), "-(O) != O");
    
    // k*G + (-k)*G = O
    for (unsigned k = 1; k <= 32; ++k) {
        SC s = SC::from_uint64(k);
        SC neg_s = s.negate();
        PT kG = G.scalar_mul(s);
        PT nkG = G.scalar_mul(neg_s);
        PT sum = kG.add(nkG);
        CHECK(sum.is_infinity(), "k*G + (-k)*G != O for k=" + std::to_string(k));
    }
}

// ----------------------------------------------------------------------------
// Test 12: In-place operations consistency
// ----------------------------------------------------------------------------
static void test_inplace() {
    std::cout << "  [12] In-place ops: next/prev/dbl_inplace vs immutable" << std::endl;
    PT G = PT::generator();
    
    // next_inplace vs add(G)
    PT p1 = G;
    PT p2 = G;
    for (int i = 0; i < 64; ++i) {
        PT immutable_next = p1.add(G);
        p2.next_inplace();
        CHECK(pt_eq(immutable_next, p2),
              "next_inplace mismatch at i=" + std::to_string(i));
        p1 = immutable_next;
    }
    
    // dbl_inplace vs dbl
    p1 = G;
    p2 = G;
    for (int i = 0; i < 32; ++i) {
        PT immutable_dbl = p1.dbl();
        p2 = p1;
        p2.dbl_inplace();
        CHECK(pt_eq(immutable_dbl, p2),
              "dbl_inplace mismatch at i=" + std::to_string(i));
        p1 = immutable_dbl;
    }
}

// ----------------------------------------------------------------------------
// Test 13: Pippenger MSM correctness
// ----------------------------------------------------------------------------
static void test_pippenger() {
    std::cout << "  [13] Pippenger MSM correctness" << std::endl;
    PT G = PT::generator();
    
    // Build n random-ish points and scalars
    const int n = 256;
    std::vector<SC> scalars(n);
    std::vector<PT> points(n);
    
    PT P = G;
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
        scalars[i] = SC::from_uint64(static_cast<uint64_t>(i * 31 + 17));
        points[i] = P;
        P = P.add(G);  // points[i] = (i+1)*G
    }
    
    // Naive computation
    PT naive = PT::infinity();
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
        naive = naive.add(points[i].scalar_mul(scalars[i]));
    }
    
    // Pippenger
    PT pip = secp256k1::pippenger_msm(scalars, points);
    CHECK(pt_eq(naive, pip), "Pippenger MSM != naive sum");
    
    // Unified MSM
    PT unified = secp256k1::msm(scalars, points);
    CHECK(pt_eq(naive, unified), "Unified MSM != naive sum");
    
    // Small n (should use Strauss)
    std::vector<SC> small_s(scalars.begin(), scalars.begin() + 8);
    std::vector<PT> small_p(points.begin(), points.begin() + 8);
    
    PT naive_small = PT::infinity();
    for (std::size_t i = 0; i < 8; ++i) {
        naive_small = naive_small.add(small_p[i].scalar_mul(small_s[i]));
    }
    PT msm_small = secp256k1::msm(small_s, small_p);
    CHECK(pt_eq(naive_small, msm_small), "MSM(n=8) != naive sum");
    
    // Edge: empty
    PT empty = secp256k1::msm(std::vector<SC>{}, std::vector<PT>{});
    CHECK(empty.is_infinity(), "MSM(empty) != infinity");
    
    // Edge: single
    SC s1 = SC::from_uint64(42);
    PT p1 = G;
    PT single = secp256k1::msm(std::vector<SC>{s1}, std::vector<PT>{p1});
    PT expected = G.scalar_mul(s1);
    CHECK(pt_eq(single, expected), "MSM(n=1) != scalar_mul");
}

// ----------------------------------------------------------------------------
// Test 14: Comb generator multiplication
// ----------------------------------------------------------------------------
static void test_comb_gen() {
    std::cout << "  [14] Comb generator: comb_mul(k) vs k*G" << std::endl;
    PT G = PT::generator();
    
    // Init with small teeth for fast test
    secp256k1::fast::CombGenContext ctx;
    ctx.init(6);  // teeth=6 -> 176KB table, fast init
    
    CHECK(ctx.ready(), "CombGenContext not ready after init");
    
    // Test small scalars
    for (unsigned k = 1; k <= 128; ++k) {
        SC s = SC::from_uint64(k);
        PT comb_result = ctx.mul(s);
        PT expected = G.scalar_mul(s);
        CHECK(pt_eq(comb_result, expected),
              "comb_mul(" + std::to_string(k) + ") != k*G");
    }
    
    // Test CT version for a subset
    for (unsigned k = 1; k <= 16; ++k) {
        SC s = SC::from_uint64(k);
        PT ct_result = ctx.mul_ct(s);
        PT expected = G.scalar_mul(s);
        CHECK(pt_eq(ct_result, expected),
              "comb_mul_ct(" + std::to_string(k) + ") != k*G");
    }
    
    // Test larger scalars
    SC large = SC::from_hex("DEADBEEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF01234567");
    PT comb_large = ctx.mul(large);
    PT expected_large = G.scalar_mul(large);
    CHECK(pt_eq(comb_large, expected_large), "comb_mul(large) != large*G");
    
    // Test global singleton
    secp256k1::fast::init_comb_gen(6);
    CHECK(secp256k1::fast::comb_gen_ready(), "Global comb gen not ready");
    
    SC s42 = SC::from_uint64(42);
    PT global_result = secp256k1::fast::comb_gen_mul(s42);
    PT expected42 = G.scalar_mul(s42);
    CHECK(pt_eq(global_result, expected42), "global comb_gen_mul(42) != 42*G");
}

// ----------------------------------------------------------------------------
// Test runner
// ----------------------------------------------------------------------------

#ifdef STANDALONE_TEST
int main() {
#else
int run_exhaustive_tests() {
#endif
    std::cout << "\n=== Exhaustive Algebraic Verification ===" << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    
    g_pass = 0;
    g_fail = 0;
    
    // Parameters (N controls exhaustive range)
    unsigned N = 256;       // k range for single-variable tests
    unsigned N_pair = 128;  // k range for pair tests
    unsigned N_ct = 64;     // k range for CT (slower)
    
    test_closure(N);
    test_additive_consistency(N);
    test_homomorphism(N_pair);
    test_scalar_mul_consistency(N);
    test_scalar_associativity();
    test_point_addition_axioms();
    test_doubling();
    test_order();
    test_scalar_arithmetic(N_pair);
    test_ct_consistency(N_ct);
    test_negation();
    test_inplace();
    test_pippenger();
    test_comb_gen();
    
    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    
    std::cout << "\n  -- Results: " << g_pass << " passed, " << g_fail << " failed"
              << " (" << ms << " ms)" << std::endl;
    
    if (g_fail > 0) {
        std::cerr << "\n  *** EXHAUSTIVE TESTS FAILED ***" << std::endl;
    } else {
        std::cout << "  All exhaustive tests PASSED [ok]" << std::endl;
    }
    
#ifdef STANDALONE_TEST
    return g_fail > 0 ? 1 : 0;
#else
    return g_fail;
#endif
}
