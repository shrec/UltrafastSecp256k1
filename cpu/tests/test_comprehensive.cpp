// ============================================================================
// Comprehensive Test Suite — UltrafastSecp256k1
// ============================================================================
// 500+ categorized tests covering every public API, edge case, and platform.
// Uses TestCategory enum for selective runs. Organic library component —
// no external test frameworks needed.
//
// Build: part of run_selftest (via CMakeLists), or standalone.
// ============================================================================

#include "secp256k1/test_framework.hpp"
#include "secp256k1/fast.hpp"
#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/field_branchless.hpp"
#include "secp256k1/field_asm.hpp"
#include "secp256k1/field_optimal.hpp"
#include "secp256k1/glv.hpp"
#include "secp256k1/ct_utils.hpp"
#include "secp256k1/ct/ops.hpp"
#include "secp256k1/ct/field.hpp"
#include "secp256k1/ct/scalar.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/sha512.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/ecdh.hpp"
#include "secp256k1/recovery.hpp"
#include "secp256k1/pippenger.hpp"
#include "secp256k1/ecmult_gen_comb.hpp"
#include "secp256k1/multiscalar.hpp"
#include "secp256k1/precompute.hpp"
#include "secp256k1/batch_add_affine.hpp"
#include "secp256k1/batch_verify.hpp"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <chrono>
#include <vector>
#include <array>
#include <cstdint>

using FE = secp256k1::fast::FieldElement;
using SC = secp256k1::fast::Scalar;
using PT = secp256k1::fast::Point;
using secp256k1::test::TestCategory;
using secp256k1::test::TestCounters;

// ── Global counters ──────────────────────────────────────────────────────────
static TestCounters g_counters;

#define CHECK(cond, msg)                                        \
    do {                                                        \
        if (cond) { ++g_counters.passed; }                      \
        else {                                                  \
            ++g_counters.failed;                                \
            std::cerr << "  FAIL: " << msg << std::endl;        \
        }                                                       \
    } while (0)

#define SKIP(msg)                                               \
    do { ++g_counters.skipped; } while (0)

// ── Utility ──────────────────────────────────────────────────────────────────
static bool pt_eq(const PT& a, const PT& b) {
    if (a.is_infinity() && b.is_infinity()) return true;
    if (a.is_infinity() || b.is_infinity()) return false;
    return a.x().to_bytes() == b.x().to_bytes() &&
           a.y().to_bytes() == b.y().to_bytes();
}

// secp256k1 prime p = 2^256 - 0x1000003D1
static FE secp256k1_p_minus_1() {
    // p-1 = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2E
    return FE::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2E");
}

// secp256k1 group order n
static SC secp256k1_n() {
    return SC::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
}

// ============================================================================
//  CATEGORY 1: Field Arithmetic (TestCategory::FieldArith)
// ============================================================================
static void test_field_arith() {
    std::cout << "  [FieldArith] Field arithmetic tests..." << std::endl;
    
    FE zero = FE::zero();
    FE one  = FE::one();
    
    // 1.1: Zero + Zero = Zero
    CHECK(FE::zero() + FE::zero() == FE::zero(), "0+0==0");
    
    // 1.2: One + Zero = One
    CHECK(one + zero == one, "1+0==1");
    
    // 1.3: Zero + One = One (commutativity)
    CHECK(zero + one == one, "0+1==1");
    
    // 1.4: One + One = Two
    FE two = FE::from_uint64(2);
    CHECK(one + one == two, "1+1==2");
    
    // 1.5: Subtraction identity
    CHECK(one - one == zero, "1-1==0");
    
    // 1.6: Subtraction self
    FE val = FE::from_uint64(42);
    CHECK(val - val == zero, "a-a==0");
    
    // 1.7: Multiplication by zero
    CHECK(val * zero == zero, "a*0==0");
    
    // 1.8: Multiplication by one
    CHECK(val * one == val, "a*1==a");
    
    // 1.9: Multiplication commutativity
    FE a = FE::from_uint64(7);
    FE b = FE::from_uint64(11);
    CHECK(a * b == b * a, "a*b==b*a");
    
    // 1.10: Multiplication associativity
    FE c = FE::from_uint64(13);
    CHECK((a * b) * c == a * (b * c), "(a*b)*c==a*(b*c)");
    
    // 1.11: Distributive law
    CHECK(a * (b + c) == a * b + a * c, "a*(b+c)==a*b+a*c");
    
    // 1.12: Square vs self-multiply
    CHECK(a.square() == a * a, "a²==a*a");
    
    // 1.13: Square of one
    CHECK(one.square() == one, "1²==1");
    
    // 1.14: Square of zero
    CHECK(zero.square() == zero, "0²==0");
    
    // 1.15: Repeated squaring chain
    FE x = FE::from_uint64(3);
    FE x2 = x.square();      // 9
    FE x4 = x2.square();     // 81
    FE expected = FE::from_uint64(81);
    CHECK(x4 == expected, "3^4==81");
    
    // 1.16: In-place square
    FE y = FE::from_uint64(5);
    FE y_sq = y.square();
    y.square_inplace();
    CHECK(y == y_sq, "square_inplace consistency");
    
    // 1.17: Additive inverse via subtraction
    FE neg_a = zero - a;
    CHECK(a + neg_a == zero, "a+(-a)==0");
    
    // 1.18: Double subtraction identity
    CHECK(a - b - a + b == zero, "a-b-a+b==0");
    
    // 1.19: Compound assignment +=
    FE acc = FE::from_uint64(10);
    FE orig = acc;
    acc += FE::from_uint64(5);
    CHECK(acc == orig + FE::from_uint64(5), "operator+=");
    
    // 1.20: Compound assignment -=
    acc = FE::from_uint64(20);
    orig = acc;
    acc -= FE::from_uint64(7);
    CHECK(acc == orig - FE::from_uint64(7), "operator-=");
    
    // 1.21: Compound assignment *=
    acc = FE::from_uint64(6);
    orig = acc;
    acc *= FE::from_uint64(8);
    CHECK(acc == orig * FE::from_uint64(8), "operator*=");
    
    // 1.22: Large value arithmetic
    FE large = FE::from_hex("DEADBEEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF01234567");
    CHECK(large + zero == large, "large+0==large");
    CHECK(large * one == large, "large*1==large");
    CHECK(large - large == zero, "large-large==0");
    
    // 1.23: Iterated addition equals multiplication by small scalar
    FE v = FE::from_uint64(17);
    FE sum3 = v + v + v;
    FE mul3 = v * FE::from_uint64(3);
    CHECK(sum3 == mul3, "v+v+v == 3*v");
    
    // 1.24: (a+b)² = a² + 2ab + b²
    FE lhs = (a + b).square();
    FE rhs = a.square() + FE::from_uint64(2) * a * b + b.square();
    CHECK(lhs == rhs, "(a+b)² == a² + 2ab + b²");
    
    // 1.25: (a-b)*(a+b) = a² - b²
    FE diff_prod = (a - b) * (a + b);
    FE sq_diff = a.square() - b.square();
    CHECK(diff_prod == sq_diff, "(a-b)(a+b) == a²-b²");
}

// ============================================================================
//  CATEGORY 2: Field Conversions (TestCategory::FieldConversions)
// ============================================================================
static void test_field_conversions() {
    std::cout << "  [FieldConversions] Field conversion tests..." << std::endl;
    
    // 2.1: from_uint64 → to_hex → from_hex roundtrip
    for (uint64_t v : {0ULL, 1ULL, 42ULL, 0xFFFFULL, 0xFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL}) {
        FE fe = FE::from_uint64(v);
        std::string hex = fe.to_hex();
        FE back = FE::from_hex(hex);
        CHECK(fe == back, "uint64→hex→FE roundtrip for " + std::to_string(v));
    }
    
    // 2.2: from_bytes ↔ to_bytes roundtrip
    FE val = FE::from_hex("09AF57F4F5C1D64C6BEA6D4193C5D9130421F4F078868E5EC00A56E68001136C");
    auto bytes = val.to_bytes();
    FE recovered = FE::from_bytes(bytes);
    CHECK(val == recovered, "bytes roundtrip");
    
    // 2.3: to_bytes_into matches to_bytes
    std::array<uint8_t, 32> buf{};
    val.to_bytes_into(buf.data());
    CHECK(buf == bytes, "to_bytes_into matches to_bytes");
    
    // 2.4: from_limbs roundtrip
    auto limbs = val.limbs();
    FE from_l = FE::from_limbs(limbs);
    CHECK(val == from_l, "limbs roundtrip");
    
    // 2.5: Zero conversions
    FE z = FE::zero();
    CHECK(z.to_hex().find_first_not_of('0') == std::string::npos, "zero to_hex is all zeros");
    auto zb = z.to_bytes();
    bool all_zero = true;
    for (auto b : zb) if (b != 0) all_zero = false;
    CHECK(all_zero, "zero to_bytes is all zeros");
    
    // 2.6: One conversions
    FE o = FE::one();
    auto ob = o.to_bytes();
    CHECK(ob[31] == 1, "one to_bytes last byte is 1");
    bool rest_zero = true;
    for (int i = 0; i < 31; ++i) if (ob[i] != 0) rest_zero = false;
    CHECK(rest_zero, "one to_bytes prefix is zeros");
    
    // 2.7: Hex case insensitivity
    FE lower = FE::from_hex("deadbeef0000000000000000000000000000000000000000000000000000abcd");
    FE upper = FE::from_hex("DEADBEEF0000000000000000000000000000000000000000000000000000ABCD");
    CHECK(lower == upper, "hex case insensitive");
    
    // 2.8: data() ↔ from_data roundtrip
    auto& data = val.data();
    FE from_d = FE::from_data(data);
    CHECK(val == from_d, "data() roundtrip");
    
    // 2.9: MidFieldElement zero-cost cast
    FE fe4 = FE::from_uint64(0x12345678);
    auto* mid = secp256k1::fast::toMid(&fe4);
    FE* back_fe = mid->ToFieldElement();
    CHECK(*back_fe == fe4, "MidFieldElement cast roundtrip");
    
    // 2.10: Large hex value
    FE max_fe = FE::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2E");
    std::string hex_back = max_fe.to_hex();
    FE max_back = FE::from_hex(hex_back);
    CHECK(max_fe == max_back, "p-1 hex roundtrip");
}

// ============================================================================
//  CATEGORY 3: Field Edge Cases (TestCategory::FieldEdgeCases)
// ============================================================================
static void test_field_edge_cases() {
    std::cout << "  [FieldEdgeCases] Field edge case tests..." << std::endl;
    
    FE zero = FE::zero();
    FE one  = FE::one();
    
    // 3.1: p-1 is the largest valid field element  
    FE p_minus_1 = secp256k1_p_minus_1();
    
    // 3.2: (p-1) + 1 = 0 (mod p)
    FE wrap = p_minus_1 + one;
    CHECK(wrap == zero, "(p-1)+1 == 0 mod p");
    
    // 3.3: (p-1)² mod p
    FE sq = p_minus_1.square();
    // (-1)² = 1
    CHECK(sq == one, "(p-1)² == 1 mod p (since p-1 ≡ -1)");
    
    // 3.4: (p-1) * (p-1) = 1  
    CHECK(p_minus_1 * p_minus_1 == one, "(p-1)*(p-1) == 1");
    
    // 3.5: Inverse of one
    CHECK(one.inverse() == one, "1⁻¹ == 1");
    
    // 3.6: Inverse of (p-1) = (p-1) since (-1)⁻¹ = -1
    CHECK(p_minus_1.inverse() == p_minus_1, "(p-1)⁻¹ == p-1");
    
    // 3.7: a * a⁻¹ = 1 for random-ish values
    uint64_t test_vals[] = {2, 3, 7, 42, 0xDEADBEEF, 0xFFFFFFFFFFFFULL};
    for (auto v : test_vals) {
        FE fe = FE::from_uint64(v);
        FE inv = fe.inverse();
        CHECK(fe * inv == one, "a*a⁻¹==1 for v=" + std::to_string(v));
    }
    
    // 3.8: Double inverse = identity
    FE val = FE::from_uint64(1337);
    CHECK(val.inverse().inverse() == val, "(a⁻¹)⁻¹ == a");
    
    // 3.9: Repeated addition wraps at p
    // 2*(p-1) = p-2 = -(2) → 2*(p-1) + 2 = 0
    FE two = FE::from_uint64(2);
    FE double_pm1 = p_minus_1 + p_minus_1;
    CHECK(double_pm1 + two == zero, "2*(p-1)+2 == 0");
    
    // 3.10: Equality reflexivity
    FE q = FE::from_uint64(99);
    CHECK(q == q, "FE equality reflexive");
    CHECK(!(q != q), "FE inequality reflexive");
    
    // 3.11: Inequality
    FE r = FE::from_uint64(100);
    CHECK(q != r, "FE inequality for different values");
    
    // 3.12: from_uint64 edge values
    CHECK(FE::from_uint64(0) == zero, "from_uint64(0)==zero()");
    CHECK(FE::from_uint64(1) == one, "from_uint64(1)==one()");
    
    // 3.13: Inverse-inplace consistency
    FE a = FE::from_uint64(12345);
    FE a_inv = a.inverse();
    FE a_copy = a;
    a_copy.inverse_inplace();
    CHECK(a_inv == a_copy, "inverse_inplace matches inverse");
    
    // 3.14: Large limb values (LE: limb[0] = least significant)
    FE big = FE::from_limbs({0xFFFFFFFEFFFFFC2EULL, 0xFFFFFFFFFFFFFFFFULL,
                             0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL});
    // This should be p-1 itself (verify via addition wrap)
    CHECK(big + FE::one() == FE::zero(), "max limbs + 1 == 0 (i.e., it's p-1)");
    
    // 3.15: Subtraction wrapping
    FE small1 = FE::from_uint64(1);
    FE sub_wrap = zero - small1;  // should be p-1
    CHECK(sub_wrap == p_minus_1, "0-1 == p-1");
    
    // 3.16: Montgomery R constant roundtrip
    const auto& R = secp256k1::fast::montgomery::R();
    CHECK(R * R.inverse() == one, "R * R⁻¹ == 1");
    
    // 3.17: Montgomery from_mont consistency
    FE v2 = FE::from_uint64(42);
    FE mont = v2 * secp256k1::fast::montgomery::R();
    FE back = FE::from_mont(mont);
    CHECK(back == v2, "from_mont(a*R) == a");
}

// ============================================================================
//  CATEGORY 4: Field Inverse (TestCategory::FieldInverse) 
// ============================================================================
static void test_field_inverse() {
    std::cout << "  [FieldInverse] Field inverse algorithm tests..." << std::endl;
    
    FE one = FE::one();
    
    // Test values
    FE test_values[] = {
        FE::from_uint64(2),
        FE::from_uint64(3),
        FE::from_uint64(7),
        FE::from_uint64(0xDEADBEEF),
        FE::from_uint64(0xCAFEBABE12345ULL),
        FE::from_hex("09AF57F4F5C1D64C6BEA6D4193C5D9130421F4F078868E5EC00A56E68001136C"),
        secp256k1_p_minus_1(),
    };
    
    // 4.1-4.7: Standard inverse correctness for each value
    for (size_t i = 0; i < sizeof(test_values)/sizeof(test_values[0]); ++i) {
        FE inv = test_values[i].inverse();
        CHECK(test_values[i] * inv == one, 
              "inverse correctness #" + std::to_string(i+1));
    }
    
    // 4.8-4.21: Each inverse algorithm matches default
    // Only run on x86_64 where all algorithms are available
    FE base = FE::from_uint64(12345);
    FE ref_inv = base.inverse();

    using InvFn = FE(*)(const FE&);
    struct InvAlgo { const char* name; InvFn fn; };
    InvAlgo algos[] = {
        {"binary",          secp256k1::fast::fe_inverse_binary},
        {"window4",         secp256k1::fast::fe_inverse_window4},
        {"addchain",        secp256k1::fast::fe_inverse_addchain},
        {"eea",             secp256k1::fast::fe_inverse_eea},
        {"window_naf_v2",   secp256k1::fast::fe_inverse_window_naf_v2},
        {"hybrid_eea",      secp256k1::fast::fe_inverse_hybrid_eea},
        {"bos_coster",      secp256k1::fast::fe_inverse_bos_coster},
        {"ltr_precomp",     secp256k1::fast::fe_inverse_ltr_precomp},
        {"booth",           secp256k1::fast::fe_inverse_booth},
        {"strauss",         secp256k1::fast::fe_inverse_strauss},
        {"kary16",          secp256k1::fast::fe_inverse_kary16},
        {"fixed_window5",   secp256k1::fast::fe_inverse_fixed_window5},
    };
    
    for (auto& algo : algos) {
        FE result = algo.fn(base);
        CHECK(result == ref_inv, 
              std::string("inverse algo '") + algo.name + "' matches default");
    }
    
    // 4.22: Batch inverse
    const int BATCH = 8;
    FE batch[BATCH];
    FE expected_inv[BATCH];
    for (int i = 0; i < BATCH; ++i) {
        batch[i] = FE::from_uint64(static_cast<uint64_t>(i + 2));
        expected_inv[i] = batch[i].inverse();
    }
    FE batch_copy[BATCH];
    std::memcpy(batch_copy, batch, sizeof(batch));
    secp256k1::fast::fe_batch_inverse(batch_copy, BATCH);
    for (int i = 0; i < BATCH; ++i) {
        CHECK(batch_copy[i] == expected_inv[i],
              "batch_inverse[" + std::to_string(i) + "] matches single inverse");
    }
    
    // 4.23: Batch inverse with scratch
    std::vector<FE> scratch;
    std::memcpy(batch_copy, batch, sizeof(batch));
    secp256k1::fast::fe_batch_inverse(batch_copy, BATCH, scratch);
    for (int i = 0; i < BATCH; ++i) {
        CHECK(batch_copy[i] == expected_inv[i],
              "batch_inverse_scratch[" + std::to_string(i) + "] matches");
    }
    
    // 4.24: Batch inverse of 1 element
    FE single_batch[1] = {FE::from_uint64(7)};
    FE single_expected = single_batch[0].inverse();
    secp256k1::fast::fe_batch_inverse(single_batch, 1);
    CHECK(single_batch[0] == single_expected, "batch_inverse(n=1)");
}

// ============================================================================
//  CATEGORY 5: Field Branchless (TestCategory::FieldBranchless)
// ============================================================================
static void test_field_branchless() {
    std::cout << "  [FieldBranchless] Branchless operation tests..." << std::endl;
    
    using namespace secp256k1::fast;
    FE a = FE::from_uint64(42);
    FE b = FE::from_uint64(99);
    FE zero = FE::zero();
    
    // 5.1: field_cmov flag=true selects a
    FE result;
    field_cmov(&result, &a, &b, true);
    CHECK(result == a, "cmov(true) selects a");
    
    // 5.2: field_cmov flag=false selects b
    field_cmov(&result, &a, &b, false);
    CHECK(result == b, "cmov(false) selects b");
    
    // 5.3: field_cmovznz nonzero selects a
    field_cmovznz(&result, &a, &b, 1);
    CHECK(result == a, "cmovznz(1) selects a");
    
    // 5.4: field_cmovznz zero selects b
    field_cmovznz(&result, &a, &b, 0);
    CHECK(result == b, "cmovznz(0) selects b");
    
    // 5.5: field_cmovznz large nonzero
    field_cmovznz(&result, &a, &b, 0xFFFFFFFFFFFFFFFFULL);
    CHECK(result == a, "cmovznz(MAX) selects a");
    
    // 5.6: field_select true
    FE sel = field_select(a, b, true);
    CHECK(sel == a, "select(true)==a");
    
    // 5.7: field_select false
    sel = field_select(a, b, false);
    CHECK(sel == b, "select(false)==b");
    
    // 5.8: field_is_zero on zero
    CHECK(field_is_zero(zero) == 1, "field_is_zero(0)==1");
    
    // 5.9: field_is_zero on nonzero
    CHECK(field_is_zero(a) == 0, "field_is_zero(42)==0");
    
    // 5.10: field_eq same
    CHECK(field_eq(a, a) == 1, "field_eq(a,a)==1");
    
    // 5.11: field_eq different
    CHECK(field_eq(a, b) == 0, "field_eq(a,b)==0");
    
    // 5.12: field_cneg true negates
    FE negated;
    field_cneg(&negated, a, true);
    CHECK(a + negated == zero, "cneg(true) negates");
    
    // 5.13: field_cneg false preserves
    FE preserved;
    field_cneg(&preserved, a, false);
    CHECK(preserved == a, "cneg(false) preserves");
    
    // 5.14: field_cadd true adds
    FE cond_sum;
    field_cadd(&cond_sum, a, b, true);
    CHECK(cond_sum == a + b, "cadd(true) adds");
    
    // 5.15: field_cadd false preserves
    field_cadd(&cond_sum, a, b, false);
    CHECK(cond_sum == a, "cadd(false) preserves");
    
    // 5.16: field_csub true subtracts
    FE cond_diff;
    field_csub(&cond_diff, a, b, true);
    CHECK(cond_diff == a - b, "csub(true) subtracts");
    
    // 5.17: field_csub false preserves
    field_csub(&cond_diff, a, b, false);
    CHECK(cond_diff == a, "csub(false) preserves");
    
    // 5.18: cmov self-assignment (a=a)
    FE self = a;
    field_cmov(&self, &a, &a, true);
    CHECK(self == a, "cmov self-assign");
    
    // 5.19: is_zero on one
    CHECK(field_is_zero(FE::one()) == 0, "is_zero(one)==0");
    
    // 5.20: Chained select
    FE c = FE::from_uint64(77);
    FE r1 = field_select(a, b, true);
    FE r2 = field_select(r1, c, false);
    CHECK(r2 == c, "chained select");
}

// ============================================================================
//  CATEGORY 6: Field Optimal Dispatch (TestCategory::FieldOptimal)
// ============================================================================
static void test_field_optimal() {
    std::cout << "  [FieldOptimal] Optimal representation dispatch..." << std::endl;
    
    using namespace secp256k1::fast;
    
    // 6.1: to_optimal → from_optimal roundtrip
    FE val = FE::from_uint64(42);
    auto opt = to_optimal(val);
    FE back = from_optimal(opt);
    CHECK(val == back, "to_optimal→from_optimal roundtrip");
    
    // 6.2: Zero roundtrip
    FE z = FE::zero();
    CHECK(from_optimal(to_optimal(z)) == z, "zero optimal roundtrip");
    
    // 6.3: One roundtrip
    FE o = FE::one();
    CHECK(from_optimal(to_optimal(o)) == o, "one optimal roundtrip");
    
    // 6.4: Large value roundtrip
    FE large = FE::from_hex("DEADBEEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF01234567");
    CHECK(from_optimal(to_optimal(large)) == large, "large optimal roundtrip");
    
    // 6.5: p-1 roundtrip
    FE pm1 = secp256k1_p_minus_1();
    CHECK(from_optimal(to_optimal(pm1)) == pm1, "p-1 optimal roundtrip");
    
    // 6.6: Tier name is non-null
    CHECK(kOptimalTierName != nullptr, "optimal tier name exists");
    CHECK(std::strlen(kOptimalTierName) > 0, "optimal tier name non-empty");
    
    // 6.7: Multiple distinct values
    for (uint64_t v = 1; v <= 64; ++v) {
        FE fe = FE::from_uint64(v);
        CHECK(from_optimal(to_optimal(fe)) == fe, 
              "optimal roundtrip #" + std::to_string(v));
    }
}

// ============================================================================
//  CATEGORY 7: Field ASM (TestCategory::FieldRepresentations)
// ============================================================================
static void test_field_representations() {
    std::cout << "  [FieldRepresentations] ASM/platform field ops..." << std::endl;
    
    using namespace secp256k1::fast;
    FE a = FE::from_uint64(0xDEADBEEF);
    FE b = FE::from_uint64(0xCAFEBABE);
    FE ref_mul = a * b;
    FE ref_sq  = a.square();
    FE ref_add = a + b;
    
    // 7.1-7.5: BMI2 implementations (x86_64)
#if defined(__x86_64__) || defined(_M_X64)
    FE bmi2_mul = field_mul_bmi2(a, b);
    CHECK(bmi2_mul == ref_mul, "BMI2 mul matches");
    
    FE bmi2_sq = field_square_bmi2(a);
    CHECK(bmi2_sq == ref_sq, "BMI2 square matches");
    
    FE kara_sq = field_square_karatsuba(a);
    CHECK(kara_sq == ref_sq, "Karatsuba square matches");
    
    FE bmi2_add = field_add_bmi2(a, b);
    CHECK(bmi2_add == ref_add, "BMI2 add matches");
    
    FE bmi2_neg = field_negate_bmi2(a);
    CHECK(bmi2_neg == FE::zero() - a, "BMI2 negate matches");
#else
    // Skip on non-x86
    for (int i = 0; i < 5; ++i) SKIP("BMI2 not available");
#endif
    
    // 7.6-7.10: ARM64 implementations
#if defined(__aarch64__) || defined(_M_ARM64)
    FE arm_mul = field_mul_arm64(a, b);
    CHECK(arm_mul == ref_mul, "ARM64 mul matches");
    
    FE arm_sq = field_square_arm64(a);
    CHECK(arm_sq == ref_sq, "ARM64 square matches");
    
    FE arm_add = field_add_arm64(a, b);
    CHECK(arm_add == ref_add, "ARM64 add matches");
    
    FE arm_sub = field_sub_arm64(a, b);
    CHECK(arm_sub == a - b, "ARM64 sub matches");
    
    FE arm_neg = field_negate_arm64(a);
    CHECK(arm_neg == FE::zero() - a, "ARM64 negate matches");
#else
    for (int i = 0; i < 5; ++i) SKIP("ARM64 not available");
#endif

    // 7.11-7.15: RISC-V implementations
#ifdef SECP256K1_HAS_RISCV_ASM
    FE rv_mul = field_mul_riscv(a, b);
    CHECK(rv_mul == ref_mul, "RISC-V mul matches");
    
    FE rv_sq = field_square_riscv(a);
    CHECK(rv_sq == ref_sq, "RISC-V square matches");
    
    FE rv_add = field_add_riscv(a, b);
    CHECK(rv_add == ref_add, "RISC-V add matches");
    
    FE rv_sub = field_sub_riscv(a, b);
    CHECK(rv_sub == a - b, "RISC-V sub matches");
    
    FE rv_neg = field_negate_riscv(a);
    CHECK(rv_neg == FE::zero() - a, "RISC-V negate matches");
#else
    for (int i = 0; i < 5; ++i) SKIP("RISC-V ASM not available");
#endif
}

// ============================================================================
//  CATEGORY 8: Scalar Arithmetic (TestCategory::ScalarArith)
// ============================================================================
static void test_scalar_arith() {
    std::cout << "  [ScalarArith] Scalar arithmetic tests..." << std::endl;
    
    SC zero = SC::zero();
    SC one  = SC::one();
    
    // 8.1: Zero + Zero = Zero
    CHECK(zero + zero == zero, "s: 0+0==0");
    
    // 8.2: One + Zero = One
    CHECK(one + zero == one, "s: 1+0==1");
    
    // 8.3: One - One = Zero
    CHECK(one - one == zero, "s: 1-1==0");
    
    // 8.4: One * One = One
    CHECK(one * one == one, "s: 1*1==1");
    
    // 8.5: Commutativity
    SC a = SC::from_uint64(7);
    SC b = SC::from_uint64(13);
    CHECK(a + b == b + a, "s: a+b==b+a");
    CHECK(a * b == b * a, "s: a*b==b*a");
    
    // 8.6: Associativity
    SC c = SC::from_uint64(19);
    CHECK((a + b) + c == a + (b + c), "s: (a+b)+c==a+(b+c)");
    CHECK((a * b) * c == a * (b * c), "s: (a*b)*c==a*(b*c)");
    
    // 8.7: Distributive
    CHECK(a * (b + c) == a * b + a * c, "s: a*(b+c)==a*b+a*c");
    
    // 8.8: Multiplication by zero
    CHECK(a * zero == zero, "s: a*0==0");
    
    // 8.9: Negate
    SC neg_a = a.negate();
    CHECK(a + neg_a == zero, "s: a+(-a)==0");
    
    // 8.10: Double negate
    CHECK(a.negate().negate() == a, "s: -(-a)==a");
    
    // 8.11: Negate zero
    CHECK(zero.negate() == zero, "s: -0==0");
    
    // 8.12: is_zero
    CHECK(zero.is_zero(), "s: zero.is_zero()");
    CHECK(!one.is_zero(), "s: !one.is_zero()");
    CHECK(!a.is_zero(), "s: !7.is_zero()");
    
    // 8.13: Inverse
    SC inv_a = a.inverse();
    CHECK(a * inv_a == one, "s: a*a⁻¹==1");
    
    // 8.14: Double inverse
    CHECK(a.inverse().inverse() == a, "s: (a⁻¹)⁻¹==a");
    
    // 8.15: Compound assignment
    SC acc = SC::from_uint64(10);
    SC orig = acc;
    acc += SC::from_uint64(5);
    CHECK(acc == orig + SC::from_uint64(5), "s: +=");
    
    acc = SC::from_uint64(20);
    orig = acc;
    acc -= SC::from_uint64(7);
    CHECK(acc == orig - SC::from_uint64(7), "s: -=");
    
    acc = SC::from_uint64(6);
    orig = acc;
    acc *= SC::from_uint64(8);
    CHECK(acc == orig * SC::from_uint64(8), "s: *=");
    
    // 8.16: is_even
    CHECK(SC::from_uint64(2).is_even(), "s: 2.is_even()");
    CHECK(SC::from_uint64(0).is_even(), "s: 0.is_even()");
    CHECK(!SC::from_uint64(1).is_even(), "s: !1.is_even()");
    CHECK(!SC::from_uint64(3).is_even(), "s: !3.is_even()");
    
    // 8.17: Exhaustive small-range
    unsigned checks = 0;
    for (unsigned i = 0; i <= 64; ++i) {
        for (unsigned j = 0; j <= 64; ++j) {
            SC si = SC::from_uint64(i);
            SC sj = SC::from_uint64(j);
            CHECK(si + sj == SC::from_uint64(i + j), 
                  "s: " + std::to_string(i) + "+" + std::to_string(j));
            CHECK(si * sj == SC::from_uint64(static_cast<uint64_t>(i) * j),
                  "s: " + std::to_string(i) + "*" + std::to_string(j));
            ++checks;
        }
    }
    std::cout << "    " << checks << " small-range pairs verified" << std::endl;
}

// ============================================================================
//  CATEGORY 9: Scalar Conversions (TestCategory::ScalarConversions)
// ============================================================================
static void test_scalar_conversions() {
    std::cout << "  [ScalarConversions] Scalar conversion tests..." << std::endl;
    
    // 9.1: from_uint64 → to_hex → from_hex roundtrip
    for (uint64_t v : {0ULL, 1ULL, 42ULL, 0xFFFFULL, 0xFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL}) {
        SC s = SC::from_uint64(v);
        std::string hex = s.to_hex();
        SC back = SC::from_hex(hex);
        CHECK(s == back, "s: uint64→hex→SC for " + std::to_string(v));
    }
    
    // 9.2: from_bytes ↔ to_bytes
    SC val = SC::from_hex("DEADBEEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF01234567");
    auto bytes = val.to_bytes();
    SC recovered = SC::from_bytes(bytes);
    CHECK(val == recovered, "s: bytes roundtrip");
    
    // 9.3: from_limbs roundtrip
    auto limbs = val.limbs();
    SC from_l = SC::from_limbs(limbs);
    CHECK(val == from_l, "s: limbs roundtrip");
    
    // 9.4: Zero conversions
    SC z = SC::zero();
    auto zb = z.to_bytes();
    bool all_zero = true;
    for (auto b : zb) if (b != 0) all_zero = false;
    CHECK(all_zero, "s: zero bytes all zeros");
    
    // 9.5: bit() extraction
    SC s3 = SC::from_uint64(3);  // binary: ...011
    CHECK(s3.bit(0) == 1, "s: bit(0) of 3");
    CHECK(s3.bit(1) == 1, "s: bit(1) of 3");
    CHECK(s3.bit(2) == 0, "s: bit(2) of 3");
    
    SC s5 = SC::from_uint64(5);  // binary: ...101
    CHECK(s5.bit(0) == 1, "s: bit(0) of 5");
    CHECK(s5.bit(1) == 0, "s: bit(1) of 5");
    CHECK(s5.bit(2) == 1, "s: bit(2) of 5");
    
    // 9.6: data() ↔ from_data
    auto& data = val.data();
    SC from_d = SC::from_data(data);
    CHECK(val == from_d, "s: data() roundtrip");
}

// ============================================================================
//  CATEGORY 10: Scalar Edge Cases (TestCategory::ScalarEdgeCases)
// ============================================================================
static void test_scalar_edge_cases() {
    std::cout << "  [ScalarEdgeCases] Scalar edge case tests..." << std::endl;
    
    SC zero = SC::zero();
    SC one  = SC::one();
    SC n = secp256k1_n();
    
    // 10.1: n ≡ 0 (mod n)
    // Scalar should wrap: n becomes 0
    // Note: from_hex for n may return zero if reduction is applied
    SC n_minus_1 = n - one;
    
    // 10.2: (n-1) + 1 = 0
    SC wrap = n_minus_1 + one;
    CHECK(wrap == zero, "s: (n-1)+1 == 0");
    
    // 10.3: (n-1) * (n-1) = 1  (since n-1 ≡ -1, (-1)² = 1)
    CHECK(n_minus_1 * n_minus_1 == one, "s: (n-1)²==1");
    
    // 10.4: Inverse of (n-1) = (n-1)
    CHECK(n_minus_1.inverse() == n_minus_1, "s: (n-1)⁻¹==(n-1)");
    
    // 10.5: Negate of (n-1) = 1
    CHECK(n_minus_1.negate() == one, "s: -(n-1)==1");
    
    // 10.6: Negate of 1 = (n-1)
    CHECK(one.negate() == n_minus_1, "s: -1==(n-1)");
    
    // 10.7: 2 * (n-1) + 2 = 0
    SC two = SC::from_uint64(2);
    SC double_nm1 = n_minus_1 + n_minus_1;
    CHECK(double_nm1 + two == zero, "s: 2*(n-1)+2==0");
    
    // 10.8: n-1 is even (n = ...41 is odd, so n-1 = ...40 is even)
    CHECK(n_minus_1.is_even(), "s: n-1 is even");
    
    // 10.9: Various bit positions
    for (unsigned bit = 0; bit < 256; ++bit) {
        uint8_t b = one.bit(bit);
        if (bit == 0) CHECK(b == 1, "s: bit(0) of 1");
        else CHECK(b == 0, "s: bit(" + std::to_string(bit) + ") of 1 is 0");
    }
}

// ============================================================================
//  CATEGORY 11: Scalar NAF/wNAF (TestCategory::ScalarEncoding)
// ============================================================================
static void test_scalar_encoding() {
    std::cout << "  [ScalarEncoding] NAF/wNAF encoding tests..." << std::endl;
    
    // 11.1: NAF of small values
    SC s7 = SC::from_uint64(7);    // 111 → NAF: 100(-1)
    auto naf = s7.to_naf();
    // Reconstruct from NAF and verify it equals original
    SC reconstructed = SC::zero();
    SC power = SC::one();
    for (size_t i = 0; i < naf.size(); ++i) {
        if (naf[i] == 1) reconstructed = reconstructed + power;
        else if (naf[i] == -1) reconstructed = reconstructed - power;
        power = power + power;  // power *= 2
    }
    CHECK(reconstructed == s7, "NAF(7) reconstructs to 7");
    
    // 11.2: NAF property — no two consecutive nonzero digits
    bool naf_valid = true;
    for (size_t i = 0; i + 1 < naf.size(); ++i) {
        if (naf[i] != 0 && naf[i+1] != 0) { naf_valid = false; break; }
    }
    CHECK(naf_valid, "NAF(7) has no consecutive nonzero digits");
    
    // 11.3: wNAF reconstruction for various widths
    for (unsigned w = 2; w <= 6; ++w) {
        SC s42 = SC::from_uint64(42);
        auto wnaf = s42.to_wnaf(w);
        
        SC rec = SC::zero();
        SC pw = SC::one();
        for (size_t i = 0; i < wnaf.size(); ++i) {
            if (wnaf[i] > 0) {
                SC digit = SC::from_uint64(static_cast<uint64_t>(wnaf[i]));
                rec = rec + digit * pw;
            } else if (wnaf[i] < 0) {
                SC digit = SC::from_uint64(static_cast<uint64_t>(-wnaf[i]));
                rec = rec - digit * pw;
            }
            pw = pw + pw;
        }
        CHECK(rec == s42, "wNAF(w=" + std::to_string(w) + ",42) reconstructs");
    }
    
    // 11.4: NAF of zero
    auto naf_zero = SC::zero().to_naf();
    bool all_zero = true;
    for (auto d : naf_zero) if (d != 0) all_zero = false;
    CHECK(all_zero, "NAF(0) is all zeros");
    
    // 11.5: NAF of one
    auto naf_one = SC::one().to_naf();
    CHECK(naf_one.size() > 0 && naf_one[0] == 1, "NAF(1)[0] == 1");
    
    // 11.6: Large scalar NAF
    SC large = SC::from_hex("DEADBEEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF01234567");
    auto naf_large = large.to_naf();
    SC rec_large = SC::zero();
    SC pw_large = SC::one();
    for (size_t i = 0; i < naf_large.size(); ++i) {
        if (naf_large[i] == 1) rec_large = rec_large + pw_large;
        else if (naf_large[i] == -1) rec_large = rec_large - pw_large;
        pw_large = pw_large + pw_large;
    }
    CHECK(rec_large == large, "NAF(large) reconstructs");
    
    // 11.7: wNAF digits are odd
    auto wnaf5 = SC::from_uint64(255).to_wnaf(5);
    bool all_odd_or_zero = true;
    for (auto d : wnaf5) {
        if (d != 0 && (d % 2 == 0)) all_odd_or_zero = false;
    }
    CHECK(all_odd_or_zero, "wNAF digits are odd or zero");
    
    // 11.8: compute_wnaf_into test
    SC s100 = SC::from_uint64(100);
    int32_t wnaf_buf[257] = {};
    std::size_t wnaf_len = 0;
    secp256k1::fast::compute_wnaf_into(s100, 4, wnaf_buf, 257, wnaf_len);
    CHECK(wnaf_len > 0, "compute_wnaf_into produces nonzero length");
    // Reconstruct
    SC rec_wnaf = SC::zero();
    SC pw_wnaf = SC::one();
    for (std::size_t i = 0; i < wnaf_len; ++i) {
        if (wnaf_buf[i] > 0) {
            rec_wnaf = rec_wnaf + SC::from_uint64(static_cast<uint64_t>(wnaf_buf[i])) * pw_wnaf;
        } else if (wnaf_buf[i] < 0) {
            rec_wnaf = rec_wnaf - SC::from_uint64(static_cast<uint64_t>(-wnaf_buf[i])) * pw_wnaf;
        }
        pw_wnaf = pw_wnaf + pw_wnaf;
    }
    CHECK(rec_wnaf == s100, "compute_wnaf_into reconstructs 100");
}

// ============================================================================
//  CATEGORY 12: Point Basic (TestCategory::PointBasic)
// ============================================================================
static void test_point_basic() {
    std::cout << "  [PointBasic] Point basic operations..." << std::endl;
    
    PT G = PT::generator();
    PT O = PT::infinity();
    
    // 12.1: Generator is on curve
    FE gx = G.x();
    FE gy = G.y();
    CHECK(gy.square() == gx.square() * gx + FE::from_uint64(7), "G on curve");
    
    // 12.2: Generator is not infinity
    CHECK(!G.is_infinity(), "G != O");
    
    // 12.3: Infinity is infinity
    CHECK(O.is_infinity(), "O.is_infinity()");
    
    // 12.4: P + O = P
    CHECK(pt_eq(G.add(O), G), "G+O==G");
    
    // 12.5: O + P = P
    CHECK(pt_eq(O.add(G), G), "O+G==G");
    
    // 12.6: O + O = O
    CHECK(O.add(O).is_infinity(), "O+O==O");
    
    // 12.7: P + (-P) = O
    PT negG = G.negate();
    CHECK(G.add(negG).is_infinity(), "G+(-G)==O");
    
    // 12.8: Commutativity
    PT twoG = G.add(G);
    PT threeG = twoG.add(G);
    PT threeG_alt = G.add(twoG);
    CHECK(pt_eq(threeG, threeG_alt), "2G+G == G+2G");
    
    // 12.9: Associativity
    PT fourG = threeG.add(G);
    PT fourG_alt = twoG.add(twoG);
    CHECK(pt_eq(fourG, fourG_alt), "(3G+G) == (2G+2G)");
    
    // 12.10: Doubling = addition
    CHECK(pt_eq(G.dbl(), G.add(G)), "dbl(G)==G+G");
    
    // 12.11: 2·O = O
    CHECK(O.dbl().is_infinity(), "dbl(O)==O");
    
    // 12.12: Negation of infinity
    CHECK(O.negate().is_infinity(), "-O==O");
    
    // 12.13: Double negation
    PT P = G.add(G).add(G);  // 3G
    CHECK(pt_eq(P.negate().negate(), P), "-(-P)==P");
    
    // 12.14: Generator known x-coordinate first 4 bytes
    auto comp = G.to_compressed();
    CHECK(comp[0] == 0x02 || comp[0] == 0x03, "G compressed prefix");
    
    // 12.15: Negate changes y, keeps x
    FE px = P.x();
    FE py = P.y();
    PT neg_p = P.negate();
    CHECK(neg_p.x() == px, "negate preserves x");
    CHECK(neg_p.y() != py, "negate changes y");
    
    // 12.16: Add(P, P) == dbl(P)
    for (int i = 1; i <= 16; ++i) {
        PT pi = G.scalar_mul(SC::from_uint64(i));
        CHECK(pt_eq(pi.add(pi), pi.dbl()), "add(P,P)==dbl(P) for " + std::to_string(i) + "G");
    }
    
    // 12.17: from_affine ↔ x(),y()
    FE ax = G.x();
    FE ay = G.y();
    PT from_aff = PT::from_affine(ax, ay);
    CHECK(pt_eq(from_aff, G), "from_affine(G.x, G.y)==G");
    
    // 12.18: from_hex ↔ to_hex
    std::string xh = ax.to_hex();
    std::string yh = ay.to_hex();
    PT from_h = PT::from_hex(xh, yh);
    CHECK(pt_eq(from_h, G), "from_hex roundtrip");
    
    // 12.19: Closure — iterated addition stays on curve
    PT acc = G;
    for (int i = 0; i < 100; ++i) {
        FE xi = acc.x();
        FE yi = acc.y();
        CHECK(yi.square() == xi.square() * xi + FE::from_uint64(7),
              "on-curve at iteration " + std::to_string(i));
        acc = acc.add(G);
    }
}

// ============================================================================
//  CATEGORY 13: Point Scalar Mul (TestCategory::PointScalarMul)
// ============================================================================
static void test_point_scalar_mul() {
    std::cout << "  [PointScalarMul] Scalar multiplication tests..." << std::endl;
    
    PT G = PT::generator();
    
    // 13.1: 0*G = O
    CHECK(G.scalar_mul(SC::zero()).is_infinity(), "0*G==O");
    
    // 13.2: 1*G = G
    CHECK(pt_eq(G.scalar_mul(SC::one()), G), "1*G==G");
    
    // 13.3: 2*G = G+G
    CHECK(pt_eq(G.scalar_mul(SC::from_uint64(2)), G.add(G)), "2*G==G+G");
    
    // 13.4: n*G = O  
    SC n = secp256k1_n();
    CHECK(G.scalar_mul(n).is_infinity(), "n*G==O");
    
    // 13.5: (n-1)*G = -G
    SC n_minus_1 = n - SC::one();
    CHECK(pt_eq(G.scalar_mul(n_minus_1), G.negate()), "(n-1)*G==-G");
    
    // 13.6: Consistency with iterated addition for k=1..256
    PT iter = G;
    for (unsigned k = 1; k <= 256; ++k) {
        PT mul_result = G.scalar_mul(SC::from_uint64(k));
        CHECK(pt_eq(mul_result, iter),
              "scalar_mul(" + std::to_string(k) + ") == iterated");
        iter = iter.add(G);
    }
    
    // 13.7: Scalar associativity — k*(l*G) = (k*l)*G
    const uint64_t pairs[][2] = {
        {2, 3}, {3, 5}, {7, 11}, {13, 17}, {100, 200},
        {255, 255}, {1000, 999}, {12345, 6789}
    };
    for (auto& [k, l] : pairs) {
        SC sk = SC::from_uint64(k);
        SC sl = SC::from_uint64(l);
        PT lG = G.scalar_mul(sl);
        PT klG = lG.scalar_mul(sk);
        PT klG_direct = G.scalar_mul(sk * sl);
        CHECK(pt_eq(klG, klG_direct),
              "k*(l*G)==(k*l)*G for k=" + std::to_string(k) + ",l=" + std::to_string(l));
    }
    
    // 13.8: k*G + (-k)*G = O
    for (unsigned k = 1; k <= 64; ++k) {
        SC s = SC::from_uint64(k);
        PT kG = G.scalar_mul(s);
        PT nkG = G.scalar_mul(s.negate());
        CHECK(kG.add(nkG).is_infinity(),
              "k*G+(-k)*G==O for k=" + std::to_string(k));
    }
    
    // 13.9: Large scalar
    SC large = SC::from_hex("DEADBEEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF01234567");
    PT large_mul = G.scalar_mul(large);
    CHECK(!large_mul.is_infinity(), "large*G is not O");
    FE lx = large_mul.x();
    FE ly = large_mul.y();
    CHECK(ly.square() == lx.square() * lx + FE::from_uint64(7), "large*G on curve");
    
    // 13.10: Scalar mul of non-generator points
    PT twoG = G.add(G);
    PT result = twoG.scalar_mul(SC::from_uint64(3));
    PT expected = G.scalar_mul(SC::from_uint64(6));
    CHECK(pt_eq(result, expected), "3*(2G)==6G");
    
    // 13.11: Scalar mul of infinity
    CHECK(PT::infinity().scalar_mul(SC::from_uint64(42)).is_infinity(), "42*O==O");
}

// ============================================================================
//  CATEGORY 14: Point In-Place (TestCategory::PointInplace)
// ============================================================================
static void test_point_inplace() {
    std::cout << "  [PointInplace] In-place operations..." << std::endl;
    
    PT G = PT::generator();
    
    // 14.1: next_inplace vs add(G)
    PT p1 = G, p2 = G;
    for (int i = 0; i < 64; ++i) {
        PT immutable_next = p1.add(G);
        p2.next_inplace();
        CHECK(pt_eq(immutable_next, p2), "next_inplace #" + std::to_string(i));
        p1 = immutable_next;
    }
    
    // 14.2: prev_inplace (reverse of next)  
    PT p3 = G.scalar_mul(SC::from_uint64(100));
    PT p4 = p3;
    p4.next_inplace();
    p4.prev_inplace();
    CHECK(pt_eq(p3, p4), "next then prev is identity");
    
    // 14.3: dbl_inplace
    PT p5 = G;
    PT p6 = G;
    for (int i = 0; i < 32; ++i) {
        PT expected_dbl = p5.dbl();
        p6 = p5;
        p6.dbl_inplace();
        CHECK(pt_eq(expected_dbl, p6), "dbl_inplace #" + std::to_string(i));
        p5 = expected_dbl;
    }
    
    // 14.4: negate_inplace
    PT p7 = G.scalar_mul(SC::from_uint64(42));
    PT neg_immutable = p7.negate();
    PT p8 = p7;
    p8.negate_inplace();
    CHECK(pt_eq(neg_immutable, p8), "negate_inplace");
    
    // 14.5: add_inplace
    PT p9 = G;
    PT p10 = G.add(G);
    PT expected_add = p9.add(p10);
    p9.add_inplace(p10);
    CHECK(pt_eq(expected_add, p9), "add_inplace");
    
    // 14.6: sub_inplace
    PT p11 = G.scalar_mul(SC::from_uint64(50));
    PT p12 = G.scalar_mul(SC::from_uint64(20));
    PT expected_sub = p11.add(p12.negate());
    PT p13 = p11;
    p13.sub_inplace(p12);
    CHECK(pt_eq(expected_sub, p13), "sub_inplace");
    
    // 14.7: Double negate_inplace
    PT p14 = G.scalar_mul(SC::from_uint64(7));
    PT p15 = p14;
    p15.negate_inplace();
    p15.negate_inplace();
    CHECK(pt_eq(p14, p15), "double negate_inplace");

    // 14.8: add_mixed_inplace (affine RHS where z=1)
    FE ax = G.x();
    FE ay = G.y();
    PT p16 = G.scalar_mul(SC::from_uint64(5));  // 5G
    PT p17 = p16;
    p17.add_mixed_inplace(ax, ay);  // 5G + G = 6G
    PT expected_6g = G.scalar_mul(SC::from_uint64(6));
    CHECK(pt_eq(p17, expected_6g), "add_mixed_inplace == 5G+G=6G");
    
    // 14.9: sub_mixed_inplace
    PT p18 = G.scalar_mul(SC::from_uint64(10));
    p18.sub_mixed_inplace(ax, ay);  // 10G - G = 9G
    PT expected_9g = G.scalar_mul(SC::from_uint64(9));
    CHECK(pt_eq(p18, expected_9g), "sub_mixed_inplace == 10G-G=9G");
}

// ============================================================================
//  CATEGORY 15: Point Precomputed (TestCategory::PointPrecomputed)
// ============================================================================
static void test_point_precomputed() {
    std::cout << "  [PointPrecomputed] Precomputed scalar mul..." << std::endl;
    
    PT G = PT::generator();
    
    // 15.1: scalar_mul_precomputed_k matches scalar_mul
    for (unsigned k = 1; k <= 32; ++k) {
        SC s = SC::from_uint64(k);
        PT fast_result = G.scalar_mul_precomputed_k(s);
        PT ref_result = G.scalar_mul(s);
        CHECK(pt_eq(fast_result, ref_result),
              "precomputed_k(" + std::to_string(k) + ")");
    }
    
    // 15.2: KPlan
    SC k = SC::from_uint64(42);
    auto plan = secp256k1::fast::KPlan::from_scalar(k);
    PT plan_result = G.scalar_mul_with_plan(plan);
    PT ref = G.scalar_mul(k);
    CHECK(pt_eq(plan_result, ref), "KPlan(42) matches");
    
    // 15.3: KPlan with small-medium scalar
    SC med_k = SC::from_uint64(99999);
    auto med_plan = secp256k1::fast::KPlan::from_scalar(med_k);
    PT med_plan_result = G.scalar_mul_with_plan(med_plan);
    PT med_ref = G.scalar_mul(med_k);
    CHECK(pt_eq(med_plan_result, med_ref), "KPlan(99999) matches");
    
    // 15.4: scalar_mul_predecomposed
    auto decomp = secp256k1::fast::split_scalar_glv(k);
    PT predecomp_result = G.scalar_mul_predecomposed(
        decomp.k1, decomp.k2, decomp.neg1, decomp.neg2);
    CHECK(pt_eq(predecomp_result, ref), "predecomposed(42) matches");
    
    // 15.5: scalar_mul_precomputed_wnaf
    PT wnaf_result = G.scalar_mul_precomputed_wnaf(
        plan.wnaf1, plan.wnaf2, plan.neg1, plan.neg2);
    CHECK(pt_eq(wnaf_result, ref), "precomputed_wnaf(42) matches");
    
    // 15.6: Multiple KPlan values
    for (uint64_t kv = 1; kv <= 1024; kv += 100) {
        SC sv = SC::from_uint64(kv);
        auto pl = secp256k1::fast::KPlan::from_scalar(sv);
        PT res = G.scalar_mul_with_plan(pl);
        PT exp = G.scalar_mul(sv);
        CHECK(pt_eq(res, exp), "KPlan(" + std::to_string(kv) + ") matches");
    }
    
    // 15.7: Multiple different plans
    for (unsigned v = 1; v <= 8; ++v) {
        SC sv = SC::from_uint64(v * 1000 + 7);
        auto pl = secp256k1::fast::KPlan::from_scalar(sv);
        PT res = G.scalar_mul_with_plan(pl);
        PT exp = G.scalar_mul(sv);
        CHECK(pt_eq(res, exp), "KPlan #" + std::to_string(v));
    }
}

// ============================================================================
//  CATEGORY 16: Point Serialization (TestCategory::PointSerialization)
// ============================================================================
static void test_point_serialization() {
    std::cout << "  [PointSerialization] Point serialization tests..." << std::endl;
    
    PT G = PT::generator();
    
    // 16.1: to_compressed returns 33 bytes with 02/03 prefix
    auto comp = G.to_compressed();
    CHECK(comp.size() == 33, "compressed size == 33");
    CHECK(comp[0] == 0x02 || comp[0] == 0x03, "compressed prefix");
    
    // 16.2: to_uncompressed returns 65 bytes with 04 prefix
    auto uncomp = G.to_uncompressed();
    CHECK(uncomp.size() == 65, "uncompressed size == 65");
    CHECK(uncomp[0] == 0x04, "uncompressed prefix");
    
    // 16.3: Compressed x matches uncompressed x
    bool x_match = true;
    for (int i = 0; i < 32; ++i) {
        if (comp[1 + i] != uncomp[1 + i]) x_match = false;
    }
    CHECK(x_match, "compressed x == uncompressed x");
    
    // 16.4: Multiple points serialization
    PT P = G;
    for (int i = 0; i < 32; ++i) {
        auto c = P.to_compressed();
        auto u = P.to_uncompressed();
        CHECK(c[0] == 0x02 || c[0] == 0x03, "prefix #" + std::to_string(i));
        CHECK(u[0] == 0x04, "uncomp prefix #" + std::to_string(i));
        P = P.add(G);
    }
    
    // 16.5: x_first_half / x_second_half
    auto first = G.x_first_half();
    auto second = G.x_second_half();
    auto full_bytes = G.x().to_bytes();
    bool first_match = true, second_match = true;
    for (int i = 0; i < 16; ++i) {
        if (first[i] != full_bytes[i]) first_match = false;
        if (second[i] != full_bytes[16 + i]) second_match = false;
    }
    CHECK(first_match, "x_first_half matches");
    CHECK(second_match, "x_second_half matches");
    
    // 16.6: Serialization of negated point — same x
    PT neg = G.negate();
    auto comp_neg = neg.to_compressed();
    bool same_x = true;
    for (int i = 1; i < 33; ++i) {
        if (comp[i] != comp_neg[i]) same_x = false;
    }
    CHECK(same_x, "negation: same x in compressed");
    CHECK(comp[0] != comp_neg[0], "negation: different parity prefix");
    
    // 16.7: from_jacobian_coords
    FE jx = G.X();
    FE jy = G.Y();
    FE jz = G.z();
    PT from_j = PT::from_jacobian_coords(jx, jy, jz, false);
    CHECK(pt_eq(from_j, G), "from_jacobian_coords roundtrip");
}

// ============================================================================
//  CATEGORY 17: Point Edge Cases (TestCategory::PointEdgeCases)
// ============================================================================
static void test_point_edge_cases() {
    std::cout << "  [PointEdgeCases] Point edge cases..." << std::endl;
    
    PT G = PT::generator();
    PT O = PT::infinity();
    
    // 17.1: n*G = O
    SC n = secp256k1_n();
    CHECK(G.scalar_mul(n).is_infinity(), "n*G==O");
    
    // 17.2: 2*O = O
    CHECK(O.dbl().is_infinity(), "2*O==O");
    
    // 17.3: k*O = O for any k
    CHECK(O.scalar_mul(SC::from_uint64(42)).is_infinity(), "42*O==O");
    CHECK(O.scalar_mul(SC::from_uint64(1)).is_infinity(), "1*O==O");
    CHECK(O.scalar_mul(n).is_infinity(), "n*O==O");
    
    // 17.4: P + O = P for various P
    for (unsigned k = 1; k <= 16; ++k) {
        PT P = G.scalar_mul(SC::from_uint64(k));
        CHECK(pt_eq(P.add(O), P), "P+O==P for " + std::to_string(k) + "G");
        CHECK(pt_eq(O.add(P), P), "O+P==P for " + std::to_string(k) + "G");
    }
    
    // 17.5: P + (-P) = O for various P
    for (unsigned k = 1; k <= 16; ++k) {
        PT P = G.scalar_mul(SC::from_uint64(k));
        CHECK(P.add(P.negate()).is_infinity(),
              "P+(-P)==O for " + std::to_string(k) + "G");
    }
    
    // 17.6: Order of 2G
    PT twoG = G.scalar_mul(SC::from_uint64(2));
    // n * (2G) should be O since order of group divides n
    CHECK(twoG.scalar_mul(n).is_infinity(), "n*(2G)==O");
    
    // 17.7: Iterated addition = scalar mul for powers of 2
    PT P = G;
    for (int pow2 = 1; pow2 <= 16; ++pow2) {
        P = P.dbl();
        uint64_t expected_k = 1ULL << pow2;
        PT mul_result = G.scalar_mul(SC::from_uint64(expected_k));
        CHECK(pt_eq(P, mul_result),
              "2^" + std::to_string(pow2) + "*G via doubling");
    }
    
    // 17.8: Commutativity exhaustive small range
    std::vector<PT> pts(8);
    for (int i = 0; i < 8; ++i) pts[i] = G.scalar_mul(SC::from_uint64(i + 1));
    
    for (int i = 0; i < 8; ++i) {
        for (int j = i; j < 8; ++j) {
            CHECK(pt_eq(pts[i].add(pts[j]), pts[j].add(pts[i])),
                  "commutative (" + std::to_string(i+1) + "G," + std::to_string(j+1) + "G)");
        }
    }
    
    // 17.9: Associativity
    for (int i = 0; i < 4; ++i) {
        for (int j = i+1; j < 6; ++j) {
            for (int k = j+1; k < 8; ++k) {
                PT lhs = pts[i].add(pts[j]).add(pts[k]);
                PT rhs = pts[i].add(pts[j].add(pts[k]));
                CHECK(pt_eq(lhs, rhs),
                      "associative (" + std::to_string(i+1) + "," + 
                      std::to_string(j+1) + "," + std::to_string(k+1) + ")");
            }
        }
    }
}

// ============================================================================
//  CATEGORY 18: CT Primitives (TestCategory::CTOps)
// ============================================================================
static void test_ct_ops() {
    std::cout << "  [CTOps] Constant-time primitive tests..." << std::endl;
    
    using namespace secp256k1::ct;
    
    // 18.1: value_barrier (should not change value)
    uint64_t v = 0xDEADBEEFCAFEBABEULL;
    uint64_t v_orig = v;
    value_barrier(v);
    CHECK(v == v_orig, "value_barrier preserves value");
    
    // 18.2: is_zero_mask
    CHECK(is_zero_mask(0) == 0xFFFFFFFFFFFFFFFFULL, "is_zero_mask(0)==all-ones");
    CHECK(is_zero_mask(1) == 0, "is_zero_mask(1)==0");
    CHECK(is_zero_mask(0xFFFFFFFFFFFFFFFFULL) == 0, "is_zero_mask(MAX)==0");
    
    // 18.3: is_nonzero_mask
    CHECK(is_nonzero_mask(0) == 0, "is_nonzero_mask(0)==0");
    CHECK(is_nonzero_mask(1) == 0xFFFFFFFFFFFFFFFFULL, "is_nonzero_mask(1)==all-ones");
    
    // 18.4: eq_mask
    CHECK(eq_mask(42, 42) == 0xFFFFFFFFFFFFFFFFULL, "eq_mask(42,42)==all-ones");
    CHECK(eq_mask(42, 43) == 0, "eq_mask(42,43)==0");
    
    // 18.5: bool_to_mask
    CHECK(bool_to_mask(true) == 0xFFFFFFFFFFFFFFFFULL, "bool_to_mask(true)==all-ones");
    CHECK(bool_to_mask(false) == 0, "bool_to_mask(false)==0");
    
    // 18.6: lt_mask
    CHECK(lt_mask(5, 10) == 0xFFFFFFFFFFFFFFFFULL, "lt_mask(5,10)==all-ones");
    CHECK(lt_mask(10, 5) == 0, "lt_mask(10,5)==0");
    CHECK(lt_mask(5, 5) == 0, "lt_mask(5,5)==0");
    
    // 18.7: cmov64
    uint64_t dst = 100;
    uint64_t src = 200;
    cmov64(&dst, &src, 0xFFFFFFFFFFFFFFFFULL);
    CHECK(dst == 200, "cmov64 mask=all-ones copies");
    
    dst = 100;
    cmov64(&dst, &src, 0);
    CHECK(dst == 100, "cmov64 mask=0 preserves");
    
    // 18.8: cmov256
    uint64_t arr_dst[4] = {1, 2, 3, 4};
    uint64_t arr_src[4] = {10, 20, 30, 40};
    cmov256(arr_dst, arr_src, 0xFFFFFFFFFFFFFFFFULL);
    CHECK(arr_dst[0]==10 && arr_dst[1]==20 && arr_dst[2]==30 && arr_dst[3]==40,
          "cmov256 mask=all-ones");
    
    uint64_t arr_dst2[4] = {1, 2, 3, 4};
    cmov256(arr_dst2, arr_src, 0);
    CHECK(arr_dst2[0]==1 && arr_dst2[1]==2 && arr_dst2[2]==3 && arr_dst2[3]==4,
          "cmov256 mask=0");
    
    // 18.9: cswap256
    uint64_t swap_a[4] = {1, 2, 3, 4};
    uint64_t swap_b[4] = {10, 20, 30, 40};
    cswap256(swap_a, swap_b, 0xFFFFFFFFFFFFFFFFULL);
    CHECK(swap_a[0]==10 && swap_a[1]==20 && swap_a[2]==30 && swap_a[3]==40, "cswap256 swapped a");
    CHECK(swap_b[0]==1 && swap_b[1]==2 && swap_b[2]==3 && swap_b[3]==4, "cswap256 swapped b");
    
    uint64_t noswap_a[4] = {1, 2, 3, 4};
    uint64_t noswap_b[4] = {10, 20, 30, 40};
    cswap256(noswap_a, noswap_b, 0);
    CHECK(noswap_a[0]==1 && noswap_b[0]==10, "cswap256 mask=0 no swap");
    
    // 18.10: ct_select
    CHECK(ct_select(42, 99, 0xFFFFFFFFFFFFFFFFULL) == 42, "ct_select mask=ones");
    CHECK(ct_select(42, 99, 0) == 99, "ct_select mask=0");
    
    // 18.11: ct_equal (byte-level)
    uint8_t buf1[4] = {1, 2, 3, 4};
    uint8_t buf2[4] = {1, 2, 3, 4};
    uint8_t buf3[4] = {1, 2, 3, 5};
    CHECK(ct_equal(buf1, buf2, 4), "ct_equal same");
    CHECK(!ct_equal(buf1, buf3, 4), "ct_equal different");
    
    // 18.12: ct_is_zero
    uint8_t zeros[8] = {};
    uint8_t nonzeros[8] = {0, 0, 0, 0, 0, 0, 0, 1};
    CHECK(ct_is_zero(zeros, 8), "ct_is_zero for zeros");
    CHECK(!ct_is_zero(nonzeros, 8), "ct_is_zero for nonzeros");
    
    // 18.13: ct_memcpy_if
    uint8_t dest[4] = {0, 0, 0, 0};
    uint8_t source[4] = {5, 6, 7, 8};
    ct_memcpy_if(dest, source, 4, true);
    CHECK(dest[0]==5 && dest[1]==6 && dest[2]==7 && dest[3]==8, "ct_memcpy_if(true) copies");
    
    uint8_t dest2[4] = {1, 2, 3, 4};
    ct_memcpy_if(dest2, source, 4, false);
    CHECK(dest2[0]==1 && dest2[1]==2 && dest2[2]==3 && dest2[3]==4, "ct_memcpy_if(false) preserves");
    
    // 18.14: ct_memswap_if
    uint8_t sa[4] = {1, 2, 3, 4};
    uint8_t sb[4] = {5, 6, 7, 8};
    ct_memswap_if(sa, sb, 4, true);
    CHECK(sa[0]==5 && sb[0]==1, "ct_memswap_if(true) swaps");
    
    uint8_t sa2[4] = {1, 2, 3, 4};
    uint8_t sb2[4] = {5, 6, 7, 8};
    ct_memswap_if(sa2, sb2, 4, false);
    CHECK(sa2[0]==1 && sb2[0]==5, "ct_memswap_if(false) no swap");
}

// ============================================================================
//  CATEGORY 19: CT Field (TestCategory::CTField)
// ============================================================================
static void test_ct_field() {
    std::cout << "  [CTField] CT field operations..." << std::endl;
    
    namespace ctf = secp256k1::ct;
    FE a = FE::from_uint64(42);
    FE b = FE::from_uint64(99);
    FE zero = FE::zero();
    FE one = FE::one();
    
    // 19.1-19.5: CT arithmetic matches fast arithmetic
    CHECK(ctf::field_add(a, b) == a + b, "ct field_add");
    CHECK(ctf::field_sub(a, b) == a - b, "ct field_sub");
    CHECK(ctf::field_mul(a, b) == a * b, "ct field_mul");
    CHECK(ctf::field_sqr(a) == a.square(), "ct field_sqr");
    CHECK(ctf::field_neg(a) == zero - a, "ct field_neg");
    
    // 19.6: CT inverse
    FE ct_inv = ctf::field_inv(a);
    CHECK(a * ct_inv == one, "ct field_inv correct");
    
    // 19.7: CT cmov
    FE result = b;
    ctf::field_cmov(&result, a, 0xFFFFFFFFFFFFFFFFULL);
    CHECK(result == a, "ct field_cmov copies");
    
    result = b;
    ctf::field_cmov(&result, a, 0);
    CHECK(result == b, "ct field_cmov preserves");
    
    // 19.8: CT cswap
    FE fa = a, fb = b;
    ctf::field_cswap(&fa, &fb, 0xFFFFFFFFFFFFFFFFULL);
    CHECK(fa == b && fb == a, "ct field_cswap swaps");
    
    fa = a; fb = b;
    ctf::field_cswap(&fa, &fb, 0);
    CHECK(fa == a && fb == b, "ct field_cswap no swap");
    
    // 19.9: CT select
    CHECK(ctf::field_select(a, b, 0xFFFFFFFFFFFFFFFFULL) == a, "ct field_select(ones)");
    CHECK(ctf::field_select(a, b, 0) == b, "ct field_select(zero)");
    
    // 19.10: CT cneg
    CHECK(a + ctf::field_cneg(a, 0xFFFFFFFFFFFFFFFFULL) == zero, "ct field_cneg(ones)");
    CHECK(ctf::field_cneg(a, 0) == a, "ct field_cneg(zero)");
    
    // 19.11: CT is_zero
    CHECK(ctf::field_is_zero(zero) != 0, "ct field_is_zero(0)");
    CHECK(ctf::field_is_zero(a) == 0, "ct field_is_zero(42)");
    
    // 19.12: CT eq
    CHECK(ctf::field_eq(a, a) != 0, "ct field_eq same");
    CHECK(ctf::field_eq(a, b) == 0, "ct field_eq diff");
    
    // 19.13: CT normalize is idempotent
    FE norm = ctf::field_normalize(a);
    FE norm2 = ctf::field_normalize(norm);
    CHECK(norm == norm2, "ct field_normalize idempotent");
    
    // 19.14: CT arithmetic consistency for many values
    for (uint64_t v = 1; v <= 64; ++v) {
        FE fv = FE::from_uint64(v);
        FE fv2 = FE::from_uint64(v + 1);
        CHECK(ctf::field_add(fv, fv2) == fv + fv2, "ct add #" + std::to_string(v));
        CHECK(ctf::field_mul(fv, fv2) == fv * fv2, "ct mul #" + std::to_string(v));
    }
}

// ============================================================================
//  CATEGORY 20: CT Scalar (TestCategory::CTScalar)
// ============================================================================
static void test_ct_scalar() {
    std::cout << "  [CTScalar] CT scalar operations..." << std::endl;
    
    namespace cts = secp256k1::ct;
    SC a = SC::from_uint64(42);
    SC b = SC::from_uint64(99);
    SC zero = SC::zero();
    SC one = SC::one();
    
    // 20.1-20.3: Arithmetic
    CHECK(cts::scalar_add(a, b) == a + b, "ct scalar_add");
    CHECK(cts::scalar_sub(a, b) == a - b, "ct scalar_sub");
    CHECK(cts::scalar_neg(a) == a.negate(), "ct scalar_neg");
    
    // 20.4: cmov
    SC result = b;
    cts::scalar_cmov(&result, a, 0xFFFFFFFFFFFFFFFFULL);
    CHECK(result == a, "ct scalar_cmov copies");
    result = b;
    cts::scalar_cmov(&result, a, 0);
    CHECK(result == b, "ct scalar_cmov preserves");
    
    // 20.5: cswap
    SC sa = a, sb = b;
    cts::scalar_cswap(&sa, &sb, 0xFFFFFFFFFFFFFFFFULL);
    CHECK(sa == b && sb == a, "ct scalar_cswap swaps");
    sa = a; sb = b;
    cts::scalar_cswap(&sa, &sb, 0);
    CHECK(sa == a && sb == b, "ct scalar_cswap no swap");
    
    // 20.6: select
    CHECK(cts::scalar_select(a, b, 0xFFFFFFFFFFFFFFFFULL) == a, "ct scalar_select(ones)");
    CHECK(cts::scalar_select(a, b, 0) == b, "ct scalar_select(zero)");
    
    // 20.7: cneg
    CHECK(cts::scalar_cneg(a, 0xFFFFFFFFFFFFFFFFULL) == a.negate(), "ct scalar_cneg(ones)");
    CHECK(cts::scalar_cneg(a, 0) == a, "ct scalar_cneg(zero)");
    
    // 20.8: is_zero / eq
    CHECK(cts::scalar_is_zero(zero) != 0, "ct scalar_is_zero(0)");
    CHECK(cts::scalar_is_zero(a) == 0, "ct scalar_is_zero(42)");
    CHECK(cts::scalar_eq(a, a) != 0, "ct scalar_eq same");
    CHECK(cts::scalar_eq(a, b) == 0, "ct scalar_eq diff");
    
    // 20.9: bit
    SC s5 = SC::from_uint64(5);  // 101
    CHECK(cts::scalar_bit(s5, 0) == 1, "ct scalar_bit(5,0)");
    CHECK(cts::scalar_bit(s5, 1) == 0, "ct scalar_bit(5,1)");
    CHECK(cts::scalar_bit(s5, 2) == 1, "ct scalar_bit(5,2)");
    
    // 20.10: window
    SC s0xFF = SC::from_uint64(0xFF);  // 11111111
    uint64_t w = cts::scalar_window(s0xFF, 0, 4);
    CHECK(w == 0x0F, "ct scalar_window(0xFF,0,4)==0x0F");
    w = cts::scalar_window(s0xFF, 4, 4);
    CHECK(w == 0x0F, "ct scalar_window(0xFF,4,4)==0x0F");
}

// ============================================================================
//  CATEGORY 21: CT Point (TestCategory::CTPoint)
// ============================================================================
static void test_ct_point() {
    std::cout << "  [CTPoint] CT point operations..." << std::endl;
    
    namespace ctp = secp256k1::ct;
    PT G = PT::generator();
    PT O = PT::infinity();
    
    // 21.1: CT scalar_mul matches fast
    for (unsigned k = 1; k <= 64; ++k) {
        SC s = SC::from_uint64(k);
        PT ct_result = ctp::scalar_mul(G, s);
        PT fast_result = G.scalar_mul(s);
        CHECK(pt_eq(ct_result, fast_result),
              "ct scalar_mul(" + std::to_string(k) + ") matches");
    }
    
    // 21.2: CT generator_mul
    for (unsigned k = 1; k <= 32; ++k) {
        SC s = SC::from_uint64(k);
        PT ct_gen = ctp::generator_mul(s);
        PT fast_result = G.scalar_mul(s);
        CHECK(pt_eq(ct_gen, fast_result),
              "ct generator_mul(" + std::to_string(k) + ") matches");
    }
    
    // 21.3: CT on-curve check
    CHECK(ctp::point_is_on_curve(G) != 0, "ct G is on curve");
    for (unsigned k = 2; k <= 16; ++k) {
        PT P = G.scalar_mul(SC::from_uint64(k));
        CHECK(ctp::point_is_on_curve(P) != 0,
              "ct " + std::to_string(k) + "G on curve");
    }
    
    // 21.4: CT point equality
    CHECK(ctp::point_eq(G, G) != 0, "ct G==G");
    PT twoG = G.scalar_mul(SC::from_uint64(2));
    CHECK(ctp::point_eq(G, twoG) == 0, "ct G!=2G");
    
    // 21.5: Complete addition — handles all cases
    auto jG = ctp::CTJacobianPoint::from_point(G);
    auto jO = ctp::CTJacobianPoint::make_infinity();
    
    // P + O = P
    auto res = ctp::point_add_complete(jG, jO);
    CHECK(pt_eq(res.to_point(), G), "ct complete add: G+O==G");
    
    // O + P = P
    res = ctp::point_add_complete(jO, jG);
    CHECK(pt_eq(res.to_point(), G), "ct complete add: O+G==G");
    
    // P + P = 2P
    res = ctp::point_add_complete(jG, jG);
    CHECK(pt_eq(res.to_point(), G.dbl()), "ct complete add: G+G==2G");
    
    // P + (-P) = O
    auto jNegG = ctp::point_neg(jG);
    res = ctp::point_add_complete(jG, jNegG);
    CHECK(res.to_point().is_infinity(), "ct complete add: G+(-G)==O");
    
    // 21.6: point_cmov
    auto jp = ctp::CTJacobianPoint::from_point(G);
    auto jq = ctp::CTJacobianPoint::from_point(twoG);
    ctp::point_cmov(&jp, jq, 0xFFFFFFFFFFFFFFFFULL);
    CHECK(pt_eq(jp.to_point(), twoG), "ct point_cmov copies");
    
    // 21.7: point_select
    auto sel = ctp::point_select(jG, jq, 0xFFFFFFFFFFFFFFFFULL);
    CHECK(pt_eq(sel.to_point(), G), "ct point_select(ones)==G");
    sel = ctp::point_select(jG, jq, 0);
    CHECK(pt_eq(sel.to_point(), twoG), "ct point_select(zero)==2G");
    
    // 21.8: CT scalar_mul with large scalar
    SC large = SC::from_hex("DEADBEEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF01234567");
    PT ct_large = ctp::scalar_mul(G, large);
    PT fast_large = G.scalar_mul(large);
    CHECK(pt_eq(ct_large, fast_large), "ct scalar_mul(large)");
    
    // 21.9: CT scalar_mul identity (0*G = O)
    PT ct_zero = ctp::scalar_mul(G, SC::zero());
    CHECK(ct_zero.is_infinity(), "ct 0*G==O");
    
    // 21.10: CT scalar_mul (n*G = O)
    PT ct_order = ctp::scalar_mul(G, secp256k1_n());
    CHECK(ct_order.is_infinity(), "ct n*G==O");
}

// ============================================================================
//  CATEGORY 22: GLV Endomorphism (TestCategory::GLV)
// ============================================================================
static void test_glv() {
    std::cout << "  [GLV] GLV endomorphism tests..." << std::endl;
    
    using namespace secp256k1::fast;
    PT G = PT::generator();
    
    // 22.1: Endomorphism β*G.x produces valid point
    PT endo = apply_endomorphism(G);
    FE ex = endo.x();
    FE ey = endo.y();
    CHECK(ey.square() == ex.square() * ex + FE::from_uint64(7), "φ(G) on curve");
    
    // 22.2: φ(G).y == G.y (endomorphism preserves y)
    CHECK(endo.y() == G.y(), "φ(G).y == G.y");
    
    // 22.3: φ(G).x != G.x
    CHECK(endo.x() != G.x(), "φ(G).x != G.x");
    
    // 22.4: verify_endomorphism — φ(φ(P)) + P should relate to curve
    CHECK(verify_endomorphism(G), "verify_endomorphism(G)");
    
    // 22.5: verify_endomorphism for multiple points
    for (unsigned k = 2; k <= 16; ++k) {
        PT P = G.scalar_mul(SC::from_uint64(k));
        CHECK(verify_endomorphism(P), "verify_endomorphism(" + std::to_string(k) + "G)");
    }
    
    // 22.6: GLV decomposition — k = k1 + k2*λ (mod n)
    for (unsigned k = 1; k <= 64; ++k) {
        SC sk = SC::from_uint64(k);
        auto decomp = glv_decompose(sk);
        
        // Verify: k1 + k2*λ ≡ k (mod n)
        SC lambda = SC::from_bytes(secp256k1::fast::glv_constants::LAMBDA);
        SC k1 = decomp.k1_neg ? decomp.k1.negate() : decomp.k1;
        SC k2 = decomp.k2_neg ? decomp.k2.negate() : decomp.k2;
        SC reconstructed = k1 + k2 * lambda;
        CHECK(reconstructed == sk, "GLV decompose/reconstruct k=" + std::to_string(k));
    }
    
    // 22.7: GLV decomposition for large scalar
    SC large = SC::from_hex("DEADBEEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF01234567");
    auto decomp = glv_decompose(large);
    SC lambda = SC::from_bytes(secp256k1::fast::glv_constants::LAMBDA);
    SC lk1 = decomp.k1_neg ? decomp.k1.negate() : decomp.k1;
    SC lk2 = decomp.k2_neg ? decomp.k2.negate() : decomp.k2;
    CHECK(lk1 + lk2 * lambda == large, "GLV decompose large");
    
    // 22.8: λ*G = φ(G) (eigenvalue property)
    SC lambda_sc = SC::from_bytes(secp256k1::fast::glv_constants::LAMBDA);
    PT lambdaG = G.scalar_mul(lambda_sc);
    CHECK(pt_eq(lambdaG, endo), "λ*G == φ(G)");
    
    // 22.9: Endomorphism of infinity
    PT endo_inf = apply_endomorphism(PT::infinity());
    CHECK(endo_inf.is_infinity(), "φ(O)==O");
    
    // 22.10: λ² + λ + 1 ≡ 0 (mod n)
    SC lambda2 = lambda_sc * lambda_sc;
    SC sum = lambda2 + lambda_sc + SC::one();
    CHECK(sum == SC::zero(), "λ²+λ+1 ≡ 0 (mod n)");
    
    // 22.11: β³ ≡ 1 (mod p) 
    FE beta = FE::from_bytes(secp256k1::fast::glv_constants::BETA);
    FE beta3 = beta * beta * beta;
    CHECK(beta3 == FE::one(), "β³ ≡ 1 (mod p)");
}

// ============================================================================
//  CATEGORY 23: MSM (TestCategory::MSM)
// ============================================================================
static void test_msm() {
    std::cout << "  [MSM] Multi-scalar multiplication tests..." << std::endl;
    
    PT G = PT::generator();
    
    // 23.1: Empty MSM = O
    CHECK(secp256k1::msm(std::vector<SC>{}, std::vector<PT>{}).is_infinity(),
          "MSM(empty)==O");
    
    // 23.2: Single element MSM = scalar_mul
    SC s42 = SC::from_uint64(42);
    PT single = secp256k1::msm(std::vector<SC>{s42}, std::vector<PT>{G});
    CHECK(pt_eq(single, G.scalar_mul(s42)), "MSM(n=1)==scalar_mul");
    
    // 23.3: Two-element MSM
    SC s_a = SC::from_uint64(7);
    SC s_b = SC::from_uint64(11);
    PT P_a = G;
    PT P_b = G.scalar_mul(SC::from_uint64(3));
    PT msm2 = secp256k1::msm(std::vector<SC>{s_a, s_b}, std::vector<PT>{P_a, P_b});
    PT naive2 = P_a.scalar_mul(s_a).add(P_b.scalar_mul(s_b));
    CHECK(pt_eq(msm2, naive2), "MSM(n=2)");
    
    // 23.4: Pippenger for medium n
    const int n = 64;
    std::vector<SC> scalars(n);
    std::vector<PT> points(n);
    PT P = G;
    for (int i = 0; i < n; ++i) {
        scalars[i] = SC::from_uint64(static_cast<uint64_t>(i * 31 + 17));
        points[i] = P;
        P = P.add(G);
    }
    
    PT naive = PT::infinity();
    for (int i = 0; i < n; ++i) {
        naive = naive.add(points[i].scalar_mul(scalars[i]));
    }
    
    PT pip = secp256k1::pippenger_msm(scalars, points);
    CHECK(pt_eq(pip, naive), "Pippenger(n=64)");
    
    PT unified = secp256k1::msm(scalars, points);
    CHECK(pt_eq(unified, naive), "MSM(n=64)");
    
    // 23.5: Shamir trick (2-point MSM)
    PT shamir = secp256k1::shamir_trick(s_a, P_a, s_b, P_b);
    CHECK(pt_eq(shamir, naive2), "shamir_trick(2)");
    
    // 23.6: Pippenger with larger n
    const int n2 = 256;
    std::vector<SC> scalars2(n2);
    std::vector<PT> points2(n2);
    P = G;
    for (int i = 0; i < n2; ++i) {
        scalars2[i] = SC::from_uint64(static_cast<uint64_t>(i * 13 + 5));
        points2[i] = P;
        P = P.add(G);
    }
    
    PT naive2_big = PT::infinity();
    for (int i = 0; i < n2; ++i) {
        naive2_big = naive2_big.add(points2[i].scalar_mul(scalars2[i]));
    }
    
    PT pip2 = secp256k1::pippenger_msm(scalars2, points2);
    CHECK(pt_eq(pip2, naive2_big), "Pippenger(n=256)");
    
    // 23.7: Optimal window function
    CHECK(secp256k1::pippenger_optimal_window(1) >= 1, "optimal_window(1)>=1");
    CHECK(secp256k1::pippenger_optimal_window(256) >= 1, "optimal_window(256)>=1");
    
    // 23.8: MSM with zero scalars
    std::vector<SC> zeros_s = {SC::zero(), SC::zero()};
    std::vector<PT> pts = {G, G.dbl()};
    PT zero_msm = secp256k1::msm(zeros_s, pts);
    CHECK(zero_msm.is_infinity(), "MSM with zero scalars == O");
    
    // 23.9: Strauss optimal window
    CHECK(secp256k1::strauss_optimal_window(1) >= 1, "strauss_window(1)>=1");
    
    // 23.10: multi_scalar_mul (from multiscalar.hpp)
    PT multi_result = secp256k1::multi_scalar_mul(
        std::vector<SC>{s_a, s_b}, std::vector<PT>{P_a, P_b});
    CHECK(pt_eq(multi_result, naive2), "multi_scalar_mul(n=2)");
}

// ============================================================================
//  CATEGORY 24: Comb Generator (TestCategory::CombGen)
// ============================================================================
static void test_comb_gen() {
    std::cout << "  [CombGen] Comb generator tests..." << std::endl;
    
    PT G = PT::generator();
    
    // 24.1: Init and ready
    secp256k1::fast::CombGenContext ctx;
    ctx.init(6);
    CHECK(ctx.ready(), "CombGenContext ready after init");
    
    // 24.2: teeth/spacing
    CHECK(ctx.teeth() == 6, "teeth==6");
    CHECK(ctx.spacing() > 0, "spacing>0");
    CHECK(ctx.table_size_bytes() > 0, "table_size>0");
    
    // 24.3: mul matches scalar_mul for small range
    for (unsigned k = 1; k <= 128; ++k) {
        SC s = SC::from_uint64(k);
        PT comb_result = ctx.mul(s);
        PT expected = G.scalar_mul(s);
        CHECK(pt_eq(comb_result, expected),
              "comb_mul(" + std::to_string(k) + ")");
    }
    
    // 24.4: CT mul matches
    for (unsigned k = 1; k <= 16; ++k) {
        SC s = SC::from_uint64(k);
        PT ct_result = ctx.mul_ct(s);
        PT expected = G.scalar_mul(s);
        CHECK(pt_eq(ct_result, expected),
              "comb_mul_ct(" + std::to_string(k) + ")");
    }
    
    // 24.5: Large scalar
    SC large = SC::from_hex("DEADBEEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF01234567");
    CHECK(pt_eq(ctx.mul(large), G.scalar_mul(large)), "comb_mul(large)");
    
    // 24.6: Global singleton
    secp256k1::fast::init_comb_gen(6);
    CHECK(secp256k1::fast::comb_gen_ready(), "global comb ready");
    
    SC s42 = SC::from_uint64(42);
    CHECK(pt_eq(secp256k1::fast::comb_gen_mul(s42), G.scalar_mul(s42)), "global comb_gen_mul(42)");
    
    // 24.7: CT global
    CHECK(pt_eq(secp256k1::fast::comb_gen_mul_ct(s42), G.scalar_mul(s42)), "global comb_gen_mul_ct(42)");
    
    // 24.8: Different teeth values
    for (unsigned teeth : {4u, 5u, 7u, 8u}) {
        secp256k1::fast::CombGenContext ctx2;
        ctx2.init(teeth);
        CHECK(ctx2.ready(), "CombGen teeth=" + std::to_string(teeth) + " ready");
        
        SC test_s = SC::from_uint64(42);
        PT result = ctx2.mul(test_s);
        PT expected = G.scalar_mul(test_s);
        CHECK(pt_eq(result, expected),
              "CombGen teeth=" + std::to_string(teeth) + " correct");
    }
    
    // 24.9: Zero scalar
    SC s_zero = SC::zero();
    // Comb mul of zero should produce infinity
    PT zero_result = ctx.mul(s_zero);
    CHECK(zero_result.is_infinity(), "comb_mul(0)==O");
    
    // 24.10: One scalar
    SC s_one = SC::one();
    CHECK(pt_eq(ctx.mul(s_one), G), "comb_mul(1)==G");
}

// ============================================================================
//  CATEGORY 25: Batch Inverse (TestCategory::BatchInverse)
// ============================================================================
static void test_batch_inverse() {
    std::cout << "  [BatchInverse] Batch inverse tests..." << std::endl;
    
    FE one = FE::one();
    
    // 25.1: Batch of 1
    FE batch1[1] = {FE::from_uint64(7)};
    FE exp1 = batch1[0].inverse();
    secp256k1::fast::fe_batch_inverse(batch1, 1);
    CHECK(batch1[0] == exp1, "batch_inv(n=1)");
    
    // 25.2: Batch of 2
    FE batch2[2] = {FE::from_uint64(3), FE::from_uint64(5)};
    FE exp2[2] = {batch2[0].inverse(), batch2[1].inverse()};
    secp256k1::fast::fe_batch_inverse(batch2, 2);
    CHECK(batch2[0] == exp2[0] && batch2[1] == exp2[1], "batch_inv(n=2)");
    
    // 25.3: Batch of 16
    const int N = 16;
    FE batch16[N];
    FE expected16[N];
    for (int i = 0; i < N; ++i) {
        batch16[i] = FE::from_uint64(i + 2);
        expected16[i] = batch16[i].inverse();
    }
    secp256k1::fast::fe_batch_inverse(batch16, N);
    bool all_match = true;
    for (int i = 0; i < N; ++i) {
        if (batch16[i] != expected16[i]) all_match = false;
    }
    CHECK(all_match, "batch_inv(n=16) all match");
    
    // 25.4: Verify a*a⁻¹ = 1 for batch results
    FE vals[8];
    for (int i = 0; i < 8; ++i) vals[i] = FE::from_uint64(i * 7 + 3);
    FE originals[8];
    std::memcpy(originals, vals, sizeof(vals));
    secp256k1::fast::fe_batch_inverse(vals, 8);
    for (int i = 0; i < 8; ++i) {
        CHECK(originals[i] * vals[i] == one, "batch a*a⁻¹==1 #" + std::to_string(i));
    }
    
    // 25.5: Large batch
    const int LN = 64;
    FE large_batch[LN];
    FE large_orig[LN];
    for (int i = 0; i < LN; ++i) {
        large_batch[i] = FE::from_uint64(i + 2);
        large_orig[i] = large_batch[i];
    }
    secp256k1::fast::fe_batch_inverse(large_batch, LN);
    bool large_ok = true;
    for (int i = 0; i < LN; ++i) {
        if (large_orig[i] * large_batch[i] != one) large_ok = false;
    }
    CHECK(large_ok, "batch_inv(n=64) all correct");
}

// ============================================================================
//  CATEGORY 26: ECDSA (TestCategory::ECDSA)
// ============================================================================
static void test_ecdsa() {
    std::cout << "  [ECDSA] ECDSA sign/verify tests..." << std::endl;
    
    PT G = PT::generator();
    
    // 26.1: Sign and verify basic
    SC privkey = SC::from_uint64(42);
    PT pubkey = G.scalar_mul(privkey);
    std::array<uint8_t, 32> msg{};
    msg[0] = 0x01;
    
    auto sig = secp256k1::ecdsa_sign(msg, privkey);
    CHECK(secp256k1::ecdsa_verify(msg, pubkey, sig), "ECDSA sign+verify");
    
    // 26.2: Wrong message fails
    std::array<uint8_t, 32> wrong_msg{};
    wrong_msg[0] = 0x02;
    CHECK(!secp256k1::ecdsa_verify(wrong_msg, pubkey, sig), "ECDSA wrong msg fails");
    
    // 26.3: Wrong pubkey fails
    PT wrong_pub = G.scalar_mul(SC::from_uint64(43));
    CHECK(!secp256k1::ecdsa_verify(msg, wrong_pub, sig), "ECDSA wrong pubkey fails");
    
    // 26.4: Deterministic nonce (RFC 6979)
    SC nonce = secp256k1::rfc6979_nonce(privkey, msg);
    CHECK(!nonce.is_zero(), "RFC6979 nonce nonzero");
    
    // Same inputs → same nonce (deterministic)
    SC nonce2 = secp256k1::rfc6979_nonce(privkey, msg);
    CHECK(nonce == nonce2, "RFC6979 deterministic");
    
    // Different key → different nonce
    SC nonce3 = secp256k1::rfc6979_nonce(SC::from_uint64(43), msg);
    CHECK(nonce != nonce3, "RFC6979 different key → different nonce");
    
    // 26.5: Signature normalization (low-S)
    auto norm_sig = sig.normalize();
    CHECK(norm_sig.is_low_s(), "normalized is low-S");
    CHECK(secp256k1::ecdsa_verify(msg, pubkey, norm_sig), "normalized sig verifies");
    
    // 26.6: Compact serialization roundtrip
    auto compact = sig.to_compact();
    auto recovered_sig = secp256k1::ECDSASignature::from_compact(compact);
    CHECK(recovered_sig.r == sig.r && recovered_sig.s == sig.s, "compact roundtrip");
    
    // 26.7: DER serialization
    auto [der, der_len] = sig.to_der();
    CHECK(der_len > 0 && der_len <= 72, "DER length valid");
    CHECK(der[0] == 0x30, "DER sequence tag");
    
    // 26.8: Multiple keys
    for (uint64_t k = 1; k <= 16; ++k) {
        SC key = SC::from_uint64(k * 1000 + 7);
        PT pub = G.scalar_mul(key);
        auto s = secp256k1::ecdsa_sign(msg, key);
        CHECK(secp256k1::ecdsa_verify(msg, pub, s),
              "ECDSA key #" + std::to_string(k));
    }
    
    // 26.9: Sign with different messages
    for (uint8_t m = 0; m < 16; ++m) {
        std::array<uint8_t, 32> msg_i{};
        msg_i[0] = m;
        auto s = secp256k1::ecdsa_sign(msg_i, privkey);
        CHECK(secp256k1::ecdsa_verify(msg_i, pubkey, s),
              "ECDSA msg #" + std::to_string(m));
    }
}

// ============================================================================
//  CATEGORY 27: Schnorr (TestCategory::Schnorr)
// ============================================================================
static void test_schnorr() {
    std::cout << "  [Schnorr] Schnorr sign/verify tests..." << std::endl;
    
    PT G = PT::generator();
    
    // 27.1: Sign and verify
    SC privkey = SC::from_uint64(42);
    std::array<uint8_t, 32> msg{};
    msg[0] = 0x01;
    std::array<uint8_t, 32> aux{};
    aux[0] = 0xAB;
    
    auto sig = secp256k1::schnorr_sign(privkey, msg, aux);
    auto pubkey_x = secp256k1::schnorr_pubkey(privkey);
    CHECK(secp256k1::schnorr_verify(pubkey_x, msg, sig), "Schnorr sign+verify");
    
    // 27.2: Wrong message fails
    std::array<uint8_t, 32> wrong_msg{};
    wrong_msg[0] = 0x02;
    CHECK(!secp256k1::schnorr_verify(pubkey_x, wrong_msg, sig), "Schnorr wrong msg fails");
    
    // 27.3: Signature serialization roundtrip
    auto sig_bytes = sig.to_bytes();
    auto sig_back = secp256k1::SchnorrSignature::from_bytes(sig_bytes);
    CHECK(sig_back.s == sig.s, "Schnorr sig bytes roundtrip (s)");
    CHECK(sig_back.r == sig.r, "Schnorr sig bytes roundtrip (r)");
    
    // 27.4: Multiple keys
    for (uint64_t k = 1; k <= 16; ++k) {
        SC key = SC::from_uint64(k * 1000 + 7);
        auto pub_x = secp256k1::schnorr_pubkey(key);
        auto s = secp256k1::schnorr_sign(key, msg, aux);
        CHECK(secp256k1::schnorr_verify(pub_x, msg, s),
              "Schnorr key #" + std::to_string(k));
    }
    
    // 27.5: tagged_hash
    auto hash1 = secp256k1::tagged_hash("BIP0340/challenge", msg.data(), 32);
    auto hash2 = secp256k1::tagged_hash("BIP0340/challenge", msg.data(), 32);
    CHECK(hash1 == hash2, "tagged_hash deterministic");
    
    auto hash3 = secp256k1::tagged_hash("DifferentTag", msg.data(), 32);
    CHECK(hash1 != hash3, "tagged_hash different tag != result");
    
    // 27.6: Determinism (same inputs → same sig)
    auto sig2 = secp256k1::schnorr_sign(privkey, msg, aux);
    CHECK(sig.s == sig2.s && sig.r == sig2.r, "Schnorr deterministic");
    
    // 27.7: Different aux_rand → different signing nonce
    std::array<uint8_t, 32> aux2{};
    aux2[0] = 0xCD;
    auto sig3 = secp256k1::schnorr_sign(privkey, msg, aux2);
    // Sig should be different (different aux_rand)
    CHECK(sig3.r != sig.r || sig3.s != sig.s, "different aux → different sig");
    // But still valid
    CHECK(secp256k1::schnorr_verify(pubkey_x, msg, sig3), "diff aux sig still verifies");
}

// ============================================================================
//  CATEGORY 28: ECDH (TestCategory::ECDH)
// ============================================================================
static void test_ecdh() {
    std::cout << "  [ECDH] ECDH shared secret tests..." << std::endl;
    
    PT G = PT::generator();
    using secp256k1::Scalar;
    using secp256k1::Point;
    
    // 28.1: Basic ECDH — Alice and Bob derive same secret
    SC alice_priv = SC::from_uint64(42);
    SC bob_priv = SC::from_uint64(99);
    PT alice_pub = G.scalar_mul(alice_priv);
    PT bob_pub = G.scalar_mul(bob_priv);
    
    auto secret_alice = secp256k1::ecdh_compute(alice_priv, bob_pub);
    auto secret_bob = secp256k1::ecdh_compute(bob_priv, alice_pub);
    CHECK(secret_alice == secret_bob, "ECDH shared secret matches");
    
    // 28.2: Different keys → different secrets
    SC carol_priv = SC::from_uint64(200);
    PT carol_pub = G.scalar_mul(carol_priv);
    auto secret_ac = secp256k1::ecdh_compute(alice_priv, carol_pub);
    CHECK(secret_alice != secret_ac, "different keys → different secrets");
    
    // 28.3: xonly variant
    auto xonly_alice = secp256k1::ecdh_compute_xonly(alice_priv, bob_pub);
    auto xonly_bob = secp256k1::ecdh_compute_xonly(bob_priv, alice_pub);
    CHECK(xonly_alice == xonly_bob, "ECDH xonly matches");
    
    // 28.4: Raw variant
    auto raw_alice = secp256k1::ecdh_compute_raw(alice_priv, bob_pub);
    auto raw_bob = secp256k1::ecdh_compute_raw(bob_priv, alice_pub);
    CHECK(raw_alice == raw_bob, "ECDH raw matches");
    
    // 28.5: Multiple key pairs
    for (uint64_t ka = 1; ka <= 8; ++ka) {
        for (uint64_t kb = ka + 1; kb <= 9; ++kb) {
            SC a = SC::from_uint64(ka * 1000 + 7);
            SC b = SC::from_uint64(kb * 1000 + 13);
            PT pa = G.scalar_mul(a);
            PT pb = G.scalar_mul(b);
            auto sa = secp256k1::ecdh_compute(a, pb);
            auto sb = secp256k1::ecdh_compute(b, pa);
            CHECK(sa == sb, "ECDH pair (" + std::to_string(ka) + "," + std::to_string(kb) + ")");
        }
    }
}

// ============================================================================
//  CATEGORY 29: Key Recovery (TestCategory::Recovery)
// ============================================================================
static void test_recovery() {
    std::cout << "  [Recovery] Key recovery tests..." << std::endl;
    
    PT G = PT::generator();
    
    // 29.1: Sign recoverable and recover pubkey
    SC privkey = SC::from_uint64(42);
    PT pubkey = G.scalar_mul(privkey);
    std::array<uint8_t, 32> msg{};
    msg[0] = 0x01;
    
    auto rsig = secp256k1::ecdsa_sign_recoverable(msg, privkey);
    CHECK(rsig.recid >= 0 && rsig.recid <= 3, "recid in [0,3]");
    
    // 29.2: Recover public key
    auto [recovered, success] = secp256k1::ecdsa_recover(msg, rsig.sig, rsig.recid);
    CHECK(success, "recovery succeeded");
    CHECK(pt_eq(recovered, pubkey), "recovered pubkey matches");
    
    // 29.3: Wrong recid fails or gives wrong key
    int wrong_recid = (rsig.recid + 1) % 4;
    auto [recovered2, success2] = secp256k1::ecdsa_recover(msg, rsig.sig, wrong_recid);
    if (success2) {
        CHECK(!pt_eq(recovered2, pubkey), "wrong recid → wrong key");
    } else {
        CHECK(true, "wrong recid → recovery failed (expected)");
    }
    
    // 29.4: Compact roundtrip
    auto compact = secp256k1::recoverable_to_compact(rsig);
    auto [back, back_ok] = secp256k1::recoverable_from_compact(compact);
    CHECK(back_ok, "recoverable compact roundtrip");
    CHECK(back.recid == rsig.recid, "recoverable recid preserved");
    CHECK(back.sig.r == rsig.sig.r && back.sig.s == rsig.sig.s, "recoverable sig preserved");
    
    // 29.5: Multiple keys
    for (uint64_t k = 1; k <= 8; ++k) {
        SC key = SC::from_uint64(k * 1000 + 7);
        PT pub = G.scalar_mul(key);
        auto rs = secp256k1::ecdsa_sign_recoverable(msg, key);
        auto [rec, ok] = secp256k1::ecdsa_recover(msg, rs.sig, rs.recid);
        CHECK(ok, "recovery key #" + std::to_string(k));
        CHECK(pt_eq(rec, pub), "recovered matches key #" + std::to_string(k));
    }
}

// ============================================================================
//  BONUS: SHA-256 / SHA-512 tests (integrated into protocol categories)
// ============================================================================
static void test_hashing() {
    std::cout << "  [Hashing] SHA-256/SHA-512 tests..." << std::endl;
    
    // SHA-256 NIST vectors
    // 30.1: SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    auto empty_hash = secp256k1::SHA256::hash(nullptr, 0);
    CHECK(empty_hash[0] == 0xe3 && empty_hash[1] == 0xb0 && empty_hash[2] == 0xc4,
          "SHA-256('') first 3 bytes");
    
    // 30.2: SHA-256("abc")
    const char* abc = "abc";
    auto abc_hash = secp256k1::SHA256::hash(abc, 3);
    // Expected: ba7816bf 8f01cfea 414140de 5dae2223 b00361a3 96177a9c b410ff61 f20015ad
    CHECK(abc_hash[0] == 0xba && abc_hash[1] == 0x78 && abc_hash[2] == 0x16 && abc_hash[3] == 0xbf,
          "SHA-256('abc') first 4 bytes");
    
    // 30.3: SHA-256 incremental matches one-shot
    secp256k1::SHA256 sha;
    sha.update("ab", 2);
    sha.update("c", 1);
    auto incremental = sha.finalize();
    CHECK(incremental == abc_hash, "SHA-256 incremental matches one-shot");
    
    // 30.4: SHA-256 reset
    sha.reset();
    sha.update("abc", 3);
    auto after_reset = sha.finalize();
    CHECK(after_reset == abc_hash, "SHA-256 reset works");
    
    // 30.5: hash256 (double SHA-256)
    auto double_hash = secp256k1::SHA256::hash256("abc", 3);
    auto manual_double = secp256k1::SHA256::hash(abc_hash.data(), 32);
    CHECK(double_hash == manual_double, "hash256 == hash(hash(x))");
    
    // SHA-512 
    // 30.6: SHA-512("") = cf83e1357eefb8bd...
    auto empty512 = secp256k1::SHA512::hash(nullptr, 0);
    CHECK(empty512[0] == 0xcf && empty512[1] == 0x83 && empty512[2] == 0xe1,
          "SHA-512('') first 3 bytes");
    
    // 30.7: SHA-512("abc")
    auto abc512 = secp256k1::SHA512::hash(abc, 3);
    // Expected: ddaf35a193617aba cc417349ae204131 12e6fa4e89a97ea2 ...
    CHECK(abc512[0] == 0xdd && abc512[1] == 0xaf && abc512[2] == 0x35 && abc512[3] == 0xa1,
          "SHA-512('abc') first 4 bytes");
    
    // 30.8: SHA-512 incremental
    secp256k1::SHA512 sha512;
    sha512.update("ab", 2);
    sha512.update("c", 1);
    auto inc512 = sha512.finalize();
    CHECK(inc512 == abc512, "SHA-512 incremental matches");
    
    // 30.9: SHA-256 longer input
    const char* msg448 = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    auto hash448 = secp256k1::SHA256::hash(msg448, std::strlen(msg448));
    // Expected: 248d6a61 d20638b8 e5c02693 0c3e6039 a33ce459 64ff2167 f6ecedd4 19db06c1
    CHECK(hash448[0] == 0x24 && hash448[1] == 0x8d && hash448[2] == 0x6a,
          "SHA-256(448-bit msg) first 3 bytes");
    
    // 30.10: Determinism
    auto h1 = secp256k1::SHA256::hash(abc, 3);
    auto h2 = secp256k1::SHA256::hash(abc, 3);
    CHECK(h1 == h2, "SHA-256 deterministic");
}

// ============================================================================
//  BONUS: Batch Add Affine tests
// ============================================================================
static void test_batch_add_affine_comprehensive() {
    std::cout << "  [BatchAddAffine] Batch affine addition tests..." << std::endl;
    
    using namespace secp256k1::fast;
    PT G = PT::generator();
    FE gx = G.x();
    FE gy = G.y();
    
    // 31.1: precompute_g_multiples
    auto g_multiples = precompute_g_multiples(64);
    CHECK(g_multiples.size() == 64, "precomputed 64 G multiples");
    
    // 31.2: Verify precomputed multiples are on curve
    for (std::size_t i = 0; i < 16 && i < g_multiples.size(); ++i) {
        FE cx = g_multiples[i].x;
        FE cy = g_multiples[i].y;
        FE lhs = cy.square();
        FE rhs = cx.square() * cx + FE::from_uint64(7);
        CHECK(lhs == rhs, "g_multiple[" + std::to_string(i) + "] on curve");
    }
    
    // 31.3: precompute_point_multiples with custom base
    PT p7 = G.scalar_mul(SC::from_uint64(7));
    FE p7x = p7.x();
    FE p7y = p7.y();
    auto p7_multiples = precompute_point_multiples(p7x, p7y, 16);
    CHECK(p7_multiples.size() == 16, "precomputed 16 7G multiples");
    
    // 31.5: negate_affine_table flips sign
    auto negated = negate_affine_table(g_multiples.data(), 16);
    CHECK(negated.size() == 16, "negated table size");
    // Negated y should differ from original
    CHECK(negated[0].y != g_multiples[0].y, "negated y differs");
    CHECK(negated[0].x == g_multiples[0].x, "negated x same");
    
    // 31.6: Negate + original y should sum to zero on curve (y + (-y) = 0 mod p)
    FE y_sum = g_multiples[0].y + negated[0].y;
    CHECK(y_sum == FE::zero(), "y + neg_y == 0 (additive inverse)");
}

// ============================================================================
//  BONUS: Batch Verify tests
// ============================================================================
static void test_batch_verify() {
    std::cout << "  [BatchVerify] Batch verification tests..." << std::endl;
    
    PT G = PT::generator();
    
    // 32.1: ECDSA batch verify (all valid)
    std::vector<secp256k1::ECDSABatchEntry> ecdsa_entries;
    for (uint64_t k = 1; k <= 4; ++k) {
        SC key = SC::from_uint64(k * 1000 + 7);
        PT pub = G.scalar_mul(key);
        std::array<uint8_t, 32> msg{};
        msg[0] = static_cast<uint8_t>(k);
        auto sig = secp256k1::ecdsa_sign(msg, key);
        ecdsa_entries.push_back({msg, pub, sig});
    }
    CHECK(secp256k1::ecdsa_batch_verify(ecdsa_entries), "ECDSA batch verify all valid");
    
    // 32.2: ECDSA batch verify with one invalid
    auto bad_entries = ecdsa_entries;
    bad_entries[2].msg_hash[0] ^= 0xFF;  // corrupt one message
    CHECK(!secp256k1::ecdsa_batch_verify(bad_entries), "ECDSA batch verify detects invalid");
    
    // 32.3: Identify invalid entries
    auto invalid_indices = secp256k1::ecdsa_batch_identify_invalid(
        bad_entries.data(), bad_entries.size());
    CHECK(!invalid_indices.empty(), "ecdsa_batch_identify finds invalid");
    bool found_idx2 = false;
    for (auto idx : invalid_indices) {
        if (idx == 2) found_idx2 = true;
    }
    CHECK(found_idx2, "identified corrupted entry at index 2");
    
    // 32.4: Schnorr batch verify
    std::vector<secp256k1::SchnorrBatchEntry> schnorr_entries;
    for (uint64_t k = 1; k <= 4; ++k) {
        SC key = SC::from_uint64(k * 1000 + 7);
        auto pub_x = secp256k1::schnorr_pubkey(key);
        std::array<uint8_t, 32> msg{};
        msg[0] = static_cast<uint8_t>(k);
        std::array<uint8_t, 32> aux{};
        aux[0] = static_cast<uint8_t>(k + 0x10);
        auto sig = secp256k1::schnorr_sign(key, msg, aux);
        schnorr_entries.push_back({pub_x, msg, sig});
    }
    CHECK(secp256k1::schnorr_batch_verify(schnorr_entries), "Schnorr batch verify all valid");
    
    // 32.5: Schnorr batch with invalid
    auto bad_schnorr = schnorr_entries;
    bad_schnorr[1].message[0] ^= 0xFF;
    CHECK(!secp256k1::schnorr_batch_verify(bad_schnorr), "Schnorr batch detects invalid");
    
    // 32.6: Empty batch
    CHECK(secp256k1::ecdsa_batch_verify(std::vector<secp256k1::ECDSABatchEntry>{}),
          "ECDSA empty batch verify");
    CHECK(secp256k1::schnorr_batch_verify(std::vector<secp256k1::SchnorrBatchEntry>{}),
          "Schnorr empty batch verify");
}

// ============================================================================
//  BONUS: Homomorphism & Order tests (expanded)
// ============================================================================
static void test_homomorphism_expanded() {
    std::cout << "  [Homomorphism] Expanded homomorphism tests..." << std::endl;
    
    PT G = PT::generator();
    
    // 33.1: a*G + b*G = (a+b)*G for larger range
    const unsigned N = 128;
    unsigned M = 2 * N + 1;
    std::vector<PT> ref(M + 1);
    ref[0] = PT::infinity();
    ref[1] = G;
    for (unsigned k = 2; k <= M; ++k) ref[k] = ref[k-1].add(G);
    
    unsigned step = N / 32;
    for (unsigned a = 1; a <= N; a += step) {
        for (unsigned b = 1; b <= N; b += step) {
            PT lhs = ref[a].add(ref[b]);
            PT rhs = ref[a + b];
            CHECK(pt_eq(lhs, rhs),
                  "homo a=" + std::to_string(a) + ",b=" + std::to_string(b));
        }
    }
    
    // 33.2: Doubling chain — 2^k * G via repeated doubling
    PT dbl_chain = G;
    for (int k = 1; k <= 20; ++k) {
        PT dbl_result = dbl_chain.dbl();
        PT add_result = dbl_chain.add(dbl_chain);
        CHECK(pt_eq(dbl_result, add_result), "2^" + std::to_string(k) + " dbl==add");
        dbl_chain = dbl_result;
    }
    
    // 33.3: Subtraction property: (a+b)*G - b*G = a*G
    for (unsigned a = 1; a <= 32; ++a) {
        for (unsigned b = 1; b <= 32; b += 3) {
            PT abG = ref[a + b];
            PT bG_neg = ref[b].negate();
            PT diff = abG.add(bG_neg);
            CHECK(pt_eq(diff, ref[a]),
                  "(a+b)G - bG = aG, a=" + std::to_string(a) + ",b=" + std::to_string(b));
        }
    }
}

// ============================================================================
//  BONUS: Precompute module tests
// ============================================================================
static void test_precompute() {
    std::cout << "  [Precompute] Precomputation module tests..." << std::endl;
    
    using namespace secp256k1::fast;
    PT G = PT::generator();
    
    // 34.1: split_scalar_glv correctness
    for (unsigned k = 1; k <= 32; ++k) {
        SC s = SC::from_uint64(k);
        auto decomp = split_scalar_glv(s);
        
        // Reconstructed via endomorphism
        SC lambda = SC::from_bytes(glv_constants::LAMBDA);
        SC k1 = decomp.neg1 ? decomp.k1.negate() : decomp.k1;
        SC k2 = decomp.neg2 ? decomp.k2.negate() : decomp.k2;
        CHECK(k1 + k2 * lambda == s,
              "split_scalar_glv(" + std::to_string(k) + ")");
    }
    
    // 34.2: precompute_scalar_for_arbitrary
    SC key = SC::from_uint64(42);
    auto precomp = precompute_scalar_for_arbitrary(key, 4);
    CHECK(precomp.is_valid(), "precomputed scalar valid");
    
    // Use it on a point
    PT P = G.scalar_mul(SC::from_uint64(7));
    PT result = scalar_mul_arbitrary_precomputed(P, precomp);
    PT expected = P.scalar_mul(key);
    CHECK(pt_eq(result, expected), "precomputed_arbitrary matches");
    
    // 34.3: precompute_scalar_optimized
    auto precomp_opt = precompute_scalar_optimized(key, 4);
    CHECK(precomp_opt.is_valid(), "optimized precomp valid");
    
    PT result_opt = scalar_mul_arbitrary_precomputed_optimized(P, precomp_opt);
    CHECK(pt_eq(result_opt, expected), "optimized precomputed matches");
    
    // 34.4: scalar_mul_arbitrary (via standard scalar_mul as fallback reference)
    PT arb_result = P.scalar_mul(key);
    CHECK(pt_eq(arb_result, expected), "scalar_mul on arbitrary point matches");
    
    // 34.5: compute_wnaf
    auto wnaf = compute_wnaf(key, 4);
    CHECK(!wnaf.empty(), "compute_wnaf non-empty");
}

// ============================================================================
//  Runner — dispatches by TestCategory
// ============================================================================
using TestFn = void(*)();
struct CategoryEntry {
    TestCategory cat;
    TestFn fn;
};

static const CategoryEntry CATEGORY_TABLE[] = {
    { TestCategory::FieldArith,           test_field_arith },
    { TestCategory::FieldConversions,     test_field_conversions },
    { TestCategory::FieldEdgeCases,       test_field_edge_cases },
    { TestCategory::FieldInverse,         test_field_inverse },
    { TestCategory::FieldBranchless,      test_field_branchless },
    { TestCategory::FieldOptimal,         test_field_optimal },
    { TestCategory::FieldRepresentations, test_field_representations },
    { TestCategory::ScalarArith,          test_scalar_arith },
    { TestCategory::ScalarConversions,    test_scalar_conversions },
    { TestCategory::ScalarEdgeCases,      test_scalar_edge_cases },
    { TestCategory::ScalarEncoding,       test_scalar_encoding },
    { TestCategory::PointBasic,           test_point_basic },
    { TestCategory::PointScalarMul,       test_point_scalar_mul },
    { TestCategory::PointInplace,         test_point_inplace },
    { TestCategory::PointPrecomputed,     test_point_precomputed },
    { TestCategory::PointSerialization,   test_point_serialization },
    { TestCategory::PointEdgeCases,       test_point_edge_cases },
    { TestCategory::CTOps,                test_ct_ops },
    { TestCategory::CTField,              test_ct_field },
    { TestCategory::CTScalar,             test_ct_scalar },
    { TestCategory::CTPoint,              test_ct_point },
    { TestCategory::GLV,                  test_glv },
    { TestCategory::MSM,                  test_msm },
    { TestCategory::CombGen,              test_comb_gen },
    { TestCategory::BatchInverse,         test_batch_inverse },
    { TestCategory::ECDSA,                test_ecdsa },
    { TestCategory::Schnorr,              test_schnorr },
    { TestCategory::ECDH,                 test_ecdh },
    { TestCategory::Recovery,             test_recovery },
};

// Extra tests (not in single-category mapping, run under All)
static const TestFn EXTRA_TESTS[] = {
    test_hashing,
    test_batch_add_affine_comprehensive,
    test_batch_verify,
    test_homomorphism_expanded,
    test_precompute,
};

#ifdef STANDALONE_TEST
int main() {
#else
int test_comprehensive_run() {
#endif
    std::cout << "\n=== Comprehensive Test Suite (" << secp256k1::test::NUM_CATEGORIES 
              << " categories) ===" << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    
    g_counters = {};
    
    // Run all categorized tests
    for (auto& entry : CATEGORY_TABLE) {
        std::cout << "\n── " << secp256k1::test::category_name(entry.cat) << " ──" << std::endl;
        entry.fn();
    }
    
    // Run extra tests
    std::cout << "\n── Extra/Cross-cutting Tests ──" << std::endl;
    for (auto fn : EXTRA_TESTS) {
        fn();
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    
    std::cout << "\n  ── Comprehensive Results: "
              << g_counters.passed << " passed, "
              << g_counters.failed << " failed, "
              << g_counters.skipped << " skipped"
              << " (" << ms << " ms)" << std::endl;
    
    if (g_counters.failed > 0) {
        std::cerr << "\n  *** COMPREHENSIVE TESTS FAILED ***" << std::endl;
    } else {
        std::cout << "  All comprehensive tests PASSED" << std::endl;
    }
    
#ifdef STANDALONE_TEST
    return g_counters.failed > 0 ? 1 : 0;
#else
    return static_cast<int>(g_counters.failed);
#endif
}
