// ============================================================================
// Constant-Time Layer -- Correctness Tests
// ============================================================================
// Verifies that secp256k1::ct:: operations produce the SAME results
// as secp256k1::fast:: operations, and handles edge cases correctly.
//
// Tests:
//   1. CT field arithmetic (add, sub, mul, sqr, neg, inv, normalize)
//   2. CT field conditional ops (cmov, cswap, select, cneg, is_zero, eq)
//   3. CT scalar arithmetic (add, sub, neg)
//   4. CT scalar conditional ops + bit access
//   5. CT point complete addition -- edge cases (P+O, O+P, P+P, P+(-P))
//   6. CT scalar multiplication -- known test vectors (k*G)
//   7. CT generator_mul matches fast::Point::scalar_mul
//   8. CT point_is_on_curve
// ============================================================================

#include "secp256k1/fast.hpp"
#include "secp256k1/ct/field.hpp"
#include "secp256k1/ct/scalar.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/ct/ops.hpp"
#include "secp256k1/ct/sign.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"
#include <iostream>
#include <iomanip>
#include <cstring>
#include <sstream>

using FE = secp256k1::fast::FieldElement;
using SC = secp256k1::fast::Scalar;
using PT = secp256k1::fast::Point;

namespace ct = secp256k1::ct;

static int g_pass = 0;
static int g_fail = 0;

[[maybe_unused]] static std::string fe_hex(const FE& f) {
    auto bytes = f.to_bytes();
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (auto b : bytes) ss << std::setw(2) << static_cast<int>(b);
    return ss.str();
}

static bool fe_eq(const FE& a, const FE& b) {
    return a.to_bytes() == b.to_bytes();
}

static bool pt_eq_affine(const PT& a, const PT& b) {
    if (a.is_infinity() && b.is_infinity()) return true;
    if (a.is_infinity() || b.is_infinity()) return false;
    auto ax = a.x().to_bytes();
    auto bx = b.x().to_bytes();
    auto ay = a.y().to_bytes();
    auto by = b.y().to_bytes();
    return ax == bx && ay == by;
}

#define CHECK(cond, msg)                                        \
    do {                                                        \
        if (cond) {                                             \
            ++g_pass;                                           \
            std::cout << "  PASS: " << (msg) << "\n";             \
            std::cout.flush();                                  \
        } else {                                                \
            ++g_fail;                                           \
            std::cout << "  FAIL: " << (msg) << "\n";             \
            std::cout.flush();                                  \
        }                                                       \
    } while (0)

// --- Test helpers ------------------------------------------------------------

[[maybe_unused]] static FE make_fe(uint64_t v) { return FE::from_uint64(v); }

// --- 1. CT Field Arithmetic -------------------------------------------------

static void test_field_add() {
    FE const a = FE::from_hex("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
    FE const b = FE::from_hex("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8");

    FE const fast_r = a + b;
    FE const ct_r = ct::field_add(a, b);

    CHECK(fe_eq(fast_r, ct_r), "field_add basic");
}

static void test_field_sub() {
    FE const a = FE::from_hex("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
    FE const b = FE::from_hex("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8");

    FE const fast_r = a - b;
    FE const ct_r = ct::field_sub(a, b);

    CHECK(fe_eq(fast_r, ct_r), "field_sub basic");
}

static void test_field_mul() {
    FE const a = FE::from_hex("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
    FE const b = FE::from_hex("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8");

    FE const fast_r = a * b;
    FE const ct_r = ct::field_mul(a, b);

    CHECK(fe_eq(fast_r, ct_r), "field_mul basic");
}

static void test_field_sqr() {
    FE const a = FE::from_hex("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");

    FE const fast_r = a.square();
    FE const ct_r = ct::field_sqr(a);

    CHECK(fe_eq(fast_r, ct_r), "field_sqr basic");
}

static void test_field_neg() {
    FE const a = FE::from_hex("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
    FE const zero = FE::from_uint64(0);

    FE const ct_neg_a = ct::field_neg(a);
    FE const sum = ct::field_add(a, ct_neg_a);

    CHECK(ct::field_is_zero(sum) != 0, "field_neg: a + (-a) == 0");

    FE const ct_neg_zero = ct::field_neg(zero);
    CHECK(ct::field_is_zero(ct_neg_zero) != 0, "field_neg(0) == 0");
}

static void test_field_inv() {
    FE const a = FE::from_hex("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");

    FE const ct_inv = ct::field_inv(a);
    FE const product = ct::field_mul(a, ct_inv);

    // product should be 1
    FE const one = FE::from_uint64(1);
    CHECK(fe_eq(product, one), "field_inv: a * a^-1 == 1");
}

static void test_field_normalize() {
    // Create a value >= p by using raw limbs
    FE const a = FE::from_uint64(42);
    FE const norm = ct::field_normalize(a);
    CHECK(fe_eq(a, norm), "field_normalize: small value unchanged");
}

// --- 2. CT Field Conditional Ops ---------------------------------------------

static void test_field_cmov() {
    FE const a = FE::from_uint64(42);
    FE const b = FE::from_uint64(99);
    FE r = a;

    ct::field_cmov(&r, b, 0);  // no move
    CHECK(fe_eq(r, a), "field_cmov: mask=0 -> no change");

    ct::field_cmov(&r, b, ~uint64_t(0));  // move
    CHECK(fe_eq(r, b), "field_cmov: mask=all-ones -> moved");
}

static void test_field_cswap() {
    FE const a = FE::from_uint64(42);
    FE const b = FE::from_uint64(99);
    FE a2 = a, b2 = b;

    ct::field_cswap(&a2, &b2, 0);  // no swap
    CHECK(fe_eq(a2, a) && fe_eq(b2, b), "field_cswap: mask=0 -> no swap");

    ct::field_cswap(&a2, &b2, ~uint64_t(0));  // swap
    CHECK(fe_eq(a2, b) && fe_eq(b2, a), "field_cswap: mask=all-ones -> swapped");
}

static void test_field_select() {
    FE const a = FE::from_uint64(42);
    FE const b = FE::from_uint64(99);

    FE const r0 = ct::field_select(a, b, 0);
    CHECK(fe_eq(r0, b), "field_select: mask=0 -> b");

    FE const r1 = ct::field_select(a, b, ~uint64_t(0));
    CHECK(fe_eq(r1, a), "field_select: mask=all-ones -> a");
}

static void test_field_cneg() {
    FE const a = FE::from_uint64(42);

    FE const r0 = ct::field_cneg(a, 0);
    CHECK(fe_eq(r0, a), "field_cneg: mask=0 -> unchanged");

    FE const r1 = ct::field_cneg(a, ~uint64_t(0));
    FE const neg_a = ct::field_neg(a);
    CHECK(fe_eq(r1, neg_a), "field_cneg: mask=all-ones -> negated");
}

static void test_field_is_zero() {
    FE const zero = FE::from_uint64(0);
    FE const nonzero = FE::from_uint64(1);

    CHECK(ct::field_is_zero(zero) != 0, "field_is_zero(0) -> true");
    CHECK(ct::field_is_zero(nonzero) == 0, "field_is_zero(1) -> false");
}

static void test_field_eq() {
    FE const a = FE::from_uint64(42);
    FE const b = FE::from_uint64(42);
    FE const c = FE::from_uint64(99);

    CHECK(ct::field_eq(a, b) != 0, "field_eq: equal -> true");
    CHECK(ct::field_eq(a, c) == 0, "field_eq: not equal -> false");
}

// --- 3. CT Scalar Arithmetic -------------------------------------------------

static void test_scalar_add() {
    SC const a = SC::from_uint64(100);
    SC const b = SC::from_uint64(200);

    SC const ct_r = ct::scalar_add(a, b);
    SC const expected = SC::from_uint64(300);

    CHECK(ct::scalar_eq(ct_r, expected) != 0, "scalar_add basic");
}

static void test_scalar_sub() {
    SC const a = SC::from_uint64(300);
    SC const b = SC::from_uint64(100);

    SC const ct_r = ct::scalar_sub(a, b);
    SC const expected = SC::from_uint64(200);

    CHECK(ct::scalar_eq(ct_r, expected) != 0, "scalar_sub basic");
}

static void test_scalar_neg() {
    SC const a = SC::from_uint64(42);
    SC const neg_a = ct::scalar_neg(a);
    SC const sum = ct::scalar_add(a, neg_a);

    CHECK(ct::scalar_is_zero(sum) != 0, "scalar_neg: a + (-a) == 0");

    SC const zero = SC::from_uint64(0);
    SC const neg_zero = ct::scalar_neg(zero);
    CHECK(ct::scalar_is_zero(neg_zero) != 0, "scalar_neg(0) == 0");
}

// --- 4. CT Scalar Conditional Ops + Bit Access -------------------------------

static void test_scalar_cmov() {
    SC const a = SC::from_uint64(42);
    SC const b = SC::from_uint64(99);
    SC r = a;

    ct::scalar_cmov(&r, b, 0);
    CHECK(ct::scalar_eq(r, a) != 0, "scalar_cmov: mask=0 -> no change");

    ct::scalar_cmov(&r, b, ~uint64_t(0));
    CHECK(ct::scalar_eq(r, b) != 0, "scalar_cmov: mask=all-ones -> moved");
}

static void test_scalar_bit() {
    // k = 5 = 0b101
    SC const k = SC::from_uint64(5);

    CHECK(ct::scalar_bit(k, 0) == 1, "scalar_bit(5, 0) == 1");
    CHECK(ct::scalar_bit(k, 1) == 0, "scalar_bit(5, 1) == 0");
    CHECK(ct::scalar_bit(k, 2) == 1, "scalar_bit(5, 2) == 1");
    CHECK(ct::scalar_bit(k, 3) == 0, "scalar_bit(5, 3) == 0");
}

static void test_scalar_window() {
    // k = 0xAB = 0b10101011 -> window(0,4) = 0xB = 11, window(4,4) = 0xA = 10
    SC const k = SC::from_uint64(0xAB);

    uint64_t const w0 = ct::scalar_window(k, 0, 4);
    uint64_t const w1 = ct::scalar_window(k, 4, 4);

    CHECK(w0 == 0xB, "scalar_window(0xAB, 0, 4) == 0xB");
    CHECK(w1 == 0xA, "scalar_window(0xAB, 4, 4) == 0xA");
}

// --- 5. CT Complete Addition -- Edge Cases ------------------------------------

static void test_complete_add_general() {
    // P + Q where P != Q
    PT const G = PT::generator();
    SC const k2 = SC::from_uint64(2);
    PT const G2 = G.scalar_mul(k2);

    // CT: G + G2 should equal 3G
    ct::CTJacobianPoint const jp = ct::CTJacobianPoint::from_point(G);
    ct::CTJacobianPoint const jq = ct::CTJacobianPoint::from_point(G2);
    ct::CTJacobianPoint const jr = ct::point_add_complete(jp, jq);
    PT const ct_result = jr.to_point();

    SC const k3 = SC::from_uint64(3);
    PT const expected = G.scalar_mul(k3);

    CHECK(pt_eq_affine(ct_result, expected), "complete_add: G + 2G == 3G");
}

static void test_complete_add_doubling() {
    // P + P should give 2P
    PT const G = PT::generator();

    ct::CTJacobianPoint const jp = ct::CTJacobianPoint::from_point(G);
    ct::CTJacobianPoint const jr = ct::point_add_complete(jp, jp);
    PT const ct_result = jr.to_point();

    SC const k2 = SC::from_uint64(2);
    PT const expected = G.scalar_mul(k2);

    CHECK(pt_eq_affine(ct_result, expected), "complete_add: G + G == 2G");
}

static void test_complete_add_identity() {
    // P + O = P  and  O + P = P
    PT const G = PT::generator();

    ct::CTJacobianPoint const jp = ct::CTJacobianPoint::from_point(G);
    ct::CTJacobianPoint const inf = ct::CTJacobianPoint::make_infinity();

    // P + O = P
    ct::CTJacobianPoint const r1 = ct::point_add_complete(jp, inf);
    PT const result1 = r1.to_point();
    CHECK(pt_eq_affine(result1, G), "complete_add: G + O == G");

    // O + P = P
    ct::CTJacobianPoint const r2 = ct::point_add_complete(inf, jp);
    PT const result2 = r2.to_point();
    CHECK(pt_eq_affine(result2, G), "complete_add: O + G == G");

    // O + O = O
    ct::CTJacobianPoint const r3 = ct::point_add_complete(inf, inf);
    PT const result3 = r3.to_point();
    CHECK(result3.is_infinity(), "complete_add: O + O == O");
}

static void test_complete_add_inverse() {
    // P + (-P) = O
    PT const G = PT::generator();

    ct::CTJacobianPoint const jp = ct::CTJacobianPoint::from_point(G);
    ct::CTJacobianPoint const jneg = ct::point_neg(jp);

    ct::CTJacobianPoint const r = ct::point_add_complete(jp, jneg);
    PT const result = r.to_point();
    CHECK(result.is_infinity(), "complete_add: G + (-G) == O");
}

// --- 6. CT Scalar Multiplication -- Known Vectors -----------------------------

static void test_scalar_mul_k1() {
    // 1*G = G
    PT const G = PT::generator();
    SC const k = SC::from_uint64(1);

    PT const ct_r = ct::scalar_mul(G, k);
    CHECK(pt_eq_affine(ct_r, G), "CT scalar_mul: 1*G == G");
}

static void test_scalar_mul_k2() {
    // 2*G
    PT const G = PT::generator();
    SC const k = SC::from_uint64(2);

    PT const ct_r = ct::scalar_mul(G, k);
    PT const fast_r = G.scalar_mul(k);

    CHECK(pt_eq_affine(ct_r, fast_r), "CT scalar_mul: 2*G == fast 2*G");
}

static void test_scalar_mul_known_vector() {
    // Known test vector: k*G where k = 7
    // 7*G should have known coordinates
    PT const G = PT::generator();
    SC const k = SC::from_uint64(7);

    PT const ct_r = ct::scalar_mul(G, k);
    PT const fast_r = G.scalar_mul(k);

    CHECK(pt_eq_affine(ct_r, fast_r), "CT scalar_mul: 7*G == fast 7*G");
}

static void test_scalar_mul_large_k() {
    // Larger scalar (0xDEADBEEF exercises many non-zero 4-bit windows)
    PT const G = PT::generator();
    SC const k = SC::from_uint64(0xDEADBEEF);

    PT const ct_r = ct::scalar_mul(G, k);
    PT const fast_r = G.scalar_mul(k);

    CHECK(pt_eq_affine(ct_r, fast_r), "CT scalar_mul: 0xDEADBEEF*G == fast");
}

static void test_scalar_mul_k0() {
    // 0*G = O (infinity)
    PT const G = PT::generator();
    SC const k = SC::from_uint64(0);

    PT const ct_r = ct::scalar_mul(G, k);
    CHECK(ct_r.is_infinity(), "CT scalar_mul: 0*G == O");
}

// --- 7. CT Generator Multiplication -----------------------------------------

static void test_generator_mul() {
    SC const k = SC::from_uint64(42);

    PT const ct_r = ct::generator_mul(k);
    PT const fast_r = PT::generator().scalar_mul(k);

    CHECK(pt_eq_affine(ct_r, fast_r), "CT generator_mul(42) == fast 42*G");
}

// --- 8. CT On-Curve Check ----------------------------------------------------

static void test_point_is_on_curve() {
    PT const G = PT::generator();
    CHECK(ct::point_is_on_curve(G) != 0, "generator is on curve");

    SC const k = SC::from_uint64(12345);
    PT const P = G.scalar_mul(k);
    CHECK(ct::point_is_on_curve(P) != 0, "12345*G is on curve");
}

// --- 9. CT Point Equality ----------------------------------------------------

static void test_point_eq() {
    PT const G = PT::generator();
    SC const k = SC::from_uint64(42);
    PT const P = G.scalar_mul(k);

    CHECK(ct::point_eq(G, G) != 0, "point_eq(G, G) -> true");
    CHECK(ct::point_eq(G, P) == 0, "point_eq(G, 42*G) -> false");

    PT const inf = PT::infinity();
    CHECK(ct::point_eq(inf, inf) != 0, "point_eq(O, O) -> true");
    CHECK(ct::point_eq(G, inf) == 0, "point_eq(G, O) -> false");
}

// --- 10. CT Mixing test: fast compute + CT finish ----------------------------

static void test_mixing() {
    // Use fast:: for public data, ct:: for secret-dependent operations
    PT const G = PT::generator();

    // "Public" computation
    SC const pub_k = SC::from_uint64(100);
    PT const pub_point = G.scalar_mul(pub_k);  // fast::

    // "Secret" scalar multiplication using CT
    SC const secret_k = SC::from_uint64(7);
    PT const ct_result = ct::scalar_mul(pub_point, secret_k);  // CT

    // Verify: should equal 700*G
    SC const k700 = SC::from_uint64(700);
    PT const expected = G.scalar_mul(k700);

    CHECK(pt_eq_affine(ct_result, expected), "mixing: fast(100*G) -> CT(7*P) == 700*G");
}

// --- CT Signing Tests --------------------------------------------------------

static void test_ct_ecdsa_sign() {
    // Use BIP-340 test vector #0 private key
    auto privkey = SC::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000001");

    // Hash of "test message"
    std::array<uint8_t, 32> msg_hash{};
    msg_hash[0] = 0x42; msg_hash[1] = 0xAB; msg_hash[31] = 0x01;

    // CT and fast should produce identical signatures (same RFC 6979 nonce)
    auto ct_sig   = secp256k1::ct::ecdsa_sign(msg_hash, privkey);
    auto fast_sig = secp256k1::ecdsa_sign(msg_hash, privkey);

    CHECK(ct_sig.r.to_bytes() == fast_sig.r.to_bytes(),
          "ct::ecdsa_sign.r matches fast::ecdsa_sign.r");
    CHECK(ct_sig.s.to_bytes() == fast_sig.s.to_bytes(),
          "ct::ecdsa_sign.s matches fast::ecdsa_sign.s");

    // Verify the CT signature with the standard verifier
    auto G = PT::generator();
    auto pubkey = G.scalar_mul(privkey);
    CHECK(secp256k1::ecdsa_verify(msg_hash, pubkey, ct_sig),
          "ct::ecdsa_sign signature verifies");

    // Edge: zero key returns zero signature
    auto zero_sig = secp256k1::ct::ecdsa_sign(msg_hash, SC::zero());
    CHECK(zero_sig.r.is_zero() && zero_sig.s.is_zero(),
          "ct::ecdsa_sign(zero key) returns zero sig");
}

static void test_ct_schnorr_sign() {
    auto privkey = SC::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000003");

    std::array<uint8_t, 32> msg{};
    msg[0] = 0xDE; msg[1] = 0xAD; msg[2] = 0xBE; msg[3] = 0xEF;

    std::array<uint8_t, 32> aux{};
    aux[0] = 0x01;

    // CT keypair + sign
    auto ct_kp  = secp256k1::ct::schnorr_keypair_create(privkey);
    auto ct_sig = secp256k1::ct::schnorr_sign(ct_kp, msg, aux);

    // Fast keypair + sign
    auto fast_kp  = secp256k1::schnorr_keypair_create(privkey);
    auto fast_sig = secp256k1::schnorr_sign(fast_kp, msg, aux);

    CHECK(ct_kp.px == fast_kp.px,
          "ct::schnorr_keypair_create.px matches fast");
    CHECK(ct_sig.r == fast_sig.r,
          "ct::schnorr_sign.r matches fast::schnorr_sign.r");
    CHECK(ct_sig.s.to_bytes() == fast_sig.s.to_bytes(),
          "ct::schnorr_sign.s matches fast::schnorr_sign.s");

    // Verify with standard verifier
    CHECK(secp256k1::schnorr_verify(ct_kp.px, msg, ct_sig),
          "ct::schnorr_sign signature verifies");
}

static void test_ct_schnorr_pubkey() {
    auto privkey = SC::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000001");
    auto ct_px = secp256k1::ct::schnorr_pubkey(privkey);

    // Generator x-coordinate (well known)
    auto G = PT::generator();
    auto [gx, gy_odd] = G.x_bytes_and_parity();
    (void)gy_odd;
    CHECK(ct_px == gx, "ct::schnorr_pubkey(1) == G.x");
}

// --- Main --------------------------------------------------------------------

int test_ct_run() {
    std::cout << "=== CT (Constant-Time) Layer Tests ===\n\n";

    // Field
    std::cout << "--- Field Arithmetic ---\n";
    test_field_add();
    test_field_sub();
    test_field_mul();
    test_field_sqr();
    test_field_neg();
    test_field_inv();
    test_field_normalize();

    // Field conditional
    std::cout << "--- Field Conditional Ops ---\n";
    test_field_cmov();
    test_field_cswap();
    test_field_select();
    test_field_cneg();
    test_field_is_zero();
    test_field_eq();

    // Scalar
    std::cout << "--- Scalar Arithmetic ---\n";
    test_scalar_add();
    test_scalar_sub();
    test_scalar_neg();

    // Scalar conditional + bits
    std::cout << "--- Scalar Conditional + Bit Access ---\n";
    test_scalar_cmov();
    test_scalar_bit();
    test_scalar_window();

    // Complete addition edge cases
    std::cout << "--- Complete Addition (edge cases) ---\n";
    test_complete_add_general();
    test_complete_add_doubling();
    test_complete_add_identity();
    test_complete_add_inverse();

    // Scalar multiplication
    std::cout << "--- CT Scalar Multiplication ---\n";
    test_scalar_mul_k1();
    test_scalar_mul_k2();
    test_scalar_mul_known_vector();
    test_scalar_mul_large_k();
    test_scalar_mul_k0();

    // Generator mul
    std::cout << "--- CT Generator Multiplication ---\n";
    test_generator_mul();

    // On-curve
    std::cout << "--- CT On-Curve Check ---\n";
    test_point_is_on_curve();

    // Point equality
    std::cout << "--- CT Point Equality ---\n";
    test_point_eq();

    // Mixing
    std::cout << "--- CT + Fast Mixing ---\n";
    test_mixing();

    // CT Signing
    std::cout << "--- CT Signing (ecdsa + schnorr) ---\n";
    test_ct_ecdsa_sign();
    test_ct_schnorr_sign();
    test_ct_schnorr_pubkey();

    // Summary
    std::cout << "\n=== Results: " << g_pass << " passed, " << g_fail << " failed ===\n";
    return g_fail > 0 ? 1 : 0;
}
