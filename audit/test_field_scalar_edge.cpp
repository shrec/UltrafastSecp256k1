// ============================================================================
// Field & Scalar Edge-Case Audit Test
// ============================================================================
// Exercises boundary conditions, carry propagation, reduction edge cases,
// and inverse correctness for FieldElement and Scalar — the two lowest-level
// algebraic types.  These are the "thin layer" that all higher-level
// cryptographic operations depend on.
//
// TESTS:
//   1.  Field reduction near p   (carry chain stress)
//   2.  Field inverse via safegcd (boundary + random)
//   3.  Field square vs mul consistency (large random)
//   4.  Field from_bytes / to_bytes roundtrip stress
//   5.  Field parse_bytes_strict boundary
//   6.  Scalar reduction near n  (carry chain stress)
//   7.  Scalar inverse via safegcd (boundary + random)
//   8.  Scalar negate / double-negate identity
//   9.  Scalar parse_bytes_strict / parse_bytes_strict_nonzero boundary
//  10.  Scalar bit() accessor exhaustive on known values
//  11.  ct_normalize_low_s boundary verification
//  12.  Field/Scalar cross-representation consistency (from_hex, from_limbs)
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>
#include <random>
#include <string>

#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/ct/ops.hpp"
#include "secp256k1/ct/scalar.hpp"

// Sanitizer-aware iteration scaling
#include "secp256k1/sanitizer_scale.hpp"

using namespace secp256k1;
using namespace secp256k1::fast;

static int g_pass = 0, g_fail = 0;
static const char* g_section = "";

#include "audit_check.hpp"

static std::mt19937_64 rng(0xF1E1D5CA'1A420001ULL);

static FieldElement random_fe() {
    std::array<uint8_t, 32> buf{};
    for (int i = 0; i < 4; ++i) {
        uint64_t v = rng();
        std::memcpy(buf.data() + static_cast<std::size_t>(i) * 8, &v, 8);
    }
    return FieldElement::from_bytes(buf);
}

static Scalar random_scalar() {
    std::array<uint8_t, 32> buf{};
    for (int i = 0; i < 4; ++i) {
        uint64_t v = rng();
        std::memcpy(buf.data() + static_cast<std::size_t>(i) * 8, &v, 8);
    }
    return Scalar::from_bytes(buf);
}

// secp256k1 prime p = 2^256 - 0x1000003D1
// p   = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
// p-1 = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2E
static FieldElement fe_p_minus_1() {
    return FieldElement::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2E");
}

// secp256k1 group order n
// n   = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
// n-1 = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140
static Scalar sc_n_minus_1() {
    return Scalar::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140");
}

// ============================================================================
//  TEST 1: Field reduction near p (carry chain stress)
// ============================================================================
static void test_field_reduction() {
    g_section = "field_reduction";
    std::printf("\n[1] Field reduction near p (carry chain stress)\n");

    const FieldElement zero = FieldElement::zero();
    const FieldElement one  = FieldElement::one();
    const FieldElement pm1  = fe_p_minus_1();

    // (p-1) + 1 = 0 mod p
    CHECK((pm1 + one) == zero, "(p-1)+1 == 0");

    // (p-1) + (p-1) = p-2 mod p
    FieldElement pm2 = FieldElement::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D");
    CHECK((pm1 + pm1) == pm2, "(p-1)+(p-1) == p-2");

    // 0 - 1 = p-1
    CHECK((zero - one) == pm1, "0-1 == p-1");

    // Carry through each limb boundary: add 1 to (2^64 - 1) patterns
    FieldElement carry_test = FieldElement::from_limbs({0xFFFFFFFFFFFFFFFFULL, 0, 0, 0});
    FieldElement carry_result = carry_test + one;
    auto limbs = carry_result.limbs();
    CHECK(limbs[0] == 0 && limbs[1] == 1, "carry propagation limb[0]->limb[1]");

    carry_test = FieldElement::from_limbs({0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0, 0});
    carry_result = carry_test + one;
    limbs = carry_result.limbs();
    CHECK(limbs[0] == 0 && limbs[1] == 0 && limbs[2] == 1,
          "carry propagation limb[0]->limb[1]->limb[2]");

    // Multiply (p-1) * 2 then add 2 — should wrap to 0
    FieldElement two = FieldElement::from_uint64(2);
    CHECK(pm1 * two + two == zero, "(p-1)*2+2 == 0");

    // Small values near p: p-k for k=1..10
    for (uint64_t k = 1; k <= 10; ++k) {
        FieldElement pk = pm1 - FieldElement::from_uint64(k - 1); // p-k
        FieldElement fk = FieldElement::from_uint64(k);
        CHECK(pk + fk == zero, "p-k + k == 0 for k=" + std::to_string(k));
    }

    std::printf("    -> OK\n");
}

// ============================================================================
//  TEST 2: Field inverse via safegcd (boundary + random)
// ============================================================================
static void test_field_inverse() {
    g_section = "field_inverse";
    std::printf("\n[2] Field inverse via safegcd\n");

    const FieldElement one = FieldElement::one();
    const FieldElement pm1 = fe_p_minus_1();

    // inv(1) = 1
    CHECK(one.inverse() == one, "inv(1)==1");

    // inv(p-1) = p-1  (since (p-1)=-1, inv(-1)=-1)
    CHECK(pm1.inverse() == pm1, "inv(p-1)==p-1");

    // inv(2) * 2 = 1
    FieldElement two = FieldElement::from_uint64(2);
    CHECK(two.inverse() * two == one, "inv(2)*2==1");

    // inv(inv(x)) = x for small values
    uint64_t test_vals[] = {2, 3, 5, 7, 11, 13, 42, 0xDEADBEEF, 0xCAFE'BABE'1234'5678ULL};
    for (auto v : test_vals) {
        FieldElement fe = FieldElement::from_uint64(v);
        CHECK(fe.inverse().inverse() == fe, "inv(inv(x))==x for v=" + std::to_string(v));
    }

    // inv(a) * a = 1 for random values
    const int N = SCALED(50, 10);
    for (int i = 0; i < N; ++i) {
        FieldElement a = random_fe();
        if (a == FieldElement::zero()) continue;
        FieldElement inv_a = a.inverse();
        CHECK(a * inv_a == one, "a*inv(a)==1 random #" + std::to_string(i));
    }

    // inverse_inplace consistency
    FieldElement x = FieldElement::from_uint64(9999);
    FieldElement x_inv = x.inverse();
    FieldElement x_copy = x;
    x_copy.inverse_inplace();
    CHECK(x_inv == x_copy, "inverse_inplace matches inverse");

    std::printf("    -> %d/%d OK\n", N, N);
}

// ============================================================================
//  TEST 3: Field square vs mul consistency (random)
// ============================================================================
static void test_field_square_mul() {
    g_section = "field_square_mul";
    std::printf("\n[3] Field square vs mul consistency\n");

    const int N = SCALED(80, 16);
    for (int i = 0; i < N; ++i) {
        FieldElement a = random_fe();
        FieldElement sq = a.square();
        FieldElement mul = a * a;
        CHECK(sq == mul, "a^2 == a*a random #" + std::to_string(i));
    }

    // square_inplace consistency
    for (int i = 0; i < 20; ++i) {
        FieldElement a = random_fe();
        FieldElement sq = a.square();
        FieldElement a_copy = a;
        a_copy.square_inplace();
        CHECK(sq == a_copy, "square_inplace consistency #" + std::to_string(i));
    }

    // Known: (p-1)^2 = 1
    FieldElement pm1 = fe_p_minus_1();
    CHECK(pm1.square() == FieldElement::one(), "(p-1)^2 == 1");

    std::printf("    -> %d/%d OK\n", N + 20, N + 20);
}

// ============================================================================
//  TEST 4: Field from_bytes / to_bytes roundtrip stress
// ============================================================================
static void test_field_bytes_roundtrip() {
    g_section = "field_bytes_roundtrip";
    std::printf("\n[4] Field from_bytes / to_bytes roundtrip\n");

    const int N = SCALED(100, 20);
    for (int i = 0; i < N; ++i) {
        FieldElement a = random_fe();
        auto bytes = a.to_bytes();
        FieldElement b = FieldElement::from_bytes(bytes);
        CHECK(a == b, "bytes roundtrip #" + std::to_string(i));
    }

    // to_bytes_into matches to_bytes
    for (int i = 0; i < 20; ++i) {
        FieldElement a = random_fe();
        auto bytes = a.to_bytes();
        std::array<uint8_t, 32> buf{};
        a.to_bytes_into(buf.data());
        CHECK(bytes == buf, "to_bytes_into consistency #" + std::to_string(i));
    }

    // Zero roundtrip
    auto z_bytes = FieldElement::zero().to_bytes();
    bool all_zero = true;
    for (auto b : z_bytes) if (b != 0) all_zero = false;
    CHECK(all_zero, "zero to_bytes all zeros");

    // One roundtrip
    auto o_bytes = FieldElement::one().to_bytes();
    CHECK(o_bytes[31] == 1, "one to_bytes last byte is 1");

    std::printf("    -> %d/%d OK\n", N + 20 + 2, N + 20 + 2);
}

// ============================================================================
//  TEST 5: Field parse_bytes_strict boundary
// ============================================================================
static void test_field_parse_strict() {
    g_section = "field_parse_strict";
    std::printf("\n[5] Field parse_bytes_strict boundary\n");

    FieldElement out;

    // p itself should fail (value == p is not < p)
    std::array<uint8_t, 32> p_bytes = {
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
        0xFF,0xFF,0xFF,0xFE,0xFF,0xFF,0xFC,0x2F
    };
    CHECK(!FieldElement::parse_bytes_strict(p_bytes, out), "p itself rejected");

    // p-1 should succeed
    std::array<uint8_t, 32> pm1_bytes = {
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
        0xFF,0xFF,0xFF,0xFE,0xFF,0xFF,0xFC,0x2E
    };
    CHECK(FieldElement::parse_bytes_strict(pm1_bytes, out), "p-1 accepted");
    CHECK(out == fe_p_minus_1(), "p-1 value correct");

    // p+1 should fail 
    std::array<uint8_t, 32> pp1_bytes = {
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
        0xFF,0xFF,0xFF,0xFE,0xFF,0xFF,0xFC,0x30
    };
    CHECK(!FieldElement::parse_bytes_strict(pp1_bytes, out), "p+1 rejected");

    // All 0xFF should fail (> p)
    std::array<uint8_t, 32> all_ff{};
    std::memset(all_ff.data(), 0xFF, 32);
    CHECK(!FieldElement::parse_bytes_strict(all_ff, out), "0xFF...FF rejected");

    // Zero should succeed
    std::array<uint8_t, 32> zero_bytes{};
    CHECK(FieldElement::parse_bytes_strict(zero_bytes, out), "zero accepted");
    CHECK(out == FieldElement::zero(), "zero value correct");

    // One should succeed
    std::array<uint8_t, 32> one_bytes{};
    one_bytes[31] = 1;
    CHECK(FieldElement::parse_bytes_strict(one_bytes, out), "one accepted");
    CHECK(out == FieldElement::one(), "one value correct");

    std::printf("    -> OK\n");
}

// ============================================================================
//  TEST 6: Scalar reduction near n (carry chain stress)
// ============================================================================
static void test_scalar_reduction() {
    g_section = "scalar_reduction";
    std::printf("\n[6] Scalar reduction near n\n");

    const Scalar zero = Scalar::zero();
    const Scalar one  = Scalar::one();
    const Scalar nm1  = sc_n_minus_1();

    // (n-1) + 1 = 0 mod n
    CHECK((nm1 + one) == zero, "(n-1)+1 == 0");

    // 0 - 1 = n-1
    CHECK((zero - one) == nm1, "0-1 == n-1");

    // (n-1) * (n-1) = 1  (since n-1 = -1)
    CHECK(nm1 * nm1 == one, "(n-1)*(n-1) == 1");

    // n-k + k = 0 for k=1..10
    for (uint64_t k = 1; k <= 10; ++k) {
        Scalar sk = Scalar::from_uint64(k);
        Scalar nmk = nm1 - sk + one; // (n-1) - k + 1 = n-k
        CHECK(nmk + sk == zero, "n-k + k == 0 for k=" + std::to_string(k));
    }

    // Carry chain: from_limbs with values at limb boundaries
    Scalar carry = Scalar::from_limbs({0xFFFFFFFFFFFFFFFFULL, 0, 0, 0});
    Scalar carry_p1 = carry + one;
    auto limbs = carry_p1.limbs();
    CHECK(limbs[0] == 0 && limbs[1] == 1, "scalar carry limb[0]->limb[1]");

    std::printf("    -> OK\n");
}

// ============================================================================
//  TEST 7: Scalar inverse via safegcd (boundary + random)
// ============================================================================
static void test_scalar_inverse() {
    g_section = "scalar_inverse";
    std::printf("\n[7] Scalar inverse via safegcd\n");

    const Scalar one = Scalar::one();
    const Scalar nm1 = sc_n_minus_1();

    // inv(1) = 1
    CHECK(one.inverse() == one, "inv(1)==1");

    // inv(n-1) = n-1
    CHECK(nm1.inverse() == nm1, "inv(n-1)==n-1");

    // inv(2) * 2 = 1
    Scalar two = Scalar::from_uint64(2);
    CHECK(two.inverse() * two == one, "inv(2)*2==1");

    // inv(inv(x)) = x
    uint64_t test_vals[] = {2, 3, 7, 42, 0xDEADBEEF};
    for (auto v : test_vals) {
        Scalar s = Scalar::from_uint64(v);
        CHECK(s.inverse().inverse() == s, "inv(inv(x))==x for v=" + std::to_string(v));
    }

    // Random: a * inv(a) = 1
    const int N = SCALED(40, 8);
    for (int i = 0; i < N; ++i) {
        Scalar a = random_scalar();
        if (a.is_zero()) continue;
        CHECK(a * a.inverse() == one, "a*inv(a)==1 random #" + std::to_string(i));
    }

    std::printf("    -> %d/%d OK\n", N, N);
}

// ============================================================================
//  TEST 8: Scalar negate / double-negate identity
// ============================================================================
static void test_scalar_negate() {
    g_section = "scalar_negate";
    std::printf("\n[8] Scalar negate / double-negate\n");

    const Scalar zero = Scalar::zero();

    // negate(0) = 0
    CHECK(zero.negate() == zero, "negate(0)==0");

    // negate(1) = n-1
    CHECK(Scalar::one().negate() == sc_n_minus_1(), "negate(1)==n-1");

    // a + negate(a) = 0
    const int N = SCALED(50, 10);
    for (int i = 0; i < N; ++i) {
        Scalar a = random_scalar();
        CHECK(a + a.negate() == zero, "a+neg(a)==0 #" + std::to_string(i));
    }

    // negate(negate(a)) = a
    for (int i = 0; i < 20; ++i) {
        Scalar a = random_scalar();
        CHECK(a.negate().negate() == a, "neg(neg(a))==a #" + std::to_string(i));
    }

    std::printf("    -> %d/%d OK\n", N + 20, N + 20);
}

// ============================================================================
//  TEST 9: Scalar parse_bytes_strict boundary
// ============================================================================
static void test_scalar_parse_strict() {
    g_section = "scalar_parse_strict";
    std::printf("\n[9] Scalar parse_bytes_strict boundary\n");

    Scalar out;

    // n itself should fail
    std::array<uint8_t, 32> n_bytes = {
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
        0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B,
        0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x41
    };
    CHECK(!Scalar::parse_bytes_strict(n_bytes, out), "n itself rejected");

    // n-1 should succeed
    std::array<uint8_t, 32> nm1_bytes = {
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
        0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B,
        0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x40
    };
    CHECK(Scalar::parse_bytes_strict(nm1_bytes, out), "n-1 accepted");
    CHECK(out == sc_n_minus_1(), "n-1 value correct");

    // All 0xFF should fail
    std::array<uint8_t, 32> all_ff{};
    std::memset(all_ff.data(), 0xFF, 32);
    CHECK(!Scalar::parse_bytes_strict(all_ff, out), "0xFF...FF rejected");

    // Zero should succeed (for parse_bytes_strict)
    std::array<uint8_t, 32> zero_bytes{};
    CHECK(Scalar::parse_bytes_strict(zero_bytes, out), "zero accepted by parse_strict");

    // Zero should FAIL for parse_bytes_strict_nonzero
    CHECK(!Scalar::parse_bytes_strict_nonzero(zero_bytes, out), "zero rejected by parse_strict_nonzero");

    // One should succeed for both
    std::array<uint8_t, 32> one_bytes{};
    one_bytes[31] = 1;
    CHECK(Scalar::parse_bytes_strict(one_bytes, out), "one accepted by parse_strict");
    CHECK(Scalar::parse_bytes_strict_nonzero(one_bytes, out), "one accepted by parse_strict_nonzero");

    std::printf("    -> OK\n");
}

// ============================================================================
//  TEST 10: Scalar bit() accessor verification
// ============================================================================
static void test_scalar_bit() {
    g_section = "scalar_bit";
    std::printf("\n[10] Scalar bit() accessor\n");

    // bit(0) of 1 is 1
    CHECK(Scalar::one().bit(0) == 1, "one.bit(0)==1");

    // bit(1) of 1 is 0
    CHECK(Scalar::one().bit(1) == 0, "one.bit(1)==0");

    // bit(0) of 2 is 0, bit(1) is 1
    Scalar two = Scalar::from_uint64(2);
    CHECK(two.bit(0) == 0, "two.bit(0)==0");
    CHECK(two.bit(1) == 1, "two.bit(1)==1");

    // All bits of zero are 0
    for (unsigned b = 0; b < 256; ++b) {
        CHECK(Scalar::zero().bit(b) == 0, "zero.bit(" + std::to_string(b) + ")==0");
    }

    // Check known value: 0xFF = 11111111 in binary
    Scalar ff = Scalar::from_uint64(0xFF);
    for (unsigned b = 0; b < 8; ++b) {
        CHECK(ff.bit(b) == 1, "0xFF.bit(" + std::to_string(b) + ")==1");
    }
    CHECK(ff.bit(8) == 0, "0xFF.bit(8)==0");

    // Cross-verify: reconstruct value from bits for random scalars
    for (int trial = 0; trial < 10; ++trial) {
        Scalar s = random_scalar();
        auto bytes = s.to_bytes();
        for (unsigned b = 0; b < 256; ++b) {
            unsigned byte_idx = 31 - (b / 8);  // big-endian
            unsigned bit_idx  = b % 8;
            uint8_t expected = (bytes[byte_idx] >> bit_idx) & 1;
            CHECK(s.bit(b) == expected, "bit cross-verify trial=" + std::to_string(trial) + " b=" + std::to_string(b));
        }
    }

    std::printf("    -> OK\n");
}

// ============================================================================
//  TEST 11: ct_normalize_low_s verification
// ============================================================================
static void test_ct_normalize_low_s() {
    g_section = "ct_normalize_low_s";
    std::printf("\n[11] CT normalize_low_s\n");

    using namespace secp256k1;
    using fast::Scalar;

    // half_n = (n-1)/2: any s <= half_n is already low-S
    Scalar half_n = Scalar::from_hex("7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A0");
    Scalar one = Scalar::one();

    // Helper: build ECDSASignature from (r=1, s) and normalize
    auto make_sig = [&](const Scalar& r, const Scalar& s) -> ECDSASignature {
        std::array<uint8_t, 64> buf{};
        auto rb = r.to_bytes(); auto sb = s.to_bytes();
        std::memcpy(buf.data(),      rb.data(), 32);
        std::memcpy(buf.data() + 32, sb.data(), 32);
        return ECDSASignature::from_compact(buf);
    };

    // [A] half_n is low-S → unchanged
    {
        auto sig = make_sig(one, half_n);
        auto norm = secp256k1::ct::ct_normalize_low_s(sig);
        CHECK(norm.s == half_n, "half_n already low-s");
    }

    // [B] half_n + 1 is high-S → should become n - (half_n+1) = half_n
    {
        Scalar high = half_n + one;
        auto sig = make_sig(one, high);
        auto norm = secp256k1::ct::ct_normalize_low_s(sig);
        Scalar expected = high.negate();
        CHECK(norm.s == expected, "half_n+1 normalized to negate");
    }

    // [C] s=1 is low-S → unchanged
    {
        auto sig = make_sig(one, one);
        auto norm = secp256k1::ct::ct_normalize_low_s(sig);
        CHECK(norm.s == one, "s=1 low-s unchanged");
    }

    // [D] s=n-1 → high-S, normalized to 1
    {
        Scalar nm1 = sc_n_minus_1();
        auto sig = make_sig(one, nm1);
        auto norm = secp256k1::ct::ct_normalize_low_s(sig);
        CHECK(norm.s == one, "s=n-1 normalized to 1");
    }

    // [E] r field must be preserved unchanged
    {
        Scalar r = Scalar::from_hex("DEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEF");
        auto sig = make_sig(r, one);
        auto norm = secp256k1::ct::ct_normalize_low_s(sig);
        CHECK(norm.r == r, "r preserved by normalize_low_s");
    }

    // [F] Random: normalized s <= half_n; double-normalize is idempotent
    for (int i = 0; i < SCALED(30, 6); ++i) {
        Scalar s = random_scalar();
        if (s.is_zero()) continue;
        auto sig  = make_sig(one, s);
        auto norm = secp256k1::ct::ct_normalize_low_s(sig);
        // norm.s == s or norm.s == s.negate()
        CHECK(norm.s == s || norm.s == s.negate(),
              "normalized is s or -s #" + std::to_string(i));
        // idempotent
        auto sig2 = make_sig(one, norm.s);
        auto norm2 = secp256k1::ct::ct_normalize_low_s(sig2);
        CHECK(norm2.s == norm.s, "double normalize idempotent #" + std::to_string(i));
    }

    std::printf("    -> OK\n");
}

// ============================================================================
//  TEST 12: Cross-representation consistency
// ============================================================================
static void test_cross_representation() {
    g_section = "cross_repr";
    std::printf("\n[12] Cross-representation consistency\n");

    // Field: from_hex and from_limbs agree
    FieldElement fe_hex = FieldElement::from_hex("0000000000000000000000000000000000000000000000000000000000000001");
    CHECK(fe_hex == FieldElement::one(), "from_hex(1)==one()");

    FieldElement fe_limbs = FieldElement::from_limbs({1, 0, 0, 0});
    CHECK(fe_limbs == FieldElement::one(), "from_limbs({1,0,0,0})==one()");

    // Scalar: from_hex and from_limbs agree
    Scalar sc_hex = Scalar::from_hex("0000000000000000000000000000000000000000000000000000000000000001");
    CHECK(sc_hex == Scalar::one(), "scalar from_hex(1)==one()");

    Scalar sc_limbs = Scalar::from_limbs({1, 0, 0, 0});
    CHECK(sc_limbs == Scalar::one(), "scalar from_limbs({1,0,0,0})==one()");

    // Field: from_uint64 = from_limbs({v,0,0,0})
    for (uint64_t v : {0ULL, 1ULL, 42ULL, 0xFFFFFFFFFFFFFFFFULL}) {
        FieldElement a = FieldElement::from_uint64(v);
        FieldElement b = FieldElement::from_limbs({v, 0, 0, 0});
        CHECK(a == b, "from_uint64 == from_limbs for v=" + std::to_string(v));
    }

    // Scalar: hex roundtrip
    for (int i = 0; i < 20; ++i) {
        Scalar s = random_scalar();
        std::string hex = s.to_hex();
        Scalar s2 = Scalar::from_hex(hex);
        CHECK(s == s2, "scalar hex roundtrip #" + std::to_string(i));
    }

    // Field: hex roundtrip
    for (int i = 0; i < 20; ++i) {
        FieldElement fe = random_fe();
        std::string hex = fe.to_hex();
        FieldElement fe2 = FieldElement::from_hex(hex);
        CHECK(fe == fe2, "field hex roundtrip #" + std::to_string(i));
    }

    // Scalar: is_even parity check
    CHECK(Scalar::zero().is_even(), "zero is even");
    CHECK(!Scalar::one().is_even(), "one is not even");
    CHECK(Scalar::from_uint64(2).is_even(), "2 is even");
    CHECK(!Scalar::from_uint64(3).is_even(), "3 is not even");

    std::printf("    -> OK\n");
}

// ============================================================================
//  MAIN
// ============================================================================

#ifdef STANDALONE_TEST
int main() {
#else
int test_field_scalar_edge_run() {
#endif
    std::printf("============================================================\n");
    std::printf("  Field & Scalar Edge-Case Audit Test\n");
    std::printf("============================================================\n");

    test_field_reduction();
    test_field_inverse();
    test_field_square_mul();
    test_field_bytes_roundtrip();
    test_field_parse_strict();
    test_scalar_reduction();
    test_scalar_inverse();
    test_scalar_negate();
    test_scalar_parse_strict();
    test_scalar_bit();
    test_ct_normalize_low_s();
    test_cross_representation();

    std::printf("\n  [field_scalar_edge] %d passed, %d failed\n\n", g_pass, g_fail);
    std::printf("============================================================\n");
    std::printf("  Result: %s\n", g_fail == 0 ? "ALL PASSED" : "FAILURES DETECTED");
    std::printf("============================================================\n");
    return g_fail == 0 ? 0 : 1;
}
