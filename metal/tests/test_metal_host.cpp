// =============================================================================
// UltrafastSecp256k1 Metal -- Host-Side Type Tests (Cross-Platform)
// =============================================================================
// Verifies Metal host types, hex conversion, type bridges, and test vectors.
// Pure C++20 -- runs on ANY platform (Windows, Linux, macOS) without Metal GPU.
//
// CTest: metal_host_test
// =============================================================================

#include "host_helpers.h"
#include "secp256k1/types.hpp"
#include "../../audit/test_vectors.hpp"

#include <iostream>
#include <cstring>

// ============================================================
// Test infrastructure (matches project convention)
// ============================================================

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "FAIL: " << (msg) << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        ++g_fail; \
    } else { ++g_pass; } \
} while(0)

using namespace secp256k1::metal;
using namespace secp256k1::test_vectors;

// Resolve ambiguity: use test_vectors::hex_equal as the canonical comparison
using secp256k1::test_vectors::hex_equal;

// ============================================================
// Test: FieldElement hex round-trip
// ============================================================

static void test_field_hex_roundtrip() {
    std::cout << "  [Field] Hex round-trip...\n";

    // Generator X
    const char* gx_hex = GENERATOR_X;
    auto fe = HostFieldElement::from_hex(gx_hex);
    auto hex_out = fe.to_hex();
    CHECK(secp256k1::test_vectors::hex_equal(hex_out, gx_hex), "Generator X hex round-trip");

    // Generator Y
    const char* gy_hex = GENERATOR_Y;
    fe = HostFieldElement::from_hex(gy_hex);
    hex_out = fe.to_hex();
    CHECK(secp256k1::test_vectors::hex_equal(hex_out, gy_hex), "Generator Y hex round-trip");

    // Field prime
    fe = HostFieldElement::from_hex(FIELD_PRIME);
    hex_out = fe.to_hex();
    CHECK(secp256k1::test_vectors::hex_equal(hex_out, FIELD_PRIME), "Field prime hex round-trip");

    // Zero
    fe = HostFieldElement::zero();
    CHECK(fe.is_zero(), "Zero field element is_zero()");
    hex_out = fe.to_hex();
    CHECK(secp256k1::test_vectors::hex_equal(hex_out, "0000000000000000000000000000000000000000000000000000000000000000"),
          "Zero hex round-trip");

    // One
    fe = HostFieldElement::one();
    CHECK(!fe.is_zero(), "One is not zero");
    CHECK(fe.limbs[0] == 1, "One.limbs[0] == 1");
    CHECK(fe.limbs[1] == 0 && fe.limbs[2] == 0 && fe.limbs[3] == 0, "One upper limbs are zero");
}

// ============================================================
// Test: Scalar hex round-trip
// ============================================================

static void test_scalar_hex_roundtrip() {
    std::cout << "  [Scalar] Hex round-trip...\n";

    auto s = HostScalar::from_hex(CURVE_ORDER);
    auto hex_out = s.to_hex();
    CHECK(secp256k1::test_vectors::hex_equal(hex_out, CURVE_ORDER), "Curve order hex round-trip");

    s = HostScalar::zero();
    CHECK(s.limbs[0] == 0 && s.limbs[1] == 0 && s.limbs[2] == 0 && s.limbs[3] == 0,
          "Zero scalar all limbs zero");

    s = HostScalar::one();
    CHECK(s.limbs[0] == 1, "One scalar limbs[0] == 1");

    s = HostScalar::from_uint64(0xDEADBEEFCAFEBABEULL);
    CHECK(s.limbs[0] == 0xDEADBEEFCAFEBABEULL, "from_uint64 stores correctly");
    CHECK(s.limbs[1] == 0 && s.limbs[2] == 0 && s.limbs[3] == 0, "from_uint64 upper limbs zero");
}

// ============================================================
// Test: to_bytes / from_bytes round-trip
// ============================================================

static void test_bytes_roundtrip() {
    std::cout << "  [Bytes] to_bytes/from_bytes round-trip...\n";

    auto fe = HostFieldElement::from_hex(GENERATOR_X);
    auto bytes = fe.to_bytes();
    auto fe2 = HostFieldElement::from_bytes(bytes);
    CHECK(fe == fe2, "FieldElement bytes round-trip");

    auto sc = HostScalar::from_hex(CURVE_ORDER);
    bytes = sc.to_bytes();
    auto sc2 = HostScalar::from_bytes(bytes);
    CHECK(sc == sc2, "Scalar bytes round-trip");
}

// ============================================================
// Test: Bridge to shared types.hpp (to_data / from_data)
// ============================================================

static void test_type_bridge() {
    std::cout << "  [Bridge] to_data/from_data with shared types...\n";

    // FieldElement <-> FieldElementData
    auto fe = HostFieldElement::from_hex(GENERATOR_X);
    secp256k1::FieldElementData fed = fe.to_data();
    for (int i = 0; i < 4; i++) {
        CHECK(fed.limbs[i] == fe.limbs[i], "FieldElementData.limbs matches HostFieldElement.limbs");
    }
    auto fe2 = HostFieldElement::from_data(fed);
    CHECK(fe == fe2, "FieldElement round-trip through FieldElementData");

    // Scalar <-> ScalarData
    auto sc = HostScalar::from_hex(CURVE_ORDER);
    secp256k1::ScalarData scd = sc.to_data();
    for (int i = 0; i < 4; i++) {
        CHECK(scd.limbs[i] == sc.limbs[i], "ScalarData.limbs matches HostScalar.limbs");
    }
    auto sc2 = HostScalar::from_data(scd);
    CHECK(sc == sc2, "Scalar round-trip through ScalarData");

    // AffinePoint <-> AffinePointData
    auto gp = generator_point();
    secp256k1::AffinePointData apd = gp.to_data();
    CHECK(apd.x.limbs[0] == gp.x.limbs[0], "AffinePointData.x.limbs[0] matches");
    CHECK(apd.y.limbs[0] == gp.y.limbs[0], "AffinePointData.y.limbs[0] matches");
    auto gp2 = HostAffinePoint::from_data(apd);
    CHECK(gp.x == gp2.x && gp.y == gp2.y, "AffinePoint round-trip through AffinePointData");

    // JacobianPoint <-> JacobianPointData
    HostJacobianPoint jp;
    jp.x = gp.x;
    jp.y = gp.y;
    jp.z = HostFieldElement::one();
    jp.infinity = 0;
    secp256k1::JacobianPointData jpd = jp.to_data();
    CHECK(jpd.infinity == 0, "JacobianPointData.infinity == 0");
    CHECK(jpd.z.limbs[0] == 1, "JacobianPointData.z.limbs[0] == 1 (normalized)");
    auto jp2 = HostJacobianPoint::from_data(jpd);
    CHECK(jp.x == jp2.x && jp.y == jp2.y && jp.z == jp2.z, "JacobianPoint round-trip");
}

// ============================================================
// Test: HostPoint factory methods
// ============================================================

static void test_host_point() {
    std::cout << "  [HostPoint] Factory methods...\n";

    // Generator
    auto g = HostPoint::generator();
    CHECK(!g.is_infinity(), "Generator is not infinity");
    auto gx_hex = g.x_fe.to_hex();
    CHECK(secp256k1::test_vectors::hex_equal(gx_hex, GENERATOR_X), "Generator X matches canonical");
    auto gy_hex = g.y_fe.to_hex();
    CHECK(secp256k1::test_vectors::hex_equal(gy_hex, GENERATOR_Y), "Generator Y matches canonical");

    // Z should be 1 (affine)
    CHECK(g.z_fe.limbs[0] == 1, "Generator Z == 1");
    CHECK(g.z_fe.limbs[1] == 0 && g.z_fe.limbs[2] == 0 && g.z_fe.limbs[3] == 0,
          "Generator Z upper limbs == 0");

    // Infinity
    auto inf = HostPoint::infinity_point();
    CHECK(inf.is_infinity(), "Infinity point detected");

    // from_affine
    auto ap = generator_point();
    auto p = HostPoint::from_host_affine(ap);
    CHECK(!p.is_infinity(), "from_host_affine not infinity");
    CHECK(p.x_fe == ap.x, "from_host_affine X matches");
    CHECK(p.y_fe == ap.y, "from_host_affine Y matches");

    // to_data / from_data
    auto jpd = g.to_data();
    auto g2 = HostPoint::from_data(jpd);
    CHECK(g.x_fe == g2.x_fe && g.y_fe == g2.y_fe, "HostPoint to_data/from_data round-trip");
}

// ============================================================
// Test: Test vectors consistency with shared test_vectors.hpp
// ============================================================

static void test_shared_vectors() {
    std::cout << "  [Vectors] Cross-check with shared test_vectors.hpp...\n";

    // Verify Metal generator_point() matches shared GENERATOR_X/Y
    auto g = generator_point();
    CHECK(secp256k1::test_vectors::hex_equal(g.x.to_hex(), GENERATOR_X), "Metal generator X == shared GENERATOR_X");
    CHECK(secp256k1::test_vectors::hex_equal(g.y.to_hex(), GENERATOR_Y), "Metal generator Y == shared GENERATOR_Y");

    // Verify 2*G and 3*G match KG_VECTORS
    auto g2 = two_g_point();
    CHECK(secp256k1::test_vectors::hex_equal(g2.x.to_hex(), KG_VECTORS[1].expected_x), "Metal 2G.x == KG_VECTORS[1].x");
    CHECK(secp256k1::test_vectors::hex_equal(g2.y.to_hex(), KG_VECTORS[1].expected_y), "Metal 2G.y == KG_VECTORS[1].y");

    auto g3 = three_g_point();
    CHECK(secp256k1::test_vectors::hex_equal(g3.x.to_hex(), KG_VECTORS[2].expected_x), "Metal 3G.x == KG_VECTORS[2].x");
    CHECK(secp256k1::test_vectors::hex_equal(g3.y.to_hex(), KG_VECTORS[2].expected_y), "Metal 3G.y == KG_VECTORS[2].y");

    // Verify hex_to_bytes matches between Metal and shared utilities
    for (int i = 0; i < KG_VECTOR_COUNT; i++) {
        auto metal_bytes = secp256k1::metal::hex_to_bytes(KG_VECTORS[i].scalar_hex);
        auto shared_bytes = secp256k1::test_vectors::hex_to_bytes_be(KG_VECTORS[i].scalar_hex);

        // Metal hex_to_bytes and shared hex_to_bytes_be should produce identical results
        // for full-length (64 char) hex strings
        bool match = (memcmp(metal_bytes.data(), shared_bytes.data(), 32) == 0);
        CHECK(match, std::string("hex_to_bytes matches shared for vector ") + KG_VECTORS[i].description);
    }
}

// ============================================================
// Test: MidFieldElementData reinterpret compatibility
// ============================================================

static void test_mid_field_reinterpret() {
    std::cout << "  [Layout] MidFieldElementData reinterpret compatibility...\n";

    auto fe = HostFieldElement::from_hex(GENERATOR_X);
    secp256k1::FieldElementData fed = fe.to_data();

    // Reinterpret as MidFieldElementData (8x32-bit view)
    auto* mid = reinterpret_cast<const secp256k1::MidFieldElementData*>(&fed);

    // Verify byte-level identity: the 32 bytes must be identical
    CHECK(memcmp(&fed, mid, 32) == 0, "FieldElementData and MidFieldElementData byte-identical");

    // Verify 32-bit limbs match the 64-bit limbs when split
    for (int i = 0; i < 4; i++) {
        uint32_t lo = static_cast<uint32_t>(fed.limbs[i] & 0xFFFFFFFF);
        uint32_t hi = static_cast<uint32_t>(fed.limbs[i] >> 32);
        CHECK(mid->limbs[2 * i] == lo,
              std::string("MidField limb ") + std::to_string(2*i) + " == lo of FieldElement limb " + std::to_string(i));
        CHECK(mid->limbs[2 * i + 1] == hi,
              std::string("MidField limb ") + std::to_string(2*i+1) + " == hi of FieldElement limb " + std::to_string(i));
    }
}

// ============================================================
// Test: Serialization (to_compressed, to_uncompressed)
// ============================================================

static void test_serialization() {
    std::cout << "  [Serialize] Compressed/uncompressed encoding...\n";

    auto g = HostPoint::generator();
    auto compressed = g.to_compressed();
    CHECK(compressed.size() == 33, "Compressed size == 33");
    CHECK(compressed[0] == 0x02 || compressed[0] == 0x03, "Compressed prefix is 02 or 03");

    // Generator Y is even, so prefix should be 0x02
    auto y_bytes = g.y_fe.to_bytes();
    bool y_even = (y_bytes[31] & 1) == 0;
    CHECK((y_even && compressed[0] == 0x02) || (!y_even && compressed[0] == 0x03),
          "Compressed prefix matches Y parity");

    auto uncompressed = g.to_uncompressed();
    CHECK(uncompressed.size() == 65, "Uncompressed size == 65");
    CHECK(uncompressed[0] == 0x04, "Uncompressed prefix is 04");

    // Verify X bytes in uncompressed match compressed
    CHECK(memcmp(compressed.data() + 1, uncompressed.data() + 1, 32) == 0,
          "X bytes match between compressed and uncompressed");
}

// ============================================================
// Test: Negate
// ============================================================

static void test_negate() {
    std::cout << "  [Negate] Point negation...\n";

    auto g = HostPoint::generator();
    auto neg_g = g.negate();

    CHECK(!neg_g.is_infinity(), "Negated G is not infinity");
    CHECK(g.x_fe == neg_g.x_fe, "Negated G has same X");
    CHECK(g.y_fe != neg_g.y_fe, "Negated G has different Y");

    // Infinity negate
    auto inf = HostPoint::infinity_point();
    auto neg_inf = inf.negate();
    CHECK(neg_inf.is_infinity(), "Negated infinity is still infinity");
}

// ============================================================
// Main
// ============================================================

int main() {
    std::cout << "+==========================================================+\n"
              << "|  UltrafastSecp256k1 -- Metal Host-Side Type Tests        |\n"
              << "|  Cross-platform (no GPU required)                       |\n"
              << "+==========================================================+\n\n";

    test_field_hex_roundtrip();
    test_scalar_hex_roundtrip();
    test_bytes_roundtrip();
    test_type_bridge();
    test_host_point();
    test_shared_vectors();
    test_mid_field_reinterpret();
    test_serialization();
    test_negate();

    std::cout << "\n=======================================================\n"
              << "  Results: " << g_pass << " passed, " << g_fail << " failed\n"
              << "=======================================================\n";

    if (g_fail > 0) {
        std::cerr << "  FAILED [x]\n";
        return 1;
    }
    std::cout << "  All tests PASSED [ok]\n";
    return 0;
}
