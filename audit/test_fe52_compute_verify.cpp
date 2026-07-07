// test_fe52_compute_verify.cpp — FE52-compute verify differential
// ============================================================================
// Pairing test for the FE52-compute verify path (commit 875d5bee:
// "perf(cpu): FE52-compute ECDSA/Schnorr verify on MSVC cl — parity with libsecp").
//
// That commit added SECP256K1_FE52_COMPUTE, which decouples the 5x52 *compute*
// path from 5x52 *storage*, and gated the ECDSA-verify dual-multiply
// (u1*G + u2*Q) plus the to_jac52/from_jac52 Jacobian bridge on it — so the cl
// (MSVC) verify ecmult runs through the 5x52 field even while the Point storage
// stays 4x64. On native __int128 (this Linux/GCC build) SECP256K1_FE52_COMPUTE
// is defined (config.hpp), so Point::dual_scalar_mul_gen_point IS the same 5x52
// dual-mul that ships in ecdsa_verify here.
//
// What this pins:
//   1. dual_scalar_mul_gen_point(u1,u2,Q)  ==  u1*G + u2*Q   computed with two
//      INDEPENDENT single scalar-muls + a point add, over many vectors. This is
//      a direct cross-check of the verify hot-path dual-mul and the
//      to_jac52/from_jac52 bridge that 875d5bee refactored: any divergence in
//      the GLV/Strauss interleave or the jac52 round-trip breaks the equality.
//   2. ECDSA sign/verify round-trip + tamper rejection — exercises the same
//      dual-mul inside the real ecdsa_verify entry point.
//   3. Schnorr (BIP-340) sign/verify round-trip + tamper rejection.
//
// (A literal FE52-compute-vs-4x64 differential would require toggling
// SECP256K1_FE52_COMPUTE at compile time, which only diverges on MSVC cl; on
// Linux the dual-mul-vs-single-mul identity + the verify KATs are the buildable
// proxy and catch the same class of regression.)
// ============================================================================
#include <cstdio>
#include <cstdint>
#include <array>
#include <cstring>

#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"

using namespace secp256k1;
using secp256k1::fast::Scalar;
using secp256k1::fast::Point;

#if defined(__GNUC__) || defined(__clang__)
#  pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

static int g_pass = 0, g_fail = 0;
#include "audit_check.hpp"

// Deterministic 32-byte value from a 64-bit seed (xorshift fill).
// from_bytes / from_seed reduce mod n, so any 32 bytes yields a valid scalar.
static std::array<std::uint8_t, 32> bytes_from_seed(std::uint64_t s) {
    std::array<std::uint8_t, 32> b{};
    for (int i = 0; i < 32; ++i) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        b[i] = static_cast<std::uint8_t>(s & 0xff);
    }
    return b;
}
static Scalar scalar_from_seed(std::uint64_t s) {
    return Scalar::from_bytes(bytes_from_seed(s));
}

static bool points_equal(const Point& a, const Point& b) {
    // Compare normalized affine encodings (parity + x). Both operands are
    // non-infinity for the randomized vectors below (P=O needs u1 + u2*d == 0).
    return a.to_compressed() == b.to_compressed();
}

// 1. Core differential: the verify dual-mul vs two independent single-muls.
static void test_dual_mul_matches_single_muls() {
    std::printf("[fe52-dualmul] dual_scalar_mul_gen_point(u1,u2,Q) == u1*G + u2*Q\n");
    const Point G = Point::generator();
    int checks = 0;
    for (std::uint64_t i = 1; i <= 200; ++i) {
        const Scalar d  = scalar_from_seed(0x9E3779B97F4A7C15ull * i + 1);
        const Scalar u1 = scalar_from_seed(0xD1B54A32D192ED03ull * i + 2);
        const Scalar u2 = scalar_from_seed(0xCA5A826395121157ull * i + 3);
        const Point  Q  = G.scalar_mul(d);                            // a valid pubkey
        const Point  via_dual = Point::dual_scalar_mul_gen_point(u1, u2, Q);
        const Point  via_single = G.scalar_mul(u1).add(Q.scalar_mul(u2));
        CHECK(points_equal(via_dual, via_single),
              "FE52 dual-mul equals independent single-mul sum");
        ++checks;
    }
    std::printf("    %d vectors checked\n", checks);
}

// 2. ECDSA verify through the real entry point (uses the FE52 dual-mul internally).
static void test_ecdsa_verify_roundtrip() {
    std::printf("[fe52-ecdsa] ECDSA sign/verify round-trip + tamper rejection\n");
    int checks = 0;
    for (std::uint64_t i = 1; i <= 48; ++i) {
        const Scalar sk = scalar_from_seed(0x2545F4914F6CDD1Dull * i + 7);
        const Point  pk = Point::generator().scalar_mul(sk);
        const std::array<std::uint8_t, 32> msg = bytes_from_seed(0xA0761D6478BD642Full * i + 11);
        const ECDSASignature sig = ecdsa_sign(msg, sk);
        CHECK(ecdsa_verify(msg, pk, sig), "valid ECDSA signature accepted");
        std::array<std::uint8_t, 32> bad = msg;
        bad[i % 32] ^= 0x01;
        CHECK(!ecdsa_verify(bad, pk, sig), "tampered-message ECDSA signature rejected");
        ++checks;
    }
    std::printf("    %d sign/verify pairs\n", checks);
}

// 3. Schnorr (BIP-340) verify through the real entry point.
static void test_schnorr_verify_roundtrip() {
    std::printf("[fe52-schnorr] Schnorr sign/verify round-trip + tamper rejection\n");
    const std::array<std::uint8_t, 32> aux{};  // deterministic nonce (safe vs reuse)
    int checks = 0;
    for (std::uint64_t i = 1; i <= 48; ++i) {
        const Scalar sk = scalar_from_seed(0x8EBC6AF09C88C6E3ull * i + 13);
        const std::array<std::uint8_t, 32> px  = schnorr_pubkey(sk);
        const std::array<std::uint8_t, 32> msg = bytes_from_seed(0x589965CC75374CC3ull * i + 17);
        const SchnorrSignature sig = schnorr_sign(sk, msg, aux);
        CHECK(schnorr_verify(px, msg, sig), "valid Schnorr signature accepted");
        std::array<std::uint8_t, 32> bad = msg;
        bad[i % 32] ^= 0x01;
        CHECK(!schnorr_verify(px, bad, sig), "tampered-message Schnorr signature rejected");
        ++checks;
    }
    std::printf("    %d sign/verify pairs\n", checks);
}

int test_fe52_compute_verify_run() {
    std::printf("[test_fe52_compute_verify] FE52-compute verify differential\n");
    test_dual_mul_matches_single_muls();
    std::printf("\n");
    test_ecdsa_verify_roundtrip();
    std::printf("\n");
    test_schnorr_verify_roundtrip();
    std::printf("\n[test_fe52_compute_verify] %d/%d passed\n", g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#ifdef STANDALONE_TEST
int main() { return test_fe52_compute_verify_run(); }
#endif
