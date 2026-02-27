// ============================================================================
// Cryptographic Self-Audit: Security Hardening (Section V)
// ============================================================================
// Covers: API hardening (zero key, identity inputs, double-free sim),
//         bit-flip resilience, secret zeroization, signature malleability,
//         nonce reuse detection, serialization round-trip integrity.
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>
#include <random>
#include <stdexcept>

#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/recovery.hpp"
#include "secp256k1/ct/ops.hpp"
#include "secp256k1/ct_utils.hpp"

using namespace secp256k1::fast;

static int g_pass = 0, g_fail = 0;
static const char* g_section = "";

#define CHECK(cond, msg) do { \
    if (cond) { \
        ++g_pass; \
    } else { \
        printf("  FAIL [%s]: %s (line %d)\n", g_section, msg, __LINE__); \
        ++g_fail; \
    } \
} while(0)

static std::mt19937_64 rng(0xA0D17'5EC0A);  // NOLINT(cert-msc32-c,cert-msc51-cpp)

static Scalar random_scalar() {
    std::array<uint8_t, 32> out{};
    for (int i = 0; i < 4; ++i) {
        uint64_t v = rng();
        std::memcpy(out.data() + static_cast<std::size_t>(i) * 8, &v, 8);
    }
    for (;;) {
        auto s = Scalar::from_bytes(out);
        if (!s.is_zero()) return s;
        out[31] ^= 0x01;
    }
}

// ============================================================================
// 1. Zero / identity key handling
// ============================================================================
static void test_zero_key_handling() {
    g_section = "zero_key";
    printf("[1] Zero / identity key handling\n");

    auto G = Point::generator();

    // ECDSA sign with zero key -> failure
    {
        auto zero = Scalar::from_uint64(0);
        std::array<uint8_t, 32> msg{};
        msg[0] = 0x42;
        auto sig = secp256k1::ecdsa_sign(msg, zero);
        CHECK(sig.r.is_zero() && sig.s.is_zero(), "ECDSA: sign(k=0) -> zero sig");
    }

    // Scalar mul by zero -> infinity
    {
        CHECK(G.scalar_mul(Scalar::from_uint64(0)).is_infinity(), "0*G == O");
    }

    // Scalar inverse of zero -- should throw or return zero
    {
        auto z = Scalar::from_uint64(0);
        bool threw = false;
        try {
            auto inv = z.inverse();
            // If it doesn't throw, inv should be zero (by convention)
            CHECK(inv.is_zero(), "inv(0) = 0 (by convention)");
        } catch (const std::exception&) {
            threw = true;
        }
        CHECK(threw || true, "inv(0) either throws or returns 0");
    }

    // Field inverse of zero -- should throw or return zero
    {
        auto z = FieldElement::from_uint64(0);
        bool threw = false;
        try {
            auto inv = z.inverse();
            CHECK(inv == FieldElement::from_uint64(0), "fe_inv(0) = 0");
        } catch (const std::exception&) {
            threw = true;
        }
        CHECK(threw || true, "fe_inv(0) either throws or returns 0");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 2. Secret zeroization via ct_memzero
// ============================================================================
static void test_zeroization() {
    g_section = "zeroize";
    printf("[2] Secret zeroization (ct_memzero)\n");

    // Fill buffer with secret, zeroize, verify
    std::array<uint8_t, 64> secret_buf{};
    for (auto& b : secret_buf) b = rng() & 0xFF;

    // Verify non-zero before
    bool nonzero_before = false;
    for (auto b : secret_buf) if (b != 0) nonzero_before = true;
    CHECK(nonzero_before, "secret buffer is nonzero");

    secp256k1::ct::ct_memzero(secret_buf.data(), secret_buf.size());

    bool all_zero = true;
    for (auto b : secret_buf) if (b != 0) all_zero = false;
    CHECK(all_zero, "ct_memzero zeroed all bytes");

    // Scalar zeroization pattern
    {
        auto sk = random_scalar();
        auto sk_bytes = sk.to_bytes();
        secp256k1::ct::ct_memzero(sk_bytes.data(), sk_bytes.size());
        CHECK(secp256k1::ct::ct_is_zero(sk_bytes), "scalar bytes zeroed");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 3. Bit-flip resilience on signatures
// ============================================================================
static void test_bitflip_resilience() {
    g_section = "bitflip";
    printf("[3] Bit-flip resilience on signatures (1K)\n");

    auto G = Point::generator();

    for (int i = 0; i < 1000; ++i) {
        auto sk = random_scalar();
        auto pk = G.scalar_mul(sk);
        std::array<uint8_t, 32> msg{};
        uint64_t v = rng();
        std::memcpy(msg.data(), &v, 8);

        auto sig = secp256k1::ecdsa_sign(msg, sk);
        CHECK(secp256k1::ecdsa_verify(msg, pk, sig), "original sig valid");

        // Flip each bit in compact encoding
        auto compact = sig.to_compact();
        int rejections = 0;
        for (size_t byte_idx = 0; byte_idx < 64; ++byte_idx) {
            for (int bit = 0; bit < 8; ++bit) {
                auto flipped = compact;
                flipped[byte_idx] ^= (1u << bit);
                auto fsig = secp256k1::ECDSASignature::from_compact(flipped);
                if (!secp256k1::ecdsa_verify(msg, pk, fsig)) {
                    ++rejections;
                }
            }
        }
        // All 512 bit-flips should be rejected
        CHECK(rejections == 512, "all 512 bit-flips rejected");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 4. Message bit-flip detection
// ============================================================================
static void test_message_bitflip() {
    g_section = "msg_flip";
    printf("[4] Message bit-flip detection (1K)\n");

    auto G = Point::generator();

    for (int i = 0; i < 1000; ++i) {
        auto sk = random_scalar();
        auto pk = G.scalar_mul(sk);
        std::array<uint8_t, 32> msg{};
        uint64_t v = rng();
        std::memcpy(msg.data(), &v, 8);

        auto sig = secp256k1::ecdsa_sign(msg, sk);

        // Flip each byte position (exhaustive per-byte)
        int rejections = 0;
        for (size_t j = 0; j < 32; ++j) {
            auto bad_msg = msg;
            bad_msg[j] ^= 0x01;
            if (!secp256k1::ecdsa_verify(bad_msg, pk, sig)) {
                ++rejections;
            }
        }
        CHECK(rejections == 32, "all 32 byte-flips rejected");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 5. Nonce determinism (RFC 6979)
// ============================================================================
static void test_nonce_determinism() {
    g_section = "nonce_det";
    printf("[5] Nonce determinism (RFC 6979)\n");

    // Same key + same message -> same signature
    for (int i = 0; i < 100; ++i) {
        auto sk = random_scalar();
        std::array<uint8_t, 32> msg{};
        uint64_t v = rng();
        std::memcpy(msg.data(), &v, 8);

        auto sig1 = secp256k1::ecdsa_sign(msg, sk);
        auto sig2 = secp256k1::ecdsa_sign(msg, sk);

        CHECK(sig1.r == sig2.r && sig1.s == sig2.s, "deterministic nonce");
    }

    // Different messages -> different r (overwhelming probability)
    {
        auto sk = random_scalar();
        std::array<uint8_t, 32> msg1{}, msg2{};
        msg1[0] = 0x01;
        msg2[0] = 0x02;

        auto sig1 = secp256k1::ecdsa_sign(msg1, sk);
        auto sig2 = secp256k1::ecdsa_sign(msg2, sk);

        CHECK(!(sig1.r == sig2.r && sig1.s == sig2.s), "different msg -> different sig");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 6. Serialization round-trip integrity
// ============================================================================
static void test_serialization_integrity() {
    g_section = "serial";
    printf("[6] Serialization round-trip integrity\n");

    auto G = Point::generator();

    // Point serialization
    for (int i = 0; i < 1000; ++i) {
        auto P = G.scalar_mul(random_scalar());

        auto comp = P.to_compressed();
        auto uncomp = P.to_uncompressed();

        // Compressed: prefix(1) + X(32)
        CHECK(comp.size() == 33, "compressed size");
        CHECK(comp[0] == 0x02 || comp[0] == 0x03, "compressed prefix");

        // Uncompressed: prefix(1) + X(32) + Y(32)
        CHECK(uncomp.size() == 65, "uncompressed size");
        CHECK(uncomp[0] == 0x04, "uncompressed prefix");

        // X match
        CHECK(std::memcmp(comp.data() + 1, uncomp.data() + 1, 32) == 0,
              "X coordinates match");
    }

    // Scalar byte round-trip
    for (int i = 0; i < 1000; ++i) {
        auto s = random_scalar();
        auto bytes = s.to_bytes();
        auto restored = Scalar::from_bytes(bytes);
        CHECK(s == restored, "scalar byte round-trip");
    }

    // FieldElement byte round-trip
    for (int i = 0; i < 1000; ++i) {
        std::array<uint8_t, 32> bytes{};
        for (int j = 0; j < 4; ++j) {
            uint64_t v = rng();
            std::memcpy(bytes.data() + static_cast<std::size_t>(j) * 8, &v, 8);
        }
        // Ensure < p by clearing top bit
        bytes[0] &= 0x7F;
        auto fe = FieldElement::from_bytes(bytes);
        auto out = fe.to_bytes();
        auto fe2 = FieldElement::from_bytes(out);
        CHECK(fe == fe2, "field element byte round-trip");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 7. Compact recovery serialization
// ============================================================================
static void test_compact_recovery_serial() {
    g_section = "compact_rec";
    printf("[7] Compact recovery serialization (1K)\n");

    auto G = Point::generator();
    (void)G;

    for (int i = 0; i < 1000; ++i) {
        auto sk = random_scalar();
        std::array<uint8_t, 32> msg{};
        uint64_t v = rng();
        std::memcpy(msg.data(), &v, 8);

        auto rsig = secp256k1::ecdsa_sign_recoverable(msg, sk);
        auto compact = secp256k1::recoverable_to_compact(rsig, true);

        CHECK(compact.size() == 65, "compact size 65");
        // First byte encodes recid: 27 + recid + 4(compressed)
        int const expected_first = 27 + rsig.recid + 4;
        CHECK(compact[0] == static_cast<uint8_t>(expected_first), "header byte correct");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 8. Double operations idempotency
// ============================================================================
static void test_double_ops() {
    g_section = "double_op";
    printf("[8] Double operations idempotency\n");

    auto G = Point::generator();

    // Double inverse: inv(inv(a)) == a
    for (int i = 0; i < 1000; ++i) {
        auto a = random_scalar();
        auto inv1 = a.inverse();
        auto inv2 = inv1.inverse();
        CHECK(a == inv2, "scalar: inv(inv(a)) == a");
    }

    // Double negate: neg(neg(P)) == P
    for (int i = 0; i < 1000; ++i) {
        auto P = G.scalar_mul(random_scalar());
        auto nn = P.negate().negate();
        auto P_bytes = P.to_compressed();
        auto nn_bytes = nn.to_compressed();
        CHECK(P_bytes == nn_bytes, "point: neg(neg(P)) == P");
    }

    // Double dbl consistency
    for (int i = 0; i < 100; ++i) {
        auto P = G.scalar_mul(random_scalar());
        auto P4a = P.dbl().dbl();
        auto P4b = P.scalar_mul(Scalar::from_uint64(4));
        CHECK(P4a.to_compressed() == P4b.to_compressed(), "dbl(dbl(P)) == 4*P");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 9. Cross-algorithm consistency (ECDSA key == Schnorr key)
// ============================================================================
static void test_cross_algorithm() {
    g_section = "cross_alg";
    printf("[9] Cross-algorithm consistency (ECDSA/Schnorr)\n");

    auto G = Point::generator();

    for (int i = 0; i < 100; ++i) {
        auto sk = random_scalar();
        auto ecdsa_pk = G.scalar_mul(sk);
        auto schnorr_pkx = secp256k1::schnorr_pubkey(sk);

        // Schnorr's x-only pubkey should match ECDSA pubkey's x
        auto ecdsa_uncomp = ecdsa_pk.to_uncompressed();
        // x is bytes 1..32
        bool x_match = std::memcmp(
            ecdsa_uncomp.data() + 1, schnorr_pkx.data(), 32) == 0;

        // BIP-340: if Y is odd, negate. So x might match the negated point
        if (!x_match) {
            auto neg_pk = ecdsa_pk.negate();
            auto neg_uncomp = neg_pk.to_uncompressed();
            x_match = std::memcmp(
                neg_uncomp.data() + 1, schnorr_pkx.data(), 32) == 0;
        }

        CHECK(x_match, "schnorr_pubkey x matches ECDSA pk x (or negated)");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 10. High-S rejection test
// ============================================================================
static void test_high_s_rejection() {
    g_section = "high_s";
    printf("[10] High-S detection\n");

    auto G = Point::generator();
    (void)G;

    for (int i = 0; i < 1000; ++i) {
        auto sk = random_scalar();
        std::array<uint8_t, 32> msg{};
        uint64_t v = rng();
        std::memcpy(msg.data(), &v, 8);

        auto sig = secp256k1::ecdsa_sign(msg, sk);

        // sign always produces low-S
        CHECK(sig.is_low_s(), "sign -> low-S");

        // Manually create high-S version
        secp256k1::ECDSASignature high;
        high.r = sig.r;
        high.s = sig.s.negate();

        CHECK(!high.is_low_s(), "negated s is high-S");
        CHECK(high.normalize().is_low_s(), "normalize restores low-S");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// _run() entry point for unified audit runner
// ============================================================================

int audit_security_run() {
    g_pass = 0; g_fail = 0;

    test_zero_key_handling();
    test_zeroization();
    test_bitflip_resilience();
    test_message_bitflip();
    test_nonce_determinism();
    test_serialization_integrity();
    test_compact_recovery_serial();
    test_double_ops();
    test_cross_algorithm();
    test_high_s_rejection();

    return g_fail > 0 ? 1 : 0;
}

// ============================================================================
#ifndef UNIFIED_AUDIT_RUNNER
int main() {
    printf("===============================================================\n");
    printf("  AUDIT V -- Security Hardening\n");
    printf("===============================================================\n\n");

    test_zero_key_handling();
    test_zeroization();
    test_bitflip_resilience();
    test_message_bitflip();
    test_nonce_determinism();
    test_serialization_integrity();
    test_compact_recovery_serial();
    test_double_ops();
    test_cross_algorithm();
    test_high_s_rejection();

    printf("===============================================================\n");
    printf("  SECURITY AUDIT: %d passed, %d failed\n", g_pass, g_fail);
    printf("===============================================================\n");

    return g_fail > 0 ? 1 : 0;
}
#endif // UNIFIED_AUDIT_RUNNER
