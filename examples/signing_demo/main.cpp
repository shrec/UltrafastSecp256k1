// ============================================================================
// UltrafastSecp256k1 -- ECDSA + Schnorr Signing Demo
// ============================================================================
// Demonstrates:
//   1. ECDSA sign + verify (RFC 6979 deterministic nonce)
//   2. Schnorr BIP-340 sign + verify
//   3. Signature serialization (DER, compact, BIP-340)
//   4. Constant-time (CT) signing alternative
//
// Build: cmake --build <build_dir> --target example_signing_demo
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>

#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"

using namespace secp256k1;
using namespace secp256k1::fast;

// -- helpers ------------------------------------------------------------------

static void print_hex(const char* label, const uint8_t* data, size_t len) {
    printf("  %s: ", label);
    for (size_t i = 0; i < len; ++i) printf("%02x", data[i]);
    printf("\n");
}

// Deterministic 32-byte hash (for demo purposes only!)
static std::array<uint8_t, 32> fake_sha256(const char* msg) {
    std::array<uint8_t, 32> out{};
    size_t len = strlen(msg);
    for (size_t i = 0; i < 32; ++i) {
        out[i] = static_cast<uint8_t>(msg[i % len] ^ (i * 0x9e + 0x37));
    }
    return out;
}

// -- main ---------------------------------------------------------------------

int main() {
    printf("=== UltrafastSecp256k1 -- Signing Demo ===\n\n");

    // ── Setup ────────────────────────────────────────────────────────────────

    // Private key (deterministic for demo)
    auto priv = Scalar::from_hex(
        "e8f32e723decf4051aefac8e2c93c9c5b214313817cdb01a1494b917c8436b35");
    auto pub = Point::generator().scalar_mul(priv);

    auto compressed = pub.to_compressed();
    printf("[Setup]\n");
    printf("  Private key : %s\n", priv.to_hex().c_str());
    print_hex("Public key  ", compressed.data(), compressed.size());
    printf("\n");

    // Message hash (normally SHA-256 of the actual message)
    auto msg_hash = fake_sha256("Hello, UltrafastSecp256k1!");
    print_hex("Message hash", msg_hash.data(), msg_hash.size());
    printf("\n");

    // ── 1. ECDSA Sign ───────────────────────────────────────────────────────

    printf("[1] ECDSA Signing (RFC 6979)\n");
    auto ecdsa_sig = ecdsa_sign(msg_hash, priv);

    // Compact encoding (64 bytes: r || s)
    auto compact = ecdsa_sig.to_compact();
    print_hex("Signature (compact)", compact.data(), compact.size());

    // DER encoding (variable length, max 72 bytes)
    auto [der_bytes, der_len] = ecdsa_sig.to_der();
    print_hex("Signature (DER)   ", der_bytes.data(), der_len);

    // Low-S check (should already be normalized by ecdsa_sign)
    printf("  Low-S: %s\n", ecdsa_sig.is_low_s() ? "yes" : "no");
    printf("\n");

    // ── 2. ECDSA Verify ─────────────────────────────────────────────────────

    printf("[2] ECDSA Verification\n");
    bool ecdsa_ok = ecdsa_verify(msg_hash, pub, ecdsa_sig);
    printf("  Verify: %s\n", ecdsa_ok ? "PASS" : "FAIL");

    // Tampered message should fail
    auto bad_hash = msg_hash;
    bad_hash[0] ^= 0x01;
    bool ecdsa_bad = ecdsa_verify(bad_hash, pub, ecdsa_sig);
    printf("  Tampered msg verify: %s (expected FAIL)\n",
           ecdsa_bad ? "PASS" : "FAIL");
    printf("\n");

    // ── 3. Schnorr BIP-340 Sign ─────────────────────────────────────────────

    printf("[3] Schnorr BIP-340 Signing\n");

    // Auxiliary randomness (can be all zeros for deterministic)
    std::array<uint8_t, 32> aux_rand{};

    // Create keypair (recommended: pre-compute once, sign many)
    auto kp = schnorr_keypair_create(priv);

    auto schnorr_sig = schnorr_sign(kp, msg_hash, aux_rand);
    auto sig_bytes = schnorr_sig.to_bytes();
    print_hex("Signature (BIP-340)", sig_bytes.data(), sig_bytes.size());
    printf("  Length: 64 bytes (R.x 32 + s 32)\n\n");

    // ── 4. Schnorr Verify ───────────────────────────────────────────────────

    printf("[4] Schnorr BIP-340 Verification\n");
    bool schnorr_ok = schnorr_verify(kp.px, msg_hash, schnorr_sig);
    printf("  Verify: %s\n", schnorr_ok ? "PASS" : "FAIL");

    bool schnorr_bad = schnorr_verify(kp.px, bad_hash, schnorr_sig);
    printf("  Tampered msg verify: %s (expected FAIL)\n",
           schnorr_bad ? "PASS" : "FAIL");
    printf("\n");

    // ── 5. Round-Trip Serialization ─────────────────────────────────────────

    printf("[5] Serialization Round-Trip\n");

    // ECDSA compact round-trip
    auto decoded_ecdsa = ECDSASignature::from_compact(compact);
    bool ecdsa_rt = ecdsa_verify(msg_hash, pub, decoded_ecdsa);
    printf("  ECDSA compact round-trip verify: %s\n",
           ecdsa_rt ? "PASS" : "FAIL");

    // Schnorr round-trip
    auto decoded_schnorr = SchnorrSignature::from_bytes(sig_bytes);
    bool schnorr_rt = schnorr_verify(kp.px, msg_hash, decoded_schnorr);
    printf("  Schnorr round-trip verify: %s\n",
           schnorr_rt ? "PASS" : "FAIL");
    printf("\n");

    // ── Summary ──────────────────────────────────────────────────────────────

    bool all_pass = ecdsa_ok && !ecdsa_bad && schnorr_ok && !schnorr_bad
                    && ecdsa_rt && schnorr_rt;

    printf("=== All checks %s ===\n", all_pass ? "PASSED" : "FAILED");
    return all_pass ? 0 : 1;
}
