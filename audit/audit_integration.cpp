// ============================================================================
// Cryptographic Self-Audit: Integration Testing (Section VI)
// ============================================================================
// Covers: ECDH key exchange, batch verification, FROST threshold signatures,
//         cross-protocol consistency, end-to-end flows, recovery round-trips.
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>
#include <vector>
#include <random>

#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/recovery.hpp"
#include "secp256k1/ecdh.hpp"
#include "secp256k1/batch_verify.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/ct_utils.hpp"
#include "secp256k1/sanitizer_scale.hpp"

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

static std::mt19937_64 rng(0xA0D17'1D7E6);  // NOLINT(cert-msc32-c,cert-msc51-cpp)

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

static bool points_equal(const Point& a, const Point& b) {
    if (a.is_infinity() && b.is_infinity()) return true;
    if (a.is_infinity() != b.is_infinity()) return false;
    return a.to_compressed() == b.to_compressed();
}

// ============================================================================
// 1. ECDH key exchange -- symmetry & correctness
// ============================================================================
static void test_ecdh() {
    g_section = "ecdh";
    printf("[1] ECDH key exchange symmetry (1K)\n");

    auto G = Point::generator();

    for (int i = 0; i < SCALED(1000, 50); ++i) {
        auto sk_a = random_scalar();
        auto sk_b = random_scalar();
        auto pk_a = G.scalar_mul(sk_a);
        auto pk_b = G.scalar_mul(sk_b);

        // shared_secret_a = ECDH(sk_a, pk_b) == ECDH(sk_b, pk_a) = shared_secret_b
        auto secret_a = secp256k1::ecdh_compute(sk_a, pk_b);
        auto secret_b = secp256k1::ecdh_compute(sk_b, pk_a);
        CHECK(secret_a == secret_b, "ECDH symmetry (hashed)");

        // xonly variant
        auto xonly_a = secp256k1::ecdh_compute_xonly(sk_a, pk_b);
        auto xonly_b = secp256k1::ecdh_compute_xonly(sk_b, pk_a);
        CHECK(xonly_a == xonly_b, "ECDH symmetry (xonly)");

        // raw variant
        auto raw_a = secp256k1::ecdh_compute_raw(sk_a, pk_b);
        auto raw_b = secp256k1::ecdh_compute_raw(sk_b, pk_a);
        CHECK(raw_a == raw_b, "ECDH symmetry (raw)");

        // raw should match x-coordinate of sk_a * pk_b
        auto shared_point = pk_b.scalar_mul(sk_a);
        auto shared_uncomp = shared_point.to_uncompressed();
        std::array<uint8_t, 32> expected_x;
        std::memcpy(expected_x.data(), shared_uncomp.data() + 1, 32);
        CHECK(raw_a == expected_x, "ECDH raw == x-coord of shared point");
    }

    // ECDH with infinity should return zeros
    {
        auto sk = random_scalar();
        auto inf = Point::infinity();
        auto result = secp256k1::ecdh_compute(sk, inf);
        std::array<uint8_t, 32> const zeros{};
        CHECK(result == zeros, "ECDH with infinity -> all-zeros");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 2. Schnorr batch verification
// ============================================================================
static void test_schnorr_batch_verify() {
    g_section = "batch_sch";
    printf("[2] Schnorr batch verification\n");

    // Create N valid Schnorr signatures
    constexpr int N = 100;
    std::vector<secp256k1::SchnorrBatchEntry> entries;
    entries.reserve(N);

    for (int i = 0; i < N; ++i) {
        auto sk = random_scalar();
        auto pkx = secp256k1::schnorr_pubkey(sk);
        std::array<uint8_t, 32> msg{};
        uint64_t v = rng();
        std::memcpy(msg.data(), &v, 8);
        std::array<uint8_t, 32> const aux{};

        auto sig = secp256k1::schnorr_sign(sk, msg, aux);
        entries.push_back({pkx, msg, sig});
    }

    bool const all_valid = secp256k1::schnorr_batch_verify(entries);
    CHECK(all_valid, "batch(100 valid) -> true");

    // Corrupt one signature
    {
        auto bad = entries;
        bad[50].signature.r[0] ^= 0x01;
        bool const bad_result = secp256k1::schnorr_batch_verify(bad);
        CHECK(!bad_result, "batch with 1 bad -> false");

        // Identify the bad one
        auto invalids = secp256k1::schnorr_batch_identify_invalid(
            bad.data(), bad.size());
        CHECK(invalids.size() == 1 && invalids[0] == 50,
              "identify_invalid finds index 50");
    }

    // Empty batch
    {
        std::vector<secp256k1::SchnorrBatchEntry> const empty;
        CHECK(secp256k1::schnorr_batch_verify(empty), "empty batch -> true");
    }

    // Single entry
    {
        std::vector<secp256k1::SchnorrBatchEntry> const single = {entries[0]};
        CHECK(secp256k1::schnorr_batch_verify(single), "single entry -> true");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 3. ECDSA batch verification
// ============================================================================
static void test_ecdsa_batch_verify() {
    g_section = "batch_ecd";
    printf("[3] ECDSA batch verification\n");

    auto G = Point::generator();
    constexpr int N = 100;
    std::vector<secp256k1::ECDSABatchEntry> entries;
    entries.reserve(N);

    for (int i = 0; i < N; ++i) {
        auto sk = random_scalar();
        auto pk = G.scalar_mul(sk);
        std::array<uint8_t, 32> msg{};
        uint64_t v = rng();
        std::memcpy(msg.data(), &v, 8);

        auto sig = secp256k1::ecdsa_sign(msg, sk);
        entries.push_back({msg, pk, sig});
    }

    bool const all_valid = secp256k1::ecdsa_batch_verify(entries);
    CHECK(all_valid, "ECDSA batch(100 valid) -> true");

    // Corrupt one
    {
        auto bad = entries;
        auto compact = bad[25].signature.to_compact();
        compact[0] ^= 0x01;
        bad[25].signature = secp256k1::ECDSASignature::from_compact(compact);

        bool const bad_result = secp256k1::ecdsa_batch_verify(bad);
        CHECK(!bad_result, "ECDSA batch with 1 bad -> false");

        auto invalids = secp256k1::ecdsa_batch_identify_invalid(
            bad.data(), bad.size());
        bool found_25 = false;
        for (auto idx : invalids) if (idx == 25) found_25 = true;
        CHECK(found_25, "ECDSA identify_invalid finds index 25");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 4. ECDSA sign -> recover -> verify (full round-trip)
// ============================================================================
static void test_ecdsa_full_roundtrip() {
    g_section = "full_rt";
    printf("[4] ECDSA sign -> recover -> verify (1K)\n");

    auto G = Point::generator();

    for (int i = 0; i < SCALED(1000, 50); ++i) {
        auto sk = random_scalar();
        auto pk = G.scalar_mul(sk);
        std::array<uint8_t, 32> msg{};
        uint64_t v = rng();
        std::memcpy(msg.data(), &v, 8);

        // Sign recoverable
        auto rsig = secp256k1::ecdsa_sign_recoverable(msg, sk);

        // Verify signature normally
        CHECK(secp256k1::ecdsa_verify(msg, pk, rsig.sig), "sig verifies");

        // Recover public key
        auto [rec_pk, ok] = secp256k1::ecdsa_recover(msg, rsig.sig, rsig.recid);
        CHECK(ok, "recovery ok");
        CHECK(points_equal(pk, rec_pk), "recovered pk matches");

        // Compact recovery serialization
        auto compact = secp256k1::recoverable_to_compact(rsig, true);
        CHECK(compact.size() == 65, "compact = 65 bytes");

        // DER round-trip
        auto [der, der_len] = rsig.sig.to_der();
        CHECK(der_len > 0 && der_len <= 72, "DER len ok");
        CHECK(der[0] == 0x30, "DER SEQUENCE tag");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 5. Schnorr sign -> verify -> batch verify (cross-path)
// ============================================================================
static void test_schnorr_cross_path() {
    g_section = "schn_cross";
    printf("[5] Schnorr cross-path: individual vs batch (500)\n");

    std::vector<secp256k1::SchnorrBatchEntry> batch;
    batch.reserve(500);

    for (int i = 0; i < SCALED(500, 30); ++i) {
        auto sk = random_scalar();
        auto pkx = secp256k1::schnorr_pubkey(sk);
        std::array<uint8_t, 32> msg{};
        uint64_t v = rng();
        std::memcpy(msg.data(), &v, 8);
        std::array<uint8_t, 32> const aux{};

        auto sig = secp256k1::schnorr_sign(sk, msg, aux);

        // Individual verify
        CHECK(secp256k1::schnorr_verify(pkx, msg, sig), "individual verify");

        // Byte round-trip
        auto bytes = sig.to_bytes();
        auto restored = secp256k1::SchnorrSignature::from_bytes(bytes);
        CHECK(secp256k1::schnorr_verify(pkx, msg, restored), "restored sig verify");

        batch.push_back({pkx, msg, sig});
    }

    // Batch verify all 500
    CHECK(secp256k1::schnorr_batch_verify(batch), "batch verify 500");

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 6. Fast vs CT scalar mul cross-check (integration consistency)
// ============================================================================
static void test_fast_vs_ct_integration() {
    g_section = "fast_ct";
    printf("[6] Fast vs CT integration cross-check (500)\n");

    auto G = Point::generator();

    for (int i = 0; i < SCALED(500, 30); ++i) {
        auto sk = random_scalar();

        // Generate pubkey via fast path
        auto pk_fast = G.scalar_mul(sk);

        // Generate pubkey via CT path
        auto pk_ct = secp256k1::ct::generator_mul(sk);

        CHECK(points_equal(pk_fast, pk_ct), "fast pubkey == CT pubkey");

        // ECDSA sign (uses fast path), verify with both pubkeys
        std::array<uint8_t, 32> msg{};
        uint64_t v = rng();
        std::memcpy(msg.data(), &v, 8);

        auto sig = secp256k1::ecdsa_sign(msg, sk);
        CHECK(secp256k1::ecdsa_verify(msg, pk_fast, sig), "verify w/ fast pk");
        CHECK(secp256k1::ecdsa_verify(msg, pk_ct, sig), "verify w/ CT pk");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 7. ECDH + ECDSA combined protocol flow
// ============================================================================
static void test_combined_protocol() {
    g_section = "protocol";
    printf("[7] Combined ECDH + ECDSA protocol flow (100)\n");

    auto G = Point::generator();

    for (int i = 0; i < 100; ++i) {
        // Alice and Bob generate keys
        auto sk_a = random_scalar(), sk_b = random_scalar();
        auto pk_a = G.scalar_mul(sk_a), pk_b = G.scalar_mul(sk_b);

        // ECDH: derive shared secret
        auto shared = secp256k1::ecdh_compute(sk_a, pk_b);
        auto shared_b = secp256k1::ecdh_compute(sk_b, pk_a);
        CHECK(shared == shared_b, "shared secret matches");

        // Alice signs the shared secret
        auto sig = secp256k1::ecdsa_sign(shared, sk_a);
        CHECK(secp256k1::ecdsa_verify(shared, pk_a, sig), "Alice's sig valid");

        // Bob signs the shared secret
        auto sig_b = secp256k1::ecdsa_sign(shared, sk_b);
        CHECK(secp256k1::ecdsa_verify(shared, pk_b, sig_b), "Bob's sig valid");

        // Cross: Bob's sig should NOT verify with Alice's key
        CHECK(!secp256k1::ecdsa_verify(shared, pk_a, sig_b), "cross-key fails");

        // Recovery: recover Alice's key from her signature
        auto rsig = secp256k1::ecdsa_sign_recoverable(shared, sk_a);
        auto [rec_pk, ok] = secp256k1::ecdsa_recover(shared, rsig.sig, rsig.recid);
        CHECK(ok && points_equal(rec_pk, pk_a), "recovered Alice's pk");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 8. Multi-key Schnorr aggregation check (naive additive)
// ============================================================================
static void test_multikey_consistency() {
    g_section = "multikey";
    printf("[8] Multi-key consistency (point addition, 200)\n");

    auto G = Point::generator();

    for (int i = 0; i < 200; ++i) {
        // k1 + k2 -> (k1+k2)*G should equal k1*G + k2*G
        auto k1 = random_scalar(), k2 = random_scalar();
        auto sum_scalar = k1 + k2;

        auto pk_sum = G.scalar_mul(sum_scalar);
        auto pk_add = G.scalar_mul(k1).add(G.scalar_mul(k2));

        CHECK(points_equal(pk_sum, pk_add), "k1*G + k2*G == (k1+k2)*G");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 9. Schnorr pubkey vs ECDSA pubkey consistency (full flow)
// ============================================================================
static void test_schnorr_ecdsa_key_consistency() {
    g_section = "key_cons";
    printf("[9] Schnorr/ECDSA key consistency (200)\n");

    auto G = Point::generator();

    for (int i = 0; i < 200; ++i) {
        auto sk = random_scalar();

        // ECDSA full pubkey
        auto pk = G.scalar_mul(sk);
        auto pk_uncomp = pk.to_uncompressed();
        std::array<uint8_t, 32> ecdsa_x;
        std::memcpy(ecdsa_x.data(), pk_uncomp.data() + 1, 32);

        // Schnorr x-only pubkey (may negate for even Y)
        auto schnorr_x = secp256k1::schnorr_pubkey(sk);

        // One of them should match (either direct or negated)
        auto neg_pk = pk.negate();
        auto neg_uncomp = neg_pk.to_uncompressed();
        std::array<uint8_t, 32> neg_x;
        std::memcpy(neg_x.data(), neg_uncomp.data() + 1, 32);

        CHECK(schnorr_x == ecdsa_x || schnorr_x == neg_x,
              "schnorr X matches ecdsa X (or negated)");

        // Both should sign and verify correctly
        std::array<uint8_t, 32> msg{};
        uint64_t v = rng();
        std::memcpy(msg.data(), &v, 8);

        auto ecdsa_sig = secp256k1::ecdsa_sign(msg, sk);
        CHECK(secp256k1::ecdsa_verify(msg, pk, ecdsa_sig), "ECDSA verifies");

        std::array<uint8_t, 32> const aux{};
        auto schnorr_sig = secp256k1::schnorr_sign(sk, msg, aux);
        CHECK(secp256k1::schnorr_verify(schnorr_x, msg, schnorr_sig), "Schnorr verifies");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 10. Stress: mixed random protocol operations (5K)
// ============================================================================
static void test_stress_mixed() {
    g_section = "stress";
    printf("[10] Stress: mixed protocol ops (5K)\n");

    auto G = Point::generator();
    int ok_count = 0;

    for (int i = 0; i < SCALED(5000, 100); ++i) {
        auto sk = random_scalar();
        auto pk = G.scalar_mul(sk);
        std::array<uint8_t, 32> msg{};
        uint64_t v = rng();
        std::memcpy(msg.data(), &v, 8);

        int const op = static_cast<int>(rng() % 5);
        switch (op) {
        case 0: { // ECDSA round-trip
            auto sig = secp256k1::ecdsa_sign(msg, sk);
            if (secp256k1::ecdsa_verify(msg, pk, sig)) ++ok_count;
            break;
        }
        case 1: { // Schnorr round-trip
            std::array<uint8_t, 32> const aux{};
            auto pkx = secp256k1::schnorr_pubkey(sk);
            auto sig = secp256k1::schnorr_sign(sk, msg, aux);
            if (secp256k1::schnorr_verify(pkx, msg, sig)) ++ok_count;
            break;
        }
        case 2: { // ECDH
            auto sk2 = random_scalar();
            auto pk2 = G.scalar_mul(sk2);
            auto s1 = secp256k1::ecdh_compute(sk, pk2);
            auto s2 = secp256k1::ecdh_compute(sk2, pk);
            if (s1 == s2) ++ok_count;
            break;
        }
        case 3: { // Recovery
            auto rsig = secp256k1::ecdsa_sign_recoverable(msg, sk);
            auto [rec, ok] = secp256k1::ecdsa_recover(msg, rsig.sig, rsig.recid);
            if (ok && points_equal(rec, pk)) ++ok_count;
            break;
        }
        case 4: { // CT scalar mul
            auto ct_pk = secp256k1::ct::generator_mul(sk);
            if (points_equal(pk, ct_pk)) ++ok_count;
            break;
        }
        }
    }

    CHECK(ok_count == 5000, "all 5K mixed ops succeeded");
    printf("    success: %d/5000\n", ok_count);
    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// _run() entry point for unified audit runner
// ============================================================================

int audit_integration_run() {
    g_pass = 0; g_fail = 0;

    test_ecdh();
    test_schnorr_batch_verify();
    test_ecdsa_batch_verify();
    test_ecdsa_full_roundtrip();
    test_schnorr_cross_path();
    test_fast_vs_ct_integration();
    test_combined_protocol();
    test_multikey_consistency();
    test_schnorr_ecdsa_key_consistency();
    test_stress_mixed();

    return g_fail > 0 ? 1 : 0;
}

// ============================================================================
#ifndef UNIFIED_AUDIT_RUNNER
int main() {
    printf("===============================================================\n");
    printf("  AUDIT VI -- Integration Testing\n");
    printf("===============================================================\n\n");

    test_ecdh();
    test_schnorr_batch_verify();
    test_ecdsa_batch_verify();
    test_ecdsa_full_roundtrip();
    test_schnorr_cross_path();
    test_fast_vs_ct_integration();
    test_combined_protocol();
    test_multikey_consistency();
    test_schnorr_ecdsa_key_consistency();
    test_stress_mixed();

    printf("===============================================================\n");
    printf("  INTEGRATION AUDIT: %d passed, %d failed\n", g_pass, g_fail);
    printf("===============================================================\n");

    return g_fail > 0 ? 1 : 0;
}
#endif // UNIFIED_AUDIT_RUNNER
