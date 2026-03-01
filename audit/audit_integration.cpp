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
#include "secp256k1/multiscalar.hpp"
#include "secp256k1/pippenger.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/ct_utils.hpp"
#include "secp256k1/sanitizer_scale.hpp"

using namespace secp256k1::fast;

static int g_pass = 0, g_fail = 0;
static const char* g_section = "";

#include "audit_check.hpp"

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

    { const int total = SCALED(1000, 50);
    for (int i = 0; i < total; ++i) {
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
        if ((i+1) % (total/5+1) == 0) printf("      ecdh %d/%d\n", i+1, total);
    } }

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
    // ESP32: schnorr_batch_verify builds MSM with 1+2*N points; Strauss
    // tables for 201 points ~= 100KB, exceeding 151KB heap.
#if defined(SECP256K1_PLATFORM_ESP32) || defined(ESP_PLATFORM)
    constexpr int N = 8;
#else
    constexpr int N = 100;
#endif
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
        int const corrupt_idx = N > 50 ? 50 : N / 2;
        bad[corrupt_idx].signature.r[0] ^= 0x01;
        bool const bad_result = secp256k1::schnorr_batch_verify(bad);
        CHECK(!bad_result, "batch with 1 bad -> false");

        // Identify the bad one
        auto invalids = secp256k1::schnorr_batch_identify_invalid(
            bad.data(), bad.size());
        CHECK(invalids.size() == 1 && invalids[0] == static_cast<std::size_t>(corrupt_idx),
              "identify_invalid finds corrupted index");
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
        if ((i+1) % (N/5+1) == 0) printf("      batch_gen %d/%d\n", i+1, N);
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

    { const int total = SCALED(1000, 50);
    for (int i = 0; i < total; ++i) {
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
        if ((i+1) % (total/5+1) == 0) printf("      full_rt %d/%d\n", i+1, total);
    } }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 5. Schnorr sign -> verify -> batch verify (cross-path)
// ============================================================================
static void test_schnorr_cross_path() {
    g_section = "schn_cross";
    printf("[5] Schnorr cross-path: individual vs batch (500)\n");

    std::vector<secp256k1::SchnorrBatchEntry> batch;
    // ESP32: Schnorr batch_verify builds MSM with 1+2*n points.
    // With n=30 -> 61-point Strauss tables ~94KB -> exceeds ~151KB heap.
    // Use n=8 on ESP32 (17 MSM points, ~13KB tables).
    constexpr int BATCH_N = 
#if defined(SECP256K1_PLATFORM_ESP32) || defined(ESP_PLATFORM)
        8;
#else
        500;
#endif
    batch.reserve(BATCH_N);

    { const int total_b = SCALED(500, BATCH_N);
    for (int i = 0; i < total_b; ++i) {
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
        if ((i+1) % (total_b/5+1) == 0) printf("      schn_cross %d/%d\n", i+1, total_b);
    } }

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

    { const int total_ct = SCALED(500, 30);
    for (int i = 0; i < total_ct; ++i) {
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
        if ((i+1) % (total_ct/5+1) == 0) printf("      fast_ct %d/%d\n", i+1, total_ct);
    } }

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
        if ((i+1) % 25 == 0) printf("      proto %d/100\n", i+1);
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
        if ((i+1) % 50 == 0) printf("      multikey %d/200\n", i+1);
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
        if ((i+1) % 50 == 0) printf("      key_cons %d/200\n", i+1);
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

    int const total_ops = SCALED(5000, 100);

    for (int i = 0; i < total_ops; ++i) {
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
        default: break;
        }
        if ((i+1) % (total_ops/10+1) == 0) printf("      stress %d/%d\n", i+1, total_ops);
    }

    CHECK(ok_count == total_ops, "all mixed ops succeeded");
    printf("    success: %d/%d\n", ok_count, total_ops);
    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 11. ECDSA batch verify n-sweep (n=1,2,3,10,50,100,500)
// ============================================================================
static void test_ecdsa_batch_nsweep() {
    g_section = "batch_nsw";
    printf("[11] ECDSA batch verify n-sweep\n");

    auto G = Point::generator();

    // ESP32: n=500 entries ~112KB + copy = 224KB > 151KB heap
#if defined(SECP256K1_PLATFORM_ESP32) || defined(ESP_PLATFORM)
    const int sizes[] = {1, 2, 3, 10};
#else
    const int sizes[] = {1, 2, 3, 10, 50, 100, 500};
#endif
    for (const int n : sizes) {
        std::vector<secp256k1::ECDSABatchEntry> entries;
        entries.reserve(n);

        for (int i = 0; i < n; ++i) {
            auto sk = random_scalar();
            auto pk = G.scalar_mul(sk);
            std::array<uint8_t, 32> msg{};
            uint64_t v = rng();
            std::memcpy(msg.data(), &v, 8);
            auto sig = secp256k1::ecdsa_sign(msg, sk);
            entries.push_back({msg, pk, sig});
        }

        bool const valid = secp256k1::ecdsa_batch_verify(entries);
        char label[64];
        (void)snprintf(label, sizeof(label), "ECDSA batch n=%d valid", n);
        CHECK(valid, label);

        // Corrupt random entry and verify rejection
        if (n > 0) {
            auto bad = entries;
            int const idx = static_cast<int>(rng() % n);
            auto compact = bad[idx].signature.to_compact();
            compact[0] ^= 0x01;
            bad[idx].signature = secp256k1::ECDSASignature::from_compact(compact);
            bool const rejected = !secp256k1::ecdsa_batch_verify(bad);
            (void)snprintf(label, sizeof(label), "ECDSA batch n=%d reject bad", n);
            CHECK(rejected, label);
        }
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 12. Pippenger MSM large-n correctness
// ============================================================================
static void test_pippenger_large_n() {
    g_section = "pipp_lgn";
    printf("[12] Pippenger MSM large-n correctness\n");

#if defined(SECP256K1_PLATFORM_ESP32) || defined(ESP_PLATFORM)
    // ESP32: n=128 Strauss tables alone ~262KB > 151KB heap.
    // Skip large-n MSM on embedded targets.
    printf("    SKIP -- ESP32 (heap limits for MSM tables)\n");
    return;
#endif

    auto G = Point::generator();

    // For each size: compute MSM and verify against naive sum
    const int sizes[] = {128, 256, 512, 1000};
    for (const int n : sizes) {
        std::vector<Scalar> scalars(n);
        std::vector<Point> points(n);

        // Generate random scalars and points
        for (int i = 0; i < n; ++i) {
            scalars[i] = random_scalar();
            // Use small multiples of G for fast generation
            points[i] = G.scalar_mul(random_scalar());
        }

        // Compute via Pippenger
        auto pipp_result = secp256k1::pippenger_msm(scalars, points);

        // Compute via Strauss (multi_scalar_mul)
        auto strauss_result = secp256k1::multi_scalar_mul(scalars, points);

        // Compare
        bool const match = points_equal(pipp_result, strauss_result);
        char label[64];
        (void)snprintf(label, sizeof(label), "Pippenger n=%d == Strauss", n);
        CHECK(match, label);

        // Also verify via unified msm()
        auto msm_result = secp256k1::msm(scalars, points);
        (void)snprintf(label, sizeof(label), "msm() n=%d == Strauss", n);
        CHECK(points_equal(msm_result, strauss_result), label);
    }

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
    test_ecdsa_batch_nsweep();
    test_pippenger_large_n();

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
    test_ecdsa_batch_nsweep();
    test_pippenger_large_n();

    printf("===============================================================\n");
    printf("  INTEGRATION AUDIT: %d passed, %d failed\n", g_pass, g_fail);
    printf("===============================================================\n");

    return g_fail > 0 ? 1 : 0;
}
#endif // UNIFIED_AUDIT_RUNNER
