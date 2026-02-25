// ============================================================================
// MuSig2 + FROST Protocol Tests (Phase II Tasks 2.1.1-2.2.2)
// ============================================================================
// - MuSig2 (BIP-327 style): key aggregation, nonce flow, partial signing,
//   partial verification, signature aggregation, Schnorr verify.
// - FROST: DKG simulation, nonce gen, partial signing, partial verification,
//   aggregation, Schnorr verify.
// - Multi-party: 2, 3, 5 signers for MuSig2; 2-of-3, 3-of-5 for FROST.
//
// Note: Our MuSig2 uses x-only (32-byte) pubkeys for hash inputs rather than
//       plain (33-byte compressed) keys as BIP-327 specifies. This means the
//       intermediate hash values (L, coefficients) may differ from BIP-327
//       reference vectors. The mathematical protocol structure is identical;
//       end-to-end correctness is verified via schnorr_verify().
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <array>
#include <vector>
#include <algorithm>
#include <random>

#include "secp256k1/musig2.hpp"
#include "secp256k1/frost.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/field.hpp"

using secp256k1::fast::Scalar;
using secp256k1::fast::Point;
using secp256k1::fast::FieldElement;

// -- Minimal test harness -----------------------------------------------------

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, label) do { \
    if (cond) { ++g_pass; } else { \
        ++g_fail; \
        std::printf("  FAIL: %s (line %d)\n", label, __LINE__); \
    } \
} while(0)

// -- Helpers ------------------------------------------------------------------

static std::array<uint8_t, 32> random32(std::mt19937_64& rng) {
    std::array<uint8_t, 32> out{};
    for (int i = 0; i < 4; ++i) {
        uint64_t v = rng();
        std::memcpy(out.data() + i * 8, &v, 8);
    }
    return out;
}

// Generate a valid private key (non-zero, < n)
static Scalar random_privkey(std::mt19937_64& rng) {
    for (;;) {
        auto bytes = random32(rng);
        auto sk = Scalar::from_bytes(bytes);
        if (!sk.is_zero()) return sk;
    }
}

// Get x-only pubkey from private key
static std::array<uint8_t, 32> xonly_pubkey(const Scalar& sk) {
    auto P = Point::generator().scalar_mul(sk);
    return P.x().to_bytes();
}

// ===============================================================================
// MuSig2 Tests
// ===============================================================================

// -- Test 1: Key Aggregation -- Determinism ------------------------------------

static void test_musig2_key_agg_determinism() {
    std::printf("[1] MuSig2 Key Aggregation: Determinism\n");

    std::mt19937_64 rng(0xDEADBEEF);
    const int N = 50;

    for (int round = 0; round < N; ++round) {
        int n_signers = 2 + (round % 4); // 2,3,4,5
        std::vector<Scalar> sks;
        std::vector<std::array<uint8_t, 32>> pks;
        for (int i = 0; i < n_signers; ++i) {
            auto sk = random_privkey(rng);
            sks.push_back(sk);
            pks.push_back(xonly_pubkey(sk));
        }

        auto ctx1 = secp256k1::musig2_key_agg(pks);
        auto ctx2 = secp256k1::musig2_key_agg(pks);

        CHECK(ctx1.Q_x == ctx2.Q_x, "agg key deterministic");
        CHECK(ctx1.Q_negated == ctx2.Q_negated, "negation flag deterministic");
        for (int i = 0; i < n_signers; ++i) {
            CHECK(ctx1.key_coefficients[i].to_bytes() ==
                  ctx2.key_coefficients[i].to_bytes(),
                  "coefficient deterministic");
        }
    }

    std::printf("    %d checks OK\n\n", g_pass);
}

// -- Test 2: Key Aggregation -- Ordering Matters ------------------------------

static void test_musig2_key_agg_ordering() {
    std::printf("[2] MuSig2 Key Aggregation: Ordering Matters\n");

    std::mt19937_64 rng(0xCAFEBABE);
    const int N = 20;

    for (int round = 0; round < N; ++round) {
        std::vector<std::array<uint8_t, 32>> pks;
        for (int i = 0; i < 3; ++i) {
            pks.push_back(xonly_pubkey(random_privkey(rng)));
        }

        auto ctx_fwd = secp256k1::musig2_key_agg(pks);

        // Reverse order
        auto pks_rev = pks;
        std::reverse(pks_rev.begin(), pks_rev.end());
        auto ctx_rev = secp256k1::musig2_key_agg(pks_rev);

        // Different ordering should (generally) give different agg key
        // because L = hash of concatenated keys in order
        bool same = (ctx_fwd.Q_x == ctx_rev.Q_x);
        // It's theoretically possible but astronomically unlikely
        CHECK(!same, "different ordering -> different agg key");
    }

    std::printf("    %d checks OK\n\n", g_pass);
}

// -- Test 3: Key Aggregation -- Duplicate Keys --------------------------------

static void test_musig2_key_agg_duplicates() {
    std::printf("[3] MuSig2 Key Aggregation: Duplicate Keys\n");

    std::mt19937_64 rng(0x12345678);
    auto sk = random_privkey(rng);
    auto pk = xonly_pubkey(sk);

    // Same key 3 times
    std::vector<std::array<uint8_t, 32>> pks = {pk, pk, pk};
    auto ctx = secp256k1::musig2_key_agg(pks);

    // Should not crash, should produce a valid x-only key (32 bytes)
    bool valid_key = false;
    for (int i = 0; i < 32; ++i) {
        if (ctx.Q_x[i] != 0) { valid_key = true; break; }
    }
    CHECK(valid_key, "duplicate keys produce non-zero agg key");

    // Deterministic
    auto ctx2 = secp256k1::musig2_key_agg(pks);
    CHECK(ctx.Q_x == ctx2.Q_x, "duplicate keys deterministic");

    std::printf("    %d checks OK\n\n", g_pass);
}

// -- Test 4: MuSig2 Full Round-Trip (parametric N signers) -------------------

static void test_musig2_round_trip(int n_signers, const char* label) {
    std::printf("[4.%s] MuSig2 Full Round-Trip: %d signers\n", label, n_signers);

    std::mt19937_64 rng(0xBEEF0000 + static_cast<uint32_t>(n_signers));

    const int ROUNDS = 20;
    for (int round = 0; round < ROUNDS; ++round) {
        // 1. Generate keys
        std::vector<Scalar> sks;
        std::vector<std::array<uint8_t, 32>> pks;
        for (int i = 0; i < n_signers; ++i) {
            auto sk = random_privkey(rng);
            sks.push_back(sk);
            pks.push_back(xonly_pubkey(sk));
        }

        // 2. Key aggregation
        auto key_agg = secp256k1::musig2_key_agg(pks);

        // 3. Message
        auto msg = random32(rng);

        // 4. Nonce generation
        std::vector<secp256k1::MuSig2SecNonce> sec_nonces;
        std::vector<secp256k1::MuSig2PubNonce> pub_nonces;
        for (int i = 0; i < n_signers; ++i) {
            auto extra = random32(rng);
            auto [sec, pub] = secp256k1::musig2_nonce_gen(
                sks[i], pks[i], key_agg.Q_x, msg, extra.data());
            sec_nonces.push_back(sec);
            pub_nonces.push_back(pub);
        }

        // 5. Nonce aggregation
        auto agg_nonce = secp256k1::musig2_nonce_agg(pub_nonces);

        // 6. Start signing session
        auto session = secp256k1::musig2_start_sign_session(agg_nonce, key_agg, msg);

        // 7. Partial signing + partial verification
        std::vector<Scalar> partial_sigs;
        for (int i = 0; i < n_signers; ++i) {
            auto s_i = secp256k1::musig2_partial_sign(
                sec_nonces[i], sks[i], key_agg, session,
                static_cast<std::size_t>(i));
            partial_sigs.push_back(s_i);

            bool pv = secp256k1::musig2_partial_verify(
                s_i, pub_nonces[i], pks[i], key_agg, session,
                static_cast<std::size_t>(i));
            CHECK(pv, "partial sig verifies");
        }

        // 8. Aggregate
        auto sig64 = secp256k1::musig2_partial_sig_agg(partial_sigs, session);

        // 9. Final Schnorr verify against aggregated pubkey
        auto schnorr_sig = secp256k1::SchnorrSignature::from_bytes(sig64);
        bool ok = secp256k1::schnorr_verify(key_agg.Q_x, msg, schnorr_sig);
        CHECK(ok, "aggregated sig passes schnorr_verify");
    }

    std::printf("    %d checks OK\n\n", g_pass);
}

// -- Test 5: MuSig2 Wrong Signer -- Expect Failure ---------------------------

static void test_musig2_wrong_signer() {
    std::printf("[5] MuSig2: Wrong Partial Sig Fails Verify\n");

    std::mt19937_64 rng(0xBAADF00D);
    const int N = 10;

    for (int round = 0; round < N; ++round) {
        std::vector<Scalar> sks;
        std::vector<std::array<uint8_t, 32>> pks;
        for (int i = 0; i < 3; ++i) {
            sks.push_back(random_privkey(rng));
            pks.push_back(xonly_pubkey(sks.back()));
        }

        auto key_agg = secp256k1::musig2_key_agg(pks);
        auto msg = random32(rng);

        std::vector<secp256k1::MuSig2SecNonce> sec_nonces;
        std::vector<secp256k1::MuSig2PubNonce> pub_nonces;
        for (int i = 0; i < 3; ++i) {
            auto [sec, pub] = secp256k1::musig2_nonce_gen(
                sks[i], pks[i], key_agg.Q_x, msg, nullptr);
            sec_nonces.push_back(sec);
            pub_nonces.push_back(pub);
        }

        auto agg_nonce = secp256k1::musig2_nonce_agg(pub_nonces);
        auto session = secp256k1::musig2_start_sign_session(agg_nonce, key_agg, msg);

        // Signer 0 signs correctly
        auto s_0 = secp256k1::musig2_partial_sign(
            sec_nonces[0], sks[0], key_agg, session, 0);

        // Verify s_0 against signer 1's nonce/pubkey -- should fail
        bool bad_pv = secp256k1::musig2_partial_verify(
            s_0, pub_nonces[1], pks[1], key_agg, session, 1);
        CHECK(!bad_pv, "wrong signer partial verify fails");
    }

    std::printf("    %d checks OK\n\n", g_pass);
}

// -- Test 6: MuSig2 Bit-Flip Invalidates Signature --------------------------

static void test_musig2_bitflip() {
    std::printf("[6] MuSig2: Bit-Flip Invalidates Final Signature\n");

    std::mt19937_64 rng(0xFACEFEED);
    const int N = 20;

    for (int round = 0; round < N; ++round) {
        std::vector<Scalar> sks;
        std::vector<std::array<uint8_t, 32>> pks;
        for (int i = 0; i < 2; ++i) {
            sks.push_back(random_privkey(rng));
            pks.push_back(xonly_pubkey(sks.back()));
        }

        auto key_agg = secp256k1::musig2_key_agg(pks);
        auto msg = random32(rng);

        std::vector<secp256k1::MuSig2SecNonce> sec_nonces;
        std::vector<secp256k1::MuSig2PubNonce> pub_nonces;
        for (int i = 0; i < 2; ++i) {
            auto extra = random32(rng);
            auto [sec, pub] = secp256k1::musig2_nonce_gen(
                sks[i], pks[i], key_agg.Q_x, msg, extra.data());
            sec_nonces.push_back(sec);
            pub_nonces.push_back(pub);
        }

        auto agg_nonce = secp256k1::musig2_nonce_agg(pub_nonces);
        auto session = secp256k1::musig2_start_sign_session(agg_nonce, key_agg, msg);

        std::vector<Scalar> partial_sigs;
        for (int i = 0; i < 2; ++i) {
            partial_sigs.push_back(secp256k1::musig2_partial_sign(
                sec_nonces[i], sks[i], key_agg, session,
                static_cast<std::size_t>(i)));
        }

        auto sig64 = secp256k1::musig2_partial_sig_agg(partial_sigs, session);

        // Verify original
        auto sig_ok = secp256k1::SchnorrSignature::from_bytes(sig64);
        CHECK(secp256k1::schnorr_verify(key_agg.Q_x, msg, sig_ok),
              "original sig valid");

        // Flip one bit in s component
        auto sig_bad = sig64;
        sig_bad[32 + (round % 32)] ^= 0x01;
        auto sig_flipped = secp256k1::SchnorrSignature::from_bytes(sig_bad);
        CHECK(!secp256k1::schnorr_verify(key_agg.Q_x, msg, sig_flipped),
              "bitflipped sig invalid");
    }

    std::printf("    %d checks OK\n\n", g_pass);
}

// ===============================================================================
// FROST Tests
// ===============================================================================

// -- Test 7: FROST DKG -- 2-of-3 ---------------------------------------------

static void test_frost_dkg(uint32_t threshold, uint32_t n_participants,
                           const char* label) {
    std::printf("[7.%s] FROST DKG: %u-of-%u\n", label, threshold, n_participants);

    std::mt19937_64 rng(0xF0057000 + threshold * 100 + n_participants);
    const int ROUNDS = 10;

    for (int round = 0; round < ROUNDS; ++round) {
        // Phase 1: Each participant generates commitments and shares
        std::vector<secp256k1::FrostCommitment> all_commitments;
        // share_matrix[i][j] = share from participant (i+1) to participant (j+1)
        std::vector<std::vector<secp256k1::FrostShare>> share_matrix;

        for (uint32_t i = 0; i < n_participants; ++i) {
            auto seed = random32(rng);
            auto [commitment, shares] = secp256k1::frost_keygen_begin(
                i + 1, threshold, n_participants, seed);
            all_commitments.push_back(commitment);
            share_matrix.push_back(shares);
        }

        // Phase 2: Each participant collects shares destined for them
        std::vector<secp256k1::FrostKeyPackage> key_packages;
        bool all_ok = true;

        for (uint32_t i = 0; i < n_participants; ++i) {
            std::vector<secp256k1::FrostShare> my_shares;
            for (uint32_t j = 0; j < n_participants; ++j) {
                // Share from participant (j+1) for participant (i+1)
                my_shares.push_back(share_matrix[j][i]);
            }

            auto [pkg, ok] = secp256k1::frost_keygen_finalize(
                i + 1, all_commitments, my_shares, threshold, n_participants);
            all_ok = all_ok && ok;
            key_packages.push_back(pkg);
        }

        CHECK(all_ok, "DKG all participants verified shares");

        // All participants should agree on the group public key
        for (uint32_t i = 1; i < n_participants; ++i) {
            auto pk0 = key_packages[0].group_public_key.to_compressed();
            auto pki = key_packages[i].group_public_key.to_compressed();
            CHECK(pk0 == pki, "all agree on group key");
        }

        // Verify: signing_share * G == verification_share
        for (uint32_t i = 0; i < n_participants; ++i) {
            auto sG = Point::generator().scalar_mul(key_packages[i].signing_share);
            auto vs = key_packages[i].verification_share;
            CHECK(sG.to_compressed() == vs.to_compressed(),
                  "signing_share*G == verification_share");
        }
    }

    std::printf("    %d checks OK\n\n", g_pass);
}

// -- Test 8: FROST Full Signing Round-Trip -----------------------------------

static void test_frost_signing(uint32_t threshold, uint32_t n_participants,
                               const char* label) {
    std::printf("[8.%s] FROST Signing: %u-of-%u\n", label, threshold, n_participants);

    std::mt19937_64 rng(0xF5160000 + threshold * 100 + n_participants);
    const int ROUNDS = 10;

    for (int round = 0; round < ROUNDS; ++round) {
        // -- DKG ----------------------------------------------------------
        std::vector<secp256k1::FrostCommitment> all_commitments;
        std::vector<std::vector<secp256k1::FrostShare>> share_matrix;

        for (uint32_t i = 0; i < n_participants; ++i) {
            auto seed = random32(rng);
            auto [commitment, shares] = secp256k1::frost_keygen_begin(
                i + 1, threshold, n_participants, seed);
            all_commitments.push_back(commitment);
            share_matrix.push_back(shares);
        }

        std::vector<secp256k1::FrostKeyPackage> key_packages;
        for (uint32_t i = 0; i < n_participants; ++i) {
            std::vector<secp256k1::FrostShare> my_shares;
            for (uint32_t j = 0; j < n_participants; ++j) {
                my_shares.push_back(share_matrix[j][i]);
            }
            auto [pkg, ok] = secp256k1::frost_keygen_finalize(
                i + 1, all_commitments, my_shares, threshold, n_participants);
            CHECK(ok, "DKG finalize OK");
            key_packages.push_back(pkg);
        }

        // -- Select t signers (first t participants) ---------------------
        std::vector<uint32_t> signer_indices;
        for (uint32_t i = 0; i < threshold; ++i) {
            signer_indices.push_back(i);
        }

        auto msg = random32(rng);

        // -- Nonce generation --------------------------------------------
        std::vector<secp256k1::FrostNonce> nonces;
        std::vector<secp256k1::FrostNonceCommitment> nonce_commitments;

        for (uint32_t idx : signer_indices) {
            auto nonce_seed = random32(rng);
            auto [nonce, commitment] = secp256k1::frost_sign_nonce_gen(
                key_packages[idx].id, nonce_seed);
            nonces.push_back(nonce);
            nonce_commitments.push_back(commitment);
        }

        // -- Partial signing ---------------------------------------------
        std::vector<secp256k1::FrostPartialSig> partial_sigs;
        for (std::size_t si = 0; si < signer_indices.size(); ++si) {
            uint32_t idx = signer_indices[si];
            auto psig = secp256k1::frost_sign(
                key_packages[idx], nonces[si], msg, nonce_commitments);
            partial_sigs.push_back(psig);
        }

        // -- Partial verification ----------------------------------------
        for (std::size_t si = 0; si < signer_indices.size(); ++si) {
            uint32_t idx = signer_indices[si];
            bool pv = secp256k1::frost_verify_partial(
                partial_sigs[si],
                nonce_commitments[si],
                key_packages[idx].verification_share,
                msg, nonce_commitments,
                key_packages[0].group_public_key);
            CHECK(pv, "FROST partial sig verifies");
        }

        // -- Aggregation -------------------------------------------------
        auto final_sig = secp256k1::frost_aggregate(
            partial_sigs, nonce_commitments,
            key_packages[0].group_public_key, msg);

        // -- Schnorr verify against group public key ---------------------
        auto gpk_x = key_packages[0].group_public_key.x().to_bytes();
        // Ensure we're using even-Y version for BIP-340
        auto gpk_y = key_packages[0].group_public_key.y().to_bytes();
        if (gpk_y[31] & 1) {
            // Negate -- but x stays the same for x-only
        }
        bool ok = secp256k1::schnorr_verify(gpk_x, msg, final_sig);
        CHECK(ok, "FROST aggregated sig passes schnorr_verify");
    }

    std::printf("    %d checks OK\n\n", g_pass);
}

// -- Test 9: FROST -- Different Signer Subsets --------------------------------

static void test_frost_different_subsets() {
    std::printf("[9] FROST: Different 2-of-3 Subsets All Valid\n");

    std::mt19937_64 rng(0xF5AB5E70);
    const int ROUNDS = 5;

    for (int round = 0; round < ROUNDS; ++round) {
        // DKG: 2-of-3
        uint32_t threshold = 2, n_parts = 3;
        std::vector<secp256k1::FrostCommitment> all_commitments;
        std::vector<std::vector<secp256k1::FrostShare>> share_matrix;

        for (uint32_t i = 0; i < n_parts; ++i) {
            auto seed = random32(rng);
            auto [commitment, shares] = secp256k1::frost_keygen_begin(
                i + 1, threshold, n_parts, seed);
            all_commitments.push_back(commitment);
            share_matrix.push_back(shares);
        }

        std::vector<secp256k1::FrostKeyPackage> key_packages;
        for (uint32_t i = 0; i < n_parts; ++i) {
            std::vector<secp256k1::FrostShare> my_shares;
            for (uint32_t j = 0; j < n_parts; ++j) {
                my_shares.push_back(share_matrix[j][i]);
            }
            auto [pkg, ok] = secp256k1::frost_keygen_finalize(
                i + 1, all_commitments, my_shares, threshold, n_parts);
            CHECK(ok, "DKG OK");
            key_packages.push_back(pkg);
        }

        auto msg = random32(rng);
        auto gpk = key_packages[0].group_public_key;
        auto gpk_x = gpk.x().to_bytes();

        // Try all 3 possible 2-signer subsets: {1,2}, {1,3}, {2,3}
        uint32_t subsets[][2] = {{0,1}, {0,2}, {1,2}};
        for (int s = 0; s < 3; ++s) {
            uint32_t a = subsets[s][0], b = subsets[s][1];

            // Nonces
            auto seed_a = random32(rng);
            auto seed_b = random32(rng);
            auto [nonce_a, nc_a] = secp256k1::frost_sign_nonce_gen(
                key_packages[a].id, seed_a);
            auto [nonce_b, nc_b] = secp256k1::frost_sign_nonce_gen(
                key_packages[b].id, seed_b);
            std::vector<secp256k1::FrostNonceCommitment> ncs = {nc_a, nc_b};

            // Sign
            auto psig_a = secp256k1::frost_sign(key_packages[a], nonce_a, msg, ncs);
            auto psig_b = secp256k1::frost_sign(key_packages[b], nonce_b, msg, ncs);

            // Aggregate
            std::vector<secp256k1::FrostPartialSig> psigs = {psig_a, psig_b};
            auto sig = secp256k1::frost_aggregate(psigs, ncs, gpk, msg);

            // Verify
            bool ok = secp256k1::schnorr_verify(gpk_x, msg, sig);
            CHECK(ok, "subset signature valid");
        }
    }

    std::printf("    %d checks OK\n\n", g_pass);
}

// -- Test 10: FROST -- Bit-Flip Invalidates Signature -------------------------

static void test_frost_bitflip() {
    std::printf("[10] FROST: Bit-Flip Invalidates Signature\n");

    std::mt19937_64 rng(0xF5B17F11);
    const int N = 10;

    for (int round = 0; round < N; ++round) {
        // Quick 2-of-3 DKG
        uint32_t t = 2, n = 3;
        std::vector<secp256k1::FrostCommitment> comms;
        std::vector<std::vector<secp256k1::FrostShare>> smatrix;
        for (uint32_t i = 0; i < n; ++i) {
            auto seed = random32(rng);
            auto [c, s] = secp256k1::frost_keygen_begin(i+1, t, n, seed);
            comms.push_back(c);
            smatrix.push_back(s);
        }
        std::vector<secp256k1::FrostKeyPackage> pkgs;
        for (uint32_t i = 0; i < n; ++i) {
            std::vector<secp256k1::FrostShare> ms;
            for (uint32_t j = 0; j < n; ++j) ms.push_back(smatrix[j][i]);
            auto [pkg, ok] = secp256k1::frost_keygen_finalize(i+1, comms, ms, t, n);
            pkgs.push_back(pkg);
        }

        auto msg = random32(rng);
        auto gpk = pkgs[0].group_public_key;
        auto gpk_x = gpk.x().to_bytes();

        // Sign with signers 1,2
        auto [n1, nc1] = secp256k1::frost_sign_nonce_gen(1, random32(rng));
        auto [n2, nc2] = secp256k1::frost_sign_nonce_gen(2, random32(rng));
        std::vector<secp256k1::FrostNonceCommitment> ncs = {nc1, nc2};
        auto ps1 = secp256k1::frost_sign(pkgs[0], n1, msg, ncs);
        auto ps2 = secp256k1::frost_sign(pkgs[1], n2, msg, ncs);
        auto sig = secp256k1::frost_aggregate({ps1, ps2}, ncs, gpk, msg);

        // Original passes
        CHECK(secp256k1::schnorr_verify(gpk_x, msg, sig), "original valid");

        // Flip bit in s
        auto sig_bad = sig;
        auto s_bytes = sig_bad.s.to_bytes();
        s_bytes[round % 32] ^= 0x01;
        sig_bad.s = Scalar::from_bytes(s_bytes);
        CHECK(!secp256k1::schnorr_verify(gpk_x, msg, sig_bad), "bitflipped invalid");
    }

    std::printf("    %d checks OK\n\n", g_pass);
}

// -- Test 11: FROST -- Wrong Partial Sig Fails --------------------------------

static void test_frost_wrong_partial() {
    std::printf("[11] FROST: Wrong Partial Sig Fails Verify\n");

    std::mt19937_64 rng(0xF5BAD510);
    const int N = 10;

    for (int round = 0; round < N; ++round) {
        uint32_t t = 2, n = 3;
        std::vector<secp256k1::FrostCommitment> comms;
        std::vector<std::vector<secp256k1::FrostShare>> smatrix;
        for (uint32_t i = 0; i < n; ++i) {
            auto seed = random32(rng);
            auto [c, s] = secp256k1::frost_keygen_begin(i+1, t, n, seed);
            comms.push_back(c);
            smatrix.push_back(s);
        }
        std::vector<secp256k1::FrostKeyPackage> pkgs;
        for (uint32_t i = 0; i < n; ++i) {
            std::vector<secp256k1::FrostShare> ms;
            for (uint32_t j = 0; j < n; ++j) ms.push_back(smatrix[j][i]);
            auto [pkg, ok] = secp256k1::frost_keygen_finalize(i+1, comms, ms, t, n);
            pkgs.push_back(pkg);
        }

        auto msg = random32(rng);
        auto gpk = pkgs[0].group_public_key;

        auto [n1, nc1] = secp256k1::frost_sign_nonce_gen(1, random32(rng));
        auto [n2, nc2] = secp256k1::frost_sign_nonce_gen(2, random32(rng));
        std::vector<secp256k1::FrostNonceCommitment> ncs = {nc1, nc2};

        auto ps1 = secp256k1::frost_sign(pkgs[0], n1, msg, ncs);

        // Verify ps1 against signer 2's verification share -- should fail
        bool bad = secp256k1::frost_verify_partial(
            ps1, nc1, pkgs[1].verification_share, msg, ncs, gpk);
        CHECK(!bad, "wrong verification share -> partial verify fails");
    }

    std::printf("    %d checks OK\n\n", g_pass);
}

// ===============================================================================
// _run() entry point for unified audit runner
// ===============================================================================

int test_musig2_frost_protocol_run() {
    g_pass = 0; g_fail = 0;

    test_musig2_key_agg_determinism();
    test_musig2_key_agg_ordering();
    test_musig2_key_agg_duplicates();
    test_musig2_round_trip(2, "2");
    test_musig2_round_trip(3, "3");
    test_musig2_round_trip(5, "5");
    test_musig2_wrong_signer();
    test_musig2_bitflip();

    test_frost_dkg(2, 3, "2of3");
    test_frost_dkg(3, 5, "3of5");
    test_frost_signing(2, 3, "2of3");
    test_frost_signing(3, 5, "3of5");
    test_frost_different_subsets();
    test_frost_bitflip();
    test_frost_wrong_partial();

    return g_fail > 0 ? 1 : 0;
}

// ===============================================================================
// Main (standalone only)
// ===============================================================================

#ifndef UNIFIED_AUDIT_RUNNER
int main() {
    std::printf("===================================================\n");
    std::printf("  MuSig2 + FROST Protocol Tests\n");
    std::printf("===================================================\n\n");

    // MuSig2
    test_musig2_key_agg_determinism();     // [1]
    test_musig2_key_agg_ordering();         // [2]
    test_musig2_key_agg_duplicates();       // [3]
    test_musig2_round_trip(2, "2");         // [4.2]
    test_musig2_round_trip(3, "3");         // [4.3]
    test_musig2_round_trip(5, "5");         // [4.5]
    test_musig2_wrong_signer();             // [5]
    test_musig2_bitflip();                  // [6]

    // FROST
    test_frost_dkg(2, 3, "2of3");           // [7.2of3]
    test_frost_dkg(3, 5, "3of5");           // [7.3of5]
    test_frost_signing(2, 3, "2of3");       // [8.2of3]
    test_frost_signing(3, 5, "3of5");       // [8.3of5]
    test_frost_different_subsets();          // [9]
    test_frost_bitflip();                   // [10]
    test_frost_wrong_partial();             // [11]

    // Summary
    std::printf("======================================================================\n");
    std::printf("TOTAL: %d passed, %d failed\n", g_pass, g_fail);
    std::printf("======================================================================\n");

    return g_fail > 0 ? 1 : 0;
}
#endif // UNIFIED_AUDIT_RUNNER
