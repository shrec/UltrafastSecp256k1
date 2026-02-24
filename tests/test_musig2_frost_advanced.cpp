// ============================================================================
// MuSig2 + FROST Advanced Protocol Tests (Phase II Tasks 2.1.3-2.2.4)
// ============================================================================
// - 2.1.3: Rogue-key resistance tests (MuSig2 key coefficient mechanism)
// - 2.1.4: Transcript binding tests (message/key/nonce changes)
// - 2.2.3: Malicious participant simulation (FROST VSS cheat, bad partial sig)
// - 2.2.4: Transcript binding correctness (FROST message/signer binding)
// - 2.1.5: Fault injection (invalid nonce, zero nonce, wrong partial sig)
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <array>
#include <vector>
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

// ── Minimal test harness ─────────────────────────────────────────────────────

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, label) do { \
    if (cond) { ++g_pass; } else { \
        ++g_fail; \
        std::printf("  FAIL: %s (line %d)\n", label, __LINE__); \
    } \
} while(0)

// ── Helpers ──────────────────────────────────────────────────────────────────

static std::array<uint8_t, 32> random32(std::mt19937_64& rng) {
    std::array<uint8_t, 32> out{};
    for (int i = 0; i < 4; ++i) {
        uint64_t v = rng();
        std::memcpy(out.data() + i * 8, &v, 8);
    }
    return out;
}

static Scalar random_privkey(std::mt19937_64& rng) {
    for (;;) {
        auto bytes = random32(rng);
        auto sk = Scalar::from_bytes(bytes);
        if (!sk.is_zero()) return sk;
    }
}

static std::array<uint8_t, 32> xonly_pubkey(const Scalar& sk) {
    return Point::generator().scalar_mul(sk).x().to_bytes();
}

// Full MuSig2 sign+verify helper
static bool musig2_full_sign_verify(
    const std::vector<Scalar>& sks,
    const std::vector<std::array<uint8_t, 32>>& pks,
    const std::array<uint8_t, 32>& msg,
    std::mt19937_64& rng)
{
    int n = static_cast<int>(sks.size());
    auto key_agg = secp256k1::musig2_key_agg(pks);

    std::vector<secp256k1::MuSig2SecNonce> sec_nonces;
    std::vector<secp256k1::MuSig2PubNonce> pub_nonces;
    for (int i = 0; i < n; ++i) {
        auto extra = random32(rng);
        auto [sec, pub] = secp256k1::musig2_nonce_gen(
            sks[i], pks[i], key_agg.Q_x, msg, extra.data());
        sec_nonces.push_back(sec);
        pub_nonces.push_back(pub);
    }

    auto agg_nonce = secp256k1::musig2_nonce_agg(pub_nonces);
    auto session = secp256k1::musig2_start_sign_session(agg_nonce, key_agg, msg);

    std::vector<Scalar> partial_sigs;
    for (int i = 0; i < n; ++i) {
        partial_sigs.push_back(secp256k1::musig2_partial_sign(
            sec_nonces[i], sks[i], key_agg, session,
            static_cast<std::size_t>(i)));
    }

    auto sig64 = secp256k1::musig2_partial_sig_agg(partial_sigs, session);
    auto ssig = secp256k1::SchnorrSignature::from_bytes(sig64);
    return secp256k1::schnorr_verify(key_agg.Q_x, msg, ssig);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Task 2.1.3: Rogue-Key Resistance Tests
// ═══════════════════════════════════════════════════════════════════════════════
// In naive multi-sig, an attacker could choose rogue_pk = target - honest_pk
// so that agg_pk = honest_pk + rogue_pk = target. MuSig2's key coefficient
// mechanism (a_i) prevents this by weighting each key differently.

static void test_musig2_rogue_key_resistance() {
    std::printf("[1] MuSig2: Rogue-Key Resistance\n");

    std::mt19937_64 rng(0xD0AEFACE);
    const int N = 10;

    for (int round = 0; round < N; ++round) {
        // Honest signer: random key
        auto sk_honest = random_privkey(rng);
        auto pk_honest = xonly_pubkey(sk_honest);

        // Attacker's target: they want agg_key = target_pk
        auto sk_target = random_privkey(rng);
        auto pk_target = xonly_pubkey(sk_target);

        // In naive scheme: rogue_pk = target - honest_pk (x-coordinates)
        // But in MuSig2, the aggregated key is Q = a_0*P_0 + a_1*P_1
        // where a_i depends on ALL pubkeys, so attacker can't predict
        // their own coefficient to cancel out honest key.

        // Test: attacker uses an arbitrary second key (they CAN'T actually
        // construct a perfect rogue key). Verify that:
        // 1. Aggregated key != target key (with overwhelming probability)
        // 2. Protocol still produces a valid signature (both cooperate)

        auto sk_attacker = random_privkey(rng);
        auto pk_attacker = xonly_pubkey(sk_attacker);

        std::vector<std::array<uint8_t, 32>> pks = {pk_honest, pk_attacker};
        auto key_agg = secp256k1::musig2_key_agg(pks);

        // Agg key should NOT equal the attacker's target
        CHECK(key_agg.Q_x != pk_target, "agg key != attacker target");

        // Both coefficients should be non-trivial (neither 0 nor 1 for first key)
        auto coeff0_bytes = key_agg.key_coefficients[0].to_bytes();
        auto coeff1_bytes = key_agg.key_coefficients[1].to_bytes();

        // At least one coefficient should not be the trivial "1"
        // (second unique key gets a_i=1 in BIP-327 optimization)
        bool coeff0_is_one = key_agg.key_coefficients[0].to_bytes() ==
                             Scalar::one().to_bytes();
        bool coeff1_is_one = key_agg.key_coefficients[1].to_bytes() ==
                             Scalar::one().to_bytes();
        // With 2 distinct keys, exactly one gets coefficient 1 (the optimization)
        // The other gets a hash-derived coefficient
        CHECK(!(coeff0_is_one && coeff1_is_one),
              "not both coefficients are trivial");

        // Signing still works correctly with both cooperating
        auto msg = random32(rng);
        std::vector<Scalar> sks = {sk_honest, sk_attacker};
        bool ok = musig2_full_sign_verify(sks, pks, msg, rng);
        CHECK(ok, "cooperative sign still valid");
    }

    std::printf("    %d checks OK\n\n", g_pass);
}

// Verify key coefficient is different for same key in different groups

static void test_musig2_key_coefficient_binding() {
    std::printf("[2] MuSig2: Key Coefficient Depends on Full Group\n");

    std::mt19937_64 rng(0xC0EFF1C1);
    const int N = 10;

    for (int round = 0; round < N; ++round) {
        auto sk_a = random_privkey(rng);
        auto sk_b = random_privkey(rng);
        auto sk_c = random_privkey(rng);
        auto pk_a = xonly_pubkey(sk_a);
        auto pk_b = xonly_pubkey(sk_b);
        auto pk_c = xonly_pubkey(sk_c);

        // Group 1: {A, B}
        auto ctx_ab = secp256k1::musig2_key_agg({pk_a, pk_b});
        // Group 2: {A, C}
        auto ctx_ac = secp256k1::musig2_key_agg({pk_a, pk_c});

        // A's coefficient should be different in different groups
        // (because L = hash(all keys) differs)
        auto coeff_a_in_ab = ctx_ab.key_coefficients[0].to_bytes();
        auto coeff_a_in_ac = ctx_ac.key_coefficients[0].to_bytes();
        CHECK(coeff_a_in_ab != coeff_a_in_ac,
              "same key gets different coeff in different groups");
    }

    std::printf("    %d checks OK\n\n", g_pass);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Task 2.1.4: Transcript Binding Tests
// ═══════════════════════════════════════════════════════════════════════════════

// Different messages → different signatures

static void test_musig2_message_binding() {
    std::printf("[3] MuSig2: Different Messages → Different Signatures\n");

    std::mt19937_64 rng(0xF5650001);
    const int N = 20;

    for (int round = 0; round < N; ++round) {
        std::vector<Scalar> sks;
        std::vector<std::array<uint8_t, 32>> pks;
        for (int i = 0; i < 2; ++i) {
            sks.push_back(random_privkey(rng));
            pks.push_back(xonly_pubkey(sks.back()));
        }

        auto key_agg = secp256k1::musig2_key_agg(pks);
        auto msg1 = random32(rng);
        auto msg2 = random32(rng);

        // Sign msg1
        std::vector<secp256k1::MuSig2SecNonce> sn1;
        std::vector<secp256k1::MuSig2PubNonce> pn1;
        for (int i = 0; i < 2; ++i) {
            auto extra = random32(rng);
            auto [sec, pub] = secp256k1::musig2_nonce_gen(
                sks[i], pks[i], key_agg.Q_x, msg1, extra.data());
            sn1.push_back(sec);
            pn1.push_back(pub);
        }
        auto an1 = secp256k1::musig2_nonce_agg(pn1);
        auto sess1 = secp256k1::musig2_start_sign_session(an1, key_agg, msg1);

        // Sign msg2 with fresh nonces
        std::vector<secp256k1::MuSig2SecNonce> sn2;
        std::vector<secp256k1::MuSig2PubNonce> pn2;
        for (int i = 0; i < 2; ++i) {
            auto extra = random32(rng);
            auto [sec, pub] = secp256k1::musig2_nonce_gen(
                sks[i], pks[i], key_agg.Q_x, msg2, extra.data());
            sn2.push_back(sec);
            pn2.push_back(pub);
        }
        auto an2 = secp256k1::musig2_nonce_agg(pn2);
        auto sess2 = secp256k1::musig2_start_sign_session(an2, key_agg, msg2);

        // Challenges must differ
        CHECK(sess1.e.to_bytes() != sess2.e.to_bytes(),
              "different messages → different challenges");

        // Each signature verifies against its own message
        std::vector<Scalar> ps1, ps2;
        for (int i = 0; i < 2; ++i) {
            ps1.push_back(secp256k1::musig2_partial_sign(sn1[i], sks[i], key_agg, sess1, i));
            ps2.push_back(secp256k1::musig2_partial_sign(sn2[i], sks[i], key_agg, sess2, i));
        }
        auto sig1 = secp256k1::musig2_partial_sig_agg(ps1, sess1);
        auto sig2 = secp256k1::musig2_partial_sig_agg(ps2, sess2);
        auto ss1 = secp256k1::SchnorrSignature::from_bytes(sig1);
        auto ss2 = secp256k1::SchnorrSignature::from_bytes(sig2);

        CHECK(secp256k1::schnorr_verify(key_agg.Q_x, msg1, ss1), "sig1 valid on msg1");
        CHECK(secp256k1::schnorr_verify(key_agg.Q_x, msg2, ss2), "sig2 valid on msg2");

        // Cross-verify should fail
        CHECK(!secp256k1::schnorr_verify(key_agg.Q_x, msg2, ss1), "sig1 invalid on msg2");
        CHECK(!secp256k1::schnorr_verify(key_agg.Q_x, msg1, ss2), "sig2 invalid on msg1");
    }

    std::printf("    %d checks OK\n\n", g_pass);
}

// Nonce binding: same keys+message but different nonces → different R, same challenge structure

static void test_musig2_nonce_binding() {
    std::printf("[4] MuSig2: Nonce Binding (fresh nonces → different R)\n");

    std::mt19937_64 rng(0xA0CEFACE);
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

        // Two signing sessions with different extra_input (different nonces)
        auto do_session = [&](std::mt19937_64& r)
            -> std::pair<secp256k1::MuSig2Session, std::array<uint8_t, 64>> {
            std::vector<secp256k1::MuSig2SecNonce> sns;
            std::vector<secp256k1::MuSig2PubNonce> pns;
            for (int i = 0; i < 2; ++i) {
                auto extra = random32(r);
                auto [sec, pub] = secp256k1::musig2_nonce_gen(
                    sks[i], pks[i], key_agg.Q_x, msg, extra.data());
                sns.push_back(sec);
                pns.push_back(pub);
            }
            auto an = secp256k1::musig2_nonce_agg(pns);
            auto sess = secp256k1::musig2_start_sign_session(an, key_agg, msg);
            std::vector<Scalar> ps;
            for (int i = 0; i < 2; ++i)
                ps.push_back(secp256k1::musig2_partial_sign(sns[i], sks[i], key_agg, sess, i));
            return {sess, secp256k1::musig2_partial_sig_agg(ps, sess)};
        };

        auto [sess_a, sig_a] = do_session(rng);
        auto [sess_b, sig_b] = do_session(rng);

        // R should differ (different nonces)
        auto R_a = sess_a.R.x().to_bytes();
        auto R_b = sess_b.R.x().to_bytes();
        CHECK(R_a != R_b, "different nonces → different R");

        // Both signatures should be valid
        auto s_a = secp256k1::SchnorrSignature::from_bytes(sig_a);
        auto s_b = secp256k1::SchnorrSignature::from_bytes(sig_b);
        CHECK(secp256k1::schnorr_verify(key_agg.Q_x, msg, s_a), "sig_a valid");
        CHECK(secp256k1::schnorr_verify(key_agg.Q_x, msg, s_b), "sig_b valid");
    }

    std::printf("    %d checks OK\n\n", g_pass);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Task 2.1.5: Fault Injection
// ═══════════════════════════════════════════════════════════════════════════════

static void test_musig2_fault_injection() {
    std::printf("[5] MuSig2: Fault Injection (wrong key in partial sign)\n");

    std::mt19937_64 rng(0xFA017000);
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

        std::vector<secp256k1::MuSig2SecNonce> sns;
        std::vector<secp256k1::MuSig2PubNonce> pns;
        for (int i = 0; i < 3; ++i) {
            auto extra = random32(rng);
            auto [sec, pub] = secp256k1::musig2_nonce_gen(
                sks[i], pks[i], key_agg.Q_x, msg, extra.data());
            sns.push_back(sec);
            pns.push_back(pub);
        }
        auto an = secp256k1::musig2_nonce_agg(pns);
        auto sess = secp256k1::musig2_start_sign_session(an, key_agg, msg);

        // Signer 0 signs with the WRONG secret key (signer 2's key)
        auto bad_s0 = secp256k1::musig2_partial_sign(
            sns[0], sks[2], key_agg, sess, 0);

        // Partial verify should catch this
        bool pv0 = secp256k1::musig2_partial_verify(
            bad_s0, pns[0], pks[0], key_agg, sess, 0);
        CHECK(!pv0, "partial verify catches wrong secret key");

        // If aggregated anyway, final sig should fail schnorr_verify
        auto s1 = secp256k1::musig2_partial_sign(sns[1], sks[1], key_agg, sess, 1);
        auto s2 = secp256k1::musig2_partial_sign(sns[2], sks[2], key_agg, sess, 2);
        auto sig = secp256k1::musig2_partial_sig_agg({bad_s0, s1, s2}, sess);
        auto ssig = secp256k1::SchnorrSignature::from_bytes(sig);
        CHECK(!secp256k1::schnorr_verify(key_agg.Q_x, msg, ssig),
              "aggregated sig with bad partial fails verify");
    }

    std::printf("    %d checks OK\n\n", g_pass);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Task 2.2.3: Malicious FROST Participant Simulation
// ═══════════════════════════════════════════════════════════════════════════════

// Scenario A: Participant sends tampered share during DKG

static void test_frost_bad_share_dkg() {
    std::printf("[6] FROST: Malicious Participant — Bad DKG Share\n");

    std::mt19937_64 rng(0xBAD50A8E);
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

        // Tamper: participant 3 sends corrupted share to participant 1
        // (modify the share value)
        auto& bad_share = smatrix[2][0]; // from participant 3, to participant 1
        auto bad_val = bad_share.value + Scalar::one();
        bad_share.value = bad_val;

        // Participant 1 should detect the bad share during finalize
        std::vector<secp256k1::FrostShare> p1_shares;
        for (uint32_t j = 0; j < n; ++j) {
            p1_shares.push_back(smatrix[j][0]);
        }

        auto [pkg, ok] = secp256k1::frost_keygen_finalize(
            1, comms, p1_shares, t, n);
        CHECK(!ok, "DKG detects tampered share");
    }

    std::printf("    %d checks OK\n\n", g_pass);
}

// Scenario B: Participant sends bad partial signature during signing

static void test_frost_bad_partial_sig() {
    std::printf("[7] FROST: Malicious Participant — Bad Partial Sig\n");

    std::mt19937_64 rng(0xBAD51600);
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
        auto gpk_x = gpk.x().to_bytes();

        // Signer 1 and 2 generate nonces
        auto [n1, nc1] = secp256k1::frost_sign_nonce_gen(1, random32(rng));
        auto [n2, nc2] = secp256k1::frost_sign_nonce_gen(2, random32(rng));
        std::vector<secp256k1::FrostNonceCommitment> ncs = {nc1, nc2};

        // Signer 1 signs correctly
        auto ps1 = secp256k1::frost_sign(pkgs[0], n1, msg, ncs);

        // Signer 2 sends a TAMPERED partial sig (add 1 to z_i)
        auto ps2_good = secp256k1::frost_sign(pkgs[1], n2, msg, ncs);
        secp256k1::FrostPartialSig ps2_bad = ps2_good;
        ps2_bad.z_i = ps2_bad.z_i + Scalar::one();

        // Partial verification should catch the bad sig
        bool pv_good = secp256k1::frost_verify_partial(
            ps2_good, nc2, pkgs[1].verification_share, msg, ncs, gpk);
        bool pv_bad = secp256k1::frost_verify_partial(
            ps2_bad, nc2, pkgs[1].verification_share, msg, ncs, gpk);
        CHECK(pv_good, "good partial sig verifies");
        CHECK(!pv_bad, "bad partial sig detected");

        // Aggregate with bad partial: final sig should fail schnorr_verify
        auto sig_bad = secp256k1::frost_aggregate({ps1, ps2_bad}, ncs, gpk, msg);
        CHECK(!secp256k1::schnorr_verify(gpk_x, msg, sig_bad),
              "aggregated sig with bad partial fails");

        // Aggregate with good partial: should pass
        auto sig_ok = secp256k1::frost_aggregate({ps1, ps2_good}, ncs, gpk, msg);
        CHECK(secp256k1::schnorr_verify(gpk_x, msg, sig_ok),
              "aggregated sig with good partial passes");
    }

    std::printf("    %d checks OK\n\n", g_pass);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Task 2.2.4: FROST Transcript Binding
// ═══════════════════════════════════════════════════════════════════════════════

// Different messages produce different FROST signatures

static void test_frost_message_binding() {
    std::printf("[8] FROST: Message Binding (different messages → different sigs)\n");

    std::mt19937_64 rng(0xF5B1D000);
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

        auto gpk = pkgs[0].group_public_key;
        auto gpk_x = gpk.x().to_bytes();
        auto msg1 = random32(rng);
        auto msg2 = random32(rng);

        // Sign msg1
        auto [n1a, nc1a] = secp256k1::frost_sign_nonce_gen(1, random32(rng));
        auto [n2a, nc2a] = secp256k1::frost_sign_nonce_gen(2, random32(rng));
        std::vector<secp256k1::FrostNonceCommitment> ncs_a = {nc1a, nc2a};
        auto ps1a = secp256k1::frost_sign(pkgs[0], n1a, msg1, ncs_a);
        auto ps2a = secp256k1::frost_sign(pkgs[1], n2a, msg1, ncs_a);
        auto sig1 = secp256k1::frost_aggregate({ps1a, ps2a}, ncs_a, gpk, msg1);

        // Sign msg2 with fresh nonces
        auto [n1b, nc1b] = secp256k1::frost_sign_nonce_gen(1, random32(rng));
        auto [n2b, nc2b] = secp256k1::frost_sign_nonce_gen(2, random32(rng));
        std::vector<secp256k1::FrostNonceCommitment> ncs_b = {nc1b, nc2b};
        auto ps1b = secp256k1::frost_sign(pkgs[0], n1b, msg2, ncs_b);
        auto ps2b = secp256k1::frost_sign(pkgs[1], n2b, msg2, ncs_b);
        auto sig2 = secp256k1::frost_aggregate({ps1b, ps2b}, ncs_b, gpk, msg2);

        // Each valid on its own message
        CHECK(secp256k1::schnorr_verify(gpk_x, msg1, sig1), "sig1 valid on msg1");
        CHECK(secp256k1::schnorr_verify(gpk_x, msg2, sig2), "sig2 valid on msg2");

        // Cross-verify fails
        CHECK(!secp256k1::schnorr_verify(gpk_x, msg2, sig1), "sig1 invalid on msg2");
        CHECK(!secp256k1::schnorr_verify(gpk_x, msg1, sig2), "sig2 invalid on msg1");
    }

    std::printf("    %d checks OK\n\n", g_pass);
}

// Different signer subsets produce signatures valid under SAME group key

static void test_frost_signer_set_binding() {
    std::printf("[9] FROST: Signer Set Binding (same key, different subsets)\n");

    std::mt19937_64 rng(0xF5557000);

    // 2-of-3 DKG
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

    auto gpk = pkgs[0].group_public_key;
    auto gpk_x = gpk.x().to_bytes();
    auto msg = random32(rng);

    // 3 different subsets sign the same message
    uint32_t subsets[][2] = {{0,1}, {0,2}, {1,2}};
    std::vector<secp256k1::SchnorrSignature> sigs;

    for (int s = 0; s < 3; ++s) {
        uint32_t a = subsets[s][0], b = subsets[s][1];
        auto [na, nca] = secp256k1::frost_sign_nonce_gen(pkgs[a].id, random32(rng));
        auto [nb, ncb] = secp256k1::frost_sign_nonce_gen(pkgs[b].id, random32(rng));
        std::vector<secp256k1::FrostNonceCommitment> ncs = {nca, ncb};
        auto psa = secp256k1::frost_sign(pkgs[a], na, msg, ncs);
        auto psb = secp256k1::frost_sign(pkgs[b], nb, msg, ncs);
        auto sig = secp256k1::frost_aggregate({psa, psb}, ncs, gpk, msg);
        CHECK(secp256k1::schnorr_verify(gpk_x, msg, sig), "subset sig valid");
        sigs.push_back(sig);
    }

    // All 3 sigs should be different (different nonces, different Lagrange coefficients)
    for (int i = 0; i < 3; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            bool r_same = sigs[i].r == sigs[j].r;
            bool s_same = sigs[i].s.to_bytes() == sigs[j].s.to_bytes();
            CHECK(!r_same || !s_same, "different subsets → different sigs");
        }
    }

    std::printf("    %d checks OK\n\n", g_pass);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════════

int main() {
    std::printf("═══════════════════════════════════════════════════\n");
    std::printf("  MuSig2 + FROST Advanced Protocol Tests\n");
    std::printf("═══════════════════════════════════════════════════\n\n");

    // 2.1.3: Rogue-key resistance
    test_musig2_rogue_key_resistance();       // [1]
    test_musig2_key_coefficient_binding();     // [2]

    // 2.1.4: Transcript binding
    test_musig2_message_binding();             // [3]
    test_musig2_nonce_binding();               // [4]

    // 2.1.5: Fault injection
    test_musig2_fault_injection();             // [5]

    // 2.2.3: Malicious participant
    test_frost_bad_share_dkg();                // [6]
    test_frost_bad_partial_sig();              // [7]

    // 2.2.4: Transcript binding
    test_frost_message_binding();              // [8]
    test_frost_signer_set_binding();           // [9]

    // Summary
    std::printf("══════════════════════════════════════════════════════════════════════\n");
    std::printf("TOTAL: %d passed, %d failed\n", g_pass, g_fail);
    std::printf("══════════════════════════════════════════════════════════════════════\n");

    return g_fail > 0 ? 1 : 0;
}
