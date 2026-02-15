// ============================================================================
// Tests: Pedersen, FROST, Adaptor, Address, Silent Payments
// ============================================================================

#include <cstdio>
#include <cstring>
#include <array>
#include <vector>
#include <string>

#include "secp256k1/pedersen.hpp"
#include "secp256k1/frost.hpp"
#include "secp256k1/adaptor.hpp"
#include "secp256k1/address.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/field.hpp"

using namespace secp256k1;
using fast::Scalar;
using fast::Point;

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, name) do { \
    if (cond) { g_pass++; std::printf("  [PASS] %s\n", name); } \
    else { g_fail++; std::printf("  [FAIL] %s (line %d)\n", name, __LINE__); } \
} while(0)

// ═══════════════════════════════════════════════════════════════════════════════
// Pedersen Commitment Tests
// ═══════════════════════════════════════════════════════════════════════════════

static void test_pedersen_basic() {
    std::printf("\n=== Pedersen Commitments ===\n");

    // Generator H should be valid non-infinity point
    const auto& H = pedersen_generator_H();
    CHECK(!H.is_infinity(), "generator_H_not_infinity");

    // H should be different from G
    auto G_comp = Point::generator().to_compressed();
    auto H_comp = H.to_compressed();
    CHECK(G_comp != H_comp, "H_differs_from_G");

    // Basic commit and verify
    Scalar value = Scalar::from_uint64(42);
    Scalar blinding = Scalar::from_uint64(12345);
    auto commitment = pedersen_commit(value, blinding);
    CHECK(!commitment.point.is_infinity(), "commitment_not_infinity");
    CHECK(commitment.verify(value, blinding), "commit_verify_roundtrip");

    // Wrong value should fail
    Scalar wrong_value = Scalar::from_uint64(43);
    CHECK(!commitment.verify(wrong_value, blinding), "wrong_value_fails");

    // Wrong blinding should fail
    Scalar wrong_blind = Scalar::from_uint64(99999);
    CHECK(!commitment.verify(value, wrong_blind), "wrong_blinding_fails");
}

static void test_pedersen_homomorphic() {
    std::printf("\n=== Pedersen Homomorphic ===\n");

    // C(v1,r1) + C(v2,r2) == C(v1+v2, r1+r2)
    Scalar v1 = Scalar::from_uint64(100);
    Scalar r1 = Scalar::from_uint64(11111);
    Scalar v2 = Scalar::from_uint64(200);
    Scalar r2 = Scalar::from_uint64(22222);

    auto c1 = pedersen_commit(v1, r1);
    auto c2 = pedersen_commit(v2, r2);
    auto c_sum = c1 + c2;
    auto c_direct = pedersen_commit(v1 + v2, r1 + r2);

    auto sum_comp = c_sum.to_compressed();
    auto dir_comp = c_direct.to_compressed();
    CHECK(sum_comp == dir_comp, "homomorphic_addition");
}

static void test_pedersen_balance() {
    std::printf("\n=== Pedersen Balance ===\n");

    // Simulate: 2 inputs (100, 200) → 2 outputs (150, 150)
    Scalar v_in1 = Scalar::from_uint64(100);
    Scalar r_in1 = Scalar::from_uint64(111);
    Scalar v_in2 = Scalar::from_uint64(200);
    Scalar r_in2 = Scalar::from_uint64(222);

    Scalar v_out1 = Scalar::from_uint64(150);
    Scalar r_out1 = Scalar::from_uint64(333);
    Scalar v_out2 = Scalar::from_uint64(150);

    // Compute balancing blinding factor
    Scalar blinds_in[] = {r_in1, r_in2};
    Scalar blinds_out[] = {r_out1};
    Scalar r_out2 = pedersen_blind_sum(blinds_in, 2, blinds_out, 1);

    auto c_in1 = pedersen_commit(v_in1, r_in1);
    auto c_in2 = pedersen_commit(v_in2, r_in2);
    auto c_out1 = pedersen_commit(v_out1, r_out1);
    auto c_out2 = pedersen_commit(v_out2, r_out2);

    PedersenCommitment pos[] = {c_in1, c_in2};
    PedersenCommitment neg[] = {c_out1, c_out2};
    CHECK(pedersen_verify_sum(pos, 2, neg, 2), "balance_proof_valid");
}

static void test_pedersen_switch() {
    std::printf("\n=== Pedersen Switch ===\n");

    const auto& J = pedersen_generator_J();
    CHECK(!J.is_infinity(), "generator_J_not_infinity");

    Scalar value = Scalar::from_uint64(777);
    Scalar blind = Scalar::from_uint64(888);
    Scalar switch_blind = Scalar::from_uint64(999);
    auto commitment = pedersen_switch_commit(value, blind, switch_blind);
    CHECK(!commitment.point.is_infinity(), "switch_commit_not_infinity");
}

static void test_pedersen_serialization() {
    std::printf("\n=== Pedersen Serialization ===\n");

    Scalar v = Scalar::from_uint64(12345);
    Scalar r = Scalar::from_uint64(67890);
    auto c = pedersen_commit(v, r);
    auto compressed = c.to_compressed();
    CHECK(compressed[0] == 0x02 || compressed[0] == 0x03, "compressed_prefix_valid");
    CHECK(compressed.size() == 33, "compressed_size_33");
}

// ═══════════════════════════════════════════════════════════════════════════════
// FROST Threshold Signature Tests
// ═══════════════════════════════════════════════════════════════════════════════

static void test_frost_lagrange() {
    std::printf("\n=== FROST Lagrange ===\n");

    // For 2-of-3, signers {1, 2}: λ_1 = 2/(2-1) = 2, λ_2 = 1/(1-2) = -1
    std::vector<ParticipantId> signers = {1, 2};
    Scalar l1 = frost_lagrange_coefficient(1, signers);
    Scalar l2 = frost_lagrange_coefficient(2, signers);

    // λ_1 should be 2
    Scalar expected_l1 = Scalar::from_uint64(2);
    CHECK(l1 == expected_l1, "lagrange_l1_equals_2");

    // λ_2 should be -1 (= n-1)
    Scalar expected_l2 = Scalar::from_uint64(1).negate();
    CHECK(l2 == expected_l2, "lagrange_l2_equals_neg1");

    // λ_1 * 1 + λ_2 * 2 should equal the secret (Lagrange property)
    // f(0) = λ_1*f(1) + λ_2*f(2) where f is degree-1 polynomial
    // Let's verify with concrete values: f(x) = 5 + 3x
    // f(1) = 8, f(2) = 11
    // λ_1*f(1) + λ_2*f(2) = 2*8 + (-1)*11 = 16 - 11 = 5 = f(0) ✓
    Scalar f1 = Scalar::from_uint64(8);
    Scalar f2 = Scalar::from_uint64(11);
    Scalar recovered = l1 * f1 + l2 * f2;
    Scalar expected = Scalar::from_uint64(5);
    CHECK(recovered == expected, "lagrange_interpolation");
}

static void test_frost_keygen() {
    std::printf("\n=== FROST Key Generation ===\n");

    const std::uint32_t T = 2;  // threshold
    const std::uint32_t N = 3;  // participants

    // Each participant generates their part
    std::array<std::uint8_t, 32> seed1 = {}; seed1[0] = 1;
    std::array<std::uint8_t, 32> seed2 = {}; seed2[0] = 2;
    std::array<std::uint8_t, 32> seed3 = {}; seed3[0] = 3;

    auto [comm1, shares1] = frost_keygen_begin(1, T, N, seed1);
    auto [comm2, shares2] = frost_keygen_begin(2, T, N, seed2);
    auto [comm3, shares3] = frost_keygen_begin(3, T, N, seed3);

    CHECK(comm1.coeffs.size() == T, "participant1_poly_degree");
    CHECK(shares1.size() == N, "participant1_share_count");

    // Each participant gets shares from others
    // Participant 1 gets: shares1[0] (own), shares2[0], shares3[0]
    std::vector<FrostCommitment> all_comms = {comm1, comm2, comm3};

    // Build received shares for participant 1: from each participant, get their share for id=1
    std::vector<FrostShare> recv1 = {shares1[0], shares2[0], shares3[0]};
    auto [pkg1, ok1] = frost_keygen_finalize(1, all_comms, recv1, T, N);
    CHECK(ok1, "participant1_keygen_ok");

    std::vector<FrostShare> recv2 = {shares1[1], shares2[1], shares3[1]};
    auto [pkg2, ok2] = frost_keygen_finalize(2, all_comms, recv2, T, N);
    CHECK(ok2, "participant2_keygen_ok");

    std::vector<FrostShare> recv3 = {shares1[2], shares2[2], shares3[2]};
    auto [pkg3, ok3] = frost_keygen_finalize(3, all_comms, recv3, T, N);
    CHECK(ok3, "participant3_keygen_ok");

    // All participants should agree on the group public key
    auto gpk1 = pkg1.group_public_key.to_compressed();
    auto gpk2 = pkg2.group_public_key.to_compressed();
    auto gpk3 = pkg3.group_public_key.to_compressed();
    CHECK(gpk1 == gpk2, "group_key_1_2_match");
    CHECK(gpk2 == gpk3, "group_key_2_3_match");
}

static void test_frost_2of3_signing() {
    std::printf("\n=== FROST 2-of-3 Signing ===\n");

    const std::uint32_t T = 2, N = 3;

    // KeyGen
    std::array<std::uint8_t, 32> seed1 = {}; seed1[0] = 0x11;
    std::array<std::uint8_t, 32> seed2 = {}; seed2[0] = 0x22;
    std::array<std::uint8_t, 32> seed3 = {}; seed3[0] = 0x33;

    auto [comm1, shares1] = frost_keygen_begin(1, T, N, seed1);
    auto [comm2, shares2] = frost_keygen_begin(2, T, N, seed2);
    auto [comm3, shares3] = frost_keygen_begin(3, T, N, seed3);

    std::vector<FrostCommitment> all_comms = {comm1, comm2, comm3};

    auto [pkg1, ok1] = frost_keygen_finalize(1, all_comms, {shares1[0], shares2[0], shares3[0]}, T, N);
    auto [pkg2, ok2] = frost_keygen_finalize(2, all_comms, {shares1[1], shares2[1], shares3[1]}, T, N);
    CHECK(ok1 && ok2, "keygen_2of3_ok");

    // Signing (signers 1 and 2)
    std::array<std::uint8_t, 32> msg = {};
    msg[0] = 0xAB; msg[1] = 0xCD;

    std::array<std::uint8_t, 32> nonce_seed1 = {}; nonce_seed1[0] = 0x41;
    std::array<std::uint8_t, 32> nonce_seed2 = {}; nonce_seed2[0] = 0x42;

    auto [nonce1, ncomm1] = frost_sign_nonce_gen(1, nonce_seed1);
    auto [nonce2, ncomm2] = frost_sign_nonce_gen(2, nonce_seed2);

    std::vector<FrostNonceCommitment> nonce_comms = {ncomm1, ncomm2};

    auto partial1 = frost_sign(pkg1, nonce1, msg, nonce_comms);
    auto partial2 = frost_sign(pkg2, nonce2, msg, nonce_comms);

    // Aggregate
    std::vector<FrostPartialSig> partials = {partial1, partial2};
    auto sig = frost_aggregate(partials, nonce_comms, pkg1.group_public_key, msg);

    // Verify with standard schnorr_verify
    auto gpk_x = schnorr_pubkey(Scalar::zero()); // we need x-only of group key
    // Actually just get x-only from group key:
    auto group_x = pkg1.group_public_key.x().to_bytes();
    bool valid = schnorr_verify(group_x, msg, sig);
    CHECK(valid, "frost_2of3_signature_valid");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Adaptor Signature Tests
// ═══════════════════════════════════════════════════════════════════════════════

static void test_schnorr_adaptor_basic() {
    std::printf("\n=== Schnorr Adaptor Basic ===\n");

    // Secret key
    Scalar sk = Scalar::from_uint64(12345678);
    Point pk = Point::generator().scalar_mul(sk);
    auto pk_x = pk.x().to_bytes();
    // Ensure even y for BIP-340
    auto pk_y = pk.y().to_bytes();
    if (pk_y[31] & 1) {
        sk = sk.negate();
        pk = pk.negate();
        pk_x = pk.x().to_bytes();
    }

    // Adaptor secret
    Scalar t = Scalar::from_uint64(87654321);
    Point T = Point::generator().scalar_mul(t);

    std::array<std::uint8_t, 32> msg = {};
    msg[0] = 0xDE; msg[1] = 0xAD;

    std::array<std::uint8_t, 32> aux = {};

    // Create adaptor pre-signature
    auto pre_sig = schnorr_adaptor_sign(sk, msg, T, aux);
    CHECK(!pre_sig.R_hat.is_infinity(), "adaptor_R_hat_not_infinity");

    // Verify pre-signature
    bool pre_valid = schnorr_adaptor_verify(pre_sig, pk_x, msg, T);
    CHECK(pre_valid, "adaptor_pre_sig_valid");

    // Adapt with secret t
    auto sig = schnorr_adaptor_adapt(pre_sig, t);

    // The adapted signature should be a valid Schnorr signature
    bool sig_valid = schnorr_verify(pk_x, msg, sig);
    CHECK(sig_valid, "adapted_sig_valid_schnorr");

    // Extract adaptor secret
    auto [extracted_t, extract_ok] = schnorr_adaptor_extract(pre_sig, sig);
    CHECK(extract_ok, "adaptor_extract_ok");
    // t or -t should match
    bool t_matches = (extracted_t == t) || (extracted_t == t.negate());
    CHECK(t_matches, "extracted_secret_matches");
}

static void test_ecdsa_adaptor_basic() {
    std::printf("\n=== ECDSA Adaptor Basic ===\n");

    Scalar sk = Scalar::from_uint64(999999);
    Point pk = Point::generator().scalar_mul(sk);

    Scalar t = Scalar::from_uint64(777777);
    Point T = Point::generator().scalar_mul(t);

    std::array<std::uint8_t, 32> msg = {};
    msg[0] = 0xCA; msg[1] = 0xFE;

    // Create ECDSA adaptor
    auto pre_sig = ecdsa_adaptor_sign(sk, msg, T);
    CHECK(!pre_sig.R_hat.is_infinity(), "ecdsa_adaptor_R_hat_valid");
    CHECK(!pre_sig.r.is_zero(), "ecdsa_adaptor_r_nonzero");

    // Verify adaptor
    bool pre_valid = ecdsa_adaptor_verify(pre_sig, pk, msg, T);
    CHECK(pre_valid, "ecdsa_adaptor_verify_ok");

    // Adapt  
    auto sig = ecdsa_adaptor_adapt(pre_sig, t);
    CHECK(!sig.r.is_zero() && !sig.s.is_zero(), "adapted_ecdsa_nonzero");

    // Extract secret
    auto [extracted_t, ok] = ecdsa_adaptor_extract(pre_sig, sig);
    CHECK(ok, "ecdsa_extract_ok");
    bool t_match = (extracted_t == t) || (extracted_t == t.negate());
    CHECK(t_match, "ecdsa_extracted_secret_matches");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Address Generation Tests
// ═══════════════════════════════════════════════════════════════════════════════

static void test_base58check() {
    std::printf("\n=== Base58Check ===\n");

    // Known vector: version=0x00, hash160=all zeros → "1111111111111111111114oLvT2"
    std::uint8_t payload[21] = {};  // version 0x00 + 20 zero bytes
    auto encoded = base58check_encode(payload, 21);
    CHECK(!encoded.empty(), "base58_encode_nonempty");
    CHECK(encoded[0] == '1', "base58_leading_ones");

    // Roundtrip
    auto [decoded, valid] = base58check_decode(encoded);
    CHECK(valid, "base58_decode_valid");
    CHECK(decoded.size() == 21, "base58_decode_size");
    bool match = true;
    for (int i = 0; i < 21; ++i) { if (decoded[i] != payload[i]) match = false; }
    CHECK(match, "base58_roundtrip");
}

static void test_bech32() {
    std::printf("\n=== Bech32/Bech32m ===\n");

    // Encode P2WPKH (v0, 20-byte program)
    std::uint8_t prog20[20] = {};
    prog20[0] = 0x75; prog20[1] = 0x1e;
    auto addr = bech32_encode("bc", 0, prog20, 20);
    CHECK(!addr.empty(), "bech32_encode_nonempty");
    CHECK(addr.substr(0, 3) == "bc1", "bech32_prefix_bc1");

    // Decode
    auto result = bech32_decode(addr);
    CHECK(result.valid, "bech32_decode_valid");
    CHECK(result.witness_version == 0, "bech32_witness_v0");
    CHECK(result.witness_program.size() == 20, "bech32_prog_20_bytes");
    CHECK(result.hrp == "bc", "bech32_hrp_bc");

    // Bech32m (P2TR, v1, 32-byte program)
    std::uint8_t prog32[32] = {};
    prog32[0] = 0xAB;
    auto addr_tr = bech32_encode("bc", 1, prog32, 32);
    CHECK(addr_tr.substr(0, 4) == "bc1p", "bech32m_prefix_bc1p");

    auto result_tr = bech32_decode(addr_tr);
    CHECK(result_tr.valid, "bech32m_decode_valid");
    CHECK(result_tr.witness_version == 1, "bech32m_witness_v1");
    CHECK(result_tr.witness_program.size() == 32, "bech32m_prog_32_bytes");
}

static void test_hash160() {
    std::printf("\n=== HASH160 ===\n");

    // HASH160 of empty string should be deterministic
    auto h1 = hash160(nullptr, 0);
    auto h2 = hash160(nullptr, 0);
    CHECK(h1 == h2, "hash160_deterministic");
    
    // Non-trivial input
    std::uint8_t data[] = {0x01, 0x02, 0x03};
    auto h3 = hash160(data, 3);
    CHECK(h3 != h1, "hash160_different_for_different_input");
}

static void test_address_p2pkh() {
    std::printf("\n=== P2PKH Address ===\n");

    Scalar sk = Scalar::from_uint64(1);
    Point pk = Point::generator().scalar_mul(sk);

    auto addr = address_p2pkh(pk, Network::Mainnet);
    CHECK(addr[0] == '1', "p2pkh_starts_with_1");
    CHECK(addr.size() >= 25 && addr.size() <= 34, "p2pkh_valid_length");

    auto addr_test = address_p2pkh(pk, Network::Testnet);
    CHECK(addr_test[0] == 'm' || addr_test[0] == 'n', "p2pkh_testnet_prefix");
}

static void test_address_p2wpkh() {
    std::printf("\n=== P2WPKH Address ===\n");

    Scalar sk = Scalar::from_uint64(1);
    Point pk = Point::generator().scalar_mul(sk);

    auto addr = address_p2wpkh(pk, Network::Mainnet);
    CHECK(addr.substr(0, 4) == "bc1q", "p2wpkh_bc1q_prefix");

    auto addr_test = address_p2wpkh(pk, Network::Testnet);
    CHECK(addr_test.substr(0, 4) == "tb1q", "p2wpkh_testnet_tb1q");

    // Roundtrip decode
    auto decoded = bech32_decode(addr);
    CHECK(decoded.valid, "p2wpkh_decode_valid");
    CHECK(decoded.witness_version == 0, "p2wpkh_version_0");
    CHECK(decoded.witness_program.size() == 20, "p2wpkh_20_byte_program");
}

static void test_address_p2tr() {
    std::printf("\n=== P2TR Address ===\n");

    Scalar sk = Scalar::from_uint64(1);
    Point pk = Point::generator().scalar_mul(sk);

    auto addr = address_p2tr(pk, Network::Mainnet);
    CHECK(addr.substr(0, 4) == "bc1p", "p2tr_bc1p_prefix");

    auto decoded = bech32_decode(addr);
    CHECK(decoded.valid, "p2tr_decode_valid");
    CHECK(decoded.witness_version == 1, "p2tr_version_1");
    CHECK(decoded.witness_program.size() == 32, "p2tr_32_byte_program");
}

static void test_wif() {
    std::printf("\n=== WIF Encode/Decode ===\n");

    Scalar sk = Scalar::from_uint64(12345);

    // Compressed mainnet
    auto wif = wif_encode(sk, true, Network::Mainnet);
    CHECK(wif[0] == 'K' || wif[0] == 'L', "wif_compressed_prefix");

    auto decoded = wif_decode(wif);
    CHECK(decoded.valid, "wif_decode_valid");
    CHECK(decoded.compressed, "wif_decode_compressed");
    CHECK(decoded.network == Network::Mainnet, "wif_network_mainnet");
    CHECK(decoded.key == sk, "wif_key_matches");

    // Uncompressed mainnet
    auto wif_unc = wif_encode(sk, false, Network::Mainnet);
    CHECK(wif_unc[0] == '5', "wif_uncompressed_prefix");

    auto dec_unc = wif_decode(wif_unc);
    CHECK(dec_unc.valid, "wif_uncompressed_valid");
    CHECK(!dec_unc.compressed, "wif_uncompressed_flag");

    // Testnet
    auto wif_test = wif_encode(sk, true, Network::Testnet);
    auto dec_test = wif_decode(wif_test);
    CHECK(dec_test.valid && dec_test.network == Network::Testnet, "wif_testnet_roundtrip");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Silent Payments Tests
// ═══════════════════════════════════════════════════════════════════════════════

static void test_silent_payment_basic() {
    std::printf("\n=== Silent Payments ===\n");

    // Receiver generates address
    Scalar b_scan = Scalar::from_uint64(111);
    Scalar b_spend = Scalar::from_uint64(222);
    auto sp_addr = silent_payment_address(b_scan, b_spend);
    CHECK(!sp_addr.scan_pubkey.is_infinity(), "sp_scan_key_valid");
    CHECK(!sp_addr.spend_pubkey.is_infinity(), "sp_spend_key_valid");

    // Encode address
    auto encoded = sp_addr.encode(Network::Mainnet);
    CHECK(!encoded.empty(), "sp_address_encoded");
    CHECK(encoded.substr(0, 2) == "sp", "sp_address_prefix");
}

static void test_silent_payment_flow() {
    std::printf("\n=== Silent Payment Flow ===\n");

    // Receiver keys
    Scalar b_scan = Scalar::from_uint64(0xBEEF);
    Scalar b_spend = Scalar::from_uint64(0xCAFE);
    auto sp_addr = silent_payment_address(b_scan, b_spend);

    // Sender creates output
    Scalar a1 = Scalar::from_uint64(0xDEAD);  // sender's input key
    std::vector<Scalar> input_keys = {a1};

    auto [output_pk, tweak] = silent_payment_create_output(input_keys, sp_addr, 0);
    CHECK(!output_pk.is_infinity(), "sp_output_key_valid");
    CHECK(!tweak.is_zero(), "sp_tweak_nonzero");

    // Receiver scans and detects
    auto A1 = Point::generator().scalar_mul(a1);
    std::vector<Point> input_pks = {A1};
    auto output_x = output_pk.x().to_bytes();
    std::vector<std::array<std::uint8_t, 32>> outputs = {output_x};

    auto detected = silent_payment_scan(b_scan, b_spend, input_pks, outputs);
    CHECK(detected.size() == 1, "sp_detected_one_output");

    if (!detected.empty()) {
        CHECK(detected[0].first == 0, "sp_detected_index_0");
        // Verify the spending key works
        Scalar d = detected[0].second;
        Point derived_pk = Point::generator().scalar_mul(d);
        auto derived_x = derived_pk.x().to_bytes();
        CHECK(derived_x == output_x, "sp_derived_key_matches_output");
    }
}

static void test_silent_payment_multiple_outputs() {
    std::printf("\n=== Silent Payment Multiple Outputs ===\n");

    Scalar b_scan = Scalar::from_uint64(0x1111);
    Scalar b_spend = Scalar::from_uint64(0x2222);
    auto sp_addr = silent_payment_address(b_scan, b_spend);

    Scalar a1 = Scalar::from_uint64(0x3333);
    std::vector<Scalar> input_keys = {a1};

    // Create 3 outputs to same recipient
    std::vector<std::array<std::uint8_t, 32>> outputs;
    for (std::uint32_t k = 0; k < 3; ++k) {
        auto [pk, tw] = silent_payment_create_output(input_keys, sp_addr, k);
        outputs.push_back(pk.x().to_bytes());
    }

    // Scan
    auto A1 = Point::generator().scalar_mul(a1);
    auto detected = silent_payment_scan(b_scan, b_spend, {A1}, outputs);
    CHECK(detected.size() == 3, "sp_detected_three_outputs");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Edge Cases
// ═══════════════════════════════════════════════════════════════════════════════

static void test_pedersen_zero_value() {
    std::printf("\n=== Edge: Zero Value Commitment ===\n");

    Scalar zero = Scalar::zero();
    Scalar blind = Scalar::from_uint64(42);
    auto c = pedersen_commit(zero, blind);

    // Should equal blind*G (since v=0, v*H = O)
    Point expected = Point::generator().scalar_mul(blind);
    auto c_comp = c.to_compressed();
    auto e_comp = expected.to_compressed();
    CHECK(c_comp == e_comp, "zero_value_equals_blind_times_G");
}

static void test_adaptor_zero_adaptor() {
    std::printf("\n=== Edge: Identity Adaptor ===\n");
    
    Scalar sk = Scalar::from_uint64(55555);
    Scalar t = Scalar::from_uint64(1);  // minimal adaptor
    Point T = Point::generator().scalar_mul(t);
    
    std::array<std::uint8_t, 32> msg = {}; msg[0] = 0xFF;
    std::array<std::uint8_t, 32> aux = {};
    
    auto pre = schnorr_adaptor_sign(sk, msg, T, aux);
    CHECK(!pre.R_hat.is_infinity(), "identity_adaptor_valid");
}

static void test_address_consistency() {
    std::printf("\n=== Address Consistency ===\n");

    // Same key should always produce same addresses
    Scalar sk = Scalar::from_uint64(7);
    Point pk = Point::generator().scalar_mul(sk);

    auto a1 = address_p2pkh(pk);
    auto a2 = address_p2pkh(pk);
    CHECK(a1 == a2, "p2pkh_deterministic");

    auto w1 = address_p2wpkh(pk);
    auto w2 = address_p2wpkh(pk);
    CHECK(w1 == w2, "p2wpkh_deterministic");

    auto t1 = address_p2tr(pk);
    auto t2 = address_p2tr(pk);
    CHECK(t1 == t2, "p2tr_deterministic");

    // Different keys → different addresses
    Scalar sk2 = Scalar::from_uint64(8);
    Point pk2 = Point::generator().scalar_mul(sk2);
    CHECK(address_p2pkh(pk) != address_p2pkh(pk2), "different_keys_different_p2pkh");
    CHECK(address_p2wpkh(pk) != address_p2wpkh(pk2), "different_keys_different_p2wpkh");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════════

int main() {
    std::printf("===========================================\n");
    std::printf("  v4.0.0 Feature Tests\n");
    std::printf("  Pedersen | FROST | Adaptor | Address | SP\n");
    std::printf("===========================================\n");

    // Pedersen
    test_pedersen_basic();
    test_pedersen_homomorphic();
    test_pedersen_balance();
    test_pedersen_switch();
    test_pedersen_serialization();
    test_pedersen_zero_value();

    // FROST
    test_frost_lagrange();
    test_frost_keygen();
    test_frost_2of3_signing();

    // Adaptor
    test_schnorr_adaptor_basic();
    test_ecdsa_adaptor_basic();
    test_adaptor_zero_adaptor();

    // Address
    test_base58check();
    test_bech32();
    test_hash160();
    test_address_p2pkh();
    test_address_p2wpkh();
    test_address_p2tr();
    test_wif();
    test_address_consistency();

    // Silent Payments
    test_silent_payment_basic();
    test_silent_payment_flow();
    test_silent_payment_multiple_outputs();

    std::printf("\n===========================================\n");
    std::printf("  Results: %d passed, %d failed\n", g_pass, g_fail);
    std::printf("===========================================\n");

    return g_fail > 0 ? 1 : 0;
}
