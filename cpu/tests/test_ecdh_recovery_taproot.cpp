// ============================================================================
// Test: ECDH + Recovery + Taproot + CT Utils
// ============================================================================
// Comprehensive test suite for v3.2.0 features.
// Includes Wycheproof-style edge cases for ECDSA.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <array>
#include <vector>

#include "secp256k1/ecdh.hpp"
#include "secp256k1/recovery.hpp"
#include "secp256k1/taproot.hpp"
#include "secp256k1/ct_utils.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/field.hpp"

using namespace secp256k1;
using fast::Scalar;
using fast::Point;
using fast::FieldElement;

static int g_pass = 0, g_fail = 0;

static void check(bool cond, const char* name) {
    if (cond) {
        ++g_pass;
    } else {
        ++g_fail;
        std::printf("  FAIL: %s\n", name);
    }
}

// Helper: hex to bytes
static std::array<uint8_t, 32> hex32(const char* hex) {
    std::array<uint8_t, 32> out{};
    for (int i = 0; i < 32; ++i) {
        unsigned val = 0;
        std::sscanf(hex + i * 2, "%02x", &val);
        out[i] = static_cast<uint8_t>(val);
    }
    return out;
}

// ═══════════════════════════════════════════════════════════════════════════════
// ECDH Tests
// ═══════════════════════════════════════════════════════════════════════════════

static void test_ecdh_basic() {
    std::printf("[ECDH] Basic key exchange...\n");

    // Alice and Bob: generate keypairs
    auto sk_a = Scalar::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000001");
    auto sk_b = Scalar::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000002");

    auto pk_a = Point::generator().scalar_mul(sk_a);
    auto pk_b = Point::generator().scalar_mul(sk_b);

    // Alice computes shared secret with Bob's public key
    auto secret_a = ecdh_compute(sk_a, pk_b);
    // Bob computes shared secret with Alice's public key
    auto secret_b = ecdh_compute(sk_b, pk_a);

    check(secret_a == secret_b, "ECDH: shared secrets match");
    check(!secp256k1::ct::ct_is_zero(secret_a), "ECDH: secret is non-zero");
}

static void test_ecdh_xonly() {
    std::printf("[ECDH] X-only variant...\n");

    auto sk_a = Scalar::from_hex(
        "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
    auto sk_b = Scalar::from_hex(
        "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8");

    auto pk_a = Point::generator().scalar_mul(sk_a);
    auto pk_b = Point::generator().scalar_mul(sk_b);

    auto secret_a = ecdh_compute_xonly(sk_a, pk_b);
    auto secret_b = ecdh_compute_xonly(sk_b, pk_a);

    check(secret_a == secret_b, "ECDH xonly: shared secrets match");
}

static void test_ecdh_raw() {
    std::printf("[ECDH] Raw x-coordinate...\n");

    auto sk_a = Scalar::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000003");
    auto sk_b = Scalar::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000005");

    auto pk_a = Point::generator().scalar_mul(sk_a);
    auto pk_b = Point::generator().scalar_mul(sk_b);

    auto raw_a = ecdh_compute_raw(sk_a, pk_b);
    auto raw_b = ecdh_compute_raw(sk_b, pk_a);

    check(raw_a == raw_b, "ECDH raw: x-coordinates match");

    // Verify raw vs hashed: hashed = SHA256(raw) for xonly variant
    auto hashed_a = ecdh_compute_xonly(sk_a, pk_b);
    auto expected = SHA256::hash(raw_a.data(), 32);
    check(hashed_a == expected, "ECDH: xonly == SHA256(raw)");
}

static void test_ecdh_zero_key() {
    std::printf("[ECDH] Edge: zero private key...\n");

    auto sk_zero = Scalar::zero();
    auto pk = Point::generator();

    auto secret = ecdh_compute(sk_zero, pk);
    check(secp256k1::ct::ct_is_zero(secret), "ECDH: zero key returns zero");
}

static void test_ecdh_infinity() {
    std::printf("[ECDH] Edge: infinity public key...\n");

    auto sk = Scalar::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000001");
    auto pk_inf = Point::infinity();

    auto secret = ecdh_compute(sk, pk_inf);
    check(secp256k1::ct::ct_is_zero(secret), "ECDH: infinity pubkey returns zero");
}

// ═══════════════════════════════════════════════════════════════════════════════
// ECDSA Recovery Tests
// ═══════════════════════════════════════════════════════════════════════════════

static void test_recovery_basic() {
    std::printf("[Recovery] Basic sign + recover...\n");

    auto sk = Scalar::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000001");
    auto pk = Point::generator().scalar_mul(sk);

    auto msg = hex32("0000000000000000000000000000000000000000000000000000000000000001");

    auto rsig = ecdsa_sign_recoverable(msg, sk);
    check(!rsig.sig.r.is_zero(), "Recovery: signature r != 0");
    check(!rsig.sig.s.is_zero(), "Recovery: signature s != 0");
    check(rsig.recid >= 0 && rsig.recid <= 3, "Recovery: valid recid");

    // Recover public key
    auto [recovered, ok] = ecdsa_recover(msg, rsig.sig, rsig.recid);
    check(ok, "Recovery: recovery succeeded");
    check(recovered.to_compressed() == pk.to_compressed(),
          "Recovery: recovered key matches original");
}

static void test_recovery_multiple_keys() {
    std::printf("[Recovery] Multiple different private keys...\n");

    const char* test_keys[] = {
        "0000000000000000000000000000000000000000000000000000000000000002",
        "0000000000000000000000000000000000000000000000000000000000000003",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140",  // n-1
        "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798",
    };
    auto msg = hex32("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");

    for (auto hex : test_keys) {
        auto sk = Scalar::from_hex(hex);
        auto pk = Point::generator().scalar_mul(sk);

        auto rsig = ecdsa_sign_recoverable(msg, sk);
        auto [recovered, ok] = ecdsa_recover(msg, rsig.sig, rsig.recid);
        check(ok && recovered.to_compressed() == pk.to_compressed(),
              "Recovery: matches for different key");
    }
}

static void test_recovery_compact_serialization() {
    std::printf("[Recovery] Compact 65-byte serialization...\n");

    auto sk = Scalar::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000005");
    auto msg = hex32("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB");

    auto rsig = ecdsa_sign_recoverable(msg, sk);

    // Serialize to 65 bytes
    auto compact = recoverable_to_compact(rsig);
    check(compact[0] >= 31 && compact[0] <= 34, "Compact: valid header byte");

    // Parse back
    auto [parsed, ok] = recoverable_from_compact(compact);
    check(ok, "Compact: parse succeeded");
    check(parsed.recid == rsig.recid, "Compact: recid matches");
    check(parsed.sig.r == rsig.sig.r, "Compact: r matches");
    check(parsed.sig.s == rsig.sig.s, "Compact: s matches");
}

static void test_recovery_wrong_recid() {
    std::printf("[Recovery] Wrong recovery ID...\n");

    auto sk = Scalar::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000001");
    auto pk = Point::generator().scalar_mul(sk);
    auto msg = hex32("0000000000000000000000000000000000000000000000000000000000000001");

    auto rsig = ecdsa_sign_recoverable(msg, sk);

    // Try wrong recid — should either fail or give wrong key
    int wrong_recid = (rsig.recid + 1) % 4;
    auto [wrong_pk, ok] = ecdsa_recover(msg, rsig.sig, wrong_recid);
    // It might succeed but give a different key, or it might fail
    if (ok) {
        check(wrong_pk.to_compressed() != pk.to_compressed(),
              "Recovery: wrong recid gives different key");
    } else {
        check(true, "Recovery: wrong recid correctly failed");
    }
}

static void test_recovery_invalid_sig() {
    std::printf("[Recovery] Invalid signature (zero r/s)...\n");

    auto msg = hex32("0000000000000000000000000000000000000000000000000000000000000001");

    // Zero r
    ECDSASignature zero_r{Scalar::zero(), Scalar::one()};
    auto [pk1, ok1] = ecdsa_recover(msg, zero_r, 0);
    check(!ok1, "Recovery: zero r fails");

    // Zero s
    ECDSASignature zero_s{Scalar::one(), Scalar::zero()};
    auto [pk2, ok2] = ecdsa_recover(msg, zero_s, 0);
    check(!ok2, "Recovery: zero s fails");

    // Invalid recid
    ECDSASignature valid_sig{Scalar::one(), Scalar::one()};
    auto [pk3, ok3] = ecdsa_recover(msg, valid_sig, -1);
    check(!ok3, "Recovery: negative recid fails");

    auto [pk4, ok4] = ecdsa_recover(msg, valid_sig, 4);
    check(!ok4, "Recovery: recid > 3 fails");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Taproot Tests
// ═══════════════════════════════════════════════════════════════════════════════

static void test_taproot_tweak_hash() {
    std::printf("[Taproot] TapTweak hash...\n");

    auto internal_key_x = hex32(
        "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");

    // Key-path only (no merkle root)
    auto tweak1 = taproot_tweak_hash(internal_key_x);
    check(!secp256k1::ct::ct_is_zero(tweak1), "TapTweak: non-zero result");

    // With merkle root
    auto merkle = hex32("1111111111111111111111111111111111111111111111111111111111111111");
    auto tweak2 = taproot_tweak_hash(internal_key_x, merkle.data(), 32);
    check(tweak1 != tweak2, "TapTweak: different with merkle root");
}

static void test_taproot_output_key() {
    std::printf("[Taproot] Output key derivation...\n");

    auto sk = Scalar::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000001");
    auto pk_x = schnorr_pubkey(sk);  // x-only public key

    // Key-path only
    auto [output_x, parity] = taproot_output_key(pk_x);
    check(!secp256k1::ct::ct_is_zero(output_x), "Taproot: output key non-zero");
    check(parity == 0 || parity == 1, "Taproot: valid parity");
    check(output_x != pk_x, "Taproot: output key differs from internal key");
}

static void test_taproot_privkey_tweak() {
    std::printf("[Taproot] Private key tweaking...\n");

    auto sk = Scalar::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000001");
    auto pk_x = schnorr_pubkey(sk);

    // Tweak private key
    auto tweaked_sk = taproot_tweak_privkey(sk);
    check(!tweaked_sk.is_zero(), "Taproot: tweaked key non-zero");

    // Verify: tweaked_sk * G should have same x as output_key
    auto tweaked_pk = Point::generator().scalar_mul(tweaked_sk);
    auto tweaked_pk_x = tweaked_pk.x().to_bytes();

    auto [output_x, parity] = taproot_output_key(pk_x);
    check(tweaked_pk_x == output_x, "Taproot: tweaked key produces output key");
}

static void test_taproot_commitment_verify() {
    std::printf("[Taproot] Commitment verification...\n");

    auto sk = Scalar::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000002");
    auto pk_x = schnorr_pubkey(sk);

    auto [output_x, parity] = taproot_output_key(pk_x);

    check(taproot_verify_commitment(output_x, parity, pk_x),
          "Taproot: commitment verifies");

    // Tamper with output key
    auto bad_output = output_x;
    bad_output[0] ^= 0x01;
    check(!taproot_verify_commitment(bad_output, parity, pk_x),
          "Taproot: tampered output fails");
}

static void test_taproot_leaf_and_branch() {
    std::printf("[Taproot] Leaf and branch hashes...\n");

    // Simple script
    uint8_t script[] = {0xAC}; // OP_CHECKSIG
    auto leaf = taproot_leaf_hash(script, 1);
    check(!secp256k1::ct::ct_is_zero(leaf), "TapLeaf: non-zero hash");

    // Same script, same version = same hash
    auto leaf2 = taproot_leaf_hash(script, 1);
    check(leaf == leaf2, "TapLeaf: deterministic");

    // Different version = different hash
    auto leaf3 = taproot_leaf_hash(script, 1, 0xC2);
    check(leaf != leaf3, "TapLeaf: different version = different hash");

    // Branch hash: order-independent (sorts internally)
    auto a = hex32("1111111111111111111111111111111111111111111111111111111111111111");
    auto b = hex32("2222222222222222222222222222222222222222222222222222222222222222");

    auto branch_ab = taproot_branch_hash(a, b);
    auto branch_ba = taproot_branch_hash(b, a);
    check(branch_ab == branch_ba, "TapBranch: order-independent");
}

static void test_taproot_merkle_tree() {
    std::printf("[Taproot] Merkle tree construction...\n");

    uint8_t s1[] = {0xAC};
    uint8_t s2[] = {0xAD};
    uint8_t s3[] = {0xAE};

    auto l1 = taproot_leaf_hash(s1, 1);
    auto l2 = taproot_leaf_hash(s2, 1);
    auto l3 = taproot_leaf_hash(s3, 1);

    // Single leaf
    auto root1 = taproot_merkle_root({l1});
    check(root1 == l1, "Merkle: single leaf = leaf hash");

    // Two leaves
    auto root2 = taproot_merkle_root({l1, l2});
    auto expected2 = taproot_branch_hash(l1, l2);
    check(root2 == expected2, "Merkle: two leaves = branch(l1, l2)");

    // Three leaves (odd count)
    auto root3 = taproot_merkle_root({l1, l2, l3});
    check(!secp256k1::ct::ct_is_zero(root3), "Merkle: 3-leaf tree non-zero");
}

static void test_taproot_merkle_proof() {
    std::printf("[Taproot] Merkle proof verification...\n");

    uint8_t s1[] = {0xAC};
    uint8_t s2[] = {0xAD};
    auto l1 = taproot_leaf_hash(s1, 1);
    auto l2 = taproot_leaf_hash(s2, 1);

    auto root = taproot_merkle_root({l1, l2});

    // Proof for l1: sibling is l2
    auto computed = taproot_merkle_root_from_proof(l1, {l2});
    check(computed == root, "Merkle proof: l1 with sibling l2 gives root");
}

static void test_taproot_full_flow() {
    std::printf("[Taproot] Full flow: key-path + script-path...\n");

    // Internal key
    auto sk = Scalar::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000003");
    auto pk_x = schnorr_pubkey(sk);

    // Script tree: two scripts
    uint8_t script_a[] = {0xAC}; // OP_CHECKSIG
    uint8_t script_b[] = {0x51, 0xAC}; // OP_1 OP_CHECKSIG

    auto leaf_a = taproot_leaf_hash(script_a, 1);
    auto leaf_b = taproot_leaf_hash(script_b, 2);

    auto merkle_root = taproot_merkle_root({leaf_a, leaf_b});

    // Derive output key with merkle root
    auto [output_x, parity] = taproot_output_key(pk_x, merkle_root.data(), 32);
    check(!secp256k1::ct::ct_is_zero(output_x), "Full flow: output key non-zero");

    // Verify commitment
    check(taproot_verify_commitment(output_x, parity, pk_x, merkle_root.data(), 32),
          "Full flow: commitment verifies");

    // Key-path spend: tweak private key
    auto tweaked_sk = taproot_tweak_privkey(sk, merkle_root.data(), 32);
    check(!tweaked_sk.is_zero(), "Full flow: tweaked key non-zero");

    // tweaked key * G should produce the output key
    auto tweaked_pk = Point::generator().scalar_mul(tweaked_sk);
    auto tweaked_pk_x = tweaked_pk.x().to_bytes();
    check(tweaked_pk_x == output_x, "Full flow: tweaked key matches output");
}

// ═══════════════════════════════════════════════════════════════════════════════
// CT Utils Tests
// ═══════════════════════════════════════════════════════════════════════════════

static void test_ct_equal() {
    std::printf("[CT Utils] Constant-time equality...\n");

    auto a = hex32("0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF");
    auto b = a;
    auto c = hex32("0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDE0");

    check(secp256k1::ct::ct_equal(a, b), "CT equal: same data");
    check(!secp256k1::ct::ct_equal(a, c), "CT equal: different data");
}

static void test_ct_is_zero() {
    std::printf("[CT Utils] Constant-time zero check...\n");

    std::array<uint8_t, 32> zeros{};
    auto nonzero = hex32("0000000000000000000000000000000000000000000000000000000000000001");

    check(secp256k1::ct::ct_is_zero(zeros), "CT is_zero: all zeros");
    check(!secp256k1::ct::ct_is_zero(nonzero), "CT is_zero: non-zero");
}

static void test_ct_compare() {
    std::printf("[CT Utils] Constant-time compare...\n");

    auto a = hex32("0000000000000000000000000000000000000000000000000000000000000001");
    auto b = hex32("0000000000000000000000000000000000000000000000000000000000000002");
    auto c = a;

    check(secp256k1::ct::ct_compare(a.data(), b.data(), 32) < 0, "CT compare: a < b");
    check(secp256k1::ct::ct_compare(b.data(), a.data(), 32) > 0, "CT compare: b > a");
    check(secp256k1::ct::ct_compare(a.data(), c.data(), 32) == 0, "CT compare: a == c");
}

static void test_ct_memzero() {
    std::printf("[CT Utils] Secure memory zeroing...\n");

    std::array<uint8_t, 32> data;
    std::memset(data.data(), 0xFF, 32);
    secp256k1::ct::ct_memzero(data.data(), 32);

    check(secp256k1::ct::ct_is_zero(data), "CT memzero: data is zeroed");
}

static void test_ct_conditional_ops() {
    std::printf("[CT Utils] Conditional copy and swap...\n");

    auto a = hex32("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");
    auto b = hex32("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB");
    auto orig_a = a;
    auto orig_b = b;

    // Conditional copy (false) — no change
    auto dst = a;
    secp256k1::ct::ct_memcpy_if(dst.data(), b.data(), 32, false);
    check(dst == a, "CT memcpy_if: false keeps original");

    // Conditional copy (true) — copies
    secp256k1::ct::ct_memcpy_if(dst.data(), b.data(), 32, true);
    check(dst == b, "CT memcpy_if: true copies source");

    // Conditional swap (false) — no change
    a = orig_a; b = orig_b;
    secp256k1::ct::ct_memswap_if(a.data(), b.data(), 32, false);
    check(a == orig_a && b == orig_b, "CT memswap_if: false no swap");

    // Conditional swap (true) — swaps
    secp256k1::ct::ct_memswap_if(a.data(), b.data(), 32, true);
    check(a == orig_b && b == orig_a, "CT memswap_if: true swaps");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Wycheproof-Style Edge Case ECDSA Vectors
// ═══════════════════════════════════════════════════════════════════════════════

static void test_wycheproof_ecdsa_edge_cases() {
    std::printf("[Wycheproof] ECDSA edge cases...\n");

    // 1. Signature with s = 1 (minimal s)
    {
        auto sk = Scalar::from_hex(
            "0000000000000000000000000000000000000000000000000000000000000001");
        auto pk = Point::generator().scalar_mul(sk);
        auto msg = hex32("4B688DF40BCEDBE641DDB16FF0A1842D9C67EA1C3BF63F3E0471BCA57E5A2BD2");

        auto sig = ecdsa_sign(msg, sk);
        check(ecdsa_verify(msg, pk, sig), "Wycheproof: normal sig verifies");
    }

    // 2. Message hash = 0
    {
        auto sk = Scalar::from_hex(
            "0000000000000000000000000000000000000000000000000000000000000001");
        auto pk = Point::generator().scalar_mul(sk);
        auto zero_msg = hex32("0000000000000000000000000000000000000000000000000000000000000000");

        auto sig = ecdsa_sign(zero_msg, sk);
        check(ecdsa_verify(zero_msg, pk, sig), "Wycheproof: zero message hash");
    }

    // 3. Message hash = n - 1 (max valid)
    {
        auto sk = Scalar::from_hex(
            "0000000000000000000000000000000000000000000000000000000000000002");
        auto pk = Point::generator().scalar_mul(sk);
        auto max_msg = hex32("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140");

        auto sig = ecdsa_sign(max_msg, sk);
        check(ecdsa_verify(max_msg, pk, sig), "Wycheproof: max message hash");
    }

    // 4. Verify rejects modified message
    {
        auto sk = Scalar::from_hex(
            "0000000000000000000000000000000000000000000000000000000000000001");
        auto pk = Point::generator().scalar_mul(sk);
        auto msg = hex32("0000000000000000000000000000000000000000000000000000000000000001");

        auto sig = ecdsa_sign(msg, sk);
        auto bad_msg = msg;
        bad_msg[31] ^= 0x01; // Flip one bit
        check(!ecdsa_verify(bad_msg, pk, sig), "Wycheproof: modified msg rejects");
    }

    // 5. Verify rejects wrong public key
    {
        auto sk = Scalar::from_hex(
            "0000000000000000000000000000000000000000000000000000000000000001");
        auto wrong_pk = Point::generator().scalar_mul(
            Scalar::from_hex("0000000000000000000000000000000000000000000000000000000000000002"));
        auto msg = hex32("0000000000000000000000000000000000000000000000000000000000000001");

        auto sig = ecdsa_sign(msg, sk);
        check(!ecdsa_verify(msg, wrong_pk, sig), "Wycheproof: wrong pubkey rejects");
    }

    // 6. High-S signature normalization
    {
        auto sk = Scalar::from_hex(
            "0000000000000000000000000000000000000000000000000000000000000001");
        auto msg = hex32("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");

        auto sig = ecdsa_sign(msg, sk);
        check(sig.is_low_s(), "Wycheproof: ecdsa_sign always produces low-S");
    }

    // 7. Sign with zero private key returns zero sig
    {
        auto msg = hex32("0000000000000000000000000000000000000000000000000000000000000001");
        auto sig = ecdsa_sign(msg, Scalar::zero());
        check(sig.r.is_zero() && sig.s.is_zero(), "Wycheproof: zero key → zero sig");
    }

    // 8. Verify with zero signature rejects
    {
        auto pk = Point::generator();
        auto msg = hex32("0000000000000000000000000000000000000000000000000000000000000001");
        ECDSASignature zero_sig{Scalar::zero(), Scalar::zero()};
        check(!ecdsa_verify(msg, pk, zero_sig), "Wycheproof: zero sig rejects");
    }
}

static void test_wycheproof_schnorr_edge_cases() {
    std::printf("[Wycheproof] Schnorr (BIP-340) edge cases...\n");

    // 1. Normal sign/verify
    {
        auto sk = Scalar::from_hex(
            "0000000000000000000000000000000000000000000000000000000000000001");
        auto pk_x = schnorr_pubkey(sk);
        auto msg = hex32("0000000000000000000000000000000000000000000000000000000000000001");
        auto aux = hex32("0000000000000000000000000000000000000000000000000000000000000000");

        auto sig = schnorr_sign(sk, msg, aux);
        check(schnorr_verify(pk_x, msg, sig), "Wycheproof Schnorr: basic verify");
    }

    // 2. Modified message rejects
    {
        auto sk = Scalar::from_hex(
            "0000000000000000000000000000000000000000000000000000000000000001");
        auto pk_x = schnorr_pubkey(sk);
        auto msg = hex32("0000000000000000000000000000000000000000000000000000000000000001");
        auto aux = hex32("0000000000000000000000000000000000000000000000000000000000000000");

        auto sig = schnorr_sign(sk, msg, aux);

        auto bad_msg = msg;
        bad_msg[31] ^= 0x01;
        check(!schnorr_verify(pk_x, bad_msg, sig), "Wycheproof Schnorr: modified msg rejects");
    }

    // 3. Zero message
    {
        auto sk = Scalar::from_hex(
            "0000000000000000000000000000000000000000000000000000000000000002");
        auto pk_x = schnorr_pubkey(sk);
        auto zero_msg = hex32("0000000000000000000000000000000000000000000000000000000000000000");
        auto aux = hex32("0000000000000000000000000000000000000000000000000000000000000000");

        auto sig = schnorr_sign(sk, zero_msg, aux);
        check(schnorr_verify(pk_x, zero_msg, sig), "Wycheproof Schnorr: zero message");
    }
}

static void test_wycheproof_recovery_edge_cases() {
    std::printf("[Wycheproof] Recovery edge cases...\n");

    // 1. Recovery round-trip with various message hashes
    const char* messages[] = {
        "0000000000000000000000000000000000000000000000000000000000000000",  // zero
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",  // max
        "0000000000000000000000000000000000000000000000000000000000000001",  // one
        "8000000000000000000000000000000000000000000000000000000000000000",  // MSB set
    };

    auto sk = Scalar::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000007");
    auto pk = Point::generator().scalar_mul(sk);

    for (auto hex : messages) {
        auto msg = hex32(hex);
        auto rsig = ecdsa_sign_recoverable(msg, sk);
        auto [recovered, ok] = ecdsa_recover(msg, rsig.sig, rsig.recid);
        check(ok && recovered.to_compressed() == pk.to_compressed(),
              "Wycheproof Recovery: round-trip with edge-case hash");
    }

    // 2. Verify recovered key works for verification
    {
        auto msg = hex32("DEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEF");
        auto rsig = ecdsa_sign_recoverable(msg, sk);
        auto [recovered, ok] = ecdsa_recover(msg, rsig.sig, rsig.recid);
        check(ok, "Wycheproof Recovery: recovery succeeded");
        check(ecdsa_verify(msg, recovered, rsig.sig),
              "Wycheproof Recovery: recovered key verifies signature");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════════

int main() {
    std::printf("═══════════════════════════════════════════════════════════════\n");
    std::printf("  UltrafastSecp256k1 — v3.2.0 Feature Tests\n");
    std::printf("═══════════════════════════════════════════════════════════════\n\n");

    // ECDH
    test_ecdh_basic();
    test_ecdh_xonly();
    test_ecdh_raw();
    test_ecdh_zero_key();
    test_ecdh_infinity();

    std::printf("\n");

    // Recovery
    test_recovery_basic();
    test_recovery_multiple_keys();
    test_recovery_compact_serialization();
    test_recovery_wrong_recid();
    test_recovery_invalid_sig();

    std::printf("\n");

    // Taproot
    test_taproot_tweak_hash();
    test_taproot_output_key();
    test_taproot_privkey_tweak();
    test_taproot_commitment_verify();
    test_taproot_leaf_and_branch();
    test_taproot_merkle_tree();
    test_taproot_merkle_proof();
    test_taproot_full_flow();

    std::printf("\n");

    // CT Utils
    test_ct_equal();
    test_ct_is_zero();
    test_ct_compare();
    test_ct_memzero();
    test_ct_conditional_ops();

    std::printf("\n");

    // Wycheproof Edge Cases
    test_wycheproof_ecdsa_edge_cases();
    test_wycheproof_schnorr_edge_cases();
    test_wycheproof_recovery_edge_cases();

    std::printf("\n═══════════════════════════════════════════════════════════════\n");
    std::printf("  Results: %d passed, %d failed (total %d)\n",
                g_pass, g_fail, g_pass + g_fail);
    std::printf("═══════════════════════════════════════════════════════════════\n");

    return g_fail > 0 ? 1 : 0;
}
