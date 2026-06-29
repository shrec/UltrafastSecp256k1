// Standalone test for the direct libbitcoin integration header
// (ufsecp/libbitcoin.hpp): signing, key operations, recovery, and utilities.
// Build: see CMakeLists.txt (test_lbtc_direct_operations).
// Returns 0 on success, 1 on any failure.
#include "ufsecp/libbitcoin.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/ct/sign.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace secp256k1;
using fast::Scalar;
using fast::Point;

namespace {
std::uint64_t g_xs = 0xDEADBEEFCAFEBABEull;
std::uint8_t nb() { g_xs ^= g_xs << 13; g_xs ^= g_xs >> 7; g_xs ^= g_xs << 17; return static_cast<std::uint8_t>(g_xs); }
int fails = 0;
void check(bool cond, const char* what) { if (!cond) { std::printf("FAIL: %s\n", what); ++fails; } }

// Generate a valid random secret key.
void rand_sk(std::uint8_t sk[32]) {
    do {
        for (int i = 0; i < 32; ++i) sk[i] = nb();
    } while (!ufsecp::lbtc::seckey_verify(sk));
}

// Generate a valid random message hash.
void rand_hash(std::uint8_t h[32]) {
    for (int i = 0; i < 32; ++i) h[i] = nb();
}
} // namespace

int main() {
    std::printf("=== libbitcoin direct operations test ===\n");
    std::uint8_t aux[32];
    std::memset(aux, 0, 32);

    // ─── ECDSA sign + verify roundtrip (CT-backed) ─────────────────────
    {
        std::uint8_t sk[32], hash[32], sig64[64];
        rand_sk(sk);
        rand_hash(hash);
        check(ufsecp::lbtc::ecdsa_sign(hash, sk, sig64), "ecdsa_sign all-valid");
        std::uint8_t pub33[33];
        check(ufsecp::lbtc::pubkey_create(sk, pub33), "pubkey_create from signing sk");
        check(ufsecp::lbtc::ecdsa_verify(pub33, hash, sig64), "ecdsa_verify after ecdsa_sign");
    }

    // ─── ECDSA sign hedged + verify ──────────────────────────────────
    {
        std::uint8_t sk[32], hash[32], aux32[32], sig64[64];
        rand_sk(sk);
        rand_hash(hash);
        for (int i = 0; i < 32; ++i) aux32[i] = nb();
        check(ufsecp::lbtc::ecdsa_sign_hedged(hash, sk, aux32, sig64), "ecdsa_sign_hedged all-valid");
        std::uint8_t pub33[33];
        check(ufsecp::lbtc::pubkey_create(sk, pub33), "pubkey_create for hedged");
        check(ufsecp::lbtc::ecdsa_verify(pub33, hash, sig64), "ecdsa_verify after ecdsa_sign_hedged");
    }

    // ─── ECDSA sign recoverable + recover ─────────────────────────────
    {
        std::uint8_t sk[32], hash[32], sig65[65], pub33[33], recovered[33];
        rand_sk(sk);
        rand_hash(hash);
        check(ufsecp::lbtc::ecdsa_sign_recoverable(hash, sk, sig65), "ecdsa_sign_recoverable all-valid");

        // Parse the 65-byte compact form properly using the API
        std::uint8_t sig64_from65[64];
        int recid = 0;
        check(ufsecp::lbtc::recoverable_from_compact(sig65, sig64_from65, recid),
              "recoverable_from_compact after sign_recoverable");
        check(recid >= 0 && recid <= 3, "recid in range");
        check(ufsecp::lbtc::ecdsa_recover(hash, sig64_from65, recid, recovered),
              "ecdsa_recover all-valid");
        check(ufsecp::lbtc::pubkey_create(sk, pub33), "pubkey_create for recoverable");
        check(std::memcmp(pub33, recovered, 33) == 0, "recovered pubkey matches created");
    }

    // ─── ECDSA recoverable_to_compact / from_compact roundtrip ────────
    {
        std::uint8_t sk[32], hash[32], sig65_a[65], sig65_b[65];
        std::uint8_t sig64[64], sig64_rt[64];
        int recid_a, recid_b;
        rand_sk(sk);
        rand_hash(hash);
        check(ufsecp::lbtc::ecdsa_sign_recoverable(hash, sk, sig65_a), "recoverable roundtrip sign");

        // Parse the 65-byte form into sig64 + recid
        check(ufsecp::lbtc::recoverable_from_compact(sig65_a, sig64, recid_a),
              "recoverable_from_compact parse");

        // Convert back to 65-byte compact
        ufsecp::lbtc::recoverable_to_compact(sig64, recid_a, sig65_b, true);
        check(std::memcmp(sig65_a, sig65_b, 65) == 0, "recoverable_to_compact roundtrip");

        // Re-parse and verify
        check(ufsecp::lbtc::recoverable_from_compact(sig65_b, sig64_rt, recid_b),
              "recoverable_from_compact re-parse");
        check(recid_a == recid_b, "recid match after roundtrip");
        check(std::memcmp(sig64, sig64_rt, 64) == 0, "sig64 match after roundtrip");
    }

    // ─── ECDSA signature normalize / serialize / parse ────────────────
    {
        std::uint8_t sk[32], hash[32], sig64[64];
        rand_sk(sk);
        rand_hash(hash);
        check(ufsecp::lbtc::ecdsa_sign(hash, sk, sig64), "normalize sign");

        // Normalize (CT signing produces low-S already → no-op)
        std::uint8_t sig64_norm[64];
        std::memcpy(sig64_norm, sig64, 64);
        bool was_normalized = ufsecp::lbtc::ecdsa_signature_normalize(sig64_norm);
        check(!was_normalized, "normalize low-S already (CT output always low-S)");

        // Serialize compact (opaque LE → big-endian)
        std::uint8_t be_compact[64];
        ufsecp::lbtc::ecdsa_signature_serialize_compact(sig64, be_compact);

        // Re-parse (big-endian → opaque LE)
        std::uint8_t parsed_opaque[64];
        check(ufsecp::lbtc::ecdsa_signature_parse_compact(be_compact, parsed_opaque),
              "parse compact strict after serialize");
        check(std::memcmp(sig64, parsed_opaque, 64) == 0,
              "signature parse(serialize(sig)) == sig roundtrip");

        // DER serialize
        std::uint8_t der[72];
        std::size_t der_len = 0;
        check(ufsecp::lbtc::ecdsa_signature_serialize_der(sig64, der, der_len), "serialize der");
        check(der_len >= 8 && der_len <= 72, "der length plausible");
    }

    // ─── pubkey_create (CT) ──────────────────────────────────────────
    {
        std::uint8_t sk[32], pub33[33];
        rand_sk(sk);
        check(ufsecp::lbtc::pubkey_create(sk, pub33), "pubkey_create");
        check(pub33[0] == 0x02 || pub33[0] == 0x03, "pubkey_create compressed prefix");
        // Invalid key must fail
        std::uint8_t bad_sk[32];
        std::memset(bad_sk, 0, 32);
        check(!ufsecp::lbtc::pubkey_create(bad_sk, pub33), "pubkey_create zero sk fails");
    }

    // ─── pubkey_parse ────────────────────────────────────────────────
    {
        std::uint8_t sk[32], pub33[33];
        rand_sk(sk);
        check(ufsecp::lbtc::pubkey_create(sk, pub33), "pubkey_parse create");
        check(ufsecp::lbtc::pubkey_parse(pub33), "pubkey_parse valid");
        // Invalid compressed key
        std::uint8_t bad33[33] = {0xFF};
        check(!ufsecp::lbtc::pubkey_parse(bad33), "pubkey_parse invalid prefix");
    }

    // ─── pubkey_negate ───────────────────────────────────────────────
    {
        std::uint8_t sk[32], pub33[33], neg33[33];
        rand_sk(sk);
        check(ufsecp::lbtc::pubkey_create(sk, pub33), "pubkey_negate create");
        std::memcpy(neg33, pub33, 33);
        check(ufsecp::lbtc::pubkey_negate(neg33), "pubkey_negate");
        // Negate again → original
        check(ufsecp::lbtc::pubkey_negate(neg33), "pubkey_negate back");
        check(std::memcmp(pub33, neg33, 33) == 0, "pubkey_negate roundtrip");
    }

    // ─── pubkey_tweak_add ────────────────────────────────────────────
    {
        std::uint8_t sk[32], pub33[33], tweak[32];
        rand_sk(sk);
        rand_sk(tweak);
        check(ufsecp::lbtc::pubkey_create(sk, pub33), "pubkey_tweak_add create");
        std::uint8_t tweaked[33];
        std::memcpy(tweaked, pub33, 33);
        check(ufsecp::lbtc::pubkey_tweak_add(tweaked, tweak), "pubkey_tweak_add");
        // Verify that a signature from sk+tweak verifies against P'
        std::uint8_t sk_tweaked[32];
        std::memcpy(sk_tweaked, sk, 32);
        check(ufsecp::lbtc::seckey_tweak_add(sk_tweaked, tweak), "seckey_tweak_add for tweak_add check");
        std::uint8_t hash[32], sig64[64];
        rand_hash(hash);
        check(ufsecp::lbtc::ecdsa_sign(hash, sk_tweaked, sig64), "ecdsa_sign with tweaked sk");
        check(ufsecp::lbtc::ecdsa_verify(tweaked, hash, sig64), "ecdsa_verify after tweak_add roundtrip");
    }

    // ─── pubkey_tweak_mul ────────────────────────────────────────────
    {
        std::uint8_t sk[32], pub33[33], tweak[32];
        rand_sk(sk);
        rand_sk(tweak);
        check(ufsecp::lbtc::pubkey_create(sk, pub33), "pubkey_tweak_mul create");
        std::uint8_t multiplied[33];
        std::memcpy(multiplied, pub33, 33);
        check(ufsecp::lbtc::pubkey_tweak_mul(multiplied, tweak), "pubkey_tweak_mul");
        // Verify sk*tweak produces same pubkey
        std::uint8_t sk_mul[32];
        std::memcpy(sk_mul, sk, 32);
        check(ufsecp::lbtc::seckey_tweak_mul(sk_mul, tweak), "seckey_tweak_mul for pubkey_tweak_mul check");
        std::uint8_t expected_pub[33];
        check(ufsecp::lbtc::pubkey_create(sk_mul, expected_pub), "pubkey_create after seckey_tweak_mul");
        check(std::memcmp(multiplied, expected_pub, 33) == 0,
              "pubkey_tweak_mul matches seckey_tweak_mul*G");
    }

    // ─── pubkey_combine ──────────────────────────────────────────────
    {
        constexpr int N = 5;
        std::uint8_t sks[N][32], pubs[N][33];
        for (int i = 0; i < N; ++i) {
            rand_sk(sks[i]);
            check(ufsecp::lbtc::pubkey_create(sks[i], pubs[i]), "pubkey_combine create keys");
        }
        // Combine via pubkey_combine (pointer array)
        std::uint8_t combined[33];
        const std::uint8_t* ptrs[N];
        for (int i = 0; i < N; ++i) ptrs[i] = pubs[i];
        check(ufsecp::lbtc::pubkey_combine(ptrs, N, combined), "pubkey_combine");
        // Verify: sum of secret keys → combined pubkey
        Scalar sum_sk = Scalar::zero();
        for (int i = 0; i < N; ++i) {
            Scalar ski;
            check(Scalar::parse_bytes_strict_nonzero(sks[i], ski), "parse sk for combine check");
            sum_sk = sum_sk + ski;
        }
        auto sum_bytes = sum_sk.to_bytes();
        std::uint8_t sum_sk_bytes[32];
        std::memcpy(sum_sk_bytes, sum_bytes.data(), 32);
        std::uint8_t expected_pub[33];
        check(ufsecp::lbtc::pubkey_create(sum_sk_bytes, expected_pub), "pubkey_create for combined sum");
        check(std::memcmp(combined, expected_pub, 33) == 0,
              "pubkey_combine matches sum(sk)*G");
    }

    // ─── seckey_verify ───────────────────────────────────────────────
    {
        std::uint8_t sk[32];
        rand_sk(sk);
        check(ufsecp::lbtc::seckey_verify(sk), "seckey_verify valid");
        std::uint8_t zero_sk[32] = {0};
        check(!ufsecp::lbtc::seckey_verify(zero_sk), "seckey_verify zero rejected");
        std::uint8_t full_sk[32];
        std::memset(full_sk, 0xFF, 32);
        check(!ufsecp::lbtc::seckey_verify(full_sk), "seckey_verify >= n rejected");
    }

    // ─── seckey_negate ───────────────────────────────────────────────
    {
        std::uint8_t sk[32], orig[32];
        rand_sk(sk);
        std::memcpy(orig, sk, 32);
        check(ufsecp::lbtc::seckey_negate(sk), "seckey_negate");
        check(ufsecp::lbtc::seckey_negate(sk), "seckey_negate back");
        check(std::memcmp(orig, sk, 32) == 0, "seckey_negate roundtrip");
    }

    // ─── seckey_tweak_add ────────────────────────────────────────────
    {
        std::uint8_t sk[32], tweak[32], orig_pub[33];
        rand_sk(sk);
        rand_sk(tweak);
        check(ufsecp::lbtc::pubkey_create(sk, orig_pub), "seckey_tweak_add orig pub");
        check(ufsecp::lbtc::seckey_tweak_add(sk, tweak), "seckey_tweak_add");
        std::uint8_t new_pub[33];
        check(ufsecp::lbtc::pubkey_create(sk, new_pub), "pubkey_create after seckey_tweak_add");
        std::uint8_t expected[33];
        std::memcpy(expected, orig_pub, 33);
        check(ufsecp::lbtc::pubkey_tweak_add(expected, tweak), "pubkey_tweak_add for comparison");
        check(std::memcmp(new_pub, expected, 33) == 0,
              "seckey_tweak_add matches pubkey_tweak_add");
    }

    // ─── seckey_tweak_mul ────────────────────────────────────────────
    {
        std::uint8_t sk[32], tweak[32], orig_pub[33];
        rand_sk(sk);
        rand_sk(tweak);
        check(ufsecp::lbtc::pubkey_create(sk, orig_pub), "seckey_tweak_mul orig pub");
        check(ufsecp::lbtc::seckey_tweak_mul(sk, tweak), "seckey_tweak_mul");
        std::uint8_t new_pub[33];
        check(ufsecp::lbtc::pubkey_create(sk, new_pub), "pubkey_create after seckey_tweak_mul");
        std::uint8_t expected[33];
        std::memcpy(expected, orig_pub, 33);
        check(ufsecp::lbtc::pubkey_tweak_mul(expected, tweak), "pubkey_tweak_mul for comparison");
        check(std::memcmp(new_pub, expected, 33) == 0,
              "seckey_tweak_mul matches pubkey_tweak_mul");
    }

    // ─── Schnorr keypair_create + sign + verify (CT-backed) ──────────
    {
        std::uint8_t sk[32], xonly[32], msg[32], sig64[64];
        rand_sk(sk);
        rand_hash(msg);
        check(ufsecp::lbtc::schnorr_keypair_create(sk, xonly), "schnorr_keypair_create");
        std::uint8_t aux32[32];
        std::memset(aux32, 0, 32);
        check(ufsecp::lbtc::schnorr_sign(xonly, sk, msg, aux32, sig64), "schnorr_sign");
        check(ufsecp::lbtc::schnorr_verify(xonly, msg, sig64), "schnorr_verify after schnorr_sign");
        // Tampered sig must fail
        sig64[0] ^= 1;
        check(!ufsecp::lbtc::schnorr_verify(xonly, msg, sig64), "schnorr_verify tampered sig fails");
    }

    // ─── Schnorr sign rejects mismatched pubkey_xonly ───────────────
    {
        std::uint8_t sk[32], xonly[32], wrong_xonly[32], msg[32], sig64[64];
        rand_sk(sk);
        rand_hash(msg);
        check(ufsecp::lbtc::schnorr_keypair_create(sk, xonly), "xonly mismatch create");
        // Generate a different keypair and use its xonly
        std::uint8_t sk2[32];
        rand_sk(sk2);
        check(ufsecp::lbtc::schnorr_keypair_create(sk2, wrong_xonly), "xonly mismatch wrong create");
        std::uint8_t aux32[32];
        std::memset(aux32, 0, 32);
        check(!ufsecp::lbtc::schnorr_sign(wrong_xonly, sk, msg, aux32, sig64),
              "schnorr_sign rejects mismatched pubkey_xonly");
        // Verify sig64 is zeroed on failure
        for (auto b : sig64) check(b == 0, "schnorr_sign mismatched xonly zeroes output");
    }

    // ─── Schnorr xonly pubkey parse ──────────────────────────────────
    {
        std::uint8_t sk[32], xonly[32];
        rand_sk(sk);
        check(ufsecp::lbtc::schnorr_keypair_create(sk, xonly), "xonly parse keypair create");
        check(ufsecp::lbtc::schnorr_xonly_pubkey_parse(xonly), "schnorr_xonly_pubkey_parse valid");
        std::uint8_t bad_xonly[32];
        std::memset(bad_xonly, 0xFF, 32);
        check(!ufsecp::lbtc::schnorr_xonly_pubkey_parse(bad_xonly),
              "schnorr_xonly_pubkey_parse invalid");
    }

    // ─── Taproot tweak_add_check ─────────────────────────────────────
    {
        std::uint8_t sk[32], internal_xonly[32];
        rand_sk(sk);
        check(ufsecp::lbtc::schnorr_keypair_create(sk, internal_xonly), "taproot internal key");
        auto tweak_hash = secp256k1::tagged_hash("TapTweak", internal_xonly, 32);
        Scalar tweak;
        check(Scalar::parse_bytes_strict(tweak_hash.data(), tweak), "taproot tweak parse");
        // Build internal compressed with even Y (BIP-341 convention)
        std::uint8_t internal_compressed[33] = {0x02};
        std::memcpy(internal_compressed + 1, internal_xonly, 32);
        std::uint8_t tweak32[32];
        std::memcpy(tweak32, tweak_hash.data(), 32);
        check(ufsecp::lbtc::pubkey_tweak_add(internal_compressed, tweak32),
              "taproot output key via pubkey_tweak_add");
        std::uint8_t output_xonly[32];
        std::memcpy(output_xonly, internal_compressed + 1, 32);
        int parity = (internal_compressed[0] == 0x03) ? 1 : 0;
        check(ufsecp::lbtc::taproot_tweak_add_check(output_xonly, parity, internal_xonly, nullptr, 0),
              "taproot_tweak_add_check key-path only");
    }

    // ─── Fail-closed: invalid secret key to signing ──────────────────
    {
        std::uint8_t zero_sk[32] = {0}, hash[32], sig64[64];
        rand_hash(hash);
        check(!ufsecp::lbtc::ecdsa_sign(hash, zero_sk, sig64), "ecdsa_sign zero sk fails");
        for (auto b : sig64) check(b == 0, "ecdsa_sign zero sk sig64 zeroed");
    }

    // ─── Fail-closed: invalid pubkey to verify ───────────────────────
    {
        std::uint8_t bad_pub[33] = {0xFF}, hash[32], sig64[64];
        rand_hash(hash);
        std::memset(sig64, 0x42, 64);
        check(!ufsecp::lbtc::ecdsa_verify(bad_pub, hash, sig64), "ecdsa_verify bad pubkey fails");
    }

    // ─── Fail-closed: invalid recovery ───────────────────────────────
    {
        std::uint8_t hash[32], sig64[64], pub33[33];
        rand_hash(hash);
        std::memset(sig64, 0, 64);
        check(!ufsecp::lbtc::ecdsa_recover(hash, sig64, 0, pub33), "ecdsa_recover zero sig fails");
        check(!ufsecp::lbtc::ecdsa_recover(hash, sig64, 5, pub33), "ecdsa_recover invalid recid fails");
    }

    // ─── Fail-closed: signature parse rejects non-canonical ──────────
    {
        std::uint8_t bad_be[64];
        std::memset(bad_be, 0xFF, 64);  // both r and s >= n (big-endian)
        std::uint8_t out[64];
        check(!ufsecp::lbtc::ecdsa_signature_parse_compact(bad_be, out),
              "parse compact rejects >=n");
    }

    std::printf("=== %s ===\n", fails == 0 ? "ALL PASS" : "SOME FAILED");
    return fails == 0 ? 0 : 1;
}
