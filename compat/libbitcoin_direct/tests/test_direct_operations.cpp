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

// Independent Bitcoin HASH256 (double-SHA256) reference.
void ref_hash256(const std::uint8_t* in, std::size_t len, std::uint8_t out[32]) {
    const auto d = secp256k1::SHA256::hash256(in, len);
    std::memcpy(out, d.data(), 32);
}

// Fake GPU hook for merkle_pair_hash_batch: writes a fixed sentinel pattern
// and reports "handled" (0). Used only to prove the wrapper actually
// consults the installed hook instead of silently falling through to CPU.
int fake_merkle_pair_hook(const std::uint8_t*, const std::uint8_t*, std::size_t count,
                          std::uint8_t* out32) {
    std::memset(out32, 0xDD, count * 32);
    return 0;
}

// Fake GPU hook for merkle_pair_hash_batch that always declines (-1),
// forcing the CPU fallback loop to run regardless of what a real production
// hook (if any is installed by a direct-GPU build) would otherwise do.
int decline_merkle_pair_hook(const std::uint8_t*, const std::uint8_t*, std::size_t,
                             std::uint8_t*) {
    return -1;
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


    // ─── pubkey_create_uncompressed (CT) ──────────────────────────────
    {
        std::uint8_t sk[32], pub65[65];
        rand_sk(sk);
        check(ufsecp::lbtc::pubkey_create_uncompressed(sk, pub65), "pubkey_create_uncompressed valid");
        check(pub65[0] == 0x04, "pubkey_create_uncompressed header 0x04");
        // Verify against compressed version: compress(uncompressed) == pubkey_create(compressed)
        std::uint8_t pub33[33], pub33_from65[33];
        check(ufsecp::lbtc::pubkey_create(sk, pub33), "pubkey_create for cross-check");
        check(ufsecp::lbtc::pubkey_compress(pub65, pub33_from65), "pubkey_compress cross-check");
        check(std::memcmp(pub33, pub33_from65, 33) == 0, "uncompressed->compressed matches direct compressed");
        // Invalid key must fail
        std::uint8_t zero_sk[32] = {0};
        check(!ufsecp::lbtc::pubkey_create_uncompressed(zero_sk, pub65), "pubkey_create_uncompressed zero sk fails");
        // Verify output zeroed
        for (int i = 0; i < 65; ++i) check(pub65[i] == 0, "pubkey_create_uncompressed zero sk zeroes output");
    }

    // ─── pubkey_parse_uncompressed ───────────────────────────────────
    {
        std::uint8_t sk[32], pub65[65];
        rand_sk(sk);
        check(ufsecp::lbtc::pubkey_create_uncompressed(sk, pub65), "pubkey_parse_uncompressed create");
        check(ufsecp::lbtc::pubkey_parse_uncompressed(pub65), "pubkey_parse_uncompressed valid");
        // Invalid header
        std::uint8_t bad65[65];
        std::memset(bad65, 0xFF, 65);
        bad65[0] = 0x05;
        check(!ufsecp::lbtc::pubkey_parse_uncompressed(bad65), "pubkey_parse_uncompressed invalid header");
        // Off-curve point (x = all-0x03, which is off-curve)
        std::uint8_t off65[65];
        off65[0] = 0x04;
        std::memset(off65 + 1, 0x03, 32);  // x = 0x0303...
        std::memset(off65 + 33, 0x00, 32); // y = 0 (not on curve)
        check(!ufsecp::lbtc::pubkey_parse_uncompressed(off65), "pubkey_parse_uncompressed off-curve rejected");
    }

    // ─── pubkey_compress / pubkey_decompress roundtrip ───────────────
    {
        std::uint8_t sk[32], pub33[33], pub65[65], pub33_rt[33];
        rand_sk(sk);
        check(ufsecp::lbtc::pubkey_create(sk, pub33), "compress-rt create compressed");
        check(ufsecp::lbtc::pubkey_decompress(pub33, pub65), "pubkey_decompress");
        check(pub65[0] == 0x04, "pubkey_decompress header check");
        check(ufsecp::lbtc::pubkey_compress(pub65, pub33_rt), "pubkey_compress back");
        check(std::memcmp(pub33, pub33_rt, 33) == 0, "compress(decompress(pub33)) == pub33 roundtrip");
        // Invalid inputs
        std::uint8_t bad33[33] = {0xFF};
        check(!ufsecp::lbtc::pubkey_decompress(bad33, pub65), "pubkey_decompress invalid fails");
        std::uint8_t bad65[65] = {0xFF};
        check(!ufsecp::lbtc::pubkey_compress(bad65, pub33_rt), "pubkey_compress invalid fails");
    }

    // ─── ECDSA sign recoverable hedged + recover (CT) ────────────────
    {
        std::uint8_t sk[32], hash[32], aux32[32], sig65[65], pub33[33], recovered[33];
        rand_sk(sk);
        rand_hash(hash);
        for (int i = 0; i < 32; ++i) aux32[i] = nb();
        check(ufsecp::lbtc::ecdsa_sign_hedged_recoverable(hash, sk, aux32, sig65),
              "ecdsa_sign_hedged_recoverable");
        // Parse back
        std::uint8_t sig64[64];
        int recid = 0;
        check(ufsecp::lbtc::recoverable_from_compact(sig65, sig64, recid),
              "hedged recoverable_from_compact");
        check(recid >= 0 && recid <= 3, "hedged recid in range");
        check(ufsecp::lbtc::ecdsa_recover(hash, sig64, recid, recovered),
              "hedged ecdsa_recover");
        check(ufsecp::lbtc::pubkey_create(sk, pub33), "hedged pubkey_create");
        check(std::memcmp(pub33, recovered, 33) == 0, "hedged recovered pubkey matches created");
        // Verify the signature
        check(ufsecp::lbtc::ecdsa_verify(pub33, hash, sig64), "hedged ecdsa_verify after recover");
        // Invalid sk must fail
        std::uint8_t zero_sk[32] = {0};
        check(!ufsecp::lbtc::ecdsa_sign_hedged_recoverable(hash, zero_sk, aux32, sig65),
              "ecdsa_sign_hedged_recoverable zero sk fails");
        for (int i = 0; i < 65; ++i) check(sig65[i] == 0, "hedged recoverable zero sk zeroes output");
        // Hedged signing uses aux randomness for defense-in-depth;
        // signatures may differ across calls even with identical inputs
        // (the aux seeds internal HMAC-DRBG state). Verify each sig independently.
        std::uint8_t sig65_b[65];
        check(ufsecp::lbtc::ecdsa_sign_hedged_recoverable(hash, sk, aux32, sig65_b),
              "hedged recoverable second call");
        // Recover and verify the second signature independently
        std::uint8_t sig64_b[64], recovered_b[33];
        int recid_b = 0;
        check(ufsecp::lbtc::recoverable_from_compact(sig65_b, sig64_b, recid_b),
              "hedged second recoverable_from_compact");
        check(ufsecp::lbtc::ecdsa_recover(hash, sig64_b, recid_b, recovered_b),
              "hedged second ecdsa_recover");
        check(std::memcmp(pub33, recovered_b, 33) == 0,
              "hedged second recovered pubkey matches created");
    }

    // ─── pubkey_combine negative test ────────────────────────────────
    {
        std::uint8_t out[33];
        check(!ufsecp::lbtc::pubkey_combine((const std::uint8_t* const*)nullptr, 0, out),
              "pubkey_combine zero count fails");
        // Single invalid key
        std::uint8_t bad33[33] = {0xFF};
        const std::uint8_t* ptrs[1] = {bad33};
        check(!ufsecp::lbtc::pubkey_combine(ptrs, 1, out), "pubkey_combine invalid key fails");
    }

    // ─── taproot_tweak_add_check with merkle root ───────────────────
    {
        std::uint8_t sk[32], internal_xonly[32];
        rand_sk(sk);
        check(ufsecp::lbtc::schnorr_keypair_create(sk, internal_xonly), "taproot mr internal key");
        // Build a fake merkle root (32 bytes)
        std::uint8_t merkle_root[32];
        rand_hash(merkle_root);
        // Compute expected output via tagged hash
        // Concatenate internal_xonly || merkle_root for BIP-341 tagged hash
        std::uint8_t tap_tweak_msg[64];
        std::memcpy(tap_tweak_msg, internal_xonly, 32);
        std::memcpy(tap_tweak_msg + 32, merkle_root, 32);
        auto tweak_hash = secp256k1::tagged_hash("TapTweak", tap_tweak_msg, 64);
        Scalar tweak;
        check(Scalar::parse_bytes_strict(tweak_hash.data(), tweak), "taproot mr tweak parse");
        std::uint8_t internal_compressed[33] = {0x02};
        std::memcpy(internal_compressed + 1, internal_xonly, 32);
        std::uint8_t tweak32[32];
        std::memcpy(tweak32, tweak_hash.data(), 32);
        check(ufsecp::lbtc::pubkey_tweak_add(internal_compressed, tweak32),
              "taproot mr output key");
        std::uint8_t output_xonly[32];
        std::memcpy(output_xonly, internal_compressed + 1, 32);
        int parity = (internal_compressed[0] == 0x03) ? 1 : 0;
        check(ufsecp::lbtc::taproot_tweak_add_check(output_xonly, parity, internal_xonly,
              merkle_root, 32), "taproot_tweak_add_check with merkle root");
    }

    // ─── Fail-closed: wrong-size pubkey to verify ───────────────────
    {
        std::uint8_t sk[32], hash[32], sig64[64];
        rand_sk(sk);
        rand_hash(hash);
        check(ufsecp::lbtc::ecdsa_sign(hash, sk, sig64), "fail-closed sign");
        // Use an uncompressed pubkey (65 bytes) where compressed (33) expected
        std::uint8_t pub65[65];
        check(ufsecp::lbtc::pubkey_create_uncompressed(sk, pub65), "fail-closed uncompressed create");
        // ecdsa_verify only accepts 33-byte compressed, but we pass 65 bytes in a 33-param
        // This would be a type error at the caller level; verify with zeroed key instead
        std::uint8_t zero_pub[33] = {0};
        check(!ufsecp::lbtc::ecdsa_verify(zero_pub, hash, sig64), "ecdsa_verify zero pubkey fails");
    }

    // ─── txid_hash_batch / wtxid_hash_batch (aliases over hash256_var_batch) ─
    {
        constexpr std::size_t BN = 300;
        const std::size_t STRIDE = 250;
        std::vector<std::uint8_t> txs(BN * STRIDE);
        for (auto& b : txs) b = nb();
        std::vector<std::uint32_t> lens(BN);
        for (std::size_t i = 0; i < BN; ++i) lens[i] = (std::uint32_t)(10 + (i % (STRIDE - 10)));

        std::vector<std::uint8_t> out_txid(BN * 32, 0);
        check(ufsecp::lbtc::txid_hash_batch(txs.data(), lens.data(), STRIDE, BN, out_txid.data()),
              "txid_hash_batch computes true");
        std::vector<std::uint8_t> out_wtxid(BN * 32, 0);
        check(ufsecp::lbtc::wtxid_hash_batch(txs.data(), lens.data(), STRIDE, BN, out_wtxid.data()),
              "wtxid_hash_batch computes true");
        // Both are thin aliases of hash256_var_batch: byte-identical output
        // for the same input, computed through the exact same call path.
        std::vector<std::uint8_t> out_ref(BN * 32, 0);
        check(ufsecp::lbtc::hash256_var_batch(txs.data(), lens.data(), STRIDE, BN, out_ref.data()),
              "hash256_var_batch reference computes true");
        check(std::memcmp(out_txid.data(), out_ref.data(), BN * 32) == 0,
              "txid_hash_batch byte-identical to hash256_var_batch");
        check(std::memcmp(out_wtxid.data(), out_ref.data(), BN * 32) == 0,
              "wtxid_hash_batch byte-identical to hash256_var_batch");
        // Independent double-SHA256 (HASH256) oracle cross-check.
        int mism_txid = 0, mism_wtxid = 0;
        for (std::size_t i = 0; i < BN; ++i) {
            std::uint8_t ref[32];
            ref_hash256(txs.data() + i * STRIDE, lens[i], ref);
            if (std::memcmp(out_txid.data() + i * 32, ref, 32) != 0) ++mism_txid;
            if (std::memcmp(out_wtxid.data() + i * 32, ref, 32) != 0) ++mism_wtxid;
        }
        check(mism_txid == 0, "txid_hash_batch bit-exact vs independent double-SHA256 oracle");
        check(mism_wtxid == 0, "wtxid_hash_batch bit-exact vs independent double-SHA256 oracle");
    }

    // ─── merkle_pair_hash_batch (SoA double-SHA256 over left32||right32 pairs) ─
    {
        constexpr std::size_t BN = 300;

        // (a) count==0 -> true, out32 completely untouched.
        {
            std::uint8_t left32[32], right32[32], out32[32];
            std::memset(left32, 0x11, 32);
            std::memset(right32, 0x22, 32);
            std::memset(out32, 0xAB, 32);
            check(ufsecp::lbtc::merkle_pair_hash_batch(left32, right32, 0, out32),
                  "merkle_pair count==0 vacuous true");
            bool untouched = true;
            for (int i = 0; i < 32; ++i) if (out32[i] != 0xAB) { untouched = false; break; }
            check(untouched, "merkle_pair count==0 leaves out untouched");
        }

        std::vector<std::uint8_t> left(BN * 32), right(BN * 32);
        for (auto& b : left) b = nb();
        for (auto& b : right) b = nb();

        // (b) null left32/right32/out32, each individually -> false, out untouched.
        {
            std::vector<std::uint8_t> out(BN * 32, 0xCD);
            check(!ufsecp::lbtc::merkle_pair_hash_batch(nullptr, right.data(), BN, out.data()),
                  "merkle_pair null left fails closed");
            check(out[0] == 0xCD, "merkle_pair null left leaves out untouched");
            check(!ufsecp::lbtc::merkle_pair_hash_batch(left.data(), nullptr, BN, out.data()),
                  "merkle_pair null right fails closed");
            check(out[0] == 0xCD, "merkle_pair null right leaves out untouched");
            check(!ufsecp::lbtc::merkle_pair_hash_batch(left.data(), right.data(), BN, nullptr),
                  "merkle_pair null out fails closed");
        }

        // (c) overflow rejection: count*32 overflows size_t.
        {
            const std::size_t ovf32 = (SIZE_MAX / 32) + 1;
            std::vector<std::uint8_t> out(BN * 32, 0xCD);
            check(!ufsecp::lbtc::merkle_pair_hash_batch(left.data(), right.data(), ovf32, out.data()),
                  "merkle_pair huge count overflow rejected");
            check(out[0] == 0xCD, "merkle_pair overflow leaves out untouched");
        }

        // Startup capture (non-destructive): whatever is installed right now
        // is either nullptr (CPU-only build) or the real production hook
        // self-installed at process start (direct-GPU profile). Swap-and-
        // restore reads it without disturbing whatever is genuinely live.
        const auto prod_merkle_hook = ufsecp::lbtc::gpu_hook::install_lbtc_merkle_pair_hook(nullptr);
        ufsecp::lbtc::gpu_hook::install_lbtc_merkle_pair_hook(prod_merkle_hook);
        check(prod_merkle_hook != fake_merkle_pair_hook,
              "merkle_pair production hook (if any) is not this test's fake");

        // (d) CPU fallback KAT: force decline (-1) so this exercises the CPU
        // loop regardless of what a real production hook would otherwise do,
        // then check against an independently computed
        // double-SHA256(left||right) oracle, including an all-zeros pair and
        // a couple of arbitrary non-trivial byte patterns.
        {
            const auto prev = ufsecp::lbtc::gpu_hook::install_lbtc_merkle_pair_hook(decline_merkle_pair_hook);

            constexpr std::size_t KATN = 3;
            std::uint8_t kl[KATN * 32], kr[KATN * 32], kout[KATN * 32];
            std::memset(kl + 0 * 32, 0x00, 32);                                          // all-zeros pair
            std::memset(kr + 0 * 32, 0x00, 32);
            for (int i = 0; i < 32; ++i) kl[1 * 32 + i] = static_cast<std::uint8_t>(i);          // 0x00..0x1F
            for (int i = 0; i < 32; ++i) kr[1 * 32 + i] = static_cast<std::uint8_t>(0xFF - i);    // 0xFF..0xE0
            for (int i = 0; i < 32; ++i) kl[2 * 32 + i] = nb();                                  // arbitrary
            for (int i = 0; i < 32; ++i) kr[2 * 32 + i] = nb();                                  // arbitrary

            check(ufsecp::lbtc::merkle_pair_hash_batch(kl, kr, KATN, kout),
                  "merkle_pair CPU-fallback KAT computes true (hook forced decline)");
            int mism = 0;
            for (std::size_t i = 0; i < KATN; ++i) {
                std::uint8_t combined[64], ref[32];
                std::memcpy(combined, kl + i * 32, 32);
                std::memcpy(combined + 32, kr + i * 32, 32);
                ref_hash256(combined, 64, ref);
                if (std::memcmp(kout + i * 32, ref, 32) != 0) ++mism;
            }
            check(mism == 0, "merkle_pair CPU fallback bit-exact vs independent double-SHA256 oracle");

            ufsecp::lbtc::gpu_hook::install_lbtc_merkle_pair_hook(prev);
        }

        // (e) hook-decline fallback over the larger BN batch: install a hook
        // that always returns -1; verify the CPU path still produces the
        // correct real hash (not just "doesn't crash").
        {
            const auto prev = ufsecp::lbtc::gpu_hook::install_lbtc_merkle_pair_hook(decline_merkle_pair_hook);
            std::vector<std::uint8_t> out(BN * 32, 0);
            check(ufsecp::lbtc::merkle_pair_hash_batch(left.data(), right.data(), BN, out.data()),
                  "merkle_pair hook-decline (-1) still computes true");
            int mism = 0;
            for (std::size_t i = 0; i < BN; ++i) {
                std::uint8_t combined[64], ref[32];
                std::memcpy(combined, left.data() + i * 32, 32);
                std::memcpy(combined + 32, right.data() + i * 32, 32);
                ref_hash256(combined, 64, ref);
                if (std::memcmp(out.data() + i * 32, ref, 32) != 0) ++mism;
            }
            check(mism == 0, "merkle_pair hook-decline path bit-exact vs independent oracle");
            ufsecp::lbtc::gpu_hook::install_lbtc_merkle_pair_hook(prev);
        }

        // (f) hook-success path: fake hook writes a recognizable sentinel and
        // returns 0 (handled). Verify out32 == sentinel exactly, proving the
        // hook path (not a silent CPU override) is what ran.
        {
            check(ufsecp::lbtc::gpu_hook::install_lbtc_merkle_pair_hook(fake_merkle_pair_hook) == prod_merkle_hook,
                  "merkle_pair pre-existing hook (production or none) matches startup capture");
            std::vector<std::uint8_t> out(BN * 32, 0);
            check(ufsecp::lbtc::merkle_pair_hash_batch(left.data(), right.data(), BN, out.data()),
                  "merkle_pair with fake hook computes true");
            bool all_sentinel = true;
            for (auto b : out) if (b != 0xDD) { all_sentinel = false; break; }
            check(all_sentinel, "merkle_pair_hash_batch used the installed hook's output");
            check(ufsecp::lbtc::gpu_hook::install_lbtc_merkle_pair_hook(prod_merkle_hook) == fake_merkle_pair_hook,
                  "merkle_pair fake hook uninstalled, previous fn returned; production hook restored");
        }

        // (g) left/right SoA byte order: for a pair where left != right,
        // verify swap(left,right) changes the output (not commutative), and
        // that the un-swapped call equals doubleSHA256(left||right)
        // specifically -- not right||left.
        {
            const auto prev = ufsecp::lbtc::gpu_hook::install_lbtc_merkle_pair_hook(decline_merkle_pair_hook);
            std::uint8_t l[32], r[32], out_lr[32], out_rl[32];
            for (int i = 0; i < 32; ++i) { l[i] = static_cast<std::uint8_t>(i); r[i] = static_cast<std::uint8_t>(200 + i); }
            check(std::memcmp(l, r, 32) != 0, "merkle_pair byte-order fixture left != right");
            check(ufsecp::lbtc::merkle_pair_hash_batch(l, r, 1, out_lr),
                  "merkle_pair byte-order left||right computes true");
            check(ufsecp::lbtc::merkle_pair_hash_batch(r, l, 1, out_rl),
                  "merkle_pair byte-order right||left (swapped) computes true");
            check(std::memcmp(out_lr, out_rl, 32) != 0,
                  "merkle_pair swap(left,right) changes output (not commutative)");
            std::uint8_t combined[64], ref_lr[32];
            std::memcpy(combined, l, 32);
            std::memcpy(combined + 32, r, 32);
            ref_hash256(combined, 64, ref_lr);
            check(std::memcmp(out_lr, ref_lr, 32) == 0,
                  "merkle_pair un-swapped output == doubleSHA256(left||right), not right||left");
            ufsecp::lbtc::gpu_hook::install_lbtc_merkle_pair_hook(prev);
        }
    }



    // ─── merkle_level_reduce_batch delegates to merkle_pair_hash_batch ─────
    {
        // merkle_level_reduce_batch is a semantic alias — it must produce
        // byte-identical output to merkle_pair_hash_batch for the same inputs.
        constexpr std::size_t N = 100;
        std::vector<std::uint8_t> left(N * 32), right(N * 32);
        for (auto& b : left) b = nb();
        for (auto& b : right) b = nb();
        std::vector<std::uint8_t> out1(N * 32), out2(N * 32);
        check(ufsecp::lbtc::merkle_pair_hash_batch(left.data(), right.data(), N, out1.data()),
              "merkle_pair_hash_batch baseline");
        check(ufsecp::lbtc::merkle_level_reduce_batch(left.data(), right.data(), N, out2.data()),
              "merkle_level_reduce_batch alias");
        check(std::memcmp(out1.data(), out2.data(), N * 32) == 0,
              "merkle_level_reduce_batch byte-identical to merkle_pair_hash_batch");
    }

    // ─── merkle_root_from_leaves: 0 leaves reject ──────────────────────────
    {
        std::uint8_t scratch[64], root[32];
        std::memset(root, 0xAB, 32);
        check(!ufsecp::lbtc::merkle_root_from_leaves(nullptr, 0, scratch, 64, root),
              "merkle_root_from_leaves count==0 rejects");
        // out_root32 must be zeroed on failure
        bool zeroed = true;
        for (int i = 0; i < 32; ++i) if (root[i] != 0) { zeroed = false; break; }
        check(zeroed, "merkle_root_from_leaves count==0 zeroes output");
    }

    // ─── merkle_root_from_leaves: 1 leaf copy ─────────────────────────────
    {
        std::uint8_t leaf[32], scratch[64], root[32];
        for (int i = 0; i < 32; ++i) leaf[i] = nb();
        check(ufsecp::lbtc::merkle_root_from_leaves(leaf, 1, scratch, 64, root),
              "merkle_root_from_leaves 1 leaf returns true");
        check(std::memcmp(root, leaf, 32) == 0,
              "merkle_root_from_leaves 1 leaf copies to root");
    }

    // ─── merkle_root_from_leaves: 2 leaves KAT vs HASH256(left||right) ────
    {
        std::uint8_t l[32], r[32], scratch[128], root[32];
        for (int i = 0; i < 32; ++i) { l[i] = nb(); r[i] = nb(); }
        // Build input: [l, r]
        std::uint8_t leaves[64];
        std::memcpy(leaves, l, 32);
        std::memcpy(leaves + 32, r, 32);
        check(ufsecp::lbtc::merkle_root_from_leaves(leaves, 2, scratch, 128, root),
              "merkle_root_from_leaves 2 leaves computes true");
        // Oracle: HASH256(l || r)
        std::uint8_t combined[64], oracle[32];
        std::memcpy(combined, l, 32);
        std::memcpy(combined + 32, r, 32);
        ref_hash256(combined, 64, oracle);
        check(std::memcmp(root, oracle, 32) == 0,
              "merkle_root_from_leaves 2 leaves matches HASH256(left||right) oracle");
    }

    // ─── merkle_root_from_leaves: 3 leaves (odd) duplicate-last semantics ──
    {
        // Tree: L0 L1 L2
        // Level 0: pair(L0,L1) -> P01, pair(L2,L2) -> P22  (duplicate last)
        // Level 1: pair(P01,P22) -> root
        std::uint8_t leaves[3 * 32], scratch[192], root[32];
        for (int i = 0; i < 3 * 32; ++i) leaves[i] = nb();
        check(ufsecp::lbtc::merkle_root_from_leaves(leaves, 3, scratch, 192, root),
              "merkle_root_from_leaves 3 leaves computes true");

        // Oracle: HASH256(HASH256(L0||L1) || HASH256(L2||L2))
        std::uint8_t l0l1[64], l2l2[64];
        std::memcpy(l0l1, leaves, 32);
        std::memcpy(l0l1 + 32, leaves + 32, 32);
        std::memcpy(l2l2, leaves + 64, 32);
        std::memcpy(l2l2 + 32, leaves + 64, 32);  // duplicate last
        std::uint8_t p01[32], p22[32];
        ref_hash256(l0l1, 64, p01);
        ref_hash256(l2l2, 64, p22);
        std::uint8_t combined[64], oracle[32];
        std::memcpy(combined, p01, 32);
        std::memcpy(combined + 32, p22, 32);
        ref_hash256(combined, 64, oracle);
        check(std::memcmp(root, oracle, 32) == 0,
              "merkle_root_from_leaves 3 leaves matches independent oracle (duplicate-last)");
    }

    // ─── merkle_root_from_leaves: multi-level (7 leaves) ──────────────────
    {
        // 7 leaves -> 4 -> 2 -> 1 (root)
        std::uint8_t leaves[7 * 32], scratch[7 * 64], root[32];
        for (int i = 0; i < 7 * 32; ++i) leaves[i] = nb();
        check(ufsecp::lbtc::merkle_root_from_leaves(leaves, 7, scratch, sizeof(scratch), root),
              "merkle_root_from_leaves 7 leaves computes true");

        // Reconstruct oracle manually
        auto h2 = [](const uint8_t* a, const uint8_t* b, uint8_t out[32]) {
            uint8_t c[64];
            std::memcpy(c, a, 32); std::memcpy(c+32, b, 32);
            ref_hash256(c, 64, out);
        };
        uint8_t lvl1[4*32], lvl2[2*32], lvl3[1*32];
        h2(leaves+0*32, leaves+1*32, lvl1+0*32);
        h2(leaves+2*32, leaves+3*32, lvl1+1*32);
        h2(leaves+4*32, leaves+5*32, lvl1+2*32);
        h2(leaves+6*32, leaves+6*32, lvl1+3*32);  // duplicate last
        h2(lvl1+0*32, lvl1+1*32, lvl2+0*32);
        h2(lvl1+2*32, lvl1+3*32, lvl2+1*32);
        h2(lvl2+0*32, lvl2+1*32, lvl3+0*32);
        check(std::memcmp(root, lvl3, 32) == 0,
              "merkle_root_from_leaves 7 leaves matches independent multi-level oracle");
    }

    // ─── merkle_root_from_leaves: null arguments ───────────────────────────
    {
        std::uint8_t leaf[32], scratch[64], root[32];
        for (int i = 0; i < 32; ++i) leaf[i] = nb();
        std::memset(root, 0xAB, 32);
        check(!ufsecp::lbtc::merkle_root_from_leaves(nullptr, 1, scratch, 64, root),
              "merkle_root_from_leaves null leaves rejects");
        check(!ufsecp::lbtc::merkle_root_from_leaves(leaf, 1, nullptr, 64, root),
              "merkle_root_from_leaves null scratch rejects");
        check(!ufsecp::lbtc::merkle_root_from_leaves(leaf, 1, scratch, 64, nullptr),
              "merkle_root_from_leaves null output rejects");
        bool zeroed = true;
        for (int i = 0; i < 32; ++i) if (root[i] != 0) { zeroed = false; break; }
        check(zeroed, "merkle_root_from_leaves null arg zeroes output");
    }

    // ─── merkle_root_from_leaves: scratch undersize ────────────────────────
    {
        std::uint8_t leaves[2 * 32], scratch[63], root[32];  // 63 < 2*64=128
        for (int i = 0; i < 2 * 32; ++i) leaves[i] = nb();
        check(!ufsecp::lbtc::merkle_root_from_leaves(leaves, 2, scratch, 63, root),
              "merkle_root_from_leaves undersize scratch rejects");
        bool zeroed = true;
        for (int i = 0; i < 32; ++i) if (root[i] != 0) { zeroed = false; break; }
        check(zeroed, "merkle_root_from_leaves undersize scratch zeroes output");
    }

    // ─── merkle_root_from_leaves: overflow guard ───────────────────────────
    {
        std::vector<std::uint8_t> big_scratch(128, 0);
        std::uint8_t root[32];
        // count * 32 overflows size_t
        const std::size_t huge = (SIZE_MAX / 32) + 1;
        check(!ufsecp::lbtc::merkle_root_from_leaves(nullptr, huge, big_scratch.data(), big_scratch.size(), root),
              "merkle_root_from_leaves overflow count rejects");
    }

    // ─── merkle_root_from_leaves: hook-decline inherited ───────────────────
    {
        // Verify that merkle_root_from_leaves composes through
        // merkle_pair_hash_batch and inherits its hook-decline behaviour.
        const auto prev = ufsecp::lbtc::gpu_hook::install_lbtc_merkle_pair_hook(decline_merkle_pair_hook);
        std::uint8_t leaves[2 * 32], scratch[128], root[32];
        for (int i = 0; i < 2 * 32; ++i) leaves[i] = nb();
        check(ufsecp::lbtc::merkle_root_from_leaves(leaves, 2, scratch, 128, root),
              "merkle_root_from_leaves hook-decline computes true");
        // Verify against oracle
        std::uint8_t combined[64], oracle[32];
        std::memcpy(combined, leaves, 32);
        std::memcpy(combined + 32, leaves + 32, 32);
        ref_hash256(combined, 64, oracle);
        check(std::memcmp(root, oracle, 32) == 0,
              "merkle_root_from_leaves hook-decline output matches oracle");
        ufsecp::lbtc::gpu_hook::install_lbtc_merkle_pair_hook(prev);
    }

    // ─── merkle_root_from_leaves: fake-hook sentinel inherited ─────────────
    {
        // Install the fake sentinel hook; merkle_root_from_leaves should
        // produce the sentinel through its internal merkle_pair_hash_batch
        // calls, proving the hook path (not CPU fallback) runs.
        const auto prod_hook = ufsecp::lbtc::gpu_hook::install_lbtc_merkle_pair_hook(fake_merkle_pair_hook);
        std::uint8_t leaves[2 * 32], scratch[128], root[32];
        for (int i = 0; i < 2 * 32; ++i) leaves[i] = nb();
        check(ufsecp::lbtc::merkle_root_from_leaves(leaves, 2, scratch, 128, root),
              "merkle_root_from_leaves fake-hook computes true");
        // The fake hook writes 0xDD to every output byte
        bool all_sentinel = true;
        for (int i = 0; i < 32; ++i) if (root[i] != 0xDD) { all_sentinel = false; break; }
        check(all_sentinel, "merkle_root_from_leaves used the installed hook (sentinel in root)");
        ufsecp::lbtc::gpu_hook::install_lbtc_merkle_pair_hook(prod_hook);
    }


    std::printf("=== %s ===\n", fails == 0 ? "ALL PASS" : "SOME FAILED");
    return fails == 0 ? 0 : 1;
}
