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
#include <string>
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

// Fake GPU hook for sighash_descriptor_hash_batch: writes a fixed sentinel
// pattern per-row (0xAA repeated) to prove the hook was called.  Returns 0
// (handled).
int fake_sighash_hook(const std::uint8_t*, std::size_t,
                       const std::uint8_t* const*, const std::uint32_t*,
                       const std::uint32_t* const*,
                       std::size_t count, std::uint8_t* out32) {
    if (!out32) return -1;
    std::memset(out32, 0xAA, count * 32);
    return 0;  // handled
}

// Fake GPU hook for sighash_descriptor_hash_batch that always declines (-1),
// forcing the CPU fallback path.
int decline_sighash_hook(const std::uint8_t*, std::size_t,
                          const std::uint8_t* const*, const std::uint32_t*,
                          const std::uint32_t* const*,
                          std::size_t, std::uint8_t*) {
    return -1;  // always decline → CPU fallback
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


    // ══════════════════════════════════════════════════════════════════════
    // sighash_descriptor_hash_batch tests
    // ══════════════════════════════════════════════════════════════════════

    // Reference oracle: assemble preimage manually and compute HASH256.
    auto sighash_oracle = [](const std::uint8_t* const* fields,
                             const std::uint32_t* flens,
                             const std::uint32_t* const* fvars,
                             const std::uint8_t* desc, std::size_t dlen,
                             std::size_t row, std::uint8_t out[32]) {
        // Re-parse descriptor to assemble preimage (mirrors the implementation).
        std::vector<std::uint8_t> preimage;
        const std::uint8_t* p = desc;
        const std::uint8_t* end = desc + dlen;
        while (p < end && *p != 0xFF) {
            std::uint16_t fid = static_cast<std::uint16_t>(p[0]) |
                                (static_cast<std::uint16_t>(p[1] & 0x0Fu) << 8);
            std::uint8_t flags = p[1] >> 4;
            bool has_len = (flags & 0x01u) != 0;
            if (has_len) {
                std::uint32_t vl = fvars[fid][row];
                const std::uint8_t* src = fields[fid] + row * flens[fid];
                preimage.insert(preimage.end(), src, src + vl);
            } else {
                // Determine fixed len from field_id
                std::uint32_t fl = 0;
                switch (fid) {
                case 0x00: fl = 4; break;
                case 0x01: fl = 32; break;
                case 0x02: fl = 32; break;
                case 0x03: fl = 36; break;
                case 0x05: fl = 8; break;
                case 0x06: fl = 4; break;
                case 0x07: fl = 32; break;
                case 0x08: fl = 4; break;
                case 0x09: fl = 4; break;
                case 0x0A: fl = 36; break;
                case 0x0B: fl = 4; break;
                // 0x0C..0x0F are Taproot-only, rejected by the parser.
                default: fl = 0; break;
                }
                const std::uint8_t* src = fields[fid] + row * flens[fid];
                preimage.insert(preimage.end(), src, src + fl);
            }
            p += 2;
        }
        ref_hash256(preimage.data(), preimage.size(), out);
    };

    // ─── Valid BIP143-style descriptor (all fixed fields) ─────────────
    {
        // Descriptor: nVersion,hashPrevouts,hashSequence,outpoint,value,
        //            nSequence,hashOutputs,nLocktime,nHashType (9 fixed fields)
        // BIP143-style (no scriptCode for simplicity — all fixed)
        std::uint8_t desc[] = {
            0x00, 0x00,  // nVersion
            0x01, 0x00,  // hashPrevouts
            0x02, 0x00,  // hashSequence
            0x03, 0x00,  // outpoint
            0x05, 0x00,  // value
            0x06, 0x00,  // nSequence
            0x07, 0x00,  // hashOutputs
            0x08, 0x00,  // nLocktime
            0x09, 0x00,  // nHashType
            0xFF         // terminator
        };
        const std::size_t dlen = sizeof(desc);

        constexpr std::size_t N = 10;
        // Allocate field data for all possible field_ids (256 entries for safety)
        std::vector<std::uint8_t> fbuf[256];
        std::vector<std::uint32_t> flens_data(256, 0);
        std::vector<std::uint32_t> fvars_buf[256];
        const std::uint8_t* fdata_ptrs[256] = {};
        const std::uint32_t* fvars_ptrs[256] = {};
        for (int i = 0; i < 256; ++i) { fdata_ptrs[i] = nullptr; fvars_ptrs[i] = nullptr; }

        // Populate fields with random data
        auto populate_fixed = [&](std::uint16_t fid, std::uint32_t flen) {
            fbuf[fid].resize(N * flen);
            for (auto& b : fbuf[fid]) b = nb();
            flens_data[fid] = flen;
            fdata_ptrs[fid] = fbuf[fid].data();
        };
        populate_fixed(0x00, 4);   // nVersion
        populate_fixed(0x01, 32);  // hashPrevouts
        populate_fixed(0x02, 32);  // hashSequence
        populate_fixed(0x03, 36);  // outpoint
        populate_fixed(0x05, 8);   // value
        populate_fixed(0x06, 4);   // nSequence
        populate_fixed(0x07, 32);  // hashOutputs
        populate_fixed(0x08, 4);   // nLocktime
        populate_fixed(0x09, 4);   // nHashType

        std::vector<std::uint8_t> out(N * 32, 0xCD);
        check(ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc, dlen, fdata_ptrs, flens_data.data(), fvars_ptrs, N, out.data()),
              "sighash BIP143-style all-fixed computes true");

        // Cross-check each row against the independent oracle
        int mism = 0;
        for (std::size_t i = 0; i < N; ++i) {
            std::uint8_t oracle[32];
            sighash_oracle(fdata_ptrs, flens_data.data(), fvars_ptrs, desc, dlen, i, oracle);
            if (std::memcmp(out.data() + i * 32, oracle, 32) != 0) ++mism;
        }
        check(mism == 0, "sighash BIP143-style bit-exact vs independent oracle");
    }

    // ─── count=0 no-op ────────────────────────────────────────────────
    {
        std::uint8_t desc[] = {0x09, 0x00, 0xFF};  // nHashType only
        std::uint8_t out[32];
        std::memset(out, 0xAB, 32);
        const std::uint8_t* dummy_ptr = out;  // non-null but unused
        std::uint32_t dummy_len = 4;
        const std::uint32_t* dummy_var = nullptr;
        check(ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc, 3, &dummy_ptr, &dummy_len, &dummy_var, 0, out),
              "sighash count==0 returns true");
        // out must be untouched
        bool untouched = true;
        for (int i = 0; i < 32; ++i) if (out[i] != 0xAB) { untouched = false; break; }
        check(untouched, "sighash count==0 leaves out untouched");
    }

    // ─── descriptor_len=0 rejected ────────────────────────────────────
    {
        std::uint8_t desc[] = {0xFF};
        std::uint8_t out[32];
        std::memset(out, 0xAB, 32);
        const std::uint8_t* dptr = desc;
        std::uint32_t dummy_len = 4;
        check(!ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc, 0, &dptr, &dummy_len, nullptr, 1, out),
              "sighash descriptor_len==0 rejected");
        bool untouched = true;
        for (int i = 0; i < 32; ++i) if (out[i] != 0xAB) { untouched = false; break; }
        check(untouched, "sighash descriptor_len==0 leaves out untouched");
    }

    // ─── Missing terminator rejected ──────────────────────────────────
    {
        std::uint8_t desc[] = {0x09, 0x00};  // no terminator
        std::uint8_t out[32];
        std::memset(out, 0xAB, 32);
        const std::uint8_t* dptr = desc;
        std::uint32_t dummy_len = 4;
        check(!ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc, 2, &dptr, &dummy_len, nullptr, 1, out),
              "sighash missing terminator rejected");
        bool untouched = true;
        for (int i = 0; i < 32; ++i) if (out[i] != 0xAB) { untouched = false; break; }
        check(untouched, "sighash missing terminator leaves out untouched");
    }

    // ─── Duplicate field rejected ─────────────────────────────────────
    {
        // nHashType appears twice
        std::uint8_t desc[] = {0x09, 0x00, 0x09, 0x00, 0xFF};
        std::uint8_t out[32];
        std::memset(out, 0xAB, 32);
        std::uint8_t nht_buf[4] = {1,0,0,0};
        const std::uint8_t* fdata[256] = {};
        std::uint32_t flens[256] = {};
        fdata[0x09] = nht_buf;
        flens[0x09] = 4;
        check(!ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc, 5, fdata, flens, nullptr, 1, out),
              "sighash duplicate field rejected");
        bool untouched = true;
        for (int i = 0; i < 32; ++i) if (out[i] != 0xAB) { untouched = false; break; }
        check(untouched, "sighash duplicate field leaves out untouched");
    }

    // ─── Reserved flag bits set → rejected ────────────────────────────
    {
        // Set bit14 (flags nibble bit2) on nVersion
        std::uint8_t desc[] = {0x00, 0x40, 0x09, 0x00, 0xFF};  // 0x40 = flags nibble 4 (bit2 set)
        std::uint8_t out[32];
        std::memset(out, 0xAB, 32);
        std::uint8_t nver_buf[4] = {2,0,0,0};
        std::uint8_t nht_buf[4] = {1,0,0,0};
        const std::uint8_t* fdata[256] = {};
        std::uint32_t flens[256] = {};
        fdata[0x00] = nver_buf; flens[0x00] = 4;
        fdata[0x09] = nht_buf;  flens[0x09] = 4;
        check(!ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc, 5, fdata, flens, nullptr, 1, out),
              "sighash reserved flag bit14 rejected");
        bool untouched = true;
        for (int i = 0; i < 32; ++i) if (out[i] != 0xAB) { untouched = false; break; }
        check(untouched, "sighash reserved flag leaves out untouched");
    }

    // ─── Reserved low-byte 0xFF field_id rejected ─────────────────────
    {
        // field_id = 0x00FF (low byte = 0xFF)
        std::uint8_t desc[] = {0xFF, 0x00, 0x09, 0x00, 0xFF};
        std::uint8_t out[32];
        std::memset(out, 0xAB, 32);
        std::uint8_t nht_buf[4] = {1,0,0,0};
        const std::uint8_t* fdata[256] = {};
        std::uint32_t flens[256] = {};
        fdata[0x09] = nht_buf; flens[0x09] = 4;
        check(!ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc, 5, fdata, flens, nullptr, 1, out),
              "sighash reserved 0xFF low-byte field rejected");
        bool untouched = true;
        for (int i = 0; i < 32; ++i) if (out[i] != 0xAB) { untouched = false; break; }
        check(untouched, "sighash reserved 0xFF low-byte leaves out untouched");
    }

    // ─── Null args rejected ───────────────────────────────────────────
    {
        std::uint8_t desc[] = {0x09, 0x00, 0xFF};
        std::uint8_t out[32];
        std::memset(out, 0xAB, 32);
        std::uint8_t nht[4] = {1,0,0,0};
        const std::uint8_t* fdata[256] = {};
        std::uint32_t flens[256] = {};
        fdata[0x09] = nht; flens[0x09] = 4;

        check(!ufsecp::lbtc::sighash_descriptor_hash_batch(
                  nullptr, 3, fdata, flens, nullptr, 1, out),
              "sighash null descriptor rejected");
        check(!ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc, 3, nullptr, flens, nullptr, 1, out),
              "sighash null field_data rejected");
        check(!ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc, 3, fdata, nullptr, nullptr, 1, out),
              "sighash null field_lengths rejected");
        check(!ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc, 3, fdata, flens, nullptr, 1, nullptr),
              "sighash null out32 rejected");
        // out32 must be untouched after null-arg failures
        bool untouched = true;
        for (int i = 0; i < 32; ++i) if (out[i] != 0xAB) { untouched = false; break; }
        check(untouched, "sighash null args leave out untouched");
    }

    // ─── Fixed + variable fields (scriptCode with HAS_LENGTH) ─────────
    {
        // Descriptor: nVersion, nHashType, scriptCode(var)
        // scriptCode (0x04) with HAS_LENGTH flag
        std::uint8_t desc[] = {
            0x00, 0x00,  // nVersion (fixed 4)
            0x04, 0x10,  // scriptCode (HAS_LENGTH, variable)
            0x09, 0x00,  // nHashType (fixed 4)
            0xFF
        };
        const std::size_t dlen = sizeof(desc);

        constexpr std::size_t N = 5;
        constexpr std::uint32_t SCRIPT_STRIDE = 128;

        std::vector<std::uint8_t> nver(N * 4);
        std::vector<std::uint8_t> scode(N * SCRIPT_STRIDE);
        std::vector<std::uint8_t> nht(N * 4);
        for (auto& b : nver) b = nb();
        for (auto& b : scode) b = nb();
        for (auto& b : nht) b = nb();

        std::vector<std::uint32_t> scode_lens(N);
        for (std::size_t i = 0; i < N; ++i)
            scode_lens[i] = static_cast<std::uint32_t>(10 + (i % 30));  // 10-39 bytes each

        const std::uint8_t* fdata[256] = {};
        std::uint32_t flens[256] = {};
        const std::uint32_t* fvars[256] = {};

        fdata[0x00] = nver.data();   flens[0x00] = 4;
        fdata[0x04] = scode.data();  flens[0x04] = SCRIPT_STRIDE;
        fdata[0x09] = nht.data();    flens[0x09] = 4;
        fvars[0x04] = scode_lens.data();

        std::vector<std::uint8_t> out(N * 32);
        check(ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc, dlen, fdata, flens, fvars, N, out.data()),
              "sighash fixed+variable (scriptCode) computes true");

        int mism = 0;
        for (std::size_t i = 0; i < N; ++i) {
            std::uint8_t oracle[32];
            sighash_oracle(fdata, flens, fvars, desc, dlen, i, oracle);
            if (std::memcmp(out.data() + i * 32, oracle, 32) != 0) ++mism;
        }
        check(mism == 0, "sighash fixed+variable bit-exact vs independent oracle");

        // var_len > stride must fail
        scode_lens[2] = SCRIPT_STRIDE + 1;
        std::memset(out.data(), 0xAB, out.size());
        check(!ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc, dlen, fdata, flens, fvars, N, out.data()),
              "sighash var_len > stride rejected");
        // Reset for subsequent tests
        scode_lens[2] = 12;

        std::memset(out.data(), 0xAB, out.size());
        check(!ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc, dlen, fdata, flens, nullptr, N, out.data()),
              "sighash null field_var_lens base rejected for variable field");
        bool untouched = true;
        for (auto b : out) {
            if (b != 0xAB) { untouched = false; break; }
        }
        check(untouched, "sighash null field_var_lens base leaves out untouched");
    }

    // ─── LE/BE negative: BE-swapped nVersion produces different hash ───
    {
        // Descriptor: nVersion (4 bytes LE), nHashType (4 bytes LE)
        std::uint8_t desc[] = {0x00, 0x00, 0x09, 0x00, 0xFF};
        const std::size_t dlen = sizeof(desc);

        // LE nVersion: 0x02000000 (version 2)
        std::uint8_t nver_le[4] = {2, 0, 0, 0};
        // BE nVersion: 0x00000002 (wrong byte order)
        std::uint8_t nver_be[4] = {0, 0, 0, 2};
        std::uint8_t nht[4] = {1, 0, 0, 0};  // SIGHASH_ALL

        const std::uint8_t* fdata_le[256] = {};
        const std::uint8_t* fdata_be[256] = {};
        std::uint32_t flens[256] = {};
        fdata_le[0x00] = nver_le;  fdata_le[0x09] = nht;
        fdata_be[0x00] = nver_be;  fdata_be[0x09] = nht;
        flens[0x00] = 4; flens[0x09] = 4;

        std::uint8_t out_le[32], out_be[32];
        check(ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc, dlen, fdata_le, flens, nullptr, 1, out_le),
              "sighash LE nVersion computes true");
        check(ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc, dlen, fdata_be, flens, nullptr, 1, out_be),
              "sighash BE nVersion computes true");
        // BE-swapped nVersion must produce a DIFFERENT hash
        check(std::memcmp(out_le, out_be, 32) != 0,
              "sighash BE nVersion produces different hash vs LE");
    }

    // ─── LE/BE negative: BE-swapped nHashType produces different hash ──
    {
        std::uint8_t desc[] = {0x00, 0x00, 0x09, 0x00, 0xFF};
        const std::size_t dlen = sizeof(desc);

        std::uint8_t nver[4] = {2, 0, 0, 0};
        std::uint8_t nht_le[4] = {1, 0, 0, 0};   // SIGHASH_ALL LE
        std::uint8_t nht_be[4] = {0, 0, 0, 1};   // SIGHASH_ALL BE (wrong)

        const std::uint8_t* fdata_le[256] = {};
        const std::uint8_t* fdata_be[256] = {};
        std::uint32_t flens[256] = {};
        fdata_le[0x00] = nver;  fdata_le[0x09] = nht_le;
        fdata_be[0x00] = nver;  fdata_be[0x09] = nht_be;
        flens[0x00] = 4; flens[0x09] = 4;

        std::uint8_t out_le[32], out_be[32];
        check(ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc, dlen, fdata_le, flens, nullptr, 1, out_le),
              "sighash LE nHashType computes true");
        check(ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc, dlen, fdata_be, flens, nullptr, 1, out_be),
              "sighash BE nHashType computes true");
        check(std::memcmp(out_le, out_be, 32) != 0,
              "sighash BE nHashType produces different hash vs LE");
    }

    // ─── LE/BE negative: BE-swapped amount (8-byte value) ─────────────
    {
        std::uint8_t desc[] = {0x05, 0x00, 0x09, 0x00, 0xFF};
        const std::size_t dlen = sizeof(desc);

        std::uint8_t val_le[8] = {0x40, 0x0D, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00}; // 200000 sat LE
        std::uint8_t val_be[8] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x0D, 0x40}; // 200000 sat BE
        std::uint8_t nht[4] = {1, 0, 0, 0};

        const std::uint8_t* fdata_le[256] = {};
        const std::uint8_t* fdata_be[256] = {};
        std::uint32_t flens[256] = {};
        fdata_le[0x05] = val_le;  fdata_le[0x09] = nht;
        fdata_be[0x05] = val_be;  fdata_be[0x09] = nht;
        flens[0x05] = 8; flens[0x09] = 4;

        std::uint8_t out_le[32], out_be[32];
        check(ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc, dlen, fdata_le, flens, nullptr, 1, out_le),
              "sighash LE amount computes true");
        check(ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc, dlen, fdata_be, flens, nullptr, 1, out_be),
              "sighash BE amount computes true");
        check(std::memcmp(out_le, out_be, 32) != 0,
              "sighash BE amount produces different hash vs LE");
    }

    // ─── Max descriptor length boundary (129 bytes = 64 fields) ───────
    {
        // Build a 129-byte descriptor with 64 distinct supported field_ids.
        // Use nVersion(0x00) through codesep_pos(0x0F) + raw_literal(0xF0) = 17 fields,
        // then repeat with different high-nibble field_ids.
        // Actually we can only use unique field_ids up to 0xF0, so we max at 17.
        // For the 129-byte boundary test, use the maximum number of fields
        // within the supported range, which is 17 fields = 35 bytes.
        // A full 64-field test needs field_ids beyond 0xF0 which are all reserved.
        // So test: descriptor_len == 129 with reserved fields → rejected.
        // And test: descriptor_len == 129 with all-zero bytes → rejected
        // (mid-stream 0xFF check catches 0xFF in low-byte positions).
        // Test that a 129-byte-max descriptor with a clean terminator is
        // properly rejected for containing unsupported field_ids.
        std::uint8_t desc129[129];
        std::memset(desc129, 0x10, 128);  // field_id=0x10, flags=0x1 (reserved field_id)
        desc129[128] = 0xFF;  // terminator
        std::uint8_t out[32];
        std::memset(out, 0xAB, 32);
        const std::uint8_t* dummy_ptr = out;
        std::uint32_t dummy_len = 4;
        check(!ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc129, 129, &dummy_ptr, &dummy_len, nullptr, 1, out),
              "sighash 129-byte descriptor with reserved fields rejected");
        bool untouched = true;
        for (int i = 0; i < 32; ++i) if (out[i] != 0xAB) { untouched = false; break; }
        check(untouched, "sighash 129-byte reserved leaves out untouched");

        // Test: valid max-length descriptor with all supported fields
        // (excluding reserved 0x0C..0x0F = 13 fields = 27 bytes)
        std::uint16_t all_fields[] = {
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
            0x08, 0x09, 0x0A, 0x0B, 0xF0
        };
        constexpr std::size_t NF = sizeof(all_fields) / sizeof(all_fields[0]);

        std::uint8_t desc_all[NF * 2 + 1];
        const std::uint8_t* fdata_all[256] = {};
        std::uint32_t flens_all[256] = {};
        const std::uint32_t* fvars_all[256] = {};

        for (std::size_t i = 0; i < NF; ++i) {
            std::uint16_t fid = all_fields[i];
            desc_all[i * 2 + 0] = static_cast<std::uint8_t>(fid & 0xFF);
            bool is_var = (fid == 0x04 || fid == 0xF0);
            std::uint8_t flags = is_var ? 0x10 : 0x00;  // HAS_LENGTH for variable
            desc_all[i * 2 + 1] = static_cast<std::uint8_t>(((fid >> 8) & 0x0F) | flags);
            // Allocate dummy buffers
            static std::vector<std::uint8_t> abuf[256];
            std::uint32_t alen = is_var ? 64u : 8u;
            switch (fid) {
            case 0x00: case 0x06: case 0x08: case 0x09: case 0x0B:
                alen = 4; break;
            case 0x01: case 0x02: case 0x07: alen = 32; break;
            case 0x03: case 0x0A: alen = 36; break;
            case 0x05: alen = 8; break;
            default: alen = 64; break;  // variable fields
            }
            abuf[fid].resize(alen);
            for (auto& b : abuf[fid]) b = nb();
            fdata_all[fid] = abuf[fid].data();
            flens_all[fid] = alen;
            if (is_var) {
                static std::vector<std::uint32_t> vbuf[256];
                vbuf[fid].resize(1);
                vbuf[fid][0] = alen / 2;
                fvars_all[fid] = vbuf[fid].data();
            }
        }
        desc_all[NF * 2] = 0xFF;

        std::uint8_t out_all[32];
        check(ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc_all, NF * 2 + 1, fdata_all, flens_all, fvars_all, 1, out_all),
              "sighash max-supported-fields (13 fields) computes true");
    }

    // ─── Missing nHashType rejected ───────────────────────────────────
    {
        std::uint8_t desc[] = {0x00, 0x00, 0xFF};  // only nVersion
        std::uint8_t out[32];
        std::memset(out, 0xAB, 32);
        std::uint8_t nver[4] = {2,0,0,0};
        const std::uint8_t* fdata[256] = {};
        std::uint32_t flens[256] = {};
        fdata[0x00] = nver; flens[0x00] = 4;
        check(!ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc, 3, fdata, flens, nullptr, 1, out),
              "sighash missing nHashType rejected");
    }

    // ─── Mid-stream 0xFF collision rejected ────────────────────────────
    {
        // Place 0xFF at an even position (field_ref low byte)
        std::uint8_t desc[] = {0xFF, 0x00, 0x09, 0x00, 0xFF};
        std::uint8_t out[32];
        std::memset(out, 0xAB, 32);
        std::uint8_t nht[4] = {1,0,0,0};
        const std::uint8_t* fdata[256] = {};
        std::uint32_t flens[256] = {};
        fdata[0x09] = nht; flens[0x09] = 4;
        check(!ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc, 5, fdata, flens, nullptr, 1, out),
              "sighash mid-stream 0xFF rejected");
    }

    // ─── field_lengths[f] * count overflow rejected ───────────────────
    {
        std::uint8_t desc[] = {0x09, 0x00, 0xFF};
        std::uint8_t out[32];
        std::memset(out, 0xAB, 32);
        std::uint8_t nht[4] = {1,0,0,0};
        const std::uint8_t* fdata[256] = {};
        std::uint32_t flens[256] = {};
        fdata[0x09] = nht; flens[0x09] = 4;
        const std::size_t huge = (SIZE_MAX / 4) + 1;
        check(!ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc, 3, fdata, flens, nullptr, huge, out),
              "sighash field_lengths*count overflow rejected");
        bool untouched = true;
        for (int i = 0; i < 32; ++i) if (out[i] != 0xAB) { untouched = false; break; }
        check(untouched, "sighash overflow leaves out untouched");
    }

    // ─── Null field_data[f] for referenced field rejected ─────────────
    {
        std::uint8_t desc[] = {0x09, 0x00, 0xFF};
        std::uint8_t out[32];
        std::memset(out, 0xAB, 32);
        const std::uint8_t* fdata[256] = {};
        std::uint32_t flens[256] = {};
        fdata[0x09] = nullptr;  // null for referenced field
        flens[0x09] = 4;
        check(!ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc, 3, fdata, flens, nullptr, 1, out),
              "sighash null field_data[0x09] rejected");
    }

    // ─── HAS_LENGTH mismatch on fixed field rejected ──────────────────
    {
        // nVersion (0x00) is fixed, but we set HAS_LENGTH flag
        std::uint8_t desc[] = {0x00, 0x10, 0x09, 0x00, 0xFF};
        std::uint8_t out[32];
        std::memset(out, 0xAB, 32);
        std::uint8_t nver[4] = {2,0,0,0};
        std::uint8_t nht[4] = {1,0,0,0};
        const std::uint8_t* fdata[256] = {};
        std::uint32_t flens[256] = {};
        fdata[0x00] = nver; flens[0x00] = 4;
        fdata[0x09] = nht;  flens[0x09] = 4;
        check(!ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc, 5, fdata, flens, nullptr, 1, out),
              "sighash HAS_LENGTH on fixed field rejected");
    }

    // ─── Missing HAS_LENGTH on variable field rejected ────────────────
    {
        // scriptCode (0x04) is variable but HAS_LENGTH flag not set
        std::uint8_t desc[] = {0x04, 0x00, 0x09, 0x00, 0xFF};
        std::uint8_t out[32];
        std::memset(out, 0xAB, 32);
        std::uint8_t sc[64], nht[4] = {1,0,0,0};
        const std::uint8_t* fdata[256] = {};
        std::uint32_t flens[256] = {};
        fdata[0x04] = sc; flens[0x04] = 64;
        fdata[0x09] = nht; flens[0x09] = 4;
        check(!ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc, 5, fdata, flens, nullptr, 1, out),
              "sighash missing HAS_LENGTH on variable field rejected");
    }

    // ─── ZERO_PAD flag accepted (no-op for preimage assembly) ─────────
    {
        // nVersion with ZERO_PAD flag set — should still work correctly
        std::uint8_t desc[] = {0x00, 0x20, 0x09, 0x00, 0xFF};  // 0x20 = ZERO_PAD flag
        const std::size_t dlen = sizeof(desc);
        std::uint8_t nver[4] = {2,0,0,0};
        std::uint8_t nht[4] = {1,0,0,0};
        const std::uint8_t* fdata[256] = {};
        std::uint32_t flens[256] = {};
        fdata[0x00] = nver; flens[0x00] = 4;
        fdata[0x09] = nht;  flens[0x09] = 4;

        std::uint8_t out[32];
        check(ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc, dlen, fdata, flens, nullptr, 1, out),
              "sighash ZERO_PAD on fixed field computes true");

        // Compare with no-ZERO_PAD version
        std::uint8_t desc_nopad[] = {0x00, 0x00, 0x09, 0x00, 0xFF};
        std::uint8_t out_nopad[32];
        check(ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc_nopad, sizeof(desc_nopad), fdata, flens, nullptr, 1, out_nopad),
              "sighash no-ZERO_PAD computes true");
        check(std::memcmp(out, out_nopad, 32) == 0,
              "sighash ZERO_PAD does not change hash (no-op for fixed fields)");
    }

    // ─── Unsupported field_ids rejected ───────────────────────────────
    {
        // field_id 0x10 is reserved_general
        std::uint8_t desc[] = {0x10, 0x00, 0x09, 0x00, 0xFF};
        std::uint8_t out[32];
        std::memset(out, 0xAB, 32);
        std::uint8_t nht[4] = {1,0,0,0};
        const std::uint8_t* fdata[256] = {};
        std::uint32_t flens[256] = {};
        fdata[0x09] = nht; flens[0x09] = 4;
        check(!ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc, 5, fdata, flens, nullptr, 1, out),
              "sighash unsupported field_id 0x10 rejected");
    }

    // ─── field_lengths[f] == 0 for referenced field rejected ──────────
    {
        std::uint8_t desc[] = {0x09, 0x00, 0xFF};
        std::uint8_t out[32];
        std::memset(out, 0xAB, 32);
        std::uint8_t nht[4] = {1,0,0,0};
        const std::uint8_t* fdata[256] = {};
        std::uint32_t flens[256] = {};
        fdata[0x09] = nht; flens[0x09] = 0;  // zero stride
        check(!ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc, 3, fdata, flens, nullptr, 1, out),
              "sighash zero stride rejected");
    }

    // ─── Even descriptor_len rejected ─────────────────────────────────
    {
        std::uint8_t desc[] = {0x09, 0x00};  // 2 bytes, even, no terminator
        std::uint8_t out[32];
        std::memset(out, 0xAB, 32);
        const std::uint8_t* dptr = desc;
        std::uint32_t dummy_len = 4;
        check(!ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc, 2, &dptr, &dummy_len, nullptr, 1, out),
              "sighash even descriptor_len rejected");
    }

    // ─── Truncated field_ref (odd length, no terminator at end) ───────
    {
        // 3 bytes: [0x00, 0x00, 0x09] — last byte is not 0xFF
        std::uint8_t desc[] = {0x00, 0x00, 0x09};
        std::uint8_t out[32];
        std::memset(out, 0xAB, 32);
        const std::uint8_t* dptr = desc;
        std::uint32_t dummy_len = 4;
        check(!ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc, 3, &dptr, &dummy_len, nullptr, 1, out),
              "sighash missing terminator at expected position rejected");
    }

    // ─── Sighash GPU hook: handled (fake hook writes sentinel) ────────
    {
        // Install a fake hook that writes 0xAA to all output rows.
        auto prev = ufsecp::lbtc::gpu_hook::install_lbtc_sighash_hook(fake_sighash_hook);

        std::uint8_t desc[] = {0x00, 0x00, 0x09, 0x00, 0xFF};
        std::uint8_t nver[4] = {2, 0, 0, 0};
        std::uint8_t nht[4]  = {1, 0, 0, 0};
        const std::uint8_t* fdata[256] = {};
        std::uint32_t flens[256] = {};
        fdata[0x00] = nver; flens[0x00] = 4;
        fdata[0x09] = nht;  flens[0x09] = 4;

        constexpr std::size_t N = 3;
        std::uint8_t out[N * 32];
        std::memset(out, 0xCD, sizeof(out));

        check(ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc, 5, fdata, flens, nullptr, N, out),
              "sighash GPU hook handled returns true");

        // Hook should have written 0xAA to every byte.
        bool all_AA = true;
        for (std::size_t i = 0; i < sizeof(out); ++i)
            if (out[i] != 0xAA) { all_AA = false; break; }
        check(all_AA, "sighash GPU hook wrote sentinel 0xAA to output");

        // Restore previous hook (or null).
        ufsecp::lbtc::gpu_hook::install_lbtc_sighash_hook(prev);
    }

    // ─── Sighash GPU hook: decline → CPU fallback (byte-identical) ────
    {
        // Install a hook that always declines.
        auto prev = ufsecp::lbtc::gpu_hook::install_lbtc_sighash_hook(decline_sighash_hook);

        std::uint8_t desc[] = {0x00, 0x00, 0x05, 0x00, 0x09, 0x00, 0xFF};
        std::uint8_t nver[4] = {2, 0, 0, 0};
        std::uint8_t val[8]  = {0x40, 0x0D, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00};
        std::uint8_t nht[4]  = {1, 0, 0, 0};
        const std::uint8_t* fdata[256] = {};
        std::uint32_t flens[256] = {};
        fdata[0x00] = nver; flens[0x00] = 4;
        fdata[0x05] = val;  flens[0x05] = 8;
        fdata[0x09] = nht;  flens[0x09] = 4;

        std::uint8_t out_decline[32];
        check(ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc, 7, fdata, flens, nullptr, 1, out_decline),
              "sighash decline hook → CPU fallback returns true");

        // Now run with no hook → pure CPU path.
        ufsecp::lbtc::gpu_hook::install_lbtc_sighash_hook(nullptr);

        std::uint8_t out_cpu[32];
        check(ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc, 7, fdata, flens, nullptr, 1, out_cpu),
              "sighash no-hook CPU returns true");

        // Results must be byte-identical.
        check(std::memcmp(out_decline, out_cpu, 32) == 0,
              "sighash decline→CPU byte-identical to pure CPU");

        // Restore.
        ufsecp::lbtc::gpu_hook::install_lbtc_sighash_hook(prev);
    }

    // ─── Sighash GPU hook: null hook → straight to CPU ─────────────────
    {
        // Ensure no hook is installed.
        auto prev = ufsecp::lbtc::gpu_hook::install_lbtc_sighash_hook(nullptr);

        std::uint8_t desc[] = {0x00, 0x00, 0x09, 0x00, 0xFF};
        std::uint8_t nver[4] = {2, 0, 0, 0};
        std::uint8_t nht[4]  = {1, 0, 0, 0};
        const std::uint8_t* fdata[256] = {};
        std::uint32_t flens[256] = {};
        fdata[0x00] = nver; flens[0x00] = 4;
        fdata[0x09] = nht;  flens[0x09] = 4;

        std::uint8_t out[32];
        check(ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc, 5, fdata, flens, nullptr, 1, out),
              "sighash null hook computes true");

        // Verify against independent oracle.
        std::uint8_t oracle[32];
        {
            // Build preimage: nVersion[4] || nHashType[4]
            std::uint8_t preimage[8];
            std::memcpy(preimage,     nver, 4);
            std::memcpy(preimage + 4, nht,  4);
            ref_hash256(preimage, 8, oracle);
        }
        check(std::memcmp(out, oracle, 32) == 0,
              "sighash null hook byte-identical vs independent oracle");

        // Restore.
        ufsecp::lbtc::gpu_hook::install_lbtc_sighash_hook(prev);
    }

    // ─── Sighash GPU hook: out32 untouched on validation failure ───────
    {
        auto prev = ufsecp::lbtc::gpu_hook::install_lbtc_sighash_hook(fake_sighash_hook);

        // Malformed descriptor (even length) → must fail BEFORE hook is called.
        std::uint8_t desc[] = {0x00, 0x00, 0x09, 0x00};  // even length, no terminator
        std::uint8_t out[32];
        std::memset(out, 0xAB, 32);
        check(!ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc, 4, nullptr, nullptr, nullptr, 1, out),
              "sighash malformed descriptor (before hook) rejected");

        // out32 must be untouched (still 0xAB).
        bool untouched = true;
        for (int i = 0; i < 32; ++i) if (out[i] != 0xAB) { untouched = false; break; }
        check(untouched, "sighash malformed descriptor leaves out untouched (hook not called)");

        ufsecp::lbtc::gpu_hook::install_lbtc_sighash_hook(prev);
    }

    // ─── Taproot field IDs 0x0C..0x0F rejected ────────────────────────
    // These four field IDs are reserved for future BIP-341 TapSighash
    // tagged-hash mode and must be rejected by the current HASH256 parser.
    {
        const std::uint16_t taproot_fids[] = {0x0C, 0x0D, 0x0E, 0x0F};
        std::uint8_t nht[4] = {1,0,0,0};
        const std::uint8_t* fdata[256] = {};
        std::uint32_t flens[256] = {};
        fdata[0x09] = nht; flens[0x09] = 4;

        for (int ti = 0; ti < 4; ++ti) {
            std::uint16_t fid = taproot_fids[ti];
            std::uint8_t desc[] = {
                static_cast<std::uint8_t>(fid & 0xFF),
                static_cast<std::uint8_t>((fid >> 8) & 0x0F),
                0x09, 0x00, 0xFF
            };
            std::uint8_t out[32];
            std::memset(out, 0xAB, 32);
            bool ok = ufsecp::lbtc::sighash_descriptor_hash_batch(
                desc, 5, fdata, flens, nullptr, 1, out);
            check(!ok, ("sighash Taproot field 0x" +
                        std::string(fid <= 0x0F ? "0" : "") +
                        std::to_string(fid) + " rejected").c_str());
            // out32 must be untouched on rejection.
            bool untouched = true;
            for (int i = 0; i < 32; ++i)
                if (out[i] != 0xAB) { untouched = false; break; }
            check(untouched, ("sighash Taproot field 0x" +
                             std::string(fid <= 0x0F ? "0" : "") +
                             std::to_string(fid) + " leaves out untouched").c_str());
        }
    }

    // ─── Field-boundary HASH256 KATs: var-len 0/1/63/64/65 ────────────
    // These lengths span every interesting SHA-256 block-boundary case:
    //   empty field (0), single byte (1), one byte short of block (63),
    //   exactly one block (64), one byte into second block (65).
    // Uses raw_literal (0xF0) with HAS_LENGTH for precise length control,
    // plus nHashType to satisfy the mandatory nHashType requirement.
    // vlen=0 still requires non-null backing storage (stride≥1) while
    // var_len[] correctly reports 0 actual bytes.
    {
        const std::uint32_t test_lens[] = {0, 1, 63, 64, 65};
        for (int ti = 0; ti < 5; ++ti) {
            std::uint32_t vlen = test_lens[ti];

            // descriptor: raw_literal (0xF0, HAS_LENGTH) + nHashType (0x09)
            std::uint8_t desc[] = {
                0xF0, 0x10,  // raw_literal, HAS_LENGTH
                0x09, 0x00,  // nHashType
                0xFF
            };

            // Non-null backing even for vlen=0: allocate at least 1 byte.
            std::vector<std::uint8_t> raw_lit(vlen > 0 ? vlen : 1u, 0);
            for (std::uint32_t j = 0; j < vlen; ++j) raw_lit[j] = static_cast<std::uint8_t>(j + ti);
            std::uint8_t nht[4] = {1, 0, 0, 0};

            const std::uint8_t* fdata[256] = {};
            std::uint32_t flens[256] = {};
            const std::uint32_t* fvars[256] = {};
            std::vector<std::uint32_t> vlens(1, vlen);

            // Stride must be ≥1 for referenced fields; for vlen=0 use stride=1
            // with var_len=0 so no bytes are actually consumed.
            fdata[0xF0] = raw_lit.data(); flens[0xF0] = vlen > 0 ? vlen : 1u;
            fdata[0x09] = nht;            flens[0x09] = 4;
            fvars[0xF0] = vlens.data();

            std::uint8_t out[32];
            bool ok = ufsecp::lbtc::sighash_descriptor_hash_batch(
                desc, 5, fdata, flens, fvars, 1, out);
            check(ok, ("sighash var-len=" + std::to_string(vlen) + " computes true").c_str());

            // Independent oracle: concatenate raw_literal (vlen bytes only) || nHashType, HASH256.
            // For vlen=0, raw_lit has 1-byte non-null backing but we must only hash 0 bytes.
            std::vector<std::uint8_t> preimage;
            preimage.insert(preimage.end(), raw_lit.begin(), raw_lit.begin() + vlen);
            preimage.insert(preimage.end(), nht, nht + 4);
            std::uint8_t oracle[32];
            ref_hash256(preimage.data(), preimage.size(), oracle);
            check(std::memcmp(out, oracle, 32) == 0,
                  ("sighash var-len=" + std::to_string(vlen) + " bit-exact vs oracle").c_str());
        }
    }

    // ─── Multi-block field HASH256 KAT (129+ bytes) ───────────────────
    {
        const std::uint32_t vlen = 129;  // spans 3 SHA-256 blocks (64+64+1)
        std::uint8_t desc[] = {
            0xF0, 0x10,  // raw_literal, HAS_LENGTH
            0x09, 0x00,  // nHashType
            0xFF
        };

        std::vector<std::uint8_t> raw_lit(vlen);
        for (std::uint32_t j = 0; j < vlen; ++j) raw_lit[j] = static_cast<std::uint8_t>(j);
        std::uint8_t nht[4] = {1, 0, 0, 0};

        const std::uint8_t* fdata[256] = {};
        std::uint32_t flens[256] = {};
        const std::uint32_t* fvars[256] = {};
        std::vector<std::uint32_t> vlens(1, vlen);

        fdata[0xF0] = raw_lit.data(); flens[0xF0] = vlen;
        fdata[0x09] = nht;            flens[0x09] = 4;
        fvars[0xF0] = vlens.data();

        std::uint8_t out[32];
        bool ok = ufsecp::lbtc::sighash_descriptor_hash_batch(
            desc, 5, fdata, flens, fvars, 1, out);
        check(ok, "sighash multi-block var-len=129 computes true");

        // Oracle
        std::vector<std::uint8_t> preimage;
        preimage.insert(preimage.end(), raw_lit.begin(), raw_lit.end());
        preimage.insert(preimage.end(), nht, nht + 4);
        std::uint8_t oracle[32];
        ref_hash256(preimage.data(), preimage.size(), oracle);
        check(std::memcmp(out, oracle, 32) == 0,
              "sighash multi-block var-len=129 bit-exact vs oracle");
    }

    // ─── Adjacent short fixed fields: cross-field partial block ───────
    // nVersion(4) + nLocktime(4) = 8 bytes → one SHA-256 block with
    // proper streaming (no separate padding per field).
    {
        std::uint8_t desc[] = {
            0x00, 0x00,  // nVersion (4)
            0x08, 0x00,  // nLocktime (4)
            0x09, 0x00,  // nHashType (4)
            0xFF
        };

        std::uint8_t nver[4] = {2, 0, 0, 0};
        std::uint8_t nlock[4] = {0x10, 0x27, 0x00, 0x00};
        std::uint8_t nht[4] = {1, 0, 0, 0};

        const std::uint8_t* fdata[256] = {};
        std::uint32_t flens[256] = {};
        fdata[0x00] = nver;  flens[0x00] = 4;
        fdata[0x08] = nlock; flens[0x08] = 4;
        fdata[0x09] = nht;   flens[0x09] = 4;

        std::uint8_t out[32];
        bool ok = ufsecp::lbtc::sighash_descriptor_hash_batch(
            desc, 7, fdata, flens, nullptr, 1, out);
        check(ok, "sighash adjacent short fixed fields computes true");

        // Oracle: nVersion || nLocktime || nHashType
        std::uint8_t preimage[12];
        std::memcpy(preimage,      nver,  4);
        std::memcpy(preimage + 4,  nlock, 4);
        std::memcpy(preimage + 8,  nht,   4);
        std::uint8_t oracle[32];
        ref_hash256(preimage, 12, oracle);
        check(std::memcmp(out, oracle, 32) == 0,
              "sighash adjacent short fields bit-exact vs oracle");
    }

    // ─── Stride < fixed_len rejected (OOB read prevention) ────────────
    // A fixed field's stride MUST be ≥ its fixed serialized length.
    // stride=1 with nVersion (fixed_len=4) would cause a 3-byte
    // out-of-bounds read per row if not rejected early.
    {
        // Descriptor: nVersion(4) + nHashType(4). Set nVersion stride=1.
        std::uint8_t desc[] = {0x00, 0x00, 0x09, 0x00, 0xFF};

        std::uint8_t nver[4] = {2, 0, 0, 0};   // 4-byte data, but stride=1
        std::uint8_t nht[4]  = {1, 0, 0, 0};

        const std::uint8_t* fdata[256] = {};
        std::uint32_t flens[256] = {};
        fdata[0x00] = nver; flens[0x00] = 1;    // Stride 1 < fixed_len=4
        fdata[0x09] = nht;  flens[0x09] = 4;

        std::uint8_t out[32];
        std::memset(out, 0xAB, 32);
        check(!ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc, 5, fdata, flens, nullptr, 1, out),
              "sighash stride<fixed_len (nVersion stride=1<4) rejected");
        // out32 must be untouched on rejection.
        bool untouched = true;
        for (int i = 0; i < 32; ++i)
            if (out[i] != 0xAB) { untouched = false; break; }
        check(untouched, "sighash stride<fixed_len leaves out untouched");
    }

    // ─── Stride == fixed_len is valid ─────────────────────────────────
    // Regression: ensure stride == fixed_len is NOT accidentally rejected.
    {
        std::uint8_t desc[] = {0x00, 0x00, 0x09, 0x00, 0xFF};

        std::uint8_t nver[4] = {2, 0, 0, 0};
        std::uint8_t nht[4]  = {1, 0, 0, 0};

        const std::uint8_t* fdata[256] = {};
        std::uint32_t flens[256] = {};
        fdata[0x00] = nver; flens[0x00] = 4;    // Stride == fixed_len=4
        fdata[0x09] = nht;  flens[0x09] = 4;

        std::uint8_t out[32];
        check(ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc, 5, fdata, flens, nullptr, 1, out),
              "sighash stride==fixed_len (nVersion stride=4==4) computes true");

        // Verify against oracle.
        std::uint8_t preimage[8];
        std::memcpy(preimage,     nver, 4);
        std::memcpy(preimage + 4, nht,  4);
        std::uint8_t oracle[32];
        ref_hash256(preimage, 8, oracle);
        check(std::memcmp(out, oracle, 32) == 0,
              "sighash stride==fixed_len bit-exact vs oracle");
    }

    // ─── stride < fixed_len with backing memory large enough ──────────
    // The backing buffer has enough bytes (4 bytes allocated) but the
    // declared stride is undersized (1). This must be rejected by the
    // host-side parser BEFORE any memory access, not relying on the
    // kernel to catch it.
    {
        std::uint8_t desc[] = {0x00, 0x00, 0x09, 0x00, 0xFF};

        // 4 bytes of valid data — enough to read fixed_len=4 if stride were correct.
        std::uint8_t nver[4] = {2, 0, 0, 0};
        std::uint8_t nht[4]  = {1, 0, 0, 0};

        const std::uint8_t* fdata[256] = {};
        std::uint32_t flens[256] = {};
        fdata[0x00] = nver; flens[0x00] = 1;    // Stride=1 < 4, but 4 bytes allocated
        fdata[0x09] = nht;  flens[0x09] = 4;

        std::uint8_t out[32];
        std::memset(out, 0xAB, 32);
        check(!ufsecp::lbtc::sighash_descriptor_hash_batch(
                  desc, 5, fdata, flens, nullptr, 1, out),
              "sighash stride<fixed_len w/ adequate backing rejected (host-side gate)");
        bool untouched = true;
        for (int i = 0; i < 32; ++i)
            if (out[i] != 0xAB) { untouched = false; break; }
        check(untouched, "sighash stride<fixed_len w/ backing leaves out untouched");
    }


    std::printf("=== %s ===\n", fails == 0 ? "ALL PASS" : "SOME FAILED");
    return fails == 0 ? 0 : 1;
}
