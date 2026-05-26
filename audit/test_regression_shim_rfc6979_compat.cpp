// ============================================================================
// test_regression_shim_rfc6979_compat.cpp
// ============================================================================
// SHIM-P3-006: rfc6979_nonce_libsecp_compat determinism + signing correctness
//
// Tests the rfc6979_nonce_libsecp_compat function (CPU layer) directly.
// This test builds WITHOUT SECP256K1_SHIM_RFC6979_COMPAT — it exercises the
// underlying nonce function independently of the shim compile flag.
//
// Test cases:
//   RFC-1: same inputs → same nonce (determinism, no-ndata)
//   RFC-2: same inputs with ndata → same nonce (determinism, with-ndata)
//   RFC-3: NULL ndata vs non-NULL ndata → different nonces
//   RFC-4: different ndata → different nonces
//   RFC-5: different message → different nonce
//   RFC-6: different key → different nonce
//   RFC-7: ecdsa_sign_libsecp_compat produces a valid, verifiable signature
//   RFC-8: ecdsa_sign_libsecp_compat_recoverable produces valid sig + recovery
//   RFC-9: NULL ndata path and non-NULL ndata path both produce non-zero nonces
// ============================================================================

#include "secp256k1/ecdsa.hpp"
#include "secp256k1/ct/sign.hpp"
#include "secp256k1/ct/scalar.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/recovery.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>

using secp256k1::fast::Scalar;
using secp256k1::fast::Point;

// ─── helpers ──────────────────────────────────────────────────────────────────

static bool scalars_equal(const Scalar& a, const Scalar& b) {
    auto ab = a.to_bytes();
    auto bb = b.to_bytes();
    return ab == bb;
}

static bool scalar_is_zero(const Scalar& s) {
    return s.is_zero_ct();
}

// ─── test body ────────────────────────────────────────────────────────────────

int test_regression_shim_rfc6979_compat_run() {
    int failures = 0;

#define CHECK(cond, label) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "  FAIL [" label "]: %s\n", #cond); \
            ++failures; \
        } else { \
            std::fprintf(stdout, "  PASS [" label "]\n"); \
        } \
    } while (0)

    std::fprintf(stdout, "=== SHIM-P3-006: rfc6979_nonce_libsecp_compat ===\n");

    // Fixed test vectors
    std::array<uint8_t, 32> key1{};
    key1[31] = 0x01;  // private key = 1

    std::array<uint8_t, 32> key2{};
    key2[31] = 0x02;  // private key = 2

    std::array<uint8_t, 32> msg1{};
    for (int i = 0; i < 32; ++i) msg1[i] = static_cast<uint8_t>(i + 1);

    std::array<uint8_t, 32> msg2{};
    for (int i = 0; i < 32; ++i) msg2[i] = static_cast<uint8_t>(i + 10);

    std::array<uint8_t, 32> ndata1{};
    for (int i = 0; i < 32; ++i) ndata1[i] = static_cast<uint8_t>(0xAB);

    std::array<uint8_t, 32> ndata2{};
    for (int i = 0; i < 32; ++i) ndata2[i] = static_cast<uint8_t>(0xCD);

    Scalar sk1{}, sk2{};
    bool ok1 = Scalar::parse_bytes_strict_nonzero(key1.data(), sk1);
    bool ok2 = Scalar::parse_bytes_strict_nonzero(key2.data(), sk2);
    CHECK(ok1, "RFC-0a: sk1 parse");
    CHECK(ok2, "RFC-0b: sk2 parse");

    // RFC-1: determinism without ndata
    {
        auto k_a = secp256k1::rfc6979_nonce_libsecp_compat(sk1, msg1, nullptr);
        auto k_b = secp256k1::rfc6979_nonce_libsecp_compat(sk1, msg1, nullptr);
        CHECK(!scalar_is_zero(k_a), "RFC-1a: nonce != 0");
        CHECK(scalars_equal(k_a, k_b), "RFC-1: same inputs → same nonce (no ndata)");
    }

    // RFC-2: determinism with ndata
    {
        auto k_a = secp256k1::rfc6979_nonce_libsecp_compat(sk1, msg1, ndata1.data());
        auto k_b = secp256k1::rfc6979_nonce_libsecp_compat(sk1, msg1, ndata1.data());
        CHECK(!scalar_is_zero(k_a), "RFC-2a: nonce != 0 with ndata");
        CHECK(scalars_equal(k_a, k_b), "RFC-2: same inputs → same nonce (with ndata)");
    }

    // RFC-3: NULL ndata vs non-NULL ndata → different nonces
    {
        auto k_no  = secp256k1::rfc6979_nonce_libsecp_compat(sk1, msg1, nullptr);
        auto k_yes = secp256k1::rfc6979_nonce_libsecp_compat(sk1, msg1, ndata1.data());
        CHECK(!scalars_equal(k_no, k_yes), "RFC-3: NULL ndata != non-NULL ndata nonce");
    }

    // RFC-4: different ndata → different nonces
    {
        auto k1_nonce = secp256k1::rfc6979_nonce_libsecp_compat(sk1, msg1, ndata1.data());
        auto k2_nonce = secp256k1::rfc6979_nonce_libsecp_compat(sk1, msg1, ndata2.data());
        CHECK(!scalars_equal(k1_nonce, k2_nonce), "RFC-4: different ndata → different nonces");
    }

    // RFC-5: different message → different nonce
    {
        auto k_m1 = secp256k1::rfc6979_nonce_libsecp_compat(sk1, msg1, nullptr);
        auto k_m2 = secp256k1::rfc6979_nonce_libsecp_compat(sk1, msg2, nullptr);
        CHECK(!scalars_equal(k_m1, k_m2), "RFC-5: different message → different nonce");
    }

    // RFC-6: different key → different nonce
    {
        auto k_s1 = secp256k1::rfc6979_nonce_libsecp_compat(sk1, msg1, nullptr);
        auto k_s2 = secp256k1::rfc6979_nonce_libsecp_compat(sk2, msg1, nullptr);
        CHECK(!scalars_equal(k_s1, k_s2), "RFC-6: different key → different nonce");
    }

    // RFC-7: ecdsa_sign_libsecp_compat produces a valid, verifiable signature (no ndata)
    {
        auto sig = secp256k1::ct::ecdsa_sign_libsecp_compat(msg1, sk1, nullptr);
        CHECK(sig.is_valid(), "RFC-7a: compat sign (no ndata) produces valid sig");

        // Derive pubkey for verification
        auto pubkey = secp256k1::ct::generator_mul(sk1);
        bool verified = secp256k1::ecdsa_verify(msg1, pubkey, sig);
        CHECK(verified, "RFC-7b: compat sig verifies");
    }

    // RFC-7c: ecdsa_sign_libsecp_compat with ndata produces valid, verifiable signature
    {
        auto sig = secp256k1::ct::ecdsa_sign_libsecp_compat(msg1, sk1, ndata1.data());
        CHECK(sig.is_valid(), "RFC-7c: compat sign (with ndata) produces valid sig");

        auto pubkey = secp256k1::ct::generator_mul(sk1);
        bool verified = secp256k1::ecdsa_verify(msg1, pubkey, sig);
        CHECK(verified, "RFC-7d: compat sig with ndata verifies");
    }

    // RFC-8: recoverable compat signing
    {
        auto rsig = secp256k1::ct::ecdsa_sign_libsecp_compat_recoverable(
            msg1, sk1, nullptr);
        CHECK(rsig.sig.is_valid(), "RFC-8a: recoverable compat sig is valid");
        CHECK(rsig.recid >= 0 && rsig.recid <= 3, "RFC-8b: recid in [0,3]");

        // Recover pubkey and verify it matches the expected pubkey
        auto [recovered_pk, rec_ok] = secp256k1::ecdsa_recover(msg1, rsig.sig, rsig.recid);
        CHECK(rec_ok && !recovered_pk.is_infinity(), "RFC-8c: recovery succeeds");

        auto expected_pk = secp256k1::ct::generator_mul(sk1);
        // Compare x-coordinates of the recovered pubkey
        expected_pk.normalize();
        recovered_pk.normalize();
        auto ex = expected_pk.x().to_bytes();
        auto rx = recovered_pk.x().to_bytes();
        CHECK(ex == rx, "RFC-8d: recovered pubkey matches expected");
    }

    // RFC-9: both NULL and non-NULL ndata paths produce non-zero nonces
    {
        // Test with multiple keys
        for (int i = 1; i <= 5; ++i) {
            std::array<uint8_t, 32> key_i{};
            key_i[31] = static_cast<uint8_t>(i);
            Scalar sk_i{};
            if (!Scalar::parse_bytes_strict_nonzero(key_i.data(), sk_i)) continue;

            auto k_no   = secp256k1::rfc6979_nonce_libsecp_compat(sk_i, msg1, nullptr);
            auto k_with = secp256k1::rfc6979_nonce_libsecp_compat(sk_i, msg1, ndata1.data());
            CHECK(!scalar_is_zero(k_no),   "RFC-9a: no-ndata nonce is non-zero");
            CHECK(!scalar_is_zero(k_with), "RFC-9b: with-ndata nonce is non-zero");
        }
    }

    if (failures == 0) {
        std::fprintf(stdout, "PASS: all %d checks passed\n",
            9 + 5 * 2); // approximate check count
    } else {
        std::fprintf(stderr, "FAIL: %d check(s) failed\n", failures);
    }

#undef CHECK
    return failures == 0 ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() {
    return test_regression_shim_rfc6979_compat_run();
}
#endif
