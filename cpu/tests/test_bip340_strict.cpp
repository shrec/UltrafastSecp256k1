// ============================================================================
// Test: BIP-340 Strict Encoding Checks
// ============================================================================
// Validates that non-canonical encodings are rejected per BIP-340:
//   - r >= p must be rejected (not reduced)
//   - s >= n must be rejected (not reduced)
//   - s == 0 must be rejected
//   - pubkey x >= p must be rejected (not reduced)
//   - Values that would reduce to valid inputs must still be rejected
//
// Reference: https://github.com/bitcoin/bips/blob/master/bip-0340.mediawiki
//   "The function bytes(x), where x is an integer, returns the 32-byte
//    encoding of x, most significant byte first."
//   "Fail if r >= p"
//   "Fail if s >= n"
// ============================================================================

#include "secp256k1/schnorr.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/field.hpp"

#include <cstdio>
#include <cstring>
#include <array>

using namespace secp256k1;
using fast::Scalar;
using fast::FieldElement;

static int tests_run = 0;
static int tests_passed = 0;

#define CHECK(cond, msg) do { \
    ++tests_run; \
    if (cond) { ++tests_passed; printf("  [PASS] %s\n", msg); } \
    else { printf("  [FAIL] %s\n", msg); } \
} while(0)

// -- Hex helpers (allocation-free) --------------------------------------------

static void hex_to_bytes(const char* hex, uint8_t* out, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        unsigned hi = 0, lo = 0;
        char const c0 = hex[i * 2], c1 = hex[i * 2 + 1];
        if (c0 >= '0' && c0 <= '9') {
            hi = static_cast<unsigned>(c0 - '0');
        } else if (c0 >= 'a' && c0 <= 'f') {
            hi = static_cast<unsigned>(c0 - 'a' + 10);
        } else if (c0 >= 'A' && c0 <= 'F') {
            hi = static_cast<unsigned>(c0 - 'A' + 10);
        }
        if (c1 >= '0' && c1 <= '9') {
            lo = static_cast<unsigned>(c1 - '0');
        } else if (c1 >= 'a' && c1 <= 'f') {
            lo = static_cast<unsigned>(c1 - 'a' + 10);
        } else if (c1 >= 'A' && c1 <= 'F') {
            lo = static_cast<unsigned>(c1 - 'A' + 10);
        }
        out[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
}

static std::array<uint8_t, 32> h32(const char* hex) {
    std::array<uint8_t, 32> r{};
    hex_to_bytes(hex, r.data(), 32);
    return r;
}

static std::array<uint8_t, 64> h64(const char* hex) {
    std::array<uint8_t, 64> r{};
    hex_to_bytes(hex, r.data(), 64);
    return r;
}

// ============================================================================
// secp256k1 constants (hex)
//   p = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
//   n = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
// ============================================================================

// -- Scalar strict parsing tests ----------------------------------------------

static void test_scalar_strict() {
    printf("\n  -- Scalar::parse_bytes_strict --\n");

    // Valid: s = 1 (canonical)
    {
        auto bytes = h32("0000000000000000000000000000000000000000000000000000000000000001");
        Scalar out;
        CHECK(Scalar::parse_bytes_strict(bytes, out), "s=1 accepted");
        CHECK(!out.is_zero(), "s=1 is nonzero");
    }

    // Valid: s = n-1 (max valid scalar)
    {
        auto bytes = h32("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140");
        Scalar out;
        CHECK(Scalar::parse_bytes_strict(bytes, out), "s=n-1 accepted");
    }

    // Invalid: s = n (must reject, not reduce to 0)
    {
        auto bytes = h32("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
        Scalar out;
        CHECK(!Scalar::parse_bytes_strict(bytes, out), "s=n rejected (not reduced)");
    }

    // Invalid: s = n+1 (must reject, not reduce to 1)
    {
        auto bytes = h32("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364142");
        Scalar out;
        CHECK(!Scalar::parse_bytes_strict(bytes, out), "s=n+1 rejected (not reduced)");
    }

    // Invalid: s = 2^256 - 1 (max 256-bit value, must reject)
    {
        auto bytes = h32("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF");
        Scalar out;
        CHECK(!Scalar::parse_bytes_strict(bytes, out), "s=2^256-1 rejected");
    }

    // Valid: s = 0 (parse_bytes_strict allows zero; nonzero variant rejects)
    {
        auto bytes = h32("0000000000000000000000000000000000000000000000000000000000000000");
        Scalar out;
        CHECK(Scalar::parse_bytes_strict(bytes, out), "s=0: strict accepts");
        CHECK(!Scalar::parse_bytes_strict_nonzero(bytes, out), "s=0: strict_nonzero rejects");
    }

    // parse_bytes_strict_nonzero: n rejected
    {
        auto bytes = h32("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
        Scalar out;
        CHECK(!Scalar::parse_bytes_strict_nonzero(bytes, out), "s=n: strict_nonzero rejects");
    }

    // parse_bytes_strict_nonzero: 1 accepted
    {
        auto bytes = h32("0000000000000000000000000000000000000000000000000000000000000001");
        Scalar out;
        CHECK(Scalar::parse_bytes_strict_nonzero(bytes, out), "s=1: strict_nonzero accepts");
    }
}

// -- FieldElement strict parsing tests ----------------------------------------

static void test_field_strict() {
    printf("\n  -- FieldElement::parse_bytes_strict --\n");

    // Valid: x = 1
    {
        auto bytes = h32("0000000000000000000000000000000000000000000000000000000000000001");
        FieldElement out;
        CHECK(FieldElement::parse_bytes_strict(bytes, out), "x=1 accepted");
    }

    // Valid: x = p-1 (max valid field element)
    {
        auto bytes = h32("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2E");
        FieldElement out;
        CHECK(FieldElement::parse_bytes_strict(bytes, out), "x=p-1 accepted");
    }

    // Invalid: x = p (must reject, not reduce to 0)
    {
        auto bytes = h32("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
        FieldElement out;
        CHECK(!FieldElement::parse_bytes_strict(bytes, out), "x=p rejected (not reduced)");
    }

    // Invalid: x = p+1 (must reject, not reduce to 1)
    {
        auto bytes = h32("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC30");
        FieldElement out;
        CHECK(!FieldElement::parse_bytes_strict(bytes, out), "x=p+1 rejected (not reduced)");
    }

    // Invalid: x = 2^256 - 1
    {
        auto bytes = h32("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF");
        FieldElement out;
        CHECK(!FieldElement::parse_bytes_strict(bytes, out), "x=2^256-1 rejected");
    }

    // Valid: x = 0
    {
        auto bytes = h32("0000000000000000000000000000000000000000000000000000000000000000");
        FieldElement out;
        CHECK(FieldElement::parse_bytes_strict(bytes, out), "x=0 accepted");
    }
}

// -- SchnorrSignature strict parsing tests ------------------------------------

static void test_schnorr_sig_strict() {
    printf("\n  -- SchnorrSignature::parse_strict --\n");

    // Valid canonical signature (from BIP-340 vector 0)
    {
        auto sig_bytes = h64(
            "E907831F80848D1069A5371B402410364BDF1C5F8307B0084C55F1CE2DCA8215"
            "25F66A4A85EA8B71E482A74F382D2CE5EBEEE8FDB2172F477DF4900D310536C0");
        SchnorrSignature sig;
        CHECK(SchnorrSignature::parse_strict(sig_bytes, sig), "Valid BIP-340 sig accepted");
    }

    // Invalid: r = p (BIP-340 vector 12)
    {
        auto sig_bytes = h64(
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F"
            "69E89B4C5564D00349106B8497785DD7D1D713A8AE82B32FA79D5F7FC407D39B");
        SchnorrSignature sig;
        CHECK(!SchnorrSignature::parse_strict(sig_bytes, sig), "r=p rejected by strict parse");
    }

    // Invalid: r = p+1
    {
        auto sig_bytes = h64(
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC30"
            "69E89B4C5564D00349106B8497785DD7D1D713A8AE82B32FA79D5F7FC407D39B");
        SchnorrSignature sig;
        CHECK(!SchnorrSignature::parse_strict(sig_bytes, sig), "r=p+1 rejected by strict parse");
    }

    // Invalid: s = n (BIP-340 vector 13)
    {
        auto sig_bytes = h64(
            "6CFF5C3BA86C69EA4B7376F31A9BCB4F74C1976089B2D9963DA2E5543E177769"
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
        SchnorrSignature sig;
        CHECK(!SchnorrSignature::parse_strict(sig_bytes, sig), "s=n rejected by strict parse");
    }

    // Invalid: s = n+1
    {
        auto sig_bytes = h64(
            "6CFF5C3BA86C69EA4B7376F31A9BCB4F74C1976089B2D9963DA2E5543E177769"
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364142");
        SchnorrSignature sig;
        CHECK(!SchnorrSignature::parse_strict(sig_bytes, sig), "s=n+1 rejected by strict parse");
    }

    // Invalid: s = 0
    {
        auto sig_bytes = h64(
            "6CFF5C3BA86C69EA4B7376F31A9BCB4F74C1976089B2D9963DA2E5543E177769"
            "0000000000000000000000000000000000000000000000000000000000000000");
        SchnorrSignature sig;
        CHECK(!SchnorrSignature::parse_strict(sig_bytes, sig), "s=0 rejected by strict parse");
    }

    // Invalid: s = 2^256 - 1
    {
        auto sig_bytes = h64(
            "6CFF5C3BA86C69EA4B7376F31A9BCB4F74C1976089B2D9963DA2E5543E177769"
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF");
        SchnorrSignature sig;
        CHECK(!SchnorrSignature::parse_strict(sig_bytes, sig), "s=2^256-1 rejected by strict parse");
    }
}

// -- Schnorr verify with non-canonical inputs ---------------------------------

static void test_schnorr_verify_strict() {
    printf("\n  -- schnorr_verify with non-canonical inputs --\n");

    // Use BIP-340 vector 1 pubkey for tests
    auto pk = h32("DFF1D77F2A671C5F36183726DB2341BE58FEAE1DA2DECED843240F7B502BA659");
    auto msg = h32("243F6A8885A308D313198A2E03707344A4093822299F31D0082EFA98EC4E6C89");

    // BIP-340 V12: r == p => must fail
    {
        auto sig_arr = h64(
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F"
            "69E89B4C5564D00349106B8497785DD7D1D713A8AE82B32FA79D5F7FC407D39B");
        auto sig = SchnorrSignature::from_bytes(sig_arr);
        CHECK(!schnorr_verify(pk, msg, sig), "V12: r==p rejected by schnorr_verify");
    }

    // BIP-340 V13: s == n => must fail
    {
        auto sig_arr = h64(
            "6CFF5C3BA86C69EA4B7376F31A9BCB4F74C1976089B2D9963DA2E5543E177769"
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
        auto sig = SchnorrSignature::from_bytes(sig_arr);
        CHECK(!schnorr_verify(pk, msg, sig), "V13: s==n rejected by schnorr_verify");
    }

    // BIP-340 V14: pk.x >= p => must fail
    {
        auto bad_pk = h32("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC30");
        auto sig_arr = h64(
            "6CFF5C3BA86C69EA4B7376F31A9BCB4F74C1976089B2D9963DA2E5543E177769"
            "69E89B4C5564D00349106B8497785DD7D1D713A8AE82B32FA79D5F7FC407D39B");
        auto sig = SchnorrSignature::from_bytes(sig_arr);
        CHECK(!schnorr_verify(bad_pk, msg, sig), "V14: pk>=p rejected by schnorr_verify");
    }

    // Extra: s = n+1 (reduces to 1 with from_bytes, but schnorr_verify
    // does NOT rely on parse_strict for callers using from_bytes --
    // the s!=0 check passes (reduced to 1), but the math won't match
    // the intended signature. This is different from strict parse.)
    // The key point: parse_strict would reject at parse time.
    {
        auto sig_bytes = h64(
            "E907831F80848D1069A5371B402410364BDF1C5F8307B0084C55F1CE2DCA8215"
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364142");
        SchnorrSignature sig;
        CHECK(!SchnorrSignature::parse_strict(sig_bytes, sig),
              "s=n+1: parse_strict rejects at parse time");
    }
}

// -- X-only pubkey parse with non-canonical inputs ----------------------------

static void test_xonly_pubkey_strict() {
    printf("\n  -- schnorr_xonly_pubkey_parse with non-canonical x --\n");

    // Valid pubkey (BIP-340 vector 0 pk)
    {
        auto pk_bytes = h32("F9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9");
        SchnorrXonlyPubkey pub{};
        CHECK(schnorr_xonly_pubkey_parse(pub, pk_bytes), "Valid pk accepted");
    }

    // Invalid: x = p (must reject)
    {
        auto pk_bytes = h32("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
        SchnorrXonlyPubkey pub{};
        CHECK(!schnorr_xonly_pubkey_parse(pub, pk_bytes), "x=p rejected");
    }

    // Invalid: x = p+1 (must reject, not reduce to 1)
    {
        auto pk_bytes = h32("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC30");
        SchnorrXonlyPubkey pub{};
        CHECK(!schnorr_xonly_pubkey_parse(pub, pk_bytes), "x=p+1 rejected (not reduced)");
    }

    // Invalid: x = 2^256 - 1 (must reject)
    {
        auto pk_bytes = h32("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF");
        SchnorrXonlyPubkey pub{};
        CHECK(!schnorr_xonly_pubkey_parse(pub, pk_bytes), "x=2^256-1 rejected");
    }
}

// -- Entry point --------------------------------------------------------------

int test_bip340_strict_run() {
    printf("================================================================\n");
    printf("  BIP-340 Strict Encoding Tests (non-canonical rejection)\n");
    printf("================================================================\n");

    test_scalar_strict();
    test_field_strict();
    test_schnorr_sig_strict();
    test_schnorr_verify_strict();
    test_xonly_pubkey_strict();

    printf("\n================================================================\n");
    printf("  BIP-340 Strict Results: %d / %d passed\n", tests_passed, tests_run);
    printf("================================================================\n");

    return (tests_passed == tests_run) ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() {
    return test_bip340_strict_run();
}
#endif
