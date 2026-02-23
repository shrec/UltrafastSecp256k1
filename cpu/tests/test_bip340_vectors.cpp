// ============================================================================
// Test: BIP-340 Official Test Vectors (0-14 from bitcoin/bips)
// ============================================================================
// Source: https://github.com/bitcoin/bips/blob/master/bip-0340/test-vectors.csv
// Reference: https://github.com/bitcoin-core/secp256k1/blob/master/src/modules/schnorrsig/tests_impl.h
//
// Vectors 0-3: signing + verification (with known secret key)
// Vectors 4-14: verification only (some expected to fail)
// ============================================================================

#include "secp256k1/schnorr.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/sha256.hpp"

#include <cstdio>
#include <cstring>
#include <array>

using namespace secp256k1;
using fast::Scalar;

static int tests_run = 0;
static int tests_passed = 0;

#define CHECK(cond, msg) do { \
    ++tests_run; \
    if (cond) { ++tests_passed; printf("  [PASS] %s\n", msg); } \
    else { printf("  [FAIL] %s\n", msg); } \
} while(0)

// -- Hex conversion (allocation-free) ----------------------------------------

static void hex_to_bytes(const char* hex, uint8_t* out, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        unsigned v = 0;
        std::sscanf(hex + i * 2, "%02x", &v);
        out[i] = static_cast<uint8_t>(v);
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

// -- Diagnostic hex printer ---------------------------------------------------

static void print_hex(const char* label, const uint8_t* data, size_t len) {
    printf("    %s: ", label);
    for (size_t i = 0; i < len; ++i)
        printf("%02X", data[i]);
    printf("\n");
}

// ============================================================================
// Vectors 0-3: Signing + Verification
// ============================================================================
// Byte arrays transcribed from bitcoin-core/secp256k1 tests_impl.h
// (the canonical reference implementation of these test vectors).

static void test_bip340_sign_vector_0() {
    printf("\n  -- Vector 0 (sk=3, msg=00..00) --\n");

    // From reference: sk = 0x00..03
    auto sk = Scalar::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000003");

    // BIP-340 x-only pubkey for sk=3 (official test-vectors.csv)
    const auto pk = h32(
        "F9308A019258C31049344F85F89D5229"
        "B531C845836F99B08601F113BCE036F9");

    const auto aux = h32(
        "0000000000000000000000000000000000000000000000000000000000000000");
    const auto msg = h32(
        "0000000000000000000000000000000000000000000000000000000000000000");
    const auto expected_sig = h64(
        "E907831F80848D1069A5371B402410364BDF1C5F8307B0084C55F1CE2DCA8215"
        "25F66A4A85EA8B71E482A74F382D2CE5EBEEE8FDB2172F477DF4900D310536C0");

    // 1. Pubkey derivation
    auto derived_pk = schnorr_pubkey(sk);
    if (derived_pk != pk) {
        print_hex("expected pk", pk.data(), 32);
        print_hex("actual   pk", derived_pk.data(), 32);
    }
    CHECK(derived_pk == pk, "V0: pubkey matches expected");

    // 2. Signing
    auto sig = schnorr_sign(sk, msg, aux);
    auto sig_bytes = sig.to_bytes();
    if (sig_bytes != expected_sig) {
        print_hex("expected sig", expected_sig.data(), 64);
        print_hex("actual   sig", sig_bytes.data(), 64);
    }
    CHECK(sig_bytes == expected_sig, "V0: signature matches expected");

    // 3. Verification (using expected sig)
    auto parsed = SchnorrSignature::from_bytes(expected_sig);
    CHECK(schnorr_verify(pk, msg, parsed), "V0: verification passes");

    // 4. Verification (using our computed sig, should always pass)
    CHECK(schnorr_verify(derived_pk, msg, sig), "V0: verify(our_sig) passes");
}

static void test_bip340_sign_vector_1() {
    printf("\n  -- Vector 1 --\n");

    auto sk = Scalar::from_hex(
        "B7E151628AED2A6ABF7158809CF4F3C762E7160F38B4DA56A784D9045190CFEF");
    const auto pk = h32(
        "DFF1D77F2A671C5F36183726DB2341BE58FEAE1DA2DECED843240F7B502BA659");
    const auto aux = h32(
        "0000000000000000000000000000000000000000000000000000000000000001");
    // Message from reference byte array: 0x24,0x3F,...,0xEC,0x4E,0x6C,0x89
    const auto msg = h32(
        "243F6A8885A308D313198A2E03707344A4093822299F31D0082EFA98EC4E6C89");
    const auto expected_sig = h64(
        "6896BD60EEAE296DB48A229FF71DFE071BDE413E6D43F917DC8DCF8C78DE3341"
        "8906D11AC976ABCCB20B091292BFF4EA897EFCB639EA871CFA95F6DE339E4B0A");

    auto derived_pk = schnorr_pubkey(sk);
    if (derived_pk != pk) {
        print_hex("expected pk", pk.data(), 32);
        print_hex("actual   pk", derived_pk.data(), 32);
    }
    CHECK(derived_pk == pk, "V1: pubkey matches expected");

    auto sig = schnorr_sign(sk, msg, aux);
    auto sig_bytes = sig.to_bytes();
    if (sig_bytes != expected_sig) {
        print_hex("expected sig", expected_sig.data(), 64);
        print_hex("actual   sig", sig_bytes.data(), 64);
    }
    CHECK(sig_bytes == expected_sig, "V1: signature matches expected");

    auto parsed = SchnorrSignature::from_bytes(expected_sig);
    CHECK(schnorr_verify(pk, msg, parsed), "V1: verification passes");
    CHECK(schnorr_verify(derived_pk, msg, sig), "V1: verify(our_sig) passes");
}

static void test_bip340_sign_vector_2() {
    printf("\n  -- Vector 2 --\n");

    auto sk = Scalar::from_hex(
        "C90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B14E5C9");
    const auto pk = h32(
        "DD308AFEC5777E13121FA72B9CC1B7CC0139715309B086C960E18FD969774EB8");
    const auto aux = h32(
        "C87AA53824B4D7AE2EB035A2B5BBBCCC080E76CDC6D1692C4B0B62D798E6D906");
    const auto msg = h32(
        "7E2D58D8B3BCDF1ABADEC7829054F90DDA9805AAB56C77333024B9D0A508B75C");
    const auto expected_sig = h64(
        "5831AAEED7B44BB74E5EAB94BA9D4294C49BCF2A60728D8B4C200F50DD313C1B"
        "AB745879A5AD954A72C45A91C3A51D3C7ADEA98D82F8481E0E1E03674A6F3FB7");

    auto derived_pk = schnorr_pubkey(sk);
    if (derived_pk != pk) {
        print_hex("expected pk", pk.data(), 32);
        print_hex("actual   pk", derived_pk.data(), 32);
    }
    CHECK(derived_pk == pk, "V2: pubkey matches expected");

    auto sig = schnorr_sign(sk, msg, aux);
    auto sig_bytes = sig.to_bytes();
    if (sig_bytes != expected_sig) {
        print_hex("expected sig", expected_sig.data(), 64);
        print_hex("actual   sig", sig_bytes.data(), 64);
    }
    CHECK(sig_bytes == expected_sig, "V2: signature matches expected");

    auto parsed = SchnorrSignature::from_bytes(expected_sig);
    CHECK(schnorr_verify(pk, msg, parsed), "V2: verification passes");
    CHECK(schnorr_verify(derived_pk, msg, sig), "V2: verify(our_sig) passes");
}

static void test_bip340_sign_vector_3() {
    printf("\n  -- Vector 3 --\n");

    auto sk = Scalar::from_hex(
        "0B432B2677937381AEF05BB02A66ECD012773062CF3FA2549E44F58ED2401710");
    const auto pk = h32(
        "25D1DFF95105F5253C4022F628A996AD3A0D95FBF21D468A1B33F8C160D8F517");
    const auto aux = h32(
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF");
    const auto msg = h32(
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF");
    const auto expected_sig = h64(
        "7EB0509757E246F19449885651611CB965ECC1A187DD51B64FDA1EDC9637D5EC"
        "97582B9CB13DB3933705B32BA982AF5AF25FD78881EBB32771FC5922EFC66EA3");

    auto derived_pk = schnorr_pubkey(sk);
    if (derived_pk != pk) {
        print_hex("expected pk", pk.data(), 32);
        print_hex("actual   pk", derived_pk.data(), 32);
    }
    CHECK(derived_pk == pk, "V3: pubkey matches expected");

    auto sig = schnorr_sign(sk, msg, aux);
    auto sig_bytes = sig.to_bytes();
    if (sig_bytes != expected_sig) {
        print_hex("expected sig", expected_sig.data(), 64);
        print_hex("actual   sig", sig_bytes.data(), 64);
    }
    CHECK(sig_bytes == expected_sig, "V3: signature matches expected");

    auto parsed = SchnorrSignature::from_bytes(expected_sig);
    CHECK(schnorr_verify(pk, msg, parsed), "V3: verification passes");
    CHECK(schnorr_verify(derived_pk, msg, sig), "V3: verify(our_sig) passes");
}

// ============================================================================
// Vectors 4-14: Verification Only
// ============================================================================
// Hex values from reference byte arrays in bitcoin-core/secp256k1.
// Common message: 243F6A88...EC4E6C89 (first 32 bytes of hex digits of pi)
// Common pubkey (V6-V13): DFF1D77F...502BA659 (V1 pubkey)

static const char* MSG_PI =
    "243F6A8885A308D313198A2E03707344A4093822299F31D0082EFA98EC4E6C89";
static const char* PK_V1 =
    "DFF1D77F2A671C5F36183726DB2341BE58FEAE1DA2DECED843240F7B502BA659";

static void test_bip340_verify_vectors() {
    printf("\n--- BIP-340 Verify-Only Vectors (4-14) ---\n");

    // Vector 4: valid signature
    {
        auto pk  = h32("D69C3509BB99E412E68B0FE8544E72837DFA30746D8BE2AA65975F29D22DC7B9");
        auto msg = h32("4DF3C3F68FCC83B27E9D42C90431A72499F17875C81A599B566C9889B9696703");
        auto sig = SchnorrSignature::from_bytes(h64(
            "00000000000000000000003B78CE563F89A0ED9414F5AA28AD0D96D6795F9C63"
            "76AFB1548AF603B3EB45C9F8207DEE1060CB71C04E80F593060B07D28308D7F4"));
        CHECK(schnorr_verify(pk, msg, sig), "V4: valid sig");
    }

    // Vector 5: public key not on the curve (schnorr_verify should reject)
    {
        auto pk  = h32("EEFDEA4CDB677750A420FEE807EACF21EB9898AE79B9768766E4FAA04A2D4A34");
        auto msg = h32(MSG_PI);
        auto sig = SchnorrSignature::from_bytes(h64(
            "6CFF5C3BA86C69EA4B7376F31A9BCB4F74C1976089B2D9963DA2E5543E177769"
            "69E89B4C5564D00349106B8497785DD7D1D713A8AE82B32FA79D5F7FC407D39B"));
        CHECK(!schnorr_verify(pk, msg, sig), "V5: pk not on curve => reject");
    }

    // Vector 6: has_even_y(R) is false
    {
        auto pk  = h32(PK_V1);
        auto msg = h32(MSG_PI);
        auto sig = SchnorrSignature::from_bytes(h64(
            "FFF97BD5755EEEA420453A14355235D382F6472F8568A18B2F057A1460297556"
            "3CC27944640AC607CD107AE10923D9EF7A73C643E166BE5EBEAFA34B1AC553E2"));
        CHECK(!schnorr_verify(pk, msg, sig), "V6: R has odd Y => reject");
    }

    // Vector 7: negated message
    {
        auto pk  = h32(PK_V1);
        auto msg = h32(MSG_PI);
        auto sig = SchnorrSignature::from_bytes(h64(
            "1FA62E331EDBC21C394792D2AB1100A7B432B013DF3F6FF4F99FCB33E0E1515F"
            "28890B3EDB6E7189B630448B515CE4F8622A954CFE545735AAEA5134FCCDB2BD"));
        CHECK(!schnorr_verify(pk, msg, sig), "V7: negated message => reject");
    }

    // Vector 8: negated s value
    {
        auto pk  = h32(PK_V1);
        auto msg = h32(MSG_PI);
        auto sig = SchnorrSignature::from_bytes(h64(
            "6CFF5C3BA86C69EA4B7376F31A9BCB4F74C1976089B2D9963DA2E5543E177769"
            "961764B3AA9B2FFCB6EF947B6887A226E8D7C93E00C5ED0C1834FF0D0C2E6DA6"));
        CHECK(!schnorr_verify(pk, msg, sig), "V8: negated s => reject");
    }

    // Vector 9: sG - eP is infinite (R at infinity)
    {
        auto pk  = h32(PK_V1);
        auto msg = h32(MSG_PI);
        auto sig = SchnorrSignature::from_bytes(h64(
            "0000000000000000000000000000000000000000000000000000000000000000"
            "123DDA8328AF9C23A94C1FEECFD123BA4FB73476F0D594DCB65C6425BD186051"));
        CHECK(!schnorr_verify(pk, msg, sig), "V9: R at infinity => reject");
    }

    // Vector 10: sG - eP is infinite (x(inf) defined as 1)
    {
        auto pk  = h32(PK_V1);
        auto msg = h32(MSG_PI);
        auto sig = SchnorrSignature::from_bytes(h64(
            "0000000000000000000000000000000000000000000000000000000000000001"
            "7615FBAF5AE28864013C099742DEADB4DBA87F11AC6754F93780D5A1837CF197"));
        CHECK(!schnorr_verify(pk, msg, sig), "V10: R at inf (x=1) => reject");
    }

    // Vector 11: sig[0:32] is not an X coordinate on the curve
    {
        auto pk  = h32(PK_V1);
        auto msg = h32(MSG_PI);
        auto sig = SchnorrSignature::from_bytes(h64(
            "4A298DACAE57395A15D0795DDBFD1DCB564DA82B0F269BC70A74F8220429BA1D"
            "69E89B4C5564D00349106B8497785DD7D1D713A8AE82B32FA79D5F7FC407D39B"));
        CHECK(!schnorr_verify(pk, msg, sig), "V11: R.x not on curve => reject");
    }

    // Vector 12: sig[0:32] is equal to field size p
    {
        auto pk  = h32(PK_V1);
        auto msg = h32(MSG_PI);
        auto sig = SchnorrSignature::from_bytes(h64(
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F"
            "69E89B4C5564D00349106B8497785DD7D1D713A8AE82B32FA79D5F7FC407D39B"));
        CHECK(!schnorr_verify(pk, msg, sig), "V12: R.x == p => reject");
    }

    // Vector 13: sig[32:64] is equal to curve order n
    {
        auto pk  = h32(PK_V1);
        auto msg = h32(MSG_PI);
        auto sig = SchnorrSignature::from_bytes(h64(
            "6CFF5C3BA86C69EA4B7376F31A9BCB4F74C1976089B2D9963DA2E5543E177769"
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141"));
        CHECK(!schnorr_verify(pk, msg, sig), "V13: s == n => reject");
    }

    // Vector 14: public key exceeds field size (>= p)
    {
        auto pk  = h32("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC30");
        auto msg = h32(MSG_PI);
        auto sig = SchnorrSignature::from_bytes(h64(
            "6CFF5C3BA86C69EA4B7376F31A9BCB4F74C1976089B2D9963DA2E5543E177769"
            "69E89B4C5564D00349106B8497785DD7D1D713A8AE82B32FA79D5F7FC407D39B"));
        CHECK(!schnorr_verify(pk, msg, sig), "V14: pk >= p => reject");
    }
}

// -- Entry -------------------------------------------------------------------

int test_bip340_vectors_run() {
    printf("================================================================\n");
    printf("  BIP-340 Official Test Vectors (bitcoin/bips)\n");
    printf("================================================================\n");

    test_bip340_sign_vector_0();
    test_bip340_sign_vector_1();
    test_bip340_sign_vector_2();
    test_bip340_sign_vector_3();
    test_bip340_verify_vectors();

    printf("\n================================================================\n");
    printf("  BIP-340 Results: %d / %d passed\n", tests_passed, tests_run);
    printf("================================================================\n");

    return (tests_passed == tests_run) ? 0 : 1;
}

// Standalone mode
#ifdef STANDALONE_TEST
int main() {
    return test_bip340_vectors_run();
}
#endif
