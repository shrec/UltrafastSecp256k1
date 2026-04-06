// ============================================================================
// Wycheproof ECDSA secp256k1 SHA-256 Extended Coverage
// ============================================================================
// Source: ecdsa_secp256k1_sha256_test.json, Version v1 (474 tests)
//         https://github.com/google/wycheproof (Apache 2.0 / CC0)
//
// This file covers Wycheproof flag categories NOT in the base file
// test_wycheproof_ecdsa_secp256k1_sha256.cpp.
//
// Categories:
//   A  RangeCheck (tcIds 152-162, 6 vectors)
//      Signatures where r is out of range (r ≥ n or negative DER encoding);
//      all must be rejected.  Modifications tested:
//        tcId 152: r = r_valid + n   (33-byte DER INTEGER, leading 0x01)
//        tcId 153: r modified (32-byte but >= n after scalar parse)
//        tcId 154: r = r_valid + 2n  (34-byte DER INTEGER, leading 0x0100)
//        tcId 160: r = s_valid + n   (33-byte DER INTEGER, leading 0x01)
//        tcId 161: r = s_valid + 2n  (high byte 0xFF = negative DER)
//        tcId 162: r = s_valid + 2n  (34-byte DER INTEGER, leading 0x0100)
//
//   B  InvalidTypesInSignature (representative subset, tcIds 232-258, 27 vectors)
//      r or s use the wrong ASN.1 type tag: REAL(0x09), BOOLEAN(0x01),
//      NULL(0x05), UTF8String(0x0c), SEQUENCE(0x30). Must be rejected.
//      Tests: 9 vectors with r=0, 9 with r=1, 9 with r=-1 (negative → also rejected).
//
//   C  EdgeCaseShamirMultiplication (tcId 295, 1 vector)
//      Valid signature exercising a Shamir-trick arithmetic edge path.
//
//   D  SmallRandS / CVE-2020-13895 (tcIds 355-360, 6 vectors)
//      Valid signatures with r, s ∈ {1, 2, 3}.
//      Some implementations incorrectly reject these — all must ACCEPT.
//
//   E  PointDuplication (tcIds 390, 427-428, 442-445, 7 vectors)
//      u1*G + u2*Q produces an edge-case intermediate value in verify.
//      Mixed valid/invalid.
//
//   F  EdgeCasePublicKey (tcIds 446-463, 18 vectors)
//      Valid signatures with extreme public key coordinates:
//        - wy small (leading zero bytes)
//        - wy near the field prime p
//        - wx small (leading zero bytes)
//        - wx/wy with many trailing 1-bits
//        - wx with many trailing 0-bits
//
// Total: 65 test vectors (6+27+1+6+7+18). All must pass.
//
// All tcId numbers reference ecdsa_secp256k1_sha256_test.json.
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>

#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/sha256.hpp"

using namespace secp256k1;
using fast::Scalar;
using fast::Point;
using fast::FieldElement;

static int g_pass = 0, g_fail = 0;
static const char* g_section = "";

#include "audit_check.hpp"

// ── Hex helpers ──────────────────────────────────────────────────────────────

static uint8_t hex_nibble(char c) {
    if (c >= '0' && c <= '9') return static_cast<uint8_t>(c - '0');
    if (c >= 'a' && c <= 'f') return static_cast<uint8_t>(c - 'a' + 10);
    if (c >= 'A' && c <= 'F') return static_cast<uint8_t>(c - 'A' + 10);
    return 0;
}

static std::array<uint8_t, 32> hex32(const char* h) {
    std::array<uint8_t, 32> out{};
    for (size_t i = 0; i < 32; ++i)
        out[i] = static_cast<uint8_t>(
            (hex_nibble(h[2*i]) << 4) | hex_nibble(h[2*i+1]));
    return out;
}

static size_t hex_decode(const char* hex, uint8_t* out, size_t max_out) {
    size_t hex_len = strlen(hex);
    if (hex_len % 2 != 0) return 0;
    size_t n = hex_len / 2;
    if (n > max_out) return 0;
    for (size_t i = 0; i < n; ++i)
        out[i] = static_cast<uint8_t>(
            (hex_nibble(hex[2*i]) << 4) | hex_nibble(hex[2*i+1]));
    return n;
}

// ── Strict DER ECDSA parser ───────────────────────────────────────────────────
// Accepts only canonical DER: 0x30 <short-form len> 0x02 <rlen> <r> 0x02 <slen> <s>
// Rejects all BER extensions, negative integers, integers > 32 bytes, trailing bytes.
static bool der_parse_strict(const uint8_t* der, size_t len,
                              std::array<uint8_t,32>& r_out,
                              std::array<uint8_t,32>& s_out) {
    if (!der || len < 8) return false;
    size_t off = 0;
    if (der[off++] != 0x30) return false;
    if (off >= len) return false;
    if (der[off] >= 0x80) return false;   // reject BER long-form length
    size_t seq_len = der[off++];
    if (off + seq_len != len) return false;

    auto read_int = [&](std::array<uint8_t,32>& out) -> bool {
        if (off >= len || der[off] != 0x02) return false;
        off++;
        if (off >= len) return false;
        if (der[off] >= 0x80) return false; // reject BER long-form
        size_t int_len = der[off++];
        if (int_len == 0 || off + int_len > len) return false;
        const uint8_t* p = der + off;
        size_t n = int_len;
        // Reject non-minimal leading zero (second byte < 0x80 implies no need for 0x00)
        if (n > 1 && p[0] == 0x00 && p[1] < 0x80) return false;
        // Reject negative integer (high bit set without leading 0x00)
        if (n > 0 && p[0] >= 0x80) return false;
        // Strip canonical leading 0x00
        if (n > 0 && p[0] == 0x00) { p++; n--; }
        if (n > 32) return false;
        out.fill(0);
        if (n > 0) std::memcpy(out.data() + 32 - n, p, n);
        off += int_len;
        return true;
    };

    if (!read_int(r_out)) return false;
    if (!read_int(s_out)) return false;
    if (off != len) return false; // reject trailing bytes
    return true;
}

// ── Public key construction ──────────────────────────────────────────────────

// hex32 version: both wx_hex and wy_hex must be exactly 64 hex chars (32 bytes)
static Point make_pubkey(const char* wx_hex, const char* wy_hex) {
    auto x = FieldElement::from_bytes(hex32(wx_hex));
    auto y = FieldElement::from_bytes(hex32(wy_hex));
    return Point::from_affine(x, y);
}

// Padded version: wx_hex and wy_hex may be shorter than 64 hex chars;
// they are zero-padded on the left (big-endian most-significant bytes = 0).
static Point make_pubkey_padded(const char* wx_hex, const char* wy_hex) {
    char wx64[65] = {};
    char wy64[65] = {};
    size_t wx_len = strlen(wx_hex);
    size_t wy_len = strlen(wy_hex);
    if (wx_len <= 64) {
        size_t pad = 64 - wx_len;
        std::memset(wx64, '0', pad);
        std::memcpy(wx64 + pad, wx_hex, wx_len);
    }
    if (wy_len <= 64) {
        size_t pad = 64 - wy_len;
        std::memset(wy64, '0', pad);
        std::memcpy(wy64 + pad, wy_hex, wy_len);
    }
    return make_pubkey(wx64, wy64);
}

// ── Scalar range check ────────────────────────────────────────────────────────

static bool parse_scalar_strict(const std::array<uint8_t,32>& bytes, Scalar& out) {
    return Scalar::parse_bytes_strict_nonzero(bytes, out);
}

// ── Full DER-verify: parse + scalar checks + ECDSA verify ────────────────────

static bool der_verify(const char* sig_hex,
                       const std::array<uint8_t,32>& hash,
                       const Point& pk) {
    uint8_t buf[300] = {};
    size_t len = (strlen(sig_hex) == 0) ? 0
                                        : hex_decode(sig_hex, buf, sizeof(buf));
    if (len == 0 && strlen(sig_hex) != 0) return false;

    std::array<uint8_t,32> r_b{}, s_b{};
    if (!der_parse_strict(buf, len, r_b, s_b)) return false;

    Scalar r_s{}, s_s{};
    if (!parse_scalar_strict(r_b, r_s)) return false;
    if (!parse_scalar_strict(s_b, s_s)) return false;

    ECDSASignature sig{ r_s, s_s };
    return ecdsa_verify(hash.data(), pk, sig);
}

// ── Message hash helpers ──────────────────────────────────────────────────────

// msg = "313233343030" = hex bytes 0x31 0x32 0x33 0x34 0x30 0x30 = ASCII "123400"
static std::array<uint8_t,32> hash_std() {
    uint8_t msg[] = { 0x31, 0x32, 0x33, 0x34, 0x30, 0x30 };
    return SHA256::hash(msg, sizeof(msg));
}

// hex-encoded message → SHA-256
static std::array<uint8_t,32> hash_from_hex(const char* msg_hex) {
    uint8_t buf[512];
    size_t n = hex_decode(msg_hex, buf, sizeof(buf));
    return SHA256::hash(buf, n);
}

// ── Group 2 pubkey (tcIds 152-295) ──────────────────────────────────────────
static Point pk_group2() {
    return make_pubkey(
        "b838ff44e5bc177bf21189d0766082fc9d843226887fc9760371100b7ee20a6f",
        "f0c9d75bfba7b31a6bca1974496eeb56de357071955d83c4b1badaa0b21832e9");
}

// ============================================================================
// A — RangeCheck (tcIds 152-162)
// "r replaced by r + n" — r exceeds group order n; must be rejected.
// All share group2 pubkey and hash_std().
// ============================================================================
static void test_range_check() {
    g_section = "RangeCheck";
    std::printf("  [A] RangeCheck — r = r_valid + n (tcIds 152,153,154,160,161,162)\n");
    auto pk = pk_group2();
    auto h  = hash_std();

    // tcId 152: r has leading 0x01 byte (33-byte DER integer, value = r_valid + n)
    CHECK(!der_verify(
        "3045022101813ef79ccefa9a56f7ba805f0e478583b90deabca4b05c4574e49b5899b964a6"
        "02206ff18a52dcc0336f7af62400a6dd9b810732baf1ff758000d6f613a556eb31ba",
        h, pk), "tcId 152: r+n rejected");

    // tcId 153: r reduced mod 2^256 (still >= n)
    CHECK(!der_verify(
        "30440220813ef79ccefa9a56f7ba805f0e47858643b030ef461f1bcdf53fde3ef94ce224"
        "02206ff18a52dcc0336f7af62400a6dd9b810732baf1ff758000d6f613a556eb31ba",
        h, pk), "tcId 153: r modified rejected");

    // tcId 154: r = r_valid + 2n (leads to 34-byte DER with 0x0100 prefix)
    CHECK(!der_verify(
        "304602220100813ef79ccefa9a56f7ba805f0e47843fad3bf4853e07f7c98770c99bffc46465"
        "02206ff18a52dcc0336f7af62400a6dd9b810732baf1ff758000d6f613a556eb31ba",
        h, pk), "tcId 154: r+2n rejected");

    // tcId 160: r = (s_value + n) — 33-byte INTEGER with leading 0x01
    CHECK(!der_verify(
        "30450221016ff18a52dcc0336f7af62400a6dd9b7fc1e197d8aebe203c96c87232272172fb"
        "02206ff18a52dcc0336f7af62400a6dd9b810732baf1ff758000d6f613a556eb31ba",
        h, pk), "tcId 160: r out of range (33B) rejected");

    // tcId 161: r = (s_value + 2n) wraps to 0xFF high byte — DER parser rejects negative INTEGER
    CHECK(!der_verify(
        "30450221ff6ff18a52dcc0336f7af62400a6dd9b824c83de0b502cdfc51723b51886b4f079"
        "02206ff18a52dcc0336f7af62400a6dd9b810732baf1ff758000d6f613a556eb31ba",
        h, pk), "tcId 161: r negative-encoded rejected");

    // tcId 162: r = (s_value + 2n) — 34-byte INTEGER with leading 0x0100 prefix
    CHECK(!der_verify(
        "3046022201006ff18a52dcc0336f7af62400a6dd9a3bb60fa1a14815bbc0a954a0758d2c72ba"
        "02206ff18a52dcc0336f7af62400a6dd9b810732baf1ff758000d6f613a556eb31ba",
        h, pk), "tcId 162: r out of range (34B) rejected");
}

// ============================================================================
// B — InvalidTypesInSignature (representative subset, tcIds 232-258)
// r or s element uses wrong DER type tag.  All must be rejected by the parser.
// All share group2 pubkey and hash_std().
// ============================================================================
static void test_invalid_type_signatures() {
    g_section = "InvalidTypesInSignature";
    std::printf("  [B] InvalidTypesInSignature — wrong ASN.1 type tags (tcIds 232-258, subset)\n");
    auto pk = pk_group2();
    auto h  = hash_std();

    // ── r = INTEGER(0), s = wrong type ──────────────────────────────────────
    // tcId 232: s = REAL(0.25) [tag 0x09]
    CHECK(!der_verify("3008020100090380fe01", h, pk), "tcId 232: s=REAL(0.25)");
    // tcId 233: s = REAL(NaN) [tag 0x09]
    CHECK(!der_verify("3006020100090142",     h, pk), "tcId 233: s=REAL(NaN)");
    // tcId 234: s = BOOLEAN(TRUE) [tag 0x01]
    CHECK(!der_verify("3006020100010101",     h, pk), "tcId 234: s=BOOL(true)");
    // tcId 235: s = BOOLEAN(FALSE) [tag 0x01]
    CHECK(!der_verify("3006020100010100",     h, pk), "tcId 235: s=BOOL(false)");
    // tcId 236: s = NULL [tag 0x05]
    CHECK(!der_verify("30050201000500",       h, pk), "tcId 236: s=NULL");
    // tcId 237: s = UTF8String("") [tag 0x0c]
    CHECK(!der_verify("30050201000c00",       h, pk), "tcId 237: s=UTF8\"\"");
    // tcId 238: s = UTF8String("0") [tag 0x0c]
    CHECK(!der_verify("30060201000c0130",     h, pk), "tcId 238: s=UTF8\"0\"");
    // tcId 239: s = SEQUENCE{} [tag 0x30]
    CHECK(!der_verify("30050201003000",       h, pk), "tcId 239: s=SEQ{}");
    // tcId 240: s = SEQUENCE{INTEGER(0)} [tag 0x30]
    CHECK(!der_verify("30080201003003020100", h, pk), "tcId 240: s=SEQ{INT 0}");

    // ── r = INTEGER(1), s = wrong type ──────────────────────────────────────
    // tcId 241: s = REAL(0.25)
    CHECK(!der_verify("3008020101090380fe01", h, pk), "tcId 241: r=1 s=REAL");
    // tcId 242: s = REAL(NaN)
    CHECK(!der_verify("3006020101090142",     h, pk), "tcId 242: r=1 s=NaN");
    // tcId 243: s = BOOLEAN(TRUE)
    CHECK(!der_verify("3006020101010101",     h, pk), "tcId 243: r=1 s=BOOL");
    // tcId 244: s = BOOLEAN(FALSE)
    CHECK(!der_verify("3006020101010100",     h, pk), "tcId 244: r=1 s=BOOL(F)");
    // tcId 245: s = NULL
    CHECK(!der_verify("30050201010500",       h, pk), "tcId 245: r=1 s=NULL");
    // tcId 246: s = UTF8String("")
    CHECK(!der_verify("30050201010c00",       h, pk), "tcId 246: r=1 s=UTF8\"\"");
    // tcId 247: s = UTF8String("0")
    CHECK(!der_verify("30060201010c0130",     h, pk), "tcId 247: r=1 s=UTF8\"0\"");
    // tcId 248: s = SEQUENCE{}
    CHECK(!der_verify("30050201013000",       h, pk), "tcId 248: r=1 s=SEQ{}");
    // tcId 249: s = SEQUENCE{INTEGER(0)}
    CHECK(!der_verify("30080201013003020100", h, pk), "tcId 249: r=1 s=SEQ{0}");

    // ── r = INTEGER(-1) [negative = invalid by strict DER], s = wrong type ──
    // tcId 250: r = -1, s = REAL(0.25)
    CHECK(!der_verify("30080201ff090380fe01", h, pk), "tcId 250: r=-1 s=REAL");
    // tcId 251: r = -1, s = REAL(NaN)
    CHECK(!der_verify("30060201ff090142",     h, pk), "tcId 251: r=-1 s=NaN");
    // tcId 252: r = -1, s = BOOLEAN(TRUE)
    CHECK(!der_verify("30060201ff010101",     h, pk), "tcId 252: r=-1 s=BOOL");
    // tcId 253: r = -1, s = BOOLEAN(FALSE)
    CHECK(!der_verify("30060201ff010100",     h, pk), "tcId 253: r=-1 s=BOOL(F)");
    // tcId 254: r = -1, s = NULL
    CHECK(!der_verify("30050201ff0500",       h, pk), "tcId 254: r=-1 s=NULL");
    // tcId 255: r = -1, s = UTF8String("")
    CHECK(!der_verify("30050201ff0c00",       h, pk), "tcId 255: r=-1 s=UTF8\"\"");
    // tcId 256: r = -1, s = UTF8String("0")
    CHECK(!der_verify("30060201ff0c0130",     h, pk), "tcId 256: r=-1 s=UTF8\"0\"");
    // tcId 257: r = -1, s = SEQUENCE{}
    CHECK(!der_verify("30050201ff3000",       h, pk), "tcId 257: r=-1 s=SEQ{}");
    // tcId 258: r = -1, s = SEQUENCE{INTEGER(0)}
    CHECK(!der_verify("30080201ff3003020100", h, pk), "tcId 258: r=-1 s=SEQ{0}");
}

// ============================================================================
// C — EdgeCaseShamirMultiplication (tcId 295)
// Valid signature exercising an edge case in the Shamir simultaneous
// multiplication code path. Uses group2 pubkey but a different message.
// msg = "3235353835" (hex literal) = ASCII "25585"
// ============================================================================
static void test_shamir_edge_case() {
    g_section = "EdgeCaseShamirMultiplication";
    std::printf("  [C] EdgeCaseShamirMultiplication (tcId 295)\n");

    auto pk = pk_group2();
    // msg_hex "3235353835" → SHA-256 → hash
    auto h = hash_from_hex("3235353835");

    // tcId 295: valid
    CHECK(der_verify(
        "3045022100dd1b7d09a7bd8218961034a39a87fecf5314f00c4d25eb58a07ac85e85eab516"
        "022035138c401ef8d3493d65c9002fe62b43aee568731b744548358996d9cc427e06",
        h, pk), "tcId 295: Shamir edge case valid");
}

// ============================================================================
// D — SmallRandS / CVE-2020-13895 (tcIds 355-360)
// Valid signatures with r, s ∈ {1, 2, 3}.
// Affected libraries incorrectly rejected them due to modular arithmetic bugs
// or inappropriate range checks.  All must ACCEPT.
// Each test case uses a distinct public key crafted so that the signature
// equation holds with the given tiny r and s values.
// ============================================================================
static void test_small_r_and_s() {
    g_section = "SmallRandS";
    std::printf("  [D] SmallRandS / CVE-2020-13895 (tcIds 355-360)\n");
    auto h = hash_std();

    // tcId 355: r = 1, s = 1
    {
        auto pk = make_pubkey(
            "1877045be25d34a1d0600f9d5c00d0645a2a54379b6ceefad2e6bf5c2a3352ce",
            "821a532cc1751ee1d36d41c3d6ab4e9b143e44ec46d73478ea6a79a5c0e54159");
        CHECK(der_verify("3006020101020101", h, pk), "tcId 355: r=1 s=1 valid");
    }
    // tcId 356: r = 1, s = 2
    {
        auto pk = make_pubkey(
            "455439fcc3d2deeceddeaece60e7bd17304f36ebb602adf5a22e0b8f1db46a50",
            "aec38fb2baf221e9a8d1887c7bf6222dd1834634e77263315af6d23609d04f77");
        CHECK(der_verify("3006020101020102", h, pk), "tcId 356: r=1 s=2 valid");
    }
    // tcId 357: r = 1, s = 3
    {
        auto pk = make_pubkey(
            "2e1f466b024c0c3ace2437de09127fed04b706f94b19a21bb1c2acf35cece718",
            "0449ae3523d72534e964972cfd3b38af0bddd9619e5af223e4d1a40f34cf9f1d");
        CHECK(der_verify("3006020101020103", h, pk), "tcId 357: r=1 s=3 valid");
    }
    // tcId 358: r = 2, s = 1
    {
        auto pk = make_pubkey(
            "8e7abdbbd18de7452374c1879a1c3b01d13261e7d4571c3b47a1c76c55a23373",
            "26ed897cd517a4f5349db809780f6d2f2b9f6299d8b5a89077f1119a718fd7b3");
        CHECK(der_verify("3006020102020101", h, pk), "tcId 358: r=2 s=1 valid");
    }
    // tcId 359: r = 2, s = 2
    {
        auto pk = make_pubkey(
            "7b333d4340d3d718dd3e6aff7de7bbf8b72bfd616c8420056052842376b9af19",
            "42117c5afeac755d6f376fc6329a7d76051b87123a4a5d0bc4a539380f03de7b");
        CHECK(der_verify("3006020102020102", h, pk), "tcId 359: r=2 s=2 valid");
    }
    // tcId 360: r = 2, s = 3
    {
        auto pk = make_pubkey(
            "d30ca4a0ddb6616c851d30ced682c40f83c62758a1f2759988d6763a88f1c0e5",
            "03a80d5415650d41239784e8e2fb1235e9fe991d112ebb81186cbf0da2de3aff");
        CHECK(der_verify("3006020102020103", h, pk), "tcId 360: r=2 s=3 valid");
    }
}

// ============================================================================
// E — PointDuplication (tcIds 390, 427-428, 442-445)
// Intermediate point arithmetic produces special values during ECDSA verify.
//
//   tcId 390:  u1*G + u2*Q = O (point at infinity) → REJECT
//   tcId 427:  valid sig exercising the point-duplication arithmetic → ACCEPT
//   tcId 428:  same sig as 427 but Q = -Q (negated pubkey) → REJECT
//   tcId 442:  pubkey = G (even-y), crafted sig → REJECT
//   tcId 443:  pubkey = G (even-y), different crafted sig → REJECT
//   tcId 444:  pubkey = -G (odd-y), same sigs as tcId 442 → REJECT
//   tcId 445:  pubkey = -G (odd-y), same sigs as tcId 443 → REJECT
// ============================================================================
static void test_point_duplication() {
    g_section = "PointDuplication";
    std::printf("  [E] PointDuplication (tcIds 390,427,428,442-445)\n");
    auto h = hash_std();

    // tcId 390: u1*G + u2*Q = O → invalid
    {
        auto pk = make_pubkey(
            "d533b789a4af890fa7a82a1fae58c404f9a62a50b49adafab349c513b4150874",
            "01b4171b803e76b34a9861e10f7bc289a066fd01bd29f84c987a10a5fb18c2d4");
        CHECK(!der_verify(
            "304402207fffffffffffffffffffffffffffffff5d576e7357a4501ddfe92f46681b20a0"
            "022055555555555555555555555555555554e8e4f44ce51835693ff0ca2ef01215c0",
            h, pk), "tcId 390: u1G+u2Q=inf must reject");
    }

    // tcId 427: valid — specific point-duplication path
    {
        auto pk = make_pubkey(
            "2ea7133432339c69d27f9b267281bd2ddd5f19d6338d400a05cd3647b157a385",
            "3547808298448edb5e701ade84cd5fb1ac9567ba5e8fb68a6b933ec4b5cc84cc");
        CHECK(der_verify(
            "3045022032b0d10d8d0e04bc8d4d064d270699e87cffc9b49c5c20730e1c26f6105ddcda"
            "022100d612c2984c2afa416aa7f2882a486d4a8426cb6cfc91ed5b737278f9fca8be68",
            h, pk), "tcId 427: point-dup path valid");
    }

    // tcId 428: same sig as 427 but negated pubkey (odd y) → invalid
    {
        auto pk = make_pubkey(
            "2ea7133432339c69d27f9b267281bd2ddd5f19d6338d400a05cd3647b157a385",
            "cab87f7d67bb7124a18fe5217b32a04e536a9845a1704975946cc13a4a337763");
        CHECK(!der_verify(
            "3045022032b0d10d8d0e04bc8d4d064d270699e87cffc9b49c5c20730e1c26f6105ddcda"
            "022100d612c2984c2afa416aa7f2882a486d4a8426cb6cfc91ed5b737278f9fca8be68",
            h, pk), "tcId 428: negated pubkey must reject");
    }

    // Generator G coordinates (secp256k1)
    const char* g_x = "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798";
    const char* g_y_even = "483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8";
    const char* g_y_odd  = "b7c52588d95c3b9aa25b0403f1eef75702e84bb7597aabe663b82f6f04ef2777";

    // sig_442 and sig_443 are crafted so that with pubkey = ±G,
    // the verify computation hits a point-duplication edge case
    const char* sig_442 =
        "3045022100bb5a52f42f9c9261ed4361f59422a1e30036e7c32b270c8807a419feca605023"
        "02202492492492492492492492492492492463cfd66a190a6008891e0d81d49a0952";
    const char* sig_443 =
        "3044022044a5ad0bd0636d9e12bc9e0a6bdd5e1bba77f523842193b3b82e448e05d5f11e"
        "02202492492492492492492492492492492463cfd66a190a6008891e0d81d49a0952";

    // tcId 442: pubkey = G (even y)
    CHECK(!der_verify(sig_442, h, make_pubkey(g_x, g_y_even)), "tcId 442: G even must reject");
    // tcId 443: pubkey = G (even y), different r
    CHECK(!der_verify(sig_443, h, make_pubkey(g_x, g_y_even)), "tcId 443: G even must reject");
    // tcId 444: pubkey = -G (odd y)
    CHECK(!der_verify(sig_442, h, make_pubkey(g_x, g_y_odd)),  "tcId 444: -G odd must reject");
    // tcId 445: pubkey = -G (odd y), different r
    CHECK(!der_verify(sig_443, h, make_pubkey(g_x, g_y_odd)),  "tcId 445: -G odd must reject");
}

// ============================================================================
// F — EdgeCasePublicKey (tcIds 446-463)
// Valid signatures whose public keys have extreme coordinate values.
// All use msg_hex = "4d657373616765" (ASCII "Message").
// ============================================================================
static void test_edge_case_pubkey() {
    g_section = "EdgeCasePublicKey";
    std::printf("  [F] EdgeCasePublicKey (tcIds 446-463)\n");
    auto h = hash_from_hex("4d657373616765"); // "Message"

    // ── Group 1: wy has leading zero bytes (wy is a 29-byte value, ≈ 2^224) ──
    // wy = 0x000000 01060492d5a5673e0f25d8d50fb7e58c49d86d46d4216955e0aa3d40e1
    {
        auto pk = make_pubkey_padded(
            "6e823555452914099182c6b2c1d6f0b5d28d50ccd005af2ce1bba541aa40caff",
            "01060492d5a5673e0f25d8d50fb7e58c49d86d46d4216955e0aa3d40e1");

        CHECK(der_verify(
            "304402206d6a4f556ccce154e7fb9f19e76c3deca13d59cc2aeb4ecad968aab2ded45965"
            "022053b9fa74803ede0fc4441bf683d56c564d3e274e09ccf47390badd1471c05fb7",
            h, pk), "tcId 446: small wy #1");

        CHECK(der_verify(
            "3046022100aad503de9b9fd66b948e9acf596f0a0e65e700b28b26ec56e6e45e846489b3c4"
            "022100fff223c5d0765447e8447a3f9d31fd0696e89d244422022ff61a110b2a8c2f04",
            h, pk), "tcId 447: small wy #2");

        CHECK(der_verify(
            "30460221009182cebd3bb8ab572e167174397209ef4b1d439af3b200cdf003620089e43225"
            "022100abb88367d15fe62d1efffb6803da03109ee22e90bc9c78e8b4ed23630b82ea9d",
            h, pk), "tcId 448: small wy #3");
    }

    // ── Group 2: wy close to field prime p ────────────────────────────────────
    // wy = fffffffef9fb6d2a5a98c1f0da272af0481a73b62792b92bde96aa1e55c2bb4e
    {
        auto pk = make_pubkey(
            "6e823555452914099182c6b2c1d6f0b5d28d50ccd005af2ce1bba541aa40caff",
            "fffffffef9fb6d2a5a98c1f0da272af0481a73b62792b92bde96aa1e55c2bb4e");

        CHECK(der_verify(
            "304502203854a3998aebdf2dbc28adac4181462ccac7873907ab7f212c42db0e69b56ed8"
            "022100c12c09475c772fd0c1b2060d5163e42bf71d727e4ae7c03eeba954bf50b43bb3",
            h, pk), "tcId 449: large wy #1");

        CHECK(der_verify(
            "3046022100e94dbdc38795fe5c904d8f16d969d3b587f0a25d2de90b6d8c5c53ff887e3607"
            "022100856b8c963e9b68dade44750bf97ec4d11b1a0a3804f4cb79aa27bdea78ac14e4",
            h, pk), "tcId 450: large wy #2");

        CHECK(der_verify(
            "3044022049fc102a08ca47b60e0858cd0284d22cddd7233f94aaffbb2db1dd2cf08425e1"
            "02205b16fca5a12cdb39701697ad8e39ffd6bdec0024298afaa2326aea09200b14d6",
            h, pk), "tcId 451: large wy #3");
    }

    // ── Group 3: wx has leading zero bytes (wx is a 29-byte value, ≈ 2^224) ──
    // wx = 0x000000 013fd22248d64d95f73c29b48ab48631850be503fd00f8468b5f0f70e0
    {
        auto pk = make_pubkey_padded(
            "013fd22248d64d95f73c29b48ab48631850be503fd00f8468b5f0f70e0",
            "f6ee7aa43bc2c6fd25b1d8269241cbdd9dbb0dac96dc96231f430705f838717d");

        CHECK(der_verify(
            "3045022041efa7d3f05a0010675fcb918a45c693da4b348df21a59d6f9cd73e0d831d67a"
            "022100bbab52596c1a1d9484296cdc92cbf07e665259a13791a8fe8845e2c07cf3fc67",
            h, pk), "tcId 452: small wx #1");

        CHECK(der_verify(
            "3046022100b615698c358b35920dd883eca625a6c5f7563970cdfc378f8fe0cee17092144c"
            "022100da0b84cd94a41e049ef477aeac157b2a9bfa6b7ac8de06ed3858c5eede6ddd6d",
            h, pk), "tcId 453: small wx #2");

        CHECK(der_verify(
            "304602210087cf8c0eb82d44f69c60a2ff5457d3aaa322e7ec61ae5aecfd678ae1c1932b0e"
            "022100c522c4eea7eafb82914cbf5c1ff76760109f55ddddcf58274d41c9bc4311e06e",
            h, pk), "tcId 454: small wx #3");
    }

    // ── Group 4: wx with many trailing 1-bits (ffffffff in low limb) ─────────
    {
        auto pk = make_pubkey(
            "25afd689acabaed67c1f296de59406f8c550f57146a0b4ec2c97876dffffffff",
            "fa46a76e520322dfbc491ec4f0cc197420fc4ea5883d8f6dd53c354bc4f67c35");

        CHECK(der_verify(
            "3045022062f48ef71ace27bf5a01834de1f7e3f948b9dce1ca1e911d5e13d3b104471d82"
            "022100a1570cc0f388768d3ba7df7f212564caa256ff825df997f21f72f5280d53011f",
            h, pk), "tcId 455: wx trailing-ff #1");

        CHECK(der_verify(
            "3046022100f6b0e2f6fe020cf7c0c20137434344ed7add6c4be51861e2d14cbda472a6ffb4"
            "0221009be93722c1a3ad7d4cf91723700cb5486de5479d8c1b38ae4e8e5ba1638e9732",
            h, pk), "tcId 456: wx trailing-ff #2");

        CHECK(der_verify(
            "3045022100db09d8460f05eff23bc7e436b67da563fa4b4edb58ac24ce201fa8a358125057"
            "022046da116754602940c8999c8d665f786c50f5772c0a3cdbda075e77eabc64df16",
            h, pk), "tcId 457: wx trailing-ff #3");
    }

    // ── Group 5: wy with many trailing 1-bits (ffffffff in low limb) ─────────
    {
        auto pk = make_pubkey(
            "d12e6c66b67734c3c84d2601cf5d35dc097e27637f0aca4a4fdb74b6aadd3bb9",
            "3f5bdff88bd5736df898e699006ed750f11cf07c5866cd7ad70c7121ffffffff");

        CHECK(der_verify(
            "30450220592c41e16517f12fcabd98267674f974b588e9f35d35406c1a7bb2ed1d19b7b8"
            "022100c19a5f942607c3551484ff0dc97281f0cdc82bc48e2205a0645c0cf3d7f59da0",
            h, pk), "tcId 458: wy trailing-ff #1");

        CHECK(der_verify(
            "3046022100be0d70887d5e40821a61b68047de4ea03debfdf51cdf4d4b195558b959a032b2"
            "0221008266b4d270e24414ecacb14c091a233134b918d37320c6557d60ad0a63544ac4",
            h, pk), "tcId 459: wy trailing-ff #2");

        CHECK(der_verify(
            "3046022100fae92dfcb2ee392d270af3a5739faa26d4f97bfd39ed3cbee4d29e26af3b206a"
            "02210093645c80605595e02c09a0dc4b17ac2a51846a728b3e8d60442ed6449fd3342b",
            h, pk), "tcId 460: wy trailing-ff #3");
    }

    // ── Group 6: wx with many trailing 0-bits (00000000 in low limb) ─────────
    {
        auto pk = make_pubkey(
            "6d4a7f60d4774a4f0aa8bbdedb953c7eea7909407e3164755664bc2800000000",
            "e659d34e4df38d9e8c9eaadfba36612c769195be86c77aac3f36e78b538680fb");

        CHECK(der_verify(
            "30450220176a2557566ffa518b11226694eb9802ed2098bfe278e5570fe1d5d7af18a943"
            "022100ed6e2095f12a03f2eaf6718f430ec5fe2829fd1646ab648701656fd31221b97d",
            h, pk), "tcId 461: wx trailing-00 #1");

        CHECK(der_verify(
            "3045022060be20c3dbc162dd34d26780621c104bbe5dace630171b2daef0d826409ee5c2"
            "022100bd8081b27762ab6e8f425956bf604e332fa066a99b59f87e27dc1198b26f5caa",
            h, pk), "tcId 462: wx trailing-00 #2");

        CHECK(der_verify(
            "3046022100edf03cf63f658883289a1a593d1007895b9f236d27c9c1f1313089aaed6b16ae"
            "022100e5b22903f7eb23adc2e01057e39b0408d495f694c83f306f1216c9bf87506074",
            h, pk), "tcId 463: wx trailing-00 #3");
    }
}

// ============================================================================
// Entry point
// ============================================================================
int test_wycheproof_ecdsa_extended_run() {
    std::printf("=== Wycheproof ECDSA secp256k1 SHA-256 Extended Coverage ===\n");
    test_range_check();
    test_invalid_type_signatures();
    test_shamir_edge_case();
    test_small_r_and_s();
    test_point_duplication();
    test_edge_case_pubkey();
    std::printf("--- Result: %d passed, %d failed ---\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() {
    return test_wycheproof_ecdsa_extended_run();
}
#endif
