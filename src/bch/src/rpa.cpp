// ============================================================================
// rpa.cpp — BCH Reusable Payment Addresses implementation
// ============================================================================
#include "secp256k1/bch/rpa.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/ct/sign.hpp"
#include <cstring>
#include <vector>

namespace secp256k1::bch {

// ── SHA256 helpers ────────────────────────────────────────────────────────────

static SHA256::digest_type sha256_bytes(const uint8_t* data, size_t len) noexcept {
    return SHA256::hash(data, len);
}

static SHA256::digest_type sha256_double(const uint8_t* data, size_t len) noexcept {
    return SHA256::hash256(data, len);
}

// ── Shared secret derivation — ported from BIP-352 with RPA hash domain ────────
//
// BIP-352: t_k = SHA256_tagged("BIP0352/SharedSecret", ser(S) || ser32(k))
// RPA:     c   = SHA256(SHA256(ser_compressed(S)) || outpoint_bytes)
//
// Key optimisation from BIP-352 scanner (address.cpp):
//   Pre-compute SHA256 midstate over SHA256(S_compressed) once per tx.
//   Loop over outputs/indices only re-feeds outpoint (cheap).

// Build the SHA256 base state: inner = SHA256(compressed_S)
// Returns the SHA256 context already fed with inner — caller adds outpoint.
static SHA256 rpa_shared_secret_base(const fast::Point& ecdh_point) noexcept {
    auto compressed = ecdh_point.to_compressed();
    auto inner = SHA256::hash(compressed.data(), compressed.size()); // SHA256(S_comp)
    SHA256 h;
    h.update(inner.data(), 32); // seed with inner hash
    return h;
}

// Complete the shared secret: SHA256(inner || outpoint)
static RpaSharedSecret rpa_shared_secret_finish(
    SHA256 h_base,  // copy of base state (caller holds original)
    const uint8_t* outpoint_bytes,
    size_t outpoint_len) noexcept {

    if (outpoint_len > 0)
        h_base.update(outpoint_bytes, outpoint_len);
    RpaSharedSecret result{};
    result.value = h_base.finalize();
    return result;
}

// Full derivation (single call convenience)
static RpaSharedSecret derive_shared_secret(
    const fast::Point& ecdh_point,
    const uint8_t* outpoint_bytes,
    size_t outpoint_len) noexcept {

    auto h = rpa_shared_secret_base(ecdh_point);
    return rpa_shared_secret_finish(h, outpoint_bytes, outpoint_len);
}

// Parse compressed 33-byte pubkey → Point. Returns infinity on failure.
static fast::Point parse_pubkey(const uint8_t* pubkey33) noexcept {
    EcdsaPublicKey epk{};
    if (!secp256k1::ecdsa_pubkey_parse(epk, pubkey33, 33))
        return fast::Point::infinity();
    return epk.point;
}

RpaSharedSecret rpa_sender_shared_secret(
    const fast::Scalar& input_privkey,
    const uint8_t* scan_pubkey33,
    const uint8_t* outpoint_bytes,
    size_t outpoint_len) noexcept {

    fast::Point Q = parse_pubkey(scan_pubkey33);
    if (Q.is_infinity()) return {};
    // CT: input_privkey is secret
    fast::Point ecdh_point = secp256k1::ct::scalar_mul(Q, input_privkey);
    return derive_shared_secret(ecdh_point, outpoint_bytes, outpoint_len);
}

RpaSharedSecret rpa_receiver_shared_secret(
    const fast::Scalar& scan_privkey,
    const uint8_t* input_pubkey33,
    const uint8_t* outpoint_bytes,
    size_t outpoint_len) noexcept {

    fast::Point P = parse_pubkey(input_pubkey33);
    if (P.is_infinity()) return {};
    // CT: scan_privkey is secret
    fast::Point ecdh_point = secp256k1::ct::scalar_mul(P, scan_privkey);
    return derive_shared_secret(ecdh_point, outpoint_bytes, outpoint_len);
}

// ── Payment address derivation ────────────────────────────────────────────────

// Build SHA256 midstate over (spend_pubkey33 || secret) — call once per tx.
// Returns SHA256 context ready for appending the 4-byte index.
// PERF: avoids re-hashing the 65 static bytes for every key index.
SHA256 rpa_payment_key_base(const uint8_t* spend_pubkey33,
                            const RpaSharedSecret& secret) noexcept {
    SHA256 h;
    h.update(spend_pubkey33, 33);
    h.update(secret.value.data(), 32);
    return h;
}

// Derive payment pubkey for a given index using pre-built midstate + pre-parsed Point.
// PERF: spend_point must be pre-parsed once (lift_x is expensive — ~1.6µs).
//       h_base must be pre-built once per tx via rpa_payment_key_base().
std::array<uint8_t, 33> rpa_derive_payment_pubkey_fast(
    const fast::Point& spend_point,
    SHA256 h_base,               // copy — caller keeps original for next index
    uint32_t index) noexcept {

    // Append 4-byte index and finalize: tweak = SHA256(spend_pubkey || secret || index_BE)
    uint8_t idx_be[4] = {
        uint8_t(index >> 24), uint8_t(index >> 16),
        uint8_t(index >>  8), uint8_t(index)
    };
    h_base.update(idx_be, 4);
    auto tweak_bytes = h_base.finalize();
    fast::Scalar t = fast::Scalar::from_bytes(tweak_bytes.data());

    // child = spend_point + t*G using ct::generator_mul (precomputed table, ~33µs)
    // PERF: ct::generator_mul is 25× faster than Point::generator().scalar_mul()
    //       which uses the cold FAST path (~826µs).
    fast::Point tG    = secp256k1::ct::generator_mul(t);
    fast::Point child = spend_point.add(tG);
    if (child.is_infinity()) return {};

    auto comp = child.to_compressed();
    std::array<uint8_t, 33> result{};
    std::memcpy(result.data(), comp.data(), 33);
    return result;
}

// Public convenience: derive from raw bytes (single call, parses spend_pubkey each time).
// Use rpa_derive_payment_pubkey_fast() in hot paths.
std::array<uint8_t, 33> rpa_derive_payment_pubkey(
    const uint8_t* spend_pubkey33,
    const RpaSharedSecret& secret,
    uint32_t index) noexcept {

    fast::Point P = parse_pubkey(spend_pubkey33);
    if (P.is_infinity()) return {};
    auto h_base = rpa_payment_key_base(spend_pubkey33, secret);
    return rpa_derive_payment_pubkey_fast(P, h_base, index);
}

// ── Prefix matching ────────────────────────────────────────────────────────────

std::array<uint8_t, 32> rpa_sig_hash(const uint8_t* sig64) noexcept {
    SHA256::digest_type inner = sha256_bytes(sig64, 64);
    return sha256_bytes(inner.data(), 32);
}

bool rpa_prefix_matches(const uint8_t* sig64,
                        uint8_t prefix_bits,
                        const uint8_t* prefix_data) noexcept {
    if (prefix_bits == 0) return true;

    // PERF: Early-exit after first SHA256 — avoids 2nd SHA256 for non-matching sigs.
    // For 8-bit prefix: saves 255/256 of second SHA256 ≈ 50% total hash work.
    // For 16-bit prefix: saves ~99.6% of second SHA256.
    auto inner = SHA256::hash(sig64, 64);         // SHA256(sig) — first hash

    uint8_t full_bytes = prefix_bits / 8;
    uint8_t rem_bits   = prefix_bits % 8;

    // Early check on first SHA256 output
    if (std::memcmp(inner.data(), prefix_data, full_bytes) != 0) return false;
    if (rem_bits != 0) {
        uint8_t mask = static_cast<uint8_t>(0xff << (8 - rem_bits));
        if ((inner[full_bytes] & mask) != (prefix_data[full_bytes] & mask)) return false;
    }

    // Prefix matched first SHA256 → compute second SHA256 to verify double-hash
    auto outer = SHA256::hash(inner.data(), 32);  // SHA256(SHA256(sig)) — second hash
    if (std::memcmp(outer.data(), prefix_data, full_bytes) != 0) return false;
    if (rem_bits == 0) return true;
    uint8_t mask = static_cast<uint8_t>(0xff << (8 - rem_bits));
    return (outer[full_bytes] & mask) == (prefix_data[full_bytes] & mask);
}

// ── EC Grinding (CPU) ─────────────────────────────────────────────────────────

GrindResult rpa_grind_cpu(
    const fast::Scalar& input_privkey,
    const uint8_t* msg32,
    uint8_t prefix_bits,
    const uint8_t* prefix_data,
    uint32_t max_tries,
    GrindProgressFn on_progress) noexcept {

    GrindResult result{};

    for (uint32_t nonce = 0; max_tries == 0 || nonce < max_tries; ++nonce) {
        if (on_progress && (nonce & 0xfff) == 0)
            on_progress(nonce);

        // RFC6979 extra data = 4-byte nonce counter
        std::array<uint8_t, 4> extra = {
            uint8_t(nonce >> 24), uint8_t(nonce >> 16),
            uint8_t(nonce >>  8), uint8_t(nonce)
        };

        // CT ECDSA sign hedged — RFC6979 + extra entropy for grinding
        std::array<uint8_t, 32> msg_arr{};
        std::memcpy(msg_arr.data(), msg32, 32);
        std::array<uint8_t, 32> aux_arr{};
        // embed 4-byte nonce in first 4 bytes of aux_rand
        std::memcpy(aux_arr.data(), extra.data(), 4);
        auto sig = secp256k1::ct::ecdsa_sign_hedged(msg_arr, input_privkey, aux_arr);
        {

        auto compact = sig.to_compact();
        if (rpa_prefix_matches(compact.data(), prefix_bits, prefix_data)) {
            result.found = true;
            result.nonce = nonce;
            std::memcpy(result.signature.data(), compact.data(), 64);
            result.input_hash = rpa_sig_hash(compact.data());
            return result;
        }
        } // end sig scope
    }
    return result;
}

// ── Paycode parse / encode ────────────────────────────────────────────────────
// RPA paycode format (base32, CashAddr charset):
//   "paycode:" + base32(version[1] + prefix_bits[1] + scan_pubkey[33]
//                       + spend_pubkey[33] + expiry[4]) + checksum[8]
// Total raw bytes: 1 + 1 + 33 + 33 + 4 = 72 bytes → ~115 base32 chars

static constexpr char BASE32_CHARSET[] = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";
static constexpr std::string_view PAYCODE_PREFIX = "paycode";

static uint64_t paycode_polymod(const std::vector<uint8_t>& v) noexcept {
    static constexpr uint64_t GEN[5] = {
        0x98f2bc8e61ULL, 0x79b76d99e2ULL,
        0xf33e5fb3c4ULL, 0xae2eabe2a8ULL, 0x1e4f43e470ULL
    };
    uint64_t c = 1;
    for (uint8_t d : v) {
        uint8_t c0 = static_cast<uint8_t>(c >> 35);
        c = ((c & 0x07ffffffffULL) << 5) ^ d;
        for (int i = 0; i < 5; ++i)
            if ((c0 >> i) & 1) c ^= GEN[i];
    }
    return c ^ 1;
}

static void bytes_to_base32(std::vector<uint8_t>& out,
                             const uint8_t* in, size_t len) {
    int acc = 0, bits = 0;
    for (size_t i = 0; i < len; ++i) {
        acc = (acc << 8) | in[i];
        bits += 8;
        while (bits >= 5) {
            bits -= 5;
            out.push_back((acc >> bits) & 31);
        }
    }
    if (bits > 0) out.push_back((acc << (5 - bits)) & 31);
}

static bool base32_to_bytes(std::vector<uint8_t>& out,
                             const uint8_t* in, size_t len) {
    int acc = 0, bits = 0;
    for (size_t i = 0; i < len; ++i) {
        if (in[i] >= 32) return false;
        acc = (acc << 5) | in[i];
        bits += 5;
        if (bits >= 8) {
            bits -= 8;
            out.push_back((acc >> bits) & 0xff);
        }
    }
    return true;
}

std::optional<RpaPaycode> rpa_parse_paycode(std::string_view s) noexcept {
    // strip "paycode:" prefix (case-insensitive)
    if (s.size() > 8 && s.substr(0, 8) == "paycode:")
        s = s.substr(8);
    else if (s.size() > 8 && s.substr(0, 8) == "PAYCODE:")
        s = s.substr(8);

    // decode chars → 5-bit groups
    std::vector<uint8_t> data5;
    data5.reserve(s.size());
    for (char c : s) {
        const char* p = std::strchr(BASE32_CHARSET, std::tolower(static_cast<unsigned char>(c)));
        if (!p) return std::nullopt;
        data5.push_back(static_cast<uint8_t>(p - BASE32_CHARSET));
    }
    if (data5.size() < 8) return std::nullopt;

    // verify checksum
    std::vector<uint8_t> chk_input;
    for (char c : PAYCODE_PREFIX)
        chk_input.push_back(static_cast<uint8_t>(c) & 0x1f);
    chk_input.push_back(0);
    chk_input.insert(chk_input.end(), data5.begin(), data5.end());
    if (paycode_polymod(chk_input) != 0) return std::nullopt;

    // strip checksum
    data5.resize(data5.size() - 8);

    // decode to bytes
    std::vector<uint8_t> raw;
    if (!base32_to_bytes(raw, data5.data(), data5.size())) return std::nullopt;
    if (raw.size() < 72) return std::nullopt;  // 1+1+33+33+4

    RpaPaycode pc;
    size_t pos = 0;
    pc.version     = raw[pos++];
    pc.prefix_bits = raw[pos++];
    std::memcpy(pc.scan_pubkey.data(),  raw.data() + pos, 33); pos += 33;
    std::memcpy(pc.spend_pubkey.data(), raw.data() + pos, 33); pos += 33;
    pc.expiry  = (uint32_t(raw[pos]) << 24) | (uint32_t(raw[pos+1]) << 16)
               | (uint32_t(raw[pos+2]) << 8) | raw[pos+3];
    return pc;
}

std::string rpa_encode_paycode(const RpaPaycode& pc) noexcept {
    // pack raw bytes: version + prefix_bits + scan_pubkey + spend_pubkey + expiry
    uint8_t raw[72];
    raw[0] = pc.version;
    raw[1] = pc.prefix_bits;
    std::memcpy(raw + 2,  pc.scan_pubkey.data(),  33);
    std::memcpy(raw + 35, pc.spend_pubkey.data(), 33);
    raw[68] = (pc.expiry >> 24) & 0xff;
    raw[69] = (pc.expiry >> 16) & 0xff;
    raw[70] = (pc.expiry >>  8) & 0xff;
    raw[71] =  pc.expiry        & 0xff;

    // encode to 5-bit groups
    std::vector<uint8_t> data5;
    bytes_to_base32(data5, raw, sizeof(raw));

    // compute checksum
    std::vector<uint8_t> chk_input;
    for (char c : PAYCODE_PREFIX)
        chk_input.push_back(static_cast<uint8_t>(c) & 0x1f);
    chk_input.push_back(0);
    chk_input.insert(chk_input.end(), data5.begin(), data5.end());
    for (int i = 0; i < 8; ++i) chk_input.push_back(0);
    uint64_t cksum = paycode_polymod(chk_input);

    std::string result = "paycode:";
    for (uint8_t v : data5) result += BASE32_CHARSET[v];
    for (int i = 7; i >= 0; --i)
        result += BASE32_CHARSET[(cksum >> (5 * i)) & 0x1f];
    return result;
}

} // namespace secp256k1::bch
