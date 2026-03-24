// ============================================================================
// Message Signing -- Implementation
// ============================================================================

#include "secp256k1/coins/message_signing.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/recovery.hpp"
#include "secp256k1/ct/sign.hpp"
#include "secp256k1/hash_accel.hpp"
#include "secp256k1/detail/secure_erase.hpp"
#include <cstring>

namespace secp256k1::coins {

using secp256k1::hash::sha256;

// -- Bitcoin varint encoding --------------------------------------------------

// Write Bitcoin-style compact size (varint) into buffer.
// Returns number of bytes written (1, 3, 5, or 9).
static std::size_t write_varint(std::uint8_t* out, std::uint64_t val) {
    if (val < 0xFD) {
        out[0] = static_cast<std::uint8_t>(val);
        return 1;
    } else if (val <= 0xFFFF) {
        out[0] = 0xFD;
        out[1] = static_cast<std::uint8_t>(val & 0xFF);
        out[2] = static_cast<std::uint8_t>((val >> 8) & 0xFF);
        return 3;
    } else if (val <= 0xFFFFFFFF) {
        out[0] = 0xFE;
        for (int i = 0; i < 4; ++i) {
            out[1 + i] = static_cast<std::uint8_t>((val >> (8 * i)) & 0xFF);
        }
        return 5;
    } else {
        out[0] = 0xFF;
        for (int i = 0; i < 8; ++i) {
            out[1 + i] = static_cast<std::uint8_t>((val >> (8 * i)) & 0xFF);
        }
        return 9;
    }
}

// -- Bitcoin Message Hash -----------------------------------------------------

// "\x18Bitcoin Signed Message:\n"
static constexpr std::uint8_t BITCOIN_MSG_PREFIX[] = {
    0x18, // length of "Bitcoin Signed Message:\n" (24 bytes)
    'B','i','t','c','o','i','n',' ',
    'S','i','g','n','e','d',' ',
    'M','e','s','s','a','g','e',':','\n'
};
static constexpr std::size_t BITCOIN_MSG_PREFIX_LEN = sizeof(BITCOIN_MSG_PREFIX);

std::array<std::uint8_t, 32> bitcoin_message_hash(const std::uint8_t* msg,
                                                   std::size_t msg_len) {
    // Construct: prefix + varint(msg_len) + msg
    // Then double-SHA256 the result
    std::uint8_t varint_buf[9];
    const std::size_t varint_len = write_varint(varint_buf, msg_len);

    // Overflow guard: prefix(25) + varint(<=9) + msg_len
    if (msg_len > SIZE_MAX - BITCOIN_MSG_PREFIX_LEN - 9) {
        return {};
    }

    // Total payload size
    const std::size_t total = BITCOIN_MSG_PREFIX_LEN + varint_len + msg_len;

    // Stack buffer for small messages, heap for large
    constexpr std::size_t STACK_MAX = 512;
    std::uint8_t stack_buf[STACK_MAX];
    std::uint8_t* buf = (total <= STACK_MAX) ? stack_buf : new std::uint8_t[total];

    std::memcpy(buf, BITCOIN_MSG_PREFIX, BITCOIN_MSG_PREFIX_LEN);
    std::memcpy(buf + BITCOIN_MSG_PREFIX_LEN, varint_buf, varint_len);
    if (msg_len > 0) {
        std::memcpy(buf + BITCOIN_MSG_PREFIX_LEN + varint_len, msg, msg_len);
    }

    // Double SHA-256
    auto hash1 = sha256(buf, total);
    auto hash2 = sha256(hash1.data(), 32);

    // Erase intermediate hash and message buffer so sensitive content does not
    // linger in heap or stack memory after this function returns.
    secp256k1::detail::secure_erase(hash1.data(), hash1.size());
    if (buf != stack_buf) {
        secp256k1::detail::secure_erase(buf, total);
        delete[] buf;
    } else {
        secp256k1::detail::secure_erase(stack_buf, STACK_MAX);
    }

    return hash2;
}

// -- Bitcoin Sign Message -----------------------------------------------------

RecoverableSignature bitcoin_sign_message(const std::uint8_t* msg,
                                          std::size_t msg_len,
                                          const fast::Scalar& private_key) {
    auto hash = bitcoin_message_hash(msg, msg_len);
    // Use CT path: private key and RFC-6979 nonce must not leak via timing.
    // (Q-07: fast::ecdsa_sign_recoverable is variable-time on the nonce.)
    return ct::ecdsa_sign_recoverable(hash, private_key);
}

// -- Bitcoin Verify Message ---------------------------------------------------

bool bitcoin_verify_message(const std::uint8_t* msg,
                            std::size_t msg_len,
                            const fast::Point& pubkey,
                            const ECDSASignature& sig) {
    auto hash = bitcoin_message_hash(msg, msg_len);
    return ecdsa_verify(hash.data(), pubkey, sig);
}

// -- Bitcoin Recover Message --------------------------------------------------

std::pair<fast::Point, bool>
bitcoin_recover_message(const std::uint8_t* msg,
                        std::size_t msg_len,
                        const ECDSASignature& sig,
                        int recid) {
    auto hash = bitcoin_message_hash(msg, msg_len);
    return ecdsa_recover(hash, sig, recid);
}

// -- Base64 Encode/Decode (minimal, for 65-byte sigs) -------------------------

static constexpr char BASE64_CHARS[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static std::string base64_encode(const std::uint8_t* data, std::size_t len) {
    std::string result;
    result.reserve(((len + 2) / 3) * 4);
    for (std::size_t i = 0; i < len; i += 3) {
        std::uint32_t triple = static_cast<std::uint32_t>(data[i]) << 16;
        if (i + 1 < len) triple |= static_cast<std::uint32_t>(data[i + 1]) << 8;
        if (i + 2 < len) triple |= static_cast<std::uint32_t>(data[i + 2]);

        result += BASE64_CHARS[(triple >> 18) & 0x3F];
        result += BASE64_CHARS[(triple >> 12) & 0x3F];
        result += (i + 1 < len) ? BASE64_CHARS[(triple >> 6) & 0x3F] : '=';
        result += (i + 2 < len) ? BASE64_CHARS[triple & 0x3F] : '=';
    }
    return result;
}

static int base64_char_value(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

static bool base64_decode(const std::string& b64, std::uint8_t* out, std::size_t expected_len) {
    if (b64.size() != ((expected_len + 2) / 3) * 4) return false;

    std::size_t out_idx = 0;
    for (std::size_t i = 0; i < b64.size(); i += 4) {
        const int a = base64_char_value(b64[i]);
        const int b = base64_char_value(b64[i + 1]);
        int c = (b64[i + 2] == '=') ? 0 : base64_char_value(b64[i + 2]);
        int d = (b64[i + 3] == '=') ? 0 : base64_char_value(b64[i + 3]);

        if (a < 0 || b < 0 || c < 0 || d < 0) return false;

        std::uint32_t triple = (static_cast<std::uint32_t>(a) << 18) |
                               (static_cast<std::uint32_t>(b) << 12) |
                               (static_cast<std::uint32_t>(c) << 6) |
                               static_cast<std::uint32_t>(d);

        if (out_idx < expected_len) out[out_idx++] = static_cast<std::uint8_t>((triple >> 16) & 0xFF);
        if (out_idx < expected_len) out[out_idx++] = static_cast<std::uint8_t>((triple >> 8) & 0xFF);
        if (out_idx < expected_len) out[out_idx++] = static_cast<std::uint8_t>(triple & 0xFF);
    }
    return out_idx == expected_len;
}

// -- Bitcoin Sig Base64 -------------------------------------------------------

std::string bitcoin_sig_to_base64(const RecoverableSignature& rsig,
                                  bool compressed) {
    // 65-byte format: [header] [r:32] [s:32]
    // header = 27 + recid + (compressed ? 4 : 0)
    std::uint8_t buf[65];
    buf[0] = static_cast<std::uint8_t>(27 + rsig.recid + (compressed ? 4 : 0));

    auto compact = rsig.sig.to_compact();
    std::memcpy(buf + 1, compact.data(), 64);

    return base64_encode(buf, 65);
}

BitcoinSigDecodeResult bitcoin_sig_from_base64(const std::string& base64_str) {
    BitcoinSigDecodeResult result{};
    result.valid = false;

    std::uint8_t buf[65];
    if (!base64_decode(base64_str, buf, 65)) return result;

    std::uint8_t header = buf[0];
    if (header < 27 || header > 34) return result;

    int flag = header - 27;
    result.recid = flag & 3;
    result.compressed = (flag & 4) != 0;

    result.sig = ECDSASignature::from_compact(buf + 1);
    result.valid = true;
    return result;
}

} // namespace secp256k1::coins
