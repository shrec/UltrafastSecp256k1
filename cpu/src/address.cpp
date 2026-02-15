// ============================================================================
// Address Generation + Silent Payments — Implementation
// ============================================================================

#include "secp256k1/address.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/field.hpp"
#include <algorithm>
#include <cstring>

namespace secp256k1 {

using fast::Point;
using fast::Scalar;
using fast::FieldElement;

// ═══════════════════════════════════════════════════════════════════════════════
// RIPEMD-160 (self-contained, needed for HASH160)
// ═══════════════════════════════════════════════════════════════════════════════

namespace {

class RIPEMD160 {
public:
    static std::array<std::uint8_t, 20> hash(const std::uint8_t* data, std::size_t len) {
        RIPEMD160 ctx;
        ctx.update(data, len);
        return ctx.finalize();
    }

    RIPEMD160() {
        h_[0] = 0x67452301u; h_[1] = 0xEFCDAB89u;
        h_[2] = 0x98BADCFEu; h_[3] = 0x10325476u;
        h_[4] = 0xC3D2E1F0u;
        total_ = 0; buf_len_ = 0;
    }

    void update(const std::uint8_t* data, std::size_t len) {
        total_ += len;
        if (buf_len_ > 0) {
            std::size_t fill = 64 - buf_len_;
            if (len < fill) { std::memcpy(buf_ + buf_len_, data, len); buf_len_ += len; return; }
            std::memcpy(buf_ + buf_len_, data, fill);
            compress(buf_); data += fill; len -= fill; buf_len_ = 0;
        }
        while (len >= 64) { compress(data); data += 64; len -= 64; }
        if (len > 0) { std::memcpy(buf_, data, len); buf_len_ = len; }
    }

    std::array<std::uint8_t, 20> finalize() {
        std::uint64_t bits = total_ * 8;
        std::uint8_t pad = 0x80;
        update(&pad, 1);
        pad = 0;
        while (buf_len_ != 56) update(&pad, 1);
        std::uint8_t len_le[8];
        for (int i = 0; i < 8; ++i) len_le[i] = std::uint8_t(bits >> (i * 8));
        update(len_le, 8);
        std::array<std::uint8_t, 20> out;
        for (int i = 0; i < 5; ++i) {
            out[i*4+0] = std::uint8_t(h_[i]); out[i*4+1] = std::uint8_t(h_[i]>>8);
            out[i*4+2] = std::uint8_t(h_[i]>>16); out[i*4+3] = std::uint8_t(h_[i]>>24);
        }
        return out;
    }

private:
    static std::uint32_t rotl(std::uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }
    static std::uint32_t f(int j, std::uint32_t x, std::uint32_t y, std::uint32_t z) {
        if (j < 16) return x ^ y ^ z;
        if (j < 32) return (x & y) | (~x & z);
        if (j < 48) return (x | ~y) ^ z;
        if (j < 64) return (x & z) | (y & ~z);
        return x ^ (y | ~z);
    }

    void compress(const std::uint8_t* block) {
        std::uint32_t X[16];
        for (int i = 0; i < 16; ++i)
            X[i] = std::uint32_t(block[i*4]) | (std::uint32_t(block[i*4+1])<<8) |
                   (std::uint32_t(block[i*4+2])<<16) | (std::uint32_t(block[i*4+3])<<24);

        static constexpr int rl[80] = {
            0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
            7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
            3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
            1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
            4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13
        };
        static constexpr int rr[80] = {
            5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
            6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
            15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,
            8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
            12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11
        };
        static constexpr int sl[80] = {
            11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
            7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
            11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
            11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
            9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6
        };
        static constexpr int sr[80] = {
            8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
            9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
            9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
            15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
            8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11
        };
        static constexpr std::uint32_t KL[5] = {0, 0x5A827999u, 0x6ED9EBA1u, 0x8F1BBCDCu, 0xA953FD4Eu};
        static constexpr std::uint32_t KR[5] = {0x50A28BE6u, 0x5C4DD124u, 0x6D703EF3u, 0x7A6D76E9u, 0};

        std::uint32_t al=h_[0],bl=h_[1],cl=h_[2],dl=h_[3],el=h_[4];
        std::uint32_t ar=h_[0],br=h_[1],cr=h_[2],dr=h_[3],er=h_[4];

        for (int j = 0; j < 80; ++j) {
            std::uint32_t tl = al + f(j,bl,cl,dl) + X[rl[j]] + KL[j/16];
            tl = rotl(tl, sl[j]) + el;
            al = el; el = dl; dl = rotl(cl, 10); cl = bl; bl = tl;

            std::uint32_t tr = ar + f(79-j,br,cr,dr) + X[rr[j]] + KR[j/16];
            tr = rotl(tr, sr[j]) + er;
            ar = er; er = dr; dr = rotl(cr, 10); cr = br; br = tr;
        }

        std::uint32_t t = h_[1] + cl + dr;
        h_[1] = h_[2] + dl + er;
        h_[2] = h_[3] + el + ar;
        h_[3] = h_[4] + al + br;
        h_[4] = h_[0] + bl + cr;
        h_[0] = t;
    }

    std::uint32_t h_[5];
    std::uint8_t buf_[64];
    std::size_t buf_len_;
    std::uint64_t total_;
};

} // anonymous namespace

// ═══════════════════════════════════════════════════════════════════════════════
// HASH160
// ═══════════════════════════════════════════════════════════════════════════════

std::array<std::uint8_t, 20> hash160(const std::uint8_t* data, std::size_t len) {
    auto sha = SHA256::hash(data, len);
    return RIPEMD160::hash(sha.data(), 32);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Base58Check
// ═══════════════════════════════════════════════════════════════════════════════

static const char BASE58_ALPHABET[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

static int base58_char_value(char c) {
    if (c >= '1' && c <= '9') return c - '1';
    if (c >= 'A' && c <= 'H') return c - 'A' + 9;
    if (c >= 'J' && c <= 'N') return c - 'J' + 17;
    if (c >= 'P' && c <= 'Z') return c - 'P' + 22;
    if (c >= 'a' && c <= 'k') return c - 'a' + 33;
    if (c >= 'm' && c <= 'z') return c - 'm' + 44;
    return -1;
}

std::string base58check_encode(const std::uint8_t* data, std::size_t len) {
    // Append 4-byte checksum
    auto checksum_hash1 = SHA256::hash(data, len);
    auto checksum_hash2 = SHA256::hash(checksum_hash1.data(), 32);

    std::vector<std::uint8_t> payload(len + 4);
    std::memcpy(payload.data(), data, len);
    std::memcpy(payload.data() + len, checksum_hash2.data(), 4);

    // Count leading zeros
    std::size_t leading_zeros = 0;
    while (leading_zeros < payload.size() && payload[leading_zeros] == 0) ++leading_zeros;

    // Base58 encode (big number division)
    std::string result;
    result.reserve(payload.size() * 138 / 100 + 1);

    // Use a copy for division
    std::vector<std::uint8_t> num(payload.begin(), payload.end());
    while (!num.empty()) {
        int remainder = 0;
        std::vector<std::uint8_t> quotient;
        for (std::size_t i = 0; i < num.size(); ++i) {
            int acc = remainder * 256 + num[i];
            int digit = acc / 58;
            remainder = acc % 58;
            if (!quotient.empty() || digit > 0) {
                quotient.push_back(static_cast<std::uint8_t>(digit));
            }
        }
        result.push_back(BASE58_ALPHABET[remainder]);
        num = std::move(quotient);
    }

    // Add '1' for each leading zero byte
    for (std::size_t i = 0; i < leading_zeros; ++i) {
        result.push_back('1');
    }

    std::reverse(result.begin(), result.end());
    return result;
}

std::pair<std::vector<std::uint8_t>, bool>
base58check_decode(const std::string& encoded) {
    // Decode from base58
    std::vector<std::uint8_t> bytes;
    bytes.reserve(encoded.size());

    // Count leading '1's
    std::size_t leading_ones = 0;
    while (leading_ones < encoded.size() && encoded[leading_ones] == '1') ++leading_ones;

    // Convert from base58 to base256
    std::vector<int> digits;
    for (char c : encoded) {
        int val = base58_char_value(c);
        if (val < 0) return {{}, false};

        int carry = val;
        for (auto it = digits.rbegin(); it != digits.rend(); ++it) {
            int acc = *it * 58 + carry;
            *it = acc % 256;
            carry = acc / 256;
        }
        while (carry > 0) {
            digits.insert(digits.begin(), carry % 256);
            carry /= 256;
        }
    }

    // Prepend leading zeros
    for (std::size_t i = 0; i < leading_ones; ++i) {
        digits.insert(digits.begin(), 0);
    }

    if (digits.size() < 4) return {{}, false};

    // Verify checksum
    std::size_t payload_len = digits.size() - 4;
    std::vector<std::uint8_t> payload(digits.begin(), digits.begin() + static_cast<std::ptrdiff_t>(payload_len));
    auto h1 = SHA256::hash(payload.data(), payload_len);
    auto h2 = SHA256::hash(h1.data(), 32);

    for (int i = 0; i < 4; ++i) {
        if (digits[payload_len + i] != static_cast<int>(h2[i])) return {{}, false};
    }

    return {payload, true};
}

// ═══════════════════════════════════════════════════════════════════════════════
// Bech32 / Bech32m (BIP-173 / BIP-350)
// ═══════════════════════════════════════════════════════════════════════════════

static const char BECH32_CHARSET[] = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";

static int bech32_charset_value(char c) {
    const char* p = std::strchr(BECH32_CHARSET, c);
    if (!p) return -1;
    return static_cast<int>(p - BECH32_CHARSET);
}

static std::uint32_t bech32_polymod(const std::vector<std::uint8_t>& values) {
    static constexpr std::uint32_t GEN[5] = {
        0x3b6a57b2u, 0x26508e6du, 0x1ea119fau, 0x3d4233ddu, 0x2a1462b3u
    };
    std::uint32_t chk = 1;
    for (auto v : values) {
        std::uint32_t top = chk >> 25;
        chk = ((chk & 0x1ffffffu) << 5) ^ v;
        for (int i = 0; i < 5; ++i) {
            if ((top >> i) & 1) chk ^= GEN[i];
        }
    }
    return chk;
}

static std::vector<std::uint8_t> bech32_hrp_expand(const std::string& hrp) {
    std::vector<std::uint8_t> ret;
    ret.reserve(hrp.size() * 2 + 1);
    for (char c : hrp) ret.push_back(static_cast<std::uint8_t>(c >> 5));
    ret.push_back(0);
    for (char c : hrp) ret.push_back(static_cast<std::uint8_t>(c & 31));
    return ret;
}

static bool convert_bits(std::vector<std::uint8_t>& out,
                          const std::uint8_t* data, std::size_t len,
                          int frombits, int tobits, bool pad) {
    int acc = 0;
    int bits = 0;
    int maxv = (1 << tobits) - 1;
    for (std::size_t i = 0; i < len; ++i) {
        int value = data[i];
        if (value < 0 || value >> frombits) return false;
        acc = (acc << frombits) | value;
        bits += frombits;
        while (bits >= tobits) {
            bits -= tobits;
            out.push_back(static_cast<std::uint8_t>((acc >> bits) & maxv));
        }
    }
    if (pad) {
        if (bits > 0) {
            out.push_back(static_cast<std::uint8_t>((acc << (tobits - bits)) & maxv));
        }
    } else if (bits >= frombits || ((acc << (tobits - bits)) & maxv)) {
        return false;
    }
    return true;
}

std::string bech32_encode(const std::string& hrp,
                          std::uint8_t witness_version,
                          const std::uint8_t* witness_program,
                          std::size_t prog_len) {
    // Determine encoding: v0 = BECH32, v1+ = BECH32M
    std::uint32_t encoding_const = (witness_version == 0) ? 1u : 0x2bc830a3u;

    // Convert 8-bit data to 5-bit groups
    std::vector<std::uint8_t> data5;
    data5.push_back(witness_version);
    convert_bits(data5, witness_program, prog_len, 8, 5, true);

    // Compute checksum
    auto hrp_exp = bech32_hrp_expand(hrp);
    std::vector<std::uint8_t> values(hrp_exp);
    values.insert(values.end(), data5.begin(), data5.end());
    values.resize(values.size() + 6, 0);
    std::uint32_t polymod = bech32_polymod(values) ^ encoding_const;

    // Build result
    std::string result = hrp + "1";
    for (auto v : data5) result.push_back(BECH32_CHARSET[v]);
    for (int i = 0; i < 6; ++i) {
        result.push_back(BECH32_CHARSET[(polymod >> (5 * (5 - i))) & 31]);
    }

    return result;
}

Bech32DecodeResult bech32_decode(const std::string& addr) {
    Bech32DecodeResult result;
    result.valid = false;
    result.witness_version = -1;

    // Find separator '1'
    auto sep = addr.rfind('1');
    if (sep == std::string::npos || sep < 1 || sep + 7 > addr.size()) return result;

    std::string hrp_str;
    for (std::size_t i = 0; i < sep; ++i) {
        char c = addr[i];
        if (c < 33 || c > 126) return result;
        hrp_str.push_back(static_cast<char>(c >= 'A' && c <= 'Z' ? c + 32 : c));
    }

    // Decode data part
    std::vector<std::uint8_t> data5;
    for (std::size_t i = sep + 1; i < addr.size(); ++i) {
        char c = addr[i];
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + 32);
        int val = bech32_charset_value(c);
        if (val < 0) return result;
        data5.push_back(static_cast<std::uint8_t>(val));
    }

    if (data5.size() < 6) return result;

    // Verify checksum
    auto hrp_exp = bech32_hrp_expand(hrp_str);
    std::vector<std::uint8_t> values(hrp_exp);
    values.insert(values.end(), data5.begin(), data5.end());
    std::uint32_t polymod = bech32_polymod(values);

    Bech32Encoding enc;
    if (polymod == 1) enc = Bech32Encoding::BECH32;
    else if (polymod == 0x2bc830a3u) enc = Bech32Encoding::BECH32M;
    else return result;

    // Extract witness version and program
    std::uint8_t wit_ver = data5[0];
    if (wit_ver > 16) return result;
    if (wit_ver == 0 && enc != Bech32Encoding::BECH32) return result;
    if (wit_ver != 0 && enc != Bech32Encoding::BECH32M) return result;

    std::vector<std::uint8_t> prog;
    if (!convert_bits(prog, data5.data() + 1, data5.size() - 7, 5, 8, false)) return result;

    if (prog.size() < 2 || prog.size() > 40) return result;
    if (wit_ver == 0 && prog.size() != 20 && prog.size() != 32) return result;

    result.hrp = hrp_str;
    result.witness_version = wit_ver;
    result.witness_program = std::move(prog);
    result.valid = true;
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Address Derivation
// ═══════════════════════════════════════════════════════════════════════════════

std::string address_p2pkh(const Point& pubkey, Network net) {
    auto compressed = pubkey.to_compressed();
    auto h160 = hash160(compressed.data(), 33);

    // Version byte + hash160
    std::uint8_t payload[21];
    payload[0] = (net == Network::Mainnet) ? 0x00 : 0x6F;
    std::memcpy(payload + 1, h160.data(), 20);

    return base58check_encode(payload, 21);
}

std::string address_p2wpkh(const Point& pubkey, Network net) {
    auto compressed = pubkey.to_compressed();
    auto h160 = hash160(compressed.data(), 33);

    std::string hrp = (net == Network::Mainnet) ? "bc" : "tb";
    return bech32_encode(hrp, 0, h160.data(), 20);
}

std::string address_p2tr(const Point& internal_key, Network net) {
    // For keypath-only spend: output_key = internal_key (no tweak)
    // A proper Taproot output key uses taproot_output_key() from taproot.hpp
    // Here we just encode the x-only key
    auto x_bytes = internal_key.x().to_bytes();
    return address_p2tr_raw(x_bytes, net);
}

std::string address_p2tr_raw(const std::array<std::uint8_t, 32>& output_key_x,
                             Network net) {
    std::string hrp = (net == Network::Mainnet) ? "bc" : "tb";
    return bech32_encode(hrp, 1, output_key_x.data(), 32);
}

// ═══════════════════════════════════════════════════════════════════════════════
// WIF (Wallet Import Format)
// ═══════════════════════════════════════════════════════════════════════════════

std::string wif_encode(const Scalar& private_key, bool compressed, Network net) {
    auto key_bytes = private_key.to_bytes();
    std::size_t payload_len = compressed ? 34 : 33;
    std::vector<std::uint8_t> payload(payload_len);

    payload[0] = (net == Network::Mainnet) ? 0x80 : 0xEF;
    std::memcpy(payload.data() + 1, key_bytes.data(), 32);
    if (compressed) payload[33] = 0x01;

    return base58check_encode(payload.data(), payload_len);
}

WIFDecodeResult wif_decode(const std::string& wif) {
    auto [data, valid] = base58check_decode(wif);
    WIFDecodeResult result{};
    result.valid = false;

    if (!valid || data.empty()) return result;

    std::uint8_t version = data[0];
    if (version != 0x80 && version != 0xEF) return result;

    result.network = (version == 0x80) ? Network::Mainnet : Network::Testnet;

    if (data.size() == 34 && data[33] == 0x01) {
        result.compressed = true;
    } else if (data.size() == 33) {
        result.compressed = false;
    } else {
        return result;
    }

    std::array<std::uint8_t, 32> key_bytes;
    std::memcpy(key_bytes.data(), data.data() + 1, 32);
    result.key = Scalar::from_bytes(key_bytes);
    result.valid = true;
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════════
// BIP-352 Silent Payments
// ═══════════════════════════════════════════════════════════════════════════════

// Helper: lift_x with even y (try-and-increment)
static Point lift_x_even(const FieldElement& x_in) {
    FieldElement x = x_in;
    auto exp = FieldElement::from_hex(
        "3fffffffffffffffffffffffffffffffffffffffffffffffffffffffbfffff0c");
    for (int attempt = 0; attempt < 256; ++attempt) {
        FieldElement x2 = x * x;
        FieldElement x3 = x2 * x;
        FieldElement rhs = x3 + FieldElement::from_uint64(7);
        // sqrt via exp: y = rhs^((p+1)/4), p ≡ 3 (mod 4)
        auto y = FieldElement::one();
        auto base = rhs;
        auto exp_bytes = exp.to_bytes();
        for (int i = 0; i < 256; ++i) {
            y = y.square();
            int byte_idx = i / 8;
            int bit_idx = 7 - (i % 8);
            if ((exp_bytes[byte_idx] >> bit_idx) & 1) {
                y = y * base;
            }
        }
        if (y.square() == rhs) {
            auto y_bytes = y.to_bytes();
            if (y_bytes[31] & 1) y = FieldElement::zero() - y;
            return Point::from_affine(x, y);
        }
        x = x + FieldElement::one();
    }
    return Point::infinity();
}

SilentPaymentAddress
silent_payment_address(const Scalar& scan_privkey,
                       const Scalar& spend_privkey) {
    SilentPaymentAddress addr;
    addr.scan_pubkey = Point::generator().scalar_mul(scan_privkey);
    addr.spend_pubkey = Point::generator().scalar_mul(spend_privkey);
    return addr;
}

std::string SilentPaymentAddress::encode(Network net) const {
    // BIP-352 silent payment address format:
    // sp1q + scan_pubkey_x(32) + spend_pubkey_x(32) → bech32m
    auto scan_x = scan_pubkey.x().to_bytes();
    auto spend_x = spend_pubkey.x().to_bytes();

    // Concatenate scan_x || spend_x
    std::uint8_t data[64];
    std::memcpy(data, scan_x.data(), 32);
    std::memcpy(data + 32, spend_x.data(), 32);

    std::string hrp = (net == Network::Mainnet) ? "sp" : "tsp";
    // Use witness version 1 (Bech32m) for silent payments 
    // Note: BIP-352 uses a custom HRP, not standard witness program
    // For simplicity, we encode as bech32m with witness version 0
    // Real BIP-352 uses a dedicated encoding
    return bech32_encode(hrp, 1, data, 64);
}

std::pair<Point, Scalar>
silent_payment_create_output(const std::vector<Scalar>& input_privkeys,
                             const SilentPaymentAddress& recipient,
                             std::uint32_t k) {
    // Sum of input private keys: a = Σ a_i
    Scalar a_sum = Scalar::zero();
    for (const auto& a : input_privkeys) {
        a_sum = a_sum + a;
    }

    // Shared secret: S = a_sum * B_scan
    Point S = recipient.scan_pubkey.scalar_mul(a_sum);

    // t_k = SHA256(tagged_hash("BIP0352/SharedSecret", ser(S)) || ser32(k))
    auto S_comp = S.to_compressed();
    
    SHA256 h;
    // Tagged hash
    auto tag_hash = SHA256::hash(reinterpret_cast<const std::uint8_t*>("BIP0352/SharedSecret"), 20);
    h.update(tag_hash.data(), 32);
    h.update(tag_hash.data(), 32);
    h.update(S_comp.data(), 33);
    std::uint8_t k_be[4] = {
        std::uint8_t(k >> 24), std::uint8_t(k >> 16),
        std::uint8_t(k >> 8), std::uint8_t(k)
    };
    h.update(k_be, 4);
    auto t_hash = h.finalize();
    Scalar t_k = Scalar::from_bytes(t_hash);

    // Output key: P_output = B_spend + t_k * G
    Point P_output = recipient.spend_pubkey.add(Point::generator().scalar_mul(t_k));

    return {P_output, t_k};
}

std::vector<std::pair<std::uint32_t, Scalar>>
silent_payment_scan(const Scalar& scan_privkey,
                    const Scalar& spend_privkey,
                    const std::vector<Point>& input_pubkeys,
                    const std::vector<std::array<std::uint8_t, 32>>& output_pubkeys) {
    std::vector<std::pair<std::uint32_t, Scalar>> results;

    // Sum of input public keys: A = Σ A_i
    Point A_sum = Point::infinity();
    for (const auto& A : input_pubkeys) {
        A_sum = A_sum.add(A);
    }

    // Shared secret: S = b_scan * A_sum
    Point S = A_sum.scalar_mul(scan_privkey);

    // Check each output
    for (std::uint32_t k = 0; k < static_cast<std::uint32_t>(output_pubkeys.size()); ++k) {
        // t_k = tagged_hash("BIP0352/SharedSecret", ser(S) || ser32(k))
        auto S_comp = S.to_compressed();

        SHA256 h;
        auto tag_hash = SHA256::hash(reinterpret_cast<const std::uint8_t*>("BIP0352/SharedSecret"), 20);
        h.update(tag_hash.data(), 32);
        h.update(tag_hash.data(), 32);
        h.update(S_comp.data(), 33);
        std::uint8_t k_be[4] = {
            std::uint8_t(k >> 24), std::uint8_t(k >> 16),
            std::uint8_t(k >> 8), std::uint8_t(k)
        };
        h.update(k_be, 4);
        auto t_hash = h.finalize();
        Scalar t_k = Scalar::from_bytes(t_hash);

        // Expected output: P = B_spend + t_k * G
        Point B_spend = Point::generator().scalar_mul(spend_privkey);
        Point expected = B_spend.add(Point::generator().scalar_mul(t_k));
        auto expected_x = expected.x().to_bytes();

        // Compare x-coordinate
        if (expected_x == output_pubkeys[k]) {
            // Compute spending private key: d = b_spend + t_k
            Scalar d = spend_privkey + t_k;
            results.push_back({k, d});
        }
    }

    return results;
}

} // namespace secp256k1
