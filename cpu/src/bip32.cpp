// ============================================================================
// BIP-32: Hierarchical Deterministic Key Derivation
// ============================================================================

#include "secp256k1/bip32.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/sha512.hpp"
#include <cstring>
#include <cctype>

namespace secp256k1 {

using fast::Scalar;
using fast::Point;
using fast::FieldElement;

// -- HMAC-SHA512 --------------------------------------------------------------

std::array<std::uint8_t, 64> hmac_sha512(const uint8_t* key, std::size_t key_len,
                                          const uint8_t* data, std::size_t data_len) {
    uint8_t k_buf[128]{};

    if (key_len > 128) {
        auto h = SHA512::hash(key, key_len);
        std::memcpy(k_buf, h.data(), 64);
    } else {
        std::memcpy(k_buf, key, key_len);
    }

    uint8_t ipad[128], opad[128];
    for (int i = 0; i < 128; ++i) {
        ipad[i] = k_buf[i] ^ 0x36;
        opad[i] = k_buf[i] ^ 0x5c;
    }

    // inner = SHA512(ipad || data)
    SHA512 inner;
    inner.update(ipad, 128);
    inner.update(data, data_len);
    auto inner_hash = inner.finalize();

    // outer = SHA512(opad || inner_hash)
    SHA512 outer;
    outer.update(opad, 128);
    outer.update(inner_hash.data(), 64);
    return outer.finalize();
}

// -- RIPEMD-160 (minimal) -----------------------------------------------------
// Needed for HASH160 = RIPEMD160(SHA256(data))

namespace {

class RIPEMD160 {
public:
    static std::array<uint8_t, 20> hash(const void* data, std::size_t len) {
        RIPEMD160 ctx;
        ctx.update(static_cast<const uint8_t*>(data), len);
        return ctx.finalize();
    }

private:
    uint32_t h_[5];
    uint64_t total_;
    uint8_t buf_[64];
    size_t buf_len_;

    RIPEMD160() {
        h_[0] = 0x67452301u; h_[1] = 0xEFCDAB89u;
        h_[2] = 0x98BADCFEu; h_[3] = 0x10325476u;
        h_[4] = 0xC3D2E1F0u;
        total_ = 0;
        buf_len_ = 0;
    }

    void update(const uint8_t* data, size_t len) {
        total_ += len;
        if (buf_len_ > 0) {
            size_t fill = 64 - buf_len_;
            if (len < fill) {
                std::memcpy(buf_ + buf_len_, data, len);
                buf_len_ += len;
                return;
            }
            std::memcpy(buf_ + buf_len_, data, fill);
            compress(buf_);
            data += fill;
            len -= fill;
            buf_len_ = 0;
        }
        while (len >= 64) {
            compress(data);
            data += 64;
            len -= 64;
        }
        if (len > 0) {
            std::memcpy(buf_, data, len);
            buf_len_ = len;
        }
    }

    std::array<uint8_t, 20> finalize() {
        uint64_t bit_len = total_ * 8;
        uint8_t pad = 0x80;
        update(&pad, 1);
        while (buf_len_ != 56) {
            uint8_t z = 0;
            update(&z, 1);
        }
        uint8_t len_buf[8];
        for (std::size_t i = 0; i < 8; ++i)
            len_buf[i] = static_cast<uint8_t>(bit_len >> (i * 8));
        update(len_buf, 8);

        std::array<uint8_t, 20> d{};
        for (std::size_t i = 0; i < 5; ++i) {
            d[i * 4 + 0] = static_cast<uint8_t>(h_[i]);
            d[i * 4 + 1] = static_cast<uint8_t>(h_[i] >> 8);
            d[i * 4 + 2] = static_cast<uint8_t>(h_[i] >> 16);
            d[i * 4 + 3] = static_cast<uint8_t>(h_[i] >> 24);
        }
        return d;
    }

    static uint32_t rotl(uint32_t x, unsigned n) { return (x << n) | (x >> (32 - n)); }

    static uint32_t f(unsigned j, uint32_t x, uint32_t y, uint32_t z) {
        if (j < 16)      return x ^ y ^ z;
        else if (j < 32) return (x & y) | (~x & z);
        else if (j < 48) return (x | ~y) ^ z;
        else if (j < 64) return (x & z) | (y & ~z);
        else              return x ^ (y | ~z);
    }

    static uint32_t K_left(unsigned j) {
        if (j < 16)      return 0x00000000u;
        else if (j < 32) return 0x5A827999u;
        else if (j < 48) return 0x6ED9EBA1u;
        else if (j < 64) return 0x8F1BBCDCu;
        else              return 0xA953FD4Eu;
    }

    static uint32_t K_right(unsigned j) {
        if (j < 16)      return 0x50A28BE6u;
        else if (j < 32) return 0x5C4DD124u;
        else if (j < 48) return 0x6D703EF3u;
        else if (j < 64) return 0x7A6D76E9u;
        else              return 0x00000000u;
    }

    void compress(const uint8_t* block) {
        static constexpr uint8_t RL[80] = {
            0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
            7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
            3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
            1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
            4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13
        };
        static constexpr uint8_t RR[80] = {
            5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
            6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
            15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,
            8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
            12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11
        };
        static constexpr uint8_t SL[80] = {
            11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
            7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
            11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
            11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
            9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6
        };
        static constexpr uint8_t SR[80] = {
            8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
            9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
            9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
            15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
            8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11
        };

        uint32_t X[16];
        for (int i = 0; i < 16; ++i) {
            X[i] = static_cast<uint32_t>(block[i * 4]) |
                   (static_cast<uint32_t>(block[i * 4 + 1]) << 8) |
                   (static_cast<uint32_t>(block[i * 4 + 2]) << 16) |
                   (static_cast<uint32_t>(block[i * 4 + 3]) << 24);
        }

        uint32_t al = h_[0], bl = h_[1], cl = h_[2], dl = h_[3], el = h_[4];
        uint32_t ar = h_[0], br = h_[1], cr = h_[2], dr = h_[3], er = h_[4];

        for (unsigned j = 0; j < 80; ++j) {
            uint32_t tl = rotl(al + f(j, bl, cl, dl) + X[RL[j]] + K_left(j), SL[j]) + el;
            al = el; el = dl; dl = rotl(cl, 10); cl = bl; bl = tl;

            uint32_t tr = rotl(ar + f(79 - j, br, cr, dr) + X[RR[j]] + K_right(j), SR[j]) + er;
            ar = er; er = dr; dr = rotl(cr, 10); cr = br; br = tr;
        }

        uint32_t t = h_[1] + cl + dr;
        h_[1] = h_[2] + dl + er;
        h_[2] = h_[3] + el + ar;
        h_[3] = h_[4] + al + br;
        h_[4] = h_[0] + bl + cr;
        h_[0] = t;
    }
};

// HASH160 = RIPEMD160(SHA256(data))
std::array<uint8_t, 20> hash160(const void* data, std::size_t len) {
    auto sha = SHA256::hash(data, len);
    return RIPEMD160::hash(sha.data(), 32);
}

} // anonymous namespace

// -- ExtendedKey --------------------------------------------------------------

fast::Point ExtendedKey::public_key() const {
    if (is_private) {
        auto sk = Scalar::from_bytes(key);
        return Point::generator().scalar_mul(sk);
    }
    // Public key: decompress from pub_prefix + key (x-coordinate)
    // y^2 = x^3 + 7, then pick y matching parity
    auto x = fast::FieldElement::from_bytes(key);
    auto x2 = x * x;
    auto x3 = x2 * x;
    auto seven = fast::FieldElement::from_uint64(7);
    auto y2 = x3 + seven;
    auto y = y2.sqrt();
    // Check parity: prefix 0x02 = even y, 0x03 = odd y
    auto y_bytes = y.to_bytes();
    bool y_is_odd = (y_bytes[31] & 1) != 0;
    bool need_odd = (pub_prefix == 0x03);
    if (y_is_odd != need_odd) {
        y = y.negate();
    }
    return Point::from_affine(x, y);
}

fast::Scalar ExtendedKey::private_key() const {
    return Scalar::from_bytes(key);
}

ExtendedKey ExtendedKey::to_public() const {
    if (!is_private) return *this;

    ExtendedKey pub{};
    auto pk = public_key();
    auto compressed = pk.to_compressed();
    // Store prefix (0x02 or 0x03) and x-coordinate separately
    pub.pub_prefix = compressed[0];
    auto x_bytes = pk.x().to_bytes();
    pub.key = x_bytes;
    pub.chain_code = chain_code;
    pub.depth = depth;
    pub.child_number = child_number;
    pub.parent_fingerprint = parent_fingerprint;
    pub.is_private = false;
    return pub;
}

std::array<uint8_t, 4> ExtendedKey::fingerprint() const {
    auto pk = public_key();
    auto compressed = pk.to_compressed();
    auto h = hash160(compressed.data(), 33);
    std::array<uint8_t, 4> fp{};
    std::memcpy(fp.data(), h.data(), 4);
    return fp;
}

std::array<uint8_t, 78> ExtendedKey::serialize() const {
    std::array<uint8_t, 78> out{};
    // Version bytes
    uint32_t version = is_private ? 0x0488ADE4u : 0x0488B21Eu;  // xprv / xpub
    out[0] = static_cast<uint8_t>(version >> 24);
    out[1] = static_cast<uint8_t>(version >> 16);
    out[2] = static_cast<uint8_t>(version >> 8);
    out[3] = static_cast<uint8_t>(version);
    out[4] = depth;
    std::memcpy(out.data() + 5, parent_fingerprint.data(), 4);
    out[9] = static_cast<uint8_t>(child_number >> 24);
    out[10] = static_cast<uint8_t>(child_number >> 16);
    out[11] = static_cast<uint8_t>(child_number >> 8);
    out[12] = static_cast<uint8_t>(child_number);
    std::memcpy(out.data() + 13, chain_code.data(), 32);

    if (is_private) {
        out[45] = 0x00;
        std::memcpy(out.data() + 46, key.data(), 32);
    } else {
        // Reconstruct compressed pubkey from prefix + x-coordinate
        out[45] = pub_prefix;
        std::memcpy(out.data() + 46, key.data(), 32);
    }
    return out;
}

std::pair<ExtendedKey, bool> ExtendedKey::derive_child(uint32_t index) const {
    bool hardened = (index & 0x80000000u) != 0;

    // Hardened derivation requires private key
    if (hardened && !is_private) {
        return {ExtendedKey{}, false};
    }

    // Data = [compressed_pubkey(33) || index(4)] for normal
    // Data = [0x00 || private_key(32) || index(4)] for hardened
    uint8_t data[37];
    if (hardened) {
        data[0] = 0x00;
        std::memcpy(data + 1, key.data(), 32);
    } else {
        auto pk = public_key();
        auto compressed = pk.to_compressed();
        std::memcpy(data, compressed.data(), 33);
    }
    // Append index (big-endian)
    data[33] = static_cast<uint8_t>(index >> 24);
    data[34] = static_cast<uint8_t>(index >> 16);
    data[35] = static_cast<uint8_t>(index >> 8);
    data[36] = static_cast<uint8_t>(index);

    auto I = hmac_sha512(chain_code.data(), 32, data, hardened ? 37 : 37);

    std::array<uint8_t, 32> IL{}, IR{};
    std::memcpy(IL.data(), I.data(), 32);
    std::memcpy(IR.data(), I.data() + 32, 32);

    auto il_scalar = Scalar::from_bytes(IL);
    // Check IL is valid (less than curve order -- from_bytes handles mod n)
    // But we need to check it wasn't zero after reduction when original was >= n
    // from_bytes already reduces mod n, so just check non-zero
    if (il_scalar.is_zero()) return {ExtendedKey{}, false};

    ExtendedKey child{};
    child.chain_code = IR;
    child.depth = static_cast<uint8_t>(depth + 1);
    child.child_number = index;
    child.parent_fingerprint = fingerprint();

    if (is_private) {
        // child_key = (IL + parent_key) mod n
        auto parent_scalar = Scalar::from_bytes(key);
        auto child_scalar = il_scalar + parent_scalar;
        if (child_scalar.is_zero()) return {ExtendedKey{}, false};
        child.key = child_scalar.to_bytes();
        child.is_private = true;
    } else {
        // child_key = point(IL) + parent_pubkey
        auto IL_point = Point::generator().scalar_mul(il_scalar);
        auto parent_point = public_key();
        auto child_point = IL_point.add(parent_point);
        if (child_point.is_infinity()) return {ExtendedKey{}, false};
        auto compressed = child_point.to_compressed();
        child.pub_prefix = compressed[0];
        child.key = child_point.x().to_bytes();
        child.is_private = false;
    }

    return {child, true};
}

// -- Master Key ---------------------------------------------------------------

std::pair<ExtendedKey, bool> bip32_master_key(const uint8_t* seed, std::size_t seed_len) {
    if (seed_len < 16 || seed_len > 64) return {ExtendedKey{}, false};

    // I = HMAC-SHA512(Key="Bitcoin seed", Data=Seed)
    const char* hmac_key = "Bitcoin seed";
    auto I = hmac_sha512(reinterpret_cast<const uint8_t*>(hmac_key), 12,
                          seed, seed_len);

    std::array<uint8_t, 32> IL{}, IR{};
    std::memcpy(IL.data(), I.data(), 32);
    std::memcpy(IR.data(), I.data() + 32, 32);

    auto master_key = Scalar::from_bytes(IL);
    if (master_key.is_zero()) return {ExtendedKey{}, false};

    ExtendedKey ext{};
    ext.key = master_key.to_bytes();
    ext.chain_code = IR;
    ext.depth = 0;
    ext.child_number = 0;
    ext.parent_fingerprint = {0, 0, 0, 0};
    ext.is_private = true;

    return {ext, true};
}

// -- Path Derivation ----------------------------------------------------------

std::pair<ExtendedKey, bool> bip32_derive_path(const ExtendedKey& master,
                                                const std::string& path) {
    // Format: "m/44'/0'/0'/0/0"
    if (path.empty() || path[0] != 'm') return {ExtendedKey{}, false};

    ExtendedKey current = master;
    std::size_t pos = 1; // skip 'm'

    while (pos < path.size()) {
        if (path[pos] == '/') {
            ++pos;
            continue;
        }

        // Parse number
        uint32_t index = 0;
        bool has_digit = false;
        while (pos < path.size() && std::isdigit(static_cast<unsigned char>(path[pos]))) {
            index = index * 10 + static_cast<uint32_t>(path[pos] - '0');
            ++pos;
            has_digit = true;
        }
        if (!has_digit) return {ExtendedKey{}, false};

        // Check for hardened marker
        bool hardened = false;
        if (pos < path.size() && (path[pos] == '\'' || path[pos] == 'h' || path[pos] == 'H')) {
            hardened = true;
            ++pos;
        }

        uint32_t child_index = hardened ? (index | 0x80000000u) : index;
        auto [child, ok] = current.derive_child(child_index);
        if (!ok) return {ExtendedKey{}, false};
        current = child;
    }

    return {current, true};
}

} // namespace secp256k1
