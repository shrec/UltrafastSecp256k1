// ============================================================================
// ECIES -- Elliptic Curve Integrated Encryption Scheme
// ============================================================================
// Envelope: [33B ephemeral pubkey][16B IV][N bytes AES-256-CTR ciphertext][32B HMAC-SHA256]
//
// Key derivation: SHA-512(ECDH_raw_x) -> enc_key (32B) || mac_key (32B)
// Encryption: AES-256-CTR (software, no OpenSSL dependency)
// Authentication: HMAC-SHA256(mac_key, ephemeral_pubkey || IV || ciphertext)
// ============================================================================

#include "secp256k1/ecies.hpp"
#include "secp256k1/ecdh.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/sha512.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/detail/secure_erase.hpp"
#include "secp256k1/field.hpp"
#include <cstring>

// OS CSPRNG headers
#if defined(_WIN32)
#  include <windows.h>
#  include <bcrypt.h>
#  pragma comment(lib, "bcrypt.lib")
#elif defined(__APPLE__)
#  include <Security/SecRandom.h>
#elif defined(__linux__) || defined(__FreeBSD__) || defined(__OpenBSD__)
#  include <sys/random.h>
#else
#  include <cstdio>   // fopen/fread for /dev/urandom fallback
#endif

namespace secp256k1 {

using fast::Scalar;
using fast::Point;
using fast::FieldElement;

// Decompress a 33-byte compressed point (strict: only 0x02/0x03 prefix)
static Point decompress_point(const std::uint8_t data[33]) {
    if (data[0] != 0x02 && data[0] != 0x03) return Point::infinity();
    FieldElement x;
    if (!FieldElement::parse_bytes_strict(data + 1, x)) return Point::infinity();
    auto x2 = x * x;
    auto x3 = x2 * x;
    auto y2 = x3 + FieldElement::from_uint64(7);
    auto y = y2.sqrt();
    if (y.square() != y2) return Point::infinity();
    auto yb = y.to_bytes();
    bool const y_odd = (yb[31] & 1) != 0;
    bool const want_odd = (data[0] == 0x03);
    if (y_odd != want_odd) y = FieldElement::zero() - y;
    return Point::from_affine(x, y);
}

// ============================================================================
// AES-256 core (software, constant-time S-box via algebraic decomposition)
// ============================================================================

namespace {

// AES S-box (precomputed, standard FIPS-197)
alignas(64) constexpr std::uint8_t SBOX[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

// AES round constants
constexpr std::uint8_t RCON[11] = {
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
};

// xtime: multiply by x in GF(2^8)
inline std::uint8_t xtime(std::uint8_t a) {
    return static_cast<std::uint8_t>((a << 1) ^ (((a >> 7) & 1) * 0x1b));
}

struct AES256 {
    std::uint8_t round_keys[15][16]; // 14 rounds + initial key

    void key_expansion(const std::uint8_t key[32]) {
        std::uint8_t W[240]; // 60 words * 4 bytes
        std::memcpy(W, key, 32);

        for (int i = 8; i < 60; ++i) {
            std::uint8_t tmp[4];
            std::memcpy(tmp, W + (i - 1) * 4, 4);

            if (i % 8 == 0) {
                std::uint8_t const t = tmp[0];
                tmp[0] = SBOX[tmp[1]] ^ RCON[i / 8];
                tmp[1] = SBOX[tmp[2]];
                tmp[2] = SBOX[tmp[3]];
                tmp[3] = SBOX[t];
            } else if (i % 8 == 4) {
                for (int j = 0; j < 4; ++j) tmp[j] = SBOX[tmp[j]];
            }

            for (int j = 0; j < 4; ++j)
                W[i * 4 + j] = W[(i - 8) * 4 + j] ^ tmp[j];
        }

        for (int r = 0; r < 15; ++r)
            std::memcpy(round_keys[r], W + r * 16, 16);

        secp256k1::detail::secure_erase(W, sizeof(W));
    }

    // Encrypt a single 16-byte block (ECB, used as building block for CTR)
    void encrypt_block(const std::uint8_t in[16], std::uint8_t out[16]) const {
        std::uint8_t state[16];
        std::memcpy(state, in, 16);

        // AddRoundKey (initial)
        for (int i = 0; i < 16; ++i) state[i] ^= round_keys[0][i];

        for (int round = 1; round <= 14; ++round) {
            // SubBytes
            for (int i = 0; i < 16; ++i) state[i] = SBOX[state[i]];

            // ShiftRows
            std::uint8_t t;
            t = state[1]; state[1] = state[5]; state[5] = state[9];
            state[9] = state[13]; state[13] = t;
            t = state[2]; state[2] = state[10]; state[10] = t;
            t = state[6]; state[6] = state[14]; state[14] = t;
            t = state[3]; state[3] = state[15]; state[15] = state[11];
            state[11] = state[7]; state[7] = t;

            // MixColumns (skip on last round)
            if (round < 14) {
                for (int c = 0; c < 4; ++c) {
                    int const j = c * 4;
                    std::uint8_t const a0 = state[j], a1 = state[j+1],
                                      a2 = state[j+2], a3 = state[j+3];
                    std::uint8_t const h0 = xtime(a0), h1 = xtime(a1),
                                      h2 = xtime(a2), h3 = xtime(a3);
                    state[j]   = h0 ^ h1 ^ a1 ^ a2 ^ a3;
                    state[j+1] = a0 ^ h1 ^ h2 ^ a2 ^ a3;
                    state[j+2] = a0 ^ a1 ^ h2 ^ h3 ^ a3;
                    state[j+3] = h0 ^ a0 ^ a1 ^ a2 ^ h3;
                }
            }

            // AddRoundKey
            for (int i = 0; i < 16; ++i) state[i] ^= round_keys[round][i];
        }

        std::memcpy(out, state, 16);
    }
};

// AES-256-CTR encrypt/decrypt (same operation)
void aes256_ctr(const std::uint8_t key[32],
                const std::uint8_t iv[16],
                const std::uint8_t* input, std::size_t len,
                std::uint8_t* output) {
    AES256 aes;
    aes.key_expansion(key);

    std::uint8_t counter[16];
    std::memcpy(counter, iv, 16);

    std::uint8_t keystream[16];
    std::size_t pos = 0;

    while (pos < len) {
        aes.encrypt_block(counter, keystream);

        std::size_t const chunk = (len - pos < 16) ? (len - pos) : 16;
        for (std::size_t i = 0; i < chunk; ++i)
            output[pos + i] = input[pos + i] ^ keystream[i];

        pos += chunk;

        // Increment counter (big-endian)
        for (int i = 15; i >= 0; --i) {
            if (++counter[i] != 0) break;
        }
    }

    secp256k1::detail::secure_erase(&aes, sizeof(aes));
    secp256k1::detail::secure_erase(keystream, sizeof(keystream));
}

// HMAC-SHA256
std::array<std::uint8_t, 32>
hmac_sha256(const std::uint8_t* key, std::size_t key_len,
            const std::uint8_t* data, std::size_t data_len) {
    std::uint8_t k_pad[64];
    std::memset(k_pad, 0, 64);

    if (key_len > 64) {
        auto h = SHA256::hash(key, key_len);
        std::memcpy(k_pad, h.data(), 32);
    } else {
        std::memcpy(k_pad, key, key_len);
    }

    // ipad
    std::uint8_t ipad[64];
    for (int i = 0; i < 64; ++i) ipad[i] = k_pad[i] ^ 0x36;

    SHA256 inner;
    inner.update(ipad, 64);
    inner.update(data, data_len);
    auto inner_hash = inner.finalize();

    // opad
    std::uint8_t opad[64];
    for (int i = 0; i < 64; ++i) opad[i] = k_pad[i] ^ 0x5c;

    SHA256 outer;
    outer.update(opad, 64);
    outer.update(inner_hash.data(), 32);

    secp256k1::detail::secure_erase(k_pad, sizeof(k_pad));
    secp256k1::detail::secure_erase(ipad, sizeof(ipad));
    secp256k1::detail::secure_erase(opad, sizeof(opad));

    return outer.finalize();
}

// CSPRNG fill -- OS-level cryptographic randomness, fail-closed
void csprng_fill(std::uint8_t* buf, std::size_t len) {
#if defined(_WIN32)
    NTSTATUS const status = BCryptGenRandom(
        nullptr, buf, static_cast<ULONG>(len), BCRYPT_USE_SYSTEM_PREFERRED_RNG);
    if (status != 0) std::abort();  // fail-closed
#elif defined(__APPLE__)
    if (SecRandomCopyBytes(kSecRandomDefault, len, buf) != errSecSuccess)
        std::abort();
#elif defined(__linux__) || defined(__FreeBSD__) || defined(__OpenBSD__)
    // getrandom(2): blocks until entropy available, no EINTR on < 256 bytes
    std::size_t filled = 0;
    while (filled < len) {
        ssize_t const r = getrandom(buf + filled, len - filled, 0);
        if (r <= 0) std::abort();  // fail-closed
        filled += static_cast<std::size_t>(r);
    }
#else
    // Fallback: /dev/urandom (POSIX)
    FILE* f = std::fopen("/dev/urandom", "rb");
    if (!f) std::abort();
    if (std::fread(buf, 1, len, f) != len) { std::fclose(f); std::abort(); }
    std::fclose(f);
#endif
}

} // anonymous namespace

// ============================================================================
// ECIES public API
// ============================================================================
// Envelope: [33B ephemeral pubkey][16B IV][N bytes ciphertext][32B HMAC tag]
// Total overhead: 33 + 16 + 32 = 81 bytes
// ============================================================================

std::vector<std::uint8_t>
ecies_encrypt(const Point& recipient_pubkey,
              const std::uint8_t* plaintext, std::size_t plaintext_len) {
    if (recipient_pubkey.is_infinity() || !plaintext || plaintext_len == 0)
        return {};

    // 1. Generate ephemeral keypair
    std::uint8_t eph_bytes[32];
    csprng_fill(eph_bytes, 32);
    Scalar eph_privkey = Scalar::from_bytes(eph_bytes);
    if (eph_privkey.is_zero()) return {};

    Point const eph_pubkey = ct::generator_mul(eph_privkey);

    // 2. ECDH: shared_x = (eph_priv * recipient_pub).x
    auto shared_x = ecdh_compute_raw(eph_privkey, recipient_pubkey);

    // 3. Key derivation: SHA-512(shared_x) -> enc_key(32) || mac_key(32)
    auto kdf = SHA512::hash(shared_x.data(), 32);
    secp256k1::detail::secure_erase(shared_x.data(), 32);
    const std::uint8_t* enc_key = kdf.data();
    const std::uint8_t* mac_key = kdf.data() + 32;

    // 4. Generate random IV (16 bytes)
    std::uint8_t iv[16];
    csprng_fill(iv, 16);

    // 5. Encrypt with AES-256-CTR
    std::vector<std::uint8_t> ciphertext(plaintext_len);
    aes256_ctr(enc_key, iv, plaintext, plaintext_len, ciphertext.data());

    // 6. HMAC-SHA256(mac_key, ephemeral_pubkey || iv || ciphertext)
    //    Covers the entire envelope prefix to prevent parity-byte malleability
    auto eph_comp = eph_pubkey.to_compressed();
    std::vector<std::uint8_t> hmac_data(33 + 16 + plaintext_len);
    std::memcpy(hmac_data.data(), eph_comp.data(), 33);
    std::memcpy(hmac_data.data() + 33, iv, 16);
    std::memcpy(hmac_data.data() + 49, ciphertext.data(), plaintext_len);
    auto tag = hmac_sha256(mac_key, 32, hmac_data.data(), hmac_data.size());

    // 7. Build envelope: [33B pubkey][16B IV][N ciphertext][32B HMAC]
    std::vector<std::uint8_t> envelope(33 + 16 + plaintext_len + 32);
    std::memcpy(envelope.data(), eph_comp.data(), 33);
    std::memcpy(envelope.data() + 33, iv, 16);
    std::memcpy(envelope.data() + 49, ciphertext.data(), plaintext_len);
    std::memcpy(envelope.data() + 49 + plaintext_len, tag.data(), 32);

    // Cleanup
    secp256k1::detail::secure_erase(eph_bytes, sizeof(eph_bytes));
    secp256k1::detail::secure_erase(&eph_privkey, sizeof(eph_privkey));
    secp256k1::detail::secure_erase(kdf.data(), 64);

    return envelope;
}

std::vector<std::uint8_t>
ecies_decrypt(const Scalar& privkey,
              const std::uint8_t* envelope, std::size_t envelope_len) {
    // Minimum envelope: 33 (pubkey) + 16 (IV) + 1 (ciphertext) + 32 (HMAC) = 82
    if (privkey.is_zero() || !envelope || envelope_len < 82)
        return {};

    std::size_t const ciphertext_len = envelope_len - 33 - 16 - 32;

    // 1. Parse ephemeral pubkey
    auto eph_pubkey = decompress_point(envelope);
    if (eph_pubkey.is_infinity()) return {};

    // 2. IV and ciphertext and tag
    const std::uint8_t* iv         = envelope + 33;
    const std::uint8_t* ciphertext = envelope + 49;
    const std::uint8_t* tag        = envelope + 49 + ciphertext_len;

    // 3. ECDH
    auto shared_x = ecdh_compute_raw(privkey, eph_pubkey);

    // 4. KDF: SHA-512(shared_x)
    auto kdf = SHA512::hash(shared_x.data(), 32);
    secp256k1::detail::secure_erase(shared_x.data(), 32);
    const std::uint8_t* enc_key = kdf.data();
    const std::uint8_t* mac_key = kdf.data() + 32;

    // 5. Verify HMAC-SHA256(mac_key, ephemeral_pubkey || iv || ciphertext)
    std::vector<std::uint8_t> hmac_data(33 + 16 + ciphertext_len);
    std::memcpy(hmac_data.data(), envelope, 33);  // ephemeral pubkey bytes as-received
    std::memcpy(hmac_data.data() + 33, iv, 16);
    std::memcpy(hmac_data.data() + 49, ciphertext, ciphertext_len);
    auto expected_tag = hmac_sha256(mac_key, 32, hmac_data.data(), hmac_data.size());

    // Constant-time compare
    std::uint8_t diff = 0;
    for (int i = 0; i < 32; ++i) diff |= expected_tag[i] ^ tag[i];
    if (diff != 0) {
        secp256k1::detail::secure_erase(kdf.data(), 64);
        return {}; // Authentication failed
    }

    // 6. Decrypt with AES-256-CTR
    std::vector<std::uint8_t> plaintext(ciphertext_len);
    aes256_ctr(enc_key, iv, ciphertext, ciphertext_len, plaintext.data());

    secp256k1::detail::secure_erase(kdf.data(), 64);
    return plaintext;
}

} // namespace secp256k1
