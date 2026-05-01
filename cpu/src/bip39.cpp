// BIP-39: Mnemonic Code for Generating Deterministic Keys
// Reference: https://github.com/bitcoin/bips/blob/master/bip-0039.mediawiki

#include "secp256k1/bip39.hpp"
#include "secp256k1/bip39_wordlist.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/bip32.hpp"  // hmac_sha512
#include "secp256k1/detail/secure_erase.hpp"
#include "secp256k1/unicode_nfkd.hpp"

#include <algorithm>
#include <cstring>
#include <sstream>

#if defined(_WIN32)
#  include <windows.h>
#  include <bcrypt.h>
#else
#  include <cstdio>
#endif

namespace secp256k1 {

// ---------------------------------------------------------------------------
// CSPRNG – fill buffer with cryptographically secure random bytes
// ---------------------------------------------------------------------------
static bool csprng_fill(uint8_t* buf, size_t len) {
#if defined(_WIN32)
    return BCryptGenRandom(nullptr, buf, static_cast<ULONG>(len),
                           BCRYPT_USE_SYSTEM_PREFERRED_RNG) == 0;
#else
    FILE* f = std::fopen("/dev/urandom", "rb");
    if (!f) return false;
    const bool ok = (std::fread(buf, 1, len, f) == len);
    (void)std::fclose(f);
    return ok;
#endif
}

// ---------------------------------------------------------------------------
// Wordlist helpers
// ---------------------------------------------------------------------------
static int word_index(const char* word) {
    // Binary search in the sorted BIP-39 english wordlist
    int lo = 0, hi = 2047;
    while (lo <= hi) {
        const int mid = lo + (hi - lo) / 2;
        const int cmp = std::strcmp(word, detail::bip39_english[mid]);
        if (cmp == 0) return mid;
        if (cmp < 0) {
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }
    return -1;
}

// Split mnemonic string into words
static std::vector<std::string> split_words(const std::string& mnemonic) {
    std::vector<std::string> words;
    std::istringstream iss(mnemonic);
    std::string w;
    while (iss >> w) words.push_back(w);
    return words;
}

// ---------------------------------------------------------------------------
// PBKDF2-HMAC-SHA512
// ---------------------------------------------------------------------------
void pbkdf2_hmac_sha512(const uint8_t* password, size_t password_len,
                         const uint8_t* salt, size_t salt_len,
                         uint32_t iterations,
                         uint8_t* output, size_t output_len) {
    // BIP-39 always uses output_len = 64 (one block), but we implement
    // the general multi-block version for correctness.
    uint32_t block_num = 1;
    size_t offset = 0;

    while (offset < output_len) {
        // U_1 = HMAC-SHA512(password, salt || INT_32_BE(block_num))
        // Build salt || INT_32_BE(block_num)
        std::vector<uint8_t> salt_block(salt_len + 4);
        std::memcpy(salt_block.data(), salt, salt_len);
        salt_block[salt_len + 0] = static_cast<uint8_t>((block_num >> 24) & 0xFF);
        salt_block[salt_len + 1] = static_cast<uint8_t>((block_num >> 16) & 0xFF);
        salt_block[salt_len + 2] = static_cast<uint8_t>((block_num >> 8) & 0xFF);
        salt_block[salt_len + 3] = static_cast<uint8_t>(block_num & 0xFF);

        auto u = hmac_sha512(password, password_len,
                             salt_block.data(), salt_block.size());

        std::array<uint8_t, 64> result = u;

        for (uint32_t i = 1; i < iterations; ++i) {
            u = hmac_sha512(password, password_len, u.data(), u.size());
            for (size_t j = 0; j < 64; ++j) {
                result[j] ^= u[j];
            }
        }

        const size_t to_copy = std::min<size_t>(64, output_len - offset);
        std::memcpy(output + offset, result.data(), to_copy);
        detail::secure_erase(result.data(), result.size());
        detail::secure_erase(u.data(), u.size());
        offset += to_copy;
        ++block_num;
    }
}

// ---------------------------------------------------------------------------
// bip39_generate
// ---------------------------------------------------------------------------
std::pair<std::string, bool>
bip39_generate(size_t entropy_bytes, const uint8_t* entropy_in) {
    // Valid entropy sizes: 16, 20, 24, 28, 32 bytes
    if (entropy_bytes < 16 || entropy_bytes > 32 || (entropy_bytes % 4) != 0) {
        return {"", false};
    }

    uint8_t entropy[32];
    if (entropy_in) {
        std::memcpy(entropy, entropy_in, entropy_bytes);
    } else {
        if (!csprng_fill(entropy, entropy_bytes)) {
            return {"", false};
        }
    }

    // Compute SHA-256 checksum of entropy
    auto hash = SHA256::hash(entropy, entropy_bytes);

    // Build the bit stream: entropy bits + checksum bits
    // checksum_bits = entropy_bytes * 8 / 32 = entropy_bytes / 4
    const size_t entropy_bits = entropy_bytes * 8;
    const size_t checksum_bits = entropy_bytes / 4;
    const size_t total_bits = entropy_bits + checksum_bits;
    const size_t word_count = total_bits / 11;

    // Extract 11-bit indices from the combined entropy+checksum bit stream
    std::string mnemonic;
    for (size_t i = 0; i < word_count; ++i) {
        uint32_t index = 0;
        for (size_t b = 0; b < 11; ++b) {
            const size_t bit_pos = i * 11 + b;
            uint8_t byte_val = 0;
            if (bit_pos < entropy_bits) {
                byte_val = entropy[bit_pos / 8];
            }
            else {
                byte_val = hash[(bit_pos - entropy_bits) / 8];
            }

            size_t bit_in_byte = 7 - (bit_pos % 8);
            if (bit_pos >= entropy_bits) {
                const size_t cs_bit = bit_pos - entropy_bits;
                byte_val = hash[cs_bit / 8];
                bit_in_byte = 7 - (cs_bit % 8);
            }

            if (byte_val & (1u << bit_in_byte)) {
                index |= (1u << (10 - b));
            }
        }

        if (i > 0) mnemonic += ' ';
        mnemonic += detail::bip39_english[index];
    }

    // Clear sensitive data
    detail::secure_erase(entropy, sizeof(entropy));

    return {mnemonic, true};
}

// ---------------------------------------------------------------------------
// bip39_validate
// ---------------------------------------------------------------------------
bool bip39_validate(const std::string& mnemonic) {
    auto words = split_words(mnemonic);

    // Valid word counts: 12, 15, 18, 21, 24
    if (words.size() < 12 || words.size() > 24 || (words.size() % 3) != 0) {
        return false;
    }

    // Look up each word index
    std::vector<int> indices(words.size());
    for (size_t i = 0; i < words.size(); ++i) {
        indices[i] = word_index(words[i].c_str());
        if (indices[i] < 0) {
            detail::secure_erase(indices.data(), indices.size() * sizeof(int));
            return false;
        }
    }

    // Reconstruct entropy + checksum bits
    const size_t total_bits = words.size() * 11;
    const size_t checksum_bits = words.size() / 3;
    const size_t entropy_bits = total_bits - checksum_bits;
    const size_t entropy_bytes = entropy_bits / 8;

    uint8_t entropy[32] = {};
    uint8_t checksum_byte = 0;

    for (size_t i = 0; i < words.size(); ++i) {
        const auto idx = static_cast<uint32_t>(indices[i]);
        for (size_t b = 0; b < 11; ++b) {
            const size_t bit_pos = i * 11 + b;
            const bool bit_set = (idx >> (10 - b)) & 1;
            if (bit_pos < entropy_bits) {
                if (bit_set) {
                    entropy[bit_pos / 8] |= (1u << (7 - (bit_pos % 8)));
                }
            } else {
                const size_t cs_bit = bit_pos - entropy_bits;
                if (bit_set) {
                    checksum_byte |= (1u << (7 - cs_bit));
                }
            }
        }
    }

    // Verify checksum
    auto hash = SHA256::hash(entropy, entropy_bytes);
    const uint8_t expected_cs = hash[0] >> (8 - checksum_bits);
    const uint8_t actual_cs = checksum_byte >> (8 - checksum_bits);

    detail::secure_erase(entropy, sizeof(entropy));
    detail::secure_erase(indices.data(), indices.size() * sizeof(int));
    return expected_cs == actual_cs;
}

// ---------------------------------------------------------------------------
// bip39_mnemonic_to_seed
// ---------------------------------------------------------------------------
std::pair<std::array<uint8_t, 64>, bool>
bip39_mnemonic_to_seed(const std::string& mnemonic,
                       const std::string& passphrase) {
    std::array<uint8_t, 64> seed{};

    if (mnemonic.empty()) {
        return {seed, false};
    }

    // BIP-39 requires NFKD normalization of both mnemonic and passphrase
    // before PBKDF2: PBKDF2(NFKD(mnemonic), "mnemonic" + NFKD(passphrase))
    std::string norm_mnemonic = nfkd_normalize(mnemonic);
    std::string salt_str      = "mnemonic" + nfkd_normalize(passphrase);

    pbkdf2_hmac_sha512(
        reinterpret_cast<const uint8_t*>(norm_mnemonic.data()), norm_mnemonic.size(),
        reinterpret_cast<const uint8_t*>(salt_str.data()), salt_str.size(),
        2048,
        seed.data(), 64);

    detail::secure_erase(norm_mnemonic.data(), norm_mnemonic.size());
    detail::secure_erase(salt_str.data(), salt_str.size());

    return {seed, true};
}

// ---------------------------------------------------------------------------
// bip39_mnemonic_to_entropy
// ---------------------------------------------------------------------------
std::pair<Bip39Entropy, bool>
bip39_mnemonic_to_entropy(const std::string& mnemonic) {
    Bip39Entropy ent{};

    auto words = split_words(mnemonic);
    if (words.size() < 12 || words.size() > 24 || (words.size() % 3) != 0) {
        return {ent, false};
    }

    std::vector<int> indices(words.size());
    for (size_t i = 0; i < words.size(); ++i) {
        indices[i] = word_index(words[i].c_str());
        if (indices[i] < 0) {
            detail::secure_erase(indices.data(), indices.size() * sizeof(int));
            return {ent, false};
        }
    }

    const size_t total_bits = words.size() * 11;
    const size_t checksum_bits = words.size() / 3;
    const size_t entropy_bits = total_bits - checksum_bits;
    const size_t entropy_bytes = entropy_bits / 8;

    uint8_t entropy[32] = {};
    uint8_t checksum_byte = 0;

    for (size_t i = 0; i < words.size(); ++i) {
        const auto idx = static_cast<uint32_t>(indices[i]);
        for (size_t b = 0; b < 11; ++b) {
            const size_t bit_pos = i * 11 + b;
            const bool bit_set = (idx >> (10 - b)) & 1;
            if (bit_pos < entropy_bits) {
                if (bit_set) {
                    entropy[bit_pos / 8] |= (1u << (7 - (bit_pos % 8)));
                }
            } else {
                const size_t cs_bit = bit_pos - entropy_bits;
                if (bit_set) {
                    checksum_byte |= (1u << (7 - cs_bit));
                }
            }
        }
    }

    // Verify checksum
    auto hash = SHA256::hash(entropy, entropy_bytes);
    const uint8_t expected_cs = hash[0] >> (8 - checksum_bits);
    const uint8_t actual_cs = checksum_byte >> (8 - checksum_bits);

    if (expected_cs != actual_cs) {
        detail::secure_erase(entropy, sizeof(entropy));
        detail::secure_erase(indices.data(), indices.size() * sizeof(int));
        return {ent, false};
    }

    std::memcpy(ent.data.data(), entropy, entropy_bytes);
    ent.length = entropy_bytes;
    detail::secure_erase(entropy, sizeof(entropy));
    detail::secure_erase(indices.data(), indices.size() * sizeof(int));

    return {ent, true};
}

// ---------------------------------------------------------------------------
// bip39_wordlist_english
// ---------------------------------------------------------------------------
const char* const* bip39_wordlist_english() {
    return detail::bip39_english;
}

} // namespace secp256k1
