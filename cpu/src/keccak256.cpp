// ============================================================================
// Keccak-256 Implementation
// ============================================================================
// Standard Keccak-f[1600] permutation with rate=1088, capacity=512.
// Ethereum-compatible: uses 0x01 padding (NOT SHA3's 0x06).
//
// Reference: NIST FIPS 202 (Section 3), but with Keccak padding.
// ============================================================================

#include "secp256k1/coins/keccak256.hpp"
#include <cstring>

namespace secp256k1::coins {

// -- Keccak-f[1600] Round Constants -------------------------------------------

static constexpr std::uint64_t KECCAK_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL,
};

// -- Rotation offsets ---------------------------------------------------------

static constexpr int KECCAK_ROT[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
};

// -- Helper -------------------------------------------------------------------

static inline std::uint64_t rotl64(std::uint64_t x, int n) {
    // Mask shift counts to avoid UB when n=0 (KECCAK_ROT[0]=0 -> x>>64 is UB)
    return (x << (n & 63)) | (x >> ((64 - n) & 63));
}

// -- Keccak-f[1600] Permutation -----------------------------------------------

static void keccak_f1600(std::uint64_t state[25]) {
    for (int round = 0; round < 24; ++round) {
        // theta (theta)
        std::uint64_t C[5];
        for (int x = 0; x < 5; ++x) {
            C[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
}

        std::uint64_t D[5];
        for (int x = 0; x < 5; ++x) {
            D[x] = C[(x + 4) % 5] ^ rotl64(C[(x + 1) % 5], 1);
}

        for (int x = 0; x < 5; ++x) {
            for (int y = 0; y < 5; ++y) {
                state[x + 5 * y] ^= D[x];
}
}

        // rho (rho) + pi (pi)
        std::uint64_t B[25];
        for (int x = 0; x < 5; ++x) {
            for (int y = 0; y < 5; ++y) {
                B[y + 5 * ((2 * x + 3 * y) % 5)] = rotl64(state[x + 5 * y], KECCAK_ROT[x + 5 * y]);
}
}

        //  (chi)
        for (int x = 0; x < 5; ++x) {
            for (int y = 0; y < 5; ++y) {
                state[x + 5 * y] = B[x + 5 * y] ^ ((~B[((x + 1) % 5) + 5 * y]) & B[((x + 2) % 5) + 5 * y]);
}
}

        //  (iota)
        state[0] ^= KECCAK_RC[round];
    }
}

// -- Keccak256State -----------------------------------------------------------

Keccak256State::Keccak256State() : buf_pos(0) {
    std::memset(state, 0, sizeof(state));
    std::memset(buf, 0, sizeof(buf));
}

void Keccak256State::update(const std::uint8_t* data, std::size_t len) {
    constexpr std::size_t RATE = 136; // 1088 bits / 8
    
    while (len > 0) {
        std::size_t const space = RATE - buf_pos;
        std::size_t const to_copy = (len < space) ? len : space;
        
        std::memcpy(buf + buf_pos, data, to_copy);
        buf_pos += to_copy;
        data += to_copy;
        len -= to_copy;
        
        if (buf_pos == RATE) {
            // XOR buffer into state (little-endian lanes)
            for (std::size_t i = 0; i < RATE / 8; ++i) {
                std::uint64_t lane = 0;
                std::memcpy(&lane, buf + i * 8, 8);
                state[i] ^= lane;
            }
            keccak_f1600(state);
            buf_pos = 0;
        }
    }
}

std::array<std::uint8_t, 32> Keccak256State::finalize() {
    constexpr std::size_t RATE = 136;
    
    // Keccak padding: append 0x01, then pad with zeros, then set last bit
    // NOTE: Keccak uses 0x01, SHA3 uses 0x06
    buf[buf_pos] = 0x01;
    std::memset(buf + buf_pos + 1, 0, RATE - buf_pos - 1);
    buf[RATE - 1] |= 0x80;
    
    // XOR final block into state
    for (std::size_t i = 0; i < RATE / 8; ++i) {
        std::uint64_t lane = 0;
        std::memcpy(&lane, buf + i * 8, 8);
        state[i] ^= lane;
    }
    keccak_f1600(state);
    
    // Squeeze: extract first 32 bytes (256 bits)
    std::array<std::uint8_t, 32> output;
    for (int i = 0; i < 4; ++i) {
        std::memcpy(output.data() + static_cast<std::size_t>(i) * 8, &state[i], 8);
    }
    return output;
}

// -- One-Shot -----------------------------------------------------------------

std::array<std::uint8_t, 32> keccak256(const std::uint8_t* data, std::size_t len) {
    Keccak256State ctx;
    ctx.update(data, len);
    return ctx.finalize();
}

} // namespace secp256k1::coins
