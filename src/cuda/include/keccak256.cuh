// =============================================================================
// UltrafastSecp256k1 CUDA -- Keccak-256 (Ethereum-compatible)
// =============================================================================
// Standard Keccak-f[1600] permutation, rate=1088, capacity=512.
// Uses 0x01 padding (NOT SHA3's 0x06) for Ethereum compatibility.
// =============================================================================

#pragma once
#ifndef KECCAK256_CUH
#define KECCAK256_CUH

#include <cstdint>

namespace secp256k1_gpu {

// Keccak-f[1600] round constants
__device__ __constant__ static const uint64_t KECCAK_RC[24] = {
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

// Rotation offsets for rho step
__device__ __constant__ static const int KECCAK_ROT_OFF[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
};

__device__ __forceinline__ uint64_t keccak_rotl64(uint64_t x, int n) {
    return (x << (n & 63)) | (x >> ((64 - n) & 63));
}

__device__ __forceinline__ void keccak_f1600(uint64_t state[25]) {
    for (int round = 0; round < 24; ++round) {
        // theta
        uint64_t C[5];
        for (int x = 0; x < 5; ++x)
            C[x] = state[x] ^ state[x+5] ^ state[x+10] ^ state[x+15] ^ state[x+20];
        uint64_t D[5];
        for (int x = 0; x < 5; ++x)
            D[x] = C[(x+4) % 5] ^ keccak_rotl64(C[(x+1) % 5], 1);
        for (int i = 0; i < 25; ++i)
            state[i] ^= D[i % 5];

        // rho + pi
        uint64_t B[25];
        for (int x = 0; x < 5; ++x)
            for (int y = 0; y < 5; ++y)
                B[y + 5 * ((2*x + 3*y) % 5)] = keccak_rotl64(state[x + 5*y], KECCAK_ROT_OFF[x + 5*y]);

        // chi
        for (int x = 0; x < 5; ++x)
            for (int y = 0; y < 5; ++y)
                state[x + 5*y] = B[x + 5*y] ^ ((~B[((x+1)%5) + 5*y]) & B[((x+2)%5) + 5*y]);

        // iota
        state[0] ^= KECCAK_RC[round];
    }
}

// One-shot Keccak-256: hash arbitrary-length data -> 32 bytes
__device__ __forceinline__ void keccak256(const uint8_t* data, uint32_t len, uint8_t out[32]) {
    uint64_t state[25];
    for (int i = 0; i < 25; i++) state[i] = 0;

    const uint32_t RATE = 136;

    uint32_t pos = 0;
    // Absorb full blocks
    while (pos + RATE <= len) {
        for (uint32_t i = 0; i < RATE / 8; i++) {
            uint64_t lane = 0;
            for (int b = 0; b < 8; b++)
                lane |= ((uint64_t)data[pos + i*8 + b]) << (b * 8);
            state[i] ^= lane;
        }
        keccak_f1600(state);
        pos += RATE;
    }

    // Absorb final partial block + Keccak padding (0x01)
    uint8_t padded[136];
    for (uint32_t i = 0; i < RATE; i++) padded[i] = 0;
    uint32_t remaining = len - pos;
    for (uint32_t i = 0; i < remaining; i++) padded[i] = data[pos + i];
    padded[remaining] = 0x01;
    padded[RATE - 1] |= 0x80;

    for (uint32_t i = 0; i < RATE / 8; i++) {
        uint64_t lane = 0;
        for (int b = 0; b < 8; b++)
            lane |= ((uint64_t)padded[i*8 + b]) << (b * 8);
        state[i] ^= lane;
    }
    keccak_f1600(state);

    // Squeeze
    for (int i = 0; i < 4; i++)
        for (int b = 0; b < 8; b++)
            out[i*8 + b] = (uint8_t)(state[i] >> (b * 8));
}

// Ethereum address: keccak256(pubkey_xy) -> last 20 bytes
__device__ __forceinline__ void eth_address(const uint8_t* pubkey_xy, uint32_t xy_len, uint8_t addr[20]) {
    uint8_t hash[32];
    keccak256(pubkey_xy, xy_len, hash);
    for (int i = 0; i < 20; i++) addr[i] = hash[12 + i];
}

// EIP-55 checksum encoding
__device__ __forceinline__ void eip55_checksum(const uint8_t addr[20], uint8_t hex_out[40]) {
    const uint8_t hx[16] = {'0','1','2','3','4','5','6','7',
                              '8','9','a','b','c','d','e','f'};
    uint8_t lower_hex[40];
    for (int i = 0; i < 20; i++) {
        lower_hex[i*2]     = hx[(addr[i] >> 4) & 0x0F];
        lower_hex[i*2 + 1] = hx[addr[i] & 0x0F];
    }

    uint8_t addr_hash[32];
    keccak256(lower_hex, 40, addr_hash);

    for (int i = 0; i < 40; i++) {
        uint8_t nibble = (addr_hash[i / 2] >> ((1 - (i % 2)) * 4)) & 0x0F;
        hex_out[i] = lower_hex[i];
        if (hex_out[i] >= 'a' && hex_out[i] <= 'f' && nibble >= 8)
            hex_out[i] -= 32;
    }
}

} // namespace secp256k1_gpu

#endif // KECCAK256_CUH
