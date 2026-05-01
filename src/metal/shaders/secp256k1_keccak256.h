// =============================================================================
// UltrafastSecp256k1 Metal -- Keccak-256 (Ethereum-compatible)
// =============================================================================
// Standard Keccak-f[1600] permutation, rate=1088, capacity=512.
// Uses 0x01 padding (NOT SHA3's 0x06) for Ethereum compatibility.
// Metal has no native 64-bit integers: each lane is uint2 (lo, hi).
// =============================================================================

#ifndef SECP256K1_KECCAK256_H
#define SECP256K1_KECCAK256_H

#include <metal_stdlib>
using namespace metal;

// uint2-based 64-bit lane: .x = lo, .y = hi
// Keccak-f[1600] round constants as uint2 (lo, hi)
constant uint2 KECCAK_RC[24] = {
    uint2(0x00000001, 0x00000000), uint2(0x00008082, 0x00000000),
    uint2(0x0000808A, 0x80000000), uint2(0x80008000, 0x80000000),
    uint2(0x0000808B, 0x00000000), uint2(0x80000001, 0x00000000),
    uint2(0x80008081, 0x80000000), uint2(0x00008009, 0x80000000),
    uint2(0x0000008A, 0x00000000), uint2(0x00000088, 0x00000000),
    uint2(0x80008009, 0x00000000), uint2(0x8000000A, 0x00000000),
    uint2(0x8000808B, 0x00000000), uint2(0x0000008B, 0x80000000),
    uint2(0x00008089, 0x80000000), uint2(0x00008003, 0x80000000),
    uint2(0x00008002, 0x80000000), uint2(0x00000080, 0x80000000),
    uint2(0x0000800A, 0x00000000), uint2(0x8000000A, 0x80000000),
    uint2(0x80008081, 0x80000000), uint2(0x00008080, 0x80000000),
    uint2(0x80000001, 0x00000000), uint2(0x80008008, 0x80000000),
};

// Rotation offsets for rho step
constant int KECCAK_ROT_OFF[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
};

// XOR two uint2 lanes
inline uint2 keccak_xor(uint2 a, uint2 b) { return uint2(a.x ^ b.x, a.y ^ b.y); }

// NOT a lane
inline uint2 keccak_not(uint2 a) { return uint2(~a.x, ~a.y); }

// AND two lanes
inline uint2 keccak_and(uint2 a, uint2 b) { return uint2(a.x & b.x, a.y & b.y); }

// Rotate left a uint2 "64-bit" value by n bits (0 <= n < 64)
inline uint2 keccak_rotl64(uint2 v, int n) {
    if (n == 0) return v;
    if (n == 32) return uint2(v.y, v.x);
    if (n < 32) {
        return uint2((v.x << n) | (v.y >> (32 - n)),
                     (v.y << n) | (v.x >> (32 - n)));
    }
    // n > 32
    int m = n - 32;
    return uint2((v.y << m) | (v.x >> (32 - m)),
                 (v.x << m) | (v.y >> (32 - m)));
}

inline void keccak_f1600_metal(thread uint2 state[25]) {
    for (int round = 0; round < 24; ++round) {
        // theta
        uint2 C[5];
        for (int x = 0; x < 5; ++x)
            C[x] = keccak_xor(keccak_xor(keccak_xor(state[x], state[x+5]),
                               keccak_xor(state[x+10], state[x+15])), state[x+20]);
        uint2 D[5];
        for (int x = 0; x < 5; ++x)
            D[x] = keccak_xor(C[(x+4) % 5], keccak_rotl64(C[(x+1) % 5], 1));
        for (int i = 0; i < 25; ++i)
            state[i] = keccak_xor(state[i], D[i % 5]);

        // rho + pi
        uint2 B[25];
        for (int x = 0; x < 5; ++x)
            for (int y = 0; y < 5; ++y)
                B[y + 5 * ((2*x + 3*y) % 5)] = keccak_rotl64(state[x + 5*y], KECCAK_ROT_OFF[x + 5*y]);

        // chi
        for (int x = 0; x < 5; ++x)
            for (int y = 0; y < 5; ++y)
                state[x + 5*y] = keccak_xor(B[x + 5*y],
                    keccak_and(keccak_not(B[((x+1)%5) + 5*y]), B[((x+2)%5) + 5*y]));

        // iota
        state[0] = keccak_xor(state[0], KECCAK_RC[round]);
    }
}

// Load a byte from a uint2 lane (little-endian, lane index within squeezed state)
inline uchar keccak_lane_byte(uint2 lane, int byte_idx) {
    // byte_idx 0..3 from .x (lo), 4..7 from .y (hi)
    if (byte_idx < 4)
        return (uchar)((lane.x >> (byte_idx * 8)) & 0xFF);
    return (uchar)((lane.y >> ((byte_idx - 4) * 8)) & 0xFF);
}

// XOR a byte into a uint2 lane at position byte_idx
inline uint2 keccak_lane_xor_byte(uint2 lane, int byte_idx, uchar val) {
    if (byte_idx < 4)
        return uint2(lane.x ^ ((uint)val << (byte_idx * 8)), lane.y);
    return uint2(lane.x, lane.y ^ ((uint)val << ((byte_idx - 4) * 8)));
}

// One-shot Keccak-256: hash arbitrary-length data -> 32 bytes
inline void keccak256_metal(thread const uchar* data, uint len, thread uchar out[32]) {
    uint2 state[25];
    for (int i = 0; i < 25; i++) state[i] = uint2(0, 0);

    const uint RATE = 136;
    uint pos = 0;

    // Absorb full blocks
    while (pos + RATE <= len) {
        for (uint i = 0; i < RATE / 8; i++) {
            uint lo = 0, hi = 0;
            for (int b = 0; b < 4; b++) {
                lo |= (uint)data[pos + i*8 + b] << (b * 8);
                hi |= (uint)data[pos + i*8 + 4 + b] << (b * 8);
            }
            state[i] = keccak_xor(state[i], uint2(lo, hi));
        }
        keccak_f1600_metal(state);
        pos += RATE;
    }

    // Absorb final partial block + Keccak padding (0x01)
    uchar padded[136];
    for (uint i = 0; i < RATE; i++) padded[i] = 0;
    uint remaining = len - pos;
    for (uint i = 0; i < remaining; i++) padded[i] = data[pos + i];
    padded[remaining] = 0x01;
    padded[RATE - 1] |= 0x80;

    for (uint i = 0; i < RATE / 8; i++) {
        uint lo = 0, hi = 0;
        for (int b = 0; b < 4; b++) {
            lo |= (uint)padded[i*8 + b] << (b * 8);
            hi |= (uint)padded[i*8 + 4 + b] << (b * 8);
        }
        state[i] = keccak_xor(state[i], uint2(lo, hi));
    }
    keccak_f1600_metal(state);

    // Squeeze: extract 32 bytes (4 lanes)
    for (int i = 0; i < 4; i++)
        for (int b = 0; b < 8; b++)
            out[i*8 + b] = keccak_lane_byte(state[i], b);
}

// Ethereum address: keccak256(pubkey_xy) -> last 20 bytes
inline void eth_address_metal(thread const uchar* pubkey_xy, uint xy_len, thread uchar addr[20]) {
    uchar hash[32];
    keccak256_metal(pubkey_xy, xy_len, hash);
    for (int i = 0; i < 20; i++) addr[i] = hash[12 + i];
}

// EIP-55 checksum encoding
inline void eip55_checksum_metal(thread const uchar addr[20], thread uchar hex_out[40]) {
    const uchar hx[16] = {'0','1','2','3','4','5','6','7',
                            '8','9','a','b','c','d','e','f'};
    uchar lower_hex[40];
    for (int i = 0; i < 20; i++) {
        lower_hex[i*2]     = hx[(addr[i] >> 4) & 0x0F];
        lower_hex[i*2 + 1] = hx[addr[i] & 0x0F];
    }

    uchar addr_hash[32];
    keccak256_metal(lower_hex, 40, addr_hash);

    for (int i = 0; i < 40; i++) {
        uchar nibble = (addr_hash[i / 2] >> ((1 - (i % 2)) * 4)) & 0x0F;
        hex_out[i] = lower_hex[i];
        if (hex_out[i] >= 'a' && hex_out[i] <= 'f' && nibble >= 8)
            hex_out[i] -= 32;
    }
}

#endif // SECP256K1_KECCAK256_H
