// =============================================================================
// UltrafastSecp256k1 OpenCL -- Keccak-256 (Ethereum-compatible)
// =============================================================================
// Standard Keccak-f[1600] permutation, rate=1088, capacity=512.
// Uses 0x01 padding (NOT SHA3's 0x06) for Ethereum compatibility.
// Provides: keccak256_impl (one-shot), eth_address_impl (pubkey -> 20-byte addr),
//           eip55_checksum_impl (hex address -> EIP-55 checksummed string).
// =============================================================================

#ifndef SECP256K1_KECCAK256_CL
#define SECP256K1_KECCAK256_CL

// Keccak-f[1600] round constants
__constant ulong KECCAK_RC[24] = {
    0x0000000000000001UL, 0x0000000000008082UL,
    0x800000000000808AUL, 0x8000000080008000UL,
    0x000000000000808BUL, 0x0000000080000001UL,
    0x8000000080008081UL, 0x8000000000008009UL,
    0x000000000000008AUL, 0x0000000000000088UL,
    0x0000000080008009UL, 0x000000008000000AUL,
    0x000000008000808BUL, 0x800000000000008BUL,
    0x8000000000008089UL, 0x8000000000008003UL,
    0x8000000000008002UL, 0x8000000000000080UL,
    0x000000000000800AUL, 0x800000008000000AUL,
    0x8000000080008081UL, 0x8000000000008080UL,
    0x0000000080000001UL, 0x8000000080008008UL,
};

// Rotation offsets for rho step
__constant int KECCAK_ROT[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
};

inline ulong keccak_rotl64(ulong x, int n) {
    return (x << (n & 63)) | (x >> ((64 - n) & 63));
}

inline void keccak_f1600_impl(ulong state[25]) {
    for (int round = 0; round < 24; ++round) {
        // theta
        ulong C[5];
        for (int x = 0; x < 5; ++x)
            C[x] = state[x] ^ state[x+5] ^ state[x+10] ^ state[x+15] ^ state[x+20];
        ulong D[5];
        for (int x = 0; x < 5; ++x)
            D[x] = C[(x+4) % 5] ^ keccak_rotl64(C[(x+1) % 5], 1);
        for (int i = 0; i < 25; ++i)
            state[i] ^= D[i % 5];

        // rho + pi
        ulong B[25];
        for (int x = 0; x < 5; ++x)
            for (int y = 0; y < 5; ++y)
                B[y + 5 * ((2*x + 3*y) % 5)] = keccak_rotl64(state[x + 5*y], KECCAK_ROT[x + 5*y]);

        // chi
        for (int x = 0; x < 5; ++x)
            for (int y = 0; y < 5; ++y)
                state[x + 5*y] = B[x + 5*y] ^ ((~B[((x+1)%5) + 5*y]) & B[((x+2)%5) + 5*y]);

        // iota
        state[0] ^= KECCAK_RC[round];
    }
}

// One-shot Keccak-256: hash arbitrary-length data -> 32 bytes
inline void keccak256_impl(const uchar* data, uint len, uchar out[32]) {
    ulong state[25];
    for (int i = 0; i < 25; i++) state[i] = 0;

    const uint RATE = 136;  // 1088 bits / 8
    uint pos = 0;

    // Absorb full blocks
    while (pos + RATE <= len) {
        for (uint i = 0; i < RATE / 8; i++) {
            ulong lane = 0;
            for (int b = 0; b < 8; b++)
                lane |= ((ulong)data[pos + i*8 + b]) << (b * 8);
            state[i] ^= lane;
        }
        keccak_f1600_impl(state);
        pos += RATE;
    }

    // Absorb final partial block + Keccak padding (0x01, NOT SHA3's 0x06)
    uchar padded[136];
    for (uint i = 0; i < RATE; i++) padded[i] = 0;
    uint remaining = len - pos;
    for (uint i = 0; i < remaining; i++) padded[i] = data[pos + i];
    padded[remaining] = 0x01;
    padded[RATE - 1] |= 0x80;

    for (uint i = 0; i < RATE / 8; i++) {
        ulong lane = 0;
        for (int b = 0; b < 8; b++)
            lane |= ((ulong)padded[i*8 + b]) << (b * 8);
        state[i] ^= lane;
    }
    keccak_f1600_impl(state);

    // Squeeze: extract 32 bytes
    for (int i = 0; i < 4; i++)
        for (int b = 0; b < 8; b++)
            out[i*8 + b] = (uchar)(state[i] >> (b * 8));
}

// Ethereum address: keccak256(uncompressed_pubkey[1..64]) -> last 20 bytes
// Input: 65-byte uncompressed (04||x||y) or 64-byte raw (x||y)
inline void eth_address_impl(const uchar* pubkey_xy, uint xy_len, uchar addr[20]) {
    uchar hash[32];
    keccak256_impl(pubkey_xy, xy_len, hash);
    for (int i = 0; i < 20; i++) addr[i] = hash[12 + i];
}

// EIP-55 checksum encoding: lowercase hex addr -> checksummed hex string
// Input: 20-byte raw address, Output: 40-byte hex string (no 0x prefix)
inline void eip55_checksum_impl(const uchar addr[20], uchar hex_out[40]) {
    // Convert to lowercase hex
    const uchar hex_chars[16] = {'0','1','2','3','4','5','6','7',
                                  '8','9','a','b','c','d','e','f'};
    uchar lower_hex[40];
    for (int i = 0; i < 20; i++) {
        lower_hex[i*2]     = hex_chars[(addr[i] >> 4) & 0x0F];
        lower_hex[i*2 + 1] = hex_chars[addr[i] & 0x0F];
    }

    // Hash the lowercase hex string
    uchar addr_hash[32];
    keccak256_impl(lower_hex, 40, addr_hash);

    // Apply checksum: uppercase if corresponding nibble >= 8
    for (int i = 0; i < 40; i++) {
        uchar nibble = (addr_hash[i / 2] >> ((1 - (i % 2)) * 4)) & 0x0F;
        hex_out[i] = lower_hex[i];
        if (hex_out[i] >= 'a' && hex_out[i] <= 'f' && nibble >= 8)
            hex_out[i] -= 32;  // uppercase
    }
}

// =============================================================================
// Batch kernels
// =============================================================================

// Batch keccak256: hash multiple fixed-length messages
__kernel void keccak256_batch(
    __global const uchar* data,
    __global uchar* hashes,
    uint msg_len,
    uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    uchar local_data[256];  // max supported single message size
    uint clamped = (msg_len < 256) ? msg_len : 256;
    for (uint i = 0; i < clamped; i++)
        local_data[i] = data[gid * msg_len + i];

    uchar hash[32];
    keccak256_impl(local_data, clamped, hash);
    for (int i = 0; i < 32; i++)
        hashes[gid * 32 + i] = hash[i];
}

// Batch Ethereum address derivation from uncompressed public keys
// Input: N x 64-byte (x||y) public keys, Output: N x 20-byte addresses
__kernel void eth_address_batch(
    __global const uchar* pubkeys_xy,
    __global uchar* addresses,
    uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    uchar xy[64];
    for (int i = 0; i < 64; i++)
        xy[i] = pubkeys_xy[gid * 64 + i];

    uchar addr[20];
    eth_address_impl(xy, 64, addr);
    for (int i = 0; i < 20; i++)
        addresses[gid * 20 + i] = addr[i];
}

#endif // SECP256K1_KECCAK256_CL
