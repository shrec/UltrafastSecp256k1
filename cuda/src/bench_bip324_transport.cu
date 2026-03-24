// ============================================================================
// BIP-324 Transport Benchmark — CUDA Batch Parallel
// ============================================================================
// GPU-accelerated BIP-324 packet processing: each CUDA thread handles one
// independent packet encryption or decryption (simulating batch processing
// of many peer connections simultaneously).
//
//   1. ChaCha20 block kernel       — batch keystream generation
//   2. ChaCha20-Poly1305 AEAD      — batch encrypt / decrypt
//   3. transport_mixed             — realistic payload size distribution
//   4. transport_with_decoys       — decoy packet CPU tax (GPU perspective)
//   5. latency comparison          — GPU batch vs CPU single-packet
//
// Reference: BIP-324, RFC 8439
// ============================================================================

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>
#include <chrono>
#include <random>

// ============================================================================
// Error checking
// ============================================================================
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                     cudaGetErrorString(err), __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while(0)

// ============================================================================
// CUDA Timer
// ============================================================================
class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    float stop_ms() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
private:
    cudaEvent_t start_, stop_;
};

// ============================================================================
// Device helpers
// ============================================================================

__device__ __forceinline__ std::uint32_t d_load32_le(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0])
         | (static_cast<std::uint32_t>(p[1]) << 8)
         | (static_cast<std::uint32_t>(p[2]) << 16)
         | (static_cast<std::uint32_t>(p[3]) << 24);
}

__device__ __forceinline__ void d_store32_le(std::uint8_t* p, std::uint32_t v) {
    p[0] = static_cast<std::uint8_t>(v);
    p[1] = static_cast<std::uint8_t>(v >> 8);
    p[2] = static_cast<std::uint8_t>(v >> 16);
    p[3] = static_cast<std::uint8_t>(v >> 24);
}

__device__ __forceinline__ std::uint64_t d_load64_le(const std::uint8_t* p) {
    return static_cast<std::uint64_t>(p[0])
         | (static_cast<std::uint64_t>(p[1]) << 8)
         | (static_cast<std::uint64_t>(p[2]) << 16)
         | (static_cast<std::uint64_t>(p[3]) << 24)
         | (static_cast<std::uint64_t>(p[4]) << 32)
         | (static_cast<std::uint64_t>(p[5]) << 40)
         | (static_cast<std::uint64_t>(p[6]) << 48)
         | (static_cast<std::uint64_t>(p[7]) << 56);
}

__device__ __forceinline__ void d_store64_le(std::uint8_t* p, std::uint64_t v) {
    for (int i = 0; i < 8; ++i)
        p[i] = static_cast<std::uint8_t>(v >> (i * 8));
}

__device__ __forceinline__ std::uint32_t d_rotl32(std::uint32_t v, int n) {
    return (v << n) | (v >> (32 - n));
}

// ============================================================================
// Device ChaCha20 (RFC 8439)
// ============================================================================

__device__ void d_chacha20_quarter_round(std::uint32_t& a, std::uint32_t& b,
                                          std::uint32_t& c, std::uint32_t& d) {
    a += b; d ^= a; d = __byte_perm(d, 0, 0x1032);       // rotl32(16) via PTX
    c += d; b ^= c; b = __funnelshift_l(b, b, 12);       // rotl32(12) via PTX
    a += b; d ^= a; d = __byte_perm(d, 0, 0x0321);       // rotl32(8)  via PTX
    c += d; b ^= c; b = __funnelshift_l(b, b, 7);        // rotl32(7)  via PTX
}

__device__ void d_chacha20_block(const std::uint32_t input[16],
                                  std::uint8_t output[64]) {
    std::uint32_t x[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) x[i] = input[i];

    #pragma unroll
    for (int i = 0; i < 10; ++i) {
        d_chacha20_quarter_round(x[0], x[4], x[ 8], x[12]);
        d_chacha20_quarter_round(x[1], x[5], x[ 9], x[13]);
        d_chacha20_quarter_round(x[2], x[6], x[10], x[14]);
        d_chacha20_quarter_round(x[3], x[7], x[11], x[15]);
        d_chacha20_quarter_round(x[0], x[5], x[10], x[15]);
        d_chacha20_quarter_round(x[1], x[6], x[11], x[12]);
        d_chacha20_quarter_round(x[2], x[7], x[ 8], x[13]);
        d_chacha20_quarter_round(x[3], x[4], x[ 9], x[14]);
    }

    #pragma unroll
    for (int i = 0; i < 16; ++i)
        d_store32_le(output + i * 4, x[i] + input[i]);
}

__device__ void d_chacha20_setup(std::uint32_t state[16],
                                  const std::uint8_t key[32],
                                  const std::uint8_t nonce[12],
                                  std::uint32_t counter) {
    state[ 0] = 0x61707865u;
    state[ 1] = 0x3320646eu;
    state[ 2] = 0x79622d32u;
    state[ 3] = 0x6b206574u;
    for (int i = 0; i < 8; ++i) state[4 + i] = d_load32_le(key + i * 4);
    state[12] = counter;
    state[13] = d_load32_le(nonce);
    state[14] = d_load32_le(nonce + 4);
    state[15] = d_load32_le(nonce + 8);
}

__device__ void d_chacha20_crypt(const std::uint8_t key[32],
                                  const std::uint8_t nonce[12],
                                  std::uint32_t counter,
                                  std::uint8_t* data, std::size_t len) {
    std::uint32_t state[16];
    d_chacha20_setup(state, key, nonce, counter);
    std::uint8_t block[64];
    std::size_t offset = 0;
    while (offset < len) {
        d_chacha20_block(state, block);
        state[12]++;
        std::size_t use = (len - offset < 64) ? (len - offset) : 64;
        for (std::size_t i = 0; i < use; ++i)
            data[offset + i] ^= block[i];
        offset += use;
    }
}

// ============================================================================
// Device Poly1305 (RFC 8439) — 5×26-bit limbs (no __int128 on GPU)
// ============================================================================

struct DevPoly1305 {
    std::uint32_t r[5], s[4], h[5];

    __device__ void init(const std::uint8_t key[32]) {
        std::uint32_t t0 = d_load32_le(key +  0) & 0x0FFFFFFFu;
        std::uint32_t t1 = d_load32_le(key +  4) & 0x0FFFFFFCu;
        std::uint32_t t2 = d_load32_le(key +  8) & 0x0FFFFFFCu;
        std::uint32_t t3 = d_load32_le(key + 12) & 0x0FFFFFFCu;

        r[0] =  t0                        & 0x3FFFFFF;
        r[1] = ((t0 >> 26) | (t1 <<  6)) & 0x3FFFFFF;
        r[2] = ((t1 >> 20) | (t2 << 12)) & 0x3FFFFFF;
        r[3] = ((t2 >> 14) | (t3 << 18)) & 0x3FFFFFF;
        r[4] =  (t3 >>  8);

        s[0] = d_load32_le(key + 16);
        s[1] = d_load32_le(key + 20);
        s[2] = d_load32_le(key + 24);
        s[3] = d_load32_le(key + 28);

        h[0] = h[1] = h[2] = h[3] = h[4] = 0;
    }

    __device__ void block(const std::uint8_t* msg, std::size_t len) {
        std::uint8_t buf[17];
        for (int i = 0; i < 17; ++i) buf[i] = 0;
        for (std::size_t i = 0; i < len; ++i) buf[i] = msg[i];
        buf[len] = 1;

        std::uint32_t t0 = d_load32_le(buf);
        std::uint32_t t1 = d_load32_le(buf + 4);
        std::uint32_t t2 = d_load32_le(buf + 8);
        std::uint32_t t3 = d_load32_le(buf + 12);
        std::uint32_t hibit = static_cast<std::uint32_t>(buf[16]);

        h[0] += t0 & 0x3FFFFFF;
        h[1] += ((t0 >> 26) | (t1 << 6)) & 0x3FFFFFF;
        h[2] += ((t1 >> 20) | (t2 << 12)) & 0x3FFFFFF;
        h[3] += ((t2 >> 14) | (t3 << 18)) & 0x3FFFFFF;
        h[4] += (t3 >> 8) | (hibit << 24);

        std::uint32_t r0 = r[0], r1 = r[1], r2 = r[2], r3 = r[3], r4 = r[4];
        std::uint32_t s1 = r1 * 5, s2 = r2 * 5, s3 = r3 * 5, s4 = r4 * 5;

        std::uint64_t d0 = (std::uint64_t)h[0]*r0 + (std::uint64_t)h[1]*s4
                         + (std::uint64_t)h[2]*s3 + (std::uint64_t)h[3]*s2
                         + (std::uint64_t)h[4]*s1;
        std::uint64_t d1 = (std::uint64_t)h[0]*r1 + (std::uint64_t)h[1]*r0
                         + (std::uint64_t)h[2]*s4 + (std::uint64_t)h[3]*s3
                         + (std::uint64_t)h[4]*s2;
        std::uint64_t d2 = (std::uint64_t)h[0]*r2 + (std::uint64_t)h[1]*r1
                         + (std::uint64_t)h[2]*r0 + (std::uint64_t)h[3]*s4
                         + (std::uint64_t)h[4]*s3;
        std::uint64_t d3 = (std::uint64_t)h[0]*r3 + (std::uint64_t)h[1]*r2
                         + (std::uint64_t)h[2]*r1 + (std::uint64_t)h[3]*r0
                         + (std::uint64_t)h[4]*s4;
        std::uint64_t d4 = (std::uint64_t)h[0]*r4 + (std::uint64_t)h[1]*r3
                         + (std::uint64_t)h[2]*r2 + (std::uint64_t)h[3]*r1
                         + (std::uint64_t)h[4]*r0;

        std::uint32_t c;
        c = (std::uint32_t)(d0 >> 26); h[0] = (std::uint32_t)d0 & 0x3FFFFFF;
        d1 += c; c = (std::uint32_t)(d1 >> 26); h[1] = (std::uint32_t)d1 & 0x3FFFFFF;
        d2 += c; c = (std::uint32_t)(d2 >> 26); h[2] = (std::uint32_t)d2 & 0x3FFFFFF;
        d3 += c; c = (std::uint32_t)(d3 >> 26); h[3] = (std::uint32_t)d3 & 0x3FFFFFF;
        d4 += c; c = (std::uint32_t)(d4 >> 26); h[4] = (std::uint32_t)d4 & 0x3FFFFFF;
        h[0] += c * 5; c = h[0] >> 26; h[0] &= 0x3FFFFFF;
        h[1] += c;
    }

    __device__ void finish(std::uint8_t tag[16]) {
        std::uint32_t c;
        c = h[1] >> 26; h[1] &= 0x3FFFFFF;
        h[2] += c; c = h[2] >> 26; h[2] &= 0x3FFFFFF;
        h[3] += c; c = h[3] >> 26; h[3] &= 0x3FFFFFF;
        h[4] += c; c = h[4] >> 26; h[4] &= 0x3FFFFFF;
        h[0] += c * 5; c = h[0] >> 26; h[0] &= 0x3FFFFFF;
        h[1] += c;

        std::uint32_t g[5];
        c = h[0] + 5; g[0] = c & 0x3FFFFFF; c >>= 26;
        c += h[1];    g[1] = c & 0x3FFFFFF; c >>= 26;
        c += h[2];    g[2] = c & 0x3FFFFFF; c >>= 26;
        c += h[3];    g[3] = c & 0x3FFFFFF; c >>= 26;
        c += h[4];    g[4] = c & 0x3FFFFFF; c >>= 26;

        std::uint32_t mask = ~(c - 1);
        for (int i = 0; i < 5; ++i)
            h[i] = (h[i] & ~mask) | (g[i] & mask);

        std::uint64_t f;
        f  = (std::uint64_t)h[0] | ((std::uint64_t)h[1] << 26);
        std::uint32_t h0 = (std::uint32_t)f;
        f  = (f >> 32) | ((std::uint64_t)h[2] << 20);
        std::uint32_t h1 = (std::uint32_t)f;
        f  = (f >> 32) | ((std::uint64_t)h[3] << 14);
        std::uint32_t h2 = (std::uint32_t)f;
        f  = (f >> 32) | ((std::uint64_t)h[4] <<  8);
        std::uint32_t h3 = (std::uint32_t)f;

        std::uint64_t t;
        t = (std::uint64_t)h0 + s[0];              h0 = (std::uint32_t)t;
        t = (std::uint64_t)h1 + s[1] + (t >> 32);  h1 = (std::uint32_t)t;
        t = (std::uint64_t)h2 + s[2] + (t >> 32);  h2 = (std::uint32_t)t;
        t = (std::uint64_t)h3 + s[3] + (t >> 32);  h3 = (std::uint32_t)t;

        d_store32_le(tag +  0, h0);
        d_store32_le(tag +  4, h1);
        d_store32_le(tag +  8, h2);
        d_store32_le(tag + 12, h3);
    }
};

// ============================================================================
// Device AEAD ChaCha20-Poly1305 (RFC 8439)
// ============================================================================

__device__ void d_aead_encrypt(const std::uint8_t key[32],
                                const std::uint8_t nonce[12],
                                const std::uint8_t* plaintext,
                                std::size_t plaintext_len,
                                std::uint8_t* ciphertext,
                                std::uint8_t tag[16]) {
    // 1. Setup state once (reused for poly key + encryption)
    std::uint32_t state[16];
    d_chacha20_setup(state, key, nonce, 0);

    // 2. Poly1305 key from block 0
    std::uint8_t poly_key[64];
    d_chacha20_block(state, poly_key);

    // 3. Encrypt (counter=1) — reuse state, skip redundant setup
    state[12] = 1;
    {
        std::uint8_t block[64];
        std::size_t offset = 0;
        while (offset < plaintext_len) {
            d_chacha20_block(state, block);
            state[12]++;
            std::size_t use = (plaintext_len - offset < 64)
                            ? (plaintext_len - offset) : 64;
            for (std::size_t j = 0; j < use; ++j)
                ciphertext[offset + j] = plaintext[offset + j] ^ block[j];
            offset += use;
        }
    }

    // 4. Poly1305 MAC over (empty AAD || ciphertext || lengths)
    DevPoly1305 st;
    st.init(poly_key);

    // AAD padding (empty AAD → skip to ct)
    std::size_t off = 0;
    while (off + 16 <= plaintext_len) {
        st.block(ciphertext + off, 16);
        off += 16;
    }
    if (off < plaintext_len)
        st.block(ciphertext + off, plaintext_len - off);

    std::size_t ct_pad = (16 - (plaintext_len % 16)) % 16;
    if (ct_pad > 0) {
        std::uint8_t zeros[16] = {};
        st.block(zeros, ct_pad);
    }

    std::uint8_t lens[16];
    d_store64_le(lens, 0); // AAD len = 0
    d_store64_le(lens + 8, static_cast<std::uint64_t>(plaintext_len));
    st.block(lens, 16);
    st.finish(tag);
}

__device__ bool d_aead_decrypt(const std::uint8_t key[32],
                                const std::uint8_t nonce[12],
                                const std::uint8_t* ciphertext,
                                std::size_t ciphertext_len,
                                const std::uint8_t expected_tag[16],
                                std::uint8_t* plaintext) {
    // 1. Setup state once
    std::uint32_t state[16];
    d_chacha20_setup(state, key, nonce, 0);

    // 2. Poly1305 key from block 0
    std::uint8_t poly_key[64];
    d_chacha20_block(state, poly_key);

    // 3. Verify tag
    DevPoly1305 st;
    st.init(poly_key);

    std::size_t off = 0;
    while (off + 16 <= ciphertext_len) {
        st.block(ciphertext + off, 16);
        off += 16;
    }
    if (off < ciphertext_len)
        st.block(ciphertext + off, ciphertext_len - off);

    std::size_t ct_pad = (16 - (ciphertext_len % 16)) % 16;
    if (ct_pad > 0) {
        std::uint8_t zeros[16] = {};
        st.block(zeros, ct_pad);
    }

    std::uint8_t lens[16];
    d_store64_le(lens, 0);
    d_store64_le(lens + 8, static_cast<std::uint64_t>(ciphertext_len));
    st.block(lens, 16);

    std::uint8_t computed[16];
    st.finish(computed);

    std::uint8_t diff = 0;
    for (int i = 0; i < 16; ++i) diff |= computed[i] ^ expected_tag[i];

    if (diff != 0) return false;

    // 4. Decrypt (counter=1) — reuse state, skip redundant setup
    state[12] = 1;
    {
        std::uint8_t block[64];
        std::size_t offset = 0;
        while (offset < ciphertext_len) {
            d_chacha20_block(state, block);
            state[12]++;
            std::size_t use = (ciphertext_len - offset < 64)
                            ? (ciphertext_len - offset) : 64;
            for (std::size_t j = 0; j < use; ++j)
                plaintext[offset + j] = ciphertext[offset + j] ^ block[j];
            offset += use;
        }
    }
    return true;
}

// ============================================================================
// BIP-324 packet format on GPU:
//   Wire = [3B encrypted length] [payload] [16B tag]
//   We simplify: each thread encrypts/decrypts one fixed-size payload using
//   its own key + nonce (simulating independent peer connections).
// ============================================================================

static constexpr std::size_t BIP324_OVERHEAD = 3 + 16; // 19 bytes
static constexpr int THREADS_PER_BLOCK = 128; // Lower occupancy per block,
                                               // but more registers for crypto state

// ============================================================================
// Kernel: batch AEAD encrypt
// ============================================================================
// Each thread encrypts one packet.
// Input:  d_plaintexts (N * max_payload_bytes contiguous)
// Output: d_ciphertexts (N * (max_payload_bytes + 16) contiguous), d_headers (N * 3)
// Keys/nonces are per-packet.

__global__ void kernel_batch_aead_encrypt(
    const std::uint8_t* __restrict__ d_keys,       // N * 32
    const std::uint8_t* __restrict__ d_nonces,     // N * 12
    const std::uint8_t* __restrict__ d_plaintexts, // N * max_payload
    const std::uint32_t* __restrict__ d_sizes,     // N payload sizes
    std::uint8_t* __restrict__ d_wire_out,         // N * (max_payload + BIP324_OVERHEAD)
    std::uint32_t max_payload,
    int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    std::uint32_t payload_sz = d_sizes[idx];
    const std::uint8_t* key = d_keys + idx * 32;
    const std::uint8_t* nonce = d_nonces + idx * 12;
    const std::uint8_t* pt = d_plaintexts + (std::size_t)idx * max_payload;

    std::size_t wire_stride = max_payload + BIP324_OVERHEAD;
    std::uint8_t* wire = d_wire_out + (std::size_t)idx * wire_stride;

    // BIP-324: 3-byte encrypted length header
    // (simplified — just store payload_sz as 3 LE bytes, not reallly encrypted for benchmark)
    wire[0] = static_cast<std::uint8_t>(payload_sz);
    wire[1] = static_cast<std::uint8_t>(payload_sz >> 8);
    wire[2] = static_cast<std::uint8_t>(payload_sz >> 16);

    // AEAD encrypt → ciphertext at wire+3, tag at wire+3+payload_sz
    d_aead_encrypt(key, nonce, pt, payload_sz,
                   wire + 3, wire + 3 + payload_sz);
}

// ============================================================================
// Kernel: batch AEAD decrypt
// ============================================================================

__global__ void kernel_batch_aead_decrypt(
    const std::uint8_t* __restrict__ d_keys,
    const std::uint8_t* __restrict__ d_nonces,
    const std::uint8_t* __restrict__ d_wire_in,     // N * (max_payload + BIP324_OVERHEAD)
    const std::uint32_t* __restrict__ d_sizes,
    std::uint8_t* __restrict__ d_plaintext_out,    // N * max_payload
    std::uint32_t* __restrict__ d_ok,              // N success flags
    std::uint32_t max_payload,
    int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    std::uint32_t payload_sz = d_sizes[idx];
    const std::uint8_t* key = d_keys + idx * 32;
    const std::uint8_t* nonce = d_nonces + idx * 12;

    std::size_t wire_stride = max_payload + BIP324_OVERHEAD;
    const std::uint8_t* wire = d_wire_in + (std::size_t)idx * wire_stride;
    std::uint8_t* pt_out = d_plaintext_out + (std::size_t)idx * max_payload;

    const std::uint8_t* ct = wire + 3;
    const std::uint8_t* tag = wire + 3 + payload_sz;

    bool ok = d_aead_decrypt(key, nonce, ct, payload_sz, tag, pt_out);
    d_ok[idx] = ok ? 1 : 0;
}

// ============================================================================
// Kernel: batch ChaCha20 block only (throughput ceiling)
// ============================================================================

__global__ void kernel_chacha20_block_batch(
    const std::uint8_t* __restrict__ d_keys,
    const std::uint8_t* __restrict__ d_nonces,
    std::uint8_t* __restrict__ d_out, // N * 64
    int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    std::uint32_t state[16];
    d_chacha20_setup(state, d_keys + idx * 32, d_nonces + idx * 12, 0);
    d_chacha20_block(state, d_out + idx * 64);
}

// ============================================================================
// Host benchmark harness
// ============================================================================

struct GpuBenchResult {
    double total_ms;
    double packets_per_sec;
    double goodput_mbps;
    double wire_mbps;
    double overhead_pct;
    int count;
    double avg_payload;
};

// Generate random keys and nonces on host
static void fill_random_keys(std::uint8_t* keys, std::uint8_t* nonces, int count,
                              std::mt19937_64& rng) {
    for (int i = 0; i < count * 32; ++i)
        keys[i] = static_cast<std::uint8_t>(rng());
    for (int i = 0; i < count * 12; ++i)
        nonces[i] = static_cast<std::uint8_t>(rng());
}

GpuBenchResult bench_batch_encrypt_decrypt(const std::uint32_t* h_sizes,
                                            int count,
                                            std::uint32_t max_payload,
                                            int warmup_rounds,
                                            int measure_rounds) {
    std::mt19937_64 rng(0xDEAD1234BEEFULL);

    // Compute stats
    std::size_t total_payload = 0;
    for (int i = 0; i < count; ++i) total_payload += h_sizes[i];
    std::size_t total_wire = total_payload + (std::size_t)count * BIP324_OVERHEAD;
    double avg_payload = static_cast<double>(total_payload) / count;

    // Allocate host data
    std::vector<std::uint8_t> h_keys(count * 32);
    std::vector<std::uint8_t> h_nonces(count * 12);
    std::vector<std::uint8_t> h_plaintext(static_cast<std::size_t>(count) * max_payload);
    fill_random_keys(h_keys.data(), h_nonces.data(), count, rng);
    for (auto& b : h_plaintext) b = static_cast<std::uint8_t>(rng());

    // GPU allocations
    std::uint8_t *d_keys, *d_nonces, *d_plaintext, *d_wire, *d_dec_out;
    std::uint32_t *d_sizes, *d_ok;
    std::size_t wire_stride = max_payload + BIP324_OVERHEAD;

    CUDA_CHECK(cudaMalloc(&d_keys,      count * 32));
    CUDA_CHECK(cudaMalloc(&d_nonces,    count * 12));
    CUDA_CHECK(cudaMalloc(&d_plaintext, static_cast<std::size_t>(count) * max_payload));
    CUDA_CHECK(cudaMalloc(&d_wire,      static_cast<std::size_t>(count) * wire_stride));
    CUDA_CHECK(cudaMalloc(&d_dec_out,   static_cast<std::size_t>(count) * max_payload));
    CUDA_CHECK(cudaMalloc(&d_sizes,     count * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_ok,        count * sizeof(std::uint32_t)));

    CUDA_CHECK(cudaMemcpy(d_keys,      h_keys.data(),      count * 32,       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nonces,    h_nonces.data(),    count * 12,       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_plaintext, h_plaintext.data(),
                          static_cast<std::size_t>(count) * max_payload, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sizes,     h_sizes,            count * sizeof(std::uint32_t), cudaMemcpyHostToDevice));

    int blocks = (count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Warmup
    for (int w = 0; w < warmup_rounds; ++w) {
        kernel_batch_aead_encrypt<<<blocks, THREADS_PER_BLOCK>>>(
            d_keys, d_nonces, d_plaintext, d_sizes, d_wire, max_payload, count);
        kernel_batch_aead_decrypt<<<blocks, THREADS_PER_BLOCK>>>(
            d_keys, d_nonces, d_wire, d_sizes, d_dec_out, d_ok, max_payload, count);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Measure
    CudaTimer timer;
    std::vector<double> times_ms;
    times_ms.reserve(measure_rounds);

    for (int r = 0; r < measure_rounds; ++r) {
        timer.start();
        kernel_batch_aead_encrypt<<<blocks, THREADS_PER_BLOCK>>>(
            d_keys, d_nonces, d_plaintext, d_sizes, d_wire, max_payload, count);
        kernel_batch_aead_decrypt<<<blocks, THREADS_PER_BLOCK>>>(
            d_keys, d_nonces, d_wire, d_sizes, d_dec_out, d_ok, max_payload, count);
        float ms = timer.stop_ms();
        times_ms.push_back(ms);
    }

    // Use median
    std::sort(times_ms.begin(), times_ms.end());
    double median_ms = times_ms[times_ms.size() / 2];

    // Verify decrypts succeeded
    std::vector<std::uint32_t> h_ok(count);
    CUDA_CHECK(cudaMemcpy(h_ok.data(), d_ok, count * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
    int ok_count = 0;
    for (int i = 0; i < count; ++i) ok_count += h_ok[i];

    if (ok_count != count) {
        std::printf("  WARNING: %d/%d decrypts failed!\n", count - ok_count, count);
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_nonces));
    CUDA_CHECK(cudaFree(d_plaintext));
    CUDA_CHECK(cudaFree(d_wire));
    CUDA_CHECK(cudaFree(d_dec_out));
    CUDA_CHECK(cudaFree(d_sizes));
    CUDA_CHECK(cudaFree(d_ok));

    double total_sec = median_ms / 1e3;
    GpuBenchResult res;
    res.total_ms = median_ms;
    res.count = count;
    res.avg_payload = avg_payload;
    res.packets_per_sec = static_cast<double>(count) / total_sec;
    res.goodput_mbps = (static_cast<double>(total_payload) / total_sec) / (1024.0 * 1024.0);
    res.wire_mbps = (static_cast<double>(total_wire) / total_sec) / (1024.0 * 1024.0);
    res.overhead_pct = 100.0 * (1.0 - static_cast<double>(total_payload)
                                       / static_cast<double>(total_wire));
    return res;
}

// ============================================================================
// Simple LCG for deterministic size generation
// ============================================================================
struct HostRng {
    std::uint64_t state;
    HostRng(std::uint64_t seed = 0xDEADBEEFULL) : state(seed) {}
    std::uint32_t next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<std::uint32_t>(state >> 32);
    }
    std::uint32_t range(std::uint32_t lo, std::uint32_t hi) {
        return lo + (next() % (hi - lo + 1));
    }
};

// ============================================================================
// 1. ChaCha20 block throughput ceiling
// ============================================================================

static void bench_chacha20_blocks() {
    std::printf("--- 1. ChaCha20 Block Throughput (GPU ceiling) ---\n");

    static constexpr int BATCH = 1 << 20; // 1M blocks
    std::mt19937_64 rng(42);

    std::vector<std::uint8_t> h_keys(BATCH * 32), h_nonces(BATCH * 12);
    fill_random_keys(h_keys.data(), h_nonces.data(), BATCH, rng);

    std::uint8_t *d_keys, *d_nonces, *d_out;
    CUDA_CHECK(cudaMalloc(&d_keys,   BATCH * 32));
    CUDA_CHECK(cudaMalloc(&d_nonces, BATCH * 12));
    CUDA_CHECK(cudaMalloc(&d_out,    BATCH * 64));
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), BATCH * 32, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nonces, h_nonces.data(), BATCH * 12, cudaMemcpyHostToDevice));

    int blocks = (BATCH + 256 - 1) / 256;

    // Warmup
    for (int w = 0; w < 3; ++w)
        kernel_chacha20_block_batch<<<blocks, 256>>>(d_keys, d_nonces, d_out, BATCH);
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaTimer timer;
    std::vector<float> times;
    for (int r = 0; r < 7; ++r) {
        timer.start();
        kernel_chacha20_block_batch<<<blocks, 256>>>(d_keys, d_nonces, d_out, BATCH);
        times.push_back(timer.stop_ms());
    }
    std::sort(times.begin(), times.end());
    double median_ms = times[times.size() / 2];

    double total_bytes = static_cast<double>(BATCH) * 64.0;
    double mbps = (total_bytes / (median_ms / 1e3)) / (1024.0 * 1024.0);
    double blocks_per_sec = static_cast<double>(BATCH) / (median_ms / 1e3);
    double ns_per_block = (median_ms * 1e6) / BATCH;

    std::printf("  batch:         %d blocks (1M)\n", BATCH);
    std::printf("  time:          %.3f ms\n", median_ms);
    std::printf("  blocks/sec:    %.0f M\n", blocks_per_sec / 1e6);
    std::printf("  throughput:    %.0f MB/s\n", mbps);
    std::printf("  ns/block:      %.1f\n", ns_per_block);
    std::printf("\n");

    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_nonces));
    CUDA_CHECK(cudaFree(d_out));
}

// ============================================================================
// 2. Fixed-size batch AEAD roundtrip
// ============================================================================

static void bench_fixed_size_aead() {
    std::printf("--- 2. Batch AEAD Encrypt+Decrypt (fixed payload sizes) ---\n");

    static constexpr std::uint32_t SIZES[] = {32, 128, 512, 4096};
    static constexpr int COUNTS[] = {1 << 18, 1 << 17, 1 << 16, 1 << 14};

    std::printf("  %-10s  %8s  %12s  %10s  %10s  %8s\n",
                "payload", "batch", "time ms", "pkt/sec", "goodput", "overhead");

    for (int si = 0; si < 4; ++si) {
        std::uint32_t sz = SIZES[si];
        int count = COUNTS[si];

        std::vector<std::uint32_t> sizes(count, sz);
        auto res = bench_batch_encrypt_decrypt(sizes.data(), count, sz, 3, 7);

        char pkt_s[32], gp[32];
        if (res.packets_per_sec >= 1e6)
            std::snprintf(pkt_s, sizeof(pkt_s), "%.2f M", res.packets_per_sec / 1e6);
        else
            std::snprintf(pkt_s, sizeof(pkt_s), "%.0f K", res.packets_per_sec / 1e3);
        std::snprintf(gp, sizeof(gp), "%.0f MB/s", res.goodput_mbps);

        std::printf("  %-10u  %8d  %12.3f  %10s  %10s  %7.1f%%\n",
                    sz, count, res.total_ms, pkt_s, gp, res.overhead_pct);
    }
    std::printf("\n");
}

// ============================================================================
// 3. Mixed payload distribution (same as CPU benchmark)
// ============================================================================

static void bench_mixed_transport() {
    std::printf("--- 3. Transport Mixed (realistic distribution, GPU batch) ---\n");

    static constexpr int COUNT = 1 << 17; // 128K packets
    static constexpr std::uint32_t MAX_PAYLOAD = 4096;

    HostRng rng(0x1234ABCD);
    std::vector<std::uint32_t> sizes(COUNT);
    int bucket[4] = {};
    std::size_t total_payload = 0;

    for (int i = 0; i < COUNT; ++i) {
        std::uint32_t r = rng.range(0, 99);
        std::uint32_t sz;
        if (r < 40)      { sz = rng.range(1, 32);    bucket[0]++; }
        else if (r < 70) { sz = rng.range(33, 128);   bucket[1]++; }
        else if (r < 90) { sz = rng.range(129, 512);  bucket[2]++; }
        else              { sz = rng.range(513, 4096); bucket[3]++; }
        sizes[i] = sz;
        total_payload += sz;
    }

    auto res = bench_batch_encrypt_decrypt(sizes.data(), COUNT, MAX_PAYLOAD, 3, 7);

    std::printf("  packets:       %d (128K)\n", COUNT);
    std::printf("  distribution:  0-32B=%d%%  33-128B=%d%%  129-512B=%d%%  513-4096B=%d%%\n",
                bucket[0] * 100 / COUNT, bucket[1] * 100 / COUNT,
                bucket[2] * 100 / COUNT, bucket[3] * 100 / COUNT);
    std::printf("  avg payload:   %.1f B\n", res.avg_payload);
    std::printf("  GPU time:      %.3f ms\n", res.total_ms);
    std::printf("  packets/sec:   %.2f M\n", res.packets_per_sec / 1e6);
    std::printf("  goodput:       %.0f MB/s (payload only)\n", res.goodput_mbps);
    std::printf("  wire rate:     %.0f MB/s\n", res.wire_mbps);
    std::printf("  overhead:      %.1f%%\n", res.overhead_pct);
    std::printf("\n");
}

// ============================================================================
// 4. Decoy overhead (GPU batch)
// ============================================================================

static void bench_decoys_gpu(double decoy_rate) {
    std::printf("--- 4. Transport Decoys (rate=%.0f%%, GPU batch) ---\n", decoy_rate * 100);

    static constexpr int COUNT = 1 << 17;
    static constexpr std::uint32_t MAX_PAYLOAD = 4096;

    HostRng rng(0x5678CDEF);
    std::vector<std::uint32_t> all_sizes(COUNT);
    std::vector<std::uint32_t> real_sizes;
    int real_count = 0, decoy_count = 0;
    std::size_t real_payload = 0;

    for (int i = 0; i < COUNT; ++i) {
        bool decoy = (rng.range(0, 999) < static_cast<std::uint32_t>(decoy_rate * 1000));
        std::uint32_t sz;
        if (decoy) {
            sz = rng.range(0, 64);
            decoy_count++;
        } else {
            std::uint32_t r = rng.range(0, 99);
            if (r < 40)      sz = rng.range(1, 32);
            else if (r < 70) sz = rng.range(33, 128);
            else if (r < 90) sz = rng.range(129, 512);
            else              sz = rng.range(513, 4096);
            real_count++;
            real_payload += sz;
            real_sizes.push_back(sz);
        }
        all_sizes[i] = sz;
    }

    // With decoys
    auto res_all = bench_batch_encrypt_decrypt(all_sizes.data(), COUNT, MAX_PAYLOAD, 3, 7);

    // Without decoys (real packets only)
    auto res_real = bench_batch_encrypt_decrypt(real_sizes.data(),
                                                 static_cast<int>(real_sizes.size()),
                                                 MAX_PAYLOAD, 3, 7);

    double throughput_drop = 100.0 * (1.0 - res_real.total_ms / res_all.total_ms);

    std::printf("  total packets: %d (real=%d, decoy=%d)\n", COUNT, real_count, decoy_count);
    std::printf("  GPU time (all):  %.3f ms\n", res_all.total_ms);
    std::printf("  GPU time (real): %.3f ms\n", res_real.total_ms);
    std::printf("  useful goodput:  %.0f MB/s\n",
                (real_payload / (res_all.total_ms / 1e3)) / (1024.0 * 1024.0));
    std::printf("  throughput drop vs no-decoy: %.1f%%\n", throughput_drop);
    std::printf("\n");
}

// ============================================================================
// 5. Batch size scaling (GPU occupancy sweet spot)
// ============================================================================

static void bench_scaling() {
    std::printf("--- 5. Batch Size Scaling (GPU occupancy) ---\n");

    static constexpr std::uint32_t PAYLOAD = 128;
    static constexpr int BATCH_SIZES[] = {256, 1024, 4096, 16384, 65536, 262144};

    std::printf("  %-10s  %12s  %12s  %10s\n", "batch", "time ms", "pkt/sec", "MB/s");

    for (int bs : BATCH_SIZES) {
        std::vector<std::uint32_t> sizes(bs, PAYLOAD);
        auto res = bench_batch_encrypt_decrypt(sizes.data(), bs, PAYLOAD, 3, 7);

        char pkt_s[32];
        if (res.packets_per_sec >= 1e6)
            std::snprintf(pkt_s, sizeof(pkt_s), "%.2f M", res.packets_per_sec / 1e6);
        else
            std::snprintf(pkt_s, sizeof(pkt_s), "%.0f K", res.packets_per_sec / 1e3);

        std::printf("  %-10d  %12.3f  %12s  %9.0f\n",
                    bs, res.total_ms, pkt_s, res.goodput_mbps);
    }
    std::printf("\n");
}

// ============================================================================
// 6. PCIe-Aware End-to-End (H2D + Kernel + D2H)
// ============================================================================

static void bench_pcie_aware() {
    std::printf("--- 6. PCIe-Aware End-to-End (H2D + Kernel + D2H) ---\n");

    static constexpr int COUNT = 1 << 16; // 64K packets
    static constexpr std::uint32_t PAYLOADS[] = {128, 512, 4096};

    std::printf("  %-8s  %9s  %9s  %9s  %9s  %10s  %9s\n",
                "payload", "H2D ms", "kern ms", "D2H ms", "total ms",
                "eff MB/s", "kern%");

    for (auto PAYLOAD : PAYLOADS) {
        std::size_t wire_stride = PAYLOAD + BIP324_OVERHEAD;

        // Pinned host memory for accurate PCIe measurement
        std::uint8_t *h_keys, *h_nonces, *h_pt, *h_dec;
        std::uint32_t *h_sizes, *h_ok;
        CUDA_CHECK(cudaMallocHost(&h_keys,   COUNT * 32));
        CUDA_CHECK(cudaMallocHost(&h_nonces, COUNT * 12));
        CUDA_CHECK(cudaMallocHost(&h_pt,     (std::size_t)COUNT * PAYLOAD));
        CUDA_CHECK(cudaMallocHost(&h_dec,    (std::size_t)COUNT * PAYLOAD));
        CUDA_CHECK(cudaMallocHost(&h_sizes,  COUNT * sizeof(std::uint32_t)));
        CUDA_CHECK(cudaMallocHost(&h_ok,     COUNT * sizeof(std::uint32_t)));

        std::mt19937_64 rng(0x42FC1EULL);
        for (int i = 0; i < COUNT * 32; ++i)
            h_keys[i] = static_cast<std::uint8_t>(rng());
        for (int i = 0; i < COUNT * 12; ++i)
            h_nonces[i] = static_cast<std::uint8_t>(rng());
        for (std::size_t i = 0; i < (std::size_t)COUNT * PAYLOAD; ++i)
            h_pt[i] = static_cast<std::uint8_t>(rng());
        for (int i = 0; i < COUNT; ++i) h_sizes[i] = PAYLOAD;

        // Device allocations
        std::uint8_t *d_keys, *d_nonces, *d_pt, *d_wire, *d_dec;
        std::uint32_t *d_sizes, *d_ok;
        CUDA_CHECK(cudaMalloc(&d_keys,   COUNT * 32));
        CUDA_CHECK(cudaMalloc(&d_nonces, COUNT * 12));
        CUDA_CHECK(cudaMalloc(&d_pt,     (std::size_t)COUNT * PAYLOAD));
        CUDA_CHECK(cudaMalloc(&d_wire,   (std::size_t)COUNT * wire_stride));
        CUDA_CHECK(cudaMalloc(&d_dec,    (std::size_t)COUNT * PAYLOAD));
        CUDA_CHECK(cudaMalloc(&d_sizes,  COUNT * sizeof(std::uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_ok,     COUNT * sizeof(std::uint32_t)));

        int nblocks = (COUNT + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        // Warmup
        CUDA_CHECK(cudaMemcpy(d_keys, h_keys, COUNT * 32, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_nonces, h_nonces, COUNT * 12, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pt, h_pt, (std::size_t)COUNT * PAYLOAD, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sizes, h_sizes, COUNT * sizeof(std::uint32_t), cudaMemcpyHostToDevice));
        kernel_batch_aead_encrypt<<<nblocks, THREADS_PER_BLOCK>>>(
            d_keys, d_nonces, d_pt, d_sizes, d_wire, PAYLOAD, COUNT);
        kernel_batch_aead_decrypt<<<nblocks, THREADS_PER_BLOCK>>>(
            d_keys, d_nonces, d_wire, d_sizes, d_dec, d_ok, PAYLOAD, COUNT);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Measure phases separately (median of 7 runs)
        CudaTimer timer;
        std::vector<float> t_h2d(7), t_kern(7), t_d2h(7);
        for (int r = 0; r < 7; ++r) {
            // H2D phase
            timer.start();
            CUDA_CHECK(cudaMemcpy(d_keys, h_keys, COUNT * 32, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_nonces, h_nonces, COUNT * 12, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_pt, h_pt, (std::size_t)COUNT * PAYLOAD, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_sizes, h_sizes, COUNT * sizeof(std::uint32_t), cudaMemcpyHostToDevice));
            t_h2d[r] = timer.stop_ms();

            // Kernel phase (encrypt + decrypt)
            timer.start();
            kernel_batch_aead_encrypt<<<nblocks, THREADS_PER_BLOCK>>>(
                d_keys, d_nonces, d_pt, d_sizes, d_wire, PAYLOAD, COUNT);
            kernel_batch_aead_decrypt<<<nblocks, THREADS_PER_BLOCK>>>(
                d_keys, d_nonces, d_wire, d_sizes, d_dec, d_ok, PAYLOAD, COUNT);
            t_kern[r] = timer.stop_ms();

            // D2H phase (transfer decrypted data + status)
            timer.start();
            CUDA_CHECK(cudaMemcpy(h_dec, d_dec, (std::size_t)COUNT * PAYLOAD, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_ok, d_ok, COUNT * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
            t_d2h[r] = timer.stop_ms();
        }

        std::sort(t_h2d.begin(), t_h2d.end());
        std::sort(t_kern.begin(), t_kern.end());
        std::sort(t_d2h.begin(), t_d2h.end());

        float h2d  = t_h2d[3];
        float kern = t_kern[3];
        float d2h  = t_d2h[3];
        float total = h2d + kern + d2h;

        double total_payload = static_cast<double>((std::size_t)COUNT * PAYLOAD);
        double eff_mbps = (total_payload / (total / 1e3)) / (1024.0 * 1024.0);

        std::printf("  %-8u  %9.3f  %9.3f  %9.3f  %9.3f  %9.0f  %8.1f%%\n",
                    PAYLOAD, h2d, kern, d2h, total, eff_mbps,
                    100.0 * kern / total);

        CUDA_CHECK(cudaFree(d_keys));  CUDA_CHECK(cudaFree(d_nonces));
        CUDA_CHECK(cudaFree(d_pt));    CUDA_CHECK(cudaFree(d_wire));
        CUDA_CHECK(cudaFree(d_dec));   CUDA_CHECK(cudaFree(d_sizes));
        CUDA_CHECK(cudaFree(d_ok));
        CUDA_CHECK(cudaFreeHost(h_keys));  CUDA_CHECK(cudaFreeHost(h_nonces));
        CUDA_CHECK(cudaFreeHost(h_pt));    CUDA_CHECK(cudaFreeHost(h_dec));
        CUDA_CHECK(cudaFreeHost(h_sizes)); CUDA_CHECK(cudaFreeHost(h_ok));
    }
    std::printf("\n");
}

// ============================================================================
// 7. Per-Packet Latency vs Batch Size
// ============================================================================

static void bench_latency_curve() {
    std::printf("--- 7. Per-Packet Latency vs Batch Size ---\n");

    static constexpr std::uint32_t PAYLOAD = 128;
    static constexpr int BATCHES[] = {1, 4, 16, 64, 256, 1024, 4096, 16384, 65536};

    std::printf("  %-10s  %10s  %12s  %12s\n",
                "batch", "total us", "us/packet", "pkt/sec");

    for (int bs : BATCHES) {
        std::vector<std::uint32_t> sizes(bs, PAYLOAD);
        auto res = bench_batch_encrypt_decrypt(sizes.data(), bs, PAYLOAD, 5, 11);
        double us_total = res.total_ms * 1000.0;
        double us_per_pkt = us_total / bs;

        char pps[32];
        if (res.packets_per_sec >= 1e6)
            std::snprintf(pps, sizeof(pps), "%.2f M", res.packets_per_sec / 1e6);
        else if (res.packets_per_sec >= 1e3)
            std::snprintf(pps, sizeof(pps), "%.1f K", res.packets_per_sec / 1e3);
        else
            std::snprintf(pps, sizeof(pps), "%.0f", res.packets_per_sec);

        std::printf("  %-10d  %10.1f  %12.3f  %12s\n",
                    bs, us_total, us_per_pkt, pps);
    }
    std::printf("\n");
}

// ============================================================================
// 8. Multi-Stream Pipeline (Copy/Compute Overlap)
// ============================================================================

static void bench_multi_stream() {
    std::printf("--- 8. Multi-Stream Pipeline (Copy/Compute Overlap) ---\n");

    static constexpr int TOTAL = 1 << 17; // 128K packets
    static constexpr std::uint32_t PAYLOAD = 128;
    static constexpr int STREAM_COUNTS[] = {1, 2, 4, 8};
    std::size_t wire_stride = PAYLOAD + BIP324_OVERHEAD;

    // Pinned host memory (required for cudaMemcpyAsync)
    std::uint8_t *h_keys, *h_nonces, *h_pt;
    std::uint32_t *h_sizes, *h_ok;
    CUDA_CHECK(cudaMallocHost(&h_keys,   TOTAL * 32));
    CUDA_CHECK(cudaMallocHost(&h_nonces, TOTAL * 12));
    CUDA_CHECK(cudaMallocHost(&h_pt,     (std::size_t)TOTAL * PAYLOAD));
    CUDA_CHECK(cudaMallocHost(&h_sizes,  TOTAL * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMallocHost(&h_ok,     TOTAL * sizeof(std::uint32_t)));

    std::mt19937_64 rng(0x5782EA00ULL);
    for (int i = 0; i < TOTAL * 32; ++i)
        h_keys[i] = static_cast<std::uint8_t>(rng());
    for (int i = 0; i < TOTAL * 12; ++i)
        h_nonces[i] = static_cast<std::uint8_t>(rng());
    for (std::size_t i = 0; i < (std::size_t)TOTAL * PAYLOAD; ++i)
        h_pt[i] = static_cast<std::uint8_t>(rng());
    for (int i = 0; i < TOTAL; ++i) h_sizes[i] = PAYLOAD;

    // Device memory (one contiguous allocation, split across streams)
    std::uint8_t *d_keys, *d_nonces, *d_pt, *d_wire, *d_dec;
    std::uint32_t *d_sizes, *d_ok;
    CUDA_CHECK(cudaMalloc(&d_keys,   TOTAL * 32));
    CUDA_CHECK(cudaMalloc(&d_nonces, TOTAL * 12));
    CUDA_CHECK(cudaMalloc(&d_pt,     (std::size_t)TOTAL * PAYLOAD));
    CUDA_CHECK(cudaMalloc(&d_wire,   (std::size_t)TOTAL * wire_stride));
    CUDA_CHECK(cudaMalloc(&d_dec,    (std::size_t)TOTAL * PAYLOAD));
    CUDA_CHECK(cudaMalloc(&d_sizes,  TOTAL * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_ok,     TOTAL * sizeof(std::uint32_t)));

    std::printf("  %-10s  %12s  %12s  %10s\n",
                "streams", "time ms", "pkt/sec", "speedup");

    double baseline_ms = 0;

    for (int ns : STREAM_COUNTS) {
        int chunk = TOTAL / ns;
        int nblk = (chunk + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        std::vector<cudaStream_t> streams(ns);
        for (int i = 0; i < ns; ++i)
            CUDA_CHECK(cudaStreamCreate(&streams[i]));

        // Warmup (2 rounds)
        for (int w = 0; w < 2; ++w) {
            for (int i = 0; i < ns; ++i) {
                std::size_t off = (std::size_t)i * chunk;
                cudaMemcpyAsync(d_keys + off * 32, h_keys + off * 32,
                    chunk * 32, cudaMemcpyHostToDevice, streams[i]);
                cudaMemcpyAsync(d_nonces + off * 12, h_nonces + off * 12,
                    chunk * 12, cudaMemcpyHostToDevice, streams[i]);
                cudaMemcpyAsync(d_pt + off * PAYLOAD, h_pt + off * PAYLOAD,
                    (std::size_t)chunk * PAYLOAD, cudaMemcpyHostToDevice, streams[i]);
                cudaMemcpyAsync(d_sizes + off, h_sizes + off,
                    chunk * sizeof(std::uint32_t), cudaMemcpyHostToDevice, streams[i]);

                kernel_batch_aead_encrypt<<<nblk, THREADS_PER_BLOCK, 0, streams[i]>>>(
                    d_keys + off * 32, d_nonces + off * 12,
                    d_pt + off * PAYLOAD, d_sizes + off,
                    d_wire + off * wire_stride, PAYLOAD, chunk);
                kernel_batch_aead_decrypt<<<nblk, THREADS_PER_BLOCK, 0, streams[i]>>>(
                    d_keys + off * 32, d_nonces + off * 12,
                    d_wire + off * wire_stride, d_sizes + off,
                    d_dec + off * PAYLOAD, d_ok + off, PAYLOAD, chunk);

                cudaMemcpyAsync(h_ok + off, d_ok + off,
                    chunk * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, streams[i]);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Measure (median of 7, host timing for cross-stream accuracy)
        std::vector<double> times(7);
        for (int r = 0; r < 7; ++r) {
            CUDA_CHECK(cudaDeviceSynchronize());
            auto t0 = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < ns; ++i) {
                std::size_t off = (std::size_t)i * chunk;
                cudaMemcpyAsync(d_keys + off * 32, h_keys + off * 32,
                    chunk * 32, cudaMemcpyHostToDevice, streams[i]);
                cudaMemcpyAsync(d_nonces + off * 12, h_nonces + off * 12,
                    chunk * 12, cudaMemcpyHostToDevice, streams[i]);
                cudaMemcpyAsync(d_pt + off * PAYLOAD, h_pt + off * PAYLOAD,
                    (std::size_t)chunk * PAYLOAD, cudaMemcpyHostToDevice, streams[i]);
                cudaMemcpyAsync(d_sizes + off, h_sizes + off,
                    chunk * sizeof(std::uint32_t), cudaMemcpyHostToDevice, streams[i]);

                kernel_batch_aead_encrypt<<<nblk, THREADS_PER_BLOCK, 0, streams[i]>>>(
                    d_keys + off * 32, d_nonces + off * 12,
                    d_pt + off * PAYLOAD, d_sizes + off,
                    d_wire + off * wire_stride, PAYLOAD, chunk);
                kernel_batch_aead_decrypt<<<nblk, THREADS_PER_BLOCK, 0, streams[i]>>>(
                    d_keys + off * 32, d_nonces + off * 12,
                    d_wire + off * wire_stride, d_sizes + off,
                    d_dec + off * PAYLOAD, d_ok + off, PAYLOAD, chunk);

                cudaMemcpyAsync(h_ok + off, d_ok + off,
                    chunk * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, streams[i]);
            }

            CUDA_CHECK(cudaDeviceSynchronize());
            auto t1 = std::chrono::high_resolution_clock::now();
            times[r] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        std::sort(times.begin(), times.end());
        double ms = times[3];

        if (ns == 1) baseline_ms = ms;
        double speedup = (baseline_ms > 0) ? baseline_ms / ms : 1.0;
        double pps = TOTAL / (ms / 1e3);

        char pps_s[32];
        if (pps >= 1e6)
            std::snprintf(pps_s, sizeof(pps_s), "%.2f M", pps / 1e6);
        else
            std::snprintf(pps_s, sizeof(pps_s), "%.0f K", pps / 1e3);

        std::printf("  %-10d  %12.3f  %12s  %9.2fx\n", ns, ms, pps_s, speedup);

        for (auto& s : streams) CUDA_CHECK(cudaStreamDestroy(s));
    }

    // Verify correctness on last run
    int ok_count = 0;
    for (int i = 0; i < TOTAL; ++i) ok_count += h_ok[i];
    if (ok_count != TOTAL)
        std::printf("  WARNING: %d/%d decrypts failed!\n", TOTAL - ok_count, TOTAL);

    CUDA_CHECK(cudaFree(d_keys));  CUDA_CHECK(cudaFree(d_nonces));
    CUDA_CHECK(cudaFree(d_pt));    CUDA_CHECK(cudaFree(d_wire));
    CUDA_CHECK(cudaFree(d_dec));   CUDA_CHECK(cudaFree(d_sizes));
    CUDA_CHECK(cudaFree(d_ok));
    CUDA_CHECK(cudaFreeHost(h_keys));  CUDA_CHECK(cudaFreeHost(h_nonces));
    CUDA_CHECK(cudaFreeHost(h_pt));    CUDA_CHECK(cudaFreeHost(h_sizes));
    CUDA_CHECK(cudaFreeHost(h_ok));

    std::printf("\n");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    // Print GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("================================================================\n");
    std::printf("  BIP-324 Transport Benchmark — CUDA Batch Parallel\n");
    std::printf("================================================================\n");
    std::printf("  GPU:           %s\n", prop.name);
    std::printf("  SMs:           %d\n", prop.multiProcessorCount);
#if CUDART_VERSION >= 13000
    { int _clk = 0; cudaDeviceGetAttribute(&_clk, cudaDevAttrClockRate, 0);
      std::printf("  Clock:         %d MHz\n", _clk / 1000); }
#else
    std::printf("  Clock:         %d MHz\n", prop.clockRate / 1000);
#endif
    std::printf("  Memory:        %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
    std::printf("  Compute:       sm_%d%d\n", prop.major, prop.minor);
    std::printf("================================================================\n\n");

    // 1. ChaCha20 throughput ceiling
    bench_chacha20_blocks();

    // 2. Fixed-size AEAD roundtrip
    bench_fixed_size_aead();

    // 3. Mixed traffic distribution
    bench_mixed_transport();

    // 4. Decoy overhead
    bench_decoys_gpu(0.05);
    bench_decoys_gpu(0.20);

    // 5. Batch size scaling
    bench_scaling();

    // 6. PCIe-aware end-to-end
    bench_pcie_aware();

    // 7. Per-packet latency curve
    bench_latency_curve();

    // 8. Multi-stream pipeline
    bench_multi_stream();

    std::printf("================================================================\n");
    std::printf("  Done.\n");
    std::printf("================================================================\n");

    return 0;
}
