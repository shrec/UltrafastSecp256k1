// ============================================================================
// BIP-324 Benchmark Suite
// ============================================================================
// Measures throughput of the BIP-324 v2 encrypted transport stack:
//   - ChaCha20 stream cipher (keystream gen + encrypt)
//   - Poly1305 MAC
//   - ChaCha20-Poly1305 AEAD (encrypt + decrypt)
//   - HMAC-SHA256 / HKDF-SHA256 key derivation
//   - ElligatorSwift encode / ECDH
//   - Full BIP-324 session (handshake + packet encrypt/decrypt)
//
// All operations benchmarked at multiple payload sizes to show throughput
// scaling: 64B (control msgs), 256B (typical), 1KB, 4KB (blocks).
// ============================================================================

#include "secp256k1/chacha20_poly1305.hpp"
#include "secp256k1/hkdf.hpp"
#include "secp256k1/ellswift.hpp"
#include "secp256k1/bip324.hpp"
#include "secp256k1/benchmark_harness.hpp"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>
#include <vector>

static bench::Harness H(500, 11);

// Fixed test key / nonce (not secret — benchmark only)
static const std::uint8_t KEY[32] = {
    0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
    0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,
    0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,
    0x18,0x19,0x1a,0x1b,0x1c,0x1d,0x1e,0x1f
};
static const std::uint8_t NONCE[12] = {
    0x00,0x00,0x00,0x09,0x00,0x00,0x00,0x4a,
    0x00,0x00,0x00,0x00
};
static const std::uint8_t PRIVKEY[32] = {
    0xe8,0xf3,0x2e,0x72,0x3d,0xec,0xf4,0x05,
    0x1a,0xef,0xac,0x8e,0x2c,0x93,0xc9,0xc5,
    0xb2,0x14,0x31,0x38,0x17,0xcd,0xb0,0x1a,
    0x14,0x94,0xb9,0x17,0xc8,0x43,0x6b,0x35
};

// Payload sizes to benchmark
static constexpr std::size_t SIZES[] = { 64, 256, 1024, 4096 };
static constexpr int         N_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);

// Iteration counts (scaled inversely by payload size for ~constant wall time)
static constexpr int ITERS_CHACHA[]  = { 100000, 50000, 20000, 5000 };
static constexpr int ITERS_AEAD[]    = {  50000, 20000, 10000, 2000 };
static constexpr int ITERS_SESSION[] = {  10000,  5000,  2000, 1000 };

static void print_throughput(const char* label, double ns_per_iter, std::size_t bytes) {
    double mb_per_sec = (static_cast<double>(bytes) / (ns_per_iter / 1e9)) / (1024.0 * 1024.0);
    std::printf("  %-38s %8.1f ns   %7.1f MB/s   (%zu B)\n",
                label, ns_per_iter, mb_per_sec, bytes);
}

// ============================================================================
int main() {
    bench::pin_thread_and_elevate();

    std::printf("================================================================\n");
    std::printf("  BIP-324 Benchmark Suite\n");
    std::printf("================================================================\n");
    H.print_config();
    std::printf("\n");

    // Pre-allocate buffers for the largest size
    std::vector<std::uint8_t> buf(SIZES[N_SIZES - 1]);
    std::vector<std::uint8_t> out(SIZES[N_SIZES - 1] + 16);
    std::uint8_t tag[16];
    // Fill with non-zero data
    for (std::size_t i = 0; i < buf.size(); ++i)
        buf[i] = static_cast<std::uint8_t>(i & 0xFF);

    // =====================================================================
    // 1. ChaCha20 keystream generation
    // =====================================================================
    std::printf("--- ChaCha20 Block (64-byte keystream) ---\n");
    {
        std::uint8_t block[64];
        double ns = H.run(200000, [&]() {
            secp256k1::chacha20_block(KEY, NONCE, 1, block);
            bench::DoNotOptimize(block);
        });
        std::printf("  chacha20_block (1 block)              %8.1f ns   %7.1f MB/s\n",
                    ns, (64.0 / (ns / 1e9)) / (1024.0 * 1024.0));
    }
    std::printf("\n");

    // =====================================================================
    // 2. ChaCha20 stream encrypt at various sizes
    // =====================================================================
    std::printf("--- ChaCha20 Encrypt ---\n");
    for (int si = 0; si < N_SIZES; ++si) {
        std::size_t sz = SIZES[si];
        std::vector<std::uint8_t> data(sz);
        std::memcpy(data.data(), buf.data(), sz);

        char label[64];
        std::snprintf(label, sizeof(label), "chacha20_crypt %zu B", sz);
        double ns = H.run(ITERS_CHACHA[si], [&]() {
            secp256k1::chacha20_crypt(KEY, NONCE, 1, data.data(), sz);
            bench::DoNotOptimize(data[0]);
        });
        print_throughput(label, ns, sz);
    }
    std::printf("\n");

    // =====================================================================
    // 3. Poly1305 MAC at various sizes
    // =====================================================================
    std::printf("--- Poly1305 MAC ---\n");
    for (int si = 0; si < N_SIZES; ++si) {
        std::size_t sz = SIZES[si];
        char label[64];
        std::snprintf(label, sizeof(label), "poly1305_mac %zu B", sz);
        double ns = H.run(ITERS_CHACHA[si], [&]() {
            auto t = secp256k1::poly1305_mac(KEY, buf.data(), sz);
            bench::DoNotOptimize(t);
        });
        print_throughput(label, ns, sz);
    }
    std::printf("\n");

    // =====================================================================
    // 4. ChaCha20-Poly1305 AEAD encrypt + decrypt
    // =====================================================================
    std::printf("--- ChaCha20-Poly1305 AEAD Encrypt ---\n");
    {
        std::uint8_t aad[12] = {0x50,0x51,0x52,0x53,0xc0,0xc1,0xc2,0xc3,0xc4,0xc5,0xc6,0xc7};
        for (int si = 0; si < N_SIZES; ++si) {
            std::size_t sz = SIZES[si];
            std::vector<std::uint8_t> ct(sz);

            char label[64];
            std::snprintf(label, sizeof(label), "aead_encrypt %zu B", sz);
            double ns = H.run(ITERS_AEAD[si], [&]() {
                secp256k1::aead_chacha20_poly1305_encrypt(
                    KEY, NONCE, aad, sizeof(aad),
                    buf.data(), sz, ct.data(), tag);
                bench::DoNotOptimize(ct[0]);
            });
            print_throughput(label, ns, sz);
        }
    }
    std::printf("\n");

    std::printf("--- ChaCha20-Poly1305 AEAD Decrypt ---\n");
    {
        std::uint8_t aad[12] = {0x50,0x51,0x52,0x53,0xc0,0xc1,0xc2,0xc3,0xc4,0xc5,0xc6,0xc7};
        for (int si = 0; si < N_SIZES; ++si) {
            std::size_t sz = SIZES[si];
            // Produce valid ciphertext + tag first
            std::vector<std::uint8_t> ct(sz);
            secp256k1::aead_chacha20_poly1305_encrypt(
                KEY, NONCE, aad, sizeof(aad),
                buf.data(), sz, ct.data(), tag);

            std::vector<std::uint8_t> pt(sz);
            char label[64];
            std::snprintf(label, sizeof(label), "aead_decrypt %zu B", sz);
            double ns = H.run(ITERS_AEAD[si], [&]() {
                bool ok = secp256k1::aead_chacha20_poly1305_decrypt(
                    KEY, NONCE, aad, sizeof(aad),
                    ct.data(), sz, tag, pt.data());
                bench::DoNotOptimize(ok);
            });
            print_throughput(label, ns, sz);
        }
    }
    std::printf("\n");

    // =====================================================================
    // 5. HMAC-SHA256
    // =====================================================================
    std::printf("--- HMAC-SHA256 ---\n");
    for (int si = 0; si < N_SIZES; ++si) {
        std::size_t sz = SIZES[si];
        char label[64];
        std::snprintf(label, sizeof(label), "hmac_sha256 %zu B", sz);
        double ns = H.run(ITERS_AEAD[si], [&]() {
            auto h = secp256k1::hmac_sha256(KEY, 32, buf.data(), sz);
            bench::DoNotOptimize(h);
        });
        print_throughput(label, ns, sz);
    }
    std::printf("\n");

    // =====================================================================
    // 6. HKDF-SHA256 Extract + Expand
    // =====================================================================
    std::printf("--- HKDF-SHA256 ---\n");
    {
        const char* salt = "bitcoin_v2_shared_secret";
        auto salt_len = std::strlen(salt);

        double ns_extract = H.run(100000, [&]() {
            auto prk = secp256k1::hkdf_sha256_extract(
                reinterpret_cast<const std::uint8_t*>(salt), salt_len,
                KEY, 32);
            bench::DoNotOptimize(prk);
        });
        std::printf("  hkdf_extract (32B IKM)                %8.1f ns\n", ns_extract);

        auto prk = secp256k1::hkdf_sha256_extract(
            reinterpret_cast<const std::uint8_t*>(salt), salt_len, KEY, 32);

        const char* info = "initiator_L";
        auto info_len = std::strlen(info);
        std::uint8_t okm[32];
        double ns_expand = H.run(100000, [&]() {
            secp256k1::hkdf_sha256_expand(prk.data(),
                reinterpret_cast<const std::uint8_t*>(info), info_len,
                okm, 32);
            bench::DoNotOptimize(okm);
        });
        std::printf("  hkdf_expand  (32B OKM)                %8.1f ns\n", ns_expand);
        std::printf("  hkdf_total   (extract+expand)         %8.1f ns\n",
                    ns_extract + ns_expand);
    }
    std::printf("\n");

    // =====================================================================
    // 7. ElligatorSwift encode (private key -> 64B encoding)
    // =====================================================================
    std::printf("--- ElligatorSwift ---\n");
    {
        double ns_create = H.run(200, [&]() {
            auto enc = secp256k1::ellswift_create(
                secp256k1::fast::Scalar::from_bytes(PRIVKEY));
            bench::DoNotOptimize(enc);
        });
        std::printf("  ellswift_create                       %8.1f ns   (%8.1f us)\n",
                    ns_create, ns_create / 1000.0);

        // Prepare two encodings for XDH
        auto enc_a = secp256k1::ellswift_create(
            secp256k1::fast::Scalar::from_bytes(PRIVKEY));
        std::uint8_t privkey2[32];
        std::memcpy(privkey2, PRIVKEY, 32);
        privkey2[0] ^= 0x42;
        auto enc_b = secp256k1::ellswift_create(
            secp256k1::fast::Scalar::from_bytes(privkey2));

        double ns_xdh = H.run(500, [&]() {
            auto privk = secp256k1::fast::Scalar::from_bytes(PRIVKEY);
            auto secret = secp256k1::ellswift_xdh(
                enc_a.data(), enc_b.data(), privk, true);
            bench::DoNotOptimize(secret);
        });
        std::printf("  ellswift_xdh (ECDH)                   %8.1f ns   (%8.1f us)\n",
                    ns_xdh, ns_xdh / 1000.0);
    }
    std::printf("\n");

    // =====================================================================
    // 8. Full BIP-324 session: handshake
    // =====================================================================
    std::printf("--- BIP-324 Session ---\n");
    {
        double ns_handshake = H.run(100, [&]() {
            secp256k1::Bip324Session initiator(true);
            secp256k1::Bip324Session responder(false);

            responder.complete_handshake(initiator.our_ellswift_encoding().data());
            initiator.complete_handshake(responder.our_ellswift_encoding().data());
            bench::DoNotOptimize(initiator.session_id());
        });
        std::printf("  full_handshake (both sides)            %8.1f ns   (%8.1f us)\n",
                    ns_handshake, ns_handshake / 1000.0);
    }
    std::printf("\n");

    // =====================================================================
    // 9. Full BIP-324 session: packet encrypt/decrypt at various sizes
    // =====================================================================
    std::printf("--- BIP-324 Packet Encrypt ---\n");
    {
        // Set up an established session pair
        secp256k1::Bip324Session initiator(true, PRIVKEY);
        std::uint8_t resp_priv[32];
        std::memcpy(resp_priv, PRIVKEY, 32);
        resp_priv[0] ^= 0x42;
        secp256k1::Bip324Session responder(false, resp_priv);

        responder.complete_handshake(initiator.our_ellswift_encoding().data());
        initiator.complete_handshake(responder.our_ellswift_encoding().data());

        for (int si = 0; si < N_SIZES; ++si) {
            std::size_t sz = SIZES[si];
            char label[64];
            std::snprintf(label, sizeof(label), "session_encrypt %zu B", sz);
            double ns = H.run(ITERS_SESSION[si], [&]() {
                auto pkt = initiator.encrypt(buf.data(), sz);
                bench::DoNotOptimize(pkt);
            });
            print_throughput(label, ns, sz);
        }
    }
    std::printf("\n");

    std::printf("--- BIP-324 Packet Decrypt ---\n");
    {
        // Benchmark encrypt→decrypt round-trip, report decrypt half.
        // Each iteration encrypts (to advance sender nonce) then decrypts
        // (to advance receiver nonce), so nonces stay in sync.
        secp256k1::Bip324Session initiator(true, PRIVKEY);
        std::uint8_t resp_priv[32];
        std::memcpy(resp_priv, PRIVKEY, 32);
        resp_priv[0] ^= 0x42;
        secp256k1::Bip324Session responder(false, resp_priv);

        responder.complete_handshake(initiator.our_ellswift_encoding().data());
        initiator.complete_handshake(responder.our_ellswift_encoding().data());

        for (int si = 0; si < N_SIZES; ++si) {
            std::size_t sz = SIZES[si];

            char label[64];
            std::snprintf(label, sizeof(label), "session_roundtrip %zu B", sz);
            double ns = H.run(ITERS_SESSION[si], [&]() {
                // Encrypt (advances initiator nonce)
                auto pkt = initiator.encrypt(buf.data(), sz);
                // Decrypt (advances responder nonce — stays in sync)
                auto dec = responder.decrypt(
                    pkt.data(), pkt.data() + 3, pkt.size() - 3);
                bench::DoNotOptimize(dec);
            });
            print_throughput(label, ns, sz);
        }
    }
    std::printf("\n");

    // =====================================================================
    // Summary table
    // =====================================================================
    std::printf("================================================================\n");
    std::printf("  Done.\n");
    std::printf("================================================================\n");

    return 0;
}
