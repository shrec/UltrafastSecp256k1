// ============================================================================
// BIP-324 Transport Benchmark — Real-World Protocol Simulation
// ============================================================================
// Extends the primitive benchmark with production-grade metrics:
//
//   1. transport_mixed   — realistic payload size distribution
//   2. transport_decoys  — measures CPU tax from decoy (ignore) packets
//   3. latency_mode      — per-packet p50/p95/p99 latency histograms
//   4. e2e_socket        — localhost TCP full-duplex with syscall overhead
//
// Reference: BIP-324 Section "Packet handling"
// ============================================================================

#include "secp256k1/bip324.hpp"
#include "secp256k1/benchmark_harness.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <numeric>
#include <vector>

// Socket headers (Linux / POSIX)
#if defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__)
#define HAS_SOCKETS 1
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <thread>
#include <atomic>
#else
#define HAS_SOCKETS 0
#endif

// ============================================================================
// Helpers
// ============================================================================

static bench::Harness H(200, 11);

static const std::uint8_t PRIVKEY_A[32] = {
    0xe8,0xf3,0x2e,0x72,0x3d,0xec,0xf4,0x05,
    0x1a,0xef,0xac,0x8e,0x2c,0x93,0xc9,0xc5,
    0xb2,0x14,0x31,0x38,0x17,0xcd,0xb0,0x1a,
    0x14,0x94,0xb9,0x17,0xc8,0x43,0x6b,0x35
};
static const std::uint8_t PRIVKEY_B[32] = {
    0xaa,0xbb,0xcc,0xdd,0x11,0x22,0x33,0x44,
    0x55,0x66,0x77,0x88,0x99,0x00,0xab,0xcd,
    0xef,0x01,0x23,0x45,0x67,0x89,0xab,0xcd,
    0xef,0xfe,0xdc,0xba,0x98,0x76,0x54,0x32
};

// Simple LCG PRNG for deterministic, fast pseudo-random sizes
struct FastRng {
    std::uint64_t state;
    explicit FastRng(std::uint64_t seed = 0xDEADBEEF42ULL) : state(seed) {}
    std::uint32_t next() noexcept {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<std::uint32_t>(state >> 32);
    }
    // Uniform in [lo, hi]
    std::uint32_t range(std::uint32_t lo, std::uint32_t hi) noexcept {
        return lo + (next() % (hi - lo + 1));
    }
};

// BIP-324 per-packet wire overhead: 3B encrypted length + 16B Poly1305 tag
static constexpr std::size_t BIP324_OVERHEAD = 3 + 16;

// Set up a connected session pair (deterministic keys for reproducibility)
struct SessionPair {
    secp256k1::Bip324Session initiator;
    secp256k1::Bip324Session responder;

    SessionPair()
        : initiator(true, PRIVKEY_A), responder(false, PRIVKEY_B) {
        responder.complete_handshake(initiator.our_ellswift_encoding().data());
        initiator.complete_handshake(responder.our_ellswift_encoding().data());
    }
};

static std::vector<std::uint8_t> decrypt_packet(secp256k1::Bip324Session& session,
                                                const std::vector<std::uint8_t>& packet) {
    std::vector<std::uint8_t> plaintext;
    if (!session.decrypt(packet.data(), packet.data() + 3, packet.size() - 3, plaintext)) {
        std::fprintf(stderr, "BIP-324 transport benchmark decrypt failed\n");
        std::abort();
    }
    return plaintext;
}

// Percentile from sorted array (linear interpolation)
static double percentile(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) return 0.0;
    double idx = p * static_cast<double>(sorted.size() - 1);
    auto lo = static_cast<std::size_t>(idx);
    auto hi = lo + 1;
    if (hi >= sorted.size()) return sorted.back();
    double frac = idx - static_cast<double>(lo);
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

// ============================================================================
// 1. transport_mixed — Realistic payload size distribution
// ============================================================================
// Bitcoin P2P traffic profile (Erlay + compact blocks + inv/getdata flow):
//   40% 0–32 B   (pings, inv, sendcmpct, verack, feefilter)
//   30% 33–128 B  (getheaders, getdata, addr)
//   20% 129–512 B (cmpctblock, headers, blocktxn)
//   10% 513–4096 B (full blocks, large batches)
// ============================================================================

static void bench_transport_mixed(int total_packets) {
    std::printf("--- Transport Mixed (realistic distribution) ---\n");

    SessionPair sess;
    FastRng rng(0x1234ABCD);

    // Pre-generate the packet schedule
    struct PacketSpec { std::size_t payload_size; };
    std::vector<PacketSpec> schedule(static_cast<std::size_t>(total_packets));

    std::size_t total_payload_bytes = 0;
    std::size_t total_wire_bytes = 0;
    int bucket_count[4] = {};

    for (std::size_t i = 0; i < static_cast<std::size_t>(total_packets); ++i) {
        std::uint32_t r = rng.range(0, 99);
        std::size_t sz;
        if (r < 40) {
            sz = rng.range(1, 32);
            bucket_count[0]++;
        } else if (r < 70) {
            sz = rng.range(33, 128);
            bucket_count[1]++;
        } else if (r < 90) {
            sz = rng.range(129, 512);
            bucket_count[2]++;
        } else {
            sz = rng.range(513, 4096);
            bucket_count[3]++;
        }
        schedule[i].payload_size = sz;
        total_payload_bytes += sz;
        total_wire_bytes += sz + BIP324_OVERHEAD;
    }

    // Fill source buffer
    std::vector<std::uint8_t> src(4096);
    for (std::size_t i = 0; i < src.size(); ++i)
        src[i] = static_cast<std::uint8_t>(i & 0xFF);

    // Timed roundtrip: encrypt all + decrypt all (nonce-synced)
    double ns = H.run(1, [&]() {
        // Re-create sessions per pass to reset nonce counters
        SessionPair sp;
        for (std::size_t i = 0; i < static_cast<std::size_t>(total_packets); ++i) {
            auto pkt = sp.initiator.encrypt(src.data(), schedule[i].payload_size);
            auto dec = decrypt_packet(sp.responder, pkt);
            bench::DoNotOptimize(dec);
        }
    });

    double total_sec = ns / 1e9;
    double pps = static_cast<double>(total_packets) / total_sec;
    double goodput_mbps = (static_cast<double>(total_payload_bytes) / total_sec) / (1024.0 * 1024.0);
    double wire_mbps = (static_cast<double>(total_wire_bytes) / total_sec) / (1024.0 * 1024.0);
    double overhead_pct = 100.0 * (1.0 - static_cast<double>(total_payload_bytes)
                                          / static_cast<double>(total_wire_bytes));
    double avg_payload = static_cast<double>(total_payload_bytes) / total_packets;

    std::printf("  packets:       %d\n", total_packets);
    std::printf("  distribution:  0-32B=%d%%  33-128B=%d%%  129-512B=%d%%  513-4096B=%d%%\n",
                bucket_count[0] * 100 / total_packets,
                bucket_count[1] * 100 / total_packets,
                bucket_count[2] * 100 / total_packets,
                bucket_count[3] * 100 / total_packets);
    std::printf("  avg payload:   %.1f B\n", avg_payload);
    std::printf("  total time:    %.3f ms\n", ns / 1e6);
    std::printf("  packets/sec:   %.0f\n", pps);
    std::printf("  goodput:       %.1f MB/s (payload only)\n", goodput_mbps);
    std::printf("  wire rate:     %.1f MB/s (with BIP324 framing)\n", wire_mbps);
    std::printf("  overhead:      %.1f%%\n", overhead_pct);
    std::printf("\n");
}

// ============================================================================
// 2. transport_with_decoys — Decoy (ignore) packet CPU tax
// ============================================================================
// BIP-324 supports "ignore" packets (decoys) for traffic analysis resistance.
// These are encrypted normally but the receiver discards the plaintext.
// We measure the CPU cost of processing decoys at various injection rates.
// ============================================================================

static void bench_transport_decoys(int total_packets, double decoy_rate) {
    char hdr[64];
    std::snprintf(hdr, sizeof(hdr), "--- Transport Decoys (rate=%.0f%%) ---", decoy_rate * 100);
    std::printf("%s\n", hdr);

    FastRng rng(0x5678CDEF);

    // Pre-generate schedule: each packet is either real or decoy
    struct PacketSpec {
        std::size_t payload_size;
        bool is_decoy;
    };
    std::vector<PacketSpec> schedule(static_cast<std::size_t>(total_packets));

    int real_count = 0, decoy_count = 0;
    std::size_t real_payload_bytes = 0, total_wire_bytes = 0;

    for (std::size_t i = 0; i < static_cast<std::size_t>(total_packets); ++i) {
        bool decoy = (rng.range(0, 999) < static_cast<std::uint32_t>(decoy_rate * 1000));
        // Decoys are typically small (0-64B)
        std::size_t sz;
        if (decoy) {
            sz = rng.range(0, 64);
            decoy_count++;
        } else {
            // Real packets: use the same mixed distribution
            std::uint32_t r = rng.range(0, 99);
            if (r < 40)      sz = rng.range(1, 32);
            else if (r < 70) sz = rng.range(33, 128);
            else if (r < 90) sz = rng.range(129, 512);
            else              sz = rng.range(513, 4096);
            real_count++;
            real_payload_bytes += sz;
        }
        schedule[i] = {sz, decoy};
        total_wire_bytes += sz + BIP324_OVERHEAD;
    }

    std::vector<std::uint8_t> src(4096);
    for (std::size_t i = 0; i < src.size(); ++i)
        src[i] = static_cast<std::uint8_t>(i & 0xFF);

    // Measure: encrypt + decrypt all. For decoys, we still decrypt (cost is the same),
    // but the receiver would discard the plaintext.
    double ns = H.run(1, [&]() {
        SessionPair sp;
        for (std::size_t i = 0; i < static_cast<std::size_t>(total_packets); ++i) {
            auto pkt = sp.initiator.encrypt(src.data(), schedule[i].payload_size);
            auto dec = decrypt_packet(sp.responder, pkt);
            bench::DoNotOptimize(dec);
        }
    });

    double total_sec = ns / 1e9;
    double useful_mbps = (static_cast<double>(real_payload_bytes) / total_sec) / (1024.0 * 1024.0);
    double wire_mbps = (static_cast<double>(total_wire_bytes) / total_sec) / (1024.0 * 1024.0);
    double decoy_tax = (real_count > 0)
        ? 100.0 * static_cast<double>(decoy_count) / static_cast<double>(total_packets)
        : 0.0;

    // Also run without decoys for comparison (same real packet count)
    double ns_no_decoy = H.run(1, [&]() {
        SessionPair sp;
        for (std::size_t i = 0; i < static_cast<std::size_t>(total_packets); ++i) {
            if (schedule[i].is_decoy) continue; // skip decoys
            auto pkt = sp.initiator.encrypt(src.data(), schedule[i].payload_size);
            auto dec = decrypt_packet(sp.responder, pkt);
            bench::DoNotOptimize(dec);
        }
    });

    double throughput_drop = 100.0 * (1.0 - (ns_no_decoy > 0 ? ns_no_decoy / ns : 0.0));

    std::printf("  total packets: %d (real=%d, decoy=%d)\n", total_packets, real_count, decoy_count);
    std::printf("  decoy rate:    %.1f%%\n", decoy_rate * 100);
    std::printf("  total time:    %.3f ms\n", ns / 1e6);
    std::printf("  useful goodput:%.1f MB/s (payload of real packets only)\n", useful_mbps);
    std::printf("  wire rate:     %.1f MB/s\n", wire_mbps);
    std::printf("  decoy CPU tax: %.1f%% of total time spent on decoys\n", decoy_tax);
    std::printf("  throughput drop vs no-decoy: %.1f%%\n", throughput_drop);
    std::printf("\n");
}

// ============================================================================
// 3. latency_mode — Per-packet timing with percentile histogram
// ============================================================================
// Measures individual packet encrypt+decrypt latencies and reports
// p50 / p95 / p99 / max to expose tail latency behavior.
// ============================================================================

static void bench_latency_mode(int num_packets) {
    std::printf("--- Latency Mode (per-packet ns) ---\n");

    // Size buckets matching Bitcoin P2P
    struct SizeBucket {
        const char* name;
        std::size_t lo, hi;
    };
    static const SizeBucket BUCKETS[] = {
        {"  0-32 B  (control)  ", 1, 32},
        {"  33-128 B (typical) ", 33, 128},
        {"  129-512 B (medium) ", 129, 512},
        {"  513-4096 B (large) ", 513, 4096},
    };

    std::vector<std::uint8_t> src(4096);
    for (std::size_t i = 0; i < src.size(); ++i)
        src[i] = static_cast<std::uint8_t>(i & 0xFF);

    FastRng rng(0xABCD1234);

    std::printf("  %-22s  %8s  %8s  %8s  %8s  %8s\n",
                "bucket", "p50", "p95", "p99", "max", "ns/pkt");

    for (const auto& bkt : BUCKETS) {
        // Collect individual roundtrip latencies
        std::vector<double> latencies;
        latencies.reserve(num_packets);

        SessionPair sp;

        // Warmup
        for (int w = 0; w < 100; ++w) {
            std::size_t sz = rng.range(static_cast<std::uint32_t>(bkt.lo),
                                        static_cast<std::uint32_t>(bkt.hi));
            auto pkt = sp.initiator.encrypt(src.data(), sz);
            auto dec = decrypt_packet(sp.responder, pkt);
            bench::DoNotOptimize(dec);
        }

        // Measure
        for (std::size_t i = 0; i < static_cast<std::size_t>(num_packets); ++i) {
            std::size_t sz = rng.range(static_cast<std::uint32_t>(bkt.lo),
                                        static_cast<std::uint32_t>(bkt.hi));

            std::uint64_t t0 = bench::Timer::now();
            auto pkt = sp.initiator.encrypt(src.data(), sz);
            auto dec = decrypt_packet(sp.responder, pkt);
            bench::DoNotOptimize(dec);
            std::uint64_t t1 = bench::Timer::now();

            latencies.push_back(bench::Timer::ticks_to_ns(t1 - t0));
        }

        std::sort(latencies.begin(), latencies.end());

        double p50 = percentile(latencies, 0.50);
        double p95 = percentile(latencies, 0.95);
        double p99 = percentile(latencies, 0.99);
        double mx  = latencies.back();
        double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0)
                     / static_cast<double>(latencies.size());

        std::printf("  %-22s  %7.0f   %7.0f   %7.0f   %7.0f   %7.0f\n",
                    bkt.name, p50, p95, p99, mx, avg);
    }

    // Also measure mixed distribution
    {
        std::vector<double> latencies;
        latencies.reserve(num_packets);

        SessionPair sp;
        // Warmup
        for (int w = 0; w < 100; ++w) {
            const std::size_t sz = rng.range(1, 4096);
            auto pkt = sp.initiator.encrypt(src.data(), sz);
            auto dec = decrypt_packet(sp.responder, pkt);
            bench::DoNotOptimize(dec);
        }

        for (std::size_t i = 0; i < static_cast<std::size_t>(num_packets); ++i) {
            std::uint32_t r = rng.range(0, 99);
            std::size_t sz;
            if (r < 40)      sz = rng.range(1, 32);
            else if (r < 70) sz = rng.range(33, 128);
            else if (r < 90) sz = rng.range(129, 512);
            else              sz = rng.range(513, 4096);

            std::uint64_t t0 = bench::Timer::now();
            auto pkt = sp.initiator.encrypt(src.data(), sz);
            auto dec = decrypt_packet(sp.responder, pkt);
            bench::DoNotOptimize(dec);
            std::uint64_t t1 = bench::Timer::now();

            latencies.push_back(bench::Timer::ticks_to_ns(t1 - t0));
        }

        std::sort(latencies.begin(), latencies.end());

        double p50 = percentile(latencies, 0.50);
        double p95 = percentile(latencies, 0.95);
        double p99 = percentile(latencies, 0.99);
        double mx  = latencies.back();
        double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0)
                     / static_cast<double>(latencies.size());

        // Jitter = stddev
        double var = 0.0;
        for (auto v : latencies) {
            double d = v - avg;
            var += d * d;
        }
        double jitter = std::sqrt(var / static_cast<double>(latencies.size()));

        std::printf("  %-22s  %7.0f   %7.0f   %7.0f   %7.0f   %7.0f\n",
                    "  mixed (real dist)  ", p50, p95, p99, mx, avg);
        std::printf("  jitter (stddev):   %.0f ns\n", jitter);
    }
    std::printf("\n");
}

// ============================================================================
// 4. e2e_socket — Localhost TCP full-duplex benchmark
// ============================================================================
// Creates a TCP socketpair (localhost), sets TCP_NODELAY, and measures
// end-to-end encrypt→send→recv→decrypt latency including syscall overhead.
// ============================================================================

#if HAS_SOCKETS

// Send exactly n bytes
static bool send_all(int fd, const void* buf, std::size_t n) {
    auto p = static_cast<const std::uint8_t*>(buf);
    while (n > 0) {
        ssize_t r = send(fd, p, n, MSG_NOSIGNAL);
        if (r <= 0) return false;
        p += r;
        n -= static_cast<std::size_t>(r);
    }
    return true;
}

// Receive exactly n bytes
static bool recv_all(int fd, void* buf, std::size_t n) {
    auto p = static_cast<std::uint8_t*>(buf);
    while (n > 0) {
        ssize_t r = recv(fd, p, n, 0);
        if (r <= 0) return false;
        p += r;
        n -= static_cast<std::size_t>(r);
    }
    return true;
}

static void bench_e2e_socket(int num_roundtrips) {
    std::printf("--- E2E Socket (localhost TCP, TCP_NODELAY) ---\n");

    // Create TCP listener
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) { std::printf("  ERROR: socket() failed\n\n"); return; }

    int one = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0; // let OS pick a port

    if (bind(listen_fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::printf("  ERROR: bind() failed\n\n");
        close(listen_fd);
        return;
    }

    socklen_t alen = sizeof(addr);
    getsockname(listen_fd, reinterpret_cast<struct sockaddr*>(&addr), &alen);

    if (listen(listen_fd, 1) < 0) {
        std::printf("  ERROR: listen() failed\n\n");
        close(listen_fd);
        return;
    }

    // Connect from client side
    int client_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (connect(client_fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::printf("  ERROR: connect() failed\n\n");
        close(client_fd);
        close(listen_fd);
        return;
    }

    int server_fd = accept(listen_fd, nullptr, nullptr);
    close(listen_fd);

    if (server_fd < 0) {
        std::printf("  ERROR: accept() failed\n\n");
        close(client_fd);
        return;
    }

    // Set TCP_NODELAY on both
    setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    setsockopt(server_fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    // Set up BIP-324 sessions
    secp256k1::Bip324Session client_sess(true, PRIVKEY_A);
    secp256k1::Bip324Session server_sess(false, PRIVKEY_B);
    server_sess.complete_handshake(client_sess.our_ellswift_encoding().data());
    client_sess.complete_handshake(server_sess.our_ellswift_encoding().data());

    // Benchmark sizes
    static constexpr std::size_t SIZES[] = {32, 128, 512, 4096};
    static constexpr int N_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);

    std::vector<std::uint8_t> src(4096);
    for (std::size_t i = 0; i < src.size(); ++i)
        src[i] = static_cast<std::uint8_t>(i & 0xFF);

    std::printf("  %-22s  %8s  %8s  %8s  %8s  %8s\n",
                "payload", "p50", "p95", "p99", "avg", "MB/s");

    for (std::size_t si = 0; si < N_SIZES; ++si) {
        std::size_t sz = SIZES[si];
        int iters = std::max(100, num_roundtrips / (static_cast<int>(sz) / 32 + 1));

        // For nonce synchronization, we need coordinated encrypt/decrypt.
        // Since both sides share the same process, we do it sequentially:
        //   client: encrypt → send over socket
        //   server: recv → decrypt → encrypt reply → send back
        //   client: recv reply → decrypt

        // Re-create sessions to reset nonces
        secp256k1::Bip324Session c_sess(true, PRIVKEY_A);
        secp256k1::Bip324Session s_sess(false, PRIVKEY_B);
        s_sess.complete_handshake(c_sess.our_ellswift_encoding().data());
        c_sess.complete_handshake(s_sess.our_ellswift_encoding().data());

        std::vector<double> latencies;
        latencies.reserve(iters);

        // Server thread: reads request, decrypts, encrypts echo reply, sends back
        std::atomic<bool> done{false};
        std::atomic<bool> server_ok{true};

        std::thread server_thread([&]() {
            std::vector<std::uint8_t> recv_buf(sz + BIP324_OVERHEAD + 64);
            for (int i = 0; i < iters && server_ok.load(); ++i) {
                // Read full packet (we know the wire size: sz + BIP324_OVERHEAD)
                std::size_t wire_sz = sz + BIP324_OVERHEAD;
                if (!recv_all(server_fd, recv_buf.data(), wire_sz)) {
                    server_ok.store(false);
                    break;
                }

                // Decrypt
                std::vector<std::uint8_t> dec;
                if (!s_sess.decrypt(recv_buf.data(), recv_buf.data() + 3, wire_sz - 3, dec)) {
                    server_ok.store(false);
                    break;
                }

                // Echo reply: encrypt the decrypted payload back
                auto reply = s_sess.encrypt(dec.data(), dec.size());
                if (!send_all(server_fd, reply.data(), reply.size())) {
                    server_ok.store(false);
                    break;
                }
            }
            done.store(true);
        });

        // Warmup: a few roundtrips before timing
        int warmup = std::min(10, iters / 4);
        for (int w = 0; w < warmup; ++w) {
            auto pkt = c_sess.encrypt(src.data(), sz);
            send_all(client_fd, pkt.data(), pkt.size());

            std::vector<std::uint8_t> reply_buf(sz + BIP324_OVERHEAD);
            recv_all(client_fd, reply_buf.data(), reply_buf.size());
            auto rep = decrypt_packet(c_sess, reply_buf);
            bench::DoNotOptimize(rep);
        }

        // Timed iterations
        int timed_iters = iters - warmup;
        for (int i = 0; i < timed_iters && server_ok.load(); ++i) {
            std::uint64_t t0 = bench::Timer::now();

            // Client: encrypt + send
            auto pkt = c_sess.encrypt(src.data(), sz);
            send_all(client_fd, pkt.data(), pkt.size());

            // Client: recv reply + decrypt
            std::vector<std::uint8_t> reply_buf(sz + BIP324_OVERHEAD);
            recv_all(client_fd, reply_buf.data(), reply_buf.size());
            auto rep = decrypt_packet(c_sess, reply_buf);
            bench::DoNotOptimize(rep);

            std::uint64_t t1 = bench::Timer::now();
            latencies.push_back(bench::Timer::ticks_to_ns(t1 - t0));
        }

        server_thread.join();

        if (!server_ok.load() || latencies.empty()) {
            std::printf("  %-22zu  ERROR: socket I/O failed\n", sz);
            continue;
        }

        std::sort(latencies.begin(), latencies.end());
        double p50 = percentile(latencies, 0.50);
        double p95 = percentile(latencies, 0.95);
        double p99 = percentile(latencies, 0.99);
        double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0)
                     / static_cast<double>(latencies.size());
        // Throughput: one roundtrip = 2 * payload bytes
        double mbps = (2.0 * static_cast<double>(sz) / (avg / 1e9)) / (1024.0 * 1024.0);

        char label[32];
        std::snprintf(label, sizeof(label), "%zu B", sz);
        std::printf("  %-22s  %7.0f   %7.0f   %7.0f   %7.0f   %7.1f\n",
                    label, p50, p95, p99, avg, mbps);
    }

    // Also report in-memory vs socket throughput drop for 128B
    {
        secp256k1::Bip324Session m_init(true, PRIVKEY_A);
        secp256k1::Bip324Session m_resp(false, PRIVKEY_B);
        m_resp.complete_handshake(m_init.our_ellswift_encoding().data());
        m_init.complete_handshake(m_resp.our_ellswift_encoding().data());

        const double mem_ns = H.run(5000, [&]() {
            auto pkt = m_init.encrypt(src.data(), 128);
            auto dec = decrypt_packet(m_resp, pkt);
            bench::DoNotOptimize(dec);
        });

        std::printf("\n  memory-only 128 B roundtrip: %.0f ns\n", mem_ns);
        std::printf("  (compare with socket p50 above to see syscall overhead)\n");
    }

    close(client_fd);
    close(server_fd);

    std::printf("\n");
}

#endif // HAS_SOCKETS

// ============================================================================
// Main
// ============================================================================

int main() {
    bench::pin_thread_and_elevate();

    std::printf("================================================================\n");
    std::printf("  BIP-324 Transport Benchmark — Real-World Simulation\n");
    std::printf("================================================================\n");
    H.print_config();
    std::printf("\n");

    // =====================================================================
    // 1. Mixed payload distribution
    // =====================================================================
    bench_transport_mixed(10000);

    // =====================================================================
    // 2. Decoy packet overhead at different rates
    // =====================================================================
    bench_transport_decoys(10000, 0.05);   // 5% decoys
    bench_transport_decoys(10000, 0.20);   // 20% decoys

    // =====================================================================
    // 3. Per-packet latency percentiles
    // =====================================================================
    bench_latency_mode(10000);

    // =====================================================================
    // 4. E2E socket (localhost TCP)
    // =====================================================================
#if HAS_SOCKETS
    bench_e2e_socket(1000);
#else
    std::printf("--- E2E Socket: skipped (no POSIX sockets on this platform) ---\n\n");
#endif

    // =====================================================================
    std::printf("================================================================\n");
    std::printf("  Done.\n");
    std::printf("================================================================\n");

    return 0;
}
