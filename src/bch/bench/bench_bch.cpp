// ============================================================================
// bench_bch.cpp — BCH RPA CPU benchmark
// ============================================================================
// Measures:
//   1. EC Grinding throughput (ECDSA sign + double-SHA256 prefix check)
//   2. RPA Scan throughput (ECDH + SHA256 midstate + pubkey derivation)
//   3. Multi-threaded versions of both
//
// Build: cmake -DSECP256K1_BUILD_BCH=ON && make bench_bch
// Run:   taskset -c 0-15 nice -20 ./bench_bch
// ============================================================================
#include "secp256k1/bch/rpa.hpp"
#include "secp256k1/bch/bch_scan.hpp"
#include "secp256k1/bch/cashaddr.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/ct/sign.hpp"
#include "secp256k1/ct/point.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>
#include <thread>
#include <atomic>
#include <random>
#include <algorithm>

using namespace secp256k1::bch;
using namespace std::chrono;

// ── Helpers ───────────────────────────────────────────────────────────────────

static secp256k1::fast::Scalar random_scalar() {
    static std::mt19937_64 rng(0xBC352CAFEu);
    secp256k1::fast::Scalar s{};
    do {
        std::array<uint8_t, 32> b{};
        for (int i = 0; i < 4; ++i) {
            uint64_t v = rng();
            std::memcpy(b.data() + i*8, &v, 8);
        }
        b[0] &= 0x7f; // keep < p (rough)
        secp256k1::fast::Scalar::parse_bytes_strict(b.data(), s);
    } while (s.is_zero());
    return s;
}

static double ms_since(steady_clock::time_point t0) {
    return duration_cast<microseconds>(steady_clock::now() - t0).count() / 1000.0;
}

// ── Benchmark 1: EC Grinding (single-thread) ─────────────────────────────────
// How many ECDSA signs + double-SHA256 prefix checks per second?

static void bench_grinding_single(int prefix_bits, uint32_t n_attempts) {
    auto sk = random_scalar();
    std::array<uint8_t, 32> msg{};
    msg[31] = 0x42;

    // Fake prefix target (first byte of scan_pubkey)
    auto pk = secp256k1::ct::generator_mul(sk);
    auto pk_comp = pk.to_compressed();
    const uint8_t* prefix_data = pk_comp.data();

    uint32_t found = 0;
    auto t0 = steady_clock::now();

    for (uint32_t nonce = 0; nonce < n_attempts; ++nonce) {
        std::array<uint8_t, 32> aux{};
        std::memcpy(aux.data(), &nonce, 4);
        auto sig = secp256k1::ct::ecdsa_sign_hedged(msg, sk, aux);
        auto compact = sig.to_compact();
        if (rpa_prefix_matches(compact.data(), prefix_bits, prefix_data))
            ++found;
    }

    double ms = ms_since(t0);
    double rate = n_attempts / (ms / 1000.0);
    printf("  Grinding (%2d-bit prefix): %.0f k/s  [found=%u in %.0f ms]\n",
           prefix_bits, rate / 1000.0, found, ms);
}

// ── Benchmark 2: EC Grinding (multi-thread) ───────────────────────────────────

static void bench_grinding_mt(int prefix_bits, uint32_t n_per_thread, int n_threads) {
    auto sk = random_scalar();
    std::array<uint8_t, 32> msg{};
    msg[31] = 0x42;
    auto pk = secp256k1::ct::generator_mul(sk);
    auto pk_comp = pk.to_compressed();
    const uint8_t* prefix_data = pk_comp.data();

    std::atomic<uint64_t> total_signs{0};
    std::atomic<uint32_t> total_found{0};

    auto t0 = steady_clock::now();
    std::vector<std::thread> threads;
    threads.reserve(n_threads);

    for (int t = 0; t < n_threads; ++t) {
        threads.emplace_back([&, t]() {
            uint32_t base = static_cast<uint32_t>(t) * n_per_thread;
            for (uint32_t i = 0; i < n_per_thread; ++i) {
                uint32_t nonce = base + i;
                std::array<uint8_t, 32> aux{};
                std::memcpy(aux.data(), &nonce, 4);
                auto sig = secp256k1::ct::ecdsa_sign_hedged(msg, sk, aux);
                auto compact = sig.to_compact();
                if (rpa_prefix_matches(compact.data(), prefix_bits, prefix_data))
                    ++total_found;
            }
            total_signs += n_per_thread;
        });
    }
    for (auto& th : threads) th.join();

    double ms = ms_since(t0);
    uint64_t total = total_signs.load();
    double rate = total / (ms / 1000.0);
    printf("  Grinding (%2d-bit, %2d thr): %.0f k/s  [%.2f M total, %.0f ms]\n",
           prefix_bits, n_threads, rate / 1000.0, total / 1e6, ms);
}

// ── Benchmark 3: RPA Scan throughput (single-thread) ─────────────────────────

static void bench_scan_single(uint32_t n_txs, uint8_t prefix_bits) {
    // Setup scanner
    auto scan_sk = random_scalar();
    auto spend_sk = random_scalar();

    RpaPaycode pc{};
    pc.version = 1;
    pc.prefix_bits = prefix_bits;
    auto scan_pk = secp256k1::ct::generator_mul(scan_sk);
    auto sp = scan_pk.to_compressed();
    std::memcpy(pc.scan_pubkey.data(), sp.data(), 33);
    auto spend_pk = secp256k1::ct::generator_mul(spend_sk);
    auto spp = spend_pk.to_compressed();
    std::memcpy(pc.spend_pubkey.data(), spp.data(), 33);

    RpaScanner scanner(pc, scan_sk);

    // Build random tx pool
    std::mt19937_64 rng(0xBC1u);
    std::vector<ScanTx> txs(n_txs);
    for (auto& tx : txs) {
        // random txid
        for (auto& b : tx.txid) b = static_cast<uint8_t>(rng());
        tx.vout = 0;
        // random input pubkey (valid compressed)
        auto sender_sk = random_scalar();
        auto sender_pk = secp256k1::ct::generator_mul(sender_sk);
        tx.input_pubkey = sender_pk.to_compressed();
        // random output pubkeys (no match)
        tx.outputs.resize(2);
        for (auto& out : tx.outputs) {
            auto k = random_scalar();
            out = secp256k1::ct::generator_mul(k).to_compressed();
        }
    }

    auto t0 = steady_clock::now();
    auto matches = scanner.scan_batch_cpu(txs, 0);
    double ms = ms_since(t0);
    double rate = n_txs / (ms / 1000.0);
    printf("  Scan (1 thr, %2d-bit prefix): %.0f tx/s  [%.1f ms, %zu matches]\n",
           prefix_bits, rate, ms, matches.size());
}

// ── Benchmark 4: RPA Scan multi-threaded ─────────────────────────────────────

static void bench_scan_mt(uint32_t n_txs, uint8_t prefix_bits, int n_threads) {
    auto scan_sk = random_scalar();
    auto spend_sk = random_scalar();

    RpaPaycode pc{};
    pc.version = 1;
    pc.prefix_bits = prefix_bits;
    auto sc = secp256k1::ct::generator_mul(scan_sk).to_compressed();
    auto sp = secp256k1::ct::generator_mul(spend_sk).to_compressed();
    std::memcpy(pc.scan_pubkey.data(), sc.data(), 33);
    std::memcpy(pc.spend_pubkey.data(), sp.data(), 33);

    RpaScanner scanner(pc, scan_sk);

    // Build tx pool
    std::vector<ScanTx> txs(n_txs);
    for (size_t i = 0; i < txs.size(); ++i) {
        txs[i].txid[0] = static_cast<uint8_t>(i);
        txs[i].txid[1] = static_cast<uint8_t>(i >> 8);
        txs[i].vout = 0;
        auto k = random_scalar();
        txs[i].input_pubkey = secp256k1::ct::generator_mul(k).to_compressed();
        txs[i].outputs.resize(1);
        auto k2 = random_scalar();
        txs[i].outputs[0] = secp256k1::ct::generator_mul(k2).to_compressed();
    }

    // Partition across threads
    std::atomic<size_t> total_matches{0};
    uint32_t chunk = n_txs / n_threads;

    auto t0 = steady_clock::now();
    std::vector<std::thread> threads;
    for (int t = 0; t < n_threads; ++t) {
        threads.emplace_back([&, t]() {
            size_t start = static_cast<size_t>(t) * chunk;
            size_t end = (t == n_threads - 1) ? n_txs : start + chunk;
            std::vector<ScanTx> sub(txs.begin() + start, txs.begin() + end);
            auto m = scanner.scan_batch_cpu(sub, 0);
            total_matches += m.size();
        });
    }
    for (auto& th : threads) th.join();

    double ms = ms_since(t0);
    double rate = n_txs / (ms / 1000.0);
    printf("  Scan (%2d thr, %2d-bit prefix): %.0f tx/s  [%.1f ms, %zu matches]\n",
           n_threads, prefix_bits, rate, ms, total_matches.load());
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    int n_threads = std::thread::hardware_concurrency();
    if (argc > 1) n_threads = std::atoi(argv[1]);
    if (n_threads < 1) n_threads = 1;

    printf("====================================================================\n");
    printf("  BCH RPA CPU Benchmark\n");
    printf("  Threads available: %d  |  Using: %d\n",
           (int)std::thread::hardware_concurrency(), n_threads);
    printf("====================================================================\n\n");

    printf("[1] EC Grinding throughput (single thread)\n");
    bench_grinding_single(0,  200000);
    bench_grinding_single(8,  200000);
    bench_grinding_single(16, 200000);

    printf("\n[2] EC Grinding throughput (%d threads)\n", n_threads);
    bench_grinding_mt(0,  200000 / n_threads, n_threads);
    bench_grinding_mt(8,  200000 / n_threads, n_threads);
    bench_grinding_mt(16, 200000 / n_threads, n_threads);

    printf("\n[3] RPA Scan throughput (single thread)\n");
    bench_scan_single(10000, 0);
    bench_scan_single(10000, 8);
    bench_scan_single(10000, 16);

    printf("\n[4] RPA Scan throughput (%d threads)\n", n_threads);
    bench_scan_mt(100000, 0,  n_threads);
    bench_scan_mt(100000, 8,  n_threads);
    bench_scan_mt(100000, 16, n_threads);

    printf("\n====================================================================\n");
    printf("  BCH mainnet: ~1M tx/day = ~11.5 tx/s\n");
    printf("  16-bit prefix: 1/65536 txs require full scan per tx\n");
    printf("====================================================================\n");
    return 0;
}
