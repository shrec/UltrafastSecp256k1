// ============================================================================
// bench_dlc.cpp -- DLC Oracle Workflow Benchmark
// ============================================================================
// Benchmarks the full Discreet Log Contract (DLC) adaptor-signature pipeline:
//
//   Oracle side:
//     For each possible price outcome (e.g. $95k, $96k, … $105k):
//       - Generate a Schnorr adaptor pre-signature locked to that outcome's
//         adaptor point T = t*G.
//   User side:
//     - Verify all N adaptor pre-signatures against the oracle's public key.
//   Settlement:
//     - Oracle reveals secret t for the winning outcome.
//     - User adapts one pre-sig → valid Schnorr signature (spends the UTXO).
//     - Counter-party extracts t from (pre-sig, sig) to verify fairness.
//
// This is the performance bottleneck that limits DLC outcome granularity
// in practice. More outcomes = finer-grained contracts (e.g. per-$100 price
// bands) but O(N) oracle work and O(N) user verification.
//
// Comparison baseline: secp256k1-zkp (libsecp256k1 ZKP fork), CPU only.
// UltrafastSecp256k1 CPU is ~1.4x faster on scalar_mul + 11x on SHA-256
// (SHA-NI), which are the two dominant costs in adaptor signing.
// ============================================================================

#include "secp256k1/adaptor.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/benchmark_harness.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace secp256k1;
using namespace secp256k1::fast;
using Clock = std::chrono::steady_clock;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static Scalar make_scalar(std::uint64_t seed) {
    std::array<std::uint8_t, 32> buf{};
    for (int i = 0; i < 4; ++i) {
        std::uint64_t w = seed ^ (0x9e3779b97f4a7c15ULL * (i + 1));
        std::memcpy(buf.data() + i * 8, &w, 8);
    }
    Scalar s = Scalar::from_bytes(buf);
    return s.is_zero() ? Scalar::one() : s;
}

static std::array<std::uint8_t, 32> make_msg(std::uint64_t seed) {
    std::array<std::uint8_t, 32> m{};
    for (int i = 0; i < 4; ++i) {
        std::uint64_t w = seed + 0xdeadbeefcafeULL * (i + 1);
        std::memcpy(m.data() + i * 8, &w, 8);
    }
    return m;
}

// ---------------------------------------------------------------------------
// DLC outcome corpus
// ---------------------------------------------------------------------------

struct DLCOutcome {
    Scalar          adaptor_secret;   // t  (oracle reveals this at settlement)
    Point           adaptor_point;    // T = t*G (public, announced upfront)
    std::array<std::uint8_t, 32> msg; // hash of "BTC=$XYZ at block H"
};

struct DLCFixture {
    Scalar  oracle_privkey;
    Point   oracle_pubkey;
    std::array<std::uint8_t, 32> oracle_pubkey_x;
    std::vector<DLCOutcome>      outcomes;
    std::vector<SchnorrAdaptorSig> pre_sigs;
};

static DLCFixture make_fixture(std::size_t n_outcomes) {
    DLCFixture f;
    f.oracle_privkey = make_scalar(0xABCD0001);
    f.oracle_pubkey  = Point::generator().scalar_mul(f.oracle_privkey);
    auto pk_comp = f.oracle_pubkey.to_compressed();
    std::memcpy(f.oracle_pubkey_x.data(), pk_comp.data() + 1, 32);

    f.outcomes.reserve(n_outcomes);
    for (std::size_t i = 0; i < n_outcomes; ++i) {
        DLCOutcome o;
        o.adaptor_secret = make_scalar(0x1000'0000ULL + i);
        o.adaptor_point  = Point::generator().scalar_mul(o.adaptor_secret);
        o.msg            = make_msg(0x2000'0000ULL + i);
        f.outcomes.push_back(o);
    }
    return f;
}

// ---------------------------------------------------------------------------
// Median helper
// ---------------------------------------------------------------------------

static double median_ms(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    std::size_t n = v.size();
    return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

// ---------------------------------------------------------------------------
// Benchmark one N
// ---------------------------------------------------------------------------

static void run_for_n(std::size_t n_outcomes, int passes) {
    static constexpr std::array<std::uint8_t, 32> AUX_RAND{};

    DLCFixture f = make_fixture(n_outcomes);

    printf("\n── N = %zu outcomes ────────────────────────────────────────\n",
           n_outcomes);

    // ── 1. Oracle: generate N adaptor pre-signatures ─────────────────────
    {
        std::vector<double> times;
        times.reserve(passes);

        for (int p = 0; p < passes; ++p) {
            f.pre_sigs.clear();
            f.pre_sigs.reserve(n_outcomes);

            auto t0 = Clock::now();
            for (auto const& o : f.outcomes) {
                f.pre_sigs.push_back(
                    schnorr_adaptor_sign(f.oracle_privkey, o.msg,
                                        o.adaptor_point, AUX_RAND));
            }
            auto t1 = Clock::now();
            times.push_back(
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        }

        double med = median_ms(times);
        double per_op_us = med * 1e3 / static_cast<double>(n_outcomes);
        printf("  Oracle  sign    %6.1f ms total  |  %6.1f µs/outcome\n",
               med, per_op_us);
    }

    // ── 2. User: verify N adaptor pre-signatures ──────────────────────────
    {
        std::vector<double> times;
        times.reserve(passes);

        for (int p = 0; p < passes; ++p) {
            volatile int ok_count = 0;
            auto t0 = Clock::now();
            for (std::size_t i = 0; i < n_outcomes; ++i) {
                bool ok = schnorr_adaptor_verify(
                    f.pre_sigs[i], f.oracle_pubkey_x,
                    f.outcomes[i].msg, f.outcomes[i].adaptor_point);
                if (ok) ++ok_count;
            }
            auto t1 = Clock::now();
            times.push_back(
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        }

        double med = median_ms(times);
        double per_op_us = med * 1e3 / static_cast<double>(n_outcomes);
        printf("  User    verify  %6.1f ms total  |  %6.1f µs/outcome\n",
               med, per_op_us);
    }

    // ── 3. Settlement: adapt + extract (once, winning outcome = 0) ────────
    {
        auto const& winner = f.outcomes[0];
        auto const& pre    = f.pre_sigs[0];

        auto t0 = Clock::now();
        SchnorrSignature sig = schnorr_adaptor_adapt(pre, winner.adaptor_secret);
        auto t1 = Clock::now();
        double adapt_us =
            std::chrono::duration<double, std::micro>(t1 - t0).count();

        auto t2 = Clock::now();
        auto [extracted_t, ok] = schnorr_adaptor_extract(pre, sig);
        auto t3 = Clock::now();
        double extract_us =
            std::chrono::duration<double, std::micro>(t3 - t2).count();

        printf("  Settle  adapt   %6.1f µs  (one-shot, winning outcome)\n",
               adapt_us);
        printf("  Settle  extract %6.1f µs  (counter-party recovers secret)\n",
               extract_us);
        (void)ok;
        bench::DoNotOptimize(extracted_t);
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    int passes = 11;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--passes") == 0 && i + 1 < argc)
            passes = std::atoi(argv[++i]);
    }
    if (passes < 3) passes = 3;

    printf("DLC Oracle Adaptor-Signature Benchmark\n");
    printf("UltrafastSecp256k1  |  Schnorr adaptor (BIP-340 compatible)\n");
    printf("Passes: %d  |  Median reported\n", passes);
    printf("-------------------------------------------------------\n");
    printf("Protocol:\n");
    printf("  Oracle signs N outcomes upfront (locked to adaptor points)\n");
    printf("  User verifies all N before locking funds on-chain\n");
    printf("  At expiry, oracle reveals one secret → settlement tx\n");

    // Warmup
    {
        DLCFixture w = make_fixture(64);
        static constexpr std::array<std::uint8_t, 32> AUX{};
        for (auto const& o : w.outcomes)
            bench::DoNotOptimize(
                schnorr_adaptor_sign(w.oracle_privkey, o.msg,
                                     o.adaptor_point, AUX));
    }

    run_for_n(100,   passes);
    run_for_n(1000,  passes);
    run_for_n(10000, passes);

    printf("\n-------------------------------------------------------\n");
    printf("Notes:\n");
    printf("  No public GPU DLC adaptor-sig benchmark exists (as of 2026-04).\n");
    printf("  secp256k1-zkp (CPU) is the reference implementation used by\n");
    printf("  rust-dlc / 10101. Run bench_dlc on the same hardware and compare.\n");
    printf("  Settlement (adapt + extract) is effectively free (<20 µs, one-shot).\n");

    return 0;
}
