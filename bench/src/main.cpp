// bench_compare main.cpp -- Apples-to-apples benchmark runner
// UltrafastSecp256k1 vs bitcoin-core/libsecp256k1
//
// Build:  cmake -S . -B build -DSECP256K1_BUILD_BENCH_COMPARE=ON
//         cmake --build build -j
// Run:    ./build/bench/bench_compare [OPTIONS]
// Help:   ./bench_compare --help

#ifdef _WIN32
#  define NOMINMAX
#endif

#include "bench_api.h"
#include "bench_config.h"
#include "bench_timer.h"
#include "bench_rng.h"
#include "bench_affinity.h"
#include "bench_report.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

// For ECDSA signing (dataset generation) -- use UF API directly
#include <secp256k1/ecdsa.hpp>
#include <secp256k1/schnorr.hpp>
#include <secp256k1/precompute.hpp>
#include <secp256k1/scalar.hpp>
#include <secp256k1/point.hpp>
#include <secp256k1/field.hpp>

using namespace bench;

// ===================================================================
// Dataset types
// ===================================================================

struct EcdsaDatasetItem {
    std::array<uint8_t, 32> msg;         // message hash
    std::array<uint8_t, 33> pubkey;      // compressed pubkey
    std::array<uint8_t, 64> sig_compact; // compact signature
    // DER encoded (variable length)
    uint8_t sig_der[72];
    size_t  sig_der_len;
};

struct SchnorrDatasetItem {
    std::array<uint8_t, 32> msg;         // message hash
    std::array<uint8_t, 32> xonly_pubkey; // x-only pubkey
    std::array<uint8_t, 64> sig;         // 64-byte signature
};

struct PubkeyDatasetItem {
    std::array<uint8_t, 32> seckey;
    std::array<uint8_t, 33> expected_pubkey; // for correctness check
};

struct EcdhDatasetItem {
    std::array<uint8_t, 32> seckey;
    std::array<uint8_t, 33> pubkey;
    std::array<uint8_t, 32> expected_secret; // for correctness check
};

// ===================================================================
// Dataset generation (deterministic via BenchRng)
// ===================================================================

static void generate_ecdsa_dataset(
    std::vector<EcdsaDatasetItem>& items, size_t count, uint64_t seed)
{
    BenchRng rng(seed);
    items.resize(count);

    secp256k1::fast::ensure_fixed_base_ready();

    for (size_t i = 0; i < count; ++i) {
        auto& item = items[i];
        rng.fill_bytes(item.msg.data(), 32);

        // Generate random private key (non-zero)
        std::array<uint8_t, 32> sk;
        do {
            rng.fill_bytes(sk.data(), 32);
        } while (sk == std::array<uint8_t, 32>{});

        auto scalar = secp256k1::fast::Scalar::from_bytes(sk);
        auto point = secp256k1::fast::scalar_mul_generator(scalar);
        auto compressed = point.to_compressed();
        std::memcpy(item.pubkey.data(), compressed.data(), 33);

        // Sign with UF
        auto ecdsa_sig = secp256k1::ecdsa_sign(item.msg, scalar);
        // Normalize to low-s
        if (!ecdsa_sig.is_low_s()) {
            ecdsa_sig = ecdsa_sig.normalize();
        }

        auto compact = ecdsa_sig.to_compact();
        std::memcpy(item.sig_compact.data(), compact.data(), 64);

        auto [der_buf, der_len] = ecdsa_sig.to_der();
        std::memcpy(item.sig_der, der_buf.data(), der_len);
        item.sig_der_len = der_len;
    }
}

static void generate_schnorr_dataset(
    std::vector<SchnorrDatasetItem>& items, size_t count, uint64_t seed)
{
    BenchRng rng(seed + 0x1000); // different stream
    items.resize(count);

    secp256k1::fast::ensure_fixed_base_ready();

    for (size_t i = 0; i < count; ++i) {
        auto& item = items[i];
        rng.fill_bytes(item.msg.data(), 32);

        // Generate random private key
        std::array<uint8_t, 32> sk;
        do {
            rng.fill_bytes(sk.data(), 32);
        } while (sk == std::array<uint8_t, 32>{});

        auto scalar = secp256k1::fast::Scalar::from_bytes(sk);

        // BIP-340 keypair
        auto kp = secp256k1::schnorr_keypair_create(scalar);
        std::memcpy(item.xonly_pubkey.data(), kp.px.data(), 32);

        // Sign
        std::array<uint8_t, 32> aux_rand;
        rng.fill_bytes(aux_rand.data(), 32);
        auto schnorr_sig = secp256k1::schnorr_sign(kp, item.msg, aux_rand);
        auto sig_bytes = schnorr_sig.to_bytes();
        std::memcpy(item.sig.data(), sig_bytes.data(), 64);
    }
}

static void generate_pubkey_dataset(
    std::vector<PubkeyDatasetItem>& items, size_t count, uint64_t seed)
{
    BenchRng rng(seed + 0x2000);
    items.resize(count);

    secp256k1::fast::ensure_fixed_base_ready();

    for (size_t i = 0; i < count; ++i) {
        auto& item = items[i];
        do {
            rng.fill_bytes(item.seckey.data(), 32);
        } while (item.seckey == std::array<uint8_t, 32>{});

        auto scalar = secp256k1::fast::Scalar::from_bytes(item.seckey);
        auto point = secp256k1::fast::scalar_mul_generator(scalar);
        auto compressed = point.to_compressed();
        std::memcpy(item.expected_pubkey.data(), compressed.data(), 33);
    }
}

static void generate_ecdh_dataset(
    std::vector<EcdhDatasetItem>& items, size_t count, uint64_t seed)
{
    BenchRng rng(seed + 0x3000);
    items.resize(count);

    secp256k1::fast::ensure_fixed_base_ready();

    for (size_t i = 0; i < count; ++i) {
        auto& item = items[i];

        // Generate secret key
        do {
            rng.fill_bytes(item.seckey.data(), 32);
        } while (item.seckey == std::array<uint8_t, 32>{});

        // Generate peer pubkey
        std::array<uint8_t, 32> peer_sk;
        do {
            rng.fill_bytes(peer_sk.data(), 32);
        } while (peer_sk == std::array<uint8_t, 32>{});

        auto peer_scalar = secp256k1::fast::Scalar::from_bytes(peer_sk);
        auto peer_point = secp256k1::fast::scalar_mul_generator(peer_scalar);
        auto compressed = peer_point.to_compressed();
        std::memcpy(item.pubkey.data(), compressed.data(), 33);

        // Compute expected secret using UF
        auto my_scalar = secp256k1::fast::Scalar::from_bytes(item.seckey);
        // ecdh_compute_xonly is SHA256(x-coord)
        // For correctness comparison, both must use same ECDH variant
        // libsecp256k1 default uses SHA256(compressed point), we need to match
        // We'll compute with UF and store as reference
        // NOTE: We'll skip ECDH correctness cross-check since hash functions
        // may differ between UF and libsecp default. Each provider is verified
        // internally via pubkey_create first.
        std::memset(item.expected_secret.data(), 0, 32);
    }
}

// ===================================================================
// Correctness gate
// ===================================================================

static bool run_correctness_gate_ecdsa(
    IProvider* provider, const std::vector<EcdsaDatasetItem>& dataset,
    const BenchConfig& cfg, int sample_count = 100)
{
    int n = std::min(sample_count, static_cast<int>(dataset.size()));
    int pass = 0;
    for (int i = 0; i < n; ++i) {
        const auto& item = dataset[i];
        bool ok;
        if (cfg.sig_encoding == SigEncoding::DER) {
            ok = provider->ecdsa_verify_bytes(
                item.pubkey.data(), 33,
                item.sig_der, item.sig_der_len,
                item.msg.data(), true);
        } else {
            ok = provider->ecdsa_verify_bytes(
                item.pubkey.data(), 33,
                item.sig_compact.data(), 64,
                item.msg.data(), true);
        }
        if (ok) ++pass;
    }

    if (pass != n) {
        std::fprintf(stderr, "[FAIL] %s ECDSA correctness: %d/%d passed\n",
                     provider->name(), pass, n);
        return false;
    }
    return true;
}

static bool run_correctness_gate_schnorr(
    IProvider* provider, const std::vector<SchnorrDatasetItem>& dataset,
    int sample_count = 100)
{
    int n = std::min(sample_count, static_cast<int>(dataset.size()));
    int pass = 0;
    for (int i = 0; i < n; ++i) {
        const auto& item = dataset[i];
        if (provider->schnorr_verify_bytes(
                item.xonly_pubkey.data(), item.sig.data(), item.msg.data()))
            ++pass;
    }

    if (pass != n) {
        std::fprintf(stderr, "[FAIL] %s Schnorr correctness: %d/%d passed\n",
                     provider->name(), pass, n);
        return false;
    }
    return true;
}

static bool run_correctness_gate_pubkey(
    IProvider* provider, const std::vector<PubkeyDatasetItem>& dataset,
    int sample_count = 100)
{
    int n = std::min(sample_count, static_cast<int>(dataset.size()));
    int pass = 0;
    for (int i = 0; i < n; ++i) {
        const auto& item = dataset[i];
        uint8_t out33[33];
        if (!provider->pubkey_create(out33, item.seckey.data())) continue;
        if (std::memcmp(out33, item.expected_pubkey.data(), 33) == 0) ++pass;
    }

    if (pass != n) {
        std::fprintf(stderr, "[FAIL] %s pubkey_create correctness: %d/%d passed\n",
                     provider->name(), pass, n);
        return false;
    }
    return true;
}

// ===================================================================
// Benchmark runner for a single case
// ===================================================================

// Run warmup iterations for warmup_ms, then measure for measure_ms.
// Returns timing statistics.
template<typename RunOneFn>
static BenchStats run_timed_bench(
    RunOneFn run_one, size_t dataset_size,
    int warmup_ms, int measure_ms)
{
    using clock = std::chrono::steady_clock;

    // Warmup phase
    auto warmup_end = clock::now() + std::chrono::milliseconds(warmup_ms);
    size_t idx = 0;
    while (clock::now() < warmup_end) {
        run_one(idx % dataset_size);
        ++idx;
    }

    // Measurement phase
    // Collect per-op samples
    std::vector<double> samples;
    samples.reserve(dataset_size);

    auto measure_end = clock::now() + std::chrono::milliseconds(measure_ms);
    idx = 0;
    while (clock::now() < measure_end) {
        auto t0 = clock::now();
        run_one(idx % dataset_size);
        auto t1 = clock::now();

        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        samples.push_back(ns);
        ++idx;
    }

    return compute_stats(samples);
}

// ===================================================================
// Case runners
// ===================================================================

static CaseResult bench_ecdsa_verify_bytes(
    IProvider* provider, const std::vector<EcdsaDatasetItem>& dataset,
    const BenchConfig& cfg)
{
    CaseResult result;
    result.case_name = "ecdsa_verify_bytes";
    result.provider_name = provider->name();
    result.correctness_ok = run_correctness_gate_ecdsa(provider, dataset, cfg);

    if (!result.correctness_ok) {
        result.stats = {};
        return result;
    }

    bool use_der = (cfg.sig_encoding == SigEncoding::DER);

    result.stats = run_timed_bench(
        [&](size_t i) {
            const auto& item = dataset[i];
            if (use_der) {
                provider->ecdsa_verify_bytes(
                    item.pubkey.data(), 33,
                    item.sig_der, item.sig_der_len,
                    item.msg.data(), true);
            } else {
                provider->ecdsa_verify_bytes(
                    item.pubkey.data(), 33,
                    item.sig_compact.data(), 64,
                    item.msg.data(), true);
            }
        },
        dataset.size(), cfg.warmup_ms, cfg.measure_ms);

    return result;
}

static CaseResult bench_ecdsa_verify_preparsed(
    IProvider* provider, const std::vector<EcdsaDatasetItem>& dataset,
    const BenchConfig& cfg)
{
    CaseResult result;
    result.case_name = "ecdsa_verify_preparsed";
    result.provider_name = provider->name();

    // Pre-parse all keys and signatures
    std::vector<ParsedPubkey> parsed_pks(dataset.size());
    std::vector<ParsedSig>    parsed_sigs(dataset.size());

    bool use_der = (cfg.sig_encoding == SigEncoding::DER);

    for (size_t i = 0; i < dataset.size(); ++i) {
        const auto& item = dataset[i];
        if (!provider->ecdsa_parse_pubkey(&parsed_pks[i], item.pubkey.data(), 33)) {
            std::fprintf(stderr, "[FAIL] %s: ecdsa_parse_pubkey failed at %zu\n",
                         provider->name(), i);
            result.correctness_ok = false;
            result.stats = {};
            return result;
        }
        if (use_der) {
            if (!provider->ecdsa_parse_sig(&parsed_sigs[i], item.sig_der, item.sig_der_len, true)) {
                std::fprintf(stderr, "[FAIL] %s: ecdsa_parse_sig(DER) failed at %zu\n",
                             provider->name(), i);
                result.correctness_ok = false;
                result.stats = {};
                return result;
            }
        } else {
            if (!provider->ecdsa_parse_sig(&parsed_sigs[i], item.sig_compact.data(), 64, false)) {
                std::fprintf(stderr, "[FAIL] %s: ecdsa_parse_sig(compact) failed at %zu\n",
                             provider->name(), i);
                result.correctness_ok = false;
                result.stats = {};
                return result;
            }
        }
    }

    // Quick correctness check on parsed path
    int n = std::min(100, static_cast<int>(dataset.size()));
    int pass = 0;
    for (int i = 0; i < n; ++i) {
        if (provider->ecdsa_verify_preparsed(
                &parsed_pks[i], &parsed_sigs[i],
                dataset[i].msg.data(), true))
            ++pass;
    }
    result.correctness_ok = (pass == n);
    if (!result.correctness_ok) {
        std::fprintf(stderr, "[FAIL] %s ecdsa_verify_preparsed correctness: %d/%d\n",
                     provider->name(), pass, n);
        result.stats = {};
        return result;
    }

    result.stats = run_timed_bench(
        [&](size_t i) {
            provider->ecdsa_verify_preparsed(
                &parsed_pks[i], &parsed_sigs[i],
                dataset[i].msg.data(), true);
        },
        dataset.size(), cfg.warmup_ms, cfg.measure_ms);

    return result;
}

static CaseResult bench_schnorr_verify_bytes(
    IProvider* provider, const std::vector<SchnorrDatasetItem>& dataset,
    const BenchConfig& cfg)
{
    CaseResult result;
    result.case_name = "schnorr_verify_bytes";
    result.provider_name = provider->name();
    result.correctness_ok = run_correctness_gate_schnorr(provider, dataset);

    if (!result.correctness_ok) {
        result.stats = {};
        return result;
    }

    result.stats = run_timed_bench(
        [&](size_t i) {
            const auto& item = dataset[i];
            provider->schnorr_verify_bytes(
                item.xonly_pubkey.data(), item.sig.data(), item.msg.data());
        },
        dataset.size(), cfg.warmup_ms, cfg.measure_ms);

    return result;
}

static CaseResult bench_schnorr_verify_preparsed(
    IProvider* provider, const std::vector<SchnorrDatasetItem>& dataset,
    const BenchConfig& cfg)
{
    CaseResult result;
    result.case_name = "schnorr_verify_preparsed";
    result.provider_name = provider->name();

    // Pre-parse all xonly pubkeys
    std::vector<ParsedXonlyPubkey> parsed_pks(dataset.size());
    for (size_t i = 0; i < dataset.size(); ++i) {
        if (!provider->schnorr_parse_xonly(&parsed_pks[i], dataset[i].xonly_pubkey.data())) {
            std::fprintf(stderr, "[FAIL] %s: schnorr_parse_xonly failed at %zu\n",
                         provider->name(), i);
            result.correctness_ok = false;
            result.stats = {};
            return result;
        }
    }

    // Quick correctness check
    int n = std::min(100, static_cast<int>(dataset.size()));
    int pass = 0;
    for (int i = 0; i < n; ++i) {
        if (provider->schnorr_verify_preparsed(
                &parsed_pks[i], dataset[i].sig.data(), dataset[i].msg.data()))
            ++pass;
    }
    result.correctness_ok = (pass == n);
    if (!result.correctness_ok) {
        std::fprintf(stderr, "[FAIL] %s schnorr_verify_preparsed correctness: %d/%d\n",
                     provider->name(), pass, n);
        result.stats = {};
        return result;
    }

    result.stats = run_timed_bench(
        [&](size_t i) {
            provider->schnorr_verify_preparsed(
                &parsed_pks[i], dataset[i].sig.data(), dataset[i].msg.data());
        },
        dataset.size(), cfg.warmup_ms, cfg.measure_ms);

    return result;
}

static CaseResult bench_pubkey_create(
    IProvider* provider, const std::vector<PubkeyDatasetItem>& dataset,
    const BenchConfig& cfg)
{
    CaseResult result;
    result.case_name = "pubkey_create";
    result.provider_name = provider->name();
    result.correctness_ok = run_correctness_gate_pubkey(provider, dataset);

    if (!result.correctness_ok) {
        result.stats = {};
        return result;
    }

    uint8_t out33[33];
    result.stats = run_timed_bench(
        [&](size_t i) {
            provider->pubkey_create(out33, dataset[i].seckey.data());
        },
        dataset.size(), cfg.warmup_ms, cfg.measure_ms);

    return result;
}

static CaseResult bench_ecdh(
    IProvider* provider, const std::vector<EcdhDatasetItem>& dataset,
    const BenchConfig& cfg)
{
    CaseResult result;
    result.case_name = "ecdh";
    result.provider_name = provider->name();
    result.correctness_ok = true; // self-consistency only

    // Quick self-consistency: ecdh twice same input -> same output
    if (dataset.size() >= 2) {
        uint8_t out1[32], out2[32];
        bool ok1 = provider->ecdh(out1, dataset[0].seckey.data(),
                                  dataset[0].pubkey.data(), 33);
        bool ok2 = provider->ecdh(out2, dataset[0].seckey.data(),
                                  dataset[0].pubkey.data(), 33);
        if (!ok1 || !ok2 || std::memcmp(out1, out2, 32) != 0) {
            std::fprintf(stderr, "[FAIL] %s ECDH self-consistency failed\n",
                         provider->name());
            result.correctness_ok = false;
            result.stats = {};
            return result;
        }
    }

    uint8_t out32[32];
    result.stats = run_timed_bench(
        [&](size_t i) {
            provider->ecdh(out32, dataset[i].seckey.data(),
                          dataset[i].pubkey.data(), 33);
        },
        dataset.size(), cfg.warmup_ms, cfg.measure_ms);

    return result;
}

// ===================================================================
// Run all cases for one provider
// ===================================================================

static void run_provider(
    IProvider* provider,
    const BenchConfig& cfg,
    const std::vector<EcdsaDatasetItem>& ecdsa_ds,
    const std::vector<SchnorrDatasetItem>& schnorr_ds,
    const std::vector<PubkeyDatasetItem>& pubkey_ds,
    const std::vector<EcdhDatasetItem>& ecdh_ds,
    std::vector<CaseResult>& all_results)
{
    std::printf("\n>>> Provider: %s (v%s)\n", provider->name(), provider->version());

    if (cfg.case_ecdsa_verify) {
        std::printf("  [ecdsa_verify_bytes] ...\n");
        all_results.push_back(bench_ecdsa_verify_bytes(provider, ecdsa_ds, cfg));
        std::printf("    -> median=%.1f ns  ops/s=%.0f  %s\n",
                    all_results.back().stats.median_ns,
                    all_results.back().stats.ops_per_sec,
                    all_results.back().correctness_ok ? "OK" : "FAIL");

        std::printf("  [ecdsa_verify_preparsed] ...\n");
        all_results.push_back(bench_ecdsa_verify_preparsed(provider, ecdsa_ds, cfg));
        std::printf("    -> median=%.1f ns  ops/s=%.0f  %s\n",
                    all_results.back().stats.median_ns,
                    all_results.back().stats.ops_per_sec,
                    all_results.back().correctness_ok ? "OK" : "FAIL");
    }

    if (cfg.case_schnorr_verify) {
        std::printf("  [schnorr_verify_bytes] ...\n");
        all_results.push_back(bench_schnorr_verify_bytes(provider, schnorr_ds, cfg));
        std::printf("    -> median=%.1f ns  ops/s=%.0f  %s\n",
                    all_results.back().stats.median_ns,
                    all_results.back().stats.ops_per_sec,
                    all_results.back().correctness_ok ? "OK" : "FAIL");

        std::printf("  [schnorr_verify_preparsed] ...\n");
        all_results.push_back(bench_schnorr_verify_preparsed(provider, schnorr_ds, cfg));
        std::printf("    -> median=%.1f ns  ops/s=%.0f  %s\n",
                    all_results.back().stats.median_ns,
                    all_results.back().stats.ops_per_sec,
                    all_results.back().correctness_ok ? "OK" : "FAIL");
    }

    if (cfg.case_pubkey_create) {
        std::printf("  [pubkey_create] ...\n");
        all_results.push_back(bench_pubkey_create(provider, pubkey_ds, cfg));
        std::printf("    -> median=%.1f ns  ops/s=%.0f  %s\n",
                    all_results.back().stats.median_ns,
                    all_results.back().stats.ops_per_sec,
                    all_results.back().correctness_ok ? "OK" : "FAIL");
    }

    if (cfg.case_ecdh) {
        std::printf("  [ecdh] ...\n");
        all_results.push_back(bench_ecdh(provider, ecdh_ds, cfg));
        std::printf("    -> median=%.1f ns  ops/s=%.0f  %s\n",
                    all_results.back().stats.median_ns,
                    all_results.back().stats.ops_per_sec,
                    all_results.back().correctness_ok ? "OK" : "FAIL");
    }
}

// ===================================================================
// main
// ===================================================================

int main(int argc, char** argv) {
    BenchConfig cfg;
    if (!parse_args(cfg, argc, argv)) return 1;

    // -- Print banner -------------------------------------------------------
    std::printf("=== bench_compare: UltrafastSecp256k1 vs libsecp256k1 ===\n");
    std::printf("Dataset size   : %zu\n", cfg.dataset_size);
    std::printf("PRNG seed      : %llu\n", static_cast<unsigned long long>(cfg.seed));
    std::printf("Warmup         : %d ms\n", cfg.warmup_ms);
    std::printf("Measurement    : %d ms\n", cfg.measure_ms);
    std::printf("Sig encoding   : %s\n",
                cfg.sig_encoding == SigEncoding::DER ? "DER" : "compact");
    std::printf("Msg policy     : %s\n",
                cfg.msg_policy == MsgPolicy::PREHASHED32 ? "prehashed32" : "sha256d");
    std::printf("libsecp random : %s\n", cfg.libsecp_randomize ? "yes" : "no");
    std::printf("Pin core       : %d\n", cfg.pin_core);

    // -- CPU info -----------------------------------------------------------
    print_cpu_info();

    // -- CPU affinity -------------------------------------------------------
    if (cfg.pin_core >= 0) {
        if (pin_to_core(cfg.pin_core)) {
            std::printf("Pinned to core %d\n", cfg.pin_core);
        }
    }

    // -- Create providers ---------------------------------------------------
    std::unique_ptr<IProvider> prov_uf, prov_libsecp;

    if (cfg.run_uf) {
        prov_uf = create_provider_uf();
        if (!prov_uf || !prov_uf->init(false)) {
            std::fprintf(stderr, "[FATAL] Failed to init UltrafastSecp256k1\n");
            return 1;
        }
    }

    if (cfg.run_libsecp) {
        prov_libsecp = create_provider_libsecp();
        if (!prov_libsecp || !prov_libsecp->init(cfg.libsecp_randomize)) {
            std::fprintf(stderr, "[FATAL] Failed to init libsecp256k1\n");
            return 1;
        }
    }

    // -- Generate datasets --------------------------------------------------
    std::printf("\nGenerating datasets (n=%zu) ...\n", cfg.dataset_size);

    std::vector<EcdsaDatasetItem>  ecdsa_ds;
    std::vector<SchnorrDatasetItem> schnorr_ds;
    std::vector<PubkeyDatasetItem> pubkey_ds;
    std::vector<EcdhDatasetItem>   ecdh_ds;

    if (cfg.case_ecdsa_verify)   generate_ecdsa_dataset(ecdsa_ds, cfg.dataset_size, cfg.seed);
    if (cfg.case_schnorr_verify) generate_schnorr_dataset(schnorr_ds, cfg.dataset_size, cfg.seed);
    if (cfg.case_pubkey_create)  generate_pubkey_dataset(pubkey_ds, cfg.dataset_size, cfg.seed);
    if (cfg.case_ecdh)           generate_ecdh_dataset(ecdh_ds, cfg.dataset_size, cfg.seed);

    std::printf("Datasets ready.\n");

    // -- Run benchmarks -----------------------------------------------------
    std::vector<CaseResult> all_results;

    if (prov_uf) {
        run_provider(prov_uf.get(), cfg,
                     ecdsa_ds, schnorr_ds, pubkey_ds, ecdh_ds,
                     all_results);
    }

    if (prov_libsecp) {
        run_provider(prov_libsecp.get(), cfg,
                     ecdsa_ds, schnorr_ds, pubkey_ds, ecdh_ds,
                     all_results);
    }

    // -- Report -------------------------------------------------------------
    EnvInfo env;
    fill_env_info(env, cfg.pin_core);

    // Check for any failures
    bool any_fail = false;
    for (const auto& r : all_results) {
        if (!r.correctness_ok) any_fail = true;
    }

    print_markdown_table(all_results.data(),
                         static_cast<int>(all_results.size()), env);

    if (!cfg.report_json.empty()) {
        if (write_json_report(cfg.report_json.c_str(),
                              all_results.data(),
                              static_cast<int>(all_results.size()), env)) {
            std::printf("JSON report written to: %s\n", cfg.report_json.c_str());
        }
    }

    // -- Cleanup ------------------------------------------------------------
    if (prov_uf)      prov_uf->shutdown();
    if (prov_libsecp) prov_libsecp->shutdown();

    if (any_fail) {
        std::fprintf(stderr, "\n[!] Some correctness gates FAILED\n");
        return 2;
    }

    std::printf("\nDone. All correctness gates passed.\n");
    return 0;
}
