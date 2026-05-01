// ============================================================================
// bench_hotpaths.cpp -- Focused benchmarks for optimization candidates
// ============================================================================

#include "secp256k1/address.hpp"
#include "secp256k1/batch_verify.hpp"
#include "secp256k1/benchmark_harness.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"
#include "ufsecp/ufsecp.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace secp256k1;
using namespace secp256k1::fast;

namespace {

struct CliOptions {
    int passes = 11;
    bool quick = false;
};

CliOptions parse_cli(int argc, char** argv) {
    CliOptions opts;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--quick") == 0) {
            opts.quick = true;
        } else if (std::strcmp(argv[i], "--passes") == 0 && i + 1 < argc) {
            opts.passes = std::atoi(argv[++i]);
            if (opts.passes < 3) {
                opts.passes = 3;
            }
        }
    }
    return opts;
}

std::array<std::uint8_t, 32> make_hash(std::uint64_t seed) {
    std::array<std::uint8_t, 32> out{};
    for (std::size_t i = 0; i < 4; ++i) {
        std::uint64_t word = seed + 0x9e3779b97f4a7c15ULL * (i + 1);
        std::memcpy(out.data() + i * 8, &word, sizeof(word));
    }
    return out;
}

Scalar make_nonzero_scalar(std::uint64_t seed) {
    Scalar scalar = Scalar::from_bytes(make_hash(seed));
    if (scalar.is_zero()) {
        scalar = Scalar::one();
    }
    return scalar;
}

void cpu_warmup() {
    Point const generator = Point::generator();
    Scalar scalar = make_nonzero_scalar(0xC0FFEE);
    volatile std::uint8_t sink = 0;
    for (int i = 0; i < 512; ++i) {
        Point point = generator.scalar_mul(scalar);
        auto compressed = point.to_compressed();
        sink ^= compressed[1];
        scalar += Scalar::one();
    }
    bench::DoNotOptimize(sink);
}

struct BatchBenchFixture {
    std::vector<ECDSABatchEntry> valid_entries;
    std::vector<ECDSABatchEntry> one_invalid_entries;
};

BatchBenchFixture make_batch_fixture(std::size_t count) {
    BatchBenchFixture fixture;
    fixture.valid_entries.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        Scalar const seckey = make_nonzero_scalar(0x1000 + i);
        auto const msg = make_hash(0xABC000 + i);
        ECDSASignature const sig = ecdsa_sign(msg, seckey);
        fixture.valid_entries.push_back(ECDSABatchEntry{msg, Point::generator().scalar_mul(seckey), sig});
    }
    fixture.one_invalid_entries = fixture.valid_entries;
    fixture.one_invalid_entries.back().msg_hash[0] ^= 0x01;
    return fixture;
}

struct FrostBenchFixture {
    ufsecp_ctx* ctx = nullptr;
    std::vector<std::uint8_t> all_commits;
    std::vector<std::uint8_t> received_shares;
    std::array<std::uint8_t, UFSECP_FROST_KEYPKG_LEN> keypkg{};
    std::uint32_t threshold = 5;
    std::uint32_t num_participants = 8;
    std::uint32_t participant_id = 1;
};

bool init_frost_fixture(FrostBenchFixture& fixture) {
    if (ufsecp_ctx_create(&fixture.ctx) != UFSECP_OK) {
        return false;
    }

    std::size_t const commit_record_len = 8 + static_cast<std::size_t>(fixture.threshold) * 33;
    fixture.all_commits.resize(static_cast<std::size_t>(fixture.num_participants) * commit_record_len);
    fixture.received_shares.resize(static_cast<std::size_t>(fixture.num_participants) * UFSECP_FROST_SHARE_LEN);

    std::vector<std::uint8_t> commit_buf(commit_record_len);
    std::vector<std::uint8_t> shares_buf(static_cast<std::size_t>(fixture.num_participants) * UFSECP_FROST_SHARE_LEN);

    for (std::uint32_t sender = 1; sender <= fixture.num_participants; ++sender) {
        std::array<std::uint8_t, 32> seed = make_hash(0xF200 + sender);
        std::size_t commit_len = commit_buf.size();
        std::size_t shares_len = shares_buf.size();
        if (ufsecp_frost_keygen_begin(
                fixture.ctx,
                sender,
                fixture.threshold,
                fixture.num_participants,
                seed.data(),
                commit_buf.data(),
                &commit_len,
                shares_buf.data(),
                &shares_len) != UFSECP_OK) {
            return false;
        }
        if (commit_len != commit_record_len || shares_len != shares_buf.size()) {
            return false;
        }

        std::memcpy(
            fixture.all_commits.data() + static_cast<std::size_t>(sender - 1) * commit_record_len,
            commit_buf.data(),
            commit_record_len);
        std::memcpy(
            fixture.received_shares.data() + static_cast<std::size_t>(sender - 1) * UFSECP_FROST_SHARE_LEN,
            shares_buf.data() + static_cast<std::size_t>(fixture.participant_id - 1) * UFSECP_FROST_SHARE_LEN,
            UFSECP_FROST_SHARE_LEN);
    }

    return ufsecp_frost_keygen_finalize(
               fixture.ctx,
               fixture.participant_id,
               fixture.all_commits.data(),
               fixture.all_commits.size(),
               fixture.received_shares.data(),
               fixture.received_shares.size(),
               fixture.threshold,
               fixture.num_participants,
               fixture.keypkg.data()) == UFSECP_OK;
}

void destroy_frost_fixture(FrostBenchFixture& fixture) {
    ufsecp_ctx_destroy(fixture.ctx);
    fixture.ctx = nullptr;
}

} // namespace

int main(int argc, char** argv) {
    CliOptions const opts = parse_cli(argc, argv);
    bench::pin_thread_and_elevate();
    cpu_warmup();

    bench::Harness harness(opts.quick ? 100 : 300,
                           static_cast<std::size_t>(opts.quick ? 5 : opts.passes));

    BatchBenchFixture const batch_fixture = make_batch_fixture(opts.quick ? 32U : 64U);

    FrostBenchFixture frost_fixture;
    if (!init_frost_fixture(frost_fixture)) {
        std::fprintf(stderr, "failed to initialize FROST benchmark fixture\n");
        destroy_frost_fixture(frost_fixture);
        return 1;
    }

    Point const address_pubkey = Point::generator().scalar_mul(make_nonzero_scalar(0x4242));

    std::printf("Hot Path Benchmarks\n");
    std::printf("  Timer:  %s\n", bench::Timer::timer_name());
    std::printf("  Passes: %zu\n", harness.passes);
    std::printf("  Batch:  %zu ECDSA entries\n", batch_fixture.valid_entries.size());
    std::printf("  FROST:  t=%u n=%u\n\n", frost_fixture.threshold, frost_fixture.num_participants);

    double const batch_verify_ns = harness.run_and_print(
        "ecdsa_batch_verify(valid)",
        opts.quick ? 16 : 32,
        [&]() {
            bool const ok = ecdsa_batch_verify(batch_fixture.valid_entries.data(), batch_fixture.valid_entries.size());
            bench::DoNotOptimize(ok);
        });

    double const batch_identify_ns = harness.run_and_print(
        "ecdsa_batch_identify_invalid",
        opts.quick ? 8 : 16,
        [&]() {
            auto invalid = ecdsa_batch_identify_invalid(
                batch_fixture.one_invalid_entries.data(),
                batch_fixture.one_invalid_entries.size());
            bench::DoNotOptimize(invalid);
        });

    double const frost_finalize_ns = harness.run_and_print(
        "ufsecp_frost_keygen_finalize",
        opts.quick ? 20 : 40,
        [&]() {
            ufsecp_error_t const err = ufsecp_frost_keygen_finalize(
                frost_fixture.ctx,
                frost_fixture.participant_id,
                frost_fixture.all_commits.data(),
                frost_fixture.all_commits.size(),
                frost_fixture.received_shares.data(),
                frost_fixture.received_shares.size(),
                frost_fixture.threshold,
                frost_fixture.num_participants,
                frost_fixture.keypkg.data());
            bench::DoNotOptimize(err);
        });

    double const p2pkh_ns = harness.run_and_print(
        "address_p2pkh",
        opts.quick ? 128 : 256,
        [&]() {
            std::string address = address_p2pkh(address_pubkey, Network::Mainnet);
            bench::DoNotOptimize(address);
        });

    double const p2wpkh_ns = harness.run_and_print(
        "address_p2wpkh",
        opts.quick ? 192 : 384,
        [&]() {
            std::string address = address_p2wpkh(address_pubkey, Network::Mainnet);
            bench::DoNotOptimize(address);
        });

    std::printf("\nSummary\n");
    std::printf("  batch_verify_ns=%.2f\n", batch_verify_ns);
    std::printf("  batch_identify_invalid_ns=%.2f\n", batch_identify_ns);
    std::printf("  frost_finalize_ns=%.2f\n", frost_finalize_ns);
    std::printf("  address_p2pkh_ns=%.2f\n", p2pkh_ns);
    std::printf("  address_p2wpkh_ns=%.2f\n", p2wpkh_ns);

    destroy_frost_fixture(frost_fixture);
    return 0;
}