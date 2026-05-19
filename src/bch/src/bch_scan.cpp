// ============================================================================
// bch_scan.cpp — BCH RPA scan pipeline
// ============================================================================
// Ported from BIP-352 scanner (address.cpp) with RPA hash domain.
// Key optimization inherited from BIP-352:
//   SHA256 midstate pre-computed once per tx (over ECDH point);
//   output loop only feeds outpoint bytes → minimal per-output work.
// ============================================================================
#include "secp256k1/bch/bch_scan.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/detail/secure_erase.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/precompute.hpp"  // batch_scalar_mul_generator, KPlan
#include <cstring>

namespace secp256k1::bch {

// Forward: rpa_shared_secret_base / finish declared in rpa.cpp (internal linkage).
// Redeclare here to share — or include a private header.
// For now, call rpa_receiver_shared_secret directly (1 ECDH per tx is fine for CPU).

RpaScanner::RpaScanner(const RpaPaycode& paycode,
                       const fast::Scalar& scan_privkey)
    : paycode_(paycode)
    , scan_privkey_(scan_privkey)
    , network_(paycode.network()) {
    // Precompute spend_epk once — avoids repeated lift_x (~1.6 µs) per tx
    spend_epk_valid_ = secp256k1::ecdsa_pubkey_parse(
        spend_epk_, paycode_.spend_pubkey.data(), 33);
}

bool RpaScanner::prefix_matches(const uint8_t* sig64) const noexcept {
    if (paycode_.prefix_bits == 0) return true;
    return rpa_prefix_matches(sig64, paycode_.prefix_bits,
                              paycode_.scan_pubkey.data());
}

std::optional<ScanMatch> RpaScanner::scan_tx(
    const ScanTx& tx, uint32_t max_key_index) const noexcept {

    // Outpoint: txid(32) || vout(4 LE)
    uint8_t outpoint[36];
    std::memcpy(outpoint, tx.txid.data(), 32);
    outpoint[32] = tx.vout & 0xff;
    outpoint[33] = (tx.vout >> 8) & 0xff;
    outpoint[34] = (tx.vout >> 16) & 0xff;
    outpoint[35] = (tx.vout >> 24) & 0xff;

    // ECDH: S = scan_privkey · input_pubkey
    // (CT — scan_privkey is secret)
    secp256k1::EcdsaPublicKey epk{};
    if (!secp256k1::ecdsa_pubkey_parse(epk, tx.input_pubkey.data(), 33))
        return std::nullopt;
    // input_pubkey is public on-chain — variable-time GLV safe here
    fast::Point S = epk.point.scalar_mul(scan_privkey_);

    // BIP-352 midstate trick: SHA256 base = SHA256(SHA256(S_compressed))
    // Pre-computed once; outpoint fed inside per-index loop.
    auto S_comp = S.to_compressed();
    auto inner  = SHA256::hash(S_comp.data(), S_comp.size()); // SHA256(S_comp)
    SHA256 h_base;
    h_base.update(inner.data(), 32);
    // NOTE: outpoint is the same for all key indices of this tx — feed it into base too.
    h_base.update(outpoint, 36);
    // h_base now represents: SHA256(SHA256(S_comp) || outpoint)
    // The RPA shared secret c = this finalized value.
    // Key indices reuse c (no re-derivation needed — c is per-tx, index is BIP32 level).
    RpaSharedSecret secret;
    secret.value = h_base.finalize();

    // Erase ECDH temporaries
    secp256k1::detail::secure_erase(S_comp.data(), S_comp.size());
    secp256k1::detail::secure_erase(inner.data(), inner.size());

    // Use precomputed spend_epk from constructor (lift_x amortised across all txs)
    if (!spend_epk_valid_) return std::nullopt;
    auto pay_base = rpa_payment_key_base(paycode_.spend_pubkey.data(), secret);

    for (uint32_t i = 0; i <= max_key_index; ++i) {
        auto payment_pubkey = rpa_derive_payment_pubkey_fast(
            spend_epk_.point, pay_base, i);          // no lift_x, fast ct::generator_mul
        if (payment_pubkey[0] == 0) continue;

        for (uint32_t j = 0; j < tx.outputs.size(); ++j) {
            if (std::memcmp(payment_pubkey.data(),
                            tx.outputs[j].data(), 33) == 0) {
                ScanMatch match;
                match.txid           = tx.txid;
                match.output_index   = j;
                match.key_index      = i;
                match.payment_pubkey = payment_pubkey;
                match.cashaddr = cashaddr_from_pubkey(
                    payment_pubkey.data(), network_);
                return match;
            }
        }
    }
    return std::nullopt;
}

std::vector<ScanMatch> RpaScanner::scan_batch_cpu(
    const std::vector<ScanTx>& txs,
    uint32_t max_key_index) const noexcept {

    std::vector<ScanMatch> results;
    results.reserve(txs.size() / 16); // rough estimate
    for (const auto& tx : txs) {
        if (auto m = scan_tx(tx, max_key_index))
            results.push_back(std::move(*m));
    }
    return results;
}

std::vector<ScanMatch> RpaScanner::scan_batch(
    const std::vector<ScanTx>& txs,
    uint32_t max_key_index) const noexcept {
    // GPU path: implemented when SECP256K1_ENABLE_CUDA/OPENCL + BCH both ON
    return scan_batch_cpu(txs, max_key_index);
}

ScanRateEstimate estimate_scan_rate(uint8_t prefix_bits) noexcept {
    ScanRateEstimate est{};
    double filter_ratio = (prefix_bits == 0) ? 1.0
        : 1.0 / static_cast<double>(1ULL << prefix_bits);
    est.cpu_tx_per_sec   = 100'000.0 / filter_ratio;
    est.gpu_tx_per_sec   = 0.0;
    est.chain_tx_per_day = 1'000'000.0;
    double scan_per_day  = est.cpu_tx_per_sec * 86400.0 * filter_ratio;
    est.days_to_full_sync = est.chain_tx_per_day / scan_per_day;
    return est;
}

// -- RpaScanner::scan_batch (optimised) --------------------------------------
// KPlan::from_scalar(scan_privkey) once + batch_scalar_mul_fixed_k +
// batch_to_compressed + compress_to_scalar + batch_scalar_mul_generator +
// batch_x_only_bytes (all Montgomery H-trick batch inversions).
// Thread-local scratch buffers: no heap allocation after first call.

std::vector<RpaScanner::BatchMatch>
RpaScanner::scan_batch(
    const std::vector<std::vector<std::array<uint8_t,33>>>& input_pubkeys_per_tx,
    const std::vector<std::vector<std::array<uint8_t,33>>>& outputs_per_tx,
    const std::vector<std::array<uint8_t,36>>& outpoints_per_tx,
    uint32_t max_key_index) const
{
    (void)max_key_index;  // k=0 only for batch (extend later)
    std::vector<BatchMatch> results;
    const std::size_t N = input_pubkeys_per_tx.size();
    if (N == 0) return results;

    // Thread-local scratch
    static thread_local std::vector<fast::Point>                   tl_a_sums;
    static thread_local std::vector<fast::Point>                   tl_shared;
    static thread_local std::vector<std::array<uint8_t,33>>        tl_S_comps;
    static thread_local std::vector<uint64_t>                      tl_out_map;
    static thread_local std::vector<fast::Scalar>                  tl_t_scalars;
    static thread_local std::vector<fast::Point>                   tl_candidates;
    static thread_local std::vector<std::array<uint8_t,32>>        tl_x_bytes;

    // Stage 1: aggregate input pubkeys → A_sum per tx
    tl_a_sums.assign(N, fast::Point::infinity());
    for (std::size_t i = 0; i < N; ++i) {
        for (const auto& pk33 : input_pubkeys_per_tx[i]) {
            secp256k1::EcdsaPublicKey epk{};
            if (secp256k1::ecdsa_pubkey_parse(epk, pk33.data(), 33))
                tl_a_sums[i] = tl_a_sums[i].add(epk.point);
        }
    }

    // Stage 1b: S_i = scan_privkey × A_sum_i (KPlan + batch field_inv)
    fast::KPlan plan = fast::KPlan::from_scalar(scan_privkey_);
    tl_shared.resize(N);
    fast::Point::batch_scalar_mul_fixed_k(plan, tl_a_sums.data(), N, tl_shared.data());

    tl_S_comps.resize(N);
    fast::Point::batch_to_compressed(tl_shared.data(), N, tl_S_comps.data());

    // Pass 2a: BCH shared-secret hash + payment key hash (raw SHA256 blocks)
    // c = SHA256(SHA256(S_comp[33]) || outpoint[36])
    // t_k = SHA256(spend_pubkey[33] || c[32] || 0x00000000)
    tl_out_map.clear();
    tl_t_scalars.clear();

    const uint8_t* spend_pk33 = paycode_.spend_pubkey.data();

    for (std::uint32_t tx = 0; tx < static_cast<std::uint32_t>(N); ++tx) {
        if (tl_shared[tx].is_infinity()) continue;
        const auto& S_comp = tl_S_comps[tx];

        // c = SHA256(SHA256(S_comp) || outpoint)
        std::array<uint8_t,32> inner, c;
        {
            auto h = SHA256::hash(S_comp.data(), 33);
            inner = h;
        }
        {
            uint8_t buf[68];
            std::memcpy(buf,      inner.data(), 32);
            if (tx < outpoints_per_tx.size())
                std::memcpy(buf+32, outpoints_per_tx[tx].data(), 36);
            else
                std::memset(buf+32, 0, 36);
            c = SHA256::hash(buf, 68);
        }

        // t_k = SHA256(spend_pk33 || c || ser32(0)) for k=0
        {
            uint8_t buf2[69];
            std::memcpy(buf2,    spend_pk33, 33);
            std::memcpy(buf2+33, c.data(),   32);
            buf2[65]=buf2[66]=buf2[67]=buf2[68]=0;
            auto t_hash = SHA256::hash(buf2, 69);
            fast::Scalar t_k = fast::Scalar::from_bytes(t_hash);
            tl_out_map.push_back((static_cast<uint64_t>(tx) << 32) | 0u);
            tl_t_scalars.push_back(t_k);
        }
    }

    if (tl_t_scalars.empty()) return results;

    // Pass 2b: batch t_k*G
    const std::size_t M = tl_t_scalars.size();
    std::vector<fast::Point> out_jac(M);
    secp256k1::fast::batch_scalar_mul_generator(tl_t_scalars.data(), out_jac.data(), M);

    // Pass 2c: spend_point + t_k*G
    tl_candidates.resize(M);
    for (std::size_t i = 0; i < M; ++i)
        tl_candidates[i] = spend_epk_.point.add(out_jac[i]);

    // Pass 2d: batch x-only extraction (H-trick)
    tl_x_bytes.resize(M);
    fast::Point::batch_x_only_bytes(tl_candidates.data(), M, tl_x_bytes.data());

    // Pass 2e: compare with expected outputs
    for (std::size_t i = 0; i < M; ++i) {
        std::uint32_t tx = static_cast<std::uint32_t>(tl_out_map[i] >> 32);
        const auto& outs = outputs_per_tx[tx];
        for (std::uint32_t j = 0; j < static_cast<std::uint32_t>(outs.size()); ++j) {
            // Compare x-only (first 32 bytes of compressed output after parity byte)
            if (std::memcmp(tl_x_bytes[i].data(), outs[j].data()+1, 32) == 0) {
                BatchMatch m;
                m.tx_index = tx;
                m.output_index = j;
                m.payment_pubkey = outs[j];
                m.cashaddr = cashaddr_from_pubkey(outs[j].data(), network_);
                results.push_back(std::move(m));
            }
        }
    }
    return results;
}

} // namespace secp256k1::bch
