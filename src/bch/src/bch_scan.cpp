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
#include <cstring>

namespace secp256k1::bch {

// Forward: rpa_shared_secret_base / finish declared in rpa.cpp (internal linkage).
// Redeclare here to share — or include a private header.
// For now, call rpa_receiver_shared_secret directly (1 ECDH per tx is fine for CPU).

RpaScanner::RpaScanner(const RpaPaycode& paycode,
                       const fast::Scalar& scan_privkey)
    : paycode_(paycode)
    , scan_privkey_(scan_privkey)
    , network_(paycode.network()) {}

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
    fast::Point S = secp256k1::ct::scalar_mul(epk.point, scan_privkey_);

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

    // PERF: Pre-parse spend_pubkey ONCE (lift_x sqrt ~1.6µs) + pre-build
    // SHA256 midstate over (spend_pubkey || secret) — amortised across all indices.
    secp256k1::EcdsaPublicKey spend_epk{};
    if (!secp256k1::ecdsa_pubkey_parse(spend_epk, paycode_.spend_pubkey.data(), 33))
        return std::nullopt;
    auto pay_base = rpa_payment_key_base(paycode_.spend_pubkey.data(), secret);

    for (uint32_t i = 0; i <= max_key_index; ++i) {
        auto payment_pubkey = rpa_derive_payment_pubkey_fast(
            spend_epk.point, pay_base, i);          // no lift_x, fast ct::generator_mul
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

} // namespace secp256k1::bch
