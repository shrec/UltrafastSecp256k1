#pragma once
#include "bch_types.hpp"
#include "rpa.hpp"
#include "cashaddr.hpp"
#include "secp256k1/ecdsa.hpp"
#include <vector>
#include <functional>

namespace secp256k1::bch {

struct ScanTx {
    std::array<uint8_t, 32> txid;
    uint32_t                vout;
    std::array<uint8_t, 33> input_pubkey;
    std::vector<std::array<uint8_t, 33>> outputs;
};

struct ScanMatch {
    std::array<uint8_t, 32> txid;
    uint32_t   output_index;
    uint32_t   key_index;
    std::array<uint8_t, 33> payment_pubkey;
    std::string cashaddr;
};

class RpaScanner {
public:
    explicit RpaScanner(const RpaPaycode& paycode, const fast::Scalar& scan_privkey);
    [[nodiscard]] std::optional<ScanMatch> scan_tx(const ScanTx& tx,
        uint32_t max_key_index = 30) const noexcept;
    [[nodiscard]] std::vector<ScanMatch> scan_batch_cpu(
        const std::vector<ScanTx>& txs, uint32_t max_key_index = 30) const noexcept;
    [[nodiscard]] std::vector<ScanMatch> scan_batch(
        const std::vector<ScanTx>& txs, uint32_t max_key_index = 30) const noexcept;
    [[nodiscard]] bool prefix_matches(const uint8_t* sig64) const noexcept;
    const RpaPaycode& paycode() const noexcept { return paycode_; }

    // Batch scanner: KPlan + batch field_inv across N transactions.
    // input_pubkeys_per_tx[i]: compressed input pubkeys for tx i (33 bytes each)
    // outputs_per_tx[i]:       compressed output pubkeys for tx i (33 bytes each)
    // outpoints_per_tx[i]:     txid[32]+vout[4] for tx i (36 bytes)
    struct BatchMatch {
        std::uint32_t tx_index;
        std::uint32_t output_index;
        std::array<uint8_t,33> payment_pubkey;
        std::string cashaddr;
    };
    [[nodiscard]] std::vector<BatchMatch>
    scan_batch(const std::vector<std::vector<std::array<uint8_t,33>>>& input_pubkeys_per_tx,
               const std::vector<std::vector<std::array<uint8_t,33>>>& outputs_per_tx,
               const std::vector<std::array<uint8_t,36>>& outpoints_per_tx,
               uint32_t max_key_index = 0) const;
private:
    RpaPaycode        paycode_;
    fast::Scalar      scan_privkey_;
    Network           network_;
    // Precomputed once — avoids repeated lift_x (√ ~1.6 µs) per tx
    secp256k1::EcdsaPublicKey spend_epk_;   // parsed spend_pubkey from paycode
    bool              spend_epk_valid_ = false;
};

struct ScanRateEstimate {
    double cpu_tx_per_sec;
    double gpu_tx_per_sec;
    double chain_tx_per_day;
    double days_to_full_sync;
};
[[nodiscard]] ScanRateEstimate estimate_scan_rate(uint8_t prefix_bits) noexcept;

} // namespace secp256k1::bch
