// =============================================================================
// ltc_sp.hpp — Litecoin Silent Payments (LTC-SP)
// =============================================================================
// Privacy-preserving reusable addresses for Litecoin, analogous to BIP-352
// (Bitcoin Silent Payments) but with LTC-specific tagged hash domain.
//
// Protocol overview:
//   Recipient publishes one (scan_pubkey, spend_pubkey) pair encoded as
//   ltcsp1... paycode.  Sender creates unique P2TR output per transaction
//   using ECDH over the scan key.  Only the recipient can detect outputs.
//
// Differences from BIP-352:
//   - Tagged hash domain: "LTCSP/SharedSecret", "LTCSP/Inputs"
//     (prevents cross-chain replay: a LTC-SP address rejects BTC payments)
//   - Paycode HRP:  "ltcsp" (BTC: "sp")
//   - Output HRP:   "ltc"   (BTC: "bc") — standard Litecoin P2TR (ltc1p...)
//
// Protocol is otherwise identical to BIP-352 §3.
//
// Usage:
//   // Receiver: generate paycode
//   auto addr = ltcsp_address(scan_sk, spend_sk);
//   printf("%s\n", addr.paycode.c_str()); // ltcsp1q...
//
//   // Sender: compute output
//   auto [out_pubkey, tweak] = ltcsp_create_output(input_privkeys, addr);
//   // encode out_pubkey as ltc1p... P2TR output
//
//   // Receiver: scan transaction
//   auto matches = ltcsp_scan(scan_sk, spend_sk, input_pubkeys, output_pubkeys);
//   for (auto [idx, privkey] : matches) { /* spend output[idx] */ }
//
// Requires: SECP256K1_BUILD_BIP352=ON (reuses address.hpp bech32_encode)
// =============================================================================

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"

namespace secp256k1::ltc {

// -- LTC-SP address (paycode) -------------------------------------------------

struct LtcSpAddress {
    fast::Point scan_pubkey;   // B_scan
    fast::Point spend_pubkey;  // B_spend

    // Encode as ltcsp1... paycode (bech32m, HRP = "ltcsp")
    // Format: witness_version=0 + scan_pubkey(33) + spend_pubkey(33)
    std::string encode() const;

    // Decode ltcsp1... paycode
    // Returns: {address, valid}
    static std::pair<LtcSpAddress, bool> decode(const std::string& paycode);
};

// Generate LTC-SP address from scan and spend private keys
LtcSpAddress ltcsp_address(const fast::Scalar& scan_privkey,
                            const fast::Scalar& spend_privkey);

// -- Sender API ---------------------------------------------------------------

// Compute LTC-SP output public key.
// input_privkeys: sender's input private keys (for ECDH over B_scan)
// recipient:      decoded ltcsp1... address
// k:              output index (for multiple outputs to same recipient in one tx)
//
// Returns: {output_pubkey (32-byte x-only for P2TR), tweak_scalar}
// The output address is: bech32m("ltc", 1, output_pubkey.x_bytes())
std::pair<fast::Point, fast::Scalar>
ltcsp_create_output(const std::vector<fast::Scalar>& input_privkeys,
                    const LtcSpAddress& recipient,
                    std::uint32_t k = 0);

// -- Receiver API -------------------------------------------------------------

// Scan transaction outputs for LTC-SP payments.
// scan_privkey:   receiver's scan private key
// spend_privkey:  receiver's spend private key
// input_pubkeys:  all input public keys from the transaction
// output_pubkeys: x-only public keys of all outputs to scan (32 bytes each)
//
// Returns: {output_index, spend_privkey_for_output} for each matched output.
//          The returned privkey lets the receiver spend the output directly.
std::vector<std::pair<std::uint32_t, fast::Scalar>>
ltcsp_scan(const fast::Scalar& scan_privkey,
           const fast::Scalar& spend_privkey,
           const std::vector<fast::Point>& input_pubkeys,
           const std::vector<std::array<std::uint8_t, 32>>& output_pubkeys);

// =============================================================================
// LtcSpScanner — wallet-optimised batch scanner
// =============================================================================
// For scanning the full LTC blockchain, calling ltcsp_scan() per transaction
// is suboptimal: GLV decomposition of scan_privkey is recomputed every call.
//
// LtcSpScanner precomputes the GLV tables for scan_privkey ONCE and reuses
// them for every transaction, identical to BIP-352's fast_scan_batch approach.
// This closes the 13x gap between ltcsp_scan (54k tx/s) and BIP-352 fast scan
// (638k tx/s on 16 cores).
//
// Usage:
//   LtcSpScanner scanner(scan_privkey, spend_privkey);
//   for (auto& tx : block) {
//     auto matches = scanner.scan_tx(tx.input_pubkeys, tx.output_pubkeys);
//   }
struct LtcSpScanner {
    explicit LtcSpScanner(const fast::Scalar& scan_sk,
                          const fast::Scalar& spend_sk);

    // Scan a single transaction.
    // input_pubkeys:  all input public keys (sum computed internally)
    // output_pubkeys: x-only output public keys (32 bytes each)
    // Returns: {output_index, spend_privkey} for each matched output
    std::vector<std::pair<std::uint32_t, fast::Scalar>>
    scan_tx(const std::vector<fast::Point>& input_pubkeys,
            const std::vector<std::array<std::uint8_t, 32>>& output_pubkeys) const;

    // Batch scanner: amortizes KPlan wNAF + batch field_inv across N transactions.
    // N=10000+ gives best throughput. Same optimizations as BTC fast_scan_batch.
    // Returns: {tx_index, output_index, spend_privkey} for each match.
    struct BatchMatch {
        std::uint32_t tx_index;
        std::uint32_t output_index;
        fast::Scalar  spend_privkey;
    };
    std::vector<BatchMatch>
    scan_batch(const std::vector<std::vector<fast::Point>>& input_pubkeys_per_tx,
               const std::vector<std::vector<std::array<std::uint8_t, 32>>>& outputs_per_tx) const;

private:
    fast::Scalar scan_privkey_;
    fast::Scalar spend_privkey_;
    fast::Point  spend_pubkey_;   // B_spend = spend_sk * G, precomputed
};

} // namespace secp256k1::ltc
