#ifndef SECP256K1_ADDRESS_HPP
#define SECP256K1_ADDRESS_HPP
#pragma once

// ============================================================================
// Bitcoin Address Generation + Encoding for secp256k1
// ============================================================================
// Address types:
//   P2PKH  -- Pay-to-Public-Key-Hash (1..., legacy, Base58Check)
//   P2WPKH -- Pay-to-Witness-Public-Key-Hash (bc1q..., SegWit v0, Bech32)
//   P2TR   -- Pay-to-Taproot (bc1p..., SegWit v1, Bech32m)
//
// Encoding:
//   Base58Check -- P2PKH addresses + WIF private keys
//   Bech32      -- SegWit v0 (BIP-173)
//   Bech32m     -- SegWit v1+ (BIP-350)
//
// BIP-352 Silent Payments:
//   Privacy-preserving addresses. Sender computes unique output from
//   public scan/spend keys; only recipient can detect and spend.
// ============================================================================

#include <array>
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <utility>
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"

namespace secp256k1 {

// -- Network ------------------------------------------------------------------

enum class Network : std::uint8_t {
    Mainnet = 0,
    Testnet = 1
};

// -- Base58Check Encoding -----------------------------------------------------

// Encode data with 4-byte SHA256d checksum in Base58
std::string base58check_encode(const std::uint8_t* data, std::size_t len);

// Decode Base58Check, returns pair of (data, valid)
std::pair<std::vector<std::uint8_t>, bool>
base58check_decode(const std::string& encoded);

// -- Bech32 / Bech32m Encoding (BIP-173 / BIP-350) ---------------------------

enum class Bech32Encoding {
    BECH32,    // SegWit v0 (BIP-173)
    BECH32M    // SegWit v1+ (BIP-350)
};

// Encode a witness program to bech32/bech32m address
// hrp: "bc" for mainnet, "tb" for testnet
// witness_version: 0 for P2WPKH, 1 for P2TR
// witness_program: 20 bytes (v0) or 32 bytes (v1)
std::string bech32_encode(const std::string& hrp,
                          std::uint8_t witness_version,
                          const std::uint8_t* witness_program,
                          std::size_t prog_len);

// Decode bech32/bech32m address
// Returns: {hrp, witness_version, witness_program, valid}
struct Bech32DecodeResult {
    std::string hrp;
    int witness_version;  // -1 if invalid
    std::vector<std::uint8_t> witness_program;
    bool valid;
};
Bech32DecodeResult bech32_decode(const std::string& addr);

// -- HASH160 ------------------------------------------------------------------

// HASH160: RIPEMD160 applied to SHA256 digest
std::array<std::uint8_t, 20> hash160(const std::uint8_t* data, std::size_t len);

// -- Address Derivation -------------------------------------------------------

// P2PKH address from public key (compressed 33 bytes or uncompressed 65 bytes)
// Returns: "1..." (mainnet) or "m/n..." (testnet)
std::string address_p2pkh(const fast::Point& pubkey,
                          Network net = Network::Mainnet);

// P2WPKH address from public key (native SegWit v0)
// Returns: "bc1q..." (mainnet) or "tb1q..." (testnet)
std::string address_p2wpkh(const fast::Point& pubkey,
                           Network net = Network::Mainnet);

// P2TR address from x-only public key (Taproot, SegWit v1)
// Returns: "bc1p..." (mainnet) or "tb1p..." (testnet)
// If internal_key only (no script tree): uses untwisted key
std::string address_p2tr(const fast::Point& internal_key,
                         Network net = Network::Mainnet);

// P2TR address from x-only output key bytes (32 bytes)
std::string address_p2tr_raw(const std::array<std::uint8_t, 32>& output_key_x,
                             Network net = Network::Mainnet);

// P2SH-P2WPKH address (nested/wrapped SegWit, "3..." on mainnet)
// Wraps P2WPKH witness program inside P2SH for backward compatibility
std::string address_p2sh_p2wpkh(const fast::Point& pubkey,
                                Network net = Network::Mainnet);

// P2SH address from a 20-byte script hash (generic)
std::string address_p2sh(const std::array<std::uint8_t, 20>& script_hash,
                         Network net = Network::Mainnet);

// P2WSH address from a 32-byte witness script hash (SegWit v0)
std::string address_p2wsh(const std::array<std::uint8_t, 32>& witness_script_hash,
                          Network net = Network::Mainnet);

// -- CashAddr (Bitcoin Cash BIP-0185) -----------------------------------------

// Encode a hash160 as CashAddr address
// type: 0 = P2PKH, 1 = P2SH
std::string cashaddr_encode(const std::array<std::uint8_t, 20>& hash,
                            const std::string& prefix,
                            std::uint8_t type = 0);

// CashAddr P2PKH from public key
std::string address_cashaddr(const fast::Point& pubkey,
                             const std::string& prefix = "bitcoincash");

// -- WIF (Wallet Import Format) -----------------------------------------------

// Encode private key as WIF string
std::string wif_encode(const fast::Scalar& private_key,
                       bool compressed = true,
                       Network net = Network::Mainnet);

// Decode WIF string to private key
// Returns: {scalar, compressed, network, valid}
struct WIFDecodeResult {
    fast::Scalar key;
    bool compressed;
    Network network;
    bool valid;
};
WIFDecodeResult wif_decode(const std::string& wif);

// -- BIP-352 Silent Payments --------------------------------------------------

// Silent payment address: (scan_pubkey, spend_pubkey) pair
struct SilentPaymentAddress {
    fast::Point scan_pubkey;     // B_scan
    fast::Point spend_pubkey;    // B_spend
    
    // Encode to sp1q... address (mainnet) or tsp1q... (testnet)
    std::string encode(Network net = Network::Mainnet) const;
};

// Generate silent payment address from scan and spend private keys
SilentPaymentAddress
silent_payment_address(const fast::Scalar& scan_privkey,
                       const fast::Scalar& spend_privkey);

// Sender: Compute output public key for a silent payment
// input_privkeys: sender's input private keys (for ECDH)
// input_pubkeys: corresponding public keys
// recipient: recipient's silent payment address
// k: output index (for multiple outputs to same recipient)
// Returns: {output_pubkey, output_tweaked_key}
std::pair<fast::Point, fast::Scalar>
silent_payment_create_output(const std::vector<fast::Scalar>& input_privkeys,
                             const SilentPaymentAddress& recipient,
                             std::uint32_t k = 0);

// Receiver: Scan transaction to detect silent payment outputs
// scan_privkey: receiver's scan private key
// spend_privkey: receiver's spend private key
// input_pubkeys: all input public keys from the transaction
// output_pubkeys: all output x-only public keys to check
// Returns: vector of {output_index, tweaked_privkey} for detected outputs
std::vector<std::pair<std::uint32_t, fast::Scalar>>
silent_payment_scan(const fast::Scalar& scan_privkey,
                    const fast::Scalar& spend_privkey,
                    const std::vector<fast::Point>& input_pubkeys,
                    const std::vector<std::array<std::uint8_t, 32>>& output_pubkeys);

// ── Fast batch scanner (wallet-optimised, non-CT) ────────────────────────────
//
// For full-blockchain scanning: scan_privkey is constant across millions of txs.
// This API amortizes four expensive costs:
//   (1) KPlan::from_scalar(scan_privkey): GLV decompose + wNAF — computed ONCE.
//   (2) batch_scalar_mul_fixed_k for Stage 1: shared wNAF loop, 1 field_inv per
//       chunk (chunk_size≈2048). Treats scan as "one execution schedule, N data".
//   (3) batch_to_compressed for Stage 1: single field inversion across all N secrets.
//   (4) batch_x_only_bytes for Stage 2: single field inversion across all N outputs.
//
// Compared to N × silent_payment_scan() on i5-14400F (50K txs, 1 thread):
//   silent_payment_scan: ~47K ns/tx  (CT Stage 1 + ct::generator_mul per tx)
//   fast_scan_batch:     ~23K ns/tx  →  2.06× speedup
//
// Stage 2 standalone throughput (16T, w=18 table): 1.60 M/s
// Full pipeline (Stage 1 + Stage 2) at 16T: ~206K tx/s (Stage 1 dominates)
//
// Not constant-time: suitable for trusted-environment wallet scanning.
// For server-side scanning where side-channel matters, use silent_payment_scan().

// One effective input pubkey + its candidate outputs per transaction.
// a_eff = input_hash × A_sum (caller's responsibility to aggregate inputs).
// For raw-input aggregation, use ScanTxRaw + compute_a_eff() below.
struct ScanTx {
    fast::Point a_eff;                                    // input_hash × A_sum
    std::vector<std::array<std::uint8_t, 32>> outputs;   // x-only output pubkeys to test
};

// Raw-input variant: caller provides individual input pubkeys per transaction.
// compute_a_eff() aggregates them into a_eff using Pippenger/Strauss batch MSM
// (all scalars = 1, so this reduces to batch point addition with one field_inv).
struct ScanTxRaw {
    std::vector<fast::Point> input_pubkeys;               // all P_j from tx inputs
    std::array<std::uint8_t, 36> smallest_outpoint;      // BIP-352 input hash seed
    std::vector<std::array<std::uint8_t, 32>> outputs;   // x-only output pubkeys
};

// Aggregate raw inputs into ScanTx.a_eff using batch MSM (scalars all = 1):
//   a_eff = input_hash(outpoint, A_sum) × A_sum
//   where A_sum = Σ input_pubkeys[j]
// Uses multi_scalar_mul (Strauss GLV + effective-affine) for the summation.
// Pippenger crossover applies at n_inputs > ~128; Strauss is used for all n < 128.
ScanTx compute_a_eff(const ScanTxRaw& raw);

// One detected match.
struct ScanMatch {
    std::uint32_t tx_index;       // index into the txs[] vector
    std::uint32_t output_index;   // which output_pubkeys[] entry matched
    fast::Scalar  tweaked_privkey; // spend_privkey + t_k (spendable key)
};

// Batch scanner: processes txs.size() transactions in one call.
// Stage 1 uses batch_scalar_mul_fixed_k (shared wNAF schedule, lockstep execution).
// Stage 2 uses batch_x_only_bytes (one field_inv for all output candidates).
std::vector<ScanMatch>
fast_scan_batch(const fast::Scalar& scan_privkey,
                const fast::Scalar& spend_privkey,
                const std::vector<ScanTx>& txs);

} // namespace secp256k1

#endif // SECP256K1_ADDRESS_HPP
