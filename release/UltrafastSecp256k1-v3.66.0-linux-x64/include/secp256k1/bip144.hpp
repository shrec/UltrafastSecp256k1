#ifndef SECP256K1_BIP144_HPP
#define SECP256K1_BIP144_HPP

// ============================================================================
// BIP-144: Segregated Witness (Peer Services) — Witness Serialization
// ============================================================================
// Implements the witness transaction serialization format and wtxid computation:
//
//   [nVersion][marker][flag][txins][txouts][witness][nLockTime]
//
// Where marker=0x00, flag=0x01 indicate a witness transaction.
//
// Also provides:
//   - wtxid computation (double-SHA256 of witness-serialized transaction)
//   - txid computation  (double-SHA256 of legacy serialization, no witness)
//   - witness commitment (for coinbase witness commitment)
//
// Reference: BIP-144, https://github.com/bitcoin/bips/blob/master/bip-0144.mediawiki
// ============================================================================

#include <array>
#include <cstdint>
#include <cstddef>
#include <vector>

namespace secp256k1 {

// A single witness item (byte vector)
using WitnessItem = std::vector<std::uint8_t>;

// Witness stack for one input
using WitnessStack = std::vector<WitnessItem>;

// A transaction input for BIP-144 serialization
struct TxInput {
    std::array<std::uint8_t, 32> prev_txid;  // LE txid
    std::uint32_t prev_vout;
    std::vector<std::uint8_t> script_sig;    // scriptSig (usually empty for segwit)
    std::uint32_t sequence;
};

// A transaction output for BIP-144 serialization
struct TxOut {
    std::uint64_t value;
    std::vector<std::uint8_t> script_pubkey;
};

// Witness transaction data
struct WitnessTx {
    std::uint32_t version;
    std::vector<TxInput>      inputs;
    std::vector<TxOut>        outputs;
    std::vector<WitnessStack> witness;  // One WitnessStack per input
    std::uint32_t locktime;
};

// Serialize a transaction in witness format (BIP-144).
// Returns the full serialized bytes:
//   [nVersion][0x00][0x01][vin][vout][witness][nLockTime]
std::vector<std::uint8_t> witness_serialize(const WitnessTx& tx) noexcept;

// Serialize a transaction in legacy format (no marker/flag, no witness).
// Returns: [nVersion][vin][vout][nLockTime]
std::vector<std::uint8_t> legacy_serialize(const WitnessTx& tx) noexcept;

// Compute txid: double-SHA256 of legacy serialization (LE byte order).
std::array<std::uint8_t, 32> compute_txid(const WitnessTx& tx) noexcept;

// Compute wtxid: double-SHA256 of witness serialization (LE byte order).
// For coinbase: wtxid is defined as 32 zero bytes.
std::array<std::uint8_t, 32> compute_wtxid(const WitnessTx& tx) noexcept;

// Compute witness commitment hash for coinbase:
//   SHA256(SHA256(witness_root || witness_nonce))
// witness_root: merkle root of wtxid tree
// witness_nonce: 32-byte nonce from coinbase witness (typically all zeros)
std::array<std::uint8_t, 32> witness_commitment(
    const std::array<std::uint8_t, 32>& witness_root,
    const std::array<std::uint8_t, 32>& witness_nonce) noexcept;

// Check if a transaction has non-empty witness data.
bool has_witness(const WitnessTx& tx) noexcept;

// Compute the weight of a transaction (BIP-141 weight units).
// weight = base_size * 3 + total_size
// Where base_size is legacy serialization size,
// and total_size is witness serialization size.
std::uint64_t tx_weight(const WitnessTx& tx) noexcept;

// Compute virtual size (vsize) = ceil(weight / 4)
std::uint64_t tx_vsize(const WitnessTx& tx) noexcept;

} // namespace secp256k1

#endif // SECP256K1_BIP144_HPP
