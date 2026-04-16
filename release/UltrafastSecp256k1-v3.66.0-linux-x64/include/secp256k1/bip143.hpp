#ifndef SECP256K1_BIP143_HPP
#define SECP256K1_BIP143_HPP

// ============================================================================
// BIP-143: Transaction Signature Verification for Version 0 Witness Program
// ============================================================================
// Implements the SegWit v0 sighash algorithm as specified in BIP-143.
// This defines how transaction digests are computed for signing SegWit v0
// inputs (P2WPKH and P2WSH), replacing the legacy sighash algorithm.
//
// The BIP-143 digest commits to:
//   - nVersion, hashPrevouts, hashSequence
//   - outpoint (txid + vout)
//   - scriptCode, value
//   - nSequence, hashOutputs
//   - nLockTime, nHashType
//
// This prevents the quadratic hashing problem of legacy transactions.
//
// Reference: BIP-143, https://github.com/bitcoin/bips/blob/master/bip-0143.mediawiki
// ============================================================================

#include <array>
#include <cstdint>
#include <cstddef>
#include <vector>

namespace secp256k1 {

// Sighash types (same as legacy, used in witness sighash)
enum class SighashType : std::uint32_t {
    ALL          = 0x01,
    NONE         = 0x02,
    SINGLE       = 0x03,
    ANYONECANPAY = 0x80,
};

// A single transaction outpoint (txid + output index)
struct Outpoint {
    std::array<std::uint8_t, 32> txid;  // LE txid
    std::uint32_t vout;
};

// A single transaction output (value + scriptPubKey)
struct TxOutput {
    std::uint64_t value;
    std::vector<std::uint8_t> script_pubkey;
};

// Precomputed hash components for BIP-143 sighash.
// Reusable across multiple inputs of the same transaction.
struct Bip143Preimage {
    std::uint32_t version;
    std::array<std::uint8_t, 32> hash_prevouts;    // dSHA256 of all outpoints
    std::array<std::uint8_t, 32> hash_sequence;    // dSHA256 of all sequences
    std::array<std::uint8_t, 32> hash_outputs;     // dSHA256 of all outputs
    std::uint32_t locktime;
};

// Compute hashPrevouts: double-SHA256 of all outpoints concatenated.
std::array<std::uint8_t, 32> bip143_hash_prevouts(
    const Outpoint* outpoints, std::size_t count) noexcept;

// Compute hashSequence: double-SHA256 of all nSequence values concatenated.
std::array<std::uint8_t, 32> bip143_hash_sequence(
    const std::uint32_t* sequences, std::size_t count) noexcept;

// Compute hashOutputs: double-SHA256 of all outputs serialized.
std::array<std::uint8_t, 32> bip143_hash_outputs(
    const TxOutput* outputs, std::size_t count) noexcept;

// Build reusable preimage components from transaction data.
Bip143Preimage bip143_build_preimage(
    std::uint32_t version,
    const Outpoint* outpoints, std::size_t input_count,
    const std::uint32_t* sequences,
    const TxOutput* outputs, std::size_t output_count,
    std::uint32_t locktime) noexcept;

// Compute BIP-143 sighash for a specific input.
// script_code: the scriptCode for this input (P2WPKH: OP_DUP OP_HASH160...;
//              P2WSH: the witness script).
// value: the value in satoshis of the output being spent.
// sighash_type: SIGHASH_ALL, SIGHASH_NONE, SIGHASH_SINGLE, |ANYONECANPAY
//
// For ANYONECANPAY: hashPrevouts and hashSequence are zeroed.
// For NONE: hashOutputs is zeroed.
// For SINGLE: hashOutputs is hash of the output at same index (or zeros if
//             the index exceeds output count).
std::array<std::uint8_t, 32> bip143_sighash(
    const Bip143Preimage& preimage,
    const Outpoint& outpoint,
    const std::uint8_t* script_code, std::size_t script_code_len,
    std::uint64_t value,
    std::uint32_t sequence,
    std::uint32_t sighash_type) noexcept;

// Convenience: build P2WPKH scriptCode from a 20-byte pubkey hash.
// Returns: OP_DUP OP_HASH160 <20 bytes> OP_EQUALVERIFY OP_CHECKSIG (25 bytes)
std::array<std::uint8_t, 25> bip143_p2wpkh_script_code(
    const std::uint8_t pubkey_hash[20]) noexcept;

} // namespace secp256k1

#endif // SECP256K1_BIP143_HPP
