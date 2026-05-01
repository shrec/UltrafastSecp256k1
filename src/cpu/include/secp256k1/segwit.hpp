#ifndef SECP256K1_SEGWIT_HPP
#define SECP256K1_SEGWIT_HPP

// ============================================================================
// BIP-141: Segregated Witness (Consensus Layer)
// ============================================================================
// Implements witness program creation and validation as defined in BIP-141.
//
// A witness program is defined by:
//   - A version byte (0–16, encoded as OP_0 to OP_16)
//   - A data push of 2–40 bytes
//
// Standard witness programs:
//   v0, 20 bytes → P2WPKH (BIP-141 §4.1)
//   v0, 32 bytes → P2WSH  (BIP-141 §4.2)
//   v1, 32 bytes → P2TR   (BIP-341)
//
// This module provides:
//   - scriptPubKey construction for witness outputs
//   - Witness program extraction from scriptPubKey
//   - Witness program type classification
//   - P2WPKH and P2WSH script code generation for signing
//
// Reference: BIP-141, https://github.com/bitcoin/bips/blob/master/bip-0141.mediawiki
// ============================================================================

#include <array>
#include <cstdint>
#include <cstddef>
#include <vector>

namespace secp256k1 {

// Witness program type classification
enum class WitnessProgramType : std::uint8_t {
    NONE     = 0,  // Not a witness program
    P2WPKH   = 1,  // v0, 20-byte program (BIP-141 §4.1)
    P2WSH    = 2,  // v0, 32-byte program (BIP-141 §4.2)
    P2TR     = 3,  // v1, 32-byte program (BIP-341)
    UNKNOWN  = 4,  // Valid witness program but unknown type (v2-v16, or non-standard length)
};

// Parsed witness program
struct WitnessProgram {
    int version;                          // -1 = not a witness program, 0-16 otherwise
    std::vector<std::uint8_t> program;    // 2-40 bytes
    WitnessProgramType type;
};

// Create a P2WPKH scriptPubKey from a 20-byte pubkey hash.
// Output: OP_0 <20 bytes> = [0x00, 0x14, ...20 bytes...]
std::array<std::uint8_t, 22> segwit_scriptpubkey_p2wpkh(
    const std::uint8_t pubkey_hash[20]) noexcept;

// Create a P2WSH scriptPubKey from a 32-byte witness script hash.
// Output: OP_0 <32 bytes> = [0x00, 0x20, ...32 bytes...]
std::array<std::uint8_t, 34> segwit_scriptpubkey_p2wsh(
    const std::uint8_t script_hash[32]) noexcept;

// Create a P2TR scriptPubKey from a 32-byte x-only output key.
// Output: OP_1 <32 bytes> = [0x51, 0x20, ...32 bytes...]
std::array<std::uint8_t, 34> segwit_scriptpubkey_p2tr(
    const std::uint8_t output_key[32]) noexcept;

// Create a general witness scriptPubKey from version & program.
// version: 0-16
// program: 2-40 bytes
// Returns: [OP_n, push_len, ...program...]
std::vector<std::uint8_t> segwit_scriptpubkey(
    std::uint8_t version,
    const std::uint8_t* program,
    std::size_t program_len) noexcept;

// Check if a scriptPubKey is a witness program.
// BIP-141: A scriptPubKey is a witness program if:
//   - Its length is 4-42 bytes
//   - First byte is OP_0 (0x00) or OP_1..OP_16 (0x51..0x60)
//   - Second byte is a direct data push of 2-40 bytes equal to remaining length
bool is_witness_program(
    const std::uint8_t* script, std::size_t script_len) noexcept;

// Extract and classify a witness program from a scriptPubKey.
// Returns a WitnessProgram struct. If not a valid witness program,
// version = -1 and type = NONE.
WitnessProgram parse_witness_program(
    const std::uint8_t* script, std::size_t script_len) noexcept;

// Compute the witness script hash (SHA256) for P2WSH.
// This is the raw SHA256 (not double-SHA256) of the witness script.
std::array<std::uint8_t, 32> witness_script_hash(
    const std::uint8_t* script, std::size_t script_len) noexcept;

// Compute the P2WPKH scriptCode for BIP-143 signing from a 20-byte pubkey hash.
// Returns: OP_DUP OP_HASH160 <20 bytes> OP_EQUALVERIFY OP_CHECKSIG (25 bytes)
// This is the script that replaces the witness program in the sighash preimage.
std::array<std::uint8_t, 25> p2wpkh_script_code(
    const std::uint8_t pubkey_hash[20]) noexcept;

// Validate P2WPKH witness (BIP-141 §4.1):
// Witness must be exactly [<signature>, <pubkey>]
// pubkey must be 33 bytes (compressed), hash160(pubkey) must match program
bool validate_p2wpkh_witness(
    const std::vector<std::vector<std::uint8_t>>& witness,
    const std::uint8_t program[20]) noexcept;

// Validate P2WSH witness (BIP-141 §4.2):
// Last witness item is the witness script.
// SHA256(witnessScript) must match the 32-byte program.
bool validate_p2wsh_witness(
    const std::vector<std::vector<std::uint8_t>>& witness,
    const std::uint8_t program[32]) noexcept;

// Compute witness weight contribution for a single input.
// witness items count + all items with their lengths.
std::size_t witness_weight(
    const std::vector<std::vector<std::uint8_t>>& witness) noexcept;

} // namespace secp256k1

#endif // SECP256K1_SEGWIT_HPP
