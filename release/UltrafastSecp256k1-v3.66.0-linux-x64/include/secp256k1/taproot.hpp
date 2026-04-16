#ifndef SECP256K1_TAPROOT_HPP
#define SECP256K1_TAPROOT_HPP

// ============================================================================
// Taproot (BIP-341 / BIP-342) -- secp256k1
// ============================================================================
// Implements BIP-341 key tweaking and output key derivation for Taproot.
//
// Taproot uses x-only (32-byte) public keys with implicit even Y.
// A Taproot output key Q is derived from an internal key P and a tweak:
//   Q = P + t*G  where  t = tagged_hash("TapTweak", P.x || merkle_root)
//
// Key concepts:
//   - Internal key (P): the actual signer's public key
//   - Tweak (t): scalar derived from merkle root of script tree
//   - Output key (Q): the key that appears on-chain (P tweaked by t)
//   - Key-path spend: sign with tweaked private key
//   - Script-path spend: reveal internal key + merkle proof + script
//
// Usage:
//   // Key-path: derive tweaked keypair
//   auto [output_key, parity] = taproot_output_key(internal_key_x, merkle_root);
//   auto tweaked_sk = taproot_tweak_privkey(private_key, merkle_root);
//
//   // Script tree construction
//   auto leaf = taproot_leaf_hash(script_bytes);
//   auto branch = taproot_branch_hash(left, right);
// ============================================================================

#include <array>
#include <cstdint>
#include <cstddef>
#include <vector>
#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"

namespace secp256k1 {

using fast::Scalar;
using fast::Point;

// -- Taproot Tagged Hashes (BIP-341 S.5.2) -------------------------------------

// TapTweak hash: t = H_TapTweak(internal_key_x || data)
// If merkle_root is empty (key-path only), uses just internal_key_x.
std::array<std::uint8_t, 32> taproot_tweak_hash(
    const std::array<std::uint8_t, 32>& internal_key_x,
    const std::uint8_t* merkle_root = nullptr,
    std::size_t merkle_root_len = 0);

// TapLeaf hash: H_TapLeaf(leaf_version || compact_size(script) || script)
std::array<std::uint8_t, 32> taproot_leaf_hash(
    const std::uint8_t* script, std::size_t script_len,
    std::uint8_t leaf_version = 0xC0);

// TapBranch hash: H_TapBranch(sorted(a, b))
// Sorts the two 32-byte hashes lexicographically before hashing.
std::array<std::uint8_t, 32> taproot_branch_hash(
    const std::array<std::uint8_t, 32>& a,
    const std::array<std::uint8_t, 32>& b);

// -- Output Key Derivation ----------------------------------------------------

// Derive Taproot output key Q = P + t*G
// Returns {output_key_x (32 bytes), parity (0 = even, 1 = odd)}
// merkle_root can be nullptr for key-path-only outputs.
std::pair<std::array<std::uint8_t, 32>, int> taproot_output_key(
    const std::array<std::uint8_t, 32>& internal_key_x,
    const std::uint8_t* merkle_root = nullptr,
    std::size_t merkle_root_len = 0);

// -- Private Key Tweaking -----------------------------------------------------

// Tweak a private key for key-path spending:
//   d' = d + t  (if P has even y)
//   d' = n - d + t  (if P has odd y, negate first)
// where t = H_TapTweak(P.x || merkle_root)
// Returns tweaked private key (zero on failure).
Scalar taproot_tweak_privkey(
    const Scalar& private_key,
    const std::uint8_t* merkle_root = nullptr,
    std::size_t merkle_root_len = 0);

// -- Taproot Signature Validation ---------------------------------------------

// Verify that output_key was correctly derived from internal_key + merkle_root.
// This is the "control block" validation from BIP-341 S.4.2.
bool taproot_verify_commitment(
    const std::array<std::uint8_t, 32>& output_key_x,
    int output_key_parity,
    const std::array<std::uint8_t, 32>& internal_key_x,
    const std::uint8_t* merkle_root = nullptr,
    std::size_t merkle_root_len = 0);

// -- Script Path: Merkle Proof ------------------------------------------------

// Compute merkle root from a leaf hash and proof path.
// Each proof element is a 32-byte sibling hash. The leaf is combined
// with each sibling in order using taproot_branch_hash.
std::array<std::uint8_t, 32> taproot_merkle_root_from_proof(
    const std::array<std::uint8_t, 32>& leaf_hash,
    const std::vector<std::array<std::uint8_t, 32>>& proof);

// -- TapScript Utilities ------------------------------------------------------

// Construct a simple Merkle tree from a list of TapLeaf hashes.
// Returns the Merkle root. Handles odd-count leaves by promoting the last one.
std::array<std::uint8_t, 32> taproot_merkle_root(
    const std::vector<std::array<std::uint8_t, 32>>& leaf_hashes);

// ============================================================================
// BIP-342: Validation of Taproot Scripts (Tapscript Sighash)
// ============================================================================
// Implements the signature message (SigMsg) for tapscript spending as defined
// in BIP-342 §5. This is an extension of BIP-341 common signature message
// with additional tapscript-specific fields:
//   - tapleaf_hash (tagged hash of the executed script + leaf version)
//   - key_version (0x00 for BIP-342)
//   - code_separator_position (opcode position of last OP_CODESEPARATOR)
//
// The epoch byte (0x00) is prepended to distinguish from future upgrades.

// Input amounts and scriptPubKeys for all inputs
// (required for BIP-341 common signature message)
struct TapSighashTxData {
    std::uint32_t version;
    std::uint32_t locktime;

    // Per-input data
    std::size_t input_count;
    const std::array<std::uint8_t, 32>* prevout_txids; // input_count elements
    const std::uint32_t* prevout_vouts;                 // input_count elements
    const std::uint64_t* input_amounts;                 // input_count elements
    const std::uint32_t* input_sequences;               // input_count elements
    // Per-input scriptPubKeys (for sha_scriptpubkeys)
    const std::uint8_t* const* input_scriptpubkeys;     // input_count pointers
    const std::size_t* input_scriptpubkey_lens;          // input_count lengths

    // Output data
    std::size_t output_count;
    const std::uint64_t* output_values;
    const std::uint8_t* const* output_scriptpubkeys;
    const std::size_t* output_scriptpubkey_lens;
};

// Sighash types for BIP-341/342 (different encoding from legacy)
// 0x00 = SIGHASH_DEFAULT (treated as ALL)
// 0x01 = SIGHASH_ALL
// 0x02 = SIGHASH_NONE
// 0x03 = SIGHASH_SINGLE
// 0x81 = SIGHASH_ALL|ANYONECANPAY
// 0x82 = SIGHASH_NONE|ANYONECANPAY
// 0x83 = SIGHASH_SINGLE|ANYONECANPAY

// Compute BIP-342 tapscript signature hash.
//
// tx_data:     Transaction data (all inputs/outputs)
// input_index: Index of the input being signed
// hash_type:   Sighash type (0x00=DEFAULT, 0x01=ALL, etc.)
// tapleaf_hash: H_TapLeaf(leaf_version || compact_size(script) || script)
// key_version: 0x00 for BIP-342
// code_separator_pos: Position of last OP_CODESEPARATOR, or 0xFFFFFFFF if none
// annex:       Optional annex data (may be nullptr if annex_len == 0)
// annex_len:   Length of annex data
//
// Returns: 32-byte signature hash
std::array<std::uint8_t, 32> tapscript_sighash(
    const TapSighashTxData& tx_data,
    std::size_t input_index,
    std::uint8_t hash_type,
    const std::array<std::uint8_t, 32>& tapleaf_hash,
    std::uint8_t key_version,
    std::uint32_t code_separator_pos,
    const std::uint8_t* annex = nullptr,
    std::size_t annex_len = 0) noexcept;

// Compute BIP-341 key-path signature hash.
// Same as tapscript_sighash but without ext_flag / tapscript-specific data.
std::array<std::uint8_t, 32> taproot_keypath_sighash(
    const TapSighashTxData& tx_data,
    std::size_t input_index,
    std::uint8_t hash_type,
    const std::uint8_t* annex = nullptr,
    std::size_t annex_len = 0) noexcept;

} // namespace secp256k1

#endif // SECP256K1_TAPROOT_HPP
