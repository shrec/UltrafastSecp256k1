#ifndef SECP256K1_TAPROOT_HPP
#define SECP256K1_TAPROOT_HPP

// ============================================================================
// Taproot (BIP-341 / BIP-342) — secp256k1
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

// ── Taproot Tagged Hashes (BIP-341 §5.2) ─────────────────────────────────────

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

// ── Output Key Derivation ────────────────────────────────────────────────────

// Derive Taproot output key Q = P + t*G
// Returns {output_key_x (32 bytes), parity (0 = even, 1 = odd)}
// merkle_root can be nullptr for key-path-only outputs.
std::pair<std::array<std::uint8_t, 32>, int> taproot_output_key(
    const std::array<std::uint8_t, 32>& internal_key_x,
    const std::uint8_t* merkle_root = nullptr,
    std::size_t merkle_root_len = 0);

// ── Private Key Tweaking ─────────────────────────────────────────────────────

// Tweak a private key for key-path spending:
//   d' = d + t  (if P has even y)
//   d' = n - d + t  (if P has odd y, negate first)
// where t = H_TapTweak(P.x || merkle_root)
// Returns tweaked private key (zero on failure).
Scalar taproot_tweak_privkey(
    const Scalar& private_key,
    const std::uint8_t* merkle_root = nullptr,
    std::size_t merkle_root_len = 0);

// ── Taproot Signature Validation ─────────────────────────────────────────────

// Verify that output_key was correctly derived from internal_key + merkle_root.
// This is the "control block" validation from BIP-341 §4.2.
bool taproot_verify_commitment(
    const std::array<std::uint8_t, 32>& output_key_x,
    int output_key_parity,
    const std::array<std::uint8_t, 32>& internal_key_x,
    const std::uint8_t* merkle_root = nullptr,
    std::size_t merkle_root_len = 0);

// ── Script Path: Merkle Proof ────────────────────────────────────────────────

// Compute merkle root from a leaf hash and proof path.
// Each proof element is a 32-byte sibling hash. The leaf is combined
// with each sibling in order using taproot_branch_hash.
std::array<std::uint8_t, 32> taproot_merkle_root_from_proof(
    const std::array<std::uint8_t, 32>& leaf_hash,
    const std::vector<std::array<std::uint8_t, 32>>& proof);

// ── TapScript Utilities ──────────────────────────────────────────────────────

// Construct a simple Merkle tree from a list of TapLeaf hashes.
// Returns the Merkle root. Handles odd-count leaves by promoting the last one.
std::array<std::uint8_t, 32> taproot_merkle_root(
    const std::vector<std::array<std::uint8_t, 32>>& leaf_hashes);

} // namespace secp256k1

#endif // SECP256K1_TAPROOT_HPP
