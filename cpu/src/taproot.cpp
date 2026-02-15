#include "secp256k1/taproot.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/sha256.hpp"
#include <cstring>
#include <algorithm>

namespace secp256k1 {

using fast::Scalar;
using fast::Point;
using fast::FieldElement;

// ── TapTweak Hash ────────────────────────────────────────────────────────────

std::array<uint8_t, 32> taproot_tweak_hash(
    const std::array<uint8_t, 32>& internal_key_x,
    const uint8_t* merkle_root,
    std::size_t merkle_root_len) {

    // Concatenate: internal_key_x [|| merkle_root]
    std::size_t total = 32 + merkle_root_len;
    uint8_t buf[64]; // max 32 + 32
    std::memcpy(buf, internal_key_x.data(), 32);
    if (merkle_root != nullptr && merkle_root_len > 0) {
        std::memcpy(buf + 32, merkle_root, merkle_root_len);
    }

    return tagged_hash("TapTweak", buf, total);
}

// ── TapLeaf Hash ─────────────────────────────────────────────────────────────

std::array<uint8_t, 32> taproot_leaf_hash(
    const uint8_t* script, std::size_t script_len,
    uint8_t leaf_version) {

    // H_TapLeaf(leaf_version || compact_size(script_len) || script)
    // For compact_size: values < 253 are 1 byte, 253-65535 use 3 bytes, etc.
    // We support scripts up to 64KB for practical use.

    // Pre-tagged hash for "TapLeaf"
    auto tag_hash = SHA256::hash("TapLeaf", 7);

    SHA256 ctx;
    ctx.update(tag_hash.data(), 32);  // tag prefix
    ctx.update(tag_hash.data(), 32);  // tag prefix (twice)
    ctx.update(&leaf_version, 1);

    // CompactSize encoding
    if (script_len < 253) {
        uint8_t len_byte = static_cast<uint8_t>(script_len);
        ctx.update(&len_byte, 1);
    } else if (script_len <= 0xFFFF) {
        uint8_t prefix = 0xFD;
        ctx.update(&prefix, 1);
        uint8_t len_le[2] = {
            static_cast<uint8_t>(script_len & 0xFF),
            static_cast<uint8_t>((script_len >> 8) & 0xFF)
        };
        ctx.update(len_le, 2);
    } else {
        uint8_t prefix = 0xFE;
        ctx.update(&prefix, 1);
        uint8_t len_le[4] = {
            static_cast<uint8_t>(script_len & 0xFF),
            static_cast<uint8_t>((script_len >> 8) & 0xFF),
            static_cast<uint8_t>((script_len >> 16) & 0xFF),
            static_cast<uint8_t>((script_len >> 24) & 0xFF)
        };
        ctx.update(len_le, 4);
    }

    ctx.update(script, script_len);
    return ctx.finalize();
}

// ── TapBranch Hash ───────────────────────────────────────────────────────────

std::array<uint8_t, 32> taproot_branch_hash(
    const std::array<uint8_t, 32>& a,
    const std::array<uint8_t, 32>& b) {

    // Sort lexicographically: smaller first
    const auto* first = a.data();
    const auto* second = b.data();
    if (std::memcmp(a.data(), b.data(), 32) > 0) {
        first = b.data();
        second = a.data();
    }

    uint8_t buf[64];
    std::memcpy(buf, first, 32);
    std::memcpy(buf + 32, second, 32);

    return tagged_hash("TapBranch", buf, 64);
}

// ── Output Key Derivation ────────────────────────────────────────────────────

// Helper: lift x-only key to point with even y
static std::pair<Point, bool> lift_x_even(const std::array<uint8_t, 32>& x_bytes) {
    auto px_fe = FieldElement::from_bytes(x_bytes);

    // y² = x³ + 7
    auto x3 = px_fe.square() * px_fe;
    auto seven = FieldElement::from_uint64(7);
    auto y2 = x3 + seven;

    // sqrt: y = y2^((p+1)/4)
    auto exp = FieldElement::from_hex(
        "3fffffffffffffffffffffffffffffffffffffffffffffffffffffffbfffff0c");

    auto y = FieldElement::one();
    auto base = y2;
    auto exp_bytes = exp.to_bytes();

    for (int i = 0; i < 256; ++i) {
        y = y.square();
        int byte_idx = i / 8;
        int bit_idx = 7 - (i % 8);
        if ((exp_bytes[byte_idx] >> bit_idx) & 1) {
            y = y * base;
        }
    }

    // Verify sqrt
    auto y_check = y.square();
    if (y_check != y2) return {Point::infinity(), false};

    // Force even y (BIP-341 convention)
    auto y_bytes = y.to_bytes();
    if (y_bytes[31] & 1) {
        y = FieldElement::zero() - y;
    }

    return {Point::from_affine(px_fe, y), true};
}

std::pair<std::array<uint8_t, 32>, int> taproot_output_key(
    const std::array<uint8_t, 32>& internal_key_x,
    const uint8_t* merkle_root,
    std::size_t merkle_root_len) {

    // P = lift_x(internal_key_x) — with even y
    auto [P, valid] = lift_x_even(internal_key_x);
    if (!valid) return {{}, 0};

    // t = H_TapTweak(internal_key_x || merkle_root)
    auto t_bytes = taproot_tweak_hash(internal_key_x, merkle_root, merkle_root_len);
    auto t = Scalar::from_bytes(t_bytes);
    if (t.is_zero()) return {{}, 0};

    // Q = P + t*G
    auto tG = Point::generator().scalar_mul(t);
    auto Q = P.add(tG);
    if (Q.is_infinity()) return {{}, 0};

    // Output x-only key
    auto q_x = Q.x().to_bytes();

    // Parity: check if Q.y is odd
    auto Q_uncomp = Q.to_uncompressed();
    int parity = (Q_uncomp[64] & 1) != 0 ? 1 : 0;

    return {q_x, parity};
}

// ── Private Key Tweaking ─────────────────────────────────────────────────────

Scalar taproot_tweak_privkey(
    const Scalar& private_key,
    const uint8_t* merkle_root,
    std::size_t merkle_root_len) {

    if (private_key.is_zero()) return Scalar::zero();

    // P = d * G
    auto P = Point::generator().scalar_mul(private_key);
    auto P_uncomp = P.to_uncompressed();
    bool p_y_odd = (P_uncomp[64] & 1) != 0;

    // If P has odd y, negate d
    auto d = p_y_odd ? private_key.negate() : private_key;

    // t = H_TapTweak(P.x || merkle_root)
    auto px = P.x().to_bytes();
    auto t_bytes = taproot_tweak_hash(px, merkle_root, merkle_root_len);
    auto t = Scalar::from_bytes(t_bytes);

    // Tweaked private key = d + t
    auto tweaked = d + t;
    if (tweaked.is_zero()) return Scalar::zero();

    return tweaked;
}

// ── Taproot Commitment Verification ──────────────────────────────────────────

bool taproot_verify_commitment(
    const std::array<uint8_t, 32>& output_key_x,
    int output_key_parity,
    const std::array<uint8_t, 32>& internal_key_x,
    const uint8_t* merkle_root,
    std::size_t merkle_root_len) {

    // Derive expected output key
    auto [expected_x, expected_parity] = taproot_output_key(
        internal_key_x, merkle_root, merkle_root_len);

    // Compare
    return (expected_x == output_key_x) &&
           (expected_parity == output_key_parity);
}

// ── Merkle Root from Proof ───────────────────────────────────────────────────

std::array<uint8_t, 32> taproot_merkle_root_from_proof(
    const std::array<uint8_t, 32>& leaf_hash,
    const std::vector<std::array<uint8_t, 32>>& proof) {

    auto current = leaf_hash;
    for (const auto& sibling : proof) {
        current = taproot_branch_hash(current, sibling);
    }
    return current;
}

// ── Merkle Root from Leaf List ───────────────────────────────────────────────

std::array<uint8_t, 32> taproot_merkle_root(
    const std::vector<std::array<uint8_t, 32>>& leaf_hashes) {

    if (leaf_hashes.empty()) return {};
    if (leaf_hashes.size() == 1) return leaf_hashes[0];

    // Build tree bottom-up
    std::vector<std::array<uint8_t, 32>> level = leaf_hashes;

    while (level.size() > 1) {
        std::vector<std::array<uint8_t, 32>> next_level;
        for (std::size_t i = 0; i < level.size(); i += 2) {
            if (i + 1 < level.size()) {
                next_level.push_back(taproot_branch_hash(level[i], level[i + 1]));
            } else {
                // Odd leaf — promote to next level
                next_level.push_back(level[i]);
            }
        }
        level = std::move(next_level);
    }

    return level[0];
}

} // namespace secp256k1
