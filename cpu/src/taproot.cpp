#include "secp256k1/taproot.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/ct/scalar.hpp"
#include "secp256k1/detail/secure_erase.hpp"
#include <cstring>
#include <algorithm>

namespace secp256k1 {

using fast::Scalar;
using fast::Point;
using fast::FieldElement;

// -- TapTweak Hash ------------------------------------------------------------

std::array<uint8_t, 32> taproot_tweak_hash(
    const std::array<uint8_t, 32>& internal_key_x,
    const uint8_t* merkle_root,
    std::size_t merkle_root_len) {

    // merkle_root is either absent (0) or exactly 32 bytes
    if (merkle_root_len > 32) return {};

    // Concatenate: internal_key_x [|| merkle_root]
    std::size_t const total = 32 + merkle_root_len;
    uint8_t buf[64]; // max 32 + 32
    std::memcpy(buf, internal_key_x.data(), 32);
    if (merkle_root != nullptr && merkle_root_len > 0) {
        std::memcpy(buf + 32, merkle_root, merkle_root_len);
    }

    return tagged_hash("TapTweak", buf, total);
}

// -- TapLeaf Hash -------------------------------------------------------------

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
        auto len_byte = static_cast<uint8_t>(script_len);
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

// -- TapBranch Hash -----------------------------------------------------------

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

// -- Output Key Derivation ----------------------------------------------------

// Helper: lift x-only key to point with even y
static std::pair<Point, bool> lift_x_even(const std::array<uint8_t, 32>& x_bytes) {
    // Strict: reject x >= p
    FieldElement px_fe;
    if (!FieldElement::parse_bytes_strict(x_bytes, px_fe))
        return {Point::infinity(), false};

    // y^2 = x^3 + 7
    auto x3 = px_fe.square() * px_fe;
    auto y2 = x3 + FieldElement::from_uint64(7);

    // Optimized sqrt via addition chain
    auto y = y2.sqrt();

    // Verify sqrt
    if (y.square() != y2) return {Point::infinity(), false};

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

    // P = lift_x(internal_key_x) -- with even y
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
    int const parity = (Q_uncomp[64] & 1) != 0 ? 1 : 0;

    return {q_x, parity};
}

// -- Private Key Tweaking -----------------------------------------------------

Scalar taproot_tweak_privkey(
    const Scalar& private_key,
    const uint8_t* merkle_root,
    std::size_t merkle_root_len) {

    if (private_key.is_zero()) return Scalar::zero();

    // P = d * G (CT)
    auto P = ct::generator_mul(private_key);
    auto [px_bytes, p_y_odd] = P.x_bytes_and_parity();

    // If P has odd y, negate d (CT branchless)
    std::uint64_t const neg_mask = static_cast<std::uint64_t>(p_y_odd)
                                 * UINT64_C(0xFFFFFFFFFFFFFFFF);
    auto d = ct::scalar_cneg(private_key, neg_mask);

    // t = H_TapTweak(P.x || merkle_root)
    auto px = P.x().to_bytes();
    auto t_bytes = taproot_tweak_hash(px, merkle_root, merkle_root_len);
    auto t = Scalar::from_bytes(t_bytes);

    // Tweaked private key = d + t
    auto tweaked = d + t;

    detail::secure_erase(&d, sizeof(d));

    if (tweaked.is_zero()) return Scalar::zero();

    return tweaked;
}

// -- Taproot Commitment Verification ------------------------------------------

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

// -- Merkle Root from Proof ---------------------------------------------------

std::array<uint8_t, 32> taproot_merkle_root_from_proof(
    const std::array<uint8_t, 32>& leaf_hash,
    const std::vector<std::array<uint8_t, 32>>& proof) {

    auto current = leaf_hash;
    for (const auto& sibling : proof) {
        current = taproot_branch_hash(current, sibling);
    }
    return current;
}

// -- Merkle Root from Leaf List -----------------------------------------------

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
                // Odd leaf -- promote to next level
                next_level.push_back(level[i]);
            }
        }
        level = std::move(next_level);
    }

    return level[0];
}

// ============================================================================
// BIP-342: Tapscript Sighash (+ BIP-341 Key-Path Sighash)
// ============================================================================

// Internal: LE serialization helpers
static inline void write_le32(uint8_t* dst, uint32_t v) noexcept {
    dst[0] = static_cast<uint8_t>(v);
    dst[1] = static_cast<uint8_t>(v >> 8);
    dst[2] = static_cast<uint8_t>(v >> 16);
    dst[3] = static_cast<uint8_t>(v >> 24);
}

static inline void write_le64(uint8_t* dst, uint64_t v) noexcept {
    dst[0] = static_cast<uint8_t>(v);
    dst[1] = static_cast<uint8_t>(v >> 8);
    dst[2] = static_cast<uint8_t>(v >> 16);
    dst[3] = static_cast<uint8_t>(v >> 24);
    dst[4] = static_cast<uint8_t>(v >> 32);
    dst[5] = static_cast<uint8_t>(v >> 40);
    dst[6] = static_cast<uint8_t>(v >> 48);
    dst[7] = static_cast<uint8_t>(v >> 56);
}

// Internal: write compactSize to SHA256 context
static void sha_compact_size(SHA256& ctx, uint64_t n) {
    if (n < 253) {
        auto b = static_cast<uint8_t>(n);
        ctx.update(&b, 1);
    } else if (n <= 0xFFFF) {
        uint8_t buf[3];
        buf[0] = 0xFD;
        buf[1] = static_cast<uint8_t>(n & 0xFF);
        buf[2] = static_cast<uint8_t>((n >> 8) & 0xFF);
        ctx.update(buf, 3);
    } else if (n <= 0xFFFFFFFF) {
        uint8_t buf[5];
        buf[0] = 0xFE;
        write_le32(buf + 1, static_cast<uint32_t>(n));
        ctx.update(buf, 5);
    } else {
        uint8_t buf[9];
        buf[0] = 0xFF;
        write_le64(buf + 1, n);
        ctx.update(buf, 9);
    }
}

// Internal: compute sha_prevouts for BIP-341
static std::array<uint8_t, 32> tap_sha_prevouts(const TapSighashTxData& tx) {
    SHA256 ctx;
    for (std::size_t i = 0; i < tx.input_count; ++i) {
        ctx.update(tx.prevout_txids[i].data(), 32);
        uint8_t vout_le[4];
        write_le32(vout_le, tx.prevout_vouts[i]);
        ctx.update(vout_le, 4);
    }
    return ctx.finalize();
}

// Internal: compute sha_amounts for BIP-341
static std::array<uint8_t, 32> tap_sha_amounts(const TapSighashTxData& tx) {
    SHA256 ctx;
    for (std::size_t i = 0; i < tx.input_count; ++i) {
        uint8_t val_le[8];
        write_le64(val_le, tx.input_amounts[i]);
        ctx.update(val_le, 8);
    }
    return ctx.finalize();
}

// Internal: compute sha_scriptpubkeys for BIP-341
static std::array<uint8_t, 32> tap_sha_scriptpubkeys(const TapSighashTxData& tx) {
    SHA256 ctx;
    for (std::size_t i = 0; i < tx.input_count; ++i) {
        sha_compact_size(ctx, tx.input_scriptpubkey_lens[i]);
        ctx.update(tx.input_scriptpubkeys[i], tx.input_scriptpubkey_lens[i]);
    }
    return ctx.finalize();
}

// Internal: compute sha_sequences for BIP-341
static std::array<uint8_t, 32> tap_sha_sequences(const TapSighashTxData& tx) {
    SHA256 ctx;
    for (std::size_t i = 0; i < tx.input_count; ++i) {
        uint8_t seq_le[4];
        write_le32(seq_le, tx.input_sequences[i]);
        ctx.update(seq_le, 4);
    }
    return ctx.finalize();
}

// Internal: compute sha_outputs for BIP-341
static std::array<uint8_t, 32> tap_sha_outputs(const TapSighashTxData& tx) {
    SHA256 ctx;
    for (std::size_t i = 0; i < tx.output_count; ++i) {
        uint8_t val_le[8];
        write_le64(val_le, tx.output_values[i]);
        ctx.update(val_le, 8);
        sha_compact_size(ctx, tx.output_scriptpubkey_lens[i]);
        ctx.update(tx.output_scriptpubkeys[i], tx.output_scriptpubkey_lens[i]);
    }
    return ctx.finalize();
}

// Precomputed SHA256("TapSighash") — avoids recomputing on every sighash call.
// echo -n "TapSighash" | sha256sum → f40a48df...e4a031
static constexpr std::array<uint8_t, 32> TAP_SIGHASH_TAG = {
    0xf4, 0x0a, 0x48, 0xdf, 0x4b, 0x2a, 0x70, 0xc8,
    0xb4, 0x92, 0x4b, 0xf2, 0x65, 0x46, 0x61, 0xed,
    0x3d, 0x95, 0xfd, 0x66, 0xa3, 0x13, 0xeb, 0x87,
    0x23, 0x75, 0x97, 0xc6, 0x28, 0xe4, 0xa0, 0x31
};

// Internal: build BIP-341 common signature message and return tagged hash.
// ext_flag: 0x00 for key path, 0x01 for tapscript
// Extension data is appended by the caller via ext_data/ext_len.
static std::array<uint8_t, 32> tap_sighash_common(
    const TapSighashTxData& tx_data,
    std::size_t input_index,
    uint8_t hash_type,
    uint8_t ext_flag,
    const uint8_t* ext_data, std::size_t ext_len,
    const uint8_t* annex, std::size_t annex_len) noexcept {

    // Bounds check: input_index must be within tx inputs
    if (input_index >= tx_data.input_count) return {};

    uint8_t const output_type = (hash_type == 0x00) ? 0x01 : (hash_type & 0x03);
    bool const anyone = (hash_type & 0x80) != 0;

    // Tagged hash with precomputed SHA256("TapSighash")
    SHA256 ctx;
    ctx.update(TAP_SIGHASH_TAG.data(), 32);
    ctx.update(TAP_SIGHASH_TAG.data(), 32);

    // Epoch (0x00)
    uint8_t epoch = 0x00;
    ctx.update(&epoch, 1);

    // hash_type
    ctx.update(&hash_type, 1);

    // nVersion (LE)
    uint8_t ver_le[4];
    write_le32(ver_le, tx_data.version);
    ctx.update(ver_le, 4);

    // nLockTime (LE)
    uint8_t lt_le[4];
    write_le32(lt_le, tx_data.locktime);
    ctx.update(lt_le, 4);

    // If not ANYONECANPAY: sha_prevouts, sha_amounts, sha_scriptpubkeys, sha_sequences
    if (!anyone) {
        auto hp = tap_sha_prevouts(tx_data);
        ctx.update(hp.data(), 32);
        auto ha = tap_sha_amounts(tx_data);
        ctx.update(ha.data(), 32);
        auto hsp = tap_sha_scriptpubkeys(tx_data);
        ctx.update(hsp.data(), 32);
        auto hs = tap_sha_sequences(tx_data);
        ctx.update(hs.data(), 32);
    }

    // If output_type == ALL (0x01): sha_outputs
    if (output_type == 0x01) {
        auto ho = tap_sha_outputs(tx_data);
        ctx.update(ho.data(), 32);
    }

    // spend_type = (ext_flag * 2) + annex_present
    uint8_t const annex_present = (annex != nullptr && annex_len > 0) ? 1 : 0;
    uint8_t const spend_type = static_cast<uint8_t>(ext_flag * 2 + annex_present);
    ctx.update(&spend_type, 1);

    // If ANYONECANPAY: serialize this input's prevout, amount, scriptPubKey, sequence
    if (anyone) {
        ctx.update(tx_data.prevout_txids[input_index].data(), 32);
        uint8_t vout_le[4];
        write_le32(vout_le, tx_data.prevout_vouts[input_index]);
        ctx.update(vout_le, 4);
        uint8_t amt_le[8];
        write_le64(amt_le, tx_data.input_amounts[input_index]);
        ctx.update(amt_le, 8);
        sha_compact_size(ctx, tx_data.input_scriptpubkey_lens[input_index]);
        ctx.update(tx_data.input_scriptpubkeys[input_index],
                   tx_data.input_scriptpubkey_lens[input_index]);
        uint8_t seq_le[4];
        write_le32(seq_le, tx_data.input_sequences[input_index]);
        ctx.update(seq_le, 4);
    } else {
        // input_index (LE u32)
        uint8_t idx_le[4];
        write_le32(idx_le, static_cast<uint32_t>(input_index));
        ctx.update(idx_le, 4);
    }

    // If annex present: sha_annex = SHA256(compact_size(annex_len) || annex)
    if (annex_present) {
        SHA256 annex_ctx;
        sha_compact_size(annex_ctx, annex_len);
        annex_ctx.update(annex, annex_len);
        auto sha_annex = annex_ctx.finalize();
        ctx.update(sha_annex.data(), 32);
    }

    // If SINGLE: sha_single_output (SHA256 of the output at input_index)
    if (output_type == 0x03) {
        if (input_index < tx_data.output_count) {
            SHA256 out_ctx;
            uint8_t val_le[8];
            write_le64(val_le, tx_data.output_values[input_index]);
            out_ctx.update(val_le, 8);
            sha_compact_size(out_ctx, tx_data.output_scriptpubkey_lens[input_index]);
            out_ctx.update(tx_data.output_scriptpubkeys[input_index],
                           tx_data.output_scriptpubkey_lens[input_index]);
            auto sha_single = out_ctx.finalize();
            ctx.update(sha_single.data(), 32);
        }
        // else: no corresponding output → omit (as per BIP-341)
    }

    // Extension data (tapscript-specific, appended by caller)
    if (ext_data != nullptr && ext_len > 0) {
        ctx.update(ext_data, ext_len);
    }

    return ctx.finalize();
}

// -- BIP-342: Tapscript sighash -----------------------------------------------

std::array<uint8_t, 32> tapscript_sighash(
    const TapSighashTxData& tx_data,
    std::size_t input_index,
    uint8_t hash_type,
    const std::array<uint8_t, 32>& tapleaf_hash,
    uint8_t key_version,
    uint32_t code_separator_pos,
    const uint8_t* annex,
    std::size_t annex_len) noexcept {

    // Build extension data: tapleaf_hash(32) || key_version(1) || codesep_pos(4 LE)
    uint8_t ext[37];
    std::memcpy(ext, tapleaf_hash.data(), 32);
    ext[32] = key_version;
    write_le32(ext + 33, code_separator_pos);

    return tap_sighash_common(tx_data, input_index, hash_type,
                              0x01, ext, 37, annex, annex_len);
}

// -- BIP-341: Key-path sighash ------------------------------------------------

std::array<uint8_t, 32> taproot_keypath_sighash(
    const TapSighashTxData& tx_data,
    std::size_t input_index,
    uint8_t hash_type,
    const uint8_t* annex,
    std::size_t annex_len) noexcept {

    return tap_sighash_common(tx_data, input_index, hash_type,
                              0x00, nullptr, 0, annex, annex_len);
}

} // namespace secp256k1
