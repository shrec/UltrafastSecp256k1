// ============================================================================
// BIP-144: Segregated Witness (Peer Services) — Witness Serialization
// ============================================================================

#include "secp256k1/bip144.hpp"
#include "secp256k1/sha256.hpp"
#include <cstring>

namespace secp256k1 {

// -- LE serialization helpers (local) -----------------------------------------

static inline void ser_le32(std::vector<std::uint8_t>& buf, std::uint32_t v) {
    buf.push_back(static_cast<std::uint8_t>(v));
    buf.push_back(static_cast<std::uint8_t>(v >> 8));
    buf.push_back(static_cast<std::uint8_t>(v >> 16));
    buf.push_back(static_cast<std::uint8_t>(v >> 24));
}

static inline void ser_le64(std::vector<std::uint8_t>& buf, std::uint64_t v) {
    buf.push_back(static_cast<std::uint8_t>(v));
    buf.push_back(static_cast<std::uint8_t>(v >> 8));
    buf.push_back(static_cast<std::uint8_t>(v >> 16));
    buf.push_back(static_cast<std::uint8_t>(v >> 24));
    buf.push_back(static_cast<std::uint8_t>(v >> 32));
    buf.push_back(static_cast<std::uint8_t>(v >> 40));
    buf.push_back(static_cast<std::uint8_t>(v >> 48));
    buf.push_back(static_cast<std::uint8_t>(v >> 56));
}

// CompactSize encoding (Bitcoin varint)
static inline void ser_compact_size(std::vector<std::uint8_t>& buf, std::uint64_t n) {
    if (n < 253) {
        buf.push_back(static_cast<std::uint8_t>(n));
    } else if (n <= 0xFFFF) {
        buf.push_back(0xFD);
        buf.push_back(static_cast<std::uint8_t>(n & 0xFF));
        buf.push_back(static_cast<std::uint8_t>((n >> 8) & 0xFF));
    } else if (n <= 0xFFFFFFFF) {
        buf.push_back(0xFE);
        ser_le32(buf, static_cast<std::uint32_t>(n));
    } else {
        buf.push_back(0xFF);
        ser_le64(buf, n);
    }
}

// Serialize a vector of bytes with compactSize length prefix
static inline void ser_bytes(std::vector<std::uint8_t>& buf,
                             const std::vector<std::uint8_t>& data) {
    ser_compact_size(buf, data.size());
    buf.insert(buf.end(), data.begin(), data.end());
}

// Serialize inputs (shared between legacy and witness formats)
static void ser_inputs(std::vector<std::uint8_t>& buf,
                       const std::vector<TxInput>& inputs) {
    ser_compact_size(buf, inputs.size());
    for (auto const& in : inputs) {
        // prevout: txid(32) + vout(4)
        buf.insert(buf.end(), in.prev_txid.begin(), in.prev_txid.end());
        ser_le32(buf, in.prev_vout);
        // scriptSig
        ser_bytes(buf, in.script_sig);
        // nSequence
        ser_le32(buf, in.sequence);
    }
}

// Serialize outputs
static void ser_outputs(std::vector<std::uint8_t>& buf,
                        const std::vector<TxOut>& outputs) {
    ser_compact_size(buf, outputs.size());
    for (auto const& out : outputs) {
        ser_le64(buf, out.value);
        ser_bytes(buf, out.script_pubkey);
    }
}

// -- Public API ---------------------------------------------------------------

std::vector<std::uint8_t> witness_serialize(const WitnessTx& tx) noexcept {
    std::vector<std::uint8_t> buf;
    buf.reserve(256);

    // nVersion
    ser_le32(buf, tx.version);

    // marker + flag
    buf.push_back(0x00); // marker
    buf.push_back(0x01); // flag

    // inputs
    ser_inputs(buf, tx.inputs);

    // outputs
    ser_outputs(buf, tx.outputs);

    // witness data: one stack per input
    for (std::size_t i = 0; i < tx.inputs.size(); ++i) {
        if (i < tx.witness.size()) {
            auto const& stack = tx.witness[i];
            ser_compact_size(buf, stack.size());
            for (auto const& item : stack) {
                ser_bytes(buf, item);
            }
        } else {
            // Empty witness for this input
            buf.push_back(0x00);
        }
    }

    // nLockTime
    ser_le32(buf, tx.locktime);

    return buf;
}

std::vector<std::uint8_t> legacy_serialize(const WitnessTx& tx) noexcept {
    std::vector<std::uint8_t> buf;
    buf.reserve(256);

    // nVersion
    ser_le32(buf, tx.version);

    // inputs (no marker/flag)
    ser_inputs(buf, tx.inputs);

    // outputs
    ser_outputs(buf, tx.outputs);

    // nLockTime (no witness)
    ser_le32(buf, tx.locktime);

    return buf;
}

std::array<std::uint8_t, 32> compute_txid(const WitnessTx& tx) noexcept {
    auto data = legacy_serialize(tx);
    return SHA256::hash256(data.data(), data.size());
}

std::array<std::uint8_t, 32> compute_wtxid(const WitnessTx& tx) noexcept {
    if (!has_witness(tx)) {
        // No witness → wtxid == txid
        return compute_txid(tx);
    }
    auto data = witness_serialize(tx);
    return SHA256::hash256(data.data(), data.size());
}

std::array<std::uint8_t, 32> witness_commitment(
    const std::array<std::uint8_t, 32>& witness_root,
    const std::array<std::uint8_t, 32>& witness_nonce) noexcept {

    // SHA256(SHA256(witness_root || witness_nonce))
    std::uint8_t buf[64];
    std::memcpy(buf, witness_root.data(), 32);
    std::memcpy(buf + 32, witness_nonce.data(), 32);
    return SHA256::hash256(buf, 64);
}

bool has_witness(const WitnessTx& tx) noexcept {
    for (auto const& stack : tx.witness) {
        if (!stack.empty()) return true;
    }
    return false;
}

std::uint64_t tx_weight(const WitnessTx& tx) noexcept {
    auto legacy_data = legacy_serialize(tx);
    auto witness_data = witness_serialize(tx);
    // weight = base_size * 3 + total_size
    return legacy_data.size() * 3 + witness_data.size();
}

std::uint64_t tx_vsize(const WitnessTx& tx) noexcept {
    // vsize = ceil(weight / 4)
    auto w = tx_weight(tx);
    return (w + 3) / 4;
}

} // namespace secp256k1
