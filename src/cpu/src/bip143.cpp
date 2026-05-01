// ============================================================================
// BIP-143: Transaction Signature Verification for Version 0 Witness Program
// ============================================================================

#include "secp256k1/bip143.hpp"
#include "secp256k1/sha256.hpp"
#include <cstring>

namespace secp256k1 {

// -- LE serialization helpers (local) -----------------------------------------

static inline void put_le32(std::uint8_t* dst, std::uint32_t v) noexcept {
    dst[0] = static_cast<std::uint8_t>(v);
    dst[1] = static_cast<std::uint8_t>(v >> 8);
    dst[2] = static_cast<std::uint8_t>(v >> 16);
    dst[3] = static_cast<std::uint8_t>(v >> 24);
}

static inline void put_le64(std::uint8_t* dst, std::uint64_t v) noexcept {
    dst[0] = static_cast<std::uint8_t>(v);
    dst[1] = static_cast<std::uint8_t>(v >> 8);
    dst[2] = static_cast<std::uint8_t>(v >> 16);
    dst[3] = static_cast<std::uint8_t>(v >> 24);
    dst[4] = static_cast<std::uint8_t>(v >> 32);
    dst[5] = static_cast<std::uint8_t>(v >> 40);
    dst[6] = static_cast<std::uint8_t>(v >> 48);
    dst[7] = static_cast<std::uint8_t>(v >> 56);
}

// -- hashPrevouts -------------------------------------------------------------

std::array<std::uint8_t, 32> bip143_hash_prevouts(
    const Outpoint* outpoints, std::size_t count) noexcept {

    SHA256 ctx;
    for (std::size_t i = 0; i < count; ++i) {
        ctx.update(outpoints[i].txid.data(), 32);
        std::uint8_t vout_le[4];
        put_le32(vout_le, outpoints[i].vout);
        ctx.update(vout_le, 4);
    }
    auto h1 = ctx.finalize();
    return SHA256::hash(h1.data(), 32);
}

// -- hashSequence -------------------------------------------------------------

std::array<std::uint8_t, 32> bip143_hash_sequence(
    const std::uint32_t* sequences, std::size_t count) noexcept {

    SHA256 ctx;
    for (std::size_t i = 0; i < count; ++i) {
        std::uint8_t seq_le[4];
        put_le32(seq_le, sequences[i]);
        ctx.update(seq_le, 4);
    }
    auto h1 = ctx.finalize();
    return SHA256::hash(h1.data(), 32);
}

// -- hashOutputs --------------------------------------------------------------

std::array<std::uint8_t, 32> bip143_hash_outputs(
    const TxOutput* outputs, std::size_t count) noexcept {

    SHA256 ctx;
    for (std::size_t i = 0; i < count; ++i) {
        std::uint8_t val_le[8];
        put_le64(val_le, outputs[i].value);
        ctx.update(val_le, 8);

        // scriptPubKey with compactSize length prefix
        std::size_t slen = outputs[i].script_pubkey.size();
        if (slen < 253) {
            auto len_byte = static_cast<std::uint8_t>(slen);
            ctx.update(&len_byte, 1);
        } else if (slen <= 0xFFFF) {
            std::uint8_t prefix = 0xFD;
            ctx.update(&prefix, 1);
            std::uint8_t len_le[2] = {
                static_cast<std::uint8_t>(slen & 0xFF),
                static_cast<std::uint8_t>((slen >> 8) & 0xFF)
            };
            ctx.update(len_le, 2);
        } else {
            std::uint8_t prefix = 0xFE;
            ctx.update(&prefix, 1);
            std::uint8_t len_le[4];
            put_le32(len_le, static_cast<std::uint32_t>(slen));
            ctx.update(len_le, 4);
        }

        ctx.update(outputs[i].script_pubkey.data(), slen);
    }
    auto h1 = ctx.finalize();
    return SHA256::hash(h1.data(), 32);
}

// -- Build preimage -----------------------------------------------------------

Bip143Preimage bip143_build_preimage(
    std::uint32_t version,
    const Outpoint* outpoints, std::size_t input_count,
    const std::uint32_t* sequences,
    const TxOutput* outputs, std::size_t output_count,
    std::uint32_t locktime) noexcept {

    Bip143Preimage p{};
    p.version = version;
    p.hash_prevouts = bip143_hash_prevouts(outpoints, input_count);
    p.hash_sequence = bip143_hash_sequence(sequences, input_count);
    p.hash_outputs  = bip143_hash_outputs(outputs, output_count);
    p.locktime = locktime;
    return p;
}

// -- BIP-143 Sighash ---------------------------------------------------------

std::array<std::uint8_t, 32> bip143_sighash(
    const Bip143Preimage& preimage,
    const Outpoint& outpoint,
    const std::uint8_t* script_code, std::size_t script_code_len,
    std::uint64_t value,
    std::uint32_t sequence,
    std::uint32_t sighash_type) noexcept {

    constexpr std::uint32_t SIGHASH_NONE         = 0x02;
    constexpr std::uint32_t SIGHASH_SINGLE       = 0x03;
    constexpr std::uint32_t SIGHASH_ANYONECANPAY = 0x80;

    std::uint32_t const base_type = sighash_type & 0x1F;
    bool const anyone = (sighash_type & SIGHASH_ANYONECANPAY) != 0;

    // Start building the preimage for double-SHA256
    SHA256 ctx;

    // 1. nVersion (4 bytes LE)
    std::uint8_t ver_le[4];
    put_le32(ver_le, preimage.version);
    ctx.update(ver_le, 4);

    // 2. hashPrevouts (32 bytes) -- zeroed for ANYONECANPAY
    if (!anyone) {
        ctx.update(preimage.hash_prevouts.data(), 32);
    } else {
        std::uint8_t zeros[32]{};
        ctx.update(zeros, 32);
    }

    // 3. hashSequence (32 bytes) -- zeroed for ANYONECANPAY, NONE, SINGLE
    if (!anyone && base_type != SIGHASH_NONE && base_type != SIGHASH_SINGLE) {
        ctx.update(preimage.hash_sequence.data(), 32);
    } else {
        std::uint8_t zeros[32]{};
        ctx.update(zeros, 32);
    }

    // 4. outpoint (32-byte txid + 4-byte vout LE)
    ctx.update(outpoint.txid.data(), 32);
    std::uint8_t vout_le[4];
    put_le32(vout_le, outpoint.vout);
    ctx.update(vout_le, 4);

    // 5. scriptCode (with compactSize length prefix)
    if (script_code_len < 253) {
        auto len_byte = static_cast<std::uint8_t>(script_code_len);
        ctx.update(&len_byte, 1);
    } else if (script_code_len <= 0xFFFF) {
        std::uint8_t prefix = 0xFD;
        ctx.update(&prefix, 1);
        std::uint8_t len_le[2] = {
            static_cast<std::uint8_t>(script_code_len & 0xFF),
            static_cast<std::uint8_t>((script_code_len >> 8) & 0xFF)
        };
        ctx.update(len_le, 2);
    } else {
        std::uint8_t prefix = 0xFE;
        ctx.update(&prefix, 1);
        std::uint8_t len_le[4];
        put_le32(len_le, static_cast<std::uint32_t>(script_code_len));
        ctx.update(len_le, 4);
    }
    ctx.update(script_code, script_code_len);

    // 6. value (8 bytes LE)
    std::uint8_t val_le[8];
    put_le64(val_le, value);
    ctx.update(val_le, 8);

    // 7. nSequence (4 bytes LE)
    std::uint8_t seq_le[4];
    put_le32(seq_le, sequence);
    ctx.update(seq_le, 4);

    // 8. hashOutputs (32 bytes)
    // NONE: zeroed
    // SINGLE: hash of output at same index (we use preimage.hash_outputs
    //         which caller should set accordingly), or zeros if no matching output.
    // ALL: full hashOutputs
    if (base_type != SIGHASH_NONE) {
        // For SINGLE, caller is expected to set hash_outputs to the hash of
        // the corresponding output. If no such output, zeros are used.
        ctx.update(preimage.hash_outputs.data(), 32);
    } else {
        std::uint8_t zeros[32]{};
        ctx.update(zeros, 32);
    }

    // 9. nLocktime (4 bytes LE)
    std::uint8_t lt_le[4];
    put_le32(lt_le, preimage.locktime);
    ctx.update(lt_le, 4);

    // 10. nHashType (4 bytes LE)
    std::uint8_t ht_le[4];
    put_le32(ht_le, sighash_type);
    ctx.update(ht_le, 4);

    // Double SHA256
    auto h1 = ctx.finalize();
    return SHA256::hash(h1.data(), 32);
}

// -- P2WPKH scriptCode -------------------------------------------------------

std::array<std::uint8_t, 25> bip143_p2wpkh_script_code(
    const std::uint8_t pubkey_hash[20]) noexcept {

    // OP_DUP OP_HASH160 <20 bytes> OP_EQUALVERIFY OP_CHECKSIG
    std::array<std::uint8_t, 25> sc{};
    sc[0] = 0x76; // OP_DUP
    sc[1] = 0xA9; // OP_HASH160
    sc[2] = 0x14; // Push 20 bytes
    std::memcpy(sc.data() + 3, pubkey_hash, 20);
    sc[23] = 0x88; // OP_EQUALVERIFY
    sc[24] = 0xAC; // OP_CHECKSIG
    return sc;
}

} // namespace secp256k1
