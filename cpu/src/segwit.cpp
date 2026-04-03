// ============================================================================
// BIP-141: Segregated Witness — Witness Program Operations
// ============================================================================

#include "secp256k1/segwit.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/address.hpp"
#include <cstring>

namespace secp256k1 {

// -- Hash160 (local) ----------------------------------------------------------

static std::array<std::uint8_t, 20> local_hash160(
    const std::uint8_t* data, std::size_t len) noexcept {
    // Minimal RIPEMD160 — we import from address module
    // Actually, address.hpp already has hash160. For self-containment,
    // use SHA256 + inline RIPEMD160 constants.
    // But since the library has hash160 in address.cpp, we use SHA256 only
    // for witness_script_hash and do hash160 externally.
    // For validate_p2wpkh_witness, we need it, so include the RIPEMD160 path.

    return hash160(data, len);
}

// -- scriptPubKey construction ------------------------------------------------

std::array<std::uint8_t, 22> segwit_scriptpubkey_p2wpkh(
    const std::uint8_t pubkey_hash[20]) noexcept {
    std::array<std::uint8_t, 22> spk{};
    spk[0] = 0x00; // OP_0
    spk[1] = 0x14; // Push 20 bytes
    std::memcpy(spk.data() + 2, pubkey_hash, 20);
    return spk;
}

std::array<std::uint8_t, 34> segwit_scriptpubkey_p2wsh(
    const std::uint8_t script_hash[32]) noexcept {
    std::array<std::uint8_t, 34> spk{};
    spk[0] = 0x00; // OP_0
    spk[1] = 0x20; // Push 32 bytes
    std::memcpy(spk.data() + 2, script_hash, 32);
    return spk;
}

std::array<std::uint8_t, 34> segwit_scriptpubkey_p2tr(
    const std::uint8_t output_key[32]) noexcept {
    std::array<std::uint8_t, 34> spk{};
    spk[0] = 0x51; // OP_1
    spk[1] = 0x20; // Push 32 bytes
    std::memcpy(spk.data() + 2, output_key, 32);
    return spk;
}

std::vector<std::uint8_t> segwit_scriptpubkey(
    std::uint8_t version,
    const std::uint8_t* program,
    std::size_t program_len) noexcept {
    std::vector<std::uint8_t> spk;
    if (program_len < 2 || program_len > 40 || version > 16) return spk;

    spk.reserve(2 + program_len);
    // OP_0 = 0x00, OP_1..OP_16 = 0x51..0x60
    spk.push_back(version == 0 ? 0x00 : static_cast<std::uint8_t>(0x50 + version));
    spk.push_back(static_cast<std::uint8_t>(program_len));
    spk.insert(spk.end(), program, program + program_len);
    return spk;
}

// -- Witness program detection ------------------------------------------------

bool is_witness_program(
    const std::uint8_t* script, std::size_t script_len) noexcept {
    // BIP-141: scriptPubKey must be:
    //   - 4-42 bytes total
    //   - First byte: OP_0 (0x00) or OP_1..OP_16 (0x51..0x60)
    //   - Second byte: push length == remaining bytes (2-40)
    if (script_len < 4 || script_len > 42) return false;

    std::uint8_t const opcode = script[0];
    if (opcode != 0x00 && (opcode < 0x51 || opcode > 0x60)) return false;

    std::uint8_t const push_len = script[1];
    if (push_len < 2 || push_len > 40) return false;

    return (push_len + 2 == static_cast<std::uint8_t>(script_len));
}

WitnessProgram parse_witness_program(
    const std::uint8_t* script, std::size_t script_len) noexcept {
    WitnessProgram wp;
    wp.version = -1;
    wp.type = WitnessProgramType::NONE;

    if (!is_witness_program(script, script_len)) return wp;

    std::uint8_t const opcode = script[0];
    wp.version = (opcode == 0x00) ? 0 : (opcode - 0x50);

    std::size_t const plen = script[1];
    wp.program.assign(script + 2, script + 2 + plen);

    // Classify
    if (wp.version == 0 && plen == 20) {
        wp.type = WitnessProgramType::P2WPKH;
    } else if (wp.version == 0 && plen == 32) {
        wp.type = WitnessProgramType::P2WSH;
    } else if (wp.version == 1 && plen == 32) {
        wp.type = WitnessProgramType::P2TR;
    } else {
        wp.type = WitnessProgramType::UNKNOWN;
    }

    return wp;
}

// -- Witness script hash (SHA256, single, for P2WSH) --------------------------

std::array<std::uint8_t, 32> witness_script_hash(
    const std::uint8_t* script, std::size_t script_len) noexcept {
    return SHA256::hash(script, script_len);
}

// -- P2WPKH scriptCode -------------------------------------------------------

std::array<std::uint8_t, 25> p2wpkh_script_code(
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

// -- Witness validation -------------------------------------------------------

bool validate_p2wpkh_witness(
    const std::vector<std::vector<std::uint8_t>>& witness,
    const std::uint8_t program[20]) noexcept {
    // BIP-141 §4.1: witness must be [<sig>, <pubkey>]
    if (witness.size() != 2) return false;

    auto const& pubkey = witness[1];
    // Must be compressed pubkey (33 bytes, starting with 0x02 or 0x03)
    if (pubkey.size() != 33) return false;
    if (pubkey[0] != 0x02 && pubkey[0] != 0x03) return false;

    // hash160(pubkey) must match program
    auto h = local_hash160(pubkey.data(), pubkey.size());
    return std::memcmp(h.data(), program, 20) == 0;
}

bool validate_p2wsh_witness(
    const std::vector<std::vector<std::uint8_t>>& witness,
    const std::uint8_t program[32]) noexcept {
    // BIP-141 §4.2: last witness item is the witness script
    if (witness.empty()) return false;

    auto const& witness_script = witness.back();
    if (witness_script.empty()) return false;

    // SHA256(witnessScript) must match program
    auto h = SHA256::hash(witness_script.data(), witness_script.size());
    return std::memcmp(h.data(), program, 32) == 0;
}

// -- Witness weight -----------------------------------------------------------

std::size_t witness_weight(
    const std::vector<std::vector<std::uint8_t>>& witness) noexcept {
    // CompactSize(item_count) + sum of (CompactSize(item_len) + item_data)
    std::size_t w = 0;

    // Item count (compactSize encoding)
    std::size_t n = witness.size();
    if (n < 253) w += 1;
    else if (n <= 0xFFFF) w += 3;
    else w += 5;

    for (auto const& item : witness) {
        std::size_t ilen = item.size();
        if (ilen < 253) w += 1;
        else if (ilen <= 0xFFFF) w += 3;
        else w += 5;
        w += ilen;
    }
    return w;
}

} // namespace secp256k1
