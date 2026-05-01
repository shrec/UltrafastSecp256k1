// ============================================================================
// Test: BIP-143, BIP-144, BIP-141 — SegWit Sighash, Serialization, Programs
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <array>
#include <string>
#include <vector>

#include "secp256k1/bip143.hpp"
#include "secp256k1/bip144.hpp"
#include "secp256k1/segwit.hpp"
#include "secp256k1/sha256.hpp"

using namespace secp256k1;

static int g_pass = 0, g_fail = 0;

static void check(bool cond, const char* name) {
    if (cond) {
        ++g_pass;
    } else {
        ++g_fail;
        (void)std::printf("  FAIL: %s\n", name);
    }
}

// Helper: hex to bytes
static std::array<uint8_t, 32> hex32(const char* hex) {
    std::array<uint8_t, 32> out{};
    for (std::size_t i = 0; i < 32; ++i) {
        char byte_buf[3] = {hex[i * 2], hex[i * 2 + 1], '\0'};
        unsigned long val = std::strtoul(byte_buf, nullptr, 16);
        out[i] = static_cast<uint8_t>(val);
    }
    return out;
}

static std::vector<uint8_t> hex_to_vec(const char* hex) {
    std::vector<uint8_t> out;
    std::size_t len = std::strlen(hex);
    for (std::size_t i = 0; i + 1 < len; i += 2) {
        char byte_buf[3] = {hex[i], hex[i + 1], '\0'};
        unsigned long val = std::strtoul(byte_buf, nullptr, 16);
        out.push_back(static_cast<uint8_t>(val));
    }
    return out;
}

// ===========================================================================
// BIP-143 Tests
// ===========================================================================

static void test_bip143_p2wpkh_script_code() {
    (void)std::printf("[BIP-143] P2WPKH scriptCode construction...\n");

    // Known pubkey hash
    uint8_t pkh[20] = {
        0x14, 0x76, 0xa9, 0x14, 0x79,
        0x09, 0x1b, 0x49, 0xc2, 0x41,
        0x01, 0xa8, 0x63, 0xa7, 0x3c,
        0xd8, 0x46, 0x56, 0xa0, 0xb8
    };

    auto sc = bip143_p2wpkh_script_code(pkh);
    check(sc[0] == 0x76, "scriptCode starts with OP_DUP");
    check(sc[1] == 0xA9, "scriptCode has OP_HASH160");
    check(sc[2] == 0x14, "scriptCode push 20 bytes");
    check(sc[23] == 0x88, "scriptCode has OP_EQUALVERIFY");
    check(sc[24] == 0xAC, "scriptCode ends with OP_CHECKSIG");
    check(std::memcmp(sc.data() + 3, pkh, 20) == 0, "scriptCode contains pubkey hash");
}

static void test_bip143_hash_prevouts() {
    (void)std::printf("[BIP-143] hashPrevouts computation...\n");

    // BIP-143 Example: native P2WPKH (from BIP)
    Outpoint outpoints[2];
    outpoints[0].txid = hex32("0100000000000000000000000000000000000000000000000000000000000000");
    outpoints[0].vout = 0;
    outpoints[1].txid = hex32("0200000000000000000000000000000000000000000000000000000000000000");
    outpoints[1].vout = 1;

    auto hp = bip143_hash_prevouts(outpoints, 2);
    // The result is deterministic: double-SHA256 of the concatenated outpoints
    // Verify it's a valid 32-byte hash (non-zero)
    bool non_zero = false;
    for (auto b : hp) if (b != 0) { non_zero = true; break; }
    check(non_zero, "hashPrevouts is non-zero");
}

static void test_bip143_hash_sequence() {
    (void)std::printf("[BIP-143] hashSequence computation...\n");

    uint32_t seqs[2] = { 0xFFFFFFFF, 0xFFFFFFFE };
    auto hs = bip143_hash_sequence(seqs, 2);

    bool non_zero = false;
    for (auto b : hs) if (b != 0) { non_zero = true; break; }
    check(non_zero, "hashSequence is non-zero");
}

static void test_bip143_hash_outputs() {
    (void)std::printf("[BIP-143] hashOutputs computation...\n");

    TxOutput outputs[1];
    outputs[0].value = 100000;
    outputs[0].script_pubkey = hex_to_vec("76a914000000000000000000000000000000000000000088ac");

    auto ho = bip143_hash_outputs(outputs, 1);
    bool non_zero = false;
    for (auto b : ho) if (b != 0) { non_zero = true; break; }
    check(non_zero, "hashOutputs is non-zero");
}

static void test_bip143_sighash_deterministic() {
    (void)std::printf("[BIP-143] Sighash determinism...\n");

    // Build a simple preimage and verify same inputs → same hash
    Outpoint outpoints[1];
    outpoints[0].txid = hex32("c37af31116d1b27caf68aae9e3ac82f1477929014d5b917657d0eb49478cb670");
    outpoints[0].vout = 1;

    uint32_t seqs[1] = { 0xFFFFFFFE };

    TxOutput outputs[2];
    outputs[0].value = 199996600;
    outputs[0].script_pubkey = hex_to_vec("76a914a457b684d7f0d539a46a45bbc043f35b59d0d96388ac");
    outputs[1].value = 800000000;
    outputs[1].script_pubkey = hex_to_vec("76a914fd270b1ee6abcaea97fea7ad0402e8bd8ad6d77c88ac");

    auto pre = bip143_build_preimage(1, outpoints, 1, seqs, outputs, 2, 0x11000000);

    uint8_t pkh[20] = {
        0xa4, 0x57, 0xb6, 0x84, 0xd7, 0xf0, 0xd5, 0x39, 0xa4, 0x6a,
        0x45, 0xbb, 0xc0, 0x43, 0xf3, 0x5b, 0x59, 0xd0, 0xd9, 0x63
    };
    auto sc = bip143_p2wpkh_script_code(pkh);

    auto h1 = bip143_sighash(pre, outpoints[0], sc.data(), sc.size(),
                              1000000000, 0xFFFFFFFE, 0x01);
    auto h2 = bip143_sighash(pre, outpoints[0], sc.data(), sc.size(),
                              1000000000, 0xFFFFFFFE, 0x01);
    check(h1 == h2, "same inputs produce same sighash");

    // Different sighash type should produce different hash
    auto h3 = bip143_sighash(pre, outpoints[0], sc.data(), sc.size(),
                              1000000000, 0xFFFFFFFE, 0x02);
    check(h1 != h3, "different sighash_type → different digest");
}

static void test_bip143_anyonecanpay() {
    (void)std::printf("[BIP-143] ANYONECANPAY modifier...\n");

    Outpoint outpoints[1];
    outpoints[0].txid = hex32("0000000000000000000000000000000000000000000000000000000000000001");
    outpoints[0].vout = 0;
    uint32_t seqs[1] = { 0xFFFFFFFF };
    TxOutput outputs[1];
    outputs[0].value = 50000;
    outputs[0].script_pubkey = hex_to_vec("76a91400000000000000000000000000000000000000ac");

    auto pre = bip143_build_preimage(2, outpoints, 1, seqs, outputs, 1, 0);
    uint8_t sc[25]{};
    auto h_all = bip143_sighash(pre, outpoints[0], sc, 25, 100000, 0xFFFFFFFF, 0x01);
    auto h_acp = bip143_sighash(pre, outpoints[0], sc, 25, 100000, 0xFFFFFFFF, 0x81);
    check(h_all != h_acp, "ANYONECANPAY produces different digest");
}

// ===========================================================================
// BIP-144 Tests
// ===========================================================================

static void test_bip144_legacy_serialize() {
    (void)std::printf("[BIP-144] Legacy serialization...\n");

    WitnessTx tx;
    tx.version = 2;
    tx.locktime = 0;

    TxInput in;
    in.prev_txid = hex32("0000000000000000000000000000000000000000000000000000000000000001");
    in.prev_vout = 0;
    in.sequence = 0xFFFFFFFF;
    tx.inputs.push_back(in);

    TxOut out;
    out.value = 50000;
    out.script_pubkey = hex_to_vec("76a91400000000000000000000000000000000000000ac");
    tx.outputs.push_back(out);

    auto data = legacy_serialize(tx);
    check(data.size() > 0, "legacy serialization non-empty");

    // Should start with version 2 LE
    check(data[0] == 0x02 && data[1] == 0x00 && data[2] == 0x00 && data[3] == 0x00,
          "starts with version 2 LE");

    // Should NOT have marker/flag
    check(data[4] != 0x00, "no marker byte in legacy format");
}

static void test_bip144_witness_serialize() {
    (void)std::printf("[BIP-144] Witness serialization...\n");

    WitnessTx tx;
    tx.version = 2;
    tx.locktime = 0;

    TxInput in;
    in.prev_txid = hex32("0000000000000000000000000000000000000000000000000000000000000001");
    in.prev_vout = 0;
    in.sequence = 0xFFFFFFFF;
    tx.inputs.push_back(in);

    TxOut out;
    out.value = 50000;
    out.script_pubkey = hex_to_vec("76a91400000000000000000000000000000000000000ac");
    tx.outputs.push_back(out);

    // Add witness: [<sig>, <pubkey>]
    WitnessStack ws;
    ws.push_back(hex_to_vec("3045022100abcdef022100abcdef"));  // mock sig
    ws.push_back(hex_to_vec("02" "0000000000000000000000000000000000000000000000000000000000000001"));
    tx.witness.push_back(ws);

    auto data = witness_serialize(tx);
    check(data.size() > 0, "witness serialization non-empty");
    // marker=0x00, flag=0x01 after version
    check(data[4] == 0x00, "marker byte present");
    check(data[5] == 0x01, "flag byte present");
}

static void test_bip144_txid_vs_wtxid() {
    (void)std::printf("[BIP-144] txid vs wtxid...\n");

    WitnessTx tx;
    tx.version = 2;
    tx.locktime = 0;

    TxInput in;
    in.prev_txid = hex32("0000000000000000000000000000000000000000000000000000000000000001");
    in.prev_vout = 0;
    in.sequence = 0xFFFFFFFF;
    tx.inputs.push_back(in);

    TxOut out;
    out.value = 50000;
    out.script_pubkey = hex_to_vec("001400000000000000000000000000000000000000");
    tx.outputs.push_back(out);

    // Add witness
    WitnessStack ws;
    ws.push_back(hex_to_vec("3045022100abcd"));
    ws.push_back(hex_to_vec("0200000000000000000000000000000000000000000000000000000000000001"));
    tx.witness.push_back(ws);

    auto txid = compute_txid(tx);
    auto wtxid = compute_wtxid(tx);

    // txid should NOT include witness data, so they should differ
    check(txid != wtxid, "txid != wtxid when witness data present");

    // Both should be non-zero
    bool txid_nz = false, wtxid_nz = false;
    for (auto b : txid) if (b) { txid_nz = true; break; }
    for (auto b : wtxid) if (b) { wtxid_nz = true; break; }
    check(txid_nz, "txid is non-zero");
    check(wtxid_nz, "wtxid is non-zero");
}

static void test_bip144_no_witness_txid_eq_wtxid() {
    (void)std::printf("[BIP-144] No witness: txid == wtxid...\n");

    WitnessTx tx;
    tx.version = 1;
    tx.locktime = 0;

    TxInput in;
    in.prev_txid = hex32("0000000000000000000000000000000000000000000000000000000000000001");
    in.prev_vout = 0;
    in.sequence = 0xFFFFFFFF;
    tx.inputs.push_back(in);

    TxOut out;
    out.value = 50000;
    out.script_pubkey = hex_to_vec("76a91400000000000000000000000000000000000000ac");
    tx.outputs.push_back(out);

    // No witness → wtxid should equal txid
    auto txid = compute_txid(tx);
    auto wtxid = compute_wtxid(tx);
    check(txid == wtxid, "no witness: txid == wtxid");
}

static void test_bip144_witness_commitment() {
    (void)std::printf("[BIP-144] Witness commitment...\n");

    std::array<uint8_t, 32> root{}, nonce{};
    root.fill(0xAA);
    nonce.fill(0x00);

    auto c = witness_commitment(root, nonce);
    bool non_zero = false;
    for (auto b : c) if (b) { non_zero = true; break; }
    check(non_zero, "witness commitment non-zero");

    // Deterministic
    auto c2 = witness_commitment(root, nonce);
    check(c == c2, "witness commitment deterministic");
}

static void test_bip144_weight_vsize() {
    (void)std::printf("[BIP-144] tx weight & vsize...\n");

    WitnessTx tx;
    tx.version = 2;
    tx.locktime = 0;

    TxInput in;
    in.prev_txid = hex32("0000000000000000000000000000000000000000000000000000000000000001");
    in.prev_vout = 0;
    in.sequence = 0xFFFFFFFF;
    tx.inputs.push_back(in);

    TxOut out;
    out.value = 50000;
    out.script_pubkey = hex_to_vec("001400000000000000000000000000000000000000");
    tx.outputs.push_back(out);

    WitnessStack ws;
    ws.push_back(hex_to_vec("3045022100abcd"));
    ws.push_back(hex_to_vec("0200000000000000000000000000000000000000000000000000000000000001"));
    tx.witness.push_back(ws);

    auto w = tx_weight(tx);
    auto vs = tx_vsize(tx);

    check(w > 0, "weight > 0");
    check(vs > 0, "vsize > 0");
    check(vs <= w, "vsize <= weight");
    // weight = base * 3 + total, vsize = ceil(weight/4)
    check(vs == (w + 3) / 4, "vsize == ceil(weight/4)");
}

static void test_bip144_has_witness() {
    (void)std::printf("[BIP-144] has_witness detection...\n");

    WitnessTx tx;
    tx.version = 1;
    tx.locktime = 0;
    check(!has_witness(tx), "empty tx has no witness");

    WitnessStack ws;
    ws.push_back({0x01});
    tx.witness.push_back(ws);
    check(has_witness(tx), "tx with witness items detected");
}

// ===========================================================================
// BIP-141 Tests
// ===========================================================================

static void test_segwit_scriptpubkey_p2wpkh() {
    (void)std::printf("[BIP-141] P2WPKH scriptPubKey...\n");

    uint8_t pkh[20];
    std::memset(pkh, 0xAB, 20);
    auto spk = segwit_scriptpubkey_p2wpkh(pkh);
    check(spk[0] == 0x00, "starts with OP_0");
    check(spk[1] == 0x14, "push 20 bytes");
    check(std::memcmp(spk.data() + 2, pkh, 20) == 0, "contains pubkey hash");
}

static void test_segwit_scriptpubkey_p2wsh() {
    (void)std::printf("[BIP-141] P2WSH scriptPubKey...\n");

    uint8_t sh[32];
    std::memset(sh, 0xCD, 32);
    auto spk = segwit_scriptpubkey_p2wsh(sh);
    check(spk[0] == 0x00, "starts with OP_0");
    check(spk[1] == 0x20, "push 32 bytes");
    check(std::memcmp(spk.data() + 2, sh, 32) == 0, "contains script hash");
}

static void test_segwit_scriptpubkey_p2tr() {
    (void)std::printf("[BIP-141] P2TR scriptPubKey...\n");

    uint8_t key[32];
    std::memset(key, 0xEF, 32);
    auto spk = segwit_scriptpubkey_p2tr(key);
    check(spk[0] == 0x51, "starts with OP_1");
    check(spk[1] == 0x20, "push 32 bytes");
    check(std::memcmp(spk.data() + 2, key, 32) == 0, "contains output key");
}

static void test_segwit_is_witness_program() {
    (void)std::printf("[BIP-141] is_witness_program detection...\n");

    // P2WPKH: OP_0 <20 bytes> = 22 bytes
    uint8_t p2wpkh[22] = {0x00, 0x14};
    std::memset(p2wpkh + 2, 0, 20);
    check(is_witness_program(p2wpkh, 22), "P2WPKH is witness program");

    // P2WSH: OP_0 <32 bytes> = 34 bytes
    uint8_t p2wsh[34] = {0x00, 0x20};
    std::memset(p2wsh + 2, 0, 32);
    check(is_witness_program(p2wsh, 34), "P2WSH is witness program");

    // P2TR: OP_1 <32 bytes> = 34 bytes
    uint8_t p2tr[34] = {0x51, 0x20};
    std::memset(p2tr + 2, 0, 32);
    check(is_witness_program(p2tr, 34), "P2TR is witness program");

    // Non-witness: too short
    uint8_t short_script[3] = {0x00, 0x01, 0xFF};
    check(!is_witness_program(short_script, 3), "too short not witness");

    // Non-witness: wrong opcode
    uint8_t bad_opcode[22] = {0x61, 0x14};
    check(!is_witness_program(bad_opcode, 22), "bad opcode not witness");

    // Non-witness: length mismatch
    uint8_t len_mismatch[22] = {0x00, 0x13};
    check(!is_witness_program(len_mismatch, 22), "length mismatch not witness");
}

static void test_segwit_parse_witness_program() {
    (void)std::printf("[BIP-141] parse_witness_program...\n");

    // P2WPKH
    uint8_t p2wpkh[22] = {0x00, 0x14};
    std::memset(p2wpkh + 2, 0xAA, 20);
    auto wp = parse_witness_program(p2wpkh, 22);
    check(wp.version == 0, "P2WPKH version 0");
    check(wp.type == WitnessProgramType::P2WPKH, "classified as P2WPKH");
    check(wp.program.size() == 20, "program is 20 bytes");

    // P2WSH
    uint8_t p2wsh[34] = {0x00, 0x20};
    std::memset(p2wsh + 2, 0xBB, 32);
    wp = parse_witness_program(p2wsh, 34);
    check(wp.version == 0, "P2WSH version 0");
    check(wp.type == WitnessProgramType::P2WSH, "classified as P2WSH");
    check(wp.program.size() == 32, "program is 32 bytes");

    // P2TR
    uint8_t p2tr[34] = {0x51, 0x20};
    std::memset(p2tr + 2, 0xCC, 32);
    wp = parse_witness_program(p2tr, 34);
    check(wp.version == 1, "P2TR version 1");
    check(wp.type == WitnessProgramType::P2TR, "classified as P2TR");

    // Unknown: v2, 32 bytes
    uint8_t v2[34] = {0x52, 0x20};
    std::memset(v2 + 2, 0xDD, 32);
    wp = parse_witness_program(v2, 34);
    check(wp.version == 2, "unknown v2 program");
    check(wp.type == WitnessProgramType::UNKNOWN, "classified as UNKNOWN");

    // Not witness
    uint8_t not_wp[3] = {0x00, 0x01, 0xFF};
    wp = parse_witness_program(not_wp, 3);
    check(wp.version == -1, "not a witness program");
    check(wp.type == WitnessProgramType::NONE, "classified as NONE");
}

static void test_segwit_witness_script_hash() {
    (void)std::printf("[BIP-141] witness_script_hash (SHA256)...\n");

    uint8_t script[] = {0x52, 0x21, 0x02};  // Sample script fragment
    auto h = witness_script_hash(script, 3);

    // SHA256 of 3-byte input is deterministic
    auto expected = SHA256::hash(script, 3);
    check(h == expected, "witness_script_hash == SHA256(script)");
}

static void test_segwit_p2wpkh_script_code() {
    (void)std::printf("[BIP-141] p2wpkh_script_code...\n");

    uint8_t pkh[20];
    std::memset(pkh, 0x42, 20);
    auto sc = p2wpkh_script_code(pkh);
    check(sc[0] == 0x76, "OP_DUP");
    check(sc[1] == 0xA9, "OP_HASH160");
    check(sc[2] == 0x14, "push 20");
    check(sc[23] == 0x88, "OP_EQUALVERIFY");
    check(sc[24] == 0xAC, "OP_CHECKSIG");
}

static void test_segwit_general_scriptpubkey() {
    (void)std::printf("[BIP-141] General segwit_scriptpubkey...\n");

    // v0, 20-byte program
    uint8_t prog[20];
    std::memset(prog, 0xAA, 20);
    auto spk = segwit_scriptpubkey(0, prog, 20);
    check(spk.size() == 22, "v0 20-byte spk is 22 bytes");
    check(spk[0] == 0x00, "v0 starts with OP_0");
    check(spk[1] == 0x14, "push 20 bytes");

    // v1, 32-byte program
    uint8_t prog32[32];
    std::memset(prog32, 0xBB, 32);
    spk = segwit_scriptpubkey(1, prog32, 32);
    check(spk.size() == 34, "v1 32-byte spk is 34 bytes");
    check(spk[0] == 0x51, "v1 starts with OP_1");

    // Invalid: program too short
    spk = segwit_scriptpubkey(0, prog, 1);
    check(spk.empty(), "program too short → empty");

    // Invalid: version too high
    spk = segwit_scriptpubkey(17, prog, 20);
    check(spk.empty(), "version 17 → empty");
}

static void test_segwit_witness_weight() {
    (void)std::printf("[BIP-141] witness_weight computation...\n");

    std::vector<std::vector<uint8_t>> witness;
    witness.push_back(hex_to_vec("3045022100abcd"));  // 7-byte sig
    witness.push_back(hex_to_vec("020000000000000000000000000000000000000000000000000000000000000001"));  // 33-byte pubkey

    auto w = witness_weight(witness);
    // Expected: 1 (item count) + 1 (sig len) + 7 (sig) + 1 (pubkey len) + 33 (pubkey) = 43
    check(w == 43, "witness weight matches expected");
}

static void test_segwit_validate_p2wsh_witness() {
    (void)std::printf("[BIP-141] validate_p2wsh_witness...\n");

    // Create a simple witness script
    std::vector<uint8_t> witness_script = {0x52, 0x21, 0x02, 0x00}; // mock

    // Compute its SHA256 hash
    auto script_hash = SHA256::hash(witness_script.data(), witness_script.size());

    // Valid witness: [<item>, ..., <witnessScript>]
    std::vector<std::vector<uint8_t>> witness;
    witness.push_back({0x01, 0x02}); // dummy stack item
    witness.push_back(witness_script);

    check(validate_p2wsh_witness(witness, script_hash.data()),
          "P2WSH: valid witness matches program");

    // Invalid: wrong script hash
    uint8_t bad_hash[32] = {};
    check(!validate_p2wsh_witness(witness, bad_hash),
          "P2WSH: wrong hash fails validation");
}

// ===========================================================================
// Roundtrip Tests
// ===========================================================================

static void test_roundtrip_scriptpubkey() {
    (void)std::printf("[Roundtrip] Build scriptPubKey -> parse witness program...\n");

    // P2WPKH roundtrip
    uint8_t pkh[20];
    std::memset(pkh, 0x42, 20);
    auto spk = segwit_scriptpubkey_p2wpkh(pkh);
    auto wp = parse_witness_program(spk.data(), spk.size());
    check(wp.version == 0, "roundtrip P2WPKH: version 0");
    check(wp.type == WitnessProgramType::P2WPKH, "roundtrip P2WPKH: type correct");
    check(std::memcmp(wp.program.data(), pkh, 20) == 0, "roundtrip P2WPKH: program matches");

    // P2TR roundtrip
    uint8_t key[32];
    std::memset(key, 0xEF, 32);
    auto spk_tr = segwit_scriptpubkey_p2tr(key);
    wp = parse_witness_program(spk_tr.data(), spk_tr.size());
    check(wp.version == 1, "roundtrip P2TR: version 1");
    check(wp.type == WitnessProgramType::P2TR, "roundtrip P2TR: type correct");
    check(std::memcmp(wp.program.data(), key, 32) == 0, "roundtrip P2TR: program matches");
}

// ===========================================================================
// Main
// ===========================================================================

static int run_all_bip141_143_144_tests() {
    (void)std::printf("=== BIP-143 / BIP-144 / BIP-141 Test Suite ===\n\n");

    // BIP-143
    test_bip143_p2wpkh_script_code();
    test_bip143_hash_prevouts();
    test_bip143_hash_sequence();
    test_bip143_hash_outputs();
    test_bip143_sighash_deterministic();
    test_bip143_anyonecanpay();

    // BIP-144
    test_bip144_legacy_serialize();
    test_bip144_witness_serialize();
    test_bip144_txid_vs_wtxid();
    test_bip144_no_witness_txid_eq_wtxid();
    test_bip144_witness_commitment();
    test_bip144_weight_vsize();
    test_bip144_has_witness();

    // BIP-141
    test_segwit_scriptpubkey_p2wpkh();
    test_segwit_scriptpubkey_p2wsh();
    test_segwit_scriptpubkey_p2tr();
    test_segwit_is_witness_program();
    test_segwit_parse_witness_program();
    test_segwit_witness_script_hash();
    test_segwit_p2wpkh_script_code();
    test_segwit_general_scriptpubkey();
    test_segwit_witness_weight();
    test_segwit_validate_p2wsh_witness();

    // Roundtrip
    test_roundtrip_scriptpubkey();

    (void)std::printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}

int test_bip141_143_144_run() { return run_all_bip141_143_144_tests(); }

#ifdef STANDALONE_TEST
int main() { return run_all_bip141_143_144_tests(); }
#endif
