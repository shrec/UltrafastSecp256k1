// ============================================================================
// Test: BIP-342 — Tapscript Sighash (+ BIP-341 Key-Path Sighash)
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <array>
#include <vector>

#include "secp256k1/taproot.hpp"
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

static std::array<uint8_t, 32> hex32(const char* hex) {
    std::array<uint8_t, 32> out{};
    for (std::size_t i = 0; i < 32; ++i) {
        char byte_buf[3] = {hex[i * 2], hex[i * 2 + 1], '\0'};
        unsigned long val = std::strtoul(byte_buf, nullptr, 16);
        out[i] = static_cast<uint8_t>(val);
    }
    return out;
}

// ===========================================================================
// BIP-341 Key-Path Sighash Tests
// ===========================================================================

static void test_keypath_sighash_deterministic() {
    (void)std::printf("[BIP-341] Key-path sighash determinism...\n");

    auto txid = hex32("c37af31116d1b27caf68aae9e3ac82f1477929014d5b917657d0eb49478cb670");
    uint8_t full_spk[34];
    full_spk[0] = 0x51; full_spk[1] = 0x20;
    std::memset(full_spk + 2, 0xAA, 32);
    uint8_t output_spk[34];
    output_spk[0] = 0x51; output_spk[1] = 0x20;
    std::memset(output_spk + 2, 0xBB, 32);

    uint32_t vout, seq;
    uint64_t amt, oval;
    const uint8_t* ispk = full_spk;
    size_t ispk_len = 34;
    const uint8_t* ospk = output_spk;
    size_t ospk_len = 34;

    TapSighashTxData td{};
    td.version = 2;
    td.locktime = 0;
    td.input_count = 1;
    td.prevout_txids = &txid;
    vout = 0;
    td.prevout_vouts = &vout;
    amt = 100000;
    td.input_amounts = &amt;
    seq = 0xFFFFFFFF;
    td.input_sequences = &seq;
    td.input_scriptpubkeys = &ispk;
    td.input_scriptpubkey_lens = &ispk_len;
    td.output_count = 1;
    oval = 90000;
    td.output_values = &oval;
    td.output_scriptpubkeys = &ospk;
    td.output_scriptpubkey_lens = &ospk_len;

    auto h1 = taproot_keypath_sighash(td, 0, 0x00);
    auto h2 = taproot_keypath_sighash(td, 0, 0x00);
    check(h1 == h2, "same inputs → same key-path sighash");

    // Different hash_type → different hash
    auto h3 = taproot_keypath_sighash(td, 0, 0x01);
    // 0x00 (DEFAULT) and 0x01 (ALL) produce different hashes because
    // hash_type byte itself is part of the preimage
    check(h1 != h3, "DEFAULT (0x00) vs ALL (0x01) differ in preimage");
}

static void test_keypath_sighash_anyonecanpay() {
    (void)std::printf("[BIP-341] Key-path ANYONECANPAY...\n");

    auto txid = hex32("0000000000000000000000000000000000000000000000000000000000000001");
    uint8_t spk[34] = {0x51, 0x20};
    std::memset(spk + 2, 0x11, 32);

    uint32_t vout = 0, seq = 0xFFFFFFFF;
    uint64_t amt = 50000, oval = 40000;
    const uint8_t* ispk = spk;
    size_t ispk_len = 34;
    const uint8_t* ospk = spk;
    size_t ospk_len = 34;

    TapSighashTxData td{};
    td.version = 2; td.locktime = 0;
    td.input_count = 1;
    td.prevout_txids = &txid;
    td.prevout_vouts = &vout;
    td.input_amounts = &amt;
    td.input_sequences = &seq;
    td.input_scriptpubkeys = &ispk;
    td.input_scriptpubkey_lens = &ispk_len;
    td.output_count = 1;
    td.output_values = &oval;
    td.output_scriptpubkeys = &ospk;
    td.output_scriptpubkey_lens = &ospk_len;

    auto h_all = taproot_keypath_sighash(td, 0, 0x01);
    auto h_acp = taproot_keypath_sighash(td, 0, 0x81);
    check(h_all != h_acp, "ALL vs ALL|ANYONECANPAY differ");
}

static void test_keypath_sighash_with_annex() {
    (void)std::printf("[BIP-341] Key-path with annex...\n");

    auto txid = hex32("0000000000000000000000000000000000000000000000000000000000000001");
    uint8_t spk[34] = {0x51, 0x20};
    std::memset(spk + 2, 0x22, 32);

    uint32_t vout = 0, seq = 0xFFFFFFFF;
    uint64_t amt = 50000, oval = 40000;
    const uint8_t* ispk = spk;
    size_t ispk_len = 34;
    const uint8_t* ospk = spk;
    size_t ospk_len = 34;

    TapSighashTxData td{};
    td.version = 2; td.locktime = 0;
    td.input_count = 1;
    td.prevout_txids = &txid;
    td.prevout_vouts = &vout;
    td.input_amounts = &amt;
    td.input_sequences = &seq;
    td.input_scriptpubkeys = &ispk;
    td.input_scriptpubkey_lens = &ispk_len;
    td.output_count = 1;
    td.output_values = &oval;
    td.output_scriptpubkeys = &ospk;
    td.output_scriptpubkey_lens = &ospk_len;

    auto h_no_annex = taproot_keypath_sighash(td, 0, 0x00);
    uint8_t annex[] = {0x50, 0x01, 0x02, 0x03}; // annex starts with 0x50
    auto h_annex = taproot_keypath_sighash(td, 0, 0x00, annex, 4);
    check(h_no_annex != h_annex, "annex changes the sighash");
}

// ===========================================================================
// BIP-342 Tapscript Sighash Tests
// ===========================================================================

static void test_tapscript_sighash_deterministic() {
    (void)std::printf("[BIP-342] Tapscript sighash determinism...\n");

    auto txid = hex32("0000000000000000000000000000000000000000000000000000000000000001");
    uint8_t spk[34] = {0x51, 0x20};
    std::memset(spk + 2, 0x33, 32);

    uint32_t vout = 0, seq = 0xFFFFFFFF;
    uint64_t amt = 100000, oval = 90000;
    const uint8_t* ispk = spk;
    size_t ispk_len = 34;
    const uint8_t* ospk = spk;
    size_t ospk_len = 34;

    TapSighashTxData td{};
    td.version = 2; td.locktime = 0;
    td.input_count = 1;
    td.prevout_txids = &txid;
    td.prevout_vouts = &vout;
    td.input_amounts = &amt;
    td.input_sequences = &seq;
    td.input_scriptpubkeys = &ispk;
    td.input_scriptpubkey_lens = &ispk_len;
    td.output_count = 1;
    td.output_values = &oval;
    td.output_scriptpubkeys = &ospk;
    td.output_scriptpubkey_lens = &ospk_len;

    // Build a tapleaf hash
    uint8_t script[] = {0xAC}; // OP_CHECKSIG
    auto tlh = taproot_leaf_hash(script, 1, 0xC0);

    auto h1 = tapscript_sighash(td, 0, 0x00, tlh, 0x00, 0xFFFFFFFF);
    auto h2 = tapscript_sighash(td, 0, 0x00, tlh, 0x00, 0xFFFFFFFF);
    check(h1 == h2, "tapscript: same inputs → same sighash");
}

static void test_tapscript_vs_keypath() {
    (void)std::printf("[BIP-342] Tapscript vs key-path sighash differ...\n");

    auto txid = hex32("0000000000000000000000000000000000000000000000000000000000000001");
    uint8_t spk[34] = {0x51, 0x20};
    std::memset(spk + 2, 0x44, 32);

    uint32_t vout = 0, seq = 0xFFFFFFFF;
    uint64_t amt = 100000, oval = 90000;
    const uint8_t* ispk = spk;
    size_t ispk_len = 34;
    const uint8_t* ospk = spk;
    size_t ospk_len = 34;

    TapSighashTxData td{};
    td.version = 2; td.locktime = 0;
    td.input_count = 1;
    td.prevout_txids = &txid;
    td.prevout_vouts = &vout;
    td.input_amounts = &amt;
    td.input_sequences = &seq;
    td.input_scriptpubkeys = &ispk;
    td.input_scriptpubkey_lens = &ispk_len;
    td.output_count = 1;
    td.output_values = &oval;
    td.output_scriptpubkeys = &ospk;
    td.output_scriptpubkey_lens = &ospk_len;

    auto kp = taproot_keypath_sighash(td, 0, 0x00);

    uint8_t script[] = {0xAC};
    auto tlh = taproot_leaf_hash(script, 1, 0xC0);
    auto ts = tapscript_sighash(td, 0, 0x00, tlh, 0x00, 0xFFFFFFFF);

    check(kp != ts, "key-path and tapscript sighash differ (ext_flag=0 vs 1)");
}

static void test_tapscript_code_separator() {
    (void)std::printf("[BIP-342] code_separator_pos affects sighash...\n");

    auto txid = hex32("0000000000000000000000000000000000000000000000000000000000000002");
    uint8_t spk[34] = {0x51, 0x20};
    std::memset(spk + 2, 0x55, 32);

    uint32_t vout = 0, seq = 0xFFFFFFFF;
    uint64_t amt = 200000, oval = 190000;
    const uint8_t* ispk = spk;
    size_t ispk_len = 34;
    const uint8_t* ospk = spk;
    size_t ospk_len = 34;

    TapSighashTxData td{};
    td.version = 2; td.locktime = 0;
    td.input_count = 1;
    td.prevout_txids = &txid;
    td.prevout_vouts = &vout;
    td.input_amounts = &amt;
    td.input_sequences = &seq;
    td.input_scriptpubkeys = &ispk;
    td.input_scriptpubkey_lens = &ispk_len;
    td.output_count = 1;
    td.output_values = &oval;
    td.output_scriptpubkeys = &ospk;
    td.output_scriptpubkey_lens = &ospk_len;

    uint8_t script[] = {0xAB, 0xAC}; // OP_CODESEPARATOR OP_CHECKSIG
    auto tlh = taproot_leaf_hash(script, 2, 0xC0);

    auto h1 = tapscript_sighash(td, 0, 0x00, tlh, 0x00, 0xFFFFFFFF);
    auto h2 = tapscript_sighash(td, 0, 0x00, tlh, 0x00, 0);
    check(h1 != h2, "different code_separator_pos → different sighash");
}

static void test_tapscript_leaf_version() {
    (void)std::printf("[BIP-342] Different leaf versions...\n");

    uint8_t script[] = {0xAC};
    auto tlh_c0 = taproot_leaf_hash(script, 1, 0xC0);
    auto tlh_c1 = taproot_leaf_hash(script, 1, 0xC1);

    // Different leaf versions → different tapleaf hashes
    check(tlh_c0 != tlh_c1, "different leaf_version → different tapleaf_hash");
}

static void test_tapscript_sighash_types() {
    (void)std::printf("[BIP-342] Sighash type variations...\n");

    auto txid = hex32("0000000000000000000000000000000000000000000000000000000000000003");
    uint8_t spk[34] = {0x51, 0x20};
    std::memset(spk + 2, 0x66, 32);

    uint32_t vout = 0, seq = 0xFFFFFFFF;
    uint64_t amt = 50000, oval = 40000;
    const uint8_t* ispk = spk;
    size_t ispk_len = 34;
    const uint8_t* ospk = spk;
    size_t ospk_len = 34;

    TapSighashTxData td{};
    td.version = 2; td.locktime = 0;
    td.input_count = 1;
    td.prevout_txids = &txid;
    td.prevout_vouts = &vout;
    td.input_amounts = &amt;
    td.input_sequences = &seq;
    td.input_scriptpubkeys = &ispk;
    td.input_scriptpubkey_lens = &ispk_len;
    td.output_count = 1;
    td.output_values = &oval;
    td.output_scriptpubkeys = &ospk;
    td.output_scriptpubkey_lens = &ospk_len;

    uint8_t script[] = {0xAC};
    auto tlh = taproot_leaf_hash(script, 1, 0xC0);

    auto h_default = tapscript_sighash(td, 0, 0x00, tlh, 0x00, 0xFFFFFFFF);
    auto h_all     = tapscript_sighash(td, 0, 0x01, tlh, 0x00, 0xFFFFFFFF);
    auto h_none    = tapscript_sighash(td, 0, 0x02, tlh, 0x00, 0xFFFFFFFF);
    auto h_single  = tapscript_sighash(td, 0, 0x03, tlh, 0x00, 0xFFFFFFFF);
    auto h_acp     = tapscript_sighash(td, 0, 0x81, tlh, 0x00, 0xFFFFFFFF);

    // All should be different
    check(h_default != h_none, "DEFAULT != NONE");
    check(h_all != h_none, "ALL != NONE");
    check(h_none != h_single, "NONE != SINGLE");
    check(h_all != h_acp, "ALL != ALL|ANYONECANPAY");
}

// ===========================================================================
// Main
// ===========================================================================

static int run_all_bip342_tests() {
    (void)std::printf("=== BIP-341/342 Sighash Test Suite ===\n\n");

    // BIP-341 key-path
    test_keypath_sighash_deterministic();
    test_keypath_sighash_anyonecanpay();
    test_keypath_sighash_with_annex();

    // BIP-342 tapscript
    test_tapscript_sighash_deterministic();
    test_tapscript_vs_keypath();
    test_tapscript_code_separator();
    test_tapscript_leaf_version();
    test_tapscript_sighash_types();

    (void)std::printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}

int test_bip342_run() { return run_all_bip342_tests(); }

#ifdef STANDALONE_TEST
int main() { return run_all_bip342_tests(); }
#endif
