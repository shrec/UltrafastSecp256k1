// ============================================================================
// UltrafastSecp256k1 — Ethereum ZK-Layer Demo
// ============================================================================
// Demonstrates a realistic ZK-rollup / Layer-2 workflow on Ethereum using
// only the UltrafastSecp256k1 C++ library.  No external dependencies.
//
// Build:
//   cmake --build <build_dir> --target example_eth_zk_layer
//
// What this demo covers
// ---------------------
// 1. BIP-32 HD key derivation (m/44'/60'/0'/0/0) → Ethereum wallet
// 2. EIP-55 checksum address (compatible with MetaMask / SafePal / Bitget)
// 3. EIP-191 personal_sign — the "Sign Message" flow used by every EVM wallet
// 4. ecrecover — backend wallet validation (equivalent to Solidity ecrecover)
// 5. ZK proof bundle for a confidential L2 transaction:
//    a. Pedersen commitment  — hides the transfer amount on-chain
//    b. Bulletproof range proof — proves 0 <= amount < 2^64 without revealing it
//    c. Knowledge proof      — proves the sender owns the signing key
//    d. DLEQ proof           — binding between L2 key and commitment key
// 6. Batch ECDSA verification — sequencer block processing throughput demo
//
// Real-world integration notes (for the Ethereum developer)
// ----------------------------------------------------------
// * Steps 3-4 implement standard "Connect Wallet" authentication for your
//   backend ZK-layer node.  MetaMask / SafePal / Bitget call eth_sign or
//   personal_sign via JSON-RPC; your backend calls ufsecp_eth_ecrecover to
//   confirm the recovered address matches the claimed wallet address.
// * The ZK bundle (step 5) is what a Solidity verifier contract receives.
//   Commitments and proofs are serialised and submitted to the L2 node or
//   posted on-chain as calldata.
// * Batch verify (step 6) is the sequencer hot-path: verify every user
//   transaction in a block before including it in a ZK proof batch.
// ============================================================================

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

// Public C ABI — the only interface this demo needs
#include "ufsecp/ufsecp.h"

// ── error checking ───────────────────────────────────────────────────────────
// CHECK() is used instead of assert() so it works regardless of NDEBUG/build type.
// It calls the API unconditionally and terminates on any error.
#define CHECK(expr)                                                            \
    do {                                                                       \
        ufsecp_error_t _rc_ = (expr);                                          \
        if (_rc_ != UFSECP_OK) {                                               \
            fprintf(stderr, "\n  FAIL [line %d]: %s => error %d\n",           \
                    __LINE__, #expr, (int)_rc_);                               \
            ufsecp_ctx_destroy(ctx);                                           \
            return 1;                                                          \
        }                                                                      \
    } while (0)

// ── helpers ──────────────────────────────────────────────────────────────────

static void print_hex(const char* label, const uint8_t* d, size_t n) {
    printf("    %-28s 0x", label);
    for (size_t i = 0; i < n; ++i) printf("%02x", d[i]);
    printf("\n");
}

// Deterministic non-CSPRNG — DEMO ONLY; use OS CSPRNG in production
static void demo_rand(uint8_t* out, size_t n, uint64_t seed) {
    uint64_t s = seed ^ 0x9E3779B97F4A7C15ULL;
    for (size_t i = 0; i < n; ++i) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        out[i] = static_cast<uint8_t>(s & 0xFF);
    }
}

static void section(int n, const char* title) {
    printf("\n[%d] %s\n", n, title);
    printf("    %s\n",
           "----------------------------------------------------------------------");
}

// ── main ─────────────────────────────────────────────────────────────────────

int main() {
    printf("==========================================================================\n");
    printf(" UltrafastSecp256k1 -- Ethereum ZK-Layer Demo\n");
    printf(" Compatible with: MetaMask / SafePal / Bitget / any EVM wallet\n");
    printf("==========================================================================\n");

    // ── Create library context ────────────────────────────────────────────────
    ufsecp_ctx* ctx = nullptr;
    {
        ufsecp_error_t rc = ufsecp_ctx_create(&ctx);
        if (rc != UFSECP_OK || !ctx) {
            fprintf(stderr, "  FAIL: ufsecp_ctx_create => %d\n", (int)rc);
            return 1;
        }
    }

    // ── 0) Library version & ABI sanity check ────────────────────────────────
    section(0, "Library version & ABI sanity check");
    printf("    Version: %d.%d.%d  ABI: %u\n",
           UFSECP_VERSION_MAJOR, UFSECP_VERSION_MINOR, UFSECP_VERSION_PATCH,
           ufsecp_abi_version());
    if (ufsecp_abi_version() != UFSECP_ABI_VERSION) {
        fprintf(stderr, "  FAIL: ABI version mismatch (expected %u, got %u)\n",
                UFSECP_ABI_VERSION, ufsecp_abi_version());
        ufsecp_ctx_destroy(ctx);
        return 1;
    }
    {
        // Quick sign+verify round-trip to confirm the library is operational
        const uint8_t tsk[32] = {
            0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
            0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
            0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
            0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01,
        };
        const uint8_t tmsg[32] = {
            0xde,0xad,0xbe,0xef,0xde,0xad,0xbe,0xef,
            0xde,0xad,0xbe,0xef,0xde,0xad,0xbe,0xef,
            0xde,0xad,0xbe,0xef,0xde,0xad,0xbe,0xef,
            0xde,0xad,0xbe,0xef,0xde,0xad,0xbe,0xef,
        };
        uint8_t tpub[33] = {}, tsig[64] = {};
        CHECK(ufsecp_pubkey_create(ctx, tsk, tpub));
        CHECK(ufsecp_ecdsa_sign(ctx, tmsg, tsk, tsig));
        CHECK(ufsecp_ecdsa_verify(ctx, tmsg, tsig, tpub));
    }
    printf("    PASS\n");

    // ========================================================================
    // STEP 1 -- BIP-32 HD key derivation  (Ethereum path m/44'/60'/0'/0/0)
    // ========================================================================
    section(1, "BIP-32 HD key derivation (m/44'/60'/0'/0/0)");

    // BIP-39 seed: in production = PBKDF2-HMAC-SHA512(mnemonic, passphrase)
    const uint8_t demo_seed[64] = {
        0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
        0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,
        0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,
        0x18,0x19,0x1a,0x1b,0x1c,0x1d,0x1e,0x1f,
        0x20,0x21,0x22,0x23,0x24,0x25,0x26,0x27,
        0x28,0x29,0x2a,0x2b,0x2c,0x2d,0x2e,0x2f,
        0x30,0x31,0x32,0x33,0x34,0x35,0x36,0x37,
        0x38,0x39,0x3a,0x3b,0x3c,0x3d,0x3e,0x3f,
    };

    ufsecp_bip32_key master = {};
    CHECK(ufsecp_bip32_master(ctx, demo_seed, sizeof(demo_seed), &master));

    ufsecp_bip32_key child = {};
    CHECK(ufsecp_bip32_derive_path(ctx, &master, "m/44'/60'/0'/0/0", &child));

    uint8_t privkey[32] = {};
    uint8_t pubkey33[33] = {};
    CHECK(ufsecp_bip32_privkey(ctx, &child, privkey));
    CHECK(ufsecp_bip32_pubkey(ctx, &child, pubkey33));

    print_hex("Private key:", privkey, 32);
    print_hex("Public key (compressed):", pubkey33, 33);

    // ========================================================================
    // STEP 2 -- Ethereum address (EIP-55 checksummed)
    // ========================================================================
    section(2, "Ethereum address (EIP-55 checksum -- shown in every EVM wallet)");

    uint8_t addr20[20] = {};
    char    addr_str[44] = {};
    size_t  addr_len = sizeof(addr_str);

    CHECK(ufsecp_eth_address(ctx, pubkey33, addr20));
    CHECK(ufsecp_eth_address_checksummed(ctx, pubkey33, addr_str, &addr_len));

    printf("    Raw (20 bytes):    0x");
    for (int i = 0; i < 20; ++i) printf("%02x", addr20[i]);
    printf("\n");
    printf("    EIP-55 checksum:   %s\n", addr_str);
    printf("\n    >> Paste into MetaMask 'Import Account' to verify.\n");

    // ========================================================================
    // STEP 3 -- EIP-191 personal_sign  (wallet connect / authentication)
    //
    //   Your node activator dashboard sends a challenge string to the wallet.
    //   MetaMask / SafePal / Bitget show it to the user as "Sign Message".
    //   The frontend relays (v, r, s) back to your backend.
    // ========================================================================
    section(3, "EIP-191 personal_sign  (wallet connect authentication)");

    const char* challenge =
        "Connect to UltrafastSecp256k1 ZK-Layer Node\nNonce: 9f3a21b4c7d0e852";
    uint8_t personal_hash[32] = {};
    CHECK(ufsecp_eth_personal_hash(
            reinterpret_cast<const uint8_t*>(challenge), strlen(challenge),
            personal_hash));
    print_hex("Challenge hash (EIP-191):", personal_hash, 32);

    uint8_t sig_r[32] = {}, sig_s[32] = {};
    uint64_t sig_v = 0;
    const uint64_t CHAIN_ID = 1;  // 11155111 for Sepolia testnet
    CHECK(ufsecp_eth_sign(ctx, personal_hash, privkey,
                         sig_r, sig_s, &sig_v, CHAIN_ID));
    print_hex("r:", sig_r, 32);
    print_hex("s:", sig_s, 32);
    printf("    %-28s %llu\n", "v (EIP-155):", (unsigned long long)sig_v);
    printf("\n    >> Wallet returns (v, r, s) to your backend via JSON-RPC.\n");

    // ========================================================================
    // STEP 4 -- ecrecover  (backend wallet validation)
    //   Equivalent to Solidity: require(ecrecover(hash, v, r, s) == claimedAddr)
    // ========================================================================
    section(4, "ecrecover -- backend wallet address validation");

    uint8_t recovered_addr[20] = {};
    CHECK(ufsecp_eth_ecrecover(ctx, personal_hash, sig_r, sig_s, sig_v,
                              recovered_addr));

    printf("    Claimed  address:  0x");
    for (int i = 0; i < 20; ++i) printf("%02x", addr20[i]);
    printf("\n");
    printf("    Recovered address: 0x");
    for (int i = 0; i < 20; ++i) printf("%02x", recovered_addr[i]);
    printf("\n");

    bool wallet_ok = (memcmp(addr20, recovered_addr, 20) == 0);
    printf("    Result: %s\n",
           wallet_ok ? "VALID -- wallet owns this Ethereum address"
                     : "INVALID");
    if (!wallet_ok) { ufsecp_ctx_destroy(ctx); return 1; }
    printf("\n    >> Backend confirmed: wallet identity is authentic.\n");

    // ========================================================================
    // STEP 5 -- ZK proof bundle  (confidential L2 transaction)
    //
    //   The user proves four statements about a 1.5 ETH transfer without
    //   revealing the amount to on-chain observers:
    //   a. Pedersen commitment C = amount*H + blinding*G  (amount is hidden)
    //   b. Range proof: 0 <= amount < 2^64  (Bulletproof)
    //   c. Knowledge proof: sender owns privkey that controls P = privkey*G
    //   d. DLEQ proof: same privkey used for P and Q = privkey*H
    //
    //   The bundle (C, rp, kp, dleq) is submitted to the ZK-layer sequencer.
    // ========================================================================
    section(5, "ZK proof bundle (confidential L2 transaction - 1.5 ETH)");

    const uint64_t amount_wei = 1500000000000000000ULL;  // 1.5 ETH

    uint8_t blinding[32] = {};
    uint8_t aux_rand[32] = {};
    demo_rand(blinding, 32, 0xABCD1234FEEDULL);
    demo_rand(aux_rand, 32, 0x0102030405060708ULL);

    // Amount as 32-byte big-endian (Pedersen API requires scalar-sized input)
    uint8_t amount_be[32] = {};
    for (int i = 0; i < 8; ++i)
        amount_be[31 - i] = static_cast<uint8_t>((amount_wei >> (i * 8)) & 0xFF);

    // [5a] Pedersen commitment
    printf("\n  [5a] Pedersen commitment  C = amount*H + blinding*G\n");
    uint8_t commitment[33] = {};
    CHECK(ufsecp_pedersen_commit(ctx, amount_be, blinding, commitment));
    print_hex("Commitment:", commitment, 33);
    CHECK(ufsecp_pedersen_verify(ctx, commitment, amount_be, blinding));
    printf("         Verify: PASS\n");

    // [5b] Bulletproof range proof
    printf("\n  [5b] Bulletproof range proof  (0 <= amount < 2^64)\n");
    uint8_t range_proof[UFSECP_ZK_RANGE_PROOF_MAX_LEN] = {};
    size_t  rp_len = sizeof(range_proof);
    CHECK(ufsecp_zk_range_prove(ctx, amount_wei, blinding, commitment,
                               aux_rand, range_proof, &rp_len));
    printf("         Proof size: %zu bytes\n", rp_len);
    CHECK(ufsecp_zk_range_verify(ctx, commitment, range_proof, rp_len));
    printf("         Verify: PASS\n");

    // [5c] Knowledge proof (sender owns the signing key)
    //   Bind proof to Ethereum address for replay protection.
    printf("\n  [5c] Knowledge proof  (sender controls signing key)\n");
    uint8_t kp_msg[32] = {};
    memcpy(kp_msg, addr20, 20);
    uint8_t knowledge_proof[UFSECP_ZK_KNOWLEDGE_PROOF_LEN] = {};
    CHECK(ufsecp_zk_knowledge_prove(ctx, privkey, pubkey33, kp_msg,
                                   aux_rand, knowledge_proof));
    print_hex("Proof:", knowledge_proof, UFSECP_ZK_KNOWLEDGE_PROOF_LEN);
    CHECK(ufsecp_zk_knowledge_verify(ctx, knowledge_proof, pubkey33, kp_msg));
    printf("         Verify: PASS\n");

    // [5d] DLEQ proof (same privkey for P=privkey*G and Q=privkey*H)
    //   H is a domain-separated public generator derived via Keccak-256.
    printf("\n  [5d] DLEQ proof  (L2 key binds to commitment generator)\n");

    // secp256k1 standard generator G (compressed, 33 bytes — well-known constant)
    const uint8_t G33[33] = {
        0x02,
        0x79,0xBE,0x66,0x7E,0xF9,0xDC,0xBB,0xAC,
        0x55,0xA0,0x62,0x95,0xCE,0x87,0x0B,0x07,
        0x02,0x9B,0xFC,0xDB,0x2D,0xCE,0x28,0xD9,
        0x59,0xF2,0x81,0x5B,0x16,0xF8,0x17,0x98,
    };

    const char* h_tag = "UltrafastSecp256k1/DLEQ/H-v1";
    uint8_t h_hash[32] = {};
    CHECK(ufsecp_keccak256(
            reinterpret_cast<const uint8_t*>(h_tag), strlen(h_tag),
            h_hash));

    // H33 = h_hash * G  (treat h_hash as a private key; ufsecp_pubkey_create = privkey * G)
    uint8_t H33[33] = {};
    CHECK(ufsecp_pubkey_create(ctx, h_hash, H33));

    // Q33 = privkey * H  (ufsecp_pubkey_tweak_mul computes tweak * pubkey)
    uint8_t Q33[33] = {};
    CHECK(ufsecp_pubkey_tweak_mul(ctx, H33, privkey, Q33));

    uint8_t dleq_proof[UFSECP_ZK_DLEQ_PROOF_LEN] = {};
    CHECK(ufsecp_zk_dleq_prove(ctx, privkey, G33, H33, pubkey33, Q33,
                               aux_rand, dleq_proof));
    print_hex("Proof:", dleq_proof, UFSECP_ZK_DLEQ_PROOF_LEN);
    CHECK(ufsecp_zk_dleq_verify(ctx, dleq_proof, G33, H33, pubkey33, Q33));
    printf("         Verify: PASS\n");

    size_t bundle_bytes = 33 + rp_len
                        + UFSECP_ZK_KNOWLEDGE_PROOF_LEN
                        + UFSECP_ZK_DLEQ_PROOF_LEN;
    printf("\n    >> ZK bundle ready: commitment=33 + rp=%zu + kp=%d + dleq=%d = %zu bytes\n",
           rp_len, UFSECP_ZK_KNOWLEDGE_PROOF_LEN, UFSECP_ZK_DLEQ_PROOF_LEN, bundle_bytes);

    // ========================================================================
    // STEP 6 -- Batch ECDSA verification  (ZK rollup sequencer hot-path)
    //
    //   Entry layout for ufsecp_ecdsa_batch_verify:
    //     [32-byte msg | 33-byte compressed pubkey | 64-byte compact sig] = 129 bytes
    // ========================================================================
    section(6, "Batch ECDSA verification (100 transactions / sequencer block)");

    const int N_TX = 100;
    const int ENTRY_SZ = 32 + 33 + 64;  // 129 bytes
    std::vector<uint8_t> batch(static_cast<size_t>(N_TX * ENTRY_SZ), 0);

    for (int i = 0; i < N_TX; ++i) {
        uint8_t* e = batch.data() + i * ENTRY_SZ;
        uint8_t  tx_priv[32] = {}, tx_pub[33] = {}, tx_msg[32] = {}, tx_sig[64] = {};

        demo_rand(tx_priv, 32, static_cast<uint64_t>(i) * 0x123456789ABCULL + 1);
        tx_priv[0] |= 0x01;  // prevent zero scalar
        demo_rand(tx_msg,  32, static_cast<uint64_t>(i) * 0xFEDCBA9876ULL + 2);

        CHECK(ufsecp_pubkey_create(ctx, tx_priv, tx_pub));
        CHECK(ufsecp_ecdsa_sign(ctx, tx_msg, tx_priv, tx_sig));

        memcpy(e +  0, tx_msg,  32);
        memcpy(e + 32, tx_pub,  33);
        memcpy(e + 65, tx_sig,  64);
    }

    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now();
    ufsecp_error_t rc = ufsecp_ecdsa_batch_verify(ctx, batch.data(), N_TX);
    double us = std::chrono::duration<double, std::micro>(clk::now() - t0).count();

    printf("    %d sigs in %.1f us  (%.2f us/sig,  ~%.0f tx/sec)\n",
           N_TX, us, us / N_TX, 1.0e6 / (us / N_TX));
    printf("    Result: %s\n", (rc == UFSECP_OK) ? "ALL VALID" : "INVALID");

    if (rc != UFSECP_OK) { ufsecp_ctx_destroy(ctx); return 1; }

    // ── Summary ──────────────────────────────────────────────────────────────
    printf("\n");
    printf("==========================================================================\n");
    printf(" All steps PASSED\n");
    printf(" Ethereum ZK-layer integration checklist:\n");
    printf("   BIP-32 derivation     m/44'/60'/0'/0/0\n");
    printf("   EIP-55 address        paste into any EVM wallet\n");
    printf("   personal_sign EIP-191 wallet connect authentication\n");
    printf("   ecrecover             backend wallet validation\n");
    printf("   Pedersen commitment   confidential amount on-chain\n");
    printf("   Bulletproof range     0 <= amount < 2^64\n");
    printf("   Knowledge proof       wallet proves key ownership\n");
    printf("   DLEQ proof            L2 key binds to commitment\n");
    printf("   Batch ECDSA verify    sequencer block processing\n");
    printf("==========================================================================\n");

    ufsecp_ctx_destroy(ctx);
    return 0;
}
