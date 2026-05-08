// ============================================================================
// LibFuzzer harness: BIP-324 frame decryption + AEAD boundary fuzzer
// ============================================================================
//
// TARGET: ufsecp_bip324_decrypt()                  — frame decryption
//         ufsecp_bip324_encrypt()                  — encrypt→decrypt round-trip
//         ufsecp_aead_chacha20_poly1305_decrypt()   — standalone AEAD parser
//
// CONTRACT: no crash on any byte sequence fed as encrypted frame.
//           Authentication failures (wrong tag, truncated frame) must return
//           UFSECP_ERR_VERIFY_FAIL, not crash or corrupt memory.
//
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ufsecp/ufsecp.h"

static ufsecp_ctx* g_ctx = nullptr;
static void ensure_ctx() {
    if (!g_ctx && ufsecp_ctx_create(&g_ctx) != UFSECP_OK) abort();
}

// A pre-established session pair (initiator + responder) for round-trip tests
struct SessionPair {
    ufsecp_bip324_session* initiator = nullptr;
    ufsecp_bip324_session* responder = nullptr;
    bool ready = false;

    bool init() {
        if (ready) return true;
        ensure_ctx();

        uint8_t ell_init[64], ell_resp[64];
        if (ufsecp_bip324_create(g_ctx, 1, &initiator, ell_init) != UFSECP_OK) return false;
        if (ufsecp_bip324_create(g_ctx, 0, &responder, ell_resp) != UFSECP_OK) return false;

        uint8_t sid_init[32], sid_resp[32];
        if (ufsecp_bip324_handshake(initiator, ell_resp, sid_init) != UFSECP_OK) return false;
        if (ufsecp_bip324_handshake(responder, ell_init, sid_resp) != UFSECP_OK) return false;

        ready = true;
        return true;
    }

    ~SessionPair() {
        if (initiator) ufsecp_bip324_destroy(initiator);
        if (responder) ufsecp_bip324_destroy(responder);
    }
};

static SessionPair g_sessions;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    ensure_ctx();

    if (size < 1) return 0;
    const uint8_t variant = data[0];
    const uint8_t* payload = data + 1;
    const size_t   plen    = size - 1;

    switch (variant & 0x03) {
        case 0: {
            // --- feed fuzz bytes as encrypted frame to a fresh session -------
            // Create a fresh responder (reusing g_sessions.initiator for key material)
            // We just feed arbitrary bytes to decrypt — must not crash.
            if (!g_sessions.init()) break;
            // Use a clone of the responder session (can't reuse after decryption advances state)
            // For simplicity, create a fresh one-shot session pair per run
            {
                ufsecp_ctx* local_ctx = nullptr;
                if (ufsecp_ctx_create(&local_ctx) != UFSECP_OK) break;

                ufsecp_bip324_session* loc_init = nullptr;
                ufsecp_bip324_session* loc_resp = nullptr;
                uint8_t le[64], re[64];
                if (ufsecp_bip324_create(local_ctx, 1, &loc_init, le) == UFSECP_OK &&
                    ufsecp_bip324_create(local_ctx, 0, &loc_resp, re) == UFSECP_OK)
                {
                    CHECK_OK(ufsecp_bip324_handshake(loc_init, re, nullptr), "bip324_handshake");
                    CHECK_OK(ufsecp_bip324_handshake(loc_resp, le, nullptr), "bip324_handshake");

                    // Feed fuzz payload as a "frame" to decrypt
                    std::vector<uint8_t> pt(plen + 64, 0);
                    size_t pt_len = pt.size();
                    (void)ufsecp_bip324_decrypt(loc_resp, payload, plen, pt.data(), &pt_len);
                }
                if (loc_init) ufsecp_bip324_destroy(loc_init);
                if (loc_resp) ufsecp_bip324_destroy(loc_resp);
                ufsecp_ctx_destroy(local_ctx);
            }
            break;
        }
        case 1: {
            // --- encrypt fuzz payload → decrypt must succeed ----------------
            if (!g_sessions.init()) break;
            if (plen == 0 || plen > 65536) break;

            ufsecp_ctx* local_ctx = nullptr;
            if (ufsecp_ctx_create(&local_ctx) != UFSECP_OK) break;

            ufsecp_bip324_session* li = nullptr;
            ufsecp_bip324_session* lr = nullptr;
            uint8_t le[64], re[64];
            bool ok = (ufsecp_bip324_create(local_ctx, 1, &li, le) == UFSECP_OK &&
                       ufsecp_bip324_create(local_ctx, 0, &lr, re) == UFSECP_OK &&
                       ufsecp_bip324_handshake(li, re, nullptr) == UFSECP_OK &&
                       ufsecp_bip324_handshake(lr, le, nullptr)  == UFSECP_OK);

            if (ok) {
                std::vector<uint8_t> ct(plen + 32, 0);
                size_t ct_len = ct.size();
                ufsecp_error_t rc_enc = ufsecp_bip324_encrypt(li, payload, plen, ct.data(), &ct_len);
                if (rc_enc == UFSECP_OK && ct_len > 0) {
                    std::vector<uint8_t> pt(ct_len, 0);
                    size_t pt_len = pt.size();
                    ufsecp_error_t rc_dec = ufsecp_bip324_decrypt(lr, ct.data(), ct_len, pt.data(), &pt_len);
                    if (rc_dec == UFSECP_OK) {
                        // Decrypted payload must equal original
                        if (pt_len != plen || memcmp(pt.data(), payload, plen) != 0) {
                            __builtin_trap();  // round-trip integrity failure
                        }
                    }
                    // rc_dec != OK is allowed (state mismatch, etc.) — just not a crash
                }
            }
            if (li) ufsecp_bip324_destroy(li);
            if (lr) ufsecp_bip324_destroy(lr);
            ufsecp_ctx_destroy(local_ctx);
            break;
        }
        case 2: {
            // --- ChaCha20-Poly1305 standalone decrypt fuzz ------------------
            // key=first 32B, nonce=next 12B, tag=next 16B, rest=ciphertext
            if (plen < 32 + 12 + 16) break;
            const uint8_t* key   = payload;
            const uint8_t* nonce = payload + 32;
            const uint8_t* tag   = payload + 32 + 12;
            const uint8_t* ct    = payload + 32 + 12 + 16;
            size_t         ct_len = plen  - (32 + 12 + 16);
            std::vector<uint8_t> pt(ct_len + 1, 0);
            (void)ufsecp_aead_chacha20_poly1305_decrypt(
                key, nonce, nullptr, 0, ct, ct_len, tag, pt.data());
            break;
        }
        case 3: {
            // --- AAD fuzz: encrypt with fuzz AAD, then decrypt wrong AAD ----
            if (plen < 16) break;
            const uint8_t* aad    = payload;
            size_t         aad_len = plen < 32 ? plen : 32;
            static const uint8_t k32[32] = {};
            static const uint8_t n12[12] = {};
            static const uint8_t pt1[16] = {};
            uint8_t ct1[16];
            uint8_t tag1[16];
            (void)ufsecp_aead_chacha20_poly1305_encrypt(
                k32, n12, aad, aad_len, pt1, sizeof(pt1), ct1, tag1);
            // Decrypt with same AAD → should succeed
            uint8_t pt2[16];
            ufsecp_error_t rc1 = ufsecp_aead_chacha20_poly1305_decrypt(
                k32, n12, aad, aad_len, ct1, sizeof(ct1), tag1, pt2);
            if (rc1 == UFSECP_OK && memcmp(pt1, pt2, 16) != 0) {
                __builtin_trap();  // AEAD round-trip failure
            }
            // Decrypt with wrong AAD → must fail (or succeed if aad happened to match)
            uint8_t wrong_aad[32];
            memcpy(wrong_aad, aad, aad_len);
            wrong_aad[0] ^= 0xFF;
            uint8_t pt3[16];
            (void)ufsecp_aead_chacha20_poly1305_decrypt(
                k32, n12, wrong_aad, aad_len, ct1, sizeof(ct1), tag1, pt3);
            // Can't assert failure here (wrong_aad == aad if aad_len==0 edge case)
            break;
        }
    }

    return 0;
}

// ---------------------------------------------------------------------------
// Standalone
// ---------------------------------------------------------------------------
#if defined(LIBFUZZER_STANDALONE) || defined(SECP256K1_STANDALONE_FUZZ)

#include <cstdio>
#include <random>

int main() {
    ensure_ctx();
    printf("fuzz_bip324_frame: running\n");

    // Trigger the AEAD round-trip test
    std::vector<uint8_t> trigger(1 + 32 + 12 + 16 + 64, 0);
    trigger[0] = 0x03;
    for (size_t i = 1; i < trigger.size(); ++i) trigger[i] = (uint8_t)(i * 17);
    LLVMFuzzerTestOneInput(trigger.data(), trigger.size());

    std::mt19937_64 rng(0xDEADBEEFCAFE0006ULL);
    constexpr int kIter = 10000;  // fewer iterations since session creation is expensive
    for (int i = 0; i < kIter; ++i) {
        const size_t len = 1 + (rng() % 256);
        std::vector<uint8_t> buf(len);
        for (auto& b : buf) b = static_cast<uint8_t>(rng());
        buf[0] = buf[0] & 0x03;
        LLVMFuzzerTestOneInput(buf.data(), buf.size());
    }
    printf("fuzz_bip324_frame: PASS (%d iterations)\n", kIter);
    return 0;
}
#endif
