/**
 * test_lbtc_bridge.cpp — correctness test for the libbitcoin acceleration bridge.
 *
 * Covers: all-valid batches, single-corruption identification, the variable
 * opaque key column (stride handling), ECDSA + Schnorr, and the empty batch.
 * Runs against whichever backend the controller binds (CPU or GPU) — the
 * contract semantics are identical.
 *
 * Standalone build (links the engine ufsecp library):
 *   g++ -std=c++17 -I ../include -I ../../../include/ufsecp \
 *       test_lbtc_bridge.cpp -lufsecp -o test_lbtc_bridge
 */
#include "ufsecp_libbitcoin.h"
#include "ufsecp.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) { std::printf("  FAIL: %s\n", msg); ++g_fail; }            \
        else         { std::printf("  ok  : %s\n", msg); }                     \
    } while (0)

namespace {

void make_sk(uint8_t sk[32], uint32_t seed) {
    std::memset(sk, 0, 32);
    sk[28] = (uint8_t)(seed >> 24); sk[29] = (uint8_t)(seed >> 16);
    sk[30] = (uint8_t)(seed >> 8);  sk[31] = (uint8_t)(seed | 1u); /* nonzero */
}
void make_msg(uint8_t msg[32], uint32_t seed) {
    for (int i = 0; i < 32; ++i) msg[i] = (uint8_t)(seed * 2654435761u >> (i % 24));
}

/* Build a packed ECDSA table: n rows of (129 + key_size) bytes. */
std::vector<uint8_t> build_ecdsa(ufsecp_ctx* ctx, size_t n, size_t key_size) {
    const size_t stride = UFSECP_LBTC_ECDSA_RECORD + key_size;
    std::vector<uint8_t> rows(n * stride, 0);
    for (size_t i = 0; i < n; ++i) {
        uint8_t* r = rows.data() + i * stride;
        uint8_t sk[32], msg[32], pub[33];
        make_sk(sk, (uint32_t)(i + 1));
        make_msg(msg, (uint32_t)(i + 1));
        if (ufsecp_pubkey_create(ctx, sk, pub) != UFSECP_OK) ++g_fail;
        std::memcpy(r, msg, 32);          /* 32 msg    */
        std::memcpy(r + 32, pub, 33);     /* 33 pubkey */
        if (ufsecp_ecdsa_sign(ctx, msg, sk, r + 65) != UFSECP_OK) ++g_fail; /* 64 sig */
        if (key_size) for (size_t k = 0; k < key_size; ++k) /* opaque tag = row id */
            r[UFSECP_LBTC_ECDSA_RECORD + k] = (uint8_t)(i >> (8 * k));
    }
    return rows;
}

/* Build a packed Schnorr table: n rows of (128 + key_size) bytes. */
std::vector<uint8_t> build_schnorr(ufsecp_ctx* ctx, size_t n, size_t key_size) {
    const size_t stride = UFSECP_LBTC_SCHNORR_RECORD + key_size;
    std::vector<uint8_t> rows(n * stride, 0);
    for (size_t i = 0; i < n; ++i) {
        uint8_t* r = rows.data() + i * stride;
        uint8_t sk[32], msg[32], pub[33], aux[32] = {0};
        make_sk(sk, (uint32_t)(i + 101));
        make_msg(msg, (uint32_t)(i + 101));
        aux[0] = (uint8_t)i;
        if (ufsecp_pubkey_create(ctx, sk, pub) != UFSECP_OK) ++g_fail;
        std::memcpy(r, msg, 32);              /* 32 msg/sighash          */
        std::memcpy(r + 32, pub + 1, 32);     /* 32 xonly (x-coordinate) */
        if (ufsecp_schnorr_sign(ctx, msg, sk, aux, r + 64) != UFSECP_OK) ++g_fail; /* 64 sig */
        if (key_size) for (size_t k = 0; k < key_size; ++k)
            r[UFSECP_LBTC_SCHNORR_RECORD + k] = (uint8_t)(i >> (8 * k));
    }
    return rows;
}

} // namespace

int main() {
    ufsecp_lbtc_ctrl* ctrl = nullptr;
    if (ufsecp_lbtc_ctrl_create(&ctrl, UFSECP_LBTC_AUTO) != UFSECP_OK || !ctrl) {
        std::printf("FATAL: controller create failed\n");
        return 1;
    }
    const char* be[] = {"CPU", "CUDA", "OpenCL", "Metal"};
    std::printf("bridge bound backend: %s (%s)\n",
                be[ufsecp_lbtc_ctrl_backend(ctrl)],
                ufsecp_lbtc_ctrl_device_name(ctrl));

    /* A signing context to generate test vectors. */
    ufsecp_ctx* sctx = nullptr;
    if (ufsecp_ctx_create(&sctx) != UFSECP_OK) { std::printf("FATAL: signing ctx\n"); return 1; }

    /* --- ECDSA, no key column, all valid --- */
    {
        const size_t N = 64;
        auto rows = build_ecdsa(sctx, N, 0);
        std::vector<uint8_t> res(N, 0xAA);
        size_t inv[8], ninv = 0;
        auto rc = ufsecp_lbtc_verify_ecdsa(ctrl, rows.data(), N, 0,
                                           res.data(), inv, 8, &ninv);
        CHECK(rc == UFSECP_OK, "ecdsa: rc OK");
        CHECK(ninv == 0, "ecdsa: whole batch verifies");
        bool all1 = true; for (auto v : res) if (v != 1) all1 = false;
        CHECK(all1, "ecdsa: every result == 1");
    }

    /* --- ECDSA, corrupt one --- */
    {
        const size_t N = 50;
        auto rows = build_ecdsa(sctx, N, 0);
        rows[25 * UFSECP_LBTC_ECDSA_RECORD + 65] ^= 0x01; /* flip a sig byte */
        std::vector<uint8_t> res(N, 0xAA);
        size_t inv[8], ninv = 0;
        ufsecp_lbtc_verify_ecdsa(ctrl, rows.data(), N, 0, res.data(), inv, 8, &ninv);
        CHECK(ninv == 1, "ecdsa: exactly 1 invalid after corruption");
        CHECK(ninv == 1 && inv[0] == 25, "ecdsa: invalid index == 25");
        CHECK(res[25] == 0 && res[24] == 1, "ecdsa: per-row result marks row 25");
    }

    /* --- ECDSA, 4-byte opaque key column, corrupt one --- */
    {
        const size_t N = 40, KS = 4;
        auto rows = build_ecdsa(sctx, N, KS);
        const size_t stride = UFSECP_LBTC_ECDSA_RECORD + KS;
        rows[10 * stride + 65] ^= 0x02;
        std::vector<uint8_t> res(N, 0xAA);
        size_t inv[8], ninv = 0;
        ufsecp_lbtc_verify_ecdsa(ctrl, rows.data(), N, KS, res.data(), inv, 8, &ninv);
        CHECK(ninv == 1 && inv[0] == 10, "ecdsa+key: invalid index == 10 (stride ok)");
        /* opaque tag at the failing row is the caller's to read back */
        const uint8_t* tag = rows.data() + 10 * stride + UFSECP_LBTC_ECDSA_RECORD;
        CHECK(tag[0] == 10, "ecdsa+key: opaque tag preserved (= row id)");
    }

    /* --- Schnorr, all valid + corrupt one --- */
    {
        const size_t N = 32;
        auto rows = build_schnorr(sctx, N, 0);
        std::vector<uint8_t> res(N, 0xAA);
        size_t inv[8], ninv = 0;
        ufsecp_lbtc_verify_schnorr(ctrl, rows.data(), N, 0, res.data(), inv, 8, &ninv);
        CHECK(ninv == 0, "schnorr: whole batch verifies");

        rows[7 * UFSECP_LBTC_SCHNORR_RECORD + 64] ^= 0x01;
        ninv = 0;
        ufsecp_lbtc_verify_schnorr(ctrl, rows.data(), N, 0, res.data(), inv, 8, &ninv);
        CHECK(ninv == 1 && inv[0] == 7, "schnorr: invalid index == 7 after corruption");
    }

    /* --- C++ wrapper: sizing contract = record count + key size, no buffer
     *     size. The span overload takes the COUNT from span.size() (never a byte
     *     buffer size), with key_size implied 0. --- */
    {
        const size_t N = 16;
        auto rows = build_ecdsa(sctx, N, 0); /* tightly packed EcdsaRecords */
        ufsecp::lbtc::Controller wrap;       /* RAII, AUTO backend */
        std::vector<uint8_t> res(N, 0xAA);
        size_t ninv = 0;
#ifdef UFSECP_LBTC_HAS_SPAN
        const auto* recs = reinterpret_cast<const ufsecp::lbtc::EcdsaRecord*>(rows.data());
        auto rc = wrap.verify(std::span<const ufsecp::lbtc::EcdsaRecord>(recs, N),
                              res.data(), nullptr, 0, &ninv);
#else
        auto rc = wrap.verify_ecdsa(rows.data(), N, 0, res.data(), nullptr, 0, &ninv);
#endif
        CHECK(rc == UFSECP_OK && ninv == 0, "wrapper: count from span.size(), all valid");
        rows[3 * UFSECP_LBTC_ECDSA_RECORD + 65] ^= 0x04; /* flip a sig byte */
        ninv = 0;
#ifdef UFSECP_LBTC_HAS_SPAN
        wrap.verify(std::span<const ufsecp::lbtc::EcdsaRecord>(
                        reinterpret_cast<const ufsecp::lbtc::EcdsaRecord*>(rows.data()), N),
                    res.data(), nullptr, 0, &ninv);
#else
        wrap.verify_ecdsa(rows.data(), N, 0, res.data(), nullptr, 0, &ninv);
#endif
        CHECK(ninv == 1 && res[3] == 0, "wrapper: corruption detected, row 3 marked");
    }

    /* --- empty batch --- */
    {
        size_t ninv = 123;
        auto rc = ufsecp_lbtc_verify_ecdsa(ctrl, nullptr, 0, 0, nullptr, nullptr, 0, &ninv);
        CHECK(rc == UFSECP_OK && ninv == 0, "empty batch returns OK");
    }

    ufsecp_ctx_destroy(sctx);
    ufsecp_lbtc_ctrl_destroy(ctrl);

    std::printf("\n%s\n", g_fail == 0 ? "ALL PASS" : "FAILURES PRESENT");
    return g_fail == 0 ? 0 : 1;
}
