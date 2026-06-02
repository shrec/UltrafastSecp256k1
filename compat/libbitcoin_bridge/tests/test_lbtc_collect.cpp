/**
 * test_lbtc_collect.cpp — correctness test for the in-place "collect" verify
 * (ufsecp_lbtc_verify_ecdsa_collect / _schnorr_collect).
 *
 * Collect collapses each row's verdict INTO that row's trailing key cell:
 *   valid   -> key cell zeroed
 *   invalid -> key cell left intact (the caller's id survives)
 * so the caller collects every row with a non-zero key cell = the rejected set.
 *
 * The test is chunk-size-agnostic: built normally it runs single-chunk; built
 * with -DUFSECP_LBTC_CHUNK_OVERRIDE=8 (the bridge source recompiled) the SAME
 * corpus straddles chunk boundaries, exercising the (base+i)*stride global
 * indexing of CollectSink across chunks. Runs against whichever backend the
 * controller binds (CPU or GPU) — the contract is identical.
 *
 * Standalone:
 *   g++ -std=c++20 -I ../include -I ../../../include/ufsecp \
 *       test_lbtc_collect.cpp -lufsecp -o test_lbtc_collect
 */
#include "ufsecp_libbitcoin.h"
#include "ufsecp.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
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

/* id written into a row's key cell. MUST be non-zero and distinct per row so a
 * surviving cell is distinguishable from a zeroed (valid) cell — even for row 0.
 * Encodes (i + 1) little-endian across key_size bytes. */
void write_id(uint8_t* cell, size_t ks, size_t i) {
    const uint64_t id = (uint64_t)i + 1u;
    for (size_t k = 0; k < ks; ++k) cell[k] = (uint8_t)(id >> (8 * k));
}
bool cell_is_zero(const uint8_t* cell, size_t ks) {
    for (size_t k = 0; k < ks; ++k) if (cell[k]) return false;
    return true;
}
bool cell_is_id(const uint8_t* cell, size_t ks, size_t i) {
    uint8_t want[16]; write_id(want, ks, i);
    return std::memcmp(cell, want, ks) == 0;
}

enum Kind { ECDSA, SCHNORR };

size_t rec_size(Kind k) {
    return k == ECDSA ? UFSECP_LBTC_ECDSA_RECORD : UFSECP_LBTC_SCHNORR_RECORD;
}

/* Build a packed table with a non-zero distinct id in each row's key cell. */
std::vector<uint8_t> build(ufsecp_ctx* ctx, Kind k, size_t n, size_t ks) {
    const size_t rec = rec_size(k);
    const size_t stride = rec + ks;
    std::vector<uint8_t> rows(n * stride, 0);
    for (size_t i = 0; i < n; ++i) {
        uint8_t* r = rows.data() + i * stride;
        uint8_t sk[32], msg[32], pub[33], aux[32] = {0};
        make_sk(sk, (uint32_t)(i + 1));
        make_msg(msg, (uint32_t)(i + 1));
        if (ufsecp_pubkey_create(ctx, sk, pub) != UFSECP_OK) ++g_fail;
        if (k == ECDSA) {
            std::memcpy(r, msg, 32); std::memcpy(r + 32, pub, 33);
            if (ufsecp_ecdsa_sign(ctx, msg, sk, r + 65) != UFSECP_OK) ++g_fail;
        } else {
            std::memcpy(r, msg, 32); std::memcpy(r + 32, pub + 1, 32);
            if (ufsecp_schnorr_sign(ctx, msg, sk, aux, r + 64) != UFSECP_OK) ++g_fail;
        }
        write_id(r + rec, ks, i);  /* id = i+1, non-zero, distinct */
    }
    return rows;
}

void corrupt_sig(std::vector<uint8_t>& rows, Kind k, size_t i, size_t ks) {
    const size_t stride = rec_size(k) + ks;
    const size_t sig_off = (k == ECDSA) ? 65 : 64;
    rows[i * stride + sig_off] ^= 0x01;  /* flip a signature byte -> invalid */
}

void run_collect(ufsecp_lbtc_ctrl* ctrl, Kind k, uint8_t* rows, size_t n, size_t ks) {
    if (k == ECDSA) ufsecp_lbtc_verify_ecdsa_collect(ctrl, rows, n, ks);
    else            ufsecp_lbtc_verify_schnorr_collect(ctrl, rows, n, ks);
}
void run_results(ufsecp_lbtc_ctrl* ctrl, Kind k, const uint8_t* rows, size_t n,
                 size_t ks, uint8_t* res) {
    if (k == ECDSA) ufsecp_lbtc_verify_ecdsa(ctrl, rows, n, ks, res);
    else            ufsecp_lbtc_verify_schnorr(ctrl, rows, n, ks, res);
}

/* Full collect contract for one kind, with invalids straddling chunk boundaries
 * (indices chosen to cross 8/16/24/32 when kChunk==8). */
void test_kind(ufsecp_lbtc_ctrl* ctrl, ufsecp_ctx* sctx, Kind k) {
    const char* name = (k == ECDSA) ? "ecdsa" : "schnorr";
    const size_t N = 40, KS = 4;
    const size_t rec = rec_size(k), stride = rec + KS;

    /* boundary-straddling invalids */
    const size_t inv[] = {7, 8, 9, 16, 25, 31, 32};
    auto is_invalid = [&](size_t i) {
        for (size_t x : inv) if (x == i) return true; return false;
    };

    auto rows = build(sctx, k, N, KS);
    for (size_t i : inv) corrupt_sig(rows, k, i, KS);

    run_collect(ctrl, k, rows.data(), N, KS);

    /* (1) valid rows zeroed; (2) invalid rows keep their id */
    bool valid_ok = true, inv_ok = true;
    for (size_t i = 0; i < N; ++i) {
        const uint8_t* cell = rows.data() + i * stride + rec;
        if (is_invalid(i)) { if (!cell_is_id(cell, KS, i)) inv_ok = false; }
        else               { if (!cell_is_zero(cell, KS))  valid_ok = false; }
    }
    char buf[96];
    std::snprintf(buf, sizeof buf, "%s: valid rows have key cell zeroed", name);
    CHECK(valid_ok, buf);
    std::snprintf(buf, sizeof buf, "%s: invalid rows keep their id (= i+1)", name);
    CHECK(inv_ok, buf);

    /* (3) surviving-key set == rejected set (the caller's real contract) */
    std::vector<size_t> survivors;
    for (size_t i = 0; i < N; ++i)
        if (!cell_is_zero(rows.data() + i * stride + rec, KS)) survivors.push_back(i);
    bool set_ok = survivors.size() == (sizeof(inv) / sizeof(inv[0]));
    for (size_t i = 0; set_ok && i < survivors.size(); ++i)
        if (!is_invalid(survivors[i])) set_ok = false;
    std::snprintf(buf, sizeof buf, "%s: surviving-key set == rejected set", name);
    CHECK(set_ok, buf);

    /* (6) neighbor isolation around row 25 */
    bool neigh = cell_is_id(rows.data() + 25 * stride + rec, KS, 25) &&
                 cell_is_zero(rows.data() + 24 * stride + rec, KS) &&
                 cell_is_zero(rows.data() + 26 * stride + rec, KS);
    std::snprintf(buf, sizeof buf, "%s: neighbor isolation (24,26 zeroed, 25 survives)", name);
    CHECK(neigh, buf);

    /* (4) all-valid corpus -> no survivors */
    auto clean = build(sctx, k, N, KS);
    run_collect(ctrl, k, clean.data(), N, KS);
    bool none = true;
    for (size_t i = 0; i < N; ++i)
        if (!cell_is_zero(clean.data() + i * stride + rec, KS)) none = false;
    std::snprintf(buf, sizeof buf, "%s: all-valid corpus leaves zero survivors", name);
    CHECK(none, buf);

    /* (7) collect-vs-results parity (ties the new channel to the consensus-gated
     *     results channel): results[i]==1 <=> key cell zeroed. */
    auto a = build(sctx, k, N, KS), b = build(sctx, k, N, KS);
    for (size_t i : inv) { corrupt_sig(a, k, i, KS); corrupt_sig(b, k, i, KS); }
    std::vector<uint8_t> res(N, 0xAA);
    run_results(ctrl, k, a.data(), N, KS, res.data());
    run_collect(ctrl, k, b.data(), N, KS);
    bool parity = true;
    for (size_t i = 0; i < N; ++i) {
        const bool zeroed = cell_is_zero(b.data() + i * stride + rec, KS);
        if ((res[i] == 1) != zeroed) parity = false;
    }
    std::snprintf(buf, sizeof buf, "%s: collect-vs-results parity (valid<=>zeroed)", name);
    CHECK(parity, buf);
}

} // namespace

int main() {
    ufsecp_lbtc_ctrl* ctrl = nullptr;
    if (ufsecp_lbtc_ctrl_create(&ctrl, UFSECP_LBTC_AUTO) != UFSECP_OK || !ctrl) {
        std::printf("FATAL: controller create failed\n");
        return 1;
    }
    const char* be[] = {"CPU", "CUDA", "OpenCL", "Metal"};
    std::printf("collect: bound backend: %s (%s), kChunk=%zu\n",
                be[ufsecp_lbtc_ctrl_backend(ctrl)],
                ufsecp_lbtc_ctrl_device_name(ctrl),
#ifdef UFSECP_LBTC_CHUNK_OVERRIDE
                (size_t)UFSECP_LBTC_CHUNK_OVERRIDE
#else
                (size_t)0  /* 0 == default 262144 (single chunk for this corpus) */
#endif
    );

    ufsecp_ctx* sctx = nullptr;
    if (ufsecp_ctx_create(&sctx) != UFSECP_OK) { std::printf("FATAL: signing ctx\n"); return 1; }

    test_kind(ctrl, sctx, ECDSA);
    test_kind(ctrl, sctx, SCHNORR);

    /* (5) degenerate calls are no-ops that write nothing: an empty batch, a zero
     *     key size, and a NULL rows pointer each leave the caller's buffer exactly
     *     as supplied (so every id survives = rejected, never falsely accepted). */
    {
        const size_t N = 4, KS = 4, stride = UFSECP_LBTC_ECDSA_RECORD + KS;
        std::vector<uint8_t> rows(N * stride, 0xCC);  /* sentinel everywhere */
        auto unchanged = [&](const std::vector<uint8_t>& v) {
            for (auto x : v) if (x != 0xCC) return false; return true;
        };
        std::vector<uint8_t> a = rows;
        ufsecp_lbtc_verify_ecdsa_collect(ctrl, a.data(), 0, KS);   /* empty batch */
        CHECK(unchanged(a), "empty batch is a no-op: rows preserved");
        std::vector<uint8_t> z = rows;
        ufsecp_lbtc_verify_ecdsa_collect(ctrl, z.data(), N, 0);    /* zero key size */
        CHECK(unchanged(z), "zero key size is a no-op: rows preserved");

        /* A NULL rows pointer must write nothing and must not corrupt the
         * controller. Assert the latter with a real post-condition: a normal
         * verify issued AFTER the NULL call still zeroes its valid rows. */
        ufsecp_lbtc_verify_ecdsa_collect(ctrl, nullptr, N, KS);    /* NULL rows */
        auto live = build(sctx, ECDSA, 4, KS);
        ufsecp_lbtc_verify_ecdsa_collect(ctrl, live.data(), 4, KS);
        bool live_ok = true;
        for (size_t i = 0; i < 4; ++i)
            if (!cell_is_zero(live.data() + i * stride + UFSECP_LBTC_ECDSA_RECORD, KS))
                live_ok = false;
        CHECK(live_ok, "NULL rows leaves the controller usable for later batches");
    }

    /* (8) C++ typed-span collect overload (MUTABLE span). */
    {
#pragma pack(push, 1)
        struct Triple {
            uint8_t record[UFSECP_LBTC_ECDSA_RECORD];
            uint8_t identifier[4];
        };
#pragma pack(pop)
        static_assert(sizeof(Triple) == UFSECP_LBTC_ECDSA_RECORD + 4, "packed");
        const size_t N = 16, KS = 4;
        auto raw = build(sctx, ECDSA, N, KS);  /* [record|4-byte id] == Triple */
        Triple* batch = reinterpret_cast<Triple*>(raw.data());
        const size_t stride = sizeof(Triple);
        raw[5 * stride + 65] ^= 0x01;          /* corrupt row 5 */
        ufsecp::lbtc::Controller wrap;
        wrap.collect_ecdsa(std::span<Triple>(batch, N));
        bool ok = !cell_is_zero(batch[5].identifier, KS) &&
                  cell_is_id(batch[5].identifier, KS, 5);
        for (size_t i = 0; i < N; ++i)
            if (i != 5 && !cell_is_zero(batch[i].identifier, KS)) ok = false;
        CHECK(ok, "span<Triple> collect: only row 5 survives with its id");
    }

    ufsecp_ctx_destroy(sctx);
    ufsecp_lbtc_ctrl_destroy(ctrl);
    std::printf("\n%s\n", g_fail == 0 ? "ALL PASS" : "FAILURES PRESENT");
    return g_fail == 0 ? 0 : 1;
}
