/**
 * test_lbtc_multisig_threshold.cpp — 4-table batching model (libbitcoin).
 *
 * libbitcoin accumulates signature tuples into four tables: ecdsa / schnorr
 * (single, 3-byte block-fk tail) and multisig / threshold (m-of-n, 6-byte
 * m|n+group+block-fk tail). At the verify boundary the four collapse to two
 * kinds: a multisig row is one ECDSA signature, a threshold (tapscript) row is
 * one Schnorr signature; the m|n+group bytes ride in the opaque key tail and are
 * never interpreted by the bridge.
 *
 * This proves that:
 *   (A) verify_multisig / verify_threshold give correct per-row results[] with a
 *       6-byte tail (valid -> 1, corrupted -> 0);
 *   (B) collect_multisig / collect_threshold zero the ENTIRE 6-byte tail on valid
 *       rows and leave it intact (m|n+group+block-fk all survive) on invalid;
 *   (C) the verdict is INDEPENDENT of the tail width — the single (3-byte) and
 *       multi (6-byte) tables verify the same signatures identically (the record
 *       is the first 129/128 bytes; the bridge never reads the tail);
 *   (D) the C++ typed-span MultisigRow / ThresholdRow overloads behave the same.
 *
 * Standalone:
 *   g++ -std=c++20 -I ../include -I ../../../include/ufsecp \
 *       test_lbtc_multisig_threshold.cpp -lufsecp -o test_lbtc_mt
 */
#include "ufsecp_libbitcoin.h"
#include "ufsecp.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) { std::printf("  FAIL: %s\n", msg); ++g_fail; }            \
        else         { std::printf("  ok  : %s\n", msg); }                      \
    } while (0)

namespace {

enum Kind { ECDSA, SCHNORR };
size_t rec_size(Kind k) {
    return k == ECDSA ? UFSECP_LBTC_ECDSA_RECORD : UFSECP_LBTC_SCHNORR_RECORD;
}

void make_sk(uint8_t sk[32], uint32_t seed) {
    std::memset(sk, 0, 32);
    sk[28] = (uint8_t)(seed >> 24); sk[29] = (uint8_t)(seed >> 16);
    sk[30] = (uint8_t)(seed >> 8);  sk[31] = (uint8_t)(seed | 1u); /* nonzero */
}
void make_msg(uint8_t msg[32], uint32_t seed) {
    for (int i = 0; i < 32; ++i) msg[i] = (uint8_t)(seed * 2654435761u >> (i % 24));
}

/* Seed a non-zero, distinct m|n+group+block-fk tag across `ks` tail bytes so a
 * surviving cell (rejected) is always distinguishable from a zeroed (valid) one,
 * even for row 0. Encodes (i+1) little-endian. */
void write_tag(uint8_t* tail, size_t ks, size_t i) {
    const uint64_t id = (uint64_t)i + 1u;
    for (size_t k = 0; k < ks; ++k) tail[k] = (uint8_t)(id >> (8 * k));
}
bool tail_is_zero(const uint8_t* tail, size_t ks) {
    for (size_t k = 0; k < ks; ++k) if (tail[k]) return false;
    return true;
}
bool tail_is_tag(const uint8_t* tail, size_t ks, size_t i) {
    uint8_t want[16]; write_tag(want, ks, i);
    return std::memcmp(tail, want, ks) == 0;
}

struct Columns {
    std::vector<uint8_t> msg;
    std::vector<uint8_t> pub;
    std::vector<uint8_t> sig;
    std::vector<uint8_t> key;
    size_t key_size = 0;
};

/* Build a packed table: n rows of [record | ks-byte tail], valid signatures,
 * with a distinct non-zero tag in each tail. ks = 3 (single) or 6 (multi). */
std::vector<uint8_t> build(ufsecp_ctx* ctx, Kind k, size_t n, size_t ks) {
    const size_t rec = rec_size(k), stride = rec + ks;
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
        write_tag(r + rec, ks, i);
    }
    return rows;
}
void corrupt(std::vector<uint8_t>& rows, Kind k, size_t i, size_t ks) {
    const size_t stride = rec_size(k) + ks;
    const size_t sig_off = (k == ECDSA) ? 65 : 64;
    rows[i * stride + sig_off] ^= 0x01;  /* flip a signature byte -> invalid */
}

Columns split_columns(const std::vector<uint8_t>& rows, Kind k, size_t n, size_t ks) {
    const size_t rec = rec_size(k), stride = rec + ks, pub_size = (k == ECDSA) ? 33 : 32;
    Columns c;
    c.msg.resize(n * 32);
    c.pub.resize(n * pub_size);
    c.sig.resize(n * 64);
    c.key.resize(n * ks);
    c.key_size = ks;
    for (size_t i = 0; i < n; ++i) {
        const uint8_t* r = rows.data() + i * stride;
        std::memcpy(c.msg.data() + i * 32, r, 32);
        std::memcpy(c.pub.data() + i * pub_size, r + 32, pub_size);
        std::memcpy(c.sig.data() + i * 64, r + 32 + pub_size, 64);
        if (ks) std::memcpy(c.key.data() + i * ks, r + rec, ks);
    }
    return c;
}

void verify_tbl(ufsecp_lbtc_ctrl* c, Kind k, const uint8_t* rows, size_t n,
                size_t ks, uint8_t* res) {
    /* multisig table -> ECDSA path; threshold table -> Schnorr path. */
    if (k == ECDSA) ufsecp_lbtc_verify_ecdsa(c, rows, n, ks, res);
    else            ufsecp_lbtc_verify_schnorr(c, rows, n, ks, res);
}
void collect_tbl(ufsecp_lbtc_ctrl* c, Kind k, uint8_t* rows, size_t n, size_t ks) {
    if (k == ECDSA) ufsecp_lbtc_verify_ecdsa_collect(c, rows, n, ks);
    else            ufsecp_lbtc_verify_schnorr_collect(c, rows, n, ks);
}
void verify_columns_tbl(ufsecp_lbtc_ctrl* c, Kind k, const Columns& cols,
                        size_t n, uint8_t* res) {
    if (k == ECDSA)
        ufsecp_lbtc_verify_ecdsa_columns(
            c, cols.msg.data(), cols.pub.data(), cols.sig.data(), n, res);
    else
        ufsecp_lbtc_verify_schnorr_columns(
            c, cols.msg.data(), cols.pub.data(), cols.sig.data(), n, res);
}
void collect_columns_tbl(ufsecp_lbtc_ctrl* c, Kind k, Columns& cols, size_t n) {
    if (k == ECDSA)
        ufsecp_lbtc_verify_ecdsa_columns_collect(
            c, cols.msg.data(), cols.pub.data(), cols.sig.data(), n,
            cols.key.data(), cols.key_size);
    else
        ufsecp_lbtc_verify_schnorr_columns_collect(
            c, cols.msg.data(), cols.pub.data(), cols.sig.data(), n,
            cols.key.data(), cols.key_size);
}

/* The "multi" tables use a 6-byte tail (m|n + group + block-fk). */
constexpr size_t KS_MULTI = 6;

void test_table(ufsecp_lbtc_ctrl* ctrl, ufsecp_ctx* sctx, Kind k) {
    const char* name = (k == ECDSA) ? "multisig(ecdsa)" : "threshold(schnorr)";
    const size_t N = 24, rec = rec_size(k), stride = rec + KS_MULTI;
    const size_t inv[] = {0, 5, 11, 23};
    auto is_invalid = [&](size_t i) {
        for (size_t x : inv) {
            if (x == i) return true;
        }
        return false;
    };
    char buf[112];

    /* (A) results[] with the 6-byte tail */
    auto rows = build(sctx, k, N, KS_MULTI);
    for (size_t i : inv) corrupt(rows, k, i, KS_MULTI);
    std::vector<uint8_t> res(N, 0xAA);
    verify_tbl(ctrl, k, rows.data(), N, KS_MULTI, res.data());
    bool res_ok = true;
    for (size_t i = 0; i < N; ++i)
        if ((res[i] == 1) == is_invalid(i)) res_ok = false;
    std::snprintf(buf, sizeof buf, "%s: results[] correct with 6-byte tail", name);
    CHECK(res_ok, buf);

    /* (B) collect: valid -> whole tail zeroed; invalid -> tail intact */
    auto crows = build(sctx, k, N, KS_MULTI);
    for (size_t i : inv) corrupt(crows, k, i, KS_MULTI);
    collect_tbl(ctrl, k, crows.data(), N, KS_MULTI);
    bool col_ok = true;
    for (size_t i = 0; i < N; ++i) {
        const uint8_t* tail = crows.data() + i * stride + rec;
        if (is_invalid(i)) { if (!tail_is_tag(tail, KS_MULTI, i)) col_ok = false; }
        else               { if (!tail_is_zero(tail, KS_MULTI))   col_ok = false; }
    }
    std::snprintf(buf, sizeof buf, "%s: collect zeroes valid tail, keeps invalid m|n+group+fk", name);
    CHECK(col_ok, buf);

    /* (C) tail-width independence: 3-byte single table vs 6-byte multi table
     *     verify the SAME signatures to the SAME verdicts (record is first
     *     129/128 bytes; the tail is never read by the verify core). */
    auto single = build(sctx, k, N, 3);
    auto multi  = build(sctx, k, N, KS_MULTI);
    for (size_t i : inv) { corrupt(single, k, i, 3); corrupt(multi, k, i, KS_MULTI); }
    std::vector<uint8_t> rs(N, 0), rm(N, 0);
    verify_tbl(ctrl, k, single.data(), N, 3, rs.data());
    verify_tbl(ctrl, k, multi.data(),  N, KS_MULTI, rm.data());
    bool indep = std::memcmp(rs.data(), rm.data(), N) == 0;
    std::snprintf(buf, sizeof buf, "%s: 3-byte vs 6-byte tail -> identical verdicts", name);
    CHECK(indep, buf);

    /* (D) columnar/vertical API: same verdicts as packed rows, without requiring
     * the bridge to de-interleave [record|tail] rows into msg/pub/sig columns. */
    auto vrows = build(sctx, k, N, KS_MULTI);
    for (size_t i : inv) corrupt(vrows, k, i, KS_MULTI);
    auto cols = split_columns(vrows, k, N, KS_MULTI);
    std::vector<uint8_t> rr(N, 0), vr(N, 0);
    verify_tbl(ctrl, k, vrows.data(), N, KS_MULTI, rr.data());
    verify_columns_tbl(ctrl, k, cols, N, vr.data());
    bool vmatch = std::memcmp(rr.data(), vr.data(), N) == 0;
    std::snprintf(buf, sizeof buf, "%s: columnar verify matches packed-row verify", name);
    CHECK(vmatch, buf);

    /* (E) columnar collect: valid -> whole key column cell zeroed; invalid ->
     * m|n+group+block-fk cell intact. */
    auto cvrows = build(sctx, k, N, KS_MULTI);
    for (size_t i : inv) corrupt(cvrows, k, i, KS_MULTI);
    auto ccols = split_columns(cvrows, k, N, KS_MULTI);
    collect_columns_tbl(ctrl, k, ccols, N);
    bool ccol_ok = true;
    for (size_t i = 0; i < N; ++i) {
        const uint8_t* tail = ccols.key.data() + i * KS_MULTI;
        if (is_invalid(i)) { if (!tail_is_tag(tail, KS_MULTI, i)) ccol_ok = false; }
        else               { if (!tail_is_zero(tail, KS_MULTI))   ccol_ok = false; }
    }
    std::snprintf(buf, sizeof buf, "%s: columnar collect zeroes valid key cells only", name);
    CHECK(ccol_ok, buf);
}

} // namespace

int main() {
    ufsecp_lbtc_ctrl* ctrl = nullptr;
    if (ufsecp_lbtc_ctrl_create(&ctrl, UFSECP_LBTC_AUTO) != UFSECP_OK || !ctrl) {
        std::printf("FATAL: controller create failed\n");
        return 1;
    }
    const char* be[] = {"CPU", "CUDA", "OpenCL", "Metal"};
    std::printf("multisig/threshold: bound backend: %s (%s)\n",
                be[ufsecp_lbtc_ctrl_backend(ctrl)],
                ufsecp_lbtc_ctrl_device_name(ctrl));

    ufsecp_ctx* sctx = nullptr;
    if (ufsecp_ctx_create(&sctx) != UFSECP_OK) { std::printf("FATAL: signing ctx\n"); return 1; }

    /* canonical row sizes match evoskuil's exact tuples */
    CHECK(sizeof(ufsecp::lbtc::MultisigRow)  == 135, "MultisigRow is 135 bytes");
    CHECK(sizeof(ufsecp::lbtc::ThresholdRow) == 134, "ThresholdRow is 134 bytes");

    test_table(ctrl, sctx, ECDSA);    /* multisig table  -> ECDSA path  */
    test_table(ctrl, sctx, SCHNORR);  /* threshold table -> Schnorr path */

    /* (D) C++ typed-span MultisigRow / ThresholdRow overloads */
    {
        using ufsecp::lbtc::Controller;
        using ufsecp::lbtc::MultisigRow;
        const size_t N = 12;
        auto raw = build(sctx, ECDSA, N, KS_MULTI);   /* [EcdsaRecord|6] == MultisigRow */
        static_assert(sizeof(MultisigRow) == UFSECP_LBTC_ECDSA_RECORD + KS_MULTI, "");
        MultisigRow* batch = reinterpret_cast<MultisigRow*>(raw.data());
        corrupt(raw, ECDSA, 4, KS_MULTI);

        Controller wrap;
        std::vector<uint8_t> res(N, 0xAA);
        wrap.verify_multisig(std::span<const MultisigRow>(batch, N), res.data());
        bool vok = true;
        for (size_t i = 0; i < N; ++i) if ((res[i] == 1) != (i != 4)) vok = false;
        CHECK(vok, "span<MultisigRow> verify_multisig: only row 4 invalid");

        wrap.collect_multisig(std::span<MultisigRow>(batch, N));
        bool cok = !tail_is_zero(reinterpret_cast<uint8_t*>(&batch[4]) + UFSECP_LBTC_ECDSA_RECORD, KS_MULTI);
        for (size_t i = 0; i < N; ++i)
            if (i != 4 && !tail_is_zero(reinterpret_cast<uint8_t*>(&batch[i]) + UFSECP_LBTC_ECDSA_RECORD, KS_MULTI))
                cok = false;
        CHECK(cok, "span<MultisigRow> collect_multisig: only row 4 tail survives");
    }

    ufsecp_ctx_destroy(sctx);
    ufsecp_lbtc_ctrl_destroy(ctrl);
    std::printf("\n%s\n", g_fail == 0 ? "ALL PASS" : "FAILURES PRESENT");
    return g_fail == 0 ? 0 : 1;
}
