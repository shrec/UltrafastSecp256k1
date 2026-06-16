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

static constexpr uint8_t kScalarOrder[32] = {
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfe,
    0xba, 0xae, 0xdc, 0xe6, 0xaf, 0x48, 0xa0, 0x3b,
    0xbf, 0xd2, 0x5e, 0x8c, 0xd0, 0x36, 0x41, 0x41
};

static constexpr uint8_t kScalarHalfOrder[32] = {
    0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0x5d, 0x57, 0x6e, 0x73, 0x57, 0xa4, 0x50, 0x1d,
    0xdf, 0xe9, 0x2f, 0x46, 0x68, 0x1b, 0x20, 0xa0
};

int cmp32_be(const uint8_t* a, const uint8_t* b) {
    for (size_t i = 0; i < 32; ++i) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

void scalar_order_minus(uint8_t out[32], const uint8_t x[32]) {
    uint16_t borrow = 0;
    for (int i = 31; i >= 0; --i) {
        const uint16_t lhs = kScalarOrder[(size_t)i];
        const uint16_t rhs = (uint16_t)x[i] + borrow;
        if (lhs < rhs) {
            out[i] = (uint8_t)(lhs + 256u - rhs);
            borrow = 1;
        } else {
            out[i] = (uint8_t)(lhs - rhs);
            borrow = 0;
        }
    }
}

bool ecdsa_s_is_high(const uint8_t* sig64) {
    return cmp32_be(sig64 + 32, kScalarOrder) < 0 &&
           cmp32_be(sig64 + 32, kScalarHalfOrder) > 0;
}

void scalar_be_to_internal(uint8_t out[32], const uint8_t in[32]) {
    for (size_t i = 0; i < 32; ++i) out[i] = in[31 - i];
}

void scalar_internal_to_be(uint8_t out[32], const uint8_t in[32]) {
    for (size_t i = 0; i < 32; ++i) out[i] = in[31 - i];
}

void compact_to_lbtc_opaque(uint8_t* sig64) {
    uint8_t tmp[64];
    scalar_be_to_internal(tmp, sig64);
    scalar_be_to_internal(tmp + 32, sig64 + 32);
    std::memcpy(sig64, tmp, 64);
}

void lbtc_opaque_to_compact(const uint8_t* sig64, uint8_t out[64]) {
    scalar_internal_to_be(out, sig64);
    scalar_internal_to_be(out + 32, sig64 + 32);
}

bool lbtc_ecdsa_s_is_high(const uint8_t* sig64) {
    uint8_t compact[64];
    lbtc_opaque_to_compact(sig64, compact);
    return ecdsa_s_is_high(compact);
}

void make_high_s(uint8_t* sig64) {
    uint8_t high_s[32];
    scalar_order_minus(high_s, sig64 + 32);
    std::memcpy(sig64 + 32, high_s, 32);
}

void make_lbtc_high_s(uint8_t* sig64) {
    uint8_t compact[64];
    lbtc_opaque_to_compact(sig64, compact);
    make_high_s(compact);
    compact_to_lbtc_opaque(compact);
    std::memcpy(sig64, compact, 64);
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
        compact_to_lbtc_opaque(r + 65);   /* libbitcoin ec_signature opaque layout */
        if (key_size) for (size_t k = 0; k < key_size; ++k) /* opaque tag = row id */
            r[UFSECP_LBTC_ECDSA_RECORD + k] = (uint8_t)(i >> (8 * k));
    }
    return rows;
}

struct EcdsaColumns {
    std::vector<uint8_t> msg;
    std::vector<uint8_t> pub;
    std::vector<uint8_t> sig;
};

EcdsaColumns build_ecdsa_columns(ufsecp_ctx* ctx, size_t n) {
    auto rows = build_ecdsa(ctx, n, 0);
    EcdsaColumns cols;
    cols.msg.resize(n * 32);
    cols.pub.resize(n * 33);
    cols.sig.resize(n * 64);
    for (size_t i = 0; i < n; ++i) {
        const uint8_t* r = rows.data() + i * UFSECP_LBTC_ECDSA_RECORD;
        std::memcpy(cols.msg.data() + i * 32, r, 32);
        std::memcpy(cols.pub.data() + i * 33, r + 32, 33);
        std::memcpy(cols.sig.data() + i * 64, r + 65, 64);
    }
    return cols;
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

/* Derive failure info from the per-row results array — the bridge's only output.
 * The consumer (e.g. libbitcoin) maps failures to tokens the same way. */
struct InvalidInfo { size_t count; size_t first; };
InvalidInfo invalids(const std::vector<uint8_t>& res) {
    InvalidInfo r{0, (size_t)-1};
    for (size_t i = 0; i < res.size(); ++i)
        if (res[i] == 0) { if (r.first == (size_t)-1) r.first = i; ++r.count; }
    return r;
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
        uint8_t* high_sig = rows.data() + 11 * UFSECP_LBTC_ECDSA_RECORD + 65;
        make_lbtc_high_s(high_sig);
        CHECK(lbtc_ecdsa_s_is_high(high_sig), "ecdsa: fixture contains high-S signature");
        std::vector<uint8_t> res(N, 0xAA);
        ufsecp_lbtc_verify_ecdsa(ctrl, rows.data(), N, 0, res.data(), nullptr, 0, nullptr);
        CHECK(invalids(res).count == 0, "ecdsa: whole batch verifies, including high-S");
        bool all1 = true; for (auto v : res) if (v != 1) all1 = false;
        CHECK(all1, "ecdsa: every result == 1");
        CHECK(lbtc_ecdsa_s_is_high(high_sig), "ecdsa: source high-S row is not rewritten");
    }

    /* --- ECDSA sig packing: opaque ec_signature -> GPU-native compact (normalized),
     *     pre-built table verifies identically via the compact columns API.
     *     This is the "conversion algorithm" a caller uses to pass normalized
     *     signatures directly (no per-call conversion at verify time). --- */
    {
        const size_t N = 48;
        auto cols = build_ecdsa_columns(sctx, N);   /* opaque (ec_signature) sigs */
        make_lbtc_high_s(cols.sig.data() + 7 * 64); /* row 7 is consensus-valid high-S */
        CHECK(lbtc_ecdsa_s_is_high(cols.sig.data() + 7 * 64),
              "pack: fixture row 7 is high-S (opaque)");

        /* Reference: verify the opaque column directly (engine normalizes internally). */
        std::vector<uint8_t> res_opaque(N, 0xAA);
        ufsecp_lbtc_verify_ecdsa_columns(ctrl, cols.msg.data(), cols.pub.data(),
                                         cols.sig.data(), N, res_opaque.data());

        /* Pack the whole opaque column once -> public big-endian compact, low-S. */
        std::vector<uint8_t> packed(N * 64);
        ufsecp_lbtc_ecdsa_sigs_pack(cols.sig.data(), N, /*input_is_opaque=*/1,
                                    packed.data());

        bool any_high = false, fmt_ok = true;
        for (size_t i = 0; i < N; ++i) {
            if (ecdsa_s_is_high(packed.data() + i * 64)) any_high = true; /* compact is_high */
            /* packed must equal opaque->compact then low-S normalize. */
            uint8_t expect[64];
            lbtc_opaque_to_compact(cols.sig.data() + i * 64, expect);
            if (ecdsa_s_is_high(expect)) make_high_s(expect); /* n-s == low-S of the pair */
            if (std::memcmp(expect, packed.data() + i * 64, 64) != 0) fmt_ok = false;
        }
        CHECK(!any_high, "pack: every packed sig is low-S normalized");
        CHECK(fmt_ok, "pack: packed == byte-reversed opaque, low-S normalized");

        /* single-sig pack matches the batch pack (incl. in-place aliasing). */
        uint8_t one[64];
        std::memcpy(one, cols.sig.data() + 7 * 64, 64);
        ufsecp_lbtc_ecdsa_sig_pack(one, /*input_is_opaque=*/1, one); /* alias in==out */
        CHECK(std::memcmp(one, packed.data() + 7 * 64, 64) == 0,
              "pack: single (in-place) == batch pack for the high-S row");

        /* Verify the PRE-PACKED compact column: identical verdict, same backend. */
        std::vector<uint8_t> res_compact(N, 0xAA);
        ufsecp_lbtc_verify_ecdsa_columns_compact(ctrl, cols.msg.data(),
                                                 cols.pub.data(), packed.data(), N,
                                                 res_compact.data());
        CHECK(invalids(res_compact).count == 0,
              "pack: pre-packed compact column verifies (incl. the high-S row)");
        CHECK(res_opaque == res_compact,
              "pack: compact-column verdict == opaque-column verdict (parity)");
    }

    /* --- ECDSA row form: explicit _opaque / _compact + invalid_idx/_count --- */
    {
        const size_t N = 40;
        /* opaque rows (libbitcoin ec_signature layout), one high-S */
        auto rows_op = build_ecdsa(sctx, N, 3);
        const size_t stride = UFSECP_LBTC_ECDSA_RECORD + 3;
        make_lbtc_high_s(rows_op.data() + 9 * stride + 65);

        /* compact rows: same records, but the sig field is big-endian compact */
        auto rows_cp = rows_op;
        for (size_t i = 0; i < N; ++i) {
            uint8_t comp[64];
            lbtc_opaque_to_compact(rows_cp.data() + i * stride + 65, comp);
            std::memcpy(rows_cp.data() + i * stride + 65, comp, 64);
        }

        std::vector<uint8_t> r_op(N, 0xAA), r_cp(N, 0xAA);
        size_t inv_op = 1, inv_cp = 1;
        auto e1 = ufsecp_lbtc_verify_ecdsa_opaque(ctrl, rows_op.data(), N, 3,
                                                  r_op.data(), nullptr, 0, &inv_op);
        auto e2 = ufsecp_lbtc_verify_ecdsa_compact(ctrl, rows_cp.data(), N, 3,
                                                   r_cp.data(), nullptr, 0, &inv_cp);
        CHECK(e1 == UFSECP_OK && e2 == UFSECP_OK, "rowform: both return UFSECP_OK");
        CHECK(inv_op == 0, "rowform: _opaque all valid (incl high-S), invalid_count==0");
        CHECK(inv_cp == 0, "rowform: _compact all valid (incl high-S), invalid_count==0");
        CHECK(r_op == r_cp, "rowform: _opaque verdict == _compact verdict (parity)");

        /* unsuffixed default == _opaque */
        std::vector<uint8_t> r_def(N, 0xAA);
        ufsecp_lbtc_verify_ecdsa(ctrl, rows_op.data(), N, 3, r_def.data(),
                                 nullptr, 0, nullptr);
        CHECK(r_def == r_op, "rowform: ufsecp_lbtc_verify_ecdsa == _opaque");

        /* corrupt one opaque row -> invalid_idx + invalid_count report it */
        rows_op[17 * stride + 65] ^= 0x01;
        size_t idx[4] = {99, 99, 99, 99}, inv = 0;
        ufsecp_lbtc_verify_ecdsa_opaque(ctrl, rows_op.data(), N, 3, nullptr,
                                        idx, 4, &inv);
        CHECK(inv == 1, "rowform: one corruption -> invalid_count == 1");
        CHECK(idx[0] == 17, "rowform: invalid_idx[0] == corrupted row 17");
    }

    /* --- ECDSA, corrupt one --- */
    {
        const size_t N = 50;
        auto rows = build_ecdsa(sctx, N, 0);
        rows[25 * UFSECP_LBTC_ECDSA_RECORD + 65] ^= 0x01; /* flip a sig byte */
        std::vector<uint8_t> res(N, 0xAA);
        ufsecp_lbtc_verify_ecdsa(ctrl, rows.data(), N, 0, res.data(), nullptr, 0, nullptr);
        auto iv = invalids(res);
        CHECK(iv.count == 1, "ecdsa: exactly 1 invalid after corruption");
        CHECK(iv.count == 1 && iv.first == 25, "ecdsa: invalid row == 25");
        CHECK(res[25] == 0 && res[24] == 1, "ecdsa: per-row result marks row 25");
    }

    /* --- ECDSA, 4-byte opaque key column, corrupt one --- */
    {
        const size_t N = 40, KS = 4;
        auto rows = build_ecdsa(sctx, N, KS);
        const size_t stride = UFSECP_LBTC_ECDSA_RECORD + KS;
        uint8_t* high_sig = rows.data() + 9 * stride + 65;
        make_lbtc_high_s(high_sig);
        CHECK(lbtc_ecdsa_s_is_high(high_sig), "ecdsa+key: fixture contains high-S signature");
        rows[10 * stride + 65] ^= 0x02;
        std::vector<uint8_t> res(N, 0xAA);
        ufsecp_lbtc_verify_ecdsa(ctrl, rows.data(), N, KS, res.data(), nullptr, 0, nullptr);
        auto iv = invalids(res);
        CHECK(iv.count == 1 && iv.first == 10, "ecdsa+key: invalid row == 10 (stride ok)");
        CHECK(res[9] == 1, "ecdsa+key: high-S row remains valid beside corruption");
        CHECK(lbtc_ecdsa_s_is_high(high_sig), "ecdsa+key: source high-S row is not rewritten");
        /* opaque tag at the failing row is the caller's to read back */
        const uint8_t* tag = rows.data() + 10 * stride + UFSECP_LBTC_ECDSA_RECORD;
        CHECK(tag[0] == 10, "ecdsa+key: opaque tag preserved (= row id)");
    }

    /* --- ECDSA columns, all valid with high-S --- */
    {
        const size_t N = 24;
        auto cols = build_ecdsa_columns(sctx, N);
        uint8_t* high_sig = cols.sig.data() + 8 * 64;
        make_lbtc_high_s(high_sig);
        CHECK(lbtc_ecdsa_s_is_high(high_sig), "ecdsa columns: fixture contains high-S signature");
        std::vector<uint8_t> res(N, 0xAA);
        ufsecp_lbtc_verify_ecdsa_columns(ctrl, cols.msg.data(), cols.pub.data(),
                                         cols.sig.data(), N, res.data());
        CHECK(invalids(res).count == 0, "ecdsa columns: whole batch verifies, including high-S");
        CHECK(res[8] == 1, "ecdsa columns: high-S row remains valid");
        CHECK(lbtc_ecdsa_s_is_high(high_sig), "ecdsa columns: source high-S sig is not rewritten");

        cols.sig[13 * 64 + 7] ^= 0x01;
        ufsecp_lbtc_verify_ecdsa_columns(ctrl, cols.msg.data(), cols.pub.data(),
                                         cols.sig.data(), N, res.data());
        auto iv = invalids(res);
        CHECK(iv.count == 1 && iv.first == 13, "ecdsa columns: invalid row == 13 after corruption");
        CHECK(res[8] == 1, "ecdsa columns: high-S row remains valid beside corruption");
    }

    /* --- ECDSA columns collect, high-S valid row is zeroed --- */
    {
        const size_t N = 20, KS = 3;
        auto cols = build_ecdsa_columns(sctx, N);
        uint8_t* high_sig = cols.sig.data() + 4 * 64;
        make_lbtc_high_s(high_sig);
        CHECK(lbtc_ecdsa_s_is_high(high_sig), "ecdsa columns collect: fixture contains high-S signature");
        cols.sig[12 * 64 + 9] ^= 0x02;
        std::vector<uint8_t> keys(N * KS, 0);
        for (size_t i = 0; i < N; ++i)
            for (size_t k = 0; k < KS; ++k)
                keys[i * KS + k] = (uint8_t)(0xA0u + i + k);
        ufsecp_lbtc_verify_ecdsa_columns_collect(ctrl, cols.msg.data(), cols.pub.data(),
                                                 cols.sig.data(), N, keys.data(), KS);
        bool high_zero = true;
        for (size_t k = 0; k < KS; ++k) high_zero &= keys[4 * KS + k] == 0;
        bool invalid_kept = true;
        for (size_t k = 0; k < KS; ++k) invalid_kept &= keys[12 * KS + k] != 0;
        CHECK(high_zero, "ecdsa columns collect: high-S valid row key is zeroed");
        CHECK(invalid_kept, "ecdsa columns collect: invalid row key survives");
        CHECK(lbtc_ecdsa_s_is_high(high_sig), "ecdsa columns collect: source high-S sig is not rewritten");
    }

    /* --- Schnorr, all valid + corrupt one --- */
    {
        const size_t N = 32;
        auto rows = build_schnorr(sctx, N, 0);
        std::vector<uint8_t> res(N, 0xAA);
        ufsecp_lbtc_verify_schnorr(ctrl, rows.data(), N, 0, res.data(), nullptr, 0, nullptr);
        CHECK(invalids(res).count == 0, "schnorr: whole batch verifies");

        rows[7 * UFSECP_LBTC_SCHNORR_RECORD + 64] ^= 0x01;
        ufsecp_lbtc_verify_schnorr(ctrl, rows.data(), N, 0, res.data(), nullptr, 0, nullptr);
        auto iv = invalids(res);
        CHECK(iv.count == 1 && iv.first == 7, "schnorr: invalid row == 7 after corruption");
    }

    /* --- C++ wrapper: pass the row pointer + record COUNT + KEY SIZE + results.
     *     No buffer size, no invalid-index outputs — failures come from results[]. --- */
    {
        const size_t N = 16, KS = 4;
        auto rows = build_ecdsa(sctx, N, KS); /* rows of [EcdsaRecord | 4-byte key] */
        ufsecp::lbtc::Controller wrap;        /* RAII, AUTO backend */
        std::vector<uint8_t> res(N, 0xAA);
        const size_t stride = UFSECP_LBTC_ECDSA_RECORD + KS;
        wrap.verify_ecdsa(rows.data(), N, KS, res.data());
        CHECK(invalids(res).count == 0, "wrapper: count + key_size, buffer implied");
        rows[3 * stride + 65] ^= 0x04; /* flip a sig byte in row 3 */
        wrap.verify_ecdsa(rows.data(), N, KS, res.data());
        CHECK(invalids(res).count == 1 && res[3] == 0, "wrapper: corruption detected, row 3 marked");
    }

    /* --- typed-span overload: pass a packed struct span; count + key_size are both
     *     recovered from the element type, NOTHING about size is restated at the call
     *     site. Mirrors libbitcoin's `secp256k1::ecdsa::triple` (#pragma pack(1):
     *     { hash_digest, ec_compressed, ec_signature, token } == 129 + 3). --- */
    {
#pragma pack(push, 1)
        struct Triple {                  // == libbitcoin ecdsa::triple layout
            uint8_t record[UFSECP_LBTC_ECDSA_RECORD]; // 129: hash|point|sig
            uint8_t identifier[3];                    // 3-byte opaque token
        };
#pragma pack(pop)
        static_assert(sizeof(Triple) == UFSECP_LBTC_ECDSA_RECORD + 3,
                      "Triple must be tightly packed (132 bytes)");
        const size_t N = 16;
        auto raw = build_ecdsa(sctx, N, 3);   /* [record|3-byte key] rows == Triple */
        const Triple* batch = reinterpret_cast<const Triple*>(raw.data());
        ufsecp::lbtc::Controller wrap;
        std::vector<uint8_t> res(N, 0xAA);
        uint8_t* high_sig = raw.data() + 6 * sizeof(Triple) + 65;
        make_lbtc_high_s(high_sig);
        CHECK(lbtc_ecdsa_s_is_high(high_sig), "span<Triple>: fixture contains high-S signature");
        /* key_size (3) is derived from sizeof(Triple)-RECORD; count from span.size() */
        wrap.verify_ecdsa(std::span<const Triple>(batch, N), res.data());
        CHECK(invalids(res).count == 0, "span<Triple>: count+key_size from type, all valid with high-S");
        CHECK(lbtc_ecdsa_s_is_high(high_sig), "span<Triple>: source high-S row is not rewritten");
        raw[5 * sizeof(Triple) + 65] ^= 0x08; /* flip a sig byte in row 5 */
        wrap.verify_ecdsa(std::span<const Triple>(batch, N), res.data());
        CHECK(invalids(res).count == 1 && res[5] == 0, "span<Triple>: 3-byte stride correct, row 5 marked");
        CHECK(res[6] == 1, "span<Triple>: high-S row remains valid beside corruption");
    }

    /* --- empty batch (no-op; results untouched) --- */
    {
        std::vector<uint8_t> res(1, 0xAA);
        ufsecp_lbtc_verify_ecdsa(ctrl, nullptr, 0, 0, res.data(), nullptr, 0, nullptr);
        CHECK(res[0] == 0xAA, "empty batch is a no-op (results untouched)");
    }

    ufsecp_ctx_destroy(sctx);
    ufsecp_lbtc_ctrl_destroy(ctrl);

    std::printf("\n%s\n", g_fail == 0 ? "ALL PASS" : "FAILURES PRESENT");
    return g_fail == 0 ? 0 : 1;
}
