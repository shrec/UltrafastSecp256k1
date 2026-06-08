/**
 * test_lbtc_commitment.cpp — BIP-341 Taproot commitment batch
 * (ufsecp_lbtc_verify_commitment).
 *
 * Each row checks: Q = lift_x(internal, even-y) + tweak*G has x(Q)==tweaked_x and
 * y-parity(Q)==parity. Ground truth is generated with, and cross-checked against,
 * the shim's native secp256k1_xonly_pubkey_tweak_add / _tweak_add_check (so the
 * batch is validated against the canonical per-call path, not against itself).
 *
 * Standalone:
 *   g++ -std=c++20 -I ../include -I ../../libsecp256k1_shim/include \
 *       -I ../../../include/ufsecp test_lbtc_commitment.cpp -lufsecp -o test_lbtc_commitment
 */
#include "ufsecp_libbitcoin.h"
#include "ufsecp.h"
#include "ufsecp_gpu.h"
#include "secp256k1.h"
#include "secp256k1_extrakeys.h"
#include "secp256k1_schnorrsig.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) { std::printf("  FAIL: %s\n", msg); ++g_fail; }            \
        else         { std::printf("  ok  : %s\n", msg); }                      \
    } while (0)

int main() {
    secp256k1_context* sctx =
        secp256k1_context_create(SECP256K1_CONTEXT_VERIFY | SECP256K1_CONTEXT_SIGN);
    ufsecp_ctx* uctx = nullptr;
    if (ufsecp_ctx_create(&uctx) != UFSECP_OK) { std::printf("ctx fail\n"); return 1; }
    ufsecp_lbtc_ctrl* ctrl = nullptr;
    if (ufsecp_lbtc_ctrl_create(&ctrl, UFSECP_LBTC_AUTO) != UFSECP_OK || !ctrl) {
        std::printf("ctrl fail\n"); return 1;
    }

    const size_t N = 1000;
    std::vector<uint8_t> ix(N*32), tw(N*32), tx(N*32), par(N);
    std::vector<secp256k1_xonly_pubkey> ixo(N);
    for (size_t i = 0; i < N; ++i) {
        uint8_t sk[32] = {0};
        sk[24]=(uint8_t)(i>>24); sk[25]=(uint8_t)(i>>16); sk[26]=(uint8_t)(i>>8); sk[31]=(uint8_t)(i|1u);
        uint8_t p33[33];
        if (ufsecp_pubkey_create(uctx, sk, p33) != UFSECP_OK) { ++g_fail; continue; }
        std::memcpy(ix.data()+i*32, p33+1, 32);              /* x-only internal */
        for (int b=0;b<32;++b) tw[i*32+b]=(uint8_t)(((i*2654435761u+b*7u+1)>>(b%24))|((b==31)?1u:0u));
        secp256k1_xonly_pubkey_parse(sctx, &ixo[i], ix.data()+i*32);
        secp256k1_pubkey twk;
        if (!secp256k1_xonly_pubkey_tweak_add(sctx, &twk, &ixo[i], tw.data()+i*32)) { ++g_fail; continue; }
        uint8_t c33[33]; size_t L=33;
        secp256k1_ec_pubkey_serialize(sctx, c33, &L, &twk, SECP256K1_EC_COMPRESSED);
        std::memcpy(tx.data()+i*32, c33+1, 32);
        par[i] = (c33[0]==0x03) ? 1 : 0;
    }

    /* (1) all-valid batch -> every result 1, and matches the shim per-call check */
    std::vector<uint8_t> res(N, 0);
    ufsecp_lbtc_verify_commitment(ctrl, ix.data(), tw.data(), tx.data(), par.data(), N, res.data());
    bool all_one = true, match_shim = true;
    for (size_t i = 0; i < N; ++i) {
        if (res[i] != 1) all_one = false;
        int shim = secp256k1_xonly_pubkey_tweak_add_check(sctx, tx.data()+i*32, par[i], &ixo[i], tw.data()+i*32);
        if ((res[i]==1) != (shim==1)) match_shim = false;
    }
    CHECK(all_one, "all-valid batch: every result == 1");
    CHECK(match_shim, "batch verdict matches shim tweak_add_check per row");

    /* (2) corrupt a subset of tweaked keys -> exactly those rows reject */
    const size_t bad[] = {0, 3, 17, 499, 999};
    for (size_t i : bad) tx[i*32] ^= 0x01;                    /* flip tweaked_x */
    std::fill(res.begin(), res.end(), 0xAA);
    ufsecp_lbtc_verify_commitment(ctrl, ix.data(), tw.data(), tx.data(), par.data(), N, res.data());
    auto is_bad = [&](size_t i){ for (size_t x:bad) if (x==i) return true; return false; };
    bool sel_ok = true;
    for (size_t i = 0; i < N; ++i) if ((res[i]==1) == is_bad(i)) sel_ok = false;
    CHECK(sel_ok, "corrupted rows reject, all others accept");
    for (size_t i : bad) tx[i*32] ^= 0x01;                    /* restore */

    /* (3) wrong parity -> reject */
    std::fill(res.begin(), res.end(), 0);
    std::vector<uint8_t> badpar = par; badpar[7] ^= 1;
    ufsecp_lbtc_verify_commitment(ctrl, ix.data(), tw.data(), tx.data(), badpar.data(), N, res.data());
    CHECK(res[7]==0 && res[6]==1 && res[8]==1, "flipped parity rejects only that row");

    /* (4) degenerate calls are fail-closed no-ops */
    std::vector<uint8_t> z(4, 0xCC);
    ufsecp_lbtc_verify_commitment(ctrl, ix.data(), tw.data(), tx.data(), par.data(), 0, z.data());
    CHECK(z[0]==0xCC, "n==0 is a no-op (results untouched)");
    std::vector<uint8_t> z2(4, 0xCC);
    ufsecp_lbtc_verify_commitment(ctrl, nullptr, tw.data(), tx.data(), par.data(), 4, z2.data());
    bool null_all_zeroed = true;
    for (uint8_t b : z2) if (b != 0) null_all_zeroed = false;  /* every byte zeroed = no row accepted */
    CHECK(null_all_zeroed, "NULL input zeroes every result byte (no row left accepted)");

    /* (5) GPU RLC aggregate fast-check (commitment_batch_ok). Skips when no GPU. */
    {
        int ok = ufsecp_lbtc_commitment_batch_ok(ctrl, ix.data(), tw.data(), tx.data(), par.data(), N);
        if (ok < 0) {
            std::printf("  skip: commitment_batch_ok — no GPU build/device (returned -1)\n");
        } else {
            CHECK(ok == 1, "GPU RLC: all-valid batch returns 1");
            uint8_t saved = tx[11*32]; tx[11*32] ^= 0x01;     /* one invalid row */
            int bad = ufsecp_lbtc_commitment_batch_ok(ctrl, ix.data(), tw.data(), tx.data(), par.data(), N);
            tx[11*32] = saved;
            CHECK(bad == 0, "GPU RLC: one corrupted row returns 0 (Fiat-Shamir catches it)");
            /* a second, different corrupted position also rejects */
            uint8_t s2 = tx[800*32]; tx[800*32] ^= 0x80;
            int bad2 = ufsecp_lbtc_commitment_batch_ok(ctrl, ix.data(), tw.data(), tx.data(), par.data(), N);
            tx[800*32] = s2;
            CHECK(bad2 == 0, "GPU RLC: a different corrupted row also returns 0");
        }
    }

    /* (6) single-buffer (AoS) rows path: internal_x|tweak|tweaked_comp (97B) +tail */
    {
        const size_t REC = UFSECP_LBTC_COMMITMENT_RECORD;   /* 97 */
        const size_t TAIL = 3, STRIDE = REC + TAIL;          /* exercise a correlation tail */
        std::vector<uint8_t> rows(N*STRIDE, 0);
        for (size_t i = 0; i < N; ++i) {
            uint8_t* r = rows.data() + i*STRIDE;
            std::memcpy(r,      ix.data()+i*32, 32);
            std::memcpy(r + 32, tw.data()+i*32, 32);
            r[64] = par[i] ? 0x03 : 0x02;                    /* compressed prefix = parity */
            std::memcpy(r + 65, tx.data()+i*32, 32);
            r[97]=0xAA; r[98]=0xBB; r[99]=0xCC;              /* junk tail — must be ignored */
        }
        std::vector<uint8_t> res(N, 0);
        ufsecp_lbtc_verify_commitment_rows(ctrl, rows.data(), N, STRIDE, res.data());
        bool all1 = true; for (auto v : res) if (v != 1) all1 = false;
        CHECK(all1, "AoS rows: all-valid -> every result 1 (correlation tail ignored)");

        uint8_t s = rows[10*STRIDE+64]; rows[10*STRIDE+64] ^= 0x01;   /* flip parity prefix */
        std::fill(res.begin(), res.end(), 0);
        ufsecp_lbtc_verify_commitment_rows(ctrl, rows.data(), N, STRIDE, res.data());
        bool sel = (res[10]==0); for (size_t i=0;i<N;++i) if (i!=10 && res[i]!=1) sel=false;
        rows[10*STRIDE+64] = s;
        CHECK(sel, "AoS rows: corrupted parity prefix rejects only that row");

        int ok = ufsecp_lbtc_commitment_batch_ok_rows(ctrl, rows.data(), N, STRIDE);
        if (ok < 0) {
            std::printf("  skip: batch_ok_rows — no GPU build/device\n");
        } else {
            CHECK(ok == 1, "AoS rows GPU RLC: all-valid -> 1");
            uint8_t s2 = rows[20*STRIDE+70]; rows[20*STRIDE+70] ^= 0x01;
            int bad = ufsecp_lbtc_commitment_batch_ok_rows(ctrl, rows.data(), N, STRIDE);
            rows[20*STRIDE+70] = s2;
            CHECK(bad == 0, "AoS rows GPU RLC: corrupted row -> 0");
        }
    }

    /* (7) batch x-only pubkey validation (lift_x on-curve check) */
    {
        const size_t M = 64, STR = 32;
        std::vector<uint8_t> keys(M*STR, 0);
        for (size_t i = 0; i < M; ++i) {
            if (i & 1) std::memset(keys.data()+i*STR, 0xFF, 32);        /* >= p -> invalid */
            else       std::memcpy(keys.data()+i*STR, ix.data()+(i%N)*32, 32); /* valid */
        }
        std::vector<uint8_t> res(M, 0xAA);
        ufsecp_lbtc_validate_xonly(ctrl, keys.data(), M, STR, res.data());
        bool ok = true;
        for (size_t i = 0; i < M; ++i) {
            bool want = !(i & 1);
            secp256k1_xonly_pubkey xp;                                  /* shim ground truth */
            bool shim = secp256k1_xonly_pubkey_parse(sctx, &xp, keys.data()+i*STR) == 1;
            if ((res[i]==1) != want) ok = false;
            if ((res[i]==1) != shim) ok = false;
        }
        CHECK(ok, "validate_xonly: valid accept, >=p reject, matches shim xonly_parse");
    }

    /* (8) tagged-hash batch (TapBranch, 64-byte inputs) vs shim tagged_sha256 */
    {
        const size_t M = 128, ML = 64, STR = ML;
        std::vector<uint8_t> msgs(M*STR), out(M*32);
        for (size_t i = 0; i < M*STR; ++i) msgs[i] = (uint8_t)((i*2654435761u) >> (i%24));
        ufsecp_lbtc_tagged_hash_batch(ctrl, "TapBranch", msgs.data(), ML, M, STR, out.data());
        bool ok = true;
        for (size_t i = 0; i < M; ++i) {
            uint8_t ref[32];
            secp256k1_tagged_sha256(sctx, ref, (const unsigned char*)"TapBranch", 9,
                                    msgs.data()+i*STR, ML);
            if (std::memcmp(out.data()+i*32, ref, 32) != 0) ok = false;
        }
        CHECK(ok, "tagged_hash_batch TapBranch == shim secp256k1_tagged_sha256 per item");
    }

    /* (9) Direct GPU ABI per-item kernels vs shim ground truth. Probes each backend
     * and uses the first that actually IMPLEMENTS the kernels (CUDA); OpenCL/Metal/CPU
     * return UFSECP_ERR_GPU_UNSUPPORTED -> skipped (consensus served by the CPU path). */
    {
        ufsecp_gpu_ctx* g = nullptr;
        const uint32_t order[3] = {UFSECP_GPU_BACKEND_CUDA, UFSECP_GPU_BACKEND_OPENCL,
                                   UFSECP_GPU_BACKEND_METAL};
        for (uint32_t b : order) {
            if (!ufsecp_gpu_is_available(b)) continue;
            ufsecp_gpu_ctx* cand = nullptr;
            if (ufsecp_gpu_ctx_create(&cand, b, 0) == UFSECP_OK && ufsecp_gpu_is_ready(cand)) {
                uint8_t k[32]; std::memcpy(k, ix.data(), 32); uint8_t r = 0xAA;
                if (ufsecp_gpu_xonly_validate(cand, k, 1, &r) == UFSECP_OK) { g = cand; break; }
            }
            if (cand) ufsecp_gpu_ctx_destroy(cand);
        }
        if (!g) {
            std::printf("  skip: direct GPU ABI kernels — no GPU device implements them\n");
        } else {
            /* 9a: xonly_validate */
            const size_t M = 256;
            std::vector<uint8_t> keys(M*32), gr(M, 0xAA);
            for (size_t i = 0; i < M; ++i) {
                if (i & 1) std::memset(keys.data()+i*32, 0xFF, 32);                 /* >=p invalid */
                else       std::memcpy(keys.data()+i*32, ix.data()+(i%N)*32, 32);   /* valid */
            }
            bool xok = ufsecp_gpu_xonly_validate(g, keys.data(), M, gr.data()) == UFSECP_OK;
            for (size_t i = 0; xok && i < M; ++i) {
                secp256k1_xonly_pubkey xp;
                bool shim = secp256k1_xonly_pubkey_parse(sctx, &xp, keys.data()+i*32) == 1;
                if ((gr[i]==1) != shim) xok = false;
            }
            CHECK(xok, "GPU ufsecp_gpu_xonly_validate matches shim xonly_parse per key");

            /* 9b: commitment_verify (all-valid matches shim, then one corruption rejects) */
            std::vector<uint8_t> cr(N, 0xAA);
            bool cok = ufsecp_gpu_commitment_verify(g, ix.data(), tw.data(), tx.data(),
                                                    par.data(), N, cr.data()) == UFSECP_OK;
            for (size_t i = 0; cok && i < N; ++i) {
                int shim = secp256k1_xonly_pubkey_tweak_add_check(sctx, tx.data()+i*32, par[i],
                                                                  &ixo[i], tw.data()+i*32);
                if ((cr[i]==1) != (shim==1)) cok = false;
            }
            CHECK(cok, "GPU ufsecp_gpu_commitment_verify matches shim tweak_add_check per row");
            uint8_t sv = tx[5*32]; tx[5*32] ^= 0x01;
            ufsecp_gpu_commitment_verify(g, ix.data(), tw.data(), tx.data(), par.data(), N, cr.data());
            bool crej = (cr[5]==0 && cr[4]==1 && cr[6]==1);
            tx[5*32] = sv;
            CHECK(crej, "GPU commitment_verify: corrupted row rejects, neighbours accept");

            /* 9c: tagged_hash (TapBranch) vs shim tagged_sha256 */
            const size_t TM = 128, ML = 64;
            std::vector<uint8_t> tmsg(TM*ML), tout(TM*32);
            for (size_t i = 0; i < TM*ML; ++i) tmsg[i] = (uint8_t)((i*40503u) >> (i%24));
            uint8_t th[32]; ufsecp_sha256((const uint8_t*)"TapBranch", 9, th);
            bool tok = ufsecp_gpu_tagged_hash(g, th, tmsg.data(), ML, TM, tout.data()) == UFSECP_OK;
            for (size_t i = 0; tok && i < TM; ++i) {
                uint8_t ref[32];
                secp256k1_tagged_sha256(sctx, ref, (const unsigned char*)"TapBranch", 9,
                                        tmsg.data()+i*ML, ML);
                if (std::memcmp(tout.data()+i*32, ref, 32) != 0) tok = false;
            }
            CHECK(tok, "GPU ufsecp_gpu_tagged_hash TapBranch == shim tagged_sha256 per item");

            ufsecp_gpu_ctx_destroy(g);
        }
    }

    /* (10) full compressed-pubkey validation vs shim ec_pubkey_parse */
    {
        const size_t M = 256, STR = 33;
        std::vector<uint8_t> keys(M*STR, 0);
        for (size_t i = 0; i < M; ++i) {
            if (i & 1) { keys[i*STR] = 0x02; std::memset(keys.data()+i*STR+1, 0xFF, 32); } /* x>=p */
            else { uint8_t sk[32]={0}; sk[24]=(uint8_t)(i>>8); sk[31]=(uint8_t)(i|1u);
                   if (ufsecp_pubkey_create(uctx, sk, keys.data()+i*STR) != UFSECP_OK) ++g_fail; } /* valid */
        }
        std::vector<uint8_t> res(M, 0xAA);
        ufsecp_lbtc_validate_pubkeys(ctrl, keys.data(), M, STR, res.data());
        bool ok = true;
        for (size_t i = 0; i < M; ++i) {
            secp256k1_pubkey pk;
            bool shim = secp256k1_ec_pubkey_parse(sctx, &pk, keys.data()+i*STR, 33) == 1;
            if ((res[i]==1) != shim) ok = false;
        }
        CHECK(ok, "validate_pubkeys: valid accept, bad reject, matches shim ec_pubkey_parse");
    }

    /* (11) per-item-length tagged hash (TapLeaf) vs shim tagged_sha256 */
    {
        const size_t M = 1000, STR = 256;
        std::vector<uint8_t> msgs(M*STR), out(M*32); std::vector<uint32_t> lens(M);
        for (size_t i = 0; i < M; ++i) { lens[i] = 32u + (uint32_t)(i % 200);
            for (uint32_t b = 0; b < lens[i]; ++b) msgs[i*STR+b] = (uint8_t)((i*131u+b*7u)>>(b%19)); }
        ufsecp_lbtc_tagged_hash_var(ctrl, "TapLeaf", msgs.data(), lens.data(), STR, M, out.data());
        bool ok = true;
        for (size_t i = 0; i < M; ++i) { uint8_t ref[32];
            secp256k1_tagged_sha256(sctx, ref, (const unsigned char*)"TapLeaf", 7, msgs.data()+i*STR, lens[i]);
            if (std::memcmp(out.data()+i*32, ref, 32) != 0) ok = false; }
        CHECK(ok, "tagged_hash_var TapLeaf (per-item len) == shim tagged_sha256");
    }

    /* (12) batch HASH256 of 64-byte pairs (merkle) vs SHA256d reference */
    {
        const size_t M = 1024, IL = 64;
        std::vector<uint8_t> in(M*IL), out(M*32);
        for (size_t i = 0; i < M*IL; ++i) in[i] = (uint8_t)((i*40503u)>>(i%19));
        ufsecp_lbtc_hash256(ctrl, in.data(), IL, M, out.data());
        bool ok = true;
        for (size_t i = 0; i < M; ++i) { uint8_t h1[32], h2[32];
            if (ufsecp_sha256(in.data()+i*IL, IL, h1) != UFSECP_OK ||
                ufsecp_sha256(h1, 32, h2) != UFSECP_OK) { ok = false; break; }
            if (std::memcmp(out.data()+i*32, h2, 32) != 0) ok = false; }
        CHECK(ok, "hash256 (merkle 64B) == SHA256d reference");
    }

    /* (13) Aggregate (RLC) Schnorr batch verify. All-valid -> 1; any corruption -> 0.
     * Fiat-Shamir weights mean a corrupted batch cannot false-accept. Every generated
     * sig is also confirmed valid by the shim, so a "1" verdict is sound ground truth. */
    {
        const size_t M = 1000;
        std::vector<uint8_t> sm(M*32), sx(M*32), ss(M*64);
        int gen_ok = 1;
        for (size_t i = 0; i < M; ++i) {
            uint8_t sk[32]={0}; sk[23]=(uint8_t)(i>>16); sk[24]=(uint8_t)(i>>8); sk[31]=(uint8_t)(i|1u);
            uint8_t comp[33];
            if (ufsecp_pubkey_create(uctx, sk, comp) != UFSECP_OK) { gen_ok = 0; break; }
            std::memcpy(sx.data()+i*32, comp+1, 32);
            for (int b = 0; b < 32; ++b) sm[i*32+b] = (uint8_t)((i*2654435761u+b*7u)>>(b%24));
            uint8_t aux[32]; for (int b = 0; b < 32; ++b) aux[b] = (uint8_t)(i*131u+b);
            if (ufsecp_schnorr_sign(uctx, sm.data()+i*32, sk, aux, ss.data()+i*64) != UFSECP_OK) { gen_ok = 0; break; }
        }
        CHECK(gen_ok == 1, "aggregate setup: generated valid Schnorr signatures");

        int agg = ufsecp_lbtc_schnorr_aggregate_verify(ctrl, sm.data(), sx.data(), ss.data(), M);
        if (agg < 0) {
            std::printf("  skip: schnorr_aggregate_verify — no GPU (returned -1)\n");
        } else {
            CHECK(agg == 1, "aggregate: all-valid batch returns 1");

            bool all_shim = true;                                /* sigs are genuinely valid per shim */
            for (size_t i = 0; i < M; ++i) {
                secp256k1_xonly_pubkey xp;
                if (!secp256k1_xonly_pubkey_parse(sctx, &xp, sx.data()+i*32)) { all_shim = false; continue; }
                if (!secp256k1_schnorrsig_verify(sctx, ss.data()+i*64, sm.data()+i*32, 32, &xp)) all_shim = false;
            }
            CHECK(all_shim, "aggregate: every generated sig passes shim schnorrsig_verify (1 is sound)");

            uint8_t sv = ss[5*64+40]; ss[5*64+40] ^= 0x01;       /* corrupt s of one sig */
            CHECK(ufsecp_lbtc_schnorr_aggregate_verify(ctrl, sm.data(), sx.data(), ss.data(), M) == 0,
                  "aggregate: corrupted s rejects (returns 0)");
            ss[5*64+40] = sv;

            uint8_t rv = ss[800*64+3]; ss[800*64+3] ^= 0x80;     /* corrupt R.x of a different sig */
            CHECK(ufsecp_lbtc_schnorr_aggregate_verify(ctrl, sm.data(), sx.data(), ss.data(), M) == 0,
                  "aggregate: corrupted R.x rejects (returns 0)");
            ss[800*64+3] = rv;

            uint8_t mv = sm[100*32+9]; sm[100*32+9] ^= 0x01;     /* corrupt a message */
            CHECK(ufsecp_lbtc_schnorr_aggregate_verify(ctrl, sm.data(), sx.data(), ss.data(), M) == 0,
                  "aggregate: corrupted message rejects (returns 0)");
            sm[100*32+9] = mv;

            CHECK(ufsecp_lbtc_schnorr_aggregate_verify(ctrl, sm.data(), sx.data(), ss.data(), M) == 1,
                  "aggregate: restored batch is valid again (1)");
        }
    }

    ufsecp_lbtc_ctrl_destroy(ctrl);
    ufsecp_ctx_destroy(uctx);
    secp256k1_context_destroy(sctx);
    std::printf("\n%s\n", g_fail==0 ? "ALL PASS" : "FAILURES PRESENT");
    return g_fail==0 ? 0 : 1;
}
