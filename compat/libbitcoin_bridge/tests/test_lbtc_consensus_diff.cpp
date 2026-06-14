/**
 * test_lbtc_consensus_diff.cpp — GPU-vs-CPU consensus differential for the
 * libbitcoin batch script-signature verification bridge.
 *
 * The GPU path is consensus-bearing: for block validation it must agree with the
 * CPU reference path BIT-FOR-BIT on the accept/reject verdict of every signature,
 * including consensus edge cases (corrupted sig, off-curve pubkey, tampered
 * message, non-canonical encodings). The CPU path is itself gated bit-for-bit
 * against libsecp256k1 (cross_libsecp256k1 + reverse bridge), so GPU==CPU here
 * means GPU==libsecp transitively.
 *
 * This test builds one mixed corpus and verifies it through a GPU controller and
 * a CPU controller; ANY per-row verdict mismatch is a consensus failure. If no
 * GPU is present it advisory-skips (exit 77) per the GPU-local-only policy.
 *
 * Standalone:
 *   g++ -std=c++17 -I ../include -I ../../../include/ufsecp \
 *       test_lbtc_consensus_diff.cpp -lufsecp -o test_lbtc_consensus_diff
 */
#include "ufsecp_libbitcoin.h"
#include "ufsecp.h"

#ifdef UFSECP_LBTC_HAVE_LIBSECP
/* Direct libsecp256k1 leg: the same corpus is verified through the reference
 * library so the gate proves GPU==CPU==libsecp256k1 directly, not just GPU==CPU
 * with CPU==libsecp transitive. Defined by the build when the secp256k1_shim
 * target is on the test's link line. */
#include <secp256k1.h>
#include <secp256k1_schnorrsig.h>
#include <secp256k1_extrakeys.h>
#endif

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, msg)                                                       \
    do { if (!(cond)) { std::printf("  FAIL: %s\n", msg); ++g_fail; }           \
         else         { std::printf("  ok  : %s\n", msg); } } while (0)

namespace {

constexpr int ADVISORY_SKIP_CODE = 77;

void make_sk(uint8_t sk[32], uint32_t seed) {
    std::memset(sk, 0, 32);
    sk[28]=(uint8_t)(seed>>24); sk[29]=(uint8_t)(seed>>16);
    sk[30]=(uint8_t)(seed>>8);  sk[31]=(uint8_t)(seed|1u);
}
void make_msg(uint8_t msg[32], uint32_t seed) {
    for (int i=0;i<32;++i) msg[i]=(uint8_t)((seed*2654435761u)>>(i%24));
}

/* Build a corpus that mixes valid signatures with every consensus-relevant
 * rejection class. With ks > 0 each row carries a distinct non-zero correlation
 * id (= i+1, little-endian) in its trailing key cell — required for the collect
 * differential, where a surviving non-zero cell IS the rejected marker. */
enum Kind { ECDSA, SCHNORR };
std::vector<uint8_t> build_corpus(ufsecp_ctx* ctx, Kind k, size_t n, size_t ks=0) {
    const size_t rec = (k==ECDSA) ? UFSECP_LBTC_ECDSA_RECORD : UFSECP_LBTC_SCHNORR_RECORD;
    const size_t stride = rec + ks;
    std::vector<uint8_t> rows(n*stride, 0);
    for (size_t i=0;i<n;++i) {
        uint8_t* r = rows.data()+i*stride;
        uint8_t sk[32], msg[32], pub[33], aux[32]={0};
        make_sk(sk,(uint32_t)(i+1)); make_msg(msg,(uint32_t)(i+1));
        if (ufsecp_pubkey_create(ctx, sk, pub) != UFSECP_OK) ++g_fail;
        if (k==ECDSA) {
            std::memcpy(r, msg, 32); std::memcpy(r+32, pub, 33);
            if (ufsecp_ecdsa_sign(ctx, msg, sk, r+65) != UFSECP_OK) ++g_fail;
        } else {
            std::memcpy(r, msg, 32); std::memcpy(r+32, pub+1, 32); /* msg | xonly */
            if (ufsecp_schnorr_sign(ctx, msg, sk, aux, r+64) != UFSECP_OK) ++g_fail;
        }
        /* Inject a rejection class every few rows (deterministic mix). */
        const size_t sig_off = (k==ECDSA) ? 65 : 64;
        switch (i % 7) {
            case 0: break;                                  /* valid                */
            case 1: r[sig_off] ^= 0x01; break;              /* corrupted sig byte   */
            case 2: r[0] ^= 0x01; break;                    /* tampered msg (off 0) */
            case 3: r[sig_off+31] ^= 0x80; break;           /* high bit of r/R.x    */
            case 4: std::memset(r+sig_off, 0, 32); break;   /* zero r/R.x           */
            case 5: std::memset(r+sig_off+32, 0xff, 32); break; /* s = 0xff..ff (>=n) */
            case 6: r[32] ^= 0x01; break;                   /* flip pubkey byte: ECDSA prefix / Schnorr x-only (both @32) */
        }
        /* distinct non-zero id in the key cell (only used by the collect leg) */
        const uint64_t id = (uint64_t)i + 1u;
        for (size_t b=0;b<ks;++b) r[rec+b] = (uint8_t)(id >> (8*b));
    }
    return rows;
}

#ifdef UFSECP_LBTC_HAVE_LIBSECP
/* Per-row libsecp256k1 verdict on a packed bridge record: 1 = accept, 0 = reject.
 * A parse failure on a malformed row counts as reject (consensus: invalid). The
 * record layout matches the bridge: ECDSA [msg32|pub33|sig64], Schnorr
 * [msg32|xonly32|sig64]; the 64-byte sig is compact r||s (ECDSA) / R.x||s (Schnorr). */
int libsecp_verify_row(const secp256k1_context* ls, Kind k, const uint8_t* r) {
    if (k == ECDSA) {
        secp256k1_pubkey pk;
        if (!secp256k1_ec_pubkey_parse(ls, &pk, r + 32, 33)) return 0;
        secp256k1_ecdsa_signature sig;
        if (!secp256k1_ecdsa_signature_parse_compact(ls, &sig, r + 65)) return 0;
        return secp256k1_ecdsa_verify(ls, &sig, r + 0, &pk) == 1 ? 1 : 0;
    }
    secp256k1_xonly_pubkey xpk;
    if (!secp256k1_xonly_pubkey_parse(ls, &xpk, r + 32)) return 0;
    return secp256k1_schnorrsig_verify(ls, r + 64, r + 0, 32, &xpk) == 1 ? 1 : 0;
}
#endif

bool diff_kind(ufsecp_lbtc_ctrl* gpu, ufsecp_lbtc_ctrl* cpu, const void* ls_opaque,
               Kind k, size_t n) {
    const size_t rec = (k==ECDSA) ? UFSECP_LBTC_ECDSA_RECORD : UFSECP_LBTC_SCHNORR_RECORD;
    ufsecp_ctx* sctx=nullptr;
    if (ufsecp_ctx_create(&sctx)!=UFSECP_OK) return false;
    auto rows = build_corpus(sctx, k, n);
    ufsecp_ctx_destroy(sctx);

    std::vector<uint8_t> g_res(n), c_res(n);  /* zero-init: bridge writes 1=valid/0=invalid */
    if (k==ECDSA) {
        ufsecp_lbtc_verify_ecdsa(gpu, rows.data(), n, 0, g_res.data(), nullptr, 0, nullptr);
        ufsecp_lbtc_verify_ecdsa(cpu, rows.data(), n, 0, c_res.data(), nullptr, 0, nullptr);
    } else {
        ufsecp_lbtc_verify_schnorr(gpu, rows.data(), n, 0, g_res.data(), nullptr, 0, nullptr);
        ufsecp_lbtc_verify_schnorr(cpu, rows.data(), n, 0, c_res.data(), nullptr, 0, nullptr);
    }
    /* Invalid counts are derived from the per-row results — the bridge's only
     * output (it returns void; failures are mapped by the caller from results[]). */
    size_t g_ninv=0, c_ninv=0;
    for (size_t i=0;i<n;++i) { if (!g_res[i]) ++g_ninv; if (!c_res[i]) ++c_ninv; }
    const char* name = (k==ECDSA)?"ECDSA":"Schnorr";

    /* Per-row agreement across GPU, CPU and (when linked) libsecp256k1 is the
     * consensus property. The GPU path is consensus-bearing for block validation:
     * its accept/reject verdict for every signature must match the CPU reference
     * AND the ecosystem-standard libsecp256k1 bit-for-bit. */
    bool have_ls = false;
#ifdef UFSECP_LBTC_HAVE_LIBSECP
    have_ls = (ls_opaque != nullptr);
#else
    (void)ls_opaque;
#endif
    size_t mismatch = 0, first = (size_t)-1, ls_ninv = 0;
    for (size_t i=0;i<n;++i) {
        int gv = g_res[i], cv = c_res[i], lv = c_res[i];
#ifdef UFSECP_LBTC_HAVE_LIBSECP
        if (have_ls) {
            lv = libsecp_verify_row((const secp256k1_context*)ls_opaque, k, rows.data()+i*rec);
            if (lv == 0) ++ls_ninv;
        }
#endif
        if (gv != cv || gv != lv) { if (first==(size_t)-1) first=i; ++mismatch; }
    }
    if (mismatch) {
        int lf = -1;
#ifdef UFSECP_LBTC_HAVE_LIBSECP
        if (have_ls) lf = libsecp_verify_row((const secp256k1_context*)ls_opaque, k, rows.data()+first*rec);
#endif
        std::printf("  CONSENSUS DIVERGENCE [%s]: %zu/%zu rows differ (first @%zu: gpu=%d cpu=%d libsecp=%d)\n",
                    name, mismatch, n, first, g_res[first], c_res[first], lf);
        return false;
    }
    if (have_ls) {
        if (ls_ninv != c_ninv) {
            std::printf("  CONSENSUS DIVERGENCE [%s]: libsecp invalid-count %zu != CPU %zu\n",
                        name, ls_ninv, c_ninv);
            return false;
        }
        std::printf("  %s: GPU==CPU==libsecp256k1 bit-for-bit on %zu rows (%zu rejected); counts %zu==%zu==%zu\n",
                    name, n, c_ninv, g_ninv, c_ninv, ls_ninv);
    } else {
        std::printf("  %s: GPU==CPU bit-for-bit on %zu rows (%zu rejected); counts %zu==%zu  [libsecp leg not linked]\n",
                    name, n, c_ninv, g_ninv, c_ninv);
    }
    return (g_ninv==c_ninv);
}

/* Collect-mode consensus differential: same mixed corpus, but each row carries a
 * non-zero id and the verdict is collapsed IN PLACE into the key cell. The
 * rejected set is {i : key cell non-zero} after the call; it must match across
 * GPU, CPU and (when linked) libsecp256k1 — proving the in-place collect channel
 * is consensus-identical to the results channel. */
bool diff_kind_collect(ufsecp_lbtc_ctrl* gpu, ufsecp_lbtc_ctrl* cpu, const void* ls_opaque,
                       Kind k, size_t n) {
    const size_t KS = 4;
    const size_t rec = (k==ECDSA) ? UFSECP_LBTC_ECDSA_RECORD : UFSECP_LBTC_SCHNORR_RECORD;
    const size_t stride = rec + KS;
    ufsecp_ctx* sctx=nullptr;
    if (ufsecp_ctx_create(&sctx)!=UFSECP_OK) return false;
    auto base = build_corpus(sctx, k, n, KS);
    ufsecp_ctx_destroy(sctx);

    /* two mutable copies — collect writes the key cells in place */
    std::vector<uint8_t> g_rows = base, c_rows = base;
    if (k==ECDSA) {
        ufsecp_lbtc_verify_ecdsa_collect(gpu, g_rows.data(), n, KS);
        ufsecp_lbtc_verify_ecdsa_collect(cpu, c_rows.data(), n, KS);
    } else {
        ufsecp_lbtc_verify_schnorr_collect(gpu, g_rows.data(), n, KS);
        ufsecp_lbtc_verify_schnorr_collect(cpu, c_rows.data(), n, KS);
    }
    auto key_nonzero = [&](const std::vector<uint8_t>& v, size_t i) {
        const uint8_t* cell = v.data() + i*stride + rec;
        for (size_t b=0;b<KS;++b) if (cell[b]) return true; return false;
    };
    const char* name = (k==ECDSA)?"ECDSA":"Schnorr";

    bool have_ls = false;
#ifdef UFSECP_LBTC_HAVE_LIBSECP
    have_ls = (ls_opaque != nullptr);
#else
    (void)ls_opaque;
#endif
    size_t mismatch=0, first=(size_t)-1, g_rej=0, c_rej=0, ls_rej=0;
    for (size_t i=0;i<n;++i) {
        const bool gz = key_nonzero(g_rows, i);   /* GPU rejected row i  */
        const bool cz = key_nonzero(c_rows, i);   /* CPU rejected row i  */
        bool lz = cz;
#ifdef UFSECP_LBTC_HAVE_LIBSECP
        if (have_ls) {
            /* records are untouched by collect — verify from the GPU copy */
            lz = libsecp_verify_row((const secp256k1_context*)ls_opaque, k,
                                    g_rows.data()+i*stride) == 0;
        }
#endif
        if (gz) ++g_rej; if (cz) ++c_rej; if (lz) ++ls_rej;
        if (gz != cz || gz != lz) { if (first==(size_t)-1) first=i; ++mismatch; }
    }
    if (mismatch) {
        std::printf("  COLLECT DIVERGENCE [%s]: %zu/%zu rows differ (first @%zu)\n",
                    name, mismatch, n, first);
        return false;
    }
    /* Stronger than the rejected-set counts: after collect the GPU and CPU row
     * buffers must be byte-for-byte identical — records are untouched, valid key
     * cells are zeroed in both, invalid ids are preserved in both. This catches
     * any in-place-write or per-row divergence the counts alone could miss (e.g.
     * the GPU zeroing the WRONG rows but the same NUMBER of rows). */
    if (g_rows != c_rows) {
        std::printf("  COLLECT DIVERGENCE [%s]: GPU and CPU key buffers differ byte-for-byte\n", name);
        return false;
    }
    if (have_ls)
        std::printf("  %s collect: GPU==CPU==libsecp rejected-set on %zu rows (%zu rejected); %zu==%zu==%zu (buffers byte-equal)\n",
                    name, n, c_rej, g_rej, c_rej, ls_rej);
    else
        std::printf("  %s collect: GPU==CPU rejected-set on %zu rows (%zu rejected); %zu==%zu (buffers byte-equal)  [libsecp leg not linked]\n",
                    name, n, c_rej, g_rej, c_rej);
    return (g_rej==c_rej) && (!have_ls || ls_rej==c_rej);
}

} // namespace

int test_lbtc_consensus_diff_run() {
    ufsecp_lbtc_ctrl *gpu=nullptr, *cpu=nullptr;
    if (ufsecp_lbtc_ctrl_create(&gpu, UFSECP_LBTC_GPU)!=UFSECP_OK || !gpu) {
        std::printf("[lbtc_consensus_diff] no GPU backend — advisory skip\n");
        return ADVISORY_SKIP_CODE;
    }
    if (ufsecp_lbtc_ctrl_create(&cpu, UFSECP_LBTC_CPU)!=UFSECP_OK || !cpu) {
        std::printf("FATAL: CPU controller create failed\n");
        ufsecp_lbtc_ctrl_destroy(gpu); return 1;
    }
    const char* names[]={"CPU","CUDA","OpenCL","Metal"};
    std::printf("GPU-vs-CPU consensus differential (gpu=%s, cpu=%s)\n",
                names[ufsecp_lbtc_ctrl_backend(gpu)], names[ufsecp_lbtc_ctrl_backend(cpu)]);

    const void* ls = nullptr;
#ifdef UFSECP_LBTC_HAVE_LIBSECP
    secp256k1_context* lsctx = secp256k1_context_create(SECP256K1_CONTEXT_VERIFY);
    ls = lsctx;
    std::printf("  direct libsecp256k1 leg: ENABLED (3-way GPU==CPU==libsecp)\n");
#else
    std::printf("  direct libsecp256k1 leg: not linked — GPU==CPU only (libsecp transitive)\n");
#endif

    const size_t N = 20000;
    CHECK(diff_kind(gpu, cpu, ls, ECDSA,   N), "ECDSA  GPU==CPU==libsecp consensus (mixed corpus)");
    CHECK(diff_kind(gpu, cpu, ls, SCHNORR, N), "Schnorr GPU==CPU==libsecp consensus (mixed corpus)");
    /* in-place collect channel must be consensus-identical to the results channel */
    CHECK(diff_kind_collect(gpu, cpu, ls, ECDSA,   N), "ECDSA  collect GPU==CPU==libsecp rejected-set");
    CHECK(diff_kind_collect(gpu, cpu, ls, SCHNORR, N), "Schnorr collect GPU==CPU==libsecp rejected-set");

#ifdef UFSECP_LBTC_HAVE_LIBSECP
    secp256k1_context_destroy(lsctx);
#endif
    ufsecp_lbtc_ctrl_destroy(cpu);
    ufsecp_lbtc_ctrl_destroy(gpu);
    std::printf("\n%s\n", g_fail==0 ? "CONSENSUS OK (GPU == CPU bit-for-bit)" : "CONSENSUS FAILURES");
    return g_fail==0 ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() { return test_lbtc_consensus_diff_run(); }
#endif
