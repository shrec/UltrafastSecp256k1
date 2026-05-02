// ============================================================================
// bench_vs_libsecp.cpp — Ultra vs libsecp256k1 quick comparison
// ============================================================================
// Runs only sign/verify, prints ratio table. ~25s vs bench_unified ~5min.
//
//   bench_vs_libsecp                  # 11 passes, warmup
//   bench_vs_libsecp --passes 7       # 7 passes, faster
//   bench_vs_libsecp --no-warmup      # skip warmup (~10s total)
// ============================================================================

#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ct/sign.hpp"
#include "secp256k1/benchmark_harness.hpp"
#include "secp256k1/precompute.hpp"

#include "secp256k1.h"
#include "secp256k1_extrakeys.h"
#include "secp256k1_schnorrsig.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <cstdlib>

using namespace secp256k1;
using namespace secp256k1::fast;

static constexpr std::size_t POOL = 64;
static constexpr int         N    = 200;   // iterations per pass

static void print_row(const char* name, double u, double lib) {
    double ratio = lib / u;
    printf("  %-34s  %8.1f  %9.1f  %6.2fx %s\n",
           name, u, lib, ratio, ratio >= 1.0 ? "Ultra ✓" : "libsecp");
}

int main(int argc, char** argv) {
    int  passes    = 11;
    bool no_warmup = false;

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--passes") && i + 1 < argc)
            passes = std::max(3, std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--no-warmup"))
            no_warmup = true;
    }

    bench::pin_thread_and_elevate();
    bench::Harness H(500, static_cast<std::size_t>(passes));
    configure_fixed_base({});

    // Warmup: run a few hundred k*G to stabilize CPU frequency
    if (!no_warmup) {
        printf("  warmup...");
        fflush(stdout);
        Scalar s = Scalar::from_bytes({1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                       0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5});
        for (int i = 0; i < 5000; ++i) {
            auto p = Point::generator().scalar_mul(s);
            bench::DoNotOptimize(p);
            s += Scalar::one();
        }
        printf(" done\n");
    }

    // Key/message pool
    std::array<Scalar,       POOL> sk;
    std::array<std::array<uint8_t,32>, POOL> msg, aux;
    std::array<SchnorrKeypair,    POOL> kps;
    std::array<std::array<uint8_t,32>, POOL> schnorr_pk;
    std::array<ECDSASignature,    POOL> ecdsa_sigs;
    std::array<SchnorrSignature,  POOL> schnorr_sigs;
    std::array<Point,             POOL> pubkeys;

    for (int i = 0; i < POOL; ++i) {
        std::array<uint8_t,32> kb{}; kb[31]=(uint8_t)(i+1); kb[0]=(uint8_t)(0x11+i);
        sk[i] = Scalar::from_bytes(kb);
        for (int j=0;j<32;j++) msg[i][j]=(uint8_t)(i*3+j);
        for (int j=0;j<32;j++) aux[i][j]=(uint8_t)(i+j+7);
        kps[i]          = schnorr_keypair_create(sk[i]);
        schnorr_pk[i]   = kps[i].px;
        pubkeys[i]      = Point::generator().scalar_mul(sk[i]);
        ecdsa_sigs[i]   = ecdsa_sign(msg[i], sk[i]);
        schnorr_sigs[i] = schnorr_sign(kps[i], msg[i], aux[i]);
    }

    // libsecp context + pools
    secp256k1_context* lctx = secp256k1_context_create(
        SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    std::array<secp256k1_pubkey,         POOL> lpubs;
    std::array<secp256k1_ecdsa_signature,POOL> lecdsa;
    std::array<secp256k1_keypair,        POOL> lkps;
    std::array<std::array<uint8_t,64>,   POOL> lschnorr;
    std::array<std::array<uint8_t,32>,   POOL> lxonly;

    for (int i = 0; i < POOL; ++i) {
        std::array<uint8_t,32> kb{}; kb[31]=(uint8_t)(i+1); kb[0]=(uint8_t)(0x11+i);
        secp256k1_ec_pubkey_create(lctx, &lpubs[i], kb.data());
        secp256k1_ecdsa_sign(lctx, &lecdsa[i], msg[i].data(), kb.data(), nullptr, nullptr);
        secp256k1_keypair_create(lctx, &lkps[i], kb.data());
        secp256k1_schnorrsig_sign32(lctx, lschnorr[i].data(), msg[i].data(), &lkps[i], aux[i].data());
        // x-only pubkey = bytes 1..33 of compressed pubkey
        std::array<uint8_t,33> comp; std::size_t sz=33;
        secp256k1_ec_pubkey_serialize(lctx,comp.data(),&sz,&lpubs[i],SECP256K1_EC_COMPRESSED);
        std::memcpy(lxonly[i].data(), comp.data()+1, 32);
    }

    printf("\n");
    printf("  =====================================================================\n");
    printf("  Ultra vs libsecp256k1 | %d passes | pool=%d | %d iters/pass\n",
           passes, POOL, N);
    printf("  ratio > 1.0 = Ultra wins\n");
    printf("  =====================================================================\n");
    printf("  %-34s  %8s  %9s  %7s\n",
           "Operation", "Ultra ns", "libsecp ns", "ratio");
    printf("  %-34s  %8s  %9s  %7s\n",
           "----------------------------------", "--------", "---------", "-------");

    int idx = 0;

    // ECDSA sign (CT vs CT — fair comparison)
    idx = 0;
    double u = H.run(N, [&]() {
        auto s = ct::ecdsa_sign(msg[idx%POOL], sk[idx%POOL]);
        bench::DoNotOptimize(s); ++idx;
    });
    idx = 0;
    double l = H.run(N, [&]() {
        secp256k1_ecdsa_signature s;
        std::array<uint8_t,32> kb{}; kb[31]=(uint8_t)((idx%POOL)+1); kb[0]=(uint8_t)(0x11+idx%POOL);
        secp256k1_ecdsa_sign(lctx,&s,msg[idx%POOL].data(),kb.data(),nullptr,nullptr);
        bench::DoNotOptimize(s); ++idx;
    });
    print_row("ECDSA sign (CT vs CT)", u, l);

    // ECDSA verify
    idx = 0;
    u = H.run(N, [&]() {
        bool ok = ecdsa_verify(msg[idx%POOL], pubkeys[idx%POOL], ecdsa_sigs[idx%POOL]);
        bench::DoNotOptimize(ok); ++idx;
    });
    idx = 0;
    l = H.run(N, [&]() {
        int ok = secp256k1_ecdsa_verify(lctx,&lecdsa[idx%POOL],msg[idx%POOL].data(),&lpubs[idx%POOL]);
        bench::DoNotOptimize(ok); ++idx;
    });
    print_row("ECDSA verify", u, l);

    printf("  %-34s\n", "");

    // Schnorr keypair create
    idx = 0;
    u = H.run(N, [&]() {
        auto kp = schnorr_keypair_create(sk[idx%POOL]);
        bench::DoNotOptimize(kp); ++idx;
    });
    idx = 0;
    l = H.run(N, [&]() {
        secp256k1_keypair kp;
        std::array<uint8_t,32> kb{}; kb[31]=(uint8_t)((idx%POOL)+1); kb[0]=(uint8_t)(0x11+idx%POOL);
        secp256k1_keypair_create(lctx,&kp,kb.data());
        bench::DoNotOptimize(kp); ++idx;
    });
    print_row("Schnorr keypair create", u, l);

    // Schnorr sign (CT vs CT)
    idx = 0;
    u = H.run(N, [&]() {
        auto s = ct::schnorr_sign(kps[idx%POOL], msg[idx%POOL], aux[idx%POOL]);
        bench::DoNotOptimize(s); ++idx;
    });
    idx = 0;
    l = H.run(N, [&]() {
        std::array<uint8_t,64> s;
        secp256k1_schnorrsig_sign32(lctx,s.data(),msg[idx%POOL].data(),&lkps[idx%POOL],aux[idx%POOL].data());
        bench::DoNotOptimize(s); ++idx;
    });
    print_row("Schnorr sign (CT vs CT)", u, l);

    // Schnorr verify — raw (different pubkey each time)
    idx = 0;
    u = H.run(N, [&]() {
        bool ok = schnorr_verify(schnorr_pk[idx%POOL].data(), msg[idx%POOL].data(), schnorr_sigs[idx%POOL]);
        bench::DoNotOptimize(ok); ++idx;
    });
    idx = 0;
    l = H.run(N, [&]() {
        secp256k1_xonly_pubkey xpk;
        secp256k1_xonly_pubkey_parse(lctx,&xpk,lxonly[idx%POOL].data());
        int ok = secp256k1_schnorrsig_verify(lctx,lschnorr[idx%POOL].data(),msg[idx%POOL].data(),32,&xpk);
        bench::DoNotOptimize(ok); ++idx;
    });
    print_row("Schnorr verify (raw, 64 keys)", u, l);

    // Schnorr verify — cached (same pubkey = lift_x cache hit)
    {
        auto cpk = schnorr_pk[0]; auto cmsg = msg[0]; auto csig = schnorr_sigs[0];
        u = H.run(N, [&]() {
            bool ok = schnorr_verify(cpk.data(), cmsg.data(), csig);
            bench::DoNotOptimize(ok);
        });
        secp256k1_xonly_pubkey xpk;
        secp256k1_xonly_pubkey_parse(lctx,&xpk,lxonly[0].data());
        l = H.run(N, [&]() {
            int ok = secp256k1_schnorrsig_verify(lctx,lschnorr[0].data(),cmsg.data(),32,&xpk);
            bench::DoNotOptimize(ok);
        });
        print_row("Schnorr verify (cached pubkey)", u, l);
    }

    // Schnorr verify — SchnorrXonlyPubkey with GLV table cache (P0 optimization)
    // Pre-parses pubkeys into SchnorrXonlyPubkey (builds tbl_P/tbl_phi once).
    // Eliminates ~1,954 ns of build_glv52_table_zr per verify.
    {
        // Parse 64 pubkeys into SchnorrXonlyPubkey with pre-built GLV tables
        std::array<secp256k1::SchnorrXonlyPubkey, POOL> xonly_pks;
        for (std::size_t i = 0; i < POOL; ++i)
            secp256k1::schnorr_xonly_pubkey_parse(xonly_pks[i], schnorr_pk[i].data());

        idx = 0;
        u = H.run(N, [&]() {
            bool ok = secp256k1::schnorr_verify(xonly_pks[idx%POOL], msg[idx%POOL].data(), schnorr_sigs[idx%POOL]);
            bench::DoNotOptimize(ok); ++idx;
        });
        // libsecp baseline: parse each time (no pre-built tables)
        idx = 0;
        l = H.run(N, [&]() {
            secp256k1_xonly_pubkey xpk;
            secp256k1_xonly_pubkey_parse(lctx,&xpk,lxonly[idx%POOL].data());
            int ok = secp256k1_schnorrsig_verify(lctx,lschnorr[idx%POOL].data(),msg[idx%POOL].data(),32,&xpk);
            bench::DoNotOptimize(ok); ++idx;
        });
        print_row("Schnorr verify (xonly+GLV cache)", u, l);
    }

    printf("  %-34s\n", "");

    // pubkey_create
    idx = 0;
    u = H.run(N, [&]() {
        auto p = Point::generator().scalar_mul(sk[idx%POOL]);
        bench::DoNotOptimize(p); ++idx;
    });
    idx = 0;
    l = H.run(N, [&]() {
        secp256k1_pubkey p;
        std::array<uint8_t,32> kb{}; kb[31]=(uint8_t)((idx%POOL)+1); kb[0]=(uint8_t)(0x11+idx%POOL);
        secp256k1_ec_pubkey_create(lctx,&p,kb.data());
        bench::DoNotOptimize(p); ++idx;
    });
    print_row("pubkey_create (k*G)", u, l);

    printf("  =====================================================================\n\n");
    secp256k1_context_destroy(lctx);
    return 0;
}
