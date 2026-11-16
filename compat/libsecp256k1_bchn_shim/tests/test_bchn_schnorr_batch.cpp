// BCHN shim: secp256k1_schnorr_verify_batch contract (Knuth PR #826, issue #466).
// Uses only the public shim API: keys from secp256k1_ec_pubkey_create,
// signatures from secp256k1_schnorr_sign.
#include "secp256k1.h"
#include "secp256k1_schnorr.h"

#include <cstdio>
#include <cstring>
#include <vector>

static int g_fail = 0;
static void check(bool cond, const char* msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL: %s\n", msg);
    }
}

int main() {
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_NONE);
    if (!ctx) {
        std::printf("FAIL: context_create\n");
        return 1;
    }

    const std::size_t n = 48;
    std::vector<unsigned char> sigs(n * 64), msgs(n * 32);
    std::vector<secp256k1_pubkey> pks(n);
    for (std::size_t i = 0; i < n; ++i) {
        unsigned char sk[32];
        std::memset(sk, 0, sizeof(sk));
        sk[31] = static_cast<unsigned char>(i + 1);
        sk[0] = 0x42;
        for (std::size_t j = 0; j < 32; ++j) msgs[i * 32 + j] = static_cast<unsigned char>(i * 7 + j);
        check(secp256k1_ec_pubkey_create(ctx, &pks[i], sk) == 1, "pubkey_create");
        check(secp256k1_schnorr_sign(ctx, &sigs[i * 64], &msgs[i * 32], sk, nullptr, nullptr) == 1,
              "schnorr_sign");
        check(secp256k1_schnorr_verify(ctx, &sigs[i * 64], &msgs[i * 32], &pks[i]) == 1,
              "single verify of an honest signature");
    }

    std::vector<const unsigned char*> sp(n), mp(n);
    std::vector<const secp256k1_pubkey*> pp(n);
    for (std::size_t i = 0; i < n; ++i) {
        sp[i] = &sigs[i * 64];
        mp[i] = &msgs[i * 32];
        pp[i] = &pks[i];
    }
    unsigned char seed[32];
    std::memset(seed, 0x77, sizeof(seed));

    check(secp256k1_schnorr_verify_batch(ctx, sp.data(), mp.data(), pp.data(), n, nullptr) == 1,
          "honest batch, no seed");
    check(secp256k1_schnorr_verify_batch(ctx, sp.data(), mp.data(), pp.data(), n, seed) == 1,
          "honest batch, with seed");
    check(secp256k1_schnorr_verify_batch(ctx, nullptr, nullptr, nullptr, 0, nullptr) == 1, "n == 0 -> 1");
    check(secp256k1_schnorr_verify_batch(ctx, nullptr, mp.data(), pp.data(), n, nullptr) == 0,
          "NULL sig array -> 0");
    check(secp256k1_schnorr_verify_batch(ctx, sp.data(), nullptr, pp.data(), n, nullptr) == 0,
          "NULL msg array -> 0");
    check(secp256k1_schnorr_verify_batch(ctx, sp.data(), mp.data(), nullptr, n, nullptr) == 0,
          "NULL pubkey array -> 0");

    // One bad entry at every position (small batch and MSM-sized batch).
    for (std::size_t m : {std::size_t{3}, n}) {
        for (std::size_t pos = 0; pos < m; ++pos) {
            sigs[pos * 64 + 40] ^= 0x01;
            check(secp256k1_schnorr_verify_batch(ctx, sp.data(), mp.data(), pp.data(), m, nullptr) == 0,
                  "batch with one corrupted signature -> 0");
            sigs[pos * 64 + 40] ^= 0x01;
        }
    }
    check(secp256k1_schnorr_verify_batch(ctx, sp.data(), mp.data(), pp.data(), n, nullptr) == 1,
          "restored batch verifies again");

    secp256k1_context_destroy(ctx);
    if (g_fail == 0) std::printf("PASS: bchn_schnorr_batch\n");
    return g_fail == 0 ? 0 : 1;
}
