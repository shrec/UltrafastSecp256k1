/* ============================================================================
 * UltrafastSecp256k1 -- GPU Ops Equivalence Test
 * ============================================================================
 * For each first-wave GPU operation, compares GPU result against CPU reference.
 * Only tests ops that the current backend actually supports (not UNSUPPORTED).
 *
 * Ops tested (where supported):
 *   1. generator_mul_batch   -- k*G
 *   2. ecdsa_verify_batch    -- ECDSA verification
 *   3. schnorr_verify_batch  -- BIP-340 Schnorr verification
 *   3b. bch_schnorr_verify_batch -- Bitcoin Cash 2019 Schnorr verification
 *       (valid + every invalid class, mixed batch, vs ufsecp_bch_schnorr_verify)
 *   4. ecdh_batch            -- ECDH shared secret
 *   5. hash160_pubkey_batch  -- RIPEMD160(SHA256(pubkey))
 *   6. msm                   -- multi-scalar multiplication
 *
 * Requires at least one GPU backend with devices. Skips gracefully otherwise.
 * ============================================================================ */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <array>
#include <vector>

#include "ufsecp/ufsecp_gpu.h"
#include "ufsecp/ufsecp.h"

static int g_pass = 0;
static int g_fail = 0;
static int g_skip = 0;

#define CHECK(cond, msg)                                            \
    do {                                                            \
        if (cond) { ++g_pass; }                                     \
        else { ++g_fail; std::printf("  FAIL: %s\n", msg); }       \
    } while (0)

#define CHECK_OK(expr, msg) CHECK((expr) == UFSECP_OK, msg)

#define SKIP(msg)                                                   \
    do { ++g_skip; std::printf("  SKIP: %s\n", msg); } while (0)

/* Deterministic pseudo-random bytes for test vectors */
static void fill_deterministic(uint8_t* buf, size_t len, uint8_t seed) {
    uint32_t state = seed;
    for (size_t i = 0; i < len; ++i) {
        state = state * 1103515245u + 12345u;
        buf[i] = static_cast<uint8_t>((state >> 16) & 0xFF);
    }
}

/* ============================================================================
 * 1. generator_mul_batch equivalence
 * ============================================================================ */
static void test_generator_mul_equiv(ufsecp_gpu_ctx* ctx) {
    std::printf("[gpu_equiv] generator_mul_batch\n");

    constexpr size_t N = 8;
    uint8_t scalars[N * 32];
    fill_deterministic(scalars, sizeof(scalars), 0x42);
    /* Ensure scalars are valid (set high byte < 0xFF to stay in range) */
    for (size_t i = 0; i < N; ++i) scalars[i * 32] &= 0x7F;

    uint8_t gpu_pubs[N * 33];
    auto err = ufsecp_gpu_generator_mul_batch(ctx, scalars, N, gpu_pubs);
    if (err == UFSECP_ERR_GPU_UNSUPPORTED) { SKIP("generator_mul_batch unsupported"); return; }
    CHECK(err == UFSECP_OK, "GPU generator_mul_batch succeeds");
    if (err != UFSECP_OK) return;

    /* CPU reference */
    ufsecp_ctx* cpu_ctx = nullptr;
    CHECK_OK(ufsecp_ctx_create(&cpu_ctx), "CPU ctx_create succeeds");

    for (size_t i = 0; i < N; ++i) {
        uint8_t cpu_pub[33];
        auto cerr = ufsecp_pubkey_create(cpu_ctx, scalars + i * 32, cpu_pub);
        CHECK(cerr == UFSECP_OK, "CPU pubkey_create succeeds");
        if (cerr == UFSECP_OK) {
            char msg[128];
            std::snprintf(msg, sizeof(msg), "generator_mul[%zu] GPU == CPU", i);
            CHECK(std::memcmp(gpu_pubs + i * 33, cpu_pub, 33) == 0, msg);
        }
    }
    ufsecp_ctx_destroy(cpu_ctx);
}

/* ============================================================================
 * 2. ecdsa_verify_batch equivalence
 * ============================================================================ */
static void test_ecdsa_verify_equiv(ufsecp_gpu_ctx* ctx) {
    std::printf("[gpu_equiv] ecdsa_verify_batch\n");

    constexpr size_t N = 4;
    ufsecp_ctx* cpu_ctx = nullptr;
    CHECK_OK(ufsecp_ctx_create(&cpu_ctx), "CPU ctx_create succeeds");

    uint8_t msg_hashes[N * 32];
    uint8_t privkeys[N * 32];
    uint8_t pubkeys[N * 33];
    uint8_t sigs[N * 64];

    fill_deterministic(msg_hashes, sizeof(msg_hashes), 0xAA);
    fill_deterministic(privkeys, sizeof(privkeys), 0xBB);

    /* Generate keys and sign with CPU */
    for (size_t i = 0; i < N; ++i) {
        privkeys[i * 32] &= 0x7F;
        CHECK_OK(ufsecp_pubkey_create(cpu_ctx, privkeys + i * 32, pubkeys + i * 33),
                 "CPU pubkey_create succeeds");

        CHECK_OK(ufsecp_ecdsa_sign(cpu_ctx, msg_hashes + i * 32, privkeys + i * 32,
                 sigs + i * 64),
                 "CPU ecdsa_sign succeeds");
    }

    /* GPU verify */
    uint8_t gpu_results[N];
    auto err = ufsecp_gpu_ecdsa_verify_batch(ctx, msg_hashes, pubkeys, sigs, N, gpu_results);
    if (err == UFSECP_ERR_GPU_UNSUPPORTED) { SKIP("ecdsa_verify_batch unsupported"); goto cleanup; }
    CHECK(err == UFSECP_OK, "GPU ecdsa_verify_batch succeeds");
    if (err != UFSECP_OK) goto cleanup;

    /* CPU verify for comparison */
    for (size_t i = 0; i < N; ++i) {
        auto cerr = ufsecp_ecdsa_verify(cpu_ctx, msg_hashes + i * 32,
                                         sigs + i * 64, pubkeys + i * 33);
        uint8_t cpu_ok = (cerr == UFSECP_OK) ? 1 : 0;
        char msg[128];
        std::snprintf(msg, sizeof(msg), "ecdsa_verify[%zu] GPU(%d) == CPU(%d)", i,
                      gpu_results[i], cpu_ok);
        CHECK(gpu_results[i] == cpu_ok, msg);
    }

cleanup:
    ufsecp_ctx_destroy(cpu_ctx);
}

/* ============================================================================
 * 3. schnorr_verify_batch equivalence
 * ============================================================================ */
static void test_schnorr_verify_equiv(ufsecp_gpu_ctx* ctx) {
    std::printf("[gpu_equiv] schnorr_verify_batch\n");

    constexpr size_t N = 4;
    ufsecp_ctx* cpu_ctx = nullptr;
    CHECK_OK(ufsecp_ctx_create(&cpu_ctx), "CPU ctx_create succeeds");

    uint8_t msg_hashes[N * 32];
    uint8_t privkeys[N * 32];
    uint8_t pubkeys_x[N * 32];
    uint8_t sigs[N * 64];

    fill_deterministic(msg_hashes, sizeof(msg_hashes), 0xCC);
    fill_deterministic(privkeys, sizeof(privkeys), 0xDD);

    /* Generate keypairs and sign with CPU */
    for (size_t i = 0; i < N; ++i) {
        privkeys[i * 32] &= 0x7F;
        uint8_t xonly[32];
        CHECK_OK(ufsecp_pubkey_xonly(cpu_ctx, privkeys + i * 32, xonly),
                 "CPU pubkey_xonly succeeds");
        std::memcpy(pubkeys_x + i * 32, xonly, 32);

        uint8_t aux[32] = {};
        CHECK_OK(ufsecp_schnorr_sign(cpu_ctx, msg_hashes + i * 32, privkeys + i * 32,
                 aux, sigs + i * 64),
                 "CPU schnorr_sign succeeds");
    }

    /* GPU verify */
    uint8_t gpu_results[N];
    auto err = ufsecp_gpu_schnorr_verify_batch(ctx, msg_hashes, pubkeys_x, sigs, N, gpu_results);
    if (err == UFSECP_ERR_GPU_UNSUPPORTED) { SKIP("schnorr_verify_batch unsupported"); goto cleanup; }
    CHECK(err == UFSECP_OK, "GPU schnorr_verify_batch succeeds");
    if (err != UFSECP_OK) goto cleanup;

    for (size_t i = 0; i < N; ++i) {
        auto cerr = ufsecp_schnorr_verify(cpu_ctx, msg_hashes + i * 32,
                                           sigs + i * 64, pubkeys_x + i * 32);
        uint8_t cpu_ok = (cerr == UFSECP_OK) ? 1 : 0;
        char msg[128];
        std::snprintf(msg, sizeof(msg), "schnorr_verify[%zu] GPU(%d) == CPU(%d)", i,
                      gpu_results[i], cpu_ok);
        CHECK(gpu_results[i] == cpu_ok, msg);
    }

cleanup:
    ufsecp_ctx_destroy(cpu_ctx);
}

/* ============================================================================
 * 3b. bch_schnorr_verify_batch equivalence (Bitcoin Cash 2019 Schnorr)
 * ============================================================================
 * Every item's GPU verdict must equal ufsecp_bch_schnorr_verify, and must equal
 * the verdict the item was built to have. */

/* p and n, big-endian. */
static const uint8_t kBchFieldP[32] = {
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFE, 0xFF, 0xFF, 0xFC, 0x2F};
static const uint8_t kBchOrderN[32] = {
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFE, 0xBA, 0xAE, 0xDC, 0xE6, 0xAF, 0x48, 0xA0, 0x3B,
    0xBF, 0xD2, 0x5E, 0x8C, 0xD0, 0x36, 0x41, 0x41};

/* A BCH 2019 Schnorr signature built from C ABI primitives (the ABI has no BCH
 * signer): R = k*G, e = SHA256(r || P33 || m) mod n, s = k + e*x. Exactly one
 * of R, -R has a quadratic-residue y (p = 3 mod 4), and both have x = r, so the
 * two candidates s = e*x + k and s = e*x - k are tried against the CPU
 * verifier: the accepted one goes to sig_out, the other (whose R has a
 * non-residue y) to neg_sig_out. Returns false if the construction fails or
 * the CPU verifier does not accept exactly one candidate. */
static bool bch_test_sign(ufsecp_ctx* cpu, const uint8_t priv[32], const uint8_t msg[32],
                          const uint8_t nonce[32], const uint8_t pub33[33],
                          uint8_t sig_out[64], uint8_t neg_sig_out[64]) {
    uint8_t R33[33];
    if (ufsecp_pubkey_create(cpu, nonce, R33) != UFSECP_OK) return false;
    uint8_t chal[97];
    std::memcpy(chal, R33 + 1, 32);
    std::memcpy(chal + 32, pub33, 33);
    std::memcpy(chal + 65, msg, 32);
    uint8_t e[32];
    if (ufsecp_sha256(chal, sizeof(chal), e) != UFSECP_OK) return false;
    uint8_t ex[32];
    std::memcpy(ex, priv, 32);
    if (ufsecp_seckey_tweak_mul(cpu, ex, e) != UFSECP_OK) return false;  /* e >= n: retry seed */

    uint8_t cand[2][64];
    for (int c = 0; c < 2; ++c) {
        uint8_t k[32];
        std::memcpy(k, nonce, 32);
        if (c == 1 && ufsecp_seckey_negate(cpu, k) != UFSECP_OK) return false;
        uint8_t s[32];
        std::memcpy(s, ex, 32);
        if (ufsecp_seckey_tweak_add(cpu, s, k) != UFSECP_OK) return false;
        std::memcpy(cand[c], R33 + 1, 32);
        std::memcpy(cand[c] + 32, s, 32);
    }
    const bool ok0 = ufsecp_bch_schnorr_verify(cpu, msg, cand[0], pub33) == UFSECP_OK;
    const bool ok1 = ufsecp_bch_schnorr_verify(cpu, msg, cand[1], pub33) == UFSECP_OK;
    if (ok0 == ok1) return false;
    std::memcpy(sig_out, ok0 ? cand[0] : cand[1], 64);
    std::memcpy(neg_sig_out, ok0 ? cand[1] : cand[0], 64);
    return true;
}

/* Smallest x in [1, 255] (as 32 BE bytes) that is / is not the x of a curve
 * point, probed through the CPU key parser (BAD_PUBKEY <=> off-curve). */
static bool bch_small_x(ufsecp_ctx* cpu, bool on_curve, uint8_t x_out[32]) {
    uint8_t msg[32] = {};
    uint8_t sig[64] = {};
    sig[63] = 1;  /* r = 0 < p, s = 1: well-formed, so only the key decides */
    for (unsigned v = 1; v < 256; ++v) {
        uint8_t pub[33] = {};
        pub[0] = 0x02;
        pub[32] = static_cast<uint8_t>(v);
        const bool on = ufsecp_bch_schnorr_verify(cpu, msg, sig, pub) != UFSECP_ERR_BAD_PUBKEY;
        if (on == on_curve) { std::memcpy(x_out, pub + 1, 32); return true; }
    }
    return false;
}

struct BchEquivItem {
    uint8_t msg[32];
    uint8_t pub[33];
    uint8_t sig[64];
    uint8_t neg_sig[64];  /* same key/msg, R negated (non-residue y) */
    uint8_t expect;
    const char* what;
};

static void test_bch_schnorr_verify_equiv(ufsecp_gpu_ctx* ctx) {
    std::printf("[gpu_equiv] bch_schnorr_verify_batch\n");

    ufsecp_ctx* cpu = nullptr;
    CHECK_OK(ufsecp_ctx_create(&cpu), "CPU ctx_create succeeds");
    if (!cpu) return;

    /* Random valid signatures (deterministic seeds). */
    constexpr size_t kValid = 16;
    std::vector<BchEquivItem> valid;
    for (unsigned i = 0; i < kValid + 8 && valid.size() < kValid; ++i) {
        BchEquivItem it{};
        uint8_t priv[32], nonce[32];
        fill_deterministic(priv, 32, static_cast<uint8_t>(0x10 + i));
        fill_deterministic(nonce, 32, static_cast<uint8_t>(0x90 + i));
        fill_deterministic(it.msg, 32, static_cast<uint8_t>(0x50 + i));
        priv[0] &= 0x7F;  priv[31] |= 1;
        nonce[0] &= 0x7F; nonce[31] |= 1;
        if (ufsecp_pubkey_create(cpu, priv, it.pub) != UFSECP_OK) continue;
        if (!bch_test_sign(cpu, priv, it.msg, nonce, it.pub, it.sig, it.neg_sig)) continue;
        it.expect = 1;
        it.what = "valid";
        valid.push_back(it);
    }
    CHECK(valid.size() == kValid, "bch: every valid test signature was built");
    if (valid.empty()) { ufsecp_ctx_destroy(cpu); return; }

    uint8_t x_off[32], x_on[32];
    const bool have_off = bch_small_x(cpu, false, x_off);
    const bool have_on  = bch_small_x(cpu, true, x_on);
    CHECK(have_off && have_on, "bch: found small on-curve and off-curve x values");

    /* Invalid items, each derived from a valid one. */
    std::vector<BchEquivItem> invalid;
    size_t base = 0;
    auto add = [&](const char* what, auto mutate) {
        BchEquivItem it = valid[base++ % valid.size()];
        mutate(it);
        it.expect = 0;
        it.what = what;
        invalid.push_back(it);
    };
    add("msg bit flipped",             [](BchEquivItem& it) { it.msg[0] ^= 0x01; });
    add("r bit flipped",               [](BchEquivItem& it) { it.sig[5] ^= 0x10; });
    add("s bit flipped",               [](BchEquivItem& it) { it.sig[40] ^= 0x01; });
    add("pubkey x bit flipped",        [](BchEquivItem& it) { it.pub[10] ^= 0x04; });
    add("pubkey other parity (-P)",    [](BchEquivItem& it) { it.pub[0] ^= 0x01; });
    add("r = p",                       [](BchEquivItem& it) { std::memcpy(it.sig, kBchFieldP, 32); });
    add("r = 2^256 - 1",               [](BchEquivItem& it) { std::memset(it.sig, 0xFF, 32); });
    if (have_off)
        add("r = off-curve x",         [&](BchEquivItem& it) { std::memcpy(it.sig, x_off, 32); });
    add("R negated (non-residue y)",   [](BchEquivItem& it) { std::memcpy(it.sig, it.neg_sig, 64); });
    add("s = n",                       [](BchEquivItem& it) { std::memcpy(it.sig + 32, kBchOrderN, 32); });
    add("s = 2^256 - 1",               [](BchEquivItem& it) { std::memset(it.sig + 32, 0xFF, 32); });
    add("s = 0",                       [](BchEquivItem& it) { std::memset(it.sig + 32, 0x00, 32); });
    add("pubkey prefix 0x04",          [](BchEquivItem& it) { it.pub[0] = 0x04; });
    add("pubkey prefix 0x00",          [](BchEquivItem& it) { it.pub[0] = 0x00; });
    if (have_off)
        add("pubkey off-curve x",      [&](BchEquivItem& it) { std::memcpy(it.pub + 1, x_off, 32); });
    add("pubkey x = p (non-canonical)", [](BchEquivItem& it) { std::memcpy(it.pub + 1, kBchFieldP, 32); });
    if (have_on) {
        /* x + p aliases a curve point mod p: non-canonical, must be rejected. */
        add("pubkey x = x0 + p (non-canonical alias)", [&](BchEquivItem& it) {
            unsigned carry = 0;
            for (int b = 31; b >= 0; --b) {
                const unsigned v = unsigned(kBchFieldP[b]) + unsigned(x_on[b]) + carry;
                it.pub[1 + b] = static_cast<uint8_t>(v);
                carry = v >> 8;
            }
        });
    }

    /* Mixed batch: valid and invalid interleaved, then the remaining valid. */
    std::vector<BchEquivItem> batch;
    size_t vi = 0;
    for (const auto& bad : invalid) {
        if (vi < valid.size()) batch.push_back(valid[vi++]);
        batch.push_back(bad);
    }
    while (vi < valid.size()) batch.push_back(valid[vi++]);

    const size_t n = batch.size();
    std::vector<uint8_t> msgs(n * 32), pubs(n * 33), sigs(n * 64), res(n, 0xEE);
    for (size_t i = 0; i < n; ++i) {
        std::memcpy(msgs.data() + i * 32, batch[i].msg, 32);
        std::memcpy(pubs.data() + i * 33, batch[i].pub, 33);
        std::memcpy(sigs.data() + i * 64, batch[i].sig, 64);
    }

    const auto err = ufsecp_gpu_bch_schnorr_verify_batch(
        ctx, msgs.data(), pubs.data(), sigs.data(), n, res.data());
    if (err == UFSECP_ERR_GPU_UNSUPPORTED) {
        SKIP("bch_schnorr_verify_batch unsupported");
        ufsecp_ctx_destroy(cpu);
        return;
    }
    CHECK(err == UFSECP_OK, "GPU bch_schnorr_verify_batch succeeds");
    if (err != UFSECP_OK) { ufsecp_ctx_destroy(cpu); return; }

    for (size_t i = 0; i < n; ++i) {
        const uint8_t cpu_ok = ufsecp_bch_schnorr_verify(
            cpu, batch[i].msg, batch[i].sig, batch[i].pub) == UFSECP_OK ? 1 : 0;
        char msg[192];
        std::snprintf(msg, sizeof(msg), "bch_schnorr_verify[%zu] (%s) GPU(%d) == CPU(%d)",
                      i, batch[i].what, res[i], cpu_ok);
        CHECK(res[i] == cpu_ok, msg);
        std::snprintf(msg, sizeof(msg), "bch_schnorr_verify[%zu] (%s) verdict == expected %d",
                      i, batch[i].what, batch[i].expect);
        CHECK(res[i] == batch[i].expect, msg);
    }

    /* Single-item batch (tail of a partially filled block). */
    uint8_t one = 0xEE;
    CHECK(ufsecp_gpu_bch_schnorr_verify_batch(ctx, valid[0].msg, valid[0].pub, valid[0].sig,
                                              1, &one) == UFSECP_OK && one == 1,
          "bch_schnorr_verify single valid item accepted");

    ufsecp_ctx_destroy(cpu);
}

/* ============================================================================
 * 4. ecdh_batch equivalence
 * ============================================================================ */
static void test_ecdh_equiv(ufsecp_gpu_ctx* ctx) {
    std::printf("[gpu_equiv] ecdh_batch\n");

    constexpr size_t N = 4;
    ufsecp_ctx* cpu_ctx = nullptr;
    CHECK_OK(ufsecp_ctx_create(&cpu_ctx), "CPU ctx_create succeeds");

    uint8_t alice_keys[N * 32];
    uint8_t bob_keys[N * 32];
    uint8_t bob_pubs[N * 33];

    fill_deterministic(alice_keys, sizeof(alice_keys), 0xA1);
    fill_deterministic(bob_keys, sizeof(bob_keys), 0xB2);

    for (size_t i = 0; i < N; ++i) {
        alice_keys[i * 32] &= 0x7F;
        bob_keys[i * 32] &= 0x7F;
        CHECK_OK(ufsecp_pubkey_create(cpu_ctx, bob_keys + i * 32, bob_pubs + i * 33),
                 "CPU pubkey_create succeeds");
    }

    /* GPU ECDH */
    uint8_t gpu_secrets[N * 32];
    auto err = ufsecp_gpu_ecdh_batch(ctx, alice_keys, bob_pubs, N, gpu_secrets);
    if (err == UFSECP_ERR_GPU_UNSUPPORTED) { SKIP("ecdh_batch unsupported"); goto cleanup; }
    CHECK(err == UFSECP_OK, "GPU ecdh_batch succeeds");
    if (err != UFSECP_OK) goto cleanup;

    /* CPU ECDH for comparison */
    for (size_t i = 0; i < N; ++i) {
        uint8_t cpu_secret[32];
        auto cerr = ufsecp_ecdh(cpu_ctx, alice_keys + i * 32,
                                bob_pubs + i * 33, cpu_secret);
        CHECK(cerr == UFSECP_OK, "CPU ecdh succeeds");
        if (cerr == UFSECP_OK) {
            char msg[128];
            std::snprintf(msg, sizeof(msg), "ecdh[%zu] GPU == CPU", i);
            CHECK(std::memcmp(gpu_secrets + i * 32, cpu_secret, 32) == 0, msg);
        }
    }

cleanup:
    ufsecp_ctx_destroy(cpu_ctx);
}

/* ============================================================================
 * 5. hash160_pubkey_batch equivalence
 * ============================================================================ */
static void test_hash160_equiv(ufsecp_gpu_ctx* ctx) {
    std::printf("[gpu_equiv] hash160_pubkey_batch\n");

    constexpr size_t N = 8;
    ufsecp_ctx* cpu_ctx = nullptr;
    CHECK_OK(ufsecp_ctx_create(&cpu_ctx), "CPU ctx_create succeeds");

    uint8_t privkeys[N * 32];
    uint8_t pubkeys[N * 33];

    fill_deterministic(privkeys, sizeof(privkeys), 0xE1);
    for (size_t i = 0; i < N; ++i) {
        privkeys[i * 32] &= 0x7F;
        CHECK_OK(ufsecp_pubkey_create(cpu_ctx, privkeys + i * 32, pubkeys + i * 33),
                 "CPU pubkey_create succeeds");
    }

    /* GPU hash160 */
    uint8_t gpu_hashes[N * 20];
    auto err = ufsecp_gpu_hash160_pubkey_batch(ctx, pubkeys, N, gpu_hashes);
    if (err == UFSECP_ERR_GPU_UNSUPPORTED) { SKIP("hash160_pubkey_batch unsupported"); goto cleanup; }
    CHECK(err == UFSECP_OK, "GPU hash160_pubkey_batch succeeds");
    if (err != UFSECP_OK) goto cleanup;

    /* CPU hash160 for comparison */
    for (size_t i = 0; i < N; ++i) {
        uint8_t cpu_hash[20];
        auto cerr = ufsecp_hash160(pubkeys + i * 33, 33, cpu_hash);
        CHECK(cerr == UFSECP_OK, "CPU hash160 succeeds");
        if (cerr == UFSECP_OK) {
            char msg[128];
            std::snprintf(msg, sizeof(msg), "hash160[%zu] GPU == CPU", i);
            CHECK(std::memcmp(gpu_hashes + i * 20, cpu_hash, 20) == 0, msg);
        }
    }

cleanup:
    ufsecp_ctx_destroy(cpu_ctx);
}

/* ============================================================================
 * 6. ecrecover_batch equivalence
 * ============================================================================ */
static void test_ecrecover_equiv(ufsecp_gpu_ctx* ctx) {
    std::printf("[gpu_equiv] ecrecover_batch\n");

    constexpr size_t N = 4;
    ufsecp_ctx* cpu_ctx = nullptr;
    CHECK_OK(ufsecp_ctx_create(&cpu_ctx), "CPU ctx_create succeeds");

    uint8_t msg_hashes[N * 32];
    uint8_t privkeys[N * 32];
    uint8_t sigs[N * 64];
    int recids[N] = {};
    uint8_t expected_pubs[N * 33];
    ufsecp_error_t err = UFSECP_OK;

    fill_deterministic(msg_hashes, sizeof(msg_hashes), 0x91);
    fill_deterministic(privkeys, sizeof(privkeys), 0xA7);

    for (size_t i = 0; i < N; ++i) {
        privkeys[i * 32] &= 0x7F;
        auto perr = ufsecp_pubkey_create(cpu_ctx, privkeys + i * 32, expected_pubs + i * 33);
        CHECK(perr == UFSECP_OK, "CPU pubkey_create succeeds");
        if (perr != UFSECP_OK) goto cleanup;

        auto serr = ufsecp_ecdsa_sign_recoverable(cpu_ctx,
                                                  msg_hashes + i * 32,
                                                  privkeys + i * 32,
                                                  sigs + i * 64,
                                                  &recids[i]);
        CHECK(serr == UFSECP_OK, "CPU ecdsa_sign_recoverable succeeds");
        if (serr != UFSECP_OK) goto cleanup;
    }

    uint8_t gpu_pubs[N * 33];
    uint8_t gpu_valid[N];
    err = ufsecp_gpu_ecrecover_batch(ctx, msg_hashes, sigs, recids, N, gpu_pubs, gpu_valid);
    if (err == UFSECP_ERR_GPU_UNSUPPORTED) { SKIP("ecrecover_batch unsupported"); goto cleanup; }
    CHECK(err == UFSECP_OK, "GPU ecrecover_batch succeeds");
    if (err != UFSECP_OK) goto cleanup;

    for (size_t i = 0; i < N; ++i) {
        uint8_t cpu_pub[33];
        auto rerr = ufsecp_ecdsa_recover(cpu_ctx,
                                         msg_hashes + i * 32,
                                         sigs + i * 64,
                                         recids[i],
                                         cpu_pub);
        CHECK(rerr == UFSECP_OK, "CPU ecdsa_recover succeeds");
        if (rerr != UFSECP_OK) continue;

        char msg_valid[128];
        std::snprintf(msg_valid, sizeof(msg_valid), "ecrecover[%zu] valid flag set", i);
        CHECK(gpu_valid[i] == 1, msg_valid);

        char msg_gpu_cpu[128];
        std::snprintf(msg_gpu_cpu, sizeof(msg_gpu_cpu), "ecrecover[%zu] GPU == CPU recover", i);
        CHECK(std::memcmp(gpu_pubs + i * 33, cpu_pub, 33) == 0, msg_gpu_cpu);

        char msg_expected[128];
        std::snprintf(msg_expected, sizeof(msg_expected), "ecrecover[%zu] recovered pubkey matches signer", i);
        CHECK(std::memcmp(gpu_pubs + i * 33, expected_pubs + i * 33, 33) == 0, msg_expected);
    }

cleanup:
    ufsecp_ctx_destroy(cpu_ctx);
}

/* ============================================================================
 * 7. msm equivalence
 * ============================================================================ */
static void test_msm_equiv(ufsecp_gpu_ctx* ctx) {
    std::printf("[gpu_equiv] msm\n");

    constexpr size_t N = 4;
    ufsecp_ctx* cpu_ctx = nullptr;
    CHECK_OK(ufsecp_ctx_create(&cpu_ctx), "CPU ctx_create succeeds");

    /* Use known points: k[i]*G for small k values */
    uint8_t base_scalars[N * 32] = {};
    uint8_t points[N * 33];
    uint8_t msm_scalars[N * 32];

    /* base_scalars = {1, 2, 3, 4} → points = {G, 2G, 3G, 4G} */
    for (size_t i = 0; i < N; ++i) {
        base_scalars[i * 32 + 31] = static_cast<uint8_t>(i + 1);
        CHECK_OK(ufsecp_pubkey_create(cpu_ctx, base_scalars + i * 32, points + i * 33),
                 "CPU pubkey_create succeeds");
    }

    /* msm_scalars = {5, 6, 7, 8} */
    for (size_t i = 0; i < N; ++i) {
        std::memset(msm_scalars + i * 32, 0, 32);
        msm_scalars[i * 32 + 31] = static_cast<uint8_t>(i + 5);
    }

    /* GPU MSM: sum = 5*G + 6*2G + 7*3G + 8*4G = (5+12+21+32)*G = 70*G */
    uint8_t gpu_result[33];
    auto err = ufsecp_gpu_msm(ctx, msm_scalars, points, N, gpu_result);
    if (err == UFSECP_ERR_GPU_UNSUPPORTED) { SKIP("msm unsupported"); goto cleanup; }
    CHECK(err == UFSECP_OK, "GPU msm succeeds");
    if (err != UFSECP_OK) goto cleanup;

    /* CPU reference: 70*G */
    {
        uint8_t scalar_70[32] = {};
        scalar_70[31] = 70;
        uint8_t cpu_pub[33];
        CHECK_OK(ufsecp_pubkey_create(cpu_ctx, scalar_70, cpu_pub),
                 "CPU pubkey_create for MSM reference succeeds");
        CHECK(std::memcmp(gpu_result, cpu_pub, 33) == 0, "msm result == 70*G (CPU reference)");
    }

cleanup:
    ufsecp_ctx_destroy(cpu_ctx);
}

/* ============================================================================
 * 8. msm equivalence at a batch large enough to reach the multi-level tail
 * ============================================================================ */
static void test_msm_large_n_equiv(ufsecp_gpu_ctx* ctx) {
    std::printf("[gpu_equiv] msm (large N, multi-level tail reduce)\n");

    /* The CUDA backend block-reduces the scatter output and then finishes on a
     * single thread, so every point left to that finisher costs a serial point
     * addition. msm() therefore keeps block-reducing while the tail is longer
     * than 32, and N here is picked to make that loop actually run: 70000 points
     * give 274 blocks, one pass takes 274 down to 2, and the finisher sees 2 --
     * covering the loop body, the ping-pong buffer swap, and a finisher count
     * that is neither the old full block count nor 1.
     *
     * A SECOND loop pass would need N > 2M, far too slow for a CPU-reference MSM
     * in an audit test, and it re-runs the same code the first pass does.
     * test_msm_equiv (N=4) covers the other side: a tail short enough that the
     * loop must not run at all. */
    constexpr size_t N = 70000;
    constexpr size_t kDistinctPoints = 256;

    ufsecp_ctx* cpu_ctx = nullptr;
    CHECK_OK(ufsecp_ctx_create(&cpu_ctx), "CPU ctx_create succeeds");
    if (!cpu_ctx) return;

    /* Points repeat on a 256-entry cycle: the reduction tree is indifferent to
     * which point sits where, and generating 70000 distinct pubkeys on the CPU
     * would dominate the test's runtime for no added coverage. */
    std::vector<uint8_t> base(kDistinctPoints * 33);
    for (size_t i = 0; i < kDistinctPoints; ++i) {
        uint8_t sk[32] = {};
        fill_deterministic(sk, 32, static_cast<uint8_t>(i + 1));
        sk[0] = 0;                        /* stay well below the group order */
        sk[31] = static_cast<uint8_t>(sk[31] | 1u);   /* and non-zero */
        CHECK_OK(ufsecp_pubkey_create(cpu_ctx, sk, base.data() + i * 33),
                 "CPU pubkey_create succeeds");
    }

    std::vector<uint8_t> points(N * 33);
    std::vector<uint8_t> scalars(N * 32);
    for (size_t i = 0; i < N; ++i) {
        std::memcpy(points.data() + i * 33,
                    base.data() + (i % kDistinctPoints) * 33, 33);
        uint8_t* s = scalars.data() + i * 32;
        fill_deterministic(s, 32, static_cast<uint8_t>(i & 0xFF));
        s[0] = 0;                                   /* < group order */
        s[1] = static_cast<uint8_t>(i >> 8);        /* all N scalars distinct */
        s[31] = static_cast<uint8_t>(s[31] | 1u);   /* non-zero */
    }

    uint8_t gpu_result[33] = {};
    auto err = ufsecp_gpu_msm(ctx, scalars.data(), points.data(), N, gpu_result);
    if (err == UFSECP_ERR_GPU_UNSUPPORTED) {
        SKIP("msm unsupported");
        ufsecp_ctx_destroy(cpu_ctx);
        return;
    }
    CHECK(err == UFSECP_OK, "GPU msm (N=70000) succeeds");

    uint8_t cpu_result[33] = {};
    auto cpu_err = ufsecp_multi_scalar_mul(cpu_ctx, scalars.data(), points.data(),
                                           N, cpu_result);
    CHECK(cpu_err == UFSECP_OK, "CPU multi_scalar_mul (N=70000) succeeds");

    if (err == UFSECP_OK && cpu_err == UFSECP_OK) {
        CHECK(std::memcmp(gpu_result, cpu_result, 33) == 0,
              "GPU msm == CPU multi_scalar_mul at N=70000");
    }

    ufsecp_ctx_destroy(cpu_ctx);
}

/* ============================================================================
 * 9. Grow / shrink reuse of the backends' device scratch
 * ============================================================================
 * The CUDA and OpenCL backends keep grow-only device scratch buffers and reuse
 * them from one call to the next. Every batch op is called three times on the
 * same context: a small batch, a larger one (the scratch grows) and a smaller
 * one again (the larger buffers are reused). Each call uses rows the previous
 * call did not, and each verify call carries one tampered row, so a result
 * that leaked from a previous call's data cannot match the CPU reference. */
static constexpr size_t kGsSizes[3] = {5, 64, 3};
static constexpr size_t kGsPool = 5 + 64 + 3;

struct GrowShrinkPool {
    std::vector<uint8_t> msgs, sks, pubs, xonly, ecdsa_sigs, schnorr_sigs;
    std::vector<uint8_t> rec_sigs, peer_pubs;
    std::vector<int>     recids;
};

static bool gs_build_pool(ufsecp_ctx* cpu, GrowShrinkPool& p) {
    p.msgs.resize(kGsPool * 32);
    p.sks.resize(kGsPool * 32);
    p.pubs.resize(kGsPool * 33);
    p.xonly.resize(kGsPool * 32);
    p.ecdsa_sigs.resize(kGsPool * 64);
    p.schnorr_sigs.resize(kGsPool * 64);
    p.rec_sigs.resize(kGsPool * 64);
    p.peer_pubs.resize(kGsPool * 33);
    p.recids.resize(kGsPool);
    for (size_t i = 0; i < kGsPool; ++i) {
        uint8_t* sk  = p.sks.data() + i * 32;
        uint8_t* msg = p.msgs.data() + i * 32;
        fill_deterministic(sk, 32, static_cast<uint8_t>(3 * i + 1));
        sk[0] &= 0x7F;                               /* < group order */
        sk[31] = static_cast<uint8_t>(sk[31] | 1u);  /* non-zero */
        fill_deterministic(msg, 32, static_cast<uint8_t>(3 * i + 2));
        uint8_t peer_sk[32];
        fill_deterministic(peer_sk, 32, static_cast<uint8_t>(3 * i + 3));
        peer_sk[0] &= 0x7F;
        peer_sk[31] = static_cast<uint8_t>(peer_sk[31] | 1u);
        uint8_t aux[32] = {};
        if (ufsecp_pubkey_create(cpu, sk, p.pubs.data() + i * 33) != UFSECP_OK) return false;
        if (ufsecp_pubkey_xonly(cpu, sk, p.xonly.data() + i * 32) != UFSECP_OK) return false;
        if (ufsecp_pubkey_create(cpu, peer_sk, p.peer_pubs.data() + i * 33) != UFSECP_OK) return false;
        if (ufsecp_ecdsa_sign(cpu, msg, sk, p.ecdsa_sigs.data() + i * 64) != UFSECP_OK) return false;
        if (ufsecp_schnorr_sign(cpu, msg, sk, aux, p.schnorr_sigs.data() + i * 64) != UFSECP_OK) return false;
        if (ufsecp_ecdsa_sign_recoverable(cpu, msg, sk, p.rec_sigs.data() + i * 64,
                                          &p.recids[i]) != UFSECP_OK) return false;
    }
    return true;
}

static void test_batch_grow_shrink(ufsecp_gpu_ctx* ctx, const char* backend) {
    std::printf("[gpu_equiv] grow/shrink scratch reuse (%s)\n", backend);

    ufsecp_ctx* cpu = nullptr;
    CHECK_OK(ufsecp_ctx_create(&cpu), "CPU ctx_create succeeds");
    if (!cpu) return;
    GrowShrinkPool p;
    const bool built = gs_build_pool(cpu, p);
    CHECK(built, "grow/shrink: CPU fixtures built");
    if (!built) { ufsecp_ctx_destroy(cpu); return; }

    enum { kGen, kEcdsa, kSchnorr, kEcdh, kHash160, kEcrecover, kOps };
    static const char* const kOpName[kOps] = {
        "generator_mul_batch", "ecdsa_verify_batch", "schnorr_verify_batch",
        "ecdh_batch", "hash160_pubkey_batch", "ecrecover_batch"};
    bool unsupported[kOps] = {};
    char msg[192];

    /* Returns true when the op ran; records a SKIP once per op otherwise. */
    auto ran = [&](int op, ufsecp_error_t e, size_t n) {
        if (e == UFSECP_ERR_GPU_UNSUPPORTED) {
            if (!unsupported[op]) {
                std::snprintf(msg, sizeof(msg), "%s grow/shrink: %s unsupported",
                              backend, kOpName[op]);
                SKIP(msg);
            }
            unsupported[op] = true;
            return false;
        }
        std::snprintf(msg, sizeof(msg), "%s %s n=%zu succeeds", backend, kOpName[op], n);
        CHECK(e == UFSECP_OK, msg);
        return e == UFSECP_OK;
    };

    size_t off = 0;
    for (size_t round = 0; round < 3; ++round) {
        const size_t n   = kGsSizes[round];
        const size_t bad = (round + 1) % n;   /* tampered verify row */

        /* generator_mul_batch: k*G must equal the CPU pubkeys of THIS call. */
        if (!unsupported[kGen]) {
            std::vector<uint8_t> out(n * 33, 0xEE);
            auto e = ufsecp_gpu_generator_mul_batch(ctx, p.sks.data() + off * 32, n, out.data());
            if (ran(kGen, e, n)) {
                std::snprintf(msg, sizeof(msg), "%s generator_mul n=%zu GPU == CPU", backend, n);
                CHECK(std::memcmp(out.data(), p.pubs.data() + off * 33, n * 33) == 0, msg);
            }
        }

        /* ecdsa_verify_batch with one tampered message. */
        if (!unsupported[kEcdsa]) {
            std::vector<uint8_t> m(p.msgs.begin() + off * 32, p.msgs.begin() + (off + n) * 32);
            m[bad * 32] ^= 0x01;
            std::vector<uint8_t> res(n, 0xEE);
            auto e = ufsecp_gpu_ecdsa_verify_batch(ctx, m.data(), p.pubs.data() + off * 33,
                                                   p.ecdsa_sigs.data() + off * 64, n, res.data());
            if (ran(kEcdsa, e, n)) {
                size_t mismatch = 0;
                for (size_t i = 0; i < n; ++i) {
                    const uint8_t cpu_ok = ufsecp_ecdsa_verify(cpu, m.data() + i * 32,
                        p.ecdsa_sigs.data() + (off + i) * 64,
                        p.pubs.data() + (off + i) * 33) == UFSECP_OK ? 1 : 0;
                    if (res[i] != cpu_ok) ++mismatch;
                }
                std::snprintf(msg, sizeof(msg), "%s ecdsa_verify n=%zu GPU == CPU per row", backend, n);
                CHECK(mismatch == 0, msg);
                std::snprintf(msg, sizeof(msg), "%s ecdsa_verify n=%zu tampered row rejected", backend, n);
                CHECK(res[bad] == 0, msg);
            }
        }

        /* schnorr_verify_batch with one tampered message. */
        if (!unsupported[kSchnorr]) {
            std::vector<uint8_t> m(p.msgs.begin() + off * 32, p.msgs.begin() + (off + n) * 32);
            m[bad * 32] ^= 0x01;
            std::vector<uint8_t> res(n, 0xEE);
            auto e = ufsecp_gpu_schnorr_verify_batch(ctx, m.data(), p.xonly.data() + off * 32,
                                                     p.schnorr_sigs.data() + off * 64, n, res.data());
            if (ran(kSchnorr, e, n)) {
                size_t mismatch = 0;
                for (size_t i = 0; i < n; ++i) {
                    const uint8_t cpu_ok = ufsecp_schnorr_verify(cpu, m.data() + i * 32,
                        p.schnorr_sigs.data() + (off + i) * 64,
                        p.xonly.data() + (off + i) * 32) == UFSECP_OK ? 1 : 0;
                    if (res[i] != cpu_ok) ++mismatch;
                }
                std::snprintf(msg, sizeof(msg), "%s schnorr_verify n=%zu GPU == CPU per row", backend, n);
                CHECK(mismatch == 0, msg);
                std::snprintf(msg, sizeof(msg), "%s schnorr_verify n=%zu tampered row rejected", backend, n);
                CHECK(res[bad] == 0, msg);
            }
        }

        /* ecdh_batch: secret slots are zeroed after every call; the next call
         * must still compute its own secrets exactly. */
        if (!unsupported[kEcdh]) {
            std::vector<uint8_t> out(n * 32, 0xEE);
            auto e = ufsecp_gpu_ecdh_batch(ctx, p.sks.data() + off * 32,
                                           p.peer_pubs.data() + off * 33, n, out.data());
            if (ran(kEcdh, e, n)) {
                size_t mismatch = 0;
                for (size_t i = 0; i < n; ++i) {
                    uint8_t ref[32];
                    if (ufsecp_ecdh(cpu, p.sks.data() + (off + i) * 32,
                                    p.peer_pubs.data() + (off + i) * 33, ref) != UFSECP_OK ||
                        std::memcmp(out.data() + i * 32, ref, 32) != 0)
                        ++mismatch;
                }
                std::snprintf(msg, sizeof(msg), "%s ecdh n=%zu GPU == CPU", backend, n);
                CHECK(mismatch == 0, msg);
            }
        }

        /* hash160_pubkey_batch */
        if (!unsupported[kHash160]) {
            std::vector<uint8_t> out(n * 20, 0xEE);
            auto e = ufsecp_gpu_hash160_pubkey_batch(ctx, p.pubs.data() + off * 33, n, out.data());
            if (ran(kHash160, e, n)) {
                size_t mismatch = 0;
                for (size_t i = 0; i < n; ++i) {
                    uint8_t ref[20];
                    if (ufsecp_hash160(p.pubs.data() + (off + i) * 33, 33, ref) != UFSECP_OK ||
                        std::memcmp(out.data() + i * 20, ref, 20) != 0)
                        ++mismatch;
                }
                std::snprintf(msg, sizeof(msg), "%s hash160 n=%zu GPU == CPU", backend, n);
                CHECK(mismatch == 0, msg);
            }
        }

        /* ecrecover_batch: every recovered key must be THIS call's signer. */
        if (!unsupported[kEcrecover]) {
            std::vector<uint8_t> out(n * 33, 0xEE);
            std::vector<uint8_t> valid(n, 0xEE);
            auto e = ufsecp_gpu_ecrecover_batch(ctx, p.msgs.data() + off * 32,
                                                p.rec_sigs.data() + off * 64,
                                                p.recids.data() + off, n,
                                                out.data(), valid.data());
            if (ran(kEcrecover, e, n)) {
                size_t mismatch = 0;
                for (size_t i = 0; i < n; ++i)
                    if (valid[i] != 1 ||
                        std::memcmp(out.data() + i * 33, p.pubs.data() + (off + i) * 33, 33) != 0)
                        ++mismatch;
                std::snprintf(msg, sizeof(msg), "%s ecrecover n=%zu recovers this call's signers",
                              backend, n);
                CHECK(mismatch == 0, msg);
            }
        }

        off += n;
    }

    ufsecp_ctx_destroy(cpu);
}

/* ============================================================================ */

int main() {
    std::printf("=== GPU Ops Equivalence Test ===\n\n");

    /* Find first available backend */
    uint32_t ids[4] = {};
    uint32_t n = ufsecp_gpu_backend_count(ids, 4);
    uint32_t avail_id = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (ufsecp_gpu_is_available(ids[i])) {
            avail_id = ids[i];
            break;
        }
    }

    if (avail_id == 0) {
        std::printf("  No GPU backend available -- skipping all equivalence tests.\n");
        std::printf("\n=== Results: 0 passed, 0 failed, 0 skipped (no GPU) ===\n");
        return 0;
    }

    std::printf("  Using backend: %s\n\n", ufsecp_gpu_backend_name(avail_id));

    ufsecp_gpu_ctx* ctx = nullptr;
    auto err = ufsecp_gpu_ctx_create(&ctx, avail_id, 0);
    if (err != UFSECP_OK || !ctx) {
        std::printf("  FAIL: could not create GPU context (err=%d)\n", err);
        return 1;
    }

    test_generator_mul_equiv(ctx);
    test_ecdsa_verify_equiv(ctx);
    test_schnorr_verify_equiv(ctx);
    test_bch_schnorr_verify_equiv(ctx);
    test_ecdh_equiv(ctx);
    test_hash160_equiv(ctx);
    test_ecrecover_equiv(ctx);
    test_msm_equiv(ctx);
    test_msm_large_n_equiv(ctx);

    ufsecp_gpu_ctx_destroy(ctx);

    /* Grow/shrink scratch-reuse check on EVERY available backend (not only the
     * first): CUDA and OpenCL each keep their own grow-only device scratch. */
    for (uint32_t i = 0; i < n; ++i) {
        if (!ufsecp_gpu_is_available(ids[i])) continue;
        ufsecp_gpu_ctx* gctx = nullptr;
        if (ufsecp_gpu_ctx_create(&gctx, ids[i], 0) != UFSECP_OK || !gctx) {
            std::printf("  FAIL: could not create GPU context for %s\n",
                        ufsecp_gpu_backend_name(ids[i]));
            ++g_fail;
            continue;
        }
        test_batch_grow_shrink(gctx, ufsecp_gpu_backend_name(ids[i]));
        ufsecp_gpu_ctx_destroy(gctx);
    }

    std::printf("\n=== Results: %d passed, %d failed, %d skipped ===\n",
                g_pass, g_fail, g_skip);
    return g_fail > 0 ? 1 : 0;
}
