/* ============================================================================
 * UltrafastSecp256k1 -- GPU collect-verify parity test
 * ============================================================================
 * Proves the native GPU "collect" verify entrypoints
 *   ufsecp_gpu_ecdsa_verify_collect   (digests32 | pubkeys33 | compact sig64)
 *   ufsecp_gpu_schnorr_verify_collect (digests32 | xonly32   | BIP-340 sig64)
 * produce a per-row verdict that is bit-for-bit equivalent to the corresponding
 *   ufsecp_gpu_ecdsa_verify_batch / ufsecp_gpu_schnorr_verify_batch
 * over the SAME Structure-of-Arrays corpus, once collapsed to the collect
 * output convention.
 *
 * The two encodings differ ONLY in how the verdict is written:
 *   - verify_batch  writes a WRITE-ONLY 1-byte-per-row results buffer that the
 *     ABI pre-clears to 0; 1 == valid, 0 == invalid.
 *   - verify_collect takes a 1-byte-per-row key_buffer the caller SEEDS with its
 *     own non-zero markers; the kernel writes 0 ONLY on a VALID verdict and
 *     leaves invalid rows at their seeded (non-zero) value. The ABI never
 *     pre-clears it (0 == valid would otherwise be a mass false-accept).
 * Per-row parity predicate:  (key_buffer[i] == 0) == (batch_out[i] == 1).
 *
 * PUBLIC-DATA verification, variable-time: every input (message digest, public
 * key, signature) is public on-chain data. There is NO secret path here and the
 * key_buffer carries only opaque verdict markers, not key material -- constant
 * time is neither required nor beneficial (see CT-vs-VT boundary rule: verify is
 * variable-time by design).
 *
 * No GPU required to build/run: the null-ctx contract checks always run; the
 * on-device parity checks self-skip cleanly when no GPU backend is present or a
 * backend has not (yet) implemented a native collect override (operational GPU
 * errors and "unsupported" surface as SKIP, never a false PASS).
 * ============================================================================ */

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

#include "ufsecp/ufsecp.h"
#include "ufsecp/ufsecp_gpu.h"

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg)                                      \
    do {                                                      \
        if (cond) { ++g_pass; }                               \
        else { ++g_fail; std::printf("  FAIL: %s\n", msg); }  \
    } while (0)

namespace {

// A backend runtime error (or a backend without a native collect override) means
// "collect parity is not assertable on this runner" -- treat as skip, never a
// false FAIL and never a consensus-invalid all-zero result. Mirrors the sibling
// test_gpu_lbtc_columns_diff is_skip_err() set.
bool is_skip_err(ufsecp_error_t e) {
    return e == UFSECP_ERR_GPU_LAUNCH || e == UFSECP_ERR_GPU_MEMORY ||
           e == UFSECP_ERR_GPU_BACKEND || e == UFSECP_ERR_GPU_QUEUE ||
           e == UFSECP_ERR_GPU_DEVICE || e == UFSECP_ERR_GPU_UNSUPPORTED;
}

// Build M valid ECDSA rows in column (SoA) layout for the batch/collect ABI:
// digest32 | pubkey33 | COMPACT sig64 (r||s). Returns false if the CPU signing
// infrastructure is unavailable.
bool build_ecdsa(ufsecp_ctx* sc, size_t M,
                 std::vector<uint8_t>& dig, std::vector<uint8_t>& pub,
                 std::vector<uint8_t>& sig_compact) {
    dig.assign(M * 32, 0);
    pub.assign(M * 33, 0);
    sig_compact.assign(M * 64, 0);
    for (size_t i = 0; i < M; ++i) {
        uint8_t sk[32] = {0};
        sk[31] = (uint8_t)((i % 250) + 3);
        sk[30] = (uint8_t)((i >> 8) & 0xff);
        uint8_t msg[32];
        for (int j = 0; j < 32; ++j) msg[j] = (uint8_t)(i * 13 + j * 7 + 1);
        uint8_t p33[33], s64[64];
        if (ufsecp_pubkey_create(sc, sk, p33) != UFSECP_OK) return false;
        if (ufsecp_ecdsa_sign(sc, msg, sk, s64) != UFSECP_OK) return false;
        std::memcpy(&dig[i * 32], msg, 32);
        std::memcpy(&pub[i * 33], p33, 33);
        std::memcpy(&sig_compact[i * 64], s64, 64);
    }
    return true;
}

// Build M valid Schnorr rows in column layout: digest32 | xonly32 | BIP-340 sig64.
// xonly = compressed pubkey x-coord (drop the SEC1 prefix byte).
bool build_schnorr(ufsecp_ctx* sc, size_t M,
                   std::vector<uint8_t>& dig, std::vector<uint8_t>& xonly,
                   std::vector<uint8_t>& sig) {
    dig.assign(M * 32, 0);
    xonly.assign(M * 32, 0);
    sig.assign(M * 64, 0);
    uint8_t aux[32] = {0};
    for (size_t i = 0; i < M; ++i) {
        uint8_t sk[32] = {0};
        sk[31] = (uint8_t)((i % 250) + 5);
        sk[30] = (uint8_t)((i >> 8) & 0xff);
        uint8_t msg[32];
        for (int j = 0; j < 32; ++j) msg[j] = (uint8_t)(i * 11 + j * 3 + 2);
        uint8_t p33[33], s64[64];
        if (ufsecp_pubkey_create(sc, sk, p33) != UFSECP_OK) return false;
        if (ufsecp_schnorr_sign(sc, msg, sk, aux, s64) != UFSECP_OK) return false;
        std::memcpy(&dig[i * 32], msg, 32);
        std::memcpy(&xonly[i * 32], p33 + 1, 32);  // x-only = compressed[1..33]
        std::memcpy(&sig[i * 64], s64, 64);
    }
    return true;
}

// -- null-ctx contract (no GPU required, MANDATORY on every runner) --------------
// The collect ABI rejects a NULL ctx before any other check (see
// ufsecp_gpu_{ecdsa,schnorr}_verify_collect in ufsecp_gpu_impl.cpp), so these run
// with no hardware and MUST pass everywhere including CPU-only GitHub runners.
void test_null_ctx_contract() {
    std::printf("[gpu_collect_verify_parity] null-ctx contract (no GPU needed)\n");
    uint8_t d[32] = {0}, p[33] = {0}, s[64] = {0}, kb[1] = {0xEE};
    CHECK(ufsecp_gpu_ecdsa_verify_collect(nullptr, d, p, s, 1, kb) == UFSECP_ERR_NULL_ARG,
          "ecdsa_verify_collect(NULL ctx) -> NULL_ARG");
    CHECK(ufsecp_gpu_schnorr_verify_collect(nullptr, d, p, s, 1, kb) == UFSECP_ERR_NULL_ARG,
          "schnorr_verify_collect(NULL ctx) -> NULL_ARG");
}

// -- ctx-dependent input contract (needs a real ctx; runs per available backend) -
// With a valid ctx the collect ABI checks, in order: count==0 -> OK (no-op, never
// touches key_buffer), NULL buffers with count>0 -> NULL_ARG, count>kMaxGpuBatchN
// (1<<26) -> BAD_INPUT. The oversize-count check returns BEFORE the backend is
// invoked, so tiny dummy buffers are never dereferenced.
void check_input_contract(ufsecp_gpu_ctx* ctx) {
    uint8_t d[32] = {0}, p[33] = {0}, s[64] = {0}, kb[1] = {0xEE};
    const size_t kOversize = ((size_t)1 << 26) + 1;  // kMaxGpuBatchN + 1

    CHECK(ufsecp_gpu_ecdsa_verify_collect(ctx, d, p, s, 0, kb) == UFSECP_OK,
          "ecdsa_verify_collect(count=0) -> OK no-op");
    CHECK(ufsecp_gpu_ecdsa_verify_collect(ctx, nullptr, p, s, 1, kb) == UFSECP_ERR_NULL_ARG,
          "ecdsa_verify_collect(NULL digests) -> NULL_ARG");
    CHECK(ufsecp_gpu_ecdsa_verify_collect(ctx, d, p, s, 1, nullptr) == UFSECP_ERR_NULL_ARG,
          "ecdsa_verify_collect(NULL key_buffer) -> NULL_ARG");
    CHECK(ufsecp_gpu_ecdsa_verify_collect(ctx, d, p, s, kOversize, kb) == UFSECP_ERR_BAD_INPUT,
          "ecdsa_verify_collect(count>max) -> BAD_INPUT");

    CHECK(ufsecp_gpu_schnorr_verify_collect(ctx, d, p, s, 0, kb) == UFSECP_OK,
          "schnorr_verify_collect(count=0) -> OK no-op");
    CHECK(ufsecp_gpu_schnorr_verify_collect(ctx, nullptr, p, s, 1, kb) == UFSECP_ERR_NULL_ARG,
          "schnorr_verify_collect(NULL digests) -> NULL_ARG");
    CHECK(ufsecp_gpu_schnorr_verify_collect(ctx, d, p, s, 1, nullptr) == UFSECP_ERR_NULL_ARG,
          "schnorr_verify_collect(NULL key_buffer) -> NULL_ARG");
    CHECK(ufsecp_gpu_schnorr_verify_collect(ctx, d, p, s, kOversize, kb) == UFSECP_ERR_BAD_INPUT,
          "schnorr_verify_collect(count>max) -> BAD_INPUT");
}

// Per-row parity predicate: collect writes 0 iff valid; batch writes 1 iff valid.
bool parity_row(uint8_t key_cell, uint8_t batch_cell) {
    return (key_cell == 0) == (batch_cell == 1);
}

// ECDSA on-device parity for one backend. Returns void; skips (no failures) when
// the backend declines/errors. Asserts collect==batch per row and that invalid
// rows are LEFT seeded (proves invalid -> non-zero, not merely all-zero).
void ecdsa_parity(ufsecp_gpu_ctx* ctx, ufsecp_ctx* sc, size_t M) {
    std::vector<uint8_t> dig, pub, sig;
    if (!build_ecdsa(sc, M, dig, pub, sig)) {
        std::printf("  (ecdsa signing infra unavailable -- skip ecdsa parity)\n");
        return;
    }

    const uint8_t kSeed = 0xEE;

    // ---- all-valid corpus ----
    std::vector<uint8_t> kb(M, kSeed);
    ufsecp_error_t ec = ufsecp_gpu_ecdsa_verify_collect(ctx, dig.data(), pub.data(), sig.data(), M, kb.data());
    if (is_skip_err(ec)) { std::printf("  (ecdsa collect: backend skip %d (%s))\n", ec, ufsecp_gpu_error_str(ec)); return; }
    CHECK(ec == UFSECP_OK, "ecdsa collect valid batch returns OK");
    if (ec != UFSECP_OK) return;

    std::vector<uint8_t> batch(M, 2);
    ufsecp_error_t eb = ufsecp_gpu_ecdsa_verify_batch(ctx, dig.data(), pub.data(), sig.data(), M, batch.data());
    if (is_skip_err(eb)) { std::printf("  (ecdsa batch: backend skip -- cannot cross-check)\n"); return; }
    CHECK(eb == UFSECP_OK, "ecdsa verify_batch valid corpus -> OK");

    bool all_valid_collapse = true, per_row = true;
    for (size_t i = 0; i < M; ++i) {
        if (kb[i] != 0) all_valid_collapse = false;
        if (!parity_row(kb[i], batch[i])) per_row = false;
    }
    CHECK(all_valid_collapse, "ecdsa collect: all-valid corpus collapses every key_buffer[i]==0");
    CHECK(per_row, "ecdsa collect == verify_batch verdict per-row (valid corpus)");

    // ---- tampered corpus: flip a sig byte in 3 rows -> those rows invalid ----
    std::vector<uint8_t> t = sig;
    const size_t bad[3] = {5, 137, 400};
    for (size_t k = 0; k < 3; ++k) t[bad[k] * 64 + 10] ^= 0x55;

    std::vector<uint8_t> kb2(M, kSeed);
    ec = ufsecp_gpu_ecdsa_verify_collect(ctx, dig.data(), pub.data(), t.data(), M, kb2.data());
    std::vector<uint8_t> batch2(M, 2);
    eb = ufsecp_gpu_ecdsa_verify_batch(ctx, dig.data(), pub.data(), t.data(), M, batch2.data());
    if (is_skip_err(ec) || is_skip_err(eb)) { std::printf("  (ecdsa tamper: runtime skip)\n"); return; }
    if (ec == UFSECP_OK && eb == UFSECP_OK) {
        bool tampered_seeded = true, untouched_zero = true, per_row2 = true;
        for (size_t k = 0; k < 3; ++k) {
            // invalid -> key_buffer LEFT at its non-zero seed (not zeroed)
            if (kb2[bad[k]] == 0) tampered_seeded = false;
            if (batch2[bad[k]] != 0) tampered_seeded = false;
        }
        if (kb2[0] != 0 || batch2[0] != 1) untouched_zero = false;
        for (size_t i = 0; i < M; ++i) if (!parity_row(kb2[i], batch2[i])) per_row2 = false;
        CHECK(tampered_seeded, "ecdsa collect: tampered rows stay seeded (invalid left non-zero) and verify_batch rejects them");
        CHECK(untouched_zero, "ecdsa collect: untouched row key_buffer[0]==0 & batch[0]==1");
        CHECK(per_row2, "ecdsa collect == verify_batch verdict per-row (tampered corpus)");
    } else {
        CHECK(0, "ecdsa collect/batch tamper: unexpected error");
    }
}

// Schnorr on-device parity for one backend (mirror of ecdsa_parity).
void schnorr_parity(ufsecp_gpu_ctx* ctx, ufsecp_ctx* sc, size_t M) {
    std::vector<uint8_t> dig, xonly, sig;
    if (!build_schnorr(sc, M, dig, xonly, sig)) {
        std::printf("  (schnorr signing infra unavailable -- skip schnorr parity)\n");
        return;
    }

    const uint8_t kSeed = 0xEE;

    std::vector<uint8_t> kb(M, kSeed);
    ufsecp_error_t ec = ufsecp_gpu_schnorr_verify_collect(ctx, dig.data(), xonly.data(), sig.data(), M, kb.data());
    if (is_skip_err(ec)) { std::printf("  (schnorr collect: backend skip %d (%s))\n", ec, ufsecp_gpu_error_str(ec)); return; }
    CHECK(ec == UFSECP_OK, "schnorr collect valid batch returns OK");
    if (ec != UFSECP_OK) return;

    std::vector<uint8_t> batch(M, 2);
    ufsecp_error_t eb = ufsecp_gpu_schnorr_verify_batch(ctx, dig.data(), xonly.data(), sig.data(), M, batch.data());
    if (is_skip_err(eb)) { std::printf("  (schnorr batch: backend skip -- cannot cross-check)\n"); return; }
    CHECK(eb == UFSECP_OK, "schnorr verify_batch valid corpus -> OK");

    bool all_valid_collapse = true, per_row = true;
    for (size_t i = 0; i < M; ++i) {
        if (kb[i] != 0) all_valid_collapse = false;
        if (!parity_row(kb[i], batch[i])) per_row = false;
    }
    CHECK(all_valid_collapse, "schnorr collect: all-valid corpus collapses every key_buffer[i]==0");
    CHECK(per_row, "schnorr collect == verify_batch verdict per-row (valid corpus)");

    std::vector<uint8_t> t = sig;
    const size_t bad[3] = {3, 200, 511};
    for (size_t k = 0; k < 3; ++k) t[bad[k] * 64 + 40] ^= 0x33;

    std::vector<uint8_t> kb2(M, kSeed);
    ec = ufsecp_gpu_schnorr_verify_collect(ctx, dig.data(), xonly.data(), t.data(), M, kb2.data());
    std::vector<uint8_t> batch2(M, 2);
    eb = ufsecp_gpu_schnorr_verify_batch(ctx, dig.data(), xonly.data(), t.data(), M, batch2.data());
    if (is_skip_err(ec) || is_skip_err(eb)) { std::printf("  (schnorr tamper: runtime skip)\n"); return; }
    if (ec == UFSECP_OK && eb == UFSECP_OK) {
        bool tampered_seeded = true, untouched_zero = true, per_row2 = true;
        for (size_t k = 0; k < 3; ++k) {
            if (kb2[bad[k]] == 0) tampered_seeded = false;
            if (batch2[bad[k]] != 0) tampered_seeded = false;
        }
        if (kb2[0] != 0 || batch2[0] != 1) untouched_zero = false;
        for (size_t i = 0; i < M; ++i) if (!parity_row(kb2[i], batch2[i])) per_row2 = false;
        CHECK(tampered_seeded, "schnorr collect: tampered rows stay seeded (invalid left non-zero) and verify_batch rejects them");
        CHECK(untouched_zero, "schnorr collect: untouched row key_buffer[0]==0 & batch[0]==1");
        CHECK(per_row2, "schnorr collect == verify_batch verdict per-row (tampered corpus)");
    } else {
        CHECK(0, "schnorr collect/batch tamper: unexpected error");
    }
}

// Run the ctx-input-contract + on-device parity for one backend id.
void run_backend(uint32_t bid) {
    ufsecp_gpu_ctx* ctx = nullptr;
    if (ufsecp_gpu_ctx_create(&ctx, bid, 0) != UFSECP_OK || !ctx) {
        std::printf("  (%s: ctx create failed -- skipping)\n", ufsecp_gpu_backend_name(bid));
        return;
    }
    ufsecp_ctx* sc = nullptr;
    if (ufsecp_ctx_create(&sc) != UFSECP_OK || !sc) {
        std::printf("  (cpu ctx create failed -- skipping)\n");
        ufsecp_gpu_ctx_destroy(ctx);
        return;
    }
    std::printf("  Backend: %s\n", ufsecp_gpu_backend_name(bid));

    check_input_contract(ctx);

    const size_t M = 512;
    ecdsa_parity(ctx, sc, M);
    schnorr_parity(ctx, sc, M);

    ufsecp_ctx_destroy(sc);
    ufsecp_gpu_ctx_destroy(ctx);
}

// -- on-device collect-vs-verify_batch parity (self-skipping) --------------------
void test_collect_parity() {
    std::printf("[gpu_collect_verify_parity] collect == verify_batch per-row (if GPU available)\n");

    uint32_t ids[8] = {};
    const uint32_t n = ufsecp_gpu_backend_count(ids, 8);
    bool any = false;
    // Exercise EVERY available backend (CUDA, OpenCL, Metal) so a native collect
    // override on each is held to verify_batch parity when present.
    for (uint32_t i = 0; i < n; ++i) {
        if (ufsecp_gpu_is_available(ids[i])) { any = true; run_backend(ids[i]); }
    }
    if (!any) std::printf("  (no GPU available -- skipping parity)\n");
}

}  // namespace

int test_gpu_collect_verify_parity_run() {
    g_pass = 0; g_fail = 0;
    std::printf("=== GPU collect-verify parity (native collect == verify_batch) ===\n");
    test_null_ctx_contract();
    test_collect_parity();
    std::printf("[gpu_collect_verify_parity] pass=%d fail=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() { return test_gpu_collect_verify_parity_run(); }
#endif
