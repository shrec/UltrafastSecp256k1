/* ============================================================================
 * UltrafastSecp256k1 -- hash256_var batch structural/boundary KAT coverage
 * ============================================================================
 * Differentially validates ufsecp_gpu_hash256_var() (batch variable-length
 * Bitcoin HASH256 = SHA256(SHA256(row))) against the independent CPU oracle
 * secp256k1::SHA256::hash256(). No hardcoded external digests -- every
 * expected value is computed from the same row bytes via the oracle
 * (differential testing), per the documented ABI contract in
 * include/ufsecp/ufsecp_gpu.h:
 *
 *   - n==0 is a no-op (UFSECP_OK, out32 untouched)
 *   - row i occupies inputs[i*stride .. i*stride+input_lens[i]); bytes from
 *     input_lens[i] up to stride are ignored padding
 *   - stride==len (no padding) and stride>len (padding present) with the
 *     SAME row content must produce the SAME digest
 *   - n==1 and moderate-n batches with varying per-row lengths, verified
 *     row-by-row against the oracle
 *
 * PUBLIC-DATA operation: Bitcoin tx/merkle-tree preimages are public on-chain
 * data. No secret key, nonce, or scalar is touched -- variable-time is
 * correct and no ct::* boundary applies here (see CT-vs-VT boundary rule).
 *
 * No GPU required to build/run: the null-ctx contract always runs; the
 * on-device KAT/boundary checks self-skip cleanly when no GPU backend is
 * compiled in or a device is unavailable. Operational GPU errors surface as
 * SKIP, never a false PASS and never a silently-accepted wrong digest.
 * ============================================================================ */

#include <array>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

#include "ufsecp/ufsecp_gpu.h"
#include "secp256k1/sha256.hpp"

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg)                                      \
    do {                                                      \
        if (cond) { ++g_pass; }                               \
        else { ++g_fail; std::printf("  FAIL: %s\n", msg); }  \
    } while (0)

namespace {

// A backend runtime error (or a backend without hash256_var wired up yet)
// means "not assertable on this runner" -- treat as skip, never a false FAIL.
// Mirrors the is_skip_err() convention shared by every GPU differential test
// in this directory (test_gpu_collect_verify_parity.cpp, test_gpu_lbtc_columns_diff.cpp).
bool is_skip_err(ufsecp_error_t e) {
    return e == UFSECP_ERR_GPU_LAUNCH || e == UFSECP_ERR_GPU_MEMORY ||
           e == UFSECP_ERR_GPU_BACKEND || e == UFSECP_ERR_GPU_QUEUE ||
           e == UFSECP_ERR_GPU_DEVICE || e == UFSECP_ERR_GPU_UNSUPPORTED;
}

// Deterministic, index-derived row content -- no hardcoded test vectors.
void fill_row(std::vector<uint8_t>& buf, size_t off, size_t len, uint8_t seed) {
    for (size_t j = 0; j < len; ++j) buf[off + j] = (uint8_t)(seed + j * 7 + 1);
}

// CPU reference oracle: Bitcoin HASH256 = SHA256(SHA256(x)).
std::array<uint8_t, 32> oracle(const uint8_t* data, size_t len) {
    return secp256k1::SHA256::hash256(data, len);
}

// -- null-ctx contract (no GPU required, MANDATORY on every runner) --------
void test_null_ctx_contract() {
    std::printf("[hash256_var_batch] null-ctx contract (no GPU needed)\n");
    uint8_t in[32] = {0};
    uint32_t len32 = 32;
    uint8_t out[32] = {0};
    CHECK(ufsecp_gpu_hash256_var(nullptr, in, &len32, 32, 1, out) == UFSECP_ERR_NULL_ARG,
          "hash256_var(NULL ctx) -> NULL_ARG");
}

// -- KAT-style structural coverage: 1B, 32B, 64B (merkle-pair-shaped), and a
// ~1KB (tx-shaped) row, packed at a shared stride with different pad filler
// than the row content (so any pad leakage would change the digest).
void test_kat_rows(ufsecp_gpu_ctx* ctx) {
    const size_t lens[4] = {1, 32, 64, 1000};
    const size_t n = 4;
    const size_t stride = 1000;
    std::vector<uint8_t> inputs(n * stride, 0xAA);
    std::vector<uint32_t> input_lens(n);
    for (size_t i = 0; i < n; ++i) {
        input_lens[i] = (uint32_t)lens[i];
        fill_row(inputs, i * stride, lens[i], (uint8_t)(0x11 * (i + 1)));
    }
    std::vector<uint8_t> out(n * 32, 0xEE);
    ufsecp_error_t e = ufsecp_gpu_hash256_var(ctx, inputs.data(), input_lens.data(), stride, n, out.data());
    if (is_skip_err(e)) { std::printf("  (kat rows: backend skip %d (%s))\n", e, ufsecp_gpu_error_str(e)); return; }
    CHECK(e == UFSECP_OK, "hash256_var KAT rows: returns OK");
    if (e != UFSECP_OK) return;
    for (size_t i = 0; i < n; ++i) {
        auto exp = oracle(inputs.data() + i * stride, lens[i]);
        CHECK(std::memcmp(out.data() + i * 32, exp.data(), 32) == 0,
              "hash256_var KAT row matches secp256k1::SHA256::hash256 oracle");
    }
}

// -- stride==len (no padding) vs stride>len (padding present): SAME row
// content must produce the SAME digest -- padding bytes must not leak in.
void test_stride_padding(ufsecp_gpu_ctx* ctx) {
    const size_t len = 64;
    std::vector<uint8_t> row(len);
    fill_row(row, 0, len, 0x42);
    auto expected = oracle(row.data(), len);

    // stride == len (tight packing, no padding)
    {
        uint32_t l32 = (uint32_t)len;
        std::vector<uint8_t> out(32, 0xEE);
        ufsecp_error_t e = ufsecp_gpu_hash256_var(ctx, row.data(), &l32, len, 1, out.data());
        if (is_skip_err(e)) { std::printf("  (stride==len: backend skip)\n"); return; }
        CHECK(e == UFSECP_OK, "hash256_var stride==len: returns OK");
        if (e == UFSECP_OK)
            CHECK(std::memcmp(out.data(), expected.data(), 32) == 0,
                  "hash256_var stride==len matches oracle");
    }
    // stride > len, padded with a DIFFERENT filler byte than the row content.
    {
        const size_t stride = len + 37;
        std::vector<uint8_t> padded(stride, 0x5A);
        std::memcpy(padded.data(), row.data(), len);
        uint32_t l32 = (uint32_t)len;
        std::vector<uint8_t> out(32, 0xEE);
        ufsecp_error_t e = ufsecp_gpu_hash256_var(ctx, padded.data(), &l32, stride, 1, out.data());
        if (is_skip_err(e)) { std::printf("  (stride>len: backend skip)\n"); return; }
        CHECK(e == UFSECP_OK, "hash256_var stride>len: returns OK");
        if (e == UFSECP_OK)
            CHECK(std::memcmp(out.data(), expected.data(), 32) == 0,
                  "hash256_var stride>len (padded) matches oracle (padding ignored)");
    }
}

// -- n==0: no-op, OK, out32 left untouched -----------------------------------
void test_n_zero(ufsecp_gpu_ctx* ctx) {
    uint8_t sentinel[32];
    std::memset(sentinel, 0x77, sizeof(sentinel));
    uint8_t out[32];
    std::memcpy(out, sentinel, sizeof(out));
    // n==0 must be a no-op regardless of stride/pointer values (never dereferenced).
    ufsecp_error_t e = ufsecp_gpu_hash256_var(ctx, nullptr, nullptr, 0, 0, out);
    CHECK(e == UFSECP_OK, "hash256_var(n=0) -> OK no-op");
    CHECK(std::memcmp(out, sentinel, sizeof(out)) == 0,
          "hash256_var(n=0) leaves out32 untouched");
}

// -- n==1 and a moderately large n with varying per-row lengths --------------
void test_n_one_and_moderate(ufsecp_gpu_ctx* ctx) {
    // n==1
    {
        const size_t len = 55;
        std::vector<uint8_t> row(len);
        fill_row(row, 0, len, 0x09);
        uint32_t l32 = (uint32_t)len;
        std::vector<uint8_t> out(32, 0xEE);
        ufsecp_error_t e = ufsecp_gpu_hash256_var(ctx, row.data(), &l32, len, 1, out.data());
        if (is_skip_err(e)) {
            std::printf("  (n=1: backend skip)\n");
        } else {
            CHECK(e == UFSECP_OK, "hash256_var(n=1): returns OK");
            if (e == UFSECP_OK) {
                auto exp = oracle(row.data(), len);
                CHECK(std::memcmp(out.data(), exp.data(), 32) == 0, "hash256_var(n=1) matches oracle");
            }
        }
    }
    // moderate n, varying per-row lengths in [1, stride]
    {
        const size_t n = 300;
        const size_t stride = 128;
        std::vector<uint8_t> inputs(n * stride, 0xCC);
        std::vector<uint32_t> input_lens(n);
        for (size_t i = 0; i < n; ++i) {
            const size_t len = 1 + (i * 37) % stride;
            input_lens[i] = (uint32_t)len;
            fill_row(inputs, i * stride, len, (uint8_t)(i * 3 + 1));
        }
        std::vector<uint8_t> out(n * 32, 0xEE);
        ufsecp_error_t e = ufsecp_gpu_hash256_var(ctx, inputs.data(), input_lens.data(), stride, n, out.data());
        if (is_skip_err(e)) { std::printf("  (moderate n: backend skip)\n"); return; }
        CHECK(e == UFSECP_OK, "hash256_var(moderate n) returns OK");
        if (e != UFSECP_OK) return;
        bool all_match = true;
        for (size_t i = 0; i < n; ++i) {
            auto exp = oracle(inputs.data() + i * stride, input_lens[i]);
            if (std::memcmp(out.data() + i * 32, exp.data(), 32) != 0) { all_match = false; break; }
        }
        CHECK(all_match, "hash256_var(moderate n) matches oracle row-by-row");
    }
}

void run_backend(uint32_t bid) {
    ufsecp_gpu_ctx* ctx = nullptr;
    if (ufsecp_gpu_ctx_create(&ctx, bid, 0) != UFSECP_OK || !ctx) {
        std::printf("  (%s: ctx create failed -- skipping)\n", ufsecp_gpu_backend_name(bid));
        return;
    }
    std::printf("  Backend: %s\n", ufsecp_gpu_backend_name(bid));
    test_kat_rows(ctx);
    test_stride_padding(ctx);
    test_n_zero(ctx);
    test_n_one_and_moderate(ctx);
    ufsecp_gpu_ctx_destroy(ctx);
}

void test_on_device() {
    std::printf("[hash256_var_batch] KAT/boundary coverage vs CPU oracle (if GPU available)\n");
    uint32_t ids[8] = {};
    const uint32_t n = ufsecp_gpu_backend_count(ids, 8);
    bool any = false;
    for (uint32_t i = 0; i < n; ++i) {
        if (ufsecp_gpu_is_available(ids[i])) { any = true; run_backend(ids[i]); }
    }
    if (!any) std::printf("  (no GPU available -- skipping on-device coverage)\n");
}

}  // namespace

int test_regression_hash256_var_batch_run() {
    g_pass = 0; g_fail = 0;
    std::printf("=== hash256_var batch structural/boundary KAT ===\n");
    test_null_ctx_contract();
    test_on_device();
    std::printf("[hash256_var_batch] pass=%d fail=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_hash256_var_batch_run(); }
#endif
