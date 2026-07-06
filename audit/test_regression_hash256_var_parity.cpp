/* ============================================================================
 * UltrafastSecp256k1 -- hash256_var cross-backend byte-identical parity
 * ============================================================================
 * Proves that every currently-available GPU backend (CUDA / OpenCL / Metal)
 * produces BYTE-IDENTICAL ufsecp_gpu_hash256_var() output for the exact same
 * input corpus. This is a DIFFERENT check from the sibling
 * test_regression_hash256_var_batch.cpp, which validates each backend
 * independently against the CPU oracle secp256k1::SHA256::hash256(); this
 * file instead cross-checks backend-vs-backend agreement directly -- no CPU
 * oracle is involved in the comparison itself.
 *
 * PUBLIC-DATA operation: Bitcoin tx/merkle-tree preimages are public on-chain
 * data. No secret key, nonce, or scalar is touched -- variable-time is
 * correct and no ct::* boundary applies here (see CT-vs-VT boundary rule).
 *
 * No GPU required to build/run: the null-ctx contract always runs on any
 * runner (no GPU needed). The on-device cross-backend parity check self-skips
 * cleanly (never a false FAIL, never a false PASS) whenever fewer than two
 * GPU backends are available AND creatable on the current machine -- e.g. a
 * Linux runner with only CUDA/OpenCL compiled in has no Metal device, so this
 * check reports SKIP rather than asserting anything about the missing
 * backend, and never falls back to comparing a single backend to itself.
 * ============================================================================ */

#include <array>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <utility>
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
// in this directory (test_regression_hash256_var_batch.cpp,
// test_gpu_collect_verify_parity.cpp, test_gpu_lbtc_columns_diff.cpp).
bool is_skip_err(ufsecp_error_t e) {
    return e == UFSECP_ERR_GPU_LAUNCH || e == UFSECP_ERR_GPU_MEMORY ||
           e == UFSECP_ERR_GPU_BACKEND || e == UFSECP_ERR_GPU_QUEUE ||
           e == UFSECP_ERR_GPU_DEVICE || e == UFSECP_ERR_GPU_UNSUPPORTED;
}

// Deterministic, index-derived row content -- no hardcoded test vectors.
void fill_row(std::vector<uint8_t>& buf, size_t off, size_t len, uint8_t seed) {
    for (size_t j = 0; j < len; ++j) buf[off + j] = (uint8_t)(seed + j * 7 + 1);
}

// -- null-ctx contract (no GPU required, MANDATORY on every runner) --------
void test_null_ctx_contract() {
    std::printf("[hash256_var_parity] null-ctx contract (no GPU needed)\n");
    uint8_t in[32] = {0};
    uint32_t len32 = 32;
    uint8_t out[32] = {0};
    CHECK(ufsecp_gpu_hash256_var(nullptr, in, &len32, 32, 1, out) == UFSECP_ERR_NULL_ARG,
          "hash256_var(NULL ctx) -> NULL_ARG");
}

// Shared deterministic corpus: 4 KAT-shaped rows (1B, 32B, 64B, ~1KB) followed
// by a moderate batch of 150 rows with varying per-row lengths in [1, stride].
// stride is fixed at 1000 so the ~1KB KAT row and the moderate batch share the
// SAME flat layout, packed into a single ufsecp_gpu_hash256_var() call -- this
// is the ONE corpus reused, unmodified, across every backend under test.
struct Corpus {
    size_t stride = 0;
    size_t n = 0;
    std::vector<uint8_t> inputs;
    std::vector<uint32_t> lens;
};

Corpus build_corpus() {
    Corpus c;
    c.stride = 1000;
    const size_t kat_n = 4;
    const size_t moderate_n = 150;
    c.n = kat_n + moderate_n;
    c.inputs.assign(c.n * c.stride, 0xAA);
    c.lens.assign(c.n, 0);

    const size_t kat_lens[kat_n] = {1, 32, 64, 1000};
    for (size_t i = 0; i < kat_n; ++i) {
        c.lens[i] = (uint32_t)kat_lens[i];
        fill_row(c.inputs, i * c.stride, kat_lens[i], (uint8_t)(0x11 * (i + 1)));
    }
    for (size_t i = 0; i < moderate_n; ++i) {
        const size_t row = kat_n + i;
        const size_t len = 1 + (i * 37) % c.stride;
        c.lens[row] = (uint32_t)len;
        fill_row(c.inputs, row * c.stride, len, (uint8_t)(i * 3 + 1));
    }
    return c;
}

// Attempt to run hash256_var on one backend against the shared corpus.
// Returns true and fills out32 (n*32 bytes) on a directly-comparable success.
// Returns false (no CHECK failure recorded) on ctx-create failure or a
// skip-class operational error -- those backends are simply excluded from
// the cross-backend comparison, never asserted against.
bool run_backend_capture(uint32_t bid, const Corpus& c, std::vector<uint8_t>& out32) {
    ufsecp_gpu_ctx* ctx = nullptr;
    if (ufsecp_gpu_ctx_create(&ctx, bid, 0) != UFSECP_OK || !ctx) {
        std::printf("  (%s: ctx create failed -- excluded from parity comparison)\n",
                     ufsecp_gpu_backend_name(bid));
        return false;
    }
    out32.assign(c.n * 32, 0xEE);
    ufsecp_error_t e = ufsecp_gpu_hash256_var(ctx, c.inputs.data(), c.lens.data(), c.stride, c.n, out32.data());
    ufsecp_gpu_ctx_destroy(ctx);
    if (is_skip_err(e)) {
        std::printf("  (%s: backend skip %d (%s) -- excluded from parity comparison)\n",
                     ufsecp_gpu_backend_name(bid), e, ufsecp_gpu_error_str(e));
        return false;
    }
    CHECK(e == UFSECP_OK, "hash256_var cross-backend corpus: returns OK");
    return e == UFSECP_OK;
}

// -- cross-backend byte-identical parity (the point of this file) -----------
void test_cross_backend_parity() {
    std::printf("[hash256_var_parity] cross-backend byte-identical output (if >=2 GPU backends available)\n");
    Corpus c = build_corpus();

    uint32_t ids[8] = {};
    const uint32_t nb = ufsecp_gpu_backend_count(ids, 8);

    struct Captured {
        uint32_t bid;
        std::vector<uint8_t> out32;
    };
    std::vector<Captured> captured;

    for (uint32_t i = 0; i < nb; ++i) {
        if (!ufsecp_gpu_is_available(ids[i])) continue;
        std::vector<uint8_t> out32;
        if (run_backend_capture(ids[i], c, out32)) {
            captured.push_back(Captured{ids[i], std::move(out32)});
        }
    }

    if (captured.size() < 2) {
        std::printf("  (only %zu backend(s) available/creatable on this machine -- "
                     "skipping cross-backend parity comparison, nothing to compare)\n",
                     captured.size());
        return;
    }

    for (size_t i = 0; i < captured.size(); ++i) {
        for (size_t j = i + 1; j < captured.size(); ++j) {
            char msg[160];
            std::snprintf(msg, sizeof(msg),
                          "hash256_var(%s) == hash256_var(%s) byte-identical over full corpus",
                          ufsecp_gpu_backend_name(captured[i].bid), ufsecp_gpu_backend_name(captured[j].bid));
            CHECK(std::memcmp(captured[i].out32.data(), captured[j].out32.data(), c.n * 32) == 0, msg);
        }
    }
}

}  // namespace

int test_regression_hash256_var_parity_run() {
    g_pass = 0; g_fail = 0;
    std::printf("=== hash256_var cross-backend byte-identical parity ===\n");
    test_null_ctx_contract();
    test_cross_backend_parity();
    std::printf("[hash256_var_parity] pass=%d fail=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_hash256_var_parity_run(); }
#endif
