/* ============================================================================
 * UltrafastSecp256k1 -- merkle_pair_hash structural/boundary KAT + cross-backend parity
 * ============================================================================
 * Differentially validates ufsecp_gpu_merkle_pair_hash() (batch Bitcoin
 * Merkle-pair HASH256 = SHA256(SHA256(left32 || right32)), Structure-of-Arrays
 * column layout) against the independent CPU oracle
 * secp256k1::SHA256::hash256(). No hardcoded external digests -- every
 * expected value is computed from the same left/right column bytes via the
 * oracle (differential testing), per the documented ABI contract in
 * include/ufsecp/ufsecp_gpu.h:
 *
 *   - n==0 is a no-op (UFSECP_OK, out32 untouched)
 *   - row i = HASH256(left32[i*32 .. i*32+32) || right32[i*32 .. i*32+32))
 *   - fixed 64-byte combined input per row -- no stride/input_len parameter,
 *     unlike the sibling hash256_var/tagged_hash_var ABI entrypoints
 *   - n==1 and moderate-n batches, verified row-by-row against the oracle
 *   - left32 == right32 (same buffer, Bitcoin's odd-leaf duplication case)
 *     must hash correctly with no special-casing that could accidentally
 *     skip or alias the right column
 *   - every currently-available GPU backend that overrides merkle_pair_hash
 *     natively (CUDA/OpenCL/Metal) must produce byte-identical output for the
 *     SAME input corpus (cross-backend parity, mirrors the convention in
 *     test_regression_hash256_var_parity.cpp)
 *
 * PUBLIC-DATA operation: Bitcoin Merkle-tree preimages are public on-chain
 * data. No secret key, nonce, or scalar is touched -- variable-time is
 * correct and no ct::* boundary applies here (see CT-vs-VT boundary rule).
 *
 * No GPU required to build/run: the null-ctx contract always runs on any
 * runner (no GPU needed). The on-device KAT and cross-backend parity checks
 * self-skip cleanly whenever no GPU backend overriding merkle_pair_hash is
 * compiled in / available -- the base gpu_backend.hpp virtual default
 * returns GpuError::Unsupported (-> UFSECP_ERR_GPU_UNSUPPORTED, a documented
 * skip class) -- never a false FAIL and never a silently-accepted wrong
 * digest.
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

// A backend runtime error (or a backend without merkle_pair_hash wired up
// natively -- i.e. the abstract-safe base default) means "not assertable on
// this runner" -- treat as skip, never a false FAIL. Mirrors the is_skip_err()
// convention shared by every GPU differential test in this directory
// (test_regression_hash256_var_batch.cpp, test_regression_hash256_var_parity.cpp).
bool is_skip_err(ufsecp_error_t e) {
    return e == UFSECP_ERR_GPU_LAUNCH || e == UFSECP_ERR_GPU_MEMORY ||
           e == UFSECP_ERR_GPU_BACKEND || e == UFSECP_ERR_GPU_QUEUE ||
           e == UFSECP_ERR_GPU_DEVICE || e == UFSECP_ERR_GPU_UNSUPPORTED;
}

// Deterministic, index-derived 32-byte column content -- no hardcoded vectors.
void fill_col(std::vector<uint8_t>& buf, size_t off, uint8_t seed) {
    for (size_t j = 0; j < 32; ++j) buf[off + j] = (uint8_t)(seed + j * 7 + 1);
}

// CPU reference oracle: Bitcoin Merkle-pair HASH256(left32 || right32).
std::array<uint8_t, 32> oracle(const uint8_t* left, const uint8_t* right) {
    uint8_t combined[64];
    std::memcpy(combined, left, 32);
    std::memcpy(combined + 32, right, 32);
    return secp256k1::SHA256::hash256(combined, 64);
}

// -- null-ctx contract (no GPU required, MANDATORY on every runner) --------
void test_null_ctx_contract() {
    std::printf("[merkle_pair_hash] null-ctx contract (no GPU needed)\n");
    uint8_t left[32] = {0};
    uint8_t right[32] = {0};
    uint8_t out[32] = {0};
    CHECK(ufsecp_gpu_merkle_pair_hash(nullptr, left, right, 1, out) == UFSECP_ERR_NULL_ARG,
          "merkle_pair_hash(NULL ctx) -> NULL_ARG");
}

// -- n==0: no-op, OK, out32 left byte-for-byte untouched regardless of
// null/garbage left32/right32 (neither is ever dereferenced when n==0).
void test_n_zero(ufsecp_gpu_ctx* ctx) {
    uint8_t sentinel[32];
    std::memset(sentinel, 0x77, sizeof(sentinel));
    uint8_t out[32];
    std::memcpy(out, sentinel, sizeof(out));
    ufsecp_error_t e = ufsecp_gpu_merkle_pair_hash(ctx, nullptr, nullptr, 0, out);
    CHECK(e == UFSECP_OK, "merkle_pair_hash(n=0, NULL left32/right32) -> OK no-op");
    CHECK(std::memcmp(out, sentinel, sizeof(out)) == 0,
          "merkle_pair_hash(n=0) leaves out32 byte-for-byte untouched");
}

// -- n==1 and a moderately large n, verified row-by-row against the oracle --
void test_n_one_and_moderate(ufsecp_gpu_ctx* ctx) {
    // n==1
    {
        std::vector<uint8_t> left(32), right(32);
        fill_col(left, 0, 0x09);
        fill_col(right, 0, 0x5C);
        std::vector<uint8_t> out(32, 0xEE);
        ufsecp_error_t e = ufsecp_gpu_merkle_pair_hash(ctx, left.data(), right.data(), 1, out.data());
        if (is_skip_err(e)) {
            std::printf("  (n=1: backend skip %d (%s))\n", e, ufsecp_gpu_error_str(e));
        } else {
            CHECK(e == UFSECP_OK, "merkle_pair_hash(n=1): returns OK");
            if (e == UFSECP_OK) {
                auto exp = oracle(left.data(), right.data());
                CHECK(std::memcmp(out.data(), exp.data(), 32) == 0, "merkle_pair_hash(n=1) matches oracle");
            }
        }
    }
    // moderate n, distinct per-row left/right content
    {
        const size_t n = 300;
        std::vector<uint8_t> left(n * 32), right(n * 32);
        for (size_t i = 0; i < n; ++i) {
            fill_col(left, i * 32, (uint8_t)(i * 3 + 1));
            fill_col(right, i * 32, (uint8_t)(i * 5 + 2));
        }
        std::vector<uint8_t> out(n * 32, 0xEE);
        ufsecp_error_t e = ufsecp_gpu_merkle_pair_hash(ctx, left.data(), right.data(), n, out.data());
        if (is_skip_err(e)) { std::printf("  (moderate n: backend skip %d (%s))\n", e, ufsecp_gpu_error_str(e)); return; }
        CHECK(e == UFSECP_OK, "merkle_pair_hash(moderate n) returns OK");
        if (e != UFSECP_OK) return;
        bool all_match = true;
        for (size_t i = 0; i < n; ++i) {
            auto exp = oracle(left.data() + i * 32, right.data() + i * 32);
            if (std::memcmp(out.data() + i * 32, exp.data(), 32) != 0) { all_match = false; break; }
        }
        CHECK(all_match, "merkle_pair_hash(moderate n) matches oracle row-by-row");
    }
}

// -- Bitcoin's odd-leaf-count Merkle padding duplicates the last leaf, so
// left32 == right32 (the SAME 32-byte buffer passed for both columns of a
// row) is a real, expected input shape -- must hash correctly with no
// special-casing that could accidentally skip or alias the right column.
void test_left_eq_right(ufsecp_gpu_ctx* ctx) {
    std::vector<uint8_t> left(32);
    fill_col(left, 0, 0x21);
    std::vector<uint8_t> out(32, 0xEE);
    ufsecp_error_t e = ufsecp_gpu_merkle_pair_hash(ctx, left.data(), left.data(), 1, out.data());
    if (is_skip_err(e)) { std::printf("  (left==right: backend skip %d (%s))\n", e, ufsecp_gpu_error_str(e)); return; }
    CHECK(e == UFSECP_OK, "merkle_pair_hash(left==right, same buffer) returns OK");
    if (e == UFSECP_OK) {
        auto exp = oracle(left.data(), left.data());
        CHECK(std::memcmp(out.data(), exp.data(), 32) == 0,
              "merkle_pair_hash(left==right) matches oracle (Bitcoin odd-leaf duplication case)");
    }
}

void run_backend(uint32_t bid) {
    ufsecp_gpu_ctx* ctx = nullptr;
    if (ufsecp_gpu_ctx_create(&ctx, bid, 0) != UFSECP_OK || !ctx) {
        std::printf("  (%s: ctx create failed -- skipping)\n", ufsecp_gpu_backend_name(bid));
        return;
    }
    std::printf("  Backend: %s\n", ufsecp_gpu_backend_name(bid));
    test_n_zero(ctx);
    test_n_one_and_moderate(ctx);
    test_left_eq_right(ctx);
    ufsecp_gpu_ctx_destroy(ctx);
}

void test_on_device() {
    std::printf("[merkle_pair_hash] KAT/boundary coverage vs CPU oracle (if GPU available)\n");
    uint32_t ids[8] = {};
    const uint32_t n = ufsecp_gpu_backend_count(ids, 8);
    bool any = false;
    for (uint32_t i = 0; i < n; ++i) {
        if (ufsecp_gpu_is_available(ids[i])) { any = true; run_backend(ids[i]); }
    }
    if (!any) std::printf("  (no GPU available -- skipping on-device coverage)\n");
}

// Shared deterministic corpus reused, unmodified, across every backend under
// cross-backend parity comparison -- 150 rows with distinct left/right
// column content per row.
struct Corpus {
    size_t n = 0;
    std::vector<uint8_t> left;
    std::vector<uint8_t> right;
};

Corpus build_corpus() {
    Corpus c;
    c.n = 150;
    c.left.assign(c.n * 32, 0);
    c.right.assign(c.n * 32, 0);
    for (size_t i = 0; i < c.n; ++i) {
        fill_col(c.left, i * 32, (uint8_t)(i * 3 + 1));
        fill_col(c.right, i * 32, (uint8_t)(i * 5 + 2));
    }
    return c;
}

// Attempt to run merkle_pair_hash on one backend against the shared corpus.
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
    ufsecp_error_t e = ufsecp_gpu_merkle_pair_hash(ctx, c.left.data(), c.right.data(), c.n, out32.data());
    ufsecp_gpu_ctx_destroy(ctx);
    if (is_skip_err(e)) {
        std::printf("  (%s: backend skip %d (%s) -- excluded from parity comparison)\n",
                     ufsecp_gpu_backend_name(bid), e, ufsecp_gpu_error_str(e));
        return false;
    }
    CHECK(e == UFSECP_OK, "merkle_pair_hash cross-backend corpus: returns OK");
    return e == UFSECP_OK;
}

// -- cross-backend byte-identical parity (mirrors test_regression_hash256_var_parity.cpp)
void test_cross_backend_parity() {
    std::printf("[merkle_pair_hash] cross-backend byte-identical output (if >=2 GPU backends available)\n");
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
                          "merkle_pair_hash(%s) == merkle_pair_hash(%s) byte-identical over full corpus",
                          ufsecp_gpu_backend_name(captured[i].bid), ufsecp_gpu_backend_name(captured[j].bid));
            CHECK(std::memcmp(captured[i].out32.data(), captured[j].out32.data(), c.n * 32) == 0, msg);
        }
    }
}

}  // namespace

int test_regression_merkle_pair_hash_run() {
    g_pass = 0; g_fail = 0;
    std::printf("=== merkle_pair_hash structural/boundary KAT + cross-backend parity ===\n");
    test_null_ctx_contract();
    test_on_device();
    test_cross_backend_parity();
    std::printf("[merkle_pair_hash] pass=%d fail=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_merkle_pair_hash_run(); }
#endif
