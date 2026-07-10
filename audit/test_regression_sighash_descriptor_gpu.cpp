/* ============================================================================
 * UltrafastSecp256k1 -- sighash_descriptor_hash structural/boundary KAT
 * ============================================================================
 * Differentially validates ufsecp_gpu_sighash_descriptor_hash() (descriptor-
 * shaped concatenation of per-row field columns, then HASH256) against an
 * independent CPU oracle that concatenates the SAME field bytes by direct
 * construction (not by re-parsing the descriptor) and hashes them with
 * secp256k1::SHA256::hash256(). No hardcoded external digests -- every
 * expected value is derived from the same row bytes the test itself builds.
 *
 * Contract under test (compat/libbitcoin_direct/include/ufsecp/libbitcoin.hpp,
 * ufsecp::lbtc::sighash_descriptor_hash_batch doc comment):
 *   - count==0 is a no-op (UFSECP_OK, out32 untouched)
 *   - descriptor: 2-byte-per-field little-endian refs, terminated by 0xFF
 *   - fixed-length fields use one shared column per field_id; variable-length
 *     (HAS_LENGTH) fields use a strided column + a per-row length array
 *   - nHashType (field_id 0x09) is mandatory
 *   - HASH256(preimage) = SHA256(SHA256(concatenated field bytes in
 *     descriptor order))
 *
 * PUBLIC-DATA operation: Bitcoin sighash preimages are public on-chain data.
 * No secret key, nonce, or scalar is touched -- variable-time is correct and
 * no ct::* boundary applies here (see CT-vs-VT boundary rule).
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
#include <thread>
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

// A backend runtime error (or a backend without sighash_descriptor_hash wired
// up yet) means "not assertable on this runner" -- treat as skip, never a
// false FAIL. Mirrors the is_skip_err() convention shared by every GPU
// differential test in this directory.
bool is_skip_err(ufsecp_error_t e) {
    return e == UFSECP_ERR_GPU_LAUNCH || e == UFSECP_ERR_GPU_MEMORY ||
           e == UFSECP_ERR_GPU_BACKEND || e == UFSECP_ERR_GPU_QUEUE ||
           e == UFSECP_ERR_GPU_DEVICE || e == UFSECP_ERR_GPU_UNSUPPORTED;
}

// Deterministic, index-derived content -- no hardcoded test vectors.
void fill_bytes(uint8_t* buf, size_t len, uint8_t seed) {
    for (size_t j = 0; j < len; ++j) buf[j] = (uint8_t)(seed + j * 7 + 1);
}

// CPU reference oracle: Bitcoin HASH256 = SHA256(SHA256(x)).
std::array<uint8_t, 32> oracle(const uint8_t* data, size_t len) {
    return secp256k1::SHA256::hash256(data, len);
}

// -- null-ctx contract (no GPU required, MANDATORY on every runner) --------
void test_null_ctx_contract() {
    std::printf("[sighash_descriptor_gpu] null-ctx contract (no GPU needed)\n");
    uint8_t desc[] = {0x09, 0x00, 0xFF};  // nHashType only + terminator
    const uint8_t* fdata[256] = {};
    uint32_t flens[256] = {};
    uint8_t nht[4] = {0};
    fdata[0x09] = nht;
    flens[0x09] = 4;
    uint8_t out[32] = {0};
    CHECK(ufsecp_gpu_sighash_descriptor_hash(nullptr, desc, sizeof(desc), fdata, flens,
                                              nullptr, 1, out) == UFSECP_ERR_NULL_ARG,
          "sighash_descriptor_hash(NULL ctx) -> NULL_ARG");
}

// Legacy BIP143-shaped descriptor with 9 all-fixed fields (no scriptCode),
// matching compat/libbitcoin_direct/tests/test_direct_operations.cpp's
// established legacy fixture exactly. field order: nVersion, hashPrevouts,
// hashSequence, outpoint, value, nSequence, hashOutputs, nLocktime, nHashType.
struct LegacyFixture {
    static constexpr uint8_t kFieldIds[9] = {0x00, 0x01, 0x02, 0x03, 0x05, 0x06, 0x07, 0x08, 0x09};
    static constexpr uint32_t kFieldLens[9] = {4, 32, 32, 36, 8, 4, 32, 4, 4};
    uint8_t descriptor[19] = {
        0x00, 0x00, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00, 0x05, 0x00,
        0x06, 0x00, 0x07, 0x00, 0x08, 0x00, 0x09, 0x00, 0xFF,
    };
    std::vector<std::vector<uint8_t>> columns;  // one per kFieldIds entry, n*len bytes
    const uint8_t* fdata[256] = {};
    uint32_t flens[256] = {};

    // tag: extra distinguishing offset folded into every row's content
    // (default 0 reproduces the exact bytes used by every pre-existing
    // caller of build() below). Lets callers that build several independent
    // fixtures -- e.g. one per GPU context, or one per worker thread --
    // give each one visibly distinct content, so a cross-context or
    // cross-thread buffer mix-up produces a detectably wrong digest instead
    // of an accidental match.
    void build(size_t n, uint8_t tag = 0) {
        columns.assign(9, {});
        for (size_t f = 0; f < 9; ++f) {
            columns[f].assign(n * kFieldLens[f], 0);
            for (size_t row = 0; row < n; ++row)
                fill_bytes(columns[f].data() + row * kFieldLens[f], kFieldLens[f],
                           (uint8_t)(0x11 * (f + 1) + row * 13 + tag));
            fdata[kFieldIds[f]] = columns[f].data();
            flens[kFieldIds[f]] = kFieldLens[f];
        }
    }

    // Independent oracle: concatenate this row's 9 fields in descriptor
    // order (built directly from the same per-row buffers, NOT by
    // re-parsing the descriptor) and HASH256 the result.
    std::array<uint8_t, 32> expected_row(size_t row) const {
        std::vector<uint8_t> preimage;
        for (size_t f = 0; f < 9; ++f) {
            const uint8_t* p = columns[f].data() + row * kFieldLens[f];
            preimage.insert(preimage.end(), p, p + kFieldLens[f]);
        }
        return oracle(preimage.data(), preimage.size());
    }
};
constexpr uint8_t LegacyFixture::kFieldIds[9];
constexpr uint32_t LegacyFixture::kFieldLens[9];

void test_kat_legacy_all_fixed(ufsecp_gpu_ctx* ctx) {
    const size_t n = 5;
    LegacyFixture fx;
    fx.build(n);
    std::vector<uint8_t> out(n * 32, 0xEE);
    ufsecp_error_t e = ufsecp_gpu_sighash_descriptor_hash(
        ctx, fx.descriptor, sizeof(fx.descriptor), fx.fdata, fx.flens, nullptr, n, out.data());
    if (is_skip_err(e)) { std::printf("  (legacy KAT: backend skip %d (%s))\n", e, ufsecp_gpu_error_str(e)); return; }
    CHECK(e == UFSECP_OK, "sighash_descriptor_hash legacy KAT: returns OK");
    if (e != UFSECP_OK) return;
    for (size_t row = 0; row < n; ++row) {
        auto exp = fx.expected_row(row);
        CHECK(std::memcmp(out.data() + row * 32, exp.data(), 32) == 0,
              "sighash_descriptor_hash legacy KAT row matches direct-concatenation oracle");
    }
}

// Variable-length scriptCode descriptor: nVersion, scriptCode(HAS_LENGTH), nHashType.
// Exercises the on-device unbounded-length streaming path across the exact
// SHA-256 block-boundary edges: 0 bytes (empty variable field), 1 byte
// (minimal), 63/64/65 bytes (one byte short of, exactly at, and one byte past
// a full 64-byte block -- combined with the 4-byte nVersion preamble and
// 4-byte nHashType postamble this also lands the block boundary at several
// different offsets *within* the variable field itself, not just at its
// edges), and a >2 KB row that crosses several 64-byte SHA-256 blocks.
void test_kat_variable_scriptcode(ufsecp_gpu_ctx* ctx) {
    const uint8_t desc[] = {0x00, 0x00, 0x04, 0x10, 0x09, 0x00, 0xFF};
    const size_t n = 6;
    const size_t stride = 2048;
    const size_t script_lens[n] = {0, 1, 63, 64, 65, 2001};

    std::vector<uint8_t> nver(n * 4);
    std::vector<uint8_t> script(n * stride, 0x5A);  // padding filler distinct from content
    std::vector<uint8_t> nht(n * 4);
    std::vector<uint32_t> script_var_lens(n);
    for (size_t row = 0; row < n; ++row) {
        fill_bytes(nver.data() + row * 4, 4, (uint8_t)(0x20 + row));
        fill_bytes(script.data() + row * stride, script_lens[row], (uint8_t)(0x30 + row));
        fill_bytes(nht.data() + row * 4, 4, (uint8_t)(0x40 + row));
        script_var_lens[row] = (uint32_t)script_lens[row];
    }

    const uint8_t* fdata[256] = {};
    uint32_t flens[256] = {};
    const uint32_t* fvarlens[256] = {};
    fdata[0x00] = nver.data();   flens[0x00] = 4;
    fdata[0x04] = script.data(); flens[0x04] = (uint32_t)stride;
    fdata[0x09] = nht.data();    flens[0x09] = 4;
    fvarlens[0x04] = script_var_lens.data();

    std::vector<uint8_t> out(n * 32, 0xEE);
    ufsecp_error_t e = ufsecp_gpu_sighash_descriptor_hash(
        ctx, desc, sizeof(desc), fdata, flens, fvarlens, n, out.data());
    if (is_skip_err(e)) { std::printf("  (variable scriptCode: backend skip %d (%s))\n", e, ufsecp_gpu_error_str(e)); return; }
    CHECK(e == UFSECP_OK, "sighash_descriptor_hash variable scriptCode: returns OK");
    if (e != UFSECP_OK) return;
    for (size_t row = 0; row < n; ++row) {
        std::vector<uint8_t> preimage;
        preimage.insert(preimage.end(), nver.data() + row * 4, nver.data() + row * 4 + 4);
        preimage.insert(preimage.end(), script.data() + row * stride,
                        script.data() + row * stride + script_lens[row]);
        preimage.insert(preimage.end(), nht.data() + row * 4, nht.data() + row * 4 + 4);
        auto exp = oracle(preimage.data(), preimage.size());
        CHECK(std::memcmp(out.data() + row * 32, exp.data(), 32) == 0,
              "sighash_descriptor_hash variable scriptCode row matches oracle (padding ignored)");
    }
}

// -- count==0: no-op, OK, out32 left untouched -------------------------------
void test_n_zero(ufsecp_gpu_ctx* ctx) {
    const uint8_t desc[] = {0x09, 0x00, 0xFF};
    const uint8_t* fdata[256] = {};
    uint32_t flens[256] = {};
    uint8_t nht[4] = {0};
    fdata[0x09] = nht;
    flens[0x09] = 4;
    uint8_t sentinel[32];
    std::memset(sentinel, 0x77, sizeof(sentinel));
    uint8_t out[32];
    std::memcpy(out, sentinel, sizeof(out));
    ufsecp_error_t e = ufsecp_gpu_sighash_descriptor_hash(ctx, desc, sizeof(desc), fdata, flens, nullptr, 0, out);
    CHECK(e == UFSECP_OK, "sighash_descriptor_hash(count=0) -> OK no-op");
    CHECK(std::memcmp(out, sentinel, sizeof(out)) == 0,
          "sighash_descriptor_hash(count=0) leaves out32 untouched");
}

// -- moderate count with the legacy fixture, verified row-by-row ------------
void test_moderate_count(ufsecp_gpu_ctx* ctx) {
    const size_t n = 257;  // deliberately not a power of two / warp multiple
    LegacyFixture fx;
    fx.build(n);
    std::vector<uint8_t> out(n * 32, 0xEE);
    ufsecp_error_t e = ufsecp_gpu_sighash_descriptor_hash(
        ctx, fx.descriptor, sizeof(fx.descriptor), fx.fdata, fx.flens, nullptr, n, out.data());
    if (is_skip_err(e)) { std::printf("  (moderate count: backend skip %d (%s))\n", e, ufsecp_gpu_error_str(e)); return; }
    CHECK(e == UFSECP_OK, "sighash_descriptor_hash(moderate count) returns OK");
    if (e != UFSECP_OK) return;
    bool all_match = true;
    for (size_t row = 0; row < n; ++row) {
        auto exp = fx.expected_row(row);
        if (std::memcmp(out.data() + row * 32, exp.data(), 32) != 0) { all_match = false; break; }
    }
    CHECK(all_match, "sighash_descriptor_hash(moderate count) matches oracle row-by-row");
}

// Runs one legacy-fixture call against `ctx` and CHECKs both the return code
// and the digest match against the independent oracle. Shared by calls 2 & 3
// of test_ocl_pool_context_switch() below, once the backend's availability
// has already been established by that test's first (probe) call -- so a
// non-OK return here is real signal, not a skip condition.
void check_legacy_call(ufsecp_gpu_ctx* ctx, size_t n, uint8_t tag,
                        const char* ok_msg, const char* match_msg) {
    LegacyFixture fx;
    fx.build(n, tag);
    std::vector<uint8_t> out(n * 32, 0xEE);
    ufsecp_error_t e = ufsecp_gpu_sighash_descriptor_hash(
        ctx, fx.descriptor, sizeof(fx.descriptor), fx.fdata, fx.flens, nullptr, n, out.data());
    CHECK(e == UFSECP_OK, ok_msg);
    if (e != UFSECP_OK) return;
    bool all_match = true;
    for (size_t row = 0; row < n; ++row) {
        auto exp = fx.expected_row(row);
        if (std::memcmp(out.data() + row * 32, exp.data(), 32) != 0) { all_match = false; break; }
    }
    CHECK(all_match, match_msg);
}

// -- OpenCL thread_local pool context-tracking regression --------------------
// OclSighashPool (thread_local per calling thread) now records the cl_context
// its device buffers were last allocated against (owning_ctx). ensure_meta()
// frees and reallocates whenever the SAME thread calls this op against a
// DIFFERENT cl_context than the one the pool currently owns, instead of
// handing back cl_mem handles bound to a stale/foreign context. Exercised
// here by creating TWO independent OpenCL GPU contexts from this thread and
// calling the op A -> B -> A, checking every call's digest against the same
// independent CPU oracle used everywhere else in this file.
void test_ocl_pool_context_switch() {
    std::printf("[sighash_descriptor_gpu] OpenCL pool context-switch A -> B -> A (thread_local pool re-targets device buffers)\n");
    ufsecp_gpu_ctx* ctx_a = nullptr;
    if (ufsecp_gpu_ctx_create(&ctx_a, UFSECP_GPU_BACKEND_OPENCL, 0) != UFSECP_OK || !ctx_a) {
        std::printf("  (OpenCL: ctx A create failed -- skipping pool context-switch test)\n");
        return;
    }
    ufsecp_gpu_ctx* ctx_b = nullptr;
    if (ufsecp_gpu_ctx_create(&ctx_b, UFSECP_GPU_BACKEND_OPENCL, 0) != UFSECP_OK || !ctx_b) {
        std::printf("  (OpenCL: ctx B create failed -- skipping pool context-switch test)\n");
        ufsecp_gpu_ctx_destroy(ctx_a);
        return;
    }

    const size_t n = 3;

    // Call 1 (ctx A): also serves as the availability probe -- if this op
    // itself is unsupported/unavailable on this backend, skip the whole test
    // cleanly instead of reporting three cascading FAILs below.
    LegacyFixture fx_a1;
    fx_a1.build(n, 0x10);
    std::vector<uint8_t> out_a1(n * 32, 0xEE);
    ufsecp_error_t e_a1 = ufsecp_gpu_sighash_descriptor_hash(
        ctx_a, fx_a1.descriptor, sizeof(fx_a1.descriptor), fx_a1.fdata, fx_a1.flens, nullptr, n, out_a1.data());
    if (is_skip_err(e_a1)) {
        std::printf("  (pool context-switch: backend skip %d (%s) -- skipping)\n", e_a1, ufsecp_gpu_error_str(e_a1));
        ufsecp_gpu_ctx_destroy(ctx_a);
        ufsecp_gpu_ctx_destroy(ctx_b);
        return;
    }
    CHECK(e_a1 == UFSECP_OK, "pool context-switch: call on ctx A (first alloc) returns OK");
    if (e_a1 == UFSECP_OK) {
        bool match_a1 = true;
        for (size_t row = 0; row < n; ++row) {
            auto exp = fx_a1.expected_row(row);
            if (std::memcmp(out_a1.data() + row * 32, exp.data(), 32) != 0) { match_a1 = false; break; }
        }
        CHECK(match_a1, "pool context-switch: call on ctx A (first alloc) matches oracle");
    }

    // Call 2 (ctx B): forces ensure_meta() to detect owning_ctx != ctx and
    // free_all()+reallocate against B.
    check_legacy_call(ctx_b, n, 0x20,
                       "pool context-switch: call on ctx B (forces realloc) returns OK",
                       "pool context-switch: call on ctx B (forces realloc) matches oracle");

    // Call 3 (ctx A again): forces a second free_all()+realloc, proving the
    // pool correctly tracks whichever context it was most recently used
    // against rather than only handling a single one-shot switch.
    check_legacy_call(ctx_a, n, 0x30,
                       "pool context-switch: call back on ctx A (forces second realloc) returns OK",
                       "pool context-switch: call back on ctx A (forces second realloc) matches oracle");

    ufsecp_gpu_ctx_destroy(ctx_a);
    ufsecp_gpu_ctx_destroy(ctx_b);
}

// -- OpenCL shared-kernel-object concurrency regression ----------------------
// ext_sighash_descriptor_ (the shared cl_kernel object) and the command queue
// are backend-instance members, not thread_local -- sighash_dispatch_mtx_
// now serializes the whole dispatch (buffer upload through kernel-arg-bind
// through readback) so unsynchronized concurrent clSetKernelArg calls from
// different threads can no longer interleave and launch with a mix of both
// callers' buffers. Exercised here by spawning several real std::thread
// workers that all call the op CONCURRENTLY on the SAME GPU context/backend
// instance, each with a distinct, thread-derived row payload -- a
// cross-thread mix-up would produce a visibly wrong digest, not an
// accidentally-matching one. CHECK() itself is only ever called from the
// main thread (after every worker has joined) since g_pass/g_fail are not
// synchronized for concurrent increment.
void test_ocl_concurrent_dispatch() {
    std::printf("[sighash_descriptor_gpu] OpenCL concurrent dispatch across threads (shared kernel/queue mutex)\n");
    ufsecp_gpu_ctx* ctx = nullptr;
    if (ufsecp_gpu_ctx_create(&ctx, UFSECP_GPU_BACKEND_OPENCL, 0) != UFSECP_OK || !ctx) {
        std::printf("  (OpenCL: ctx create failed -- skipping concurrent dispatch test)\n");
        return;
    }

    constexpr size_t kRows = 4;

    // Warm-up probe on the main thread, before any worker is spawned: turns a
    // genuine backend-unavailable condition into a clean skip instead of N
    // cascading thread failures below.
    LegacyFixture warm;
    warm.build(kRows, 0x05);
    std::vector<uint8_t> warm_out(kRows * 32, 0xEE);
    ufsecp_error_t warm_e = ufsecp_gpu_sighash_descriptor_hash(
        ctx, warm.descriptor, sizeof(warm.descriptor), warm.fdata, warm.flens, nullptr, kRows, warm_out.data());
    if (is_skip_err(warm_e)) {
        std::printf("  (concurrent dispatch: backend skip %d (%s) -- skipping)\n", warm_e, ufsecp_gpu_error_str(warm_e));
        ufsecp_gpu_ctx_destroy(ctx);
        return;
    }
    CHECK(warm_e == UFSECP_OK, "concurrent dispatch: warm-up call returns OK");

    constexpr int kThreads = 6;
    struct ThreadResult { ufsecp_error_t err; bool all_match; };
    std::vector<ThreadResult> results(kThreads, ThreadResult{UFSECP_ERR_INTERNAL, false});
    std::vector<std::thread> workers;
    workers.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        workers.emplace_back([ctx, t, &results, kRows]() {
            LegacyFixture fx;
            // Distinct, thread-derived tag: the exact race the
            // sighash_dispatch_mtx_ fix closes would interleave one
            // thread's kernel args with another's -- with per-thread
            // content this distinctly different, that shows up as a wrong
            // digest rather than an accidental match.
            fx.build(kRows, (uint8_t)(0x40 + t * 0x11));
            std::vector<uint8_t> out(kRows * 32, 0xEE);
            ufsecp_error_t e = ufsecp_gpu_sighash_descriptor_hash(
                ctx, fx.descriptor, sizeof(fx.descriptor), fx.fdata, fx.flens, nullptr, kRows, out.data());
            bool all_match = false;
            if (e == UFSECP_OK) {
                all_match = true;
                for (size_t row = 0; row < kRows; ++row) {
                    auto exp = fx.expected_row(row);
                    if (std::memcmp(out.data() + row * 32, exp.data(), 32) != 0) { all_match = false; break; }
                }
            }
            results[t] = ThreadResult{e, all_match};  // each thread writes only its own index -- no shared mutable state
        });
    }
    for (auto& th : workers) th.join();

    for (int t = 0; t < kThreads; ++t) {
        CHECK(results[t].err == UFSECP_OK, "concurrent dispatch: thread call returns OK");
        CHECK(results[t].all_match,
              "concurrent dispatch: thread digest matches its own oracle (no cross-thread mix-up)");
    }

    ufsecp_gpu_ctx_destroy(ctx);
}

// -- OpenCL COLD lazy-init concurrency regression -----------------------
// test_ocl_concurrent_dispatch() above deliberately warms up
// ext_sighash_descriptor_ on the main thread BEFORE spawning workers, so
// by the time any worker thread runs, ensure_extended_kernels() has
// already short-circuited on its cache-check and every worker only
// exercises the (already-locked) upload/kernel-arg/launch/readback span --
// it never exercises the lazy init itself. This variant spawns every
// worker directly against a freshly created, never-yet-dispatched ctx, so
// the very first ensure_extended_kernels() call races across threads --
// closing the gap a pre-warmed kernel would otherwise mask. Without
// sighash_dispatch_mtx_ covering the lazy init (not just the
// pool/upload/launch span), concurrent cold callers could race the
// sequential clCreateKernel/clReleaseKernel/clReleaseProgram cleanup
// chains inside ensure_extended_kernels() and its unguarded
// ext_init_attempted_ read-then-write.
void test_ocl_cold_concurrent_dispatch() {
    std::printf("[sighash_descriptor_gpu] OpenCL COLD concurrent dispatch across threads (races lazy kernel init)\n");
    ufsecp_gpu_ctx* ctx = nullptr;
    if (ufsecp_gpu_ctx_create(&ctx, UFSECP_GPU_BACKEND_OPENCL, 0) != UFSECP_OK || !ctx) {
        std::printf("  (OpenCL: ctx create failed -- skipping cold concurrent dispatch test)\n");
        return;
    }

    constexpr size_t kRows = 4;
    constexpr int kThreads = 6;
    struct ThreadResult { ufsecp_error_t err; bool all_match; };
    std::vector<ThreadResult> results(kThreads, ThreadResult{UFSECP_ERR_INTERNAL, false});
    std::vector<std::thread> workers;
    workers.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        workers.emplace_back([ctx, t, &results, kRows]() {
            LegacyFixture fx;
            // Distinct, thread-derived tag (0x80 base, disjoint from the warm
            // test's 0x40 base) so a cross-thread buffer mix-up during the
            // init race would show up as a wrong digest, not an accidental
            // match.
            fx.build(kRows, (uint8_t)(0x80 + t * 0x11));
            std::vector<uint8_t> out(kRows * 32, 0xEE);
            ufsecp_error_t e = ufsecp_gpu_sighash_descriptor_hash(
                ctx, fx.descriptor, sizeof(fx.descriptor), fx.fdata, fx.flens, nullptr, kRows, out.data());
            bool all_match = false;
            if (e == UFSECP_OK) {
                all_match = true;
                for (size_t row = 0; row < kRows; ++row) {
                    auto exp = fx.expected_row(row);
                    if (std::memcmp(out.data() + row * 32, exp.data(), 32) != 0) { all_match = false; break; }
                }
            }
            results[t] = ThreadResult{e, all_match};  // each thread writes only its own index -- no shared mutable state
        });
    }
    for (auto& th : workers) th.join();

    // No pre-flight warm-up here (that's the point) -- treat the first
    // thread's error as the availability signal, same convention as the
    // warm test's pre-spawn probe: genuine backend/device unavailability is
    // a ctx-level fact, not a per-call one, so it surfaces identically on
    // every thread.
    if (is_skip_err(results[0].err)) {
        std::printf("  (cold concurrent dispatch: backend skip %d (%s) -- skipping)\n",
                     results[0].err, ufsecp_gpu_error_str(results[0].err));
        ufsecp_gpu_ctx_destroy(ctx);
        return;
    }

    for (int t = 0; t < kThreads; ++t) {
        CHECK(results[t].err == UFSECP_OK, "cold concurrent dispatch: thread call returns OK");
        CHECK(results[t].all_match,
              "cold concurrent dispatch: thread digest matches its own oracle (no cross-thread mix-up during lazy init race)");
    }

    ufsecp_gpu_ctx_destroy(ctx);
}

void run_backend(uint32_t bid) {
    ufsecp_gpu_ctx* ctx = nullptr;
    if (ufsecp_gpu_ctx_create(&ctx, bid, 0) != UFSECP_OK || !ctx) {
        std::printf("  (%s: ctx create failed -- skipping)\n", ufsecp_gpu_backend_name(bid));
        return;
    }
    std::printf("  Backend: %s\n", ufsecp_gpu_backend_name(bid));
    test_kat_legacy_all_fixed(ctx);
    test_kat_variable_scriptcode(ctx);
    test_n_zero(ctx);
    test_moderate_count(ctx);
    ufsecp_gpu_ctx_destroy(ctx);
}

void test_on_device() {
    std::printf("[sighash_descriptor_gpu] KAT/boundary coverage vs CPU oracle (if GPU available)\n");
    uint32_t ids[8] = {};
    const uint32_t n = ufsecp_gpu_backend_count(ids, 8);
    bool any = false;
    for (uint32_t i = 0; i < n; ++i) {
        if (ufsecp_gpu_is_available(ids[i])) { any = true; run_backend(ids[i]); }
    }
    if (!any) std::printf("  (no GPU available -- skipping on-device coverage)\n");
}

}  // namespace

int test_regression_sighash_descriptor_gpu_run() {
    g_pass = 0; g_fail = 0;
    std::printf("=== sighash_descriptor_hash structural/boundary KAT ===\n");
    test_null_ctx_contract();
    test_on_device();
    test_ocl_pool_context_switch();
    test_ocl_concurrent_dispatch();
    test_ocl_cold_concurrent_dispatch();
    std::printf("[sighash_descriptor_gpu] pass=%d fail=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_sighash_descriptor_gpu_run(); }
#endif
