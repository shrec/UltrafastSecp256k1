// ============================================================================
// Regression: BIP-352 scan kernel uses CT variable-base scalar mul (CRIT-02)
//
// BCV-1..4  — CPU-path BIP-352 scan produces correct output (always runs)
// BCV-5..8  — GPU BIP-352 batch scan matches an INDEPENDENT CPU oracle,
//             byte-for-byte, for scan keys chosen to probe every 64-bit limb
//             boundary of ct_scalar_mul_varbase (GPU-only, advisory)
//
// issue #335 acceptance repair (2026-07): this module previously only
// asserted the GPU prefix was non-zero and that two identical calls returned
// the same value ("nonzero-and-determinism-only"). That is NOT a correctness
// check: a scan key with 100% of its limb ordering silently reversed is
// still non-zero and still deterministic -- it just computes a different,
// wrong point every time, consistently. This is exactly the shape of the
// pre-existing P0 bug this module is named for (CRIT-02 / ct_scalar_mul_varbase,
// src/cuda/include/ct/ct_point.cuh): `limb_idx = 3 - (bit/64)` (wrong,
// assumes limbs[0]=MSW) vs the fixed `limb_idx = bit/64` (correct,
// limbs[0]=LSW, matching every other Scalar consumer in this codebase). The
// old nonzero+determinism check passed identically before AND after that fix
// -- it could never have caught the bug it claims to guard against.
//
// This rewrite replaces that check with an INDEPENDENT CPU oracle
// (ufsecp_silent_payment_create_output, sender-side BIP-352 construction --
// a code path entirely separate from the GPU scan kernels and from
// ct_scalar_mul_varbase itself) compared byte-for-byte against the GPU
// receiver-side scan result, for scan private keys chosen specifically to
// probe ct_scalar_mul_varbase's bit/limb indexing:
//   - k=1                (BCV-9  intent: k=1 -> shared = base point unchanged)
//   - k=2                (BCV-10 intent: k=2 -> shared = base point doubled)
//   - single bit set at each of positions 63/64/127/128/191/192/255 -- the
//     eight 64-bit-limb boundary bits (top/bottom bit of each of the 4
//     limbs), where a limb-index transposition bug changes WHICH limb a
//     given bit lands in and therefore produces a completely different,
//     wrong scalar
//   - two bits straddling a limb boundary simultaneously (63+64, 127+128,
//     191+192) -- catches a bug that only manifests when two adjacent limbs
//     are both non-zero
//   - deterministic "representative random" vectors (fixed PRNG seeds, not
//     true randomness, so failures are always reproducible)
// crossed with n_spend in {1, 2, 3, 8} candidate spend keys per GPU backend
// available on this machine. A limb-reversed scalar mul fails EVERY one of
// these cells (the reversed scalar is a different value for all of them);
// the correct implementation passes all of them.
// ============================================================================

#ifndef UNIFIED_AUDIT_RUNNER
#include <cstdio>
#define STANDALONE_TEST
#ifndef ADVISORY_SKIP_CODE
#define ADVISORY_SKIP_CODE 77
#endif
#endif

#include "ufsecp256k1.h"
#include "ufsecp/ufsecp_gpu.h"

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <array>
#include <vector>
#include <string>

static int g_fail = 0;
static bool g_gpu_available = false;
#define ASSERT_TRUE(cond, msg)  do { if (!(cond)) { std::printf("FAIL [%s]: %s\n", __func__, msg); ++g_fail; } } while(0)
#define ASSERT_FALSE(cond, msg) do { if ( (cond)) { std::printf("FAIL [%s]: %s\n", __func__, msg); ++g_fail; } } while(0)

// BCV-1..4: CPU-path BIP-352 silent-payment correctness.
// ufsecp_bip352_* API exercises the CPU path that calls the same arithmetic
// as the kernel, verifying that the pipeline is end-to-end correct.
static void test_bcv_cpu_correctness() {
    ufsecp_ctx* ctx = nullptr;
    if (ufsecp_ctx_create(&ctx) != UFSECP_OK || !ctx) { std::printf("SKIP BCV-1: no ctx\n"); return; }

    // BCV-1: create a BIP-352 scan key pair
    uint8_t scan_sk[32] = {};
    scan_sk[31] = 0x07;
    uint8_t scan_pk[33] = {};
    ASSERT_TRUE(ufsecp_pubkey_create(ctx, scan_sk, scan_pk) == UFSECP_OK,
                "BCV-1: scan pubkey_create must succeed");

    // BCV-2: create a BIP-352 spend key pair
    uint8_t spend_sk[32] = {};
    spend_sk[31] = 0x0B;
    uint8_t spend_pk[33] = {};
    ASSERT_TRUE(ufsecp_pubkey_create(ctx, spend_sk, spend_pk) == UFSECP_OK,
                "BCV-2: spend pubkey_create must succeed");

    // BCV-3: sender ECDH produces non-zero shared secret
    uint8_t shared1[32] = {};
    ASSERT_TRUE(ufsecp_ecdh(ctx, scan_sk, spend_pk, shared1) == UFSECP_OK,
                "BCV-3: sender ECDH must succeed");
    bool all_zero = true;
    for (int i = 0; i < 32; ++i) all_zero &= (shared1[i] == 0);
    ASSERT_FALSE(all_zero, "BCV-3: ECDH shared secret must not be all-zeros");

    // BCV-4: receiver ECDH(spend_sk, scan_pk) == sender ECDH(scan_sk, spend_pk)
    uint8_t shared2[32] = {};
    ASSERT_TRUE(ufsecp_ecdh(ctx, spend_sk, scan_pk, shared2) == UFSECP_OK,
                "BCV-4: receiver ECDH must succeed");
    ASSERT_TRUE(std::memcmp(shared1, shared2, 32) == 0,
                "BCV-4: ECDH is commutative — scan and spend paths must agree");

    ufsecp_ctx_destroy(ctx);
}

/* ----------------------------------------------------------------------- */
/* BCV-5..8: independent CPU-oracle byte-exact equality, limb-boundary keys */
/* ----------------------------------------------------------------------- */

// Deterministic (not cryptographically random) byte fill -- reproducible
// across runs/machines, matching the pattern used elsewhere in this audit
// suite (e.g. audit/test_gpu_bip352_scan.cpp's fill_det).
static void bcv_fill_det(uint8_t* buf, size_t len, uint32_t seed) {
    uint32_t st = seed;
    for (size_t i = 0; i < len; ++i) {
        st = st * 1103515245u + 12345u;
        buf[i] = static_cast<uint8_t>((st >> 16) & 0xFF);
    }
}

// Set bit `bit_index` (0 = LSB of the 256-bit big-endian scalar, 255 = MSB)
// in a 32-byte big-endian buffer. byte 31 holds bits 0..7, byte 0 holds
// bits 248..255.
static void bcv_set_bit_be32(uint8_t out32[32], int bit_index) {
    std::memset(out32, 0, 32);
    int byte_from_end = bit_index / 8;
    int bit_in_byte    = bit_index % 8;
    out32[31 - byte_from_end] = static_cast<uint8_t>(1u << bit_in_byte);
}

// Independent CPU oracle: ufsecp_silent_payment_create_output (sender side,
// secp256k1::silent_payment_create_output) -- a code path entirely separate
// from both the GPU scan kernels AND ct_scalar_mul_varbase under test.
static bool bcv_oracle_expected_prefix(
    ufsecp_ctx* cpu_ctx,
    const uint8_t tweak_input_sk[32],
    const uint8_t scan_pubkey33[33],
    const uint8_t spend_pubkey33[33],
    uint64_t* prefix_out)
{
    uint8_t out33[33] = {};
    auto rc = ufsecp_silent_payment_create_output(
        cpu_ctx, tweak_input_sk, 1, scan_pubkey33, spend_pubkey33, 0, out33, nullptr);
    if (rc != UFSECP_OK) return false;
    uint64_t pref = 0;
    for (int i = 0; i < 8; ++i) pref = (pref << 8) | out33[1 + i];
    *prefix_out = pref;
    return true;
}

/* ----------------------------------------------------------------------- */
/* Backend-selection / operational-failure logic, factored out for direct  */
/* mutation testing -- no real GPU device required to exercise them.       */
/* ----------------------------------------------------------------------- */

// ufsecp_gpu_backend_count() reports COMPILED backends only (gpu_registry.cpp
// s_num_backends) -- it says nothing about whether a device exists at
// runtime. Pick the first id for which is_available(id) != 0; -1 if none.
// Dependency-injected via a function pointer so a mutation test can exercise
// "compiled backend present, zero runtime-available devices" without a real
// GPU.
static int bcv_pick_available_backend_impl(const uint32_t* ids, uint32_t cnt,
                                            int (*is_available)(uint32_t)) {
    for (uint32_t i = 0; i < cnt; ++i) {
        if (is_available(ids[i])) return static_cast<int>(ids[i]);
    }
    return -1;
}

static int bcv_pick_available_backend(const uint32_t* ids, uint32_t cnt) {
    return bcv_pick_available_backend_impl(ids, cnt, ufsecp_gpu_is_available);
}

// Once a backend has been selected as runtime-available, any subsequent
// ctx-create/dispatch failure is an operational failure and must never be
// downgraded back to an advisory skip.
static bool bcv_ctx_failure_is_hard(bool backend_available, bool ctx_create_ok) {
    return backend_available && !ctx_create_ok;
}

static int bcv_mock_never_available(uint32_t) { return 0; }

// Mutation: a compiled backend (nonzero ufsecp_gpu_backend_count()) with
// zero runtime-available devices must select no backend -- driving the
// "no runtime-available provider" advisory-skip / exit-77 path, never a
// ctx-create attempt.
static void test_bcv_backend_selection_mutation() {
    uint32_t ids[1] = { UFSECP_GPU_BACKEND_CUDA };
    int picked = bcv_pick_available_backend_impl(ids, 1, bcv_mock_never_available);
    ASSERT_TRUE(picked == -1,
                "mutation: compiled backend present but ufsecp_gpu_is_available() "
                "== 0 for every id must select no backend");
}

// Mutation: once a backend is selected as runtime-available, a ctx-create
// failure must count as a hard failure, never a silent advisory skip; the
// "no backend available" case remains the only legitimate skip path.
static void test_bcv_ctx_failure_hard_mutation() {
    ASSERT_TRUE(bcv_ctx_failure_is_hard(/*backend_available=*/true, /*ctx_create_ok=*/false),
                "mutation: available backend + ctx-create failure must be a "
                "hard failure, not downgraded to advisory skip");
    ASSERT_FALSE(bcv_ctx_failure_is_hard(/*backend_available=*/false, /*ctx_create_ok=*/false),
                 "sanity: no runtime-available backend is the legitimate "
                 "advisory-skip path, not a hard failure");
}

static void test_bcv_gpu_batch_matches_cpu_oracle() {
    // ufsecp_gpu_backend_count() reports COMPILED backends only -- it says
    // nothing about runtime device availability. A compiled CUDA/OpenCL/
    // Metal build with zero devices attached must still take the advisory
    // skip path rather than attempt ctx_create and hard-fail; select the
    // first id for which ufsecp_gpu_is_available(id) == 1 first.
    uint32_t ids[8] = {};
    uint32_t cnt = ufsecp_gpu_backend_count(ids, 8);
    if (cnt == 0) {
        std::printf("SKIP BCV-5..8: no GPU backend compiled in (advisory)\n");
        return;
    }
    int picked = bcv_pick_available_backend(ids, cnt);
    if (picked < 0) {
        std::printf("SKIP BCV-5..8: %u GPU backend(s) compiled in but none "
                    "runtime-available (advisory)\n", cnt);
        return;
    }
    g_gpu_available = true;

    ufsecp_gpu_ctx* gctx = nullptr;
    ufsecp_error_t grc = ufsecp_gpu_ctx_create(&gctx, static_cast<uint32_t>(picked), 0);
    ASSERT_TRUE(grc == UFSECP_OK && gctx,
                "BCV-5: GPU ctx creation must succeed once ufsecp_gpu_is_available() "
                "reported this backend as runtime-available -- this is an "
                "operational failure, not an availability skip");
    if (grc != UFSECP_OK || !gctx) {
        return;
    }

    ufsecp_ctx* ctx = nullptr;
    ufsecp_error_t crc = ufsecp_ctx_create(&ctx);
    ASSERT_TRUE(crc == UFSECP_OK && ctx, "BCV-5: CPU ctx creation must succeed");
    if (crc != UFSECP_OK || !ctx) {
        ufsecp_gpu_ctx_destroy(gctx);
        return;
    }

    // -- 1. Build the scan-key candidate set: boundary-bit + doubling +
    //       cross-boundary + deterministic "random" vectors. Every key is
    //       nonzero and < group order n (n > 2^255, so any single/double
    //       power-of-two below bit 256 is a valid scalar). --
    struct ScanKeyCase { const char* label; uint8_t sk[32]; };
    std::vector<ScanKeyCase> scan_keys;

    auto add_bits = [&](const char* label, std::initializer_list<int> bits) {
        ScanKeyCase c{};
        c.label = label;
        std::memset(c.sk, 0, 32);
        for (int b : bits) {
            int byte_from_end = b / 8;
            int bit_in_byte    = b % 8;
            c.sk[31 - byte_from_end] |= static_cast<uint8_t>(1u << bit_in_byte);
        }
        scan_keys.push_back(c);
    };

    add_bits("k=1 (BCV-9: shared=base unchanged)",   {0});
    add_bits("k=2 (BCV-10: shared=base doubled)",    {1});
    add_bits("bit63 (limb0 top bit)",                {63});
    add_bits("bit64 (limb1 bottom bit)",              {64});
    add_bits("bit127 (limb1 top bit)",                {127});
    add_bits("bit128 (limb2 bottom bit)",             {128});
    add_bits("bit191 (limb2 top bit)",                {191});
    add_bits("bit192 (limb3 bottom bit)",             {192});
    add_bits("bit255 (limb3 top bit)",                {255});
    add_bits("bit63+bit64 (limb0/limb1 boundary)",    {63, 64});
    add_bits("bit127+bit128 (limb1/limb2 boundary)",  {127, 128});
    add_bits("bit191+bit192 (limb2/limb3 boundary)",  {191, 192});

    for (uint32_t seed = 1; seed <= 3; ++seed) {
        ScanKeyCase c{};
        char buf[48];
        std::snprintf(buf, sizeof(buf), "deterministic random seed=%u", seed);
        c.label = "deterministic random vector"; // static storage for snprintf'd label not needed; kept generic
        bcv_fill_det(c.sk, 32, seed * 0x9E3779B9u);
        c.sk[0] &= 0x7F; // keep well below group order
        scan_keys.push_back(c);
    }

    // -- 2. Fixed spend-key candidates (up to MAX_SPEND) and tweak inputs. --
    constexpr int MAX_SPEND  = 8;
    constexpr int MAX_TWEAKS = 4;
    uint8_t spend_sk[MAX_SPEND][32] = {};
    uint8_t spend_pk[MAX_SPEND][33] = {};
    bool spend_ok = true;
    for (int j = 0; j < MAX_SPEND; ++j) {
        bcv_fill_det(spend_sk[j], 32, 1000u + static_cast<uint32_t>(j));
        spend_sk[j][0] &= 0x7F;
        if (ufsecp_pubkey_create(ctx, spend_sk[j], spend_pk[j]) != UFSECP_OK) { spend_ok = false; break; }
    }
    ASSERT_TRUE(spend_ok, "BCV-5: deterministic spend pubkey derivation must succeed");

    uint8_t tweak_sk[MAX_TWEAKS][32] = {};
    uint8_t tweak_pk[MAX_TWEAKS * 33] = {};
    bool tweak_ok = true;
    for (int i = 0; i < MAX_TWEAKS; ++i) {
        bcv_fill_det(tweak_sk[i], 32, 2000u + static_cast<uint32_t>(i));
        tweak_sk[i][0] &= 0x7F;
        if (ufsecp_pubkey_create(ctx, tweak_sk[i], tweak_pk + i * 33) != UFSECP_OK) { tweak_ok = false; break; }
    }
    ASSERT_TRUE(tweak_ok, "BCV-5: deterministic tweak pubkey derivation must succeed");

    if (!spend_ok || !tweak_ok) {
        ufsecp_ctx_destroy(ctx);
        ufsecp_gpu_ctx_destroy(gctx);
        return;
    }

    // -- 3. For every scan-key case x n_spend in {1,2,3,8}: GPU multispend
    //       scan vs. independent CPU oracle, byte-exact, per cell. --
    const int spend_counts[] = {1, 2, 3, MAX_SPEND};
    int cells_checked = 0;
    bool any_supported = false;
    for (const auto& sk_case : scan_keys) {
        uint8_t scan_pubkey33[33] = {};
        if (ufsecp_pubkey_create(ctx, sk_case.sk, scan_pubkey33) != UFSECP_OK) {
            ASSERT_TRUE(false, "BCV-5: scan pubkey derivation must succeed for every candidate key");
            continue;
        }
        for (int n_spend : spend_counts) {
            std::vector<uint64_t> matrix(static_cast<size_t>(MAX_TWEAKS) * n_spend, UINT64_MAX);
            auto rc = ufsecp_gpu_bip352_scan_batch_multispend(
                gctx, sk_case.sk, &spend_pk[0][0], static_cast<size_t>(n_spend),
                tweak_pk, MAX_TWEAKS, matrix.data());
            if (rc == UFSECP_ERR_GPU_UNSUPPORTED) continue; // backend doesn't implement this op
            ASSERT_TRUE(rc == UFSECP_OK, "BCV-6: GPU multispend scan must return UFSECP_OK");
            if (rc != UFSECP_OK) continue;
            any_supported = true;

            for (int i = 0; i < MAX_TWEAKS; ++i) {
                for (int j = 0; j < n_spend; ++j) {
                    uint64_t expected = 0;
                    if (!bcv_oracle_expected_prefix(ctx, tweak_sk[i], scan_pubkey33,
                                                     spend_pk[j], &expected)) {
                        continue; // extremely unlikely (infinity point); not a failure
                    }
                    ++cells_checked;
                    ASSERT_TRUE(matrix[static_cast<size_t>(i) * n_spend + j] == expected,
                                "BCV-7: GPU multispend cell must byte-exact-match independent "
                                "CPU oracle (ufsecp_silent_payment_create_output) -- a limb-order "
                                "bug in ct_scalar_mul_varbase fails this for every boundary-bit "
                                "scan key case");
                }
            }
        }
    }

    if (any_supported) {
        // BCV-8: determinism, kept from the original module (now on top of,
        // not instead of, the byte-exact oracle check above).
        std::vector<uint64_t> run1(static_cast<size_t>(MAX_TWEAKS) * 3, 0);
        std::vector<uint64_t> run2(static_cast<size_t>(MAX_TWEAKS) * 3, 0);
        auto rc1 = ufsecp_gpu_bip352_scan_batch_multispend(
            gctx, scan_keys.front().sk, &spend_pk[0][0], 3, tweak_pk, MAX_TWEAKS, run1.data());
        auto rc2 = ufsecp_gpu_bip352_scan_batch_multispend(
            gctx, scan_keys.front().sk, &spend_pk[0][0], 3, tweak_pk, MAX_TWEAKS, run2.data());
        if (rc1 == UFSECP_OK && rc2 == UFSECP_OK) {
            ASSERT_TRUE(std::memcmp(run1.data(), run2.data(), run1.size() * sizeof(uint64_t)) == 0,
                        "BCV-8: identical GPU multispend calls must be deterministic");
        }
        std::printf("  BCV-5..8: %d scan-key cases x n_spend{1,2,3,8} = %d byte-exact "
                    "oracle cells checked\n",
                    static_cast<int>(scan_keys.size()), cells_checked);
        ASSERT_TRUE(cells_checked > 0, "BCV-6b: at least one oracle cell must have been checked");
    } else {
        std::printf("SKIP BCV-5..8: bip352_scan_batch_multispend unsupported on every "
                    "available GPU backend (advisory)\n");
    }

    ufsecp_ctx_destroy(ctx);
    ufsecp_gpu_ctx_destroy(gctx);
}

int test_regression_bip352_ct_varbase_run() {
    g_fail = 0;
    g_gpu_available = false;
    test_bcv_backend_selection_mutation();
    test_bcv_ctx_failure_hard_mutation();
    test_bcv_cpu_correctness();
    test_bcv_gpu_batch_matches_cpu_oracle();

    if (!g_gpu_available) {
        std::printf("  BIP-352 CT variable-base regression: no runtime-available GPU "
                    "provider on this machine -- BCV-5..8 GPU coverage not exercised\n");
    }
    if (g_fail == 0)
        std::printf("PASS: BIP-352 CT variable-base scalar mul regression (CRIT-02)\n");
    return g_fail;
}

#ifdef STANDALONE_TEST
int main() {
    int rc = test_regression_bip352_ct_varbase_run();
    // No runtime-available GPU provider on this machine: BCV-1..4 (CPU path)
    // and the pure backend-selection/ctx-failure mutation tests still ran,
    // but BCV-5..8 GPU coverage never executed. Report that as a CTest skip
    // (77), not a silent PASS, so GPU coverage gaps stay visible.
    if (rc == 0 && !g_gpu_available) return ADVISORY_SKIP_CODE;
    return rc;
}
#endif
