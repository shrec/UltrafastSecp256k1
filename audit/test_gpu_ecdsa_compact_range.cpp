/* ============================================================================
 * UltrafastSecp256k1 -- GPU ECDSA compact-signature strict-range regression
 * ============================================================================
 * Consensus-relevant finding: for a compact ECDSA signature whose r or s is
 * OUT of the scalar range [1, n-1] (i.e. r >= n or s >= n), the verification
 * verdict was NOT uniform across the library's own paths:
 *
 *   CUDA  ecdsa_verify_batch_kernel     strict  (ecdsa_sig_parse_compact_strict)
 *   CUDA  ecdsa_verify_collect_kernel   permissive (host bytes_to_ecdsa_sig +
 *           ecdsa_verify, which only rejected zero scalars and reduced the
 *           limbs mod n)
 *   Metal ecdsa_verify_batch_compressed permissive (same mod-n reduction)
 *   Metal lbtc_ecdsa_verify_collect     permissive (same mod-n reduction)
 *   OpenCL ecdsa_verify_compressed/rows strict  (lbtc_parse_compact_signature)
 *   CPU   ufsecp_ecdsa_verify           strict  (parse_compact_strict + low-S)
 *
 * For an encoding (r, s+n) with s+n < 2^256 the scalar s+n is congruent to s
 * mod n, so the permissive paths returned the SAME verdict as for the valid
 * (r, s) -- i.e. they accepted a non-canonical compact encoding that the
 * strict paths reject. Ubiquitous txs use low-S, and ufsecp_ecdsa_sign only
 * emits canonical r/s < n, so a valid base with s < 2^256 - n is infeasible to
 * synthesize; the divergence is therefore mostly theoretical in the wild BUT
 * it still breaks the documented invariant "collect EC verdict is bit-identical
 * to verify_batch" (gpu_backend_cuda.cu, ecdsa_verify_collect_kernel) and the
 * analogous invariant in the Metal shaders.
 *
 * FIX: each backend's device-side ecdsa_verify() entry point now rejects
 * r >= n or s >= n up front (single choke point shared by single/batch/collect
 * and the sign-and-verify countermeasure path), restoring one uniform strict
 * compact contract across CPU, CUDA, OpenCL and Metal.
 *
 * This module pins the fix two ways:
 *   [A] CPU-only SOURCE GATE (always runs, every runner): locates each
 *       backend's ecdsa_verify() / ecdsa_verify_impl() function definition,
 *       brace-matches its body and asserts the strict r/s >= n guard appears
 *       INSIDE that body, before the scalar_inverse() call. A guard stranded
 *       in a helper no verify path calls no longer satisfies the gate. This
 *       FAILED against the pre-fix sources, so it is the real always-bit-rot
 *       regression gate.
 *   [B] On-device BOUNDARY-SCALAR differential (advisory; self-skips when no
 *       GPU backend is available): builds a valid base signature with a SMALL
 *       r and SMALL s (r = 1, s = 0x0307) via ufsecp_ecdsa_recover, so every
 *       congruent malleation -- (r, s+n), (r+n, s), (r+n, s+n) -- is
 *       32-byte-representable and the collect-vs-batch divergence is actually
 *       exercisable on hardware. Rows are fed through BOTH the batch and
 *       collect entrypoints and asserted uniform per-row against the CPU strict
 *       oracle; the base row must pass both (a guard that rejects everything
 *       would fail here too).
 *
 * PUBLIC-DATA inputs only; verify is variable-time by design. No GPU needed to
 * build/run the source gate; the on-device part self-skips cleanly (operational
 * errors surface as SKIP, never a false PASS), mirroring
 * test_gpu_collect_verify_parity.
 * ============================================================================ */

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "ufsecp/ufsecp.h"
#include "ufsecp/ufsecp_gpu.h"
#include "audit_check.hpp"

static int g_pass = 0;
static int g_fail = 0;

namespace {

// Group order n, big-endian 32 bytes (0xFFFFFFFF...D0364141).
static const uint8_t kOrderBe[32] = {
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE,
    0xBA, 0xAE, 0xDC, 0xE6, 0xAF, 0x48, 0xA0, 0x3B,
    0xBF, 0xD2, 0x5E, 0x8C, 0xD0, 0x36, 0x41, 0x41
};

// n-1, big-endian.
static const uint8_t kOrderMinus1Be[32] = {
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE,
    0xBA, 0xAE, 0xDC, 0xE6, 0xAF, 0x48, 0xA0, 0x3B,
    0xBF, 0xD2, 0x5E, 0x8C, 0xD0, 0x36, 0x41, 0x40
};

// 2^256 - 1, big-endian (all ones; > n).
static const uint8_t kMax32Be[32] = {
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF
};

static const uint8_t kZero32[32] = {0};

// s := s + n mod 2^256 (big-endian 32-byte add). Returns false if it would
// overflow 32 bytes (encoding not representable). This is the s-congruent
// malleation class; only representable when s < 2^256 - n.
bool be_add_order(const uint8_t* s, uint8_t out[32]) {
    unsigned carry = 0;
    for (int i = 31; i >= 0; --i) {
        unsigned v = (unsigned)s[i] + kOrderBe[i] + carry;
        out[i] = (uint8_t)v;
        carry = v >> 8;
    }
    return carry == 0;
}

// A backend runtime error (or a backend without the native entrypoints) means
// "not assertable on this runner" -> treat as skip, never a false FAIL.
bool is_skip_err(ufsecp_error_t e) {
    return e == UFSECP_ERR_GPU_LAUNCH || e == UFSECP_ERR_GPU_MEMORY ||
           e == UFSECP_ERR_GPU_BACKEND || e == UFSECP_ERR_GPU_QUEUE ||
           e == UFSECP_ERR_GPU_DEVICE || e == UFSECP_ERR_GPU_UNSUPPORTED;
}

void copy32(const uint8_t src[32], uint8_t dst[32]) { std::memcpy(dst, src, 32); }

// ---------------------------------------------------------------------------
// [A] CPU-only source gate -- runs on EVERY runner, no GPU required. Each
//     backend's device ecdsa_verify() must carry the strict >= n guard, scoped
//     to the actual function body and ordered before the scalar_inverse() call;
//     the CUDA batch kernel must keep its strict compact parse. These CHECKs
//     fail against the pre-fix sources (and against a guard stranded in a
//     helper no verify path calls), so this section trees the regression even
//     on CPU-only CI (no GPU ever exercised it).
// ---------------------------------------------------------------------------

// Extract the body of a function whose signature starts at sig_pos: from the
// opening brace to the matching close, skipping strings, line and block
// comments. Empty string on any failure (which the CHECKs then flag).
std::string extract_function_body(const std::string& src, size_t sig_pos) {
    const size_t n = src.size();
    const size_t brace = src.find('{', sig_pos);
    if (brace == std::string::npos) return {};
    int depth = 0;
    bool in_str = false, in_line = false, in_block = false;
    for (size_t i = brace; i < n; ++i) {
        const char c = src[i];
        if (in_line) {
            if (c == '\n') in_line = false;
            continue;
        }
        if (in_block) {
            if (c == '*' && i + 1 < n && src[i + 1] == '/') { in_block = false; ++i; }
            continue;
        }
        if (in_str) {
            if (c == '\\') { ++i; continue; }
            if (c == '"') in_str = false;
            continue;
        }
        if (c == '/' && i + 1 < n) {
            if (src[i + 1] == '/') { in_line = true; ++i; continue; }
            if (src[i + 1] == '*') { in_block = true; ++i; continue; }
        }
        if (c == '"') { in_str = true; continue; }
        if (c == '{') { ++depth; continue; }
        if (c == '}') {
            --depth;
            if (depth == 0) return src.substr(brace + 1, i - brace - 1);
        }
    }
    return {};
}

// True when guard appears in body before the scalar_inverse() call
// ("scalar_inverse" also matches scalar_inverse_impl in the OpenCL kernel).
bool guard_before_inverse(const std::string& body, const std::string& guard) {
    const size_t g = body.find(guard);
    if (g == std::string::npos) return false;
    const size_t inv = body.find("scalar_inverse");
    return inv != std::string::npos && g < inv;
}

void test_source_gate() {
    AUDIT_LOG("[gpu_ecdsa_compact_range] source gate (CPU-only, always runs)\n");

    // -- CUDA: strict guard inside ecdsa_verify(), before scalar_inverse(),
    //    and the batch kernel keeps its strict compact parse. ------------
    {
        std::string cuh = audit_read_source_file("src/cuda/include/ecdsa.cuh");
        CHECK(!cuh.empty(), "src/cuda/include/ecdsa.cuh must be readable (in-tree source)");
        if (!cuh.empty()) {
            const size_t sig = cuh.find("bool ecdsa_verify(");
            const std::string body = extract_function_body(cuh, sig);
            CHECK(!body.empty(), "[A] CUDA ecdsa_verify( definition found and brace-matched");
            if (!body.empty()) {
                CHECK(guard_before_inverse(body, "scalar_ge(&sig->r, ORDER)"),
                      "[A] CUDA ecdsa_verify rejects r >= n inside its body, before scalar_inverse");
                CHECK(guard_before_inverse(body, "scalar_ge(&sig->s, ORDER)"),
                      "[A] CUDA ecdsa_verify rejects s >= n inside its body, before scalar_inverse");
            }
        }
        std::string cu = audit_read_source_file("src/cuda/src/secp256k1.cu");
        CHECK(!cu.empty(), "src/cuda/src/secp256k1.cu must be readable (in-tree source)");
        if (!cu.empty()) {
            CHECK(cu.find("ecdsa_sig_parse_compact_strict") != std::string::npos,
                  "[A] CUDA batch kernel keeps strict compact parse (parse_compact_strict)");
        }
    }

    // -- Metal: guard inside the 3-arg ecdsa_verify( overload -- the choke
    //    point the 4-arg overload and both batch/collect kernels delegate to --
    //    comparing sig.r/sig.s against the order built from SECP256K1_N. ---
    {
        std::string mh = audit_read_source_file("src/metal/shaders/secp256k1_extended.h");
        CHECK(!mh.empty(), "src/metal/shaders/secp256k1_extended.h must be readable (in-tree source)");
        if (!mh.empty()) {
            const size_t sig = mh.find("bool ecdsa_verify(thread const uchar");
            const std::string body = extract_function_body(mh, sig);
            CHECK(!body.empty(), "[A] Metal ecdsa_verify( 3-arg definition found and brace-matched");
            if (!body.empty()) {
                CHECK(guard_before_inverse(body, "scalar256_ge(sig.r, order_n)"),
                      "[A] Metal ecdsa_verify rejects r >= n inside its body, before scalar_inverse");
                CHECK(guard_before_inverse(body, "scalar256_ge(sig.s, order_n)"),
                      "[A] Metal ecdsa_verify rejects s >= n inside its body, before scalar_inverse");
            }
        }
    }

    // -- OpenCL: guard inside ecdsa_verify_impl(), before
    //    scalar_inverse_impl(). Defense in depth (OpenCL parsers were already
    //    strict) but the choke point must still carry it. -----------------
    {
        std::string cl = audit_read_source_file("src/opencl/kernels/secp256k1_extended.cl");
        CHECK(!cl.empty(), "src/opencl/kernels/secp256k1_extended.cl must be readable (in-tree source)");
        if (!cl.empty()) {
            const size_t sig = cl.find("int ecdsa_verify_impl(");
            const std::string body = extract_function_body(cl, sig);
            CHECK(!body.empty(), "[A] OpenCL ecdsa_verify_impl( definition found and brace-matched");
            if (!body.empty()) {
                CHECK(guard_before_inverse(body, "lbtc_scalar_ge_order(&sig->r)"),
                      "[A] OpenCL ecdsa_verify_impl rejects r >= n inside its body, before scalar_inverse");
                CHECK(guard_before_inverse(body, "lbtc_scalar_ge_order(&sig->s)"),
                      "[A] OpenCL ecdsa_verify_impl rejects s >= n inside its body, before scalar_inverse");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// [B] On-device boundary-scalar differential (advisory; self-skips w/o GPU).
//     Builds one valid compact row whose r and s are SMALL (r = 1, s = 0x0307)
//     by signing-recovering the matching public key, so every congruent
//     malleation (r, s+n), (r+n, s), (r+n, s+n) fits in 32 bytes and actually
//     reaches the device. Verdicts are asserted uniform with the CPU strict
//     oracle (ufsecp_ecdsa_verify: parse_compact_strict + low-S).
// ---------------------------------------------------------------------------
struct Row {
    const char* label;
    bool       want_valid;   // expected per the CPU strict oracle
};

// Sign-recover the pubkey for a hand-built small-(r, s) signature.
//   r[31] = r_small, s[62] = 0x03, s[63] = 0x07  ->  r = small, s = 0x0307.
// A canonical low-S ufsecp_ecdsa_sign output (s ~ 2^255) would make s+n
// unrepresentable in 32 bytes; a small s makes the malleation class reachable.
// Returns false only if 32 candidates fail to recover (infrastructure issue).
bool make_small_valid_row(ufsecp_ctx* sc, uint8_t msg[32], uint8_t pub33[33],
                          uint8_t sig64[64]) {
    for (int j = 0; j < 32; ++j)
        msg[j] = (uint8_t)(0xa5 + j * 7);
    uint8_t sig[64];
    for (uint8_t r_small = 1; r_small < 32; ++r_small) {
        std::memset(sig, 0, 64);
        sig[31] = r_small;                       // r = r_small
        sig[62] = 0x03; sig[63] = 0x07;          // s = 0x0307 (tiny, low-S)
        for (int recid = 0; recid < 2; ++recid) {
            if (ufsecp_ecdsa_recover(sc, msg, sig, recid, pub33) != UFSECP_OK) continue;
            if (ufsecp_ecdsa_verify(sc, msg, sig, pub33) == UFSECP_OK) {
                std::memcpy(sig64, sig, 64);
                return true;
            }
        }
    }
    return false;
}

// Build the boundary corpus and run batch+collect on one backend id.
void run_backend(uint32_t bid) {
    ufsecp_gpu_ctx* ctx = nullptr;
    if (ufsecp_gpu_ctx_create(&ctx, bid, 0) != UFSECP_OK || !ctx) {
        AUDIT_LOG("  (%s: ctx create failed -- skipping)\n", ufsecp_gpu_backend_name(bid));
        return;
    }
    ufsecp_ctx* sc = nullptr;
    if (ufsecp_ctx_create(&sc) != UFSECP_OK || !sc) {
        AUDIT_LOG("  (%s: cpu ctx create failed -- skipping)\n", ufsecp_gpu_backend_name(bid));
        ufsecp_gpu_ctx_destroy(ctx);
        return;
    }
    AUDIT_LOG("  Backend: %s\n", ufsecp_gpu_backend_name(bid));

    uint8_t msg[32], pub[33], base[64];
    if (!make_small_valid_row(sc, msg, pub, base)) {
        AUDIT_LOG("  (ufsecp_ecdsa_recover infra unavailable -- skip boundary differential)\n");
        ufsecp_ctx_destroy(sc);
        ufsecp_gpu_ctx_destroy(ctx);
        return;
    }

    // Row corpus: (label, expected-by-CPU-strict-oracle). Rows 1-3 are the
    // congruent-malleation class of the small base -- representable in 32
    // bytes because r = 1 and s = 0x0307 are far below 2^256 - n.
    std::vector<Row> rows;
    rows.push_back({"valid base (r=small, s=small)", true});
    rows.push_back({"r, s+n (congruent malleation)", false});
    rows.push_back({"r+n, s  (congruent malleation)", false});
    rows.push_back({"r+n, s+n (congruent malleation)", false});
    rows.push_back({"r = n", false});
    rows.push_back({"s = n", false});
    rows.push_back({"r = n-1", false});
    rows.push_back({"s = n-1", false});
    rows.push_back({"r = 0", false});
    rows.push_back({"s = 0", false});
    rows.push_back({"r = 2^256-1", false});
    rows.push_back({"s = 2^256-1", false});

    const size_t M = rows.size();
    std::vector<uint8_t> dig(M * 32), pks(M * 33), sigs(M * 64);

    // r/s := wildcard per-row; then the congruent rows get +n via 32-byte add.
    bool all_congruent_representable = true;
    for (size_t i = 0; i < M; ++i) {
        std::memcpy(&dig[i * 32], msg, 32);
        std::memcpy(&pks[i * 33], pub, 33);
        uint8_t r[32], s[32];
        copy32(base, r);
        copy32(base + 32, s);
        switch (i) {
            case 0: break;  // base (valid, small r/s)
            case 1: break;  // (r, s+n)
            case 2: break;  // (r+n, s)
            case 3: break;  // (r+n, s+n)
            case 4: copy32(kOrderBe, r); break;
            case 5: copy32(kOrderBe, s); break;
            case 6: copy32(kOrderMinus1Be, r); break;
            case 7: copy32(kOrderMinus1Be, s); break;
            case 8: copy32(kZero32, r); break;
            case 9: copy32(kZero32, s); break;
            case 10: copy32(kMax32Be, r); break;
            case 11: copy32(kMax32Be, s); break;
            default: break;
        }
        if (i == 1) all_congruent_representable &= be_add_order(base + 32, s);  // s := s+n
        if (i == 2) all_congruent_representable &= be_add_order(base, r);        // r := r+n
        if (i == 3) {
            all_congruent_representable &= be_add_order(base + 32, s);           // s := s+n
            all_congruent_representable &= be_add_order(base, r);                // r := r+n
        }
        std::memcpy(&sigs[i * 64], r, 32);
        std::memcpy(&sigs[i * 64 + 32], s, 32);

        // CPU strict oracle for this row.
        rows[i].want_valid = (ufsecp_ecdsa_verify(sc, msg, &sigs[i * 64], pub) == UFSECP_OK);
    }

    // With the small base the s+n / r+n classes MUST be representable: if this
    // ever stops holding, the differential has silently lost its point.
    CHECK(all_congruent_representable,
          "[B] congruent-malleation rows (r, s+n)/(r+n, s)/(r+n, s+n) are all 32-byte-representable");
    CHECK(rows[0].want_valid, "[B] CPU oracle accepts the recovered small-(r, s) base row");
    CHECK(!rows[1].want_valid, "[B] CPU oracle rejects (r, s+n) (strict parse rejects s >= n)");
    CHECK(!rows[2].want_valid, "[B] CPU oracle rejects (r+n, s) (strict parse rejects r >= n)");
    CHECK(!rows[3].want_valid, "[B] CPU oracle rejects (r+n, s+n)");

    // ---- batch ----
    std::vector<uint8_t> batch(M, 2);
    ufsecp_error_t eb = ufsecp_gpu_ecdsa_verify_batch(ctx, dig.data(), pks.data(), sigs.data(), M, batch.data());
    if (is_skip_err(eb)) { AUDIT_LOG("  (ecdsa batch: runtime skip %d (%s))\n", eb, ufsecp_gpu_error_str(eb)); ufsecp_ctx_destroy(sc); ufsecp_gpu_ctx_destroy(ctx); return; }
    CHECK(eb == UFSECP_OK, "[B] verify_batch boundary corpus -> OK");

    // ---- collect ----
    const uint8_t kSeed = 0xEE;
    std::vector<uint8_t> key(M, kSeed);
    ufsecp_error_t ec = ufsecp_gpu_ecdsa_verify_collect(ctx, dig.data(), pks.data(), sigs.data(), M, key.data());
    if (is_skip_err(ec)) { AUDIT_LOG("  (ecdsa collect: runtime skip %d (%s))\n", ec, ufsecp_gpu_error_str(ec)); ufsecp_ctx_destroy(sc); ufsecp_gpu_ctx_destroy(ctx); return; }
    CHECK(ec == UFSECP_OK, "[B] ecdsa_verify_collect boundary corpus -> OK");

    if (eb == UFSECP_OK && ec == UFSECP_OK) {
        bool batch_match = true, collect_match = true, per_row = true, seeded = true;
        for (size_t i = 0; i < M; ++i) {
            const bool ok = rows[i].want_valid;
            if (batch[i] != (ok ? 1u : 0u)) batch_match = false;
            if ((key[i] == 0) != ok) collect_match = false;
            if ((key[i] == 0) != (batch[i] == 1)) per_row = false;
            if (!ok && key[i] != kSeed) seeded = false;   // invalid row stays seeded
        }
        CHECK(batch_match,   "[B] batch verdict == CPU strict oracle per boundary row");
        CHECK(collect_match, "[B] collect verdict == CPU strict oracle per boundary row");
        CHECK(per_row,       "[B] collect == verify_batch verdict per boundary row");
        CHECK(seeded,        "[B] invalid rows are left at the seeded marker (fail-closed)");

        // Explicit pins so a guard that rejects EVERYTHING cannot pass: the
        // valid base must verify on both entrypoints and each congruent
        // malleation must be rejected on both.
        CHECK(batch[0] == 1u && key[0] == 0,
              "[B] valid base accepted by BOTH verify_batch and ecdsa_verify_collect");
        CHECK(batch[1] == 0u && key[1] == kSeed,
              "[B] (r, s+n) rejected by BOTH entrypoints (collect cell stays seeded)");
        CHECK(batch[2] == 0u && key[2] == kSeed,
              "[B] (r+n, s) rejected by BOTH entrypoints (collect cell stays seeded)");
        CHECK(batch[3] == 0u && key[3] == kSeed,
              "[B] (r+n, s+n) rejected by BOTH entrypoints (collect cell stays seeded)");
    }

    ufsecp_ctx_destroy(sc);
    ufsecp_gpu_ctx_destroy(ctx);
}

void test_backend_differential() {
    AUDIT_LOG("[gpu_ecdsa_compact_range] boundary-scalar differential (if GPU available)\n");
    uint32_t ids[8] = {};
    const uint32_t n = ufsecp_gpu_backend_count(ids, 8);
    bool any = false;
    for (uint32_t i = 0; i < n; ++i) {
        if (ufsecp_gpu_is_available(ids[i])) { any = true; run_backend(ids[i]); }
    }
    if (!any) AUDIT_LOG("  (no GPU available -- skipping on-device differential)\n");
}

}  // namespace

int test_gpu_ecdsa_compact_range_run() {
    g_pass = 0; g_fail = 0;
    AUDIT_LOG("=== GPU ECDSA compact-sig strict-range regression ===\n");
    test_source_gate();
    test_backend_differential();
    AUDIT_LOG("[gpu_ecdsa_compact_range] pass=%d fail=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() { return test_gpu_ecdsa_compact_range_run(); }
#endif
