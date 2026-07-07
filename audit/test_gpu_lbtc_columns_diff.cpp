/* ============================================================================
 * UltrafastSecp256k1 -- GPU libbitcoin COLUMN verify differential test
 * ============================================================================
 * Differentially validates the native GPU columnar verify entrypoints
 *   ufsecp_gpu_ecdsa_verify_lbtc_columns   (digests32 | pubkeys33 | opaque-LE sig64)
 *   ufsecp_gpu_schnorr_verify_lbtc_columns (digests32 | xonly32   | BIP-340 sig64)
 * against an INDEPENDENT CPU per-row oracle (ufsecp_ecdsa_verify /
 * ufsecp_schnorr_verify) over the SAME Structure-of-Arrays corpus.
 *
 * Cases: valid batch, tampered rows, malformed pubkeys/signatures, null/zero-count
 * handling, and forced small chunks (UFSECP_GPU_COLUMNS_CHUNK, an internal/test
 * knob — never a caller-facing parameter).
 *
 * Design notes (per project GPU/libbitcoin contract):
 *   - GPU is an INTERNAL accelerator. This test drives the C ABI completeness
 *     surface directly; the libbitcoin direct caller surface is separate and not
 *     exercised here.
 *   - Results are uint8_t 1=valid / 0=invalid (host contract).
 *   - Operational GPU failures (software/emulated device on CI) surface as
 *     non-OK GpuError and are treated as SKIP, never a false PASS and never a
 *     consensus-invalid all-zero result.
 *
 * No GPU required to build/run: without hardware the null-ctx contract checks
 * still run; hardware-only differential checks are skipped cleanly.
 * ============================================================================ */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>

#include "ufsecp/ufsecp.h"
#include "ufsecp/ufsecp_gpu.h"
#include "secp256k1/batch_verify.hpp"

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg)                                      \
    do {                                                      \
        if (cond) { ++g_pass; }                               \
        else { ++g_fail; std::printf("  FAIL: %s\n", msg); }  \
    } while (0)

namespace {

// A backend runtime error means "GPU not usable on this runner" — treat as skip.
bool is_skip_err(ufsecp_error_t e) {
    return e == UFSECP_ERR_GPU_LAUNCH || e == UFSECP_ERR_GPU_MEMORY ||
           e == UFSECP_ERR_GPU_BACKEND || e == UFSECP_ERR_GPU_QUEUE ||
           e == UFSECP_ERR_GPU_DEVICE || e == UFSECP_ERR_GPU_UNSUPPORTED;
}

// Portable set/clear for the internal UFSECP_GPU_COLUMNS_CHUNK test knob. POSIX has
// setenv/unsetenv; MSVC/Windows provides neither, so use _putenv_s (both declared in
// <cstdlib>). Setting an empty value with _putenv_s removes the variable, matching
// unsetenv semantics so getenv() returns NULL after portable_unsetenv on every target.
void portable_setenv(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}
void portable_unsetenv(const char* name) {
#if defined(_WIN32)
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

// Build M valid ECDSA rows in column (SoA) layout. sigs are opaque-LE (GPU input);
// compact sigs are kept for the independent CPU oracle. Returns false if the CPU
// signing infrastructure is unavailable.
bool build_ecdsa(ufsecp_ctx* sc, size_t M,
                 std::vector<uint8_t>& dig, std::vector<uint8_t>& pub,
                 std::vector<uint8_t>& sig_opaque, std::vector<uint8_t>& sig_compact) {
    dig.assign(M * 32, 0);
    pub.assign(M * 33, 0);
    sig_opaque.assign(M * 64, 0);
    sig_compact.assign(M * 64, 0);
    for (size_t i = 0; i < M; ++i) {
        uint8_t sk[32] = {0};
        sk[31] = (uint8_t)((i % 250) + 3);
        sk[30] = (uint8_t)((i >> 8) & 0xff);
        uint8_t msg[32];
        for (int j = 0; j < 32; ++j) msg[j] = (uint8_t)(i * 13 + j * 7 + 1);
        uint8_t p33[33], s64[64], op64[64];
        if (ufsecp_pubkey_create(sc, sk, p33) != UFSECP_OK) return false;
        if (ufsecp_ecdsa_sign(sc, msg, sk, s64) != UFSECP_OK) return false;
        if (ufsecp_ecdsa_sig_compact_to_opaque(sc, s64, op64) != UFSECP_OK) return false;
        std::memcpy(&dig[i * 32], msg, 32);
        std::memcpy(&pub[i * 33], p33, 33);
        std::memcpy(&sig_opaque[i * 64], op64, 64);
        std::memcpy(&sig_compact[i * 64], s64, 64);
    }
    return true;
}

// Build M valid Schnorr rows in column layout. xonly = compressed pubkey x-coord
// (drop the SEC1 prefix). sig is BIP-340 (GPU + CPU consume it directly).
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

// CPU oracle: 1 if row i verifies on the CPU single-sig path, else 0.
uint8_t cpu_ecdsa_ok(ufsecp_ctx* sc, const uint8_t* dig, const uint8_t* pub,
                     const uint8_t* compact, size_t i) {
    return ufsecp_ecdsa_verify(sc, dig + i * 32, compact + i * 64, pub + i * 33) == UFSECP_OK ? 1 : 0;
}
uint8_t cpu_schnorr_ok(ufsecp_ctx* sc, const uint8_t* dig, const uint8_t* xonly,
                       const uint8_t* sig, size_t i) {
    return ufsecp_schnorr_verify(sc, dig + i * 32, sig + i * 64, xonly + i * 32) == UFSECP_OK ? 1 : 0;
}

// Compare a full GPU column result against the CPU per-row oracle.
bool matches_cpu_ecdsa(ufsecp_ctx* sc, const std::vector<uint8_t>& dig,
                       const std::vector<uint8_t>& pub, const std::vector<uint8_t>& compact,
                       const std::vector<uint8_t>& gpu_out, size_t M) {
    for (size_t i = 0; i < M; ++i)
        if (gpu_out[i] != cpu_ecdsa_ok(sc, dig.data(), pub.data(), compact.data(), i))
            return false;
    return true;
}
bool matches_cpu_schnorr(ufsecp_ctx* sc, const std::vector<uint8_t>& dig,
                         const std::vector<uint8_t>& xonly, const std::vector<uint8_t>& sig,
                         const std::vector<uint8_t>& gpu_out, size_t M) {
    for (size_t i = 0; i < M; ++i)
        if (gpu_out[i] != cpu_schnorr_ok(sc, dig.data(), xonly.data(), sig.data(), i))
            return false;
    return true;
}

// -- null-ctx contract (no GPU required) ----------------------------------------
void test_null_ctx_contract() {
    std::printf("[gpu_lbtc_columns_diff] null-ctx contract (no GPU needed)\n");
    uint8_t d[32] = {0}, p[33] = {0}, s[64] = {0}, out[1] = {0};
    CHECK(ufsecp_gpu_ecdsa_verify_lbtc_columns(nullptr, d, p, s, 1, out) == UFSECP_ERR_NULL_ARG,
          "ecdsa_verify_lbtc_columns(NULL ctx) -> NULL_ARG");
    CHECK(ufsecp_gpu_schnorr_verify_lbtc_columns(nullptr, d, p, s, 1, out) == UFSECP_ERR_NULL_ARG,
          "schnorr_verify_lbtc_columns(NULL ctx) -> NULL_ARG");
}

// -- differential over an available GPU backend ---------------------------------
void test_differential() {
    std::printf("[gpu_lbtc_columns_diff] GPU vs CPU columns differential (if available)\n");

    uint32_t ids[4] = {};
    const uint32_t n = ufsecp_gpu_backend_count(ids, 4);
    uint32_t avail = 0;
    for (uint32_t i = 0; i < n; ++i)
        if (ufsecp_gpu_is_available(ids[i])) { avail = ids[i]; break; }
    if (avail == 0) { std::printf("  (no GPU available -- skipping differential)\n"); return; }
    std::printf("  Using backend: %s\n", ufsecp_gpu_backend_name(avail));

    ufsecp_gpu_ctx* ctx = nullptr;
    if (ufsecp_gpu_ctx_create(&ctx, avail, 0) != UFSECP_OK || !ctx) {
        std::printf("  (gpu ctx create failed -- skipping)\n");
        return;
    }
    ufsecp_ctx* sc = nullptr;
    if (ufsecp_ctx_create(&sc) != UFSECP_OK || !sc) {
        std::printf("  (cpu ctx create failed -- skipping)\n");
        ufsecp_gpu_ctx_destroy(ctx);
        return;
    }

    const size_t M = 512;

    // ---------- ECDSA ----------
    {
        std::vector<uint8_t> dig, pub, sopq, scmp;
        if (build_ecdsa(sc, M, dig, pub, sopq, scmp)) {
            std::vector<uint8_t> out(M, 0xEE);
            ufsecp_error_t e = ufsecp_gpu_ecdsa_verify_lbtc_columns(ctx, dig.data(), pub.data(), sopq.data(), M, out.data());
            if (e == UFSECP_OK) {
                bool all_valid = true;
                for (size_t i = 0; i < M; ++i) if (out[i] != 1) all_valid = false;
                CHECK(all_valid, "ecdsa columns: all valid rows verify (==1)");
                CHECK(matches_cpu_ecdsa(sc, dig, pub, scmp, out, M), "ecdsa columns == CPU per-row oracle (valid)");

                // Tampered rows: flip a signature byte -> those rows must fail.
                std::vector<uint8_t> t = sopq;
                const size_t bad[3] = {5, 137, 400};
                for (size_t k = 0; k < 3; ++k) t[bad[k] * 64 + 10] ^= 0x55;
                std::vector<uint8_t> out2(M, 0xEE);
                e = ufsecp_gpu_ecdsa_verify_lbtc_columns(ctx, dig.data(), pub.data(), t.data(), M, out2.data());
                if (e == UFSECP_OK) {
                    bool ok = true;
                    for (size_t k = 0; k < 3; ++k) if (out2[bad[k]] != 0) ok = false;
                    CHECK(ok, "ecdsa columns: tampered rows zeroed (==0)");
                    CHECK(out2[0] == 1 && out2[M - 1] == 1, "ecdsa columns: untouched rows still valid");
                } else if (is_skip_err(e)) { std::printf("  (ecdsa tamper: runtime skip)\n"); }
                else CHECK(0, "ecdsa columns tamper: unexpected error");

                // Malformed pubkey (bad SEC1 prefix) + malformed sig (r >= n) -> 0.
                std::vector<uint8_t> mp = pub, ms = sopq;
                mp[7 * 33] = 0x00;                              // invalid prefix
                for (int j = 0; j < 32; ++j) ms[9 * 64 + j] = 0xFF;  // r >= n
                std::vector<uint8_t> out3(M, 0xEE);
                e = ufsecp_gpu_ecdsa_verify_lbtc_columns(ctx, dig.data(), mp.data(), ms.data(), M, out3.data());
                if (e == UFSECP_OK) {
                    CHECK(out3[7] == 0, "ecdsa columns: malformed pubkey row zeroed");
                    CHECK(out3[9] == 0, "ecdsa columns: malformed sig (r>=n) row zeroed");
                    CHECK(out3[0] == 1, "ecdsa columns: well-formed row alongside malformed still valid");
                } else if (is_skip_err(e)) { std::printf("  (ecdsa malformed: runtime skip)\n"); }

                // Consensus-differential rows the GPU backend must match against
                // the CPU column reference:
                //   row hs  : high-S ECDSA sig (S' = n - S > n/2), valid in the
                //             libbitcoin consensus/direct verify path.
                //   row xge : compressed pubkey with x-coord >= field prime p,
                //             invalid under strict decompression.
                // No hardcoded n/p: S' = n - S via ufsecp_seckey_negate (x <- -x mod n,
                // strict-nonzero-<n parse); a low-S signature's S is nonzero and < n, so
                // its negation is a valid high-S scalar. x >= p uses the all-0xFF X
                // encoding, which is > p (p = FFFF..FFFFFC2F) yet a well-formed 33-byte
                // 0x02||X compressed layout the strict decompressor must reject.
                {
                    const size_t hs = 128;   // high-S row (non-first)
                    const size_t xge = 300;  // x>=p pubkey row (non-first)

                    std::vector<uint8_t> hp = pub, hopq = sopq;

                    // Build high-S: negate the compact S (bytes 32..63 of scmp[hs]),
                    // re-encode to opaque. Skip cleanly if the ABI declines the scalar.
                    uint8_t s_hi[32];
                    std::memcpy(s_hi, &scmp[hs * 64 + 32], 32);
                    bool hs_ready = (ufsecp_seckey_negate(sc, s_hi) == UFSECP_OK);
                    if (hs_ready) {
                        uint8_t cmp_hi[64];
                        std::memcpy(cmp_hi, &scmp[hs * 64], 32);   // R unchanged
                        std::memcpy(cmp_hi + 32, s_hi, 32);        // S' = n - S
                        uint8_t opq_hi[64];
                        hs_ready = (ufsecp_ecdsa_sig_compact_to_opaque(sc, cmp_hi, opq_hi) == UFSECP_OK);
                        if (hs_ready) std::memcpy(&hopq[hs * 64], opq_hi, 64);
                    }

                    // Build x>=p pubkey: 0x02 || 0xFF..FF (32 bytes) at row xge.
                    hp[xge * 33] = 0x02;
                    for (int j = 1; j < 33; ++j) hp[xge * 33 + j] = 0xFF;

                    std::vector<uint8_t> outc(M, 0xEE);
                    e = ufsecp_gpu_ecdsa_verify_lbtc_columns(ctx, dig.data(), hp.data(), hopq.data(), M, outc.data());
                    if (e == UFSECP_OK) {
                        // Independent CPU column reference over the SAME corpus (this is
                        // the is_low_s + strict-decompress path, NOT the high-S-normalizing
                        // single-sig verify), forced onto CPU by clearing any hook.
                        secp256k1::GpuColumnsVerifyHook saved =
                            secp256k1::install_gpu_columns_verify_hook(nullptr);
                        std::vector<uint8_t> cpu_ref(M, 0xEE);
                        (void)secp256k1::ecdsa_batch_verify_opaque_columns(
                            dig.data(), hp.data(), hopq.data(), M, cpu_ref.data());
                        secp256k1::install_gpu_columns_verify_hook(saved);

                        bool row_match = true;
                        for (size_t i = 0; i < M; ++i) if (outc[i] != cpu_ref[i]) row_match = false;
                        CHECK(row_match, "ecdsa columns: GPU verdict == CPU column reference (high-S + x>=p corpus)");
                        if (hs_ready)
                            CHECK(outc[hs] == 1 && cpu_ref[hs] == 1,
                                  "ecdsa columns: high-S (s>n/2) accepted on GPU and CPU");
                        else
                            std::printf("  (ecdsa high-S: ABI declined scalar negate -- skip hs assert)\n");
                        CHECK(outc[xge] == 0 && cpu_ref[xge] == 0,
                              "ecdsa columns: pubkey x>=p rejected on GPU and CPU");
                        CHECK(outc[0] == 1 && cpu_ref[0] == 1,
                              "ecdsa columns: control row valid alongside high-S/x>=p rows");
                    } else if (is_skip_err(e)) { std::printf("  (ecdsa high-S/x>=p: runtime skip)\n"); }
                    else CHECK(0, "ecdsa columns high-S/x>=p: unexpected error");
                }

                // Forced small chunks: exercise the engine-internal chunk loop.
                portable_setenv("UFSECP_GPU_COLUMNS_CHUNK", "64");
                std::vector<uint8_t> out4(M, 0xEE);
                e = ufsecp_gpu_ecdsa_verify_lbtc_columns(ctx, dig.data(), pub.data(), sopq.data(), M, out4.data());
                portable_unsetenv("UFSECP_GPU_COLUMNS_CHUNK");
                if (e == UFSECP_OK)
                    CHECK(matches_cpu_ecdsa(sc, dig, pub, scmp, out4, M),
                          "ecdsa columns: forced small chunks (64) == CPU oracle across boundaries");
                else if (is_skip_err(e)) std::printf("  (ecdsa small-chunk: runtime skip)\n");

                // zero-count no-op + null buffer rejection.
                CHECK(ufsecp_gpu_ecdsa_verify_lbtc_columns(ctx, dig.data(), pub.data(), sopq.data(), 0, out.data()) == UFSECP_OK,
                      "ecdsa columns(count=0) -> OK no-op");
                CHECK(ufsecp_gpu_ecdsa_verify_lbtc_columns(ctx, nullptr, pub.data(), sopq.data(), M, out.data()) == UFSECP_ERR_NULL_ARG,
                      "ecdsa columns(NULL digests) -> NULL_ARG");
            } else if (is_skip_err(e)) {
                std::printf("  (ecdsa columns: backend runtime err %d (%s) -- skip)\n", e, ufsecp_gpu_error_str(e));
            } else {
                CHECK(0, "ecdsa columns valid batch: unexpected error");
            }
        } else {
            std::printf("  (ecdsa signing infra unavailable -- skip ecdsa)\n");
        }
    }

    // ---------- Schnorr ----------
    {
        std::vector<uint8_t> dig, xonly, sig;
        if (build_schnorr(sc, M, dig, xonly, sig)) {
            std::vector<uint8_t> out(M, 0xEE);
            ufsecp_error_t e = ufsecp_gpu_schnorr_verify_lbtc_columns(ctx, dig.data(), xonly.data(), sig.data(), M, out.data());
            if (e == UFSECP_OK) {
                bool all_valid = true;
                for (size_t i = 0; i < M; ++i) if (out[i] != 1) all_valid = false;
                CHECK(all_valid, "schnorr columns: all valid rows verify (==1)");
                CHECK(matches_cpu_schnorr(sc, dig, xonly, sig, out, M), "schnorr columns == CPU per-row oracle (valid)");

                std::vector<uint8_t> t = sig;
                const size_t bad[3] = {3, 200, 511};
                for (size_t k = 0; k < 3; ++k) t[bad[k] * 64 + 40] ^= 0x33;
                std::vector<uint8_t> out2(M, 0xEE);
                e = ufsecp_gpu_schnorr_verify_lbtc_columns(ctx, dig.data(), xonly.data(), t.data(), M, out2.data());
                if (e == UFSECP_OK) {
                    bool ok = true;
                    for (size_t k = 0; k < 3; ++k) if (out2[bad[k]] != 0) ok = false;
                    CHECK(ok, "schnorr columns: tampered rows zeroed (==0)");
                } else if (is_skip_err(e)) { std::printf("  (schnorr tamper: runtime skip)\n"); }

                portable_setenv("UFSECP_GPU_COLUMNS_CHUNK", "100");
                std::vector<uint8_t> out3(M, 0xEE);
                e = ufsecp_gpu_schnorr_verify_lbtc_columns(ctx, dig.data(), xonly.data(), sig.data(), M, out3.data());
                portable_unsetenv("UFSECP_GPU_COLUMNS_CHUNK");
                if (e == UFSECP_OK)
                    CHECK(matches_cpu_schnorr(sc, dig, xonly, sig, out3, M),
                          "schnorr columns: forced small chunks (100) == CPU oracle across boundaries");
                else if (is_skip_err(e)) std::printf("  (schnorr small-chunk: runtime skip)\n");

                // Malformed Schnorr inputs must yield the SAME verdict on GPU and CPU
                // (fail-closed parity): an x-only pubkey with x >= p (lift_x must reject)
                // and a BIP-340 signature with s >= n (parse_strict must reject). The
                // s >= n row exercises the CUDA/Metal column kernels' explicit strict-s
                // reject that keeps them byte-identical to the OpenCL/CPU reference — a
                // GPU that reduced s mod n instead would false-accept s' = s + n.
                {
                    std::vector<uint8_t> mxo = xonly, msig = sig;
                    const size_t row_badx = 7;    // x-only x >= p
                    const size_t row_bads = 250;  // BIP-340 s >= n
                    for (int i = 0; i < 32; ++i) mxo[row_badx * 32 + i] = 0xFF;       // x = 0xFF..FF > p
                    for (int i = 0; i < 32; ++i) msig[row_bads * 64 + 32 + i] = 0xFF; // s = 0xFF..FF >= n
                    std::vector<uint8_t> out5(M, 0xEE);
                    e = ufsecp_gpu_schnorr_verify_lbtc_columns(ctx, dig.data(), mxo.data(), msig.data(), M, out5.data());
                    if (e == UFSECP_OK) {
                        CHECK(out5[row_badx] == 0, "schnorr columns: x-only x>=p row rejected (==0)");
                        CHECK(out5[row_bads] == 0, "schnorr columns: BIP-340 s>=n row rejected (==0)");
                        CHECK(matches_cpu_schnorr(sc, dig, mxo, msig, out5, M),
                              "schnorr columns: malformed (x>=p, s>=n) rows == CPU oracle (GPU/CPU parity)");
                    } else if (is_skip_err(e)) { std::printf("  (schnorr malformed: runtime skip)\n"); }
                }

                CHECK(ufsecp_gpu_schnorr_verify_lbtc_columns(ctx, dig.data(), xonly.data(), sig.data(), 0, out.data()) == UFSECP_OK,
                      "schnorr columns(count=0) -> OK no-op");
                CHECK(ufsecp_gpu_schnorr_verify_lbtc_columns(ctx, dig.data(), nullptr, sig.data(), M, out.data()) == UFSECP_ERR_NULL_ARG,
                      "schnorr columns(NULL xonly) -> NULL_ARG");
            } else if (is_skip_err(e)) {
                std::printf("  (schnorr columns: backend runtime err %d (%s) -- skip)\n", e, ufsecp_gpu_error_str(e));
            } else {
                CHECK(0, "schnorr columns valid batch: unexpected error");
            }
        } else {
            std::printf("  (schnorr signing infra unavailable -- skip schnorr)\n");
        }
    }

    ufsecp_ctx_destroy(sc);
    ufsecp_gpu_ctx_destroy(ctx);
}

// -- engine dispatcher: CPU-only GPU-columns-hook fallback contract -------------
// Exercises the installable secp256k1::GpuColumnsVerifyHook that the two engine
// column entrypoints (ecdsa_batch_verify_opaque_columns /
// schnorr_batch_verify_bip340_columns) consult internally. This runs entirely on
// the CPU (the hook is a test double, not a real GPU) and MUST pass everywhere,
// including GitHub runners with no GPU. It proves:
//   - no hook  -> pure CPU path, all-valid corpus verifies;
//   - decline (-1) -> engine consults hook then falls back to CPU (correct rows);
//   - handled-all-valid (1) -> engine trusts the hook's all-valid verdict;
//   - handled-some-invalid (0) -> engine honours per-row verdict, fail-closed;
//   - decline + tampered corpus -> CPU fallback yields the CORRECT per-row result
//     (an operational GPU decline is fatal-not-invalid: never an all-zero buffer);
//   - count==0 and out_results==nullptr are no-ops that never mis-verify.

static int g_hook_calls = 0;
static int g_hook_mode = 0;   // 0=decline(-1); 1=return-all-valid(1); 2=return-with-forced-invalid(0)
static size_t g_hook_forced_invalid_row = 0;

static int test_columns_hook(int kind, const uint8_t* d, const uint8_t* k,
                             const uint8_t* s, size_t n, uint8_t* out) noexcept {
    (void)kind; (void)d; (void)k; (void)s;
    ++g_hook_calls;
    if (g_hook_mode == 0) return -1;
    for (size_t i = 0; i < n; ++i) out[i] = 1;
    if (g_hook_mode == 2) {
        if (g_hook_forced_invalid_row < n) out[g_hook_forced_invalid_row] = 0;
        return 0;
    }
    return 1;
}

void test_engine_dispatcher() {
    std::printf("[gpu_lbtc_columns_diff] engine dispatcher CPU fallback (no GPU needed)\n");

    ufsecp_ctx* sc = nullptr;
    if (ufsecp_ctx_create(&sc) != UFSECP_OK || !sc) {
        std::printf("  (cpu ctx create failed -- skipping engine dispatcher)\n");
        return;
    }

    const size_t M = 64;

    // Save-and-clear any pre-installed hook so we exercise the pure CPU path first.
    secp256k1::GpuColumnsVerifyHook prev =
        secp256k1::install_gpu_columns_verify_hook(nullptr);

    // Enablement proof: the GPU-host layer (src/gpu/src/gpu_engine_hook.cpp) is
    // linked into this audit build, so its self-installing provider MUST have
    // registered a default hook before any test double ran. A null 'prev' means
    // the engine surface could never reach the GPU -- the exact wiring gap a prior
    // review flagged when enablement hung on an undefined compile-time macro. On a
    // GPU-less runner the provider simply declines (-1) and the engine falls back
    // to CPU, so this holds with or without hardware.
    CHECK(prev != nullptr,
          "engine GPU-columns provider self-installed (enablement wired, not a dead macro)");

    // ---------------- ECDSA (opaque-LE sigs) ----------------
    {
        std::vector<uint8_t> dig, pub, sopq, scmp;
        if (build_ecdsa(sc, M, dig, pub, sopq, scmp)) {
            // Case 1: no hook -> pure CPU path, all valid.
            {
                std::vector<uint8_t> out(M, 0xEE);
                bool r = secp256k1::ecdsa_batch_verify_opaque_columns(
                    dig.data(), pub.data(), sopq.data(), M, out.data());
                CHECK(r, "ecdsa dispatch: no-hook CPU path returns true (all valid)");
                bool all1 = true;
                for (size_t i = 0; i < M; ++i) if (out[i] != 1) all1 = false;
                CHECK(all1, "ecdsa dispatch: no-hook CPU path writes all out[i]==1");
            }

            // Case 2: decline hook (-1) -> engine consults hook, falls back to CPU.
            {
                secp256k1::install_gpu_columns_verify_hook(test_columns_hook);
                g_hook_mode = 0;
                g_hook_calls = 0;
                std::vector<uint8_t> out(M, 0xEE);
                bool r = secp256k1::ecdsa_batch_verify_opaque_columns(
                    dig.data(), pub.data(), sopq.data(), M, out.data());
                CHECK(r, "ecdsa dispatch: decline-hook falls back, returns true");
                bool all1 = true;
                for (size_t i = 0; i < M; ++i) if (out[i] != 1) all1 = false;
                CHECK(all1, "ecdsa dispatch: decline-hook CPU fallback writes all out[i]==1");
                CHECK(g_hook_calls >= 1, "ecdsa dispatch: decline-hook was consulted (attempted GPU)");
            }

            // Case 3: handled-all-valid hook (1) -> engine trusts the hook verdict.
            {
                secp256k1::install_gpu_columns_verify_hook(test_columns_hook);
                g_hook_mode = 1;
                g_hook_calls = 0;
                std::vector<uint8_t> out(M, 0xEE);  // pre-filled non-1 to prove hook wrote it
                bool r = secp256k1::ecdsa_batch_verify_opaque_columns(
                    dig.data(), pub.data(), sopq.data(), M, out.data());
                CHECK(r, "ecdsa dispatch: handled-all-valid hook returns true");
                bool all1 = true;
                for (size_t i = 0; i < M; ++i) if (out[i] != 1) all1 = false;
                CHECK(all1, "ecdsa dispatch: handled-all-valid hook result trusted (out[i]==1)");
                CHECK(g_hook_calls >= 1, "ecdsa dispatch: handled-all-valid hook consulted");
            }

            // Case 4: handled-some-invalid hook (0, forced row 7) -> per-row honoured.
            {
                secp256k1::install_gpu_columns_verify_hook(test_columns_hook);
                g_hook_mode = 2;
                g_hook_forced_invalid_row = 7;
                g_hook_calls = 0;
                std::vector<uint8_t> out(M, 0xEE);
                bool r = secp256k1::ecdsa_batch_verify_opaque_columns(
                    dig.data(), pub.data(), sopq.data(), M, out.data());
                CHECK(!r, "ecdsa dispatch: handled-some-invalid hook returns false (fail-closed)");
                CHECK(out[7] == 0, "ecdsa dispatch: handled-some-bad forced row zeroed (out[7]==0)");
                CHECK(out[0] == 1, "ecdsa dispatch: handled-some-invalid keeps valid rows (out[0]==1)");
                CHECK(g_hook_calls >= 1, "ecdsa dispatch: handled-some-invalid hook consulted");
            }

            // Case 5: decline + genuinely tampered corpus -> CPU fallback yields the
            // CORRECT per-row verdict (fatal-not-invalid: a decline must NOT force an
            // all-zero buffer; the true invalid row must be pinpointed by the CPU).
            {
                const size_t r_bad = 21;
                std::vector<uint8_t> t = sopq;
                t[r_bad * 64 + 3] ^= 0x5A;  // flip a byte in the opaque sig
                secp256k1::install_gpu_columns_verify_hook(test_columns_hook);
                g_hook_mode = 0;  // decline -> CPU fallback
                g_hook_calls = 0;
                std::vector<uint8_t> out(M, 0xEE);
                bool r = secp256k1::ecdsa_batch_verify_opaque_columns(
                    dig.data(), pub.data(), t.data(), M, out.data());
                CHECK(!r, "ecdsa dispatch: decline + tampered corpus returns false");
                CHECK(out[r_bad] == 0, "ecdsa dispatch: decline CPU fallback flags the true bad row");
                size_t untouched = (r_bad == 0) ? 1 : 0;
                CHECK(out[untouched] == 1,
                      "ecdsa dispatch: decline CPU fallback keeps an untouched row valid (not all-zero)");
                CHECK(g_hook_calls >= 1, "ecdsa dispatch: decline+tamper hook consulted");
            }

            // Case 5b: decline + UNPARSEABLE rows (off-curve/bad-prefix pubkey and
            // r>=n sig) in NON-first positions -> CPU fallback drives the parse-
            // failure branch (parse_ecdsa_opaque_entry returns false) and flags
            // exactly those rows invalid while untouched rows stay valid. Case 5
            // only byte-flips a still-parseable sig; this covers the distinct
            // "row does not parse" path that the malformed-input differential
            // otherwise exercises only on a real GPU.
            {
                const size_t bad_pub = 30;   // invalid SEC1 prefix -> decompress fails
                const size_t bad_sig = 47;   // r >= n -> scalar reject on verify
                std::vector<uint8_t> mp = pub, ms = sopq;
                mp[bad_pub * 33] = 0x00;                       // not 0x02/0x03/0x04
                for (int j = 0; j < 32; ++j) ms[bad_sig * 64 + j] = 0xFF;  // r column = 0xFF..FF
                secp256k1::install_gpu_columns_verify_hook(test_columns_hook);
                g_hook_mode = 0;  // decline -> CPU fallback
                g_hook_calls = 0;
                std::vector<uint8_t> out(M, 0xEE);
                bool r = secp256k1::ecdsa_batch_verify_opaque_columns(
                    dig.data(), mp.data(), ms.data(), M, out.data());
                CHECK(!r, "ecdsa dispatch: decline + unparseable rows returns false");
                CHECK(out[bad_pub] == 0,
                      "ecdsa dispatch: decline CPU fallback flags off-curve pubkey row (parse-fail)");
                CHECK(out[bad_sig] == 0,
                      "ecdsa dispatch: decline CPU fallback flags r>=n sig row");
                CHECK(out[0] == 1 && out[M - 1] == 1,
                      "ecdsa dispatch: decline CPU fallback keeps well-formed rows valid");
                CHECK(g_hook_calls >= 1, "ecdsa dispatch: decline+unparseable hook consulted");
            }

            // Case 6: count==0 and out_results==nullptr no-ops must never mis-verify.
            {
                // A hook that WOULD corrupt if consulted; then prove it is not needed.
                secp256k1::install_gpu_columns_verify_hook(test_columns_hook);
                g_hook_mode = 2;
                g_hook_forced_invalid_row = 0;

                std::vector<uint8_t> out(M, 0xEE);
                bool r0 = secp256k1::ecdsa_batch_verify_opaque_columns(
                    dig.data(), pub.data(), sopq.data(), 0, out.data());
                CHECK(r0, "ecdsa dispatch: count==0 returns true (no-op)");

                g_hook_calls = 0;
                bool rn = secp256k1::ecdsa_batch_verify_opaque_columns(
                    dig.data(), pub.data(), sopq.data(), M, nullptr);
                CHECK(rn, "ecdsa dispatch: out_results==nullptr valid batch returns true");
            }
        } else {
            std::printf("  (ecdsa signing infra unavailable -- skip ecdsa dispatcher)\n");
        }
    }

    // ---------------- Schnorr (BIP-340 sigs) — mirror decline + forced-invalid ---
    {
        std::vector<uint8_t> dig, xonly, sig;
        if (build_schnorr(sc, M, dig, xonly, sig)) {
            // No hook: pure CPU path, all valid. Clear any hook left installed by
            // the ECDSA section above so this genuinely exercises the CPU path.
            {
                secp256k1::install_gpu_columns_verify_hook(nullptr);
                std::vector<uint8_t> out(M, 0xEE);
                bool r = secp256k1::schnorr_batch_verify_bip340_columns(
                    dig.data(), xonly.data(), sig.data(), M, out.data());
                CHECK(r, "schnorr dispatch: no-hook CPU path returns true (all valid)");
                bool all1 = true;
                for (size_t i = 0; i < M; ++i) if (out[i] != 1) all1 = false;
                CHECK(all1, "schnorr dispatch: no-hook CPU path writes all out[i]==1");
            }

            // Handled-some-invalid hook (0, forced row 7) -> per-row honoured.
            {
                secp256k1::install_gpu_columns_verify_hook(test_columns_hook);
                g_hook_mode = 2;
                g_hook_forced_invalid_row = 7;
                g_hook_calls = 0;
                std::vector<uint8_t> out(M, 0xEE);
                bool r = secp256k1::schnorr_batch_verify_bip340_columns(
                    dig.data(), xonly.data(), sig.data(), M, out.data());
                CHECK(!r, "schnorr dispatch: handled-some-invalid hook returns false (fail-closed)");
                CHECK(out[7] == 0, "schnorr dispatch: handled-some-bad forced row zeroed (out[7]==0)");
                CHECK(out[0] == 1, "schnorr dispatch: handled-some-invalid keeps valid rows (out[0]==1)");
                CHECK(g_hook_calls >= 1, "schnorr dispatch: handled-some-invalid hook consulted");
            }

            // Decline + genuinely tampered corpus -> CPU fallback pinpoints bad row.
            {
                const size_t r_bad = 33;
                std::vector<uint8_t> t = sig;
                t[r_bad * 64 + 40] ^= 0x3C;  // flip a byte in the BIP-340 sig
                secp256k1::install_gpu_columns_verify_hook(test_columns_hook);
                g_hook_mode = 0;  // decline -> CPU fallback
                g_hook_calls = 0;
                std::vector<uint8_t> out(M, 0xEE);
                bool r = secp256k1::schnorr_batch_verify_bip340_columns(
                    dig.data(), xonly.data(), t.data(), M, out.data());
                CHECK(!r, "schnorr dispatch: decline + tampered corpus returns false");
                CHECK(out[r_bad] == 0, "schnorr dispatch: decline CPU fallback flags the true bad row");
                size_t untouched = (r_bad == 0) ? 1 : 0;
                CHECK(out[untouched] == 1,
                      "schnorr dispatch: decline CPU fallback keeps an untouched row valid (not all-zero)");
                CHECK(g_hook_calls >= 1, "schnorr dispatch: decline+tamper hook consulted");
            }
        } else {
            std::printf("  (schnorr signing infra unavailable -- skip schnorr dispatcher)\n");
        }
    }

    // Restore whatever hook was installed before this test ran.
    secp256k1::install_gpu_columns_verify_hook(prev);
    ufsecp_ctx_destroy(sc);
}

// -- CPU-only consensus edges (runs on EVERY runner, no GPU needed) -------------
// The consensus-critical edge cases — high-S ECDSA acceptance, pubkey x >= p
// rejection, BIP-340 s >= n rejection, BIP-340 s == 0 rejection — are also
// exercised in test_differential(), but ONLY when a real GPU is present. On a
// GPU-less runner (e.g. GitHub CI) those rows never execute, so the engine CPU
// column reference (ecdsa_batch_verify_opaque_columns /
// schnorr_batch_verify_bip340_columns) would go unverified on the behavior the
// GPU kernels are held to. This drives those edges directly on the CPU with the
// GPU hook cleared, so they run everywhere.
void test_cpu_consensus_edges() {
    std::printf("[gpu_lbtc_columns_diff] CPU column consensus edges (no GPU needed)\n");
    ufsecp_ctx* sc = nullptr;
    if (ufsecp_ctx_create(&sc) != UFSECP_OK || !sc) {
        std::printf("  (cpu ctx create failed -- skipping consensus rejects)\n");
        return;
    }
    const size_t M = 64;

    // Force the CPU column path regardless of any linked GPU provider.
    secp256k1::GpuColumnsVerifyHook prev =
        secp256k1::install_gpu_columns_verify_hook(nullptr);

    // ---------------- ECDSA: high-S accepted + pubkey x>=p rejected --------
    {
        std::vector<uint8_t> dig, pub, sopq, scmp;
        if (build_ecdsa(sc, M, dig, pub, sopq, scmp)) {
            const size_t hs  = 20;   // high-S row
            const size_t xge = 40;   // x>=p pubkey row
            std::vector<uint8_t> hp = pub, hopq = sopq;

            // high-S: S' = n - S via seckey_negate on the compact S, re-encode opaque.
            // A low-S S is nonzero and < n, so its negation is a valid high-S scalar.
            bool hs_ready = false;
            uint8_t s_hi[32];
            std::memcpy(s_hi, &scmp[hs * 64 + 32], 32);
            if (ufsecp_seckey_negate(sc, s_hi) == UFSECP_OK) {
                uint8_t cmp_hi[64], opq_hi[64];
                std::memcpy(cmp_hi, &scmp[hs * 64], 32);   // R unchanged
                std::memcpy(cmp_hi + 32, s_hi, 32);        // S' = n - S
                if (ufsecp_ecdsa_sig_compact_to_opaque(sc, cmp_hi, opq_hi) == UFSECP_OK) {
                    std::memcpy(&hopq[hs * 64], opq_hi, 64);
                    hs_ready = true;
                }
            }
            // x>=p pubkey: 0x02 || 0xFF..FF (> p = FFFF..FFFFFC2F), strict decompress rejects.
            hp[xge * 33] = 0x02;
            for (int j = 1; j < 33; ++j) hp[xge * 33 + j] = 0xFF;

            std::vector<uint8_t> out(M, 0xEE);
            bool r = secp256k1::ecdsa_batch_verify_opaque_columns(
                dig.data(), hp.data(), hopq.data(), M, out.data());
            CHECK(!r, "ecdsa CPU columns: batch with high-S/x>=p returns false because x>=p is invalid");
            if (hs_ready)
                CHECK(out[hs] == 1, "ecdsa CPU columns: high-S (s>n/2) accepted");
            else
                std::printf("  (ecdsa high-S: ABI declined scalar negate -- skip hs assert)\n");
            CHECK(out[xge] == 0, "ecdsa CPU columns: pubkey x>=p rejected");
            CHECK(out[0] == 1, "ecdsa CPU columns: control row valid alongside x>=p reject (not all-zero)");
        } else {
            std::printf("  (ecdsa signing infra unavailable -- skip ecdsa consensus rejects)\n");
        }
    }

    // ---------------- Schnorr: x-only x>=p + BIP-340 s>=n + s==0 ----------------
    {
        std::vector<uint8_t> dig, xonly, sig;
        if (build_schnorr(sc, M, dig, xonly, sig)) {
            std::vector<uint8_t> mxo = xonly, msig = sig;
            const size_t row_badx = 7;    // x-only x >= p  (lift_x must reject)
            const size_t row_bads = 25;   // BIP-340 s >= n (strict-s must reject)
            const size_t row_s0   = 44;   // BIP-340 s == 0 (0 < s < n required)
            for (int i = 0; i < 32; ++i) mxo[row_badx * 32 + i] = 0xFF;         // x = 0xFF..FF > p
            for (int i = 0; i < 32; ++i) msig[row_bads * 64 + 32 + i] = 0xFF;   // s = 0xFF..FF >= n
            for (int i = 0; i < 32; ++i) msig[row_s0  * 64 + 32 + i] = 0x00;    // s = 0

            std::vector<uint8_t> out(M, 0xEE);
            bool r = secp256k1::schnorr_batch_verify_bip340_columns(
                dig.data(), mxo.data(), msig.data(), M, out.data());
            CHECK(!r, "schnorr CPU columns: batch with x>=p/s>=n/s==0 returns false");
            CHECK(out[row_badx] == 0, "schnorr CPU columns: x-only x>=p rejected");
            CHECK(out[row_bads] == 0, "schnorr CPU columns: BIP-340 s>=n rejected");
            CHECK(out[row_s0]  == 0, "schnorr CPU columns: BIP-340 s==0 rejected");
            CHECK(out[0] == 1, "schnorr CPU columns: control row valid alongside rejects (not all-zero)");
        } else {
            std::printf("  (schnorr signing infra unavailable -- skip schnorr consensus rejects)\n");
        }
    }

    secp256k1::install_gpu_columns_verify_hook(prev);
    ufsecp_ctx_destroy(sc);
}

}  // namespace

int test_gpu_lbtc_columns_diff_run() {
    g_pass = 0; g_fail = 0;
    std::printf("=== GPU libbitcoin column verify differential ===\n");
    test_null_ctx_contract();
    test_engine_dispatcher();
    test_cpu_consensus_edges();
    test_differential();
    std::printf("[gpu_lbtc_columns_diff] pass=%d fail=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() { return test_gpu_lbtc_columns_diff_run(); }
#endif
