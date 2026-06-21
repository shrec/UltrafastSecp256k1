/**
 * ufsecp_libbitcoin.cpp — implementation of the libbitcoin acceleration bridge.
 *
 * The controller is dispatch + marshalling only. It owns a CPU context (always)
 * and, when built with GPU support and a device is available, a GPU context.
 * It reuses the engine's existing, tested public primitives:
 *
 *   CPU:  ufsecp_ecdsa_verify_opaque_rows / ufsecp_schnorr_batch_verify
 *           (libbitcoin ECDSA rows are passed directly to the engine's opaque
 *            row C ABI; no intermediate compact row table is built)
 *         ufsecp_ecdsa_verify_opaque_batch / ufsecp_schnorr_verify
 *           (columnar fallback: no row re-pack)
 *   GPU:  ufsecp_gpu_ecdsa_verify_opaque_rows / ufsecp_gpu_schnorr_verify_batch
 *           (ECDSA reads the caller's libbitcoin rows directly; per-item
 *            results — one signature per thread)
 *   SP :  ufsecp_gpu_bip352_scan_batch
 *
 * GPU dispatch is compile-time optional: define UFSECP_LBTC_WITH_GPU to link the
 * engine GPU ABI. Without it the bridge is a CPU-only build (the consensus
 * reference path) that links a CPU-only engine library. The C ABI is identical
 * in both configurations.
 *
 * Per-row semantics: every row gets a 1/0 result. A structurally-malformed
 * record (off-curve pubkey, s>=n, R.x>=p) is reported as invalid (0); it never
 * aborts the batch. This is obtained on CPU by interpreting the return code of a
 * single-entry verify, and on GPU by the kernel's per-item out_results.
 *
 * Mandatory CPU fallback: if a GPU chunk fails at the device level, that chunk
 * is transparently re-run on the CPU. The CPU path is the consensus reference.
 */
#include "ufsecp_libbitcoin.h"

#include "ufsecp.h"     /* ufsecp_ctx, ufsecp_ctx_create/destroy,
                           ufsecp_ecdsa_batch_verify, ufsecp_schnorr_batch_verify */
#ifdef UFSECP_LBTC_WITH_GPU
#include "ufsecp_gpu.h"          /* GPU C ABI + UFSECP_GPU_BACKEND_*, UFSECP_ERR_GPU_* */
#include "secp256k1/scalar.hpp"  /* Scalar — Fiat-Shamir g_coeff for the commitment RLC */
#endif

#include <cstring>
#include <new>
#include <type_traits>
#include <vector>
#include <thread>
#include <algorithm>

/* UFSECP_ERR_GPU_UNAVAILABLE lives in ufsecp_gpu.h; provide it for CPU-only builds. */
#ifndef UFSECP_ERR_GPU_UNAVAILABLE
#define UFSECP_ERR_GPU_UNAVAILABLE 100
#endif

/* ------------------------------------------------------------------------- */
/* Controller state                                                           */
/* ------------------------------------------------------------------------- */

struct ufsecp_lbtc_ctrl {
    ufsecp_ctx*       cpu  = nullptr;                 /* always present        */
#ifdef UFSECP_LBTC_WITH_GPU
    ufsecp_gpu_ctx*   gpu  = nullptr;                 /* null when CPU-bound   */
#endif
    ufsecp_lbtc_bound bound = UFSECP_LBTC_BOUND_CPU;
    char              device_name[128] = {'C', 'P', 'U', '\0'};
};

namespace {

/* Per-call chunk size. Well below kMaxGpuBatchN (64 M) and any CPU limit, large
 * enough to amortize device transfer for IBD-scale batches. Tunable.
 *
 * The default is fixed at 262144. UFSECP_LBTC_CHUNK_OVERRIDE is a TEST-ONLY
 * compile seam (e.g. -DUFSECP_LBTC_CHUNK_OVERRIDE=8) that lets the multi-chunk
 * boundary be exercised deterministically without a 262k-row corpus. Production
 * builds never define it. */
#ifndef UFSECP_LBTC_CHUNK_OVERRIDE
constexpr std::size_t kChunk = std::size_t{1} << 18; /* 262144 */
#else
constexpr std::size_t kChunk = UFSECP_LBTC_CHUNK_OVERRIDE;
#endif

enum class Kind { Ecdsa, Schnorr };
enum class EcdsaSigFormat { Compact, Opaque };

inline std::size_t record_size(Kind k) {
    return k == Kind::Ecdsa ? UFSECP_LBTC_ECDSA_RECORD : UFSECP_LBTC_SCHNORR_RECORD;
}

inline std::size_t pubkey_size(Kind k) {
    return k == Kind::Ecdsa ? 33u : 32u;
}

inline bool cancel_requested(const ufsecp_cancel_token* cancel) noexcept {
    if (!cancel || !cancel->is_cancelled) return false;
    try {
        return cancel->is_cancelled(cancel->user) != 0;
    } catch (...) {
        return true;
    }
}

inline std::size_t cancel_chunk_size(const ufsecp_cancel_token* cancel) noexcept {
    if (!cancel || cancel->check_interval == 0) return kChunk;
    const std::size_t requested = static_cast<std::size_t>(cancel->check_interval);
    return requested < kChunk ? requested : kChunk;
}

/* secp256k1 group order and half-order, big-endian. These are public signature
 * normalization constants used only at the libbitcoin bridge boundary. */
static constexpr uint8_t kScalarOrder[32] = {
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfe,
    0xba, 0xae, 0xdc, 0xe6, 0xaf, 0x48, 0xa0, 0x3b,
    0xbf, 0xd2, 0x5e, 0x8c, 0xd0, 0x36, 0x41, 0x41
};

static constexpr uint8_t kScalarHalfOrder[32] = {
    0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0x5d, 0x57, 0x6e, 0x73, 0x57, 0xa4, 0x50, 0x1d,
    0xdf, 0xe9, 0x2f, 0x46, 0x68, 0x1b, 0x20, 0xa0
};

inline int cmp32_be(const uint8_t* a, const uint8_t* b) noexcept {
    for (std::size_t i = 0; i < 32; ++i) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

inline bool is_zero32(const uint8_t* v) noexcept {
    uint8_t acc = 0;
    for (std::size_t i = 0; i < 32; ++i) acc |= v[i];
    return acc == 0;
}

inline void scalar_order_minus(uint8_t out[32], const uint8_t x[32]) noexcept {
    uint16_t borrow = 0;
    for (int i = 31; i >= 0; --i) {
        const uint16_t lhs = kScalarOrder[static_cast<std::size_t>(i)];
        const uint16_t rhs = static_cast<uint16_t>(x[i]) + borrow;
        if (lhs < rhs) {
            out[i] = static_cast<uint8_t>(lhs + 256u - rhs);
            borrow = 1;
        } else {
            out[i] = static_cast<uint8_t>(lhs - rhs);
            borrow = 0;
        }
    }
}

/* libbitcoin's ec_signature is a copied secp256k1_ecdsa_signature object. In
 * libsecp-compatible memory that is opaque internal scalar storage, not public
 * compact r||s. The shim stores each 32-byte scalar byte-reversed relative to
 * compact big-endian form, matching libsecp on little-endian hosts. */
inline void scalar_internal_to_be(uint8_t out[32], const uint8_t in[32]) noexcept {
    for (std::size_t i = 0; i < 32; ++i)
        out[i] = in[31u - i];
}

inline void ecdsa_opaque_sig_to_compact(const uint8_t* opaque,
                                        uint8_t compact[64]) noexcept {
    scalar_internal_to_be(compact, opaque);
    scalar_internal_to_be(compact + 32, opaque + 32);
}

inline bool ecdsa_s_is_high(const uint8_t* sig64) noexcept {
    const uint8_t* s = sig64 + 32;
    return !is_zero32(s) &&
           cmp32_be(s, kScalarOrder) < 0 &&
           cmp32_be(s, kScalarHalfOrder) > 0;
}

inline void normalize_ecdsa_sig_low_s(uint8_t sig64[64]) noexcept {
    if (!ecdsa_s_is_high(sig64)) return;
    uint8_t low_s[32];
    scalar_order_minus(low_s, sig64 + 32);
    std::memcpy(sig64 + 32, low_s, 32);
}

inline void copy_ecdsa_signature_normalized(const uint8_t* opaque,
                                            uint8_t compact[64],
                                            EcdsaSigFormat format) noexcept {
    if (format == EcdsaSigFormat::Opaque)
        ecdsa_opaque_sig_to_compact(opaque, compact);
    else
        std::memcpy(compact, opaque, 64);
    normalize_ecdsa_sig_low_s(compact);
}

inline ufsecp_error_t cpu_verify_one(ufsecp_ctx* ctx, Kind k, const uint8_t* rec) {
    /* For n == 1 the packed verify reads exactly record_size() bytes, so the
     * row's opaque key tail (if any) is never touched. */
    if (k == Kind::Ecdsa) {
        uint8_t result = 0;
        const auto rc = ufsecp_ecdsa_verify_opaque_rows(
            ctx, rec, UFSECP_LBTC_ECDSA_RECORD, 1, &result);
        return (rc == UFSECP_OK && result != 0) ? UFSECP_OK
                                                : UFSECP_ERR_VERIFY_FAIL;
    }
    return ufsecp_schnorr_batch_verify(ctx, rec, 1);
}

inline ufsecp_error_t cpu_verify_run(ufsecp_ctx* ctx, Kind k,
                                     const uint8_t* recs, std::size_t cnt) {
    return k == Kind::Ecdsa ? ufsecp_ecdsa_batch_verify(ctx, recs, cnt)
                            : ufsecp_schnorr_batch_verify(ctx, recs, cnt);
}

/* Output channel for one verified row. The chunk paths (cpu_chunk / gpu_chunk)
 * compute the per-row valid/invalid verdict and ONLY call sink.mark(...) /
 * sink.mark_all_valid(...). Two concrete sinks plug into the SAME chunk bodies,
 * so the verify cores are reused verbatim — only where the verdict is written
 * differs (consensus-parity by construction):
 *
 *   ResultSink  — writes results[i] = 1/0      (the original results API)
 *   CollectSink — collapses the verdict INTO the row's trailing key cell:
 *                   valid   -> memset(key cell, 0, key_size)   (key zeroed)
 *                   invalid -> leave the key bytes untouched   (id survives)
 *                 The caller then scans for non-zero key cells = rejected ids.
 */
struct ResultSink {
    uint8_t* results;                 /* may be NULL (no-op) */

    inline void mark(std::size_t global, bool valid) {
        if (results) results[global] = valid ? 1u : 0u;
    }
    inline void mark_all_valid(std::size_t base, std::size_t cnt) {
        if (results) std::memset(results + base, 1, cnt);
    }
};

/* In-place sink. Owns the MUTABLE rows buffer + geometry so it can locate the
 * key cell of any global row index. The key cell of global row i lives at
 * rows + i*stride + rec, length key_size. VALID -> zero it; INVALID -> leave.
 *
 * No aliasing hazard with the const `rows` view read by cpu_chunk/gpu_chunk:
 * every write happens through mark(...) only AFTER that chunk's reads complete,
 * and it touches only the key tail [rec, rec+key_size) — never any record byte
 * the verify core reads. key_size > 0 is guaranteed by the entry point. */
struct CollectSink {
    uint8_t*    rows;       /* mutable caller buffer (separate from the const view) */
    std::size_t stride;     /* rec + key_size */
    std::size_t rec;        /* record_size(k) — offset of the key cell */
    std::size_t key_size;   /* > 0 */

    inline void mark(std::size_t global, bool valid) {
        if (valid)
            std::memset(rows + global * stride + rec, 0, key_size);
        /* invalid: leave the key bytes untouched (id survives = rejected) */
    }
    inline void mark_all_valid(std::size_t base, std::size_t cnt) {
        for (std::size_t i = 0; i < cnt; ++i)
            std::memset(rows + (base + i) * stride + rec, 0, key_size);
    }
};

/* In-place sink for the vertical/columnar API. The key column is separate from
 * the verified msg/pub/sig columns, so valid rows zero only
 * key_cells[i*key_size .. i*key_size+key_size). */
struct KeyColumnSink {
    uint8_t*    key_cells;  /* mutable n * key_size correlation-key column */
    std::size_t key_size;   /* > 0 */

    inline void mark(std::size_t global, bool valid) {
        if (valid)
            std::memset(key_cells + global * key_size, 0, key_size);
    }
    inline void mark_all_valid(std::size_t base, std::size_t cnt) {
        for (std::size_t i = 0; i < cnt; ++i)
            std::memset(key_cells + (base + i) * key_size, 0, key_size);
    }
};

struct LbtcByteScratch {
    std::vector<uint8_t> a;
    std::vector<uint8_t> b;
    std::vector<uint8_t> c;
    std::vector<uint8_t> d;

    uint8_t* bytes(std::vector<uint8_t>& slot, std::size_t n) {
        if (slot.size() < n) slot.resize(n);
        return slot.data();
    }

    uint8_t* fill(std::vector<uint8_t>& slot, std::size_t n, uint8_t value) {
        uint8_t* out = bytes(slot, n);
        std::fill_n(out, n, value);
        return out;
    }
};

static thread_local LbtcByteScratch g_lbtc_byte_scratch;

/* Translate one libbitcoin row into the engine's packed record layout.
 *   ECDSA   row == engine record:        32 msg | 33 pubkey | 64 sig.
 *   Schnorr row (uniform with ECDSA):    32 msg | 32 xonly  | 64 sig,
 *           but the engine record is     32 xonly | 32 msg  | 64 sig — so the
 *           first two fields are swapped on the way in.
 * `out` must hold record_size(k) bytes. */
inline void to_engine_record(Kind k, const uint8_t* row, uint8_t* out,
                             EcdsaSigFormat format) {
    if (k == Kind::Ecdsa) {
        std::memcpy(out, row, 65);
        copy_ecdsa_signature_normalized(row + 65, out + 65, format);
    } else {
        std::memcpy(out,      row + 32, 32);  /* xonly <- libbitcoin offset 32 */
        std::memcpy(out + 32, row,      32);  /* msg   <- libbitcoin offset 0  */
        std::memcpy(out + 64, row + 64, 64);  /* sig                           */
    }
}

/* CPU path for one chunk [base, base+cnt). Never aborts: malformed → invalid.
 * Templated on the sink so ResultSink and CollectSink share the identical body
 * (zero virtual-dispatch cost). */
template <class Sink>
void cpu_chunk(ufsecp_ctx* ctx, Kind k, const uint8_t* rows,
               std::size_t base, std::size_t cnt, std::size_t stride,
               EcdsaSigFormat format, Sink& sink) {
    const std::size_t rec = record_size(k);

    /* ECDSA OPAQUE rows: libbitcoin's native ec_signature layout. The engine's
     * opaque-row C ABI parses each row in place into the batch verifier and low-S
     * normalizes internally — fast, zero scratch. */
    if (k == Kind::Ecdsa && format == EcdsaSigFormat::Opaque) {
        uint8_t* verdicts = g_lbtc_byte_scratch.bytes(g_lbtc_byte_scratch.a, cnt);
        const auto rc = ufsecp_ecdsa_verify_opaque_rows(
            ctx, rows + base * stride, stride, cnt, verdicts);
        for (std::size_t i = 0; i < cnt; ++i) {
            sink.mark(base + i, rc == UFSECP_OK && verdicts[i] != 0);
        }
        return;
    }

    /* ECDSA COMPACT rows (public big-endian r||s in the row's sig field): the
     * opaque-row fast path would mis-read them, so verify per row — extract
     * msg(32)|pubkey(33)|sig(64), low-S normalize the compact sig, and verify. */
    if (k == Kind::Ecdsa) {
        for (std::size_t i = 0; i < cnt; ++i) {
            const uint8_t* r = rows + (base + i) * stride;
            uint8_t sig_norm[64];
            copy_ecdsa_signature_normalized(r + 65, sig_norm, EcdsaSigFormat::Compact);
            sink.mark(base + i,
                      ufsecp_ecdsa_verify(ctx, r, sig_norm, r + 32) == UFSECP_OK);
        }
        return;
    }

    /* Schnorr: the libbitcoin field order differs from the engine record, so
     * reorder the chunk into an engine-format contiguous scratch (stride == rec)
     * and reuse the same fast/per-row engine verify. */
    std::vector<uint8_t> eng(cnt * rec);
    for (std::size_t i = 0; i < cnt; ++i)
        to_engine_record(k, rows + (base + i) * stride, eng.data() + i * rec,
                         format);

    if (cpu_verify_run(ctx, k, eng.data(), cnt) == UFSECP_OK) {
        sink.mark_all_valid(base, cnt);
        return;
    }
    for (std::size_t i = 0; i < cnt; ++i)
        sink.mark(base + i, cpu_verify_one(ctx, k, eng.data() + i * rec) == UFSECP_OK);
}

inline ufsecp_error_t cpu_verify_columns_one(ufsecp_ctx* ctx, Kind k,
                                             const uint8_t* msg,
                                             const uint8_t* pub,
                                             const uint8_t* sig,
                                             EcdsaSigFormat format) {
    if (k == Kind::Ecdsa) {
        uint8_t sig_norm[64];
        copy_ecdsa_signature_normalized(sig, sig_norm, format);
        return ufsecp_ecdsa_verify(ctx, msg, sig_norm, pub);
    }
    return ufsecp_schnorr_verify(ctx, msg, sig, pub);
}

/* CPU fallback for vertical/columnar inputs. This deliberately avoids rebuilding
 * packed records; each row is verified directly from its three public columns. */
template <class Sink>
void cpu_columns_chunk(ufsecp_ctx* ctx, Kind k,
                       const uint8_t* msgs32, const uint8_t* pubs,
                       const uint8_t* sigs64, std::size_t base,
                       std::size_t cnt, EcdsaSigFormat format, Sink& sink) {
    const std::size_t pub = pubkey_size(k);
    /* Fast batch path for OPAQUE ECDSA columns (libbitcoin's native ec_signature
     * layout): the engine's opaque-batch C ABI parses + low-S normalizes directly.
     * A COMPACT column (public big-endian r||s, e.g. ufsecp_lbtc_ecdsa_sigs_pack
     * output) must NOT take this path — it would be mis-read as opaque — so it
     * falls through to the per-row loop, which honours `format`. */
    if (k == Kind::Ecdsa && format == EcdsaSigFormat::Opaque) {
        uint8_t* verdicts = g_lbtc_byte_scratch.bytes(g_lbtc_byte_scratch.a, cnt);
        const auto rc = ufsecp_ecdsa_verify_opaque_batch(
            ctx,
            msgs32 + base * 32,
            pubs + base * pub,
            sigs64 + base * 64,
            cnt,
            verdicts);
        for (std::size_t i = 0; i < cnt; ++i) {
            sink.mark(base + i, rc == UFSECP_OK && verdicts[i] != 0);
        }
        return;
    }

    for (std::size_t i = 0; i < cnt; ++i) {
        const std::size_t idx = base + i;
        sink.mark(idx,
                  cpu_verify_columns_one(ctx, k,
                                         msgs32 + idx * 32,
                                         pubs + idx * pub,
                                         sigs64 + idx * 64,
                                         format) == UFSECP_OK);
    }
}

#ifdef UFSECP_LBTC_WITH_GPU
/* GPU path for one chunk. Returns false on device-level failure (caller then
 * falls back to CPU for this chunk). Per-row validity comes from out_results.
 *
 * This is the GENERIC host-collapse path: the GPU computes the verdict, the host
 * writes it through the sink. With CollectSink the host zeros valid key cells.
 * It is correct and parity-safe for both sinks, and is the mandatory fallback
 * for the dedicated on-device collect kernel. Templated on the sink type. */
template <class Sink>
bool gpu_chunk(ufsecp_gpu_ctx* gpu, Kind k, const uint8_t* rows,
               std::size_t base, std::size_t cnt, std::size_t stride,
               EcdsaSigFormat format, Sink& sink) {
    if (k == Kind::Ecdsa && format == EcdsaSigFormat::Opaque) {
        uint8_t* res = g_lbtc_byte_scratch.bytes(g_lbtc_byte_scratch.a, cnt);
        const ufsecp_error_t rc =
            ufsecp_gpu_ecdsa_verify_opaque_rows(
                gpu, rows + base * stride, stride, cnt, res);
        if (rc != UFSECP_OK) return false;
        for (std::size_t i = 0; i < cnt; ++i)
            sink.mark(base + i, res[i] != 0);
        return true;
    }

    uint8_t* msg = g_lbtc_byte_scratch.bytes(g_lbtc_byte_scratch.a, cnt * 32);
    uint8_t* sig = g_lbtc_byte_scratch.bytes(g_lbtc_byte_scratch.b, cnt * 64);
    uint8_t* res = g_lbtc_byte_scratch.bytes(g_lbtc_byte_scratch.c, cnt);
    uint8_t* pub = g_lbtc_byte_scratch.bytes(
        g_lbtc_byte_scratch.d, cnt * (k == Kind::Ecdsa ? 33u : 32u));

    for (std::size_t i = 0; i < cnt; ++i) {
        const uint8_t* r = rows + (base + i) * stride;
        if (k == Kind::Ecdsa) {
            /* record: 32 msg | 33 pubkey | 64 sig */
            std::memcpy(msg + i * 32, r, 32);
            std::memcpy(pub + i * 33, r + 32, 33);
            copy_ecdsa_signature_normalized(r + 65, sig + i * 64, format);
        } else {
            /* libbitcoin row: 32 msg | 32 xonly | 64 sig. The engine GPU ABI
             * takes (msg, pubkey_x, sig) — extract at the libbitcoin offsets. */
            std::memcpy(msg + i * 32, r, 32);
            std::memcpy(pub + i * 32, r + 32, 32);
            std::memcpy(sig + i * 64, r + 64, 64);
        }
    }

    const ufsecp_error_t rc =
        k == Kind::Ecdsa
            ? ufsecp_gpu_ecdsa_verify_batch(gpu, msg, pub, sig, cnt, res)
            : ufsecp_gpu_schnorr_verify_batch(gpu, msg, pub, sig, cnt, res);
    if (rc != UFSECP_OK) return false; /* fall back to CPU */

    for (std::size_t i = 0; i < cnt; ++i)
        sink.mark(base + i, res[i] != 0);
    return true;
}

/* DEDICATED on-device collect path (libbitcoin specialization). De-interleaves
 * exactly like gpu_chunk, but calls the dedicated *_verify_collect kernel which
 * writes a 1-byte/row verdict on device (the kernel zeroes the verdict byte on
 * VALID and leaves the caller-seeded non-zero byte on INVALID). The host then
 * applies the SAME sink (so the variable-width real key-cell zeroing — or the
 * results[] write — stays host-side and byte-identical to every other path).
 * Returns false if the backend does not implement collect (OpenCL/Metal return
 * Unsupported) or on device failure → caller falls back to gpu_chunk then CPU.
 *
 * Verdict convention here is the INVERSE of gpu_chunk's: the collect kernel sets
 * keys[i]==0 for VALID (and leaves it non-zero for invalid), whereas verify_batch
 * returns res[i]==1 for valid. So "valid" is `keys[i] == 0`. */
template <class Sink>
bool gpu_chunk_collect(ufsecp_gpu_ctx* gpu, Kind k, const uint8_t* rows,
                       std::size_t base, std::size_t cnt, std::size_t stride,
                       EcdsaSigFormat format, Sink& sink) {
    if (k == Kind::Ecdsa && format == EcdsaSigFormat::Opaque)
        return gpu_chunk(gpu, k, rows, base, cnt, stride, format, sink);

    uint8_t* msg = g_lbtc_byte_scratch.bytes(g_lbtc_byte_scratch.a, cnt * 32);
    uint8_t* sig = g_lbtc_byte_scratch.bytes(g_lbtc_byte_scratch.b, cnt * 64);
    uint8_t* pub = g_lbtc_byte_scratch.bytes(
        g_lbtc_byte_scratch.c, cnt * (k == Kind::Ecdsa ? 33u : 32u));
    uint8_t* keys = g_lbtc_byte_scratch.fill(
        g_lbtc_byte_scratch.d, cnt, 1u); /* seed non-zero → unwritten rows = rejected */

    for (std::size_t i = 0; i < cnt; ++i) {
        const uint8_t* r = rows + (base + i) * stride;
        if (k == Kind::Ecdsa) {
            std::memcpy(msg + i * 32, r, 32);
            std::memcpy(pub + i * 33, r + 32, 33);
            copy_ecdsa_signature_normalized(r + 65, sig + i * 64, format);
        } else {
            std::memcpy(msg + i * 32, r, 32);       /* msg   @ 0  */
            std::memcpy(pub + i * 32, r + 32, 32);  /* xonly @ 32 */
            std::memcpy(sig + i * 64, r + 64, 64);  /* sig   @ 64 */
        }
    }

    const ufsecp_error_t rc =
        k == Kind::Ecdsa
            ? ufsecp_gpu_ecdsa_verify_collect(gpu, msg, pub, sig, cnt, keys)
            : ufsecp_gpu_schnorr_verify_collect(gpu, msg, pub, sig, cnt, keys);
    if (rc != UFSECP_OK) return false; /* Unsupported / device error → fall back */

    for (std::size_t i = 0; i < cnt; ++i)
        sink.mark(base + i, keys[i] == 0u); /* keys[i]==0 ⇔ valid (see above) */
    return true;
}

bool gpu_columns_results(ufsecp_gpu_ctx* gpu, Kind k,
                         const uint8_t* msgs32, const uint8_t* pubs,
                         const uint8_t* sigs64, std::size_t base,
                         std::size_t cnt, EcdsaSigFormat format,
                         uint8_t* results) {
    const std::size_t pub = pubkey_size(k);
    uint8_t* out = results ? results + base : nullptr;
    if (!out)
        out = g_lbtc_byte_scratch.bytes(g_lbtc_byte_scratch.a, cnt);

    const uint8_t* sig_ptr = sigs64 + base * 64;
    if (k == Kind::Ecdsa) {
        uint8_t* sig_norm = g_lbtc_byte_scratch.bytes(g_lbtc_byte_scratch.b, cnt * 64);
        for (std::size_t i = 0; i < cnt; ++i)
            copy_ecdsa_signature_normalized(sig_ptr + i * 64,
                                            sig_norm + i * 64,
                                            format);
        sig_ptr = sig_norm;
    }

    const ufsecp_error_t rc =
        k == Kind::Ecdsa
            ? ufsecp_gpu_ecdsa_verify_batch(gpu, msgs32 + base * 32,
                                            pubs + base * pub,
                                            sig_ptr, cnt, out)
            : ufsecp_gpu_schnorr_verify_batch(gpu, msgs32 + base * 32,
                                              pubs + base * pub,
                                              sig_ptr, cnt, out);
    return rc == UFSECP_OK;
}

bool gpu_columns_collect(ufsecp_gpu_ctx* gpu, Kind k,
                         const uint8_t* msgs32, const uint8_t* pubs,
                         const uint8_t* sigs64, std::size_t base,
                         std::size_t cnt, EcdsaSigFormat format,
                         KeyColumnSink& sink) {
    const std::size_t pub = pubkey_size(k);
    /* Use a 1-byte temporary verdict column even when the caller's key column is
     * also 1 byte. That keeps the caller-owned key cells untouched until the
     * backend reports success, so a device failure can safely fall back to CPU. */
    uint8_t* markers = g_lbtc_byte_scratch.fill(g_lbtc_byte_scratch.a, cnt, 1u);
    const uint8_t* sig_ptr = sigs64 + base * 64;
    if (k == Kind::Ecdsa) {
        uint8_t* sig_norm = g_lbtc_byte_scratch.bytes(g_lbtc_byte_scratch.b, cnt * 64);
        for (std::size_t i = 0; i < cnt; ++i)
            copy_ecdsa_signature_normalized(sig_ptr + i * 64,
                                            sig_norm + i * 64,
                                            format);
        sig_ptr = sig_norm;
    }
    const ufsecp_error_t rc =
        k == Kind::Ecdsa
            ? ufsecp_gpu_ecdsa_verify_collect(gpu, msgs32 + base * 32,
                                              pubs + base * pub,
                                              sig_ptr, cnt, markers)
            : ufsecp_gpu_schnorr_verify_collect(gpu, msgs32 + base * 32,
                                                pubs + base * pub,
                                                sig_ptr, cnt, markers);
    if (rc != UFSECP_OK) return false;

    for (std::size_t i = 0; i < cnt; ++i)
        sink.mark(base + i, markers[i] == 0u);
    return true;
}
#endif /* UFSECP_LBTC_WITH_GPU */

/* The shared chunk loop, generic over the output sink. `rows` is the const view
 * the verify cores read; a CollectSink carries its own mutable pointer to the
 * same buffer for the in-place write.
 *
 * The chunk paths allocate scratch / marshalling buffers (the GPU per-field
 * arrays and the Schnorr CPU reorder). A mid-batch allocation failure is
 * unrecoverable — never let an exception escape the extern "C" boundary. On such
 * a failure the loop is abandoned: rows already processed keep their verdict and
 * rows not yet reached keep the caller's initial state, i.e. fail-closed (results
 * stay zero-init = invalid; collect key cells stay non-zero = rejected), never
 * falsely accepted. */
template <class Sink>
ufsecp_error_t verify_core(ufsecp_lbtc_ctrl* ctrl, Kind k, const uint8_t* rows,
                           std::size_t n, std::size_t stride,
                           EcdsaSigFormat format, Sink& sink,
                           const ufsecp_cancel_token* cancel) {
    try {
        const std::size_t chunk = cancel_chunk_size(cancel);
        for (std::size_t base = 0; base < n; base += chunk) {
            if (cancel_requested(cancel)) return UFSECP_ERR_CANCELLED;
            const std::size_t cnt = (n - base) < chunk ? (n - base) : chunk;
#ifdef UFSECP_LBTC_WITH_GPU
            if (ctrl->gpu) {
                /* Collect path PREFERS the dedicated on-device kernel; the results
                 * path (ResultSink) is left byte-for-byte unchanged — it never
                 * touches the new kernel. Both fall back to the host-collapse
                 * gpu_chunk, then to CPU. UFSECP_LBTC_DISABLE_DEDICATED is a
                 * TEST/BENCH seam that forces the host-collapse control arm. */
                if constexpr (std::is_same_v<Sink, CollectSink>) {
#ifndef UFSECP_LBTC_DISABLE_DEDICATED
                    if (gpu_chunk_collect(ctrl->gpu, k, rows, base, cnt, stride,
                                          format, sink))
                        continue;
#endif
                }
                if (gpu_chunk(ctrl->gpu, k, rows, base, cnt, stride, format, sink))
                    continue;
                /* device-level failure → mandatory CPU fallback for this chunk */
            }
#endif
            cpu_chunk(ctrl->cpu, k, rows, base, cnt, stride, format, sink);
        }
    } catch (...) {
        return UFSECP_OK; /* fail-closed: unprocessed rows remain at caller initial state */
    }
    return UFSECP_OK;
}

/* results[] variant — original behavior, unchanged for existing callers. */
ufsecp_error_t verify_results_impl(ufsecp_lbtc_ctrl* ctrl, Kind k,
                         const uint8_t* rows, std::size_t n,
                         std::size_t key_size, EcdsaSigFormat format,
                         uint8_t* results,
                         const ufsecp_cancel_token* cancel) {
    /* No-op on a degenerate call: a NULL ctrl/rows or empty batch leaves results
     * exactly as the caller initialized it. Callers zero-initialize results, so a
     * degenerate call reads as "all invalid" (fail-closed), never falsely valid. */
    if (!ctrl || n == 0 || !rows) return UFSECP_OK;
    const std::size_t stride = record_size(k) + key_size;
    ResultSink sink{results};
    return verify_core(ctrl, k, rows, n, stride, format, sink, cancel);
}

/* In-place "collect" variant — verdict collapsed into each row's key cell.
 * key_size == 0 is a no-op (no cell to write the verdict into → nothing can be
 * reported → fail-closed: every id survives = all rejected, never falsely
 * accepted). rows MUST be writable (the C++ span overload enforces non-const). */
ufsecp_error_t verify_collect_impl(ufsecp_lbtc_ctrl* ctrl, Kind k,
                         uint8_t* rows, std::size_t n, std::size_t key_size,
                         EcdsaSigFormat format,
                         const ufsecp_cancel_token* cancel) {
    if (!ctrl || n == 0 || !rows || key_size == 0) return UFSECP_OK;
    const std::size_t rec    = record_size(k);
    const std::size_t stride = rec + key_size;
    CollectSink sink{rows, stride, rec, key_size};
    /* rows passed twice: as the const read view to verify_core, and (inside the
     * sink) as the mutable write target. No aliasing — see CollectSink. */
    return verify_core(ctrl, k, static_cast<const uint8_t*>(rows), n, stride, format,
                       sink, cancel);
}

/* Vertical/columnar results[] variant. This is the copy-avoidance path for a
 * caller that stores its batch as independent msg/pub/sig columns already. */
ufsecp_error_t verify_columns_results_impl(ufsecp_lbtc_ctrl* ctrl, Kind k,
                                 const uint8_t* msgs32, const uint8_t* pubs,
                                 const uint8_t* sigs64, std::size_t n,
                                 EcdsaSigFormat format, uint8_t* results,
                                 const ufsecp_cancel_token* cancel) {
    if (!ctrl || n == 0 || !msgs32 || !pubs || !sigs64) return UFSECP_OK;
    ResultSink sink{results};
    try {
        const std::size_t chunk = cancel_chunk_size(cancel);
        for (std::size_t base = 0; base < n; base += chunk) {
            if (cancel_requested(cancel)) return UFSECP_ERR_CANCELLED;
            const std::size_t cnt = (n - base) < chunk ? (n - base) : chunk;
#ifdef UFSECP_LBTC_WITH_GPU
            if (ctrl->gpu &&
                gpu_columns_results(ctrl->gpu, k, msgs32, pubs, sigs64, base, cnt,
                                    format, results))
                continue;
#endif
            cpu_columns_chunk(ctrl->cpu, k, msgs32, pubs, sigs64, base, cnt,
                              format, sink);
        }
    } catch (...) {
        return UFSECP_OK; /* fail-closed: caller-initialized result bytes remain untouched */
    }
    return UFSECP_OK;
}

/* Vertical/columnar collect variant. key_cells is an independent n*key_size
 * correlation-key column. VALID rows are zeroed; INVALID rows are left intact. */
ufsecp_error_t verify_columns_collect_impl(ufsecp_lbtc_ctrl* ctrl, Kind k,
                                 const uint8_t* msgs32, const uint8_t* pubs,
                                 const uint8_t* sigs64, std::size_t n,
                                 uint8_t* key_cells, std::size_t key_size,
                                 EcdsaSigFormat format,
                                 const ufsecp_cancel_token* cancel) {
    if (!ctrl || n == 0 || !msgs32 || !pubs || !sigs64 || !key_cells || key_size == 0)
        return UFSECP_OK;
    KeyColumnSink sink{key_cells, key_size};
    try {
        const std::size_t chunk = cancel_chunk_size(cancel);
        for (std::size_t base = 0; base < n; base += chunk) {
            if (cancel_requested(cancel)) return UFSECP_ERR_CANCELLED;
            const std::size_t cnt = (n - base) < chunk ? (n - base) : chunk;
#ifdef UFSECP_LBTC_WITH_GPU
            if (ctrl->gpu &&
                gpu_columns_collect(ctrl->gpu, k, msgs32, pubs, sigs64, base, cnt,
                                    format, sink))
                continue;
#endif
            cpu_columns_chunk(ctrl->cpu, k, msgs32, pubs, sigs64, base, cnt,
                              format, sink);
        }
    } catch (...) {
        return UFSECP_OK; /* fail-closed: unprocessed key cells stay non-zero/rejected */
    }
    return UFSECP_OK;
}

/* Shared implementation of the public packed-row verify entry points (the shape
 * libbitcoin's ecdsa::batch_verify / schnorr::batch_verify call). Verifies n rows
 * of (record + key_size) bytes and reports per row:
 *   results        OUT, optional: results[i] = 1 valid / 0 invalid.
 *   invalid_idx    OUT, optional: indices of failing rows, up to invalid_cap.
 *   invalid_count  OUT, optional: TOTAL failing rows (all valid iff *invalid_count==0;
 *                  may exceed invalid_cap, so the caller can detect truncation).
 * Returns UFSECP_OK for any well-formed batch (per-row failures are NOT errors — an
 * invalid signature is a verdict, reported via the outputs, never an aborting return,
 * matching libbitcoin's `if (status != UFSECP_OK) std::abort()` fail-fast contract).
 * Fail-closed: a NULL/degenerate call or an unprocessed row reads as invalid. */
ufsecp_error_t verify_rows_impl(ufsecp_lbtc_ctrl* ctrl, Kind k,
                                const uint8_t* rows, std::size_t n, std::size_t key_size,
                                uint8_t* results, std::size_t* invalid_idx,
                                std::size_t invalid_cap, std::size_t* invalid_count,
                                EcdsaSigFormat format,
                                const ufsecp_cancel_token* cancel) {
    if (invalid_count) *invalid_count = 0;
    if (!ctrl) return UFSECP_ERR_NULL_ARG;
    if (n == 0) return UFSECP_OK;                 /* empty batch: vacuously valid */
    if (!rows) return UFSECP_ERR_NULL_ARG;
    if (cancel_requested(cancel)) return UFSECP_ERR_CANCELLED;

    /* Verify into the caller's results[] when provided, else a local scratch.
     * Zero-init = invalid, so any row the verify core does not reach (mid-batch
     * scratch OOM) stays invalid — fail-closed. */
    std::vector<uint8_t> local;
    uint8_t* res = results;
    if (!res) { local.assign(n, 0u); res = local.data(); }
    else std::memset(res, 0, n);

    const ufsecp_error_t rc = verify_results_impl(ctrl, k, rows, n, key_size, format, res, cancel);
    if (rc == UFSECP_ERR_CANCELLED) return rc;

    std::size_t failed = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (res[i] == 0u) {
            if (invalid_idx && failed < invalid_cap) invalid_idx[failed] = i;
            ++failed;
        }
    }
    if (invalid_count) *invalid_count = failed;
    return UFSECP_OK;
}

} // namespace

/* ------------------------------------------------------------------------- */
/* Public API                                                                 */
/* ------------------------------------------------------------------------- */

extern "C" {

ufsecp_error_t ufsecp_lbtc_ctrl_create(ufsecp_lbtc_ctrl** out,
                                       ufsecp_lbtc_backend backend) {
    if (!out) return UFSECP_ERR_NULL_ARG;
    *out = nullptr;

    auto* c = new (std::nothrow) ufsecp_lbtc_ctrl{};
    if (!c) return UFSECP_ERR_INTERNAL;

    if (ufsecp_ctx_create(&c->cpu) != UFSECP_OK || !c->cpu) {
        delete c;
        return UFSECP_ERR_INTERNAL;
    }

#ifdef UFSECP_LBTC_WITH_GPU
    if (backend != UFSECP_LBTC_CPU) {
        const uint32_t order[3] = {UFSECP_GPU_BACKEND_CUDA,
                                   UFSECP_GPU_BACKEND_OPENCL,
                                   UFSECP_GPU_BACKEND_METAL};
        for (uint32_t b : order) {
            if (!ufsecp_gpu_is_available(b)) continue;
            if (ufsecp_gpu_ctx_create(&c->gpu, b, 0) == UFSECP_OK &&
                ufsecp_gpu_is_ready(c->gpu)) {
                c->bound = (b == UFSECP_GPU_BACKEND_CUDA)   ? UFSECP_LBTC_BOUND_CUDA
                         : (b == UFSECP_GPU_BACKEND_OPENCL) ? UFSECP_LBTC_BOUND_OPENCL
                                                            : UFSECP_LBTC_BOUND_METAL;
                ufsecp_gpu_device_info_t info;
                if (ufsecp_gpu_device_info(b, 0, &info) == UFSECP_OK) {
                    std::strncpy(c->device_name, info.name,
                                 sizeof(c->device_name) - 1);
                    c->device_name[sizeof(c->device_name) - 1] = '\0';
                }
                break;
            }
            if (c->gpu) {
                ufsecp_gpu_ctx_destroy(c->gpu);
                c->gpu = nullptr;
            }
        }
        if (!c->gpu && backend == UFSECP_LBTC_GPU) {
            ufsecp_ctx_destroy(c->cpu);
            delete c;
            return UFSECP_ERR_GPU_UNAVAILABLE;
        }
    }
#else
    if (backend == UFSECP_LBTC_GPU) { /* GPU support not compiled in */
        ufsecp_ctx_destroy(c->cpu);
        delete c;
        return UFSECP_ERR_GPU_UNAVAILABLE;
    }
#endif

    *out = c;
    return UFSECP_OK;
}

void ufsecp_lbtc_ctrl_destroy(ufsecp_lbtc_ctrl* ctrl) {
    if (!ctrl) return;
#ifdef UFSECP_LBTC_WITH_GPU
    if (ctrl->gpu) ufsecp_gpu_ctx_destroy(ctrl->gpu);
#endif
    if (ctrl->cpu) ufsecp_ctx_destroy(ctrl->cpu);
    delete ctrl;
}

ufsecp_lbtc_bound ufsecp_lbtc_ctrl_backend(const ufsecp_lbtc_ctrl* ctrl) {
    return ctrl ? ctrl->bound : UFSECP_LBTC_BOUND_CPU;
}

const char* ufsecp_lbtc_ctrl_device_name(const ufsecp_lbtc_ctrl* ctrl) {
    return ctrl ? ctrl->device_name : "";
}

/* ECDSA packed-row verify. `ufsecp_lbtc_verify_ecdsa` defaults to the OPAQUE
 * signature form (libbitcoin's ec_signature, zero-copy); `_opaque` / `_compact`
 * name the form explicitly so an integrator picks whichever it stores. All share
 * the same 8-argument shape and verdict/parity semantics. */
ufsecp_error_t ufsecp_lbtc_verify_ecdsa_opaque(
        ufsecp_lbtc_ctrl* ctrl, const uint8_t* rows, size_t n, size_t key_size,
        uint8_t* results, size_t* invalid_idx, size_t invalid_cap,
        size_t* invalid_count, const ufsecp_cancel_token* cancel) {
    return verify_rows_impl(ctrl, Kind::Ecdsa, rows, n, key_size, results,
                            invalid_idx, invalid_cap, invalid_count,
                            EcdsaSigFormat::Opaque, cancel);
}

ufsecp_error_t ufsecp_lbtc_verify_ecdsa_compact(
        ufsecp_lbtc_ctrl* ctrl, const uint8_t* rows, size_t n, size_t key_size,
        uint8_t* results, size_t* invalid_idx, size_t invalid_cap,
        size_t* invalid_count, const ufsecp_cancel_token* cancel) {
    return verify_rows_impl(ctrl, Kind::Ecdsa, rows, n, key_size, results,
                            invalid_idx, invalid_cap, invalid_count,
                            EcdsaSigFormat::Compact, cancel);
}

ufsecp_error_t ufsecp_lbtc_verify_ecdsa(
        ufsecp_lbtc_ctrl* ctrl, const uint8_t* rows, size_t n, size_t key_size,
        uint8_t* results, size_t* invalid_idx, size_t invalid_cap,
        size_t* invalid_count, const ufsecp_cancel_token* cancel) {
    return ufsecp_lbtc_verify_ecdsa_opaque(ctrl, rows, n, key_size, results,
                                           invalid_idx, invalid_cap, invalid_count,
                                           cancel);
}

/* Schnorr signatures have a single BIP-340 form (no opaque/compact distinction). */
ufsecp_error_t ufsecp_lbtc_verify_schnorr(
        ufsecp_lbtc_ctrl* ctrl, const uint8_t* rows, size_t n, size_t key_size,
        uint8_t* results, size_t* invalid_idx, size_t invalid_cap,
        size_t* invalid_count, const ufsecp_cancel_token* cancel) {
    return verify_rows_impl(ctrl, Kind::Schnorr, rows, n, key_size, results,
                            invalid_idx, invalid_cap, invalid_count,
                            EcdsaSigFormat::Compact, cancel);
}

ufsecp_error_t ufsecp_lbtc_verify_ecdsa_collect(ufsecp_lbtc_ctrl* ctrl,
                                                uint8_t* rows, size_t n,
                                                size_t key_size,
                                                const ufsecp_cancel_token* cancel) {
    return verify_collect_impl(ctrl, Kind::Ecdsa, rows, n, key_size,
                               EcdsaSigFormat::Opaque, cancel);
}

ufsecp_error_t ufsecp_lbtc_verify_schnorr_collect(ufsecp_lbtc_ctrl* ctrl,
                                                  uint8_t* rows, size_t n,
                                                  size_t key_size,
                                                  const ufsecp_cancel_token* cancel) {
    return verify_collect_impl(ctrl, Kind::Schnorr, rows, n, key_size,
                               EcdsaSigFormat::Compact, cancel);
}

ufsecp_error_t ufsecp_lbtc_verify_ecdsa_columns(ufsecp_lbtc_ctrl* ctrl,
                                                const uint8_t* msg_hashes32,
                                                const uint8_t* pubkeys33,
                                                const uint8_t* sigs64,
                                                size_t n,
                                                uint8_t* results,
                                                const ufsecp_cancel_token* cancel) {
    return verify_columns_results_impl(ctrl, Kind::Ecdsa, msg_hashes32, pubkeys33,
                                       sigs64, n, EcdsaSigFormat::Opaque, results,
                                       cancel);
}

ufsecp_error_t ufsecp_lbtc_verify_schnorr_columns(ufsecp_lbtc_ctrl* ctrl,
                                                  const uint8_t* msg_hashes32,
                                                  const uint8_t* pubkeys_x32,
                                                  const uint8_t* sigs64,
                                                  size_t n,
                                                  uint8_t* results,
                                                  const ufsecp_cancel_token* cancel) {
    return verify_columns_results_impl(ctrl, Kind::Schnorr, msg_hashes32, pubkeys_x32,
                                       sigs64, n, EcdsaSigFormat::Compact, results,
                                       cancel);
}

ufsecp_error_t ufsecp_lbtc_verify_ecdsa_columns_collect(ufsecp_lbtc_ctrl* ctrl,
                                                        const uint8_t* msg_hashes32,
                                                        const uint8_t* pubkeys33,
                                                        const uint8_t* sigs64,
                                                        size_t n,
                                                        uint8_t* key_cells,
                                                        size_t key_size,
                                                        const ufsecp_cancel_token* cancel) {
    return verify_columns_collect_impl(ctrl, Kind::Ecdsa, msg_hashes32, pubkeys33,
                                       sigs64, n, key_cells, key_size,
                                       EcdsaSigFormat::Opaque, cancel);
}

ufsecp_error_t ufsecp_lbtc_verify_schnorr_columns_collect(ufsecp_lbtc_ctrl* ctrl,
                                                          const uint8_t* msg_hashes32,
                                                          const uint8_t* pubkeys_x32,
                                                          const uint8_t* sigs64,
                                                          size_t n,
                                                          uint8_t* key_cells,
                                                          size_t key_size,
                                                          const ufsecp_cancel_token* cancel) {
    return verify_columns_collect_impl(ctrl, Kind::Schnorr, msg_hashes32, pubkeys_x32,
                                       sigs64, n, key_cells, key_size,
                                       EcdsaSigFormat::Compact, cancel);
}

/* ---------------------------------------------------------------------------
 * ECDSA signature table packing (build a GPU-native sig column once).
 *
 * For libbitcoin's existing tables NO packing is required: the ECDSA verify
 * entry points (ufsecp_lbtc_verify_ecdsa / _columns) consume the OPAQUE
 * secp256k1_ecdsa_signature bytes (== ec_signature, internal little-endian scalar
 * layout produced by ecdsa_signature_parse_der_lax / _parse_compact) DIRECTLY and
 * low-S normalize internally on both the CPU and the on-device GPU path.
 *
 * These helpers are for callers that want to build the signature column ONCE in
 * the engine's native compact form — big-endian r||s, low-S normalized — so the
 * verify call needs no per-row reformat, and for non-libbitcoin integrators that
 * already store public compact r||s rather than the opaque object. Pair the
 * packed column with ufsecp_lbtc_verify_ecdsa_columns_compact below.
 * ------------------------------------------------------------------------- */
void ufsecp_lbtc_ecdsa_sig_pack(const uint8_t* in, int input_is_opaque,
                                uint8_t out64[64]) {
    if (!in || !out64) return;
    /* Stage through a temp so `out64` may alias `in` (the Opaque path byte-reverses
     * in place, which would corrupt a self-aliased buffer). */
    uint8_t tmp[64];
    copy_ecdsa_signature_normalized(
        in, tmp,
        input_is_opaque ? EcdsaSigFormat::Opaque : EcdsaSigFormat::Compact);
    std::memcpy(out64, tmp, 64);
}

void ufsecp_lbtc_ecdsa_sigs_pack(const uint8_t* in, size_t n,
                                 int input_is_opaque, uint8_t* out) {
    if (!in || !out || n == 0) return;
    const EcdsaSigFormat fmt = input_is_opaque ? EcdsaSigFormat::Opaque
                                               : EcdsaSigFormat::Compact;
    for (size_t i = 0; i < n; ++i) {
        uint8_t tmp[64];
        copy_ecdsa_signature_normalized(in + i * 64, tmp, fmt);
        std::memcpy(out + i * 64, tmp, 64);
    }
}

/* Verify a column of PUBLIC big-endian compact r||s ECDSA signatures (e.g. the
 * output of ufsecp_lbtc_ecdsa_sigs_pack, or any caller that already stores compact
 * sigs). Same verdict + same CPU/GPU parity as ufsecp_lbtc_verify_ecdsa_columns;
 * the only difference is the input sig format (compact instead of opaque). High-S
 * is still low-S normalized, so a not-yet-normalized compact column is also safe. */
ufsecp_error_t ufsecp_lbtc_verify_ecdsa_columns_compact(ufsecp_lbtc_ctrl* ctrl,
                                              const uint8_t* msg_hashes32,
                                              const uint8_t* pubkeys33,
                                              const uint8_t* sigs64,
                                              size_t n,
                                              uint8_t* results,
                                              const ufsecp_cancel_token* cancel) {
    return verify_columns_results_impl(ctrl, Kind::Ecdsa, msg_hashes32, pubkeys33,
                                       sigs64, n, EcdsaSigFormat::Compact, results,
                                       cancel);
}

ufsecp_error_t ufsecp_lbtc_verify_ecdsa_columns_compact_collect(ufsecp_lbtc_ctrl* ctrl,
                                                      const uint8_t* msg_hashes32,
                                                      const uint8_t* pubkeys33,
                                                      const uint8_t* sigs64,
                                                      size_t n,
                                                      uint8_t* key_cells,
                                                      size_t key_size,
                                                      const ufsecp_cancel_token* cancel) {
    return verify_columns_collect_impl(ctrl, Kind::Ecdsa, msg_hashes32, pubkeys33,
                                       sigs64, n, key_cells, key_size,
                                       EcdsaSigFormat::Compact, cancel);
}

/* ---------------------------------------------------------------------------
 * BIP-341 Taproot commitment batch (x-only tweak-add-check).
 *
 * Per item: Q = lift_x(internal, even-y) + tweak*G; accept iff x(Q)==tweaked_x
 * and the y-parity of Q equals the claimed parity. ALL inputs are PUBLIC
 * (variable-time correct). Each check is computed engine-native as one 2-term
 * MSM — Q = 1*P + tweak*G, where P is the compressed even-y point 0x02||internal_x
 * (the MSM's compressed-point parse performs the lift). It is INDEPENDENT and
 * EXACT per row (no random-linear-combination), so it is consensus-safe with no
 * aggregate randomness, and the CPU path fans the rows across a thread pool.
 *
 * (A GPU random-linear-combination fast-check — collapsing the batch to two
 * device MSMs via ufsecp_gpu_msm — is a measured follow-up. For a consensus path
 * its per-row weights r_i MUST be Fiat-Shamir-derived from the batch data, else a
 * crafted block could force the aggregate to pass; that is deliberately not shipped
 * here. A dedicated per-item GPU kernel gives per-row GPU verdicts at a higher
 * ceiling — also a follow-up.)
 * ------------------------------------------------------------------------- */
static inline bool lbtc_commit_one(ufsecp_ctx* ctx, const uint8_t Gc[33],
                                   const uint8_t* internal_x32, const uint8_t* tweak32,
                                   const uint8_t* tweaked_x32, uint8_t parity) {
    uint8_t pts[66], scl[64], Q[33], exp[33];
    pts[0] = 0x02; std::memcpy(pts + 1, internal_x32, 32);  /* lift(internal, even-y) */
    std::memcpy(pts + 33, Gc, 33);                          /* generator             */
    std::memset(scl, 0, 64); scl[31] = 1;                   /* scalar 1 * P          */
    std::memcpy(scl + 32, tweak32, 32);                     /* scalar tweak * G      */
    if (ufsecp_multi_scalar_mul(ctx, scl, pts, 2, Q) != UFSECP_OK) return false;
    exp[0] = parity ? 0x03 : 0x02; std::memcpy(exp + 1, tweaked_x32, 32);
    return std::memcmp(Q, exp, 33) == 0;
}

void ufsecp_lbtc_verify_commitment(ufsecp_lbtc_ctrl* ctrl,
                                   const uint8_t* internal_x32,
                                   const uint8_t* tweak32,
                                   const uint8_t* tweaked_x32,
                                   const uint8_t* parity,
                                   size_t n, uint8_t* results) {
    if (!results || n == 0) return;
    std::memset(results, 0, n);  /* fail-closed default: any early return = all invalid */
    if (!ctrl || !internal_x32 || !tweak32 || !tweaked_x32 || !parity) return;

#ifdef UFSECP_LBTC_WITH_GPU
    /* GPU per-item tweak-add-check (one thread per row). Exact per-row verdict (no
     * RLC) — consensus-identical to the CPU path; fall back on device failure. */
    if (ctrl->gpu) {
        if (ufsecp_gpu_commitment_verify(ctrl->gpu, internal_x32, tweak32,
                                         tweaked_x32, parity, n, results) == UFSECP_OK) return;
        std::memset(results, 0, n);
    }
#endif

    /* Hoisted invariant: G (compressed) computed ONCE for the whole batch. */
    uint8_t Gc[33]; { uint8_t one[32] = {0}; one[31] = 1;
        ufsecp_ctx* gctx = nullptr;
        if (ufsecp_ctx_create(&gctx) != UFSECP_OK || !gctx) return;
        const ufsecp_error_t ge = ufsecp_pubkey_create(gctx, one, Gc);
        ufsecp_ctx_destroy(gctx);
        if (ge != UFSECP_OK) return;
    }

    unsigned T = std::thread::hardware_concurrency(); if (!T) T = 4;
    if (n < 512) T = 1;  /* small batches: thread setup is not worth it */
    std::vector<std::thread> ths; ths.reserve(T);
    const size_t per = (n + T - 1) / T;
    for (unsigned t = 0; t < T; ++t) {
        ths.emplace_back([&, t]() {
            ufsecp_ctx* c = nullptr;                 /* per-thread ctx (ctx is not shared-safe) */
            if (ufsecp_ctx_create(&c) != UFSECP_OK || !c) return;  /* results stay 0 = fail-closed */
            const size_t lo = (size_t)t * per, hi = std::min(n, lo + per);
            for (size_t i = lo; i < hi; ++i)
                results[i] = lbtc_commit_one(c, Gc, internal_x32 + i*32, tweak32 + i*32,
                                             tweaked_x32 + i*32, parity[i]) ? 1u : 0u;
            ufsecp_ctx_destroy(c);
        });
    }
    for (auto& th : ths) th.join();
}

/*
 * GPU random-linear-combination fast-check for the commitment batch.
 *   returns  1 = all commitments valid,
 *            0 = at least one invalid,
 *           -1 = GPU unavailable / device error (caller falls back to the per-row
 *                ufsecp_lbtc_verify_commitment, which is exact and locates failures).
 *
 * Aggregate identity: with weights r_i,
 *     Sum r_i*P_i + (Sum r_i*t_i)*G  ==  Sum r_i*Q_i   iff   every P_i+t_i*G == Q_i.
 * Evaluated as two device MSMs (ufsecp_gpu_msm). The weights r_i are FIAT-SHAMIR
 * derived from a SHA-256 over the ENTIRE batch — e = H(internal_x||tweak||tweaked_x
 * ||parity), r_i = H(e || i) mod n — so a crafted block cannot choose data that
 * forces a false cancellation (that would require grinding SHA-256 into a fixed
 * group relation). A constant r would be forgeable; the per-row CPU path stays the
 * exact consensus reference. PUBLIC data only -> variable-time.
 */
int ufsecp_lbtc_commitment_batch_ok(ufsecp_lbtc_ctrl* ctrl,
                                    const uint8_t* internal_x32,
                                    const uint8_t* tweak32,
                                    const uint8_t* tweaked_x32,
                                    const uint8_t* parity,
                                    size_t n) {
    if (!ctrl || n == 0 || !internal_x32 || !tweak32 || !tweaked_x32 || !parity) return -1;
#ifdef UFSECP_LBTC_WITH_GPU
    if (!ctrl->gpu) return -1;
    using secp256k1::fast::Scalar;
    try {
        /* Fiat-Shamir batch digest e = SHA256(internal_x || tweak || tweaked_x || parity). */
        std::vector<uint8_t> tr;
        tr.reserve(n*32*3 + n);
        tr.insert(tr.end(), internal_x32, internal_x32 + n*32);
        tr.insert(tr.end(), tweak32,      tweak32      + n*32);
        tr.insert(tr.end(), tweaked_x32,  tweaked_x32  + n*32);
        tr.insert(tr.end(), parity,       parity       + n);
        uint8_t e[32];
        if (ufsecp_sha256(tr.data(), tr.size(), e) != UFSECP_OK) return -1;

        /* Generator (compressed), once. */
        uint8_t Gc[33]; { uint8_t one[32]={0}; one[31]=1; ufsecp_ctx* g=nullptr;
            if (ufsecp_ctx_create(&g)!=UFSECP_OK || !g) return -1;
            const ufsecp_error_t ge = ufsecp_pubkey_create(g, one, Gc);
            ufsecp_ctx_destroy(g);
            if (ge != UFSECP_OK) return -1; }

        std::vector<uint8_t> lp((n+1)*33), ls((n+1)*32), qp(n*33);
        Scalar g_coeff = Scalar::zero();
        for (size_t i = 0; i < n; ++i) {
            uint8_t seed[36]; std::memcpy(seed, e, 32);
            seed[32]=(uint8_t)i; seed[33]=(uint8_t)(i>>8);
            seed[34]=(uint8_t)(i>>16); seed[35]=(uint8_t)(i>>24);
            uint8_t rb[32]; if (ufsecp_sha256(seed, 36, rb) != UFSECP_OK) return -1;
            const Scalar r = Scalar::from_bytes(rb);           /* reduced mod n */
            const auto ra = r.to_bytes();
            std::memcpy(ls.data() + i*32, ra.data(), 32);
            lp[i*33] = 0x02; std::memcpy(lp.data()+i*33+1, internal_x32+i*32, 32); /* even-y P_i */
            qp[i*33] = parity[i] ? 0x03 : 0x02; std::memcpy(qp.data()+i*33+1, tweaked_x32+i*32, 32); /* Q_i */
            g_coeff = g_coeff + r * Scalar::from_bytes(tweak32 + i*32);
        }
        std::memcpy(lp.data() + n*33, Gc, 33);
        { const auto ga = g_coeff.to_bytes(); std::memcpy(ls.data() + n*32, ga.data(), 32); }

        uint8_t lhs[33], rhs[33];
        if (ufsecp_gpu_msm(ctrl->gpu, ls.data(), lp.data(), n+1, lhs) != UFSECP_OK) return -1;
        if (ufsecp_gpu_msm(ctrl->gpu, ls.data(), qp.data(), n,   rhs) != UFSECP_OK) return -1;
        return std::memcmp(lhs, rhs, 33) == 0 ? 1 : 0;
    } catch (...) { return -1; }
#else
    return -1;  /* no GPU build — caller uses ufsecp_lbtc_verify_commitment */
#endif
}

/* ---------------------------------------------------------------------------
 * Single-buffer (AoS) commitment batch — one horizontal mmap'd table, one ptr.
 *
 * Each row is a tightly-packed 97-byte record (+ optional caller tail counted in
 * `stride`):
 *     [ internal_x(32) ][ tweak(32) ][ tweaked(33) ]
 * where `tweaked` is the COMPRESSED output key (0x02/0x03 prefix folds in the
 * y-parity — no separate parity column). The verified record is the first 97
 * bytes; stride may be larger (a block-fk / correlation tail the bridge ignores).
 *
 * CPU path: each row is read in place from the mmap at base + i*stride (strided,
 * ZERO copy) and verified independently — Q = lift_x(internal,even)+tweak*G,
 * accept iff its compressed form equals the stored `tweaked` (33 bytes; this
 * single compare captures both x AND parity). Rows fan across a thread pool.
 * ------------------------------------------------------------------------- */
static inline bool lbtc_commit_one_comp(ufsecp_ctx* ctx, const uint8_t Gc[33],
                                        const uint8_t* internal_x32,
                                        const uint8_t* tweak32,
                                        const uint8_t* tweaked_comp33) {
    uint8_t pts[66], scl[64], Q[33];
    pts[0] = 0x02; std::memcpy(pts + 1, internal_x32, 32);  /* lift(internal, even-y) */
    std::memcpy(pts + 33, Gc, 33);
    std::memset(scl, 0, 64); scl[31] = 1;
    std::memcpy(scl + 32, tweak32, 32);
    if (ufsecp_multi_scalar_mul(ctx, scl, pts, 2, Q) != UFSECP_OK) return false;
    return std::memcmp(Q, tweaked_comp33, 33) == 0;  /* x + parity in one compare */
}

void ufsecp_lbtc_verify_commitment_rows(ufsecp_lbtc_ctrl* ctrl,
                                        const uint8_t* rows, size_t n,
                                        size_t stride, uint8_t* results) {
    if (!results || n == 0) return;
    std::memset(results, 0, n);  /* fail-closed default */
    if (!ctrl || !rows || stride < UFSECP_LBTC_COMMITMENT_RECORD) return;

#ifdef UFSECP_LBTC_WITH_GPU
    /* GPU per-item path over the AoS buffer. De-interleave the 97-byte records into
     * columns; the stored `tweaked` is compressed (prefix=parity, x=bytes 1..33). A
     * malformed prefix (not 0x02/0x03) is forced invalid AFTER the kernel so the
     * verdict matches the CPU path's full 33-byte compare bit-for-bit. */
    if (ctrl->gpu) {
        std::vector<uint8_t> ix(n*32), tw(n*32), tx(n*32), par(n);
        for (size_t i = 0; i < n; ++i) {
            const uint8_t* r = rows + i*stride;
            std::memcpy(ix.data()+i*32, r,      32);
            std::memcpy(tw.data()+i*32, r + 32, 32);
            std::memcpy(tx.data()+i*32, r + 65, 32);          /* tweaked_x = comp[1..33] */
            par[i] = (r[64] == 0x03) ? 1u : 0u;               /* y-parity from comp prefix */
        }
        if (ufsecp_gpu_commitment_verify(ctrl->gpu, ix.data(), tw.data(),
                                         tx.data(), par.data(), n, results) == UFSECP_OK) {
            for (size_t i = 0; i < n; ++i) {                  /* match CPU full-compress compare */
                const uint8_t pfx = rows[i*stride + 64];
                if (pfx != 0x02 && pfx != 0x03) results[i] = 0u;
            }
            return;
        }
        std::memset(results, 0, n);
    }
#endif

    uint8_t Gc[33]; { uint8_t one[32] = {0}; one[31] = 1;
        ufsecp_ctx* g = nullptr;
        if (ufsecp_ctx_create(&g) != UFSECP_OK || !g) return;
        const ufsecp_error_t ge = ufsecp_pubkey_create(g, one, Gc);
        ufsecp_ctx_destroy(g);
        if (ge != UFSECP_OK) return;
    }

    unsigned T = std::thread::hardware_concurrency(); if (!T) T = 4;
    if (n < 512) T = 1;
    std::vector<std::thread> ths; ths.reserve(T);
    const size_t per = (n + T - 1) / T;
    for (unsigned t = 0; t < T; ++t) {
        ths.emplace_back([&, t]() {
            ufsecp_ctx* c = nullptr;
            if (ufsecp_ctx_create(&c) != UFSECP_OK || !c) return;  /* results stay 0 */
            const size_t lo = (size_t)t * per, hi = std::min(n, lo + per);
            for (size_t i = lo; i < hi; ++i) {
                const uint8_t* r = rows + i * stride;          /* in-place strided read */
                results[i] = lbtc_commit_one_comp(c, Gc, r, r + 32, r + 64) ? 1u : 0u;
            }
            ufsecp_ctx_destroy(c);
        });
    }
    for (auto& th : ths) th.join();
}

/* GPU RLC fast-check over the single AoS buffer (see ufsecp_lbtc_commitment_batch_ok).
 * Q_i is read straight from the row's compressed `tweaked` field — no repack. */
int ufsecp_lbtc_commitment_batch_ok_rows(ufsecp_lbtc_ctrl* ctrl,
                                         const uint8_t* rows, size_t n, size_t stride) {
    if (!ctrl || n == 0 || !rows || stride < UFSECP_LBTC_COMMITMENT_RECORD) return -1;
#ifdef UFSECP_LBTC_WITH_GPU
    if (!ctrl->gpu) return -1;
    using secp256k1::fast::Scalar;
    try {
        /* Fiat-Shamir digest e = SHA256 over the 97-byte records (excludes any tail). */
        std::vector<uint8_t> tr(n * UFSECP_LBTC_COMMITMENT_RECORD);
        for (size_t i = 0; i < n; ++i)
            std::memcpy(tr.data() + i * UFSECP_LBTC_COMMITMENT_RECORD,
                        rows + i * stride, UFSECP_LBTC_COMMITMENT_RECORD);
        uint8_t e[32];
        if (ufsecp_sha256(tr.data(), tr.size(), e) != UFSECP_OK) return -1;

        uint8_t Gc[33]; { uint8_t one[32]={0}; one[31]=1; ufsecp_ctx* g=nullptr;
            if (ufsecp_ctx_create(&g)!=UFSECP_OK || !g) return -1;
            const ufsecp_error_t ge = ufsecp_pubkey_create(g, one, Gc);
            ufsecp_ctx_destroy(g);
            if (ge != UFSECP_OK) return -1; }

        std::vector<uint8_t> lp((n+1)*33), ls((n+1)*32), qp(n*33);
        Scalar g_coeff = Scalar::zero();
        for (size_t i = 0; i < n; ++i) {
            const uint8_t* r = rows + i * stride;
            uint8_t seed[36]; std::memcpy(seed, e, 32);
            seed[32]=(uint8_t)i; seed[33]=(uint8_t)(i>>8);
            seed[34]=(uint8_t)(i>>16); seed[35]=(uint8_t)(i>>24);
            uint8_t rb[32]; if (ufsecp_sha256(seed, 36, rb) != UFSECP_OK) return -1;
            const Scalar rs = Scalar::from_bytes(rb);
            const auto ra = rs.to_bytes();
            std::memcpy(ls.data() + i*32, ra.data(), 32);
            lp[i*33] = 0x02; std::memcpy(lp.data()+i*33+1, r, 32);   /* P_i = even-y(internal) */
            std::memcpy(qp.data()+i*33, r + 64, 33);                 /* Q_i = stored compressed */
            g_coeff = g_coeff + rs * Scalar::from_bytes(r + 32);     /* + r_i * tweak_i */
        }
        std::memcpy(lp.data() + n*33, Gc, 33);
        { const auto ga = g_coeff.to_bytes(); std::memcpy(ls.data() + n*32, ga.data(), 32); }

        uint8_t lhs[33], rhs[33];
        if (ufsecp_gpu_msm(ctrl->gpu, ls.data(), lp.data(), n+1, lhs) != UFSECP_OK) return -1;
        if (ufsecp_gpu_msm(ctrl->gpu, ls.data(), qp.data(), n,   rhs) != UFSECP_OK) return -1;
        return std::memcmp(lhs, rhs, 33) == 0 ? 1 : 0;
    } catch (...) { return -1; }
#else
    (void)rows; (void)n; (void)stride; return -1;
#endif
}

/* ---------------------------------------------------------------------------
 * Batch x-only pubkey validation (lift_x on-curve check), AoS single-buffer.
 *
 * For each 32-byte x-only key, decide whether it is a valid pubkey x-coordinate
 * (x < p AND x^3+7 is a quadratic residue, i.e. a point lifts). One lift_x per
 * key — a field sqrt + QR test; PUBLIC data, variable-time. results[i] = 1 valid.
 * Engine-native: validate via ufsecp_pubkey_parse(0x02||x). CPU-threaded.
 *
 * Use case: a node's PARALLEL pre-validation pass (validate a block's x-only keys
 * in bulk before sequential script eval). NOTE: the ECDSA/Schnorr/commitment
 * verify batches already lift their keys internally, so this is for a SEPARATE
 * bulk validation — not a redundant second lift on keys you already verify.
 * (A dedicated GPU lift_x kernel — no MSM reuse here, unlike the commitment RLC —
 * is a follow-up; this ships the threaded CPU path.)
 * ------------------------------------------------------------------------- */
static inline bool lbtc_validate_xonly_one(ufsecp_ctx* ctx, const uint8_t* x32) {
    uint8_t comp[33], out[33];
    comp[0] = 0x02; std::memcpy(comp + 1, x32, 32);  /* even-y compressed candidate */
    return ufsecp_pubkey_parse(ctx, comp, 33, out) == UFSECP_OK;
}

void ufsecp_lbtc_validate_xonly(ufsecp_lbtc_ctrl* ctrl,
                                const uint8_t* keys, size_t n,
                                size_t stride, uint8_t* results) {
    if (!results || n == 0) return;
    std::memset(results, 0, n);  /* fail-closed default */
    if (!ctrl || !keys || stride < 32) return;

#ifdef UFSECP_LBTC_WITH_GPU
    /* GPU per-key lift_x fast path (one thread per key). On any device failure we
     * reset and fall through to the CPU reference — consensus-identical. */
    if (ctrl->gpu) {
        const uint8_t* contig = keys;
        std::vector<uint8_t> packed;
        if (stride != 32) {                       /* de-stride into contiguous n*32 */
            packed.resize(n * 32);
            for (size_t i = 0; i < n; ++i) std::memcpy(packed.data() + i*32, keys + i*stride, 32);
            contig = packed.data();
        }
        if (ufsecp_gpu_xonly_validate(ctrl->gpu, contig, n, results) == UFSECP_OK) return;
        std::memset(results, 0, n);               /* device failed → CPU fallback */
    }
#endif

    unsigned T = std::thread::hardware_concurrency(); if (!T) T = 4;
    if (n < 512) T = 1;
    std::vector<std::thread> ths; ths.reserve(T);
    const size_t per = (n + T - 1) / T;
    for (unsigned t = 0; t < T; ++t) {
        ths.emplace_back([&, t]() {
            ufsecp_ctx* c = nullptr;
            if (ufsecp_ctx_create(&c) != UFSECP_OK || !c) return;  /* results stay 0 */
            const size_t lo = (size_t)t * per, hi = std::min(n, lo + per);
            for (size_t i = lo; i < hi; ++i)
                results[i] = lbtc_validate_xonly_one(c, keys + i * stride) ? 1u : 0u;
            ufsecp_ctx_destroy(c);
        });
    }
    for (auto& th : ths) th.join();
}

/* ---------------------------------------------------------------------------
 * Batch BIP-340 tagged hash — Taproot script-tree (leaf / branch) hashing.
 *
 * out_i = tagged_hash(tag, msg_i) = SHA256( SHA256(tag) || SHA256(tag) || msg_i ),
 * for n messages of fixed msg_len (AoS, stride >= msg_len; tail ignored).
 *   TapBranch: tag="TapBranch", msg_len=64 — the caller supplies the two child
 *              hashes ALREADY ordered (min||max) per BIP-341.
 *   TapLeaf  : tag="TapLeaf"  — for fixed-size leaves (variable scripts need a
 *              lengths[] variant; not provided here).
 * PUBLIC data, CPU-threaded. ufsecp_tagged_hash is stateless (no ctx); one warmup
 * call first absorbs any lazily-built tag midstate before the pool starts. A node
 * uses this to hash a block's merkle-tree internal nodes in bulk during parallel
 * pre-validation. `ctrl` is accepted for API uniformity / future GPU SHA dispatch.
 * ------------------------------------------------------------------------- */
void ufsecp_lbtc_tagged_hash_batch(ufsecp_lbtc_ctrl* ctrl, const char* tag,
                                   const uint8_t* msgs, size_t msg_len,
                                   size_t n, size_t stride, uint8_t* out32) {
    (void)ctrl;
    if (!out32 || n == 0) return;
    if (!tag || !msgs || msg_len == 0 || stride < msg_len) return;

#ifdef UFSECP_LBTC_WITH_GPU
    /* GPU path (one thread per message). tag_hash = SHA256(tag) is precomputed once
     * on host; the kernel does SHA256(tag_hash||tag_hash||msg). msg_len capped at
     * 256 on the device; fall back to the CPU pool on any failure. */
    if (ctrl && ctrl->gpu && msg_len <= 256) {
        uint8_t th[32];
        if (ufsecp_sha256(reinterpret_cast<const uint8_t*>(tag), std::strlen(tag), th) == UFSECP_OK) {
            const uint8_t* contig = msgs;
            std::vector<uint8_t> packed;
            if (stride != msg_len) {                  /* de-stride into contiguous n*msg_len */
                packed.resize(n * msg_len);
                for (size_t i = 0; i < n; ++i) std::memcpy(packed.data()+i*msg_len, msgs+i*stride, msg_len);
                contig = packed.data();
            }
            if (ufsecp_gpu_tagged_hash(ctrl->gpu, th, contig, msg_len, n, out32) == UFSECP_OK) return;
        }
    }
#endif

    uint8_t warm[32];                                   /* serialize any lazy tag-midstate init */
    if (ufsecp_tagged_hash(tag, msgs, msg_len, warm) != UFSECP_OK) return;

    unsigned T = std::thread::hardware_concurrency(); if (!T) T = 4;
    if (n < 512) T = 1;
    std::vector<std::thread> ths; ths.reserve(T);
    const size_t per = (n + T - 1) / T;
    for (unsigned t = 0; t < T; ++t) {
        ths.emplace_back([&, t]() {
            const size_t lo = (size_t)t * per, hi = std::min(n, lo + per);
            for (size_t i = lo; i < hi; ++i)
                ufsecp_tagged_hash(tag, msgs + i * stride, msg_len, out32 + i * 32);
        });
    }
    for (auto& th : ths) th.join();
}

/* ---------------------------------------------------------------------------
 * Batch full compressed-pubkey validation (prefix 0x02/0x03 + x<p + on-curve).
 * For CHECKSIG bulk pre-validation. PUBLIC data, variable-time, CPU-threaded;
 * GPU per-key kernel when the controller is GPU-bound (ufsecp_gpu_pubkey_validate).
 * ------------------------------------------------------------------------- */
static inline bool lbtc_validate_pubkey_one(ufsecp_ctx* ctx, const uint8_t* pk33) {
    uint8_t out[33];
    return ufsecp_pubkey_parse(ctx, pk33, 33, out) == UFSECP_OK;
}

void ufsecp_lbtc_validate_pubkeys(ufsecp_lbtc_ctrl* ctrl,
                                  const uint8_t* keys, size_t n,
                                  size_t stride, uint8_t* results) {
    if (!results || n == 0) return;
    std::memset(results, 0, n);
    if (!ctrl || !keys || stride < 33) return;

#ifdef UFSECP_LBTC_WITH_GPU
    if (ctrl->gpu) {
        const uint8_t* contig = keys;
        std::vector<uint8_t> packed;
        if (stride != 33) {
            packed.resize(n * 33);
            for (size_t i = 0; i < n; ++i) std::memcpy(packed.data() + i*33, keys + i*stride, 33);
            contig = packed.data();
        }
        if (ufsecp_gpu_pubkey_validate(ctrl->gpu, contig, n, results) == UFSECP_OK) return;
        std::memset(results, 0, n);
    }
#endif

    unsigned T = std::thread::hardware_concurrency(); if (!T) T = 4;
    if (n < 512) T = 1;
    std::vector<std::thread> ths; ths.reserve(T);
    const size_t per = (n + T - 1) / T;
    for (unsigned t = 0; t < T; ++t) {
        ths.emplace_back([&, t]() {
            ufsecp_ctx* c = nullptr;
            if (ufsecp_ctx_create(&c) != UFSECP_OK || !c) return;
            const size_t lo = (size_t)t * per, hi = std::min(n, lo + per);
            for (size_t i = lo; i < hi; ++i)
                results[i] = lbtc_validate_pubkey_one(c, keys + i * stride) ? 1u : 0u;
            ufsecp_ctx_destroy(c);
        });
    }
    for (auto& th : ths) th.join();
}

/* ---------------------------------------------------------------------------
 * Batch Taproot tagged hash with PER-ITEM length (TapLeaf scripts). PUBLIC data,
 * CPU-threaded; GPU when bound (ufsecp_gpu_tagged_hash_var). lens[i] in 1..256.
 * ------------------------------------------------------------------------- */
void ufsecp_lbtc_tagged_hash_var(ufsecp_lbtc_ctrl* ctrl, const char* tag,
                                 const uint8_t* msgs, const uint32_t* lens,
                                 size_t stride, size_t n, uint8_t* out32) {
    if (!out32 || n == 0) return;
    if (!tag || !msgs || !lens || stride == 0 || stride > 256) return;

    uint8_t warm[32];                                   /* serialize lazy tag-midstate init */
    const uint8_t one = 0;
    if (ufsecp_tagged_hash(tag, &one, 1, warm) != UFSECP_OK) return;

#ifdef UFSECP_LBTC_WITH_GPU
    if (ctrl && ctrl->gpu) {
        uint8_t th[32];
        if (ufsecp_sha256(reinterpret_cast<const uint8_t*>(tag), std::strlen(tag), th) == UFSECP_OK) {
            if (ufsecp_gpu_tagged_hash_var(ctrl->gpu, th, msgs, lens, stride, n, out32) == UFSECP_OK) return;
        }
    }
#endif

    unsigned T = std::thread::hardware_concurrency(); if (!T) T = 4;
    if (n < 512) T = 1;
    std::vector<std::thread> ths; ths.reserve(T);
    const size_t per = (n + T - 1) / T;
    for (unsigned t = 0; t < T; ++t) {
        ths.emplace_back([&, t]() {
            const size_t lo = (size_t)t * per, hi = std::min(n, lo + per);
            for (size_t i = lo; i < hi; ++i) {
                size_t L = lens[i]; if (L > 256) L = 256;
                ufsecp_tagged_hash(tag, msgs + i * stride, L, out32 + i * 32);
            }
        });
    }
    for (auto& th : ths) th.join();
}

/* ---------------------------------------------------------------------------
 * Batch HASH256 (double SHA-256) of fixed-length inputs — merkle node hashing.
 * out32[i*32..] = SHA256(SHA256(inputs[i*input_len..])). PUBLIC, CPU-threaded;
 * GPU when bound (ufsecp_gpu_hash256).
 * ------------------------------------------------------------------------- */
void ufsecp_lbtc_hash256(ufsecp_lbtc_ctrl* ctrl, const uint8_t* inputs,
                         size_t input_len, size_t n, uint8_t* out32) {
    if (!out32 || n == 0) return;
    if (!inputs || input_len == 0) return;

#ifdef UFSECP_LBTC_WITH_GPU
    if (ctrl && ctrl->gpu && input_len <= 320) {
        if (ufsecp_gpu_hash256(ctrl->gpu, inputs, input_len, n, out32) == UFSECP_OK) return;
    }
#endif

    unsigned T = std::thread::hardware_concurrency(); if (!T) T = 4;
    if (n < 512) T = 1;
    std::vector<std::thread> ths; ths.reserve(T);
    const size_t per = (n + T - 1) / T;
    for (unsigned t = 0; t < T; ++t) {
        ths.emplace_back([&, t]() {
            const size_t lo = (size_t)t * per, hi = std::min(n, lo + per);
            for (size_t i = lo; i < hi; ++i) {
                uint8_t h1[32];
                ufsecp_sha256(inputs + i * input_len, input_len, h1);
                ufsecp_sha256(h1, 32, out32 + i * 32);
            }
        });
    }
    for (auto& th : ths) th.join();
}

/* ---------------------------------------------------------------------------
 * Aggregate (random-linear-combination) BIP-340 Schnorr batch verification.
 *
 * For each i a valid signature satisfies  sᵢ·G = Rᵢ + eᵢ·Pᵢ  (BIP-340), with
 * Rᵢ=lift_x(rᵢ), Pᵢ=lift_x(pubkey_xᵢ), eᵢ=tagged_hash("BIP0340/challenge",rᵢ‖Pᵢ‖mᵢ).
 * Summing with random weights aᵢ:  (Σaᵢsᵢ)·G == Σaᵢ·Rᵢ + Σ(aᵢeᵢ)·Pᵢ  iff every
 * signature is valid. Evaluated as two device MSMs (LHS = (Σaᵢsᵢ)·G over the single
 * generator; RHS = the 2n-point combination). The weights aᵢ are FIAT-SHAMIR-derived
 * from a SHA-256 over the whole batch, so a crafted block cannot grind a false
 * cancellation (Bellare-Garay-Rabin; identical construction to the commitment RLC).
 * Returns 1 all-valid / 0 some-invalid / -1 no-GPU-or-device-error. PUBLIC data.
 * ------------------------------------------------------------------------- */
int ufsecp_lbtc_schnorr_aggregate_verify(ufsecp_lbtc_ctrl* ctrl,
                                         const uint8_t* msgs32,
                                         const uint8_t* pubkeys_x32,
                                         const uint8_t* sigs64,
                                         size_t n) {
    if (!ctrl || n == 0 || !msgs32 || !pubkeys_x32 || !sigs64) return -1;
#ifdef UFSECP_LBTC_WITH_GPU
    if (!ctrl->gpu) return -1;
    using secp256k1::fast::Scalar;
    try {
        /* Fiat-Shamir batch digest e = SHA256(msgs || pubkeys || sigs). */
        std::vector<uint8_t> tr; tr.reserve(n*32 + n*32 + n*64);
        tr.insert(tr.end(), msgs32,      msgs32      + n*32);
        tr.insert(tr.end(), pubkeys_x32, pubkeys_x32 + n*32);
        tr.insert(tr.end(), sigs64,      sigs64      + n*64);
        uint8_t e[32];
        if (ufsecp_sha256(tr.data(), tr.size(), e) != UFSECP_OK) return -1;

        /* Generator (compressed), once. */
        uint8_t Gc[33]; { uint8_t one[32]={0}; one[31]=1; ufsecp_ctx* g=nullptr;
            if (ufsecp_ctx_create(&g)!=UFSECP_OK || !g) return -1;
            const ufsecp_error_t ge = ufsecp_pubkey_create(g, one, Gc);
            ufsecp_ctx_destroy(g);
            if (ge != UFSECP_OK) return -1; }

        /* RHS = Σ aᵢ·Rᵢ + Σ (aᵢeᵢ)·Pᵢ  → 2n points; LHS scalar = Σ aᵢ·sᵢ. */
        std::vector<uint8_t> rp(2*n*33), rs(2*n*32);
        Scalar lhs_s = Scalar::zero();
        for (size_t i = 0; i < n; ++i) {
            const uint8_t* r = sigs64 + i*64;          /* R.x */
            const uint8_t* s = sigs64 + i*64 + 32;     /* s   */
            const uint8_t* P = pubkeys_x32 + i*32;     /* P.x (x-only) */
            const uint8_t* m = msgs32 + i*32;

            uint8_t cb[96]; std::memcpy(cb, r, 32); std::memcpy(cb+32, P, 32); std::memcpy(cb+64, m, 32);
            uint8_t eh[32];
            if (ufsecp_tagged_hash("BIP0340/challenge", cb, 96, eh) != UFSECP_OK) return -1;
            const Scalar ei = Scalar::from_bytes(eh);

            uint8_t seed[36]; std::memcpy(seed, e, 32);
            seed[32]=(uint8_t)i; seed[33]=(uint8_t)(i>>8);
            seed[34]=(uint8_t)(i>>16); seed[35]=(uint8_t)(i>>24);
            uint8_t ab[32];
            if (ufsecp_sha256(seed, 36, ab) != UFSECP_OK) return -1;
            const Scalar ai = Scalar::from_bytes(ab);
            const Scalar si = Scalar::from_bytes(s);
            lhs_s = lhs_s + ai * si;

            rp[i*33] = 0x02; std::memcpy(rp.data()+i*33+1, r, 32);          /* Rᵢ = lift_x(r) even-y */
            { const auto a = ai.to_bytes(); std::memcpy(rs.data()+i*32, a.data(), 32); }
            rp[(n+i)*33] = 0x02; std::memcpy(rp.data()+(n+i)*33+1, P, 32);  /* Pᵢ = lift_x(P.x) even-y */
            { const auto ae = (ai * ei).to_bytes(); std::memcpy(rs.data()+(n+i)*32, ae.data(), 32); }
        }

        uint8_t rhs[33], lhs[33];
        if (ufsecp_gpu_msm(ctrl->gpu, rs.data(), rp.data(), 2*n, rhs) != UFSECP_OK) return -1;
        { const auto ls = lhs_s.to_bytes();
          if (ufsecp_gpu_msm(ctrl->gpu, ls.data(), Gc, 1, lhs) != UFSECP_OK) return -1; }
        return std::memcmp(lhs, rhs, 33) == 0 ? 1 : 0;
    } catch (...) { return -1; }
#else
    (void)msgs32; (void)pubkeys_x32; (void)sigs64; return -1;
#endif
}

ufsecp_error_t ufsecp_lbtc_sp_scan(ufsecp_lbtc_ctrl* ctrl,
                                   const uint8_t scan_privkey32[32],
                                   const uint8_t spend_pubkey33[33],
                                   const uint8_t* tweak_pubkeys33, size_t n,
                                   uint64_t* prefix64_out) {
    if (!ctrl) return UFSECP_ERR_NULL_ARG;
    if (n == 0) return UFSECP_OK;
    if (!scan_privkey32 || !spend_pubkey33 || !tweak_pubkeys33 || !prefix64_out)
        return UFSECP_ERR_NULL_ARG;

#ifdef UFSECP_LBTC_WITH_GPU
    /* v1: SP scan is GPU-only (CUDA/OpenCL). CPU SP-scan fallback is a follow-up. */
    if (!ctrl->gpu) return UFSECP_ERR_GPU_UNAVAILABLE;

    for (size_t base = 0; base < n; base += kChunk) {
        const size_t cnt = (n - base) < kChunk ? (n - base) : kChunk;
        const ufsecp_error_t rc = ufsecp_gpu_bip352_scan_batch(
            ctrl->gpu, scan_privkey32, spend_pubkey33,
            tweak_pubkeys33 + base * 33, cnt, prefix64_out + base);
        if (rc != UFSECP_OK) return rc;
    }
    return UFSECP_OK;
#else
    (void)spend_pubkey33; (void)tweak_pubkeys33; (void)prefix64_out;
    return UFSECP_ERR_GPU_UNAVAILABLE; /* GPU support not compiled in */
#endif
}

} // extern "C"
