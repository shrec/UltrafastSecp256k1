/**
 * ufsecp_libbitcoin.cpp — implementation of the libbitcoin acceleration bridge.
 *
 * The controller is dispatch + marshalling only. It owns a CPU context (always)
 * and, when built with GPU support and a device is available, a GPU context.
 * It reuses the engine's existing, tested public primitives:
 *
 *   CPU:  ufsecp_ecdsa_batch_verify / ufsecp_schnorr_batch_verify
 *           (packed records — exactly the row layout this bridge documents)
 *   GPU:  ufsecp_gpu_ecdsa_verify_batch / ufsecp_gpu_schnorr_verify_batch
 *           (per-item results — one signature per thread)
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
#include "ufsecp_gpu.h" /* GPU C ABI + UFSECP_GPU_BACKEND_*, UFSECP_ERR_GPU_* */
#endif

#include <cstring>
#include <new>
#include <vector>

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

inline std::size_t record_size(Kind k) {
    return k == Kind::Ecdsa ? UFSECP_LBTC_ECDSA_RECORD : UFSECP_LBTC_SCHNORR_RECORD;
}

inline ufsecp_error_t cpu_verify_one(ufsecp_ctx* ctx, Kind k, const uint8_t* rec) {
    /* For n == 1 the packed verify reads exactly record_size() bytes, so the
     * row's opaque key tail (if any) is never touched. */
    return k == Kind::Ecdsa ? ufsecp_ecdsa_batch_verify(ctx, rec, 1)
                            : ufsecp_schnorr_batch_verify(ctx, rec, 1);
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

/* Translate one libbitcoin row into the engine's packed record layout.
 *   ECDSA   row == engine record:        32 msg | 33 pubkey | 64 sig.
 *   Schnorr row (uniform with ECDSA):    32 msg | 32 xonly  | 64 sig,
 *           but the engine record is     32 xonly | 32 msg  | 64 sig — so the
 *           first two fields are swapped on the way in.
 * `out` must hold record_size(k) bytes. */
inline void to_engine_record(Kind k, const uint8_t* row, uint8_t* out) {
    if (k == Kind::Ecdsa) {
        std::memcpy(out, row, UFSECP_LBTC_ECDSA_RECORD);   /* identical layout */
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
               Sink& sink) {
    const std::size_t rec = record_size(k);

    /* ECDSA: the libbitcoin row IS the engine record, so verify in place. The
     * fast all-valid path applies when rows are contiguous (no opaque key
     * column splitting the stride). */
    if (k == Kind::Ecdsa) {
        if (stride == rec &&
            cpu_verify_run(ctx, k, rows + base * stride, cnt) == UFSECP_OK) {
            sink.mark_all_valid(base, cnt);
            return;
        }
        for (std::size_t i = 0; i < cnt; ++i)
            sink.mark(base + i,
                      cpu_verify_one(ctx, k, rows + (base + i) * stride) == UFSECP_OK);
        return;
    }

    /* Schnorr: the libbitcoin field order differs from the engine record, so
     * reorder the chunk into an engine-format contiguous scratch (stride == rec)
     * and reuse the same fast/per-row engine verify. */
    std::vector<uint8_t> eng(cnt * rec);
    for (std::size_t i = 0; i < cnt; ++i)
        to_engine_record(k, rows + (base + i) * stride, eng.data() + i * rec);

    if (cpu_verify_run(ctx, k, eng.data(), cnt) == UFSECP_OK) {
        sink.mark_all_valid(base, cnt);
        return;
    }
    for (std::size_t i = 0; i < cnt; ++i)
        sink.mark(base + i, cpu_verify_one(ctx, k, eng.data() + i * rec) == UFSECP_OK);
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
               Sink& sink) {
    std::vector<uint8_t> msg(cnt * 32), sig(cnt * 64), res(cnt);
    std::vector<uint8_t> pub(cnt * (k == Kind::Ecdsa ? 33u : 32u));

    for (std::size_t i = 0; i < cnt; ++i) {
        const uint8_t* r = rows + (base + i) * stride;
        if (k == Kind::Ecdsa) {
            /* record: 32 msg | 33 pubkey | 64 sig */
            std::memcpy(msg.data() + i * 32, r, 32);
            std::memcpy(pub.data() + i * 33, r + 32, 33);
            std::memcpy(sig.data() + i * 64, r + 65, 64);
        } else {
            /* libbitcoin row: 32 msg | 32 xonly | 64 sig. The engine GPU ABI
             * takes (msg, pubkey_x, sig) — extract at the libbitcoin offsets. */
            std::memcpy(msg.data() + i * 32, r, 32);
            std::memcpy(pub.data() + i * 32, r + 32, 32);
            std::memcpy(sig.data() + i * 64, r + 64, 64);
        }
    }

    const ufsecp_error_t rc =
        k == Kind::Ecdsa
            ? ufsecp_gpu_ecdsa_verify_batch(gpu, msg.data(), pub.data(),
                                            sig.data(), cnt, res.data())
            : ufsecp_gpu_schnorr_verify_batch(gpu, msg.data(), pub.data(),
                                              sig.data(), cnt, res.data());
    if (rc != UFSECP_OK) return false; /* fall back to CPU */

    for (std::size_t i = 0; i < cnt; ++i)
        sink.mark(base + i, res[i] != 0);
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
void verify_core(ufsecp_lbtc_ctrl* ctrl, Kind k, const uint8_t* rows,
                 std::size_t n, std::size_t stride, Sink& sink) {
    try {
        for (std::size_t base = 0; base < n; base += kChunk) {
            const std::size_t cnt = (n - base) < kChunk ? (n - base) : kChunk;
#ifdef UFSECP_LBTC_WITH_GPU
            if (ctrl->gpu) {
                if (gpu_chunk(ctrl->gpu, k, rows, base, cnt, stride, sink)) continue;
                /* device-level failure → mandatory CPU fallback for this chunk */
            }
#endif
            cpu_chunk(ctrl->cpu, k, rows, base, cnt, stride, sink);
        }
    } catch (...) {
        return; /* fail-closed: unprocessed rows remain at caller initial state */
    }
}

/* results[] variant — original behavior, unchanged for existing callers. */
void verify_results_impl(ufsecp_lbtc_ctrl* ctrl, Kind k,
                         const uint8_t* rows, std::size_t n,
                         std::size_t key_size, uint8_t* results) {
    /* No-op on a degenerate call: a NULL ctrl/rows or empty batch leaves results
     * exactly as the caller initialized it. Callers zero-initialize results, so a
     * degenerate call reads as "all invalid" (fail-closed), never falsely valid. */
    if (!ctrl || n == 0 || !rows) return;
    const std::size_t stride = record_size(k) + key_size;
    ResultSink sink{results};
    verify_core(ctrl, k, rows, n, stride, sink);
}

/* In-place "collect" variant — verdict collapsed into each row's key cell.
 * key_size == 0 is a no-op (no cell to write the verdict into → nothing can be
 * reported → fail-closed: every id survives = all rejected, never falsely
 * accepted). rows MUST be writable (the C++ span overload enforces non-const). */
void verify_collect_impl(ufsecp_lbtc_ctrl* ctrl, Kind k,
                         uint8_t* rows, std::size_t n, std::size_t key_size) {
    if (!ctrl || n == 0 || !rows || key_size == 0) return;
    const std::size_t rec    = record_size(k);
    const std::size_t stride = rec + key_size;
    CollectSink sink{rows, stride, rec, key_size};
    /* rows passed twice: as the const read view to verify_core, and (inside the
     * sink) as the mutable write target. No aliasing — see CollectSink. */
    verify_core(ctrl, k, static_cast<const uint8_t*>(rows), n, stride, sink);
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

void ufsecp_lbtc_verify_ecdsa(ufsecp_lbtc_ctrl* ctrl,
                              const uint8_t* rows, size_t n,
                              size_t key_size, uint8_t* results) {
    verify_results_impl(ctrl, Kind::Ecdsa, rows, n, key_size, results);
}

void ufsecp_lbtc_verify_schnorr(ufsecp_lbtc_ctrl* ctrl,
                                const uint8_t* rows, size_t n,
                                size_t key_size, uint8_t* results) {
    verify_results_impl(ctrl, Kind::Schnorr, rows, n, key_size, results);
}

void ufsecp_lbtc_verify_ecdsa_collect(ufsecp_lbtc_ctrl* ctrl,
                                      uint8_t* rows, size_t n,
                                      size_t key_size) {
    verify_collect_impl(ctrl, Kind::Ecdsa, rows, n, key_size);
}

void ufsecp_lbtc_verify_schnorr_collect(ufsecp_lbtc_ctrl* ctrl,
                                        uint8_t* rows, size_t n,
                                        size_t key_size) {
    verify_collect_impl(ctrl, Kind::Schnorr, rows, n, key_size);
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
