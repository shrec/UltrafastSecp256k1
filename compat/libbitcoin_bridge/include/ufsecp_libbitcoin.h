/**
 * ufsecp_libbitcoin.h — UltrafastSecp256k1 ⇄ libbitcoin acceleration bridge.
 *
 * A single "shim controller" entry point that gives a Bitcoin full node (or any
 * consumer) two GPU-accelerated batch capabilities with a MANDATORY CPU
 * fallback, behind one tiny C ABI. Pure C / C++ — no FFI, no language bindings
 * required.
 *
 *   1. Script-signature batch verification  (ECDSA + Schnorr / BIP-340)
 *        - The signature/key/sighash triples extracted from Bitcoin scripts
 *          (CHECKSIG / CHECKMULTISIG / Taproot CHECKSIG / CHECKSIGADD).
 *        - Signatures are ~95% of script validation cost; this is the big win
 *          for IBD / historical block validation. NOT a mempool-latency path.
 *        - GPU verifies each signature on its own thread, so PER-ROW pass/fail
 *          is produced naturally — exactly what a node needs to locate the bad
 *          signature and map it back to a block/tx.
 *
 *   2. BIP-352 Silent Payments scan batch
 *        - GPU-accelerated ECDH scan for silent-payment indexing (Electrum /
 *          server use case). Reuses the existing engine SP scan pipeline.
 *
 * Design contract (agreed with the libbitcoin maintainer):
 *   - ECDSA and Schnorr are SEPARATE calls. The data stays homogeneous and
 *     uniform-sized, and the two kinds stack independently (one GPU stream
 *     each, or one card each).
 *   - The caller passes ONE unified table per call. Each row is:
 *         [ signature record ][ optional opaque correlation key ]
 *     The bridge verifies only the signature record and never interprets the
 *     opaque key bytes. The key is the caller's own tag (e.g. 3-byte block id,
 *     4-byte tx id). It is carried purely so an invalid row can be mapped back
 *     to its block/tx without a second side table.
 *   - The opaque key column is VARIABLE-sized: pass `key_size` on the call.
 *     `key_size == 0` disables it entirely (fastest; row stride == record size).
 *   - SIZING CONTRACT (per the libbitcoin maintainer): the call passes exactly
 *     two of {record count, key size, buffer size} — namely the RECORD COUNT and
 *     the KEY SIZE. The buffer size is IMPLIED, always = count * (RECORD +
 *     key_size). There is no buffer-size argument, so there is no buffer/stride
 *     mismatch and no corresponding error condition.
 *   - Results are returned directly: a per-row pass/fail byte array (results[i]
 *     == 1 valid / 0 invalid). The verify call returns void — there is no error
 *     code (the only recoverable failure, no usable backend, is reported at
 *     controller creation). The caller maps a failing row back to its block/tx
 *     via the opaque correlation tag carried in that row.
 *   - Two storage formats are supported:
 *       * row API:     [record][opaque key] repeated, for a single packed table.
 *       * column API:  msg_hashes[], pubkeys[], sigs[] plus optional key column,
 *                      for callers that already maintain vertical mmap columns and
 *                      want to avoid bridge-side row de-interleave.
 *   - ECDSA rows mirror libbitcoin's ec_signature storage, which is a copied
 *     secp256k1_ecdsa_signature object (opaque libsecp-compatible scalar
 *     layout), not public compact r||s. This is the accepted bridge contract:
 *     libbitcoin can pass its existing rows unchanged. On the CPU batch path the
 *     bridge passes those bytes to the engine's opaque-row C ABI; on the GPU row
 *     path the backend receives the same strided rows and parses the opaque
 *     scalars on device. No intermediate compact row table or msg/pub/sig staging
 *     columns are built for the row API. Consensus-valid high-S signatures are
 *     accepted and normalized during parse, preserving
 *     libsecp256k1/fallback verification semantics without rewriting caller-owned
 *     rows or columns.
 *
 * Consensus note:
 *   For block validation the GPU path is used as a consensus-bearing
 *   accelerator. Correctness is anchored on the CPU/libsecp256k1-equivalent
 *   reference: the GPU result MUST match the CPU result bit-for-bit, which is
 *   enforced by a differential test gate. The CPU fallback is always available
 *   and is never optional.
 */
#ifndef UFSECP_LIBBITCOIN_H
#define UFSECP_LIBBITCOIN_H

#include <stddef.h>
#include <stdint.h>
#include "ufsecp/ufsecp_error.h" /* ufsecp_error_t, UFSECP_OK, UFSECP_ERR_* */
#include "ufsecp/ufsecp_cancel.h" /* ufsecp_cancel_fn, ufsecp_cancel_token, UFSECP_CANCEL_DEFAULT */

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------- */
/* Record layouts (signature payload only — excludes any opaque key column).  */
/* ------------------------------------------------------------------------- */

/* Uniform field order across both kinds — hash/msg first, then pubkey, then
 * signature (matches the libbitcoin ecdsa::triple / schnorr::triple structs,
 * so a packed struct array forwards into the row API directly). */
/* ECDSA row:   32-byte msghash      | 33-byte compressed pubkey | 64-byte opaque secp256k1_ecdsa_signature. */
#define UFSECP_LBTC_ECDSA_RECORD   129u
/* Schnorr row: 32-byte msg/sighash  | 32-byte x-only pubkey     | 64-byte sig. */
#define UFSECP_LBTC_SCHNORR_RECORD 128u
/* BIP-341 commitment AoS record (single-buffer / mmap-direct):
 *   32-byte x-only internal key | 32-byte tweak | 33-byte COMPRESSED tweaked key
 * The compressed prefix (0x02/0x03) folds in the y-parity, so there is no separate
 * parity column. The verified record is 97 bytes; the row stride may be larger to
 * carry a caller correlation tail (block-fk etc.) the bridge ignores. */
#define UFSECP_LBTC_COMMITMENT_RECORD 97u

/* ------------------------------------------------------------------------- */
/* Controller lifecycle.                                                      */
/* ------------------------------------------------------------------------- */

/* Opaque controller. Create once, reuse across many batches. It owns the GPU
 * backend (if any) and the CPU fallback. Not internally synchronized: use one
 * controller per worker thread, or serialize calls externally. */
typedef struct ufsecp_lbtc_ctrl ufsecp_lbtc_ctrl;

/* Backend selection at creation time. */
typedef enum {
    UFSECP_LBTC_AUTO = 0, /* GPU if usable, else CPU (recommended).            */
    UFSECP_LBTC_GPU  = 1, /* Require a GPU; create fails with no usable GPU.   */
    UFSECP_LBTC_CPU  = 2  /* Force the CPU fallback only.                      */
} ufsecp_lbtc_backend;

/* The backend the controller actually bound (query after create). */
typedef enum {
    UFSECP_LBTC_BOUND_CPU    = 0,
    UFSECP_LBTC_BOUND_CUDA   = 1,
    UFSECP_LBTC_BOUND_OPENCL = 2,
    UFSECP_LBTC_BOUND_METAL  = 3
} ufsecp_lbtc_bound;

/* Optional caller-driven cancellation token for long batch verification.
 * NULL preserves the original behavior. A non-NULL token is polled between
 * chunks; check_interval == 0 uses the engine default chunk size. Returning
 * non-zero from is_cancelled yields UFSECP_ERR_CANCELLED. Cancellation is an
 * execution status: partial results/key-cells are caller-discarded and must not
 * be interpreted as consensus verdicts.
 *
 * ufsecp_cancel_fn / ufsecp_cancel_token / UFSECP_CANCEL_DEFAULT now live in the
 * shared header "ufsecp/ufsecp_cancel.h" (single canonical definition so this
 * bridge and the shim batch-verify API agree on layout). The struct is unchanged,
 * byte-for-byte; the historical UFSECP_LBTC_CANCEL_DEFAULT name is kept as an
 * alias for source compatibility. */
#ifndef UFSECP_LBTC_CANCEL_DEFAULT
#define UFSECP_LBTC_CANCEL_DEFAULT UFSECP_CANCEL_DEFAULT
#endif

/* Create / destroy the controller. On success *out receives a non-NULL handle.
 * UFSECP_LBTC_AUTO never fails for lack of a GPU — it silently binds CPU. */
ufsecp_error_t ufsecp_lbtc_ctrl_create(ufsecp_lbtc_ctrl** out,
                                       ufsecp_lbtc_backend backend);
void           ufsecp_lbtc_ctrl_destroy(ufsecp_lbtc_ctrl* ctrl);

/* Which backend is bound, and a human-readable device name (may be NULL). */
ufsecp_lbtc_bound ufsecp_lbtc_ctrl_backend(const ufsecp_lbtc_ctrl* ctrl);
const char*       ufsecp_lbtc_ctrl_device_name(const ufsecp_lbtc_ctrl* ctrl);

/* ------------------------------------------------------------------------- */
/* 1. Script-signature batch verification.                                    */
/* ------------------------------------------------------------------------- */

/*
 * Verify a homogeneous batch of ECDSA (resp. Schnorr) signatures.
 *
 *   ctrl      the controller (one per worker thread). Must be non-NULL and
 *             successfully created — controller creation is where the only
 *             recoverable failure (no usable backend) is reported; once you
 *             hold a live controller, verification itself has no error return.
 *   rows      n rows, each (RECORD + key_size) bytes, tightly packed. The
 *             signature record occupies the first RECORD bytes; the trailing
 *             key_size bytes are the caller's opaque correlation tag, carried
 *             but never interpreted.
 *   n         number of rows (the record COUNT). Together with key_size this
 *             fully determines the buffer — n * (RECORD + key_size) bytes — so
 *             there is no buffer-size argument and no size-mismatch error.
 *   key_size  opaque trailing bytes per row; 0 to disable. For a packed record
 *             struct carrying a trailing id this is sizeof(id) (e.g. 3).
 *   results   OUT, n bytes. results[i] == 1 if row i's signature is valid,
 *             0 if invalid (structurally malformed rows — off-curve pubkey,
 *             s >= n, R.x >= p — verify to 0, never abort the batch). ECDSA
 *             signature bytes are libsecp-compatible opaque scalar storage, as
 *             used by libbitcoin::ec_signature when libsecp256k1 is the backing
 *             ABI. The CPU batch path parses that existing layout directly into
 *             engine batch entries; the GPU row path parses the same row layout
 *             on device, avoiding bridge-side compact/signature-column staging.
 *             High-S signatures with s < n are consensus-valid and verify to 1;
 *             they are normalized internally without mutating the row. The
 *             caller maps a failing row back to its block/tx via the opaque
 *             tag at rows[i] (or its own table) — no second side table needed.
 *
 *   results       OUT, optional. n bytes; results[i] = 1 valid / 0 invalid.
 *   invalid_idx   OUT, optional. Receives the indices of failing rows, up to
 *                 invalid_cap entries (ignored if NULL).
 *   invalid_cap   capacity of invalid_idx in entries.
 *   invalid_count OUT, optional. Set to the TOTAL number of failing rows — which
 *                 may exceed invalid_cap, so the caller can detect truncation.
 *                 All rows valid iff *invalid_count == 0.
 *
 * Returns UFSECP_OK for any well-formed batch: a signature outcome is a normal
 * per-row result, never an aborting return (so a node may treat any non-OK return
 * as an unrecoverable fault and fail fast). A NULL ctrl returns UFSECP_ERR_NULL_ARG;
 * n == 0 is a vacuously-valid no-op (*invalid_count = 0). Pass `results` and/or
 * `invalid_idx`/`invalid_count` — any combination, or none.
 *
 * ECDSA signature form:
 *   ufsecp_lbtc_verify_ecdsa          — OPAQUE (== ufsecp_lbtc_verify_ecdsa_opaque):
 *                                       the libsecp-compatible ec_signature bytes,
 *                                       zero-copy; the engine low-S normalizes.
 *   ufsecp_lbtc_verify_ecdsa_opaque   — same, named explicitly.
 *   ufsecp_lbtc_verify_ecdsa_compact  — public big-endian compact r||s in the row's
 *                                       signature field (also low-S normalized).
 *   Pick whichever matches how you store signatures. Schnorr has a single BIP-340
 *   form, so `ufsecp_lbtc_verify_schnorr` takes no opaque/compact variant.
 */
ufsecp_error_t ufsecp_lbtc_verify_ecdsa(ufsecp_lbtc_ctrl* ctrl,
                                        const uint8_t* rows, size_t n,
                                        size_t key_size, uint8_t* results,
                                        size_t* invalid_idx, size_t invalid_cap,
                                        size_t* invalid_count,
                                        const ufsecp_cancel_token* cancel UFSECP_LBTC_CANCEL_DEFAULT);

ufsecp_error_t ufsecp_lbtc_verify_ecdsa_opaque(ufsecp_lbtc_ctrl* ctrl,
                                        const uint8_t* rows, size_t n,
                                        size_t key_size, uint8_t* results,
                                        size_t* invalid_idx, size_t invalid_cap,
                                        size_t* invalid_count,
                                        const ufsecp_cancel_token* cancel UFSECP_LBTC_CANCEL_DEFAULT);

ufsecp_error_t ufsecp_lbtc_verify_ecdsa_compact(ufsecp_lbtc_ctrl* ctrl,
                                        const uint8_t* rows, size_t n,
                                        size_t key_size, uint8_t* results,
                                        size_t* invalid_idx, size_t invalid_cap,
                                        size_t* invalid_count,
                                        const ufsecp_cancel_token* cancel UFSECP_LBTC_CANCEL_DEFAULT);

ufsecp_error_t ufsecp_lbtc_verify_schnorr(ufsecp_lbtc_ctrl* ctrl,
                                        const uint8_t* rows, size_t n,
                                        size_t key_size, uint8_t* results,
                                        size_t* invalid_idx, size_t invalid_cap,
                                        size_t* invalid_count,
                                        const ufsecp_cancel_token* cancel UFSECP_LBTC_CANCEL_DEFAULT);

/*
 * Multi-threaded twins of the packed-row verify entry points. Identical shape and
 * per-row verdict/invalid_idx/invalid_count/parity semantics, plus a `max_threads`
 * budget that fans the CPU verify across cores:
 *     max_threads == 0  -> auto (std::thread::hardware_concurrency)
 *     max_threads == 1  -> serial (byte-for-byte identical to the non-mt function)
 *     max_threads == N  -> up to N
 * Verification is variable-time over PUBLIC data only, so threading is a pure
 * throughput change; the per-row verdict is identical to the serial path for any
 * thread count. The GPU path is unaffected (max_threads governs the CPU fallback
 * only). Cancellation still polls the token between chunks. The single-threaded
 * functions above remain for integrators that already shard across their own
 * thread pool (use those to avoid nested pools / oversubscription).
 * NOTE: ufsecp_lbtc_verify_ecdsa_compact_mt accepts max_threads for API symmetry
 * but its CPU fallback verifies per-row (serial); prefer the opaque form for MT.
 */
ufsecp_error_t ufsecp_lbtc_verify_ecdsa_mt(ufsecp_lbtc_ctrl* ctrl,
                                        const uint8_t* rows, size_t n,
                                        size_t key_size, uint8_t* results,
                                        size_t* invalid_idx, size_t invalid_cap,
                                        size_t* invalid_count, size_t max_threads,
                                        const ufsecp_cancel_token* cancel UFSECP_LBTC_CANCEL_DEFAULT);

ufsecp_error_t ufsecp_lbtc_verify_ecdsa_opaque_mt(ufsecp_lbtc_ctrl* ctrl,
                                        const uint8_t* rows, size_t n,
                                        size_t key_size, uint8_t* results,
                                        size_t* invalid_idx, size_t invalid_cap,
                                        size_t* invalid_count, size_t max_threads,
                                        const ufsecp_cancel_token* cancel UFSECP_LBTC_CANCEL_DEFAULT);

ufsecp_error_t ufsecp_lbtc_verify_ecdsa_compact_mt(ufsecp_lbtc_ctrl* ctrl,
                                        const uint8_t* rows, size_t n,
                                        size_t key_size, uint8_t* results,
                                        size_t* invalid_idx, size_t invalid_cap,
                                        size_t* invalid_count, size_t max_threads,
                                        const ufsecp_cancel_token* cancel UFSECP_LBTC_CANCEL_DEFAULT);

ufsecp_error_t ufsecp_lbtc_verify_schnorr_mt(ufsecp_lbtc_ctrl* ctrl,
                                        const uint8_t* rows, size_t n,
                                        size_t key_size, uint8_t* results,
                                        size_t* invalid_idx, size_t invalid_cap,
                                        size_t* invalid_count, size_t max_threads,
                                        const ufsecp_cancel_token* cancel UFSECP_LBTC_CANCEL_DEFAULT);

/*
 * Columnar/vertical form of the same verification API. These calls are intended
 * for large batch producers that already store independent columns:
 *
 *     ECDSA:   msg_hashes32[n][32], pubkeys33[n][33], sigs64[n][64]
 *     Schnorr: msg_hashes32[n][32], pubkeys_x32[n][32], sigs64[n][64]
 *
 * The per-row verdict is byte-identical to the packed-row API. ECDSA signature
 * column entries use the same libsecp-compatible opaque layout as the packed-row
 * API. CPU verification parses them directly into batch entries; GPU verification
 * still stages compact signature columns for device upload. High-S signatures are
 * normalized in the parsed/staged entry; the caller-owned signature column is not
 * mutated. Degenerate calls are no-ops; callers should zero-initialize results
 * for fail-closed behavior.
 */
ufsecp_error_t ufsecp_lbtc_verify_ecdsa_columns(ufsecp_lbtc_ctrl* ctrl,
                                      const uint8_t* msg_hashes32,
                                      const uint8_t* pubkeys33,
                                      const uint8_t* sigs64,
                                      size_t n,
                                      uint8_t* results,
                                      const ufsecp_cancel_token* cancel UFSECP_LBTC_CANCEL_DEFAULT);

ufsecp_error_t ufsecp_lbtc_verify_schnorr_columns(ufsecp_lbtc_ctrl* ctrl,
                                        const uint8_t* msg_hashes32,
                                        const uint8_t* pubkeys_x32,
                                        const uint8_t* sigs64,
                                        size_t n,
                                        uint8_t* results,
                                        const ufsecp_cancel_token* cancel UFSECP_LBTC_CANCEL_DEFAULT);

/* ------------------------------------------------------------------------- */
/* 1a'. ECDSA signature packing — build a GPU-native sig table/column once.    */
/* ------------------------------------------------------------------------- */
/*
 * IMPORTANT — for libbitcoin you do NOT need these. The ECDSA verify entry points
 * (ufsecp_lbtc_verify_ecdsa / _columns) consume the OPAQUE secp256k1_ecdsa_signature
 * bytes — i.e. your ec_signature, the output of ecdsa_signature_parse_der_lax /
 * secp256k1_ecdsa_signature_parse_compact — DIRECTLY, and low-S normalize internally
 * on both the CPU and the on-device GPU path. Pass your existing rows/columns
 * unchanged.
 *
 * Use these helpers only if you want to PRE-BUILD the signature table once in the
 * engine's native compact form (public big-endian r||s, low-S normalized) so the
 * verify call does zero per-row reformatting, OR if you are a non-libbitcoin
 * integrator that already stores public compact r||s instead of the opaque object.
 *
 * ufsecp_lbtc_ecdsa_sig_pack:
 *   in              64 bytes. input_is_opaque != 0 -> `in` is an opaque
 *                   secp256k1_ecdsa_signature (== ec_signature, internal little-endian
 *                   scalar layout). input_is_opaque == 0 -> `in` is public big-endian
 *                   compact r||s.
 *   out64           64-byte normalized big-endian compact r||s (low-S). MAY alias `in`.
 *   No validation of scalar range (verify rejects out-of-range/zero). Never fails.
 *
 * ufsecp_lbtc_ecdsa_sigs_pack: the same, applied to n signatures (stride 64). `out`
 *   may alias `in` for an in-place pack.
 *
 * Verify a packed/compact column with ufsecp_lbtc_verify_ecdsa_columns_compact
 * (results) or ..._compact_collect (in-place key-cell verdict): identical verdict
 * and CPU/GPU parity to the opaque columns API — only the input sig format differs.
 */
void ufsecp_lbtc_ecdsa_sig_pack(const uint8_t* in, int input_is_opaque,
                                uint8_t out64[64]);

void ufsecp_lbtc_ecdsa_sigs_pack(const uint8_t* in, size_t n,
                                 int input_is_opaque, uint8_t* out);

ufsecp_error_t ufsecp_lbtc_verify_ecdsa_columns_compact(ufsecp_lbtc_ctrl* ctrl,
                                              const uint8_t* msg_hashes32,
                                              const uint8_t* pubkeys33,
                                              const uint8_t* sigs64,
                                              size_t n,
                                              uint8_t* results,
                                              const ufsecp_cancel_token* cancel UFSECP_LBTC_CANCEL_DEFAULT);

ufsecp_error_t ufsecp_lbtc_verify_ecdsa_columns_compact_collect(ufsecp_lbtc_ctrl* ctrl,
                                                      const uint8_t* msg_hashes32,
                                                      const uint8_t* pubkeys33,
                                                      const uint8_t* sigs64,
                                                      size_t n,
                                                      uint8_t* key_cells,
                                                      size_t key_size,
                                                      const ufsecp_cancel_token* cancel UFSECP_LBTC_CANCEL_DEFAULT);

/* ------------------------------------------------------------------------- */
/* 1b. In-place "collect" verify — verdict collapsed into the row's key cell.  */
/* ------------------------------------------------------------------------- */

/*
 * Same verification as ufsecp_lbtc_verify_ecdsa/_schnorr, but with NO results
 * array. The per-row verdict is written IN PLACE into each row's trailing
 * key_size-byte cell:
 *
 *     VALID   row  ->  memset(key cell, 0, key_size)   (key zeroed)
 *     INVALID row  ->  key cell left exactly as the caller supplied it
 *
 * The caller's correlation id therefore SURVIVES only on rejected rows. A single
 * post-pass that "collects every row whose key cell is non-zero" yields the
 * rejected-id set directly — no second side table, no results[]. (This is the
 * inverse of the results-based variant, which NEVER touches the key cell. They
 * are mutually exclusive views of the same verification; pick one per call.)
 *
 *   ctrl      bound controller (ufsecp_lbtc_ctrl_create).
 *   rows      IN/OUT, MUTABLE, n rows of (RECORD + key_size) bytes, tightly
 *             packed. The first RECORD bytes are the signature record (read
 *             only); the trailing key_size bytes are the caller's id AND the
 *             result channel (zeroed on valid, untouched on invalid). MUST point
 *             to writable memory — a caller that mmap'd the source read-only must
 *             remap MAP_PRIVATE / PROT_WRITE first.
 *   n         record COUNT. May be arbitrarily large: n is walked internally in
 *             device-sized chunks, so there is no maximum and no size error.
 *   key_size  trailing bytes per row, and the result channel. MUST be > 0:
 *             with key_size == 0 there is no cell to write the verdict into, so
 *             the call is a NO-OP (nothing can be reported — fail-closed).
 *
 * Returns UFSECP_OK or UFSECP_ERR_CANCELLED. A degenerate call
 * (NULL rows, n == 0, or key_size == 0) writes nothing; the key cells stay as
 * supplied, i.e. every id survives = "all rejected" = fail-closed, never falsely
 * accepted. On cancellation the caller must discard the partially-mutated key
 * cells. An unrecoverable mid-batch condition (scratch OOM) abandons the loop:
 * processed rows keep their zeroed/left verdict; rows not yet reached keep their
 * non-zero id = rejected. Fail-closed end to end.
 */
ufsecp_error_t ufsecp_lbtc_verify_ecdsa_collect(ufsecp_lbtc_ctrl* ctrl,
                                      uint8_t* rows, size_t n,
                                      size_t key_size,
                                      const ufsecp_cancel_token* cancel UFSECP_LBTC_CANCEL_DEFAULT);

ufsecp_error_t ufsecp_lbtc_verify_schnorr_collect(ufsecp_lbtc_ctrl* ctrl,
                                        uint8_t* rows, size_t n,
                                        size_t key_size,
                                        const ufsecp_cancel_token* cancel UFSECP_LBTC_CANCEL_DEFAULT);

/* Multi-threaded twins of the in-place collect entry points (same verdict
 * semantics; max_threads 0=auto/all cores, 1=serial, N=cap — see the packed-row
 * _mt note above for the threading/GPU/cancellation contract). */
ufsecp_error_t ufsecp_lbtc_verify_ecdsa_collect_mt(ufsecp_lbtc_ctrl* ctrl,
                                        uint8_t* rows, size_t n,
                                        size_t key_size, size_t max_threads,
                                        const ufsecp_cancel_token* cancel UFSECP_LBTC_CANCEL_DEFAULT);

ufsecp_error_t ufsecp_lbtc_verify_schnorr_collect_mt(ufsecp_lbtc_ctrl* ctrl,
                                        uint8_t* rows, size_t n,
                                        size_t key_size, size_t max_threads,
                                        const ufsecp_cancel_token* cancel UFSECP_LBTC_CANCEL_DEFAULT);

/*
 * Columnar/vertical collect form. The verified bytes live in msg/pub/sig columns;
 * key_cells is a separate n * key_size correlation-key column. VALID rows have
 * their whole key cell zeroed; INVALID rows keep their key cell intact. With
 * key_size == 0 the call is a no-op/fail-closed, matching the packed collect API.
 */
ufsecp_error_t ufsecp_lbtc_verify_ecdsa_columns_collect(ufsecp_lbtc_ctrl* ctrl,
                                              const uint8_t* msg_hashes32,
                                              const uint8_t* pubkeys33,
                                              const uint8_t* sigs64,
                                              size_t n,
                                              uint8_t* key_cells,
                                              size_t key_size,
                                              const ufsecp_cancel_token* cancel UFSECP_LBTC_CANCEL_DEFAULT);

ufsecp_error_t ufsecp_lbtc_verify_schnorr_columns_collect(ufsecp_lbtc_ctrl* ctrl,
                                                const uint8_t* msg_hashes32,
                                                const uint8_t* pubkeys_x32,
                                                const uint8_t* sigs64,
                                                size_t n,
                                                uint8_t* key_cells,
                                                size_t key_size,
                                                const ufsecp_cancel_token* cancel UFSECP_LBTC_CANCEL_DEFAULT);

/* ------------------------------------------------------------------------- */
/* 2. BIP-352 Silent Payments scan batch.                                     */
/* ------------------------------------------------------------------------- */

/*
 * Scan a batch of input-tweak public keys for a single recipient (scan key +
 * spend pubkey), producing one 64-bit output prefix per tweak for fast wallet
 * matching. Mirrors the engine GPU scan pipeline (ufsecp_gpu_bip352_scan_batch)
 * with a CPU fallback.
 *
 *   scan_privkey32  the recipient scan private key (secret — CT path on CPU,
 *                   branchless on GPU; erased after use).
 *   spend_pubkey33  the recipient spend public key (compressed).
 *   tweak_pubkeys33 n * 33 bytes, one compressed input-tweak pubkey per entry.
 *   n               number of tweaks.
 *   prefix64_out    OUT, n * uint64_t output prefixes for matching.
 */
/* ------------------------------------------------------------------------- */
/* 1c. BIP-341 Taproot commitment batch (x-only tweak-add-check).             */
/* ------------------------------------------------------------------------- */

/*
 * Batch the BIP-341 key-path commitment check
 * (secp256k1_xonly_pubkey_tweak_add_check): for each item, accept iff
 *     Q = lift_x(internal_x, even-y) + tweak*G   has   x(Q) == tweaked_x
 *     and the y-parity of Q equals `parity`.
 * All inputs are PUBLIC (variable-time correct). Columns / struct-of-arrays:
 *
 *   internal_x32  n*32  x-only internal keys.
 *   tweak32       n*32  BIP-341 tweaks (each must be in [1, n)).
 *   tweaked_x32   n*32  claimed tweaked output x-only keys.
 *   parity        n     claimed y-parity of each tweaked key (0 even / 1 odd).
 *   n             rows.
 *   results       OUT, n bytes. results[i] == 1 if the commitment holds, else 0.
 *                 A malformed row (internal_x off-curve, tweak >= n, Q at
 *                 infinity) verifies to 0 — never aborts the batch.
 *
 * Each row is INDEPENDENT and EXACT (no random-linear-combination), so the batch
 * is per-row and consensus-safe with no aggregate randomness; the CPU path runs
 * the rows across a thread pool. (A GPU random-linear-combination fast-check and
 * a per-item GPU kernel are follow-ups — see the .cpp note. `ctrl` is accepted
 * for ABI stability / future GPU dispatch.) Returns void; a degenerate call
 * (NULL args, n == 0) leaves results as the caller initialized it — zero-init for
 * a fail-closed "all invalid".
 */
void ufsecp_lbtc_verify_commitment(ufsecp_lbtc_ctrl* ctrl,
                                   const uint8_t* internal_x32,
                                   const uint8_t* tweak32,
                                   const uint8_t* tweaked_x32,
                                   const uint8_t* parity,
                                   size_t n, uint8_t* results);

/*
 * GPU random-linear-combination fast-check (the "GPU version" of the commitment
 * batch). Collapses the whole batch to two device multi-scalar-mults and returns
 * a single aggregate verdict:
 *     1 = ALL commitments valid,
 *     0 = at least one invalid,
 *    -1 = no usable GPU / device error  → use ufsecp_lbtc_verify_commitment instead.
 * On 0 (or -1), call ufsecp_lbtc_verify_commitment to get per-row results[] and
 * locate the failure(s). The weights are Fiat-Shamir-derived from a SHA-256 over
 * the entire batch, so the aggregate is not forgeable (a constant weight would be).
 * All inputs PUBLIC -> variable-time. Backend-agnostic: rides ufsecp_gpu_msm, which
 * has full Pippenger MSM on CUDA, OpenCL and Metal, so it runs on whichever GPU the
 * controller binds. Measured ceiling ~2.5 M checks/s (CUDA, RTX-class).
 */
int ufsecp_lbtc_commitment_batch_ok(ufsecp_lbtc_ctrl* ctrl,
                                    const uint8_t* internal_x32,
                                    const uint8_t* tweak32,
                                    const uint8_t* tweaked_x32,
                                    const uint8_t* parity,
                                    size_t n);

/*
 * Single-buffer (array-of-structs) commitment batch — one horizontal mmap'd table,
 * one pointer, no repack. Each row is a tightly-packed UFSECP_LBTC_COMMITMENT_RECORD
 * (97-byte) record, optionally followed by a caller tail (counted in `stride`):
 *     [ internal_x(32) ][ tweak(32) ][ tweaked(33, compressed: prefix = parity) ]
 *
 *   rows     n rows of `stride` bytes, tightly packed (your mmap'd table verbatim).
 *   n        row count.   stride  >= 97 (record + optional correlation tail).
 *   results  OUT n bytes, 1 valid / 0 invalid.
 *
 * CPU path reads each record IN PLACE at rows + i*stride (zero copy) and verifies
 * it independently across a thread pool. Degenerate call (NULL/n==0/stride<97) ->
 * fail-closed (results zeroed). Same exact per-row semantics as the columns
 * verify_commitment; the tweaked key being compressed makes x+parity a single
 * compare.
 */
void ufsecp_lbtc_verify_commitment_rows(ufsecp_lbtc_ctrl* ctrl,
                                        const uint8_t* rows, size_t n,
                                        size_t stride, uint8_t* results);

/* GPU RLC aggregate fast-check over the single AoS buffer (1 all-valid / 0 some-
 * invalid / -1 no GPU). Q_i is read straight from each row's compressed `tweaked`
 * field (no repack). Same Fiat-Shamir contract as ufsecp_lbtc_commitment_batch_ok. */
int ufsecp_lbtc_commitment_batch_ok_rows(ufsecp_lbtc_ctrl* ctrl,
                                         const uint8_t* rows, size_t n, size_t stride);

/*
 * Batch x-only pubkey validation (lift_x on-curve check), AoS single-buffer.
 * For each 32-byte x-only key, results[i] = 1 if it is a valid pubkey x-coordinate
 * (x < p and x^3+7 is a quadratic residue, i.e. a point lifts), else 0. One lift_x
 * per key (field sqrt + QR), PUBLIC data, variable-time, CPU-threaded.
 *
 *   keys     n keys of `stride` bytes (the 32-byte x first; stride >= 32 may carry
 *            a caller tail the bridge ignores). Pass your mmap'd column verbatim.
 *   results  OUT n bytes, 1 valid / 0 invalid. Degenerate call -> fail-closed.
 *
 * For a node's PARALLEL pre-validation. The verify batches already lift internally,
 * so use this only for SEPARATE bulk validation. (A GPU lift_x kernel is a follow-up
 * — there is no MSM reuse here, unlike the commitment RLC.)
 */
void ufsecp_lbtc_validate_xonly(ufsecp_lbtc_ctrl* ctrl,
                                const uint8_t* keys, size_t n,
                                size_t stride, uint8_t* results);

/*
 * Batch BIP-340 tagged hash — Taproot script-tree (leaf / branch) hashing.
 * out32[i*32..] = tagged_hash(tag, msg_i) for n fixed-length messages (AoS,
 * stride >= msg_len; trailing bytes ignored). PUBLIC data, CPU-threaded.
 *   TapBranch: tag="TapBranch", msg_len=64 (caller supplies child hashes already
 *              ordered min||max per BIP-341).
 *   TapLeaf  : tag="TapLeaf" for fixed-size leaves.
 *   msgs     n messages of `stride` bytes each (first msg_len bytes hashed).
 *   out32    OUT n*32 bytes. Degenerate call (NULL/n==0/msg_len==0/stride<msg_len)
 *            -> no-op. `ctrl` accepted for API uniformity (CPU-threaded today).
 */
void ufsecp_lbtc_tagged_hash_batch(ufsecp_lbtc_ctrl* ctrl, const char* tag,
                                   const uint8_t* msgs, size_t msg_len,
                                   size_t n, size_t stride, uint8_t* out32);

/*
 * Batch FULL compressed-pubkey validation (CHECKSIG pre-validation pass).
 * results[i] = 1 iff keys[i*stride..] is a valid compressed pubkey (prefix
 * 0x02/0x03, x < p, on-curve). PUBLIC data, variable-time; stride >= 33 may carry
 * a tail the bridge ignores. Complements validate_xonly (which is x-only / 32B).
 */
void ufsecp_lbtc_validate_pubkeys(ufsecp_lbtc_ctrl* ctrl,
                                  const uint8_t* keys, size_t n,
                                  size_t stride, uint8_t* results);

/*
 * Batch Taproot tagged hash with PER-ITEM length (TapLeaf scripts of varying size).
 * out32[i*32..] = tagged_hash(tag, msgs[i*stride .. +lens[i]]). Each lens[i] in
 * 1..256. PUBLIC data, CPU-threaded (GPU when bound). The fixed-length
 * ufsecp_lbtc_tagged_hash_batch stays for the uniform TapBranch case.
 */
void ufsecp_lbtc_tagged_hash_var(ufsecp_lbtc_ctrl* ctrl, const char* tag,
                                 const uint8_t* msgs, const uint32_t* lens,
                                 size_t stride, size_t n, uint8_t* out32);

/*
 * Batch HASH256 (double SHA-256) of fixed-length inputs — e.g. merkle-tree node
 * hashing (input_len = 64 for a left||right pair). out32[i*32..] =
 * SHA256(SHA256(inputs[i*input_len..])). PUBLIC data, CPU-threaded (GPU when bound).
 */
void ufsecp_lbtc_hash256(ufsecp_lbtc_ctrl* ctrl, const uint8_t* inputs,
                         size_t input_len, size_t n, uint8_t* out32);

/*
 * Aggregate (random-linear-combination) BIP-340 Schnorr batch verification.
 *   returns  1 = all n signatures valid,
 *            0 = at least one invalid,
 *           -1 = GPU unavailable / device error (caller falls back to per-row
 *                ufsecp_lbtc_verify_schnorr to locate the failure).
 *
 * Checks Σaᵢ·sᵢ·G == Σaᵢ·Rᵢ + Σ(aᵢ·eᵢ)·Pᵢ via two device MSMs, where Rᵢ=lift_x(rᵢ),
 * Pᵢ=lift_x(pubkey_xᵢ), eᵢ=tagged_hash("BIP0340/challenge", rᵢ‖Pᵢ‖mᵢ). The weights
 * aᵢ are FIAT-SHAMIR-derived from a SHA-256 over the whole batch (e=H(msgs‖pubkeys‖
 * sigs), aᵢ=H(e‖i) mod n) — a crafted batch cannot force a false cancellation
 * (Bellare-Garay-Rabin small-exponents test; same construction as the commitment
 * RLC). Columns input: msgs32 = n*32, pubkeys_x32 = n*32 (x-only), sigs64 = n*64.
 * PUBLIC data -> variable-time. GPU-only (rides ufsecp_gpu_msm); -1 without a GPU.
 */
int ufsecp_lbtc_schnorr_aggregate_verify(ufsecp_lbtc_ctrl* ctrl,
                                         const uint8_t* msgs32,
                                         const uint8_t* pubkeys_x32,
                                         const uint8_t* sigs64,
                                         size_t n);

ufsecp_error_t ufsecp_lbtc_sp_scan(ufsecp_lbtc_ctrl* ctrl,
                                   const uint8_t scan_privkey32[32],
                                   const uint8_t spend_pubkey33[33],
                                   const uint8_t* tweak_pubkeys33, size_t n,
                                   uint64_t* prefix64_out);

/*
 * BIP-352 silent-payment scan + 8-byte prefix match (issue #312).
 *
 * A direct port of the per-row Silent Payments scanning pipeline that the DuckDB extension
 * (`duckdb-ufsecp-extension`, used by Sparrow/Frigate) runs in `ProcessBatch`, exposed so a
 * node can call it without DuckDB. The byte layouts match that extension exactly so the same
 * data can flow through unchanged. For each of `count` rows, given the tweak point
 * `tweaks[i]`:
 *
 *     shared = scan_privkey * tweak_point                     (k*P, via the GLV plan)
 *     hash   = TaggedHash("BIP0352/SharedSecret", compress(shared) || counter_be32(0))
 *     output = spend_pubkey + hash * G
 *     prefix = ExtractUpper64(output.x)   = big-endian top 8 bytes of the output x-coordinate
 *     matches[i] = (prefix == prefixes[i]) ? 1 : 0
 *
 * Byte layouts (EXACTLY as the DuckDB extension / Frigate send them):
 *   - scan_privkey32 : 32-byte scalar, LITTLE-ENDIAN.
 *   - spend_pubkey64 : uncompressed point, x(32 LE) || y(32 LE).
 *   - tweaks         : count * 64 bytes, each an uncompressed point x(32 LE) || y(32 LE).
 *   - prefixes       : count uint64, each the BIG-ENDIAN top 8 bytes of a candidate output's
 *                      x-coordinate (the taproot-output prefix the scanner joined to this row;
 *                      same convention as ufsecp_lbtc_sp_scan's prefix64_out / extract_upper_64).
 *   - matches        : count bytes (caller-allocated), written 0/1.
 *
 * CPU implementation (matches the extension's CPU `ProcessBatch`); works on any build. The
 * heavy work is the per-row scalar multiply. Returns the number of matching rows (>= 0), or
 * -1 on a NULL-argument / malformed-key error.
 */
int ufsecp_lbtc_match_silent_prefixes(const uint8_t scan_privkey32[32],
                                      const uint8_t spend_pubkey64[64],
                                      const uint8_t* tweaks,
                                      const uint64_t* prefixes,
                                      size_t count,
                                      uint8_t* matches);

#ifdef __cplusplus
} /* extern "C" */

#if __cplusplus >= 202002L
#  include <span>
#  include <array>
#  include <type_traits>
#endif

/* ------------------------------------------------------------------------- */
/* Optional thin C++ RAII convenience wrapper (header-only, zero overhead).   */
/* ------------------------------------------------------------------------- */
namespace ufsecp {
namespace lbtc {

// ---------------------------------------------------------------------------
// Canonical packed record structs — the single source of truth for the on-wire
// byte layout, so a node's DB / mmap layout and the engine agree on the exact
// bytes. Verified fields are FIRST, in on-wire order; any correlation id is the
// caller's own trailing bytes appended to each record. Each on-wire row is
// [ EcdsaRecord(129) | key_size bytes ] (key_size may be 0). To verify, pass the
// row pointer + the record COUNT + the KEY SIZE. The buffer carries NO size: it
// is fully determined by count * (RECORD + key_size), so it can never mismatch
// and there is no size-related error condition. results[i] maps back to record i.
//
// FIELD ORDER — IMPORTANT: the message / sighash is FIRST for BOTH kinds. Schnorr
// is UNIFORM with ECDSA (hash, then key, then sig); it is NOT x-only-key-first at
// this public boundary. (The engine's internal Schnorr record is xonly-first; the
// bridge re-marshals it for you, so callers never see that order.)
// ---------------------------------------------------------------------------
#pragma pack(push, 1)
struct EcdsaRecord {     // == UFSECP_LBTC_ECDSA_RECORD (129) bytes
    uint8_t hash[32];    // message hash (sighash)
    uint8_t point[33];   // compressed public key
    uint8_t sig[64];     // opaque libsecp-compatible secp256k1_ecdsa_signature
};
struct SchnorrRecord {   // == UFSECP_LBTC_SCHNORR_RECORD (128) bytes
    uint8_t hash[32];    // message / sighash  (FIRST — uniform with ECDSA)
    uint8_t xonly[32];   // x-only public key (BIP-340)
    uint8_t sig[64];     // BIP-340 signature: R.x || s
};
#pragma pack(pop)
static_assert(sizeof(EcdsaRecord) == UFSECP_LBTC_ECDSA_RECORD,
              "EcdsaRecord must be exactly 129 bytes, tightly packed");
static_assert(sizeof(SchnorrRecord) == UFSECP_LBTC_SCHNORR_RECORD,
              "SchnorrRecord must be exactly 128 bytes, tightly packed");

// ---------------------------------------------------------------------------
// Multisig / threshold table rows — libbitcoin's 4-table batching model.
//
// libbitcoin accumulates signature tuples into FOUR homogeneous tables so each
// stacks independently at a uniform row size:
//
//     ecdsa     [ EcdsaRecord(129)   ][ block-fk(3) ]                       single
//     schnorr   [ SchnorrRecord(128) ][ block-fk(3) ]                       single
//     multisig  [ EcdsaRecord(129)   ][ m|n(1) ][ group(2) ][ block-fk(3) ] m-of-n
//     threshold [ SchnorrRecord(128) ][ m|n(1) ][ group(2) ][ block-fk(3) ] m-of-n
//
// At THIS verification boundary the four tables collapse to TWO verify kinds:
// each multisig row is one ECDSA signature over (hash, point, sig); each
// threshold row (tapscript CHECKSIG / CHECKSIGADD) is one Schnorr signature over
// (hash, xonly, sig). The m|n + group + block-fk bytes are libbitcoin accounting
// metadata, carried in the opaque correlation tail and NEVER interpreted by the
// bridge — m-of-n satisfaction is the node's job, downstream of per-row verify.
//
// So multisig verifies through the ECDSA path and threshold through the Schnorr
// path, both with key_size == 6 (the m|n + group + block-fk tail). The structs
// below are the canonical on-wire rows so a node's packed table forwards through
// verify_multisig / verify_threshold (and the collect twins) without a side table.
//
// COLLECT note: collect zeroes the ENTIRE key tail on a VALID row and leaves it
// intact on INVALID. A rejected row therefore retains m|n, group AND block-fk
// for mapping; a valid row's accounting bytes are cleared (valid rows are purged
// per window, so this is intended). A caller that must keep the accounting bytes
// for valid rows uses the results[] variant, which never writes the row.
// ---------------------------------------------------------------------------
#pragma pack(push, 1)
struct MultisigRow {            // 135 bytes: ECDSA record + m|n + group + block-fk
    EcdsaRecord record;         // 129: hash | point | sig    (the verified bytes)
    uint8_t     mn;             //   1: packed m|n            (accounting; opaque)
    uint8_t     group[2];       //   2: multisig group id     (accounting; opaque)
    uint8_t     block_fk[3];    //   3: block correlation id  (opaque)
};
struct ThresholdRow {           // 134 bytes: Schnorr record + m|n + group + block-fk
    SchnorrRecord record;       // 128: hash | xonly | sig    (the verified bytes)
    uint8_t       mn;           //   1: packed m|n            (accounting; opaque)
    uint8_t       group[2];     //   2: threshold group id    (accounting; opaque)
    uint8_t       block_fk[3];  //   3: block correlation id  (opaque)
};
#pragma pack(pop)
static_assert(sizeof(MultisigRow) == UFSECP_LBTC_ECDSA_RECORD + 6,
              "MultisigRow must be exactly 135 bytes (129 record + 6 tag), packed");
static_assert(sizeof(ThresholdRow) == UFSECP_LBTC_SCHNORR_RECORD + 6,
              "ThresholdRow must be exactly 134 bytes (128 record + 6 tag), packed");

// Single-buffer (AoS) BIP-341 commitment record — the canonical on-wire row for
// ufsecp_lbtc_verify_commitment_rows. The tweaked key is COMPRESSED so its
// 0x02/0x03 prefix carries the y-parity (no separate parity column). Append your
// own correlation bytes after it; the row stride = sizeof(CommitmentRow) + tail.
#pragma pack(push, 1)
struct CommitmentRow {          // 97 bytes
    uint8_t internal_x[32];     // x-only internal key (even-y implicit per BIP-341)
    uint8_t tweak[32];          // BIP-341 tweak (in [1, n))
    uint8_t tweaked[33];        // compressed output key: 0x02/0x03 prefix = y-parity
};
#pragma pack(pop)
static_assert(sizeof(CommitmentRow) == UFSECP_LBTC_COMMITMENT_RECORD,
              "CommitmentRow must be exactly 97 bytes (32+32+33), tightly packed");

class Controller {
public:
    explicit Controller(ufsecp_lbtc_backend backend = UFSECP_LBTC_AUTO) {
        (void)ufsecp_lbtc_ctrl_create(&ctrl_, backend);
    }
    ~Controller() { ufsecp_lbtc_ctrl_destroy(ctrl_); }
    Controller(const Controller&) = delete;
    Controller& operator=(const Controller&) = delete;

    bool ok() const { return ctrl_ != nullptr; }
    ufsecp_lbtc_ctrl* get() const { return ctrl_; }
    ufsecp_lbtc_bound backend() const { return ufsecp_lbtc_ctrl_backend(ctrl_); }

    // --- Batch verify --------------------------------------------------------
    // Pass the record COUNT and the KEY SIZE (key_size == 0 == no correlation key)
    // plus the per-row results buffer. The row pointer has NO size argument: the
    // buffer is fully determined by count * (RECORD + key_size) bytes, so it can
    // never mismatch and there is no size-related error condition. Returns
    // UFSECP_OK or UFSECP_ERR_CANCELLED when an optional cancel token asks the
    // batch to stop. results[i] (1=valid/0=invalid) is valid only on UFSECP_OK;
    // the caller maps a failing row back to its block/tx via the opaque tag at
    // rows[i]. ECDSA high-S rows are normalized in scratch, never in the caller
    // buffer. (Per evoskuil: the buffer size is redundant with count+key_size.)
    ufsecp_error_t verify_ecdsa(const uint8_t* rows, size_t count, size_t key_size,
                      uint8_t* results,
                      const ufsecp_cancel_token* cancel = nullptr) const {
        return ufsecp_lbtc_verify_ecdsa(ctrl_, rows, count, key_size, results,
                                        nullptr, 0, nullptr, cancel);
    }
    ufsecp_error_t verify_schnorr(const uint8_t* rows, size_t count, size_t key_size,
                        uint8_t* results,
                        const ufsecp_cancel_token* cancel = nullptr) const {
        return ufsecp_lbtc_verify_schnorr(ctrl_, rows, count, key_size, results,
                                          nullptr, 0, nullptr, cancel);
    }

    // Columnar / vertical verify. Use when the node already stores independent
    // msg/pub/sig columns and wants to avoid bridge-side row de-interleave.
    ufsecp_error_t verify_ecdsa_columns(const uint8_t* msg_hashes32, const uint8_t* pubkeys33,
                              const uint8_t* sigs64, size_t count,
                              uint8_t* results,
                              const ufsecp_cancel_token* cancel = nullptr) const {
        return ufsecp_lbtc_verify_ecdsa_columns(
            ctrl_, msg_hashes32, pubkeys33, sigs64, count, results, cancel);
    }
    ufsecp_error_t verify_schnorr_columns(const uint8_t* msg_hashes32, const uint8_t* pubkeys_x32,
                                const uint8_t* sigs64, size_t count,
                                uint8_t* results,
                                const ufsecp_cancel_token* cancel = nullptr) const {
        return ufsecp_lbtc_verify_schnorr_columns(
            ctrl_, msg_hashes32, pubkeys_x32, sigs64, count, results, cancel);
    }

    // Multisig (ECDSA m-of-n) and threshold (Schnorr / tapscript m-of-n) tables.
    // Intent-revealing aliases for verify_ecdsa / verify_schnorr — identical
    // verification, named for the libbitcoin table they consume. The m|n+group
    // bytes ride in the opaque key tail; key_size is its width (6 for the
    // canonical MultisigRow / ThresholdRow: m|n + group + block-fk).
    ufsecp_error_t verify_multisig(const uint8_t* rows, size_t count, size_t key_size,
                         uint8_t* results,
                         const ufsecp_cancel_token* cancel = nullptr) const {
        return ufsecp_lbtc_verify_ecdsa(ctrl_, rows, count, key_size, results,
                                        nullptr, 0, nullptr, cancel);
    }
    ufsecp_error_t verify_threshold(const uint8_t* rows, size_t count, size_t key_size,
                          uint8_t* results,
                          const ufsecp_cancel_token* cancel = nullptr) const {
        return ufsecp_lbtc_verify_schnorr(ctrl_, rows, count, key_size, results,
                                          nullptr, 0, nullptr, cancel);
    }
    ufsecp_error_t verify_multisig_columns(const uint8_t* msg_hashes32, const uint8_t* pubkeys33,
                                 const uint8_t* sigs64, size_t count,
                                 uint8_t* results,
                                 const ufsecp_cancel_token* cancel = nullptr) const {
        return verify_ecdsa_columns(msg_hashes32, pubkeys33, sigs64, count, results, cancel);
    }
    ufsecp_error_t verify_threshold_columns(const uint8_t* msg_hashes32, const uint8_t* pubkeys_x32,
                                  const uint8_t* sigs64, size_t count,
                                  uint8_t* results,
                                  const ufsecp_cancel_token* cancel = nullptr) const {
        return verify_schnorr_columns(msg_hashes32, pubkeys_x32, sigs64, count, results, cancel);
    }

    // --- Collect (in-place) verify ------------------------------------------
    // No results buffer: the verdict is written into each row's trailing key
    // cell (key_size MUST be > 0) — zeroed on valid, left intact on invalid. The
    // caller then collects every row whose key cell is still non-zero = the
    // rejected-id set. `rows` MUST be writable. key_size == 0 is a no-op.
    ufsecp_error_t collect_ecdsa(uint8_t* rows, size_t count, size_t key_size,
                                const ufsecp_cancel_token* cancel = nullptr) const {
        return ufsecp_lbtc_verify_ecdsa_collect(ctrl_, rows, count, key_size, cancel);
    }
    ufsecp_error_t collect_schnorr(uint8_t* rows, size_t count, size_t key_size,
                                  const ufsecp_cancel_token* cancel = nullptr) const {
        return ufsecp_lbtc_verify_schnorr_collect(ctrl_, rows, count, key_size, cancel);
    }

    // Columnar / vertical collect. key_cells is n * key_size bytes and is the
    // only mutable column; msg/pub/sig columns remain read-only.
    ufsecp_error_t collect_ecdsa_columns(const uint8_t* msg_hashes32, const uint8_t* pubkeys33,
                               const uint8_t* sigs64, size_t count,
                               uint8_t* key_cells, size_t key_size,
                               const ufsecp_cancel_token* cancel = nullptr) const {
        return ufsecp_lbtc_verify_ecdsa_columns_collect(
            ctrl_, msg_hashes32, pubkeys33, sigs64, count, key_cells, key_size, cancel);
    }
    ufsecp_error_t collect_schnorr_columns(const uint8_t* msg_hashes32, const uint8_t* pubkeys_x32,
                                 const uint8_t* sigs64, size_t count,
                                 uint8_t* key_cells, size_t key_size,
                                 const ufsecp_cancel_token* cancel = nullptr) const {
        return ufsecp_lbtc_verify_schnorr_columns_collect(
            ctrl_, msg_hashes32, pubkeys_x32, sigs64, count, key_cells, key_size, cancel);
    }

    // Collect twins for the multisig / threshold tables (see verify_multisig).
    // key_size MUST be > 0 (the whole tail is the verdict cell: zeroed on valid,
    // intact on invalid). For the canonical rows that is 6 (m|n + group + block-fk).
    ufsecp_error_t collect_multisig(uint8_t* rows, size_t count, size_t key_size,
                                    const ufsecp_cancel_token* cancel = nullptr) const {
        return ufsecp_lbtc_verify_ecdsa_collect(ctrl_, rows, count, key_size, cancel);
    }
    ufsecp_error_t collect_threshold(uint8_t* rows, size_t count, size_t key_size,
                                     const ufsecp_cancel_token* cancel = nullptr) const {
        return ufsecp_lbtc_verify_schnorr_collect(ctrl_, rows, count, key_size, cancel);
    }

    // --- BIP-341 Taproot commitment batch (x-only tweak-add-check) -----------
    // Columns/SoA: internal_x[n*32], tweak[n*32], tweaked_x[n*32], parity[n].
    // results[i] == 1 if lift_x(internal)+tweak*G matches (tweaked_x, parity).
    void verify_commitment(const uint8_t* internal_x32, const uint8_t* tweak32,
                           const uint8_t* tweaked_x32, const uint8_t* parity,
                           size_t n, uint8_t* results) const {
        ufsecp_lbtc_verify_commitment(ctrl_, internal_x32, tweak32, tweaked_x32,
                                      parity, n, results);
    }
    // GPU RLC aggregate fast-check: 1 all-valid / 0 some-invalid / -1 no GPU.
    int commitment_batch_ok(const uint8_t* internal_x32, const uint8_t* tweak32,
                            const uint8_t* tweaked_x32, const uint8_t* parity,
                            size_t n) const {
        return ufsecp_lbtc_commitment_batch_ok(ctrl_, internal_x32, tweak32,
                                               tweaked_x32, parity, n);
    }

    // Single-buffer (AoS) commitment batch — one mmap'd table, one pointer.
    // Each row: internal_x[32] | tweak[32] | tweaked[33, compressed]. stride >= 97.
    void verify_commitment_rows(const uint8_t* rows, size_t n, size_t stride,
                                uint8_t* results) const {
        ufsecp_lbtc_verify_commitment_rows(ctrl_, rows, n, stride, results);
    }
    int commitment_batch_ok_rows(const uint8_t* rows, size_t n, size_t stride) const {
        return ufsecp_lbtc_commitment_batch_ok_rows(ctrl_, rows, n, stride);
    }

    // Batch x-only pubkey validation (lift_x on-curve check). keys: n × stride
    // (32-byte x first; stride >= 32). results[i] == 1 if a valid x-only pubkey.
    void validate_xonly(const uint8_t* keys, size_t n, size_t stride,
                        uint8_t* results) const {
        ufsecp_lbtc_validate_xonly(ctrl_, keys, n, stride, results);
    }

    // Batch BIP-340 tagged hash (Taproot leaf/branch). TapBranch: tag="TapBranch",
    // msg_len=64 (children pre-ordered min||max). out32 = n*32.
    void tagged_hash_batch(const char* tag, const uint8_t* msgs, size_t msg_len,
                           size_t n, size_t stride, uint8_t* out32) const {
        ufsecp_lbtc_tagged_hash_batch(ctrl_, tag, msgs, msg_len, n, stride, out32);
    }

    // Batch full compressed-pubkey validation (prefix 0x02/0x03 + x<p + on-curve).
    // keys: n × stride (33-byte pubkey first; stride >= 33).
    void validate_pubkeys(const uint8_t* keys, size_t n, size_t stride,
                          uint8_t* results) const {
        ufsecp_lbtc_validate_pubkeys(ctrl_, keys, n, stride, results);
    }
    // Batch Taproot tagged hash with per-item length (TapLeaf). lens[i] in 1..256.
    void tagged_hash_var(const char* tag, const uint8_t* msgs, const uint32_t* lens,
                         size_t stride, size_t n, uint8_t* out32) const {
        ufsecp_lbtc_tagged_hash_var(ctrl_, tag, msgs, lens, stride, n, out32);
    }
    // Batch HASH256 (double SHA-256) of fixed-length inputs (merkle-tree nodes).
    void hash256(const uint8_t* inputs, size_t input_len, size_t n,
                 uint8_t* out32) const {
        ufsecp_lbtc_hash256(ctrl_, inputs, input_len, n, out32);
    }
    // Aggregate BIP-340 Schnorr batch verify (RLC). 1 all-valid / 0 some-bad / -1 no-GPU.
    int schnorr_aggregate_verify(const uint8_t* msgs32, const uint8_t* pubkeys_x32,
                                 const uint8_t* sigs64, size_t n) const {
        return ufsecp_lbtc_schnorr_aggregate_verify(ctrl_, msgs32, pubkeys_x32, sigs64, n);
    }
#if __cplusplus >= 202002L
    // Typed-span over the canonical CommitmentRow (stride = sizeof(Row) recovers
    // any trailing correlation tail). Zero-copy from your packed table.
    template <class Row>
    void verify_commitment(std::span<const Row> batch, uint8_t* results) const {
        static_assert(sizeof(Row) >= UFSECP_LBTC_COMMITMENT_RECORD,
                      "Row must contain the 97-byte commitment record first");
        static_assert(std::is_standard_layout_v<Row>,
                      "Row must be standard-layout, tightly packed (#pragma pack(1))");
        ufsecp_lbtc_verify_commitment_rows(
            ctrl_, reinterpret_cast<const uint8_t*>(batch.data()), batch.size(),
            sizeof(Row), results);
    }
    template <class Row>
    int commitment_batch_ok(std::span<const Row> batch) const {
        static_assert(sizeof(Row) >= UFSECP_LBTC_COMMITMENT_RECORD,
                      "Row must contain the 97-byte commitment record first");
        return ufsecp_lbtc_commitment_batch_ok_rows(
            ctrl_, reinterpret_cast<const uint8_t*>(batch.data()), batch.size(),
            sizeof(Row));
    }
    // Typed-span x-only validation: stride = sizeof(Key) (e.g. std::array<uint8_t,32>,
    // or a wider packed row whose 32-byte x is first). results[i] == 1 if valid.
    template <class Key>
    void validate_xonly(std::span<const Key> keys, uint8_t* results) const {
        static_assert(sizeof(Key) >= 32, "Key must hold the 32-byte x-coordinate first");
        validate_xonly(reinterpret_cast<const uint8_t*>(keys.data()), keys.size(),
                       sizeof(Key), results);
    }
    // Typed-span tagged hash: msg_len = stride = sizeof(Msg) (e.g. std::array<uint8_t,64>
    // for TapBranch min||max). out[i] = tagged_hash(tag, msg_i).
    template <class Msg>
    void tagged_hash_batch(const char* tag, std::span<const Msg> msgs,
                           std::array<uint8_t, 32>* out) const {
        tagged_hash_batch(tag, reinterpret_cast<const uint8_t*>(msgs.data()),
                          sizeof(Msg), msgs.size(), sizeof(Msg),
                          reinterpret_cast<uint8_t*>(out));
    }
#endif
    ufsecp_error_t collect_multisig_columns(const uint8_t* msg_hashes32, const uint8_t* pubkeys33,
                                            const uint8_t* sigs64, size_t count,
                                            uint8_t* key_cells, size_t key_size,
                                            const ufsecp_cancel_token* cancel = nullptr) const {
        return collect_ecdsa_columns(msg_hashes32, pubkeys33, sigs64, count,
                                     key_cells, key_size, cancel);
    }
    ufsecp_error_t collect_threshold_columns(const uint8_t* msg_hashes32,
                                             const uint8_t* pubkeys_x32,
                                             const uint8_t* sigs64, size_t count,
                                             uint8_t* key_cells, size_t key_size,
                                             const ufsecp_cancel_token* cancel = nullptr) const {
        return collect_schnorr_columns(msg_hashes32, pubkeys_x32, sigs64, count,
                                       key_cells, key_size, cancel);
    }

#if __cplusplus >= 202002L
    // --- Typed-span overloads (C++20) ---------------------------------------
    // Pass the packed record span directly; NOTHING about its size is restated at
    // the call site. The element type IS the layout, so count = batch.size() and
    // key_size = sizeof(Row) - RECORD (the trailing correlation id) are both
    // recovered from the type. Equivalent to computing key_size yourself and
    // calling the C ABI, e.g. evoskuil's
    //     constexpr auto id_size = array_count<decltype(triple::identifier)>; // 3
    //     ufsecp_lbtc_verify_ecdsa(ctx, in, count, id_size, out);
    //
    //     std::span<const ecdsa::triple> batch = ...;
    //     std::vector<uint8_t> results(batch.size());
    //     ctrl.verify_ecdsa(batch, results.data());
    //
    // CONTRACT — `Row` MUST be a tightly-packed (#pragma pack(1)) standard-layout
    // struct whose FIRST RECORD bytes are the on-wire record in order (hash, key,
    // sig) with NO leading or internal padding; any trailing bytes (including zero)
    // are the opaque correlation key. key_size = sizeof(Row) - RECORD, so padding
    // folded into sizeof(Row) before/within the record would silently misread —
    // hence the standard-layout assert and the packing requirement. evoskuil's
    // `secp256k1::ecdsa::triple` satisfies this: it is #pragma pack(1) over
    // byte-array members, so the record is the first 129 bytes exactly.
    template <class Row>
    ufsecp_error_t verify_ecdsa(std::span<const Row> batch, uint8_t* results,
                      const ufsecp_cancel_token* cancel = nullptr) const {
        static_assert(sizeof(Row) >= UFSECP_LBTC_ECDSA_RECORD,
                      "Row must contain the 129-byte ECDSA record (hash|point|sig) first");
        static_assert(std::is_standard_layout_v<Row>,
                      "Row must be a standard-layout, tightly-packed (#pragma pack(1)) "
                      "struct so the first 129 bytes are the contiguous on-wire record");
        return ufsecp_lbtc_verify_ecdsa(
            ctrl_, reinterpret_cast<const uint8_t*>(batch.data()), batch.size(),
            sizeof(Row) - UFSECP_LBTC_ECDSA_RECORD, results, nullptr, 0, nullptr, cancel);
    }
    template <class Row>
    ufsecp_error_t verify_schnorr(std::span<const Row> batch, uint8_t* results,
                        const ufsecp_cancel_token* cancel = nullptr) const {
        static_assert(sizeof(Row) >= UFSECP_LBTC_SCHNORR_RECORD,
                      "Row must contain the 128-byte Schnorr record (hash|xonly|sig) first");
        static_assert(std::is_standard_layout_v<Row>,
                      "Row must be a standard-layout, tightly-packed (#pragma pack(1)) "
                      "struct so the first 128 bytes are the contiguous on-wire record");
        return ufsecp_lbtc_verify_schnorr(
            ctrl_, reinterpret_cast<const uint8_t*>(batch.data()), batch.size(),
            sizeof(Row) - UFSECP_LBTC_SCHNORR_RECORD, results, nullptr, 0, nullptr, cancel);
    }

    // --- Typed-span COLLECT overloads (C++20) -------------------------------
    // The in-place collect path writes each row's key cell, so the span is over
    // NON-const Row (mutable) — the type system enforces caller writability at
    // the C++ boundary (this is exactly why the span must not be const). count =
    // batch.size(); key_size = sizeof(Row) - RECORD, which MUST be > 0 (the row
    // must carry a non-empty key cell to hold the verdict — enforced by a
    // static_assert). After the call, scan `batch` for any Row whose key cell is
    // non-zero: those are the rejected rows (their correlation id survived).
    //
    //     std::span<ecdsa::triple> batch = ...;   // MUTABLE
    //     ctrl.collect_ecdsa(batch);
    //     for (auto& row : batch) if (key_nonzero(row)) reject(row);
    template <class Row>
    ufsecp_error_t collect_ecdsa(std::span<Row> batch,
                                const ufsecp_cancel_token* cancel = nullptr) const {
        static_assert(sizeof(Row) > UFSECP_LBTC_ECDSA_RECORD,
                      "collect requires a non-empty key cell: sizeof(Row) must "
                      "exceed the 129-byte ECDSA record");
        static_assert(std::is_standard_layout_v<Row>,
                      "Row must be a standard-layout, tightly-packed (#pragma pack(1)) "
                      "struct so the first 129 bytes are the contiguous on-wire record");
        static_assert(!std::is_const_v<Row>,
                      "collect writes the key cell in place — the span must be over "
                      "non-const Row (mutable rows)");
        return ufsecp_lbtc_verify_ecdsa_collect(
            ctrl_, reinterpret_cast<uint8_t*>(batch.data()), batch.size(),
            sizeof(Row) - UFSECP_LBTC_ECDSA_RECORD, cancel);
    }
    template <class Row>
    ufsecp_error_t collect_schnorr(std::span<Row> batch,
                                  const ufsecp_cancel_token* cancel = nullptr) const {
        static_assert(sizeof(Row) > UFSECP_LBTC_SCHNORR_RECORD,
                      "collect requires a non-empty key cell: sizeof(Row) must "
                      "exceed the 128-byte Schnorr record");
        static_assert(std::is_standard_layout_v<Row>,
                      "Row must be a standard-layout, tightly-packed (#pragma pack(1)) "
                      "struct so the first 128 bytes are the contiguous on-wire record");
        static_assert(!std::is_const_v<Row>,
                      "collect writes the key cell in place — the span must be over "
                      "non-const Row (mutable rows)");
        return ufsecp_lbtc_verify_schnorr_collect(
            ctrl_, reinterpret_cast<uint8_t*>(batch.data()), batch.size(),
            sizeof(Row) - UFSECP_LBTC_SCHNORR_RECORD, cancel);
    }

    // --- Typed-span multisig / threshold overloads (C++20) ------------------
    // Pass the canonical MultisigRow / ThresholdRow span (or any packed Row whose
    // first RECORD bytes are the ECDSA / Schnorr record). key_size = sizeof(Row) -
    // RECORD is recovered from the type (6 for the canonical rows). These forward
    // to the ECDSA / Schnorr verify+collect cores — same crypto, named for intent.
    template <class Row>
    ufsecp_error_t verify_multisig(std::span<const Row> batch, uint8_t* results,
                         const ufsecp_cancel_token* cancel = nullptr) const {
        return verify_ecdsa<Row>(batch, results, cancel);
    }
    template <class Row>
    ufsecp_error_t verify_threshold(std::span<const Row> batch, uint8_t* results,
                          const ufsecp_cancel_token* cancel = nullptr) const {
        return verify_schnorr<Row>(batch, results, cancel);
    }
    template <class Row>
    ufsecp_error_t collect_multisig(std::span<Row> batch,
                                  const ufsecp_cancel_token* cancel = nullptr) const {
        return collect_ecdsa<Row>(batch, cancel);
    }
    template <class Row>
    ufsecp_error_t collect_threshold(std::span<Row> batch,
                                   const ufsecp_cancel_token* cancel = nullptr) const {
        return collect_schnorr<Row>(batch, cancel);
    }
#endif /* C++20 std::span */

private:
    ufsecp_lbtc_ctrl* ctrl_ = nullptr;
};

} // namespace lbtc
} // namespace ufsecp
#endif /* __cplusplus */

#endif /* UFSECP_LIBBITCOIN_H */
