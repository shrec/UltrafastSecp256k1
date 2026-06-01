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
 *   - Results are returned directly: a per-row pass/fail array AND, optionally,
 *     a compact list of the failing row indices.
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
#include "ufsecp_error.h" /* ufsecp_error_t, UFSECP_OK, UFSECP_ERR_* */

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------- */
/* Record layouts (signature payload only — excludes any opaque key column).  */
/* ------------------------------------------------------------------------- */

/* Uniform field order across both kinds — hash/msg first, then pubkey, then
 * signature (matches the libbitcoin ecdsa::triple / schnorr::triple structs,
 * so a packed struct array forwards into these calls with zero copy). */
/* ECDSA row:   32-byte msghash      | 33-byte compressed pubkey | 64-byte sig. */
#define UFSECP_LBTC_ECDSA_RECORD   129u
/* Schnorr row: 32-byte msg/sighash  | 32-byte x-only pubkey     | 64-byte sig. */
#define UFSECP_LBTC_SCHNORR_RECORD 128u

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
 *   rows          n rows, each (RECORD + key_size) bytes, tightly packed.
 *                 The signature record occupies the first RECORD bytes; the
 *                 trailing key_size bytes are the caller's opaque tag.
 *   n             number of rows (the record COUNT). With key_size this fully
 *                 determines the buffer: n * (RECORD + key_size) bytes. No
 *                 separate buffer-size argument is taken or needed.
 *   key_size      opaque trailing bytes per row; 0 to disable.
 *   results       OUT, optional. If non-NULL, must be n bytes: results[i] is
 *                 1 if row i is valid, 0 if invalid.
 *   invalid_idx   OUT, optional. If non-NULL, receives the indices of failing
 *                 rows, up to invalid_cap entries.
 *   invalid_cap   capacity of invalid_idx (entries); ignored if invalid_idx is
 *                 NULL.
 *   invalid_count OUT, optional. Set to the TOTAL number of failing rows (which
 *                 may exceed invalid_cap — the caller can detect truncation).
 *
 * Return value:
 *   UFSECP_OK              the batch was processed (inspect results / counts);
 *                          all rows valid iff *invalid_count == 0.
 *   UFSECP_ERR_BAD_INPUT   reserved for arithmetic overflow of n * stride. There
 *                          is NO buffer-size / stride-mismatch error: the buffer
 *                          is implied by (count, key_size), so it is always
 *                          self-consistent by construction.
 *   UFSECP_ERR_BAD_PUBKEY  a row's pubkey is not a valid curve point.
 *   UFSECP_ERR_BAD_SIG     a row's signature is structurally invalid (e.g.
 *                          s >= n, R.x >= p). Such a row counts as invalid.
 *   UFSECP_ERR_NULL_ARG    ctrl or rows is NULL with n > 0.
 *
 * A NULL `rows` with n == 0 is the empty batch: vacuously valid, returns
 * UFSECP_OK with *invalid_count == 0.
 */
ufsecp_error_t ufsecp_lbtc_verify_ecdsa(ufsecp_lbtc_ctrl* ctrl,
                                        const uint8_t* rows, size_t n,
                                        size_t key_size,
                                        uint8_t* results,
                                        size_t* invalid_idx, size_t invalid_cap,
                                        size_t* invalid_count);

ufsecp_error_t ufsecp_lbtc_verify_schnorr(ufsecp_lbtc_ctrl* ctrl,
                                          const uint8_t* rows, size_t n,
                                          size_t key_size,
                                          uint8_t* results,
                                          size_t* invalid_idx, size_t invalid_cap,
                                          size_t* invalid_count);

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
ufsecp_error_t ufsecp_lbtc_sp_scan(ufsecp_lbtc_ctrl* ctrl,
                                   const uint8_t scan_privkey32[32],
                                   const uint8_t spend_pubkey33[33],
                                   const uint8_t* tweak_pubkeys33, size_t n,
                                   uint64_t* prefix64_out);

#ifdef __cplusplus
} /* extern "C" */

#if __cplusplus >= 202002L
#  include <span>
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
    uint8_t sig[64];     // compact signature: r || s
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
    // Pass the record COUNT and the KEY SIZE (key_size == 0 == no correlation key).
    // The row pointer has NO size argument: the buffer is fully determined by the
    // two values — exactly count * (RECORD + key_size) bytes — so it can never
    // mismatch and there is no size-related error condition. (Per evoskuil: the
    // buffer size is redundant with count + key_size; passing it would only add an
    // unnecessary error path.)
    ufsecp_error_t verify_ecdsa(const uint8_t* rows, size_t count, size_t key_size,
                                uint8_t* results = nullptr, size_t* invalid_idx = nullptr,
                                size_t invalid_cap = 0, size_t* invalid_count = nullptr) const {
        return ufsecp_lbtc_verify_ecdsa(ctrl_, rows, count, key_size, results,
                                        invalid_idx, invalid_cap, invalid_count);
    }
    ufsecp_error_t verify_schnorr(const uint8_t* rows, size_t count, size_t key_size,
                                  uint8_t* results = nullptr, size_t* invalid_idx = nullptr,
                                  size_t invalid_cap = 0, size_t* invalid_count = nullptr) const {
        return ufsecp_lbtc_verify_schnorr(ctrl_, rows, count, key_size, results,
                                          invalid_idx, invalid_cap, invalid_count);
    }

#if __cplusplus >= 202002L
    // --- Typed-span overloads (C++20) ---------------------------------------
    // Pass the packed record span directly; NOTHING about its size is restated at
    // the call site. The element type IS the layout, so:
    //     count    = batch.size()
    //     key_size = sizeof(Row) - RECORD            (the trailing correlation id)
    // are both recovered from the type — no count, no key_size, no buffer size to
    // pass or mismatch. This is the form a node uses with its own packed struct,
    // e.g. libbitcoin's `secp256k1::ecdsa::triple` (a #pragma pack(1) struct of
    // { hash_digest, ec_compressed, ec_signature, token } == 129 + sizeof(token)):
    //
    //     std::span<const ecdsa::triple> batch = ...;
    //     std::vector<uint8_t> results(batch.size());
    //     size_t fails = 0;
    //     ctrl.verify_ecdsa(batch, results.data(), nullptr, 0, &fails);
    //
    // CONTRACT — `Row` MUST be a tightly-packed (#pragma pack(1)) standard-layout
    // struct whose FIRST RECORD bytes are the on-wire record in order (hash, key,
    // sig) with NO leading or internal padding; any trailing bytes (including zero)
    // are the opaque correlation key. key_size = sizeof(Row) - RECORD, so any
    // padding folded into sizeof(Row) before the record, or between the record
    // fields, would silently misread — hence the standard-layout assert and the
    // packing requirement. evoskuil's `secp256k1::ecdsa::triple` satisfies this:
    // it is #pragma pack(1) over byte-array members (hash_digest, ec_compressed,
    // ec_signature, data_array<3>), so the record is the first 129 bytes exactly.
    template <class Row>
    ufsecp_error_t verify_ecdsa(std::span<const Row> batch,
                                uint8_t* results = nullptr, size_t* invalid_idx = nullptr,
                                size_t invalid_cap = 0, size_t* invalid_count = nullptr) const {
        static_assert(sizeof(Row) >= UFSECP_LBTC_ECDSA_RECORD,
                      "Row must contain the 129-byte ECDSA record (hash|point|sig) first");
        static_assert(std::is_standard_layout_v<Row>,
                      "Row must be a standard-layout, tightly-packed (#pragma pack(1)) "
                      "struct so the first 129 bytes are the contiguous on-wire record");
        return ufsecp_lbtc_verify_ecdsa(
            ctrl_, reinterpret_cast<const uint8_t*>(batch.data()), batch.size(),
            sizeof(Row) - UFSECP_LBTC_ECDSA_RECORD, results,
            invalid_idx, invalid_cap, invalid_count);
    }
    template <class Row>
    ufsecp_error_t verify_schnorr(std::span<const Row> batch,
                                  uint8_t* results = nullptr, size_t* invalid_idx = nullptr,
                                  size_t invalid_cap = 0, size_t* invalid_count = nullptr) const {
        static_assert(sizeof(Row) >= UFSECP_LBTC_SCHNORR_RECORD,
                      "Row must contain the 128-byte Schnorr record (hash|xonly|sig) first");
        static_assert(std::is_standard_layout_v<Row>,
                      "Row must be a standard-layout, tightly-packed (#pragma pack(1)) "
                      "struct so the first 128 bytes are the contiguous on-wire record");
        return ufsecp_lbtc_verify_schnorr(
            ctrl_, reinterpret_cast<const uint8_t*>(batch.data()), batch.size(),
            sizeof(Row) - UFSECP_LBTC_SCHNORR_RECORD, results,
            invalid_idx, invalid_cap, invalid_count);
    }
#endif /* C++20 std::span */

private:
    ufsecp_lbtc_ctrl* ctrl_ = nullptr;
};

} // namespace lbtc
} // namespace ufsecp
#endif /* __cplusplus */

#endif /* UFSECP_LIBBITCOIN_H */
