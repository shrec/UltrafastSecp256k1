// ============================================================================
// shim_batch_verify.cpp -- Batch ECDSA and Schnorr verification
// ============================================================================
// Exposes secp256k1_schnorrsig_verify_batch() / secp256k1_ecdsa_verify_batch()
// (and the _mt / _results thread-controlled + per-row variants) as
// libsecp256k1-compatible shim functions backed by UltrafastSecp256k1's
// multi-scalar multiplication (Pippenger) plus the engine's first-class
// multi-threaded batch verify. These are ADDITIVE symbols -- the existing
// secp256k1_schnorrsig_verify / secp256k1_ecdsa_verify are unchanged.
//
// Performance vs n individual verifications:
//   Schnorr batch: ~2-3x faster (MSM); MT scales further across CPU cores
//   ECDSA batch:   ~1.5-2x faster (per-sig modular inverse); MT scales further
//
// Threading model (the single standard surface -- no bespoke bridge needed):
//   max_threads = 0 -> auto (hardware_concurrency)
//   max_threads = 1 -> serial (use this when calling from your OWN pool)
//   max_threads = N -> up to N (reduced only to what the hardware can run)
// The boolean ("all valid") result is identical for any thread count;
// verification is variable-time over PUBLIC data only -> zero CT impact.
//
// No-failure contract: never throw across the C ABI. If internal thread
// creation fails, fall back to serial verification; the result is deterministic
// and identical to the serial path.
//
// API design:
//   n = 0: returns 1 (vacuously valid)
//   n < 8: falls back to individual verification (batch overhead > benefit)
//   Any invalid input pointer (all-or-nothing path): returns 0 (fail-closed)
//   results path: malformed/NULL row -> results[i]=0, other rows still verified
// ============================================================================

#include "secp256k1_batch.h"
#include "shim_internal.hpp"
#include "shim_pubkey_helpers.hpp"

#include <cstring>
#include <array>
#include <vector>
#include <cstdint>
#include <algorithm>

#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/batch_verify.hpp"

using namespace secp256k1::fast;

// Minimum batch size below which individual verify is cheaper.
static constexpr size_t kBatchMinSchnorr = 8;
static constexpr size_t kBatchMinEcdsa   = 8;

// Context helpers from shim_internal.hpp
using secp256k1_shim_internal::ctx_can_verify;

namespace {

// -- Single-signature verify helpers (shared by small-n and results paths) ----

bool ecdsa_verify_one(const secp256k1_ecdsa_signature* sig,
                      const unsigned char*             msg32,
                      const secp256k1_pubkey*          pubkey)
{
    if (!sig || !msg32 || !pubkey) return false;
    Scalar r, s;
    // Opaque secp256k1_ecdsa_signature.data is the engine's native little-endian
    // limb form (see shim_ecdsa.cpp ecdsa_sig_from_data), NOT big-endian compact.
    if (!Scalar::parse_bytes_strict_le(sig->data,      r)) return false;
    if (!Scalar::parse_bytes_strict_le(sig->data + 32, s)) return false;
    // No is_low_s() check: batch verify accepts high-S to match single
    // secp256k1_ecdsa_verify behaviour (SHIM-008).
    secp256k1::ECDSASignature isig{r, s};
    // Trust contract: pubkey->data was validated at ec_pubkey_parse time.
    using secp256k1_shim_internal::pubkey_data_to_point;
    auto pt = pubkey_data_to_point(pubkey->data);
    std::array<uint8_t, 32> msg{};
    std::memcpy(msg.data(), msg32, 32);
    return secp256k1::ecdsa_verify(msg, pt, isig);
}

bool schnorr_verify_one(const unsigned char*          sig64,
                        const unsigned char*          msg,
                        size_t                        msglen,
                        const secp256k1_xonly_pubkey* pubkey)
{
    if (!sig64 || !msg || !pubkey) return false;
    secp256k1::SchnorrSignature sig;
    if (!secp256k1::SchnorrSignature::parse_strict(sig64, sig)) return false;
    if (msglen != 32) {
        return secp256k1::schnorr_verify(pubkey->data, msg, msglen, sig);
    }
    // PERF-007: use Y-stored Point overload when the parse cached an even-Y,
    // avoiding a lift_x sqrt per call (xonly_pubkey.data[0..31]=X, [32..63]=Y).
    const uint8_t* xb = pubkey->data;
    const uint8_t* yb = pubkey->data + 32;
    secp256k1::fast::FieldElement x_fe, y_fe;
    if (!secp256k1::fast::FieldElement::parse_bytes_strict(xb, x_fe)) return false;
    if (secp256k1::fast::FieldElement::parse_bytes_strict(yb, y_fe)) {
        auto P = secp256k1::fast::Point::from_affine(x_fe, y_fe);
        return secp256k1::schnorr_verify(P, xb, msg, sig);
    }
    return secp256k1::schnorr_verify(xb, msg, sig);
}

// Dispatch a fully-marshalled ECDSA batch with the no-failure contract:
// try MT, fall back to serial if thread creation throws.
bool ecdsa_dispatch(const secp256k1::ECDSABatchEntry* b, size_t n, size_t max_threads) {
    try {
        return secp256k1::ecdsa_batch_verify_mt(b, n, max_threads);
    } catch (...) {
        return secp256k1::ecdsa_batch_verify(b, n);  // serial, no thread spawn
    }
}

bool schnorr_dispatch(const secp256k1::SchnorrBatchEntry* b, size_t n, size_t max_threads) {
    try {
        return secp256k1::schnorr_batch_verify_mt(b, n, max_threads);
    } catch (...) {
        return secp256k1::schnorr_batch_verify(b, n);  // serial, no thread spawn
    }
}

// -- External cancellation (verify path = public data only -> no CT impact) ----
// A NULL token is the hot path (handled entirely by the *_batch_core null branch,
// untouched). A non-NULL token is polled BETWEEN chunks on the caller's thread,
// so there is no concurrent callback invocation. Default chunk size matches the
// libbitcoin bridge (262144); a smaller check_interval trades MSM batching for
// cancellation latency and is clamped up to the batch minimum.

inline bool cancel_requested(const ufsecp_cancel_token* cancel) noexcept {
    if (!cancel || !cancel->is_cancelled) return false;
    try {
        return cancel->is_cancelled(cancel->user) != 0;
    } catch (...) {
        return true;  // fail-closed: a throwing callback cancels the batch
    }
}

static constexpr size_t kCancelChunkDefault = size_t{1} << 18;  // 262144 (matches bridge kChunk)

inline size_t cancel_chunk_size(const ufsecp_cancel_token* cancel) noexcept {
    if (!cancel || cancel->check_interval == 0) return kCancelChunkDefault;
    size_t req = static_cast<size_t>(cancel->check_interval);
    if (req < kBatchMinEcdsa) req = kBatchMinEcdsa;  // keep each chunk worth batching
    return req < kCancelChunkDefault ? req : kCancelChunkDefault;
}

// Cancellable ECDSA core (cancel != nullptr; ctx/n/array guards already done by
// the caller). Chunked so the batch can be aborted from outside; fail-closed.
int ecdsa_batch_core_cancel(
    const secp256k1_ecdsa_signature* const* sigs,
    const unsigned char* const*             msgs32,
    const secp256k1_pubkey* const*          pubkeys,
    size_t                                  n,
    size_t                                  max_threads,
    int*                                    results,
    const ufsecp_cancel_token*              cancel)
{
    using secp256k1_shim_internal::pubkey_data_to_point;
    const size_t csz = cancel_chunk_size(cancel);

    if (results) {
        for (size_t i = 0; i < n; ++i) results[i] = 0;  // pre-zero: unreached rows stay 0
        int all_valid = 1;
        static thread_local std::vector<secp256k1::ECDSABatchEntry> rbatch;
        static thread_local std::vector<size_t>                     ridx;
        for (size_t base = 0; base < n; base += csz) {
            if (cancel_requested(cancel)) return 0;
            const size_t cn = std::min(csz, n - base);
            rbatch.clear(); ridx.clear();
            rbatch.reserve(cn); ridx.reserve(cn);
            for (size_t i = base; i < base + cn; ++i) {
                if (!sigs[i] || !msgs32[i] || !pubkeys[i]) { all_valid = 0; continue; }
                Scalar r, s;
                if (!Scalar::parse_bytes_strict_le(sigs[i]->data,      r)) { all_valid = 0; continue; }
                if (!Scalar::parse_bytes_strict_le(sigs[i]->data + 32, s)) { all_valid = 0; continue; }
                secp256k1::ECDSABatchEntry e{};
                std::memcpy(e.msg_hash.data(), msgs32[i], 32);
                e.signature  = secp256k1::ECDSASignature{r, s};
                e.public_key = pubkey_data_to_point(pubkeys[i]->data);
                rbatch.push_back(e); ridx.push_back(i);
            }
            const bool chunk_all_parsed = (rbatch.size() == cn);
            if (chunk_all_parsed && ecdsa_dispatch(rbatch.data(), rbatch.size(), max_threads)) {
                for (size_t i = base; i < base + cn; ++i) results[i] = 1;
                continue;
            }
            std::vector<size_t> invalid;
            secp256k1::ecdsa_batch_identify_invalid(rbatch.data(), rbatch.size(), invalid);
            for (size_t j = 0; j < ridx.size(); ++j) results[ridx[j]] = 1;
            for (size_t inv : invalid) results[ridx[inv]] = 0;
            if (!(chunk_all_parsed && invalid.empty())) all_valid = 0;
        }
        return all_valid;
    }

    // All-or-nothing: any malformed row or invalid chunk -> 0 (fail-closed).
    static thread_local std::vector<secp256k1::ECDSABatchEntry> obatch;
    for (size_t base = 0; base < n; base += csz) {
        if (cancel_requested(cancel)) return 0;
        const size_t cn = std::min(csz, n - base);
        obatch.clear(); obatch.reserve(cn);
        for (size_t i = base; i < base + cn; ++i) {
            if (!sigs[i] || !msgs32[i] || !pubkeys[i]) return 0;
            secp256k1::ECDSABatchEntry e{};
            std::memcpy(e.msg_hash.data(), msgs32[i], 32);
            Scalar r, s;
            if (!Scalar::parse_bytes_strict_le(sigs[i]->data,      r)) return 0;
            if (!Scalar::parse_bytes_strict_le(sigs[i]->data + 32, s)) return 0;
            e.signature  = secp256k1::ECDSASignature{r, s};
            e.public_key = pubkey_data_to_point(pubkeys[i]->data);
            obatch.push_back(e);
        }
        if (!ecdsa_dispatch(obatch.data(), obatch.size(), max_threads)) return 0;
    }
    return 1;
}

// Cancellable Schnorr core (cancel != nullptr; caller did the guards).
int schnorr_batch_core_cancel(
    const unsigned char* const*           sigs64,
    const unsigned char* const*           msgs,
    size_t                                msglen,
    const secp256k1_xonly_pubkey* const*  pubkeys,
    size_t                                n,
    size_t                                max_threads,
    int*                                  results,
    const ufsecp_cancel_token*            cancel)
{
    const size_t csz = cancel_chunk_size(cancel);

    if (results) {
        for (size_t i = 0; i < n; ++i) results[i] = 0;  // pre-zero
        if (msglen != 32) {  // variable-length: per-row verify with periodic polls
            int all = 1;
            for (size_t i = 0; i < n; ++i) {
                if ((i % csz) == 0 && cancel_requested(cancel)) return 0;
                results[i] = schnorr_verify_one(sigs64[i], msgs[i], msglen, pubkeys[i]) ? 1 : 0;
                if (!results[i]) all = 0;
            }
            return all;
        }
        int all_valid = 1;
        static thread_local std::vector<secp256k1::SchnorrBatchEntry> rbatch;
        static thread_local std::vector<size_t>                       ridx;
        for (size_t base = 0; base < n; base += csz) {
            if (cancel_requested(cancel)) return 0;
            const size_t cn = std::min(csz, n - base);
            rbatch.clear(); ridx.clear();
            rbatch.reserve(cn); ridx.reserve(cn);
            for (size_t i = base; i < base + cn; ++i) {
                if (!sigs64[i] || !msgs[i] || !pubkeys[i]) { all_valid = 0; continue; }
                secp256k1::SchnorrBatchEntry e{};
                std::memcpy(e.pubkey_x.data(), pubkeys[i]->data, 32);
                std::memcpy(e.message.data(),  msgs[i],          32);
                std::array<uint8_t, 64> sb{};
                std::memcpy(sb.data(), sigs64[i], 64);
                if (!secp256k1::SchnorrSignature::parse_strict(sb, e.signature)) { all_valid = 0; continue; }
                rbatch.push_back(e); ridx.push_back(i);
            }
            const bool chunk_all_parsed = (rbatch.size() == cn);
            if (chunk_all_parsed && schnorr_dispatch(rbatch.data(), rbatch.size(), max_threads)) {
                for (size_t i = base; i < base + cn; ++i) results[i] = 1;
                continue;
            }
            std::vector<size_t> invalid;
            secp256k1::schnorr_batch_identify_invalid(rbatch.data(), rbatch.size(), invalid);
            for (size_t j = 0; j < ridx.size(); ++j) results[ridx[j]] = 1;
            for (size_t inv : invalid) results[ridx[inv]] = 0;
            if (!(chunk_all_parsed && invalid.empty())) all_valid = 0;
        }
        return all_valid;
    }

    // All-or-nothing.
    if (msglen != 32) {
        for (size_t i = 0; i < n; ++i) {
            if ((i % csz) == 0 && cancel_requested(cancel)) return 0;
            if (!schnorr_verify_one(sigs64[i], msgs[i], msglen, pubkeys[i])) return 0;
        }
        return 1;
    }
    static thread_local std::vector<secp256k1::SchnorrBatchEntry> obatch;
    for (size_t base = 0; base < n; base += csz) {
        if (cancel_requested(cancel)) return 0;
        const size_t cn = std::min(csz, n - base);
        obatch.clear(); obatch.reserve(cn);
        for (size_t i = base; i < base + cn; ++i) {
            if (!sigs64[i] || !msgs[i] || !pubkeys[i]) return 0;
            secp256k1::SchnorrBatchEntry e{};
            std::memcpy(e.pubkey_x.data(), pubkeys[i]->data, 32);
            std::memcpy(e.message.data(),  msgs[i],          32);
            std::array<uint8_t, 64> sb{};
            std::memcpy(sb.data(), sigs64[i], 64);
            if (!secp256k1::SchnorrSignature::parse_strict(sb, e.signature)) return 0;
            obatch.push_back(e);
        }
        if (!schnorr_dispatch(obatch.data(), obatch.size(), max_threads)) return 0;
    }
    return 1;
}

// ---------------------------------------------------------------------------
// ECDSA batch core. results == nullptr => all-or-nothing (fail-closed on any
// bad pointer); results != nullptr => per-row verdict (each results[i] set).
// A non-NULL `cancel` routes to the chunked, cancellable path above; NULL keeps
// the original single-dispatch hot path below byte-for-byte.
// ---------------------------------------------------------------------------
int ecdsa_batch_core(
    const secp256k1_context*               ctx,
    const secp256k1_ecdsa_signature* const* sigs,
    const unsigned char* const*             msgs32,
    const secp256k1_pubkey* const*          pubkeys,
    size_t                                  n,
    size_t                                  max_threads,
    int*                                    results,
    const ufsecp_cancel_token*              cancel = nullptr)
{
    if (!ctx_can_verify(ctx)) return 0;
    if (n == 0) return 1;  // vacuously valid
    if (!sigs || !msgs32 || !pubkeys) return 0;
    if (cancel) return ecdsa_batch_core_cancel(sigs, msgs32, pubkeys, n, max_threads, results, cancel);

    // -- Per-row results path -------------------------------------------------
    if (results) {
        static thread_local std::vector<secp256k1::ECDSABatchEntry> rbatch;
        static thread_local std::vector<size_t>                     ridx;
        rbatch.clear(); ridx.clear();
        rbatch.reserve(n); ridx.reserve(n);

        for (size_t i = 0; i < n; ++i) {
            results[i] = 0;
            if (!sigs[i] || !msgs32[i] || !pubkeys[i]) continue;
            Scalar r, s;
            if (!Scalar::parse_bytes_strict_le(sigs[i]->data,      r)) continue;
            if (!Scalar::parse_bytes_strict_le(sigs[i]->data + 32, s)) continue;
            secp256k1::ECDSABatchEntry e{};
            std::memcpy(e.msg_hash.data(), msgs32[i], 32);
            e.signature  = secp256k1::ECDSASignature{r, s};
            using secp256k1_shim_internal::pubkey_data_to_point;
            e.public_key = pubkey_data_to_point(pubkeys[i]->data);
            rbatch.push_back(e);
            ridx.push_back(i);
        }

        const bool all_parsed = (rbatch.size() == n);
        // Fast path: every row parsed and the whole batch is valid.
        if (all_parsed && ecdsa_dispatch(rbatch.data(), rbatch.size(), max_threads)) {
            for (size_t i = 0; i < n; ++i) results[i] = 1;
            return 1;
        }
        // Pinpoint the invalid rows among the parseable subset.
        std::vector<size_t> invalid;
        secp256k1::ecdsa_batch_identify_invalid(rbatch.data(), rbatch.size(), invalid);
        for (size_t j = 0; j < ridx.size(); ++j) results[ridx[j]] = 1;
        for (size_t inv : invalid) results[ridx[inv]] = 0;
        return (all_parsed && invalid.empty()) ? 1 : 0;
    }

    // -- All-or-nothing path --------------------------------------------------
    if (n < kBatchMinEcdsa) {
        for (size_t i = 0; i < n; ++i) {
            if (!ecdsa_verify_one(sigs[i], msgs32[i], pubkeys[i])) return 0;
        }
        return 1;
    }

    // Marshalling scratch is a thread_local grow-only vector: it is cleared (not
    // freed) between calls so capacity is retained for amortized performance on
    // repeated large batches. shrink_to_fit() removed (PERF-004) — shrinking would
    // force a reallocation on the next batch and defeat the amortization.
    static thread_local std::vector<secp256k1::ECDSABatchEntry> batch;
    batch.clear();
    batch.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (!sigs[i] || !msgs32[i] || !pubkeys[i]) return 0;
        secp256k1::ECDSABatchEntry e{};
        std::memcpy(e.msg_hash.data(), msgs32[i], 32);
        Scalar r, s;
        if (!Scalar::parse_bytes_strict_le(sigs[i]->data,      r)) return 0;
        if (!Scalar::parse_bytes_strict_le(sigs[i]->data + 32, s)) return 0;
        e.signature = secp256k1::ECDSASignature{r, s};
        using secp256k1_shim_internal::pubkey_data_to_point;
        e.public_key = pubkey_data_to_point(pubkeys[i]->data);
        batch.push_back(e);
    }
    return ecdsa_dispatch(batch.data(), batch.size(), max_threads) ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Schnorr batch core. Mirrors ecdsa_batch_core; msglen != 32 is served via
// per-signature verify (MSM requires fixed 32-byte message slots, matching
// upstream BIP-340 semantics for arbitrary-length messages).
// ---------------------------------------------------------------------------
int schnorr_batch_core(
    const secp256k1_context*              ctx,
    const unsigned char* const*           sigs64,
    const unsigned char* const*           msgs,
    size_t                                msglen,
    const secp256k1_xonly_pubkey* const*  pubkeys,
    size_t                                n,
    size_t                                max_threads,
    int*                                  results,
    const ufsecp_cancel_token*            cancel = nullptr)
{
    if (!ctx_can_verify(ctx)) return 0;
    if (n == 0) return 1;  // vacuously valid
    if (!sigs64 || !msgs || !pubkeys) return 0;
    if (cancel) return schnorr_batch_core_cancel(sigs64, msgs, msglen, pubkeys, n, max_threads, results, cancel);

    // -- Per-row results path -------------------------------------------------
    if (results) {
        // Variable-length messages cannot be batched -> per-row verify.
        if (msglen != 32) {
            int all = 1;
            for (size_t i = 0; i < n; ++i) {
                results[i] = schnorr_verify_one(sigs64[i], msgs[i], msglen, pubkeys[i]) ? 1 : 0;
                if (!results[i]) all = 0;
            }
            return all;
        }

        static thread_local std::vector<secp256k1::SchnorrBatchEntry> rbatch;
        static thread_local std::vector<size_t>                       ridx;
        rbatch.clear(); ridx.clear();
        rbatch.reserve(n); ridx.reserve(n);

        for (size_t i = 0; i < n; ++i) {
            results[i] = 0;
            if (!sigs64[i] || !msgs[i] || !pubkeys[i]) continue;
            secp256k1::SchnorrBatchEntry e{};
            std::memcpy(e.pubkey_x.data(), pubkeys[i]->data, 32);
            std::memcpy(e.message.data(),  msgs[i],          32);
            std::array<uint8_t, 64> sb{};
            std::memcpy(sb.data(), sigs64[i], 64);
            if (!secp256k1::SchnorrSignature::parse_strict(sb, e.signature)) continue;
            rbatch.push_back(e);
            ridx.push_back(i);
        }

        const bool all_parsed = (rbatch.size() == n);
        if (all_parsed && schnorr_dispatch(rbatch.data(), rbatch.size(), max_threads)) {
            for (size_t i = 0; i < n; ++i) results[i] = 1;
            return 1;
        }
        std::vector<size_t> invalid;
        secp256k1::schnorr_batch_identify_invalid(rbatch.data(), rbatch.size(), invalid);
        for (size_t j = 0; j < ridx.size(); ++j) results[ridx[j]] = 1;
        for (size_t inv : invalid) results[ridx[inv]] = 0;
        return (all_parsed && invalid.empty()) ? 1 : 0;
    }

    // -- All-or-nothing path --------------------------------------------------
    // Variable-length messages: per-signature verify (any msglen valid per BIP-340).
    if (msglen != 32) {
        for (size_t i = 0; i < n; ++i) {
            if (!schnorr_verify_one(sigs64[i], msgs[i], msglen, pubkeys[i])) return 0;
        }
        return 1;
    }

    // msglen == 32: small batches fall back to individual verify (lower overhead).
    if (n < kBatchMinSchnorr) {
        for (size_t i = 0; i < n; ++i) {
            if (!schnorr_verify_one(sigs64[i], msgs[i], 32, pubkeys[i])) return 0;
        }
        return 1;
    }

    static thread_local std::vector<secp256k1::SchnorrBatchEntry> batch;
    batch.clear();
    batch.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (!sigs64[i] || !msgs[i] || !pubkeys[i]) return 0;
        secp256k1::SchnorrBatchEntry e{};
        std::memcpy(e.pubkey_x.data(), pubkeys[i]->data, 32);
        std::memcpy(e.message.data(),  msgs[i],          32);
        std::array<uint8_t, 64> sb{};
        std::memcpy(sb.data(), sigs64[i], 64);
        if (!secp256k1::SchnorrSignature::parse_strict(sb, e.signature)) return 0;
        batch.push_back(e);
    }
    return schnorr_dispatch(batch.data(), batch.size(), max_threads) ? 1 : 0;
}

} // namespace

extern "C" {

int secp256k1_schnorrsig_verify_batch(
    const secp256k1_context*         ctx,
    const unsigned char* const*      sigs64,
    const unsigned char* const*      msgs,
    size_t                           msglen,
    const secp256k1_xonly_pubkey* const* pubkeys,
    size_t                           n,
    const ufsecp_cancel_token*       cancel)
{
    if (!ctx) {
        secp256k1_shim_call_illegal_cb(NULL, "secp256k1_schnorrsig_verify_batch: NULL context");
        return 0;
    }
    return schnorr_batch_core(ctx, sigs64, msgs, msglen, pubkeys, n, 0 /*auto*/, nullptr, cancel);
}

int secp256k1_schnorrsig_verify_batch_mt(
    const secp256k1_context*         ctx,
    const unsigned char* const*      sigs64,
    const unsigned char* const*      msgs,
    size_t                           msglen,
    const secp256k1_xonly_pubkey* const* pubkeys,
    size_t                           n,
    size_t                           max_threads,
    const ufsecp_cancel_token*       cancel)
{
    if (!ctx) {
        secp256k1_shim_call_illegal_cb(NULL, "secp256k1_schnorrsig_verify_batch_mt: NULL context");
        return 0;
    }
    return schnorr_batch_core(ctx, sigs64, msgs, msglen, pubkeys, n, max_threads, nullptr, cancel);
}

int secp256k1_schnorrsig_verify_batch_results(
    const secp256k1_context*         ctx,
    const unsigned char* const*      sigs64,
    const unsigned char* const*      msgs,
    size_t                           msglen,
    const secp256k1_xonly_pubkey* const* pubkeys,
    size_t                           n,
    size_t                           max_threads,
    int*                             results,
    const ufsecp_cancel_token*       cancel)
{
    if (!ctx) {
        secp256k1_shim_call_illegal_cb(NULL, "secp256k1_schnorrsig_verify_batch_results: NULL context");
        return 0;
    }
    if (!results) return 0;
    return schnorr_batch_core(ctx, sigs64, msgs, msglen, pubkeys, n, max_threads, results, cancel);
}

int secp256k1_ecdsa_verify_batch(
    const secp256k1_context*               ctx,
    const secp256k1_ecdsa_signature* const* sigs,
    const unsigned char* const*             msgs32,
    const secp256k1_pubkey* const*          pubkeys,
    size_t                                  n,
    const ufsecp_cancel_token*              cancel)
{
    if (!ctx) {
        secp256k1_shim_call_illegal_cb(NULL, "secp256k1_ecdsa_verify_batch: NULL context");
        return 0;
    }
    return ecdsa_batch_core(ctx, sigs, msgs32, pubkeys, n, 0 /*auto*/, nullptr, cancel);
}

int secp256k1_ecdsa_verify_batch_mt(
    const secp256k1_context*               ctx,
    const secp256k1_ecdsa_signature* const* sigs,
    const unsigned char* const*             msgs32,
    const secp256k1_pubkey* const*          pubkeys,
    size_t                                  n,
    size_t                                  max_threads,
    const ufsecp_cancel_token*              cancel)
{
    if (!ctx) {
        secp256k1_shim_call_illegal_cb(NULL, "secp256k1_ecdsa_verify_batch_mt: NULL context");
        return 0;
    }
    return ecdsa_batch_core(ctx, sigs, msgs32, pubkeys, n, max_threads, nullptr, cancel);
}

int secp256k1_ecdsa_verify_batch_results(
    const secp256k1_context*               ctx,
    const secp256k1_ecdsa_signature* const* sigs,
    const unsigned char* const*             msgs32,
    const secp256k1_pubkey* const*          pubkeys,
    size_t                                  n,
    size_t                                  max_threads,
    int*                                    results,
    const ufsecp_cancel_token*              cancel)
{
    if (!ctx) {
        secp256k1_shim_call_illegal_cb(NULL, "secp256k1_ecdsa_verify_batch_results: NULL context");
        return 0;
    }
    if (!results) return 0;
    return ecdsa_batch_core(ctx, sigs, msgs32, pubkeys, n, max_threads, results, cancel);
}

} // extern "C"
