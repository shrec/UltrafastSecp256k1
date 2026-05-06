// ============================================================================
// shim_batch_verify.cpp -- Batch ECDSA and Schnorr verification
// ============================================================================
// Exposes secp256k1_schnorrsig_verify_batch() and secp256k1_ecdsa_verify_batch()
// as libsecp256k1-compatible shim functions backed by UltrafastSecp256k1's
// multi-scalar multiplication (Pippenger). These are ADDITIVE symbols — the
// existing secp256k1_schnorrsig_verify / secp256k1_ecdsa_verify are unchanged.
//
// Performance vs n individual verifications (Pippenger at n >= 32):
//   Schnorr batch: ~2-3x faster
//   ECDSA batch:   ~1.5-2x faster (limited by per-sig modular inverse)
//
// API design:
//   n = 0: returns 1 (vacuously valid)
//   n < 8: falls back to individual verification (batch overhead > benefit)
//   Any invalid input pointer: returns 0 (fail-closed)
// ============================================================================

#include "secp256k1_batch.h"
#include "shim_internal.hpp"

#include <cstring>
#include <array>
#include <vector>
#include <cstdint>

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

extern "C" {

int secp256k1_schnorrsig_verify_batch(
    const secp256k1_context*         ctx,
    const unsigned char* const*      sigs64,
    const unsigned char* const*      msgs,
    size_t                           msglen,
    const secp256k1_xonly_pubkey* const* pubkeys,
    size_t                           n)
{
    if (!ctx_can_verify(ctx)) return 0;
    if (n == 0) return 1;  // vacuously valid
    if (!sigs64 || !msgs || !pubkeys) return 0;

    // Only 32-byte messages supported by internal batch_verify.
    if (msglen != 32) return 0;

    // Small batches: fall back to individual verify (lower overhead).
    if (n < kBatchMinSchnorr) {
        for (size_t i = 0; i < n; ++i) {
            if (!sigs64[i] || !msgs[i] || !pubkeys[i]) return 0;
            secp256k1::SchnorrSignature sig;
            std::array<uint8_t, 64> sb{};
            std::memcpy(sb.data(), sigs64[i], 64);
            if (!secp256k1::SchnorrSignature::parse_strict(sb, sig)) return 0;
            std::array<uint8_t, 32> pk_bytes{};
            std::memcpy(pk_bytes.data(), pubkeys[i]->data, 32);
            std::array<uint8_t, 32> msg32{};
            std::memcpy(msg32.data(), msgs[i], 32);
            if (!secp256k1::schnorr_verify(pk_bytes, msg32, sig)) return 0;
        }
        return 1;
    }

    // Build batch entries.
    std::vector<secp256k1::SchnorrBatchEntry> batch;
    batch.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        if (!sigs64[i] || !msgs[i] || !pubkeys[i]) return 0;

        secp256k1::SchnorrBatchEntry e{};
        std::memcpy(e.pubkey_x.data(), pubkeys[i]->data, 32);
        std::memcpy(e.message.data(),  msgs[i],           32);

        std::array<uint8_t, 64> sb{};
        std::memcpy(sb.data(), sigs64[i], 64);
        if (!secp256k1::SchnorrSignature::parse_strict(sb, e.signature))
            return 0;

        batch.push_back(e);
    }

    return secp256k1::schnorr_batch_verify(batch) ? 1 : 0;
}

int secp256k1_ecdsa_verify_batch(
    const secp256k1_context*               ctx,
    const secp256k1_ecdsa_signature* const* sigs,
    const unsigned char* const*             msgs32,
    const secp256k1_pubkey* const*          pubkeys,
    size_t                                  n)
{
    if (!ctx_can_verify(ctx)) return 0;
    if (n == 0) return 1;
    if (!sigs || !msgs32 || !pubkeys) return 0;

    // Small batches: fall back to individual verify.
    if (n < kBatchMinEcdsa) {
        for (size_t i = 0; i < n; ++i) {
            if (!sigs[i] || !msgs32[i] || !pubkeys[i]) return 0;
            // Reconstruct ECDSASignature from opaque compact format.
            Scalar r, s;
            if (!Scalar::parse_bytes_strict(sigs[i]->data,      r)) return 0;
            if (!Scalar::parse_bytes_strict(sigs[i]->data + 32, s)) return 0;
            secp256k1::ECDSASignature sig{r, s};
            if (!sig.is_low_s()) return 0;

            // Reconstruct Point from opaque uncompressed x||y.
            std::array<uint8_t, 32> xb{}, yb{};
            std::memcpy(xb.data(), pubkeys[i]->data,      32);
            std::memcpy(yb.data(), pubkeys[i]->data + 32, 32);
            auto x = FieldElement::from_bytes(xb);
            auto y = FieldElement::from_bytes(yb);
            auto pt = Point::from_affine(x, y);
            if (pt.is_infinity()) return 0;

            std::array<uint8_t, 32> msg{};
            std::memcpy(msg.data(), msgs32[i], 32);
            if (!secp256k1::ecdsa_verify(msg, pt, sig)) return 0;
        }
        return 1;
    }

    // Build batch entries.
    std::vector<secp256k1::ECDSABatchEntry> batch;
    batch.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        if (!sigs[i] || !msgs32[i] || !pubkeys[i]) return 0;

        secp256k1::ECDSABatchEntry e{};
        std::memcpy(e.msg_hash.data(), msgs32[i], 32);

        Scalar r, s;
        if (!Scalar::parse_bytes_strict(sigs[i]->data,      r)) return 0;
        if (!Scalar::parse_bytes_strict(sigs[i]->data + 32, s)) return 0;
        e.signature = secp256k1::ECDSASignature{r, s};
        if (!e.signature.is_low_s()) return 0;

        std::array<uint8_t, 32> xb{}, yb{};
        std::memcpy(xb.data(), pubkeys[i]->data,      32);
        std::memcpy(yb.data(), pubkeys[i]->data + 32, 32);
        auto x = FieldElement::from_bytes(xb);
        auto y = FieldElement::from_bytes(yb);
        e.public_key = Point::from_affine(x, y);
        if (e.public_key.is_infinity()) return 0;

        batch.push_back(e);
    }

    return secp256k1::ecdsa_batch_verify(batch) ? 1 : 0;
}

} // extern "C"
