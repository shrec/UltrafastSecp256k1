// ============================================================================
// shim_recovery.cpp -- ECDSA sign-recoverable + public key recovery
// ============================================================================
// Implements the libsecp256k1 recovery module API using the internal
// ct::ecdsa_sign_recoverable() (constant-time) and ecdsa_recover() (fast,
// recovery is a public operation).
// ============================================================================

#include "secp256k1_recovery.h"
#include "shim_internal.hpp"

#include <cstring>
#include <array>

#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/field.hpp"
#include "secp256k1/recovery.hpp"
#include "secp256k1/ct/sign.hpp"
#include "secp256k1/ct/point.hpp"

using namespace secp256k1::fast;

// Context flag helpers — use the canonical implementations from shim_internal.hpp.
// NULL ctx returns false (matches libsecp256k1: triggers illegal callback, returns 0).
using secp256k1_shim_internal::ctx_flags;
using secp256k1_shim_internal::ctx_can_sign;
using secp256k1_shim_internal::ctx_can_verify;

// Internal helpers that mirror shim_ecdsa.cpp conventions ----------------

static void point_to_pubkey_data(const Point& P, unsigned char data[64]) {
    // PERF-001: fast path for affine points (Z=1). ecdsa_recover always returns
    // an affine point, so this path is taken on every call — saves ~1,300 ns/call
    // vs the unconditional to_uncompressed() that performs a full field inversion.
    if (P.is_normalized()) {
        P.x_raw().to_bytes_into(reinterpret_cast<uint8_t*>(data));
        P.y_raw().to_bytes_into(reinterpret_cast<uint8_t*>(data) + 32);
    } else {
        auto unc = P.to_uncompressed(); // [0x04] [x:32] [y:32]
        std::memcpy(data,      unc.data() + 1,  32);
        std::memcpy(data + 32, unc.data() + 33, 32);
    }
}

// Recoverable sig opaque layout: data[0] = recid, data[1..32] = r, data[33..64] = s

static void rsig_to_data(const secp256k1::RecoverableSignature& rsig,
                         unsigned char data[65]) {
    data[0] = static_cast<unsigned char>(rsig.recid & 0x03);
    auto rb = rsig.sig.r.to_bytes();
    auto sb = rsig.sig.s.to_bytes();
    std::memcpy(data + 1,  rb.data(), 32);
    std::memcpy(data + 33, sb.data(), 32);
}

static secp256k1::RecoverableSignature rsig_from_data(const unsigned char data[65]) {
    // T-07: strict parse — values >= n are cleared to zero so downstream verify fails cleanly.
    Scalar r_scalar, s_scalar;
    if (!Scalar::parse_bytes_strict(data + 1,  r_scalar)) r_scalar = Scalar::zero();
    if (!Scalar::parse_bytes_strict(data + 33, s_scalar)) s_scalar = Scalar::zero();
    return {
        { r_scalar, s_scalar },
        static_cast<int>(data[0] & 0x03)
    };
}

extern "C" {

// -- Parse / Serialize --------------------------------------------------------

int secp256k1_ecdsa_recoverable_signature_parse_compact(
    const secp256k1_context *ctx,
    secp256k1_ecdsa_recoverable_signature *sig,
    const unsigned char *input64,
    int recid)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!sig || !input64) return 0;
    if (recid < 0 || recid > 3) return 0;
    // Validate r and s are in (0, n-1] — matches libsecp strict contract.
    Scalar r, s;
    if (!Scalar::parse_bytes_strict_nonzero(
            reinterpret_cast<const uint8_t*>(input64),      r)) return 0;
    if (!Scalar::parse_bytes_strict_nonzero(
            reinterpret_cast<const uint8_t*>(input64 + 32), s)) return 0;
    sig->data[0] = static_cast<unsigned char>(recid);
    std::memcpy(sig->data + 1, input64, 64);
    return 1;
}

int secp256k1_ecdsa_recoverable_signature_serialize_compact(
    const secp256k1_context *ctx,
    unsigned char *output64,
    int *recid,
    const secp256k1_ecdsa_recoverable_signature *sig)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!output64 || !recid || !sig) return 0;
    *recid = static_cast<int>(sig->data[0] & 0x03);
    std::memcpy(output64, sig->data + 1, 64);
    return 1;
}

int secp256k1_ecdsa_recoverable_signature_convert(
    const secp256k1_context *ctx,
    secp256k1_ecdsa_signature *sig,
    const secp256k1_ecdsa_recoverable_signature *sigin)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!sig || !sigin) return 0;
    // Non-recoverable sig is r||s (64 bytes); strip the recid byte.
    std::memcpy(sig->data, sigin->data + 1, 64);
    return 1;
}

// -- Sign (recoverable) -------------------------------------------------------

int secp256k1_ecdsa_sign_recoverable(
    const secp256k1_context *ctx,
    secp256k1_ecdsa_recoverable_signature *sig,
    const unsigned char *msghash32,
    const unsigned char *seckey,
    secp256k1_nonce_function noncefp,
    const void *ndata)
{
    if (!ctx) {
        secp256k1_shim_call_illegal_cb(nullptr, "secp256k1_ecdsa_sign_recoverable: NULL context");
        return 0;
    }
    if (!ctx_can_sign(ctx)) return 0;
    // SHIM-005: NULL args must fire the illegal callback (matching libsecp behavior)
    if (!sig || !msghash32 || !seckey) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ecdsa_sign_recoverable: NULL argument");
        return 0;
    }
    secp256k1_shim_internal::ContextBlindingScope _blind(ctx);
    if (noncefp != nullptr &&
        noncefp != secp256k1_nonce_function_rfc6979 &&
        noncefp != secp256k1_nonce_function_default) {
        return 0;
    }

    std::array<uint8_t, 32> msg{};
    std::memcpy(msg.data(), msghash32, 32);

    Scalar privkey_scalar;
    if (!Scalar::parse_bytes_strict_nonzero(
            reinterpret_cast<const uint8_t*>(seckey), privkey_scalar)) return 0;

    secp256k1::RecoverableSignature rsig;
    if (ndata) {
        // RFC 6979 + extra entropy: sign hedged and derive recid from R's y-parity
        // during signing — no post-sign ecdsa_recover loop needed.
        // Previously this called ecdsa_sign_hedged() then 4x ecdsa_recover (4-5×
        // overhead per grind iteration). ecdsa_sign_hedged_recoverable() eliminates
        // the recovery loop by computing recid from K's affine y-coordinate and
        // x-overflow flag, the same as ct::ecdsa_sign_recoverable().
        std::array<uint8_t, 32> aux{};
        std::memcpy(aux.data(), ndata, 32);
        rsig = secp256k1::ct::ecdsa_sign_hedged_recoverable(msg, privkey_scalar, aux);
        if (rsig.sig.r.is_zero() || rsig.sig.s.is_zero()) return 0;
    } else {
        rsig = secp256k1::ct::ecdsa_sign_recoverable(msg, privkey_scalar);
    }

    if (rsig.sig.r.is_zero() || rsig.sig.s.is_zero()) return 0;

    rsig_to_data(rsig, sig->data);
    return 1;
}

// -- Recover public key -------------------------------------------------------

int secp256k1_ecdsa_recover(
    const secp256k1_context *ctx,
    secp256k1_pubkey *pubkey,
    const secp256k1_ecdsa_recoverable_signature *sig,
    const unsigned char *msghash32)
{
    if (!ctx_can_verify(ctx)) return 0;
    if (!pubkey || !sig || !msghash32) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ecdsa_recover: NULL argument");
        return 0;
    }

    std::array<uint8_t, 32> msg{};
    std::memcpy(msg.data(), msghash32, 32);

    auto rsig = rsig_from_data(sig->data);

    auto [pk, ok] = secp256k1::ecdsa_recover(msg, rsig.sig, rsig.recid);
    if (!ok || pk.is_infinity()) return 0;

    point_to_pubkey_data(pk, pubkey->data);
    return 1;
}

} // extern "C"
