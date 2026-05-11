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
    auto unc = P.to_uncompressed(); // [0x04] [x:32] [y:32]
    std::memcpy(data,      unc.data() + 1,  32); // x
    std::memcpy(data + 32, unc.data() + 33, 32); // y
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
    std::array<uint8_t, 32> rb{}, sb{};
    std::memcpy(rb.data(), data + 1,  32);
    std::memcpy(sb.data(), data + 33, 32);
    return {
        { Scalar::from_bytes(rb), Scalar::from_bytes(sb) },
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
    (void)ctx;
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
    (void)ctx;
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
    (void)ctx;
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
    if (!sig || !msghash32 || !seckey) return 0;
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
        // RFC 6979 + extra entropy: sign hedged then derive recovery ID.
        std::array<uint8_t, 32> aux{};
        std::memcpy(aux.data(), ndata, 32);
        auto plain_sig = secp256k1::ct::ecdsa_sign_hedged(msg, privkey_scalar, aux);
        if (plain_sig.r.is_zero() || plain_sig.s.is_zero()) return 0;

        // Derive recid: always evaluate all 4 candidates unconditionally so
        // the iteration count does not depend on which recid is correct.
        // Cache expected_pk serialization once outside the loop.
        auto expected_pk = secp256k1::ct::generator_mul(privkey_scalar);
        auto expected_unc = expected_pk.to_uncompressed();
        int found_recid = -1;
        for (int rid = 0; rid < 4; ++rid) {
            auto [pk_try, ok] = secp256k1::ecdsa_recover(msg, plain_sig, rid);
            if (ok && !pk_try.is_infinity() &&
                pk_try.to_uncompressed() == expected_unc) {
                if (found_recid < 0) found_recid = rid; // no early exit
            }
        }
        if (found_recid < 0) return 0;
        rsig = {plain_sig, found_recid};
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
    if (!pubkey || !sig || !msghash32) return 0;

    std::array<uint8_t, 32> msg{};
    std::memcpy(msg.data(), msghash32, 32);

    auto rsig = rsig_from_data(sig->data);

    auto [pk, ok] = secp256k1::ecdsa_recover(msg, rsig.sig, rsig.recid);
    if (!ok || pk.is_infinity()) return 0;

    point_to_pubkey_data(pk, pubkey->data);
    return 1;
}

} // extern "C"
