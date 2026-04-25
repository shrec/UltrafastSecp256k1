// ============================================================================
// shim_schnorr.cpp -- BIP-340 Schnorr sign/verify
// ============================================================================
#include "secp256k1_schnorrsig.h"
#include "secp256k1_extrakeys.h"

#include <cstring>
#include <array>

#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/schnorr.hpp"

using namespace secp256k1::fast;

extern "C" {

// -- BIP-340 nonce function (stub -- sign uses internal BIP-340 nonce) -----

static int nonce_function_bip340_stub(unsigned char *, const unsigned char *,
    size_t, const unsigned char *, const unsigned char *,
    const unsigned char *, size_t, void *)
{ return 1; }

const secp256k1_nonce_function_hardened secp256k1_nonce_function_bip340 =
    nonce_function_bip340_stub;

// -- Sign -----------------------------------------------------------------

int secp256k1_schnorrsig_sign32(
    const secp256k1_context *ctx,
    unsigned char *sig64,
    const unsigned char *msg32,
    const secp256k1_keypair *keypair,
    const unsigned char *aux_rand32)
{
    (void)ctx;
    if (!sig64 || !msg32 || !keypair) return 0;

    try {
        // Extract secret key from keypair (first 32 bytes)
        std::array<uint8_t, 32> sk_bytes{};
        std::memcpy(sk_bytes.data(), keypair->data, 32);
        auto sk = Scalar::from_bytes(sk_bytes);
        if (sk.is_zero()) return 0;

        std::array<uint8_t, 32> msg{};
        std::memcpy(msg.data(), msg32, 32);

        std::array<uint8_t, 32> aux{};
        if (aux_rand32) std::memcpy(aux.data(), aux_rand32, 32);

        auto sig = secp256k1::schnorr_sign(sk, msg, aux);

        // Output: R.x (32) || s (32)
        auto sig_bytes = sig.to_bytes();
        std::memcpy(sig64, sig_bytes.data(), 64);
        return 1;
    } catch (...) { return 0; }
}

// -- Verify ---------------------------------------------------------------

int secp256k1_schnorrsig_verify(
    const secp256k1_context *ctx,
    const unsigned char *sig64,
    const unsigned char *msg, size_t msglen,
    const secp256k1_xonly_pubkey *pubkey)
{
    (void)ctx;
    if (!sig64 || !pubkey) return 0;
    // BIP-340 standard: 32-byte messages
    if (msglen != 32 || !msg) return 0;

    try {
        // Use pre-cached pubkey (X || Y stored at parse time — no sqrt needed).
        std::array<uint8_t, 32> pk_x{}, pk_y_bytes{};
        std::memcpy(pk_x.data(),       pubkey->data,      32);
        std::memcpy(pk_y_bytes.data(), pubkey->data + 32, 32);

        std::array<uint8_t, 32> msg32{};
        std::memcpy(msg32.data(), msg, 32);

        std::array<uint8_t, 64> sig_bytes{};
        std::memcpy(sig_bytes.data(), sig64, 64);
        auto sig = secp256k1::SchnorrSignature::from_bytes(sig_bytes);

        // Build pre-cached pubkey: reconstruct Point from (X, Y) — no sqrt.
        secp256k1::SchnorrXonlyPubkey xpub{};
        xpub.x_bytes = pk_x;
        auto px = FieldElement::from_bytes(pk_x);
        auto py = FieldElement::from_bytes(pk_y_bytes);
        xpub.point = Point::from_affine(px, py);

        return secp256k1::schnorr_verify(xpub, msg32, sig) ? 1 : 0;
    } catch (...) { return 0; }
}

} // extern "C"
