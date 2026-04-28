#include "secp256k1_schnorrsig.h"
#include "secp256k1_extrakeys.h"

#include <cstring>
#include <array>

#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/schnorr.hpp"

using namespace secp256k1::fast;

extern "C" {

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

    Scalar sk;
    if (!Scalar::parse_bytes_strict_nonzero(keypair->data, sk)) return 0;

    std::array<uint8_t, 32> msg{};
    std::memcpy(msg.data(), msg32, 32);

    std::array<uint8_t, 32> aux{};
    if (aux_rand32) std::memcpy(aux.data(), aux_rand32, 32);

    auto sig = secp256k1::schnorr_sign(sk, msg, aux);
    auto sig_bytes = sig.to_bytes();
    std::memcpy(sig64, sig_bytes.data(), 64);
    return 1;
}

int secp256k1_schnorrsig_sign_custom(
    const secp256k1_context *ctx,
    unsigned char *sig64,
    const unsigned char *msg,
    size_t msglen,
    const secp256k1_keypair *keypair,
    secp256k1_nonce_function_hardened noncefp,
    void *ndata)
{
    (void)noncefp; (void)ndata;
    // NOTE: This implementation restricts messages to exactly 32 bytes, matching
    // BIP-340 strict mode. Upstream libsecp256k1 sign_custom accepts variable-length
    // messages by including them verbatim in H_BIP0340/challenge(R_x||P_x||msg).
    // Variable-length support requires extending the internal schnorr_sign API.
    // Bitcoin Core itself only calls sign_custom with 32-byte hashes.
    if (!msg || msglen != 32) return 0;
    return secp256k1_schnorrsig_sign32(ctx, sig64, msg, keypair, nullptr);
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
    if (!msg || msglen != 32) return 0;

    secp256k1::SchnorrSignature sig;
    std::array<uint8_t, 64> sig_buf{};
    std::memcpy(sig_buf.data(), sig64, 64);
    if (!secp256k1::SchnorrSignature::parse_strict(sig_buf, sig)) return 0;

    std::array<uint8_t, 32> msg32{};
    std::memcpy(msg32.data(), msg, 32);

    // x-only key is stored in the first 32 bytes of the opaque 64-byte struct.
    // schnorr_verify handles the lift (sqrt) internally and rejects invalid x.
    std::array<uint8_t, 32> pk_x{};
    std::memcpy(pk_x.data(), pubkey->data, 32);
    return secp256k1::schnorr_verify(pk_x, msg32, sig) ? 1 : 0;
}

} // extern "C"
