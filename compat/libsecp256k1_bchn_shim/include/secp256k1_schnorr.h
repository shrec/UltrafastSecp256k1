/* ============================================================================
 * secp256k1_schnorr.h -- Bitcoin Cash Node (BCHN) legacy Schnorr signatures
 *
 * Drop-in compatible with the bchn/secp256k1 fork's secp256k1_schnorr.h.
 *
 * Scheme (NOT BIP-340):
 *   nonce k = RFC6979(seckey, msg32)
 *   R = k * G
 *   e = SHA256(R.x[32] || P_compressed[33] || msg32[32])
 *   s = k + e * seckey  (mod n)
 *   signature: R.x[32] || s[32]  (64 bytes)
 * ========================================================================== */
#ifndef SECP256K1_SCHNORR_H
#define SECP256K1_SCHNORR_H

#include "secp256k1.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Sign a 32-byte message hash with a Schnorr signature (BCH scheme).
 * sig64:   64-byte output: R.x[32] || s[32]
 * msg32:   32-byte message hash
 * seckey:  32-byte private key
 * noncefp: optional custom nonce function (NULL uses RFC6979)
 * ndata:   extra data passed to noncefp (may be NULL)
 * Returns 1 on success, 0 on failure. */
SECP256K1_API int secp256k1_schnorr_sign(
    const secp256k1_context *ctx,
    unsigned char *sig64,
    const unsigned char *msg32,
    const unsigned char *seckey,
    secp256k1_nonce_function noncefp,
    const void *ndata
) SECP256K1_WARN_UNUSED_RESULT;

/* Verify a BCH Schnorr signature.
 * sig64:  64-byte signature: R.x[32] || s[32]
 * msg32:  32-byte message hash
 * pubkey: signer's public key
 * Returns 1 if valid, 0 if invalid. */
SECP256K1_API int secp256k1_schnorr_verify(
    const secp256k1_context *ctx,
    const unsigned char *sig64,
    const unsigned char *msg32,
    const secp256k1_pubkey *pubkey
) SECP256K1_WARN_UNUSED_RESULT;

#ifdef __cplusplus
}
#endif

#endif /* SECP256K1_SCHNORR_H */
