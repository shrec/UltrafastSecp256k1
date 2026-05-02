/* ============================================================================
 * secp256k1_ecdh.h -- ECDH key exchange (shim header)
 *
 * Drop-in compatible with bitcoin-core/secp256k1's secp256k1_ecdh.h.
 * ========================================================================== */
#ifndef SECP256K1_ECDH_ULTRAFAST_H
#define SECP256K1_ECDH_ULTRAFAST_H

#include "secp256k1.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Hash function type for secp256k1_ecdh.
 * Receives: output, x32 (x-coordinate), y32 (y-coordinate), data.
 * Returns 1 on success, 0 on failure. */
typedef int (*secp256k1_ecdh_hashfp)(
    unsigned char *output,
    const unsigned char *x32,
    const unsigned char *y32,
    void *data);

/* Built-in hash: SHA-256(02/03 || x) (compressed-point hash, libsecp default) */
extern const secp256k1_ecdh_hashfp secp256k1_ecdh_hashfp_sha256;

/* Compute ECDH shared secret.
 * pubkey: the other party's public key.
 * seckey: our 32-byte private key.
 * hashfp: hash function applied to the resulting point (NULL uses default SHA-256).
 * data: extra data passed to hashfp (may be NULL).
 * output: 32-byte shared secret output.
 * Returns 1 on success, 0 on failure. */
SECP256K1_API int secp256k1_ecdh(
    const secp256k1_context *ctx,
    unsigned char *output,
    const secp256k1_pubkey *pubkey,
    const unsigned char *seckey,
    secp256k1_ecdh_hashfp hashfp,
    void *data
) SECP256K1_WARN_UNUSED_RESULT;

#ifdef __cplusplus
}
#endif

#endif /* SECP256K1_ECDH_ULTRAFAST_H */
