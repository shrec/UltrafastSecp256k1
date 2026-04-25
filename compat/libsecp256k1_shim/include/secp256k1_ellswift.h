/* ============================================================================
 * secp256k1_ellswift.h -- BIP-324 ElligatorSwift encoding (shim header)
 *
 * Drop-in compatible with bitcoin-core/secp256k1's secp256k1_ellswift.h.
 * ========================================================================== */
#ifndef SECP256K1_ELLSWIFT_H
#define SECP256K1_ELLSWIFT_H

#include "secp256k1.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Hash function type for secp256k1_ellswift_xdh.
 * Receives: output[32], x32 (ECDH x-coord), ell_a64, ell_b64, user data.
 * Returns 1 on success, 0 on failure. */
typedef int (*secp256k1_ellswift_xdh_hash_function)(
    unsigned char *output,
    const unsigned char *x32,
    const unsigned char *ell_a64,
    const unsigned char *ell_b64,
    void *data);

/* Built-in BIP-324 hash function (SHA256 tagged "bip324_ellswift_xonly_ecdh") */
extern const secp256k1_ellswift_xdh_hash_function
    secp256k1_ellswift_xdh_hash_function_bip324;

/* Built-in prefix-based hash: SHA256(prefix64 || ell_a64 || ell_b64 || x32) */
extern const secp256k1_ellswift_xdh_hash_function
    secp256k1_ellswift_xdh_hash_function_prefix;

/* Encode a public key to a 64-byte ElligatorSwift encoding.
 * rnd32: 32 bytes of randomness (may be NULL for deterministic but still safe).
 * Returns 1 on success. */
SECP256K1_API int secp256k1_ellswift_encode(
    const secp256k1_context *ctx,
    unsigned char *ell64,
    const secp256k1_pubkey *pubkey,
    const unsigned char *rnd32
) SECP256K1_WARN_UNUSED_RESULT;

/* Decode a 64-byte ElligatorSwift encoding to a public key.
 * Returns 1 on success. */
SECP256K1_API int secp256k1_ellswift_decode(
    const secp256k1_context *ctx,
    secp256k1_pubkey *pubkey,
    const unsigned char *ell64
) SECP256K1_WARN_UNUSED_RESULT;

/* Generate a fresh ephemeral key and its ElligatorSwift encoding.
 * seckey32: 32-byte private key output.
 * ell64: 64-byte ElligatorSwift encoding output.
 * auxrnd32: optional 32 bytes of extra randomness (may be NULL).
 * Returns 1 on success. */
SECP256K1_API int secp256k1_ellswift_create(
    const secp256k1_context *ctx,
    unsigned char *ell64,
    const unsigned char *seckey32,
    const unsigned char *auxrnd32
) SECP256K1_WARN_UNUSED_RESULT;

/* Perform x-only ECDH with ElligatorSwift-encoded keys and custom hash.
 * ell_a64: initiator's 64-byte encoding.
 * ell_b64: responder's 64-byte encoding.
 * seckey32: our 32-byte private key.
 * party: 0 = initiator (A), 1 = responder (B).
 * hashfp: hash function (use secp256k1_ellswift_xdh_hash_function_bip324 for BIP-324).
 * data: extra data passed to hashfp (may be NULL).
 * output: 32-byte shared secret output.
 * Returns 1 on success. */
SECP256K1_API int secp256k1_ellswift_xdh(
    const secp256k1_context *ctx,
    unsigned char *output,
    const unsigned char *ell_a64,
    const unsigned char *ell_b64,
    const unsigned char *seckey32,
    int party,
    secp256k1_ellswift_xdh_hash_function hashfp,
    void *data
) SECP256K1_WARN_UNUSED_RESULT;

#ifdef __cplusplus
}
#endif

#endif /* SECP256K1_ELLSWIFT_H */
