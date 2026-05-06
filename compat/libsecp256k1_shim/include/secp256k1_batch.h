#ifndef SECP256K1_BATCH_H
#define SECP256K1_BATCH_H

#include "secp256k1.h"
#include "secp256k1_schnorrsig.h"
#include "secp256k1_extrakeys.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Batch Schnorr (BIP-340) verification.
 *
 * Verifies n signatures using a single multi-scalar multiplication.
 * Performance: ~2-3x faster than n individual secp256k1_schnorrsig_verify()
 * calls for n >= 32. For n < 8, falls back to individual verification.
 *
 * If any signature is invalid the function returns 0. To identify WHICH
 * signature(s) are invalid, the caller must re-verify individually.
 *
 * Inputs:
 *   ctx:     context object (must have VERIFY flag)
 *   sigs64:  array of n pointers to 64-byte Schnorr signatures
 *   msgs:    array of n pointers to message buffers
 *   msglen:  byte length of each message (all messages must be the same length)
 *   pubkeys: array of n pointers to secp256k1_xonly_pubkey
 *   n:       number of signatures (0 returns 1, i.e. vacuously valid)
 *
 * Returns 1 if all n signatures are valid; 0 otherwise or on error.
 */
SECP256K1_API int secp256k1_schnorrsig_verify_batch(
    const secp256k1_context*        ctx,
    const unsigned char* const*     sigs64,
    const unsigned char* const*     msgs,
    size_t                          msglen,
    const secp256k1_xonly_pubkey* const* pubkeys,
    size_t                          n
) SECP256K1_ARG_NONNULL(1);

/* Batch ECDSA verification.
 *
 * Verifies n ECDSA signatures using a single multi-scalar multiplication.
 * Performance: ~1.5-2x faster than n individual secp256k1_ecdsa_verify()
 * calls for n >= 32. For n < 8, falls back to individual verification.
 *
 * All signatures must already be in low-S normalized form (as returned by
 * secp256k1_ecdsa_signature_normalize). High-S signatures are rejected.
 *
 * If any signature is invalid the function returns 0. The caller must
 * re-verify individually to identify which one(s) are invalid.
 *
 * Inputs:
 *   ctx:     context object (must have VERIFY flag)
 *   sigs:    array of n pointers to secp256k1_ecdsa_signature
 *   msgs32:  array of n pointers to 32-byte message hashes
 *   pubkeys: array of n pointers to secp256k1_pubkey
 *   n:       number of signatures (0 returns 1, i.e. vacuously valid)
 *
 * Returns 1 if all n signatures are valid; 0 otherwise or on error.
 */
SECP256K1_API int secp256k1_ecdsa_verify_batch(
    const secp256k1_context*              ctx,
    const secp256k1_ecdsa_signature* const* sigs,
    const unsigned char* const*           msgs32,
    const secp256k1_pubkey* const*        pubkeys,
    size_t                                n
) SECP256K1_ARG_NONNULL(1);

#ifdef __cplusplus
}
#endif

#endif /* SECP256K1_BATCH_H */
