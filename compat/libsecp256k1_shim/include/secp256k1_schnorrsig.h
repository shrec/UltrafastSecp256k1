/* ============================================================================
 * libsecp256k1-compatible Schnorr API -- backed by UltrafastSecp256k1
 * ========================================================================== */
#ifndef SECP256K1_SCHNORRSIG_ULTRAFAST_H
#define SECP256K1_SCHNORRSIG_ULTRAFAST_H

#include "secp256k1.h"
#include "secp256k1_extrakeys.h"

#ifdef __cplusplus
extern "C" {
#endif

/* -- Nonce function type for Schnorr -------------------------------------- */
typedef int (*secp256k1_nonce_function_hardened)(
    unsigned char *nonce32,
    const unsigned char *msg, size_t msglen,
    const unsigned char *key32,
    const unsigned char *xonly_pk32,
    const unsigned char *algo, size_t algolen,
    void *data
);

SECP256K1_API const secp256k1_nonce_function_hardened secp256k1_nonce_function_bip340;

/* -- Extra params --------------------------------------------------------- */
typedef struct secp256k1_schnorrsig_extraparams {
    unsigned char magic[4];
    secp256k1_nonce_function_hardened noncefp;
    void *ndata;
} secp256k1_schnorrsig_extraparams;

#define SECP256K1_SCHNORRSIG_EXTRAPARAMS_MAGIC { 0xda, 0x6f, 0xb3, 0x8c }
#define SECP256K1_SCHNORRSIG_EXTRAPARAMS_INIT { \
    SECP256K1_SCHNORRSIG_EXTRAPARAMS_MAGIC, \
    NULL, \
    NULL  \
}

/* -- Sign / Verify -------------------------------------------------------- */
SECP256K1_API int secp256k1_schnorrsig_sign32(
    const secp256k1_context *ctx,
    unsigned char *sig64,
    const unsigned char *msg32,
    const secp256k1_keypair *keypair,
    const unsigned char *aux_rand32);

/* secp256k1_schnorrsig_sign_custom — BIP-340 Schnorr sign with custom options.
 *
 * This implementation follows the upstream libsecp256k1 "Bitcoin 32-byte-only
 * profile" with the following documented divergences:
 *
 * 1. NONCE FUNCTION: only NULL and secp256k1_nonce_function_bip340 are
 *    accepted.  Any other non-NULL noncefp causes an immediate return of 0
 *    (fail-closed). The ndata argument is accepted but ignored — aux entropy
 *    must be embedded in extraparams->ndata by callers using the upstream API.
 *    DIVERGENCE: upstream passes ndata to the nonce function; this shim ignores
 *    ndata because the nonce function is not forwarded to an external callback.
 *
 * 2. AUX RAND: when noncefp == NULL (deterministic BIP-340 path), aux entropy
 *    is hardcoded to 32 zero bytes.  This matches the behaviour of upstream
 *    libsecp256k1 when secp256k1_schnorrsig_sign_custom is called with
 *    extraparams == NULL or with extraparams->ndata == NULL.
 *    SECURITY: this gives deterministic (but still CT-safe) nonces. Pass a
 *    non-NULL extraparams->ndata through sign32() for randomised nonces.
 *
 * 3. VARIABLE-LENGTH MESSAGES: sign_custom supports msglen != 32 via the full
 *    BIP-340 H_BIP0340/nonce(t‖P_x‖msg) and H_BIP0340/challenge(R_x‖P_x‖msg)
 *    construction.  secp256k1_schnorrsig_verify ONLY accepts msglen == 32 and
 *    returns 0 for any other length (upstream verify also only supports 32-byte
 *    messages in the default BIP-340 profile). Callers that sign with msglen != 32
 *    must verify externally or with a matching custom verifier.
 *
 * Context flag: SECP256K1_CONTEXT_SIGN is required (enforced; returns 0 on mismatch).
 */
SECP256K1_API int secp256k1_schnorrsig_sign_custom(
    const secp256k1_context *ctx,
    unsigned char *sig64,
    const unsigned char *msg,
    size_t msglen,
    const secp256k1_keypair *keypair,
    secp256k1_schnorrsig_extraparams *extraparams);

/* secp256k1_schnorrsig_verify — BIP-340 Schnorr signature verification.
 *
 * NOTE: only msglen == 32 is accepted; any other length returns 0 immediately.
 * This matches the default BIP-340 32-byte message profile of upstream
 * libsecp256k1.  Signatures produced by sign_custom with msglen != 32 cannot
 * be verified by this function.
 */
SECP256K1_API int secp256k1_schnorrsig_verify(
    const secp256k1_context *ctx,
    const unsigned char *sig64,
    const unsigned char *msg, size_t msglen,
    const secp256k1_xonly_pubkey *pubkey);

/* -- Pre-computed xonly pubkey for fast Schnorr verify -------------------- */
/* Embeds pre-built GLV tables + cached lifted point.
 * Eliminates ~2,600 ns lift_x (sqrt) + ~1,954 ns GLV table rebuild per verify.
 * Size: sizeof(secp256k1::SchnorrXonlyPubkey), typically 1,400-1,504 bytes. */
#ifdef __cplusplus
}  /* suspend extern "C" — schnorr.hpp contains C++ templates, illegal under C linkage */
#  include "secp256k1/schnorr.hpp"
   struct secp256k1_xonly_pubkey_precomp { secp256k1::SchnorrXonlyPubkey epk; };
extern "C" {  /* reopen C linkage for the rest of the shim API */
#else
#  define SECP256K1_XONLY_PUBKEY_PRECOMP_SIZE 1504
   struct secp256k1_xonly_pubkey_precomp {
#    ifdef __STDC_VERSION__
       _Alignas(8) unsigned char _data[SECP256K1_XONLY_PUBKEY_PRECOMP_SIZE];
#    else
       unsigned char _data[SECP256K1_XONLY_PUBKEY_PRECOMP_SIZE];
#    endif
   };
#endif
typedef struct secp256k1_xonly_pubkey_precomp secp256k1_xonly_pubkey_precomp;

/* Build pre-computed form from parsed secp256k1_xonly_pubkey. Returns 1/0. */
SECP256K1_API int secp256k1_xonly_ec_pubkey_precomp(
    const secp256k1_context *ctx,
    secp256k1_xonly_pubkey_precomp *out,
    const secp256k1_xonly_pubkey *pubkey);

/* Parse x-only pubkey bytes and build pre-computed tables in one step. */
SECP256K1_API int secp256k1_xonly_pubkey_parse_precomp(
    const secp256k1_context *ctx,
    secp256k1_xonly_pubkey_precomp *out,
    const unsigned char *pubkey_x32);

/* Verify a Schnorr signature against a pre-computed xonly pubkey.
 * Zero lift_x and zero GLV table rebuild overhead.
 * Only msglen == 32 is accepted (same constraint as schnorrsig_verify). */
SECP256K1_API int secp256k1_schnorrsig_verify_precomp(
    const secp256k1_context *ctx,
    const unsigned char *sig64,
    const unsigned char *msg32,
    const secp256k1_xonly_pubkey_precomp *pubkey);

#ifdef __cplusplus
}
#endif

#endif /* SECP256K1_SCHNORRSIG_ULTRAFAST_H */
