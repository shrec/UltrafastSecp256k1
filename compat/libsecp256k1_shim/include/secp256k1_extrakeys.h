/* ============================================================================
 * libsecp256k1-compatible extrakeys API -- backed by UltrafastSecp256k1
 * ========================================================================== */
#ifndef SECP256K1_EXTRAKEYS_ULTRAFAST_H
#define SECP256K1_EXTRAKEYS_ULTRAFAST_H

#include "secp256k1.h"

#ifdef __cplusplus
extern "C" {
#endif

/* -- X-only public key (64 bytes opaque) ---------------------------------- */
typedef struct secp256k1_xonly_pubkey {
    unsigned char data[64];
} secp256k1_xonly_pubkey;

/* -- Keypair (96 bytes opaque) -------------------------------------------- */
typedef struct secp256k1_keypair {
    unsigned char data[96];
} secp256k1_keypair;

/* -- X-only pubkey operations --------------------------------------------- */
SECP256K1_API int secp256k1_xonly_pubkey_parse(
    const secp256k1_context *ctx, secp256k1_xonly_pubkey *pubkey,
    const unsigned char *input32);

SECP256K1_API int secp256k1_xonly_pubkey_serialize(
    const secp256k1_context *ctx, unsigned char *output32,
    const secp256k1_xonly_pubkey *pubkey);

SECP256K1_API int secp256k1_xonly_pubkey_cmp(
    const secp256k1_context *ctx,
    const secp256k1_xonly_pubkey *pk1,
    const secp256k1_xonly_pubkey *pk2);

SECP256K1_API int secp256k1_xonly_pubkey_from_pubkey(
    const secp256k1_context *ctx, secp256k1_xonly_pubkey *xonly_pubkey,
    int *pk_parity, const secp256k1_pubkey *pubkey);

/* -- Keypair operations --------------------------------------------------- */
SECP256K1_API int secp256k1_keypair_create(
    const secp256k1_context *ctx, secp256k1_keypair *keypair,
    const unsigned char *seckey);

SECP256K1_API int secp256k1_keypair_sec(
    const secp256k1_context *ctx, unsigned char *seckey,
    const secp256k1_keypair *keypair);

SECP256K1_API int secp256k1_keypair_pub(
    const secp256k1_context *ctx, secp256k1_pubkey *pubkey,
    const secp256k1_keypair *keypair);

SECP256K1_API int secp256k1_keypair_xonly_pub(
    const secp256k1_context *ctx, secp256k1_xonly_pubkey *pubkey,
    int *pk_parity, const secp256k1_keypair *keypair);

/* -- Taproot tweak operations --------------------------------------------- */

/** Compute tweaked public key: output = internal_pubkey + tweak32 * G.
 *  output_pubkey: full secp256k1_pubkey (X || Y).
 *  Returns 1 on success, 0 if tweak is invalid or result is infinity. */
SECP256K1_API int secp256k1_xonly_pubkey_tweak_add(
    const secp256k1_context *ctx,
    secp256k1_pubkey *output_pubkey,
    const secp256k1_xonly_pubkey *internal_pubkey,
    const unsigned char *tweak32);

/** Verify Taproot tweak commitment.
 *  Returns 1 if tweaked_pubkey32 (with tweaked_pk_parity) == internal_pubkey + tweak32 * G. */
SECP256K1_API int secp256k1_xonly_pubkey_tweak_add_check(
    const secp256k1_context *ctx,
    const unsigned char *tweaked_pubkey32,
    int tweaked_pk_parity,
    const secp256k1_xonly_pubkey *internal_pubkey,
    const unsigned char *tweak32);

/** Tweak a keypair in-place for Taproot key-path spending.
 *  If the internal pubkey has odd Y, negates the secret key first, then adds tweak.
 *  Returns 1 on success, 0 if tweak is invalid or result is infinity. */
SECP256K1_API int secp256k1_keypair_xonly_tweak_add(
    const secp256k1_context *ctx,
    secp256k1_keypair *keypair,
    const unsigned char *tweak32);

#ifdef __cplusplus
}
#endif

#endif /* SECP256K1_EXTRAKEYS_ULTRAFAST_H */
