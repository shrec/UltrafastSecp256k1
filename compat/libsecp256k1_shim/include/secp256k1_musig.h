#ifndef SECP256K1_MUSIG_ULTRAFAST_H
#define SECP256K1_MUSIG_ULTRAFAST_H

#include "secp256k1.h"
#include "secp256k1_extrakeys.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque data structures matching libsecp256k1 byte-layout guarantees.
   Internal layout is implementation-defined; do not read/write directly. */

typedef struct secp256k1_musig_keyagg_cache { unsigned char data[197]; } secp256k1_musig_keyagg_cache;
typedef struct secp256k1_musig_secnonce    { unsigned char data[132]; } secp256k1_musig_secnonce;
typedef struct secp256k1_musig_pubnonce    { unsigned char data[132]; } secp256k1_musig_pubnonce;
typedef struct secp256k1_musig_aggnonce    { unsigned char data[132]; } secp256k1_musig_aggnonce;
/* WARNING: secp256k1_musig_session contains internal heap pointers (bytes 98-105).
   It is NOT safely serializable (memcpy to disk/network will produce dangling pointers
   on restore). It is NOT safely memcpy-able across process boundaries.
   Use secp256k1_musig_session_save/load for persistence across calls. */
typedef struct secp256k1_musig_session     { unsigned char data[133]; } secp256k1_musig_session;
typedef struct secp256k1_musig_partial_sig { unsigned char data[36];  } secp256k1_musig_partial_sig;

/* Key aggregation -----------------------------------------------------------*/

SECP256K1_API int secp256k1_musig_pubkey_agg(
    const secp256k1_context *ctx,
    secp256k1_xonly_pubkey *agg_pk,
    secp256k1_musig_keyagg_cache *keyagg_cache,
    const secp256k1_pubkey * const *pubkeys,
    size_t n_pubkeys
) SECP256K1_ARG_NONNULL(1) SECP256K1_ARG_NONNULL(3) SECP256K1_ARG_NONNULL(4);

SECP256K1_API int secp256k1_musig_pubkey_get(
    const secp256k1_context *ctx,
    secp256k1_pubkey *agg_pk,
    const secp256k1_musig_keyagg_cache *keyagg_cache
) SECP256K1_ARG_NONNULL(1) SECP256K1_ARG_NONNULL(2) SECP256K1_ARG_NONNULL(3);

SECP256K1_API int secp256k1_musig_pubkey_ec_tweak_add(
    const secp256k1_context *ctx,
    secp256k1_pubkey *output_pubkey,
    secp256k1_musig_keyagg_cache *keyagg_cache,
    const unsigned char *tweak32
) SECP256K1_ARG_NONNULL(1) SECP256K1_ARG_NONNULL(3) SECP256K1_ARG_NONNULL(4);

SECP256K1_API int secp256k1_musig_pubkey_xonly_tweak_add(
    const secp256k1_context *ctx,
    secp256k1_pubkey *output_pubkey,
    secp256k1_musig_keyagg_cache *keyagg_cache,
    const unsigned char *tweak32
) SECP256K1_ARG_NONNULL(1) SECP256K1_ARG_NONNULL(3) SECP256K1_ARG_NONNULL(4);

/* Nonce generation ----------------------------------------------------------*/

SECP256K1_API int secp256k1_musig_nonce_gen(
    const secp256k1_context *ctx,
    secp256k1_musig_secnonce *secnonce,
    secp256k1_musig_pubnonce *pubnonce,
    const unsigned char *session_id32,
    const unsigned char *seckey,
    const secp256k1_pubkey *pubkey,
    const unsigned char *msg32,
    const secp256k1_musig_keyagg_cache *keyagg_cache,
    const unsigned char *extra_input32
) SECP256K1_ARG_NONNULL(1) SECP256K1_ARG_NONNULL(2) SECP256K1_ARG_NONNULL(3)
  SECP256K1_ARG_NONNULL(4);

SECP256K1_API int secp256k1_musig_pubnonce_serialize(
    const secp256k1_context *ctx,
    unsigned char *out66,
    const secp256k1_musig_pubnonce *nonce
) SECP256K1_ARG_NONNULL(1) SECP256K1_ARG_NONNULL(2) SECP256K1_ARG_NONNULL(3);

SECP256K1_API int secp256k1_musig_pubnonce_parse(
    const secp256k1_context *ctx,
    secp256k1_musig_pubnonce *nonce,
    const unsigned char *in66
) SECP256K1_ARG_NONNULL(1) SECP256K1_ARG_NONNULL(2) SECP256K1_ARG_NONNULL(3);

SECP256K1_API int secp256k1_musig_nonce_agg(
    const secp256k1_context *ctx,
    secp256k1_musig_aggnonce *aggnonce,
    const secp256k1_musig_pubnonce * const *pubnonces,
    size_t n_pubnonces
) SECP256K1_ARG_NONNULL(1) SECP256K1_ARG_NONNULL(2) SECP256K1_ARG_NONNULL(3);

/* Session signing -----------------------------------------------------------*/

SECP256K1_API int secp256k1_musig_nonce_process(
    const secp256k1_context *ctx,
    secp256k1_musig_session *session,
    const secp256k1_musig_aggnonce *aggnonce,
    const unsigned char *msg32,
    const secp256k1_musig_keyagg_cache *keyagg_cache
) SECP256K1_ARG_NONNULL(1) SECP256K1_ARG_NONNULL(2) SECP256K1_ARG_NONNULL(3)
  SECP256K1_ARG_NONNULL(4) SECP256K1_ARG_NONNULL(5);

SECP256K1_API int secp256k1_musig_partial_sign(
    const secp256k1_context *ctx,
    secp256k1_musig_partial_sig *partial_sig,
    secp256k1_musig_secnonce *secnonce,
    const secp256k1_keypair *keypair,
    const secp256k1_musig_keyagg_cache *keyagg_cache,
    const secp256k1_musig_session *session
) SECP256K1_ARG_NONNULL(1) SECP256K1_ARG_NONNULL(2) SECP256K1_ARG_NONNULL(3)
  SECP256K1_ARG_NONNULL(4) SECP256K1_ARG_NONNULL(5) SECP256K1_ARG_NONNULL(6);

SECP256K1_API int secp256k1_musig_partial_sig_verify(
    const secp256k1_context *ctx,
    const secp256k1_musig_partial_sig *partial_sig,
    const secp256k1_musig_pubnonce *pubnonce,
    const secp256k1_pubkey *pubkey,
    const secp256k1_musig_keyagg_cache *keyagg_cache,
    const secp256k1_musig_session *session
) SECP256K1_ARG_NONNULL(1) SECP256K1_ARG_NONNULL(2) SECP256K1_ARG_NONNULL(3)
  SECP256K1_ARG_NONNULL(4) SECP256K1_ARG_NONNULL(5) SECP256K1_ARG_NONNULL(6);

SECP256K1_API int secp256k1_musig_partial_sig_agg(
    const secp256k1_context *ctx,
    unsigned char *sig64,
    const secp256k1_musig_session *session,
    const secp256k1_musig_partial_sig * const *partial_sigs,
    size_t n_sigs
) SECP256K1_ARG_NONNULL(1) SECP256K1_ARG_NONNULL(2) SECP256K1_ARG_NONNULL(3)
  SECP256K1_ARG_NONNULL(4);

SECP256K1_API int secp256k1_musig_partial_sig_serialize(
    const secp256k1_context *ctx,
    unsigned char *out32,
    const secp256k1_musig_partial_sig *partial_sig
) SECP256K1_ARG_NONNULL(1) SECP256K1_ARG_NONNULL(2) SECP256K1_ARG_NONNULL(3);

SECP256K1_API int secp256k1_musig_partial_sig_parse(
    const secp256k1_context *ctx,
    secp256k1_musig_partial_sig *partial_sig,
    const unsigned char *in32
) SECP256K1_ARG_NONNULL(1) SECP256K1_ARG_NONNULL(2) SECP256K1_ARG_NONNULL(3);

/* Optional shim extension: explicitly release the internal state for a
 * keyagg_cache abandoned before partial_sig_agg (e.g. on protocol abort).
 * Calling on an already-cleared or never-initialised cache is a no-op.
 * No counterpart in upstream libsecp256k1; shim-only API. */
SECP256K1_API void secp256k1_musig_keyagg_cache_clear(
    secp256k1_musig_keyagg_cache *keyagg_cache
) SECP256K1_ARG_NONNULL(1);

#ifdef __cplusplus
}
#endif

#endif /* SECP256K1_MUSIG_ULTRAFAST_H */
