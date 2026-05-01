/* ============================================================================
 * libsecp256k1-compatible C API -- backed by UltrafastSecp256k1
 *
 * This header provides the same types and function declarations as
 * bitcoin-core/secp256k1's secp256k1.h so that existing C/C++ code
 * can link against UltrafastSecp256k1 without source changes.
 * ========================================================================== */
#ifndef SECP256K1_H
#define SECP256K1_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/* -- Visibility ----------------------------------------------------------- */
#ifndef SECP256K1_API
#  define SECP256K1_API extern
#endif
#define SECP256K1_WARN_UNUSED_RESULT
#define SECP256K1_ARG_NONNULL(_x)
#define SECP256K1_DEPRECATED(_msg)

/* -- Opaque context ------------------------------------------------------- */
typedef struct secp256k1_context_struct secp256k1_context;

/* -- Public key (64 bytes opaque) ----------------------------------------- */
typedef struct secp256k1_pubkey {
    unsigned char data[64];
} secp256k1_pubkey;

/* -- ECDSA signature (64 bytes opaque) ------------------------------------ */
typedef struct secp256k1_ecdsa_signature {
    unsigned char data[64];
} secp256k1_ecdsa_signature;

/* -- Nonce function type -------------------------------------------------- */
typedef int (*secp256k1_nonce_function)(
    unsigned char *nonce32,
    const unsigned char *msg32,
    const unsigned char *key32,
    const unsigned char *algo16,
    void *data,
    unsigned int attempt
);

/* -- Flags ---------------------------------------------------------------- */
#define SECP256K1_FLAGS_TYPE_MASK         ((1 << 8) - 1)
#define SECP256K1_FLAGS_TYPE_CONTEXT      (1 << 0)
#define SECP256K1_FLAGS_TYPE_COMPRESSION  (1 << 1)
#define SECP256K1_FLAGS_BIT_CONTEXT_VERIFY    (1 << 8)
#define SECP256K1_FLAGS_BIT_CONTEXT_SIGN      (1 << 9)
#define SECP256K1_FLAGS_BIT_COMPRESSION       (1 << 8)

#define SECP256K1_CONTEXT_NONE     (SECP256K1_FLAGS_TYPE_CONTEXT)
#define SECP256K1_CONTEXT_VERIFY   (SECP256K1_FLAGS_TYPE_CONTEXT | SECP256K1_FLAGS_BIT_CONTEXT_VERIFY)
#define SECP256K1_CONTEXT_SIGN     (SECP256K1_FLAGS_TYPE_CONTEXT | SECP256K1_FLAGS_BIT_CONTEXT_SIGN)

#define SECP256K1_EC_COMPRESSED    (SECP256K1_FLAGS_TYPE_COMPRESSION | SECP256K1_FLAGS_BIT_COMPRESSION)
#define SECP256K1_EC_UNCOMPRESSED  (SECP256K1_FLAGS_TYPE_COMPRESSION)

#define SECP256K1_TAG_PUBKEY_EVEN          0x02
#define SECP256K1_TAG_PUBKEY_ODD           0x03
#define SECP256K1_TAG_PUBKEY_UNCOMPRESSED  0x04

/* -- Illegal/error callback ----------------------------------------------- */
/* Called when an illegal API usage or internal error is detected.
 * The default behavior (when not set) is to call abort(). Pass a no-op
 * function pointer to suppress the default abort (Bitcoin Core pattern). */
typedef void (*secp256k1_callback_fn)(const char *text, void *data);

/* -- Static context ------------------------------------------------------- */
SECP256K1_API const secp256k1_context * const secp256k1_context_static;

/* -- Context lifecycle ---------------------------------------------------- */
SECP256K1_API secp256k1_context *secp256k1_context_create(unsigned int flags);
SECP256K1_API secp256k1_context *secp256k1_context_clone(const secp256k1_context *ctx);
SECP256K1_API void secp256k1_context_destroy(secp256k1_context *ctx);
SECP256K1_API int  secp256k1_context_randomize(secp256k1_context *ctx, const unsigned char *seed32);
SECP256K1_API void secp256k1_selftest(void);

/* Install a callback invoked on illegal API usage (NULL ctx, bad arg, etc.).
 * The default callback calls abort(). Pass a no-op to suppress abort. */
SECP256K1_API void secp256k1_context_set_illegal_callback(
    secp256k1_context *ctx,
    secp256k1_callback_fn fun,
    const void *data);

/* Install a callback for internal library errors (e.g., OOM during context ops).
 * The default callback calls abort(). */
SECP256K1_API void secp256k1_context_set_error_callback(
    secp256k1_context *ctx,
    secp256k1_callback_fn fun,
    const void *data);

/* -- Public key operations ------------------------------------------------ */
SECP256K1_API int secp256k1_ec_pubkey_parse(
    const secp256k1_context *ctx, secp256k1_pubkey *pubkey,
    const unsigned char *input, size_t inputlen);

SECP256K1_API int secp256k1_ec_pubkey_serialize(
    const secp256k1_context *ctx, unsigned char *output, size_t *outputlen,
    const secp256k1_pubkey *pubkey, unsigned int flags);

SECP256K1_API int secp256k1_ec_pubkey_cmp(
    const secp256k1_context *ctx,
    const secp256k1_pubkey *pubkey1, const secp256k1_pubkey *pubkey2);

SECP256K1_API int secp256k1_ec_pubkey_create(
    const secp256k1_context *ctx, secp256k1_pubkey *pubkey,
    const unsigned char *seckey);

SECP256K1_API int secp256k1_ec_pubkey_negate(
    const secp256k1_context *ctx, secp256k1_pubkey *pubkey);

SECP256K1_API int secp256k1_ec_pubkey_tweak_add(
    const secp256k1_context *ctx, secp256k1_pubkey *pubkey,
    const unsigned char *tweak32);

SECP256K1_API int secp256k1_ec_pubkey_tweak_mul(
    const secp256k1_context *ctx, secp256k1_pubkey *pubkey,
    const unsigned char *tweak32);

SECP256K1_API int secp256k1_ec_pubkey_combine(
    const secp256k1_context *ctx, secp256k1_pubkey *out,
    const secp256k1_pubkey * const *ins, size_t n);

SECP256K1_API int secp256k1_ec_pubkey_sort(
    const secp256k1_context *ctx,
    const secp256k1_pubkey **pubkeys,
    size_t n_pubkeys);

/* -- Secret key operations ------------------------------------------------ */
SECP256K1_API int secp256k1_ec_seckey_verify(
    const secp256k1_context *ctx, const unsigned char *seckey);

SECP256K1_API int secp256k1_ec_seckey_negate(
    const secp256k1_context *ctx, unsigned char *seckey);

SECP256K1_API int secp256k1_ec_seckey_tweak_add(
    const secp256k1_context *ctx, unsigned char *seckey,
    const unsigned char *tweak32);

SECP256K1_API int secp256k1_ec_seckey_tweak_mul(
    const secp256k1_context *ctx, unsigned char *seckey,
    const unsigned char *tweak32);

/* -- ECDSA ---------------------------------------------------------------- */
SECP256K1_API int secp256k1_ecdsa_signature_parse_compact(
    const secp256k1_context *ctx, secp256k1_ecdsa_signature *sig,
    const unsigned char *input64);

SECP256K1_API int secp256k1_ecdsa_signature_parse_der(
    const secp256k1_context *ctx, secp256k1_ecdsa_signature *sig,
    const unsigned char *input, size_t inputlen);

SECP256K1_API int secp256k1_ecdsa_signature_serialize_compact(
    const secp256k1_context *ctx, unsigned char *output64,
    const secp256k1_ecdsa_signature *sig);

SECP256K1_API int secp256k1_ecdsa_signature_serialize_der(
    const secp256k1_context *ctx, unsigned char *output, size_t *outputlen,
    const secp256k1_ecdsa_signature *sig);

SECP256K1_API int secp256k1_ecdsa_signature_normalize(
    const secp256k1_context *ctx, secp256k1_ecdsa_signature *sigout,
    const secp256k1_ecdsa_signature *sigin);

SECP256K1_API int secp256k1_ecdsa_verify(
    const secp256k1_context *ctx, const secp256k1_ecdsa_signature *sig,
    const unsigned char *msghash32, const secp256k1_pubkey *pubkey);

/* NOTE (shim divergence): custom noncefp values (other than NULL /
 * secp256k1_nonce_function_rfc6979 / secp256k1_nonce_function_default)
 * are rejected (return 0) rather than silently ignored. RFC 6979 is always
 * used. ndata IS respected as auxiliary entropy (hedged signing path), which
 * is how Bitcoin Core's R-grinding loop works. See BITCOIN_CORE_PR_BLOCKERS.md §B. */
SECP256K1_API int secp256k1_ecdsa_sign(
    const secp256k1_context *ctx, secp256k1_ecdsa_signature *sig,
    const unsigned char *msghash32, const unsigned char *seckey,
    secp256k1_nonce_function noncefp, const void *ndata);

SECP256K1_API const secp256k1_nonce_function secp256k1_nonce_function_rfc6979;
SECP256K1_API const secp256k1_nonce_function secp256k1_nonce_function_default;

/* -- Tagged hash ---------------------------------------------------------- */
SECP256K1_API int secp256k1_tagged_sha256(
    const secp256k1_context *ctx, unsigned char *hash32,
    const unsigned char *tag, size_t taglen,
    const unsigned char *msg, size_t msglen);

#ifdef __cplusplus
}
#endif

/* Pull in ElligatorSwift (BIP-324 v2 transport) -- required by Bitcoin Core */
#include "secp256k1_ellswift.h"

#endif /* SECP256K1_H */
