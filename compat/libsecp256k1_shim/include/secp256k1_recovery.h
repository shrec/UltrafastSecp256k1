#ifndef SECP256K1_RECOVERY_ULTRAFAST_H
#define SECP256K1_RECOVERY_ULTRAFAST_H

// ============================================================================
// secp256k1_recovery.h -- libsecp256k1-compatible ECDSA recovery module
// ============================================================================
// Provides secp256k1_ecdsa_recoverable_signature, secp256k1_ecdsa_sign_recoverable,
// and secp256k1_ecdsa_recover as a drop-in replacement for the corresponding
// libsecp256k1 recovery module (include/secp256k1_recovery.h).
//
// Used by: ECIES implementations, Ethereum ecrecover, light clients that need
// to recover the signer's public key from an (r, s, v) tuple without storing it.
//
// All signing uses the CT path (ct::ecdsa_sign_recoverable) internally -- the
// nonce and private key are never processed by variable-time code.
// ============================================================================

#include "secp256k1.h"

#ifdef __cplusplus
extern "C" {
#endif

// -- Opaque recoverable signature type ----------------------------------------
// Layout: [recid: 1 byte] [r: 32 bytes] [s: 32 bytes] = 65 bytes, zero-padded
// to match libsecp256k1's 65-byte internal representation.
typedef struct {
    unsigned char data[65];
} secp256k1_ecdsa_recoverable_signature;

// -- Parse / Serialize --------------------------------------------------------

// Parse a compact (64-byte) recoverable signature.
// recid must be 0, 1, 2, or 3.
// Returns 1 on success, 0 on failure.
int secp256k1_ecdsa_recoverable_signature_parse_compact(
    const secp256k1_context *ctx,
    secp256k1_ecdsa_recoverable_signature *sig,
    const unsigned char *input64,
    int recid);

// Serialize a recoverable signature to compact (64 bytes) + output recid.
// Returns 1 on success.
int secp256k1_ecdsa_recoverable_signature_serialize_compact(
    const secp256k1_context *ctx,
    unsigned char *output64,
    int *recid,
    const secp256k1_ecdsa_recoverable_signature *sig);

// Convert a recoverable signature to a non-recoverable one (drops recid).
// Returns 1 on success.
int secp256k1_ecdsa_recoverable_signature_convert(
    const secp256k1_context *ctx,
    secp256k1_ecdsa_signature *sig,
    const secp256k1_ecdsa_recoverable_signature *sigin);

// -- Sign (recoverable) -------------------------------------------------------

// Sign a 32-byte message hash and produce a recoverable signature.
// Uses ct::ecdsa_sign_recoverable internally -- constant-time on privkey/nonce.
// noncefp and ndata are accepted for ABI compatibility but ignored (RFC 6979 is
// always used).
// Returns 1 on success, 0 on failure (null args, zeroed privkey).
int secp256k1_ecdsa_sign_recoverable(
    const secp256k1_context *ctx,
    secp256k1_ecdsa_recoverable_signature *sig,
    const unsigned char *msghash32,
    const unsigned char *seckey,
    secp256k1_nonce_function noncefp,
    const void *ndata);

// -- Recover public key -------------------------------------------------------

// Recover the public key from a recoverable signature and the original message
// hash.  Needed by Ethereum ecrecover, ECIES, and light-client implementations.
// Returns 1 if recovery succeeded, 0 otherwise.
int secp256k1_ecdsa_recover(
    const secp256k1_context *ctx,
    secp256k1_pubkey *pubkey,
    const secp256k1_ecdsa_recoverable_signature *sig,
    const unsigned char *msghash32);

#ifdef __cplusplus
}
#endif

#endif // SECP256K1_RECOVERY_H
