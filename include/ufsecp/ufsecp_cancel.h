#ifndef UFSECP_CANCEL_H
#define UFSECP_CANCEL_H

/* ufsecp_cancel.h — shared caller-driven cancellation token.
 *
 * Single canonical definition of the cancellation token used by the long-running
 * batch entry points across UltrafastSecp256k1: the libbitcoin acceleration
 * bridge (ufsecp_libbitcoin.h) and the libsecp256k1-compatible shim batch-verify
 * surface (secp256k1_batch.h). Both include THIS header so they agree on layout —
 * a program may include both without a conflicting redefinition.
 *
 * Semantics:
 *   - A NULL token (or NULL is_cancelled) means "no cancellation" and preserves
 *     the original, byte-for-byte behavior with zero overhead.
 *   - A non-NULL token is polled between work chunks. check_interval == 0 selects
 *     the engine default chunk size; otherwise it is the number of items between
 *     polls (rounded up to the batch minimum where smaller would be pathological).
 *   - Returning non-zero from is_cancelled aborts the operation FAIL-CLOSED: a
 *     cancelled batch never reports success. Cancellation is an execution status,
 *     NOT a verdict — any partial results must be discarded by the caller.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Caller cancellation predicate. Returns non-zero to request cancellation.
 * `user` is the opaque pointer carried in the token (never dereferenced here). */
typedef int (*ufsecp_cancel_fn)(const void* user);

typedef struct ufsecp_cancel_token {
    ufsecp_cancel_fn is_cancelled;   /* NULL => never cancelled                  */
    const void*      user;           /* opaque, passed back to is_cancelled      */
    uint32_t         check_interval; /* items between polls; 0 => engine default */
} ufsecp_cancel_token;

#ifdef __cplusplus
} /* extern "C" */
#endif

/* Convenience for C++ default arguments on cancellation-aware declarations:
 *   int f(..., const ufsecp_cancel_token* cancel UFSECP_CANCEL_DEFAULT);
 * In C it expands to nothing (C has no default arguments — pass NULL). */
#ifdef __cplusplus
#define UFSECP_CANCEL_DEFAULT = nullptr
#else
#define UFSECP_CANCEL_DEFAULT
#endif

#endif /* UFSECP_CANCEL_H */
