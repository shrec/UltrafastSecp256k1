/* ============================================================================
 * UltrafastSecp256k1 -- Error Model
 * ============================================================================
 * Every ufsecp_* function returns ufsecp_error_t (0 = success).
 * For detailed diagnostics, call ufsecp_last_error() / ufsecp_last_error_msg()
 * on the context that returned the error.
 *
 * Thread safety: each ufsecp_ctx owns its own last-error slot.
 * ============================================================================ */

#ifndef UFSECP_ERROR_H
#define UFSECP_ERROR_H

#include "ufsecp_version.h"   /* pulls in UFSECP_API */

#ifdef __cplusplus
extern "C" {
#endif

/* -- Error codes ------------------------------------------------------------ */

typedef int ufsecp_error_t;

#define UFSECP_OK                0   /**< Success                              */
#define UFSECP_ERR_NULL_ARG      1   /**< Required pointer argument was NULL   */
#define UFSECP_ERR_BAD_KEY       2   /**< Invalid private key (zero, >= order) */
#define UFSECP_ERR_BAD_PUBKEY    3   /**< Invalid / unparseable public key     */
#define UFSECP_ERR_BAD_SIG       4   /**< Invalid / malformed signature        */
#define UFSECP_ERR_BAD_INPUT     5   /**< Malformed input (wrong length, etc.) */
#define UFSECP_ERR_VERIFY_FAIL   6   /**< Signature verification failed        */
#define UFSECP_ERR_ARITH         7   /**< Scalar/field arithmetic overflow     */
#define UFSECP_ERR_SELFTEST      8   /**< Library self-test failed             */
#define UFSECP_ERR_INTERNAL      9   /**< Unexpected internal error            */
#define UFSECP_ERR_BUF_TOO_SMALL 10  /**< Output buffer too small              */
#define UFSECP_ERR_NOT_FOUND     11  /**< Item not found (e.g. GCS filter match) */

/* -- Error inspection ------------------------------------------------------- */

/** Map error code to a short English description (never NULL). */
UFSECP_API const char* ufsecp_error_str(ufsecp_error_t err);

/*
 * Per-context error inspection (see ufsecp.h for ufsecp_ctx):
 *   ufsecp_error_t  ufsecp_last_error    (const ufsecp_ctx* ctx);
 *   const char*     ufsecp_last_error_msg(const ufsecp_ctx* ctx);
 */

#ifdef __cplusplus
}
#endif

#endif /* UFSECP_ERROR_H */
