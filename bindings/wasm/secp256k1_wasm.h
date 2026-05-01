/* ============================================================================
 * UltrafastSecp256k1 -- WebAssembly C API
 * ============================================================================
 * Flat C interface suitable for Emscripten export and JS/TS interop.
 * All buffers are caller-owned, fixed-size, and use big-endian byte order
 * (standard hex encoding).
 *
 * Return convention:
 *   1  = success
 *   0  = failure (invalid input, verification failed, etc.)
 * ============================================================================
 */
#ifndef SECP256K1_WASM_H
#define SECP256K1_WASM_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -- Library Info ----------------------------------------------------------- */

/** Run built-in self-test. Returns 1 on success, 0 on failure. */
int secp256k1_wasm_selftest(void);

/** Return library version as a static string (e.g. "3.0.0"). */
const char* secp256k1_wasm_version(void);

/* -- Key Generation --------------------------------------------------------- */

/**
 * Compute public key from private key: P = privkey * G.
 *
 * @param seckey32    [in]  32-byte private key (big-endian)
 * @param pubkey_x32  [out] 32-byte X coordinate (big-endian)
 * @param pubkey_y32  [out] 32-byte Y coordinate (big-endian)
 * @return 1 on success, 0 if seckey is zero or >= curve order
 */
int secp256k1_wasm_pubkey_create(const uint8_t* seckey32,
                                  uint8_t* pubkey_x32,
                                  uint8_t* pubkey_y32);

/* -- Point Arithmetic ------------------------------------------------------- */

/**
 * Scalar x Point multiplication: R = scalar * P.
 *
 * @param point_x32  [in]  32-byte X of point P (big-endian)
 * @param point_y32  [in]  32-byte Y of point P (big-endian)
 * @param scalar32   [in]  32-byte scalar (big-endian)
 * @param out_x32    [out] 32-byte X of result R (big-endian)
 * @param out_y32    [out] 32-byte Y of result R (big-endian)
 * @return 1 on success, 0 if point is invalid or scalar is zero
 */
int secp256k1_wasm_point_mul(const uint8_t* point_x32,
                              const uint8_t* point_y32,
                              const uint8_t* scalar32,
                              uint8_t* out_x32,
                              uint8_t* out_y32);

/**
 * Point addition: R = P + Q.
 *
 * @param p_x32   [in]  32-byte X of point P
 * @param p_y32   [in]  32-byte Y of point P
 * @param q_x32   [in]  32-byte X of point Q
 * @param q_y32   [in]  32-byte Y of point Q
 * @param out_x32 [out] 32-byte X of result R
 * @param out_y32 [out] 32-byte Y of result R
 * @return 1 on success
 */
int secp256k1_wasm_point_add(const uint8_t* p_x32, const uint8_t* p_y32,
                              const uint8_t* q_x32, const uint8_t* q_y32,
                              uint8_t* out_x32, uint8_t* out_y32);

/* -- ECDSA ------------------------------------------------------------------ */

/**
 * ECDSA sign (RFC 6979 deterministic nonce, low-S normalized).
 *
 * @param msg32     [in]  32-byte message hash
 * @param seckey32  [in]  32-byte private key
 * @param sig64     [out] 64-byte compact signature (r || s)
 * @return 1 on success, 0 on failure
 */
int secp256k1_wasm_ecdsa_sign(const uint8_t* msg32,
                               const uint8_t* seckey32,
                               uint8_t* sig64);

/**
 * ECDSA verify.
 *
 * @param msg32       [in] 32-byte message hash
 * @param pubkey_x32  [in] 32-byte public key X
 * @param pubkey_y32  [in] 32-byte public key Y
 * @param sig64       [in] 64-byte compact signature (r || s)
 * @return 1 if valid, 0 if invalid
 */
int secp256k1_wasm_ecdsa_verify(const uint8_t* msg32,
                                 const uint8_t* pubkey_x32,
                                 const uint8_t* pubkey_y32,
                                 const uint8_t* sig64);

/* -- Schnorr BIP-340 -------------------------------------------------------- */

/**
 * Schnorr BIP-340 sign.
 *
 * @param seckey32  [in]  32-byte private key
 * @param msg32     [in]  32-byte message
 * @param aux32     [in]  32-byte auxiliary randomness (may be all zeros)
 * @param sig64     [out] 64-byte signature (R.x || s)
 * @return 1 on success, 0 on failure
 */
int secp256k1_wasm_schnorr_sign(const uint8_t* seckey32,
                                 const uint8_t* msg32,
                                 const uint8_t* aux32,
                                 uint8_t* sig64);

/**
 * Schnorr BIP-340 verify.
 *
 * @param pubkey_x32  [in] 32-byte x-only public key
 * @param msg32       [in] 32-byte message
 * @param sig64       [in] 64-byte signature (R.x || s)
 * @return 1 if valid, 0 if invalid
 */
int secp256k1_wasm_schnorr_verify(const uint8_t* pubkey_x32,
                                   const uint8_t* msg32,
                                   const uint8_t* sig64);

/**
 * Derive x-only public key for Schnorr (BIP-340).
 *
 * @param seckey32     [in]  32-byte private key
 * @param pubkey_x32   [out] 32-byte x-only public key
 * @return 1 on success, 0 on failure
 */
int secp256k1_wasm_schnorr_pubkey(const uint8_t* seckey32,
                                   uint8_t* pubkey_x32);

/* -- SHA-256 ---------------------------------------------------------------- */

/**
 * SHA-256 hash.
 *
 * @param data    [in]  Input data
 * @param len     [in]  Length in bytes
 * @param out32   [out] 32-byte hash output
 */
void secp256k1_wasm_sha256(const uint8_t* data, size_t len, uint8_t* out32);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SECP256K1_WASM_H */
