/* ============================================================================
 * UltrafastSecp256k1 -- Stable C ABI
 * ============================================================================
 *
 * SINGLE HEADER that exposes the entire public C API.
 * Include only this file from application / binding code.
 *
 * ## Design principles
 *
 *   1. Opaque context (`ufsecp_ctx*`) -- all state lives here.
 *   2. Every function returns `ufsecp_error_t` (0 = OK).
 *   3. No internal types leak -- all I/O is `uint8_t[]` with documented sizes.
 *   4. ABI version checked at link time via `ufsecp_abi_version()`.
 *   5. Thread safety: each ctx is single-thread; create one per thread or
 *      protect externally.
 *   6. Dual-layer constant-time: secret-dependent operations (scalar mul,
 *      nonce gen, key tweak) ALWAYS use the CT layer; public operations
 *      (verification, point serialisation) ALWAYS use the fast layer.
 *      Both layers are architecturally wired -- no flag, no opt-in.
 *      This eliminates the human factor entirely.
 *
 * ## Naming
 *
 *   ufsecp_<noun>_<verb>()   e.g. ufsecp_ecdsa_sign()
 *   UFSECP_<CONSTANT>        e.g. UFSECP_PUBKEY_COMPRESSED_LEN
 *
 * ## Memory
 *
 *   Caller always owns output buffers.
 *   Library never allocates on behalf of caller (except ctx create/clone).
 *
 * ============================================================================ */

#ifndef UFSECP_H
#define UFSECP_H

#include "ufsecp_version.h"
#include "ufsecp_error.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -- Size constants --------------------------------------------------------- */

#define UFSECP_PRIVKEY_LEN          32
#define UFSECP_PUBKEY_COMPRESSED_LEN 33
#define UFSECP_PUBKEY_UNCOMPRESSED_LEN 65
#define UFSECP_PUBKEY_XONLY_LEN     32
#define UFSECP_SIG_COMPACT_LEN      64  /* R||S for ECDSA, r||s for Schnorr */
#define UFSECP_SIG_DER_MAX_LEN      72
#define UFSECP_HASH_LEN             32
#define UFSECP_HASH160_LEN          20
#define UFSECP_SHARED_SECRET_LEN    32
#define UFSECP_BIP32_SERIALIZED_LEN 78

/* -- Network constants ------------------------------------------------------ */

#define UFSECP_NET_MAINNET  0
#define UFSECP_NET_TESTNET  1

/* ===========================================================================
 * Context
 * ===========================================================================
 *
 * Constant-time safety is architectural, NOT flag-based.
 *
 *   +-------------------------------------------------------------+
 *   |  Layer 1 -- FAST:  public operations (verify, point arith)  |
 *   |  Layer 2 -- CT  :  secret operations (sign, nonce, tweak)   |
 *   |  Both layers are ALWAYS ACTIVE simultaneously.              |
 *   |  No opt-in / opt-out.  Human-error-proof by design.        |
 *   +-------------------------------------------------------------+
 *
 * CT layer guarantees:
 *   - Complete addition formula (branchless, 12M+2S)
 *   - Fixed-trace scalar multiplication (no early exit)
 *   - CT table lookup (scans all entries)
 *   - Valgrind/MSAN verifiable (SECP256K1_CLASSIFY / DECLASSIFY)
 *
 * =========================================================================== */

/** Opaque context handle.  One per thread (or externally synchronised). */
typedef struct ufsecp_ctx ufsecp_ctx;

/** Create a new context.
 *  Runs library self-test on first call (cached globally).
 *  Both fast and CT layers are always active -- no flags needed.
 *  @param ctx_out  receives the new context pointer.
 *  @return UFSECP_OK on success. */
UFSECP_API ufsecp_error_t ufsecp_ctx_create(ufsecp_ctx** ctx_out);

/** Clone an existing context (deep copy). */
UFSECP_API ufsecp_error_t ufsecp_ctx_clone(const ufsecp_ctx* src,
                                           ufsecp_ctx** ctx_out);

/** Destroy context and free resources. NULL is safe. */
UFSECP_API void ufsecp_ctx_destroy(ufsecp_ctx* ctx);

/** Last error code on this context (0 = none). */
UFSECP_API ufsecp_error_t ufsecp_last_error(const ufsecp_ctx* ctx);

/** Last error message on this context (never NULL). */
UFSECP_API const char* ufsecp_last_error_msg(const ufsecp_ctx* ctx);

/** Size of the compiled ufsecp_ctx struct (for FFI layout assertions). */
UFSECP_API size_t ufsecp_ctx_size(void);

/* ===========================================================================
 * Private key utilities
 * =========================================================================== */

/** Verify that privkey[32] is valid (non-zero, < order).
 *  Returns UFSECP_OK if valid, UFSECP_ERR_BAD_KEY otherwise. */
UFSECP_API ufsecp_error_t ufsecp_seckey_verify(const ufsecp_ctx* ctx,
                                               const uint8_t privkey[32]);

/** Negate privkey in-place: key <- -key mod n. */
UFSECP_API ufsecp_error_t ufsecp_seckey_negate(ufsecp_ctx* ctx,
                                               uint8_t privkey[32]);

/** privkey <- (privkey + tweak) mod n. */
UFSECP_API ufsecp_error_t ufsecp_seckey_tweak_add(ufsecp_ctx* ctx,
                                                  uint8_t privkey[32],
                                                  const uint8_t tweak[32]);

/** privkey <- (privkey x tweak) mod n. */
UFSECP_API ufsecp_error_t ufsecp_seckey_tweak_mul(ufsecp_ctx* ctx,
                                                  uint8_t privkey[32],
                                                  const uint8_t tweak[32]);

/* ===========================================================================
 * Public key
 * =========================================================================== */

/** Derive compressed public key (33 bytes) from private key. */
UFSECP_API ufsecp_error_t ufsecp_pubkey_create(ufsecp_ctx* ctx,
                                               const uint8_t privkey[32],
                                               uint8_t pubkey33_out[33]);

/** Derive uncompressed public key (65 bytes) from private key. */
UFSECP_API ufsecp_error_t ufsecp_pubkey_create_uncompressed(
    ufsecp_ctx* ctx,
    const uint8_t privkey[32],
    uint8_t pubkey65_out[65]);

/** Parse any public key (33 compressed or 65 uncompressed).
 *  Output is always 33-byte compressed. */
UFSECP_API ufsecp_error_t ufsecp_pubkey_parse(ufsecp_ctx* ctx,
                                              const uint8_t* input,
                                              size_t input_len,
                                              uint8_t pubkey33_out[33]);

/** Derive x-only (32 bytes, BIP-340) public key from private key. */
UFSECP_API ufsecp_error_t ufsecp_pubkey_xonly(ufsecp_ctx* ctx,
                                              const uint8_t privkey[32],
                                              uint8_t xonly32_out[32]);

/* ===========================================================================
 * ECDSA (secp256k1, RFC 6979 deterministic nonce)
 * =========================================================================== */

/** Sign a 32-byte hash. Output: 64-byte compact R||S (low-S normalised). */
UFSECP_API ufsecp_error_t ufsecp_ecdsa_sign(ufsecp_ctx* ctx,
                                            const uint8_t msg32[32],
                                            const uint8_t privkey[32],
                                            uint8_t sig64_out[64]);

/** Sign + verify (FIPS 186-4 fault attack countermeasure).
 *  Verifies the produced signature before returning it.
 *  Use this when fault injection resistance is required. */
UFSECP_API ufsecp_error_t ufsecp_ecdsa_sign_verified(ufsecp_ctx* ctx,
                                                     const uint8_t msg32[32],
                                                     const uint8_t privkey[32],
                                                     uint8_t sig64_out[64]);

/** Verify an ECDSA compact signature.
 *  Returns UFSECP_OK if valid, UFSECP_ERR_VERIFY_FAIL if invalid. */
UFSECP_API ufsecp_error_t ufsecp_ecdsa_verify(ufsecp_ctx* ctx,
                                              const uint8_t msg32[32],
                                              const uint8_t sig64[64],
                                              const uint8_t pubkey33[33]);

/** Encode compact sig to DER.
 *  der_len: in = buffer size (>=72), out = actual DER length. */
UFSECP_API ufsecp_error_t ufsecp_ecdsa_sig_to_der(ufsecp_ctx* ctx,
                                                   const uint8_t sig64[64],
                                                   uint8_t* der_out,
                                                   size_t* der_len);

/** Decode DER-encoded sig back to compact 64 bytes. */
UFSECP_API ufsecp_error_t ufsecp_ecdsa_sig_from_der(ufsecp_ctx* ctx,
                                                    const uint8_t* der,
                                                    size_t der_len,
                                                    uint8_t sig64_out[64]);

/* -- ECDSA recovery --------------------------------------------------------- */

/** Sign with recovery id.
 *  recid_out: recovery id (0-3). */
UFSECP_API ufsecp_error_t ufsecp_ecdsa_sign_recoverable(
    ufsecp_ctx* ctx,
    const uint8_t msg32[32],
    const uint8_t privkey[32],
    uint8_t sig64_out[64],
    int* recid_out);

/** Recover public key from an ECDSA recoverable signature. */
UFSECP_API ufsecp_error_t ufsecp_ecdsa_recover(ufsecp_ctx* ctx,
                                               const uint8_t msg32[32],
                                               const uint8_t sig64[64],
                                               int recid,
                                               uint8_t pubkey33_out[33]);

/* ===========================================================================
 * Schnorr / BIP-340
 * =========================================================================== */

/** BIP-340 Schnorr sign.
 *  aux_rand: 32 bytes auxiliary randomness (all-zeros for deterministic). */
UFSECP_API ufsecp_error_t ufsecp_schnorr_sign(ufsecp_ctx* ctx,
                                              const uint8_t msg32[32],
                                              const uint8_t privkey[32],
                                              const uint8_t aux_rand[32],
                                              uint8_t sig64_out[64]);

/** BIP-340 Schnorr sign + verify (FIPS 186-4 fault attack countermeasure).
 *  Verifies the produced signature before returning it. */
UFSECP_API ufsecp_error_t ufsecp_schnorr_sign_verified(ufsecp_ctx* ctx,
                                                       const uint8_t msg32[32],
                                                       const uint8_t privkey[32],
                                                       const uint8_t aux_rand[32],
                                                       uint8_t sig64_out[64]);

/** BIP-340 Schnorr verify.
 *  pubkey_x: 32-byte x-only public key. */
UFSECP_API ufsecp_error_t ufsecp_schnorr_verify(ufsecp_ctx* ctx,
                                                const uint8_t msg32[32],
                                                const uint8_t sig64[64],
                                                const uint8_t pubkey_x[32]);

/* ===========================================================================
 * ECDH (Diffie-Hellman key agreement)
 * =========================================================================== */

/** ECDH shared secret: SHA256(compressed shared point). */
UFSECP_API ufsecp_error_t ufsecp_ecdh(ufsecp_ctx* ctx,
                                      const uint8_t privkey[32],
                                      const uint8_t pubkey33[33],
                                      uint8_t secret32_out[32]);

/** ECDH x-only: SHA256(x-coordinate). */
UFSECP_API ufsecp_error_t ufsecp_ecdh_xonly(ufsecp_ctx* ctx,
                                            const uint8_t privkey[32],
                                            const uint8_t pubkey33[33],
                                            uint8_t secret32_out[32]);

/** ECDH raw: raw x-coordinate (32 bytes, no hash). */
UFSECP_API ufsecp_error_t ufsecp_ecdh_raw(ufsecp_ctx* ctx,
                                          const uint8_t privkey[32],
                                          const uint8_t pubkey33[33],
                                          uint8_t secret32_out[32]);

/* ===========================================================================
 * Hashing
 * =========================================================================== */

/** SHA-256 (hardware-accelerated when available). */
UFSECP_API ufsecp_error_t ufsecp_sha256(const uint8_t* data, size_t len,
                                        uint8_t digest32_out[32]);

/** RIPEMD160(SHA256(data)) = Hash160. */
UFSECP_API ufsecp_error_t ufsecp_hash160(const uint8_t* data, size_t len,
                                         uint8_t digest20_out[20]);

/** BIP-340 tagged hash. */
UFSECP_API ufsecp_error_t ufsecp_tagged_hash(const char* tag,
                                             const uint8_t* data, size_t len,
                                             uint8_t digest32_out[32]);

/* ===========================================================================
 * Bitcoin addresses
 * =========================================================================== */

/** P2PKH address from compressed pubkey.
 *  addr_len: in = buffer size, out = strlen (excl. NUL). */
UFSECP_API ufsecp_error_t ufsecp_addr_p2pkh(ufsecp_ctx* ctx,
                                            const uint8_t pubkey33[33],
                                            int network,
                                            char* addr_out, size_t* addr_len);

/** P2WPKH (Bech32, SegWit v0). */
UFSECP_API ufsecp_error_t ufsecp_addr_p2wpkh(ufsecp_ctx* ctx,
                                             const uint8_t pubkey33[33],
                                             int network,
                                             char* addr_out, size_t* addr_len);

/** P2TR (Bech32m, Taproot) from x-only internal key. */
UFSECP_API ufsecp_error_t ufsecp_addr_p2tr(ufsecp_ctx* ctx,
                                           const uint8_t internal_key_x[32],
                                           int network,
                                           char* addr_out, size_t* addr_len);

/* ===========================================================================
 * WIF (Wallet Import Format)
 * =========================================================================== */

/** Encode private key -> WIF string.
 *  wif_len: in = buf size, out = strlen. */
UFSECP_API ufsecp_error_t ufsecp_wif_encode(ufsecp_ctx* ctx,
                                            const uint8_t privkey[32],
                                            int compressed, int network,
                                            char* wif_out, size_t* wif_len);

/** Decode WIF string -> private key. */
UFSECP_API ufsecp_error_t ufsecp_wif_decode(ufsecp_ctx* ctx,
                                            const char* wif,
                                            uint8_t privkey32_out[32],
                                            int* compressed_out,
                                            int* network_out);

/* ===========================================================================
 * BIP-32 (HD key derivation)
 * =========================================================================== */

/** Opaque serialised BIP-32 extended key. */
typedef struct {
    uint8_t data[UFSECP_BIP32_SERIALIZED_LEN];
    uint8_t is_private;   /**< 1 = xprv, 0 = xpub */
    uint8_t _pad[3];      /**< Reserved, must be zero */
} ufsecp_bip32_key;

/** Master key from seed (16-64 bytes). */
UFSECP_API ufsecp_error_t ufsecp_bip32_master(ufsecp_ctx* ctx,
                                              const uint8_t* seed, size_t seed_len,
                                              ufsecp_bip32_key* key_out);

/** Normal or hardened child derivation (index >= 0x80000000 = hardened). */
UFSECP_API ufsecp_error_t ufsecp_bip32_derive(ufsecp_ctx* ctx,
                                              const ufsecp_bip32_key* parent,
                                              uint32_t index,
                                              ufsecp_bip32_key* child_out);

/** Full path derivation, e.g. "m/44'/0'/0'/0/0". */
UFSECP_API ufsecp_error_t ufsecp_bip32_derive_path(ufsecp_ctx* ctx,
                                                   const ufsecp_bip32_key* master,
                                                   const char* path,
                                                   ufsecp_bip32_key* key_out);

/** Extract 32-byte private key (fails if xpub). */
UFSECP_API ufsecp_error_t ufsecp_bip32_privkey(ufsecp_ctx* ctx,
                                               const ufsecp_bip32_key* key,
                                               uint8_t privkey32_out[32]);

/** Extract 33-byte compressed public key. */
UFSECP_API ufsecp_error_t ufsecp_bip32_pubkey(ufsecp_ctx* ctx,
                                              const ufsecp_bip32_key* key,
                                              uint8_t pubkey33_out[33]);

/* ===========================================================================
 * Taproot (BIP-341)
 * =========================================================================== */

/** Derive Taproot output key from internal key.
 *  merkle_root: 32 bytes or NULL for key-path-only. */
UFSECP_API ufsecp_error_t ufsecp_taproot_output_key(
    ufsecp_ctx* ctx,
    const uint8_t internal_x[32],
    const uint8_t* merkle_root,
    uint8_t output_x_out[32],
    int* parity_out);

/** Tweak a private key for Taproot key-path spending. */
UFSECP_API ufsecp_error_t ufsecp_taproot_tweak_seckey(
    ufsecp_ctx* ctx,
    const uint8_t privkey[32],
    const uint8_t* merkle_root,
    uint8_t tweaked32_out[32]);

/** Verify Taproot commitment. Returns UFSECP_OK if valid. */
UFSECP_API ufsecp_error_t ufsecp_taproot_verify(
    ufsecp_ctx* ctx,
    const uint8_t output_x[32], int output_parity,
    const uint8_t internal_x[32],
    const uint8_t* merkle_root, size_t merkle_root_len);

#ifdef __cplusplus
}

/* -- ABI layout guards (C++ only) ------------------------------------------ */
/* These fire at compile time if struct layout changes, preventing silent ABI  */
/* breaks when bindings or cached objects assume a fixed layout.               */
static_assert(sizeof(ufsecp_bip32_key) == 82,
              "ABI break: ufsecp_bip32_key size changed (expected 82)");
static_assert(UFSECP_BIP32_SERIALIZED_LEN == 78,
              "ABI break: UFSECP_BIP32_SERIALIZED_LEN changed (expected 78)");
static_assert(UFSECP_PRIVKEY_LEN == 32,
              "ABI break: UFSECP_PRIVKEY_LEN changed");
static_assert(UFSECP_PUBKEY_COMPRESSED_LEN == 33,
              "ABI break: UFSECP_PUBKEY_COMPRESSED_LEN changed");
static_assert(UFSECP_SIG_COMPACT_LEN == 64,
              "ABI break: UFSECP_SIG_COMPACT_LEN changed");
#else
/* C11 _Static_assert equivalent for pure-C consumers */
_Static_assert(sizeof(ufsecp_bip32_key) == 82,
               "ABI break: ufsecp_bip32_key size changed (expected 82)");
#endif

#endif /* UFSECP_H */
