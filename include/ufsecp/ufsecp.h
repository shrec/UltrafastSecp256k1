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
 *      This removes flag-based routing mistakes inside the core ABI.
 *      Bindings and callers must still enforce context ownership, buffer
 *      lifetime, and secret-handling discipline.
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
 *   |  No opt-in / opt-out in the core ABI.                       |
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

/** Last error message on this context (never NULL).
 *  The returned pointer is borrowed storage owned by ctx.
 *  It remains valid until the next call that mutates the same ctx, or until
 *  ufsecp_ctx_destroy(ctx). Copy it if it must outlive the context/call. */
UFSECP_API const char* ufsecp_last_error_msg(const ufsecp_ctx* ctx);

/** Size of the compiled ufsecp_ctx struct (for FFI layout assertions). */
UFSECP_API size_t ufsecp_ctx_size(void);

/** Randomize scalar blinding for constant-time signing operations.
 *
 *  Installs a fresh random blinding scalar r derived from seed32 into the
 *  calling thread's blinding state.  Subsequent signing calls (ECDSA, Schnorr,
 *  recoverable ECDSA) compute (k+r)*G - r*G instead of k*G, protecting against
 *  DPA and fault-injection attacks without changing the output signature.
 *
 *  @param ctx    valid context (used only for error reporting).
 *  @param seed32 32 bytes of entropy.  Pass NULL to clear blinding.
 *  @return UFSECP_OK on success.
 *
 *  Thread safety: blinding state is thread-local; each thread must call this
 *  independently.  A single context may be shared by multiple threads as long
 *  as each thread randomizes its own blinding state.
 *
 *  Recommended: call once after context creation and periodically thereafter
 *  (e.g. every 2^32 signing operations or whenever a new session begins). */
UFSECP_API ufsecp_error_t ufsecp_context_randomize(ufsecp_ctx*    ctx,
                                                    const uint8_t* seed32);

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
 * Batch signing (CPU constant-time dispatch -- private keys never leave host)
 * =========================================================================== */

/** ECDSA sign a batch of messages.
 *  Signs each (msgs32[i], privkeys32[i]) pair in order using the CT sign path.
 *  The private key for each entry is immediately erased from memory after use.
 *  Returns on the first failure; already-written entries remain valid.
 *
 *  @param ctx         CPU context.
 *  @param count       Number of (message, key) pairs.
 *  @param msgs32      Input: count * 32 bytes (message hashes, contiguous).
 *  @param privkeys32  Input: count * 32 bytes (private keys, contiguous).
 *  @param sigs64_out  Output: count * 64 bytes (compact R||S per entry). */
UFSECP_API ufsecp_error_t ufsecp_ecdsa_sign_batch(
    ufsecp_ctx* ctx,
    size_t count,
    const uint8_t* msgs32,
    const uint8_t* privkeys32,
    uint8_t* sigs64_out);

/** BIP-340 Schnorr sign a batch of messages.
 *  Signs each (msgs32[i], privkeys32[i], aux_rands32[i]) triple in order.
 *  The private key for each entry is immediately erased from memory after use.
 *  Returns on the first failure; already-written entries remain valid.
 *
 *  @param ctx         CPU context.
 *  @param count       Number of (message, key) pairs.
 *  @param msgs32      Input: count * 32 bytes (message hashes, contiguous).
 *  @param privkeys32  Input: count * 32 bytes (private keys, contiguous).
 *  @param aux_rands32 Input: count * 32 bytes (aux randomness); pass NULL to
 *                     use all-zero aux for every entry.
 *  @param sigs64_out  Output: count * 64 bytes (BIP-340 Schnorr signatures). */
UFSECP_API ufsecp_error_t ufsecp_schnorr_sign_batch(
    ufsecp_ctx* ctx,
    size_t count,
    const uint8_t* msgs32,
    const uint8_t* privkeys32,
    const uint8_t* aux_rands32,
    uint8_t* sigs64_out);

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

/** P2SH address from arbitrary redeem script.
 *  addr_len: in = buffer size (min 36), out = strlen (excl. NUL). */
UFSECP_API ufsecp_error_t ufsecp_addr_p2sh(
    const uint8_t* redeem_script, size_t redeem_script_len,
    int network,
    char* addr_out, size_t* addr_len);

/** P2SH-P2WPKH (WrappedSegWit) address from compressed pubkey.
 *  addr_len: in = buffer size (min 36), out = strlen (excl. NUL). */
UFSECP_API ufsecp_error_t ufsecp_addr_p2sh_p2wpkh(
    ufsecp_ctx* ctx,
    const uint8_t pubkey33[33],
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

/** Opaque serialised BIP-32 extended key.
 *  Reserved bytes must remain zero.
 *  Caller-supplied keys are rejected unless is_private, serialized version bytes,
 *  and serialized key material form a valid xprv/xpub encoding. */
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

/* ===========================================================================
 * BIP-143: SegWit v0 Sighash
 * =========================================================================== */

/** Compute BIP-143 sighash digest for a SegWit v0 input.
 *  hash_prevouts, hash_sequence, hash_outputs: precomputed 32-byte hashes.
 *  outpoint_txid: 32-byte LE txid of the input being signed.
 *  outpoint_vout: output index of the input being signed.
 *  script_code / script_code_len: the scriptCode for this input.
 *  value: satoshi amount of the output being spent.
 *  sequence: nSequence of this input.
 *  sighash_type: SIGHASH_ALL etc. */
UFSECP_API ufsecp_error_t ufsecp_bip143_sighash(
    ufsecp_ctx* ctx,
    uint32_t version,
    const uint8_t hash_prevouts[32],
    const uint8_t hash_sequence[32],
    const uint8_t outpoint_txid[32], uint32_t outpoint_vout,
    const uint8_t* script_code, size_t script_code_len,
    uint64_t value,
    uint32_t sequence,
    const uint8_t hash_outputs[32],
    uint32_t locktime,
    uint32_t sighash_type,
    uint8_t sighash_out[32]);

/** Build P2WPKH scriptCode (25 bytes) from a 20-byte pubkey hash. */
UFSECP_API ufsecp_error_t ufsecp_bip143_p2wpkh_script_code(
    const uint8_t pubkey_hash[20],
    uint8_t script_code_out[25]);

/* ===========================================================================
 * BIP-144: Witness Transaction Serialization
 * =========================================================================== */

/** Compute txid (legacy hash, no witness) from raw witness-serialized tx.
 *  raw_tx/raw_tx_len: complete witness-format transaction bytes.
 *  txid_out: 32-byte LE txid. */
UFSECP_API ufsecp_error_t ufsecp_bip144_txid(
    ufsecp_ctx* ctx,
    const uint8_t* raw_tx, size_t raw_tx_len,
    uint8_t txid_out[32]);

/** Compute wtxid from raw witness-serialized transaction bytes. */
UFSECP_API ufsecp_error_t ufsecp_bip144_wtxid(
    ufsecp_ctx* ctx,
    const uint8_t* raw_tx, size_t raw_tx_len,
    uint8_t wtxid_out[32]);

/** Compute witness commitment: SHA256d(witness_root || witness_nonce). */
UFSECP_API ufsecp_error_t ufsecp_bip144_witness_commitment(
    const uint8_t witness_root[32],
    const uint8_t witness_nonce[32],
    uint8_t commitment_out[32]);

/* ===========================================================================
 * BIP-141: Segregated Witness — Witness Programs
 * =========================================================================== */

/** Check if a scriptPubKey is a witness program. Returns 1 if yes, 0 if no. */
UFSECP_API int ufsecp_segwit_is_witness_program(
    const uint8_t* script, size_t script_len);

/** Parse a witness program from a scriptPubKey.
 *  version_out: witness version (0-16), or -1 if not a witness program.
 *  program_out: buffer for the program (at least 40 bytes).
 *  program_len_out: actual program length.
 *  Returns UFSECP_OK on success, UFSECP_ERR_BAD_INPUT if not a witness program. */
UFSECP_API ufsecp_error_t ufsecp_segwit_parse_program(
    const uint8_t* script, size_t script_len,
    int* version_out,
    uint8_t* program_out, size_t* program_len_out);

/** Build P2WPKH scriptPubKey (22 bytes) from 20-byte pubkey hash. */
UFSECP_API ufsecp_error_t ufsecp_segwit_p2wpkh_spk(
    const uint8_t pubkey_hash[20],
    uint8_t spk_out[22]);

/** Build P2WSH scriptPubKey (34 bytes) from 32-byte script hash. */
UFSECP_API ufsecp_error_t ufsecp_segwit_p2wsh_spk(
    const uint8_t script_hash[32],
    uint8_t spk_out[34]);

/** Build P2TR scriptPubKey (34 bytes) from 32-byte x-only output key. */
UFSECP_API ufsecp_error_t ufsecp_segwit_p2tr_spk(
    const uint8_t output_key[32],
    uint8_t spk_out[34]);

/** Compute SHA256 of witness script (for P2WSH program). */
UFSECP_API ufsecp_error_t ufsecp_segwit_witness_script_hash(
    const uint8_t* script, size_t script_len,
    uint8_t hash_out[32]);

/* ===========================================================================
 * BIP-342: Tapscript Sighash
 * =========================================================================== */

/** Compute BIP-341 key-path sighash.
 *  All input prevout txids, vouts, amounts, sequences, and scriptPubKeys
 *  must be provided as flat arrays. */
UFSECP_API ufsecp_error_t ufsecp_taproot_keypath_sighash(
    ufsecp_ctx* ctx,
    uint32_t version, uint32_t locktime,
    size_t input_count,
    const uint8_t* prevout_txids,   /* input_count*32 bytes, flattened */
    const uint32_t* prevout_vouts,
    const uint64_t* input_amounts,
    const uint32_t* input_sequences,
    const uint8_t* const* input_spks,
    const size_t* input_spk_lens,
    size_t output_count,
    const uint64_t* output_values,
    const uint8_t* const* output_spks,
    const size_t* output_spk_lens,
    size_t input_index,
    uint8_t hash_type,
    const uint8_t* annex, size_t annex_len,
    uint8_t sighash_out[32]);

/** Compute BIP-342 tapscript sighash. Same as key-path + extension data. */
UFSECP_API ufsecp_error_t ufsecp_tapscript_sighash(
    ufsecp_ctx* ctx,
    uint32_t version, uint32_t locktime,
    size_t input_count,
    const uint8_t* prevout_txids,
    const uint32_t* prevout_vouts,
    const uint64_t* input_amounts,
    const uint32_t* input_sequences,
    const uint8_t* const* input_spks,
    const size_t* input_spk_lens,
    size_t output_count,
    const uint64_t* output_values,
    const uint8_t* const* output_spks,
    const size_t* output_spk_lens,
    size_t input_index,
    uint8_t hash_type,
    const uint8_t tapleaf_hash[32],
    uint8_t key_version,
    uint32_t code_separator_pos,
    const uint8_t* annex, size_t annex_len,
    uint8_t sighash_out[32]);

/* ===========================================================================
 * Ethereum (conditional: SECP256K1_BUILD_ETHEREUM)
 * =========================================================================== */

/* ===========================================================================
 * Public key arithmetic
 * =========================================================================== */

/** Add two compressed public keys: out = a + b. */
UFSECP_API ufsecp_error_t ufsecp_pubkey_add(ufsecp_ctx* ctx,
                                            const uint8_t a33[33],
                                            const uint8_t b33[33],
                                            uint8_t out33[33]);

/** Negate a compressed public key: out = -P. */
UFSECP_API ufsecp_error_t ufsecp_pubkey_negate(ufsecp_ctx* ctx,
                                               const uint8_t pubkey33[33],
                                               uint8_t out33[33]);

/** Tweak-add a public key: out = P + tweak*G. */
UFSECP_API ufsecp_error_t ufsecp_pubkey_tweak_add(ufsecp_ctx* ctx,
                                                  const uint8_t pubkey33[33],
                                                  const uint8_t tweak[32],
                                                  uint8_t out33[33]);

/** Tweak-mul a public key: out = tweak * P. */
UFSECP_API ufsecp_error_t ufsecp_pubkey_tweak_mul(ufsecp_ctx* ctx,
                                                  const uint8_t pubkey33[33],
                                                  const uint8_t tweak[32],
                                                  uint8_t out33[33]);

/** Combine N compressed public keys: out = sum(pubkeys[i]).
 *  pubkeys: array of 33-byte compressed keys, contiguous.
 *  The total contiguous byte span n * 33 must fit in size_t. */
UFSECP_API ufsecp_error_t ufsecp_pubkey_combine(ufsecp_ctx* ctx,
                                                const uint8_t* pubkeys,
                                                size_t n,
                                                uint8_t out33[33]);

/* ===========================================================================
 * BIP-39 (Mnemonic seed phrases)
 * =========================================================================== */

/** Generate BIP-39 mnemonic from entropy.
 *  entropy_bytes: 16 (12 words), 20 (15), 24 (18), 28 (21), 32 (24 words).
 *  entropy_in: NULL for random, or pointer to entropy bytes.
 *  mnemonic_out: buffer for NUL-terminated mnemonic.
 *  mnemonic_len: in = buffer size, out = strlen. */
UFSECP_API ufsecp_error_t ufsecp_bip39_generate(ufsecp_ctx* ctx,
                                                size_t entropy_bytes,
                                                const uint8_t* entropy_in,
                                                char* mnemonic_out,
                                                size_t* mnemonic_len);

/** Validate BIP-39 mnemonic (checksum + word list).
 *  Returns UFSECP_OK if valid, UFSECP_ERR_BAD_INPUT if invalid. */
UFSECP_API ufsecp_error_t ufsecp_bip39_validate(const ufsecp_ctx* ctx,
                                                const char* mnemonic);

/** Convert mnemonic to 64-byte seed (PBKDF2-HMAC-SHA512, 2048 rounds).
 *  passphrase: optional BIP-39 passphrase (NULL or "" for none). */
UFSECP_API ufsecp_error_t ufsecp_bip39_to_seed(ufsecp_ctx* ctx,
                                               const char* mnemonic,
                                               const char* passphrase,
                                               uint8_t seed64_out[64]);

/** Convert mnemonic back to raw entropy bytes.
 *  entropy_out: buffer (>=32 bytes).
 *  entropy_len: out = actual entropy length. */
UFSECP_API ufsecp_error_t ufsecp_bip39_to_entropy(ufsecp_ctx* ctx,
                                                  const char* mnemonic,
                                                  uint8_t* entropy_out,
                                                  size_t* entropy_len);

/* ===========================================================================
 * Batch verification
 * =========================================================================== */

/** Schnorr batch verify: verify N signatures in one call.
 *  Each entry: [32-byte xonly pubkey | 32-byte msg | 64-byte sig] = 128 bytes.
 *  Returns UFSECP_OK if ALL valid. */
UFSECP_API ufsecp_error_t ufsecp_schnorr_batch_verify(
    ufsecp_ctx* ctx,
    const uint8_t* entries, size_t n);

/** ECDSA batch verify: verify N signatures in one call.
 *  Each entry: [32-byte msg | 33-byte pubkey | 64-byte sig] = 129 bytes.
 *  Returns UFSECP_OK if ALL valid. */
UFSECP_API ufsecp_error_t ufsecp_ecdsa_batch_verify(
    ufsecp_ctx* ctx,
    const uint8_t* entries, size_t n);

/** Schnorr batch identify invalid: returns indices of invalid sigs.
 *  invalid_out: caller-owned array of size_t.
 *  invalid_count: in = invalid_out capacity, out = total number of invalid entries. */
UFSECP_API ufsecp_error_t ufsecp_schnorr_batch_identify_invalid(
    ufsecp_ctx* ctx,
    const uint8_t* entries, size_t n,
    size_t* invalid_out, size_t* invalid_count);

/** ECDSA batch identify invalid: returns indices of invalid sigs.
 *  invalid_out: caller-owned array of size_t.
 *  invalid_count: in = invalid_out capacity, out = total number of invalid entries. */
UFSECP_API ufsecp_error_t ufsecp_ecdsa_batch_identify_invalid(
    ufsecp_ctx* ctx,
    const uint8_t* entries, size_t n,
    size_t* invalid_out, size_t* invalid_count);

/* ===========================================================================
 * SHA-512
 * =========================================================================== */

/** SHA-512 hash. */
UFSECP_API ufsecp_error_t ufsecp_sha512(const uint8_t* data, size_t len,
                                        uint8_t digest64_out[64]);

/* ===========================================================================
 * Multi-scalar multiplication
 * =========================================================================== */

/** Shamir's trick: compute a*P + b*Q.
 *  All scalars are 32-byte big-endian. All points are 33-byte compressed. */
UFSECP_API ufsecp_error_t ufsecp_shamir_trick(
    ufsecp_ctx* ctx,
    const uint8_t a[32], const uint8_t P33[33],
    const uint8_t b[32], const uint8_t Q33[33],
    uint8_t out33[33]);

/** Multi-scalar multiplication: compute sum(scalars[i] * points[i]).
 *  scalars: n * 32 bytes contiguous. points: n * 33 bytes contiguous.
 *  Both contiguous byte spans must fit in size_t. */
UFSECP_API ufsecp_error_t ufsecp_multi_scalar_mul(
    ufsecp_ctx* ctx,
    const uint8_t* scalars, const uint8_t* points, size_t n,
    uint8_t out33[33]);

/* ===========================================================================
 * MuSig2 (BIP-327 multi-signatures)
 * =========================================================================== */

#define UFSECP_MUSIG2_PUBNONCE_LEN   66  /**< 33 + 33 bytes */
#define UFSECP_MUSIG2_AGGNONCE_LEN   66
#define UFSECP_MUSIG2_KEYAGG_LEN     165 /**< opaque serialised key agg context */
#define UFSECP_MUSIG2_SESSION_LEN    165 /**< opaque serialised session state, including participant count metadata */
#define UFSECP_MUSIG2_SECNONCE_LEN   64  /**< secret nonce (2 x 32 bytes) */

/** Aggregate public keys for MuSig2.
 *  pubkeys: n * 32 bytes (x-only). keyagg_out: opaque context.
 *  The current fixed-size keyagg/session format supports 2 to 3 participants. */
UFSECP_API ufsecp_error_t ufsecp_musig2_key_agg(
    ufsecp_ctx* ctx,
    const uint8_t* pubkeys, size_t n,
    uint8_t keyagg_out[UFSECP_MUSIG2_KEYAGG_LEN],
    uint8_t agg_pubkey32_out[32]);

/** Generate MuSig2 nonce pair. */
UFSECP_API ufsecp_error_t ufsecp_musig2_nonce_gen(
    ufsecp_ctx* ctx,
    const uint8_t privkey[32],
    const uint8_t pubkey32[32],
    const uint8_t agg_pubkey32[32],
    const uint8_t msg32[32],
    const uint8_t extra_in[32],
    uint8_t secnonce_out[UFSECP_MUSIG2_SECNONCE_LEN],
    uint8_t pubnonce_out[UFSECP_MUSIG2_PUBNONCE_LEN]);

/** Aggregate public nonces.
 *  pubnonces must contain exactly n records of UFSECP_MUSIG2_PUBNONCE_LEN bytes.
 *  Each record must contain two valid 33-byte compressed curve points.
 *  n must be at least 2. */
UFSECP_API ufsecp_error_t ufsecp_musig2_nonce_agg(
    ufsecp_ctx* ctx,
    const uint8_t* pubnonces, size_t n,
    uint8_t aggnonce_out[UFSECP_MUSIG2_AGGNONCE_LEN]);

/** Start a MuSig2 signing session.
 *  keyagg must be a valid opaque context previously produced by ufsecp_musig2_key_agg.
 *  session_out binds the participant count from keyagg and must later be paired
 *  with exactly the same signer set arity during partial signing, verification,
 *  and aggregation. */
UFSECP_API ufsecp_error_t ufsecp_musig2_start_sign_session(
    ufsecp_ctx* ctx,
    const uint8_t aggnonce[UFSECP_MUSIG2_AGGNONCE_LEN],
    const uint8_t keyagg[UFSECP_MUSIG2_KEYAGG_LEN],
    const uint8_t msg32[32],
    uint8_t session_out[UFSECP_MUSIG2_SESSION_LEN]);

/** Produce a partial signature.
 *  IMPORTANT: secnonce is zeroed after use to prevent nonce reuse.
 *  keyagg must be a valid opaque context previously produced by ufsecp_musig2_key_agg.
 *  signer_index must be a valid participant index within the aggregated key set.
 *  session must carry the same participant count as keyagg. */
UFSECP_API ufsecp_error_t ufsecp_musig2_partial_sign(
    ufsecp_ctx* ctx,
    uint8_t secnonce[UFSECP_MUSIG2_SECNONCE_LEN],
    const uint8_t privkey[32],
    const uint8_t keyagg[UFSECP_MUSIG2_KEYAGG_LEN],
    const uint8_t session[UFSECP_MUSIG2_SESSION_LEN],
    size_t signer_index,
    uint8_t partial_sig32_out[32]);

/** Verify a partial signature.
 *  keyagg must be a valid opaque context previously produced by ufsecp_musig2_key_agg.
 *  signer_index must be a valid participant index within the aggregated key set.
 *  session must carry the same participant count as keyagg. */
UFSECP_API ufsecp_error_t ufsecp_musig2_partial_verify(
    ufsecp_ctx* ctx,
    const uint8_t partial_sig32[32],
    const uint8_t pubnonce[UFSECP_MUSIG2_PUBNONCE_LEN],
    const uint8_t pubkey32[32],
    const uint8_t keyagg[UFSECP_MUSIG2_KEYAGG_LEN],
    const uint8_t session[UFSECP_MUSIG2_SESSION_LEN],
    size_t signer_index);

/** Aggregate partial signatures into a final BIP-340 Schnorr signature.
 *  partial_sigs must contain exactly n records of 32 bytes.
 *  n must be non-zero and must match the participant count bound into session. */
UFSECP_API ufsecp_error_t ufsecp_musig2_partial_sig_agg(
    ufsecp_ctx* ctx,
    const uint8_t* partial_sigs, size_t n,
    const uint8_t session[UFSECP_MUSIG2_SESSION_LEN],
    uint8_t sig64_out[64]);

/* ===========================================================================
 * FROST (Threshold signatures)
 * =========================================================================== */

#define UFSECP_FROST_SHARE_LEN         36   /**< 4 (from) + 32 (value) */
#define UFSECP_FROST_KEYPKG_LEN        141  /**< serialised key package */
#define UFSECP_FROST_NONCE_LEN         64   /**< hiding + binding nonce */
#define UFSECP_FROST_NONCE_COMMIT_LEN  70   /**< id + hiding_pt + binding_pt */

/** FROST key generation phase 1: produce commitment + shares.
 *  participant_id must be in [1, num_participants] and threshold must satisfy
 *  2 <= threshold <= num_participants.
 *  commits_out must have room for 8 + threshold * 33 bytes.
 *  shares_out must have room for num_participants * UFSECP_FROST_SHARE_LEN bytes.
 *  commits_out: commitment blob. shares_out: n shares of UFSECP_FROST_SHARE_LEN each. */
UFSECP_API ufsecp_error_t ufsecp_frost_keygen_begin(
    ufsecp_ctx* ctx,
    uint32_t participant_id, uint32_t threshold, uint32_t num_participants,
    const uint8_t seed[32],
    uint8_t* commits_out, size_t* commits_len,
    uint8_t* shares_out, size_t* shares_len);

/** FROST key generation phase 2: finalise key package.
 *  participant_id must be in [1, num_participants] and threshold must satisfy
 *  2 <= threshold <= num_participants.
 *  all_commits length must equal num_participants * (8 + threshold * 33) and
 *  must contain exactly num_participants unique commitment records, each with
 *  exactly threshold coefficients.
 *  received_shares length must equal num_participants * UFSECP_FROST_SHARE_LEN
 *  and must contain exactly num_participants unique share records. */
UFSECP_API ufsecp_error_t ufsecp_frost_keygen_finalize(
    ufsecp_ctx* ctx,
    uint32_t participant_id,
    const uint8_t* all_commits, size_t commits_len,
    const uint8_t* received_shares, size_t shares_len,
    uint32_t threshold, uint32_t num_participants,
    uint8_t keypkg_out[UFSECP_FROST_KEYPKG_LEN]);

/** Generate FROST signing nonce.
 *  participant_id must be non-zero and use the protocol's 1-based participant numbering. */
UFSECP_API ufsecp_error_t ufsecp_frost_sign_nonce_gen(
    ufsecp_ctx* ctx,
    uint32_t participant_id,
    const uint8_t nonce_seed[32],
    uint8_t nonce_out[UFSECP_FROST_NONCE_LEN],
    uint8_t nonce_commit_out[UFSECP_FROST_NONCE_COMMIT_LEN]);

/** Produce FROST partial signature.
 *  nonce_commits must contain exactly n_signers records of
 *  UFSECP_FROST_NONCE_COMMIT_LEN bytes and n_signers must be non-zero.
 *  Signer IDs in nonce_commits must be unique, within the key package participant
 *  range, and must include the caller's own participant ID exactly once. */
UFSECP_API ufsecp_error_t ufsecp_frost_sign(
    ufsecp_ctx* ctx,
    const uint8_t keypkg[UFSECP_FROST_KEYPKG_LEN],
    uint8_t nonce[UFSECP_FROST_NONCE_LEN],
    const uint8_t msg32[32],
    const uint8_t* nonce_commits, size_t n_signers,
    uint8_t partial_sig_out[36]);

/** Verify FROST partial signature.
 *  verification_share33: 33-byte compressed signer verification share Y_i.
 *  nonce_commits must contain exactly n_signers records of
 *  UFSECP_FROST_NONCE_COMMIT_LEN bytes and n_signers must be non-zero.
 *  partial_sig[0..3] and all nonce commitment signer IDs must be non-zero and
 *  unique, and partial_sig's signer ID must appear exactly once in
 *  nonce_commits. */
UFSECP_API ufsecp_error_t ufsecp_frost_verify_partial(
    ufsecp_ctx* ctx,
    const uint8_t partial_sig[36],
    const uint8_t verification_share33[33],
    const uint8_t* nonce_commits, size_t n_signers,
    const uint8_t msg32[32],
    const uint8_t group_pubkey33[33]);

/** Aggregate FROST partial signatures into final Schnorr signature.
 *  partial_sigs must contain exactly n records of 36 bytes.
 *  nonce_commits must contain exactly n_signers records of
 *  UFSECP_FROST_NONCE_COMMIT_LEN bytes.
 *  Both n and n_signers must be non-zero and must describe the same signer set.
 *  Partial signature IDs and nonce commitment IDs must be unique, non-zero, and
 *  each partial signature signer must appear exactly once in nonce_commits. */
UFSECP_API ufsecp_error_t ufsecp_frost_aggregate(
    ufsecp_ctx* ctx,
    const uint8_t* partial_sigs, size_t n,
    const uint8_t* nonce_commits, size_t n_signers,
    const uint8_t group_pubkey33[33],
    const uint8_t msg32[32],
    uint8_t sig64_out[64]);

/* ===========================================================================
 * Adaptor signatures (Atomic swaps / DLCs)
 * =========================================================================== */

#define UFSECP_SCHNORR_ADAPTOR_SIG_LEN 97  /**< 33 R_hat + 32 s_hat + 32 proof */
#define UFSECP_ECDSA_ADAPTOR_SIG_LEN   130 /**< 33 R_hat + 32 s_hat + 33 r_proof + 32 dleq_e */

/** BIP-340 Schnorr adaptor pre-sign. adaptor_point: 33-byte compressed. */
UFSECP_API ufsecp_error_t ufsecp_schnorr_adaptor_sign(
    ufsecp_ctx* ctx,
    const uint8_t privkey[32],
    const uint8_t msg32[32],
    const uint8_t adaptor_point33[33],
    const uint8_t aux_rand[32],
    uint8_t pre_sig_out[UFSECP_SCHNORR_ADAPTOR_SIG_LEN]);

/** Verify Schnorr adaptor pre-signature. */
UFSECP_API ufsecp_error_t ufsecp_schnorr_adaptor_verify(
    ufsecp_ctx* ctx,
    const uint8_t pre_sig[UFSECP_SCHNORR_ADAPTOR_SIG_LEN],
    const uint8_t pubkey_x[32],
    const uint8_t msg32[32],
    const uint8_t adaptor_point33[33]);

/** Adapt a Schnorr pre-signature into a valid signature. */
UFSECP_API ufsecp_error_t ufsecp_schnorr_adaptor_adapt(
    ufsecp_ctx* ctx,
    const uint8_t pre_sig[UFSECP_SCHNORR_ADAPTOR_SIG_LEN],
    const uint8_t adaptor_secret[32],
    uint8_t sig64_out[64]);

/** Extract adaptor secret from pre-signature + completed signature. */
UFSECP_API ufsecp_error_t ufsecp_schnorr_adaptor_extract(
    ufsecp_ctx* ctx,
    const uint8_t pre_sig[UFSECP_SCHNORR_ADAPTOR_SIG_LEN],
    const uint8_t sig64[64],
    uint8_t secret32_out[32]);

/** ECDSA adaptor pre-sign. */
UFSECP_API ufsecp_error_t ufsecp_ecdsa_adaptor_sign(
    ufsecp_ctx* ctx,
    const uint8_t privkey[32],
    const uint8_t msg32[32],
    const uint8_t adaptor_point33[33],
    uint8_t pre_sig_out[UFSECP_ECDSA_ADAPTOR_SIG_LEN]);

/** Verify ECDSA adaptor pre-signature. */
UFSECP_API ufsecp_error_t ufsecp_ecdsa_adaptor_verify(
    ufsecp_ctx* ctx,
    const uint8_t pre_sig[UFSECP_ECDSA_ADAPTOR_SIG_LEN],
    const uint8_t pubkey33[33],
    const uint8_t msg32[32],
    const uint8_t adaptor_point33[33]);

/** Adapt ECDSA pre-signature into valid signature. */
UFSECP_API ufsecp_error_t ufsecp_ecdsa_adaptor_adapt(
    ufsecp_ctx* ctx,
    const uint8_t pre_sig[UFSECP_ECDSA_ADAPTOR_SIG_LEN],
    const uint8_t adaptor_secret[32],
    uint8_t sig64_out[64]);

/** Extract adaptor secret from ECDSA pre-sig + completed sig. */
UFSECP_API ufsecp_error_t ufsecp_ecdsa_adaptor_extract(
    ufsecp_ctx* ctx,
    const uint8_t pre_sig[UFSECP_ECDSA_ADAPTOR_SIG_LEN],
    const uint8_t sig64[64],
    uint8_t secret32_out[32]);

/* ===========================================================================
 * Pedersen commitments
 * =========================================================================== */

/** Pedersen commitment: C = value * H + blinding * G.
 *  commitment33_out: 33-byte compressed point. */
UFSECP_API ufsecp_error_t ufsecp_pedersen_commit(
    ufsecp_ctx* ctx,
    const uint8_t value[32],
    const uint8_t blinding[32],
    uint8_t commitment33_out[33]);

/** Verify Pedersen commitment. */
UFSECP_API ufsecp_error_t ufsecp_pedersen_verify(
    ufsecp_ctx* ctx,
    const uint8_t commitment33[33],
    const uint8_t value[32],
    const uint8_t blinding[32]);

/** Verify that sum of positive commitments equals sum of negative commitments.
 *  pos/neg: arrays of 33-byte compressed commitments. */
UFSECP_API ufsecp_error_t ufsecp_pedersen_verify_sum(
    ufsecp_ctx* ctx,
    const uint8_t* pos, size_t n_pos,
    const uint8_t* neg, size_t n_neg);

/** Compute blinding sum: sum(in) - sum(out).
 *  blinds: all blindings contiguous (32 bytes each), first n_in are inputs. */
UFSECP_API ufsecp_error_t ufsecp_pedersen_blind_sum(
    ufsecp_ctx* ctx,
    const uint8_t* blinds_in, size_t n_in,
    const uint8_t* blinds_out, size_t n_out,
    uint8_t sum32_out[32]);

/** Switch commitment: C = value*H + blinding*G + switch_blind*J. */
UFSECP_API ufsecp_error_t ufsecp_pedersen_switch_commit(
    ufsecp_ctx* ctx,
    const uint8_t value[32],
    const uint8_t blinding[32],
    const uint8_t switch_blind[32],
    uint8_t commitment33_out[33]);

/* ===========================================================================
 * Zero-knowledge proofs
 * =========================================================================== */

#define UFSECP_ZK_KNOWLEDGE_PROOF_LEN  64  /**< 32 rx + 32 s */
#define UFSECP_ZK_DLEQ_PROOF_LEN       64  /**< 32 e + 32 s */
#define UFSECP_ZK_RANGE_PROOF_MAX_LEN  688 /**< max Bulletproof range proof */

/** Knowledge proof: prove knowledge of discrete log. */
UFSECP_API ufsecp_error_t ufsecp_zk_knowledge_prove(
    ufsecp_ctx* ctx,
    const uint8_t secret[32],
    const uint8_t pubkey33[33],
    const uint8_t msg32[32],
    const uint8_t aux_rand[32],
    uint8_t proof_out[UFSECP_ZK_KNOWLEDGE_PROOF_LEN]);

/** Verify knowledge proof. */
UFSECP_API ufsecp_error_t ufsecp_zk_knowledge_verify(
    ufsecp_ctx* ctx,
    const uint8_t proof[UFSECP_ZK_KNOWLEDGE_PROOF_LEN],
    const uint8_t pubkey33[33],
    const uint8_t msg32[32]);

/** DLEQ proof: prove that P/G == Q/H (same discrete log).
 *  G, H, P, Q: 33-byte compressed points. */
UFSECP_API ufsecp_error_t ufsecp_zk_dleq_prove(
    ufsecp_ctx* ctx,
    const uint8_t secret[32],
    const uint8_t G33[33], const uint8_t H33[33],
    const uint8_t P33[33], const uint8_t Q33[33],
    const uint8_t aux_rand[32],
    uint8_t proof_out[UFSECP_ZK_DLEQ_PROOF_LEN]);

/** Verify DLEQ proof. */
UFSECP_API ufsecp_error_t ufsecp_zk_dleq_verify(
    ufsecp_ctx* ctx,
    const uint8_t proof[UFSECP_ZK_DLEQ_PROOF_LEN],
    const uint8_t G33[33], const uint8_t H33[33],
    const uint8_t P33[33], const uint8_t Q33[33]);

/** Bulletproof range proof: prove commitment hides value in [0, 2^64).
 *  proof_len: in = buffer size, out = actual proof size. */
UFSECP_API ufsecp_error_t ufsecp_zk_range_prove(
    ufsecp_ctx* ctx,
    uint64_t value,
    const uint8_t blinding[32],
    const uint8_t commitment33[33],
    const uint8_t aux_rand[32],
    uint8_t* proof_out, size_t* proof_len);

/** Verify Bulletproof range proof.
 *  proof must be exactly one serialized range-proof record. */
UFSECP_API ufsecp_error_t ufsecp_zk_range_verify(
    ufsecp_ctx* ctx,
    const uint8_t commitment33[33],
    const uint8_t* proof, size_t proof_len);

/** Foreign-field 5×52-bit limb decomposition (eprint 2025/695).
 *  5 × uint64_t, each holding ≤ 52 bits of the field/scalar value. */
typedef struct {
    uint64_t limbs[5];  /**< little-endian 52-bit limbs */
} ufsecp_ff_limbs_t;

/** ECDSA-in-SNARK prover witness  (eprint 2025/695 §5).
 *
 *  Contains every intermediate value required by a PLONK circuit to verify
 *  one secp256k1 ECDSA signature using foreign-field arithmetic.
 *  Values are provided in canonical 32-byte big-endian encoding AND in
 *  5×52-bit limb form (ufsecp_ff_limbs_t) for direct PLONK gate wiring.
 *
 *  ECDSA verify circuit steps:
 *    s_inv = sig_s^{-1} mod n
 *    u1    = msg * s_inv mod n
 *    u2    = sig_r * s_inv mod n
 *    R     = u1*G + u2*pubkey
 *    valid = (R ≠ ∞) AND (R.x mod n == sig_r)
 */
typedef struct {
    /* ── public inputs ─────────────────────────────────────────────────── */
    uint8_t msg[32];       /**< message hash e (big-endian)   */
    uint8_t sig_r[32];     /**< signature r                   */
    uint8_t sig_s[32];     /**< signature s                   */
    uint8_t pub_x[32];     /**< public key P.x (Fp element)   */
    uint8_t pub_y[32];     /**< public key P.y (Fp element)   */

    /* ── private witness (circuit signals) ─────────────────────────────── */
    uint8_t s_inv[32];             /**< s^{-1} mod n           */
    uint8_t u1[32];                /**< e * s^{-1} mod n       */
    uint8_t u2[32];                /**< r * s^{-1} mod n       */
    uint8_t result_x[32];          /**< R.x (Fp)               */
    uint8_t result_y[32];          /**< R.y (Fp)               */
    uint8_t result_x_mod_n[32];    /**< R.x mod n (Fr)         */

    /* ── 5×52-bit foreign-field limb decompositions ─────────────────────── */
    ufsecp_ff_limbs_t lmb_sig_r;           /**< sig_r limbs       */
    ufsecp_ff_limbs_t lmb_sig_s;           /**< sig_s limbs       */
    ufsecp_ff_limbs_t lmb_pub_x;           /**< pub_x limbs       */
    ufsecp_ff_limbs_t lmb_pub_y;           /**< pub_y limbs       */
    ufsecp_ff_limbs_t lmb_s_inv;           /**< s_inv limbs       */
    ufsecp_ff_limbs_t lmb_u1;              /**< u1 limbs          */
    ufsecp_ff_limbs_t lmb_u2;              /**< u2 limbs          */
    ufsecp_ff_limbs_t lmb_result_x;        /**< R.x limbs         */
    ufsecp_ff_limbs_t lmb_result_y;        /**< R.y limbs         */
    ufsecp_ff_limbs_t lmb_result_x_mod_n;  /**< R.x mod n limbs   */

    /* ── verdict ──────────────────────────────────────────────────────── */
    int valid;  /**< 1 = signature valid, 0 = invalid */
} ufsecp_ecdsa_snark_witness_t;

/** Compute ECDSA foreign-field prover witness (eprint 2025/695).
 *
 *  Runs a full secp256k1 ECDSA verification and captures all intermediate
 *  values (s_inv, u1, u2, MSM result) in both canonical bytes and 5×52-bit
 *  foreign-field limb form — ready to feed into any PLONK framework.
 *
 *  msg_hash32  : 32-byte big-endian message hash
 *  pubkey33    : 33-byte compressed secp256k1 public key
 *  sig64       : 64-byte compact signature  (r[32] || s[32], big-endian)
 *  out         : caller-allocated output structure (zero-initialised on call)
 *
 *  Returns UFSECP_OK on all well-formed inputs, even when the signature is
 *  invalid — check out->valid for the verification result.
 *  Returns UFSECP_ERR_BAD_INPUT on malformed pubkey, r=0, or s=0. */
UFSECP_API ufsecp_error_t ufsecp_zk_ecdsa_snark_witness(
    ufsecp_ctx* ctx,
    const uint8_t msg_hash32[32],
    const uint8_t pubkey33[33],
    const uint8_t sig64[64],
    ufsecp_ecdsa_snark_witness_t* out);

/* ===========================================================================
 * BIP340 Schnorr-in-SNARK Foreign-Field Witness
 * =========================================================================== */

/** Complete PLONK prover witness for one BIP-340 Schnorr verification.
 *
 *  BIP-340 verify steps:
 *    R      = lift_x(sig_r)       (even Y)
 *    P      = lift_x(pubkey_x)    (even Y)
 *    e      = H("BIP0340/challenge" || R.x || P.x || msg) mod n
 *    valid  = (s*G == R + e*P) AND (R.y is even)
 */
typedef struct {
    /* ── public inputs ─────────────────────────────────────────────────── */
    uint8_t msg[32];       /**< message (32 bytes, per BIP-340)   */
    uint8_t sig_r[32];     /**< R.x from signature                */
    uint8_t sig_s[32];     /**< s scalar from signature           */
    uint8_t pub_x[32];     /**< x-only public key                 */

    /* ── private witness (circuit signals) ─────────────────────────────── */
    uint8_t r_y[32];       /**< R.y (lifted, even Y)              */
    uint8_t pub_y[32];     /**< P.y (lifted, even Y)              */
    uint8_t e[32];         /**< challenge scalar                  */

    /* ── 5×52-bit foreign-field limb decompositions ─────────────────────── */
    ufsecp_ff_limbs_t lmb_sig_r;           /**< R.x limbs         */
    ufsecp_ff_limbs_t lmb_sig_s;           /**< s limbs           */
    ufsecp_ff_limbs_t lmb_pub_x;           /**< P.x limbs         */
    ufsecp_ff_limbs_t lmb_r_y;             /**< R.y limbs         */
    ufsecp_ff_limbs_t lmb_pub_y;           /**< P.y limbs         */
    ufsecp_ff_limbs_t lmb_e;               /**< challenge limbs   */

    /* ── verdict ──────────────────────────────────────────────────────── */
    int valid;  /**< 1 = signature valid, 0 = invalid */
} ufsecp_schnorr_snark_witness_t;

/** Compute BIP-340 Schnorr foreign-field prover witness.
 *
 *  Runs a full BIP-340 Schnorr verification and captures all intermediate
 *  values (R.y, P.y, challenge e) in both canonical bytes and 5×52-bit
 *  foreign-field limb form — ready to feed into any PLONK framework.
 *
 *  msg32       : 32-byte message (per BIP-340 spec)
 *  pubkey_x32  : 32-byte x-only public key (big-endian)
 *  sig64       : 64-byte BIP-340 signature (R.x[32] || s[32])
 *  out         : caller-allocated output structure
 *
 *  Returns UFSECP_OK on all well-formed inputs, even when the signature is
 *  invalid — check out->valid for the verification result.
 *  Returns UFSECP_ERR_BAD_INPUT on malformed inputs . */
UFSECP_API ufsecp_error_t ufsecp_zk_schnorr_snark_witness(
    ufsecp_ctx* ctx,
    const uint8_t msg32[32],
    const uint8_t pubkey_x32[32],
    const uint8_t sig64[64],
    ufsecp_schnorr_snark_witness_t* out);

/* ===========================================================================
 * Multi-coin wallet infrastructure
 * =========================================================================== */

/** Maximum address string length for any supported coin. */
#define UFSECP_COIN_ADDR_MAX_LEN 128

/** Coin identifiers (BIP-44 coin_type). */
#define UFSECP_COIN_BITCOIN      0
#define UFSECP_COIN_LITECOIN     2
#define UFSECP_COIN_DOGECOIN     3
#define UFSECP_COIN_DASH         5
#define UFSECP_COIN_ETHEREUM     60
#define UFSECP_COIN_BITCOIN_CASH 145
#define UFSECP_COIN_TRON         195

/** Get default address for a coin from a compressed public key.
 *  coin_type: BIP-44 coin type index.
 *  addr_out: buffer for NUL-terminated address.
 *  addr_len: in = buffer size, out = strlen. */
UFSECP_API ufsecp_error_t ufsecp_coin_address(
    ufsecp_ctx* ctx,
    const uint8_t pubkey33[33],
    uint32_t coin_type, int testnet,
    char* addr_out, size_t* addr_len);

/** Derive full key from seed for a specific coin.
 *  seed must be 16 to 64 bytes.
 *  Derives using best_purpose for the coin.
 *  privkey32_out, pubkey33_out: optional (NULL to skip).
 *  addr_out and addr_len are optional as a pair and must be both NULL or both non-NULL. */
UFSECP_API ufsecp_error_t ufsecp_coin_derive_from_seed(
    ufsecp_ctx* ctx,
    const uint8_t* seed, size_t seed_len,
    uint32_t coin_type, uint32_t account, int change, uint32_t index,
    int testnet,
    uint8_t* privkey32_out,
    uint8_t* pubkey33_out,
    char* addr_out, size_t* addr_len);

/** Encode WIF for any supported coin. */
UFSECP_API ufsecp_error_t ufsecp_coin_wif_encode(
    ufsecp_ctx* ctx,
    const uint8_t privkey[32],
    uint32_t coin_type, int testnet,
    char* wif_out, size_t* wif_len);

/** Bitcoin message signing (BIP-137).
 *  base64_out: buffer for base64-encoded signature.
 *  base64_len: in = buffer size, out = strlen. */
UFSECP_API ufsecp_error_t ufsecp_btc_message_sign(
    ufsecp_ctx* ctx,
    const uint8_t* msg, size_t msg_len,
    const uint8_t privkey[32],
    char* base64_out, size_t* base64_len);

/** Bitcoin message verify.
 *  Returns UFSECP_OK if signature is valid. */
UFSECP_API ufsecp_error_t ufsecp_btc_message_verify(
    ufsecp_ctx* ctx,
    const uint8_t* msg, size_t msg_len,
    const uint8_t pubkey33[33],
    const char* base64_sig);

/** Bitcoin message hash (double SHA-256 with prefix). */
UFSECP_API ufsecp_error_t ufsecp_btc_message_hash(
    const uint8_t* msg, size_t msg_len,
    uint8_t digest32_out[32]);

/* ===========================================================================
 * BIP-352 Silent Payments
 * =========================================================================== */

/** Generate a Silent Payment address from scan and spend private keys.
 *  scan_privkey:  32-byte scan private key.
 *  spend_privkey: 32-byte spend private key.
 *  scan_pubkey33_out:  33-byte compressed scan public key (B_scan).
 *  spend_pubkey33_out: 33-byte compressed spend public key (B_spend).
 *  addr_out: buffer for bech32m-encoded address (min 128 bytes).
 *  addr_len: in = buffer size, out = strlen (excl. NUL). */
UFSECP_API ufsecp_error_t ufsecp_silent_payment_address(
    ufsecp_ctx* ctx,
    const uint8_t scan_privkey[32],
    const uint8_t spend_privkey[32],
    uint8_t scan_pubkey33_out[33],
    uint8_t spend_pubkey33_out[33],
    char* addr_out, size_t* addr_len);

/** Create a Silent Payment output (sender side).
 *  Computes the tweaked output pubkey for the recipient.
 *  input_privkeys: array of 32-byte private keys (N keys, one per input).
 *  n_inputs: number of input private keys.
 *  scan_pubkey33:  33-byte recipient scan pubkey (B_scan).
 *  spend_pubkey33: 33-byte recipient spend pubkey (B_spend).
 *  k: output index (for multiple outputs to same recipient).
 *  output_pubkey33_out: 33-byte compressed tweaked output pubkey.
 *  tweak32_out: 32-byte tweak scalar (optional, may be NULL). */
UFSECP_API ufsecp_error_t ufsecp_silent_payment_create_output(
    ufsecp_ctx* ctx,
    const uint8_t* input_privkeys, size_t n_inputs,
    const uint8_t scan_pubkey33[33],
    const uint8_t spend_pubkey33[33],
    uint32_t k,
    uint8_t output_pubkey33_out[33],
    uint8_t* tweak32_out);

/** Scan for Silent Payment outputs (receiver side).
 *  scan_privkey:  32-byte scan private key.
 *  spend_privkey: 32-byte spend private key.
 *  input_pubkeys33: array of 33-byte compressed pubkeys (sender inputs).
 *  n_input_pubkeys: number of input pubkeys.
 *  output_xonly32: array of 32-byte x-only output pubkeys to check.
 *  n_outputs: number of output pubkeys.
 *  found_indices_out: array to receive indices of matched outputs.
 *  found_privkeys_out: array to receive 32-byte spending private keys (one per match).
 *  n_found: in = array capacity, out = number of matches found. */
UFSECP_API ufsecp_error_t ufsecp_silent_payment_scan(
    ufsecp_ctx* ctx,
    const uint8_t scan_privkey[32],
    const uint8_t spend_privkey[32],
    const uint8_t* input_pubkeys33, size_t n_input_pubkeys,
    const uint8_t* output_xonly32, size_t n_outputs,
    uint32_t* found_indices_out,
    uint8_t* found_privkeys_out,
    size_t* n_found);

/* ===========================================================================
 * ECIES (Elliptic Curve Integrated Encryption Scheme)
 * =========================================================================== */

/** ECIES envelope overhead: 33 (ephemeral pubkey) + 16 (IV) + 32 (HMAC) = 81 */
#define UFSECP_ECIES_OVERHEAD 81

/** ECIES encrypt: encrypt plaintext for a recipient's public key.
 *  recipient_pubkey33: 33-byte compressed public key.
 *  plaintext, plaintext_len: message to encrypt.
 *  envelope_out: buffer for encrypted envelope (min plaintext_len + 81).
 *  envelope_len: in = buffer size, out = actual envelope size. */
UFSECP_API ufsecp_error_t ufsecp_ecies_encrypt(
    ufsecp_ctx* ctx,
    const uint8_t recipient_pubkey33[33],
    const uint8_t* plaintext, size_t plaintext_len,
    uint8_t* envelope_out, size_t* envelope_len);

/** ECIES decrypt: decrypt an ECIES envelope with a private key.
 *  privkey: 32-byte private key.
 *  envelope, envelope_len: encrypted envelope.
 *  plaintext_out: buffer for decrypted plaintext (min envelope_len - 81).
 *  plaintext_len: in = buffer size, out = actual plaintext size. */
UFSECP_API ufsecp_error_t ufsecp_ecies_decrypt(
    ufsecp_ctx* ctx,
    const uint8_t privkey[32],
    const uint8_t* envelope, size_t envelope_len,
    uint8_t* plaintext_out, size_t* plaintext_len);

/* ========================================================================== */
/* BIP-324: Version 2 P2P Encrypted Transport (conditional: SECP256K1_BIP324) */
/* ========================================================================== */
#ifdef SECP256K1_BIP324

/** Opaque BIP-324 session handle. */
typedef struct ufsecp_bip324_session ufsecp_bip324_session;

/** Create a new BIP-324 session.
 *  initiator: must be exactly 1 if this peer initiates the connection, or 0 if responder.
 *  session_out: receives the session handle (caller must free with ufsecp_bip324_destroy).
 *  ellswift64_out: receives our 64-byte ElligatorSwift-encoded public key to send to the peer. */
UFSECP_API ufsecp_error_t ufsecp_bip324_create(
    ufsecp_ctx* ctx,
    int initiator,
    ufsecp_bip324_session** session_out,
    uint8_t ellswift64_out[64]);

/** Complete the BIP-324 handshake by providing the peer's 64-byte ElligatorSwift encoding.
 *  session_id32_out: if non-NULL, receives the 32-byte session ID.
 *  Returns UFSECP_OK on success. */
UFSECP_API ufsecp_error_t ufsecp_bip324_handshake(
    ufsecp_bip324_session* session,
    const uint8_t peer_ellswift64[64],
    uint8_t session_id32_out[32]);

/** Encrypt a BIP-324 packet.
 *  plaintext: the message payload.
 *  out: buffer for encrypted output ([3B enc length][encrypted payload][16B tag]).
 *  out_len: in = buffer size, out = bytes written. Must be >= plaintext_len + 19. */
UFSECP_API ufsecp_error_t ufsecp_bip324_encrypt(
    ufsecp_bip324_session* session,
    const uint8_t* plaintext, size_t plaintext_len,
    uint8_t* out, size_t* out_len);

/** Decrypt a BIP-324 packet.
 *  encrypted: the full encrypted packet ([3B header][payload][16B tag]).
 *  encrypted_len: total length of encrypted data.
 *  plaintext_out: buffer for decrypted payload.
 *  plaintext_len: in = buffer size, out = payload bytes written.
 *  Returns UFSECP_OK on success, including valid zero-length payloads.
 *  Returns UFSECP_ERR_VERIFY_FAIL on authentication or integrity failure. */
UFSECP_API ufsecp_error_t ufsecp_bip324_decrypt(
    ufsecp_bip324_session* session,
    const uint8_t* encrypted, size_t encrypted_len,
    uint8_t* plaintext_out, size_t* plaintext_len);

/** Destroy a BIP-324 session and securely erase key material. */
UFSECP_API void ufsecp_bip324_destroy(ufsecp_bip324_session* session);

/** ChaCha20-Poly1305 AEAD encrypt (standalone, independent of BIP-324 sessions).
 *  key32: 256-bit key.
 *  nonce12: 96-bit nonce.
 *  aad/aad_len: additional authenticated data.
 *  plaintext/plaintext_len: input.
 *  out: ciphertext (same length as plaintext).
 *  tag16_out: 128-bit authentication tag. */
UFSECP_API ufsecp_error_t ufsecp_aead_chacha20_poly1305_encrypt(
    const uint8_t key[32], const uint8_t nonce[12],
    const uint8_t* aad, size_t aad_len,
    const uint8_t* plaintext, size_t plaintext_len,
    uint8_t* out, uint8_t tag[16]);

/** ChaCha20-Poly1305 AEAD decrypt (standalone).
 *  Returns UFSECP_OK if tag is valid, UFSECP_ERR_VERIFY_FAIL on authentication failure. */
UFSECP_API ufsecp_error_t ufsecp_aead_chacha20_poly1305_decrypt(
    const uint8_t key[32], const uint8_t nonce[12],
    const uint8_t* aad, size_t aad_len,
    const uint8_t* ciphertext, size_t ciphertext_len,
    const uint8_t tag[16], uint8_t* out);

/** ElligatorSwift: create a 64-byte encoding of a public key.
 *  privkey32: private key (used to compute the public key).
 *  encoding64_out: receives the 64-byte uniformly random-looking encoding. */
UFSECP_API ufsecp_error_t ufsecp_ellswift_create(
    ufsecp_ctx* ctx,
    const uint8_t privkey[32],
    uint8_t encoding64_out[64]);

/** ElligatorSwift ECDH: compute shared secret from ElligatorSwift encodings.
 *  ell_a64: initiator's 64-byte encoding.
 *  ell_b64: responder's 64-byte encoding.
 *  our_privkey32: our private key.
 *  initiating: 1 if we are the initiator, 0 if responder.
 *  secret32_out: 32-byte shared secret. */
UFSECP_API ufsecp_error_t ufsecp_ellswift_xdh(
    ufsecp_ctx* ctx,
    const uint8_t ell_a64[64],
    const uint8_t ell_b64[64],
    const uint8_t our_privkey[32],
    int initiating,
    uint8_t secret32_out[32]);

#endif /* SECP256K1_BIP324 */

#ifdef SECP256K1_BUILD_ETHEREUM

/** Ethereum address size (20 bytes). */
#define UFSECP_ETH_ADDR_LEN 20

/** Keccak-256 hash (Ethereum variant, NOT SHA3-256).
 *  Output: 32 bytes. */
UFSECP_API ufsecp_error_t ufsecp_keccak256(const uint8_t* data, size_t len,
                                           uint8_t digest32_out[32]);

/** Derive Ethereum address (20 bytes) from compressed public key.
 *  pubkey33: 33-byte compressed public key.
 *  addr20_out: 20-byte Ethereum address. */
UFSECP_API ufsecp_error_t ufsecp_eth_address(ufsecp_ctx* ctx,
                                             const uint8_t pubkey33[33],
                                             uint8_t addr20_out[20]);

/** Derive EIP-55 checksummed Ethereum address string from compressed pubkey.
 *  addr_out: buffer for "0x" + 40 hex chars + NUL (min 43 bytes).
 *  addr_len: in = buffer size, out = strlen (excl. NUL). */
UFSECP_API ufsecp_error_t ufsecp_eth_address_checksummed(
    ufsecp_ctx* ctx,
    const uint8_t pubkey33[33],
    char* addr_out, size_t* addr_len);

/** EIP-191 personal_sign: hash a message with Ethereum prefix.
 *  Computes Keccak256("\x19Ethereum Signed Message:\n" + len(msg) + msg).
 *  digest32_out: 32-byte hash. */
UFSECP_API ufsecp_error_t ufsecp_eth_personal_hash(const uint8_t* msg, size_t msg_len,
                                                   uint8_t digest32_out[32]);

/** Sign message hash with ECDSA recovery (Ethereum v,r,s format).
 *  msg32: 32-byte message hash (pre-hashed, e.g. output of personal_hash).
 *  privkey: 32-byte private key.
 *  r_out, s_out: 32 bytes each.
 *  v_out: EIP-155 v value (27+recid for legacy, 35+2*chainId+recid). */
UFSECP_API ufsecp_error_t ufsecp_eth_sign(ufsecp_ctx* ctx,
                                          const uint8_t msg32[32],
                                          const uint8_t privkey[32],
                                          uint8_t r_out[32],
                                          uint8_t s_out[32],
                                          uint64_t* v_out,
                                          uint64_t chain_id);

/** ecrecover: recover 20-byte Ethereum address from ECDSA(v,r,s) + msg hash.
 *  This is Ethereum's ecrecover precompile (address 0x01).
 *  Returns UFSECP_OK if recovery succeeds. */
UFSECP_API ufsecp_error_t ufsecp_eth_ecrecover(ufsecp_ctx* ctx,
                                               const uint8_t msg32[32],
                                               const uint8_t r[32],
                                               const uint8_t s[32],
                                               uint64_t v,
                                               uint8_t addr20_out[20]);

#endif /* SECP256K1_BUILD_ETHEREUM */

/* ===========================================================================
 * BIP-85 — Deterministic Entropy from BIP-32 Keychains
 * =========================================================================== */

/** Derive application entropy from a BIP-32 master xprv.
 *  path: BIP-85 derivation path string, e.g. "m/83696968'/2'/0'"
 *  entropy_out: output buffer (caller-supplied, min entropy_len bytes).
 *  entropy_len: number of entropy bytes to derive (16, 24, or 32).
 *  Internally: HMAC-SHA512(key="bip-85", data=derived_privkey),
 *  take first entropy_len bytes. */
UFSECP_API ufsecp_error_t ufsecp_bip85_entropy(
    ufsecp_ctx* ctx,
    const ufsecp_bip32_key* master_xprv,
    const char* path,
    uint8_t* entropy_out, size_t entropy_len);

/** Derive a BIP-39 mnemonic using BIP-85.
 *  words: 12, 18, or 24.
 *  language_index: 0=English.
 *  index: child index.
 *  mnemonic_out: buffer, min 500 bytes. */
UFSECP_API ufsecp_error_t ufsecp_bip85_bip39(
    ufsecp_ctx* ctx,
    const ufsecp_bip32_key* master_xprv,
    uint32_t words, uint32_t language_index, uint32_t index,
    char* mnemonic_out, size_t* mnemonic_len);

/* ===========================================================================
 * BIP-340 Variable-Length Schnorr
 * =========================================================================== */

/** Sign an arbitrary-length message with BIP-340 Schnorr.
 *  Internally: msg_hash = tagged_hash("BIP0340/msg", msg, msg_len).
 *  Use this instead of ufsecp_schnorr_sign when msg is not exactly 32 bytes. */
UFSECP_API ufsecp_error_t ufsecp_schnorr_sign_msg(
    ufsecp_ctx* ctx,
    const uint8_t privkey[32],
    const uint8_t* msg, size_t msg_len,
    const uint8_t* aux_rand32,
    uint8_t sig64_out[64]);

/** Verify Schnorr signature over arbitrary-length message. */
UFSECP_API ufsecp_error_t ufsecp_schnorr_verify_msg(
    ufsecp_ctx* ctx,
    const uint8_t pubkey_x[32],
    const uint8_t* msg, size_t msg_len,
    const uint8_t sig64[64]);

/* ===========================================================================
 * BIP-322 — Generic Message Signing
 * =========================================================================== */

typedef enum {
    UFSECP_BIP322_ADDR_P2PKH        = 0,
    UFSECP_BIP322_ADDR_P2WPKH       = 1,
    UFSECP_BIP322_ADDR_P2TR         = 2,
    UFSECP_BIP322_ADDR_P2SH_P2WPKH  = 3,
} ufsecp_bip322_addr_type;

/** Sign a message using BIP-322 "simple" type.
 *  privkey: 32-byte private key.
 *  addr_type: address type (determines signing scheme and sighash).
 *  sig_out: buffer for the witness/signature bytes (min 128 bytes).
 *  sig_len: in = buffer size, out = actual bytes written. */
UFSECP_API ufsecp_error_t ufsecp_bip322_sign(
    ufsecp_ctx* ctx,
    const uint8_t privkey[32],
    ufsecp_bip322_addr_type addr_type,
    const uint8_t* msg, size_t msg_len,
    uint8_t* sig_out, size_t* sig_len);

/** Verify a BIP-322 "simple" signature.
 *  pubkey: 33-byte compressed (P2PKH/P2WPKH/P2SH-P2WPKH) or 32-byte x-only (P2TR).
 *  pubkey_len: 33 or 32.
 *  Returns UFSECP_OK if valid. */
UFSECP_API ufsecp_error_t ufsecp_bip322_verify(
    ufsecp_ctx* ctx,
    const uint8_t* pubkey, size_t pubkey_len,
    ufsecp_bip322_addr_type addr_type,
    const uint8_t* msg, size_t msg_len,
    const uint8_t* sig, size_t sig_len);

/* ===========================================================================
 * BIP-157/158 — Compact Block Filters (Golomb-Coded Set)
 * =========================================================================== */

/** Build a BIP-158 "basic" GCS filter.
 *  key: 16-byte SipHash key (from block hash).
 *  data: array of count variable-length items (each a script or txid).
 *  data_sizes: array of count sizes for each data item.
 *  filter_out: output buffer for encoded filter (caller-supplied).
 *  filter_len: in = buffer size, out = actual bytes written.
 *  N = count of items, P = 19, M = 784931 (BIP-158 defaults). */
UFSECP_API ufsecp_error_t ufsecp_gcs_build(
    const uint8_t key[16],
    const uint8_t** data, const size_t* data_sizes, size_t count,
    uint8_t* filter_out, size_t* filter_len);

/** Test if a single item is in the filter.
 *  Returns UFSECP_OK if item is in filter, UFSECP_ERR_NOT_FOUND if not. */
UFSECP_API ufsecp_error_t ufsecp_gcs_match(
    const uint8_t key[16],
    const uint8_t* filter, size_t filter_len,
    size_t n_items,
    const uint8_t* item, size_t item_len);

/** Test if any of the query items is in the filter (OR match).
 *  Returns UFSECP_OK if any item matches. */
UFSECP_API ufsecp_error_t ufsecp_gcs_match_any(
    const uint8_t key[16],
    const uint8_t* filter, size_t filter_len,
    size_t n_items,
    const uint8_t** query, const size_t* query_sizes, size_t query_count);

/* ===========================================================================
 * BIP-174/370 — PSBT Signing Helpers
 * =========================================================================== */

/** PSBT input sighash types. */
#define UFSECP_SIGHASH_ALL          0x01
#define UFSECP_SIGHASH_NONE         0x02
#define UFSECP_SIGHASH_SINGLE       0x03
#define UFSECP_SIGHASH_ANYONECANPAY 0x80
#define UFSECP_SIGHASH_DEFAULT      0x00  /* BIP-341 Taproot default */

/** Sign a PSBT non-witness input (legacy P2PKH).
 *  sighash: 32-byte BIP-143 or BIP-341 sighash pre-image digest.
 *  privkey: signing private key.
 *  sig_out: DER+sighash_type, min 73 bytes.
 *  sig_len: in = buffer size, out = actual bytes. */
UFSECP_API ufsecp_error_t ufsecp_psbt_sign_legacy(
    ufsecp_ctx* ctx,
    const uint8_t sighash32[32],
    const uint8_t privkey[32],
    uint8_t sighash_type,
    uint8_t* sig_out, size_t* sig_len);

/** Sign a PSBT SegWit v0 input (P2WPKH or P2WSH).
 *  Returns compact ECDSA sig (64 bytes) + sighash_type (1 byte) = 65 bytes total. */
UFSECP_API ufsecp_error_t ufsecp_psbt_sign_segwit(
    ufsecp_ctx* ctx,
    const uint8_t sighash32[32],
    const uint8_t privkey[32],
    uint8_t sighash_type,
    uint8_t* sig_out, size_t* sig_len);

/** Sign a PSBT Taproot key-path input (P2TR).
 *  Returns 64-byte Schnorr sig (+ optional sighash_type byte if not SIGHASH_DEFAULT). */
UFSECP_API ufsecp_error_t ufsecp_psbt_sign_taproot(
    ufsecp_ctx* ctx,
    const uint8_t sighash32[32],
    const uint8_t privkey[32],
    uint8_t sighash_type,
    const uint8_t* aux_rand32,
    uint8_t* sig_out, size_t* sig_len);

/** Derive the signing key from a BIP-32 xprv + key-path record.
 *  key_path: e.g. "m/84'/0'/0'/0/0"
 *  privkey_out: 32-byte derived private key. */
UFSECP_API ufsecp_error_t ufsecp_psbt_derive_key(
    ufsecp_ctx* ctx,
    const ufsecp_bip32_key* master_xprv,
    const char* key_path,
    uint8_t privkey_out[32]);

/* ===========================================================================
 * BIP-380..386 — Output Descriptors (key expression parser)
 * =========================================================================== */

typedef enum {
    UFSECP_DESC_PK       = 0,   /**< pk(KEY) — P2PK */
    UFSECP_DESC_PKH      = 1,   /**< pkh(KEY) — P2PKH */
    UFSECP_DESC_WPKH     = 2,   /**< wpkh(KEY) — P2WPKH */
    UFSECP_DESC_TR       = 3,   /**< tr(KEY) — P2TR key-path only */
    UFSECP_DESC_SH_WPKH  = 4,   /**< sh(wpkh(KEY)) — P2SH-P2WPKH */
} ufsecp_desc_type;

/** Parsed descriptor result. */
typedef struct {
    ufsecp_desc_type type;
    uint8_t pubkey[33];        /**< Compressed pubkey (or x-only [32] for TR). */
    uint8_t pubkey_len;        /**< 33 for compressed, 32 for x-only. */
    int network;               /**< UFSECP_NET_MAINNET or UFSECP_NET_TESTNET. */
    char path[64];             /**< Derivation path suffix, e.g. "/0/0", or empty. */
} ufsecp_desc_key;

/** Parse a descriptor string and derive the key + address type.
 *  descriptor: e.g. "wpkh(xpub.../<0;1>/[*])" or "tr(xpub.../0/0)"
 *  index: child index to resolve (replaces * wildcard).
 *  key_out: receives the parsed key information.
 *  addr_out: buffer for the derived address (min 128 bytes), or NULL.
 *  addr_len: in/out for address buffer. */
UFSECP_API ufsecp_error_t ufsecp_descriptor_parse(
    ufsecp_ctx* ctx,
    const char* descriptor,
    uint32_t index,
    ufsecp_desc_key* key_out,
    char* addr_out, size_t* addr_len);

/** Derive address directly from a descriptor string.
 *  Convenience wrapper around ufsecp_descriptor_parse. */
UFSECP_API ufsecp_error_t ufsecp_descriptor_address(
    ufsecp_ctx* ctx,
    const char* descriptor,
    uint32_t index,
    char* addr_out, size_t* addr_len);

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
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(ufsecp_bip32_key) == 82,
               "ABI break: ufsecp_bip32_key size changed (expected 82)");
#endif
#endif

#endif /* UFSECP_H */
