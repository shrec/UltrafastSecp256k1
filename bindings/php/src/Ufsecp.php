<?php

declare(strict_types=1);

/**
 * UltrafastSecp256k1 — PHP FFI Bindings (ufsecp stable C ABI)
 *
 * High-performance secp256k1 elliptic curve cryptography with dual-layer
 * constant-time architecture: secret operations always use CT layer,
 * public operations always use fast layer. Both always active — no opt-in.
 *
 * Requirements: PHP 8.1+ with ext-ffi enabled.
 *
 * Usage:
 *   $ctx = new \Ultrafast\Ufsecp\Ufsecp();
 *   $privkey = str_repeat("\x00", 31) . "\x01";
 *   $pubkey = $ctx->pubkeyCreate($privkey);
 *   $sig = $ctx->ecdsaSign(str_repeat("\x00", 32), $privkey);
 *   $ok = $ctx->ecdsaVerify(str_repeat("\x00", 32), $sig, $pubkey);
 */

namespace Ultrafast\Ufsecp;

use FFI;
use RuntimeException;
use InvalidArgumentException;

class Ufsecp
{
    public const EXPECTED_ABI = 4;
    public const NET_MAINNET = 0;
    public const NET_TESTNET = 1;

    // Error codes
    public const OK                = 0;
    public const ERR_NULL_ARG      = 1;
    public const ERR_BAD_KEY       = 2;
    public const ERR_BAD_PUBKEY    = 3;
    public const ERR_BAD_SIG       = 4;
    public const ERR_BAD_INPUT     = 5;
    public const ERR_VERIFY_FAIL   = 6;
    public const ERR_ARITH         = 7;
    public const ERR_SELFTEST      = 8;
    public const ERR_INTERNAL      = 9;
    public const ERR_BUF_TOO_SMALL = 10;

    private FFI $ffi;

    /** @var FFI\CData Opaque ufsecp_ctx* */
    private FFI\CData $ctx;

    private const C_HEADER = <<<'CDEF'
    // Version / ABI
    unsigned int ufsecp_version(void);
    unsigned int ufsecp_abi_version(void);
    const char*  ufsecp_version_string(void);
    const char*  ufsecp_error_str(int err);

    // Context (opaque pointer)
    typedef struct ufsecp_ctx ufsecp_ctx;
    int         ufsecp_ctx_create(ufsecp_ctx** ctx_out);
    int         ufsecp_ctx_clone(const ufsecp_ctx* src, ufsecp_ctx** ctx_out);
    void        ufsecp_ctx_destroy(ufsecp_ctx* ctx);
    int         ufsecp_last_error(const ufsecp_ctx* ctx);
    const char* ufsecp_last_error_msg(const ufsecp_ctx* ctx);
    size_t      ufsecp_ctx_size(void);

    // Private key
    int ufsecp_seckey_verify(const ufsecp_ctx* ctx, const uint8_t* privkey);
    int ufsecp_seckey_negate(ufsecp_ctx* ctx, uint8_t* privkey);
    int ufsecp_seckey_tweak_add(ufsecp_ctx* ctx, uint8_t* privkey, const uint8_t* tweak);
    int ufsecp_seckey_tweak_mul(ufsecp_ctx* ctx, uint8_t* privkey, const uint8_t* tweak);

    // Public key
    int ufsecp_pubkey_create(ufsecp_ctx* ctx, const uint8_t* privkey, uint8_t* pubkey33_out);
    int ufsecp_pubkey_create_uncompressed(ufsecp_ctx* ctx, const uint8_t* privkey, uint8_t* pubkey65_out);
    int ufsecp_pubkey_parse(ufsecp_ctx* ctx, const uint8_t* input, size_t input_len, uint8_t* pubkey33_out);
    int ufsecp_pubkey_xonly(ufsecp_ctx* ctx, const uint8_t* privkey, uint8_t* xonly32_out);

    // ECDSA
    int ufsecp_ecdsa_sign(ufsecp_ctx* ctx, const uint8_t* msg32, const uint8_t* privkey, uint8_t* sig64_out);
    int ufsecp_ecdsa_verify(ufsecp_ctx* ctx, const uint8_t* msg32, const uint8_t* sig64, const uint8_t* pubkey33);
    int ufsecp_ecdsa_sig_to_der(ufsecp_ctx* ctx, const uint8_t* sig64, uint8_t* der_out, size_t* der_len);
    int ufsecp_ecdsa_sig_from_der(ufsecp_ctx* ctx, const uint8_t* der, size_t der_len, uint8_t* sig64_out);

    // Recovery
    int ufsecp_ecdsa_sign_recoverable(ufsecp_ctx* ctx, const uint8_t* msg32, const uint8_t* privkey, uint8_t* sig64_out, int* recid_out);
    int ufsecp_ecdsa_recover(ufsecp_ctx* ctx, const uint8_t* msg32, const uint8_t* sig64, int recid, uint8_t* pubkey33_out);

    // Schnorr
    int ufsecp_schnorr_sign(ufsecp_ctx* ctx, const uint8_t* msg32, const uint8_t* privkey, const uint8_t* aux_rand, uint8_t* sig64_out);
    int ufsecp_schnorr_verify(ufsecp_ctx* ctx, const uint8_t* msg32, const uint8_t* sig64, const uint8_t* pubkey_x);

    // ECDH
    int ufsecp_ecdh(ufsecp_ctx* ctx, const uint8_t* privkey, const uint8_t* pubkey33, uint8_t* secret32_out);
    int ufsecp_ecdh_xonly(ufsecp_ctx* ctx, const uint8_t* privkey, const uint8_t* pubkey33, uint8_t* secret32_out);
    int ufsecp_ecdh_raw(ufsecp_ctx* ctx, const uint8_t* privkey, const uint8_t* pubkey33, uint8_t* secret32_out);

    // Hashing (context-free)
    int ufsecp_sha256(const uint8_t* data, size_t len, uint8_t* digest32_out);
    int ufsecp_hash160(const uint8_t* data, size_t len, uint8_t* digest20_out);
    int ufsecp_tagged_hash(const char* tag, const uint8_t* data, size_t len, uint8_t* digest32_out);

    // Addresses
    int ufsecp_addr_p2pkh(ufsecp_ctx* ctx, const uint8_t* pubkey33, int network, char* addr_out, size_t* addr_len);
    int ufsecp_addr_p2wpkh(ufsecp_ctx* ctx, const uint8_t* pubkey33, int network, char* addr_out, size_t* addr_len);
    int ufsecp_addr_p2tr(ufsecp_ctx* ctx, const uint8_t* internal_key_x, int network, char* addr_out, size_t* addr_len);

    // WIF
    int ufsecp_wif_encode(ufsecp_ctx* ctx, const uint8_t* privkey, int compressed, int network, char* wif_out, size_t* wif_len);
    int ufsecp_wif_decode(ufsecp_ctx* ctx, const char* wif, uint8_t* privkey32_out, int* compressed_out, int* network_out);

    // BIP-32
    typedef struct { uint8_t data[78]; uint8_t is_private; uint8_t _pad[3]; } ufsecp_bip32_key;
    int ufsecp_bip32_master(ufsecp_ctx* ctx, const uint8_t* seed, size_t seed_len, ufsecp_bip32_key* key_out);
    int ufsecp_bip32_derive(ufsecp_ctx* ctx, const ufsecp_bip32_key* parent, uint32_t index, ufsecp_bip32_key* child_out);
    int ufsecp_bip32_derive_path(ufsecp_ctx* ctx, const ufsecp_bip32_key* master, const char* path, ufsecp_bip32_key* key_out);
    int ufsecp_bip32_privkey(ufsecp_ctx* ctx, const ufsecp_bip32_key* key, uint8_t* privkey32_out);
    int ufsecp_bip32_pubkey(ufsecp_ctx* ctx, const ufsecp_bip32_key* key, uint8_t* pubkey33_out);

    // Taproot
    int ufsecp_taproot_output_key(ufsecp_ctx* ctx, const uint8_t* internal_x, const uint8_t* merkle_root, uint8_t* output_x_out, int* parity_out);
    int ufsecp_taproot_tweak_seckey(ufsecp_ctx* ctx, const uint8_t* privkey, const uint8_t* merkle_root, uint8_t* tweaked32_out);
    int ufsecp_taproot_verify(ufsecp_ctx* ctx, const uint8_t* output_x, int output_parity, const uint8_t* internal_x, const uint8_t* merkle_root, size_t merkle_root_len);
    CDEF;

    /** @param string|null $libPath Path to the ufsecp shared library. Auto-detected if null. */
    public function __construct(?string $libPath = null)
    {
        $path = $libPath ?? self::findLibrary();
        $this->ffi = FFI::cdef(self::C_HEADER, $path);

        $abi = $this->ffi->ufsecp_abi_version();
        if ($abi !== self::EXPECTED_ABI) {
            throw new RuntimeException(
                "ABI mismatch: wrapper expects ABI " . self::EXPECTED_ABI . ", lib reports ABI $abi."
            );
        }

        $ctxPtr = $this->ffi->new('ufsecp_ctx*');
        $rc = $this->ffi->ufsecp_ctx_create(FFI::addr($ctxPtr));
        if ($rc !== self::OK) {
            throw new RuntimeException('ufsecp_ctx_create() failed: ' . self::errorName($rc));
        }
        $this->ctx = $ctxPtr;
    }

    public function __destruct()
    {
        $this->close();
    }

    public function close(): void
    {
        if (isset($this->ctx)) {
            $this->ffi->ufsecp_ctx_destroy($this->ctx);
            unset($this->ctx);
        }
    }

    // ── Version ──────────────────────────────────────────────────────────

    /** Packed version number. */
    public function version(): int
    {
        return $this->ffi->ufsecp_version();
    }

    /** ABI version (should match UFSECP_ABI_VERSION). */
    public function abiVersion(): int
    {
        return $this->ffi->ufsecp_abi_version();
    }

    /** Human-readable version string, e.g. "3.4.0". */
    public function versionString(): string
    {
        return FFI::string($this->ffi->ufsecp_version_string());
    }

    /** Last error code on this context. */
    public function lastError(): int
    {
        return $this->ffi->ufsecp_last_error($this->ctx);
    }

    /** Last error message on this context. */
    public function lastErrorMsg(): string
    {
        return FFI::string($this->ffi->ufsecp_last_error_msg($this->ctx));
    }

    // ── Key Operations ───────────────────────────────────────────────────

    /** Compressed public key (33 bytes) from private key (32 bytes). */
    public function pubkeyCreate(string $privkey): string
    {
        self::check($privkey, 32, 'privkey');
        $out = FFI::new('uint8_t[33]');
        $rc = $this->ffi->ufsecp_pubkey_create($this->ctx, self::buf($privkey), $out);
        self::throwOnError($rc, 'pubkey_create');
        return FFI::string($out, 33);
    }

    /** Uncompressed public key (65 bytes). */
    public function pubkeyCreateUncompressed(string $privkey): string
    {
        self::check($privkey, 32, 'privkey');
        $out = FFI::new('uint8_t[65]');
        $rc = $this->ffi->ufsecp_pubkey_create_uncompressed($this->ctx, self::buf($privkey), $out);
        self::throwOnError($rc, 'pubkey_create_uncompressed');
        return FFI::string($out, 65);
    }

    /** Parse compressed (33) or uncompressed (65) pubkey → compressed 33 bytes. */
    public function pubkeyParse(string $pubkey): string
    {
        $out = FFI::new('uint8_t[33]');
        $rc = $this->ffi->ufsecp_pubkey_parse($this->ctx, self::buf($pubkey), strlen($pubkey), $out);
        self::throwOnError($rc, 'pubkey_parse');
        return FFI::string($out, 33);
    }

    /** X-only (32 bytes, BIP-340) public key from private key. */
    public function pubkeyXonly(string $privkey): string
    {
        self::check($privkey, 32, 'privkey');
        $out = FFI::new('uint8_t[32]');
        $rc = $this->ffi->ufsecp_pubkey_xonly($this->ctx, self::buf($privkey), $out);
        self::throwOnError($rc, 'pubkey_xonly');
        return FFI::string($out, 32);
    }

    /** Verify secret key validity. */
    public function seckeyVerify(string $privkey): bool
    {
        self::check($privkey, 32, 'privkey');
        return $this->ffi->ufsecp_seckey_verify($this->ctx, self::buf($privkey)) === self::OK;
    }

    /** Negate private key: key ← −key mod n. */
    public function seckeyNegate(string $privkey): string
    {
        self::check($privkey, 32, 'privkey');
        $buf = self::bufCopy($privkey, 32);
        $rc = $this->ffi->ufsecp_seckey_negate($this->ctx, $buf);
        self::throwOnError($rc, 'seckey_negate');
        return FFI::string($buf, 32);
    }

    /** Add tweak to private key: key ← (key + tweak) mod n. */
    public function seckeyTweakAdd(string $privkey, string $tweak): string
    {
        self::check($privkey, 32, 'privkey');
        self::check($tweak, 32, 'tweak');
        $buf = self::bufCopy($privkey, 32);
        $rc = $this->ffi->ufsecp_seckey_tweak_add($this->ctx, $buf, self::buf($tweak));
        self::throwOnError($rc, 'seckey_tweak_add');
        return FFI::string($buf, 32);
    }

    /** Multiply private key by tweak: key ← (key × tweak) mod n. */
    public function seckeyTweakMul(string $privkey, string $tweak): string
    {
        self::check($privkey, 32, 'privkey');
        self::check($tweak, 32, 'tweak');
        $buf = self::bufCopy($privkey, 32);
        $rc = $this->ffi->ufsecp_seckey_tweak_mul($this->ctx, $buf, self::buf($tweak));
        self::throwOnError($rc, 'seckey_tweak_mul');
        return FFI::string($buf, 32);
    }

    // ── ECDSA ────────────────────────────────────────────────────────────

    /** ECDSA sign (RFC 6979). Returns 64-byte compact signature. */
    public function ecdsaSign(string $msgHash, string $privkey): string
    {
        self::check($msgHash, 32, 'msgHash');
        self::check($privkey, 32, 'privkey');
        $sig = FFI::new('uint8_t[64]');
        $rc = $this->ffi->ufsecp_ecdsa_sign($this->ctx, self::buf($msgHash), self::buf($privkey), $sig);
        self::throwOnError($rc, 'ecdsa_sign');
        return FFI::string($sig, 64);
    }

    /** Verify ECDSA compact signature. Returns true if valid. */
    public function ecdsaVerify(string $msgHash, string $sig, string $pubkey): bool
    {
        self::check($msgHash, 32, 'msgHash');
        self::check($sig, 64, 'sig');
        self::check($pubkey, 33, 'pubkey');
        $rc = $this->ffi->ufsecp_ecdsa_verify(
            $this->ctx, self::buf($msgHash), self::buf($sig), self::buf($pubkey)
        );
        return $rc === self::OK;
    }

    /** Compact sig → DER format. */
    public function ecdsaSigToDer(string $sig): string
    {
        self::check($sig, 64, 'sig');
        $der = FFI::new('uint8_t[72]');
        $len = FFI::new('size_t');
        $len->cdata = 72;
        $rc = $this->ffi->ufsecp_ecdsa_sig_to_der($this->ctx, self::buf($sig), $der, FFI::addr($len));
        self::throwOnError($rc, 'ecdsa_sig_to_der');
        return FFI::string($der, $len->cdata);
    }

    /** DER → compact 64-byte sig. */
    public function ecdsaSigFromDer(string $der): string
    {
        $sig = FFI::new('uint8_t[64]');
        $rc = $this->ffi->ufsecp_ecdsa_sig_from_der($this->ctx, self::buf($der), strlen($der), $sig);
        self::throwOnError($rc, 'ecdsa_sig_from_der');
        return FFI::string($sig, 64);
    }

    // ── Recovery ─────────────────────────────────────────────────────────

    /** Sign with recovery id. Returns [signature(64), recid(int)]. */
    public function ecdsaSignRecoverable(string $msgHash, string $privkey): array
    {
        self::check($msgHash, 32, 'msgHash');
        self::check($privkey, 32, 'privkey');
        $sig = FFI::new('uint8_t[64]');
        $recid = FFI::new('int');
        $rc = $this->ffi->ufsecp_ecdsa_sign_recoverable(
            $this->ctx, self::buf($msgHash), self::buf($privkey), $sig, FFI::addr($recid)
        );
        self::throwOnError($rc, 'ecdsa_sign_recoverable');
        return [FFI::string($sig, 64), $recid->cdata];
    }

    /** Recover compressed public key (33 bytes). */
    public function ecdsaRecover(string $msgHash, string $sig, int $recid): string
    {
        self::check($msgHash, 32, 'msgHash');
        self::check($sig, 64, 'sig');
        $pubkey = FFI::new('uint8_t[33]');
        $rc = $this->ffi->ufsecp_ecdsa_recover(
            $this->ctx, self::buf($msgHash), self::buf($sig), $recid, $pubkey
        );
        self::throwOnError($rc, 'ecdsa_recover');
        return FFI::string($pubkey, 33);
    }

    // ── Schnorr ──────────────────────────────────────────────────────────

    /** BIP-340 Schnorr sign. Returns 64-byte signature. */
    public function schnorrSign(string $msg, string $privkey, string $auxRand): string
    {
        self::check($msg, 32, 'msg');
        self::check($privkey, 32, 'privkey');
        self::check($auxRand, 32, 'auxRand');
        $sig = FFI::new('uint8_t[64]');
        $rc = $this->ffi->ufsecp_schnorr_sign(
            $this->ctx, self::buf($msg), self::buf($privkey), self::buf($auxRand), $sig
        );
        self::throwOnError($rc, 'schnorr_sign');
        return FFI::string($sig, 64);
    }

    /** BIP-340 Schnorr verify. */
    public function schnorrVerify(string $msg, string $sig, string $pubkeyX): bool
    {
        self::check($msg, 32, 'msg');
        self::check($sig, 64, 'sig');
        self::check($pubkeyX, 32, 'pubkeyX');
        $rc = $this->ffi->ufsecp_schnorr_verify(
            $this->ctx, self::buf($msg), self::buf($sig), self::buf($pubkeyX)
        );
        return $rc === self::OK;
    }

    // ── ECDH ─────────────────────────────────────────────────────────────

    /** ECDH: SHA256(compressed shared point). */
    public function ecdh(string $privkey, string $pubkey): string
    {
        self::check($privkey, 32, 'privkey');
        self::check($pubkey, 33, 'pubkey');
        $out = FFI::new('uint8_t[32]');
        $rc = $this->ffi->ufsecp_ecdh($this->ctx, self::buf($privkey), self::buf($pubkey), $out);
        self::throwOnError($rc, 'ecdh');
        return FFI::string($out, 32);
    }

    /** ECDH x-only. */
    public function ecdhXonly(string $privkey, string $pubkey): string
    {
        self::check($privkey, 32, 'privkey');
        self::check($pubkey, 33, 'pubkey');
        $out = FFI::new('uint8_t[32]');
        $rc = $this->ffi->ufsecp_ecdh_xonly($this->ctx, self::buf($privkey), self::buf($pubkey), $out);
        self::throwOnError($rc, 'ecdh_xonly');
        return FFI::string($out, 32);
    }

    /** ECDH raw x-coordinate. */
    public function ecdhRaw(string $privkey, string $pubkey): string
    {
        self::check($privkey, 32, 'privkey');
        self::check($pubkey, 33, 'pubkey');
        $out = FFI::new('uint8_t[32]');
        $rc = $this->ffi->ufsecp_ecdh_raw($this->ctx, self::buf($privkey), self::buf($pubkey), $out);
        self::throwOnError($rc, 'ecdh_raw');
        return FFI::string($out, 32);
    }

    // ── Hashing (context-free) ───────────────────────────────────────────

    /** SHA-256 (hardware-accelerated when available). Returns 32 bytes. */
    public function sha256(string $data): string
    {
        $out = FFI::new('uint8_t[32]');
        $rc = $this->ffi->ufsecp_sha256(self::buf($data), strlen($data), $out);
        self::throwOnError($rc, 'sha256');
        return FFI::string($out, 32);
    }

    /** RIPEMD160(SHA256(data)). Returns 20 bytes. */
    public function hash160(string $data): string
    {
        $out = FFI::new('uint8_t[20]');
        $rc = $this->ffi->ufsecp_hash160(self::buf($data), strlen($data), $out);
        self::throwOnError($rc, 'hash160');
        return FFI::string($out, 20);
    }

    /** BIP-340 tagged hash. Returns 32 bytes. */
    public function taggedHash(string $tag, string $data): string
    {
        $out = FFI::new('uint8_t[32]');
        $rc = $this->ffi->ufsecp_tagged_hash($tag, self::buf($data), strlen($data), $out);
        self::throwOnError($rc, 'tagged_hash');
        return FFI::string($out, 32);
    }

    // ── Addresses ────────────────────────────────────────────────────────

    /** P2PKH address. */
    public function addrP2PKH(string $pubkey, int $network = self::NET_MAINNET): string
    {
        self::check($pubkey, 33, 'pubkey');
        return $this->getAddr(
            fn($buf, $len) => $this->ffi->ufsecp_addr_p2pkh($this->ctx, self::buf($pubkey), $network, $buf, $len)
        );
    }

    /** P2WPKH (Bech32, SegWit v0) address. */
    public function addrP2WPKH(string $pubkey, int $network = self::NET_MAINNET): string
    {
        self::check($pubkey, 33, 'pubkey');
        return $this->getAddr(
            fn($buf, $len) => $this->ffi->ufsecp_addr_p2wpkh($this->ctx, self::buf($pubkey), $network, $buf, $len)
        );
    }

    /** P2TR (Bech32m, Taproot) address from x-only key. */
    public function addrP2TR(string $internalKeyX, int $network = self::NET_MAINNET): string
    {
        self::check($internalKeyX, 32, 'internalKeyX');
        return $this->getAddr(
            fn($buf, $len) => $this->ffi->ufsecp_addr_p2tr($this->ctx, self::buf($internalKeyX), $network, $buf, $len)
        );
    }

    // ── WIF ──────────────────────────────────────────────────────────────

    /** Encode private key as WIF. */
    public function wifEncode(string $privkey, bool $compressed = true, int $network = self::NET_MAINNET): string
    {
        self::check($privkey, 32, 'privkey');
        return $this->getAddr(
            fn($buf, $len) => $this->ffi->ufsecp_wif_encode(
                $this->ctx, self::buf($privkey), $compressed ? 1 : 0, $network, $buf, $len
            )
        );
    }

    /** Decode WIF. Returns ['privkey' => string(32), 'compressed' => bool, 'network' => int]. */
    public function wifDecode(string $wif): array
    {
        $privkey = FFI::new('uint8_t[32]');
        $comp = FFI::new('int');
        $net = FFI::new('int');
        $rc = $this->ffi->ufsecp_wif_decode($this->ctx, $wif, $privkey, FFI::addr($comp), FFI::addr($net));
        self::throwOnError($rc, 'wif_decode');
        return [
            'privkey'    => FFI::string($privkey, 32),
            'compressed' => $comp->cdata === 1,
            'network'    => $net->cdata,
        ];
    }

    // ── BIP-32 ───────────────────────────────────────────────────────────

    /** Master key from seed (16–64 bytes). Returns opaque key (82 bytes). */
    public function bip32Master(string $seed): string
    {
        $len = strlen($seed);
        if ($len < 16 || $len > 64) {
            throw new InvalidArgumentException("Seed must be 16-64 bytes, got $len");
        }
        $key = FFI::new('ufsecp_bip32_key');
        $rc = $this->ffi->ufsecp_bip32_master($this->ctx, self::buf($seed), $len, FFI::addr($key));
        self::throwOnError($rc, 'bip32_master');
        return FFI::string(FFI::addr($key), 82);
    }

    /** Derive child key by index (>= 0x80000000 for hardened). */
    public function bip32Derive(string $parent, int $index): string
    {
        self::check($parent, 82, 'parent');
        $child = FFI::new('ufsecp_bip32_key');
        $parentKey = FFI::new('ufsecp_bip32_key');
        FFI::memcpy(FFI::addr($parentKey), $parent, 82);
        $rc = $this->ffi->ufsecp_bip32_derive($this->ctx, FFI::addr($parentKey), $index, FFI::addr($child));
        self::throwOnError($rc, 'bip32_derive');
        return FFI::string(FFI::addr($child), 82);
    }

    /** Derive from path, e.g. "m/44'/0'/0'/0/0". */
    public function bip32DerivePath(string $master, string $path): string
    {
        self::check($master, 82, 'master');
        $key = FFI::new('ufsecp_bip32_key');
        $masterKey = FFI::new('ufsecp_bip32_key');
        FFI::memcpy(FFI::addr($masterKey), $master, 82);
        $rc = $this->ffi->ufsecp_bip32_derive_path($this->ctx, FFI::addr($masterKey), $path, FFI::addr($key));
        self::throwOnError($rc, 'bip32_derive_path');
        return FFI::string(FFI::addr($key), 82);
    }

    /** Extract 32-byte private key from extended key. */
    public function bip32Privkey(string $key): string
    {
        self::check($key, 82, 'key');
        $privkey = FFI::new('uint8_t[32]');
        $bip32Key = FFI::new('ufsecp_bip32_key');
        FFI::memcpy(FFI::addr($bip32Key), $key, 82);
        $rc = $this->ffi->ufsecp_bip32_privkey($this->ctx, FFI::addr($bip32Key), $privkey);
        self::throwOnError($rc, 'bip32_privkey');
        return FFI::string($privkey, 32);
    }

    /** Extract compressed 33-byte public key from extended key. */
    public function bip32Pubkey(string $key): string
    {
        self::check($key, 82, 'key');
        $pubkey = FFI::new('uint8_t[33]');
        $bip32Key = FFI::new('ufsecp_bip32_key');
        FFI::memcpy(FFI::addr($bip32Key), $key, 82);
        $rc = $this->ffi->ufsecp_bip32_pubkey($this->ctx, FFI::addr($bip32Key), $pubkey);
        self::throwOnError($rc, 'bip32_pubkey');
        return FFI::string($pubkey, 33);
    }

    // ── Taproot ──────────────────────────────────────────────────────────

    /** Taproot output key. Returns ['outputKeyX' => string(32), 'parity' => int]. */
    public function taprootOutputKey(string $internalKeyX, ?string $merkleRoot = null): array
    {
        self::check($internalKeyX, 32, 'internalKeyX');
        $out = FFI::new('uint8_t[32]');
        $parity = FFI::new('int');
        $mr = $merkleRoot !== null ? self::buf($merkleRoot) : null;
        $rc = $this->ffi->ufsecp_taproot_output_key($this->ctx, self::buf($internalKeyX), $mr, $out, FFI::addr($parity));
        self::throwOnError($rc, 'taproot_output_key');
        return ['outputKeyX' => FFI::string($out, 32), 'parity' => $parity->cdata];
    }

    /** Tweak private key for Taproot spending. */
    public function taprootTweakSeckey(string $privkey, ?string $merkleRoot = null): string
    {
        self::check($privkey, 32, 'privkey');
        $out = FFI::new('uint8_t[32]');
        $mr = $merkleRoot !== null ? self::buf($merkleRoot) : null;
        $rc = $this->ffi->ufsecp_taproot_tweak_seckey($this->ctx, self::buf($privkey), $mr, $out);
        self::throwOnError($rc, 'taproot_tweak_seckey');
        return FFI::string($out, 32);
    }

    /** Verify Taproot commitment. */
    public function taprootVerify(string $outputKeyX, int $parity, string $internalKeyX, ?string $merkleRoot = null): bool
    {
        self::check($outputKeyX, 32, 'outputKeyX');
        self::check($internalKeyX, 32, 'internalKeyX');
        $mr = $merkleRoot !== null ? self::buf($merkleRoot) : null;
        $mrLen = $merkleRoot !== null ? strlen($merkleRoot) : 0;
        $rc = $this->ffi->ufsecp_taproot_verify(
            $this->ctx, self::buf($outputKeyX), $parity, self::buf($internalKeyX), $mr, $mrLen
        );
        return $rc === self::OK;
    }

    // ── Internal helpers ─────────────────────────────────────────────────

    private static function buf(string $data): FFI\CData
    {
        $len = strlen($data);
        $b = FFI::new('uint8_t[' . max(1, $len) . ']');
        if ($len > 0) {
            FFI::memcpy($b, $data, $len);
        }
        return $b;
    }

    private static function bufCopy(string $data, int $size): FFI\CData
    {
        $b = FFI::new("uint8_t[$size]");
        FFI::memcpy($b, $data, $size);
        return $b;
    }

    private function getAddr(callable $fn): string
    {
        $buf = FFI::new('char[128]');
        $len = FFI::new('size_t');
        $len->cdata = 128;
        $rc = $fn($buf, FFI::addr($len));
        self::throwOnError($rc, 'address');
        return FFI::string($buf, $len->cdata);
    }

    private static function check(string $data, int $expected, string $name): void
    {
        $len = strlen($data);
        if ($len !== $expected) {
            throw new InvalidArgumentException("$name must be $expected bytes, got $len");
        }
    }

    private static function throwOnError(int $rc, string $op): void
    {
        if ($rc !== self::OK) {
            throw new RuntimeException("ufsecp $op failed: " . self::errorName($rc));
        }
    }

    private static function errorName(int $rc): string
    {
        return match ($rc) {
            self::ERR_NULL_ARG      => 'null argument',
            self::ERR_BAD_KEY       => 'invalid private key',
            self::ERR_BAD_PUBKEY    => 'invalid public key',
            self::ERR_BAD_SIG       => 'invalid signature',
            self::ERR_BAD_INPUT     => 'bad input',
            self::ERR_VERIFY_FAIL   => 'verification failed',
            self::ERR_ARITH         => 'arithmetic error',
            self::ERR_SELFTEST      => 'selftest failed',
            self::ERR_INTERNAL      => 'internal error',
            self::ERR_BUF_TOO_SMALL => 'buffer too small',
            default                 => "unknown error ($rc)",
        };
    }

    private static function findLibrary(): string
    {
        $names = PHP_OS_FAMILY === 'Windows'
            ? ['ufsecp.dll']
            : (PHP_OS_FAMILY === 'Darwin'
                ? ['libufsecp.dylib']
                : ['libufsecp.so']);

        // 1. Environment variable
        $env = getenv('UFSECP_LIB');
        if ($env !== false && file_exists($env)) {
            return $env;
        }

        // 2. Next to this file
        foreach ($names as $name) {
            $p = __DIR__ . "/$name";
            if (file_exists($p)) {
                return $p;
            }
        }

        // 3. Common build dirs
        $root = realpath(__DIR__ . '/../..');
        if ($root) {
            $dirs = [
                "$root/build_rel",
                "$root/build-linux",
                "$root/build",
            ];
            foreach ($dirs as $dir) {
                foreach ($names as $name) {
                    $p = "$dir/$name";
                    if (file_exists($p)) {
                        return $p;
                    }
                }
            }
        }

        // 4. System default
        return $names[0];
    }
}
