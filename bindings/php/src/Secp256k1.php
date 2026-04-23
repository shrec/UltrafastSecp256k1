<?php

declare(strict_types=1);

/**
 * UltrafastSecp256k1 — PHP FFI Bindings
 *
 * High-performance secp256k1 elliptic curve cryptography.
 *
 * Requirements: PHP 8.1+ with ext-ffi enabled.
 *
 * Usage:
 *   $lib = new \Ultrafast\Secp256k1\Secp256k1();
 *   $privkey = str_repeat("\x00", 31) . "\x01";
 *   $pubkey = $lib->ecPubkeyCreate($privkey);
 *   $sig = $lib->ecdsaSign(str_repeat("\x00", 32), $privkey);
 *   $valid = $lib->ecdsaVerify(str_repeat("\x00", 32), $sig, $pubkey);
 */

namespace Ultrafast\Secp256k1;

use FFI;
use RuntimeException;
use InvalidArgumentException;

class Secp256k1
{
    public const NETWORK_MAINNET = 0;
    public const NETWORK_TESTNET = 1;

    private FFI $ffi;

    private const C_HEADER = <<<'CDEF'
    const char* secp256k1_version(void);
    int secp256k1_init(void);

    int secp256k1_ec_pubkey_create(const uint8_t* privkey, uint8_t* pubkey_out);
    int secp256k1_ec_pubkey_create_uncompressed(const uint8_t* privkey, uint8_t* pubkey_out);
    int secp256k1_ec_pubkey_parse(const uint8_t* input, size_t input_len, uint8_t* pubkey_out);
    int secp256k1_ec_seckey_verify(const uint8_t* privkey);
    int secp256k1_ec_privkey_negate(uint8_t* privkey);
    int secp256k1_ec_privkey_tweak_add(uint8_t* privkey, const uint8_t* tweak);
    int secp256k1_ec_privkey_tweak_mul(uint8_t* privkey, const uint8_t* tweak);

    int secp256k1_ecdsa_sign(const uint8_t* msg_hash, const uint8_t* privkey, uint8_t* sig_out);
    int secp256k1_ecdsa_verify(const uint8_t* msg_hash, const uint8_t* sig, const uint8_t* pubkey);
    int secp256k1_ecdsa_signature_serialize_der(const uint8_t* sig, uint8_t* der_out, size_t* der_len);

    int secp256k1_ecdsa_sign_recoverable(const uint8_t* msg_hash, const uint8_t* privkey, uint8_t* sig_out, int* recid_out);
    int secp256k1_ecdsa_recover(const uint8_t* msg_hash, const uint8_t* sig, int recid, uint8_t* pubkey_out);

    int secp256k1_schnorr_sign(const uint8_t* msg, const uint8_t* privkey, const uint8_t* aux_rand, uint8_t* sig_out);
    int secp256k1_schnorr_verify(const uint8_t* msg, const uint8_t* sig, const uint8_t* pubkey_x);
    int secp256k1_schnorr_pubkey(const uint8_t* privkey, uint8_t* pubkey_x_out);

    int secp256k1_ecdh(const uint8_t* privkey, const uint8_t* pubkey, uint8_t* secret_out);
    int secp256k1_ecdh_xonly(const uint8_t* privkey, const uint8_t* pubkey, uint8_t* secret_out);
    int secp256k1_ecdh_raw(const uint8_t* privkey, const uint8_t* pubkey, uint8_t* secret_out);

    void secp256k1_sha256(const uint8_t* data, size_t data_len, uint8_t* digest_out);
    void secp256k1_hash160(const uint8_t* data, size_t data_len, uint8_t* digest_out);
    void secp256k1_tagged_hash(const char* tag, const uint8_t* data, size_t data_len, uint8_t* digest_out);

    int secp256k1_address_p2pkh(const uint8_t* pubkey, int network, char* addr_out, size_t* addr_len);
    int secp256k1_address_p2wpkh(const uint8_t* pubkey, int network, char* addr_out, size_t* addr_len);
    int secp256k1_address_p2tr(const uint8_t* internal_key_x, int network, char* addr_out, size_t* addr_len);

    int secp256k1_wif_encode(const uint8_t* privkey, int compressed, int network, char* wif_out, size_t* wif_len);
    int secp256k1_wif_decode(const char* wif, uint8_t* privkey_out, int* compressed_out, int* network_out);

    int secp256k1_bip32_master_key(const uint8_t* seed, size_t seed_len, uint8_t* key_out);
    int secp256k1_bip32_derive_path(const uint8_t* master, const char* path, uint8_t* key_out);
    int secp256k1_bip32_get_privkey(const uint8_t* key, uint8_t* privkey_out);
    int secp256k1_bip32_get_pubkey(const uint8_t* key, uint8_t* pubkey_out);

    int secp256k1_taproot_output_key(const uint8_t* internal_key_x, const uint8_t* merkle_root, uint8_t* output_key_x_out, int* parity_out);
    int secp256k1_taproot_tweak_privkey(const uint8_t* privkey, const uint8_t* merkle_root, uint8_t* tweaked_privkey_out);
    int secp256k1_taproot_verify_commitment(const uint8_t* output_key_x, int output_key_parity, const uint8_t* internal_key_x, const uint8_t* merkle_root, size_t merkle_root_len);
    CDEF;

    /**
     * @param string|null $libPath Path to the shared library. Auto-detected if null.
     */
    public function __construct(?string $libPath = null)
    {
        $path = $libPath ?? $this->findLibrary();
        $this->ffi = FFI::cdef(self::C_HEADER, $path);

        $rc = $this->ffi->secp256k1_init();
        if ($rc !== 0) {
            throw new RuntimeException('secp256k1_init() failed: library selftest failure');
        }
    }

    public function version(): string
    {
        return FFI::string($this->ffi->secp256k1_version());
    }

    // ── Key Operations ───────────────────────────────────────────────────

    /** Compressed public key (33 bytes) from private key (32 bytes). */
    public function ecPubkeyCreate(string $privkey): string
    {
        self::check($privkey, 32, 'privkey');
        $out = $this->alloc(33);
        $rc = $this->ffi->secp256k1_ec_pubkey_create($this->ptr($privkey), $out);
        if ($rc !== 0) throw new InvalidArgumentException('Invalid private key');
        return FFI::string($out, 33);
    }

    /** Uncompressed public key (65 bytes). */
    public function ecPubkeyCreateUncompressed(string $privkey): string
    {
        self::check($privkey, 32, 'privkey');
        $out = $this->alloc(65);
        $rc = $this->ffi->secp256k1_ec_pubkey_create_uncompressed($this->ptr($privkey), $out);
        if ($rc !== 0) throw new InvalidArgumentException('Invalid private key');
        return FFI::string($out, 65);
    }

    /** Parse compressed (33) or uncompressed (65) pubkey. Returns compressed. */
    public function ecPubkeyParse(string $pubkey): string
    {
        $out = $this->alloc(33);
        $rc = $this->ffi->secp256k1_ec_pubkey_parse($this->ptr($pubkey), strlen($pubkey), $out);
        if ($rc !== 0) throw new InvalidArgumentException('Invalid public key');
        return FFI::string($out, 33);
    }

    /** Verify secret key validity. */
    public function ecSeckeyVerify(string $privkey): bool
    {
        self::check($privkey, 32, 'privkey');
        return $this->ffi->secp256k1_ec_seckey_verify($this->ptr($privkey)) === 1;
    }

    /** Negate private key. */
    public function ecPrivkeyNegate(string $privkey): string
    {
        self::check($privkey, 32, 'privkey');
        $buf = $this->allocCopy($privkey, 32);
        $rc = $this->ffi->secp256k1_ec_privkey_negate($buf);
        if ($rc !== 0) throw new \RuntimeException('ec_privkey_negate failed: invalid (zero) key');
        return FFI::string($buf, 32);
    }

    /** Add tweak to private key. */
    public function ecPrivkeyTweakAdd(string $privkey, string $tweak): string
    {
        self::check($privkey, 32, 'privkey');
        self::check($tweak, 32, 'tweak');
        $buf = $this->allocCopy($privkey, 32);
        $rc = $this->ffi->secp256k1_ec_privkey_tweak_add($buf, $this->ptr($tweak));
        if ($rc !== 0) throw new RuntimeException('Tweak add produced invalid key');
        return FFI::string($buf, 32);
    }

    /** Multiply private key by tweak. */
    public function ecPrivkeyTweakMul(string $privkey, string $tweak): string
    {
        self::check($privkey, 32, 'privkey');
        self::check($tweak, 32, 'tweak');
        $buf = $this->allocCopy($privkey, 32);
        $rc = $this->ffi->secp256k1_ec_privkey_tweak_mul($buf, $this->ptr($tweak));
        if ($rc !== 0) throw new RuntimeException('Tweak mul produced invalid key');
        return FFI::string($buf, 32);
    }

    // ── ECDSA ────────────────────────────────────────────────────────────

    /** Sign with ECDSA (RFC 6979). Returns 64-byte compact signature. */
    public function ecdsaSign(string $msgHash, string $privkey): string
    {
        self::check($msgHash, 32, 'msgHash');
        self::check($privkey, 32, 'privkey');
        $sig = $this->alloc(64);
        $rc = $this->ffi->secp256k1_ecdsa_sign($this->ptr($msgHash), $this->ptr($privkey), $sig);
        if ($rc !== 0) throw new RuntimeException('ECDSA signing failed');
        return FFI::string($sig, 64);
    }

    /** Verify ECDSA signature. */
    public function ecdsaVerify(string $msgHash, string $sig, string $pubkey): bool
    {
        self::check($msgHash, 32, 'msgHash');
        self::check($sig, 64, 'sig');
        self::check($pubkey, 33, 'pubkey');
        return $this->ffi->secp256k1_ecdsa_verify(
            $this->ptr($msgHash), $this->ptr($sig), $this->ptr($pubkey)
        ) === 1;
    }

    /** Serialize compact sig to DER. */
    public function ecdsaSerializeDer(string $sig): string
    {
        self::check($sig, 64, 'sig');
        $der = $this->alloc(72);
        $len = $this->allocSize(72);
        $rc = $this->ffi->secp256k1_ecdsa_signature_serialize_der($this->ptr($sig), $der, $len);
        if ($rc !== 0) throw new RuntimeException('DER serialization failed');
        return FFI::string($der, $len[0]);
    }

    // ── Recovery ─────────────────────────────────────────────────────────

    /** Sign with recovery id. Returns [signature(64), recid(int)]. */
    public function ecdsaSignRecoverable(string $msgHash, string $privkey): array
    {
        self::check($msgHash, 32, 'msgHash');
        self::check($privkey, 32, 'privkey');
        $sig = $this->alloc(64);
        $recid = $this->allocInt();
        $rc = $this->ffi->secp256k1_ecdsa_sign_recoverable(
            $this->ptr($msgHash), $this->ptr($privkey), $sig, $recid
        );
        if ($rc !== 0) throw new RuntimeException('Recoverable signing failed');
        return [FFI::string($sig, 64), $recid[0]];
    }

    /** Recover public key. Returns 33-byte compressed pubkey. */
    public function ecdsaRecover(string $msgHash, string $sig, int $recid): string
    {
        self::check($msgHash, 32, 'msgHash');
        self::check($sig, 64, 'sig');
        $pubkey = $this->alloc(33);
        $rc = $this->ffi->secp256k1_ecdsa_recover(
            $this->ptr($msgHash), $this->ptr($sig), $recid, $pubkey
        );
        if ($rc !== 0) throw new RuntimeException('Recovery failed');
        return FFI::string($pubkey, 33);
    }

    // ── Schnorr ──────────────────────────────────────────────────────────

    /** BIP-340 Schnorr sign. Returns 64-byte signature. */
    public function schnorrSign(string $msg, string $privkey, string $auxRand): string
    {
        self::check($msg, 32, 'msg');
        self::check($privkey, 32, 'privkey');
        self::check($auxRand, 32, 'auxRand');
        $sig = $this->alloc(64);
        $rc = $this->ffi->secp256k1_schnorr_sign(
            $this->ptr($msg), $this->ptr($privkey), $this->ptr($auxRand), $sig
        );
        if ($rc !== 0) throw new RuntimeException('Schnorr signing failed');
        return FFI::string($sig, 64);
    }

    /** Verify Schnorr signature. */
    public function schnorrVerify(string $msg, string $sig, string $pubkeyX): bool
    {
        self::check($msg, 32, 'msg');
        self::check($sig, 64, 'sig');
        self::check($pubkeyX, 32, 'pubkeyX');
        return $this->ffi->secp256k1_schnorr_verify(
            $this->ptr($msg), $this->ptr($sig), $this->ptr($pubkeyX)
        ) === 1;
    }

    /** Get x-only public key (32 bytes). */
    public function schnorrPubkey(string $privkey): string
    {
        self::check($privkey, 32, 'privkey');
        $out = $this->alloc(32);
        $rc = $this->ffi->secp256k1_schnorr_pubkey($this->ptr($privkey), $out);
        if ($rc !== 0) throw new InvalidArgumentException('Invalid private key');
        return FFI::string($out, 32);
    }

    // ── ECDH ─────────────────────────────────────────────────────────────

    /** ECDH: SHA256(compressed shared point). */
    public function ecdh(string $privkey, string $pubkey): string
    {
        self::check($privkey, 32, 'privkey');
        self::check($pubkey, 33, 'pubkey');
        $out = $this->alloc(32);
        $rc = $this->ffi->secp256k1_ecdh($this->ptr($privkey), $this->ptr($pubkey), $out);
        if ($rc !== 0) throw new RuntimeException('ECDH failed');
        return FFI::string($out, 32);
    }

    /** ECDH x-only. */
    public function ecdhXonly(string $privkey, string $pubkey): string
    {
        self::check($privkey, 32, 'privkey');
        self::check($pubkey, 33, 'pubkey');
        $out = $this->alloc(32);
        $rc = $this->ffi->secp256k1_ecdh_xonly($this->ptr($privkey), $this->ptr($pubkey), $out);
        if ($rc !== 0) throw new RuntimeException('ECDH xonly failed');
        return FFI::string($out, 32);
    }

    /** ECDH raw x-coordinate. */
    public function ecdhRaw(string $privkey, string $pubkey): string
    {
        self::check($privkey, 32, 'privkey');
        self::check($pubkey, 33, 'pubkey');
        $out = $this->alloc(32);
        $rc = $this->ffi->secp256k1_ecdh_raw($this->ptr($privkey), $this->ptr($pubkey), $out);
        if ($rc !== 0) throw new RuntimeException('ECDH raw failed');
        return FFI::string($out, 32);
    }

    // ── Hashing ──────────────────────────────────────────────────────────

    /** SHA-256. Returns 32 bytes. */
    public function sha256(string $data): string
    {
        $out = $this->alloc(32);
        $this->ffi->secp256k1_sha256($this->ptr($data), strlen($data), $out);
        return FFI::string($out, 32);
    }

    /** HASH160. Returns 20 bytes. */
    public function hash160(string $data): string
    {
        $out = $this->alloc(20);
        $this->ffi->secp256k1_hash160($this->ptr($data), strlen($data), $out);
        return FFI::string($out, 20);
    }

    /** Tagged hash (BIP-340). Returns 32 bytes. */
    public function taggedHash(string $tag, string $data): string
    {
        $out = $this->alloc(32);
        $this->ffi->secp256k1_tagged_hash($tag, $this->ptr($data), strlen($data), $out);
        return FFI::string($out, 32);
    }

    // ── Addresses ────────────────────────────────────────────────────────

    /** P2PKH address. */
    public function addressP2PKH(string $pubkey, int $network = self::NETWORK_MAINNET): string
    {
        self::check($pubkey, 33, 'pubkey');
        return $this->getAddress(
            fn($buf, $len) => $this->ffi->secp256k1_address_p2pkh($this->ptr($pubkey), $network, $buf, $len)
        );
    }

    /** P2WPKH address. */
    public function addressP2WPKH(string $pubkey, int $network = self::NETWORK_MAINNET): string
    {
        self::check($pubkey, 33, 'pubkey');
        return $this->getAddress(
            fn($buf, $len) => $this->ffi->secp256k1_address_p2wpkh($this->ptr($pubkey), $network, $buf, $len)
        );
    }

    /** P2TR address from x-only key. */
    public function addressP2TR(string $internalKeyX, int $network = self::NETWORK_MAINNET): string
    {
        self::check($internalKeyX, 32, 'internalKeyX');
        return $this->getAddress(
            fn($buf, $len) => $this->ffi->secp256k1_address_p2tr($this->ptr($internalKeyX), $network, $buf, $len)
        );
    }

    // ── WIF ──────────────────────────────────────────────────────────────

    /** Encode private key as WIF. */
    public function wifEncode(string $privkey, bool $compressed = true, int $network = self::NETWORK_MAINNET): string
    {
        self::check($privkey, 32, 'privkey');
        return $this->getAddress(
            fn($buf, $len) => $this->ffi->secp256k1_wif_encode(
                $this->ptr($privkey), $compressed ? 1 : 0, $network, $buf, $len
            )
        );
    }

    /** Decode WIF. Returns [privkey(32), compressed(bool), network(int)]. */
    public function wifDecode(string $wif): array
    {
        $privkey = $this->alloc(32);
        $comp = $this->allocInt();
        $net = $this->allocInt();
        $rc = $this->ffi->secp256k1_wif_decode($wif, $privkey, $comp, $net);
        if ($rc !== 0) throw new InvalidArgumentException('Invalid WIF string');
        return [FFI::string($privkey, 32), $comp[0] === 1, $net[0]];
    }

    // ── BIP-32 ───────────────────────────────────────────────────────────

    /** Master key from seed. Returns 79-byte opaque key. */
    public function bip32MasterKey(string $seed): string
    {
        $len = strlen($seed);
        if ($len < 16 || $len > 64) throw new InvalidArgumentException('Seed must be 16-64 bytes');
        $key = $this->alloc(79);
        $rc = $this->ffi->secp256k1_bip32_master_key($this->ptr($seed), $len, $key);
        if ($rc !== 0) throw new RuntimeException('Master key generation failed');
        return FFI::string($key, 79);
    }

    /** Derive key from path. */
    public function bip32DerivePath(string $masterKey, string $path): string
    {
        self::check($masterKey, 79, 'masterKey');
        $key = $this->alloc(79);
        $rc = $this->ffi->secp256k1_bip32_derive_path($this->ptr($masterKey), $path, $key);
        if ($rc !== 0) throw new RuntimeException("Path derivation failed: $path");
        return FFI::string($key, 79);
    }

    /** Get private key from extended key. */
    public function bip32GetPrivkey(string $key): string
    {
        self::check($key, 79, 'key');
        $privkey = $this->alloc(32);
        $rc = $this->ffi->secp256k1_bip32_get_privkey($this->ptr($key), $privkey);
        if ($rc !== 0) throw new RuntimeException('Key is not a private key');
        return FFI::string($privkey, 32);
    }

    /** Get compressed public key from extended key. */
    public function bip32GetPubkey(string $key): string
    {
        self::check($key, 79, 'key');
        $pubkey = $this->alloc(33);
        $rc = $this->ffi->secp256k1_bip32_get_pubkey($this->ptr($key), $pubkey);
        if ($rc !== 0) throw new RuntimeException('Public key extraction failed');
        return FFI::string($pubkey, 33);
    }

    // ── Taproot ──────────────────────────────────────────────────────────

    /** Taproot output key. Returns [outputKeyX(32), parity(int)]. */
    public function taprootOutputKey(string $internalKeyX, ?string $merkleRoot = null): array
    {
        self::check($internalKeyX, 32, 'internalKeyX');
        $out = $this->alloc(32);
        $parity = $this->allocInt();
        $mr = $merkleRoot !== null ? $this->ptr($merkleRoot) : null;
        $rc = $this->ffi->secp256k1_taproot_output_key($this->ptr($internalKeyX), $mr, $out, $parity);
        if ($rc !== 0) throw new RuntimeException('Taproot output key failed');
        return [FFI::string($out, 32), $parity[0]];
    }

    /** Tweak private key for Taproot. */
    public function taprootTweakPrivkey(string $privkey, ?string $merkleRoot = null): string
    {
        self::check($privkey, 32, 'privkey');
        $out = $this->alloc(32);
        $mr = $merkleRoot !== null ? $this->ptr($merkleRoot) : null;
        $rc = $this->ffi->secp256k1_taproot_tweak_privkey($this->ptr($privkey), $mr, $out);
        if ($rc !== 0) throw new RuntimeException('Taproot tweak failed');
        return FFI::string($out, 32);
    }

    // ── Internal helpers ─────────────────────────────────────────────────

    private function alloc(int $size): FFI\CData
    {
        return FFI::new("uint8_t[$size]");
    }

    private function allocCopy(string $data, int $size): FFI\CData
    {
        $buf = FFI::new("uint8_t[$size]");
        FFI::memcpy($buf, $data, $size);
        return $buf;
    }

    private function allocSize(int $value): FFI\CData
    {
        $s = FFI::new('size_t');
        $s->cdata = $value;
        return FFI::addr($s);
    }

    private function allocInt(): FFI\CData
    {
        $i = FFI::new('int');
        $i->cdata = 0;
        return FFI::addr($i);
    }

    private function ptr(string $data): FFI\CData
    {
        $len = strlen($data);
        $buf = FFI::new("uint8_t[$len]");
        FFI::memcpy($buf, $data, $len);
        return $buf;
    }

    private function getAddress(callable $fn): string
    {
        $buf = FFI::new('char[128]');
        $len = FFI::new('size_t');
        $len->cdata = 128;
        $lenPtr = FFI::addr($len);
        $rc = $fn($buf, $lenPtr);
        if ($rc !== 0) throw new RuntimeException('Address generation failed');
        return FFI::string($buf, $len->cdata);
    }

    private function findLibrary(): string
    {
        $libNames = PHP_OS_FAMILY === 'Windows'
            ? ['ultrafast_secp256k1.dll', 'libultrafast_secp256k1.dll']
            : (PHP_OS_FAMILY === 'Darwin'
                ? ['libultrafast_secp256k1.dylib']
                : ['libultrafast_secp256k1.so']);

        // 1. Environment variable
        $env = getenv('ULTRAFAST_SECP256K1_LIB');
        if ($env !== false && file_exists($env)) return $env;

        // 2. Next to this file
        $base = __DIR__;
        foreach ($libNames as $name) {
            $p = "$base/$name";
            if (file_exists($p)) return $p;
        }

        // 3. Common build dirs
        $root = realpath("$base/../..");
        if ($root) {
            $dirs = [
                "$root/bindings/c_api/build",
                "$root/bindings/c_api/build/Release",
                "$root/build_rel",
                "$root/build-linux",
            ];
            foreach ($dirs as $dir) {
                foreach ($libNames as $name) {
                    $p = "$dir/$name";
                    if (file_exists($p)) return $p;
                }
            }
        }

        // 4. System default
        return 'libultrafast_secp256k1' . (PHP_OS_FAMILY === 'Windows' ? '.dll' : '.so');
    }

    private static function check(string $data, int $expected, string $name): void
    {
        $len = strlen($data);
        if ($len !== $expected) {
            throw new InvalidArgumentException("$name must be $expected bytes, got $len");
        }
    }
}
