// Package ufsecp provides Go bindings for the UltrafastSecp256k1 stable C ABI (ufsecp).
//
// High-performance secp256k1 elliptic curve cryptography: key operations,
// ECDSA (RFC 6979), Schnorr (BIP-340), ECDH, recovery, hashing, Bitcoin
// addresses (P2PKH/P2WPKH/P2TR), WIF, BIP-32, and Taproot.
//
// Dual-layer constant-time architecture: secret-dependent operations always
// use the CT layer; public operations use the fast layer. Both layers are
// always active — no flag, no opt-in.
//
// Build: the ufsecp shared library must be installed or pointed to via
// CGO_LDFLAGS / LD_LIBRARY_PATH.
package ufsecp

/*
#cgo LDFLAGS: -lufsecp
#cgo CFLAGS: -I${SRCDIR}/../../include/ufsecp

#include "ufsecp.h"
#include <stdlib.h>
#include <string.h>
*/
import "C"
import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

// ── Constants ────────────────────────────────────────────────────────────────

const (
	ExpectedABI        = 4
	PrivkeyLen         = 32
	PubkeyCompressedLen = 33
	PubkeyUncompressedLen = 65
	PubkeyXonlyLen     = 32
	SigCompactLen      = 64
	SigDERMaxLen       = 72
	HashLen            = 32
	Hash160Len         = 20
	SharedSecretLen    = 32
	BIP32KeySize       = 82 // 78 data + 1 is_private + 3 padding
)

// Network selects mainnet or testnet address encoding.
type Network int

const (
	Mainnet Network = C.UFSECP_NET_MAINNET
	Testnet Network = C.UFSECP_NET_TESTNET
)

// Common errors.
var (
	ErrNullArg      = errors.New("ufsecp: null argument")
	ErrBadKey       = errors.New("ufsecp: invalid private key")
	ErrBadPubkey    = errors.New("ufsecp: invalid public key")
	ErrBadSig       = errors.New("ufsecp: invalid signature")
	ErrBadInput     = errors.New("ufsecp: bad input")
	ErrVerifyFail   = errors.New("ufsecp: verification failed")
	ErrArith        = errors.New("ufsecp: arithmetic error")
	ErrSelftest     = errors.New("ufsecp: selftest failed")
	ErrInternal     = errors.New("ufsecp: internal error")
	ErrBufTooSmall  = errors.New("ufsecp: buffer too small")
)

func errFromCode(code C.int) error {
	switch code {
	case C.UFSECP_OK:
		return nil
	case C.UFSECP_ERR_NULL_ARG:
		return ErrNullArg
	case C.UFSECP_ERR_BAD_KEY:
		return ErrBadKey
	case C.UFSECP_ERR_BAD_PUBKEY:
		return ErrBadPubkey
	case C.UFSECP_ERR_BAD_SIG:
		return ErrBadSig
	case C.UFSECP_ERR_BAD_INPUT:
		return ErrBadInput
	case C.UFSECP_ERR_VERIFY_FAIL:
		return ErrVerifyFail
	case C.UFSECP_ERR_ARITH:
		return ErrArith
	case C.UFSECP_ERR_SELFTEST:
		return ErrSelftest
	case C.UFSECP_ERR_BUF_TOO_SMALL:
		return ErrBufTooSmall
	default:
		return fmt.Errorf("ufsecp: error code %d", code)
	}
}

// ── Context ──────────────────────────────────────────────────────────────────

// Context wraps an opaque ufsecp_ctx. One per goroutine (or externally synced).
// Dual-layer CT is always active — no flags needed.
type Context struct {
	ctx *C.ufsecp_ctx
}

// NewContext creates a new ufsecp context (runs selftest on first call).
func NewContext() (*Context, error) {
	if abi := ABIVersion(); abi != ExpectedABI {
		return nil, fmt.Errorf("ufsecp: ABI mismatch: wrapper expects ABI %d, lib reports ABI %d", ExpectedABI, abi)
	}
	var ctx *C.ufsecp_ctx
	rc := C.ufsecp_ctx_create(&ctx)
	if rc != C.UFSECP_OK {
		return nil, errFromCode(rc)
	}
	c := &Context{ctx: ctx}
	runtime.SetFinalizer(c, func(c *Context) { c.Destroy() })
	return c, nil
}

// Clone deep-copies the context.
func (c *Context) Clone() (*Context, error) {
	var ctx *C.ufsecp_ctx
	rc := C.ufsecp_ctx_clone(c.ctx, &ctx)
	if rc != C.UFSECP_OK {
		return nil, errFromCode(rc)
	}
	cl := &Context{ctx: ctx}
	runtime.SetFinalizer(cl, func(cl *Context) { cl.Destroy() })
	return cl, nil
}

// Destroy frees the context. Safe to call multiple times.
func (c *Context) Destroy() {
	if c.ctx != nil {
		runtime.SetFinalizer(c, nil)
		C.ufsecp_ctx_destroy(c.ctx)
		c.ctx = nil
	}
}

// LastError returns the last error code on this context.
func (c *Context) LastError() int {
	return int(C.ufsecp_last_error(c.ctx))
}

// LastErrorMsg returns the last error message on this context.
func (c *Context) LastErrorMsg() string {
	return C.GoString(C.ufsecp_last_error_msg(c.ctx))
}

// ── Version ──────────────────────────────────────────────────────────────────

// Version returns the packed version number.
func Version() uint {
	return uint(C.ufsecp_version())
}

// ABIVersion returns the ABI version.
func ABIVersion() uint {
	return uint(C.ufsecp_abi_version())
}

// VersionString returns the human-readable version string (e.g. "3.4.0").
func VersionString() string {
	return C.GoString(C.ufsecp_version_string())
}

// ── Private key utilities ────────────────────────────────────────────────────

// SeckeyVerify checks whether a 32-byte private key is valid (non-zero, < order).
func (c *Context) SeckeyVerify(privkey [32]byte) bool {
	rc := C.ufsecp_seckey_verify(c.ctx, (*C.uint8_t)(unsafe.Pointer(&privkey[0])))
	return rc == C.UFSECP_OK
}

// SeckeyNegate negates a private key in-place: key ← −key mod n.
func (c *Context) SeckeyNegate(privkey [32]byte) ([32]byte, error) {
	out := privkey
	rc := C.ufsecp_seckey_negate(c.ctx, (*C.uint8_t)(unsafe.Pointer(&out[0])))
	return out, errFromCode(rc)
}

// SeckeyTweakAdd: privkey ← (privkey + tweak) mod n.
func (c *Context) SeckeyTweakAdd(privkey, tweak [32]byte) ([32]byte, error) {
	out := privkey
	rc := C.ufsecp_seckey_tweak_add(
		c.ctx,
		(*C.uint8_t)(unsafe.Pointer(&out[0])),
		(*C.uint8_t)(unsafe.Pointer(&tweak[0])),
	)
	return out, errFromCode(rc)
}

// SeckeyTweakMul: privkey ← (privkey × tweak) mod n.
func (c *Context) SeckeyTweakMul(privkey, tweak [32]byte) ([32]byte, error) {
	out := privkey
	rc := C.ufsecp_seckey_tweak_mul(
		c.ctx,
		(*C.uint8_t)(unsafe.Pointer(&out[0])),
		(*C.uint8_t)(unsafe.Pointer(&tweak[0])),
	)
	return out, errFromCode(rc)
}

// ── Public key ───────────────────────────────────────────────────────────────

// PubkeyCreate computes a compressed (33-byte) public key from a 32-byte private key.
func (c *Context) PubkeyCreate(privkey [32]byte) ([33]byte, error) {
	var out [33]byte
	rc := C.ufsecp_pubkey_create(
		c.ctx,
		(*C.uint8_t)(unsafe.Pointer(&privkey[0])),
		(*C.uint8_t)(unsafe.Pointer(&out[0])),
	)
	return out, errFromCode(rc)
}

// PubkeyCreateUncompressed computes an uncompressed (65-byte) public key.
func (c *Context) PubkeyCreateUncompressed(privkey [32]byte) ([65]byte, error) {
	var out [65]byte
	rc := C.ufsecp_pubkey_create_uncompressed(
		c.ctx,
		(*C.uint8_t)(unsafe.Pointer(&privkey[0])),
		(*C.uint8_t)(unsafe.Pointer(&out[0])),
	)
	return out, errFromCode(rc)
}

// PubkeyParse parses a compressed (33) or uncompressed (65) public key.
// Returns compressed form.
func (c *Context) PubkeyParse(input []byte) ([33]byte, error) {
	if len(input) != 33 && len(input) != 65 {
		return [33]byte{}, fmt.Errorf("ufsecp: pubkey must be 33 or 65 bytes, got %d", len(input))
	}
	var out [33]byte
	rc := C.ufsecp_pubkey_parse(
		c.ctx,
		(*C.uint8_t)(unsafe.Pointer(&input[0])),
		C.size_t(len(input)),
		(*C.uint8_t)(unsafe.Pointer(&out[0])),
	)
	return out, errFromCode(rc)
}

// PubkeyXonly derives a 32-byte x-only (BIP-340) public key from a private key.
func (c *Context) PubkeyXonly(privkey [32]byte) ([32]byte, error) {
	var out [32]byte
	rc := C.ufsecp_pubkey_xonly(
		c.ctx,
		(*C.uint8_t)(unsafe.Pointer(&privkey[0])),
		(*C.uint8_t)(unsafe.Pointer(&out[0])),
	)
	return out, errFromCode(rc)
}

// ── ECDSA ────────────────────────────────────────────────────────────────────

// EcdsaSign signs a 32-byte hash with ECDSA (RFC 6979). Returns 64-byte compact sig.
func (c *Context) EcdsaSign(msgHash, privkey [32]byte) ([64]byte, error) {
	var sig [64]byte
	rc := C.ufsecp_ecdsa_sign(
		c.ctx,
		(*C.uint8_t)(unsafe.Pointer(&msgHash[0])),
		(*C.uint8_t)(unsafe.Pointer(&privkey[0])),
		(*C.uint8_t)(unsafe.Pointer(&sig[0])),
	)
	return sig, errFromCode(rc)
}

// EcdsaVerify verifies an ECDSA compact signature.
// Returns nil on valid, ErrVerifyFail on invalid.
func (c *Context) EcdsaVerify(msgHash [32]byte, sig [64]byte, pubkey [33]byte) error {
	rc := C.ufsecp_ecdsa_verify(
		c.ctx,
		(*C.uint8_t)(unsafe.Pointer(&msgHash[0])),
		(*C.uint8_t)(unsafe.Pointer(&sig[0])),
		(*C.uint8_t)(unsafe.Pointer(&pubkey[0])),
	)
	return errFromCode(rc)
}

// EcdsaSigToDER encodes a compact 64-byte sig into DER format.
func (c *Context) EcdsaSigToDER(sig [64]byte) ([]byte, error) {
	var der [72]byte
	derLen := C.size_t(72)
	rc := C.ufsecp_ecdsa_sig_to_der(
		c.ctx,
		(*C.uint8_t)(unsafe.Pointer(&sig[0])),
		(*C.uint8_t)(unsafe.Pointer(&der[0])),
		&derLen,
	)
	if err := errFromCode(rc); err != nil {
		return nil, err
	}
	return append([]byte{}, der[:derLen]...), nil
}

// EcdsaSigFromDER decodes a DER-encoded signature to compact 64 bytes.
func (c *Context) EcdsaSigFromDER(der []byte) ([64]byte, error) {
	var sig [64]byte
	rc := C.ufsecp_ecdsa_sig_from_der(
		c.ctx,
		(*C.uint8_t)(unsafe.Pointer(&der[0])),
		C.size_t(len(der)),
		(*C.uint8_t)(unsafe.Pointer(&sig[0])),
	)
	return sig, errFromCode(rc)
}

// ── Recovery ─────────────────────────────────────────────────────────────────

// EcdsaSignRecoverable signs with a recovery id (0-3).
func (c *Context) EcdsaSignRecoverable(msgHash, privkey [32]byte) (sig [64]byte, recid int, err error) {
	var cRecid C.int
	rc := C.ufsecp_ecdsa_sign_recoverable(
		c.ctx,
		(*C.uint8_t)(unsafe.Pointer(&msgHash[0])),
		(*C.uint8_t)(unsafe.Pointer(&privkey[0])),
		(*C.uint8_t)(unsafe.Pointer(&sig[0])),
		&cRecid,
	)
	return sig, int(cRecid), errFromCode(rc)
}

// EcdsaRecover recovers a compressed public key from a recoverable signature.
func (c *Context) EcdsaRecover(msgHash [32]byte, sig [64]byte, recid int) ([33]byte, error) {
	var pubkey [33]byte
	rc := C.ufsecp_ecdsa_recover(
		c.ctx,
		(*C.uint8_t)(unsafe.Pointer(&msgHash[0])),
		(*C.uint8_t)(unsafe.Pointer(&sig[0])),
		C.int(recid),
		(*C.uint8_t)(unsafe.Pointer(&pubkey[0])),
	)
	return pubkey, errFromCode(rc)
}

// ── Schnorr (BIP-340) ───────────────────────────────────────────────────────

// SchnorrSign creates a BIP-340 Schnorr signature (64 bytes).
func (c *Context) SchnorrSign(msg, privkey, auxRand [32]byte) ([64]byte, error) {
	var sig [64]byte
	rc := C.ufsecp_schnorr_sign(
		c.ctx,
		(*C.uint8_t)(unsafe.Pointer(&msg[0])),
		(*C.uint8_t)(unsafe.Pointer(&privkey[0])),
		(*C.uint8_t)(unsafe.Pointer(&auxRand[0])),
		(*C.uint8_t)(unsafe.Pointer(&sig[0])),
	)
	return sig, errFromCode(rc)
}

// SchnorrVerify verifies a BIP-340 Schnorr signature.
// Returns nil on valid, ErrVerifyFail on invalid.
func (c *Context) SchnorrVerify(msg [32]byte, sig [64]byte, pubkeyX [32]byte) error {
	rc := C.ufsecp_schnorr_verify(
		c.ctx,
		(*C.uint8_t)(unsafe.Pointer(&msg[0])),
		(*C.uint8_t)(unsafe.Pointer(&sig[0])),
		(*C.uint8_t)(unsafe.Pointer(&pubkeyX[0])),
	)
	return errFromCode(rc)
}

// ── ECDH ─────────────────────────────────────────────────────────────────────

// ECDH computes a shared secret: SHA256(compressed_shared_point).
func (c *Context) ECDH(privkey [32]byte, pubkey [33]byte) ([32]byte, error) {
	var out [32]byte
	rc := C.ufsecp_ecdh(
		c.ctx,
		(*C.uint8_t)(unsafe.Pointer(&privkey[0])),
		(*C.uint8_t)(unsafe.Pointer(&pubkey[0])),
		(*C.uint8_t)(unsafe.Pointer(&out[0])),
	)
	return out, errFromCode(rc)
}

// ECDHXonly computes a shared secret from x-coordinate only.
func (c *Context) ECDHXonly(privkey [32]byte, pubkey [33]byte) ([32]byte, error) {
	var out [32]byte
	rc := C.ufsecp_ecdh_xonly(
		c.ctx,
		(*C.uint8_t)(unsafe.Pointer(&privkey[0])),
		(*C.uint8_t)(unsafe.Pointer(&pubkey[0])),
		(*C.uint8_t)(unsafe.Pointer(&out[0])),
	)
	return out, errFromCode(rc)
}

// ECDHRaw returns the raw x-coordinate of the shared point.
func (c *Context) ECDHRaw(privkey [32]byte, pubkey [33]byte) ([32]byte, error) {
	var out [32]byte
	rc := C.ufsecp_ecdh_raw(
		c.ctx,
		(*C.uint8_t)(unsafe.Pointer(&privkey[0])),
		(*C.uint8_t)(unsafe.Pointer(&pubkey[0])),
		(*C.uint8_t)(unsafe.Pointer(&out[0])),
	)
	return out, errFromCode(rc)
}

// ── Hashing (context-free) ───────────────────────────────────────────────────

// SHA256 computes a SHA-256 digest (hardware-accelerated when available).
func SHA256(data []byte) ([32]byte, error) {
	var out [32]byte
	var placeholder C.uint8_t
	var p *C.uint8_t = &placeholder
	if len(data) > 0 {
		p = (*C.uint8_t)(unsafe.Pointer(&data[0]))
	}
	rc := C.ufsecp_sha256(p, C.size_t(len(data)), (*C.uint8_t)(unsafe.Pointer(&out[0])))
	return out, errFromCode(rc)
}

// Hash160 computes RIPEMD160(SHA256(data)).
func Hash160(data []byte) ([20]byte, error) {
	var out [20]byte
	var placeholder C.uint8_t
	var p *C.uint8_t = &placeholder
	if len(data) > 0 {
		p = (*C.uint8_t)(unsafe.Pointer(&data[0]))
	}
	rc := C.ufsecp_hash160(p, C.size_t(len(data)), (*C.uint8_t)(unsafe.Pointer(&out[0])))
	return out, errFromCode(rc)
}

// TaggedHash computes BIP-340 tagged hash: SHA256(SHA256(tag)||SHA256(tag)||data).
func TaggedHash(tag string, data []byte) ([32]byte, error) {
	var out [32]byte
	cTag := C.CString(tag)
	defer C.free(unsafe.Pointer(cTag))
	var placeholder C.uint8_t
	var p *C.uint8_t = &placeholder
	if len(data) > 0 {
		p = (*C.uint8_t)(unsafe.Pointer(&data[0]))
	}
	rc := C.ufsecp_tagged_hash(cTag, p, C.size_t(len(data)), (*C.uint8_t)(unsafe.Pointer(&out[0])))
	return out, errFromCode(rc)
}

// ── Addresses ────────────────────────────────────────────────────────────────

func (c *Context) getAddress(fn func(buf *C.char, bufLen *C.size_t) C.int) (string, error) {
	var buf [128]C.char
	bufLen := C.size_t(128)
	rc := fn(&buf[0], &bufLen)
	if err := errFromCode(rc); err != nil {
		return "", err
	}
	return C.GoStringN(&buf[0], C.int(bufLen)), nil
}

// AddrP2PKH generates a P2PKH address from a compressed public key.
func (c *Context) AddrP2PKH(pubkey [33]byte, net Network) (string, error) {
	return c.getAddress(func(buf *C.char, bufLen *C.size_t) C.int {
		return C.ufsecp_addr_p2pkh(
			c.ctx,
			(*C.uint8_t)(unsafe.Pointer(&pubkey[0])),
			C.int(net), buf, bufLen,
		)
	})
}

// AddrP2WPKH generates a P2WPKH (SegWit v0, Bech32) address.
func (c *Context) AddrP2WPKH(pubkey [33]byte, net Network) (string, error) {
	return c.getAddress(func(buf *C.char, bufLen *C.size_t) C.int {
		return C.ufsecp_addr_p2wpkh(
			c.ctx,
			(*C.uint8_t)(unsafe.Pointer(&pubkey[0])),
			C.int(net), buf, bufLen,
		)
	})
}

// AddrP2TR generates a P2TR (Taproot, Bech32m) address from an x-only key.
func (c *Context) AddrP2TR(internalKeyX [32]byte, net Network) (string, error) {
	return c.getAddress(func(buf *C.char, bufLen *C.size_t) C.int {
		return C.ufsecp_addr_p2tr(
			c.ctx,
			(*C.uint8_t)(unsafe.Pointer(&internalKeyX[0])),
			C.int(net), buf, bufLen,
		)
	})
}

// ── WIF ──────────────────────────────────────────────────────────────────────

// WIFEncode encodes a private key as WIF.
func (c *Context) WIFEncode(privkey [32]byte, compressed bool, net Network) (string, error) {
	comp := C.int(0)
	if compressed {
		comp = 1
	}
	return c.getAddress(func(buf *C.char, bufLen *C.size_t) C.int {
		return C.ufsecp_wif_encode(
			c.ctx,
			(*C.uint8_t)(unsafe.Pointer(&privkey[0])),
			comp, C.int(net), buf, bufLen,
		)
	})
}

// WIFDecodeResult holds the decoded WIF data.
type WIFDecodeResult struct {
	Privkey    [32]byte
	Compressed bool
	Network    Network
}

// WIFDecode decodes a WIF string.
func (c *Context) WIFDecode(wif string) (WIFDecodeResult, error) {
	cWIF := C.CString(wif)
	defer C.free(unsafe.Pointer(cWIF))
	var result WIFDecodeResult
	var comp, net C.int
	rc := C.ufsecp_wif_decode(
		c.ctx,
		cWIF,
		(*C.uint8_t)(unsafe.Pointer(&result.Privkey[0])),
		&comp, &net,
	)
	if err := errFromCode(rc); err != nil {
		return result, err
	}
	result.Compressed = comp == 1
	result.Network = Network(net)
	return result, nil
}

// ── BIP-32 ───────────────────────────────────────────────────────────────────

// BIP32Key wraps the ufsecp_bip32_key struct (82 bytes).
type BIP32Key [BIP32KeySize]byte

// BIP32Master creates a master key from seed (16–64 bytes).
func (c *Context) BIP32Master(seed []byte) (BIP32Key, error) {
	if len(seed) < 16 || len(seed) > 64 {
		return BIP32Key{}, fmt.Errorf("ufsecp: seed must be 16-64 bytes, got %d", len(seed))
	}
	var key BIP32Key
	rc := C.ufsecp_bip32_master(
		c.ctx,
		(*C.uint8_t)(unsafe.Pointer(&seed[0])),
		C.size_t(len(seed)),
		(*C.ufsecp_bip32_key)(unsafe.Pointer(&key[0])),
	)
	return key, errFromCode(rc)
}

// BIP32Derive derives a child key by index (>= 0x80000000 = hardened).
func (c *Context) BIP32Derive(parent BIP32Key, index uint32) (BIP32Key, error) {
	var child BIP32Key
	rc := C.ufsecp_bip32_derive(
		c.ctx,
		(*C.ufsecp_bip32_key)(unsafe.Pointer(&parent[0])),
		C.uint32_t(index),
		(*C.ufsecp_bip32_key)(unsafe.Pointer(&child[0])),
	)
	return child, errFromCode(rc)
}

// BIP32DerivePath derives from a path string, e.g. "m/44'/0'/0'/0/0".
func (c *Context) BIP32DerivePath(master BIP32Key, path string) (BIP32Key, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	var key BIP32Key
	rc := C.ufsecp_bip32_derive_path(
		c.ctx,
		(*C.ufsecp_bip32_key)(unsafe.Pointer(&master[0])),
		cPath,
		(*C.ufsecp_bip32_key)(unsafe.Pointer(&key[0])),
	)
	return key, errFromCode(rc)
}

// BIP32Privkey extracts the 32-byte private key from an extended key.
func (c *Context) BIP32Privkey(key BIP32Key) ([32]byte, error) {
	var privkey [32]byte
	rc := C.ufsecp_bip32_privkey(
		c.ctx,
		(*C.ufsecp_bip32_key)(unsafe.Pointer(&key[0])),
		(*C.uint8_t)(unsafe.Pointer(&privkey[0])),
	)
	return privkey, errFromCode(rc)
}

// BIP32Pubkey extracts the compressed 33-byte public key from an extended key.
func (c *Context) BIP32Pubkey(key BIP32Key) ([33]byte, error) {
	var pubkey [33]byte
	rc := C.ufsecp_bip32_pubkey(
		c.ctx,
		(*C.ufsecp_bip32_key)(unsafe.Pointer(&key[0])),
		(*C.uint8_t)(unsafe.Pointer(&pubkey[0])),
	)
	return pubkey, errFromCode(rc)
}

// ── Taproot ──────────────────────────────────────────────────────────────────

// TaprootOutputKeyResult holds the output key and parity.
type TaprootOutputKeyResult struct {
	OutputKeyX [32]byte
	Parity     int
}

// TaprootOutputKey derives the Taproot output key from an internal key.
// merkleRoot can be nil for key-path only.
func (c *Context) TaprootOutputKey(internalKeyX [32]byte, merkleRoot *[32]byte) (TaprootOutputKeyResult, error) {
	var mr *C.uint8_t
	if merkleRoot != nil {
		mr = (*C.uint8_t)(unsafe.Pointer(&merkleRoot[0]))
	}
	var result TaprootOutputKeyResult
	var cParity C.int
	rc := C.ufsecp_taproot_output_key(
		c.ctx,
		(*C.uint8_t)(unsafe.Pointer(&internalKeyX[0])),
		mr,
		(*C.uint8_t)(unsafe.Pointer(&result.OutputKeyX[0])),
		&cParity,
	)
	result.Parity = int(cParity)
	return result, errFromCode(rc)
}

// TaprootTweakSeckey tweaks a private key for Taproot key-path spending.
func (c *Context) TaprootTweakSeckey(privkey [32]byte, merkleRoot *[32]byte) ([32]byte, error) {
	var mr *C.uint8_t
	if merkleRoot != nil {
		mr = (*C.uint8_t)(unsafe.Pointer(&merkleRoot[0]))
	}
	var out [32]byte
	rc := C.ufsecp_taproot_tweak_seckey(
		c.ctx,
		(*C.uint8_t)(unsafe.Pointer(&privkey[0])),
		mr,
		(*C.uint8_t)(unsafe.Pointer(&out[0])),
	)
	return out, errFromCode(rc)
}

// TaprootVerify verifies a Taproot commitment.
func (c *Context) TaprootVerify(outputKeyX [32]byte, parity int, internalKeyX [32]byte, merkleRoot []byte) error {
	var mr *C.uint8_t
	mrLen := C.size_t(len(merkleRoot))
	if len(merkleRoot) > 0 {
		mr = (*C.uint8_t)(unsafe.Pointer(&merkleRoot[0]))
	}
	rc := C.ufsecp_taproot_verify(
		c.ctx,
		(*C.uint8_t)(unsafe.Pointer(&outputKeyX[0])),
		C.int(parity),
		(*C.uint8_t)(unsafe.Pointer(&internalKeyX[0])),
		mr, mrLen,
	)
	return errFromCode(rc)
}
