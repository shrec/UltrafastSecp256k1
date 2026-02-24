// ============================================================================
// UltrafastSecp256k1 — Go Binding Smoke Test (Golden Vectors)
// ============================================================================
// Verifies CGo/FFI boundary correctness using deterministic known-answer tests.
// Runs in <2 seconds.
//
// Usage:
//   go test -v -run TestSmoke ./...
// ============================================================================

package ufsecp_test

import (
	"bytes"
	"encoding/hex"
	"strings"
	"testing"

	"github.com/nicenemo/ufsecp"
)

// ── Golden Vectors ──────────────────────────────────────────────────────

func mustHex(s string) []byte {
	b, err := hex.DecodeString(s)
	if err != nil {
		panic(err)
	}
	return b
}

func mustHex32(s string) [32]byte {
	b := mustHex(s)
	var out [32]byte
	copy(out[:], b)
	return out
}

func mustHex33(s string) [33]byte {
	b := mustHex(s)
	var out [33]byte
	copy(out[:], b)
	return out
}

var (
	knownPrivkey = mustHex32("0000000000000000000000000000000000000000000000000000000000000001")
	knownPubkey  = mustHex33("0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798")
	knownXonly   = mustHex32("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798")
	sha256Empty  = mustHex32("E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855")
	msg32        = [32]byte{}
	aux32        = [32]byte{}
)

// ── Tests ───────────────────────────────────────────────────────────────

func TestSmokeCtxCreateAbi(t *testing.T) {
	ctx, err := ufsecp.NewContext()
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Destroy()

	abi := ctx.AbiVersion()
	if abi < 1 {
		t.Fatalf("ABI %d < 1", abi)
	}
}

func TestSmokePubkeyCreateGolden(t *testing.T) {
	ctx, err := ufsecp.NewContext()
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Destroy()

	pub, err := ctx.PubkeyCreate(knownPrivkey)
	if err != nil {
		t.Fatalf("PubkeyCreate: %v", err)
	}
	if pub != knownPubkey {
		t.Fatalf("pubkey mismatch: got %x, want %x", pub, knownPubkey)
	}
}

func TestSmokePubkeyXonlyGolden(t *testing.T) {
	ctx, err := ufsecp.NewContext()
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Destroy()

	xonly, err := ctx.PubkeyXonly(knownPrivkey)
	if err != nil {
		t.Fatalf("PubkeyXonly: %v", err)
	}
	if xonly != knownXonly {
		t.Fatalf("xonly mismatch: got %x, want %x", xonly, knownXonly)
	}
}

func TestSmokeEcdsaSignVerify(t *testing.T) {
	ctx, err := ufsecp.NewContext()
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Destroy()

	sig, err := ctx.EcdsaSign(msg32, knownPrivkey)
	if err != nil {
		t.Fatalf("EcdsaSign: %v", err)
	}

	if err := ctx.EcdsaVerify(msg32, sig, knownPubkey); err != nil {
		t.Fatalf("EcdsaVerify: %v", err)
	}

	// Mutated → fail
	bad := sig
	bad[0] ^= 0x01
	if err := ctx.EcdsaVerify(msg32, bad, knownPubkey); err == nil {
		t.Fatal("mutated sig should fail")
	}
}

func TestSmokeSchnorrSignVerify(t *testing.T) {
	ctx, err := ufsecp.NewContext()
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Destroy()

	sig, err := ctx.SchnorrSign(msg32, knownPrivkey, aux32)
	if err != nil {
		t.Fatalf("SchnorrSign: %v", err)
	}

	if err := ctx.SchnorrVerify(msg32, sig, knownXonly); err != nil {
		t.Fatalf("SchnorrVerify: %v", err)
	}
}

func TestSmokeEcdsaRecover(t *testing.T) {
	ctx, err := ufsecp.NewContext()
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Destroy()

	sig, recid, err := ctx.EcdsaSignRecoverable(msg32, knownPrivkey)
	if err != nil {
		t.Fatalf("EcdsaSignRecoverable: %v", err)
	}
	if recid < 0 || recid > 3 {
		t.Fatalf("recid %d out of range", recid)
	}

	pub, err := ctx.EcdsaRecover(msg32, sig, recid)
	if err != nil {
		t.Fatalf("EcdsaRecover: %v", err)
	}
	if pub != knownPubkey {
		t.Fatalf("recovered pubkey mismatch: got %x, want %x", pub, knownPubkey)
	}
}

func TestSmokeSha256Golden(t *testing.T) {
	digest, err := ufsecp.SHA256([]byte{})
	if err != nil {
		t.Fatalf("SHA256: %v", err)
	}
	if digest != sha256Empty {
		t.Fatalf("SHA-256 empty mismatch: got %x, want %x", digest, sha256Empty)
	}
}

func TestSmokeAddrP2wpkh(t *testing.T) {
	ctx, err := ufsecp.NewContext()
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Destroy()

	addr, err := ctx.AddrP2WPKH(knownPubkey, ufsecp.NetworkMainnet)
	if err != nil {
		t.Fatalf("AddrP2WPKH: %v", err)
	}
	if !strings.HasPrefix(addr, "bc1q") {
		t.Fatalf("expected bc1q..., got %s", addr)
	}
}

func TestSmokeWifRoundtrip(t *testing.T) {
	ctx, err := ufsecp.NewContext()
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Destroy()

	wif, err := ctx.WIFEncode(knownPrivkey, true, ufsecp.NetworkMainnet)
	if err != nil {
		t.Fatalf("WIFEncode: %v", err)
	}

	result, err := ctx.WIFDecode(wif)
	if err != nil {
		t.Fatalf("WIFDecode: %v", err)
	}
	if !bytes.Equal(result.Privkey[:], knownPrivkey[:]) {
		t.Fatalf("WIF privkey mismatch")
	}
	if !result.Compressed {
		t.Fatal("WIF should be compressed")
	}
}

func TestSmokeEcdhSymmetric(t *testing.T) {
	ctx, err := ufsecp.NewContext()
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Destroy()

	k2 := mustHex32("0000000000000000000000000000000000000000000000000000000000000002")
	pub1, err := ctx.PubkeyCreate(knownPrivkey)
	if err != nil {
		t.Fatalf("PubkeyCreate(k1): %v", err)
	}
	pub2, err := ctx.PubkeyCreate(k2)
	if err != nil {
		t.Fatalf("PubkeyCreate(k2): %v", err)
	}

	s12, err := ctx.ECDH(knownPrivkey, pub2)
	if err != nil {
		t.Fatalf("ECDH(k1,pub2): %v", err)
	}
	s21, err := ctx.ECDH(k2, pub1)
	if err != nil {
		t.Fatalf("ECDH(k2,pub1): %v", err)
	}
	if s12 != s21 {
		t.Fatalf("ECDH asymmetric: %x != %x", s12, s21)
	}
}

func TestSmokeErrorPath(t *testing.T) {
	ctx, err := ufsecp.NewContext()
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Destroy()

	zeroes := [32]byte{}
	_, err = ctx.PubkeyCreate(zeroes)
	if err == nil {
		t.Fatal("zero key should fail")
	}
}

func TestSmokeEcdsaDeterministic(t *testing.T) {
	ctx, err := ufsecp.NewContext()
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Destroy()

	sig1, err := ctx.EcdsaSign(msg32, knownPrivkey)
	if err != nil {
		t.Fatalf("EcdsaSign(1): %v", err)
	}
	sig2, err := ctx.EcdsaSign(msg32, knownPrivkey)
	if err != nil {
		t.Fatalf("EcdsaSign(2): %v", err)
	}
	if sig1 != sig2 {
		t.Fatalf("RFC 6979 non-deterministic")
	}
}
