// UltrafastSecp256k1 -- Go Example (CPU + GPU)
//
// Demonstrates pure cgo calls to the ufsecp C ABI: key ops, ECDSA, Schnorr,
// ECDH, hashing, Bitcoin addresses, BIP-32, Taproot, and GPU batch operations.
//
// Build & Run:
//
//	CGO_CFLAGS="-I../../include/ufsecp" \
//	CGO_LDFLAGS="-L../../build-linux/include/ufsecp -lufsecp" \
//	LD_LIBRARY_PATH=../../build-linux/include/ufsecp \
//	go run example.go
package main

/*
#cgo LDFLAGS: -lufsecp
#cgo CFLAGS: -I../../include/ufsecp

#include "ufsecp.h"
#include "ufsecp_gpu.h"
#include <stdlib.h>
#include <string.h>
*/
import "C"
import (
	"encoding/hex"
	"fmt"
	"os"
	"unsafe"
)

func hexs(data []byte) string { return hex.EncodeToString(data) }

func check(rc C.int, op string) {
	if rc != 0 {
		fmt.Fprintf(os.Stderr, "[FAIL] %s: %s (code %d)\n", op, C.GoString(C.ufsecp_error_str(rc)), rc)
		os.Exit(1)
	}
}

func boolStr(cond bool, t, f string) string {
	if cond {
		return t
	}
	return f
}

func demoCPU(ctx *C.ufsecp_ctx) {
	fmt.Println("=== CPU Operations ===")
	fmt.Println()

	var privkey, privkey2 [32]byte
	privkey[31] = 1
	privkey2[31] = 2
	sk := (*C.uint8_t)(unsafe.Pointer(&privkey[0]))
	sk2 := (*C.uint8_t)(unsafe.Pointer(&privkey2[0]))

	// 1. Key Generation
	fmt.Println("[1] Key Generation")
	var pub33 [33]byte
	var pub65 [65]byte
	var xonly [32]byte
	check(C.ufsecp_pubkey_create(ctx, sk, (*C.uint8_t)(&pub33[0])), "pubkey_create")
	check(C.ufsecp_pubkey_create_uncompressed(ctx, sk, (*C.uint8_t)(&pub65[0])), "pubkey_uncompressed")
	check(C.ufsecp_pubkey_xonly(ctx, sk, (*C.uint8_t)(&xonly[0])), "pubkey_xonly")
	fmt.Printf("  Private key:        %s\n", hexs(privkey[:]))
	fmt.Printf("  Compressed (33B):   %s\n", hexs(pub33[:]))
	fmt.Printf("  Uncompressed (65B): %s\n", hexs(pub65[:]))
	fmt.Printf("  X-only (32B):       %s\n", hexs(xonly[:]))
	fmt.Println()

	// 2. ECDSA
	fmt.Println("[2] ECDSA Sign / Verify (RFC 6979)")
	msgStr := []byte("Hello UltrafastSecp256k1!")
	var msg [32]byte
	check(C.ufsecp_sha256((*C.uint8_t)(&msgStr[0]), C.size_t(len(msgStr)), (*C.uint8_t)(&msg[0])), "sha256")
	fmt.Printf("  Message hash:       %s\n", hexs(msg[:]))

	var sig [64]byte
	check(C.ufsecp_ecdsa_sign(ctx, (*C.uint8_t)(&msg[0]), sk, (*C.uint8_t)(&sig[0])), "ecdsa_sign")
	fmt.Printf("  ECDSA signature:    %s\n", hexs(sig[:]))

	rc := C.ufsecp_ecdsa_verify(ctx, (*C.uint8_t)(&msg[0]), (*C.uint8_t)(&sig[0]), (*C.uint8_t)(&pub33[0]))
	fmt.Printf("  Verify:             %s\n", boolStr(rc == 0, "VALID", "INVALID"))

	// DER
	var der [72]byte
	derLen := C.size_t(72)
	check(C.ufsecp_ecdsa_sig_to_der(ctx, (*C.uint8_t)(&sig[0]), (*C.uint8_t)(&der[0]), &derLen), "sig_to_der")
	fmt.Printf("  DER length:         %d bytes\n", derLen)

	var sigBack [64]byte
	check(C.ufsecp_ecdsa_sig_from_der(ctx, (*C.uint8_t)(&der[0]), derLen, (*C.uint8_t)(&sigBack[0])), "sig_from_der")
	fmt.Printf("  DER roundtrip:      %s\n", boolStr(sig == sigBack, "match", "MISMATCH"))

	// Recovery
	var rsig [64]byte
	var recid C.int
	check(C.ufsecp_ecdsa_sign_recoverable(ctx, (*C.uint8_t)(&msg[0]), sk, (*C.uint8_t)(&rsig[0]), &recid), "sign_recoverable")
	var recovered [33]byte
	check(C.ufsecp_ecdsa_recover(ctx, (*C.uint8_t)(&msg[0]), (*C.uint8_t)(&rsig[0]), recid, (*C.uint8_t)(&recovered[0])), "ecdsa_recover")
	fmt.Printf("  Recovery:           recid=%d, match=%s\n", recid, boolStr(recovered == pub33, "YES", "NO"))
	fmt.Println()

	// 3. Schnorr
	fmt.Println("[3] Schnorr Sign / Verify (BIP-340)")
	var aux [32]byte
	var schnorrSig [64]byte
	check(C.ufsecp_schnorr_sign(ctx, (*C.uint8_t)(&msg[0]), sk, (*C.uint8_t)(&aux[0]), (*C.uint8_t)(&schnorrSig[0])), "schnorr_sign")
	fmt.Printf("  Schnorr signature:  %s\n", hexs(schnorrSig[:]))
	rc = C.ufsecp_schnorr_verify(ctx, (*C.uint8_t)(&msg[0]), (*C.uint8_t)(&schnorrSig[0]), (*C.uint8_t)(&xonly[0]))
	fmt.Printf("  Verify:             %s\n", boolStr(rc == 0, "VALID", "INVALID"))
	fmt.Println()

	// 4. ECDH
	fmt.Println("[4] ECDH Key Agreement")
	var pub2 [33]byte
	check(C.ufsecp_pubkey_create(ctx, sk2, (*C.uint8_t)(&pub2[0])), "pubkey2")
	var secretA, secretB [32]byte
	check(C.ufsecp_ecdh(ctx, sk, (*C.uint8_t)(&pub2[0]), (*C.uint8_t)(&secretA[0])), "ecdh_a")
	check(C.ufsecp_ecdh(ctx, sk2, (*C.uint8_t)(&pub33[0]), (*C.uint8_t)(&secretB[0])), "ecdh_b")
	fmt.Printf("  Secret (A->B):      %s\n", hexs(secretA[:]))
	fmt.Printf("  Secret (B->A):      %s\n", hexs(secretB[:]))
	fmt.Printf("  Match:              %s\n", boolStr(secretA == secretB, "YES", "NO"))
	fmt.Println()

	// 5. Hashing
	fmt.Println("[5] Hashing")
	var sha [32]byte
	var h160 [20]byte
	check(C.ufsecp_sha256((*C.uint8_t)(&pub33[0]), 33, (*C.uint8_t)(&sha[0])), "sha256_pub")
	check(C.ufsecp_hash160((*C.uint8_t)(&pub33[0]), 33, (*C.uint8_t)(&h160[0])), "hash160_pub")
	fmt.Printf("  SHA-256(pubkey):    %s\n", hexs(sha[:]))
	fmt.Printf("  Hash160(pubkey):    %s\n", hexs(h160[:]))
	fmt.Println()

	// 6. Bitcoin Addresses
	fmt.Println("[6] Bitcoin Addresses")
	var addrBuf [128]byte
	var addrLen C.size_t

	addrLen = 128
	check(C.ufsecp_addr_p2pkh(ctx, (*C.uint8_t)(&pub33[0]), 0, (*C.char)(unsafe.Pointer(&addrBuf[0])), &addrLen), "p2pkh")
	fmt.Printf("  P2PKH:              %s\n", string(addrBuf[:addrLen]))

	addrLen = 128
	check(C.ufsecp_addr_p2wpkh(ctx, (*C.uint8_t)(&pub33[0]), 0, (*C.char)(unsafe.Pointer(&addrBuf[0])), &addrLen), "p2wpkh")
	fmt.Printf("  P2WPKH:             %s\n", string(addrBuf[:addrLen]))

	addrLen = 128
	check(C.ufsecp_addr_p2tr(ctx, (*C.uint8_t)(&xonly[0]), 0, (*C.char)(unsafe.Pointer(&addrBuf[0])), &addrLen), "p2tr")
	fmt.Printf("  P2TR:               %s\n", string(addrBuf[:addrLen]))
	fmt.Println()

	// 7. WIF
	fmt.Println("[7] WIF Encoding")
	wifBuf := make([]byte, 128)
	wifLen := C.size_t(128)
	check(C.ufsecp_wif_encode(ctx, sk, 1, 0, (*C.char)(unsafe.Pointer(&wifBuf[0])), &wifLen), "wif_encode")
	wif := string(wifBuf[:wifLen])
	fmt.Printf("  WIF:                %s\n", wif)
	var decoded [32]byte
	var comp, net C.int
	cWif := C.CString(wif)
	check(C.ufsecp_wif_decode(ctx, cWif, (*C.uint8_t)(&decoded[0]), &comp, &net), "wif_decode")
	C.free(unsafe.Pointer(cWif))
	fmt.Printf("  Decode roundtrip:   match=%s\n", boolStr(decoded == privkey, "YES", "NO"))
	fmt.Println()

	// 8. BIP-32
	fmt.Println("[8] BIP-32 HD Key Derivation")
	var seed [64]byte
	for i := range seed {
		seed[i] = 0x42
	}
	var master C.ufsecp_bip32_key
	check(C.ufsecp_bip32_master(ctx, (*C.uint8_t)(&seed[0]), 64, &master), "bip32_master")
	var childKey C.ufsecp_bip32_key
	path := C.CString("m/44'/0'/0'/0/0")
	check(C.ufsecp_bip32_derive_path(ctx, &master, path, &childKey), "bip32_derive")
	C.free(unsafe.Pointer(path))
	var childPriv [32]byte
	var childPub [33]byte
	check(C.ufsecp_bip32_privkey(ctx, &childKey, (*C.uint8_t)(&childPriv[0])), "bip32_privkey")
	check(C.ufsecp_bip32_pubkey(ctx, &childKey, (*C.uint8_t)(&childPub[0])), "bip32_pubkey")
	fmt.Printf("  BIP-32 child priv:  %s\n", hexs(childPriv[:]))
	fmt.Printf("  BIP-32 child pub:   %s\n", hexs(childPub[:]))
	fmt.Println()

	// 9. Taproot
	fmt.Println("[9] Taproot (BIP-341)")
	var outputX [32]byte
	var parity C.int
	check(C.ufsecp_taproot_output_key(ctx, (*C.uint8_t)(&xonly[0]), nil, (*C.uint8_t)(&outputX[0]), &parity), "taproot_output_key")
	fmt.Printf("  Output key:         %s\n", hexs(outputX[:]))
	fmt.Printf("  Parity:             %d\n", parity)
	rc = C.ufsecp_taproot_verify(ctx, (*C.uint8_t)(&outputX[0]), parity, (*C.uint8_t)(&xonly[0]), nil, 0)
	fmt.Printf("  Verify:             %s\n", boolStr(rc == 0, "VALID", "INVALID"))
	fmt.Println()

	// 10. Pedersen Commitment
	fmt.Println("[10] Pedersen Commitment")
	var pedValue [32]byte
	pedValue[31] = 42
	var pedBlinding [32]byte
	pedBlinding[31] = 7
	var pedCommit [33]byte
	check(C.ufsecp_pedersen_commit(ctx, (*C.uint8_t)(&pedValue[0]), (*C.uint8_t)(&pedBlinding[0]), (*C.uint8_t)(&pedCommit[0])), "pedersen_commit")
	fmt.Printf("  Commitment:         %s\n", hexs(pedCommit[:]))
	rc = C.ufsecp_pedersen_verify(ctx, (*C.uint8_t)(&pedCommit[0]), (*C.uint8_t)(&pedValue[0]), (*C.uint8_t)(&pedBlinding[0]))
	fmt.Printf("  Verify:             %s\n", boolStr(rc == 0, "VALID", "INVALID"))
	fmt.Println()
}

func demoGPU(cpuCtx *C.ufsecp_ctx) {
	fmt.Println("=== GPU Operations ===")
	fmt.Println()

	// 10. Backend Discovery
	fmt.Println("[10] GPU Backend Discovery")
	var bids [4]C.uint32_t
	nBackends := C.ufsecp_gpu_backend_count(&bids[0], 4)
	fmt.Printf("  Backends compiled:  %d\n", nBackends)

	var useBackend C.uint32_t
	for i := C.uint32_t(0); i < nBackends; i++ {
		bid := bids[i]
		name := C.GoString(C.ufsecp_gpu_backend_name(bid))
		avail := C.ufsecp_gpu_is_available(bid)
		devs := C.ufsecp_gpu_device_count(bid)
		fmt.Printf("  Backend %d: %-8s available=%d devices=%d\n", bid, name, avail, devs)
		if avail != 0 && useBackend == 0 {
			useBackend = bid
		}
	}

	if useBackend == 0 {
		fmt.Println("  No GPU backends available -- skipping GPU demos.")
		fmt.Println()
		return
	}

	// Create GPU context
	var gpu *C.ufsecp_gpu_ctx
	rc := C.ufsecp_gpu_ctx_create(&gpu, useBackend, 0)
	if rc != 0 {
		fmt.Printf("  GPU context creation failed: %s\n", C.GoString(C.ufsecp_gpu_error_str(rc)))
		return
	}
	defer C.ufsecp_gpu_ctx_destroy(gpu)

	const N = 4

	// 11. Batch Key Generation
	fmt.Println()
	fmt.Println("[11] GPU Batch Key Generation (4 keys)")
	var scalars [N * 32]byte
	for i := 0; i < N; i++ {
		scalars[i*32+31] = byte(i + 1)
	}

	var pubkeys [N * 33]byte
	rc = C.ufsecp_gpu_generator_mul_batch(
		gpu,
		(*C.uint8_t)(unsafe.Pointer(&scalars[0])),
		C.size_t(N),
		(*C.uint8_t)(unsafe.Pointer(&pubkeys[0])),
	)
	if rc == 0 {
		for i := 0; i < N; i++ {
			fmt.Printf("  GPU pubkey[%d]:      %s\n", i, hexs(pubkeys[i*33:(i+1)*33]))
		}
	} else {
		fmt.Printf("  gpu_generator_mul_batch: %s\n", C.GoString(C.ufsecp_gpu_error_str(rc)))
	}

	// 12. ECDSA Batch Verify
	fmt.Println()
	fmt.Println("[12] GPU ECDSA Batch Verify")
	var msgs [N * 32]byte
	var sigs [N * 64]byte
	var pubs [N * 33]byte

	for i := 0; i < N; i++ {
		var msgHash [32]byte
		b := C.uint8_t(i)
		C.ufsecp_sha256(&b, 1, (*C.uint8_t)(&msgHash[0]))
		copy(msgs[i*32:(i+1)*32], msgHash[:])

		var sk [32]byte
		sk[31] = byte(i + 1)
		var sig [64]byte
		C.ufsecp_ecdsa_sign(cpuCtx, (*C.uint8_t)(&msgHash[0]), (*C.uint8_t)(&sk[0]), (*C.uint8_t)(&sig[0]))
		copy(sigs[i*64:(i+1)*64], sig[:])
		copy(pubs[i*33:(i+1)*33], pubkeys[i*33:(i+1)*33])
	}

	var results [N]byte
	rc = C.ufsecp_gpu_ecdsa_verify_batch(
		gpu,
		(*C.uint8_t)(unsafe.Pointer(&msgs[0])),
		(*C.uint8_t)(unsafe.Pointer(&pubs[0])),
		(*C.uint8_t)(unsafe.Pointer(&sigs[0])),
		C.size_t(N),
		(*C.uint8_t)(unsafe.Pointer(&results[0])),
	)
	if rc == 0 {
		fmt.Printf("  Results: ")
		for i := 0; i < N; i++ {
			fmt.Printf("[%d]=%s ", i, boolStr(results[i] != 0, "VALID", "INVALID"))
		}
		fmt.Println()
	} else {
		fmt.Printf("  gpu_ecdsa_verify_batch: %s\n", C.GoString(C.ufsecp_gpu_error_str(rc)))
	}

	// 13. Hash160 Batch
	fmt.Println()
	fmt.Println("[13] GPU Hash160 Batch")
	var hashes [N * 20]byte
	rc = C.ufsecp_gpu_hash160_pubkey_batch(
		gpu,
		(*C.uint8_t)(unsafe.Pointer(&pubkeys[0])),
		C.size_t(N),
		(*C.uint8_t)(unsafe.Pointer(&hashes[0])),
	)
	if rc == 0 {
		for i := 0; i < N; i++ {
			fmt.Printf("  Hash160[%d]:         %s\n", i, hexs(hashes[i*20:(i+1)*20]))
		}
	} else {
		fmt.Printf("  gpu_hash160_pubkey_batch: %s\n", C.GoString(C.ufsecp_gpu_error_str(rc)))
	}

	// 14. MSM
	fmt.Println()
	fmt.Println("[14] GPU Multi-Scalar Multiplication")
	var msmResult [33]byte
	rc = C.ufsecp_gpu_msm(
		gpu,
		(*C.uint8_t)(unsafe.Pointer(&scalars[0])),
		(*C.uint8_t)(unsafe.Pointer(&pubkeys[0])),
		C.size_t(N),
		(*C.uint8_t)(unsafe.Pointer(&msmResult[0])),
	)
	if rc == 0 {
		fmt.Printf("  MSM result:         %s\n", hexs(msmResult[:]))
	} else {
		fmt.Printf("  gpu_msm: %s\n", C.GoString(C.ufsecp_gpu_error_str(rc)))
	}
	fmt.Println()
}

func main() {
	fmt.Println("UltrafastSecp256k1 -- Go Example")
	fmt.Printf("ABI version: %d\n", C.ufsecp_abi_version())
	fmt.Printf("Library:     %s\n", C.GoString(C.ufsecp_version_string()))
	fmt.Println()

	var ctx *C.ufsecp_ctx
	check(C.ufsecp_ctx_create(&ctx), "ctx_create")
	defer C.ufsecp_ctx_destroy(ctx)

	demoCPU(ctx)
	demoGPU(ctx)

	fmt.Println("All examples completed successfully.")
}
