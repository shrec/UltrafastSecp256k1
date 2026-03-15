/**
 * UltrafastSecp256k1 -- Node.js Example (CPU + GPU)
 *
 * Demonstrates the ufsecp C ABI via koffi FFI: key ops, ECDSA, Schnorr,
 * ECDH, hashing, Bitcoin addresses, WIF, BIP-32, Taproot, Pedersen,
 * and GPU batch operations.
 *
 * Requirements:
 *   npm install koffi
 *
 * Run:
 *   UFSECP_LIB=../../build-linux/include/ufsecp/libufsecp.so node example.js
 */

'use strict';

const koffi = require('koffi');
const path = require('path');

// ── Load Library ─────────────────────────────────────────────────────────

const LIB_PATH = process.env.UFSECP_LIB ||
    path.join(__dirname, '..', '..', 'build-linux', 'include', 'ufsecp', 'libufsecp.so');

const lib = koffi.load(LIB_PATH);

// ── CPU Function Declarations ────────────────────────────────────────────

const UFSECP_OK = 0;

// Context
const ctx_create      = lib.func('int ufsecp_ctx_create(_Out_ void **ctx)');
const ctx_destroy     = lib.func('void ufsecp_ctx_destroy(void *ctx)');
const abi_version     = lib.func('uint32_t ufsecp_abi_version()');
const version_string  = lib.func('const char *ufsecp_version_string()');
const error_str       = lib.func('const char *ufsecp_error_str(int code)');

// Keys
const seckey_verify   = lib.func('int ufsecp_seckey_verify(void *ctx, const uint8_t *k32)');
const pubkey_create   = lib.func('int ufsecp_pubkey_create(void *ctx, const uint8_t *sk32, uint8_t *pk33)');
const pubkey_xonly    = lib.func('int ufsecp_pubkey_xonly(void *ctx, const uint8_t *sk32, uint8_t *xo32)');

// ECDSA
const ecdsa_sign      = lib.func('int ufsecp_ecdsa_sign(void *ctx, const uint8_t *msg32, const uint8_t *sk32, uint8_t *sig64)');
const ecdsa_verify    = lib.func('int ufsecp_ecdsa_verify(void *ctx, const uint8_t *msg32, const uint8_t *sig64, const uint8_t *pk33)');
const ecdsa_sign_rec  = lib.func('int ufsecp_ecdsa_sign_recoverable(void *ctx, const uint8_t *msg32, const uint8_t *sk32, uint8_t *sig64, _Out_ int *recid)');
const ecdsa_recover   = lib.func('int ufsecp_ecdsa_recover(void *ctx, const uint8_t *msg32, const uint8_t *sig64, int recid, uint8_t *pk33)');
const ecdsa_to_der    = lib.func('int ufsecp_ecdsa_sig_to_der(void *ctx, const uint8_t *sig64, uint8_t *der, _Inout_ size_t *len)');

// Schnorr
const schnorr_sign    = lib.func('int ufsecp_schnorr_sign(void *ctx, const uint8_t *msg32, const uint8_t *sk32, const uint8_t *aux32, uint8_t *sig64)');
const schnorr_verify  = lib.func('int ufsecp_schnorr_verify(void *ctx, const uint8_t *msg32, const uint8_t *sig64, const uint8_t *xo32)');

// ECDH
const ecdh            = lib.func('int ufsecp_ecdh(void *ctx, const uint8_t *sk32, const uint8_t *pk33, uint8_t *secret32)');

// Hashing
const sha256          = lib.func('int ufsecp_sha256(const uint8_t *data, size_t len, uint8_t *digest32)');
const hash160         = lib.func('int ufsecp_hash160(const uint8_t *data, size_t len, uint8_t *digest20)');

// Addresses
const addr_p2pkh      = lib.func('int ufsecp_addr_p2pkh(void *ctx, const uint8_t *pk33, int net, uint8_t *addr, _Inout_ size_t *len)');
const addr_p2wpkh     = lib.func('int ufsecp_addr_p2wpkh(void *ctx, const uint8_t *pk33, int net, uint8_t *addr, _Inout_ size_t *len)');
const addr_p2tr       = lib.func('int ufsecp_addr_p2tr(void *ctx, const uint8_t *xo32, int net, uint8_t *addr, _Inout_ size_t *len)');

// WIF
const wif_encode      = lib.func('int ufsecp_wif_encode(void *ctx, const uint8_t *sk32, int comp, int net, uint8_t *wif, _Inout_ size_t *len)');

// BIP-32
const bip32_master    = lib.func('int ufsecp_bip32_master(void *ctx, const uint8_t *seed, size_t seed_len, uint8_t *key82)');
const bip32_derive    = lib.func('int ufsecp_bip32_derive_path(void *ctx, const uint8_t *master82, const char *path, uint8_t *key82)');
const bip32_privkey   = lib.func('int ufsecp_bip32_privkey(void *ctx, const uint8_t *key82, uint8_t *priv32)');
const bip32_pubkey    = lib.func('int ufsecp_bip32_pubkey(void *ctx, const uint8_t *key82, uint8_t *pub33)');

// Taproot
const taproot_output  = lib.func('int ufsecp_taproot_output_key(void *ctx, const uint8_t *ix32, const uint8_t *mr, uint8_t *ox32, _Out_ int *parity)');
const taproot_verify  = lib.func('int ufsecp_taproot_verify(void *ctx, const uint8_t *ox32, int parity, const uint8_t *ix32, const uint8_t *mr, size_t mr_len)');

// Pedersen
const pedersen_commit = lib.func('int ufsecp_pedersen_commit(void *ctx, const uint8_t *v32, const uint8_t *b32, uint8_t *c33)');
const pedersen_verify = lib.func('int ufsecp_pedersen_verify(void *ctx, const uint8_t *c33, const uint8_t *v32, const uint8_t *b32)');

// GPU
const gpu_backend_count    = lib.func('uint32_t ufsecp_gpu_backend_count(_Out_ uint32_t *ids, uint32_t max)');
const gpu_backend_name     = lib.func('const char *ufsecp_gpu_backend_name(uint32_t bid)');
const gpu_is_available     = lib.func('int ufsecp_gpu_is_available(uint32_t bid)');
const gpu_device_count     = lib.func('uint32_t ufsecp_gpu_device_count(uint32_t bid)');
const gpu_ctx_create       = lib.func('int ufsecp_gpu_ctx_create(_Out_ void **ctx, uint32_t bid, uint32_t dev)');
const gpu_ctx_destroy      = lib.func('void ufsecp_gpu_ctx_destroy(void *ctx)');
const gpu_generator_mul    = lib.func('int ufsecp_gpu_generator_mul_batch(void *ctx, const uint8_t *s32, size_t n, uint8_t *pk33)');
const gpu_ecdsa_verify     = lib.func('int ufsecp_gpu_ecdsa_verify_batch(void *ctx, const uint8_t *msg, const uint8_t *pk, const uint8_t *sig, size_t n, uint8_t *res)');
const gpu_hash160          = lib.func('int ufsecp_gpu_hash160_pubkey_batch(void *ctx, const uint8_t *pk33, size_t n, uint8_t *h20)');
const gpu_msm              = lib.func('int ufsecp_gpu_msm(void *ctx, const uint8_t *s32, const uint8_t *p33, size_t n, uint8_t *out33)');
const gpu_error_str        = lib.func('const char *ufsecp_gpu_error_str(int code)');

// ── Helpers ──────────────────────────────────────────────────────────────

function hex(buf) { return Buffer.from(buf).toString('hex'); }

function check(rc, op) {
    if (rc !== UFSECP_OK) {
        throw new Error(`${op} failed: ${error_str(rc)} (code ${rc})`);
    }
}

function createCtx() {
    const pp = [null];
    check(ctx_create(pp), 'ctx_create');
    return pp[0];
}

function getAddr(fn, ctx, key, net) {
    const buf = Buffer.alloc(128);
    const len = [128];
    check(fn(ctx, key, net, buf, len), 'addr');
    return buf.slice(0, len[0]).toString();
}

// ── Golden Vectors ───────────────────────────────────────────────────────

const PRIVKEY  = Buffer.from('0000000000000000000000000000000000000000000000000000000000000001', 'hex');
const PRIVKEY2 = Buffer.from('0000000000000000000000000000000000000000000000000000000000000002', 'hex');

// ── CPU Demo ─────────────────────────────────────────────────────────────

function demoCPU() {
    console.log('=== CPU Operations ===\n');
    const ctx = createCtx();

    // 1. Key Generation
    console.log('[1] Key Generation');
    const pub33 = Buffer.alloc(33);
    const xonly = Buffer.alloc(32);
    check(pubkey_create(ctx, PRIVKEY, pub33), 'pubkey_create');
    check(pubkey_xonly(ctx, PRIVKEY, xonly), 'pubkey_xonly');
    console.log(`  Private key:        ${hex(PRIVKEY)}`);
    console.log(`  Compressed (33B):   ${hex(pub33)}`);
    console.log(`  X-only (32B):       ${hex(xonly)}`);
    console.log();

    // 2. ECDSA
    console.log('[2] ECDSA Sign / Verify (RFC 6979)');
    const msg = Buffer.alloc(32);
    check(sha256(Buffer.from('Hello UltrafastSecp256k1!'), 24, msg), 'sha256');
    console.log(`  Message hash:       ${hex(msg)}`);

    const sig = Buffer.alloc(64);
    check(ecdsa_sign(ctx, msg, PRIVKEY, sig), 'ecdsa_sign');
    console.log(`  ECDSA signature:    ${hex(sig)}`);

    const vrc = ecdsa_verify(ctx, msg, sig, pub33);
    console.log(`  Verify:             ${vrc === UFSECP_OK ? 'VALID' : 'INVALID'}`);

    // DER encoding
    const der = Buffer.alloc(72);
    const derLen = [72];
    check(ecdsa_to_der(ctx, sig, der, derLen), 'sig_to_der');
    console.log(`  DER length:         ${derLen[0]} bytes`);

    // Recovery
    const rsig = Buffer.alloc(64);
    const recid = [0];
    check(ecdsa_sign_rec(ctx, msg, PRIVKEY, rsig, recid), 'sign_recoverable');
    const recovered = Buffer.alloc(33);
    check(ecdsa_recover(ctx, msg, rsig, recid[0], recovered), 'recover');
    console.log(`  Recovery:           recid=${recid[0]}, match=${recovered.equals(pub33) ? 'YES' : 'NO'}`);
    console.log();

    // 3. Schnorr
    console.log('[3] Schnorr Sign / Verify (BIP-340)');
    const aux = Buffer.alloc(32);
    const schnorrSig = Buffer.alloc(64);
    check(schnorr_sign(ctx, msg, PRIVKEY, aux, schnorrSig), 'schnorr_sign');
    console.log(`  Schnorr signature:  ${hex(schnorrSig)}`);
    const sv = schnorr_verify(ctx, msg, schnorrSig, xonly);
    console.log(`  Verify:             ${sv === UFSECP_OK ? 'VALID' : 'INVALID'}`);
    console.log();

    // 4. ECDH
    console.log('[4] ECDH Key Agreement');
    const pub2 = Buffer.alloc(33);
    check(pubkey_create(ctx, PRIVKEY2, pub2), 'pubkey2');
    const secretA = Buffer.alloc(32);
    const secretB = Buffer.alloc(32);
    check(ecdh(ctx, PRIVKEY, pub2, secretA), 'ecdh_a');
    check(ecdh(ctx, PRIVKEY2, pub33, secretB), 'ecdh_b');
    console.log(`  Secret (A->B):      ${hex(secretA)}`);
    console.log(`  Secret (B->A):      ${hex(secretB)}`);
    console.log(`  Match:              ${secretA.equals(secretB) ? 'YES' : 'NO'}`);
    console.log();

    // 5. Hashing
    console.log('[5] Hashing');
    const sha = Buffer.alloc(32);
    const h160 = Buffer.alloc(20);
    check(sha256(pub33, 33, sha), 'sha256_pub');
    check(hash160(pub33, 33, h160), 'hash160_pub');
    console.log(`  SHA-256(pubkey):    ${hex(sha)}`);
    console.log(`  Hash160(pubkey):    ${hex(h160)}`);
    console.log();

    // 6. Bitcoin Addresses
    console.log('[6] Bitcoin Addresses');
    console.log(`  P2PKH:              ${getAddr(addr_p2pkh, ctx, pub33, 0)}`);
    console.log(`  P2WPKH:             ${getAddr(addr_p2wpkh, ctx, pub33, 0)}`);
    console.log(`  P2TR:               ${getAddr(addr_p2tr, ctx, xonly, 0)}`);
    console.log();

    // 7. WIF
    console.log('[7] WIF Encoding');
    const wifBuf = Buffer.alloc(128);
    const wifLen = [128];
    check(wif_encode(ctx, PRIVKEY, 1, 0, wifBuf, wifLen), 'wif_encode');
    const wif = wifBuf.slice(0, wifLen[0]).toString();
    console.log(`  WIF:                ${wif}`);
    console.log();

    // 8. BIP-32
    console.log('[8] BIP-32 HD Key Derivation');
    const seed = Buffer.alloc(64, 0x42);
    const masterKey = Buffer.alloc(82);
    check(bip32_master(ctx, seed, 64, masterKey), 'bip32_master');
    const childKey = Buffer.alloc(82);
    check(bip32_derive(ctx, masterKey, "m/44'/0'/0'/0/0", childKey), 'bip32_derive');
    const childPriv = Buffer.alloc(32);
    const childPub = Buffer.alloc(33);
    check(bip32_privkey(ctx, childKey, childPriv), 'bip32_privkey');
    check(bip32_pubkey(ctx, childKey, childPub), 'bip32_pubkey');
    console.log(`  BIP-32 child priv:  ${hex(childPriv)}`);
    console.log(`  BIP-32 child pub:   ${hex(childPub)}`);
    console.log();

    // 9. Taproot
    console.log('[9] Taproot (BIP-341)');
    const tapOut = Buffer.alloc(32);
    const tapParity = [0];
    check(taproot_output(ctx, xonly, null, tapOut, tapParity), 'taproot_output');
    console.log(`  Output key:         ${hex(tapOut)}`);
    console.log(`  Parity:             ${tapParity[0]}`);
    const tapVrc = taproot_verify(ctx, tapOut, tapParity[0], xonly, null, 0);
    console.log(`  Verify:             ${tapVrc === UFSECP_OK ? 'VALID' : 'INVALID'}`);
    console.log();

    // 10. Pedersen
    console.log('[10] Pedersen Commitment');
    const pedVal = Buffer.alloc(32); pedVal[31] = 42;
    const pedBlind = Buffer.alloc(32); pedBlind[31] = 7;
    const pedCommit = Buffer.alloc(33);
    check(pedersen_commit(ctx, pedVal, pedBlind, pedCommit), 'pedersen_commit');
    console.log(`  Commitment:         ${hex(pedCommit)}`);
    const pvrc = pedersen_verify(ctx, pedCommit, pedVal, pedBlind);
    console.log(`  Verify:             ${pvrc === UFSECP_OK ? 'VALID' : 'INVALID'}`);
    console.log();

    ctx_destroy(ctx);
}

// ── GPU Demo ─────────────────────────────────────────────────────────────

function demoGPU() {
    console.log('=== GPU Operations ===\n');

    // 11. Backend Discovery
    console.log('[11] GPU Backend Discovery');
    const bids = new Uint32Array(4);
    const nBackends = gpu_backend_count(bids, 4);
    console.log(`  Backends compiled:  ${nBackends}`);

    let useBackend = 0;
    for (let i = 0; i < nBackends; i++) {
        const bid = bids[i];
        const name = gpu_backend_name(bid);
        const avail = gpu_is_available(bid);
        const devs = gpu_device_count(bid);
        console.log(`  Backend ${bid}: ${name.padEnd(8)} available=${avail} devices=${devs}`);
        if (avail && !useBackend) useBackend = bid;
    }

    if (!useBackend) {
        console.log('  No GPU backends available -- skipping GPU demos.\n');
        return;
    }

    // Create GPU context
    const gpp = [null];
    const grc = gpu_ctx_create(gpp, useBackend, 0);
    if (grc !== UFSECP_OK) {
        console.log(`  GPU context creation failed: ${gpu_error_str(grc)}`);
        return;
    }
    const gpu = gpp[0];

    const N = 4;

    // 12. Batch Key Generation
    console.log('\n[12] GPU Batch Key Generation (4 keys)');
    const scalars = Buffer.alloc(N * 32);
    for (let i = 0; i < N; i++) scalars[i * 32 + 31] = i + 1;

    const pubkeys = Buffer.alloc(N * 33);
    let rc = gpu_generator_mul(gpu, scalars, N, pubkeys);
    if (rc === UFSECP_OK) {
        for (let i = 0; i < N; i++) {
            console.log(`  GPU pubkey[${i}]:      ${hex(pubkeys.slice(i*33, (i+1)*33))}`);
        }
    } else {
        console.log(`  gpu_generator_mul_batch: ${gpu_error_str(rc)}`);
    }

    // 13. ECDSA Batch Verify
    console.log('\n[13] GPU ECDSA Batch Verify');
    const cpuCtx = createCtx();
    const msgs = Buffer.alloc(N * 32);
    const sigs = Buffer.alloc(N * 64);
    const pubs = Buffer.alloc(N * 33);

    for (let i = 0; i < N; i++) {
        const msgHash = Buffer.alloc(32);
        sha256(Buffer.from([i]), 1, msgHash);
        msgHash.copy(msgs, i * 32);

        const sk = Buffer.alloc(32);
        sk[31] = i + 1;
        const s = Buffer.alloc(64);
        ecdsa_sign(cpuCtx, msgHash, sk, s);
        s.copy(sigs, i * 64);

        pubkeys.copy(pubs, i * 33, i * 33, (i + 1) * 33);
    }
    ctx_destroy(cpuCtx);

    const results = Buffer.alloc(N);
    rc = gpu_ecdsa_verify(gpu, msgs, pubs, sigs, N, results);
    if (rc === UFSECP_OK) {
        const parts = [];
        for (let i = 0; i < N; i++) parts.push(`[${i}]=${results[i] ? 'VALID' : 'INVALID'}`);
        console.log(`  Results: ${parts.join(' ')}`);
    } else {
        console.log(`  gpu_ecdsa_verify_batch: ${gpu_error_str(rc)}`);
    }

    // 14. Hash160 Batch
    console.log('\n[14] GPU Hash160 Batch');
    const hashes = Buffer.alloc(N * 20);
    rc = gpu_hash160(gpu, pubkeys, N, hashes);
    if (rc === UFSECP_OK) {
        for (let i = 0; i < N; i++) {
            console.log(`  Hash160[${i}]:         ${hex(hashes.slice(i*20, (i+1)*20))}`);
        }
    } else {
        console.log(`  gpu_hash160_pubkey_batch: ${gpu_error_str(rc)}`);
    }

    // 15. MSM
    console.log('\n[15] GPU Multi-Scalar Multiplication');
    const msmResult = Buffer.alloc(33);
    rc = gpu_msm(gpu, scalars, pubkeys, N, msmResult);
    if (rc === UFSECP_OK) {
        console.log(`  MSM result:         ${hex(msmResult)}`);
    } else {
        console.log(`  gpu_msm: ${gpu_error_str(rc)}`);
    }

    gpu_ctx_destroy(gpu);
    console.log();
}

// ── Main ─────────────────────────────────────────────────────────────────

console.log('UltrafastSecp256k1 -- Node.js Example');
console.log(`ABI version: ${abi_version()}`);
console.log(`Library:     ${version_string()}`);
console.log();

demoCPU();
demoGPU();

console.log('All examples completed successfully.');
