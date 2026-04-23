/**
 * UltrafastSecp256k1 -- Node.js Smoke Test via koffi (compatible with Node 22+)
 *
 * Tests the ufsecp C ABI directly through koffi FFI.
 * Same golden vectors as the ffi-napi smoke test.
 *
 * Usage:
 *   UFSECP_LIB=/path/to/libufsecp.so node tests/smoke_koffi.js
 */

'use strict';

const koffi = require('koffi');
const assert = require('assert');
const path = require('path');

// ── Load library ────────────────────────────────────────────────────────

const LIB_PATH = process.env.UFSECP_LIB ||
    path.join(__dirname, '..', '..', '..', 'build-linux', 'include', 'ufsecp', 'libufsecp.so');

const lib = koffi.load(LIB_PATH);

// ── Type declarations ───────────────────────────────────────────────────

const UFSECP_OK = 0;

const ufsecp_ctx_create    = lib.func('int ufsecp_ctx_create(_Out_ void **ctx)');
const ufsecp_ctx_destroy   = lib.func('void ufsecp_ctx_destroy(void *ctx)');
const ufsecp_abi_version   = lib.func('uint32_t ufsecp_abi_version()');
const ufsecp_version_string = lib.func('const char *ufsecp_version_string()');
const ufsecp_error_str     = lib.func('const char *ufsecp_error_str(int code)');
const ufsecp_seckey_verify = lib.func('int ufsecp_seckey_verify(void *ctx, const uint8_t *key32)');
const ufsecp_pubkey_create = lib.func('int ufsecp_pubkey_create(void *ctx, const uint8_t *seckey32, uint8_t *pubkey33)');
const ufsecp_pubkey_xonly  = lib.func('int ufsecp_pubkey_xonly(void *ctx, const uint8_t *seckey32, uint8_t *xonly32, _Out_ int *parity)');
const ufsecp_ecdsa_sign    = lib.func('int ufsecp_ecdsa_sign(void *ctx, const uint8_t *msg32, const uint8_t *seckey32, uint8_t *sig64)');
const ufsecp_ecdsa_verify  = lib.func('int ufsecp_ecdsa_verify(void *ctx, const uint8_t *msg32, const uint8_t *sig64, const uint8_t *pubkey33)');
const ufsecp_schnorr_sign  = lib.func('int ufsecp_schnorr_sign(void *ctx, const uint8_t *msg32, const uint8_t *seckey32, const uint8_t *aux32, uint8_t *sig64)');
const ufsecp_schnorr_verify = lib.func('int ufsecp_schnorr_verify(void *ctx, const uint8_t *msg32, const uint8_t *sig64, const uint8_t *xonly32)');
const ufsecp_ecdsa_sign_recoverable = lib.func('int ufsecp_ecdsa_sign_recoverable(void *ctx, const uint8_t *msg32, const uint8_t *seckey32, uint8_t *sig64, _Out_ int *recid)');
const ufsecp_ecdsa_recover = lib.func('int ufsecp_ecdsa_recover(void *ctx, const uint8_t *msg32, const uint8_t *sig64, int recid, uint8_t *pubkey33)');
const ufsecp_ecdh          = lib.func('int ufsecp_ecdh(void *ctx, const uint8_t *seckey32, const uint8_t *pubkey33, uint8_t *secret32)');
const ufsecp_sha256        = lib.func('int ufsecp_sha256(const uint8_t *data, size_t len, uint8_t *digest32)');
const ufsecp_hash160       = lib.func('int ufsecp_hash160(const uint8_t *data, size_t len, uint8_t *digest20)');

// ── Golden Vectors ──────────────────────────────────────────────────────

const KNOWN_PRIVKEY = Buffer.from(
    '0000000000000000000000000000000000000000000000000000000000000001', 'hex');

const KNOWN_PUBKEY_COMPRESSED = Buffer.from(
    '0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798', 'hex');

const KNOWN_PUBKEY_XONLY = Buffer.from(
    '79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798', 'hex');

const SHA256_EMPTY = Buffer.from(
    'E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855', 'hex');

const HASH160_PUBKEY = Buffer.from(
    '751e76e8199196d454941c45d1b3a323f1433bd6', 'hex');

const RFC6979_MSG = Buffer.alloc(32);
const BIP340_AUX  = Buffer.alloc(32);

// ── Helper ──────────────────────────────────────────────────────────────

function createCtx() {
    const pp = [null];
    const err = ufsecp_ctx_create(pp);
    assert.strictEqual(err, UFSECP_OK, 'ctx_create failed');
    return pp[0];
}

// ── Tests ───────────────────────────────────────────────────────────────

const tests = [];
function test(name, fn) { tests.push({ name, fn }); }

test('ctx_create_abi', () => {
    const ctx = createCtx();
    const abi = ufsecp_abi_version();
    assert(abi >= 1, `ABI version ${abi} < 1`);
    const ver = ufsecp_version_string();
    assert(typeof ver === 'string' && ver.length > 0);
    ufsecp_ctx_destroy(ctx);
});

test('error_str', () => {
    const s = ufsecp_error_str(0);
    assert(typeof s === 'string');
    const s2 = ufsecp_error_str(2);
    assert(typeof s2 === 'string');
});

test('seckey_verify', () => {
    const ctx = createCtx();
    const err = ufsecp_seckey_verify(ctx, KNOWN_PRIVKEY);
    assert.strictEqual(err, UFSECP_OK);
    // Zero key should fail
    const zero = Buffer.alloc(32);
    const err2 = ufsecp_seckey_verify(ctx, zero);
    assert.notStrictEqual(err2, UFSECP_OK);
    ufsecp_ctx_destroy(ctx);
});

test('pubkey_create_golden', () => {
    const ctx = createCtx();
    const pub = Buffer.alloc(33);
    const err = ufsecp_pubkey_create(ctx, KNOWN_PRIVKEY, pub);
    assert.strictEqual(err, UFSECP_OK);
    assert(pub.equals(KNOWN_PUBKEY_COMPRESSED),
        `Expected ${KNOWN_PUBKEY_COMPRESSED.toString('hex')}, got ${pub.toString('hex')}`);
    ufsecp_ctx_destroy(ctx);
});

test('pubkey_xonly_golden', () => {
    const ctx = createCtx();
    const xonly = Buffer.alloc(32);
    const parity = [0];
    const err = ufsecp_pubkey_xonly(ctx, KNOWN_PRIVKEY, xonly, parity);
    assert.strictEqual(err, UFSECP_OK);
    assert(xonly.equals(KNOWN_PUBKEY_XONLY));
    ufsecp_ctx_destroy(ctx);
});

test('ecdsa_sign_verify', () => {
    const ctx = createCtx();
    const sig = Buffer.alloc(64);
    let err = ufsecp_ecdsa_sign(ctx, RFC6979_MSG, KNOWN_PRIVKEY, sig);
    assert.strictEqual(err, UFSECP_OK);
    err = ufsecp_ecdsa_verify(ctx, RFC6979_MSG, sig, KNOWN_PUBKEY_COMPRESSED);
    assert.strictEqual(err, UFSECP_OK);

    // Mutated sig must fail
    const bad = Buffer.from(sig);
    bad[0] ^= 0x01;
    err = ufsecp_ecdsa_verify(ctx, RFC6979_MSG, bad, KNOWN_PUBKEY_COMPRESSED);
    assert.notStrictEqual(err, UFSECP_OK);
    ufsecp_ctx_destroy(ctx);
});

test('ecdsa_deterministic', () => {
    const ctx = createCtx();
    const sig1 = Buffer.alloc(64);
    const sig2 = Buffer.alloc(64);
    let err = ufsecp_ecdsa_sign(ctx, RFC6979_MSG, KNOWN_PRIVKEY, sig1);
    assert.strictEqual(err, UFSECP_OK, `ecdsa_sign (1) returned error ${err}`);
    err = ufsecp_ecdsa_sign(ctx, RFC6979_MSG, KNOWN_PRIVKEY, sig2);
    assert.strictEqual(err, UFSECP_OK, `ecdsa_sign (2) returned error ${err}`);
    assert(sig1.equals(sig2), 'RFC 6979 must be deterministic');
    ufsecp_ctx_destroy(ctx);
});

test('schnorr_sign_verify', () => {
    const ctx = createCtx();
    const sig = Buffer.alloc(64);
    let err = ufsecp_schnorr_sign(ctx, RFC6979_MSG, KNOWN_PRIVKEY, BIP340_AUX, sig);
    assert.strictEqual(err, UFSECP_OK);
    err = ufsecp_schnorr_verify(ctx, RFC6979_MSG, sig, KNOWN_PUBKEY_XONLY);
    assert.strictEqual(err, UFSECP_OK);

    // Mutated sig must fail
    const bad = Buffer.from(sig);
    bad[0] ^= 0x01;
    err = ufsecp_schnorr_verify(ctx, RFC6979_MSG, bad, KNOWN_PUBKEY_XONLY);
    assert.notStrictEqual(err, UFSECP_OK);
    ufsecp_ctx_destroy(ctx);
});

test('ecdsa_recover', () => {
    const ctx = createCtx();
    const sig = Buffer.alloc(64);
    const recid = [0];
    let err = ufsecp_ecdsa_sign_recoverable(ctx, RFC6979_MSG, KNOWN_PRIVKEY, sig, recid);
    assert.strictEqual(err, UFSECP_OK);
    assert(recid[0] >= 0 && recid[0] <= 3);

    const recovered = Buffer.alloc(33);
    err = ufsecp_ecdsa_recover(ctx, RFC6979_MSG, sig, recid[0], recovered);
    assert.strictEqual(err, UFSECP_OK);
    assert(recovered.equals(KNOWN_PUBKEY_COMPRESSED),
        `Recovered ${recovered.toString('hex')} != ${KNOWN_PUBKEY_COMPRESSED.toString('hex')}`);
    ufsecp_ctx_destroy(ctx);
});

test('sha256_golden', () => {
    const digest = Buffer.alloc(32);
    // Pass a 1-byte dummy buffer with length 0 to avoid NULL pointer
    const err = ufsecp_sha256(Buffer.alloc(1), 0, digest);
    assert.strictEqual(err, UFSECP_OK, `sha256 returned error ${err}`);
    assert(digest.equals(SHA256_EMPTY),
        `SHA256("") = ${digest.toString('hex')}, expected ${SHA256_EMPTY.toString('hex')}`);
});

test('hash160_golden', () => {
    const digest = Buffer.alloc(20);
    const err = ufsecp_hash160(KNOWN_PUBKEY_COMPRESSED, 33, digest);
    assert.strictEqual(err, UFSECP_OK);
    assert(digest.equals(HASH160_PUBKEY),
        `hash160(G) = ${digest.toString('hex')}, expected ${HASH160_PUBKEY.toString('hex')}`);
});

test('ecdh_symmetric', () => {
    const ctx = createCtx();
    const k2 = Buffer.from(
        '0000000000000000000000000000000000000000000000000000000000000002', 'hex');
    const pub1 = Buffer.alloc(33);
    const pub2 = Buffer.alloc(33);
    let err2 = ufsecp_pubkey_create(ctx, KNOWN_PRIVKEY, pub1);
    assert.strictEqual(err2, UFSECP_OK, `pubkey_create (k1) returned error ${err2}`);
    err2 = ufsecp_pubkey_create(ctx, k2, pub2);
    assert.strictEqual(err2, UFSECP_OK, `pubkey_create (k2) returned error ${err2}`);

    const s12 = Buffer.alloc(32);
    const s21 = Buffer.alloc(32);
    let err3 = ufsecp_ecdh(ctx, KNOWN_PRIVKEY, pub2, s12);
    assert.strictEqual(err3, UFSECP_OK, `ecdh (k1,P2) returned error ${err3}`);
    err3 = ufsecp_ecdh(ctx, k2, pub1, s21);
    assert.strictEqual(err3, UFSECP_OK, `ecdh (k2,P1) returned error ${err3}`);
    assert(s12.equals(s21), 'ECDH must be symmetric');
    ufsecp_ctx_destroy(ctx);
});

// ── Runner ──────────────────────────────────────────────────────────────

async function run() {
    let passed = 0, failed = 0;
    console.log(`  Library: ${LIB_PATH}`);
    console.log(`  Node.js: ${process.version}\n`);
    for (const { name, fn } of tests) {
        try {
            await fn();
            console.log(`  [OK]   ${name}`);
            passed++;
        } catch (e) {
            console.log(`  [FAIL] ${name}: ${e.message}`);
            failed++;
        }
    }
    console.log(`\n${'='.repeat(60)}`);
    console.log(`  Node.js koffi smoke test: ${passed} passed, ${failed} failed`);
    console.log(`${'='.repeat(60)}`);
    process.exit(failed > 0 ? 1 : 0);
}

run();
