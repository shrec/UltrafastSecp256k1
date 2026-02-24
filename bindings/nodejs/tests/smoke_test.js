/**
 * UltrafastSecp256k1 -- Node.js Binding Smoke Test (Golden Vectors)
 *
 * Verifies FFI boundary correctness using deterministic known-answer tests.
 * Runs in <2 seconds. Requires the ufsecp native addon.
 *
 * Usage:
 *   node smoke_test.js
 *   npx jest smoke_test.js
 */

'use strict';

const path = require('path');
const assert = require('assert');

// Attempt to load ufsecp
let Ufsecp;
try {
  Ufsecp = require(path.join(__dirname, '..', 'lib', 'ufsecp'));
} catch (e) {
  try {
    Ufsecp = require('../lib/ufsecp');
  } catch (e2) {
    console.error('Cannot load ufsecp binding. Build the native addon first.');
    process.exit(2);
  }
}

// ── Golden Vectors ──────────────────────────────────────────────────────────

const KNOWN_PRIVKEY = Buffer.from(
  '0000000000000000000000000000000000000000000000000000000000000001', 'hex');

const KNOWN_PUBKEY_COMPRESSED = Buffer.from(
  '0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798', 'hex');

const KNOWN_PUBKEY_XONLY = Buffer.from(
  '79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798', 'hex');

const SHA256_EMPTY = Buffer.from(
  'E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855', 'hex');

const RFC6979_MSG = Buffer.alloc(32);
const BIP340_AUX = Buffer.alloc(32);

// ── Tests ───────────────────────────────────────────────────────────────────

const tests = [];
function test(name, fn) { tests.push({ name, fn }); }

test('ctx_create_destroy', () => {
  const ctx = new Ufsecp();
  const abi = ctx.abiVersion();
  assert(abi >= 1, `ABI version ${abi} < 1`);
  ctx.close();
});

test('pubkey_create_golden', () => {
  const ctx = new Ufsecp();
  const pub = ctx.pubkeyCreate(KNOWN_PRIVKEY);
  assert(pub.equals(KNOWN_PUBKEY_COMPRESSED),
    `Expected ${KNOWN_PUBKEY_COMPRESSED.toString('hex')}, got ${pub.toString('hex')}`);
  ctx.close();
});

test('pubkey_xonly_golden', () => {
  const ctx = new Ufsecp();
  const xonly = ctx.pubkeyXonly(KNOWN_PRIVKEY);
  assert(xonly.equals(KNOWN_PUBKEY_XONLY));
  ctx.close();
});

test('ecdsa_sign_verify', () => {
  const ctx = new Ufsecp();
  const sig = ctx.ecdsaSign(RFC6979_MSG, KNOWN_PRIVKEY);
  assert.strictEqual(sig.length, 64);
  ctx.ecdsaVerify(RFC6979_MSG, sig, KNOWN_PUBKEY_COMPRESSED);

  // Mutated sig must fail
  const bad = Buffer.from(sig);
  bad[0] ^= 0x01;
  assert.throws(() => ctx.ecdsaVerify(RFC6979_MSG, bad, KNOWN_PUBKEY_COMPRESSED));
  ctx.close();
});

test('ecdsa_der_roundtrip', () => {
  const ctx = new Ufsecp();
  const sig = ctx.ecdsaSign(RFC6979_MSG, KNOWN_PRIVKEY);
  const der = ctx.ecdsaSigToDer(sig);
  const recovered = ctx.ecdsaSigFromDer(der);
  assert(recovered.equals(sig));
  ctx.close();
});

test('schnorr_sign_verify', () => {
  const ctx = new Ufsecp();
  const sig = ctx.schnorrSign(RFC6979_MSG, KNOWN_PRIVKEY, BIP340_AUX);
  assert.strictEqual(sig.length, 64);
  ctx.schnorrVerify(RFC6979_MSG, sig, KNOWN_PUBKEY_XONLY);

  const bad = Buffer.from(sig);
  bad[0] ^= 0x01;
  assert.throws(() => ctx.schnorrVerify(RFC6979_MSG, bad, KNOWN_PUBKEY_XONLY));
  ctx.close();
});

test('ecdsa_recover', () => {
  const ctx = new Ufsecp();
  const { signature, recid } = ctx.ecdsaSignRecoverable(RFC6979_MSG, KNOWN_PRIVKEY);
  assert(recid >= 0 && recid <= 3);
  const recovered = ctx.ecdsaRecover(RFC6979_MSG, signature, recid);
  assert(recovered.equals(KNOWN_PUBKEY_COMPRESSED));
  ctx.close();
});

test('sha256_golden', () => {
  const ctx = new Ufsecp();
  const digest = ctx.sha256(Buffer.alloc(0));
  assert(digest.equals(SHA256_EMPTY));
  ctx.close();
});

test('addr_p2wpkh', () => {
  const ctx = new Ufsecp();
  const addr = ctx.addrP2wpkh(KNOWN_PUBKEY_COMPRESSED, 0);
  assert(addr.startsWith('bc1q'), `Expected bc1q..., got ${addr}`);
  ctx.close();
});

test('wif_roundtrip', () => {
  const ctx = new Ufsecp();
  const wif = ctx.wifEncode(KNOWN_PRIVKEY, true, 0);
  const { privkey, compressed, network } = ctx.wifDecode(wif);
  assert(privkey.equals(KNOWN_PRIVKEY));
  assert.strictEqual(compressed, 1);
  assert.strictEqual(network, 0);
  ctx.close();
});

test('ecdh_symmetric', () => {
  const ctx = new Ufsecp();
  const k1 = KNOWN_PRIVKEY;
  const k2 = Buffer.from(
    '0000000000000000000000000000000000000000000000000000000000000002', 'hex');
  const pub1 = ctx.pubkeyCreate(k1);
  const pub2 = ctx.pubkeyCreate(k2);
  const s12 = ctx.ecdh(k1, pub2);
  const s21 = ctx.ecdh(k2, pub1);
  assert(s12.equals(s21), 'ECDH must be symmetric');
  ctx.close();
});

test('error_path', () => {
  const ctx = new Ufsecp();
  assert.throws(() => ctx.seckeyVerify(Buffer.alloc(32)));
  ctx.close();
});

test('golden_ecdsa_deterministic', () => {
  const ctx = new Ufsecp();
  const sig1 = ctx.ecdsaSign(RFC6979_MSG, KNOWN_PRIVKEY);
  const sig2 = ctx.ecdsaSign(RFC6979_MSG, KNOWN_PRIVKEY);
  assert(sig1.equals(sig2), 'RFC 6979 must be deterministic');
  ctx.close();
});

test('golden_schnorr_deterministic', () => {
  const ctx = new Ufsecp();
  const sig1 = ctx.schnorrSign(RFC6979_MSG, KNOWN_PRIVKEY, BIP340_AUX);
  const sig2 = ctx.schnorrSign(RFC6979_MSG, KNOWN_PRIVKEY, BIP340_AUX);
  assert(sig1.equals(sig2), 'Schnorr must be deterministic');
  ctx.close();
});

// ── Runner ──────────────────────────────────────────────────────────────────

async function run() {
  let passed = 0, failed = 0;
  for (const { name, fn } of tests) {
    try {
      await fn();
      console.log(`  [OK] ${name}`);
      passed++;
    } catch (e) {
      console.log(`  [FAIL] ${name}: ${e.message}`);
      failed++;
    }
  }
  console.log(`\n${'='.repeat(60)}`);
  console.log(`  Node.js smoke test: ${passed} passed, ${failed} failed`);
  console.log(`${'='.repeat(60)}`);
  process.exit(failed > 0 ? 1 : 0);
}

run();
