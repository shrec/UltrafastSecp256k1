// ============================================================================
// UltrafastSecp256k1 — React Native Binding Smoke Test (Golden Vectors)
// ============================================================================
// Verifies RN bridge boundary correctness using deterministic known-answer
// tests.  Runs in <2 seconds.  Designed for Jest or any async test runner.
//
// Usage:
//   npx jest tests/smoke_test.js
//
// NOTE: Requires a mock or real native module.  In CI, run via a RN test
// harness or mock NativeModules.Ufsecp for compile-check only.
// ============================================================================

const { UfsecpContext } = require('../lib/ufsecp');

// ── Golden Vectors ──────────────────────────────────────────────────────

const KNOWN_PRIVKEY = '0000000000000000000000000000000000000000000000000000000000000001';
const KNOWN_PUBKEY  = '0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798';
const KNOWN_XONLY   = '79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798';
const SHA256_EMPTY  = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855';
const MSG32         = '00'.repeat(32);
const AUX32         = '00'.repeat(32);

// ── Tests ───────────────────────────────────────────────────────────────

let ctx;

beforeAll(async () => {
  ctx = await UfsecpContext.create();
});

afterAll(async () => {
  if (ctx) await ctx.destroy();
});

test('ctx_create_abi', async () => {
  const abi = ctx.abiVersion;
  expect(abi).toBeGreaterThanOrEqual(1);
});

test('pubkey_create_golden', async () => {
  const pub = await ctx.pubkeyCreate(KNOWN_PRIVKEY);
  expect(pub.toLowerCase()).toBe(KNOWN_PUBKEY);
});

test('pubkey_xonly_golden', async () => {
  const xonly = await ctx.pubkeyXonly(KNOWN_PRIVKEY);
  expect(xonly.toLowerCase()).toBe(KNOWN_XONLY);
});

test('ecdsa_sign_verify', async () => {
  const sig = await ctx.ecdsaSign(MSG32, KNOWN_PRIVKEY);
  expect(sig.length).toBe(128); // 64 bytes hex

  const ok = await ctx.ecdsaVerify(MSG32, sig, KNOWN_PUBKEY);
  expect(ok).toBe(true);

  // Mutated → fail
  const firstByte = parseInt(sig.substring(0, 2), 16) ^ 0x01;
  const bad = firstByte.toString(16).padStart(2, '0') + sig.substring(2);
  const fail = await ctx.ecdsaVerify(MSG32, bad, KNOWN_PUBKEY);
  expect(fail).toBe(false);
});

test('schnorr_sign_verify', async () => {
  const sig = await ctx.schnorrSign(MSG32, KNOWN_PRIVKEY, AUX32);
  expect(sig.length).toBe(128);

  const ok = await ctx.schnorrVerify(MSG32, sig, KNOWN_XONLY);
  expect(ok).toBe(true);
});

test('ecdsa_recover', async () => {
  const { signature, recid } = await ctx.ecdsaSignRecoverable(MSG32, KNOWN_PRIVKEY);
  expect(recid).toBeGreaterThanOrEqual(0);
  expect(recid).toBeLessThanOrEqual(3);

  const pub = await ctx.ecdsaRecover(MSG32, signature, recid);
  expect(pub.toLowerCase()).toBe(KNOWN_PUBKEY);
});

test('sha256_golden', async () => {
  const digest = await UfsecpContext.sha256('');
  expect(digest.toLowerCase()).toBe(SHA256_EMPTY);
});

test('addr_p2wpkh', async () => {
  const addr = await ctx.addrP2wpkh(KNOWN_PUBKEY, 0);
  expect(addr.startsWith('bc1q')).toBe(true);
});

test('wif_roundtrip', async () => {
  const wif = await ctx.wifEncode(KNOWN_PRIVKEY, true, 0);
  const decoded = await ctx.wifDecode(wif);
  expect(decoded.privkey.toLowerCase()).toBe(KNOWN_PRIVKEY);
  expect(decoded.compressed).toBe(true);
  expect(decoded.network).toBe(0);
});

test('ecdh_symmetric', async () => {
  const k2 = '0000000000000000000000000000000000000000000000000000000000000002';
  const pub1 = await ctx.pubkeyCreate(KNOWN_PRIVKEY);
  const pub2 = await ctx.pubkeyCreate(k2);
  const s12 = await ctx.ecdh(KNOWN_PRIVKEY, pub2);
  const s21 = await ctx.ecdh(k2, pub1);
  expect(s12.toLowerCase()).toBe(s21.toLowerCase());
});

test('error_path', async () => {
  const zeroes = '00'.repeat(32);
  await expect(ctx.pubkeyCreate(zeroes)).rejects.toThrow();
});

test('ecdsa_deterministic', async () => {
  const sig1 = await ctx.ecdsaSign(MSG32, KNOWN_PRIVKEY);
  const sig2 = await ctx.ecdsaSign(MSG32, KNOWN_PRIVKEY);
  expect(sig1).toBe(sig2);
});
