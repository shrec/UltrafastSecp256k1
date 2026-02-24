// ============================================================================
// UltrafastSecp256k1 — Dart Binding Smoke Test (Golden Vectors)
// ============================================================================
// Verifies dart:ffi boundary correctness using deterministic known-answer tests.
// Runs in <2 seconds.
//
// Usage:
//   dart test test/smoke_test.dart
// ============================================================================

import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:ufsecp/ufsecp.dart';

// ── Golden Vectors ──────────────────────────────────────────────────────

Uint8List hexToBytes(String hex) {
  final result = Uint8List(hex.length ~/ 2);
  for (var i = 0; i < result.length; i++) {
    result[i] = int.parse(hex.substring(i * 2, i * 2 + 2), radix: 16);
  }
  return result;
}

final knownPrivkey = hexToBytes(
    '0000000000000000000000000000000000000000000000000000000000000001');

final knownPubkey = hexToBytes(
    '0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798');

final knownXonly = hexToBytes(
    '79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798');

final sha256Empty = hexToBytes(
    'E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855');

final msg32 = Uint8List(32);
final aux32 = Uint8List(32);

// ── Tests ───────────────────────────────────────────────────────────────

void main() {
  late UfsecpContext ctx;

  setUp(() {
    ctx = UfsecpContext();
  });

  tearDown(() {
    ctx.destroy();
  });

  test('ctx_create_abi', () {
    final abi = ctx.abiVersion;
    expect(abi, greaterThanOrEqualTo(1));
  });

  test('pubkey_create_golden', () {
    final pub = ctx.pubkeyCreate(knownPrivkey);
    expect(pub, equals(knownPubkey));
  });

  test('pubkey_xonly_golden', () {
    final xonly = ctx.pubkeyXonly(knownPrivkey);
    expect(xonly, equals(knownXonly));
  });

  test('ecdsa_sign_verify', () {
    final sig = ctx.ecdsaSign(msg32, knownPrivkey);
    expect(sig.length, equals(64));
    expect(ctx.ecdsaVerify(msg32, sig, knownPubkey), isTrue);

    // Mutated → fail
    final bad = Uint8List.fromList(sig);
    bad[0] ^= 0x01;
    expect(ctx.ecdsaVerify(msg32, bad, knownPubkey), isFalse);
  });

  test('schnorr_sign_verify', () {
    final sig = ctx.schnorrSign(msg32, knownPrivkey, aux32);
    expect(sig.length, equals(64));
    expect(ctx.schnorrVerify(msg32, sig, knownXonly), isTrue);
  });

  test('ecdsa_recover', () {
    final rec = ctx.ecdsaSignRecoverable(msg32, knownPrivkey);
    expect(rec.recoveryId, inInclusiveRange(0, 3));
    final pub = ctx.ecdsaRecover(msg32, rec.signature, rec.recoveryId);
    expect(pub, equals(knownPubkey));
  });

  test('sha256_golden', () {
    final digest = UfsecpContext.sha256(Uint8List(0));
    expect(digest, equals(sha256Empty));
  });

  test('addr_p2wpkh', () {
    final addr = ctx.addrP2wpkh(knownPubkey, Network.mainnet);
    expect(addr, startsWith('bc1q'));
  });

  test('wif_roundtrip', () {
    final wif = ctx.wifEncode(knownPrivkey, true, Network.mainnet);
    final decoded = ctx.wifDecode(wif);
    expect(decoded.privkey, equals(knownPrivkey));
    expect(decoded.compressed, isTrue);
  });

  test('ecdh_symmetric', () {
    final k2 = hexToBytes(
        '0000000000000000000000000000000000000000000000000000000000000002');
    final pub1 = ctx.pubkeyCreate(knownPrivkey);
    final pub2 = ctx.pubkeyCreate(k2);
    final s12 = ctx.ecdh(knownPrivkey, pub2);
    final s21 = ctx.ecdh(k2, pub1);
    expect(s12, equals(s21));
  });

  test('error_path', () {
    final zeroes = Uint8List(32);
    expect(() => ctx.pubkeyCreate(zeroes), throwsA(isA<UfsecpException>()));
  });

  test('ecdsa_deterministic', () {
    final sig1 = ctx.ecdsaSign(msg32, knownPrivkey);
    final sig2 = ctx.ecdsaSign(msg32, knownPrivkey);
    expect(sig1, equals(sig2));
  });
}
