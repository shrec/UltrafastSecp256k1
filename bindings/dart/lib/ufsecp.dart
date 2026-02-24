/// UltrafastSecp256k1 — Dart FFI bindings (ufsecp stable C ABI v1).
///
/// High-performance secp256k1 elliptic curve cryptography with dual-layer
/// constant-time architecture. Context-based API: one `UfsecpContext` per
/// isolate; `destroy()` is automatic via `Finalizer`.
///
/// ```dart
/// import 'package:ufsecp/ufsecp.dart';
///
/// final ctx = UfsecpContext();
/// final pubkey = ctx.pubkeyCreate(Uint8List(32)..[31] = 1);
/// ctx.destroy();
/// ```
library ufsecp;

import 'dart:ffi' as ffi;
import 'dart:io' show Platform;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

// ── Error codes ────────────────────────────────────────────────────────

enum UfsecpError {
  ok(0),
  nullArg(1),
  badKey(2),
  badPubkey(3),
  badSig(4),
  badInput(5),
  verifyFail(6),
  arith(7),
  selftest(8),
  internal(9),
  bufTooSmall(10);

  final int code;
  const UfsecpError(this.code);

  static UfsecpError fromCode(int c) =>
      UfsecpError.values.firstWhere((e) => e.code == c,
          orElse: () => UfsecpError.internal);
}

// ── Result types ───────────────────────────────────────────────────────

enum Network { mainnet, testnet }

class RecoverableSignature {
  final Uint8List signature;
  final int recoveryId;
  RecoverableSignature(this.signature, this.recoveryId);
}

class TaprootOutputKeyResult {
  final Uint8List outputKeyX;
  final int parity;
  TaprootOutputKeyResult(this.outputKeyX, this.parity);
}

class WifDecodeResult {
  final Uint8List privkey;
  final bool compressed;
  final Network network;
  WifDecodeResult(this.privkey, this.compressed, this.network);
}

class UfsecpException implements Exception {
  final String operation;
  final UfsecpError error;
  UfsecpException(this.operation, this.error);

  @override
  String toString() => 'UfsecpException($operation): ${error.name} (${error.code})';
}

// ── C typedef pairs ────────────────────────────────────────────────────

// Context lifecycle
typedef _CtxCreateC = ffi.Int32 Function(ffi.Pointer<ffi.Pointer<ffi.Void>>);
typedef _CtxCreateDart = int Function(ffi.Pointer<ffi.Pointer<ffi.Void>>);
typedef _CtxDestroyC = ffi.Void Function(ffi.Pointer<ffi.Void>);
typedef _CtxDestroyDart = void Function(ffi.Pointer<ffi.Void>);
typedef _CtxCloneC = ffi.Int32 Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Pointer<ffi.Void>>);
typedef _CtxCloneDart = int Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Pointer<ffi.Void>>);

// Version
typedef _VersionC = ffi.Uint32 Function();
typedef _VersionDart = int Function();
typedef _VersionStringC = ffi.Pointer<Utf8> Function();
typedef _VersionStringDart = ffi.Pointer<Utf8> Function();
typedef _ErrorStrC = ffi.Pointer<Utf8> Function(ffi.Int32);
typedef _ErrorStrDart = ffi.Pointer<Utf8> Function(int);

// Context error
typedef _LastErrorC = ffi.Int32 Function(ffi.Pointer<ffi.Void>);
typedef _LastErrorDart = int Function(ffi.Pointer<ffi.Void>);
typedef _LastErrorMsgC = ffi.Pointer<Utf8> Function(ffi.Pointer<ffi.Void>);
typedef _LastErrorMsgDart = ffi.Pointer<Utf8> Function(ffi.Pointer<ffi.Void>);

// Key ops (ctx, in, out)
typedef _PubkeyCreateC = ffi.Int32 Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>);
typedef _PubkeyCreateDart = int Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>);
typedef _PubkeyParseC = ffi.Int32 Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, ffi.Size, ffi.Pointer<ffi.Uint8>);
typedef _PubkeyParseDart = int Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, int, ffi.Pointer<ffi.Uint8>);
typedef _SeckeyVerifyC = ffi.Int32 Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>);
typedef _SeckeyVerifyDart = int Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>);
typedef _SeckeyMutateC = ffi.Int32 Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>);
typedef _SeckeyMutateDart = int Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>);
typedef _SeckeyTweakC = ffi.Int32 Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>);
typedef _SeckeyTweakDart = int Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>);

// ECDSA (ctx, msg, key/sig, out)
typedef _EcdsaSignC = ffi.Int32 Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>);
typedef _EcdsaSignDart = int Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>);
typedef _EcdsaVerifyC = ffi.Int32 Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>);
typedef _EcdsaVerifyDart = int Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>);
typedef _SigDerC = ffi.Int32 Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Size>);
typedef _SigDerDart = int Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Size>);
typedef _SigFromDerC = ffi.Int32 Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, ffi.Size, ffi.Pointer<ffi.Uint8>);
typedef _SigFromDerDart = int Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, int, ffi.Pointer<ffi.Uint8>);

// Recovery
typedef _SignRecoverableC = ffi.Int32 Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Int32>);
typedef _SignRecoverableDart = int Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Int32>);
typedef _RecoverC = ffi.Int32 Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>, ffi.Int32, ffi.Pointer<ffi.Uint8>);
typedef _RecoverDart = int Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>, int, ffi.Pointer<ffi.Uint8>);

// Schnorr
typedef _SchnorrSignC = ffi.Int32 Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>);
typedef _SchnorrSignDart = int Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>);
typedef _SchnorrVerifyC = ffi.Int32 Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>);
typedef _SchnorrVerifyDart = int Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>);

// ECDH
typedef _EcdhC = ffi.Int32 Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>);
typedef _EcdhDart = int Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>);

// Hashing (no ctx)
typedef _Sha256C = ffi.Int32 Function(ffi.Pointer<ffi.Uint8>, ffi.Size, ffi.Pointer<ffi.Uint8>);
typedef _Sha256Dart = int Function(ffi.Pointer<ffi.Uint8>, int, ffi.Pointer<ffi.Uint8>);
typedef _Hash160C = ffi.Int32 Function(ffi.Pointer<ffi.Uint8>, ffi.Size, ffi.Pointer<ffi.Uint8>);
typedef _Hash160Dart = int Function(ffi.Pointer<ffi.Uint8>, int, ffi.Pointer<ffi.Uint8>);
typedef _TaggedHashC = ffi.Int32 Function(ffi.Pointer<Utf8>, ffi.Pointer<ffi.Uint8>, ffi.Size, ffi.Pointer<ffi.Uint8>);
typedef _TaggedHashDart = int Function(ffi.Pointer<Utf8>, ffi.Pointer<ffi.Uint8>, int, ffi.Pointer<ffi.Uint8>);

// Addresses (ctx, pubkey, network, buf, len)
typedef _AddressC = ffi.Int32 Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, ffi.Int32, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Size>);
typedef _AddressDart = int Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, int, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Size>);

// WIF
typedef _WifEncodeC = ffi.Int32 Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, ffi.Int32, ffi.Int32, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Size>);
typedef _WifEncodeDart = int Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, int, int, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Size>);
typedef _WifDecodeC = ffi.Int32 Function(ffi.Pointer<ffi.Void>, ffi.Pointer<Utf8>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Int32>, ffi.Pointer<ffi.Int32>);
typedef _WifDecodeDart = int Function(ffi.Pointer<ffi.Void>, ffi.Pointer<Utf8>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Int32>, ffi.Pointer<ffi.Int32>);

// BIP-32 (ctx, ...)
typedef _Bip32MasterC = ffi.Int32 Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, ffi.Size, ffi.Pointer<ffi.Uint8>);
typedef _Bip32MasterDart = int Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, int, ffi.Pointer<ffi.Uint8>);
typedef _Bip32DeriveC = ffi.Int32 Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, ffi.Uint32, ffi.Pointer<ffi.Uint8>);
typedef _Bip32DeriveDart = int Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, int, ffi.Pointer<ffi.Uint8>);
typedef _Bip32DerivePathC = ffi.Int32 Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<Utf8>, ffi.Pointer<ffi.Uint8>);
typedef _Bip32DerivePathDart = int Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<Utf8>, ffi.Pointer<ffi.Uint8>);
typedef _Bip32GetKeyC = ffi.Int32 Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>);
typedef _Bip32GetKeyDart = int Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>);

// Taproot
typedef _TaprootOutputC = ffi.Int32 Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Int32>);
typedef _TaprootOutputDart = int Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Int32>);
typedef _TaprootTweakC = ffi.Int32 Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>);
typedef _TaprootTweakDart = int Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>);
typedef _TaprootVerifyC = ffi.Int32 Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, ffi.Int32, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>, ffi.Size);
typedef _TaprootVerifyDart = int Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, int, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>, int);

// ── Context class ──────────────────────────────────────────────────────

/// UfsecpContext wraps an opaque `ufsecp_ctx*`. Thread-safe, one per isolate.
///
/// Finalizer will call `ufsecp_ctx_destroy` when GCed, but explicit `destroy()`
/// is recommended for deterministic cleanup.
class UfsecpContext {
  late final ffi.DynamicLibrary _lib;
  ffi.Pointer<ffi.Void> _ctx = ffi.nullptr;
  bool _destroyed = false;

  // ── cached lookups ──
  late final _CtxDestroyDart _ctxDestroy;
  late final _CtxCloneDart _ctxClone;
  late final _LastErrorDart _lastError;
  late final _LastErrorMsgDart _lastErrorMsg;
  late final _VersionDart _version;
  late final _VersionDart _abiVersion;
  late final _VersionStringDart _versionString;
  late final _ErrorStrDart _errorStr;
  late final _PubkeyCreateDart _pubkeyCreate;
  late final _PubkeyCreateDart _pubkeyCreateUncompressed;
  late final _PubkeyParseDart _pubkeyParse;
  late final _PubkeyCreateDart _pubkeyXonly;
  late final _SeckeyVerifyDart _seckeyVerify;
  late final _SeckeyMutateDart _seckeyNegate;
  late final _SeckeyTweakDart _seckeyTweakAdd;
  late final _SeckeyTweakDart _seckeyTweakMul;
  late final _EcdsaSignDart _ecdsaSign;
  late final _EcdsaVerifyDart _ecdsaVerify;
  late final _SigDerDart _sigToDer;
  late final _SigFromDerDart _sigFromDer;
  late final _SignRecoverableDart _signRecoverable;
  late final _RecoverDart _recover;
  late final _SchnorrSignDart _schnorrSign;
  late final _SchnorrVerifyDart _schnorrVerify;
  late final _EcdhDart _ecdh;
  late final _EcdhDart _ecdhXonly;
  late final _EcdhDart _ecdhRaw;
  late final _Sha256Dart _sha256;
  late final _Hash160Dart _hash160;
  late final _TaggedHashDart _taggedHash;
  late final _AddressDart _addrP2pkh;
  late final _AddressDart _addrP2wpkh;
  late final _AddressDart _addrP2tr;
  late final _WifEncodeDart _wifEncode;
  late final _WifDecodeDart _wifDecode;
  late final _Bip32MasterDart _bip32Master;
  late final _Bip32DeriveDart _bip32Derive;
  late final _Bip32DerivePathDart _bip32DerivePath;
  late final _Bip32GetKeyDart _bip32Privkey;
  late final _Bip32GetKeyDart _bip32Pubkey;
  late final _TaprootOutputDart _taprootOutput;
  late final _TaprootTweakDart _taprootTweak;
  late final _TaprootVerifyDart _taprootVerify;

  /// Open the ufsecp native library and create a context.
  UfsecpContext({String? libraryPath}) {
    _lib = ffi.DynamicLibrary.open(libraryPath ?? _defaultLibName());
    _bindAll();

    final pp = calloc<ffi.Pointer<ffi.Void>>();
    try {
      final rc = _lib.lookupFunction<_CtxCreateC, _CtxCreateDart>(
          'ufsecp_ctx_create')(pp);
      if (rc != 0) {
        throw UfsecpException('ctx_create', UfsecpError.fromCode(rc));
      }
      _ctx = pp.value;
    } finally {
      calloc.free(pp);
    }
  }

  static String _defaultLibName() {
    if (Platform.isWindows) return 'ufsecp.dll';
    if (Platform.isMacOS) return 'libufsecp.dylib';
    return 'libufsecp.so';
  }

  /// Explicitly destroy the context. Safe to call multiple times.
  void destroy() {
    if (!_destroyed && _ctx != ffi.nullptr) {
      _ctxDestroy(_ctx);
      _ctx = ffi.nullptr;
      _destroyed = true;
    }
  }

  /// Deep-copy this context into a new independent context.
  UfsecpContext._fromPointer(this._lib, this._ctx) {
    _bindAll();
  }

  UfsecpContext cloneCtx() {
    _ensureAlive();
    final pp = calloc<ffi.Pointer<ffi.Void>>();
    try {
      final rc = _ctxClone(_ctx, pp);
      if (rc != 0) throw UfsecpException('ctx_clone', UfsecpError.fromCode(rc));
      return UfsecpContext._fromPointer(_lib, pp.value);
    } finally {
      calloc.free(pp);
    }
  }

  void _ensureAlive() {
    if (_destroyed) throw StateError('UfsecpContext already destroyed');
  }

  // ── Version ──────────────────────────────────────────────────────────

  int get version => _version();
  int get abiVersion => _abiVersion();
  String get versionString => _versionString().toDartString();
  /// Get human-readable string for an error code.
  String errorString(int code) => _errorStr(code).toDartString();

  int get lastError {
    _ensureAlive();
    return _lastError(_ctx);
  }

  String get lastErrorMsg {
    _ensureAlive();
    return _lastErrorMsg(_ctx).toDartString();
  }

  // ── Key operations ────────────────────────────────────────────────────

  /// Compressed public key (33 bytes).
  Uint8List pubkeyCreate(Uint8List privkey) {
    _check(privkey, 32, 'privkey');
    _ensureAlive();
    final pPriv = _allocCopy(privkey);
    final pOut = calloc<ffi.Uint8>(33);
    try {
      _throw(_pubkeyCreate(_ctx, pPriv, pOut), 'pubkey_create');
      return _read(pOut, 33);
    } finally {
      calloc.free(pPriv);
      calloc.free(pOut);
    }
  }

  /// Uncompressed public key (65 bytes).
  Uint8List pubkeyCreateUncompressed(Uint8List privkey) {
    _check(privkey, 32, 'privkey');
    _ensureAlive();
    final pPriv = _allocCopy(privkey);
    final pOut = calloc<ffi.Uint8>(65);
    try {
      _throw(_pubkeyCreateUncompressed(_ctx, pPriv, pOut), 'pubkey_create_uncompressed');
      return _read(pOut, 65);
    } finally {
      calloc.free(pPriv);
      calloc.free(pOut);
    }
  }

  /// Parse compressed (33) or uncompressed (65) → compressed 33 bytes.
  Uint8List pubkeyParse(Uint8List pubkey) {
    _ensureAlive();
    final pIn = _allocCopy(pubkey);
    final pOut = calloc<ffi.Uint8>(33);
    try {
      _throw(_pubkeyParse(_ctx, pIn, pubkey.length, pOut), 'pubkey_parse');
      return _read(pOut, 33);
    } finally {
      calloc.free(pIn);
      calloc.free(pOut);
    }
  }

  /// X-only public key (32 bytes, BIP-340).
  Uint8List pubkeyXonly(Uint8List privkey) {
    _check(privkey, 32, 'privkey');
    _ensureAlive();
    final pPriv = _allocCopy(privkey);
    final pOut = calloc<ffi.Uint8>(32);
    try {
      _throw(_pubkeyXonly(_ctx, pPriv, pOut), 'pubkey_xonly');
      return _read(pOut, 32);
    } finally {
      calloc.free(pPriv);
      calloc.free(pOut);
    }
  }

  /// Verify secret key is valid for secp256k1.
  bool seckeyVerify(Uint8List privkey) {
    _check(privkey, 32, 'privkey');
    _ensureAlive();
    final p = _allocCopy(privkey);
    try {
      return _seckeyVerify(_ctx, p) == 0;
    } finally {
      calloc.free(p);
    }
  }

  /// Negate secret key in-place.
  Uint8List seckeyNegate(Uint8List privkey) {
    _check(privkey, 32, 'privkey');
    _ensureAlive();
    final p = _allocCopy(privkey);
    try {
      _throw(_seckeyNegate(_ctx, p), 'seckey_negate');
      return _read(p, 32);
    } finally {
      calloc.free(p);
    }
  }

  /// Add tweak to secret key: key ← (key + tweak) mod n.
  Uint8List seckeyTweakAdd(Uint8List privkey, Uint8List tweak) {
    _check(privkey, 32, 'privkey');
    _check(tweak, 32, 'tweak');
    _ensureAlive();
    final pKey = _allocCopy(privkey);
    final pTw = _allocCopy(tweak);
    try {
      _throw(_seckeyTweakAdd(_ctx, pKey, pTw), 'seckey_tweak_add');
      return _read(pKey, 32);
    } finally {
      calloc.free(pKey);
      calloc.free(pTw);
    }
  }

  /// Multiply secret key by tweak: key ← (key × tweak) mod n.
  Uint8List seckeyTweakMul(Uint8List privkey, Uint8List tweak) {
    _check(privkey, 32, 'privkey');
    _check(tweak, 32, 'tweak');
    _ensureAlive();
    final pKey = _allocCopy(privkey);
    final pTw = _allocCopy(tweak);
    try {
      _throw(_seckeyTweakMul(_ctx, pKey, pTw), 'seckey_tweak_mul');
      return _read(pKey, 32);
    } finally {
      calloc.free(pKey);
      calloc.free(pTw);
    }
  }

  // ── ECDSA ─────────────────────────────────────────────────────────────

  /// ECDSA sign (RFC 6979). Returns 64-byte compact signature.
  Uint8List ecdsaSign(Uint8List msgHash, Uint8List privkey) {
    _check(msgHash, 32, 'msgHash');
    _check(privkey, 32, 'privkey');
    _ensureAlive();
    final pMsg = _allocCopy(msgHash);
    final pKey = _allocCopy(privkey);
    final pSig = calloc<ffi.Uint8>(64);
    try {
      _throw(_ecdsaSign(_ctx, pMsg, pKey, pSig), 'ecdsa_sign');
      return _read(pSig, 64);
    } finally {
      calloc.free(pMsg);
      calloc.free(pKey);
      calloc.free(pSig);
    }
  }

  /// Verify ECDSA compact signature.
  bool ecdsaVerify(Uint8List msgHash, Uint8List sig, Uint8List pubkey) {
    _check(msgHash, 32, 'msgHash');
    _check(sig, 64, 'sig');
    _check(pubkey, 33, 'pubkey');
    _ensureAlive();
    final pMsg = _allocCopy(msgHash);
    final pSig = _allocCopy(sig);
    final pPub = _allocCopy(pubkey);
    try {
      return _ecdsaVerify(_ctx, pMsg, pSig, pPub) == 0;
    } finally {
      calloc.free(pMsg);
      calloc.free(pSig);
      calloc.free(pPub);
    }
  }

  /// Compact sig → DER format.
  Uint8List ecdsaSigToDer(Uint8List sig) {
    _check(sig, 64, 'sig');
    _ensureAlive();
    final pSig = _allocCopy(sig);
    final pDer = calloc<ffi.Uint8>(72);
    final pLen = calloc<ffi.Size>(1);
    pLen.value = 72;
    try {
      _throw(_sigToDer(_ctx, pSig, pDer, pLen), 'ecdsa_sig_to_der');
      return _read(pDer, pLen.value);
    } finally {
      calloc.free(pSig);
      calloc.free(pDer);
      calloc.free(pLen);
    }
  }

  /// DER → compact 64-byte sig.
  Uint8List ecdsaSigFromDer(Uint8List der) {
    _ensureAlive();
    final pDer = _allocCopy(der);
    final pSig = calloc<ffi.Uint8>(64);
    try {
      _throw(_sigFromDer(_ctx, pDer, der.length, pSig), 'ecdsa_sig_from_der');
      return _read(pSig, 64);
    } finally {
      calloc.free(pDer);
      calloc.free(pSig);
    }
  }

  // ── Recovery ──────────────────────────────────────────────────────────

  /// Sign with recovery id.
  RecoverableSignature ecdsaSignRecoverable(Uint8List msgHash, Uint8List privkey) {
    _check(msgHash, 32, 'msgHash');
    _check(privkey, 32, 'privkey');
    _ensureAlive();
    final pMsg = _allocCopy(msgHash);
    final pKey = _allocCopy(privkey);
    final pSig = calloc<ffi.Uint8>(64);
    final pRec = calloc<ffi.Int32>(1);
    try {
      _throw(_signRecoverable(_ctx, pMsg, pKey, pSig, pRec), 'ecdsa_sign_recoverable');
      return RecoverableSignature(_read(pSig, 64), pRec.value);
    } finally {
      calloc.free(pMsg);
      calloc.free(pKey);
      calloc.free(pSig);
      calloc.free(pRec);
    }
  }

  /// Recover compressed public key from recoverable signature.
  Uint8List ecdsaRecover(Uint8List msgHash, Uint8List sig, int recid) {
    _check(msgHash, 32, 'msgHash');
    _check(sig, 64, 'sig');
    _ensureAlive();
    final pMsg = _allocCopy(msgHash);
    final pSig = _allocCopy(sig);
    final pPub = calloc<ffi.Uint8>(33);
    try {
      _throw(_recover(_ctx, pMsg, pSig, recid, pPub), 'ecdsa_recover');
      return _read(pPub, 33);
    } finally {
      calloc.free(pMsg);
      calloc.free(pSig);
      calloc.free(pPub);
    }
  }

  // ── Schnorr ───────────────────────────────────────────────────────────

  /// BIP-340 Schnorr sign. Returns 64-byte signature.
  Uint8List schnorrSign(Uint8List msg, Uint8List privkey, Uint8List auxRand) {
    _check(msg, 32, 'msg');
    _check(privkey, 32, 'privkey');
    _check(auxRand, 32, 'auxRand');
    _ensureAlive();
    final pMsg = _allocCopy(msg);
    final pKey = _allocCopy(privkey);
    final pAux = _allocCopy(auxRand);
    final pSig = calloc<ffi.Uint8>(64);
    try {
      _throw(_schnorrSign(_ctx, pMsg, pKey, pAux, pSig), 'schnorr_sign');
      return _read(pSig, 64);
    } finally {
      calloc.free(pMsg);
      calloc.free(pKey);
      calloc.free(pAux);
      calloc.free(pSig);
    }
  }

  /// BIP-340 Schnorr verify.
  bool schnorrVerify(Uint8List msg, Uint8List sig, Uint8List pubkeyX) {
    _check(msg, 32, 'msg');
    _check(sig, 64, 'sig');
    _check(pubkeyX, 32, 'pubkeyX');
    _ensureAlive();
    final pMsg = _allocCopy(msg);
    final pSig = _allocCopy(sig);
    final pPub = _allocCopy(pubkeyX);
    try {
      return _schnorrVerify(_ctx, pMsg, pSig, pPub) == 0;
    } finally {
      calloc.free(pMsg);
      calloc.free(pSig);
      calloc.free(pPub);
    }
  }

  // ── ECDH ──────────────────────────────────────────────────────────────

  /// ECDH: SHA256(compressed shared point).
  Uint8List ecdh(Uint8List privkey, Uint8List pubkey) {
    _check(privkey, 32, 'privkey');
    _check(pubkey, 33, 'pubkey');
    _ensureAlive();
    final pPriv = _allocCopy(privkey);
    final pPub = _allocCopy(pubkey);
    final pOut = calloc<ffi.Uint8>(32);
    try {
      _throw(_ecdh(_ctx, pPriv, pPub, pOut), 'ecdh');
      return _read(pOut, 32);
    } finally {
      calloc.free(pPriv);
      calloc.free(pPub);
      calloc.free(pOut);
    }
  }

  /// ECDH x-only variant.
  Uint8List ecdhXonly(Uint8List privkey, Uint8List pubkey) {
    _check(privkey, 32, 'privkey');
    _check(pubkey, 33, 'pubkey');
    _ensureAlive();
    final pPriv = _allocCopy(privkey);
    final pPub = _allocCopy(pubkey);
    final pOut = calloc<ffi.Uint8>(32);
    try {
      _throw(_ecdhXonly(_ctx, pPriv, pPub, pOut), 'ecdh_xonly');
      return _read(pOut, 32);
    } finally {
      calloc.free(pPriv);
      calloc.free(pPub);
      calloc.free(pOut);
    }
  }

  /// ECDH raw x-coordinate.
  Uint8List ecdhRaw(Uint8List privkey, Uint8List pubkey) {
    _check(privkey, 32, 'privkey');
    _check(pubkey, 33, 'pubkey');
    _ensureAlive();
    final pPriv = _allocCopy(privkey);
    final pPub = _allocCopy(pubkey);
    final pOut = calloc<ffi.Uint8>(32);
    try {
      _throw(_ecdhRaw(_ctx, pPriv, pPub, pOut), 'ecdh_raw');
      return _read(pOut, 32);
    } finally {
      calloc.free(pPriv);
      calloc.free(pPub);
      calloc.free(pOut);
    }
  }

  // ── Hashing (context-free) ────────────────────────────────────────────

  /// SHA-256.
  Uint8List sha256(Uint8List data) {
    _ensureAlive();
    final pIn = _allocCopy(data);
    final pOut = calloc<ffi.Uint8>(32);
    try {
      _throw(_sha256(pIn, data.length, pOut), 'sha256');
      return _read(pOut, 32);
    } finally {
      calloc.free(pIn);
      calloc.free(pOut);
    }
  }

  /// RIPEMD160(SHA256(data)).
  Uint8List hash160(Uint8List data) {
    _ensureAlive();
    final pIn = _allocCopy(data);
    final pOut = calloc<ffi.Uint8>(20);
    try {
      _throw(_hash160(pIn, data.length, pOut), 'hash160');
      return _read(pOut, 20);
    } finally {
      calloc.free(pIn);
      calloc.free(pOut);
    }
  }

  /// BIP-340 tagged hash.
  Uint8List taggedHash(String tag, Uint8List data) {
    _ensureAlive();
    final pTag = tag.toNativeUtf8();
    final pIn = _allocCopy(data);
    final pOut = calloc<ffi.Uint8>(32);
    try {
      _throw(_taggedHash(pTag, pIn, data.length, pOut), 'tagged_hash');
      return _read(pOut, 32);
    } finally {
      calloc.free(pTag);
      calloc.free(pIn);
      calloc.free(pOut);
    }
  }

  // ── Addresses ─────────────────────────────────────────────────────────

  /// P2PKH address.
  String addrP2PKH(Uint8List pubkey, {Network network = Network.mainnet}) {
    _check(pubkey, 33, 'pubkey');
    return _getAddress(_addrP2pkh, pubkey, network);
  }

  /// P2WPKH (Bech32, SegWit v0) address.
  String addrP2WPKH(Uint8List pubkey, {Network network = Network.mainnet}) {
    _check(pubkey, 33, 'pubkey');
    return _getAddress(_addrP2wpkh, pubkey, network);
  }

  /// P2TR (Bech32m, Taproot, SegWit v1) address from x-only key.
  String addrP2TR(Uint8List xonlyKey, {Network network = Network.mainnet}) {
    _check(xonlyKey, 32, 'xonlyKey');
    return _getAddress(_addrP2tr, xonlyKey, network);
  }

  // ── WIF ───────────────────────────────────────────────────────────────

  /// Encode private key as WIF.
  String wifEncode(Uint8List privkey, {bool compressed = true, Network network = Network.mainnet}) {
    _check(privkey, 32, 'privkey');
    _ensureAlive();
    final pKey = _allocCopy(privkey);
    final pBuf = calloc<ffi.Uint8>(128);
    final pLen = calloc<ffi.Size>(1);
    pLen.value = 128;
    try {
      _throw(_wifEncode(_ctx, pKey, compressed ? 1 : 0, network.index, pBuf, pLen), 'wif_encode');
      return String.fromCharCodes(pBuf.asTypedList(pLen.value));
    } finally {
      calloc.free(pKey);
      calloc.free(pBuf);
      calloc.free(pLen);
    }
  }

  /// Decode WIF to private key + metadata.
  WifDecodeResult wifDecode(String wif) {
    _ensureAlive();
    final pWif = wif.toNativeUtf8();
    final pKey = calloc<ffi.Uint8>(32);
    final pComp = calloc<ffi.Int32>(1);
    final pNet = calloc<ffi.Int32>(1);
    try {
      _throw(_wifDecode(_ctx, pWif, pKey, pComp, pNet), 'wif_decode');
      return WifDecodeResult(
        _read(pKey, 32),
        pComp.value == 1,
        pNet.value == 0 ? Network.mainnet : Network.testnet,
      );
    } finally {
      calloc.free(pWif);
      calloc.free(pKey);
      calloc.free(pComp);
      calloc.free(pNet);
    }
  }

  // ── BIP-32 ─────────────────────────────────────────────────────────────

  /// Master key from seed (16-64 bytes). Returns opaque key (82 bytes).
  Uint8List bip32Master(Uint8List seed) {
    if (seed.length < 16 || seed.length > 64) {
      throw ArgumentError('Seed must be 16-64 bytes');
    }
    _ensureAlive();
    final pSeed = _allocCopy(seed);
    final pKey = calloc<ffi.Uint8>(82);
    try {
      _throw(_bip32Master(_ctx, pSeed, seed.length, pKey), 'bip32_master');
      return _read(pKey, 82);
    } finally {
      calloc.free(pSeed);
      calloc.free(pKey);
    }
  }

  /// Derive child by index (>= 0x80000000 for hardened).
  Uint8List bip32Derive(Uint8List parent, int index) {
    _check(parent, 82, 'parent');
    _ensureAlive();
    final pPar = _allocCopy(parent);
    final pChild = calloc<ffi.Uint8>(82);
    try {
      _throw(_bip32Derive(_ctx, pPar, index, pChild), 'bip32_derive');
      return _read(pChild, 82);
    } finally {
      calloc.free(pPar);
      calloc.free(pChild);
    }
  }

  /// Derive from path, e.g. "m/44'/0'/0'/0/0".
  Uint8List bip32DerivePath(Uint8List master, String path) {
    _check(master, 82, 'master');
    _ensureAlive();
    final pMaster = _allocCopy(master);
    final pPath = path.toNativeUtf8();
    final pKey = calloc<ffi.Uint8>(82);
    try {
      _throw(_bip32DerivePath(_ctx, pMaster, pPath, pKey), 'bip32_derive_path');
      return _read(pKey, 82);
    } finally {
      calloc.free(pMaster);
      calloc.free(pPath);
      calloc.free(pKey);
    }
  }

  /// Extract 32-byte private key from extended key.
  Uint8List bip32Privkey(Uint8List key) {
    _check(key, 82, 'key');
    _ensureAlive();
    final pKey = _allocCopy(key);
    final pPriv = calloc<ffi.Uint8>(32);
    try {
      _throw(_bip32Privkey(_ctx, pKey, pPriv), 'bip32_privkey');
      return _read(pPriv, 32);
    } finally {
      calloc.free(pKey);
      calloc.free(pPriv);
    }
  }

  /// Extract compressed 33-byte public key from extended key.
  Uint8List bip32Pubkey(Uint8List key) {
    _check(key, 82, 'key');
    _ensureAlive();
    final pKey = _allocCopy(key);
    final pPub = calloc<ffi.Uint8>(33);
    try {
      _throw(_bip32Pubkey(_ctx, pKey, pPub), 'bip32_pubkey');
      return _read(pPub, 33);
    } finally {
      calloc.free(pKey);
      calloc.free(pPub);
    }
  }

  // ── Taproot ───────────────────────────────────────────────────────────

  /// Taproot output key.
  TaprootOutputKeyResult taprootOutputKey(Uint8List internalKeyX, {Uint8List? merkleRoot}) {
    _check(internalKeyX, 32, 'internalKeyX');
    _ensureAlive();
    final pInt = _allocCopy(internalKeyX);
    final pMr = merkleRoot != null ? _allocCopy(merkleRoot) : ffi.nullptr.cast<ffi.Uint8>();
    final pOut = calloc<ffi.Uint8>(32);
    final pPar = calloc<ffi.Int32>(1);
    try {
      _throw(_taprootOutput(_ctx, pInt, pMr, pOut, pPar), 'taproot_output_key');
      return TaprootOutputKeyResult(_read(pOut, 32), pPar.value);
    } finally {
      calloc.free(pInt);
      if (merkleRoot != null) calloc.free(pMr);
      calloc.free(pOut);
      calloc.free(pPar);
    }
  }

  /// Tweak private key for Taproot spending.
  Uint8List taprootTweakSeckey(Uint8List privkey, {Uint8List? merkleRoot}) {
    _check(privkey, 32, 'privkey');
    _ensureAlive();
    final pKey = _allocCopy(privkey);
    final pMr = merkleRoot != null ? _allocCopy(merkleRoot) : ffi.nullptr.cast<ffi.Uint8>();
    final pOut = calloc<ffi.Uint8>(32);
    try {
      _throw(_taprootTweak(_ctx, pKey, pMr, pOut), 'taproot_tweak_seckey');
      return _read(pOut, 32);
    } finally {
      calloc.free(pKey);
      if (merkleRoot != null) calloc.free(pMr);
      calloc.free(pOut);
    }
  }

  /// Verify Taproot commitment.
  bool taprootVerify(Uint8List outputKeyX, int parity, Uint8List internalKeyX, {Uint8List? merkleRoot}) {
    _check(outputKeyX, 32, 'outputKeyX');
    _check(internalKeyX, 32, 'internalKeyX');
    _ensureAlive();
    final pOut = _allocCopy(outputKeyX);
    final pInt = _allocCopy(internalKeyX);
    final pMr = merkleRoot != null ? _allocCopy(merkleRoot) : ffi.nullptr.cast<ffi.Uint8>();
    final mrLen = merkleRoot?.length ?? 0;
    try {
      return _taprootVerify(_ctx, pOut, parity, pInt, pMr, mrLen) == 0;
    } finally {
      calloc.free(pOut);
      calloc.free(pInt);
      if (merkleRoot != null) calloc.free(pMr);
    }
  }

  // ── Internal helpers ──────────────────────────────────────────────────

  void _bindAll() {
    _ctxDestroy = _lib.lookupFunction<_CtxDestroyC, _CtxDestroyDart>('ufsecp_ctx_destroy');
    _ctxClone = _lib.lookupFunction<_CtxCloneC, _CtxCloneDart>('ufsecp_ctx_clone');
    _lastError = _lib.lookupFunction<_LastErrorC, _LastErrorDart>('ufsecp_last_error');
    _lastErrorMsg = _lib.lookupFunction<_LastErrorMsgC, _LastErrorMsgDart>('ufsecp_last_error_msg');
    _version = _lib.lookupFunction<_VersionC, _VersionDart>('ufsecp_version');
    _abiVersion = _lib.lookupFunction<_VersionC, _VersionDart>('ufsecp_abi_version');
    _versionString = _lib.lookupFunction<_VersionStringC, _VersionStringDart>('ufsecp_version_string');
    _errorStr = _lib.lookupFunction<_ErrorStrC, _ErrorStrDart>('ufsecp_error_str');
    _pubkeyCreate = _lib.lookupFunction<_PubkeyCreateC, _PubkeyCreateDart>('ufsecp_pubkey_create');
    _pubkeyCreateUncompressed = _lib.lookupFunction<_PubkeyCreateC, _PubkeyCreateDart>('ufsecp_pubkey_create_uncompressed');
    _pubkeyParse = _lib.lookupFunction<_PubkeyParseC, _PubkeyParseDart>('ufsecp_pubkey_parse');
    _pubkeyXonly = _lib.lookupFunction<_PubkeyCreateC, _PubkeyCreateDart>('ufsecp_pubkey_xonly');
    _seckeyVerify = _lib.lookupFunction<_SeckeyVerifyC, _SeckeyVerifyDart>('ufsecp_seckey_verify');
    _seckeyNegate = _lib.lookupFunction<_SeckeyMutateC, _SeckeyMutateDart>('ufsecp_seckey_negate');
    _seckeyTweakAdd = _lib.lookupFunction<_SeckeyTweakC, _SeckeyTweakDart>('ufsecp_seckey_tweak_add');
    _seckeyTweakMul = _lib.lookupFunction<_SeckeyTweakC, _SeckeyTweakDart>('ufsecp_seckey_tweak_mul');
    _ecdsaSign = _lib.lookupFunction<_EcdsaSignC, _EcdsaSignDart>('ufsecp_ecdsa_sign');
    _ecdsaVerify = _lib.lookupFunction<_EcdsaVerifyC, _EcdsaVerifyDart>('ufsecp_ecdsa_verify');
    _sigToDer = _lib.lookupFunction<_SigDerC, _SigDerDart>('ufsecp_ecdsa_sig_to_der');
    _sigFromDer = _lib.lookupFunction<_SigFromDerC, _SigFromDerDart>('ufsecp_ecdsa_sig_from_der');
    _signRecoverable = _lib.lookupFunction<_SignRecoverableC, _SignRecoverableDart>('ufsecp_ecdsa_sign_recoverable');
    _recover = _lib.lookupFunction<_RecoverC, _RecoverDart>('ufsecp_ecdsa_recover');
    _schnorrSign = _lib.lookupFunction<_SchnorrSignC, _SchnorrSignDart>('ufsecp_schnorr_sign');
    _schnorrVerify = _lib.lookupFunction<_SchnorrVerifyC, _SchnorrVerifyDart>('ufsecp_schnorr_verify');
    _ecdh = _lib.lookupFunction<_EcdhC, _EcdhDart>('ufsecp_ecdh');
    _ecdhXonly = _lib.lookupFunction<_EcdhC, _EcdhDart>('ufsecp_ecdh_xonly');
    _ecdhRaw = _lib.lookupFunction<_EcdhC, _EcdhDart>('ufsecp_ecdh_raw');
    _sha256 = _lib.lookupFunction<_Sha256C, _Sha256Dart>('ufsecp_sha256');
    _hash160 = _lib.lookupFunction<_Hash160C, _Hash160Dart>('ufsecp_hash160');
    _taggedHash = _lib.lookupFunction<_TaggedHashC, _TaggedHashDart>('ufsecp_tagged_hash');
    _addrP2pkh = _lib.lookupFunction<_AddressC, _AddressDart>('ufsecp_addr_p2pkh');
    _addrP2wpkh = _lib.lookupFunction<_AddressC, _AddressDart>('ufsecp_addr_p2wpkh');
    _addrP2tr = _lib.lookupFunction<_AddressC, _AddressDart>('ufsecp_addr_p2tr');
    _wifEncode = _lib.lookupFunction<_WifEncodeC, _WifEncodeDart>('ufsecp_wif_encode');
    _wifDecode = _lib.lookupFunction<_WifDecodeC, _WifDecodeDart>('ufsecp_wif_decode');
    _bip32Master = _lib.lookupFunction<_Bip32MasterC, _Bip32MasterDart>('ufsecp_bip32_master');
    _bip32Derive = _lib.lookupFunction<_Bip32DeriveC, _Bip32DeriveDart>('ufsecp_bip32_derive');
    _bip32DerivePath = _lib.lookupFunction<_Bip32DerivePathC, _Bip32DerivePathDart>('ufsecp_bip32_derive_path');
    _bip32Privkey = _lib.lookupFunction<_Bip32GetKeyC, _Bip32GetKeyDart>('ufsecp_bip32_privkey');
    _bip32Pubkey = _lib.lookupFunction<_Bip32GetKeyC, _Bip32GetKeyDart>('ufsecp_bip32_pubkey');
    _taprootOutput = _lib.lookupFunction<_TaprootOutputC, _TaprootOutputDart>('ufsecp_taproot_output_key');
    _taprootTweak = _lib.lookupFunction<_TaprootTweakC, _TaprootTweakDart>('ufsecp_taproot_tweak_seckey');
    _taprootVerify = _lib.lookupFunction<_TaprootVerifyC, _TaprootVerifyDart>('ufsecp_taproot_verify');
  }

  String _getAddress(_AddressDart fn, Uint8List key, Network network) {
    _ensureAlive();
    final pKey = _allocCopy(key);
    final pBuf = calloc<ffi.Uint8>(128);
    final pLen = calloc<ffi.Size>(1);
    pLen.value = 128;
    try {
      _throw(fn(_ctx, pKey, network.index, pBuf, pLen), 'address');
      return String.fromCharCodes(pBuf.asTypedList(pLen.value));
    } finally {
      calloc.free(pKey);
      calloc.free(pBuf);
      calloc.free(pLen);
    }
  }

  static ffi.Pointer<ffi.Uint8> _allocCopy(Uint8List data) {
    final p = calloc<ffi.Uint8>(data.length);
    p.asTypedList(data.length).setAll(0, data);
    return p;
  }

  static Uint8List _read(ffi.Pointer<ffi.Uint8> p, int len) {
    return Uint8List.fromList(p.asTypedList(len));
  }

  static void _check(Uint8List data, int expected, String name) {
    if (data.length != expected) {
      throw ArgumentError('$name must be $expected bytes, got ${data.length}');
    }
  }

  static void _throw(int rc, String op) {
    if (rc != 0) {
      throw UfsecpException(op, UfsecpError.fromCode(rc));
    }
  }
}
