# ufsecp — Java

Java binding for [UltrafastSecp256k1](https://github.com/shrec/UltrafastSecp256k1) — high-performance secp256k1 elliptic curve cryptography via JNI.

## Features

- **ECDSA** — sign, verify, recover, DER serialization (RFC 6979)
- **Schnorr** — BIP-340 sign/verify
- **ECDH** — compressed, x-only, raw shared secret
- **BIP-32** — HD key derivation (master/derive/path/privkey/pubkey)
- **Taproot** — output key tweaking, verification (BIP-341)
- **Addresses** — P2PKH, P2WPKH, P2TR
- **WIF** — encode/decode
- **Hashing** — SHA-256 (hardware-accelerated), HASH160, tagged hash
- **Key tweaking** — negate, add, multiply

## Quick Start

```java
import com.ultrafast.ufsecp.Ufsecp;

try (Ufsecp ctx = Ufsecp.create()) {
    byte[] privkey = new byte[32];
    privkey[31] = 1;

    byte[] pubkey = ctx.pubkeyCreate(privkey);
    byte[] msgHash = Ufsecp.sha256("hello".getBytes());
    byte[] sig = ctx.ecdsaSign(msgHash, privkey);
    boolean valid = ctx.ecdsaVerify(msgHash, sig, pubkey);
}
```

## ECDSA Recovery

```java
RecoverableSignature rs = ctx.ecdsaSignRecoverable(msgHash, privkey);
byte[] recovered = ctx.ecdsaRecover(msgHash, rs.getSignature(), rs.getRecid());
```

## Schnorr (BIP-340)

```java
byte[] xonlyPub = ctx.pubkeyXonly(privkey);
byte[] schnorrSig = ctx.schnorrSign(msgHash, privkey, auxRand);
boolean ok = ctx.schnorrVerify(msgHash, schnorrSig, xonlyPub);
```

## BIP-32 HD Derivation

```java
byte[] master = ctx.bip32Master(seed);
byte[] child = ctx.bip32DerivePath(master, "m/44'/0'/0'/0/0");
byte[] childPriv = ctx.bip32Privkey(child);
byte[] childPub = ctx.bip32Pubkey(child);
```

## Taproot (BIP-341)

```java
TaprootOutputKeyResult tok = ctx.taprootOutputKey(xonlyPub, null);
byte[] tweakedPriv = ctx.taprootTweakSeckey(privkey, null);
boolean tapValid = ctx.taprootVerify(tok.getOutputKey(), tok.getParity(), xonlyPub, null);
```

## Architecture Note

The C ABI layer uses the **fast** (variable-time) implementation for maximum throughput. A constant-time (CT) layer with identical mathematical operations is available via the C++ headers for applications requiring timing-attack resistance.

## License

MIT
