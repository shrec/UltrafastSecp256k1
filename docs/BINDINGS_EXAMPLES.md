# Bindings Quick-Start Examples
## UltrafastSecp256k1 -- Copy-Paste Recipes per Language

> **3 examples per binding**: Sign/Verify, Address Derive, Error Handling  
> All examples use the **stable `ufsecp` ABI** (context-based API).

---

## Python

### Example 1: Sign & Verify (ECDSA + Schnorr)

```python
import os
from ufsecp import Ufsecp

with Ufsecp() as ctx:
    # Generate random private key
    privkey = os.urandom(32)
    while not ctx.seckey_verify(privkey):
        privkey = os.urandom(32)

    # Create public key
    pubkey = ctx.pubkey_create(privkey)            # 33 bytes compressed
    xonly  = ctx.pubkey_xonly(privkey)              # 32 bytes x-only

    # ECDSA sign + verify
    msg = ctx.sha256(b"Hello, secp256k1!")
    sig = ctx.ecdsa_sign(msg, privkey)
    assert ctx.ecdsa_verify(msg, sig, pubkey)      # True
    print(f"ECDSA OK  sig={sig.hex()[:32]}...")

    # Schnorr sign + verify
    aux = os.urandom(32)
    schnorr_sig = ctx.schnorr_sign(msg, privkey, aux)
    assert ctx.schnorr_verify(msg, schnorr_sig, xonly)
    print(f"Schnorr OK  sig={schnorr_sig.hex()[:32]}...")
```

### Example 2: Address Derivation (BIP-32 -> P2WPKH)

```python
from ufsecp import Ufsecp

with Ufsecp() as ctx:
    seed = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
    master = ctx.bip32_master(seed)
    child  = ctx.bip32_derive_path(master, "m/84'/0'/0'/0/0")

    # child is the derived private key
    pubkey = ctx.pubkey_create(child[:32])
    addr = ctx.addr_p2wpkh(pubkey, 0)              # 0 = mainnet
    print(f"Address: {addr}")                       # bc1q...
```

### Example 3: Error Handling

```python
from ufsecp import Ufsecp, UfsecpError

with Ufsecp() as ctx:
    # Invalid key -> UfsecpError
    try:
        ctx.pubkey_create(b"\x00" * 32)
    except UfsecpError as e:
        print(f"Error: {e.operation} -> code {e.code}: {e.message}")

    # Bad signature -> returns False (no exception!)
    msg = b"\x00" * 32
    sig = b"\xff" * 64
    pub = b"\x02" + b"\x00" * 32
    result = ctx.ecdsa_verify(msg, sig, pub)
    print(f"Invalid sig verify = {result}")         # False
```

---

## Node.js

### Example 1: Sign & Verify

```javascript
const { UfsecpContext } = require('@ultrafast/ufsecp');
const crypto = require('crypto');

const ctx = new UfsecpContext();
const privkey = crypto.randomBytes(32);
const pubkey  = ctx.pubkeyCreate(privkey);
const xonly   = ctx.pubkeyXonly(privkey);

const msg = ctx.sha256(Buffer.from('Hello, secp256k1!'));

// ECDSA
const sig = ctx.ecdsaSign(msg, privkey);
console.log('ECDSA valid:', ctx.ecdsaVerify(msg, sig, pubkey));

// Schnorr
const aux = crypto.randomBytes(32);
const schnorrSig = ctx.schnorrSign(msg, privkey, aux);
console.log('Schnorr valid:', ctx.schnorrVerify(msg, schnorrSig, xonly));

ctx.destroy();
```

### Example 2: Address Derivation

```javascript
const { UfsecpContext } = require('@ultrafast/ufsecp');

const ctx = new UfsecpContext();
const seed = Buffer.from('000102030405060708090a0b0c0d0e0f', 'hex');
const master = ctx.bip32Master(seed);
const child  = ctx.bip32DerivePath(master, "m/84'/0'/0'/0/0");

const pubkey = ctx.pubkeyCreate(child.slice(0, 32));
const addr = ctx.addrP2wpkh(pubkey, 0);
console.log('Address:', addr);

ctx.destroy();
```

### Example 3: Error Handling

```javascript
const { UfsecpContext, UfsecpError } = require('@ultrafast/ufsecp');

const ctx = new UfsecpContext();

try {
    ctx.pubkeyCreate(Buffer.alloc(32));  // zero key
} catch (e) {
    if (e instanceof UfsecpError) {
        console.log(`Error: ${e.operation} -> code ${e.code}: ${e.message}`);
    }
}

// Verify failure -> false, not exception
const ok = ctx.ecdsaVerify(Buffer.alloc(32), Buffer.alloc(64), Buffer.alloc(33));
console.log('Invalid verify:', ok);  // false

ctx.destroy();
```

---

## C#

### Example 1: Sign & Verify

```csharp
using System.Security.Cryptography;
using Ufsecp;

using var ctx = new Ufsecp.Ufsecp();

byte[] privkey = RandomNumberGenerator.GetBytes(32);
byte[] pubkey  = ctx.PubkeyCreate(privkey);
byte[] xonly   = ctx.PubkeyXonly(privkey);

byte[] msg = ctx.Sha256(System.Text.Encoding.UTF8.GetBytes("Hello, secp256k1!"));

// ECDSA
byte[] sig = ctx.EcdsaSign(msg, privkey);
Console.WriteLine($"ECDSA valid: {ctx.EcdsaVerify(msg, sig, pubkey)}");

// Schnorr
byte[] aux = RandomNumberGenerator.GetBytes(32);
byte[] schnorrSig = ctx.SchnorrSign(msg, privkey, aux);
Console.WriteLine($"Schnorr valid: {ctx.SchnorrVerify(msg, schnorrSig, xonly)}");
```

### Example 2: Address Derivation

```csharp
using Ufsecp;

using var ctx = new Ufsecp.Ufsecp();
byte[] seed = Convert.FromHexString("000102030405060708090a0b0c0d0e0f");
byte[] master = ctx.Bip32Master(seed);
byte[] child  = ctx.Bip32DerivePath(master, "m/84'/0'/0'/0/0");

byte[] pubkey = ctx.PubkeyCreate(child[..32]);
string addr = ctx.AddrP2wpkh(pubkey, 0);
Console.WriteLine($"Address: {addr}");
```

### Example 3: Error Handling

```csharp
using Ufsecp;

using var ctx = new Ufsecp.Ufsecp();

try {
    ctx.PubkeyCreate(new byte[32]);  // zero key
} catch (UfsecpException ex) {
    Console.WriteLine($"Error: {ex.Operation} -> code {ex.Code}: {ex.Message}");
}

// Verify failure -> false
bool ok = ctx.EcdsaVerify(new byte[32], new byte[64], new byte[33]);
Console.WriteLine($"Invalid verify: {ok}");  // False
```

---

## Java

### Example 1: Sign & Verify

```java
import com.ultrafast.ufsecp.Ufsecp;
import java.security.SecureRandom;

try (Ufsecp ctx = new Ufsecp()) {
    SecureRandom rng = SecureRandom.getInstanceStrong();
    byte[] privkey = new byte[32];
    rng.nextBytes(privkey);
    byte[] pubkey = ctx.pubkeyCreate(privkey);
    byte[] xonly  = ctx.pubkeyXonly(privkey);

    byte[] msg = Ufsecp.sha256("Hello, secp256k1!".getBytes());

    // ECDSA
    byte[] sig = ctx.ecdsaSign(msg, privkey);
    System.out.println("ECDSA valid: " + ctx.ecdsaVerify(msg, sig, pubkey));

    // Schnorr
    byte[] aux = new byte[32];
    rng.nextBytes(aux);
    byte[] schnorrSig = ctx.schnorrSign(msg, privkey, aux);
    System.out.println("Schnorr valid: " + ctx.schnorrVerify(msg, schnorrSig, xonly));
}
```

### Example 2: Address Derivation

```java
import com.ultrafast.ufsecp.Ufsecp;

try (Ufsecp ctx = new Ufsecp()) {
    byte[] seed = hexToBytes("000102030405060708090a0b0c0d0e0f");
    byte[] master = ctx.bip32Master(seed);
    byte[] child = ctx.bip32DerivePath(master, "m/84'/0'/0'/0/0");

    byte[] pubkey = ctx.pubkeyCreate(java.util.Arrays.copyOf(child, 32));
    String addr = ctx.addrP2wpkh(pubkey, 0);
    System.out.println("Address: " + addr);
}
```

### Example 3: Error Handling

```java
import com.ultrafast.ufsecp.Ufsecp;

try (Ufsecp ctx = new Ufsecp()) {
    // Java uses null-return + lastError() pattern
    byte[] pub = ctx.pubkeyCreate(new byte[32]);
    if (pub == null) {
        int code = ctx.lastError();
        String msg = ctx.lastErrorMsg();
        System.out.println("Error: code " + code + " -> " + msg);
    }

    // Verify failure -> false
    boolean ok = ctx.ecdsaVerify(new byte[32], new byte[64], new byte[33]);
    System.out.println("Invalid verify: " + ok);  // false
}
```

---

## Swift

### Example 1: Sign & Verify

```swift
import Ufsecp
import Security

let ctx = try UfsecpContext()
var privkey = Data(count: 32)
_ = SecRandomCopyBytes(kSecRandomDefault, 32, privkey.withUnsafeMutableBytes { $0.baseAddress! })

let pubkey = try ctx.pubkeyCreate(privkey: privkey)
let xonly  = try ctx.pubkeyXonly(privkey: privkey)
let msg = try ctx.sha256(Data("Hello, secp256k1!".utf8))

// ECDSA
let sig = try ctx.ecdsaSign(msgHash: msg, privkey: privkey)
print("ECDSA valid:", try ctx.ecdsaVerify(msgHash: msg, sig: sig, pubkey: pubkey))

// Schnorr
var aux = Data(count: 32)
_ = SecRandomCopyBytes(kSecRandomDefault, 32, aux.withUnsafeMutableBytes { $0.baseAddress! })
let schnorrSig = try ctx.schnorrSign(msg: msg, privkey: privkey, auxRand: aux)
print("Schnorr valid:", try ctx.schnorrVerify(msg: msg, sig: schnorrSig, pubkeyX: xonly))
```

### Example 2: Address Derivation

```swift
import Ufsecp

let ctx = try UfsecpContext()
let seed = Data(hexString: "000102030405060708090a0b0c0d0e0f")!
let master = try ctx.bip32Master(seed: seed)
let child  = try ctx.bip32DerivePath(master: master, path: "m/84'/0'/0'/0/0")

let pubkey = try ctx.pubkeyCreate(privkey: child.prefix(32))
let addr = try ctx.addrP2wpkh(pubkey: pubkey, network: .mainnet)
print("Address:", addr)
```

### Example 3: Error Handling

```swift
import Ufsecp

let ctx = try UfsecpContext()

do {
    _ = try ctx.pubkeyCreate(privkey: Data(count: 32))
} catch let error as UfsecpError {
    print("Error: \(error.operation) -> \(error.code)")
}

// Verify failure -> false (never throws)
let ok = try ctx.ecdsaVerify(msgHash: Data(count: 32), sig: Data(count: 64), pubkey: Data(count: 33))
print("Invalid verify:", ok)  // false
```

---

## Go

### Example 1: Sign & Verify

```go
package main

import (
    "crypto/rand"
    "fmt"
    "github.com/nicenemo/ufsecp"
)

func main() {
    ctx, _ := ufsecp.NewContext()
    defer ctx.Destroy()

    var privkey [32]byte
    rand.Read(privkey[:])
    pubkey, _ := ctx.PubkeyCreate(privkey)
    xonly, _  := ctx.PubkeyXonly(privkey)

    msg, _ := ufsecp.SHA256([]byte("Hello, secp256k1!"))

    // ECDSA
    sig, _ := ctx.EcdsaSign(msg, privkey)
    err := ctx.EcdsaVerify(msg, sig, pubkey)
    fmt.Println("ECDSA valid:", err == nil)

    // Schnorr
    var aux [32]byte
    rand.Read(aux[:])
    schnorrSig, _ := ctx.SchnorrSign(msg, privkey, aux)
    err = ctx.SchnorrVerify(msg, schnorrSig, xonly)
    fmt.Println("Schnorr valid:", err == nil)
}
```

### Example 2: Address Derivation

```go
package main

import (
    "encoding/hex"
    "fmt"
    "github.com/nicenemo/ufsecp"
)

func main() {
    ctx, _ := ufsecp.NewContext()
    defer ctx.Destroy()

    seed, _ := hex.DecodeString("000102030405060708090a0b0c0d0e0f")
    master, _ := ctx.BIP32Master(seed)
    child, _ := ctx.BIP32DerivePath(master, "m/84'/0'/0'/0/0")

    var privkey [32]byte
    copy(privkey[:], child.Privkey[:32])
    pubkey, _ := ctx.PubkeyCreate(privkey)
    addr, _ := ctx.AddrP2WPKH(pubkey, ufsecp.NetworkMainnet)
    fmt.Println("Address:", addr)
}
```

### Example 3: Error Handling

```go
package main

import (
    "errors"
    "fmt"
    "github.com/nicenemo/ufsecp"
)

func main() {
    ctx, _ := ufsecp.NewContext()
    defer ctx.Destroy()

    // Zero key -> error
    var zero [32]byte
    _, err := ctx.PubkeyCreate(zero)
    if errors.Is(err, ufsecp.ErrBadKey) {
        fmt.Println("Error:", err)
    }

    // Verify failure -> non-nil error
    var sig [64]byte
    var pub [33]byte
    err = ctx.EcdsaVerify(zero, sig, pub)
    fmt.Println("Invalid verify:", err != nil) // true
}
```

---

## Rust

### Example 1: Sign & Verify

```rust
use ufsecp::{Context, Network};
use rand::RngCore;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = Context::new()?;
    let mut privkey = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut privkey);

    let pubkey = ctx.pubkey_create(&privkey)?;
    let xonly  = ctx.pubkey_xonly(&privkey)?;
    let msg = Context::sha256(b"Hello, secp256k1!")?;

    // ECDSA
    let sig = ctx.ecdsa_sign(&msg, &privkey)?;
    println!("ECDSA valid: {}", ctx.ecdsa_verify(&msg, &sig, &pubkey));

    // Schnorr
    let mut aux = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut aux);
    let schnorr_sig = ctx.schnorr_sign(&msg, &privkey, &aux)?;
    println!("Schnorr valid: {}", ctx.schnorr_verify(&msg, &schnorr_sig, &xonly));

    Ok(())
}
```

### Example 2: Address Derivation

```rust
use ufsecp::{Context, Network};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = Context::new()?;
    let seed = hex::decode("000102030405060708090a0b0c0d0e0f")?;
    let master = ctx.bip32_master(&seed)?;
    let child = ctx.bip32_derive_path(&master, "m/84'/0'/0'/0/0")?;

    let pubkey = ctx.pubkey_create(&child[..32].try_into()?)?;
    let addr = ctx.addr_p2wpkh(&pubkey, Network::Mainnet)?;
    println!("Address: {}", addr);
    Ok(())
}
```

### Example 3: Error Handling

```rust
use ufsecp::{Context, Error, ErrorCode};

fn main() {
    let ctx = Context::new().unwrap();

    // Zero key -> Err
    match ctx.pubkey_create(&[0u8; 32]) {
        Err(e) if e.code == ErrorCode::BadKey => println!("Error: {:?}", e),
        Err(e) => println!("Unexpected: {:?}", e),
        Ok(_) => unreachable!(),
    }

    // Verify failure -> false (not Err)
    let ok = ctx.ecdsa_verify(&[0u8; 32], &[0u8; 64], &[0u8; 33]);
    println!("Invalid verify: {}", ok);  // false
}
```

---

## Dart

### Example 1: Sign & Verify

```dart
import 'dart:math';
import 'dart:typed_data';
import 'package:ufsecp/ufsecp.dart';

void main() {
  final ctx = UfsecpContext();
  final rng = Random.secure();
  final privkey = Uint8List(32);
  for (var i = 0; i < 32; i++) privkey[i] = rng.nextInt(256);

  final pubkey = ctx.pubkeyCreate(privkey);
  final xonly  = ctx.pubkeyXonly(privkey);
  final msg = UfsecpContext.sha256(Uint8List.fromList('Hello, secp256k1!'.codeUnits));

  // ECDSA
  final sig = ctx.ecdsaSign(msg, privkey);
  print('ECDSA valid: ${ctx.ecdsaVerify(msg, sig, pubkey)}');

  // Schnorr
  final aux = Uint8List(32);
  for (var i = 0; i < 32; i++) aux[i] = rng.nextInt(256);
  final schnorrSig = ctx.schnorrSign(msg, privkey, aux);
  print('Schnorr valid: ${ctx.schnorrVerify(msg, schnorrSig, xonly)}');

  ctx.destroy();
}
```

### Example 2: Address Derivation

```dart
import 'dart:typed_data';
import 'package:ufsecp/ufsecp.dart';

Uint8List hexToBytes(String hex) {
  final result = Uint8List(hex.length ~/ 2);
  for (var i = 0; i < result.length; i++) {
    result[i] = int.parse(hex.substring(i * 2, i * 2 + 2), radix: 16);
  }
  return result;
}

void main() {
  final ctx = UfsecpContext();
  final seed = hexToBytes('000102030405060708090a0b0c0d0e0f');
  final master = ctx.bip32Master(seed);
  final child = ctx.bip32DerivePath(master, "m/84'/0'/0'/0/0");

  final pubkey = ctx.pubkeyCreate(child.sublist(0, 32));
  final addr = ctx.addrP2wpkh(pubkey, Network.mainnet);
  print('Address: $addr');

  ctx.destroy();
}
```

### Example 3: Error Handling

```dart
import 'dart:typed_data';
import 'package:ufsecp/ufsecp.dart';

void main() {
  final ctx = UfsecpContext();

  try {
    ctx.pubkeyCreate(Uint8List(32));  // zero key
  } on UfsecpException catch (e) {
    print('Error: ${e.operation} -> ${e.error}');
  }

  // Verify failure -> false
  final ok = ctx.ecdsaVerify(Uint8List(32), Uint8List(64), Uint8List(33));
  print('Invalid verify: $ok');  // false

  ctx.destroy();
}
```

---

## PHP

### Example 1: Sign & Verify

```php
<?php
require_once 'vendor/autoload.php';
use Ultrafast\Ufsecp\Ufsecp;

$ctx = new Ufsecp();
$privkey = random_bytes(32);
$pubkey  = $ctx->pubkeyCreate($privkey);
$xonly   = $ctx->pubkeyXonly($privkey);
$msg = Ufsecp::sha256('Hello, secp256k1!');

// ECDSA
$sig = $ctx->ecdsaSign($msg, $privkey);
echo "ECDSA valid: " . ($ctx->ecdsaVerify($msg, $sig, $pubkey) ? "true" : "false") . "\n";

// Schnorr
$aux = random_bytes(32);
$schnorrSig = $ctx->schnorrSign($msg, $privkey, $aux);
echo "Schnorr valid: " . ($ctx->schnorrVerify($msg, $schnorrSig, $xonly) ? "true" : "false") . "\n";
```

### Example 2: Address Derivation

```php
<?php
use Ultrafast\Ufsecp\Ufsecp;

$ctx = new Ufsecp();
$seed = hex2bin('000102030405060708090a0b0c0d0e0f');
$master = $ctx->bip32Master($seed);
$child  = $ctx->bip32DerivePath($master, "m/84'/0'/0'/0/0");

$pubkey = $ctx->pubkeyCreate(substr($child, 0, 32));
$addr = $ctx->addrP2wpkh($pubkey, 0);
echo "Address: $addr\n";
```

### Example 3: Error Handling

```php
<?php
use Ultrafast\Ufsecp\Ufsecp;

$ctx = new Ufsecp();

try {
    $ctx->pubkeyCreate(str_repeat("\x00", 32));
} catch (\RuntimeException $e) {
    echo "Error: {$e->getMessage()}\n";
}

// Verify failure -> false
$ok = $ctx->ecdsaVerify(str_repeat("\x00", 32), str_repeat("\x00", 64), str_repeat("\x00", 33));
echo "Invalid verify: " . ($ok ? "true" : "false") . "\n";  // false
```

---

## Ruby

### Example 1: Sign & Verify

```ruby
require 'ufsecp'
require 'securerandom'

ctx = Ufsecp::Context.new
privkey = SecureRandom.random_bytes(32)
pubkey  = ctx.pubkey_create(privkey)
xonly   = ctx.pubkey_xonly(privkey)
msg = ctx.sha256("Hello, secp256k1!")

# ECDSA
sig = ctx.ecdsa_sign(msg, privkey)
puts "ECDSA valid: #{ctx.ecdsa_verify(msg, sig, pubkey)}"

# Schnorr
aux = SecureRandom.random_bytes(32)
schnorr_sig = ctx.schnorr_sign(msg, privkey, aux)
puts "Schnorr valid: #{ctx.schnorr_verify(msg, schnorr_sig, xonly)}"

ctx.destroy
```

### Example 2: Address Derivation

```ruby
require 'ufsecp'

ctx = Ufsecp::Context.new
seed = ["000102030405060708090a0b0c0d0e0f"].pack('H*')
master = ctx.bip32_master(seed)
child  = ctx.bip32_derive_path(master, "m/84'/0'/0'/0/0")

pubkey = ctx.pubkey_create(child[0, 32])
addr = ctx.addr_p2wpkh(pubkey, 0)
puts "Address: #{addr}"

ctx.destroy
```

### Example 3: Error Handling

```ruby
require 'ufsecp'

ctx = Ufsecp::Context.new

begin
  ctx.pubkey_create("\x00" * 32)
rescue Ufsecp::Error => e
  puts "Error: #{e.operation} -> code #{e.code}"
end

# Verify failure -> false
ok = ctx.ecdsa_verify("\x00" * 32, "\x00" * 64, "\x00" * 33)
puts "Invalid verify: #{ok}"  # false

ctx.destroy
```

---

## React Native

### Example 1: Sign & Verify

```javascript
import { UfsecpContext } from 'react-native-ufsecp';
import { randomBytes } from 'react-native-get-random-values';

const ctx = await UfsecpContext.create();
const privkeyHex = Array.from(randomBytes(32), b => b.toString(16).padStart(2,'0')).join('');
const pubkey = await ctx.pubkeyCreate(privkeyHex);
const xonly  = await ctx.pubkeyXonly(privkeyHex);

const msg = await UfsecpContext.sha256(Buffer.from('Hello, secp256k1!').toString('hex'));

// ECDSA
const sig = await ctx.ecdsaSign(msg, privkeyHex);
console.log('ECDSA valid:', await ctx.ecdsaVerify(msg, sig, pubkey));

// Schnorr
const aux = Array.from(randomBytes(32), b => b.toString(16).padStart(2,'0')).join('');
const schnorrSig = await ctx.schnorrSign(msg, privkeyHex, aux);
console.log('Schnorr valid:', await ctx.schnorrVerify(msg, schnorrSig, xonly));

await ctx.destroy();
```

### Example 2: Address Derivation

```javascript
import { UfsecpContext } from 'react-native-ufsecp';

const ctx = await UfsecpContext.create();
const seed = '000102030405060708090a0b0c0d0e0f';
const master = await ctx.bip32Master(seed);
const child  = await ctx.bip32DerivePath(master, "m/84'/0'/0'/0/0");

const pubkey = await ctx.pubkeyCreate(child.slice(0, 64));  // 32 bytes hex = 64 chars
const addr = await ctx.addrP2wpkh(pubkey, 0);
console.log('Address:', addr);

await ctx.destroy();
```

### Example 3: Error Handling

```javascript
import { UfsecpContext, UfsecpError } from 'react-native-ufsecp';

const ctx = await UfsecpContext.create();

try {
    await ctx.pubkeyCreate('00'.repeat(32));  // zero key
} catch (e) {
    if (e instanceof UfsecpError) {
        console.log(`Error: ${e.operation} -> code ${e.code}`);
    }
}

// Verify failure -> false (no exception)
const ok = await ctx.ecdsaVerify('00'.repeat(32), '00'.repeat(64), '00'.repeat(33));
console.log('Invalid verify:', ok);  // false

await ctx.destroy();
```
