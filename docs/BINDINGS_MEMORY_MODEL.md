# Bindings Memory & Secret Handling Model
## UltrafastSecp256k1 Cross-Language Security Boundary Contract

> **Scope**: Documents how secret material (private keys, nonces, shared secrets) crosses FFI boundaries in each binding.  
> **Honest policy**: States what IS guaranteed and what IS NOT — no overclaiming.

---

## 1. C ABI Layer (Ground Truth)

### What the C library guarantees:

| Guarantee | Status | Mechanism |
|---|:---:|---|
| No heap allocation in hot paths | **YES** | Caller-provided buffers only |
| Private key never stored in context | **YES** | Keys passed per-call, not retained |
| Nonce (k) zeroized after signing | **YES** | `memset_explicit` in sign paths |
| Context scratchpad zeroized on destroy | **YES** | `ufsecp_ctx_destroy()` zeroes internals |
| No key material in error messages | **YES** | Error strings are static constants |
| Constant-time operations | **YES** | See `docs/CT_EMPIRICAL_REPORT.md` |

### What the C library does NOT guarantee:

| Non-Guarantee | Reason |
|---|---|
| Caller buffer zeroization | Caller's responsibility to zero after use |
| Protection against memory dumps | OS-level concern (mlock, etc.) |
| Side-channel resistance on all CPUs | Verified on x86-64/ARM64 only |

---

## 2. Per-Language FFI Boundary Analysis

### 2.1 Python (ctypes)

| Aspect | Behavior | Risk Level |
|---|---|:---:|
| Key input type | `bytes` (immutable) | ⚠️ |
| Key in Python memory | Immutable — **cannot be zeroized** by wrapper | ⚠️ |
| ctypes buffer lifetime | Temporary `c_char` arrays, freed on scope exit | Low |
| GC timing | Non-deterministic — key bytes may linger in heap | ⚠️ |
| Mitigation | Wrapper creates `ctypes.create_string_buffer()` for native calls and zeros it after | ✅ |

**Honest statement**: Python `bytes` objects are immutable. The wrapper zeros its own ctypes buffers, but the caller's `bytes` object containing the private key **cannot be securely erased** from Python. Callers should use `bytearray` when possible and manually zero after use.

```python
# Recommended pattern:
key = bytearray(os.urandom(32))
try:
    sig = ctx.ecdsa_sign(msg, bytes(key))
finally:
    for i in range(len(key)):
        key[i] = 0
```

### 2.2 Node.js (N-API / node-ffi-napi)

| Aspect | Behavior | Risk Level |
|---|---|:---:|
| Key input type | `Buffer` or `Uint8Array` | Low |
| Buffer zeroization | `Buffer.alloc()` supports `.fill(0)` | ✅ |
| GC timing | V8 GC non-deterministic; old Buffer data may linger | ⚠️ |
| Native copy | Data is copied to native heap for FFI call | Low |
| Mitigation | Wrapper copies to native allocation, zeros after | ✅ |

**Honest statement**: Node `Buffer` CAN be zeroed synchronously via `.fill(0)`. However, V8 may have copied the buffer contents during optimization (JIT, deopt, etc.). No guaranteed protection against heap inspection.

```javascript
// Recommended pattern:
const key = Buffer.alloc(32);
crypto.randomFillSync(key);
try {
    const sig = ctx.ecdsaSign(msg, key);
} finally {
    key.fill(0);
}
```

### 2.3 C# (P/Invoke)

| Aspect | Behavior | Risk Level |
|---|---|:---:|
| Key input type | `byte[]` (managed heap) | ⚠️ |
| Pinning | Required for P/Invoke — `GCHandle.Alloc(Pinned)` | ✅ |
| GC compaction | May copy `byte[]` before pinning | ⚠️ |
| `SecureString` support | Not applicable (binary data, not strings) | — |
| Mitigation | Wrapper pins arrays, zeros after unpinning | ✅ |

**Honest statement**: .NET GC may copy array contents during compaction before the wrapper pins them. The wrapper zeros the pinned buffer after the FFI call, but prior GC copies are not erasable. For maximum security, use `stackalloc byte[32]` with `Span<byte>`.

```csharp
// Recommended pattern:
Span<byte> key = stackalloc byte[32];
RandomNumberGenerator.Fill(key);
try {
    var sig = ctx.EcdsaSign(msg, key.ToArray());
} finally {
    key.Clear();
}
```

### 2.4 Java (JNI)

| Aspect | Behavior | Risk Level |
|---|---|:---:|
| Key input type | `byte[]` (Java heap) | ⚠️ |
| JNI copy | `GetByteArrayElements` may copy or pin | ⚠️ |
| GC behavior | Generational GC — objects may be copied between regions | ⚠️ |
| Zeroization in JNI | JNI bridge zeros the C-side copy after use | ✅ |
| Mitigation | `Arrays.fill(key, (byte)0)` after use recommended | ✅ |

**Honest statement**: Java's GC may relocate `byte[]` arrays between generations, leaving copies in old heap regions. The JNI native bridge zeros its working copy. The Java-side array should be zeroed by the caller.

```java
// Recommended pattern:
byte[] key = new byte[32];
SecureRandom.getInstanceStrong().nextBytes(key);
try {
    byte[] sig = ctx.ecdsaSign(msg, key);
} finally {
    Arrays.fill(key, (byte) 0);
}
```

### 2.5 Swift (C Interop)

| Aspect | Behavior | Risk Level |
|---|---|:---:|
| Key input type | `Data` (heap-allocated, COW) | ⚠️ |
| Copy-on-Write | `Data` may share backing storage until mutation | ⚠️ |
| Swift bridging | `Data.withUnsafeBytes` provides direct pointer — no copy | ✅ |
| ARC dealloc | Deterministic refcount — freed promptly when last ref drops | Low |
| Mitigation | Wrapper operates via `withUnsafeBytes`; recommends `resetBytes(in:)` | ✅ |

**Honest statement**: Swift `Data` uses COW. If the caller has a single reference, `resetBytes(in:)` will zero in-place. If multiple refs exist, a copy may have been made. ARC ensures deterministic deallocation but does not guarantee zeroization of freed pages.

```swift
// Recommended pattern:
var key = Data(count: 32)
key.withUnsafeMutableBytes { SecRandomCopyBytes(kSecRandomDefault, 32, $0.baseAddress!) }
defer { key.resetBytes(in: 0..<32) }
let sig = try ctx.ecdsaSign(msgHash: msg, privkey: key)
```

### 2.6 Go (CGo)

| Aspect | Behavior | Risk Level |
|---|---|:---:|
| Key input type | `[32]byte` (stack or heap, fixed-size) | Low |
| CGo marshaling | Go passes pointer directly if stack-allocated | ✅ |
| GC behavior | Moving GC — but `[32]byte` on stack is not GC-managed | Low |
| Mitigation | Fixed-size arrays encourage stack allocation; wrapper does not copy | ✅ |

**Honest statement**: Go's `[32]byte` arrays, when stack-allocated, are not subject to GC relocation. CGo passes the address directly. Heap-escaped arrays ARE subject to GC movement. Callers should zero arrays explicitly after use.

```go
// Recommended pattern:
var key [32]byte
rand.Read(key[:])
defer func() {
    for i := range key {
        key[i] = 0
    }
}()
sig, err := ctx.EcdsaSign(msg, key)
```

### 2.7 Rust (FFI)

| Aspect | Behavior | Risk Level |
|---|---|:---:|
| Key input type | `&[u8; 32]` (borrowed reference) | ✅ |
| No hidden copies | Rust guarantees no implicit copy of `&[u8; 32]` | ✅ |
| Drop semantics | Deterministic — `Drop` impl runs immediately | ✅ |
| `zeroize` crate | Available for caller use; not enforced by wrapper | ✅ |
| Mitigation | Wrapper passes the slice pointer directly to C FFI | ✅ |

**Honest statement**: Rust provides the strongest guarantees. No implicit copies, deterministic drop, and the `zeroize` crate can be used for guaranteed zeroization. The wrapper does not add overhead or copies.

```rust
// Recommended pattern:
use zeroize::Zeroize;
let mut key = [0u8; 32];
OsRng.fill_bytes(&mut key);
let sig = ctx.ecdsa_sign(&[0u8; 32], &key)?;
key.zeroize();
```

### 2.8 Dart (dart:ffi)

| Aspect | Behavior | Risk Level |
|---|---|:---:|
| Key input type | `Uint8List` (heap-allocated, typed array) | ⚠️ |
| FFI marshaling | `allocate<Uint8>()` + memcpy to native heap | ⚠️ |
| Finalizer | `NativeFinalizer` zeroizes native allocation | ✅ |
| Dart GC | Non-deterministic; old allocations may linger | ⚠️ |
| Mitigation | Wrapper zeros the native-side copy after FFI call | ✅ |

**Honest statement**: Dart's GC is non-deterministic. The wrapper zeros its native-side temporary allocations. The Dart-side `Uint8List` CAN be zeroed manually but may have been copied by the runtime.

```dart
// Recommended pattern:
final key = Uint8List(32);
Random.secure().nextBytes(key);
try {
    final sig = ctx.ecdsaSign(msg, key);
} finally {
    key.fillRange(0, 32, 0);
}
```

### 2.9 PHP (ext-ffi)

| Aspect | Behavior | Risk Level |
|---|---|:---:|
| Key input type | `string` (binary, COW, immutable once created) | ⚠️ |
| PHP string interning | Strings may be interned by the engine | ⚠️ |
| FFI marshaling | `FFI::memcpy` to native buffer | Low |
| Mitigation | Wrapper zeros native buffer after use | ✅ |

**Honest statement**: PHP strings are immutable and reference-counted. The original `$privkey` string CANNOT be zeroized from userland. The wrapper's native-side buffer IS zeroed after use.

### 2.10 Ruby (ffi gem)

| Aspect | Behavior | Risk Level |
|---|---|:---:|
| Key input type | `String` (mutable binary) | Low |
| Ruby string mutation | Strings ARE mutable — can be zeroed in place | ✅ |
| FFI copy | `ffi` gem copies to native heap for call | ⚠️ |
| GC timing | Mark-and-sweep, non-deterministic | ⚠️ |
| Mitigation | Wrapper zeros native copy; caller can zero Ruby string | ✅ |

```ruby
# Recommended pattern:
key = SecureRandom.random_bytes(32)
begin
  sig = ctx.ecdsa_sign(msg, key)
ensure
  key.replace("\x00" * 32)
end
```

### 2.11 React Native (NativeModules Bridge)

| Aspect | Behavior | Risk Level |
|---|---|:---:|
| Key input type | Hex `string` (JavaScript) | ⚠️ |
| Bridge marshaling | JSON/MessageQueue → native Java/ObjC | ⚠️ |
| JS string immutable | JavaScript strings are immutable | ⚠️ |
| Native side | Java/ObjC module does hex→bytes conversion, calls C, seros | ✅ |
| Mitigation | Native module zeros byte arrays; JS strings cannot be zeroed | ⚠️ |

**Honest statement**: React Native passes hex strings over the bridge. JavaScript strings are immutable and may be interned by the JS engine. The native module zeros its working buffers. **No protection for the JS-side hex string.** This is an inherent limitation of RN's bridge architecture.

---

## 3. Summary Classification

| Language | Can Caller Zero Key? | Wrapper Zeros Native Copy? | GC Risk | Overall Risk |
|---|:---:|:---:|:---:|:---:|
| **Rust** | ✅ (zeroize) | N/A (no copy) | None | **Low** |
| **Go** | ✅ (manual loop) | N/A (direct ptr) | Low | **Low** |
| **Swift** | ✅ (resetBytes) | ✅ | Low (ARC) | **Low** |
| **C#** | ⚠️ (pre-pin copies) | ✅ | Medium | **Medium** |
| **Java** | ⚠️ (GC copies) | ✅ | Medium | **Medium** |
| **Node.js** | ✅ (Buffer.fill) | ✅ | Medium (V8) | **Medium** |
| **Ruby** | ✅ (mutable String) | ✅ | Medium | **Medium** |
| **Dart** | ⚠️ (VM copies) | ✅ | Medium | **Medium** |
| **Python** | ❌ (bytes immutable) | ✅ (ctypes buf) | High | **High** |
| **PHP** | ❌ (string immutable) | ✅ | High | **High** |
| **React Native** | ❌ (JS string) | ✅ | High | **High** |

---

## 4. Library-Side Guarantees (Universal)

Regardless of language, the C library guarantees:

1. **No key retention**: Private keys are NEVER stored in `ufsecp_ctx`. They are used for the duration of the call and the library's stack frame is zeroed.
2. **Nonce zeroization**: Signing nonces (`k` values) are always zeroed after use via `memset_explicit`.
3. **Scratch zeroization**: `ufsecp_ctx_destroy()` zeros all internal scratch buffers.
4. **No key in diagnostics**: Error messages never contain key material.
5. **Constant-time**: All operations on secret data run in constant time (see CT report).

---

## 5. Recommendations

### For Library Consumers

1. **Always zero private key material after use** — use language-appropriate pattern from §2
2. **Prefer stack allocation** (Rust, Go, C#/Span) over heap for key material
3. **Do not log key material** — not even in debug mode
4. **Use OS secure memory** (`mlock`, `VirtualLock`) for long-lived keys
5. **Single-use context pattern** — create → use → destroy for paranoid security

### For Binding Authors

1. **Zero all temporary native buffers** after FFI calls return
2. **Document what you CAN'T guarantee** (GC copies, interned strings, etc.)
3. **Never store keys in wrapper struct fields** — pass-through only
4. **Use `memset_explicit`/`SecureZeroMemory`** on the native side (not `memset` which may be optimized away)
