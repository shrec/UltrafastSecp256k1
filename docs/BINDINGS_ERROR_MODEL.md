# Bindings Error Model
## UltrafastSecp256k1 Cross-Language Error Contract

> **Canonical source**: `include/ufsecp/ufsecp_error.h`  
> **ABI version**: 1+  
> **Last updated**: 2025-01-XX

---

## 1. C ABI Error Codes (Single Source of Truth)

Every `ufsecp_*` function returns `ufsecp_error_t` (`int`, 0 = success).

| Code | Name | Meaning | Recoverable? |
|---:|------|---------|:---:|
| 0 | `UFSECP_OK` | Success | — |
| 1 | `UFSECP_ERR_NULL_ARG` | Required pointer was NULL | Yes |
| 2 | `UFSECP_ERR_BAD_KEY` | Invalid private key (zero, ≥ order) | Yes |
| 3 | `UFSECP_ERR_BAD_PUBKEY` | Unparseable / invalid public key | Yes |
| 4 | `UFSECP_ERR_BAD_SIG` | Malformed signature | Yes |
| 5 | `UFSECP_ERR_BAD_INPUT` | Wrong length, bad format | Yes |
| 6 | `UFSECP_ERR_VERIFY_FAIL` | Signature verification failed | Yes |
| 7 | `UFSECP_ERR_ARITH` | Scalar/field arithmetic overflow | Yes |
| 8 | `UFSECP_ERR_SELFTEST` | Library self-test failed on init | **Fatal** |
| 9 | `UFSECP_ERR_INTERNAL` | Unexpected internal error | **Fatal** |
| 10 | `UFSECP_ERR_BUF_TOO_SMALL` | Output buffer too small | Yes |

### Recoverable vs Fatal

- **Recoverable** (codes 1-7, 10): Invalid caller input. The context remains valid. Caller should fix the input and retry.
- **Fatal** (codes 8-9): Library integrity compromised. Context should be destroyed. Do not retry.

### Per-Context Diagnostics

```c
ufsecp_error_t  ufsecp_last_error(const ufsecp_ctx* ctx);
const char*     ufsecp_last_error_msg(const ufsecp_ctx* ctx);
const char*     ufsecp_error_str(ufsecp_error_t err);   // Static lookup
```

Each `ufsecp_ctx` owns its own last-error slot. Thread safety: one context per thread (no locking).

---

## 2. Language-Specific Error Mapping

### 2.1 Python (`ufsecp/__init__.py`)

| Error Pattern | Mechanism |
|---|---|
| Recoverable (1-7, 10) | Raises `UfsecpError(op, code, msg)` |
| Fatal (8-9) | Raises `UfsecpError` — context should not be reused |
| Verify failure (code 6) | Returns `False` (does **not** throw) |
| Input validation | `ValueError` / `TypeError` before FFI call |

```python
class UfsecpError(Exception):
    def __init__(self, operation: str, code: int, message: str): ...
```

### 2.2 Node.js (`lib/ufsecp.js`)

| Error Pattern | Mechanism |
|---|---|
| Recoverable (1-7, 10) | Throws `UfsecpError { operation, code, message }` |
| Fatal (8-9) | Throws `UfsecpError` |
| Verify failure (code 6) | Returns `false` |
| Input validation | `TypeError` before FFI call |

```javascript
class UfsecpError extends Error {
    constructor(operation, code, message) { ... }
}
```

### 2.3 C# (`Ufsecp/Ufsecp.cs`)

| Error Pattern | Mechanism |
|---|---|
| Recoverable (1-7, 10) | Throws `UfsecpException(op, code, msg)` |
| Fatal (8-9) | Throws `UfsecpException` |
| Verify failure (code 6) | Returns `false` |
| Input validation | `ArgumentException` / `ArgumentNullException` |
| Context destroyed | `ObjectDisposedException` |

```csharp
public class UfsecpException : Exception {
    public string Operation { get; }
    public int Code { get; }
}
```

### 2.4 Java (`com.ultrafast.ufsecp.Ufsecp`)

| Error Pattern | Mechanism |
|---|---|
| Recoverable (1-7, 10) | Returns `null` + sets `lastError()` / `lastErrorMsg()` |
| Fatal (8-9) | Returns `null` + sets `lastError()` |
| Verify failure (code 6) | Returns `false` |
| Context destroyed | Throws `IllegalStateException` |
| Input validation | `IllegalArgumentException` |

> **Java design note**: JNI overhead makes exception creation expensive. The null-return + last-error pattern allows callers to check errors only when needed.

```java
// Post-call error check
byte[] pub = ctx.pubkeyCreate(privkey);
if (pub == null) {
    int code = ctx.lastError();        // UFSECP_ERR_BAD_KEY
    String msg = ctx.lastErrorMsg();   // "Invalid private key"
}
```

### 2.5 Swift (`Ufsecp/UfsecpContext`)

| Error Pattern | Mechanism |
|---|---|
| Recoverable (1-7, 10) | Throws `UfsecpError(operation:, code:)` |
| Fatal (8-9) | Throws `UfsecpError` |
| Verify failure (code 6) | Returns `false` (no throw) |
| Input validation | `precondition` / `UfsecpError` with `.badInput` |

```swift
struct UfsecpError: Error {
    let operation: String
    let code: UfsecpErrorCode  // enum: .ok, .nullArg, .badKey, ...
}
```

### 2.6 Go (`ufsecp.Context`)

| Error Pattern | Mechanism |
|---|---|
| Recoverable (1-7, 10) | Returns `(zero, ErrXxx)` sentinel error |
| Fatal (8-9) | Returns `(zero, ErrInternal)` / `(zero, ErrSelftest)` |
| Verify failure (code 6) | Returns `error` (non-nil) |
| Input validation | Returns `ErrNullArg` / `ErrBadInput` |

```go
var (
    ErrNullArg    = errors.New("ufsecp: null argument")
    ErrBadKey     = errors.New("ufsecp: invalid private key")
    ErrBadPubkey  = errors.New("ufsecp: invalid public key")
    ErrBadSig     = errors.New("ufsecp: invalid signature")
    ErrBadInput   = errors.New("ufsecp: bad input")
    ErrVerifyFail = errors.New("ufsecp: verification failed")
    ErrArith      = errors.New("ufsecp: arithmetic overflow")
    ErrSelftest   = errors.New("ufsecp: self-test failed")
    ErrInternal   = errors.New("ufsecp: internal error")
    ErrBufSmall   = errors.New("ufsecp: buffer too small")
)
```

> **Go design note**: Go idiom is `(value, error)` returns. The `error` interface is used uniformly. Callers use `errors.Is(err, ufsecp.ErrBadKey)` for matching.

### 2.7 Rust (`ufsecp::Context`)

| Error Pattern | Mechanism |
|---|---|
| Recoverable (1-7, 10) | Returns `Err(Error { op, code })` |
| Fatal (8-9) | Returns `Err(Error { op, code })` |
| Verify failure (code 6) | Returns `false` (no `Result`) |
| Input validation | Compile-time via `&[u8; 32]` fixed-size slices |

```rust
pub struct Error {
    pub op: &'static str,
    pub code: ErrorCode,
}

pub enum ErrorCode {
    NullArg, BadKey, BadPubkey, BadSig, BadInput,
    VerifyFail, Arith, Selftest, Internal, BufTooSmall,
    Unknown(i32),
}
```

### 2.8 Dart (`UfsecpContext`)

| Error Pattern | Mechanism |
|---|---|
| Recoverable (1-7, 10) | Throws `UfsecpException(operation, error)` |
| Fatal (8-9) | Throws `UfsecpException` |
| Verify failure (code 6) | Returns `false` |
| Input validation | Throws `ArgumentError` for size mismatches |

```dart
class UfsecpException implements Exception {
    final String operation;
    final UfsecpError error;  // enum values matching C codes
}
```

### 2.9 PHP (`Ultrafast\Ufsecp\Ufsecp`)

| Error Pattern | Mechanism |
|---|---|
| Recoverable (1-7, 10) | Throws `RuntimeException` |
| Fatal (8-9) | Throws `RuntimeException` |
| Verify failure (code 6) | Returns `false` |
| Input validation | Throws `InvalidArgumentException` |

### 2.10 Ruby (`Ufsecp::Context`)

| Error Pattern | Mechanism |
|---|---|
| Recoverable (1-7, 10) | Raises `Ufsecp::Error` (with `code`, `operation`) |
| Fatal (8-9) | Raises `Ufsecp::Error` |
| Verify failure (code 6) | Returns `false` |
| Input validation | `ArgumentError` |

### 2.11 React Native (`UfsecpContext`)

| Error Pattern | Mechanism |
|---|---|
| Recoverable (1-7, 10) | Rejects Promise with `UfsecpError` |
| Fatal (8-9) | Rejects Promise with `UfsecpError` |
| Verify failure (code 6) | Resolves to `false` |
| Input validation | Throws `TypeError` synchronously |

---

## 3. Cross-Language Error Invariants

These MUST hold across ALL bindings:

| Invariant | Description |
|---|---|
| **E-1** | `ecdsaVerify` / `schnorrVerify` NEVER throw/error for invalid signatures; they return `false` |
| **E-2** | Zero private key (32 zero bytes) MUST produce error code 2 (`BAD_KEY`) |
| **E-3** | NULL/nil context access MUST produce immediate error (not segfault) |
| **E-4** | Error messages are English-only, short, deterministic |
| **E-5** | Context remains valid after any recoverable error (codes 1-7, 10) |
| **E-6** | Fatal errors (8-9) MUST be propagated immediately — no silent fallback |
| **E-7** | All errors include the operation name for diagnostics |

---

## 4. Error Classification Matrix

| Operation Category | Possible Errors | Notes |
|---|---|---|
| `ctx_create` / `ctx_destroy` | 8 (SELFTEST), 9 (INTERNAL) | Fatal if selftest fails |
| `seckey_verify` | 2 (BAD_KEY) | |
| `pubkey_create` / `pubkey_xonly` | 1, 2 | |
| `ecdsa_sign` / `schnorr_sign` | 1, 2, 9 | |
| `ecdsa_verify` / `schnorr_verify` | Returns false on failure | Never errors, never throws |
| `ecdsa_sign_recoverable` | 1, 2, 9 | |
| `ecdsa_recover` | 1, 4, 5 | |
| `ecdsa_der_*` | 1, 4, 5, 10 | |
| `sha256` / `hash160` | 1 | |
| `addr_*` | 1, 3, 5 | |
| `wif_encode` / `wif_decode` | 1, 2, 5 | |
| `bip32_*` | 1, 5, 7 | |
| `ecdh` | 1, 2, 3 | |
| `taproot_*` | 1, 2, 3, 5 | |

---

## 5. Testing Requirements

Each binding's smoke test MUST verify:

1. **Golden path**: valid inputs → success
2. **Error path**: zero key → error code 2 or equivalent exception
3. **Verify rejection**: mutated sig → returns `false` (not exception)
4. **Determinism**: same inputs → same error code

See `bindings/<lang>/tests/smoke_test.*` for implementations.
