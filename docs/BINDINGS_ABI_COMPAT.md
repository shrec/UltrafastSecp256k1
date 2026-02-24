# ABI Compatibility & Version Gates
## UltrafastSecp256k1 Cross-Language ABI Contract

> **Canonical header**: `include/ufsecp/ufsecp_version.h`  
> **Current ABI version**: 1

---

## 1. Versioning Scheme

### Semantic Versioning (Library)

| Component | Meaning | ABI Impact |
|---|---|---|
| `MAJOR` bump | Breaking struct layout / removed functions | ABI-incompatible |
| `MINOR` bump | New functions added (existing untouched) | ABI-compatible |
| `PATCH` bump | Bug fixes only | ABI-compatible |

### ABI Version (Binary Compatibility Tag)

`UFSECP_ABI_VERSION` is an integer incremented **ONLY** when binary-incompatible changes occur:
- Struct layout changes
- Function signature changes
- Function removals
- Enum value reordering

It does NOT increment for:
- New function additions
- Bug fixes
- Performance improvements

### Runtime Query Functions

```c
unsigned int ufsecp_version(void);         // Packed: (major<<16)|(minor<<8)|patch
unsigned int ufsecp_abi_version(void);     // ABI integer (currently 1)
const char*  ufsecp_version_string(void);  // Human-readable, e.g. "3.14.0"
```

---

## 2. Wrapper ABI Gate Contract

**Every binding wrapper MUST** check ABI compatibility at initialization time:

### Required Behavior

```
On context creation:
  1. Call ufsecp_abi_version()
  2. Compare against EXPECTED_ABI (compiled-in constant)
  3. If mismatch → fail with clear error message
  4. If match → proceed normally
```

### Mismatch Error Format

```
UfsecpError: ABI version mismatch. Wrapper expects ABI 1, library reports ABI 2.
Please update the wrapper to match the installed native library.
```

---

## 3. Per-Language ABI Gate Implementation

### 3.1 Python

```python
class Ufsecp:
    EXPECTED_ABI = 1
    
    def __init__(self):
        self._ctx = _lib.ufsecp_ctx_create()
        abi = _lib.ufsecp_abi_version()
        if abi != self.EXPECTED_ABI:
            raise UfsecpError("init", -1, 
                f"ABI mismatch: wrapper expects {self.EXPECTED_ABI}, lib reports {abi}")
```

### 3.2 Node.js

```javascript
class UfsecpContext {
    static EXPECTED_ABI = 1;
    
    constructor() {
        const abi = lib.ufsecp_abi_version();
        if (abi !== UfsecpContext.EXPECTED_ABI) {
            throw new UfsecpError('init', -1,
                `ABI mismatch: wrapper expects ${UfsecpContext.EXPECTED_ABI}, lib reports ${abi}`);
        }
    }
}
```

### 3.3 C# 

```csharp
public class Ufsecp : IDisposable {
    private const int ExpectedAbi = 1;
    
    public Ufsecp() {
        var abi = NativeMethods.ufsecp_abi_version();
        if (abi != ExpectedAbi)
            throw new UfsecpException("init", 
                $"ABI mismatch: wrapper expects {ExpectedAbi}, lib reports {abi}");
    }
}
```

### 3.4 Java

```java
public class Ufsecp implements AutoCloseable {
    private static final int EXPECTED_ABI = 1;
    
    public Ufsecp() {
        int abi = nativeAbiVersion();
        if (abi != EXPECTED_ABI)
            throw new IllegalStateException(
                "ABI mismatch: wrapper expects " + EXPECTED_ABI + ", lib reports " + abi);
    }
}
```

### 3.5 Swift

```swift
class UfsecpContext {
    static let expectedABI: UInt32 = 1
    
    init() throws {
        let abi = ufsecp_abi_version()
        guard abi == Self.expectedABI else {
            throw UfsecpError(operation: "init", 
                code: .internal, 
                message: "ABI mismatch: wrapper expects \(Self.expectedABI), lib reports \(abi)")
        }
    }
}
```

### 3.6 Go

```go
const expectedABI = 1

func NewContext() (*Context, error) {
    abi := C.ufsecp_abi_version()
    if int(abi) != expectedABI {
        return nil, fmt.Errorf("ufsecp: ABI mismatch: wrapper expects %d, lib reports %d",
            expectedABI, abi)
    }
    // ...
}
```

### 3.7 Rust

```rust
const EXPECTED_ABI: u32 = 1;

impl Context {
    pub fn new() -> Result<Self> {
        let abi = unsafe { ffi::ufsecp_abi_version() };
        if abi != EXPECTED_ABI {
            return Err(Error { 
                op: "init", 
                code: ErrorCode::Internal,
            });
        }
        // ...
    }
}
```

### 3.8 Dart

```dart
class UfsecpContext {
    static const int _expectedAbi = 1;
    
    UfsecpContext() {
        final abi = _bindings.ufsecp_abi_version();
        if (abi != _expectedAbi) {
            throw UfsecpException('init',
                UfsecpError.internal,
                'ABI mismatch: wrapper expects $_expectedAbi, lib reports $abi');
        }
    }
}
```

---

## 4. Version Compatibility Matrix

| Wrapper Version | Min Library ABI | Max Library ABI | Notes |
|---|:---:|:---:|---|
| 1.x.x | 1 | 1 | Current |
| 2.x.x | 2 | 2 | Future (when ABI breaks) |

### Forward/Backward Compatibility Rules

| Scenario | Behavior |
|---|---|
| Wrapper ABI == Library ABI | ✅ Full compatibility |
| Wrapper ABI < Library ABI | ❌ Wrapper too old — must upgrade wrapper |
| Wrapper ABI > Library ABI | ❌ Library too old — must upgrade library |

---

## 5. Wrapper Version ↔ Library Version Mapping

Each wrapper release pins to a specific minimum library version:

| Wrapper Release | Min Library Version | ABI Required |
|---|---|:---:|
| All current (v1.x) | 3.14.0+ | 1 |

### How Wrapper Versioning Works

- Wrapper version follows its own semver (e.g., npm `ufsecp@1.2.0`)
- Wrapper version is **independent** of library version
- The ONLY coupling is `EXPECTED_ABI` constant
- When ABI breaks (rare), ALL wrappers must be updated simultaneously

---

## 6. Release Checklist (ABI Changes)

When an ABI-breaking change is required:

- [ ] Increment `UFSECP_ABI_VERSION` in `ufsecp_version.h.in`
- [ ] Bump `UFSECP_VERSION_MAJOR`
- [ ] Update `EXPECTED_ABI` in ALL 11 binding wrappers
- [ ] Update this document's compatibility matrix
- [ ] Release note: "⚠️ ABI BREAK — all wrapper packages must be updated"
- [ ] Tag release with `abi-N` label

---

## 7. Smoke Test Verification

Each binding's smoke test verifies ABI compatibility:

```
Test: ctx_create_abi
  - Creates context successfully
  - Reads abi_version() ≥ 1
  - Confirms no ABI mismatch exception
```

This test is the first test in every language's smoke suite (see `bindings/<lang>/tests/smoke_test.*`).
