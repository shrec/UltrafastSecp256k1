# ABI Versioning Policy

> **Applies to:** `libufsecp` (the C-ABI shared/static library produced by UltrafastSecp256k1)

---

## 1. Version Numbers

UltrafastSecp256k1 uses **three-component semantic versioning** (`MAJOR.MINOR.PATCH`)
plus a separate integer **ABI version** that tracks binary compatibility independently.

| Component | File | Macro / Variable |
|-----------|------|-----------------|
| `MAJOR.MINOR.PATCH` | `VERSION.txt` | `UFSECP_VERSION_MAJOR/MINOR/PATCH` |
| ABI version | `include/ufsecp/ufsecp_version.h.in` | `UFSECP_ABI_VERSION` |
| SO version (ELF) | `cpu/CMakeLists.txt` | `SOVERSION ${PROJECT_VERSION_MAJOR}` |

**Single source of truth for the release version:** `VERSION.txt` in the repository root.
CMake reads it at configure time and propagates it to headers, `pkg-config`, and install targets.

---

## 2. Bump Rules

### MAJOR (e.g. 3 -> 4)
A **MAJOR** bump indicates an ABI-incompatible change. Consumers **must** recompile.

Triggers:
- Removal or rename of any `UFSECP_API` function
- Change of function signature (parameter types, return type)
- Change of opaque struct layout that affects allocation (`ufsecp_ctx_size()`)
- Change of error-code semantics for existing codes

Actions on MAJOR bump:
- Increment `UFSECP_ABI_VERSION` in `ufsecp_version.h.in`
- Increment `SOVERSION` in CMake (`PROJECT_VERSION_MAJOR` tracks this automatically)
- Document the breaking changes in `CHANGELOG.md` under **[!] Breaking**
- Add a migration note in `CHANGELOG.md`

### MINOR (e.g. 3.14 -> 3.15)
A **MINOR** bump adds functionality in a backwards-compatible manner. Existing consumers
continue to work **without** recompilation if they only use previously existing symbols.

Triggers:
- New `UFSECP_API` functions added
- New error codes added (existing codes unchanged)
- New optional feature flags

Actions on MINOR bump:
- Do **not** change `UFSECP_ABI_VERSION`
- Do **not** change `SOVERSION`
- Document new API in `CHANGELOG.md` under **Added**

### PATCH (e.g. 3.14.0 -> 3.14.1)
A **PATCH** bump is a backwards-compatible bug fix. No API surface changes.

Triggers:
- Correctness fix in existing functions
- Performance improvements (same inputs -> same outputs)
- Documentation / CI fixes

Actions on PATCH bump:
- Do **not** change `UFSECP_ABI_VERSION`
- Document in `CHANGELOG.md` under **Fixed**

---

## 3. ABI Version (`UFSECP_ABI_VERSION`)

The ABI version is a monotonically increasing integer that **only** increments on
binary-incompatible changes. It is independent of the release version:

```
UFSECP_ABI_VERSION = 1    // since initial stable release
```

### Runtime Check

Consumers can verify ABI compatibility at load time:

```c
#include <ufsecp/ufsecp_version.h>

assert(ufsecp_abi_version() == UFSECP_ABI_VERSION);
```

The runtime function `ufsecp_abi_version()` returns the ABI version compiled into the
library binary. Comparing it with the compile-time macro detects mismatched headers/library
pairs.

### Packed Version Check

For minimum-version guards:

```c
// Require at least v3.14.0
#if UFSECP_VERSION_PACKED < 0x030E00
  #error "UltrafastSecp256k1 >= 3.14.0 required"
#endif
```

Or at runtime:

```c
if (ufsecp_version() < 0x030E00) {
    fprintf(stderr, "Upgrade libufsecp to >= 3.14.0\n");
    exit(1);
}
```

---

## 4. Shared Library Naming (ELF / Linux)

```
libfastsecp256k1.so               -> symlink to current
libfastsecp256k1.so.3             -> SOVERSION (= MAJOR)
libfastsecp256k1.so.3.14.0        -> full version
```

CMake sets this via:
```cmake
set_target_properties(fastsecp256k1 PROPERTIES
    VERSION   ${PROJECT_VERSION}        # e.g. 3.14.0
    SOVERSION ${PROJECT_VERSION_MAJOR}  # e.g. 3
)
```

The `SOVERSION` changes **only** on MAJOR bumps, allowing the dynamic linker to
resolve compatible versions automatically.

### Windows (MSVC / Clang-CL)

On Windows, DLL versioning uses the PE version resource. The DLL name includes the
ABI version: `fastsecp256k1-3.dll`. Import library: `fastsecp256k1.lib`.

### macOS

```
libfastsecp256k1.dylib              -> symlink
libfastsecp256k1.3.dylib            -> compatibility version
libfastsecp256k1.3.14.0.dylib       -> current version
```

---

## 5. Stability Guarantees

### Stable ABI Surface

All functions declared with `UFSECP_API` in `include/ufsecp/ufsecp.h` are part of
the stable ABI. Currently 41 functions across these categories:

| Category | Count | Functions |
|----------|-------|-----------|
| Context | 5 | `ctx_create`, `ctx_clone`, `ctx_destroy`, `last_error`, `last_error_msg`, `ctx_size` |
| Secret Key | 4 | `seckey_verify`, `seckey_negate`, `seckey_tweak_add`, `seckey_tweak_mul` |
| Public Key | 4 | `pubkey_create`, `pubkey_create_uncompressed`, `pubkey_parse`, `pubkey_xonly` |
| ECDSA | 4 | `ecdsa_sign`, `ecdsa_verify`, `ecdsa_sig_to_der`, `ecdsa_sig_from_der` |
| Recovery | 2 | `ecdsa_sign_recoverable`, `ecdsa_recover` |
| Schnorr | 2 | `schnorr_sign`, `schnorr_verify` |
| ECDH | 1 | `ecdh` |
| BIP32 | 4 | `bip32_master_key`, `bip32_derive_child`, `bip32_pubkey`, `bip32_serialize` |
| Address | 4 | `address_p2pkh`, `address_p2sh_p2wpkh`, `address_p2wpkh`, `address_p2tr` |
| Hash | 5 | `sha256`, `sha256_init`, `sha256_update`, `sha256_finalize`, `hash160` |
| Version | 3 | `version`, `abi_version`, `version_string` |
| Selftest | 3 | `selftest_run`, `selftest_report`, `selftest_passed` |

### Unstable / Internal

- C++ headers under `cpu/include/secp256k1/` (`secp256k1::fast::*`, `secp256k1::ct::*`)
  are **internal** and may change without notice between any versions.
- Struct internals of opaque types (`ufsecp_ctx`) are not part of the ABI.
- Functions not marked `UFSECP_API` may be removed or changed freely.

---

## 6. Deprecation Process

1. Mark the function with `UFSECP_DEPRECATED` attribute (one MINOR release warning period).
2. Document deprecation in `CHANGELOG.md` under **Deprecated** with migration guidance.
3. Remove the function in the next MAJOR release.

```c
// Example:
UFSECP_API UFSECP_DEPRECATED("use ufsecp_pubkey_create() instead")
ufsecp_error_t ufsecp_ec_pubkey_create(ufsecp_ctx* ctx, ...);
```

Minimum deprecation window: **one MINOR release cycle**.

---

## 7. pkg-config Integration

The installed `ufsecp.pc` (generated by CMake) exposes:

```
prefix=/usr/local
libdir=${prefix}/lib
includedir=${prefix}/include

Name: libufsecp
Description: Ultra high-performance secp256k1 elliptic curve cryptography
Version: 3.14.0
Libs: -L${libdir} -lfastsecp256k1
Cflags: -I${includedir}
```

Consumers should use:
```bash
pkg-config --modversion ufsecp   # -> 3.14.0
pkg-config --libs ufsecp         # -> -L/usr/local/lib -lfastsecp256k1
```

---

## 8. Binding Compatibility Matrix

| Binding | Minimum ABI | Notes |
|---------|-------------|-------|
| Python (ctypes) | 1 | Full 41-fn coverage |
| Rust (FFI) | 1 | Full 41-fn coverage |
| Go (CGo) | 1 | Full 41-fn coverage |
| C# (P/Invoke) | 1 | Full 41-fn coverage |
| Java (JNI) | 1 | Full 41-fn coverage |
| Swift | 1 | Full 41-fn coverage |
| Dart (FFI) | 1 | Full 41-fn coverage |
| React Native | 1 | Full 41-fn coverage |
| Node.js (NAPI) | 1 | Full 41-fn coverage |
| Node.js (WASM) | 1 | Full 41-fn coverage |
| Ruby (FFI) | 1 | Full 41-fn coverage |
| Kotlin (JNI) | 1 | Full 41-fn coverage |

All bindings target `UFSECP_ABI_VERSION >= 1`. When ABI version bumps, binding
maintainers update their minimum version requirement and adjust any changed signatures.

---

## 9. Checklist for Releases

- [ ] `VERSION.txt` updated
- [ ] `CHANGELOG.md` has entry for new version
- [ ] If ABI-breaking: `UFSECP_ABI_VERSION` incremented in `ufsecp_version.h.in`
- [ ] If ABI-breaking: migration notes in `CHANGELOG.md`
- [ ] `git tag -a v<version>` on release commit
- [ ] CI passes (all CTest targets + cross-library differential)
- [ ] Shared library SOVERSION correct (`readelf -d` / `otool -L`)
