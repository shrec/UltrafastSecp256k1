# UltrafastSecp256k1 v3.14.0 -- Full Language Binding Coverage

**Release Date**: 2026-02-25
**Tag**: `v3.14.0`
**Commits since v3.13.1**: 4

---

## Highlights

### 🔗 12 Language Bindings -- Full 41-Function C API Parity

All 12 officially supported language bindings now cover the complete `ufsecp` C API (41 exported functions):

| Language | New Functions | Status |
|----------|:---:|--------|
| **Java** | +22 JNI + 3 helper classes | [OK] Complete |
| **Swift** | +20 | [OK] Complete |
| **React Native** | +15 | [OK] Complete |
| **Python** | +3 | [OK] Complete |
| **Rust** | +2 | [OK] Complete |
| **Dart** | +1 | [OK] Complete |
| **Go** | -- | [OK] Already complete |
| **Node.js** | -- | [OK] Already complete |
| **C#** | -- | [OK] Already complete |
| **Ruby** | -- | [OK] Already complete |
| **PHP** | -- | [OK] Already complete |
| **C API** | -- | [OK] Reference implementation |

### Java Details
- 22 new JNI functions covering: DER encode/decode, recoverable signing, ECDH, Schnorr (BIP-340), BIP-32 HD derivation, BIP-39 mnemonic, taproot key generation, WIF encode/decode, address encoding, tagged hash
- 3 new helper classes: `RecoverableSignature`, `WifDecoded`, `TaprootOutputKeyResult`

### Swift Details
- 20 new functions: DER serialization, recovery signatures, ECDH shared secret, tagged hashing, BIP-32/39, taproot output key, WIF handling, base58/bech32 address encoding

### React Native Details
- 15 new functions bridged through the JS layer for mobile DApp development

### 📚 9 New Binding READMEs
Comprehensive documentation added for: `c_api`, `dart`, `go`, `java`, `php`, `python`, `ruby`, `rust`, `swift` -- each with API reference, build instructions, and usage examples.

### 📦 Package Naming Cleanup
All documentation and packaging files now reference the correct library names:
- **Shared library**: `libufsecp.so` / `ufsecp.dll` / `libufsecp.dylib`
- **Static library**: `libfastsecp256k1.a`
- **Debian**: `libufsecp3` / `libufsecp-dev`
- **RPM**: `libufsecp` / `libufsecp-devel`
- **Arch**: `libufsecp`
- **CMake**: `find_package(secp256k1-fast)` -> `secp256k1::fast`
- **pkg-config**: `pkg-config --libs secp256k1-fast` -> `-lfastsecp256k1`

### 🏗 Selftest Report API (Foundation)
- `SelftestReport` and `SelftestCase` structs added to `selftest.hpp`
- `tally()` refactored for programmatic access to test results
- Function bodies (`selftest_report()`, `to_text()`, `to_json()`) planned for next release

---

## CI / Build Fixes
- `[[maybe_unused]]` on `get_platform_string()` -- eliminates `-Werror=unused-function` in release builds
- `Dockerfile.local-ci` -- `ubuntu:24.04` pinned by SHA digest (Scorecard compliance)

---

## Files Changed
- **38 files changed**, +1,579 insertions, -108 deletions
- **22 binding files** modified/created
- **13 documentation/packaging files** corrected

## Verification
```bash
cmake -S . -B build_rel -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build_rel -j
ctest --test-dir build_rel --output-on-failure
```

## Upgrade Notes
- **No breaking changes** -- drop-in upgrade from v3.13.x
- **SOVERSION unchanged** -- remains `3` (`libufsecp.so.3`)
- **ABI compatible** -- no changes to C API function signatures
- Binding code additions are pure additions; existing binding users unaffected

---

**Full Changelog**: [`v3.13.1...v3.14.0`](https://github.com/shrec/UltrafastSecp256k1/compare/v3.13.1...v3.14.0)
