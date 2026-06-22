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
binary-incompatible changes. It equals `PROJECT_VERSION_MAJOR` and is propagated
automatically by CMake into `include/ufsecp/ufsecp_version.h.in`:

```
UFSECP_ABI_VERSION = 4    // equals PROJECT_VERSION_MAJOR; increments on every breaking release
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
// Require at least v3.14.0 (example using hypothetical v3.x naming — current MAJOR is 4)
#if UFSECP_VERSION_PACKED < 0x030E00
  #error "UltrafastSecp256k1 >= 3.14.0 required"
#endif
```

Or at runtime:

```c
// Example using hypothetical v3.x version — current MAJOR is 4 (SONAME 4)
if (ufsecp_version() < 0x030E00) {
    fprintf(stderr, "Upgrade libufsecp to >= 3.14.0\n");
    exit(1);
}
```

---

## 4. Shared Library Naming (ELF / Linux)

```
libfastsecp256k1.so               -> symlink to current
libfastsecp256k1.so.4             -> SOVERSION (= MAJOR; current MAJOR is 4)
```

> **Note:** The examples below use hypothetical version numbers for illustration.
> Current MAJOR = 4; SONAME = `libfastsecp256k1.so.4`.

```
libfastsecp256k1.so               -> symlink to current
libfastsecp256k1.so.3             -> example SOVERSION (MAJOR = 3, hypothetical)
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
the stable ABI. There are **163** such functions (the authoritative list is every
`UFSECP_API`-marked declaration in the header; counts and example names below are
verified against it by `ci/check_abi_count.py`). Functions carry the `ufsecp_`
prefix in the header (e.g. `ufsecp_ctx_create`); the short names are used here.

| Category | Count | Example functions |
|----------|-------|-----------|
| Context & lifecycle | 7 | `ctx_create`, `ctx_clone`, `ctx_destroy`, `ctx_size`, `context_randomize`, `last_error`, `last_error_msg` |
| Secret key | 4 | `seckey_verify`, `seckey_negate`, `seckey_tweak_add`, `seckey_tweak_mul` |
| Public key | 9 | `pubkey_create`, `pubkey_create_uncompressed`, `pubkey_parse`, `pubkey_xonly`, `pubkey_add`, `pubkey_negate`, `pubkey_tweak_add`, `pubkey_tweak_mul`, `pubkey_combine` |
| ECDSA (sign/verify/recover/batch/adaptor) | 15 | `ecdsa_sign`, `ecdsa_verify`, `ecdsa_sig_to_der`, `ecdsa_sig_from_der`, `ecdsa_sign_recoverable`, `ecdsa_recover`, `ecdsa_sign_batch`, `ecdsa_batch_verify`, `ecdsa_batch_verify_mt`, `ecdsa_verify_opaque_rows`, `ecdsa_verify_opaque_rows_mt`, `ecdsa_adaptor_sign`, `ecdsa_adaptor_adapt` |
| Schnorr (sign/verify/batch/adaptor/msg) | 13 | `schnorr_sign`, `schnorr_verify`, `schnorr_sign_batch`, `schnorr_batch_verify`, `schnorr_batch_verify_mt`, `schnorr_adaptor_sign`, `schnorr_sign_msg`, `schnorr_verify_msg` |
| ECDH | 3 | `ecdh`, `ecdh_xonly`, `ecdh_raw` |
| MuSig2 | 8 | `musig2_key_agg`, `musig2_nonce_gen`, `musig2_nonce_agg`, `musig2_partial_sign`, `musig2_partial_sign_v2`, `musig2_partial_verify`, `musig2_partial_sig_agg` |
| FROST | 6 | `frost_keygen_begin`, `frost_keygen_finalize`, `frost_sign_nonce_gen`, `frost_sign`, `frost_verify_partial`, `frost_aggregate` |
| Taproot & Tapscript | 5 | `taproot_output_key`, `taproot_tweak_seckey`, `taproot_verify`, `taproot_keypath_sighash`, `tapscript_sighash` |
| BIP32 / BIP39 / BIP85 derivation | 11 | `bip32_master`, `bip32_derive`, `bip32_derive_path`, `bip32_pubkey`, `bip39_generate`, `bip39_to_seed`, `bip85_entropy` |
| BIP143 / BIP144 sighash & txid | 5 | `bip143_sighash`, `bip143_p2wpkh_script_code`, `bip144_txid`, `bip144_wtxid`, `bip144_witness_commitment` |
| BIP322 / BTC / Ethereum message | 10 | `bip322_sign`, `bip322_verify`, `btc_message_sign`, `btc_message_verify`, `eth_sign`, `eth_ecrecover`, `eth_address` |
| BIP324 v2 transport | 5 | `bip324_create`, `bip324_handshake`, `bip324_encrypt`, `bip324_decrypt`, `bip324_destroy` |
| Addresses / SegWit / descriptors / WIF | 18 | `addr_p2pkh`, `addr_p2wpkh`, `addr_p2tr`, `segwit_p2wpkh_spk`, `segwit_p2tr_spk`, `descriptor_parse`, `wif_encode`, `wif_decode`, `coin_address` |
| Hashing | 5 | `sha256`, `sha512`, `hash160`, `keccak256`, `tagged_hash` |
| AEAD & ECIES | 4 | `aead_chacha20_poly1305_encrypt`, `aead_chacha20_poly1305_decrypt`, `ecies_encrypt`, `ecies_decrypt` |
| ellswift (BIP324 encoding) | 2 | `ellswift_create`, `ellswift_xdh` |
| Pedersen / ZK / GCS / Silent Payments / PSBT / misc | 25 | `pedersen_commit`, `zk_dleq_prove`, `zk_range_prove`, `gcs_build`, `gcs_match`, `silent_payment_scan`, `psbt_sign_taproot`, `shamir_trick`, `multi_scalar_mul` |

### Unstable / Internal

- C++ headers under `src/cpu/include/secp256k1/` (`secp256k1::fast::*`, `secp256k1::ct::*`)
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

A standard root install ships `secp256k1-fast.pc` (generated by CMake from
`secp256k1-fast.pc.in`):

```
prefix=/usr/local
exec_prefix=${prefix}
libdir=${exec_prefix}/lib
includedir=${prefix}/include

Name: secp256k1-fast
Description: High-performance secp256k1 elliptic curve cryptography library
Version: 4.4.0
Cflags: -I${includedir}
Libs: -L${libdir} -lufsecp
Libs.private: -lpthread
```

Consumers should use:
```bash
pkg-config --modversion secp256k1-fast   # -> 4.1.0
pkg-config --libs secp256k1-fast          # -> -L/usr/local/lib -lufsecp
```

A second module, `ufsecp.pc` (`Name: ufsecp`, same `-lufsecp`), is generated from
`include/ufsecp/ufsecp.pc.in` for the stable C ABI; use whichever module name your
build installs (`secp256k1-fast` on the standard root build).

---

## 8. Binding Compatibility Matrix

| Binding | Minimum ABI | Notes |
|---------|-------------|-------|
| Python (ctypes) | 4 | Targets ABI v4 (163 stable C functions) |
| Rust (FFI) | 4 | Targets ABI v4 (163 stable C functions) |
| Go (CGo) | 4 | Targets ABI v4 (163 stable C functions) |
| C# (P/Invoke) | 4 | Targets ABI v4 (163 stable C functions) |
| Java (JNI) | 4 | Targets ABI v4 (163 stable C functions) |
| Swift | 4 | Targets ABI v4 (163 stable C functions) |
| Dart (FFI) | 4 | Targets ABI v4 (163 stable C functions) |
| React Native | 4 | Targets ABI v4 (163 stable C functions) |
| Node.js (NAPI) | 4 | Targets ABI v4 (163 stable C functions) |
| Node.js (WASM) | 4 | Targets ABI v4 (163 stable C functions) |
| Ruby (FFI) | 4 | Targets ABI v4 (163 stable C functions) |
| Kotlin (JNI) | 4 | Targets ABI v4 (163 stable C functions) |

All bindings target `UFSECP_ABI_VERSION == 4` (the current MAJOR); every binding's
`EXPECTED_ABI` constant is gated to 4 by `ci/check_abi_version_sync.py` (the fix for
REL-ABI-VERSION-MISMATCH-001, where bindings hardcoded ABI 1 against an ABI 4 library).
When the ABI MAJOR bumps, binding maintainers update `EXPECTED_ABI` and adjust any changed signatures.

---

## 9. Checklist for Releases

- [ ] `VERSION.txt` updated
- [ ] `CHANGELOG.md` has entry for new version
- [ ] If ABI-breaking: `UFSECP_ABI_VERSION` incremented in `ufsecp_version.h.in`
- [ ] If ABI-breaking: migration notes in `CHANGELOG.md`
- [ ] `git tag -a v<version>` on release commit
- [ ] CI passes (all CTest targets + cross-library differential)
- [ ] Shared library SOVERSION correct (`readelf -d` / `otool -L`)
