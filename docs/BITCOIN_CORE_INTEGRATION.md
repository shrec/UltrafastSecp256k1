# Bitcoin Core Integration Guide

How to use UltrafastSecp256k1 as a **compile-time alternative** to bitcoin-core/secp256k1
inside Bitcoin Core's build system.

---

## TL;DR

UltrafastSecp256k1 ships a `compat/libsecp256k1_shim` that provides the **identical
`secp256k1.h` API surface** as bitcoin-core/secp256k1. No Bitcoin Core source changes
are required — only the CMake configuration changes.

---

## Architecture

```
Bitcoin Core source (unchanged)
        │
        │ #include <secp256k1.h>
        │ secp256k1_ecdsa_sign(ctx, ...)
        ▼
┌─────────────────────────────┐
│  compat/libsecp256k1_shim   │  ← thin ABI mapping layer
│  (C API, identical types)   │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  UltrafastSecp256k1 (C++)   │  ← actual implementation
│  CPU: w18/GLV/SafeGCD       │
│  GPU: CUDA / OpenCL         │
└─────────────────────────────┘
```

---

## Covered API Surface

| Header | Functions | Status |
|--------|-----------|--------|
| `secp256k1.h` | context lifecycle, pubkey parse/serialize/negate/tweak, ECDSA sign/verify/parse/normalize, DER, tagged_sha256 | ✅ Complete |
| `secp256k1_schnorrsig.h` | `schnorrsig_sign32`, `schnorrsig_verify` (BIP-340) | ✅ Complete |
| `secp256k1_extrakeys.h` | x-only pubkey, keypair, tweaks (BIP-340/341) | ✅ Complete |
| `secp256k1_recovery.h` | ECDSA recoverable signatures | ✅ Complete |
| `secp256k1_ecdh.h` | ECDH with custom hash function | ✅ Complete |
| `secp256k1_ellswift.h` | ElligatorSwift encoding/decoding (BIP-324) | ✅ Complete |

---

## How to Wire Into Bitcoin Core's CMake

Bitcoin Core (cmake/modules) uses `find_package(secp256k1)` or builds the bundled
submodule at `src/secp256k1`. Adding UltrafastSecp256k1 requires two changes:

### Option A — FetchContent replacement (recommended for CI testing)

Add to Bitcoin Core's top-level `CMakeLists.txt`:

```cmake
option(USE_ULTRAFAST_SECP256K1 "Use UltrafastSecp256k1 instead of bundled secp256k1" OFF)

if(USE_ULTRAFAST_SECP256K1)
    include(FetchContent)
    FetchContent_Declare(
        UltrafastSecp256k1
        GIT_REPOSITORY https://github.com/shrec/UltrafastSecp256k1.git
        GIT_TAG        main   # pin to a specific commit hash for reproducibility
    )
    FetchContent_MakeAvailable(UltrafastSecp256k1)

    # The shim provides identical secp256k1.h headers
    add_subdirectory(
        ${ultrafastsecp256k1_SOURCE_DIR}/compat/libsecp256k1_shim
        secp256k1_shim_build
    )
    # Create an alias that satisfies the existing secp256k1 CMake target name
    add_library(secp256k1 ALIAS secp256k1_shim)
    target_include_directories(secp256k1_shim PUBLIC
        ${ultrafastsecp256k1_SOURCE_DIR}/compat/libsecp256k1_shim/include
    )
else()
    # Default: use bundled bitcoin-core/secp256k1
    add_subdirectory(src/secp256k1)
endif()
```

Build with:
```bash
cmake -S . -B build -DUSE_ULTRAFAST_SECP256K1=ON
cmake --build build -j$(nproc)
```

### Option B — Submodule replacement

Replace the `src/secp256k1` submodule reference:
```bash
git submodule deinit src/secp256k1
git rm src/secp256k1
git submodule add https://github.com/shrec/UltrafastSecp256k1.git src/secp256k1_uf
```

Then in `CMakeLists.txt`, include the shim instead of `src/secp256k1`.

---

## Verifying Correctness

UltrafastSecp256k1 is already differential-tested against bitcoin-core/secp256k1:

```bash
# Runs 10,000+ random cross-validation cases
cmake --build build --target run_differential_tests

# Or: nightly 100 rounds × 13,000 checks each = ~1,300,000 validations
cmake --build build --target run_differential_nightly
```

All NIST/Wycheproof/BIP-340 test vectors pass. BIP-340 (27/27), RFC 6979 (35/35),
MuSig2 BIP-327 (full KAT suite), FROST (full KAT suite), Wycheproof ECDSA/ECDH.

---

## Key Differences vs. bitcoin-core/secp256k1

| Aspect | bitcoin-core/secp256k1 | UltrafastSecp256k1 |
|--------|----------------------|---------------------|
| Language | C11 | C++20 (C ABI via shim) |
| Generator precomputed table | w=15 (8 KB) | w=18 (64 KB) — 2× faster G·k |
| Scalar mult | Strauss w=5 | GLV + Strauss w=18 |
| Scalar inversion | Fermat chain | SafeGCD (Bernstein-Yang) — 6.5× faster |
| Context | Required, heap-allocated | No-op (thread-safe by design) |
| GPU acceleration | None | CUDA, OpenCL, Apple Metal |
| CT verification | Valgrind + disassembly | LLVM ct-verif + Valgrind + dudect |
| Audit system | Review + fuzzing | CAAS — 173 exploit PoC modules, CI every commit |
| BIP-352 batch scan | Not present | 11.00 M/s on RTX 5060 Ti |

---

## Context Differences (Important)

bitcoin-core/secp256k1 uses context objects (`secp256k1_context*`) for:
- Thread safety
- Context randomization (blinding)

UltrafastSecp256k1's shim accepts context pointers and ignores them safely.
Thread safety is provided by `thread_local` precomputed tables — no shared
mutable state. Context randomization is not implemented (blinding is not used
in the current implementation; this is a documented limitation).

If Bitcoin Core requires context randomization for security properties:
- The shim can add CSPRNG-based blinding in a future update
- Or the context object can carry a per-call blinding factor

---

## Minimum Viable PR to Bitcoin Core

A minimal PR to [bitcoin/bitcoin](https://github.com/bitcoin/bitcoin) would:

1. **Add `cmake/secp256k1_backend.cmake`** — contains the `USE_ULTRAFAST_SECP256K1`
   option and the FetchContent/alias wiring shown above.
2. **Modify `CMakeLists.txt`** to `include(secp256k1_backend)` before the existing
   `add_subdirectory(src/secp256k1)`.
3. **Add CI job** that builds Bitcoin Core with `-DUSE_ULTRAFAST_SECP256K1=ON` and
   runs the full test suite.
4. **No changes to any Bitcoin Core source file** — only the build system changes.

---

## Current Limitations

- `secp256k1_context_randomize` is a no-op (blinding not implemented)
- `secp256k1_context_static` points to a shared static instance (matches libsecp256k1 behavior)
- The shim does not expose UltrafastSecp256k1-specific APIs (GPU batching, BIP-352
  fast scan) — those are available only through the native C++ API or `ufsecp_*` C ABI

---

## Building the Shim Standalone

```bash
git clone https://github.com/shrec/UltrafastSecp256k1.git
cd UltrafastSecp256k1
cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DSECP256K1_BUILD_SHIM=ON \
    -DSECP256K1_SHIM_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build -R shim --output-on-failure
```
