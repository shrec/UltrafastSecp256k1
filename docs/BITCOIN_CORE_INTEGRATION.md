# Bitcoin Core Integration Guide

How to use UltrafastSecp256k1 as a **compile-time alternative** to bitcoin-core/secp256k1
inside Bitcoin Core's build system.

---

## TL;DR

UltrafastSecp256k1 ships a `compat/libsecp256k1_shim` that provides the **identical
`secp256k1.h` API surface** as bitcoin-core/secp256k1. No Bitcoin Core source changes
are required — only the CMake configuration changes.

Current integration status: **749/749 test_bitcoin pass, 0 failures** (GCC 14.2.0, 2026-05-11).
Evidence: `docs/BITCOIN_CORE_BENCH_RESULTS.json` (most recent) · `docs/BITCOIN_CORE_TEST_RESULTS.json` (earlier Clang 19 run, 693/693).

---

## Architecture

```
Bitcoin Core source (unchanged)
        │
        │ #include <secp256k1.h>
        │ secp256k1_ecdsa_sign(ctx, ...)
        ▼
┌─────────────────────────────┐
│  compat/libsecp256k1_shim   │  ← thin ABI mapping layer (C, identical types)
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  UltrafastSecp256k1 (C++)   │  ← actual implementation
│  CPU: w=18 / SafeGCD / CT   │
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

### Recommended: local checkout

Pin to a specific commit hash. Do not use `GIT_TAG main` — the library is
under active development and `main` is not a stable reference for CI:

```cmake
option(SECP256K1_USE_ULTRAFAST "Use UltrafastSecp256k1 instead of bundled secp256k1" OFF)

if(SECP256K1_USE_ULTRAFAST)
    include(FetchContent)
    FetchContent_Declare(
        UltrafastSecp256k1
        GIT_REPOSITORY https://github.com/shrec/UltrafastSecp256k1.git
        GIT_TAG        c1df659e  # pin to a tested commit hash, never "main"
    )
    FetchContent_MakeAvailable(UltrafastSecp256k1)

    add_subdirectory(
        ${ultrafastsecp256k1_SOURCE_DIR}/compat/libsecp256k1_shim
        secp256k1_shim_build
    )
    add_library(secp256k1 ALIAS secp256k1_shim)
    target_include_directories(secp256k1_shim PUBLIC
        ${ultrafastsecp256k1_SOURCE_DIR}/compat/libsecp256k1_shim/include
    )
else()
    add_subdirectory(src/secp256k1)
endif()
```

Build:
```bash
cmake -S . -B build -DSECP256K1_USE_ULTRAFAST=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

### Alternative: submodule replacement

```bash
git submodule deinit src/secp256k1
git rm src/secp256k1
git submodule add https://github.com/shrec/UltrafastSecp256k1.git src/secp256k1_uf
```

Then point CMake at `src/secp256k1_uf/compat/libsecp256k1_shim` instead of
`src/secp256k1`.

---

## Verifying Correctness

UltrafastSecp256k1 is differential-tested against bitcoin-core/secp256k1:

```bash
# 10,000+ random cross-validation cases
cmake --build build --target run_differential_tests

# Nightly: 100 rounds × 13,000 checks = ~1,300,000 validations
cmake --build build --target run_differential_nightly
```

All NIST / Wycheproof / BIP-340 test vectors pass:
BIP-340 (27/27), RFC 6979 (35/35), MuSig2 BIP-327 (full KAT suite),
FROST (full KAT suite), Wycheproof ECDSA/ECDH (11 suites, CI-gated weekly).

---

## Key Differences vs. bitcoin-core/secp256k1

| Aspect | bitcoin-core/secp256k1 | UltrafastSecp256k1 |
|--------|----------------------|---------------------|
| Language | C11 | C++20 (C ABI via shim) |
| C++ standard exposed to consumers | C11 only | C17 minimum (`PRIVATE cxx_std_20` — not propagated) |
| Generator precomputed table | w=15 (8 KB) | w=18 (64 KB) — 2× faster G·k |
| Scalar mult | Strauss w=5 | GLV + Strauss w=18 |
| Scalar inversion | Fermat chain | SafeGCD (Bernstein-Yang) — 6.5× faster |
| Context | Required, heap-allocated | Accepted; thread-safe by design |
| CT verification | Valgrind + disassembly | LLVM ct-verif + Valgrind + dudect |
| Audit system | Review + fuzzing | CAAS — 256 exploit PoCs modules, CI every commit |
| BIP-352 batch scan | Not present | Available via native `ufsecp_*` API (not relevant to Core) |

---

## Context and Randomization

bitcoin-core/secp256k1 uses `secp256k1_context*` for two purposes:
1. **Thread isolation** — each ctx is conceptually bound to a thread
2. **Context randomization** — `secp256k1_context_randomize(ctx, seed32)` activates
   additive scalar blinding to mitigate side-channel attacks during signing

UltrafastSecp256k1's shim implements **both** behaviors:

**Thread safety:** thread safety is provided by `thread_local` precomputed tables.
No shared mutable state between threads. Context pointers are accepted and tracked
per-context but thread safety does not depend on the caller's context discipline.

**Context randomization:** `secp256k1_context_randomize` is **not a no-op**.
When called with a 32-byte seed:

```cpp
// shim_context.cpp — actual implementation
Scalar r = Scalar::from_bytes(seed_arr);   // reduce seed mod n
auto r_G = secp256k1::ct::generator_mul(r); // CT scalar mult
secp256k1::ct::set_blinding(r, r_G);        // activate thread-local additive blinding
```

This activates additive scalar blinding on this thread's signing path. The blinding
factor `r` and its public key `r_G` are stored thread-locally. Every subsequent
`secp256k1_ecdsa_sign` call on this thread uses blinding.

Calling `secp256k1_context_randomize(ctx, NULL)` clears blinding via
`secp256k1::ct::clear_blinding()`.

Bitcoin Core calls `secp256k1_context_randomize` periodically to refresh the
blinding factor. This is fully supported and has the same security effect as in
libsecp256k1.

---

## Minimum Viable PR to Bitcoin Core

A minimal PR to [bitcoin/bitcoin](https://github.com/bitcoin/bitcoin) would:

1. **Add `cmake/secp256k1_backend.cmake`** — `SECP256K1_USE_ULTRAFAST` option with
   FetchContent/alias wiring.
2. **Modify `CMakeLists.txt`** — `include(secp256k1_backend)` before existing
   `add_subdirectory(src/secp256k1)`.
3. **Add CI job** — builds Bitcoin Core with `-DSECP256K1_USE_ULTRAFAST=ON`
   and runs the full test suite.
4. **No changes to any Bitcoin Core source file** — build system only.

---

## Current Limitations

- The shim does not expose UltrafastSecp256k1-specific APIs (BIP-352 fast scan,
  `ufsecp_*` C ABI) — those are available only through the native C++ or C ABI.
- `secp256k1_context_static` points to a shared static instance (matches
  libsecp256k1 behavior; blinding on the static context is not recommended).
- GPU acceleration is out of scope for the Bitcoin Core integration path — Bitcoin
  Core does not use GPU compute. GPU features are available through the native API.

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
