# UltrafastSecp256k1 — Build Integration Guide

This guide explains how to embed UltrafastSecp256k1 in another project,
the available build modes, and how to choose the right one.

## Quick Reference

| Mode | CMake flags | Best for |
|------|-------------|----------|
| Static library (default) | _(none)_ | Development, CI |
| Unity static | `SECP256K1_UNITY_BUILD=ON` | Production no-LTO |
| LTO static | `SECP256K1_USE_LTO=ON` | Production, maximum speed |
| Object library | `SECP256K1_BUILD_AS_OBJECT=ON` | Direct embedding |
| Object + Unity | `SECP256K1_BUILD_AS_OBJECT=ON SECP256K1_UNITY_BUILD=ON` | Tightest integration |

---

## Which Library Should I Link?

For native C++ integrations, Bitcoin-family node integrations, and the
libsecp256k1-compatible shim, link the engine target:

```cmake
add_subdirectory(path/to/UltrafastSecp256k1 uf_build)
target_link_libraries(myapp PRIVATE secp256k1::fast)
```

With the compatibility shim enabled, the shim code is compiled into
`fastsecp256k1`; `secp256k1_shim` is only the compatibility target that exposes
the `secp256k1_*` headers/symbol contract to the parent project.

`libufsecp` is the optional stable C ABI package for C callers, language
bindings, and explicit bridge consumers. It is not required just to use the
engine. The top-level install now keeps that distinction explicit:

```bash
# Native engine package only (default): installs libfastsecp256k1 + secp256k1-fast.pc.
cmake -S . -B out/install-fast -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build out/install-fast --target install

# Install both packages in one configure when you really need the C ABI package.
cmake -S . -B out/install-both -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSECP256K1_BUILD_CABI=ON \
  -DSECP256K1_INSTALL_CABI=ON
cmake --build out/install-both --target install
```

`secp256k1-fast.pc` links `-lfastsecp256k1`. `ufsecp.pc` is emitted only when
`SECP256K1_INSTALL_CABI=ON` and links `-lufsecp`.

---

## Measured Performance (bitcoin-core-dev, GCC 14.2.0, i5-14400F, 2026-05-21)

All numbers are relative to bundled libsecp256k1.

| Build mode | ConnectBlock ECDSA | ConnectBlock Schnorr | Taproot sign |
|------------|--------------------|---------------------|--------------|
| libsecp256k1 (baseline) | 0% | 0% | 0% |
| Ultra STATIC no-LTO | −8% | parity | +7% |
| Ultra STATIC+UNITY no-LTO | −2.5% | parity | **+6%** |
| Ultra STATIC+LTO | **+1.2%** | **+0.9%** | **+35%** |
| Ultra OBJECT+UNITY no-LTO | −2.5% | parity | +6% |
| Ultra OBJECT+LTO | +1.2% | +0.9% | +35% |

Binary size (stripped, LTO): Ultra = libsecp256k1 (same total; +250 KB .text).

---

## Mode 1: Static Library (default)

Standard CMake static library. A `libfastsecp256k1.a` file is produced and
linked into the parent project.

```cmake
# In your CMakeLists.txt:
add_subdirectory(path/to/UltrafastSecp256k1 uf_build)
target_link_libraries(myapp PRIVATE secp256k1::fast)
```

**Trade-off:** Separate compilation units — without LTO, the compiler cannot
inline across the boundary. Signing still beats libsecp256k1; ConnectBlock
ECDSA verify is ~8% slower than libsecp without LTO.

---

## Mode 2: Unity Static Library (`SECP256K1_UNITY_BUILD=ON`)

CMake's `UNITY_BUILD` concatenates all `.cpp` files into a single translation
unit before compiling. The compiler sees the entire library as one `.cpp`, so
it can inline across all internal boundaries — without LTO.

```bash
cmake -B build \
  -DSECP256K1_UNITY_BUILD=ON \
  -DCMAKE_BUILD_TYPE=Release
```

Or in CMakeLists:
```cmake
set(SECP256K1_UNITY_BUILD ON CACHE BOOL "" FORCE)
add_subdirectory(path/to/UltrafastSecp256k1 uf_build)
target_link_libraries(myapp PRIVATE secp256k1::fast)
```

**Result:** 
- ConnectBlock ECDSA: −2.5% vs libsecp (vs −8% standard)
- Signing: +6% vs libsecp
- Binary `.text`: 213 KB **smaller** than libsecp (inlining eliminates call stubs)
- Compile time: slightly longer (all files in one TU — less parallelism)

**Recommended for:** Production builds where LTO is unavailable or too slow
(distro packages, cross-compilation, Windows MSVC).

---

## Mode 3: LTO (`SECP256K1_USE_LTO=ON` or `CMAKE_INTERPROCEDURAL_OPTIMIZATION=ON`)

Link Time Optimization lets the compiler inline across `.o` file and `.a` file
boundaries at link time. Achieves the same inlining benefit as Unity build, with
better compile-time parallelism. Slower to link.

```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON
```

**Result:**
- ConnectBlock: +1.2% ECDSA, +0.9% Schnorr vs libsecp
- Taproot signing: +35% vs libsecp
- Compile time: 3–5× slower link step

**Recommended for:** Production nodes where link time is acceptable.

---

## Mode 4: Object Library (`SECP256K1_BUILD_AS_OBJECT=ON`)

Builds `fastsecp256k1` as a CMake `OBJECT` library instead of a `STATIC`
library. No `.a` file is created — all compiled object files are included
**directly** in the parent project's link step, as if the sources were part
of the parent project itself.

```cmake
set(SECP256K1_BUILD_AS_OBJECT ON CACHE BOOL "" FORCE)
add_subdirectory(path/to/UltrafastSecp256k1 uf_build)
target_link_libraries(myapp PRIVATE secp256k1::fast)
# CMake automatically includes $<TARGET_OBJECTS:fastsecp256k1> in myapp's link
```

**When to use:**
- Embedding in a project that manages its own link step precisely
- When you want to avoid installing a separate `.a` file
- When combined with LTO: eliminates inter-library IR boundary (better LTO quality)
- Wallet libraries, embedded firmware (ESP32, STM32) where a separate `.a` adds overhead

**Limitation:** Without LTO or Unity, OBJECT mode alone does NOT enable
cross-source-file inlining (each `.cpp` is still compiled separately).

---

## Mode 5: Object + Unity (tightest integration)

Combines both flags: one translation unit (`UNITY_BUILD`), no `.a` file
(`BUILD_AS_OBJECT`). The entire library is compiled as a single `.cpp` and
its object is linked directly into the parent binary.

```cmake
set(SECP256K1_BUILD_AS_OBJECT ON CACHE BOOL "" FORCE)
set(SECP256K1_UNITY_BUILD     ON CACHE BOOL "" FORCE)
add_subdirectory(path/to/UltrafastSecp256k1 uf_build)
target_link_libraries(myapp PRIVATE secp256k1::fast)
```

**Effect:** From the compiler's perspective, UltrafastSecp256k1 is a single
`.cpp` file compiled alongside your project. No boundaries, no separate
library, maximum inlining. This is the closest to "writing the crypto code
directly in your project."

**Recommended for:**
- Bitcoin Core fork integration (best no-LTO performance)
- Wallet applications that need maximum signing throughput without LTO
- Embedded systems where the linker cannot do LTO

---

## Bitcoin Core Integration

In the `cmake/secp256k1.cmake` file of a Bitcoin Core fork:

```cmake
# Recommended production configuration:
set(SECP256K1_BUILD_AS_OBJECT ON  CACHE BOOL "" FORCE)  # no separate .a
set(SECP256K1_UNITY_BUILD     ON  CACHE BOOL "" FORCE)  # single TU, full inlining
set(SECP256K1_BUILD_SHIM      ON  CACHE BOOL "" FORCE)  # shim inside engine
set(SECP256K1_CORE_BACKEND_MODE ON CACHE BOOL "" FORCE) # CT enforce + strict ABI

add_subdirectory(${_UF_ROOT} ultrafast_secp256k1_build)
add_library(secp256k1 ALIAS secp256k1_shim)
```

With LTO as an alternative:
```cmake
set(SECP256K1_BUILD_AS_OBJECT OFF CACHE BOOL "" FORCE)
set(SECP256K1_UNITY_BUILD     OFF CACHE BOOL "" FORCE)
# LTO is set globally:
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
```

---

## Choosing a Mode

```
Do you have LTO available and acceptable compile time?
  YES → Use LTO (Mode 3). Best absolute performance.
  NO  →
    Do you need maximum no-LTO performance?
      YES → Use Object+Unity (Mode 5). Best no-LTO result.
      NO  → Is this a development/CI build?
              YES → Default static (Mode 1). Fastest incremental builds.
              NO  → Use Unity static (Mode 2). Good balance.
```

---

## Interaction with Other Flags

| Flag | Compatible with | Notes |
|------|----------------|-------|
| `SECP256K1_BUILD_AS_OBJECT` | `SECP256K1_UNITY_BUILD`, `SECP256K1_BUILD_SHIM`, `SECP256K1_CORE_BACKEND_MODE` | Cannot be installed as a library |
| `SECP256K1_UNITY_BUILD` | All modes | Increases compile time ~20%; reduces binary size |
| `SECP256K1_BUILD_SHIM` | All modes | Compiles shim sources inside fastsecp256k1 |
| `SECP256K1_USE_LTO` | STATIC only (recommended) | Redundant with UNITY_BUILD but additive |
| `SECP256K1_CORE_BACKEND_MODE` | All modes | Enables CT enforcement + static precomputed table |

---

## Summary

The key insight: **what limits performance without LTO is not binary size but
cross-translation-unit call boundaries**. The linker dead-code-eliminates unused
features (FROST, BIP-352, etc.) regardless of which `SECP256K1_BUILD_*` flags
are set — only the functions Bitcoin Core actually calls end up in the binary.
The +250 KB difference vs libsecp in `.text` section fits in L2 cache and is
not the source of the no-LTO deficit.

`SECP256K1_UNITY_BUILD=ON` is the primary lever for no-LTO performance.
`SECP256K1_BUILD_AS_OBJECT=ON` is the lever for tightest embedding.
Both can be combined.

## GPU Feature Modules (independent of CPU modules)

The GPU op surface has its own per-feature build flags, **separate** from the
CPU feature modules — a consumer may want, e.g., BIP-324 on the CPU but stripped
from the GPU build, or a node that only needs GPU ECDSA/Schnorr signature
verification with everything else removed:

| Flag (default ON) | Strips when OFF |
|-------------------|-----------------|
| `SECP256K1_GPU_BUILD_ZK`     | ZK GPU ops: knowledge / DLEQ / bulletproof / SNARK witness kernels + dispatch |
| `SECP256K1_GPU_BUILD_BIP324` | BIP-324 AEAD encrypt/decrypt kernels + dispatch |
| `SECP256K1_GPU_BUILD_FROST`  | FROST partial-verify kernel + dispatch |
| `SECP256K1_GPU_BUILD_BIP352` | BIP-352 silent-payment scan kernel + dispatch |
| `SECP256K1_GPU_BUILD_ECDH`   | ECDH batch kernel + dispatch |
| `SECP256K1_GPU_BUILD_MSM`    | multi-scalar-multiplication kernels + dispatch |
| `SECP256K1_GPU_BUILD_HASH160`| Hash160(pubkey) kernel + dispatch |
| `SECP256K1_GPU_BUILD_ECRECOVER` | ECDSA recovery kernel + dispatch |

Turning **all** of the above OFF yields the minimal node GPU build — only
generator-mul + ECDSA/Schnorr batch verification remain.

When a module is OFF, its kernels are not compiled into any GPU backend
(CUDA/OpenCL/Metal) and the stable C ABI entry point returns
`UFSECP_ERR_GPU_UNSUPPORTED` rather than dispatching. Core ECDSA/Schnorr batch
verification is never gated.

Measured (CUDA, RTX 5060 Ti build, static-archive sizes — all four GPU modules
OFF): `secp256k1_gpu_host` ≈ −17% (27.5 → 22.7 MB), `secp256k1_cuda_lib` ≈ −20%
(9.6 → 7.6 MB). Core batch verify still passes (ECDSA + Schnorr) with all
optional modules stripped.

```bash
# Minimal GPU node build: only ECDSA/Schnorr batch verify on the GPU.
cmake --preset cpu-release -DSECP256K1_BUILD_CUDA=ON \
    -DSECP256K1_GPU_BUILD_ZK=OFF -DSECP256K1_GPU_BUILD_BIP324=OFF \
    -DSECP256K1_GPU_BUILD_FROST=OFF -DSECP256K1_GPU_BUILD_BIP352=OFF \
    -DSECP256K1_GPU_BUILD_ECDH=OFF -DSECP256K1_GPU_BUILD_MSM=OFF \
    -DSECP256K1_GPU_BUILD_HASH160=OFF -DSECP256K1_GPU_BUILD_ECRECOVER=OFF
# Verified (RTX 5060 Ti): all eight OFF builds; every optional kernel stripped;
# ECDSA + Schnorr batch verify still pass.
```

## Verifying GPU == CPU consensus (batch sig verify)

The GPU batch verification path is consensus-bearing: for block validation the
GPU verdict on every script signature must match the CPU reference bit-for-bit
(the CPU path is itself gated against libsecp256k1). A standalone differential is
provided so an integrator can reproduce that proof on their own hardware:

```bash
# Build the consensus differential from the central source (CUDA shown; the
# libbitcoin object-mode profile compiles the bridge + GPU host layer in-tree).
cmake -S . -B out/lbtc-consensus -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DSECP256K1_BUILD_LIBBITCOIN=ON -DSECP256K1_BUILD_LIBBITCOIN_TESTS=ON \
    -DSECP256K1_BUILD_CUDA=ON
cmake --build out/lbtc-consensus --target test_lbtc_consensus_diff -j"$(nproc)"

# Run it. Exit 0 = GPU==CPU bit-for-bit; exit 77 = no GPU (skipped).
./out/lbtc-consensus/include/ufsecp/test_lbtc_consensus_diff
```

It builds one mixed corpus (valid signatures plus six rejection classes) and
verifies it through a GPU controller and a CPU controller, failing on any per-row
verdict mismatch. In our local GPU CI (`gpu-selfhosted.yml`) the same target runs
as a `ctest` gate (`lbtc_consensus_diff`); it is local-only because GPU hardware
is not available on GitHub-hosted runners.

### Measured throughput (RTX 5060 Ti, 2026-05-30)

Measured with `bench_lbtc_batch` (batch = 1 048 576 rows, best-of-10, turbo-locked,
`taskset -c 0 nice -20`); artifact:
`benchmarks/gpu/cuda/rtx-50xx/lbtc_batch_verify_rtx5060ti_20260530.txt`.

| Batch verify | GPU (CUDA) | CPU bridge fallback | GPU speedup |
|--------------|-----------:|--------------------:|------------:|
| ECDSA   | **3.56 M sig/s** (281 ns/sig) | 0.02 M sig/s | ~177× |
| Schnorr | **4.53 M sig/s** (221 ns/sig) | 0.04 M sig/s | ~104× |

libsecp256k1 has no GPU batch-verify path, so the CPU column is the bridge's own
fallback (the consensus reference). The GPU verdict matches that reference
bit-for-bit (`test_lbtc_consensus_diff`), so this is throughput, not a new
correctness claim. OpenCL/Metal throughput: not yet measured — benchmark required.
