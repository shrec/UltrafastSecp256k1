# GPU Testing & Benchmark Guide
## UltrafastSecp256k1 — OpenCL / CUDA / Metal

> This document guides testing of ALL GPU backends when switching to Linux/Apple.

---

## 1. File Inventory (What Was Created)

### CUDA (reference — already complete)
- `cuda/include/hash160.cuh` — SHA-256 + RIPEMD-160 + Hash160
- `cuda/include/ecdsa.cuh` — ECDSA sign/verify
- `cuda/include/schnorr.cuh` — Schnorr BIP-340
- `cuda/include/ecdh.cuh` — ECDH shared secret
- `cuda/include/recovery.cuh` — Key recovery
- `cuda/include/msm.cuh` — Multi-scalar multiplication
- `cuda/src/test_suite.cu` — Full test suite

### OpenCL
- `opencl/kernels/secp256k1_field.cl` — Field arithmetic (4×64-bit)
- `opencl/kernels/secp256k1_point.cl` — EC point operations
- `opencl/kernels/secp256k1_batch.cl` — Batch operations
- `opencl/kernels/secp256k1_affine.cl` — Affine conversions
- `opencl/kernels/secp256k1_extended.cl` — Scalar, SHA-256, HMAC, RFC6979, ECDSA, Schnorr, ECDH, Recovery, MSM (~1370 lines)
- `opencl/kernels/secp256k1_hash160.cl` — **NEW** — SHA-256 one-shot + RIPEMD-160 + Hash160
- `opencl/tests/opencl_extended_test.cpp` — **NEW** — Host-side test+bench
- `opencl/src/opencl_selftest.cpp` — Existing 40-test suite (field/point)

### Metal
- `metal/shaders/secp256k1_field.h` — Field arithmetic (8×32-bit)
- `metal/shaders/secp256k1_point.h` — EC point operations
- `metal/shaders/secp256k1_affine.h` — Affine conversions
- `metal/shaders/secp256k1_bloom.h` — Bloom filter (external — not part of this project)
- `metal/shaders/secp256k1_extended.h` — Scalar, SHA-256, HMAC, RFC6979, ECDSA, Schnorr, ECDH, Recovery, MSM (~680 lines)
- `metal/shaders/secp256k1_hash160.h` — **NEW** — SHA-256 one-shot + RIPEMD-160 + Hash160
- `metal/shaders/secp256k1_kernels.metal` — **UPDATED** — Now includes extended.h + hash160.h, 18 kernels total
- `metal/tests/metal_extended_test.mm` — **NEW** — Host-side test+bench
- `metal/src/metal_runtime.mm` — Existing Metal runtime

---

## 2. Feature Coverage Matrix

| Feature           | CUDA | OpenCL | Metal | Notes |
|-------------------|------|--------|-------|-------|
| Field add/sub/mul | ✅   | ✅     | ✅    |       |
| Field inv/sqr     | ✅   | ✅     | ✅    |       |
| Field sqrt        | ✅   | ✅     | ✅    |       |
| Point add/double  | ✅   | ✅     | ✅    |       |
| Scalar mul (4-bit)| ✅   | ✅     | ✅    |       |
| Batch inverse     | ✅   | ✅     | ✅    |       |
| Affine convert    | ✅   | ✅     | ✅    |       |
| Scalar mod-n ops  | ✅   | ✅     | ✅    |       |
| GLV endomorphism  | ✅   | ✅     | ✅    |       |
| SHA-256 streaming | ✅   | ✅     | ✅    |       |
| SHA-256 one-shot  | ✅   | ✅     | ✅    | For Hash160 |
| HMAC-SHA256       | ✅   | ✅     | ✅    |       |
| RFC 6979          | ✅   | ✅     | ✅    |       |
| ECDSA sign/verify | ✅   | ✅     | ✅    |       |
| Schnorr BIP-340   | ✅   | ✅     | ✅    |       |
| ECDH              | ✅   | ✅     | ✅    |       |
| Key Recovery      | ✅   | ✅     | ✅    |       |
| MSM / Pippenger   | ✅   | ✅     | ✅    |       |
| RIPEMD-160        | ✅   | ✅     | ✅    |       |
| Hash160           | ✅   | ✅     | ✅    |       |
| Bloom filter      | ✅   | ❌     | ✅*   | *External, not part of project |

---

## 3. Linux Testing — CUDA

### Prerequisites
```bash
# NVIDIA driver + CUDA toolkit
nvidia-smi  # Verify GPU
nvcc --version  # Verify CUDA
```

### Build
```bash
cd libs/UltrafastSecp256k1
cmake -S Secp256K1fast -B Secp256K1fast/build_rel -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build Secp256K1fast/build_rel -j
```

### Test
```bash
ctest --test-dir Secp256K1fast/build_rel --output-on-failure
```

### Expected Results
- All CUDA tests pass (P0 scalar/field, P1 ECDSA, P2 Schnorr/ECDH/Recovery/MSM)
- Hash160 test vectors:
  - `Hash160(compressed key=1)` = `751e76e8199196d454941c45d1b3a323f1433bd6`
  - `Hash160(uncompressed key=1)` = `91b24bf9f5288532960ac687abb035127b1d28a5`

---

## 4. Linux Testing — OpenCL

### Prerequisites
```bash
# Install OpenCL ICD + headers
sudo apt install ocl-icd-opencl-dev opencl-headers
# For NVIDIA GPU:
sudo apt install nvidia-opencl-dev
# Or for Intel:
sudo apt install intel-opencl-icd
# Verify:
clinfo | head -20
```

### Build Test
```bash
cd libs/UltrafastSecp256k1/opencl

# Compile the test (standalone)
g++ -std=c++17 -O2 \
  -I kernels/ \
  tests/opencl_extended_test.cpp \
  -lOpenCL \
  -o opencl_extended_test
```

### Run Tests
```bash
# Copy kernels next to the binary
cp kernels/*.cl .

# Run tests
./opencl_extended_test --verbose

# Run benchmarks
./opencl_extended_test --bench --count 131072
```

### Build Existing Self-Test (field/point)
```bash
g++ -std=c++17 -O2 \
  -I . \
  src/opencl_selftest.cpp src/opencl_context.cpp \
  src/opencl_field.cpp src/opencl_point.cpp src/opencl_batch.cpp \
  -lOpenCL \
  -o opencl_selftest
./opencl_selftest
```

### Expected Test Results
```
Hash160(compressed key=1):    751e76e8199196d454941c45d1b3a323f1433bd6
Hash160(uncompressed key=1):  91b24bf9f5288532960ac687abb035127b1d28a5
All 40 existing field/point tests: PASS
```

### Troubleshooting
- If kernel build fails: check `-cl-std=CL2.0` support, try removing it
- If `ulong` not available: device doesn't support 64-bit int — unusual for GPUs
- Include path issues: ensure `-I kernels/` or place all `.cl` files in CWD

---

## 5. Apple Metal Testing

### Prerequisites
- macOS 12+ with Apple Silicon (M1/M2/M3) or Intel Mac with Metal support
- Xcode Command Line Tools: `xcode-select --install`

### Build Metal Library
```bash
cd libs/UltrafastSecp256k1/metal

# Compile shader to .air
xcrun -sdk macosx metal -c shaders/secp256k1_kernels.metal \
  -o secp256k1.air \
  -I shaders/

# Link to .metallib
xcrun -sdk macosx metallib secp256k1.air -o secp256k1.metallib
```

### Build Test
```bash
# Compile test + runtime
clang++ -std=c++17 -O2 -fobjc-arc \
  -framework Metal -framework Foundation \
  tests/metal_extended_test.mm \
  src/metal_runtime.mm \
  -I src/ -I shaders/ \
  -o metal_extended_test
```

### Run Tests
```bash
# Make sure metallib or .metal source is accessible
cp secp256k1.metallib .  # Or the test will compile from source

# Run tests
./metal_extended_test --verbose

# Run benchmarks (default: 65536 items)
./metal_extended_test --bench --count 131072
```

### Expected Results
```
Hash160(compressed key=1):   751e76e8199196d454941c45d1b3a323f1433bd6
Hash160(uncompressed key=1): 91b24bf9f5288532960ac687abb035127b1d28a5
field_mul(2, 3) = 6:         PASS
1*G = G:                     PASS
```

### Metal Kernel List (18 kernels in secp256k1_kernels.metal)
1. `search_kernel` — Batch ECC search
2. `scalar_mul_batch` — Batch P×k
3. `generator_mul_batch` — Batch G×k
4. `field_mul_bench` — Benchmark
5. `field_sqr_bench` — Benchmark
6. `field_add_bench` — Benchmark
7. `field_sub_bench` — Benchmark
8. `field_inv_bench` — Benchmark
9. `batch_inverse` — Chunked Montgomery
10. `point_add_kernel` — Testing
11. `point_double_kernel` — Testing
12. `ecdsa_sign_batch` — Batch ECDSA sign
13. `ecdsa_verify_batch` — Batch ECDSA verify
14. `schnorr_sign_batch` — Batch Schnorr sign
15. `schnorr_verify_batch` — Batch Schnorr verify
16. `ecdh_batch` — Batch ECDH
17. `hash160_batch` — Batch Hash160
18. `ecrecover_batch` — Batch key recovery
19. `sha256_bench` — SHA-256 benchmark
20. `hash160_bench` — Hash160 benchmark
21. `ecdsa_bench` — ECDSA sign+verify benchmark

### Troubleshooting (Metal)
- "Function not found" — Add `#include "secp256k1_extended.h"` to kernels.metal (already done)
- Compile error on 64-bit int — Metal uses 8×32-bit limbs, no `ulong` needed
- MTLGPUFamilyApple9 error — Update Xcode or use `@available(macOS 14.0, *)`

---

## 6. Benchmark Comparison Template

Run on each platform and fill in:

| Operation            | CUDA (RTX)  | OpenCL (GPU) | Metal (M-series) |
|----------------------|-------------|--------------|-------------------|
| Field mul            |             |              |                   |
| Field inv            |             |              |                   |
| Field sqr            |             |              |                   |
| Generator mul (k*G)  |             |              |                   |
| Scalar mul (P*k)     |             |              |                   |
| Batch inverse        |             |              |                   |
| SHA-256              |             |              |                   |
| Hash160              |             |              |                   |
| ECDSA sign           |             |              |                   |
| ECDSA verify         |             |              |                   |
| Schnorr sign         |             |              |                   |
| Schnorr verify       |             |              |                   |
| ECDH                 |             |              |                   |
| Key recovery         |             |              |                   |

Units: **ops/sec** (batch size = 131072)

---

## 7. Test Vectors (Cross-Platform Verification)

### Hash160
```
Input:  0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798
Output: 751e76e8199196d454941c45d1b3a323f1433bd6

Input:  0479be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798
        483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8
Output: 91b24bf9f5288532960ac687abb035127b1d28a5
```

### Generator Point (1*G)
```
X: 79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798
Y: 483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8
```

### 2*G
```
X: c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5
Y: 1ae168fea63dc339a3c58419466ceaeef7f632653266d0e1236431a950cfe52a
```

### SHA-256("abc")
```
ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
```

---

## 8. Quick Command Reference

### Linux (CUDA + OpenCL)
```bash
# CUDA tests
ctest --test-dir Secp256K1fast/build_rel --output-on-failure

# OpenCL tests
cd libs/UltrafastSecp256k1/opencl
g++ -std=c++17 -O2 -I kernels/ tests/opencl_extended_test.cpp -lOpenCL -o opencl_extended_test
cp kernels/*.cl .
./opencl_extended_test --bench --count 131072
```

### Apple (Metal)
```bash
cd libs/UltrafastSecp256k1/metal
xcrun -sdk macosx metal -c shaders/secp256k1_kernels.metal -o secp256k1.air -I shaders/
xcrun -sdk macosx metallib secp256k1.air -o secp256k1.metallib
clang++ -std=c++17 -O2 -fobjc-arc -framework Metal -framework Foundation \
  tests/metal_extended_test.mm src/metal_runtime.mm -I src/ -I shaders/ -o metal_extended_test
./metal_extended_test --bench --count 131072
```

---

## 9. Architecture Notes

### Limb Sizes
- **CUDA**: 4×`uint64_t` (native 64-bit, PTX `mul.hi.u64`)
- **OpenCL**: 4×`ulong` (64-bit, `mul_hi()`)
- **Metal**: 8×`uint32_t` (no 64-bit int on Apple GPU!)

### Key Differences
- Metal has NO 64-bit integer support on GPU → 8×32-bit with carry chains
- Metal uses `constant` instead of `__constant`
- Metal uses `thread` qualifier for private pointers
- Metal uses `[[buffer(N)]]` for buffer bindings
- OpenCL uses `_impl` suffix convention for inline functions
- CUDA has `__device__ __forceinline__` qualifiers

### Hash160 Pipeline
```
pubkey (33 or 65 bytes)
  → SHA-256 (one-shot, big-endian output, 32 bytes)
  → RIPEMD-160 (two parallel chains, little-endian output, 20 bytes)
  = Hash160 (20 bytes)
```

---

> **Reminder**: Bloom filters are NOT part of this project — they should be external.
