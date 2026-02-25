# Porting Guide -- UltrafastSecp256k1

How to add a new CPU architecture, embedded target, or GPU backend to UltrafastSecp256k1.

---

## Overview

UltrafastSecp256k1 is designed for portability. The core library is pure C++20 with **zero external dependencies**. Platform-specific acceleration is layered on top via optional assembly and GPU backends. The portable C++ path compiles on any conforming compiler -- all optimizations are additive.

---

## 1. New CPU Architecture (e.g., LoongArch, MIPS64, s390x)

### Minimum Requirements

- 64-bit architecture (32-bit works but slower)
- C++20 conforming compiler (GCC 11+, Clang 15+)
- `uint64_t` support

### Steps

1. **Verify portable build compiles and tests pass**
   ```bash
   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSECP256K1_USE_ASM=OFF
   cmake --build build -j
   ctest --test-dir build --output-on-failure
   ```
   If all 35+ tests pass, the portable path works. You're done for basic support.

2. **Add architecture detection** to `cpu/include/secp256k1/platform_detect.hpp`:
   ```cpp
   #elif defined(__loongarch64)
   #define SECP256K1_ARCH_LOONGARCH64 1
   #define SECP256K1_ARCH_64BIT 1
   ```

3. **Optional: Add assembly optimizations**
   - Create `cpu/src/field_asm_<arch>.S` (GAS syntax) or `.asm` (MASM)
   - Implement at minimum: `field_mul_asm`, `field_sqr_asm`, `field_add_asm`
   - Guard with `#ifdef SECP256K1_ARCH_<ARCH>` preprocessor checks
   - Add to `cpu/CMakeLists.txt` with architecture detection

4. **Optional: `__int128` support**
   - If compiler supports `__int128`, the 5x52 field representation is used automatically
   - If not (e.g., MSVC), the 4x64 portable path is used

5. **Run benchmarks** -- compare against portable C++ baseline:
   ```bash
   ./bench_comprehensive
   ```

6. **Add CI job** in `.github/workflows/ci.yml` (cross-compilation or native runner).

### Files to Touch

| File | Change |
|------|--------|
| `cpu/include/secp256k1/platform_detect.hpp` | Architecture detection macro |
| `cpu/src/field_asm_<arch>.S` | Assembly (optional) |
| `cpu/CMakeLists.txt` | Source file + flag guards |
| `.github/workflows/ci.yml` | CI job |
| `README.md` | Badge + benchmark table entry |

---

## 2. New Embedded Target (e.g., nRF52, RP2040, RISC-V MCU)

### Minimum Requirements

- 32-bit or 64-bit CPU
- ~8 KB stack (for Jacobian->Affine batch operations)
- ~2 KB flash for minimal field/scalar code
- C++20 compiler (or C++17 with minor adjustments)

### Steps

1. **Create example directory**: `examples/<board>_test/`

2. **Add toolchain file** (CMake cross-compilation):
   ```cmake
   # cmake/<board>.toolchain.cmake
   set(CMAKE_SYSTEM_NAME Generic)
   set(CMAKE_SYSTEM_PROCESSOR arm)  # or riscv32, etc.
   set(CMAKE_C_COMPILER arm-none-eabi-gcc)
   set(CMAKE_CXX_COMPILER arm-none-eabi-g++)
   ```

3. **Create minimal test**: Port `selftest.cpp` to run on target with UART output.

4. **Disable features that don't fit**:
   - `-DSECP256K1_USE_ASM=OFF` (if no asm for this arch)
   - Small batch sizes (reduce stack usage)
   - No `std::vector`, no heap (embedded hot-path contract)

5. **Benchmark key operations**: At minimum, measure `Field Mul`, `Field Inv`, `Scalar x G`.

6. **Document in README**: Add to embedded comparison table.

### Reference Ports

| Target | Directory | Notes |
|--------|-----------|-------|
| ESP32-S3 | `examples/esp32_test/` | Xtensa LX7, ESP-IDF, portable C++ |
| ESP32 | `examples/esp32_test/` | Xtensa LX6, dual-core |
| STM32F103 | `examples/stm32_test/` | ARM Cortex-M3, inline asm for mul |

---

## 3. New GPU Backend (e.g., Vulkan Compute, Intel oneAPI/SYCL)

### Minimum Requirements

- 32-bit integer arithmetic on device
- Shared memory / local data share (for batch inverse)
- Ability to launch large thread counts (>10K)

### Steps

1. **Create backend directory**: `<backend>/` (e.g., `vulkan/` or `sycl/`)

2. **Port field arithmetic first**:
   - `field_mul`, `field_sqr`, `field_add`, `field_sub`, `field_inv` (Fermat)
   - 8x32-bit limb representation (like Metal) or 4x64-bit if hardware supports 64-bit int

3. **Port point operations**:
   - `point_add` (Jacobian), `point_dbl` (Jacobian)
   - `jacobian_add_mixed` (Jacobian + Affine, 7M+4S)

4. **Port batch inverse** (Montgomery trick):
   - Forward pass: cumulative products
   - Single inversion (Fermat)
   - Backward pass: extract individual inverses

5. **Port scalar multiplication**:
   - wNAF or fixed-window for kxG
   - GLV endomorphism (optional, for 2x speedup)

6. **Add kernel benchmarks**: Field/Point/ScalarMul microbenchmarks.

7. **Add CMake build option**: `SECP256K1_BUILD_<BACKEND>=OFF`

8. **Add selftest**: Port GPU test vectors from `cuda/tests/`.

### Reference Implementations

| Backend | Directory | Limb Repr | Notes |
|---------|-----------|-----------|-------|
| CUDA | `cuda/` | 4x64-bit | `__int128`-like via PTX `mul.hi.u64` |
| OpenCL | `opencl/` | 4x64-bit | PTX inline asm on NVIDIA |
| Metal | `metal/` | 8x32-bit Comba | Apple GPU, no 64-bit int |
| ROCm/HIP | via `cuda/` | 4x64-bit | `__int128` fallback |

### Key Kernel Files to Study

| File | Purpose |
|------|---------|
| `cuda/include/field_ops.cuh` | Device-side field arithmetic |
| `cuda/include/point_ops.cuh` | Device-side point operations |
| `cuda/include/batch_inversion.cuh` | Montgomery batch inverse kernel |
| `cuda/include/scalar_mul.cuh` | Scalar multiplication kernel |
| `metal/shaders/secp256k1_kernels.metal` | Metal equivalent |
| `opencl/kernels/secp256k1_field.cl` | OpenCL equivalent |

---

## 4. Checklist for Any New Port

- [ ] Portable C++ build compiles without errors
- [ ] All 35+ selftest vectors pass
- [ ] No heap allocation in hot paths
- [ ] Fixed-size POD types only (no `std::vector` in kernels)
- [ ] Endianness handled correctly (`from_limbs` = little-endian, `from_bytes` = big-endian)
- [ ] Benchmark results documented
- [ ] CI job added (or cross-compilation instructions provided)
- [ ] README updated with badge + benchmark entry
- [ ] No new external dependencies added

---

## 5. Testing Your Port

### Minimum Test Coverage

```bash
# Build with tests enabled
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSECP256K1_BUILD_TESTS=ON
cmake --build build -j

# Run all tests
ctest --test-dir build --output-on-failure

# Run selftest explicitly (smoke mode for embedded)
./build/selftest --mode=smoke
```

### Known Test Vectors

The selftest includes deterministic KAT vectors for:
- Field arithmetic (add, sub, mul, square, inverse, negate)
- Scalar arithmetic (add, sub, mul, inverse, negate)
- Point operations (generator, add, dbl, scalar mul)
- ECDSA (sign, verify, recovery)
- Schnorr BIP-340 (sign, verify)
- Batch inverse (sweep from 1 to 1024)
- Boundary conditions (zero, one, p-1, n-1)

---

## 6. Submitting Your Port

1. Fork the repository
2. Create a branch: `port/<platform>`
3. Add your port following this guide
4. Include benchmark results in the PR description
5. Ensure CI passes (or explain cross-compilation setup)
6. Submit PR with:
   - What platform/architecture
   - Benchmark results (at least Field Mul, Field Inv, Scalar x G)
   - Test results (selftest pass/fail count)

---

*UltrafastSecp256k1 v3.6.0 -- Porting Guide*
