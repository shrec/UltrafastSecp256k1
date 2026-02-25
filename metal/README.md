# UltrafastSecp256k1 -- Apple Metal Backend

**The first secp256k1 library with Apple Metal GPU support.**

The Metal backend provides secp256k1 elliptic curve operations on Apple Silicon GPUs
(M1, M2, M3, M4, and others) using Metal Shading Language (MSL).

---

## Quick Start (M1/M2/M3/M4 MacBook)

### Prerequisites

```bash
# Xcode Command Line Tools (if not installed)
xcode-select --install

# CMake + Ninja (via Homebrew)
brew install cmake ninja

# Verification
cmake --version   # 3.21+
xcrun metal --version   # Metal compiler
```

### Build and Test (all commands together)

```bash
# 1. Clone
git clone https://github.com/shrec/UltrafastSecp256k1.git
cd UltrafastSecp256k1

# 2. Configure with Metal
cmake -S . -B build_metal -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DSECP256K1_BUILD_METAL=ON

# 3. Build
cmake --build build_metal -j

# 4. Run all tests (host + GPU)
ctest --test-dir build_metal --output-on-failure
```

### Run GPU tests/benchmarks only

```bash
# GPU tests (Gx1, Gx2, Gx3 verification + field_mul check)
./build_metal/metal/metal_secp256k1_test

# GPU benchmark (field_mul 1M ops, scalar_mul 4K ops)
./build_metal/metal/metal_secp256k1_test --bench
```

### Run Host tests only (without GPU)

```bash
./build_metal/metal/metal_host_test
# Expected: "Results: 76 passed, 0 failed"
```

---

## Architecture

### 8x32-bit Limb Model (in Shaders)

Metal Shading Language does not support 64-bit integers (`uint64_t`) in shader functions.
The CUDA backend uses 4x64-bit limbs with PTX inline assembly, while **Metal shaders use
8x32-bit limbs** with explicit carry propagation using `ulong` (64-bit) temporary variables.

**Host-side types** (`host_helpers.h`) use `uint64_t limbs[4]` -- exactly the same as
CUDA's `HostFieldElement` and shared `FieldElementData` (`types.hpp`). This ensures cross-backend
compatibility. Buffer I/O is zero-cost since `FieldElementData{uint64_t[4]}` and
`MidFieldElementData{uint32_t[8]}` are the same 32 bytes on little-endian.

| Backend | Shader Limb | Host Limb | Carry Method |
|---------|-------------|------------|--------------|
| CUDA    | 64-bit (4)  | 64-bit (4) | PTX addc     |
| Metal   | 32-bit (8)  | 64-bit (4) | ulong cast   |
| OpenCL  | 64-bit (4)  | 64-bit (4) | mul_hi()     |

### Apple Silicon Unified Memory

Apple Silicon's unified memory architecture enables zero-copy buffer
usage (`MTLResourceStorageModeShared`), eliminating explicit host<->device
data copies.

---

## File Structure

```
metal/
+-- CMakeLists.txt              # Build configuration
+-- README.md                   # This file
+-- shaders/
|   +-- secp256k1_field.h       # Field arithmetic (add, sub, mul, sqr, inv)
|   +-- secp256k1_point.h       # Point operations (double, add_mixed, scalar_mul)
|   +-- secp256k1_kernels.metal # Compute kernels (search, batch_inverse, benchmarks)
+-- include/
|   +-- gpu_compat_metal.h      # Platform macros (CUDA gpu_compat.h pattern)
|   +-- metal_runtime.h         # C++ interface (PIMPL, Obj-C types hidden)
|   +-- host_helpers.h          # Host-side types (uint64_t[4]), types.hpp integration
+-- src/
|   +-- metal_runtime.mm        # Objective-C++ runtime (ARC, pipeline caching)
+-- app/
    +-- metal_test.mm           # Tests + benchmarks
```

---

## Implemented Operations

### Field Arithmetic (`secp256k1_field.h`)
- `field_add` -- Modular addition, branchless (mod p)
- `field_sub` -- Modular subtraction, branchless (mod p)
- `field_negate` -- Modular negation
- `field_mul` -- **Comba product scanning** (CUDA PTX MAD_ACC equivalent, column-by-column accumulation)
- `field_sqr` -- **Comba + symmetry optimization** (36 multiplies instead of 64)
- `field_reduce_512` -- 512->256 bit reduction K = 0x1000003D1, branchless final subtract
- `field_inv` -- Fermat inversion (a^(p-2) mod p, 255 sqr + 14 mul chain)
- `field_sqr_n` -- Multi-squaring (sqr xN)
- `field_mul_small` -- Multiplication by scalar (< 2^32), branchless reduction
- `METAL_MAD_ACC` -- PTX `mad.lo.cc.u64/madc.hi.cc.u64/addc.u64` macro equivalent

### Point Operations (`secp256k1_point.h`)
- `jacobian_double` -- dbl-2001-b (3M + 4S)
- `jacobian_add_mixed` -- madd-2007-bl (7M + 4S)
- `jacobian_add` -- Full Jacobian addition (11M + 5S)
- `scalar_mul` -- **4-bit fixed window** (64 double + 64 add, ~35% faster than naive)
- `affine_select` -- **branchless** table read (no GPU divergence)
- `jacobian_to_affine` -- Jacobian -> Affine conversion
- `apply_endomorphism` -- GLV endomorphism (beta*x mod p)

### Compute Kernels (`secp256k1_kernels.metal`)
- `search_kernel` -- Main search kernel (**O(1) per-thread** offset, scalar_mul)
- `scalar_mul_batch` -- Scalar multiplication batch (4-bit windowed)
- `generator_mul_batch` -- Generator point multiplication (4-bit windowed)
- `field_mul_bench` -- Field multiplication benchmark (Comba)
- `field_sqr_bench` -- Field squaring benchmark (Comba + symmetry)
- `batch_inverse` -- **Chunked** Montgomery batch inversion (parallel threadgroups)
- `point_add_kernel` -- Point addition
- `point_double_kernel` -- Point doubling

---

## Build -- Detailed

For quick build instructions see the "Quick Start" section above.

### Shader Compilation

CMake automatically compiles shaders:
1. `.metal` -> `.air` (xcrun metal -O2 -std=metal2.4)
2. `.air` -> `.metallib` (xcrun metallib)

Runtime fallback: if the `.metallib` file is not found, the runtime automatically
compiles the `.metal` source file.

---

## Usage

### C++ API (metal_runtime.h)

```cpp
#include "metal_runtime.h"
#include "host_helpers.h"

// Runtime initialization
secp256k1::metal::MetalRuntime runtime;
runtime.init();

// Load shader library
runtime.load_library_from_path("secp256k1_kernels.metallib");

// Create pipeline
auto pipeline = runtime.make_pipeline("generator_mul_batch");

// Allocate buffers (zero-copy unified memory)
auto scalars_buf = runtime.alloc_buffer_shared(n * sizeof(HostScalar));
auto points_buf  = runtime.alloc_buffer_shared(n * sizeof(HostAffinePoint));

// Kernel dispatch
runtime.dispatch_1d(pipeline, n, /* threadgroup_size */ 256,
    {scalars_buf, points_buf});
runtime.synchronize();
```

---

## Performance Characteristics

### Apple Silicon GPU Specifics
- 32-bit ALU throughput: Very high (Metal is optimized for 32-bit operations)
- Unified memory: Zero copy overhead
- Threadgroup memory: 32KB per threadgroup (M1/M2), 64KB (M3/M4)
- Max threads per threadgroup: 1024

### Expected Performance
| Operation | M1 (est.) | M2 (est.) | M3 Pro (est.) |
|----------|-----------|-----------|---------------|
| field_mul | ~300M/s | ~400M/s | ~550M/s |
| scalar_mul | ~150K/s | ~200K/s | ~300K/s |

> Note: These are approximate estimates. Actual benchmarks are in the `metal_test` application.

---

## Supported Devices

| Device | GPU Family | Support |
|--------|------------|---------|
| M1 / M1 Pro / M1 Max / M1 Ultra | Apple7 | [OK] |
| M2 / M2 Pro / M2 Max / M2 Ultra | Apple8 | [OK] |
| M3 / M3 Pro / M3 Max | Apple9 | [OK] |
| M4 / M4 Pro / M4 Max | Apple9+ | [OK] |
| A14+ (iPhone/iPad) | Apple7+ | [OK] |
| Apple Vision Pro | Apple9 | [OK] |

---

## CUDA Compatibility

The Metal backend uses algorithms identical to the CUDA backend:
- Same Fermat inversion chain (x2->x3->x6->x9->x11->x22->x44->x88->x176->x220->x223->tail)
- Same Jacobian formulas (dbl-2001-b, madd-2007-bl)
- Same bloom filter hash functions (FNV-1a + SplitMix64)
- Same Montgomery batch inversion
- **Comba product scanning** -- `METAL_MAD_ACC` macro equivalent to PTX `mad.lo.cc.u64 / madc.hi.cc.u64 / addc.u64`
- **4-bit windowed scalar_mul** -- Matching CUDA's wNAF/fixed-window approach

Limb size: 4x64 -> 8x32 (in shaders), mathematical correctness is identical.

---

## Acceleration Strategy (Instead of Assembly)

CUDA uses PTX inline assembly for hardware carry-chains. Metal **does not have** inline
assembly -- Apple GPU ISA is closed. Instead:

| CUDA PTX | Metal Equivalent | Purpose |
|----------|-------------------|-------------|
| `mad.lo.cc.u64` | `METAL_MAD_ACC` macro | 96-bit accumulator column accumulation |
| `madc.hi.cc.u64` | `ulong(a)*ulong(b)` compiler MAC fusion | Hardware MAC instruction mapping |
| `addc.u64` | Explicit carry propagation | Apple Silicon compiler optimizes this |
| wNAF scalar mul | 4-bit fixed window | Precomputed table[16] + branchless select |
| `__syncthreads()` | Chunked threadgroups | Each threadgroup = independent batch |

---

## License

Same license as the main UltrafastSecp256k1 project.
