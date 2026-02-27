# Secp256k1 CUDA -- GPU ECC Library

> **English summary**: Full secp256k1 ECC library for NVIDIA GPUs -- header-only core with PTX inline assembly. Supports CUDA and ROCm/HIP (via `gpu_compat.h` abstraction layer). Priority: maximum throughput for batch operations. Not side-channel resistant (research/development use). See [docs/API_REFERENCE.md](../docs/API_REFERENCE.md) for the full API and [docs/BUILDING.md](../docs/BUILDING.md) for build instructions.

Full secp256k1 ECC library for NVIDIA GPUs -- header-only core with PTX inline assembly.

**Priority**: Maximum throughput for batch operations. Not side-channel resistant (research/dev use).

---

## Architecture

All code resides in the `secp256k1::cuda` namespace. The core is **header-only** -- `secp256k1.cuh` contains all device functions. Data types are interoperable with the CPU library (`secp256k1/types.hpp` POD structs).

### Compile-Time Configuration (3 backends)

| Macro | Default | Description |
|-------|---------|--------|
| `SECP256K1_CUDA_USE_HYBRID_MUL` | **ON** | 32-bit Comba mul + 64-bit reduction (1.10x faster) |
| `SECP256K1_CUDA_USE_MONTGOMERY` | OFF | Montgomery residue domain (mont_reduce_512) |
| `SECP256K1_CUDA_LIMBS_32` | OFF | Full 8x32-bit limbs (separate backend) |

**Default path** (64-bit hybrid): `field_mul` -> `field_mul_hybrid` -> 32-bit Comba PTX -> `reduce_512_to_256`

---

## Functionality

### Field Arithmetic (Fp)
- **add/sub**: PTX inline asm with carry chains (ADDC.CC/SUBC.CC)
- **mul**: 32-bit Comba hybrid -> 64-bit secp256k1 fast reduction (P = 2^2⁵⁶ - 2^3^2 - 977)
- **sqr**: Optimized squaring (cross-product doubling)
- **inverse**: Fermat chain `a^{p-2}` (255 sqr + 16 mul)
- **mul_small**: Multiplication by uint32 (for reduction constants)
- **Montgomery**: `field_to_mont`, `field_from_mont`, `mont_reduce_512` (optional backend)

### Scalar Arithmetic (Fn)
- **add/sub**: Modular arithmetic mod curve order N
- **bit extraction**: Fast bit access for scalar processing

### Point Operations (Jacobian coordinates)
- **doubling**: `dbl-2001-b` (3M+4S, a=0 curves)
- **mixed addition**: 6 variants optimized for different scenarios:
  - `jacobian_add_mixed` -- madd-2007-bl (7M+4S) general
  - `jacobian_add_mixed_h` -- madd-2004-hmv (8M+3S), H output for batch inversion
  - `jacobian_add_mixed_h_z1` -- Z=1 specialized (5M+2S), first step
  - `jacobian_add_mixed_const` -- branchless (8M+3S), constant-point
  - `jacobian_add_mixed_const_7m4s` -- branchless 7M+4S + 2H output
- **general add**: `jacobian_add` (11M+5S, Jacobian + Jacobian)
- **GLV endomorphism**: `apply_endomorphism` phi(x,y) = (beta*x, y)

### Scalar Multiplication
- **double-and-add**: Simple, register-efficient (wNAF is expensive on GPU due to register pressure)
- **Batch kernels**: `scalar_mul_batch_kernel`, `generator_mul_batch_kernel`

### Batch Inversion
- **Montgomery trick**: prefix/suffix scan (default, one inversion for N elements)
- **Fermat**: `a^{p-2}` for each element (fallback)
- **naive**: Direct GCD (debug/reference)

### Hash160 (SHA-256 + RIPEMD-160)
- `hash160_pubkey_kernel` -- pubkey -> Hash160 device-side

### Bloom Filter
- `DeviceBloom` -- FNV-1a + SplitMix hashing
- `test` / `add` device functions + batch kernels

---

## File Structure

```
cuda/
+-- CMakeLists.txt                              # Build: lib + test + bench
+-- README.md
+-- include/
|   +-- secp256k1.cuh                           # Core -- field/point/scalar device functions (1800+ lines)
|   +-- ptx_math.cuh                            # PTX inline asm (256x256->512 Comba multiply)
|   +-- secp256k1_32.cuh                        # Alternative: 8x32-bit limbs + Montgomery backend
|   +-- secp256k1_32_hybrid_final.cuh           # 32-bit Comba mul -> 64-bit reduction (default mul path)
|   +-- batch_inversion.cuh                     # Montgomery trick / Fermat / naive batch inverse
|   +-- bloom.cuh                               # Device-side Bloom filter (FNV-1a + SplitMix)
|   +-- hash160.cuh                             # SHA-256 + RIPEMD-160 -> Hash160
|   +-- host_helpers.cuh                        # Host-side wrappers (1-thread kernels, test-only)
|   +-- gpu_compat.h                            # CUDA <-> HIP (ROCm) compatibility layer
+-- src/
|   +-- secp256k1.cu                            # Kernel definitions (thin wrappers)
|   +-- test_suite.cu                           # 30 vector tests
|   +-- bench_cuda.cu                           # Benchmark harness
```

---

## Build

```bash
# Via parent CMakeLists.txt (or standalone)
cmake -S cuda -B cuda/build -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build cuda/build -j

# Tests
./cuda/build/secp256k1_cuda_test

# Benchmark
./cuda/build/secp256k1_cuda_bench
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_CUDA_ARCHITECTURES` | 89 (Ada) | NVIDIA GPU architecture (75/80/86/89/90) |
| `SECP256K1_CUDA_USE_MONTGOMERY` | OFF | Montgomery domain |
| `SECP256K1_CUDA_LIMBS_32` | OFF | 8x32-bit limb backend |
| `SECP256K1_BUILD_ROCM` | OFF | AMD ROCm/HIP build (portable math) |
| `CMAKE_HIP_ARCHITECTURES` | -- | AMD GPU architectures (gfx906/gfx1030/gfx1100/...) |

### Requirements
- **NVIDIA**: CUDA Toolkit 12.0+, GPU Compute Capability 7.0+ (Volta+), CMake 3.18+
- **AMD**: ROCm 5.0+ (HIP SDK), CMake 3.21+, gfx9/gfx10/gfx11 GPU

### ROCm/HIP Build (AMD GPU)

```bash
# With ROCm Docker or native installation
cmake -S . -B build-rocm -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSECP256K1_BUILD_ROCM=ON \
  -DCMAKE_HIP_ARCHITECTURES="gfx1030;gfx1100"

cmake --build build-rocm -j
./build-rocm/cuda_rocm/secp256k1_cuda_test
```

> **Note**: In ROCm builds, PTX inline asm is automatically replaced with portable
> `__int128` fallbacks (`gpu_compat.h` -> `SECP256K1_USE_PTX=0`).
> The 32-bit hybrid mul backend (PTX-dependent) is automatically disabled on HIP.

---

## Usage

### Device Functions

```cpp
#include "secp256k1.cuh"

__global__ void my_kernel(const Scalar* scalars, JacobianPoint* results, int n) {
    using namespace secp256k1::cuda;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // G * k -- GENERATOR_JACOBIAN is embedded at compile time
    JacobianPoint G = GENERATOR_JACOBIAN;
    scalar_mul(&G, &scalars[idx], &results[idx]);
}
```

### Batch Processing

```cpp
#include "secp256k1.cuh"

const int N = 1 << 20;  // ~1M operations
Scalar* d_scalars;
JacobianPoint* d_results;

cudaMalloc(&d_scalars, N * sizeof(Scalar));
cudaMalloc(&d_results, N * sizeof(JacobianPoint));

// Generator multiplication batch
int block = 256;
int grid = (N + block - 1) / block;
generator_mul_batch_kernel<<<grid, block>>>(d_scalars, d_results, N);
cudaDeviceSynchronize();
```

---

## Tests

30 vector tests in `test_suite.cu`:
- Field arithmetic: identity, inverse, commutativity, associativity
- Scalar arithmetic: add, sub, boundary
- Point operations: doubling, mixed addition, identity
- Scalar multiplication: known vectors, generator mul
- GLV endomorphism: phi(phi(P)) + P = -phi(P)
- Batch inversion: Montgomery trick correctness
- Cross-backend: CPU <-> CUDA result comparison

---

## CPU <-> CUDA Compatibility

Data types share layout via `secp256k1/types.hpp`:

```cpp
static_assert(sizeof(FieldElement) == 32);
static_assert(sizeof(Scalar) == 32);
static_assert(sizeof(AffinePoint) == 64);
static_assert(offsetof(FieldElement, limbs) == 0);
```

CPU-computed data transfers directly to GPU via `cudaMemcpy` (little-endian, same POD layout).

---

## Cross-Platform Benchmarks

### Android ARM64 -- RK3588 (Cortex-A55/A76), ARM64 inline ASM (MUL/UMULH)

| Operation | Time |
|-----------|------|
| field_mul (a*b mod p) | 85 ns |
| field_sqr (a^2 mod p) | 66 ns |
| field_add (a+b mod p) | 18 ns |
| field_sub (a-b mod p) | 16 ns |
| field_inverse | 2,621 ns |
| **fast scalar_mul (k*G)** | **7.6 us** |
| fast scalar_mul (k*P) | 77.6 us |
| CT scalar_mul (k*G) | 545 us |
| ECDH (full CT) | 545 us |

> Backend: ARM64 inline assembly (MUL/UMULH). ~5x faster than generic C++.

---

## License

MIT -- see [LICENSE](../LICENSE)

---

## Credits

**Port**: Direct CUDA adaptation of the C++ library  
**Focus**: Maximum throughput for batch ECC operations  
**Philosophy**: Speed > Security (research/development)

---

*UltrafastSecp256k1 v3.0.0 -- CUDA/ROCm GPU Library*
