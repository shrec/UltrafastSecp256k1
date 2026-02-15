# UltrafastSecp256k1 CUDA — GPU ECC Library

სრული secp256k1 ECC ბიბლიოთეკა NVIDIA GPU-სთვის — header-only ბირთვი PTX inline assembly-ით.

**პრიორიტეტი**: მაქსიმალური throughput batch ოპერაციებისთვის. Side-channel დაცვა არ არის (კვლევა/dev გამოყენებისთვის).

---

## არქიტექტურა

კოდი მთლიანად `secp256k1::cuda` namespace-შია. ბირთვი **header-only** — `secp256k1.cuh` მოიცავს ყველა device ფუნქციას. მონაცემთა ტიპები CPU ბიბლიოთეკასთან ურთიერთთავსებადია (`secp256k1/types.hpp`-ის POD სტრუქტურები).

### Compile-Time კონფიგურაცია (3 backend)

| Macro | Default | აღწერა |
|-------|---------|--------|
| `SECP256K1_CUDA_USE_HYBRID_MUL` | **ON** | 32-bit Comba mul + 64-bit reduction (1.10× ჩქარი) |
| `SECP256K1_CUDA_USE_MONTGOMERY` | OFF | Montgomery residue domain (mont_reduce_512) |
| `SECP256K1_CUDA_LIMBS_32` | OFF | სრულიად 8×32-bit limbs (სეპარატული backend) |

**Default path** (64-bit hybrid): `field_mul` → `field_mul_hybrid` → 32-bit Comba PTX → `reduce_512_to_256`

---

## ფუნქციონალი

### Field არითმეტიკა (Fp)
- **add/sub**: PTX inline asm carry chain-ებით (ADDC.CC/SUBC.CC)
- **mul**: 32-bit Comba hybrid → 64-bit secp256k1 fast reduction (P = 2²⁵⁶ − 2³² − 977)
- **sqr**: ოპტიმიზებული squaring (cross-product doubling)
- **inverse**: Fermat chain `a^{p-2}` (255 sqr + 16 mul)
- **mul_small**: uint32-ზე გამრავლება (reduction constant-ისთვის)
- **Montgomery**: `field_to_mont`, `field_from_mont`, `mont_reduce_512` (optional backend)

### Scalar არითმეტიკა (Fn)
- **add/sub**: Modular arithmetic mod curve order N
- **bit extraction**: Fast bit access scalar processing-ისთვის

### Point ოპერაციები (Jacobian კოორდინატები)
- **doubling**: `dbl-2001-b` (3M+4S, a=0 curves)
- **mixed addition**: 6 ვარიანტი ოპტიმიზებული სხვადასხვა სცენარისთვის:
  - `jacobian_add_mixed` — madd-2007-bl (7M+4S) ზოგადი
  - `jacobian_add_mixed_h` — madd-2004-hmv (8M+3S), H output batch inversion-ისთვის
  - `jacobian_add_mixed_h_z1` — Z=1 სპეციალიზებული (5M+2S), პირველი ნაბიჯი
  - `jacobian_add_mixed_const` — branchless (8M+3S), constant-point
  - `jacobian_add_mixed_const_7m4s` — branchless 7M+4S + 2H output
- **general add**: `jacobian_add` (11M+5S, Jacobian + Jacobian)
- **GLV endomorphism**: `apply_endomorphism` φ(x,y) = (β·x, y)

### Scalar Multiplication
- **double-and-add**: მარტივი, რეგისტრების ეფექტური (GPU-ზე wNAF ძვირია რეგისტრ-pressure-ის გამო)
- **Batch kernels**: `scalar_mul_batch_kernel`, `generator_mul_batch_kernel`

### Batch Inversion
- **Montgomery trick**: prefix/suffix scan (ნაგულისხმევი, ერთი inversion N ელემენტისთვის)
- **Fermat**: `a^{p-2}` თითოეულისთვის (fallback)
- **naive**: პირდაპირი GCD (debug/reference)

### Hash160 (SHA-256 + RIPEMD-160)
- `hash160_pubkey_kernel` — pubkey → Hash160 device-side

### Bloom Filter
- `DeviceBloom` — FNV-1a + SplitMix хეშირებით
- `test` / `add` device ფუნქციები + batch kernels

### Search Kernels (ექსპერიმენტული)
- `search_simple.cuh` — პროტოტიპი (naive per-thread loop)
- `search_cpu_identical.cuh` — CPU-identical incremental add algorithm (init → add → batch_inv → bloom)

> **შენიშვნა**: Search kernels ექსპერიმენტულია. პროდაქშენ search app სეპარატულად არის: `Secp256K1fast/apps/secp256k1_search_gpu/`

---

## ფაილთა სტრუქტურა

```
cuda/
├── CMakeLists.txt                              # Build: lib + test + bench
├── README.md
├── include/
│   ├── secp256k1.cuh                           # ბირთვი — field/point/scalar device ფუნქციები (1800+ ხაზი)
│   ├── ptx_math.cuh                            # PTX inline asm (256×256→512 Comba multiply)
│   ├── secp256k1_32.cuh                        # Alternative: 8×32-bit limbs + Montgomery backend
│   ├── secp256k1_32_hybrid_final.cuh           # 32-bit Comba mul → 64-bit reduction (default mul path)
│   ├── batch_inversion.cuh                     # Montgomery trick / Fermat / naive batch inverse
│   ├── bloom.cuh                               # Device-side Bloom filter (FNV-1a + SplitMix)
│   ├── hash160.cuh                             # SHA-256 + RIPEMD-160 → Hash160
│   ├── host_helpers.cuh                        # Host-side wrappers (1-thread kernels, test-only)
│   ├── search_simple.cuh                       # Search prototype (experimental)
│   └── search_cpu_identical.cuh                # CPU-identical search algorithm (experimental)
├── src/
│   ├── secp256k1.cu                            # Kernel definitions (thin wrappers)
│   ├── test_suite.cu                           # 30 vector tests
│   └── bench_cuda.cu                           # Benchmark harness
```

---

## Build

```bash
# Parent CMakeLists.txt-ით (ან standalone)
cmake -S cuda -B cuda/build -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build cuda/build -j

# ტესტები
./cuda/build/secp256k1_cuda_test

# ბენჩმარკი
./cuda/build/secp256k1_cuda_bench
```

### Build ოფციები

| ოფცია | Default | აღწერა |
|-------|---------|--------|
| `CMAKE_CUDA_ARCHITECTURES` | 89 (Ada) | GPU არქიტექტურა (75/80/86/89/90) |
| `SECP256K1_CUDA_USE_MONTGOMERY` | OFF | Montgomery domain |
| `SECP256K1_CUDA_LIMBS_32` | OFF | 8×32-bit limb backend |

### მოთხოვნები
- CUDA Toolkit 12.0+
- NVIDIA GPU Compute Capability 7.0+ (Volta+)
- CMake 3.18+

---

## გამოყენება

### Device ფუნქციები

```cpp
#include "secp256k1.cuh"

__global__ void my_kernel(const Scalar* scalars, JacobianPoint* results, int n) {
    using namespace secp256k1::cuda;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // G * k — GENERATOR_JACOBIAN კომპილაციის დროს ჩაშენებულია
    JacobianPoint G = GENERATOR_JACOBIAN;
    scalar_mul(&G, &scalars[idx], &results[idx]);
}
```

### Batch Processing

```cpp
#include "secp256k1.cuh"

const int N = 1 << 20;  // ~1M ოპერაცია
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

## ტესტები

30 vector test `test_suite.cu`-ში:
- Field არითმეტიკა: identity, inverse, commutativity, associativity
- Scalar არითმეტიკა: add, sub, boundary
- Point ოპერაციები: doubling, mixed addition, identity
- Scalar multiplication: known vectors, generator mul
- GLV endomorphism: φ(φ(P)) + P = -φ(P)
- Batch inversion: Montgomery trick correctness
- Cross-backend: CPU ↔ CUDA შედეგების შედარება

---

## CPU ↔ CUDA თავსებადობა

მონაცემთა ტიპები იზიარებს layout-ს `secp256k1/types.hpp`-ით:

```cpp
static_assert(sizeof(FieldElement) == 32);
static_assert(sizeof(Scalar) == 32);
static_assert(sizeof(AffinePoint) == 64);
static_assert(offsetof(FieldElement, limbs) == 0);
```

CPU-ზე გამოთვლილი მონაცემები პირდაპირ `cudaMemcpy`-ით გადადის GPU-ზე (little-endian, same POD layout).

---

## ლიცენზია

იგივე რაც მშობელი პროექტი (UltrafastSecp256k1)

---

## კრედიტები

**პორტი**: C++ ბიბლიოთეკის პირდაპირი CUDA ადაპტაცია  
**ფოკუსი**: მაქსიმალური throughput batch ECC ოპერაციებისთვის  
**ფილოსოფია**: სიჩქარე > უსაფრთხოება (კვლევა/development)
