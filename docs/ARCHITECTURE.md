# Architecture

**UltrafastSecp256k1 v3.12.1** -- Technical Architecture for Auditors

---

## System Diagram

```
+-----------------------------------------------------------------+
|                     Application Layer                           |
|  (Wallet, Signer, Verifier, Key Manager, Address Generator)     |
+-----------------------------------------------------------------+
|                     Protocol Layer                              |
|  ECDSA (RFC 6979) | Schnorr (BIP-340) | MuSig2 | FROST        |
|  Adaptor Sigs     | Pedersen Commit    | Taproot| HD (BIP-32)  |
+-----------------------------------------------------------------+
|                  Dispatch / Utility Layer                        |
|  27-Coin Dispatch | SHA-256 | RIPEMD-160 | Batch Inverse       |
+-----------------------------------------------------------------+
|                  Core Arithmetic Layer                          |
|  +----------------------+----------------------+               |
|  |  FAST (variable-time)|  CT (constant-time)  |               |
|  |  secp256k1::fast::   |  secp256k1::ct::     |               |
|  |  +----------------+  |  +----------------+  |               |
|  |  | FieldElement   |  |  | ct::FieldOps   |  |               |
|  |  | Scalar         |  |  | ct::ScalarOps  |  |               |
|  |  | Point (Jac/Aff)|  |  | ct::Point      |  |               |
|  |  | GLV Endo.      |  |  | ct::scalar_mul |  |               |
|  |  | Hamburg Comb    |  |  | ct::gen_mul    |  |               |
|  |  +----------------+  |  +----------------+  |               |
|  +----------------------+----------------------+               |
+-----------------------------------------------------------------+
|                  Platform Backend Layer                         |
|  x86-64 BMI2/ADX | ARM64 MUL/UMULH | RISC-V RV64GC           |
|  CUDA PTX        | ROCm/HIP        | OpenCL                   |
|  Metal           | WASM            | Xtensa (ESP32)           |
+-----------------------------------------------------------------+
```

---

## Field Element Representation

The fundamental data type. All higher-level operations build on field arithmetic.

```
FieldElement: 4 x uint64_t limbs (little-endian)

  limbs[0]   limbs[1]   limbs[2]   limbs[3]
  +--------+--------+--------+--------+
  | [0:63] |[64:127]|[128:191]|[192:255]|  = 256 bits total
  +--------+--------+--------+--------+
  LSB                              MSB

Prime p = 2^256 - 2^32 - 977
       = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F

Reduction: After arithmetic, normalize() ensures 0 <= result < p
           by checking if limbs >= PRIME and subtracting if needed.
```

### Key Files

| File | Purpose |
|------|---------|
| `cpu/include/secp256k1/field.hpp` | Class declaration, `from_limbs`, `from_bytes` |
| `cpu/src/field.cpp` | `add_impl`, `sub_impl`, `mul_impl`, `square_impl`, `normalize` |
| `cpu/include/secp256k1/field_branchless.hpp` | `field_select` -- branchless cmov |

### MidFieldElement (32-bit View)

```cpp
struct MidFieldElement {
    uint32_t limbs[8];   // Same memory, 32-bit interpretation
};
// sizeof(MidFieldElement) == sizeof(FieldElement) == 32 bytes
```

Zero-cost reinterpretation for operations where 32-bit multiplication is faster (~1.10x on some uarch). Memory layout is identical.

### Endianness Convention

| Function | Byte Order | Use Case |
|----------|------------|----------|
| `from_limbs()` | **Little-endian** (native x86/64) | Internal binary I/O, database, index files |
| `from_bytes()` | **Big-endian** (standard crypto) | Hex strings, test vectors, interop |
| `to_bytes()` | Big-endian output | Serialization for external consumers |

**Rule**: `from_limbs` is the PRIMARY function. Use `from_bytes` only for standard vectors.

---

## Scalar Representation

```
Scalar: 4 x uint64_t limbs (little-endian)

Order n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

Represented as 4x64-bit limbs. All operations reduce mod n.
Scalar::zero(), Scalar::one(), inverse via SafeGCD or Fermat exponentiation.
```

---

## Point Representations

### Jacobian Coordinates (default for computation)

```
(X, Y, Z) where affine (x, y) = (X/Z^2, Y/Z^3)

Advantages:
  - Addition: no inversion needed
  - Doubling: no inversion needed
  - Only need inversion when converting back to affine

Memory: 3 x FieldElement = 96 bytes
```

### Affine Coordinates (for storage/lookup)

```
(x, y) -- direct curve point

Memory: 2 x FieldElement = 64 bytes
Used for: precomputed tables, serialization, final output
```

---

## Scalar Multiplication Strategies

### FAST Layer: GLV Endomorphism + Windowed

```
scalar_mul(P, k):
  1. GLV decompose: k -> k1 + k2*lambda (mod n)
     where lambda^3 == 1 (mod n), beta^3 == 1 (mod p)
     and P' = (beta*x, y) satisfies k2*P' computation
  2. Both k1, k2 are ~128 bits (half the scalar width)
  3. Windowed simultaneous evaluation of k1*P + k2*P'
  
  Result: ~2x speedup over naive double-and-add
```

### FAST Layer: Hamburg Signed-Digit Comb (Generator)

```
generator_mul(k):
  1. Transform: v = (k + 2^256 - 1) / 2 mod n
  2. Every 4-bit window yields guaranteed odd digit
  3. Precomputed table: 8 entries per window group
  4. Cost: 64 unified_add + 64 signed_lookups(8)
  5. No doublings needed (comb structure handles it)
  
  ~3x faster than generic scalar_mul(G, k)
```

### CT Layer: GLV + Signed-Digit

```
ct::scalar_mul(P, k):
  1. k -> (k + K) / 2, GLV split -> v1, v2 (~129 bits each)
  2. 26 groups of 5 bits, each -> non-zero odd digit
  3. Table: 16 odd multiples per curve ([1P..31P], [1lambdaP..31lambdaP])
  4. Cost: 125 dbl + 52 unified_add + 52 signed_lookups(16)
  5. ALL operations are constant-time (no branches on secret bits)

ct::generator_mul(k):
  - Hamburg signed-digit encoding
  - Cost: 64 unified_add + 64 signed_lookups(8)
  - No doublings, no cmov-skip
```

---

## Field Inversion

Two primary algorithms:

### SafeGCD (Bernstein-Yang divsteps)

```
Default on platforms with __int128:
  fe_inverse_safegcd_impl(x)  -- 62-bit divsteps
  ~3x faster than binary EEA for secp256k1

Fallback (no __int128):
  field_safegcd30::inverse_impl(x)  -- 30-bit divsteps
  ~130us on ESP32 vs ~3ms Fermat chain
```

### Fermat's Little Theorem (multiple strategies)

```
a^(-1) = a^(p-2) mod p

17 variants implemented for benchmarking/research:
  - Binary, Window-4, Addchain, EEA, K-ary-16, Fixed-Window-5
  - RTL Binary, Sliding Dynamic, Fermat GPU, Montgomery REDC
  - Branchless, Parallel Window, Binary Euclidean
  - Lehmer, Stein, Warp-Optimized, Double-Base, Compact Table

Active inversion: selected by compile-time SECP256K1_FE_INV_METHOD
Default: SafeGCD (most platforms), Addchain (ESP32)
```

### Batch Inversion (Montgomery Trick)

```
fe_batch_inverse(elements[], count):
  Cost: 1 inversion + 3*(count-1) multiplications
  For N=8: ~8us instead of ~28us (3.5x speedup)
  Sweep-tested up to 8192 elements
```

---

## Platform Assembly Backends

| Platform | File | Key Operations |
|----------|------|----------------|
| x86-64 | `field_asm_x64.asm` | BMI2 `MULX`, ADX `ADCX`/`ADOX` for carry-free mul |
| ARM64 | `field_asm_arm64.cpp` | `MUL`/`UMULH` intrinsics for 64x64->128 |
| RISC-V | `field_asm_riscv64.S` | `MUL`/`MULHU` for 64x64->128 |
| ESP32 | `field.cpp` (generic) | 32-bit portable path |

Assembly dispatch is compile-time: preprocessor selects the optimal path based on `__x86_64__`, `__aarch64__`, `__riscv`, or falls back to portable C++.

---

## GPU Architecture

### CUDA

```
cuda/
+-- include/
|   +-- secp256k1.cuh           -- All device functions
|   +-- ptx_math.cuh            -- PTX inline asm (with __int128 fallback)
|   +-- gpu_compat.h            -- CUDA <-> HIP API mapping
|   +-- batch_inversion.cuh     -- Montgomery trick on GPU
|   +-- bloom.cuh               -- Device-side Bloom filter
|   +-- hash160.cuh             -- SHA-256 + RIPEMD-160
+-- app/                        -- Search kernels
+-- src/                        -- Kernel wrappers, tests
```

**GPU Contract**:
- No dynamic allocation in device hot loops
- No per-iteration host/device sync
- Launch parameters derived from config.json
- NOT constant-time -- for public-data workloads only

### OpenCL

```
opencl/kernels/
+-- secp256k1_field.cl          -- Field arithmetic
+-- secp256k1_extended.cl       -- GLV, signatures
+-- ...
```

### Metal

```
metal/shaders/
+-- secp256k1_field.h           -- 8x32-bit limbs (Metal uint)
+-- ...
```

**Note**: Metal uses 8x32-bit limbs (vs 4x64-bit on CPU) due to Metal Shading Language constraints.

---

## Memory Model

### Hot Path Contract

```
MUST:
  OK Allocation-free hot paths
  OK Explicit buffers (out*, in*, scratch*)
  OK Fixed-size POD types
  OK In-place mutation only
  OK Deterministic memory layout
  OK alignas(32/64) where applicable

NEVER:
  X Heap allocation (new, malloc, push_back, resize)
  X Exceptions / RTTI / virtual calls
  X Strings / iostreams / formatting
  X Hidden temporaries
  X % or / (use Montgomery/Barrett)
```

### Scratchpad Pattern

```
Single allocation -> full reuse
Thread-local scratch on CPU
Pointer-based reset (no memset in loops)
Caller owns all buffers
```

---

## Signature Schemes

### ECDSA (RFC 6979)

```
sign(hash, privkey):
  1. k = RFC6979_nonce(hash, privkey)    -- deterministic
  2. R = k*G
  3. r = R.x mod n
  4. s = k^(-1) * (hash + r*privkey) mod n
  5. return (r, s)

verify(hash, pubkey, r, s):
  1. w = s^(-1) mod n
  2. u1 = hash * w mod n
  3. u2 = r * w mod n
  4. R' = u1*G + u2*pubkey
  5. return R'.x == r
```

### Schnorr (BIP-340)

```
sign(hash, privkey):
  1. d = privkey (adjusted for even y)
  2. aux = tagged_hash("BIP0340/aux", rand)
  3. t = d XOR aux
  4. k = tagged_hash("BIP0340/nonce", t || pubkey || hash)
  5. R = k*G (ensure even y)
  6. e = tagged_hash("BIP0340/challenge", R.x || pubkey || hash)
  7. s = k + e*d mod n
  8. return (R.x, s)
```

### Experimental: MuSig2 / FROST / Adaptor

- **MuSig2**: 2-round multi-signature (Musig2 paper)
- **FROST**: Threshold signature (t-of-n)
- **Adaptor**: Signature adaptors for atomic swaps

All marked **Experimental** -- APIs may change, limited test coverage.

---

## Build System

```
CMakeLists.txt
+-- lib: UltrafastSecp256k1 (STATIC)
|   +-- cpu/src/*.cpp
|   +-- platform-specific ASM (conditional)
|   +-- Public headers in cpu/include/
+-- tests/ (CTest targets)
+-- bench/ (benchmark targets)
+-- fuzz/ (libFuzzer targets, clang only)
+-- cuda/ (optional, requires CUDA toolkit)
+-- opencl/ (optional, requires OpenCL SDK)
+-- wasm/ (optional, requires Emscripten)

Key CMake Options:
  -DCMAKE_BUILD_TYPE=Release       -- Optimized build
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined"  -- Sanitizer build
  -DSECP256K1_USE_ROCKSDB=ON       -- Enable RocksDB-dependent tools
  -DSECP256K1_SPEED_FIRST=ON       -- Aggressive speed optimizations
  -DCMAKE_CUDA_ARCHITECTURES=86;89 -- CUDA target architectures
```

---

## Data Flow: Sign -> Verify

```
+---------+    +----------+    +----------+    +----------+
| Message  |---->| SHA-256  |---->|  Sign    |---->| (r, s)   |
| (bytes)  |    | hash()   |    | ECDSA/   |    | signature|
+---------+    +----------+    | Schnorr  |    +----------+
                               +----------+
                                    |
                                    ▼
                              +----------+
                              |  privkey  | (Scalar)
                              |  -> k*G   | (RFC 6979 nonce)
                              |  -> r, s  | (signature components)
                              +----------+

Verification:
+----------+   +----------+   +----------+   +------+
| (r, s)   |--->| Verify   |--->| u1*G +   |--->| bool |
| + hash   |   | decompose|   | u2*pubkey|   | pass |
| + pubkey |   | u1, u2   |   | ?= R     |   +------+
+----------+   +----------+   +----------+
```

---

## Security Boundaries

```
+---------------------------------------------+
|            THIS LIBRARY CONTROLS            |
|                                             |
|  OK Arithmetic correctness (F_p, Z_n, E)    |
|  OK CT layer timing properties               |
|  OK Deterministic nonce generation           |
|  OK Input validation (on-curve, range)       |
|  OK Memory layout (no hidden alloc)          |
|  OK Platform dispatch (ASM selection)        |
+---------------------------------------------+

+---------------------------------------------+
|          CALLER RESPONSIBILITY              |
|                                             |
|  X Key storage and lifecycle                |
|  X Buffer zeroing after use                 |
|  X FAST vs CT selection                     |
|  X Network security / transport             |
|  X Entropy source (if randomness needed)    |
|  X GPU memory isolation                     |
+---------------------------------------------+
```

---

*UltrafastSecp256k1 v3.12.1 -- Architecture*
