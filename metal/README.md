# UltrafastSecp256k1 — Apple Metal Backend

**პირველი secp256k1 ბიბლიოთეკა Apple Metal GPU მხარდაჭერით.**

Metal backend-ი უზრუნველყოფს secp256k1 ელიფსური მრუდის ოპერაციებს Apple Silicon GPU-ზე
(M1, M2, M3, M4 და სხვა) Metal Shading Language (MSL) გამოყენებით.

---

## სწრაფი დაწყება (M1/M2/M3/M4 MacBook-ზე)

### წინაპირობები

```bash
# Xcode Command Line Tools (თუ არ გაქვთ)
xcode-select --install

# CMake + Ninja (Homebrew-ით)
brew install cmake ninja

# ვერიფიკაცია
cmake --version   # 3.21+
xcrun metal --version   # Metal compiler
```

### აშენება და ტესტირება (ყველა ბრძანება ერთად)

```bash
# 1. კლონირება
git clone https://github.com/shrec/Secp256K1fast.git
cd UltrafastSecp256k1

# 2. კონფიგურაცია Metal-ით
cmake -S . -B build_metal -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DSECP256K1_BUILD_METAL=ON

# 3. აშენება
cmake --build build_metal -j

# 4. ყველა ტესტის გაშვება (host + GPU)
ctest --test-dir build_metal --output-on-failure
```

### მხოლოდ GPU ტესტების/ბენჩმარკების გაშვება

```bash
# GPU ტესტები (G×1, G×2, G×3 ვერიფიკაცია + field_mul check)
./build_metal/metal/metal_secp256k1_test

# GPU ბენჩმარკი (field_mul 1M ops, scalar_mul 4K ops)
./build_metal/metal/metal_secp256k1_test --bench
```

### მხოლოდ Host ტესტების გაშვება (GPU-ს გარეშე)

```bash
./build_metal/metal/metal_host_test
# მოსალოდნელი: "Results: 76 passed, 0 failed"
```

---

## არქიტექტურა

### 8×32-ბიტიანი ლიმბების მოდელი (Shader-ებში)

Metal Shading Language არ უჭერს მხარს 64-ბიტიან მთელ რიცხვებს (`uint64_t`) shader ფუნქციებში.
CUDA backend იყენებს 4×64-ბიტიან ლიმბებს PTX ინლაინ ასემბლით, ხოლო **Metal shader-ები იყენებენ
8×32-ბიტიან ლიმბებს** explicit carry propagation-ით `ulong` (64-bit) ტიპის დროებითი ცვლადებით.

**Host-side ტიპები** (`host_helpers.h`) იყენებენ `uint64_t limbs[4]` — ზუსტად ისეთივე, როგორც
CUDA-ს `HostFieldElement` და shared `FieldElementData` (`types.hpp`). ეს უზრუნველყოფს cross-backend
თავსებადობას. ბუფერების I/O zero-cost-ია, რადგან `FieldElementData{uint64_t[4]}` და
`MidFieldElementData{uint32_t[8]}` ერთი და იგივე 32 ბაიტია little-endian-ზე.

| Backend | Shader ლიმბი | Host ლიმბი | Carry მეთოდი |
|---------|-------------|------------|--------------|
| CUDA    | 64-bit (4)  | 64-bit (4) | PTX addc     |
| Metal   | 32-bit (8)  | 64-bit (4) | ulong cast   |
| OpenCL  | 64-bit (4)  | 64-bit (4) | mul_hi()     |

### Apple Silicon Unified Memory

Apple Silicon-ის unified memory არქიტექტურა საშუალებას იძლევა zero-copy ბუფერების
გამოყენება (`MTLResourceStorageModeShared`), რაც გამორიცხავს ექსპლიციტურ host↔device
მონაცემთა კოპირებას.

---

## ფაილების სტრუქტურა

```
metal/
├── CMakeLists.txt              # Build configuration
├── README.md                   # ეს ფაილი
├── shaders/
│   ├── secp256k1_field.h       # ველის არითმეტიკა (add, sub, mul, sqr, inv)
│   ├── secp256k1_point.h       # წერტილის ოპერაციები (double, add_mixed, scalar_mul)
│   └── secp256k1_kernels.metal # Compute kernels (search, batch_inverse, benchmarks)
├── include/
│   ├── gpu_compat_metal.h      # პლატფორმის მაკროსები (CUDA gpu_compat.h pattern)
│   ├── metal_runtime.h         # C++ ინტერფეისი (PIMPL, Obj-C types hidden)
│   └── host_helpers.h          # Host-side ტიპები (uint64_t[4]), types.hpp integration
├── src/
│   └── metal_runtime.mm        # Objective-C++ runtime (ARC, pipeline caching)
└── app/
    └── metal_test.mm           # ტესტები + ბენჩმარკები
```

---

## იმპლემენტირებული ოპერაციები

### ველის არითმეტიკა (`secp256k1_field.h`)
- `field_add` — მოდულარული ჯამი, branchless (mod p)
- `field_sub` — მოდულარული გამოკლება, branchless (mod p)
- `field_negate` — მოდულარული უარყოფა
- `field_mul` — **Comba product scanning** (CUDA PTX MAD_ACC ექვივალენტი, column-by-column accumulation)
- `field_sqr` — **Comba + სიმეტრიის ოპტიმიზაცია** (36 multiply ნაცვლად 64-ისა)
- `field_reduce_512` — 512→256 bit რედუქცია K = 0x1000003D1, branchless final subtract
- `field_inv` — Fermat ინვერსია (a^(p-2) mod p, 255 sqr + 14 mul chain)
- `field_sqr_n` — მრავალჯერადი კვადრატი (sqr ×N)
- `field_mul_small` — გამრავლება სკალარზე (< 2^32), branchless რედუქცია
- `METAL_MAD_ACC` — PTX `mad.lo.cc.u64/madc.hi.cc.u64/addc.u64` მაკროს ექვივალენტი

### წერტილის ოპერაციები (`secp256k1_point.h`)
- `jacobian_double` — dbl-2001-b (3M + 4S)
- `jacobian_add_mixed` — madd-2007-bl (7M + 4S)
- `jacobian_add` — სრული Jacobian ჯამი (11M + 5S)
- `scalar_mul` — **4-bit fixed window** (64 double + 64 add, ~35% სწრაფი ვიდრე naive)
- `affine_select` — **branchless** table წაკითხვა (GPU divergence-ს არ იწვევს)
- `jacobian_to_affine` — Jacobian → Affine კონვერსია
- `apply_endomorphism` — GLV ენდომორფიზმი (β·x mod p)

### Compute Kernels (`secp256k1_kernels.metal`)
- `search_kernel` — ძიების მთავარი kernel (**O(1) per-thread** offset, scalar_mul)
- `scalar_mul_batch` — სკალარული გამრავლების ბეჩი (4-bit windowed)
- `generator_mul_batch` — გენერატორის წერტილზე გამრავლება (4-bit windowed)
- `field_mul_bench` — ველის გამრავლების ბენჩმარკი (Comba)
- `field_sqr_bench` — ველის კვადრატის ბენჩმარკი (Comba + symmetry)
- `batch_inverse` — **Chunked** Montgomery batch ინვერსია (parallel threadgroups)
- `point_add_kernel` — წერტილების ჯამი
- `point_double_kernel` — წერტილის გაორმაგება

---

## აშენება (Build) — დეტალური

სწრაფი build ინსტრუქცია იხ. ზემოთ „სწრაფი დაწყება" სექციაში.

### Shader კომპილაცია

CMake ავტომატურად ახდენს shader-ების კომპილაციას:
1. `.metal` → `.air` (xcrun metal -O2 -std=metal2.4)
2. `.air` → `.metallib` (xcrun metallib)

Runtime fallback: თუ `.metallib` ფაილი ვერ მოიძებნება, runtime ავტომატურად
კომპილირებს `.metal` სორს ფაილს.

---

## გამოყენება

### C++ API (metal_runtime.h)

```cpp
#include "metal_runtime.h"
#include "host_helpers.h"

// Runtime ინიციალიზაცია
secp256k1::metal::MetalRuntime runtime;
runtime.init();

// Shader ბიბლიოთეკის ჩატვირთვა
runtime.load_library_from_path("secp256k1_kernels.metallib");

// Pipeline შექმნა
auto pipeline = runtime.make_pipeline("generator_mul_batch");

// ბუფერების ალოკაცია (zero-copy unified memory)
auto scalars_buf = runtime.alloc_buffer_shared(n * sizeof(HostScalar));
auto points_buf  = runtime.alloc_buffer_shared(n * sizeof(HostAffinePoint));

// Kernel dispatch
runtime.dispatch_1d(pipeline, n, /* threadgroup_size */ 256,
    {scalars_buf, points_buf});
runtime.synchronize();
```

---

## შესრულების მახასიათებლები

### Apple Silicon GPU სპეციფიკა
- 32-bit ALU throughput: ძალიან მაღალი (Metal ოპტიმიზირებულია 32-bit ოპერაციებზე)
- Unified memory: ნულოვანი კოპირების ხარჯი
- Threadgroup memory: 32KB per threadgroup (M1/M2), 64KB (M3/M4)
- Max threads per threadgroup: 1024

### მოსალოდნელი წარმადობა
| ოპერაცია | M1 (est.) | M2 (est.) | M3 Pro (est.) |
|----------|-----------|-----------|---------------|
| field_mul | ~300M/s | ~400M/s | ~550M/s |
| scalar_mul | ~150K/s | ~200K/s | ~300K/s |

> შენიშვნა: ეს მიახლოებითი შეფასებებია. რეალური ბენჩმარკები `metal_test` აპლიკაციაში.

---

## მხარდაჭერილი მოწყობილობები

| მოწყობილობა | GPU Family | მხარდაჭერა |
|-------------|------------|------------|
| M1 / M1 Pro / M1 Max / M1 Ultra | Apple7 | ✅ |
| M2 / M2 Pro / M2 Max / M2 Ultra | Apple8 | ✅ |
| M3 / M3 Pro / M3 Max | Apple9 | ✅ |
| M4 / M4 Pro / M4 Max | Apple9+ | ✅ |
| A14+ (iPhone/iPad) | Apple7+ | ✅ |
| Apple Vision Pro | Apple9 | ✅ |

---

## CUDA-სთან თავსებადობა

Metal backend იყენებს CUDA backend-ის იდენტურ ალგორითმებს:
- იგივე Fermat inversion chain (x2→x3→x6→x9→x11→x22→x44→x88→x176→x220→x223→tail)
- იგივე Jacobian ფორმულები (dbl-2001-b, madd-2007-bl)
- იგივე bloom filter ჰეშ ფუნქციები (FNV-1a + SplitMix64)
- იგივე Montgomery batch inversion
- **Comba product scanning** — PTX `mad.lo.cc.u64 / madc.hi.cc.u64 / addc.u64`-ის ექვივალენტური `METAL_MAD_ACC` მაკრო
- **4-bit windowed scalar_mul** — CUDA-ს wNAF/fixed-window-ს შესატყვისი

ლიმბის ზომა: 4×64 → 8×32 (shader-ებში), მათემატიკური კორექტულობა იდენტურია.

---

## ამაჩქარების სტრატეგია (Assembly-ის ნაცვლად)

CUDA იყენებს PTX inline assembly-ს hardware carry-chain-ისთვის. Metal-ს **არ აქვს** inline
assembly — Apple GPU ISA დახურულია. სამაგიეროდ:

| CUDA PTX | Metal ექვივალენტი | რას აკეთებს |
|----------|-------------------|-------------|
| `mad.lo.cc.u64` | `METAL_MAD_ACC` macro | 96-bit accumulator column accumulation |
| `madc.hi.cc.u64` | `ulong(a)*ulong(b)` compiler MAC fusion | Hardware MAC instruction mapping |
| `addc.u64` | Explicit carry propagation | Apple Silicon compiler optimizes this |
| wNAF scalar mul | 4-bit fixed window | Precomputed table[16] + branchless select |
| `__syncthreads()` | Chunked threadgroups | Each threadgroup = independent batch |

---

## ლიცენზია

იგივე ლიცენზია, როგორც მთავარი UltrafastSecp256k1 პროექტი.
