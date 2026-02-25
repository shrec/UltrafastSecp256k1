# RISC-V Linux Benchmarks

## Available Results

### Milk-V Mars (StarFive JH7110)

- **File:** `milkv-mars-20260208.txt`
- **Platform:** RISC-V 64-bit RV64GC
- **CPU:** StarFive JH7110 @ 1.5 GHz (4x U74 cores)
- **Features:** Assembly + RVV Vector Extension + Fast Modular Reduction
- **Compiler:** Clang 19.1.7
- **Date:** 2026-02-11

#### Performance Summary
| Operation | Time |
|-----------|------|
| Field Multiplication | 200 ns |
| Point Scalar Multiply | 665 us |
| Generator Multiply | 44 us |
| Batch Inverse (1000) | 611 ns/element |

OK All 29/29 self-tests passed

---

## Contributing Results

To submit benchmark results for other RISC-V boards:

```bash
# Build with benchmarks enabled
cmake -B build -DCMAKE_BUILD_TYPE=Release -DSECP256K1_BUILD_BENCH=ON
cmake --build build -j

# Run comprehensive benchmark
./build/libs/UltrafastSecp256k1/cpu/bench_comprehensive_riscv > results.txt

# Name format: <board>-<date>.txt
# e.g., visionfive2-20260208.txt
```

Please include:
- CPU model and frequency
- RISC-V extensions enabled (G, V, C, etc.)
- Compiler version and flags
- OS/kernel version

