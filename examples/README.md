# UltrafastSecp256k1 Examples

Usage examples for different platforms, languages, and use cases.

## Prerequisites

Build the library first (from repo root):

```bash
cmake -S . -B build-linux -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-linux -j
```

The shared library will be at `build-linux/include/ufsecp/libufsecp.so`.
Headers live in `include/ufsecp/` (`ufsecp.h`, `ufsecp_gpu.h`).

---

## Language Binding Examples

Full-feature examples covering CPU + GPU operations for 6 languages.
Each covers: key generation, ECDSA, Schnorr, ECDH, hashing, Bitcoin addresses,
WIF, BIP-32, Taproot, Pedersen commitments, and GPU batch operations
(backend discovery, batch keygen, ECDSA batch verify, Hash160 batch, MSM).

### C -- [`c_example/`](c_example/)

Direct C ABI usage. Includes all CPU features and GPU batch operations.

**Build & Run:**
```bash
cd examples/c_example

# Compile:
gcc -O2 -o demo main.c \
    -I../../include/ufsecp \
    -L../../build-linux/include/ufsecp -lufsecp \
    -Wl,-rpath,../../build-linux/include/ufsecp

# Run:
./demo
```

**Sections:** 16 (10 CPU + 6 GPU) | **Status:** [STABLE]

---

### Python -- [`python_example/`](python_example/)

Uses `ctypes` to load `libufsecp.so` directly. No pip packages needed.

**Build & Run:**
```bash
cd examples/python_example

# Set library path and run:
UFSECP_LIB=../../build-linux/include/ufsecp/libufsecp.so \
    python3 example.py
```

**Sections:** 15 (10 CPU + 5 GPU) | **Status:** [STABLE]

---

### Rust -- [`rust_example/`](rust_example/)

Uses the `ufsecp` safe wrapper crate for CPU operations and raw FFI
(`ufsecp::ufsecp_sys`) for GPU and Pedersen operations.

**Build & Run:**
```bash
cd examples/rust_example

# Build and run (UFSECP_LIB_DIR tells the sys crate where to find libufsecp):
UFSECP_LIB_DIR=../../build-linux/include/ufsecp \
LD_LIBRARY_PATH=../../build-linux/include/ufsecp \
    cargo run
```

**Dependencies:** `ufsecp` (path dep), `hex`

**Sections:** 15 (10 CPU + 5 GPU) | **Status:** [STABLE]

---

### Node.js -- [`nodejs_example/`](nodejs_example/)

Uses `koffi` FFI library (compatible with Node.js 18+, no native compilation needed).

**Build & Run:**
```bash
cd examples/nodejs_example

# Install FFI library:
npm install

# Run:
UFSECP_LIB=../../build-linux/include/ufsecp/libufsecp.so \
    node example.js
```

**Dependencies:** `koffi` (npm)

**Sections:** 12 (7 CPU + 5 GPU) | **Status:** [STABLE]

---

### Go -- [`go_example/`](go_example/)

Pure cgo FFI -- calls the C ABI directly via `#cgo` directives. No Go binding
package needed.

**Build & Run:**
```bash
cd examples/go_example

# Set cgo flags and run:
CGO_CFLAGS="-I../../include/ufsecp" \
CGO_LDFLAGS="-L../../build-linux/include/ufsecp -lufsecp -Wl,-rpath,../../build-linux/include/ufsecp" \
    go run example.go
```

**Dependencies:** None (cgo only)

**Sections:** 15 (10 CPU + 5 GPU) | **Status:** [STABLE]

---

### Java -- [`java_example/`](java_example/)

Uses JNA (Java Native Access) for FFI. Downloads `jna.jar` from Maven Central
if not present.

**Build & Run:**
```bash
cd examples/java_example

# Download JNA if needed:
[ -f jna.jar ] || curl -sL -o jna.jar \
    https://repo1.maven.org/maven2/net/java/dev/jna/jna/5.14.0/jna-5.14.0.jar

# Compile:
javac -cp jna.jar Example.java

# Run:
java -cp .:jna.jar \
    -Djna.library.path=../../build-linux/include/ufsecp \
    Example
```

**Dependencies:** JNA 5.14+ (jar)

**Sections:** 15 (10 CPU + 5 GPU) | **Status:** [STABLE]

---

## C++ Desktop Examples

Built automatically by CMake (linked against `fastsecp256k1` internal target).

### basic_usage -- Core API Demo [STABLE]

**Location:** [`basic_usage/`](basic_usage/)

Key generation, ECDSA signing/verification, Schnorr signing/verification,
field arithmetic.

### signing_demo -- ECDSA + Schnorr Signing [STABLE]

**Location:** [`signing_demo/`](signing_demo/)

End-to-end signing and verification demo.

### threshold_demo -- Threshold Signatures [STABLE]

**Location:** [`threshold_demo/`](threshold_demo/)

Threshold signature scheme demonstration.

---

## Embedded Platform Examples

### esp32_test -- ESP32-S3 Selftest & Benchmark [STABLE]

**Location:** [`esp32_test/`](esp32_test/)

Complete ESP32-S3 example: self-test, field arithmetic benchmarks,
point multiplication performance. 28/28 tests pass.

```bash
cd esp32_test
idf.py set-target esp32s3
idf.py build
idf.py flash monitor
```

See [esp32_test/README.md](esp32_test/README.md) for setup details.

### esp32_bench_hornet -- ESP32-S3 bench_hornet [STABLE]

**Location:** [`esp32_bench_hornet/`](esp32_bench_hornet/)

Full bench_hornet benchmark suite: 6-operation comparison vs libsecp256k1,
CT and FAST mode, block validation simulation.

```bash
cd esp32_bench_hornet
idf.py set-target esp32s3
idf.py build
idf.py flash monitor
```

### esp32c6_bench_hornet -- ESP32-C6 bench_hornet [STABLE]

**Location:** [`esp32c6_bench_hornet/`](esp32c6_bench_hornet/)

### esp32p4_bench_hornet -- ESP32-P4 bench_hornet [STABLE]

**Location:** [`esp32p4_bench_hornet/`](esp32p4_bench_hornet/)

### stm32_test -- STM32 Embedded Port [EXPERIMENTAL]

**Location:** [`stm32_test/`](stm32_test/)

STM32F103 (Cortex-M3) port. Runs core field arithmetic and point operations
on extremely constrained hardware (72 MHz, 20 KB SRAM).

See [stm32_test/README.md](stm32_test/README.md) for wiring and flashing.

---

## Feature Coverage Matrix

| Feature | C | Python | Rust | Node.js | Go | Java |
|---------|:-:|:------:|:----:|:-------:|:--:|:----:|
| Key Generation | + | + | + | + | + | + |
| ECDSA Sign/Verify | + | + | + | + | + | + |
| Schnorr (BIP-340) | + | + | + | + | + | + |
| ECDH | + | + | + | + | + | + |
| Hashing (SHA-256, Hash160) | + | + | + | + | + | + |
| Bitcoin Addresses | + | + | + | + | + | + |
| WIF Encoding | + | + | + | + | + | + |
| BIP-32 HD Keys | + | + | + | + | + | + |
| Taproot (BIP-341) | + | + | + | + | + | + |
| Pedersen Commitments | + | + | + | - | + | + |
| GPU Backend Discovery | + | + | + | + | + | + |
| GPU Batch Keygen | + | + | + | + | + | + |
| GPU ECDSA Batch Verify | + | + | + | + | + | + |
| GPU Hash160 Batch | + | + | + | + | + | + |
| GPU MSM | + | + | + | + | + | + |

## Troubleshooting

**Library not found at runtime:**
```
error while loading shared libraries: libufsecp.so.3: cannot open
```
Set `LD_LIBRARY_PATH` to point to the build output directory:
```bash
export LD_LIBRARY_PATH=/path/to/build-linux/include/ufsecp
```

**GPU backends show `available=0`:**
- CUDA: install NVIDIA driver 535+ and verify with `nvidia-smi`
- OpenCL: install `ocl-icd-opencl-dev` and an OpenCL runtime (e.g., `nvidia-opencl-icd`)

**Rust build fails with "cannot find -lufsecp":**
Set `UFSECP_LIB_DIR` to the directory containing `libufsecp.so`:
```bash
export UFSECP_LIB_DIR=/path/to/build-linux/include/ufsecp
```

**Node.js "koffi" install fails:**
Requires Node.js 18+ (recommended 22+). Use `npm install` in the `nodejs_example/` dir.

**Java "UnsatisfiedLinkError":**
Pass `-Djna.library.path=` pointing to the directory with `libufsecp.so`.
