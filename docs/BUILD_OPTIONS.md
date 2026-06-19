# Build Options Reference

> **Auto-generated — do not edit by hand.** Regenerate with `python3 ci/gen_build_options.py`. The source of truth is the `option()` / `cmake_dependent_option()` declarations in the project `CMakeLists.txt` files (each carries its own description + default).
>
> Defaults below are the **CMake declaration defaults**. Named build profiles (see [CMakePresets.json](../CMakePresets.json) and [BUILDING.md](BUILDING.md)) override many of them for a minimal footprint per coin / use case. A `cmake_dependent_option` is only honoured when its guard condition holds (otherwise it is forced off).

**79 options** across 8 scope(s). Set any flag at configure time with `-D<FLAG>=ON|OFF`.

```bash
# Example: CPU build with the shim + MuSig2, no ZK/FROST
cmake -S . -B out/mybuild -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSECP256K1_BUILD_SHIM=ON -DSECP256K1_BUILD_MUSIG2=ON \
  -DSECP256K1_BUILD_ZK=OFF -DSECP256K1_BUILD_FROST=OFF
```

## Global / top-level (backends, GPU op selection, install)

| Flag | Default | Description |
|------|---------|-------------|
| `SECP256K1_BCHN_SHIM_BUILD_TESTS` | `OFF` | Build BCHN libsecp256k1 compatibility shim tests |
| `SECP256K1_BUILD_BCH` | `OFF` | Build BCH-specific modules: RPA (Reusable Payment Addresses), CashAddr, EC grinding pipeline |
| `SECP256K1_BUILD_BENCH` | `ON` | Build benchmarks |
| `SECP256K1_BUILD_CABI` | `ON` | Build optional libufsecp C ABI package (ufsecp_* FFI/bridge surface; native engine consumers link secp256k1::fast) |
| `SECP256K1_BUILD_CPU` | `ON` | Build CPU implementation |
| `SECP256K1_BUILD_CUDA` | `OFF` | Build CUDA GPU support |
| `SECP256K1_BUILD_ETHEREUM` | `ON` | Build Ethereum module (Keccak, EIP-55/155/191, ecrecover) |
| `SECP256K1_BUILD_EXAMPLES` | `ON` | Build example programs |
| `SECP256K1_BUILD_JAVA` | `ON` | Build Java JNI bindings |
| `SECP256K1_BUILD_KNOTS` | `OFF` | [Bitcoin Knots] Minimal libsecp256k1 backend: ecdsa+recovery+schnorr+extrakeys+ellswift; everything else off |
| `SECP256K1_BUILD_LIBBITCOIN` | `OFF` | [libbitcoin] Minimal node profile: shim + GPU/CPU batch script-sig bridge + BIP-352; extras off |
| `SECP256K1_BUILD_METAL` | `OFF` | Build Apple Metal GPU support |
| `SECP256K1_BUILD_OPENCL` | `OFF` | Build OpenCL support |
| `SECP256K1_BUILD_ROCM` | `OFF` | Build ROCm/HIP GPU support (AMD) |
| `SECP256K1_BUILD_SHARED` | `OFF` | Build shared library |
| `SECP256K1_BUILD_TESTS` | `ON` | Build test suite |
| `SECP256K1_CORE_BACKEND_MODE` | `OFF` | Bitcoin Core alternative backend: CT enforced, strict ABI, RFC 6979 deterministic signing |
| `SECP256K1_GPU_BUILD_BIP324` | `ON` | GPU BIP-324 AEAD encrypt/decrypt ops |
| `SECP256K1_GPU_BUILD_BIP352` | `ON` | GPU BIP-352 silent-payment scan op |
| `SECP256K1_GPU_BUILD_ECDH` | `ON` | GPU ECDH batch op (secret-bearing) |
| `SECP256K1_GPU_BUILD_ECRECOVER` | `ON` | GPU ECDSA public-key recovery batch op |
| `SECP256K1_GPU_BUILD_FROST` | `ON` | GPU FROST partial-signature verify op |
| `SECP256K1_GPU_BUILD_HASH160` | `ON` | GPU Hash160(pubkey) batch op |
| `SECP256K1_GPU_BUILD_MSM` | `ON` | GPU multi-scalar multiplication op |
| `SECP256K1_GPU_BUILD_ZK` | `ON` | GPU ZK ops (knowledge / DLEQ / bulletproof / SNARK witness) |
| `SECP256K1_INSTALL` | `ON` | Generate install target |
| `SECP256K1_INSTALL_CABI` | `OFF` | Install optional libufsecp C ABI package during the top-level install |
| `SECP256K1_INSTALL_PKGCONFIG` | `ON` | Install pkg-config file |
| `SECP256K1_MSVC_OB3` | `ON` | MSVC: /Ob3 most-aggressive inlining |
| `SECP256K1_MSVC_WPO` | `OFF` | MSVC: /GL whole-program + /LTCG + /OPT:REF,ICF (opt-in; a /GL static lib forces consumer /LTCG) |
| `SECP256K1_REQUIRE_CT` | `OFF` | Deprecate non-CT sign functions (compile warnings on fast:: signing) |
| `SECP256K1_SPEED_FIRST` | `OFF` | Reserved: SECP256K1_FAST_NO_SECURITY_CHECKS compile definition (currently has no effect — CT guards are always on) |
| `SECP256K1_USE_ULTRAFAST` | `OFF` | [Bitcoin Core] Use UltrafastSecp256k1 as the secp256k1 backend (enables shim + CT + strict ABI) |
| `SECP256K1_WERROR` | `OFF` | Treat compiler warnings as errors (-Werror / /WX) |
| `UFSECP_BITCOIN_STRICT` | `ON` | Enforce BIP-340 strict encoding in public API (reject r>=p, s>=n) |
| `UFSECP_REFRESH_SOURCE_GRAPH` | `OFF` | Refresh the repo source graph during builds (mutates source tree — OFF by default for reproducibility) |
| `UFSECP_REPRODUCIBLE` | `OFF` | Enable reproducible-build flags (path stripping, fixed march) |

## CPU implementation (crypto modules, optimization, integration)

| Flag | Default | Description |
|------|---------|-------------|
| `SECP256K1_BUILD_ADAPTOR` | `ON` | Adaptor signatures (Schnorr + ECDSA) |
| `SECP256K1_BUILD_AS_OBJECT` | `OFF` | Build fastsecp256k1 as OBJECT library (no .a — objects go directly into parent link step) |
| `SECP256K1_BUILD_BIP324` | `ON` | Enable BIP-324 v2 encrypted P2P transport (ChaCha20-Poly1305, ElligatorSwift, HKDF) |
| `SECP256K1_BUILD_BIP352` | `ON` | BIP-352 Silent Payments (scan key ECDH + output derivation) |
| `SECP256K1_BUILD_ECIES` | `ON` | ECIES authenticated encryption (AES-256-CTR + HMAC-SHA256) |
| `SECP256K1_BUILD_FROST` | `ON` | FROST threshold signatures (t-of-n) |
| `SECP256K1_BUILD_LTC_SP` | `ON` | Litecoin Silent Payments (LTC-SP, ltcsp1... paycodes, ltc1p... outputs) |
| `SECP256K1_BUILD_MUSIG2` | `ON` | MuSig2 multi-signatures (BIP-327) |
| `SECP256K1_BUILD_PIPPENGER` | `ON` | Pippenger MSM + comb generator mul + batch affine (large MSM algorithms) |
| `SECP256K1_BUILD_SHIM` | `OFF` | Include libsecp256k1-compatible shim inside fastsecp256k1 (engine-integrated) |
| `SECP256K1_BUILD_WALLET` | `ON` | HD wallet stack (BIP-32/39, coin types, Bitcoin message signing) |
| `SECP256K1_BUILD_ZK` | `ON` | ZK proofs (Bulletproofs, Pedersen commitments, DLEQ) |
| `SECP256K1_ENABLE_OPENMP` | `OFF` | Enable OpenMP for batch parallel operations |
| `SECP256K1_RISCV_USE_PREFETCH` | `ON` | Enable prefetch hints for cache optimization |
| `SECP256K1_RISCV_USE_VECTOR` | `ON` | Enable RISC-V Vector Extension (RVV) if available |
| `SECP256K1_UNITY_BUILD` | `OFF` | Compile core as single TU (matches libsecp256k1 model) |
| `SECP256K1_USE_ASM` | `ON` | Enable inline assembly optimizations (x64/RISC-V, 2-5x speedup) |
| `SECP256K1_USE_FAST_REDUCTION` | `ON` | Use fast modular reduction (RISC-V asm, x64 BMI2) |
| `SECP256K1_USE_LTO` | `ON` | Enable Link Time Optimization (LTO) for C++ code |
| `SECP256K1_USE_PGO_GEN` | `OFF` | Enable Profile-Guided Optimization - Generate profile |
| `SECP256K1_USE_PGO_USE` | `OFF` | Enable Profile-Guided Optimization - Use profile |
| `SECP256K1_USE_RISCV_FE52_ASM` | `OFF` | Use hand-written RISC-V assembly for 5x52 field multiply/square (slower on in-order cores like U74) |

## CUDA backend

| Flag | Default | Description |
|------|---------|-------------|
| `SECP256K1_CUDA_LIMBS_32` | `OFF` | Use 8x32-bit limbs for field arithmetic |
| `SECP256K1_CUDA_USE_MONTGOMERY` | `OFF` | Use Montgomery field arithmetic backend in CUDA |

## OpenCL backend

| Flag | Default | Description |
|------|---------|-------------|
| `SECP256K1_USE_OPENCL` | `ON` | Enable OpenCL GPU acceleration |

## C ABI library (ufsecp_* shared/static)

| Flag | Default | Description |
|------|---------|-------------|
| `SECP256K1_BUILD_LIBBITCOIN_BENCH` | `OFF` | Build the libbitcoin batch sig-verify benchmark (bench_lbtc_batch) |
| `SECP256K1_BUILD_LIBBITCOIN_TESTS` | `OFF` | Build the libbitcoin GPU/CPU consensus differential (test_lbtc_consensus_diff) |
| `UFSECP_BUILD_SHARED` | `ON` | Build shared library |
| `UFSECP_BUILD_STATIC` | `ON` | Build static library |

## libsecp256k1 compatibility shim

| Flag | Default | Description |
|------|---------|-------------|
| `SECP256K1_SHIM_BUILD_SHARED` | `OFF` | Also build a self-contained shared ultrafast_secp256k1 (DLL/.so) exporting the libsecp256k1 ABI |
| `SECP256K1_SHIM_BUILD_TESTS` | `OFF` | Build shim compatibility test |
| `SECP256K1_SHIM_INSTALL` | `ON` | Install the shim shared lib + secp256k1*.h ABI headers + ultrafast_secp256k1.pc (our-name pkg-config; integrator aliases secp256k1 explicitly) |
| `SECP256K1_SHIM_RFC6979_COMPAT` | `OFF` | Match upstream libsecp256k1 nonce bytes exactly (includes ECDSA algo16 tag). Disables fault-attack resistance of hedged nonce. |

## libbitcoin bridge (script-sig batch verify + scan)

| Flag | Default | Description |
|------|---------|-------------|
| `UFSECP_LBTC_BUILD_BENCH` | `ON` | Build the bridge throughput benchmark |
| `UFSECP_LBTC_BUILD_EXAMPLE` | `ON` | Build the example harness |
| `UFSECP_LBTC_BUILD_TESTS` | `ON` | Build the bridge correctness test |
| `UFSECP_LBTC_WITH_GPU` | `OFF` | Enable GPU dispatch (requires engine GPU ABI) |

## Audit / test harness (differential, fuzz, protocol tests)

| Flag | Default | Description |
|------|---------|-------------|
| `SECP256K1_BUILD_CROSS_TESTS` | `OFF` | Build in-process differential tests against bitcoin-core/libsecp256k1 |
| `SECP256K1_BUILD_FUZZ_TESTS` | `OFF` | Build deterministic fuzz tests for parsers (DER, Schnorr, Pubkey) |
| `SECP256K1_BUILD_LIBFUZZER` | `OFF` | Build LibFuzzer harnesses (requires clang -fsanitize=fuzzer,address) |
| `SECP256K1_BUILD_LIBFUZZER_STANDALONE` | `OFF` | Build LibFuzzer harnesses in standalone deterministc mode (no fuzzer runtime) |
| `SECP256K1_BUILD_PROTOCOL_TESTS` | `OFF` | Build MuSig2 + FROST protocol tests |

---

_Generated from:_ `CMakeLists.txt`, `audit/CMakeLists.txt`, `bindings/android/CMakeLists.txt`, `bindings/android/example/src/main/cpp/CMakeLists.txt`, `bindings/c_api/CMakeLists.txt`, `bindings/java/CMakeLists.txt`, `bindings/wasm/CMakeLists.txt`, `compat/libbitcoin_bridge/CMakeLists.txt`, `compat/libsecp256k1_bchn_shim/CMakeLists.txt`, `compat/libsecp256k1_shim/CMakeLists.txt`, `compat/litecoin_shim/CMakeLists.txt`, `examples/CMakeLists.txt`, `examples/esp32_bench_hornet/CMakeLists.txt`, `examples/esp32_bench_hornet/main/CMakeLists.txt`, `examples/esp32_test/CMakeLists.txt`, `examples/esp32_test/main/CMakeLists.txt`, `examples/esp32c6_bench_hornet/CMakeLists.txt`, `examples/esp32c6_bench_hornet/main/CMakeLists.txt`, `examples/esp32p4_bench_hornet/CMakeLists.txt`, `examples/esp32p4_bench_hornet/main/CMakeLists.txt`, `examples/stm32_test/CMakeLists.txt`, `include/ufsecp/CMakeLists.txt`, `src/bch/CMakeLists.txt`, `src/cpu/CMakeLists.txt`, `src/cuda/CMakeLists.txt`, `src/gpu/CMakeLists.txt`, `src/ltc/cuda/CMakeLists.txt`, `src/metal/CMakeLists.txt`, `src/opencl/CMakeLists.txt`, `tests/esp32_audit/CMakeLists.txt`, `tests/esp32_audit/main/CMakeLists.txt`, `tests/esp32c6_audit/CMakeLists.txt`, `tests/esp32c6_audit/main/CMakeLists.txt`, `tests/esp32p4_audit/CMakeLists.txt`, `tests/esp32p4_audit/main/CMakeLists.txt`
