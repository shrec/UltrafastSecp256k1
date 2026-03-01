# Cross-Platform Audit Report

> Generated: 2026-03-01 | Library: UltrafastSecp256k1 v3.16.0
> Audit Framework: v2.0.0

## Summary

| # | Platform            | OS      | Arch      | Compiler       | Build   | Modules | Verdict      | Time    |
|---|---------------------|---------|-----------|----------------|---------|---------|--------------|---------|
| 1 | Windows (local)     | Windows | x86-64    | Clang 21.1.0   | Release | 48/49   | AUDIT-READY  | 39.3 s  |
| 2 | Linux Docker (local)| Linux   | x86-64    | GCC 13.3.0     | Release | 48/49   | AUDIT-READY  | 46.0 s  |
| 3 | Linux CI            | Linux   | x86-64    | Clang 17.0.6   | Release | 46/46   | AUDIT-READY  | 44.3 s  |
| 4 | Linux CI            | Linux   | x86-64    | GCC 13.3.0     | Release | 46/46   | AUDIT-READY  | 48.4 s  |
| 5 | Windows CI          | Windows | x86-64    | MSVC 1944      | Release | 45/45   | AUDIT-READY  | 139.1 s |
| 6 | ESP32-S3 (real HW)  | FreeRTOS| Xtensa    | GCC 14.2.0     | Release | 40/40   | AUDIT-READY  | 583 s   |
| 7 | Milk-V Mars (real HW)| Linux  | RISC-V 64 | GCC 13.3.0     | Release | 48/49   | AUDIT-READY  | 250 s   |

**All 7 platform configurations: AUDIT-READY**

## Notes

- Rows 1-2 run against v3.16.0 (commit 28a40d0a) with 49 modules (includes BIP-340 strict, MuSig2 BIP-327 vectors, FFI round-trip, RFC 9591 invariants/3-of-5)
- Rows 3-5 run against v3.15.2 (commit 03b1661a) with 45-46 modules (before BIP-340 strict + FROST RFC 9591 modules were added)
- Row 6: ESP32-S3 real hardware (Xtensa LX7 240 MHz), 8 modules skipped (platform-incompatible: __int128, AVX2, SHA-NI, exhaustive, comprehensive, FFI, desktop fuzz)
- Row 7: Milk-V Mars real hardware (SiFive U74-MC rv64gc_zba_zbb @ 1.5 GHz), all 49 modules run, cross-compiled from x86-64
- Module count difference: v3.16.0 added 3 new modules (BIP-340 strict, MuSig2 BIP-327, FFI round-trip)
- "48/49" means 1 advisory warning: Side-channel dudect smoke test (probabilistic timing; flakes under shared-runner / hypervisor noise)
- FieldElement52 (5x52) test skipped on MSVC (no __uint128_t), hence 45 vs 46 on CI

## Section-by-Section Breakdown (v3.16.0, Windows Clang 21.1.0)

### Section 1: Mathematical Invariants (Fp, Zn, Group Laws) -- 13/13 PASS
| Module | Result | Time |
|--------|--------|------|
| Field Fp deep audit (add/mul/inv/sqrt/batch) | PASS | 360 ms |
| Scalar Zn deep audit (mod/GLV/edge/inv) | PASS | 65 ms |
| Point ops deep audit (Jac/affine/sigs) | PASS | 1416 ms |
| Field & scalar arithmetic | PASS | 3 ms |
| Arithmetic correctness | PASS | 18 ms |
| Scalar multiplication | PASS | 245 ms |
| Exhaustive algebraic verification | PASS | 25 ms |
| Comprehensive 500+ suite | PASS | 44 ms |
| ECC property-based invariants | PASS | 92 ms |
| Affine batch addition | PASS | 269 ms |
| Carry chain stress (limb boundary) | PASS | 2 ms |
| FieldElement52 (5x52) vs 4x64 | PASS | 4 ms |
| FieldElement26 (10x26) vs 4x64 | PASS | 4 ms |

### Section 2: Constant-Time & Side-Channel Analysis -- 4/5 PASS (1 advisory)
| Module | Result | Time |
|--------|--------|------|
| CT deep audit (masks/cmov/cswap/timing) | PASS | 139 ms |
| Constant-time layer | PASS | 10 ms |
| FAST == CT equivalence | PASS | 15 ms |
| Side-channel dudect (smoke) | WARN | 78 ms |
| CT scalar_mul vs fast (diagnostic) | PASS | 5 ms |

### Section 3: Differential & Cross-Library Testing -- 3/3 PASS
| Module | Result | Time |
|--------|--------|------|
| Differential correctness | PASS | 299 ms |
| Fiat-Crypto reference vectors | PASS | 3 ms |
| Cross-platform KAT | PASS | 2 ms |

### Section 4: Standard Test Vectors (BIP-340, RFC-6979, BIP-32) -- 6/6 PASS
| Module | Result | Time |
|--------|--------|------|
| BIP-340 official vectors | PASS | 24 ms |
| BIP-340 strict encoding (non-canonical) | PASS | 28 ms |
| BIP-32 official vectors TV1-5 | PASS | 83 ms |
| RFC 6979 ECDSA vectors | PASS | 33 ms |
| FROST reference KAT vectors | PASS | 17 ms |
| MuSig2 BIP-327 reference vectors | PASS | 17 ms |

### Section 5: Fuzzing & Adversarial Attack Resilience -- 4/4 PASS
| Module | Result | Time |
|--------|--------|------|
| Adversarial fuzz (malform/edge) | PASS | 303 ms |
| Parser fuzz (DER/Schnorr/Pubkey) | PASS | 8316 ms |
| Address/BIP32/FFI boundary fuzz | PASS | 2361 ms |
| Fault injection simulation | PASS | 110 ms |

### Section 6: Protocol Security (ECDSA, Schnorr, MuSig2, FROST) -- 9/9 PASS
| Module | Result | Time |
|--------|--------|------|
| ECDSA + Schnorr | PASS | 24 ms |
| BIP-32 HD derivation | PASS | 26 ms |
| MuSig2 | PASS | 20 ms |
| ECDH + recovery + taproot | PASS | 5 ms |
| v4 (Pedersen/FROST/etc) | PASS | 76 ms |
| Coins layer | PASS | 52 ms |
| MuSig2 + FROST protocol suite | PASS | 172 ms |
| MuSig2 + FROST advanced/adversar | PASS | 83 ms |
| Integration (ECDH/batch/cross-proto) | PASS | 1037 ms |

### Section 7: ABI & Memory Safety (zeroization, hardening) -- 4/4 PASS
| Module | Result | Time |
|--------|--------|------|
| Security hardening (zero/bitflip/nonce) | PASS | 21837 ms |
| Debug invariant assertions | PASS | 3 ms |
| ABI version gate (compile-time) | PASS | 1 ms |
| Cross-ABI/FFI round-trip (ufsecp C API) | PASS | 10 ms |

### Section 8: Performance Validation & Regression -- 4/4 PASS
| Module | Result | Time |
|--------|--------|------|
| Accelerated hashing | PASS | 656 ms |
| SIMD batch operations | PASS | 4 ms |
| Multi-scalar & batch verify | PASS | 21 ms |
| Performance smoke (sign/verify roundtrip) | PASS | 0 ms |

## Raw Report Files

Individual reports are stored as text and JSON in this directory:

- `windows-x86_64-clang21.txt` / `.json` -- Windows local (v3.16.0)
- `linux-x86_64-gcc13-docker.txt` / `.json` -- Linux Docker CI (v3.16.0)
- `linux-x86_64-clang17-ci.txt` / `.json` -- Linux GitHub CI (v3.15.2)
- `linux-x86_64-gcc13-ci.txt` / `.json` -- Linux GitHub CI (v3.15.2)
- `windows-x86_64-msvc-ci.txt` / `.json` -- Windows GitHub CI (v3.15.2)
- `esp32s3-xtensa-idf551.txt` / `.json` -- ESP32-S3 real hardware (v3.16.0)
- `riscv64-gcc13-hw.txt` / `.json` -- Milk-V Mars RISC-V real hardware (v3.16.0)
