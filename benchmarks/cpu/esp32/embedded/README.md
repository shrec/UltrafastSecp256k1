# UltrafastSecp256k1 -- ESP32-S3 Benchmark (Hornet Format)

Full Bitcoin consensus benchmark on real ESP32-S3 hardware.
Raw output: `esp32s3-xtensa-idf54-bench-hornet.txt`

## Platform

| Property | Value |
|----------|-------|
| **Chip** | ESP32-S3 (rev 0.1) |
| **CPU** | 2 x Xtensa LX7 @ 240 MHz |
| **RAM** | 512 KB SRAM (301 KB free heap at benchmark start) |
| **Flash** | 8 MB |
| **Build** | Portable C++ (no assembly, no `__int128`) |
| **ESP-IDF** | v5.4 |
| **Compiler** | xtensa-esp32s3-elf-g++ 14.2.0 |
| **Optimization** | `-O3` |
| **Library** | UltrafastSecp256k1 v3.16.0 |
| **Field** | 4x64 limbs (emulated 64-bit on 32-bit core) |
| **Scalar** | 10x26 limbs (`uint32_t`), Barrett reduction |
| **Point mul** | GLV endomorphism + wNAF (w=5) |
| **Dual mul** | Shamir's trick (a*G + b*P) |
| **Timer** | `esp_timer` (1 us resolution) |
| **Method** | Median of 3 runs, per-op warmup, 16-key pool |
| **Date** | 2026-03-07 |

## ECDSA (RFC 6979)

| Operation | us/op | ops/sec |
|-----------|------:|--------:|
| ecdsa_sign (deterministic nonce) | 7,443 | 134 |
| ecdsa_verify (full) | 18,670 | 54 |

## Schnorr / BIP-340 (Taproot)

| Operation | us/op | ops/sec |
|-----------|------:|--------:|
| schnorr_sign (pre-computed keypair) | 6,467 | 155 |
| schnorr_sign (from raw privkey) | 12,811 | 78 |
| schnorr_verify (x-only 32B pubkey) | 19,947 | 50 |
| schnorr_verify (pre-parsed pubkey) | 18,424 | 54 |

## Batch Verification (N=16)

| Operation | us/op | ops/sec | vs individual |
|-----------|------:|--------:|--------------:|
| schnorr_batch_verify (per sig) | 19,936 | 50 | 1.00x |
| ecdsa_batch_verify (per sig) | 18,394 | 54 | 1.02x |

## Key Generation

| Operation | us/op | ops/sec |
|-----------|------:|--------:|
| pubkey_create (k*G, GLV+wNAF) | 6,134 | 163 |
| schnorr_keypair_create | 6,204 | 161 |

## Point Arithmetic

| Operation | us/op | ops/sec |
|-----------|------:|--------:|
| k*P (arbitrary point, GLV+wNAF) | 12,752 | 78 |
| a*G + b*P (Shamir dual mul) | 18,296 | 55 |
| point_add (Jacobian mixed) | 479 | 2,088 |
| point_dbl (Jacobian) | 330 | 3,030 |

## Field Arithmetic

| Operation | ns/op | ops/sec |
|-----------|------:|--------:|
| field_mul | 5,910 | 169.2 k |
| field_sqr | 4,848 | 206.3 k |
| field_inv (Fermat, 256-bit exp) | 130,100 | 7.7 k |
| field_add (mod p) | 572 | 1.75 M |
| field_sub (mod p) | 814 | 1.23 M |
| field_negate (mod p) | 510 | 1.96 M |

## Scalar Arithmetic (mod n)

| Operation | ns/op | ops/sec |
|-----------|------:|--------:|
| scalar_mul | 18,890 | 52.9 k |
| scalar_inv | 132,950 | 7.5 k |
| scalar_add | 652 | 1.53 M |
| scalar_negate | 706 | 1.42 M |

## Serialization

| Operation | us/op | ops/sec |
|-----------|------:|--------:|
| pubkey_serialize (33B compressed) | 162 | 6,176 |
| ecdsa_sig_to_der (DER encode) | 6.0 | 166,300 |
| schnorr_sig_to_bytes (64B) | 2.3 | 432,900 |

## Constant-Time Signing (CT layer)

| Operation | us/op | ops/sec | CT overhead |
|-----------|------:|--------:|------------:|
| ct::ecdsa_sign | 13,741 | 73 | 1.85x |
| ct::schnorr_sign | 7,574 | 132 | 1.17x |

## libsecp256k1 Comparison (bitcoin-core v0.7.2, same hardware)

| Operation | libsecp (us) | Ultra FAST (us) | Ratio |
|-----------|-------------:|----------------:|------:|
| Generator * k | 7,264 | 6,134 | **1.18x** |
| ECDSA Sign | 9,419 | 7,443 | **1.27x** |
| ECDSA Verify | 31,657 | 18,670 | **1.70x** |
| Schnorr Keypair | 7,270 | 6,204 | **1.17x** |
| Schnorr Sign | 9,345 | 6,467 | **1.45x** |
| Schnorr Verify | 32,326 | 19,947 | **1.62x** |

### CT-vs-CT (Ultra CT vs libsecp CT-equivalent)

| Operation | Ratio |
|-----------|------:|
| ECDSA Sign (CT vs CT) | 0.69x |
| ECDSA Verify | **1.70x** |
| Schnorr Sign (CT vs CT) | **1.23x** |
| Schnorr Verify | **1.62x** |

## Bitcoin Block Validation Estimates (1 core)

| Scenario | Individual | Batch (N=16) |
|----------|----------:|-------------:|
| Pre-Taproot (~3000 ECDSA verify) | 56.0 s | 55.2 s |
| Taproot (~2000 Schnorr + ~1000 ECDSA) | 58.6 s | 58.3 s |

| Metric | Value |
|--------|------:|
| ECDSA tx throughput | 54 tx/sec |
| Schnorr tx throughput | 50 tx/sec |
| Pre-Taproot blocks/sec | 0.02 |
| Taproot blocks/sec | 0.02 |

## Cross-Platform Comparison

| Operation | ESP32-S3 (240 MHz) | Milk-V Mars RV64 (1.5 GHz) | x86-64 i5-14400F (2.5 GHz) |
|-----------|-------------------:|----------------------------:|---------------------------:|
| field_mul | 5,910 ns | 200 ns | 12.1 ns |
| field_sqr | 4,848 ns | 185 ns | 11.3 ns |
| field_add | 572 ns | 36 ns | 4.4 ns |
| k*G | 6,134 us | 44 us | 5.4 us |
| k*P | 12,752 us | 665 us | 17.8 us |
| ECDSA sign | 7,443 us | - | 6.6 us |
| ECDSA verify | 18,670 us | - | 22.7 us |
| Schnorr sign | 6,467 us | - | 5.9 us |
| Schnorr verify | 19,947 us | - | 21.5 us |

**Relative performance (field_mul):**
ESP32-S3 is ~30x slower than RISC-V (6.25x clock ratio) and ~488x slower than x86-64 (10.4x clock ratio).
Per-clock, ESP32-S3 field_mul is ~5x slower than RISC-V and ~47x slower than x86-64 -- expected for emulated 64-bit on 32-bit core without `__int128`.

## Build Configuration

```cmake
-DSECP256K1_PLATFORM_ESP32=1
-DSECP256K1_NO_INT128=1
-DSECP256K1_NO_ASM=1
-DNDEBUG=1
```

```ini
# sdkconfig.defaults
CONFIG_IDF_TARGET="esp32s3"
CONFIG_COMPILER_OPTIMIZATION_PERF=y
CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240=y
CONFIG_ESP_MAIN_TASK_STACK_SIZE=32768
CONFIG_ESPTOOLPY_FLASHSIZE_8MB=y
CONFIG_ESP_IPC_TASK_STACK_SIZE=4096
```

- Stack overflow protection: Requires `CONFIG_ESP_MAIN_TASK_STACK_SIZE=32768` minimum
- All tests pass on real hardware (ESP32-S3 development board)
- Serial output at 115200 baud
