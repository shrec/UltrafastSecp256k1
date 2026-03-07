# Cross-Platform Benchmark Comparison

All results: UltrafastSecp256k1 FAST path, single-threaded, median timing.

## Bitcoin Consensus Operations

| Operation | x86-64 i5-14400F (2.5 GHz) | Milk-V Mars RV64 (1.5 GHz) | ESP32-S3 Xtensa (240 MHz) |
|-----------|---------------------------:|----------------------------:|--------------------------:|
| ECDSA sign | 6.6 us | - | 7,443 us |
| ECDSA verify | 22.7 us | - | 18,670 us |
| Schnorr sign (keypair) | 5.9 us | - | 6,467 us |
| Schnorr verify (cached) | 20.6 us | - | 18,424 us |
| Schnorr verify (raw) | 24.3 us | - | 19,947 us |
| k*G (pubkey_create) | 5.4 us | 44 us | 6,134 us |
| k*P | 17.8 us | 665 us | 12,752 us |
| a*G+b*P (Shamir) | 21.2 us | - | 18,296 us |

## Field / Scalar Primitives

| Operation | x86-64 (2.5 GHz) | RV64 (1.5 GHz) | ESP32-S3 (240 MHz) |
|-----------|------------------:|----------------:|-------------------:|
| field_mul | 12.1 ns | 200 ns | 5,910 ns |
| field_sqr | 11.3 ns | 185 ns | 4,848 ns |
| field_inv | 743 ns | 18,000 ns | 130,100 ns |
| field_add | 4.4 ns | 36 ns | 572 ns |
| scalar_mul | 22.4 ns | - | 18,890 ns |
| scalar_inv | 964 ns | - | 132,950 ns |

## vs libsecp256k1 (on same hardware)

| Operation | x86-64 ratio | ESP32-S3 ratio |
|-----------|-------------:|---------------:|
| Generator * k | - | 1.18x |
| ECDSA Sign | - | 1.27x |
| ECDSA Verify | - | 1.70x |
| Schnorr Sign | - | 1.45x |
| Schnorr Verify | - | 1.62x |

## Result Files

| Platform | File |
|----------|------|
| x86-64 Linux | `bench_unified_full_local_20260307_schnorr_opt2.txt` |
| RISC-V Linux | `../cpu/riscv64/linux/milkv-mars-20260208.txt` |
| ESP32-S3 | `../cpu/esp32/embedded/esp32s3-xtensa-idf54-bench-hornet.txt` |
