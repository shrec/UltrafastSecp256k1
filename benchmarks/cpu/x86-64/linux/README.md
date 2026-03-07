# x86-64 Linux Benchmark (bench_unified)

Full results: `../../comparison/bench_unified_full_local_20260307_schnorr_opt2.txt`

## Platform

| Property | Value |
|----------|-------|
| **CPU** | Intel Core i5-14400F @ 2.496 GHz |
| **Arch** | x86-64 |
| **Compiler** | GCC 14.2.0 |
| **Harness** | 3s ramp-up, 500 warmup/op, 11 passes, IQR outlier removal, median |
| **Timer** | RDTSCP |
| **Pool** | 64 independent key/msg/sig sets |
| **Date** | 2026-03-07 |

## Key Results (Ultra FAST, ns/op)

| Category | Operation | ns/op |
|----------|-----------|------:|
| Field | field_mul | 12.1 |
| Field | field_sqr | 11.3 |
| Field | field_inv | 743.1 |
| Field | field_add | 4.4 |
| Scalar | scalar_mul | 22.4 |
| Scalar | scalar_inv | 964.3 |
| Point | k*G | 5,421.8 |
| Point | k*P | 17,801.5 |
| Point | a*G+b*P | 21,187.2 |
| Point | point_add (J+A) | 128.9 |
| Point | point_dbl | 75.9 |
| ECDSA | ecdsa_sign | 6,589.3 |
| ECDSA | ecdsa_verify | 22,660.8 |
| Schnorr | schnorr_sign | 5,853.9 |
| Schnorr | schnorr_verify (cached) | 20,636.2 |
| Schnorr | schnorr_verify (raw) | 24,320.8 |
| CT | ct::ecdsa_sign | 14,502.6 |
| CT | ct::schnorr_sign | 11,833.5 |
