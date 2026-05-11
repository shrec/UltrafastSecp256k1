# Adoption

Independently verified integrations of UltrafastSecp256k1 in third-party projects.

---

## Sparrow Wallet / Frigate

[Frigate](https://github.com/sparrowwallet/frigate) uses UltrafastSecp256k1 as the compute layer behind its DuckDB extension for Silent Payments scanning.

### Integration details

- Frigate 1.4.0 switched its DuckDB extension to `ufsecp.duckdb_extension` using UltrafastSecp256k1 by `@shrec`
- Frigate added CUDA, OpenCL and Metal backend support with automatic detection/fallback
- Frigate's scanning pipeline uses a custom DuckDB extension wrapping UltrafastSecp256k1 and exposes `ufsecp_scan(...)`

### Evidence

- **Frigate 1.4.0 release notes:**
  - "Change DuckDB extension to `ufsecp.duckdb_extension` using UltrafastSecp256k1 by `@shrec`"
  - [Frigate 1.4.0 release](https://github.com/sparrowwallet/frigate/releases/tag/1.4.0)
- **Frigate README:**
  - Frigate uses a custom DuckDB extension wrapping UltrafastSecp256k1
  - Scanning is performed with `ufsecp_scan(...)`
  - UltrafastSecp256k1 is described as a high-performance secp256k1 library with CPU and GPU backends
  - [Frigate README](https://github.com/sparrowwallet/frigate/blob/master/README.md)

### Why this matters

- Real-world use in a Bitcoin Silent Payments scanning system
- Independent integration by another project (Sparrow Wallet team)
- Practical validation of CPU/GPU backend design
- Published with permission from Craig Raw

### Independent performance results

Frigate's `benchmark.py` scanning mainnet to block 914,000 shows:

| Hardware | Backend | 2-year scan (133M tweaks) | Throughput |
|----------|---------|--------------------------|------------|
| 2× NVIDIA RTX 5090 | CUDA | 3.2 s | ~41.5 M/s |
| NVIDIA RTX 5080 | CUDA | 7.7 s | ~17.3 M/s |
| Apple M1 Pro | Metal | 3m 47s | ~584 K/s |
| Intel Core Ultra 9 285K | CPU (24 cores) | 3m 50s | ~577 K/s |
| Apple M1 Pro | CPU (10 cores) | 7m 47s | ~284 K/s |

Source: [Frigate README — Performance](https://github.com/sparrowwallet/frigate/blob/master/README.md#performance)

---

## Ecosystem signals

Public package traction provides a secondary adoption signal.

**Snapshot date:** 2026-03-29

| Surface | Package | Metric | Value |
|---|---|---|---|
| npm | [`ufsecp`](https://www.npmjs.com/package/ufsecp) | Downloads, last 30 days | 1,192 |
| npm | [`react-native-ufsecp`](https://www.npmjs.com/package/react-native-ufsecp) | Downloads, last 30 days | 1,295 |
| NuGet | [`Ufsecp`](https://www.nuget.org/packages/Ufsecp) | Total downloads | 1,491 |

---

See also: [ADOPTERS.md](ADOPTERS.md) for the full adopter list with contact and contribution details.
