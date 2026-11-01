# UltrafastSecp256k1 adoption: Sparrow Frigate and other integrations

Third-party project integrations of UltrafastSecp256k1. Where an integration was contributed
by the maintainer (`@shrec`), it is hosted and shipped by the third party; this is noted per
entry rather than framed as independent verification.

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
- **DuckDB extension source:** [sparrowwallet/duckdb-ufsecp-extension](https://github.com/sparrowwallet/duckdb-ufsecp-extension)
  (CUDA, OpenCL and Metal sources wrapping `libufsecp`)
- **Third-party coverage:** [Bitcoin Optech Newsletter, 2026-04-17](https://bitcoinops.org/en/newsletters/2026/04/17/)
  reports that Frigate v1.4.0 "now uses the UltrafastSecp256k1 library in conjunction with modern GPU computation"

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

> **Note:** Community throughput figures above reflect Frigate's full scanning pipeline
> (BIP-352 transaction parsing, output matching, and result filtering), not raw
> `ufsecp_gpu_bip352_scan_batch` kernel throughput. These numbers are not directly
> comparable to the kernel-level diagnostic figures in
> [docs/WHY_ULTRAFASTSECP256K1.md](WHY_ULTRAFASTSECP256K1.md).

---

## libbitcoin (former integration)

From June to October 2026, [libbitcoin](https://github.com/libbitcoin) offered UltrafastSecp256k1 as a
build-time secp256k1 backend across its C++ libraries: libbitcoin-system, -database, -network, -node,
-server, -explorer, and the libbitcoin-build generator. A libbitcoin build with the option enabled ran
its secp256k1 operations on UltrafastSecp256k1.

On 2026-10-02, [libbitcoin-system #2048](https://github.com/libbitcoin/libbitcoin-system/pull/2048)
("Batch verify internally regardless of secp256k1 linkage.") removed the option. libbitcoin `master`
now uses its own internal secp256k1, optionally bitcoin-core/secp256k1 (`--with-secp256k1`), and its
own CUDA batch verification ([#2049](https://github.com/libbitcoin/libbitcoin-system/pull/2049)).
The integration never reached a tagged libbitcoin release; the latest, v3.8.0, is from 2023.

### Integration details (while it was active)

- Build options: `--with-ultrafast` (GNU autotools), `-Dwith-ultrafast=ON` (CMake), and the
  `UltrafastSecp256k1-vc145` NuGet package (MSVC). libbitcoin-build added the NuGet dependency
  disabled by default, then bumped it from v4.3 to v4.5 and v4.6
- The secp256k1, ultrafast and ultrafast/CUDA build options could be imported across the libbitcoin repositories
- libbitcoin-system detected an available GPU; libbitcoin-node and libbitcoin-server gated batch
  validation on whether a GPU was present
- libbitcoin-system #2017 (2026-09-28) made libbitcoin's own internal secp256k1 the default, with
  libsecp256k1 or UltrafastSecp256k1 as replacements; #2018 rejected linking both at once
- On this repository's side the integration used the bridge-free direct profile (`compat/libbitcoin_direct`,
  target `secp256k1::fastsecp256k1_libbitcoin`, `#include <ufsecp/libbitcoin.hpp>`); see
  [LIBBITCOIN_INTEGRATION.md](LIBBITCOIN_INTEGRATION.md)

### Evidence

- **Merged libbitcoin PRs that added and maintained the integration:**
  - [libbitcoin-build #426](https://github.com/libbitcoin/libbitcoin-build/pull/426): "Add ultrafast NuGet dependency (disabled by default)." (2026-06-13)
  - [libbitcoin-build #433](https://github.com/libbitcoin/libbitcoin-build/pull/433): "GNU toolchain update, supporting ultrafast, etc." (2026-07-02)
  - [libbitcoin-system #1913](https://github.com/libbitcoin/libbitcoin-system/pull/1913): "Make vs secp256k1, ultrafast, and uf/cuda options importable." (2026-07-20)
  - [libbitcoin-system #1941](https://github.com/libbitcoin/libbitcoin-system/pull/1941): "Update ultrafast NuGet, add GPU detection." (2026-08-19)
  - [libbitcoin-node #1101](https://github.com/libbitcoin/libbitcoin-node/pull/1101): "Update ultrafast NuGet, gate batching on presence of GPU." (2026-08-19)
  - [libbitcoin-system #2003](https://github.com/libbitcoin/libbitcoin-system/pull/2003): "Bump ultrafast to NuGet 4.6, regenerate artifacts." (2026-09-25; the same bump landed in -database, -network, -node, -server, -explorer and -build)
  - [libbitcoin-system #2017](https://github.com/libbitcoin/libbitcoin-system/pull/2017): "Add internal secp256k1". External linkage now defaults to none and allows replacement via libsecp256k1 or ultrafast (2026-09-28)
- **Merged libbitcoin PRs that removed it:**
  - [libbitcoin-system #2018](https://github.com/libbitcoin/libbitcoin-system/pull/2018): "Report secp256k1 lib/ver, reject link secp256k1 + uf, fix internal secp interface." (2026-09-28)
  - [libbitcoin-system #2048](https://github.com/libbitcoin/libbitcoin-system/pull/2048): "Batch verify internally regardless of secp256k1 linkage." Removes `with-ultrafast` from CMake and autotools and `UltrafastSecp256k1-vc145` from `packages.config` (2026-10-02)
  - [libbitcoin-system #2049](https://github.com/libbitcoin/libbitcoin-system/pull/2049): "Integrate cuda and generator tool." libbitcoin compiles its own sha256 and secp256k1 source to PTX for GPU batch verification (2026-10-02)
- **Upstream requests from libbitcoin's maintainer, Eric Voskuil:**
  [#434](https://github.com/shrec/UltrafastSecp256k1/issues/434) (OpenCL GPU support for static MSVC consumers of the libbitcoin-direct profile),
  [#435](https://github.com/shrec/UltrafastSecp256k1/issues/435) (GPU support for libbitcoin-direct consumers on Linux and macOS); both closed 2026-09-25
- **NuGet packaging:** [eynhaender/getUltrafastSecp256k1](https://github.com/eynhaender/getUltrafastSecp256k1)
  builds `UltrafastSecp256k1-vc145`, the package libbitcoin's MSVC builds consumed. This repository's
  [`nuget-native.yml`](../.github/workflows/nuget-native.yml) builds the `vc143` package in the same convention

---

## Other public integrations

| Project | How it uses UltrafastSecp256k1 | Evidence |
|---|---|---|
| [Robinhood Chain sequencer feed](https://github.com/groovg/ultrafast-robinhood-sequencer-feed) (Rust) | Opt-in `--features ufsecp` alternative to the default `secp256k1` crate for ECDSA sender recovery. The library is built from source (`scripts/build-ufsecp.sh`), and a test checks that both libraries return the same results | Project README, `build.rs`, `scripts/build-ufsecp.sh` |
| [Satobox](https://github.com/embetrix/satobox) (Yocto/OE embedded-Linux Bitcoin full-node distribution) | Recipe `meta-satobox/recipes-support/ultrafastsecp256k1/ultrafastsecp256k1_3.15.3.bb` builds the CPU library with x86-64 / AArch64 / RISC-V platform detection; `satobox-image.bb` installs `ultrafastsecp256k1-examples`. Added 2026-02-17 and updated to v3.15.3 on 2026-03-04 | Recipe and image files in the Satobox repository |

---

## Ecosystem signals

Public package traction provides a secondary adoption signal.

> **Historical snapshot — 2026-03-29.** These figures were collected on that date and are not automatically updated. Current numbers may differ.

**Snapshot date:** 2026-03-29

| Surface | Package | Metric | Value |
|---|---|---|---|
| npm | [`ufsecp`](https://www.npmjs.com/package/ufsecp) | Downloads, last 30 days | 1,192 |
| npm | [`react-native-ufsecp`](https://www.npmjs.com/package/react-native-ufsecp) | Downloads, last 30 days | 1,295 |
| NuGet | [`Ufsecp`](https://www.nuget.org/packages/Ufsecp) | Total downloads | 1,491 |

---

See also: [ADOPTERS.md](ADOPTERS.md) for the full adopter list with contact and contribution details.
