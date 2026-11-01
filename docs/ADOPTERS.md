# Who uses UltrafastSecp256k1: adopters and integrations

Organizations and projects using UltrafastSecp256k1 in production or development, including
[Sparrow Wallet's Frigate](https://github.com/sparrowwallet/frigate) (BIP-352 Silent Payments GPU scanning).
Former integrations, such as libbitcoin's `--with-ultrafast` build option, are listed under
[Former Integrations](#former-integrations).
This page also tracks public ecosystem signals so adoption evidence is not scattered
across package registries and release notes.

If you use UltrafastSecp256k1, please add yourself via PR or open a
[GitHub Discussion](https://github.com/shrec/UltrafastSecp256k1/discussions/categories/show-tell)
so we can list you here.

## Production

<!-- Template (copy, fill, submit PR):
| [YourOrg](https://example.com) | Brief description of use case | ECDSA / Schnorr / MuSig2 / FROST | since vX.Y |
-->

| Organization | Use Case | Features Used | Since |
|---|---|---|---|
| [SparrowWallet Frigate](https://github.com/sparrowwallet/frigate) | DuckDB-based Silent Payments scanning via `ufsecp.duckdb_extension` ([sparrowwallet/duckdb-ufsecp-extension](https://github.com/sparrowwallet/duckdb-ufsecp-extension)) wrapping UltrafastSecp256k1; uses `ufsecp_scan(...)` with CUDA, OpenCL and Metal backends | ECDSA, Schnorr, ECDH, Silent Payments, GPU batch | 2026-03 (v1.4.0) |

## Development / Research

| Organization | Use Case | Features Used | Since |
|---|---|---|---|
| [Robinhood Chain sequencer feed](https://github.com/groovg/ultrafast-robinhood-sequencer-feed) | Rust sequencer-feed client; UltrafastSecp256k1 is the opt-in (`--features ufsecp`) alternative to libsecp256k1 for ECDSA sender recovery, with a test that both give identical results | ECDSA recovery | 2026-09 |

## Packaging & Distribution

| Project | What it ships | Version | Since |
|---|---|---|---|
| [Satobox](https://github.com/embetrix/satobox) | Yocto/OE recipe in the `meta-satobox` layer (CPU build, x86-64 / AArch64 / RISC-V); `satobox-image` installs `ultrafastsecp256k1-examples`. Satobox is an embedded-Linux Bitcoin full-node distribution | v3.15.3 | 2026-02 |
| [getUltrafastSecp256k1](https://github.com/eynhaender/getUltrafastSecp256k1) | NuGet packaging pipeline for `UltrafastSecp256k1-vc145` (Visual Studio 2026, x64, static/shared, Release/Debug); libbitcoin's MSVC builds consumed this package until 2026-10-02 | tracks `dev` | 2026-06 |

## Former Integrations

| Organization | Use Case | Features Used | Period |
|---|---|---|---|
| [libbitcoin](https://github.com/libbitcoin) | Build-time secp256k1 backend across libbitcoin-system, -database, -network, -node, -server, -explorer and libbitcoin-build: `--with-ultrafast` (GNU autotools), `-Dwith-ultrafast=ON` (CMake) and the `UltrafastSecp256k1-vc145` NuGet package (MSVC), with GPU detection and GPU-gated batch validation. It was on libbitcoin `master` only and never reached a tagged release (the latest is v3.8.0, 2023). [libbitcoin-system #2048](https://github.com/libbitcoin/libbitcoin-system/pull/2048) removed the option; libbitcoin `master` now uses its own internal secp256k1 and its own CUDA batch verification | Signature validation (ECDSA, Schnorr), GPU batch | 2026-06-13 to 2026-10-02 |

## Personal / Hobby Projects

| Project | Use Case | Features Used | Since |
|---|---|---|---|

## Ecosystem Signals

Public package traction is a useful secondary signal alongside direct adopter disclosures.
These numbers are snapshots and will move over time.

**Snapshot date:** 2026-03-29

| Surface | Package | Metric | Value |
|---|---|---|---|
| npm | [`ufsecp`](https://www.npmjs.com/package/ufsecp) | Downloads, last 30 days | 1,192 |
| npm | [`react-native-ufsecp`](https://www.npmjs.com/package/react-native-ufsecp) | Downloads, last 30 days | 1,295 |
| NuGet | [`Ufsecp`](https://www.nuget.org/packages/Ufsecp) | Total downloads | 1,491 |

### Disclosure Notes

- The SparrowWallet Frigate entry is published with permission from Craig Raw.
- The libbitcoin, Robinhood Chain sequencer feed, Satobox and getUltrafastSecp256k1 entries
  cite those projects' own public repositories (merged PRs, READMEs, build recipes); see
  [ADOPTION.md](ADOPTION.md) for the per-project evidence. They were checked on 2026-11-01.
- Registry metrics were checked from the public npm and NuGet package endpoints on 2026-03-29.

---

### How to add yourself

1. Fork the repo and edit this file, or
2. Post in [Show & Tell](https://github.com/shrec/UltrafastSecp256k1/discussions/categories/show-tell).

Please include:

- **Name** -- organization, project, or handle
- **URL** -- homepage or repo (optional)
- **Use case** -- one-liner (e.g. "Bitcoin wallet Schnorr signing")
- **Features** -- ECDSA, Schnorr, MuSig2, FROST, GPU batch, etc.
- **Since** -- library version you started with
