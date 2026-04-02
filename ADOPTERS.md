# Adopters

Organizations and projects using UltrafastSecp256k1 in production or development.
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
| [SparrowWallet Frigate](https://github.com/sparrowwallet/frigate) | DuckDB-based Silent Payments scanning via `ufsecp.duckdb_extension` wrapping UltrafastSecp256k1; uses `ufsecp_scan(...)` with CUDA, OpenCL and Metal backends | ECDSA, Schnorr, ECDH, Silent Payments, GPU batch | 2026-03 (v1.4.0) |

## Development / Research

| Organization | Use Case | Features Used | Since |
|---|---|---|---|
| *Be the first!* | | | |

## Personal / Hobby Projects

| Project | Use Case | Features Used | Since |
|---|---|---|---|
| *Be the first!* | | | |

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
