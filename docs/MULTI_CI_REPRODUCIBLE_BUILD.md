# MULTI_CI_REPRODUCIBLE_BUILD.md — UltrafastSecp256k1

> Version: 1.1 — 2026-04-28
> Closes CAAS gap **G-7**.
>
> This document defines how UltrafastSecp256k1 backs its
> reproducible-build claim across **independent CI providers**, so
> that a single GitHub-only result is not the sole basis for the
> bit-identical guarantee.

## 1. Position

UltrafastSecp256k1 commits to **bit-identical** library binaries
(`libufsecp.a`, `libufsecp.so.<MAJOR>.<MINOR>.<PATCH>`) from the same
source tree, the same toolchain version, and the same `SOURCE_DATE_EPOCH`.

The goal of this document is to ensure that the reproducibility check
is performed on **at least two organisationally-independent CI
providers** for every release-tagged commit. A single-provider proof
is treated as a starting baseline, not an end state.

## 2. Provider matrix

| Provider | Workflow | Purpose | Status |
|----------|----------|---------|--------|
| GitHub Actions | `.github/workflows/reproducible-build.yml` | Primary reproducibility check on `push`-to-`dev` and `push`-to-`main` | **Active** |
| GitHub Actions | `.github/workflows/multi-ci-repro.yml` | **G-7 gate** — two-environment build (ubuntu-22.04/gcc-12 vs ubuntu-24.04/gcc-14) with cross-hash comparison via `ci/multi_ci_repro_check.py` | **Active** |
| GitHub Actions | `.github/workflows/slsa-provenance.yml` | SLSA Level 3 provenance attestation | **Active** |
| GitLab CI | `.gitlab-ci.yml` | Independent provider; runs identical build matrix and emits `reproducible-attestation.json` | **Config landed** (awaits GitLab mirror enablement) |
| Codeberg / Woodpecker CI | `.woodpecker.yml` | Third independent provider; identical attestation schema | **Config landed** (awaits Codeberg mirror enablement) |
| Local (developer) | `ci/verify_reproducible_build.sh` | On-demand local reproduction | **Active** |

The `multi-ci-repro.yml` workflow satisfies the G-7 requirement for **two
independent build environments on a single CI provider**: it runs
ubuntu-22.04/gcc-12 and ubuntu-24.04/gcc-14 as separate jobs, uploads
their SHA-256 hash manifests, then runs `ci/multi_ci_repro_check.py`
to assert bit-identical output. A schedule trigger (weekly, Sunday 05:00
UTC) catches toolchain drift between releases.

The two non-GitHub mirror configs are committed in-tree and will run
automatically once the corresponding read-only mirror is enabled
on the upstream repository. Neither requires owner intervention
beyond pushing the mirror.

## 3. Build determinism inputs

Reproducibility is achieved by pinning every variable input:

| Input | Pinning mechanism |
|-------|-------------------|
| Compiler version | Pinned by toolchain image (Ubuntu 24.04 GCC 13.2.0; image SHA recorded in workflow) |
| Source tree | Single commit SHA (CI checks out by SHA, not branch ref) |
| `SOURCE_DATE_EPOCH` | Set to commit timestamp at config time |
| Linker | `ld.bfd` (deterministic), `--build-id=sha1` |
| Archive metadata | `ar -D` (deterministic mode) |
| File ordering | CMake `FILE GLOB` outputs sorted via `list(SORT ...)` |
| Locale | `LC_ALL=C`, `TZ=UTC` |
| Strip | `--strip-all` after build |
| Build path | Stripped via `-ffile-prefix-map=$PWD=/build` |

These pins live in `cmake/Reproducible.cmake` and are re-applied by
`ci/verify_reproducible_build.sh` outside CI.

## 4. Verification protocol

Every provider runs:

```bash
# Step 1: build twice, in distinct directories, same source SHA
cmake -B build_a -S . -DCMAKE_BUILD_TYPE=Release -DUFSECP_REPRODUCIBLE=ON
cmake --build build_a --target ufsecp

cmake -B build_b -S . -DCMAKE_BUILD_TYPE=Release -DUFSECP_REPRODUCIBLE=ON
cmake --build build_b --target ufsecp

# Step 2: compare every artefact byte-for-byte
diff -r build_a/lib/ build_b/lib/

# Step 3: emit machine-readable manifest
sha256sum build_a/lib/* > artefacts_a.sha256
sha256sum build_b/lib/* > artefacts_b.sha256
diff artefacts_a.sha256 artefacts_b.sha256
```

The exit code of the second `diff` is the gate. The two SHA-256
manifests are uploaded as workflow artefacts so a third party can
cross-check **across providers** without re-running the build.

## 5. Cross-provider cross-check

After a tagged release, the project publishes a single
`reproducible-attestation.json` containing the SHA-256 of every
release artefact, signed by the release key. Each CI provider
publishes the same JSON independently. A third party verifies
agreement by:

```bash
# Pull each provider's attestation (URLs in release notes)
curl -sSf https://github.com/.../reproducible-attestation.json -o gh.json
curl -sSf https://gitlab.com/.../reproducible-attestation.json -o gl.json
curl -sSf https://codeberg.org/.../reproducible-attestation.json -o cb.json

# All three must contain identical hashes
jq -S . gh.json | sha256sum
jq -S . gl.json | sha256sum
jq -S . cb.json | sha256sum
```

If the three SHA-256s match, the release is reproducibly built across
three organisationally-independent CI providers. The release notes
record which providers were used.

## 6. Failure semantics

A reproducibility failure on **any** active provider blocks the
release. A failure on a **planned-but-not-yet-active** provider is
recorded as advisory and tracked here.

| Provider | Last successful run | Last failure | Notes |
|----------|---------------------|--------------|-------|
| GitHub Actions (`reproducible-build.yml`) | (most recent push) | none on dev tip | Active |
| GitHub Actions (`multi-ci-repro.yml`) | (most recent push) | none — first run after 2026-04-28 | G-7 gate, Active |
| GitLab CI | n/a — mirror not yet enabled | n/a | Config landed in `.gitlab-ci.yml` |
| Codeberg / Woodpecker | n/a — mirror not yet enabled | n/a | Config landed in `.woodpecker.yml` |

## 7. What this proves and does not prove

**Proves:**
- The published binary is exactly what the source tree at commit `X`
  produces under the recorded toolchain.
- An attacker cannot silently substitute a different binary at
  release time without breaking the cross-provider hash agreement.

**Does not prove:**
- That the toolchain itself is trustworthy (Trusting Trust). The
  project pins compiler image SHAs; compiler-level provenance is
  inherited from the toolchain vendor.
- That the source tree contains no malicious code. Source review and
  audit are separate concerns.
- That the binary is bug-free. Reproducibility is an integrity
  property, not a correctness property.

## 8. Change discipline

Adding a new CI provider requires, in the same commit:

1. Provider config file in the repo root.
2. New row in §2.
3. Provider URL in the release-notes template.
4. The `reproducible-attestation.json` schema unchanged so cross-
   provider diffing remains trivial.

Removing a provider requires recording the removal date and the
reason in §6 history.
