# Supply Chain Local Parity (CAAS H-6)

> Companion document for `scripts/supply_chain_gate.py`. Lists which P15
> supply-chain controls run **locally without GitHub services**, which
> require GitHub, and how the gap is mitigated.

## Threat model

A reviewer with no network access and no GitHub credentials should still
be able to make a meaningful supply-chain pass on a clone of the repo at
a given commit. They will not match the GitHub-side checks 1:1, but they
should be able to detect the most damaging supply-chain regressions
(unpinned actions, mutated dependency manifest, mismatched artifact hash).

## Coverage matrix

| P15 control | Local-runnable | GitHub-only | Notes |
|---|---|---|---|
| Pinned-action SHA enforcement | ✅ | — | Scan all `.github/workflows/*.yml` for `uses: foo@` and require the suffix to be a 40-char SHA, not a tag or branch. |
| SBOM generation (CycloneDX) | ✅ | — | `cyclonedx-py` / `cyclonedx-cmake` produce the SBOM locally. |
| SBOM diff vs prior tag | ✅ | — | `local_supply_chain_check.py --sbom-diff <prev>` compares against last tagged SBOM if checked in. |
| Dependency-Review (PR delta) | — | ✅ | GitHub Advisory Database is the source of truth; not reproducible offline. |
| SLSA provenance (build attestation) | — | ✅ | Requires GitHub OIDC token; provenance file can be **verified** locally with `slsa-verifier` once downloaded. |
| Cosign signature on artifacts | partial | partial | Verification is offline once the public key + signature are in hand. Generation requires Sigstore. |
| Scorecard | — | ✅ | Scorecard runs against the public repo metadata. Local mode degraded to scoring rules that don't need API access. |
| Pinned-runner OS digest | ✅ | — | Workflow YAMLs reference `runs-on:` strings; `local_supply_chain_check.py` flags tag-style runners (`ubuntu-latest`) and prefers immutable `ubuntu-24.04`. |
| Repo permissions hardening | partial | ✅ | Repo-side settings only checkable via GitHub API; local lints catch `permissions:` blocks missing from workflows. |
| Secret scanning baseline | ✅ | ✅ | `gitleaks` / `trufflehog` run offline on the working tree; GitHub adds historical-blob scanning. |

## Local-only commands

```bash
# Run the local supply-chain gate (5 sub-gates: build-input pinning,
# reproducible-build digest, SLSA provenance, artifact hash manifest,
# build hardening flags). All checks are filesystem-local and need no
# GitHub credentials.
python3 scripts/supply_chain_gate.py --json -o supply_chain_local.json
```

Exit code is non-zero if any local-runnable control fails. The
`generated_at` and per-check booleans are written to the JSON report so
the result can be carried forward into the CAAS evidence chain.

## Residual GitHub-only controls

A reviewer running fully offline will not be able to verify:

- **GitHub Advisory Database deltas** (Dependency-Review action). These
  only matter for the *new* portion of a PR; the existing dependency set
  is captured in the SBOM and can be compared against an offline mirror
  of the OSV database (`osv-scanner` ships an offline mode).
- **OIDC-attested SLSA provenance** at build time. The verification step
  is offline; only generation needs OIDC.
- **Public-repo scorecard data** (issue / PR cadence, branch protection
  policy). These are repo-meta signals, not code-supply-chain signals.

These gaps are accepted in `docs/RESIDUAL_RISK_REGISTER.md` RR-002
(workflow parity). Promotion of any of these to "verified offline"
requires a documented offline source of truth (SBOM mirror, signed
provenance bundle, etc.).

## Acceptance

A reviewer with no network access can run
`scripts/supply_chain_gate.py` on a clean clone and:

1. Enumerate every workflow action and its pinned SHA.
2. Diff the current SBOM against any prior committed SBOM.
3. Receive a non-zero exit if any supply-chain regression is detected
   from local data alone.

This satisfies CAAS hardening item **H-6** in
`docs/CAAS_HARDENING_TODO.md`.
