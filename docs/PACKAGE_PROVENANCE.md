# Package / Release Provenance Binding

> **This is provenance _binding_, not release _authorization_.**
> Nothing in this document or its gate publishes a package, pushes a tag, merges
> to `main`, or authorizes a release. It only refuses to call a package or binary
> "audited" unless that artifact is cryptographically bound to the exact audited
> source. Releases remain owner-authorized only (see *Release Authorization* in
> `CLAUDE.md`).

## The problem

A built package — a NuGet `.nupkg`, a Linux `.deb`/`.a`, a Node N-API addon, a
wasm bundle, a signed release tarball — is just bytes. On its own it cannot prove
*which* source commit it was built from, *whether* that commit passed the audit,
or *which* CAAS evidence bundle was current at build time. A consumer (libbitcoin,
Bitcoin Core, a wallet) that trusts a package without that binding is trusting a
filename.

## The binding

Bastion B20 makes every package surface declare a **binding contract**: an
artifact is only trustworthy when it is tied to all four of —

| Binding target | Source of truth |
|----------------|-----------------|
| `source_commit` | `git HEAD` of the audited `dev` tree at build time |
| `caas_bundle_sha256` | sha256 of the committed `docs/EXTERNAL_AUDIT_BUNDLE.json` (recorded in `docs/EXTERNAL_AUDIT_BUNDLE.sha256`) |
| `audit_gate_verdict` | the `audit_gate.py` overall verdict for that commit (must be `pass`) |
| `artifact_sha256` | sha256 of the produced package/binary |
| `workflow_run_id` | the producing CI run, for forensics |

The ledger lives in **`docs/PACKAGE_PROVENANCE_STATUS.json`** and is enforced by
**`ci/check_package_provenance_binding.py`** (`audit_gate.py --package-provenance-binding`,
G-20). It reuses the existing supply-chain infrastructure rather than duplicating
it: `generate_slsa_provenance.py` / `verify_slsa_provenance.py` (SLSA v1.0 in-toto
statements), `slsa-provenance.yml` (SLSA L3 + cosign on tags), `nuget-native.yml`,
`packaging.yml`, `bindings.yml`, and `supply_chain_gate.py` (build-trust capability).
G-20 adds the one thing those did not: a verified binding of each artifact to the
*audited commit + CAAS evidence + verdict*.

## Status model (honest about the dev tree)

Nothing is built or published from `dev`, so the committed manifest never carries
fake current values:

- **`template`** — the dev-branch binding *contract*. The binding fields hold
  recognized sentinels (`source_commit: "@HEAD"`, `caas_bundle_sha256: "@committed-bundle"`,
  `audit_gate_verdict: "@audit-gate"`) and `artifact_sha256` / `workflow_run_id`
  are `null`. The producing workflow substitutes real values at build time; the
  gate then re-checks the result as `bound`.
- **`bound`** — a real built artifact. The gate **fails closed** unless
  `source_commit == HEAD`, `caas_bundle_sha256 ==` the committed bundle digest,
  `artifact_sha256` is a real 64-hex hash, `audit_gate_verdict == "pass"`, and a
  `workflow_run_id` is present. Stale provenance (wrong commit) or a mismatched
  CAAS bundle hash is rejected.
- **`owner_gated`** — a release artifact (wasm bundle, signed release tarball +
  SLSA + cosign). It is **never current evidence in the dev tree**: `artifact_sha256`
  and `workflow_run_id` stay `null` and it carries `@release` sentinels. A dev
  manifest that marks a release artifact current (a real hash or run id) **fails**
  the gate — you cannot present a release as audited-and-current without an
  owner-authorized release run.

## Current dev surfaces

| Surface | Producer | Status | Why |
|---------|----------|--------|-----|
| NuGet native static-lib (`vc143`/`vc145`) | `nuget-native.yml` | template | dispatch-only build, no nuget.org publish |
| Node N-API + React Native addons | `bindings.yml` | template (warning) | dispatch-only, no npm publish |
| Linux packages (`.deb`/`.rpm`) + C ABI binaries | `packaging.yml` | owner_gated | tag-triggered; `publish` job attaches to the GitHub Release + deploys a public APT repo |
| WebAssembly package | `release.yml` | owner_gated | only packaged inside a release run |
| Signed release tarball + SLSA + cosign | `slsa-provenance.yml` | owner_gated | tag-triggered, owner-authorized |

(2 `template`, 3 `owner_gated`. The `packaging.yml` surface was corrected from
`template` to `owner_gated` by the B20 adversarial verification pass once its
tag-triggered `publish`-to-GitHub-Release + APT job was confirmed — a published
release artifact is never a dev template.)

## What the gate does **not** do

- It does not publish, pack, sign, tag, or upload anything.
- It does not authorize a release or merge `dev` → `main`.
- It does not decide *when* a package is shipped — only *whether a shipped
  artifact may call itself audited*.


## Package content boundary

Provenance binding proves which commit produced an artifact; the release content
guard proves the archive contains only product payload. Desktop archives are
checked by `ci/check_release_package_contents.py`, which rejects build-tree
libraries from audit, exploit, fuzz, benchmark, standalone test, or unexpected
internal targets. Those libraries may exist in CI build directories, but only the
native `ufsecp*` ABI libraries and the `ultrafast_secp256k1*`
libsecp256k1-compatible ABI libraries may appear under `lib/static` or
`lib/shared` in a shipped release package.

When the owner runs a real release/package workflow, that workflow is expected to
substitute the sentinels with the build's real commit, the committed CAAS bundle
digest, the `audit_gate` verdict, the artifact hash, and the run id — moving the
surface from `template` to `bound`. The gate then verifies the binding holds. The
authorization to run that release stays with the owner.
