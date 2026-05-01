# Replay Capsule Specification

> Version: 1.0 — 2026-04-27
> Applies to: UltrafastSecp256k1 CAAS pipeline

---

## 1. Purpose and Design Goals

A **replay capsule** is a single JSON file that captures everything a third party needs
to independently reproduce the CAAS (Continuous Audit as a Service) pipeline state for
a specific commit, without any assistance from the repository maintainers.

### Design goals

| Goal | Description |
|------|-------------|
| **Self-contained** | All inputs required to reproduce a pipeline run are recorded in one file |
| **Integrity-pinned** | Every significant output is identified by its SHA-256 hash; a reviewer can detect tampering without trusting the producer |
| **Deterministic** | Replay of the recorded commands on the recorded commit must produce matching output hashes |
| **Honest about residuals** | Known non-determinisms, deferred items, and residual risks are listed explicitly rather than omitted |
| **No maintainer required** | A reviewer with a fresh clone and the tool versions listed in the capsule can verify the full pipeline state independently |

### What a capsule is not

- It is not a replacement for the `docs/EXTERNAL_AUDIT_BUNDLE.json` evidence record; the two artifacts are complementary. The audit bundle records what was claimed; the replay capsule records how to verify the claim.
- It is not a release artifact; it is a reproducibility audit aid.
- It does not replace code review or a formal security audit.

---

## 2. JSON Schema

A valid capsule conforms to the following schema. All fields are required unless
marked optional. String lengths and format constraints are enforcement rules, not
suggestions.

```json
{
  "schema_version": "1.0",
  "generated_at": "<ISO-8601 UTC>",
  "commit_hash": "<full git SHA>",
  "dirty": false,
  "profile": "bitcoin-core-backend",
  "build_flags": {
    "cmake_build_type": "Release",
    "compiler": "clang++-19",
    "compiler_version": "19.x",
    "cxx_standard": "20"
  },
  "tool_versions": {
    "python": "3.x",
    "cmake": "3.x",
    "ninja": "1.x"
  },
  "graph_hash": "<SHA-256 of .project_graph.db>",
  "evidence_bundle_hash": "<SHA-256 of EXTERNAL_AUDIT_BUNDLE.sha256>",
  "stages": [
    {
      "id": "scanner",
      "script": "audit_test_quality_scanner.py",
      "exit_code": 0,
      "duration_s": 12.3,
      "output_hash": "<SHA-256 of stdout>"
    }
  ],
  "replay_commands": [
    "cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release",
    "ninja -C build",
    "python3 ci/caas_runner.py --profile bitcoin-core-backend --json -o caas_report.json",
    "python3 ci/verify_external_audit_bundle.py --json"
  ],
  "expected_evidence_hash": "<SHA-256>",
  "known_residuals": ["RR-001", "RR-002", "RR-003"]
}
```

---

## 3. Field Descriptions

### Top-level fields

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | string | Schema version of this capsule format. Currently `"1.0"`. |
| `generated_at` | string | ISO-8601 UTC timestamp of capsule creation, e.g. `"2026-04-27T14:32:00Z"`. |
| `commit_hash` | string | Full 40-character hex git SHA of the commit that was audited. Must be a real commit reachable from the repository. |
| `dirty` | boolean | Whether the working tree had uncommitted changes when the capsule was generated. Must be `false` for any release or publicly distributed capsule. |
| `profile` | string | CAAS profile identifier. Use `"bitcoin-core-backend"` for Bitcoin Core alternative backend evaluations. Other valid values are defined in `ci/caas_runner.py`. |

### `build_flags` object

Captures the exact CMake and compiler configuration used when the capsule was generated.
A reviewer must use matching flags to reproduce output hashes.

| Field | Type | Description |
|-------|------|-------------|
| `cmake_build_type` | string | CMake build type, e.g. `"Release"`, `"RelWithDebInfo"`. |
| `compiler` | string | Compiler binary name as it would appear on `PATH`, e.g. `"clang++-19"`. |
| `compiler_version` | string | Full compiler version string from `--version`, e.g. `"19.1.0"`. |
| `cxx_standard` | string | C++ standard passed to CMake, e.g. `"20"`. |

### `tool_versions` object

Python, CMake, and Ninja version strings captured from the build environment. Used to
detect version-induced non-determinism during replay.

| Field | Type | Description |
|-------|------|-------------|
| `python` | string | Output of `python3 --version`. |
| `cmake` | string | Output of `cmake --version` (first line). |
| `ninja` | string | Output of `ninja --version`. |

### Hash fields

| Field | Type | Description |
|-------|------|-------------|
| `graph_hash` | string | SHA-256 of `tools/source_graph_kit/source_graph.db` at capsule generation time. Used to verify graph coverage has not changed. |
| `evidence_bundle_hash` | string | SHA-256 of `docs/EXTERNAL_AUDIT_BUNDLE.sha256`. Used to cross-reference the evidence bundle without re-running the full pipeline. |
| `expected_evidence_hash` | string | SHA-256 of `docs/EXTERNAL_AUDIT_BUNDLE.json`. A reviewer who regenerates the bundle can compare this hash to verify that the regenerated bundle matches the recorded one. |

### `stages` array

Each entry records one CAAS stage execution. The set of stages and their ordering is
defined by the profile; the capsule records what actually ran.

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Short stable stage identifier, e.g. `"scanner"`, `"audit_gate"`, `"security_autonomy"`. |
| `script` | string | Python script filename relative to `ci/`, e.g. `"audit_test_quality_scanner.py"`. |
| `exit_code` | integer | Exit code returned by the script. Must be `0` for a passing stage. |
| `duration_s` | number | Wall-clock execution time in seconds, recorded to one decimal place. |
| `output_hash` | string | SHA-256 of the combined stdout+stderr of the stage run. A reviewer replaying the stage can verify output matches this hash. |

### `replay_commands` array

Ordered list of shell commands a reviewer must run, in sequence, to reproduce the full
capsule state. Each command is a string. Commands are recorded verbatim, including all
flags, so that a reviewer does not need to consult any other documentation.

### `known_residuals` array

List of residual risk register IDs (defined in `docs/RESIDUAL_RISK_REGISTER.md`) that
were open at the time of capsule generation. An empty array means no open residuals.
An honest capsule records residuals; omitting them is a capsule integrity violation.

**Current residuals included in bitcoin-core-backend capsules:**

| ID | Summary |
|----|---------|
| RR-001 | CT evidence confidence ceiling — some CT claims carry formal/manual follow-up nuance |
| RR-002 | Local vs GitHub-native workflow parity — some workflow services are GitHub-hosted by nature |
| RR-003 | ROCm/HIP real-device evidence — AMD hardware-backed evidence is intentionally deferred |

---

## 4. How a Capsule Is Generated

Capsules are produced by `ci/create_replay_capsule.py`. The script:

1. Verifies the working tree is clean (`dirty == false`).
2. Captures the current git commit hash.
3. Records compiler and tool versions from the environment.
4. Runs each CAAS stage defined by the requested profile, capturing exit codes,
   wall-clock durations, and SHA-256 hashes of combined stdout+stderr.
5. Hashes `tools/source_graph_kit/source_graph.db`, `docs/EXTERNAL_AUDIT_BUNDLE.sha256`,
   and `docs/EXTERNAL_AUDIT_BUNDLE.json`.
6. Records the `known_residuals` from `docs/RESIDUAL_RISK_REGISTER.md` (active entries only).
7. Writes the capsule JSON to `docs/REPLAY_CAPSULE.json` (or a path specified with `--output`).

```bash
# Generate a capsule for the bitcoin-core-backend profile
python3 ci/create_replay_capsule.py --profile bitcoin-core-backend

# Generate to a custom path
python3 ci/create_replay_capsule.py --profile bitcoin-core-backend \
    --output /tmp/replay_capsule_$(git rev-parse --short HEAD).json
```

The script exits non-zero if:
- The working tree is dirty.
- Any required stage exits non-zero.
- Any required file for hashing is missing.
- The git commit cannot be resolved to a full 40-char SHA.

---

## 5. How a Capsule Is Verified

A reviewer who receives a capsule can verify it using `ci/verify_replay_capsule.py`:

```bash
python3 ci/verify_replay_capsule.py docs/REPLAY_CAPSULE.json
```

The verifier checks:

1. **Schema conformance** — all required fields are present and have correct types.
2. **Commit reachability** — `commit_hash` is reachable in the local git history.
3. **Dirty flag** — warns if `dirty == true` in a capsule presented as a release artifact.
4. **Evidence bundle hash** — recomputes SHA-256 of `docs/EXTERNAL_AUDIT_BUNDLE.sha256`
   and compares against `evidence_bundle_hash`.
5. **Graph hash** — recomputes SHA-256 of `tools/source_graph_kit/source_graph.db` and
   compares against `graph_hash`. A mismatch means the graph has been rebuilt since the
   capsule was generated; this is not necessarily an error, but should be investigated.
6. **Stage output hashes** — re-runs each stage script and compares the output hash
   against the recorded value. A mismatch indicates non-determinism or environment drift.

```bash
# Strict mode: re-run all stages and compare output hashes
python3 ci/verify_replay_capsule.py docs/REPLAY_CAPSULE.json --replay-stages

# JSON output (for CI integration)
python3 ci/verify_replay_capsule.py docs/REPLAY_CAPSULE.json --json -o capsule_verify.json
```

---

## 6. Replay Procedure

Step-by-step instructions for an independent third-party reviewer.

### Prerequisites

- Git, CMake 3.20+, Ninja, Clang 19+ (or GCC 13+), Python 3.10+
- A clean clone of the repository at the commit recorded in the capsule

### Step 1 — Obtain and inspect the capsule

```bash
# Download or locate the capsule
cat docs/REPLAY_CAPSULE.json | python3 -m json.tool | head -40
```

Note the `commit_hash`, `profile`, `build_flags`, and `known_residuals` fields before
proceeding. Confirm that `dirty` is `false`.

### Step 2 — Check out the recorded commit

```bash
git fetch origin
git checkout <commit_hash recorded in capsule>
```

If the commit is not reachable, the capsule may be for a commit that was force-pushed.
This is a red flag that should be reported to the maintainer.

### Step 3 — Match the build environment

Install the compiler version and tool versions recorded in `build_flags` and
`tool_versions`. Version mismatches can cause output hash differences.

### Step 4 — Build

Run the build commands verbatim from `replay_commands[0]` and `replay_commands[1]`:

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build
```

A build failure means either an environment mismatch or a genuine build regression.

### Step 5 — Run the CAAS pipeline

Run the profile command from `replay_commands[2]`:

```bash
python3 ci/caas_runner.py --profile bitcoin-core-backend --json -o caas_report.json
```

### Step 6 — Verify the evidence bundle

Run the verification command from `replay_commands[3]`:

```bash
python3 ci/verify_external_audit_bundle.py --json
```

### Step 7 — Cross-check hashes

```bash
python3 ci/verify_replay_capsule.py docs/REPLAY_CAPSULE.json --replay-stages
```

A clean run produces exit code `0` and a JSON report with all checks passing.

### Step 8 — Evaluate residuals

Review each entry in `known_residuals` against `docs/RESIDUAL_RISK_REGISTER.md`. For
each residual, decide independently whether the documented justification is acceptable
for your evaluation context. Residual acceptance is the reviewer's judgment, not the
maintainer's.

---

## 7. Validation Rules

The following rules are enforced by `ci/verify_replay_capsule.py`. A capsule that
violates any of these is considered malformed.

| Rule | Constraint |
|------|-----------|
| `schema_version` | Must be the string `"1.0"` |
| `commit_hash` | Must be exactly 40 lowercase hex characters |
| `dirty` | Must be the boolean `false` for any capsule presented as a release or audit artifact |
| `generated_at` | Must parse as a valid ISO-8601 datetime with UTC designator (`Z` or `+00:00`) |
| `profile` | Must match a profile name known to `ci/caas_runner.py` |
| `evidence_bundle_hash` | Must be exactly 64 lowercase hex characters (SHA-256) |
| `graph_hash` | Must be exactly 64 lowercase hex characters |
| `expected_evidence_hash` | Must be exactly 64 lowercase hex characters |
| `stages[*].exit_code` | Must be `0` for a passing capsule |
| `stages[*].output_hash` | Must be exactly 64 lowercase hex characters |
| `known_residuals` | Must be an array; entries must match IDs in `docs/RESIDUAL_RISK_REGISTER.md` |
| `replay_commands` | Must be a non-empty array of non-empty strings |

---

## 8. Relationship to Other Evidence Artifacts

| Artifact | Purpose | Relationship to capsule |
|----------|---------|------------------------|
| `docs/EXTERNAL_AUDIT_BUNDLE.json` | Records what claims were made and what evidence files were hashed | The capsule hashes this bundle and records the expected hash |
| `docs/EXTERNAL_AUDIT_BUNDLE.sha256` | Digest file for the bundle JSON | The capsule records the SHA-256 of this file in `evidence_bundle_hash` |
| `docs/RESIDUAL_RISK_REGISTER.md` | Human-readable risk register | Active entry IDs are recorded in `known_residuals` |
| `tools/source_graph_kit/source_graph.db` | Project knowledge graph | Hashed in `graph_hash` to detect coverage drift |
| `caas_report.json` | CAAS pipeline output for a specific run | Produced during replay; compared against recorded stage hashes |
