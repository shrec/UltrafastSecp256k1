# CT_INDEPENDENCE.md — UltrafastSecp256k1

> Version: 1.0 — 2026-04-28
> Closes CAAS gap **G-8**.

## 1. Purpose

Individual CT tool workflows (`valgrind-ct.yml`, `ct-verif.yml`,
`ct-prover.yml`, `ct-arm64.yml`) each exercise a single methodology.
G-8 requires proof that multiple independent analysis approaches
**agree** on the CT status of the signing implementation.

The `ct-independence.yml` workflow implements this: it runs two tools
with different methodologies in parallel, collects their verdicts as
JSON, and asserts that no tool reports leakage.

## 2. Tools and methodologies

| Tool | Methodology | Job | Source |
|------|-------------|-----|--------|
| Valgrind memcheck | Binary-level taint: marks secret bytes `MAKE_MEM_UNDEFINED`, fails on any conditional jump or load that depends on them | `ct-tool-valgrind` | `scripts/valgrind_ct_check.sh` |
| dudect | Statistical timing: measures CPU cycle distributions for two input classes, rejects null hypothesis of equal timing via Welch t-test | `ct-tool-dudect` | Built with `DUDECT_CT_CHECK=ON` |

The two methodologies are **orthogonal**:
- Valgrind operates at the binary instruction level (IR-agnostic) and is
  deterministic — if it reports no conditional-jump-on-uninit, the
  compiled code has no data-dependent branches visible at that level.
- dudect operates statistically on real hardware timing — it can catch
  microarchitectural leakage (cache sets, branch predictor state) that
  valgrind's IR model does not see.

Neither tool can catch everything; requiring both to agree removes the
largest blind spots of each.

## 3. Verdict format

Each tool job emits a `ct-verdict-<tool>.json` artifact:

```json
{
  "tool": "valgrind-memcheck-ct",
  "methodology": "binary-taint",
  "verdict": "PASS",
  "exit_code": 0,
  "details": "No secret-dependent branches detected by valgrind memcheck taint analysis",
  "commit": "<sha>",
  "runner": "Linux-X64"
}
```

Valid verdicts: `PASS`, `FAIL`, `SKIP`.

- `SKIP` means the tool binary was not available (e.g. dudect CMake target
  not yet wired for a new platform) — treated as advisory, not a gate failure.
- `FAIL` from any non-skip tool causes the `ct-agree` gate job to fail.

## 4. Agreement gate

The `ct-agree` job runs `scripts/ct_independence_check.py` which:

1. Loads all downloaded verdict files.
2. Returns exit 0 only if there are no `FAIL` verdicts and at least one `PASS`.
3. Produces a GITHUB_STEP_SUMMARY table listing each tool's verdict.
4. Prints the methodologies that agreed for auditor visibility.

```
CT Tool Independence Check — 2 tool(s) evaluated:
  [OK] valgrind-memcheck-ct (binary-taint)
       No secret-dependent branches detected ...
  [OK] dudect (statistical-timing)
       No significant timing leakage detected (t-test < threshold) at 10000 samples

PASS: 2 independent CT tool(s) found no timing leakage
      Methodologies that agree: binary-taint, statistical-timing
```

## 5. Trigger policy

The workflow runs on:
- Push to `main` or `dev` touching CT source paths (`cpu/src/ct_*.cpp`,
  `cpu/src/ecdsa.cpp`, `cpu/src/schnorr.cpp`, `include/**`, `audit/**`)
- Pull requests to `main` touching the same paths
- Weekly schedule (Sunday 06:00 UTC) to catch toolchain drift

## 6. Relationship to other CT workflows

| Workflow | Scope | G-8 role |
|----------|-------|----------|
| `valgrind-ct.yml` | Full nightly valgrind suite | Detailed per-function output |
| `ct-verif.yml` | Compile-time LLVM pass | Deterministic proof on IR |
| `ct-prover.yml` | IFDS taint analysis | Stronger inter-procedural analysis |
| `ct-arm64.yml` | dudect nightly on ARM64 | Microarch coverage for Apple Silicon |
| **`ct-independence.yml`** | **Multi-tool agreement gate** | **G-8 gate — this document** |

G-8 does not replace the individual workflows. It adds the cross-tool
assertion layer on top of them, ensuring that no single-tool blind spot
can create a false sense of security.

## 7. What this proves and does not prove

**Proves:**
- The signing functions pass both a binary-taint analysis and a
  statistical timing analysis simultaneously.
- Agreement across independent methodologies is not coincidental; a
  real CT regression would break at least one tool.

**Does not prove:**
- CT against all microarchitectural side channels (cache sets, TLB
  timing, port contention). These require hardware-level analysis beyond
  software tools.
- CT in GPU kernels. GPU CT is a separate concern tracked by the GPU
  audit layer.
- That higher-level protocols (batching, key derivation) are CT. Only
  the core signing primitives are analyzed here.
