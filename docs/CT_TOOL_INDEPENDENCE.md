# CT_TOOL_INDEPENDENCE.md — UltrafastSecp256k1

> Version: 1.0 — 2026-04-21
> Closes CAAS gap **G-8**.
>
> This document defines how UltrafastSecp256k1 backs its
> constant-time claim with **three organisationally-independent
> tools** that must agree before a CT property is recorded as
> verified. A single tool's "no leakage detected" output is a
> starting baseline, not the conclusion.

## 1. Position

UltrafastSecp256k1 makes constant-time claims only on functions
listed in [`PROTOCOL_SPEC.md`](PROTOCOL_SPEC.md) §4. For each such
function the project requires **two-of-three** independent verifiers
to report no leakage; the third must not contradict.

This rule defends against:

- A bug in any single CT tool silently passing a leak.
- A platform-specific tool blind spot (e.g. an architecture the tool
  does not understand).
- A vendor-controlled tool being deprecated or compromised.

## 2. Tool matrix

| Tool | Methodology | Author / vendor | Workflow | Status |
|------|-------------|-----------------|----------|--------|
| **dudect** | Statistical timing test on real hardware (Welch t-test on measured cycles) | Reparaz et al., independent academic | `.github/workflows/cycle-ct.yml` (advisory) + `audit/test_ct_sidechannel.cpp` | Active |
| **Valgrind CT** (`memcheck` "uninitialized" abuse) | Bit-precise data-flow tracking of secret bytes through every memory and ALU op | Valgrind project / FOSS | `.github/workflows/valgrind-ct.yml` + `audit/test_ct_sidechannel.cpp` | Active |
| **ct-verif** (LLVM-based symbolic) | Formal symbolic proof of no secret-dependent control flow / memory address | Almeida et al., independent academic | `.github/workflows/ct-verif.yml` + `audit/test_ct_verif_formal.cpp` | Active |
| **ct-prover (Cryptol-backed)** | Bit-precise property over IR | Galois / Cryptol | `.github/workflows/ct-prover.yml` | Active (advisory cross-check) |
| **ct-arm64** (architecture-specific) | Re-runs dudect/Valgrind on ARM64 to detect arch-specific leaks | UltrafastSecp256k1 in-house | `.github/workflows/ct-arm64.yml` | Active |

The first three are the **primary** verifiers; the fourth and fifth
are cross-checks that increase confidence without being decisive.

## 3. Independence properties

| Pair | Source-of-truth independence? | Methodology independence? | Vendor independence? |
|------|------------------------------|---------------------------|----------------------|
| dudect vs Valgrind CT | Yes (cycles vs data-flow) | Yes (statistical vs precise) | Yes (academic vs FOSS) |
| dudect vs ct-verif | Yes (dynamic vs static) | Yes (statistical vs formal) | Yes (academic vs academic; different teams) |
| Valgrind CT vs ct-verif | Yes (run-time vs compile-time) | Yes (data-flow vs symbolic) | Yes (FOSS vs academic) |

The three primary tools agree on a CT claim only when **all three
methodologies converge on the same conclusion**, which requires that
a leak escape three orthogonal detection mechanisms simultaneously.

## 4. Verification protocol

For every function in `PROTOCOL_SPEC.md` §4:

1. **dudect** runs on real hardware, ≥ 1e7 measurements, with the
   Welch t-test threshold pinned at the published BBR value
   (`|t| < 5` for "no leakage at this measurement count").
2. **Valgrind CT** runs the function with the secret bytes marked
   "uninitialised" via `VALGRIND_MAKE_MEM_UNDEFINED`; any read or
   conditional branch on those bytes triggers a Valgrind error.
3. **ct-verif** symbolically executes the function with secret
   inputs marked tainted; any tainted address or tainted branch
   condition is a proof failure.

The CAAS sub-gate `audit_gate.py --ct-tool-agreement` (planned)
walks the JSON outputs of all three workflows and refuses the
release if any function has fewer than two "no leak" verdicts or any
"leak detected" verdict.

## 5. Failure semantics

| Combination | Verdict |
|-------------|---------|
| 3/3 say "no leak" | **Verified CT** |
| 2/3 say "no leak", 1/3 inconclusive (timeout, unsupported instruction) | **Verified CT (advisory: list missing tool)** |
| 2/3 say "no leak", 1/3 says "leak" | **HARD FAIL — investigate the disagreement** |
| 1/3 says "no leak" | **Not verified** — block the CT claim |
| Any tool says "leak" | **HARD FAIL** |

A tool reporting a leak is never silently downgraded to "advisory".
The disagreement-investigation outcome is a code fix or a
documented tool-bug attribution; the claim does not move to
"verified" until the disagreement is resolved.

## 6. Coverage today

| Function | dudect | Valgrind CT | ct-verif | Verdict |
|----------|--------|-------------|----------|---------|
| `ufsecp_ecdsa_sign*` | OK | OK | OK | Verified |
| `ufsecp_schnorr_sign*` | OK | OK | OK | Verified |
| `ufsecp_ecdh*` | OK | OK | OK | Verified |
| `ufsecp_keygen_*` | OK | OK | OK | Verified |
| `ufsecp_bip32_*` (priv path) | OK | OK | OK | Verified |
| `ufsecp_bip324_*` (handshake KDF) | OK | OK | OK | Verified |
| `ufsecp_musig2_*` (partial sign) | OK | OK | OK | Verified |
| `ufsecp_frost_*` (sign path) | OK | OK | OK | Verified |
| Field / scalar / group primitives in CT layer | OK | OK | OK | Verified |

The full machine-readable verdict table lives in
`gpu_ct_leakage_report.json` (despite the GPU-prefixed filename it
covers the CPU CT layer too) and `build*/py_ct_*report.json`.

## 7. What the three-tool rule does NOT cover

See [HARDWARE_SIDE_CHANNEL_METHODOLOGY.md](HARDWARE_SIDE_CHANNEL_METHODOLOGY.md):

- Power / EM / fault side channels (out of scope by design).
- Hertzbleed-class frequency-scaling channels (RR-002).
- Cross-process Spectre v2 BTI on adversary co-location (RR-001).
- Microarchitectural channels that all three tools cannot model
  (e.g. CPU-internal port contention not exposed in IR).

## 8. Change discipline

Adding a new CT-claimed function requires, in the same commit:

1. Add the function to `PROTOCOL_SPEC.md` §4.
2. Add a row to §6 above with all three verdicts.
3. Wire the function into the relevant CT workflow if the suite does
   not already cover it.

Replacing a CT tool requires a transition window in which the old
and new tools both run and agree, before the old tool is removed.

A tool may be promoted from advisory to primary (or demoted from
primary to advisory) only with a one-release notice in the
changelog.
