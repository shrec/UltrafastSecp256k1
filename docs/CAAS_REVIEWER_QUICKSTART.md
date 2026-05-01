# CAAS Reviewer Quickstart

> **For Bitcoin Core reviewers, security researchers, and wallet integrators.**
> You can verify everything on this page without trusting the maintainer.

---

## What you are looking at

UltrafastSecp256k1 is an alternative secp256k1 backend targeting Bitcoin Core.
It ships with CAAS (Continuous Audit as a Service) — a self-contained,
continuously replayable audit system where every security claim maps to
an executable test and every reviewer can independently reproduce the
current assurance state.

**CAAS is not a marketing page.** It is a pipeline. This document tells you
exactly what to run and what to expect.

---

## 5-Minute Overview

| What you want to know | Where to look |
|-----------------------|---------------|
| What is claimed? | [`docs/SECURITY_CLAIMS.md`](SECURITY_CLAIMS.md) |
| What is in scope? | [`docs/AUDIT_SCOPE.md`](AUDIT_SCOPE.md) |
| What is NOT in scope? | [`docs/RESIDUAL_RISK_REGISTER.md`](RESIDUAL_RISK_REGISTER.md) |
| What passed/failed? | [`docs/AUDIT_DASHBOARD.md`](AUDIT_DASHBOARD.md) |
| How does CAAS work? | [`docs/CAAS_PROTOCOL.md`](CAAS_PROTOCOL.md) |
| Why no external audit? | [`docs/CAAS_FAQ.md`](CAAS_FAQ.md) |
| CT verification evidence | [`docs/CT_VERIFICATION.md`](CT_VERIFICATION.md) |
| 205 exploit PoC catalog | [`docs/EXPLOIT_TEST_CATALOG.md`](EXPLOIT_TEST_CATALOG.md) |
| libsecp parity status | [`docs/BACKEND_PARITY.md`](BACKEND_PARITY.md) |
| Thread safety guarantees | [`docs/THREAD_SAFETY.md`](THREAD_SAFETY.md) |
| ABI versioning policy | [`docs/ABI_VERSIONING.md`](ABI_VERSIONING.md) |
| Known residual risks | [`docs/RESIDUAL_RISK_REGISTER.md`](RESIDUAL_RISK_REGISTER.md) |

---

## One-Command Replay

```bash
# 1. Clone and build
git clone https://github.com/shrec/UltrafastSecp256k1.git
cd UltrafastSecp256k1
# Recommended: canonical audit profile under out/audit
python3 ci/configure_build.py audit   # or: cmake -B out/audit -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C out/audit

# 2. Run the full CAAS pipeline (all 5 stages)
python3 ci/caas_runner.py --no-fail-fast --json -o caas_report.json

# 3. Open the interactive audit dashboard in your browser
python3 ci/audit_viewer.py
```

Expected result: all stages PASS, JSON report written. The audit viewer
launches at `http://localhost:8765` with the full evidence tree.

---

## Bitcoin Core Backend — Specific Verification

For reviewers evaluating the Bitcoin Core alternative-backend proposal:

```bash
# Run the Bitcoin Core profile only
python3 ci/caas_runner.py --profile bitcoin-core-backend --json -o btc_core_report.json

# Check shim parser parity (strict accept/reject matching libsecp)
python3 ci/check_libsecp_shim_parity.py

# Check build-mode determinism (no FetchContent, no config.ini required)
python3 ci/check_core_build_mode.py

# Verify the evidence bundle cryptographic integrity
python3 ci/verify_external_audit_bundle.py --json
```

The `bitcoin-core-backend` profile verifies:

| Gate | What it checks |
|------|----------------|
| Parser parity | Every scalar/field boundary uses strict range validation (reject >= n / >= p) |
| CT signing | All secret-path operations route through `secp256k1::ct::*` primitives |
| C++20 isolation | `PUBLIC cxx_std_20` not propagated to consumers |
| Nonce function | ndata/R-grinding compatible with Bitcoin Core's `CKey::Sign` |
| Build determinism | No runtime config required; reproducible with standard CMake |
| ECDSA / Schnorr / Taproot | All shim functions parity-tested against libsecp test vectors |

---

## What CAAS Proves

CAAS makes the following **verifiable** claims:

- **205 exploit PoC tests** covering known attack classes against secp256k1 — all pass on every commit.
- **Constant-time signing** — verified by three independent pipelines (ct-verif/LLVM IR, Valgrind taint, dudect statistical) on x86-64 and arm64.
- **Parser strict parity** — shim boundary functions reject inputs outside the valid domain exactly as libsecp does.
- **Zero ABI-incompatible changes** without an ABI version bump (enforced by test_abi_gate).
- **Self-test on every context create** — library verifies its own arithmetic on first use.
- **Deterministic ECDSA nonces** via RFC 6979 (with hedged variant supporting ndata/R-grinding).
- **BIP-340 Schnorr** conformant to the spec; differential-tested against libsecp.

---

## What CAAS Does NOT Prove

CAAS is honest about its limits. The following are **not claimed**:

- Physical side channels (EM, power, fault injection) — see [RR-006](RESIDUAL_RISK_REGISTER.md).
- Post-quantum security — secp256k1 is classical; see [RR-007](RESIDUAL_RISK_REGISTER.md).
- Compiler-correctness (trust your toolchain).
- Application-layer replay protection (caller responsibility); see [RR-008](RESIDUAL_RISK_REGISTER.md).
- GPU CT (GPU operates exclusively on public data — no secrets ever enter GPU memory).
- Formal proof of algorithmic correctness (this is empirical + symbolic, not Coq/Lean proof).

The complete list is in [`docs/RESIDUAL_RISK_REGISTER.md`](RESIDUAL_RISK_REGISTER.md)
and [`docs/NEGATIVE_RESULTS_LEDGER.md`](NEGATIVE_RESULTS_LEDGER.md).

---

## Reproducing the Evidence Bundle

The current evidence bundle is at:

- `docs/EXTERNAL_AUDIT_BUNDLE.json` — all artifacts with SHA-256 hashes
- `docs/EXTERNAL_AUDIT_BUNDLE.sha256` — detached digest of the bundle itself

Verify integrity independently:

```bash
python3 ci/verify_external_audit_bundle.py --json
```

Regenerate from scratch:

```bash
python3 ci/external_audit_bundle.py
python3 ci/verify_external_audit_bundle.py --json
```

The bundle is regenerated on every push to `dev` by the `caas-evidence-refresh` CI workflow.

---

## Inspecting a Specific Function

If you want to verify how a specific function (e.g. `secp256k1_ecdsa_sign`) is handled:

```bash
# Query the source graph
python3 ci/query_graph.py "secp256k1_ecdsa_sign"

# Check CT coverage for the signing path
python3 ci/audit_gate.py --ct-integrity

# Check shim parser parity for the function
python3 ci/check_libsecp_shim_parity.py --function secp256k1_ecdsa_sign
```

---

## Verifying Source Graph Freshness

The source graph powers most CAAS analysis. Verify it covers the key directories:

```bash
python3 ci/check_source_graph_quality.py
```

If stale, rebuild:

```bash
python3 ci/build_project_graph.py --rebuild
python3 ci/check_source_graph_quality.py
```

---

## Interactive Audit Dashboard

```bash
python3 ci/audit_viewer.py
# Opens http://localhost:8765
```

The viewer shows:
- CAAS pipeline status
- Evidence artifact tree with hashes
- Exploit PoC catalog with attack class mapping
- CT verification evidence
- Residual risk register
- Claim → test → CI traceability
- Profile-specific status (default, bitcoin-core-backend, cpu-signing)

---

## Common Reviewer Questions

**"How do I know the CI results match what I can run locally?"**
Run `python3 ci/caas_runner.py` — it executes the same gates as CI.
The evidence bundle contains git commit hash and dirty-state flag.

**"What if I find a bug in CAAS itself?"**
See [`docs/CAAS_THREAT_MODEL.md`](CAAS_THREAT_MODEL.md) for known threats against CAAS
and their mitigations. Report any issues via GitHub Issues.

**"Why should I trust the evidence bundle?"**
You shouldn't — *verify* it. The bundle is hash-pinned. Run
`verify_external_audit_bundle.py` to confirm every artifact matches its
recorded hash. Then regenerate from scratch and compare.

**"This is a lot. What's the minimum I need to read?"**
- [`SECURITY_CLAIMS.md`](SECURITY_CLAIMS.md) — what is claimed (5 min)
- [`RESIDUAL_RISK_REGISTER.md`](RESIDUAL_RISK_REGISTER.md) — what is not (5 min)
- [`CAAS_FAQ.md`](CAAS_FAQ.md) — common objections (10 min)
- Run `python3 ci/caas_runner.py --profile bitcoin-core-backend` (10 min)

Total: ~30 minutes to a well-grounded first opinion.
