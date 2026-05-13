# Continuous Adversarial Audit Standard (CAAS)

**Version 1.0 — UltrafastSecp256k1**

---

## What This Is

This document defines the audit methodology used in this project as a formal,
reproducible specification. It is not a marketing document. It is a technical
standard that can be evaluated, challenged, and compared against alternatives.

The methodology has a name: **Continuous Adversarial Audit System (CAAS)**.

> An audit is not a document. An audit is a continuously verifiable adversarial
> evidence system.

---

## The Problem With Existing Audit Models

The dominant model in the industry is:

- A third party reviews the code for a bounded period (days to weeks)
- The findings are documented in a PDF
- The PDF is published as a "trust badge"
- The engagement ends

This model has a structural flaw: **it is a snapshot, not a system**.

A snapshot tells you the state of the code at the moment of review. It says
nothing about what happens after — after new features, after dependencies
change, after new attack research is published, after the next commit.

The snapshot model also creates perverse incentives:

- Vendors optimize for findable, reproducible bugs (easy to document)
- Novel, logic-level attacks require deep context that a bounded engagement
  cannot build
- The reviewer has no stake in whether the fix holds under subsequent changes
- The "audit passed" badge persists even if the code is completely rewritten

**CAAS replaces the snapshot with a continuously operating adversarial system.**

---

## The Seven Principles

### 1. Every Claim Has Executable Evidence

A security property that cannot be linked to a running test does not exist.
"We believe this is constant-time" is not a claim. "ct-verif, Valgrind taint
analysis, and dudect all pass for this function" is a claim.

Every public security guarantee in this project links to:
- A specific test artifact
- A CI workflow that runs it
- A traceability document that records the link

Claims that cannot be linked are removed.

### 2. Every Discovered Weakness Becomes a Permanent Test

When an attack class is discovered — whether by internal analysis, external
report, published research, or fuzz crash — it becomes a regression test
that runs on every subsequent commit, permanently.

The weakness cannot silently return. The knowledge is encoded in the test
suite, not in someone's memory.

This property is called **irreversible learning**.

### 3. Audit Is Continuous, Not Periodic

Security review does not happen at release time. It happens on every commit,
because every commit is a potential regression surface.

The CI pipeline is the audit infrastructure. Every gate in the pipeline
(static analysis, CT verification, fuzz, differential checks, exploit PoC
suite) runs on every merge.

### 4. Exploit Writing Is Part of Audit

Understanding an attack class academically is not sufficient. The attack must
be expressed as executable code that demonstrates the vulnerability, or
demonstrates that it is not present under specific conditions.

260 exploit PoCs modules exist in this repository. Each one:
- Documents a known attack class (CVE, ePrint, published exploit)
- Attempts the attack against the current implementation
- Passes only if the implementation is not vulnerable
- Runs on every commit

### 5. Adversarial Review Requires Multiple Perspectives

No single reviewer posture covers the full attack surface. This project
deploys structured adversarial review across six distinct roles on every
major change:

| Role | Adversarial Posture |
|------|-------------------|
| Systematic Auditor | Methodical claim→evidence gap detection |
| Bug Bounty Hunter | High-impact, reward-maximizing target selection |
| Attacker / Red Teamer | Assumption-breaking, worst-case adversarial reasoning |
| Documentation Reviewer | Accuracy enforcement; inaccuracy treated as vulnerability |
| Parity Auditor | Cross-backend completeness enforcement |
| Regression Hunter | Pattern-aware history-sensitive re-evaluation |

Each role asks structurally different questions and surfaces different
vulnerability classes.

### 6. Research Ingestion Must Be Systematic and Time-Bounded

New cryptographic attack research is published continuously. The gap between
publication and evaluation against this codebase is bounded by one working
day, not by a release schedule.

The ingestion loop:
1. Monitor ePrint, NVD, and relevant bug bounty disclosures daily
2. Match new signals against the source graph: which functions, call paths,
   or data flows are in scope?
3. Construct a PoC. Evaluate. Patch if vulnerable. Document if not.
4. Store the finding in AI Memory with date, applicability, and outcome.

The result: the project's exploit coverage tracks the state of published
cryptographic attack research with a one-day lag.

### 7. Audit Quality Is Measured by Evidence, Not Authority

An audit is not better because a more prestigious firm performed it.
An audit is better when it covers more attack surface, produces more
executable evidence, and creates more durable regression protection.

The measurable dimensions of audit quality under CAAS:

| Metric | Definition |
|--------|-----------|
| **Exploit Coverage** | Number of known attack classes with executable PoC |
| **Regression Depth** | Number of previously discovered issues with permanent tests |
| **Time-to-Ingest** | Hours from attack publication to PoC evaluation |
| **Time-to-Test** | Hours from vulnerability discovery to CI-enforced regression test |
| **Adversarial Diversity** | Number of distinct adversarial roles applied per major change |
| **CT Coverage** | Percentage of secret-bearing functions verified by all three CT pipelines |
| **Differential Coverage** | Number of nightly cross-checks against reference implementation |
| **Evidence Traceability** | Percentage of public security claims linked to executable evidence |

These metrics are queryable from the source graph and the audit traceability
matrix. They are not self-reported estimates.

---

## Current Metrics (as of v3.64.0)

| Metric | Value |
|--------|-------|
| Exploit Coverage | 189 PoC modules covering distinct attack classes |
| Regression Depth | Every module permanent; 0 removed after discovery |
| Time-to-Ingest | ≤ 1 working day from ePrint/CVE publication |
| Time-to-Test | ≤ 1 working day from discovery to CI gate |
| Adversarial Diversity | 6 distinct roles per major change |
| CT Coverage (x86-64 + arm64) | 3-pipeline: ct-verif LLVM + Valgrind taint + dudect |
| Differential Coverage | 1,300,000+ nightly checks vs reference implementation |
| Formal Algebraic | 45 Cryptol properties covering group law and edge cases |
| Evidence Traceability | All public claims linked; unlinked claims removed |
| Fuzz Coverage | 11 libFuzzer harnesses, continuous, with sanitizers |
| Source Graph Index | 9,071 functions, 8,638 symbol-level audit scores |
| GPU Backend Parity | CUDA + OpenCL + Metal; 0 missing-metal, 0 missing-opencl |

---

## What CAAS Does Not Claim

Honesty is as important as coverage. CAAS does not claim:

- **Absolute security.** No system does. CAAS claims adaptive security:
  when an attack succeeds, the system learns and that class cannot recur.
- **Coverage of unknown unknowns.** Novel attacks by definition have no
  prior PoC. The mitigations for novel attacks are fuzz coverage (11
  harnesses) and CT analysis (three pipelines) — both continuous.
- **Replacement of all external expertise.** An external reviewer may ask
  a question that no existing pattern or role covers. The value of that
  input is real; CAAS narrows the surface where it is necessary, not to zero.
- **Physical side-channel resistance.** Power analysis, EM, fault injection
  — out of scope for this library.
- **Post-quantum security.** secp256k1 is not quantum-resistant. Stated explicitly.

---

## How CAAS Compares to Existing Approaches

| Approach | Coverage Type | Continuous? | Learns? | Adversarial? | Executable Evidence? |
|----------|--------------|-------------|---------|--------------|---------------------|
| Snapshot PDF review | Code review + some fuzzing | No (snapshot) | No | Partially | PDF (not executable) |
| Bug bounty platforms | Adversarial input | No (event-driven) | No (per-researcher) | Yes | Partial |
| Formal verification (Coq, K, Certora) | Mathematical correctness | Depends | No | No | Yes (proofs) |
| Continuous fuzzing (OSS-Fuzz) | Input space / memory safety | Yes | No | No | Crash reproducers |
| **CAAS (this project)** | All of the above, unified | **Yes** | **Yes** | **Yes** | **Yes, executable CI** |

---

## The Compound Intelligence Layer

CAAS operates with two persistent intelligence systems that give it
an architectural advantage over bounded-engagement models:

**AI Memory** (`tools/ai_memory/ai_memory.py`) retains across all sessions:
- Every architectural decision and its rationale
- Every previously evaluated attack class and its applicability assessment
- Every dead end (approaches tried and abandoned, and why)
- Every discovered bug: root cause, fix, and PoC location

This prevents re-evaluating known findings as if they were new, and ensures
that a session six months later begins with full institutional context.

**Code Grapher** (`tools/source_graph_kit/source_graph.py`) provides:
- 9,071 indexed functions with exact file and line ranges
- 4,766 test→function mappings for coverage analysis
- 8,638 symbol-level audit scores across 17 dimensions
- Live call-graph for impact analysis on any proposed change
- Backend parity status for all GPU operations

Together: AI Memory provides the "why and what happened before"; the Code
Grapher provides the "what exists now and where"; combined they enable
precise, evidence-backed adversarial reasoning rather than intuition.

---

## How to Evaluate This Standard

A reviewer who wants to assess whether this project meets the CAAS standard
can run:

```bash
# Full audit package in one command
python3 ci/external_audit_prep.sh

# Source graph with current coverage scores
python3 tools/source_graph_kit/source_graph.py hotspots 20
python3 tools/source_graph_kit/source_graph.py coverage ecdsa_sign

# Exploit PoC suite
./build/unified_audit_runner --all

# CT verification pipeline
python3 ci/run_ct_verif.py
python3 ci/run_dudect.py

# Differential suite
python3 ci/run_differential.py --count 1000000

# Assurance export (machine-readable)
python3 ci/export_assurance.py -o assurance_report.json
```

Every claim in this document is backed by an artifact that a reviewer can
inspect, run, and independently evaluate. That is what "audit" means here.

---

## Relationship to Independent Human Review

CAAS treats human review as most valuable when it is replayable, evidence-driven,
and converted into permanent regression protection:

1. Any review input should be evaluated by the same metrics: how much attack
   surface does it cover, what executable evidence does it produce, and how
   durable is the regression protection it creates?

2. Human review is highest leverage for novel, non-pattern-based reasoning and
   hypothesis-breaking analysis — areas where a different mental model surfaces
   attack paths that no existing pattern would identify.

3. A reviewer engaging with this project does not start from scratch — they
   start from a system that has already systematically evaluated 171 known
   attack classes, run 1.3M differential checks, and verified CT correctness
   on three independent pipelines.

The practical result: review on a CAAS project is targeted at gaps, methodology,
and novel attack hypotheses. Anything accepted as a finding is converted into
executable evidence so the same class cannot silently regress.

---

## Complete Toolchain Inventory

Every script listed here is in the repository, runnable by any reviewer.
These are not aspirational — they are operational, run on CI or on demand.

### Layer 1 — AI + Persistent Memory

| Script / Tool | What It Does |
|--------------|-------------|
| `tools/ai_memory/ai_memory.py` | Persistent SQLite-backed cross-session memory. Every architectural decision, bug root-cause, dead-end, and ePrint evaluation is stored permanently. Each session begins with full institutional context, not a blank slate. |
| `ci/log_ai_review_event.py` | Records structured LLM review events to `docs/AI_REVIEW_EVENTS.json` with role (`auditor`, `attacker`, `bug-bounty`, `performance-skeptic`, `documentation-skeptic`), finding class, and acceptance status. |
| `ci/audit_ai_findings.py` | Separates AI-suggested tests from manually confirmed tests. AI findings are kept in a quarantine bucket, excluded from official audit score totals until confirmed. |

---

### Layer 2 — Attack Research Ingestion (Daily)

| Script / Tool | What It Does |
|--------------|-------------|
| `ci/research_monitor.py` | Fetches ePrint and NVD feeds daily. Filters for secp256k1-relevant signals (`attack`, `nonce`, `ecdsa`, `schnorr`, `frost`, `timing`, `side-channel`, `break`, `exploit`). Compares against `docs/RESEARCH_SIGNAL_MATRIX.json`. Output: new signals not yet evaluated. |
| `ci/nonce_bias_detector.py` | Statistical nonce-bias measurement: MSB/LSB frequency test, chi-squared across all 256 bit positions, Kolmogorov-Smirnov uniformity test, nonce collision detection over 50,000 signatures. Catches Minerva (2020), TPM-FAIL (2019), and lattice-reduction attack preconditions. |
| `ci/rfc6979_spec_verifier.py` | Verifies deterministic nonce derivation against the RFC 6979 test vectors and edge-case spec requirements. |

---

### Layer 3 — Adversarial / Exploit Layer

| Script / Tool | What It Does |
|--------------|-------------|
| `ci/dev_bug_scanner.py` | 28-class static bug scanner. Detects: `CT_VIOLATION` (fast:: in secret path), `SECRET_UNERASED` (scalar without secure_erase), `RANDOM_IN_SIGNING` (non-deterministic RNG in RFC 6979 path), `TAGGED_HASH_BYPASS` (plain sha256 where BIP-340 tagged_hash required), `BINDING_NO_VALIDATION` (public ABI without NULL-check), `DANGLING_ELSE` (goto-fail pattern), `SHIFT_UB`, `DOUBLE_LOCK`, `HARDCODED_SECRET`, `UNSAFE_FMT`, and 18 others. |
| `ci/bug_capsule_gen.py` | Converts bug capsule JSON into: (a) permanent regression test, (b) exploit PoC test, (c) CMakeLists.txt CTest integration. Every discovered bug becomes an executable, permanent CI gate. |
| `ci/auditor_mode.py` | Emulates independent reviewer first-pass: extracts all registered exploit probes from the audit runner, compares against a curated high-risk attack vector baseline, emits machine-readable gap reports. |
| `ci/mutation_kill_rate.py` | Applies AOR / ROR / COR / LOR / BIT mutation operators to `src/cpu/src/`, rebuilds, runs the audit binary. Kill Rate = killed / total. Rate below 80% on any critical subsystem = documented test gap. |
| `ci/stateful_sequences.py` | Stateful API sequence verifier. Tests use-after-error, context reuse across many operations, BIP-32 chained derivation consistency, context destroy/recreate clearing. Finds state-machine bugs invisible to unit tests. |
| `ci/generate_abi_negative_tests.py` | Generates negative tests for every public ABI entry point: null pointers, wrong buffer sizes, out-of-range scalars, invalid pubkey encodings. |
| `ci/invalid_input_grammar.py` | Grammar-based fuzzing input generation for parser entry points. |
| `ci/glv_exhaustive_check.py` | Exhaustive correctness check of the GLV endomorphism decomposition on sampled inputs. |
| `ci/semantic_props.py` | Semantic property tests: commutativity, associativity, identity, inverse, cofactor handling. |

---

### Layer 4 — Constant-Time Verification (3-Pipeline)

| Script / Tool | What It Does |
|--------------|-------------|
| `ci/collect_ct_evidence.py` | Collects and summarizes evidence from all three CT pipelines into a single artifact for traceability. |
| `ci/valgrind_ct_check.sh` | Valgrind Memcheck taint propagation — marks secret bytes undefined, runs signing path, catches any branch or memory address derived from secret data at runtime. |
| `ci/ctgrind_validate.sh` | Ctgrind validation (Valgrind-based CT tool). |
| `ci/verify_ct_disasm.sh` | Disassembly-level CT verification — inspects compiled output for conditional branches on secret-derived values. |
| `ci/cachegrind_ct_analysis.sh` | Cachegrind cache-access timing analysis for cache-timing side-channels. |
| `ci/cross_compiler_ct_stress.sh` | CT stress test across multiple compilers and optimization levels. Catches optimizer-introduced branches. |
| `ci/check_secret_path_changes.py` | Pre-commit enforcement: any change to a CT-annotated file requires simultaneous update to `CT_VERIFICATION.md`, `SECRET_LIFECYCLE.md`, or `SECURITY_CLAIMS.md`. No silent changes to the secret-bearing path. |

---

### Layer 4b — Formal Verification (Machine-Checked Proofs)

| Script / Tool | What It Does |
|--------------|-------------|
| `audit/formal/safegcd_z3_proof.py` | Z3 SMT formal verification of SafeGCD/Bernstein-Yang divstep: 7 theorems, 17 proofs covering GCD preservation, determinant invariant, zeta boundedness, g-convergence, CT mask correctness, 590-step sufficiency (secp256k1), CT≡branching equivalence. Exit code 1 = proof failure = CI blocks. |
| `audit/formal/lean/SafeGCD/Divstep.lean` | Lean 4 core divstep theorems: g_sum evenness, absorbing state, zeta bounds, 9 computational 590-step witnesses for secp256k1 prime boundary values via `native_decide`. |
| `audit/formal/lean/SafeGCD/CTMasks.lean` | Lean 4 CT mask proofs: arithmetic-shift mask binary, negated-parity mask binary, combined mask binary, XOR-negate/identity — 8-bit exhaustive via `native_decide`. |
| `audit/formal/lean/SafeGCD/Equivalence.lean` | Lean 4 CT≡branching equivalence: branchless mask-based divstep = branching reference for all 2²⁴ 8-bit inputs via `native_decide`. |
| `.github/workflows/formal-verification.yml` | CI gate: runs Z3 SMT and Lean 4 prover jobs in parallel on every push/PR touching `src/cpu/src/ct_field.cpp` or `audit/formal/**`. |

---

### Layer 5 — Assurance Export and Traceability

| Script / Tool | What It Does |
|--------------|-------------|
| `ci/export_assurance.py` | Machine-readable assurance report: coverage by dimension, risk matrix, backend parity status, claim→evidence links. Output: `assurance_report.json`. |
| `ci/external_audit_prep.sh` | One command produces the full evidence package: preflight verification, assurance export, traceability artifacts, SBOM, full source archive. Starting point for any independent reviewer. |
| `ci/generate_traceability.sh` | Generates `AUDIT_TRACEABILITY.md`: claim → test function → CI workflow → evidence artifact. |
| `ci/generate_self_audit_report.sh` | Full self-audit narrative report. |
| `ci/audit_gap_report.py` | Queries the source graph for functions with audit scores below risk-tier thresholds. Output: prioritized gap list. |
| `ci/audit_gate.py` | CI gate: compares live audit score against the minimum threshold. Fails the build if the score drops. |
| `ci/auditor_kit.sh` | One-command toolkit for independent reviewers: environment check + evidence collection + gap report. |
| `ci/audit_test_quality_scanner.py` | Scans each audit test for assertion density and coverage depth. Flags tests that are too shallow to be meaningful. |
| `ci/validate_assurance.py` | Validates `assurance_report.json` against the schema. Catches stale or malformed assurance output. |
| `ci/preflight.py` | Platform and toolchain verification before any audit run. |
| `ci/sync_audit_report_version.py` | Keeps version references consistent across all audit artifacts. |
| `ci/sync_module_count.py` | Keeps exploit module count in sync across docs and the audit runner. |

---

### Layer 6 — Source Graph (Code Grapher)

| Script / Tool | What It Does |
|--------------|-------------|
| `tools/source_graph_kit/source_graph.py` | Live engineering index: 9,071 functions, 8,638 symbol-level audit scores across 17 dimensions, 4,766 test→function mappings, call graph, backend parity tracking, hotspot and bottleneck data, `bodygrep` for literal search inside function bodies. Queryable from CI and from any LLM session. |
| `ci/audit_gap_report.py` | Source-graph-driven gap report: which functions have no CT annotation, no test mapping, or audit score below tier threshold. |
| `ci/ci_gate_detect.py` | Detects CI environment and enforces the appropriate gate set for the detected platform. |
| `ci/query_graph.py` | Direct SQL query interface to the source graph database. |

---

### Layer 7 — Build Integrity and Supply Chain

| Script / Tool | What It Does |
|--------------|-------------|
| `ci/verify_slsa_provenance.py` | Verifies SLSA Level 3 provenance attestations for every release artifact. |
| `ci/verify_reproducible_build.sh` | Bit-for-bit reproducible build verification. |
| `ci/generate_sbom.sh` | SBOM (Software Bill of Materials) generation in CycloneDX format. |
| `ci/report_provenance.py` | Provenance chain reporting for release artifacts. |
| `ci/perf_regression_check.sh` | Performance regression detection — blocks commits that regress benchmark results beyond a tolerance band. |
| `ci/generate_dudect_badge.sh` | Generates a machine-readable dudect status badge from the latest CT run. |

---

### How the Layers Connect

```
  ePrint / CVE / NVD (daily)
         │
         ▼
  research_monitor.py ──────────────────────────────────────────────┐
         │                                                           │
         ▼                                                           ▼
  ai_memory.py (context) ◄──── log_ai_review_event.py      RESEARCH_SIGNAL_MATRIX.json
         │
         ▼
  dev_bug_scanner.py │ auditor_mode.py │ nonce_bias_detector.py
         │
         ▼
  bug_capsule_gen.py ──► audit/exploits/ (189 PoC modules, permanent)
         │
         ▼
  unified_audit_runner ──► CT pipeline ──► fuzz ──► differential
         │
         ▼
  export_assurance.py ──► assurance_report.json
         │
         ▼
  external_audit_prep.sh  (one command, full reviewable package)
         │
         ▼
  source_graph.py  ──► audit_gap_report.py ──► audit_gate.py (CI gate)
```

No single layer is sufficient alone. Each layer finds a class of problem
the others cannot. The integration is what constitutes CAAS.

---

*This document is a living specification. As the methodology evolves,
this document is updated in the same commit. Methodology changes without
documentation updates are a policy violation.*
