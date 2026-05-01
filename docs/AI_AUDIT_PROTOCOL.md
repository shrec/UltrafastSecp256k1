# AI Audit Protocol

Defines how AI-assisted review is used inside the UltrafastSecp256k1 assurance model.

This protocol treats LLMs as adversarial review amplifiers, not as sources of truth.
An AI finding is only accepted when it is reproduced, narrowed, and converted into
deterministic repository evidence.

---

## Purpose

AI-assisted review is used to simulate multiple external viewpoints at high cadence:

1. Independent auditor
2. Adversarial attacker
3. Bug bounty hunter
4. Performance skeptic
5. Documentation skeptic
6. API misuse reviewer

The goal is not to replace tests, proofs, or human judgment.
The goal is to increase the rate at which blind spots are discovered and fed back
into the repository's reproducible audit framework.

Completeness rule:

AI-assisted review should help close coverage by failure class, not merely add
more findings or more text. A review loop is only useful when it reduces the
set of known classes that still lack deterministic repository evidence.

---

## Core Rule

No AI finding counts as assurance by itself.

An AI finding only becomes part of the assurance system when it is converted into
one or more of the following:

1. A failing test that is then fixed
2. A new exploit-style PoC test
3. A new invariant or audit assertion
4. A new CI gate or workflow adjustment
5. A corrected benchmark or documentation claim
6. A graph/config update that closes a review blind spot

---

## Review Modes

### 1. Auditor Mode

Use when asking the model to evaluate correctness, assurance coverage, or missing tests.

Expected outputs:

1. Findings ordered by severity
2. File references
3. Assumptions and confidence
4. Suggested deterministic follow-up actions

### 2. Attacker Mode

Use when asking the model to reason like a malicious user or exploit author.

Expected outputs:

1. Candidate exploit paths
2. Inputs most likely to break assumptions
3. Parser boundary risks
4. Invariant failures worth turning into PoC tests

### 3. Bug Bounty Mode

Use when looking for practical, reportable bugs with reproduction value.

Expected outputs:

1. Concrete bug scenarios
2. Trigger conditions
3. Potential impact
4. Minimal repro suggestions

### 4. Performance Skeptic Mode

Use when evaluating benchmark claims, regressions, or overfitted tuning.

Expected outputs:

1. Likely benchmark blind spots
2. Missing hardware coverage
3. Overclaim risks
4. Candidate regression gates

### 5. Documentation Skeptic Mode

Use when checking whether public wording still matches repository truth.

Expected outputs:

1. Stale counts
2. Inconsistent capability claims
3. Missing scope qualifiers
4. Docs that should be linked or downgraded

---

## Required Workflow For Accepted Findings

Every accepted AI finding should move through this pipeline:

1. Record the raw finding and the review mode.
2. Reproduce or falsify it locally.
3. If valid, classify it.
4. Convert it into deterministic repository evidence.
5. Update docs, workflows, or tests.
6. Store the outcome in memory or changelog notes when appropriate.

Accepted finding classes:

1. `bug`
2. `security`
3. `performance`
4. `docs-drift`
5. `coverage-gap`
6. `graph-gap`
7. `false-positive` (kept for future prompt tuning)

---

## Review Event Log

Accepted, rejected, and unconfirmed AI review outcomes should be recorded in:

- `docs/AI_REVIEW_EVENTS.json`

The canonical append utility is:

- `ci/log_ai_review_event.py`

Required fields per event:

1. `event_id`
2. `reviewed_at`
3. `review_mode`
4. `finding_class`
5. `status`
6. `target`
7. `summary`
8. `reproduced`
9. `repository_evidence`
10. `resulting_changes`

Rule:

Accepted findings must reference concrete repository evidence and the exact files changed as a result.

---

## Evidence Conversion Matrix

| Finding Type | Required Conversion |
|--------------|---------------------|
| Correctness bug | Add or update deterministic test |
| Exploit path | Add exploit PoC test or invariant |
| CT concern | Add CT test, Valgrind CT case, or formal-check coverage |
| Perf concern | Add regression gate, benchmark note, or rerun artifact |
| Docs drift | Update claim and add source-of-truth link |
| Coverage blind spot | Add workflow, test target, or matrix row |
| Graph blind spot | Update source graph config/tooling and rebuild |

## Failure-Class Coverage Rule

The self-audit system should be evaluated against named problem classes rather
than total test count alone. At minimum, review and audit expansion should keep
covering these classes:

1. Arithmetic correctness
2. Serialization and parser boundaries
3. Constant-time regressions and secret-data routing
4. Protocol misuse and transcript binding failures
5. ABI misuse and hostile-caller behavior
6. GPU host/runtime misuse and backend capability drift
7. Documentation drift and public overclaim risk
8. Benchmark methodology and publishability drift

For each accepted finding, the preferred outcome is one of:

1. Expand coverage for an existing class
2. Add a missing class-to-evidence mapping
3. Record a residual-risk item when deterministic coverage is not yet possible

Rule:

Do not treat high raw assertion counts as sufficient by themselves. If a known
failure class lacks a deterministic surface, the self-audit system is still
incomplete regardless of total test volume.

---

## Metrics To Track

These metrics should be tracked over time:

1. AI findings accepted vs rejected
2. Accepted findings converted into tests
3. Accepted findings converted into documentation fixes
4. Accepted findings converted into workflow gates
5. False-positive rate by review mode
6. Time from finding to deterministic evidence

---

## Safety Rules

1. Never treat AI output as proof.
2. Never ship a claim based only on AI review text.
3. Never store secrets, keys, or credentials in prompts or artifacts.
4. Prefer smaller, scoped review tasks over vague broad prompts.
5. When a finding cannot be reproduced, label it explicitly as unconfirmed.

---

## Integration Targets

The protocol becomes stronger when tied to the following repository surfaces:

1. `docs/ASSURANCE_LEDGER.md`
2. `AUDIT_GUIDE.md`
3. `docs/CI_ENFORCEMENT.md`
4. `tools/source_graph_kit/source_graph.py`
5. exploit-style audit tests under `audit/test_exploit_*.cpp`

---

## Immediate Next Steps

1. Add a short reviewer checklist for each AI review mode.
2. Feed repeated false positives back into prompt and workflow tuning.
3. Expand the event log with performance-skeptic and attacker-mode entries.
4. Link accepted review events directly to exploit PoCs and workflow gates where possible.