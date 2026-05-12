# CAAS Gap-Closure Roadmap — Self-Contained Evidence

> Version: 1.1 — 2026-04-26
> Status: H-1..H-12 complete (`docs/CAAS_HARDENING_TODO.md`). G-1..G-4
> closed 2026-04-26 (docs created). G-5..G-6 partial. This file
> identifies the **structural gaps that prevent CAAS from being fully
> self-contained, replayable evidence infrastructure** and the work needed to
> close each one.
>
> Goal: when this file is fully checked, CAAS has enough executable evidence
> that independent reviewers focus on methodology, replay, and novel hypotheses
> rather than rediscovering known bug classes.

## Why this file exists

CAAS today (P0–P20, autonomy 100/100) catches what the current automated
evidence suite knows how to check. Remaining gaps are **areas the CAAS model
has not yet automated or pinned as replayable evidence**:

| Review / assurance deliverable | Currently in CAAS? |
|---------------------------------|--------------------|
| Source code review | ✓ scanner + audit_gate |
| Memory-safety sanitizer matrix | ✓ ASan/UBSan/TSan/MSan/Valgrind in CI |
| Constant-time review (IR level) | ✓ ct-verif, ct-prover, valgrind-ct |
| Constant-time review (binary level, per-arch, per-compiler) | **partial** — IR only, no binary CT verifier |
| Fuzzing campaigns | ✓ cflite + cryptofuzz CI |
| Differential testing vs reference | ✓ Wycheproof + project_wycheproof catalog |
| Formal verification | ✓ SAW/Cryptol via formal-verification.yml (advisory) |
| Symbolic execution | ✓ KLEE workflow |
| Supply-chain (SLSA, SBOM, scorecard) | ✓ slsa-provenance + scorecard + cbom + dependency-review |
| Reproducible build | ✓ reproducible-build.yml (single CI provider) |
| Mutation testing | ✓ mutation-weekly |
| Threat model document | ✓ **THREAT_MODEL.md** (closed 2026-04-26) |
| RNG entropy attestation methodology | ✓ **RNG_ENTROPY_ATTESTATION.md** (closed 2026-04-26) |
| Public protocol spec (publishable, with vectors) | **partial** (NORMALIZATION_SPEC + FORMAL_INVARIANTS_SPEC) |
| Spec-to-test traceability matrix (every normative MUST → test ID) | **partial** |
| Compliance stance (FIPS / NIST / CC explicit claim or non-claim) | ✓ **COMPLIANCE_STANCE.md** (closed 2026-04-26) |
| Hardware side-channel methodology (why power/EM deferred) | ✓ **HARDWARE_SIDE_CHANNEL_METHODOLOGY.md** (closed 2026-04-26) |
| Cross-implementation interop matrix (libsecp256k1, OpenSSL, Botan, BC) | **missing** as published doc |
| Multi-CI / clean-room reproducible-build attestation | **partial** (single provider) |
| Two-tool independence for CT verification | **partial** (one CT-verif chain) |
| Public security disclosure / VDP | ✓ SECURITY.md + BUG_BOUNTY.md |
| Reviewer-grade reproducibility bundle | ✓ EXTERNAL_AUDIT_BUNDLE (P19) |
| Exploit PoC catalog (per-CVE / per-ePrint) | ✓ 189 PoCs in `EXPLOIT_TEST_CATALOG.md` + `EXPLOIT_COVERAGE_MAP.md` |
| Attack class taxonomy doc | ✓ `ATTACK_GUIDE.md` (541 lines, 30+ classes) |
| Research-signal tracking (CVE / ePrint mapping) | ✓ `RESEARCH_SIGNAL_MATRIX.json` (578 entries) |
| Exploit backlog tracking | ✓ `EXPLOIT_BACKLOG.md` (cleared, all 7 PoCs landed) |

Every row marked **missing** or **partial** is a CAAS evidence gap. Closing
these rows turns the repository into self-contained, replayable assurance
infrastructure.

### What is already CAAS-strong (do not duplicate)

Four exploit-catalog dimensions are already at audit-firm parity or above
(comparing typical Trail-of-Bits / NCC Group / Cure53 deliverables for
crypto libraries):

| Dimension | Our state | Typical paid-audit deliverable |
|-----------|-----------|--------------------------------|
| PoC test count | 189 | 20–60 |
| Per-CVE/per-ePrint anchoring | every PoC has a citation | usually only critical findings cited |
| Backlog discipline | cleared, conversion standard documented | informal |
| Research-signal matrix | machine-readable JSON, 578 entries | usually narrative-only in report |

The gap registry below therefore focuses on **publishable doc surfaces**
and **automation gates** that wrap this exploit catalog into an
auditor-replacement package — not on adding more PoCs.

---

## Gap registry

### G-1. THREAT_MODEL.md (STRIDE per ABI surface)

**Why it matters.** Every reputable audit deliverable opens with a threat
model. Without one published, an auditor will write one and bill for it,
and reviewers cannot independently verify the "what we defend against"
boundary.

**What to add.**
- `docs/THREAT_MODEL.md` listing assets, trust boundaries, attacker
  models (network attacker, malicious caller, malicious peer in MuSig2/FROST,
  hostile signer, hostile verifier, OS-side timing attacker, hardware
  side-channel attacker), STRIDE table per ABI surface (key gen, sign,
  verify, ECDH, BIP-32, MuSig, FROST, address, GPU offload).
- For each STRIDE row: covered control, residual risk note, link to
  audit/test that exercises it.

**Gate.** `audit_gate.py --threat-model`: every exported ABI symbol must
be in the threat-model surface table; missing rows block.

**Acceptance.** Reviewer can answer "what attacker is this defending
against, where is the boundary, what is the residual?" in under 5 minutes
without reading source.

### G-2. RNG_ENTROPY_ATTESTATION.md

**Why it matters.** RNG quality is the single most-recurring root cause
of ECDSA private-key compromise (Sony PS3, Bitcoin Android 2013, Debian
OpenSSL). Auditors always ask "where does entropy come from, how do you
attest it?" — without a doc, every audit re-asks.

**What to add.**
- Source enumeration for `ufsecp_random_*` (per OS: Linux `getrandom(2)`,
  macOS `getentropy(3)`, Windows `BCryptGenRandom`).
- Failure mode table: what happens on syscall failure, on insufficient
  entropy at boot, on FIPS-mode constraints.
- Statistical attestation: which NIST SP 800-22 / 800-90B tests we run
  on the OS RNG output during CI; what failure threshold triggers FAIL.
- Nonce derivation: explicit citation of RFC 6979 deterministic path
  with no entropy dependency, plus the random-extra-data extension and
  why it cannot weaken the deterministic path.
- Threat scenarios: VM rollback, container clone, embedded boot before
  entropy pool seeded, fork-after-seed.

**Gate.** `audit_gate.py --rng-attestation`: doc exists and references
each `ufsecp_random_*` ABI export and each `*_sign*` deterministic-nonce
path.

**Acceptance.** A wallet vendor integrating UltrafastSecp256k1 can decide
"is the RNG path safe in my deployment?" without reading C code.

### G-3. HARDWARE_SIDE_CHANNEL_METHODOLOGY.md

**Why it matters.** Power / EM / fault-injection side channels are
hardware-deferred in the residual-risk register, but the deferral has no
methodology doc. An auditor will ask "what would change to lift this?"
and right now the answer is implicit.

**What to add.**
- Explicit out-of-scope statement: power, EM, microarchitectural noise
  injection, fault injection (clock glitch, voltage glitch), are deferred
  to operating-environment auditors.
- Methodology that would lift the deferral: dudect on real silicon, MIRA
  / DPA Workstation runs, T-test methodology, sample-count thresholds.
- What we do cover instead: IR-level CT (ct-verif), valgrind-ct, dudect
  on macro-CPU timing, per-arch CT-arm64 + cycle-CT.
- Tracked residual: linked from `RESIDUAL_RISK_REGISTER.md` as a
  permanent residual with explicit acceptance criteria.

**Gate.** `audit_gate.py --hw-sca-methodology`: doc exists and is
referenced from `RESIDUAL_RISK_REGISTER.md`.

**Acceptance.** Reviewer can answer "what does this library NOT defend
against in hardware, and what would it take to defend?" without
guessing.

### G-4. INTEROP_MATRIX.md (cross-implementation)

**Why it matters.** "Bug-compatible with reference" is the only honest
demonstration of correctness for an interoperable crypto library.
Auditors otherwise build it themselves.

**What to add.**
- Per-operation interop table: ECDSA sign/verify, ECDH, Schnorr (BIP-340),
  MuSig2, FROST, BIP-32 child derivation.
- Per-counterparty: libsecp256k1 (reference), OpenSSL, Botan, Bouncy
  Castle, Trezor firmware test vectors, Bitcoin Core integration test.
- Status per cell: PASS / KNOWN_DIVERGENCE_DOCUMENTED / NOT_RUN.
- Test artifact for each PASS cell: a JSON file in `tests/interop/` with
  reproducible vectors and pinned counterpart commit/version.

**Gate.** `audit_gate.py --interop-matrix`: every PASS cell must have a
corresponding test artifact; every KNOWN_DIVERGENCE row must have a
linked residual-risk register entry.

**Acceptance.** Wallet/exchange integrator can pick our library knowing
it interoperates with their existing stack without source diff.

### G-5. SPEC_TRACEABILITY_MATRIX.md

**Why it matters.** Auditors check "is every normative MUST in BIP-340 /
BIP-32 / BIP-352 / RFC 6979 / FROST RFC actually exercised by a test?"
We have `FORMAL_INVARIANTS_SPEC.json` for math invariants but no
spec-clause → test-ID matrix.

**What to add.**
- One row per normative MUST in: BIP-32, BIP-39, BIP-340, BIP-341,
  BIP-352, RFC 6979, RFC 8032 (if Ed25519 stub), FROST IRTF draft, MuSig2
  draft.
- Each row: spec citation (`BIP-340 §3.4 line N`), test ID
  (`tests/test_schnorr_kat.cpp::Bip340VectorN`), audit module if any.
- Auto-generated coverage ratio at the end (covered / total).

**Gate.** `audit_gate.py --spec-traceability`: must hit ≥ 95% MUST
coverage; remaining MUSTs need explicit deferral notes.

**Acceptance.** Reviewer asks "does this implement BIP-340 §3.4
requirement K?" → answer in one grep, not one source review.

### G-6. COMPLIANCE_STANCE.md

**Why it matters.** "Are you FIPS-validated? CC-evaluated?" comes up in
every enterprise integration. An explicit non-claim is the right answer
and prevents misuse.

**What to add.**
- FIPS 140-3: explicit non-claim plus statement of why (algorithm-suite
  outside FIPS scope; Schnorr/BIP-340 not FIPS-approved; constant-time
  primitives match FIPS Annex D guidance but module boundary is not
  drawn).
- NIST SP 800-90A/B/C: non-claim; we use OS RNG which the integrator
  must validate.
- Common Criteria: non-claim; protection profile not asserted.
- Bitcoin BIP compliance: what we DO claim (BIP-32, 39, 340, 341, 352
  conformance per spec-traceability matrix).
- Export-control: ECCN classification statement, US/EU export footing.

**Gate.** `audit_gate.py --compliance-stance`: doc exists and every
"DO claim" row links to traceability matrix evidence.

**Acceptance.** Compliance / procurement reviewer accepts or rejects
the library on documented grounds without phone calls.

### G-7. MULTI_CI_REPRODUCIBLE_BUILD.md + workflow

**Why it matters.** Reproducible build proven on one CI provider proves
"this CI provider produces the same artifact". Reproducible across two
providers proves "the toolchain produces the same artifact regardless
of CI". Auditors care about the latter.

**What to add.**
- Add a second reproducible-build job using a different runtime
  (e.g. `ubuntu-24.04` vs `nixos-stable` on a self-hosted runner).
- Both jobs upload artifact SHA-256 + cosign attestation.
- A new gate (`ci/multi_ci_repro_check.py`) downloads both
  attestations from the latest run and asserts equality.
- Doc explains the methodology, accepted divergence sources (timestamp
  embedding, build-id), and accepted toolchain matrix.

**Gate.** `audit_gate.py --multi-ci-repro`: latest two attestations from
distinct providers must match for the canonical release artifact.

**Acceptance.** Reviewer can verify "this binary is what the source
produces" without trusting a single CI provider.

### G-8. CT_TOOL_INDEPENDENCE.md (two-tool CT proof)

**Why it matters.** A single CT-verif chain has a single point of bug.
Two independent tools agreeing on the same secret-handling boundary is
audit-grade evidence.

**What to add.**
- Document the two tools (current chain: ct-verif/Cryptoline-style + a
  second independent tool such as Binsec/Rel or MicroWalk).
- Per CT-flagged ABI surface, both tools must produce a "leak-free"
  verdict; disagreement is FAIL.
- Add a CI workflow `.github/workflows/ct-independence.yml` running
  the second tool on the same surface.

**Gate.** `audit_gate.py --ct-independence`: per-symbol agreement
table must be 100%.

**Acceptance.** Reviewer can answer "what is the probability our CT
claim is wrong because of a single tool bug?" with a documented
two-tool disagreement search history.

### G-9. PROTOCOL_SPEC.md (publishable, citation-ready)

**Why it matters.** Currently there is no single document an auditor or
researcher can cite as "the UltrafastSecp256k1 protocol spec". Pieces
live in `NORMALIZATION_SPEC`, `FORMAL_INVARIANTS_SPEC.json`,
`FROST_COMPLIANCE`, `SECURITY_CLAIMS.md`. A single citable doc is the
deliverable an auditor would write.

**What to add.**
- One-document overview of the algorithm suite (math primitives, ECDSA
  contract, Schnorr contract, MuSig2/FROST contract, BIP-32 contract,
  BIP-352 contract, ECDH contract, address derivation contract).
- Each section: normative claim + citation to the upstream spec + link
  to the test that demonstrates conformance + link to the formal-invariant
  row.
- Pinned semver and commit; doc may be cited as
  `UltrafastSecp256k1 PROTOCOL_SPEC v3.X.Y`.

**Gate.** `audit_gate.py --protocol-spec`: doc must exist and contain
a section for every ABI surface family; CI updates pinned semver
on release.

**Acceptance.** Researcher can cite the doc in an academic paper without
needing source access.

### G-9b. EXPLOIT_CATALOG ↔ THREAT_MODEL ↔ SPEC traceability join

**Why it matters.** The 257 exploit PoCs, the threat model (G-1), and
the spec traceability matrix (G-5) currently live in three separate
docs. An auditor wants a single join: "for attacker model A targeting
spec clause C, what PoCs exercise it and what is the residual?"

**What to add.**
- `ci/exploit_traceability_join.py`: reads
  `EXPLOIT_TEST_CATALOG.md`, `THREAT_MODEL.md`, and
  `SPEC_TRACEABILITY_MATRIX.md`; emits a single
  `docs/EXPLOIT_TRACEABILITY_JOIN.md` with one row per (attacker,
  spec-clause, PoC) triple.
- Gate failure when a high-severity attacker × in-scope spec clause
  has zero PoC coverage AND no residual-risk register entry.

**Gate.** `audit_gate.py --exploit-traceability`: zero gaps in the
join matrix; new entries added in same commit as new PoCs.

**Acceptance.** Auditor opens one doc and answers "is attack X on
clause Y exercised?" without cross-referencing four files.

### G-10. SECURITY_DISCLOSURE_SLA tightening

**Why it matters.** SECURITY.md exists with a 72h ack SLA. Auditors
also expect: GPG-signed advisory channel, CVE assignment process, named
security contact rotation, public timeline-of-known-incidents file.

**What to add.**
- `SECURITY.md` extension: GPG key fingerprint and rotation policy,
  CVE assignment authority (we either assign via CNA partnership or
  GitHub Security Advisories — declare which), incident-response
  timeline target (acknowledge 72h, fix dev candidate 7d, public
  disclosure 30/60/90d).
- New file `docs/SECURITY_INCIDENT_TIMELINE.md` listing every published
  advisory with timeline data (could start empty with structure).
- `.well-known/security.txt` at repo root for RFC 9116 conformance.

**Gate.** `audit_gate.py --disclosure-policy`: SECURITY.md plus
security.txt plus the incident timeline file all exist; security.txt
is RFC 9116 valid.

**Acceptance.** A reporter discovering a vulnerability can find every
piece of information needed to disclose it within 60 seconds of opening
the repo.

---

## CAAS gate extension: P21 — CAAS Completeness

After all G-1..G-10 land, register a new audit principle in
`AUDIT_MANIFEST.md`:

> **P21 — CAAS Completeness.** Every known review gap is closed by an automated
> CAAS gate or by a published CAAS-pinned document. The set of such gaps and
> gates is enumerated in
> `docs/CAAS_GAP_CLOSURE_ROADMAP.md`.

The gate:

- Checks each G-N item for either (a) presence of the documented
  artifact at the documented path, or (b) presence of the registered
  audit_gate sub-check.
- Fails if any G-N is regressed.

This is the formal statement that every known line item is enumerated,
automated, and CI-blocked.

---

## Execution order (impact-per-effort)

1. G-1 (threat model) — universal audit prerequisite, doc-only.
2. G-2 (RNG attestation) — answers the most-asked auditor question, doc-only.
3. G-6 (compliance stance) — unblocks enterprise procurement, doc-only.
4. G-3 (HW SCA methodology) — closes the residual-risk explanation gap.
5. G-5 (spec traceability) — high-value, mostly aggregation work.
6. G-9 (protocol spec) — citation-ready master doc.
7. G-9b (exploit traceability join) — depends on G-1 + G-5; very small
   script to land once both dependencies exist.
8. G-10 (disclosure SLA tightening) — small additions to SECURITY.md.
9. G-4 (interop matrix) — moderate effort; pulls in real test data.
10. G-7 (multi-CI repro) — workflow + script.
11. G-8 (two-tool CT) — workflow + script (depends on choosing the
    second CT tool).

After all 10 close, register P21 in AUDIT_MANIFEST.md and the
`external-audit-replacement` sub-gate in `audit_gate.py`.

---

## Done criteria for self-contained CAAS evidence

- [ ] G-1..G-10 plus G-9b all closed (per acceptance criteria above).
- [ ] P21 registered and CI-enforced.
- [ ] Existing scorecard score ≥ 9.5 (currently 9.0; OpenSSF dashboard).
- [ ] An independent reviewer presented with the bundle, the threat model,
      the spec, the traceability matrix, and the methodology docs can replay
      the published claims and focus on methodology or novel hypotheses.
