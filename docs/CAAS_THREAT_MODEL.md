# CAAS Threat Model

**UltrafastSecp256k1 — Threats Against the Audit System Itself**

Version: 1.0
Date: 2026-04-27
Status: Active

---

## 1. Overview

CAAS (Continuous Audit as a Service) is a 5-stage pipeline that runs on every
commit: Static Analysis → Audit Gate (P0–P20) → Security Autonomy → Bundle
Produce → Bundle Verify. Its job is to produce independently verifiable
assurance evidence for the cryptographic library.

Like any control plane, CAAS has its own attack surface. A compromised or
degraded audit system can produce green results from broken code. This document
identifies the ways CAAS can be undermined, what mitigations exist, and what
residual risks remain even when every mitigation is active.

This document is complementary to, and distinct from,
[THREAT_MODEL.md](THREAT_MODEL.md), which covers threats against the
cryptographic library itself. The threat model here is about threats against
the trust infrastructure — the audit system that makes claims about the library.

---

## 2. Threat Categories

### T1 — Stale Evidence

**Description**: Evidence files (timing measurements, audit JSON reports,
differential test outputs, coverage artifacts) are generated at a point in
time and stored in the repository or CI artifact cache. If code changes after
evidence is generated but before the next CI run, or if a developer commits
pre-generated evidence without regenerating it, the pipeline may pass while
evaluating evidence that does not correspond to the current code.

**Attack scenario**: A developer makes a correctness fix to a signing path that
incidentally degrades a timing property. Old timing evidence passes the gate.
The stale evidence is bundled and shipped, claiming CT compliance that no longer
exists for the current code.

**Mitigations**:
- `ci/audit_gate.py` `check_freshness()` function enforces a maximum
  evidence age of 7 days. Evidence older than this causes a hard gate failure
  regardless of content.
- A dedicated `caas-evidence-refresh` CI workflow runs nightly, unconditionally
  regenerating all evidence artifacts and committing them back. Any code change
  that degrades evidence will be caught within 24 hours.
- Bundle Verify (Stage 5) re-checks artifact timestamps against the commit
  timestamp. A bundle with evidence older than the most recent relevant commit
  is rejected.

**Residual risk**: A fast adversary who pushes a bad commit and immediately
triggers a manual bundle produce (before nightly refresh) could create a bundle
whose evidence is "fresh" (generated after the commit) but evaluated against a
previous build. This requires write access to the repository and CI, which is
treated as a maintainer-level threat (see Section 3).

---

### T2 — Broken Source Graph

**Description**: The CAAS pipeline uses the SQLite source knowledge graph
(`tools/source_graph_kit/source_graph.db`) to map symbols to files, functions
to tests, and test coverage to security claims. If the graph is not rebuilt
after code changes, queries return stale symbol maps. A new function may not
appear in the graph, an old function that was deleted may still appear, or
coverage entries may point to tests that no longer exist.

**Attack scenario**: A new signing entry point is added to `src/cpu/src/ecdsa.cpp`
but the graph is not rebuilt. The graph still reports full coverage. The audit
gate passes because it queries the graph, not the source files directly.

**Mitigations**:
- `ci/check_source_graph_quality.py` runs as CAAS Stage 0. It compares
  the graph's indexed file list against the current working tree and fails if
  any `.cpp` or `.hpp` file in audited directories is absent from the graph
  or has a graph mtime older than its filesystem mtime.
- Gate check P6 ("graph freshness") enforces that the graph was rebuilt after
  the most recent modification to any file in `cpu/`, `include/`, `audit/`, or
  `ci/`.
- The graph rebuild is a required step in the evidence refresh workflow. It
  cannot be skipped without failing the freshness check.

**Residual risk**: Files outside the indexed directory set will not be covered
by the graph. The `source_graph.toml` configuration must be kept up to date
when new source directories are added. This is a configuration hygiene
requirement, not an automatic mitigation.

---

### T3 — Vacuous Tests

**Description**: A test that always returns pass regardless of the
implementation provides no assurance. Vacuous tests can arise from:
- Missing assertions (test runs but never calls any check)
- Assertions on fixed constants rather than on computed values
- Tests that catch exceptions and swallow them as "expected"
- Tests that depend on a reference implementation that shares the same bug

A vacuous test is indistinguishable from a meaningful test in a CI log unless
the test's kill rate under mutation testing is measured.

**Attack scenario**: An exploit PoC is added for a known vulnerability. The
PoC constructs the attack input and calls the relevant function, but the
assertion checks the wrong output field. The test passes on every version of
the code, including versions that are still vulnerable. The audit gate reports
the exploit as covered.

**Mitigations**:
- `ci/audit_test_quality_scanner.py` performs static analysis on all
  registered exploit PoC files. It checks for: at least one assertion-level
  call (`ASSERT_*`, `REQUIRE_*`, `CHECK_*`, or explicit abort), no unreachable
  assertion patterns (assertion inside dead branch), and no trivially constant
  inputs to the function under test.
- Mutation testing (CAAS Stage P2) applies Mull or an equivalent mutation
  framework to the exploit PoC suite and computes a kill rate. A kill rate
  below 70% causes a P2 advisory failure. Advisory failures block bundle
  production when the kill rate falls below 50%.
- Differential testing against libsecp256k1 (scripts in `tests/differential/`)
  provides an independent reference. A test cannot be vacuous if it compares
  against libsecp output and libsecp gives a different answer.

**Residual risk**: Mutation testing cannot catch all forms of vacuity, in
particular tests that depend on a property that is coincidentally preserved
across all mutations. The scanner also cannot detect all dead-code patterns in
complex C++ template expansions. Human review of newly added tests is expected
as a complementary control.

---

### T4 — Un-wired Exploit Tests

**Description**: An exploit PoC file (`audit/test_exploit_*.cpp`) may exist
in the repository — building, linking, and passing as a standalone CTest
binary — without being registered in `audit/unified_audit_runner.cpp`. In this
state, the test is not part of the CAAS pipeline. It will not appear in the
JSON audit report. The exploit class it covers will be silently absent from
the assurance bundle.

This is the most common form of incomplete test coverage, because the path of
least resistance when writing a new PoC is to add only the CTest target.

**Attack scenario**: A developer adds `audit/test_exploit_new_bias.cpp` for a
new nonce bias class. CTest passes. The audit gate reports the test as covered
(because it queries the graph, and the graph was rebuilt). But the gate query
uses the runner registration list, not the file list. The test is not wired.
The bias class is not actually verified by the CAAS pipeline.

**Mitigations**:
- `ci/check_exploit_wiring.py` runs as CAAS Stage 0 (before any other
  check). For every file matching `audit/test_exploit_*.cpp`, it checks that
  the corresponding `_run()` symbol appears in
  `audit/unified_audit_runner.cpp`. Any missing registration causes an
  immediate hard failure with a message identifying the specific file.
- The same check is a required gate in both `.github/workflows/preflight.yml`
  and `.github/workflows/caas.yml`. It cannot be bypassed by skipping one
  workflow.
- `audit/CMakeLists.txt` requires that every file added to the
  `add_executable(unified_audit_runner ...)` target also has a corresponding
  forward declaration in `unified_audit_runner.cpp`. CMake configure will fail
  if the runner source list and the declaration list are out of sync (enforced
  by a configure-time consistency check).

**Residual risk**: A developer could add a file to `unified_audit_runner.cpp`
with a wired registration but an implementation that does nothing (`_run()`
returns 0 immediately). The wiring check cannot detect this; it only checks
that the symbol is referenced. This reduces to the T3 (vacuous test) threat,
mitigated by T3 controls.

---

### T5 — CI Bypass

**Description**: If code can be merged to `main` without passing the CAAS
pipeline, the assurance claims attached to `main` do not reflect the actual
pipeline results. The simplest form is a direct push to `main` that bypasses
branch protection.

**Attack scenario**: A maintainer (or compromised maintainer account) directly
pushes a commit to `main` that includes a security regression. The commit
bypasses the pre-merge CAAS requirement. The bundle attached to the release
refers to evidence from a previous clean commit.

**Mitigations**:
- GitHub branch protection is configured on `main` with:
  - Required status checks: all CAAS pipeline jobs must pass
  - Require pull request reviews: at least 1 approval required
  - Restrict pushes: no direct push, including by administrators
- A GitHub ruleset (separate from branch protection) is configured with
  `enforcement=active`, which cannot be bypassed even by organization owners
  through the normal UI.
- The Bundle Verify step (Stage 5) records the commit SHA of the HEAD that
  produced the bundle. If the deployed bundle SHA does not match the `main`
  HEAD SHA, the bundle is invalid by definition.

**Residual risk**: GitHub platform-level access by a sufficiently privileged
GitHub employee, or a GitHub infrastructure compromise, could bypass these
controls. This is treated as the same threat class as nation-state compromise
of the CI infrastructure (Section 3).

---

### T6 — Malicious Evidence Bundle

**Description**: An evidence bundle is a ZIP or tarball containing audit JSON
reports, timing measurements, coverage data, and a manifest. If a bundle is
tampered with after generation — report pass/fail flipped, timing values
adjusted, coverage entries added — and the tampered bundle is used as the
assurance artifact for a release, the release claims are falsified.

**Attack scenario**: An attacker with access to the artifact storage location
(GitHub Actions artifact cache, an S3 bucket, a release attachment) modifies
the JSON in the bundle to change a FAIL result to PASS for a critical gate.
The bundle hash is not checked by the downstream consumer. The release ships
with falsified assurance.

**Mitigations**:
- Every bundle produced by Stage 4 (Bundle Produce) includes a
  `EXTERNAL_AUDIT_BUNDLE.sha256` file containing the SHA-256 hash of every
  file in the bundle, computed at the time of production.
- `ci/verify_external_audit_bundle.py` recomputes the hashes at verify
  time (Stage 5) and fails if any hash does not match. This is the mandatory
  final step before a bundle is attached to a release.
- The bundle manifest includes the git commit SHA, the CI run ID, and the
  pipeline version. Any modification that changes content will invalidate the
  hashes. Any modification that changes only the hash file will not match the
  expected hash of the hash file itself (the hash file is itself hashed by the
  manifest).
- Released bundles are signed with a GPG key held by the repository owner.
  Signature verification is documented in `docs/AUDITOR_QUICKSTART.md`.

**Residual risk**: Compromise of the GPG key used for bundle signing would
allow production of indistinguishable forged bundles. The key must be held
offline and its compromise would require a separate incident disclosure.

---

### T7 — Toolchain Compromise

**Description**: If the compiler, linker, or any build tool in the CI
environment is compromised (e.g., via a malicious package update in the CI
runner's package manager), the compiled binary may differ from what the source
code specifies. Evidence generated from this binary — timing measurements,
coverage data, differential test results — reflects the behavior of the
compromised binary, not the intended implementation.

**Attack scenario**: A CI package mirror serves a trojaned version of `clang`
that inserts a timing channel into the CT field multiplication. Timing evidence
is collected from this binary. The evidence passes (because the trojan is
subtle). A clean binary produced by the end user does not have the timing
channel.

**Mitigations**:
- Reproducible builds: `docs/REPRODUCIBLE_BUILDS.md` specifies the exact
  toolchain versions, flags, and build steps required to produce a
  bit-for-bit identical binary from source. An auditor can reproduce the build
  independently and compare.
- Multi-CI cross-validation: the pipeline runs on both GitHub Actions
  (ubuntu-22.04 runner) and a separate self-hosted runner with a different
  package mirror. Divergent results between runners trigger an alert.
- SBOM: the pipeline generates a Software Bill of Materials (CycloneDX format)
  listing every build dependency with its version and hash. This is included
  in the evidence bundle.
- CAAS Stage 1 (Static Analysis) runs on source, not on the compiled binary.
  Static results are independent of the compiler.

**Residual risk**: A sufficiently sophisticated toolchain compromise that
produces identical binaries across multiple independent build environments
(a rare capability, associated with nation-state threat actors) is not
detectable by reproducible build checks. This is treated as an out-of-scope
residual risk.

---

### T8 — Green Pipeline With Incomplete Scope

**Description**: The CAAS pipeline can produce a fully green result while
covering only a subset of the library's security-relevant surface. If the
pipeline's scope definition omits important components, the green result is
technically accurate but misleading. A consumer who reads "CAAS passed" may
infer full coverage when coverage is partial.

**Attack scenario**: A new signing function `ecdsa_sign_batch_streaming` is
added to `src/cpu/src/ecdsa.cpp`. It is not included in the CAAS scope definition.
The pipeline passes with no reference to this function. The function has a
timing vulnerability. The pipeline result is cited as evidence of CT compliance.

**Mitigations**:
- CAAS uses named profiles, each of which specifies an explicit in-scope
  surface list and an explicit out-of-scope list. Every bundle produced by
  Stage 4 records the profile name and version. The assurance claim is
  scoped to the profile, not to the whole library.
- The source graph quality gate (T2 mitigation) checks that all files in
  audited directories appear in the graph. If `ecdsa.cpp` is in scope, all
  functions it contains must appear in the graph, and the coverage map must
  account for each one (either covered or explicitly marked `NOT_IN_SCOPE`).
- `AUDIT_SCOPE.md` is a required document that explicitly lists in-scope and
  out-of-scope components. Any new source file in `src/cpu/src/` must be added to
  this document before it can land in `main`.

**Residual risk**: The scope definition itself may omit components. The
`NOT_IN_SCOPE` label in the coverage map must be audited periodically.
Components marked `NOT_IN_SCOPE` are not covered by the assurance claim and
must be clearly communicated to downstream consumers.

---

### T9 — Self-Referential Evidence

**Description**: Evidence that cites itself as proof of its own validity
provides no assurance. For example: an audit report that claims "this report
was generated by a trusted pipeline" with no external verification of that
claim, or a test that imports the implementation and uses its own output as
the expected value.

**Attack scenario**: A timing test generates a timing measurement, stores it
as a baseline, and then on subsequent runs checks that the measurement matches
the baseline. The baseline was generated from a version of the code with a
timing vulnerability. Every subsequent run passes because it compares against
the (vulnerable) baseline rather than against an absolute threshold or an
independent reference.

**Mitigations**:
- The CAAS evidence chain is: claim → test → CI artifact → independently
  verifiable artifact. At no point does an artifact cite itself.
- Timing gate thresholds are specified in absolute nanoseconds in
  `audit/timing_thresholds.json`, not derived from baseline measurements.
  The thresholds are reviewed and updated by a human on a documented schedule.
- Differential test expected values come from libsecp256k1 (an independent
  implementation), not from UltrafastSecp256k1's own output.
- Wycheproof test vectors are sourced from the Google Wycheproof repository
  at a pinned commit. They are not generated by the library under test.

**Residual risk**: The absolute timing thresholds in `timing_thresholds.json`
were originally calibrated against a reference machine. If that machine's
timing behavior was itself compromised at calibration time, the thresholds
could be set too loosely. Threshold calibration should be replayable and
periodically reviewed as part of CAAS evidence governance.

---

### T10 — Evidence Freshness Reporting Error

**Description**: The audit dashboard or status page may display evidence dates
that do not reflect the actual evidence timestamps. If the dashboard reads from
a cached metadata file rather than from the evidence files themselves, a stale
display could mislead a consumer into believing evidence is current when it is
not.

**Attack scenario**: The dashboard metadata file (`docs/AUDIT_DASHBOARD.md`)
is updated by a CI step that runs on a schedule. The schedule fails for one
week. The dashboard shows last week's date. The evidence files themselves are
also one week old, but the age gate (T1 mitigation) has not yet triggered
because it only runs on push events. A consumer reads the dashboard and
concludes evidence is current.

**Mitigations**:
- All evidence files contain embedded timestamps. The age gate in
  `ci/audit_gate.py` reads the timestamp from each evidence file directly,
  not from the dashboard metadata. A hard failure occurs if any evidence file
  is older than 7 days at gate evaluation time.
- The dashboard generation step reads evidence file timestamps, not a separate
  metadata cache. Dashboard timestamps are derived from the same source as the
  gate checks.
- A separate CI job (`caas-freshness-check`) runs on a 6-hour schedule
  independently of the push-triggered pipeline. It evaluates the age gate
  against the current HEAD evidence and posts a failure status if evidence has
  gone stale. This ensures the freshness check runs even when no commits are
  pushed.

**Residual risk**: A gap between the 6-hour freshness-check interval and the
7-day expiry threshold means evidence could be up to 6 hours stale before the
check catches it. In practice this is not material, but real-time freshness
monitoring would eliminate it.

---

## 3. Residual CAAS-Level Risks

These are risks that the current mitigation set cannot reduce further, given
the threat model's practical constraints. They are documented here so
downstream consumers can make informed decisions about how much trust to place
in the CAAS output.

### 3.1 Deliberately Malicious Maintainer

A maintainer with write access to the repository could commit a vacuous exploit
test that wires correctly, passes the quality scanner, and achieves a
non-trivial mutation kill rate against non-critical mutations, while failing
to detect the specific vulnerability it is supposed to cover. This requires
sustained intentional effort and would survive only if code review also failed
to catch it.

**CAAS cannot fully protect against this alone.** The mitigation is independent
review of the CAAS methodology, reproducible replay, and transparency of the
test source in the public repository.

### 3.2 Nation-State CI Infrastructure Compromise

Compromise of GitHub Actions infrastructure at the platform level — not a
specific workflow, but the runner provisioning, the artifact storage, or the
secrets management — would allow silent modification of any pipeline artifact
without leaving evidence in the git history. Reproducible builds (T7
mitigation) would detect binary differences, but the tampered artifacts could
include falsified reproducible build attestations.

**CAAS cannot protect against this.** The mitigation requires running the
pipeline on independently operated infrastructure and cross-checking results.

### 3.3 Malicious Test Vectors

Test vectors that appear to exercise a vulnerability class but are in fact
constructed so that the vulnerable and non-vulnerable implementations produce
the same output for those specific inputs. This would allow a flawed
implementation to pass differential testing against libsecp for all tested
inputs.

**CAAS reduces but cannot eliminate this risk.** Wycheproof vectors and
libsecp differential testing provide good coverage of known vector classes.
Novel vectors for undiscovered bugs are not coverable by definition.

---

## 4. How to Audit CAAS Itself

An auditor who wants to verify the integrity of the CAAS pipeline, not just
its output, should take the following steps:

1. **Run the pipeline from scratch.** Clone the repository on an independent
   machine. Run `cmake --preset ci-audit && cmake --build out/release-ci-audit &&
   ctest --test-dir out/ci-audit`. Compare results to the released bundle.
   Any divergence is a finding.

2. **Inspect the gate scripts.** `ci/audit_gate.py`,
   `ci/check_exploit_wiring.py`, `ci/check_source_graph_quality.py`,
   and `ci/verify_external_audit_bundle.py` are the primary controls.
   They are pure Python, readable, and have no external dependencies beyond
   the standard library.

3. **Rebuild the source graph.** Delete `tools/source_graph_kit/source_graph.db`
   and run `python3 tools/source_graph_kit/build_graph.py`. Re-run the coverage
   queries. Confirm the graph reflects the actual source.

4. **Check the freshness gate.** Manually call `audit_gate.check_freshness()`
   against the evidence files. Verify that the 7-day threshold is enforced and
   that the timestamps in the evidence files match the CI run timestamps in the
   GitHub Actions log.

5. **Verify the bundle hash.** Run `python3 ci/verify_external_audit_bundle.py
   <bundle_path>`. Confirm the exit code is 0 and all hashes match.

6. **Read the scope definition.** `docs/AUDIT_SCOPE.md` lists in-scope and
   out-of-scope components. Evaluate whether the scope is appropriate for the
   deployment context. Identify any components marked `NOT_IN_SCOPE` that the
   deploying organization considers critical.

CAAS is a tool. Its trustworthiness derives from the verifiability of its
components, not from the authority of the organization that runs it.
