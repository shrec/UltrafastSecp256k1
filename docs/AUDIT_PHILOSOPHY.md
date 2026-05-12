# Audit Philosophy — UltrafastSecp256k1

> **Core principle:** Every security claim must be backed by a concrete,
> executable test — and that test must pass on every commit.

---

## Our View

In UltrafastSecp256k1, auditing is treated as a **continuous practice**, not
a one-time event. The approach rests on a single premise:

> **What is not verified on every commit is not guaranteed.**

Every security claim — "constant-time", "NULL-safe", "reject invalid inputs",
"matches reference implementation" — is tied to a specific, executable test.
Code and evidence evolve together and are verified together.

This is the audit model: continuous, executable, and replayable. Full
transparency means that every claim, every test mapping, and every coverage
gap is already documented and machine-readable. Reviewers arrive at a fully
evidenced system, not a blank canvas. The claim→test→evidence chain is
already in place.

---

## Four Pillars

### I. Claim → Evidence Traceability

Every security claim must exist as three components:

```
claim             →  test                  →  CI gate
"CT on x86-64"   →  ct_verif + dudect     →  ct-verif.yml
"no bias"         →  126 CT tests          →  security-audit.yml
"reject twist"    →  exploit_ecdh_twist    →  security-audit.yml
```

`docs/AUDIT_TRACEABILITY.md` is the canonical table for this mapping.
A claim without a test is not a claim — it is an intention.

### II. Adversarial Regression Culture

Every research paper or CVE relevant to the secp256k1 ecosystem becomes a
PoC test. The library must survive that test on every commit.

**257 exploit PoCs test files (all 189 wired as runner modules)** — covering:

| Attack category | Examples |
|-----------------|----------|
| Nonce bias / lattice | ePrint 2023/841, 2024/296 (HNP lattice sieve) |
| DFA / fault injection | ePrint 2017/975 (RFC 6979 DFA) |
| Timing / CT leaks | ePrint 2024/1380 (EUCLEAK), 2024/589 (cmov SPA) |
| FROST / threshold | ePrint 2026/075 (weak binding), 2020/852 (rogue-key) |
| Batch verify bypass | ePrint 2026/663 (modified batch verify) |
| Schnorr hash order | ePrint 2025/1846 (Fiat-Shamir violation) |
| Type confusion | CVE-2024-49364, CVE-2022-41340 |
| Nonce reuse cascade | ePrint 2025/654 (cross-key) |
| ROS attack | ePrint 2020/945 (concurrent Schnorr) |
| ZVP-DCP | ePrint 2025/076 (GLV multiscalar) |

Principle: **document the attack, build the PoC, gate it in CI**.
The team's memory does not define the security posture — CI's memory does.
When a new CVE or ePrint arrives, it is met with a PoC first.
After the patch, that PoC remains as a permanent regression guard.

### III. Formal + Empirical CT Verification

Constant-time correctness is verified through three independent pipelines:

```
ct-verif.yml      — LLVM IR symbolic CT analysis   (compile-time)
valgrind-ct.yml   — Valgrind Memcheck taint prop.  (runtime)
ct-prover.yml     — dudect statistical timing test (empirical)
```

All three run on x86-64. `ct-arm64.yml` runs the full pipeline on arm64.

**45 Cryptol properties** across 4 `.cry` files verify algebraic correctness
of field arithmetic, ECDSA, and Schnorr via the Cryptol/SAW platform.

A CT claim is not "the code looks side-channel-free."
A CT claim is "ct-verif says OK, Valgrind says OK, dudect says OK" —
on every commit, across two architectures.

### IV. Layered Risk Stratification

Not every function warrants the same scrutiny. The system applies three tiers:

| Tier | Scope | Requirements |
|------|-------|-------------|
| **Tier 1** | Core ECC, ECDSA, Schnorr, ECDH, CT paths, C ABI | Full coverage, CT + fuzz + exploit PoC |
| **Tier 2** | BIP-32, BIP-340 advanced, FROST, MuSig2, ZK | High coverage, fuzz, regression |
| **Tier 3** | GPU batch ops, language bindings, tooling, benchmarks | ABI-level testing, backend audit |

The GPU layer has branchless CT primitive implementations across all three backends
(6 OpenCL CT kernel files, 6 CUDA CT headers, 6 Metal CT shaders) covering field,
scalar, point, and signing operations. These carry **code-discipline CT guarantees**
(no secret-dependent branches in source, branchless cmov/cswap), but not the same
formal pipeline guarantees as the CPU CT path (no ct-verif LLVM analysis, no Valgrind
taint tracking, no dudect timing measurement on GPU kernels). Vendor JIT compilers
can transform Metal/OpenCL/CUDA kernels at runtime; this caveat is documented and
somoke-tested but not formally verified. Production private-key signing always routes
through the CPU CT layer.

---

## Continuous Evidence Pipeline

```
code change
    │
    ▼
build + static analysis (clang-tidy, CodeQL)
    │
    ▼
unit + integration tests  ──────────────────►  coverage gate
    │
    ▼
unified_audit_runner
 ├── 51 non-exploit audit modules  (correctness, edge cases, ABI)
 └── 257 exploit PoCs modules       (CVE/ePrint adversarial cascade)
    │                                          ~1,000,000+ assertions / build
    ▼
CT verification  (ct-verif + Valgrind + dudect, x86-64 + arm64)
    │
    ▼
fuzz  (11 harnesses, libFuzzer, sanitizers)
    │
    ▼
1.3M nightly differential checks vs reference implementation
    │
    ▼
SLSA provenance + assurance export → AUDIT_TRACEABILITY.md
```

Every stage is a CI gate. A failing gate blocks merge.

---

## System Properties

### 1. Claim = Evidence, Not Intent

Every security claim has a corresponding CI workflow, test file, and evidence
artifact documented alongside it. `WHY_ULTRAFASTSECP256K1.md` exposes this
mapping in full. A claim that cannot be linked to a test is removed.

### 2. Attack-First Design

The starting question is never "does this code look correct?"
The starting question is "is this attack vector viable on this code?"
PoC first. Test first. Claim only after both pass.

### 3. Machine-Readable Audit Index

`tools/source_graph_kit/source_graph.py` maintains a live engineering index:

- **9,071** indexed functions across **406,505** LOC and **17** project dimensions
- **4,766** test→function mappings discovered via call-mention analysis
- **1,033** per-file audit coverage rows
- **8,638** symbol-level audit scores
- Backend parity tracking: **83 full**, **119 gpu-full**, **0 missing-metal**, **0 missing-opencl**
- Active review queue: **7 real audit gaps**, **6 untested hotspots**, **9 high-gain targets**

The graph allows machine-readable queries: "Which functions have CT coverage?",
"Which files have audit gaps?", "Which GPU operations are missing from which backend?"

### 4. Always Audit-Ready

`ci/external_audit_prep.sh` produces in a single command:
- preflight output (platform and toolchain verification)
- assurance export (machine-readable coverage and risk matrix)
- traceability artifacts (claim → test → evidence links)
- full audit package (source, build, coverage, SLSA provenance)

This infrastructure exists first for per-commit discipline.
The side effect: any party who wants to audit the system can start immediately
from this evidence base rather than from first principles.

---

## On the "Single-Maintainer Risk" Question

This project is sometimes characterised as a single-maintainer project,
with the associated concern that one person's absence could stall security work.
That characterisation is accurate for governance and community size,
but it misrepresents how security review actually works here.

**The project operates under continuous LLM-assisted audit.**

Every non-trivial change is developed in collaboration with large language
models (currently Claude Sonnet) acting as a persistent, structured security
reviewer. This is not autocomplete — it is a deliberate workflow:

- The LLM reviews each change for CT correctness, null-safety, ABI stability,
  and exploit surface before the code is committed.
- It cross-references every security claim against indexed evidence in the
  source graph and the audit traceability matrix.
- It flags regressions in exploit PoC coverage, missing parity across GPU
  backends, and stale documentation, all within the same session as the change.
- It has read-level access to 9,071 indexed functions and the full audit
  infrastructure, making its review as wide as the codebase, not just the diff.

The result is a review cycle that runs on every change, not periodically.
It is not a replacement for human cryptographic expertise, but it is a
qualitatively different security posture from a solo developer committing code
without review.

Practically, this means:

| Traditional solo project | This project |
|--------------------------|-------------|
| Changes reviewed by one person | Every change reviewed by maintainer + LLM audit session |
| Security gaps accumulate between audits | Security gaps flagged at commit time |
| Institutional memory lives in one head | Institutional memory lives in CI, source graph, and audit DB |
| Bus factor = 1 | CI, tests, and evidence infrastructure are fully reproducible by any successor |

The CI pipelines, audit runner, source graph, exploit PoC suite, and
traceability documents are all self-contained and documented.
A successor or independent reviewer can reproduce the full audit state from the
repository alone, without any knowledge transfer from the original maintainer.

That is the deeper purpose of "audit-as-code": it is also bus-factor mitigation.

---

## What This System Does Not Claim

Honesty is as important as coverage:

- **Physical attacks:** Power analysis, EM side-channels, cold boot — out of scope.
- **Compromised toolchain:** No guarantees if the build environment is untrusted.
- **OS-level memory disclosure:** Beyond this library's boundary.
- **Post-quantum:** secp256k1 is not quantum-resistant — stated explicitly.
- **GPU CT formal guarantees:** The GPU CT layer uses branchless primitives (code-discipline
  CT) tested by per-backend smoke tests. Vendor JIT may still introduce branches in compiled
  kernels. The 3-pipeline formal CT verification (ct-verif LLVM + Valgrind + dudect) applies
  to the CPU path only. Production signing is CPU-only.
- **External reviewer support:** The system is designed so outside reviewers can replay all evidence — `bash ci/external_audit_prep.sh` produces a reproducible auditor-facing bundle with preflight outputs, assurance export, and traceability artifacts.

---

## "Audit-as-Code" — a Culture, Not a Tool

The `audit/` directory with 189 PoC test files is **institutional memory**.

Behind each PoC is the question: "Has this attack vector been demonstrated
against the secp256k1 ecosystem?" If yes — that PoC lives in CI, on every
commit. After a patch, the PoC remains as a permanent regression guard.
The library does not just get patched against known attacks.
It carries proof of resistance on every build.

---

## Known Open Questions

Full transparency means known gaps are visible too:

| Question | Status |
|----------|--------|
| External reviewer support | Full reproducible evidence bundle available via `bash ci/external_audit_prep.sh` |
| ROCm/AMD hardware validation | Experimental |
| GPU CT formal guarantees | Code-discipline branchless CT + smoke tests on all 3 backends; 3-pipeline formal verification (ct-verif LLVM/Valgrind/dudect) is CPU-only — vendor JIT caveat applies to GPU |
| Differential fuzzing vs libsecp256k1 | Partial, not exhaustive |
| Formal verification (Fiat-Crypto level) | SafeGCD/Bernstein-Yang divstep verified with dual provers (Z3 SMT: 17 proofs + Lean 4: 19 theorems); Cryptol covers field arithmetic; broader coverage is future work |

Publishing these gaps openly means any external party can focus their effort
precisely where it adds the most value, without redundant discovery work.

---

## How This Simplifies Independent Review

CAAS is designed so independent reviewers can start from executable evidence
instead of reconstructing the project from scratch:

- **Claim inventory** — `docs/AUDIT_TRACEABILITY.md` maps every security claim
  to a test. An auditor starts from this list, not from scratch.
- **Machine-readable index** — the source graph covers 9,071 functions with
  coverage scores, audit gaps, and dependency chains queryable in seconds.
- **Evidence artifacts** — `ci/external_audit_prep.sh` produces a full
  audit package (source, tests, coverage, SLSA provenance) in one command.
- **Attack surface** — `docs/ATTACK_GUIDE.md` and the 189 PoC modules show
  exactly which attack vectors are already covered. An auditor verifies
  existing coverage and focuses on what is still open.
- **Reproducible builds** — SLSA provenance allows any party to independently
  verify build integrity.

This does not replace external scrutiny — it makes that scrutiny more targeted.

---

## Addressing Common Objections

### "Continuous CI — how is this better than a snapshot audit?"

Consider the track record: Heartbleed lived in OpenSSL for two years — a library
that had been reviewed by many expert eyes and was trusted everywhere. The
problem was not that too few people looked; it was that no system *continuously*
checked the specific property that failed. Our CT pipelines, fuzzer harnesses,
and 257 exploit PoCs modules run on every commit. A bug of the Heartbleed class
— a missing bounds check on a secret-bearing input path — is exactly what the
combination of fuzz-with-ASan and CT taint propagation is designed to catch
before it reaches `main`.

| Snapshot report | This model |
|----------------|------------|
| Certifies code at one moment | Verifies code on every commit |
| Finding post-patch bugs requires another engagement | CI flags regressions automatically |
| Report age grows while code evolves | Evidence age is always ≤ last commit |
| Technical guarantee: bounded in time | Technical guarantee: continuous |

### "29 stars and 13 forks indicate a micro-community — this is a red flag"

Community size correlates with **visibility**, not with security depth.

`bitcoin-core/libsecp256k1` has 2,000+ stars because it is embedded in Bitcoin
Core, which has millions of users and a decade of network effect. The star count
reflects the network, not the cryptographic quality of the code. `libsecp256k1`
would have the same security properties if it had 29 stars — the math does not
change with GitHub metrics.

For a narrow, high-security native library with a C ABI, low star counts are
structurally expected. The audience is not general developers — it is system
engineers with a specific, well-defined need. The most security-critical software
in existence — HSM firmware, embedded TLS stacks, formal provers, secure enclaves
— routinely has tiny public GitHub presences. Visibility is a marketing metric.
It does not belong in a security assessment.

What would a large community actually add here? More PRs creates a larger review
surface and a wider attack surface for supply-chain compromise. Correctness for
this class of library is not determined by how many people looked at it informally
— it is determined by whether the claims are verified by automated systems on
every change. That is what this project has.

The meaningful adoption signal is not star count but this: **Sparrow Wallet and
Frigate**, privacy-focused wallets making real Bitcoin transactions for real users,
integrated this library into a path that handles actual funds. That is a stronger
signal than 2,000 stars on a repository that most users pull in transitively
without knowing it.

### "npm download counts are low — production adoption is thin"

npm downloads are the wrong metric for a native cryptographic library entirely.

The core of this library is a C ABI (`libufsecp.so`). The npm package is a thin
N-API wrapper. Downloads of the npm package reflect Node.js developers
specifically, not C/C++/Rust/Python/Swift/Go/Dart consumers. Measuring adoption
of a multi-platform native library via a single language package manager's
download counter is like measuring a port's cargo volume by counting passenger
cars on the ferry.

The adoption metric that matters for a library at this stage is: **who uses it
for what, and has anything broken in production?** Sparrow and Frigate use it for
Silent Payments scanning and BIP-340 verification on real mainnet transactions.
Zero security incidents in that deployment. That is the meaningful number.

### "GPU is not safe for secrets — critical technical risk"

This is not a risk. It is the design.

The library's architecture is built around an explicit, permanent separation:

- **CPU layer** — all secret-bearing signing operations (ECDSA sign, Schnorr sign,
  key derivation). Constant-time, verified by ct-verif + Valgrind + dudect on
  every commit across x86-64 and arm64.
- **GPU layer** — primarily public-data pipelines (batch verify, address generation,
  point compression). A small number of GPU operations accept secret material for
  high-throughput workloads: `ecdh_batch`, `bip352_scan_batch`, and
  `bip324_aead_*_batch`. These secret-bearing GPU operations require a trusted
  single-tenant environment and are documented as such in the C ABI.

The GPU layer does have branchless CT primitive implementations (6 OpenCL kernel files,
6 CUDA headers, 6 Metal shaders) covering field, scalar, point, and signing operations.
These provide **code-discipline CT guarantees**: all secret-dependent operations use
branchless cmov/cswap with no data-dependent branches in source. However, vendor JIT
compilers (Metal, PTX assembler, OpenCL runtime) transform kernels at runtime and can
silently introduce branches — the formal 3-pipeline CT verification that the CPU path
has (ct-verif LLVM analysis, Valgrind taint, dudect timing) is not applicable to GPU.

The canonical engineering response is exactly what this library does: the GPU CT layer
is smoke-tested for correctness, but production private-key signing is kept on the CPU
CT layer where formal guarantees hold. Secret-bearing GPU operations (ECDH, BIP-352,
BIP-324) are restricted to trusted single-tenant environments. This layered approach is
not a gap. It is the correct architecture.

### "Cannot be used as a primary signer core"

The CPU layer of this library IS a signer core. It signs ECDSA and Schnorr
transactions with constant-time guarantees verified by three independent
pipelines. It implements RFC 6979 deterministic nonce generation with hedged
entropy support. It handles BIP-340 signing, ECDH, key derivation, and the full
secret-bearing path that a wallet or custody system requires.

What cannot happen is: running the signing path on the GPU. That is correct
behaviour, not a limitation. The correct system architecture is CPU for signing,
GPU for batch verification and chain scanning — which is how wallets like Frigate
actually use this library. The characterisation "cannot be used as primary signer
core" is factually incorrect for the CPU signing path.

The reproducible evidence bundle (`ci/external_audit_prep.sh`) and structured traceability artifacts (`docs/AUDIT_TRACEABILITY.md`, source graph) are designed so reviewers can engage with the implementation at any depth without waiting on any intermediary.

### "Backend parity is not fully proven"

The parity tracker reports at the current HEAD:

- **0 missing-metal** operations
- **0 missing-opencl** operations
- **83 full-parity** symbols (all backends)
- **119 gpu-full-parity** symbols

This is not a claim — it is a machine-generated output from the source graph
re-indexed on every build, cross-referencing the `GpuBackend` virtual interface
against CUDA, OpenCL, and Metal implementations. Any parity gap introduced by a
commit is flagged immediately in the CI parity audit workflow. Temporary
`GpuError::Unsupported` stubs are only permitted with a `TODO(parity):` tracking
comment and an entry in `docs/BACKEND_ASSURANCE_MATRIX.md`.

The claim "backend parity not fully proven" would need to identify a specific
operation that is implemented on one backend and missing on another without a
documented tracking comment. If such an instance exists, it is a concrete,
actionable bug report — not a structural concern about the architecture.

### "Consensus-level equivalence is not confirmed"

**1.3 million nightly differential checks** run against the reference
implementation (`bitcoin-core/secp256k1`) covering ECDSA sign and verify,
Schnorr sign and verify, point operations, and scalar arithmetic across
randomly-generated and edge-case inputs. Additionally, 45 Cryptol properties
verify the algebraic identities of field arithmetic and signature schemes against
formal mathematical specifications.

The differential suite is not sampling. It covers the full input domain
reachable by the random generator, biased toward boundary values, with special
cases for the curve's known edge inputs (identity, low-order points, cofactor
edges). A divergence from the reference implementation on any check fails the
nightly CI run.

"Not fully confirmed" would require specifying what additional confirmation is
expected beyond 1.3M cross-checked calls per night and formal algebraic
verification. If the concern is a specific input class not covered by the
differential suite, that is a concrete improvement target — not a generic
characterisation of the current evidence.

### "ICICLE is a comparable GPU competitor"

ICICLE (Ingonyama) is a GPU acceleration framework for **zero-knowledge proof
primitives**: MSM (multi-scalar multiplication over BLS/BN curves), NTT
(number-theoretic transforms), and polynomial commitment schemes. It is not a
secp256k1 library. It does not implement ECDSA, Schnorr, ECDH, BIP-340, BIP-352,
FROST, MuSig2, or any of the operations this library provides.

The overlapping surface is narrow: both use GPU for EC point operations. The use
cases, target curves, APIs, and security models are entirely different. ICICLE is
the right tool if you are building a ZK prover over BLS12-381. It is not a drop-in
alternative — or any kind of alternative — for GPU-accelerated secp256k1
batch verification or Silent Payments scanning.

### "A security incident would cause total trust loss"

This is a generic statement that applies equally to every cryptographic library
in existence, including libsecp256k1, OpenSSL, BouncyCastle, and Tink. It is not
a differentiating risk factor for this library specifically.

What differentiates the response to a potential security incident here is the
adversarial regression culture: 257 exploit PoCs modules covering known attack
classes run on every commit. If a vulnerability is discovered and patched, a PoC
is added to the suite and it runs permanently. The attack class can never silently
regress. Novel, previously unknown vulnerabilities carry the same risk profile for
this library as for any other — because by definition no existing test covers them.
The correct mitigation for novel vulnerabilities is fuzz coverage (11 harnesses,
continuous) and CT analysis (three pipelines, continuous). Both are in place.

The "total trust loss" scenario would require a vulnerability that: (a) is novel
and not covered by any known attack class, (b) survives 11 fuzz harnesses with
sanitizers, (c) is not caught by CT taint analysis on the secret-bearing path,
and (d) is not surfaced by 1.3M nightly differential checks. That is not an
impossible combination of conditions, but it is not a scenario unique to this
library, and the probability is materially lower here than for libraries without
this verification infrastructure.

---

## The Instrument Layer — Tools, Roles, and Continuous Growth

Every claim in the previous sections is backed not by intent but by specific
instrumentation. This section catalogs each tool: what class of bug it finds,
how the tool itself is sharpened when new attack patterns emerge, and how the
feedback loop ensures quality grows rather than stagnates.

### Instrument Taxonomy

The verification stack is stratified by what it can and cannot observe:

| Layer | Instrument | Bug Class Targeted | How It Improves When New Pattern Emerges |
|-------|-----------|-------------------|------------------------------------------|
| Compile-time | `ct-verif` (LLVM IR symbolic analysis) | CT violations where a secret-dependent branch or memory address is visible in IR | New LLVM taint query or IR annotation added targeting the new pattern; zero false negative rate for tested functions |
| Runtime secret taint | Valgrind Memcheck (custom `VALGRIND_MAKE_MEM_UNDEFINED` markers) | Any memory read or branch whose address or condition is reachable from secret bytes at runtime | New `classify/declassify` marker pair added for every new secret-bearing function entry and exit |
| Statistical timing | `dudect` (Welch t-test on timing distributions) | Empirical timing variations that are statistically significant across repeated executions | A new dudect harness is added whenever a new operation joins the signing or key-agreement code path; threshold tightened if a near-miss is observed |
| Memory safety | libFuzzer + ASan + UBSan + MSan (11 harnesses) | OOB reads/writes, use-after-free, undefined behaviour, uninitialized reads | A new harness is created for every new parser, encoding path, or API entry point; corpus is seeded with any input that previously triggered a regression |
| Adversarial regression | 257 exploit PoCs modules (`audit/` directory) | Known CVE-class, ePrint-class, and published attack patterns | Within days of a new ePrint or CVE being published, a PoC is added; the module runs on every commit permanently |
| Logic correctness | 51 non-exploit audit modules | Edge-case correctness, ABI contract violations, cross-platform numeric consistency | Every bug discovered in any review session becomes a new audit module, not just a fixed line |
| Formal algebraic | 45 Cryptol properties | Group law violations, modular arithmetic invariants, identity/cofactor edge cases | When a new algebraic identity is identified during research, a Cryptol property is written before the implementation is merged |
| Semantic equivalence | 1.3M nightly differential checks vs reference implementation | Any deviation from the canonical reference implementation on any input class | When a new operation is added, a differential test is required before the operation is exposed in the ABI |
| Data-flow | CodeQL + clang-tidy | Complex inter-procedural data flows, AST-level coding pattern violations, potential injection sites | CodeQL queries and clang-tidy checks are extended when a new pattern class is identified in a security review |
| Build integrity | SLSA Level 3 provenance | Build environment tampering, artifact substitution, dependency injection | Provenance attestations are generated for every release artifact; the supply chain model is re-evaluated on every toolchain update |
| Coverage gap detection | `source_graph.py` (Code Grapher) | Audit blind spots, untested hotspots, functions missing CT tags, GPU parity gaps | The graph schema and coverage queries evolve when new audit dimensions are added; rebuilding the graph after any coverage expansion immediately reveals newly visible gaps |

None of these instruments is optional. Each covers a failure mode the others
cannot reach. A CT violation in LLVM IR may not exhibit a measurable timing
difference on a specific microarchitecture. A statistical timing anomaly may not
correspond to any identifiable IR branch. A fuzzer crash is invisible to both.
The redundancy is engineered.

### How a New Attack Pattern Enters the Stack

The lifecycle is deterministic:

1. **Signal acquisition.** A new ePrint paper, CVE disclosure, bug bounty report,
   or LLM-assisted adversarial review identifies a potential attack class.

2. **Applicability triage.** The source graph is queried: which functions, call
   paths, or data flows are in scope? The AI Memory is queried: has this class been
   evaluated before? Was a previous PoC attempted and ruled out?

3. **PoC construction.** A PoC module is written against the current code. If the
   library is vulnerable: the PoC fails the test suite, the root cause is isolated,
   a fix is made, and the PoC is kept permanently as a regression test. If the
   library is not vulnerable: the PoC is kept as evidence of non-vulnerability —
   equally valuable for a due diligence reviewer.

4. **Instrument update.** If the attack class reveals a gap in an existing
   instrument (e.g., a timing channel dudect does not distinguish, or a fuzzer
   corpus with no coverage of the affected input), the instrument itself is updated.
   The gap is never papered over with documentation.

5. **Source graph rebuild.** After any instrumentation update, `source_graph.py
   build -i` is run. The new coverage is immediately queryable. The audit traceability
   matrix is updated. The CHANGELOG records the addition.

This loop runs daily. It is not triggered by incidents — it runs because new
cryptographic research is published every week and the integration lag must be
measured in days, not release cycles.

### The Daily Research Loop

Every day the following happens, in order:

- **Signal scan.** New ePrint preprints, CVEs in the NVD feed, and bug bounty
  disclosures relevant to elliptic curve cryptography are reviewed. The primary
  filters are: secp256k1 operations, ECDSA/Schnorr signing, nonce handling,
  batch verification, and side-channel attacks on field arithmetic.

- **Applicability review.** Each signal is matched against the source graph:
  `python3 source_graph.py focus <term> 24 --core`. If the graph has no coverage
  of the relevant symbol family, that itself is flagged as a gap.

- **Resolution.** One of three outcomes: (a) PoC added + non-vulnerable documented;
  (b) bug found + patched + PoC added; (c) attack class inapplicable — documented
  with reasoning in `docs/AUDIT_CHANGELOG.md`.

- **Memory commit.** The finding is stored in AI Memory with tags
  (`ePrint`, `CVE`, `bug`, `fix`, `dead-end`) so future sessions start with this
  context already loaded, not from a blank slate.

The daily cadence means the gap between a published attack and its evaluation
against this codebase is bounded by one working day, not by a release schedule.

---

## LLM Multi-Role Audit Scanning

LLMs are deployed as a structured, adversarial review layer — not as a chatbot.
The same model is invoked across multiple roles in a single session, each with a
different adversarial posture. Each role asks fundamentally different questions.

### Role 1 — The Systematic Auditor

Posture: methodical, exhaustive, documentation-first.

Questions asked:
- Does every public API function have a corresponding audit module?
- Does every security claim in `WHY_ULTRAFASTSECP256K1.md` link to a specific test artifact?
- Are there CT annotations on every function in the secret-bearing call path?
- Are there any functions in the source graph with an audit score below the risk tier threshold?

The auditor role is constraint-driven: it operates against the source graph,
the audit traceability matrix, and the existing coverage maps. It surfaces
gaps systematically rather than discovering novel attacks.

### Role 2 — The Bug Bounty Hunter

Posture: high-impact focused, reward-maximizing.

Questions asked:
- Which functions handle nonce generation or nonce reuse? What happens if an
  attacker can influence the RNG state?
- Which API entry points accept untrusted byte buffers? Is every length boundary
  validated before the first memory read?
- Which edge cases in the batch verifier could produce a false-positive result
  (batch verification accepts an invalid signature)?
- Is there any path where a public key or message hash is used as a secret
  in a timing-sensitive branch?

The bug bounty hunter skips low-severity findings and focuses on critical-path
vulnerabilities in signing, key derivation, and batch verification — the surfaces
with the highest real-world exploit value.

### Role 3 — The Attacker / Red Teamer

Posture: adversarial, assumption-breaking, worst-case.

Questions asked:
- If I control the first 16 bytes of the nonce input, can I bias the nonce
  distribution to recover the private key in fewer than 2^32 signing operations?
- If I can cause a fault during scalar multiplication, does the library detect it?
- If I can observe which branch of a conditional is taken via a timing oracle,
  what is the maximum information I gain per signature?
- If I build the library with a malicious compiler that optimizes away memset
  calls, does any secret survive in stack memory?

The attacker role explicitly assumes the cryptographic assumptions hold but the
implementation has a flaw. It asks "how would I exploit this?" before asking
"is this exploitable?" — because the answer to the second question is always
"maybe," and only evidence from the first question settles it.

### Role 4 — The Documentation Reviewer

Posture: accuracy-first, adversarial about claims.

Questions asked:
- Does `WHY_ULTRAFASTSECP256K1.md` claim anything the tests cannot demonstrate?
- Does `AUDIT_PHILOSOPHY.md` describe infrastructure that does not exist in CI?
- Are the benchmark numbers reproducible from the current build configuration,
  or are they from a configuration that no longer compiles?
- Does `BACKEND_ASSURANCE_MATRIX.md` reflect the current stub status of every
  GPU operation, or are resolved stubs still listed as pending?

The documentation reviewer treats documentation inaccuracies as security
vulnerabilities: a reviewer who reads an inaccurate assurance document and
makes a deployment decision based on it is as harmed as if the code itself
were wrong.

### Role 5 — The Parity Auditor

Posture: cross-backend completeness enforcer.

Questions asked:
- Every CUDA kernel: does an equivalent OpenCL kernel exist? Does a Metal shader exist?
- Every CPU C ABI function: is it accessible via the GPU path? Is the GPU result
  cross-checked against the CPU result in the differential suite?
- Every GPU backend: does it have CT smoke tests? Does it have a coverage entry
  in the assurance matrix?

GPU backend parity is a surface-area problem: an attacker who can push a
deployment toward an untested backend has effectively bypassed the verification
infrastructure for that backend. The parity auditor role closes this surface.

### Role 6 — The Regression Hunter

Posture: pattern-aware, history-sensitive.

Questions asked:
- This call looks similar to `ecdsa_sign_nonce_rfc6979` — was there a nonce
  reuse bug in that function class before? Is the fix present here?
- This field element conversion path was audited in commit `763bc2e5` — has
  anything in the call chain changed since then?
- The AI Memory records a previous "near-miss" on Valgrind taint analysis for
  this function family. Is the new change in scope for that concern?

The regression hunter operates specifically against AI Memory and the source
graph's change history. It is the role most dependent on persistent cross-session
context, because regressions are only visible when you know what was previously
fixed.

---

## AI Memory and Code Grapher — The Compound Intelligence Layer

Individual tools produce individual results. AI Memory and the Code Grapher
combine all results into a persistent, queryable, compound intelligence layer
that grows in value over time.

### What AI Memory Does

AI Memory (`tools/ai_memory/ai_memory.py`) is a persistent SQLite-backed note
store that survives across all sessions indefinitely. Every session reads from
and writes to the same database.

What is stored:
- **Architectural decisions** with rationale: why RFC 6979 nonce derivation was
  chosen over a simpler CSPRNG approach; why the CPU signing path is separated
  from the GPU path even when the GPU is present
- **Security reasoning**: why a specific Valgrind finding was classified as a
  false positive, with the reasoning that would allow a future reviewer to
  challenge or confirm that classification
- **Discovered bugs and their root causes**: not just "fixed" but "was caused by
  X, fixed by Y, PoC at Z"
- **Dead ends**: approaches tried and abandoned, with the reason, so future
  sessions do not repeat the same wrong path
- **Research signals**: ePrint papers evaluated, their applicability status, and
  the date of evaluation

What this prevents:
- Re-evaluating the same ePrint paper in a new session as if it were unknown
- Misclassifying a known false-positive Valgrind finding as a new bug
- Forgetting that a specific optimization was tried and reverted because it
  broken constant-time behavior
- Starting from first principles on a problem that was already fully understood
  in a previous session

Without AI Memory, each session is effectively a new hire who starts from
scratch. With AI Memory, each session is a continuation of an ongoing, accumulating
expertise that cannot be lost between restarts.

### What the Code Grapher Does

`source_graph.py` maintains a live machine-readable engineering index of the
entire repository. It is rebuilt incrementally on every meaningful change.

What it knows:
- **9,071 indexed functions** and their exact file and line ranges
- **4,766 test→function mappings** discovered by call-mention analysis
- **1,033 per-file audit coverage scores** across 17 audit dimensions
- **8,638 symbol-level audit scores** for precise gap identification
- **Backend parity status** for every GPU operation (CUDA, OpenCL, Metal)
- **Call-graph edges**: which functions call which, making impact analysis instant
- **Hotspot and bottleneck data**: which functions are at highest risk and highest
  coverage cost
- **Active review queue**: 7 real audit gaps, 6 untested hotspots, 9 high-gain targets

What this enables:
- Asking "which functions are CT-annotated but have no dudect harness?" and
  getting an immediate, exhaustive answer — not a guess
- Asking "what is the blast radius of changing this field-arithmetic function?"
  and getting a call-graph answer, not an approximation
- Asking "does this new function family have an audit score above the risk tier
  threshold?" as part of every code review

### The Compound Effect

Individually, AI Memory provides historical context and the Code Grapher provides
current structure. Together, they enable a qualitatively different kind of review:

**Scenario:** A new paper on lattice attacks against biased ECDSA nonces is published.

Without AI Memory + Code Grapher:
- LLM searches the codebase manually, probably misses some nonce-adjacent call paths
- Cannot know whether this exact attack class was evaluated before
- Cannot know whether a previous fix for a related class introduced a regression in the nonce path
- Conclusion: partial coverage, uncertain confidence

With AI Memory + Code Grapher:
- AI Memory is queried: "context nonce bias lattice attack" — reveals that a previous
  evaluation in March found path X safe, path Y marginally in-scope
- Code Grapher is queried: `focus nonce_derive 24 --core` — identifies the 7 functions
  in the nonce derivation chain and their current audit scores
- Cross-reference: the 7 functions are checked against the prior evaluation; 6 match
  the previous "safe" classification; 1 is new since the previous evaluation
- Action: PoC written targeting only the 1 new function; existing coverage confirmed
  sufficient for the other 6; AI Memory updated with the new evaluation date and result
- Conclusion: precise, evidence-backed, cost-efficient

The compound intelligence layer converts what would be a week-long manual audit
into a session-length targeted review, without sacrificing coverage.

### Memory Hygiene

AI Memory is periodically garbage-collected (`gc --days 7`) to remove stale
session-scoped entries while preserving persistent architectural knowledge.
The store is gitignored: it is local working state, not source, and not shared
across installations. A new installation starts with an empty AI Memory but
can be seeded from the source graph, the traceability matrix, and
`docs/AUDIT_CHANGELOG.md`, all of which capture the auditable record.

---

## Summary of Core Strengths

For reviewers who want a consolidated view of what makes this audit system
structurally different from a typical solo project:

| Strength | Mechanism | Evidence |
|----------|-----------|---------|
| Attack-first culture | 257 exploit PoCs modules, CVE/ePrint cascade | `audit/exploits/` directory, `EXPLOIT_TEST_CATALOG.md` |
| Formal CT verification | ct-verif LLVM + Valgrind taint + dudect (3 pipelines) | `ct/`, `ci/`, CI workflow |
| Continuous semantic equivalence | 1.3M nightly differential checks vs reference | Nightly CI, `AUDIT_TRACEABILITY.md` |
| Formal algebraic verification | 45 Cryptol properties covering group law + edge cases | `formal/`, Cryptol CI job |
| Fuzz coverage | 11 libFuzzer harnesses with sanitizers, continuous | `fuzz/`, CI fuzzer job |
| GPU backend completeness | Mandatory parity across CUDA, OpenCL, Metal | `BACKEND_ASSURANCE_MATRIX.md` |
| Machine-readable audit index | Source graph: 8,638 symbol scores, 4,766 test mappings | `tools/source_graph_kit/` |
| Persistent institutional memory | AI Memory: cross-session context, decision rationale | `tools/ai_memory/` |
| Daily attack-signal integration | ePrint/CVE scan → PoC → instrument update cycle | `docs/AUDIT_CHANGELOG.md` |
| LLM multi-role adversarial review | 6 distinct adversarial roles per major change | This document, per-session workflow |
| Bus-factor mitigation | Full audit state reproducible from repo alone | `ci/external_audit_prep.sh` |
| Supply chain integrity | SLSA Level 3 provenance on every release | CI provenance attestations |
| Documentation truth enforcement | Docs treated as security artifacts; inaccuracy = vulnerability | Documentation Reviewer LLM role |

No single strength above is unique in isolation. The structural advantage is
that all of them are present simultaneously, integrated into a single pipeline,
and enforced on every commit — not episodically, not at release time, and not
as aspirational goals.

---

## Related Documents

| Document | Contents |
|----------|----------|
| [AUDIT_TRACEABILITY.md](AUDIT_TRACEABILITY.md) | Claim → test → CI gate mapping |
| [AUDIT_GUIDE.md](AUDIT_GUIDE.md) | How to run the audit suite |
| [AUDITOR_QUICKSTART.md](AUDITOR_QUICKSTART.md) | 3-command start for external reviewers |
| [AUDIT_SCOPE.md](AUDIT_SCOPE.md) | External engagement scope definition |
| [BACKEND_ASSURANCE_MATRIX.md](BACKEND_ASSURANCE_MATRIX.md) | Per-backend assurance levels |
| [ATTACK_GUIDE.md](ATTACK_GUIDE.md) | Attack vectors and PoC references |
| [AUDIT_REPORT.md](AUDIT_REPORT.md) | Historical baseline (641,194 checks) |
| [WHY_ULTRAFASTSECP256K1.md](WHY_ULTRAFASTSECP256K1.md) | Engineering discipline and audit narrative |

---

*UltrafastSecp256k1 — Continuous evidence. Full transparency. Audit-ready by design.*
