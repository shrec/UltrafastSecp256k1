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

This does not make formal external audit redundant — quite the opposite.
Full transparency means that every claim, every test mapping, and every
coverage gap is already documented and machine-readable. If anyone ever
wants or needs a formal engagement, they arrive at a fully evidenced system,
not a blank canvas. The claim→test→evidence chain is already in place.
That substantially reduces the cost and duration of any external review.

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

**171 exploit-PoC registered modules, 187 test files** — covering:

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
 └── 171 exploit-PoC modules       (CVE/ePrint adversarial cascade)
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

`scripts/external_audit_prep.sh` produces in a single command:
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
It is not a replacement for human cryptographic expertise or for formal
third-party engagement — but it is a qualitatively different security posture
from a solo developer committing code without review.

Practically, this means:

| Traditional solo project | This project |
|--------------------------|-------------|
| Changes reviewed by one person | Every change reviewed by maintainer + LLM audit session |
| Security gaps accumulate between audits | Security gaps flagged at commit time |
| Institutional memory lives in one head | Institutional memory lives in CI, source graph, and audit DB |
| Bus factor = 1 | CI, tests, and evidence infrastructure are fully reproducible by any successor |

The CI pipelines, audit runner, source graph, exploit PoC suite, and
traceability documents are all self-contained and documented.
A successor or external auditor can reproduce the full audit state from the
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
- **Formal third-party audit:** Not yet conducted. The system is designed to make one as
  efficient as possible when it happens.

---

## "Audit-as-Code" — a Culture, Not a Tool

The `audit/` directory with 187 PoC test files is **institutional memory**.

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
| Formal third-party cryptographic audit | Not yet conducted — this system is designed to enable one efficiently |
| ROCm/AMD hardware validation | Experimental |
| GPU CT formal guarantees | Code-discipline branchless CT + smoke tests on all 3 backends; 3-pipeline formal verification (ct-verif LLVM/Valgrind/dudect) is CPU-only — vendor JIT caveat applies to GPU |
| Differential fuzzing vs libsecp256k1 | Partial, not exhaustive |
| Formal verification (Fiat-Crypto level) | Cryptol covers field arithmetic; broader coverage is future work |

Publishing these gaps openly means any external party can focus their effort
precisely where it adds the most value, without redundant discovery work.

---

## How This Simplifies External Engagement

If anyone ever wants or needs a formal third-party audit of this library,
the system substantially reduces the cost and time of that engagement:

- **Claim inventory** — `docs/AUDIT_TRACEABILITY.md` maps every security claim
  to a test. An auditor starts from this list, not from scratch.
- **Machine-readable index** — the source graph covers 9,071 functions with
  coverage scores, audit gaps, and dependency chains queryable in seconds.
- **Evidence artifacts** — `scripts/external_audit_prep.sh` produces a full
  audit package (source, tests, coverage, SLSA provenance) in one command.
- **Attack surface** — `docs/ATTACK_GUIDE.md` and the 187 PoC modules show
  exactly which attack vectors are already covered. An auditor verifies
  existing coverage and focuses on what is still open.
- **Reproducible builds** — SLSA provenance allows any party to independently
  verify build integrity.

This does not replace external scrutiny — it makes that scrutiny more targeted.

---

## Addressing Common Objections

### "There is no third-party audit — so this isn't production-ready"

This conflates two different things: **institutional trust** and **technical
security**. They are related but not the same.

A third-party audit produces a report certifying the code at a single point in
time. If one commit later introduces a timing side-channel or a null-dereference
on malformed input, the audit report still says "clean." The report does not
update. The CI does.

Consider the track record: Heartbleed lived in OpenSSL for two years — a library
that had been reviewed by many expert eyes and was trusted everywhere. The
problem was not that too few people looked; it was that no system *continuously*
checked the specific property that failed. Our CT pipelines, fuzzer harnesses,
and 187 exploit-PoC modules run on every commit. A bug of the Heartbleed class
— a missing bounds check on a secret-bearing input path — is exactly what the
combination of fuzz-with-ASan and CT taint propagation is designed to catch
before it reaches `main`.

The honest summary:

| Snapshot audit | This model |
|----------------|------------|
| Certifies code at one moment | Verifies code on every commit |
| Finding post-audit bugs requires another engagement | CI flags regressions automatically |
| Audit age grows while code evolves | Evidence age is always ≤ last commit |
| Institutional trust: high | Institutional trust: not yet established |
| Technical guarantee: bounded in time | Technical guarantee: continuous |

The institutional trust gap is real and acknowledged. The technical security gap
is not what a third-party audit would primarily close — it would close the
*perception* gap, which matters for enterprise procurement but is distinct from
the actual security posture of the code running in production.

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

- **CPU layer** — all secret-bearing operations (ECDSA sign, Schnorr sign, ECDH,
  key derivation). Constant-time, verified by ct-verif + Valgrind + dudect on
  every commit across x86-64 and arm64. Secrets never leave this layer.
- **GPU layer** — public-data pipelines only (batch verify, BIP-352 scan,
  address generation, point compression). No secret ever enters GPU memory.

The GPU layer does have branchless CT primitive implementations (6 OpenCL kernel files,
6 CUDA headers, 6 Metal shaders) covering field, scalar, point, and signing operations.
These provide **code-discipline CT guarantees**: all secret-dependent operations use
branchless cmov/cswap with no data-dependent branches in source. However, vendor JIT
compilers (Metal, PTX assembler, OpenCL runtime) transform kernels at runtime and can
silently introduce branches — the formal 3-pipeline CT verification that the CPU path
has (ct-verif LLVM analysis, Valgrind taint, dudect timing) is not applicable to GPU.

The canonical engineering response is exactly what this library does: the GPU CT layer
is smoke-tested for correctness, but production private-key signing is kept on the CPU
CT layer where formal guarantees hold. ECDH, BIP-352, and BIP-324 delegate to GPU only
under trusted single-tenant environments. This layered approach is not a gap. It is the
correct architecture.

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

The genuine constraint is the absence of a third-party audit report, which may
affect procurement decisions in regulated environments. That is accurately framed
as an institutional trust gap, not a technical capability gap.

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
adversarial regression culture: 187 exploit-PoC modules covering known attack
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

## Related Documents

| Document | Contents |
|----------|----------|
| [AUDIT_TRACEABILITY.md](AUDIT_TRACEABILITY.md) | Claim → test → CI gate mapping |
| [AUDIT_GUIDE.md](AUDIT_GUIDE.md) | How to run the audit suite |
| [AUDITOR_QUICKSTART.md](AUDITOR_QUICKSTART.md) | 3-command start for external reviewers |
| [AUDIT_SCOPE.md](AUDIT_SCOPE.md) | External engagement scope definition |
| [BACKEND_ASSURANCE_MATRIX.md](BACKEND_ASSURANCE_MATRIX.md) | Per-backend assurance levels |
| [ATTACK_GUIDE.md](ATTACK_GUIDE.md) | Attack vectors and PoC references |
| [AUDIT_REPORT.md](../AUDIT_REPORT.md) | Historical baseline (641,194 checks) |
| [WHY_ULTRAFASTSECP256K1.md](../WHY_ULTRAFASTSECP256K1.md) | Engineering discipline and audit narrative |

---

*UltrafastSecp256k1 — Continuous evidence. Full transparency. Audit-ready by design.*
