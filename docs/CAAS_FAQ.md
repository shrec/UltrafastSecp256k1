# CAAS FAQ — Common Reviewer Objections

> **Honest answers, no defensiveness.**
> These are the questions a skeptical reviewer should ask.
> If the answers are not satisfying, the evidence is there to check.

---

## "Why CAAS instead of a snapshot PDF?"

**Because CAAS makes the verifiable parts of review continuous, replayable, and
regression-protected.**

A traditional audit produces a PDF snapshot taken at a specific commit, by a
specific firm, at a specific date. That snapshot has three properties:

1. It describes the code *as of that commit*, not the current one.
2. It reflects the findings of reviewers who may not have run the code.
3. It cannot be re-executed by a third party to verify currency.

CAAS replaces the *verifiable* parts of an audit with something stronger:

| Traditional audit | CAAS |
|---|---|
| Snapshot (one commit) | Continuous (every commit) |
| Trust the firm's name | Verify reproducible evidence |
| Cannot be re-run | Fully reproducible in one command |
| $40k–$250k | $0 marginal cost |
| Claim → reviewer's opinion | Claim → executable test → CI gate |

What a traditional firm would do:
- Check for known attack patterns → **262 exploit PoCs tests do this continuously**
- Verify constant-time properties → **three CT pipelines do this on every commit**
- Review parser boundary behavior → **check_libsecp_shim_parity.py does this continuously**
- Check for known CVEs in the category → **RESEARCH_SIGNAL_MATRIX.json tracks 60+ ePrint/CVE references**

What CAAS does not replace:
- The reputational weight of a brand-name firm (cannot be automated)
- Human reviewer creativity for novel attack classes not yet in the PoC suite
- Formal proof of algorithmic correctness (this uses symbolic + empirical)

The evidence chain is: **claim → test → CI gate → evidence bundle → hash**. 
You can inspect, re-run, or challenge any link in that chain. With a PDF,
you cannot.

---

## "Solo maintainer. What happens if the maintainer disappears?"

**The project is designed for this.**

CAAS reproducibility means a new maintainer can:
1. Clone the repository.
2. Run `python3 ci/caas_runner.py` and get the same results.
3. Inspect every claim via the source graph and exploit catalog.
4. Run `python3 ci/audit_viewer.py` and browse the full evidence tree.

Nothing is in the maintainer's head that is not also in a test, a doc, or
a gate. The CAAS pipeline is the bus-factor mitigation — it is the
institutional memory.

Additionally, the library is pure C++ / C with no runtime dependencies beyond
a C++20 compiler. There is no service, no license server, no cloud API.

---

## "Low GitHub stars. Nobody uses this."

**Star count is a social signal, not a correctness signal.**

Mathematical correctness does not scale with adoption. Either `y² = x³ + 7`
or it does not. Either the Schnorr nonce generation matches BIP-340 or it
does not. Either the CT verification passes or it does not.

That said:
- The differential test suite compares every output against libsecp256k1's
  output on the same input. If outputs differ, the gate fails.
- The Wycheproof test suite runs against this library.
- BIP-340 test vectors are embedded in CI.

Stars measure popularity. The gates measure correctness.

---

## "GPU CT caveat — you can't prove GPU is constant-time."

**Correct. GPU is public-data-only. No secrets ever enter GPU memory.**

The GPU backend accelerates:
- Batch signature verification (public keys + signatures = public data)
- BIP-352 batch scanning (output scanning = public data)
- Point multiplication with public scalars only

The CPU CT layer handles:
- ECDSA signing (secret key + nonce)
- Schnorr signing (secret key + nonce)
- Key derivation (secret key operations)

The architecture enforces this separation at the type level. There is no
signing code path that routes through the GPU. GPU CT verification is
therefore not required — and not claimed. See [`docs/AUDIT_SCOPE.md`](AUDIT_SCOPE.md).

---

## "Why not just use libsecp256k1?"

**The library is positioned as an alternative backend, not a replacement demand.**

Bitcoin Core already uses libsecp256k1 and it is excellent. The proposal is:

> "Allow a compile-time alternative backend with identical external behavior,
> the same test suite passing, and additional performance on modern hardware."

The gains on x86-64 AVX2 and ARM64 Neon justify the additional backend for
use cases that need throughput. Core developers decide whether that tradeoff
is worth the maintenance surface. The library's job is to make the parity
case as strong as possible — which is what CAAS measures.

---

## "What does CAAS actually prove?"

CAAS proves the following, verifiably:

1. **262 exploit PoCs tests pass** on every commit — covering nonce bias,
   fault injection, CT leaks, type confusion, batch verify bypass, and 10+
   other attack classes.
2. **Constant-time signing** on x86-64 and arm64 — verified by ct-verif
   (LLVM IR), Valgrind taint propagation, and dudect statistical testing.
3. **Parser strict parity** — the libsecp shim rejects all inputs that
   libsecp rejects (checked by `check_libsecp_shim_parity.py`).
4. **Deterministic ECDSA nonces** via RFC 6979, verified by
   `rfc6979_spec_verifier.py` and differential comparison with libsecp.
5. **BIP-340 Schnorr conformance** — verified against BIP test vectors and
   differential-tested against libsecp's Schnorr implementation.
6. **Zero ABI regressions** across the test matrix — enforced by `test_abi_gate`.
7. **Library self-test on every context creation** — arithmetic invariants
   checked at runtime.
8. **Thread safety** — described in [`docs/THREAD_SAFETY.md`](THREAD_SAFETY.md);
   thread-local blinding enforced by design.

---

## "What does CAAS NOT prove?"

CAAS is explicit about non-claims:

1. **Physical side channels** — power analysis, EM, fault injection. See [RR-006](RESIDUAL_RISK_REGISTER.md).
2. **Post-quantum security** — secp256k1 is a classical curve. See [RR-007](RESIDUAL_RISK_REGISTER.md).
3. **Compiler correctness** — we trust the toolchain (GCC, Clang, MSVC).
4. **Novel attack classes not yet in the PoC suite** — by definition, unknown unknowns are not covered.
5. **Application-layer replay protection** — caller's responsibility. See [RR-008](RESIDUAL_RISK_REGISTER.md).
6. **GPU constant-time** — GPU handles public data only; see above.
7. **Formal proof of correctness** — CAAS uses symbolic + empirical verification,
   not Coq or Lean formal proof.
8. **Auditability of CAAS itself** — CAAS has its own threat model; see
   [`docs/CAAS_THREAT_MODEL.md`](CAAS_THREAT_MODEL.md).

The full list is in [`docs/RESIDUAL_RISK_REGISTER.md`](RESIDUAL_RISK_REGISTER.md)
and [`docs/NEGATIVE_RESULTS_LEDGER.md`](NEGATIVE_RESULTS_LEDGER.md).

---

## "How do I reproduce the evidence independently?"

```bash
# Clone
git clone https://github.com/shrec/UltrafastSecp256k1.git
cd UltrafastSecp256k1

# Build
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build

# Run full CAAS pipeline
python3 ci/caas_runner.py --json -o caas_report.json

# Open interactive dashboard
python3 ci/audit_viewer.py
```

The pipeline runs the same gates as CI. The JSON report includes:
- git commit hash
- dirty state flag
- per-stage pass/fail with timing
- evidence artifact hashes

If you want to verify the published bundle matches what you can generate:

```bash
python3 ci/external_audit_bundle.py        # regenerate
python3 ci/verify_external_audit_bundle.py # verify
# Compare docs/EXTERNAL_AUDIT_BUNDLE.sha256 with the published value
```

---

## "I found a potential bug. What now?"

Open a GitHub Issue or responsible-disclose via the contact in
[`SECURITY.md`](../SECURITY.md). The CAAS principle is: every real bug
becomes a permanent PoC test after the fix. That test will run on every
subsequent commit.

See [`docs/INCIDENT_RESPONSE.md`](INCIDENT_RESPONSE.md) for the full process.

---

## "Is CAAS itself trustworthy?"

CAAS has its own threat model. Known threats against CAAS are documented in
[`docs/CAAS_THREAT_MODEL.md`](CAAS_THREAT_MODEL.md), including:
- Stale evidence (evidence age gate)
- Vacuous tests (audit_test_quality_scanner)
- Un-wired tests (check_exploit_wiring.py)
- CI bypass (branch protection + ruleset)
- Malicious bundle (hash verification)

CAAS gates itself. A CAAS pipeline with stale evidence, a broken source
graph, or un-wired tests will fail its own gate before claiming success.
