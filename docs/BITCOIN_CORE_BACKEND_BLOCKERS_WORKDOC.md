# Bitcoin Core Alternative Backend Blockers Work Document

> Scope: readiness assessment for proposing UltrafastSecp256k1 as an optional,
> non-default, build-time alternative backend to `bitcoin-core/secp256k1`.
>
> Position: CAAS is not a weakness or substitute apology. CAAS is the trust
> model: continuous, executable, reproducible audit evidence that travels with
> the code.

## Executive Verdict

UltrafastSecp256k1 is close enough for a serious draft/RFC-style discussion, but
not yet ready as a merge-target Bitcoin Core backend PR.

Readiness split:

- CAAS / evidence readiness: high
- Engine technical maturity: high for its own API and C ABI
- Bitcoin Core backend readiness: blocked by shim semantic parity and build
  integration details
- Merge readiness: not yet
- Draft/RFC readiness: yes, with narrow CPU-only framing

The blockers are not "no external audit" and not adoption optics. The real
blockers are the points where an optional backend may behave differently from
`bitcoin-core/secp256k1` or may be unacceptable to Bitcoin Core's build,
consensus, and reproducibility expectations.

## CAAS Positioning

The correct PR narrative is:

> This is not asking Bitcoin Core to trust an unaudited replacement. This is an
> optional, default-off backend candidate with a reproducible CAAS evidence
> chain: claim -> executable test -> CI gate -> audit bundle -> replayable
> verification.

Relevant CAAS properties:

- Every security claim should map to executable tests.
- CAAS runs on every push/PR.
- A fixed bug becomes a permanent regression test.
- The repo carries its audit infrastructure: scripts, workflows, PoCs, bundle
  producer, bundle verifier, and source graph.
- Third parties can replay the evidence instead of trusting a static PDF.

CAAS is therefore the argument. It should be packaged as a reviewer-facing
artifact, not described defensively.

## Real Blockers

### 1. libsecp256k1 Shim Strict Accept/Reject Parity

This is the highest-priority blocker.

The shim must be byte/return-code/accept-reject compatible with
`bitcoin-core/secp256k1` for every API path Bitcoin Core uses.

Current graph signals:

- `focus libsecp256k1_shim --core` surfaces the shim as the primary Bitcoin Core
  compatibility surface.
- `focus bitcoin_core --core` prioritizes:
  - `secp256k1_ec_seckey_verify`
  - `secp256k1_ecdsa_verify`
  - `secp256k1_schnorrsig_verify`
  - `secp256k1_ec_pubkey_create`
  - `secp256k1_ecdsa_recover`
- Call frontier shows shim parsing paths calling reducing constructors such as
  `FieldElement::from_bytes` and `Scalar::from_bytes`.

Risk:

Internal reducing constructors are appropriate for internal math, but
libsecp-compatible public boundaries must reject non-canonical input rather than
reduce it.

Specific surfaces to verify/fix:

- `compat/libsecp256k1_shim/src/shim_seckey.cpp`
  - `secp256k1_ec_seckey_verify`
  - `secp256k1_ec_seckey_negate`
  - `secp256k1_ec_seckey_tweak_add`
  - `secp256k1_ec_seckey_tweak_mul`
- `compat/libsecp256k1_shim/src/shim_pubkey.cpp`
  - `secp256k1_ec_pubkey_parse`
  - `secp256k1_ec_pubkey_create`
  - `secp256k1_ec_pubkey_tweak_add`
  - `secp256k1_ec_pubkey_tweak_mul`
- `compat/libsecp256k1_shim/src/shim_ecdsa.cpp`
  - compact parse
  - DER parse
  - verify
  - sign
- `compat/libsecp256k1_shim/src/shim_schnorr.cpp`
  - Schnorr sign/verify
- `compat/libsecp256k1_shim/src/shim_extrakeys.cpp`
  - x-only pubkey and keypair operations
- `compat/libsecp256k1_shim/src/shim_recovery.cpp`
  - recoverable ECDSA signatures

Required behavior:

- Secret keys: reject `0`, reject `>= n`.
- Scalar tweaks: reject `>= n`; reject zero only where libsecp rejects zero.
- Field elements: reject `>= p` on public parse boundaries.
- Pubkeys: reject bad prefix, off-curve, infinity, non-canonical coordinates.
- ECDSA compact: reject `r == 0`, `s == 0`, `r >= n`, `s >= n`.
- DER: match libsecp's strict DER behavior exactly where Bitcoin Core expects
  strict parsing.
- Schnorr: reject `r >= p`, `s == 0`, `s >= n`, invalid x-only keys.

Definition of done:

- Upstream libsecp test vectors pass against the shim.
- A dedicated shim parity test compares libsecp and shim return codes on hostile
  inputs.
- Source graph marks these functions as `compat-shim`, `bitcoin-core`,
  `strict-parser`, and high-priority.

### 2. CT Default Cleanup for Public C++ Signing

This is a Core-readiness blocker because the shim currently calls public C++
signing APIs.

Current shim use:

- `shim_ecdsa.cpp` calls:
  - `secp256k1::ecdsa_sign`
  - `secp256k1::ecdsa_sign_hedged`
- `shim_schnorr.cpp` calls:
  - `secp256k1::schnorr_sign`

If these public C++ defaults route through variable-time internals, then the
shim inherits that behavior.

Required cleanup:

- Public default `secp256k1::ecdsa_sign*` must dispatch to
  `secp256k1::ct::ecdsa_sign*`.
- Public default `secp256k1::ecdsa_sign_recoverable` must dispatch to
  `secp256k1::ct::ecdsa_sign_recoverable`.
- Public default `secp256k1::schnorr_pubkey`,
  `secp256k1::schnorr_keypair_create`, `secp256k1::schnorr_sign`, and
  `secp256k1::schnorr_sign_verified` must dispatch to CT implementations.
- Variable-time signing may remain only behind explicit unsafe/fast naming or a
  build-time opt-in such as `SECP256K1_ALLOW_FAST_SIGN`.

Reference workdoc:

- `docs/CT_DEFAULT_CLEANUP_WORKDOC.md`

Definition of done:

- Shim signing path is CT by construction.
- Source graph reports public default signing symbols as CT-covered.
- Non-CT signing no longer appears as a normal production API.

### 3. Nonce Callback and R-Grinding Compatibility

The ECDSA shim currently needs exact compatibility with libsecp nonce behavior
where Bitcoin Core relies on it.

Risk areas:

- `secp256k1_ecdsa_sign` accepts `noncefp` and `ndata`.
- Current shim behavior must be checked against libsecp's nonce callback
  contract.
- Bitcoin Core's R-grinding behavior must not loop, diverge, or silently produce
  incompatible signatures.

Required work:

- Document which Bitcoin Core signing paths pass `noncefp` / `ndata`.
- Add tests for Core-style R-grinding calls.
- Either fully implement libsecp nonce callback semantics, or prove that Core's
  used subset is exactly supported.

Definition of done:

- Same call pattern as Bitcoin Core produces valid signatures.
- No infinite-loop risk.
- Return codes match libsecp on callback failure.
- `ndata` behavior is documented and tested.

### 4. `context_randomize` / Blinding Semantics

Bitcoin Core and libsecp expose `secp256k1_context_randomize`.

Current state observed in the shim:

- The context stores `blind[32]` and `blinded`.
- The stored seed is not clearly shown as participating in signing/keygen
  blinding.

Risk:

Reviewers will ask whether libsecp's context randomization security property is
preserved, intentionally replaced by CT guarantees, or ignored.

Acceptable resolutions:

1. Implement real context blinding compatible with the shim architecture.
2. Document a formal argument that CT signing makes libsecp-style blinding
   unnecessary in this backend.
3. Scope Core backend mode so the difference is explicit, tested, and accepted.

Definition of done:

- `context_randomize` behavior is documented in Bitcoin Core integration docs.
- Tests show return behavior matches libsecp.
- The security property is either implemented or explicitly replaced by a
  documented CT-based argument.

### 5. Bitcoin Core Full Test Evidence

The repo has cross-lib differential tests, but a Bitcoin Core backend PR needs
evidence from Bitcoin Core itself.

Required evidence:

- Bitcoin Core builds with default bundled `libsecp256k1`.
- Bitcoin Core builds with UltrafastSecp256k1 backend.
- Full unit and functional test suites run under both builds.
- Script/signature/taproot relevant tests are highlighted.
- The results are pinned in a CAAS-style evidence bundle.

Definition of done:

- One command builds and tests the UF backend inside Bitcoin Core.
- Report compares default backend vs UF backend.
- Any skipped tests are justified.

### 6. Core-Style Build Integration

The existing integration guide is useful but not acceptable as-is for upstream
Bitcoin Core.

Current risk:

- `FetchContent` with remote GitHub URL and branch/tag style is not a good
  upstream Core integration model.
- Core should not fetch consensus crypto code during configure.

Required integration properties:

- Default `OFF`.
- No network during configure/build.
- Pinned source or explicit local path.
- Reproducible source hash.
- CI job explicitly enables UF backend.
- No GPU or optional runtime features in the Core backend path.

Definition of done:

- `-DUSE_ULTRAFAST_SECP256K1=ON` or equivalent works without network access.
- Default build remains untouched.
- Build files are minimal and reviewable.

### 7. Runtime Env/Cache Determinism

Bitcoin Core's crypto backend must not depend on ambient environment state.

Risk surface:

- Fixed-base setup may consult environment variables, config files, or cache
  paths.
- Core backend mode should not allow CWD `config.ini` or external cache files to
  alter crypto backend behavior.

Required Core mode:

- No `SECP256K1_CONFIG` influence by default.
- No CWD `config.ini`.
- No external cache dependency.
- Deterministic built-in or build-pinned table selection.
- No hidden runtime generation that changes test behavior.

Definition of done:

- Core backend mode is deterministic from source + build flags.
- Any cache/table behavior is compile-time selected or disabled.

### 8. CAAS Reviewer-Facing Package

CAAS is strong, but reviewers need a compact package.

Required artifacts:

- Latest CAAS green status.
- `EXTERNAL_AUDIT_BUNDLE.json` or Core-specific equivalent.
- Bundle SHA256.
- One replay command.
- Short human-readable dashboard or summary.
- Explicit residual risks and scope exclusions.

Recommended framing:

- CPU-only.
- `libsecp256k1` shim only.
- Default OFF.
- No GPU claims.
- No multicoin/ZK/FROST/MuSig2 claims in the Bitcoin Core PR.

Definition of done:

- A reviewer can run one command and reproduce the evidence.
- CAAS story is visible without reading the whole repo.

## Non-Blockers

These are not real technical blockers for the Bitcoin Core optional backend
proposal:

- No external audit.
- Low GitHub stars or public adoption metrics.
- GPU formal CT limitations, if PR is CPU-only.
- Multicoin, ZK, FROST, MuSig2 maturity, if out of scope.
- Performance benchmark artifacts, if no performance claim is made.
- Lack of precedent; this is a review-strategy issue, not a technical blocker.

## Priority Order

1. Fix shim strict accept/reject parity.
2. Make public C++ signing default to CT because the shim calls it.
3. Resolve nonce callback / R-grinding compatibility.
4. Resolve `context_randomize` / blinding semantics.
5. Add deterministic Core backend build mode.
6. Remove ambient env/cache behavior from Core mode.
7. Run Bitcoin Core full tests with both backends.
8. Produce Core-specific CAAS evidence bundle.
9. Open draft/RFC PR with narrow CPU-only framing.

## Suggested PR Framing

Title direction:

```text
build: add experimental optional secp256k1 backend hook
```

Positioning:

```text
This PR does not replace the default Bitcoin Core secp256k1 backend. It adds a
default-off experimental backend path for evaluating a libsecp256k1-compatible
shim backed by UltrafastSecp256k1. The backend is CPU-only, build-time selected,
and accompanied by reproducible CAAS evidence and differential parity results.
```

Avoid:

- performance-first language
- GPU mentions
- “replacement” language
- external-audit debate
- broad feature claims outside libsecp compatibility

## Bottom Line

CAAS is the strength. The remaining blockers are not about credibility in the
abstract; they are concrete compatibility and integration issues:

- public inputs must reject exactly like libsecp;
- signing paths must be CT by default;
- nonce/context semantics must match or be precisely justified;
- Core build/test evidence must be produced in Bitcoin Core itself;
- the backend path must be deterministic and default-off.

Once these are closed, the proposal becomes a serious first-of-its-kind
experimental backend discussion rather than a risky replacement request.
