# Audit — remaining issues to fix (best done on Linux)

After the Windows audit-parity work (`ci/_ufsecp.py` cross-platform `find_lib`,
Z3 + Lean installed), `ci_local.sh` drops from **5/29 → 2/28** failing gates on
Windows, and the audit now runs end-to-end. The items below are the **remaining
real bugs** — none are Windows-specific; they fail identically on Linux and are
easiest to fix there (full dev environment, Cryptol native).

> **UPDATE 2026-06-23 (Linux):** ALL THREE fixed. BUG 2 (bundle hash drift) and BUG 3
> (FE52-compute commit lacked an in-commit test) in commit `c9ae316c`. BUG 1 (Cryptol
> formal specs) is now fixed too: cryptol 3.5.0 installed, all four specs parse + type-check
> and their tractable properties pass `:check` (Field 15/15, Point 10/10, ECDSA 8/8 incl.
> sign→verify, Schnorr structural), and the gate now runs real `.icry` property checks
> instead of the no-op `cryptol -b <file>.cry` REPL batch. `run_formal_verification.py`:
> z3 ✓ lean ✓ cryptol ✓ all PROVED.

---

## BUG 1 — Cryptol formal specs do not parse / are not actually proven  *(blocks the Cryptol formal gate)*  ✅ FIXED

> **Fixed (Linux, cryptol 3.5.0).** Beyond the two defects below, the specs also had
> mathematically-wrong arithmetic (Cryptol's `(+)`/`(*)` on `[256]` truncate mod 2^256, so
> `field_mul_ref`/`field_add`/`scalar_mod_*` silently dropped the high half) and an invalid
> `primitive Maybe` construct — all corrected to full-width zero-extended ops + a record
> `Maybe`. The gate (`ci/run_formal_verification.py`) now runs per-spec `.icry` runners
> (`:load` + `:check`) so a type error or counterexample fails the gate; the old
> `cryptol -b <file>.cry` executed the file as a REPL batch and checked **nothing**.
> Verified end-to-end: `run_formal_verification.py` → z3 ✓ lean ✓ cryptol ✓ all PROVED.
> Linux install: download `cryptol-3.5.0-ubuntu-24.04-X64.tar.gz` from the GaloisInc/cryptol
> 3.5.0 release; `:check` needs no SMT solver (it is randomized testing).

Two independent defects (now fixed) kept the Cryptol part from ever running:

### 1a. Pre-3.5 syntax in the `.cry` specs (parse error)
`audit/formal/cryptol/Secp256k1Field.cry:182` — a `let … in` expression inside a
`property` body is rejected by Cryptol **3.5**:

```cryptol
property field_sqrt_correct a =
    (a < P /\ field_is_square a) ==>
    let s = field_sqrt a                     // <-- Parse error: unexpected `let`
    in  field_mul_ref s s == a \/ field_mul_ref (field_neg s) (field_neg s) == a
```

Fix: move the binding to a `where` clause (or inline it):

```cryptol
property field_sqrt_correct a =
    (a < P /\ field_is_square a) ==>
    (field_mul_ref s s == a \/ field_mul_ref (field_neg s) (field_neg s) == a)
    where s = field_sqrt a
```

Audit **all four** specs for the same pattern and any other 3.x deprecations:
`Secp256k1Field.cry`, `Secp256k1Point.cry`, `Secp256k1ECDSA.cry`,
`Secp256k1Schnorr.cry`. Confirm each loads clean: `cryptol <file>.cry`.

### 1b. The gate invocation runs the spec as a REPL batch, not a module
`ci/run_formal_verification.py:134`:

```python
if run_tool(f"cryptol/{cry_file.name}", ["cryptol", "-b", str(cry_file)]) != 0:
```

`cryptol -b FILE` executes FILE as a sequence of REPL commands, so top-level
`field_*` definitions do not persist and the properties report
`Value not in scope: field_sub` even when the file is otherwise valid. The gate
should **load the module and check/prove its properties**, e.g. drive an `.icry`
script per file:

```
:load audit/formal/cryptol/Secp256k1Field.cry
:check        // random-test all properties  (or :prove NAME for SMT, slow on 256-bit)
:quit
```

i.e. `cryptol -b <generated.icry>` (NOT the `.cry` itself), or `cryptol -c ":check" <file>.cry`.

Note: full `:prove` over GF(p) with a 256-bit modulus may not terminate in the
SMT backend; `:check` (exhaustive for small types, randomized for large) is the
realistic gate. Decide proof vs check policy when wiring this.

Once 1a + 1b land, Cryptol becomes a real REQUIRED formal gate instead of an
advisory skip. The Windows toolchain is already in place
(`cryptol-3.5.0-...-with-solvers`, see `docs/AUDIT_WINDOWS_SETUP.md`).

---

## BUG 2 — External audit-evidence bundle hash drift  *(Bundle integrity gate FAIL)*  ✅ FIXED (c9ae316c)

`ci/verify_external_audit_bundle.py` reported `passing:false` — the recorded
`expected_hash` no longer matched the recomputed `actual_hash` (drifted evidence:
`docs/FFI_HOSTILE_CALLER.md`). Regenerated via `ci/external_audit_bundle.py`;
`verify_external_audit_bundle.py --allow-commit-mismatch` now passes 12/12.

---

## BUG 3 — FE52-compute perf commit lacks an in-commit test  *(Security-fix-has-test gate FAIL)*  ✅ FIXED (c9ae316c)

> Fixed differently from the "amend / age-out" suggestion below: added
> `audit/test_fe52_compute_verify.cpp` (module `fe52_compute_verify`, section differential)
> wired into the runner, and recorded `875d5bee9f` in `RETROACTIVELY_COVERED`
> (`check_security_fix_has_test.py`, FROZEN_COUNT 63→64). The gate now passes without a
> force-push. The test pins `dual_scalar_mul_gen_point(u1,u2,Q) == u1*G + u2*Q` (200 vectors)
> + ECDSA/Schnorr verify round-trip + tamper rejection (392/392).

`ci/check_security_fix_has_test.py --commits 10` flags `875d5bee9`
(`perf(cpu): FE52-compute …`): it changed `src/cpu/src/field_52.cpp` and
`src/cpu/src/point.cpp` (security paths) with no test file in the SAME commit
(CLAUDE.md "Exploit / Audit Test Conversion Standard").

The behavior is already covered by `run_selftest` (31/31, KAT + CT-equivalence),
but the rule is per-commit and mechanical. To satisfy it, add an FE52-compute
differential/KAT test (e.g. `audit/test_fe52_compute_verify.cpp`: assert the
FE52-compute ECDSA/Schnorr verify result is bit-identical to the 4×64 path on a
vector set) and wire it into `audit/unified_audit_runner.cpp`. Since `875d5bee9`
is already pushed, either amend it (force-push `dev`) or add the test in a
follow-up commit and let the per-commit flag age out of the 10-commit window.
