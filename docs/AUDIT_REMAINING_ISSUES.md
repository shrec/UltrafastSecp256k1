# Audit — remaining issues to fix (best done on Linux)

After the Windows audit-parity work (`ci/_ufsecp.py` cross-platform `find_lib`,
Z3 + Lean installed), `ci_local.sh` drops from **5/29 → 2/28** failing gates on
Windows, and the audit now runs end-to-end. The items below are the **remaining
real bugs** — none are Windows-specific; they fail identically on Linux and are
easiest to fix there (full dev environment, Cryptol native).

---

## BUG 1 — Cryptol formal specs do not parse / are not actually proven  *(blocks the Cryptol formal gate)*

Two independent defects keep the Cryptol part of formal verification from ever
running (today it ADVISORY-SKIPs because Cryptol is normally absent in CI):

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

## BUG 2 — External audit-evidence bundle hash drift  *(Bundle integrity gate FAIL)*

`ci/verify_external_audit_bundle.py` reports `passing:false` — the recorded
`expected_hash` no longer matches the recomputed `actual_hash`. The bundle was
not regenerated after recent evidence changes. Regenerate it (owner/release step)
so the manifest hash matches current evidence, then re-verify.

---

## BUG 3 — FE52-compute perf commit lacks an in-commit test  *(Security-fix-has-test gate FAIL)*

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
