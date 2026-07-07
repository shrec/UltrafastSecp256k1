# Running the local audit on Windows

`ci/ci_local.sh` runs the same gate suite as CI. On Windows it now behaves the
same as on Linux once the toolchain below is present. One-shot installer:

```powershell
pwsh ci/setup_windows_audit.ps1            # Z3 + Lean + python3 shim
pwsh ci/setup_windows_audit.ps1 -WithCryptol   # also download Cryptol (advisory)
```

Then build the C-ABI shared library and run the audit from Git Bash:

```bash
cmake --build out/<profile> --target ufsecp_shared   # emits out/<profile>/include/ufsecp/ufsecp.dll
./ci/ci_local.sh
```

## Why it failed before

| Gate | Symptom on Windows | Fix |
|------|--------------------|-----|
| Determinism gate (+ 22 `_ufsecp`-importing audit scripts) | `Cannot locate a loadable libufsecp.so` | `ci/_ufsecp.py` `find_lib()` is now cross-platform — it discovers `ufsecp.dll` (Windows) / `.dylib` (macOS) / `.so` (Linux) and adds the lib's own dir to the Windows DLL search. The build already emits `out/<profile>/include/ufsecp/ufsecp.dll`. |
| Formal verification (Z3 + Lean + Cryptol) | `z3 MISSING`, `lake/elan not installed` | Install Z3 (`pip install z3-solver`) and Lean (`elan` → `lake`). Both are **required** and now PROVE on Windows. |

`.so` is a Linux/Unix shared-object extension; Windows produces `.dll` instead.
The library was never "missing" — the audit's discovery logic only searched for
the Linux name. That is the one-line-per-platform fix in `find_lib`.

## Tooling specifics

- **Z3** — `pip install z3-solver` (the `z3` Python module). The Z3 SafeGCD proof
  (`audit/formal/safegcd_z3_proof.py`) runs in well under a second.
- **Lean 4** — `elan` (Lean version manager). `lake build` in `audit/formal/lean/`
  fetches the pinned toolchain (`lean-toolchain`, currently `v4.18.0`) on first run
  and builds the SafeGCD proofs. No mathlib dependency, so the build is fast.
- **Cryptol** — `cryptol-3.5.0-windows-...-with-solvers` bundles `cryptol.exe` plus
  `z3.exe`/CVC. **Advisory only:** Cryptol absent → the formal gate ADVISORY-SKIPs
  (exit 77), the same state as Linux CI where Cryptol is typically not installed.
  The bundled specs `audit/formal/cryptol/*.cry` currently use pre-3.5 syntax
  (e.g. `let … in` inside a `property`, parse error at `Secp256k1Field.cry:182`) and
  the gate's `cryptol -b <file>.cry` batch invocation does not load them as modules,
  so they are **not yet runnable** — keep Cryptol OFF the audit PATH until the specs
  are updated, so the gate skips cleanly instead of hard-failing.

## What still fails (and why it is NOT a Windows-portability issue)

These fail identically on Linux without the same provisioning:

- **Bundle integrity** — the external audit-evidence bundle hash drifted from the
  recorded value; regenerate the bundle (a release/owner step), unrelated to OS.
- **Security fix has test** — `check_security_fix_has_test.py` flags any commit
  touching `src/cpu/src/` without a test file in the *same* commit (CLAUDE.md
  "Exploit / Audit Test Conversion Standard"); a content rule, not OS-specific.
- **Source graph quality** — the graph DB goes stale after new commits; rebuild with
  `python3 tools/source_graph_kit/source_graph.py build -i`.
