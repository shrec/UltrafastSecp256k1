#!/usr/bin/env python3
"""
Formal verification runner for UltrafastSecp256k1.

Runs available formal verification tools and reports results.
Returns ADVISORY_SKIP_CODE (77) if no tools are installed.
Returns 0 if all available tools pass.
Returns 1 if any tool reports a proof failure.

Tools (in order of priority):
  Z3 SMT    — audit/formal/safegcd_z3_proof.py   (fast ~2s, pip install z3-solver)
  Lean 4    — audit/formal/lean/  (lake build)    (slow ~10min, requires elan)
  Cryptol   — audit/formal/cryptol/              (requires cryptol binary)

CAAS role: advisory gate — runs in ci_local.sh [4] and caas.yml Stage 3.
Exit codes:
  0  all available tools passed
  1  at least one tool failed
  77 no formal tools available (advisory skip)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ADVISORY_SKIP_CODE = 77
REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAL_DIR = REPO_ROOT / "audit" / "formal"

g_pass = 0
g_fail = 0
g_skip = 0


def run(label: str, cmd: list[str], cwd: Path | None = None) -> int:
    """Run a command and print result. Returns exit code."""
    global g_pass, g_fail
    print(f"  [{label}]", end=" ", flush=True)
    try:
        result = subprocess.run(
            cmd, cwd=cwd or REPO_ROOT,
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            print("PROVED")
            g_pass += 1
        else:
            print("FAILED")
            for line in (result.stdout + result.stderr).splitlines()[-10:]:
                print(f"    {line}")
            g_fail += 1
        return result.returncode
    except FileNotFoundError:
        print("SKIP (tool not found)")
        g_skip += 1
        return ADVISORY_SKIP_CODE
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        g_fail += 1
        return 1


def check_z3() -> bool:
    """Return True if z3 Python package is importable."""
    try:
        import z3  # noqa: F401
        return True
    except ImportError:
        return False


def check_lean() -> bool:
    """Return True if `lake` binary is available."""
    try:
        subprocess.run(["lake", "--version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_cryptol() -> bool:
    """Return True if `cryptol` binary is available."""
    try:
        subprocess.run(["cryptol", "--version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def main() -> int:
    global g_pass, g_fail, g_skip

    print("Formal Verification — UltrafastSecp256k1")
    print(f"  Formal dir: {FORMAL_DIR.relative_to(REPO_ROOT)}")
    print()

    # ── Z3 SMT proofs ────────────────────────────────────────────────────────
    z3_script = FORMAL_DIR / "safegcd_z3_proof.py"
    if check_z3() and z3_script.exists():
        print("[Z3 SMT] SafeGCD / Bernstein-Yang divstep proofs")
        run("z3", [sys.executable, str(z3_script)])
    else:
        reason = "z3 not installed" if not check_z3() else "script missing"
        print(f"[Z3 SMT] SKIP ({reason})")
        g_skip += 1

    # ── Lean 4 proofs ─────────────────────────────────────────────────────────
    lean_dir = FORMAL_DIR / "lean"
    if check_lean() and lean_dir.exists():
        print("[Lean 4] SafeGCD formal proofs (lake build)")
        run("lean", ["lake", "build"], cwd=lean_dir)
    else:
        reason = "lake not found" if not check_lean() else "lean dir missing"
        print(f"[Lean 4] SKIP ({reason})")
        g_skip += 1

    # ── Cryptol type-check ────────────────────────────────────────────────────
    cryptol_dir = FORMAL_DIR / "cryptol"
    if check_cryptol() and cryptol_dir.exists():
        print("[Cryptol] Type-check all .cry specifications")
        for cry_file in sorted(cryptol_dir.glob("*.cry")):
            run(f"cryptol/{cry_file.name}", ["cryptol", "-b", str(cry_file)])
    else:
        reason = "cryptol not installed" if not check_cryptol() else "cryptol dir missing"
        print(f"[Cryptol] SKIP ({reason})")
        g_skip += 1

    print()
    print(f"Result: {g_pass} proved, {g_fail} failed, {g_skip} skipped")

    if g_fail > 0:
        return 1
    if g_pass == 0:
        # Nothing ran — advisory skip
        print("(no formal tools available — advisory skip)")
        return ADVISORY_SKIP_CODE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
