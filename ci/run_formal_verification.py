#!/usr/bin/env python3
"""
Formal verification runner for UltrafastSecp256k1.

BLOCKING gate: Z3 SMT and Lean 4 proofs MUST pass.
If either tool is absent or fails → exit 1 (hard failure).

Cryptol type-check is advisory: absent Cryptol binary → skip (not a failure).
If Cryptol IS installed and type-check fails → exit 1.

Tools:
  Z3 SMT    — audit/formal/safegcd_z3_proof.py   (~2s, pip install z3-solver)
  Lean 4    — audit/formal/lean/  (lake build)    (~5min after elan install)
  Cryptol   — audit/formal/cryptol/ (.cry files)  (advisory: skip if absent)

Exit codes:
  0   all required tools passed (Cryptol skipped if absent)
  1   any required tool missing OR any proof failed
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAL_DIR = REPO_ROOT / "audit" / "formal"

g_pass = 0
g_fail = 0
g_skip = 0


def run_tool(label: str, cmd: list[str], cwd: Path | None = None) -> int:
    """Run a command. Returns 0 on pass, 1 on fail/timeout."""
    global g_pass, g_fail
    print(f"  [{label}]", end=" ", flush=True)
    try:
        result = subprocess.run(
            cmd, cwd=cwd or REPO_ROOT,
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            print("PROVED")
            g_pass += 1
            return 0
        else:
            print("FAILED")
            for line in (result.stdout + result.stderr).splitlines()[-15:]:
                print(f"    {line}")
            g_fail += 1
            return 1
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        g_fail += 1
        return 1


def tool_available(name: str) -> bool:
    try:
        subprocess.run([name, "--version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def z3_available() -> bool:
    try:
        import z3  # noqa: F401
        return True
    except ImportError:
        return False


def main() -> int:
    global g_pass, g_fail, g_skip
    rc = 0

    print("Formal Verification — UltrafastSecp256k1")
    print(f"  Formal dir: {FORMAL_DIR.relative_to(REPO_ROOT)}")
    print()

    # ── Z3 SMT proofs (REQUIRED) ─────────────────────────────────────────────
    print("[Z3 SMT] SafeGCD / Bernstein-Yang divstep proofs  [REQUIRED]")
    z3_script = FORMAL_DIR / "safegcd_z3_proof.py"
    if not z3_available():
        print("  [z3] MISSING — z3-solver not installed (pip install z3-solver)")
        g_fail += 1
        rc = 1
    elif not z3_script.exists():
        print(f"  [z3] MISSING — {z3_script} not found")
        g_fail += 1
        rc = 1
    else:
        if run_tool("z3", [sys.executable, str(z3_script)]) != 0:
            rc = 1

    # ── Lean 4 proofs (REQUIRED) ─────────────────────────────────────────────
    print("[Lean 4] SafeGCD formal proofs  [REQUIRED]")
    lean_dir = FORMAL_DIR / "lean"
    if not tool_available("lake"):
        print("  [lean] MISSING — lake/elan not installed (see audit/formal/lean/lean-toolchain)")
        g_fail += 1
        rc = 1
    elif not lean_dir.exists():
        print(f"  [lean] MISSING — {lean_dir} not found")
        g_fail += 1
        rc = 1
    else:
        if run_tool("lean", ["lake", "build"], cwd=lean_dir) != 0:
            rc = 1

    # ── Cryptol type-check (ADVISORY: skip if absent) ────────────────────────
    print("[Cryptol] Type-check .cry specifications  [advisory — skip if absent]")
    cryptol_dir = FORMAL_DIR / "cryptol"
    if not tool_available("cryptol"):
        print("  [cryptol] SKIP (cryptol not installed — not required)")
        g_skip += 1
    elif not cryptol_dir.exists():
        print(f"  [cryptol] SKIP ({cryptol_dir} not found)")
        g_skip += 1
    else:
        for cry_file in sorted(cryptol_dir.glob("*.cry")):
            if run_tool(f"cryptol/{cry_file.name}", ["cryptol", "-b", str(cry_file)]) != 0:
                rc = 1

    print()
    print(f"Result: {g_pass} proved, {g_fail} failed, {g_skip} skipped")
    if rc != 0:
        print("FORMAL VERIFICATION FAILED — see errors above")
    else:
        print("FORMAL VERIFICATION PASSED")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
