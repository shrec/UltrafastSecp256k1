#!/usr/bin/env python3
"""Backend parity gate — detects copy-paste divergence across signing/ECDH backend families.

Catches:
  C1 — scalar_is_even used instead of scalar_is_low_s for low-S normalization
  C2 — non-CT scalar_mul called with secret nonce k before signature computation
  C3 — early `break` inside overflow-check loop (variable-time comparison on rx_bytes)
  C8 — hardcoded `prefix = 0x02` in ECDH (ignores Y parity)

Each family defines files that must implement equivalent security properties.
The gate reads each file via the source graph and applies pattern-level checks.

Exit codes:
  0  — all families clean
  1  — one or more violations found
  77 — advisory-skip (no source graph DB / all target files absent)
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT   = SCRIPT_DIR.parent
GRAPH_DB   = LIB_ROOT / "tools" / "source_graph_kit" / "source_graph.db"

# ---------------------------------------------------------------------------
# File families and their check specifications
# ---------------------------------------------------------------------------

# Family 1: ECDSA recoverable sign backends
# These files implement the recoverable ECDSA signing primitive. They MUST
# use CT-correct patterns that were confirmed working in the reference
# implementation (kernels/secp256k1_extended.cl, include/ct/ct_sign.cuh).
FAMILY_SIGN_FILES = [
    "kernels/secp256k1_recovery.cl",
    "kernels/secp256k1_extended.cl",
    "src/cuda/include/recovery.cuh",
    "src/metal/shaders/secp256k1_recovery.h",
]

# Family 2: ECDH compute backends
# These files implement ECDH — they MUST derive Y parity and NOT hardcode 0x02.
FAMILY_ECDH_FILES = [
    "kernels/secp256k1_ecdh.cl",
    "kernels/secp256k1_extended.cl",
    "src/metal/shaders/secp256k1_extended.h",
]

# ---------------------------------------------------------------------------
# Pattern checkers
# ---------------------------------------------------------------------------

# C1: `scalar_is_even` used for low-S normalization.
# The correct predicate is scalar_is_low_s / ct_scalar_is_high.
# We flag any file that contains "scalar_is_even" WITHOUT a comment explicitly
# saying it is a *test* for the even/low-s distinction (those live in audit/).
_RE_C1_BAD     = re.compile(r'\bscalar_is_even\b(?!_impl\s*\(.*test)')
_RE_C1_AUDIT   = re.compile(r'test_low_s_neq_even|audit|test_exploit')

# C2: non-CT scalar_mul on the nonce variable k.
# Pattern: "scalar_mul_impl(" or "scalar_mul_generator(" appearing in a signing
# function with a local variable named `k` and before the result is stored as r.
# We match lines that call variable-time scalar_mul with argument `k`.
_RE_C2_BAD = re.compile(
    r'\b(scalar_mul_impl|scalar_mul_generator)\s*\('   # variable-time mul
    r'(?:[^;]{0,120},\s*)?'                            # optional earlier args
    r'[&*]?\bk\b'                                      # secret nonce arg
)

# C3: early break inside overflow comparison loop on rx_bytes.
# The break is variable-time: it exits at the first differing byte, leaking the
# position of the first byte where R.x differs from the group order.
# Pattern: a for-loop iterating rx_bytes that contains `break`.
_RE_C3_LOOP_START = re.compile(r'for\s*\(.*rx_bytes')
_RE_C3_BREAK      = re.compile(r'\bbreak\b')

# C8: hardcoded prefix = 0x02 in ECDH (ignores Y parity).
# The correct version derives parity from y_aff and sets prefix to 0x02 or 0x03.
_RE_C8_BAD = re.compile(r'prefix\s*=\s*0x02\b')
# We require 0x03 to be present somewhere in the same file (proof Y parity used)
_RE_C8_GOOD = re.compile(r'0x03\b')


def _fetch_file_content(conn: sqlite3.Connection, rel_path: str) -> str | None:
    """Return the full source text for a file from the graph DB, or None."""
    # Try function_bodies table first (aggregated per-function content)
    # Fall back to reading from disk using the relative path.
    abs_path = LIB_ROOT / rel_path
    if abs_path.exists():
        try:
            return abs_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
    return None


def _check_c1_c2_c3(content: str, rel_path: str) -> list[dict]:
    """Check for C1 (scalar_is_even), C2 (non-CT scalar_mul on k), C3 (early break)."""
    violations = []
    lines = content.splitlines()

    in_overflow_loop = False
    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()

        # --- C1: scalar_is_even used for signing normalization ---
        if _RE_C1_BAD.search(stripped):
            # Skip audit test files — they intentionally reference the wrong predicate
            if not _RE_C1_AUDIT.search(rel_path):
                violations.append({
                    "bug": "C1",
                    "file": rel_path,
                    "line": lineno,
                    "text": stripped[:120],
                    "message": "scalar_is_even used for low-S normalization (should be scalar_is_low_s/ct_scalar_is_high)",
                })

        # --- C2: variable-time scalar_mul called with nonce k ---
        if _RE_C2_BAD.search(stripped):
            violations.append({
                "bug": "C2",
                "file": rel_path,
                "line": lineno,
                "text": stripped[:120],
                "message": "non-CT scalar_mul called with nonce k (use ct_generator_mul_impl / ct_generator_mul)",
            })

        # --- C3: early break inside overflow loop over rx_bytes ---
        if _RE_C3_LOOP_START.search(stripped):
            in_overflow_loop = True
        if in_overflow_loop:
            if _RE_C3_BREAK.search(stripped):
                violations.append({
                    "bug": "C3",
                    "file": rel_path,
                    "line": lineno,
                    "text": stripped[:120],
                    "message": "early 'break' inside rx_bytes overflow loop (variable-time — use branchless MSB cascade)",
                })
                in_overflow_loop = False  # report once per loop
            # End of loop block heuristic
            if stripped.startswith("}") and in_overflow_loop:
                in_overflow_loop = False

    return violations


def _check_c8(content: str, rel_path: str) -> list[dict]:
    """Check for C8: hardcoded prefix = 0x02 without Y parity computation."""
    violations = []
    if not _RE_C8_BAD.search(content):
        return violations

    # The file has `prefix = 0x02`.  If it ALSO has 0x03 nearby we consider it
    # intentional (both branches present → Y parity is computed correctly).
    if _RE_C8_GOOD.search(content):
        return violations  # both 0x02 and 0x03 present — parity is handled

    # 0x02 present but 0x03 absent → hardcoded, Y parity ignored
    lines = content.splitlines()
    for lineno, line in enumerate(lines, 1):
        if _RE_C8_BAD.search(line.strip()):
            violations.append({
                "bug": "C8",
                "file": rel_path,
                "line": lineno,
                "text": line.strip()[:120],
                "message": "prefix = 0x02 hardcoded in ECDH (Y parity ignored; must set 0x02/0x03 from y_aff)",
            })
    return violations


def run(json_mode: bool, out_file: str | None) -> int:
    conn = None
    if GRAPH_DB.exists():
        try:
            conn = sqlite3.connect(str(GRAPH_DB))
        except Exception:
            conn = None

    violations: list[dict] = []
    files_checked = 0
    files_absent  = 0

    # ── Family 1: signing families — C1, C2, C3 ────────────────────────────
    for rel_path in FAMILY_SIGN_FILES:
        content = _fetch_file_content(conn, rel_path)
        if content is None:
            files_absent += 1
            continue
        files_checked += 1
        violations.extend(_check_c1_c2_c3(content, rel_path))

    # ── Family 2: ECDH families — C8 ───────────────────────────────────────
    for rel_path in FAMILY_ECDH_FILES:
        # Avoid re-checking files already in family 1 for C1/C2/C3
        content = _fetch_file_content(conn, rel_path)
        if content is None:
            files_absent += 1
            continue
        files_checked += 1
        violations.extend(_check_c8(content, rel_path))

    if conn:
        conn.close()

    # ── Advisory skip if no files could be read ─────────────────────────────
    total_files = len(FAMILY_SIGN_FILES) + len(FAMILY_ECDH_FILES)
    if files_checked == 0:
        msg = "ADVISORY-SKIP: no target files found — backend parity check skipped"
        if json_mode:
            print(json.dumps({"advisory_skip": True, "reason": msg}))
        else:
            print(msg)
        return 77

    overall_pass = len(violations) == 0

    report = {
        "overall_pass": overall_pass,
        "files_checked": files_checked,
        "files_absent": files_absent,
        "total_target_files": total_files,
        "violations": violations,
        "violation_count": len(violations),
    }

    rendered = json.dumps(report, indent=2)
    if out_file:
        Path(out_file).write_text(rendered, encoding="utf-8")

    if json_mode:
        print(rendered)
    else:
        if violations:
            print(f"  Backend parity violations ({len(violations)} total):")
            for v in violations:
                print(f"    [{v['bug']}] {v['file']}:{v['line']}")
                print(f"         {v['message']}")
                print(f"         >> {v['text']}")
        else:
            print(f"  Checked {files_checked}/{total_files} files — no copy-paste divergence found")

        if overall_pass:
            print("PASS backend-parity gate")
        else:
            print(f"FAIL {len(violations)} backend parity violation(s) — see details above")

    return 0 if overall_pass else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("-o", dest="out_file", help="Write report to file")
    args = parser.parse_args()
    return run(args.json, args.out_file)


if __name__ == "__main__":
    raise SystemExit(main())
