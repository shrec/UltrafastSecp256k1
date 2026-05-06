#!/usr/bin/env python3
"""Protocol invariants gate — enforces ABI-boundary checks for MuSig2/FROST.

Catches:
  C5 — FROST: n_signers < kp.threshold check missing in ufsecp_frost_sign
       (signing with fewer than threshold participants produces silently invalid sigs)

Also asserts that the following EXISTING invariants have not been regressed:
  - ufsecp_frost_sign:          n_signers > kp.num_participants  check present
  - ufsecp_musig2_partial_sign: signer_index >= kagg.key_coefficients.size() check present

Source of truth: src/impl/ufsecp_musig2.cpp (via source graph body query or direct read).

Exit codes:
  0  — all invariants present
  1  — one or more invariants missing
  77 — advisory-skip (source file not found)
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

# Source graph records this as src/impl/ufsecp_musig2.cpp but the actual
# on-disk layout is src/cpu/src/impl/ufsecp_musig2.cpp.
_MUSIG2_CANDIDATES = [
    LIB_ROOT / "src" / "impl" / "ufsecp_musig2.cpp",
    LIB_ROOT / "src" / "cpu" / "src" / "impl" / "ufsecp_musig2.cpp",
    LIB_ROOT / "src" / "cpu" / "src" / "ufsecp_musig2.cpp",
]
MUSIG2_FILE = next((p for p in _MUSIG2_CANDIDATES if p.exists()), _MUSIG2_CANDIDATES[0])


def _extract_function_body(content: str, func_name: str) -> str | None:
    """Extract the body of a C++ function by name from file content."""
    # Find line where function name appears followed by '(' on the same or next few lines
    lines = content.splitlines()
    start_line = None
    for i, line in enumerate(lines):
        if func_name in line and re.search(r'\bfunc_name\b|\bfunc_name\s*\(', line.replace('func_name', func_name)):
            start_line = i
            break
        # More relaxed: function name then ( within 3 lines
        if re.search(r'\b' + re.escape(func_name) + r'\b', line):
            # Check next 3 lines for opening paren / brace
            context = " ".join(lines[i:min(i+4, len(lines))])
            if re.search(r'\b' + re.escape(func_name) + r'\s*\(', context):
                start_line = i
                break

    if start_line is None:
        return None

    # Find the opening brace of the function body
    brace_depth = 0
    body_start  = None
    for i in range(start_line, min(start_line + 20, len(lines))):
        for ch in lines[i]:
            if ch == '{':
                if body_start is None:
                    body_start = i
                brace_depth += 1
            elif ch == '}':
                brace_depth -= 1

    if body_start is None:
        return None

    # Collect until closing brace of the function
    brace_depth = 0
    body_lines  = []
    in_body     = False
    for line in lines[body_start:]:
        for ch in line:
            if ch == '{':
                brace_depth += 1
                in_body = True
            elif ch == '}':
                brace_depth -= 1
        body_lines.append(line)
        if in_body and brace_depth == 0:
            break

    return "\n".join(body_lines)


# ---------------------------------------------------------------------------
# Invariant specifications
# ---------------------------------------------------------------------------

INVARIANTS = [
    # C5 — FROST threshold enforcement (NEW — what the bug missed)
    {
        "id": "C5",
        "function": "ufsecp_frost_sign",
        "description": "FROST threshold: n_signers < kp.threshold check",
        "pattern": re.compile(r'n_signers\s*<\s*kp\.threshold'),
        "must_be_present": True,
        "rationale": (
            "FROST is a t-of-n threshold scheme. Signing with fewer than t participants "
            "produces an unverifiable partial signature that silently succeeds at the ABI "
            "boundary. The check n_signers < kp.threshold MUST be present."
        ),
    },
    # Regression guard: existing upper-bound check in frost_sign
    {
        "id": "FROST_UPPER",
        "function": "ufsecp_frost_sign",
        "description": "FROST upper bound: n_signers > kp.num_participants check",
        "pattern": re.compile(r'n_signers\s*>\s*kp\.num_participants'),
        "must_be_present": True,
        "rationale": (
            "Regression guard: n_signers > kp.num_participants check must remain in "
            "ufsecp_frost_sign to prevent over-committed signer count."
        ),
    },
    # Regression guard: signer_index bounds check in musig2_partial_sign
    {
        "id": "MUSIG2_SIGNER_INDEX",
        "function": "ufsecp_musig2_partial_sign",
        "description": "MuSig2 signer_index out-of-range check",
        "pattern": re.compile(r'signer_index\s*>=\s*kagg\.key_coefficients\.size\(\)'),
        "must_be_present": True,
        "rationale": (
            "Regression guard: the signer_index >= kagg.key_coefficients.size() check "
            "prevents out-of-bounds access in ufsecp_musig2_partial_sign."
        ),
    },
]


def run(json_mode: bool, out_file: str | None) -> int:
    if not MUSIG2_FILE.exists():
        msg = f"ADVISORY-SKIP: {MUSIG2_FILE.relative_to(LIB_ROOT)} not found"
        if json_mode:
            print(json.dumps({"advisory_skip": True, "reason": msg}))
        else:
            print(msg)
        return 77

    content = MUSIG2_FILE.read_text(encoding="utf-8", errors="replace")

    results: list[dict] = []
    violations: list[dict] = []

    for inv in INVARIANTS:
        body = _extract_function_body(content, inv["function"])
        if body is None:
            # Function not found in file — treat as missing invariant
            entry = {
                "id": inv["id"],
                "function": inv["function"],
                "description": inv["description"],
                "found": False,
                "body_found": False,
                "rationale": inv["rationale"],
            }
            results.append(entry)
            if inv["must_be_present"]:
                violations.append(entry)
            continue

        found = bool(inv["pattern"].search(body))
        entry = {
            "id": inv["id"],
            "function": inv["function"],
            "description": inv["description"],
            "found": found,
            "body_found": True,
            "rationale": inv["rationale"],
        }
        results.append(entry)
        if inv["must_be_present"] and not found:
            violations.append(entry)

    overall_pass = len(violations) == 0

    report = {
        "overall_pass": overall_pass,
        "source_file": str(MUSIG2_FILE.relative_to(LIB_ROOT)),
        "invariants_checked": len(INVARIANTS),
        "violations": violations,
        "violation_count": len(violations),
        "results": results,
    }

    rendered = json.dumps(report, indent=2)
    if out_file:
        Path(out_file).write_text(rendered, encoding="utf-8")

    if json_mode:
        print(rendered)
    else:
        if violations:
            print(f"  Protocol invariant violations ({len(violations)} total):")
            for v in violations:
                status = "PRESENT" if v["found"] else "MISSING"
                print(f"    [{v['id']}] {v['function']}: {v['description']} — {status}")
                if not v["body_found"]:
                    print(f"         (function body not found in source file)")
                print(f"         {v['rationale']}")
        else:
            print(f"  All {len(INVARIANTS)} protocol invariants present in"
                  f" {MUSIG2_FILE.name}")

        if overall_pass:
            print("PASS protocol-invariants gate")
        else:
            print(f"FAIL {len(violations)} protocol invariant(s) missing — see details above")

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
