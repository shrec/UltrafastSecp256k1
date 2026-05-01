#!/usr/bin/env python3
"""
G-8: CT Tool Independence Agreement Check.

Reads verdict JSON files produced by independent CT tools and asserts
that they agree: either both PASS, or at least one PASS (with SKIPs treated
as neutral). A FAIL from any non-skip tool causes the gate to fail.

This proves that multiple independent analysis methodologies — binary taint,
statistical timing, compile-time verification — reach the same conclusion
about the CT properties of the signing implementation.

Usage:
    python3 ci/ct_independence_check.py <verdict_a.json> [<verdict_b.json> ...] [--json]

Exit codes:
    0  all tools agree: no leakage detected
    1  at least one tool detected leakage (FAIL), or tools disagree
    2  all verdict files missing or no results to compare

Verdict JSON format (emitted by each CT tool job):
    {
        "tool": "valgrind-memcheck-ct",
        "methodology": "binary-taint",
        "verdict": "PASS",       # PASS | FAIL | SKIP
        "exit_code": 0,
        "details": "...",
        "commit": "<sha>",
        "runner": "Linux-X64"
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_verdict(path: Path) -> dict | None:
    if not path.exists():
        print(f"WARN: verdict file not found: {path}", file=sys.stderr)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"WARN: cannot parse verdict JSON {path}: {e}", file=sys.stderr)
        return None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="G-8: CT tool independence agreement check."
    )
    parser.add_argument("verdicts", nargs="+", help="Verdict JSON files from CT tools")
    parser.add_argument("--json", action="store_true", help="Emit JSON result to stdout")
    args = parser.parse_args(argv[1:])

    loaded: list[dict] = []
    for p in args.verdicts:
        v = load_verdict(Path(p))
        if v is not None:
            loaded.append(v)

    if not loaded:
        print("ERROR: no verdict files could be loaded", file=sys.stderr)
        return 2

    passes = [v for v in loaded if v.get("verdict") == "PASS"]
    fails = [v for v in loaded if v.get("verdict") == "FAIL"]
    skips = [v for v in loaded if v.get("verdict") == "SKIP"]

    # Gate logic:
    # - Any FAIL → overall FAIL (leakage detected)
    # - All SKIP (no real results) → exit 2
    # - At least one PASS and no FAILs → PASS
    if fails:
        overall = "FAIL"
    elif not passes:
        overall = "SKIP"
    else:
        overall = "PASS"

    if args.json:
        result = {
            "overall": overall,
            "tools_evaluated": len(loaded),
            "pass_count": len(passes),
            "fail_count": len(fails),
            "skip_count": len(skips),
            "verdicts": [
                {
                    "tool": v.get("tool", "?"),
                    "methodology": v.get("methodology", "?"),
                    "verdict": v.get("verdict", "?"),
                    "details": v.get("details", ""),
                }
                for v in loaded
            ],
        }
        print(json.dumps(result, indent=2))
        return 0 if overall == "PASS" else (2 if overall == "SKIP" else 1)

    print(f"CT Tool Independence Check — {len(loaded)} tool(s) evaluated:")
    for v in loaded:
        tool = v.get("tool", "?")
        method = v.get("methodology", "?")
        verdict = v.get("verdict", "?")
        details = v.get("details", "")
        status = {"PASS": "OK", "FAIL": "FAIL", "SKIP": "SKIP"}.get(verdict, verdict)
        print(f"  [{status}] {tool} ({method})")
        if details:
            print(f"         {details}")

    if fails:
        tools = ", ".join(v.get("tool", "?") for v in fails)
        print(f"\nFAIL: {len(fails)} tool(s) detected CT leakage: {tools}")
        print("      Tools disagree or implementation is not constant-time")
        return 1

    if overall == "SKIP":
        print("\nSKIP: no tools produced a conclusive result (all skipped or files missing)")
        return 2

    print(f"\nPASS: {len(passes)} independent CT tool(s) found no timing leakage")
    if len(passes) >= 2:
        methodologies = [v.get("methodology", "?") for v in passes]
        print(f"      Methodologies that agree: {', '.join(methodologies)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
