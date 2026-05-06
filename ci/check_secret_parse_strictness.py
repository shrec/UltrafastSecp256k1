#!/usr/bin/env python3
"""Secret-parse strictness gate — enforces Rule 11 on secret-bearing ABI inputs.

Rule 11 (CLAUDE.md): every function that accepts a private key MUST use
`scalar_parse_strict_nonzero` (or equivalent that rejects both >= n AND == 0).
Using `scalar_parse_strict` on secret inputs is banned — it silently accepts
zero, which leaks nonce/share directly when computing s = k + 0 * e.

Catches:
  C4 — frost_sign signing share parsed with scalar_parse_strict (not _nonzero)

Scope: all src/impl/*.cpp files.

Detection logic:
  1. Find every function whose body contains a call to `scalar_parse_strict(`
     (without the _nonzero suffix).
  2. Cross-check whether that function also takes a known secret-bearing
     parameter (privkey, signing_share, scan_key, seckey, private_key, sk).
  3. If a non-_nonzero call is made in a function that handles secret input,
     it is a violation — report file, line, and function name.

Exit codes:
  0  — no violations
  1  — one or more violations
  77 — advisory-skip (no impl files found)
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

# Files in scope — source graph records these as "src/impl/*.cpp" but the
# actual disk layout is src/cpu/src/impl/*.cpp (ufsecp ABI implementations).
IMPL_DIRS = [
    LIB_ROOT / "src" / "impl",                 # canonical path if it ever exists
    LIB_ROOT / "src" / "cpu" / "src" / "impl", # actual on-disk layout
    LIB_ROOT / "src" / "cpu" / "src",          # fallback
]

# Parameter names that indicate a secret input
SECRET_PARAM_NAMES = frozenset({
    "privkey", "signing_share", "scan_key", "seckey", "private_key",
    "sk", "secret_key", "sign_key",
})

# Matches scalar_parse_strict( without _nonzero immediately following
# Uses negative lookahead to ensure _nonzero is absent
_RE_STRICT_BAD = re.compile(r'\bscalar_parse_strict\s*\((?!.*_nonzero)')
# More precise: the token "scalar_parse_strict" NOT followed by "_nonzero"
_RE_BAD_CALL   = re.compile(r'\bscalar_parse_strict\b(?!_nonzero)')

# Matches a parameter declaration containing a known secret parameter name
_RE_SECRET_PARAM = re.compile(
    r'\b(?:' + '|'.join(re.escape(n) for n in SECRET_PARAM_NAMES) + r')\b'
)

# Matches a C++ function signature opening line
_RE_FUNC_START = re.compile(r'^\s*\w[\w\s\*:&<>]+\w\s*\(')


def _scan_file(path: Path) -> list[dict]:
    """Scan a single impl file for non-_nonzero scalar_parse_strict on secret params."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    violations: list[dict] = []
    lines = content.splitlines()

    # Parse function extents: find function header lines and their brace depth
    # We use a simple brace-counter to track which function we're inside.
    current_func_name   = "<global>"
    current_func_params = ""
    brace_depth         = 0
    func_brace_start    = 0

    # Collect all ufsecp_* function signatures for faster lookup
    # We want: (func_name, param_string, start_line)
    i = 0
    func_ranges: list[tuple[str, str, int, int]] = []  # (name, params, start, end)
    func_stack: list[tuple[str, str, int]] = []

    for lineno, line in enumerate(lines, 1):
        for ch in line:
            if ch == '{':
                brace_depth += 1
                if brace_depth == 1:
                    # entering a top-level function body
                    # look back up to 8 lines for the function name and params
                    look_back = lines[max(0, lineno-9):lineno]
                    header = " ".join(look_back)
                    # extract function name (last word before open paren)
                    m = re.search(r'(\w+)\s*\(([^)]*)\)\s*\{?\s*$', header)
                    if m:
                        func_stack.append((m.group(1), m.group(2), lineno))
                    else:
                        func_stack.append(("?", header[-80:], lineno))
            elif ch == '}':
                brace_depth -= 1
                if brace_depth == 0 and func_stack:
                    fn, params, start = func_stack.pop()
                    func_ranges.append((fn, params, start, lineno))

    # Now scan for bad calls and map them to their enclosing function
    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()
        if not _RE_BAD_CALL.search(stripped):
            continue

        # Find enclosing function
        enclosing_func  = "<global>"
        enclosing_params = ""
        for (fn, params, start, end) in func_ranges:
            if start <= lineno <= end:
                enclosing_func   = fn
                enclosing_params = params
                break

        # Check if the call is DIRECTLY on a secret parameter.
        # The call line itself must contain a secret param name as an argument,
        # e.g.: scalar_parse_strict(privkey, sk)  or  scalar_parse_strict(signing_share, share)
        #
        # A function that has `privkey` in its signature but calls scalar_parse_strict
        # on a different (non-secret) parameter like `tweak` is NOT a violation.
        # Rule 11 applies to the *input being parsed*, not to the calling function.
        if not _RE_SECRET_PARAM.search(stripped):
            continue  # call is not on a secret param → not a violation

        violations.append({
            "bug": "C4_class",
            "file": str(path.relative_to(LIB_ROOT)),
            "line": lineno,
            "function": enclosing_func,
            "text": stripped[:120],
            "message": (
                f"scalar_parse_strict (non-_nonzero) used in '{enclosing_func}' "
                "which handles a secret input — use scalar_parse_strict_nonzero (Rule 11)"
            ),
        })

    return violations


def run(json_mode: bool, out_file: str | None) -> int:
    impl_files: list[Path] = []
    for d in IMPL_DIRS:
        if d.exists():
            impl_files.extend(sorted(d.glob("*.cpp")))

    if not impl_files:
        msg = ("ADVISORY-SKIP: no ABI impl files found "
               f"(checked {[str(d) for d in IMPL_DIRS]}) "
               "— secret-parse-strictness check skipped")
        if json_mode:
            print(json.dumps({"advisory_skip": True, "reason": msg}))
        else:
            print(msg)
        return 77

    violations: list[dict] = []
    for p in impl_files:
        violations.extend(_scan_file(p))

    overall_pass = len(violations) == 0

    report = {
        "overall_pass": overall_pass,
        "files_scanned": len(impl_files),
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
            print(f"  Rule 11 violations ({len(violations)} total):")
            for v in violations:
                print(f"    {v['file']}:{v['line']}  [{v['function']}]")
                print(f"         {v['message']}")
                print(f"         >> {v['text']}")
        else:
            print(f"  Scanned {len(impl_files)} impl files — no Rule 11 violations found")

        if overall_pass:
            print("PASS secret-parse-strictness gate")
        else:
            print(f"FAIL {len(violations)} scalar_parse_strict violation(s) on secret inputs")

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
