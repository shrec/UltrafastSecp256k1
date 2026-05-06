#!/usr/bin/env python3
"""Nonce erase coverage gate — verifies RAII scope guards for secret nonce buffers.

Catches:
  C6 — secnonce not zeroed on all error paths in ufsecp_musig2_partial_sign (BIP-327)
  C7 — nonce not zeroed on all error paths in ufsecp_frost_sign (FROST nonce reuse)

The correct pattern is to place a ScopeSecureErase<uint8_t> guard immediately
after the null-check (so it fires on every subsequent exit — success, error,
and exception).  Functions that manually call secure_erase only on the happy
path are insufficient.

Detection logic:
  1. For every function in src/impl/*.cpp that has a mutable uint8_t nonce or
     secnonce buffer parameter, verify that:
       a. A ScopeSecureErase guard is declared for that buffer, OR
       b. secure_erase is called before EVERY `return` statement in the body.
  2. If neither condition holds, the function is a violation.

Exit codes:
  0  — all mutable nonce parameters have scope-guard coverage
  1  — one or more functions lack coverage
  77 — advisory-skip (src/impl/ not found)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT   = SCRIPT_DIR.parent

# Files to scan — source graph records these as src/impl/*.cpp but the
# actual on-disk layout is src/cpu/src/impl/*.cpp.
_IMPL_DIRS = [
    LIB_ROOT / "src" / "impl",
    LIB_ROOT / "src" / "cpu" / "src" / "impl",
    LIB_ROOT / "src" / "cpu" / "src",
]
IMPL_DIR = next((d for d in _IMPL_DIRS if d.exists()), _IMPL_DIRS[0])

# Parameter declarations we care about: MUTABLE (non-const) uint8_t nonce/secnonce buffers.
# Must NOT be preceded by `const` — const nonce params (e.g. ChaCha20 IVs) are public
# and do not need zeroing.  Only writable signing/FROST/MuSig2 nonce buffers matter.
#
# Matches:
#   uint8_t nonce[UFSECP_FROST_NONCE_LEN]
#   uint8_t secnonce[UFSECP_MUSIG2_SECNONCE_LEN]
# Does NOT match:
#   const uint8_t nonce[12]     (ChaCha20 IV — public, non-secret)
_RE_NONCE_PARAM = re.compile(
    r'(?<!const\s)(?<!const )uint8_t\s+(?P<name>nonce|secnonce)\s*\['
)

# More reliable: require that the nonce param appears WITHOUT a preceding `const`
# on the same parameter entry.  We apply this as a post-filter on matched params.
def _is_mutable_nonce_param(param_text: str, name: str) -> bool:
    """Return True if `name` appears as a non-const uint8_t array param."""
    # Split param text into individual parameters (comma-separated, roughly)
    # and check each one
    for segment in re.split(r',\s*', param_text):
        # Does this segment declare our nonce name?
        if not re.search(r'\b' + re.escape(name) + r'\b', segment):
            continue
        # Is it const?
        if re.search(r'\bconst\b', segment):
            return False  # const → not mutable, skip
        # Is it uint8_t?
        if re.search(r'\buint8_t\b', segment):
            return True
    return False

# ScopeSecureErase guard for the nonce: ScopeSecureErase<uint8_t> {name}_guard(...)
_RE_SCOPE_GUARD = re.compile(r'\bScopeSecureErase\b')

# A return statement (any)
_RE_RETURN = re.compile(r'\breturn\b')

# A secure_erase call that mentions the nonce variable
def _re_erase_for(name: str) -> re.Pattern:
    return re.compile(r'\bsecure_erase\s*\([^;]*\b' + re.escape(name) + r'\b')


def _extract_top_level_functions(content: str) -> list[tuple[str, str, list[str]]]:
    """
    Return list of (func_name, param_string, body_lines) for each top-level
    function in the file.  Uses a simple brace-depth counter.
    """
    lines     = content.splitlines()
    functions = []

    brace_depth  = 0
    func_name    = None
    func_params  = None
    body_start   = None
    body_lines: list[str] = []

    # Regex to detect a function definition opening:
    # ReturnType func_name(params...) {   — must match at brace_depth == 0
    _RE_FUNC_DEF = re.compile(
        r'^\s*\w[\w\s:*&<>,]*\b(\w+)\s*\(([^)]*)\)\s*(?:noexcept\s*)?\{'
    )
    # Multi-line version: signature line that ends without a brace
    _RE_SIG_LINE = re.compile(
        r'^\s*(?:ufsecp_error_t|void|int|bool|auto)\s+(\w+)\s*\('
    )

    pending_sig_name: str | None = None
    pending_sig_params: str = ""
    pending_sig_lines: list[str] = []
    PENDING_MAX = 12

    i = 0
    while i < len(lines):
        line = lines[i]

        if brace_depth == 0:
            # Single-line function definition
            m = _RE_FUNC_DEF.search(line)
            if m and brace_depth == 0:
                # Count the brace depth change in this line
                for ch in line:
                    if ch == '{':
                        brace_depth += 1
                    elif ch == '}':
                        brace_depth -= 1
                func_name   = m.group(1)
                func_params = m.group(2)
                body_start  = i
                body_lines  = [line]
                i += 1
                continue

            # Multi-line: signature opens but brace on later line
            ms = _RE_SIG_LINE.match(line)
            if ms:
                pending_sig_name   = ms.group(1)
                pending_sig_lines  = [line]
                pending_sig_params = ""
                # Accumulate param text
                m2 = re.search(r'\((.*)$', line)
                if m2:
                    pending_sig_params += m2.group(1) + " "
                i += 1
                continue

            if pending_sig_name and len(pending_sig_lines) < PENDING_MAX:
                pending_sig_lines.append(line)
                # Collect params until we see ')'
                if ')' in line and '{' in line:
                    # Opening brace found — function starts here
                    m3 = re.search(r'\)([^)]*)\{', line)
                    for ch in line:
                        if ch == '{':
                            brace_depth += 1
                        elif ch == '}':
                            brace_depth -= 1
                    func_name   = pending_sig_name
                    # Build param string from accumulated lines
                    full_header = " ".join(pending_sig_lines)
                    m4 = re.search(r'\(([^)]*)\)', full_header)
                    func_params = m4.group(1) if m4 else ""
                    body_start  = i
                    body_lines  = [line]
                    pending_sig_name = None
                    i += 1
                    continue
                elif ')' in line:
                    m2 = re.match(r'([^)]*)\)', line)
                    if m2:
                        pending_sig_params += m2.group(1)
                elif '{' in line and ')' not in line:
                    # orphan brace — drop pending
                    pending_sig_name = None
                else:
                    # Accumulate params
                    m2 = re.search(r'(?<=\()(.*)$', line)
                    if m2:
                        pending_sig_params += m2.group(1) + " "
            else:
                pending_sig_name = None

        else:
            # Inside function body
            if func_name is not None:
                body_lines.append(line)
            for ch in line:
                if ch == '{':
                    brace_depth += 1
                elif ch == '}':
                    brace_depth -= 1
            if brace_depth == 0 and func_name is not None:
                functions.append((func_name, func_params or "", body_lines[:]))
                func_name   = None
                func_params = None
                body_start  = None
                body_lines  = []

        i += 1

    return functions


def _check_nonce_erase(func_name: str, params: str, body_lines: list[str]) -> list[dict]:
    """
    Check that every mutable nonce/secnonce parameter has a scope guard or
    a secure_erase before every return in the function body.
    """
    violations = []

    # Find which nonce params this function has — mutable only
    raw_nonce_params = _RE_NONCE_PARAM.findall(params)
    nonce_params = [n for n in raw_nonce_params if _is_mutable_nonce_param(params, n)]
    if not nonce_params:
        return []

    body = "\n".join(body_lines)

    for name in nonce_params:
        # Check 1: does the body have a ScopeSecureErase covering this nonce?
        has_scope_guard = bool(_RE_SCOPE_GUARD.search(body))
        if has_scope_guard:
            # Additionally verify the guard is for THIS specific nonce
            # Pattern: ScopeSecureErase<uint8_t> {name}_guard({name}, ...
            specific_guard = re.search(
                r'ScopeSecureErase\s*<\s*uint8_t\s*>\s+\w*\s*\(\s*' + re.escape(name),
                body,
            )
            if specific_guard:
                continue  # covered by scope guard — pass

        # Check 2: secure_erase called before every return
        re_erase = _re_erase_for(name)
        return_positions   = [i for i, ln in enumerate(body_lines) if _RE_RETURN.search(ln)]
        erase_positions    = [i for i, ln in enumerate(body_lines) if re_erase.search(ln)]

        if not return_positions:
            continue  # no returns — no issue

        # For each return, there must be an erase somewhere before it
        uncovered_returns = []
        for ret_pos in return_positions:
            # null-check returns at the top (before any nonce is constructed) are OK
            # heuristic: if the return is within the first 5 lines of the body, skip
            if ret_pos < 5:
                continue
            has_prior_erase = any(ep < ret_pos for ep in erase_positions)
            if not has_prior_erase:
                uncovered_returns.append(ret_pos + 1)  # 1-based for display

        if uncovered_returns:
            violations.append({
                "bug": "C6_C7_class",
                "function": func_name,
                "nonce_param": name,
                "uncovered_return_lines_in_body": uncovered_returns[:5],
                "has_scope_guard": has_scope_guard,
                "message": (
                    f"'{name}' in '{func_name}' lacks ScopeSecureErase scope guard; "
                    f"secure_erase missing before {len(uncovered_returns)} return(s) "
                    "(nonce reuse possible on error path)"
                ),
            })

    return violations


def _scan_file(path: Path) -> list[dict]:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    rel = str(path.relative_to(LIB_ROOT))
    functions = _extract_top_level_functions(content)
    violations = []
    for func_name, params, body_lines in functions:
        for v in _check_nonce_erase(func_name, params, body_lines):
            v["file"] = rel
            violations.append(v)
    return violations


def run(json_mode: bool, out_file: str | None) -> int:
    impl_files: list[Path] = []
    for d in _IMPL_DIRS:
        if d.exists():
            impl_files.extend(sorted(d.glob("*.cpp")))

    if not impl_files:
        msg = ("ADVISORY-SKIP: no ABI impl files found "
               f"(checked {[str(d) for d in _IMPL_DIRS]}) "
               "— nonce-erase coverage check skipped")
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
            print(f"  Nonce erase coverage violations ({len(violations)} total):")
            for v in violations:
                print(f"    {v['file']}  [{v['function']}]  param={v['nonce_param']}")
                print(f"         {v['message']}")
        else:
            print(f"  Scanned {len(impl_files)} impl files — all nonce params have scope-guard coverage")

        if overall_pass:
            print("PASS nonce-erase-coverage gate")
        else:
            print(f"FAIL {len(violations)} nonce erase coverage violation(s) — see details above")

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
