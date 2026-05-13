#!/usr/bin/env python3
"""
check_libsecp_shim_parity.py
============================
Static analysis script that verifies the libsecp256k1 shim source files use
strict parse functions at all external input boundaries — ensuring the shim
cannot silently accept inputs that libsecp would reject.

Exit codes:
  0 — all checks pass
  1 — one or more parity violations found

Usage:
  python3 check_libsecp_shim_parity.py [--json] [--function <name>]
"""

import sys
import os
import re
import json
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Path resolution — library root is parent of ci/
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT   = SCRIPT_DIR.parent
SHIM_SRC   = LIB_ROOT / "compat" / "libsecp256k1_shim" / "src"

# ---------------------------------------------------------------------------
# Check registry
# Each entry describes one function-level parity check.
#
# Fields:
#   file        — shim .cpp filename (relative to SHIM_SRC)
#   function    — C function name (used for reporting and --function filter)
#   description — human-readable intent
#   checks      — list of dicts, each with:
#       kind    — "require" | "forbid"
#       pattern — regex searched within the function's source text
#       message — what to say if the check fails
# ---------------------------------------------------------------------------

CHECKS = [
    # ---- shim_seckey.cpp -----------------------------------------------
    {
        "file": "shim_seckey.cpp",
        "function": "secp256k1_ec_seckey_verify",
        "description": "seckey_verify must reject non-canonical / zero secret keys",
        "checks": [
            {
                "kind": "require",
                "pattern": r"Scalar::parse_bytes_strict_nonzero",
                "message": "must use Scalar::parse_bytes_strict_nonzero (not from_bytes) for the seckey parameter",
            },
            {
                "kind": "forbid",
                "pattern": r"Scalar::from_bytes",
                "message": "must not use Scalar::from_bytes at an external boundary (does not reject overflow or zero)",
            },
        ],
    },
    {
        "file": "shim_seckey.cpp",
        "function": "secp256k1_ec_seckey_negate",
        "description": "seckey_negate must reject non-canonical / zero secret keys",
        "checks": [
            {
                "kind": "require",
                "pattern": r"Scalar::parse_bytes_strict_nonzero",
                "message": "must use Scalar::parse_bytes_strict_nonzero for the seckey parameter",
            },
            {
                "kind": "forbid",
                "pattern": r"Scalar::from_bytes",
                "message": "must not use Scalar::from_bytes at an external boundary",
            },
        ],
    },
    {
        "file": "shim_seckey.cpp",
        "function": "secp256k1_ec_seckey_tweak_add",
        "description": "seckey_tweak_add must strictly validate seckey and tweak",
        "checks": [
            {
                "kind": "require",
                "pattern": r"Scalar::parse_bytes_strict_nonzero",
                "message": "must use Scalar::parse_bytes_strict_nonzero for the seckey parameter",
            },
            {
                "kind": "require",
                "pattern": r"Scalar::parse_bytes_strict",
                "message": "must use Scalar::parse_bytes_strict (or _nonzero) for the tweak parameter",
            },
            {
                "kind": "forbid",
                "pattern": r"Scalar::from_bytes",
                "message": "must not use Scalar::from_bytes at an external boundary",
            },
        ],
    },
    {
        "file": "shim_seckey.cpp",
        "function": "secp256k1_ec_seckey_tweak_mul",
        "description": "seckey_tweak_mul must strictly validate seckey and tweak",
        "checks": [
            {
                "kind": "require",
                "pattern": r"Scalar::parse_bytes_strict_nonzero",
                "message": "must use Scalar::parse_bytes_strict_nonzero for both seckey and tweak",
            },
            {
                "kind": "forbid",
                "pattern": r"Scalar::from_bytes",
                "message": "must not use Scalar::from_bytes at an external boundary",
            },
        ],
    },

    # ---- shim_pubkey.cpp -----------------------------------------------
    {
        "file": "shim_pubkey.cpp",
        "function": "secp256k1_ec_pubkey_parse",
        "description": "pubkey_parse must reject out-of-range coordinates and points not on curve",
        "checks": [
            {
                "kind": "require",
                "pattern": r"FieldElement::parse_bytes_strict",
                "message": "must use FieldElement::parse_bytes_strict for coordinate parsing (rejects x >= p, y >= p)",
            },
            {
                "kind": "require",
                "pattern": r"y\.square\(\)\s*==\s*y2|y2\s*==\s*y\.square\(\)|rhs|y\^2",
                "message": "must verify the point is on the curve (y^2 == x^3 + 7 check)",
            },
        ],
    },
    {
        "file": "shim_pubkey.cpp",
        "function": "secp256k1_ec_pubkey_create",
        "description": "pubkey_create must reject invalid secret keys",
        "checks": [
            {
                "kind": "require",
                "pattern": r"Scalar::parse_bytes_strict_nonzero",
                "message": "must use Scalar::parse_bytes_strict_nonzero for the seckey parameter",
            },
            {
                "kind": "forbid",
                "pattern": r"Scalar::from_bytes",
                "message": "must not use Scalar::from_bytes at an external boundary",
            },
        ],
    },
    {
        "file": "shim_pubkey.cpp",
        "function": "secp256k1_ec_pubkey_tweak_add",
        "description": "pubkey_tweak_add must strictly validate the tweak scalar",
        "checks": [
            {
                "kind": "require",
                "pattern": r"Scalar::parse_bytes_strict",
                "message": "must use Scalar::parse_bytes_strict (or _nonzero) for the tweak parameter",
            },
            {
                "kind": "forbid",
                "pattern": r"Scalar::from_bytes",
                "message": "must not use Scalar::from_bytes at an external boundary",
            },
        ],
    },
    {
        "file": "shim_pubkey.cpp",
        "function": "secp256k1_ec_pubkey_tweak_mul",
        "description": "pubkey_tweak_mul must strictly validate the tweak scalar",
        "checks": [
            {
                "kind": "require",
                "pattern": r"Scalar::parse_bytes_strict_nonzero",
                "message": "must use Scalar::parse_bytes_strict_nonzero for the tweak parameter",
            },
            {
                "kind": "forbid",
                "pattern": r"Scalar::from_bytes",
                "message": "must not use Scalar::from_bytes at an external boundary",
            },
        ],
    },

    # ---- shim_ecdsa.cpp -----------------------------------------------
    {
        "file": "shim_ecdsa.cpp",
        "function": "secp256k1_ecdsa_signature_parse_compact",
        "description": "parse_compact must validate r and s (range check) before storing into sig->data",
        "checks": [
            {
                "kind": "require",
                # libsecp accepts r=0/s=0 at parse time (verify rejects them).
                # Only r>=n or s>=n triggers a parse failure.
                # parse_bytes_strict allows zero but rejects overflow; that matches libsecp.
                # parse_bytes_strict_nonzero is also acceptable (stricter but compatible
                # because no caller passes r=0/s=0 in practice on the signing side).
                "pattern": r"Scalar::parse_bytes_strict",
                "message": "must use Scalar::parse_bytes_strict (or strict_nonzero) for both r and s (rejects overflow)",
            },
            # A bare memcpy as the SOLE operation — with no parse anywhere in
            # the body — would be detected by the absence of the "require" above.
            # We add one more check: the function must NOT have a memcpy to
            # sig->data as an EARLY return path with zero parse calls before it.
            # Concretely: if parse_bytes_strict_nonzero is absent from the body
            # AND memcpy to sig->data is present, that is a bare-copy bug.
            # Since the "require" check already rejects the absence case, we
            # add a structural check: parse must appear BEFORE the memcpy.
            # Approximation: the string "parse_bytes_strict" must appear
            # somewhere in the body text before "memcpy(sig->data".
            {
                "kind": "require",
                "pattern": r"(?s)parse_bytes_strict.*memcpy\(sig->data",
                "message": (
                    "Scalar::parse_bytes_strict (or strict_nonzero) must appear before "
                    "the memcpy(sig->data, input64, 64) store — a bare memcpy "
                    "without prior validation is a libsecp parity violation"
                ),
            },
        ],
    },
    {
        "file": "shim_ecdsa.cpp",
        "function": "secp256k1_ecdsa_sign",
        "description": "ecdsa_sign must reject invalid secret keys",
        "checks": [
            {
                "kind": "require",
                "pattern": r"Scalar::parse_bytes_strict_nonzero",
                "message": "must use Scalar::parse_bytes_strict_nonzero for the seckey parameter",
            },
            {
                "kind": "forbid",
                "pattern": r"Scalar::from_bytes",
                "message": "must not use Scalar::from_bytes at an external boundary",
            },
        ],
    },

    # ---- shim_schnorr.cpp ---------------------------------------------
    {
        "file": "shim_schnorr.cpp",
        "function": "secp256k1_schnorrsig_sign32",
        "description": "schnorr_sign must reject invalid secret keys",
        "checks": [
            {
                "kind": "require",
                "pattern": r"Scalar::parse_bytes_strict_nonzero",
                "message": "must use Scalar::parse_bytes_strict_nonzero for the keypair secret key",
            },
            {
                "kind": "forbid",
                "pattern": r"Scalar::from_bytes",
                "message": "must not use Scalar::from_bytes at an external boundary",
            },
        ],
    },

    # ---- shim_extrakeys.cpp -------------------------------------------
    {
        "file": "shim_extrakeys.cpp",
        "function": "secp256k1_xonly_pubkey_parse",
        "description": "xonly_pubkey_parse must reject x >= p and x not a valid curve coordinate",
        "checks": [
            {
                "kind": "require",
                "pattern": r"FieldElement::parse_bytes_strict",
                "message": "must use FieldElement::parse_bytes_strict for the x-coordinate (rejects x >= p)",
            },
            {
                "kind": "require",
                "pattern": r"y\.square\(\)\s*==\s*y2|y2\s*==\s*y\.square\(\)",
                "message": "must verify x is a valid curve x-coordinate with y.square() == y2 check",
            },
        ],
    },
    {
        "file": "shim_extrakeys.cpp",
        "function": "secp256k1_keypair_create",
        "description": "keypair_create must reject invalid secret keys",
        "checks": [
            {
                "kind": "require",
                "pattern": r"Scalar::parse_bytes_strict_nonzero",
                "message": "must use Scalar::parse_bytes_strict_nonzero for the seckey parameter",
            },
            {
                "kind": "forbid",
                "pattern": r"Scalar::from_bytes",
                "message": "must not use Scalar::from_bytes at an external boundary",
            },
        ],
    },
    {
        "file": "shim_extrakeys.cpp",
        "function": "secp256k1_keypair_xonly_tweak_add",
        "description": "keypair_xonly_tweak_add must validate the secret key and tweak",
        "checks": [
            {
                "kind": "require",
                "pattern": r"Scalar::parse_bytes_strict_nonzero",
                "message": "must use Scalar::parse_bytes_strict_nonzero for the keypair secret key",
            },
            {
                "kind": "require",
                "pattern": r"Scalar::parse_bytes_strict",
                "message": "must use Scalar::parse_bytes_strict (or _nonzero) for the tweak",
            },
        ],
    },
    {
        "file": "shim_extrakeys.cpp",
        "function": "secp256k1_xonly_pubkey_tweak_add",
        "description": "xonly_pubkey_tweak_add must validate the tweak scalar",
        "checks": [
            {
                "kind": "require",
                "pattern": r"Scalar::parse_bytes_strict",
                "message": "must use Scalar::parse_bytes_strict (or _nonzero) for the tweak",
            },
            {
                "kind": "forbid",
                "pattern": r"Scalar::from_bytes",
                "message": "must not use Scalar::from_bytes at an external boundary",
            },
        ],
    },
    # ---- shim_ecdh.cpp -----------------------------------------------
    # P1-SEC-NEW-001 (2026-05-13 v8 audit): secp256k1_ecdh now uses
    # parse_bytes_strict_nonzero per CLAUDE.md Rule 11 (every function
    # accepting a private key byte array MUST use strict parsing). The
    # behavioral divergence from upstream libsecp256k1 (which silently
    # reduces mod n) is intentional and documented in
    # docs/SHIM_KNOWN_DIVERGENCES.md ("secp256k1_ecdh — private key >= curve
    # order rejected"). This check now ENFORCES the strict parsing so a
    # regression to from_bytes is caught at the parity gate.
    {
        "file": "shim_ecdh.cpp",
        "function": "secp256k1_ecdh",
        "description": "ECDH uses parse_bytes_strict_nonzero (Rule 11) — intentional divergence from libsecp documented in SHIM_KNOWN_DIVERGENCES.md",
        "checks": [
            {
                "kind": "require",
                "pattern": r"parse_bytes_strict_nonzero",
                "message": "ECDH must use parse_bytes_strict_nonzero per CLAUDE.md Rule 11 (P1-SEC-NEW-001)",
            },
            {
                "kind": "forbid",
                "pattern": r"Scalar::from_bytes\s*\(",
                "message": "ECDH must NOT regress to Scalar::from_bytes (silent mod-n reduction violates Rule 11)",
            },
        ],
    },

    {
        "file": "shim_extrakeys.cpp",
        "function": "secp256k1_xonly_pubkey_tweak_add_check",
        "description": "xonly_pubkey_tweak_add_check must validate the tweak scalar",
        "checks": [
            {
                "kind": "require",
                "pattern": r"Scalar::parse_bytes_strict",
                "message": "must use Scalar::parse_bytes_strict (or _nonzero) for the tweak",
            },
            {
                "kind": "forbid",
                "pattern": r"Scalar::from_bytes",
                "message": "must not use Scalar::from_bytes at an external boundary",
            },
        ],
    },
]

# ---------------------------------------------------------------------------
# Source file cache
# ---------------------------------------------------------------------------

_source_cache: dict[str, str] = {}


def load_source(filename: str) -> str | None:
    if filename in _source_cache:
        return _source_cache[filename]
    path = SHIM_SRC / filename
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    _source_cache[filename] = text
    return text


# ---------------------------------------------------------------------------
# Function body extractor
#
# Strategy: find the function signature line, then walk forward counting
# braces until the outer brace closes. This handles nested lambdas/blocks.
# ---------------------------------------------------------------------------

def extract_function_body(source: str, func_name: str) -> str | None:
    """
    Return the text of the C++ function body (including braces) for the first
    occurrence of a function whose name matches func_name. Returns None if not
    found.
    """
    # Match the function name followed (on the same or adjacent line) by '('
    # This is intentionally permissive — we just need a rough position.
    pattern = re.compile(
        r'\b' + re.escape(func_name) + r'\s*\n?\s*\('
    )
    m = pattern.search(source)
    if m is None:
        return None

    # Find the opening brace of the function body after the match position.
    start_pos = m.start()
    brace_pos = source.find('{', start_pos)
    if brace_pos == -1:
        return None

    depth = 0
    end_pos = brace_pos
    for i, ch in enumerate(source[brace_pos:], start=brace_pos):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end_pos = i
                break
    else:
        # Never closed — return what we found (malformed source)
        return source[brace_pos:]

    return source[brace_pos : end_pos + 1]


# ---------------------------------------------------------------------------
# Run a single check entry
# Returns a result dict.
# ---------------------------------------------------------------------------

def run_check(entry: dict) -> dict:
    filename  = entry["file"]
    func_name = entry["function"]
    source    = load_source(filename)

    result = {
        "file":        filename,
        "function":    func_name,
        "description": entry["description"],
        "status":      "PASS",
        "violations":  [],
        "file_found":  True,
        "func_found":  True,
    }

    if source is None:
        result["status"]     = "ERROR"
        result["file_found"] = False
        result["violations"] = [f"Source file not found: {SHIM_SRC / filename}"]
        return result

    body = extract_function_body(source, func_name)
    if body is None:
        result["status"]     = "ERROR"
        result["func_found"] = False
        result["violations"] = [f"Function body not found in {filename}"]
        return result

    for chk in entry["checks"]:
        pat     = re.compile(chk["pattern"], re.DOTALL)
        found   = pat.search(body) is not None
        kind    = chk["kind"]
        message = chk["message"]

        if kind == "require" and not found:
            result["violations"].append(f"MISSING: {message}")
        elif kind == "forbid" and found:
            result["violations"].append(f"FORBIDDEN pattern found: {message}")

    if result["violations"]:
        result["status"] = "FAIL"

    return result


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

STATUS_WIDTH  = 5
FUNC_WIDTH    = 50
FILE_WIDTH    = 22

COL_RESET  = "\033[0m"
COL_GREEN  = "\033[32m"
COL_RED    = "\033[31m"
COL_YELLOW = "\033[33m"
COL_CYAN   = "\033[36m"
COL_BOLD   = "\033[1m"


def _colorize(text: str, color: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{color}{text}{COL_RESET}"


def print_table(results: list[dict]) -> None:
    header = (
        f"{'STATUS':<{STATUS_WIDTH}}  "
        f"{'FILE':<{FILE_WIDTH}}  "
        f"{'FUNCTION':<{FUNC_WIDTH}}"
    )
    separator = "-" * (STATUS_WIDTH + 2 + FILE_WIDTH + 2 + FUNC_WIDTH)

    print()
    print(_colorize(header, COL_BOLD))
    print(separator)

    for r in results:
        status = r["status"]
        if status == "PASS":
            colored_status = _colorize(f"{'PASS':<{STATUS_WIDTH}}", COL_GREEN)
        elif status == "FAIL":
            colored_status = _colorize(f"{'FAIL':<{STATUS_WIDTH}}", COL_RED)
        else:
            colored_status = _colorize(f"{'ERROR':<{STATUS_WIDTH}}", COL_YELLOW)

        func_col = r["function"]
        file_col = r["file"]

        print(f"{colored_status}  {file_col:<{FILE_WIDTH}}  {func_col:<{FUNC_WIDTH}}")
        for v in r["violations"]:
            print(f"         {_colorize('>', COL_CYAN)} {v}")

    print(separator)

    passes   = sum(1 for r in results if r["status"] == "PASS")
    fails    = sum(1 for r in results if r["status"] == "FAIL")
    errors   = sum(1 for r in results if r["status"] == "ERROR")
    total    = len(results)

    summary_parts = [f"{total} checks"]
    if passes:
        summary_parts.append(_colorize(f"{passes} passed", COL_GREEN))
    if fails:
        summary_parts.append(_colorize(f"{fails} failed", COL_RED))
    if errors:
        summary_parts.append(_colorize(f"{errors} errors", COL_YELLOW))

    print("  " + ", ".join(summary_parts))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify libsecp256k1 shim uses strict parse functions at all input boundaries."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of a table",
    )
    parser.add_argument(
        "--function",
        metavar="NAME",
        help="Check only the function matching NAME",
    )
    args = parser.parse_args()

    # Verify shim src directory exists
    if not SHIM_SRC.exists():
        msg = f"Shim source directory not found: {SHIM_SRC}"
        if args.json:
            print(json.dumps({"error": msg, "results": []}, indent=2))
        else:
            print(f"ERROR: {msg}", file=sys.stderr)
        return 1

    # Filter checks if --function supplied
    checks_to_run = CHECKS
    if args.function:
        checks_to_run = [c for c in CHECKS if c["function"] == args.function]
        if not checks_to_run:
            available = sorted({c["function"] for c in CHECKS})
            msg = (
                f"No check registered for function '{args.function}'. "
                f"Available: {', '.join(available)}"
            )
            if args.json:
                print(json.dumps({"error": msg, "results": []}, indent=2))
            else:
                print(f"ERROR: {msg}", file=sys.stderr)
            return 1

    results = [run_check(entry) for entry in checks_to_run]

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        shim_path_display = str(SHIM_SRC.relative_to(LIB_ROOT))
        print(f"\nLibsecp256k1 shim parity check — scanning: {shim_path_display}")
        print_table(results)

    any_fail = any(r["status"] in ("FAIL", "ERROR") for r in results)
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
