#!/usr/bin/env python3
"""
check_bitcoin_core_test_results.py
====================================
CAAS Stage 2e — Bitcoin Core test_bitcoin gate.

Validates docs/BITCOIN_CORE_TEST_RESULTS.json:
  - total  >= REQUIRED_TOTAL  (693 is the current full suite)
  - failed == 0
  - backend_commit field is present and non-empty

Exit codes:
  0 — all checks pass
  1 — one or more FAIL findings

Usage:
  python3 ci/check_bitcoin_core_test_results.py [--json]
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone

SCRIPT_DIR     = Path(__file__).resolve().parent
LIB_ROOT       = SCRIPT_DIR.parent
RESULTS_FILE   = LIB_ROOT / "docs" / "BITCOIN_CORE_TEST_RESULTS.json"
REQUIRED_TOTAL = 693
REQUIRED_FAIL  = 0


def _load_results() -> tuple[dict | None, str | None]:
    if not RESULTS_FILE.exists():
        return None, f"Results file not found: {RESULTS_FILE.relative_to(LIB_ROOT)}"
    try:
        data = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"JSON parse error in {RESULTS_FILE.name}: {exc}"
    return data, None


def check_file_exists() -> dict:
    _, err = _load_results()
    if err and "not found" in err:
        return {
            "id": "btc_results_file",
            "name": "docs/BITCOIN_CORE_TEST_RESULTS.json exists",
            "status": "FAIL",
            "detail": [err],
        }
    return {
        "id": "btc_results_file",
        "name": "docs/BITCOIN_CORE_TEST_RESULTS.json exists",
        "status": "PASS",
        "detail": [f"Found {RESULTS_FILE.relative_to(LIB_ROOT)}"],
    }


def check_zero_failures() -> dict:
    data, err = _load_results()
    if err:
        return {
            "id": "btc_zero_failures",
            "name": "Bitcoin Core test_bitcoin: 0 failures",
            "status": "FAIL",
            "detail": [err or "Could not load results"],
        }
    summary = data.get("summary", {})
    failed  = summary.get("failed", -1)
    total   = summary.get("total", 0)
    passed  = summary.get("passed", 0)

    if failed != REQUIRED_FAIL:
        return {
            "id": "btc_zero_failures",
            "name": "Bitcoin Core test_bitcoin: 0 failures",
            "status": "FAIL",
            "detail": [
                f"failed={failed} (required {REQUIRED_FAIL})",
                f"total={total}, passed={passed}",
            ],
        }
    return {
        "id": "btc_zero_failures",
        "name": "Bitcoin Core test_bitcoin: 0 failures",
        "status": "PASS",
        "detail": [f"failed=0, total={total}, passed={passed}"],
    }


def check_full_suite() -> dict:
    data, err = _load_results()
    if err:
        return {
            "id": "btc_full_suite",
            "name": f"Bitcoin Core test_bitcoin: >= {REQUIRED_TOTAL} tests",
            "status": "FAIL",
            "detail": [err or "Could not load results"],
        }
    summary = data.get("summary", {})
    total   = summary.get("total", 0)

    if total < REQUIRED_TOTAL:
        return {
            "id": "btc_full_suite",
            "name": f"Bitcoin Core test_bitcoin: >= {REQUIRED_TOTAL} tests",
            "status": "FAIL",
            "detail": [
                f"Only {total} tests recorded (required >= {REQUIRED_TOTAL})",
                "Re-run against the full Bitcoin Core test suite",
            ],
        }
    return {
        "id": "btc_full_suite",
        "name": f"Bitcoin Core test_bitcoin: >= {REQUIRED_TOTAL} tests",
        "status": "PASS",
        "detail": [f"{total} tests >= {REQUIRED_TOTAL} required"],
    }


def check_commit_present() -> dict:
    data, err = _load_results()
    if err:
        return {
            "id": "btc_commit_present",
            "name": "Bitcoin Core test results: backend_commit recorded",
            "status": "FAIL",
            "detail": [err or "Could not load results"],
        }
    commit = data.get("backend_commit", "")
    if not commit:
        return {
            "id": "btc_commit_present",
            "name": "Bitcoin Core test results: backend_commit recorded",
            "status": "FAIL",
            "detail": ["backend_commit field is empty — results are not pinned to a commit"],
        }
    return {
        "id": "btc_commit_present",
        "name": "Bitcoin Core test results: backend_commit recorded",
        "status": "PASS",
        "detail": [f"backend_commit={commit!r}"],
    }


def check_pass_rate() -> dict:
    data, err = _load_results()
    if err:
        return {
            "id": "btc_pass_rate",
            "name": "Bitcoin Core test_bitcoin: 100% pass rate",
            "status": "FAIL",
            "detail": [err or "Could not load results"],
        }
    summary = data.get("summary", {})
    total   = summary.get("total", 0)
    passed  = summary.get("passed", 0)
    failed  = summary.get("failed", -1)

    if total == 0:
        return {
            "id": "btc_pass_rate",
            "name": "Bitcoin Core test_bitcoin: 100% pass rate",
            "status": "FAIL",
            "detail": ["total=0 — no tests recorded"],
        }

    rate = passed / total * 100
    if rate < 100.0 or failed != 0:
        return {
            "id": "btc_pass_rate",
            "name": "Bitcoin Core test_bitcoin: 100% pass rate",
            "status": "FAIL",
            "detail": [f"Pass rate {rate:.1f}% ({passed}/{total}), failed={failed}"],
        }
    return {
        "id": "btc_pass_rate",
        "name": "Bitcoin Core test_bitcoin: 100% pass rate",
        "status": "PASS",
        "detail": [f"100% ({passed}/{total}), failed=0"],
    }


def check_commit_freshness() -> dict:
    """Warn if backend_commit in the results file does not match the current
    HEAD commit short SHA.  This is a warning (not FAIL) because the results
    file is updated manually when Bitcoin Core is re-run against the shim, and
    a commit bump without a re-run is expected during normal development.
    A mismatch is surfaced so a reviewer can decide whether a re-run is needed.
    """
    data, err = _load_results()
    if err:
        return {
            "id": "btc_commit_freshness",
            "name": "Bitcoin Core test results: backend_commit matches HEAD",
            "status": "WARN",
            "detail": [f"Could not load results to check freshness: {err}"],
        }

    import subprocess as _sp
    try:
        head = _sp.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(LIB_ROOT),
            stderr=_sp.DEVNULL,
        ).decode().strip()
    except Exception as exc:
        return {
            "id": "btc_commit_freshness",
            "name": "Bitcoin Core test results: backend_commit matches HEAD",
            "status": "WARN",
            "detail": [f"Could not determine HEAD commit: {exc}"],
        }

    backend_commit = data.get("backend_commit", "")
    # Compare using the shorter of the two lengths (backend_commit may be 8
    # or 40 chars; HEAD is always 40).
    cmp_len = min(len(backend_commit), len(head))
    if cmp_len < 7:
        return {
            "id": "btc_commit_freshness",
            "name": "Bitcoin Core test results: backend_commit matches HEAD",
            "status": "WARN",
            "detail": [f"backend_commit is too short to compare: {backend_commit!r}"],
        }

    if head[:cmp_len] != backend_commit[:cmp_len]:
        return {
            "id": "btc_commit_freshness",
            "name": "Bitcoin Core test results: backend_commit matches HEAD",
            "status": "WARN",
            "detail": [
                f"backend_commit={backend_commit!r} does not match HEAD={head[:8]!r}",
                "Re-run Bitcoin Core test_bitcoin against the current shim to refresh.",
            ],
        }

    return {
        "id": "btc_commit_freshness",
        "name": "Bitcoin Core test results: backend_commit matches HEAD",
        "status": "PASS",
        "detail": [f"backend_commit={backend_commit!r} matches HEAD={head[:8]!r}"],
    }


ALL_CHECKS = [
    check_file_exists,
    check_zero_failures,
    check_full_suite,
    check_commit_present,
    check_pass_rate,
]

# ---------------------------------------------------------------------------
# Reporting (matches style of check_core_build_mode.py)
# ---------------------------------------------------------------------------

COL_RESET  = "\033[0m"
COL_GREEN  = "\033[32m"
COL_RED    = "\033[31m"
COL_YELLOW = "\033[33m"
COL_CYAN   = "\033[36m"
COL_BOLD   = "\033[1m"

STATUS_WIDTH = 5
NAME_WIDTH   = 70


def _colorize(text: str, color: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{color}{text}{COL_RESET}"


def print_table(results: list[dict]) -> None:
    header    = f"{'STATUS':<{STATUS_WIDTH}}  {'CHECK'}"
    separator = "-" * (STATUS_WIDTH + 2 + NAME_WIDTH)

    print()
    print(_colorize(header, COL_BOLD))
    print(separator)

    for r in results:
        status = r["status"]
        if status == "PASS":
            colored_status = _colorize(f"{'PASS':<{STATUS_WIDTH}}", COL_GREEN)
        elif status == "WARN":
            colored_status = _colorize(f"{'WARN':<{STATUS_WIDTH}}", COL_YELLOW)
        else:
            colored_status = _colorize(f"{'FAIL':<{STATUS_WIDTH}}", COL_RED)

        print(f"{colored_status}  {r['name']}")
        for line in r["detail"]:
            indent_color = COL_CYAN if status == "PASS" else COL_YELLOW if status == "WARN" else COL_RED
            print(f"         {_colorize('>', indent_color)} {line}")

    print(separator)

    passes = sum(1 for r in results if r["status"] == "PASS")
    warns  = sum(1 for r in results if r["status"] == "WARN")
    fails  = sum(1 for r in results if r["status"] == "FAIL")
    total  = len(results)

    parts = [f"{total} checks"]
    if passes:
        parts.append(_colorize(f"{passes} passed", COL_GREEN))
    if warns:
        parts.append(_colorize(f"{warns} warnings", COL_YELLOW))
    if fails:
        parts.append(_colorize(f"{fails} failed", COL_RED))

    print("  " + ", ".join(parts))
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "CAAS Stage 2e: validate docs/BITCOIN_CORE_TEST_RESULTS.json "
            "against the required 693/693 Bitcoin Core test_bitcoin baseline."
        )
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    results = [fn() for fn in ALL_CHECKS]

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"\nCAAS Stage 2e — Bitcoin Core test_bitcoin gate — {LIB_ROOT}")
        print_table(results)

    return 1 if any(r["status"] == "FAIL" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
