#!/usr/bin/env python3
"""
import_bitcoin_core_test_results.py — import and summarize Bitcoin Core test results
                                       when the UF backend is used instead of libsecp.

Usage:
    # After running Bitcoin Core's test suite with UF backend:
    python3 scripts/import_bitcoin_core_test_results.py \
        --results-dir /path/to/bitcoin-core/build/test/results \
        --output docs/BITCOIN_CORE_TEST_RESULTS.json

    # Or parse a CTest XML output:
    python3 scripts/import_bitcoin_core_test_results.py \
        --ctest-xml /path/to/LastTestsFailed.log \
        --output docs/BITCOIN_CORE_TEST_RESULTS.json

    # Verify an existing results file:
    python3 scripts/import_bitcoin_core_test_results.py \
        --verify docs/BITCOIN_CORE_TEST_RESULTS.json

Exit codes:
    0  all tests passed (or verify succeeded)
    1  one or more tests failed
    2  invocation error
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent
DOCS_DIR = LIB_ROOT / "docs"
DEFAULT_OUTPUT = DOCS_DIR / "BITCOIN_CORE_TEST_RESULTS.json"

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_ctest_log(path: Path) -> list[dict[str, Any]]:
    """Parse a CTest LastTestsFailed.log or test output log."""
    tests: list[dict[str, Any]] = []
    text = path.read_text(errors="replace")

    # Pattern: "  N/M Test #K: <name> ...(Passed|Failed|Timeout)"
    pattern = re.compile(
        r"(?:(\d+)/(\d+)\s+)?Test\s+#(\d+):\s+(\S+)\s+\.+\s*(Passed|Failed|Timeout|Not Run)",
        re.IGNORECASE,
    )
    for m in pattern.finditer(text):
        tests.append({
            "test_id": int(m.group(3)),
            "name": m.group(4),
            "status": m.group(5).lower(),
        })

    # Also scan for simple "FAILED: <name>" lines from pytest-style output
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("FAILED ") or line.startswith("ERROR "):
            parts = line.split(None, 1)
            if len(parts) == 2:
                tests.append({"test_id": None, "name": parts[1], "status": "failed"})

    return tests


def _parse_ctest_xml(path: Path) -> list[dict[str, Any]]:
    """Parse CTest XML output (Tag/YYYY-MM-DD-HH-MM/Test.xml)."""
    import xml.etree.ElementTree as ET
    tests: list[dict[str, Any]] = []
    try:
        tree = ET.parse(str(path))
        root = tree.getroot()
        for test in root.iter("Test"):
            status = test.get("Status", "unknown").lower()
            name_el = test.find("Name")
            name = name_el.text if name_el is not None else "unknown"
            tests.append({"test_id": None, "name": name, "status": status})
    except Exception as e:
        print(f"{YELLOW}WARN: XML parse error: {e}{RESET}", file=sys.stderr)
    return tests


def _scan_results_dir(directory: Path) -> list[dict[str, Any]]:
    """Scan a directory for CTest result files."""
    tests: list[dict[str, Any]] = []
    for xml_file in sorted(directory.glob("**/*.xml")):
        tests.extend(_parse_ctest_xml(xml_file))
    for log_file in sorted(directory.glob("**/*.log")):
        tests.extend(_parse_ctest_log(log_file))
    return tests


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _build_summary(tests: list[dict[str, Any]], source: str) -> dict[str, Any]:
    total = len(tests)
    passed = sum(1 for t in tests if t["status"] in ("passed", "pass"))
    failed = sum(1 for t in tests if t["status"] in ("failed", "fail", "error"))
    skipped = total - passed - failed

    failed_tests = [t["name"] for t in tests if t["status"] in ("failed", "fail", "error")]

    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "backend": "UltrafastSecp256k1",
        "bitcoin_core_version": _get_bitcoin_core_version(),
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "pass_rate": f"{100 * passed / total:.1f}%" if total else "N/A",
        },
        "overall_pass": failed == 0 and total > 0,
        "failed_tests": failed_tests,
        "tests": tests,
    }


def _get_bitcoin_core_version() -> str:
    """Try to detect Bitcoin Core version from environment."""
    try:
        result = subprocess.run(
            ["bitcoind", "--version"],
            capture_output=True, text=True, timeout=5
        )
        first_line = result.stdout.splitlines()[0] if result.stdout else ""
        return first_line.strip()
    except Exception:
        return os.environ.get("BITCOIN_CORE_VERSION", "unknown")


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def _print_summary(report: dict[str, Any]) -> None:
    s = report["summary"]
    passed = report["overall_pass"]
    color = GREEN if passed else RED
    status = "PASS" if passed else "FAIL"

    print(f"\n{BOLD}Bitcoin Core Test Results — UltrafastSecp256k1 Backend{RESET}")
    print(f"  Source:    {report['source']}")
    print(f"  Generated: {report['generated_at']}")
    print(f"  Core ver:  {report['bitcoin_core_version']}")
    print()
    print(f"  Total:   {s['total']}")
    print(f"  {GREEN}Passed:  {s['passed']}{RESET}")
    if s["failed"]:
        print(f"  {RED}Failed:  {s['failed']}{RESET}")
    if s["skipped"]:
        print(f"  {YELLOW}Skipped: {s['skipped']}{RESET}")
    print(f"  Rate:    {s['pass_rate']}")
    print()
    print(f"  Overall: {color}{BOLD}{status}{RESET}")

    if report["failed_tests"]:
        print(f"\n{RED}Failed tests:{RESET}")
        for name in report["failed_tests"][:20]:
            print(f"    - {name}")
        if len(report["failed_tests"]) > 20:
            print(f"    ... and {len(report['failed_tests']) - 20} more")


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

def _verify(results_path: Path) -> int:
    if not results_path.exists():
        print(f"{RED}ERROR: {results_path} not found{RESET}", file=sys.stderr)
        return 2

    report = json.loads(results_path.read_text())
    _print_summary(report)

    if not report.get("overall_pass"):
        failed = report.get("failed_tests", [])
        print(f"\n{RED}FAIL: {len(failed)} tests failed in the recorded results.{RESET}", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--results-dir",
        metavar="DIR",
        help="Directory containing CTest result files (.xml, .log)",
    )
    parser.add_argument(
        "--ctest-xml",
        metavar="FILE",
        help="Path to a single CTest XML result file",
    )
    parser.add_argument(
        "--ctest-log",
        metavar="FILE",
        help="Path to a CTest LastTestsFailed.log or output log",
    )
    parser.add_argument(
        "--output", "-o",
        default=str(DEFAULT_OUTPUT),
        metavar="FILE",
        help=f"Output JSON path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--verify",
        metavar="FILE",
        help="Verify an existing results JSON file and exit",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON summary to stdout",
    )
    args = parser.parse_args(argv)

    # Verify mode
    if args.verify:
        return _verify(Path(args.verify))

    # Import mode
    tests: list[dict[str, Any]] = []
    source = "unknown"

    if args.results_dir:
        d = Path(args.results_dir)
        if not d.is_dir():
            print(f"{RED}ERROR: not a directory: {d}{RESET}", file=sys.stderr)
            return 2
        tests = _scan_results_dir(d)
        source = str(d)

    elif args.ctest_xml:
        p = Path(args.ctest_xml)
        if not p.exists():
            print(f"{RED}ERROR: file not found: {p}{RESET}", file=sys.stderr)
            return 2
        tests = _parse_ctest_xml(p)
        source = str(p)

    elif args.ctest_log:
        p = Path(args.ctest_log)
        if not p.exists():
            print(f"{RED}ERROR: file not found: {p}{RESET}", file=sys.stderr)
            return 2
        tests = _parse_ctest_log(p)
        source = str(p)

    else:
        print(f"{RED}ERROR: specify --results-dir, --ctest-xml, --ctest-log, or --verify{RESET}",
              file=sys.stderr)
        parser.print_help(sys.stderr)
        return 2

    if not tests:
        print(f"{YELLOW}WARN: no test results found in {source}{RESET}", file=sys.stderr)
        print("If using CTest, ensure the build completed and tests were run.", file=sys.stderr)
        return 1

    report = _build_summary(tests, source)

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.json:
        compact = {k: report[k] for k in ("summary", "overall_pass", "failed_tests", "generated_at")}
        print(json.dumps(compact, indent=2))
    else:
        _print_summary(report)
        print(f"\n  Written to: {out_path}")

    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
