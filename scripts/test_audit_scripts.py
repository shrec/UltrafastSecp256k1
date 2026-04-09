#!/usr/bin/env python3
"""
test_audit_scripts.py  —  Self-test for Python audit infrastructure

Validates that every Python audit/quality script in scripts/:
  1. Has valid syntax (py_compile)
  2. Has a shebang and docstring
  3. Runs --help (or -h) without crashing
  4. Key scripts produce expected output on real repo data

This is the audit-of-the-auditors layer: the Python tooling itself must
be tested, not just the C++ code it checks.

Usage:
    python3 scripts/test_audit_scripts.py           # full self-test
    python3 scripts/test_audit_scripts.py --quick    # syntax + import only
"""

import importlib.util
import json
import os
import py_compile
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent

# All audit-relevant Python scripts (relative to scripts/)
AUDIT_SCRIPTS = [
    "audit_ai_findings.py",
    "audit_gap_report.py",
    "audit_gate.py",
    "auditor_mode.py",
    "audit_test_quality_scanner.py",
    "build_owner_audit_bundle.py",
    "check_secret_path_changes.py",
    "collect_ct_evidence.py",
    "dev_bug_scanner.py",
    "differential_cross_impl.py",
    "export_assurance.py",
    "generate_abi_negative_tests.py",
    "hot_path_alloc_scanner.py",
    "invalid_input_grammar.py",
    "log_ai_review_event.py",
    "mutation_kill_rate.py",
    "nonce_bias_detector.py",
    "preflight.py",
    "query_graph.py",
    "release_diff.py",
    "research_monitor.py",
    "run_code_quality.py",
    "rfc6979_spec_verifier.py",
    "semantic_props.py",
    "stateful_sequences.py",
    "sync_audit_report_version.py",
    "sync_docs.py",
    "sync_module_count.py",
    "sync_version_refs.py",
    "validate_assurance.py",
    "verify_slsa_provenance.py",
    "test_audit_scripts.py",
]

# Scripts that have --help / -h support
HELPABLE_SCRIPTS = [
    "dev_bug_scanner.py",
    "export_assurance.py",
    "preflight.py",
    "run_code_quality.py",
]

# ANSI
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

pass_count = 0
fail_count = 0
skip_count = 0


def ok(tag: str, msg: str) -> None:
    global pass_count
    pass_count += 1
    print(f"  {GREEN}PASS{RESET} [{tag}] {msg}")


def fail(tag: str, msg: str) -> None:
    global fail_count
    fail_count += 1
    print(f"  {RED}FAIL{RESET} [{tag}] {msg}")


def skip(tag: str, msg: str) -> None:
    global skip_count
    skip_count += 1
    print(f"  {YELLOW}SKIP{RESET} [{tag}] {msg}")


def check_syntax(name: str, path: Path) -> bool:
    """py_compile check — catches SyntaxError."""
    try:
        py_compile.compile(str(path), doraise=True)
        ok("SYNTAX", name)
        return True
    except py_compile.PyCompileError as exc:
        fail("SYNTAX", f"{name}: {exc}")
        return False


def check_shebang_docstring(name: str, path: Path) -> None:
    """Verify shebang and module docstring exist."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n", 5)
    if not lines or not lines[0].startswith("#!"):
        fail("SHEBANG", f"{name}: missing shebang")
    else:
        ok("SHEBANG", name)

    if '"""' not in text[:500] and "'''" not in text[:500]:
        fail("DOCSTR", f"{name}: missing module docstring in first 500 chars")
    else:
        ok("DOCSTR", name)


def check_help(name: str, path: Path) -> None:
    """Run script --help and verify exit 0."""
    try:
        result = subprocess.run(
            [sys.executable, str(path), "--help"],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(LIB_ROOT),
        )
        if result.returncode == 0:
            ok("HELP", f"{name} --help → exit 0")
        else:
            fail(
                "HELP",
                f"{name} --help → exit {result.returncode}: "
                f"{(result.stderr or '').strip()[:120]}",
            )
    except subprocess.TimeoutExpired:
        fail("HELP", f"{name} --help timed out")
    except Exception as exc:
        fail("HELP", f"{name} --help: {exc}")


def check_dev_bug_scanner_smoke() -> None:
    """Smoke-test: dev_bug_scanner with --json produces valid JSON array."""
    tag = "SMOKE:dev_bug_scanner"
    path = SCRIPT_DIR / "dev_bug_scanner.py"
    try:
        result = subprocess.run(
            [sys.executable, str(path), "--json", "--min-severity", "HIGH"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(LIB_ROOT),
        )
        if result.returncode != 0:
            fail(tag, f"exit {result.returncode}: {result.stderr[:200]}")
            return
        data = json.loads(result.stdout)
        if not isinstance(data, list):
            fail(tag, f"expected JSON list, got {type(data).__name__}")
            return
        # Each item should have required keys
        required_keys = {"file", "line", "severity", "category", "message"}
        for item in data[:3]:
            missing = required_keys - set(item.keys())
            if missing:
                fail(tag, f"finding missing keys: {missing}")
                return
        ok(tag, f"valid JSON, {len(data)} findings")
    except json.JSONDecodeError as exc:
        fail(tag, f"invalid JSON output: {exc}")
    except subprocess.TimeoutExpired:
        fail(tag, "timed out (120s)")
    except Exception as exc:
        fail(tag, str(exc))


def check_preflight_smoke() -> None:
    """Smoke-test: preflight --bug-scan runs without crash."""
    tag = "SMOKE:preflight"
    path = SCRIPT_DIR / "preflight.py"
    try:
        result = subprocess.run(
            [sys.executable, str(path), "--bug-scan"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(LIB_ROOT),
        )
        # preflight may exit 0 or 1, but should not crash
        output = result.stdout + result.stderr
        if "Traceback" in output:
            fail(tag, f"traceback in output: {output[-300:]}")
        elif "Code Quality Gate" in result.stdout or "[13/14]" in result.stdout:
            ok(tag, "--bug-scan ran, code quality gate visible")
        else:
            fail(tag, f"--bug-scan output missing expected header")
    except subprocess.TimeoutExpired:
        fail(tag, "timed out (120s)")
    except Exception as exc:
        fail(tag, str(exc))


def check_validate_assurance_smoke() -> None:
    """Smoke-test: validate_assurance.py runs without crash."""
    tag = "SMOKE:validate_assurance"
    path = SCRIPT_DIR / "validate_assurance.py"
    if not path.exists():
        skip(tag, "validate_assurance.py not found")
        return
    try:
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(LIB_ROOT),
        )
        output = result.stdout + result.stderr
        if "Traceback" in output:
            fail(tag, f"traceback: {output[-300:]}")
        else:
            ok(tag, f"exit {result.returncode}, no traceback")
    except subprocess.TimeoutExpired:
        fail(tag, "timed out")
    except Exception as exc:
        fail(tag, str(exc))


def check_export_assurance_smoke() -> None:
    """Smoke-test: export_assurance.py -o /dev/null runs without crash."""
    tag = "SMOKE:export_assurance"
    path = SCRIPT_DIR / "export_assurance.py"
    if not path.exists():
        skip(tag, "export_assurance.py not found")
        return
    try:
        result = subprocess.run(
            [sys.executable, str(path), "-o", "/dev/null"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(LIB_ROOT),
        )
        output = result.stdout + result.stderr
        if "Traceback" in output:
            fail(tag, f"traceback: {output[-300:]}")
        else:
            ok(tag, f"exit {result.returncode}, no traceback")
    except subprocess.TimeoutExpired:
        fail(tag, "timed out")
    except Exception as exc:
        fail(tag, str(exc))


def check_category_coverage() -> None:
    """Verify dev_bug_scanner has all 5 crypto-specific categories."""
    tag = "CATEGORIES"
    path = SCRIPT_DIR / "dev_bug_scanner.py"
    text = path.read_text(encoding="utf-8")
    required = [
        "SECRET_UNERASED",
        "CT_VIOLATION",
        "TAGGED_HASH_BYPASS",
        "RANDOM_IN_SIGNING",
        "BINDING_NO_VALIDATION",
    ]
    missing = [c for c in required if c not in text]
    if missing:
        fail(tag, f"dev_bug_scanner missing categories: {missing}")
    else:
        ok(tag, "all 5 crypto-specific categories present")


def check_preflight_step_count() -> None:
    """Verify preflight.py has [14/14] (all steps numbered correctly)."""
    tag = "STEPS"
    path = SCRIPT_DIR / "preflight.py"
    text = path.read_text(encoding="utf-8")
    for i in range(1, 15):
        marker = f"[{i}/14]"
        if marker not in text:
            fail(tag, f"preflight.py missing step {marker}")
            return
    ok(tag, "preflight.py has all 14/14 steps")


def check_code_quality_runner_smoke() -> None:
    """Smoke-test: run_code_quality.py produces valid JSON."""
    tag = "SMOKE:run_code_quality"
    path = SCRIPT_DIR / "run_code_quality.py"
    if not path.exists():
        skip(tag, "run_code_quality.py not found")
        return
    try:
        result = subprocess.run(
            [sys.executable, str(path), "--json"],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(LIB_ROOT),
        )
        output = result.stdout + result.stderr
        if "Traceback" in output:
            fail(tag, f"traceback: {output[-300:]}")
            return
        data = json.loads(result.stdout)
        if "scanners" not in data or "total_findings" not in data:
            fail(tag, "JSON output missing expected keys")
        else:
            ok(tag, f"exit {result.returncode}, {data['total_findings']} findings, "
                     f"{len(data.get('regressions', []))} regressions")
    except json.JSONDecodeError as exc:
        fail(tag, f"invalid JSON: {exc}")
    except subprocess.TimeoutExpired:
        fail(tag, "timed out (300s)")
    except Exception as exc:
        fail(tag, str(exc))


def main() -> int:
    quick = "--quick" in sys.argv

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Python Audit Infrastructure Self-Test{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")

    # Phase 1: Syntax + structure
    print(f"{BOLD}[1/4] Syntax & Structure{RESET}")
    existing = []
    for name in AUDIT_SCRIPTS:
        path = SCRIPT_DIR / name
        if not path.exists():
            skip("EXISTS", f"{name} not found")
            continue
        existing.append((name, path))
        if check_syntax(name, path):
            check_shebang_docstring(name, path)

    # Phase 2: --help
    print(f"\n{BOLD}[2/4] --help Validation{RESET}")
    if quick:
        print(f"  {YELLOW}SKIPPED (--quick){RESET}")
    else:
        for name in HELPABLE_SCRIPTS:
            path = SCRIPT_DIR / name
            if path.exists():
                check_help(name, path)

    # Phase 3: Structural checks
    print(f"\n{BOLD}[3/4] Structural Integrity{RESET}")
    check_category_coverage()
    check_preflight_step_count()

    # Phase 4: Smoke tests
    print(f"\n{BOLD}[4/4] Smoke Tests{RESET}")
    if quick:
        print(f"  {YELLOW}SKIPPED (--quick){RESET}")
    else:
        check_dev_bug_scanner_smoke()
        check_code_quality_runner_smoke()
        check_preflight_smoke()
        check_validate_assurance_smoke()
        check_export_assurance_smoke()

    # Summary
    print(f"\n{BOLD}{'='*60}{RESET}")
    total = pass_count + fail_count
    if fail_count == 0:
        print(
            f"{GREEN}{BOLD}  PYTHON AUDIT SELF-TEST PASSED "
            f"({pass_count} passed, {skip_count} skipped){RESET}"
        )
    else:
        print(
            f"{RED}{BOLD}  PYTHON AUDIT SELF-TEST: {fail_count} FAILED "
            f"({pass_count} passed, {skip_count} skipped){RESET}"
        )
    print(f"{BOLD}{'='*60}{RESET}\n")

    return 1 if fail_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
