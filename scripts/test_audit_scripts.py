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
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent

# All audit-relevant Python scripts (relative to scripts/)
AUDIT_SCRIPTS = [
    "external_audit_bundle.py",
    "verify_external_audit_bundle.py",
    "audit_ai_findings.py",
    "audit_gap_report.py",
    "audit_gate.py",
    "audit_verdict.py",
    "auditor_mode.py",
    "audit_test_quality_scanner.py",
    "build_owner_audit_bundle.py",
    "check_secret_path_changes.py",
    "check_api_contracts.py",
    "check_determinism_gate.py",
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
    "external_audit_bundle.py",
    "verify_external_audit_bundle.py",
    "audit_verdict.py",
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


def check_preflight_ctest_registry_smoke() -> None:
    """Smoke-test: preflight --ctest-registry runs without crash."""
    tag = "SMOKE:preflight_ctest_registry"
    path = SCRIPT_DIR / "preflight.py"
    try:
        result = subprocess.run(
            [sys.executable, str(path), "--ctest-registry"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(LIB_ROOT),
        )
        output = result.stdout + result.stderr
        if "Traceback" in output:
            fail(tag, f"traceback in output: {output[-300:]}")
            return
        if "CTest Registry Health" not in result.stdout:
            fail(tag, "--ctest-registry output missing expected header")
            return
        if result.returncode not in (0, 1):
            fail(tag, f"unexpected exit code {result.returncode}")
            return
        ok(tag, f"--ctest-registry ran (exit {result.returncode})")
    except subprocess.TimeoutExpired:
        fail(tag, "timed out (120s)")
    except Exception as exc:
        fail(tag, str(exc))


def check_preflight_ctest_registry_classification() -> None:
    """Verify ctest-registry classifies missing binaries correctly."""
    tag = "CLASSIFY:ctest_registry"
    path = SCRIPT_DIR / "preflight.py"
    try:
        spec = importlib.util.spec_from_file_location("preflight_selftest", str(path))
        if spec is None or spec.loader is None:
            fail(tag, "could not load preflight.py module spec")
            return

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        original_lib_root = module.LIB_ROOT
        original_run = module.subprocess.run

        def fake_run(argv, capture_output=False, text=False, cwd=None, check=False):
            exe = argv[0]
            if exe == "ninja":
                return subprocess.CompletedProcess(
                    argv,
                    0,
                    stdout="test_unbuilt: phony\nhelper_target: phony\n",
                    stderr="",
                )
            if exe == "ctest":
                payload = {
                    "tests": [
                        {"name": "unbuilt-test", "command": ["bin/test_unbuilt"]},
                        {"name": "stale-test", "command": ["bin/test_stale"]},
                        {"name": "launcher-test", "command": ["python3", "scripts/helper.py"]},
                    ]
                }
                return subprocess.CompletedProcess(
                    argv,
                    0,
                    stdout=json.dumps(payload),
                    stderr="",
                )
            raise AssertionError(f"unexpected subprocess invocation: {argv}")

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            build_rel = temp_root / "build_rel"
            build_rel.mkdir(parents=True)
            (build_rel / "CTestTestfile.cmake").write_text("# synthetic ctest registry\n", encoding="utf-8")
            (build_rel / "build.ninja").write_text("# synthetic ninja file\n", encoding="utf-8")

            module.LIB_ROOT = temp_root
            module.subprocess.run = fake_run
            try:
                issues = module.check_ctest_registry_health()
            finally:
                module.LIB_ROOT = original_lib_root
                module.subprocess.run = original_run

        if not any("UNBUILT-TEST" in issue and "unbuilt-test" in issue for issue in issues):
            fail(tag, f"missing UNBUILT-TEST classification: {issues}")
            return
        if not any("STALE-CTEST" in issue and "stale-test" in issue for issue in issues):
            fail(tag, f"missing STALE-CTEST classification: {issues}")
            return
        if any("launcher-test" in issue for issue in issues):
            fail(tag, f"launcher command should have been ignored: {issues}")
            return
        ok(tag, "UNBUILT-TEST and STALE-CTEST classifications are both stable")
    except Exception as exc:
        fail(tag, str(exc))


def check_api_contracts_smoke() -> None:
    """Smoke-test: check_api_contracts emits valid JSON schema and passes."""
    tag = "SMOKE:api_contracts"
    path = SCRIPT_DIR / "check_api_contracts.py"
    if not path.exists():
        skip(tag, "check_api_contracts.py not found")
        return
    try:
        result = subprocess.run(
            [sys.executable, str(path), "--json"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(LIB_ROOT),
        )
        output = result.stdout + result.stderr
        if "Traceback" in output:
            fail(tag, f"traceback in output: {output[-300:]}")
            return
        if result.returncode != 0:
            fail(tag, f"expected exit 0, got {result.returncode}: {output[:200]}")
            return
        data = json.loads(result.stdout)
        required = {"contract_file", "changed_files", "issues", "entries"}
        missing = required - set(data.keys())
        if missing:
            fail(tag, f"missing JSON keys: {sorted(missing)}")
            return
        if not isinstance(data.get("entries"), int) or data["entries"] <= 0:
            fail(tag, "expected positive entries count")
            return
        if data.get("issues"):
            fail(tag, f"unexpected contract issues: {data['issues'][:3]}")
            return
        ok(tag, f"valid JSON, {data['entries']} API contract entries")
    except json.JSONDecodeError as exc:
        fail(tag, f"invalid JSON output: {exc}")
    except subprocess.TimeoutExpired:
        fail(tag, "timed out (30s)")
    except Exception as exc:
        fail(tag, str(exc))


def check_determinism_gate_smoke() -> None:
    """Smoke-test: determinism checker emits valid JSON and passes locked vectors."""
    tag = "SMOKE:determinism_gate"
    path = SCRIPT_DIR / "check_determinism_gate.py"
    if not path.exists():
        skip(tag, "check_determinism_gate.py not found")
        return
    try:
        result = subprocess.run(
            [sys.executable, str(path), "--json", "--repeat", "3"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(LIB_ROOT),
        )
        output = result.stdout + result.stderr
        if "Traceback" in output:
            fail(tag, f"traceback in output: {output[-300:]}")
            return
        if result.returncode != 0:
            fail(tag, f"expected exit 0, got {result.returncode}: {output[:200]}")
            return

        data = json.loads(result.stdout)
        required = {"library", "repeat", "overall_pass", "issues", "checks"}
        missing = required - set(data.keys())
        if missing:
            fail(tag, f"missing JSON keys: {sorted(missing)}")
            return
        if not data.get("overall_pass", False):
            fail(tag, f"determinism gate reported failure: {data.get('issues', [])[:3]}")
            return
        checks = data.get("checks", {})
        for key in ("ecdsa_vectors", "ecdh_pairs", "bip32_paths"):
            if key not in checks:
                fail(tag, f"checks missing section: {key}")
                return
        ok(tag, "valid JSON and deterministic behavior for locked vectors")
    except json.JSONDecodeError as exc:
        fail(tag, f"invalid JSON output: {exc}")
    except subprocess.TimeoutExpired:
        fail(tag, "timed out (60s)")
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
    """Verify preflight.py uses contiguous [i/N] step markers."""
    tag = "STEPS"
    path = SCRIPT_DIR / "preflight.py"
    text = path.read_text(encoding="utf-8")
    matches = re.findall(r"\[(\d+)/(\d+)\]", text)
    if not matches:
        fail(tag, "preflight.py has no [i/N] step markers")
        return

    numerators = sorted({int(n) for n, _ in matches})
    denominators = {int(d) for _, d in matches}

    if len(denominators) != 1:
        fail(tag, f"inconsistent denominators in preflight markers: {sorted(denominators)}")
        return

    total = denominators.pop()
    expected = list(range(1, total + 1))
    if numerators != expected:
        fail(tag, f"preflight.py step markers are non-contiguous: got {numerators}, expected {expected}")
        return

    ok(tag, f"preflight.py has contiguous {total}/{total} step markers")


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


def check_hot_path_alloc_scanner_quality() -> None:
    """Unit-smoke: scanner truthfulness on synthetic fixtures."""
    tag = "QUALITY:hot_path_alloc_scanner"
    path = SCRIPT_DIR / "hot_path_alloc_scanner.py"
    if not path.exists():
        skip(tag, "hot_path_alloc_scanner.py not found")
        return

    try:
        spec = importlib.util.spec_from_file_location("hot_path_alloc_scanner_selftest", str(path))
        if spec is None or spec.loader is None:
            fail(tag, "could not load hot_path_alloc_scanner.py module spec")
            return

        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # 1) One-time static initializer with new[] should be ignored.
            one_time = root / "cpu/src/point.cpp"
            one_time.parent.mkdir(parents=True, exist_ok=True)
            one_time.write_text(
                """
Point dual_scalar_mul_gen_point() {
    static const auto* t = []() {
        auto* p = new int[16];
        return p;
    }();
    (void)t;
    return Point::infinity();
}
""".strip()
                + "\n",
                encoding="utf-8",
            )

            findings = module.scan_file(one_time, root)
            if any(f.category == "HEAP_NEW" for f in findings):
                fail(tag, f"one-time static init was incorrectly flagged: {findings}")
                return

            # 2) True positive in hot CPU path should still be flagged.
            true_pos = root / "cpu/src/scalar.cpp"
            true_pos.write_text(
                """
int scalar_mul_probe() {
    std::vector<int> v(8);
    return static_cast<int>(v.size());
}
""".strip()
                + "\n",
                encoding="utf-8",
            )
            findings = module.scan_file(true_pos, root)
            if not any(f.category == "HEAP_VEC" and f.function == "scalar_mul_probe" for f in findings):
                fail(tag, f"expected HEAP_VEC true positive missing: {findings}")
                return

            # 3) Benchmark helper vector-return should be ignored.
            bench = root / "opencl/benchmarks/bench_opencl.cpp"
            bench.parent.mkdir(parents=True, exist_ok=True)
            bench.write_text(
                """
std::vector<int> zk_pubkeys() {
    return {1, 2, 3};
}
""".strip()
                + "\n",
                encoding="utf-8",
            )
            findings = module.scan_file(bench, root)
            if any(f.category == "HEAP_RET" for f in findings):
                fail(tag, f"benchmark helper return-by-value was incorrectly flagged: {findings}")
                return

            # 4) GPU marshalling helper vector-return should be ignored.
            gpu = root / "gpu/src/gpu_backend_cuda.cu"
            gpu.parent.mkdir(parents=True, exist_ok=True)
            gpu.write_text(
                """
std::vector<int> h_tweaks() {
    return {1, 2, 3};
}
""".strip()
                + "\n",
                encoding="utf-8",
            )
            findings = module.scan_file(gpu, root)
            if any(f.category == "HEAP_RET" for f in findings):
                fail(tag, f"GPU marshalling helper return-by-value was incorrectly flagged: {findings}")
                return

        ok(tag, "one-time/benchmark/GPU exemptions hold and true positives remain detectable")
    except Exception as exc:
        fail(tag, str(exc))


def check_audit_verdict_smoke() -> None:
    """Smoke-test: cancelled platforms do not fail aggregate verdict."""
    tag = "SMOKE:audit_verdict"
    path = SCRIPT_DIR / "audit_verdict.py"
    if not path.exists():
        skip(tag, "audit_verdict.py not found")
        return
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gcc_dir = root / "audit-report-linux-gcc13"
            gcc_dir.mkdir(parents=True)
            (gcc_dir / "audit_report.json").write_text(
                json.dumps({"audit_verdict": "PASS"}), encoding="utf-8"
            )
            msvc_dir = root / "audit-report-windows-msvc"
            msvc_dir.mkdir(parents=True)
            (msvc_dir / "audit_report.json").write_text(
                json.dumps({"audit_verdict": "AUDIT-READY"}), encoding="utf-8"
            )
            summary = root / "summary.md"

            result = subprocess.run(
                [
                    sys.executable,
                    str(path),
                    "--artifact-root",
                    str(root),
                    "--summary-file",
                    str(summary),
                    "--platform",
                    "audit-report-linux-gcc13=success",
                    "--platform",
                    "audit-report-linux-clang17=cancelled",
                    "--platform",
                    "audit-report-windows-msvc=success",
                ],
                capture_output=True,
                text=True,
                timeout=15,
                cwd=str(LIB_ROOT),
            )
            output = result.stdout + result.stderr
            if result.returncode != 0:
                fail(tag, f"expected exit 0, got {result.returncode}: {output[:200]}")
                return
            summary_text = summary.read_text(encoding="utf-8")
            if "NO REPORT (cancelled)" not in summary_text:
                fail(tag, "summary missing cancelled-platform row")
                return
            if "**Overall: PASS**" not in summary_text:
                fail(tag, "summary missing overall PASS")
                return
            ok(tag, "cancelled platform is summarized without failing verdict")
    except subprocess.TimeoutExpired:
        fail(tag, "timed out")
    except Exception as exc:
        fail(tag, str(exc))


def check_audit_verdict_requires_evidence() -> None:
    """Smoke-test: aggregate verdict fails closed if no platform produced a report."""
    tag = "SMOKE:audit_verdict_no_evidence"
    path = SCRIPT_DIR / "audit_verdict.py"
    if not path.exists():
        skip(tag, "audit_verdict.py not found")
        return
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = root / "summary.md"

            result = subprocess.run(
                [
                    sys.executable,
                    str(path),
                    "--artifact-root",
                    str(root),
                    "--summary-file",
                    str(summary),
                    "--platform",
                    "audit-report-linux-gcc13=cancelled",
                    "--platform",
                    "audit-report-linux-clang17=cancelled",
                    "--platform",
                    "audit-report-windows-msvc=skipped",
                ],
                capture_output=True,
                text=True,
                timeout=15,
                cwd=str(LIB_ROOT),
            )
            output = result.stdout + result.stderr
            if result.returncode == 0:
                fail(tag, "expected non-zero exit when no audit evidence exists")
                return
            if "no audit evidence was produced on any platform" not in output:
                fail(tag, f"missing fail-closed error in output: {output[:200]}")
                return
            summary_text = summary.read_text(encoding="utf-8")
            if "No usable audit_report.json artifact was produced on any platform." not in summary_text:
                fail(tag, "summary missing no-evidence explanation")
                return
            if "**Overall: FAIL**" not in summary_text:
                fail(tag, "summary missing overall FAIL")
                return
            ok(tag, "no-evidence case fails closed")
    except subprocess.TimeoutExpired:
        fail(tag, "timed out")
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
        check_audit_verdict_smoke()
        check_audit_verdict_requires_evidence()
        check_api_contracts_smoke()
        check_determinism_gate_smoke()
        check_code_quality_runner_smoke()
        check_hot_path_alloc_scanner_quality()
        check_preflight_smoke()
        check_preflight_ctest_registry_smoke()
        check_preflight_ctest_registry_classification()
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
