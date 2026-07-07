#!/usr/bin/env python3
"""
test_audit_scripts.py  —  Self-test for Python audit infrastructure

Validates that every Python audit/quality script in ci/:
  1. Has valid syntax (py_compile)
  2. Has a shebang and docstring
  3. Runs --help (or -h) without crashing
  4. Key scripts produce expected output on real repo data

This is the audit-of-the-auditors layer: the Python tooling itself must
be tested, not just the C++ code it checks.

Usage:
    python3 ci/test_audit_scripts.py           # full self-test
    python3 ci/test_audit_scripts.py --quick    # syntax + import only
"""

import importlib.util
import contextlib
import io
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

# All audit-relevant Python scripts (relative to ci/)
AUDIT_SCRIPTS = [
    "caas_dashboard.py",
    "caas_runner.py",
    "install_caas_hooks.py",
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
    "test_sync_module_count.py",
    "sync_version_refs.py",
    "validate_assurance.py",
    "verify_slsa_provenance.py",
    "check_abi_version_sync.py",
    "check_randomize_claim_consistency.py",
    "check_required_checks_match_jobs.py",
    "check_advisory_json_rule16.py",
    "check_doc_module_counts.py",
    "check_source_graph_quality.py",
    "check_integration_evidence.py",
    "check_ct_evidence_status.py",
    "check_fuzz_campaign_status.py",
    "check_gpu_hardware_evidence.py",
    "check_bench_target_context.py",
    "check_research_signal_matrix.py",
    "check_evidence_refresh_coverage.py",
    "check_package_provenance_binding.py",
    "check_release_package_contents.py",
    "check_libbitcoin_perf_matrix.py",
    "test_caas_integrity.py",
    "test_audit_scripts.py",
]

# Scripts that have --help / -h support
HELPABLE_SCRIPTS = [
    "caas_runner.py",
    "install_caas_hooks.py",
    "external_audit_bundle.py",
    "verify_external_audit_bundle.py",
    "audit_verdict.py",
    "dev_bug_scanner.py",
    "export_assurance.py",
    "preflight.py",
    "run_code_quality.py",
    "check_release_package_contents.py",
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
                        {"name": "launcher-test", "command": ["python3", "ci/helper.py"]},
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
            # check_ctest_registry_health scans LIB_ROOT/out/* (BUILD-DIR-001 moved
            # all builds under out/). The synthetic build dir must live under out/,
            # not at the repo root — otherwise the scan finds nothing and returns
            # "no CTest build directory found", and this classification test fails.
            build_rel = temp_root / "out" / "build_rel"
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
    """Verify dev_bug_scanner has all required crypto-specific categories."""
    tag = "CATEGORIES"
    path = SCRIPT_DIR / "dev_bug_scanner.py"
    text = path.read_text(encoding="utf-8")
    required = [
        "SECRET_UNERASED",
        "CT_VIOLATION",
        "TAGGED_HASH_BYPASS",
        "RANDOM_IN_SIGNING",
        "BINDING_NO_VALIDATION",
        "SECRET_TABLE_INDEX",
        "UNALIGNED_WORD_LOAD",
        "OUTPUT_FAIL_OPEN",
    ]
    missing = [c for c in required if c not in text]
    if missing:
        fail(tag, f"dev_bug_scanner missing categories: {missing}")
    else:
        ok(tag, f"all {len(required)} crypto-specific categories present")


def check_dev_bug_scanner_deep_patterns() -> None:
    """Unit-smoke: high-signal crypto bug patterns on synthetic fixtures."""
    tag = "QUALITY:dev_bug_scanner_deep"
    path = SCRIPT_DIR / "dev_bug_scanner.py"
    try:
        spec = importlib.util.spec_from_file_location("dev_bug_scanner_selftest", str(path))
        if spec is None or spec.loader is None:
            fail(tag, "could not load dev_bug_scanner.py module spec")
            return

        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            src = root / "src" / "cpu" / "src" / "ecdsa.cpp"
            src.parent.mkdir(parents=True, exist_ok=True)
            src.write_text(
                """
#include <cstdint>
#include <cstring>

int vulnerable_sign_lookup(const uint8_t* seckey, const unsigned char* input, unsigned char* output32) {
    int table[16] = {};
    auto idx = seckey[0] & 15;
    output32[0] = static_cast<unsigned char>(*reinterpret_cast<const uint64_t*>(input));
    if (idx == 7) return 0;
    return table[idx];
}

int secp256k1_ecdsa_sign(const void*, void*, const unsigned char*, const unsigned char*, void*, void*) {
    return 1;
}

int secp256k1_recover_secret(const void*, unsigned char* output32, int bad) {
    output32[0] = 1;
    if (bad) return 0;
    return 1;
}

int safe_secret_read(const uint8_t* seckey, size_t i) {
    return seckey[i];
}

int secp256k1_safe_recover(const void*, unsigned char* output32, int bad) {
    output32[0] = 1;
    if (bad) {
        memset(output32, 0, 32);
        return 0;
    }
    return 1;
}

int secp256k1_cache_then_clear(const void*, unsigned char* output32, int hit, int bad) {
    if (hit) {
        memset(output32, 7, 32);
        return 1;
    }
    memset(output32, 0, 32);
    if (bad) return 0;
    return 1;
}
""".strip()
                + "\n",
                encoding="utf-8",
            )

            findings = module.scan_file(src, root)
            categories = {f.category for f in findings}
            for expected in ("SECRET_TABLE_INDEX", "UNALIGNED_WORD_LOAD", "OUTPUT_FAIL_OPEN"):
                if expected not in categories:
                    fail(tag, f"expected {expected} true positive missing: {findings}")
                    return
            if any(f.category == "SECRET_TABLE_INDEX" and "safe_secret_read" in f.snippet for f in findings):
                fail(tag, f"secret buffer read was incorrectly flagged: {findings}")
                return
            if any(f.category == "OUTPUT_FAIL_OPEN" and "secp256k1_safe_recover" in f.message for f in findings):
                fail(tag, f"cleared failure path was incorrectly flagged: {findings}")
                return
            if any(f.category == "OUTPUT_FAIL_OPEN" and "secp256k1_cache_then_clear" in f.message for f in findings):
                fail(tag, f"clear after successful write branch was incorrectly flagged: {findings}")
                return

        ok(tag, "secret-index, unaligned-load, and output-fail-open detectors catch synthetic bugs")
    except Exception as exc:
        fail(tag, str(exc))


def check_research_monitor_resilience() -> None:
    """Unit-smoke: research monitor should tolerate malformed source metadata."""
    tag = "QUALITY:research_monitor_resilience"
    path = SCRIPT_DIR / "research_monitor.py"
    try:
        spec = importlib.util.spec_from_file_location("research_monitor_selftest", str(path))
        if spec is None or spec.loader is None:
            fail(tag, "could not load research_monitor.py module spec")
            return

        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        zero_month = module.parse_crossref_date_parts([[2026, 0, 0]])
        if (zero_month.year, zero_month.month, zero_month.day) != (2026, 1, 1):
            fail(tag, f"Crossref zero month/day was not sanitized: {zero_month!r}")
            return

        invalid_day = module.parse_crossref_date_parts([[2025, 2, 31]])
        if (invalid_day.year, invalid_day.month, invalid_day.day) != (2025, 2, 1):
            fail(tag, f"Crossref invalid calendar day was not clamped: {invalid_day!r}")
            return

        long_error = "socket timeout\n" * 40
        compact_error = module.compact_report_error(long_error, limit=80)
        if "\n" in compact_error or len(compact_error) > 80:
            fail(tag, f"source error was not compacted: {compact_error!r}")
            return

        rss = """<?xml version="1.0" encoding="UTF-8"?>
<rss xmlns:dc="http://purl.org/dc/elements/1.1/" version="2.0">
  <channel>
    <item>
      <title>ECDSA nonce bias lattice attack on elliptic curve signatures</title>
      <link>https://eprint.iacr.org/2026/999</link>
      <description>New secp256k1 attack surface for biased nonces.</description>
      <guid isPermaLink="true">https://eprint.iacr.org/2026/999</guid>
      <category>Attacks and cryptanalysis</category>
      <pubDate>Fri, 12 Jun 2026 12:00:00 +0000</pubDate>
      <dc:creator>A. Researcher</dc:creator>
    </item>
    <item>
      <title>Unrelated post-quantum KEM note</title>
      <link>https://eprint.iacr.org/2026/998</link>
      <description>Code-based cryptography only.</description>
      <guid isPermaLink="true">https://eprint.iacr.org/2026/998</guid>
      <category>Public-key cryptography</category>
      <pubDate>Fri, 12 Jun 2026 12:00:00 +0000</pubDate>
    </item>
  </channel>
</rss>
"""
        old_http_get_text = module.http_get_text
        try:
            module.http_get_text = lambda *args, **kwargs: rss
            eprint_items = module.fetch_eprint(
                "ecdsa nonce bias lattice",
                5,
                module.datetime(2026, 1, 1, tzinfo=module.timezone.utc),
            )
        finally:
            module.http_get_text = old_http_get_text
        if len(eprint_items) != 1 or eprint_items[0].source != "IACR ePrint":
            fail(tag, f"IACR ePrint RSS filtering failed: {eprint_items!r}")
            return

        bio_item = module.SourceItem(
            source="Crossref",
            item_id="10.1101/gr.281517.125",
            title="Elevated DNA insertion rate in Cyanophora paradoxa reveals unique repair signature",
            summary="Spontaneous mutation rates have been estimated in approximately 200 species.",
            published=module.datetime(2026, 6, 12, tzinfo=module.timezone.utc),
            updated=module.datetime(2026, 6, 12, tzinfo=module.timezone.utc),
            url="https://doi.org/10.1101/gr.281517.125",
        )
        ros_signal = module.SignalClass(
            signal_id="ros_concurrent_schnorr_forgery",
            status="covered",
            priority="high",
            action="monitor",
            keywords=("ROS", "blind Schnorr"),
            repo_evidence=("audit/test_exploit_ros_concurrent_schnorr.cpp",),
            reason="covered regression fixture",
        )
        if module.term_matches(bio_item.summary, "ecies") or module.term_matches(bio_item.title, "ROS"):
            fail(tag, "crypto terms matched inside unrelated biological words")
            return
        bio_classification = module.classify_item(bio_item, [ros_signal])
        if bio_classification["bucket"] != "discard":
            fail(tag, f"biological repair signature false-positive was not discarded: {bio_classification}")
            return

        published = module.datetime(2026, 6, 12, tzinfo=module.timezone.utc)
        item = module.SourceItem(
            source="Crossref",
            item_id="10.0000/demo",
            title="secp256k1 ECDSA nonce bias",
            summary="lattice attack against biased ECDSA nonces",
            published=published,
            updated=published,
            url="https://doi.org/10.0000/demo",
        )
        review_item = module.SourceItem(
            source="IACR ePrint",
            item_id="https://eprint.iacr.org/2026/1000",
            title="Curve25519 implementation note",
            summary="Portability update for elliptic-curve software.",
            published=published,
            updated=published,
            url="https://eprint.iacr.org/2026/1000",
        )
        report = module.build_report(
            [item, review_item],
            [],
            "secp256k1",
            14,
            [{"source": "Crossref", "query": "secp256k1", "count": 1, "status": "ok"}],
            [{"source": "NVD", "query": "libsecp256k1", "error": "timeout"}],
        )
        markdown = module.render_markdown(report)
        text = module.render_text(report)
        mail = module.render_mail_body(report)
        if "Crossref [secp256k1]: ok (1 raw items)" not in markdown:
            fail(tag, "markdown source status is missing query context")
            return
        for rendered in (markdown, text, mail):
            if "NVD [libsecp256k1]: timeout" not in rendered:
                fail(tag, "source error is missing query context")
                return
            if "Curve25519 implementation note" not in rendered:
                fail(tag, "needs-review item details are missing from rendered output")
                return

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "github-output.txt"
            module.write_github_outputs(output, report)
            output_text = output.read_text(encoding="utf-8")
        if "research_signal_count=2" not in output_text:
            fail(tag, "GitHub output is missing research_signal_count")
            return

        calls = []

        def fake_run_label_fallback(cmd, **kwargs):
            calls.append(cmd)
            if cmd[:3] == ["gh", "issue", "list"]:
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            if "--label" in cmd:
                raise subprocess.CalledProcessError(1, cmd, stderr="Label does not exist")
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout="https://github.com/shrec/UltrafastSecp256k1/issues/1\n",
                stderr="",
            )

        old_run = module.subprocess.run
        try:
            module.subprocess.run = fake_run_label_fallback
            with contextlib.redirect_stdout(io.StringIO()):
                module.open_github_issue(report, include_review=True)
        finally:
            module.subprocess.run = old_run
        create_calls = [cmd for cmd in calls if cmd[:3] == ["gh", "issue", "create"]]
        if len(create_calls) != 2 or "--label" not in create_calls[0] or "--label" in create_calls[1]:
            fail(tag, f"issue label fallback did not retry without labels: {calls}")
            return

        duplicate_calls = []

        def fake_run_duplicate(cmd, **kwargs):
            duplicate_calls.append(cmd)
            if cmd[:3] == ["gh", "issue", "list"]:
                return subprocess.CompletedProcess(cmd, 0, stdout="7\n", stderr="")
            raise AssertionError(f"duplicate path should not create issue: {cmd}")

        try:
            module.subprocess.run = fake_run_duplicate
            with contextlib.redirect_stdout(io.StringIO()):
                module.open_github_issue(report, include_review=True)
        finally:
            module.subprocess.run = old_run
        if any(cmd[:3] == ["gh", "issue", "create"] for cmd in duplicate_calls):
            fail(tag, f"duplicate issue path attempted create: {duplicate_calls}")
            return

        pq_noise = module.SourceItem(
            source="Crossref",
            item_id="10.1145/3807506",
            title="Deep Learning Based Side-Channel Attack on Polynomial Multiplication in Post-Quantum Cryptography",
            summary="",
            published=published,
            updated=published,
            url="https://doi.org/10.1145/3807506",
        )
        matrix_classes = module.load_signal_matrix(module.DEFAULT_MATRIX)
        pq_classification = module.classify_item(pq_noise, matrix_classes)
        if pq_classification["bucket"] != "discard":
            fail(tag, f"post-quantum polynomial side-channel noise was not discarded: {pq_classification}")
            return

        workflow = (LIB_ROOT / ".github" / "workflows" / "research-monitor.yml").read_text(encoding="utf-8")
        if "OPEN_REVIEW='false'" not in workflow or "open_review_issue" not in workflow:
            fail(tag, "research monitor workflow does not default scheduled review escalation to false")
            return
        if "OPEN_REVIEW='${{ github.event.inputs.open_review_issue || 'false' }}'" not in workflow:
            fail(tag, "manual review-escalation fallback is not fail-closed")
            return

        ok(tag, "ePrint RSS, term boundaries, PQ noise discard, report rendering, and issue escalation fallbacks are covered")
    except Exception as exc:
        fail(tag, str(exc))


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
            one_time = root / "src/cpu/src/point.cpp"
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
            true_pos = root / "src/cpu/src/scalar.cpp"
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
            bench = root / "src/opencl/benchmarks/bench_opencl.cpp"
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
            gpu = root / "src/gpu/src/gpu_backend_cuda.cu"
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


def check_secret_path_changes_fail_closed() -> None:
    """CAAS-09 regression for CAAS-06: the secret-path gate must FAIL CLOSED when
    `git diff <ref>..HEAD` errors (unreachable ref / shallow clone). Before the
    2026-05-28 fix the returncode was ignored, so a failed diff produced an empty
    change set and the gate PASSED — fail-open, even if a secret-bearing file was
    modified. This monkeypatches subprocess.run to simulate the diff failure and
    asserts get_changed_files raises (does not silently return an empty list)."""
    tag = "CAAS-06:secret_path_fail_closed"
    path = SCRIPT_DIR / "check_secret_path_changes.py"
    try:
        spec = importlib.util.spec_from_file_location("secret_path_selftest", str(path))
        if spec is None or spec.loader is None:
            fail(tag, "could not load check_secret_path_changes.py module spec")
            return
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        original_run = module.subprocess.run

        def failing_git(argv, capture_output=False, text=False, cwd=None, check=False):
            # Simulate `git diff <bad-ref>..HEAD` failing (rc=128 = bad revision).
            return subprocess.CompletedProcess(
                argv, 128, stdout="", stderr="fatal: bad revision 'bad-ref..HEAD'"
            )

        module.subprocess.run = failing_git
        raised = False
        result = None
        try:
            result = module.get_changed_files(base="this-ref-does-not-exist-xyz")
        except SystemExit:
            raised = True
        except Exception:
            # Any hard error also counts as fail-closed (the gate did not pass silently).
            raised = True
        finally:
            module.subprocess.run = original_run

        if raised:
            ok(tag, "git-diff failure raises (fail-closed), not a silent empty-pass")
        elif not result:
            fail(tag, "git-diff failure returned an empty change set — FAIL-OPEN (CAAS-06 regression)")
        else:
            fail(tag, f"git-diff failure neither raised nor empty: {result!r}")
    except Exception as exc:
        fail(tag, str(exc))


def check_secret_path_before_sha_fallback() -> None:
    """Regression: force-push events can provide a `before` SHA that is not present
    in the checkout even with fetch-depth:0. The gate must not pass empty, but it
    may conservatively fall back to the base ref when that diff is available."""
    tag = "CAAS-06:secret_path_before_sha_fallback"
    path = SCRIPT_DIR / "check_secret_path_changes.py"
    try:
        spec = importlib.util.spec_from_file_location("secret_path_fallback_selftest", str(path))
        if spec is None or spec.loader is None:
            fail(tag, "could not load check_secret_path_changes.py module spec")
            return
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        original_run = module.subprocess.run
        bad_before = "a" * 40

        def force_push_git(argv, capture_output=False, text=False, cwd=None, check=False):
            command = " ".join(argv)
            if argv[:3] == ["git", "diff", "--name-only"] and f"{bad_before}..HEAD" in command:
                return subprocess.CompletedProcess(
                    argv, 128, stdout="", stderr="fatal: Invalid revision range"
                )
            if argv[:4] == ["git", "fetch", "--no-tags", "--depth=1"] and bad_before in command:
                return subprocess.CompletedProcess(
                    argv, 128, stdout="", stderr="fatal: couldn't find remote ref"
                )
            if argv[:3] == ["git", "diff", "--name-only"] and "origin/main..HEAD" in command:
                return subprocess.CompletedProcess(
                    argv,
                    0,
                    stdout="src/cpu/src/ecdsa.cpp\ndocs/CT_VERIFICATION.md\n",
                    stderr="",
                )
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="unexpected command")

        module.subprocess.run = force_push_git
        try:
            files = module.get_changed_files(base="origin/main", before_sha=bad_before)
        finally:
            module.subprocess.run = original_run

        if files == ["docs/CT_VERIFICATION.md", "src/cpu/src/ecdsa.cpp"]:
            ok(tag, "unreachable before-sha falls back to base diff without empty-pass")
        else:
            fail(tag, f"unexpected fallback changed files: {files!r}")
    except Exception as exc:
        fail(tag, str(exc))


def check_rule16_json_smoke() -> None:
    """Smoke-test (CAAS-CI-002): check_advisory_json_rule16.py must pass a clean
    report and FAIL a GPU advisory module that returned 0 instead of 77 (silent
    false-PASS). This is the real Rule-16 enforcement that replaced the inert
    standalone-binary path at the CI choke points."""
    tag = "SMOKE:rule16_json"
    path = SCRIPT_DIR / "check_advisory_json_rule16.py"
    if not path.exists():
        skip(tag, "check_advisory_json_rule16.py not found")
        return
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            clean = root / "clean.json"
            clean.write_text(json.dumps({"sections": [{"modules": [
                {"id": "test_exploit_cuda_key_erase", "advisory": True, "passed": False, "return_code": 77},
                {"id": "cryptol_specs", "advisory": True, "passed": False, "return_code": 1},
                {"id": "test_ecdsa_sign", "advisory": False, "passed": True, "return_code": 0},
            ]}]}), encoding="utf-8")
            bad = root / "bad.json"
            bad.write_text(json.dumps({"sections": [{"modules": [
                {"id": "test_exploit_cuda_key_erase", "advisory": True, "passed": True, "return_code": 0},
            ]}]}), encoding="utf-8")
            non_advisory_bad = root / "non_advisory_bad.json"
            non_advisory_bad.write_text(json.dumps({"sections": [{"modules": [
                {"id": "test_ecdsa_sign", "advisory": False, "passed": False, "return_code": 1},
            ]}]}), encoding="utf-8")

            def run(arg: Path) -> int:
                return subprocess.run(
                    [sys.executable, str(path), str(arg)],
                    capture_output=True, text=True, timeout=15, cwd=str(LIB_ROOT),
                ).returncode

            rc_clean = run(clean)
            rc_bad = run(bad)
            rc_non_advisory_bad = run(non_advisory_bad)
        if rc_clean != 0:
            fail(tag, f"clean/advisory-degraded report should pass (exit 0), got {rc_clean}")
            return
        if rc_bad != 1:
            fail(tag, f"GPU advisory false-PASS (0 not 77) should fail (exit 1), got {rc_bad}")
            return
        if rc_non_advisory_bad != 1:
            fail(tag, f"non-advisory failure should fail (exit 1), got {rc_non_advisory_bad}")
            return
        ok(tag, "advisory-degraded report passes; false-PASS and non-advisory failures are caught")
    except subprocess.TimeoutExpired:
        fail(tag, "timed out")
    except Exception as exc:
        fail(tag, str(exc))


def check_caas_integrity_json_purity() -> None:
    """CAAS self-test --json must emit parseable JSON with no text prelude."""
    tag = "SMOKE:caas_integrity_json"
    path = SCRIPT_DIR / "test_caas_integrity.py"
    if not path.exists():
        skip(tag, "test_caas_integrity.py not found")
        return
    try:
        result = subprocess.run(
            [sys.executable, str(path), "--json"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(LIB_ROOT),
            check=False,
        )
    except subprocess.TimeoutExpired:
        fail(tag, "timed out")
        return
    if result.returncode != 0:
        fail(tag, f"test_caas_integrity.py --json exited {result.returncode}: {result.stderr[:200]}")
        return
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        fail(tag, f"--json stdout is not pure JSON: {exc}; prefix={result.stdout[:120]!r}")
        return
    if payload.get("suite") != "test_caas_integrity" or payload.get("failed") != 0:
        fail(tag, f"unexpected CAAS integrity JSON payload: {payload}")
        return
    ok(tag, "test_caas_integrity.py --json emits pure passing JSON")


def _load_audit_gate_module():
    """Import audit_gate.py in-process for white-box negative testing."""
    path = SCRIPT_DIR / "audit_gate.py"
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location("audit_gate_p21_selftest", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("could not build module spec for audit_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_p21_semantic_requirement_map() -> None:
    """B1/P21: the external-audit-replacement gate must be SEMANTIC, not
    presence-only. It must FAIL closed on a bad requirement map — missing
    artifact, gate not registered in CHECK_MAP, stale last_verified, empty
    documented-residual, missing map file — and PASS on a well-formed one and on
    the real committed docs/CAAS_BASTION_REQUIREMENTS.json. This is the negative
    proof that P21 binds each closed gap to a live gate, not to prose."""
    from datetime import datetime, timezone, timedelta

    tag = "P21:semantic_requirement_map"
    try:
        module = _load_audit_gate_module()
    except Exception as exc:
        fail(tag, f"could not import audit_gate.py: {exc}")
        return

    if not hasattr(module, "REQUIREMENTS_PATH") or not hasattr(module, "check_external_audit_replacement"):
        fail(tag, "audit_gate.py is missing REQUIREMENTS_PATH / check_external_audit_replacement (presence-only?)")
        return

    today = datetime.now(timezone.utc).date()
    fresh = today.isoformat()
    stale = (today - timedelta(days=10_000)).isoformat()

    def has_fail(reqmap_obj_or_none, point_at_missing=False) -> bool:
        saved = module.REQUIREMENTS_PATH
        tmp = None
        try:
            if point_at_missing:
                module.REQUIREMENTS_PATH = Path(tempfile.gettempdir()) / "p21_does_not_exist_xyz.json"
            else:
                tf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
                json.dump(reqmap_obj_or_none, tf)
                tf.close()
                tmp = Path(tf.name)
                module.REQUIREMENTS_PATH = tmp
            _, findings = module.check_external_audit_replacement(None)
            return any(f[0] == "FAIL" for f in findings)
        finally:
            module.REQUIREMENTS_PATH = saved
            if tmp is not None:
                tmp.unlink(missing_ok=True)

    def good_map():
        return {
            "schema_version": "1.0.0",
            "sla": {"last_verified_warn_days": 180, "last_verified_fail_days": 540},
            "requirements": [
                {
                    "id": "T-1", "claim": "gated row",
                    "artifact_paths": ["docs/THREAT_MODEL.md"],
                    "gate": "audit_gate.py --threat-model", "gate_kind": "audit_gate",
                    "status": "gated", "residual_risk": "", "last_verified": fresh,
                }
            ],
        }

    failures = []

    # (0) well-formed crafted map must PASS (no FAIL findings)
    if has_fail(good_map()):
        failures.append("well-formed map produced a FAIL")

    # (1) missing artifact -> FAIL
    m = good_map(); m["requirements"][0]["artifact_paths"] = ["docs/__p21_nonexistent__.md"]
    if not has_fail(m):
        failures.append("missing artifact did not FAIL")

    # (2) gate not registered in CHECK_MAP -> FAIL
    m = good_map(); m["requirements"][0]["gate"] = "audit_gate.py --not-a-real-gate-xyz"
    if not has_fail(m):
        failures.append("unregistered gate flag did not FAIL")

    # (3) status=gated but gate_kind not executable -> FAIL
    m = good_map(); m["requirements"][0]["gate_kind"] = "presence"
    if not has_fail(m):
        failures.append("gated row with non-executable gate_kind did not FAIL")

    # (4) stale last_verified beyond SLA -> FAIL
    m = good_map(); m["requirements"][0]["last_verified"] = stale
    if not has_fail(m):
        failures.append("stale last_verified did not FAIL")

    # (5) documented_residual with empty residual_risk -> FAIL
    m = good_map()
    m["requirements"][0].update({"status": "documented_residual", "gate_kind": "presence",
                                 "gate": "presence:x", "residual_risk": ""})
    if not has_fail(m):
        failures.append("empty documented_residual did not FAIL")

    # (6) missing requirement-map file -> FAIL
    if not has_fail(None, point_at_missing=True):
        failures.append("missing CAAS_BASTION_REQUIREMENTS.json did not FAIL")

    # (7) the REAL committed map must PASS
    if has_fail(json.loads((LIB_ROOT / "docs" / "CAAS_BASTION_REQUIREMENTS.json").read_text())):
        failures.append("the committed CAAS_BASTION_REQUIREMENTS.json produced a FAIL")

    if failures:
        fail(tag, "; ".join(failures))
    else:
        ok(tag, "P21 fails closed on 6 bad-map cases and passes the real + well-formed maps")


def check_audit_sla_pre_alert_and_block() -> None:
    """B3: audit_sla_check.py must (a) BLOCK when critical evidence exceeds its
    threshold, (b) emit a non-blocking PRE-ALERT inside the buffer window, (c) be
    OK when fresh, and (d) report days_until_block for every artifact. Proves the
    freshness SLA fails closed at the deadline AND warns before it — no silent
    green->blocked jump (the failure class that dropped autonomy 100->90)."""
    tag = "B3:audit_sla_pre_alert"
    path = SCRIPT_DIR / "audit_sla_check.py"
    try:
        spec = importlib.util.spec_from_file_location("audit_sla_selftest", str(path))
        if spec is None or spec.loader is None:
            fail(tag, "could not build module spec for audit_sla_check.py")
            return
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:
        fail(tag, f"import failed: {exc}")
        return

    sla = {"slos": {"critical_evidence_freshness_days":
                    {"threshold": 14, "pre_alert_buffer_days": 4, "severity": "blocking"}}}

    def with_age(age):
        module._file_age_days = lambda p: age
        module._dir_newest_age_days = lambda p: age
        return module.check_critical_freshness(sla)

    failures = []

    # (a) stale (age 20 > 14) -> a blocking 'stale' finding + negative days_until_block
    f, s = with_age(20.0)
    if not any(x.get("severity") == "blocking" and x.get("status") == "stale" for x in f):
        failures.append("age 20 (>14) did not produce a blocking stale finding")
    if not any(st.get("days_until_block") is not None and st["days_until_block"] < 0 for st in s):
        failures.append("stale artifact missing negative days_until_block")

    # (b) pre-alert (10 < age 12 <= 14) -> warning, NOT blocking
    f, s = with_age(12.0)
    if any(x.get("severity") == "blocking" for x in f):
        failures.append("age 12 (pre-alert window) incorrectly blocked")
    if not any(x.get("status") == "pre_alert" and x.get("severity") == "warning" for x in f):
        failures.append("age 12 did not emit a non-blocking pre_alert warning")

    # (c) fresh (age 2) -> no findings, all statuses 'ok' with days_until_block set
    f, s = with_age(2.0)
    if f:
        failures.append(f"age 2 (fresh) produced findings: {[x.get('status') for x in f]}")
    if not all(st["state"] == "ok" for st in s):
        failures.append("fresh artifacts not classified ok")
    if not all(st.get("days_until_block") is not None for st in s):
        failures.append("fresh statuses missing days_until_block")

    if failures:
        fail(tag, "; ".join(failures))
    else:
        ok(tag, "SLA blocks at deadline, pre-alerts in buffer window, reports days_until_block")


def check_audit_sla_build_report_not_tracked() -> None:
    """B3 regression: risk_surface_report is a build-scoped generated artifact.

    It must not be committed under out/reports/, otherwise audit_sla_check.py
    evaluates the git commit date instead of current build evidence and the
    autonomy gate can fall from 100 to 90 when a stale generated file crosses
    the freshness threshold.
    """
    tag = "B3:audit_sla_build_report_untracked"
    rel = "out/reports/risk_surface_report.json"
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", rel],
        cwd=str(LIB_ROOT),
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", rel],
        cwd=str(LIB_ROOT),
        capture_output=True,
        text=True,
    )
    ignored = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", rel],
        cwd=str(LIB_ROOT),
        capture_output=True,
        text=True,
    )
    failures = []
    pending_delete = status.stdout.startswith("D ") or status.stdout.startswith(" D")
    if tracked.returncode == 0 and not pending_delete:
        failures.append(f"{rel} is tracked but must remain build-only")
    if ignored.returncode != 0:
        failures.append(f"{rel} is not covered by .gitignore")
    if failures:
        fail(tag, "; ".join(failures))
    else:
        ok(tag, "build-scoped risk report is ignored and not tracked")


def check_external_audit_bundle_negative_fixtures() -> None:
    """B4/B5: verify_external_audit_bundle.py must fail closed on a tampered
    digest, a missing evidence file, an evidence hash mismatch, a stale commit
    (unless --allow-commit-mismatch), and a malformed/non-object bundle — and
    pass a well-formed current-commit bundle. Proves the independent-review gold
    path cannot be satisfied by a stale or tampered snapshot."""
    tag = "B4:bundle_negative_fixtures"
    path = SCRIPT_DIR / "verify_external_audit_bundle.py"
    try:
        spec = importlib.util.spec_from_file_location("verify_bundle_selftest", str(path))
        if spec is None or spec.loader is None:
            fail(tag, "could not build module spec for verify_external_audit_bundle.py")
            return
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:
        fail(tag, f"import failed: {exc}")
        return

    real_evidence = "docs/THREAD_SAFETY.md"
    if not (module.LIB_ROOT / real_evidence).exists():
        skip(tag, f"{real_evidence} absent — cannot build a valid evidence row")
        return
    good_hash = module._sha256_file(module.LIB_ROOT / real_evidence)
    head = module._git_head()
    if not head:
        skip(tag, "git HEAD unavailable")
        return

    def build(tmp, *, commit, evidence, tamper_digest=False, raw=None):
        bundle_path = Path(tmp) / "EXTERNAL_AUDIT_BUNDLE.json"
        if raw is not None:
            bundle_path.write_text(raw, encoding="utf-8")
        else:
            obj = {"git": {"commit": commit}, "evidence": evidence, "gate_results": []}
            bundle_path.write_text(json.dumps(obj), encoding="utf-8")
        digest = "0" * 64 if tamper_digest else module._sha256_bytes(bundle_path.read_bytes())
        digest_path = Path(tmp) / "EXTERNAL_AUDIT_BUNDLE.sha256"
        digest_path.write_text(f"{digest}  EXTERNAL_AUDIT_BUNDLE.json\n", encoding="utf-8")
        return bundle_path, digest_path

    def rc(bundle_path, digest_path, allow=False):
        with contextlib.redirect_stdout(io.StringIO()):
            return module.verify(bundle_path, digest_path, False, allow, False)

    good_ev = [{"path": real_evidence, "sha256": good_hash}]
    failures = []

    with tempfile.TemporaryDirectory() as d:
        # (0) well-formed current-commit bundle -> PASS
        bp, dp = build(d, commit=head, evidence=good_ev)
        if rc(bp, dp) != 0:
            failures.append("well-formed current-commit bundle did not PASS")

        # (1) tampered digest -> FAIL
        bp, dp = build(d, commit=head, evidence=good_ev, tamper_digest=True)
        if rc(bp, dp) == 0:
            failures.append("tampered digest did not FAIL")

        # (2) missing evidence file -> FAIL
        bp, dp = build(d, commit=head, evidence=[{"path": "docs/__nope_xyz__.md", "sha256": "0" * 64}])
        if rc(bp, dp) == 0:
            failures.append("missing evidence file did not FAIL")

        # (3) evidence hash mismatch -> FAIL
        bp, dp = build(d, commit=head, evidence=[{"path": real_evidence, "sha256": "0" * 64}])
        if rc(bp, dp) == 0:
            failures.append("evidence hash mismatch did not FAIL")

        # (4) stale commit -> FAIL without flag, PASS with --allow-commit-mismatch
        bp, dp = build(d, commit="0" * 40, evidence=good_ev)
        if rc(bp, dp, allow=False) == 0:
            failures.append("stale commit did not FAIL")
        if rc(bp, dp, allow=True) != 0:
            failures.append("stale commit + allow_commit_mismatch did not PASS")

        # (5) malformed JSON -> FAIL gracefully (no exception)
        bp, dp = build(d, commit=head, evidence=good_ev, raw="{ this is : not json ")
        try:
            if rc(bp, dp) == 0:
                failures.append("malformed JSON bundle did not FAIL")
        except Exception as exc:
            failures.append(f"malformed JSON crashed instead of failing closed: {exc}")

        # (6) non-object JSON root -> FAIL gracefully
        bp, dp = build(d, commit=head, evidence=good_ev, raw="[1, 2, 3]")
        try:
            if rc(bp, dp) == 0:
                failures.append("non-object bundle root did not FAIL")
        except Exception as exc:
            failures.append(f"non-object bundle crashed instead of failing closed: {exc}")

    if failures:
        fail(tag, "; ".join(failures))
    else:
        ok(tag, "bundle verify fails closed on digest/evidence/commit/malformed; passes a valid current bundle")


def _load_ci_module(filename: str, modname: str):
    """Import a ci/*.py module in-process for white-box negative testing."""
    path = SCRIPT_DIR / filename
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(modname, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not build module spec for {filename}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses / self-referential module code resolve.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def check_ct_independence_negative_fixtures() -> None:
    """B5: ct_independence_check.py must NOT report a false-green. One PASS + one
    SKIP is INCONCLUSIVE (exit 2), not PASS; any FAIL is exit 1; a missing
    required tool is exit 1; two DISTINCT PASS is exit 0. This is the negative
    proof of the P7-CAAS-001 false-green fix."""
    tag = "B5:ct_independence"
    script = SCRIPT_DIR / "ct_independence_check.py"
    if not script.exists():
        skip(tag, "ct_independence_check.py absent")
        return

    def run_with(files, *extra):
        return subprocess.run(
            [sys.executable, str(script), *files, *extra, "--json"],
            capture_output=True, text=True, timeout=30, cwd=str(LIB_ROOT)).returncode

    failures = []
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)

        def w(name, tool, verdict):
            p = d / name
            p.write_text(json.dumps({"tool": tool, "methodology": tool, "verdict": verdict}))
            return str(p)

        a_pass = w("a.json", "valgrind-ct", "PASS")
        b_pass = w("b.json", "dudect", "PASS")
        b_skip = w("c.json", "dudect", "SKIP")
        a_fail = w("f.json", "valgrind-ct", "FAIL")

        if run_with([a_pass, b_pass], "--min-tools", "2") != 0:
            failures.append("two distinct PASS was not exit 0 (PASS)")
        if run_with([a_pass, b_skip], "--min-tools", "2") != 2:
            failures.append("PASS+SKIP was not exit 2 (INCONCLUSIVE) — false-green risk")
        if run_with([a_pass, a_fail], "--min-tools", "2") != 1:
            failures.append("a FAIL verdict was not exit 1")
        if run_with([a_pass, b_pass], "--min-tools", "2", "--require-tools", "binsec-rel") != 1:
            failures.append("missing required tool was not exit 1")

    if failures:
        fail(tag, "; ".join(failures))
    else:
        ok(tag, "PASS+SKIP is INCONCLUSIVE not PASS; FAIL/missing-required block; 2 distinct PASS pass")


def check_multi_ci_repro_negative_fixtures() -> None:
    """B5: multi_ci_repro_check.py must FAIL on mismatched hashes, FAIL/error on
    no-common-artifacts, and error on an empty hash file; PASS only on
    bit-identical artifacts across providers."""
    tag = "B5:multi_ci_repro"
    script = SCRIPT_DIR / "multi_ci_repro_check.py"
    if not script.exists():
        skip(tag, "multi_ci_repro_check.py absent")
        return

    def run2(fa, fb):
        return subprocess.run([sys.executable, str(script), fa, fb, "--json"],
                              capture_output=True, text=True, timeout=30, cwd=str(LIB_ROOT)).returncode

    failures = []
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)

        def w(name, content):
            p = d / name
            p.write_text(content)
            return str(p)

        sha = "a" * 64
        other = "b" * 64
        match_a = w("ma.txt", f"{sha}  build/libufsecp.a\n")
        match_b = w("mb.txt", f"{sha}  other/libufsecp.a\n")          # same basename+sha
        diff_b = w("db.txt", f"{other}  build/libufsecp.a\n")         # same basename, diff sha
        nocommon_b = w("nc.txt", f"{sha}  build/libsomethingelse.a\n")  # different basename
        empty = w("empty.txt", "")

        if run2(match_a, match_b) != 0:
            failures.append("bit-identical artifacts were not PASS (exit 0)")
        if run2(match_a, diff_b) == 0:
            failures.append("hash mismatch did not FAIL")
        if run2(match_a, nocommon_b) == 0:
            failures.append("no common artifacts did not FAIL")
        if run2(match_a, empty) == 0:
            failures.append("empty hash file did not error/FAIL")

    if failures:
        fail(tag, "; ".join(failures))
    else:
        ok(tag, "fails on mismatch/no-common/empty; passes only bit-identical artifacts")


def check_security_autonomy_forced_failure() -> None:
    """B5: security_autonomy_check.py must drop autonomy_ready=false (and exit 1)
    when any active sub-gate fails, return exit 77 when all gates advisory-skip,
    and report ready/100 only when all gates pass. Tests the scoring logic with a
    monkeypatched gate runner (no real sub-gates run; KPI write redirected)."""
    tag = "B5:security_autonomy"
    try:
        module = _load_ci_module("security_autonomy_check.py", "autonomy_selftest")
    except Exception as exc:
        fail(tag, f"import failed: {exc}")
        return

    saved_lib, saved_runner = module.LIB_ROOT, module._run_gate
    failures = []
    try:
        with tempfile.TemporaryDirectory() as d:
            module.LIB_ROOT = Path(d)  # redirect the KPI write away from the real file

            def all_pass(g, timeout=300):
                return {"gate": g["name"], "weight": g["weight"], "status": "ran",
                        "passing": True, "score": g["weight"], "returncode": 0}

            def one_fail(g, timeout=300):
                p = g["name"] != "audit_sla"
                return {"gate": g["name"], "weight": g["weight"], "status": "ran",
                        "passing": p, "score": g["weight"] if p else 0, "returncode": 0 if p else 1}

            def all_skip(g, timeout=300):
                return {"gate": g["name"], "weight": g["weight"], "status": "advisory_skip",
                        "passing": False, "score": 0, "advisory_skip": True,
                        "did_run": False, "returncode": 77}

            def rep(fn, outname):
                module._run_gate = fn
                outp = Path(d) / outname
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = module.run(False, str(outp), 1)
                return rc, json.loads(outp.read_text())

            rc, r = rep(all_pass, "pass.json")
            if not (rc == 0 and r["autonomy_ready"] and r["autonomy_score"] == 100):
                failures.append("all-pass was not ready/score-100/exit-0")

            rc, r = rep(one_fail, "fail.json")
            if not (rc == 1 and not r["autonomy_ready"] and r["autonomy_score"] < 100 and not r["overall_pass"]):
                failures.append("forced sub-gate failure still reported ready")

            rc, r = rep(all_skip, "skip.json")
            if not (rc == 77 and r["advisory_skip_all"] and not r["autonomy_ready"]):
                failures.append("all-advisory-skip was not exit-77/advisory_skip_all")
    finally:
        module.LIB_ROOT, module._run_gate = saved_lib, saved_runner

    if failures:
        fail(tag, "; ".join(failures))
    else:
        ok(tag, "ready only on all-pass; one failure blocks; all-skip => exit 77 (not false 100)")


def check_supply_chain_negative_fixtures() -> None:
    """B5: supply_chain_gate.py must FAIL when build inputs are unpinned and
    provenance/SBOM/hash artifacts are absent (empty tree), and PASS on the real
    repo. Proves the 5-sub-gate trust chain is fail-closed, not presence-cosmetic."""
    tag = "B5:supply_chain"
    try:
        module = _load_ci_module("supply_chain_gate.py", "supply_chain_selftest")
    except Exception as exc:
        fail(tag, f"import failed: {exc}")
        return

    saved_lib, saved_sd = module.LIB_ROOT, module.SCRIPT_DIR
    failures = []
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            rc_good = module.run(False, None)
        if rc_good != 0:
            failures.append("real-repo supply-chain gate did not PASS")

        with tempfile.TemporaryDirectory() as d:
            module.LIB_ROOT = Path(d)
            module.SCRIPT_DIR = Path(d)  # no CMakeLists, no scripts, no provenance
            with contextlib.redirect_stdout(io.StringIO()):
                rc_bad = module.run(False, None)
        if rc_bad == 0:
            failures.append("empty-tree (no pinning/provenance/hash) did not FAIL")
    finally:
        module.LIB_ROOT, module.SCRIPT_DIR = saved_lib, saved_sd

    if failures:
        fail(tag, "; ".join(failures))
    else:
        ok(tag, "fails closed when pinning/provenance/SBOM/hardening absent; passes real repo")


def check_source_graph_quality_negative_fixtures() -> None:
    """B5: check_source_graph_quality.py must FAIL (exit 1) on an empty/stale
    graph DB (missing tables, below row floor, no build revision). Proves the
    graph-quality gate cannot false-green on a broken database."""
    tag = "B5:source_graph_quality"
    script = SCRIPT_DIR / "check_source_graph_quality.py"
    if not script.exists():
        skip(tag, "check_source_graph_quality.py absent")
        return
    failures = []
    with tempfile.TemporaryDirectory() as d:
        empty = Path(d) / "empty.db"
        import sqlite3
        sqlite3.connect(str(empty)).close()
        rc = subprocess.run([sys.executable, str(script), "--db", str(empty)],
                            capture_output=True, text=True, timeout=60, cwd=str(LIB_ROOT)).returncode
        if rc == 0:
            failures.append("empty graph DB did not FAIL (exit 0) — false-green risk")
    if failures:
        fail(tag, "; ".join(failures))
    else:
        ok(tag, "empty/broken graph DB fails closed (exit 1)")


def check_bench_artifact_sanity_fixtures() -> None:
    """B8: check_bench_doc_consistency.check_bench_artifact_sanity must reject
    corrupt benchmark artifacts — zero / negative / non-finite / non-numeric /
    sub-physical ns and a non-list results array — and accept a clean one. A perf
    claim sourced from a corrupt artifact is not evidence."""
    tag = "B8:bench_artifact_sanity"
    try:
        module = _load_ci_module("check_bench_doc_consistency.py", "bench_consistency_selftest")
    except Exception as exc:
        fail(tag, f"import failed: {exc}")
        return
    if not hasattr(module, "check_bench_artifact_sanity") or not hasattr(module, "_coerce_ns"):
        fail(tag, "check_bench_doc_consistency.py missing artifact-sanity hardening")
        return

    san = module.check_bench_artifact_sanity

    def flagged(results) -> bool:
        return bool(san({"results": results}, "fixture.json"))

    failures = []
    if flagged([{"section": "FIELD", "name": "mul", "ns": 12.3}]):
        failures.append("clean artifact was flagged")
    for desc, ns in [("zero", 0), ("negative", -5), ("inf", float("inf")),
                     ("nan", float("nan")), ("non-numeric", "123ns456"),
                     ("sub-physical", 0.001), ("boolean", True)]:
        if not flagged([{"section": "X", "name": "a", "ns": ns}]):
            failures.append(f"{desc} ns was not flagged")
    if not san({"results": "not-a-list"}, "f.json"):
        failures.append("non-list results was not flagged")
    # _coerce_ns: invalid -> None, valid -> float
    if module._coerce_ns(0) is not None or module._coerce_ns("123ns456") is not None:
        failures.append("_coerce_ns did not reject invalid ns")
    if module._coerce_ns(12.3) != 12.3:
        failures.append("_coerce_ns rejected a valid ns")

    if failures:
        fail(tag, "; ".join(failures))
    else:
        ok(tag, "rejects zero/neg/inf/nan/non-numeric/sub-physical ns + non-list results; accepts clean")


def check_incident_drills_real_injection() -> None:
    """B9: the CI-poisoning and dependency-compromise drills must perform REAL
    fault injection and assert the corresponding gate DETECTS it (detected=True
    with injected_fault + detection_gate provenance) — not just check that a
    script exists. A drill that cannot fail is not a drill."""
    tag = "B9:incident_drills_real_injection"
    try:
        module = _load_ci_module("incident_drills.py", "incident_drills_selftest")
    except Exception as exc:
        fail(tag, f"import failed: {exc}")
        return

    failures = []
    for fn_name in ("drill_ci_poisoning", "drill_dependency_compromise"):
        fn = getattr(module, fn_name, None)
        if fn is None:
            failures.append(f"{fn_name} missing")
            continue
        r = fn()
        if not r.get("detected"):
            failures.append(f"{r.get('drill', fn_name)} did not DETECT its injected fault")
        if not r.get("passing"):
            failures.append(f"{r.get('drill', fn_name)} drill did not pass")
        if not r.get("injected_fault") or not r.get("detection_gate"):
            failures.append(f"{r.get('drill', fn_name)} missing injected_fault/detection_gate provenance")

    if failures:
        fail(tag, "; ".join(failures))
    else:
        ok(tag, "ci_poisoning + dependency_compromise inject real faults and the gate detects them")


def check_research_monitor_actionable_body() -> None:
    """B6: a high-confidence research finding must render an Affected Surface, the
    existing-evidence paths, and a Patch plan with a first-verification command —
    turning a signal into actionable work, not a bare citation. An empty report
    must render cleanly (no crash)."""
    tag = "B6:research_monitor_actionable"
    try:
        module = _load_ci_module("research_monitor.py", "research_monitor_body_selftest")
    except Exception as exc:
        fail(tag, f"import failed: {exc}")
        return
    if not hasattr(module, "render_markdown"):
        fail(tag, "research_monitor.py missing render_markdown")
        return

    report = {
        "generated_at": "2026-06-13T00:00:00Z", "query": "secp256k1 nonce",
        "lookback_days": 14,
        "counts": {"high_confidence": 1, "needs_review": 0, "discarded": 0, "total_fetched": 1},
        "sources": [], "source_errors": [],
        "items": [{
            "bucket": "high_confidence", "title": "New ECDSA nonce-bias result",
            "source": "eprint", "score": 12, "status": "gap", "action": "monitor",
            "published": "2026-06-10", "url": "https://eprint.iacr.org/2026/123",
            "reason": "matched nonce-bias keyword", "summary": "lattice attack on biased nonces",
            "matches": [{"id": "SIG-NONCE", "status": "gap",
                         "repo_evidence": ["audit/test_nonce_bias.cpp"]}],
        }],
    }
    failures = []
    md = module.render_markdown(report)
    for token in ("Affected surface", "Patch plan", "First verification",
                  "audit/test_nonce_bias.cpp"):
        if token not in md:
            failures.append(f"rendered finding missing '{token}'")

    empty = dict(report, items=[],
                 counts={"high_confidence": 0, "needs_review": 0, "discarded": 0, "total_fetched": 0})
    try:
        emd = module.render_markdown(empty)
        if "No high-confidence findings" not in emd:
            failures.append("empty report missing the no-findings line")
    except Exception as exc:
        failures.append(f"empty report crashed: {exc}")

    if failures:
        fail(tag, "; ".join(failures))
    else:
        ok(tag, "high-conf findings render affected-surface + evidence + patch-plan; empty report is clean")


def check_integration_evidence_fixtures() -> None:
    """B13: the integration-evidence gate must FAIL a `blocking` row with missing
    evidence / stale last_verified / malformed date, treat `warning` staleness as
    advisory, never silently count an `owner_gated` row as current evidence, and
    PASS a valid manifest (and the real committed one). Time-independent via an
    injected `today`."""
    from datetime import date
    tag = "B13:integration_evidence"
    try:
        module = _load_ci_module("check_integration_evidence.py", "integration_evidence_selftest")
    except Exception as exc:
        fail(tag, f"import failed: {exc}")
        return

    today = date(2026, 6, 13)

    def row(**kw):
        base = {"id": "R", "surface": "s",
                "evidence_path": "ci/check_integration_evidence.py",
                "reproduce_command": "x", "freshness_days": 30, "severity": "blocking",
                "last_verified": "2026-06-13", "status": "pass", "notes": ""}
        base.update(kw)
        return base

    def ev(rows):
        return module.evaluate({"default_pre_alert_buffer_days": 7, "rows": rows}, today=today)

    failures = []

    # valid blocking row -> pass
    if not ev([row()])["overall_pass"]:
        failures.append("valid blocking row did not pass")

    # missing evidence (blocking) -> FAIL + listed
    r = ev([row(evidence_path="docs/__nope_xyz__.md")])
    if r["overall_pass"] or "R" not in r["missing_rows"]:
        failures.append("missing blocking evidence did not fail")

    # stale (blocking, far-past date) -> FAIL + listed
    r = ev([row(last_verified="2026-01-01")])
    if r["overall_pass"] or "R" not in r["stale_rows"]:
        failures.append("stale blocking row did not fail")

    # malformed date (blocking) -> FAIL
    if ev([row(last_verified="not-a-date")])["overall_pass"]:
        failures.append("malformed date on blocking row did not fail")

    # warning staleness -> advisory (overall pass) but listed in stale_rows
    r = ev([row(severity="warning", last_verified="2026-01-01")])
    if not r["overall_pass"] or "R" not in r["stale_rows"]:
        failures.append("warning staleness should be advisory and listed")

    # owner_gated -> never blocks, listed explicitly, NOT counted as pass
    r = ev([row(severity="owner_gated", last_verified="2026-01-01",
                evidence_path="docs/__nope_xyz__.md")])
    if not r["overall_pass"]:
        failures.append("owner_gated row blocked the gate")
    if "R" not in r["owner_gated_rows"]:
        failures.append("owner_gated row not listed explicitly")
    if r["rows"][0]["computed_status"] == "pass":
        failures.append("owner_gated row was silently counted as current (pass)")

    # the real committed manifest must pass
    rep, _ = module.load_and_evaluate(module.MANIFEST_PATH, today=today)
    if not rep.get("overall_pass"):
        failures.append("the committed INTEGRATION_EVIDENCE_STATUS.json did not pass")

    if failures:
        fail(tag, "; ".join(failures))
    else:
        ok(tag, "blocking missing/stale/malformed fail; warning advisory; owner_gated explicit (never pass); real manifest passes")


def check_ct_evidence_status_fixtures() -> None:
    """B14: the CT-evidence gate must FAIL a blocking row with missing evidence /
    stale or malformed last_verified; when tool verdicts ARE evaluated, a required
    tool FAIL or a single PASS + SKIP is INCONCLUSIVE (never pass) and blocks; an
    owner_gated row is explicit and never counted as current; and a valid manifest
    (push path, no verdicts) passes. Time-independent via injected `today`."""
    from datetime import date
    tag = "B14:ct_evidence_status"
    try:
        module = _load_ci_module("check_ct_evidence_status.py", "ct_evidence_selftest")
    except Exception as exc:
        fail(tag, f"import failed: {exc}")
        return

    today = date(2026, 6, 13)

    def row(**kw):
        base = {"id": "R", "surface": "s", "ct_claim": "c",
                "evidence_paths": ["ci/check_ct_evidence_status.py"],
                "required_tools": ["valgrind-ct", "ct-verif"], "optional_tools": [],
                "freshness_days": 90, "severity": "blocking",
                "last_verified": "2026-06-13", "status": "pass", "notes": ""}
        base.update(kw)
        return base

    def ev(rows, verdicts=None):
        return module.evaluate({"default_pre_alert_buffer_days": 14, "rows": rows},
                               today=today, verdicts=verdicts)

    failures = []

    # valid blocking row, push path (no verdicts) -> pass
    if not ev([row()])["overall_pass"]:
        failures.append("valid blocking row (no verdicts) did not pass")

    # missing evidence -> fail + listed
    r = ev([row(evidence_paths=["docs/__nope_xyz__.md"])])
    if r["overall_pass"] or "R" not in r["missing_rows"]:
        failures.append("missing evidence did not fail")

    # stale -> fail + listed
    r = ev([row(last_verified="2026-01-01")])
    if r["overall_pass"] or "R" not in r["stale_rows"]:
        failures.append("stale blocking row did not fail")

    # malformed date -> fail
    if ev([row(last_verified="not-a-date")])["overall_pass"]:
        failures.append("malformed date did not fail")

    # required tool SKIP (verdicts evaluated) -> INCONCLUSIVE (never pass) + blocking fail
    r = ev([row()], verdicts={"valgrind-ct": "PASS", "ct-verif": "SKIP"})
    if r["overall_pass"] or "R" not in r["inconclusive_rows"]:
        failures.append("PASS+SKIP was not inconclusive/blocking")

    # required tool FAIL -> blocking fail
    if ev([row()], verdicts={"valgrind-ct": "FAIL", "ct-verif": "PASS"})["overall_pass"]:
        failures.append("required tool FAIL did not block")

    # all required PASS -> pass
    if not ev([row()], verdicts={"valgrind-ct": "PASS", "ct-verif": "PASS"})["overall_pass"]:
        failures.append("all required tools PASS did not pass")

    # warning staleness -> advisory + listed
    r = ev([row(severity="warning", last_verified="2026-01-01")])
    if not r["overall_pass"] or "R" not in r["stale_rows"]:
        failures.append("warning staleness should be advisory and listed")

    # owner_gated -> explicit, never current, never blocks
    r = ev([row(severity="owner_gated", last_verified="2026-01-01",
                evidence_paths=["docs/__nope_xyz__.md"])])
    if not r["overall_pass"]:
        failures.append("owner_gated row blocked the gate")
    if "R" not in r["owner_gated_rows"]:
        failures.append("owner_gated row not listed explicitly")
    if r["rows"][0]["computed_status"] == "pass":
        failures.append("owner_gated row was silently counted as current (pass)")

    # the real committed manifest must pass (push path)
    rep, _ = module.load_and_evaluate(module.MANIFEST_PATH, today=today)
    if not rep.get("overall_pass"):
        failures.append("the committed CT_EVIDENCE_STATUS.json did not pass")

    if failures:
        fail(tag, "; ".join(failures))
    else:
        ok(tag, "missing/stale/malformed blocking fail; PASS+SKIP inconclusive+block; FAIL blocks; owner_gated explicit; real manifest passes")


def check_fuzz_campaign_status_fixtures() -> None:
    """B15: the fuzz-campaign gate must FAIL a blocking row with missing corpus /
    stale or malformed last_verified, and a crash artifact without a matching
    regression (crash_unconverted) for ANY non-owner severity; warning staleness is
    advisory; owner_gated is explicit and never current; a valid manifest passes."""
    from datetime import date
    tag = "B15:fuzz_campaign_status"
    try:
        module = _load_ci_module("check_fuzz_campaign_status.py", "fuzz_campaign_selftest")
    except Exception as exc:
        fail(tag, f"import failed: {exc}")
        return

    today = date(2026, 6, 13)
    present = "ci/check_fuzz_campaign_status.py"  # a committed file that exists

    def row(**kw):
        base = {"id": "R", "target": "t", "corpus_path": present, "crash_path": "",
                "regression_path": present, "replay_command": "x", "freshness_days": 90,
                "severity": "blocking", "last_verified": "2026-06-13", "status": "pass", "notes": ""}
        base.update(kw)
        return base

    def ev(rows):
        return module.evaluate({"default_pre_alert_buffer_days": 14, "rows": rows}, today=today)

    failures = []

    if not ev([row()])["overall_pass"]:
        failures.append("valid blocking row did not pass")

    r = ev([row(corpus_path="docs/__nope_xyz__.md")])
    if r["overall_pass"] or "R" not in r["missing_rows"]:
        failures.append("missing corpus did not fail")

    r = ev([row(last_verified="2026-01-01")])
    if r["overall_pass"] or "R" not in r["stale_rows"]:
        failures.append("stale blocking row did not fail")

    if ev([row(last_verified="not-a-date")])["overall_pass"]:
        failures.append("malformed date did not fail")

    # crash artifact without regression -> crash_unconverted + blocking fail
    with tempfile.TemporaryDirectory() as d:
        cd = Path(d) / "crashes"
        cd.mkdir()
        (cd / "crash-deadbeef").write_text("poc")
        r = ev([row(crash_path=str(cd), regression_path="docs/__no_regression_xyz__.md")])
        if r["overall_pass"] or "R" not in r["crash_unconverted_rows"]:
            failures.append("unconverted crash did not fail")
        # crash WITH a present regression -> converted -> pass
        if not ev([row(crash_path=str(cd), regression_path=present)])["overall_pass"]:
            failures.append("converted crash (regression present) did not pass")
        # a warning row with an unconverted crash STILL blocks (correctness gap)
        if ev([row(severity="warning", crash_path=str(cd),
                   regression_path="docs/__no_xyz__.md")])["overall_pass"]:
            failures.append("warning row with unconverted crash did not block")

    # warning staleness -> advisory + listed
    r = ev([row(severity="warning", last_verified="2026-01-01")])
    if not r["overall_pass"] or "R" not in r["stale_rows"]:
        failures.append("warning staleness should be advisory and listed")

    # owner_gated -> explicit, never current, never blocks (on missing/stale)
    r = ev([row(severity="owner_gated", last_verified="2026-01-01",
                corpus_path="docs/__nope_xyz__.md")])
    if not r["overall_pass"]:
        failures.append("owner_gated row blocked the gate")
    if "R" not in r["owner_gated_rows"]:
        failures.append("owner_gated row not listed explicitly")
    if r["rows"][0]["computed_status"] == "pass":
        failures.append("owner_gated row was silently counted as current (pass)")

    rep, _ = module.load_and_evaluate(module.MANIFEST_PATH, today=today)
    if not rep.get("overall_pass"):
        failures.append("the committed FUZZ_CAMPAIGN_STATUS.json did not pass")

    if failures:
        fail(tag, "; ".join(failures))
    else:
        ok(tag, "missing/stale/malformed blocking fail; unconverted-crash blocks (incl. warning); owner_gated explicit; real manifest passes")


def check_gpu_hardware_evidence_fixtures() -> None:
    """B16: the GPU/hardware-evidence gate must FAIL a blocking row with missing
    evidence / stale or malformed last_verified; FAIL a documented_residual whose
    RR-id does not resolve; FAIL a `performance` row that names a fallback_path
    (fallback mislabeled as native) and a `fallback_correctness` row without a
    fallback_path; treat owner_gated as explicit/never-current; and pass a valid +
    the real manifest. Time-independent via injected today + rr_ids."""
    from datetime import date
    tag = "B16:gpu_hardware_evidence"
    try:
        module = _load_ci_module("check_gpu_hardware_evidence.py", "gpu_hw_selftest")
    except Exception as exc:
        fail(tag, f"import failed: {exc}")
        return

    today = date(2026, 6, 13)
    rr_ids = {"RR-005"}  # controlled residual register for crafted rows
    present = "ci/check_gpu_hardware_evidence.py"

    def row(**kw):
        base = {"id": "R", "backend": "cuda", "operation": "op", "claim_type": "correctness",
                "evidence_path": present, "replay_command": "x", "hardware_required": False,
                "freshness_days": 90, "severity": "blocking", "last_verified": "2026-06-13",
                "status": "pass", "notes": ""}
        base.update(kw)
        return base

    def ev(rows):
        return module.evaluate({"default_pre_alert_buffer_days": 14, "rows": rows},
                               today=today, rr_ids=rr_ids)

    failures = []

    if not ev([row()])["overall_pass"]:
        failures.append("valid blocking row did not pass")

    r = ev([row(evidence_path="docs/__nope_xyz__.md")])
    if r["overall_pass"] or "R" not in r["missing_rows"]:
        failures.append("missing evidence did not fail")

    r = ev([row(last_verified="2026-01-01")])
    if r["overall_pass"] or "R" not in r["stale_rows"]:
        failures.append("stale blocking row did not fail")

    if ev([row(last_verified="not-a-date")])["overall_pass"]:
        failures.append("malformed date did not fail")

    # documented_residual with an unresolved RR-id -> fail + listed
    r = ev([row(severity="documented_residual", residual_risk_id="RR-DOESNOTEXIST")])
    if r["overall_pass"] or "R" not in r["unresolved_residual_rows"]:
        failures.append("unresolved documented_residual did not fail")

    # documented_residual with a resolved RR-id -> pass + listed
    r = ev([row(severity="documented_residual", residual_risk_id="RR-005")])
    if not r["overall_pass"] or "R" not in r["documented_residual_rows"]:
        failures.append("resolved documented_residual did not pass/list")

    # owner_gated -> explicit, never current, never blocks
    r = ev([row(severity="owner_gated", last_verified="2026-01-01", evidence_path="docs/__nope_xyz__.md")])
    if not r["overall_pass"]:
        failures.append("owner_gated row blocked")
    if "R" not in r["owner_gated_rows"]:
        failures.append("owner_gated not listed")
    if r["rows"][0]["computed_status"] == "pass":
        failures.append("owner_gated counted as current (pass)")

    # fallback mislabeled as native performance -> fail
    if ev([row(claim_type="performance", fallback_path=present)])["overall_pass"]:
        failures.append("performance row naming a fallback_path did not fail")

    # fallback_correctness without a fallback_path -> fail
    if ev([row(claim_type="fallback_correctness", fallback_path="")])["overall_pass"]:
        failures.append("fallback_correctness without fallback_path did not fail")

    # warning staleness -> advisory + listed
    r = ev([row(severity="warning", last_verified="2026-01-01")])
    if not r["overall_pass"] or "R" not in r["stale_rows"]:
        failures.append("warning staleness should be advisory and listed")

    rep, _ = module.load_and_evaluate(module.MANIFEST_PATH, today=today)
    if not rep.get("overall_pass"):
        failures.append("the committed GPU_HARDWARE_EVIDENCE_STATUS.json did not pass")

    if failures:
        fail(tag, "; ".join(failures))
    else:
        ok(tag, "blocking missing/stale/malformed fail; unresolved residual fails; fallback-vs-performance mislabel fails; owner_gated explicit; real manifest passes")


def check_bench_target_context_fixtures() -> None:
    """B17: the benchmark target-context gate must FAIL a missing/invalid
    target_context, a timed artifact without a claim_scope, a missing
    security_gate_dependency, a gpu_public_data benchmark claiming native hardware
    performance, a bitcoin_core/libbitcoin claim lacking integration_evidence, and
    an unknown_owner_gated context without an explicit owner_gated note; an
    owner_gated context with a note is visible (not current proof) and passes; the
    real canonical artifacts pass."""
    tag = "B17:bench_target_context"
    try:
        module = _load_ci_module("check_bench_target_context.py", "bench_context_selftest")
    except Exception as exc:
        fail(tag, f"import failed: {exc}")
        return

    enum = ["microbench", "batch_verify", "bitcoin_core", "libbitcoin", "gpu_public_data",
            "gpu_hardware", "wasm", "package_integration", "unknown_owner_gated"]

    def row(**kw):
        base = {"id": "R", "target_context": "microbench", "operation": "op", "claim_scope": "scope",
                "evidence_path": "docs/x.json", "security_gate_dependency": "CT green",
                "integration_evidence": None, "native_hardware_claim": False, "notes": "",
                "has_timings": True}
        base.update(kw)
        return base

    def ev(rows):
        return module.evaluate(rows, enum=enum)

    failures = []

    if not ev([row()])["overall_pass"]:
        failures.append("valid microbench row did not pass")

    r = ev([row(target_context=None)])
    if r["overall_pass"] or "R" not in r["missing_context_rows"]:
        failures.append("missing target_context did not fail")

    r = ev([row(target_context="bogus_ctx")])
    if r["overall_pass"] or "R" not in r["invalid_context_rows"]:
        failures.append("invalid target_context did not fail")

    r = ev([row(claim_scope=None)])
    if r["overall_pass"] or "R" not in r["scope_mismatch_rows"]:
        failures.append("timed artifact without claim_scope did not fail")

    if ev([row(security_gate_dependency=None)])["overall_pass"]:
        failures.append("missing security_gate_dependency did not fail")

    r = ev([row(target_context="gpu_public_data", native_hardware_claim=True)])
    if r["overall_pass"] or "R" not in r["scope_mismatch_rows"]:
        failures.append("gpu_public_data claiming native hardware perf did not fail")

    if ev([row(target_context="bitcoin_core", integration_evidence=None)])["overall_pass"]:
        failures.append("bitcoin_core without integration_evidence did not fail")
    if ev([row(target_context="libbitcoin", integration_evidence=None)])["overall_pass"]:
        failures.append("libbitcoin without integration_evidence did not fail")

    if ev([row(target_context="unknown_owner_gated", notes="")])["overall_pass"]:
        failures.append("unknown_owner_gated without owner_gated note did not fail")

    r = ev([row(target_context="unknown_owner_gated", claim_scope="owner device run",
                notes="owner_gated: device-only, not current proof")])
    if not r["overall_pass"] or "R" not in r["owner_gated_rows"]:
        failures.append("owner_gated context with note should pass + be listed")

    rep, _ = module.load_and_evaluate()
    if not rep.get("overall_pass"):
        failures.append("the canonical bench artifacts did not pass the context gate")

    if failures:
        fail(tag, "; ".join(failures))
    else:
        ok(tag, "missing/invalid context + scope/gpu-native/integration mismatches fail; owner_gated explicit; real artifacts pass")


def check_research_signal_matrix_fixtures() -> None:
    """B18: the research signal-matrix gate must FAIL a missing/invalid attack_class,
    a covered class with a missing expected_evidence path or an unresolved
    expected_gate, a candidate without a missing_evidence_action, and an out_of_scope
    class without rationale; the real matrix passes; and the rendered research report
    includes attack_class + affected_surface + expected_gate + a patch-plan."""
    tag = "B18:research_signal_matrix"
    try:
        mod = _load_ci_module("check_research_signal_matrix.py", "rsm_selftest")
    except Exception as exc:
        fail(tag, f"import failed: {exc}")
        return

    enum = ["nonce_bias_or_reuse", "side_channel_ct", "out_of_scope", "batch_verification", "parser_boundary"]
    flags = {"--exploit-traceability"}
    present = "ci/check_research_signal_matrix.py"

    def cls(**kw):
        base = {"id": "R", "status": "covered", "attack_class": "nonce_bias_or_reuse",
                "expected_evidence": [present], "expected_gate": "audit_gate.py --exploit-traceability",
                "missing_evidence_action": "", "reason": "r", "rationale": ""}
        base.update(kw)
        return base

    def ev(classes):
        return mod.evaluate(classes, enum, flags)

    failures = []

    if not ev([cls()])["overall_pass"]:
        failures.append("valid covered class did not pass")
    r = ev([cls(attack_class=None)])
    if r["overall_pass"] or "R" not in r["missing_attack_class"]:
        failures.append("missing attack_class did not fail")
    r = ev([cls(attack_class="bogus_class")])
    if r["overall_pass"] or "R" not in r["invalid_attack_class"]:
        failures.append("invalid attack_class did not fail")
    r = ev([cls(expected_evidence=["docs/__nope_xyz__.md"])])
    if r["overall_pass"] or "R" not in r["missing_evidence"]:
        failures.append("covered with missing expected_evidence did not fail")
    r = ev([cls(expected_gate="audit_gate.py --not-a-real-flag")])
    if r["overall_pass"] or "R" not in r["unresolved_gate"]:
        failures.append("covered with unresolved expected_gate did not fail")
    r = ev([cls(status="candidate", missing_evidence_action="", expected_evidence=[])])
    if r["overall_pass"] or "R" not in r["missing_action"]:
        failures.append("candidate without missing_evidence_action did not fail")
    if not ev([cls(status="candidate", missing_evidence_action="add a PoC", expected_evidence=[])])["overall_pass"]:
        failures.append("candidate with action did not pass")
    if ev([cls(status="out_of_scope", reason="", rationale="", expected_evidence=[])])["overall_pass"]:
        failures.append("out_of_scope without rationale did not fail")
    if not ev([cls(status="out_of_scope", rationale="deferred to operating-env auditors",
                   expected_evidence=["docs/__nope_xyz__.md"])])["overall_pass"]:
        failures.append("out_of_scope (evidence-exempt) with rationale did not pass")

    rep, _ = mod.load_and_evaluate()
    if not rep.get("overall_pass"):
        failures.append("the committed RESEARCH_SIGNAL_MATRIX.json did not pass")

    # rendered report must surface attack_class + affected_surface + expected_gate + patch-plan
    try:
        rm = _load_ci_module("research_monitor.py", "rm_render_selftest")
        report = {
            "generated_at": "x", "query": "q", "lookback_days": 14,
            "counts": {"high_confidence": 1, "needs_review": 0, "discarded": 0, "total_fetched": 1},
            "sources": [], "source_errors": [],
            "items": [{"bucket": "high_confidence", "title": "nonce-bias result", "source": "eprint",
                       "score": 12, "status": "candidate", "action": "review", "published": "2026-06-13",
                       "url": "https://eprint.iacr.org/2026/1", "reason": "matched", "summary": "...",
                       "matches": [{"id": "ladderleak_subbit_nonce", "status": "covered",
                                    "repo_evidence": ["audit/test_exploit_ladderleak_subbit_nonce.cpp"],
                                    "attack_class": "nonce_bias_or_reuse",
                                    "affected_primitive": "rfc6979_nonce",
                                    "affected_surface": "signing nonce derivation",
                                    "expected_gate": "audit_gate.py --exploit-traceability",
                                    "missing_evidence_action": "monitor: re-run PoC"}]}],
        }
        md = rm.render_markdown(report)
        for token in ("Attack class", "Affected surface", "Expected gate", "Patch plan",
                      "nonce_bias_or_reuse", "monitor: re-run PoC"):
            if token not in md:
                failures.append(f"rendered report missing '{token}'")
    except Exception as exc:
        failures.append(f"render check failed: {exc}")

    if failures:
        fail(tag, "; ".join(failures))
    else:
        ok(tag, "missing/invalid attack_class + missing evidence/unresolved gate/missing action fail; real matrix passes; render routes attack_class+surface+gate+patch-plan")


def check_evidence_refresh_coverage_fixtures() -> None:
    """B19: the evidence-refresh coverage gate must FAIL a blocking freshness
    artifact whose `auto` producer does not actually commit its path, a malformed
    refresh mode, a `residual` whose id does not resolve, and a tracked artifact
    with no disposition at all; the real manifest passes and reports the
    incident_drill_log as auto-refreshed. It also proves the RR-BAS-02 promotion
    is safe: audit_sla_check.py BLOCKS a simulated stale drill log under blocking
    severity, while the live advisory (warning) severity only warns — never a
    false pass."""
    tag = "B19:evidence_refresh_coverage"
    try:
        mod = _load_ci_module("check_evidence_refresh_coverage.py", "erc_selftest")
    except Exception as exc:
        fail(tag, f"import failed: {exc}")
        return

    ev = mod.evaluate
    tracked = {"assurance_report", "incident_drill_log", "ct_evidence"}
    committed = {"assurance_report.json", "docs/INCIDENT_DRILL_LOG.json"}
    rids = {"RR-BAS-01"}
    wfs = {".github/workflows/caas-evidence-refresh.yml"}

    def auto(name, path, sev="blocking"):
        return {"name": name, "severity": sev, "refresh": {
            "mode": "auto", "producer_workflow": ".github/workflows/caas-evidence-refresh.yml",
            "committed_paths": [path]}}

    def residual(name, rid="RR-BAS-01", sev="blocking", reason="manual chore"):
        return {"name": name, "severity": sev,
                "refresh": {"mode": "residual", "residual_id": rid, "reason": reason}}

    healthy = [auto("assurance_report", "assurance_report.json"),
               auto("incident_drill_log", "docs/INCIDENT_DRILL_LOG.json", sev="warning"),
               residual("ct_evidence")]

    failures = []

    r = ev(healthy, tracked, committed, rids, wfs)
    if not r["overall_pass"]:
        failures.append(f"healthy manifest did not pass: {r}")
    if not r["incident_drill_autorefresh"]:
        failures.append("healthy manifest: incident_drill_autorefresh not detected")

    # (1) missing refresh producer for a blocking artifact -> fail
    bad = [auto("assurance_report", "NOT_COMMITTED_BY_LANE.json")] + healthy[1:]
    r = ev(bad, tracked, committed, rids, wfs)
    if r["overall_pass"] or "assurance_report" not in r["missing_producer"]:
        failures.append("auto producer not committed by workflow did not fail")

    # (2) malformed refresh status -> fail
    bad = [dict(healthy[0], refresh={"mode": "sometimes"})] + healthy[1:]
    r = ev(bad, tracked, committed, rids, wfs)
    if r["overall_pass"] or "assurance_report" not in r["malformed"]:
        failures.append("malformed refresh mode did not fail")

    # (3) residual with an unresolved id -> fail
    bad = healthy[:2] + [residual("ct_evidence", rid="RR-DOES-NOT-EXIST")]
    r = ev(bad, tracked, committed, rids, wfs)
    if r["overall_pass"] or "ct_evidence" not in r["unresolved_residual"]:
        failures.append("unresolved residual id did not fail")

    # (4) tracked artifact missing a disposition -> fail
    r = ev(healthy[:2], tracked, committed, rids, wfs)
    if r["overall_pass"] or "ct_evidence" not in r["uncovered_tracked"]:
        failures.append("tracked artifact without a disposition did not fail")

    # (5) the committed manifest passes for real
    rep, _ = mod.load_and_evaluate()
    if not rep.get("overall_pass"):
        failures.append(f"the committed freshness_artifacts manifest did not pass: {rep}")
    if not rep.get("incident_drill_autorefresh"):
        failures.append("real manifest: incident_drill_log not reported as auto-refreshed")

    # (5b) CAAS-009 regression: a missing CAAS_BOT_TOKEN must not kill the
    # scheduled lane before diagnostics can be regenerated and uploaded. The
    # workflow may still hard-fail at the final push if branch protection rejects
    # the fallback token, but it must not exit in the token-mode reporting step.
    workflow = (LIB_ROOT / ".github" / "workflows" / "caas-evidence-refresh.yml").read_text(
        encoding="utf-8", errors="replace")
    if "::error::CAAS_BOT_TOKEN secret is not configured" in workflow:
        failures.append("evidence refresh workflow still hard-fails on missing CAAS_BOT_TOKEN")
    if "token: ${{ secrets.CAAS_BOT_TOKEN || github.token }}" not in workflow:
        failures.append("checkout does not use CAAS_BOT_TOKEN || github.token fallback")
    if "GH_TOKEN: ${{ secrets.CAAS_BOT_TOKEN || github.token }}" not in workflow:
        failures.append("commit/push does not use CAAS_BOT_TOKEN || github.token fallback")
    if "HAS_CAAS_BOT_TOKEN: ${{ secrets.CAAS_BOT_TOKEN != '' }}" not in workflow:
        failures.append("commit/push does not expose a CAAS_BOT_TOKEN presence guard")
    if "CAAS_BOT_TOKEN is required to commit refreshed evidence to protected dev" not in workflow:
        failures.append("commit/push does not fail closed with a clear protected-dev token message")

    # (6) RR-BAS-02 promotion safety: a stale drill log BLOCKS under blocking
    #     severity, but only WARNS under the live advisory severity (no false pass).
    try:
        sla = _load_ci_module("audit_sla_check.py", "sla_drill_selftest")
        sla._file_age_days = lambda p: 99.0  # simulate a long-stale drill log
        blk = {"slos": {"incident_drill_freshness_days":
                        {"threshold": 14, "pre_alert_buffer_days": 4, "severity": "blocking"}}}
        f_blk, _ = sla.check_incident_drill_freshness(blk)
        if not any(x.get("severity") == "blocking" and x.get("status") == "stale" for x in f_blk):
            failures.append("stale drill log under blocking severity did not block")
        warn = {"slos": {"incident_drill_freshness_days":
                         {"threshold": 14, "pre_alert_buffer_days": 4, "severity": "warning"}}}
        f_warn, _ = sla.check_incident_drill_freshness(warn)
        if any(x.get("severity") == "blocking" for x in f_warn):
            failures.append("advisory (warning) drill severity incorrectly blocked")
        if not any(x.get("severity") == "warning" and x.get("status") == "stale" for x in f_warn):
            failures.append("advisory drill: stale log did not even warn (false pass)")
    except Exception as exc:
        failures.append(f"incident-drill block self-test failed: {exc}")

    if failures:
        fail(tag, "; ".join(failures))
    else:
        ok(tag, "missing producer / malformed mode / unresolved residual / uncovered tracked all fail; "
                "real manifest passes; stale drill blocks under blocking severity, only warns under advisory")


def check_package_provenance_binding_fixtures() -> None:
    """B20: the package provenance binding gate must FAIL a surface missing a binding
    field, a `bound` artifact whose commit/CAAS-bundle-hash do not match or whose
    artifact hash is absent, and an `owner_gated` release artifact marked current
    (real hash/run id) without an owner release; the real dev manifest (templates +
    owner_gated sentinels) passes. Provenance binding, not release authorization."""
    tag = "B20:package_provenance_binding"
    try:
        mod = _load_ci_module("check_package_provenance_binding.py", "ppb_selftest")
    except Exception as exc:
        fail(tag, f"import failed: {exc}")
        return

    ev = mod.evaluate
    HEAD = "72e7d4346de328fd70ab9295ad34d0decbe6e887"
    BUN = "1ad8eac7a924f39dba2b3fbe8c5a8604c3f4350b7114299969471283f3280f08"
    wfs = {".github/workflows/nuget-native.yml", ".github/workflows/release.yml"}

    def tmpl(**kw):
        b = {"artifact": "nuget", "producer_workflow": ".github/workflows/nuget-native.yml",
             "source_commit": "@HEAD", "source_branch": "dev", "artifact_sha256": None,
             "caas_bundle_sha256": "@committed-bundle", "audit_gate_verdict": "@audit-gate",
             "workflow_run_id": None, "status": "template", "severity": "blocking"}
        b.update(kw); return b

    def bound(**kw):
        b = {"artifact": "nuget", "producer_workflow": ".github/workflows/nuget-native.yml",
             "source_commit": HEAD, "source_branch": "dev", "artifact_sha256": "a" * 64,
             "caas_bundle_sha256": BUN, "audit_gate_verdict": "pass",
             "workflow_run_id": "123", "status": "bound", "severity": "blocking"}
        b.update(kw); return b

    def owner(**kw):
        b = {"artifact": "release", "producer_workflow": ".github/workflows/release.yml",
             "source_commit": "@release", "source_branch": "refs/tags/v*", "artifact_sha256": None,
             "caas_bundle_sha256": "@release-bundle", "audit_gate_verdict": "@release",
             "workflow_run_id": None, "status": "owner_gated", "severity": "owner_gated"}
        b.update(kw); return b

    failures = []

    if not ev([tmpl(), owner()], HEAD, BUN, wfs)["overall_pass"]:
        failures.append("healthy template+owner_gated did not pass")
    if not ev([bound()], HEAD, BUN, wfs)["overall_pass"]:
        failures.append("healthy bound surface did not pass")

    # (1) missing provenance: a surface missing a binding field
    s = tmpl(); del s["caas_bundle_sha256"]
    r = ev([s], HEAD, BUN, wfs)
    if r["overall_pass"] or "missing_binding" not in r["problems"]:
        failures.append("missing binding field did not fail")

    # (2) wrong commit on a bound artifact
    r = ev([bound(source_commit="b" * 40)], HEAD, BUN, wfs)
    if r["overall_pass"] or "commit_mismatch" not in r["problems"]:
        failures.append("wrong commit did not fail")

    # (3) wrong CAAS bundle hash on a bound artifact
    r = ev([bound(caas_bundle_sha256="c" * 64)], HEAD, BUN, wfs)
    if r["overall_pass"] or "bundle_mismatch" not in r["problems"]:
        failures.append("wrong CAAS bundle hash did not fail")

    # (4) missing artifact hash on a bound artifact
    r = ev([bound(artifact_sha256=None)], HEAD, BUN, wfs)
    if r["overall_pass"] or "missing_artifact_hash" not in r["problems"]:
        failures.append("missing artifact hash did not fail")

    # (5) release artifact marked current without an owner release
    r = ev([owner(artifact_sha256="d" * 64, workflow_run_id="999")], HEAD, BUN, wfs)
    if r["overall_pass"] or "release_marked_current" not in r["problems"]:
        failures.append("release artifact marked current did not fail")

    # (6) producer workflow that does not exist
    r = ev([tmpl(producer_workflow=".github/workflows/__nope__.yml")], HEAD, BUN, wfs)
    if r["overall_pass"] or "unknown_workflow" not in r["problems"]:
        failures.append("unknown producer workflow did not fail")

    # (7) the committed dev manifest passes for real
    rep, _ = mod.load_and_evaluate()
    if not rep.get("overall_pass"):
        failures.append(f"the committed PACKAGE_PROVENANCE_STATUS.json did not pass: {rep.get('problems')}")

    if failures:
        fail(tag, "; ".join(failures))
    else:
        ok(tag, "missing field / wrong commit / wrong bundle hash / missing artifact hash / "
                "release-marked-current / unknown workflow all fail; real dev manifest passes")


def check_release_package_contents_fixtures() -> None:
    """Release packages must contain only product libraries. Test/audit/exploit
    standalone .lib/.a artifacts may exist in the build tree for CI, but the
    release archive must fail closed if any of them is copied into lib/static or
    lib/shared."""
    tag = "REL:package_contents"
    try:
        mod = _load_ci_module("check_release_package_contents.py", "release_contents_selftest")
    except Exception as exc:
        fail(tag, f"import failed: {exc}")
        return

    failures = []
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        good = root / "good"
        (good / "lib" / "static").mkdir(parents=True)
        (good / "lib" / "shared").mkdir(parents=True)
        (good / "lib" / "static" / "ultrafast_secp256k1.lib").write_bytes(b"x")
        (good / "lib" / "static" / "ufsecp_s.lib").write_bytes(b"x")
        (good / "lib" / "shared" / "ufsecp.dll").write_bytes(b"x")
        if not mod.scan_package(good)["overall_pass"]:
            failures.append("valid product-only package did not pass")

        bad = root / "bad"
        (bad / "lib" / "static").mkdir(parents=True)
        (bad / "lib" / "static" / "ultrafast_secp256k1.lib").write_bytes(b"x")
        (bad / "lib" / "static" / "test_exploit_batch_sign_standalone.lib").write_bytes(b"x")
        rep = mod.scan_package(bad)
        if rep["overall_pass"]:
            failures.append("test/exploit standalone library did not fail")
        if not any(v["problem"] == "forbidden_test_or_audit_library" for v in rep["violations"]):
            failures.append("forbidden standalone library was not classified")

        unexpected = root / "unexpected"
        (unexpected / "lib" / "static").mkdir(parents=True)
        (unexpected / "lib" / "static" / "ultrafast_secp256k1.lib").write_bytes(b"x")
        (unexpected / "lib" / "static" / "internal_helper.lib").write_bytes(b"x")
        rep = mod.scan_package(unexpected)
        if rep["overall_pass"] or not any(v["problem"] == "unexpected_library" for v in rep["violations"]):
            failures.append("unexpected internal library did not fail")

        empty = root / "empty"
        (empty / "lib" / "static").mkdir(parents=True)
        rep = mod.scan_package(empty)
        if rep["overall_pass"] or not any(v["problem"] == "missing_product_library" for v in rep["violations"]):
            failures.append("empty lib package did not fail")

    if failures:
        fail(tag, "; ".join(failures))
    else:
        ok(tag, "product-only package passes; test/exploit/internal/empty library packages fail closed")


def check_libbitcoin_perf_matrix_fixtures() -> None:
    """B21: the libbitcoin performance matrix gate must fail missing surfaces,
    wrong target_context, missing evidence, native-hardware overclaim, and missing
    JSON benchmark artifact contracts; the real manifest passes."""
    tag = "B21:libbitcoin_perf_matrix"
    try:
        mod = _load_ci_module("check_libbitcoin_perf_matrix.py", "lbtc_perf_selftest")
    except Exception as exc:
        fail(tag, f"import failed: {exc}")
        return

    ev = mod.evaluate

    def row(id_, **kw):
        base = {
            "id": id_,
            "surface": id_,
            "target_context": "libbitcoin",
            "status": "implemented",
            "severity": "blocking",
            "claim_scope": "test scope",
            "evidence_paths": ["docs/LIBBITCOIN_INTEGRATION.md"],
            "reproduce_command": "python3 ci/audit_gate.py --libbitcoin-perf-matrix",
            "native_hardware_claim": False,
            "benchmark_artifact_contract": False,
            "copy_policy": "test",
        }
        base.update(kw)
        return base

    healthy = [
        row("lbtc_cpp_default_controller"),
        row("lbtc_benchmark_json_artifact", benchmark_artifact_contract=True),
        row("lbtc_cuda_row_persistent_scratch", native_hardware_claim=True,
            benchmark_artifact_contract=True),
        row("lbtc_msvc_windows_profile", status="measured_external", severity="warning"),
        row("lbtc_vertical_opaque_contract", status="documented_current"),
        row("lbtc_caas_perf_matrix_gate"),
    ]
    failures = []

    if not ev(healthy)["overall_pass"]:
        failures.append("healthy synthetic matrix did not pass")

    r = ev(healthy[:-1])
    if r["overall_pass"] or "missing_required_surface" not in r["problems"]:
        failures.append("missing required surface did not fail")

    bad_context = list(healthy)
    bad_context[0] = row("lbtc_cpp_default_controller", target_context="microbench")
    r = ev(bad_context)
    if r["overall_pass"] or "bad_context" not in r["problems"]:
        failures.append("wrong target_context did not fail")

    missing_evidence = list(healthy)
    missing_evidence[0] = row("lbtc_cpp_default_controller", evidence_paths=["docs/__missing_lbtc.md"])
    r = ev(missing_evidence)
    if r["overall_pass"] or "missing_evidence" not in r["problems"]:
        failures.append("missing evidence path did not fail")

    native_overclaim = list(healthy)
    native_overclaim[0] = row("lbtc_cpp_default_controller", status="documented_current",
                              native_hardware_claim=True, benchmark_artifact_contract=True)
    r = ev(native_overclaim)
    if r["overall_pass"] or "native_overclaim" not in r["problems"]:
        failures.append("native-hardware overclaim did not fail")

    no_artifact = list(healthy)
    no_artifact[1] = row("lbtc_benchmark_json_artifact", benchmark_artifact_contract=False)
    r = ev(no_artifact)
    if r["overall_pass"] or "missing_benchmark_artifact_contract" not in r["problems"]:
        failures.append("benchmark JSON contract omission did not fail")

    rep, _ = mod.load_and_evaluate()
    if not rep.get("overall_pass"):
        failures.append(f"the committed LIBBITCOIN_PERF_MATRIX_STATUS.json did not pass: {rep.get('problems')}")

    if failures:
        fail(tag, "; ".join(failures))
    else:
        ok(tag, "missing surface / wrong context / missing evidence / native overclaim / "
                "missing JSON artifact contract all fail; real manifest passes")


def check_gpu_backend_parity_fixtures() -> None:
    """The GPU backend native-parity gate must classify native/fallback/stub/
    missing overrides correctly on synthetic fixtures (proof-it-blocks), and
    must not silently pass when docs overclaim parity or an operation has no
    ABI mapping. Also pins the checker's result against the real repo so a
    NEW violation trips this test -- the repo is expected to be fully clean
    (zero violations) as of the hash160_pubkey_batch/OpenCL native-kernel fix."""
    tag = "GPU-PARITY:backend_parity_fixtures"
    try:
        mod = _load_ci_module("check_gpu_backend_parity.py", "gpu_backend_parity_selftest")
    except Exception as exc:
        fail(tag, f"import failed: {exc}")
        return

    failures = []

    # --- classify_override(): each forbidden gap class + the success case ---
    native_cuda_body = "{ my_kernel<<<blocks, threads>>>(a, b, c); }"
    if mod.classify_override(native_cuda_body, "cuda") != "native":
        failures.append("CUDA kernel-launch body not classified as native")

    native_opencl_body = "{ clEnqueueNDRangeKernel(queue, k, 1, nullptr, &global, nullptr, 0, nullptr, nullptr); }"
    if mod.classify_override(native_opencl_body, "opencl") != "native":
        failures.append("OpenCL clEnqueueNDRangeKernel body not classified as native")

    native_metal_body = "{ runtime_->dispatch_sync(pipe, (uint32_t)count, 64u, {&buf}); }"
    if mod.classify_override(native_metal_body, "metal") != "native":
        failures.append("Metal dispatch_sync body not classified as native")

    fallback_body = "{ for (size_t i = 0; i < count; ++i) { host_fn(i); } }"
    if mod.classify_override(fallback_body, "opencl") != "fallback_only":
        failures.append("pure CPU loop body not classified as fallback_only")

    stub_body = "{ (void)a; (void)b; return GpuError::Unsupported; }"
    if mod.classify_override(stub_body, "cuda") != "stub_unsupported":
        failures.append("bare return-Unsupported body not classified as stub_unsupported")

    if mod.classify_override(None, "cuda") != "missing_no_override":
        failures.append("absent override not classified as missing_no_override")

    if mod.classify_override(stub_body, "cuda", is_lifecycle=True) != "lifecycle_present":
        failures.append("lifecycle op with a found body not classified as lifecycle_present")

    # --- parse_gpu_backend_operations(): pure-virtual + virtual-with-default ---
    synthetic_hpp = """
    class GpuBackend {
    public:
        virtual ~GpuBackend() = default;
        virtual uint32_t backend_id() const = 0;
        virtual GpuError widget_batch(const uint8_t* in, size_t n, uint8_t* out)
        {
            (void)in; (void)n; (void)out;
            return GpuError::Unsupported;
        }
    };
    """
    ops = mod.parse_gpu_backend_operations(synthetic_hpp)
    op_names = {op["name"] for op in ops}
    if op_names != {"backend_id", "widget_batch"}:
        failures.append(f"header parser found wrong op set: {sorted(op_names)}")
    else:
        by_name = {op["name"]: op for op in ops}
        if not by_name["backend_id"]["pure_virtual"]:
            failures.append("pure-virtual op misclassified as having a default body")
        if by_name["widget_batch"]["pure_virtual"] or not by_name["widget_batch"]["has_default_body"]:
            failures.append("virtual-with-default op misclassified as pure virtual / no body")

    # --- parse_permanent_exceptions(): doc-ledger exception lookup ---
    synthetic_doc = """
## Permanent Architecture Exceptions

| Operation | Backend | Reason |
|---|---|---|
| `widget_batch` | OpenCL | Synthetic fixture exception for the self-test. |

---
"""
    exceptions = mod.parse_permanent_exceptions(synthetic_doc)
    if ("widget_batch", "opencl") not in exceptions:
        failures.append("Permanent Architecture Exceptions table row not parsed into the ledger")

    # --- check_abi_exposure(): missing ABI mapping / missing ABI symbol fail closed ---
    fake_ops = [{"name": "hash256", "return_type": "GpuError", "pure_virtual": False, "has_default_body": True}]
    abi_present = mod.check_abi_exposure("GpuError ufsecp_gpu_hash256(...);", fake_ops)
    if abi_present:
        failures.append("present ABI symbol incorrectly flagged as missing")
    abi_absent = mod.check_abi_exposure("GpuError ufsecp_gpu_something_else(...);", fake_ops)
    if not any(v["kind"] == "abi_missing" for v in abi_absent):
        failures.append("absent ABI symbol did not fail closed")

    # --- cross_check_docs(): doc overclaim must fail, honest claim must not ---
    doc_rows = {"ufsecp_gpu_hash256": {"cuda": "Y", "opencl": "Y", "metal": "Y"}}
    status_all_native = {("hash256", "cuda"): "native", ("hash256", "opencl"): "native", ("hash256", "metal"): "native"}
    if mod.cross_check_docs(doc_rows, status_all_native):
        failures.append("honest doc claim (matches code) incorrectly flagged as overclaim")
    status_opencl_gap = dict(status_all_native)
    status_opencl_gap[("hash256", "opencl")] = "fallback_only"
    overclaim = mod.cross_check_docs(doc_rows, status_opencl_gap)
    if not any(v["kind"] == "doc_overclaim" and v["backend"] == "opencl" for v in overclaim):
        failures.append("doc claiming Y where code is non-native did not fail (doc-overclaim)")

    # --- real repo: pin the currently-known violation set (regression guard) ---
    report = mod.evaluate()
    real_ops_with_violations = {v["op"] for v in report.get("violations", [])}
    known_gap_ops: set[str] = set()
    if real_ops_with_violations - known_gap_ops:
        failures.append(
            f"real repo has NEW unexpected GPU backend parity violation(s): "
            f"{sorted(real_ops_with_violations - known_gap_ops)}"
        )

    if failures:
        fail(tag, "; ".join(failures))
    else:
        ok(tag, "native/fallback/stub/missing/lifecycle classification, header parsing, "
                "exception-ledger lookup, ABI fail-closed check, and doc-overclaim check "
                "all correct; real repo has zero GPU backend parity violations")


def check_caas_dashboard_evidence_browser() -> None:
    """The CAAS dashboard must expose the committed evidence/status manifests in
    one central browser. This prevents the UI from regressing into scattered
    cards where reviewers cannot inspect CI-backed evidence rows directly."""
    tag = "DASH:evidence_browser"
    try:
        mod = _load_ci_module("caas_dashboard.py", "caas_dashboard_selftest")
    except Exception as exc:
        fail(tag, f"import failed: {exc}")
        return

    failures = []
    try:
        rows = mod.collect_evidence_browser()
    except Exception as exc:
        fail(tag, f"collect_evidence_browser crashed: {exc}")
        return

    domains = {r.get("domain") for r in rows}
    expected_domains = {
        "Integration Evidence",
        "CT Evidence",
        "Fuzz Campaign",
        "GPU / Hardware",
        "Package Provenance",
        "Libbitcoin Perf Matrix",
        "External Audit Bundle",
    }
    missing_domains = sorted(expected_domains - domains)
    if missing_domains:
        failures.append("missing domains: " + ", ".join(missing_domains))

    ids = {str(r.get("id", "")) for r in rows}
    for expected in ("INT-SHIM-PARITY", "CT-ECDSA-SIGN", "FUZZ-DER-PARSER",
                     "lbtc_cuda_row_persistent_scratch"):
        if expected not in ids:
            failures.append(f"missing evidence row {expected}")

    if len(rows) < 30:
        failures.append(f"too few evidence rows: {len(rows)}")

    html = mod.render_section_evidence(rows)
    for token in ('data-evidence-dashboard', 'id="evidence-table"', 'id="evidence-search"',
                  "Integration Evidence", "Libbitcoin Perf Matrix"):
        if token not in html:
            failures.append(f"rendered evidence section missing {token}")

    if failures:
        fail(tag, "; ".join(failures))
    else:
        ok(tag, f"central evidence browser renders {len(rows)} rows across {len(domains)} domains")


def check_caas_gate_negative_fixture_coverage() -> None:
    """B5 completeness critic: every high-value CAAS gate must have a registered
    negative fixture in this file. A green gate without a proof that it fails on
    bad input is a trust assumption, not evidence."""
    tag = "B5:fixture_coverage"
    required = {
        "audit_gate.py (P21)": "check_p21_semantic_requirement_map",
        "audit_sla_check.py": "check_audit_sla_pre_alert_and_block",
        "verify_external_audit_bundle.py": "check_external_audit_bundle_negative_fixtures",
        "ct_independence_check.py": "check_ct_independence_negative_fixtures",
        "multi_ci_repro_check.py": "check_multi_ci_repro_negative_fixtures",
        "security_autonomy_check.py": "check_security_autonomy_forced_failure",
        "supply_chain_gate.py": "check_supply_chain_negative_fixtures",
        "check_source_graph_quality.py": "check_source_graph_quality_negative_fixtures",
        "check_bench_doc_consistency.py": "check_bench_artifact_sanity_fixtures",
        "incident_drills.py": "check_incident_drills_real_injection",
        "research_monitor.py": "check_research_monitor_resilience",
        "check_integration_evidence.py": "check_integration_evidence_fixtures",
        "check_ct_evidence_status.py": "check_ct_evidence_status_fixtures",
        "check_fuzz_campaign_status.py": "check_fuzz_campaign_status_fixtures",
        "check_gpu_hardware_evidence.py": "check_gpu_hardware_evidence_fixtures",
        "check_bench_target_context.py": "check_bench_target_context_fixtures",
        "check_research_signal_matrix.py": "check_research_signal_matrix_fixtures",
        "check_evidence_refresh_coverage.py": "check_evidence_refresh_coverage_fixtures",
        "check_package_provenance_binding.py": "check_package_provenance_binding_fixtures",
        "check_release_package_contents.py": "check_release_package_contents_fixtures",
        "check_libbitcoin_perf_matrix.py": "check_libbitcoin_perf_matrix_fixtures",
        "check_gpu_backend_parity.py": "check_gpu_backend_parity_fixtures",
    }
    g = globals()
    missing = [f"{gate} -> {fn}" for gate, fn in required.items()
               if fn not in g or not callable(g[fn])]
    if missing:
        fail(tag, "high-value gates missing a negative fixture: " + "; ".join(missing))
    else:
        ok(tag, f"all {len(required)} high-value CAAS gates have a registered negative fixture")


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
    check_caas_integrity_json_purity()
    check_research_monitor_resilience()
    check_p21_semantic_requirement_map()
    check_audit_sla_pre_alert_and_block()
    check_audit_sla_build_report_not_tracked()
    check_external_audit_bundle_negative_fixtures()
    check_ct_independence_negative_fixtures()
    check_multi_ci_repro_negative_fixtures()
    check_security_autonomy_forced_failure()
    check_supply_chain_negative_fixtures()
    check_source_graph_quality_negative_fixtures()
    check_bench_artifact_sanity_fixtures()
    check_incident_drills_real_injection()
    check_research_monitor_actionable_body()
    check_integration_evidence_fixtures()
    check_ct_evidence_status_fixtures()
    check_fuzz_campaign_status_fixtures()
    check_gpu_hardware_evidence_fixtures()
    check_bench_target_context_fixtures()
    check_research_signal_matrix_fixtures()
    check_evidence_refresh_coverage_fixtures()
    check_package_provenance_binding_fixtures()
    check_release_package_contents_fixtures()
    check_libbitcoin_perf_matrix_fixtures()
    check_gpu_backend_parity_fixtures()
    check_caas_dashboard_evidence_browser()
    check_caas_gate_negative_fixture_coverage()
    check_secret_path_before_sha_fallback()

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
        check_dev_bug_scanner_deep_patterns()
        check_hot_path_alloc_scanner_quality()
        check_preflight_smoke()
        check_preflight_ctest_registry_smoke()
        check_preflight_ctest_registry_classification()
        check_validate_assurance_smoke()
        check_export_assurance_smoke()
        check_secret_path_changes_fail_closed()
        check_rule16_json_smoke()

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
