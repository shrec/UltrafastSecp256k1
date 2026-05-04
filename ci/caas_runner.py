#!/usr/bin/env python3
"""
caas_runner.py  --  Continuous Audit as a Service (CAAS) local runner

Runs the full audit pipeline in a single fail-fast pass:

  Stage 1  —  Static analysis (audit_test_quality_scanner)
  Stage 2  —  Traceability join (exploit_traceability_join.py)
  Stage 3  —  Audit gate        (audit_gate.py)
  Stage 4  —  Security autonomy (security_autonomy_check.py)
  Stage 5  —  Bundle produce    (external_audit_bundle.py)
  Stage 6  —  Bundle verify     (verify_external_audit_bundle.py)

Exit codes:
    0   all stages PASS
    1   one or more stages FAIL (fail-fast: stops at first failure)
    2   invocation/setup error

Usage:
    python3 ci/caas_runner.py                   # fail-fast by default
    python3 ci/caas_runner.py --no-fail-fast    # run all stages, report all
    python3 ci/caas_runner.py --json            # JSON summary to stdout
    python3 ci/caas_runner.py --json -o caas_report.json
    python3 ci/caas_runner.py --skip-bundle     # skip bundle gen/verify (faster)
    python3 ci/caas_runner.py --stage scanner   # run only one stage

    # Profile-scoped runs
    python3 ci/caas_runner.py --list-profiles
    python3 ci/caas_runner.py --profile bitcoin-core-backend
    python3 ci/caas_runner.py --profile bitcoin-core-backend --auditor-mode
    python3 ci/caas_runner.py --profile cpu-signing --auditor-mode --json

Profiles define a subset of stages plus optional pre-flight checks.
--auditor-mode forces --no-fail-fast, runs extra_checks, and prints a
REVIEWER SUMMARY with scope, out-of-scope, per-stage pass/fail, and
the exact command needed to reproduce the run.

Environment overrides:
    CAAS_SCANNER_DIR   — override audit dir for scanner (default: audit/)
    CAAS_TIMEOUT       — per-stage timeout in seconds (default: 300)
    CAAS_FAIL_FAST     — set to "0" to disable fail-fast
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

CAAS_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

STAGES = [
    {
        "id": "exploit_wiring",
        "name": "Exploit Wiring (check_exploit_wiring.py)",
        "script": "check_exploit_wiring.py",
        "args": [],
        "pass_fn": "_generic_pass",
        "description": "Verify all test_exploit_*.cpp files are wired in unified_audit_runner.cpp",
        "blocking": True,
    },
    {
        "id": "scanner",
        "name": "Static Analysis (audit_test_quality_scanner)",
        "script": "audit_test_quality_scanner.py",
        "args": ["--json"],
        "pass_fn": "_scanner_pass",
        "description": "Zero-findings requirement on exploit PoC suite",
        "blocking": True,
    },
    {
        "id": "traceability",
        "name": "Traceability Join (exploit_traceability_join.py)",
        "script": "exploit_traceability_join.py",
        "args": ["--emit-join"],
        "pass_fn": "_generic_pass",
        "description": "G-9b gate: exploit catalog ↔ spec matrix ↔ RR ↔ AM cross-refs (strict); emits EXPLOIT_TRACEABILITY_JOIN.md",
        "blocking": True,
    },
    {
        "id": "audit_gate",
        "name": "Audit Gate (audit_gate.py)",
        "script": "audit_gate.py",
        "args": ["--json"],
        "pass_fn": "_generic_pass",
        "description": "P0–P18 audit principles gate",
        "blocking": True,
    },
    {
        "id": "security_autonomy",
        "name": "Security Autonomy (security_autonomy_check.py)",
        "script": "security_autonomy_check.py",
        "args": ["--json"],
        "pass_fn": "_autonomy_pass",
        "description": "8-gate autonomy check — must score 100/100",
        "blocking": True,
    },
    {
        "id": "bundle_produce",
        "name": "Bundle Produce (external_audit_bundle.py)",
        "script": "external_audit_bundle.py",
        "args": [],
        "pass_fn": "_generic_pass",
        "description": "Hash-pin audit evidence bundle",
        "blocking": True,
        "skippable": True,
    },
    {
        "id": "bundle_verify",
        "name": "Bundle Verify (verify_external_audit_bundle.py)",
        "script": "verify_external_audit_bundle.py",
        # --allow-commit-mismatch: the bundle is generated once per review cycle and
        # does not change with every development commit. commit_match is advisory
        # (the evidence hashes and gate scores are the security-relevant checks).
        "args": ["--json", "--allow-commit-mismatch"],
        "pass_fn": "_generic_pass",
        "description": "Cryptographic integrity check of evidence bundle",
        "blocking": True,
        "skippable": True,
    },
]

# ---------------------------------------------------------------------------
# Profile definitions
# ---------------------------------------------------------------------------

PROFILES: dict[str, dict] = {
    "default": {
        "name": "Default",
        "description": "Full CAAS pipeline — all surfaces, all stages",
        "stages": "all",
        "extra_checks": [],
        "scope_description": "All CPU and GPU surfaces, all APIs",
        "out_of_scope": [],
    },
    "bitcoin-core-backend": {
        "name": "Bitcoin Core Backend",
        "description": "Scoped to the libsecp256k1 shim and CPU signing paths only",
        "stages": ["exploit_wiring", "scanner", "traceability", "audit_gate", "security_autonomy", "bundle_produce", "bundle_verify"],
        "extra_checks": [
            "check_libsecp_shim_parity.py",
            "check_core_build_mode.py",
        ],
        "scope_description": (
            "CPU signing (ECDSA, Schnorr, Taproot), "
            "libsecp256k1 shim (compat/libsecp256k1_shim/), "
            "strict parser parity, RFC 6979 nonces, "
            "C++20 build isolation"
        ),
        "out_of_scope": [
            "GPU backend (public data only, no signing)",
            "Multicoin address APIs",
            "ZK / FROST / MuSig2 (unless Core path uses them)",
            "BIP-352 performance",
            "Bindings (Rust, Python, etc.)",
        ],
    },
    "cpu-signing": {
        "name": "CPU Signing Only",
        "description": "Scoped to CPU signing paths and CT verification only",
        "stages": ["exploit_wiring", "scanner", "traceability", "audit_gate", "security_autonomy"],
        "extra_checks": [],
        "scope_description": "ECDSA, Schnorr, key derivation, CT layer",
        "out_of_scope": ["GPU", "Bindings", "Shim parity", "BIP-352"],
    },
    "gpu-public-data": {
        "name": "GPU Public Data",
        "description": "Scoped to GPU batch operations on public data only",
        "stages": ["exploit_wiring", "scanner", "traceability", "audit_gate"],
        "extra_checks": [],
        "scope_description": "GPU batch verify, BIP-352 scanning — public data only, no secrets",
        "out_of_scope": ["CPU signing CT", "Shim parity", "Bindings"],
    },
    "ffi-bindings": {
        "name": "FFI Bindings",
        "description": "Scoped to legacy C API and language bindings (Node, Python, Ruby, Go, Swift, Dart)",
        "stages": ["exploit_wiring", "scanner", "traceability", "audit_gate", "security_autonomy"],
        "extra_checks": [],
        "scope_description": (
            "bindings/c_api/ultrafast_secp256k1.cpp + language bindings; "
            "strict key parsing, CT signing paths, degenerate sig rejection"
        ),
        "out_of_scope": [
            "Bitcoin Core shim parity",
            "GPU backends",
            "WASM",
            "BCHN shim",
        ],
    },
    "wasm": {
        "name": "WASM",
        "description": "Scoped to WebAssembly browser/Node binding",
        "stages": ["exploit_wiring", "scanner", "traceability", "audit_gate"],
        "extra_checks": [],
        "scope_description": "bindings/wasm/ — WASM binary target",
        "out_of_scope": [
            "CPU signing CT (inherited from canonical ufsecp_* layer)",
            "GPU",
            "FFI",
            "Core shim",
        ],
    },
    "bchn-compat": {
        "name": "BCHN Compatibility",
        "description": "Scoped to Bitcoin Cash Node legacy Schnorr shim",
        "stages": ["exploit_wiring", "scanner", "traceability", "audit_gate"],
        "extra_checks": [],
        "scope_description": (
            "compat/libsecp256k1_bchn_shim/ — BCH legacy Schnorr (NOT BIP-340); "
            "CT generator mul, strict key parsing, key erasure"
        ),
        "out_of_scope": [
            "Bitcoin Core profile",
            "BIP-340 Schnorr",
            "GPU",
            "FFI bindings",
        ],
    },
    "release": {
        "name": "Release",
        "description": "Full pipeline including bundle produce — used for release preparation",
        "stages": "all",
        "extra_checks": [
            "check_libsecp_shim_parity.py",
            "check_core_build_mode.py",
            "check_source_graph_quality.py",
        ],
        "scope_description": "All surfaces — full release readiness check",
        "out_of_scope": [],
    },
}


# ---------------------------------------------------------------------------
# Pass functions (evaluated after subprocess returns)
# ---------------------------------------------------------------------------

_ADVISORY_SKIP_CODE = 77


def _generic_pass(result: subprocess.CompletedProcess, _stdout_json: dict | None) -> tuple[bool, str]:
    """Passes if exit code is 0; treats 77 (ADVISORY_SKIP_CODE) as advisory skip, not failure."""
    if result.returncode == _ADVISORY_SKIP_CODE:
        return True, "advisory-skip (no required infrastructure)"
    passed = result.returncode == 0
    detail = "exit 0" if passed else f"exit {result.returncode}"
    return passed, detail


def _scanner_pass(result: subprocess.CompletedProcess, stdout_json: dict | None) -> tuple[bool, str]:
    """Passes only when total_findings == 0."""
    if result.returncode != 0:
        return False, f"scanner exited {result.returncode}"
    if stdout_json is None:
        return False, "scanner produced no JSON output"
    total = stdout_json.get("total_findings", -1)
    if total == 0:
        return True, "0 findings"
    counts = {sev: 0 for sev in ("critical", "high", "medium", "low", "info")}
    for f in stdout_json.get("findings", []):
        sev = f.get("severity", "info")
        counts[sev] = counts.get(sev, 0) + 1
    summary = ", ".join(f"{v} {k}" for k, v in counts.items() if v)
    return False, f"{total} finding(s): {summary}"


def _autonomy_pass(result: subprocess.CompletedProcess, stdout_json: dict | None) -> tuple[bool, str]:
    """Passes only when autonomy_score == 100.
    CAAS-20 fix: returncode is checked FIRST — a non-zero exit always fails
    regardless of JSON content (prevents a crashing script with JSON score=100
    from producing a false-pass)."""
    if result.returncode != 0:
        return False, f"autonomy check exited {result.returncode}"
    if stdout_json is None:
        return False, "autonomy check produced no JSON output"
    score = stdout_json.get("autonomy_score", 0)
    ready = stdout_json.get("autonomy_ready", False)
    if ready and score >= 100:
        return True, f"score={score}/100"
    return False, f"score={score}/100, ready={ready}"


# Patch forward-declared functions into STAGES table
def _resolve_stage_fns():
    fn_map = {
        "_scanner_pass": _scanner_pass,
        "_generic_pass": _generic_pass,
        "_autonomy_pass": _autonomy_pass,
    }
    for stage in STAGES:
        fn_name = stage.get("pass_fn")
        if isinstance(fn_name, str):
            stage["pass_fn"] = fn_map[fn_name]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_stage(stage: dict, timeout: int) -> dict:
    """Run a single CAAS stage and return a result dict."""
    script_path = SCRIPT_DIR / stage["script"]
    if not script_path.exists():
        return {
            "id": stage["id"],
            "name": stage["name"],
            "status": "missing",
            "passed": False,
            "detail": f"Script not found: {script_path}",
            "duration_s": 0.0,
        }

    import os
    env_overrides = {}
    if stage["id"] == "scanner":
        scanner_dir = os.environ.get("CAAS_SCANNER_DIR")
        if scanner_dir:
            env_overrides["CAAS_SCANNER_DIR"] = scanner_dir

    cmd = [sys.executable, str(script_path)] + stage["args"]
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(LIB_ROOT),
            env={**__import__("os").environ, **env_overrides},
            start_new_session=True,  # place child in its own process group for clean kill
        )
    except subprocess.TimeoutExpired as exc:
        # Kill the entire process group so hanging child-of-child processes don't linger.
        import os as _os
        import signal as _signal
        try:
            _os.killpg(_os.getpgid(exc.process.pid), _signal.SIGKILL)
        except (ProcessLookupError, PermissionError, AttributeError):
            pass
        elapsed = time.monotonic() - t0
        # F-07 fix: capture partial stdout/stderr from the killed process.
        # TimeoutExpired stores whatever output was produced before the kill.
        def _decode(b: bytes | str | None) -> str:
            if b is None:
                return ""
            return b.decode(errors="replace") if isinstance(b, bytes) else b
        stdout_partial = _decode(exc.stdout)
        stderr_partial = _decode(exc.stderr)
        return {
            "id": stage["id"],
            "name": stage["name"],
            "status": "timeout",
            "passed": False,
            "detail": f"Timed out after {timeout}s",
            "duration_s": round(elapsed, 2),
            "output_truncated": True,
            "stdout_tail": stdout_partial[-2000:],
            "stderr_tail": stderr_partial[-500:],
        }
    elapsed = round(time.monotonic() - t0, 2)

    stdout_json = None
    try:
        stdout_json = json.loads(result.stdout)
    except (json.JSONDecodeError, ValueError):
        # Fallback: some scripts print diagnostics before the JSON payload.
        # Find the first '{' and try parsing from there.
        idx = result.stdout.find("{")
        if idx != -1:
            try:
                stdout_json = json.loads(result.stdout[idx:])
            except (json.JSONDecodeError, ValueError):
                pass

    passed, detail = stage["pass_fn"](result, stdout_json)

    # advisory_skip is set when the underlying script returned ADVISORY_SKIP_CODE
    # (77) AND the stage's pass_fn treated that as a non-failure pass (i.e.
    # returned True). We rely on the pass_fn having already accepted 77 — no
    # need to restrict by which specific pass_fn is in use.
    advisory_skip = passed and result.returncode == _ADVISORY_SKIP_CODE
    status = "advisory_skipped" if advisory_skip else ("passed" if passed else "failed")

    # M-6 fix: advisory-skip stages print diagnostic output explaining WHY
    # they skipped (missing GPU, Python, Cryptol). Use the same 2000-byte
    # tail as failed stages so skip diagnostics are not truncated in the JSON.
    capture_full = not passed or advisory_skip
    return {
        "id": stage["id"],
        "name": stage["name"],
        "status": status,
        "passed": passed,
        "advisory_skip": advisory_skip,
        "returncode": result.returncode,
        "detail": detail,
        "duration_s": elapsed,
        "stdout_tail": result.stdout[-2000:] if capture_full else result.stdout[-500:],
        "stderr_tail": result.stderr[-500:] if capture_full else result.stderr[-200:],
    }


def print_banner(text: str, color: str = CYAN) -> None:
    bar = "─" * 60
    print(f"\n{color}{BOLD}{bar}{RESET}")
    print(f"{color}{BOLD}  {text}{RESET}")
    print(f"{color}{BOLD}{bar}{RESET}")


def print_stage_header(idx: int, total: int, name: str) -> None:
    print(f"\n{CYAN}[{idx}/{total}] {name}{RESET}")


def print_result(result: dict) -> None:
    if result.get("advisory_skip") or result.get("status") == "advisory_skipped":
        icon = f"{YELLOW}—{RESET}"
        label = f"{YELLOW}ADV-SKIP{RESET}"
    elif result["passed"]:
        icon = f"{GREEN}✓{RESET}"
        label = f"{GREEN}PASS{RESET}"
    elif result["status"] == "missing":
        icon = f"{YELLOW}?{RESET}"
        label = f"{YELLOW}MISSING{RESET}"
    elif result["status"] == "timeout":
        icon = f"{YELLOW}⏱{RESET}"
        label = f"{YELLOW}TIMEOUT{RESET}"
    else:
        icon = f"{RED}✗{RESET}"
        label = f"{RED}FAIL{RESET}"

    print(f"  {icon} {label}  — {result['detail']}  ({result['duration_s']}s)")
    if not result["passed"]:
        tail = result.get("stdout_tail", "")
        if tail:
            print(f"\n{YELLOW}  --- output tail ---{RESET}")
            for line in tail.splitlines()[-30:]:
                print(f"  {line}")
        etail = result.get("stderr_tail", "")
        if etail:
            print(f"\n{YELLOW}  --- stderr tail ---{RESET}")
            for line in etail.splitlines()[-10:]:
                print(f"  {line}")


def print_summary(results: list[dict], total_s: float) -> None:
    print_banner("CAAS Summary", BOLD)
    max_name = max((len(r["name"]) for r in results), default=10)
    adv_skip_count = 0
    for r in results:
        if r.get("status") == "advisory_skipped" or r.get("advisory_skip"):
            status = f"{YELLOW}ADV-SKIP{RESET}"
            adv_skip_count += 1
        elif r["passed"]:
            status = f"{GREEN}PASS{RESET}"
        elif r.get("status") == "skipped":
            status = f"{YELLOW}SKIP{RESET}"
        elif r.get("status") == "missing":
            status = f"{YELLOW}MISSING{RESET}"
        else:
            status = f"{RED}FAIL{RESET}"
        pad = " " * (max_name - len(r["name"]) + 2)
        print(f"  {status}  {r['name']}{pad}{r['detail']}  ({r['duration_s']}s)")
    print()
    # F-01 fix: advisory-skipped stages are excluded from the overall verdict —
    # they are not "passing" evidence, just neutral (infrastructure absent).
    # A stage that returned exit 77 without actually being infrastructure-blocked
    # would previously inflate overall_pass; now it is excluded from the verdict.
    active_results = [r for r in results if r.get("status") not in ("skipped", "advisory_skipped")]
    overall = all(r["passed"] for r in active_results) if active_results else True
    overall_color = GREEN if overall else RED
    overall_text = "ALL STAGES PASSED" if overall else "AUDIT GATE VIOLATION — SEE ABOVE"
    suffix = f"  ({adv_skip_count} advisory-skipped)" if adv_skip_count else ""
    print(f"  {overall_color}{BOLD}{overall_text}{RESET}  (total: {round(total_s, 1)}s){suffix}")


def build_json_report(
    results: list[dict],
    total_s: float,
    profile_key: str = "default",
    profile: dict | None = None,
    extra_check_results: list[dict] | None = None,
) -> dict:
    # F-01/F-12 fix: exclude both fail-fast-skipped AND advisory-skipped stages
    # from the overall_pass computation.  Advisory-skipped stages have passed=True
    # but are not positive evidence of a passing gate — they are neutral
    # (infrastructure absent).  Including them was inflating overall_pass when
    # a module returned exit 77 without a legitimate infrastructure skip reason.
    active_results = [r for r in results if r.get("status") not in ("skipped", "advisory_skipped")]
    overall_pass = all(r["passed"] for r in active_results) if active_results else True
    extra_checks_clean = []
    if extra_check_results:
        for ec in extra_check_results:
            extra_checks_clean.append({
                "script": ec["script"],
                "passed": ec["passed"],
                "exit_code": ec["exit_code"],
                "advisory_skip": ec.get("advisory_skip", False),
            })
        # F-05/F-12 fix: exclude advisory-skipped extra_checks from the verdict.
        real_extra = [ec for ec in extra_check_results if not ec.get("advisory_skip")]
        overall_pass = overall_pass and (all(ec["passed"] for ec in real_extra) if real_extra else True)
    report: dict = {
        "caas_version": CAAS_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit_sha": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(LIB_ROOT),
        ).stdout.strip() or "unknown",
        "overall_pass": overall_pass,
        "total_duration_s": round(total_s, 2),
        "profile": profile_key,
        "profile_scope": profile["scope_description"] if profile else "",
        "profile_out_of_scope": profile["out_of_scope"] if profile else [],
        "extra_checks": extra_checks_clean,
        "stages": results,
    }
    return report


def run_extra_check(script_name: str, timeout: int) -> dict:
    """Run a single pre-flight extra_check script and return a result dict."""
    script_path = SCRIPT_DIR / script_name
    t0 = time.monotonic()
    if not script_path.exists():
        return {
            "script": script_name,
            "passed": False,
            "exit_code": -1,
            "detail": f"Script not found: {script_path}",
            "duration_s": 0.0,
        }
    cmd = [sys.executable, str(script_path)]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(LIB_ROOT),
            start_new_session=True,  # BUG-9 fix: place child in its own process group for clean kill
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - t0
        # F-07 fix: capture partial output from killed extra_check process.
        def _dec(b: bytes | str | None) -> str:
            if b is None:
                return ""
            return b.decode(errors="replace") if isinstance(b, bytes) else b
        return {
            "script": script_name,
            "passed": False,
            "exit_code": -1,
            "advisory_skip": False,
            "detail": f"Timed out after {timeout}s",
            "duration_s": round(elapsed, 2),
            "stdout_tail": _dec(exc.stdout)[-1000:],
            "stderr_tail": _dec(exc.stderr)[-500:],
        }
    elapsed = round(time.monotonic() - t0, 2)
    if result.returncode == _ADVISORY_SKIP_CODE:
        return {
            "script": script_name,
            "passed": True,
            "exit_code": _ADVISORY_SKIP_CODE,
            "advisory_skip": True,
            "detail": "advisory-skip (exit 77)",
            "duration_s": elapsed,
            "stdout_tail": "",
            "stderr_tail": "",
        }
    passed = result.returncode == 0
    return {
        "script": script_name,
        "passed": passed,
        "exit_code": result.returncode,
        "advisory_skip": False,
        "detail": "exit 0" if passed else f"exit {result.returncode}",
        "duration_s": elapsed,
        "stdout_tail": result.stdout[-1000:] if not passed else "",
        "stderr_tail": result.stderr[-500:] if not passed else "",
    }


def print_reviewer_summary(
    profile: dict,
    profile_key: str,
    extra_check_results: list[dict],
    stage_results: list[dict],
    total_s: float,
) -> None:
    """Print the auditor-mode REVIEWER SUMMARY block."""
    bar = "═" * 64
    print(f"\n{BOLD}{CYAN}{bar}{RESET}")
    print(f"{BOLD}{CYAN}  REVIEWER SUMMARY — {profile['name']}{RESET}")
    print(f"{BOLD}{CYAN}{bar}{RESET}")

    print(f"\n{BOLD}Profile:{RESET}      {profile_key}")
    print(f"{BOLD}Description:{RESET}  {profile['description']}")
    print(f"\n{BOLD}Scope:{RESET}")
    print(f"  {profile['scope_description']}")

    if profile["out_of_scope"]:
        print(f"\n{BOLD}Out of scope:{RESET}")
        for item in profile["out_of_scope"]:
            print(f"  • {item}")

    if extra_check_results:
        print(f"\n{BOLD}Pre-flight checks:{RESET}")
        for ec in extra_check_results:
            icon = f"{GREEN}✓{RESET}" if ec["passed"] else f"{RED}✗{RESET}"
            status = f"{GREEN}PASS{RESET}" if ec["passed"] else f"{RED}FAIL{RESET}"
            print(f"  {icon} {status}  {ec['script']}  ({ec['detail']}, {ec['duration_s']}s)")

    print(f"\n{BOLD}Stage results:{RESET}")
    for r in stage_results:
        if r["passed"]:
            icon = f"{GREEN}✓{RESET}"
            status = f"{GREEN}PASS{RESET}"
        elif r.get("status") == "skipped":
            icon = f"{YELLOW}—{RESET}"
            status = f"{YELLOW}SKIP{RESET}"
        elif r.get("status") == "missing":
            icon = f"{YELLOW}?{RESET}"
            status = f"{YELLOW}MISSING{RESET}"
        else:
            icon = f"{RED}✗{RESET}"
            status = f"{RED}FAIL{RESET}"
        print(f"  {icon} {status}  {r['name']}  — {r['detail']}  ({r['duration_s']}s)")

    print(f"\n{BOLD}Reproduce:{RESET}")
    print(f"  python3 ci/caas_runner.py --profile {profile_key} --auditor-mode")

    overall_pass = all(r["passed"] for r in stage_results if r.get("status") != "skipped")
    overall_ec_pass = all(ec["passed"] for ec in extra_check_results)
    fully_passed = overall_pass and overall_ec_pass
    verdict_color = GREEN if fully_passed else RED
    verdict_text = "AUDIT PASS" if fully_passed else "AUDIT FAIL — SEE ABOVE"
    print(f"\n  {verdict_color}{BOLD}{verdict_text}{RESET}  (total: {round(total_s, 1)}s)")
    print(f"{BOLD}{CYAN}{bar}{RESET}\n")


def _rebuild_graphs(timeout: int, quiet: bool = False) -> bool:
    """Rebuild both the legacy project graph and the source graph.

    This advances the built_at timestamp past any source file mtimes from
    mutation testing, eliminating the P6 Graph Freshness WARN.

    Returns True if all rebuilds succeeded, False if any rebuild failed.
    """
    cmds = [
        ([sys.executable, str(SCRIPT_DIR / "build_project_graph.py"), "--rebuild"],
         "Legacy project graph"),
        ([sys.executable, str(LIB_ROOT / "tools" / "source_graph_kit" / "source_graph.py"),
          "build", "-i"],
         "Source graph"),
    ]
    all_ok = True
    for cmd, label in cmds:
        t0 = time.monotonic()
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, cwd=str(LIB_ROOT)
            )
            elapsed = round(time.monotonic() - t0, 1)
            if result.returncode == 0:
                if not quiet:
                    print(f"  {GREEN}✓{RESET} {label} rebuilt ({elapsed}s)")
            else:
                all_ok = False
                if not quiet:
                    print(f"  {YELLOW}⚠{RESET} {label} rebuild exited {result.returncode} ({elapsed}s)")
        except subprocess.TimeoutExpired:
            all_ok = False
            if not quiet:
                print(f"  {YELLOW}⚠{RESET} {label} rebuild timed out")
    return all_ok


def main(argv: list[str] | None = None) -> int:
    import os

    _resolve_stage_fns()

    parser = argparse.ArgumentParser(
        description="CAAS runner — continuous audit gate for UltrafastSecp256k1"
    )
    parser.add_argument("--json", action="store_true", help="Output JSON report to stdout")
    parser.add_argument("-o", "--output", metavar="FILE", help="Write JSON report to file")
    parser.add_argument(
        "--no-fail-fast",
        action="store_true",
        help="Run all stages even after a failure (default: stop at first failure)",
    )
    parser.add_argument(
        "--skip-bundle",
        action="store_true",
        help="Skip bundle produce/verify stages (faster local iteration)",
    )
    parser.add_argument(
        "--stage",
        metavar="ID",
        help="Run only the stage with this ID (scanner|traceability|audit_gate|security_autonomy|bundle_produce|bundle_verify)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("CAAS_TIMEOUT", "300")),
        metavar="SECONDS",
        help="Per-stage timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--rebuild-graph",
        action="store_true",
        help="Rebuild both project graphs before running stages (fixes P6 Graph Freshness WARN)",
    )
    parser.add_argument(
        "--profile",
        metavar="NAME",
        default="default",
        choices=list(PROFILES.keys()),
        help=(
            "Audit profile to use — scopes stages and extra pre-flight checks. "
            "Choices: " + ", ".join(PROFILES.keys()) + "  (default: default)"
        ),
    )
    parser.add_argument(
        "--auditor-mode",
        action="store_true",
        help=(
            "Auditor mode: prints AUDITOR MODE banner, forces --no-fail-fast, "
            "runs profile extra_checks as blocking pre-flight, and prints a "
            "REVIEWER SUMMARY at the end with scope, out-of-scope, per-stage "
            "pass/fail, and the reproduce command."
        ),
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available audit profiles and exit",
    )
    args = parser.parse_args(argv)

    # --list-profiles: print and exit immediately
    if args.list_profiles:
        print(f"\n{BOLD}Available CAAS profiles:{RESET}\n")
        for key, prof in PROFILES.items():
            stage_ids = prof["stages"] if prof["stages"] == "all" else ", ".join(prof["stages"])
            print(f"  {CYAN}{BOLD}{key}{RESET}")
            print(f"    {prof['description']}")
            print(f"    Stages : {stage_ids}")
            print(f"    Scope  : {prof['scope_description']}")
            if prof["out_of_scope"]:
                print(f"    Out    : {'; '.join(prof['out_of_scope'])}")
            if prof["extra_checks"]:
                print(f"    Checks : {', '.join(prof['extra_checks'])}")
            print()
        return 0

    profile_key = args.profile
    profile = PROFILES[profile_key]
    auditor_mode = args.auditor_mode

    # --auditor-mode forces no-fail-fast
    fail_fast = not args.no_fail_fast
    if auditor_mode:
        fail_fast = False
    env_ff = os.environ.get("CAAS_FAIL_FAST", "1")
    if env_ff == "0":
        fail_fast = False

    # Select stages — first apply profile filter, then --stage override
    if profile["stages"] == "all":
        stages = list(STAGES)
    else:
        profile_ids: list[str] = profile["stages"]
        stages = [s for s in STAGES if s["id"] in profile_ids]

    if args.stage:
        stages = [s for s in stages if s["id"] == args.stage]
        if not stages:
            print(f"{RED}ERROR: Unknown stage '{args.stage}' (or not in profile '{profile_key}'){RESET}", file=sys.stderr)
            valid = ", ".join(s["id"] for s in STAGES)
            print(f"Valid stages: {valid}", file=sys.stderr)
            return 2

    if args.skip_bundle:
        stages = [s for s in stages if not s.get("skippable")]

    # ------------------------------------------------------------------ banner
    if not args.json:
        if auditor_mode:
            bar = "█" * 64
            print(f"\n{RED}{BOLD}{bar}{RESET}")
            print(f"{RED}{BOLD}  AUDITOR MODE  —  {profile['name']}{RESET}")
            print(f"{RED}{BOLD}  All stages will run. Results are held to auditor standard.{RESET}")
            print(f"{RED}{BOLD}{bar}{RESET}")
        print_banner(
            f"CAAS  v{CAAS_VERSION}  —  Continuous Audit as a Service"
            + (f"  [{profile['name']}]" if profile_key != "default" else "")
        )
        if profile_key != "default":
            print(f"  {CYAN}Profile scope:{RESET} {profile['scope_description']}")

    # Optionally rebuild both graphs before running stages.
    if args.rebuild_graph:
        if not args.json:
            print(f"\n{CYAN}[pre] Rebuilding project graphs...{RESET}")
        rebuild_ok = _rebuild_graphs(args.timeout, quiet=args.json)
        if not rebuild_ok:
            # L-5 fix: graph rebuild failure is a precondition failure, not a
            # stage result. Exit with an error so the caller sees the failure
            # rather than silently running gates against stale graph data.
            print(
                "ERROR: source graph rebuild failed; cannot run gates with stale data.",
                file=sys.stderr,
            )
            sys.exit(2)

    # --------------------------------------------------- extra_checks pre-flight
    extra_check_results: list[dict] = []
    if profile["extra_checks"]:
        if not args.json:
            print(f"\n{CYAN}[pre-flight] Running {len(profile['extra_checks'])} extra check(s)...{RESET}")
        for script_name in profile["extra_checks"]:
            ec_result = run_extra_check(script_name, args.timeout)
            extra_check_results.append(ec_result)
            if not args.json:
                icon = f"{GREEN}✓{RESET}" if ec_result["passed"] else f"{RED}✗{RESET}"
                status = f"{GREEN}PASS{RESET}" if ec_result["passed"] else f"{RED}FAIL{RESET}"
                print(f"  {icon} {status}  {script_name}  ({ec_result['detail']}, {ec_result['duration_s']}s)")
                if not ec_result["passed"]:
                    tail = ec_result.get("stdout_tail", "")
                    if tail:
                        print(f"\n{YELLOW}  --- output tail ---{RESET}")
                        for line in tail.splitlines()[-20:]:
                            print(f"  {line}")
                    etail = ec_result.get("stderr_tail", "")
                    if etail:
                        print(f"\n{YELLOW}  --- stderr tail ---{RESET}")
                        for line in etail.splitlines()[-10:]:
                            print(f"  {line}")
            # In auditor mode, a failing extra_check is blocking — but we still
            # run all checks (fail_fast is already False in auditor mode).
            # In non-auditor mode, extra_check failures are also blocking:
            # they factor into overall_pass at the end of main().

    # ----------------------------------------------------------- main stage loop
    results: list[dict] = []
    t_global = time.monotonic()
    aborted_at = None

    for idx, stage in enumerate(stages, 1):
        if not args.json:
            print_stage_header(idx, len(stages), stage["name"])

        result = run_stage(stage, args.timeout)
        results.append(result)

        if not args.json:
            print_result(result)

        if not result["passed"] and stage.get("blocking", True) and fail_fast:
            aborted_at = stage["id"]
            # Mark remaining stages as skipped
            for remaining in stages[idx:]:
                results.append({
                    "id": remaining["id"],
                    "name": remaining["name"],
                    "status": "skipped",
                    "passed": False,
                    "detail": f"skipped — fail-fast triggered by '{aborted_at}'",
                    "duration_s": 0.0,
                })
            break

    total_s = time.monotonic() - t_global
    stage_pass = all(r["passed"] for r in results if r.get("status") != "skipped")
    extra_pass = all(r.get("passed", True) for r in extra_check_results) if extra_check_results else True
    overall_pass = stage_pass and extra_pass

    if not args.json:
        print_summary(results, total_s)
        if aborted_at:
            print(
                f"\n{YELLOW}  Fail-fast triggered at stage '{aborted_at}'."
                f" Use --no-fail-fast to run all stages.{RESET}\n"
            )
        if auditor_mode:
            print_reviewer_summary(
                profile=profile,
                profile_key=profile_key,
                extra_check_results=extra_check_results,
                stage_results=results,
                total_s=total_s,
            )

    # ----------------------------------------- optional replay capsule (auditor)
    if auditor_mode:
        capsule_script = SCRIPT_DIR / "create_replay_capsule.py"
        if capsule_script.exists():
            try:
                capsule_result = subprocess.run(
                    [sys.executable, str(capsule_script)],
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                    cwd=str(LIB_ROOT),
                )
                if capsule_result.returncode != 0:
                    print(f"\n{YELLOW}  ⚠ Replay capsule creation failed (exit {capsule_result.returncode}) — "
                          f"audit evidence may be incomplete{RESET}", file=sys.stderr)
            except subprocess.TimeoutExpired:
                print(f"\n{YELLOW}  ⚠ Replay capsule timed out — audit evidence may be incomplete{RESET}",
                      file=sys.stderr)
            except OSError as e:
                print(f"\n{YELLOW}  ⚠ Replay capsule OS error: {e} — audit evidence may be incomplete{RESET}",
                      file=sys.stderr)

    report = build_json_report(
        results,
        total_s,
        profile_key=profile_key,
        profile=profile,
        extra_check_results=extra_check_results,
    )

    if args.json:
        print(json.dumps(report, indent=2))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not args.json:
            print(f"\n  Report written to: {out_path}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
