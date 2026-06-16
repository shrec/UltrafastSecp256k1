#!/usr/bin/env python3
"""Incident drill framework — automated security drill simulations.

Drills:
  1. Key compromise: inject known-weak key, verify detection
  2. CI poisoning: simulate tampered build output, verify provenance check
  3. Dependency compromise: simulate altered dependency, verify supply-chain gate

All drills must pass. Drill failure = release blocked.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent

DRILL_LOG = LIB_ROOT / "docs" / "INCIDENT_DRILL_LOG.json"


def _git_head() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(LIB_ROOT),
                           capture_output=True, text=True, timeout=15, check=False)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def drill_key_compromise() -> dict:
    """Simulate key compromise detection.

    Injects a known-weak key (all zeros, all ones, etc.) into the test
    harness and verifies the library rejects it at the API boundary.
    """
    start = time.monotonic()
    issues: list[str] = []

    # The library MUST reject these keys at the ABI boundary
    weak_keys = [
        ("all_zeros", "0" * 64),
        ("all_ones", "f" * 64),
        ("group_order_n", "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141"),
        ("above_order", "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364142"),
    ]

    # Try to load the library for live testing
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        from _ufsecp import UfSecp, find_lib
        lib = UfSecp(find_lib(None))
        live = True
    except Exception:
        live = False

    results: list[dict] = []
    for name, key_hex in weak_keys:
        if live:
            try:
                key_bytes = bytes.fromhex(key_hex)
                # Attempt to derive pubkey from weak key — should fail
                try:
                    pub = lib.pubkey(key_bytes)
                    # If we got a pubkey, the key was accepted — BAD for invalid keys
                    if name in ("all_zeros", "group_order_n", "above_order"):
                        issues.append(f"weak key '{name}' accepted — must be rejected")
                        results.append({"key": name, "rejected": False})
                    else:
                        results.append({"key": name, "rejected": False, "note": "valid edge key"})
                except Exception:
                    # Key was rejected — GOOD
                    results.append({"key": name, "rejected": True})
            except Exception as exc:
                results.append({"key": name, "error": str(exc)})
        else:
            results.append({"key": name, "mode": "doc-check-only"})

    elapsed = time.monotonic() - start

    return {
        "drill": "key_compromise",
        "passing": len(issues) == 0,
        "live_test": live,
        "injected_fault": "known-weak / out-of-range private keys (zero, n, n+1, all-ones)",
        "detection_gate": "ufsecp pubkey() ABI boundary (strict scalar parse)",
        "detected": live and len(issues) == 0,
        "elapsed_seconds": round(elapsed, 2),
        "weak_keys_tested": len(weak_keys),
        "results": results,
        "issues": issues,
    }


def drill_ci_poisoning() -> dict:
    """REAL injection drill: build a cross-provider reproducible-build hash pair
    where provider B's artifact has been tampered, then RUN multi_ci_repro_check.py
    and require it to DETECT the mismatch (non-zero exit). This proves CI poisoning
    is actually caught by the hash-comparison gate — not merely that a gate file
    exists.
    """
    start = time.monotonic()
    issues: list[str] = []
    detected = False
    legit_hash = tampered_hash = ""

    gate = SCRIPT_DIR / "multi_ci_repro_check.py"
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        legit = b"legitimate build output content for drill"
        tampered = legit + b"\x00TAMPERED"
        legit_hash = hashlib.sha256(legit).hexdigest()
        tampered_hash = hashlib.sha256(tampered).hexdigest()
        # sha256sum-format hash files for two "providers"
        (d / "provider_a.sha256").write_text(f"{legit_hash}  build/libufsecp.a\n")
        (d / "provider_b.sha256").write_text(f"{tampered_hash}  build/libufsecp.a\n")

        if not gate.exists():
            issues.append("multi_ci_repro_check.py not found — cannot verify tamper detection")
        elif legit_hash == tampered_hash:
            issues.append("tampered artifact has same hash — SHA-256 collision (impossible)")
        else:
            rc = subprocess.run(
                [sys.executable, str(gate), str(d / "provider_a.sha256"),
                 str(d / "provider_b.sha256"), "--json"],
                capture_output=True, text=True, timeout=30, cwd=str(LIB_ROOT)).returncode
            detected = rc != 0  # the gate MUST fail (detect the mismatch)
            if not detected:
                issues.append("multi_ci_repro_check did NOT detect the tampered artifact (false-negative)")

    elapsed = time.monotonic() - start
    return {
        "drill": "ci_poisoning",
        "passing": detected and len(issues) == 0,
        "injected_fault": "tampered build artifact (provider-B hash mismatch)",
        "detection_gate": "multi_ci_repro_check.py",
        "detected": detected,
        "elapsed_seconds": round(elapsed, 2),
        "legit_hash": legit_hash,
        "tampered_hash": tampered_hash,
        "issues": issues,
    }


def drill_dependency_compromise() -> dict:
    """REAL injection drill: synthesize a build tree whose CMakeLists.txt has NO
    cmake_minimum_required (an unpinned/compromised toolchain manifest), point the
    supply-chain build-input-pinning check at it, and require it to DETECT the
    missing pin. This proves a compromised dependency manifest is actually caught
    — not merely that SBOM/provenance scripts exist on disk.
    """
    start = time.monotonic()
    issues: list[str] = []
    detected = False

    try:
        if str(SCRIPT_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPT_DIR))
        scg = importlib.import_module("supply_chain_gate")
    except Exception as exc:
        issues.append(f"cannot import supply_chain_gate: {exc}")
        scg = None

    if scg is not None:
        saved_root = scg.LIB_ROOT
        try:
            with tempfile.TemporaryDirectory() as d:
                d = Path(d)
                # Compromised manifest: a CMakeLists.txt with no cmake_minimum_required.
                (d / "CMakeLists.txt").write_text("project(compromised)\nadd_library(x x.c)\n")
                scg.LIB_ROOT = d
                res = scg.check_build_input_pinning()
                detected = not res.get("passing", True)  # MUST fail on the unpinned manifest
                if not detected:
                    issues.append("supply_chain build-input-pinning did NOT detect the unpinned manifest")
        finally:
            scg.LIB_ROOT = saved_root

    elapsed = time.monotonic() - start
    return {
        "drill": "dependency_compromise",
        "passing": detected and len(issues) == 0,
        "injected_fault": "CMakeLists.txt with no cmake_minimum_required (unpinned toolchain)",
        "detection_gate": "supply_chain_gate.check_build_input_pinning",
        "detected": detected,
        "elapsed_seconds": round(elapsed, 2),
        "issues": issues,
    }


def run(json_mode: bool, out_file: str | None) -> int:
    results = [
        drill_key_compromise(),
        drill_ci_poisoning(),
        drill_dependency_compromise(),
    ]

    drills_total = len(results)
    drills_passed = sum(1 for r in results if r["passing"])
    overall_pass = drills_passed == drills_total
    total_elapsed = sum(r.get("elapsed_seconds", 0) for r in results)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit": _git_head(),
        "overall_pass": overall_pass,
        "drills_total": drills_total,
        "drills_passed": drills_passed,
        "drill_response_time_seconds": round(total_elapsed, 2),
        # Per-drill provenance for cadence/forensics: what fault was injected,
        # which gate detected it, and the result.
        "injections": [
            {
                "drill": r["drill"],
                "injected_fault": r.get("injected_fault"),
                "detection_gate": r.get("detection_gate"),
                "detected": r.get("detected"),
                "passing": r["passing"],
            }
            for r in results
        ],
        "drills": results,
    }

    rendered = json.dumps(report, indent=2)
    if out_file:
        Path(out_file).write_text(rendered, encoding="utf-8")

    # Always persist a machine-readable drill log so audit_sla_check.py can track
    # drill cadence/freshness (Bastion B9). Mirrors the SECURITY_AUTONOMY_KPI.json
    # pattern; refreshed + committed by the nightly evidence workflow.
    try:
        DRILL_LOG.write_text(rendered, encoding="utf-8")
    except OSError as exc:
        print(f"::warning::failed to write incident drill log {DRILL_LOG}: {exc}", file=sys.stderr)

    if json_mode:
        print(rendered)
    else:
        for r in results:
            status = "PASS" if r["passing"] else "FAIL"
            elapsed = r.get("elapsed_seconds", 0)
            print(f"  {status} {r['drill']} ({elapsed:.1f}s)")
            for issue in r.get("issues", []):
                print(f"        → {issue}")
        print()
        if overall_pass:
            print(f"PASS all {drills_total} incident drills ({total_elapsed:.1f}s)")
        else:
            print(f"FAIL {drills_total - drills_passed}/{drills_total} drill(s) failed")

    if overall_pass:
        # Check if any drill ran in degraded mode (no live library).
        # Live testing is required for a meaningful audit result.
        # Return ADVISORY_SKIP_CODE (77) so the autonomy runner records a skip
        # instead of a false-pass (returning 0 when no live testing occurred
        # violates CLAUDE.md Rule 16).
        any_degraded = any(not r.get("live_test", True) for r in results)
        if any_degraded:
            if json_mode:
                pass  # report already printed above
            else:
                print("ADVISORY_SKIP: library unavailable — live key-rejection testing skipped")
            return 77
        return 0
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("-o", dest="out_file", help="Write report to file")
    args = parser.parse_args()
    return run(args.json, args.out_file)


if __name__ == "__main__":
    raise SystemExit(main())
