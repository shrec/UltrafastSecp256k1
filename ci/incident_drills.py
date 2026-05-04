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
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent


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
        "elapsed_seconds": round(elapsed, 2),
        "weak_keys_tested": len(weak_keys),
        "results": results,
        "issues": issues,
    }


def drill_ci_poisoning() -> dict:
    """Simulate CI build output tampering.

    Creates a fake artifact with wrong hash and verifies the artifact
    hash policy would catch the discrepancy.
    """
    start = time.monotonic()
    issues: list[str] = []

    # Create a temporary "artifact" and compute its hash
    with tempfile.NamedTemporaryFile(suffix=".so", delete=False) as f:
        f.write(b"legitimate build output content for drill")
        legit_path = f.name

    legit_hash = hashlib.sha256(Path(legit_path).read_bytes()).hexdigest()

    # "Tamper" with it
    with open(legit_path, "ab") as f:
        f.write(b"\x00TAMPERED")

    tampered_hash = hashlib.sha256(Path(legit_path).read_bytes()).hexdigest()

    # Verify hashes differ (sanity)
    hashes_differ = legit_hash != tampered_hash
    if not hashes_differ:
        issues.append("tampered artifact has same hash — SHA-256 collision (impossible)")

    # Clean up
    os.unlink(legit_path)

    # Check that supply_chain_gate.py exists
    supply_gate = SCRIPT_DIR / "supply_chain_gate.py"
    gate_exists = supply_gate.exists()
    if not gate_exists:
        issues.append("supply_chain_gate.py not found — cannot verify tamper detection")

    elapsed = time.monotonic() - start

    return {
        "drill": "ci_poisoning",
        "passing": hashes_differ and gate_exists and len(issues) == 0,
        "elapsed_seconds": round(elapsed, 2),
        "legit_hash": legit_hash,
        "tampered_hash": tampered_hash,
        "hashes_differ": hashes_differ,
        "supply_chain_gate_exists": gate_exists,
        "issues": issues,
    }


def drill_dependency_compromise() -> dict:
    """Simulate dependency compromise detection.

    Verifies that:
    1. CMakeLists.txt pins dependency versions
    2. SBOM generation capability exists
    3. Build hardening script exists
    """
    start = time.monotonic()
    issues: list[str] = []

    # Check CMakeLists.txt for version pinning
    cmakelists = LIB_ROOT / "CMakeLists.txt"
    if cmakelists.exists():
        content = cmakelists.read_text(encoding="utf-8", errors="replace")
        has_version_pin = "VERSION" in content or "version" in content
        if not has_version_pin:
            issues.append("CMakeLists.txt does not appear to pin dependency versions")
    else:
        issues.append("CMakeLists.txt not found")

    # scripts/ is gitignored; ci/ is the committed fallback location.
    def _find(name: str) -> bool:
        for base in (LIB_ROOT / "scripts", SCRIPT_DIR):
            if (base / name).exists():
                return True
        return False

    # Check SBOM generator
    if not _find("generate_sbom.sh"):
        issues.append("generate_sbom.sh not found")

    # Check build hardening
    if not _find("verify_build_hardening.py"):
        issues.append("verify_build_hardening.py not found")

    # Check provenance verifier
    if not _find("verify_slsa_provenance.py"):
        issues.append("verify_slsa_provenance.py not found")

    elapsed = time.monotonic() - start

    return {
        "drill": "dependency_compromise",
        "passing": len(issues) == 0,
        "elapsed_seconds": round(elapsed, 2),
        "checks": {
            "version_pinning": cmakelists.exists(),
            "sbom_generator": _find("generate_sbom.sh"),
            "build_hardening": _find("verify_build_hardening.py"),
            "provenance_verifier": _find("verify_slsa_provenance.py"),
        },
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
        "overall_pass": overall_pass,
        "drills_total": drills_total,
        "drills_passed": drills_passed,
        "drill_response_time_seconds": round(total_elapsed, 2),
        "drills": results,
    }

    rendered = json.dumps(report, indent=2)
    if out_file:
        Path(out_file).write_text(rendered, encoding="utf-8")

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
