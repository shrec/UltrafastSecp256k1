#!/usr/bin/env python3
"""
test_caas_integrity.py -- CAAS Pipeline Integrity Self-Test

Tests that the CAAS audit pipeline scripts themselves are correct and
not vulnerable to the bugs discovered in the 2026-04-27 quality audit:

  C-2: supply_chain_gate.py stunt-double gates (--help instead of real check)
  C-3: evidence_governance.py HMAC excludes `reason` field
  H-3: HMAC key hardcoded without env-var override

Each test is numbered CAAS-GATE-*, CAAS-HMAC-*, CAAS-KEY-* for traceability
to the quality report finding IDs.

Usage:
    python3 ci/test_caas_integrity.py        # all tests
    python3 ci/test_caas_integrity.py --json  # JSON output
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent

g_pass = 0
g_fail = 0
g_results: list[dict[str, Any]] = []

def check(cond: bool, name: str, detail: str = "") -> bool:
    global g_pass, g_fail
    status = "PASS" if cond else "FAIL"
    msg = f"  [{status}] {name}"
    if detail and not cond:
        msg += f"\n         {detail}"
    print(msg)
    sys.stdout.flush()
    if cond:
        g_pass += 1
    else:
        g_fail += 1
    g_results.append({"name": name, "pass": cond, "detail": detail})
    return cond


def load_module(path: Path):
    """Load a Python module from a file path without executing __main__."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================================
# C-2: supply_chain_gate.py stunt-double gates
# ============================================================================

def test_supply_chain_gate_no_stunt_double():
    """CAAS-GATE-1..3: supply_chain_gate.py must not call --help for real checks."""
    gate_path = SCRIPT_DIR / "supply_chain_gate.py"
    if not gate_path.exists():
        check(False, "CAAS-GATE-0: supply_chain_gate.py exists", f"not found: {gate_path}")
        return

    source = gate_path.read_text(encoding="utf-8", errors="replace")

    # CAAS-GATE-1: check_build_hardening must not call --help as the main check
    # The old stunt-double was: subprocess.run([..., "--help"], ...)
    # Fixed version: real CMakeCache inspection without --help subprocess
    def _has_help_stunt(source: str, funcname: str) -> bool:
        """Return True if funcname in source contains a subprocess call with --help."""
        import re
        # Find the function body
        pattern = re.compile(
            rf'def\s+{re.escape(funcname)}\s*\(.*?\n'
            rf'((?:    .*\n|\n)*)',
            re.MULTILINE,
        )
        m = pattern.search(source)
        if not m:
            return False
        body = m.group(1)
        # Look for subprocess.run(..., "--help", ...) pattern
        return bool(re.search(r'subprocess\.run\s*\(.*?["\']--help["\']', body, re.DOTALL))

    check(
        not _has_help_stunt(source, "check_build_hardening"),
        "CAAS-GATE-1: check_build_hardening does not call --help (stunt-double fixed)",
        "The function still uses subprocess with --help instead of real verification.",
    )
    check(
        not _has_help_stunt(source, "check_slsa_provenance"),
        "CAAS-GATE-2: check_slsa_provenance does not call --help (stunt-double fixed)",
        "The function still uses subprocess with --help instead of real verification.",
    )

    # CAAS-GATE-3: check_build_hardening must inspect actual hardening flags
    # It should look for fstack-protector or _FORTIFY_SOURCE in CMake config
    has_real_check = (
        "fstack-protector" in source or
        "FORTIFY_SOURCE" in source or
        "CMakeCache" in source or
        "CMakeLists" in source
    )
    check(
        has_real_check,
        "CAAS-GATE-3: check_build_hardening inspects real CMake/compiler flags",
        "No fstack-protector/FORTIFY_SOURCE/CMakeCache reference found in source.",
    )


# ============================================================================
# C-3: evidence_governance.py HMAC includes reason field
# ============================================================================

def test_evidence_governance_hmac_reason():
    """CAAS-HMAC-1..4: evidence_governance.py must include `reason` in HMAC."""
    gov_path = SCRIPT_DIR / "evidence_governance.py"
    if not gov_path.exists():
        check(False, "CAAS-HMAC-0: evidence_governance.py exists", f"not found: {gov_path}")
        return

    source = gov_path.read_text(encoding="utf-8", errors="replace")

    # CAAS-HMAC-1: The old bug was `if key == 'reason': continue` in HMAC loop
    # or explicit exclusion of reason from the HMAC payload dict.
    # Check that the source does NOT exclude reason from HMAC.
    bad_patterns = [
        "if key == 'reason': continue",
        "if key == \"reason\": continue",
        "# exclude.*reason",
        "reason.*excluded",
    ]
    import re
    reason_excluded = any(re.search(p, source, re.IGNORECASE) for p in bad_patterns)
    check(
        not reason_excluded,
        "CAAS-HMAC-1: evidence_governance.py does not exclude `reason` from HMAC (C-3 fixed)",
        "Found explicit exclusion of `reason` field from HMAC computation.",
    )

    # CAAS-HMAC-2: The `reason` key must appear in the HMAC payload construction.
    # After fix, _compute_hmac should include reason in the payload dict.
    reason_in_hmac = "reason" in source and (
        '"reason"' in source or "'reason'" in source
    )
    check(
        reason_in_hmac,
        "CAAS-HMAC-2: `reason` field referenced in evidence_governance.py source",
        "No reference to `reason` field found — HMAC may still exclude it.",
    )

    # CAAS-HMAC-3: Behavioral test — load the module and verify that changing
    # the reason field changes the HMAC.
    try:
        # Set a known key for reproducibility
        old_key = os.environ.get("CAAS_HMAC_KEY")
        os.environ["CAAS_HMAC_KEY"] = "test-integrity-key-caas-hmac-3"
        try:
            mod = load_module(gov_path)
        finally:
            if old_key is None:
                os.environ.pop("CAAS_HMAC_KEY", None)
            else:
                os.environ["CAAS_HMAC_KEY"] = old_key

        if mod is None:
            check(False, "CAAS-HMAC-3: evidence_governance module loads", "load failed")
            return

        # Find the compute_hmac function
        compute_fn = getattr(mod, "_compute_hmac", None)
        if compute_fn is None:
            check(False, "CAAS-HMAC-3: _compute_hmac function exists in module")
            return

        base_record = {
            "who": "test",
            "what": "test action",
            "when": "2026-04-27T00:00:00Z",
            "commit": "abc123",
            "binary_hash": "deadbeef",
            "verdict": "PASS",
            "reason": "original reason",
        }
        record_modified_reason = dict(base_record)
        record_modified_reason["reason"] = "MODIFIED REASON — should change HMAC"

        try:
            hmac_original = compute_fn(base_record)
            hmac_modified = compute_fn(record_modified_reason)
        except Exception as e:
            check(False, "CAAS-HMAC-3: _compute_hmac callable", str(e))
            return

        check(
            hmac_original != hmac_modified,
            "CAAS-HMAC-3: changing `reason` field changes HMAC (reason is included in MAC)",
            f"HMAC unchanged: {hmac_original!r} — `reason` is still excluded from HMAC.",
        )

        # CAAS-HMAC-4: Changing a non-reason field also changes the HMAC (sanity)
        record_modified_who = dict(base_record)
        record_modified_who["who"] = "attacker"
        hmac_modified_who = compute_fn(record_modified_who)
        check(
            hmac_original != hmac_modified_who,
            "CAAS-HMAC-4: changing `who` field also changes HMAC (sanity check)",
        )

    except Exception as e:
        check(False, "CAAS-HMAC-3: behavioral HMAC test", str(e))


# ============================================================================
# H-3: HMAC key env-var override
# ============================================================================

def test_hmac_key_env_override():
    """CAAS-KEY-1..2: evidence_governance.py must support CAAS_HMAC_KEY env var."""
    gov_path = SCRIPT_DIR / "evidence_governance.py"
    if not gov_path.exists():
        return

    source = gov_path.read_text(encoding="utf-8", errors="replace")

    # CAAS-KEY-1: Source must reference CAAS_HMAC_KEY env var
    check(
        "CAAS_HMAC_KEY" in source,
        "CAAS-KEY-1: evidence_governance.py references CAAS_HMAC_KEY env var (H-3 fixed)",
        "No reference to CAAS_HMAC_KEY — HMAC key is still hardcoded.",
    )

    # CAAS-KEY-2: Behavioral test — two keys produce different HMACs
    try:
        os.environ["CAAS_HMAC_KEY"] = "key-alpha-test"
        mod_alpha = load_module(gov_path)
        os.environ["CAAS_HMAC_KEY"] = "key-beta-test"
        mod_beta = load_module(gov_path)
        os.environ.pop("CAAS_HMAC_KEY", None)

        if mod_alpha is None or mod_beta is None:
            check(False, "CAAS-KEY-2: modules load with different CAAS_HMAC_KEY")
            return

        fn_a = getattr(mod_alpha, "_compute_hmac", None)
        fn_b = getattr(mod_beta, "_compute_hmac", None)
        if fn_a is None or fn_b is None:
            check(False, "CAAS-KEY-2: _compute_hmac found in both module instances")
            return

        record = {"who": "x", "what": "y", "when": "z",
                  "commit": "c", "binary_hash": "b", "verdict": "v", "reason": "r"}
        try:
            hmac_a = fn_a(record)
            hmac_b = fn_b(record)
            check(
                hmac_a != hmac_b,
                "CAAS-KEY-2: different CAAS_HMAC_KEY values produce different HMACs",
                "Same HMAC produced with different keys — key is not being read.",
            )
        except Exception as e:
            check(False, "CAAS-KEY-2: _compute_hmac callable with both keys", str(e))

    except Exception as e:
        check(False, "CAAS-KEY-2: HMAC key env-var behavioral test", str(e))


# ============================================================================
# Main
# ============================================================================

def main():
    use_json = "--json" in sys.argv

    if not use_json:
        print("[test_caas_integrity] CAAS Pipeline Integrity Self-Test")
        print("[test_caas_integrity] Testing C-2 (stunt-double gates),"
              " C-3 (HMAC reason), H-3 (env-var key)")
        print()

    test_supply_chain_gate_no_stunt_double()
    test_evidence_governance_hmac_reason()
    test_hmac_key_env_override()

    total = g_pass + g_fail
    if not use_json:
        print()
        print(f"[test_caas_integrity] {g_pass}/{total} passed", end="")
        if g_fail:
            print(f"  ({g_fail} FAILED)")
        else:
            print("  (ALL PASS)")

    if use_json:
        print(json.dumps({
            "suite": "test_caas_integrity",
            "passed": g_pass,
            "failed": g_fail,
            "total": total,
            "results": g_results,
        }, indent=2))

    return 1 if g_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
