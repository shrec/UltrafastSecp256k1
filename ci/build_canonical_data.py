#!/usr/bin/env python3
"""
build_canonical_data.py
=======================
Generates docs/canonical_data.json from source-of-truth files.

All numbers in this file are DERIVED — never manually edited.
Run this after any structural change (new exploit test, new workflow, etc.)
and commit the updated canonical_data.json.

Sources:
  version          ← VERSION.txt
  exploit_poc_*    ← audit/unified_audit_runner.cpp  (ALL_MODULES table)
  ci_workflow_*    ← .github/workflows/*.yml count
  ct_pipelines     ← .github/workflows/ct-*.yml count
  bitcoin_core_*   ← docs/BITCOIN_CORE_TEST_RESULTS.json
  last_updated     ← today's date (UTC)
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT       = SCRIPT_DIR.parent
OUT        = ROOT / "docs" / "canonical_data.json"


# ---------------------------------------------------------------------------
def _version() -> str:
    p = ROOT / "VERSION.txt"
    return p.read_text().strip() if p.exists() else "unknown"


def _module_counts() -> dict:
    runner = ROOT / "audit" / "unified_audit_runner.cpp"
    if not runner.exists():
        return {"exploit_poc_count": 0, "non_exploit_modules": 0, "total_modules": 0}

    text = runner.read_text(errors="replace")

    # Extract the ALL_MODULES[] block (same logic as sync_module_count.py)
    m = re.search(
        r'static const AuditModule ALL_MODULES\[\]\s*=\s*\{(.+?)^\};',
        text, re.DOTALL | re.MULTILINE
    )
    if not m:
        return {"exploit_poc_count": 0, "non_exploit_modules": 0, "total_modules": 0}

    block = m.group(1)
    total   = len(re.findall(r'^\s*\{\s*"[a-z]', block, re.MULTILINE))
    exploit = len(re.findall(r'^\s*\{\s*"exploit_', block, re.MULTILINE))
    non_exploit = total - exploit

    return {
        "exploit_poc_count":   exploit,
        "non_exploit_modules": non_exploit,
        "total_modules":       total,
    }


def _workflow_counts() -> dict:
    wf_dir = ROOT / ".github" / "workflows"
    if not wf_dir.exists():
        return {"ci_workflow_count": 0, "ct_pipeline_count": 0}

    all_ymls = list(wf_dir.glob("*.yml")) + list(wf_dir.glob("*.yaml"))
    ct_ymls  = [f for f in all_ymls if re.search(r'ct[-_]|valgrind.*ct|dudect', f.name, re.I)]

    return {
        "ci_workflow_count": len(all_ymls),
        "ct_pipeline_count": len(ct_ymls),
    }


def _bitcoin_core_tests() -> dict:
    p = ROOT / "docs" / "BITCOIN_CORE_TEST_RESULTS.json"
    if not p.exists():
        return {"bitcoin_core_tests_pass": 0, "bitcoin_core_tests_total": 0}
    try:
        data = json.loads(p.read_text())
        summary = data.get("summary", data)  # may be nested under "summary"
        return {
            "bitcoin_core_tests_pass":  summary.get("passed", 0),
            "bitcoin_core_tests_total": summary.get("total",  0),
        }
    except Exception:
        return {"bitcoin_core_tests_pass": 0, "bitcoin_core_tests_total": 0}


def _shim_functions() -> dict:
    """Count SECP256K1_API declarations in shim headers."""
    shim_inc = ROOT / "compat" / "libsecp256k1_shim" / "include"
    if not shim_inc.exists():
        return {"shim_api_function_count": 0}
    count = 0
    for h in shim_inc.rglob("*.h"):
        count += len(re.findall(r'SECP256K1_API\b', h.read_text(errors="replace")))
    return {"shim_api_function_count": count}


# ---------------------------------------------------------------------------
def build() -> dict:
    data: dict = {}
    data["version"]      = _version()
    data["last_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    data.update(_module_counts())
    data.update(_workflow_counts())
    data.update(_bitcoin_core_tests())
    data.update(_shim_functions())
    return data


def main() -> int:
    dry_run = "--dry-run" in sys.argv

    data = build()

    print("canonical_data:")
    for k, v in data.items():
        print(f"  {k}: {v}")

    if dry_run:
        # In dry-run: check if existing file matches; fail if drift
        if OUT.exists():
            existing = json.loads(OUT.read_text())
            existing.pop("last_updated", None)
            new = dict(data)
            new.pop("last_updated", None)
            drift = {k: (existing.get(k), v) for k, v in new.items() if existing.get(k) != v}
            if drift:
                print("\n[DRIFT DETECTED]")
                for k, (old, new_v) in drift.items():
                    print(f"  {k}: {old!r} → {new_v!r}")
                print("Run: python3 ci/build_canonical_data.py")
                return 1
            print("\n[OK] canonical_data.json is up-to-date")
        else:
            print("\n[WARN] canonical_data.json does not exist yet")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2) + "\n")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
