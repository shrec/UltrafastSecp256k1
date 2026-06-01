#!/usr/bin/env python3
"""check_advisory_json_rule16.py — CAAS-CI-002 real Rule-16 enforcement.

Rule 16 (CLAUDE.md): advisory CAAS modules that skip due to absent infrastructure
MUST return ADVISORY_SKIP_CODE (77), never 0 — because 0 means PASS and a skipped
advisory would be falsely reported as passing.

The original enforcement (ci/check_advisory_skip_returns.sh) runs per-module
`*_standalone` binaries. CAAS-CI-002 found those binaries are NEVER built at the
three CI choke points (the jobs build only ufsecp_shared / unified_audit_runner),
so that check permanently soft-skips (exit 77 → ::warning) and enforces nothing.

This gate enforces Rule 16 from the unified_audit_runner JSON report — which IS
produced at every choke point — mirroring the working inline check in
gate.yml's shim-gate (the only place Rule 16 was actually enforced). A GPU/infra
advisory module that reports passed=true AND return_code==0 in an environment
where its runtime is absent is a silent false-PASS.

Usage:
  check_advisory_json_rule16.py [REPORT.json]
If no path is given, searches $RULE16_AUDIT_REPORT then common report locations.

Exit 0 = no Rule-16 violation, 1 = violation/real failure, 77 = no report found.
"""
import json
import os
import sys
from pathlib import Path

# Source-content advisory tests that legitimately return 0 when their target
# source file is present at the runtime cwd (no GPU runtime invoked). Kept in
# sync with gate.yml shim-gate's inline allowlist.
SOURCE_CONTENT_ALLOWLIST = {
    "test_exploit_metal_schnorr_aux_rand",
    "test_exploit_metal_batch_failclosed",
    "test_exploit_opencl_runner_key_erase",
}
GPU_KEYWORDS = ("gpu", "cuda", "metal", "opencl", "rocm", "hip")

SEARCH = [
    os.environ.get("RULE16_AUDIT_REPORT", ""),
    "audit_report.json",
    "shim_audit_report.json",
    "out/ci-shim/audit_report.json",
    "out/ci-audit/audit_report.json",
    "out/release/audit_report.json",
    "out/auditor/audit_report.json",
]


def find_report(argv) -> Path | None:
    if len(argv) > 1 and argv[1]:
        p = Path(argv[1])
        return p if p.exists() else None
    for cand in SEARCH:
        if cand and Path(cand).exists():
            return Path(cand)
    # last resort: any audit_report.json under out/
    hits = sorted(Path("out").glob("**/audit_report.json")) if Path("out").is_dir() else []
    return hits[0] if hits else None


def main(argv) -> int:
    report = find_report(argv)
    if report is None:
        print("::warning::check_advisory_json_rule16: no unified_audit_runner JSON "
              "report found — cannot enforce Rule 16 from JSON")
        return 77

    try:
        r = json.loads(report.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"::error::check_advisory_json_rule16: cannot read {report}: {e}")
        return 1

    modules = [m for s in r.get("sections", []) for m in s.get("modules", [])]
    if not modules:
        print(f"::error::check_advisory_json_rule16: no modules in {report} "
              "(sections parse failed or empty report)")
        return 1

    # GPU/infra advisory modules that returned 0 (PASS) where they should skip (77).
    false_pass = []
    for mod in modules:
        if mod.get("advisory") is True and mod.get("passed") is True and mod.get("return_code") == 0:
            mid = mod.get("id", "")
            if mid in SOURCE_CONTENT_ALLOWLIST:
                continue
            if any(kw in mid.lower() for kw in GPU_KEYWORDS):
                false_pass.append(mid)
    if false_pass:
        print(f"::error::check_advisory_json_rule16: GPU/infra advisory modules "
              f"returned 0 instead of 77 (Rule 16 false-PASS): {false_pass}")
        return 1

    # Non-advisory real failures.
    failed = [m.get("id", "?") for m in modules
              if not m.get("passed") and not (m.get("advisory") and m.get("return_code") == 77)]
    if failed:
        print(f"::error::check_advisory_json_rule16: non-advisory modules FAILED: {failed}")
        return 1

    skipped = len([m for m in modules
                   if m.get("advisory") is True and m.get("return_code") == 77])
    print(f"check_advisory_json_rule16: Rule 16 OK from {report.name} — "
          f"{len(modules)} modules, {skipped} advisory-skipped (77), 0 false-PASS, 0 failed [PASS]")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
