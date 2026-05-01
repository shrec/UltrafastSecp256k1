#!/usr/bin/env python3
"""
generate_verification_report.py — Stamp verification_report.md for a release.

Reads docs/AUDIT_READINESS_REPORT_v1.md as template and replaces live
placeholders with values derived from VERSION.txt and the source tree.

Usage:
    python3 ci/generate_verification_report.py [-o OUTPUT] [--version X.Y.Z]

If --version is omitted, reads VERSION.txt.
Output defaults to verification_report.md in the repo root.
"""

import argparse
import glob
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


def count_files(pattern: str, root: Path) -> int:
    """Count files matching a glob pattern under root."""
    return len(glob.glob(str(root / pattern)))


def detect_counts(root: Path) -> dict:
    """Derive live counts from the source tree."""
    audit_dir = root / "audit"
    exploit = count_files("test_exploit_*.cpp", audit_dir)
    wycheproof = count_files("test_wycheproof_*.cpp", audit_dir)
    mutation = count_files("test_mutation_*.cpp", audit_dir)
    catalog_total = exploit + wycheproof + mutation

    # Try to get CTest target count from any available build dir
    ctest_targets = 0
    for bdir_name in ("build", "build_opencl", "build_rel", "build-cuda"):
        ctest_file = root / bdir_name / "CTestTestfile.cmake"
        if not ctest_file.exists():
            # Check parent (suite repo wraps library)
            ctest_file = root.parent.parent / bdir_name / "CTestTestfile.cmake"
        if ctest_file.exists():
            try:
                import subprocess
                result = subprocess.run(
                    ["ctest", "--test-dir", str(ctest_file.parent), "-N"],
                    capture_output=True, text=True, timeout=30
                )
                m = re.search(r"Total Tests:\s*(\d+)", result.stdout)
                if m:
                    ctest_targets = int(m.group(1))
                    break
            except Exception:
                pass

    # Fallback: estimate from files if ctest unavailable
    if ctest_targets == 0:
        total_test_cpp = count_files("test_*.cpp", audit_dir)
        # Add rough estimate for non-audit tests
        tests_dir = root / "tests"
        if tests_dir.exists():
            total_test_cpp += count_files("test_*.cpp", tests_dir)
        ctest_targets = total_test_cpp

    non_exploit = ctest_targets - exploit if ctest_targets > exploit else 0

    return {
        "exploit_count": exploit,
        "wycheproof_count": wycheproof,
        "mutation_count": mutation,
        "catalog_total": catalog_total,
        "ctest_targets": ctest_targets,
        "non_exploit": non_exploit,
    }


def stamp_template(template: str, version: str, counts: dict) -> str:
    """Replace all live placeholders and hardcoded stale values."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    exploit = counts["exploit_count"]
    modules = counts["ctest_targets"]
    non_exploit = counts["non_exploit"]

    out = template

    # --- Placeholder-based substitutions (preferred path) ---
    subs = {
        "{{VERSION}}": version,
        "{{DATE}}": today,
        "{{EXPLOIT_COUNT}}": str(exploit),
        "{{MODULE_COUNT}}": str(modules),
        "{{NON_EXPLOIT_COUNT}}": str(non_exploit),
        "{{CTEST_TARGETS}}": str(modules),
        "{{CATALOG_TOTAL}}": str(counts["catalog_total"]),
    }
    for placeholder, value in subs.items():
        out = out.replace(placeholder, value)

    # --- Regex-based fallback for any remaining hardcoded values ---

    # "This report covers **UltrafastSecp256k1 vX.Y.Z+**"
    out = re.sub(
        r'(\*\*UltrafastSecp256k1\s+v)[\d]+\.[\d]+\.[\d]+(\+?\*\*)',
        rf'\g<1>{version}\g<2>',
        out
    )

    # "| Version | X.Y.Z |"
    out = re.sub(
        r'(\|\s*Version\s*\|\s*)[\d]+\.[\d]+\.[\d]+(\s*\|)',
        rf'\g<1>{version}\g<2>',
        out
    )

    # "| Report Date | YYYY-MM-DD |"
    out = re.sub(
        r'(\|\s*Report Date\s*\|\s*)[\d]{4}-[\d]{2}-[\d]{2}(\s*\|)',
        rf'\g<1>{today}\g<2>',
        out
    )

    # Footer: "*UltrafastSecp256k1 vX.Y.Z -- ..."
    out = re.sub(
        r'(\*UltrafastSecp256k1\s+v)[\d]+\.[\d]+\.[\d]+(\s+--)',
        rf'\g<1>{version}\g<2>',
        out
    )

    # "NNN exploit PoC" anywhere
    out = re.sub(
        r'\b\d+ exploit PoC',
        f'{exploit} exploit PoC',
        out
    )

    # "NNN unified\n> audit modules" or "NNN unified audit modules"
    out = re.sub(
        r'\b\d+(?= unified\s*\n?>\s*audit modules| unified audit modules)',
        str(modules),
        out
    )

    # "(NNN modules, N failure classes"
    out = re.sub(
        r'\(\d+ modules,',
        f'({modules} modules,',
        out
    )

    # "NNN probes)"
    out = re.sub(
        r'\b\d+ probes\)',
        f'{exploit} probes)',
        out
    )

    # "NN non-exploit modules + NNN exploit-PoC modules (NNN total)"
    out = re.sub(
        r'\d+ non-exploit modules \+ \d+ exploit-PoC modules \(\d+ total\)',
        f'{non_exploit} non-exploit modules + {exploit} exploit-PoC modules ({modules} total)',
        out
    )

    # "NN non-exploit + NNN exploit-PoC modules"
    out = re.sub(
        r'\d+ non-exploit \+ \d+ exploit-PoC modules',
        f'{non_exploit} non-exploit + {exploit} exploit-PoC modules',
        out
    )

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Generate verification_report.md for a release"
    )
    parser.add_argument("-o", "--output", default="verification_report.md",
                        help="Output file (default: verification_report.md)")
    parser.add_argument("--version",
                        help="Version string (default: read from VERSION.txt)")
    parser.add_argument("--template",
                        help="Template file (default: docs/AUDIT_READINESS_REPORT_v1.md)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent

    # Read version
    if args.version:
        version = args.version.lstrip("v")
    else:
        vfile = root / "VERSION.txt"
        if not vfile.exists():
            print("ERROR: VERSION.txt not found and --version not given", file=sys.stderr)
            sys.exit(1)
        version = vfile.read_text().strip()

    # Read template
    template_path = Path(args.template) if args.template else root / "docs" / "AUDIT_READINESS_REPORT_v1.md"
    if not template_path.exists():
        print(f"ERROR: Template not found: {template_path}", file=sys.stderr)
        sys.exit(1)
    template = template_path.read_text(encoding="utf-8")

    # Detect counts
    counts = detect_counts(root)
    print(f"Version:     {version}")
    print(f"Exploit PoC: {counts['exploit_count']}")
    print(f"CTest total: {counts['ctest_targets']}")
    print(f"Non-exploit: {counts['non_exploit']}")
    print(f"Catalog:     {counts['catalog_total']}")

    # Stamp
    result = stamp_template(template, version, counts)

    # Write
    out_path = Path(args.output)
    out_path.write_text(result, encoding="utf-8")
    print(f"Wrote {out_path} ({len(result)} bytes)")


if __name__ == "__main__":
    main()
