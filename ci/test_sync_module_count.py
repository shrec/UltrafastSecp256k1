#!/usr/bin/env python3
"""Self-test for canonical audit/module count documentation replacement."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_PATH = SCRIPT_DIR / "sync_module_count.py"


def load_sync_module_count():
    spec = importlib.util.spec_from_file_location("sync_module_count_selftest", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paired_count_replacement(module) -> list[str]:
    """make_replacements() must update paired headline counts and leave no
    stale numbers behind. Returns a list of failure messages (empty = pass)."""
    original = "\n".join([
        "# Summary",
        "GitHub cached README: 262 exploit PoCs / 367 modules.",
        "Standalone short form: 262 exploit PoCs.",
        "",
    ])
    updated, changes = module.make_replacements(
        original,
        total=431,
        exploit_mods=269,
        non_exploit=162,
        exploit_files=257,
        n_sections=10,
    )

    required = [
        "269 exploit PoCs / 431 modules",
        "Standalone short form: 269 exploit PoCs",
    ]
    missing = [needle for needle in required if needle not in updated]
    stale = ["262 exploit PoCs", "367 modules"]
    leftovers = [needle for needle in stale if needle in updated]

    failures = []
    if changes < 2 or missing or leftovers:
        failures.append("sync_module_count paired-count regression failed")
        failures.append(f"changes={changes}")
        failures.append(f"missing={missing}")
        failures.append(f"leftovers={leftovers}")
        failures.append(updated)
    return failures


def _write_fixture_tree(base: Path) -> Path:
    """Build a minimal, isolated fixture tree (NOT the real repo) that
    sync_module_count.py's main() can run against: a fake
    audit/unified_audit_runner.cpp with a known ALL_MODULES[]/SECTIONS[]
    shape (total=3, exploit_mods=2, non_exploit=1, n_sections=2), two
    test_exploit_*.cpp files (exploit_files=2), and a docs/ fixture doc
    with deliberately STALE counts. Returns the path to the fixture doc.
    """
    audit_dir = base / "audit"
    docs_dir = base / "docs"
    audit_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    runner_cpp = "\n".join([
        "static const AuditModule ALL_MODULES[] = {",
        '    {"mod_a", "exploit_poc"},',
        '    {"mod_b", "exploit_poc"},',
        '    {"mod_c", "math_invariants"},',
        "};",
        "",
        "static constexpr int NUM_MODULES = 3;",
        "",
        "static const SectionInfo SECTIONS[] = {",
        '    {"exploit_poc"},',
        '    {"math_invariants"},',
        "};",
        "",
        "static constexpr int NUM_SECTIONS = 2;",
        "",
    ])
    (audit_dir / "unified_audit_runner.cpp").write_text(runner_cpp)

    (audit_dir / "test_exploit_a.cpp").write_text("// fixture\n")
    (audit_dir / "test_exploit_b.cpp").write_text("// fixture\n")

    # WHY_ULTRAFASTSECP256K1.md is in the fixed DOC_FILES list, so it is
    # always scanned regardless of the dynamic docs/*.md glob.
    doc_path = docs_dir / "WHY_ULTRAFASTSECP256K1.md"
    doc_path.write_text("# Fixture Doc\n\nSummary: 99 exploit PoC modules registered.\n")
    return doc_path


def test_check_mode_performs_zero_filesystem_writes(module) -> list[str]:
    """Locks in the invariant (independently verified live against this repo:
    `git status --short` shows zero diff before/after `--check`) that
    sync_module_count.py's --check/--dry-run mode NEVER writes to disk, even
    when it correctly detects drift and reports a nonzero exit code. Builds
    an isolated fixture tree (never touches the real repo docs), points
    module.BASE at it, and asserts:
      1. --check reports drift (exit code 1) against the deliberately stale
         fixture doc.
      2. The fixture doc's mtime and byte content are UNCHANGED after
         --check runs (an artificially old mtime is set first so any write
         would be unambiguously detectable, not just within clock
         resolution).
      3. The SAME fixture, run WITHOUT --check, DOES write and DOES change
         the mtime/content -- proving the drift was real and that --check's
         "no write" is a deliberate skip, not an artifact of there being
         nothing to change.
    Returns a list of failure messages (empty = pass).
    """
    failures: list[str] = []
    original_base = module.BASE
    original_argv = sys.argv

    try:
        with tempfile.TemporaryDirectory(prefix="sync_module_count_selftest_") as tmp:
            base = Path(tmp)
            doc_path = _write_fixture_tree(base)
            module.BASE = base

            stale_bytes_before = doc_path.read_bytes()
            old_mtime = 1_000_000_000  # 2001-09-09, unambiguously not "now"
            os.utime(doc_path, (old_mtime, old_mtime))

            # -- (1)+(2): --check must detect drift but must NOT write -----
            sys.argv = ["sync_module_count.py", "--check"]
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc_check = module.main()

            if rc_check != 1:
                failures.append(
                    f"expected --check to report drift (exit 1) against the stale fixture, got {rc_check}"
                )

            stat_after_check = doc_path.stat()
            if stat_after_check.st_mtime != old_mtime:
                failures.append(
                    "sync_module_count.py --check modified the fixture doc's mtime "
                    f"(expected {old_mtime}, got {stat_after_check.st_mtime}) -- --check must never write"
                )
            bytes_after_check = doc_path.read_bytes()
            if bytes_after_check != stale_bytes_before:
                failures.append(
                    "sync_module_count.py --check modified the fixture doc's content -- "
                    "--check must never write"
                )

            # -- (3): the same fixture, run for real, DOES write -----------
            sys.argv = ["sync_module_count.py"]
            buf2 = io.StringIO()
            with contextlib.redirect_stdout(buf2):
                rc_real = module.main()

            if rc_real != 0:
                failures.append(f"expected a real (non-check) run to succeed (exit 0), got {rc_real}")

            bytes_after_real = doc_path.read_bytes()
            if bytes_after_real == stale_bytes_before:
                failures.append(
                    "sync_module_count.py without --check left the fixture doc unchanged -- "
                    "the drift was not real, so this test doesn't prove --check skipped a write"
                )
            if b"2 exploit-PoC modules" not in bytes_after_real:
                failures.append(
                    "sync_module_count.py (real run) did not rewrite the stale count to the "
                    f"expected fixed value; content={bytes_after_real!r}"
                )
    finally:
        module.BASE = original_base
        sys.argv = original_argv

    return failures


def main() -> int:
    module = load_sync_module_count()

    failures = []
    failures += test_paired_count_replacement(module)
    failures += test_check_mode_performs_zero_filesystem_writes(module)

    if failures:
        for line in failures:
            print(line, file=sys.stderr)
        return 1

    print("sync_module_count paired-count regression: OK")
    print("sync_module_count --check zero-filesystem-writes regression: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
