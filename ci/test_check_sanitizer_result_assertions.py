#!/usr/bin/env python3
"""Self-test for check_sanitizer_result_assertions.py (CAAS6-01 gate).

Proves the gate FLAGS a fail-open memcheck block and PASSES a fail-closed one — so
the gate cannot silently regress to a no-op. Mirrors test_check_ct_branches.py.

Exit 0 = gate behaves correctly; 1 = gate is broken.
"""
from __future__ import annotations

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load():
    spec = importlib.util.spec_from_file_location(
        "check_sanitizer_result_assertions",
        os.path.join(_HERE, "check_sanitizer_result_assertions.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FAIL_OPEN = """\
jobs:
  valgrind:
    name: Valgrind Memcheck
    steps:
      - name: Run tests under Valgrind
        run: |
          ctest --test-dir build -T MemCheck || true
          if grep -q 'ERROR SUMMARY: [1-9]' build/Testing/Temporary/MemoryChecker.*.log 2>/dev/null; then
            exit 1
          fi
"""

FAIL_CLOSED = """\
jobs:
  valgrind:
    name: Valgrind Memcheck
    steps:
      - name: Run tests under Valgrind
        run: |
          ctest --test-dir build -T MemCheck || true
          shopt -s nullglob
          memcheck_logs=(build/Testing/Temporary/MemoryChecker.*.log)
          if [ "${#memcheck_logs[@]}" -eq 0 ]; then
            echo "::error::no logs"
            exit 1
          fi
          if grep -q 'ERROR SUMMARY: [1-9]' build/Testing/Temporary/MemoryChecker.*.log 2>/dev/null; then
            exit 1
          fi
"""

# A job with no memcheck at all must never be flagged.
NO_MEMCHECK = """\
jobs:
  tsan:
    steps:
      - name: Test under TSan
        run: ctest --test-dir build --output-on-failure
"""


def main() -> int:
    mod = _load()
    failures = []

    open_problems = mod.scan_workflow_text(FAIL_OPEN)
    if not open_problems:
        failures.append("FAIL: gate did NOT flag a fail-open memcheck block")

    closed_problems = mod.scan_workflow_text(FAIL_CLOSED)
    if closed_problems:
        failures.append(f"FAIL: gate wrongly flagged a fail-closed block: {closed_problems}")

    none_problems = mod.scan_workflow_text(NO_MEMCHECK)
    if none_problems:
        failures.append(f"FAIL: gate flagged a non-memcheck job: {none_problems}")

    if failures:
        for f in failures:
            print(" ", f)
        return 1
    print("test_check_sanitizer_result_assertions: OK (flags fail-open, passes fail-closed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
