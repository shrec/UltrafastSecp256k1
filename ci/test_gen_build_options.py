#!/usr/bin/env python3
"""Self-test for gen_build_options.py (the BUILD_OPTIONS.md drift gate).

Paired unit test for the gate wired into ci/run_fast_gates.sh by commit 0d2edda6
(a SECURITY_CI_FILE change, so the convention requires a paired ci/test_*.py — see
check_security_fix_has_test.RETROACTIVELY_COVERED). Proves the gate's parser and the
deterministic renderer behave, and that the committed doc is in sync.

Run: python3 ci/test_gen_build_options.py   (exit 0 = all assertions pass)
"""
from __future__ import annotations

import sys

import gen_build_options as g


def main() -> int:
    fails = 0

    def check(name: str, cond: bool) -> None:
        nonlocal fails
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")
        if not cond:
            fails += 1

    # 1. parse_options extracts (name, desc, default, kind) for option() and
    #    cmake_dependent_option(), collapsing whitespace in the description.
    snippet = (
        'option(SECP256K1_BUILD_FOO  "Build the\n   foo module"  OFF)\n'
        'cmake_dependent_option(SECP256K1_BUILD_BAR "Bar needs foo" ON\n'
        '    "SECP256K1_BUILD_FOO" OFF)\n'
    )
    opts = {o[0]: o for o in g.parse_options(snippet)}
    check("parses option() name+desc+default+kind",
          opts.get("SECP256K1_BUILD_FOO") ==
          ("SECP256K1_BUILD_FOO", "Build the foo module", "OFF", "option"))
    check("parses cmake_dependent_option() (kind tagged, default=ON)",
          opts.get("SECP256K1_BUILD_BAR")[:3] == ("SECP256K1_BUILD_BAR", "Bar needs foo", "ON")
          and opts["SECP256K1_BUILD_BAR"][3] == "cmake_dependent_option")

    # 2. A non-option call must NOT be picked up.
    check("ignores non-option calls",
          len(g.parse_options('add_library(foo STATIC foo.c)\nif(BAR)\nendif()\n')) == 0)

    # 3. render() is deterministic (no timestamps) — required for a stable --check.
    check("render() is deterministic", g.render() == g.render())

    # 4. The committed docs/BUILD_OPTIONS.md must currently be in sync (gate green).
    #    gen_build_options.main() reads sys.argv directly, so drive --check via argv.
    saved = sys.argv
    try:
        sys.argv = ["gen_build_options.py", "--check"]
        in_sync = (g.main() == 0)
    finally:
        sys.argv = saved
    check("live BUILD_OPTIONS.md is in sync (--check passes)", in_sync)

    print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILURE(S)'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
