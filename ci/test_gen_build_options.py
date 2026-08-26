#!/usr/bin/env python3
"""Self-test for gen_build_options.py (the BUILD_OPTIONS.md drift gate).

Paired unit test for the gate wired into ci/run_fast_gates.sh by commit 0d2edda6
(a SECURITY_CI_FILE change, so the convention requires a paired ci/test_*.py — see
check_security_fix_has_test.RETROACTIVELY_COVERED). Proves the gate's parser and the
deterministic renderer behave, and that the committed doc is in sync.

Run: python3 ci/test_gen_build_options.py   (exit 0 = all assertions pass)
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import gen_build_options as g

# Scratch fixtures for the tests below must stay inside the repo tree (never
# the system temp dir) per repo policy: all scratch/temp state lives under an
# ignored, repo-local out/ path. Captured from g.ROOT before any test mutates
# it, so this always resolves to the real repo root regardless of test order.
SCRATCH_PARENT = g.ROOT / "out" / "ci_scratch"


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

    # 5. .aiworkhub is excluded from recursive CMake discovery, even a
    #    CMakeLists.txt nested several directories deep under a runtime
    #    worktree path (.aiworkhub/runtime/worktrees/<id>/worktree/...).
    SCRATCH_PARENT.mkdir(parents=True, exist_ok=True)
    saved_root = g.ROOT
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT,
                                      prefix="gen_build_options_test_") as tmp_root:
        try:
            tmp_root = Path(tmp_root)
            (tmp_root / "CMakeLists.txt").write_text(
                'option(SECP256K1_BUILD_TOP "Top-level option" ON)\n'
            )
            nested = tmp_root / ".aiworkhub" / "runtime" / "worktrees" / "deadbeef" / "worktree"
            nested.mkdir(parents=True)
            (nested / "CMakeLists.txt").write_text(
                'option(SECP256K1_BUILD_NESTED_AIWORKHUB "Should never surface" ON)\n'
            )

            g.ROOT = tmp_root
            grouped, total, files = g.collect()

            all_names = {n for opts in grouped.values() for n in opts}
            check(".aiworkhub subtree option is excluded",
                  "SECP256K1_BUILD_NESTED_AIWORKHUB" not in all_names)
            check("sibling top-level option outside .aiworkhub is still found",
                  "SECP256K1_BUILD_TOP" in all_names)
            check("Generated-from file list contains no .aiworkhub path",
                  not any(".aiworkhub" in str(f) for f in files))

            # 6. A nested runtime worktree that disappears between scans (a
            #    concurrent AIWorkHub task tearing it down) must not change the
            #    result for the rest of the tree — output stays deterministic.
            shutil.rmtree(tmp_root / ".aiworkhub")
            grouped2, total2, files2 = g.collect()
            check("collect() is stable when the .aiworkhub subtree vanishes entirely",
                  grouped2 == grouped and total2 == total and files2 == files)
        finally:
            g.ROOT = saved_root

    # 7. A CMakeLists.txt removed between discovery (the os.walk pass) and
    #    read (e.g. a nested runtime worktree torn down mid-scan) must be
    #    skipped, not raise — this is the actual race, not just the .aiworkhub
    #    prune from checks 5/6.
    saved_root = g.ROOT
    saved_iter = g._iter_cmakelists
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT,
                                      prefix="gen_build_options_test_race_") as tmp_race:
        try:
            tmp_race = Path(tmp_race)
            ghost_dir = tmp_race / "ghost"
            ghost_dir.mkdir(parents=True)
            ghost = ghost_dir / "CMakeLists.txt"
            ghost.write_text('option(SECP256K1_BUILD_GHOST "Ghost" ON)\n')
            real_iter = saved_iter

            def _iter_then_delete():
                for p in real_iter():
                    if p == ghost:
                        ghost.unlink()
                    yield p

            g.ROOT = tmp_race
            g._iter_cmakelists = _iter_then_delete
            crashed = False
            try:
                grouped3, _total3, files3 = g.collect()
            except OSError:
                crashed = True
            check("collect() tolerates a CMakeLists.txt removed after discovery, before read",
                  not crashed)
            if not crashed:
                check("the removed CMakeLists.txt contributes no option and no file entry",
                      "SECP256K1_BUILD_GHOST" not in
                      {n for opts in grouped3.values() for n in opts}
                      and not any(str(f) == str(ghost.relative_to(tmp_race)) for f in files3))
        finally:
            g.ROOT = saved_root
            g._iter_cmakelists = saved_iter

    print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILURE(S)'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
