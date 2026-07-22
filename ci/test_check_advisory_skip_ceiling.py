#!/usr/bin/env python3
"""
test_check_advisory_skip_ceiling.py — unit test for ci/check_advisory_skip_ceiling.py.

Guards the meta-gate against silent loosening: the advisory-module ceiling must stay
TIGHT (== actual count, no unreviewed slack), the frozen twin must match, and the gate
must FAIL when the advisory count exceeds the ceiling.

Self-contained. Exit 0 = pass, 1 = fail.
"""
import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE = os.path.join(ROOT, "ci", "check_advisory_skip_ceiling.py")

failures = []


def check(cond, msg):
    print(("  ok  : " if cond else "  FAIL: ") + msg)
    if not cond:
        failures.append(msg)


def main() -> int:
    spec = importlib.util.spec_from_file_location("chk", GATE)
    chk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chk)  # import-time assert(CEILING == FROZEN) runs here

    # 1. Frozen twin matches (silent loosening is blocked at import).
    check(chk.ADVISORY_CEILING == chk.ADVISORY_CEILING_FROZEN,
          f"ceiling == frozen twin ({chk.ADVISORY_CEILING})")

    # 2. Ceiling is TIGHT: no slack above the actual advisory count in the runner.
    actual = chk.count_advisory_modules(chk.ROOT / chk.RUNNER_PATH
                                        if hasattr(chk, "ROOT") else chk.RUNNER_PATH)
    if actual < 0:  # runner path may be relative to repo root
        actual = chk.count_advisory_modules(chk.RUNNER_PATH)
    check(actual >= 0, f"advisory count is readable ({actual})")
    check(chk.ADVISORY_CEILING == actual,
          f"ceiling is tight: ceiling({chk.ADVISORY_CEILING}) == actual({actual}) "
          f"(no unreviewed slack)")

    # 3. Gate isn't rubber-stamped: simulate a HYPOTHETICAL (count+1)-th advisory
    #    module against the CURRENT (un-bumped) ceiling and prove main() would
    #    reject it. This is what actually stops a silent ceiling bump-to-match —
    #    without this, checks 1-2 above would pass again the moment someone bumps
    #    ADVISORY_CEILING to whatever count() returns, rubber-stamping any growth.
    #    We monkeypatch count_advisory_modules() to return actual+1 and re-run the
    #    real main() logic (not a re-implementation) so a future refactor of the
    #    threshold comparison in main() is exercised by this test too.
    real_count_fn = chk.count_advisory_modules
    try:
        chk.count_advisory_modules = lambda *_a, **_kw: actual + 1
        simulated_rc = chk.main()
    finally:
        chk.count_advisory_modules = real_count_fn
    check(simulated_rc == 1,
          f"gate FAILS closed on a hypothetical advisory count of {actual + 1} "
          f"against the current ceiling ({chk.ADVISORY_CEILING}) — main() returned "
          f"{simulated_rc}, expected 1")

    # 4. Sanity: the real (unpatched) count must not itself already exceed the
    #    ceiling — i.e. check 3 above is testing a genuinely hypothetical case,
    #    not silently validating against an already-broken gate.
    check(actual <= chk.ADVISORY_CEILING,
          f"actual advisory count ({actual}) does not exceed ceiling "
          f"({chk.ADVISORY_CEILING}) outside of the simulation")

    # 5. Regression (2026-07-18, issue #335 round 6): a row with an inline
    #    `/* ... */` comment between `true` and the closing `}` -- the exact
    #    shape of the real "fiat_crypto_link" row -- must still be counted.
    #    A comment is not whitespace; a resolver that does not strip
    #    comments first silently undercounts rows shaped this way. Build a
    #    minimal fixture with one plain advisory row and one
    #    comment-trailing advisory row and confirm both are counted.
    import tempfile
    fixture_src = (
        'static const AuditModule ALL_MODULES[] = {\n'
        '    { "mandatory_one", "n", "s", mandatory_one_run, false },\n'
        '    { "advisory_plain", "n", "s", advisory_plain_run, true },\n'
        '    { "advisory_trailing_comment", "n", "s", advisory_comment_run, true'
        '  /* advisory=true: requires __int128 (MSVC skips with code 77) */ },\n'
        '};\n'
        'static constexpr int NUM_MODULES = sizeof(ALL_MODULES) / sizeof(ALL_MODULES[0]);\n'
    )
    with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as f:
        f.write(fixture_src)
        fixture_path = f.name
    try:
        from pathlib import Path
        fixture_count = chk.count_advisory_modules(Path(fixture_path))
        check(fixture_count == 2,
              f"a row with a trailing inline /* ... */ comment between `true` and "
              f"`}}` is still counted (expected 2 advisory rows, got {fixture_count})")

        # fail-before anchor: the OLD (pre-fix) pattern applied to the SAME
        # fixture genuinely undercounts -- if this ever starts finding 2, the
        # fixture stopped exercising the bug this test guards against.
        import re as _re
        old_style_count = len(_re.findall(r',\s*true\s*\}', fixture_src))
        check(old_style_count == 1,
              f"fail-before: the old non-comment-stripping pattern undercounts "
              f"this fixture (found {old_style_count}, expected 1) -- anchors "
              f"why comment-stripping in count_advisory_modules() is necessary")
    finally:
        os.unlink(fixture_path)

    print("\n" + ("ALL PASS" if not failures else f"FAILURES: {len(failures)}"))
    return 1 if failures else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
