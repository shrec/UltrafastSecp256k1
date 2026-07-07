#!/usr/bin/env python3
"""
Self-test for ci/mutation_kill_rate.py reporting path — issue #313 regression.

issue #313: the weekly mutation workflow opened an issue with `None` metrics
even though the report artifact had valid schema fields, and a baseline
`unified_audit` timeout (an infrastructure failure with no kill rate measured)
was reported as a normal kill-rate regression.

We do not trust that the reporting path is correct — we PROVE it:
  1. classify_result() distinguishes baseline-infra failure from kill-rate
     regression, insufficient sample, and pass.
  2. render_issue_body() never emits the literal "None" and references real
     schema keys, for both a baseline-timeout report and a regression report.
  3. A baseline-failure report is labelled as infrastructure, not regression.
  4. Every key render_issue_body() reads exists on a freshly-constructed
     KillReport (so the schema and the renderer cannot drift apart).

Run:  python3 tests/ci/test_mutation_reporting.py
"""
import dataclasses
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load():
    spec = importlib.util.spec_from_file_location(
        "mkr", os.path.join(ROOT, "ci", "mutation_kill_rate.py"))
    m = importlib.util.module_from_spec(spec)
    # Register before exec: dataclasses + `from __future__ import annotations`
    # resolve string annotations via sys.modules[cls.__module__].
    sys.modules["mkr"] = m
    spec.loader.exec_module(m)
    return m


def _baseline_timeout_report(m):
    """A report exactly as produced when the baseline unified_audit times out."""
    r = m.KillReport(
        timestamp="2026-06-29T06:00:00Z",
        threshold_pct=75.0,
        targets=["src/cpu/src/field.cpp"],
        test_commands=["ctest --test-dir build_rel -R unified_audit -j1"],
    )
    r.baseline_build_ok = True
    r.baseline_test_ok = False
    r.baseline_note = "baseline tests failed: test timeout: ctest --test-dir build_rel -R unified_audit"
    r.passed = False
    r.failure_class = "baseline_infrastructure"
    # No mutants tested — fail-closed.
    return dataclasses.asdict(r)


def _regression_report(m):
    r = m.KillReport(
        timestamp="2026-06-29T06:00:00Z",
        total=40, killed=20, survived=20, testable=40,
        kill_rate_pct=50.0, threshold_pct=75.0,
        passed=False, failure_class="kill_rate_regression",
        pass_reason="kill rate 50.0% < 75.0%",
    )
    return dataclasses.asdict(r)


def main() -> int:
    m = _load()
    fails = []

    base = _baseline_timeout_report(m)
    regr = _regression_report(m)

    # 1. Classification.
    if m.classify_result(base) != "baseline_infrastructure":
        fails.append("baseline-timeout report must classify as baseline_infrastructure")
    if m.classify_result(regr) != "kill_rate_regression":
        fails.append("regression report must classify as kill_rate_regression")

    # classify_result must derive baseline_infrastructure even without the
    # explicit field (older reports).
    legacy = dict(base)
    legacy.pop("failure_class", None)
    if m.classify_result(legacy) != "baseline_infrastructure":
        fails.append("legacy report (no failure_class) with failed baseline must derive baseline_infrastructure")

    pass_report = dict(regr)
    pass_report.update(passed=True, failure_class="pass", kill_rate_pct=90.0)
    if m.classify_result(pass_report) != "pass":
        fails.append("passing report must classify as pass")

    # 2. Rendering: no literal "None" in either body.
    base_body = m.render_issue_body(base)
    regr_body = m.render_issue_body(regr)
    if "None" in base_body:
        fails.append(f"baseline body must not contain 'None':\n{base_body}")
    if "None" in regr_body:
        fails.append(f"regression body must not contain 'None':\n{regr_body}")

    # 3. Baseline body labelled infrastructure, NOT regression.
    low = base_body.lower()
    if "infrastructure" not in low:
        fails.append("baseline body must name it an infrastructure failure")
    if "kill-rate regression" not in low and "not a mutation" not in low:
        fails.append("baseline body must state it is not a kill-rate regression")
    if "failure_class: baseline_infrastructure" not in base_body:
        fails.append("baseline body must surface failure_class")

    # 4. Regression body carries the real metric keys.
    for token in ("kill_rate_pct:", "threshold_pct:", "total:", "survived:"):
        if token not in regr_body:
            fails.append(f"regression body missing real schema key '{token}'")

    # 5. Renderer and schema cannot drift: every field the renderer reads must
    #    exist on a fresh KillReport.
    schema_keys = set(dataclasses.asdict(m.KillReport()).keys())
    for key in ("failure_class", "kill_rate_pct", "threshold_pct", "total",
                "testable", "killed", "survived", "pass_reason",
                "baseline_build_ok", "baseline_test_ok", "baseline_note"):
        if key not in schema_keys:
            fails.append(f"renderer reads '{key}' but it is not a KillReport field")

    # 6. Documented compatibility aliases map to real fields.
    for alias, real in m.SCHEMA_ALIASES.items():
        if real not in schema_keys:
            fails.append(f"alias {alias} -> {real}, but {real} is not a KillReport field")

    if fails:
        print("FAIL: mutation reporting self-test")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("PASS: mutation reporting self-test (issue #313)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
