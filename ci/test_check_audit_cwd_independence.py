#!/usr/bin/env python3
"""test_check_audit_cwd_independence.py

Self-test (proof-it-blocks) for ci/check_audit_cwd_independence.py.

Uses synthetic ALL_MODULES source snippets and synthetic console-output
fixtures shaped exactly like real unified_audit_runner output -- no real
build required, so this stays in the FAST gate tier (run_fast_gates.sh)
even though the gate it tests only makes sense against a real binary.

The key case (test_ct_blinding_nonce_class_regression_detected) reproduces
the EXACT shape of Codex's round-5 finding: a module whose overall status
is PASS in both the repo-root and /tmp runs (so a status-only comparison,
which is what a naive dual-CWD check might do, sees no difference), but
whose /tmp run silently lost one internal check (a source-scan sub-check
that printed "[SKIP] ... not found" and returned early instead of calling
CHECK()). test_naive_status_only_comparison_would_false_pass proves that a
status-only comparator -- the "old logic" this gate replaces -- genuinely
misses this class, anchoring why the extra skip-marker/check-count
comparisons in compare_runs() are necessary, not incidental.

Round 8 additions: test_returncode_1_never_an_acceptable_baseline and
test_summary_all_passed_false_is_hard_error_even_at_returncode_0 anchor
Codex's round-8 finding that returncode==1 (or a report whose own
summary.all_passed is not true) must never be usable as a CWD-consistency
baseline. test_90_of_100_returncode_0_no_longer_false_passes is Codex's
exact round-8 reproducer: a 90-of-100 shared subset at returncode==0 sat
exactly AT the OLD 90%-completeness-floor's own boundary and used to be
silently accepted -- the round-8 exact-equality check (against the
compiled binary's own ``--list-modules`` output, not a fuzzy percentage of
the source-declared count) now rejects it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_audit_cwd_independence as gate  # noqa: E402


def _fixture_source(rows: list[dict]) -> str:
    lines = ["// synthetic fixture\nAuditModule ALL_MODULES[] = {\n"]
    for r in rows:
        adv = "true" if r["advisory"] else "false"
        lines.append(
            f'    {{ "{r["id"]}", "{r["name"]}", "{r["section"]}", '
            f'{r["symbol"]}, {adv} }},\n'
        )
    lines.append("};\n")
    return "".join(lines)


def _fixture_report(rows: list[dict], entries: list[dict]) -> dict:
    """Build a synthetic audit_report.json matching write_json_report()'s
    real schema (summary{all_passed,failed,...}, sections[].modules[], each
    {id, name, passed, advisory, return_code, time_ms}) for the given
    entries. Mirrors _fixture_stdout's ``entries`` shape so a test can build
    matching stdout+report pairs from ONE entries list -- callers that need
    a mismatch construct them from different entries lists on purpose.
    ``summary.all_passed``/``summary.failed`` mirror real write_json_report()
    semantics: only a MANDATORY (non-advisory) module's failure counts
    against ``total_fail``, matching unified_audit_runner.cpp's own
    ``total_fail`` computation."""
    advisory_by_id = {r["id"]: r["advisory"] for r in rows}
    modules = []
    for e in entries:
        status = e.get("status", "PASS")
        advisory = advisory_by_id.get(e["id"], False)
        if "report_return_code" in e:
            rc = e["report_return_code"]
            passed = (rc == 0)
        elif status == "PASS":
            rc, passed = 0, True
        elif status == "SKIP" and advisory:
            rc, passed = gate.ADVISORY_SKIP_CODE, False
        else:
            rc, passed = 1, False
        modules.append({
            "id": e["id"], "name": "fixture", "passed": passed,
            "advisory": advisory, "return_code": rc, "time_ms": 5.0,
        })
    total_fail = sum(1 for m in modules if not m["advisory"] and not m["passed"])
    return {
        "summary": {"all_passed": total_fail == 0, "failed": total_fail},
        "sections": [{"id": "test", "modules": modules}],
    }


def _build_result(rows: list[dict], entries: list[dict], label: str,
                   returncode: int = 0, report_override=None,
                   stdout_override: str | None = None,
                   authoritative_ids: set[str] | None = None):
    """Combine _fixture_stdout + _fixture_report into one build_run_result()
    call -- the common case where both are derived from the same entries.
    ``authoritative_ids`` defaults to the full ``rows`` id set (i.e. "this
    binary's --list-modules would report every declared row"); pass an
    explicit subset to exercise the round-8 exact-equality check against a
    binary that legitimately compiled out some modules."""
    stdout_text = stdout_override if stdout_override is not None else _fixture_stdout(entries, rows)
    report = report_override if report_override is not None else _fixture_report(rows, entries)
    if authoritative_ids is None:
        authoritative_ids = {r["id"] for r in rows}
    return gate.build_run_result(stdout_text, report, returncode, rows, label, authoritative_ids)


def _fixture_stdout(entries: list[dict], rows: list[dict] | None = None) -> str:
    """entries: [{id, status, checks_total, checks_pass, skip_marker}].
    The printed header uses each module's NAME field (looked up from
    `rows`, defaulting to _ROWS, by id), exactly like the real runner's
    "  [%2d/%d] %-45s " (m.name, not m.id) -- module identity is resolved
    by name, not id."""
    id_to_name = {r["id"]: r["name"] for r in (rows if rows is not None else _ROWS)}
    total = len(entries)
    out = ["================\n[Phase 1/3] selftest...\n\n"
           "[Phase 2/3] Running modules...\n\n"]
    for idx, e in enumerate(entries, start=1):
        name = id_to_name[e["id"]]
        header = f"  [{idx:2d}/{total}] {name:<45} "
        inner = ""
        if e.get("skip_marker"):
            inner += "  [SKIP] some_source.cpp not found — run from repo root\n"
        if e.get("checks_total") is not None:
            checks_pass = e.get("checks_pass", e["checks_total"])
            inner += f"[{e['id']}] {checks_pass}/{e['checks_total']} checks passed\n"
        status = e.get("status", "PASS")
        out.append(f"{header}{inner}{status}  (5 ms)\n")
    return "".join(out)


_ROWS = [
    {"id": "mod_clean", "name": "clean module", "section": "core",
     "symbol": "test_mod_clean_run", "advisory": False},
    {"id": "ct_blinding_nonce", "name": "CT nonce path uses generator_mul_blinded",
     "section": "ct_analysis", "symbol": "test_regression_ct_blinding_nonce_path_run",
     "advisory": False},
    {"id": "mod_advisory", "name": "optional external tool", "section": "formal",
     "symbol": "mod_advisory_run", "advisory": True},
]


def _parse_rows():
    rows = gate.parse_all_modules_ordered(_fixture_source(_ROWS))
    assert len(rows) == 3, f"expected 3 fixture rows, got {len(rows)}"
    assert [r["id"] for r in rows] == ["mod_clean", "ct_blinding_nonce", "mod_advisory"]
    return rows


def test_duplicate_name_rows_resolved_in_declaration_order():
    """Regression (2026-07-18, live real-binary run): two real ALL_MODULES[]
    rows -- "exploit_bip39_entropy" and "exploit_bip39_mnemonic" -- share the
    EXACT SAME description string. A name->id lookup that assumes names are
    unique collapses both real headers onto whichever id it saw first (a
    dict.setdefault), silently losing the second module entirely and
    reporting a false "duplicate" for the first. Each successive occurrence
    of a shared name must resolve to the NEXT id sharing it, in declaration
    order -- correct because same-named rows still execute in their fixed
    relative declaration order."""
    dup_rows = [
        {"id": "first_of_pair", "name": "BIP-39 Mnemonic Security Properties",
         "section": "exploit_poc", "symbol": "test_first_of_pair_run", "advisory": False},
        {"id": "second_of_pair", "name": "BIP-39 Mnemonic Security Properties",
         "section": "exploit_poc", "symbol": "test_second_of_pair_run", "advisory": False},
    ]
    rows = gate.parse_all_modules_ordered(_fixture_source(dup_rows))
    assert [r["id"] for r in rows] == ["first_of_pair", "second_of_pair"]

    stdout_text = (
        "[Phase 2/3] Running modules...\n\n"
        "  [ 1/2] BIP-39 Mnemonic Security Properties           PASS  (5 ms)\n"
        "  [ 2/2] BIP-39 Mnemonic Security Properties           PASS  (5 ms)\n"
    )
    pairs = gate.split_module_chunks_by_name(stdout_text, rows)
    assert [mid for mid, _ in pairs] == ["first_of_pair", "second_of_pair"], pairs
    print("  OK: two rows sharing the same name resolve to distinct ids in "
          "declaration order, not a collapsed/duplicated single id")


def test_structural_parser_preserves_execution_order():
    rows = _parse_rows()
    assert rows[0]["advisory"] is False
    assert rows[1]["symbol"] == "test_regression_ct_blinding_nonce_path_run"
    assert rows[2]["advisory"] is True
    print("  OK: parse_all_modules_ordered preserves declaration/execution order")


def test_clean_pair_dual_cwd_consistent_passes():
    rows = _parse_rows()
    common = [
        {"id": "mod_clean", "status": "PASS", "checks_total": 4},
        {"id": "ct_blinding_nonce", "status": "PASS", "checks_total": 6},
        {"id": "mod_advisory", "status": "SKIP", "checks_total": 0},
    ]
    root = _build_result(rows, common, "root")
    tmp = _build_result(rows, common, "tmp")
    assert root.ok, root.error
    assert tmp.ok, tmp.error
    violations = gate.compare_runs(root, tmp)
    assert violations == [], f"expected no violations, got {violations}"
    print("  OK: identical root/tmp runs produce zero violations")


def test_ct_blinding_nonce_class_regression_detected():
    rows = _parse_rows()
    root_entries = [
        {"id": "mod_clean", "status": "PASS", "checks_total": 4},
        {"id": "ct_blinding_nonce", "status": "PASS", "checks_total": 6,
         "skip_marker": False},
        {"id": "mod_advisory", "status": "SKIP", "checks_total": 0},
    ]
    # Exact shape of the round-5 finding: ct_blinding_nonce's internal
    # source-scan sub-check silently returns after printing "[SKIP] ... not
    # found" instead of calling CHECK() -- one fewer check executes, but the
    # module's overall g_fail stays 0 so it STILL reports status=PASS.
    tmp_entries = [
        {"id": "mod_clean", "status": "PASS", "checks_total": 4},
        {"id": "ct_blinding_nonce", "status": "PASS", "checks_total": 5,
         "skip_marker": True},
        {"id": "mod_advisory", "status": "SKIP", "checks_total": 0},
    ]
    root = _build_result(rows, root_entries, "root")
    tmp = _build_result(rows, tmp_entries, "tmp")
    assert root.ok, root.error
    assert tmp.ok, tmp.error

    violations = gate.compare_runs(root, tmp)
    kinds = {(v["module"], v["kind"]) for v in violations}
    assert ("ct_blinding_nonce", "skip_marker_from_tmp_only") in kinds, violations
    assert ("ct_blinding_nonce", "fewer_checks_from_tmp") in kinds, violations
    # The clean module and the advisory module must NOT be flagged.
    assert not any(v["module"] == "mod_clean" for v in violations), violations
    assert not any(v["module"] == "mod_advisory" for v in violations), violations
    print("  OK: the exact round-5 ct_blinding_nonce regression shape is detected "
          f"({len(violations)} violation(s))")


def test_naive_status_only_comparison_would_false_pass():
    """Anchors WHY compare_runs checks skip-marker/check-count and not just
    status: on the identical round-5-shaped fixture above, a comparator that
    only looks at PASS/FAIL/SKIP/WARN status sees no difference at all."""
    rows = _parse_rows()
    root_entries = [
        {"id": "ct_blinding_nonce", "status": "PASS", "checks_total": 6, "skip_marker": False},
    ]
    tmp_entries = [
        {"id": "ct_blinding_nonce", "status": "PASS", "checks_total": 5, "skip_marker": True},
    ]
    narrow_rows = [r for r in rows if r["id"] == "ct_blinding_nonce"]
    root = _build_result(narrow_rows, root_entries, "root")
    tmp = _build_result(narrow_rows, tmp_entries, "tmp")
    assert root.ok, root.error
    assert tmp.ok, tmp.error

    def naive_status_only_compare(root_r, tmp_r):
        return [
            m for m, r in root_r.modules.items()
            if not r["advisory"] and tmp_r.modules.get(m, {}).get("status") != r["status"]
        ]

    naive_violations = naive_status_only_compare(root, tmp)
    assert naive_violations == [], (
        "fail-before: a status-only comparator must genuinely miss this case "
        f"for the test to anchor anything real, but it found {naive_violations}"
    )

    real_violations = gate.compare_runs(root, tmp)
    assert len(real_violations) >= 1, "pass-after: compare_runs must catch it"
    print("  OK: status-only comparison false-passes the round-5 shape; "
          "compare_runs() (skip-marker + check-count aware) catches it")


def test_disappeared_mandatory_module_detected():
    root = gate.RunResult("root")
    root.ok = True
    root.modules = {
        "mod_a": {"advisory": False, "status": "PASS", "has_skip_marker": False,
                  "check_total": 3, "report_return_code": 0},
        "mod_b": {"advisory": False, "status": "PASS", "has_skip_marker": False,
                  "check_total": 2, "report_return_code": 0},
    }
    tmp = gate.RunResult("tmp")
    tmp.ok = True
    tmp.modules = {
        "mod_a": {"advisory": False, "status": "PASS", "has_skip_marker": False,
                  "check_total": 3, "report_return_code": 0},
    }
    violations = gate.compare_runs(root, tmp)
    assert any(v["module"] == "mod_b" and v["kind"] == "disappeared" for v in violations), violations
    print("  OK: a mandatory module missing from the /tmp run is flagged as 'disappeared'")


_MANY_ROWS = [
    {"id": f"mod_{i}", "name": f"module number {i} description text",
     "section": "core", "symbol": f"test_mod_{i}_run", "advisory": False}
    for i in range(10)
]


def test_conditionally_compiled_subset_matches_authoritative_set():
    """Round 8: a build with an optional feature (FROST/ZK/BIP352/WALLET/...)
    disabled runs a strict SUBSET of the rows ALL_MODULES[] declares in
    source -- the exact "453 ALL_MODULES rows vs N observed" class. This is
    legitimate ONLY when the observed set EXACTLY matches the binary's own
    --list-modules output (``authoritative_ids`` here) -- there is no longer
    a fuzzy floor; the authoritative set (which itself already reflects the
    compiled-in feature flags) is the single source of truth for what
    "complete" means for THIS binary."""
    rows = gate.parse_all_modules_ordered(_fixture_source(_MANY_ROWS))
    assert len(rows) == 10
    # mod_9 is compiled out of THIS build -- --list-modules would report
    # only mod_0..mod_8, which is exactly what this run also produced.
    authoritative_ids = {f"mod_{i}" for i in range(9)}
    entries = [{"id": r["id"], "status": "PASS", "checks_total": 4} for r in rows[:9]]
    result = _build_result(rows, entries, "root", authoritative_ids=authoritative_ids)
    assert result.ok, result.error
    assert set(result.modules) == {f"mod_{i}" for i in range(9)}
    print("  OK: a subset run that exactly matches the binary's own "
          "authoritative --list-modules set is accepted, not flagged")


def test_exact_module_set_mismatch_is_hard_error():
    """Round 8, replacing the old fuzzy 90% floor: ANY deviation from the
    binary's own authoritative --list-modules set is a hard failure, not
    just a large one. A run that produced only 2 of 10 authoritative modules
    (returncode=0, each individually 'clean') must fail."""
    rows = gate.parse_all_modules_ordered(_fixture_source(_MANY_ROWS))
    authoritative_ids = {r["id"] for r in rows}  # binary expects all 10
    entries = [{"id": r["id"], "status": "PASS", "checks_total": 4} for r in rows[:2]]
    result = _build_result(rows, entries, "root", authoritative_ids=authoritative_ids)
    assert not result.ok
    assert "does not EXACTLY match" in result.error, result.error
    print("  OK: a run covering only a fraction of the authoritative module "
          "set is a hard exact-equality failure, not a silent partial pass")


def test_90_of_100_returncode_0_no_longer_false_passes():
    """Codex round-8's exact reproducer: a returncode=0, 90-of-100 shared
    subset -- exactly AT the OLD 90% floor's own boundary, which the old
    completeness-floor logic accepted outright as "not below the floor" --
    must now be REJECTED, because it is not exact equality against the
    authoritative set. Preserved as a fail-before/pass-after self-test."""
    rows_100 = [
        {"id": f"m{i}", "name": f"module {i} description text",
         "section": "core", "symbol": f"test_m{i}_run", "advisory": False}
        for i in range(100)
    ]
    rows = gate.parse_all_modules_ordered(_fixture_source(rows_100))
    assert len(rows) == 100
    authoritative_ids = {r["id"] for r in rows}  # binary expects all 100
    entries = [{"id": r["id"], "status": "PASS", "checks_total": 4} for r in rows[:90]]
    result = _build_result(rows, entries, "root", authoritative_ids=authoritative_ids,
                            returncode=0)
    assert not result.ok, (
        "fail-before/pass-after anchor: a 90-of-100 run at returncode=0 sat "
        "exactly AT the old 90% floor and used to be silently accepted -- "
        "the round-8 exact-equality check must reject it"
    )
    assert "does not EXACTLY match" in result.error, result.error
    print("  OK: a 90-of-100 returncode=0 shared-subset run — exactly at the "
          "old floor's own boundary — is now correctly rejected")


def test_returncode_1_never_an_acceptable_baseline():
    """Round 8: the round-7 fix accepted returncode in {0, 1} as normal
    completion. Codex's round-8 finding: returncode==1 means the audit
    itself is genuinely failing (some real module failed;
    summary.all_passed=false) and must NEVER be usable as a CWD-consistency
    baseline, even though the process did complete without crashing."""
    rows = _parse_rows()
    authoritative_ids = {r["id"] for r in rows}
    entries = [{"id": r["id"], "status": "PASS", "checks_total": 4} for r in rows]
    stdout_text = _fixture_stdout(entries, rows)
    report = _fixture_report(rows, entries)
    result = gate.build_run_result(stdout_text, report, 1, rows, "root", authoritative_ids)
    assert not result.ok
    assert "returncode" in result.error and "0" in result.error, result.error
    print("  OK: returncode==1 is rejected outright, never treated as an "
          "acceptable 'some module failed but run completed' baseline")


def test_summary_all_passed_false_is_hard_error_even_at_returncode_0():
    """Independent cross-check dimension: even if returncode somehow reads
    0, a report whose own summary.all_passed is not true (or summary.failed
    != 0) must also be rejected -- the two signals are checked
    independently, not as a single combined condition, matching
    unified_audit_runner.cpp's own summary schema."""
    rows = _parse_rows()
    authoritative_ids = {r["id"] for r in rows}
    entries = [{"id": r["id"], "status": "PASS", "checks_total": 4} for r in rows]
    stdout_text = _fixture_stdout(entries, rows)
    report = _fixture_report(rows, entries)
    report["summary"] = {"all_passed": False, "failed": 1}
    result = gate.build_run_result(stdout_text, report, 0, rows, "root", authoritative_ids)
    assert not result.ok
    assert "summary.all_passed" in result.error, result.error
    print("  OK: summary.all_passed=false is independently rejected even "
          "when returncode==0")


def test_stdout_report_mismatch_is_hard_error():
    """audit_report.json and the console text it was captured alongside must
    describe the SAME run. If the module set parsed from stdout does not
    match the module set in the report (a stdout-parser bug, or the two
    artifacts genuinely being from different runs), that must be a hard
    error -- never silently trust one source over the other."""
    rows = gate.parse_all_modules_ordered(_fixture_source(_MANY_ROWS))
    authoritative_ids = {r["id"] for r in rows}
    report_entries = [{"id": r["id"], "status": "PASS", "checks_total": 4} for r in rows]
    report = _fixture_report(rows, report_entries)
    # stdout only mentions a DIFFERENT subset than the report claims to cover.
    stdout_entries = [{"id": r["id"], "status": "PASS", "checks_total": 4} for r in rows[:5]]
    stdout_text = _fixture_stdout(stdout_entries, rows)
    result = gate.build_run_result(stdout_text, report, 0, rows, "root", authoritative_ids)
    assert not result.ok
    assert "does not match audit_report.json" in result.error, result.error
    print("  OK: a stdout/report module-set mismatch is a hard error")


def test_signal_killed_process_is_hard_error_not_parsed():
    """A negative return code (Python subprocess convention for 'killed by
    signal N') must be an immediate hard failure -- must NEVER attempt to
    parse whatever partial stdout a crashed process happened to produce."""
    rows = _parse_rows()
    authoritative_ids = {r["id"] for r in rows}
    result = gate.build_run_result("some partial garbage output", None, -11, rows, "root",
                                    authoritative_ids)
    assert not result.ok
    assert "crashed or was killed" in result.error, result.error
    print("  OK: a signal-killed process (negative returncode) is a hard "
          "error, never treated as a parseable partial result")


def test_missing_report_with_clean_exit_is_hard_error():
    """A returncode==0 alone is not sufficient evidence of completion --
    write_json_report() must have actually run. A missing/unparseable report
    despite a 'clean' exit code is itself suspicious and must fail closed."""
    rows = _parse_rows()
    authoritative_ids = {r["id"] for r in rows}
    result = gate.build_run_result("looks fine\nPASS  (5 ms)\n", None, 0, rows, "root",
                                    authoritative_ids)
    assert not result.ok
    assert "audit_report.json could not be obtained" in result.error, result.error
    print("  OK: returncode=0 with no obtainable audit_report.json is a hard error")


def test_shared_truncation_no_longer_false_passes():
    """THE Codex round-6 finding, reproduced directly: two runs that are
    identically (and severely) truncated -- e.g. both processed only 1 of 10
    declared modules, because of a shared crash/bug unrelated to CWD -- used
    to compare as 'consistent' (0 violations) because compare_runs() only
    ever checked root vs tmp against EACH OTHER, never against the declared
    module set. Both runs must now independently fail the completeness
    floor, so compare_runs() short-circuits on root_run_failed/tmp_run_failed
    and never reaches 'zero violations'."""
    rows = gate.parse_all_modules_ordered(_fixture_source(_MANY_ROWS))
    truncated_entries = [{"id": rows[0]["id"], "status": "PASS", "checks_total": 1}]
    root = _build_result(rows, truncated_entries, "root")
    tmp = _build_result(rows, truncated_entries, "tmp")
    assert not root.ok and not tmp.ok, "both runs must independently fail closed"

    violations = gate.compare_runs(root, tmp)
    assert violations, "must NOT report zero violations for two shared-truncated runs"
    assert violations[0]["kind"] == "root_run_failed", violations
    print("  OK: two identically truncated runs no longer false-pass as "
          "'consistent' -- compare_runs() reports root_run_failed, not PASS")


def test_internal_header_shaped_text_does_not_split_a_real_module():
    """Regression (2026-07-18, live real-binary run): a module can print its
    OWN internal sub-test progress in a shape that coincidentally matches
    the bare header pattern "^  \\[N/M\\] " (observed for real: a field-
    multiply-reduce test's own "  [0/100] ..." progress line). With header
    detection and name-matching as two separate passes, that false header
    split the real module's own trailing output into a bogus second
    "unmatched" chunk. The header regex must be anchored to a KNOWN name
    immediately after the bracket, so a look-alike line with no real name
    following it is never treated as a module boundary at all."""
    rows = _parse_rows()
    only_clean = [r for r in rows if r["id"] == "mod_clean"]
    total = 1
    name = only_clean[0]["name"]
    stdout_text = (
        "[Phase 2/3] Running modules...\n\n"
        f"  [ 1/{total}] {name:<45} \n"
        "  [0/100] ...\n"
        "[MFR-6] Multiplication commutativity\n"
        "[MFR-7] Verify acc[8] overflow for issue #226 input\n"
        "Results: 14 passed, 0 failed\n"
        "[mod_clean] 14/14 checks passed\n"
        "PASS  (5 ms)\n"
    )
    report = _fixture_report(only_clean, [{"id": "mod_clean", "status": "PASS"}])
    authoritative_ids = {"mod_clean"}
    result = gate.build_run_result(stdout_text, report, 0, only_clean, "root", authoritative_ids)
    assert result.ok, result.error
    assert set(result.modules) == {"mod_clean"}, result.modules
    assert result.modules["mod_clean"]["check_total"] == 14, result.modules
    assert result.modules["mod_clean"]["status"] == "PASS", result.modules
    print("  OK: an internal header-shaped progress line with no real module "
          "name after it does not split or corrupt the real module's chunk")


def test_advisory_module_divergence_not_flagged():
    root = gate.RunResult("root")
    root.ok = True
    root.modules = {
        "adv": {"advisory": True, "status": "PASS", "has_skip_marker": False, "check_total": 3},
    }
    tmp = gate.RunResult("tmp")
    tmp.ok = True
    tmp.modules = {
        "adv": {"advisory": True, "status": "SKIP", "has_skip_marker": True, "check_total": None},
    }
    violations = gate.compare_runs(root, tmp)
    assert violations == [], (
        f"advisory modules legitimately diverge from /tmp; must not be flagged, got {violations}"
    )
    print("  OK: advisory-module divergence from /tmp is correctly ignored")


def test_run_failure_reported_not_silently_passed():
    root = gate.RunResult("root")
    root.ok = False
    root.error = "boom"
    tmp = gate.RunResult("tmp")
    tmp.ok = True
    violations = gate.compare_runs(root, tmp)
    assert violations and violations[0]["kind"] == "root_run_failed", violations
    print("  OK: a failed run is reported as a violation, never silently treated as PASS")


TESTS = [
    test_duplicate_name_rows_resolved_in_declaration_order,
    test_structural_parser_preserves_execution_order,
    test_clean_pair_dual_cwd_consistent_passes,
    test_ct_blinding_nonce_class_regression_detected,
    test_naive_status_only_comparison_would_false_pass,
    test_disappeared_mandatory_module_detected,
    test_conditionally_compiled_subset_matches_authoritative_set,
    test_exact_module_set_mismatch_is_hard_error,
    test_90_of_100_returncode_0_no_longer_false_passes,
    test_returncode_1_never_an_acceptable_baseline,
    test_summary_all_passed_false_is_hard_error_even_at_returncode_0,
    test_stdout_report_mismatch_is_hard_error,
    test_signal_killed_process_is_hard_error_not_parsed,
    test_missing_report_with_clean_exit_is_hard_error,
    test_shared_truncation_no_longer_false_passes,
    test_internal_header_shaped_text_does_not_split_a_real_module,
    test_advisory_module_divergence_not_flagged,
    test_run_failure_reported_not_silently_passed,
]


def main() -> int:
    failures = 0
    for t in TESTS:
        print(f"[{t.__name__}]")
        try:
            t()
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL: {exc}")
    print()
    if failures:
        print(f"RESULT: FAIL ({failures}/{len(TESTS)} failed)")
        return 1
    print(f"RESULT: PASS ({len(TESTS)}/{len(TESTS)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
