#!/usr/bin/env python3
"""check_audit_cwd_independence.py

Issue #335 acceptance repair (round 6, hardened round 7, hardened again
round 8) — the exhaustive dual-CWD gate Codex asked for after round 5's
re-review.

Round 7 addendum: Codex's round-6 review found that ``run_once()`` discarded
the subprocess return code and trusted whatever partial stdout it captured,
so a bug that crashes/truncates BOTH the repo-root and the ``/tmp`` run
identically would produce two internally-"consistent" (because equally
broken) ``RunResult``s and ``compare_runs()`` would report a false PASS —
Codex's literal example: two identical one-module ``RunResult``s comparing
clean. Fixed by binding each run to its own ``audit_report.json`` (written
by ``write_json_report()`` only after every ``ALL_MODULES[]`` entry has
actually executed) as an INDEPENDENT completeness signal, not just an extra
thing to parse.

Round 8 addendum: Codex's round-7 review found that fix still had two
holes. (1) ``build_run_result()`` accepted returncode ``0`` OR ``1`` as a
normal completion, so a run whose audit itself FAILED (some module really
did fail, ``summary.all_passed=false``) could still be used as a "clean"
baseline for comparing CWD consistency — this gate's job is to compare
CWD-*consistency*, not to also double as the audit pass/fail gate, so a
failed run must never be an acceptable baseline. (2) the completeness check
was a fuzzy ``>= 90%`` floor against the SOURCE-declared row count, which
(a) is not exact equality and (b) compares against the wrong reference:
``ALL_MODULES[]`` rows can be wrapped in ``#if SECP256K1_HAS_*`` blocks a
Python source-text regex cannot evaluate, so "90% of the source-declared
count" is neither a completeness guarantee nor a correctness guarantee — a
90-of-100 run and a 100-of-100 run both clear a 90% floor identically.
Fixed by asking the compiled binary itself for its exact active module set
(``unified_audit_runner --list-modules``, added round 8 — zero I/O, exits
before any CWD-dependent code runs, so it is CWD-invariant by construction)
and requiring EXACT set equality against it, plus requiring returncode
``== 0`` exactly and ``audit_report.json``'s own
``summary.all_passed == true`` / ``summary.failed == 0`` as two independent
signals of "this run is a valid pass/fail-clean baseline" — see
``build_run_result()`` and ``get_authoritative_module_ids()``.

Background
----------
Round 5 fixed CWD-relative-only source resolution in a handful of audit
modules by introducing ``audit_read_source_file()`` (compile-time
``UFSECP_SOURCE_ROOT``-based, see ``audit/audit_check.hpp``) and added
``audit/test_regression_audit_source_root_cwd_independence.cpp`` as an
in-process meta-regression. Codex's round-5 re-review reproduced a FRESH
``unified_audit_runner`` build, ran it once from the repo root and once from
an unrelated ``/tmp`` directory, and found that a module NOT covered by that
meta-regression's hand-maintained 4-probe table (``ct_blinding_nonce``)
still silently lost its only source-reading sub-check from ``/tmp`` while
still returning an overall PASS (8/8). The in-process meta-regression can
only ever probe the handful of modules it knows to call directly — it
cannot, by construction, sweep the other ~450 modules in ``ALL_MODULES[]``.

This script is the sweep. It runs the REAL ``unified_audit_runner`` binary
twice — once with CWD = repository root, once with CWD = an unrelated
directory outside the repo tree — and structurally compares, for every
MANDATORY (``advisory=false``) module, whether its outcome is identical
between the two runs. A mandatory module is a violation if, between the two
runs, it:

  * disappears (the module count / module list differ — the binary itself
    changed shape, which should never happen from a CWD change alone), or
  * changes PASS/FAIL/SKIP/WARN status, or
  * gains a silent-skip marker ("not found", "[SKIP]", etc. — the exact
    tell-tale text the old CWD-only resolvers printed instead of calling
    CHECK()) that was absent from the repo-root run, or
  * reports fewer executed checks than the repo-root run while still
    reporting the SAME pass status (the "vacuous 0-check PASS" class —
    this is precisely what let ``ct_blinding_nonce`` slip through round 5).

Advisory (``advisory=true``) modules are intentionally excluded: several of
them legitimately advisory-skip from an unrelated CWD because they depend on
genuinely optional, path-relative external tooling (Cryptol, SAW, fuzz
corpora) — that is correct, documented behavior, not the bug class this gate
guards against.

This gate needs a real compiled binary and is therefore NOT part of the fast
(~30s, no-build) gate tier in ``run_fast_gates.sh``. Wire it into the build+
test tier (``ci_local.sh --full``) instead. Its pure comparison logic
(``compare_runs``) is unit-tested without any real build by
``ci/test_check_audit_cwd_independence.py``, which IS fast and IS wired into
``run_fast_gates.sh`` (matching the ``check_advisory_skip_ceiling.py`` /
``test_check_advisory_skip_ceiling.py`` pattern already used in this repo).

Exit status
-----------
  0   -- every mandatory module is dual-CWD-consistent
  1   -- one or more mandatory modules diverged between repo-root and /tmp
  77  -- could not obtain a usable binary (advisory skip: no compiler / no
         build infrastructure available in this environment)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

LIB_ROOT = Path(__file__).resolve().parents[1]
RUNNER_SRC = LIB_ROOT / "audit" / "unified_audit_runner.cpp"

ADVISORY_SKIP_CODE = 77

# ---------------------------------------------------------------------------
# Structural ALL_MODULES[] parser (self-contained: deliberately NOT imported
# from check_exploit_wiring.py, which is a separately hardened P0 gate this
# script must not risk perturbing). Same proven row shape/escaping rules.
# ---------------------------------------------------------------------------
_STRING_FIELD = r'"(?:[^"\\]|\\.)*"'
_ALL_MODULES_ROW_RE = re.compile(
    r'\{\s*(' + _STRING_FIELD + r')\s*,\s*(' + _STRING_FIELD + r')\s*,\s*(' +
    _STRING_FIELD + r')\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*(true|false)\s*\}'
)
_ALL_MODULES_ARRAY_START_RE = re.compile(r'AuditModule\s+ALL_MODULES\s*\[\s*\]\s*=\s*\{')
_ALL_MODULES_ARRAY_END_RE = re.compile(r'^\s*\}\s*;', re.MULTILINE)

_COMMENT_RE = re.compile(r'//[^\n]*|/\*.*?\*/', re.DOTALL)


def _unquote(lit: str) -> str:
    """Strip the surrounding quotes and undo simple backslash escapes."""
    inner = lit[1:-1]
    return inner.replace(r'\"', '"').replace(r'\\', '\\')


def parse_all_modules_ordered(runner_text: str) -> list[dict]:
    """Structurally parse ``ALL_MODULES[]`` and return its rows IN ARRAY
    ORDER (== the exact order ``main()``'s ``for`` loop runs them in — a
    plain sequential loop over a static array has no other possible
    execution order). Each row: ``{id, name, section, symbol, advisory}``.

    Returns an empty list if the array cannot be located (caller must treat
    that as a hard error, not "0 modules").
    """
    stripped = _COMMENT_RE.sub(lambda m: "\n" if "\n" in m.group(0) else "", runner_text)
    start_m = _ALL_MODULES_ARRAY_START_RE.search(stripped)
    if not start_m:
        return []
    body_start = start_m.end()
    end_m = _ALL_MODULES_ARRAY_END_RE.search(stripped, body_start)
    body_end = end_m.start() if end_m else len(stripped)
    body = stripped[body_start:body_end]

    rows = []
    for m in _ALL_MODULES_ROW_RE.finditer(body):
        rows.append({
            "id": _unquote(m.group(1)),
            "name": _unquote(m.group(2)),
            "section": _unquote(m.group(3)),
            "symbol": m.group(4),
            "advisory": (m.group(5) == "true"),
        })
    return rows


# ---------------------------------------------------------------------------
# stdout splitting: each module's console header is printed by main() as
# "  [%2d/%d] %-45s " immediately before that module's own run() output, and
# its status trailer ("PASS  (12 ms)\n" etc.) immediately after. Headers are
# always the first thing on their line, in strict execution order.
#
# Header detection and module-identity resolution are done as ONE combined
# regex match (header shape immediately followed by a KNOWN ALL_MODULES[]
# name), not two separate passes. A real run surfaced why this matters: at
# least one module prints its OWN internal sub-test progress in a shape that
# coincidentally matches a bare "^  \[N/M\] " header pattern (e.g. a
# field-multiply-reduce test's own "  [0/100] ..." progress line) -- with
# header-detection and name-matching as separate passes, that false header
# split a real module's output into two bogus chunks. Anchoring the header
# regex to be IMMEDIATELY followed by one of the 453 known, long, distinctive
# ALL_MODULES[] description strings makes a coincidental false match
# vanishingly unlikely: internal progress text is never going to happen to
# be followed by e.g. "CT nonce path uses generator_mul_blinded".
#
# Module identity is resolved this way (name-anchored), not by zipping
# against a fixed position/count, because unified_audit_runner.cpp
# conditionally compiles several modules out via `#if SECP256K1_HAS_{FROST,
# ZK,ECIES,BIP352,ADAPTOR,WALLET}` -- a build with any of those disabled runs
# a SUBSET of the rows ALL_MODULES[] declares in source, in the same relative
# order but not the same count (this is exactly the "453 ALL_MODULES rows vs
# 452 observed CPU runtime" class of discrepancy: nothing is missing, one
# module is compiled out by a feature flag). Position-based zipping would
# silently misalign every entry after the first excluded one; name-anchored
# matching does not.
# ---------------------------------------------------------------------------
_STATUS_RE = re.compile(r'(PASS|FAIL|SKIP|WARN)\s+\(([\d.]+) ms\)')
_CHECKS_RE = re.compile(r'(\d+)/(\d+)\s+checks passed')
_SKIP_MARKERS = ("not found", "not readable", "skipped", "[SKIP]", "source tree absent")


def split_module_chunks_by_name(stdout_text: str, ordered_modules: list[dict]):
    """Return a list of ``(module_id, chunk_text)`` pairs in execution
    order, found via a single header-shape+known-name-anchored regex pass
    (see module docstring above for why this must be one combined match,
    not header-detection followed by separate name-matching).

    Names are NOT guaranteed unique: a real ALL_MODULES[] row pair
    ("exploit_bip39_entropy" / "exploit_bip39_mnemonic") shares the exact
    same description string. When a name maps to more than one id, each
    successive match of that name in the stdout stream is assigned to the
    NEXT id sharing it, in declaration order -- correct because same-named
    rows still execute in their fixed relative declaration order (the outer
    loop is a plain sequential scan), it is only the direct name->id lookup
    that is ambiguous, not the execution order itself.
    """
    name_to_ids: dict[str, list[str]] = {}
    for m in ordered_modules:
        name_to_ids.setdefault(m["name"], []).append(m["id"])
    if not name_to_ids:
        return []
    # Longest-first so one name being a literal prefix of another can never
    # cause a short/wrong match.
    names_sorted = sorted(name_to_ids, key=len, reverse=True)
    header_re = re.compile(
        r'(?m)^  \[\s*\d+/\d+\] (' + '|'.join(re.escape(n) for n in names_sorted) + r')'
    )

    headers = list(header_re.finditer(stdout_text))
    if not headers:
        return []

    consumed = {name: 0 for name in name_to_ids}
    pairs = []
    for i, h in enumerate(headers):
        chunk_end = headers[i + 1].start() if i + 1 < len(headers) else len(stdout_text)
        chunk = stdout_text[h.start():chunk_end]
        name = h.group(1)
        ids_for_name = name_to_ids[name]
        idx = min(consumed[name], len(ids_for_name) - 1)
        consumed[name] += 1
        pairs.append((ids_for_name[idx], chunk))
    return pairs


def has_skip_marker(chunk: str) -> bool:
    return any(marker in chunk for marker in _SKIP_MARKERS)


def extract_status(chunk: str) -> str | None:
    # main() prints the PASS/FAIL/SKIP/WARN trailer as the LAST thing in a
    # module's chunk. Take the last match, not the first, in case a module's
    # own internal output happens to contain similar-looking text earlier.
    matches = list(_STATUS_RE.finditer(chunk))
    return matches[-1].group(1) if matches else None


def extract_check_total(chunk: str) -> int | None:
    matches = list(_CHECKS_RE.finditer(chunk))
    return int(matches[-1].group(2)) if matches else None


# ---------------------------------------------------------------------------
# Run + compare
# ---------------------------------------------------------------------------

class RunResult:
    def __init__(self, label: str):
        self.label = label
        self.ok = False
        self.error: str | None = None
        # module_id -> {advisory, status, has_skip_marker, check_total,
        #               report_passed, report_return_code}
        self.modules: dict[str, dict] = {}
        self.header_count = 0
        self.returncode: int | None = None
        self.report: dict | None = None


# Codex round-6 rejection: run_once() discarded the subprocess return code
# entirely and trusted whatever partial stdout it captured, so a shared
# crash/truncation affecting BOTH the repo-root and /tmp runs identically
# would produce two "consistent" (because both are equally broken)
# RunResults and compare_runs() would report a false PASS. A completeness
# check on ITS OWN isn't enough either -- it must be checked against an
# INDEPENDENT source of truth, not just stdout parsing self-consistency.
# audit_report.json (written by write_json_report() only after every
# ALL_MODULES[] entry has actually run -- see unified_audit_runner.cpp
# Phase 3) is that independent source: its mere EXISTENCE proves the run
# reached completion, and its module list is authoritative for identity/
# pass-fail, cross-checked against (not simply trusted over) the stdout
# parse.
#
# Codex round-7 rejection (this gate's round-7 fix is now known-wrong,
# fixed again round 8): the round-7 fix required returncode in {0, 1} and a
# module count >= 90% of the SOURCE-declared row count. Both were too loose:
#   * returncode==1 means unified_audit_runner's own audit genuinely FAILED
#     (some module really did fail: summary.all_passed=false). This gate's
#     job is CWD-*consistency*, not also re-deciding pass/fail -- a failed
#     run must never be usable as a "clean" baseline for that comparison.
#   * a >=90% floor against the SOURCE-declared count is neither exact nor
#     correctly anchored: ALL_MODULES[] rows can be wrapped in
#     `#if SECP256K1_HAS_*` blocks a Python regex cannot evaluate, so the
#     source-declared count is not even the right denominator, and "90%" of
#     it is not equality -- a 90-of-100 run (exactly at the floor) and a
#     100-of-100 run both clear it identically. Round 8 fixes both:
#
# What counts as "the run genuinely didn't complete / isn't a valid
# baseline", checked in order:
#   1. returncode != 0 -- unified_audit_runner's main() ends with
#      `return total_fail > 0 ? 1 : 0`; anything other than exactly 0 means
#      either the run crashed/was signal-killed (negative returncode) or
#      the audit itself is failing (returncode==1), and this gate never
#      treats either as an acceptable CWD-consistency baseline.
#   2. audit_report.json missing -- write_json_report() runs unconditionally
#      in Phase 3, after every module has executed; if it's absent despite
#      a 0 exit, the process did not reach Phase 3, full stop.
#   3. audit_report.json's own summary.all_passed is not exactly true, or
#      summary.failed is not exactly 0 -- an INDEPENDENT cross-check of (1)
#      against the report's own self-reported verdict, not just trusting
#      the process exit code alone.
#   4. audit_report.json's module-id set does not EXACTLY equal the
#      authoritative set the compiled binary itself reports via
#      `--list-modules` (see get_authoritative_module_ids()) -- catches a
#      run that exits "cleanly" but only because a section filter, early
#      return, or shared bug processed an arbitrary subset rather than the
#      real suite (Codex's literal "two identical one-module RunResults"
#      example), with NO fuzzy floor: any missing or extra id is a hard
#      failure, because the authoritative set already reflects this
#      binary's own compiled-in feature flags exactly.
#   5. the module id set parsed from stdout does not match the module id
#      set in the JSON report -- catches a stdout-parsing bug OR genuinely
#      mismatched inputs, rather than silently trusting one source alone.


def get_authoritative_module_ids(binary: Path, cwd: Path) -> set[str] | None:
    """Ask the compiled binary itself (round 8: `--list-modules`, added to
    unified_audit_runner.cpp) for its EXACT active module-id set -- the set
    ALL_MODULES[] resolves to for THIS build's `#if SECP256K1_HAS_*` flags,
    which a Python source-text regex cannot evaluate. `--list-modules` does
    zero file I/O and exits before any CWD-dependent code path runs, so it
    is CWD-invariant by construction; ``cwd`` is still threaded through and
    the two call sites in main() are cross-checked against each other as a
    belt-and-suspenders proof that invariant actually holds on this binary,
    rather than merely being assumed.

    Returns None if the binary doesn't support the flag (stale, pre-round-8
    binary), crashes, times out, or prints nothing parseable -- callers must
    treat that as a hard error requiring a rebuild, never as "0 modules"."""
    try:
        proc = subprocess.run(
            [str(binary), "--list-modules"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    ids = {line.strip() for line in proc.stdout.splitlines() if line.strip()}
    return ids or None


def build_run_result(stdout_text: str, report: dict | None, returncode: int | None,
                      ordered_modules: list[dict], label: str,
                      authoritative_ids: set[str] | None,
                      source_desc: str = "source") -> RunResult:
    """Pure: combine one run's captured console text, its parsed
    audit_report.json (or None if it could not be obtained), its process
    return code, and the binary's own authoritative module-id set (from
    `--list-modules`, or None to skip that specific check -- used by unit
    tests exercising other dimensions) into a RunResult. No filesystem or
    process access of its own -- this is what the self-test exercises
    directly with synthetic fixtures."""
    result = RunResult(label)
    result.returncode = returncode

    if returncode != 0:
        result.error = (
            f"{label} run exited with returncode={returncode!r} -- this "
            f"gate requires the underlying audit to be fully green "
            f"(returncode exactly 0) before it can be used as a baseline "
            f"for comparing CWD consistency. returncode==1 means "
            f"unified_audit_runner's own audit is failing (some module "
            f"really did fail; summary.all_passed=false) -- fix the "
            f"underlying failing module(s) first, or run this gate against "
            f"a build where the mandatory audit suite is fully green. A "
            f"negative returncode means the process crashed or was killed "
            f"by a signal (e.g. SIGSEGV/SIGABRT) before finishing."
        )
        return result

    if report is None:
        result.error = (
            f"{label} run exited with returncode={returncode} but its "
            f"audit_report.json could not be obtained -- write_json_report() "
            f"only runs after every module in ALL_MODULES[] has finished "
            f"(Phase 3), so a missing/unparseable report means the run did "
            f"not actually complete despite the exit code"
        )
        return result
    result.report = report

    summary = report.get("summary", {})
    if summary.get("all_passed") is not True or summary.get("failed") != 0:
        result.error = (
            f"{label} run: returncode==0 but audit_report.json's own "
            f"summary.all_passed={summary.get('all_passed')!r} / "
            f"summary.failed={summary.get('failed')!r} -- both must be "
            f"exactly true / 0 for this run to be a valid CWD-independence "
            f"baseline (independent cross-check of the process exit code "
            f"against the report's own self-reported verdict)"
        )
        return result

    report_modules: dict[str, dict] = {}
    for section in report.get("sections", []):
        for m in section.get("modules", []):
            report_modules[m["id"]] = {
                "advisory": bool(m["advisory"]),
                "report_passed": bool(m["passed"]),
                "report_return_code": m["return_code"],
            }

    report_ids = set(report_modules)
    if authoritative_ids is not None and report_ids != authoritative_ids:
        missing = sorted(authoritative_ids - report_ids)
        extra = sorted(report_ids - authoritative_ids)
        result.error = (
            f"{label} run's audit_report.json module-id set does not "
            f"EXACTLY match this binary's own --list-modules output (the "
            f"authoritative active-module set for THIS compiled binary, "
            f"{len(authoritative_ids)} id(s)) -- "
            f"missing={missing[:8]}{'...' if len(missing) > 8 else ''} "
            f"({len(missing)} total), "
            f"extra={extra[:8]}{'...' if len(extra) > 8 else ''} "
            f"({len(extra)} total) vs {len(report_ids)} observed. Every "
            f"omission must already be reflected in --list-modules (which "
            f"itself resolves this binary's #if SECP256K1_HAS_* feature "
            f"flags) -- a completeness floor or a blanket "
            f"'conditionally_excluded' label is not an acceptable "
            f"substitute for exact equality (source declares "
            f"{len(ordered_modules)} row(s) in {source_desc}, which may "
            f"legitimately exceed the authoritative runtime set -- that gap "
            f"is informational only and is NOT what is being checked here)."
        )
        return result

    pairs = split_module_chunks_by_name(stdout_text, ordered_modules)
    stdout_ids = {mid for mid, _ in pairs}
    report_ids = set(report_modules)
    if stdout_ids != report_ids:
        only_stdout = sorted(stdout_ids - report_ids)
        only_report = sorted(report_ids - stdout_ids)
        result.error = (
            f"{label} run: module set parsed from console output does not "
            f"match audit_report.json's module set -- "
            f"only_in_stdout={only_stdout[:5]}{'...' if len(only_stdout) > 5 else ''}, "
            f"only_in_report={only_report[:5]}{'...' if len(only_report) > 5 else ''} "
            f"(stdout parser bug, or the console output and the JSON report "
            f"are not actually from the same run)"
        )
        return result

    stdout_by_id: dict[str, str] = {}
    for mid, chunk in pairs:
        stdout_by_id[mid] = chunk

    for mod_id, rm in report_modules.items():
        chunk = stdout_by_id.get(mod_id, "")
        result.modules[mod_id] = {
            "advisory": rm["advisory"],
            "status": extract_status(chunk),
            "has_skip_marker": has_skip_marker(chunk),
            "check_total": extract_check_total(chunk),
            "report_passed": rm["report_passed"],
            "report_return_code": rm["report_return_code"],
        }
    result.header_count = len(pairs)
    result.ok = True
    return result


def run_once(binary: Path, cwd: Path, ordered_modules: list[dict],
             authoritative_ids: set[str] | None, label: str) -> RunResult:
    report_dir = tempfile.mkdtemp(prefix=f"ufsecp_cwd_gate_report_{label}_")
    try:
        try:
            proc = subprocess.run(
                [str(binary), "--report-dir", report_dir],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=900,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            result = RunResult(label)
            result.error = f"failed to execute {binary} from {cwd}: {exc}"
            return result

        report_path = Path(report_dir) / "audit_report.json"
        report = None
        if report_path.is_file():
            try:
                report = json.loads(report_path.read_text())
            except (OSError, json.JSONDecodeError):
                report = None

        return build_run_result(proc.stdout, report, proc.returncode,
                                 ordered_modules, label, authoritative_ids,
                                 source_desc=str(RUNNER_SRC))
    finally:
        shutil.rmtree(report_dir, ignore_errors=True)


def compare_runs(root: RunResult, tmp: RunResult) -> list[dict]:
    """Pure comparison logic over two already-parsed RunResults. No
    filesystem or process access -- this is what
    ci/test_check_audit_cwd_independence.py exercises directly with
    synthetic fixtures, without needing a real build."""
    violations: list[dict] = []

    if not root.ok:
        return [{"module": None, "kind": "root_run_failed", "detail": root.error}]
    if not tmp.ok:
        return [{"module": None, "kind": "tmp_run_failed", "detail": tmp.error}]

    for mod_id, r in root.modules.items():
        if r["advisory"]:
            continue  # advisory modules may legitimately diverge from /tmp

        if mod_id not in tmp.modules:
            violations.append({
                "module": mod_id, "kind": "disappeared",
                "detail": "mandatory module present in repo-root run but "
                          "missing from /tmp run",
            })
            continue

        t = tmp.modules[mod_id]

        if r["status"] != t["status"]:
            violations.append({
                "module": mod_id, "kind": "status_changed",
                "detail": f"status={r['status']!r} at repo root but "
                          f"status={t['status']!r} from /tmp",
            })

        # Independent of stdout text parsing: audit_report.json's own
        # return_code field for this module must also agree between CWDs.
        if r["report_return_code"] != t["report_return_code"]:
            violations.append({
                "module": mod_id, "kind": "report_return_code_changed",
                "detail": f"audit_report.json return_code={r['report_return_code']!r} "
                          f"at repo root but return_code={t['report_return_code']!r} "
                          f"from /tmp (independent of console-text parsing)",
            })

        if t["has_skip_marker"] and not r["has_skip_marker"]:
            violations.append({
                "module": mod_id, "kind": "skip_marker_from_tmp_only",
                "detail": "no silent-skip marker at repo root, but a "
                          "silent-skip marker (\"not found\"/\"[SKIP]\"/etc.) "
                          "appeared when run from /tmp -- source resolution "
                          "is CWD-dependent for this module",
            })

        if (r["check_total"] is not None and t["check_total"] is not None
                and t["check_total"] < r["check_total"]):
            violations.append({
                "module": mod_id, "kind": "fewer_checks_from_tmp",
                "detail": f"repo-root run executed {r['check_total']} checks, "
                          f"/tmp run executed only {t['check_total']} while "
                          f"still reporting status={t['status']!r} -- vacuous "
                          f"partial-pass class",
            })

    for mod_id, t in tmp.modules.items():
        if mod_id not in root.modules and not t["advisory"]:
            violations.append({
                "module": mod_id, "kind": "appeared_only_in_tmp",
                "detail": "mandatory module present in /tmp run but missing "
                          "from repo-root run",
            })

    return violations


def discover_binary(lib_root: Path) -> Path | None:
    patterns = [
        "build/audit/unified_audit_runner",
        "build/*/audit/unified_audit_runner",
        "build*/audit/unified_audit_runner",
        "build*/**/audit/unified_audit_runner",
    ]
    candidates: list[Path] = []
    for pattern in patterns:
        for path in lib_root.glob(pattern):
            if path.is_file():
                candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def build_fresh(lib_root: Path, build_dir: Path) -> Path | None:
    """Configure + build ONLY the unified_audit_runner target into a bounded
    scratch directory (never a new in-repo directory -- build-hygiene rule).
    Returns the binary path, or None if cmake/ninja/a compiler is missing."""
    if shutil.which("cmake") is None:
        return None
    cmake_cmd = [
        "cmake", "-S", str(lib_root), "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DSECP256K1_BUILD_CUDA=OFF",
        "-DSECP256K1_BUILD_OPENCL=OFF",
        "-DSECP256K1_BUILD_METAL=OFF",
        # unified_audit_runner is only configured when the audit/ subdirectory
        # is added, which is gated on SECP256K1_BUILD_TESTS -- must be ON even
        # though this gate only needs the ONE audit_runner target, not the
        # full test suite.
        "-DSECP256K1_BUILD_TESTS=ON",
        "-DSECP256K1_BUILD_BENCH=OFF",
        "-DSECP256K1_BUILD_EXAMPLES=OFF",
    ]
    if shutil.which("ninja") is not None:
        cmake_cmd += ["-G", "Ninja"]
    configure = subprocess.run(cmake_cmd, capture_output=True, text=True)
    if configure.returncode != 0:
        print("build_fresh: cmake configure failed:\n" + configure.stdout[-4000:] +
              configure.stderr[-4000:], file=sys.stderr)
        return None
    build = subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "unified_audit_runner", "-j"],
        capture_output=True, text=True,
    )
    if build.returncode != 0:
        print("build_fresh: cmake build failed:\n" + build.stdout[-4000:] +
              build.stderr[-4000:], file=sys.stderr)
        return None
    for candidate in build_dir.glob("**/unified_audit_runner"):
        if candidate.is_file():
            return candidate
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, default=None,
                     help="Pre-built unified_audit_runner to use instead of "
                          "discovering/building one")
    ap.add_argument("--build", action="store_true",
                     help="Force a fresh build even if an existing binary is "
                          "discovered (matches Codex's 'freshly built' repro)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not RUNNER_SRC.is_file():
        print(f"ERROR: {RUNNER_SRC} not found", file=sys.stderr)
        return 1

    ordered_modules = parse_all_modules_ordered(RUNNER_SRC.read_text())
    if not ordered_modules:
        print("ERROR: could not structurally parse ALL_MODULES[] from "
              f"{RUNNER_SRC}", file=sys.stderr)
        return 1

    scratch_ctx = None
    binary = args.binary
    if binary is None and not args.build:
        binary = discover_binary(LIB_ROOT)
    if binary is None:
        scratch_ctx = tempfile.TemporaryDirectory(prefix="ufsecp_cwd_gate_build_")
        binary = build_fresh(LIB_ROOT, Path(scratch_ctx.name) / "build")
        if binary is None:
            print("ADVISORY SKIP: no pre-built unified_audit_runner found and "
                  "a fresh build could not be produced in this environment "
                  "(missing cmake/compiler) -- cannot exercise the dual-CWD "
                  "gate here", file=sys.stderr)
            if scratch_ctx is not None:
                scratch_ctx.cleanup()
            return ADVISORY_SKIP_CODE

    # Round 8: resolve to an absolute path BEFORE any run_once() call changes
    # the subprocess cwd to an unrelated /tmp directory. A relative --binary
    # (or, in principle, a relative discover_binary()/build_fresh() result)
    # would otherwise fail to execute at all once cwd != the directory it was
    # relative to -- a real defect found live while building this fix.
    binary = binary.resolve()

    unrelated_ctx = tempfile.TemporaryDirectory(prefix="ufsecp_cwd_gate_probe_")
    try:
        unrelated_dir = Path(unrelated_ctx.name)

        # Round 8: ask the binary itself for its exact active module set,
        # from BOTH candidate CWDs. get_authoritative_module_ids() does zero
        # file I/O and should therefore be perfectly CWD-invariant -- cross-
        # checking both calls against each other, rather than trusting one
        # and assuming the other would match, is the same "don't take
        # invariance on faith" discipline the rest of this gate already
        # applies everywhere else.
        root_auth_ids = get_authoritative_module_ids(binary, LIB_ROOT)
        tmp_auth_ids = get_authoritative_module_ids(binary, unrelated_dir)
        if root_auth_ids is None or tmp_auth_ids is None:
            print("ERROR: unified_audit_runner --list-modules did not produce "
                  "a usable module list (binary predates the round-8 "
                  "--list-modules flag, crashed, or timed out) -- rebuild "
                  "unified_audit_runner from current source before running "
                  "this gate. This is a hard failure, not an advisory skip: "
                  "the flag is a required capability of the binary this gate "
                  "checks, not optional external infrastructure.",
                  file=sys.stderr)
            return 1
        if root_auth_ids != tmp_auth_ids:
            print("ERROR: unified_audit_runner --list-modules itself returned "
                  "a different module set depending on CWD "
                  f"(repo-root: {len(root_auth_ids)} ids, /tmp: "
                  f"{len(tmp_auth_ids)} ids) -- --list-modules is documented "
                  "to be zero-I/O and CWD-invariant by construction; this "
                  "would be a deeper bug in --list-modules itself, not in "
                  "the modules it lists.", file=sys.stderr)
            return 1
        authoritative_ids = root_auth_ids

        root_result = run_once(binary, LIB_ROOT, ordered_modules,
                                authoritative_ids, "repo_root")
        tmp_result = run_once(binary, unrelated_dir, ordered_modules,
                               authoritative_ids, "unrelated_tmp")
    finally:
        unrelated_ctx.cleanup()
        if scratch_ctx is not None:
            scratch_ctx.cleanup()

    violations = compare_runs(root_result, tmp_result)

    # Informational-only reconciliation for the "453 ALL_MODULES rows in
    # source vs N in the authoritative runtime set" class of question (issue
    # #335 item 43): the source always declares every row unconditionally
    # (a Python regex cannot evaluate `#if SECP256K1_HAS_*`), while
    # `--list-modules` already reflects this binary's actual compiled-in
    # set. This difference is reported for visibility ONLY -- it is never
    # used for pass/fail (see build_run_result()'s exact-equality check
    # against `authoritative_ids`, not against `source_ids`), and is
    # reported as a precise id list, never as a blanket
    # "conditionally_excluded" bucket.
    source_ids = {r["id"] for r in ordered_modules}
    source_rows_not_in_authoritative_set = sorted(source_ids - authoritative_ids)

    report = {
        "binary": str(binary),
        "all_modules_rows_in_source": len(ordered_modules),
        "authoritative_module_ids_count": len(authoritative_ids),
        "source_rows_not_in_authoritative_set": source_rows_not_in_authoritative_set,
        "modules_observed_at_runtime": len(root_result.modules) if root_result.ok else None,
        "mandatory_modules_checked": sum(
            1 for r in ordered_modules
            if not r["advisory"] and r["id"] in authoritative_ids
        ),
        "root_run_ok": root_result.ok,
        "root_returncode": root_result.returncode,
        "tmp_run_ok": tmp_result.ok,
        "tmp_returncode": tmp_result.returncode,
        "violations": violations,
        "result": "PASS" if not violations else "FAIL",
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if not violations else 1

    print("=" * 64)
    print("Audit CWD-Independence Gate (issue #335, round 8)")
    print("=" * 64)
    print(f"Binary                        : {binary}")
    print(f"Repo-root run                 : ok={root_result.ok} returncode={root_result.returncode}")
    print(f"Unrelated /tmp run            : ok={tmp_result.ok} returncode={tmp_result.returncode}")
    print(f"ALL_MODULES rows in source    : {report['all_modules_rows_in_source']}")
    print(f"Authoritative module ids (--list-modules): {report['authoritative_module_ids_count']}")
    print(f"Modules observed at runtime   : {report['modules_observed_at_runtime']}")
    if source_rows_not_in_authoritative_set:
        print(f"Source rows not in authoritative set (feature-flagged out of "
              f"THIS build, informational only): "
              f"{len(source_rows_not_in_authoritative_set)} -- "
              f"{source_rows_not_in_authoritative_set}")
    print(f"Mandatory modules checked     : {report['mandatory_modules_checked']}")
    print()
    if violations:
        print(f"FAIL -- {len(violations)} mandatory module(s) diverged between "
              "repo-root and /tmp:")
        for v in violations:
            print(f"  - [{v['kind']}] {v['module']}: {v['detail']}")
    else:
        print("PASS -- every mandatory module is dual-CWD-consistent")
    print()
    print("RESULT:", report["result"])
    return 0 if not violations else 1


if __name__ == "__main__":
    sys.exit(main())
