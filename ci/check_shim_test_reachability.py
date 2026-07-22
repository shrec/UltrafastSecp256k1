#!/usr/bin/env python3
"""check_shim_test_reachability.py -- deterministic, local (no CI required) gate.

Verifies that every shim-dependent CTest target in audit/CMakeLists.txt is
actually SELECTED (not just built) by the ctest invocation used in the
`shim-gate` job's "Run standalone shim regression CTest targets" step in
.github/workflows/gate.yml.

Background (the bug this script exists to catch)
--------------------------------------------------
The shim-gate CI job builds the libsecp256k1-shim-dependent standalone CTest
binaries via `cmake --build ... --target shim_security_gate_standalones`, then
selects which of them to actually RUN via a `ctest -R <regex>` (or, after this
fix, `ctest -L <label>`) expression. Because that selection expression used to
be a hand-written substring alternation
(`regression_shim|regression_ecdsa_batch_curve|exploit_shim|test_shim`), any
shim-dependent test whose CTest NAME didn't happen to contain one of those four
substrings was silently built but never executed as a mandatory check -- e.g.
`regression_musig_xonly_zero_tweak`, `regression_musig_noncegen_extra_input`,
and `regression_ecdh_xy64_erase` (21 tests total were affected; see --self-test).

This script re-derives, structurally from audit/CMakeLists.txt (not from any
hand-maintained list), the full set of CTest targets that carry the "shim"
CTest label -- the same signal gate.yml's `ctest -L "^shim$"` selects on --
then independently parses gate.yml's CURRENT selection mechanism and checks
that every one of those targets would actually be selected. It is the single
source of truth gate.yml itself consumes (via --list-targets) so the "what
needs building" and "what needs running" lists cannot drift apart the way the
old hand-maintained `_shim_security_gate_standalone_targets` CMake list and the
old regex did.

Usage
-----
  python3 ci/check_shim_test_reachability.py                # run the real gate (exit 0/1)
  python3 ci/check_shim_test_reachability.py --list-targets  # CMake target names to build (newline-separated)
  python3 ci/check_shim_test_reachability.py --list-ctest-names  # CTest NAMEs (newline-separated)
  python3 ci/check_shim_test_reachability.py --self-test     # prove the OLD regex missed the bug; NEW mechanism doesn't
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # libs/UltrafastSecp256k1
CMAKE_LISTS = REPO_ROOT / "audit" / "CMakeLists.txt"
GATE_YML = REPO_ROOT / ".github" / "workflows" / "gate.yml"

# The fragile substring-alternation regex used by the shim-gate job's ctest
# invocation BEFORE this fix. Kept here ONLY so --self-test can deterministically
# reproduce the silent-omission bug against the real, current CTest name set.
# Do NOT resurrect this as the live selection mechanism in gate.yml.
OLD_REGEX_PRE_FIX = r"regression_shim|regression_ecdsa_batch_curve|exploit_shim|test_shim"

# The 3 tests an external reviewer specifically flagged as built-but-never-run.
NAMED_REVIEW_TESTS = [
    "regression_musig_xonly_zero_tweak",
    "regression_musig_noncegen_extra_input",
    "regression_ecdh_xy64_erase",
]


def _read(path: Path) -> str:
    if not path.is_file():
        print(f"::error::check_shim_test_reachability: required file not found: {path}", file=sys.stderr)
        sys.exit(2)
    return path.read_text()


# ---------------------------------------------------------------------------
# Ground truth: shim-dependent CTest targets, derived structurally from
# audit/CMakeLists.txt.
#
# The authoritative signal is the CTest LABELS "shim" token: audit/CMakeLists.txt
# already uses this label consistently for every test that requires
# secp256k1_shim to be linked (target_link_libraries(... secp256k1_shim ...)) AND
# for a handful of source-scan / no-link-required tests that are thematically
# "shim security regression" tests by their own doc comments (e.g.
# regression_shim_seckey_erase: "no shim/GPU dependency"; regression_shim_rfc6979_compat:
# "no GPU/shim dependency"; regression_musig_keyagg_lifetime: "no shim link required")
# but are still meant to run as part of this mandatory gate. Basing ground truth
# purely on target_link_libraries(secp256k1_shim) would UNDER-select (miss those
# source-scan tests, confirmed missing from a real on-machine ctest -L "^shim$"
# run during development of this script); basing it purely on a hardcoded name
# pattern would OVER-fit and rot. The label is what gate.yml's ctest -L actually
# selects on, so it is also what drives --list-targets here. See
# find_shim_link_dependent_targets() below for a secondary, informational
# cross-check that warns if a target structurally linking secp256k1_shim is
# missing the label (which would silently drop it from selection).
# ---------------------------------------------------------------------------

def get_all_ctest_name_to_target(cmake_text: str) -> dict[str, str]:
    """Literal add_test(NAME <ctest> COMMAND <target>) pairs (no CMake variables)."""
    at_re = re.compile(
        r"add_test\s*\(\s*NAME\s+([A-Za-z0-9_]+)\s+COMMAND\s+([A-Za-z0-9_]+)",
        re.DOTALL,
    )
    mapping: dict[str, str] = {}
    for m in at_re.finditer(cmake_text):
        mapping[m.group(1)] = m.group(2)
    return mapping


def get_all_ctest_labels(cmake_text: str) -> dict[str, list[str]]:
    """Literal set_tests_properties(<ctest> PROPERTIES ... LABELS "...") (no CMake variables)."""
    stp_re = re.compile(
        r"set_tests_properties\s*\(\s*([A-Za-z0-9_]+)\s+PROPERTIES(.*?)\)",
        re.DOTALL,
    )
    labels: dict[str, list[str]] = {}
    for m in stp_re.finditer(cmake_text):
        name, body = m.group(1), m.group(2)
        lm = re.search(r'LABELS\s+"([^"]*)"', body)
        labels[name] = lm.group(1).split(";") if lm else []
    return labels


def get_shim_exploit_test_macro_entries(cmake_text: str) -> dict[str, str]:
    """shim_exploit_test(_name _src _label _timeout) macro invocations ->
    {ctest_name: target}.

    This macro (audit/CMakeLists.txt, "Shim security standalone tests" block)
    generates its add_executable()/add_test()/set_tests_properties() calls
    entirely through CMake variable substitution (${_target}, ${_name}_shim,
    "...;${_label}"), so the generic literal regexes above cannot resolve its
    call sites (e.g. exploit_musig_unknown_signer_shim,
    exploit_context_flag_bypass_shim -- confirmed missing from the literal
    regexes' output against a real ctest -N during development of this script).
    The naming convention is derived directly from the macro's own definition:
      target      = test_<_name>_shim_standalone
      ctest name  = <_name>_shim
      labels      = audit;exploit;shim;<_label>  (always includes "shim")
    """
    call_re = re.compile(
        r'shim_exploit_test\s*\(\s*([A-Za-z0-9_]+)\s+\S+\s+"[^"]*"\s+\d+\s*\)',
        re.DOTALL,
    )
    entries: dict[str, str] = {}
    for m in call_re.finditer(cmake_text):
        name = m.group(1)
        entries[f"{name}_shim"] = f"test_{name}_shim_standalone"
    return entries


def find_shim_link_dependent_targets(cmake_text: str) -> set[str]:
    """Secondary, informational cross-check (NOT the primary ground truth):
    CMake add_executable() TARGET names that require secp256k1_shim, either by
    linking the secp256k1_shim CMake library target, or by directly compiling
    compat/libsecp256k1_shim/src/*.cpp sources into the binary. Used by
    cmd_check() to WARN if a target that structurally needs secp256k1_shim to
    link is missing the "shim" label (which would silently drop it from the
    ctest -L "^shim$" selection this gate relies on -- the same class of bug
    this whole script exists to catch, one level up the chain).
    """
    targets: set[str] = set()

    tll_re = re.compile(
        r"target_link_libraries\s*\(\s*([A-Za-z0-9_]+)\s+[^)]*\bsecp256k1_shim\b[^)]*\)",
        re.DOTALL,
    )
    targets.update(m.group(1) for m in tll_re.finditer(cmake_text))

    ae_re = re.compile(r"add_executable\s*\(\s*([A-Za-z0-9_]+)([^)]*)\)", re.DOTALL)
    for m in ae_re.finditer(cmake_text):
        if "libsecp256k1_shim/src" in m.group(2):
            targets.add(m.group(1))

    # unified_audit_runner links secp256k1_shim too, but it is a SEPARATE binary
    # validated by the "Run shim security regression modules" JSON step
    # (--json-only) earlier in the same job -- its shim-dependent modules are
    # INTENTIONALLY advisory-stubbed there (audit/ is processed before
    # compat/libsecp256k1_shim/ at CMake configure time; see the comment on that
    # step in gate.yml). It is not one of the standalone CTest targets this gate
    # tracks, so exclude it to avoid a false "missing label" warning.
    targets.discard("unified_audit_runner")
    return targets


def shim_dependent_ctest_names(cmake_text: str) -> dict[str, str]:
    """Return {ctest_name: cmake_target} for every CTest test registered in
    audit/CMakeLists.txt that carries the exact label token "shim" -- the
    ground truth used both to build (--list-targets) and to verify selection
    (default mode). Union of literal registrations and the
    shim_exploit_test(...) macro convention.
    """
    name_to_target = get_all_ctest_name_to_target(cmake_text)
    name_to_labels = get_all_ctest_labels(cmake_text)
    macro_entries = get_shim_exploit_test_macro_entries(cmake_text)

    result: dict[str, str] = {}
    for name, target in name_to_target.items():
        if "shim" in name_to_labels.get(name, []):
            result[name] = target
    result.update(macro_entries)

    # unified_audit_runner's own CTest registration (if any) is a separate
    # binary/step -- see find_shim_link_dependent_targets() docstring.
    result.pop("unified_audit", None)

    return result


def find_unlabeled_link_dependent_targets(cmake_text: str) -> set[str]:
    """Targets that structurally link secp256k1_shim (or compile its sources
    directly) but whose CTest registration does NOT carry the "shim" label --
    i.e. targets that WOULD silently fall out of ctest -L "^shim$" selection.
    Informational only (does not fail cmd_check() on its own record; see
    cmd_check()'s use of this).
    """
    link_targets = find_shim_link_dependent_targets(cmake_text)
    labeled_targets = set(shim_dependent_ctest_names(cmake_text).values())
    return link_targets - labeled_targets


# ---------------------------------------------------------------------------
# gate.yml mechanism extraction
# ---------------------------------------------------------------------------

def extract_shim_gate_job_text(gate_yml_text: str) -> str:
    m = re.search(r"\n  shim-gate:\n", gate_yml_text)
    if not m:
        print("::error::check_shim_test_reachability: could not find 'shim-gate:' job in gate.yml", file=sys.stderr)
        sys.exit(2)
    start = m.start() + 1
    tail = gate_yml_text[m.end():]
    m2 = re.search(r"\n  [A-Za-z0-9_-]+:\n", tail)
    end = m.end() + m2.start() if m2 else len(gate_yml_text)
    return gate_yml_text[start:end]


def extract_ctest_mechanism(job_text: str) -> tuple[str, str]:
    """Return ('label', pattern) or ('regex', pattern) describing how the
    'Run standalone shim regression CTest targets' step currently selects tests.
    """
    step_m = re.search(
        r"Run standalone shim regression CTest targets.*?(?=\n\s{6}- name:|\Z)",
        job_text,
        re.DOTALL,
    )
    if not step_m:
        print(
            "::error::check_shim_test_reachability: could not find the "
            "'Run standalone shim regression CTest targets' step in gate.yml",
            file=sys.stderr,
        )
        sys.exit(2)
    step_text = step_m.group(0)

    lm = re.search(r'ctest\b.*?-L\s+"([^"]+)"', step_text, re.DOTALL)
    if lm:
        return ("label", lm.group(1))
    rm = re.search(r'ctest\b.*?-R\s+"([^"]+)"', step_text, re.DOTALL)
    if rm:
        return ("regex", rm.group(1))
    print(
        "::error::check_shim_test_reachability: could not find a ctest -L or -R "
        "selection expression in the 'Run standalone shim regression CTest targets' step",
        file=sys.stderr,
    )
    sys.exit(2)


# ---------------------------------------------------------------------------
# Selection evaluation
# ---------------------------------------------------------------------------

def is_selected(mechanism: tuple[str, str], name: str, labels: list[str]) -> bool:
    kind, pattern = mechanism
    if kind == "regex":
        # ctest -R matches the regex against the CTest test NAME.
        return re.search(pattern, name) is not None
    if kind == "label":
        # ctest -L matches the regex against EACH label individually (not the
        # raw semicolon-joined string).
        return any(re.search(pattern, lbl) for lbl in labels)
    raise ValueError(f"unknown mechanism kind: {kind!r}")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def _labels_for(cmake_text: str, ctest_names: dict[str, str]) -> dict[str, list[str]]:
    """Best-effort LABELS lookup for every name in ctest_names, covering both
    literal registrations and shim_exploit_test(...) macro entries (whose
    labels are reconstructed from the macro convention, since the literal
    text uses ${_label} substitution).
    """
    literal_labels = get_all_ctest_labels(cmake_text)
    macro_call_re = re.compile(
        r'shim_exploit_test\s*\(\s*([A-Za-z0-9_]+)\s+\S+\s+"([^"]*)"\s+\d+\s*\)',
        re.DOTALL,
    )
    macro_labels: dict[str, list[str]] = {}
    for m in macro_call_re.finditer(cmake_text):
        name, label = m.group(1), m.group(2)
        macro_labels[f"{name}_shim"] = ["audit", "exploit", "shim", label]

    result: dict[str, list[str]] = {}
    for name in ctest_names:
        if name in literal_labels:
            result[name] = literal_labels[name]
        elif name in macro_labels:
            result[name] = macro_labels[name]
        else:
            result[name] = []
    return result


def _load() -> tuple[dict[str, str], dict[str, list[str]], str, str]:
    cmake_text = _read(CMAKE_LISTS)
    gate_text = _read(GATE_YML)
    ctest_names = shim_dependent_ctest_names(cmake_text)  # name -> target
    labels = _labels_for(cmake_text, ctest_names)  # name -> [labels]
    return ctest_names, labels, gate_text, cmake_text


def cmd_list_targets() -> None:
    cmake_text = _read(CMAKE_LISTS)
    ctest_names = shim_dependent_ctest_names(cmake_text)
    for target in sorted(set(ctest_names.values())):
        print(target)


def cmd_list_ctest_names() -> None:
    cmake_text = _read(CMAKE_LISTS)
    ctest_names = shim_dependent_ctest_names(cmake_text)
    for name in sorted(ctest_names):
        print(name)


def cmd_check() -> int:
    ctest_names, labels, gate_text, cmake_text = _load()
    job_text = extract_shim_gate_job_text(gate_text)
    mechanism = extract_ctest_mechanism(job_text)

    unreachable = [
        name for name in sorted(ctest_names)
        if not is_selected(mechanism, name, labels.get(name, []))
    ]

    print(f"check_shim_test_reachability: gate.yml mechanism = {mechanism[0]} {mechanism[1]!r}")
    print(f"check_shim_test_reachability: {len(ctest_names)} shim-dependent CTest target(s) found in audit/CMakeLists.txt")

    ok = True
    if unreachable:
        ok = False
        print(
            f"::error::check_shim_test_reachability: {len(unreachable)} shim-dependent "
            f"CTest target(s) are built but would NOT be selected by the current "
            f"gate.yml mechanism ({mechanism[0]} {mechanism[1]!r}):",
            file=sys.stderr,
        )
        for name in unreachable:
            print(f"  - {name}  (target: {ctest_names[name]}, labels: {labels.get(name, [])})", file=sys.stderr)

    unlabeled = sorted(find_unlabeled_link_dependent_targets(cmake_text))
    if unlabeled:
        print(
            f"::warning::check_shim_test_reachability: {len(unlabeled)} CMake target(s) "
            f"link secp256k1_shim (or compile its sources directly) but their CTest "
            f"registration is missing the \"shim\" label, so they will NOT be selected "
            f"by ctest -L \"^shim$\": {unlabeled}",
            file=sys.stderr,
        )

    if not ok:
        return 1

    print("check_shim_test_reachability: OK -- all shim-dependent CTest targets are reachable under the current gate.yml mechanism.")
    return 0


def cmd_self_test() -> int:
    ctest_names, labels, gate_text, _cmake_text = _load()
    job_text = extract_shim_gate_job_text(gate_text)
    new_mechanism = extract_ctest_mechanism(job_text)
    old_mechanism: tuple[str, str] = ("regex", OLD_REGEX_PRE_FIX)

    old_missed = [
        n for n in sorted(ctest_names)
        if not is_selected(old_mechanism, n, labels.get(n, []))
    ]
    new_missed = [
        n for n in sorted(ctest_names)
        if not is_selected(new_mechanism, n, labels.get(n, []))
    ]

    print(f"--- self-test: OLD_REGEX_PRE_FIX = {OLD_REGEX_PRE_FIX!r} ---")
    print(f"OLD mechanism misses {len(old_missed)} / {len(ctest_names)} real shim-dependent CTest targets:")
    for n in old_missed:
        print(f"  - {n}")

    ok = True
    print(f"\n--- self-test: the 3 externally-reviewed tests, OLD vs NEW ({new_mechanism[0]} {new_mechanism[1]!r}) ---")
    for name in NAMED_REVIEW_TESTS:
        if name not in ctest_names:
            print(f"::error::self-test: expected shim-dependent test {name!r} not found by the CMakeLists.txt parser", file=sys.stderr)
            ok = False
            continue
        old_sel = is_selected(old_mechanism, name, labels.get(name, []))
        new_sel = is_selected(new_mechanism, name, labels.get(name, []))
        status = "OK" if (not old_sel and new_sel) else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  {name}: OLD selected={old_sel}  NEW selected={new_sel}  [{status}]")

    print(f"\nNEW mechanism ({new_mechanism[0]} {new_mechanism[1]!r}) misses {len(new_missed)} / {len(ctest_names)}")
    if new_missed:
        for n in new_missed:
            print(f"  still unreachable under NEW mechanism: {n}")
        ok = False

    if not old_missed:
        print("::error::self-test: OLD_REGEX_PRE_FIX did not miss anything -- self-test cannot demonstrate the bug it exists to prove", file=sys.stderr)
        ok = False

    if ok:
        print(
            "\nself-test PASSED: OLD regex reproduces the silent-omission bug on the "
            "real, current CTest name set; NEW mechanism selects every shim-dependent target."
        )
        return 0
    print("\nself-test FAILED.", file=sys.stderr)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-targets", action="store_true", help="print shim-dependent CMake build target names (newline-separated)")
    ap.add_argument("--list-ctest-names", action="store_true", help="print shim-dependent CTest names (newline-separated)")
    ap.add_argument("--self-test", action="store_true", help="prove the OLD regex missed the bug and the NEW mechanism does not")
    args = ap.parse_args()

    if args.list_targets:
        cmd_list_targets()
        return 0
    if args.list_ctest_names:
        cmd_list_ctest_names()
        return 0
    if args.self_test:
        return cmd_self_test()
    return cmd_check()


if __name__ == "__main__":
    sys.exit(main())
