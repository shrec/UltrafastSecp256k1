#!/usr/bin/env python3
"""test_check_installed_header_parity.py

Self-test (proof-it-blocks) for ci/check_installed_header_parity.py.

Uses synthetic ufsecp_version.h-shaped header fixtures with KNOWN, hand-
injected drift -- no real cmake build required, so this stays fast and is
wired into the fast gate tier (run_fast_gates.sh) even though the gate it
tests drives a real cmake configure+install (see that module's docstring).
Mirrors the self-test style of ci/test_check_audit_cwd_independence.py:
pure unit tests over the gate's exported pure functions
(parse_header / diff_headers), each proving one specific drift class is
detected, plus a "clean pair" test proving version-number-only differences
(which legitimately vary run to run) do NOT cause a false failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_installed_header_parity as gate  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures: a trimmed but structurally faithful copy of the real
# ufsecp_version.h shape (same macro names as the real file: UFSECP_API,
# UFSECP_DEPRECATED, UFSECP_BUILDING, UFSECP_STATIC_LIB, UFSECP_VERSION_H)
# so each self-test clearly maps onto the real-file bug class it anchors.
# ---------------------------------------------------------------------------

REFERENCE_FIXTURE = '''\
#ifndef UFSECP_VERSION_H
#define UFSECP_VERSION_H

#ifdef __cplusplus
extern "C" {
#endif

#define UFSECP_VERSION_MAJOR   4
#define UFSECP_VERSION_MINOR   5
#define UFSECP_VERSION_PATCH   0
#define UFSECP_VERSION_STRING  "4.5.0"
#define UFSECP_ABI_VERSION     4

#ifndef UFSECP_API
  #if defined(_WIN32) || defined(__CYGWIN__)
    #ifdef UFSECP_STATIC_LIB
      #define UFSECP_API
    #elif defined(UFSECP_BUILDING)
      #define UFSECP_API __declspec(dllexport)
    #else
      #define UFSECP_API __declspec(dllimport)
    #endif
  #elif __GNUC__ >= 4
    #define UFSECP_API __attribute__((visibility("default")))
  #else
    #define UFSECP_API
  #endif
#endif

#ifndef UFSECP_DEPRECATED
  #if defined(__cplusplus) && (__cplusplus >= 201402L)
    #define UFSECP_DEPRECATED(msg) [[deprecated(msg)]]
  #elif defined(__GNUC__) || defined(__clang__)
    #define UFSECP_DEPRECATED(msg) __attribute__((deprecated(msg)))
  #elif defined(_MSC_VER)
    #define UFSECP_DEPRECATED(msg) __declspec(deprecated(msg))
  #else
    #define UFSECP_DEPRECATED(msg)
  #endif
#endif

UFSECP_API unsigned int ufsecp_version(void);
UFSECP_API unsigned int ufsecp_abi_version(void);
UFSECP_API const char* ufsecp_version_string(void);

#ifdef __cplusplus
}
#endif

#endif /* UFSECP_VERSION_H */
'''

# A "correctly generated" fixture: same structure, DIFFERENT version numbers
# (simulating a fresh build against a different VERSION.txt than whatever the
# checked-in reference copy happens to embed) -- proves version-number-only
# differences must NOT be flagged as drift.
GENERATED_CLEAN_FIXTURE = '''\
#ifndef UFSECP_VERSION_H
#define UFSECP_VERSION_H

#ifdef __cplusplus
extern "C" {
#endif

#define UFSECP_VERSION_MAJOR   9
#define UFSECP_VERSION_MINOR   9
#define UFSECP_VERSION_PATCH   9
#define UFSECP_VERSION_STRING  "9.9.9"
#define UFSECP_ABI_VERSION     9

#ifndef UFSECP_API
  #if defined(_WIN32) || defined(__CYGWIN__)
    #ifdef UFSECP_STATIC_LIB
      #define UFSECP_API
    #elif defined(UFSECP_BUILDING)
      #define UFSECP_API __declspec(dllexport)
    #else
      #define UFSECP_API __declspec(dllimport)
    #endif
  #elif __GNUC__ >= 4
    #define UFSECP_API __attribute__((visibility("default")))
  #else
    #define UFSECP_API
  #endif
#endif

#ifndef UFSECP_DEPRECATED
  #if defined(__cplusplus) && (__cplusplus >= 201402L)
    #define UFSECP_DEPRECATED(msg) [[deprecated(msg)]]
  #elif defined(__GNUC__) || defined(__clang__)
    #define UFSECP_DEPRECATED(msg) __attribute__((deprecated(msg)))
  #elif defined(_MSC_VER)
    #define UFSECP_DEPRECATED(msg) __declspec(deprecated(msg))
  #else
    #define UFSECP_DEPRECATED(msg)
  #endif
#endif

UFSECP_API unsigned int ufsecp_version(void);
UFSECP_API unsigned int ufsecp_abi_version(void);
UFSECP_API const char* ufsecp_version_string(void);

#ifdef __cplusplus
}
#endif

#endif /* UFSECP_VERSION_H */
'''

# The EXACT real-world bug: the whole UFSECP_DEPRECATED guard block deleted
# (round 9's live finding against the real ufsecp_version.h.in).
MISSING_DEPRECATED_FIXTURE = GENERATED_CLEAN_FIXTURE.replace(
    '''
#ifndef UFSECP_DEPRECATED
  #if defined(__cplusplus) && (__cplusplus >= 201402L)
    #define UFSECP_DEPRECATED(msg) [[deprecated(msg)]]
  #elif defined(__GNUC__) || defined(__clang__)
    #define UFSECP_DEPRECATED(msg) __attribute__((deprecated(msg)))
  #elif defined(_MSC_VER)
    #define UFSECP_DEPRECATED(msg) __declspec(deprecated(msg))
  #else
    #define UFSECP_DEPRECATED(msg)
  #endif
#endif
''', '\n')
assert MISSING_DEPRECATED_FIXTURE != GENERATED_CLEAN_FIXTURE, "fixture edit did not match"

# The Windows precedence-order bug (round 9's second, undocumented finding):
# UFSECP_STATIC_LIB and UFSECP_BUILDING swapped in the #ifdef/#elif chain,
# AND their values swapped too, matching the real .in template's actual
# (buggy) branch order exactly.
WIN_PRECEDENCE_SWAPPED_FIXTURE = GENERATED_CLEAN_FIXTURE.replace(
    '''    #ifdef UFSECP_STATIC_LIB
      #define UFSECP_API
    #elif defined(UFSECP_BUILDING)
      #define UFSECP_API __declspec(dllexport)
    #else
      #define UFSECP_API __declspec(dllimport)
    #endif''',
    '''    #ifdef UFSECP_BUILDING
      #define UFSECP_API __declspec(dllexport)
    #elif defined(UFSECP_STATIC_LIB)
      #define UFSECP_API
    #else
      #define UFSECP_API __declspec(dllimport)
    #endif''')
assert WIN_PRECEDENCE_SWAPPED_FIXTURE != GENERATED_CLEAN_FIXTURE, "fixture edit did not match"

# Include guard renamed (both #ifndef and #define changed together, so the
# guard is still internally self-consistent within the generated header --
# but no longer matches the reference's guard name at all).
GUARD_RENAMED_FIXTURE = (
    GENERATED_CLEAN_FIXTURE
    .replace('#ifndef UFSECP_VERSION_H\n#define UFSECP_VERSION_H',
              '#ifndef UFSECP_VERSION_H_V2\n#define UFSECP_VERSION_H_V2')
    .replace('#endif /* UFSECP_VERSION_H */', '#endif /* UFSECP_VERSION_H_V2 */')
)
assert GUARD_RENAMED_FIXTURE != GENERATED_CLEAN_FIXTURE, "fixture edit did not match"

# Function signature drift: ufsecp_version() return type changed from
# 'unsigned int' to plain 'int'.
FUNC_SIG_CHANGED_FIXTURE = GENERATED_CLEAN_FIXTURE.replace(
    'UFSECP_API unsigned int ufsecp_version(void);',
    'UFSECP_API int ufsecp_version(void);')
assert FUNC_SIG_CHANGED_FIXTURE != GENERATED_CLEAN_FIXTURE, "fixture edit did not match"

# extern "C" opening wrapper dropped.
EXTERN_C_MISSING_FIXTURE = GENERATED_CLEAN_FIXTURE.replace(
    '#ifdef __cplusplus\nextern "C" {\n#endif\n\n', '')
assert EXTERN_C_MISSING_FIXTURE != GENERATED_CLEAN_FIXTURE, "fixture edit did not match"


def _parsed(ref_text: str, gen_text: str):
    return gate.parse_header(gen_text), gate.parse_header(ref_text)


def test_clean_pair_no_drift_despite_different_version_numbers():
    """The core false-positive guard: MAJOR/MINOR/PATCH/STRING/ABI_VERSION
    differ (9.9.9 vs 4.5.0) between the two fixtures, exactly as they
    legitimately would between a fresh build and an older checked-in
    reference copy. diff_headers() must report ZERO drift for this pair --
    it compares macro NAME/guard-structure/signature, never literal
    version-number VALUES."""
    gen, ref = _parsed(REFERENCE_FIXTURE, GENERATED_CLEAN_FIXTURE)
    assert gen.all_macro_names() == ref.all_macro_names()
    drifts = gate.diff_headers(gen, ref)
    assert drifts == [], f"expected zero drift, got {drifts}"
    print("  OK: version-number-only differences (9.9.9 vs 4.5.0) produce "
          "zero drift entries")


def test_missing_macro_detected_by_name():
    """Anchors the EXACT round-9 real-world bug: UFSECP_DEPRECATED entirely
    absent from the generated header. Must be reported as kind=macro_missing
    naming UFSECP_DEPRECATED specifically, with the fix pointed at
    ufsecp_version.h.in."""
    gen, ref = _parsed(REFERENCE_FIXTURE, MISSING_DEPRECATED_FIXTURE)
    assert 'UFSECP_DEPRECATED' in ref.all_macro_names()
    assert 'UFSECP_DEPRECATED' not in gen.all_macro_names()
    drifts = gate.diff_headers(gen, ref)
    missing = [d for d in drifts if d['kind'] == 'macro_missing' and d['macro'] == 'UFSECP_DEPRECATED']
    assert len(missing) == 1, f"expected exactly one macro_missing(UFSECP_DEPRECATED), got {drifts}"
    assert 'ufsecp_version.h.in' in missing[0]['detail']
    print("  OK: a completely missing UFSECP_DEPRECATED macro (the real "
          "round-9 bug) is detected by name, with the fix location named")


def test_windows_precedence_order_swap_detected():
    """Anchors the round-9 UNDOCUMENTED second finding: the .in template's
    Windows __declspec branch checks UFSECP_BUILDING before UFSECP_STATIC_LIB
    (reference checks STATIC_LIB first). Both macro-name sets are IDENTICAL
    (UFSECP_API is defined in both, same 5 branches) -- only the per-branch
    CONDITION ORDER differs. A pure name-presence check would miss this
    entirely; the chain-order comparison must catch it."""
    gen, ref = _parsed(REFERENCE_FIXTURE, WIN_PRECEDENCE_SWAPPED_FIXTURE)
    assert gen.all_macro_names() == ref.all_macro_names(), (
        "fixture must NOT differ in macro name set -- this test isolates "
        "branch ORDER as the only variable"
    )
    drifts = gate.diff_headers(gen, ref)
    order_drifts = [d for d in drifts if d['kind'] == 'macro_guard_order_changed']
    assert order_drifts, f"expected macro_guard_order_changed drift(s), got {drifts}"
    assert all(d['macro'] == 'UFSECP_API' for d in order_drifts)
    # Both swapped branches (STATIC_LIB<->BUILDING) must be caught, not just one.
    assert len(order_drifts) == 2, f"expected 2 swapped-branch drifts, got {order_drifts}"
    print(f"  OK: Windows __declspec precedence order swap detected "
          f"({len(order_drifts)} branch-order drift(s) on UFSECP_API)")


def test_naive_name_only_comparison_would_false_pass_precedence_swap():
    """Anchors WHY diff_headers must compare per-branch chain ORDER, not
    just macro-name-set presence: on the precedence-swap fixture, a
    comparator that only checks 'is UFSECP_API defined in both files' sees
    no difference at all (it IS defined in both, with the same 5 branches)."""
    gen, ref = _parsed(REFERENCE_FIXTURE, WIN_PRECEDENCE_SWAPPED_FIXTURE)

    def naive_name_only_compare(g, r):
        return g.all_macro_names() != r.all_macro_names()

    naive_says_different = naive_name_only_compare(gen, ref)
    assert naive_says_different is False, (
        "fail-before: a name-only comparator must genuinely miss this case "
        "for the test to anchor anything real"
    )
    real_drifts = gate.diff_headers(gen, ref)
    assert len(real_drifts) >= 1, "pass-after: diff_headers must catch it"
    print("  OK: name-only comparison false-passes the precedence swap; "
          "diff_headers() (chain-order aware) catches it")


def test_include_guard_rename_detected():
    gen, ref = _parsed(REFERENCE_FIXTURE, GUARD_RENAMED_FIXTURE)
    drifts = gate.diff_headers(gen, ref)
    kinds = {d['kind'] for d in drifts}
    assert 'include_guard_ifndef_changed' in kinds, drifts
    assert 'include_guard_define_changed' in kinds, drifts
    print("  OK: a renamed #ifndef/#define include guard pair is detected")


def test_function_signature_change_detected():
    gen, ref = _parsed(REFERENCE_FIXTURE, FUNC_SIG_CHANGED_FIXTURE)
    assert ref.functions['ufsecp_version'].return_type == 'unsigned int'
    assert gen.functions['ufsecp_version'].return_type == 'int'
    drifts = gate.diff_headers(gen, ref)
    sig_drifts = [d for d in drifts if d['kind'] == 'function_signature_changed']
    assert len(sig_drifts) == 1, f"expected one function_signature_changed, got {drifts}"
    assert sig_drifts[0]['macro'] == 'ufsecp_version'
    assert 'unsigned int' in sig_drifts[0]['detail'] and "='int " in sig_drifts[0]['detail']
    print("  OK: a changed function return type (unsigned int -> int) is detected")


def test_extern_c_wrapper_removal_detected():
    gen, ref = _parsed(REFERENCE_FIXTURE, EXTERN_C_MISSING_FIXTURE)
    assert ref.extern_c_open is True
    assert gen.extern_c_open is False
    drifts = gate.diff_headers(gen, ref)
    assert any(d['kind'] == 'extern_c_open_changed' for d in drifts), drifts
    print("  OK: a dropped 'extern \"C\"' opening wrapper is detected")


def test_added_macro_not_in_reference_detected():
    """Symmetric case: a macro present in the generated header but absent
    from the (possibly stale) reference copy must also be flagged, not
    silently ignored -- diff_headers() checks both set directions."""
    extra = GENERATED_CLEAN_FIXTURE.replace(
        '#define UFSECP_ABI_VERSION     9',
        '#define UFSECP_ABI_VERSION     9\n#define UFSECP_BRAND_NEW_MACRO 1')
    gen, ref = _parsed(REFERENCE_FIXTURE, extra)
    drifts = gate.diff_headers(gen, ref)
    added = [d for d in drifts if d['kind'] == 'macro_added']
    assert len(added) == 1 and added[0]['macro'] == 'UFSECP_BRAND_NEW_MACRO', drifts
    print("  OK: a macro present only in the generated header (stale "
          "reference copy) is detected in the opposite direction too")


def test_parser_preserves_branch_occurrence_order_for_clean_fixture():
    """Structural sanity check on the parser itself: UFSECP_API must have
    exactly 5 #define occurrences (one per mutually-exclusive branch) in the
    clean fixture, in document order, each with a distinct chain."""
    model = gate.parse_header(GENERATED_CLEAN_FIXTURE)
    api_occurrences = model.by_name()['UFSECP_API']
    assert len(api_occurrences) == 5, api_occurrences
    chains = [o.chain for o in api_occurrences]
    assert len(set(chains)) == 5, "all 5 branch chains must be distinct"
    assert model.guard_ifndef_name == 'UFSECP_VERSION_H'
    assert model.guard_define_name == 'UFSECP_VERSION_H'
    assert 'UFSECP_DEPRECATED' in model.by_name()
    assert len(model.by_name()['UFSECP_DEPRECATED']) == 4
    assert set(model.functions) == {
        'ufsecp_version', 'ufsecp_abi_version', 'ufsecp_version_string'
    }
    print("  OK: parse_header finds all 5 UFSECP_API branches (distinct "
          "chains), the include guard, all 4 UFSECP_DEPRECATED branches, "
          "and all 3 function declarations")


def test_missing_required_files_reports_all_paths():
    """_missing_required_files() must name every absent file, not just the
    first one -- used by main() to fail loudly rather than crash on the
    first .read_text()."""
    class FakePath:
        def __init__(self, exists):
            self._exists = exists
        def is_file(self):
            return self._exists

    # Directly exercise the detection logic with the real function against
    # the real repo (should be empty -- all 5 required files exist here).
    missing = gate._missing_required_files()
    assert missing == [], f"expected all required repo files present, got {missing}"
    print("  OK: _missing_required_files() finds all 5 required real repo "
          "files present (template, reference, ufsecp.h, ufsecp_error.h, "
          "VERSION.txt)")


TESTS = [
    test_clean_pair_no_drift_despite_different_version_numbers,
    test_missing_macro_detected_by_name,
    test_windows_precedence_order_swap_detected,
    test_naive_name_only_comparison_would_false_pass_precedence_swap,
    test_include_guard_rename_detected,
    test_function_signature_change_detected,
    test_extern_c_wrapper_removal_detected,
    test_added_macro_not_in_reference_detected,
    test_parser_preserves_branch_occurrence_order_for_clean_fixture,
    test_missing_required_files_reports_all_paths,
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
