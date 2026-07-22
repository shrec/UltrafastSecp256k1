#!/usr/bin/env python3
"""test_check_version_sync.py

Self-test (proof-it-blocks) for ci/check_version_sync.py.

Issue #335 acceptance repair (round 11): check_version_sync.py was listed in
the round's own `validation` command list but had no self-test at all,
unlike every sibling gate in this repo (check_installed_header_parity.py /
check_audit_cwd_independence.py / check_advisory_skip_ceiling.py all pair a
`check_*.py` gate with a `test_check_*.py` self-test). This fills that gap.

Builds a complete, synthetic, mutually-consistent package-metadata tree under
a temp directory (VERSION.txt, ufsecp_version.h, CITATION.cff, podspec, rpm
spec, PKGBUILD, docs/README.md, Package.swift, conanfile.py, vcpkg.json,
.zenodo.json, plus a minimal unified_audit_runner.cpp / ufsecp_gpu.h for the
count-sync half) -- no real repo checkout or cmake build required, so this
stays fast and belongs in the fast gate tier (run_fast_gates.sh). Each test
mutates exactly ONE location away from the synced baseline and proves
check_version_sync.py's pure `check_version_sync()` / `check_count_sync()`
functions catch that specific drift, mirroring the self-test style already
used by test_check_installed_header_parity.py and
test_check_audit_cwd_independence.py in this same directory.
"""

from __future__ import annotations

import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_version_sync as gate  # noqa: E402


VERSION = "4.5.0"
MAJOR = "4"

_UFSECP_VERSION_H = f'''\
#ifndef UFSECP_VERSION_H
#define UFSECP_VERSION_H
#define UFSECP_VERSION_MAJOR   {MAJOR}
#define UFSECP_VERSION_MINOR   5
#define UFSECP_VERSION_PATCH   0
#define UFSECP_VERSION_STRING  "{VERSION}"
#endif
'''

_CITATION_CFF = f'''\
cff-version: 1.2.0
title: UltrafastSecp256k1
version: "{VERSION}"
'''

_PODSPEC = f'''\
Pod::Spec.new do |s|
  s.name     = "UltrafastSecp256k1"
  s.version  = "{VERSION}"
end
'''

_RPM_SPEC = f'''\
Name: libufsecp
Version: {VERSION}
%global soversion {MAJOR}
'''

_PKGBUILD = f'''\
pkgname=ultrafastsecp256k1
pkgver={VERSION}
pkgrel=1
'''

_README = f'''\
# UltrafastSecp256k1

> **Version {VERSION}** -- fast constant-time secp256k1.
'''

_PACKAGE_SWIFT = f'''\
// swift-tools-version:5.7
// .package(url: "https://example.invalid/repo.git", from: "{VERSION}")
'''

_CONANFILE = f'''\
from conan import ConanFile

class UltrafastSecp256k1Conan(ConanFile):
    name = "ultrafastsecp256k1"
    version = "{VERSION}"
'''

_VCPKG_JSON = f'{{"name": "ultrafastsecp256k1", "version": "{VERSION}"}}'

_ZENODO_JSON = f'{{"version": "{VERSION}", "title": "UltrafastSecp256k1"}}'

_UNIFIED_AUDIT_RUNNER = '''\
static const AuditModule ALL_MODULES[] = {
    {"exploit_a", "desc a", "exploit_poc", test_a_run, false},
    {"exploit_b", "desc b", "exploit_poc", test_b_run, false},
    {"regression_c", "desc c", "differential", test_c_run, false},
};

static constexpr int NUM_MODULES = 3;
'''

_UFSECP_GPU_H = '''\
/* stable batch-op
 *   surface currently includes 2 backend-neutral operations */
UFSECP_API ufsecp_error_t ufsecp_gpu_op_one(void);
UFSECP_API ufsecp_error_t ufsecp_gpu_op_two(void);
UFSECP_API ufsecp_error_t ufsecp_gpu_device_info(void);
'''


@contextmanager
def _synced_tree():
    """A complete, mutually-consistent fixture tree at VERSION/MAJOR/2 exploit
    PoCs/2 GPU ops. Yields the root Path; caller mutates one file per test."""
    with tempfile.TemporaryDirectory(prefix="ufsecp_version_sync_test_") as d:
        root = Path(d)
        (root / "include" / "ufsecp").mkdir(parents=True)
        (root / "packaging" / "cocoapods").mkdir(parents=True)
        (root / "packaging" / "rpm").mkdir(parents=True)
        (root / "packaging" / "arch").mkdir(parents=True)
        (root / "docs").mkdir(parents=True)
        (root / "audit").mkdir(parents=True)

        (root / "VERSION.txt").write_text(VERSION)
        (root / "include" / "ufsecp" / "ufsecp_version.h").write_text(_UFSECP_VERSION_H)
        (root / "CITATION.cff").write_text(_CITATION_CFF)
        (root / "packaging" / "cocoapods" / "UltrafastSecp256k1.podspec").write_text(_PODSPEC)
        (root / "packaging" / "rpm" / "libufsecp.spec").write_text(_RPM_SPEC)
        (root / "packaging" / "arch" / "PKGBUILD").write_text(_PKGBUILD)
        (root / "docs" / "README.md").write_text(_README)
        (root / "Package.swift").write_text(_PACKAGE_SWIFT)
        (root / "conanfile.py").write_text(_CONANFILE)
        (root / "vcpkg.json").write_text(_VCPKG_JSON)
        (root / ".zenodo.json").write_text(_ZENODO_JSON)
        (root / "audit" / "unified_audit_runner.cpp").write_text(_UNIFIED_AUDIT_RUNNER)
        (root / "include" / "ufsecp" / "ufsecp_gpu.h").write_text(_UFSECP_GPU_H)
        yield root


def test_fully_synced_tree_passes_both_checks():
    """A mutually-consistent tree must report OK everywhere -- the baseline
    every other test mutates away from exactly one location."""
    with _synced_tree() as root:
        assert gate.check_version_sync(root) is True
        assert gate.check_count_sync(root) is True
    print("  OK: a fully-synced fixture tree passes both version sync and "
          "count sync with zero mismatches")


def test_header_version_mismatch_detected():
    """A stale ufsecp_version.h (MAJOR/MINOR/PATCH) must fail check_version_sync,
    not silently pass because VERSION_STRING alone still matches."""
    with _synced_tree() as root:
        p = root / "include" / "ufsecp" / "ufsecp_version.h"
        p.write_text(_UFSECP_VERSION_H.replace("UFSECP_VERSION_PATCH   0",
                                                "UFSECP_VERSION_PATCH   9"))
        assert gate.check_version_sync(root) is False
    print("  OK: a stale ufsecp_version.h MAJOR/MINOR/PATCH triple is detected "
          "as a MISMATCH even though VERSION_STRING is untouched")


def test_missing_file_reported_as_missing_not_silently_skipped():
    """A required file that is simply absent must fail the gate (MISSING),
    not be treated as 'nothing to check here'."""
    with _synced_tree() as root:
        (root / "CITATION.cff").unlink()
        assert gate.check_version_sync(root) is False
    print("  OK: a missing CITATION.cff fails the gate as MISSING rather "
          "than being silently skipped")


def test_rpm_soversion_checked_against_major_not_full_version():
    """rpm spec soversion is a DIFFERENT expected value (MAJOR only) from
    every other row (full VERSION) -- a naive 'compare everything to
    canonical' pass would wrongly flag a correct soversion=4 as a mismatch
    against '4.5.0'. Separately, a genuinely wrong soversion integer must
    still be caught (proves the row is actually enforced, not just present
    with a permissive expected value)."""
    with _synced_tree() as root:
        # Correct soversion (matches MAJOR) must still pass.
        assert gate.check_version_sync(root) is True
        # Now corrupt soversion to a plainly wrong major integer -- must
        # fail. (The extractor's regex is digits-only, so writing the full
        # "4.5.0" string there would only ever re-extract "4" -- not a
        # meaningful corruption; a wrong integer is the real negative case.)
        p = root / "packaging" / "rpm" / "libufsecp.spec"
        p.write_text(_RPM_SPEC.replace(f"%global soversion {MAJOR}",
                                        "%global soversion 9"))
        assert gate.check_version_sync(root) is False
    print("  OK: rpm spec soversion is validated against MAJOR ("
          f"'{MAJOR}'), not the full version string -- a correct soversion "
          "passes and a genuinely wrong soversion integer is caught")


def test_package_swift_and_conanfile_and_vcpkg_and_zenodo_each_individually_checked():
    """REL9-001/002/003: conanfile.py, vcpkg.json, and .zenodo.json each have
    their own extractor -- a stale value in any ONE of them, with all others
    correct, must still fail the gate (proves each extractor is actually
    wired into the row list, not just defined and unused)."""
    mutations = {
        "Package.swift": (_PACKAGE_SWIFT, 'from: "4.5.0"', 'from: "3.0.0"'),
        "conanfile.py": (_CONANFILE, f'version = "{VERSION}"', 'version = "3.0.0"'),
        "vcpkg.json": (_VCPKG_JSON, VERSION, "3.0.0"),
        ".zenodo.json": (_ZENODO_JSON, VERSION, "3.0.0"),
    }
    for relname, (original, needle, replacement) in mutations.items():
        with _synced_tree() as root:
            (root / relname).write_text(original.replace(needle, replacement))
            assert gate.check_version_sync(root) is False, (
                f"expected a stale {relname} to fail check_version_sync")
    print("  OK: a stale Package.swift / conanfile.py / vcpkg.json / "
          ".zenodo.json individually (all else correct) each fail the gate")


def test_stale_exploit_poc_count_in_docs_detected():
    """check_count_sync must flag a doc that quotes a stale 'N exploit PoCs'
    total against the authoritative ALL_MODULES[] exploit_poc tally."""
    with _synced_tree() as root:
        (root / "docs" / "STALE.md").write_text(
            "This release ships 275 exploit PoCs across every backend.\n")
        assert gate.check_count_sync(root) is False
    print("  OK: a doc quoting a stale 3+ digit exploit PoC total (275, "
          "vs the fixture's authoritative count of 2) is detected as stale")


def test_small_incremental_exploit_count_not_falsely_flagged():
    """The exploit-count doc scan only matches 3+ digit numbers (large
    totals), so an incidental small number like '14 new exploit PoCs added'
    in a changelog-style sentence must NOT be misread as a stale total."""
    with _synced_tree() as root:
        (root / "docs" / "CHANGENOTE.md").write_text(
            "This patch adds 14 new exploit PoCs on top of the existing set.\n")
        assert gate.check_count_sync(root) is True
    print("  OK: a small incidental count ('14 new exploit PoCs') is not "
          "misclassified as a stale total (only 3+ digit totals are scanned)")


def test_stale_gpu_op_count_in_docs_detected():
    """check_count_sync must flag a doc claiming a GPU op-count that
    disagrees with ufsecp_gpu.h's own documented 'stable batch-op surface'
    prose count."""
    with _synced_tree() as root:
        (root / "docs" / "STALE_GPU.md").write_text(
            "UltrafastSecp256k1 ships a 40-op GPU C ABI surface.\n")
        assert gate.check_count_sync(root) is False
    print("  OK: a doc claiming a stale GPU op count (40-op, vs the "
          "fixture's authoritative 2) is detected as stale")


def test_version_only_and_counts_only_flags_are_mutually_exclusive_scopes():
    """--version-only must not run the count check (and vice versa) -- a
    stale doc count must not fail check_sync(version_only=True), and a
    stale header version must not fail check_sync(counts_only=True)."""
    with _synced_tree() as root:
        (root / "docs" / "STALE.md").write_text(
            "This release ships 275 exploit PoCs across every backend.\n")
        assert gate.check_sync(root, version_only=True) is True

    with _synced_tree() as root:
        p = root / "include" / "ufsecp" / "ufsecp_version.h"
        p.write_text(_UFSECP_VERSION_H.replace("UFSECP_VERSION_PATCH   0",
                                                "UFSECP_VERSION_PATCH   9"))
        assert gate.check_sync(root, counts_only=True) is True
    print("  OK: --version-only ignores count drift and --counts-only "
          "ignores version drift -- each flag scopes strictly to its own "
          "half of the gate")


def test_missing_version_txt_is_a_hard_error_not_a_vacuous_pass():
    """No VERSION.txt at all means there is no canonical value to compare
    against -- this must be a hard failure, never an empty/vacuous PASS."""
    with _synced_tree() as root:
        (root / "VERSION.txt").unlink()
        assert gate.check_version_sync(root) is False
    print("  OK: a missing VERSION.txt (no canonical source) fails the "
          "gate rather than vacuously passing")


TESTS = [
    test_fully_synced_tree_passes_both_checks,
    test_header_version_mismatch_detected,
    test_missing_file_reported_as_missing_not_silently_skipped,
    test_rpm_soversion_checked_against_major_not_full_version,
    test_package_swift_and_conanfile_and_vcpkg_and_zenodo_each_individually_checked,
    test_stale_exploit_poc_count_in_docs_detected,
    test_small_incremental_exploit_count_not_falsely_flagged,
    test_stale_gpu_op_count_in_docs_detected,
    test_version_only_and_counts_only_flags_are_mutually_exclusive_scopes,
    test_missing_version_txt_is_a_hard_error_not_a_vacuous_pass,
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
