#!/usr/bin/env python3
"""
check_bench_doc_consistency.py
================================
Validates that reviewer-facing documentation is consistent with canonical
benchmark artifacts. Catches two classes of problems:

  1. BANNED PATTERNS — stale/wrong numeric claims that must never appear
     (e.g. old "+37% GCC CT signing", old cmake flag, old compiler version).

  2. REQUIRED REFERENCES — reviewer-facing docs must link to the canonical
     benchmark JSON artifact so claims are traceable.

Source of truth:
  docs/bench_unified_<YYYY-MM-DD>_gcc14_x86-64.json  (library-level benchmarks; latest wins)
  docs/BITCOIN_CORE_BENCH_RESULTS.json                (Bitcoin Core integration)
  docs/canonical_numbers.json                         (sync source for prose)

Exit codes:
  0  — all checks pass
  1  — drift detected (specific violations printed)
  77 — skip (canonical JSON not found — should not happen in CI)

Usage:
  python3 ci/check_bench_doc_consistency.py
  python3 ci/check_bench_doc_consistency.py --verbose
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import List

# Physical sanity floor (ns). Even the fastest measured primitive op on x86-64
# (a single field multiply) is well above this; a timing at/below it is a corrupt
# artifact — zero timing, truncated/concatenated column, or impossible throughput
# — not a real measurement. A perf claim sourced from such an artifact is not
# evidence (Bastion B8).
_MIN_PLAUSIBLE_NS = 0.1

ROOT = Path(__file__).resolve().parent.parent
CANONICAL_NUMBERS = ROOT / "docs" / "canonical_numbers.json"
BENCH_UNIFIED_GLOB = "docs/bench_unified_*.json"

# ---------------------------------------------------------------------------
# Documents to check for banned patterns
# ---------------------------------------------------------------------------
REVIEWER_DOCS = [
    "docs/BITCOIN_CORE_PR_DESCRIPTION.md",
    "docs/BITCOIN_CORE_BACKEND_EVIDENCE.md",
    "docs/CAAS_REVIEWER_QUICKSTART.md",
    "docs/BITCOIN_CORE_PR_BLOCKERS.md",
    "docs/BENCHMARKS.md",  # BENCH-001: scan primary benchmark doc for stale patterns
    "README.md",
    # CI-003: previously unscanned reviewer-facing docs — added 2026-05-12
    "docs/WHY_ULTRAFASTSECP256K1.md",
    "docs/BACKEND_ASSURANCE_MATRIX.md",
    "docs/ASSURANCE_LEDGER.md",
    "docs/BITCOIN_CORE_PR_BODY.md",
]

# ---------------------------------------------------------------------------
# Banned patterns — these must NEVER appear in reviewer docs.
# Each entry: (regex, human-readable reason, optional context_hint)
# Context_hint: if provided, the ban only applies when the surrounding
#   200 chars contain this hint (avoids false positives in quoted history).
# ---------------------------------------------------------------------------
BANNED: list[tuple[str, str, str | None]] = [
    # ── Wrong cmake flag ─────────────────────────────────────────────────────
    (
        r"-DUSE_ULTRAFAST_SECP256K1=ON",
        "Wrong cmake flag — must use -DSECP256K1_BACKEND=ultrafast",
        None,
    ),
    # ── Stale GCC 13.3 CT signing percentages ────────────────────────────────
    (
        r"GCC 13\.3[^\n]{0,80}\+37%",
        "Stale GCC 13.3 CT signing claim — canonical GCC 14 CT ECDSA is +33% (1.33×), not +37%",
        None,
    ),
    (
        r"GCC 13\.3[^\n]{0,80}\+25%",
        "Stale GCC 13.3 CT signing claim — canonical GCC 14 CT Schnorr is +26% (1.26×), not +25%",
        None,
    ),
    # ── Stale GCC 13/14 "slower" CT signing claim ────────────────────────────
    (
        r"GCC 13/14[^\n]{0,80}0\.8[2345]",
        "Stale CT signing: GCC 13/14 0.82-0.85× (slower) — canonical GCC 14 is 1.26-1.33× (faster)",
        None,
    ),
    (
        r"0\.82.0\.85.*slower",
        "Stale CT signing ratio: 0.82-0.85× slower applies to GCC 13 only, not GCC 14",
        None,
    ),
    # ── Stale compiler version in benchmark methodology ───────────────────────
    (
        r"Compiler: GCC 13\.3\.0",
        "Stale compiler version in methodology note — must say GCC 14.2.0",
        None,
    ),
    # ── Wrong nice value in methodology ──────────────────────────────────────
    (
        r"`nice -19`",
        "Wrong nice value — ci_local.sh protocol uses nice -20, not nice -19",
        None,
    ),
    # ── Stale no-LTO ConnectBlock deficit ────────────────────────────────────
    (
        r"2\.5.5\.4% slower",
        "Stale no-LTO ConnectBlock deficit — actual is ~1%, not 2.5-5.4%",
        None,
    ),
    (
        r"ConnectBlock[^\n]{0,120}1.2% slower",
        "Stale ConnectBlock wording — LTO build is +1-2% faster, not slower",
        # Only flag in contexts discussing LTO performance (not historical notes)
        "Release+LTO",
    ),
    # ── Stale taproot signing range ───────────────────────────────────────────
    (
        r"22.24% faster[^\n]{0,40}[Tt]aproot",
        "Stale taproot signing claim — actual is 11-36% (SignTransaction 11%, SignSchnorrMerkle 36%)",
        None,
    ),
    (
        r"[Tt]aproot[^\n]{0,60}22.24% faster",
        "Stale taproot signing claim — actual is 11-36% from bench_bitcoin",
        None,
    ),
    # ── Stale P2TR verify claim ───────────────────────────────────────────────
    (
        r"~18% faster[^\n]{0,40}P2TR",
        "Stale P2TR script-path verify claim — actual is ~11% (1.11×)",
        None,
    ),
    (
        r"P2TR[^\n]{0,60}~18% faster",
        "Stale P2TR script-path verify claim — actual is ~11% (1.11×)",
        None,
    ),
    # ── Stale SignSchnorrWithMerkleRoot percentage ────────────────────────────
    (
        r"23\.9%[^\n]{0,40}SignSchnorr",
        "Stale SignSchnorrWithMerkleRoot speedup — actual is ~36% (1.36×)",
        None,
    ),
    (
        r"SignSchnorr[^\n]{0,60}23\.9%",
        "Stale SignSchnorrWithMerkleRoot speedup — actual is ~36% (1.36×)",
        None,
    ),
    # ── CT signing ratio freshness — enforced DYNAMICALLY, not by string bans ──
    # The current canonical GCC 14.2.0 ratios are CT ECDSA 1.33× and CT Schnorr
    # 1.26× (docs/canonical_numbers.json → ct_signing_gcc; the archived Clang 19
    # run was 1.20–1.33×). We deliberately do NOT ban literal "1.33× CT ECDSA"
    # strings here: 1.33× IS the correct current canonical ECDSA ratio, so a static
    # ban would either be inert (as the old `1\.33.*CT ECDSA` pattern was — its
    # reason text even cited the superseded 1.24×/1.09×) or, worse, flag the right
    # value. CT-ratio freshness is enforced instead by check_canonical_vs_bench_json()
    # below, which derives the ratios from the bench JSON artifact and compares them
    # to canonical_numbers.json within a 3% tolerance — the authoritative check.
    # (P8-CAAS-001 / BENCH-005: removed the hardcoded 1.24×/1.09× ban entries that
    # gave a false sense of CT-ratio-freshness enforcement.)
    # ── FAST-path ratios cited in reviewer docs ───────────────────────────────
    # These ratios (2.45× ECDSA sign, 2.34× Schnorr sign) reflect variable-time
    # FAST-path vs libsecp256k1 CT.  They must not appear in reviewer docs because
    # production UltraFast uses CT signing, making the comparison non-equivalent.
    (
        r"2\.45[^\n]{0,60}ECDSA.?sign",
        "FAST-path ECDSA sign ratio (2.45×) — variable-time vs libsecp CT; not production-equivalent; must not appear in reviewer docs",
        None,
    ),
    (
        r"ECDSA.?sign[^\n]{0,60}2\.45",
        "FAST-path ECDSA sign ratio (2.45×) — variable-time vs libsecp CT; not production-equivalent; must not appear in reviewer docs",
        None,
    ),
    (
        r"2\.34[^\n]{0,60}[Ss]chnorr.?sign",
        "FAST-path Schnorr sign ratio (2.34×) — variable-time vs libsecp CT; not production-equivalent; must not appear in reviewer docs",
        None,
    ),
    (
        r"[Ss]chnorr.?sign[^\n]{0,60}2\.34",
        "FAST-path Schnorr sign ratio (2.34×) — variable-time vs libsecp CT; not production-equivalent; must not appear in reviewer docs",
        None,
    ),
    # BENCH-001 (v9): the original patterns above match only the order
    # "2.45× ECDSA sign" or "ECDSA sign … 2.45×".  Reviewer docs were leaking the
    # reversed order "ECDSA 2.45×" / "Schnorr 2.34×" which the original regex
    # missed.  Catch both word orders defensively.
    (
        r"ECDSA[^\n]{0,8}2\.4[5-9][×x]",
        "FAST-path ECDSA sign ratio (2.45×) in reversed word order — variable-time vs libsecp CT; not production-equivalent",
        None,
    ),
    (
        r"[Ss]chnorr[^\n]{0,8}2\.3[0-9][×x]",
        "FAST-path Schnorr sign ratio (2.34×) in reversed word order — variable-time vs libsecp CT; not production-equivalent",
        None,
    ),
    # ── Invalid pubkey_create VT vs CT comparison ─────────────────────────────
    # 2.2× pubkey_create is a variable-time (VT) vs CT comparison — not valid
    # because production pubkey_create uses CT.  Any "pubkey_create.*2.2×"
    # or "2.2.*pubkey_create" claim in reviewer docs is misleading.
    (
        r"pubkey.?create[^\n]{0,60}2\.2[0-9×x]",
        "Invalid pubkey_create VT-vs-CT ratio (2.2×) — production pubkey_create uses CT; cite CT-vs-CT ratio instead",
        None,
    ),
    (
        r"2\.2[0-9×x][^\n]{0,60}pubkey.?create",
        "Invalid pubkey_create VT-vs-CT ratio (2.2×) — production pubkey_create uses CT; cite CT-vs-CT ratio instead",
        None,
    ),
    # ── Unverified derived number must not appear in reviewer docs ────────────
    (
        r"claimed_saving_ns",
        "Use 'unverified_estimate_ns' instead of 'claimed_saving_ns' in canonical_numbers.json — "
        "this field contains a derived (not measured) estimate. Reviewer docs must not reference it.",
        None,
    ),
]

# ---------------------------------------------------------------------------
# Required references — canonical artifact must be cited in primary docs
# ---------------------------------------------------------------------------
REQUIRED_REFS: list[tuple[str, str, str]] = [
    (
        "docs/BITCOIN_CORE_PR_DESCRIPTION.md",
        r"bench_unified_2026-05-\d{2}_gcc14_x86-64\.json",
        "PR description must cite the canonical bench_unified artifact",
    ),
    (
        "docs/BITCOIN_CORE_BACKEND_EVIDENCE.md",
        r"bench_unified_2026-05-\d{2}_gcc14_x86-64\.json",
        "Evidence doc must cite the canonical bench_unified artifact",
    ),
    (
        "docs/BITCOIN_CORE_PR_DESCRIPTION.md",
        r"BITCOIN_CORE_BENCH_RESULTS\.json",
        "PR description must cite BITCOIN_CORE_BENCH_RESULTS.json",
    ),
]


def load_canonical_numbers() -> dict | None:
    if not CANONICAL_NUMBERS.exists():
        return None
    with open(CANONICAL_NUMBERS) as f:
        return json.load(f)


def find_bench_unified_json() -> Path | None:
    """Find canonical bench_unified JSON.

    Prefers the path declared in canonical_numbers.json._canonical_bench_artifact
    (avoids TASK-006 glob-by-date bypass: a newer but inconsistent file would
    otherwise silently shadow the validated canonical).  Falls back to glob only
    when the declared path is absent.
    """
    # Prefer path from canonical_numbers.json
    if CANONICAL_NUMBERS.exists():
        try:
            with open(CANONICAL_NUMBERS) as f:
                cn = json.load(f)
            declared = cn.get("_canonical_bench_artifact")
            if declared:
                p = ROOT / declared
                if p.exists():
                    return p
        except Exception:
            pass
    # Refuse to silently fall back to newest glob: the declared canonical is
    # authoritative. If it is absent, force an explicit update of
    # canonical_numbers.json._canonical_bench_artifact rather than auto-selecting
    # a potentially inconsistent newer file (e.g. one with _STALE ratio fields).
    print(
        "ERROR: canonical bench artifact declared in canonical_numbers.json not found.\n"
        "  Update canonical_numbers.json._canonical_bench_artifact to point to the\n"
        "  correct file, then re-run this check.",
        file=__import__("sys").stderr,
    )
    return None


def check_internal_bench_consistency(bench: dict, bench_name: str) -> list[str]:
    """TASK-001 gate: verify ratio fields in bench JSON are consistent with ns values.

    If a bench JSON contains ns measurements AND ratio fields, the ratios must be
    derivable from the ns values (±5% tolerance).  Prevents hand-crafted files with
    mismatched ratios from propagating to canonical_numbers.json.
    """
    violations: list[str] = []

    # Check ct_vs_libsecp_ratios section (2026-05-16 schema)
    ct_ratios = bench.get("ct_vs_libsecp_ratios", {})
    ct_ns = bench.get("ct_signing_ns", {})

    libsecp_ecdsa = ct_ratios.get("libsecp_ecdsa_sign_ns")
    libsecp_schnorr = ct_ratios.get("libsecp_schnorr_sign_ns")
    ultra_ecdsa_ns = ct_ns.get("ct_ecdsa_sign")
    ultra_schnorr_ns = ct_ns.get("ct_schnorr_sign")

    # Check for *_STALE fields (indicates previously detected inconsistency)
    if any(k.endswith("_STALE") for k in ct_ratios):
        violations.append(
            f"  INTERNAL INCONSISTENCY [{bench_name}]\n"
            f"    Ratio fields marked _STALE: this file is hand-crafted.\n"
            f"    Do not use as canonical artifact. Regenerate with bench_unified --json."
        )
        return violations

    stated_ecdsa_ratio = ct_ratios.get("ct_ecdsa_sign_ratio")
    stated_schnorr_ratio = ct_ratios.get("ct_schnorr_sign_ratio")

    if (stated_ecdsa_ratio is not None and libsecp_ecdsa is not None
            and ultra_ecdsa_ns is not None and ultra_ecdsa_ns > 0):
        derived = libsecp_ecdsa / ultra_ecdsa_ns
        drift = abs(stated_ecdsa_ratio - derived)
        if drift > 0.05:
            violations.append(
                f"  INTERNAL INCONSISTENCY [{bench_name}] ct_ecdsa_sign_ratio\n"
                f"    Stated ratio  : {stated_ecdsa_ratio:.3f}×\n"
                f"    Derived ratio : {derived:.3f}×  ({libsecp_ecdsa:.1f} / {ultra_ecdsa_ns:.1f} ns)\n"
                f"    Drift         : {drift:.3f}× > 0.05 tolerance\n"
                f"    File is hand-crafted or has a computation error. Regenerate."
            )

    if (stated_schnorr_ratio is not None and libsecp_schnorr is not None
            and ultra_schnorr_ns is not None and ultra_schnorr_ns > 0):
        derived = libsecp_schnorr / ultra_schnorr_ns
        drift = abs(stated_schnorr_ratio - derived)
        if drift > 0.05:
            violations.append(
                f"  INTERNAL INCONSISTENCY [{bench_name}] ct_schnorr_sign_ratio\n"
                f"    Stated ratio  : {stated_schnorr_ratio:.3f}×\n"
                f"    Derived ratio : {derived:.3f}×  ({libsecp_schnorr:.1f} / {ultra_schnorr_ns:.1f} ns)\n"
                f"    Drift         : {drift:.3f}× > 0.05 tolerance\n"
                f"    File is hand-crafted or has a computation error. Regenerate."
            )

    return violations


def _coerce_ns(raw: object) -> float | None:
    """Coerce a bench 'ns' value to a positive finite float, or None if invalid.

    Rejects booleans, non-numeric / concatenated strings ('123ns456'), NaN/inf,
    and non-positive timings. Corrupt rows are surfaced separately by
    check_bench_artifact_sanity(); here we simply refuse to derive a ratio from a
    bad value (which would otherwise crash on float() or divide by zero)."""
    if isinstance(raw, bool):
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val) or val <= 0:
        return None
    return val


def _find_result(results: list, section_substr: str, name_substr: str) -> float | None:
    """Return the ns value of the first result whose section contains *section_substr*
    and whose name contains *name_substr* (case-insensitive).  Returns None if not found
    or if the matched row carries an invalid ns. Only rows with an 'ns' key are
    considered (ratio-only rows are skipped).
    """
    sl = section_substr.lower()
    nl = name_substr.lower()
    for row in results:
        if not isinstance(row, dict):
            continue
        if "ns" not in row:
            continue
        if sl in row.get("section", "").lower() and nl in row.get("name", "").lower():
            return _coerce_ns(row["ns"])
    return None


def check_bench_artifact_sanity(bench: dict, name: str) -> list[str]:
    """Reject corrupt benchmark artifacts (Bastion B8).

    Scans the flat ``results`` array and flags any timing that is zero/negative,
    non-finite, non-numeric (e.g. a concatenated/truncated column like '123ns456'),
    or below the physical plausibility floor (impossible throughput). A perf claim
    cannot be evidence if its source artifact contains such values."""
    violations: list[str] = []
    results = bench.get("results")
    if results is None:
        return violations  # not a results-schema artifact; other checks cover it
    if not isinstance(results, list):
        violations.append(f"  CORRUPT BENCH [{name}]: 'results' is not a list")
        return violations

    bad: list[str] = []
    for row in results:
        if not isinstance(row, dict) or "ns" not in row:
            continue
        raw = row["ns"]
        label = f"{row.get('section', '?')}/{row.get('name', '?')}"
        if isinstance(raw, bool):
            bad.append(f"{label}: ns={raw!r} is boolean")
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            bad.append(f"{label}: ns={raw!r} is not numeric")
            continue
        if not math.isfinite(val):
            bad.append(f"{label}: ns={raw!r} is not finite")
        elif val <= 0:
            bad.append(f"{label}: ns={val} <= 0 (zero/negative timing)")
        elif val < _MIN_PLAUSIBLE_NS:
            bad.append(f"{label}: ns={val} below physical floor {_MIN_PLAUSIBLE_NS} (impossible throughput)")

    if bad:
        violations.append(
            f"  CORRUPT BENCH ARTIFACT [{name}]: {len(bad)} invalid timing(s): "
            + "; ".join(bad[:5]) + (" …" if len(bad) > 5 else "")
        )
    return violations


def check_canonical_vs_bench_json(canon: dict, verbose: bool) -> list[str]:
    """Validate canonical_numbers.json ratios against the bench_unified JSON artifact.

    The bench JSON uses a flat ``results`` array (schema introduced 2026-05-21).
    Previous code looked for a ``vs_libsecp_ct_signing`` top-level key that does not
    exist in this schema, so the ratio drift check silently skipped every time.
    This implementation derives ratios from the ``results`` array directly.

    CT ECDSA ratio  = libsecp ecdsa_sign ns  / Ultra ct::ecdsa_sign ns
    CT Schnorr ratio = libsecp schnorr_sign ns / Ultra ct::schnorr_sign ns

    Tolerance: 10% (wider than the 5% internal-consistency gate because the
    canonical_numbers values may be rounded to 2 decimal places).
    """
    violations: list[str] = []
    bench_path = find_bench_unified_json()
    if bench_path is None:
        return []  # No bench JSON found — skip (not a violation)

    try:
        with open(bench_path) as f:
            bench = json.load(f)
    except Exception as e:
        violations.append(f"  ERROR: Cannot parse {bench_path.name}: {e}")
        return violations

    # ── TASK-001: internal consistency check ──────────────────────────────────
    violations.extend(check_internal_bench_consistency(bench, bench_path.name))

    # ── Extract ns values from the flat results array ─────────────────────────
    results = bench.get("results", [])

    # Ultra CT signing rows — section "CT SIGNING (Ultra CT)"
    ultra_ct_ecdsa_ns = _find_result(results, "ct signing", "ct::ecdsa_sign")
    ultra_ct_schnorr_ns = _find_result(results, "ct signing", "ct::schnorr_sign")

    # libsecp256k1 rows — section "libsecp256k1 (bitcoin-core)"
    libsecp_ecdsa_ns = _find_result(results, "libsecp256k1", "ecdsa_sign")
    libsecp_schnorr_ns = _find_result(results, "libsecp256k1", "schnorr_sign")

    # ── CT signing GCC ratios ──────────────────────────────────────────────────
    ct_gcc = canon.get("ct_signing_gcc", {})
    canon_ecdsa_ratio = ct_gcc.get("ecdsa_ratio")
    canon_schnorr_ratio = ct_gcc.get("schnorr_ratio")

    # Tolerance: 3% — tightened from 10% to catch real drift early
    TOLERANCE = 0.03

    if ultra_ct_ecdsa_ns and libsecp_ecdsa_ns and ultra_ct_ecdsa_ns > 0:
        bench_ecdsa_ratio = libsecp_ecdsa_ns / ultra_ct_ecdsa_ns
        if canon_ecdsa_ratio is not None:
            drift = abs(bench_ecdsa_ratio - canon_ecdsa_ratio)
            if drift > TOLERANCE:
                violations.append(
                    f"  CANONICAL DRIFT [ct_signing_gcc.ecdsa_ratio]\n"
                    f"    canonical_numbers.json : {canon_ecdsa_ratio:.3f}×\n"
                    f"    {bench_path.name} : {bench_ecdsa_ratio:.3f}×"
                    f" ({libsecp_ecdsa_ns:.1f} / {ultra_ct_ecdsa_ns:.1f} ns)\n"
                    f"    Drift: {drift:.3f}× > {TOLERANCE:.2f} tolerance\n"
                    f"    Fix : set ct_signing_gcc.ecdsa_ratio = {bench_ecdsa_ratio:.2f}"
                )
        elif verbose:
            print(
                f"  [info] bench CT ECDSA ratio = {bench_ecdsa_ratio:.3f}× "
                f"(canonical_numbers.json has no ecdsa_ratio to compare)"
            )
    elif verbose:
        print(
            f"  [info] CT ECDSA ratio check skipped — could not find matching rows in {bench_path.name}.\n"
            f"    Looking for: section~'ct signing' name~'ct::ecdsa_sign' AND "
            f"section~'libsecp256k1' name~'ecdsa_sign'"
        )

    if ultra_ct_schnorr_ns and libsecp_schnorr_ns and ultra_ct_schnorr_ns > 0:
        bench_schnorr_ratio = libsecp_schnorr_ns / ultra_ct_schnorr_ns
        if canon_schnorr_ratio is not None:
            drift = abs(bench_schnorr_ratio - canon_schnorr_ratio)
            if drift > TOLERANCE:
                violations.append(
                    f"  CANONICAL DRIFT [ct_signing_gcc.schnorr_ratio]\n"
                    f"    canonical_numbers.json : {canon_schnorr_ratio:.3f}×\n"
                    f"    {bench_path.name} : {bench_schnorr_ratio:.3f}×"
                    f" ({libsecp_schnorr_ns:.1f} / {ultra_ct_schnorr_ns:.1f} ns)\n"
                    f"    Drift: {drift:.3f}× > {TOLERANCE:.2f} tolerance\n"
                    f"    Fix : set ct_signing_gcc.schnorr_ratio = {bench_schnorr_ratio:.2f}"
                )
        elif verbose:
            print(
                f"  [info] bench CT Schnorr ratio = {bench_schnorr_ratio:.3f}× "
                f"(canonical_numbers.json has no schnorr_ratio to compare)"
            )
    elif verbose:
        print(
            f"  [info] CT Schnorr ratio check skipped — could not find matching rows in {bench_path.name}.\n"
            f"    Looking for: section~'ct signing' name~'ct::schnorr_sign' AND "
            f"section~'libsecp256k1' name~'schnorr_sign'"
        )

    # ── Compiler must match bench JSON metadata ───────────────────────────────
    bench_compiler = bench.get("metadata", {}).get("compiler", "")
    canon_compiler = ct_gcc.get("compiler", "")
    if bench_compiler and canon_compiler and bench_compiler not in canon_compiler:
        violations.append(
            f"  CANONICAL DRIFT [ct_signing_gcc.compiler]\n"
            f"    canonical_numbers.json : {canon_compiler!r}\n"
            f"    {bench_path.name} metadata.compiler : {bench_compiler!r}\n"
            f"    Fix : set ct_signing_gcc.compiler = {bench_compiler!r}"
        )

    return violations


def check_field_mul_vs_bench_json(verbose: bool) -> list[str]:
    """T-02/BENCH-001 gate: verify BENCHMARKS.md summary table Field Mul value
    matches the bench_unified JSON within 5% tolerance.

    The summary table's first data row (x86-64) contains the authoritative
    Field Mul value as '<float> ns'.  Compares against bench JSON results array
    for section 'FIELD ARITHMETIC (Ultra)' / name 'field_mul'.

    This catches label transpositions like the 2026-05-25 bug where the table
    showed 10.5 ns (= field_negate's value) instead of 20.1 ns (= field_mul).
    """
    violations: list[str] = []

    bench_path = find_bench_unified_json()
    if bench_path is None:
        return []

    try:
        with open(bench_path) as f:
            bench = json.load(f)
    except Exception as e:
        violations.append(f"  ERROR: Cannot parse {bench_path.name}: {e}")
        return violations

    results = bench.get("results", [])
    bench_ns = _find_result(results, "field arithmetic", "field_mul")
    if bench_ns is None:
        if verbose:
            print(
                "  [info] Field Mul check skipped — 'FIELD ARITHMETIC' field_mul row "
                f"not found in {bench_path.name}"
            )
        return violations

    benchmarks_md = ROOT / "docs" / "BENCHMARKS.md"
    if not benchmarks_md.exists():
        if verbose:
            print("  [info] Field Mul check skipped — docs/BENCHMARKS.md not found")
        return violations

    text = benchmarks_md.read_text(encoding="utf-8")

    # Locate the summary table header row containing "Field Mul"
    header_match = re.search(r"\|[^\n]*Field Mul[^\n]*\|", text)
    if not header_match:
        if verbose:
            print("  [info] Field Mul check skipped — 'Field Mul' column not found in BENCHMARKS.md")
        return violations

    header_line = header_match.group()
    cols = [c.strip() for c in header_line.split("|")]
    field_mul_col_idx: int | None = next(
        (i for i, c in enumerate(cols) if "Field Mul" in c), None
    )
    if field_mul_col_idx is None:
        return violations  # should not happen given the search above

    # Find the primary x86-64 data row after the header
    x86_match = re.search(r"\|\s*x86-64 \(i5-[^\n]+\|", text, flags=re.IGNORECASE)
    if not x86_match:
        if verbose:
            print("  [info] Field Mul check skipped — x86-64 data row not found in BENCHMARKS.md")
        return violations

    data_cols = [c.strip() for c in x86_match.group().split("|")]
    if field_mul_col_idx >= len(data_cols):
        violations.append(
            f"  ERROR [docs/BENCHMARKS.md]\n"
            f"    Field Mul column index {field_mul_col_idx} out of range in x86-64 row"
        )
        return violations

    cell = data_cols[field_mul_col_idx]
    if cell in ("—", "-", ""):
        if verbose:
            print("  [info] Field Mul check skipped — cell value is '—' (not yet measured)")
        return violations

    ns_match = re.search(r"([\d.]+)\s*ns", cell)
    if not ns_match:
        violations.append(
            f"  ERROR [docs/BENCHMARKS.md summary table]\n"
            f"    Field Mul cell value {cell!r} cannot be parsed as '<float> ns'"
        )
        return violations

    doc_ns = float(ns_match.group(1))
    drift_pct = abs(doc_ns - bench_ns) / bench_ns
    TOLERANCE = 0.05

    if drift_pct > TOLERANCE:
        violations.append(
            f"  SUMMARY TABLE DRIFT [docs/BENCHMARKS.md Field Mul]\n"
            f"    Table value   : {doc_ns:.1f} ns\n"
            f"    Bench JSON    : {bench_ns:.2f} ns  ({bench_path.name})\n"
            f"    Drift         : {drift_pct * 100:.1f}% > {TOLERANCE * 100:.0f}% tolerance\n"
            f"    Fix : update BENCHMARKS.md summary table Field Mul to {bench_ns:.1f} ns"
        )
    elif verbose:
        print(
            f"  [info] Field Mul: doc={doc_ns:.1f} ns, bench={bench_ns:.2f} ns "
            f"(drift {drift_pct * 100:.1f}% — OK)"
        )

    return violations


def build_archive_ranges(text: str) -> list[tuple[int, int]]:
    """Return a list of (start, end) character ranges that are inside
    BENCH-ARCHIVE-START / BENCH-ARCHIVE-END blocks.  Both HTML comment
    syntax (<!-- BENCH-ARCHIVE-START -->) and plain comment syntax
    (# BENCH-ARCHIVE-START) are recognised.

    Lines inside these blocks are excluded from banned-pattern checking so
    that archived tables (e.g. old Clang-19 results) do not cause false
    positives.
    """
    # Match both <!-- BENCH-ARCHIVE-START --> and # BENCH-ARCHIVE-START
    start_pat = re.compile(
        r"(?:<!--\s*BENCH-ARCHIVE-START\s*-->|#\s*BENCH-ARCHIVE-START\b)"
    )
    end_pat = re.compile(
        r"(?:<!--\s*BENCH-ARCHIVE-END\s*-->|#\s*BENCH-ARCHIVE-END\b)"
    )
    ranges: list[tuple[int, int]] = []
    pos = 0
    while pos < len(text):
        sm = start_pat.search(text, pos)
        if sm is None:
            break
        em = end_pat.search(text, sm.end())
        if em is None:
            # Unclosed archive block — treat to end-of-file
            ranges.append((sm.start(), len(text)))
            break
        ranges.append((sm.start(), em.end()))
        pos = em.end()
    return ranges


def in_archive_block(pos: int, ranges: list[tuple[int, int]]) -> bool:
    """Return True if character position *pos* falls inside any archive range."""
    for start, end in ranges:
        if start <= pos < end:
            return True
    return False


def check_banned(text: str, path: str, verbose: bool) -> list[str]:
    violations: list[str] = []
    archive_ranges = build_archive_ranges(text)
    for pattern, reason, context_hint in BANNED:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            # Skip matches that fall inside a BENCH-ARCHIVE-START/END block
            if in_archive_block(m.start(), archive_ranges):
                continue
            start = max(0, m.start() - 100)
            end = min(len(text), m.end() + 100)
            ctx = text[start:end]
            # Skip if context_hint required but not present in surrounding text
            if context_hint and context_hint.lower() not in ctx.lower():
                continue
            line_no = text[: m.start()].count("\n") + 1
            violations.append(
                f"  BANNED [{path}:{line_no}]\n"
                f"    Pattern : {pattern}\n"
                f"    Matched : {m.group()!r}\n"
                f"    Reason  : {reason}"
            )
            if verbose:
                snippet = ctx.replace("\n", " ").strip()[:120]
                violations[-1] += f"\n    Context : ...{snippet}..."
    return violations


def check_required_refs(verbose: bool) -> list[str]:
    violations: list[str] = []
    for rel_path, pattern, reason in REQUIRED_REFS:
        doc = ROOT / rel_path
        if not doc.exists():
            continue
        text = doc.read_text(encoding="utf-8")
        m = re.search(pattern, text)
        if not m:
            violations.append(
                f"  MISSING REF [{rel_path}]\n"
                f"    Pattern : {pattern}\n"
                f"    Reason  : {reason}"
            )
        else:
            # Verify that any referenced JSON artifact file actually exists on disk.
            # Extract filenames matching the bench_unified_*.json pattern from all matches.
            for match in re.finditer(pattern, text):
                matched_text = match.group()
                # Only perform file-existence check for bench_unified artifact refs
                if matched_text.startswith("bench_unified_") and matched_text.endswith(".json"):
                    artifact_path = ROOT / "docs" / matched_text
                    if not artifact_path.exists():
                        violations.append(
                            f"  ARTIFACT NOT FOUND [{rel_path}]\n"
                            f"    Referenced : docs/{matched_text}\n"
                            f"    Reason  : {reason}\n"
                            f"    Fix     : ensure docs/{matched_text} exists or update the reference"
                        )
    return violations


def build_reviewer_docs_list() -> List[str]:
    """Return the merged list of reviewer docs: hardcoded REVIEWER_DOCS plus any
    docs/*.md and README.md discovered automatically at runtime.

    Auto-discovery ensures new reviewer-facing docs added to docs/ are scanned
    for banned patterns without requiring a manual update to REVIEWER_DOCS.
    The hardcoded list takes precedence (de-duplication by relative path).
    """
    known: set[str] = set(REVIEWER_DOCS)
    discovered: List[str] = list(REVIEWER_DOCS)

    # Auto-discover all *.md files in docs/
    for md_path in sorted((ROOT / "docs").glob("*.md")):
        rel = str(md_path.relative_to(ROOT))
        if rel not in known:
            known.add(rel)
            discovered.append(rel)

    # Auto-discover README.md at repo root
    root_readme = ROOT / "README.md"
    if root_readme.exists():
        rel = "README.md"
        if rel not in known:
            known.add(rel)
            discovered.append(rel)

    return discovered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true", help="Show match context")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("-o", dest="out_file", help="Write JSON report to file")
    args = parser.parse_args()

    canon = load_canonical_numbers()
    if canon is None:
        print(f"SKIP: {CANONICAL_NUMBERS} not found", file=sys.stderr)
        return 77

    all_violations: list[str] = []

    # ── Check banned patterns ─────────────────────────────────────────────────
    effective_docs = build_reviewer_docs_list()
    for rel in effective_docs:
        doc = ROOT / rel
        if not doc.exists():
            continue
        text = doc.read_text(encoding="utf-8")
        all_violations.extend(check_banned(text, rel, args.verbose))

    # ── Check required references ─────────────────────────────────────────────
    all_violations.extend(check_required_refs(args.verbose))

    # ── Validate canonical_numbers.json against bench_unified JSON ────────────
    all_violations.extend(check_canonical_vs_bench_json(canon, args.verbose))

    # ── T-02: Validate BENCHMARKS.md summary table Field Mul vs bench JSON ────
    all_violations.extend(check_field_mul_vs_bench_json(args.verbose))

    # ── Sanity: canonical GCC CT ratios must be >= 1.0 (GCC 14 is faster) ────
    ct_gcc = canon.get("ct_signing_gcc", {})
    max_x = ct_gcc.get("speedup_max_x", 0)
    if max_x < 1.0:
        all_violations.append(
            f"  STALE CANONICAL [docs/canonical_numbers.json]\n"
            f"    ct_signing_gcc.speedup_max_x = {max_x} (< 1.0)\n"
            f"    GCC 14.2.0 CT signing is faster than libsecp (1.09-1.24×).\n"
            f"    Run: python3 ci/sync_canonical_numbers.py after updating this file."
        )

    bc = canon.get("bitcoin_core_compat", {})
    if bc.get("compiler", "").startswith("Clang") and bc.get("pass", 0) < 700:
        all_violations.append(
            f"  STALE CANONICAL [docs/canonical_numbers.json]\n"
            f"    bitcoin_core_compat: compiler={bc.get('compiler')!r}, pass={bc.get('pass')}\n"
            f"    Primary test run should be GCC 14.2.0 with 749/749.\n"
            f"    Update canonical_numbers.json bitcoin_core_compat fields."
        )

    # ── Scan all bench_unified_*.json for _STALE or _DERIVED keys ─────────────
    # Such keys indicate a corrupted or hand-crafted artifact that must not be
    # cited as a canonical benchmark source.
    for bench_path in sorted((ROOT / "docs").glob("bench_unified_*.json")):
        try:
            bench_data = json.loads(bench_path.read_text())
        except Exception as exc:
            # Fail closed: a malformed canonical bench artifact must be a
            # violation, not silently ignored (B8).
            all_violations.append(
                f"  CORRUPT BENCH [{bench_path.name}]: not valid JSON ({exc})"
            )
            continue
        if not isinstance(bench_data, dict):
            all_violations.append(
                f"  CORRUPT BENCH [{bench_path.name}]: root is not a JSON object"
            )
            continue
        stale_keys: list[str] = []
        for section in bench_data.values():
            if isinstance(section, dict):
                stale_keys.extend(
                    k for k in section
                    if "_STALE" in str(k) or "_DERIVED" in str(k)
                )
        if stale_keys:
            all_violations.append(
                f"  STALE ARTIFACT DETECTED [{bench_path.name}]: contains keys "
                f"{stale_keys}. "
                f"Regenerate with bench_unified --json before citing."
            )
        # B8: reject zero/negative/non-finite/impossible timings in the artifact.
        all_violations.extend(check_bench_artifact_sanity(bench_data, bench_path.name))

    # ── B17: benchmark target-context taxonomy + claim scope ──────────────────
    # Every canonical benchmark artifact must declare a valid target_context and
    # claim_scope so a microbenchmark is not mistaken for node throughput and a
    # GPU public-data / fallback benchmark is not a native-hardware perf claim.
    context_report = None
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from check_bench_target_context import load_and_evaluate as _ctx_eval
        context_report, _ctx_code = _ctx_eval()
        if context_report.get("error"):
            all_violations.append(f"  BENCH CONTEXT [schema]: {context_report['error']}")
        else:
            for rid in context_report["missing_context_rows"]:
                all_violations.append(f"  BENCH CONTEXT [{rid}]: missing target_context (B17)")
            for rid in context_report["invalid_context_rows"]:
                all_violations.append(f"  BENCH CONTEXT [{rid}]: target_context not in schema enum (B17)")
            for r in context_report["rows"]:
                for p in r.get("problems", []):
                    if p.startswith("scope_mismatch"):
                        all_violations.append(f"  BENCH CONTEXT [{r['id']}]: {p} (B17)")
    except Exception as exc:
        all_violations.append(f"  BENCH CONTEXT: cannot run target-context gate: {exc}")

    # ── Report ────────────────────────────────────────────────────────────────
    ok = not all_violations
    if args.json or args.out_file:
        report = {
            "check": "check_bench_doc_consistency",
            "overall_pass": ok,
            "violations": all_violations,
            "docs_scanned": len(effective_docs),
            "banned_patterns": len(BANNED),
            "required_refs": len(REQUIRED_REFS),
            "bench_target_context": context_report,
        }
        rendered = json.dumps(report, indent=2)
        if args.out_file:
            Path(args.out_file).write_text(rendered, encoding="utf-8")
        if args.json:
            print(rendered)
        return 0 if ok else 1

    if all_violations:
        print(f"check_bench_doc_consistency: {len(all_violations)} violation(s) found\n")
        for v in all_violations:
            print(v)
        print(
            f"\nTo fix: update docs/canonical_numbers.json, then run:\n"
            f"  python3 ci/sync_canonical_numbers.py\n"
            f"  python3 ci/sync_docs_from_canonical.py\n"
            f"Commit both the JSON and updated docs together."
        )
        return 1

    print(f"check_bench_doc_consistency: OK ({len(effective_docs)} docs, {len(BANNED)} banned patterns, {len(REQUIRED_REFS)} required refs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
