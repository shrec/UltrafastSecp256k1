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
  docs/bench_unified_2026-05-11_gcc14_x86-64.json  (library-level benchmarks)
  docs/BITCOIN_CORE_BENCH_RESULTS.json              (Bitcoin Core integration)
  docs/canonical_numbers.json                       (sync source for prose)

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
import re
import sys
from pathlib import Path

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
        "Stale GCC 13.3 CT signing claim — GCC 14 CT ECDSA is +24% (1.24×), not +37%",
        None,
    ),
    (
        r"GCC 13\.3[^\n]{0,80}\+25%",
        "Stale GCC 13.3 CT signing claim — GCC 14 CT Schnorr is +9% (1.09×), not +25%",
        None,
    ),
    # ── Stale GCC 13/14 "slower" CT signing claim ────────────────────────────
    (
        r"GCC 13/14[^\n]{0,80}0\.8[2345]",
        "Stale CT signing: GCC 13/14 0.82-0.85× (slower) — GCC 14 is 1.09-1.24× (faster)",
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
    # ── Clang 19 archived CT signing ratios cited without qualifier ───────────
    # These ratios (1.33× CT ECDSA, 1.20× CT Schnorr) belong to the archived
    # Clang 19 bench run.  If cited without "Clang 19" or "[archived]" context
    # they mislead reviewers into thinking the GCC 14 baseline matches them.
    # The canonical GCC 14.2.0 ratios are 1.24× ECDSA and 1.09× Schnorr.
    # Note: context-aware banning (requires adjacent "Clang 19" or "archived")
    # is not yet implemented — these patterns are documented here so future
    # reviews can detect them, and a comment-only entry marks the intent.
    #
    # TODO(BENCH-005): implement context-aware banning once the pattern engine
    # supports negative lookahead or a "must NOT have adjacent" context_hint.
    # For now, exact banned strings are listed below and rely on the reviewer-doc
    # set being small enough that false positives are unlikely.
    (
        r"1\.33.*CT ECDSA",
        "Clang 19 archived CT ECDSA ratio (1.33×) cited without 'Clang 19' or '[archived]' qualifier — canonical GCC 14 ratio is 1.24×",
        # context_hint: only ban when NOT in a section already marked archived.
        # The current pattern engine matches on presence, so this fires when
        # 1.33× CT ECDSA appears anywhere; reviewers must check context manually.
        None,
    ),
    (
        r"1\.20.*CT Schnorr",
        "Clang 19 archived CT Schnorr ratio (1.20×) cited without 'Clang 19' or '[archived]' qualifier — canonical GCC 14 ratio is 1.09×",
        None,
    ),
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
]

# ---------------------------------------------------------------------------
# Required references — canonical artifact must be cited in primary docs
# ---------------------------------------------------------------------------
REQUIRED_REFS: list[tuple[str, str, str]] = [
    (
        "docs/BITCOIN_CORE_PR_DESCRIPTION.md",
        r"bench_unified_2026-05-11_gcc14_x86-64\.json",
        "PR description must cite the canonical bench_unified artifact",
    ),
    (
        "docs/BITCOIN_CORE_BACKEND_EVIDENCE.md",
        r"bench_unified_2026-05-11_gcc14_x86-64\.json",
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
    """Find the most recent bench_unified canonical JSON."""
    candidates = sorted(ROOT.glob(BENCH_UNIFIED_GLOB), reverse=True)
    return candidates[0] if candidates else None


def check_canonical_vs_bench_json(canon: dict, verbose: bool) -> list[str]:
    """Validate canonical_numbers.json values against bench_unified JSON artifact."""
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

    # ── CT signing GCC ratios ──────────────────────────────────────────────────
    ct_gcc = canon.get("ct_signing_gcc", {})
    bench_ct = bench.get("vs_libsecp_ct_signing", {})
    bench_ecdsa_ratio = bench_ct.get("ecdsa_sign_ratio")
    bench_schnorr_ratio = bench_ct.get("schnorr_sign_ratio")

    canon_max_x = ct_gcc.get("speedup_max_x")
    canon_min_x = ct_gcc.get("speedup_min_x")

    if bench_ecdsa_ratio is not None and canon_max_x is not None:
        drift = abs(bench_ecdsa_ratio - canon_max_x)
        if drift > 0.05:  # 5% tolerance
            violations.append(
                f"  CANONICAL DRIFT [ct_signing_gcc.speedup_max_x]\n"
                f"    canonical_numbers.json : {canon_max_x:.2f}×\n"
                f"    {bench_path.name}  : {bench_ecdsa_ratio:.2f}× (CT ECDSA)\n"
                f"    Drift: {drift:.3f}× — update canonical_numbers.json ct_signing_gcc values\n"
                f"    Fix : set speedup_max_x = {bench_ecdsa_ratio:.2f}"
            )

    if bench_schnorr_ratio is not None and canon_min_x is not None:
        drift = abs(bench_schnorr_ratio - canon_min_x)
        if drift > 0.05:
            violations.append(
                f"  CANONICAL DRIFT [ct_signing_gcc.speedup_min_x]\n"
                f"    canonical_numbers.json : {canon_min_x:.2f}×\n"
                f"    {bench_path.name}  : {bench_schnorr_ratio:.2f}× (CT Schnorr)\n"
                f"    Drift: {drift:.3f}× — update canonical_numbers.json ct_signing_gcc values\n"
                f"    Fix : set speedup_min_x = {bench_schnorr_ratio:.2f}"
            )

    # ── Compiler must match bench JSON ────────────────────────────────────────
    bench_compiler = bench.get("_meta", {}).get("compiler", "")
    canon_compiler = ct_gcc.get("compiler", "")
    if bench_compiler and canon_compiler and bench_compiler not in canon_compiler:
        violations.append(
            f"  CANONICAL DRIFT [ct_signing_gcc.compiler]\n"
            f"    canonical_numbers.json : {canon_compiler!r}\n"
            f"    {bench_path.name}  : {bench_compiler!r}\n"
            f"    Fix : set ct_signing_gcc.compiler = {bench_compiler!r}"
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
        if not re.search(pattern, text):
            violations.append(
                f"  MISSING REF [{rel_path}]\n"
                f"    Pattern : {pattern}\n"
                f"    Reason  : {reason}"
            )
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true", help="Show match context")
    args = parser.parse_args()

    canon = load_canonical_numbers()
    if canon is None:
        print(f"SKIP: {CANONICAL_NUMBERS} not found", file=sys.stderr)
        return 77

    all_violations: list[str] = []

    # ── Check banned patterns ─────────────────────────────────────────────────
    for rel in REVIEWER_DOCS:
        doc = ROOT / rel
        if not doc.exists():
            continue
        text = doc.read_text(encoding="utf-8")
        all_violations.extend(check_banned(text, rel, args.verbose))

    # ── Check required references ─────────────────────────────────────────────
    all_violations.extend(check_required_refs(args.verbose))

    # ── Validate canonical_numbers.json against bench_unified JSON ────────────
    all_violations.extend(check_canonical_vs_bench_json(canon, args.verbose))

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

    # ── Report ────────────────────────────────────────────────────────────────
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

    print(f"check_bench_doc_consistency: OK ({len(REVIEWER_DOCS)} docs, {len(BANNED)} banned patterns, {len(REQUIRED_REFS)} required refs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
