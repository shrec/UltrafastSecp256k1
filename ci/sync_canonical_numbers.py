#!/usr/bin/env python3
"""
sync_canonical_numbers.py — Propagate canonical_numbers.json values into all docs.

Workflow:
  1. Run benchmarks / make measurements.
  2. Update docs/canonical_numbers.json with the new numbers.
  3. Run this script — it rewrites all referenced docs atomically.
  4. Commit docs/canonical_numbers.json + affected docs together.

Agents: NEVER manually edit benchmark numbers in README.md, PR_DESCRIPTION.md,
BACKEND_EVIDENCE.md, etc. Update canonical_numbers.json, then run this script.

Usage:
    python3 ci/sync_canonical_numbers.py [--dry-run] [--verbose]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
CANONICAL = BASE / "docs" / "canonical_numbers.json"


def load_canonical() -> dict:
    with open(CANONICAL) as f:
        return json.load(f)


def _sub(pattern: str, replacement: str, text: str) -> tuple[str, int]:
    new_text, n = re.subn(pattern, replacement, text)
    return new_text, n


def sync_file(path: Path, c: dict, dry_run: bool, verbose: bool) -> int:
    """Apply all canonical substitutions to one file. Returns number of replacements."""
    if not path.exists():
        return 0
    text = path.read_text(encoding="utf-8")
    original = text
    total = 0

    bc = c["bitcoin_core_compat"]
    cb = c["connectblock"]
    ts = c["taproot_signing"]
    tv = c["taproot_verify"]
    sm = c["schnorr_sign_merkle"]
    ct_cl = c["ct_signing_clang19"]
    ct_gc = c["ct_signing_gcc"]

    # ── Bitcoin Core test pass count ─────────────────────────────────────────
    # Matches e.g. "693/693 Bitcoin Core tests", "693/693 tests pass",
    # "693/693 `make check` tests pass" in evidence tables
    pat = r'\b\d{3}/\d{3}\b(?= Bitcoin Core| test| `make check`)'
    repl = f"{bc['pass']}/{bc['total']}"
    text, n = _sub(pat, repl, text); total += n

    # ── ConnectBlock regression ───────────────────────────────────────────────
    # When canonical is ≈0% (regression_max_pct == 0): LTO claim is already
    # correct in docs. "without LTO" percentages are accurate context — do not
    # overwrite. When canonical is a range > 0: replace out-of-range claims.
    if cb["regression_max_pct"] > 0:
        pat = r'ConnectBlock[^\n]{0,120}?[−\-](\d+\.\d+)%'
        def replace_cb(m: re.Match) -> str:
            val = float(m.group(1))
            if abs(val - cb["regression_min_pct"]) > 0.2 and abs(val - cb["regression_max_pct"]) > 0.2:
                return m.group(0).replace(m.group(1), f"{cb['regression_min_pct']}–{cb['regression_max_pct']}")
            return m.group(0)
        new_text = re.sub(pat, replace_cb, text)
        if new_text != text:
            total += 1
            text = new_text

    # ── Taproot signing speedup ───────────────────────────────────────────────
    pat = r'Taproot[^\n]{0,80}?(\d{2})–(\d{2})% faster'
    repl = f"Taproot key-path signing is {ts['speedup_min_pct']}–{ts['speedup_max_pct']}% faster"
    text, n = _sub(pat, repl, text); total += n

    # ── Taproot verify speedup ────────────────────────────────────────────────
    pat = r'P2TR[^\n]{0,80}?~?(\d{2})% faster'
    repl = f"P2TR script-path verification is ~{tv['speedup_pct']}% faster"
    text, n = _sub(pat, repl, text); total += n

    # ── SignSchnorrWithMerkleRoot speedup ─────────────────────────────────────
    pat = r'SignSchnorrWithMerkleRoot[^\n]{0,80}?(\d{2,4}\.\d)%'
    repl_pct = f"{sm['canonical_pct']}"
    def replace_ssmr(m: re.Match) -> str:
        if abs(float(m.group(1)) - sm['canonical_pct']) > 0.5:
            return m.group(0).replace(m.group(1), repl_pct)
        return m.group(0)
    new_text = re.sub(pat, replace_ssmr, text)
    if new_text != text:
        total += 1
        text = new_text

    # ── CT signing performance (Clang 19) ────────────────────────────────────
    pat = r'Clang 19[^\n]{0,60}?(\d+\.\d{2})–(\d+\.\d{2})×'
    repl = f"Clang 19: {ct_cl['speedup_min_x']:.2f}–{ct_cl['speedup_max_x']:.2f}×"
    text, n = _sub(pat, repl, text); total += n

    # ── CT signing performance (GCC) ─────────────────────────────────────────
    pat = r'GCC 1[34][^\n]{0,60}?(\d+\.\d{2})–(\d+\.\d{2})×'
    repl = f"GCC 13/14: {ct_gc['speedup_min_x']:.2f}–{ct_gc['speedup_max_x']:.2f}×"
    text, n = _sub(pat, repl, text); total += n

    # ── ConnectBlock LTO range in QUICKSTART / PR_BODY narrative ─────────────
    # Matches inline "+N.N% to +N.N% faster ... confirmed (err% ...)"
    # Used in CAAS_REVIEWER_QUICKSTART.md and similar reviewer-facing summaries.
    cb_lto_range = cb.get("wording_lto_range", "")
    if cb_lto_range:
        pat = (r'\*\*\+[\d.]+%\s*to\s*\+[\d.]+%\s*faster\*\*[^\n]*?'
               r'confirmed\s*\([^\n]*?\)\.')
        if re.search(pat, text):
            text = re.sub(pat, cb_lto_range, text)
            total += 1

    # ── ConnectBlock LTO table rows in PR_BODY ────────────────────────────────
    # Replaces individual benchmark table rows with canonical ms values + delta.
    # Format: | ConnectBlock{Key} | {libsecp} ms/blk | {ultra} ms/blk | **+{pct}%** |
    lto_rows = cb.get("lto_rows", {})
    for key, row in lto_rows.items():
        pat = (rf'\|\s*ConnectBlock{key}\s*\|'
               rf'\s*[\d.]+\s*ms/blk\s*\|\s*[\d.]+\s*ms/blk\s*\|\s*\*\*\+[\d.]+%\*\*\s*\|')
        repl = (f'| ConnectBlock{key} | {row["libsecp_ms"]} ms/blk'
                f' | {row["ultra_ms"]} ms/blk | **+{row["delta_pct"]}%** |')
        new_text, n = _sub(pat, repl, text)
        if n:
            text = new_text
            total += n

    # ── Performance table header date in PR_BODY ──────────────────────────────
    # Updates "bench_bitcoin, Release+LTO, GCC 14.2, i5-14400F, YYYY-MM-DD"
    bench_note = cb.get("wording_bench_note", "")
    if bench_note:
        pat = r'### Performance \(bench_bitcoin,[^\n]+\)'
        repl = f'### Performance ({bench_note})'
        text, n = _sub(pat, repl, text); total += n

    # ── ConnectBlock turbo-lock note in PR_BODY Known Gaps ────────────────────
    # Replaces stale "no hard turbo lock (sudo unavailable during run)"
    pat = (r'ConnectBlock benchmark uses[^\n]*?no hard turbo lock[^\n]*')
    repl = ('ConnectBlock benchmark uses governor=performance, taskset -c 0,'
            ' hard turbo lock (intel_pstate/no_turbo=1, sudo pinned, 2026-05-12).')
    text, n = _sub(pat, repl, text); total += n

    # ── QUICKSTART / PR_BODY: stale "pending re-benchmark" paragraph ─────────
    # Replaces the old "PERF-002 removed ... A controlled re-benchmark is pending"
    # blockquote section (with or without leading "> " markers) with current data.
    nolto_wording = cb.get("wording_no_lto", "")
    if nolto_wording:
        # Pattern accounts for optional blockquote "> " or ">   " line prefixes.
        pat = (r'(?:>[ \t]*)?\s*PERF-002 removed a redundant on-curve check from.*?'
               r'A controlled re-benchmark is pending[^\n]*\n'
               r'(?:>[ \t]*)?[^\n]*`results_nolto`[^\n]*\n'
               r'(?:>[ \t]*)?[^\n]*not be cited as the current no-LTO result\.')
        new_text = re.sub(pat, nolto_wording, text, flags=re.DOTALL)
        if new_text != text:
            text = new_text
            total += 1

    # ── PR_BODY Known Gaps: "~1% slower" → precise range ─────────────────────
    # Updates the bullet point about no-LTO ConnectBlock deficit.
    nolto_min = abs(cb.get("nolto_rows", {}).get("AllEcdsa", {}).get("delta_pct", -0.5))
    nolto_max = abs(cb.get("nolto_rows", {}).get("AllSchnorr", {}).get("delta_pct", -1.0))
    if nolto_min and nolto_max:
        pat = (r'Without LTO: ConnectBlock ~\d+[\.,\-–\d]*%\s+slower'
               r'[^\n]*instruction-cache pressure[^\n]*')
        repl = (f'Without LTO: ConnectBlock ~{nolto_min}–{nolto_max}% slower due to'
                f' instruction-cache pressure from larger code footprint'
                f' (~1.3 MB vs libsecp ~400 KB); gap closes with LTO')
        text, n = _sub(pat, repl, text); total += n

    # ── ConnectBlock without-LTO wording (legacy full-sentence pattern) ───────
    nolto = cb.get("wording_no_lto", "")
    if nolto:
        pat = (r'Without LTO:\s*~\d+[\.,]\d+%\s*slower due to instruction-cache pressure'
               r'[^*\n|]*?LTO eliminates this(?:\s+entirely)?(?:\.|(?=\s))')
        if re.search(pat, text):
            text = re.sub(pat, nolto, text)
            total += 1

    # ── Fuzz corpus Summary Table cell ───────────────────────────────────────
    # Replaces "NNK+ fuzz corpus" (or similar stale count) with canonical cell text.
    fc = c.get("fuzz_corpus", {}).get("summary_table_cell", "")
    if fc:
        pat = r'Wycheproof, fault injection, [\w\s,K+]+ fuzz corpus'
        text, n = _sub(pat, fc, text); total += n

    if text == original:
        return 0

    if dry_run:
        if verbose:
            print(f"  [DRY-RUN] would update {path.relative_to(BASE)} ({total} replacements)")
        return total

    path.write_text(text, encoding="utf-8")
    if verbose:
        print(f"  [UPDATED] {path.relative_to(BASE)} ({total} replacements)")
    return total


# Docs that reference benchmark numbers
TARGET_DOCS = [
    "README.md",
    "docs/BITCOIN_CORE_PR_DESCRIPTION.md",
    "docs/BITCOIN_CORE_BACKEND_EVIDENCE.md",
    "docs/BENCHMARKS.md",
    "docs/THREAD_SAFETY.md",
    "docs/BITCOIN_CORE_PR_BLOCKERS.md",
    "docs/WHY_ULTRAFASTSECP256K1.md",
    # Reviewer-facing docs that contain inline benchmark claims
    "docs/CAAS_REVIEWER_QUICKSTART.md",
    "docs/BITCOIN_CORE_PR_BODY.md",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    c = load_canonical()
    print(f"[sync_canonical_numbers] schema={c['_schema']}")

    total_replaced = 0
    for rel in TARGET_DOCS:
        path = BASE / rel
        n = sync_file(path, c, dry_run=args.dry_run, verbose=True)
        total_replaced += n

    verb = "Would replace" if args.dry_run else "Replaced"
    print(f"\n{verb} {total_replaced} value(s) across {len(TARGET_DOCS)} docs.")
    if args.dry_run and total_replaced > 0:
        print("Run: python3 scripts/sync_all_docs.py")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
