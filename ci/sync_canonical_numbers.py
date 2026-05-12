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
    # Matches e.g. "693/693 Bitcoin Core tests" or "693/693 tests pass"
    pat = r'\b\d{3}/\d{3}\b(?= Bitcoin Core| test)'
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

    # ── ConnectBlock without-LTO wording ─────────────────────────────────────
    # Replaces "Without LTO: ~N.N% slower ... LTO eliminates this [entirely]." with
    # canonical wording. Matches the full known phrase to avoid partial replacements.
    nolto = c.get("connectblock", {}).get("wording_no_lto", "")
    if nolto:
        # Match the specific stale phrase from .text section to end of that sentence.
        # Anchored tightly so it doesn't consume text it shouldn't.
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
