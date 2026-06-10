#!/usr/bin/env python3
"""
test_check_ct_branches.py — unit test for the static CT-branch gate.

Proves the gate actually CATCHES the secret-dependent conditional-reduction bug class
(not merely that it passes on today's tree), and that it does NOT fire on the
constant-time cmov idiom, structural/public branches, or documented allowlist entries.

Run: python3 ci/test_check_ct_branches.py   (exit 0 = all assertions pass)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import check_ct_branches as g


def _scan_src(rel: str, src: str) -> list[tuple[int, str]]:
    """Scan a synthetic source string as if it were CT file `rel`."""
    with tempfile.NamedTemporaryFile("w", suffix=".cuh", delete=False) as f:
        f.write(src)
        tmp = Path(f.name)
    try:
        return g.scan_file(tmp, rel)
    finally:
        tmp.unlink()


def main() -> int:
    fails = 0

    def check(name: str, cond: bool) -> None:
        nonlocal fails
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")
        if not cond:
            fails += 1

    # 1. The exact leak pattern MUST be flagged: data-dependent conditional subtract.
    leak = "  if (borrow == 0) { r->limbs[0] = t0; r->limbs[1] = t1; }\n"
    check("flags conditional-reduction on borrow", len(_scan_src("secp256k1.cuh", leak)) == 1)

    # 2. scalar_ge / >=ORDER style reduction branch MUST be flagged.
    ge = "  if (scalar_ge(&r, ORDER)) scalar_sub(&r, &r, ORDER);\n"
    check("flags scalar_ge reduction branch", len(_scan_src("secp256k1.cuh", ge)) == 1)
    ge2 = "  if (r->limbs[3] >= MODULUS) { sub(); }\n"
    check("flags >=MODULUS reduction branch", len(_scan_src("secp256k1.cuh", ge2)) == 1)

    # 3. The constant-time cmov replacement MUST NOT be flagged (mask math, no control flow).
    cmov = (
        "  uint64_t need = (uint64_t)(borrow == 0ULL);\n"
        "  const uint64_t msk = (uint64_t)0 - need;\n"
        "  r->limbs[0] = (t0 & msk) | (r0 & ~msk);\n"
    )
    check("does NOT flag cmov mask idiom", len(_scan_src("secp256k1.cuh", cmov)) == 0)

    # 4. The ternary mask idiom MUST NOT be flagged.
    tern = "  uint64_t need = (diff < borrow) ? 1ULL : 0ULL;\n"
    check("does NOT flag ternary mask", len(_scan_src("secp256k1.cuh", tern)) == 0)

    # 5. Structural/public branches (loop/index/infinity guards) MUST NOT be flagged.
    benign = (
        "  for (int i = 0; i < 4; i++) {}\n"
        "  if (idx >= count) return;\n"
        "  if (point->infinity) return;\n"
    )
    check("does NOT flag structural/public branches", len(_scan_src("secp256k1.cuh", benign)) == 0)

    # 6. Comment lines mentioning borrow/carry MUST NOT be flagged.
    comment = "  // if (borrow == 0) keep r; else subtract MODULUS — now done branchless\n"
    check("does NOT flag comments", len(_scan_src("secp256k1.cuh", comment)) == 0)

    # 7. Allowlisted wNAF verify-path branch MUST NOT be flagged in secp256k1.cuh...
    wnaf = "  if (!carry) break;\n"
    check("allowlists wNAF verify branch in secp256k1.cuh",
          len(_scan_src("secp256k1.cuh", wnaf)) == 0)
    # 7b. ...but the SAME 'borrow' branch in a NON-allowlisted file IS flagged.
    check("allowlist is file-scoped (still flags ct_sign.cuh)",
          len(_scan_src("ct/ct_sign.cuh", leak)) == 1)

    # 7c. CT-P2-02 reduction-leak shapes (GPU scalar_mul_mod_n) MUST be flagged.
    def _scan_red(src: str) -> int:
        with tempfile.NamedTemporaryFile("w", suffix=".cl", delete=False) as f:
            f.write(src); tmp = Path(f.name)
        try:
            return len(g.scan_reduction_file(tmp))
        finally:
            tmp.unlink()
    check("flags reduction skip-if-zero (prod[i]==0)",
          _scan_red("    for (int i=0;i<4;i++){ if (prod[4+i] == 0) continue; }\n") == 1)
    check("flags reduction res[4]!=0 skip",
          _scan_red("    if (res[4] != 0) { fold(); }\n") == 1)
    check("flags data-dependent && carry loop bound",
          _scan_red("    for (int k = i+3; k < 7 && carry; k++) {}\n") == 1)
    check("does NOT flag the fixed fixed-span loop",
          _scan_red("    for (int k = i+3; k < 7; k++) {}\n") == 0)
    check("does NOT flag a trailing-comment annotation of the removed pattern",
          _scan_red("    ulong carry = 0;   // no `if (prod[i]==0) continue;`\n") == 0)

    # 8. The live tree MUST currently pass the gate.
    check("live tree passes the gate", g.main() == 0)

    print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILURE(S)'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
