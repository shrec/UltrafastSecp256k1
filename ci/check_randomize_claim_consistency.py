#!/usr/bin/env python3
"""check_randomize_claim_consistency.py — REVIEWER-FRICTION-001 regression gate.

`secp256k1_context_randomize()` in the shim is NOT a no-op: it stores the seed,
pre-computes r*G, and per-context nonce blinding (generator_mul_blinded — a DPA
side-channel defense) is applied on every signing call via ContextBlindingScope
(compat/libsecp256k1_shim/src/shim_context.cpp:199-261). The PR body correctly
states "DPA defense active when secp256k1_context_randomize is called."

This gate caught REVIEWER-FRICTION-001: docs/INTEGRATION_PATCH.patch (the cmake
comment a Bitcoin Core maintainer reads first) falsely claimed randomize "is a
no-op ... instead of per-context blinding", directly contradicting the code and
the PR body's security claim.

Rule: no reviewer-facing doc may describe secp256k1_context_randomize as a no-op
or claim there is no per-context blinding. Such a claim is false and undercuts a
security claim.

Exit 0 = consistent, 1 = a false no-op/no-blinding claim was found.
"""
import re
import sys
from pathlib import Path

DOCS = [
    "docs/INTEGRATION_PATCH.patch",
    "docs/BITCOIN_CORE_PR_BODY.md",
    "docs/BITCOIN_CORE_PR_DESCRIPTION.md",
    "README.md",
]

# Regexes that are FALSE when said about context_randomize (code DOES blind).
# Note the negation lookbehind on "a no-op" so the corrected wording
# "NOT a no-op" / "not a no-op" is explicitly allowed.
BANNED = [
    r"(?<!not )a no-op",
    r"no per-context blinding",
    r"instead of per-context blinding",
    r"does not blind",
]

WINDOW = 320  # chars on each side of a context_randomize mention


def fail(msg: str) -> None:
    print(f"::error::check_randomize_claim_consistency: {msg}")


def main() -> int:
    errors = 0
    seen_any = False
    for path in DOCS:
        p = Path(path)
        if not p.exists():
            continue
        txt = p.read_text()
        low = txt.lower()
        for m in re.finditer(r"context_randomize", low):
            seen_any = True
            s = max(0, m.start() - WINDOW)
            e = min(len(low), m.end() + WINDOW)
            window = low[s:e]
            for bad in BANNED:
                if re.search(bad, window):
                    line_no = txt.count("\n", 0, m.start()) + 1
                    fail(f"{path}:{line_no}: context_randomize described with false "
                         f"claim matching /{bad}/ — randomize DOES apply per-context "
                         f"generator_mul_blinded DPA blinding (shim_context.cpp:199-261)")
                    errors += 1
                    break

    if errors:
        print(f"check_randomize_claim_consistency: {errors} false randomize claim(s)")
        return 1
    if seen_any:
        print("check_randomize_claim_consistency: randomize described consistently "
              "(per-context blinding active) [PASS]")
    else:
        print("check_randomize_claim_consistency: no context_randomize mentions [PASS]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
