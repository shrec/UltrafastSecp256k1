#!/usr/bin/env python3
"""
check_entropy_source_integrity.py — the entropy-source-integrity gate.

WHY (the bug class this prevents):
Every private key, nonce and blinding factor is only as strong as the randomness it is
drawn from. The library mandates ONE source: detail::csprng_fill — which is fail-CLOSED
(std::abort on RNG failure), so key material is never produced from a degraded source.
The blind-zone audit found bip39.cpp had shipped a SECOND, fail-OPEN csprng_fill (returned
false on /dev/urandom failure / short read) — a duplicate that both violated single-source
and could let a caller proceed past an RNG failure. The nonce-reuse / nonce-erase gates
cover REUSE and ERASURE; none covered the SOURCE. This gate does.

Rule 1 (BLOCKING): single-source + fail-closed.
  * csprng_fill may be DEFINED in exactly one place: include/secp256k1/detail/csprng.hpp.
    Any other definition is a banned local copy.
  * The canonical definition must be fail-closed: return void (not a bool a caller can
    ignore) and abort/terminate on RNG failure.

Rule 2 (ADVISORY report): weak RNG in the secret-bearing core.
  * std::random_device / std::mt19937 / rand() / srand() in src/cpu/src + compat are listed
    for triage. Some are legitimate public randomizers (batch-verify weights, aux_rand
    hedging) — those are allowlisted with justification. New ones should be reviewed; this
    is reported (not yet hard-blocking) pending per-site classification.

A self-test (ci/test_check_entropy_source_integrity.py) proves Rule 1 blocks a second def
and a fail-open canonical signature.

Exit 0 = single-source + fail-closed holds; exit 1 = a banned local csprng / fail-open canonical.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANONICAL = "src/cpu/include/secp256k1/detail/csprng.hpp"
SCAN_DIRS = ["src/cpu/src", "src/cpu/include", "compat/libsecp256k1_shim/src"]

# A csprng_fill DEFINITION (has a body brace), at any scope.
_CSPRNG_DEF = re.compile(r"^\s*(?:static\s+|inline\s+)*[\w:<>]+\s+csprng_fill\s*\([^;]*\)\s*(?:noexcept\s*)?\{", re.MULTILINE)
_WEAK = {
    "std::random_device": re.compile(r"std::random_device"),
    "std::mt19937": re.compile(r"std::mt19937"),
    "rand()": re.compile(r"(?<![_\w:])(?:std::rand|::rand|rand)\s*\("),
    "srand()": re.compile(r"\bsrand\s*\("),
}
# Triaged legitimate non-secret randomizers (public batch-verify weights; shim aux_rand
# hedging — aux_rand is defense-in-depth entropy XORed into a CSPRNG nonce, not the nonce).
_WEAK_ALLOWLIST = {
    ("src/cpu/src/batch_verify.cpp", "std::random_device"),
    ("src/cpu/src/batch_verify.cpp", "std::mt19937"),
    ("compat/libsecp256k1_shim/src/shim_ecdsa.cpp", "std::random_device"),
    ("compat/libsecp256k1_shim/src/shim_schnorr.cpp", "std::random_device"),
}


def find_csprng_defs(file_texts):
    """file_texts: {relpath: text}. Return [relpath] where csprng_fill is DEFINED."""
    return [rel for rel, text in file_texts.items() if _CSPRNG_DEF.search(text)]


def canonical_is_fail_closed(text):
    """The canonical csprng_fill must return void and abort/terminate on failure.
    Match the DEFINITION (requires a body brace) so a comment mention of csprng_fill is
    not picked up as the return type."""
    m = re.search(r"(?:inline\s+|static\s+)*(\w+)\s+csprng_fill\s*\([^;{}\n]*\)\s*(?:noexcept\s*)?\{", text)
    if not m:
        return False
    returns_void = m.group(1) == "void"
    fails_closed = ("abort(" in text) or ("std::terminate" in text) or ("__builtin_trap" in text)
    return returns_void and fails_closed


def analyze(file_texts):
    """Pure core: return (blocking[list[str]], weak_report[list[str]])."""
    blocking = []
    defs = find_csprng_defs(file_texts)
    extra = [d for d in defs if d.replace("\\", "/") != CANONICAL]
    for d in extra:
        blocking.append(f"banned local csprng_fill definition in {d} — randomness must come "
                        f"from the single fail-closed source {CANONICAL}")
    canon = file_texts.get(CANONICAL)
    if canon is None:
        blocking.append(f"canonical CSPRNG header missing: {CANONICAL}")
    elif not canonical_is_fail_closed(canon):
        blocking.append(f"{CANONICAL}: csprng_fill must be fail-closed (return void + "
                        f"abort/terminate on RNG failure) — a bool that callers can ignore is fail-open")

    weak_report = []
    for rel, text in file_texts.items():
        for name, rx in _WEAK.items():
            if rx.search(text) and (rel, name) not in _WEAK_ALLOWLIST:
                ln = text[:rx.search(text).start()].count("\n") + 1
                weak_report.append(f"{rel}:{ln}: weak RNG '{name}' in secret-bearing core (triage: "
                                   f"public randomizer? -> add to _WEAK_ALLOWLIST with justification)")
    return blocking, weak_report


def _load_files():
    out = {}
    for d in SCAN_DIRS:
        base = os.path.join(ROOT, d)
        if not os.path.isdir(base):
            continue
        for root, _, files in os.walk(base):
            for f in files:
                if f.endswith((".cpp", ".cc", ".hpp", ".h")):
                    p = os.path.join(root, f)
                    rel = os.path.relpath(p, ROOT).replace("\\", "/")
                    out[rel] = open(p, encoding="utf-8", errors="replace").read()
    # ensure canonical is present even if outside SCAN_DIRS walk
    cp = os.path.join(ROOT, CANONICAL)
    if os.path.exists(cp):
        out[CANONICAL] = open(cp, encoding="utf-8", errors="replace").read()
    return out


def run() -> int:
    files = _load_files()
    blocking, weak = analyze(files)

    print("=" * 70)
    print("  Entropy-Source-Integrity Gate (single-source + fail-closed CSPRNG)")
    print("=" * 70)
    print(f"  scanned {len(files)} file(s); canonical source: {CANONICAL}")
    if weak:
        print(f"  --- weak-RNG report (advisory, {len(weak)} site(s) to triage) ---")
        for w in weak[:20]:
            print(f"    \033[93m[WEAK]\033[0m {w}")
    if blocking:
        print()
        for b in blocking:
            print(f"  \033[91mFAIL\033[0m  {b}")
        print(f"\n\033[91m\033[1m  ENTROPY-SOURCE-INTEGRITY: {len(blocking)} blocking issue(s)\033[0m")
        return 1
    print()
    print("  OK: csprng_fill is single-source + fail-closed; no banned local entropy copies.")
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    sys.exit(main())
