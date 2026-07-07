#!/usr/bin/env python3
"""
check_soundness_coverage.py — the soundness-coverage gate (negative-test ledger).

WHY (the bug class this prevents):
GHSA-c7q2-gv3g-rgxm slipped EVERY review and every CAAS gate because the entire
test/gate stack asserts that *correct* inputs are ACCEPTED (honest sign -> verify
roundtrips). The adaptor's "binding" was mathematically vacuous yet every honest
roundtrip still passed. The only thing that catches such a hole is a test built on
the INVERTED invariant: forge an input that VIOLATES the security property and
assert the verifier BLOCKS it. If the system cannot block the wrong case, that is
the hole.

This gate makes that discipline structural. docs/SOUNDNESS_INVARIANTS.json is a
ledger: every cryptographic verify/proof function declares its security invariant
and (when covered) a NEGATIVE-soundness probe wired into the audit runner. The gate:

  1. BLOCKS if a 'covered' invariant's probe is not actually wired (regression of
     existing soundness coverage).
  2. BLOCKS if a custom-protocol verify function exists in the engine source but is
     NOT in the ledger at all — a new soundness surface must be classified
     (covered with a forge-probe, or explicitly marked 'roadmap'). This is the
     check_security_fix_has_test analogue for soundness.
  3. REPORTS the 'roadmap' holes loudly (declared-but-not-yet-probed) so the gap is
     visible and closed incrementally.

Standard signature verifies (ecdsa_verify, schnorr_verify) are intentionally out of
scope: they have a differential oracle vs libsecp256k1. The ledger targets the
NOVEL/custom protocols (adaptor, DLEQ, ZK, MuSig2, FROST, range) that have none —
exactly where GHSA-c7q2 lived.

Exit 0 = clean, exit 1 = a covered probe is unwired or an uncataloged verify exists.
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "docs", "SOUNDNESS_INVARIANTS.json")
RUNNER = os.path.join(ROOT, "audit", "unified_audit_runner.cpp")

# Soundness scope is SELF-DERIVING: scan the WHOLE engine core, not a hardcoded file
# list. A 4-file allowlist was the original blind spot — a verifier/attestation added to
# any other file (pedersen.cpp, taproot.cpp, ...) or a struct-returning attestation
# producer (*_snark_witness) escaped mandatory classification while the gate stayed green.
SCAN_DIR = "src/cpu/src"  # globbed for *.cpp at runtime
# A `bool NAME(` definition at namespace scope whose name carries "verify".
VERIFY_DEF = re.compile(r"^bool\s+([A-Za-z_]\w*verify\w*)\s*\(", re.MULTILINE)
# An ATTESTATION producer: a function returning a *Witness type (e.g.
# `EcdsaSnarkWitness ecdsa_snark_witness(`). Its `valid` field is consumed downstream as
# ground truth, so it is a verify-equivalent and MUST be soundness-classified (the GHSA-c7q2
# shape: a verifying-but-unsound attestation that the bool-verify regex never sees).
ATTEST_DEF = re.compile(r"^[A-Za-z_][\w:]*Witness\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE)
# Internal/variant verifiers that are not the primary public soundness surface.
VARIANT_MARKERS = ("_base", "_impl", "_device", "_kernel", "_var")
VARIANT_PREFIXES = ("batch_",)
# STANDARD verifiers are intentionally OUT of the custom-soundness ledger: they verify
# only PUBLIC data and are anchored by a differential oracle vs libsecp256k1 (and/or are
# not EC-protocol soundness at all — MAC/checksum). Exempting them is a conscious, reviewed
# act; the differential gate covers them. Anything NOT here must be cataloged.
STANDARD_VERIFIERS = {
    "ecdsa_verify", "schnorr_verify",            # standard sig verify — differential oracle
    "ecdsa_batch_verify", "schnorr_batch_verify",  # batch of the above
    "ecdsa_batch_verify_mt",                       # multi-threaded variant of ecdsa_batch_verify:
                                                   # identical boolean result on PUBLIC data, covered by
                                                   # differential module regression_ecdsa_batch_verify_mt
                                                   # (asserts MT == serial == libsecp for thread counts)
    "schnorr_batch_verify_mt",                     # multi-threaded twin of schnorr_batch_verify:
                                                   # identical boolean result on PUBLIC data (BIP-340 is
                                                   # variable-time over public pubkey/msg/sig); covered by
                                                   # the shim test shim_batch_mt (MT == single across thread
                                                   # counts {0,1,2,8,64} + per-row results), same class as
                                                   # ecdsa_batch_verify_mt
    "ecdsa_batch_verify_opaque_rows",              # libbitcoin row-layout adapter: parses PUBLIC
                                                   # (digest,pubkey,sig) rows, then delegates to
                                                   # ecdsa_batch_verify plus per-row ecdsa fallback.
    "ecdsa_batch_verify_opaque_columns",           # libbitcoin column-layout twin of the same standard
                                                   # ECDSA batch verifier surface.
    "schnorr_batch_verify_bip340_rows",            # BIP-340 row-layout adapter: parses PUBLIC
                                                   # (msg,xonly,sig) rows, then delegates to
                                                   # schnorr_batch_verify plus canonical schnorr_verify.
    "schnorr_batch_verify_bip340_columns",         # BIP-340 column-layout twin of the same standard
                                                   # Schnorr batch verifier surface.
    "bitcoin_verify_message", "eth_personal_verify",  # message-sig = ecdsa recover + compare
    "eip55_verify",                               # address checksum, not curve soundness
    "poly1305_verify",                            # MAC tag compare, not EC soundness
}


def fail(msg, out):
    out.append("  \033[91mFAIL\033[0m  " + msg)


def scan_soundness_symbols(root, scan_dir=None):
    """Self-deriving discovery: every custom-protocol verifier (bool *verify*) AND every
    attestation producer (*Witness returning fn) across the engine core, MINUS internal
    variants and MINUS the STANDARD_VERIFIERS allowlist. Returns {symbol: relpath}.
    Pure (takes root) so the self-test can prove scope-exhaustiveness on a temp tree."""
    sd = scan_dir or os.path.join(root, SCAN_DIR)
    found = {}
    if not os.path.isdir(sd):
        return found
    for f in sorted(os.listdir(sd)):
        if not f.endswith(".cpp"):
            continue
        rel = os.path.relpath(os.path.join(sd, f), root)
        text = open(os.path.join(sd, f), encoding="utf-8", errors="replace").read()
        for name in VERIFY_DEF.findall(text) + ATTEST_DEF.findall(text):
            if any(m in name for m in VARIANT_MARKERS) or name.startswith(VARIANT_PREFIXES):
                continue  # internal variant of a primary verifier
            if name in STANDARD_VERIFIERS:
                continue  # standard public-data verifier — differential-oracle covered
            found.setdefault(name, rel)
    return found


def main() -> int:
    out = []
    blocking = 0

    if not os.path.exists(LEDGER):
        print(f"FAIL: soundness ledger not found: {LEDGER}")
        return 1
    ledger = json.load(open(LEDGER, encoding="utf-8"))
    invs = ledger.get("invariants", [])
    by_fn = {i.get("verify_function"): i for i in invs}

    runner_src = open(RUNNER, encoding="utf-8").read() if os.path.exists(RUNNER) else ""

    covered = [i for i in invs if i.get("status") == "covered"]
    roadmap = [i for i in invs if i.get("status") == "roadmap"]

    # (1) Every covered invariant must have its probe wired into the runner.
    for inv in covered:
        sym = inv.get("probe_run_symbol")
        mod = inv.get("probe_module")
        if not sym or not mod:
            fail(f"invariant '{inv['id']}' is 'covered' but declares no probe_run_symbol/probe_module", out)
            blocking += 1
            continue
        if sym not in runner_src:
            fail(f"covered probe '{sym}' not forward-declared/referenced in unified_audit_runner.cpp", out)
            blocking += 1
        if f'"{mod}"' not in runner_src:
            fail(f"covered probe module \"{mod}\" not registered in ALL_MODULES (unified_audit_runner.cpp)", out)
            blocking += 1

    # (2) Every custom-protocol verifier AND attestation producer in the WHOLE engine
    #     core must be in the ledger. Self-deriving scope: glob src/cpu/src/*.cpp so a new
    #     file or a struct-returning *_snark_witness cannot escape classification.
    scanned = scan_soundness_symbols(ROOT)
    uncataloged = [(n, r) for n, r in sorted(scanned.items()) if n not in by_fn]
    for name, rel in uncataloged:
        fail(f"soundness-bearing symbol '{name}' ({rel}) has NO invariant in "
             f"docs/SOUNDNESS_INVARIANTS.json — add a forge-probe ('covered') or "
             f"classify it ('roadmap'). A verifier/attestation with no negative test is a "
             f"GHSA-c7q2-class hole. (If it is a standard public-data verifier with a "
             f"differential oracle, add it to STANDARD_VERIFIERS with justification.)", out)
        blocking += 1

    # Report
    print("=" * 70)
    print("  Soundness-Coverage Gate (negative-test ledger)")
    print("=" * 70)
    print(f"  invariants: {len(invs)}  |  covered (probed): {len(covered)}  |  roadmap: {len(roadmap)}")
    for inv in covered:
        print(f"    \033[92m[COVERED]\033[0m {inv['verify_function']:24} <- {inv.get('probe_module')}")
    for inv in roadmap:
        print(f"    \033[93m[ROADMAP]\033[0m {inv['verify_function']:24} (forge-probe required: {inv['id']})")
    if out:
        print()
        print("\n".join(out))

    if blocking:
        print()
        print(f"\033[91m\033[1m  SOUNDNESS-COVERAGE: {blocking} blocking issue(s)\033[0m")
        print("  A negative-soundness probe forges an input that violates the invariant and")
        print("  asserts rejection. Without it, a verifying-but-unsound input goes undetected.")
        return 1

    print()
    if roadmap:
        print(f"  OK: all {len(covered)} covered probe(s) wired; {len(roadmap)} roadmap hole(s) "
              f"declared (forge-probes pending — close incrementally).")
    else:
        print(f"  OK: all {len(invs)} soundness invariants covered by wired forge-probes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
