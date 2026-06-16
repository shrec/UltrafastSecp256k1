#!/usr/bin/env python3
"""
Self-test for ci/check_secret_erase_coverage.py — DON'T TRUST, VERIFY.

Promotes the secret-erasure threat class from trusted to verified: we PROVE the gate flags
a seckey-parsing function with no secure_erase (in BOTH the shim spelling and the ufsecp
impl spelling), and does not flag one that erases or one that touches no secret. The real
repo must pass.
"""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location(
        "sec", os.path.join(ROOT, "ci", "check_secret_erase_coverage.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["sec"] = m
    spec.loader.exec_module(m)
    return m


def main() -> int:
    # run the gate from repo root so its relative globs resolve
    os.chdir(ROOT)
    m = _load()
    fails = []

    # 0. The real repo must pass (every seckey-parsing fn erases).
    if m.main() != 0:
        fails.append("real repo should pass (all seckey-parsing fns erase)")

    # 1. impl-spelling VIOLATION must be flagged: scalar_parse_strict_nonzero, no erase.
    viol_impl = "{ Scalar s; scalar_parse_strict_nonzero(privkey, s); use(s); return OK; }"
    if not m.body_is_violation(viol_impl):
        fails.append("ufsecp impl seckey parse with NO secure_erase MUST be flagged")

    # 2. shim-spelling VIOLATION must be flagged: parse_bytes_strict_nonzero, no erase.
    viol_shim = "{ Scalar s; Scalar::parse_bytes_strict_nonzero(seckey, s); sign(s); }"
    if not m.body_is_violation(viol_shim):
        fails.append("shim seckey parse with NO secure_erase MUST be flagged")

    # 3. A function that parses AND erases must NOT be flagged.
    ok_body = "{ Scalar s; scalar_parse_strict_nonzero(privkey, s); use(s); secure_erase(&s); }"
    if m.body_is_violation(ok_body):
        fails.append("a seckey-parsing fn that erases must NOT be flagged")

    # 4. A function with no secret parse must NOT be flagged.
    nosecret = "{ Point p = parse_pubkey(pub33); verify(p); }"
    if m.body_is_violation(nosecret) or m.body_parses_seckey(nosecret):
        fails.append("a function touching no private key must NOT be flagged")

    print("=" * 60)
    if fails:
        print("  check_secret_erase_coverage SELF-TEST: FAILED")
        for f in fails:
            print("   - " + f)
        print("=" * 60)
        return 1
    print("  check_secret_erase_coverage SELF-TEST PASSED")
    print("  (flags shim + ufsecp-impl seckey parses with no erase; passes the erasing fns)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
