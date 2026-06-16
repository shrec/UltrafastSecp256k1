#!/usr/bin/env python3
"""
Self-test for ci/check_fault_countermeasure_coverage.py — DON'T TRUST, VERIFY.

We prove the fault-countermeasure gate blocks the two silent regressions it exists to catch
— a hollowed-out sign-then-verify (no verify call left) and an unwired fault probe — and
that the countermeasure model behaves correctly. The real ledger must also pass.
"""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location(
        "fcm", os.path.join(ROOT, "ci", "check_fault_countermeasure_coverage.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    m = _load()
    fails = []

    # 0. The real repo ledger must pass.
    if m.run() != 0:
        fails.append("real FAULT_COUNTERMEASURE_LEDGER.json should pass (exit 0)")

    # 1. Model: a re-verifying countermeasure CATCHES a corrupted sig; a non-verifying one EMITS it.
    if m.single_fault_caught(reverifies=True, corrupted=True) is not False:
        fails.append("a re-verifying countermeasure must CATCH a fault-corrupted sig")
    if m.single_fault_caught(reverifies=False, corrupted=True) is not True:
        fails.append("a non-verifying path must EMIT the fault-corrupted sig (forged success)")

    # 2. Presence/hollow detectors.
    if not m.countermeasure_present("ECDSASignature ecdsa_sign_verified(...) { ... }", "ecdsa_sign_verified"):
        fails.append("countermeasure_present must find a defined function")
    if m.countermeasure_present("// nothing here", "ecdsa_sign_verified"):
        fails.append("countermeasure_present must NOT find a missing function")
    if not m.countermeasure_reverifies("auto ok = ecdsa_verify(h, P, sig);"):
        fails.append("countermeasure_reverifies must detect a verify call")
    if m.countermeasure_reverifies("return sig; // no re-verification"):
        fails.append("countermeasure_reverifies must flag a hollow (no-verify) body")

    real_runner = '"exploit_ecdsa_fault_injection" test_exploit_ecdsa_fault_injection_run'

    # 3. Coverage: a hollowed-out countermeasure (file has the symbol but NO verify) MUST block.
    led_hollow = {"paths": [{
        "id": "ecdsa", "signing_path": "x", "claim": "x",
        "countermeasure_symbol": "ecdsa_sign_verified", "countermeasure_file": "fake.cpp",
        "verify_token": "verify",
        "probe_module": "exploit_ecdsa_fault_injection",
        "probe_run_symbol": "test_exploit_ecdsa_fault_injection_run", "status": "covered"}]}
    blk, _, _ = m.check_coverage(led_hollow, real_runner,
                                 read_file=lambda p: "ECDSASignature ecdsa_sign_verified(){ return sig; }")
    if not blk:
        fails.append("a hollowed-out (no re-verify) countermeasure MUST block")

    # 4. Coverage: a deleted countermeasure (symbol absent) MUST block.
    blk2, _, _ = m.check_coverage(led_hollow, real_runner,
                                  read_file=lambda p: "// the function was removed")
    if not blk2:
        fails.append("a deleted countermeasure symbol MUST block")

    # 5. Coverage: an unwired fault probe MUST block.
    led_unwired = {"paths": [{
        "id": "x", "signing_path": "x", "claim": "x",
        "countermeasure_symbol": None, "countermeasure_file": None, "verify_token": None,
        "probe_module": "DOES_NOT_EXIST", "probe_run_symbol": "test_DOES_NOT_EXIST_run",
        "status": "covered"}]}
    blk3, _, _ = m.check_coverage(led_unwired, runner_src="", read_file=lambda p: None)
    if not blk3:
        fails.append("an unwired fault probe MUST block")

    # 6. Coverage: a live countermeasure + wired probe passes; roadmap never blocks.
    led_ok = {"paths": [
        {"id": "ecdsa", "signing_path": "x", "claim": "x",
         "countermeasure_symbol": "ecdsa_sign_verified", "countermeasure_file": "fake.cpp",
         "verify_token": "verify",
         "probe_module": "exploit_ecdsa_fault_injection",
         "probe_run_symbol": "test_exploit_ecdsa_fault_injection_run", "status": "covered"},
        {"id": "later", "signing_path": "x", "claim": "x", "countermeasure_symbol": None,
         "countermeasure_file": None, "verify_token": None,
         "probe_module": None, "probe_run_symbol": None, "status": "roadmap"}]}
    blk4, cov, road = m.check_coverage(led_ok, real_runner,
                                       read_file=lambda p: "ecdsa_sign_verified(){ ecdsa_verify(x); }")
    if blk4 or len(cov) != 1 or len(road) != 1:
        fails.append("a live countermeasure + wired probe + roadmap must pass")

    print("=" * 60)
    if fails:
        print("  check_fault_countermeasure_coverage SELF-TEST: FAILED")
        for f in fails:
            print("   - " + f)
        print("=" * 60)
        return 1
    print("  check_fault_countermeasure_coverage SELF-TEST PASSED")
    print("  (gate proven to block hollowed-out/deleted countermeasures + unwired probes)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
