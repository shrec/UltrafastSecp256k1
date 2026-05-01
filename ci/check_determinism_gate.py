#!/usr/bin/env python3
"""Fail-closed determinism gate for core cryptographic API behavior."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _ufsecp import UfSecp, find_lib  # noqa: E402


ECDSA_VECTORS = [
    {
        "name": "ecdsa_det_v1",
        "sk": "1f1e1d1c1b1a191817161514131211100f0e0d0c0b0a09080706050403020101",
        "msg": "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
    },
    {
        "name": "ecdsa_det_v2",
        "sk": "4c0883a69102937d6231471b5dbb6204fe5129617082790f9f2f7f6f7f6f7f6f",
        "msg": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
    },
    {
        "name": "ecdsa_det_v3",
        "sk": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "msg": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    },
]

ECDH_PAIRS = [
    (
        "1111111111111111111111111111111111111111111111111111111111111111",
        "2222222222222222222222222222222222222222222222222222222222222222",
    ),
    (
        "3333333333333333333333333333333333333333333333333333333333333333",
        "4444444444444444444444444444444444444444444444444444444444444444",
    ),
]

BIP32_SEED = bytes.fromhex(
    "000102030405060708090a0b0c0d0e0f"
    "101112131415161718191a1b1c1d1e1f"
)
BIP32_PATHS = [
    "m/0/1",
    "m/44'/0'/0'/0/0",
    "m/84'/0'/0'/1/7",
]


def _hex32(s: str) -> bytes:
    b = bytes.fromhex(s)
    if len(b) != 32:
        raise ValueError(f"expected 32-byte hex, got {len(b)} bytes")
    return b


def run(lib_path: Optional[str], repeat: int, json_mode: bool, out_file: Optional[str]) -> int:
    issues: list[str] = []
    checks: dict = {
        "ecdsa_vectors": [],
        "ecdh_pairs": [],
        "bip32_paths": [],
    }

    try:
        resolved = find_lib(lib_path)
        lib = UfSecp(resolved)
    except Exception as exc:
        issues.append(f"load_library_failed: {exc}")
        report = {
            "library": lib_path,
            "repeat": repeat,
            "overall_pass": False,
            "issues": issues,
            "checks": checks,
        }
        rendered = json.dumps(report, indent=2)
        if out_file:
            Path(out_file).write_text(rendered, encoding="utf-8")
        if json_mode:
            print(rendered)
        else:
            print(f"FAIL {issues[0]}")
        return 1

    # 1) ECDSA determinism + self-verification
    for vec in ECDSA_VECTORS:
        sk = _hex32(vec["sk"])
        msg = _hex32(vec["msg"])
        pub = lib.pubkey(sk)

        sig0 = lib.sign(msg, sk)
        deterministic = True
        for _ in range(max(1, repeat) - 1):
            if lib.sign(msg, sk) != sig0:
                deterministic = False
                break

        verifies = lib.verify(msg, sig0, pub)
        checks["ecdsa_vectors"].append(
            {
                "name": vec["name"],
                "deterministic": deterministic,
                "verify_ok": verifies,
            }
        )
        if not deterministic:
            issues.append(f"ecdsa_nondeterministic:{vec['name']}")
        if not verifies:
            issues.append(f"ecdsa_self_verify_failed:{vec['name']}")

    # 2) ECDH symmetry + repeat determinism
    for idx, (a_hex, b_hex) in enumerate(ECDH_PAIRS, start=1):
        sk_a = _hex32(a_hex)
        sk_b = _hex32(b_hex)
        pk_a = lib.pubkey(sk_a)
        pk_b = lib.pubkey(sk_b)

        ab = lib.ecdh(sk_a, pk_b)
        ba = lib.ecdh(sk_b, pk_a)
        symmetric = ab == ba

        repeat_stable = True
        for _ in range(max(1, repeat) - 1):
            if lib.ecdh(sk_a, pk_b) != ab:
                repeat_stable = False
                break

        checks["ecdh_pairs"].append(
            {
                "name": f"ecdh_pair_{idx}",
                "symmetric": symmetric,
                "repeat_stable": repeat_stable,
            }
        )
        if not symmetric:
            issues.append(f"ecdh_symmetry_failed:pair_{idx}")
        if not repeat_stable:
            issues.append(f"ecdh_nondeterministic:pair_{idx}")

    # 3) BIP32 path determinism
    try:
        master = lib.bip32_master(BIP32_SEED)
    except Exception as exc:
        issues.append(f"bip32_master_failed:{exc}")
        master = None

    if master is not None:
        for path in BIP32_PATHS:
            k1 = lib.bip32_derive_path(master, path)
            k2 = lib.bip32_derive_path(master, path)
            equal = (bytes(k1.data) == bytes(k2.data)) and (int(k1.is_private) == int(k2.is_private))
            checks["bip32_paths"].append({"path": path, "deterministic": equal})
            if not equal:
                issues.append(f"bip32_nondeterministic:{path}")

    report = {
        "library": resolved,
        "repeat": repeat,
        "overall_pass": len(issues) == 0,
        "issues": issues,
        "checks": checks,
    }

    rendered = json.dumps(report, indent=2)
    if out_file:
        Path(out_file).write_text(rendered, encoding="utf-8")

    if json_mode:
        print(rendered)
    else:
        if issues:
            for issue in issues:
                print(f"FAIL {issue}")
            print(f"FAIL {len(issues)} determinism issue(s)")
        else:
            print("PASS determinism gate checks")

    return 1 if issues else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib", help="Path to libufsecp shared library")
    parser.add_argument("--repeat", type=int, default=5, help="Repeat count per vector (default: 5)")
    parser.add_argument("--json", action="store_true", help="Print JSON report")
    parser.add_argument("-o", dest="out_file", help="Write JSON report to file")
    args = parser.parse_args()

    return run(args.lib, args.repeat, args.json, args.out_file)


if __name__ == "__main__":
    raise SystemExit(main())
