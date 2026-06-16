#!/usr/bin/env python3
"""
Self-test for ci/check_secret_parse_strictness.py — DON'T TRUST, VERIFY.

Promotes secret-parse-strictness from trusted to verified: PROVE the gate flags both
banned forms on a secret input — `scalar_parse_strict` (non-_nonzero) AND `Scalar::from_bytes`
(silent mod-n reduce, Rule 11's primary CVE) — and does NOT flag the strict-nonzero parse or
a from_bytes on non-secret (public) data.
"""
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location(
        "sps", os.path.join(ROOT, "ci", "check_secret_parse_strictness.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["sps"] = m
    spec.loader.exec_module(m)
    return m


def _scan_snippet(m, code):
    # _scan_file uses path.relative_to(LIB_ROOT), so the temp file must live under the repo.
    d = tempfile.mkdtemp(dir=ROOT)
    p = Path(d) / "snippet.cpp"
    p.write_text(code)
    try:
        return m._scan_file(p)
    finally:
        p.unlink(missing_ok=True)
        os.rmdir(d)


def main() -> int:
    m = _load()
    fails = []

    # 0. The real repo must pass (impl uses strict-nonzero everywhere).
    if m.run(json_mode=False, out_file=None) != 0:
        fails.append("real repo should pass (no non-strict secret parse)")

    # 1. scalar_parse_strict (non-_nonzero) on a secret MUST be flagged.
    v = _scan_snippet(m, "ufsecp_error_t f(const uint8_t privkey[32]){ Scalar s; scalar_parse_strict(privkey, s); }")
    if not v:
        fails.append("scalar_parse_strict (non-_nonzero) on a secret MUST be flagged")

    # 2. Scalar::from_bytes on a secret MUST be flagged (silent mod-n reduce, Rule 11 CVE).
    v2 = _scan_snippet(m, "ufsecp_error_t g(const uint8_t seckey[32]){ Scalar s = Scalar::from_bytes(seckey); }")
    if not any(x["bug"] == "from_bytes_secret" for x in v2):
        fails.append("Scalar::from_bytes on a secret MUST be flagged (silent mod-n reduce)")

    # 3. strict-nonzero parse must NOT be flagged.
    v3 = _scan_snippet(m, "ufsecp_error_t h(const uint8_t privkey[32]){ Scalar s; scalar_parse_strict_nonzero(privkey, s); }")
    if v3:
        fails.append("scalar_parse_strict_nonzero on a secret must NOT be flagged")

    # 4. from_bytes on PUBLIC data must NOT be flagged.
    v4 = _scan_snippet(m, "ufsecp_error_t k(const uint8_t msg32[32]){ Scalar z = Scalar::from_bytes(msg32); }")
    if v4:
        fails.append("Scalar::from_bytes on public (non-secret) data must NOT be flagged")

    print("=" * 60)
    if fails:
        print("  check_secret_parse_strictness SELF-TEST: FAILED")
        for f in fails:
            print("   - " + f)
        print("=" * 60)
        return 1
    print("  check_secret_parse_strictness SELF-TEST PASSED")
    print("  (flags scalar_parse_strict + Scalar::from_bytes on secrets; passes strict-nonzero + public)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
