#!/usr/bin/env python3
"""
Self-test for ci/dudect_ct_probe.py — DON'T TRUST, VERIFY.

We do not trust the binary-CT detector; we PROVE its decision logic blocks a timing leak.
Live hardware measurement is noisy and advisory, but the Welch t-test classifier is
deterministic: feed it a known-LEAKY distribution (class1 systematically slower) and it
MUST return 'leak'; feed it a known-UNIFORM distribution (both classes same) and it MUST
NOT. A detector that cannot be shown to flag a real leak is not a detector.

Synthetic data is generated with a deterministic LCG (no wall-clock randomness) so the
result is reproducible on every machine.
"""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location(
        "dud", os.path.join(ROOT, "ci", "dudect_ct_probe.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _lcg(seed):
    """Deterministic [0,1) generator — reproducible, no Math.random / wall clock."""
    state = seed & 0xFFFFFFFF
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state / 0x7FFFFFFF


def _noise(gen, n, mean, spread):
    """n samples ~ mean +/- spread (triangular-ish from two uniforms)."""
    out = []
    for _ in range(n):
        u = (next(gen) + next(gen)) / 2.0  # mean 0.5
        out.append(mean + (u - 0.5) * 2.0 * spread)
    return out


def main() -> int:
    m = _load()
    fails = []
    N = 4000

    # 1. UNIFORM: both classes drawn from the SAME distribution -> must NOT be 'leak'.
    g = _lcg(0xC0FFEE)
    u0 = _noise(g, N, mean=100.0, spread=15.0)
    u1 = _noise(g, N, mean=100.0, spread=15.0)
    t_u, verdict_u = m.measure_pair(u0, u1)
    if verdict_u == "leak":
        fails.append(f"uniform timing wrongly flagged as leak (|t|={abs(t_u):.2f})")

    # 2. LEAKY: class1 systematically slower (a secret-dependent branch) -> MUST be 'leak'.
    g2 = _lcg(0xBADF00D)
    l0 = _noise(g2, N, mean=100.0, spread=15.0)
    l1 = _noise(g2, N, mean=112.0, spread=15.0)   # +12 cycles when secret bit set
    t_l, verdict_l = m.measure_pair(l0, l1)
    if verdict_l != "leak":
        fails.append(f"a real timing leak (+12 cyc) was NOT flagged (|t|={abs(t_l):.2f}, got '{verdict_l}')")

    # 3. classify() boundary sanity.
    if m.classify(m.T_HARD_LEAK + 0.1) != "leak":
        fails.append("classify() must call |t| above the hard threshold a leak")
    if m.classify(0.5) != "uniform":
        fails.append("classify() must call a tiny |t| uniform")

    # 4. The live entry point advisory-skips (77) when no samples file is present.
    if m.main.__module__:  # smoke that main exists; exercise the no-samples path
        import io, contextlib
        saved = sys.argv
        try:
            sys.argv = ["dudect_ct_probe.py"]
            with contextlib.redirect_stdout(io.StringIO()):
                rc = m.main()
        finally:
            sys.argv = saved
        if rc != 77:
            fails.append(f"main() with no samples must advisory-skip (77), got {rc}")

    print("=" * 60)
    if fails:
        print("  dudect_ct_probe SELF-TEST: FAILED")
        for f in fails:
            print("   - " + f)
        print("=" * 60)
        return 1
    print(f"  dudect_ct_probe SELF-TEST PASSED")
    print(f"  (detector proven: leak flagged |t|={abs(t_l):.1f}, uniform passed |t|={abs(t_u):.1f})")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
