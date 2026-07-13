#!/usr/bin/env python3
"""
check_opencl_generator_w4.py
=============================
Regression gate for task opencl-generator-w4-production-claude-v4.

`scalar_mul_generator_windowed_impl` (OpenCL, window w=4 precomputed
generator table {0*G..15*G}) used to rebuild a 16-entry AffinePoint table as
a LOCAL/private array on every kernel-thread invocation (128 individual
per-limb literal assignments, ~1056 bytes of private memory on the RTX 5060
Ti / NVIDIA OpenCL driver 580.173.02). It now reads the table from a single
program-scope `__constant AffinePoint GENERATOR_TABLE_W4[16]` declaration
instead — a storage-only change (measured ~32 bytes of private memory on the
same hardware/driver; see docs/BACKEND_ASSURANCE_MATRIX.md and
data/tasking/artifacts/opencl_generator_w4_production_claude_v4.json for the
actual clGetKernelWorkGroupInfo + benchmark evidence).

This gate is a static source scan (no GPU / OpenCL runtime required) that
prevents three classes of regression:

  1. The local per-call table[16] rebuild silently creeping back into
     scalar_mul_generator_windowed_impl (would reintroduce the ~1024 extra
     bytes of private memory and the per-call rebuild cost).
  2. GENERATOR_TABLE_W4 being declared more than once (duplicated storage
     across files defeats the point of hoisting it to a single declaration,
     and risks the two copies drifting apart).
  3. Any of the 16 canonical table entries (i*G for i=0..15) silently
     diverging from the true secp256k1 generator-point multiples — checked
     against values independently recomputed by this script's own from-
     scratch EC point-doubling/addition (not copied from the production
     file), not just presence-checked.

Exit codes:
  0  — all checks pass
  1  — a regression was detected (see printed detail)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXTENDED_CL = ROOT / "src" / "opencl" / "kernels" / "secp256k1_extended.cl"
BIP352_CL = ROOT / "src" / "opencl" / "kernels" / "secp256k1_bip352.cl"

TABLE_DECL_RE = re.compile(r"__constant\s+AffinePoint\s+GENERATOR_TABLE_W4\s*\[\s*16\s*\]")
FUNC_START_RE = re.compile(r"inline\s+void\s+scalar_mul_generator_windowed_impl\s*\([^)]*\)\s*\{")

# secp256k1 field prime p and generator point G (standard SEC2 domain
# parameters, not read from the repository) -- used to independently
# recompute i*G for i=0..15 from scratch, so this gate does not just check
# "does the file still say what it used to say" but "is the table
# mathematically the correct set of generator-point multiples."
_P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
_GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
_GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8


def _point_add(p1, p2):
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2:
        if (y1 + y2) % _P == 0:
            return None
        lam = (3 * x1 * x1) * pow(2 * y1, -1, _P) % _P
    else:
        lam = (y2 - y1) * pow((x2 - x1) % _P, -1, _P) % _P
    x3 = (lam * lam - x1 - x2) % _P
    y3 = (lam * (x1 - x3) - y1) % _P
    return (x3, y3)


def _scalar_mul(k, point):
    result = None
    addend = point
    while k:
        if k & 1:
            result = _point_add(result, addend)
        addend = _point_add(addend, addend)
        k >>= 1
    return result


def _limbs_le(v: int) -> tuple[int, int, int, int]:
    return tuple((v >> (64 * i)) & 0xFFFFFFFFFFFFFFFF for i in range(4))


def independent_canonical_table() -> dict[int, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Independently recompute {0*G..15*G} via from-scratch EC math (this
    function does not read GENERATOR_TABLE_W4 or any project source)."""
    table: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {
        0: ((0, 0, 0, 0), (0, 0, 0, 0))
    }
    for i in range(1, 16):
        p = _scalar_mul(i, (_GX, _GY))
        assert p is not None
        table[i] = (_limbs_le(p[0]), _limbs_le(p[1]))
    return table


def _parse_hex_literal(v: str) -> int:
    v = v.strip()
    while v and v[-1] in "uUlL":
        v = v[:-1]
    return int(v, 16) if v.lower().startswith("0x") else int(v)


def parse_table_from_source(src: str) -> dict[int, tuple[tuple[int, ...], tuple[int, ...]]] | None:
    m = TABLE_DECL_RE.search(src)
    if not m:
        return None
    brace_start = src.index("{", m.end())
    # Find the matching closing brace for the initializer list (simple depth
    # counter -- the initializer contains no nested braces beyond the fixed
    # {{...}},{{...}} structure per entry, so this is safe).
    depth = 0
    end = None
    for i in range(brace_start, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None:
        return None
    body = src[brace_start:end + 1]
    entry_re = re.compile(r"\{\s*\{\{([^}]*)\}\}\s*,\s*\{\{([^}]*)\}\}\s*\}")
    entries = entry_re.findall(body)
    if len(entries) != 16:
        return None
    table = {}
    for i, (xs, ys) in enumerate(entries):
        xvals = tuple(_parse_hex_literal(v) for v in xs.split(","))
        yvals = tuple(_parse_hex_literal(v) for v in ys.split(","))
        table[i] = (xvals, yvals)
    return table


def extract_function_body(src: str) -> str | None:
    m = FUNC_START_RE.search(src)
    if not m:
        return None
    brace_start = src.index("{", m.start())
    depth = 0
    for i in range(brace_start, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[m.start():i + 1]
    return None


def main() -> int:
    if not EXTENDED_CL.is_file():
        print(f"FAIL: cannot read {EXTENDED_CL}", file=sys.stderr)
        return 1
    src = EXTENDED_CL.read_text()
    failures: list[str] = []

    # (1) Exactly one GENERATOR_TABLE_W4 declaration in secp256k1_extended.cl.
    decl_count = len(TABLE_DECL_RE.findall(src))
    if decl_count != 1:
        failures.append(
            f"expected exactly ONE '__constant AffinePoint GENERATOR_TABLE_W4[16]' "
            f"declaration in {EXTENDED_CL}, found {decl_count}"
        )

    # (2) No duplicate declaration in secp256k1_bip352.cl (it #includes
    # secp256k1_extended.cl, so it must inherit the table, not redeclare it).
    if BIP352_CL.is_file():
        bip352_src = BIP352_CL.read_text()
        bip352_decls = len(TABLE_DECL_RE.findall(bip352_src))
        if bip352_decls != 0:
            failures.append(
                f"secp256k1_bip352.cl must NOT declare its own GENERATOR_TABLE_W4 "
                f"(found {bip352_decls}) -- it must inherit the single declaration "
                f"via '#include \"secp256k1_extended.cl\"'"
            )
        if '#include "secp256k1_extended.cl"' not in bip352_src:
            failures.append(
                "secp256k1_bip352.cl no longer #includes secp256k1_extended.cl -- "
                "it would lose access to GENERATOR_TABLE_W4 and scalar_mul_generator_windowed_impl"
            )

    # (3) scalar_mul_generator_windowed_impl body: local rebuild absent, reads
    # the constant table instead.
    func_body = extract_function_body(src)
    if func_body is None:
        failures.append("cannot locate scalar_mul_generator_windowed_impl body in secp256k1_extended.cl")
    else:
        if "AffinePoint table[16]" in func_body:
            failures.append(
                "scalar_mul_generator_windowed_impl still contains a local "
                "'AffinePoint table[16]' rebuild -- the per-call table "
                "reconstruction must be removed (storage moved to GENERATOR_TABLE_W4)"
            )
        if "GENERATOR_TABLE_W4[" not in func_body:
            failures.append(
                "scalar_mul_generator_windowed_impl does not reference "
                "GENERATOR_TABLE_W4[...] -- expected it to read the precomputed "
                "table from __constant address space"
            )
        # Storage-only change: the scalar-nibble extraction / doubling-and-add
        # control flow must be untouched (CT boundary requirement -- this
        # optimization must not change indexing/arithmetic/timing behavior).
        if "(w >> (nib * 4)) & 0xFUL" not in func_body:
            failures.append(
                "scalar_mul_generator_windowed_impl's nibble-extraction expression "
                "'(w >> (nib * 4)) & 0xFUL' changed -- this optimization must be "
                "storage-only (no indexing/arithmetic changes)"
            )
        for helper in ("point_double_unchecked(r, r)", "point_from_affine(r,", "point_add_mixed_unchecked(r, r,"):
            if helper not in func_body:
                failures.append(
                    f"scalar_mul_generator_windowed_impl no longer calls '{helper}' -- "
                    "control-flow/arithmetic must be unchanged (storage-only optimization)"
                )

    # (4) All 16 canonical table entries match independently recomputed values.
    parsed = parse_table_from_source(src)
    if parsed is None:
        failures.append("could not parse GENERATOR_TABLE_W4 initializer contents")
    else:
        canonical = independent_canonical_table()
        for i in range(16):
            if parsed.get(i) != canonical[i]:
                failures.append(
                    f"GENERATOR_TABLE_W4[{i}] does not match the independently "
                    f"recomputed canonical value {i}*G: "
                    f"parsed={parsed.get(i)} expected={canonical[i]}"
                )

    if failures:
        print("FAIL: opencl_generator_w4 regression gate")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"OK: GENERATOR_TABLE_W4 declared exactly once in {EXTENDED_CL.relative_to(ROOT)}, "
          "local per-call rebuild absent, all 16 canonical entries verified against an "
          "independently recomputed reference table.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
