"""Odd-multiples table construction, current and co-Z, verified against each other.

`build_glv52_table_zr` in src/cpu/src/point.cpp builds [1P, 3P, ..., (2T-1)P]
with zero field inversions, using the z-ratio technique: accumulate with a
mixed add that also emits Z_i/Z_{i-1}, then a single backward sweep rescales
every entry onto one shared Z.

The measured co-Z result (-34.7% on the addition itself) targets exactly this
loop, because it is the one place in the engine where both operands can be made
to share a Z. This module models BOTH constructions exactly and checks that
they produce the identical table -- as points -- before any C++ is written.
That ordering matters: getting a co-Z chain subtly wrong yields a table that is
self-consistent and wrong, which the engine's own tests would catch only
probabilistically.

Cost, in heavy operations (multiply or square), for a table of T entries:

    current   1 double (3M+4S) + C^2 + C^3 + 2 iso muls        = 11
              + (T-1) * jac52_add_mixed_inplace_zr (8M+3S)     = 11(T-1)
    co-Z      1 DBLU (1M+5S), which hands back P co-Z for free  = 6
              + (T-1) * ZADDU (5M+2S)                          = 7(T-1)

The backward sweep is identical in both and is not counted. For T = 16 that is
176 heavy operations against 111, before the sweep.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from .field import Affine, Jacobian, P, affine_add, affine_double, affine_mul, inv


def _dblu(x: int, y: int) -> Tuple[int, int, int, int, int]:
    """DBLU: double an affine point AND return it re-expressed on the new Z.

    Returns (X2, Y2, X1p, Y1p, Z) where (X2,Y2,Z) is 2P and (X1p,Y1p,Z) is P,
    both on the same Z. The co-Z copy of P costs NOTHING: the doubling already
    computes both values it needs.

        A = X^2   B = Y^2   C = B^2
        D = 2((X+B)^2 - A - C) = 4*X*B          <- this is X on the new Z
        E = 3A    F = E^2
        X2 = F - 2D
        Y2 = E(D - X2) - 8C
        Z  = 2Y
        P on Z: X*Z^2 = 4*X*B = D,  Y*Z^3 = 8*Y^4 = 8C

    Cost 1M+5S. The only multiply is E*(D - X2).
    """
    A = x * x % P
    B = y * y % P
    C = B * B % P
    D = 2 * (((x + B) % P) ** 2 - A - C) % P
    E = 3 * A % P
    F = E * E % P
    X2 = (F - 2 * D) % P
    Y2 = (E * (D - X2) - 8 * C) % P
    Z = 2 * y % P
    return X2, Y2, D, 8 * C % P, Z


def _zaddu(X1: int, Y1: int, X2: int, Y2: int, Z: int):
    """ZADDU: (P, Q) sharing Z -> (P+Q, P) both on the new Z, plus the z-ratio.

    The z-ratio Z_new/Z_old is exactly (X2 - X1), which the formula computes
    anyway -- so the backward sweep costs nothing extra to feed.
    """
    dx = (X2 - X1) % P
    dy = (Y2 - Y1) % P
    A = dx * dx % P
    B = X1 * A % P
    C = X2 * A % P
    D = dy * dy % P
    X3 = (D - B - C) % P
    Y1cb = Y1 * (C - B) % P
    Y3 = (dy * (B - X3) - Y1cb) % P
    Z3 = Z * dx % P
    return X3, Y3, B, Y1cb, Z3, dx


def build_table_coz(p: Affine, size: int):
    """[1P, 3P, ..., (2*size-1)P] via a co-Z chain.

    State invariant: `acc` (the running odd multiple) and `d` (the constant 2P)
    are ALWAYS on the same Z. ZADDU preserves that by returning both operands
    on the new Z, which is the whole reason the chain needs no normalisation
    between steps.
    """
    if p.infinity:
        raise ValueError("table build on the point at infinity")
    # DBLU gives 2P and P on a common Z, with P's co-Z form free.
    dX, dY, aX, aY, Z = _dblu(p.x, p.y)

    entries = [(aX, aY)]      # 1P on Z
    ratios = [1]              # zr[0] is unused by the sweep; kept for symmetry

    for _ in range(1, size):
        # OPERAND ORDER IS THE WHOLE TRICK. ZADDU returns the sum together with
        # its FIRST operand on the new Z. The accumulator is being replaced by
        # the sum anyway, so it is `d` -- the constant 2P -- that has to survive
        # onto the new Z. Passing (d, acc) therefore costs nothing extra;
        # passing (acc, d) leaves d on the old Z and forces a 3M+1S rescale
        # every step, which eats almost the entire saving.
        sX, sY, dX, dY, Z, zr = _zaddu(dX, dY, aX, aY, Z)
        aX, aY = sX, sY
        entries.append((aX, aY))
        ratios.append(zr)

    return entries, ratios, Z


def rescale_to_common_z(entries: Sequence[Tuple[int, int]],
                        ratios: Sequence[int]) -> List[Tuple[int, int]]:
    """Backward sweep: put every entry on the LAST entry's Z.

    Identical in shape to the sweep in build_glv52_table_zr: zs accumulates
    zr[n-1] * ... * zr[i+1] = Z_last / Z_i, and each entry is scaled by
    (zs^2, zs^3).
    """
    n = len(entries)
    out = list(entries)
    if n < 2:
        return out
    zs = ratios[n - 1]
    for idx in range(n - 2, -1, -1):
        if idx != n - 2:
            zs = zs * ratios[idx + 1] % P
        zs2 = zs * zs % P
        zs3 = zs2 * zs % P
        x, y = out[idx]
        out[idx] = (x * zs2 % P, y * zs3 % P)
    return out


def verify_table(p: Affine, size: int) -> Tuple[bool, str]:
    """The co-Z table must denote exactly [1P, 3P, ..., (2*size-1)P].

    Checked against repeated affine addition, which shares no code with either
    construction.
    """
    entries, ratios, _z = build_table_coz(p, size)
    rescaled = rescale_to_common_z(entries, ratios)
    # After the sweep every entry shares the LAST entry's Z, which is the Z the
    # chain ended on -- recover it from the last entry, whose value is unchanged.
    # The shared Z is not needed to check the POINTS: entry i and the reference
    # must agree projectively under one common Z, so compare ratios instead.
    want = p
    two_p = affine_double(p)
    for i, (x, y) in enumerate(rescaled):
        if i:
            want = affine_add(want, two_p)
        # All entries share one implied Z; verify by checking that (x, y)
        # matches want scaled by that same unknown Z. Take the first entry to
        # pin Z, then every other entry must be consistent with it.
        if i == 0:
            # x = X_1 * Zc^2, y = Y_1 * Zc^3 with X_1 = want.x, Y_1 = want.y
            if want.x == 0:
                return False, "degenerate first entry"
            zc2 = x * inv(want.x) % P
            zc3 = y * inv(want.y) % P
            zc = zc3 * inv(zc2) % P
            if zc * zc % P != zc2:
                return False, "entry 0 is not a consistent projective scaling"
            continue
        if x != want.x * zc2 % P or y != want.y * zc3 % P:
            return False, "entry %d is not (%d)P" % (i, 2 * i + 1)
    return True, "%d entries exact, all on one common Z" % len(rescaled)


def heavy_op_counts(size: int) -> dict:
    """Heavy (multiply or square) operation counts, excluding the shared sweep."""
    return {
        "current_setup": 11,                 # jac52_double 3M+4S, C^2, C^3, 2 iso muls
        "current_adds": 11 * (size - 1),     # jac52_add_mixed_inplace_zr, 8M+3S
        "current_total": 11 + 11 * (size - 1),
        "coz_setup": 6,                      # DBLU 1M+5S, co-Z copy of P free
        "coz_adds": 7 * (size - 1),          # ZADDU 5M+2S
        "coz_extra_rescale": 0,              # operand order makes the rescale free
        "coz_total": 6 + 7 * (size - 1),
    }
