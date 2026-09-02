"""The co-Z table build must denote exactly the same points as the current one.

A co-Z chain that is subtly wrong produces a table that is INTERNALLY
CONSISTENT and wrong -- every entry on a common Z, just not the right points.
The engine's own randomised tests would catch that only probabilistically, so
it is pinned here against repeated affine addition, which shares no code with
either construction.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from repsearch.field import P, affine_add, affine_double, curve_points, inv
from repsearch.tablebuild import (build_table_coz, heavy_op_counts,
                                  rescale_to_common_z, verify_table)

POINTS = curve_points(6, seed=0xB00C)


@pytest.mark.parametrize("size", [2, 3, 4, 5, 8, 16, 32])
def test_coz_table_matches_group_law(size):
    for p in POINTS:
        ok, msg = verify_table(p, size)
        assert ok, "size=%d: %s" % (size, msg)


def test_coz_chain_keeps_operands_on_one_z():
    """The invariant the whole construction rests on.

    If `d` ever falls behind the accumulator's Z, ZADDU is being fed operands
    from different frames and silently computes the wrong point.
    """
    p = POINTS[0]
    entries, ratios, _z = build_table_coz(p, 8)
    assert len(entries) == 8
    assert len(ratios) == 8
    # Every z-ratio must be non-zero; a zero ratio means the chain hit
    # X2 == X1, which ZADDU cannot represent.
    assert all(r % P != 0 for r in ratios)


def test_entries_share_one_z_after_sweep():
    p = POINTS[1]
    entries, ratios, _z = build_table_coz(p, 8)
    resc = rescale_to_common_z(entries, ratios)
    want = p
    two_p = affine_double(p)
    zc2 = zc3 = None
    for i, (x, y) in enumerate(resc):
        if i:
            want = affine_add(want, two_p)
        if i == 0:
            zc2 = x * inv(want.x) % P
            zc3 = y * inv(want.y) % P
            continue
        assert x == want.x * zc2 % P, "entry %d x is on a different Z" % i
        assert y == want.y * zc3 % P, "entry %d y is on a different Z" % i


def test_operand_order_is_load_bearing():
    """Passing (acc, d) instead of (d, acc) costs 3M+1S per step.

    ZADDU returns its FIRST operand on the new Z. The accumulator is replaced by
    the sum anyway, so `d` is the operand that must survive. This test pins the
    cost model that records that, because the wrong order still produces a
    CORRECT table -- just a slower one, which no correctness test would catch.
    """
    c = heavy_op_counts(16)
    assert c["coz_extra_rescale"] == 0
    assert c["coz_total"] == 6 + 7 * 15
    assert c["current_total"] == 11 + 11 * 15
    saving = (c["current_total"] - c["coz_total"]) / c["current_total"]
    assert saving > 0.35, saving


@pytest.mark.parametrize("size", [4, 8, 16, 32])
def test_saving_holds_across_table_sizes(size):
    c = heavy_op_counts(size)
    saving = (c["current_total"] - c["coz_total"]) / c["current_total"]
    assert 0.35 < saving < 0.40, (size, saving)
