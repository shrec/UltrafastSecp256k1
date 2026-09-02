"""Grammar enumeration and equivalence-class collapse.

With four binary operators there are 4! = 24 strict precedence orders.  Each one
turns the SAME token sequence into a potentially different AST, hence a
potentially different function.  This module enumerates them, evaluates each to
an exact rational function, and groups the results into equivalence classes.

The observed phenomenon this reproduces: 24 grammars collapse to a much smaller
number of distinct canonical functions, and some classes collapse dramatically
(a variable disappears, a quadratic degenerates to linear).
"""

from __future__ import annotations

import itertools
from typing import Dict, List, NamedTuple, Sequence

from .expr import BINARY_OPS, Node, parse
from .poly import RatFunc, probe_points

# All 24 strict precedence orders.  order[0] binds MOST tightly.
PRECEDENCE_ORDERS: List[tuple] = list(itertools.permutations(BINARY_OPS))

# The conventional grammar: * and / bind tighter than + and -.
CONVENTIONAL_ORDER = ("*", "/", "+", "-")


def precedence_map(order: Sequence[str]) -> Dict[str, int]:
    """order[0] gets the highest binding strength."""
    n = len(order)
    return {op: n - i for i, op in enumerate(order)}


def conventional() -> Dict[str, int]:
    return precedence_map(CONVENTIONAL_ORDER)


class Variant(NamedTuple):
    order: tuple
    ast: Node
    value: RatFunc
    ok: bool
    error: str


def enumerate_grammars(source: str, orders: Sequence[Sequence[str]] | None = None) -> List[Variant]:
    """Parse one token sequence under every precedence order."""
    out: List[Variant] = []
    for order in (orders if orders is not None else PRECEDENCE_ORDERS):
        try:
            ast = parse(source, precedence_map(order))
            value = ast.to_ratfunc()
            out.append(Variant(tuple(order), ast, value, True, ""))
        except Exception as exc:  # division by zero, parse failure, ...
            out.append(Variant(tuple(order), None, None, False, "%s: %s" % (type(exc).__name__, exc)))
    return out


class EquivalenceClass(NamedTuple):
    value: RatFunc
    members: List[Variant]

    @property
    def canonical(self) -> str:
        return str(self.value)


def collapse(variants: Sequence[Variant]) -> List[EquivalenceClass]:
    """Group variants by EXACT equality of their rational functions.

    Fingerprints bucket cheaply; every claimed equality inside a bucket is then
    confirmed by exact cross-multiplication, so no false merge is possible.
    """
    live = [v for v in variants if v.ok]
    if not live:
        return []
    variables = set()
    for v in live:
        variables |= v.value.variables()
    points = probe_points(variables or {"_"})

    buckets: Dict[tuple, List[Variant]] = {}
    for v in live:
        buckets.setdefault(v.value.fingerprint(points), []).append(v)

    classes: List[EquivalenceClass] = []
    for members in buckets.values():
        # A fingerprint bucket may in principle hold non-equal members; split
        # it by exact equality.
        groups: List[List[Variant]] = []
        for m in members:
            for g in groups:
                if g[0].value.equals(m.value):
                    g.append(m)
                    break
            else:
                groups.append([m])
        for g in groups:
            classes.append(EquivalenceClass(g[0].value, g))

    classes.sort(key=lambda c: (-len(c.members), str(c.value)))
    return classes


def report(source: str, orders: Sequence[Sequence[str]] | None = None) -> str:
    variants = enumerate_grammars(source, orders)
    classes = collapse(variants)
    failed = [v for v in variants if not v.ok]

    lines = []
    lines.append("token sequence : %s" % source)
    lines.append("grammars       : %d" % len(variants))
    lines.append("parsed          : %d" % (len(variants) - len(failed)))
    lines.append("distinct forms : %d" % len(classes))
    lines.append("")
    for i, cls in enumerate(classes, 1):
        lines.append("[%d] %-40s  (%d grammars)" % (i, cls.canonical, len(cls.members)))
        for m in cls.members:
            lines.append("      %-16s  %s" % ("".join(m.order), m.ast))
    if failed:
        lines.append("")
        lines.append("failed to evaluate:")
        for f in failed:
            lines.append("      %-16s  %s" % ("".join(f.order), f.error))
    return "\n".join(lines)
