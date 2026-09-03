"""Exact multivariate polynomial and rational-function arithmetic over Q.

This is the equivalence oracle for the representation search.  Two candidate
operation graphs are declared equal only when their rational-function forms are
proven identical by exact cross-multiplication:

    a/b == c/d   <=>   a*d - c*b == 0   (as polynomials over Q)

No floating point.  No probabilistic identity test at this layer -- fingerprints
are used ONLY to bucket candidates cheaply; every claimed equality inside a
bucket is confirmed exactly.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Dict, Iterable, Tuple

# A monomial is a sorted tuple of (variable_name, exponent) with exponent > 0.
# The empty tuple () is the constant monomial.
Monomial = Tuple[Tuple[str, int], ...]

# Fingerprint field: a 61-bit Mersenne prime.  Large enough that an accidental
# collision between two genuinely different low-degree forms is negligible, and
# every collision is resolved exactly afterwards anyway.
FP_PRIME = (1 << 61) - 1


def _mono_mul(m1: Monomial, m2: Monomial) -> Monomial:
    d: Dict[str, int] = dict(m1)
    for v, e in m2:
        d[v] = d.get(v, 0) + e
    return tuple(sorted((v, e) for v, e in d.items() if e))


class Poly:
    """Dense-in-terms, sparse-in-monomials multivariate polynomial over Q."""

    __slots__ = ("terms",)

    def __init__(self, terms: Dict[Monomial, Fraction] | None = None):
        self.terms: Dict[Monomial, Fraction] = {}
        if terms:
            for m, c in terms.items():
                c = Fraction(c)
                if c:
                    self.terms[m] = c

    # -- constructors ------------------------------------------------------
    @classmethod
    def zero(cls) -> "Poly":
        return cls()

    @classmethod
    def const(cls, c) -> "Poly":
        c = Fraction(c)
        return cls({(): c}) if c else cls()

    @classmethod
    def var(cls, name: str) -> "Poly":
        return cls({((name, 1),): Fraction(1)})

    # -- predicates --------------------------------------------------------
    def is_zero(self) -> bool:
        return not self.terms

    def is_const(self) -> bool:
        return all(m == () for m in self.terms)

    def const_value(self) -> Fraction:
        return self.terms.get((), Fraction(0))

    def degree(self) -> int:
        if not self.terms:
            return -1
        return max(sum(e for _, e in m) for m in self.terms)

    def variables(self) -> set:
        out = set()
        for m in self.terms:
            for v, _ in m:
                out.add(v)
        return out

    # -- ring operations ---------------------------------------------------
    def __add__(self, other: "Poly") -> "Poly":
        out = dict(self.terms)
        for m, c in other.terms.items():
            n = out.get(m, Fraction(0)) + c
            if n:
                out[m] = n
            else:
                out.pop(m, None)
        return Poly(out)

    def __neg__(self) -> "Poly":
        return Poly({m: -c for m, c in self.terms.items()})

    def __sub__(self, other: "Poly") -> "Poly":
        return self + (-other)

    def __mul__(self, other: "Poly") -> "Poly":
        out: Dict[Monomial, Fraction] = {}
        for m1, c1 in self.terms.items():
            for m2, c2 in other.terms.items():
                m = _mono_mul(m1, m2)
                n = out.get(m, Fraction(0)) + c1 * c2
                if n:
                    out[m] = n
                else:
                    out.pop(m, None)
        return Poly(out)

    def scale(self, k) -> "Poly":
        k = Fraction(k)
        if not k:
            return Poly()
        return Poly({m: c * k for m, c in self.terms.items()})

    def __eq__(self, other) -> bool:
        return isinstance(other, Poly) and self.terms == other.terms

    def __hash__(self):
        return hash(frozenset(self.terms.items()))

    # -- evaluation --------------------------------------------------------
    def eval_mod(self, point: Dict[str, int], p: int) -> int:
        """Evaluate mod p.  Coefficient denominators are inverted mod p."""
        acc = 0
        for m, c in self.terms.items():
            term = c.numerator % p
            den = c.denominator % p
            if den != 1:
                term = term * pow(den, p - 2, p) % p
            for v, e in m:
                term = term * pow(point[v] % p, e, p) % p
            acc = (acc + term) % p
        return acc

    def eval_exact(self, point: Dict[str, Fraction]) -> Fraction:
        acc = Fraction(0)
        for m, c in self.terms.items():
            term = c
            for v, e in m:
                term *= Fraction(point[v]) ** e
            acc += term
        return acc

    # -- integral normalisation -------------------------------------------
    def integral_form(self) -> "Poly":
        """Scale by the LCM of coefficient denominators, then divide by the GCD
        of the resulting integer coefficients.  Sign is left untouched."""
        if not self.terms:
            return Poly()
        lcm = 1
        for c in self.terms.values():
            d = c.denominator
            lcm = lcm * d // gcd(lcm, d)
        ints = {m: int(c * lcm) for m, c in self.terms.items()}
        g = 0
        for v in ints.values():
            g = gcd(g, abs(v))
        if g == 0:
            g = 1
        return Poly({m: Fraction(v // g) for m, v in ints.items()})

    def _lead_key(self):
        return min(self.terms) if self.terms else ()

    def __str__(self) -> str:
        if not self.terms:
            return "0"
        parts = []
        for m in sorted(self.terms, key=lambda mm: (-sum(e for _, e in mm), mm)):
            c = self.terms[m]
            if m == ():
                parts.append(str(c))
                continue
            fs = "*".join(v if e == 1 else "%s^%d" % (v, e) for v, e in m)
            if c == 1:
                parts.append(fs)
            elif c == -1:
                parts.append("-" + fs)
            else:
                parts.append("%s*%s" % (c, fs))
        return " + ".join(parts).replace("+ -", "- ")

    __repr__ = __str__


class RatFunc:
    """Rational function num/den over Q, with EXACT equality."""

    __slots__ = ("num", "den")

    def __init__(self, num: Poly, den: Poly | None = None):
        if den is None:
            den = Poly.const(1)
        if den.is_zero():
            raise ZeroDivisionError("rational function with zero denominator")
        num, den = self._canonical_scale(num, den)
        self.num = num
        self.den = den

    @staticmethod
    def _canonical_scale(num: Poly, den: Poly) -> Tuple[Poly, Poly]:
        """Scale num and den by the SAME rational factor so the ratio is
        preserved: multiply both by the lcm of every coefficient denominator,
        then divide both by the gcd of the resulting integer coefficients, then
        fix the sign so the denominator's leading coefficient is positive.

        This is NOT a full gcd reduction (multivariate gcd is not needed here);
        equality is decided by cross-multiplication, which is exact regardless.
        The scaling only makes fingerprints and printed forms stable.
        """
        lcm_den = 1
        for poly in (num, den):
            for c in poly.terms.values():
                d = c.denominator
                lcm_den = lcm_den * d // gcd(lcm_den, d)
        n2 = num.scale(lcm_den)
        d2 = den.scale(lcm_den)
        content = 0
        for poly in (n2, d2):
            for c in poly.terms.values():
                content = gcd(content, abs(int(c)))
        if content == 0:
            content = 1
        if content != 1:
            inv = Fraction(1, content)
            n2 = n2.scale(inv)
            d2 = d2.scale(inv)
        if d2.terms and d2.terms[d2._lead_key()] < 0:
            n2 = -n2
            d2 = -d2
        return n2, d2

    # -- constructors ------------------------------------------------------
    @classmethod
    def const(cls, c) -> "RatFunc":
        return cls(Poly.const(c))

    @classmethod
    def var(cls, name: str) -> "RatFunc":
        return cls(Poly.var(name))

    # -- field operations --------------------------------------------------
    def __add__(self, o: "RatFunc") -> "RatFunc":
        return RatFunc(self.num * o.den + o.num * self.den, self.den * o.den)

    def __neg__(self) -> "RatFunc":
        return RatFunc(-self.num, self.den)

    def __sub__(self, o: "RatFunc") -> "RatFunc":
        return self + (-o)

    def __mul__(self, o: "RatFunc") -> "RatFunc":
        return RatFunc(self.num * o.num, self.den * o.den)

    def __truediv__(self, o: "RatFunc") -> "RatFunc":
        if o.num.is_zero():
            raise ZeroDivisionError("division by the zero rational function")
        return RatFunc(self.num * o.den, self.den * o.num)

    def power(self, e: int) -> "RatFunc":
        if e < 0:
            return RatFunc.const(1) / self.power(-e)
        out = RatFunc.const(1)
        base = self
        while e:
            if e & 1:
                out = out * base
            base = base * base
            e >>= 1
        return out

    # -- exact equality ----------------------------------------------------
    def equals(self, o: "RatFunc") -> bool:
        """Exact: a/b == c/d  iff  a*d - c*b == 0 as polynomials."""
        return (self.num * o.den - o.num * self.den).is_zero()

    def __eq__(self, o) -> bool:
        return isinstance(o, RatFunc) and self.equals(o)

    def __hash__(self):
        # Deliberately weak but consistent: equal RatFuncs may hash differently
        # only if _canonical_scale left them in different unreduced forms, so
        # never rely on hashing alone -- bucket by fingerprint(), confirm by
        # equals().
        return hash((self.num.degree(), self.den.degree()))

    def is_constant(self) -> bool:
        return self.num.is_const() and self.den.is_const()

    def variables(self) -> set:
        return self.num.variables() | self.den.variables()

    def fingerprint(self, points: Iterable[Dict[str, int]], p: int = FP_PRIME) -> Tuple:
        """Cheap bucketing key: value at each probe point in F_p.

        A point where the denominator vanishes yields the marker 'inf' rather
        than an exception, so poles do not destroy the fingerprint.
        """
        out = []
        for pt in points:
            d = self.den.eval_mod(pt, p)
            n = self.num.eval_mod(pt, p)
            out.append("inf" if d == 0 else n * pow(d, p - 2, p) % p)
        return tuple(out)

    def __str__(self) -> str:
        if self.den.is_const() and self.den.const_value() == 1:
            return str(self.num)
        return "(%s) / (%s)" % (self.num, self.den)

    __repr__ = __str__


def probe_points(variables, count: int = 6, seed: int = 0x5EC0256B1) -> list:
    """Deterministic probe points for fingerprinting.  Seeded, never random."""
    vs = sorted(variables)
    pts = []
    state = seed & ((1 << 64) - 1)
    for _ in range(count):
        pt = {}
        for v in vs:
            # SplitMix64 -- deterministic, no dependency on Python's RNG state.
            state = (state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
            z = state
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
            z = z ^ (z >> 31)
            pt[v] = z % (FP_PRIME - 1) + 1
        pts.append(pt)
    return pts
