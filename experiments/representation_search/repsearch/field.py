"""secp256k1 field/group reference arithmetic -- the correctness oracle.

Deliberately simple and obviously-correct Python big-integer arithmetic.  This
is the independent reference every candidate representation is checked against;
it is NEVER used for timing, only for truth.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

# secp256k1 domain parameters
P = 2**256 - 2**32 - 977
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
A = 0
B = 7
GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

# The reduction constant that makes this prime attractive: 2^256 = 2^32 + 977.
FOLD_256 = 2**32 + 977
# In the 5x52 representation the top limb folds at 2^260.
FOLD_260 = 0x1000003D10

assert FOLD_260 == FOLD_256 * 16


def inv(x: int, m: int = P) -> int:
    if x % m == 0:
        raise ZeroDivisionError("inverse of zero")
    return pow(x % m, m - 2, m)


def is_on_curve(x: int, y: int) -> bool:
    return (y * y - x * x * x - B) % P == 0


class Affine:
    """Affine point; `None` coordinates mean the point at infinity."""

    __slots__ = ("x", "y", "infinity")

    def __init__(self, x: Optional[int] = None, y: Optional[int] = None, infinity: bool = False):
        self.infinity = infinity or x is None
        self.x = None if self.infinity else x % P
        self.y = None if self.infinity else y % P

    @classmethod
    def infinity_point(cls) -> "Affine":
        return cls(infinity=True)

    def __eq__(self, o):
        if not isinstance(o, Affine):
            return False
        if self.infinity or o.infinity:
            return self.infinity == o.infinity
        return self.x == o.x and self.y == o.y

    def __repr__(self):
        return "Affine(inf)" if self.infinity else "Affine(%064x,%064x)" % (self.x, self.y)


class Jacobian:
    """Jacobian point (X:Y:Z) with x = X/Z^2, y = Y/Z^3.  Z == 0 is infinity."""

    __slots__ = ("X", "Y", "Z")

    def __init__(self, X: int, Y: int, Z: int):
        self.X = X % P
        self.Y = Y % P
        self.Z = Z % P

    @property
    def infinity(self) -> bool:
        return self.Z == 0

    def to_affine(self) -> Affine:
        if self.Z == 0:
            return Affine.infinity_point()
        zi = inv(self.Z)
        zi2 = zi * zi % P
        zi3 = zi2 * zi % P
        return Affine(self.X * zi2 % P, self.Y * zi3 % P)

    def is_equivalent(self, o: "Jacobian") -> bool:
        """Projective equivalence: (X:Y:Z) ~ (L^2 X : L^3 Y : L Z) for L != 0.

        Tested WITHOUT computing L, by cross-multiplication:
            X' Z^2 == X Z'^2   and   Y' Z^3 == Y Z'^3
        """
        if self.Z == 0 or o.Z == 0:
            return self.Z == 0 and o.Z == 0
        z2, oz2 = self.Z * self.Z % P, o.Z * o.Z % P
        z3, oz3 = z2 * self.Z % P, oz2 * o.Z % P
        return (o.X * z2 - self.X * oz2) % P == 0 and (o.Y * z3 - self.Y * oz3) % P == 0

    def scaling_factor(self, o: "Jacobian") -> Optional[int]:
        """Return L such that o == (L^2 X, L^3 Y, L Z), or None."""
        if self.Z == 0 or o.Z == 0:
            return None
        lam = o.Z * inv(self.Z) % P
        if not self.is_equivalent(o):
            return None
        return lam

    def __repr__(self):
        return "Jacobian(%x,%x,%x)" % (self.X, self.Y, self.Z)


# -- reference group law (textbook affine, no shortcuts) -------------------

def affine_add(p: Affine, q: Affine) -> Affine:
    if p.infinity:
        return Affine(q.x, q.y, q.infinity)
    if q.infinity:
        return Affine(p.x, p.y, p.infinity)
    if p.x == q.x:
        if (p.y + q.y) % P == 0:
            return Affine.infinity_point()
        return affine_double(p)
    lam = (q.y - p.y) * inv(q.x - p.x) % P
    x3 = (lam * lam - p.x - q.x) % P
    y3 = (lam * (p.x - x3) - p.y) % P
    return Affine(x3, y3)


def affine_double(p: Affine) -> Affine:
    if p.infinity or p.y % P == 0:
        return Affine.infinity_point()
    lam = 3 * p.x * p.x % P * inv(2 * p.y) % P
    x3 = (lam * lam - 2 * p.x) % P
    y3 = (lam * (p.x - x3) - p.y) % P
    return Affine(x3, y3)


def affine_neg(p: Affine) -> Affine:
    if p.infinity:
        return p
    return Affine(p.x, (-p.y) % P)


def affine_mul(k: int, p: Affine) -> Affine:
    """Textbook double-and-add.  Reference only -- intentionally not fast."""
    k %= N
    acc = Affine.infinity_point()
    addend = Affine(p.x, p.y, p.infinity)
    while k:
        if k & 1:
            acc = affine_add(acc, addend)
        addend = affine_double(addend)
        k >>= 1
    return acc


G = Affine(GX, GY)


# -- deterministic test corpus --------------------------------------------

class Rng:
    """SplitMix64 -- deterministic, seeded, reproducible across machines."""

    __slots__ = ("state",)
    MASK = (1 << 64) - 1

    def __init__(self, seed: int):
        self.state = seed & self.MASK

    def next64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & self.MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & self.MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & self.MASK
        return z ^ (z >> 31)

    def next_field(self) -> int:
        v = 0
        for _ in range(4):
            v = (v << 64) | self.next64()
        return v % P

    def next_scalar(self) -> int:
        v = 0
        for _ in range(4):
            v = (v << 64) | self.next64()
        return v % N


def field_boundary_values() -> List[int]:
    """Adversarial field inputs the correctness contract requires."""
    return [
        0, 1, 2, 3,
        P - 1, P - 2,
        (P + 1) // 2,                 # the value 1/2 mod p
        2**32, 2**32 + 976, FOLD_256, FOLD_260,
        2**52 - 1, 2**52, 2**52 + 1,  # FE52 limb boundaries
        2**104 - 1, 2**104,
        2**208 - 1, 2**208,
        2**255, 2**256 - 1 - (2**32 + 977) + 1,
        (1 << 256) - (1 << 32) - 978,  # p - 1 written the other way
    ]


def curve_points(count: int, seed: int = 0x5EC0256B1) -> List[Affine]:
    """Deterministic valid curve points: k*G for pseudorandom k."""
    rng = Rng(seed)
    out = []
    for _ in range(count):
        k = rng.next_scalar() or 1
        out.append(affine_mul(k, G))
    return out


def jacobian_randomizations(p: Affine, count: int, seed: int = 0xC0FFEE) -> List[Jacobian]:
    """Same affine point, randomised nonzero Z -- exercises Z-dependence."""
    if p.infinity:
        return [Jacobian(0, 1, 0) for _ in range(count)]
    rng = Rng(seed)
    out = [Jacobian(p.x, p.y, 1)]
    for _ in range(count - 1):
        z = rng.next_field() or 1
        z2 = z * z % P
        z3 = z2 * z % P
        out.append(Jacobian(p.x * z2 % P, p.y * z3 % P, z))
    return out
