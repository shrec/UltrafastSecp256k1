"""Limb-level modular arithmetic: the representation search one level down.

Everything above this module treated a field multiply as an atomic 21.5 ns cost.
That is the wrong abstraction: a 5x52 multiply is 25 limb products, four
reduction multiplies by the folding constant, and a carry schedule, and *that*
is where both the time and the real representation freedom live.

The arithmetic that makes secp256k1 special:

    p = 2^256 - 2^32 - 977
    2^256 == 2^32 + 977                       (mod p)
    2^260 == 2^4 * (2^32 + 977) == 0x1000003D10   (mod p)

So for a = sum a_i 2^(52i) and b = sum b_j 2^(52j), the product columns

    c_k = sum_{i+j=k} a_i b_j        k = 0 .. 8

reduce as

    a*b == sum_{k=0..4} c_k 2^(52k) + R52 * sum_{k=5..8} c_k 2^(52(k-5))   (mod p)

with R52 = 0x1000003D10. Generic 512-bit division collapses into four extra
multiplies by a small constant. That collapse is itself a representation win,
already harvested; the open question is the SCHEDULE.

What is free to choose, with the answer held fixed:

  * the order in which the nine columns are accumulated
  * where each fold by R52 (or R52 << 12, or R52 >> 4) is applied
  * how many 128-bit accumulators are live at once
  * when each 52-bit digit is extracted and the accumulator shifted
  * whether a column is split across accumulators to lengthen a carry chain

Each choice changes register pressure, carry-chain length and ILP, and none of
them changes the answer. This module makes the schedule an explicit object that
can be generated, VERIFIED EXACTLY, bounds-checked, and costed.

The bounds check is not optional. A schedule that lets a 128-bit accumulator
overflow produces a silently wrong field element -- the exact failure mode that
has no runtime detection in this engine (issue #396). Every schedule here is
checked against a symbolic worst-case bound before it is allowed to be timed.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from .field import P, Rng

# The 5x52 representation.
LIMBS = 5
LIMB_BITS = 52
M52 = (1 << 52) - 1
M48 = (1 << 48) - 1

# 2^260 mod p.  This is the constant the whole reduction turns on.
R52 = 0x1000003D10
# 2^256 mod p, used where the fold lands on a different digit boundary.
R56 = 0x1000003D1

assert R52 == (1 << 260) % P
assert R56 == (1 << 256) % P
assert R52 == R56 * 16

MASK64 = (1 << 64) - 1
MASK128 = (1 << 128) - 1


def from_limbs(n: Sequence[int]) -> int:
    """Interpret 5 limbs as an integer.  Limbs may exceed 52 bits (magnitude)."""
    return sum(int(v) << (LIMB_BITS * i) for i, v in enumerate(n))


def to_limbs(x: int) -> List[int]:
    """Canonical 5x52 decomposition of a value in [0, 2^260)."""
    return [(x >> (LIMB_BITS * i)) & M52 for i in range(LIMBS)]


# ==========================================================================
# The schedule IR
# ==========================================================================
# A schedule is a straight-line program over two kinds of value:
#   u64   a 64-bit word
#   u128  a 128-bit accumulator
#
# Operations correspond one-to-one to what the hardware offers, so the cost
# model counts real instructions rather than abstract "operations":
#
#   mulw   d, a, b        d(u128) = a(u64) * b(u64)          MULX
#   accm   d, a, b        d(u128) += a(u64) * b(u64)         MULX + ADCX/ADOX
#   accv   d, s           d(u128) += s(u128)                 ADD + ADC
#   mulc   d, a, K        d(u128) = a(u64) * K   (constant)  MULX
#   accc   d, a, K        d(u128) += a(u64) * K              MULX + ADCX/ADOX
#   lo52   w, d           w(u64) = lo(d) & M52               AND
#   lo48   w, d           w(u64) = lo(d) & M48               AND
#   lo64   w, d           w(u64) = lo(d)                     MOV
#   shr    d, d, k        d >>= k                            SHRD + SHR
#   shl    w, a, k        w(u64) = a << k                    SHL
#   or64   w, a, b        w(u64) = a | b                     OR
#   setz   d              d = 0                              XOR


class Op(NamedTuple):
    kind: str
    dst: str
    args: Tuple[str, ...]
    imm: Optional[int] = None


# Instruction cost in "MULX-equivalents", used only for ranking.  These are
# placeholders until the limb-op microbenchmark calibrates them; they are NOT
# measurements and must never be quoted as such.
OP_COST = {
    "mulw": 1.0, "accm": 1.15, "mulc": 1.0, "accc": 1.15,
    "accv": 0.35, "lo52": 0.12, "lo48": 0.12, "lo64": 0.10,
    "shr": 0.30, "shl": 0.10, "or64": 0.10, "setz": 0.05,
}


class Schedule:
    """A verified, bounds-checked limb-level field-multiply schedule."""

    def __init__(self, name: str, ops: Sequence[Op], outputs: Sequence[str], note: str = ""):
        self.name = name
        self.ops = list(ops)
        self.outputs = list(outputs)
        self.note = note
        if len(self.outputs) != LIMBS:
            raise ValueError("%s: expected %d output limbs, got %d"
                             % (name, LIMBS, len(self.outputs)))

    # -- exact evaluation --------------------------------------------------
    def evaluate(self, a: Sequence[int], b: Sequence[int],
                 check_overflow: bool = True) -> List[int]:
        """Run the schedule with exact integer semantics.

        Every u64 and u128 write is checked against its width.  An overflow is
        raised rather than silently wrapped, because in C it would wrap and
        produce a wrong field element with no other symptom.
        """
        env: Dict[str, int] = {}
        for i in range(LIMBS):
            env["a%d" % i] = int(a[i])
            env["b%d" % i] = int(b[i])

        def w128(name: str, value: int):
            if check_overflow and not (0 <= value <= MASK128):
                raise OverflowError("%s: accumulator %s overflowed 128 bits (%d bits)"
                                    % (self.name, name, value.bit_length()))
            env[name] = value & MASK128

        def w64(name: str, value: int):
            if check_overflow and not (0 <= value <= MASK64):
                raise OverflowError("%s: word %s overflowed 64 bits (%d bits)"
                                    % (self.name, name, value.bit_length()))
            env[name] = value & MASK64

        for op in self.ops:
            k, d, ar = op.kind, op.dst, op.args
            if k == "setz":
                env[d] = 0
            elif k == "mulw":
                w128(d, env[ar[0]] * env[ar[1]])
            elif k == "accm":
                w128(d, env.get(d, 0) + env[ar[0]] * env[ar[1]])
            elif k == "mulc":
                w128(d, env[ar[0]] * op.imm)
            elif k == "accc":
                w128(d, env.get(d, 0) + env[ar[0]] * op.imm)
            elif k == "accv":
                w128(d, env.get(d, 0) + env[ar[0]])
            elif k == "lo52":
                w64(d, env[ar[0]] & M52)
            elif k == "lo48":
                w64(d, env[ar[0]] & M48)
            elif k == "lo64":
                w64(d, env[ar[0]] & MASK64)
            elif k == "shr":
                env[d] = env[ar[0]] >> op.imm
            elif k == "shl":
                w64(d, env[ar[0]] << op.imm)
            elif k == "or64":
                w64(d, env[ar[0]] | env[ar[1]])
            else:
                raise ValueError("unknown limb op %r" % k)
        return [env[o] for o in self.outputs]

    # -- worst-case accumulator analysis -----------------------------------
    def max_accumulator_bits(self, max_limb: int = (1 << 52) - 1) -> Tuple[int, str]:
        """Symbolic worst case: every limb at its maximum, tracked exactly.

        This is what decides whether a schedule is SAFE, not whether it happens
        to work on the inputs the test corpus contains.
        """
        env: Dict[str, int] = {}
        for i in range(LIMBS):
            env["a%d" % i] = max_limb
            env["b%d" % i] = max_limb
        worst, where = 0, ""
        for op in self.ops:
            k, d, ar = op.kind, op.dst, op.args
            if k == "setz":
                env[d] = 0
            elif k == "mulw":
                env[d] = env[ar[0]] * env[ar[1]]
            elif k == "accm":
                env[d] = env.get(d, 0) + env[ar[0]] * env[ar[1]]
            elif k == "mulc":
                env[d] = env[ar[0]] * op.imm
            elif k == "accc":
                env[d] = env.get(d, 0) + env[ar[0]] * op.imm
            elif k == "accv":
                env[d] = env.get(d, 0) + env[ar[0]]
            elif k in ("lo52", "lo48", "lo64"):
                env[d] = {"lo52": M52, "lo48": M48, "lo64": MASK64}[k]
            elif k == "shr":
                env[d] = env[ar[0]] >> op.imm
            elif k == "shl":
                env[d] = env[ar[0]] << op.imm
            elif k == "or64":
                env[d] = env[ar[0]] | env[ar[1]]
            bits = env[d].bit_length()
            if bits > worst:
                worst, where = bits, "%s (%s)" % (d, k)
        return worst, where

    def is_safe(self, max_limb: int = (1 << 52) - 1) -> Tuple[bool, str]:
        bits, where = self.max_accumulator_bits(max_limb)
        if bits > 128:
            return False, "worst case %d bits at %s -- OVERFLOWS" % (bits, where)
        if bits > 124:
            return False, "worst case %d bits at %s -- under 4 bits of headroom" % (bits, where)
        return True, "worst case %d bits at %s" % (bits, where)

    # -- cost --------------------------------------------------------------
    def cost(self) -> Dict[str, object]:
        counts: Dict[str, int] = {}
        for op in self.ops:
            counts[op.kind] = counts.get(op.kind, 0) + 1
        weighted = sum(OP_COST.get(k, 0.0) * v for k, v in counts.items())
        # Peak simultaneously-live values, a register-pressure proxy.
        last: Dict[str, int] = {}
        for i, op in enumerate(self.ops):
            for x in op.args:
                last[x] = i
        for o in self.outputs:
            last[o] = len(self.ops)
        live, peak = set("a%d" % i for i in range(LIMBS)) | set("b%d" % i for i in range(LIMBS)), 0
        for i, op in enumerate(self.ops):
            live.add(op.dst)
            peak = max(peak, len(live))
            for x in list(live):
                if last.get(x, -1) <= i:
                    live.discard(x)
        return {"counts": counts, "weighted": weighted, "peak_live": peak,
                "mul_count": counts.get("mulw", 0) + counts.get("accm", 0)
                             + counts.get("mulc", 0) + counts.get("accc", 0)}

    def summary(self) -> str:
        c = self.cost()
        return "%d muls, weighted %.2f, peak live %d" % (
            c["mul_count"], c["weighted"], c["peak_live"])


# ==========================================================================
# Verification against exact modular arithmetic
# ==========================================================================

def reference_mul(a: Sequence[int], b: Sequence[int]) -> int:
    """The answer every schedule must produce, as an integer mod p."""
    return (from_limbs(a) * from_limbs(b)) % P


def verify(schedule: Schedule, cases: int = 64, seed: int = 0x5EC0256B1,
           max_limb: int = M52) -> Tuple[bool, str]:
    """Exact check on random and adversarial limb vectors.

    The boundary cases matter more than the random ones: all-max limbs is
    exactly where a carry schedule with too little headroom fails, and a random
    corpus essentially never generates it.
    """
    rng = Rng(seed)
    corpus: List[Tuple[List[int], List[int]]] = []
    for _ in range(cases):
        corpus.append(([rng.next64() & max_limb for _ in range(LIMBS)],
                       [rng.next64() & max_limb for _ in range(LIMBS)]))
    allmax = [max_limb] * LIMBS
    zeros = [0] * LIMBS
    ones = [1, 0, 0, 0, 0]
    pl = to_limbs(P)
    pm1 = to_limbs(P - 1)
    for x in (allmax, zeros, ones, pl, pm1):
        for y in (allmax, zeros, ones, pl, pm1):
            corpus.append((x, y))
    # one limb at max, the rest random -- catches per-column overflows
    for i in range(LIMBS):
        x = [rng.next64() & max_limb for _ in range(LIMBS)]
        x[i] = max_limb
        corpus.append((x, allmax))

    for a, b in corpus:
        want = reference_mul(a, b)
        try:
            got_limbs = schedule.evaluate(a, b)
        except OverflowError as exc:
            return False, "overflow on a=%s b=%s: %s" % (a[:2], b[:2], exc)
        got = from_limbs(got_limbs) % P
        if got != want:
            return False, ("wrong result: a=%s b=%s\n  got  %064x\n  want %064x"
                           % (a, b, got, want))
    return True, "%d cases exact" % len(corpus)
