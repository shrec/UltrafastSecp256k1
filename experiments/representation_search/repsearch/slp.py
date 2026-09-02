"""Straight-line program IR for field-level formulas.

A candidate representation of a primitive is an SLP: an ordered sequence of
single-assignment field operations.  The SLP is the object that gets

  * evaluated exactly in F_p against the reference oracle,
  * costed (operation counts, critical-path depth, register pressure),
  * emitted as C++ FieldElement52 code for real benchmarking.

Operation vocabulary mirrors what the engine actually offers on FE52:

    input   x                       a formal input
    const   c                       a small integer constant
    add     a, b                    a + b
    sub     a, b                    a - b
    neg     a                       -a
    mul     a, b                    field multiply          (M)
    sqr     a                       field square            (S)
    mulint  a, k                    multiply by small int k (cheap, limb-wise)
    half    a                       divide by 2             (cheap, limb-wise)

`mulint` and `half` are cheap because in a 5x52 redundant representation they
are single passes over the limbs with no carry propagation -- but they consume
magnitude headroom, which the cost model tracks separately.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from .field import P, inv

CHEAP_KINDS = {"add", "sub", "neg", "mulint", "half", "const", "input"}
EXPENSIVE_KINDS = {"mul", "sqr"}


class Instr(NamedTuple):
    dst: str
    kind: str
    args: Tuple[str, ...]
    imm: Optional[int] = None


# Default cost weights, in "field multiply" units.  These are PLACEHOLDERS for
# ranking candidates cheaply; every surviving candidate must still be measured.
# They are not, and must never be presented as, benchmark results.
DEFAULT_WEIGHTS = {
    "mul": 1.00,
    "sqr": 0.93,
    "add": 0.06,
    "sub": 0.06,
    "neg": 0.04,
    "mulint": 0.05,
    "half": 0.05,
    "const": 0.0,
    "input": 0.0,
}


# -- FE52 magnitude model --------------------------------------------------
# Derived from the engine's own documented bounds:
#   negate(m)              -> magnitude m + 1     (computes (m+1)*p - a)
#   mul_int_assign(k)      -> magnitude m * k,  requires m * k < 4096
#   half_assign()          -> magnitude (m >> 1) + 1
#   mul / sqr              -> magnitude 1,  requires 5 * m_a * m_b < MUL_MAG_PRODUCT_LIMIT
#   a - b  (FE52 operator-) is a + b.negate(m_b) -> magnitude m_a + m_b + 1
#   normalize_weak()       -> magnitude 1,  safe from magnitude <= ~4000
#
# The mul bound is the 128-bit accumulator headroom quoted in point.cpp's own
# magnitude annotations ("6*6*5 = 180 < 3.3M").  It is deliberately conservative
# here: any candidate that comes within a factor of 8 of it is flagged, because
# a magnitude overflow is a silent wrong answer, not a crash.
MUL_ACC_LIMIT = 3_300_000
MUL_INT_LIMIT = 4096
NORMALIZE_WEAK_LIMIT = 4000


class MagnitudeViolation(NamedTuple):
    instr: str
    kind: str
    detail: str


class Cost(NamedTuple):
    counts: Dict[str, int]
    weighted: float
    depth: float
    max_live: int

    def summary(self) -> str:
        c = self.counts
        parts = []
        for k, label in (("mul", "M"), ("sqr", "S"), ("add", "A"), ("sub", "A-"),
                         ("neg", "N"), ("mulint", "I"), ("half", "H")):
            if c.get(k):
                parts.append("%d%s" % (c[k], label))
        return "%-24s w=%.3f depth=%.3f live=%d" % ("+".join(parts) or "-",
                                                    self.weighted, self.depth, self.max_live)


class SLP:
    def __init__(self, name: str, inputs: Sequence[str], instrs: Sequence[Instr],
                 outputs: Dict[str, str], note: str = ""):
        self.name = name
        self.inputs = list(inputs)
        self.instrs = list(instrs)
        self.outputs = dict(outputs)
        self.note = note
        self._validate()

    def _validate(self):
        defined = set(self.inputs)
        for ins in self.instrs:
            for a in ins.args:
                if a not in defined:
                    raise ValueError("%s: %s uses undefined value %r" % (self.name, ins.dst, a))
            if ins.dst in defined:
                raise ValueError("%s: %r assigned twice (SLP must be SSA)" % (self.name, ins.dst))
            defined.add(ins.dst)
        for out, val in self.outputs.items():
            if val not in defined:
                raise ValueError("%s: output %r refers to undefined %r" % (self.name, out, val))

    # -- exact evaluation --------------------------------------------------
    def evaluate(self, env: Dict[str, int], p: int = P) -> Dict[str, int]:
        vals: Dict[str, int] = {k: v % p for k, v in env.items()}
        for name in self.inputs:
            if name not in vals:
                raise KeyError("%s: missing input %r" % (self.name, name))
        half = inv(2, p)
        for ins in self.instrs:
            k = ins.kind
            if k == "const":
                vals[ins.dst] = ins.imm % p
            elif k == "add":
                vals[ins.dst] = (vals[ins.args[0]] + vals[ins.args[1]]) % p
            elif k == "sub":
                vals[ins.dst] = (vals[ins.args[0]] - vals[ins.args[1]]) % p
            elif k == "neg":
                vals[ins.dst] = (-vals[ins.args[0]]) % p
            elif k == "mul":
                vals[ins.dst] = vals[ins.args[0]] * vals[ins.args[1]] % p
            elif k == "sqr":
                v = vals[ins.args[0]]
                vals[ins.dst] = v * v % p
            elif k == "mulint":
                vals[ins.dst] = vals[ins.args[0]] * ins.imm % p
            elif k == "half":
                vals[ins.dst] = vals[ins.args[0]] * half % p
            else:
                raise ValueError("unknown op kind %r" % k)
        return {out: vals[val] for out, val in self.outputs.items()}

    # -- cost --------------------------------------------------------------
    def cost(self, weights: Dict[str, float] = None) -> Cost:
        w = weights or DEFAULT_WEIGHTS
        counts: Dict[str, int] = {}
        depth: Dict[str, float] = {name: 0.0 for name in self.inputs}
        total = 0.0
        for ins in self.instrs:
            counts[ins.kind] = counts.get(ins.kind, 0) + 1
            total += w.get(ins.kind, 0.0)
            base = max((depth.get(a, 0.0) for a in ins.args), default=0.0)
            depth[ins.dst] = base + w.get(ins.kind, 0.0)
        crit = max((depth[v] for v in self.outputs.values()), default=0.0)
        return Cost(counts, total, crit, self._max_live())

    def _max_live(self) -> int:
        """Peak simultaneously-live SSA values -- a register-pressure proxy."""
        last_use: Dict[str, int] = {}
        for i, ins in enumerate(self.instrs):
            for a in ins.args:
                last_use[a] = i
        for val in self.outputs.values():
            last_use[val] = len(self.instrs)
        live = set(self.inputs)
        peak = len(live)
        for i, ins in enumerate(self.instrs):
            live.add(ins.dst)
            peak = max(peak, len(live))
            for name in list(live):
                if last_use.get(name, -1) <= i:
                    live.discard(name)
        return peak

    # -- FE52 magnitude analysis -------------------------------------------
    def magnitudes(self, input_mags: Dict[str, int] = None):
        """Propagate FE52 magnitudes through the program.

        Returns (mags, violations, peak).  A violation is a hard correctness
        bug in the candidate, not a performance note: exceeding the accumulator
        bound produces a silently wrong field element.
        """
        mags: Dict[str, int] = {}
        for name in self.inputs:
            mags[name] = (input_mags or {}).get(name, 1)
        violations: List[MagnitudeViolation] = []
        for ins in self.instrs:
            k, a = ins.kind, ins.args
            if k == "const":
                mags[ins.dst] = 1
            elif k == "add":
                mags[ins.dst] = mags[a[0]] + mags[a[1]]
            elif k == "sub":
                # FE52 has no native subtract: a - b == a + b.negate(mag_b)
                mags[ins.dst] = mags[a[0]] + mags[a[1]] + 1
            elif k == "neg":
                mags[ins.dst] = mags[a[0]] + 1
            elif k == "half":
                mags[ins.dst] = (mags[a[0]] >> 1) + 1
            elif k == "mulint":
                m = mags[a[0]] * ins.imm
                if m >= MUL_INT_LIMIT:
                    violations.append(MagnitudeViolation(
                        ins.dst, "mul_int", "magnitude %d * %d = %d >= %d" %
                        (mags[a[0]], ins.imm, m, MUL_INT_LIMIT)))
                mags[ins.dst] = m
            elif k in ("mul", "sqr"):
                ma = mags[a[0]]
                mb = mags[a[0] if k == "sqr" else a[1]]
                acc = 5 * ma * mb
                if acc >= MUL_ACC_LIMIT:
                    violations.append(MagnitudeViolation(
                        ins.dst, k, "5 * %d * %d = %d >= %d (accumulator overflow)" %
                        (ma, mb, acc, MUL_ACC_LIMIT)))
                elif acc * 8 >= MUL_ACC_LIMIT:
                    violations.append(MagnitudeViolation(
                        ins.dst, k, "5 * %d * %d = %d is within 8x of the %d bound" %
                        (ma, mb, acc, MUL_ACC_LIMIT)))
                mags[ins.dst] = 1
            elif k == "normalize_weak":
                if mags[a[0]] > NORMALIZE_WEAK_LIMIT:
                    violations.append(MagnitudeViolation(
                        ins.dst, "normalize_weak", "input magnitude %d > %d" %
                        (mags[a[0]], NORMALIZE_WEAK_LIMIT)))
                mags[ins.dst] = 1
            else:
                raise ValueError("unknown op kind %r" % k)
        peak = max(mags.values()) if mags else 0
        return mags, violations, peak

    def output_magnitudes(self, input_mags: Dict[str, int] = None) -> Dict[str, int]:
        mags, _, _ = self.magnitudes(input_mags)
        return {out: mags[val] for out, val in self.outputs.items()}

    def is_magnitude_closed(self, input_mags: Dict[str, int]) -> bool:
        """True when the outputs fit back inside the declared input budget, i.e.
        the formula can be iterated in a loop without an inserted normalize."""
        outs = self.output_magnitudes(input_mags)
        for out, m in outs.items():
            budget = input_mags.get(out)
            if budget is not None and m > budget:
                return False
        return True

    # -- C++ FE52 emission -------------------------------------------------
    def to_cpp(self, fn_name: str = None, fe: str = "FieldElement52",
               input_mags: Dict[str, int] = None) -> str:
        fn = fn_name or self.name
        mags, _, _ = self.magnitudes(input_mags)
        params = ", ".join("const %s& %s" % (fe, i) for i in self.inputs)
        outs = ", ".join("%s& out_%s" % (fe, o) for o in sorted(self.outputs))
        lines = ["// %s%s" % (self.name, (" -- " + self.note) if self.note else ""),
                 "static inline void %s(%s, %s) noexcept {" % (fn, params, outs)]
        emitted = set(self.inputs)
        for ins in self.instrs:
            a = ins.args
            if ins.kind == "const":
                lines.append("    const %s %s = fe_small(%d);" % (fe, ins.dst, ins.imm))
            elif ins.kind == "add":
                lines.append("    %s %s = %s; %s.add_assign(%s);" % (fe, ins.dst, a[0], ins.dst, a[1]))
            elif ins.kind == "sub":
                lines.append("    %s %s = %s + %s.negate(%d);  // mag %d"
                             % (fe, ins.dst, a[0], a[1], mags[a[1]], mags[ins.dst]))
            elif ins.kind == "neg":
                lines.append("    %s %s = %s.negate(%d);  // mag %d"
                             % (fe, ins.dst, a[0], mags[a[0]], mags[ins.dst]))
            elif ins.kind == "mul":
                lines.append("    %s %s = %s * %s;" % (fe, ins.dst, a[0], a[1]))
            elif ins.kind == "sqr":
                lines.append("    %s %s = %s.square();" % (fe, ins.dst, a[0]))
            elif ins.kind == "mulint":
                lines.append("    %s %s = %s; %s.mul_int_assign(%d);" % (fe, ins.dst, a[0], ins.dst, ins.imm))
            elif ins.kind == "half":
                lines.append("    %s %s = %s; %s.half_assign();" % (fe, ins.dst, a[0], ins.dst))
            emitted.add(ins.dst)
        for o in sorted(self.outputs):
            lines.append("    out_%s = %s;" % (o, self.outputs[o]))
        lines.append("}")
        return "\n".join(lines)

    def __str__(self):
        head = "SLP %s(%s) -> %s" % (self.name, ", ".join(self.inputs),
                                     ", ".join("%s=%s" % kv for kv in sorted(self.outputs.items())))
        body = "\n".join("    %-6s = %-6s %s%s" % (i.dst, i.kind, " ".join(i.args),
                                                   "" if i.imm is None else " #%d" % i.imm)
                         for i in self.instrs)
        return head + "\n" + body


class Builder:
    """Small DSL so formulas read close to their published form."""

    def __init__(self, *inputs: str):
        self.inputs = list(inputs)
        self.instrs: List[Instr] = []
        self._n = 0

    def _fresh(self, hint: str) -> str:
        self._n += 1
        return "%s%d" % (hint, self._n)

    def _emit(self, kind: str, args, imm=None, hint="t") -> str:
        dst = self._fresh(hint)
        self.instrs.append(Instr(dst, kind, tuple(args), imm))
        return dst

    def const(self, c: int) -> str:
        return self._emit("const", (), c, "c")

    def add(self, a: str, b: str) -> str:
        return self._emit("add", (a, b))

    def sub(self, a: str, b: str) -> str:
        return self._emit("sub", (a, b))

    def neg(self, a: str) -> str:
        return self._emit("neg", (a,))

    def mul(self, a: str, b: str) -> str:
        return self._emit("mul", (a, b), hint="m")

    def sqr(self, a: str) -> str:
        return self._emit("sqr", (a,), hint="s")

    def mulint(self, a: str, k: int) -> str:
        return self._emit("mulint", (a,), k)

    def half(self, a: str) -> str:
        return self._emit("half", (a,))

    def dbl(self, a: str) -> str:
        return self.add(a, a)

    def build(self, name: str, outputs: Dict[str, str], note: str = "") -> SLP:
        return SLP(name, self.inputs, self.instrs, outputs, note)
