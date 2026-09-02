"""Automatic representation generation over straight-line programs.

This is the 24-grammar idea applied at scale to the real engine.  Instead of a
human writing down candidate formulas, the search generates them: every local
rewrite that preserves the answer is applied, the result is checked against the
original by exact evaluation in F_p, and survivors are costed and kept.

The guiding constraint, and the only one:

    same answer, correctly, passing the tests.

Nothing else is off limits.  Association, distribution, factorisation, sign
placement, multiply/square substitution, integer-multiply decomposition -- all
of them are free variables, and all of them are enumerated.

Two things make this safe rather than reckless:

  1. every generated program is verified by EXACT evaluation in F_p on a
     deterministic corpus, so an unsound rewrite is caught immediately rather
     than shipped;
  2. every generated program is checked against the FE52 magnitude rules, so a
     rewrite that would silently overflow an accumulator or violate a declared
     negate() magnitude is rejected before it is ever costed.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

from .field import P, Rng
from .slp import Instr, SLP, MUL_ACC_LIMIT, MUL_INT_LIMIT


# ==========================================================================
# Canonicalisation passes: CSE and dead-code elimination
# ==========================================================================

def _key(ins: Instr, alias: Dict[str, str]) -> tuple:
    args = tuple(alias.get(a, a) for a in ins.args)
    if ins.kind in ("add", "mul"):          # commutative
        args = tuple(sorted(args))
    return (ins.kind, args, ins.imm)


def cse(slp: SLP) -> SLP:
    """Common-subexpression elimination.  Commutative ops are keyed on sorted
    arguments, so `a*b` and `b*a` collapse to one instruction."""
    alias: Dict[str, str] = {}
    seen: Dict[tuple, str] = {}
    out: List[Instr] = []
    for ins in slp.instrs:
        k = _key(ins, alias)
        if k in seen:
            alias[ins.dst] = seen[k]
            continue
        seen[k] = ins.dst
        out.append(Instr(ins.dst, ins.kind, tuple(alias.get(a, a) for a in ins.args), ins.imm))
    outputs = {o: alias.get(v, v) for o, v in slp.outputs.items()}
    return SLP(slp.name, slp.inputs, out, outputs, slp.note)


def dce(slp: SLP) -> SLP:
    """Drop instructions whose results never reach an output."""
    live = set(slp.outputs.values())
    kept: List[Instr] = []
    for ins in reversed(slp.instrs):
        if ins.dst in live:
            kept.append(ins)
            live.update(ins.args)
    kept.reverse()
    return SLP(slp.name, slp.inputs, kept, slp.outputs, slp.note)


def normalise(slp: SLP) -> SLP:
    return dce(cse(slp))


# ==========================================================================
# Rewrite rules
# ==========================================================================
# Each rule takes (instrs, index, defs, uses) and yields replacement instruction
# lists.  Every rule is an ALGEBRAIC IDENTITY -- but none of them is trusted:
# the caller verifies every result by exact evaluation before keeping it.

class Rewrite(NamedTuple):
    name: str
    kind: str          # the search axis this rule belongs to
    instrs: List[Instr]


def _fresh(existing: set, hint: str) -> str:
    i = 0
    while True:
        i += 1
        cand = "%s_r%d" % (hint, i)
        if cand not in existing:
            existing.add(cand)
            return cand


def _defs(instrs: Sequence[Instr]) -> Dict[str, Instr]:
    return {i.dst: i for i in instrs}


def _use_counts(slp: SLP) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for ins in slp.instrs:
        for a in ins.args:
            counts[a] = counts.get(a, 0) + 1
    for v in slp.outputs.values():
        counts[v] = counts.get(v, 0) + 1
    return counts


def single_step_rewrites(slp: SLP) -> List[Rewrite]:
    """Every one-step rewrite applicable anywhere in the program."""
    out: List[Rewrite] = []
    instrs = slp.instrs
    defs = _defs(instrs)
    uses = _use_counts(slp)
    names = set(slp.inputs) | set(defs)

    def emit(name, kind, new_instrs):
        out.append(Rewrite(name, kind, new_instrs))

    for idx, ins in enumerate(instrs):
        pre, post = instrs[:idx], instrs[idx + 1:]
        a = ins.args

        # -- mul_square_trade ------------------------------------------
        if ins.kind == "mul" and a[0] == a[1]:
            emit("sqr@%s" % ins.dst, "mul_square_trade",
                 pre + [Instr(ins.dst, "sqr", (a[0],), None)] + list(post))

        if ins.kind == "sqr":
            emit("unsqr@%s" % ins.dst, "mul_square_trade",
                 pre + [Instr(ins.dst, "mul", (a[0], a[0]), None)] + list(post))

        # -- operation_fusion: a+a  <->  mul_int(a,2) ------------------
        if ins.kind == "add" and a[0] == a[1]:
            emit("dbl2int@%s" % ins.dst, "operation_fusion",
                 pre + [Instr(ins.dst, "mulint", (a[0],), 2)] + list(post))

        if ins.kind == "mulint" and ins.imm == 2:
            emit("int2dbl@%s" % ins.dst, "operation_fusion",
                 pre + [Instr(ins.dst, "add", (a[0], a[0]), None)] + list(post))

        # -- mul_int decomposition: k*a -> (k-1)*a + a -----------------
        if ins.kind == "mulint" and ins.imm and ins.imm > 2:
            tmp = _fresh(set(names), ins.dst)
            emit("intsplit@%s" % ins.dst, "operation_fusion",
                 pre + [Instr(tmp, "mulint", (a[0],), ins.imm - 1),
                        Instr(ins.dst, "add", (tmp, a[0]), None)] + list(post))

        # -- sign_placement: a - b  <->  a + (-b) ----------------------
        if ins.kind == "sub":
            tmp = _fresh(set(names), ins.dst)
            emit("sub2neg@%s" % ins.dst, "sign_placement",
                 pre + [Instr(tmp, "neg", (a[1],), None),
                        Instr(ins.dst, "add", (a[0], tmp), None)] + list(post))

        if ins.kind == "add":
            for side in (0, 1):
                src = defs.get(a[side])
                if src is not None and src.kind == "neg" and uses.get(a[side], 0) == 1:
                    other = a[1 - side]
                    emit("neg2sub@%s" % ins.dst, "sign_placement",
                         pre + [Instr(ins.dst, "sub", (other, src.args[0]), None)] + list(post))

        # -- sign_placement: -(a*b)  <->  (-a)*b -----------------------
        if ins.kind == "neg":
            src = defs.get(a[0])
            if src is not None and src.kind == "mul" and uses.get(a[0], 0) == 1:
                for side in (0, 1):
                    tmp = _fresh(set(names), ins.dst)
                    args = list(src.args)
                    inner = args[side]
                    other = args[1 - side]
                    body = [i for i in pre if i.dst != src.dst]
                    emit("pushneg%d@%s" % (side, ins.dst), "sign_placement",
                         body + [Instr(tmp, "neg", (inner,), None),
                                 Instr(ins.dst, "mul", (tmp, other), None)] + list(post))

        if ins.kind == "mul":
            for side in (0, 1):
                src = defs.get(a[side])
                if src is not None and src.kind == "neg" and uses.get(a[side], 0) == 1:
                    tmp = _fresh(set(names), ins.dst)
                    other = a[1 - side]
                    body = [i for i in pre if i.dst != src.dst]
                    emit("hoistneg%d@%s" % (side, ins.dst), "sign_placement",
                         body + [Instr(tmp, "mul", (src.args[0], other), None),
                                 Instr(ins.dst, "neg", (tmp,), None)] + list(post))

        # -- algebraic_rewrite: a*b + a*c  ->  a*(b+c) -----------------
        if ins.kind == "add":
            l, r = defs.get(a[0]), defs.get(a[1])
            if (l is not None and r is not None and l.kind == "mul" and r.kind == "mul"
                    and uses.get(a[0], 0) == 1 and uses.get(a[1], 0) == 1):
                for li in (0, 1):
                    for ri in (0, 1):
                        if l.args[li] != r.args[ri]:
                            continue
                        shared = l.args[li]
                        b1, c1 = l.args[1 - li], r.args[1 - ri]
                        tmp = _fresh(set(names), ins.dst)
                        body = [i for i in pre if i.dst not in (l.dst, r.dst)]
                        emit("factor@%s" % ins.dst, "algebraic_rewrite",
                             body + [Instr(tmp, "add", (b1, c1), None),
                                     Instr(ins.dst, "mul", (shared, tmp), None)] + list(post))

        # -- algebraic_rewrite: a*(b+c) -> a*b + a*c -------------------
        if ins.kind == "mul":
            for side in (0, 1):
                src = defs.get(a[side])
                if src is not None and src.kind == "add" and uses.get(a[side], 0) == 1:
                    other = a[1 - side]
                    t1 = _fresh(set(names), ins.dst)
                    t2 = _fresh(set(names), ins.dst)
                    body = [i for i in pre if i.dst != src.dst]
                    emit("distribute@%s" % ins.dst, "algebraic_rewrite",
                         body + [Instr(t1, "mul", (other, src.args[0]), None),
                                 Instr(t2, "mul", (other, src.args[1]), None),
                                 Instr(ins.dst, "add", (t1, t2), None)] + list(post))

        # -- mul_square_trade: 2ab = (a+b)^2 - a^2 - b^2 ---------------
        # Costs 3S+3A to save 1M.  A loser on this engine's S/M ratio, but it is
        # part of the space and the search should be allowed to find that out.
        if ins.kind == "mul" and a[0] != a[1]:
            s1 = _fresh(set(names), ins.dst)
            s2 = _fresh(set(names), ins.dst)
            s3 = _fresh(set(names), ins.dst)
            s4 = _fresh(set(names), ins.dst)
            s5 = _fresh(set(names), ins.dst)
            emit("karatsuba@%s" % ins.dst, "mul_square_trade",
                 pre + [Instr(s1, "add", (a[0], a[1]), None),
                        Instr(s2, "sqr", (s1,), None),
                        Instr(s3, "sqr", (a[0],), None),
                        Instr(s4, "sqr", (a[1],), None),
                        Instr(s5, "sub", (s2, s3), None),
                        Instr(ins.dst, "sub", (s5, s4), None)] + list(post))

    return out


# ==========================================================================
# Verification: exact evaluation in F_p on a deterministic corpus
# ==========================================================================

def make_corpus(inputs: Sequence[str], count: int = 16,
                seed: int = 0x5EC0256B1) -> List[Dict[str, int]]:
    """Deterministic random field inputs, plus adversarial boundary tuples.

    An SLP is a polynomial map; two SLPs that agree on this many independent
    random points are equal as maps with overwhelming probability, and any
    unsound rewrite disagrees on essentially every point.  The boundary tuples
    then catch the cases random sampling never reaches.
    """
    rng = Rng(seed)
    corpus = [{name: rng.next_field() or 1 for name in inputs} for _ in range(count)]
    for const in (0, 1, 2, P - 1, (P + 1) // 2, 2**52, 2**52 - 1):
        corpus.append({name: const for name in inputs})
    # mixed: one input at a boundary, the rest random
    for i, name in enumerate(inputs):
        pt = {n: rng.next_field() or 1 for n in inputs}
        pt[name] = [0, 1, P - 1][i % 3]
        corpus.append(pt)
    return corpus


def expected_outputs(reference: SLP, corpus: Sequence[Dict[str, int]]) -> List[Optional[Dict[str, int]]]:
    """Evaluate the reference once; every candidate is compared against this."""
    out = []
    for point in corpus:
        try:
            out.append(reference.evaluate(point))
        except ZeroDivisionError:
            out.append(None)
    return out


def equivalent(expected: Sequence[Optional[Dict[str, int]]], candidate: SLP,
               corpus: Sequence[Dict[str, int]], output_names: set) -> bool:
    """Exact equality of every output on every corpus point.

    This is STRONGER than projective equivalence: a peephole rewrite must
    preserve the coordinate VALUES, not merely the point they denote.  Anything
    weaker would let a rewrite silently change the projective representative and
    break a caller that depends on the raw coordinates.
    """
    if set(candidate.outputs) != output_names:
        return False
    for want, point in zip(expected, corpus):
        if want is None:
            continue
        try:
            got = candidate.evaluate(point)
        except ZeroDivisionError:
            return False
        if want != got:
            return False
    return True


def magnitude_ok(slp: SLP, input_mags: Dict[str, int],
                 declared: Dict[str, int] = None) -> Tuple[bool, str]:
    """Reject anything that would overflow, or that would exceed the magnitudes
    the consuming code declares at its negate() call sites."""
    try:
        _m, violations, _peak = slp.magnitudes(input_mags)
    except Exception as exc:
        return False, "magnitude model failed: %s" % exc
    hard = [v for v in violations if "overflow" in v.detail or ">= %d" % MUL_INT_LIMIT in v.detail]
    if hard:
        return False, "; ".join("%s: %s" % (v.kind, v.detail) for v in hard)
    if declared:
        outs = slp.output_magnitudes(input_mags)
        bad = {k: (outs[k], declared[k]) for k in declared if k in outs and outs[k] > declared[k]}
        if bad:
            return False, "exceeds declared magnitude: " + ", ".join(
                "%s=%d>%d" % (k, v[0], v[1]) for k, v in bad.items())
    return True, ""


# ==========================================================================
# The search
# ==========================================================================

class Candidate(NamedTuple):
    slp: SLP
    path: Tuple[str, ...]
    kinds: Tuple[str, ...]
    weighted: float
    depth: float
    live: int
    counts: Dict[str, int]

    def signature(self) -> tuple:
        """Program identity, so the search does not revisit the same program."""
        return tuple((i.kind, i.args, i.imm) for i in self.slp.instrs) + \
               tuple(sorted(self.slp.outputs.items()))


class SearchResult(NamedTuple):
    reference: Candidate
    candidates: List[Candidate]
    explored: int
    rejected_unsound: int
    rejected_magnitude: int


def _measure(slp: SLP, path, kinds) -> Candidate:
    c = slp.cost()
    return Candidate(slp, tuple(path), tuple(kinds), c.weighted, c.depth, c.max_live, c.counts)


def search(reference: SLP,
           input_mags: Dict[str, int],
           declared: Dict[str, int] = None,
           depth: int = 3,
           beam: int = 40,
           corpus_size: int = 16,
           progress: Callable[[str], None] = None) -> SearchResult:
    """Beam search over the rewrite graph.

    Breadth is bounded by `beam`, depth by `depth`.  Both bounds are reported so
    a truncated search is never mistaken for an exhaustive one.
    """
    say = progress or (lambda _m: None)
    corpus = make_corpus(reference.inputs, corpus_size)
    expected = expected_outputs(reference, corpus)
    output_names = set(reference.outputs)
    base = _measure(normalise(reference), (), ())

    seen = {base.signature()}
    frontier = [base]
    found: Dict[tuple, Candidate] = {base.signature(): base}
    explored = 0
    bad_math = 0
    bad_mag = 0

    for level in range(depth):
        nxt: List[Candidate] = []
        for cand in frontier:
            for rw in single_step_rewrites(cand.slp):
                explored += 1
                try:
                    raw = SLP(reference.name, reference.inputs, rw.instrs,
                              cand.slp.outputs, reference.note)
                    new = normalise(raw)
                except Exception:
                    continue
                sig = tuple((i.kind, i.args, i.imm) for i in new.instrs) + \
                    tuple(sorted(new.outputs.items()))
                if sig in seen:
                    continue
                seen.add(sig)
                if not equivalent(expected, new, corpus, output_names):
                    bad_math += 1
                    continue
                ok, _why = magnitude_ok(new, input_mags, declared)
                if not ok:
                    bad_mag += 1
                    continue
                c = _measure(new, cand.path + (rw.name,), cand.kinds + (rw.kind,))
                found[sig] = c
                nxt.append(c)
        nxt.sort(key=lambda c: (c.weighted, c.depth, c.live))
        frontier = nxt[:beam]
        say("  depth %d: %d new programs kept, frontier %d (explored %d, "
            "unsound %d, magnitude-rejected %d)"
            % (level + 1, len(nxt), len(frontier), explored, bad_math, bad_mag))
        if not frontier:
            break

    ranked = sorted(found.values(), key=lambda c: (c.weighted, c.depth, c.live))
    return SearchResult(base, ranked, explored, bad_math, bad_mag)
