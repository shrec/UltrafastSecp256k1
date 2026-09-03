"""Token sequences, precedence-parametrised parsing, and expression ASTs.

The core research object of this experiment is the pair

    (token sequence, G)

where G is a grammar -- concretely, an assignment of binding strength to the
binary operators.  A bare token sequence such as

    2 * t + 1 - t / 2

is not yet a mathematical expression.  It becomes one only after a grammar
turns it into an AST.  This module makes G an explicit, enumerable parameter
instead of a hardcoded convention.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Dict, List, Tuple

from .poly import RatFunc

BINARY_OPS = ("+", "-", "*", "/")

_TOKEN_RE = re.compile(
    r"""\s*(?:
          (?P<num>\d+\.\d+|\d+)
        | (?P<name>[A-Za-z_]\w*)
        | (?P<pow>\*\*|\^)
        | (?P<op>[-+*/])
        | (?P<lp>\()
        | (?P<rp>\))
    )""",
    re.VERBOSE,
)


class Token:
    __slots__ = ("kind", "value")

    def __init__(self, kind: str, value=None):
        self.kind = kind
        self.value = value

    def __repr__(self):
        return "Token(%s,%r)" % (self.kind, self.value)


def tokenize(source: str) -> List[Token]:
    pos = 0
    out: List[Token] = []
    while pos < len(source):
        if source[pos].isspace():
            pos += 1
            continue
        m = _TOKEN_RE.match(source, pos)
        if not m:
            raise ValueError("cannot tokenize at offset %d: %r" % (pos, source[pos:]))
        pos = m.end()
        if m.group("num") is not None:
            out.append(Token("num", Fraction(m.group("num"))))
        elif m.group("name") is not None:
            out.append(Token("name", m.group("name")))
        elif m.group("pow") is not None:
            out.append(Token("pow"))
        elif m.group("op") is not None:
            out.append(Token("op", m.group("op")))
        elif m.group("lp") is not None:
            out.append(Token("lp"))
        else:
            out.append(Token("rp"))
    return out


# -- AST -------------------------------------------------------------------

class Node:
    __slots__ = ()

    def to_ratfunc(self) -> RatFunc:
        raise NotImplementedError

    def op_counts(self) -> Dict[str, int]:
        counts = {"+": 0, "-": 0, "*": 0, "/": 0, "neg": 0, "pow": 0}
        self._count_into(counts)
        return counts

    def _count_into(self, counts):
        pass

    def depth(self) -> int:
        return 1

    def size(self) -> int:
        return 1


class Num(Node):
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = Fraction(value)

    def to_ratfunc(self):
        return RatFunc.const(self.value)

    def __str__(self):
        return str(self.value)


class Var(Node):
    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name

    def to_ratfunc(self):
        return RatFunc.var(self.name)

    def __str__(self):
        return self.name


class Neg(Node):
    __slots__ = ("operand",)

    def __init__(self, operand):
        self.operand = operand

    def to_ratfunc(self):
        return -self.operand.to_ratfunc()

    def _count_into(self, counts):
        counts["neg"] += 1
        self.operand._count_into(counts)

    def depth(self):
        return 1 + self.operand.depth()

    def size(self):
        return 1 + self.operand.size()

    def __str__(self):
        return "-(%s)" % self.operand


class Pow(Node):
    __slots__ = ("base", "exponent")

    def __init__(self, base, exponent: int):
        self.base = base
        self.exponent = exponent

    def to_ratfunc(self):
        return self.base.to_ratfunc().power(self.exponent)

    def _count_into(self, counts):
        counts["pow"] += 1
        self.base._count_into(counts)

    def depth(self):
        return 1 + self.base.depth()

    def size(self):
        return 1 + self.base.size()

    def __str__(self):
        return "(%s)^%d" % (self.base, self.exponent)


class BinOp(Node):
    __slots__ = ("op", "left", "right")

    def __init__(self, op: str, left: Node, right: Node):
        self.op = op
        self.left = left
        self.right = right

    def to_ratfunc(self):
        a = self.left.to_ratfunc()
        b = self.right.to_ratfunc()
        if self.op == "+":
            return a + b
        if self.op == "-":
            return a - b
        if self.op == "*":
            return a * b
        if self.op == "/":
            return a / b
        raise ValueError("unknown operator %r" % self.op)

    def _count_into(self, counts):
        counts[self.op] += 1
        self.left._count_into(counts)
        self.right._count_into(counts)

    def depth(self):
        return 1 + max(self.left.depth(), self.right.depth())

    def size(self):
        return 1 + self.left.size() + self.right.size()

    def __str__(self):
        return "(%s %s %s)" % (self.left, self.op, self.right)


# -- precedence-parametrised parser ----------------------------------------

class Parser:
    """Precedence-climbing parser where the precedence table is an argument.

    `precedence` maps each binary operator to an integer binding strength;
    larger binds more tightly.  All binary operators are left-associative;
    `^`/`**` is right-associative and always binds tighter than every binary
    operator; unary minus binds tighter still.  Parentheses always win, so a
    fully-parenthesised source is grammar-invariant by construction.
    """

    def __init__(self, tokens: List[Token], precedence: Dict[str, int]):
        self.tokens = tokens
        self.pos = 0
        self.precedence = precedence

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def next(self):
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse(self) -> Node:
        node = self.parse_binary(min_level=1)
        if self.pos != len(self.tokens):
            raise ValueError("trailing tokens at %d" % self.pos)
        return node

    def parse_binary(self, min_level: int) -> Node:
        left = self.parse_power()
        while True:
            tok = self.peek()
            if tok is None or tok.kind != "op":
                break
            level = self.precedence[tok.value]
            if level < min_level:
                break
            self.next()
            right = self.parse_binary(level + 1)  # left-associative
            left = BinOp(tok.value, left, right)
        return left

    def parse_power(self) -> Node:
        base = self.parse_unary()
        tok = self.peek()
        if tok is not None and tok.kind == "pow":
            self.next()
            exp_tok = self.next()
            if exp_tok.kind != "num" or exp_tok.value.denominator != 1:
                raise ValueError("only integer exponents are supported")
            return Pow(base, int(exp_tok.value))
        return base

    def parse_unary(self) -> Node:
        tok = self.peek()
        if tok is not None and tok.kind == "op" and tok.value == "-":
            self.next()
            return Neg(self.parse_power())
        return self.parse_atom()

    def parse_atom(self) -> Node:
        tok = self.next()
        if tok.kind == "num":
            return Num(tok.value)
        if tok.kind == "name":
            return Var(tok.value)
        if tok.kind == "lp":
            node = self.parse_binary(min_level=1)
            closing = self.next()
            if closing.kind != "rp":
                raise ValueError("expected ')'")
            return node
        raise ValueError("unexpected token %r" % tok)


def parse(source: str, precedence: Dict[str, int]) -> Node:
    return Parser(tokenize(source), precedence).parse()
