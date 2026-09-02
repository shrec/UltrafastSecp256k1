"""Limb-level field-multiply schedules, transcribed and generated.

`production_mul` is the schedule this engine actually runs on x86-64,
transcribed operation for operation from fe52_mul_inner in
src/cpu/include/secp256k1/field_52_impl.hpp (which is libsecp256k1's 5x52
schedule). It is the reference every generated schedule is checked against --
and it is itself checked against exact integer arithmetic mod p, so a
transcription error cannot go unnoticed.
"""

from __future__ import annotations

from typing import List

from .modarith import M52, R52, Op, Schedule


def production_mul() -> Schedule:
    """fe52_mul_inner: 25 limb products + 4 reduction folds, five columns.

    The column order is 3, 4, 0, 1, 2 -- NOT 0..4. Columns 5..8 are folded into
    the low columns as they are produced, so only two 128-bit accumulators (c
    and d) are ever live. That ordering is the representation choice; the
    arithmetic identity it implements is

        a*b == sum_{k<5} c_k 2^(52k) + R52 * sum_{k>=5} c_k 2^(52(k-5))  (mod p)
    """
    o: List[Op] = []
    A = ["a%d" % i for i in range(5)]
    B = ["b%d" % i for i in range(5)]

    # -- column 3 + reduced column 8 --------------------------------------
    o.append(Op("mulw", "d", (A[0], B[3])))
    for i, j in ((1, 2), (2, 1), (3, 0)):
        o.append(Op("accm", "d", (A[i], B[j])))
    o.append(Op("mulw", "c", (A[4], B[4])))
    o.append(Op("lo64", "clo", ("c",)))
    o.append(Op("accc", "d", ("clo",), R52))
    o.append(Op("shr", "c", ("c",), 64))
    o.append(Op("lo52", "t3", ("d",)))
    o.append(Op("shr", "d", ("d",), 52))

    # -- column 4 + column 8 carry ----------------------------------------
    for i, j in ((0, 4), (1, 3), (2, 2), (3, 1), (4, 0)):
        o.append(Op("accm", "d", (A[i], B[j])))
    o.append(Op("lo64", "clo2", ("c",)))
    o.append(Op("accc", "d", ("clo2",), R52 << 12))
    o.append(Op("lo52", "t4", ("d",)))
    o.append(Op("shr", "d", ("d",), 52))
    o.append(Op("shr", "tx", ("t4",), 48))
    o.append(Op("lo48", "t4", ("t4",)))

    # -- column 0 + reduced column 5 --------------------------------------
    o.append(Op("mulw", "c", (A[0], B[0])))
    for i, j in ((1, 4), (2, 3), (3, 2), (4, 1)):
        o.append(Op("accm", "d", (A[i], B[j])))
    o.append(Op("lo52", "u0", ("d",)))
    o.append(Op("shr", "d", ("d",), 52))
    o.append(Op("shl", "u0s", ("u0",), 4))
    o.append(Op("or64", "u0f", ("u0s", "tx")))
    o.append(Op("accc", "c", ("u0f",), R52 >> 4))
    o.append(Op("lo52", "r0", ("c",)))
    o.append(Op("shr", "c", ("c",), 52))

    # -- column 1 + reduced column 6 --------------------------------------
    for i, j in ((0, 1), (1, 0)):
        o.append(Op("accm", "c", (A[i], B[j])))
    for i, j in ((2, 4), (3, 3), (4, 2)):
        o.append(Op("accm", "d", (A[i], B[j])))
    o.append(Op("lo52", "dm", ("d",)))
    o.append(Op("accc", "c", ("dm",), R52))
    o.append(Op("shr", "d", ("d",), 52))
    o.append(Op("lo52", "r1", ("c",)))
    o.append(Op("shr", "c", ("c",), 52))

    # -- column 2 + reduced column 7 --------------------------------------
    for i, j in ((0, 2), (1, 1), (2, 0)):
        o.append(Op("accm", "c", (A[i], B[j])))
    for i, j in ((3, 4), (4, 3)):
        o.append(Op("accm", "d", (A[i], B[j])))
    o.append(Op("lo64", "dlo", ("d",)))
    o.append(Op("accc", "c", ("dlo",), R52))
    o.append(Op("shr", "d", ("d",), 64))
    o.append(Op("lo52", "r2", ("c",)))
    o.append(Op("shr", "c", ("c",), 52))

    # -- finalize columns 3 and 4 -----------------------------------------
    o.append(Op("lo64", "dlo2", ("d",)))
    o.append(Op("accc", "c", ("dlo2",), R52 << 12))
    o.append(Op("accv", "c", ("t3",)))
    o.append(Op("lo52", "r3", ("c",)))
    o.append(Op("shr", "c", ("c",), 52))
    o.append(Op("accv", "c", ("t4",)))
    o.append(Op("lo64", "r4", ("c",)))

    return Schedule("production_mul", o, ["r0", "r1", "r2", "r3", "r4"],
                    "fe52_mul_inner, column order 3,4,0,1,2, two live accumulators")


def wide_mul() -> Schedule:
    """A deliberately different SCHEDULE for the same arithmetic.

    Production interleaves the reduction with the column accumulation so that
    only two 128-bit accumulators are ever live.  This one does the opposite:
    accumulate all nine columns independently first (maximum ILP, nine live
    accumulators), then carry-propagate, then fold, then carry-propagate again.

    Same identity, opposite point on the register-pressure / parallelism
    trade-off.  Whether that trade is worth taking is a measurement question,
    which is the point of expressing it as a schedule rather than arguing it.
    """
    o: List[Op] = []
    A = ["a%d" % i for i in range(5)]
    B = ["b%d" % i for i in range(5)]

    # -- phase 1: nine independent column accumulators --------------------
    # Each column holds at most five products of 52-bit values:
    # 5 * (2^52-1)^2 < 2^107, so a 128-bit accumulator has 21 bits spare and
    # every column can be built without any inter-column dependency at all.
    for k in range(9):
        terms = [(i, k - i) for i in range(5) if 0 <= k - i < 5]
        first = True
        for i, j in terms:
            o.append(Op("mulw" if first else "accm", "c%d" % k, (A[i], B[j])))
            first = False

    # -- phase 2: carry-propagate into ten 52-bit digits -------------------
    # carry <= 2^107 / 2^52 = 2^55, so it stays inside a u64.
    for k in range(9):
        if k:
            o.append(Op("accv", "c%d" % k, ("k%d" % (k - 1),)))
        o.append(Op("lo52", "d%d" % k, ("c%d" % k,)))
        o.append(Op("shr", "c%d" % k, ("c%d" % k,), 52))
        o.append(Op("lo64", "k%d" % k, ("c%d" % k,)))
    o.append(Op("lo64", "d9", ("k8",)))

    # -- phase 3: fold the high half -------------------------------------
    # 2^(52k) == 2^(52(k-5)) * 2^260 == 2^(52(k-5)) * R52  (mod p) for k >= 5,
    # so digit k lands on digit k-5 scaled by R52.
    for k in range(5):
        o.append(Op("mulc", "e%d" % k, ("d%d" % (k + 5),), R52))
        o.append(Op("accv", "e%d" % k, ("d%d" % k,)))

    # -- phase 4: carry-propagate again ----------------------------------
    for k in range(5):
        if k:
            o.append(Op("accv", "e%d" % k, ("g%d" % (k - 1),)))
        o.append(Op("lo52", "f%d" % k, ("e%d" % k,)))
        o.append(Op("shr", "e%d" % k, ("e%d" % k,), 52))
        o.append(Op("lo64", "g%d" % k, ("e%d" % k,)))

    # -- phase 5: the carry out of digit 4 folds back to digit 0 ----------
    # It is at most a few bits, so one more short propagation closes it.
    o.append(Op("mulc", "h0", ("g4",), R52))
    o.append(Op("accv", "h0", ("f0",)))
    o.append(Op("lo52", "r0", ("h0",)))
    o.append(Op("shr", "h0", ("h0",), 52))
    o.append(Op("lo64", "hc", ("h0",)))

    o.append(Op("accv", "h1", ("hc",)))
    o.append(Op("accv", "h1", ("f1",)))
    o.append(Op("lo52", "r1", ("h1",)))
    o.append(Op("shr", "h1", ("h1",), 52))
    o.append(Op("lo64", "hc1", ("h1",)))

    o.append(Op("accv", "h2", ("hc1",)))
    o.append(Op("accv", "h2", ("f2",)))
    o.append(Op("lo52", "r2", ("h2",)))
    o.append(Op("shr", "h2", ("h2",), 52))
    o.append(Op("lo64", "hc2", ("h2",)))

    o.append(Op("accv", "h3", ("hc2",)))
    o.append(Op("accv", "h3", ("f3",)))
    o.append(Op("lo52", "r3", ("h3",)))
    o.append(Op("shr", "h3", ("h3",), 52))
    o.append(Op("lo64", "hc3", ("h3",)))

    o.append(Op("accv", "r4x", ("hc3",)))
    o.append(Op("accv", "r4x", ("f4",)))
    o.append(Op("lo64", "r4", ("r4x",)))

    return Schedule("wide_mul", o, ["r0", "r1", "r2", "r3", "r4"],
                    "nine independent column accumulators, fold after; max ILP, max registers")
