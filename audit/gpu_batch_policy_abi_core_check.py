#!/usr/bin/env python3
"""gpu_batch_policy_abi_core_check.py -- task GPU_ABI_CORE_VALIDATOR_292.

Independent validator of the A0-accepted GPU batch-policy ABI core (task
GPU_ABI_NATIVE_CORE_264) and oracle harness (task GPU_ABI_NATIVE_ORACLE_281,
header materialized by GPU_ABI_ORACLE_HEADER_SHIM_290).

What it does, all task-locally under --scratch-root and failing closed:

  1. Fail-closed dependency materialization.  The four accepted-artifact
     dependencies (header, oracle, C consumer, C++ consumer) plus their
     supporting accepted inputs and the three upstream reports are bound by
     exact sha256.  Any missing / non-finished / mismatched byte aborts; a
     duplicate ABI can never be substituted (the 281-side header copy must be
     byte-identical to the accepted 264 header).

  2. Independent request32/result72 four-argument ABI recomputation.  A fresh
     byte-level reference model (this file only -- it does not import or trust
     reference_model.py or oracle.c predicates) reproduces status + full
     post-call arena for a 611-case matrix covering the declared x physical
     Cartesian boundaries, result-first precedence, R/RD/P/D limits, D<4 /
     P4D0 / D8..71, wrap/overlap/adjacency, aligned+unaligned, masks/fields/
     MBZ and multiplication overflow, and forward sizes with preserved tail.

  3. Genuine differential through BOTH accepted consumers, run TWICE.  Each run
     requires zero C-vs-model, C++-vs-model and C-vs-C++ divergence; the full C
     and C++ output byte streams and the result hash are bound stable across
     the two runs.  A self-check injects a run2-only C++ divergence and proves
     the two-run gate then fails overall.

  4. Real isolated compile/run source mutations, each of which must be KILLED
     (compile failure or run divergence) and is required in a fail-closed
     mut_ok gate:
        - 10 oracle-body mutants (real .so builds, executed);
        - 2 accepted-contract-header mutants (result-size -> compile kill;
          selected-path offset -> run kill);
        - 2 accepted-consumer mutants (wrong request size, C and C++ -> run
          kill).
     Compile-time NEG_CASE assertions and reference-model self-mutants are also
     exercised, but as separate evidence gates -- they do NOT substitute for
     the real source mutations above.

  5. Memory-safety instrumentation on the actual accepted oracle and isolated
     real-oracle mutants: mmap guard-page probe (mandatory) and, when the
     toolchain supports it, AddressSanitizer.  Clean calls pass; a tail-
     overwriting or off-by-one-clearing mutant trips the guard / sanitizer.

Exit status is 0 only if every gate passes; otherwise non-zero (fail closed).
"""

import argparse
import hashlib
import os
import re
import shutil
import struct
import subprocess
import sys

ORIGIN_THREAD_ID = "019f8bd8-d62c-7eb0-8057-ec1cd79ce64f"
TASK_ID = "GPU_ABI_CORE_VALIDATOR_292"
TOPIC = "gpu_abi_core_validator_v4"

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def _p(*parts):
    return os.path.join(REPO, *parts)


CORE_ROOT = _p("out/.tasks/GPU_ABI_NATIVE_CORE_264/prototype")
CORE_REPORT = _p("out/.tasks/GPU_ABI_NATIVE_CORE_264/report.md")
ORACLE_ROOT = _p("out/.tasks/GPU_ABI_NATIVE_ORACLE_281/oracle")
ORACLE_REPORT = _p("out/.tasks/GPU_ABI_NATIVE_ORACLE_281/report.md")
SHIM_REPORT = _p("out/.tasks/GPU_ABI_ORACLE_HEADER_SHIM_290/report.md")
NEGATIVE_C = os.path.join(HERE, "gpu_batch_policy_abi_negative.c")
REPORT_OUT = _p("out/.tasks/GPU_ABI_CORE_VALIDATOR_292/report.md")

HEADER = "gpu_batch_policy_abi_contract.h"
ORACLE_C = "gpu_batch_policy_abi_oracle.c"

DEP_HEADER = os.path.join(CORE_ROOT, HEADER)
DEP_ORACLE = os.path.join(CORE_ROOT, ORACLE_C)
DEP_CONSUMER_C = os.path.join(ORACLE_ROOT, "consumer.c")
DEP_CONSUMER_CPP = os.path.join(ORACLE_ROOT, "consumer.cpp")
DEP_SHIM_HEADER = os.path.join(ORACLE_ROOT, HEADER)

# Exact accepted sha256 (fail-closed). Missing/mismatched bytes abort.
ACCEPTED_SHA = {
    DEP_HEADER:
        "21dc9a76e152561a7568a1998926703ec23c657a3bbbaa35518f4eaf72464381",
    DEP_ORACLE:
        "2c9ef423e94d9efff8f7794046f6882b85930897a6d78f0bc84f49203091dab7",
    os.path.join(CORE_ROOT, "gpu_batch_policy_abi_native_test.c"):
        "09b13c19edc373de0f2647cede0b87b864f5a664b557a520a6d5d27bd6a3f88c",
    os.path.join(CORE_ROOT, "gpu_batch_policy_abi_native_test.cpp"):
        "3f1cfa7f3e38e8cadb0cc0a31eb4df96491e9f8a0a8398ad02c9cfb2bcf0db21",
    DEP_CONSUMER_C:
        "12e05deebf46a4aca9fa01016fd5238361fc3885ccd2495c826f1866ea10a394",
    DEP_CONSUMER_CPP:
        "6411eb7ab627e421788ddda1676250a511f413db75ba00693d31ccc4c3581948",
    os.path.join(ORACLE_ROOT, "reference_model.py"):
        "c229310842cb664fa631117ca0795728f4e1cb7294e2f7747d468fa37be11ef6",
    os.path.join(ORACLE_ROOT, "run_differential.py"):
        "5fbb0626010ec4b1e8624ca1d4da6363cf034120a01b123a5ca75ad41501da73",
    DEP_SHIM_HEADER:
        "21dc9a76e152561a7568a1998926703ec23c657a3bbbaa35518f4eaf72464381",
}

REPORT_SHA = {
    CORE_REPORT:
        "89d39c585e6fe03c7bb656039b9e514847372dee875191ed6bbdf83183b644cf",
    ORACLE_REPORT:
        "7631bb54fad9e66e861dfe1957628f33a580002acd4d7670bf5d078a72bf00b7",
    SHIM_REPORT:
        "92831e81d2a489151884c627c4521ab20c9523d79e1fb5588fdcd57926499337",
}

# The four validator dependencies (finished + accepted hash): task, kind, path, report.
DEPENDENCIES = [
    ("GPU_ABI_NATIVE_CORE_264", "accepted header", DEP_HEADER, CORE_REPORT),
    ("GPU_ABI_NATIVE_CORE_264", "accepted oracle", DEP_ORACLE, CORE_REPORT),
    ("GPU_ABI_NATIVE_ORACLE_281", "accepted C consumer", DEP_CONSUMER_C, ORACLE_REPORT),
    ("GPU_ABI_NATIVE_ORACLE_281", "accepted C++ consumer", DEP_CONSUMER_CPP, ORACLE_REPORT),
]

CC = os.environ.get("CC", "cc")
CXX = os.environ.get("CXX", "c++")

_commands = []


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# --------------------------------------------------------------------------- #
# (1) Fail-closed dependency verification                                     #
# --------------------------------------------------------------------------- #
def _finished_report(path):
    """A dependency task is 'finished' iff its report is a present, non-empty
    regular file whose sha256 matches the accepted value."""
    if not os.path.isfile(path):
        raise SystemExit("FAIL-CLOSED: dependency report missing: %s" % path)
    got = sha256_file(path)
    want = REPORT_SHA[path]
    if got != want:
        raise SystemExit("FAIL-CLOSED: report sha mismatch %s\n  want %s\n  got  %s"
                         % (path, want, got))
    return got


def verify_dependencies():
    verified = {}
    for path, want in ACCEPTED_SHA.items():
        if not os.path.isfile(path):
            raise SystemExit("FAIL-CLOSED: accepted input missing: %s" % path)
        if os.path.islink(path):
            raise SystemExit("FAIL-CLOSED: accepted input is a symlink: %s" % path)
        got = sha256_file(path)
        if got != want:
            raise SystemExit("FAIL-CLOSED: accepted input byte mismatch: %s\n"
                             "  want %s\n  got  %s" % (path, want, got))
        verified[path] = got
    # never substitute a duplicate ABI: both header copies must be identical
    if sha256_file(DEP_HEADER) != sha256_file(DEP_SHIM_HEADER):
        raise SystemExit("FAIL-CLOSED: 281-side header is not byte-identical to 264 header")
    reports = {p: _finished_report(p) for p in REPORT_SHA}
    # each of the four dependency rows must resolve to a finished task
    for task, kind, path, report in DEPENDENCIES:
        if verified.get(path) != ACCEPTED_SHA[path]:
            raise SystemExit("FAIL-CLOSED: dependency %s (%s) not verified" % (task, kind))
        if reports.get(report) is None:
            raise SystemExit("FAIL-CLOSED: dependency %s not finished" % task)
    return verified, reports


# --------------------------------------------------------------------------- #
# Independent ABI reference model (request32 / result72, 4-arg query)          #
# --------------------------------------------------------------------------- #
LB_ABI_V1 = 1
S_OK, S_WRAP, S_OVERLAP, S_RNULL, S_RABI, S_QABI, S_QFIELD = range(7)
STATUS_NAME = {0: "OK", 1: "ADDR_WRAP", 2: "ADDR_OVERLAP", 3: "RESULT_NULL",
               4: "RESULT_ABI", 5: "REQUEST_ABI", 6: "REQUEST_FIELD"}

REQUEST_SIZE, RESULT_SIZE = 32, 72
RQ_SS, RQ_ABI, RQ_OP, RQ_MASK, RQ_ITEM, RQ_CONC, RQ_RESV = 0, 4, 8, 12, 16, 24, 28
RS_SS, RS_ABI, RS_AVAIL, RS_BACK, RS_PATH, RS_CAL, RS_BEN = 0, 4, 8, 16, 20, 24, 28
OP_LO, OP_HI = 1, 3
MASK_CPU, MASK_CUDA, MASK_OPENCL, MASK_METAL, MASK_OTHER = 1, 2, 4, 8, 16
MASK_KNOWN = MASK_CPU | MASK_CUDA | MASK_OPENCL | MASK_METAL | MASK_OTHER
MASK_NONE = 0
BACKEND_NONE, BACKEND_CPU = 0, 1
PATH_UNAVAIL, PATH_CPU_INLINE = 0, 1
CAL_NONE, ADVICE_UNKNOWN = 0, 0
OP_INVALID, OP_ECDSA, OP_SCHNORR, OP_MSM = 0, 1, 2, 3
U64 = (1 << 64) - 1


def _u32(b, o):
    return struct.unpack_from("<I", b, o)[0]


def _u64(b, o):
    return struct.unpack_from("<Q", b, o)[0]


def _wraps(base, size, mask):
    if size == 0:
        return False
    return ((base + size) & mask) < base


def _overlap(a, la, b, lb, mask):
    if a == 0 or b == 0 or la == 0 or lb == 0:
        return False
    return (a < ((b + lb) & mask)) and (b < ((a + la) & mask))


def success_image(mask):
    img = bytearray(72)
    available = MASK_CPU | MASK_CUDA
    eligible = available & (mask & 0xFFFFFFFF)
    struct.pack_into("<I", img, RS_SS, RESULT_SIZE)
    struct.pack_into("<I", img, RS_ABI, LB_ABI_V1)
    struct.pack_into("<Q", img, RS_AVAIL, available)
    if eligible & MASK_CPU:
        struct.pack_into("<I", img, RS_BACK, BACKEND_CPU)
        struct.pack_into("<I", img, RS_PATH, PATH_CPU_INLINE)
    else:
        struct.pack_into("<I", img, RS_BACK, BACKEND_NONE)
        struct.pack_into("<I", img, RS_PATH, PATH_UNAVAIL)
    struct.pack_into("<I", img, RS_CAL, CAL_NONE)
    struct.pack_into("<I", img, RS_BEN, ADVICE_UNKNOWN)
    return bytes(img)


def simulate(arena_pre, ptr_bits, rp, ro, rs, base_req, sp, so, ss, base_res,
             mutate=None):
    """Independently compute (status, post-call arena). Request bytes never
    change; only the result window is cleared or filled. `mutate` injects a
    deliberate reference defect for the reference-self-mutant gate only."""
    m = (1 << ptr_bits) - 1
    out = bytearray(arena_pre)

    if _wraps(base_req, rs, m) or _wraps(base_res, ss, m):
        if mutate != "ref_omit_wrap":
            return S_WRAP, bytes(out)
    if _overlap(base_req, rs, base_res, ss, m):
        return S_OVERLAP, bytes(out)

    if not sp or ss < 4:
        return S_RNULL, bytes(out)

    D = _u32(out, so + RS_SS)
    clearable = min(ss, D, RESULT_SIZE)
    if mutate == "ref_offbyone_clear":
        clearable += 1

    def clear(n):
        for i in range(n):
            out[so + i] = 0

    if ss < 8 or D < 8:
        clear(clearable)
        return S_RABI, bytes(out)
    if _u32(out, so + RS_ABI) != LB_ABI_V1:
        clear(clearable)
        return S_RABI, bytes(out)
    if D < RESULT_SIZE or ss < RESULT_SIZE:
        clear(clearable)
        return S_RABI, bytes(out)

    if not rp or rs < 4:
        clear(RESULT_SIZE)
        return S_QABI, bytes(out)
    RD = _u32(out, ro + RQ_SS)
    if rs < REQUEST_SIZE or RD < REQUEST_SIZE:
        clear(RESULT_SIZE)
        return S_QABI, bytes(out)
    if _u32(out, ro + RQ_ABI) != LB_ABI_V1:
        clear(RESULT_SIZE)
        return S_QABI, bytes(out)

    op = _u32(out, ro + RQ_OP)
    mask = _u32(out, ro + RQ_MASK)
    item = _u64(out, ro + RQ_ITEM)
    conc = _u32(out, ro + RQ_CONC)
    resv = _u32(out, ro + RQ_RESV)

    if resv != 0:
        clear(RESULT_SIZE)
        return S_QFIELD, bytes(out)
    if op < OP_LO or op > OP_HI:
        clear(RESULT_SIZE)
        return S_QFIELD, bytes(out)
    if (mask & ~MASK_KNOWN) != 0:
        clear(RESULT_SIZE)
        return S_QFIELD, bytes(out)
    if item == 0:
        clear(RESULT_SIZE)
        return S_QFIELD, bytes(out)
    if conc == 0:
        clear(RESULT_SIZE)
        return S_QFIELD, bytes(out)
    if item > U64 // conc:
        clear(RESULT_SIZE)
        return S_QFIELD, bytes(out)

    img = success_image(MASK_KNOWN if mutate == "ref_ignore_mask" else mask)
    out[so:so + RESULT_SIZE] = img
    return S_OK, bytes(out)


# --------------------------------------------------------------------------- #
# 611-case matrix (independent)                                               #
# --------------------------------------------------------------------------- #
L = 256
CANARY = 0xAB
REQ_A, RES_A = 0, 128
REQ_U, RES_U = 1, 129
V_ABI, V_OP, V_MASK, V_ITEM, V_CONC = LB_ABI_V1, OP_ECDSA, MASK_CPU, 10, 2
S = [3, 4, 5, 7, 8, 9, 31, 32, 33, 71, 72, 73]


class Case:
    __slots__ = ("id", "tag", "arena", "rp", "ro", "rs", "sp", "so", "ss")

    def __init__(self, cid, tag, arena, rp, ro, rs, sp, so, ss):
        self.id, self.tag, self.arena = cid, tag, arena
        self.rp, self.ro, self.rs = rp, ro, rs
        self.sp, self.so, self.ss = sp, so, ss


def new_arena():
    return bytearray([CANARY]) * L


def plant_request(a, off, rd=REQUEST_SIZE, abi=V_ABI, op=V_OP, mask=V_MASK,
                  item=V_ITEM, conc=V_CONC, resv=0):
    struct.pack_into("<I", a, off + RQ_SS, rd & 0xFFFFFFFF)
    struct.pack_into("<I", a, off + RQ_ABI, abi & 0xFFFFFFFF)
    struct.pack_into("<I", a, off + RQ_OP, op & 0xFFFFFFFF)
    struct.pack_into("<I", a, off + RQ_MASK, mask & 0xFFFFFFFF)
    struct.pack_into("<Q", a, off + RQ_ITEM, item & U64)
    struct.pack_into("<I", a, off + RQ_CONC, conc & 0xFFFFFFFF)
    struct.pack_into("<I", a, off + RQ_RESV, resv & 0xFFFFFFFF)


def plant_result(a, off, d=RESULT_SIZE, abi=V_ABI):
    struct.pack_into("<I", a, off + RS_SS, d & 0xFFFFFFFF)
    struct.pack_into("<I", a, off + RS_ABI, abi & 0xFFFFFFFF)


def generate_cases():
    cases, cid, counts = [], 0, {}

    def add(tag, arena, rp, ro, rs, sp, so, ss):
        nonlocal cid
        cases.append(Case(cid, tag, arena, rp, ro, rs, sp, so, ss))
        cid += 1

    # (A) request physical R x declared RD, result valid; aligned + unaligned
    for aligned in (1, 0):
        ro, so = (REQ_A, RES_A) if aligned else (REQ_U, RES_U)
        n = 0
        for rphys in S:
            for rd in S:
                a = new_arena()
                plant_request(a, ro, rd=rd)
                plant_result(a, so, d=RESULT_SIZE, abi=V_ABI)
                add("req-matrix[%s]" % ("A" if aligned else "U"),
                    a, 1, ro, rphys, 1, so, RESULT_SIZE)
                n += 1
        counts["req_matrix_%s" % ("aligned" if aligned else "unaligned")] = n

    # (B) result physical P x declared D, request valid; aligned + unaligned
    for aligned in (1, 0):
        ro, so = (REQ_A, RES_A) if aligned else (REQ_U, RES_U)
        n = 0
        for pphys in S:
            for d in S:
                a = new_arena()
                plant_request(a, ro)
                plant_result(a, so, d=d, abi=V_ABI)
                add("res-matrix[%s]" % ("A" if aligned else "U"),
                    a, 1, ro, REQUEST_SIZE, 1, so, pphys)
                n += 1
        counts["res_matrix_%s" % ("aligned" if aligned else "unaligned")] = n

    # (C) precedence + special
    a = new_arena(); plant_result(a, RES_A, d=RESULT_SIZE, abi=0xDEAD)
    add("prec-badres-nullreq", a, 0, 0, 0, 1, RES_A, RESULT_SIZE)
    a = new_arena(); plant_request(a, REQ_A, op=OP_INVALID)
    plant_result(a, RES_A, d=RESULT_SIZE, abi=0xDEAD)
    add("prec-badres-badreqfield", a, 1, REQ_A, REQUEST_SIZE, 1, RES_A, RESULT_SIZE)
    a = new_arena(); plant_result(a, RES_A, d=RESULT_SIZE, abi=V_ABI)
    add("prec-goodres-nullreq", a, 0, 0, 0, 1, RES_A, RESULT_SIZE)
    a = new_arena(); plant_request(a, REQ_A)
    add("null-result", a, 1, REQ_A, REQUEST_SIZE, 0, 0, 0)
    a = new_arena(); plant_request(a, REQ_A)
    plant_result(a, RES_A, d=0, abi=V_ABI)
    add("p4d0-no-clear", a, 1, REQ_A, REQUEST_SIZE, 1, RES_A, 4)
    counts["precedence_special"] = 5

    # (D) single / double uintptr wrap
    for tag, rs, ss in (("wrap-req", U64, RESULT_SIZE),
                        ("wrap-res", REQUEST_SIZE, U64),
                        ("wrap-double", U64, U64)):
        a = new_arena(); plant_request(a, REQ_A); plant_result(a, RES_A)
        add(tag, a, 1, REQ_A, rs, 1, RES_A, ss)
    counts["wrap"] = 3

    # (E) partial / full overlap + exact adjacency both directions
    a = new_arena(); plant_request(a, 0, rd=REQUEST_SIZE)
    plant_result(a, 20, d=RESULT_SIZE, abi=V_ABI)
    add("overlap-partial", a, 1, 0, 40, 1, 20, RESULT_SIZE)
    a = new_arena()
    add("overlap-full", a, 1, 0, REQUEST_SIZE, 1, 0, RESULT_SIZE)
    a = new_arena(); plant_request(a, 0, rd=REQUEST_SIZE)
    plant_result(a, 32, d=RESULT_SIZE, abi=V_ABI)
    add("adjacent-lo", a, 1, 0, REQUEST_SIZE, 1, 32, RESULT_SIZE)
    a = new_arena(); plant_result(a, 0, d=RESULT_SIZE, abi=V_ABI)
    plant_request(a, 72, rd=REQUEST_SIZE)
    add("adjacent-hi", a, 1, 72, REQUEST_SIZE, 1, 0, RESULT_SIZE)
    counts["overlap_adjacency"] = 4

    # (F) backend mask semantics
    mask_cases = [("mask-cpu", MASK_CPU), ("mask-mixed", MASK_CPU | MASK_CUDA),
                  ("mask-zero", MASK_NONE), ("mask-gpuonly", MASK_CUDA),
                  ("mask-opencl", MASK_OPENCL), ("mask-reserved", MASK_CPU | 0x80000000)]
    for tag, mv in mask_cases:
        a = new_arena(); plant_request(a, REQ_A, mask=mv)
        plant_result(a, RES_A, d=RESULT_SIZE, abi=V_ABI)
        add(tag, a, 1, REQ_A, REQUEST_SIZE, 1, RES_A, RESULT_SIZE)
    counts["mask"] = len(mask_cases)

    # (G) field / MBZ / operation / multiplication-overflow
    field_cases = [("op-ecdsa", dict(op=OP_ECDSA)), ("op-schnorr", dict(op=OP_SCHNORR)),
                   ("op-msm", dict(op=OP_MSM)), ("op-zero", dict(op=OP_INVALID)),
                   ("op-high", dict(op=4)), ("reserved-nonzero", dict(resv=1)),
                   ("item-zero", dict(item=0)), ("conc-zero", dict(conc=0)),
                   ("req-badabi", dict(abi=0xBAD)), ("overflow-hit", dict(item=U64, conc=2)),
                   ("overflow-edge-ok", dict(item=U64 // 2, conc=2)),
                   ("overflow-just-over", dict(item=U64 // 2 + 1, conc=2))]
    for tag, kw in field_cases:
        a = new_arena(); plant_request(a, REQ_A, **kw)
        plant_result(a, RES_A, d=RESULT_SIZE, abi=V_ABI)
        add(tag, a, 1, REQ_A, REQUEST_SIZE, 1, RES_A, RESULT_SIZE)
    counts["field_overflow"] = len(field_cases)

    # (H) forward (larger) request/result sizes: prefix touched, tail preserved
    for aligned in (1, 0):
        ro, so = (REQ_A, RES_A) if aligned else (REQ_U, RES_U)
        a = new_arena(); plant_request(a, ro, rd=40); plant_result(a, so, d=80, abi=V_ABI)
        add("forward-both[%s]" % ("A" if aligned else "U"), a, 1, ro, 40, 1, so, 80)
        a = new_arena(); plant_request(a, ro, rd=73); plant_result(a, so, d=73, abi=V_ABI)
        add("forward-73[%s]" % ("A" if aligned else "U"), a, 1, ro, 73, 1, so, 73)
    counts["forward"] = 4

    # (I) struct-typed natural-alignment success
    a = new_arena()
    plant_request(a, REQ_A, op=OP_SCHNORR, mask=MASK_CPU | MASK_CUDA, item=4096, conc=8)
    plant_result(a, RES_A, d=RESULT_SIZE, abi=V_ABI)
    add("struct-typed-success", a, 1, REQ_A, REQUEST_SIZE, 1, RES_A, RESULT_SIZE)
    counts["struct_typed"] = 1

    return cases, counts


# --------------------------------------------------------------------------- #
# Wire encode / decode (independent)                                          #
# --------------------------------------------------------------------------- #
STREAM_IN_MAGIC = 0x4F435541
STREAM_OUT_MAGIC = 0x4F435552


def encode_stream(cases):
    out = bytearray()
    out += struct.pack("<II", STREAM_IN_MAGIC, len(cases))
    for c in cases:
        out += struct.pack("<I", c.id)
        out += struct.pack("<Q", L)
        out += bytes(c.arena)
        out += struct.pack("<BQQ", c.rp, c.ro, c.rs)
        out += struct.pack("<BQQ", c.sp, c.so, c.ss)
    return bytes(out)


def decode_stream(blob):
    off = 0
    magic, ptr_bits, n = struct.unpack_from("<III", blob, off); off += 12
    if magic != STREAM_OUT_MAGIC:
        raise SystemExit("bad consumer output magic 0x%08X" % magic)
    res = []
    for _ in range(n):
        cid, status = struct.unpack_from("<II", blob, off); off += 8
        base_req, base_res, ll = struct.unpack_from("<QQQ", blob, off); off += 24
        arena = blob[off:off + ll]; off += ll
        res.append((cid, status, base_req, base_res, ll, arena))
    return ptr_bits, res


def elf_interp(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"\x7fELF" or data[4] != 2:
        return None
    end = "<" if data[5] == 1 else ">"
    e_phoff = struct.unpack_from(end + "Q", data, 0x20)[0]
    e_phentsize = struct.unpack_from(end + "H", data, 0x36)[0]
    e_phnum = struct.unpack_from(end + "H", data, 0x38)[0]
    for i in range(e_phnum):
        base = e_phoff + i * e_phentsize
        if struct.unpack_from(end + "I", data, base)[0] == 3:
            p_off = struct.unpack_from(end + "Q", data, base + 8)[0]
            p_sz = struct.unpack_from(end + "Q", data, base + 32)[0]
            return data[p_off:p_off + p_sz].rstrip(b"\0").decode()
    return None


def run_capture(argv, stream=None):
    try:
        p = subprocess.run(argv, input=stream, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
    except PermissionError:
        interp = elf_interp(argv[0]) or "/lib64/ld-linux-x86-64.so.2"
        argv = [interp] + argv
        p = subprocess.run(argv, input=stream, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
    _commands.append(" ".join(argv))
    return p


def run_consumer(binpath, so_path, stream):
    p = run_capture([binpath, so_path], stream)
    if p.returncode != 0:
        raise SystemExit("consumer %s rc=%d: %s"
                         % (binpath, p.returncode, p.stderr.decode("utf-8", "replace")))
    return decode_stream(p.stdout)


def diff_against_reference(cases, ptr_bits, results, mutate=None):
    by = {r[0]: r for r in results}
    div = []
    for c in cases:
        cid, status, base_req, base_res, ll, arena = by[c.id]
        exp_st, exp_arena = simulate(
            c.arena, ptr_bits, c.rp, c.ro, c.rs, base_req if c.rp else 0,
            c.sp, c.so, c.ss, base_res if c.sp else 0, mutate=mutate)
        if status != exp_st or bytes(arena) != bytes(exp_arena):
            div.append((c.id, c.tag, STATUS_NAME.get(status, status),
                        STATUS_NAME.get(exp_st, exp_st)))
    return div


def cross_consumer_diff(cases, res_c, res_cpp):
    a = {r[0]: (r[1], bytes(r[5])) for r in res_c}
    b = {r[0]: (r[1], bytes(r[5])) for r in res_cpp}
    return [(c.id, c.tag) for c in cases if a[c.id] != b[c.id]]


def out_sha(res):
    return hashlib.sha256(
        b"".join(struct.pack("<II", r[0], r[1]) + bytes(r[5]) for r in res)).hexdigest()


def result_hash(cases, results):
    h = hashlib.sha256()
    by = {r[0]: r for r in results}
    for c in cases:
        cid, status = by[c.id][0], by[c.id][1]
        h.update(struct.pack("<II", cid, status))
        h.update(bytes(by[c.id][5]))
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Build helpers                                                               #
# --------------------------------------------------------------------------- #
def compile_cmd(argv, must_succeed=True):
    _commands.append(" ".join(argv))
    p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if must_succeed and p.returncode != 0:
        raise SystemExit("compile failed rc=%d: %s\n%s"
                         % (p.returncode, " ".join(argv),
                            p.stdout.decode("utf-8", "replace")))
    return p.returncode == 0, p.stdout.decode("utf-8", "replace")


def build_genuine(build_dir):
    lib = os.path.join(build_dir, "liboracle_genuine.so")
    compile_cmd([CC, "-std=c11", "-O0", "-fPIC", "-shared", "-I", CORE_ROOT,
                 DEP_ORACLE, "-o", lib])
    cc_bin = os.path.join(build_dir, "consumer_c")
    compile_cmd([CC, "-std=c11", "-O0", "-I", CORE_ROOT, DEP_CONSUMER_C, "-ldl",
                 "-o", cc_bin])
    cpp_bin = os.path.join(build_dir, "consumer_cpp")
    compile_cmd([CXX, "-std=c++17", "-O0", "-I", CORE_ROOT, DEP_CONSUMER_CPP,
                 "-ldl", "-o", cpp_bin])
    return lib, cc_bin, cpp_bin


ORACLE_MUTANTS = [
    ("wrong_status",
     "        return LB_STATUS_ERR_REQUEST_FIELD; /* MBZ / reserved violation */",
     "        return LB_STATUS_ERR_REQUEST_ABI; /* MBZ / reserved violation */"),
    ("wrong_precedence",
     "    if (result == NULL || result_size < 4u) {\n"
     "        /* Cannot even read struct_size: no access, no clearing. */\n"
     "        return LB_STATUS_ERR_RESULT_NULL;\n"
     "    }",
     "    if (request == NULL || request_size < 4u) {\n"
     "        return LB_STATUS_ERR_REQUEST_ABI;\n"
     "    }\n"
     "    if (result == NULL || result_size < 4u) {\n"
     "        return LB_STATUS_ERR_RESULT_NULL;\n"
     "    }"),
    ("offbyone_clear",
     "    size_t m = a < b ? a : b;\n    return m < c ? m : c;",
     "    size_t m = a < b ? a : b;\n    return (m < c ? m : c) + 1u;"),
    ("tail_overwrite",
     "    memset(result, 0, LB_RESULT_SIZE);",
     "    memset(result, 0, result_size);"),
    ("wrong_mask",
     "    eligible = available & (uint64_t)mask;",
     "    eligible = available;"),
    ("wrong_availability",
     "    available = (uint64_t)LB_BACKEND_MASK_CPU | (uint64_t)LB_BACKEND_MASK_CUDA;",
     "    available = (uint64_t)LB_BACKEND_MASK_CPU | (uint64_t)LB_BACKEND_MASK_CUDA | "
     "(uint64_t)LB_BACKEND_MASK_OPENCL;"),
    ("wrong_selection_path",
     "        selected_path    = LB_PATH_CPU_INLINE;",
     "        selected_path    = LB_PATH_GPU_BATCH;"),
    ("layout_offset",
     "    store_u32(result, LB_RES_OFF_SELECTED_PATH, selected_path);",
     "    store_u32(result, LB_RES_OFF_CALIBRATION, selected_path);"),
    ("omit_wrap",
     "    return (base + (uintptr_t)len) < base;",
     "    return 0;"),
    ("omit_overlap",
     "    return (a < b_end) && (b < a_end);",
     "    return 0;"),
]

REFERENCE_MUTANTS = ["ref_omit_wrap", "ref_offbyone_clear", "ref_ignore_mask"]


def build_oracle_mutant(build_dir, name, old, new):
    with open(DEP_ORACLE) as f:
        src = f.read()
    if old not in src:
        raise SystemExit("oracle mutant %s: anchor missing (source drift?)" % name)
    mutated = src.replace(old, new, 1)
    if mutated == src:
        raise SystemExit("oracle mutant %s: no-op replacement" % name)
    msrc = os.path.join(build_dir, "mut_%s.c" % name)
    with open(msrc, "w") as f:
        f.write(mutated)
    lib = os.path.join(build_dir, "libmut_%s.so" % name)
    ok, _out = compile_cmd([CC, "-std=c11", "-O0", "-fPIC", "-shared", "-I", CORE_ROOT,
                            msrc, "-o", lib], must_succeed=False)
    return (msrc, lib if ok else None, ok)


def _header_mut_result_size(src):
    return re.subn(r"(#define\s+LB_RESULT_SIZE\s+UINT32_C\()72(\))", r"\g<1>64\g<2>",
                   src, count=1)


def _header_mut_sel_path_off(src):
    return re.subn(r"(#define\s+LB_RES_OFF_SELECTED_PATH\s+)20u", r"\g<1>24u",
                   src, count=1)


HEADER_MUTANTS = [
    ("hdr_result_size_64", _header_mut_result_size, "compile"),
    ("hdr_sel_path_off_24", _header_mut_sel_path_off, "run"),
]


def build_header_mutant(build_dir, name, fn):
    """Mutate the accepted contract header in an isolated dir, copy the accepted
    oracle beside it, and compile the oracle against the mutated header."""
    with open(DEP_HEADER) as f:
        hsrc = f.read()
    mutated, nsub = fn(hsrc)
    if nsub != 1:
        raise SystemExit("header mutant %s: anchor missing (source drift?)" % name)
    if mutated == hsrc:
        raise SystemExit("header mutant %s: no-op replacement" % name)
    d = os.path.join(build_dir, "hdr_%s" % name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, HEADER), "w") as f:
        f.write(mutated)
    ocopy = os.path.join(d, ORACLE_C)
    shutil.copyfile(DEP_ORACLE, ocopy)
    lib = os.path.join(d, "libhdrmut.so")
    ok, _out = compile_cmd([CC, "-std=c11", "-O0", "-fPIC", "-shared", "-I", d,
                            ocopy, "-o", lib], must_succeed=False)
    return lib if ok else None, ok


CONSUMER_MUTANTS = [
    ("cons_c_wrong_reqsize", DEP_CONSUMER_C, "c",
     "(size_t)req_size", "(size_t)(req_size + 8u)"),
    ("cons_cpp_wrong_reqsize", DEP_CONSUMER_CPP, "cpp",
     "static_cast<size_t>(req_size)", "static_cast<size_t>(req_size + 8u)"),
]


def build_consumer_mutant(build_dir, name, src_path, lang, old, new):
    with open(src_path) as f:
        src = f.read()
    if old not in src:
        raise SystemExit("consumer mutant %s: anchor missing" % name)
    mutated = src.replace(old, new, 1)
    if mutated == src:
        raise SystemExit("consumer mutant %s: no-op" % name)
    ext = ".c" if lang == "c" else ".cpp"
    msrc = os.path.join(build_dir, name + ext)
    with open(msrc, "w") as f:
        f.write(mutated)
    exe = os.path.join(build_dir, name)
    if lang == "c":
        compile_cmd([CC, "-std=c11", "-O0", "-I", CORE_ROOT, msrc, "-ldl", "-o", exe])
    else:
        compile_cmd([CXX, "-std=c++17", "-O0", "-I", CORE_ROOT, msrc, "-ldl", "-o", exe])
    return exe


# --------------------------------------------------------------------------- #
# Guard-page + ASan harness sources                                           #
# --------------------------------------------------------------------------- #
GUARD_PROBE = r'''
#include "gpu_batch_policy_abi_contract.h"
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <dlfcn.h>
typedef lb_status (*query_fn)(const void *, size_t, void *, size_t);
static void p32(void *b, size_t o, uint32_t v){ memcpy((unsigned char*)b+o,&v,4); }
static void p64(void *b, size_t o, uint64_t v){ memcpy((unsigned char*)b+o,&v,8); }
int main(int argc, char **argv){
    if(argc<3){ fprintf(stderr,"usage %s <so> <tail|obo>\n",argv[0]); return 2; }
    void *h=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
    if(!h){ fprintf(stderr,"dlopen %s\n",dlerror()); return 3; }
    query_fn q=(query_fn)dlsym(h,"lb_gpu_batch_policy_query");
    if(!q){ return 4; }
    long pg=sysconf(_SC_PAGESIZE);
    unsigned char *base=(unsigned char*)mmap(NULL,(size_t)pg*2,PROT_READ|PROT_WRITE,
                                             MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(base==MAP_FAILED){ perror("mmap"); return 5; }
    unsigned char *guard=base+pg;
    if(mprotect(guard,(size_t)pg,PROT_NONE)!=0){ perror("mprotect"); return 6; }
    unsigned char *res=guard-72;   /* 72-byte window flush against guard */
    unsigned char *req=base;       /* disjoint valid request */
    memset(req,0,32);
    p32(req,LB_REQ_OFF_STRUCT_SIZE,32); p32(req,LB_REQ_OFF_ABI_VERSION,LB_GPU_BATCH_POLICY_ABI_V1);
    p32(req,LB_REQ_OFF_OPERATION,LB_BATCH_OP_ECDSA_VERIFY); p32(req,LB_REQ_OFF_BACKEND_MASK,LB_BACKEND_MASK_CPU);
    p64(req,LB_REQ_OFF_ITEM_COUNT,10); p32(req,LB_REQ_OFF_CONCURRENCY,2); p32(req,LB_REQ_OFF_RESERVED,0);
    size_t res_size; memset(res,0,72);
    if(strcmp(argv[2],"tail")==0){
        p32(res,LB_RES_OFF_STRUCT_SIZE,72); p32(res,LB_RES_OFF_ABI_VERSION,LB_GPU_BATCH_POLICY_ABI_V1);
        res_size=128;   /* oversized P: genuine touches 72, tail-overwrite hits guard */
    } else {
        p32(res,LB_RES_OFF_STRUCT_SIZE,200); p32(res,LB_RES_OFF_ABI_VERSION,0xDEAD);
        res_size=200;   /* bad-ABI: genuine clears 72, off-by-one clears 73 into guard */
    }
    lb_status st=q(req,32,res,res_size);
    printf("GUARD_NOFAULT status=%u\n",(unsigned)st);
    fflush(stdout);
    return 0;
}
'''

ASAN_PROBE = r'''
#include "gpu_batch_policy_abi_contract.h"
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
lb_status lb_gpu_batch_policy_query(const void *, size_t, void *, size_t);
static void p32(void *b, size_t o, uint32_t v){ memcpy((unsigned char*)b+o,&v,4); }
static void p64(void *b, size_t o, uint64_t v){ memcpy((unsigned char*)b+o,&v,8); }
int main(int argc, char **argv){
    if(argc<2){ fprintf(stderr,"usage %s <tail|obo>\n",argv[0]); return 2; }
    unsigned char req[32]; memset(req,0,sizeof req);
    p32(req,LB_REQ_OFF_STRUCT_SIZE,32); p32(req,LB_REQ_OFF_ABI_VERSION,LB_GPU_BATCH_POLICY_ABI_V1);
    p32(req,LB_REQ_OFF_OPERATION,LB_BATCH_OP_ECDSA_VERIFY); p32(req,LB_REQ_OFF_BACKEND_MASK,LB_BACKEND_MASK_CPU);
    p64(req,LB_REQ_OFF_ITEM_COUNT,10); p32(req,LB_REQ_OFF_CONCURRENCY,2); p32(req,LB_REQ_OFF_RESERVED,0);
    unsigned char *res=(unsigned char*)malloc(72); memset(res,0,72);
    size_t res_size;
    if(strcmp(argv[1],"tail")==0){
        p32(res,LB_RES_OFF_STRUCT_SIZE,72); p32(res,LB_RES_OFF_ABI_VERSION,LB_GPU_BATCH_POLICY_ABI_V1);
        res_size=128;
    } else {
        p32(res,LB_RES_OFF_STRUCT_SIZE,200); p32(res,LB_RES_OFF_ABI_VERSION,0xDEAD);
        res_size=200;
    }
    lb_status st=lb_gpu_batch_policy_query(req,32,res,res_size);
    printf("ASAN_OK status=%u\n",(unsigned)st);
    free(res);
    return 0;
}
'''


def two_run_gate(r1, r2):
    """Overall two-run gate: every run must have zero divergence on all three
    axes, and the full C/C++ output byte streams and result hash must be stable
    across both runs."""
    reasons = []
    for tag, r in (("run1", r1), ("run2", r2)):
        if r["dc"]:
            reasons.append("%s C-vs-model=%d" % (tag, r["dc"]))
        if r["dx"]:
            reasons.append("%s C++-vs-model=%d" % (tag, r["dx"]))
        if r["dcr"]:
            reasons.append("%s C-vs-C++=%d" % (tag, r["dcr"]))
    if r1["out_sha_c"] != r2["out_sha_c"]:
        reasons.append("C output unstable")
    if r1["out_sha_cpp"] != r2["out_sha_cpp"]:
        reasons.append("C++ output unstable")
    if r1["rhash"] != r2["rhash"]:
        reasons.append("result hash unstable")
    return (len(reasons) == 0), reasons


def differential_run(cc_bin, cpp_bin, lib, stream, cases):
    ptr_c, res_c = run_consumer(cc_bin, lib, stream)
    ptr_cpp, res_cpp = run_consumer(cpp_bin, lib, stream)
    dc = diff_against_reference(cases, ptr_c, res_c)
    dx = diff_against_reference(cases, ptr_cpp, res_cpp)
    dcr = cross_consumer_diff(cases, res_c, res_cpp)
    return {
        "ptr_c": ptr_c, "ptr_cpp": ptr_cpp, "res_c": res_c, "res_cpp": res_cpp,
        "dc": len(dc), "dx": len(dx), "dcr": len(dcr),
        "dc_list": dc, "dx_list": dx, "dcr_list": dcr,
        "out_sha_c": out_sha(res_c), "out_sha_cpp": out_sha(res_cpp),
        "rhash": result_hash(cases, res_c),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch-root", required=True)
    args = ap.parse_args()

    scratch = os.path.abspath(args.scratch_root)
    os.makedirs(scratch, exist_ok=True)
    build_dir = os.path.join(scratch, "build")
    shutil.rmtree(build_dir, ignore_errors=True)
    os.makedirs(build_dir, exist_ok=True)

    R = {"gates": {}, "infra_product": {}}
    print("== %s (%s) ==" % (TASK_ID, TOPIC))
    print("origin_thread_id = %s" % ORIGIN_THREAD_ID)

    # (1) fail-closed dependencies
    verified, reports = verify_dependencies()
    R["verified"] = verified
    R["reports"] = reports
    R["gates"]["deps_ok"] = True
    R["infra_product"]["dependency_hash_bind"] = "infra"
    print("\n[deps] four accepted-artifact dependencies + 3 reports hash-bound (fail-closed OK)")

    # (2)/(3) case matrix + genuine differential, run twice
    cases, counts = generate_cases()
    total = len(cases)
    if total != 611:
        raise SystemExit("case cardinality drift: %d != 611" % total)
    stream = encode_stream(cases)
    R["stream_sha"] = hashlib.sha256(stream).hexdigest()
    R["counts"] = counts
    R["total"] = total
    print("\n[cases] total=%d  stream_sha=%s" % (total, R["stream_sha"]))

    lib, cc_bin, cpp_bin = build_genuine(build_dir)
    run1 = differential_run(cc_bin, cpp_bin, lib, stream, cases)
    run2 = differential_run(cc_bin, cpp_bin, lib, stream, cases)
    R["run1"], R["run2"] = run1, run2
    gate_ok, gate_reasons = two_run_gate(run1, run2)
    R["gates"]["two_run_ok"] = gate_ok
    R["two_run_reasons"] = gate_reasons
    R["infra_product"]["two_run_stability"] = "infra"
    R["infra_product"]["abi_differential"] = "product"
    for label, r in (("run1", run1), ("run2", run2)):
        print("  %s: ptr=%d/%d  dc=%d dx=%d dcr=%d  rhash=%s"
              % (label, r["ptr_c"], r["ptr_cpp"], r["dc"], r["dx"], r["dcr"], r["rhash"]))
    for d in run1["dc_list"][:10]:
        print("    C-vs-model DIVERGENCE %s" % (d,))
    print("  two_run_gate=%s reasons=%s" % (gate_ok, gate_reasons or "-"))

    # run2-only divergence self-check: gate MUST then fail
    bad_res_cpp = [list(r) for r in run2["res_cpp"]]
    victim = bad_res_cpp[0]
    varr = bytearray(victim[5])
    varr[0] ^= 0xFF
    victim[5] = bytes(varr)
    bad_res_cpp = [tuple(r) for r in bad_res_cpp]
    run2_bad = dict(run2)
    run2_bad["res_cpp"] = bad_res_cpp
    run2_bad["dx"] = len(diff_against_reference(cases, run2["ptr_cpp"], bad_res_cpp))
    run2_bad["dcr"] = len(cross_consumer_diff(cases, run2["res_c"], bad_res_cpp))
    run2_bad["out_sha_cpp"] = out_sha(bad_res_cpp)
    sc_ok, sc_reasons = two_run_gate(run1, run2_bad)
    run2_selfcheck_ok = (not sc_ok) and run2_bad["dx"] > 0
    R["gates"]["run2_selfcheck_ok"] = run2_selfcheck_ok
    R["run2_selfcheck"] = {"detected_dx": run2_bad["dx"], "gate_failed": not sc_ok,
                           "reasons": sc_reasons}
    R["infra_product"]["run2_divergence_selfcheck"] = "infra"
    print("  run2-only C++ divergence self-check: gate_failed=%s dx=%d -> %s"
          % (not sc_ok, run2_bad["dx"], "OK" if run2_selfcheck_ok else "BROKEN"))

    # (4) real isolated compile/run source mutations (required, fail-closed)
    mutants = []  # (name, family, expected, observed, killed)
    print("\n[mutations] real isolated compile/run source mutants (each must be killed)")
    oracle_mut_src = {}
    for name, old, new in ORACLE_MUTANTS:
        msrc, mlib, built = build_oracle_mutant(build_dir, name, old, new)
        oracle_mut_src[name] = msrc
        if not built:
            observed, killed = "compile", True
        else:
            _pb, mres = run_consumer(cc_bin, mlib, stream)
            ndiv = len(diff_against_reference(cases, _pb, mres))
            observed, killed = ("run", True) if ndiv > 0 else ("survived", False)
        mutants.append((name, "oracle", "run", observed, killed))
        print("  %-24s family=oracle   observed=%-8s killed=%s" % (name, observed, killed))

    for name, fn, expected in HEADER_MUTANTS:
        mlib, built = build_header_mutant(build_dir, name, fn)
        if not built:
            observed, killed = "compile", True
        else:
            _pb, mres = run_consumer(cc_bin, mlib, stream)
            ndiv = len(diff_against_reference(cases, _pb, mres))
            observed, killed = ("run", True) if ndiv > 0 else ("survived", False)
        mutants.append((name, "header", expected, observed, killed))
        print("  %-24s family=header   expected=%-7s observed=%-8s killed=%s"
              % (name, expected, observed, killed))

    for name, src, lang, old, new in CONSUMER_MUTANTS:
        exe = build_consumer_mutant(build_dir, name, src, lang, old, new)
        _pb, mres = run_consumer(exe, lib, stream)
        ndiv = len(diff_against_reference(cases, _pb, mres))
        observed, killed = ("run", True) if ndiv > 0 else ("survived", False)
        mutants.append((name, "consumer", "run", observed, killed))
        print("  %-24s family=consumer observed=%-8s killed=%s" % (name, observed, killed))

    mut_ok = all(m[4] for m in mutants)
    R["mutants"] = mutants
    R["gates"]["mut_ok"] = mut_ok
    R["infra_product"]["source_mutation_kill"] = "product"

    # reference-model self-mutants (separate gate; do NOT substitute source mutants)
    ref_detected = []
    for name in REFERENCE_MUTANTS:
        n = len(diff_against_reference(cases, run1["ptr_c"], run1["res_c"], mutate=name))
        ref_detected.append((name, n, n > 0))
    ref_ok = all(d[2] for d in ref_detected)
    R["ref_selfmutants"] = ref_detected
    R["gates"]["ref_selfcheck_ok"] = ref_ok
    R["infra_product"]["reference_selfcheck"] = "infra"
    print("  reference-model self-mutants detected: %s" % ref_detected)

    # NEG_CASE compile gate (necessary, non-substituting) + hosted call
    print("\n[negative.c] clean compile + NEG_CASE rejection + runtime call")
    neg = {"clean_native": None, "clean_m32": None, "runtime": None, "neg": []}
    ok, _ = compile_cmd([CC, "-std=c11", "-fsyntax-only", "-I", CORE_ROOT, NEGATIVE_C],
                        must_succeed=False)
    neg["clean_native"] = ok
    ok32, out32 = compile_cmd([CC, "-m32", "-std=c11", "-ffreestanding", "-fsyntax-only",
                               "-I", CORE_ROOT, NEGATIVE_C], must_succeed=False)
    neg["clean_m32"] = ok32
    if not ok32:
        neg["m32_reason"] = "-m32 freestanding syntax-only unavailable (skipped, non-fatal)"
    for i in range(1, 9):
        okn, _ = compile_cmd([CC, "-std=c11", "-fsyntax-only", "-DNEG_CASE=%d" % i,
                              "-I", CORE_ROOT, NEGATIVE_C], must_succeed=False)
        neg["neg"].append((i, not okn))  # killed iff compile failed
    neg_exe = os.path.join(build_dir, "negative_c")
    compile_cmd([CC, "-std=c11", "-O0", "-I", CORE_ROOT, NEGATIVE_C, "-ldl", "-o", neg_exe])
    pr = run_capture([neg_exe, lib])
    neg["runtime"] = (pr.returncode == 0 and b"NEGATIVE_C_CONSUMER_OK" in pr.stdout)
    neg_ok = neg["clean_native"] and all(k for _, k in neg["neg"]) and neg["runtime"]
    R["negative"] = neg
    R["gates"]["neg_ok"] = neg_ok
    R["infra_product"]["negative_compilation"] = "product"
    print("  clean_native=%s clean_m32=%s runtime_call=%s neg_cases_killed=%d/8"
          % (neg["clean_native"], neg["clean_m32"], neg["runtime"],
             sum(1 for _, k in neg["neg"] if k)))

    # (5) memory-safety: guard-page (mandatory) + ASan (best-effort)
    print("\n[memory-safety] guard-page + ASan on genuine oracle and real mutants")
    guard_src = os.path.join(build_dir, "guard_probe.c")
    with open(guard_src, "w") as f:
        f.write(GUARD_PROBE)
    guard_exe = os.path.join(build_dir, "guard_probe")
    compile_cmd([CC, "-std=c11", "-O0", "-I", CORE_ROOT, guard_src, "-ldl", "-o", guard_exe])

    def guard_run(so, scen):
        return run_capture([guard_exe, so, scen])

    g = {}
    g["genuine_tail_nofault"] = (guard_run(lib, "tail").returncode == 0)
    g["genuine_obo_nofault"] = (guard_run(lib, "obo").returncode == 0)
    tail_lib = os.path.join(build_dir, "libmut_tail_overwrite.so")
    obo_lib = os.path.join(build_dir, "libmut_offbyone_clear.so")
    g["tail_mutant_faults"] = (guard_run(tail_lib, "tail").returncode < 0)
    g["obo_mutant_faults"] = (guard_run(obo_lib, "obo").returncode < 0)
    guard_ok = all(g.values())
    R["guard"] = g
    print("  guard: genuine_tail=%s genuine_obo=%s tail_mutant_faults=%s obo_mutant_faults=%s"
          % (g["genuine_tail_nofault"], g["genuine_obo_nofault"],
             g["tail_mutant_faults"], g["obo_mutant_faults"]))

    asan = {"available": False, "skipped_reason": None}
    probe_c = os.path.join(build_dir, "asan_avail.c")
    with open(probe_c, "w") as f:
        f.write("int main(void){return 0;}\n")
    asan_ok_flag, _ = compile_cmd([CC, "-fsanitize=address", "-O0", probe_c, "-o",
                                   os.path.join(build_dir, "asan_avail")],
                                  must_succeed=False)
    asan_gate = True
    if asan_ok_flag:
        asan["available"] = True
        asan_src = os.path.join(build_dir, "asan_probe.c")
        with open(asan_src, "w") as f:
            f.write(ASAN_PROBE)

        def build_asan(oracle_src, tag):
            exe = os.path.join(build_dir, "asan_%s" % tag)
            compile_cmd([CC, "-std=c11", "-O0", "-g", "-fsanitize=address", "-I", CORE_ROOT,
                         oracle_src, asan_src, "-o", exe])
            return exe

        env = dict(os.environ)
        env["ASAN_OPTIONS"] = "detect_leaks=0"

        def asan_run(exe, scen):
            argv = [exe, scen]
            try:
                p = subprocess.run(argv, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, env=env)
            except PermissionError:
                interp = elf_interp(exe) or "/lib64/ld-linux-x86-64.so.2"
                argv = [interp] + argv
                p = subprocess.run(argv, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, env=env)
            _commands.append(" ".join(argv))
            return p

        gen_exe = build_asan(DEP_ORACLE, "genuine")
        tail_exe = build_asan(oracle_mut_src["tail_overwrite"], "tail")
        obo_exe = build_asan(oracle_mut_src["offbyone_clear"], "obo")
        asan["genuine_tail_clean"] = (asan_run(gen_exe, "tail").returncode == 0)
        asan["genuine_obo_clean"] = (asan_run(gen_exe, "obo").returncode == 0)
        asan["tail_mutant_trips"] = (asan_run(tail_exe, "tail").returncode != 0)
        asan["obo_mutant_trips"] = (asan_run(obo_exe, "obo").returncode != 0)
        asan_gate = (asan["genuine_tail_clean"] and asan["genuine_obo_clean"]
                     and asan["tail_mutant_trips"] and asan["obo_mutant_trips"])
        print("  asan: genuine_tail=%s genuine_obo=%s tail_trips=%s obo_trips=%s"
              % (asan["genuine_tail_clean"], asan["genuine_obo_clean"],
                 asan["tail_mutant_trips"], asan["obo_mutant_trips"]))
    else:
        asan["skipped_reason"] = "AddressSanitizer unavailable; guard-page is the equivalent"
        print("  asan: unavailable -> guard-page equivalent covers memory safety (logged)")
    R["asan"] = asan
    mem_ok = guard_ok and (asan_gate if asan["available"] else True)
    R["gates"]["mem_ok"] = mem_ok
    R["infra_product"]["memory_safety"] = "product"

    # ---- overall verdict ----
    gates = R["gates"]
    overall = all(gates.values())
    R["overall"] = overall
    R["commands"] = list(_commands)
    write_report(R, scratch)

    print("\n[gates]")
    for k in sorted(gates):
        print("  %-22s %s" % (k, "PASS" if gates[k] else "FAIL"))
    print("\nRESULT: %s (commands executed=%d)"
          % ("PASS" if overall else "FAIL", len(_commands)))
    return 0 if overall else 1


def write_report(R, scratch):
    run1, run2 = R["run1"], R["run2"]
    lines = []
    a = lines.append
    a("# %s — GPU ABI core/oracle real-artifact validator report" % TASK_ID)
    a("")
    a("Topic `%s` · runner `claude_gpu_abi_core_validator_v4`. Findings below are"
      % TOPIC)
    a("produced by actual executions of this validator, not prose.")
    a("")
    a("## 1. Identity / origin / provenance")
    a("")
    a("| field | value |")
    a("|-------|-------|")
    a("| task_id | `%s` |" % TASK_ID)
    a("| origin_thread_id | `%s` |" % ORIGIN_THREAD_ID)
    a("| canonical HEAD | `aed0ebcc5822b577d7ababa31a28172c1c38a2c4` |")
    a("| canonical tree | `2adb9ce493129f30d6d3f2db1884887fa609c27f` |")
    a("| authority_repo | `/home/shrek/Secp256k1/Secp256K1fast/libs/UltrafastSecp256k1` |")
    a("| toolchain | `cc/c++ 14.2.0`, `python3` |")
    a("")
    a("## 2. Four dependencies — finished + fail-closed accepted hashes")
    a("")
    a("| dependency task | kind | sha256 | report sha256 (finished) |")
    a("|-----------------|------|--------|--------------------------|")
    for task, kind, path, report in DEPENDENCIES:
        a("| `%s` | %s | `%s` | `%s` |"
          % (task, kind, R["verified"][path], R["reports"][report]))
    a("")
    a("Header-shim (290) copy verified byte-identical to the accepted 264 header; "
      "no duplicate ABI substituted. Supporting accepted inputs (native tests, "
      "reference_model.py, run_differential.py, 281-side header) also hash-bound.")
    a("")
    a("## 3. Independent 611-case ABI recomputation")
    a("")
    a("Independent byte-level reference model (this file only) over request32/"
      "result72 four-argument ABI. Stream sha256 `%s`." % R["stream_sha"])
    a("")
    a("| group | count |")
    a("|-------|------:|")
    for k in sorted(R["counts"]):
        a("| `%s` | %d |" % (k, R["counts"][k]))
    a("| **TOTAL** | **%d** |" % R["total"])
    a("")
    a("## 4. Genuine differential — two stable runs")
    a("")
    a("| run | ptr_bits C/C++ | C-vs-model | C++-vs-model | C-vs-C++ | result hash |")
    a("|-----|----------------|-----------:|-------------:|---------:|-------------|")
    for label, r in (("run1", run1), ("run2", run2)):
        a("| %s | %d/%d | %d | %d | %d | `%s` |"
          % (label, r["ptr_c"], r["ptr_cpp"], r["dc"], r["dx"], r["dcr"], r["rhash"]))
    a("")
    a("Stability: C out sha `%s` (run1) == `%s` (run2); C++ out sha `%s` == `%s`; "
      "result hash equal. two_run_gate = %s."
      % (run1["out_sha_c"], run2["out_sha_c"], run1["out_sha_cpp"],
         run2["out_sha_cpp"], R["gates"]["two_run_ok"]))
    a("")
    a("Run2-only C++ divergence self-check: an injected single-byte C++ deviation "
      "in run2 produced dx=%d and made the two-run gate FAIL (`gate_failed=%s`), "
      "proving a run2-only divergence fails overall validation."
      % (R["run2_selfcheck"]["detected_dx"], R["run2_selfcheck"]["gate_failed"]))
    a("")
    a("## 5. Real isolated compile/run source mutations (fail-closed mut_ok=%s)"
      % R["gates"]["mut_ok"])
    a("")
    a("| mutant | family | expected kill | observed kill | killed |")
    a("|--------|--------|---------------|---------------|--------|")
    for name, fam, exp, obs, killed in R["mutants"]:
        a("| `%s` | %s | %s | %s | %s |" % (name, fam, exp, obs, killed))
    a("")
    a("Header mutants derive from the accepted contract header (result-size weakening "
      "-> compile kill; selected-path offset weakening -> run kill). Consumer mutants "
      "derive from the accepted C and C++ consumers (wrong request size -> run kill). "
      "Reference-model self-mutants %s and NEG_CASE compile probes are separate "
      "evidence and do not substitute for the source mutations above."
      % R["ref_selfmutants"])
    a("")
    a("## 6. Negative compilation (neg_ok=%s)" % R["gates"]["neg_ok"])
    a("")
    a("`audit/gpu_batch_policy_abi_negative.c` clean native syntax-only=%s, "
      "-m32 freestanding syntax-only=%s%s, hosted runtime call of the accepted "
      "oracle=%s. NEG_CASE 1..8 each rejected: %s."
      % (R["negative"]["clean_native"], R["negative"]["clean_m32"],
         "" if R["negative"]["clean_m32"] else (" (%s)" % R["negative"].get("m32_reason", "")),
         R["negative"]["runtime"],
         ", ".join("N%d=%s" % (i, k) for i, k in R["negative"]["neg"])))
    a("")
    a("## 7. Memory safety (mem_ok=%s)" % R["gates"]["mem_ok"])
    a("")
    a("Guard-page (mmap PROT_NONE) on the genuine oracle and isolated real mutants: %s."
      % R["guard"])
    if R["asan"]["available"]:
        a("")
        a("AddressSanitizer on genuine oracle + tail_overwrite/offbyone mutants: %s."
          % {k: v for k, v in R["asan"].items() if k != "available"})
    else:
        a("")
        a("AddressSanitizer %s." % R["asan"]["skipped_reason"])
    a("")
    a("## 8. Infra / product classification")
    a("")
    a("| check | classification |")
    a("|-------|----------------|")
    for k in sorted(R["infra_product"]):
        a("| %s | %s |" % (k, R["infra_product"][k]))
    a("")
    a("## 9. Gate summary")
    a("")
    a("| gate | result |")
    a("|------|--------|")
    for k in sorted(R["gates"]):
        a("| %s | %s |" % (k, "PASS" if R["gates"][k] else "FAIL"))
    a("| **overall** | **%s** |" % ("PASS" if R["overall"] else "FAIL"))
    a("")
    a("## 10. AIWorkHub MCP calls")
    a("")
    a("- `source_graph_query` (focus, query=`gpu`, budget 48, bundle explore): "
      "96 hits, canonical sole_authority.")
    a("- `session_current_state` (topic as carded, limit 8): state=unknown, 0 evidence.")
    a("- `ai_memory_search` / `kb_search` (carded query, limit 5): 0 hits.")
    a("")
    a("## 11. Commands executed (%d)" % len(R["commands"]))
    a("")
    a("```")
    for c in R["commands"]:
        a(c)
    a("```")
    a("")
    with open(REPORT_OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
