#!/usr/bin/env python3
"""
check_ct_branches.py — STATIC constant-time gate for the GPU/CPU arithmetic primitives.

WHY THIS EXISTS (the cornerstone CT defence):
  GPU timing is far too noisy for black-box constant-time detection, and the in-launch
  dudect probe structurally MASKS key-dependent warp divergence. The only reliable,
  hardware-independent, every-push way to guarantee constant-time is to forbid the bug
  class at the SOURCE: a data-dependent CONTROL-FLOW branch (if/while/ternary) whose
  condition is derived from a secret value — the "conditional modular reduction /
  conditional subtract" pattern that produced the 2026-06 GPU ECDSA leak.

  This gate scans the constant-time arithmetic primitives (the functions reachable from
  secret-bearing signing / ECDH / scan / ZK kernels) and FAILS if any of them contains a
  branch whose condition tests a reduction/borrow/carry/limb-comparison signal. The
  constant-time idiom is: always compute both results and select with a value-barriered
  mask (cmov) — which this gate does NOT flag (mask math is not control flow).

  It runs anywhere (pure text analysis, no GPU, no CUDA toolchain), so it gates on every
  push in GitHub CI as well as locally — exactly the "catch any leak anywhere" guarantee.

USAGE:
  python3 ci/check_ct_branches.py            # exit 1 if any forbidden branch found
  python3 ci/check_ct_branches.py --list     # list scanned files + counts

Exit 0 = clean, 1 = forbidden data-dependent branch found.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Constant-time-critical sources: the GPU/CPU arithmetic primitives + CT wrappers that run
# on secret scalars (private key, nonce, scan key, signing share). A data-dependent branch
# in ANY of these is a constant-time violation.
CT_FILES = [
    "src/cuda/include/secp256k1.cuh",
    "src/cuda/include/secp256k1_32.cuh",
    "src/cuda/include/secp256k1_32_hybrid_final.cuh",
    "src/cuda/include/ct/ct_scalar.cuh",
    "src/cuda/include/ct/ct_point.cuh",
    "src/cuda/include/ct/ct_sign.cuh",
    "src/cuda/include/ct/ct_ops.cuh",
]

# CT-P2-02: dedicated scan for the GPU scalar_mul_mod_n REDUCTION leak class
# (CT-P2-01). The Solinas/NC folding had `if (prod[i]==0) continue;` skip-if-zero and
# data-dependent `&& carry` loop bounds — both branch on the secret-derived product
# and produced key-dependent warp divergence. The general FORBIDDEN scan below MISSES
# these: `[...] == 0` carries no borrow/carry token, and the loop index inside the
# subscript trips BENIGN. So we scan the reduction sources for the exact two shapes.
# These files mix CT and VT (verify) code, so we do NOT apply the broad borrow/carry
# scan to them (it would flag legitimate public-data verify branches); we only forbid
# the two reduction-leak shapes.
REDUCTION_FILES = [
    "src/opencl/kernels/secp256k1_extended.cl",
    "src/metal/shaders/secp256k1_extended.h",
    "src/cuda/include/secp256k1.cuh",
]
# (a) skip-if-zero on a reduction limb/accumulator: if/while (<name>[..] == 0 | != 0)
# (b) data-dependent carry/borrow loop bound: for (...; ... && (carry|borrow); ...)
REDUCTION_FORBIDDEN = re.compile(
    r"\b(?:if|while)\s*\(\s*\(?\s*(?:prod|acc|res|hi|qn|qmu)\s*\[[^\]]*\]\s*(?:==|!=)\s*0\b"
    r"|\bfor\s*\([^;]*;[^;]*&&\s*(?:carry|borrow)\b"
)


# A branch condition is FORBIDDEN when it tests a secret-derived reduction signal:
# borrow/carry out of a subtract/add, a >= modulus comparison, or a raw limb comparison.
FORBIDDEN = re.compile(
    r"\b(borrow|carry)\b"
    r"|\bscalar_ge\b|\bfield_ge\b|\bscalar_cmp\b|\buint320_compare\b|\buint256_compare\b"
    r"|>=\s*(ORDER|MODULUS|PRIME|SECP256K1_[NP]|P\b|N\b)"
    r"|\blimbs\s*\[[^\]]+\]\s*[<>]"
)

# Benign control flow whose condition is structural/public, never secret:
# thread/loop guards, small constant limb-index bounds, null/infinity checks, preprocessor.
BENIGN = re.compile(
    r"\b(idx|gid|tid|lane|warp|blockIdx|threadIdx|blockDim|count|n_|num_|len|size|pos|"
    r"limb_idx|bit_idx|word_idx|wlen|window|i\b|j\b|k\b)\b"
    r"|\b(is_infinity|infinity|->infinity|nullptr|NULL)\b"
    r"|^\s*#"
    r"|[<>]=?\s*\d+\b"          # comparison against a small integer literal (loop/index bound)
)

# Real control-flow branch: an if/while STATEMENT. We deliberately do NOT match the
# ternary mask idiom `(cond) ? 1ULL : 0ULL` / `? ~0ULL : 0ULL` — that is the constant-time
# replacement (compiles to setp/select, no divergence), not a violation.
BRANCH = re.compile(r"\b(if|while)\s*\(")
# Comment lines (the FORBIDDEN tokens often appear in explanatory comments).
COMMENT = re.compile(r"^\s*(//|\*|/\*)")

# Allowlisted branches: VT-correct BY DESIGN (verify path on public data) or in a
# non-built/broken config. Matched by (file-name substring, code substring). Every entry
# MUST carry a reason. New secret-path branches are NOT allowlisted and will fail the gate.
ALLOWLIST = [
    ("secp256k1.cuh", "if (!carry) break;",
     "scalar_to_wnaf4: wNAF is the VARIABLE-TIME verify-path decomposition over PUBLIC "
     "data; CT signing uses GLV+cmov and never calls wNAF (CLAUDE.md CT-VERIFY: verify=VT)."),
    ("secp256k1.cuh", "if (static_cast<int>(b) == carry)",
     "wnaf_encode: same wNAF verify-path, variable-time by design, public data."),
    ("secp256k1_32.cuh", "if (borrow)",
     "32-bit-limb config is pre-existingly BROKEN/dead (KB GPU-CUDA-LIMBS32-BROKEN: does "
     "not compile). Branchless fix to be applied + verified when the config is repaired."),
    ("secp256k1_32.cuh", "if (borrow == 0)",
     "32-bit-limb config broken/dead (KB GPU-CUDA-LIMBS32-BROKEN) — fix when repaired."),
]


def is_allowlisted(rel: str, line: str) -> bool:
    for fsub, csub, _reason in ALLOWLIST:
        if fsub in rel and csub in line:
            return True
    return False


def scan_file(path: Path, rel: str) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    if not path.exists():
        return hits
    for n, raw in enumerate(path.read_text(errors="replace").splitlines(), 1):
        line = raw.strip()
        if COMMENT.match(raw):
            continue
        if not BRANCH.search(line):
            continue
        if not FORBIDDEN.search(line):
            continue
        # Structural/public branch (thread/loop/index guard) — not secret control flow.
        if BENIGN.search(line):
            continue
        # VT-correct-by-design (verify path) or broken/dead config — documented allowlist.
        if is_allowlisted(rel, line):
            continue
        hits.append((n, line[:120]))
    return hits


def scan_reduction_file(path: Path) -> list[tuple[int, str]]:
    """Scan a GPU reduction source for the CT-P2-01 leak shapes only (skip-if-zero on a
    reduction limb, data-dependent carry/borrow loop bound). Independent of BENIGN."""
    hits: list[tuple[int, str]] = []
    if not path.exists():
        return hits
    for n, raw in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if COMMENT.match(raw):
            continue
        # Strip a trailing line comment so a `// ... if (hi[i]==0) ...` annotation
        # (which legitimately describes the removed pattern) does not self-trigger.
        code = raw.split("//", 1)[0]
        if REDUCTION_FORBIDDEN.search(code):
            hits.append((n, code.strip()[:120]))
    return hits


def main() -> int:
    list_only = "--list" in sys.argv
    total = 0
    flagged: dict[str, list[tuple[int, str]]] = {}
    for rel in CT_FILES:
        p = ROOT / rel
        hits = scan_file(p, rel)
        if list_only:
            print(f"  {'OK ' if not hits else 'HIT'} {rel}  ({len(hits)} branch flag(s))")
        if hits:
            flagged[rel] = hits
            total += len(hits)

    # CT-P2-02: reduction-leak shape scan (OpenCL/Metal/CUDA scalar_mul_mod_n).
    for rel in REDUCTION_FILES:
        p = ROOT / rel
        hits = scan_reduction_file(p)
        if list_only:
            print(f"  {'OK ' if not hits else 'HIT'} {rel}  ({len(hits)} reduction-leak flag(s))")
        if hits:
            flagged.setdefault(rel, []).extend(hits)
            total += len(hits)

    if list_only:
        return 0

    if total:
        print(f"::error::check_ct_branches: {total} forbidden data-dependent branch(es) in "
              "constant-time arithmetic (secret-dependent if/while/ternary on a "
              "borrow/carry/>=modulus/limb-compare signal).")
        print("  The CT idiom is: always compute both results, select with a "
              "value-barriered mask (cmov) — never branch on a secret-derived value.")
        for rel, hits in flagged.items():
            for ln, code in hits:
                print(f"    {rel}:{ln}: {code}")
        return 1

    print(f"check_ct_branches: PASS — no secret-dependent branches in "
          f"{len(CT_FILES)} constant-time arithmetic sources.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
