#!/usr/bin/env python3
"""
dev_bug_scanner.py  —  Classic development-bug detector for UltrafastSecp256k1 C++ source.

Detects the kind of bugs a human reviewer catches during development:
  SPTR    sizeof applied to a pointer (wrong memset/memcpy size)
  OB1     off-by-one: loop uses <= where < is almost certainly intended
  SIG     signed/unsigned comparison mismatch
  TRUNC   silent integer truncation (u64/i64 → u32/i32 narrowing)
  OVER    integer overflow in 32-bit multiply before widening cast
  MSET    memset/memcpy with hard-coded literal size that looks wrong
  NULL    null-check AFTER dereference (use-before-check pattern)
  CPASTE  copy-paste: same lvalue assigned twice without use between them
  SEMI    empty loop or if body (dangling semicolon)
  MBREAK  switch case with no break and no [[fallthrough]] / FALLTHROUGH comment
  RETVAL  return value of ufsecp_* or key-generation function silently discarded
  LOGIC   security condition using wrong logical operator (&&/|| confusion)
  ZEROIZE memset clears fewer bytes than the declared buffer size
  DBLINIT variable assigned twice consecutively with no intervening read
  UNREACH statement after unconditional return/throw/continue/break
  SECRET_UNERASED  Scalar in signing/key path without secure_erase on exit
  CT_VIOLATION     fast:: namespace call in CT-required secret path
  TAGGED_HASH_BYPASS  plain sha256() where BIP-340 tagged_hash is required
  RANDOM_IN_SIGNING   non-deterministic random in RFC 6979 signing path
  BINDING_NO_VALIDATION  public ufsecp_ function missing NULL-check on args
  DANGLING_ELSE  braceless else with multiple statements (goto-fail pattern)
  ASSERT_SIDE    assert() with side-effectful code (stripped in Release builds)
  SHIFT_UB       shift by >= type width (undefined behavior)
  DOUBLE_LOCK    mutex locked twice without unlock (deadlock risk)
  UNSAFE_FMT     sprintf/strcpy/gets without bounds check (buffer overflow)
  HARDCODED_SECRET  hardcoded private key or seed in non-test code
  CATCH_EMPTY    empty catch block silently swallows errors
  SIZEOF_MISMATCH  memcpy/memset sizeof refers to wrong variable

  Crypto bug-pattern checkers (CVE-grounded, added 2026-04-21):
  NONCE_REUSE_VAR        same nonce variable used in 2+ sign() calls (Sony PS3, Bitcoin Android 2013)
  MEMCMP_SECRET          memcmp/strcmp on auth tag/MAC/signature (timing leak — Xbox 360, Java JCE)
  MISSING_LOW_S_CHECK    ECDSA verify path missing low-S guard (BIP-62 malleability)
  SCALAR_FROM_RAND       signing nonce derived from rand()/time() (CVE-2008-0166 Debian OpenSSL)
  GOTO_FAIL_DUPLICATE    duplicate `goto label;` skipping a check (Apple CVE-2014-1266)
  POINT_NO_VALIDATION    pubkey parser without on-curve / subgroup check (invalid-curve attack)
  ECDH_OUTPUT_NOT_CHECKED  ECDH output used without zero/identity check (small-subgroup)
  HASH_NO_DOMAIN_SEP     plain SHA-256 in BIP-340/Schnorr context without tagged_hash
  DER_LAX_PARSE          DER length read without canonicalisation (CVE-2014-8275)
  TIMING_BRANCH_ON_KEY   if (key[i] ...) direct branching on secret byte
  MAC_TRUNCATION         memcmp on MAC/tag with truncated length (forgery class)
  SCALAR_NOT_REDUCED     scalar_inverse without adjacent reduce/zero check
  PRINTF_SECRET          printf/log statement of secret-bearing identifier
  JNI_GETBYTES_NULL      JNI GetByteArrayElements return value used without null-check

  Language-binding FFI checkers (added 2026-04):
  FFI_RETVAL_IGNORED     secp256k1_*/ufsecp_* call in language binding with silently
                         discarded return value (Python/C#/Swift/Go/Rust/Ruby/PHP/Dart)

  Crypto parse-return checker (added 2026-04-26, grounded in SonarCloud C Reliability):
  PARSE_RETVAL_IGNORED   Scalar::parse_bytes_strict_nonzero or parse_bytes_strict return
                         value silently discarded; output Scalar indeterminate on failure

Usage examples
--------------
  # Scan everything under the default paths, human-readable output:
  python3 scripts/dev_bug_scanner.py

  # Scan just cpu/src, emit JSON, write to file:
  python3 scripts/dev_bug_scanner.py --src-dir cpu/src --json -o findings.json

  # Only show HIGH/MEDIUM severity:
  python3 scripts/dev_bug_scanner.py --min-severity MEDIUM

  # Non-zero exit code when findings exist (useful for CI):
  python3 scripts/dev_bug_scanner.py --fail-on-findings
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────

SEVERITY_RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}


@dataclass
class Finding:
    file: str
    line: int
    severity: str          # HIGH | MEDIUM | LOW
    category: str          # SPTR | OB1 | SIG | …
    message: str
    snippet: str           # raw source line (stripped)
    fix_hint: str


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _strip_comments(line: str) -> str:
    """Remove // line comments (simple heuristic, good enough for pattern matching)."""
    in_str = False
    i = 0
    while i < len(line) - 1:
        c = line[i]
        if c == '"' and (i == 0 or line[i - 1] != '\\'):
            in_str = not in_str
        if not in_str and line[i] == '/' and line[i + 1] == '/':
            return line[:i]
        i += 1
    return line


def _context_window(lines: List[str], idx: int, radius: int = 5) -> List[Tuple[int, str]]:
    """Returns [(lineno, text), …] for a window around idx."""
    start = max(0, idx - radius)
    end = min(len(lines), idx + radius + 1)
    return [(start + i + 1, lines[start + i]) for i in range(end - start)]


# ──────────────────────────────────────────────────────────────────────────────
# Individual checkers
# ──────────────────────────────────────────────────────────────────────────────

# ── SPTR: sizeof(pointer_var) inside mem* calls ───────────────────────────────
_MEMFN_SIZEOF = re.compile(
    r'\b(mem(?:set|cpy|move|cmp|chr))\s*\([^;]+\bsizeof\s*\(\s*([*a-zA-Z_][a-zA-Z0-9_]*)\s*\)',
    re.DOTALL,
)
# pointer parameters / variables: if the identifier is a pointer param we flag it
_PTR_DECL = re.compile(r'\b(?:uint\w+_t|int\w+_t|char|void|unsigned|const)\s*\*+\s*(\w+)')

def check_sptr(path: str, lines: List[str]) -> List[Finding]:
    """sizeof applied to a pointer-typed variable inside mem* — gives 8 bytes instead of buffer size."""
    findings: List[Finding] = []
    # Collect pointer-declared names in this file
    ptr_names: set[str] = set()
    for ln in lines:
        for m in _PTR_DECL.finditer(ln):
            ptr_names.add(m.group(1))

    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        for m in _MEMFN_SIZEOF.finditer(line):
            fn_name = m.group(1)
            sz_ident = m.group(2)
            if sz_ident in ptr_names:
                findings.append(Finding(
                    file=path, line=i + 1, severity="HIGH",
                    category="SPTR",
                    message=f"`sizeof({sz_ident})` passed to `{fn_name}` but `{sz_ident}` is a pointer — sizeof gives ptr width (8), not buffer size",
                    snippet=raw.strip(),
                    fix_hint=f"Replace sizeof({sz_ident}) with the actual byte count or use sizeof(*{sz_ident}) * count",
                ))
    return findings


# ── OB1: off-by-one loop bound (i <= N where N matches a size/length name) ───
_OB1_LOOP = re.compile(
    r'\bfor\s*\([^;]*;\s*(\w+)\s*<=\s*([A-Za-z_]\w*|0x[0-9a-fA-F]+|\d+)\s*;',
)
_SIZE_SUFFIX = re.compile(r'(?:size|len|count|num|n|N|max|MAX|limit|cap|total)$', re.IGNORECASE)

def check_ob1(path: str, lines: List[str]) -> List[Finding]:
    """Loop bound `i <= N` where N is a size/length constant — usually should be `i < N`."""
    findings: List[Finding] = []
    # Skip selftest / KAT / fuzz files where 1-based inclusive loops are common
    # for iteration counts (test vector index, retry attempts, etc.).
    if re.search(r'(selftest|kat|test_|/test|/tests|fuzz|bench)', path, re.IGNORECASE):
        return findings
    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        for m in _OB1_LOOP.finditer(line):
            idx_var = m.group(1)
            bound   = m.group(2)
            # AES/cipher round loops: `round <= 14` or similar — intentional (rounds are 1-based, inclusive)
            # These are numeric bounds 10-20 where the loop var is named round/Round/nr/Nr
            if re.search(r'\b(round|Round|nr|Nr|nrounds)\b', idx_var):
                continue
            # Skip 1-based inclusive count loops: `for (... = 1; i <= N; ...)` — common for test vectors.
            if re.search(rf'\b{re.escape(idx_var)}\s*=\s*1\s*;', line):
                continue
            # Skip explicit non-zero start indices: `for (... = K; i <= N; ...)`
            # where K > 1 — typically a deliberate sliding-window or sub-range loop.
            start_m = re.search(rf'\b{re.escape(idx_var)}\s*=\s*(\d+)\s*;', line)
            if start_m and int(start_m.group(1)) >= 2:
                continue
            # Only flag when bound looks like a size/count or is at least 4 as a literal
            if _SIZE_SUFFIX.search(bound) or (bound.isdigit() and int(bound) >= 4):
                findings.append(Finding(
                    file=path, line=i + 1, severity="MEDIUM",
                    category="OB1",
                    message=f"Loop condition `{idx_var} <= {bound}` may be an off-by-one error; "
                            f"consider `{idx_var} < {bound}` unless the array is sized {bound}+1",
                    snippet=raw.strip(),
                    fix_hint=f"Change `{idx_var} <= {bound}` to `{idx_var} < {bound}` unless the array is 1-based or has {bound}+1 elements",
                ))
    return findings


# ── SIG: signed/unsigned comparison ──────────────────────────────────────────
# int/signed variable on one side, size_t / unsigned / uint on the other
_SIG_DECL = re.compile(r'\bint\s+(\w+)\b')
_SIG_CMP  = re.compile(r'\b(\w+)\s*[<>]=?\s*(size_t|uint\d+_t|unsigned)\b')
_SIG_CMP2 = re.compile(r'\b(size_t|uint\d+_t|unsigned)\s+(\w+)\s*[><=]=?[^=>]')

_UNSIGNED_DECL_RE = re.compile(
    r'\b(?:size_t|std::size_t|unsigned|uint\d+_t|auto)\s+(\w+)\b'
)

def check_sig(path: str, lines: List[str]) -> List[Finding]:
    """Signed int compared with size_t / uint* — may fail when value > INT_MAX."""
    findings: List[Finding] = []
    int_vars: set[str] = set()
    unsigned_vars: set[str] = set()
    for ln in lines:
        s = _strip_comments(ln)
        for m in _SIG_DECL.finditer(s):
            int_vars.add(m.group(1))
        for m in _UNSIGNED_DECL_RE.finditer(s):
            unsigned_vars.add(m.group(1))
    # If a name is declared as both, prefer the unsigned (later) declaration —
    # the loop variable in `for (size_t i = ...)` shadows any outer `int i`.
    int_vars -= unsigned_vars

    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        # If the same line declares the variable as size_t/unsigned, skip.
        if _UNSIGNED_DECL_RE.search(line):
            continue
        # Pattern: signed_var < vec.size() or signed_var < uint_expr
        m = re.search(r'\b(\w+)\s*[<>]=?\s*\w+\.size\(\)', line)
        if m and m.group(1) in int_vars:
            findings.append(Finding(
                file=path, line=i + 1, severity="MEDIUM",
                category="SIG",
                message=f"Signed variable `{m.group(1)}` compared with `.size()` (returns size_t); "
                        "use `static_cast<int>` or change loop var to `size_t`",
                snippet=raw.strip(),
                fix_hint=f"Change `int {m.group(1)}` to `size_t {m.group(1)}` or cast .size() to int",
            ))
    return findings


# ── TRUNC: silent integer truncation ─────────────────────────────────────────
# uint32_t x = some_u64_expression;  or   int x = (a * b) / N;  where a,b look 64-bit
_TRUNC_ASSIGN = re.compile(
    r'\b(uint32_t|int32_t|uint16_t|int16_t|uint8_t|int8_t)\s+(\w+)\s*=\s*(.*?);'
)
_WIDE_EXPR = re.compile(r'\b(uint64_t|int64_t|size_t|long long|__int128)\b')

def check_trunc(path: str, lines: List[str]) -> List[Finding]:
    """Narrow integer type assigned from a 64-bit expression — silent truncation."""
    findings: List[Finding] = []
    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        m = _TRUNC_ASSIGN.search(line)
        if m:
            narrow_type = m.group(1)
            var_name    = m.group(2)
            rhs         = m.group(3)
            # Only flag if the wide type appears OUTSIDE of [...] subscript brackets
            # e.g. `arr[static_cast<size_t>(i)]` — size_t is just the index, not the value
            rhs_no_subscripts = re.sub(r'\[[^\]]*\]', '[]', rhs)
            if _WIDE_EXPR.search(rhs_no_subscripts):
                findings.append(Finding(
                    file=path, line=i + 1, severity="HIGH",
                    category="TRUNC",
                    message=f"`{narrow_type} {var_name}` assigned from a 64-bit expression — "
                            "high 32 bits silently discarded",
                    snippet=raw.strip(),
                    fix_hint=f"Use a 64-bit type or add an explicit `({narrow_type})(...)` cast with a comment confirming the truncation is safe",
                ))
    return findings


# ── OVER: 32-bit multiply with result used as 64-bit ────────────────────────
_OVER_MUL = re.compile(
    r'\b(uint32_t|int32_t|uint|int)\s+\w+\s*=\s*([a-zA-Z_]\w*)\s*\*\s*([a-zA-Z_]\w*)\s*;'
)
_WIDE_DEST = re.compile(r'\b(uint64_t|int64_t|size_t)\s+\w+\s*=\s*([a-zA-Z_]\w*)\s*\*\s*([a-zA-Z_]\w*)\s*;')

def check_overflow(path: str, lines: List[str]) -> List[Finding]:
    """64-bit sink assigned product of two 32-bit variables — overflow before widening."""
    findings: List[Finding] = []
    u32_vars: set[str] = set()
    for ln in lines:
        for m in re.finditer(r'\b(?:uint32_t|int32_t)\s+(\w+)\b', _strip_comments(ln)):
            u32_vars.add(m.group(1))

    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        m = _WIDE_DEST.search(line)
        if m:
            a, b = m.group(2), m.group(3)
            if a in u32_vars and b in u32_vars:
                findings.append(Finding(
                    file=path, line=i + 1, severity="HIGH",
                    category="OVER",
                    message=f"Product of two 32-bit variables `{a} * {b}` assigned to 64-bit — "
                            "multiplication happens in 32-bit domain first, result overflows",
                    snippet=raw.strip(),
                    fix_hint=f"Cast one operand before multiply: `(uint64_t){a} * {b}`",
                ))
    return findings


# ── MSET: memset/memcpy with suspicious literal size ─────────────────────────
_MSET_LITERAL = re.compile(
    r'\b(memset|memcpy|memzero_explicit)\s*\([^,]+,\s*[^,]+,\s*(\d+)\s*\)'
)
# Hard-coded sizes that don't match common crypto buffer sizes.
# Note: 20 (RIPEMD-160), 28 (SHA-224), 48 (SHA-384), 12 (GCM nonce) are
# legitimate crypto sizes and are excluded to keep FP rate down.
# Note: 3 is also a legitimate size (e.g. BIP-324 plaintext header), so it is
# only flagged when paired with non-protocol-specific buffer names (handled
# below by the `bip324`/`header`/`magic` filename/snippet skip).
_SUSPICIOUS_SIZES = {3, 5, 6, 7, 9, 11, 13, 14, 15, 17, 18, 19,
                     40, 56, 60, 100, 200}

def check_mset(path: str, lines: List[str]) -> List[Finding]:
    """memset/memcpy with hard-coded size that doesn't match common crypto buffer sizes."""
    findings: List[Finding] = []
    # Common crypto sizes: 16, 32, 64, 96, 128, 33, 65
    safe_sizes = {16, 32, 33, 64, 65, 96, 128, 256, 384, 512, 1, 2, 4, 8}
    # Files that legitimately use small fixed sizes for protocol headers /
    # magic bytes / version stamps.
    plow = path.lower()
    if any(tag in plow for tag in ('bip324', 'bip32', 'bip352', 'tagged_hash',
                                   'address', 'hd_wallet', 'wallet')):
        return []
    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        # Skip lines that look like they are copying a protocol header /
        # magic / version stamp into a fixed-size field.
        if re.search(r'\b(header|magic|version|prefix|sentinel|marker|tag)\b',
                     line, re.IGNORECASE):
            continue
        m = _MSET_LITERAL.search(line)
        if m:
            fn   = m.group(1)
            size = int(m.group(2))
            if size in _SUSPICIOUS_SIZES:
                findings.append(Finding(
                    file=path, line=i + 1, severity="MEDIUM",
                    category="MSET",
                    message=f"`{fn}` uses hard-coded size {size} — verify this matches the actual buffer/struct size",
                    snippet=raw.strip(),
                    fix_hint=f"Replace literal {size} with sizeof(buffer) or a named constant to avoid silent mismatch if struct changes",
                ))
    return findings


# ── NULL: null-check AFTER dereference ───────────────────────────────────────
_DEREF_EXPR  = re.compile(r'\*\s*([a-zA-Z_]\w*)\s*[;=\[\].,)]')
_ARROW_EXPR  = re.compile(r'([a-zA-Z_]\w*)\s*->')
_NULL_CHECK  = re.compile(r'if\s*\(\s*(!?\s*([a-zA-Z_]\w*)\s*(?:==\s*nullptr|==\s*NULL|!=\s*nullptr|!=\s*NULL)?)\s*\)')

def check_null_after_deref(path: str, lines: List[str]) -> List[Finding]:
    """A pointer is dereferenced before being null-checked — if it is null the deref crashes first."""
    findings: List[Finding] = []
    # Only the negative-form check (`if (!p)`, `if (p == NULL)`, `if (p == nullptr)`)
    # is a real bug indicator. The positive form `if (p) use(p)` is the correct guard
    # pattern and must NOT trigger.
    _NEG_NULL_CHECK = re.compile(
        r'if\s*\(\s*(?:!\s*([a-zA-Z_]\w*)|([a-zA-Z_]\w*)\s*==\s*(?:nullptr|NULL))\s*[\)&|]'
    )
    # Track derefs then check for null within small window
    dereffed: dict[str, int] = {}   # name -> line index
    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        # Function / block end: clear all tracked derefs to prevent crossing scope.
        if raw.strip() == '}':
            dereffed.clear()
            continue
        # Function definition line (top-level `T fn(...) {` or just `{`):
        # any `{` at end of line is also a scope boundary worth resetting at.
        if line.rstrip().endswith('{') and re.match(r'^\s*\S', raw):
            dereffed.clear()
        # Clear old derefs outside window
        dereffed = {k: v for k, v in dereffed.items() if i - v <= 5}

        # Collect dereferences on this line. Filter out:
        #   - multiplications: `a * b`, `(x) * y`
        #   - pointer declarations: `char *p = ...`, `T* p`, `T **p`
        for m in _DEREF_EXPR.finditer(line):
            start = m.start()
            # Walk back past spaces and other `*` (for `**p`) to find the
            # previous meaningful char.
            j = start - 1
            while j >= 0 and line[j] in ' \t*&':
                j -= 1
            prev_ch = line[j] if j >= 0 else ' '
            # If prev is an identifier-ending char or `>` (template),
            # it is a declaration `T *p` or expression `a * b` — not a deref.
            if prev_ch.isalnum() or prev_ch in ('_', ')', ']', '>'):
                continue
            name = m.group(1)
            if name not in ('this', 'result', 'nullptr', 'NULL'):
                dereffed[name] = i

        for m in _ARROW_EXPR.finditer(line):
            name = m.group(1)
            if name not in ('this',):
                dereffed[name] = i

        # Check for negative null check that references a previously dereffed variable
        nc = _NEG_NULL_CHECK.search(line)
        if nc:
            inner = nc.group(1) or nc.group(2)
            if inner and inner in dereffed and dereffed[inner] < i:
                findings.append(Finding(
                    file=path, line=i + 1, severity="HIGH",
                    category="NULL",
                    message=f"Pointer `{inner}` is null-checked here (line {i+1}) but was dereferenced earlier (line {dereffed[inner]+1}) — if null, crash occurs before this check",
                    snippet=raw.strip(),
                    fix_hint=f"Move the null-check for `{inner}` to BEFORE its first dereference",
                ))
    return findings


# ── CPASTE: same lvalue assigned twice without use in between ────────────────
_ASSIGN_STMT = re.compile(r'^\s*([a-zA-Z_]\w*(?:\[[^\]]*\])?)\s*=\s*(?!=)', )
# (not `==`, not `+=`, not `!=`, etc.)
_PREPROC = re.compile(r'^\s*#\s*(if|elif|else|endif|define|undef)\b')

def check_copypaste(path: str, lines: List[str]) -> List[Finding]:
    """Same variable assigned twice with no read in between — second assignment overwrites first."""
    findings: List[Finding] = []
    last_assigned: dict[str, Tuple[int, str]] = {}   # name -> (lineno, snippet)
    for i, raw in enumerate(lines):
        line = _strip_comments(raw).strip()
        # Indexed-assignment lines like `buf[32] = 0x01;` are byte writes,
        # not copy-paste candidates — different indices write different cells.
        # Only treat plain identifiers (no `[`) as candidates.
        # Reset on block boundaries and preprocessor conditionals
        if '{' in line or '}' in line or _PREPROC.match(line):
            last_assigned.clear()
            continue
        # Reset on Python/Ruby/Dart branch and exception-handler keywords —
        # assignments in different conditional branches are not dead writes.
        if re.match(r'^\s*(elif\b|else\s*:|else\s*$|rescue\b|except\b|finally\b|ensure\b)', line):
            last_assigned.clear()
            continue
        # Look for simple assignment
        m = _ASSIGN_STMT.match(line)
        if m:
            lval = m.group(1)
            # Always clear reads on this line FIRST so subsequent skips do
            # not leave stale tracked names.
            for name in list(last_assigned.keys()):
                if name == lval:
                    continue
                if re.search(r'\b' + re.escape(name) + r'\b', line):
                    del last_assigned[name]
            # Skip indexed lvalues (buf[N] = ...) — different indices are
            # different cells and false-positive here.
            if '[' in lval:
                last_assigned.pop(lval, None)
                continue
            # Skip well-known scratch/accumulator variables that are
            # legitimately reassigned without intervening reads in the same
            # statement (e.g. limb arithmetic carry chains).
            if lval in {'carry', 'borrow', 'tmp', 'temp', 'acc', 'accum',
                        't', 't0', 't1', 't2', 't3', 't4', 'r', 'r0', 'r1',
                        'd', 'lo', 'hi', 'acc_lo', 'acc_hi', 'diff', 'sum',
                        'x', 'y', 'z', 'md', 'me', 'sd', 'se', 'fi', 'gi',
                        'cd', 'ce', 'cond_add', 'cond_sub', 'mask', 'X', 'Y'}:
                last_assigned.pop(lval, None)
                continue
            # Skip compound assignments and ==
            if re.match(r'[+\-*/%&|^]=|==', line[m.end()-1:m.end()+1]):
                continue
            # Extract RHS by removing the lval= prefix
            rhs_start = line.find('=') + 1
            rhs = line[rhs_start:]
            # Skip ternary-of-constants pattern (e.g. BIP-32 magic version):
            # `lval = cond ? CONST1 : CONST2;` — common for protocol magic
            # bytes and not a copy-paste bug.
            if re.search(r'\?\s*(?:0x[0-9A-Fa-f]+|[A-Z_][A-Z0-9_]*|\d+)[uUlL]*\s*:'
                         r'\s*(?:0x[0-9A-Fa-f]+|[A-Z_][A-Z0-9_]*|\d+)[uUlL]*',
                         rhs):
                last_assigned.pop(lval, None)
                last_assigned[lval] = (i, raw.strip())
                continue
            # Self-referential: `y = y.square()` — y is read in RHS, not a dead write
            if re.search(r'\b' + re.escape(lval) + r'\b', rhs):
                last_assigned.pop(lval, None)
                last_assigned[lval] = (i, raw.strip())
                continue
            if lval in last_assigned:
                prev_line, prev_snip = last_assigned[lval]
                # Only flag if within 12 lines (likely same block)
                if i - prev_line <= 12:
                    findings.append(Finding(
                        file=path, line=i + 1, severity="MEDIUM",
                        category="CPASTE",
                        message=f"`{lval}` assigned at line {prev_line+1} and again at line {i+1} with no read in between — first assignment is dead",
                        snippet=raw.strip(),
                        fix_hint=f"Remove the redundant assignment at line {prev_line+1}, or check for a copy-paste error where different variables were intended",
                    ))
        # Reset if the variable is read anywhere on this line
        for name in list(last_assigned.keys()):
            if re.search(r'\b' + re.escape(name) + r'\b', line):
                # If it's a read (not another assignment to it), clear it
                if not (m and m.group(1) == name):
                    del last_assigned[name]

        if m:
            last_assigned[m.group(1)] = (i, raw.strip())
    return findings


# ── SEMI: dangling semicolon after for/while/if ──────────────────────────────
_SEMI_KW = re.compile(r'\b(for|while|if)\s*\(')


def _find_matching_paren(s: str, start: int) -> int:
    """Given a string and the index of '(', return the index of the matching ')'.
    Returns -1 if not found.
    """
    depth = 0
    for i in range(start, len(s)):
        if s[i] == '(':
            depth += 1
        elif s[i] == ')':
            depth -= 1
            if depth == 0:
                return i
    return -1


def check_semi(path: str, lines: List[str]) -> List[Finding]:
    """Empty loop or if body — trailing semicolon means the body is always the empty statement."""
    findings: List[Finding] = []
    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        m = _SEMI_KW.search(line)
        if not m:
            continue
        kw = m.group(1)
        # `} while (...)` is the tail of a do-while loop — always ends with `;`, not a bug
        if kw == 'while' and re.search(r'}\s*while\s*\(', line):
            continue
        open_idx = line.index('(', m.start())
        close_idx = _find_matching_paren(line, open_idx)
        if close_idx == -1:
            continue
        # What follows the closing )?
        after = line[close_idx + 1:].lstrip()
        if after.startswith(';'):
            if kw in ('for', 'while'):
                findings.append(Finding(
                    file=path, line=i + 1, severity="HIGH",
                    category="SEMI",
                    message=f"Empty `{kw}` body — trailing `;` means the loop does nothing; the block below is unconditional",
                    snippet=raw.strip(),
                    fix_hint="Remove the `;` if a body was intended, or add `{{}}` to signal the empty body is deliberate",
                ))
            elif kw == 'if' and not re.search(r'\belse\b', line):
                findings.append(Finding(
                    file=path, line=i + 1, severity="MEDIUM",
                    category="SEMI",
                    message="Empty `if` body — trailing `;` after condition; the block below runs unconditionally",
                    snippet=raw.strip(),
                    fix_hint="Remove the `;` if a body was intended",
                ))
    return findings


# ── MBREAK: switch case without break or [[fallthrough]] ────────────────────
_CASE_LINE    = re.compile(r'^\s*case\s+')
_DEFAULT_LINE = re.compile(r'^\s*default\s*:')
_BREAK_THROW  = re.compile(r'\b(break|continue)\s*;|\b(return|throw)\b|\b(abort|exit)\s*\(')
_FALLTHROUGH  = re.compile(r'fallthrough|FALLTHROUGH|fall[\s_-]*through', re.IGNORECASE)

def check_missing_break(path: str, lines: List[str]) -> List[Finding]:
    """Switch case with no break, return, throw, or explicit [[fallthrough]] comment.
    Intentional case grouping (consecutive labels with empty body) is excluded."""
    findings: List[Finding] = []
    in_switch_depth = 0
    brace_depth = 0
    last_case_line: Optional[int] = None
    last_case_snip: str = ""
    has_exit_in_case = False
    case_body_lines = 0   # count of non-empty, non-case lines in this case body

    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        # Track brace depth to stay within the switch
        brace_depth += line.count('{') - line.count('}')

        if 'switch' in line and '{' in line:
            in_switch_depth = brace_depth

        if in_switch_depth > 0 and brace_depth < in_switch_depth:
            in_switch_depth = 0
            last_case_line = None

        if in_switch_depth > 0:
            if _CASE_LINE.match(line) or _DEFAULT_LINE.match(line):
                if last_case_line is not None and not has_exit_in_case:
                    # Only flag if the case body had actual statements (not just labels)
                    if case_body_lines > 0:
                        body_lines = lines[last_case_line:i]
                        if not any(_FALLTHROUGH.search(bl) for bl in body_lines):
                            findings.append(Finding(
                                file=path, line=last_case_line + 1, severity="MEDIUM",
                                category="MBREAK",
                                message=f"Switch case at line {last_case_line+1} falls through to next case at line {i+1} without `break`, `return`, or `[[fallthrough]]`",
                                snippet=last_case_snip,
                                fix_hint="Add `break;` at end of case, or add `[[fallthrough]];` / `// FALLTHROUGH` comment if intentional",
                            ))
                last_case_line = i
                last_case_snip = raw.strip()
                has_exit_in_case = False
                case_body_lines = 0

            if last_case_line is not None:
                stripped = line.strip()
                if stripped and not _CASE_LINE.match(line) and not _DEFAULT_LINE.match(line):
                    case_body_lines += 1
                if _BREAK_THROW.search(line):
                    has_exit_in_case = True

    return findings


# ── RETVAL: discarded return value of security-critical functions ─────────────
_UFSECP_CALL = re.compile(
    r'^\s*(?:ufsecp_\w+|secp256k1_\w+|mbedtls_\w+|RAND_bytes|EVP_\w+|SHA256_\w+|HMAC\w*)\s*\('
)
_ASSIGNED    = re.compile(r'^\s*\w[\w\s*<>:,]+\s*=\s*(?:ufsecp_|secp256k1_|mbedtls_|RAND_bytes|EVP_|SHA256_|HMAC)')
_VOID_FUNC   = re.compile(r'^\s*void\b')
_IF_CHECK    = re.compile(r'if\s*\(')
# secp256k1_sha256 / hash160 / tagged_hash return void in the current ABI
# (SECP256K1_API void ...) — they are infallible hash helpers, not error-returning
# functions.  Include them alongside sha512 and ripemd160 so bare calls are not
# flagged as discarded return values.
_VOID_COMPAT_FN = {
    'secp256k1_sha256', 'secp256k1_hash160', 'secp256k1_tagged_hash',
    'secp256k1_sha512', 'secp256k1_ripemd160',
}

def check_retval(path: str, lines: List[str]) -> List[Finding]:
    """Return value of ufsecp_* / security critical function silently discarded."""
    findings: List[Finding] = []
    # Collect macros defined in this file to skip their invocations
    defined_macros: set[str] = set()
    for ln in lines:
        dm = re.match(r'\s*#\s*define\s+(\w+)\s*\(', ln)
        if dm:
            defined_macros.add(dm.group(1))
    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        # Skip if inside an if-condition, an assignment, or a return statement
        stripped = line.strip()
        if not _UFSECP_CALL.match(stripped):
            continue
        # Skip continuation lines of multi-line expressions
        if not stripped.endswith(';'):
            continue
        # Skip wrapped args (more closing parens than opening — part of outer call)
        if stripped.count(')') > stripped.count('('):
            continue
        # It's a bare call — not used in an expression
        if any(line.lstrip().startswith(prefix) for prefix in ('if (', 'if(', 'return ', 'bool ', 'int ', 'auto ', 'const ')):
            continue
        if _ASSIGNED.match(line):
            continue

        # Extract function name
        fn_m = re.match(r'\s*(\w+)\s*\(', stripped)
        fn_name = fn_m.group(1) if fn_m else stripped[:30]

        # Skip macros defined in the same file (e.g. do{...}while(0) wrappers)
        if fn_name in defined_macros:
            continue
        # Skip known void-returning cleanup and compat hash functions
        if re.search(r'_(destroy|free|cleanup|release|close)\b', fn_name):
            continue
        if fn_name in _VOID_COMPAT_FN:
            continue

        findings.append(Finding(
            file=path, line=i + 1, severity="HIGH",
            category="RETVAL",
            message=f"Return value of `{fn_name}(...)` silently discarded — error conditions will be silently ignored",
            snippet=raw.strip(),
            fix_hint=f"Store the return value and check it: `int rc = {fn_name}(...); if (rc != 0) {{ ... }}`",
        ))
    return findings


# ── LOGIC: && vs || in security-critical conditions ───────────────────────────
# Patterns like:  `if (len == 0 || len > MAX)` is fine;
# `if (result == OK && other_check == OK)` — if it should be || becomes a bypass.
# We flag compound conditions in security rejection contexts where || is suspicious.
_REJECT_CTX  = re.compile(r'\b(reject|invalid|fail|error|bad|wrong|mismatch|tamper|forgery|attack)\b', re.I)
_COND_OR     = re.compile(r'if\s*\(.*\|\|.*\)')

def check_logic(path: str, lines: List[str]) -> List[Finding]:
    """Security rejection condition uses `||` — if any sub-condition alone bypasses the check."""
    findings: List[Finding] = []
    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        # Look for `if (A == B || C == D)` in return / error context
        if not _COND_OR.search(line):
            continue
        # Check surrounding context (3 lines above) for security keywords
        context_above = ' '.join(lines[max(0, i-3):i])
        if _REJECT_CTX.search(context_above) or _REJECT_CTX.search(line):
            # Specifically flag when the condition structure looks like a validation check
            # that should be AND (both conditions must fail to reject) but uses OR
            if re.search(r'if\s*\(.*==.*\|\|.*==.*\)', line):
                findings.append(Finding(
                    file=path, line=i + 1, severity="MEDIUM",
                    category="LOGIC",
                    message="Compound rejection condition uses `||` — verify that either sub-condition alone is sufficient to reject; "
                            "consider whether `&&` is intended (both conditions must fail to reject)",
                    snippet=raw.strip(),
                    fix_hint="Review logic: `||` means ANY condition triggers rejection; `&&` means ALL conditions must match. "
                             "Security checks usually want `||` (any badness = reject), but double-check.",
                ))
    return findings


# ── ZEROIZE: memset clearing fewer bytes than declared buffer ─────────────────
_BUF_DECL = re.compile(
    r'\b(?:uint8_t|char|unsigned char)\s+(\w+)\s*\[(\d+)\]'
)
_MEMZERO = re.compile(
    r'\bmem(?:set|zero(?:_explicit)?)\s*\(\s*(\w+)\s*,\s*0\s*,\s*(\d+)\s*\)'
)

def check_zeroize(path: str, lines: List[str]) -> List[Finding]:
    """memset zeroizes fewer bytes than the buffer was declared with — partial zeroization leak."""
    findings: List[Finding] = []
    # Map name -> declared size
    buf_sizes: dict[str, int] = {}
    for ln in lines:
        for m in _BUF_DECL.finditer(_strip_comments(ln)):
            buf_sizes[m.group(1)] = int(m.group(2))

    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        m = _MEMZERO.search(line)
        if m:
            name = m.group(1)
            size = int(m.group(2))
            if name in buf_sizes and size < buf_sizes[name]:
                # Check if remaining bytes are written by subsequent code
                remaining_written = False
                for j in range(i + 1, min(i + 10, len(lines))):
                    nl = _strip_comments(lines[j])
                    if re.search(r'\b' + re.escape(name) + r'\s*\[', nl):
                        remaining_written = True
                        break
                    if re.search(r'mem(?:set|cpy|move)\s*\(\s*' + re.escape(name) + r'\s*\+', nl):
                        remaining_written = True
                        break
                if remaining_written:
                    continue
                findings.append(Finding(
                    file=path, line=i + 1, severity="HIGH",
                    category="ZEROIZE",
                    message=f"`{name}` declared as [{buf_sizes[name]}] bytes but memset clears only {size} bytes — "
                            f"{buf_sizes[name] - size} bytes of secret data remain in memory",
                    snippet=raw.strip(),
                    fix_hint=f"Change memset size to {buf_sizes[name]} (or sizeof({name})) to clear the entire buffer",
                ))
    return findings


# ── DBLINIT: variable assigned twice with no read in between ─────────────────
# (More specific version than CPASTE — looks for immediate re-assignment on next non-blank line)
_SIMPLE_INIT = re.compile(r'^\s*(?:(?:const\s+)?(?:auto|uint\w+_t|int\w+_t|bool|int|char|size_t|std::\w+)\s+)?(\w+)\s*=\s*(?!=)(.+?);')

def check_dblinit(path: str, lines: List[str]) -> List[Finding]:
    """Variable initialized then immediately re-assigned without reading the first value."""
    findings: List[Finding] = []
    # track (name, init_line, init_snip) — check next assignment
    initialized: dict[str, Tuple[int, str]] = {}

    for i, raw in enumerate(lines):
        line = _strip_comments(raw).strip()
        if not line or line.startswith('//') or line.startswith('/*'):
            continue
        # Skip function declaration / parameter list lines: trailing `,` or `)`
        # WITHOUT `;` (statements always end with `;`). This catches
        # `bool flag = false)` default args and `int x = 0,` parameter splits
        # but does NOT catch normal function calls like `update(&pad, 1);`.
        # Still check for reads (continuation lines can reference tracked vars).
        stripped = line.rstrip()
        if stripped.endswith(',') or (stripped.endswith(')') and not stripped.endswith(';')):
            for tracked in list(initialized.keys()):
                if re.search(r'\b' + re.escape(tracked) + r'\b', line):
                    del initialized[tracked]
            continue
        # Also skip lines that look like parameter-list trailing entries:
        # `... bool x = false);` or `... int x = 0);` — assignment immediately
        # followed by `)` is a default argument, not a statement.
        # But do NOT skip lines where the `=` is inside a function call
        # (e.g. `Assert(rc == 0, "msg");`) since those DO read the variable.
        if re.search(r'=\s*[^;,()]+\)\s*;?\s*$', stripped):
            first_eq    = stripped.find('=')
            first_paren = stripped.find('(')
            # Only skip if there is no opening paren before the `=`
            if first_paren < 0 or first_paren > first_eq:
                for tracked in list(initialized.keys()):
                    if re.search(r'\b' + re.escape(tracked) + r'\b', line):
                        del initialized[tracked]
                continue
        # Block boundaries reset tracking.
        # `{` / `}` are the main boundaries; `else` / `#else` also separate
        # mutually-exclusive paths (single-statement if/else, preprocessor branches).
        if '{' in line or '}' in line:
            initialized.clear()
            continue
        if re.match(r'^\s*(else\b|#else\b|#elif\b)', line):
            initialized.clear()
            continue

        m = _SIMPLE_INIT.match(line)
        if m:
            name = m.group(1)
            rhs  = m.group(2)
            # Remaining text on the same line after the matched statement
            # (multi-statement lines: `m = foo(); use(m.x); m = bar();`)
            rest = line[m.end():]
            # Always clear reads from RHS first to avoid stale tracked names.
            for tracked in list(initialized.keys()):
                if tracked == name:
                    continue
                if re.search(r'\b' + re.escape(tracked) + r'\b', rhs):
                    del initialized[tracked]
            # Also clear tracked variables that appear in the rest of the line
            # (covers multi-statement lines where a variable is assigned then
            # immediately used: `m = mul64_full(a, b); muladd(m.x, m.y, ...);`)
            if rest:
                for tracked in list(initialized.keys()):
                    if tracked == name:
                        continue
                    if re.search(r'\b' + re.escape(tracked) + r'\b', rest):
                        del initialized[tracked]
            # Skip well-known scratch / accumulator names that are repeatedly
            # reassigned in arithmetic blocks (limb chains, carry/borrow propagation).
            if name in {'carry', 'borrow', 'tmp', 'temp', 'acc', 'accum',
                        'd', 'lo', 'hi', 'acc_lo', 'acc_hi', 'diff', 'sum',
                        't', 't0', 't1', 't2', 't3', 't4', 'r', 'r0', 'r1',
                        'x', 'y', 'z', 'md', 'me', 'sd', 'se', 'fi', 'gi',
                        'cd', 'ce', 'cond_add', 'cond_sub', 'mask', 'pad'}:
                initialized.pop(name, None)
                continue
            # Skip self-referential (x = x + 1 etc.)
            if re.search(r'\b' + re.escape(name) + r'\b', rhs):
                initialized.pop(name, None)
                continue
            if name in initialized:
                prev_line, prev_snip = initialized[name]
                # Skip the defensive-zero-init pattern: `T x = 0; ... x = expr;`
                # is idiomatic in C-style code and not a bug.
                if re.match(r'^\s*(?:[\w:]+\s+)?\w+\s*=\s*(?:0|0u|0U|0L|0UL|0x0+|false|nullptr|NULL|\{\}|"")\s*[,;]',
                            prev_snip):
                    initialized[name] = (i, raw.strip())
                    continue
                # Skip if `name` is read in the REST of the same line
                # (multi-statement: `m = foo(); use(m.x);` is NOT a double-init)
                if rest and re.search(r'\b' + re.escape(name) + r'\b', rest):
                    initialized[name] = (i, raw.strip())
                    continue
                if i - prev_line <= 5:   # consecutive or near-consecutive
                    findings.append(Finding(
                        file=path, line=i + 1, severity="LOW",
                        category="DBLINIT",
                        message=f"`{name}` first set at line {prev_line+1} then immediately overwritten at line {i+1} — first value is never read",
                        snippet=raw.strip(),
                        fix_hint=f"Remove the redundant initialization at line {prev_line+1} or check for a copy-paste error",
                    ))
            initialized[name] = (i, raw.strip())
        else:
            # Any line that reads a variable clears it from candidates
            for name in list(initialized.keys()):
                if re.search(r'\b' + re.escape(name) + r'\b', line):
                    del initialized[name]

    return findings


# ── UNREACH: statement after unconditional return/throw/break/continue ────────
_TERMINATOR = re.compile(r'^\s*(return\b|throw\b)\s+.*;')
_LOOP_TERM  = re.compile(r'^\s*(break|continue)\s*;')
_BLANK_OR_BRACE = re.compile(r'^\s*[{}]?\s*$')
_COMMENT_LINE   = re.compile(r'^\s*//')
# A line that looks like the condition of a braceless if/for/while (ends control flow for next line only)
_BRACELESS_CTRL = re.compile(r'\b(if|for|while)\s*\(.*\)\s*$')

def check_unreachable(path: str, lines: List[str]) -> List[Finding]:
    """Non-empty statement appears immediately after an unconditional return/throw
    within the SAME brace scope — uses brace-depth tracking to avoid false positives
    at function boundaries.
    """
    def _preceding_is_braceless_condition(idx: int) -> bool:
        """Return True when the return/throw at `idx` is the single-statement body of
        a braceless if/for/while (including multi-line conditions)."""
        for j in range(idx - 1, max(0, idx - 12), -1):
            cand = _strip_comments(lines[j]).strip()
            if not cand or _COMMENT_LINE.match(cand):
                continue
            # A block-opening brace at END of line signals we're in a proper block,
            # not a braceless condition.  (Avoid triggering on `size_t{32}` etc.)
            if cand.endswith('{'):
                return False
            # Line that plainly opens a scope (just `{` or `} else {`)
            if re.match(r'^[{}]+$', cand) or cand == '{}':
                return False
            if _BRACELESS_CTRL.search(cand):
                return True
            # Any line that BEGINS with if/for/while and has ( is an opener
            if re.match(r'^\s*(if|for|while)\s*\(', cand):
                return True
            if (cand.endswith(')')
                    or cand.endswith('||') or cand.endswith('&&')
                    or cand.endswith('(')
                    or cand.endswith('|') or cand.endswith('&')):
                # Likely a continuation of a multi-line condition — keep scanning
                continue
            if cand.endswith(';') or cand.endswith(','):
                # A prior statement or declaration — not a condition continuation
                return False
            return False
        return False

    findings: List[Finding] = []
    terminated = False
    term_depth = 0
    brace_depth = 0

    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        stripped = line.strip()

        opens  = line.count('{')
        closes = line.count('}')

        if closes > opens:
            terminated = False
        elif opens > closes:
            terminated = False

        if re.search(r'\belse\b', stripped):
            terminated = False
        if stripped.startswith('#'):
            terminated = False

        brace_depth += opens - closes

        if not stripped or _BLANK_OR_BRACE.match(stripped) or _COMMENT_LINE.match(stripped):
            continue

        if terminated and brace_depth == term_depth:
            if not re.match(r'^\s*(case\b|default\s*:|#|\}.*else\b|else\b|catch\b|\}.*catch\b|\}.*finally\b|finally\b)', stripped):
                findings.append(Finding(
                    file=path, line=i + 1, severity="MEDIUM",
                    category="UNREACH",
                    message="Unreachable statement — code after unconditional `return`/`throw` in same scope",
                    snippet=raw.strip(),
                    fix_hint="Remove this dead code or restructure the control flow",
                ))
            terminated = False

        if _TERMINATOR.match(line) or _LOOP_TERM.match(line):
            if _preceding_is_braceless_condition(i):
                terminated = False
            else:
                terminated = True
                term_depth = brace_depth
        elif opens == 0 and closes == 0:
            terminated = False

    return findings


# ──────────────────────────────────────────────────────────────────────────────
# Crypto-specific checkers
# ──────────────────────────────────────────────────────────────────────────────

# ── SECRET_UNERASED: Scalar on stack in signing/key path without secure_erase ─
_SCALAR_DECL   = re.compile(r'\bScalar\s+(\w+)\s*[=;({]')
_SECURE_ERASE  = re.compile(r'secure_erase\s*\(\s*(?:const_cast<[^>]+>\s*\(\s*)?(?:&\s*)?(\w+)')
_SECRET_PATH_KW = {'sign', 'nonce', 'key', 'adapt', 'bip32', 'derive', 'secret', 'blind'}
_TRIVIAL_SCALAR = {'zero', 'one', 'result', 'order', 'half_order', 'n', 'GROUP_ORDER'}

def check_secret_unerased(path: str, lines: List[str]) -> List[Finding]:
    """Scalar local variable in signing/key path not followed by secure_erase."""
    path_lower = path.lower()
    if not any(kw in path_lower for kw in _SECRET_PATH_KW):
        return []
    # GPU private memory is ephemeral — skip OpenCL kernel files
    if path_lower.endswith('.cl'):
        return []
    findings: List[Finding] = []
    full_text = '\n'.join(lines)
    has_dtor_erase = bool(re.search(
        r'~\w+\s*\([^)]*\)\s*\{[^}]*secure_erase', full_text, re.DOTALL))
    scalars: dict[str, int] = {}   # name -> line number
    erased: set[str] = set()
    returned: set[str] = set()
    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        for m in _SCALAR_DECL.finditer(line):
            name = m.group(1)
            if name in _TRIVIAL_SCALAR:
                continue
            # Skip function definitions/declarations: Scalar func_name(params)
            end_pos = m.end()
            if end_pos > 0 and end_pos <= len(line) and line[end_pos - 1] == '(':
                rest = line[end_pos:]
                if rest.lstrip().startswith(')') or re.search(
                        r'\b(const|Scalar|Point|uint\w+_t|int|size_t|Span|auto|void|std::)\b', rest):
                    continue
            # In header files, skip struct/class member and function declarations
            if path_lower.endswith(('.hpp', '.h')):
                stripped = line.strip()
                if re.match(r'^(?:\w+::)*Scalar\s+\w+\s*[;{]', stripped):
                    continue
                if re.match(r'^(?:\w+::)*Scalar\s+\w+\s*\(', stripped):
                    continue
            # Skip variables inside public-data functions (ecrecover, verify, ...)
            is_public_func = False
            context_above = '\n'.join(lines[max(0, i - 30):i + 1])
            if re.search(
                    r'\b(ecrecover|recover|verify|parse|deserialize|decode)\s*\(',
                    context_above):
                is_public_func = True
            if is_public_func:
                continue
            scalars[name] = i + 1
        for m2 in _SECURE_ERASE.finditer(line):
            erased.add(m2.group(1))
        ret_m = re.search(r'\breturn\s+(\w+)\s*;', line)
        if ret_m:
            returned.add(ret_m.group(1))
    for name, lineno in scalars.items():
        if name in erased:
            continue
        if name in returned:
            continue
        if has_dtor_erase and name.endswith('_'):
            continue
        findings.append(Finding(
            file=path, line=lineno, severity="HIGH",
            category="SECRET_UNERASED",
            message=f"Scalar `{name}` in secret path — no secure_erase found before function exit",
            snippet=lines[lineno - 1].strip(),
            fix_hint=f"Add secure_erase(&{name}, sizeof({name})) on all exit paths",
        ))
    return findings


# ── CT_VIOLATION: fast:: call in signing/CT context ───────────────────────────
_FAST_NS_CALL = re.compile(r'\bfast::(scalar_mul|field_mul|field_inv|scalar_inverse|field_sqr)\s*\(')
_CT_FILE_KW   = {'sign', 'nonce', 'ct_', 'secret', 'adaptor', 'blind', 'ecdh', 'derive'}

def check_ct_violation(path: str, lines: List[str]) -> List[Finding]:
    """fast:: namespace function called in file that should use constant-time (ct::) paths."""
    path_lower = path.lower()
    if not any(kw in path_lower for kw in _CT_FILE_KW):
        return []
    findings: List[Finding] = []
    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        m = _FAST_NS_CALL.search(line)
        if m:
            fn = m.group(1)
            findings.append(Finding(
                file=path, line=i + 1, severity="HIGH",
                category="CT_VIOLATION",
                message=f"`fast::{fn}()` in secret-bearing path — must use `ct::{fn}()` for constant-time safety",
                snippet=raw.strip(),
                fix_hint=f"Replace `fast::{fn}(...)` with `ct::{fn}(...)` to ensure CT execution",
            ))
    return findings


# ── TAGGED_HASH_BYPASS: plain sha256() in signing code where tagged hash needed
_PLAIN_SHA256  = re.compile(r'\bsha256\s*\(', re.I)
_TAGGED_HASH   = re.compile(r'tagged_hash|TaggedHash|BIP0340', re.I)
_SIGN_FILE_KW  = {'sign', 'schnorr', 'bip340', 'taproot', 'musig', 'adaptor'}

def check_tagged_hash_bypass(path: str, lines: List[str]) -> List[Finding]:
    """Plain sha256() in BIP-340/signing code where tagged_hash is required for domain separation."""
    path_lower = path.lower()
    if not any(kw in path_lower for kw in _SIGN_FILE_KW):
        return []
    findings: List[Finding] = []
    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        if _PLAIN_SHA256.search(line) and not _TAGGED_HASH.search(line):
            # Check context: if nearby lines use tagged_hash, this one should too
            ctx = ' '.join(lines[max(0, i-5):min(len(lines), i+5)])
            if _TAGGED_HASH.search(ctx) or any(kw in line.lower() for kw in ('nonce', 'challenge', 'aux')):
                findings.append(Finding(
                    file=path, line=i + 1, severity="HIGH",
                    category="TAGGED_HASH_BYPASS",
                    message="Plain `sha256()` in signing code where `tagged_hash(\"BIP0340/...\")` is required "
                            "for domain separation — silent protocol break risk",
                    snippet=raw.strip(),
                    fix_hint="Use tagged_hash(\"BIP0340/...\", data) for BIP-340 domain separation",
                ))
    return findings


# ── RANDOM_IN_SIGNING: non-deterministic random in signing path ───────────────
_RANDOM_CALL   = re.compile(r'\b(getrandom|rand\s*\(|srand|random_bytes|RAND_bytes|arc4random|randombytes)\s*\(')
_AUX_RAND_CTX  = re.compile(r'aux_rand|aux_entropy|extra_entropy|additional_entropy', re.I)

def check_random_in_signing(path: str, lines: List[str]) -> List[Finding]:
    """Random source called in signing path — RFC 6979 is deterministic, random here is suspicious."""
    path_lower = path.lower()
    if not any(kw in path_lower for kw in {'sign', 'nonce', 'rfc6979', 'k_gen'}):
        return []
    findings: List[Finding] = []
    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        m = _RANDOM_CALL.search(line)
        if m:
            fn = m.group(1)
            # Skip if it's the designated aux_rand injection path
            ctx = ' '.join(lines[max(0, i-3):min(len(lines), i+3)])
            if _AUX_RAND_CTX.search(ctx):
                continue
            findings.append(Finding(
                file=path, line=i + 1, severity="HIGH",
                category="RANDOM_IN_SIGNING",
                message=f"`{fn}()` in signing/nonce path — RFC 6979 signing must be deterministic; "
                         "random source here may introduce nonce bias or predictability",
                snippet=raw.strip(),
                fix_hint="Remove the random call or move it to the designated aux_rand/extra_entropy path",
            ))
    return findings


# ── BINDING_NO_VALIDATION: public ufsecp_ function missing NULL-check on args ─
_UFSECP_FUNC_DEF = re.compile(
    r'^(?:ufsecp_error_t|int)\s+(ufsecp_\w+)\s*\('
)
# Renamed from _NULL_CHECK to avoid collision with check_null_after_deref's regex.
# Match any `if (!identifier)` or `if (X == NULL/nullptr)` — broad enough to catch
# all common ABI-style null guards. Also accepts ternary `X ? ... : err` pattern.
# Compound forms like `if ((!a && b > 0) || !c)` are caught by the leading `!\w+`
# alternative because the first `!` precedes a bare identifier inside the
# parenthesised expression.
_BINDING_NULL_CHECK = re.compile(
    r'!\s*\w+'                                  # any !x — handles compound `if ((!a && ..) || !b)`
    r'|\bif\s*\(\s*\w+\s*==\s*(?:NULL|nullptr)' # if (x == NULL)
    r'|\b\w+\s*\?\s*\w+'                        # ctx ? ... ternary guard
)

def check_binding_no_validation(path: str, lines: List[str]) -> List[Finding]:
    """Public ufsecp_ function body does not start with NULL-pointer validation on arguments."""
    if 'impl' not in path.lower() and 'binding' not in path.lower() and 'ufsecp' not in path.lower():
        return []
    # Only applies to C/C++ source files — language binding files may contain
    # C declarations inside FFI string literals, which are not real function bodies.
    _C_CPP_EXT = {'.c', '.cpp', '.cxx', '.cc', '.h', '.hpp', '.hxx', '.cu', '.cuh', '.mm'}
    if not any(path.endswith(ext) for ext in _C_CPP_EXT):
        return []
    findings: List[Finding] = []
    i = 0
    while i < len(lines):
        line = _strip_comments(lines[i])
        m = _UFSECP_FUNC_DEF.match(line.strip())
        if m:
            fn_name = m.group(1)
            # Collect the function signature (across lines until the matching
            # closing paren) to determine whether it accepts pointer arguments
            # at all. If it does not, there is nothing to NULL-check.
            sig_text = line
            for k in range(i + 1, min(len(lines), i + 12)):
                sig_text += ' ' + _strip_comments(lines[k])
                if ')' in sig_text and sig_text.count('(') <= sig_text.count(')'):
                    break
            paren_open = sig_text.find('(')
            paren_close = sig_text.find(')', paren_open + 1) if paren_open != -1 else -1
            sig_args = sig_text[paren_open + 1:paren_close] if paren_close != -1 else ''
            has_pointer_arg = '*' in sig_args or '&' in sig_args
            if not has_pointer_arg:
                i += 1
                continue
            # Scan forward into function body (next 15 lines after opening brace)
            has_null_check = False
            brace_found = False
            for j in range(i, min(len(lines), i + 20)):
                body_line = _strip_comments(lines[j])
                if '{' in body_line:
                    brace_found = True
                if brace_found and _BINDING_NULL_CHECK.search(body_line):
                    has_null_check = True
                    break
                if brace_found and '}' in body_line and body_line.strip() == '}':
                    break  # function ended
            # Skip declaration-only lines (no opening brace within scan window)
            if not brace_found:
                i += 1
                continue
            if brace_found and not has_null_check:
                # Skip trivially-small functions (e.g. version_string)
                if fn_name not in ('ufsecp_version_string', 'ufsecp_version_major',
                                   'ufsecp_version_minor', 'ufsecp_version_patch'):
                    findings.append(Finding(
                        file=path, line=i + 1, severity="MEDIUM",
                        category="BINDING_NO_VALIDATION",
                        message=f"`{fn_name}()` — public ABI function body has no NULL-pointer "
                                "validation on arguments in the first 15 lines",
                        snippet=lines[i].strip(),
                        fix_hint=f"Add `if (!ctx || !out) return UFSECP_ERROR_ARG;` at start of {fn_name}()",
                    ))
        i += 1
    return findings


# ──────────────────────────────────────────────────────────────────────────────
# File collection
# ──────────────────────────────────────────────────────────────────────────────

_SOURCE_EXTENSIONS = {
    # C/C++/CUDA/OpenCL/Metal
    '.cpp', '.cxx', '.cc', '.c', '.cu', '.cuh', '.cl', '.hpp', '.h', '.mm',
    # Language bindings
    '.py', '.go', '.rs', '.swift', '.cs', '.dart', '.rb', '.php', '.js', '.ts',
}

# Directory name components that signal third-party / generated / build code
_SKIP_DIR_PARTS = {
    'node_modules', 'vendor', 'third_party', 'third-party', 'deps',
    'external', '.git', 'build', 'dist', 'out', '__pycache__',
    '_research_repos', 'cmake_install', 'CMakeFiles', 'obj', 'gyp',
    '.build', 'build_rel', 'build_opencl', 'build_ufsecp_swift',
}

# Extra path-substring skip patterns (checked against full path string)
_SKIP_PATH_SUBSTRS = ('node_modules/', 'gyp/', '/obj/', '__pycache__/',)


def _is_skippable(path: Path) -> bool:
    ps = str(path)
    return (any(part in _SKIP_DIR_PARTS for part in path.parts)
            or any(s in ps for s in _SKIP_PATH_SUBSTRS))


# ── DANGLING_ELSE: braceless else/if-else after multi-line block (goto-fail) ──
def check_dangling_else(path: str, lines: List[str]) -> List[Finding]:
    """Braceless else after multi-line if — the Apple 'goto fail' bug pattern."""
    findings: List[Finding] = []
    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        stripped = line.strip()
        # Look for `} else` or standalone `else` without opening brace on same line
        if re.match(r'^\}\s*else\s*$', stripped) or re.match(r'^else\s*$', stripped):
            # Next non-blank line should either be { or a SINGLE statement
            # If there are multiple statements before the next control block,
            # only the first is actually in the else branch — bug!
            body_lines = 0
            for j in range(i + 1, min(len(lines), i + 6)):
                nxt = _strip_comments(lines[j]).strip()
                if not nxt:
                    continue
                if nxt.startswith('{'):
                    break  # braced — fine
                body_lines += 1
                if body_lines >= 2:
                    findings.append(Finding(
                        file=path, line=i + 1, severity="HIGH",
                        category="DANGLING_ELSE",
                        message="Braceless `else` with multiple statements — only the first "
                                "statement is inside the else branch (Apple 'goto fail' pattern)",
                        snippet=raw.strip(),
                        fix_hint="Add braces `{ }` around the else body",
                    ))
                    break
                if nxt.endswith(';') or nxt.endswith('}'):
                    break  # single statement — OK
    return findings


# ── ASSERT_SIDE_EFFECT: side-effectful code inside assert() ───────────────────
_ASSERT_CALL = re.compile(r'\bassert\s*\((.+)\)\s*;')
_SIDE_EFFECT = re.compile(
    r'(\+\+|--|\w+\s*=[^=]|\w+\s*\(|new\s+|delete\s+|free\s*\(|push_back|emplace|insert|erase|pop)')

def check_assert_side_effect(path: str, lines: List[str]) -> List[Finding]:
    """assert() argument has side effects — stripped in Release builds, changing behavior."""
    # In JavaScript/TypeScript, assert(buf.equals(other)) is the idiomatic
    # testing pattern and `Buffer.prototype.equals` has no side effects on state.
    # Flagging JS/TS test assert calls produces only false positives.
    if path.endswith('.js') or path.endswith('.ts'):
        return []
    findings: List[Finding] = []
    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        m = _ASSERT_CALL.search(line)
        if m:
            inner = m.group(1)
            # Skip simple comparisons and boolean checks
            if _SIDE_EFFECT.search(inner):
                # Filter out false positives from comparison operators
                if '==' in inner or '!=' in inner or '>=' in inner or '<=' in inner:
                    # Check if assignment is actually inside comparison
                    cleaned = re.sub(r'[!=<>]=', '', inner)
                    if not _SIDE_EFFECT.search(cleaned):
                        continue
                findings.append(Finding(
                    file=path, line=i + 1, severity="HIGH",
                    category="ASSERT_SIDE",
                    message="assert() contains side-effectful code — will be stripped in "
                            "Release/NDEBUG builds, silently changing program behavior",
                    snippet=raw.strip(),
                    fix_hint="Move the side-effectful code before the assert and test the result",
                ))
    return findings


# ── SHIFT_UB: shift by >= type width or by negative amount ────────────────────
_SHIFT_EXPR = re.compile(
    r'(?:<<|>>)\s*(\d+)')
_TYPE_WIDTH = {
    'uint8_t': 8, 'int8_t': 8, 'uint16_t': 16, 'int16_t': 16,
    'uint32_t': 32, 'int32_t': 32, 'uint64_t': 64, 'int64_t': 64,
    'unsigned': 32, 'int': 32, 'unsigned int': 32, 'size_t': 64,
    'uint': 32,  # OpenCL/Metal
}

def check_shift_ub(path: str, lines: List[str]) -> List[Finding]:
    """Shift by >= type width is undefined behavior in C/C++."""
    findings: List[Finding] = []
    # Detect 128-bit typedefs anywhere in file (e.g. using u128 = __uint128_t;)
    full_text = '\n'.join(lines)
    _128_aliases: set[str] = set()
    for alias_m in re.finditer(
            r'(?:typedef\s+(?:unsigned\s+)?__int128\s+|using\s+)(\w+)\s*=\s*(?:__uint128_t|__int128|unsigned\s+__int128)',
            full_text):
        name = alias_m.group(1)
        if name not in ('typedef', 'using', 'unsigned'):
            _128_aliases.add(name)
    _128_pat = r'__(?:u?int)?128|unsigned\s+__int128'
    if _128_aliases:
        _128_pat += '|\\b(?:' + '|'.join(re.escape(a) for a in _128_aliases) + r')\b'

    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        for sm in _SHIFT_EXPR.finditer(line):
            shift_amount = int(sm.group(1))
            # Skip 128-bit type operations
            if re.search(_128_pat, line):
                continue
            left_of_shift = line[:sm.start()]
            if re.search(_128_pat, left_of_shift):
                continue
            if shift_amount >= 128:
                findings.append(Finding(
                    file=path, line=i + 1, severity="HIGH",
                    category="SHIFT_UB",
                    message=f"Shift by {shift_amount} — undefined behavior for any integer type",
                    snippet=raw.strip(),
                    fix_hint="Use multiplication or a multi-word shift instead",
                ))
            elif shift_amount >= 64:
                # Check context for 128-bit usage
                ctx = '\n'.join(lines[max(0, i - 5):i + 1])
                if re.search(_128_pat, ctx):
                    continue
                # Skip casts from a wider result (common in multi-precision arithmetic)
                if re.search(r'\(\s*(?:std::)?(?:uint64_t|unsigned)\s*\)\s*\(', line):
                    continue
                if re.search(r'static_cast\s*<\s*(?:std::)?uint64_t\s*>', line):
                    continue
                findings.append(Finding(
                    file=path, line=i + 1, severity="HIGH",
                    category="SHIFT_UB",
                    message=f"Shift by {shift_amount} — undefined behavior for 64-bit types "
                            "(OK only for __uint128_t)",
                    snippet=raw.strip(),
                    fix_hint="Verify operand is __uint128_t, or use a multi-word shift",
                ))
            elif shift_amount >= 32:
                # Skip casts extracting high bits from wider types
                if re.search(r'\(\s*(?:std::)?uint32_t\s*\)\s*\(', line):
                    continue
                if re.search(r'static_cast\s*<\s*(?:std::)?uint32_t\s*>\s*\(', line):
                    continue
                # Only flag explicit 32-bit variable declarations with shift on same line
                decl_m = re.match(
                    r'^\s*(?:const\s+)?(?:uint32_t|int32_t|unsigned\s+int)\s+\w+\s*=.*'
                    r'(?:<<|>>)\s*(?:3[2-9]|[4-5]\d|6[0-3])', line)
                if decl_m:
                    findings.append(Finding(
                        file=path, line=i + 1, severity="HIGH",
                        category="SHIFT_UB",
                        message=f"Shift by {shift_amount} on a 32-bit type — undefined behavior",
                        snippet=raw.strip(),
                        fix_hint="Cast to uint64_t before shifting, or use (uint64_t)val << N",
                    ))
    return findings


# ── DOUBLE_LOCK: mutex locked twice without unlock (deadlock) ─────────────────
_LOCK = re.compile(r'\b(\w+)\s*\.\s*lock\s*\(\s*\)')
_UNLOCK = re.compile(r'\b(\w+)\s*\.\s*unlock\s*\(\s*\)')
_LOCK_GUARD = re.compile(r'\b(?:std::)?(?:unique_lock|lock_guard|scoped_lock)\s*.*\b(\w+)\s*[);]')

def check_double_lock(path: str, lines: List[str]) -> List[Finding]:
    """Mutex locked twice without intervening unlock — deadlock."""
    findings: List[Finding] = []
    locked: dict[str, int] = {}  # mutex_name -> line
    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        stripped = line.strip()
        # Reset tracking at function/block boundaries
        if stripped in ('{', '}') or re.match(r'^(void|int|bool|auto|static|inline)\s', stripped):
            locked.clear()
            continue
        # RAII lock_guard — mutex is locked for scope lifetime
        m_guard = _LOCK_GUARD.search(line)
        if m_guard:
            locked.clear()
            continue
        # Explicit lock
        m_lock = _LOCK.search(line)
        if m_lock:
            name = m_lock.group(1)
            if name in locked:
                findings.append(Finding(
                    file=path, line=i + 1, severity="HIGH",
                    category="DOUBLE_LOCK",
                    message=f"Mutex `{name}` locked again without unlock (first lock at line {locked[name]})"
                            " — non-recursive mutex will deadlock",
                    snippet=raw.strip(),
                    fix_hint=f"Add `{name}.unlock()` before re-locking, or use std::recursive_mutex",
                ))
            locked[name] = i + 1
        # Unlock
        m_unlock = _UNLOCK.search(line)
        if m_unlock:
            locked.pop(m_unlock.group(1), None)
    return findings


# ── UNSAFE_SPRINTF: sprintf/vsprintf without bounds check ────────────────────
_UNSAFE_FMT = re.compile(r'\b(sprintf|vsprintf|strcpy|strcat|gets)\s*\(')

def check_unsafe_sprintf(path: str, lines: List[str]) -> List[Finding]:
    """Unbounded string format/copy — use snprintf/strncpy/strlcpy instead."""
    findings: List[Finding] = []
    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        m = _UNSAFE_FMT.search(line)
        if m:
            fn = m.group(1)
            safe = {'sprintf': 'snprintf', 'vsprintf': 'vsnprintf',
                    'strcpy': 'strncpy', 'strcat': 'strncat', 'gets': 'fgets'}
            findings.append(Finding(
                file=path, line=i + 1, severity="HIGH",
                category="UNSAFE_FMT",
                message=f"`{fn}()` has no bounds checking — buffer overflow risk",
                snippet=raw.strip(),
                fix_hint=f"Replace with `{safe.get(fn, fn)}(buf, sizeof(buf), ...)` to enforce bounds",
            ))
    return findings


# ── HARDCODED_SECRET: private key / seed bytes outside test/example/bench ─────
_HEX_PRIVKEY = re.compile(
    r'(?:priv(?:ate)?_?key|secret|seed)\s*(?:\[\s*\d+\s*\])?\s*=\s*\{[^}]{60,}'
)

def check_hardcoded_secret(path: str, lines: List[str]) -> List[Finding]:
    """Hardcoded private key / seed in production code — keys should come from entropy source."""
    path_lower = path.lower()
    # Skip test, example, benchmark, audit, doc files
    if any(kw in path_lower for kw in ('test', 'bench', 'example', 'audit', 'main.c',
                                        'demo', 'sample', 'doc', '.md')):
        return []
    findings: List[Finding] = []
    full = '\n'.join(lines)
    for m in _HEX_PRIVKEY.finditer(full):
        # Find the line number
        pos = m.start()
        lineno = full[:pos].count('\n') + 1
        findings.append(Finding(
            file=path, line=lineno, severity="HIGH",
            category="HARDCODED_SECRET",
            message="Hardcoded private key / seed in production code — use CSPRNG or key derivation",
            snippet=lines[lineno - 1].strip()[:100],
            fix_hint="Remove hardcoded key material; derive from entropy source (RAND_bytes, getrandom, etc.)",
        ))
    return findings


# ── EXCEPTION_SWALLOW: empty catch block that silently ignores errors ─────────
_CATCH_BLOCK = re.compile(r'\bcatch\s*\([^)]*\)\s*\{')

def check_exception_swallow(path: str, lines: List[str]) -> List[Finding]:
    """Empty catch block silently swallows exceptions — errors become invisible."""
    findings: List[Finding] = []
    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        m = _CATCH_BLOCK.search(line)
        if not m:
            continue
        # Same-line catch: `catch (...) { return X; }` — extract content between { }
        # and check if it's non-empty (anything other than a `}` is a handler body).
        brace_idx = m.end() - 1  # index of `{`
        rest = line[brace_idx + 1:]
        # Strip trailing `}` if present on same line
        same_line_close = rest.rfind('}')
        if same_line_close != -1:
            body_inline = rest[:same_line_close].strip()
            if body_inline:
                continue  # has a return/log/etc. inline — not empty
            # Truly empty single-line catch: `catch (...) {}`
            findings.append(Finding(
                file=path, line=i + 1, severity="MEDIUM",
                category="CATCH_EMPTY",
                message="Empty catch block silently swallows exception — error condition is invisible",
                snippet=raw.strip(),
                fix_hint="Log the error, rethrow, or add a comment explaining why ignoring is safe",
            ))
            continue
        # Multi-line catch: scan next lines for first non-blank — must not be `}`.
        for j in range(i + 1, min(len(lines), i + 4)):
            body = _strip_comments(lines[j]).strip()
            if not body:
                continue
            if body == '}':
                findings.append(Finding(
                    file=path, line=i + 1, severity="MEDIUM",
                    category="CATCH_EMPTY",
                    message="Empty catch block silently swallows exception — error condition is invisible",
                    snippet=raw.strip(),
                    fix_hint="Log the error, rethrow, or add a comment explaining why ignoring is safe",
                ))
            break  # non-empty body — fine
    return findings


# ── SIZEOF_MISMATCH: memcpy/memset size argument doesn't match destination ────
_MEMOP_SIZEOF = re.compile(
    r'\bmem(?:cpy|set|move)\s*\(\s*(\w+)\s*,.*?,\s*sizeof\s*\(\s*(\w+)\s*\)\s*\)')

def check_sizeof_mismatch(path: str, lines: List[str]) -> List[Finding]:
    """memcpy/memset sizeof argument refers to wrong variable (src instead of dst)."""
    findings: List[Finding] = []
    # Build a map of array declarations: `T name[SIZE]` per file (best-effort).
    # Two arrays declared with the same SIZE expression are treated as
    # size-compatible to avoid false positives like `memcpy(a_vec, l_x, sizeof(l_x))`
    # where both are `Scalar[RANGE_PROOF_BITS]`.
    array_size_expr: dict[str, str] = {}
    _ARR_DECL = re.compile(r'\b\w[\w:]*\s+(\w+)\s*\[\s*([^\]]+?)\s*\]\s*[;=]')
    for ln in lines:
        s = _strip_comments(ln)
        for m in _ARR_DECL.finditer(s):
            array_size_expr[m.group(1)] = m.group(2).strip()

    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        m = _MEMOP_SIZEOF.search(line)
        if m:
            dst = m.group(1)
            sizeof_arg = m.group(2)
            if dst != sizeof_arg:
                # Only flag if sizeof arg looks like a different variable (not a type)
                if sizeof_arg[0].islower() and sizeof_arg != dst:
                    # Skip if the arg name has a strong type-name suffix
                    # (`_ctx`, `_t`, `_state`, `_struct`, `_type`).
                    if re.search(r'_(?:ctx|t|state|struct|type|info|cfg|opts|hdr)$', sizeof_arg):
                        continue
                    # Skip if both are arrays declared with the same dimension expression.
                    dst_dim = array_size_expr.get(dst)
                    src_dim = array_size_expr.get(sizeof_arg)
                    if dst_dim and src_dim and dst_dim == src_dim:
                        continue
                    findings.append(Finding(
                        file=path, line=i + 1, severity="MEDIUM",
                        category="SIZEOF_MISMATCH",
                        message=f"memop destination is `{dst}` but sizeof references `{sizeof_arg}` "
                                "— possible size mismatch if types differ",
                        snippet=raw.strip(),
                        fix_hint=f"Use sizeof({dst}) to ensure the size matches the destination buffer",
                    ))
    return findings


# ──────────────────────────────────────────────────────────────────────────────
# Cryptography-specific bug pattern checkers (CVE-grounded)
# ──────────────────────────────────────────────────────────────────────────────
#
# Each checker below is rooted in a real-world incident class. Comments cite
# the historical anchor so future maintainers know why the pattern matters.
# ──────────────────────────────────────────────────────────────────────────────

# Files we treat as secret-bearing context for the crypto-specific checkers.
# Path substrings (case-insensitive). We use this to keep false-positive rate
# low: signing/key-derivation paths get the strict treatment, generic helpers
# do not.
_CRYPTO_PATH_HINTS = (
    "sign", "ecdsa", "schnorr", "musig", "frost",
    "adaptor", "ecdh", "bip32", "bip39", "bip340",
    "key", "scalar", "nonce", "rfc6979",
)

def _is_crypto_path(path: str) -> bool:
    p = path.lower()
    return any(h in p for h in _CRYPTO_PATH_HINTS)


# ── NONCE_REUSE_VAR: same nonce variable used in two sign() calls ────────────
# CVE anchor: Sony PS3 ECDSA (2010), bitcoin android wallets (2013).
# Pattern: a single `k` / `nonce` variable assigned once, then passed to two
# distinct sign() calls in the same function scope.
_SIGN_CALL_RE = re.compile(r'\b(\w*sign\w*)\s*\([^;]*?\b(k|nonce|kp|k_val)\b', re.IGNORECASE)

def check_nonce_reuse(path: str, lines: List[str]) -> List[Finding]:
    findings: List[Finding] = []
    if not _is_crypto_path(path):
        return findings
    # Track current function brace depth roughly.
    in_fn = False
    fn_start = 0
    fn_depth = 0
    sign_lines: List[Tuple[int, str, str]] = []  # (lineno, sign_fn_name, nonce_var)
    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        if not in_fn and re.search(r'\b\w+\s*\([^;]*\)\s*\{', line):
            in_fn = True
            fn_start = i
            fn_depth = line.count('{') - line.count('}')
            sign_lines = []
            continue
        if in_fn:
            fn_depth += line.count('{') - line.count('}')
            for m in _SIGN_CALL_RE.finditer(line):
                # filter out function declarations and verify_sign helpers
                if 'verify' in m.group(1).lower():
                    continue
                sign_lines.append((i + 1, m.group(1), m.group(2)))
            if fn_depth <= 0:
                in_fn = False
                # Group by nonce var name; flag if same name appears in 2+ sign calls.
                from collections import defaultdict
                by_nonce: dict[str, List[Tuple[int, str]]] = defaultdict(list)
                for ln, fn_name, nonce in sign_lines:
                    by_nonce[nonce].append((ln, fn_name))
                for nonce, calls in by_nonce.items():
                    if len(calls) >= 2:
                        first_line = calls[0][0]
                        findings.append(Finding(
                            file=path, line=first_line, severity="HIGH",
                            category="NONCE_REUSE_VAR",
                            message=f"nonce/scalar `{nonce}` passed to {len(calls)} sign() calls "
                                    "in the same function — catastrophic key recovery if reused "
                                    "(CVE-2010-... PS3, Bitcoin Android 2013)",
                            snippet=lines[first_line - 1].strip(),
                            fix_hint="Derive a fresh deterministic nonce per signature via RFC 6979 "
                                     "or BIP-340 aux_rand; never reuse a nonce variable across signs.",
                        ))
    return findings


# ── MEMCMP_SECRET: memcmp/strcmp on auth tag, MAC, signature, or HMAC ────────
# CVE anchor: numerous (Xbox 360 Hypervisor, Java SSL, OAuth signature compare).
# Pattern: memcmp() or strcmp() on identifiers whose name suggests auth material.
_AUTH_NAME_RE = re.compile(
    r'\b(?:mac|tag|hmac|auth|digest|signature|sig|expected_(?:mac|tag|hmac)|out_tag|computed_(?:mac|tag))\b',
    re.IGNORECASE,
)
_INSECURE_CMP_RE = re.compile(r'\b(memcmp|strcmp|strncmp|bcmp)\s*\(([^;]+)\)')

def check_memcmp_secret(path: str, lines: List[str]) -> List[Finding]:
    findings: List[Finding] = []
    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        m = _INSECURE_CMP_RE.search(line)
        if not m:
            continue
        args = m.group(2)
        if not _AUTH_NAME_RE.search(args):
            continue
        # Skip if line clearly already in a CT-safe wrapper or a test assertion.
        if re.search(r'\b(ct_(?:memeq|equal)|secp256k1_memcmp_var|EXPECT_|ASSERT_|REQUIRE)', line):
            continue
        findings.append(Finding(
            file=path, line=i + 1, severity="HIGH",
            category="MEMCMP_SECRET",
            message=f"`{m.group(1)}` on apparent auth material — early-exit timing leak "
                    "(forgeable MAC/tag class, Xbox 360 Hypervisor 2007, Java JCE 2014)",
            snippet=raw.strip(),
            fix_hint="Use a constant-time compare (e.g. ct_memeq, secp256k1_memcmp_var-style) "
                     "that always touches the full length.",
        ))
    return findings


# ── MISSING_LOW_S_CHECK: ECDSA verify path without low-S enforcement ─────────
# CVE anchor: BIP-62 / segwit malleability, Bitcoin Cash address scams.
# Pattern: a function name containing "verify" with "ecdsa" context that
# never references low_s, high_s, or s_high.
def check_missing_low_s(path: str, lines: List[str]) -> List[Finding]:
    findings: List[Finding] = []
    if 'ecdsa' not in path.lower() and 'verify' not in path.lower():
        return findings
    # Skip header files — they contain declarations only; the check must run on
    # the corresponding .cpp implementation file that has the actual function body.
    if path.endswith(('.hpp', '.h', '.hxx', '.hh')):
        return findings
    text = '\n'.join(lines)
    # Only flag once per file.
    if 'verify' not in text:
        return findings
    if re.search(r'(low_s|high_s|s_high|half_n|half_order|is_low|is_high)', text):
        return findings
    # Look for a verify-style function definition (not a forward declaration).
    # A declaration ends with ';' on the same line; a definition has a body '{'.
    for i, raw in enumerate(lines):
        if re.search(r'\b\w*ecdsa\w*verify\w*\s*\(', raw, re.IGNORECASE):
            # Skip forward declarations (line ends with ';' and has no '{')
            stripped = raw.strip()
            if stripped.endswith(';') and '{' not in stripped:
                continue
            findings.append(Finding(
                file=path, line=i + 1, severity="MEDIUM",
                category="MISSING_LOW_S_CHECK",
                message="ECDSA verify path has no low-S / high-S reference — signature "
                        "malleability risk (BIP-62, segwit policy)",
                snippet=stripped,
                fix_hint="Reject signatures with s > n/2 in policy-strict verify paths; "
                         "or document a deliberate non-strict mode.",
            ))
            break
    return findings


# ── SCALAR_FROM_RAND: signing nonce derived from rand()/random()/gettimeofday ─
# CVE anchor: Debian OpenSSL CVE-2008-0166, Android Bitcoin SecureRandom 2013.
_BAD_RNG_RE = re.compile(
    r'\b(rand|random|rand_r|drand48|gettimeofday|time)\s*\(\s*\)'
)
_NONCE_LHS_RE = re.compile(r'\b(k|nonce|k_value|secret_nonce)\s*=', re.IGNORECASE)

def check_scalar_from_rand(path: str, lines: List[str]) -> List[Finding]:
    findings: List[Finding] = []
    if not _is_crypto_path(path):
        return findings
    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        if not _NONCE_LHS_RE.search(line):
            continue
        if _BAD_RNG_RE.search(line):
            findings.append(Finding(
                file=path, line=i + 1, severity="HIGH",
                category="SCALAR_FROM_RAND",
                message="Signing nonce assigned from non-cryptographic RNG / timestamp "
                        "(CVE-2008-0166 Debian OpenSSL, Android Bitcoin 2013)",
                snippet=raw.strip(),
                fix_hint="Use RFC 6979 deterministic nonce or a properly-seeded CSPRNG; "
                         "never derive scalar from time() or rand().",
            ))
    return findings


# ── GOTO_FAIL_DUPLICATE: two adjacent identical `goto label;` statements ─────
# CVE anchor: Apple SSL/TLS goto fail (CVE-2014-1266).
def check_goto_fail(path: str, lines: List[str]) -> List[Finding]:
    findings: List[Finding] = []
    prev = ""
    for i, raw in enumerate(lines):
        line = _strip_comments(raw).strip()
        m = re.match(r'goto\s+(\w+)\s*;', line)
        if m and prev == line:
            findings.append(Finding(
                file=path, line=i + 1, severity="HIGH",
                category="GOTO_FAIL_DUPLICATE",
                message=f"Duplicate `{line}` on consecutive lines — Apple goto-fail "
                        "anti-pattern (CVE-2014-1266); the second goto skips intended checks",
                snippet=raw.strip(),
                fix_hint="Remove the duplicate goto; verify the intended control flow "
                         "executes the validation step that was skipped.",
            ))
        prev = line
    return findings


# ── POINT_NO_VALIDATION: pubkey accept without point_on_curve / is_valid ─────
# CVE anchor: invalid-curve attacks (Brainpool 2015, OpenSSL CVE-2015-1788).
def check_point_no_validation(path: str, lines: List[str]) -> List[Finding]:
    # Language binding files (Python/Ruby/Rust/Go/etc.) delegate to the C ABI
    # `ufsecp_pubkey_parse`, which already performs on-curve validation inside
    # the library.  Flagging binding wrappers as missing the check produces
    # only false positives.  Only apply this checker to C/C++ implementation files.
    _C_EXT = {'.c', '.cpp', '.cxx', '.cc', '.h', '.hpp', '.cu', '.cuh', '.mm'}
    if not any(path.endswith(ext) for ext in _C_EXT):
        return []
    findings: List[Finding] = []
    text = '\n'.join(lines)
    has_pub_parse = re.search(r'\b(pubkey_parse|pubkey_load|pubkey_from_bytes|deserialize_pubkey)\b', text)
    if not has_pub_parse:
        return findings
    has_validation = re.search(
        r'\b(on_curve|is_on_curve|point_on_curve|is_valid_point|validate_pubkey|'
        r'check_subgroup|in_subgroup|point_validate)\b', text)
    if has_validation:
        return findings
    # Locate the parse function for line reporting.
    for i, raw in enumerate(lines):
        if re.search(r'\b(pubkey_parse|pubkey_load)\s*\(', raw):
            findings.append(Finding(
                file=path, line=i + 1, severity="MEDIUM",
                category="POINT_NO_VALIDATION",
                message="pubkey parser present but no on-curve / subgroup validation found "
                        "in the same file — invalid-curve attack risk",
                snippet=raw.strip(),
                fix_hint="Verify the parsed point lies on secp256k1 and (where applicable) "
                         "in the prime-order subgroup before exposing it to scalar mul.",
            ))
            break
    return findings


# ── ECDH_OUTPUT_NOT_CHECKED: ECDH result used without zero / identity check ──
# CVE anchor: small-subgroup confinement, twist attacks.
def check_ecdh_output(path: str, lines: List[str]) -> List[Finding]:
    findings: List[Finding] = []
    if 'ecdh' not in path.lower():
        return findings
    text = '\n'.join(lines)
    has_ecdh = re.search(r'\bufsecp_ecdh\b|\b(ecdh|x25519)\s*\(', text)
    if not has_ecdh:
        return findings
    if re.search(r'\b(is_zero|is_identity|all_zero|secp256k1_memcmp_var.*zero|fe_is_zero)\b', text):
        return findings
    # Locate the ecdh callsite.
    for i, raw in enumerate(lines):
        if re.search(r'\becdh\s*\(', raw, re.IGNORECASE):
            findings.append(Finding(
                file=path, line=i + 1, severity="MEDIUM",
                category="ECDH_OUTPUT_NOT_CHECKED",
                message="ECDH output used without zero/identity check — small-subgroup or "
                        "twist attack can produce predictable shared secrets",
                snippet=raw.strip(),
                fix_hint="Reject ECDH outputs that are the point at infinity or have a zero "
                         "x-coordinate; document the policy in the function header.",
            ))
            break
    return findings


# ── HASH_NO_DOMAIN_SEP: SHA256 in BIP-340/Schnorr/Taproot context without tag ─
# CVE anchor: cross-protocol signature reuse (BIP-340 §3.2).
_PLAIN_SHA256_RE = re.compile(r'\b(sha256_init|sha256_update|sha256\b)\s*\(')

def check_no_domain_sep(path: str, lines: List[str]) -> List[Finding]:
    findings: List[Finding] = []
    if not re.search(r'(schnorr|bip340|taproot|musig|frost)', path, re.IGNORECASE):
        return findings
    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        if not _PLAIN_SHA256_RE.search(line):
            continue
        # Skip if same line or surrounding 5 lines reference tagged_hash.
        ctx = '\n'.join(lines[max(0, i - 5): min(len(lines), i + 6)])
        if 'tagged_hash' in ctx.lower() or 'tag_hash' in ctx.lower() or 'taghash' in ctx.lower():
            continue
        if 'sha256_initialize_tagged' in ctx.lower():
            continue
        findings.append(Finding(
            file=path, line=i + 1, severity="MEDIUM",
            category="HASH_NO_DOMAIN_SEP",
            message="Plain SHA-256 in BIP-340/Schnorr/Taproot file without tagged_hash "
                    "context — cross-protocol signature reuse risk (BIP-340 §3.2)",
            snippet=raw.strip(),
            fix_hint="Use sha256_initialize_tagged(\"BIP0340/...\") or the equivalent "
                     "tagged-hash helper instead of plain SHA-256 in BIP-340 context.",
        ))
    return findings


# ── DER_LAX_PARSE: short-circuit / shortcut DER parsing ──────────────────────
# CVE anchor: BER-vs-DER ambiguity, OpenSSL CVE-2014-8275.
def check_der_lax(path: str, lines: List[str]) -> List[Finding]:
    findings: List[Finding] = []
    if 'der' not in path.lower() and 'parse' not in path.lower() and 'sig' not in path.lower():
        return findings
    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        # length-byte taken without strict canonicalisation check
        if re.search(r'\blen\s*=\s*\*?\(?(?:input|buf|p|sig)\)?\s*\+\+', line):
            ctx = '\n'.join(lines[max(0, i - 3): min(len(lines), i + 8)])
            if not re.search(r'(strict|canonical|fail.*long|>\s*0x7f)', ctx, re.IGNORECASE):
                findings.append(Finding(
                    file=path, line=i + 1, severity="MEDIUM",
                    category="DER_LAX_PARSE",
                    message="DER length byte read without canonicalisation guard — BER/DER "
                            "ambiguity (CVE-2014-8275 OpenSSL)",
                    snippet=raw.strip(),
                    fix_hint="Reject long-form length encodings where short-form would suffice; "
                             "reject leading zeros and trailing data.",
                ))
    return findings


# ── TIMING_BRANCH_ON_KEY: `if (key[i] ...)` direct branching on secret byte ──
# CVE anchor: cache-timing attacks (Bernstein 2005 AES, etc.).
_KEY_INDEX_BRANCH_RE = re.compile(
    r'\bif\s*\(\s*(?:secret|priv(?:key)?|key|seckey|sk|d|x_only_seckey)\s*\[\s*\w+\s*\]')

def check_timing_branch_on_key(path: str, lines: List[str]) -> List[Finding]:
    findings: List[Finding] = []
    if not _is_crypto_path(path):
        return findings
    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        if _KEY_INDEX_BRANCH_RE.search(line):
            findings.append(Finding(
                file=path, line=i + 1, severity="HIGH",
                category="TIMING_BRANCH_ON_KEY",
                message="Direct branch on a secret-key byte — cache/branch timing leak "
                        "(Bernstein AES 2005 class)",
                snippet=raw.strip(),
                fix_hint="Replace the branch with a constant-time mask/select; route the "
                         "operation through the CT layer.",
            ))
    return findings


# ── MAC_TRUNCATION: MAC compared with shortened length ───────────────────────
# CVE anchor: GCM-trunc misuse, JOSE alg=none class.
def check_mac_truncation(path: str, lines: List[str]) -> List[Finding]:
    findings: List[Finding] = []
    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        m = re.search(r'\b(memcmp|strncmp|bcmp)\s*\([^,]+,[^,]+,\s*(\d+)\s*\)', line)
        if not m:
            continue
        n = int(m.group(2))
        if n in (1, 2, 4, 8) and _AUTH_NAME_RE.search(line):
            findings.append(Finding(
                file=path, line=i + 1, severity="HIGH",
                category="MAC_TRUNCATION",
                message=f"Auth-material compare with only {n} bytes — truncated MAC accepts "
                        "forged tags with 2^{8*n} collisions",
                snippet=raw.strip(),
                fix_hint="Compare the full MAC/tag length using a constant-time helper.",
            ))
    return findings


# ── SCALAR_NOT_REDUCED: scalar without explicit `mod n` reduction step ───────
# CVE anchor: Stark Bank ECDSA r∈[n,p−1] (CVE-2024-... class), bias attacks.
def check_scalar_not_reduced(path: str, lines: List[str]) -> List[Finding]:
    findings: List[Finding] = []
    if not _is_crypto_path(path):
        return findings
    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        # Look for divisions/inverses without an adjacent reduction comment or call.
        # Skip plain declarations / forward declarations (no function call on this line —
        # e.g. `Scalar scalar_inverse(const Scalar& a) noexcept;`).
        if re.search(r'\bscalar_inverse\s*\(', line):
            # A forward declaration ends with `;` and contains no assignment, no body.
            stripped = line.strip()
            if stripped.endswith(';') and '=' not in stripped and '{' not in stripped:
                continue
            ctx = '\n'.join(lines[max(0, i - 4): min(len(lines), i + 4)])
            if not re.search(r'\b(scalar_reduce|scalar_set_b32|set_int|is_zero|zero_check)\b', ctx):
                findings.append(Finding(
                    file=path, line=i + 1, severity="MEDIUM",
                    category="SCALAR_NOT_REDUCED",
                    message="scalar_inverse without adjacent reduction or zero check — risk of "
                            "inverting an unreduced or zero scalar",
                    snippet=raw.strip(),
                    fix_hint="Reduce the scalar mod n and reject is_zero before inversion.",
                ))
    return findings


# ── PRINTF_SECRET: format-string output of secret-bearing identifier ─────────
# CVE anchor: developer log leakage class (countless in mobile wallets).
_PRINTF_RE = re.compile(r'\b(printf|fprintf|cout|std::cerr|std::cout|debug_print|LOG)\s*[<\(]')
_SECRET_NAME_RE = re.compile(
    r'\b(seckey|secret_key|priv(?:_?key)?|seed|mnemonic|nonce_secret|d_scalar|secret_scalar|aux_rand)\b',
    re.IGNORECASE,
)

def check_printf_secret(path: str, lines: List[str]) -> List[Finding]:
    findings: List[Finding] = []
    # Skip test files where printing is expected.
    if re.search(r'(test|fuzz|bench)', path, re.IGNORECASE):
        return findings
    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        if _PRINTF_RE.search(line) and _SECRET_NAME_RE.search(line):
            findings.append(Finding(
                file=path, line=i + 1, severity="HIGH",
                category="PRINTF_SECRET",
                message="Output statement appears to print a secret-bearing identifier "
                        "(log/console leak class)",
                snippet=raw.strip(),
                fix_hint="Never log raw secret material; if debugging, log only a "
                         "domain-separated, length-bounded hash, and gate behind "
                         "a build-time DEBUG_SECRETS=0 guard.",
            ))
    return findings


# ── JNI: GetByteArrayElements / GetStringUTFChars without NULL check ──────────
_JNI_GET_BYTES = re.compile(
    r'\b(GetByteArrayElements|GetStringUTFChars|GetIntArrayElements|GetLongArrayElements)\s*\('
)
_JNI_FILE = re.compile(r'jni.*\.[ch]$|_jni\.[ch]$|_jni\.cpp$|jni.*\.cpp$', re.I)

def check_jni_getbytes_null(path: str, lines: List[str]) -> List[Finding]:
    """JNI GetByteArrayElements / GetStringUTFChars return value used without NULL check.

    These JNI functions CAN return NULL on out-of-memory.  Dereferencing the
    result without a prior null-check is a crash / undefined-behaviour bug.
    """
    if not _JNI_FILE.search(path):
        return []
    findings: List[Finding] = []
    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        # Assignment: `jbyte *ptr = (*env)->GetByteArrayElements(...);`
        # or C++ style: `env->GetByteArrayElements(...)`
        assign_m = re.match(
            r'^\s*(?:j\w+\s*\*+|const\s+\w+\s*\*+|\w+\s*\*+)\s*(\w+)\s*=\s*'
            r'(?:\(\s*\*env\s*\)\s*->|env\s*->)\s*'
            r'(GetByteArrayElements|GetStringUTFChars|GetIntArrayElements|GetLongArrayElements)\s*\(',
            line
        )
        if assign_m:
            var = assign_m.group(1)
            # Check if the VERY NEXT non-blank line (within 3 lines) does an
            # immediate null check  `if (!var)` or `if (var == NULL)`
            has_check = False
            for j in range(i + 1, min(len(lines), i + 4)):
                nxt = _strip_comments(lines[j]).strip()
                if not nxt:
                    continue
                if re.search(r'if\s*\(\s*!' + re.escape(var) + r'\s*\)', nxt):
                    has_check = True
                if re.search(re.escape(var) + r'\s*==\s*(NULL|nullptr|0)\s*\)', nxt):
                    has_check = True
                break
            if not has_check:
                findings.append(Finding(
                    file=path, line=i + 1, severity="MEDIUM",
                    category="JNI_NULL",
                    message=f"JNI `{assign_m.group(2)}(...)` result stored in `{var}` "
                            "without an immediate NULL check — crash if JVM is out of memory",
                    snippet=raw.strip(),
                    fix_hint=f"Add `if (!{var}) {{ throw_exc(env, \"OOM\"); return NULL; }}` "
                             f"immediately after the {assign_m.group(2)}() call",
                ))
    return findings


# ── FFI_RETVAL_IGNORED: language-binding callers that discard a fallible API
#    return value without checking it.
#
#  Covers: Python ctypes, C# P/Invoke, Swift, Go CGo, Rust FFI, Ruby FFI,
#          Dart FFI, PHP FFI.
#
#  A "fallible" call is any call to secp256k1_* / ufsecp_* that is NOT in the
#  known-infallible set (void functions: sha256, hash160, tagged_hash,
#  ctx_destroy, gpu_ctx_destroy, version, abi_version, version_string,
#  last_error, last_error_msg).
#
#  The checker looks for a call on a line that is NOT part of:
#    - an assignment (rc =, result =, ret =, _ =, let/var/const declaration)
#    - a guard/if condition (guard rc, if rc, if (rc)
#    - a return statement whose value is itself the call result
#    - a throw/raise wrapper that RECEIVES the call as argument
#    - a function whose purpose is infallible (void)
#    - a test/bench file (where crash-on-error is intentional)
#
# ─────────────────────────────────────────────────────────────────────────────

# Functions that are void / always-succeed — never flag these
_FFI_INFALLIBLE = re.compile(
    r'(?:secp256k1_|ufsecp_)'
    r'(?:sha256|hash160|tagged_hash|ctx_destroy|gpu_ctx_destroy|'
    r'abi_version|version_string|version\b|last_error_msg|last_error\b)',
    re.IGNORECASE,
)

# Per-language: pattern that identifies a "guarded" context around the call
# i.e. the return value IS being captured or checked within a ±2-line window.

# ── Python (ctypes) ──────────────────────────────────────────────────────────
_PY_CALL_RE   = re.compile(r'\bself\._lib\.(secp256k1_|ufsecp_)\w+\(')
_PY_GUARD_RE  = re.compile(
    r'^\s*(rc|result|ret|_rc)\s*='           # rc = lib.fn(...)
    r'|self\._throw\s*\('                     # self._throw(self._lib.fn(...), ...)
    r'|self\._lib\.\w+\([^)]*\)\s*==\s*\d'   # direct comparison
    r'|return\s+self\._lib\.'                 # return lib.fn(...)
    r'|\bif\s+.*self\._lib\.',                # if lib.fn(...) ...
    re.MULTILINE,
)

# ── C# P/Invoke (both legacy secp256k1_ and new ufsecp_) ─────────────────────
_CS_CALL_RE   = re.compile(r'\bNative\.(secp256k1_|ufsecp_)\w+\(')
_CS_GUARD_RE  = re.compile(
    r'^\s*(int|var|rc|result|ret)\s+\w+\s*='  # int rc = Native.fn(...);
    r'|Throw\s*\('                              # Throw(Native.fn(...), ...)
    r'|Chk\s*\('                               # Chk(Native.fn(...), ...)
    r'|throw\s+'                               # throw new...
    r'|\bif\s*\(rc'                            # if (rc ...
    r'|return\s+Native\.'                      # return Native.fn(...)
    r'|Assert\s*\('                            # Assert(rc == 0, ...)
    r'|_\s*='                                   # _ = (discard explicitly)
    r'|var\s+_',                               # var _ =
    re.MULTILINE,
)

# ── Swift ─────────────────────────────────────────────────────────────────────
_SWIFT_CALL_RE  = re.compile(r'\bCUltrafastSecp256k1\.(secp256k1_|ufsecp_)\w+\(')
_SWIFT_GUARD_RE = re.compile(
    r'^\s*let\s+rc\s*='                       # let rc = ...withUnsafe...
    r'|^\s*let\s+rc\s*:'                      # let rc: Int32 (block-level decl)
    r'|guard\s+rc\s*=='                       # guard rc == 0 else
    r'|guard\s+rc\s*!='                       # guard rc != 0 else
    r'|return\s+.*==\s*\d'                    # return ... == 1
    r'|String\(cString'                       # String(cString: version())
    r'|==\s*1\b'                              # ... == 1
    r'|!=\s*0\b',                             # ... != 0
    re.MULTILINE,
)

# ── Go CGo ────────────────────────────────────────────────────────────────────
_GO_CALL_RE   = re.compile(r'\bC\.(secp256k1_|ufsecp_)\w+\(')
_GO_GUARD_RE  = re.compile(
    r':='                                      # rc := C.fn(...)
    r'|errFromCode'                            # errFromCode(C.fn(...))
    r'|return\s+C\.'                           # return C.fn(...)  (passes rc to caller)
    r'|\bif\s+.*:=\s*C\.'                     # if rc := C.fn(...); rc != OK
    r'|C\.free'                               # C.free (cleanup, not fallible)
    r'|C\.secp256k1_tagged_hash',             # void — already in infallible set
    re.MULTILINE,
)

# ── Rust (unsafe ufsecp_sys::) ────────────────────────────────────────────────
_RS_CALL_RE   = re.compile(r'\bufsecp_sys::(secp256k1_|ufsecp_)\w+\(')
_RS_GUARD_RE  = re.compile(
    r'chk\s*\('                               # chk(ufsecp_sys::fn(...), ...)
    r'|let\s+rc\s*='                          # let rc = unsafe { ufsecp_sys::fn(...) }
    r'|==\s*0'                                # ... == 0
    r'|!=\s*0'                                # ... != 0
    r'|ufsecp_sys::ufsecp_ctx_destroy'        # void
    r'|\bResult::',
    re.MULTILINE,
)

# ── Ruby FFI ──────────────────────────────────────────────────────────────────
_RB_CALL_RE   = re.compile(r'\bNative\.(secp256k1_|ufsecp_)\w+\(')
_RB_GUARD_RE  = re.compile(
    r'^\s*rc\s*='                              # rc = Native.fn(...)
    r'|raise\s+'                              # raise Error unless rc.zero?
    r'|\bif\s+.*\brc\b'                       # if rc != 0
    r'|unless\s+rc'                           # unless rc.zero?
    r'|\brc\.zero\?'                          # rc.zero?
    r'|\brc\s*!=\b'                           # rc != 0
    r'|\brc\s*==\b',                          # rc == 0
    re.MULTILINE,
)

# ── PHP FFI ──────────────────────────────────────────────────────────────────
_PHP_CALL_RE   = re.compile(r'->(?:ffi->|)(secp256k1_|ufsecp_)\w+\(')
_PHP_GUARD_RE  = re.compile(
    r'^\s*\$rc\s*='                           # $rc = $this->ffi->fn(...)
    r'|^\s*\$result\s*='
    r'|throw\s+new'                           # throw new RuntimeException
    r'|\bif\s*\(\s*\$rc'                      # if ($rc !== 0)
    r'|return\s+.*===\s*\d'                   # return $this->ffi->fn(...) === 1
    r'|return\s+.*!==\s*\d'
    r'|\$abi\s*='                             # $abi = ... version check
    r'|return\s+\$this->ffi->',              # direct return of fn value
    re.MULTILINE,
)

# ── Dart FFI ─────────────────────────────────────────────────────────────────
_DART_CALL_RE   = re.compile(r'\b_lib\.(secp256k1_|ufsecp_)\w+\(')
_DART_GUARD_RE  = re.compile(
    r'^\s*(final|int|var)\s+\w+\s*='
    r'|_chk\s*\('
    r'|_throw\s*\('
    r'|throw\s+'
    r'|\bif\s*\(rc',
    re.MULTILINE,
)

_FFI_LANG_CONFIGS = [
    # (file-extension-set, call_re, guard_re, skip-test-files?)
    ({'.py'},    _PY_CALL_RE,    _PY_GUARD_RE,    False),
    ({'.cs'},    _CS_CALL_RE,    _CS_GUARD_RE,    False),
    ({'.swift'}, _SWIFT_CALL_RE, _SWIFT_GUARD_RE, False),
    ({'.go'},    _GO_CALL_RE,    _GO_GUARD_RE,    False),
    ({'.rs'},    _RS_CALL_RE,    _RS_GUARD_RE,    False),
    ({'.rb'},    _RB_CALL_RE,    _RB_GUARD_RE,    False),
    ({'.php'},   _PHP_CALL_RE,   _PHP_GUARD_RE,   False),
    ({'.dart'},  _DART_CALL_RE,  _DART_GUARD_RE,  False),
]

_FFI_SKIP_PATHS = re.compile(
    r'(test|spec|smoke|bench|__init__|build\.rs|/sys/)',
    re.IGNORECASE,
)


def check_ffi_retval_ignored(path: str, lines: List[str]) -> List[Finding]:
    """FFI binding call to a fallible secp256k1_*/ufsecp_* function whose
    return value is silently discarded (not assigned, not checked, not thrown).

    Covers Python, C#, Swift, Go, Rust, Ruby, PHP, Dart bindings.
    """
    suffix = Path(path).suffix.lower()
    cfg = next((c for c in _FFI_LANG_CONFIGS if suffix in c[0]), None)
    if cfg is None:
        return []
    _, call_re, guard_re, _ = cfg

    # Skip test / smoke / bench files
    if _FFI_SKIP_PATHS.search(path):
        return []

    findings: List[Finding] = []
    for i, raw in enumerate(lines):
        line = raw
        # Must contain a fallible FFI call
        if not call_re.search(line):
            continue
        # Skip infallible void functions
        if _FFI_INFALLIBLE.search(line):
            continue
        # Skip comment / doc lines
        stripped = line.strip()
        if stripped.startswith(('//','#','*','///','--','<?','"""', "'''")):
            continue
        # Skip DllImport / attach_function / extern declarations
        if re.search(r'(DllImport|extern\s+|attach_function|declare\s+function'
                     r'|^\s*\[DllImport|@DllImport|unsafe\s+extern)', line):
            continue
        # Skip lines that ARE a return statement or comparison
        if re.search(r'^\s*return\s+.*==\s*\d|^\s*return\s+.*!=\s*\d', line):
            continue
        # Skip if the call itself is part of a comparison expression (same-line check)
        # e.g. `Native.fn(...) == 1` or `C.fn(...) != 0`
        if re.search(r'(secp256k1_|ufsecp_)\w+\([^)]*\)\s*(==|!=)\s*\d', line):
            continue
        # Also check next 3 lines for closing paren + comparison (multi-line call)
        fwd_ctx = '\n'.join(lines[i:min(len(lines), i + 4)])
        if re.search(r'\)\s*(==|!=)\s*\d', fwd_ctx):
            continue

        # Check a WIDE context window (±8 lines) for a guard that is in the
        # SAME function body as the call.  We collect ALL guard matches and
        # accept the call as "guarded" only if at least one match has no
        # function-definition boundary between it and the call line.
        wide_start = max(0, i - 8)
        wide_end   = min(len(lines), i + 8)
        wide_ctx   = '\n'.join(lines[wide_start:wide_end])
        _FN_BOUNDARY = re.compile(
            r'^\s*(def |function |pub fn |fn \w|public function|private function'
            r'|protected function|func \w|impl \w)',
        )
        guarded = False
        for m in guard_re.finditer(wide_ctx):
            guard_line_offset = wide_ctx[:m.start()].count('\n')
            guard_abs = wide_start + guard_line_offset
            if guard_abs < i:
                between = lines[guard_abs:i]
            elif guard_abs > i:
                between = lines[i:guard_abs]
            else:
                # Guard is on the same line as the call → definitely guarded
                guarded = True
                break
            if not any(_FN_BOUNDARY.search(bl) for bl in between):
                guarded = True
                break
        if guarded:
            continue

        # Skip if the call is INSIDE a closure/callback argument:
        # e.g. Ruby: get_address { |buf, len| Native.fn(...) }
        # e.g. PHP:  getAddress(fn($buf, $len) => $this->ffi->fn(...))
        # e.g. Swift: getAddress { buf, len in ... CUltrafastSecp256k1.fn(...) }
        # e.g. Rust:  self.get_addr(|ctx, pk, n, buf, len| unsafe { ufsecp_sys::fn(...) })
        # Detect by checking current line or 1-5 lines above for a helper that receives closure
        _CALLBACK_HELPERS = re.compile(
            r'\b(get_address|getAddress|get_addr|getAddr|_get_addr|_with_buf|getStr)\b'
        )
        in_callback = False
        # Check current line first (handles same-line: get_address { ... Native.fn() ... })
        if _CALLBACK_HELPERS.search(line):
            in_callback = True
        if not in_callback:
            for back in range(max(0, i - 5), i):
                bline = lines[back].strip()
                if _CALLBACK_HELPERS.search(bline):
                    in_callback = True
                    break
                # Also detect single-line lambda/closure pattern (Ruby block args, Rust closures)
                # NOTE: do NOT match bare `{` on its own line — that's a function body, not a closure
                if re.search(r'(fn\s*\(|\{[\s]*\||\{\s*buf\,|=>\s*\})', bline):
                    in_callback = True
                    break
        if in_callback:
            continue

        # Also detect `let rc: Int32` (Swift) or `let rc:` (no = on the decl line,
        # assignment happens later inside a block).  Use a wider window (±20) for
        # Swift's deeply-nested closure patterns.
        for back in range(max(0, i - 20), i):
            if re.search(r'\blet\s+rc\s*:', lines[back]):
                in_callback = True
                break
        if in_callback:
            continue

        call_fn_m = re.search(r'(secp256k1_|ufsecp_)(\w+)', line)
        findings.append(Finding(
            file=path, line=i + 1, severity="MEDIUM",
            category="FFI_RETVAL_IGNORED",
            message=f"FFI call return value silently discarded — "
                    f"`{call_fn_m.group(0) if call_fn_m else '?'}` returns a non-zero "
                    f"error code on failure but the result is not checked",
            snippet=stripped[:120],
            fix_hint="Assign the return value to a variable and check it; "
                     "raise/throw an exception or return an error on non-zero result.",
        ))
    return findings


# ── MISSING_NULL_AFTER_ALLOC: malloc/calloc/new result used without NULL check ─
# Matches: `type *var = malloc(...)` / `calloc(...)` / `new T`
_ALLOC_ASSIGN = re.compile(
    r'^\s*(?:\w[\w:\s]*?)\s*\*+\s*(\w+)\s*=\s*'
    r'(?:(?:\(\s*\w[\w:\s]*\s*\*+\s*\))?\s*(?:malloc|calloc|realloc|aligned_alloc)\s*\('
    r'|std::malloc\s*\('
    r'|new\s+\w)'
)
_NULL_CHECK_AFTER_ALLOC = re.compile(
    r'if\s*\(\s*!|if\s*\(\s*\w+\s*==\s*(?:NULL|nullptr|0)\s*\)|assert\s*\(\s*\w|ASSERT\s*\(\s*\w'
)


def check_missing_null_after_alloc(path: str, lines: List[str]) -> List[Finding]:
    """malloc/calloc result assigned to a pointer without a NULL check in the
    next few lines — crash on allocation failure, potential security escalation
    if caller controls allocation size.

    NOTE: `new` without `std::nothrow` throws on failure; this checker only
    flags C-style allocators where NULL is the error signal.
    """
    # C/C++ only
    _C_EXT = {'.c', '.cpp', '.cxx', '.cc', '.h', '.hpp', '.cu', '.cuh', '.mm'}
    if not any(path.endswith(ext) for ext in _C_EXT):
        return []
    findings: List[Finding] = []
    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        m = _ALLOC_ASSIGN.match(line)
        if not m:
            continue
        # Skip new-expressions — those throw on failure (nothrow is handled later if needed)
        if 'new ' in line and 'malloc' not in line and 'calloc' not in line and 'realloc' not in line:
            continue
        var = m.group(1)
        if not var or var in ('nullptr', 'NULL', '0'):
            continue
        # Scan next 4 lines for a null check on this var
        has_check = False
        for j in range(i + 1, min(len(lines), i + 5)):
            nxt = _strip_comments(lines[j])
            if re.search(r'\b' + re.escape(var) + r'\b', nxt):
                if _NULL_CHECK_AFTER_ALLOC.search(nxt):
                    has_check = True
                    break
                # Variable is used directly without check
                break
        if not has_check:
            findings.append(Finding(
                file=path, line=i + 1, severity="MEDIUM",
                category="MISSING_NULL_AFTER_ALLOC",
                message=f"`{var}` assigned from malloc/calloc/realloc without a NULL check — "
                        "crash or security issue on allocation failure",
                snippet=raw.strip(),
                fix_hint=f"Add `if (!{var}) {{ /* handle OOM */ }}` immediately after the allocation",
            ))
    return findings


# ── UNCHECKED_REALLOC: `ptr = realloc(ptr, ...)` leaks old allocation on OOM ─
_REALLOC_LEAK = re.compile(
    r'\b(\w+)\s*=\s*(?:\(\s*\w[\w:\s]*\s*\*+\s*\))?\s*realloc\s*\(\s*\1\s*,'
)


def check_unchecked_realloc(path: str, lines: List[str]) -> List[Finding]:
    """The pattern `ptr = realloc(ptr, n)` leaks `ptr` when realloc returns NULL
    because the original pointer is overwritten before the NULL check.

    Correct idiom:
        void *tmp = realloc(ptr, n);
        if (!tmp) { free(ptr); return error; }
        ptr = tmp;
    """
    _C_EXT = {'.c', '.cpp', '.cxx', '.cc', '.h', '.hpp', '.cu', '.cuh', '.mm'}
    if not any(path.endswith(ext) for ext in _C_EXT):
        return []
    findings: List[Finding] = []
    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        m = _REALLOC_LEAK.search(line)
        if m:
            var = m.group(1)
            findings.append(Finding(
                file=path, line=i + 1, severity="MEDIUM",
                category="UNCHECKED_REALLOC",
                message=f"`{var} = realloc({var}, …)` leaks `{var}` when realloc returns NULL "
                        "— the original pointer is lost before the error can be handled",
                snippet=raw.strip(),
                fix_hint=(
                    f"Use a temporary: `void *tmp = realloc({var}, n); "
                    f"if (!tmp) {{ free({var}); return err; }} {var} = tmp;`"
                ),
            ))
    return findings


# ── PARSE_RETVAL_IGNORED: unchecked bool return from parse_bytes_strict* ────────
# Catches the class of bugs we fixed in fast_scan_batch (address.cpp 2026-04-26).
# Pattern: a standalone call to Scalar::parse_bytes_strict_nonzero() or
# FieldElement::parse_bytes_strict() where the bool return is not captured,
# tested, or returned.  Ignores lines that also appear in a condition (if/while/
# return/assignment) — those properly consume the return value.
_PARSE_STRICT_CALL = re.compile(
    r'\b(?:Scalar|FieldElement\w*|fast::Scalar)\s*::\s*parse_bytes_strict\w*\s*\('
)
_PARSE_RETURN_CONSUMED = re.compile(
    r'^\s*(?:'
    r'if\s*\('            # if (!Scalar::parse...)
    r'|while\s*\('        # while (!Scalar::parse...)
    r'|(?:bool|auto|const)\s'  # bool ok = ...; auto ok = ...; const bool ok = ...
    r'|return\s'          # return Scalar::parse...
    r'|\[\[maybe_unused\]\]'   # [[maybe_unused]] const bool ...
    r'|&&\s*Scalar\b'     # chained && in condition
    r'|\s*!?\s*Scalar\b'  # (negation in condition context)
    r'|\w+\s*='           # assignment: ok = Scalar::parse...
    r')'
)


def check_parse_retval_ignored(path: str, lines: List[str]) -> List[Finding]:
    """Detect calls to parse_bytes_strict_nonzero / parse_bytes_strict where
    the bool return value is silently discarded.

    Background: these functions return false when the byte input is out of range
    (>= curve order) or zero.  Ignoring the return leaves the output Scalar in an
    indeterminate state and can cause silent key-derivation or signature failures.
    SonarCloud flags this as a MAJOR C Reliability bug (new_reliability_rating < A).
    Two such bugs were found and fixed in cpu/src/address.cpp on 2026-04-26.
    """
    _C_EXT = {'.c', '.cpp', '.cxx', '.cc', '.h', '.hpp'}
    if not any(path.endswith(ext) for ext in _C_EXT):
        return []
    findings: List[Finding] = []
    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        if not _PARSE_STRICT_CALL.search(line):
            continue
        # If the call is inside a consuming context, skip
        if _PARSE_RETURN_CONSUMED.match(line):
            continue
        # Check if the line is part of an if/while by looking for '(' before the call
        # (handles `if (!Scalar::parse...)` where the regex anchor fails)
        stripped = line.strip()
        # Any line whose stripped form starts with '!' is a negated-call in a condition
        if stripped.startswith(('if ', 'if(', 'while ', 'while(', 'return ',
                                '!', 'bool ', 'auto ',
                                'const ', '[[', '&&', '||')):
            continue
        # Also skip if there's an '=' anywhere before the call (assignment context)
        call_pos = _PARSE_STRICT_CALL.search(line)
        if call_pos and '=' in line[:call_pos.start()]:
            continue
        findings.append(Finding(
            file=path, line=i + 1, severity="HIGH",
            category="PARSE_RETVAL_IGNORED",
            message=(
                "Return value of `parse_bytes_strict_nonzero` / `parse_bytes_strict` "
                "silently discarded — output Scalar is indeterminate if input >= N or == 0. "
                "SonarCloud flags this as MAJOR C Reliability bug."
            ),
            snippet=raw.strip(),
            fix_hint=(
                "Add: `if (!Scalar::parse_bytes_strict_nonzero(..., out)) return {};` "
                "or propagate the bool return to the caller."
            ),
        ))
    return findings


def collect_files(src_dirs: List[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for d in src_dirs:
        if d.is_file():
            if d.suffix in _SOURCE_EXTENSIONS and not _is_skippable(d):
                yield d
        elif d.is_dir():
            for p in sorted(d.rglob('*')):
                if p.suffix in _SOURCE_EXTENSIONS and p not in seen and not _is_skippable(p):
                    seen.add(p)
                    yield p


# ──────────────────────────────────────────────────────────────────────────────
# Scan orchestration
# ──────────────────────────────────────────────────────────────────────────────

CHECKERS = [
    check_sptr,
    check_ob1,
    check_sig,
    check_trunc,
    check_overflow,
    check_mset,
    check_null_after_deref,
    check_copypaste,
    check_semi,
    check_missing_break,
    check_retval,
    check_logic,
    check_zeroize,
    check_dblinit,
    check_unreachable,
    # Defensive-coding checkers
    check_dangling_else,
    check_assert_side_effect,
    check_shift_ub,
    check_double_lock,
    check_unsafe_sprintf,
    check_hardcoded_secret,
    check_exception_swallow,
    check_sizeof_mismatch,
    # Crypto-specific checkers
    check_secret_unerased,
    check_ct_violation,
    check_tagged_hash_bypass,
    check_random_in_signing,
    check_binding_no_validation,
    # Crypto bug-pattern checkers (CVE-grounded, added 2026-04-21)
    check_nonce_reuse,
    check_memcmp_secret,
    check_missing_low_s,
    check_scalar_from_rand,
    check_goto_fail,
    check_point_no_validation,
    check_ecdh_output,
    check_no_domain_sep,
    check_der_lax,
    check_timing_branch_on_key,
    check_mac_truncation,
    check_scalar_not_reduced,
    check_printf_secret,
    check_jni_getbytes_null,
    # Cross-language FFI binding checker
    check_ffi_retval_ignored,
    # Memory-safety checkers (added 2026-04-27)
    check_missing_null_after_alloc,
    check_unchecked_realloc,
    # Crypto parse-return checker (added 2026-04-26, grounded in SonarCloud C Reliability)
    check_parse_retval_ignored,
]


def scan_file(path: Path, base_dir: Path) -> List[Finding]:
    try:
        text = path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return []
    lines = text.splitlines()
    rel = str(path.relative_to(base_dir)) if path.is_relative_to(base_dir) else str(path)
    all_findings: List[Finding] = []
    for checker in CHECKERS:
        try:
            all_findings.extend(checker(rel, lines))
        except Exception as exc:
            # Never crash the whole scan because of one noisy pattern
            all_findings.append(Finding(
                file=rel, line=0, severity="LOW", category="INTERNAL",
                message=f"Scanner error in {checker.__name__}: {exc}",
                snippet="", fix_hint="",
            ))
    return all_findings


# ──────────────────────────────────────────────��───────────────────────────────
# Output formatting
# ──────────────────────────────────────────────────────────────────────────────

def _sev_color(sev: str) -> str:
    return {"HIGH": "\033[31m", "MEDIUM": "\033[33m", "LOW": "\033[36m"}.get(sev, "")


def format_human(findings: List[Finding]) -> str:
    RESET = "\033[0m"
    lines: List[str] = []
    by_file: dict[str, List[Finding]] = {}
    for f in findings:
        by_file.setdefault(f.file, []).append(f)

    for file, flist in sorted(by_file.items()):
        lines.append(f"\n{'─'*70}")
        lines.append(f"  {file}  ({len(flist)} finding{'s' if len(flist)!=1 else ''})")
        lines.append(f"{'─'*70}")
        for f in sorted(flist, key=lambda x: x.line):
            col = _sev_color(f.severity)
            lines.append(f"  Line {f.line:4d}  [{col}{f.severity:6s}{RESET}]  [{f.category}]")
            lines.append(f"           {f.message}")
            if f.snippet:
                lines.append(f"           ▶  {f.snippet}")
            lines.append(f"           ✦  {f.fix_hint}")
    return '\n'.join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def _default_src_dirs(base: Path) -> List[Path]:
    """Look for source trees relative to the script's repository root."""
    candidates = [
        base / 'cpu' / 'src',
        base / 'cpu' / 'include',
        base / 'gpu' / 'src',
        base / 'gpu' / 'include',
        base / 'opencl',
        base / 'metal',
        base / 'kernels',
        base / 'src',
        base / 'include',
        base / 'bindings',
    ]
    return [p for p in candidates if p.exists()]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Classic development-bug detector for UltrafastSecp256k1 C++ source",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        '--src-dir', dest='src_dirs', action='append', metavar='DIR',
        help='Source directory or file to scan (repeatable; default: auto-detect)',
    )
    ap.add_argument(
        '--json', action='store_true',
        help='Output findings as JSON',
    )
    ap.add_argument(
        '-o', '--output', metavar='FILE',
        help='Write output to FILE instead of stdout',
    )
    ap.add_argument(
        '--min-severity', choices=['LOW', 'MEDIUM', 'HIGH'], default='LOW',
        help='Only report findings at or above this severity (default: LOW)',
    )
    ap.add_argument(
        '--fail-on-findings', action='store_true',
        help='Exit with code 1 if any findings are reported (useful for CI)',
    )
    ap.add_argument(
        '--category', metavar='CAT',
        help='Only report findings in this category (e.g. SPTR, OB1, …)',
    )
    args = ap.parse_args(argv)

    # Resolve base dir (repo root relative to this script)
    script_dir = Path(__file__).resolve().parent
    base_dir   = script_dir.parent

    if args.src_dirs:
        src_dirs = [Path(d).resolve() for d in args.src_dirs]
    else:
        src_dirs = _default_src_dirs(base_dir)

    if not src_dirs:
        print("dev_bug_scanner: no source directories found — pass --src-dir explicitly", file=sys.stderr)
        return 2

    min_rank = SEVERITY_RANK[args.min_severity]

    all_findings: List[Finding] = []
    file_list = list(collect_files(src_dirs))

    print(f"dev_bug_scanner: scanning {len(file_list)} files in {len(src_dirs)} dir(s) …",
          file=sys.stderr)

    for p in file_list:
        findings = scan_file(p, base_dir)
        all_findings.extend(findings)

    # Filter
    all_findings = [
        f for f in all_findings
        if SEVERITY_RANK.get(f.severity, 0) >= min_rank
    ]
    if args.category:
        all_findings = [f for f in all_findings if f.category == args.category.upper()]

    # Sort: severity desc, file, line
    all_findings.sort(key=lambda f: (-SEVERITY_RANK.get(f.severity, 0), f.file, f.line))

    # Summarise to stderr
    counts: dict[str, int] = {}
    for f in all_findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1
    summary = ', '.join(f"{v} {k}" for k, v in sorted(counts.items(), key=lambda x: -SEVERITY_RANK.get(x[0], 0)))
    print(f"dev_bug_scanner: {len(all_findings)} findings  ({summary or 'none'})", file=sys.stderr)

    # Format output
    if args.json:
        output = json.dumps([asdict(f) for f in all_findings], indent=2)
    else:
        if all_findings:
            output = format_human(all_findings)
            # Append summary line
            output += f"\n\n{'═'*70}\n  TOTAL: {len(all_findings)} finding(s)  [{summary}]\n"
        else:
            output = "\ndev_bug_scanner: PASS — no findings.\n"

    if args.output:
        Path(args.output).write_text(output, encoding='utf-8')
        print(f"dev_bug_scanner: results written to {args.output}", file=sys.stderr)
    else:
        print(output)

    if args.fail_on_findings and all_findings:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
