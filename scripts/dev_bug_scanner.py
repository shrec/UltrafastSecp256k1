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
    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        for m in _OB1_LOOP.finditer(line):
            idx_var = m.group(1)
            bound   = m.group(2)
            # AES/cipher round loops: `round <= 14` or similar — intentional (rounds are 1-based, inclusive)
            # These are numeric bounds 10-20 where the loop var is named round/Round/nr/Nr
            if re.search(r'\b(round|Round|nr|Nr|nrounds)\b', idx_var):
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

def check_sig(path: str, lines: List[str]) -> List[Finding]:
    """Signed int compared with size_t / uint* — may fail when value > INT_MAX."""
    findings: List[Finding] = []
    int_vars: set[str] = set()
    for ln in lines:
        for m in _SIG_DECL.finditer(_strip_comments(ln)):
            int_vars.add(m.group(1))

    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
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
_SUSPICIOUS_SIZES = {3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20,
                     24, 28, 40, 48, 56, 60, 48, 100, 128, 200, 512}

def check_mset(path: str, lines: List[str]) -> List[Finding]:
    """memset/memcpy with hard-coded size that doesn't match common crypto buffer sizes."""
    findings: List[Finding] = []
    # Common crypto sizes: 16, 32, 64, 96, 128, 33, 65
    safe_sizes = {16, 32, 33, 64, 65, 96, 128, 256, 384, 512, 1, 2, 4, 8}
    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
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
    # Track derefs then check for null within small window
    dereffed: dict[str, int] = {}   # name -> line index
    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        # Clear old derefs outside window
        dereffed = {k: v for k, v in dereffed.items() if i - v <= 8}

        # Collect dereferences on this line
        for m in _DEREF_EXPR.finditer(line):
            name = m.group(1)
            if name not in ('this', 'result', 'nullptr', 'NULL'):
                dereffed[name] = i

        for m in _ARROW_EXPR.finditer(line):
            name = m.group(1)
            if name not in ('this',):
                dereffed[name] = i

        # Check for null check that references a previously dereffed variable
        nc = _NULL_CHECK.search(line)
        if nc:
            inner = nc.group(2)
            if inner in dereffed and dereffed[inner] < i:
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
        # Reset on block boundaries and preprocessor conditionals
        if '{' in line or '}' in line or _PREPROC.match(line):
            last_assigned.clear()
            continue
        # Look for simple assignment
        m = _ASSIGN_STMT.match(line)
        if m:
            lval = m.group(1)
            # Skip compound assignments and ==
            if re.match(r'[+\-*/%&|^]=|==', line[m.end()-1:m.end()+1]):
                continue
            # Extract RHS by removing the lval= prefix
            rhs_start = line.find('=') + 1
            rhs = line[rhs_start:]
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
_VOID_COMPAT_FN = {'secp256k1_sha256', 'secp256k1_sha512', 'secp256k1_hash160',
                   'secp256k1_tagged_hash', 'secp256k1_ripemd160'}

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
        # Block boundaries reset tracking
        if '{' in line or '}' in line:
            initialized.clear()
            continue

        m = _SIMPLE_INIT.match(line)
        if m:
            name = m.group(1)
            rhs  = m.group(2)
            # Skip self-referential (x = x + 1 etc.)
            if re.search(r'\b' + re.escape(name) + r'\b', rhs):
                initialized.pop(name, None)
                continue
            if name in initialized:
                prev_line, prev_snip = initialized[name]
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
            if not re.match(r'^\s*(case\b|default\s*:|#|\}.*else\b|else\b|catch\b)', stripped):
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
_NULL_CHECK = re.compile(r'if\s*\(\s*!(?:ctx|out|sig|pk|pubkey|msg|priv|secret|result|entries)')

def check_binding_no_validation(path: str, lines: List[str]) -> List[Finding]:
    """Public ufsecp_ function body does not start with NULL-pointer validation on arguments."""
    if 'impl' not in path.lower() and 'binding' not in path.lower() and 'ufsecp' not in path.lower():
        return []
    findings: List[Finding] = []
    i = 0
    while i < len(lines):
        line = _strip_comments(lines[i])
        m = _UFSECP_FUNC_DEF.match(line.strip())
        if m:
            fn_name = m.group(1)
            # Scan forward into function body (next 15 lines after opening brace)
            has_null_check = False
            brace_found = False
            for j in range(i, min(len(lines), i + 20)):
                body_line = _strip_comments(lines[j])
                if '{' in body_line:
                    brace_found = True
                if brace_found and _NULL_CHECK.search(body_line):
                    has_null_check = True
                    break
                if brace_found and '}' in body_line and body_line.strip() == '}':
                    break  # function ended
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

_SOURCE_EXTENSIONS = {'.cpp', '.cxx', '.cc', '.c', '.cu', '.cuh', '.cl', '.hpp', '.h', '.mm'}

# Directory name components that signal third-party / generated / build code
_SKIP_DIR_PARTS = {
    'node_modules', 'vendor', 'third_party', 'third-party', 'deps',
    'external', '.git', 'build', 'dist', 'out', '__pycache__',
    '_research_repos', 'cmake_install', 'CMakeFiles',
}


def _is_skippable(path: Path) -> bool:
    return any(part in _SKIP_DIR_PARTS for part in path.parts)


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
        if _CATCH_BLOCK.search(line):
            # Check if the catch body is empty (next non-blank line is just })
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
    for i, raw in enumerate(lines):
        line = _strip_comments(raw)
        m = _MEMOP_SIZEOF.search(line)
        if m:
            dst = m.group(1)
            sizeof_arg = m.group(2)
            if dst != sizeof_arg:
                # Only flag if sizeof arg looks like a different variable (not a type)
                if sizeof_arg[0].islower() and sizeof_arg != dst:
                    findings.append(Finding(
                        file=path, line=i + 1, severity="MEDIUM",
                        category="SIZEOF_MISMATCH",
                        message=f"memop destination is `{dst}` but sizeof references `{sizeof_arg}` "
                                "— possible size mismatch if types differ",
                        snippet=raw.strip(),
                        fix_hint=f"Use sizeof({dst}) to ensure the size matches the destination buffer",
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
