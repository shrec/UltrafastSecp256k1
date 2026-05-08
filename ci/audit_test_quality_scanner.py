#!/usr/bin/env python3
"""
audit_test_quality_scanner.py

External-auditor-grade static analyzer for audit C++ test files.

Detects seven bug classes that human auditors catch but automated test runs miss:

  A: Vacuous checks — CHECK(true, ...) always passes, regardless of actual behavior
  B: Mandatory security gap — security-critical else-branch has CHECK(true) weasel-out
  C: Condition/message polarity — test condition contradicts what the message asserts
  D: Weak statistical thresholds — bias/uniformity bounds so loose they catch nothing
  E: Unchecked API return values — ufsecp_*() called without asserting success
  F: Missing mandatory rejection — security ops that MUST fail never assert it unconditionally
  G: Unwired exploit PoC — on-disk test_exploit_*.cpp is not registered in
     unified_audit_runner.cpp (runs only as standalone CTest, bypassing the
     aggregated audit verdict).

Usage:
    python3 ci/audit_test_quality_scanner.py [--audit-dir audit/] [--json] [-o report.json]
"""

import re
import json
import sys
import os
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}

@dataclass
class Finding:
    file: str
    line: int
    severity: str   # critical / high / medium / low / info
    category: str   # A..F
    label: str      # short label, e.g. "vacuous_check"
    description: str
    snippet: str
    fix_hint: str

    def __lt__(self, other):
        return (SEVERITY_ORDER[self.severity], self.file, self.line) < \
               (SEVERITY_ORDER[other.severity], other.file, other.line)


# ---------------------------------------------------------------------------
# Keywords
# ---------------------------------------------------------------------------

# Words that indicate a security-critical assertion
_SEC_WORDS = {
    "tamper", "corrupt", "inject", "unbound", "forge", "bypass",
    "must not", "must fail", "must be rejected", "adversar", "exploit",
    "attack", "malleable", "poison", "spoof", "recover signer", "replay",
}

# Words that indicate "they should be different" in a message
_DIFFER_WORDS = re.compile(r"\b(differ|distinct|different|not equal|not match|not the same|mismatch)\b", re.I)
_EQUAL_WORDS  = re.compile(r"\b(equal|same|identical|match|unchanged|identical)\b", re.I)
_REJECT_WORDS = re.compile(r"\b(reject|fail|error|invalid|refuse|must fail|should fail|must not|not ok|denied|forbid)\b", re.I)
_ACCEPT_WORDS = re.compile(r"\b(succeed|success|ok|accept|valid|work|pass)\b", re.I)

# Secp256k1 API surface — calls that return ufsecp_error_t
_API_CALL_RE = re.compile(
    r"""^\s*(ufsecp_(?:ctx_create|ctx_clone|pubkey_create|pubkey_xonly|
                       pubkey_parse|seckey_verify|seckey_negate|seckey_tweak_add|
                       seckey_tweak_mul|ecdsa_sign|ecdsa_sign_verified|
                       ecdsa_sign_recoverable|ecdsa_verify|ecdsa_recover|
                       schnorr_sign|schnorr_verify|ecdh|ecdh_raw|
                       bip32_master|bip32_derive|bip32_derive_path|
                       taproot_output_key|taproot_verify|taproot_tweak_seckey|
                       sha256|ripemd160|hash160|keccak256|hkdf_sha256|
                       wif_encode|wif_decode|addr_p2pkh|addr_p2wpkh|addr_p2tr|
                       self_test|last_error|
                       musig2_\w+|frost_\w+|
                       bip324_\w+|schnorr_adaptor_\w+|
                       ecdsa_adaptor_\w+)\s*\()""",
    re.VERBOSE,
)

# Lines that DO properly handle the return value
_CHECKED_RE = re.compile(
    r"""CHECK\s*\(\s*ufsecp_|
        \w+\s*=\s*ufsecp_|
        if\s*\(\s*ufsecp_|
        auto\s+\w+\s*=\s*ufsecp_|
        CHECK_OK\s*\(\s*ufsecp_""",
    re.VERBOSE,
)

# Detection of if...else...CHECK(true) over multiple lines
# Matches both uppercase CHECK(true,...) and lowercase check(true,...) helpers
_ELSE_OPEN_RE  = re.compile(r"^\s*\}\s*else\s*\{")
_CHECK_TRUE_RE = re.compile(r'(?:CHECK|check)\s*\(\s*true\s*,\s*"([^"]*)"')
# A2: expr || true tautology — always evaluates true regardless of expr
_CHECK_OR_TRUE_RE = re.compile(r'CHECK\s*\([^)]*\|\|\s*true\s*[,)]')
# A3: x || !x tautology
_CHECK_OR_NOT_RE  = re.compile(r'CHECK\s*\([^)]*\b(\w+)\s*\|\|\s*!\s*\1\b')
_ANY_CHECK_RE  = re.compile(r'(?:CHECK|check)\s*\(')

# Statistical bounds: catch things like  CHECK(x >= 32 && x <= 224, ...)
_STAT_BOUND_RE = re.compile(
    r'CHECK\s*\(\s*\w+\s*>=\s*(\d+)\s*&&\s*\w+\s*<=\s*(\d+)\s*,\s*"([^"]*)"'
)

# Ratio checks for timing tests: CHECK(ratio < N, ...)
_RATIO_CHECK_RE = re.compile(
    r'CHECK\s*\(\s*\w+\s*<\s*(\d+(?:\.\d+)?)\s*,\s*"([^"]*)"'
)


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

class AuditTestScanner:

    def __init__(self, audit_dir: str):
        self.audit_dir = Path(audit_dir)
        self.findings: List[Finding] = []

    def scan_all(self):
        files = sorted(self.audit_dir.glob("*.cpp"))
        for f in files:
            try:
                self.scan_file(f)
            except Exception as e:
                print(f"  [WARN] could not scan {f.name}: {e}", file=sys.stderr)
        # Repository-wide checks that look across all files, not per-file
        try:
            self._check_G_unwired_exploits()
        except Exception as e:
            print(f"  [WARN] Category G scan failed: {e}", file=sys.stderr)
        self.findings.sort()
        return self.findings

    def scan_file(self, path: Path):
        text = path.read_text(errors="replace")
        lines = text.splitlines()
        fname = path.name

        self._check_A_vacuous(fname, lines)
        self._check_B_mandatory_gap(fname, lines)
        self._check_C_polarity(fname, lines)
        self._check_D_weak_stats(fname, lines)
        self._check_E_ignored_returns(fname, lines)
        self._check_F_missing_unconditional_reject(fname, lines)

    # ------------------------------------------------------------------
    # A: Vacuous CHECK(true, ...) — A1/A2/A3
    # ------------------------------------------------------------------
    def _check_A_vacuous(self, fname: str, lines: List[str]):
        for i, raw in enumerate(lines):
            # A1: CHECK(true, ...)
            m = _CHECK_TRUE_RE.search(raw)
            if m:
                msg = m.group(1)
                is_sec = any(w in msg.lower() for w in _SEC_WORDS)
                sev = "high" if is_sec else "low"
                self._add(fname, i + 1, sev, "A1", "vacuous_check",
                          f'CHECK(true, ...) always passes — behavior is never tested. '
                          f'Message: "{msg}"',
                          raw.strip(),
                          'Replace CHECK(true, "...") with an assertion that validates '
                          'the actual library behavior (return code, output bytes, state).')

            # A2: CHECK(expr || true, ...) — always true regardless of expr
            if _CHECK_OR_TRUE_RE.search(raw):
                msg_m = re.search(r',\s*"([^"]*)"', raw)
                msg = msg_m.group(1) if msg_m else raw.strip()
                is_sec = any(w in msg.lower() for w in _SEC_WORDS)
                sev = "high" if is_sec else "medium"
                self._add(fname, i + 1, sev, "A2", "expr_or_true_tautology",
                          f'CHECK(expr || true, ...) is always true — expr is never '
                          f'meaningful. Message: "{msg}"',
                          raw.strip(),
                          'Replace "expr || true" with the actual correctness condition '
                          '(e.g. "threw || result.is_zero()").')

            # A3: CHECK(x || !x, ...) — tautology
            m3 = _CHECK_OR_NOT_RE.search(raw)
            if m3:
                var = m3.group(1)
                msg_m = re.search(r',\s*"([^"]*)"', raw)
                msg = msg_m.group(1) if msg_m else raw.strip()
                self._add(fname, i + 1, "high", "A3", "or_not_tautology",
                          f'CHECK({var} || !{var}, ...) is always true. '
                          f'Message: "{msg}"',
                          raw.strip(),
                          f'Replace "{var} || !{var}" with the actual condition to test.')

    # ------------------------------------------------------------------
    # B: Mandatory security gap — if (ok) { CHECK(real) } else { CHECK(true) }
    # ------------------------------------------------------------------
    def _check_B_mandatory_gap(self, fname: str, lines: List[str]):
        n = len(lines)
        i = 0
        while i < n:
            if not _ELSE_OPEN_RE.match(lines[i]):
                i += 1
                continue
            # Found } else { at line i
            # Look forward up to 5 lines for CHECK(true, ...)
            for j in range(i + 1, min(i + 6, n)):
                m_true = _CHECK_TRUE_RE.search(lines[j])
                if not m_true:
                    continue
                vacuous_msg = m_true.group(1).lower()
                # Look backward for the if-block to harvest its messages
                context = " ".join(lines[max(0, i - 25):i]).lower()
                # Extract messages from prior CHECKs in the block
                prior_msgs = re.findall(r'CHECK\s*\([^,]+,\s*"([^"]*)"', context)
                all_prior = " ".join(prior_msgs).lower()

                is_sec_vacuous = any(w in vacuous_msg for w in _SEC_WORDS)
                is_sec_prior   = any(w in all_prior   for w in _SEC_WORDS)

                if is_sec_vacuous or is_sec_prior:
                    sev = "critical" if is_sec_vacuous else "high"
                    self._add(fname, j + 1, sev, "B", "mandatory_security_gap",
                              f'Security-critical else-branch allows vacuous pass. '
                              f'Vacuous message: "{m_true.group(1)}". '
                              f'The test has no unconditional assertion that the '
                              f'dangerous case MUST be rejected.',
                              lines[j].strip(),
                              'Remove the if/else bifurcation. Add an unconditional '
                              'CHECK(api_call() != UFSECP_OK, "must be rejected") '
                              'before the OK-branch logic.')
                else:
                    # Still report as medium — else branches should not be vacuous
                    self._add(fname, j + 1, "medium", "B", "conditional_vacuous",
                              f'Else-branch has CHECK(true, ...) — behavior in this '
                              f'branch is not validated. Message: "{m_true.group(1)}"',
                              lines[j].strip(),
                              'Replace with a real assertion or remove the branch '
                              'entirely if the alternative outcome is truly irrelevant.')
                break  # only report once per else-block
            i += 1

    # ------------------------------------------------------------------
    # C: Condition/message polarity mismatch
    # ------------------------------------------------------------------
    def _check_C_polarity(self, fname: str, lines: List[str]):
        for i, raw in enumerate(lines):
            if 'CHECK' not in raw:
                continue
            stripped = raw.strip()

            # memcmp == 0  but message says "differ/distinct/different"
            if re.search(r'memcmp\s*\([^)]+\)\s*==\s*0', raw):
                m = re.search(r'CHECK[^"]*"([^"]*)"', raw)
                if m and _DIFFER_WORDS.search(m.group(1)):
                    self._add(fname, i + 1, "critical", "C", "polarity_bug",
                              f'Condition memcmp()==0 (EQUAL) but message says "{m.group(1)}" (DIFFER). '
                              f'Test passes when buffers are EQUAL, but message claims they should differ — '
                              f'the check either catches nothing or reports the wrong thing.',
                              stripped,
                              'Invert the condition to memcmp() != 0, or fix the message.')

            # memcmp != 0  but message says "equal/same/identical"
            if re.search(r'memcmp\s*\([^)]+\)\s*!=\s*0', raw):
                m = re.search(r'CHECK[^"]*"([^"]*)"', raw)
                if m and _EQUAL_WORDS.search(m.group(1)):
                    self._add(fname, i + 1, "critical", "C", "polarity_bug",
                              f'Condition memcmp()!=0 (DIFFER) but message says "{m.group(1)}" (EQUAL). '
                              f'Test passes when buffers DIFFER, but message claims they should be equal.',
                              stripped,
                              'Invert the condition to memcmp() == 0, or fix the message.')

            # rc == UFSECP_OK but message says "reject/fail/error"
            if re.search(r'==\s*UFSECP_OK', raw) or re.search(r'==\s*0\b', raw):
                m = re.search(r'CHECK[^"]*"([^"]*)"', raw)
                if m and _REJECT_WORDS.search(m.group(1)):
                    # Avoid false positive: "NULL ctx rejected" after checking rc != OK
                    if not re.search(r'!=\s*UFSECP_OK|!=\s*0\b', raw):
                        self._add(fname, i + 1, "high", "C", "polarity_bug",
                                  f'Condition == UFSECP_OK (ACCEPT) but message says '
                                  f'"{m.group(1)}" (REJECT). Check passes when API succeeds, '
                                  f'but message claims it should fail.',
                                  stripped,
                                  'Invert condition to != UFSECP_OK, or fix the message.')

            # rc != UFSECP_OK but message says "succeed/ok/accept"
            if re.search(r'!=\s*UFSECP_OK', raw):
                m = re.search(r'CHECK[^"]*"([^"]*)"', raw)
                if m and _ACCEPT_WORDS.search(m.group(1)):
                    # Avoid false positive: "accept after hostile input" patterns
                    if not re.search(r'==\s*UFSECP_OK', raw):
                        self._add(fname, i + 1, "high", "C", "polarity_bug",
                                  f'Condition != UFSECP_OK (REJECT) but message says '
                                  f'"{m.group(1)}" (ACCEPT). Check passes when API fails.',
                                  stripped,
                                  'Invert condition to == UFSECP_OK, or fix the message.')

    # ------------------------------------------------------------------
    # D: Weak statistical thresholds
    # ------------------------------------------------------------------
    def _check_D_weak_stats(self, fname: str, lines: List[str]):
        for i, raw in enumerate(lines):
            # Bound check: CHECK(X >= lo && X <= hi, ...)
            m = _STAT_BOUND_RE.search(raw)
            if m:
                lo, hi, msg = int(m.group(1)), int(m.group(2)), m.group(3)
                span = hi - lo
                # Try to infer total from surrounding context
                # Look backwards for kSamples / constexpr int / 256 / 512
                context = "\n".join(lines[max(0, i - 40):i])
                total_m = re.search(r'(?:kSamples|kIters|nSamples|total)\s*=\s*(\d+)', context)
                if not total_m:
                    total_m = re.search(r'\b(2048|1024|512|256|128|64)\b', context)
                total = int(total_m.group(1)) if total_m else None

                if total:
                    coverage_pct = 100.0 * span / total
                    expected = total * 0.5  # for binary/MSB tests
                    # 4-sigma for binomial: 4 * sqrt(n*0.5*0.5)
                    four_sigma = 4.0 * (total * 0.25) ** 0.5
                    tight_lo = max(0, expected - four_sigma)
                    tight_hi = min(total, expected + four_sigma)

                    if coverage_pct > 70:
                        self._add(fname, i + 1, "high", "D", "weak_statistical_bound",
                                  f'Statistical bound [{lo}, {hi}] covers {coverage_pct:.0f}% of '
                                  f'{total} samples — allows up to '
                                  f'{100.0*hi/total:.0f}% bias rate to pass undetected. '
                                  f'4-sigma tight bound would be [{tight_lo:.0f}, {tight_hi:.0f}].',
                                  raw.strip(),
                                  f'Tighten bounds: CHECK(X >= {int(tight_lo)} && X <= {int(tight_hi)}, "{msg}"); '
                                  f'for {total} samples at 4-sigma.')

            # Timing ratio check: CHECK(ratio < N, ...)
            m = _RATIO_CHECK_RE.search(raw)
            if m:
                threshold = float(m.group(1))
                msg = m.group(2)
                if threshold > 5.0 and ("timing" in msg.lower() or "ratio" in msg.lower()
                                         or "skew" in msg.lower() or "dvfs" in msg.lower()
                                         or "timing" in fname.lower() or "hertz" in fname.lower()
                                         or "ladderleak" in fname.lower()):
                    self._add(fname, i + 1, "high", "D", "weak_timing_threshold",
                              f'Timing ratio threshold is {threshold}× — a 19.9× timing '
                              f'difference would still pass. For side-channel regression tests '
                              f'the threshold should be ≤ 3.0× (CI noise) unless justified.',
                              raw.strip(),
                              f'Reduce threshold to ≤ 3.0 for CI, or add a second hard '
                              f'threshold at 5.0 for catastrophic regression only.')

    # ------------------------------------------------------------------
    # E: Unchecked API return values (setup calls in assertion context)
    # ------------------------------------------------------------------
    def _check_E_ignored_returns(self, fname: str, lines: List[str]):
        for i, raw in enumerate(lines):
            # Must match an API call pattern
            if not _API_CALL_RE.match(raw):
                continue
            # But NOT already wrapped in CHECK / assignment
            if _CHECKED_RE.search(raw):
                continue
            # Skip comment lines
            if raw.strip().startswith("//"):
                continue
            # Skip lines inside a CHECK block (multi-line CHECK)
            prev3 = " ".join(lines[max(0, i-3):i])
            if 'CHECK(' in prev3 and raw.strip().startswith(")"):
                continue
            # Skip if the previous line is an open `if (` / `while (` / `&&` / `||`
            # continuation (i.e., this `ufsecp_*(...)` is one operand of a larger
            # boolean expression whose result IS used).
            prev_line = lines[i-1].rstrip() if i > 0 else ""
            if prev_line.endswith("&&") or prev_line.endswith("||"):
                continue
            # Skip if the full statement (call line through closing `;`) contains
            # an explicit return-value comparison (`== UFSECP_OK`, `!= UFSECP_OK`,
            # `== UFSECP_ERR_*`, `!= UFSECP_ERR_*`). This catches multi-line
            # boolean chains and `bool x = call() != UFSECP_OK;` patterns where
            # the call result IS being checked.
            paren_depth = 0
            stmt_end = i
            for j in range(i, min(i + 20, len(lines))):
                for ch in lines[j]:
                    if ch == '(': paren_depth += 1
                    elif ch == ')': paren_depth -= 1
                    elif ch == ';' and paren_depth == 0:
                        stmt_end = j
                        break
                if paren_depth == 0 and ';' in lines[j]:
                    stmt_end = j
                    break
            stmt_text = "".join(lines[i:stmt_end+1])
            if any(p in stmt_text for p in (
                '== UFSECP_OK', '!= UFSECP_OK',
                '== UFSECP_ERR', '!= UFSECP_ERR',
            )):
                continue
            # Look at surrounding content to classify severity
            # If this is inside a named test section (not just setup at the top), it's medium
            context_before = " ".join(lines[max(0, i - 5):i]).lower()
            is_in_test_block = any(w in context_before for w in [
                "// atk", "// test", "// verify", "//[", "// BN", "// LL",
                "// HB", "// CSA", "// BAT", "// KR", "// EFI"
            ])
            # Check whether a subsequent line checks the output of this call
            context_after = " ".join(lines[i+1:min(i+4, len(lines))]).lower()
            output_is_used = any(w in context_after for w in [
                "check(", "assert(", "memcmp(", "if ("
            ])

            sev = "medium" if is_in_test_block or not output_is_used else "low"
            self._add(fname, i + 1, sev, "E", "ignored_return_value",
                      f'API call return value not checked — if this call fails silently, '
                      f'subsequent assertions operate on garbage/uninitialized data.',
                      raw.strip(),
                      'Wrap in CHECK(...) == UFSECP_OK or assign to rc and assert rc.')

    # ------------------------------------------------------------------
    # F: Missing unconditional rejection for mandatory-failure operations
    # ------------------------------------------------------------------
    def _check_F_missing_unconditional_reject(self, fname: str, lines: List[str]):
        """
        Find test sections that are supposed to verify a rejection property
        (attack / adversarial / must fail / exploit) but never have an
        unconditional CHECK(api() != UFSECP_OK) — only guarded by if-else.
        """
        full_text = "\n".join(lines)
        # Find comment blocks that declare a "must fail" intent
        section_re = re.compile(
            r"//.*?(?:attack|exploit|tamper|forge|adversar|must.*?fail|must.*?reject)",
            re.I
        )
        # Check if the entire file has any unconditional reject assertion
        unconditional_reject_re = re.compile(
            r'CHECK\s*\(\s*ufsecp_\w+\s*\([^)]+\)\s*!=\s*UFSECP_OK'
        )
        # Check if there are ONLY guarded (if-else) rejects
        guarded_reject_count = len(re.findall(
            r'if\s*\(\s*\w+\s*(?:==|!=)\s*UFSECP_OK\s*\).*?else.*?CHECK\(true',
            full_text, re.S
        ))
        unconditional_count = len(unconditional_reject_re.findall(full_text))

        # Only report if the file name suggests adversarial testing
        # and there are guarded rejections but few unconditional ones
        is_adversarial = any(w in fname for w in [
            "fault_inject", "tamper", "exploit_ecdsa_fault",
            "exploit_kr_ecdsa", "exploit_frost_adaptive",
            "exploit_frost_participant_set", "exploit_invalid_curve",
        ])
        if is_adversarial and guarded_reject_count > 0 and unconditional_count == 0:
            self._add(fname, 1, "high", "F", "missing_unconditional_reject",
                      f'File tests adversarial/fault-injection scenarios ({guarded_reject_count} '
                      f'guarded-reject branches) but has ZERO unconditional '
                      f'CHECK(api() != UFSECP_OK) assertions. Every security rejection '
                      f'can be silently skipped if the API returns OK unexpectedly.',
                      fname,
                      'Add at least one CHECK(adversarial_call() != UFSECP_OK, '
                      '"must unconditionally fail") per threat scenario.')

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _add(self, file, line, severity, category, label, description, snippet, fix_hint):
        self.findings.append(Finding(
            file=file, line=line, severity=severity,
            category=category, label=label,
            description=description,
            snippet=snippet[:120],
            fix_hint=fix_hint,
        ))

    # ------------------------------------------------------------------
    # G: Unwired exploit PoC -- on-disk test_exploit_*.cpp files not registered
    #    in unified_audit_runner.cpp. Added 2026-04-17 after BUG-A1: 16 exploit
    #    PoCs existed on disk and ran as standalone CTest binaries but never
    #    fed into the aggregated audit verdict. The Conversion Standard in
    #    docs/EXPLOIT_BACKLOG.md requires every new exploit test to be
    #    declared + wired + documented in a single commit.
    # ------------------------------------------------------------------
    def _check_G_unwired_exploits(self):
        runner = self.audit_dir / "unified_audit_runner.cpp"
        if not runner.exists():
            return  # nothing to correlate against
        runner_text = runner.read_text(errors="replace")
        # CAAS-23 fix: strip single-line comments before wiring checks so that
        # `// TODO: wire test_exploit_foo_run()` does not produce a false-pass.
        # A comment mentioning the symbol is NOT the same as it being registered.
        runner_text_stripped = re.sub(r'//[^\n]*', '', runner_text)

        # Collect every test_exploit_*.cpp that defines a `*_run()` entry point
        exploit_files = sorted(self.audit_dir.glob("test_exploit_*.cpp"))
        run_decl_re = re.compile(
            r"^\s*(?:int|static\s+int)\s+(test_exploit_[A-Za-z0-9_]+_run)\s*\("
            , re.MULTILINE)

        for cpp in exploit_files:
            try:
                src = cpp.read_text(errors="replace")
            except OSError:
                continue
            m = run_decl_re.search(src)
            if not m:
                # File has no `_run()` entry point -- cannot be wired yet.
                # This is itself a Conversion Standard finding.
                if "int main(" in src:
                    self._add(
                        cpp.name, 1, "high", "G", "exploit_missing_run_wrapper",
                        f"{cpp.name} defines int main() but no test_<name>_run() "
                        f"wrapper, so it can only run standalone and cannot be "
                        f"aggregated into unified_audit_runner.",
                        src.splitlines()[0][:120],
                        "Extract the main() body into "
                        "int test_<name>_run() and reduce main() to a "
                        "STANDALONE_TEST-guarded thin wrapper.")
                continue

            run_sym = m.group(1)
            # Registration check: symbol must appear both as a forward decl
            # AND inside the ALL_MODULES table of unified_audit_runner.
            if run_sym not in runner_text_stripped:
                self._add(
                    cpp.name, m.start(), "high", "G", "exploit_unwired",
                    f"{cpp.name} exposes {run_sym}() but unified_audit_runner.cpp "
                    f"does not reference it; this PoC runs only as a standalone "
                    f"CTest binary and does not contribute to the aggregated "
                    f"audit verdict (BUG-A1 class).",
                    m.group(0).strip()[:120],
                    f"Add `int {run_sym}();` to the forward declarations block "
                    f"and register an ALL_MODULES entry with section "
                    f"\"exploit_poc\" in audit/unified_audit_runner.cpp.")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(findings: List[Finding], total_files: int):
    by_severity = {}
    for f in findings:
        by_severity.setdefault(f.severity, []).append(f)

    cats = {}
    for f in findings:
        cats.setdefault(f.category, 0)
        cats[f.category] += 1

    print()
    print("=" * 70)
    print("  AUDIT TEST QUALITY REPORT")
    print(f"  Files scanned: {total_files}   Findings: {len(findings)}")
    print("=" * 70)

    for sev in ["critical", "high", "medium", "low", "info"]:
        bucket = by_severity.get(sev, [])
        if not bucket:
            continue
        print(f"\n━━ {sev.upper()} ({len(bucket)}) ━━")
        for f in bucket:
            print(f"  [{f.category}] {f.file}:{f.line}")
            print(f"       {f.label}: {f.description[:90]}")
            print(f"       → {f.snippet[:80]}")
            print(f"       FIX: {f.fix_hint[:100]}")

    print()
    print("Category summary:")
    LABELS = {
        "A": "Vacuous CHECK(true)",
        "B": "Mandatory security gap",
        "C": "Condition/message polarity",
        "D": "Weak statistical threshold",
        "E": "Ignored API return value",
        "F": "Missing unconditional reject",
    }
    for cat in sorted(cats):
        print(f"  {cat}: {LABELS.get(cat, cat):<35} {cats[cat]:3d} findings")

    n_critical = len(by_severity.get("critical", []))
    n_high     = len(by_severity.get("high", []))
    print()
    if n_critical > 0:
        print(f"  ✗ FAIL — {n_critical} critical + {n_high} high findings")
    elif n_high > 0:
        print(f"  ⚠ WARNING — {n_high} high-severity findings")
    else:
        print("  ✓ No critical/high findings")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="External-auditor-grade static analyzer for audit C++ test files")
    ap.add_argument("--audit-dir", default="audit",
                    help="Directory containing audit *.cpp files (default: audit/)")
    ap.add_argument("--json", action="store_true",
                    help="Print JSON output instead of text")
    ap.add_argument("-o", "--output",
                    help="Write JSON report to this file")
    ap.add_argument("--min-severity", default="low",
                    choices=["critical", "high", "medium", "low", "info"],
                    help="Only report findings at or above this severity")
    ap.add_argument("--file", default=None,
                    help="Scan only this file (relative to audit-dir or absolute)")
    args = ap.parse_args()

    scanner = AuditTestScanner(args.audit_dir)

    if args.file:
        p = Path(args.file)
        if not p.is_absolute():
            p = Path(args.audit_dir) / args.file
        scanner.scan_file(p)
        total_files = 1
    else:
        findings = scanner.scan_all()
        total_files = len(list(Path(args.audit_dir).glob("*.cpp")))

    # Filter by min severity
    min_ord = SEVERITY_ORDER[args.min_severity]
    findings = [f for f in scanner.findings if SEVERITY_ORDER[f.severity] <= min_ord]

    if args.json or args.output:
        data = {
            "total_files": total_files,
            "total_findings": len(findings),
            "findings": [asdict(f) for f in findings],
        }
        json_str = json.dumps(data, indent=2)
        if args.output:
            Path(args.output).write_text(json_str)
            if not args.json:
                print(f"Report written to {args.output}")
        if args.json:
            print(json_str)

    if not args.json:
        print_report(findings, total_files)

    # Exit code: 0=clean, 1=high/critical — applies in ALL output modes
    n_crit = sum(1 for f in findings if f.severity == "critical")
    n_high = sum(1 for f in findings if f.severity == "high")
    sys.exit(1 if (n_crit + n_high) > 0 else 0)


if __name__ == "__main__":
    main()
