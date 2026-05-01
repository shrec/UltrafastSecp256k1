#!/usr/bin/env python3
"""
mutation_kill_rate.py — Mutation testing: measure audit test suite kill rate
=============================================================================

What this does
--------------
Applies simple source-code mutations to the library core (cpu/src/) one at a
time, rebuilds the library, runs the audit test binary (unified_audit_runner),
and records whether the mutation is "killed" (test fails) or "survives"
(test passes despite the code being wrong).

              kill_rate = killed / total_mutations

A kill rate < 80% on critical subsystems indicates test-suite coverage gaps.

Why this matters for external auditors
---------------------------------------
Static analysis and code review can find structural problems, but only mutation
testing can quantify how well the test suite actually *detects* wrong behavior.
A library with 100% line coverage but 40% mutation kill rate has large detection
blind spots.

Mutation operators applied (intentionally conservative)
--------------------------------------------------------
  AOR  Arithmetic operator replacement  (+→-, *→/, -→+)
  ROR  Relational operator replacement  (>→>=, <→<=, ==→!=, !=→==)
  COR  Constant replacement             (0→1, 1→0, 32→64, 0xFF→0x00)
  LOR  Logical operator replacement     (&&→||, ||→&&)
  BIT  Bitwise operator replacement     (&→|, |→&, ^→~x, >>→<<)

Scope
-----
Only mutates files listed in --targets (defaults to the high-value subset).
Skips test files, benchmark files, and files without the library source marker.

Usage
-----
  # Quick run: 50 random mutations on field.cpp + scalar.cpp
  python3 ci/mutation_kill_rate.py \\
      --build-dir build_opencl \\
      --test-cmd  "ctest --test-dir build_opencl -R unified_audit -j1 -Q" \\
      --targets cpu/src/field.cpp cpu/src/scalar.cpp \\
      --count 50 --seed 42

  # Full run with JSON report
  python3 ci/mutation_kill_rate.py \\
      --build-dir build_opencl \\
      --json -o out/reports/mutation_kill_report.json \\
      --count 200

  # Run as CTest Python test (invoked by CMake audit infra):
  python3 ci/mutation_kill_rate.py --ctest-mode --build-dir $BUILD_DIR

Output
------
  - Per-mutation result table (killed / survived / error)
  - Summary: kill rate %, subsystem breakdown
  - JSON report (--json) for CI archiving
  - Exit code 0 if kill_rate >= threshold (default 75%), else 1
"""

from __future__ import annotations

import argparse
import atexit
import copy
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT   = SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# Mutation rollback safety net
# ---------------------------------------------------------------------------
# Tracks files currently mutated so they can be restored on abnormal exit.
# Uses `git checkout -- <file>` as the ultimate fallback because it is
# immune to Python-level state corruption (OOM, segfault, signal).
_MUTATED_FILES: set[Path] = set()


def _rollback_mutated_files() -> None:
    """Restore any files mutated by this process via git checkout."""
    for src in list(_MUTATED_FILES):
        try:
            subprocess.run(
                ["git", "checkout", "--", str(src)],
                cwd=LIB_ROOT, timeout=10,
                capture_output=True,
            )
        except Exception:
            pass
    _MUTATED_FILES.clear()


def _signal_handler(signum, _frame):
    """Handle SIGTERM/SIGINT by restoring mutated files before exit."""
    _rollback_mutated_files()
    sys.exit(128 + signum)


atexit.register(_rollback_mutated_files)
signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)

# ---------------------------------------------------------------------------
# Default high-value mutation targets (relative to LIB_ROOT)
# ---------------------------------------------------------------------------
DEFAULT_TARGETS = [
    "cpu/src/field.cpp",
    "cpu/src/scalar.cpp",
    "cpu/src/point.cpp",
    "cpu/src/ecdsa.cpp",
    "cpu/src/schnorr.cpp",
    "cpu/src/ct_sign.cpp",
    "cpu/src/recovery.cpp",
]

# Fast standalone suite for --ctest-mode. This avoids running the full
# unified_audit CTest target for every mutation, which is too slow for a
# practical kill-rate lane.
DEFAULT_CTEST_MODE_COMMANDS = [
    ["./cpu/test_comprehensive_standalone"],
    ["./audit/test_fiat_crypto_vectors"],
    ["./audit/test_cross_platform_kat"],
    ["./audit/test_differential_standalone"],
    ["./audit/test_ct_equivalence_standalone"],
    ["./audit/test_ct_verif_formal_standalone"],
    ["./audit/test_parse_strictness_standalone"],
    ["./audit/test_nonce_uniqueness_standalone"],
    ["./audit/test_exploit_ecdsa_edge_cases_standalone"],
    ["./audit/test_exploit_schnorr_edge_cases_standalone"],
]

# ---------------------------------------------------------------------------
# Mutation operator definitions
# ---------------------------------------------------------------------------

@dataclass
class MutationOp:
    name: str
    category: str        # AOR / ROR / COR / LOR / BIT
    pattern: str         # regex pattern (must have exactly one group: the token)
    replacement: str     # replacement string (may reference \1)
    notes: str = ""


# Each op has a pattern + replacement.  We use non-overlapping single-match
# substitution so only ONE token is changed per mutation.

MUTATION_OPS: list[MutationOp] = [
    # -- AOR: Arithmetic operator replacement --------------------------------
    MutationOp("AOR_add_to_sub",  "AOR", r"(?<![+\-])\+(?![+=])",  "-",   "+ → -"),
    MutationOp("AOR_sub_to_add",  "AOR", r"(?<![-+])-(?![=>-])",   "+",   "- → +"),
    MutationOp("AOR_mul_to_div",  "AOR", r"\*(?!=)",               "/",   "* → /"),
    MutationOp("AOR_div_to_mul",  "AOR", r"/(?!=)",                "*",   "/ → *"),

    # -- ROR: Relational operator replacement --------------------------------
    MutationOp("ROR_eq_to_ne",    "ROR", r"==",                    "!=",  "== → !="),
    MutationOp("ROR_ne_to_eq",    "ROR", r"!=",                    "==",  "!= → =="),
    MutationOp("ROR_lt_to_le",    "ROR", r"(?<![<])< (?!=)",       "<=",  "< → <="),
    MutationOp("ROR_gt_to_ge",    "ROR", r"(?<![>])> (?!=)",       ">=",  "> → >="),
    MutationOp("ROR_le_to_lt",    "ROR", r"<=",                    "<",   "<= → <"),
    MutationOp("ROR_ge_to_gt",    "ROR", r">=",                    ">",   ">= → >"),

    # -- COR: Constant replacement -------------------------------------------
    MutationOp("COR_zero_to_one",  "COR", r"\b0\b",               "1",   "0 → 1"),
    MutationOp("COR_one_to_zero",  "COR", r"\b1\b",               "0",   "1 → 0"),
    MutationOp("COR_ff_to_zero",   "COR", r"\b0xFF\b",            "0x00","0xFF→0x00"),
    MutationOp("COR_32_to_31",     "COR", r"\b32\b",              "31",  "32 → 31"),
    MutationOp("COR_64_to_63",     "COR", r"\b64\b",              "63",  "64 → 63"),

    # -- LOR: Logical operator replacement -----------------------------------
    MutationOp("LOR_and_to_or",   "LOR", r"&&",                   "||",  "&& → ||"),
    MutationOp("LOR_or_to_and",   "LOR", r"\|\|",                 "&&",  "|| → &&"),

    # -- BIT: Bitwise operator replacement -----------------------------------
    MutationOp("BIT_and_to_or",   "BIT", r"(?<![&])&(?![&=])",    "|",   "& → |"),
    MutationOp("BIT_or_to_and",   "BIT", r"(?<![|])\|(?![|=])",   "&",   "| → &"),
    MutationOp("BIT_shr_to_shl",  "BIT", r">>(?!=)",              "<<",  ">> → <<"),
    MutationOp("BIT_shl_to_shr",  "BIT", r"<<(?!=)",              ">>",  "<< → >>"),
    MutationOp("BIT_xor_to_and",  "BIT", r"\^(?!=)",              "&",   "^ → &"),
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MutationResult:
    mutation_id: int
    file: str
    line: int
    col: int
    op_name: str
    category: str
    original_token: str
    mutated_token: str
    outcome: str    # "killed" | "survived" | "build_error" | "timeout"
    elapsed_s: float = 0.0


@dataclass
class KillReport:
    timestamp: str = ""
    total: int = 0
    killed: int = 0
    survived: int = 0
    build_errors: int = 0
    timeouts: int = 0
    kill_rate_pct: float = 0.0
    threshold_pct: float = 75.0
    passed: bool = False
    # Guard against false-green verdicts: a mutation kill rate computed over
    # a tiny sample (e.g. 1 testable mutation) is statistically meaningless.
    # Similarly, if most mutations fail to build, the mutator is broken, not
    # the code-under-test. Both conditions now fail the gate explicitly.
    min_testable: int = 20
    max_build_error_ratio: float = 0.5
    testable: int = 0
    pass_reason: str = ""
    baseline_build_ok: bool = True
    baseline_test_ok: bool = True
    baseline_note: str = ""
    targets: list[str] = field(default_factory=list)
    test_commands: list[str] = field(default_factory=list)
    mutations: list[MutationResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Mutation finder
# ---------------------------------------------------------------------------

def find_mutation_sites(src_path: Path, op: MutationOp,
                        skip_comment_lines: bool = True) -> list[tuple[int, int, str]]:
    """Return list of (line_number, col, original_token) where op matches."""
    sites = []
    try:
        text = src_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return sites

    for lineno, line in enumerate(text.splitlines(), 1):
        # Skip comment lines
        stripped = line.lstrip()
        if skip_comment_lines and (stripped.startswith("//") or stripped.startswith("*")):
            continue
        for m in re.finditer(op.pattern, line):
            sites.append((lineno, m.start(), m.group(0)))
    return sites


def apply_mutation(src_path: Path, lineno: int, col: int,
                   op: MutationOp, original_token: str) -> Optional[str]:
    """Return the full modified source text with mutation applied, or None on error."""
    try:
        lines = src_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    except OSError:
        return None
    if lineno < 1 or lineno > len(lines):
        return None
    line = lines[lineno - 1]
    prefix = line[:col]
    suffix = line[col + len(original_token):]
    mutated_line = prefix + op.replacement + suffix
    lines[lineno - 1] = mutated_line
    return "".join(lines)


# ---------------------------------------------------------------------------
# Build + test runner
# ---------------------------------------------------------------------------

def infer_build_targets(test_cmds: list[list[str]]) -> list[str]:
    targets: list[str] = []
    for test_cmd in test_cmds:
        if not test_cmd:
            continue
        exe = test_cmd[0]
        if exe.startswith("./"):
            targets.append(exe[2:])
    return list(dict.fromkeys(targets))


def rebuild_library(build_dir: Path, build_targets: Optional[list[str]] = None,
                    timeout_s: int = 120) -> tuple[bool, str]:
    """Run ninja/make in build_dir. Returns (success, stderr_snippet)."""
    ninja = shutil.which("ninja")
    make  = shutil.which("make")
    if ninja and (build_dir / "build.ninja").exists():
        cmd = [ninja, "-j4"]
    elif make:
        cmd = [make, "-j4"]
    else:
        cmd = None
    if cmd is None:
        return False, "no build tool found (ninja/make)"
    if build_targets:
        cmd.extend(build_targets)
    try:
        result = subprocess.run(
            cmd, cwd=build_dir, capture_output=True, timeout=timeout_s, text=True
        )
        return result.returncode == 0, (result.stderr or "")[-2000:]
    except subprocess.TimeoutExpired:
        return False, "build timeout"
    except Exception as e:
        return False, str(e)


def format_cmd(cmd: list[str]) -> str:
    return " ".join(cmd)


def run_tests(test_cmds: list[list[str]], build_dir: Path, timeout_s: int = 90) -> tuple[bool, str]:
    """Run one or more test commands. Returns (tests_failed, output_snippet)."""
    combined_output: list[str] = []
    try:
        for test_cmd in test_cmds:
            result = subprocess.run(
                test_cmd, cwd=build_dir, capture_output=True, timeout=timeout_s, text=True
            )
            output = (result.stdout + result.stderr).strip()
            if output:
                combined_output.append(f"$ {format_cmd(test_cmd)}\n{output[-1500:]}")
            else:
                combined_output.append(f"$ {format_cmd(test_cmd)}")
            if result.returncode != 0:
                return True, "\n\n".join(combined_output)[-3000:]
        return False, "\n\n".join(combined_output)[-3000:]
    except subprocess.TimeoutExpired as exc:
        cmd = exc.cmd if isinstance(exc.cmd, list) else [str(exc.cmd)]
        return True, f"test timeout: {format_cmd(cmd)}"
    except Exception as e:
        return False, str(e)


def preflight_baseline(
    build_dir: Path,
    test_cmds: list[list[str]],
    build_targets: Optional[list[str]],
    build_timeout: int,
    test_timeout: int,
) -> tuple[bool, str]:
    """Validate that the unmutated tree builds and the selected tests pass."""
    ok, build_log = rebuild_library(build_dir, build_targets=build_targets, timeout_s=build_timeout)
    if not ok:
        return False, f"baseline build failed: {build_log}"
    tests_failed, test_log = run_tests(test_cmds, build_dir, timeout_s=test_timeout)
    if tests_failed:
        return False, f"baseline tests failed: {test_log}"
    return True, test_log


# ---------------------------------------------------------------------------
# Main mutation testing engine
# ---------------------------------------------------------------------------

def run_mutation_testing(
    targets: list[Path],
    build_dir: Path,
    test_cmds: list[list[str]],
    build_targets: Optional[list[str]],
    count: int,
    seed: int,
    threshold: float,
    verbose: bool,
    build_timeout: int,
    test_timeout: int,
) -> KillReport:
    import random
    rng = random.Random(seed)

    report = KillReport(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        threshold_pct=threshold,
        targets=[str(p.relative_to(LIB_ROOT)) for p in targets],
        test_commands=[format_cmd(cmd) for cmd in test_cmds],
    )

    print("Running baseline preflight...")
    baseline_ok, baseline_note = preflight_baseline(
        build_dir=build_dir,
        test_cmds=test_cmds,
        build_targets=build_targets,
        build_timeout=build_timeout,
        test_timeout=test_timeout,
    )
    if not baseline_ok:
        report.baseline_build_ok = not baseline_note.startswith("baseline build failed:")
        report.baseline_test_ok = False if baseline_note.startswith("baseline tests failed:") else report.baseline_test_ok
        if baseline_note.startswith("baseline build failed:"):
            report.baseline_build_ok = False
        report.baseline_note = baseline_note[-3000:]
        report.passed = False
        print(f"  [FAIL] {report.baseline_note}")
        return report

    report.baseline_note = "baseline build + quick suite passed"
    print("  [OK] baseline build + quick suite passed")

    # Enumerate all candidate mutation sites across all targets
    candidates: list[tuple[Path, MutationOp, int, int, str]] = []
    for src in targets:
        if not src.is_file():
            print(f"  [WARN] target not found: {src}", file=sys.stderr)
            continue
        for op in MUTATION_OPS:
            for (lineno, col, orig) in find_mutation_sites(src, op):
                candidates.append((src, op, lineno, col, orig))

    print(f"Found {len(candidates)} candidate mutation sites across {len(targets)} files")

    if not candidates:
        print("  No mutation sites found — check targets and operators.")
        report.passed = True
        return report

    # Sample without replacement (or all if count >= len(candidates))
    selected = rng.sample(candidates, min(count, len(candidates)))

    mut_id = 0
    for (src, op, lineno, col, orig_tok) in selected:
        mut_id += 1
        t0 = time.monotonic()

        # Save original file
        original_text = src.read_text(encoding="utf-8", errors="replace")

        # Apply mutation
        mutated_text = apply_mutation(src, lineno, col, op, orig_tok)
        if mutated_text is None:
            mr = MutationResult(
                mutation_id=mut_id, file=str(src.relative_to(LIB_ROOT)),
                line=lineno, col=col, op_name=op.name, category=op.category,
                original_token=orig_tok, mutated_token=op.replacement,
                outcome="build_error", elapsed_s=0.0,
            )
            report.mutations.append(mr)
            report.build_errors += 1
            report.total += 1
            continue

        src.write_text(mutated_text, encoding="utf-8")
        _MUTATED_FILES.add(src)

        try:
            # Rebuild
            ok, build_log = rebuild_library(build_dir, build_targets=build_targets,
                                            timeout_s=build_timeout)
            if not ok:
                outcome = "build_error"
                report.build_errors += 1
            else:
                # Run tests
                killed, test_log = run_tests(test_cmds, build_dir, timeout_s=test_timeout)
                if killed:
                    outcome = "killed"
                    report.killed += 1
                else:
                    outcome = "survived"
                    report.survived += 1
        finally:
            # Restore original (fast path — in-memory copy)
            src.write_text(original_text, encoding="utf-8")
            _MUTATED_FILES.discard(src)

        elapsed = time.monotonic() - t0
        mr = MutationResult(
            mutation_id=mut_id,
            file=str(src.relative_to(LIB_ROOT)),
            line=lineno, col=col,
            op_name=op.name, category=op.category,
            original_token=orig_tok,
            mutated_token=op.replacement,
            outcome=outcome,
            elapsed_s=round(elapsed, 2),
        )
        report.mutations.append(mr)
        report.total += 1

        status_icon = {"killed":"✓","survived":"✗","build_error":"?","timeout":"T"}.get(outcome,"?")
        if verbose or outcome == "survived":
            print(f"  [{mut_id:4d}/{len(selected)}] {status_icon} {op.name:24s} "
                  f"{src.name}:{lineno}  '{orig_tok}' → '{op.replacement}'  ({elapsed:.1f}s)")
        else:
            print(f"  [{mut_id:4d}/{len(selected)}] {status_icon}  {op.name}  {src.name}:{lineno}",
                  end="\r")

    print()

    # Compute kill rate (build errors excluded from denominator — not meaningful)
    testable = report.killed + report.survived
    report.testable = testable
    report.kill_rate_pct = round(100.0 * report.killed / testable, 1) if testable else 0.0

    # Sample-size / build-health guards (prevent false-green verdicts).
    #
    # 2026-04-17: out/reports/mutation_kill_report.json with total=5, build_errors=4,
    # testable=1, killed=1 reported passed=true at 100% kill rate. Such a
    # report conveys no real assurance. Two guards now enforce minimum
    # sanity before a PASS can be declared:
    #   1) testable >= min_testable (default 20, override via
    #      UFSECP_MUTATION_MIN_SAMPLE)
    #   2) build_errors / total <= max_build_error_ratio (default 0.5,
    #      override via UFSECP_MUTATION_MAX_BUILD_ERROR_RATIO)
    min_testable = int(os.environ.get("UFSECP_MUTATION_MIN_SAMPLE",
                                      str(report.min_testable)))
    max_be_ratio = float(os.environ.get("UFSECP_MUTATION_MAX_BUILD_ERROR_RATIO",
                                        str(report.max_build_error_ratio)))
    report.min_testable = min_testable
    report.max_build_error_ratio = max_be_ratio

    be_ratio = (report.build_errors / report.total) if report.total else 0.0
    if testable < min_testable:
        report.passed = False
        report.pass_reason = (f"insufficient testable mutations "
                              f"({testable} < {min_testable})")
    elif be_ratio > max_be_ratio:
        report.passed = False
        report.pass_reason = (f"build_error ratio too high "
                              f"({be_ratio:.2f} > {max_be_ratio:.2f}) — "
                              "mutator operators likely broken")
    elif report.kill_rate_pct < threshold:
        report.passed = False
        report.pass_reason = (f"kill rate {report.kill_rate_pct:.1f}% < "
                              f"{threshold:.1f}%")
    else:
        report.passed = True
        report.pass_reason = (f"kill rate {report.kill_rate_pct:.1f}% >= "
                              f"{threshold:.1f}% (sample={testable})")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mutation kill-rate tracker for UltrafastSecp256k1 audit suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--build-dir", default="build_opencl",
                   help="CMake binary directory (default: build_opencl)")
    p.add_argument("--test-cmd", default=None,
                   help="Shell command to run tests; default: ctest --test-dir <build> -R unified_audit -j1")
    p.add_argument("--targets", nargs="+", default=None,
                   help="Source files to mutate (relative to repo root)")
    p.add_argument("--count", type=int, default=None,
                   help="Max number of mutations to apply (default: 20 in --ctest-mode, else 100)")
    p.add_argument("--seed", type=int, default=0xDEADBEEF,
                   help="Random seed for mutation selection")
    p.add_argument("--threshold", type=float, default=75.0,
                   help="Minimum kill rate %% to pass (default: 75.0)")
    p.add_argument("--build-timeout", type=int, default=150,
                   help="Seconds allowed for each rebuild (default: 150)")
    p.add_argument("--test-timeout", type=int, default=90,
                   help="Seconds allowed for each test run (default: 90)")
    p.add_argument("--json", action="store_true",
                   help="Write JSON report")
    p.add_argument("-o", "--output", default=None,
                   help="JSON output path (default: out/reports/mutation_kill_report.json)")
    p.add_argument("--ctest-mode", action="store_true",
                   help="Minimal mode: few mutations, no print noise")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    build_dir = Path(args.build_dir)
    if not build_dir.is_absolute():
        build_dir = LIB_ROOT / build_dir
    if not build_dir.is_dir():
        # Try relative to workspace root
        build_dir = LIB_ROOT.parent / args.build_dir
    if not build_dir.is_dir():
        print(f"ERROR: build dir not found: {build_dir}", file=sys.stderr)
        return 1

    if args.test_cmd:
        import shlex
        test_cmds = [shlex.split(args.test_cmd)]
    else:
        if args.ctest_mode:
            test_cmds = [cmd[:] for cmd in DEFAULT_CTEST_MODE_COMMANDS]
        else:
            ctest = shutil.which("ctest") or "ctest"
            test_cmds = [[ctest, "--test-dir", str(build_dir),
                          "-R", "unified_audit", "-j1", "-Q", "--output-on-failure"]]

    raw_targets = args.targets or DEFAULT_TARGETS
    targets = [LIB_ROOT / t for t in raw_targets]
    build_targets = infer_build_targets(test_cmds)

    count    = args.count if args.count is not None else (20 if args.ctest_mode else 100)
    verbose  = args.verbose and not args.ctest_mode

    print("=" * 64)
    print("Mutation Kill-Rate Tracker — UltrafastSecp256k1 Audit")
    print("=" * 64)
    print(f"Build dir : {build_dir}")
    print("Test cmds :")
    for cmd in test_cmds:
        print(f"  - {format_cmd(cmd)}")
    if build_targets:
        print(f"Build tgts: {build_targets}")
    print(f"Targets   : {[t.name for t in targets]}")
    print(f"Count     : {count}")
    print(f"Threshold : {args.threshold}%")
    print()

    report = run_mutation_testing(
        targets=targets,
        build_dir=build_dir,
        test_cmds=test_cmds,
        build_targets=build_targets,
        count=count,
        seed=args.seed,
        threshold=args.threshold,
        verbose=verbose,
        build_timeout=args.build_timeout,
        test_timeout=args.test_timeout,
    )

    # --- Summary -----------------------------------------------------------
    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"  Baseline build    : {'PASS' if report.baseline_build_ok else 'FAIL'}")
    print(f"  Baseline tests    : {'PASS' if report.baseline_test_ok else 'FAIL'}")
    if report.baseline_note:
        print(f"  Baseline note     : {report.baseline_note}")
    print(f"  Total mutations  : {report.total}")
    print(f"  Killed           : {report.killed}")
    print(f"  Survived         : {report.survived}")
    print(f"  Build errors     : {report.build_errors}")
    print(f"  Kill rate        : {report.kill_rate_pct:.1f}%  "
          f"(threshold: {report.threshold_pct:.1f}%)")
    print()

    if report.survived > 0:
        print("SURVIVING MUTATIONS (investigation needed):")
        for mr in report.mutations:
            if mr.outcome == "survived":
                print(f"  {mr.file}:{mr.line}  [{mr.op_name}]  "
                      f"'{mr.original_token}' → '{mr.mutated_token}'")
        print()

    if report.passed:
        print(f"RESULT: PASS  ({report.pass_reason})")
    else:
        print(f"RESULT: FAIL  ({report.pass_reason or 'kill rate below threshold'})")
        print("  Consider adding audit tests or exploit probes to cover surviving mutations.")

    # --- JSON output -------------------------------------------------------
    if args.json or args.output:
        out_path = Path(args.output) if args.output else LIB_ROOT / "out/reports/mutation_kill_report.json"
        out_data = asdict(report)
        out_path.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
        print(f"\nJSON report: {out_path}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
