#!/usr/bin/env python3
"""Refresh the residual CAAS evidence from its real verifiers, fail-closed.

RR-BAS-01 (docs/RESIDUAL_RISK_REGISTER.md), promoted by the owner on 2026-11-02
("bring everything into one working cycle"). Until then the freshness artifacts that
ci/audit_sla_check.py tracks but no lane regenerated -- audit/ci-evidence, the
determinism golden, the API-contract and assurance-claim files -- and the
INT-SHIM-PARITY row of docs/INTEGRATION_EVIDENCE_STATUS.json were refreshed by hand
(the chore in 121e5325). The nightly lane (.github/workflows/caas-evidence-refresh.yml)
now does exactly that chore and nothing else.

One group per artifact. A file is refreshed ONLY by re-running its real verifier and is
written ONLY when that verifier passes:

  ct_evidence         run the five audit/fuzz suites from --build-dir. If every one exits
                      0 and its own summary shows 0 failed (and 0 crashes where the suite
                      counts them), write five NEW audit/ci-evidence/<suite>_<YYYYMMDD>.txt
                      files, all or none. An existing file is never rewritten or re-dated.
  determinism_golden  ci/check_determinism_gate.py regenerates the report. Accepted only
                      when overall_pass is true and the report equals the committed golden
                      except generated_at and library; otherwise the file is untouched.
  api_contracts       ci/check_api_contracts.py passes -> stamp last_updated + update_note.
  assurance_claims    ci/validate_assurance.py passes -> stamp updated_at.
  shim_parity         ci/check_libsecp_shim_parity.py passes -> stamp last_verified + notes
                      of the INT-SHIM-PARITY row.

A stamp names the date, the HEAD short sha, the verifier command and its verdict wherever
the field has room for text. It is a text edit of exactly those string values, so the
file's formatting survives and the diff is only those fields; the result is re-parsed and
refused unless it differs from the original in the allowed fields alone.

A group is due when its artifact is at least --min-age-days old (default 7), measured by
ci/audit_sla_check.py's own age functions, or with --force. A group that fails writes
nothing and makes the exit status 1; the other groups still refresh.

Modes:
  --plan      print the due groups as JSON; run nothing, write nothing
  --dry-run   run the verifiers of the due groups; write nothing
  (default)   run the verifiers and write what passed

Exit: 0 every due group refreshed (or none due), 1 a group failed, 2 usage error.
Progress goes to stderr; stdout carries one JSON document.

Usage:
  python3 ci/refresh_residual_evidence.py --plan
  python3 ci/refresh_residual_evidence.py --build-dir out/ci-evidence-build --lib out/ci-audit
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent

SCHEMA = "ufsecp.residual_evidence_refresh.v1"
DEFAULT_MIN_AGE_DAYS = 7.0

EVIDENCE_DIR = "audit/ci-evidence"
GOLDEN_FILE = "docs/DETERMINISM_GOLDEN.json"
API_CONTRACTS_FILE = "docs/API_SECURITY_CONTRACTS.json"
ASSURANCE_CLAIMS_FILE = "docs/ASSURANCE_CLAIMS.json"
INTEGRATION_FILE = "docs/INTEGRATION_EVIDENCE_STATUS.json"
SHIM_ROW_ID = "INT-SHIM-PARITY"

# The only fields of a regenerated determinism report that may differ from the golden.
GOLDEN_VOLATILE = ("generated_at", "library")

# Upper bounds in seconds; a verifier that exceeds its bound has not passed.
VERIFIER_TIMEOUT = 900
GATE_TIMEOUT = 600


@dataclass(frozen=True)
class Suite:
    stem: str                    # audit/ci-evidence/<stem>_<YYYYMMDD>.txt
    binary: str                  # CMake target == executable name (audit/CMakeLists.txt)
    summary: "re.Pattern[str]"   # named groups: passed, failed, and crashes where counted
    timeout: int                 # seconds


SUITES = (
    Suite("adversarial_protocol", "test_adversarial_protocol_standalone",
          re.compile(r"Adversarial Summary:\s*(?P<passed>\d+) passed,\s*(?P<failed>\d+) failed"), 600),
    Suite("ecies_regression", "test_ecies_regression_standalone",
          re.compile(r"ECIES regression:\s*(?P<passed>\d+) passed,\s*(?P<failed>\d+) failed"), 300),
    Suite("regression_ct_ops", "test_regression_ct_ops_standalone",
          re.compile(r"\bpass=(?P<passed>\d+)\s+fail=(?P<failed>\d+)"), 300),
    Suite("fuzz_parsers", "test_fuzz_parsers",
          re.compile(r"TOTAL:\s*(?P<passed>\d+) passed,\s*(?P<failed>\d+) failed"), 900),
    Suite("fuzz_address_bip32_ffi", "test_fuzz_address_bip32_ffi",
          re.compile(r"PASSED:\s*(?P<passed>\d+)\s+FAILED:\s*(?P<failed>\d+)"
                     r"\s+CRASHES:\s*(?P<crashes>\d+)"), 900),
)


@dataclass
class Ctx:
    root: Path                    # library root the groups read and write
    sla: object                   # audit_sla_check, bound to root
    today: str                    # UTC date, YYYY-MM-DD
    head: Optional[str]           # HEAD short sha, None when git cannot say
    dry_run: bool = False
    build_dir: Optional[Path] = None
    lib: Optional[str] = None     # shared library (file or dir) for the determinism gate
    python: str = sys.executable


@dataclass(frozen=True)
class Group:
    name: str
    artifact: str
    needs_test_binaries: bool     # needs the audit/fuzz executables built in --build-dir
    needs_shared_lib: bool        # needs libufsecp built
    age: Callable[[Ctx], Optional[float]]
    run: Callable[[Ctx], dict]


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def result(name: str, status: str, detail: str, writes=(), verifier: Optional[str] = None) -> dict:
    out = {"name": name, "status": status, "detail": detail, "writes": list(writes)}
    if verifier:
        out["verifier"] = verifier
    return out


def tail(text: str, lines: int = 3, width: int = 300) -> str:
    kept = [ln.strip() for ln in text.strip().splitlines() if ln.strip()][-lines:]
    return " | ".join(kept)[-width:]


def run_capture(cmd, cwd, timeout, merge_stderr=False):
    """Run CMD; return (returncode, stdout, stderr). The returncode is None when the
    process could not be started or did not finish within TIMEOUT seconds."""
    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return None, "", f"timed out after {timeout}s"
    except OSError as exc:
        return None, "", f"cannot run {cmd[0]}: {exc}"
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def head_short(root: Path) -> Optional[str]:
    rc, out, _ = run_capture(["git", "rev-parse", "--short=8", "HEAD"], root, 15)
    sha = out.strip()
    return sha if rc == 0 and re.fullmatch(r"[0-9a-f]{7,40}", sha) else None


def load_sla(root: Path):
    """audit_sla_check as a private instance whose git calls run in ROOT, so the ages used
    here are the ages audit_sla_check.py itself reports for that tree."""
    path = SCRIPT_DIR / "audit_sla_check.py"
    spec = importlib.util.spec_from_file_location("audit_sla_check_refresh", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.LIB_ROOT = root
    return module


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def write_new_files(files: dict) -> None:
    """Create every file or none. Mode 'x' refuses an existing path, so an existing
    snapshot is never rewritten (re-dated); on any error the files created so far are
    removed again."""
    created = []
    try:
        for path, data in files.items():
            with open(path, "xb") as handle:
                created.append(path)
                handle.write(data)
    except BaseException:
        for path in created:
            try:
                path.unlink()
            except OSError:
                pass
        raise


def replace_file(path: Path, data: bytes) -> None:
    """Replace PATH's bytes through a sibling temp file and a rename, so no reader ever
    sees a half-written JSON file."""
    tmp = path.with_name(path.name + ".refresh-tmp")
    try:
        with open(tmp, "wb") as handle:
            handle.write(data)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# JSON stamping: edit string values in place, keep every other byte
# ---------------------------------------------------------------------------

class StampError(ValueError):
    """A stamp cannot be made safely (missing, ambiguous or non-string target)."""


class _Node:
    """One JSON value and where it sits in the text. kind: obj | arr | str | lit.
    items: obj -> [(key, node)], arr -> [node], otherwise None."""
    __slots__ = ("kind", "start", "end", "items")

    def __init__(self, kind, start, end, items=None):
        self.kind, self.start, self.end, self.items = kind, start, end, items


_LITERAL = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null")
_SPACE = " \t\r\n"


def parse_spans(text: str) -> _Node:
    """Parse TEXT as JSON into a tree that records the span of every value."""
    n = len(text)

    def skip(i):
        while i < n and text[i] in _SPACE:
            i += 1
        return i

    def string_end(i):
        j = i + 1
        while j < n:
            if text[j] == "\\":
                j += 2
            elif text[j] == '"':
                return j + 1
            else:
                j += 1
        raise StampError("unterminated string")

    def value(i):
        i = skip(i)
        if i >= n:
            raise StampError("unexpected end of document")
        c = text[i]
        if c == "{":
            return members(i)
        if c == "[":
            return elements(i)
        if c == '"':
            return _Node("str", i, string_end(i))
        m = _LITERAL.match(text, i)
        if not m:
            raise StampError(f"unexpected character {c!r} at offset {i}")
        return _Node("lit", i, m.end())

    def members(start):
        items = []
        i = skip(start + 1)
        if i < n and text[i] == "}":
            return _Node("obj", start, i + 1, items)
        while True:
            i = skip(i)
            if i >= n or text[i] != '"':
                raise StampError(f"object key expected at offset {i}")
            key_end = string_end(i)
            key = json.loads(text[i:key_end])
            i = skip(key_end)
            if i >= n or text[i] != ":":
                raise StampError(f"':' expected at offset {i}")
            node = value(i + 1)
            items.append((key, node))
            i = skip(node.end)
            if i < n and text[i] == ",":
                i += 1
            elif i < n and text[i] == "}":
                return _Node("obj", start, i + 1, items)
            else:
                raise StampError(f"',' or '}}' expected at offset {i}")

    def elements(start):
        items = []
        i = skip(start + 1)
        if i < n and text[i] == "]":
            return _Node("arr", start, i + 1, items)
        while True:
            node = value(i)
            items.append(node)
            i = skip(node.end)
            if i < n and text[i] == ",":
                i += 1
            elif i < n and text[i] == "]":
                return _Node("arr", start, i + 1, items)
            else:
                raise StampError(f"',' or ']' expected at offset {i}")

    root = value(0)
    if skip(root.end) != n:
        raise StampError("trailing data after the JSON document")
    return root


def _decode(text: str, node: _Node):
    return json.loads(text[node.start:node.end])


def _locate(text: str, root: _Node, path: tuple) -> _Node:
    """Resolve PATH to a node. A step is a key (str) or a (key, value) selector that picks
    the one object of an array whose member KEY is the string VALUE."""
    node = root
    for step in path:
        hits = []
        if isinstance(step, str):
            if node.kind == "obj":
                hits = [member for key, member in node.items if key == step]
        elif node.kind == "arr":
            sel_key, wanted = step
            for element in node.items:
                if element.kind != "obj":
                    continue
                for key, member in element.items:
                    if key == sel_key and member.kind == "str" and _decode(text, member) == wanted:
                        hits.append(element)
        if len(hits) != 1:
            raise StampError(f"{step!r} on {path!r}: expected exactly 1 match, found {len(hits)}")
        node = hits[0]
    return node


def stamp_strings(text: str, edits: list) -> str:
    """Replace the string values at the given paths and change nothing else. EDITS is a
    list of (path, new_string)."""
    root = parse_spans(text)
    spans = []
    for path, new_value in edits:
        node = _locate(text, root, path)
        if node.kind != "str":
            raise StampError(f"{path!r} is not a string value")
        spans.append((node.start, node.end, json.dumps(new_value)))
    spans.sort(reverse=True)
    limit = len(text) + 1
    for start, end, literal in spans:
        if end > limit:
            raise StampError("overlapping edits")
        text = text[:start] + literal + text[end:]
        limit = start
    return text


def concrete_path(doc, path: tuple) -> tuple:
    """PATH with every (key, value) selector replaced by the array index it selects."""
    out, cur = [], doc
    for step in path:
        if isinstance(step, str):
            if not isinstance(cur, dict) or step not in cur:
                raise StampError(f"{step!r} not found on {path!r}")
            cur = cur[step]
            out.append(step)
        else:
            sel_key, wanted = step
            if not isinstance(cur, list):
                raise StampError(f"{step!r} on {path!r}: not an array")
            hits = [i for i, el in enumerate(cur) if isinstance(el, dict) and el.get(sel_key) == wanted]
            if len(hits) != 1:
                raise StampError(f"{step!r} on {path!r}: expected exactly 1 match, found {len(hits)}")
            cur = cur[hits[0]]
            out.append(hits[0])
    return tuple(out)


def value_at(doc, concrete: tuple):
    for step in concrete:
        doc = doc[step]
    return doc


def changed_fields(old, new, path: tuple = ()) -> set:
    """Paths of the leaf values that differ between two parsed JSON documents."""
    if isinstance(old, dict) and isinstance(new, dict):
        out = set()
        for key in set(old) | set(new):
            if key not in old or key not in new:
                out.add(path + (key,))
            else:
                out |= changed_fields(old[key], new[key], path + (key,))
        return out
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return {path}
        out = set()
        for i, (a, b) in enumerate(zip(old, new)):
            out |= changed_fields(a, b, path + (i,))
        return out
    return set() if type(old) is type(new) and old == new else {path}


def scope_violations(old_doc, new_doc, allowed) -> list:
    """The changed fields that are not in ALLOWED, as 'a/b/0/c' strings."""
    outside = changed_fields(old_doc, new_doc) - set(allowed)
    return sorted("/".join(str(step) for step in path) for path in outside)


def stamp_json(ctx: Ctx, name: str, rel: str, edits: list, detail: str,
               verifier: Optional[str] = None, expect=None) -> dict:
    """Stamp the string fields in EDITS into REL, or write nothing."""
    path = ctx.root / rel
    old_text = path.read_bytes().decode("utf-8")
    old_doc = json.loads(old_text)
    allowed = {concrete_path(old_doc, p) for p, _ in edits}
    new_text = stamp_strings(old_text, edits)
    new_doc = json.loads(new_text)
    outside = scope_violations(old_doc, new_doc, allowed)
    if outside:
        return result(name, "failed", f"the stamp would change fields outside the allowed set "
                      f"({', '.join(outside)}); nothing written", verifier=verifier)
    for p, value in edits:
        if value_at(new_doc, concrete_path(new_doc, p)) != value:
            return result(name, "failed", f"{p!r} does not read back as the stamped value; "
                          "nothing written", verifier=verifier)
    if expect is not None and new_doc != expect:
        return result(name, "failed", "the stamped file does not equal the verified report; "
                      "nothing written", verifier=verifier)
    if ctx.dry_run:
        return result(name, "verified", detail + "; dry-run, nothing written", verifier=verifier)
    replace_file(path, new_text.encode("utf-8"))
    return result(name, "refreshed", detail, [rel], verifier)


# ---------------------------------------------------------------------------
# Group: ct_evidence
# ---------------------------------------------------------------------------

def judge_suite(suite: Suite, returncode: Optional[int], output: str) -> Optional[str]:
    """None when the suite passed, otherwise why it did not."""
    if returncode is None:
        return "did not finish (could not start, or timed out)"
    if returncode != 0:
        return f"exited {returncode}"
    matches = list(suite.summary.finditer(output))
    if not matches:
        return "no summary line in its output"
    failed = max(int(m.group("failed")) for m in matches)
    counts = [m.groupdict().get("crashes") for m in matches]
    crashes = max((int(c) for c in counts if c is not None), default=0)
    if failed:
        return f"summary reports {failed} failed"
    if crashes:
        return f"summary reports {crashes} crashes"
    if int(matches[-1].group("passed")) <= 0:
        return "summary reports no passing checks"
    return None


def find_binary(build_dir: Path, name: str) -> Optional[Path]:
    for candidate in (build_dir / "audit" / name, build_dir / name):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    for dirpath, dirnames, filenames in os.walk(build_dir):
        dirnames[:] = [d for d in dirnames if d != "CMakeFiles"]
        if name in filenames:
            candidate = Path(dirpath) / name
            if os.access(candidate, os.X_OK):
                return candidate
    return None


def refresh_ct_evidence(ctx: Ctx) -> dict:
    name = "ct_evidence"
    if ctx.build_dir is None:
        return result(name, "failed", "--build-dir is required: configure with -DBUILD_TESTING=ON "
                      "-DSECP256K1_BUILD_FUZZ_TESTS=ON and build the five suites")
    stamp = ctx.today.replace("-", "")
    out_dir = ctx.root / EVIDENCE_DIR
    targets = {s.stem: out_dir / f"{s.stem}_{stamp}.txt" for s in SUITES}
    present = sorted(p.name for p in targets.values() if p.exists())
    if present:
        return result(name, "skipped", "a snapshot dated today already exists and is never "
                      "rewritten: " + ", ".join(present))
    binaries = {s.stem: find_binary(ctx.build_dir, s.binary) for s in SUITES}
    missing = sorted(s.binary for s in SUITES if binaries[s.stem] is None)
    if missing:
        return result(name, "failed", f"not built under --build-dir {ctx.build_dir}: "
                      + ", ".join(missing))
    # One suite at a time, each on an idle machine with its output uninterleaved. Every
    # suite runs even after a failure so one report names all of them.
    outputs, problems = {}, []
    for suite in SUITES:
        binary = binaries[suite.stem]
        log(f"[{name}] running {suite.binary}")
        rc, out, err = run_capture([str(binary)], binary.parent, suite.timeout, merge_stderr=True)
        outputs[suite.stem] = out
        reason = judge_suite(suite, rc, out) if rc is not None else err
        if rc is None or reason:
            problems.append(f"{suite.stem}: {reason} [{tail(out)}]")
    verifier = ", ".join(s.binary for s in SUITES)
    if problems:
        return result(name, "failed", "; ".join(problems) + " -- nothing written", verifier=verifier)
    files = {targets[s.stem]: outputs[s.stem].encode("utf-8") for s in SUITES}
    rels = [f"{EVIDENCE_DIR}/{p.name}" for p in files]
    if ctx.dry_run:
        return result(name, "verified", f"{len(SUITES)} suites passed; dry-run, nothing written",
                      verifier=verifier)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_new_files(files)
    return result(name, "refreshed", f"{len(SUITES)} suites passed; wrote {len(rels)} dated snapshots",
                  rels, verifier)


# ---------------------------------------------------------------------------
# Group: determinism_golden
# ---------------------------------------------------------------------------

def golden_problems(old, new) -> list:
    """Why a regenerated determinism report cannot replace the committed golden."""
    if not isinstance(old, dict) or not isinstance(new, dict):
        return ["the golden or the report is not a JSON object"]
    problems = []
    if new.get("overall_pass") is not True:
        problems.append("report overall_pass is not true")
    if new.get("issues") != []:
        problems.append(f"report lists issues: {new.get('issues')!r}")
    for key in GOLDEN_VOLATILE:
        if not isinstance(old.get(key), str) or not isinstance(new.get(key), str):
            problems.append(f"{key} is not a string in both the golden and the report")
    stable_old = {k: v for k, v in old.items() if k not in GOLDEN_VOLATILE}
    stable_new = {k: v for k, v in new.items() if k not in GOLDEN_VOLATILE}
    drift = sorted("/".join(str(step) for step in path)
                   for path in changed_fields(stable_old, stable_new))
    if drift:
        problems.append("content differs from the committed golden at: " + ", ".join(drift[:8])
                        + (" ..." if len(drift) > 8 else ""))
    return problems


def refresh_determinism_golden(ctx: Ctx) -> dict:
    name = "determinism_golden"
    gate = ctx.root / "ci" / "check_determinism_gate.py"
    verifier = "python3 ci/check_determinism_gate.py"
    old_text = (ctx.root / GOLDEN_FILE).read_bytes().decode("utf-8")
    old = json.loads(old_text)
    with tempfile.TemporaryDirectory(prefix="ufsecp_determinism_") as tmp:
        report_path = Path(tmp) / "determinism.json"
        cmd = [ctx.python, str(gate), "-o", str(report_path)]
        if ctx.lib:
            cmd += ["--lib", ctx.lib]
        rc, out, err = run_capture(cmd, ctx.root, GATE_TIMEOUT)
        if rc != 0:
            return result(name, "failed", f"the determinism gate "
                          f"{'exited ' + str(rc) if rc is not None else err}: {tail(out + err)}",
                          verifier=verifier)
        try:
            new = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return result(name, "failed", f"the determinism gate wrote no readable report: {exc}",
                          verifier=verifier)
    problems = golden_problems(old, new)
    if problems:
        return result(name, "failed", "; ".join(problems) + "; the golden is untouched",
                      verifier=verifier)
    edits = [((key,), new[key]) for key in GOLDEN_VOLATILE]
    return stamp_json(ctx, name, GOLDEN_FILE, edits,
                      "overall_pass true; identical to the committed golden except "
                      "generated_at and library", verifier=verifier, expect=new)


# ---------------------------------------------------------------------------
# Groups: stamps (api_contracts, assurance_claims, shim_parity)
# ---------------------------------------------------------------------------

def api_note(date: str, head: str, entries: int) -> str:
    return (f"{date}: CAAS residual-evidence refresh (automated: ci/refresh_residual_evidence.py "
            f"in the caas-evidence-refresh lane, HEAD {head}). Re-verified with python3 "
            f"ci/check_api_contracts.py --json: PASS, {entries} entries, no issues (entry "
            "structure, evidence tests and docs resolve, and no sensitive API/security source "
            "is changed without a contract update). The stamp records the re-verification, not "
            "a content edit, and does not replace the human review of an API change.")


def shim_note(date: str, head: str, checks: int) -> str:
    return (f"{checks}/{checks} shim parity checks PASS at HEAD {head}. Re-verified {date} with "
            "python3 ci/check_libsecp_shim_parity.py --json (automated: "
            "ci/refresh_residual_evidence.py).")


def refresh_api_contracts(ctx: Ctx) -> dict:
    name, verifier = "api_contracts", "python3 ci/check_api_contracts.py --json"
    if ctx.head is None:
        return result(name, "failed", "cannot read HEAD, so the stamp cannot name it", verifier=verifier)
    rc, out, err = run_capture([ctx.python, str(ctx.root / "ci" / "check_api_contracts.py"), "--json"],
                               ctx.root, VERIFIER_TIMEOUT)
    if rc != 0:
        return result(name, "failed", f"the verifier {'exited ' + str(rc) if rc is not None else err}: "
                      f"{tail(out + err)}; nothing stamped", verifier=verifier)
    try:
        report = json.loads(out)
        issues, entries = report["issues"], report["entries"]
    except (ValueError, KeyError, TypeError) as exc:
        return result(name, "failed", f"unreadable verifier output ({exc}); nothing stamped",
                      verifier=verifier)
    if issues != [] or not isinstance(entries, int) or isinstance(entries, bool):
        return result(name, "failed", f"the verifier reports issues {issues!r}; nothing stamped",
                      verifier=verifier)
    edits = [(("last_updated",), ctx.today), (("update_note",), api_note(ctx.today, ctx.head, entries))]
    return stamp_json(ctx, name, API_CONTRACTS_FILE, edits,
                      f"verifier PASS, {entries} entries; stamped last_updated + update_note",
                      verifier=verifier)


def refresh_assurance_claims(ctx: Ctx) -> dict:
    name, verifier = "assurance_claims", "python3 ci/validate_assurance.py"
    rc, out, err = run_capture([ctx.python, str(ctx.root / "ci" / "validate_assurance.py")],
                               ctx.root, VERIFIER_TIMEOUT)
    if rc != 0:
        return result(name, "failed", f"the verifier {'exited ' + str(rc) if rc is not None else err}: "
                      f"{tail(out + err)}; nothing stamped", verifier=verifier)
    # The ledger has no free-text field, so only the date can be stamped; the commit that
    # carries it and the evidence-chain record name the verifier and the HEAD it ran at.
    return stamp_json(ctx, name, ASSURANCE_CLAIMS_FILE, [(("updated_at",), ctx.today)],
                      "verifier PASS; stamped updated_at", verifier=verifier)


def refresh_shim_parity(ctx: Ctx) -> dict:
    name, verifier = "shim_parity", "python3 ci/check_libsecp_shim_parity.py --json"
    if ctx.head is None:
        return result(name, "failed", "cannot read HEAD, so the stamp cannot name it", verifier=verifier)
    rc, out, err = run_capture([ctx.python, str(ctx.root / "ci" / "check_libsecp_shim_parity.py"), "--json"],
                               ctx.root, VERIFIER_TIMEOUT)
    if rc != 0:
        return result(name, "failed", f"the verifier {'exited ' + str(rc) if rc is not None else err}: "
                      f"{tail(out + err)}; nothing stamped", verifier=verifier)
    try:
        checks = json.loads(out)
        passing = [c for c in checks if isinstance(c, dict) and c.get("status") == "PASS"]
        if not isinstance(checks, list) or not checks or len(passing) != len(checks):
            raise ValueError(f"{len(passing)} of {len(checks) if isinstance(checks, list) else '?'} "
                             "checks report PASS")
    except (ValueError, TypeError) as exc:
        return result(name, "failed", f"the verifier output is not an all-PASS check list ({exc}); "
                      "nothing stamped", verifier=verifier)
    row = ("rows", ("id", SHIM_ROW_ID))
    edits = [(row + ("last_verified",), ctx.today),
             (row + ("notes",), shim_note(ctx.today, ctx.head, len(checks)))]
    return stamp_json(ctx, name, INTEGRATION_FILE, edits,
                      f"verifier PASS, {len(checks)}/{len(checks)} checks; stamped the "
                      f"{SHIM_ROW_ID} row", verifier=verifier)


# ---------------------------------------------------------------------------
# Ages, plan, main
# ---------------------------------------------------------------------------

def _age_ct_evidence(ctx: Ctx) -> Optional[float]:
    return ctx.sla._dir_newest_age_days(ctx.root / EVIDENCE_DIR)


def _age_of(rel: str) -> Callable[[Ctx], Optional[float]]:
    return lambda ctx: ctx.sla._file_age_days(ctx.root / rel)


def _age_shim_parity(ctx: Ctx) -> Optional[float]:
    """The row's own last_verified (what the integration-evidence gate measures); the file's
    git age would reset whenever any other row is edited."""
    doc = json.loads((ctx.root / INTEGRATION_FILE).read_text(encoding="utf-8"))
    row = next((r for r in doc.get("rows", []) if isinstance(r, dict) and r.get("id") == SHIM_ROW_ID), None)
    if row is None:
        return None
    verified = datetime.strptime(str(row.get("last_verified")), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    _as_of, now = ctx.sla._age_reference()
    age = (now - verified.timestamp()) / 86400.0
    return age if age >= 0 else None   # a date in the future is not a measurable age


GROUPS = (
    Group("ct_evidence", EVIDENCE_DIR, True, False, _age_ct_evidence, refresh_ct_evidence),
    Group("determinism_golden", GOLDEN_FILE, False, True, _age_of(GOLDEN_FILE),
          refresh_determinism_golden),
    Group("api_contracts", API_CONTRACTS_FILE, False, False, _age_of(API_CONTRACTS_FILE),
          refresh_api_contracts),
    Group("assurance_claims", ASSURANCE_CLAIMS_FILE, False, False, _age_of(ASSURANCE_CLAIMS_FILE),
          refresh_assurance_claims),
    Group("shim_parity", f"{INTEGRATION_FILE}#{SHIM_ROW_ID}", False, False, _age_shim_parity,
          refresh_shim_parity),
)


def decide(age: Optional[float], min_age: float, force: bool):
    if force:
        return True, "forced (--force)"
    if age is None:
        return True, "no measurable age (artifact or date missing)"
    age = max(age, 0.0)   # a file stamped a few ms "in the future" is 0 days old, not negative
    if age >= min_age:
        return True, f"{age:.1f} days old >= --min-age-days {min_age:g}"
    return False, f"{age:.1f} days old < --min-age-days {min_age:g}"


def build_plan(ctx: Ctx, groups, min_age: float, force: bool) -> dict:
    entries = []
    for group in groups:
        note = ""
        try:
            age = group.age(ctx)
        except Exception as exc:   # an unreadable age is unknown, so due; the verifier decides
            age, note = None, f" (age unreadable: {type(exc).__name__}: {exc})"
        due, reason = decide(age, min_age, force)
        entries.append({
            "name": group.name, "artifact": group.artifact,
            "age_days": None if age is None else round(age, 2),
            "due": due, "reason": reason + note,
            "needs_test_binaries": group.needs_test_binaries,
            "needs_shared_lib": group.needs_shared_lib,
        })
    return {
        "schema": SCHEMA, "date": ctx.today, "head": ctx.head,
        "as_of": ctx.sla._age_reference()[0], "min_age_days": min_age, "force": force,
        "groups": entries,
        "due": [e["name"] for e in entries if e["due"]],
        "needs_test_build": any(e["due"] and e["needs_test_binaries"] for e in entries),
        "needs_shared_lib": any(e["due"] and e["needs_shared_lib"] for e in entries),
    }


def run_group(ctx: Ctx, group: Group) -> dict:
    try:
        return group.run(ctx)
    except Exception as exc:   # fail closed: the writers below this point are all-or-nothing
        return result(group.name, "failed", f"{type(exc).__name__}: {exc}; nothing written")


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true",
                      help="print the due groups as JSON; run nothing, write nothing")
    mode.add_argument("--dry-run", action="store_true",
                      help="run the verifiers of the due groups; write nothing")
    parser.add_argument("--force", action="store_true", help="treat every group as due")
    parser.add_argument("--min-age-days", type=float, default=DEFAULT_MIN_AGE_DAYS,
                        help=f"refresh an artifact at this age or older (default {DEFAULT_MIN_AGE_DAYS:g})")
    parser.add_argument("--build-dir", help="build dir holding the audit/fuzz executables "
                        "(ct_evidence); relative paths are taken from --root")
    parser.add_argument("--lib", help="libufsecp file or directory for the determinism gate; "
                        "relative paths are taken from --root (default: the gate's own search)")
    parser.add_argument("--only", action="append", choices=[g.name for g in GROUPS],
                        help="restrict to this group (repeatable)")
    parser.add_argument("--root", help="library root to operate on (default: the one containing "
                        "this script); the self-test points it at a throwaway tree")
    args = parser.parse_args(argv)
    if args.min_age_days < 0:
        parser.error("--min-age-days must be >= 0")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve() if args.root else LIB_ROOT
    if not (root / "docs").is_dir():
        print(f"error: {root} is not a library root (no docs/ directory)", file=sys.stderr)
        return 2
    build_dir = None
    if args.build_dir:
        build_dir = Path(args.build_dir)
        build_dir = build_dir if build_dir.is_absolute() else root / build_dir
    ctx = Ctx(root=root, sla=load_sla(root), today=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
              head=head_short(root), dry_run=args.dry_run, build_dir=build_dir, lib=args.lib)
    selected = [g for g in GROUPS if not args.only or g.name in args.only]
    plan = build_plan(ctx, selected, args.min_age_days, args.force)
    if args.plan:
        print(json.dumps(plan, indent=2))
        return 0

    results = []
    for group, entry in zip(selected, plan["groups"]):
        if entry["due"]:
            log(f"[{group.name}] due ({entry['reason']})")
            res = run_group(ctx, group)
        else:
            res = result(group.name, "skipped", entry["reason"])
        res["age_days"] = entry["age_days"]
        results.append(res)
        log(f"[{group.name}] {res['status']}: {res['detail']}")
    failed = [r["name"] for r in results if r["status"] == "failed"]
    summary = {k: plan[k] for k in ("schema", "date", "head", "as_of", "min_age_days", "force")}
    summary.update({"mode": "dry-run" if args.dry_run else "write", "groups": results,
                    "failed": failed, "ok": not failed})
    print(json.dumps(summary, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
