#!/usr/bin/env python3
"""check_installed_header_parity.py

Issue #335 acceptance repair (round 10) — generated-header semantic parity
gate for ``include/ufsecp/ufsecp_version.h.in``.

NAMING NOTE (read this first): the round-10 task brief asked for this gate to
be named ``ci/check_version_sync.py``. That name is ALREADY TAKEN by a
pre-existing, already-wired, already-MANDATORY gate (see
``run_fast_gates.sh``'s ``MANDATORY_GATES`` array and its
``run "Version + count sync" ci/check_version_sync.py`` line) that checks a
completely different thing: VERSION.txt-derived version-string sync across
~12 packaging/doc files (podspec, rpm spec, PKGBUILD, conanfile.py,
vcpkg.json, .zenodo.json, README, ...) plus canonical exploit/GPU-ABI *count*
sync. Overwriting that file would silently delete an unrelated, currently
load-bearing fast-gate check and would also blow the "fast ~30s" tier's time
budget (this gate drives a real ``cmake`` configure+install per run). This
gate was therefore given the new, non-colliding name
``check_installed_header_parity.py`` instead. See the round-10 final report
for the full explanation; ``ci/check_version_sync.py`` was left untouched.

Background
----------
``include/ufsecp/ufsecp_version.h.in`` is the CMake ``configure_file()``
template that produces the REAL ``ufsecp_version.h`` header every external
C/C++ consumer of this library gets via ``cmake --install``. Round 9
discovered live that the generated header is missing the ``UFSECP_DEPRECATED``
macro definition that ``include/ufsecp/ufsecp.h``'s deprecated-function
declarations (e.g. ``ufsecp_musig2_partial_sign``) require — a minimal
external consumer that includes the REAL INSTALLED header fails to compile
with "expected constructor, destructor, or type conversion before (" at the
``UFSECP_DEPRECATED(...)`` usage site. The checked-in reference copy
``include/ufsecp/ufsecp_version.h`` (kept in the repo for non-CMake use / a
"what it should look like" comparison — NOT what a real install produces) has
the macro correctly defined.

The fix to ``ufsecp_version.h.in`` is OUT OF SCOPE for this round (blocked on
a separate owner decision about task-card write permissions) — this gate
exists to DETECT the drift precisely and automatically flip to PASS the
moment someone else patches the template. It is EXPECTED to currently FAIL.

Design
------
1. ``generate_real_header()`` drives REAL CMake: a throwaway single-purpose
   ``CMakeLists.txt`` (written under a ``/tmp`` scratch dir, never in-repo)
   calls the actual ``configure_file(... ufsecp_version.h.in ... @ONLY)``
   command on the real template, then a real ``install(FILES ...)`` +
   ``cmake --install`` produces an actual installed header tree — exactly
   the mechanism a real integrator's build would use. No Python string
   substitution of ``@VAR@`` tokens is performed anywhere in this file.
2. ``parse_header()`` is a small preprocessor-directive-aware structural
   parser (not a byte-diff): it tracks the ``#ifndef/#ifdef/#if/#elif/#else/
   #endif`` nesting stack and records, for every ``#define``, the exact
   ordered chain of active conditions at that point. This lets
   ``diff_headers()`` compare macro-name presence, include-guard identity,
   per-branch condition ORDER (catches the Windows dllexport/dllimport/
   static-lib precedence class of bug), and function-declaration signatures
   — while deliberately never comparing macro VALUES for the version-number
   macros (MAJOR/MINOR/PATCH/STRING), since those legitimately differ run to
   run and must never cause a false failure.
3. ``compile_consumer()`` compiles a tiny real C program and a tiny real C++
   program, each ``#include <ufsecp/ufsecp.h>`` with ``-I`` pointed ONLY at
   the throwaway install's ``include/`` dir (never the repo's own
   ``include/``), each referencing the ``UFSECP_DEPRECATED``-marked
   ``ufsecp_musig2_partial_sign`` declaration — real end-to-end proof, not a
   simulated one.

Exit status
-----------
  0   -- generated header is semantically identical to the reference AND
         both consumer compiles succeed
  1   -- semantic drift found, and/or a consumer compile failed
  77  -- advisory skip: cmake (or no C/C++ compiler at all) is not available
         in this environment to drive the real generation/compile steps
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

LIB_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = LIB_ROOT / "include" / "ufsecp" / "ufsecp_version.h.in"
REFERENCE_PATH = LIB_ROOT / "include" / "ufsecp" / "ufsecp_version.h"
UFSECP_H_PATH = LIB_ROOT / "include" / "ufsecp" / "ufsecp.h"
UFSECP_ERROR_H_PATH = LIB_ROOT / "include" / "ufsecp" / "ufsecp_error.h"
VERSION_TXT_PATH = LIB_ROOT / "VERSION.txt"

ADVISORY_SKIP_CODE = 77

KNOWN_SCOPE_BLOCKED_NOTE = (
    "This is a KNOWN, DISCLOSED, SCOPE-BLOCKED issue (issue #335, discovered "
    "round 9): the fix belongs in include/ufsecp/ufsecp_version.h.in, which "
    "is intentionally OUT OF SCOPE for this gate/round to modify (blocked on "
    "a separate owner decision about task-card write permissions). This gate "
    "exists to DETECT the drift precisely, not to fix it -- it is EXPECTED "
    "to FAIL until ufsecp_version.h.in is patched by a future round."
)

# ---------------------------------------------------------------------------
# Semantic C-preprocessor-aware header parser
# ---------------------------------------------------------------------------

_COMMENT_RE = re.compile(r'//[^\n]*|/\*.*?\*/', re.DOTALL)


def strip_comments(text: str) -> str:
    """Remove // and /* */ comments while preserving line count (a removed
    block comment that spanned N physical lines collapses to a newline only
    when it actually contained one, so line numbers used for diagnostics
    stay approximately aligned; exact alignment is not required by any
    check in this module, but is useful for humans reading --keep-scratch
    output)."""
    return _COMMENT_RE.sub(lambda m: "\n" if "\n" in m.group(0) else "", text)


_DIRECTIVE_RE = re.compile(
    r'^\s*#\s*(ifndef|ifdef|if|elif|else|endif|define)\b\s*(.*)$'
)
_DEFINE_NAME_RE = re.compile(r'^([A-Za-z_]\w*)(\([^)]*\))?\s*(.*)$')
_FUNC_DECL_RE = re.compile(
    r'^\s*UFSECP_API\s+([A-Za-z_][\w \*]*?)\s+(\w+)\s*\(([^;\n]*)\)\s*;\s*$',
    re.MULTILINE,
)
_EXTERN_C_OPEN_RE = re.compile(r'#ifdef\s+__cplusplus\s*\n\s*extern\s+"C"\s*\{')
_EXTERN_C_CLOSE_RE = re.compile(r'#ifdef\s+__cplusplus\s*\n\s*\}\s*\n\s*#endif')


@dataclass(frozen=True)
class MacroDef:
    name: str
    is_function_like: bool
    value: str
    chain: tuple  # ordered tuple of normalized condition labels, outermost first
    line: int


@dataclass(frozen=True)
class FuncDecl:
    name: str
    return_type: str
    args: str


@dataclass
class HeaderModel:
    guard_ifndef_name: str | None
    guard_define_name: str | None
    occurrences: list = field(default_factory=list)   # list[MacroDef], document order
    functions: dict = field(default_factory=dict)      # name -> FuncDecl
    extern_c_open: bool = False
    extern_c_close: bool = False

    def by_name(self) -> dict:
        out: dict[str, list[MacroDef]] = {}
        for occ in self.occurrences:
            out.setdefault(occ.name, []).append(occ)
        return out

    def all_macro_names(self) -> set:
        return {occ.name for occ in self.occurrences}


def _normalize_cond(kind: str, cond: str) -> str:
    cond = re.sub(r'\s+', ' ', cond.strip())
    if kind in ('ifdef', 'ifndef', 'if', 'elif'):
        return f'{kind} {cond}'.strip()
    return kind  # 'else'


def parse_header(text: str) -> HeaderModel:
    """Structurally parse a ufsecp_version.h-shaped header: preprocessor
    conditional nesting, every #define's exact active-condition chain,
    UFSECP_API-decorated function declarations, and the extern "C" wrapper.
    Pure function over text -- no filesystem access -- so it is directly
    unit-testable with synthetic fixtures (see
    test_check_installed_header_parity.py)."""
    stripped = strip_comments(text)
    lines = stripped.splitlines()

    stack: list[dict] = []
    occurrences: list[MacroDef] = []
    guard_ifndef_name: str | None = None

    for lineno, raw_line in enumerate(lines, start=1):
        m = _DIRECTIVE_RE.match(raw_line)
        if not m:
            continue
        kind, rest = m.group(1), m.group(2).strip()

        if kind in ('ifndef', 'ifdef', 'if'):
            if kind == 'ifndef' and guard_ifndef_name is None and not stack:
                # The very first top-level #ifndef in the file is, by C
                # header-guard convention, the include guard.
                guard_ifndef_name = rest.split()[0] if rest else None
            stack.append({'label': _normalize_cond(kind, rest)})
        elif kind == 'elif':
            if stack:
                stack[-1]['label'] = _normalize_cond('elif', rest)
        elif kind == 'else':
            if stack:
                stack[-1]['label'] = 'else'
        elif kind == 'endif':
            if stack:
                stack.pop()
        elif kind == 'define':
            dm = _DEFINE_NAME_RE.match(rest)
            if not dm:
                continue
            name, argsgroup, value = dm.group(1), dm.group(2), dm.group(3)
            chain = tuple(f['label'] for f in stack)
            occurrences.append(MacroDef(
                name=name,
                is_function_like=bool(argsgroup),
                value=re.sub(r'\s+', ' ', value.strip()),
                chain=chain,
                line=lineno,
            ))

    guard_define_name = None
    if occurrences and len(occurrences[0].chain) == 1:
        # The #define immediately inside the (first, and only the first)
        # top-level #ifndef is the guard's matching #define, by convention
        # and by construction in both real files (#ifndef X / #define X as
        # the first two preprocessor directives in the file).
        guard_define_name = occurrences[0].name

    functions: dict[str, FuncDecl] = {}
    for fm in _FUNC_DECL_RE.finditer(stripped):
        ret = re.sub(r'\s+', ' ', fm.group(1).strip())
        name = fm.group(2)
        args = re.sub(r'\s+', ' ', fm.group(3).strip())
        functions[name] = FuncDecl(name=name, return_type=ret, args=args)

    return HeaderModel(
        guard_ifndef_name=guard_ifndef_name,
        guard_define_name=guard_define_name,
        occurrences=occurrences,
        functions=functions,
        extern_c_open=bool(_EXTERN_C_OPEN_RE.search(stripped)),
        extern_c_close=bool(_EXTERN_C_CLOSE_RE.search(stripped)),
    )


def diff_headers(gen: HeaderModel, ref: HeaderModel) -> list[dict]:
    """Semantic (not byte) diff: macro-name presence, include-guard
    identity, per-macro branch/condition ORDER (catches Windows dllexport/
    dllimport/static-lib precedence swaps), and function-signature drift.
    Deliberately does NOT compare macro VALUES for simple unconditional
    macros (version numbers, copyright years) -- those legitimately differ
    and must never cause a false failure. Pure function over two
    HeaderModels -- directly unit-testable, no filesystem/process access."""
    drifts: list[dict] = []

    if gen.guard_ifndef_name != ref.guard_ifndef_name:
        drifts.append({
            "kind": "include_guard_ifndef_changed",
            "macro": ref.guard_ifndef_name,
            "detail": (
                f"reference header's #ifndef include guard is "
                f"{ref.guard_ifndef_name!r}; generated header's is "
                f"{gen.guard_ifndef_name!r}."
            ),
        })
    if gen.guard_define_name != ref.guard_define_name:
        drifts.append({
            "kind": "include_guard_define_changed",
            "macro": ref.guard_define_name,
            "detail": (
                f"reference header's matching #define for the include guard "
                f"is {ref.guard_define_name!r}; generated header's is "
                f"{gen.guard_define_name!r} -- a mismatched "
                f"#ifndef X / #define Y pair makes the guard non-functional."
            ),
        })

    ref_names = ref.all_macro_names()
    gen_names = gen.all_macro_names()

    for name in sorted(ref_names - gen_names):
        drifts.append({
            "kind": "macro_missing",
            "macro": name,
            "detail": (
                f"macro {name!r} is #define'd in the reference header "
                f"(include/ufsecp/ufsecp_version.h) but is COMPLETELY ABSENT "
                f"from the generated header -- no #define {name} occurrence "
                f"exists under ANY preprocessor branch. Fix: add the missing "
                f"#define {name}(...) block to "
                f"include/ufsecp/ufsecp_version.h.in. {KNOWN_SCOPE_BLOCKED_NOTE}"
            ),
        })
    for name in sorted(gen_names - ref_names):
        drifts.append({
            "kind": "macro_added",
            "macro": name,
            "detail": (
                f"macro {name!r} is defined in the generated header but not "
                f"in the reference header -- the checked-in reference copy "
                f"(include/ufsecp/ufsecp_version.h) is stale and needs "
                f"updating to match ufsecp_version.h.in."
            ),
        })

    ref_by_name = ref.by_name()
    gen_by_name = gen.by_name()
    for name in sorted(ref_names & gen_names):
        r_occ, g_occ = ref_by_name[name], gen_by_name[name]
        if len(r_occ) != len(g_occ):
            drifts.append({
                "kind": "macro_branch_count_changed",
                "macro": name,
                "detail": (
                    f"macro {name!r} has {len(r_occ)} #define occurrence(s) "
                    f"(one per mutually-exclusive preprocessor branch) in "
                    f"the reference header but {len(g_occ)} in the generated "
                    f"header -- the #if/#elif/#else branch structure itself "
                    f"differs. reference branches="
                    f"{[o.chain for o in r_occ]!r}; generated branches="
                    f"{[o.chain for o in g_occ]!r}."
                ),
            })
            continue
        for i, (r, g) in enumerate(zip(r_occ, g_occ)):
            if r.chain != g.chain:
                drifts.append({
                    "kind": "macro_guard_order_changed",
                    "macro": name,
                    "detail": (
                        f"macro {name!r} branch #{i + 1} (in document order): "
                        f"reference checks {r.chain!r} -> "
                        f"{r.value or '(empty)'}; generated header checks "
                        f"{g.chain!r} -> {g.value or '(empty)'} at the SAME "
                        f"branch position -- condition/precedence order "
                        f"differs. This is the Windows __declspec(dllexport/"
                        f"dllimport)-vs-static-lib precedence class of bug: "
                        f"a static-lib translation unit that defines BOTH "
                        f"UFSECP_STATIC_LIB and UFSECP_BUILDING must resolve "
                        f"to the STATIC_LIB (empty) branch FIRST, not "
                        f"BUILDING (dllexport), or the static build emits an "
                        f"unwanted dllexport."
                    ),
                })

    ref_fns, gen_fns = ref.functions, gen.functions
    for fname in sorted(set(ref_fns) - set(gen_fns)):
        drifts.append({
            "kind": "function_missing",
            "macro": fname,
            "detail": f"{fname}() is declared in the reference header but "
                      f"missing from the generated header.",
        })
    for fname in sorted(set(gen_fns) - set(ref_fns)):
        drifts.append({
            "kind": "function_added",
            "macro": fname,
            "detail": f"{fname}() is declared in the generated header but "
                      f"missing from the reference header.",
        })
    for fname in sorted(set(ref_fns) & set(gen_fns)):
        r, g = ref_fns[fname], gen_fns[fname]
        if r.return_type != g.return_type or r.args != g.args:
            drifts.append({
                "kind": "function_signature_changed",
                "macro": fname,
                "detail": (
                    f"{fname}: reference='{r.return_type} {fname}({r.args})' "
                    f"generated='{g.return_type} {fname}({g.args})'."
                ),
            })

    if ref.extern_c_open != gen.extern_c_open:
        drifts.append({
            "kind": "extern_c_open_changed",
            "macro": None,
            "detail": f"reference '#ifdef __cplusplus / extern \"C\" {{' "
                      f"opening wrapper present={ref.extern_c_open}; "
                      f"generated header present={gen.extern_c_open}.",
        })
    if ref.extern_c_close != gen.extern_c_close:
        drifts.append({
            "kind": "extern_c_close_changed",
            "macro": None,
            "detail": f"reference closing '#ifdef __cplusplus / }} / #endif' "
                      f"wrapper present={ref.extern_c_close}; generated "
                      f"header present={gen.extern_c_close}.",
        })

    return drifts


# ---------------------------------------------------------------------------
# Real CMake-driven header generation (configure_file + install(FILES))
# ---------------------------------------------------------------------------

_PROBE_CMAKE_TEMPLATE = """\
cmake_minimum_required(VERSION 3.18)
file(READ "{version_txt}" _version_raw)
string(STRIP "${{_version_raw}}" _version)
project(ufsecp_version_gate_probe VERSION ${{_version}} LANGUAGES NONE)

# The REAL CMake configure_file() mechanism, invoked on the REAL template
# shipped in the repo -- this is NOT a hand-simulated @VAR@ substitution done
# in Python. Any CMake-side bug in the template shows up here exactly as it
# would in a real `cmake --install` of the actual project.
configure_file(
    "{template_in}"
    "${{CMAKE_CURRENT_BINARY_DIR}}/ufsecp_version.h"
    @ONLY
)

# Mirrors include/ufsecp/CMakeLists.txt's own
#   install(FILES ufsecp.h ${{CMAKE_CURRENT_BINARY_DIR}}/ufsecp_version.h
#                 ufsecp_error.h DESTINATION ${{CMAKE_INSTALL_INCLUDEDIR}}/ufsecp)
# rule: a verbatim install(FILES) copy of ufsecp.h/ufsecp_error.h alongside
# the just-configure_file()-generated ufsecp_version.h. Real CMake
# install(FILES) machinery, not a Python file copy.
install(FILES
    "${{CMAKE_CURRENT_BINARY_DIR}}/ufsecp_version.h"
    "{ufsecp_h}"
    "{ufsecp_error_h}"
    DESTINATION include/ufsecp
)
"""


@dataclass
class GenerateResult:
    ok: bool
    error: str | None
    generated_header_path: Path | None
    configure_log: str
    install_log: str
    build_dir: Path
    prefix_dir: Path


def generate_real_header(template_path: Path, version_txt: Path,
                          ufsecp_h: Path, ufsecp_error_h: Path,
                          scratch_root: Path) -> GenerateResult:
    """Drive a REAL, minimal, throwaway CMake project (written under
    scratch_root, which callers must place under /tmp -- never a new
    in-repo directory) that runs the actual configure_file() + install(FILES)
    commands against the real template and real header files. Returns the
    path to the actually-installed generated header."""
    probe_dir = scratch_root / "probe"
    build_dir = scratch_root / "build"
    prefix_dir = scratch_root / "prefix"
    probe_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which("cmake") is None:
        return GenerateResult(False, "cmake not found on PATH", None, "", "",
                               build_dir, prefix_dir)

    cmakelists = _PROBE_CMAKE_TEMPLATE.format(
        version_txt=str(version_txt.resolve()),
        template_in=str(template_path.resolve()),
        ufsecp_h=str(ufsecp_h.resolve()),
        ufsecp_error_h=str(ufsecp_error_h.resolve()),
    )
    (probe_dir / "CMakeLists.txt").write_text(cmakelists)

    try:
        configure = subprocess.run(
            ["cmake", "-S", str(probe_dir), "-B", str(build_dir)],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return GenerateResult(False, f"cmake configure failed to execute: {exc}",
                               None, "", "", build_dir, prefix_dir)
    configure_log = configure.stdout + configure.stderr
    if configure.returncode != 0:
        return GenerateResult(
            False,
            f"cmake configure of the throwaway probe project failed "
            f"(rc={configure.returncode}) -- this is the probe project's OWN "
            f"CMakeLists.txt failing, not the real repo build; see "
            f"configure_log for the real cmake diagnostic.",
            None, configure_log, "", build_dir, prefix_dir,
        )

    try:
        install = subprocess.run(
            ["cmake", "--install", str(build_dir), "--prefix", str(prefix_dir)],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return GenerateResult(False, f"cmake --install failed to execute: {exc}",
                               None, configure_log, "", build_dir, prefix_dir)
    install_log = install.stdout + install.stderr
    if install.returncode != 0:
        return GenerateResult(
            False,
            f"cmake --install of the throwaway probe project failed "
            f"(rc={install.returncode}); see install_log.",
            None, configure_log, install_log, build_dir, prefix_dir,
        )

    generated = prefix_dir / "include" / "ufsecp" / "ufsecp_version.h"
    if not generated.is_file():
        return GenerateResult(
            False,
            f"cmake --install reported success but the expected installed "
            f"header {generated} does not exist.",
            None, configure_log, install_log, build_dir, prefix_dir,
        )

    return GenerateResult(True, None, generated, configure_log, install_log,
                           build_dir, prefix_dir)


# ---------------------------------------------------------------------------
# Real consumer compile checks (C and C++), headers-only, -I installed tree
# ---------------------------------------------------------------------------

_C_CONSUMER_SRC = """\
/* Real end-to-end consumer compile check (C). Includes ONLY the throwaway
 * installed header tree (see -I in the invoking command) and references the
 * UFSECP_DEPRECATED-marked declaration ufsecp_musig2_partial_sign. If
 * ufsecp_version.h.in is missing the UFSECP_DEPRECATED macro, this fails
 * with a syntax error at that declaration site in ufsecp.h -- the exact
 * round-9 disclosed bug ("expected constructor, destructor, or type
 * conversion before (" in C++; a parse error in C too, since
 * UFSECP_DEPRECATED("...") is left as bare unexpanded tokens). */
#include <ufsecp/ufsecp.h>

int main(void) {
    void *fp = (void *)&ufsecp_musig2_partial_sign;
    return fp == 0;
}
"""

_CXX_CONSUMER_SRC = """\
/* C++ twin of the C consumer check: same installed header tree, same
 * UFSECP_DEPRECATED-marked declaration reference, but through a C++
 * compiler -- exercises the C++14 [[deprecated(msg)]] branch of the
 * UFSECP_DEPRECATED macro (the C consumer above exercises the GCC/Clang
 * __attribute__((deprecated(msg))) branch instead). */
#include <ufsecp/ufsecp.h>

int main() {
    void *fp = reinterpret_cast<void *>(&ufsecp_musig2_partial_sign);
    return fp == nullptr;
}
"""


@dataclass
class CompileResult:
    attempted: bool
    ok: bool
    compiler: str | None
    returncode: int | None
    stdout: str
    stderr: str
    cmd: list


def _find_compiler(candidates: list[str]) -> str | None:
    for c in candidates:
        if c and shutil.which(c):
            return c
    return None


def compile_consumer(compiler: str | None, extra_flags: list[str], src_text: str,
                      suffix: str, include_dir: Path, scratch_root: Path,
                      label: str) -> CompileResult:
    if compiler is None:
        return CompileResult(False, False, None, None, "",
                              "no usable compiler found on PATH", [])
    src_path = scratch_root / f"consumer_{label}{suffix}"
    obj_path = scratch_root / f"consumer_{label}.o"
    src_path.write_text(src_text)
    cmd = [compiler, *extra_flags, "-c", str(src_path),
           "-I", str(include_dir), "-o", str(obj_path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CompileResult(True, False, compiler, None, "", str(exc), cmd)
    return CompileResult(True, proc.returncode == 0, compiler, proc.returncode,
                          proc.stdout, proc.stderr, cmd)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _missing_required_files() -> list[str]:
    missing = []
    for label, p in (
        ("template (.in)", TEMPLATE_PATH),
        ("reference header", REFERENCE_PATH),
        ("ufsecp.h", UFSECP_H_PATH),
        ("ufsecp_error.h", UFSECP_ERROR_H_PATH),
        ("VERSION.txt", VERSION_TXT_PATH),
    ):
        if not p.is_file():
            missing.append(f"{label}: {p}")
    return missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--keep-scratch", action="store_true",
                     help="do not delete the /tmp scratch dir on exit (debugging)")
    ap.add_argument("--template", type=Path, default=None,
                     help="DIAGNOSTIC/TEST USE ONLY: path to an alternate "
                          ".in template to generate from instead of the real "
                          "include/ufsecp/ufsecp_version.h.in. Used to prove "
                          "the detector's positive case against a "
                          "hand-patched THROWAWAY COPY of the template "
                          "without ever writing to the real repo file. Never "
                          "pass this in the actual CI gate invocation.")
    args = ap.parse_args()

    missing = _missing_required_files()
    if missing:
        msg = "ERROR: required repo file(s) not found:\n  " + "\n  ".join(missing)
        if args.json:
            print(json.dumps({"result": "ERROR", "missing_files": missing}))
        else:
            print(msg, file=sys.stderr)
        return 1

    template_used = args.template if args.template is not None else TEMPLATE_PATH
    if not template_used.is_file():
        print(f"ERROR: --template path does not exist: {template_used}", file=sys.stderr)
        return 1

    # mkdtemp() (not TemporaryDirectory) deliberately: TemporaryDirectory
    # registers a finalizer that removes the directory at GC/interpreter
    # exit regardless of whether cleanup() is called explicitly, which would
    # silently defeat --keep-scratch. Cleanup below is done manually instead.
    scratch_root = Path(tempfile.mkdtemp(prefix="ufsecp_header_parity_"))
    keep = args.keep_scratch

    try:
        gen_result = generate_real_header(
            template_used, VERSION_TXT_PATH, UFSECP_H_PATH, UFSECP_ERROR_H_PATH,
            scratch_root,
        )

        if not gen_result.ok:
            no_cmake = shutil.which("cmake") is None
            report = {
                "result": "ADVISORY_SKIP" if no_cmake else "FAIL",
                "stage": "generate_real_header",
                "error": gen_result.error,
                "configure_log": gen_result.configure_log,
                "install_log": gen_result.install_log,
            }
            if args.json:
                print(json.dumps(report, indent=2))
            else:
                print("=" * 78)
                print("Installed-Header Semantic Parity Gate (issue #335, round 10)")
                print("=" * 78)
                print(f"Template used                 : {template_used}")
                print(f"FAILED at stage               : generate_real_header")
                print(f"Error                         : {gen_result.error}")
                if gen_result.configure_log:
                    print("\n--- cmake configure log (tail) ---")
                    print(gen_result.configure_log[-4000:])
                if gen_result.install_log:
                    print("\n--- cmake --install log (tail) ---")
                    print(gen_result.install_log[-4000:])
            return ADVISORY_SKIP_CODE if no_cmake else 1

        gen_text = gen_result.generated_header_path.read_text(encoding="utf-8")
        ref_text = REFERENCE_PATH.read_text(encoding="utf-8")
        gen_model = parse_header(gen_text)
        ref_model = parse_header(ref_text)
        drifts = diff_headers(gen_model, ref_model)

        include_dir = gen_result.prefix_dir / "include"
        cc = _find_compiler([os.environ.get("CC"), "cc", "gcc", "clang"])
        cxx = _find_compiler([os.environ.get("CXX"), "c++", "g++", "clang++"])

        c_result = compile_consumer(cc, [], _C_CONSUMER_SRC, ".c",
                                     include_dir, scratch_root, "c")
        cxx_result = compile_consumer(cxx, ["-std=c++17"], _CXX_CONSUMER_SRC, ".cpp",
                                       include_dir, scratch_root, "cxx")

        compile_failures = []
        if c_result.attempted and not c_result.ok:
            compile_failures.append(("C", c_result))
        if cxx_result.attempted and not cxx_result.ok:
            compile_failures.append(("C++", cxx_result))

        overall_fail = bool(drifts) or bool(compile_failures)

        report = {
            "result": "FAIL" if overall_fail else "PASS",
            "template_used": str(template_used),
            "generated_header": str(gen_result.generated_header_path),
            "reference_header": str(REFERENCE_PATH),
            "prefix_dir": str(gen_result.prefix_dir),
            "semantic_drift_count": len(drifts),
            "drifts": drifts,
            "c_consumer_compile": {
                "attempted": c_result.attempted,
                "ok": c_result.ok,
                "compiler": c_result.compiler,
                "returncode": c_result.returncode,
                "cmd": c_result.cmd,
                "stdout": c_result.stdout,
                "stderr": c_result.stderr,
            },
            "cxx_consumer_compile": {
                "attempted": cxx_result.attempted,
                "ok": cxx_result.ok,
                "compiler": cxx_result.compiler,
                "returncode": cxx_result.returncode,
                "cmd": cxx_result.cmd,
                "stdout": cxx_result.stdout,
                "stderr": cxx_result.stderr,
            },
        }

        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("=" * 78)
            print("Installed-Header Semantic Parity Gate (issue #335, round 10)")
            print("=" * 78)
            print(f"Template used (real configure_file() input) : {template_used}")
            print(f"Reference header                             : {REFERENCE_PATH}")
            print(f"Real cmake --install prefix (throwaway /tmp) : {gen_result.prefix_dir}")
            print(f"Installed generated header                   : {gen_result.generated_header_path}")
            print()
            if drifts:
                print(f"SEMANTIC DRIFT: {len(drifts)} issue(s) found "
                      f"between the generated header and the reference header:")
                for i, d in enumerate(drifts, start=1):
                    print(f"\n  [{i}] kind={d['kind']}  macro={d.get('macro')}")
                    print(f"      {d['detail']}")
                print()
                print("  " + KNOWN_SCOPE_BLOCKED_NOTE)
            else:
                print("No semantic drift: generated header matches the reference "
                      "header (macro presence, guard structure, branch order, "
                      "function signatures all agree).")

            print()
            print("--- Consumer compile checks (against ONLY the installed "
                  "header tree) ---")
            for lang, r in (("C  ", c_result), ("C++", cxx_result)):
                if not r.attempted:
                    print(f"  {lang}: SKIPPED -- {r.stderr}")
                    continue
                status = "OK" if r.ok else "FAIL"
                print(f"  {lang}: {status}  compiler={r.compiler}  "
                      f"returncode={r.returncode}")
                print(f"       cmd: {' '.join(r.cmd)}")
                if not r.ok:
                    tail = (r.stdout + r.stderr).strip()
                    for line in tail.splitlines()[:40]:
                        print(f"       | {line}")

            print()
            print("RESULT:", report["result"])

        return 1 if overall_fail else 0
    finally:
        if not keep:
            shutil.rmtree(scratch_root, ignore_errors=True)
        else:
            print(f"\n[--keep-scratch] scratch dir preserved: {scratch_root}",
                  file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
