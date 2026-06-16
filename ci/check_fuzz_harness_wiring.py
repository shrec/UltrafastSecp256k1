#!/usr/bin/env python3
"""
check_fuzz_harness_wiring.py — the fuzz-liveness gate (analogue of check_exploit_wiring).

WHY (the bug class this prevents):
Coverage-guided libFuzzer harnesses only protect the code they actually RUN. A harness
that exists on disk but is wired into NO build is silently dead — it never compiles in a
campaign, never runs, never finds a bug, and no test ever notices. Symmetrically, a build
file that references a deleted harness breaks the whole fuzz build on the next campaign.
ClusterFuzzLite runs on its own schedule, so this rot is invisible until a campaign fails
days later. This gate makes the harness<->build correspondence structural and fast.

Two managed harness sets, each with a single source-of-truth build:
  * src/cpu/fuzz/fuzz_*.cpp   -> .clusterfuzzlite/build.sh   (ClusterFuzzLite / OSS-Fuzz;
                                  referenced as explicit `src/cpu/fuzz/fuzz_X.cpp` paths)
  * audit/fuzz_*.cpp          -> audit/CMakeLists.txt `set(_FUZZ_HARNESSES ...)` list
                                  (a CMake foreach turns each stem into `${_harness}.cpp`)

Enforcement:
  * BLOCK if a libFuzzer harness (defines LLVMFuzzerTestOneInput) is wired into NO build.
  * BLOCK if a build reference resolves to no harness on disk (dangling -> broken build).
  * REPORT seed-corpus coverage (absence is advisory, not a block — libFuzzer still runs).

A self-test (ci/test_check_fuzz_harness_wiring.py) proves this gate blocks a dead harness.

Exit 0 = every harness is wired and every reference resolves; exit 1 = a dead/dangling harness.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFL_BUILD = os.path.join(ROOT, ".clusterfuzzlite", "build.sh")
AUDIT_CMAKE = os.path.join(ROOT, "audit", "CMakeLists.txt")
SRC_FUZZ_DIR = os.path.join(ROOT, "src", "cpu", "fuzz")
AUDIT_DIR = os.path.join(ROOT, "audit")
SRC_CORPUS = os.path.join(SRC_FUZZ_DIR, "corpus")


def _is_libfuzzer(path):
    try:
        return "LLVMFuzzerTestOneInput" in open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return False


def _disk_harnesses(directory):
    if not os.path.isdir(directory):
        return []
    return sorted(f for f in os.listdir(directory)
                  if f.startswith("fuzz_") and f.endswith(".cpp")
                  and _is_libfuzzer(os.path.join(directory, f)))


def discover():
    return {"cfl": _disk_harnesses(SRC_FUZZ_DIR), "audit": _disk_harnesses(AUDIT_DIR)}


def cfl_referenced_stems(build_text):
    """Stems compiled by .clusterfuzzlite/build.sh — only explicit src/cpu/fuzz/ paths."""
    out = set()
    for tok in build_text.split():
        tok = tok.strip(',";\\')
        if "src/cpu/fuzz/" in tok and tok.endswith(".cpp"):
            base = tok.split("/")[-1][:-4]
            if base.startswith("fuzz_"):
                out.add(base)
    return out


def cmake_harness_list(cmake_text, varname="_FUZZ_HARNESSES"):
    """Parse the `set(VARNAME <stems...>)` list — the authoritative audit harness set."""
    out = set()
    marker = f"set({varname}"
    i = cmake_text.find(marker)
    if i < 0:
        return out
    j = cmake_text.find(")", i)
    if j < 0:
        return out
    body = cmake_text[i + len(marker):j]
    for tok in body.split():
        tok = tok.strip(',";')
        if tok.startswith("fuzz_"):
            out.add(tok[:-4] if tok.endswith(".cpp") else tok)
    return out


def analyze(harnesses, cfl_refs, audit_refs, disk_cfl, disk_audit):
    """Pure core: return (blocking[list[str]], info[list[str]]).
    harnesses: {'cfl':[*.cpp],'audit':[*.cpp]}; *_refs: set of referenced stems;
    disk_*: set of stems present on disk in each directory."""
    blocking, info = [], []

    # (1) Every discovered harness must be wired into ITS build.
    for h in harnesses.get("cfl", []):
        if h[:-4] not in cfl_refs:
            blocking.append(f"src/cpu/fuzz/{h} is a libFuzzer harness but is NOT built by "
                            f".clusterfuzzlite/build.sh — it would never run in a campaign (dead harness)")
    for h in harnesses.get("audit", []):
        if h[:-4] not in audit_refs:
            blocking.append(f"audit/{h} is a libFuzzer harness but is NOT in the "
                            f"_FUZZ_HARNESSES list in audit/CMakeLists.txt — no target builds it (dead harness)")

    # (2) Every build reference must resolve to a harness on disk (no dangling reference).
    for ref in sorted(cfl_refs):
        if ref not in disk_cfl:
            blocking.append(f".clusterfuzzlite/build.sh builds {ref}.cpp which is not on disk in "
                            f"src/cpu/fuzz/ (dangling reference breaks the fuzz build)")
    for ref in sorted(audit_refs):
        if ref not in disk_audit:
            blocking.append(f"_FUZZ_HARNESSES lists {ref} but audit/{ref}.cpp is not on disk "
                            f"(dangling reference breaks the fuzz build)")

    return blocking, info


def run() -> int:
    harnesses = discover()
    cfl_text = open(CFL_BUILD, encoding="utf-8").read() if os.path.exists(CFL_BUILD) else ""
    cmake_text = open(AUDIT_CMAKE, encoding="utf-8", errors="replace").read() if os.path.exists(AUDIT_CMAKE) else ""

    cfl_refs = cfl_referenced_stems(cfl_text)
    audit_refs = cmake_harness_list(cmake_text)
    disk_cfl = {f[:-4] for f in os.listdir(SRC_FUZZ_DIR)} if os.path.isdir(SRC_FUZZ_DIR) else set()
    disk_audit = {f[:-4] for f in os.listdir(AUDIT_DIR)
                  if f.endswith(".cpp")} if os.path.isdir(AUDIT_DIR) else set()

    blocking, _ = analyze(harnesses, cfl_refs, audit_refs, disk_cfl, disk_audit)
    corpus_have = set(os.listdir(SRC_CORPUS)) if os.path.isdir(SRC_CORPUS) else set()

    print("=" * 70)
    print("  Fuzz-Harness Wiring / Liveness Gate")
    print("=" * 70)
    total = len(harnesses["cfl"]) + len(harnesses["audit"])
    print(f"  libFuzzer harnesses: {total}  |  ClusterFuzzLite: {len(harnesses['cfl'])}  "
          f"|  CTest(audit): {len(harnesses['audit'])}")
    for h in harnesses["cfl"]:
        seed = "seeds" if h[:-4] in corpus_have else "no seed corpus (advisory)"
        print(f"    \033[92m[CFL ]\033[0m {h:28} {seed}")
    for h in harnesses["audit"]:
        print(f"    \033[92m[TEST]\033[0m {h:28} _FUZZ_HARNESSES")

    if blocking:
        print()
        for b in blocking:
            print(f"  \033[91mFAIL\033[0m  {b}")
        print(f"\n\033[91m\033[1m  FUZZ-HARNESS WIRING: {len(blocking)} blocking issue(s)\033[0m")
        print("  A harness wired into no build is dead coverage; a dangling reference breaks the build.")
        return 1

    print()
    print(f"  OK: all {total} libFuzzer harnesses are wired into a build; no dangling references.")
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    sys.exit(main())
