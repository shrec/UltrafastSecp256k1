#!/usr/bin/env python3
"""
check_locked_map_handle_escape.py — the unlock-then-use-raw-pointer lifetime gate.

WHY (the bug class this prevents):
Two confirmed use-after-free bugs in this codebase shared ONE shape: a function takes a
mutex over a container of owning pointers (std::unique_ptr / std::shared_ptr), looks up an
entry, and returns a RAW pointer to the owned object (`it->second.get()`). The lock_guard
is released at function return, so the CALLER dereferences that raw pointer AFTER the lock
is gone — and a concurrent erase frees the object mid-use (PRECOMPUTE-GCONTEXT-UAF in
precompute.cpp; the shim MuSig2 keyagg-cache ka_get in shim_musig.cpp). The fix in both
cases: hold shared_ptr and return a SNAPSHOT (`it->second`), so the caller keeps the object
alive. This gate makes that a standing rule: a NEW accessor regressing to the raw-`.get()`
escape from a lock-guarded container goes red.

Heuristic (low false-positive): flag `return <x>->second.get()` / `return <x>.get()` when
the enclosing function also takes a std::lock_guard / std::scoped_lock / std::unique_lock —
i.e. a raw owning-pointer handle escaping a critical section.

Scope: compat/libsecp256k1_shim/src + src/cpu/src (the concurrent-state surfaces).

Exit 0 = no escaping raw handles; exit 1 = an unlock-then-use-raw escape (UAF class).
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN_DIRS = ["compat/libsecp256k1_shim/src", "src/cpu/src"]

_LOCK = re.compile(r"std::(lock_guard|scoped_lock|unique_lock)\b")
# A raw owning-pointer handle escaping via return: `it->second.get()` or `slot.get()`/
# `it->second.get() : nullptr` (ternary). We target the owning-container shapes.
_ESCAPE = re.compile(r"return\s+[\w:]+(?:->second|\.\w+)?\s*\.get\(\)|->second\.get\(\)\s*:")


def _split_functions(text):
    """Yield (start_line, body) per top-level brace-balanced function-ish block.
    Rough but adequate: walk braces from column-0 signature openings."""
    lines = text.splitlines()
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        # crude function start: a line ending in '{' that looks like a signature
        if line.rstrip().endswith("{") and "(" in line and not line.lstrip().startswith(("if", "for", "while", "switch", "else", "//", "*", "struct", "class", "namespace")):
            depth = line.count("{") - line.count("}")
            start = i
            buf = [line]
            i += 1
            while i < n and depth > 0:
                depth += lines[i].count("{") - lines[i].count("}")
                buf.append(lines[i])
                i += 1
            yield start + 1, "\n".join(buf)
        else:
            i += 1


def scan(text):
    """Pure: return [(lineno_in_block, function_first_line, snippet)] of escapes inside a
    lock-guarded function. lineno is the block's start line in the file."""
    findings = []
    for start, body in _split_functions(text):
        if not _LOCK.search(body):
            continue
        for m in _ESCAPE.finditer(body):
            # locate the offending line for a readable snippet
            upto = body[:m.start()]
            ln = start + upto.count("\n")
            snippet = body.splitlines()[upto.count("\n")].strip()
            findings.append((ln, start, snippet))
    return findings


def run() -> int:
    blocking = []
    scanned = 0
    for rel in SCAN_DIRS:
        d = os.path.join(ROOT, rel)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith((".cpp", ".cc", ".hpp", ".h")):
                continue
            scanned += 1
            text = open(os.path.join(d, f), encoding="utf-8", errors="replace").read()
            for ln, _start, snip in scan(text):
                blocking.append(f"{rel}/{f}:{ln}: raw owning-handle escapes a lock-guarded "
                                f"function (unlock-then-use UAF class): {snip}")

    print("=" * 70)
    print("  Locked-Map Handle-Escape Gate (unlock-then-use-raw-pointer / UAF)")
    print("=" * 70)
    print(f"  scanned {scanned} file(s) in {', '.join(SCAN_DIRS)}")
    if blocking:
        print()
        for b in blocking:
            print(f"  \033[91mFAIL\033[0m  {b}")
        print(f"\n\033[91m\033[1m  HANDLE-ESCAPE: {len(blocking)} unlock-then-use raw handle(s)\033[0m")
        print("  Return a shared_ptr snapshot (it->second), not a raw it->second.get(), so a")
        print("  caller keeps the object alive past the critical section (see g_context / ka_get).")
        return 1
    print()
    print("  OK: no raw owning-pointer handle escapes a lock-guarded container accessor.")
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    sys.exit(main())
