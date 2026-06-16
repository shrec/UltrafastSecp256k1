#!/usr/bin/env python3
"""
Self-test for ci/check_locked_map_handle_escape.py — DON'T TRUST, VERIFY.

Prove the gate flags the unlock-then-use-raw-pointer shape (the PRECOMPUTE-GCONTEXT-UAF /
shim ka_get class) and does NOT flag the shared_ptr-snapshot fix or a raw .get() outside a
lock. The current repo tree (post-fix) must pass.
"""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location(
        "lme", os.path.join(ROOT, "ci", "check_locked_map_handle_escape.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    m = _load()
    fails = []

    # 0. The real repo (post-fix) must pass.
    if m.run() != 0:
        fails.append("real tree should pass (the ka_get/g_context fixes are in)")

    # 1. The BUG shape MUST be flagged: a lock_guard + return it->second.get().
    bug = (
        "KAEntry* ka_get(const cache* p) {\n"
        "    std::lock_guard<std::mutex> lk(g_mu);\n"
        "    auto it = g_ka.find(tok);\n"
        "    return it != g_ka.end() ? it->second.get() : nullptr;\n"
        "}\n")
    if not m.scan(bug):
        fails.append("a lock-guarded `return it->second.get()` MUST be flagged (UAF class)")

    # 2. The FIX shape must NOT be flagged: shared_ptr snapshot return.
    fix = (
        "std::shared_ptr<KAEntry> ka_get(const cache* p) {\n"
        "    std::lock_guard<std::mutex> lk(g_mu);\n"
        "    auto it = g_ka.find(tok);\n"
        "    return it != g_ka.end() ? it->second : nullptr;\n"
        "}\n")
    if m.scan(fix):
        fails.append("the shared_ptr-snapshot fix (return it->second) must NOT be flagged")

    # 3. A raw .get() with NO lock in the function must NOT be flagged (out of scope).
    nolock = (
        "Foo* get_cached() {\n"
        "    return g_local_unique.get();\n"
        "}\n")
    if m.scan(nolock):
        fails.append("a .get() return with no lock_guard must NOT be flagged")

    print("=" * 60)
    if fails:
        print("  check_locked_map_handle_escape SELF-TEST: FAILED")
        for f in fails:
            print("   - " + f)
        print("=" * 60)
        return 1
    print("  check_locked_map_handle_escape SELF-TEST PASSED")
    print("  (gate flags the lock-guarded raw-handle escape; passes the snapshot fix)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
