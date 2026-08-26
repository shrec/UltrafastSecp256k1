#!/usr/bin/env python3
"""Regression tests for check_source_graph_quality.py and its ci_local.sh wiring."""

import importlib.util
import os
import tempfile
import time
from pathlib import Path

SCRIPT = Path(__file__).with_name("check_source_graph_quality.py")
SPEC = importlib.util.spec_from_file_location("check_source_graph_quality", SCRIPT)
SGQ = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SGQ)

CI_DIR = Path(__file__).resolve().parent
LIB_ROOT = CI_DIR.parent


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def _write(path, content="fixture\n", mtime=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def test_freshness_ignores_pycache_and_pyc_pyo():
    """__pycache__/*.pyc/*.pyo churn must not falsely mark the graph DB stale.

    Uses a repo-local scratch directory under out/ (gitignored) rather than the
    system temp dir -- this test's fixtures must never touch external /tmp.
    """
    scratch_root = LIB_ROOT / "out" / "ci_scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(scratch_root)) as tmp:
        root = Path(tmp)
        db_path = root / "tools" / "source_graph_kit" / "source_graph.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_path.write_bytes(b"\x00")

        watched = root / "ci"
        real_src = watched / "check_source_graph_quality.py"

        now = time.time()
        db_mtime = now - 100
        _write(real_src, mtime=db_mtime - 5)
        os.utime(db_path, (db_mtime, db_mtime))

        old_root, old_watched = SGQ.LIB_ROOT, SGQ.WATCHED_DIRS
        SGQ.LIB_ROOT, SGQ.WATCHED_DIRS = root, ["ci"]
        try:
            _write(watched / "__pycache__" / "check_source_graph_quality.cpython-312.pyc",
                   mtime=now)
            _write(watched / "leftover.pyc", mtime=now)
            _write(watched / "leftover.pyo", mtime=now)

            result = SGQ.check_db_freshness(db_path, stale_hours=0.0)
            check(result.status == "PASS",
                  f"__pycache__/.pyc/.pyo churn falsely marked DB stale: {result.detail}")

            # A genuinely newer real source file must still trip staleness --
            # the exclusion must be scoped to pycache artifacts only.
            os.utime(real_src, (now, now))
            result2 = SGQ.check_db_freshness(db_path, stale_hours=0.0)
            check(result2.status == "FAIL",
                  "genuinely newer source file no longer triggers staleness FAIL")
        finally:
            SGQ.LIB_ROOT, SGQ.WATCHED_DIRS = old_root, old_watched


def test_ci_local_refreshes_source_graph_before_quality_gate():
    """ci_local.sh must rebuild the source_graph_kit DB (HEAD binding) before the
    quality gate, and the refresh must be a hard-fail step, not weakened/best-effort."""
    text = (CI_DIR / "ci_local.sh").read_text(encoding="utf-8")
    refresh_idx = text.find("tools/source_graph_kit/source_graph.py build -i")
    quality_idx = text.find('"Source graph quality"')
    check(refresh_idx != -1, "no source_graph.py build -i refresh step found in ci_local.sh")
    check(quality_idx != -1, "Source graph quality gate invocation not found in ci_local.sh")
    check(refresh_idx < quality_idx,
          "source graph refresh must run before the Source graph quality gate")
    refresh_line_start = text.rfind("\n", 0, refresh_idx)
    refresh_line_end = text.find("\n", refresh_idx)
    refresh_line = text[refresh_line_start:refresh_line_end]
    check("|| true" not in refresh_line and "2>/dev/null" not in refresh_line,
          "source graph refresh must not silently swallow failures")


def test_gitignore_ignores_mcp_json_exactly():
    """.mcp.json must be ignored as an exact entry, not left uncovered by a glob."""
    lines = (LIB_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    check(".mcp.json" in lines,
          ".gitignore does not contain an exact '.mcp.json' entry")


if __name__ == "__main__":
    test_freshness_ignores_pycache_and_pyc_pyo()
    test_ci_local_refreshes_source_graph_before_quality_gate()
    test_gitignore_ignores_mcp_json_exactly()
    print("source graph quality regression tests: PASS")
