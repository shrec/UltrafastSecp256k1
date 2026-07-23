#!/usr/bin/env python3
"""Regression tests for durable-input selection in build_project_graph."""

import importlib.util
import sqlite3
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).with_name("build_project_graph.py")
SPEC = importlib.util.spec_from_file_location("build_project_graph", SCRIPT)
GRAPH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GRAPH)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def test_normalization_and_policy():
    normalize = GRAPH.normalize_project_relpath
    runtime = GRAPH.is_aiworkhub_runtime_path

    check(normalize(r"tools\aiworkhub\logs\.\state.json")
          == "tools/aiworkhub/logs/state.json", "backslash normalization failed")
    check(normalize("tools/aiworkhub/logs/../docs/guide.md")
          == "tools/aiworkhub/docs/guide.md", "dot-segment normalization failed")
    check(normalize("../outside.py") is None, "root escape was accepted")
    check(normalize("docs/../../outside.py") is None, "nested root escape was accepted")
    check(runtime("./tools/aiworkhub/logs/callback_bridge_state.json"),
          "callback state was accepted")
    check(runtime(r"TOOLS\AIWORKHUB\CheckPoints\nested\state.JSON"),
          "case variant bypassed runtime policy")
    check(runtime("tools/aiworkhub/worker/runtime/deep/state.json"),
          "nested runtime directory was accepted")
    check(not runtime("tools/aiworkhub/logs/../docs/guide.md"),
          "normalized durable path was overexcluded")
    check(not runtime("tools/aiworkhub/runtime_notes.py"),
          "broad runtime substring hid durable source")


def create_source_table(connection):
    connection.execute(
        """CREATE TABLE source_files (
        id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT UNIQUE NOT NULL,
        category TEXT, subsystem TEXT, file_type TEXT, lines INTEGER,
        sha256 TEXT, layer TEXT)"""
    )


def write(path, content="fixture\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_populate_source_files():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "project"
        root.mkdir()
        callback = root / "tools/aiworkhub/logs/callback_bridge_state.json"
        write(callback, "{}\n")
        write(root / "tools/aiworkhub/worker/runtime/deep/state.json", "{}\n")
        write(root / "TOOLS/AIWORKHUB/CHECKPOINTS/state.json", "{}\n")
        write(root / "tools/aiworkhub/docs/guide.md")
        write(root / "tools/aiworkhub/runtime_notes.py")
        write(root / "ci/ordinary.py")

        outside = Path(tmp) / "outside.py"
        write(outside)
        check(
            GRAPH.durable_source_path(
                root / "./tools/aiworkhub/logs/../docs/guide.md", root
            ) == "tools/aiworkhub/docs/guide.md",
            "lexical dot segments did not select the durable target",
        )
        check(
            GRAPH.durable_source_path(root / "../outside.py", root) is None,
            "lexical parent escape was accepted",
        )
        symlinks_supported = True
        try:
            (root / "outside_alias.py").symlink_to(outside)
            (root / "nested_alias").symlink_to(Path(tmp), target_is_directory=True)
            runtime_alias = root / "runtime_alias.json"
            runtime_alias.symlink_to(callback)
        except (OSError, NotImplementedError):
            symlinks_supported = False

        connection = sqlite3.connect(":memory:")
        create_source_table(connection)
        old_root = GRAPH.LIB_ROOT
        GRAPH.LIB_ROOT = root
        try:
            count = GRAPH.populate_source_files(connection.cursor())
        finally:
            GRAPH.LIB_ROOT = old_root

        indexed = {
            row[0] for row in connection.execute("SELECT path FROM source_files")
        }
        expected = {
            "tools/aiworkhub/docs/guide.md",
            "tools/aiworkhub/runtime_notes.py",
            "ci/ordinary.py",
        }
        check(indexed == expected, f"unexpected indexed paths: {indexed!r}")
        check(count == len(expected), "populate count did not match inserted fixtures")
        check("tools/aiworkhub/logs/callback_bridge_state.json" not in indexed,
              "callback state entered source_files")
        if symlinks_supported:
            check("outside_alias.py" not in indexed, "out-of-root symlink was indexed")
            check("runtime_alias.json" not in indexed, "runtime alias was indexed")
            check(not any(path.startswith("nested_alias/") for path in indexed),
                  "nested symlink component was traversed")

        # Legacy freshness compares source mtimes represented by source_files.
        indexed_mtime = max((root / path).stat().st_mtime_ns for path in indexed)
        callback.write_text('{"advanced": true}\n', encoding="utf-8")
        check(callback.stat().st_mtime_ns > indexed_mtime,
              "callback fixture mtime did not advance")
        refreshed_mtime = max((root / path).stat().st_mtime_ns for path in indexed)
        check(refreshed_mtime == indexed_mtime,
              "excluded callback mtime affected legacy freshness inputs")

        # Feed lexical spellings through populate_source_files itself.  os.walk
        # normally returns canonical names, but callers/filesystem adapters need
        # the same fail-closed behavior.
        lexical_db = sqlite3.connect(":memory:")
        create_source_table(lexical_db)
        old_walk = GRAPH.os.walk
        GRAPH.LIB_ROOT = root
        GRAPH.os.walk = lambda unused: [(
            str(root),
            [],
            [
                "./ci/ordinary.py",
                "../outside.py",
                "tools/aiworkhub/logs/../docs/guide.md",
            ],
        )]
        try:
            GRAPH.populate_source_files(lexical_db.cursor())
        finally:
            GRAPH.os.walk = old_walk
            GRAPH.LIB_ROOT = old_root
        lexical_paths = {
            row[0] for row in lexical_db.execute("SELECT path FROM source_files")
        }
        check(
            lexical_paths == {"ci/ordinary.py", "tools/aiworkhub/docs/guide.md"},
            f"populate accepted unsafe lexical paths: {lexical_paths!r}",
        )


if __name__ == "__main__":
    test_normalization_and_policy()
    test_populate_source_files()
    print("project graph runtime exclusion tests: PASS")
