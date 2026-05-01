#!/usr/bin/env bash
# clean_local_artifacts.sh — Remove local build dirs and caches safely.
#
# Does NOT delete:
#   - tracked source files
#   - committed documentation or evidence
#   - tools/source_graph_kit/source_graph.db (rebuilt separately)
#
# Safe to run at any time. Re-run configure_build.py to recreate build dirs.
#
# Usage:
#   bash ci/clean_local_artifacts.sh          # dry-run (shows what would be removed)
#   bash ci/clean_local_artifacts.sh --delete  # actually delete

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN=true
if [[ "${1:-}" == "--delete" ]]; then
    DRY_RUN=false
fi

REMOVED=0
TOTAL_SIZE=0

_remove() {
    local path="$1"
    if [[ ! -e "$path" && ! -L "$path" ]]; then
        return
    fi
    local size
    size=$(du -sh "$path" 2>/dev/null | cut -f1 || echo "?")
    echo "  ${size}  $path"
    REMOVED=$((REMOVED + 1))
    if [[ "$DRY_RUN" == "false" ]]; then
        rm -rf "$path"
    fi
}

echo "=== UltrafastSecp256k1 local artifact cleanup ==="
if [[ "$DRY_RUN" == "true" ]]; then
    echo "(dry-run — pass --delete to actually remove)"
fi
echo ""

echo "-- Canonical output root --"
_remove "out"

echo ""
echo "-- Legacy build directories (build/, build-*, build_*) --"
for d in build build-* build_*; do
    # Skip ci/files that start with "build" but are not directories
    [[ -d "$d" ]] && _remove "$d"
done

echo ""
echo "-- CMake build dirs (cmake-build-*) --"
for d in cmake-build-*; do
    [[ -d "$d" ]] && _remove "$d"
done

echo ""
echo "-- Legacy audit output dirs (migrated to out/audit-output/) --"
for d in audit-output-* audit-evidence-*; do
    [[ -d "$d" ]] && _remove "$d"
done

echo ""
echo "-- Nested bindings build dirs --"
for d in bindings/*/build bindings/*/.build bindings/*/dist bindings/*/node_modules; do
    [[ -d "$d" ]] && _remove "$d"
done

echo ""
echo "-- SQLite WAL/SHM sidecars --"
while IFS= read -r -d '' f; do
    _remove "$f"
done < <(find . -maxdepth 5 \( -name "*.db-shm" -o -name "*.db-wal" \) -print0 2>/dev/null)

echo ""
echo "-- Binary caches --"
for f in cache_w*.bin scan_lut_bench.bin secp256k1_gen_lut_v1.bin scan_lut_bench.bin precomputed/secp256k1_gen_lut_v1.bin; do
    [[ -f "$f" ]] && _remove "$f"
done

echo ""
echo "-- Local log / scratch files --"
for f in nohup.out a.out; do
    [[ -f "$f" ]] && _remove "$f"
done

echo ""
echo "=========================================="
if [[ "$DRY_RUN" == "true" ]]; then
    echo "Dry run complete — $REMOVED item(s) would be removed."
    echo "Re-run with --delete to actually clean up."
else
    echo "Cleanup complete — $REMOVED item(s) removed."
fi
