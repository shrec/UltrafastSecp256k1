#!/usr/bin/env bash
# bench/scripts/run.sh -- Run bench_compare with sane defaults (Linux)
# Usage: bash bench/scripts/run.sh [--pin <core>] [extra bench_compare args...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$REPO_ROOT/build/bench-compare"

BIN="$BUILD_DIR/bench/bench_compare"
if [ ! -x "$BIN" ]; then
    echo "[!] bench_compare not found at $BIN"
    echo "    Run: bash bench/scripts/build.sh"
    exit 1
fi

# Default arguments
PIN_CORE=2
EXTRA_ARGS=()

# Parse our wrapper args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pin)
            PIN_CORE="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Set CPU governor to performance if available (best effort)
if command -v cpupower >/dev/null 2>&1; then
    sudo cpupower frequency-set -g performance 2>/dev/null || true
fi

echo "=== bench_compare run ==="
echo "  Binary     : $BIN"
echo "  Pin core   : $PIN_CORE"
echo "  Extra args : ${EXTRA_ARGS[*]:-<none>}"
echo ""

"$BIN" \
    --pin-core="$PIN_CORE" \
    --n=100000 \
    --warmup=500 \
    --measure=3000 \
    --json=report.json \
    "${EXTRA_ARGS[@]}"

echo ""
echo "[OK] Report written to: report.json"
