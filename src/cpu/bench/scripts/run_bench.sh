#!/usr/bin/env bash
# ============================================================================
# run_bench.sh -- Run bench_unified and generate JSON+TXT reports
# ============================================================================
#
# Usage:
#   ./run_bench.sh                    # full benchmark (default)
#   ./run_bench.sh --quick            # CI smoke test
#   ./run_bench.sh --out-dir /tmp/bench  # custom output directory
#   ./run_bench.sh --passes 21        # extra passes
#
# Outputs:
#   <out-dir>/bench_<platform>_<timestamp>.json
#   <out-dir>/bench_<platform>_<timestamp>.txt
#
# ============================================================================
set -euo pipefail

# ---- Defaults ----
OUT_DIR="."
EXTRA_ARGS=""
BENCH_BIN=""

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --quick)
            EXTRA_ARGS="$EXTRA_ARGS --quick --no-warmup"
            shift
            ;;
        --passes)
            EXTRA_ARGS="$EXTRA_ARGS --passes $2"
            shift 2
            ;;
        --bin)
            BENCH_BIN="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# ---- Find bench_unified binary ----
if [[ -z "$BENCH_BIN" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    # Try common build locations
    for candidate in \
        "$SCRIPT_DIR/../../../build_rel/libs/UltrafastSecp256k1/cpu/bench_unified" \
        "$SCRIPT_DIR/../../../../build_rel/libs/UltrafastSecp256k1/cpu/bench_unified" \
        "$SCRIPT_DIR/../bench_unified" \
        "$(dirname "$0")/../../../build/libs/UltrafastSecp256k1/cpu/bench_unified"; do
        if [[ -x "$candidate" ]]; then
            BENCH_BIN="$candidate"
            break
        fi
    done
fi

if [[ -z "$BENCH_BIN" || ! -x "$BENCH_BIN" ]]; then
    echo "[!] bench_unified binary not found. Build first or use --bin <path>"
    exit 1
fi

# ---- Platform tag ----
PLATFORM="$(uname -m)"
case "$PLATFORM" in
    x86_64)   PLATFORM="x86_64" ;;
    aarch64)  PLATFORM="arm64" ;;
    riscv64)  PLATFORM="riscv64" ;;
    *)        PLATFORM="$PLATFORM" ;;
esac

HOSTNAME_TAG="$(hostname -s 2>/dev/null || echo unknown)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TAG="${PLATFORM}_${HOSTNAME_TAG}_${TIMESTAMP}"

mkdir -p "$OUT_DIR"
JSON_FILE="$OUT_DIR/bench_${TAG}.json"
TXT_FILE="$OUT_DIR/bench_${TAG}.txt"

echo "=== bench_unified ==="
echo "  Binary:   $BENCH_BIN"
echo "  Platform: $PLATFORM ($HOSTNAME_TAG)"
echo "  JSON:     $JSON_FILE"
echo "  TXT:      $TXT_FILE"
echo "  Extra:    $EXTRA_ARGS"
echo ""

# ---- Run ----
# shellcheck disable=SC2086
"$BENCH_BIN" --json "$JSON_FILE" $EXTRA_ARGS 2>&1 | tee "$TXT_FILE"

echo ""
echo "=== Done ==="
echo "  JSON: $JSON_FILE"
echo "  TXT:  $TXT_FILE"
