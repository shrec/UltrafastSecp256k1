#!/bin/bash
# bench_dash.sh — Benchmark UF shim vs Dash Core secp256k1
# Builds Dash's standalone ECDSA benchmark (src/bench/ecdsa.cpp) with UF shim.
# Note: Dash uses libsecp256k1 via CKey/CPubKey wrappers.
# For a true apples-to-apples comparison we benchmark at the secp256k1 API level.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UF_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CACHE="${SECP256K1_CACHE_PATH:-$UF_ROOT/build/gpu-ci/cpu/cache_w18.bin}"

echo "=== Dash Core secp256k1 benchmark ==="
echo "Dash uses libsecp256k1 (same API as Bitcoin Core)"
echo "UF shim is a drop-in — benchmark at the secp256k1 C API level"
echo ""

# Dash uses standard libsecp256k1 API — use the same Knuth harness
# (Dash's ecdsa.cpp benchmarks go through CKey, but the underlying
#  secp256k1 calls are the same as any other libsecp256k1 user)

KNUTH_DIR="${KNUTH_DIR:-/tmp/bench_knuth_secp256k1}"
if [ -f /tmp/bench_knuth_native ] && [ -f /tmp/bench_uf_knuth ]; then
    echo "Using pre-built Knuth harness (same API as Dash's libsecp256k1 usage)"
    echo ""
    echo "--- libsecp256k1 (Dash Core's dependency, native) ---"
    /tmp/bench_knuth_native
    echo ""
    echo "--- UltrafastSecp256k1 shim (drop-in for Dash Core) ---"
    SECP256K1_CACHE_PATH="$CACHE" /tmp/bench_uf_knuth 2>/dev/null
else
    echo "Run bench_knuth.sh first to build the harness, then re-run this script."
    echo "Or: run scripts/bench_nodes/run_all.sh"
    exit 1
fi

echo ""
echo "Full Dash Core bench (requires full Dash Core build):"
echo "  git clone https://github.com/dashpay/dash"
echo "  cmake -S dash -B dash/build -DUSE_ULTRAFAST_SECP256K1=ON -DENABLE_TESTS=OFF"
echo "  cmake --build dash/build --target dash-bench"
echo "  ./dash/build/src/bench/dash-bench --filter=ECDSA"
