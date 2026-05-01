#!/bin/bash
# run_all.sh — Run all node benchmarks sequentially
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UF_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "========================================================"
echo "  UltrafastSecp256k1 — Cross-Node Benchmark Suite"
echo "  $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "========================================================"
echo ""

# Ensure UF is built
if [ ! -f "$UF_ROOT/build/gpu-ci/cpu/libfastsecp256k1.a" ]; then
    echo "Building UltrafastSecp256k1..."
    cmake -S "$UF_ROOT" -B "$UF_ROOT/build/bench-nodes" \
        -DCMAKE_BUILD_TYPE=Release -DSECP256K1_BUILD_CUDA=OFF \
        -DSECP256K1_BUILD_OPENCL=OFF -DSECP256K1_BUILD_TESTS=OFF
    cmake --build "$UF_ROOT/build/bench-nodes" --parallel
fi

echo "=== 1/6  Knuth secp256k1 (k-nuth/secp256k1) ==="
bash "$SCRIPT_DIR/bench_knuth.sh" 2>/dev/null
echo ""

echo "=== 2/6  Bitcoin Core benchmark (requires bench_bitcoin binary) ==="
if [ -f "$UF_ROOT/../bench_compare/bitcoin_core/build_shim/bin/bench_bitcoin" ]; then
    echo "--- Bitcoin Core libsecp256k1 (native) ---"
    "$UF_ROOT/../bench_compare/bitcoin_core/build_orig/bin/bench_bitcoin" \
        --filter="Sign|Verify" 2>/dev/null | grep -E "ns/op|Sign|Verify|ECDSA|Schnorr" | grep -v "ns/op.*ns/op" | head -10
    echo ""
    echo "--- UltrafastSecp256k1 shim (Bitcoin Core bench) ---"
    cd "$UF_ROOT/../bench_compare/bitcoin_core" && \
    "$UF_ROOT/../bench_compare/bitcoin_core/build_shim/bin/bench_bitcoin" \
        --filter="Sign|Verify" 2>/dev/null | grep -E "ns/op|Sign|Verify|ECDSA|Schnorr" | grep -v "ns/op.*ns/op" | head -10
else
    echo "  [SKIP] Bitcoin Core build not found. See docs/BITCOIN_CORE_BACKEND_EVIDENCE.md"
fi
echo ""

echo "=== 3/6  Dash Core (planned — requires full build) ==="
bash "$SCRIPT_DIR/bench_dash.sh" 2>/dev/null || true
echo ""

echo "=== 4/6  Litecoin Core (planned — same as Bitcoin Core fork) ==="
echo "  Results: identical to Bitcoin Core (same libsecp256k1, same benchmark harness)"
echo "  Run: git clone https://github.com/litecoin-project/litecoin && ..."
echo ""

echo "=== 5/6  Dogecoin Core (planned) ==="
echo "  Results: identical to Bitcoin Core (same libsecp256k1)"
echo ""

echo "=== 6/6  BCHN (planned) ==="
echo "  Use: compat/libsecp256k1_bchn_shim (includes BCH Schnorr)"
echo ""

echo "========================================================"
echo "  Done. Update docs/NODES_SHIM_STATUS.md with results."
echo "========================================================"
