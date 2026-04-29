# Node Benchmark Scripts

Each script builds a node with UltrafastSecp256k1 shim and runs its native benchmark harness.

## Usage

```bash
# Single node
bash scripts/bench_nodes/bench_bitcoin_core.sh
bash scripts/bench_nodes/bench_dash.sh
bash scripts/bench_nodes/bench_litecoin.sh
bash scripts/bench_nodes/bench_dogecoin.sh
bash scripts/bench_nodes/bench_knuth.sh
bash scripts/bench_nodes/bench_bchn.sh

# All nodes (sequential)
bash scripts/bench_nodes/run_all.sh
```

Results are written to `docs/NODES_SHIM_STATUS.md` (update manually from output).

## Requirements

- UltrafastSecp256k1 built: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build`
- Each node: cloned adjacent to UltrafastSecp256k1 (or set `NODE_DIR` env var)
- Standard build tools: cmake, gcc/clang, ninja, python3
