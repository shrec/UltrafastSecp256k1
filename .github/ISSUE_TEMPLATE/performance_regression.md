---
name: Performance Regression
about: Report a measurable performance degradation after an update
title: '[PERF] '
labels: 'performance'
assignees: ''

---

**Environment**
- OS: [e.g. Ubuntu 24.04, macOS 15, Windows 11]
- Compiler: [e.g. Clang 21, GCC 14]
- Backend: [CPU / CUDA / Metal / OpenCL]
- Architecture: [x86-64 / ARM64 / RISC-V]
- Library version (current): [e.g. v3.68.0, commit abc1234]
- Library version (baseline): [e.g. v3.68.0, commit def5678]

**Regression summary**
Which operation(s) regressed, and by how much?

| Operation | Baseline | Current | Regression |
|---|---|---|---|
| e.g. ECDSA Sign | 16 μs | 22 μs | -27% |

**How to reproduce**

```bash
# Build commands
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Benchmark command
./build/cpu/bench_unified --quick
```

**Compiler flags**
```
# Paste output of: cmake -LA build | grep -E "FLAGS|BUILD_TYPE|ASM"
```

**repro.ps1 / repro.sh output**
If available, paste the output of `pwsh tools/repro.ps1` or `bash tools/repro.sh`:
<details>
<summary>Environment report</summary>

```
(paste here)
```
</details>

**CPU microarchitecture details** (if relevant)
```bash
# Linux: lscpu | grep -i "model\|cache\|flag"
# Windows: wmic cpu get Name,NumberOfCores,L2CacheSize
```

**Analysis (optional)**
If you've investigated, describe what you found:
- Which function/loop regressed?
- Was it a code change or a compiler change?
- Are the builds using identical flags?

**Additional context**
Perf counters, VTune/perf output, flamegraphs, etc.
