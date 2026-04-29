---
name: Bug Report
about: Report incorrect behavior, crashes, or build failures
title: '[BUG] '
labels: 'bug'
assignees: ''

---

**Environment**
- OS: [e.g. Ubuntu 24.04, macOS 15, Windows 11]
- Compiler: [e.g. GCC 13, Clang 17, MSVC 2022, AppleClang]
- Backend: [CPU / CUDA / Metal / OpenCL / WASM / ROCm]
- Architecture: [x86-64 / ARM64 / RISC-V / ESP32]
- Library version: [e.g. v3.68.0]

**Describe the bug**
A clear description of the incorrect behavior.

**To Reproduce**
```bash
# Build commands used
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
# Command that triggers the bug
./build/cpu/test_runner
```

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened. Include error messages, stack traces, or test output.

**Selftest output**
If relevant, paste the selftest output:
```
./build/cpu/test_runner --selftest
```

**Additional context**
Any other information (compiler flags, config.json, hardware specs, etc.)
