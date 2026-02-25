#!/usr/bin/env bash
# ============================================================================
# repro.sh -- Reproducible Environment Report Generator (Linux/macOS)
# ============================================================================
# Usage: bash tools/repro.sh [output_file]
# ============================================================================
set -euo pipefail

OUTPUT="${1:-}"

section() { echo -e "\n$(printf '-%.0s' {1..60})\n  $1\n$(printf '-%.0s' {1..60})"; }

{
echo "UltrafastSecp256k1 -- Environment Report"
echo "Generated: $(date '+%Y-%m-%d %H:%M:%S %z')"

section "Git"
echo "  Commit:  $(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')"
echo "  Branch:  $(git branch --show-current 2>/dev/null || echo 'N/A')"
echo "  Tag:     $(git describe --tags --exact-match HEAD 2>/dev/null || echo '(no tag)')"
echo "  Dirty:   $(git diff --quiet 2>/dev/null && echo NO || echo YES)"

section "Operating System"
if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    echo "  OS:      ${PRETTY_NAME:-Linux}"
elif command -v sw_vers &>/dev/null; then
    echo "  OS:      macOS $(sw_vers -productVersion)"
fi
echo "  Kernel:  $(uname -r)"
echo "  Arch:    $(uname -m)"

section "CPU"
if [[ -f /proc/cpuinfo ]]; then
    echo "  Model:   $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
    echo "  Cores:   $(nproc)"
elif command -v sysctl &>/dev/null; then
    echo "  Model:   $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
    echo "  Cores:   $(sysctl -n hw.ncpu)"
fi

section "Memory"
if [[ -f /proc/meminfo ]]; then
    echo "  Total:   $(awk '/MemTotal/ {printf "%.1f GB", $2/1048576}' /proc/meminfo)"
elif command -v sysctl &>/dev/null; then
    echo "  Total:   $(echo "scale=1; $(sysctl -n hw.memsize) / 1073741824" | bc) GB"
fi

section "Compilers"
for cc in gcc g++ clang clang++ nvcc; do
    if command -v "$cc" &>/dev/null; then
        echo "  $cc: $($cc --version 2>&1 | head -1)"
    fi
done

section "CMake"
command -v cmake &>/dev/null && cmake --version | head -1 || echo "  (not found)"
command -v ninja &>/dev/null && echo "  Ninja: $(ninja --version)" || true

section "GPU"
if command -v nvidia-smi &>/dev/null; then
    echo "  GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    echo "  VRAM:    $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)"
    echo "  Driver:  $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
else
    echo "  (no NVIDIA GPU detected)"
fi

echo ""
echo "$(printf '-%.0s' {1..60})"
echo "  End of Report"
echo "$(printf '-%.0s' {1..60})"

} | if [[ -n "$OUTPUT" ]]; then tee "$OUTPUT"; else cat; fi
