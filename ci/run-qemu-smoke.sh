#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    cat <<'EOF'
Usage:
    bash ./ci/run-qemu-smoke.sh arm64
    bash ./ci/run-qemu-smoke.sh riscv64
    bash ./ci/run-qemu-smoke.sh all

What it does:
  - cross-configures and builds the selected target(s)
  - runs QEMU smoke coverage for:
      run_selftest smoke
      test_bip324_standalone
      bench_kP
      bench_bip324

Requirements:
  ARM64:
    g++-13-aarch64-linux-gnu libc6-arm64-cross libc6-dev-arm64-cross qemu-user-static ninja-build
  RISC-V:
    g++-13-riscv64-linux-gnu libc6-riscv64-cross libc6-dev-riscv64-cross qemu-user-static ninja-build
EOF
}

find_qemu() {
    local arch="$1"
    if [[ "$arch" == "arm64" ]]; then
        command -v qemu-aarch64-static || command -v qemu-aarch64
    else
        command -v qemu-riscv64-static || command -v qemu-riscv64
    fi
}

configure_arm64() {
    cmake -S "$ROOT_DIR" -B "$ROOT_DIR/build-arm64" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_SYSTEM_NAME=Linux \
        -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
        -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc-13 \
        -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++-13 \
        -DSECP256K1_BUILD_TESTS=ON \
        -DSECP256K1_BUILD_BENCH=ON \
        -DSECP256K1_BUILD_METAL=OFF

    cmake --build "$ROOT_DIR/build-arm64" -j"$(nproc)"
}

configure_riscv64() {
    cmake -S "$ROOT_DIR" -B "$ROOT_DIR/build-riscv64" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_SYSTEM_NAME=Linux \
        -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
        -DCMAKE_C_COMPILER=riscv64-linux-gnu-gcc-13 \
        -DCMAKE_CXX_COMPILER=riscv64-linux-gnu-g++-13 \
        -DSECP256K1_BUILD_TESTS=ON \
        -DSECP256K1_BUILD_BENCH=ON \
        -DSECP256K1_BUILD_METAL=OFF

    cmake --build "$ROOT_DIR/build-riscv64" -j"$(nproc)"
}

run_smoke() {
    local arch="$1"
    local build_dir sysroot qemu

    if [[ "$arch" == "arm64" ]]; then
        build_dir="$ROOT_DIR/build-arm64"
        sysroot="/usr/aarch64-linux-gnu"
    else
        build_dir="$ROOT_DIR/build-riscv64"
        sysroot="/usr/riscv64-linux-gnu"
    fi

    qemu="$(find_qemu "$arch")"
    if [[ -z "$qemu" ]]; then
        echo "qemu for $arch was not found in PATH" >&2
        exit 1
    fi

    local selftest bip324 bench_kp bench_bip324
    selftest="$(find "$build_dir" -name run_selftest -type f -executable | head -1)"
    bip324="$(find "$build_dir" -name test_bip324_standalone -type f -executable | head -1)"
    bench_kp="$(find "$build_dir" -name bench_kP -type f -executable | head -1)"
    bench_bip324="$(find "$build_dir" -name bench_bip324 -type f -executable | head -1)"

    test -n "$selftest"
    test -n "$bip324"
    test -n "$bench_kp"
    test -n "$bench_bip324"

    echo "== $arch: run_selftest smoke =="
    timeout 180 "$qemu" -L "$sysroot" "$selftest" smoke
    echo
    echo "== $arch: test_bip324_standalone =="
    timeout 120 "$qemu" -L "$sysroot" "$bip324"
    echo
    echo "== $arch: bench_kP =="
    timeout 180 "$qemu" -L "$sysroot" "$bench_kp"
    echo
    echo "== $arch: bench_bip324 =="
    timeout 240 "$qemu" -L "$sysroot" "$bench_bip324"
}

main() {
    local target="${1:-all}"

    case "$target" in
        arm64)
            configure_arm64
            run_smoke arm64
            ;;
        riscv64)
            configure_riscv64
            run_smoke riscv64
            ;;
        all)
            configure_arm64
            run_smoke arm64
            configure_riscv64
            run_smoke riscv64
            ;;
        -h|--help|help)
            usage
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
}

main "$@"