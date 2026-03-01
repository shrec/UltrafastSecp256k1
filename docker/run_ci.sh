#!/usr/bin/env bash
# =============================================================================
# UltrafastSecp256k1 -- Local CI Test Runner (runs inside Docker)
# =============================================================================
# Usage:
#   ./docker/run_ci.sh all            # Run everything (~5-8 min)
#   ./docker/run_ci.sh quick          # linux-gcc Release + WASM KAT (~2 min)
#   ./docker/run_ci.sh wasm           # WASM build + KAT only (~1 min)
#   ./docker/run_ci.sh linux-gcc      # GCC Release build + tests
#   ./docker/run_ci.sh linux-clang    # Clang Release build + tests
#   ./docker/run_ci.sh linux-debug    # GCC Debug build + tests
#   ./docker/run_ci.sh sanitizers     # ASan+UBSan (Clang Debug)
#   ./docker/run_ci.sh tsan           # TSan (Clang Debug)
#   ./docker/run_ci.sh valgrind       # Valgrind memcheck
#   ./docker/run_ci.sh clang-tidy     # Static analysis
#   ./docker/run_ci.sh arm64          # ARM64 cross-compile check
#   ./docker/run_ci.sh coverage       # Code coverage (LLVM)
# =============================================================================
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

NPROC=$(nproc 2>/dev/null || echo 4)
PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
declare -a FAILED_JOBS=()

banner() {
    echo ""
    echo -e "${CYAN}================================================================${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}================================================================${NC}"
}

run_job() {
    local name="$1"
    shift
    banner "$name"
    local start_time
    start_time=$(date +%s)
    if "$@"; then
        local end_time
        end_time=$(date +%s)
        local elapsed=$((end_time - start_time))
        echo -e "${GREEN}[PASS] ${name} (${elapsed}s)${NC}"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        local end_time
        end_time=$(date +%s)
        local elapsed=$((end_time - start_time))
        echo -e "${RED}[FAIL] ${name} (${elapsed}s)${NC}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_JOBS+=("$name")
    fi
}

# -- Individual jobs -----------------------------------------------------------

job_linux_gcc_release() {
    local bd="build-ci/gcc-rel"
    rm -rf "$bd"
    CC=gcc-13 CXX=g++-13 cmake -S . -B "$bd" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DSECP256K1_BUILD_TESTS=ON \
        -DSECP256K1_BUILD_BENCH=ON \
        -DSECP256K1_BUILD_EXAMPLES=ON \
        -DSECP256K1_BUILD_FUZZ_TESTS=ON \
        -DSECP256K1_BUILD_PROTOCOL_TESTS=ON
    cmake --build "$bd" -j"$NPROC"
    ctest --test-dir "$bd" --output-on-failure -j"$NPROC" -E "^ct_sidechannel"
}

job_linux_gcc_debug() {
    local bd="build-ci/gcc-dbg"
    rm -rf "$bd"
    CC=gcc-13 CXX=g++-13 cmake -S . -B "$bd" -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DSECP256K1_BUILD_TESTS=ON \
        -DSECP256K1_BUILD_BENCH=ON \
        -DSECP256K1_BUILD_EXAMPLES=ON \
        -DSECP256K1_BUILD_FUZZ_TESTS=ON \
        -DSECP256K1_BUILD_PROTOCOL_TESTS=ON
    cmake --build "$bd" -j"$NPROC"
    ctest --test-dir "$bd" --output-on-failure -j"$NPROC" -E "^ct_sidechannel"
}

job_linux_clang_release() {
    local bd="build-ci/clang-rel"
    rm -rf "$bd"
    CC=clang-17 CXX=clang++-17 cmake -S . -B "$bd" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DSECP256K1_BUILD_TESTS=ON \
        -DSECP256K1_BUILD_BENCH=ON \
        -DSECP256K1_BUILD_EXAMPLES=ON \
        -DSECP256K1_BUILD_METAL=ON \
        -DSECP256K1_BUILD_FUZZ_TESTS=ON \
        -DSECP256K1_BUILD_PROTOCOL_TESTS=ON
    cmake --build "$bd" -j"$NPROC"
    ctest --test-dir "$bd" --output-on-failure -j"$NPROC" -E "^ct_sidechannel"
}

job_linux_clang_debug() {
    local bd="build-ci/clang-dbg"
    rm -rf "$bd"
    CC=clang-17 CXX=clang++-17 cmake -S . -B "$bd" -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DSECP256K1_BUILD_TESTS=ON \
        -DSECP256K1_BUILD_FUZZ_TESTS=ON \
        -DSECP256K1_BUILD_PROTOCOL_TESTS=ON
    cmake --build "$bd" -j"$NPROC"
    ctest --test-dir "$bd" --output-on-failure -j"$NPROC" -E "^ct_sidechannel"
}

job_sanitizers_asan() {
    local bd="build-ci/asan"
    rm -rf "$bd"
    CC=clang-17 CXX=clang++-17 cmake -S . -B "$bd" -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DSECP256K1_BUILD_TESTS=ON \
        -DSECP256K1_BUILD_FUZZ_TESTS=ON \
        -DSECP256K1_BUILD_PROTOCOL_TESTS=ON \
        -DSECP256K1_USE_ASM=OFF \
        -DCMAKE_C_FLAGS="-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer" \
        -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer" \
        -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
    cmake --build "$bd" -j"$NPROC"
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
    UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    ctest --test-dir "$bd" --output-on-failure -j"$NPROC" \
        -E "^(ct_sidechannel|unified_audit)" --timeout 300
}

job_sanitizers_tsan() {
    local bd="build-ci/tsan"
    rm -rf "$bd"
    CC=clang-17 CXX=clang++-17 cmake -S . -B "$bd" -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DSECP256K1_BUILD_TESTS=ON \
        -DSECP256K1_BUILD_FUZZ_TESTS=ON \
        -DSECP256K1_BUILD_PROTOCOL_TESTS=ON \
        -DSECP256K1_USE_ASM=OFF \
        -DCMAKE_C_FLAGS="-fsanitize=thread -fno-omit-frame-pointer" \
        -DCMAKE_CXX_FLAGS="-fsanitize=thread -fno-omit-frame-pointer" \
        -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread"
    cmake --build "$bd" -j"$NPROC"
    ctest --test-dir "$bd" --output-on-failure -j"$NPROC" \
        -E "^(ct_sidechannel|unified_audit)" --timeout 300
}

job_valgrind() {
    local bd="build-ci/valgrind"
    rm -rf "$bd"
    CC=gcc-13 CXX=g++-13 cmake -S . -B "$bd" -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DSECP256K1_BUILD_TESTS=ON
    cmake --build "$bd" -j"$NPROC"
    ctest --test-dir "$bd" --output-on-failure -j"$NPROC" \
        -E "^ct_sidechannel" -T MemCheck \
        --overwrite MemoryCheckCommandOptions="--leak-check=full --error-exitcode=1"
}

job_wasm() {
    local bd="build-ci/wasm"
    rm -rf "$bd"
    # Source Emscripten env
    # shellcheck disable=SC1091
    source /emsdk/emsdk_env.sh 2>/dev/null || true
    emcmake cmake -S wasm -B "$bd" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$bd" -j"$NPROC"
    echo "WASM artifacts:"
    ls -lh "$bd/dist/secp256k1_wasm.js" "$bd/dist/secp256k1_wasm.wasm" 2>/dev/null || true
    echo "KAT test:"
    ls -lh "$bd/kat/" 2>/dev/null || true
    node "$bd/kat/wasm_kat_test.js"
}

job_arm64() {
    local bd="build-ci/arm64"
    rm -rf "$bd"
    cmake -S . -B "$bd" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_SYSTEM_NAME=Linux \
        -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
        -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc-13 \
        -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++-13 \
        -DSECP256K1_BUILD_TESTS=ON \
        -DSECP256K1_BUILD_BENCH=ON \
        -DSECP256K1_BUILD_METAL=OFF
    cmake --build "$bd" -j"$NPROC"
    echo "ARM64 library:"
    file "$bd/cpu/libfastsecp256k1.a"
    echo "Size: $(du -h "$bd/cpu/libfastsecp256k1.a" | cut -f1)"
}

job_clang_tidy() {
    local bd="build-ci/tidy"
    rm -rf "$bd"
    CC=clang-17 CXX=clang++-17 cmake -S . -B "$bd" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DSECP256K1_BUILD_TESTS=ON \
        -DSECP256K1_BUILD_BENCH=ON \
        -DSECP256K1_BUILD_EXAMPLES=ON
    cmake --build "$bd" -j"$NPROC"
    # Run clang-tidy on source files (warnings only, non-blocking)
    local files
    files=$(python3 -c "
import json, sys
with open('$bd/compile_commands.json') as f:
    cmds = json.load(f)
for c in cmds:
    f = c['file']
    if f.endswith(('.cpp','.cc','.cxx')) and '/tests/' not in f and '/bench/' not in f:
        print(f)
" 2>/dev/null || true)
    if [ -n "$files" ]; then
        echo "$files" | head -20 | xargs -P"$NPROC" -I{} \
            clang-tidy-17 -p "$bd" {} 2>&1 || true
        echo -e "${YELLOW}[INFO] clang-tidy completed (warnings only)${NC}"
    fi
}

job_coverage() {
    local bd="build-ci/cov"
    rm -rf "$bd"
    CC=clang-17 CXX=clang++-17 cmake -S . -B "$bd" -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DSECP256K1_BUILD_TESTS=ON \
        -DSECP256K1_BUILD_BENCH=OFF \
        -DSECP256K1_BUILD_FUZZ_TESTS=ON \
        -DSECP256K1_BUILD_PROTOCOL_TESTS=ON \
        -DSECP256K1_USE_ASM=OFF \
        -DCMAKE_C_FLAGS="-fprofile-instr-generate -fcoverage-mapping" \
        -DCMAKE_CXX_FLAGS="-fprofile-instr-generate -fcoverage-mapping" \
        -DCMAKE_EXE_LINKER_FLAGS="-fprofile-instr-generate"
    cmake --build "$bd" -j"$NPROC"
    LLVM_PROFILE_FILE="$bd/%p-%m.profraw" \
    ctest --test-dir "$bd" --output-on-failure -j"$NPROC" -E "^ct_sidechannel"

    echo "Merging coverage profiles..."
    find "$bd" -name '*.profraw' -print0 | \
        xargs -0 llvm-profdata-17 merge -sparse -o coverage.profdata

    OBJECTS=""
    for bin in $(find "$bd" -type f -executable); do
        if llvm-cov-17 show --instr-profile=coverage.profdata "$bin" >/dev/null 2>&1; then
            OBJECTS="$OBJECTS -object=$bin"
        fi
    done

    if [ -n "$OBJECTS" ]; then
        echo "=== Coverage Summary ==="
        # shellcheck disable=SC2086
        llvm-cov-17 report \
            --instr-profile=coverage.profdata \
            $OBJECTS \
            --ignore-filename-regex='(tests/|bench/|examples/|/usr/)' \
            | tail -10
    fi
    rm -f coverage.profdata
}

job_compiler_warnings() {
    local bd="build-ci/warnings"
    rm -rf "$bd"
    CC=gcc-13 CXX=g++-13 cmake -S . -B "$bd" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS="-Werror -Wall -Wextra -Wpedantic -Wconversion -Wshadow" \
        -DSECP256K1_BUILD_TESTS=ON
    cmake --build "$bd" -j"$NPROC"
}

job_audit() {
    # Mirrors audit-report.yml (Linux GCC-13 + Linux Clang-17)
    local pass=1
    for compiler in gcc-13 clang-17; do
        local bd="build-ci/audit-${compiler}"
        rm -rf "$bd"
        if [ "$compiler" = "gcc-13" ]; then
            local cc=gcc-13 cxx=g++-13
        else
            local cc=clang-17 cxx=clang++-17
        fi
        CC="$cc" CXX="$cxx" cmake -S . -B "$bd" -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_TESTING=ON \
            -DSECP256K1_BUILD_TESTS=ON \
            -DSECP256K1_BUILD_PROTOCOL_TESTS=ON \
            -DSECP256K1_BUILD_FUZZ_TESTS=ON
        cmake --build "$bd" -j"$NPROC"
        mkdir -p "audit-output-${compiler}"
        "$bd/audit/unified_audit_runner" \
            --report-dir "./audit-output-${compiler}" || true
        # Check verdict
        if [ -f "audit-output-${compiler}/audit_report.json" ]; then
            local verdict
            verdict=$(grep -o '"audit_verdict": *"[^"]*"' "audit-output-${compiler}/audit_report.json" | head -1 | cut -d'"' -f4)
            echo "Audit verdict ($compiler): $verdict"
            if [ "$verdict" = "FAIL" ]; then
                # Check if failures are advisory-only
                local real_fail
                real_fail=$(grep -c '"advisory": *false.*"result": *"FAIL"' "audit-output-${compiler}/audit_report.json" 2>/dev/null || echo "0")
                if [ "$real_fail" != "0" ]; then
                    pass=0
                else
                    echo "All failures are advisory -- treating as PASS"
                fi
            fi
        else
            echo "WARNING: audit report not generated for $compiler"
            pass=0
        fi
    done
    [ "$pass" -eq 1 ]
}

# -- Orchestration -------------------------------------------------------------

print_summary() {
    echo ""
    echo -e "${CYAN}================================================================${NC}"
    echo -e "${CYAN}  LOCAL CI SUMMARY${NC}"
    echo -e "${CYAN}================================================================${NC}"
    echo -e "  ${GREEN}PASSED: ${PASS_COUNT}${NC}"
    [ "$FAIL_COUNT" -gt 0 ] && echo -e "  ${RED}FAILED: ${FAIL_COUNT}${NC}" || echo -e "  FAILED: 0"
    [ "$SKIP_COUNT" -gt 0 ] && echo -e "  ${YELLOW}SKIPPED: ${SKIP_COUNT}${NC}"
    if [ "${#FAILED_JOBS[@]}" -gt 0 ]; then
        echo ""
        echo -e "  ${RED}Failed jobs:${NC}"
        for job in "${FAILED_JOBS[@]}"; do
            echo -e "    ${RED}- ${job}${NC}"
        done
    fi
    echo -e "${CYAN}================================================================${NC}"
    [ "$FAIL_COUNT" -eq 0 ]
}

case "${1:-help}" in
    all)
        run_job "Linux GCC Release"     job_linux_gcc_release
        run_job "Linux GCC Debug"       job_linux_gcc_debug
        run_job "Linux Clang Release"   job_linux_clang_release
        run_job "Linux Clang Debug"     job_linux_clang_debug
        run_job "ASan + UBSan"          job_sanitizers_asan
        run_job "TSan"                  job_sanitizers_tsan
        run_job "Valgrind"              job_valgrind
        run_job "WASM (Emscripten)"     job_wasm
        run_job "ARM64 cross-compile"   job_arm64
        run_job "Compiler Warnings"     job_compiler_warnings
        run_job "clang-tidy"            job_clang_tidy
        run_job "Unified Audit"         job_audit
        run_job "Code Coverage"         job_coverage
        print_summary
        ;;
    quick)
        run_job "Linux GCC Release"     job_linux_gcc_release
        run_job "WASM (Emscripten)"     job_wasm
        print_summary
        ;;
    pre-push)
        # Pre-push validation: the minimum set that catches 95% of CI failures
        # Runs in ~3-5 min instead of ~30 min for full CI
        run_job "Compiler Warnings"     job_compiler_warnings
        run_job "Linux GCC Release"     job_linux_gcc_release
        run_job "Linux Clang Release"   job_linux_clang_release
        run_job "ASan + UBSan"          job_sanitizers_asan
        run_job "Unified Audit"         job_audit
        print_summary
        ;;
    linux-gcc)
        run_job "Linux GCC Release"     job_linux_gcc_release
        print_summary
        ;;
    linux-clang)
        run_job "Linux Clang Release"   job_linux_clang_release
        print_summary
        ;;
    linux-debug)
        run_job "Linux GCC Debug"       job_linux_gcc_debug
        print_summary
        ;;
    sanitizers|asan)
        run_job "ASan + UBSan"          job_sanitizers_asan
        print_summary
        ;;
    tsan)
        run_job "TSan"                  job_sanitizers_tsan
        print_summary
        ;;
    valgrind)
        run_job "Valgrind"              job_valgrind
        print_summary
        ;;
    wasm)
        run_job "WASM (Emscripten)"     job_wasm
        print_summary
        ;;
    arm64)
        run_job "ARM64 cross-compile"   job_arm64
        print_summary
        ;;
    clang-tidy|tidy)
        run_job "clang-tidy"            job_clang_tidy
        print_summary
        ;;
    coverage|cov)
        run_job "Code Coverage"         job_coverage
        print_summary
        ;;
    warnings)
        run_job "Compiler Warnings"     job_compiler_warnings
        print_summary
        ;;
    audit)
        run_job "Unified Audit"         job_audit
        print_summary
        ;;
    help|*)
        echo "UltrafastSecp256k1 Local CI Runner"
        echo ""
        echo "Usage: $0 <target>"
        echo ""
        echo "Targets:"
        echo "  all           Run ALL CI jobs (~5-8 min)"
        echo "  quick         GCC Release + WASM KAT (~2 min)"
        echo "  linux-gcc     GCC 13 Release build + tests"
        echo "  linux-clang   Clang 17 Release build + tests"
        echo "  linux-debug   GCC 13 Debug build + tests"
        echo "  sanitizers    ASan + UBSan (Clang Debug)"
        echo "  tsan          ThreadSanitizer (Clang Debug)"
        echo "  valgrind      Valgrind memcheck"
        echo "  wasm          WASM (Emscripten 3.1.51) + KAT"
        echo "  arm64         ARM64 cross-compile check"
        echo "  clang-tidy    Static analysis"
        echo "  coverage      Code coverage (LLVM)"
        echo "  warnings      -Werror strict warnings"
        echo "  audit         Unified audit runner (GCC+Clang)"
        echo "  pre-push      Pre-push gate (warnings+tests+asan+audit ~5min)"
        echo "  help          This message"
        ;;
esac
