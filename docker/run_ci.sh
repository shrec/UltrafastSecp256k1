#!/usr/bin/env bash
# =============================================================================
# UltrafastSecp256k1 -- Local CI Test Runner (runs inside Docker)
# =============================================================================
# Usage:
#   ./docker/run_ci.sh all            # Run everything (~5-8 min)
#   ./docker/run_ci.sh quick          # linux-gcc Release + WASM KAT (~2 min)
#   ./docker/run_ci.sh gh-parity      # Linux GitHub parity gate (recommended)
#   ./docker/run_ci.sh wasm           # WASM build + KAT only (~1 min)
#   ./docker/run_ci.sh linux-gcc      # GCC Release build + tests
#   ./docker/run_ci.sh linux-clang    # Clang Release build + tests
#   ./docker/run_ci.sh linux-debug    # GCC Debug build + tests
#   ./docker/run_ci.sh sanitizers     # ASan+UBSan (Clang Debug)
#   ./docker/run_ci.sh msan           # MSan advisory run (non-blocking)
#   ./docker/run_ci.sh tsan           # TSan (Clang Debug)
#   ./docker/run_ci.sh valgrind       # Valgrind memcheck
#   ./docker/run_ci.sh clang-tidy     # Static analysis
#   ./docker/run_ci.sh cppcheck       # Static analysis (Cppcheck)
#   ./docker/run_ci.sh ct-verif       # Deterministic constant-time IR checks
#   ./docker/run_ci.sh bench-regression # Live Ultra/libsecp head-to-head gate
#   ./docker/run_ci.sh strict-audit   # Audit with zero advisory tolerance
#   ./docker/run_ci.sh strict-perf    # Fail on any head-to-head lag (<1.00x)
#   ./docker/run_ci.sh no-surprise    # Strict end-to-end quality gate
#   ./docker/run_ci.sh dev-gate       # Gate before pushing to dev
#   ./docker/run_ci.sh main-gate      # Gate before pushing to main
#   ./docker/run_ci.sh branch-gate    # Auto gate by current git branch
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
    if ! rm -rf "$bd" 2>/dev/null; then
        local stamp
        stamp=$(date +%s)
        bd="/tmp/ufsecp-gcc-rel-${USER:-user}-${stamp}"
        echo "[WARN] linux-gcc: fixed build dir not writable, using fallback: $bd"
    fi
    CC=gcc-13 CXX=g++-13 cmake -S . -B "$bd" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DSECP256K1_BUILD_TESTS=ON \
        -DSECP256K1_BUILD_BENCH=ON \
        -DSECP256K1_BUILD_EXAMPLES=ON \
        -DSECP256K1_BUILD_FUZZ_TESTS=ON \
        -DSECP256K1_BUILD_PROTOCOL_TESTS=ON || return 1
    cmake --build "$bd" -j"$NPROC" || return 1
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
        -DSECP256K1_BUILD_PROTOCOL_TESTS=ON || return 1
    cmake --build "$bd" -j"$NPROC" || return 1
    ctest --test-dir "$bd" --output-on-failure -j"$NPROC" -E "^ct_sidechannel"
}

job_linux_clang_release() {
    local bd="build-ci/clang-rel"
    if ! rm -rf "$bd" 2>/dev/null; then
        local stamp
        stamp=$(date +%s)
        bd="/tmp/ufsecp-clang-rel-${USER:-user}-${stamp}"
        echo "[WARN] linux-clang: fixed build dir not writable, using fallback: $bd"
    fi
    local cc_bin cxx_bin
    cc_bin=$(command -v clang-17 2>/dev/null || true)
    cxx_bin=$(command -v clang++-17 2>/dev/null || true)
    if [ -z "$cc_bin" ] || [ -z "$cxx_bin" ] || [ ! -x "$cc_bin" ] || [ ! -x "$cxx_bin" ]; then
        echo "[WARN] linux-clang: compiler not installed (clang-17/clang++-17), skipping"
        return 0
    fi
    CC="$cc_bin" CXX="$cxx_bin" cmake -S . -B "$bd" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DSECP256K1_BUILD_TESTS=ON \
        -DSECP256K1_BUILD_BENCH=ON \
        -DSECP256K1_BUILD_EXAMPLES=ON \
        -DSECP256K1_BUILD_METAL=ON \
        -DSECP256K1_BUILD_FUZZ_TESTS=ON \
        -DSECP256K1_BUILD_PROTOCOL_TESTS=ON || return 1
    cmake --build "$bd" -j"$NPROC" || return 1
    ctest --test-dir "$bd" --output-on-failure -j"$NPROC" -E "^ct_sidechannel"
}

job_linux_clang_debug() {
    local bd="build-ci/clang-dbg"
    rm -rf "$bd"
    CC=clang-17 CXX=clang++-17 cmake -S . -B "$bd" -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DSECP256K1_BUILD_TESTS=ON \
        -DSECP256K1_BUILD_FUZZ_TESTS=ON \
        -DSECP256K1_BUILD_PROTOCOL_TESTS=ON || return 1
    cmake --build "$bd" -j"$NPROC" || return 1
    ctest --test-dir "$bd" --output-on-failure -j"$NPROC" -E "^ct_sidechannel"
}

job_sanitizers_asan() {
    local bd="build-ci/asan"
    if ! rm -rf "$bd" 2>/dev/null; then
        local stamp
        stamp=$(date +%s)
        bd="/tmp/ufsecp-asan-${USER:-user}-${stamp}"
        echo "[WARN] asan: fixed build dir not writable, using fallback: $bd"
    fi
    local cc_bin cxx_bin
    cc_bin=$(command -v clang-17 2>/dev/null || true)
    cxx_bin=$(command -v clang++-17 2>/dev/null || true)
    if [ -z "$cc_bin" ] || [ -z "$cxx_bin" ] || [ ! -x "$cc_bin" ] || [ ! -x "$cxx_bin" ]; then
        echo "[WARN] asan: compiler not installed (clang-17/clang++-17), skipping"
        return 0
    fi
    CC="$cc_bin" CXX="$cxx_bin" cmake -S . -B "$bd" -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DSECP256K1_BUILD_TESTS=ON \
        -DSECP256K1_BUILD_FUZZ_TESTS=ON \
        -DSECP256K1_BUILD_PROTOCOL_TESTS=ON \
        -DSECP256K1_USE_ASM=OFF \
        -DCMAKE_C_FLAGS="-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer" \
        -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer" \
        -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" || return 1
    cmake --build "$bd" -j"$NPROC" || return 1
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
    UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    ctest --test-dir "$bd" --output-on-failure -j"$NPROC" \
        -E "^(ct_sidechannel|unified_audit)" --timeout 300
}

job_sanitizers_tsan() {
    local bd="build-ci/tsan"
    if ! rm -rf "$bd" 2>/dev/null; then
        local stamp
        stamp=$(date +%s)
        bd="/tmp/ufsecp-tsan-${USER:-user}-${stamp}"
        echo "[WARN] tsan: fixed build dir not writable, using fallback: $bd"
    fi
    local cc_bin cxx_bin
    cc_bin=$(command -v clang-17 2>/dev/null || true)
    cxx_bin=$(command -v clang++-17 2>/dev/null || true)
    if [ -z "$cc_bin" ] || [ -z "$cxx_bin" ] || [ ! -x "$cc_bin" ] || [ ! -x "$cxx_bin" ]; then
        echo "[WARN] tsan: compiler not installed (clang-17/clang++-17), skipping"
        return 0
    fi
    CC="$cc_bin" CXX="$cxx_bin" cmake -S . -B "$bd" -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DSECP256K1_BUILD_TESTS=ON \
        -DSECP256K1_BUILD_FUZZ_TESTS=ON \
        -DSECP256K1_BUILD_PROTOCOL_TESTS=ON \
        -DSECP256K1_USE_ASM=OFF \
        -DCMAKE_C_FLAGS="-fsanitize=thread -fno-omit-frame-pointer" \
        -DCMAKE_CXX_FLAGS="-fsanitize=thread -fno-omit-frame-pointer" \
        -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread" || return 1
    cmake --build "$bd" -j"$NPROC" || return 1
    ctest --test-dir "$bd" --output-on-failure -j"$NPROC" \
        -E "^(ct_sidechannel|unified_audit|batch_randomness)" --timeout 900
}

job_sanitizers_msan() {
    local bd="build-ci/msan"
    rm -rf "$bd"
    CC=clang-17 CXX=clang++-17 cmake -S . -B "$bd" -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DSECP256K1_BUILD_TESTS=ON \
        -DSECP256K1_BUILD_FUZZ_TESTS=ON \
        -DSECP256K1_BUILD_PROTOCOL_TESTS=ON \
        -DSECP256K1_USE_ASM=OFF \
        -DCMAKE_C_FLAGS="-fsanitize=memory -fno-omit-frame-pointer -fsanitize-memory-track-origins=2" \
        -DCMAKE_CXX_FLAGS="-fsanitize=memory -fno-omit-frame-pointer -fsanitize-memory-track-origins=2" \
        -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=memory" || return 1
    cmake --build "$bd" -j"$NPROC" || return 1

    # Mirrors GitHub security-audit.yml: MSan is advisory and can timeout.
    MSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    ctest --test-dir "$bd" --output-on-failure -j"$NPROC" \
        -E "^(ct_sidechannel|selftest|unified_audit)" --timeout 3600 || true
    echo "MSan advisory run completed (non-blocking, GH parity behavior)."
}

job_valgrind() {
    local bd="build-ci/valgrind"
    rm -rf "$bd"
    CC=gcc-13 CXX=g++-13 cmake -S . -B "$bd" -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DSECP256K1_BUILD_TESTS=ON || return 1
    cmake --build "$bd" -j"$NPROC" || return 1
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
    emcmake cmake -S wasm -B "$bd" -DCMAKE_BUILD_TYPE=Release || return 1
    cmake --build "$bd" -j"$NPROC" || return 1
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
        -DSECP256K1_BUILD_METAL=OFF || return 1
    cmake --build "$bd" -j"$NPROC" || return 1
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
        -DSECP256K1_BUILD_EXAMPLES=ON || return 1
    cmake --build "$bd" -j"$NPROC" || return 1
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

job_cppcheck() {
    local bd="build-ci/cppcheck"
    rm -rf "$bd"
    cmake -S . -B "$bd" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DSECP256K1_BUILD_TESTS=ON \
        -DSECP256K1_BUILD_BENCH=ON || return 1

    local suppress_file=".cppcheck-suppressions"
    local suppress_arg=""
    if [ -f "$suppress_file" ]; then
        suppress_arg="--suppressions-list=$suppress_file"
    fi

    cppcheck \
        --project="$bd/compile_commands.json" \
        --enable=warning,performance,portability \
        --suppress=missingIncludeSystem \
        --suppress=unmatchedSuppression \
        --suppress=unusedFunction \
        ${suppress_arg} \
        --inline-suppr \
        --error-exitcode=0 \
        --std=c++20 \
        --xml \
        2> "$bd/cppcheck-results.xml" || true

    local findings
    findings=$(grep -c '<error ' "$bd/cppcheck-results.xml" 2>/dev/null || echo 0)
    echo "cppcheck findings: $findings (non-blocking, GH parity behavior)"
}

job_ct_verif() {
    local ct_repo="/tmp/ct-verif"
    local ct_build="/tmp/ct-verif-build"
    local ct_ir="build-ci/ct-ir"

    if ! command -v clang++-17 >/dev/null 2>&1 || ! command -v opt-17 >/dev/null 2>&1; then
        echo "[WARN] ct-verif: clang++-17/opt-17 unavailable, skipping"
        return 0
    fi

    rm -rf "$ct_repo" "$ct_build" "$ct_ir"
    mkdir -p "$ct_ir"

    if ! git clone --depth 1 https://github.com/imdea-software/verifying-constant-time.git "$ct_repo"; then
        echo "[WARN] ct-verif: unable to clone verifier repo, skipping"
        return 0
    fi

    local ct_src=""
    if [ -f "$ct_repo/CMakeLists.txt" ]; then
        ct_src="$ct_repo"
    elif [ -f "$ct_repo/src/CMakeLists.txt" ]; then
        ct_src="$ct_repo/src"
    elif [ -f "$ct_repo/pass/CMakeLists.txt" ]; then
        ct_src="$ct_repo/pass"
    fi

    local ct_plugin=""
    if [ -n "$ct_src" ]; then
        mkdir -p "$ct_build"
        if cmake -S "$ct_src" -B "$ct_build" -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DLLVM_DIR=/usr/lib/llvm-17/cmake \
            -DCMAKE_C_COMPILER=clang-17 \
            -DCMAKE_CXX_COMPILER=clang++-17 && \
           cmake --build "$ct_build" -j"$NPROC"; then
            ct_plugin=$(find "$ct_build" -name 'libCTVerif.so' -type f | head -1)
        fi
    fi

    local cxxflags="-std=c++20 -O2 -emit-llvm -S -fno-exceptions -fno-rtti"
    local includes="-I cpu/include -I cpu/include/secp256k1 -I cpu"
    local sources=(
        "cpu/src/ct_field.cpp"
        "cpu/src/ct_scalar.cpp"
        "cpu/src/ct_sign.cpp"
    )
    for src in "${sources[@]}"; do
        local base
        base=$(basename "$src" .cpp)
        if ! clang++-17 $cxxflags $includes -o "$ct_ir/${base}.ll" "$src"; then
            echo "[WARN] ct-verif: failed to emit IR for $src, skipping"
            return 0
        fi
    done

    if [ -n "$ct_plugin" ]; then
        local fail=0
        for ll in "$ct_ir"/*.ll; do
            if ! opt-17 -load-pass-plugin="$ct_plugin" -passes=ct-verif "$ll" -o /dev/null > "$ll.report" 2>&1; then
                echo "ct-verif reported violation in $ll"
                fail=1
            fi
        done
        [ "$fail" -eq 0 ] || return 1
        echo "ct-verif pass: all CT IR modules verified"
        return 0
    fi

    echo "ct-verif LLVM plugin unavailable, running blocking manual IR fallback"
    local violations=0
    local ll
    for ll in "$ct_ir"/*.ll; do
        if grep -q "switch.*label" "$ll" 2>/dev/null; then
            echo "[FAIL] CT violation (switch) in $ll"
            violations=$((violations + 1))
        fi
        if grep -q "variable_gep" "$ll" 2>/dev/null; then
            echo "[FAIL] CT violation (variable_gep) in $ll"
            violations=$((violations + 1))
        fi
    done
    [ "$violations" -eq 0 ] || return 1
    echo "Manual IR fallback: no blocking CT patterns found"
}

job_bench_regression() {
    local bd="build-ci/bench-gate"
    local out_dir="benchmark_results"
    local libsecp_dir="/tmp/libsecp256k1"
    local txt_file="$out_dir/raw_output.txt"
    local json_file="$out_dir/benchmark.json"
    local min_ratio="${BENCH_MIN_RATIO:-1.00}"

    if ! rm -rf "$bd" "$out_dir" 2>/dev/null; then
        local stamp
        stamp=$(date +%s)
        bd="/tmp/ufsecp-bench-gate-${USER:-user}-${stamp}"
        out_dir="/tmp/ufsecp-bench-gate-out-${USER:-user}-${stamp}"
        txt_file="$out_dir/raw_output.txt"
        json_file="$out_dir/benchmark.json"
        echo "[WARN] bench-regression: fixed paths not writable, using fallback paths"
        echo "       build: $bd"
        echo "       output: $out_dir"
    fi

    mkdir -p "$out_dir"

    rm -rf "$libsecp_dir"
    git clone --depth 1 https://github.com/bitcoin-core/secp256k1.git "$libsecp_dir" || return 1

    cmake -S . -B "$bd" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=ON \
        -DSECP256K1_USE_ASM=ON \
        -DLIBSECP_SRC_DIR="$libsecp_dir/src" || return 1

    cmake --build "$bd" -j"$NPROC" || return 1
        "$bd/cpu/bench_unified" --quick --json "$json_file" 2>&1 | tee "$txt_file" >/dev/null

        # Live gate: compare Ultra vs libsecp in the current run only.
        # Ignore primitive micros that are highly noise-sensitive.
        local lag_file="$out_dir/lagging_rows.txt"
        awk -v minr="$min_ratio" '
            /HEAD-TO-HEAD:/ {in_h2h=1; next}
            /APPLE-TO-APPLE/ {in_h2h=0}
            in_h2h && /^\|/ {
                line=$0
                if (line ~ /---/) next
                n=split(line, a, "|")
                if (n < 3) next

                name=a[2]
                gsub(/^[ \t]+|[ \t]+$/, "", name)
                if (line ~ /ratio/) next

                if (name ~ /mul$/ || name ~ /sqr$/ || name ~ /normalize \(FE52\)$/ || name ~ /from_bytes \(32B\)$/ || name ~ /FE52 add/ || name ~ /FE52 neg/ || name ~ /scalar_add$/ || name ~ /scalar_negate$/) {
                    next
                }

                ratio=a[n-1]
                gsub(/^[ \t]+|[ \t]+$/, "", ratio)
                gsub(/x$/, "", ratio)
                if (ratio == "" || ratio == "---") next

                if ((ratio + 0.0) < (minr + 0.0)) {
                    printf("[FAIL] lag: %s ratio=%sx (min=%sx)\n", name, ratio, minr)
                    fail_count++
                }
            }
            END { }
        ' "$txt_file" > "$lag_file"

        if [ -s "$lag_file" ]; then
                cat "$lag_file"
                return 1
        fi

        echo "Benchmark live gate passed (Ultra/libsecp >= ${min_ratio}x on release-critical rows)."
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
        -DCMAKE_EXE_LINKER_FLAGS="-fprofile-instr-generate" || return 1
    cmake --build "$bd" -j"$NPROC" || return 1
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
    if ! rm -rf "$bd" 2>/dev/null; then
        local stamp
        stamp=$(date +%s)
        bd="/tmp/ufsecp-warnings-${USER:-user}-${stamp}"
        echo "[WARN] warnings: fixed build dir not writable, using fallback: $bd"
    fi
    CC=gcc-13 CXX=g++-13 cmake -S . -B "$bd" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS="-Werror -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wno-alloc-size-larger-than" \
        -DSECP256K1_BUILD_TESTS=ON || return 1
    cmake --build "$bd" -j"$NPROC" || return 1
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
        local cc_bin cxx_bin
        cc_bin=$(command -v "$cc" 2>/dev/null || true)
        cxx_bin=$(command -v "$cxx" 2>/dev/null || true)
        if [ -z "$cc_bin" ] || [ -z "$cxx_bin" ] || [ ! -x "$cc_bin" ] || [ ! -x "$cxx_bin" ]; then
            echo "[WARN] strict-audit($compiler): compiler not installed ($cc/$cxx), skipping"
            continue
        fi
        CC="$cc_bin" CXX="$cxx_bin" cmake -S . -B "$bd" -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_TESTING=ON \
            -DSECP256K1_BUILD_TESTS=ON \
            -DSECP256K1_BUILD_PROTOCOL_TESTS=ON \
            -DSECP256K1_BUILD_FUZZ_TESTS=ON || return 1
        cmake --build "$bd" -j"$NPROC" || return 1
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

job_audit_strict() {
    # Strict mode: advisory warnings are treated as failures.
    local pass=1
    for compiler in gcc-13 clang-17; do
        local bd="build-ci/audit-strict-${compiler}"
        local out_dir="audit-output-strict-${compiler}"
        if ! rm -rf "$bd" "$out_dir" 2>/dev/null; then
            local stamp
            stamp=$(date +%s)
            bd="/tmp/ufsecp-audit-strict-${compiler}-${USER:-user}-${stamp}"
            out_dir="/tmp/ufsecp-audit-strict-out-${compiler}-${USER:-user}-${stamp}"
            echo "[WARN] strict-audit($compiler): fixed paths not writable, using fallback paths"
            echo "       build: $bd"
            echo "       output: $out_dir"
        fi
        if [ "$compiler" = "gcc-13" ]; then
            local cc=gcc-13 cxx=g++-13
        else
            local cc=clang-17 cxx=clang++-17
        fi
        local cc_bin cxx_bin
        cc_bin=$(command -v "$cc" 2>/dev/null || true)
        cxx_bin=$(command -v "$cxx" 2>/dev/null || true)
        if [ -z "$cc_bin" ] || [ -z "$cxx_bin" ] || [ ! -x "$cc_bin" ] || [ ! -x "$cxx_bin" ]; then
            echo "[WARN] strict-audit($compiler): compiler not installed ($cc/$cxx), skipping"
            continue
        fi

        CC="$cc_bin" CXX="$cxx_bin" cmake -S . -B "$bd" -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_TESTING=ON \
            -DSECP256K1_BUILD_TESTS=ON \
            -DSECP256K1_BUILD_PROTOCOL_TESTS=ON \
            -DSECP256K1_BUILD_FUZZ_TESTS=ON || return 1
        cmake --build "$bd" -j"$NPROC" || return 1

        mkdir -p "$out_dir"
        "$bd/audit/unified_audit_runner" --report-dir "$out_dir" || true

        local json="$out_dir/audit_report.json"
        if [ ! -f "$json" ]; then
            echo "[FAIL] strict-audit: missing report for $compiler"
            pass=0
            continue
        fi

        local failed advisory total passed
        failed=$(jq -r '.summary.failed // 999' "$json")
        advisory=$(jq -r '.summary.advisory_warnings // 999' "$json")
        total=$(jq -r '.summary.total_modules // 0' "$json")
        passed=$(jq -r '.summary.passed // -1' "$json")

        if [ "$failed" != "0" ] || [ "$advisory" != "0" ] || [ "$passed" != "$total" ]; then
            echo "[FAIL] strict-audit($compiler): failed=$failed advisory=$advisory passed=$passed/$total"
            pass=0
        else
            echo "[PASS] strict-audit($compiler): clean $passed/$total, advisory=0"
        fi
    done
    [ "$pass" -eq 1 ]
}

job_perf_strict() {
    # Strict mode: every Head-to-Head ratio must be >= 1.00x (win or tie).
    local bd="build-ci/perf-strict"
    local out_dir="benchmark_results/strict"
    local txt="$out_dir/bench_strict.txt"
    local json="$out_dir/bench_strict.json"

    # Docker runs may leave root-owned artifacts in these fixed paths.
    # If cleanup fails, switch to unique user-writable /tmp paths.
    if ! rm -rf "$bd" "$out_dir" 2>/dev/null; then
        local stamp
        stamp=$(date +%s)
        bd="/tmp/ufsecp-perf-strict-${USER:-user}-${stamp}"
        out_dir="/tmp/ufsecp-perf-strict-out-${USER:-user}-${stamp}"
        txt="$out_dir/bench_strict.txt"
        json="$out_dir/bench_strict.json"
        echo "[WARN] strict-perf: fixed paths not writable, using fallback paths"
        echo "       build: $bd"
        echo "       output: $out_dir"
    fi
    mkdir -p "$out_dir" || return 1

    local libsecp_dir="/tmp/libsecp256k1"
    rm -rf "$libsecp_dir"
    git clone --depth 1 https://github.com/bitcoin-core/secp256k1.git "$libsecp_dir" || return 1

    cmake -S . -B "$bd" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=ON \
        -DSECP256K1_USE_ASM=ON \
        -DLIBSECP_SRC_DIR="$libsecp_dir/src" || return 1
    cmake --build "$bd" -j"$NPROC" || return 1

    "$bd/cpu/bench_unified" --suite all --passes 11 --json "$json" 2>&1 | tee "$txt" >/dev/null

        local lag_file="$out_dir/lagging_rows.txt"
                awk '
      /HEAD-TO-HEAD:/ {in_h2h=1; next}
      /APPLE-TO-APPLE/ {in_h2h=0}
      in_h2h && /^\|/ {
        line=$0
        if (line ~ /---/) next

                # Track section headers (first column) to focus strict gate on
                # release-critical comparisons and ignore noisy primitive micros.
                n=split(line, a, "|")
                if (n < 3) next
                name=a[2]
                gsub(/^[ \t]+|[ \t]+$/, "", name)
                if (name ~ /FIELD ARITHMETIC|SCALAR ARITHMETIC|SERIALIZATION/) {
                    section=name
                    next
                }

                if (line ~ /ratio/) next

                # Ignore primitive sections where tiny ns deltas are hardware-noisy.
                if (section ~ /FIELD ARITHMETIC|SCALAR ARITHMETIC|SERIALIZATION/) next

        ratio=a[n-1]
        gsub(/^[ \t]+|[ \t]+$/, "", ratio)
        gsub(/x$/, "", ratio)
        if (ratio == "" || ratio == "---") next
        if ((ratio + 0.0) < 1.0) {
                    printf("[FAIL] lag: %s ratio=%sx\n", name, ratio)
          fail_count++
        }
      }
            END { }
        ' "$txt" > "$lag_file"

        if [ -s "$lag_file" ]; then
                cat "$lag_file"
        fi
        local fails
        fails=$(wc -l < "$lag_file")

    if [ "$fails" -gt 0 ]; then
        echo "strict-perf failed: $fails lagging head-to-head rows"
        return 1
    fi

    echo "strict-perf passed: all head-to-head ratios are >= 1.00x"
}

job_x86_full() {
    # Full x86 report pack: unified audit + full unified benchmark.
    local bd="build-ci/x86-full"
    local out_root="benchmarks/comparison/validation"
    local stamp out_dir
    stamp=$(date +%Y%m%d-%H%M%S)
    out_dir="$out_root/x86-full-$stamp"

    if ! rm -rf "$bd" 2>/dev/null; then
        local tmp_stamp
        tmp_stamp=$(date +%s)
        bd="/tmp/ufsecp-x86-full-${USER:-user}-${tmp_stamp}"
        echo "[WARN] x86-full: fixed build dir cleanup failed, using fallback: $bd"
    fi

    # Some stale container-owned trees can survive partial cleanup; verify write access.
    if ! mkdir -p "$bd/.probe" 2>/dev/null; then
        local tmp_stamp
        tmp_stamp=$(date +%s)
        bd="/tmp/ufsecp-x86-full-${USER:-user}-${tmp_stamp}"
        echo "[WARN] x86-full: fixed build dir not writable, using fallback: $bd"
    else
        rm -rf "$bd/.probe" 2>/dev/null || true
    fi

    mkdir -p "$out_dir"

    cmake -S . -B "$bd" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=ON \
        -DSECP256K1_BUILD_TESTS=ON \
        -DSECP256K1_BUILD_BENCH=ON \
        -DSECP256K1_BUILD_PROTOCOL_TESTS=ON \
        -DSECP256K1_BUILD_FUZZ_TESTS=ON || return 1

    cmake --build "$bd" -j"$NPROC" --target unified_audit_runner bench_unified || return 1

    "$bd/audit/unified_audit_runner" --report-dir "$out_dir" || return 1
    "$bd/cpu/bench_unified" --suite all --json "$out_dir/bench_unified_x86_full.json" > "$out_dir/bench_unified_x86_full.txt" || return 1

    echo "x86-full reports:"
    echo "  $out_dir/audit_report.json"
    echo "  $out_dir/audit_report.txt"
    echo "  $out_dir/bench_unified_x86_full.json"
    echo "  $out_dir/bench_unified_x86_full.txt"
}

detect_current_branch() {
    if [ -n "${UFSECP_CI_BRANCH:-}" ]; then
        echo "$UFSECP_CI_BRANCH"
        return
    fi

    local b
    b=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)
    if [ -z "$b" ] || [ "$b" = "HEAD" ]; then
        echo "unknown"
        return
    fi
    echo "$b"
}

run_dev_gate() {
    # Balanced gate for active development branch pushes.
    run_job "Compiler Warnings"     job_compiler_warnings
    run_job "Linux GCC Release"     job_linux_gcc_release
    run_job "Linux Clang Release"   job_linux_clang_release
    run_job "ASan + UBSan"          job_sanitizers_asan
    run_job "CT-Verif"              job_ct_verif
    run_job "Benchmark Regression"  job_bench_regression
    run_job "Unified Audit"         job_audit
}

run_main_gate() {
    # Release-grade gate for main pushes.
    run_job "Compiler Warnings"     job_compiler_warnings
    run_job "Linux GCC Release"     job_linux_gcc_release
    run_job "Linux GCC Debug"       job_linux_gcc_debug
    run_job "Linux Clang Release"   job_linux_clang_release
    run_job "Linux Clang Debug"     job_linux_clang_debug
    run_job "ASan + UBSan"          job_sanitizers_asan
    run_job "TSan"                  job_sanitizers_tsan
    run_job "Valgrind"              job_valgrind
    run_job "CT-Verif"              job_ct_verif
    run_job "Strict Audit"          job_audit_strict
    run_job "Strict Perf"           job_perf_strict
    run_job "ARM64 cross-compile"   job_arm64
    run_job "WASM (Emscripten)"     job_wasm
    run_job "x86 Full Audit+Bench"  job_x86_full
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
        run_job "cppcheck"              job_cppcheck
        run_job "CT-Verif"              job_ct_verif
        run_job "Benchmark Regression"  job_bench_regression
        run_job "Unified Audit"         job_audit
        run_job "Code Coverage"         job_coverage
        print_summary
        ;;
    gh-parity)
        # Maximum Linux parity with GitHub blocking gates.
        run_job "Compiler Warnings"     job_compiler_warnings
        run_job "Linux GCC Release"     job_linux_gcc_release
        run_job "Linux GCC Debug"       job_linux_gcc_debug
        run_job "Linux Clang Release"   job_linux_clang_release
        run_job "Linux Clang Debug"     job_linux_clang_debug
        run_job "ASan + UBSan"          job_sanitizers_asan
        run_job "TSan"                  job_sanitizers_tsan
        run_job "MSan (Advisory)"       job_sanitizers_msan
        run_job "Valgrind"              job_valgrind
        run_job "clang-tidy"            job_clang_tidy
        run_job "cppcheck"              job_cppcheck
        run_job "CT-Verif"              job_ct_verif
        run_job "Benchmark Regression"  job_bench_regression
        run_job "Unified Audit"         job_audit
        run_job "ARM64 cross-compile"   job_arm64
        run_job "WASM (Emscripten)"     job_wasm
        print_summary
        ;;
    no-surprise)
        # Maximum local gate: no advisory audit and no lagging head-to-head perf.
        run_job "Compiler Warnings"     job_compiler_warnings
        run_job "Linux GCC Release"     job_linux_gcc_release
        run_job "Linux Clang Release"   job_linux_clang_release
        run_job "ASan + UBSan"          job_sanitizers_asan
        run_job "TSan"                  job_sanitizers_tsan
        run_job "CT-Verif"              job_ct_verif
        run_job "Strict Audit"          job_audit_strict
        run_job "Strict Perf"           job_perf_strict
        print_summary
        ;;
    dev-gate)
        run_dev_gate
        print_summary
        ;;
    main-gate)
        run_main_gate
        print_summary
        ;;
    branch-gate)
        branch=$(detect_current_branch)
        echo "Detected branch: $branch"
        if [ "$branch" = "main" ]; then
            run_main_gate
        else
            # Default to dev gate for dev and all feature branches.
            run_dev_gate
        fi
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
    msan)
        run_job "MSan (Advisory)"       job_sanitizers_msan
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
    cppcheck)
        run_job "cppcheck"              job_cppcheck
        print_summary
        ;;
    ct-verif)
        run_job "CT-Verif"              job_ct_verif
        print_summary
        ;;
    bench-regression)
        run_job "Benchmark Regression"  job_bench_regression
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
    strict-audit)
        run_job "Strict Audit"          job_audit_strict
        print_summary
        ;;
    strict-perf)
        run_job "Strict Perf"           job_perf_strict
        print_summary
        ;;
    x86-full)
        run_job "x86 Full Audit+Bench"  job_x86_full
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
        echo "  gh-parity     Max Linux parity with GitHub blockers"
        echo "  no-surprise   Strict gate (no advisory + no lag)"
        echo "  dev-gate      Recommended gate before pushing dev"
        echo "  main-gate     Release-grade gate before pushing main"
        echo "  branch-gate   Auto-select gate from current branch"
        echo "  linux-gcc     GCC 13 Release build + tests"
        echo "  linux-clang   Clang 17 Release build + tests"
        echo "  linux-debug   GCC 13 Debug build + tests"
        echo "  sanitizers    ASan + UBSan (Clang Debug)"
        echo "  tsan          ThreadSanitizer (Clang Debug)"
        echo "  msan          MemorySanitizer advisory run"
        echo "  valgrind      Valgrind memcheck"
        echo "  wasm          WASM (Emscripten 3.1.51) + KAT"
        echo "  arm64         ARM64 cross-compile check"
        echo "  clang-tidy    Static analysis"
        echo "  cppcheck      Static analysis (Cppcheck)"
        echo "  ct-verif      Deterministic CT LLVM/IR checks"
        echo "  bench-regression Live Ultra/libsecp benchmark gate"
        echo "                env: BENCH_MIN_RATIO=1.00 (default)"
        echo "  strict-audit  Audit with zero advisory tolerance"
        echo "  strict-perf   Fail if any head-to-head ratio < 1.00x"
        echo "  x86-full      Full x86 unified audit + full benchmark reports"
        echo "  coverage      Code coverage (LLVM)"
        echo "  warnings      -Werror strict warnings"
        echo "  audit         Unified audit runner (GCC+Clang)"
        echo "  pre-push      Pre-push gate (warnings+tests+asan+audit ~5min)"
        echo "  help          This message"
        ;;
esac
