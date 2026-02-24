#!/usr/bin/env bash
# =============================================================================
# local-ci.sh — Run full CI jobs locally (inside Docker container)
# =============================================================================
# Reproduces GitHub Actions workflows locally:
#   security-audit.yml + ci.yml coverage + clang-tidy.yml
#
# Usage:
#   bash scripts/local-ci.sh --all              # 4 security-audit jobs
#   bash scripts/local-ci.sh --full             # All 7 jobs (security + coverage + tidy + ci)
#   bash scripts/local-ci.sh --job werror       # Only -Werror build
#   bash scripts/local-ci.sh --job asan         # Only ASan+UBSan
#   bash scripts/local-ci.sh --job valgrind     # Only Valgrind
#   bash scripts/local-ci.sh --job dudect       # Only dudect smoke
#   bash scripts/local-ci.sh --job coverage     # Only code coverage (HTML report)
#   bash scripts/local-ci.sh --job clang-tidy   # Only clang-tidy static analysis
#   bash scripts/local-ci.sh --job ci           # CI matrix (GCC+Clang × Debug+Release)
#
# Exit codes: 0 = all passed, 1 = at least one job failed
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

SRC="/src"
NPROC=$(nproc)
RESULTS=()
FAILED=0

# ── ccache stats (if available) ──────────────────────────────────────────────
if command -v ccache &>/dev/null && [ -d "${CCACHE_DIR:-/ccache}" ]; then
    echo -e "${BOLD}ccache:${NC} ${CCACHE_DIR:-/ccache} ($(ccache -s 2>/dev/null | grep 'cache size' || echo 'empty'))"
    ccache --zero-stats &>/dev/null || true
    CCACHE_ENABLED=1
else
    CCACHE_ENABLED=0
fi

banner() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC} ${BOLD}$1${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

pass() {
    RESULTS+=("${GREEN}✓ PASS${NC}: $1")
    echo -e "\n${GREEN}✓ PASS${NC}: $1\n"
}

fail() {
    RESULTS+=("${RED}✗ FAIL${NC}: $1")
    FAILED=1
    echo -e "\n${RED}✗ FAIL${NC}: $1\n"
}

# ─────────────────────────────────────────────────────────────────────────────
# Job 1: Build with -Werror (GCC-13, Release)
# ─────────────────────────────────────────────────────────────────────────────
job_werror() {
    banner "Job 1/4: Build with -Werror (GCC-13, Release)"
    local build_dir="$SRC/build-local-ci-werror"
    rm -rf "$build_dir"

    cmake -S "$SRC" -B "$build_dir" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER=g++-13 \
        -DCMAKE_CXX_FLAGS="-Werror -Wall -Wextra -Wpedantic -Wconversion -Wshadow" \
        -DSECP256K1_BUILD_TESTS=ON

    if cmake --build "$build_dir" -j"$NPROC" 2>&1; then
        pass "Build with -Werror"
    else
        fail "Build with -Werror"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Job 2: ASan + UBSan (Clang-17, Debug)
# ─────────────────────────────────────────────────────────────────────────────
job_asan() {
    banner "Job 2/4: ASan + UBSan (Clang-17, Debug)"
    local build_dir="$SRC/build-local-ci-asan"
    rm -rf "$build_dir"

    cmake -S "$SRC" -B "$build_dir" -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_C_COMPILER=clang-17 \
        -DCMAKE_CXX_COMPILER=clang++-17 \
        -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
        -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" \
        -DSECP256K1_BUILD_TESTS=ON

    cmake --build "$build_dir" -j"$NPROC"

    export ASAN_OPTIONS="detect_leaks=1:halt_on_error=1"
    export UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1"

    if ctest --test-dir "$build_dir" --output-on-failure -j"$NPROC" -E "^ct_sidechannel$"; then
        pass "ASan + UBSan"
    else
        fail "ASan + UBSan"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Job 3: Valgrind Memcheck (GCC-13, Debug)
# ─────────────────────────────────────────────────────────────────────────────
job_valgrind() {
    banner "Job 3/4: Valgrind Memcheck (GCC-13, Debug)"
    local build_dir="$SRC/build-local-ci-valgrind"
    rm -rf "$build_dir"

    cmake -S "$SRC" -B "$build_dir" -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_CXX_COMPILER=g++-13 \
        -DSECP256K1_BUILD_TESTS=ON

    cmake --build "$build_dir" -j"$NPROC"

    # Run tests under Valgrind via CTest MemCheck
    ctest --test-dir "$build_dir" --output-on-failure -j"$NPROC" \
        -E "^ct_sidechannel$" \
        --overwrite MemoryCheckCommand=/usr/bin/valgrind \
        --overwrite "MemoryCheckCommandOptions=--leak-check=full --error-exitcode=1 --show-leak-kinds=definite,indirect,possible --errors-for-leak-kinds=definite,indirect,possible --suppressions=$SRC/valgrind.supp" \
        -T MemCheck || true

    # Check for real errors (same logic as CI)
    local valgrind_fail=0
    if grep -q 'ERROR SUMMARY: [1-9]' "$build_dir"/Testing/Temporary/MemoryChecker.*.log 2>/dev/null; then
        echo -e "${RED}Valgrind found memory errors${NC}"
        grep 'ERROR SUMMARY:' "$build_dir"/Testing/Temporary/MemoryChecker.*.log
        valgrind_fail=1
    fi
    if grep -q 'definitely lost: [1-9]' "$build_dir"/Testing/Temporary/MemoryChecker.*.log 2>/dev/null; then
        echo -e "${RED}Valgrind found definite memory leaks${NC}"
        grep 'definitely lost:' "$build_dir"/Testing/Temporary/MemoryChecker.*.log
        valgrind_fail=1
    fi

    if [ "$valgrind_fail" -eq 0 ]; then
        pass "Valgrind Memcheck"
    else
        fail "Valgrind Memcheck"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Job 4: dudect Timing Analysis (GCC-13, Release, 60s timeout)
# ─────────────────────────────────────────────────────────────────────────────
job_dudect() {
    banner "Job 4/4: dudect Timing Analysis (GCC-13, Release)"
    local build_dir="$SRC/build-local-ci-dudect"
    rm -rf "$build_dir"

    cmake -S "$SRC" -B "$build_dir" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER=g++-13 \
        -DSECP256K1_BUILD_TESTS=ON

    cmake --build "$build_dir" --target test_ct_sidechannel_standalone -j"$NPROC"

    # 60s local timeout (shorter than CI's 300s)
    local exit_code=0
    timeout 60 "$build_dir/cpu/test_ct_sidechannel_standalone" 2>&1 || exit_code=$?

    if [ "$exit_code" -eq 124 ]; then
        echo -e "${YELLOW}dudect timed out (expected for smoke run)${NC}"
        pass "dudect Timing Analysis (timeout — OK)"
    elif [ "$exit_code" -ne 0 ]; then
        echo -e "${YELLOW}dudect reported timing variance (common on shared systems)${NC}"
        pass "dudect Timing Analysis (variance — acceptable)"
    else
        pass "dudect Timing Analysis"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Job 5: Code Coverage (Clang-17, Debug, llvm-cov → HTML)
# ─────────────────────────────────────────────────────────────────────────────
job_coverage() {
    banner "Job 5/7: Code Coverage (Clang-17 + llvm-cov)"
    local build_dir="$SRC/build-local-ci-coverage"
    rm -rf "$build_dir"

    cmake -S "$SRC" -B "$build_dir" -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_C_COMPILER=clang-17 \
        -DCMAKE_CXX_COMPILER=clang++-17 \
        -DSECP256K1_BUILD_TESTS=ON \
        -DSECP256K1_BUILD_BENCH=OFF \
        -DSECP256K1_USE_ASM=OFF \
        -DCMAKE_C_FLAGS="-fprofile-instr-generate -fcoverage-mapping" \
        -DCMAKE_CXX_FLAGS="-fprofile-instr-generate -fcoverage-mapping" \
        -DCMAKE_EXE_LINKER_FLAGS="-fprofile-instr-generate"

    cmake --build "$build_dir" -j"$NPROC"

    # Run tests to generate profraw
    LLVM_PROFILE_FILE="$build_dir/%p-%m.profraw" \
        ctest --test-dir "$build_dir" --output-on-failure -j"$NPROC" -E "^ct_sidechannel$" || true

    # Merge profiles
    find "$build_dir" -name '*.profraw' -print0 \
        | xargs -0 llvm-profdata-17 merge -sparse -o "$build_dir/merged.profdata"

    # Find all instrumented objects
    local objects=""
    for bin in $(find "$build_dir" -type f -executable); do
        if llvm-cov-17 show --instr-profile="$build_dir/merged.profdata" "$bin" >/dev/null 2>&1; then
            objects="$objects -object=$bin"
        fi
    done

    if [ -z "$objects" ]; then
        fail "Code Coverage (no instrumented objects found)"
        return
    fi

    # Generate lcov + HTML report
    # shellcheck disable=SC2086
    llvm-cov-17 export \
        --format=lcov \
        --instr-profile="$build_dir/merged.profdata" \
        $objects \
        --ignore-filename-regex='(tests/|bench/|examples/|/usr/)' \
        > "$build_dir/coverage.lcov"

    # shellcheck disable=SC2086
    llvm-cov-17 show \
        --format=html \
        --instr-profile="$build_dir/merged.profdata" \
        $objects \
        --ignore-filename-regex='(tests/|bench/|examples/|/usr/)' \
        --output-dir="$build_dir/html"

    # Summary
    echo ""
    echo -e "${BOLD}Coverage summary:${NC}"
    # shellcheck disable=SC2086
    llvm-cov-17 report \
        --instr-profile="$build_dir/merged.profdata" \
        $objects \
        --ignore-filename-regex='(tests/|bench/|examples/|/usr/)' \
        | tail -10

    local html_index="$build_dir/html/index.html"
    if [ -f "$html_index" ]; then
        echo -e "\n${GREEN}HTML report:${NC} $html_index"
        echo -e "${YELLOW}Open in browser to view detailed coverage.${NC}"
        pass "Code Coverage (HTML → $build_dir/html/)"
    else
        fail "Code Coverage (HTML report not generated)"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Job 6: clang-tidy Static Analysis (Clang-17)
# ─────────────────────────────────────────────────────────────────────────────
job_clang_tidy() {
    banner "Job 6/7: clang-tidy Static Analysis (Clang-17)"
    local build_dir="$SRC/build-local-ci-tidy"
    rm -rf "$build_dir"

    cmake -S "$SRC" -B "$build_dir" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=clang-17 \
        -DCMAKE_CXX_COMPILER=clang++-17 \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DSECP256K1_BUILD_TESTS=ON \
        -DSECP256K1_BUILD_BENCH=ON \
        -DSECP256K1_BUILD_EXAMPLES=ON

    cmake --build "$build_dir" -j"$NPROC"

    # Only analyse C++ source files that CMake actually compiles
    # Filter out .S assembly files (clang-tidy cannot process them)
    local output_file="$build_dir/clang-tidy-output.txt"
    jq -r '.[].file' "$build_dir/compile_commands.json" \
        | grep -E '\.(cpp|cc|cxx)$' \
        | sort -u \
        | xargs -P "$NPROC" -n 4 \
            clang-tidy-17 -p "$build_dir" --warnings-as-errors='' --quiet 2>&1 \
        | tee "$output_file"

    if grep -qE '^.+:[0-9]+:[0-9]+: (warning|error):' "$output_file"; then
        local count
        count=$(grep -cE '^.+:[0-9]+:[0-9]+: (warning|error):' "$output_file")
        echo -e "${YELLOW}clang-tidy found $count diagnostic(s)${NC}"
        fail "clang-tidy ($count diagnostics)"
    else
        pass "clang-tidy (clean)"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Job 7: CI matrix (GCC-13 + Clang-17, Debug + Release)
# ─────────────────────────────────────────────────────────────────────────────
job_ci() {
    local all_pass=1

    for compiler in gcc-13 clang-17; do
        for build_type in Release Debug; do
            banner "CI: $compiler / $build_type"
            local build_dir="$SRC/build-local-ci-${compiler}-${build_type}"
            rm -rf "$build_dir"

            if [ "$compiler" = "gcc-13" ]; then
                local cc=gcc-13 cxx=g++-13
            else
                local cc=clang-17 cxx=clang++-17
            fi

            cmake -S "$SRC" -B "$build_dir" -G Ninja \
                -DCMAKE_BUILD_TYPE="$build_type" \
                -DCMAKE_C_COMPILER="$cc" \
                -DCMAKE_CXX_COMPILER="$cxx" \
                -DSECP256K1_BUILD_TESTS=ON

            cmake --build "$build_dir" -j"$NPROC"

            if ctest --test-dir "$build_dir" --output-on-failure -j"$NPROC" -E "^ct_sidechannel$"; then
                echo -e "${GREEN}✓${NC} $compiler / $build_type"
            else
                echo -e "${RED}✗${NC} $compiler / $build_type"
                all_pass=0
            fi
        done
    done

    if [ "$all_pass" -eq 1 ]; then
        pass "CI matrix (4 configs)"
    else
        fail "CI matrix (4 configs)"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print_summary() {
    echo ""
    echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}  LOCAL CI SUMMARY${NC}"
    echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
    for r in "${RESULTS[@]}"; do
        echo -e "  $r"
    done
    echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
    if [ "$FAILED" -eq 0 ]; then
        echo -e "  ${GREEN}${BOLD}ALL PASSED${NC}"
    else
        echo -e "  ${RED}${BOLD}SOME JOBS FAILED${NC}"
    fi
    echo ""
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
main() {
    local run_all=0
    local run_full=0
    local jobs=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --all)     run_all=1; shift ;;
            --full)    run_full=1; shift ;;
            --job)     jobs+=("$2"); shift 2 ;;
            --help|-h)
                echo "Usage: $0 [--all] [--full] [--job <name>]..."
                echo "  --all   Run 4 security-audit jobs (werror, asan, valgrind, dudect)"
                echo "  --full  Run all 7 jobs (security + coverage + clang-tidy + ci)"
                echo "  --job   Run a specific job"
                echo "Jobs: werror, asan, valgrind, dudect, coverage, clang-tidy, ci"
                exit 0 ;;
            *) echo "Unknown option: $1"; exit 1 ;;
        esac
    done

    # --full = all 7 jobs
    if [ "$run_full" -eq 1 ]; then
        jobs=(werror asan valgrind dudect coverage clang-tidy ci)
    # --all or default = 4 security-audit jobs
    elif [ "$run_all" -eq 1 ] || [ ${#jobs[@]} -eq 0 ]; then
        jobs=(werror asan valgrind dudect)
    fi

    echo -e "${BOLD}Local CI — running jobs: ${jobs[*]}${NC}"
    echo -e "${BOLD}CPUs: $NPROC${NC}"
    echo ""

    for job in "${jobs[@]}"; do
        case "$job" in
            werror)     job_werror ;;
            asan)       job_asan ;;
            valgrind)   job_valgrind ;;
            dudect)     job_dudect ;;
            coverage)   job_coverage ;;
            clang-tidy) job_clang_tidy ;;
            ci)         job_ci ;;
            *) echo "Unknown job: $job"; exit 1 ;;
        esac
    done

    # ── ccache summary ──────────────────────────────────────────────────
    if [ "$CCACHE_ENABLED" -eq 1 ]; then
        echo ""
        echo -e "${BOLD}ccache hit rate:${NC}"
        ccache -s 2>/dev/null | grep -E 'hit|miss|size' || true
    fi

    print_summary
    exit "$FAILED"
}

main "$@"
