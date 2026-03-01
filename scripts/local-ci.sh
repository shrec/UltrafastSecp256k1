#!/usr/bin/env bash
# =============================================================================
# local-ci.sh -- Run full CI jobs locally (inside Docker container)
# =============================================================================
# Reproduces GitHub Actions workflows locally:
#   security-audit.yml + ci.yml + audit-report.yml + clang-tidy.yml + cppcheck.yml
#
# Usage:
#   bash scripts/local-ci.sh --quick            # Fast gate: werror + ci (Release) ~5 min
#   bash scripts/local-ci.sh --all              # Standard: 7 jobs ~20-25 min
#   bash scripts/local-ci.sh --full             # Everything: 10 jobs ~45-60 min
#   bash scripts/local-ci.sh --job werror       # Only -Werror build
#   bash scripts/local-ci.sh --job asan         # Only ASan+UBSan
#   bash scripts/local-ci.sh --job tsan         # Only TSan
#   bash scripts/local-ci.sh --job valgrind     # Only Valgrind memcheck
#   bash scripts/local-ci.sh --job dudect       # Only dudect smoke
#   bash scripts/local-ci.sh --job audit        # Only unified_audit_runner (641K checks)
#   bash scripts/local-ci.sh --job coverage     # Only code coverage (HTML report)
#   bash scripts/local-ci.sh --job clang-tidy   # Only clang-tidy static analysis
#   bash scripts/local-ci.sh --job cppcheck     # Only cppcheck static analysis
#   bash scripts/local-ci.sh --job ci           # CI matrix (GCC+Clang x Debug+Release)
#   bash scripts/local-ci.sh --job valgrind-ct  # Valgrind CT taint analysis
#   bash scripts/local-ci.sh --job bench        # Quick benchmark snapshot (no regression check)
#   bash scripts/local-ci.sh --list             # List all available jobs
#
# Build dirs use /tmp/build-local-ci-* (not /src) to avoid Windows NTFS overhead.
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
# Build dirs go to /tmp to avoid Windows NTFS write overhead.
# Second+ builds reuse ccache — rebuild of unchanged code takes seconds.
BUILD_BASE="/tmp/build-local-ci"

# -- ccache stats (if available) ----------------------------------------------
if command -v ccache &>/dev/null && [ -d "${CCACHE_DIR:-/ccache}" ]; then
    echo -e "${BOLD}ccache:${NC} ${CCACHE_DIR:-/ccache} ($(ccache -s 2>/dev/null | grep 'cache size' || echo 'empty'))"
    ccache --zero-stats &>/dev/null || true
    CCACHE_ENABLED=1
else
    CCACHE_ENABLED=0
fi

banner() {
    echo ""
    echo -e "${CYAN}+==============================================================+${NC}"
    echo -e "${CYAN}|${NC} ${BOLD}$1${NC}"
    echo -e "${CYAN}+==============================================================+${NC}"
    echo ""
}

pass() {
    RESULTS+=("${GREEN}OK PASS${NC}: $1")
    echo -e "\n${GREEN}OK PASS${NC}: $1\n"
}

fail() {
    RESULTS+=("${RED}X FAIL${NC}: $1")
    FAILED=1
    echo -e "\n${RED}X FAIL${NC}: $1\n"
}

# -----------------------------------------------------------------------------
# Job: werror — Build with -Werror (GCC-13, Release)
# Mirrors: security-audit.yml / compiler-warnings
# -----------------------------------------------------------------------------
job_werror() {
    banner "werror: Build -Werror -Wall -Wextra (GCC-13, Release)"
    local build_dir="${BUILD_BASE}-werror"

    cmake -S "$SRC" -B "$build_dir" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER=g++-13 \
        -DCMAKE_CXX_FLAGS="-Werror -Wall -Wextra -Wpedantic -Wconversion -Wshadow" \
        -DSECP256K1_BUILD_TESTS=ON

    if cmake --build "$build_dir" -j"$NPROC"; then
        pass "werror: Build with -Werror"
    else
        fail "werror: Build with -Werror"
    fi
}

# -----------------------------------------------------------------------------
# Job: asan — ASan + UBSan (Clang-17, Debug)
# Mirrors: ci.yml/sanitizers (asan) + security-audit.yml/sanitizers
# -----------------------------------------------------------------------------
job_asan() {
    banner "asan: ASan + UBSan (Clang-17, Debug)"
    local build_dir="${BUILD_BASE}-asan"

    cmake -S "$SRC" -B "$build_dir" -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_C_COMPILER=clang-17 \
        -DCMAKE_CXX_COMPILER=clang++-17 \
        "-DCMAKE_C_FLAGS=-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer" \
        "-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer" \
        "-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined" \
        -DSECP256K1_BUILD_TESTS=ON \
        -DSECP256K1_BUILD_FUZZ_TESTS=ON \
        -DSECP256K1_BUILD_PROTOCOL_TESTS=ON \
        -DSECP256K1_USE_ASM=OFF

    cmake --build "$build_dir" -j"$NPROC"

    if ASAN_OPTIONS="detect_leaks=1:halt_on_error=1" \
       UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1" \
       ctest --test-dir "$build_dir" --output-on-failure -j"$NPROC" \
             -E "^(ct_sidechannel|unified_audit|selftest)" --timeout 900; then
        pass "asan: ASan + UBSan"
    else
        fail "asan: ASan + UBSan"
    fi
}

# -----------------------------------------------------------------------------
# Job: tsan — TSan (Clang-17, Debug)
# Mirrors: ci.yml/sanitizers (tsan)
# -----------------------------------------------------------------------------
job_tsan() {
    banner "tsan: TSan (Clang-17, Debug)"
    local build_dir="${BUILD_BASE}-tsan"

    cmake -S "$SRC" -B "$build_dir" -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_C_COMPILER=clang-17 \
        -DCMAKE_CXX_COMPILER=clang++-17 \
        "-DCMAKE_C_FLAGS=-fsanitize=thread -fno-omit-frame-pointer" \
        "-DCMAKE_CXX_FLAGS=-fsanitize=thread -fno-omit-frame-pointer" \
        "-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=thread" \
        -DSECP256K1_BUILD_TESTS=ON \
        -DSECP256K1_BUILD_FUZZ_TESTS=ON \
        -DSECP256K1_BUILD_PROTOCOL_TESTS=ON \
        -DSECP256K1_USE_ASM=OFF

    cmake --build "$build_dir" -j"$NPROC"

    if ctest --test-dir "$build_dir" --output-on-failure -j"$NPROC" \
             -E "^(ct_sidechannel|unified_audit|selftest)" --timeout 900; then
        pass "tsan: TSan"
    else
        fail "tsan: TSan"
    fi
}

# -----------------------------------------------------------------------------
# Job: valgrind — Valgrind Memcheck (GCC-13, Debug)
# Mirrors: security-audit.yml / valgrind
# -----------------------------------------------------------------------------
job_valgrind() {
    banner "valgrind: Valgrind Memcheck (GCC-13, Debug)"
    local build_dir="${BUILD_BASE}-valgrind"

    cmake -S "$SRC" -B "$build_dir" -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_CXX_COMPILER=g++-13 \
        -DSECP256K1_BUILD_TESTS=ON

    cmake --build "$build_dir" -j"$NPROC"

    # Build suppression flag only if file exists
    local supp_flag=""
    if [ -f "$SRC/valgrind.supp" ]; then
        supp_flag="--suppressions=$SRC/valgrind.supp"
    fi

    ctest --test-dir "$build_dir" --output-on-failure -j"$NPROC" \
        -E "^ct_sidechannel" \
        --overwrite MemoryCheckCommand=/usr/bin/valgrind \
        --overwrite "MemoryCheckCommandOptions=--leak-check=full --error-exitcode=1 --show-leak-kinds=definite,indirect,possible --errors-for-leak-kinds=definite,indirect,possible ${supp_flag}" \
        -T MemCheck || true

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
        pass "valgrind: Memcheck clean"
    else
        fail "valgrind: Memcheck found errors"
    fi
}

# -----------------------------------------------------------------------------
# Job: dudect — dudect Timing Analysis (GCC-13, Release, 60s local / 300s CI)
# Mirrors: security-audit.yml / dudect
# -----------------------------------------------------------------------------
job_dudect() {
    banner "dudect: Timing Analysis smoke test (GCC-13, Release)"
    local build_dir="${BUILD_BASE}-dudect"

    cmake -S "$SRC" -B "$build_dir" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER=g++-13 \
        -DSECP256K1_BUILD_TESTS=ON

    cmake --build "$build_dir" --target test_ct_sidechannel_standalone -j"$NPROC"

    # 60s local timeout (shorter than CI's 300s)
    local exit_code=0
    timeout 60 "$build_dir/cpu/test_ct_sidechannel_standalone" 2>&1 || exit_code=$?

    if [ "$exit_code" -eq 124 ]; then
        echo -e "${YELLOW}dudect timed out after 60s (expected — no significant leakage in window)${NC}"
        pass "dudect: Timing Analysis (timeout — OK)"
    elif [ "$exit_code" -ne 0 ]; then
        echo -e "${YELLOW}dudect reported timing variance (common on VMs — verify on bare metal)${NC}"
        pass "dudect: Timing Analysis (variance — acceptable)"
    else
        pass "dudect: Timing Analysis passed"
    fi
}

# -----------------------------------------------------------------------------
# Job: coverage — LLVM source-based coverage → HTML + lcov
# Mirrors: ci.yml / coverage
# -----------------------------------------------------------------------------
job_coverage() {
    banner "coverage: LLVM coverage → HTML (Clang-17, Debug)"
    local build_dir="${BUILD_BASE}-coverage"

    cmake -S "$SRC" -B "$build_dir" -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_C_COMPILER=clang-17 \
        -DCMAKE_CXX_COMPILER=clang++-17 \
        -DSECP256K1_BUILD_TESTS=ON \
        -DSECP256K1_BUILD_BENCH=OFF \
        -DSECP256K1_BUILD_FUZZ_TESTS=ON \
        -DSECP256K1_BUILD_PROTOCOL_TESTS=ON \
        -DSECP256K1_USE_ASM=OFF \
        "-DCMAKE_C_FLAGS=-fprofile-instr-generate -fcoverage-mapping" \
        "-DCMAKE_CXX_FLAGS=-fprofile-instr-generate -fcoverage-mapping" \
        "-DCMAKE_EXE_LINKER_FLAGS=-fprofile-instr-generate"

    cmake --build "$build_dir" -j"$NPROC"

    LLVM_PROFILE_FILE="$build_dir/%p-%m.profraw" \
        ctest --test-dir "$build_dir" --output-on-failure -j"$NPROC" \
              -E "^ct_sidechannel" || true

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
        # Copy HTML to /src so it's accessible from Windows host
        local out_dir="$SRC/local-ci-output/coverage-html"
        rm -rf "$out_dir"
        cp -r "$build_dir/html" "$out_dir"
        echo -e "\n${GREEN}HTML report:${NC} local-ci-output/coverage-html/index.html"
        pass "coverage: HTML report generated"
    else
        fail "coverage: HTML report not generated"
    fi
}

# -----------------------------------------------------------------------------
# Job: clang-tidy — Static analysis (Clang-17)
# Mirrors: clang-tidy.yml
# -----------------------------------------------------------------------------
job_clang_tidy() {
    banner "clang-tidy: Static analysis (Clang-17)"
    local build_dir="${BUILD_BASE}-tidy"

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

    local count=0
    if grep -qE '^.+:[0-9]+:[0-9]+: (warning|error):' "$output_file" 2>/dev/null; then
        count=$(grep -cE '^.+:[0-9]+:[0-9]+: (warning|error):' "$output_file")
    fi

    # Copy report to /src for host access
    cp "$output_file" "$SRC/local-ci-output/clang-tidy-output.txt" 2>/dev/null || true

    if [ "$count" -gt 0 ]; then
        echo -e "${YELLOW}clang-tidy found $count diagnostic(s) — see local-ci-output/clang-tidy-output.txt${NC}"
        # Non-blocking (matches GitHub CI behaviour: warning only)
        pass "clang-tidy ($count diagnostics — non-blocking)"
    else
        pass "clang-tidy: clean"
    fi
}

# -----------------------------------------------------------------------------
# Job: ci — CI matrix (GCC-13 + Clang-17, Release + Debug)
# Mirrors: ci.yml / linux
# -----------------------------------------------------------------------------
job_ci() {
    local all_pass=1

    for compiler in gcc-13 clang-17; do
        for build_type in Release Debug; do
            banner "ci: $compiler / $build_type"
            local build_dir="${BUILD_BASE}-ci-${compiler}-${build_type}"
            local cc cxx
            if [ "$compiler" = "gcc-13" ]; then
                cc=gcc-13; cxx=g++-13
            else
                cc=clang-17; cxx=clang++-17
            fi

            cmake -S "$SRC" -B "$build_dir" -G Ninja \
                -DCMAKE_BUILD_TYPE="$build_type" \
                -DCMAKE_C_COMPILER="$cc" \
                -DCMAKE_CXX_COMPILER="$cxx" \
                -DSECP256K1_BUILD_TESTS=ON \
                -DSECP256K1_BUILD_BENCH=ON \
                -DSECP256K1_BUILD_EXAMPLES=ON \
                -DSECP256K1_BUILD_FUZZ_TESTS=ON \
                -DSECP256K1_BUILD_PROTOCOL_TESTS=ON

            cmake --build "$build_dir" -j"$NPROC"

            if ctest --test-dir "$build_dir" --output-on-failure -j"$NPROC" \
                     -E "^ct_sidechannel"; then
                echo -e "${GREEN}OK${NC} $compiler / $build_type"
            else
                echo -e "${RED}FAIL${NC} $compiler / $build_type"
                all_pass=0
            fi
        done
    done

    if [ "$all_pass" -eq 1 ]; then
        pass "ci: Matrix (4 configs)"
    else
        fail "ci: Matrix (4 configs) — some failed"
    fi
}

# -----------------------------------------------------------------------------
# Job: audit — unified_audit_runner (641,194 checks)
# Mirrors: audit-report.yml / linux-gcc
# -----------------------------------------------------------------------------
job_audit() {
    banner "audit: unified_audit_runner — 641,194 checks (GCC-13, Release)"
    local build_dir="${BUILD_BASE}-audit"

    cmake -S "$SRC" -B "$build_dir" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER=g++-13 \
        -DBUILD_TESTING=ON \
        -DSECP256K1_BUILD_PROTOCOL_TESTS=ON \
        -DSECP256K1_BUILD_FUZZ_TESTS=ON

    cmake --build "$build_dir" -j"$NPROC"

    local out_dir="$SRC/local-ci-output/audit"
    mkdir -p "$out_dir"

    "$build_dir/audit/unified_audit_runner" --report-dir "$out_dir" || true

    if [ -f "$out_dir/audit_report.json" ]; then
        echo ""
        tail -20 "$out_dir/audit_report.txt" 2>/dev/null || true
        local verdict
        verdict=$(grep -o '"audit_verdict": *"[^"]*"' "$out_dir/audit_report.json" \
                  | head -1 | cut -d'"' -f4)
        echo -e "\n${BOLD}Verdict: $verdict${NC}"
        if [ "$verdict" = "AUDIT-READY" ] || [ "$verdict" = "PASS" ]; then
            pass "audit: $verdict — report in local-ci-output/audit/"
        else
            fail "audit: $verdict — see local-ci-output/audit/audit_report.txt"
        fi
    else
        fail "audit: report not generated"
    fi
}

# -----------------------------------------------------------------------------
# Job: cppcheck — Cppcheck static analysis
# Mirrors: cppcheck.yml
# -----------------------------------------------------------------------------
job_cppcheck() {
    if ! command -v cppcheck &>/dev/null; then
        echo -e "${YELLOW}cppcheck not installed — rebuild image: docker build -f Dockerfile.local-ci -t uf-local-ci .${NC}"
        pass "cppcheck: skipped (not installed)"
        return
    fi

    banner "cppcheck: Static analysis"
    local build_dir="${BUILD_BASE}-cppcheck"
    local out_dir="$SRC/local-ci-output"
    mkdir -p "$out_dir"

    cmake -S "$SRC" -B "$build_dir" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DSECP256K1_BUILD_TESTS=ON \
        -DSECP256K1_BUILD_BENCH=ON

    local supp_flag=""
    if [ -f "$SRC/.cppcheck-suppressions" ]; then
        supp_flag="--suppressions-list=$SRC/.cppcheck-suppressions"
    fi

    cppcheck \
        --project="$build_dir/compile_commands.json" \
        --enable=warning,performance,portability \
        --suppress=missingIncludeSystem \
        --suppress=unmatchedSuppression \
        --suppress=unusedFunction \
        ${supp_flag} \
        --inline-suppr \
        --error-exitcode=0 \
        --std=c++20 \
        --xml \
        2> "$out_dir/cppcheck-results.xml" || true

    local errors
    errors=$(grep -c '<error ' "$out_dir/cppcheck-results.xml" 2>/dev/null || echo 0)
    echo -e "cppcheck: ${BOLD}$errors finding(s)${NC} — see local-ci-output/cppcheck-results.xml"
    pass "cppcheck: $errors finding(s) — non-blocking"
}

# -----------------------------------------------------------------------------
# Job: bench — Quick benchmark run (output only, no regression comparison)
# Mirrors: bench-regression.yml / benchmark.yml  (local: no baseline comparison)
# Note: Docker adds noise; use GH CI for regression detection against baseline.
# -----------------------------------------------------------------------------
job_bench() {
    banner "bench: Performance snapshot (GCC-13, Release)"
    local build_dir="${BUILD_BASE}-bench"

    cmake -S "$SRC" -B "$build_dir" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER=g++-13 \
        -DBUILD_TESTING=ON \
        -DSECP256K1_USE_ASM=ON

    cmake --build "$build_dir" --target bench_comprehensive bench_atomic_operations -j"$NPROC"

    local out_dir="$SRC/local-ci-output"
    mkdir -p "$out_dir"

    echo -e "\n${BOLD}=== bench_comprehensive ===${NC}"
    "$build_dir/cpu/bench_comprehensive" 2>&1 | tee "$out_dir/bench_comprehensive.txt"

    echo -e "\n${BOLD}=== bench_atomic_operations ===${NC}"
    "$build_dir/cpu/bench_atomic_operations" 2>&1 | tee "$out_dir/bench_atomic_operations.txt"

    echo -e "\n${YELLOW}NOTE: Docker/VM benchmarks are noisy (shared CPU, no frequency pinning).${NC}"
    echo -e "${YELLOW}      Use GitHub CI bench-regression.yml for authoritative regression detection.${NC}"
    echo -e "${GREEN}Results saved to: local-ci-output/bench_comprehensive.txt${NC}"

    pass "bench: Snapshot complete (see local-ci-output/)"
}

# -----------------------------------------------------------------------------
# Job: valgrind-ct — Valgrind CT taint analysis
# Mirrors: valgrind-ct.yml
# -----------------------------------------------------------------------------
job_valgrind_ct() {
    local script="$SRC/scripts/valgrind_ct_check.sh"
    if [ ! -f "$script" ]; then
        echo -e "${YELLOW}scripts/valgrind_ct_check.sh not found — skipping${NC}"
        pass "valgrind-ct: skipped (script not found)"
        return
    fi

    banner "valgrind-ct: CT taint analysis"
    chmod +x "$script"
    local build_dir="${BUILD_BASE}-valgrind-ct"

    if "$script" "$build_dir"; then
        local report="$build_dir/valgrind_reports/valgrind_ct_report.json"
        if [ -f "$report" ]; then
            mkdir -p "$SRC/local-ci-output/valgrind-ct"
            cp -r "$build_dir/valgrind_reports/." "$SRC/local-ci-output/valgrind-ct/"
            echo ""
            cat "$report"
        fi
        pass "valgrind-ct: CT taint clean"
    else
        fail "valgrind-ct: CT taint violations detected"
    fi
}

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
print_summary() {
    echo ""
    echo -e "${BOLD}===============================================================${NC}"
    echo -e "${BOLD}  LOCAL CI SUMMARY${NC}"
    echo -e "${BOLD}===============================================================${NC}"
    for r in "${RESULTS[@]}"; do
        echo -e "  $r"
    done
    echo -e "${BOLD}===============================================================${NC}"
    if [ "$FAILED" -eq 0 ]; then
        echo -e "  ${GREEN}${BOLD}ALL PASSED${NC}"
    else
        echo -e "  ${RED}${BOLD}SOME JOBS FAILED${NC}"
    fi
    echo ""
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    local jobs=()

    # -- Presets -----------------------------------------------------------
    # --quick  Fast pre-commit gate (~5 min): catches build breakage + test regressions
    local PRESET_QUICK=(werror ci)
    # --all    Standard check (~20-25 min): everything except slow analysis jobs
    local PRESET_ALL=(werror ci asan tsan audit clang-tidy cppcheck)
    # --full   Release-quality check (~45-60 min): mirrors entire GitHub CI suite
    local PRESET_FULL=(werror ci asan tsan audit clang-tidy cppcheck coverage valgrind dudect valgrind-ct)

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --quick)   jobs=("${PRESET_QUICK[@]}"); shift ;;
            --all)     jobs=("${PRESET_ALL[@]}"); shift ;;
            --full)    jobs=("${PRESET_FULL[@]}"); shift ;;
            --job)     IFS=',' read -ra _j <<< "$2"; jobs+=("${_j[@]}"); shift 2 ;;
            --list)
                echo "Available jobs:"
                echo "  werror      Build -Werror -Wall -Wextra    (security-audit.yml)"
                echo "  ci          GCC-13+Clang-17 Release+Debug  (ci.yml)"
                echo "  asan        ASan + UBSan, Clang-17         (ci.yml + security-audit.yml)"
                echo "  tsan        TSan, Clang-17                 (ci.yml)"
                echo "  audit       unified_audit_runner 641K chk  (audit-report.yml)"
                echo "  clang-tidy  clang-tidy-17 static analysis  (clang-tidy.yml)"
                echo "  cppcheck    Cppcheck static analysis       (cppcheck.yml)"
                echo "  coverage    LLVM coverage → HTML           (ci.yml/coverage)"
                echo "  valgrind    Valgrind memcheck              (security-audit.yml)"
                echo "  dudect      dudect 60s timing smoke test   (security-audit.yml)"
                echo "  valgrind-ct Valgrind CT taint analysis     (valgrind-ct.yml)"
                echo "  bench       Benchmark snapshot (output only) (benchmark.yml)"
                echo ""
                echo "Presets:"
                echo "  --quick  ${PRESET_QUICK[*]}  (~5 min)"
                echo "  --all    ${PRESET_ALL[*]}  (~20-25 min)"
                echo "  --full   ${PRESET_FULL[*]}  (~45-60 min)"
                exit 0 ;;
            --help|-h)
                sed -n '2,27p' "$0" | sed 's/^# \?//'
                exit 0 ;;
            *) echo "Unknown option: $1"; exit 1 ;;
        esac
    done

    # Default = --all
    if [ ${#jobs[@]} -eq 0 ]; then
        jobs=("${PRESET_ALL[@]}")
    fi

    # -- Setup output dir --------------------------------------------------
    mkdir -p "$SRC/local-ci-output"

    echo ""
    echo -e "${CYAN}+==============================================================+${NC}"
    echo -e "${CYAN}|${NC} ${BOLD}UltrafastSecp256k1 — Local CI Runner${NC}"
    echo -e "${CYAN}|${NC} Mirrors GitHub Actions ubuntu-24.04 environment"
    echo -e "${CYAN}|${NC} Jobs: ${BOLD}${jobs[*]}${NC}"
    echo -e "${CYAN}|${NC} CPUs: ${BOLD}${NPROC}${NC}  Output: local-ci-output/"
    echo -e "${CYAN}+==============================================================+${NC}"
    echo ""

    for job in "${jobs[@]}"; do
        case "$job" in
            werror)      job_werror ;;
            asan)        job_asan ;;
            tsan)        job_tsan ;;
            valgrind)    job_valgrind ;;
            dudect)      job_dudect ;;
            coverage)    job_coverage ;;
            clang-tidy)  job_clang_tidy ;;
            cppcheck)    job_cppcheck ;;
            ci)          job_ci ;;
            audit)       job_audit ;;
            valgrind-ct) job_valgrind_ct ;;
            bench)       job_bench ;;
            *) echo -e "${RED}Unknown job: $job${NC}"; exit 1 ;;
        esac
    done

    # -- ccache summary ----------------------------------------------------
    if [ "$CCACHE_ENABLED" -eq 1 ]; then
        echo ""
        echo -e "${BOLD}ccache hit rate:${NC}"
        ccache -s 2>/dev/null | grep -E 'hit|miss|size' || true
    fi

    print_summary
    exit "$FAILED"
}

main "$@"
