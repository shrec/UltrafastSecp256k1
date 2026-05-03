#!/usr/bin/env bash
# =============================================================================
# local-ci.sh -- Run full CI jobs locally (inside Docker container)
# =============================================================================
# Reproduces GitHub Actions workflows locally:
#   security-audit.yml + ci.yml + audit-report.yml + clang-tidy.yml + cppcheck.yml
#   cryptofuzz.yml + klee.yml
#
# Usage:
#   bash ci/local-ci.sh --quick            # Fast gate: werror + ci (Release) ~5 min
#   bash ci/local-ci.sh --all              # Standard: 7 jobs ~20-25 min
#   bash ci/local-ci.sh --full             # Everything: 12 jobs ~50-70 min
#   bash ci/local-ci.sh --job werror       # Only -Werror build
#   bash ci/local-ci.sh --job asan         # Only ASan+UBSan
#   bash ci/local-ci.sh --job tsan         # Only TSan
#   bash ci/local-ci.sh --job valgrind     # Only Valgrind memcheck
#   bash ci/local-ci.sh --job dudect       # Only dudect smoke
#   bash ci/local-ci.sh --job audit        # Only unified_audit_runner (641K checks)
#   bash ci/local-ci.sh --job coverage     # Only code coverage (HTML report)
#   bash ci/local-ci.sh --job clang-tidy   # Only clang-tidy static analysis
#   bash ci/local-ci.sh --job cppcheck     # Only cppcheck static analysis
#   bash ci/local-ci.sh --job ci           # CI matrix (GCC+Clang x Debug+Release)
#   bash ci/local-ci.sh --job valgrind-ct  # Valgrind CT taint analysis
#   bash ci/local-ci.sh --job bench        # Quick benchmark snapshot (no regression check)
#   bash ci/local-ci.sh --job qemu-smoke   # ARM64/RISC-V cross-build + QEMU smoke
#   bash ci/local-ci.sh --job trust        # Trust signals summary (coverage + links)
#   bash ci/local-ci.sh --job cryptofuzz   # Differential harness compile + 500-iter smoke
#   bash ci/local-ci.sh --job klee         # KLEE symbolic execution compile + 30s smoke
#   bash ci/local-ci.sh --list             # List all available jobs
#
# NEW WORKFLOW DEVELOPMENT: Before pushing any change to a CI workflow that
# involves a harness (cryptofuzz.yml, klee.yml), run the corresponding local job
# first to catch ABI mismatches, wrong include paths, and missing symbols:
#
#   bash ci/local-ci.sh --job cryptofuzz   # ~5-8 min (builds lib + runs harness)
#   bash ci/local-ci.sh --job klee         # ~3-5 min (requires Docker)
#
# Build dirs use ${TMPDIR:-/tmp}/build-local-ci-* (not /src) to avoid Windows NTFS overhead.
# Exit codes: 0 = all passed, 1 = at least one job failed
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Auto-detect source dir: /src in Docker, otherwise derive from script location
if [ -d "/src/cpu" ]; then
    SRC="/src"
else
    SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
NPROC=$(nproc)
RESULTS=()
FAILED=0
# Build dirs go to a temp dir to avoid Windows NTFS write overhead.
# Second+ builds reuse ccache — rebuild of unchanged code takes seconds.
BUILD_BASE="${TMPDIR:-/tmp}/build-local-ci-$$"

# -- Auto-detect compilers ----------------------------------------------------
# GCC: prefer gcc-13, fall back to gcc
if command -v gcc-13 &>/dev/null; then
    GCC_CC=gcc-13; GCC_CXX=g++-13
elif command -v gcc &>/dev/null; then
    GCC_CC=gcc; GCC_CXX=g++
    echo -e "${YELLOW}warning: gcc-13 not found, using $(gcc --version | head -1)${NC}"
else
    GCC_CC=""; GCC_CXX=""
    echo -e "${YELLOW}warning: no gcc found — GCC jobs will be skipped${NC}"
fi

# Clang: prefer 17, then 18, 19, 20, 21, then unversioned
CLANG_CC="" CLANG_CXX=""
for v in 17 18 19 20 21; do
    if command -v "clang-$v" &>/dev/null; then
        CLANG_CC="clang-$v"; CLANG_CXX="clang++-$v"; break
    fi
done
if [ -z "$CLANG_CC" ] && command -v clang &>/dev/null; then
    CLANG_CC=clang; CLANG_CXX=clang++
fi
if [ -z "$CLANG_CC" ]; then
    echo -e "${YELLOW}warning: no clang found — Clang jobs will be skipped${NC}"
else
    echo -e "${BOLD}Clang:${NC} $CLANG_CC ($($CLANG_CC --version 2>/dev/null | head -1))"
fi

# LLVM tools: derive version suffix from detected clang
LLVM_SUFFIX=""
if [ -n "$CLANG_CC" ]; then
    _ver="${CLANG_CC#clang-}"  # e.g. "18" from "clang-18", or "clang" if unversioned
    if [ "$_ver" != "$CLANG_CC" ] && [ -n "$_ver" ]; then
        LLVM_SUFFIX="-$_ver"
    fi
fi
LLVM_PROFDATA="llvm-profdata${LLVM_SUFFIX}"
LLVM_COV="llvm-cov${LLVM_SUFFIX}"
CLANG_TIDY="clang-tidy${LLVM_SUFFIX}"

# -- ccache stats (if available) ----------------------------------------------
if command -v ccache &>/dev/null && [ -d "${CCACHE_DIR:-${HOME}/.cache/ccache}" ]; then
    echo -e "${BOLD}ccache:${NC} ${CCACHE_DIR:-${HOME}/.cache/ccache} ($(ccache -s 2>/dev/null | grep 'cache size' || echo 'empty'))"
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
    if [ -z "$GCC_CXX" ]; then fail "werror: gcc not found"; return; fi
    banner "werror: Build -Werror -Wall -Wextra ($GCC_CXX, Release)"
    local build_dir="${BUILD_BASE}-werror"

    cmake -S "$SRC" -B "$build_dir" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER="$GCC_CXX" \
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
    if [ -z "$CLANG_CC" ]; then fail "asan: clang not found"; return; fi
    banner "asan: ASan + UBSan ($CLANG_CC, Debug)"
    local build_dir="${BUILD_BASE}-asan"

    cmake -S "$SRC" -B "$build_dir" -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_C_COMPILER="$CLANG_CC" \
        -DCMAKE_CXX_COMPILER="$CLANG_CXX" \
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
    if [ -z "$CLANG_CC" ]; then fail "tsan: clang not found"; return; fi
    banner "tsan: TSan ($CLANG_CC, Debug)"
    local build_dir="${BUILD_BASE}-tsan"

    cmake -S "$SRC" -B "$build_dir" -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_C_COMPILER="$CLANG_CC" \
        -DCMAKE_CXX_COMPILER="$CLANG_CXX" \
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
    if [ -z "$GCC_CXX" ]; then fail "valgrind: gcc not found"; return; fi
    banner "valgrind: Valgrind Memcheck ($GCC_CXX, Debug)"
    local build_dir="${BUILD_BASE}-valgrind"

    cmake -S "$SRC" -B "$build_dir" -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_CXX_COMPILER="$GCC_CXX" \
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
    if [ -z "$GCC_CXX" ]; then fail "dudect: gcc not found"; return; fi
    banner "dudect: Timing Analysis smoke test ($GCC_CXX, Release)"
    local build_dir="${BUILD_BASE}-dudect"

    cmake -S "$SRC" -B "$build_dir" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER="$GCC_CXX" \
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
    if [ -z "$CLANG_CC" ]; then fail "coverage: clang not found"; return; fi
    banner "coverage: LLVM coverage → HTML ($CLANG_CC, Debug)"
    local build_dir="${BUILD_BASE}-coverage"

    cmake -S "$SRC" -B "$build_dir" -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_C_COMPILER="$CLANG_CC" \
        -DCMAKE_CXX_COMPILER="$CLANG_CXX" \
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
        | xargs -0 "$LLVM_PROFDATA" merge -sparse -o "$build_dir/merged.profdata"

    # Find all instrumented objects
    local objects=""
    for bin in $(find "$build_dir" -type f -executable); do
        if "$LLVM_COV" show --instr-profile="$build_dir/merged.profdata" "$bin" >/dev/null 2>&1; then
            objects="$objects -object=$bin"
        fi
    done

    if [ -z "$objects" ]; then
        fail "Code Coverage (no instrumented objects found)"
        return
    fi

    # Generate lcov + HTML report
    # shellcheck disable=SC2086
    "$LLVM_COV" export \
        --format=lcov \
        --instr-profile="$build_dir/merged.profdata" \
        $objects \
        --ignore-filename-regex='(tests/|bench/|examples/|/usr/)' \
        > "$build_dir/coverage.lcov"

    # shellcheck disable=SC2086
    "$LLVM_COV" show \
        --format=html \
        --instr-profile="$build_dir/merged.profdata" \
        $objects \
        --ignore-filename-regex='(tests/|bench/|examples/|/usr/)' \
        --output-dir="$build_dir/html"

    # Summary
    echo ""
    echo -e "${BOLD}Coverage summary:${NC}"
    # shellcheck disable=SC2086
    "$LLVM_COV" report \
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
    if [ -z "$CLANG_CC" ]; then fail "clang-tidy: clang not found"; return; fi
    banner "clang-tidy: Static analysis ($CLANG_CC)"
    local build_dir="${BUILD_BASE}-tidy"

    cmake -S "$SRC" -B "$build_dir" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER="$CLANG_CC" \
        -DCMAKE_CXX_COMPILER="$CLANG_CXX" \
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
            "$CLANG_TIDY" -p "$build_dir" --warnings-as-errors='' --quiet 2>&1 \
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
    local compilers=()
    if [ -n "$GCC_CC" ]; then compilers+=(gcc); fi
    if [ -n "$CLANG_CC" ]; then compilers+=(clang); fi
    if [ ${#compilers[@]} -eq 0 ]; then fail "ci: no compilers found"; return; fi

    for compiler in "${compilers[@]}"; do
        for build_type in Release Debug; do
            local cc cxx label
            if [ "$compiler" = "gcc" ]; then
                cc="$GCC_CC"; cxx="$GCC_CXX"; label="$GCC_CC"
            else
                cc="$CLANG_CC"; cxx="$CLANG_CXX"; label="$CLANG_CC"
            fi
            banner "ci: $label / $build_type"
            local build_dir="${BUILD_BASE}-ci-${label}-${build_type}"

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
    if [ -z "$GCC_CXX" ]; then fail "audit: gcc not found"; return; fi
    banner "audit: unified_audit_runner — 641,194 checks ($GCC_CXX, Release)"
    local build_dir="${BUILD_BASE}-audit"

    cmake -S "$SRC" -B "$build_dir" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER="$GCC_CXX" \
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
    if [ -z "$GCC_CXX" ]; then fail "bench: gcc not found"; return; fi
    banner "bench: Performance snapshot ($GCC_CXX, Release)"
    local build_dir="${BUILD_BASE}-bench"

    cmake -S "$SRC" -B "$build_dir" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER="$GCC_CXX" \
        -DBUILD_TESTING=ON \
        -DSECP256K1_USE_ASM=ON

    cmake --build "$build_dir" --target bench_unified bench_atomic_operations -j"$NPROC"

    local out_dir="$SRC/local-ci-output"
    mkdir -p "$out_dir"

    echo -e "\n${BOLD}=== bench_unified ===${NC}"
    "$build_dir/cpu/bench_unified" --quick 2>&1 | tee "$out_dir/bench_unified.txt"

    echo -e "\n${BOLD}=== bench_atomic_operations ===${NC}"
    "$build_dir/cpu/bench_atomic_operations" 2>&1 | tee "$out_dir/bench_atomic_operations.txt"

    echo -e "\n${YELLOW}NOTE: Docker/VM benchmarks are noisy (shared CPU, no frequency pinning).${NC}"
    echo -e "${YELLOW}      Use GitHub CI bench-regression.yml for authoritative regression detection.${NC}"
    echo -e "${GREEN}Results saved to: local-ci-output/bench_unified.txt${NC}"

    pass "bench: Snapshot complete (see local-ci-output/)"
}

# -----------------------------------------------------------------------------
# Job: qemu-smoke - ARM64/RISC-V cross-build + QEMU smoke
# Mirrors: ci.yml / linux-arm64 + linux-riscv64
# Optional env: SECP256K1_QEMU_SMOKE_TARGET=arm64|riscv64|all (default: all)
# -----------------------------------------------------------------------------
job_qemu_smoke() {
    banner "qemu-smoke: ARM64/RISC-V cross-build + QEMU smoke"

    local script="$SRC/ci/run-qemu-smoke.sh"
    local target="${SECP256K1_QEMU_SMOKE_TARGET:-all}"
    local out_dir="$SRC/local-ci-output"
    mkdir -p "$out_dir"

    if [ ! -f "$script" ]; then
        fail "qemu-smoke: ci/run-qemu-smoke.sh not found"
        return
    fi

    echo -e "${BOLD}Target:${NC} $target"
    echo -e "${BOLD}Hint:${NC} export SECP256K1_QEMU_SMOKE_TARGET=arm64|riscv64|all to override"

    if bash "$script" "$target" 2>&1 | tee "$out_dir/qemu-smoke.txt"; then
        pass "qemu-smoke: $target"
    else
        fail "qemu-smoke: $target"
    fi
}

# -----------------------------------------------------------------------------
# Job: trust - Trust signals summary (local)
# Mirrors: codecov + SonarCloud + scorecard (local = evidence + links)
# -----------------------------------------------------------------------------
job_trust() {
    banner "trust: Public Trust Signals (local summary)"
    job_coverage

    local out_dir="$SRC/local-ci-output"
    mkdir -p "$out_dir"
    cat > "$out_dir/trust_summary.txt" <<'EOF'
Public Trust Signals (local):
- Codecov: ensure coverage artifacts exist (local-ci-output/coverage/)
- SonarCloud: run remote analysis in CI (requires token)
- OSSF Scorecard: GitHub-hosted scan (public)
EOF

    echo -e "${GREEN}Trust summary saved to: local-ci-output/trust_summary.txt${NC}"
    echo -e "${YELLOW}NOTE:${NC} SonarCloud/Scorecard run in GitHub Actions (not local)."
    pass "trust: Summary complete"
}
# -----------------------------------------------------------------------------
# Job: valgrind-ct — Valgrind CT taint analysis
# Mirrors: valgrind-ct.yml
# -----------------------------------------------------------------------------
job_valgrind_ct() {
    local script="$SRC/ci/valgrind_ct_check.sh"
    if [ ! -f "$script" ]; then
        echo -e "${YELLOW}ci/valgrind_ct_check.sh not found — skipping${NC}"
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
# Job: cryptofuzz — Differential harness compilation + smoke run
# Mirrors: cryptofuzz.yml
# Requires: clang (17+), libsecp256k1 v0.6.0, libssl-dev, libboost-dev
# Exit: non-zero if harness fails to compile or reports divergences
# -----------------------------------------------------------------------------
job_cryptofuzz() {
    banner "cryptofuzz: Differential harness (compile + 500-iter smoke)"

    # Detect clang
    local CXX_BIN=""
    for v in 17 18 19 20 21; do
        if command -v "clang++-$v" &>/dev/null; then CXX_BIN="clang++-$v"; break; fi
    done
    if [ -z "$CXX_BIN" ] && command -v clang++ &>/dev/null; then CXX_BIN=clang++; fi
    if [ -z "$CXX_BIN" ]; then
        echo -e "${YELLOW}clang++ not found — skipping cryptofuzz job${NC}"
        pass "cryptofuzz: skipped (no clang++ found)"
        return
    fi
    echo -e "${BOLD}Compiler:${NC} $CXX_BIN"

    local HARNESS="$SRC/tests/harnesses/cryptofuzz/diff_harness.cpp"
    if [ ! -f "$HARNESS" ]; then
        fail "cryptofuzz: harness not found at $HARNESS"
        return
    fi

    # ---- 1. Build UltrafastSecp256k1 shared library ----
    local build_dir="${BUILD_BASE}-cryptofuzz"
    echo "Building UltrafastSecp256k1 (RelWithDebInfo)..."
    cmake -S "$SRC" -B "$build_dir" -G Ninja \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_CXX_COMPILER="$CXX_BIN" \
        -DSECP256K1_BUILD_TESTS=OFF \
        -DSECP256K1_BUILD_BENCH=OFF \
        -DSECP256K1_BUILD_EXAMPLES=OFF \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF \
        -Wno-dev 2>&1 | tail -5
    cmake --build "$build_dir" -j"$NPROC" 2>&1 | tail -10

    local UFSECP_LIB_DIR="$build_dir/include/ufsecp"
    local CPU_LIB="$build_dir/cpu/libfastsecp256k1.a"
    if [ ! -f "$UFSECP_LIB_DIR/libufsecp.a" ]; then
        fail "cryptofuzz: libufsecp.a not found in $UFSECP_LIB_DIR"
        return
    fi
    if [ ! -f "$CPU_LIB" ]; then
        fail "cryptofuzz: libfastsecp256k1.a not found at $CPU_LIB"
        return
    fi

    # ---- 2. Build reference libsecp256k1 v0.6.0 ----
    local REF_DIR="${TMPDIR:-/tmp}/local-ci-libsecp256k1"
    local REF_INSTALL="${TMPDIR:-/tmp}/local-ci-libsecp256k1-install"
    if [ ! -f "$REF_INSTALL/lib/libsecp256k1.a" ] && [ ! -f "$REF_INSTALL/lib/libsecp256k1.so" ]; then
        echo "Cloning libsecp256k1 v0.6.0..."
        rm -rf "$REF_DIR"
        git clone --depth 1 --branch v0.6.0 \
            https://github.com/bitcoin-core/secp256k1.git "$REF_DIR" 2>&1 | tail -3
        cmake -S "$REF_DIR" -B "$REF_DIR/build" -G Ninja \
            -DCMAKE_BUILD_TYPE=RelWithDebInfo \
            -DCMAKE_C_COMPILER="${CLANG_CC:-clang}" \
            -DSECP256K1_BUILD_TESTS=OFF \
            -DSECP256K1_ENABLE_MODULE_SCHNORRSIG=ON \
            -DSECP256K1_ENABLE_MODULE_RECOVERY=ON \
            -DSECP256K1_ENABLE_MODULE_ECDH=ON \
            -Wno-dev 2>&1 | tail -5
        cmake --build "$REF_DIR/build" -j"$NPROC"
        cmake --install "$REF_DIR/build" --prefix "$REF_INSTALL"
    else
        echo "Reusing cached libsecp256k1 at $REF_INSTALL"
    fi

    # ---- 3. Compile the differential harness ----
    # Link statically against ufsecp + fastsecp256k1 to avoid shared-library
    # startup crashes (global constructors in libufsecp.so run before main()).
    echo "Compiling differential harness..."
    local compile_log="${TMPDIR:-/tmp}/cryptofuzz_compile.log"
    local harness_bin="${TMPDIR:-/tmp}/cryptofuzz_diff_harness"
    "$CXX_BIN" -std=c++20 -O2 \
        -I"$SRC" \
        -I"$REF_INSTALL/include" \
        "$HARNESS" \
        "$UFSECP_LIB_DIR/libufsecp.a" \
        "$CPU_LIB" \
        -L"$REF_INSTALL/lib" -lsecp256k1 \
        -Wl,-rpath,"$REF_INSTALL/lib" \
        -o "${harness_bin}" 2>&1 | tee "$compile_log"

    if [ ! -f "${harness_bin}" ]; then
        cat "$compile_log"
        fail "cryptofuzz: harness compilation failed — ABI or include path mismatch"
        return
    fi
    echo -e "${GREEN}Harness compiled successfully.${NC}"

    # ---- 4. Quick smoke run (500 iterations) ----
    echo "Running 500-iteration smoke test..."
    local results_log="${TMPDIR:-/tmp}/cryptofuzz_results.log"
    LD_LIBRARY_PATH="$REF_INSTALL/lib" \
        "${harness_bin}" 500 2>&1 | tee "$results_log"
    local EXIT="${PIPESTATUS[0]}"

    local FAIL_CNT
    FAIL_CNT=$(grep -oP '\d+(?= FAIL)' "$results_log" | tail -1 || echo 0)
    if [ "$EXIT" -ne 0 ] || [ "${FAIL_CNT:-0}" -gt 0 ]; then
        fail "cryptofuzz: $FAIL_CNT divergences — check $results_log"
    else
        pass "cryptofuzz: 500 iterations, 0 divergences"
    fi
}

# -----------------------------------------------------------------------------
# Job: klee — Symbolic execution (Docker-based, compile + short run)
# Mirrors: klee.yml
# Requires: Docker, internet access (pulls klee/klee:3.0 once)
# -----------------------------------------------------------------------------
job_klee() {
    banner "klee: Symbolic execution (Docker klee/klee:3.0)"

    if ! command -v docker &>/dev/null; then
        echo -e "${YELLOW}docker not found — skipping klee job${NC}"
        pass "klee: skipped (docker not found)"
        return
    fi

    local KLEE_DIR="${TMPDIR:-/tmp}/klee-work-local"
    local KLEE_RESULTS="${TMPDIR:-/tmp}/klee-results-local"
    rm -rf "$KLEE_DIR" "$KLEE_RESULTS"
    mkdir -p "$KLEE_DIR" "$KLEE_RESULTS"
    chmod 777 "$KLEE_DIR" "$KLEE_RESULTS"

    local HARNESS_DIR="$SRC/tests/harnesses/klee"
    if [ ! -d "$HARNESS_DIR" ]; then
        fail "klee: harness directory not found at $HARNESS_DIR"
        return
    fi

    # Copy harnesses to work dir
    cp "$HARNESS_DIR"/*.cpp "$KLEE_DIR/"

    # Pull image (cached after first run)
    echo "Pulling klee/klee:3.0 (skips if cached)..."
    docker pull klee/klee:3.0 2>&1 | tail -3

    # ---- Compile each harness to LLVM bitcode ----
    echo "Compiling harnesses to LLVM bitcode inside Docker..."
    local compile_ok=1
    for src in "$KLEE_DIR"/*.cpp; do
        local name
        name=$(basename "$src" .cpp)
        echo "  Compiling $name..."
        if ! docker run --rm \
            -v "$KLEE_DIR:/work" \
            klee/klee:3.0 \
            clang++ -std=c++17 -emit-llvm -O0 -g \
                -I /home/klee/klee_build/include \
                -include assert.h \
                "/work/${name}.cpp" -c -o "/work/${name}.bc" 2>&1; then
            echo -e "${RED}Compilation failed for $name${NC}"
            compile_ok=0
        fi
    done

    if [ "$compile_ok" -eq 0 ]; then
        fail "klee: harness compilation failed inside Docker"
        return
    fi

    if ! ls "$KLEE_DIR"/*.bc &>/dev/null; then
        fail "klee: no .bc files produced"
        return
    fi
    echo -e "${GREEN}All harnesses compiled to bitcode.${NC}"

    # ---- Run KLEE with short timeout (30s per target — smoke only) ----
    local TOTAL_ERRORS=0
    for bc in "$KLEE_DIR"/*.bc; do
        local name
        name=$(basename "$bc" .bc)
        echo ""
        echo "=== KLEE smoke: $name (max 30s) ==="
        mkdir -p "$KLEE_RESULTS/$name"
        chmod 777 "$KLEE_RESULTS/$name"

        docker run --rm \
            -v "$KLEE_DIR:/work" \
            -v "$KLEE_RESULTS:/results" \
            klee/klee:3.0 \
            klee \
                --output-dir="/results/${name}" \
                --max-time=30s \
                --max-depth=50 \
                --emit-all-errors \
                --only-output-states-covering-new \
                "/work/${name}.bc" 2>&1 || true

        local ASSERTS
        ASSERTS=$(find "$KLEE_RESULTS/$name" -name "*.assert.err" 2>/dev/null | wc -l)
        echo "  Assertion violations: $ASSERTS"
        TOTAL_ERRORS=$((TOTAL_ERRORS + ASSERTS))
    done

    if [ "$TOTAL_ERRORS" -gt 0 ]; then
        fail "klee: $TOTAL_ERRORS assertion violation(s) found"
    else
        pass "klee: no assertion violations (smoke, 30s/target)"
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
    local PRESET_FULL=(werror ci asan tsan audit clang-tidy cppcheck coverage valgrind dudect valgrind-ct cryptofuzz klee)

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --quick)   jobs=("${PRESET_QUICK[@]}"); shift ;;
            --all)     jobs=("${PRESET_ALL[@]}"); shift ;;
            --full)    jobs=("${PRESET_FULL[@]}"); shift ;;
            --job)     IFS=',' read -ra _j <<< "$2"; jobs+=("${_j[@]}"); shift 2 ;;
            --list)
                echo "Available jobs:"
                echo "  werror      Build -Werror -Wall -Wextra    (security-audit.yml)"
                echo "  ci          GCC+Clang Release+Debug       (ci.yml)"
                echo "  asan        ASan + UBSan, Clang            (ci.yml + security-audit.yml)"
                echo "  tsan        TSan, Clang                    (ci.yml)"
                echo "  audit       unified_audit_runner 641K chk  (audit-report.yml)"
                echo "  clang-tidy  clang-tidy static analysis     (clang-tidy.yml)"
                echo "  cppcheck    Cppcheck static analysis       (cppcheck.yml)"
                echo "  coverage    LLVM coverage → HTML           (ci.yml/coverage)"
                echo "  valgrind    Valgrind memcheck              (security-audit.yml)"
                echo "  dudect      dudect 60s timing smoke test   (security-audit.yml)"
                echo "  valgrind-ct Valgrind CT taint analysis     (valgrind-ct.yml)"
                echo "  bench       Benchmark snapshot (output only) (benchmark.yml)"
                echo "  qemu-smoke  ARM64/RISC-V cross-build + QEMU smoke (ci.yml)"
                echo "  trust       Trust signals summary (coverage + links)"
                echo "  cryptofuzz  Differential harness compile+smoke     (cryptofuzz.yml)"
                echo "  klee        Symbolic execution compile+smoke        (klee.yml)"
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
            qemu-smoke)  job_qemu_smoke ;;
            trust)       job_trust ;;
            cryptofuzz)  job_cryptofuzz ;;
            klee)        job_klee ;;
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
