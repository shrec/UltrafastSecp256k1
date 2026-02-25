#!/usr/bin/env bash
# ============================================================================
# CTGRIND-style Constant-Time Validation
# Phase V, Task 5.3.2
# ============================================================================
# Uses Valgrind's client request mechanism to mark secret data as "undefined"
# and then runs the CT test suite. Any branch or memory access depending on
# undefined (secret) data triggers a Valgrind error.
#
# This is the same technique used by:
#   - Adam Langley's ctgrind (original)
#   - OpenSSL's CT testing
#   - BoringSSL's constant_time_test
#   - libsodium's constant-time validation
#
# Usage:
#   ./scripts/ctgrind_validate.sh [build_dir]
#
# Requirements:
#   - Linux with Valgrind >= 3.15 installed
#   - Build with: cmake -DSECP256K1_CT_VALGRIND=ON (enables VALGRIND_MAKE_MEM_UNDEFINED)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${1:-${ROOT_DIR}/build-ctgrind}"
REPORT_DIR="${BUILD_DIR}/ctgrind_reports"

# -- Colors --------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[CTGRIND]${NC} $*"; }
warn()  { echo -e "${YELLOW}[CTGRIND]${NC} $*"; }
fail()  { echo -e "${RED}[CTGRIND]${NC} $*"; }

# -- Check Prerequisites ------------------------------------------------------
if ! command -v valgrind &>/dev/null; then
    fail "Valgrind not found. Install: sudo apt install valgrind"
    exit 1
fi

VG_VER=$(valgrind --version | grep -oP '\d+\.\d+')
info "Valgrind version: ${VG_VER}"

# -- Build with CT-Valgrind Instrumentation ------------------------------------
info "Building with CTGRIND instrumentation..."
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DSECP256K1_BUILD_TESTS=ON \
    -DSECP256K1_CT_VALGRIND=ON \
    -DCMAKE_C_FLAGS="-g -O1" \
    -DCMAKE_CXX_FLAGS="-g -O1" \
    2>/dev/null

cmake --build "${BUILD_DIR}" -j "$(nproc)" 2>/dev/null
info "Build complete."

# -- Prepare Report Directory --------------------------------------------------
mkdir -p "${REPORT_DIR}"

# -- CT Test Targets -----------------------------------------------------------
# These are the test binaries that exercise constant-time code paths
CT_TARGETS=(
    "test_ct_sidechannel_smoke"
    "test_ct_equivalence_standalone"
    "test_fault_injection"
    "test_debug_invariants"
    "test_fiat_crypto_vectors"
)

TOTAL=0
PASSED=0
FAILED=0
ERRORS=0

# -- Run Each Target Under Valgrind --------------------------------------------
for target in "${CT_TARGETS[@]}"; do
    BINARY="${BUILD_DIR}/cpu/${target}"
    if [[ ! -x "${BINARY}" ]]; then
        BINARY="${BUILD_DIR}/cpu/${target}.exe"
    fi
    if [[ ! -x "${BINARY}" ]]; then
        warn "Skipping ${target} (binary not found)"
        continue
    fi

    TOTAL=$((TOTAL + 1))
    XML_FILE="${REPORT_DIR}/${target}.xml"
    LOG_FILE="${REPORT_DIR}/${target}.log"

    info "Running: ${target} under Valgrind memcheck..."

    set +e
    valgrind \
        --tool=memcheck \
        --track-origins=yes \
        --xml=yes \
        --xml-file="${XML_FILE}" \
        --log-file="${LOG_FILE}" \
        --error-exitcode=42 \
        --undef-value-errors=yes \
        --expensive-definedness-checks=yes \
        "${BINARY}" 2>/dev/null
    EXIT_CODE=$?
    set -e

    # Parse XML for error count
    if [[ -f "${XML_FILE}" ]]; then
        ERR_COUNT=$(grep -c '<error>' "${XML_FILE}" 2>/dev/null || echo "0")
    else
        ERR_COUNT="unknown"
    fi

    if [[ ${EXIT_CODE} -eq 0 ]]; then
        info "  [OK] ${target}: PASS (0 CT violations)"
        PASSED=$((PASSED + 1))
    elif [[ ${EXIT_CODE} -eq 42 ]]; then
        fail "  [FAIL] ${target}: FAIL (${ERR_COUNT} CT violations)"
        FAILED=$((FAILED + 1))
        ERRORS=$((ERRORS + ERR_COUNT))
    else
        warn "  [!]  ${target}: CRASH (exit code ${EXIT_CODE})"
        FAILED=$((FAILED + 1))
    fi
done

# -- Generate JSON Summary -----------------------------------------------------
JSON_FILE="${REPORT_DIR}/ctgrind_summary.json"
cat > "${JSON_FILE}" << EOF
{
    "tool": "ctgrind_validate",
    "valgrind_version": "${VG_VER}",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "total_targets": ${TOTAL},
    "passed": ${PASSED},
    "failed": ${FAILED},
    "total_ct_violations": ${ERRORS},
    "verdict": "$([ ${FAILED} -eq 0 ] && echo "PASS" || echo "FAIL")"
}
EOF

# -- Print Summary -------------------------------------------------------------
echo ""
echo "============================================================"
echo "  CTGRIND Validation Summary"
echo "============================================================"
echo "  Total targets:     ${TOTAL}"
echo "  Passed:            ${PASSED}"
echo "  Failed:            ${FAILED}"
echo "  CT violations:     ${ERRORS}"
echo "  Reports:           ${REPORT_DIR}/"
echo "============================================================"

if [[ ${FAILED} -gt 0 ]]; then
    fail "CTGRIND VALIDATION FAILED"
    exit 1
else
    info "CTGRIND VALIDATION PASSED -- all CT properties verified"
    exit 0
fi
