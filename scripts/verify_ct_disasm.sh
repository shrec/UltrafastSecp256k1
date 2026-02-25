#!/usr/bin/env bash
# ============================================================================
# CT Disassembly Verification Script
# Phase V, Task 5.1.5 -- Compiler disassembly verification
# ============================================================================
# Scans compiled binaries for secret-dependent branches in CT functions.
# Reports PASS/FAIL per function + total summary.
#
# Usage:
#   ./scripts/verify_ct_disasm.sh <binary> [--arch x86_64|riscv64|aarch64]
#   ./scripts/verify_ct_disasm.sh build/cpu/run_selftest
#   ./scripts/verify_ct_disasm.sh build/cpu/test_ct_sidechannel_standalone --arch riscv64
#
# Exit codes:
#   0 = all CT functions branch-free
#   1 = at least one CT function has suspicious branches
#   2 = usage error
# ============================================================================

set -euo pipefail

# -- Configuration ----------------------------------------------------------

# CT functions that MUST be branch-free (namespace::function patterns)
CT_FUNCTIONS=(
    "ct_compare"
    "ct_compare_detail"
    "ct::is_zero_mask"
    "ct::bool_to_mask"
    "ct::select"
    "ct::negate_if"
    "ct::cswap"
    "ct_cmp_pair"
    "ct_less_than"
)

# -- Argument parsing ------------------------------------------------------

BINARY=""
ARCH=""
OBJDUMP="objdump"
JSON_OUTPUT=""

usage() {
    echo "Usage: $0 <binary> [--arch x86_64|riscv64|aarch64] [--json <output.json>]"
    echo ""
    echo "Scans CT functions in <binary> for secret-dependent branches."
    echo ""
    echo "Options:"
    echo "  --arch ARCH     Target architecture (auto-detected if omitted)"
    echo "  --json FILE     Also produce JSON report"
    echo "  --objdump CMD   Objdump command (default: objdump)"
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch)    ARCH="$2"; shift 2 ;;
        --json)    JSON_OUTPUT="$2"; shift 2 ;;
        --objdump) OBJDUMP="$2"; shift 2 ;;
        --help|-h) usage ;;
        -*)        echo "Unknown option: $1"; usage ;;
        *)         BINARY="$1"; shift ;;
    esac
done

if [[ -z "$BINARY" ]]; then
    echo "ERROR: No binary specified."
    usage
fi

if [[ ! -f "$BINARY" ]]; then
    echo "ERROR: Binary not found: $BINARY"
    exit 2
fi

# -- Auto-detect architecture ----------------------------------------------

if [[ -z "$ARCH" ]]; then
    FILE_INFO=$(file "$BINARY" 2>/dev/null || echo "")
    if echo "$FILE_INFO" | grep -qi "x86-64\|x86_64\|AMD64"; then
        ARCH="x86_64"
    elif echo "$FILE_INFO" | grep -qi "aarch64\|ARM aarch64"; then
        ARCH="aarch64"
    elif echo "$FILE_INFO" | grep -qi "RISC-V\|riscv"; then
        ARCH="riscv64"
    else
        echo "WARNING: Could not auto-detect arch from: $FILE_INFO"
        echo "         Defaulting to x86_64. Use --arch to override."
        ARCH="x86_64"
    fi
fi

echo "==========================================================="
echo "  CT Disassembly Verification"
echo "==========================================================="
echo "  Binary:   $BINARY"
echo "  Arch:     $ARCH"
echo "  Objdump:  $OBJDUMP"
echo "==========================================================="
echo ""

# -- Architecture-specific branch patterns ---------------------------------
# These are CONDITIONAL branch instructions that indicate secret-dependent control flow.
# Unconditional jumps (jmp/j/b) and calls are excluded.

case "$ARCH" in
    x86_64)
        # x86 conditional branches: je, jne, jz, jnz, jg, jge, jl, jle, ja, jae, jb, jbe, etc.
        BRANCH_PATTERN='\bj(e|ne|z|nz|g|ge|l|le|a|ae|b|be|s|ns|o|no|p|np|c|nc|ecxz|rcxz)\b'
        ;;
    aarch64)
        # ARM64 conditional branches: b.eq, b.ne, b.lt, b.gt, b.le, b.ge, cbz, cbnz, tbz, tbnz
        BRANCH_PATTERN='\b(b\.(eq|ne|lt|gt|le|ge|cs|cc|mi|pl|vs|vc|hi|ls|al)|cbz|cbnz|tbz|tbnz)\b'
        ;;
    riscv64)
        # RISC-V conditional branches: beq, bne, blt, bge, bltu, bgeu, beqz, bnez
        BRANCH_PATTERN='\b(beq|bne|blt|bge|bltu|bgeu|beqz|bnez|blez|bgez|bltz|bgtz)\b'
        ;;
    *)
        echo "ERROR: Unsupported arch: $ARCH"
        exit 2
        ;;
esac

# -- Disassemble and analyze -----------------------------------------------

DISASM=$("$OBJDUMP" -d -C "$BINARY" 2>/dev/null) || {
    echo "ERROR: objdump failed on $BINARY"
    exit 2
}

TOTAL_FUNCTIONS=0
PASS_FUNCTIONS=0
FAIL_FUNCTIONS=0
FAIL_LIST=""
JSON_ENTRIES=""

for FUNC in "${CT_FUNCTIONS[@]}"; do
    # Extract function body from disassembly
    # Look for demangled function names containing our pattern
    FUNC_BODY=$(echo "$DISASM" | awk -v pat="$FUNC" '
        /<.*'"$FUNC"'.*>:$/ { found=1; next }
        found && /^$/ { found=0 }
        found && /^[0-9a-f]+ </ { found=0 }
        found { print }
    ')

    if [[ -z "$FUNC_BODY" ]]; then
        echo "  [SKIP] $FUNC -- not found in binary"
        continue
    fi

    TOTAL_FUNCTIONS=$((TOTAL_FUNCTIONS + 1))

    # Count lines in function body
    TOTAL_INSNS=$(echo "$FUNC_BODY" | wc -l)

    # Count conditional branches
    BRANCHES=$(echo "$FUNC_BODY" | grep -cEi "$BRANCH_PATTERN" || true)

    # Count CT-safe instructions (cmov, sltu, csel, etc.)
    case "$ARCH" in
        x86_64)
            SAFE_CT=$(echo "$FUNC_BODY" | grep -cEi '\b(cmov|sbb|adc|setb|sete|setne)\b' || true)
            ;;
        aarch64)
            SAFE_CT=$(echo "$FUNC_BODY" | grep -cEi '\b(csel|csinc|csinv|csneg|ccmp)\b' || true)
            ;;
        riscv64)
            SAFE_CT=$(echo "$FUNC_BODY" | grep -cEi '\b(sltu|slt|sub|xor|or|and)\b' || true)
            ;;
    esac

    if [[ "$BRANCHES" -eq 0 ]]; then
        echo "  [PASS] $FUNC -- 0 branches, $SAFE_CT CT-safe ops ($TOTAL_INSNS insns)"
        PASS_FUNCTIONS=$((PASS_FUNCTIONS + 1))
        STATUS="pass"
    else
        echo "  [FAIL] $FUNC -- $BRANCHES conditional branch(es) found!"
        # Show the offending lines
        echo "$FUNC_BODY" | grep -Ei "$BRANCH_PATTERN" | head -10 | sed 's/^/         /'
        FAIL_FUNCTIONS=$((FAIL_FUNCTIONS + 1))
        FAIL_LIST="$FAIL_LIST $FUNC"
        STATUS="fail"
    fi

    # Accumulate JSON
    if [[ -n "$JSON_OUTPUT" ]]; then
        [[ -n "$JSON_ENTRIES" ]] && JSON_ENTRIES="$JSON_ENTRIES,"
        JSON_ENTRIES="$JSON_ENTRIES
    {\"function\": \"$FUNC\", \"status\": \"$STATUS\", \"branches\": $BRANCHES, \"ct_safe_ops\": $SAFE_CT, \"total_insns\": $TOTAL_INSNS}"
    fi
done

echo ""
echo "-----------------------------------------------------------"
echo "  Summary: $PASS_FUNCTIONS/$TOTAL_FUNCTIONS PASS"
if [[ $FAIL_FUNCTIONS -gt 0 ]]; then
    echo "  FAILED:$FAIL_LIST"
fi
echo "-----------------------------------------------------------"

# -- JSON output -----------------------------------------------------------

if [[ -n "$JSON_OUTPUT" ]]; then
    cat > "$JSON_OUTPUT" <<EOF
{
  "tool": "verify_ct_disasm",
  "binary": "$BINARY",
  "arch": "$ARCH",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "total_functions": $TOTAL_FUNCTIONS,
  "pass": $PASS_FUNCTIONS,
  "fail": $FAIL_FUNCTIONS,
  "functions": [$JSON_ENTRIES
  ]
}
EOF
    echo "  JSON report: $JSON_OUTPUT"
fi

# -- Exit code -------------------------------------------------------------

if [[ $FAIL_FUNCTIONS -gt 0 ]]; then
    exit 1
else
    echo ""
    echo "  OK All CT functions are branch-free on $ARCH"
    exit 0
fi
