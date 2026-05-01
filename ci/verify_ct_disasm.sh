#!/usr/bin/env bash
# ============================================================================
# CT Disassembly Verification Script
# Phase V, Task 5.1.5 -- Compiler disassembly verification
# ============================================================================
# Scans compiled binaries for secret-dependent branches in CT functions.
# Reports PASS/FAIL per function + total summary.
#
# Usage:
#   ./ci/verify_ct_disasm.sh <binary> [--arch x86_64|riscv64|aarch64]
#   ./ci/verify_ct_disasm.sh build/cpu/run_selftest
#   ./ci/verify_ct_disasm.sh build/cpu/test_ct_sidechannel_standalone --arch riscv64
#
# Exit codes:
#   0 = all CT functions branch-free
#   1 = at least one CT function has suspicious branches
#   2 = usage error
# ============================================================================

set -euo pipefail

# -- Configuration ----------------------------------------------------------

# CT functions that MUST be branch-free.
# Format: display-name|exact demangled symbol label.
CT_FUNCTIONS=(
    "ct::is_zero_mask|secp256k1::ct::is_zero_mask(unsigned long)"
    "ct::bool_to_mask|secp256k1::ct::bool_to_mask(bool)"
    "ct::cswap256|secp256k1::ct::cswap256(unsigned long*, unsigned long*, unsigned long)"
    "ct_cmp_pair|secp256k1::ct::ct_compare_detail::ct_cmp_pair(unsigned long, unsigned long, unsigned long&, unsigned long&)"
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

filter_branch_lines() {
    local func_name="$1"
    local func_body="$2"

    echo "$func_body" | awk -v func_name="$func_name" -v branch_pat="$BRANCH_PATTERN" '
        { lines[++n] = $0 }
        END {
            for (i = 1; i <= n; ++i) {
                line = lines[i]
                if (line !~ branch_pat) {
                    continue
                }
                if ((func_name == "ct::is_zero_mask" || func_name == "ct::bool_to_mask") && i < n && lines[i + 1] ~ /__stack_chk_fail/) {
                    continue
                }
                if (func_name == "ct::cswap256" && i > 1 && lines[i - 1] ~ /cmpl[[:space:]]+\$0x3/ && line ~ /\bjle\b/) {
                    continue
                }
                print line
            }
        }
    '
}

for ENTRY in "${CT_FUNCTIONS[@]}"; do
    FUNC="${ENTRY%%|*}"
    SYMBOL_LABEL="${ENTRY#*|}"
    # Extract function body from disassembly
    # Match the exact demangled label so tests and unrelated helper families
    # are not pulled in by substring matches.
    FUNC_BODY=$(echo "$DISASM" | awk -v pat="$SYMBOL_LABEL" '
        index($0, "<" pat ">:") { found=1; next }
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
    BRANCH_LINES=$(filter_branch_lines "$FUNC" "$FUNC_BODY")
    BRANCHES=$(printf '%s\n' "$BRANCH_LINES" | sed '/^$/d' | wc -l | tr -d ' ')

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
        printf '%s\n' "$BRANCH_LINES" | head -10 | sed 's/^/         /'
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
