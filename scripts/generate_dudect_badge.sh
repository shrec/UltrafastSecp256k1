#!/usr/bin/env bash
# ============================================================================
# dudect Status Badge Generator
# Phase IV, Task 4.6.4 -- Parse nightly dudect results, generate badge JSON
# ============================================================================
# Reads dudect output (from nightly.yml artifact or local run) and generates
# a shields.io-compatible badge JSON endpoint.
#
# Usage:
#   ./scripts/generate_dudect_badge.sh <dudect_log> [output_dir]
#
# Output: dudect-badge.json (shields.io endpoint schema)
#   Place in docs/ or gh-pages branch for automatic badge updates.
#
# Badge URL: https://img.shields.io/endpoint?url=<raw_url_to_dudect-badge.json>
# ============================================================================

set -euo pipefail

DUDECT_LOG="${1:-}"
OUTPUT_DIR="${2:-docs}"

if [[ -z "$DUDECT_LOG" ]]; then
    echo "Usage: $0 <dudect_log> [output_dir]"
    echo ""
    echo "If no log file available, run dudect first:"
    echo "  ./build/cpu/test_ct_sidechannel_standalone 2>&1 | tee dudect.log"
    echo "  $0 dudect.log"
    exit 2
fi

if [[ ! -f "$DUDECT_LOG" ]]; then
    echo "ERROR: Log file not found: $DUDECT_LOG"
    exit 2
fi

mkdir -p "$OUTPUT_DIR"

# -- Parse dudect results -------------------------------------------------

# Count PASS/FAIL lines
TOTAL=$(grep -cE '\[(PASS|FAIL)\]' "$DUDECT_LOG" 2>/dev/null || echo "0")
PASSED=$(grep -c '\[PASS\]' "$DUDECT_LOG" 2>/dev/null || echo "0")
FAILED=$(grep -c '\[FAIL\]' "$DUDECT_LOG" 2>/dev/null || echo "0")

# Extract max |t| value
MAX_T=$(grep -oE '\|t\| *= *[0-9]+\.[0-9]+' "$DUDECT_LOG" 2>/dev/null | \
    grep -oE '[0-9]+\.[0-9]+' | sort -rn | head -1 || echo "N/A")

# Determine badge color and message
if [[ "$TOTAL" -eq 0 ]]; then
    COLOR="lightgrey"
    MESSAGE="no data"
    LABEL="dudect"
elif [[ "$FAILED" -eq 0 ]]; then
    COLOR="brightgreen"
    MESSAGE="${PASSED}/${TOTAL} PASS"
    LABEL="dudect CT"
elif [[ "$FAILED" -le 2 ]]; then
    COLOR="yellow"
    MESSAGE="${PASSED}/${TOTAL} (${FAILED} warn)"
    LABEL="dudect CT"
else
    COLOR="red"
    MESSAGE="${PASSED}/${TOTAL} (${FAILED} FAIL)"
    LABEL="dudect CT"
fi

# -- Generate shields.io endpoint JSON ------------------------------------

BADGE_FILE="$OUTPUT_DIR/dudect-badge.json"
cat > "$BADGE_FILE" <<EOF
{
  "schemaVersion": 1,
  "label": "$LABEL",
  "message": "$MESSAGE",
  "color": "$COLOR",
  "namedLogo": "shield",
  "cacheSeconds": 3600
}
EOF

echo "Badge JSON: $BADGE_FILE"
echo "  Status: $MESSAGE ($COLOR)"
echo "  Max |t|: $MAX_T"
echo ""
echo "Usage in README.md:"
echo "  ![dudect](https://img.shields.io/endpoint?url=<raw_url_to_${BADGE_FILE}>)"

# -- Also generate a detailed JSON report ---------------------------------

DETAIL_FILE="$OUTPUT_DIR/dudect-status.json"
cat > "$DETAIL_FILE" <<EOF
{
  "tool": "dudect_badge_generator",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "source_log": "$DUDECT_LOG",
  "total_tests": $TOTAL,
  "passed": $PASSED,
  "failed": $FAILED,
  "max_t_value": "$MAX_T",
  "threshold": 4.5,
  "verdict": "$([ "$FAILED" -eq 0 ] && echo "PASS" || echo "FAIL")"
}
EOF

echo "Detail JSON: $DETAIL_FILE"
