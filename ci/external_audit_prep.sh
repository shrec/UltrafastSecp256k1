#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build-audit"
TRACEABILITY_BUILD_DIR=""
OUTPUT_DIR=""
WITH_PACKAGE=0
SKIP_GRAPH=0
SKIP_TRACEABILITY=0

usage() {
    cat <<'EOF'
external_audit_prep.sh -- Prepare auditor-facing evidence bundle

Usage:
  bash ci/external_audit_prep.sh [options]

Options:
  --build-dir <dir>             Build directory for optional audit package
                                generation. Default: build-audit
  --traceability-build <dir>    Build directory passed to generate_traceability.sh
                                Default: build
  --output-dir <dir>            Output directory. Default: external-audit-prep-<ts>
  --with-package                Also run generate_audit_package.sh and attach the
                                resulting evidence bundle
  --skip-graph                  Skip project graph rebuild
  --skip-traceability           Skip traceability generation
  --help                        Show this message

What it does:
  1. Rebuilds the project graph
  2. Runs preflight hard-fail checks plus advisory coverage/changed-file checks
  3. Validates assurance documentation
  4. Exports machine-readable assurance JSON
  5. Generates traceability artifacts
  6. Optionally builds a full audit evidence package
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --traceability-build)
            TRACEABILITY_BUILD_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --with-package)
            WITH_PACKAGE=1
            shift
            ;;
        --skip-graph)
            SKIP_GRAPH=1
            shift
            ;;
        --skip-traceability)
            SKIP_TRACEABILITY=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"

if [[ -z "$TRACEABILITY_BUILD_DIR" ]]; then
    TRACEABILITY_BUILD_DIR="build"
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$PROJECT_ROOT/external-audit-prep-$TIMESTAMP"
fi

mkdir -p "$OUTPUT_DIR/logs"

run_step() {
    local label="$1"
    shift
    local slug="${label// /_}"
    slug="${slug//\//_}"
    echo "================================================================"
    echo "[external-audit-prep] $label"
    echo "================================================================"
    "$@" 2>&1 | tee "$OUTPUT_DIR/logs/$slug.log"
}

echo "Output directory: $OUTPUT_DIR"

if [[ "$SKIP_GRAPH" -eq 0 ]]; then
    run_step "build project graph" python3 "$PROJECT_ROOT/ci/build_project_graph.py" --rebuild
fi

run_step "preflight security" python3 "$PROJECT_ROOT/ci/preflight.py" --security
run_step "preflight abi" python3 "$PROJECT_ROOT/ci/preflight.py" --abi
run_step "preflight freshness" python3 "$PROJECT_ROOT/ci/preflight.py" --freshness
run_step "preflight drift" python3 "$PROJECT_ROOT/ci/preflight.py" --drift
run_step "preflight coverage advisory" bash -lc "python3 '$PROJECT_ROOT/ci/preflight.py' --coverage || true"
run_step "preflight changed advisory" bash -lc "python3 '$PROJECT_ROOT/ci/preflight.py' --changed || true"

run_step "validate assurance" bash -lc "python3 '$PROJECT_ROOT/ci/validate_assurance.py' --json | tee '$OUTPUT_DIR/validate_assurance.json'"
run_step "export assurance" python3 "$PROJECT_ROOT/ci/export_assurance.py" -o "$OUTPUT_DIR/out/reports/assurance_report.json"
cp "$PROJECT_ROOT/docs/ASSURANCE_CLAIMS.json" "$OUTPUT_DIR/assurance_claims.json"

if [[ "$SKIP_TRACEABILITY" -eq 0 ]]; then
    run_step "generate traceability" bash "$PROJECT_ROOT/ci/generate_traceability.sh" "$PROJECT_ROOT/$TRACEABILITY_BUILD_DIR"
    cp "$PROJECT_ROOT/docs/traceability_report.json" "$OUTPUT_DIR/traceability_report.json"
    cp "$PROJECT_ROOT/docs/traceability_summary.txt" "$OUTPUT_DIR/traceability_summary.txt"
fi

if [[ "$WITH_PACKAGE" -eq 1 ]]; then
    before_latest="$(find "$PROJECT_ROOT" -maxdepth 1 -type d -name 'audit-evidence-*' | sort | tail -1 || true)"
    run_step "generate audit package" bash "$PROJECT_ROOT/ci/generate_audit_package.sh" --build-dir "$BUILD_DIR"
    after_latest="$(find "$PROJECT_ROOT" -maxdepth 1 -type d -name 'audit-evidence-*' | sort | tail -1 || true)"
    if [[ -n "$after_latest" && "$after_latest" != "$before_latest" ]]; then
        mv "$after_latest" "$OUTPUT_DIR/full_audit_package"
    elif [[ -n "$after_latest" ]]; then
        cp -R "$after_latest" "$OUTPUT_DIR/full_audit_package"
    fi
fi

cat > "$OUTPUT_DIR/README.txt" <<EOF
External auditor preparation bundle

Generated: $TIMESTAMP
Project root: $PROJECT_ROOT
Build dir: $BUILD_DIR
Traceability build dir: $TRACEABILITY_BUILD_DIR
Graph rebuilt: $((1 - SKIP_GRAPH))
Traceability generated: $((1 - SKIP_TRACEABILITY))
Full audit package included: $WITH_PACKAGE

Included artifacts:
- out/reports/assurance_report.json
- assurance_claims.json
- validate_assurance.json
- traceability_report.json / traceability_summary.txt (unless skipped)
- logs/
- full_audit_package/ (if requested)

Primary auditor entry points:
- AUDIT_GUIDE.md
- docs/ASSURANCE_LEDGER.md
- docs/AI_AUDIT_PROTOCOL.md
- docs/FORTRESS_ROADMAP.md
- docs/AUDIT_TRACEABILITY.md
EOF

echo "External audit prep complete: $OUTPUT_DIR"