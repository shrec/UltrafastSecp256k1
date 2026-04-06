#!/usr/bin/env bash
# ===========================================================================
# auditor_kit.sh — UltrafastSecp256k1 Auditor Challenge Kit wrapper
# ===========================================================================
# One-command entry point for an independent security auditor.  This script
# builds the Docker auditor image (if needed) and runs the full audit suite,
# depositing all reports in ./audit_output/.
#
# Usage:
#   bash scripts/auditor_kit.sh [OPTIONS]
#
# Options:
#   --rebuild         Force a full Docker image rebuild (ignores layer cache)
#   --no-docker       Run everything directly on the host (requires all deps)
#   --label LABEL     Only run tests with this CTest label (default: audit)
#   --timeout SECS    Per-test timeout in seconds (default: 300)
#   --output-dir DIR  Write reports to DIR (default: ./audit_output)
#   -h, --help        Show this help and exit
#
# Output artefacts:
#   audit_output/audit_run.log           — full test run log
#   audit_output/audit_results.xml       — JUnit-compatible test results
#   audit_output/audit_ai_findings.json  — confirmed vs AI-suggested tally
#   audit_output/audit_assurance.json    — export_assurance output (if avail)
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults
REBUILD=0
USE_DOCKER=1
LABEL="audit"
TIMEOUT=300
OUTPUT_DIR="$REPO_ROOT/audit_output"
IMAGE_NAME="ufsecp-auditor"

# ---------- argument parsing ------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rebuild)    REBUILD=1 ;;
        --no-docker)  USE_DOCKER=0 ;;
        --label)      LABEL="$2"; shift ;;
        --timeout)    TIMEOUT="$2"; shift ;;
        --output-dir) OUTPUT_DIR="$2"; shift ;;
        -h|--help)
            head -34 "$0" | tail -33
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
    shift
done

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "  UltrafastSecp256k1 Auditor Challenge Kit"
echo "============================================================"
echo "  Repo   : $REPO_ROOT"
echo "  Label  : $LABEL"
echo "  Timeout: ${TIMEOUT}s"
echo "  Output : $OUTPUT_DIR"
echo "============================================================"

# ---------- Docker path -----------------------------------------------------
if [[ $USE_DOCKER -eq 1 ]]; then
    if ! command -v docker &>/dev/null; then
        echo "WARNING: Docker not found. Falling back to --no-docker mode." >&2
        USE_DOCKER=0
    fi
fi

if [[ $USE_DOCKER -eq 1 ]]; then
    # Build image
    if [[ $REBUILD -eq 1 ]] || ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
        echo ">>> Building Docker image '$IMAGE_NAME' ..."
        docker build \
            -f "$REPO_ROOT/Dockerfile.auditor" \
            -t "$IMAGE_NAME" \
            "$REPO_ROOT"
    else
        echo ">>> Using cached Docker image '$IMAGE_NAME' (use --rebuild to force rebuild)"
    fi

    # Run the full audit suite inside the container
    echo ">>> Running audit suite inside Docker container ..."
    docker run --rm \
        -v "$OUTPUT_DIR":/audit_output_host \
        "$IMAGE_NAME" \
        bash -c "
            set -euo pipefail
            ctest --test-dir /build -L \"$LABEL\" --output-on-failure \
                  -j\$(nproc) --timeout $TIMEOUT \
                  --output-junit /audit_results.xml 2>&1 | tee /audit_run.log
            python3 /src/scripts/audit_ai_findings.py \
                    --build-dir /build --json /audit_ai_findings.json -v
            # Copy artefacts to the bind-mounted output directory
            cp -f /audit_run.log       /audit_output_host/ 2>/dev/null || true
            cp -f /audit_results.xml   /audit_output_host/ 2>/dev/null || true
            cp -f /audit_ai_findings.json /audit_output_host/ 2>/dev/null || true
            cp -f /audit_assurance.json   /audit_output_host/ 2>/dev/null || true
        "

else
    # ---------- Host (no-Docker) path ----------------------------------------
    # Locate or create a build directory
    BUILD_DIR=""
    for candidate in "$REPO_ROOT/build_opencl" "$REPO_ROOT/build" "$REPO_ROOT/build_rel"; do
        if [[ -f "$candidate/CTestTestfile.cmake" ]]; then
            BUILD_DIR="$candidate"
            break
        fi
    done

    if [[ -z "$BUILD_DIR" ]]; then
        echo ">>> No existing build found; configuring a new one ..."
        BUILD_DIR="$REPO_ROOT/build_auditor"
        cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -G Ninja \
              -DCMAKE_BUILD_TYPE=RelWithDebInfo \
              -DSECP256K1_BUILD_TESTS=ON \
              -DSECP256K1_BUILD_AUDIT=ON
        cmake --build "$BUILD_DIR" -j"$(nproc)"
    else
        echo ">>> Using existing build at $BUILD_DIR"
    fi

    LOG="$OUTPUT_DIR/audit_run.log"
    XML="$OUTPUT_DIR/audit_results.xml"

    echo ">>> Running ctest (label=$LABEL, timeout=${TIMEOUT}s) ..."
    ctest --test-dir "$BUILD_DIR" \
          -L "$LABEL" \
          --output-on-failure \
          -j"$(nproc)" \
          --timeout "$TIMEOUT" \
          --output-junit "$XML" 2>&1 | tee "$LOG"

    echo ">>> Generating AI findings quarantine report ..."
    python3 "$SCRIPT_DIR/audit_ai_findings.py" \
            --build-dir "$BUILD_DIR" \
            --json "$OUTPUT_DIR/audit_ai_findings.json" \
            -v || true

    # Run assurance export if available
    if [[ -f "$SCRIPT_DIR/export_assurance.py" ]]; then
        echo ">>> Generating assurance report ..."
        python3 "$SCRIPT_DIR/export_assurance.py" \
                -o "$OUTPUT_DIR/audit_assurance.json" 2>/dev/null || true
    fi
fi

echo ""
echo "============================================================"
echo "  Audit Kit Complete"
echo "  Reports written to: $OUTPUT_DIR"
echo "    audit_run.log           — full test output"
echo "    audit_results.xml       — JUnit XML"
echo "    audit_ai_findings.json  — confirmed vs AI-suggested split"
echo "    audit_assurance.json    — assurance matrix (if available)"
echo "============================================================"
