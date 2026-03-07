#!/usr/bin/env bash
# =============================================================================
# Convenience wrapper for local Docker CI with GitHub-parity presets.
# =============================================================================
# Usage:
#   ./scripts/ci-local.sh quick
#   ./scripts/ci-local.sh pre-push
#   ./scripts/ci-local.sh gh-parity
#   ./scripts/ci-local.sh all
#   ./scripts/ci-local.sh dev-gate
#   ./scripts/ci-local.sh main-gate
#   ./scripts/ci-local.sh branch-gate
#   ./scripts/ci-local.sh install-hook
#   ./scripts/ci-local.sh <service-name>
#
# Optional:
#   ./scripts/ci-local.sh --build gh-parity
#   ./scripts/ci-local.sh --list
# =============================================================================
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
compose_file="$repo_root/docker-compose.ci.yml"
run_ci="$repo_root/docker/run_ci.sh"
build_first=0

detect_compose() {
    if docker compose version >/dev/null 2>&1; then
        echo "docker compose"
        return
    fi
    if command -v docker-compose >/dev/null 2>&1; then
        echo "docker-compose"
        return
    fi
    echo "ERROR: neither 'docker compose' nor 'docker-compose' is available." >&2
    exit 1
}

usage() {
    cat <<'EOF'
Local CI wrapper

Commands:
  quick        Fast smoke (GCC Release + WASM)
  pre-push     Pre-push gate
  gh-parity    Max Linux parity with GitHub blockers
    dev-gate     Balanced gate before push to dev
    main-gate    Release-grade gate before push to main
    branch-gate  Auto-select gate by current branch (main=main-gate, else dev-gate)
    install-hook Install git pre-push hook that runs branch-gate
    strict-audit Zero advisory tolerance in unified audit
    strict-perf  Fail on any head-to-head lag (<1.00x)
    no-surprise  Strict end-to-end gate (recommended before release)
        x86-full     Full x86 unified audit + full benchmark reports
  all          Full local CI suite
  <service>    Any service from docker-compose.ci.yml

Options:
  --build      Build ci-base image first
  --list       List available services
  -h, --help   Show this help
EOF
}

if [ $# -eq 0 ]; then
    usage
    exit 1
fi

target=""
while [ $# -gt 0 ]; do
    case "$1" in
        --build)
            build_first=1
            shift
            ;;
        --list)
            target="__list__"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            if [ -n "$target" ]; then
                echo "ERROR: multiple targets provided ('$target' and '$1')." >&2
                exit 1
            fi
            target="$1"
            shift
            ;;
    esac
done

if [ ! -f "$compose_file" ]; then
    echo "ERROR: missing $compose_file" >&2
    exit 1
fi
if [ ! -x "$run_ci" ]; then
    chmod +x "$run_ci"
fi

compose_cmd="$(detect_compose)"

if [ "$target" = "__list__" ]; then
    (cd "$repo_root" && $compose_cmd -f docker-compose.ci.yml config --services)
    exit 0
fi

if [ -z "$target" ]; then
    echo "ERROR: missing target. Use --help." >&2
    exit 1
fi

if [ "$target" = "install-hook" ]; then
    git_dir="$(cd "$repo_root" && git rev-parse --git-dir 2>/dev/null || true)"
    if [ -z "$git_dir" ]; then
        echo "ERROR: not a git repository: $repo_root" >&2
        exit 1
    fi

    if [ "${git_dir#/}" != "$git_dir" ]; then
        hook_path="$git_dir/hooks/pre-push"
    else
        hook_path="$repo_root/$git_dir/hooks/pre-push"
    fi
    mkdir -p "$(dirname "$hook_path")"

    cat > "$hook_path" <<'HOOK'
#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

echo "[pre-push] running local branch gate..."
./scripts/ci-local.sh branch-gate
HOOK

    chmod +x "$hook_path"
    echo "Installed pre-push hook at: $hook_path"
    echo "Hook behavior: runs './scripts/ci-local.sh branch-gate' before every push."
    exit 0
fi

if [ "$build_first" -eq 1 ]; then
    (cd "$repo_root" && $compose_cmd -f docker-compose.ci.yml build ci-base)
fi

(cd "$repo_root" && $compose_cmd -f docker-compose.ci.yml run --rm "$target")
