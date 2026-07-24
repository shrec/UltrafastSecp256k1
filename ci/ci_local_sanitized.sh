#!/bin/bash
# ============================================================================
# ci_local_sanitized.sh — hardened, defense-in-depth entrypoint for local /
# pre-push CI. It reduces the process to a minimal, trusted environment,
# drops every inherited shell-startup / function and Git routing / config
# hook, and then delegates the caller's UNCHANGED arguments to the tracked
# sibling ci/ci_local.sh.
#
# SUPPORTED SECURITY ENTRYPOINT (the only one):
#
#     /bin/bash -p ci/ci_local_sanitized.sh [ci_local.sh args...]
#
# Always launch it through `/bin/bash -p`: privileged mode makes bash ignore
# $ENV / $BASH_ENV and refuse to silently drop privileges. This wrapper is
# intentionally NOT relied upon as a directly-executed program — no reliance
# on the executable bit or on the shebang line being honored by the kernel.
#
# Pre-push hook install (still routed through the hardened wrapper):
#     printf '%s\n' '#!/bin/sh' 'exec /bin/bash -p "$(git rev-parse --show-toplevel)/ci/ci_local_sanitized.sh" "$@"' \
#         > .git/hooks/pre-push && chmod +x .git/hooks/pre-push
# ============================================================================

# --- builtin-only prologue --------------------------------------------------
# Set a trusted PATH using ONLY shell builtins, before ANY external command
# runs. Nothing external has executed yet, so a hostile inherited PATH cannot
# intercept this assignment; every external command below additionally uses an
# absolute path, so PATH interception is defeated twice over.
PATH=/usr/bin:/bin
export PATH

set -uo pipefail

# Private, non-secret sanitization state. These names are carried ONLY through
# this wrapper's own `env -i` re-exec and are never *trusted* when inherited
# from an unsanitized caller (the environment is re-verified, not believed).
readonly _UFCI_SEAL='__UFCI_SANITIZED_V3'
readonly _UFCI_SEAL_VALUE='sealed'

_ufci_usage() {
  cat <<'EOF'
ci_local_sanitized.sh — hardened wrapper around ci/ci_local.sh

Supported invocation (the only supported security entrypoint):
    /bin/bash -p ci/ci_local_sanitized.sh [args...]

Every argument is forwarded UNCHANGED to ci/ci_local.sh after the environment
has been reduced to a minimal, trusted set:
  * PATH is reset to /usr/bin:/bin with a builtin before any external command.
  * The process re-execs through `env -i`, dropping all inherited variables.
  * BASH_ENV, ENV, exported shell functions (BASH_FUNC_*) and inherited
    functions are removed.
  * Git routing variables (GIT_DIR, GIT_WORK_TREE, GIT_INDEX_FILE, ...) and
    inline config (GIT_CONFIG_COUNT / GIT_CONFIG_KEY_n / GIT_CONFIG_VALUE_n)
    are removed; global and system Git config are neutralized
    (GIT_CONFIG_GLOBAL=/dev/null, GIT_CONFIG_SYSTEM=/dev/null,
    GIT_CONFIG_NOSYSTEM=1).

Arguments understood by ci/ci_local.sh (forwarded verbatim):
    --full   quick + build + tests + -Werror production build
    --msan   also run the no-ASM MSan smoke path
    --gpu    also run the local GPU CTest suite

Options handled locally by this wrapper (no delegation, no side effects):
    -h, --help   show this help and exit
EOF
}

# --help / -h is handled LOCALLY: no delegation, no environment transition,
# works even under a plain `bash ci/ci_local_sanitized.sh --help`.
if [ "${1-}" = "-h" ] || [ "${1-}" = "--help" ]; then
  _ufci_usage
  exit 0
fi

# --- strict minimality check ------------------------------------------------
# Returns 0 only when the LIVE environment is already reduced to the trusted
# whitelist AND carries this wrapper's seal. Because it inspects the real
# environment instead of trusting an inherited flag, a caller cannot skip the
# transition by presetting the seal or depth variables: any extra inherited
# variable (dangerous or merely unexpected) makes this fail and forces the
# `env -i` re-exec.
_ufci_is_minimal() {
  [ "${!_UFCI_SEAL-}" = "$_UFCI_SEAL_VALUE" ] || return 1
  [ "${PATH-}" = "/usr/bin:/bin" ] || return 1
  local _name
  while IFS= read -r _name; do
    [ -n "$_name" ] || continue
    case "$_name" in
      PATH|HOME|PWD|SHLVL|OLDPWD|_) continue ;;
      GIT_CONFIG_GLOBAL|GIT_CONFIG_SYSTEM|GIT_CONFIG_NOSYSTEM) continue ;;
    esac
    [ "$_name" = "$_UFCI_SEAL" ] && continue
    return 1
  done <<EOF
$(compgen -e 2>/dev/null || true)
EOF
  return 0
}

# --- explicit in-process strip (defense in depth, builtins only) ------------
# `env -i` below is the primary, un-bypassable guarantee, but we also remove
# the dangerous state explicitly so the intent is legible and holds even if a
# future change ever weakened the re-exec. Every loop enumerates the REAL
# process environment, so it is inherently bounded by the number of variables
# actually present — a forged GIT_CONFIG_COUNT (however large) can never drive
# an unbounded loop here.
_ufci_strip_inherited() {
  # Shell startup hooks.
  unset BASH_ENV ENV 2>/dev/null || true

  local _name
  while IFS= read -r _name; do
    [ -n "$_name" ] || continue
    case "$_name" in
      # Exported/imported bash functions (BASH_FUNC_foo%% and BASH_FUNC_foo()).
      BASH_FUNC_*) unset "$_name" 2>/dev/null || true ;;
      # Trusted-git local routing variables.
      GIT_DIR|GIT_WORK_TREE|GIT_INDEX_FILE|GIT_OBJECT_DIRECTORY| \
      GIT_ALTERNATE_OBJECT_DIRECTORIES|GIT_COMMON_DIR|GIT_NAMESPACE| \
      GIT_CEILING_DIRECTORIES|GIT_DISCOVERY_ACROSS_FILESYSTEM|GIT_PREFIX| \
      GIT_WORK_TREE|GIT_CONFIG)
        unset "$_name" 2>/dev/null || true ;;
      # Validated, bounded inline config: the count itself and every
      # GIT_CONFIG_KEY_n / GIT_CONFIG_VALUE_n entry actually present.
      GIT_CONFIG_COUNT|GIT_CONFIG_KEY_*|GIT_CONFIG_VALUE_*)
        unset "$_name" 2>/dev/null || true ;;
    esac
  done <<EOF
$(compgen -e 2>/dev/null || true)
EOF
}

# --- transition to a guaranteed-minimal environment -------------------------
# `env -i` provably converges in a SINGLE hop: it drops every inherited
# variable and re-establishes only the trusted whitelist, after which
# _ufci_is_minimal is true and we fall straight through to delegation. Because
# convergence is single-hop there is no re-exec loop to bound and no internal
# stage/count variable for a caller to forge — an inherited seal or count-like
# variable is simply wiped by env -i and can neither skip nor inflate the
# transition.
if ! _ufci_is_minimal; then
  _ufci_strip_inherited

  # THE un-bypassable state change. Absolute interpreter/env paths defeat a
  # hostile inherited PATH; the caller's argv ("$@") is forwarded UNCHANGED
  # (stage-like tokens included).
  exec /usr/bin/env -i \
    PATH=/usr/bin:/bin \
    HOME=/nonexistent \
    GIT_CONFIG_GLOBAL=/dev/null \
    GIT_CONFIG_SYSTEM=/dev/null \
    GIT_CONFIG_NOSYSTEM=1 \
    "$_UFCI_SEAL=$_UFCI_SEAL_VALUE" \
    /bin/bash -p "$0" "$@"
fi

# --- stage 2: sealed, minimal environment — delegate UNCHANGED args ---------
# Drop this wrapper's own seal so the delegated ci_local.sh inherits a truly
# minimal environment (no internal markers leak downstream).
unset "$_UFCI_SEAL" 2>/dev/null || true

# Resolve the tracked sibling ci_local.sh from THIS script's own location (the
# known card path), never via discovery tooling.
_ufci_self="$0"
case "$_ufci_self" in
  /*) : ;;
  *)  _ufci_self="$PWD/$_ufci_self" ;;
esac
_ufci_dir=${_ufci_self%/*}
_ufci_target="$_ufci_dir/ci_local.sh"

if [ ! -f "$_ufci_target" ]; then
  printf 'ci_local_sanitized.sh: sibling ci_local.sh not found at %s\n' \
    "$_ufci_target" >&2
  exit 66
fi

# Hand off: absolute interpreter, tracked target, caller arguments verbatim.
exec /bin/bash -p "$_ufci_target" "$@"
