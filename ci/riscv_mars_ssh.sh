#!/usr/bin/env bash
set -euo pipefail

# Milk-V Mars (RISC-V) helper.
# Password is always entered by the user at runtime (never stored).

HOST="192.168.1.31"
USER_NAME="user"

SSH_OPTS=(
  -o PreferredAuthentications=password
  -o PubkeyAuthentication=no
  -o StrictHostKeyChecking=accept-new
)

if command -v sshpass >/dev/null 2>&1; then
  # Optional path: secure hidden prompt handled locally, then pass via env.
  read -rsp "Password for ${USER_NAME}@${HOST}: " SSHPASS
  echo
  export SSHPASS
  if [ "$#" -gt 0 ]; then
    sshpass -e ssh "${SSH_OPTS[@]}" "${USER_NAME}@${HOST}" "$@"
  else
    sshpass -e ssh "${SSH_OPTS[@]}" "${USER_NAME}@${HOST}"
  fi
  unset SSHPASS
else
  # Native SSH prompt: you type password directly in ssh prompt.
  if [ "$#" -gt 0 ]; then
    ssh "${SSH_OPTS[@]}" "${USER_NAME}@${HOST}" "$@"
  else
    ssh "${SSH_OPTS[@]}" "${USER_NAME}@${HOST}"
  fi
fi
