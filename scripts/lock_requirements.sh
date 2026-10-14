#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="g1-locomotion"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCK_FILE="$REPO_ROOT/requirements.lock.txt"
TMP_LOCK="$(mktemp "${TMPDIR:-/tmp}/g1-requirements.XXXXXX.txt")"
trap 'rm -f "$TMP_LOCK"' EXIT

command -v conda >/dev/null 2>&1 || {
  echo "ERROR: conda is required; run scripts/setup.sh first." >&2
  exit 1
}

ENV_PREFIX="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.prefix)')"
ENV_PYTHON="$ENV_PREFIX/bin/python"

conda run -n "$ENV_NAME" uv pip compile \
  "$REPO_ROOT/requirements.in" \
  --output-file "$TMP_LOCK" \
  --python "$ENV_PYTHON" \
  --python-version 3.12 \
  --python-platform x86_64-unknown-linux-gnu \
  --torch-backend cu130 \
  --generate-hashes \
  --annotation-style split \
  --emit-index-url \
  --custom-compile-command './scripts/lock_requirements.sh'

if [[ -f "$LOCK_FILE" ]] && cmp -s "$TMP_LOCK" "$LOCK_FILE"; then
  echo "Lock is already current: $LOCK_FILE"
else
  mv "$TMP_LOCK" "$LOCK_FILE"
  echo "Updated lock: $LOCK_FILE"
fi
