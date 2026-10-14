#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="g1-locomotion"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOGDIR="runs"
HOST="127.0.0.1"
PORT="6006"
BROWSER="chrome"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/tensorboard.sh [options]

Launch TensorBoard for live RSL-RL training logs.

Options:
  --logdir PATH       TensorBoard log directory, relative to repo root unless absolute. Default: runs
  --host HOST         Bind host. Default: 127.0.0.1
  --port PORT         Bind port. Default: 6006
  --browser MODE      chrome, default, or none. Default: chrome
  --dry-run           Print the resolved command and URL without starting TensorBoard.
  -h, --help          Show this help.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --logdir)
      [[ $# -ge 2 ]] || die "--logdir requires a value"
      LOGDIR="$2"
      shift 2
      ;;
    --host)
      [[ $# -ge 2 ]] || die "--host requires a value"
      HOST="$2"
      shift 2
      ;;
    --port)
      [[ $# -ge 2 ]] || die "--port requires a value"
      PORT="$2"
      shift 2
      ;;
    --browser)
      [[ $# -ge 2 ]] || die "--browser requires a value"
      BROWSER="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

case "$BROWSER" in
  chrome|default|none) ;;
  *) die "--browser must be one of: chrome, default, none" ;;
esac

if [[ "$LOGDIR" = /* ]]; then
  RESOLVED_LOGDIR="$LOGDIR"
else
  RESOLVED_LOGDIR="$REPO_ROOT/$LOGDIR"
fi

URL="http://$HOST:$PORT"

command -v conda >/dev/null 2>&1 || die "conda is required; run ./scripts/setup.sh first"
mkdir -p "$RESOLVED_LOGDIR"

TB_CMD=(conda run -n "$ENV_NAME" tensorboard --logdir "$RESOLVED_LOGDIR" --host "$HOST" --port "$PORT")

print_command() {
  printf '%q ' "${TB_CMD[@]}"
  printf '\n'
}

open_browser() {
  [[ "$BROWSER" == "none" ]] && return 0

  if [[ "$BROWSER" == "chrome" ]]; then
    if command -v powershell.exe >/dev/null 2>&1; then
      powershell.exe -NoProfile -Command "Start-Process chrome '$URL'" >/dev/null 2>&1 && return 0
    fi
    if command -v cmd.exe >/dev/null 2>&1; then
      cmd.exe /C start "" chrome "$URL" >/dev/null 2>&1 && return 0
    fi
    for candidate in google-chrome google-chrome-stable chromium chromium-browser; do
      if command -v "$candidate" >/dev/null 2>&1; then
        "$candidate" "$URL" >/dev/null 2>&1 &
        return 0
      fi
    done
  fi

  if command -v cmd.exe >/dev/null 2>&1; then
    cmd.exe /C start "" "$URL" >/dev/null 2>&1 && return 0
  fi
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$URL" >/dev/null 2>&1 &
    return 0
  fi

  echo "Could not open a browser automatically. Open this URL manually: $URL"
}

echo "Repo: $REPO_ROOT"
echo "Logdir: $RESOLVED_LOGDIR"
echo "URL: $URL"
echo "Command: $(print_command)"

if [[ "$DRY_RUN" -eq 1 ]]; then
  conda run -n "$ENV_NAME" tensorboard --version >/dev/null
  echo "Dry run passed: TensorBoard is available in conda env $ENV_NAME."
  exit 0
fi

open_browser || true
exec "${TB_CMD[@]}"
