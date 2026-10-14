#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="g1-locomotion"
EXPECTED_PYTHON="3.12.14"
EXPECTED_PIP="26.2.1"
EXPECTED_UV="0.12.17"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

echo "==> Checking WSL/Linux host"
case "$(uname -s)" in
  Linux) ;;
  *) die "This setup must run under Linux inside WSL2." ;;
esac

if ! grep -qi microsoft /proc/version /proc/sys/kernel/osrelease 2>/dev/null; then
  die "This shell does not appear to be running under WSL."
fi

if ! grep -qi wsl2 /proc/sys/kernel/osrelease 2>/dev/null && ! uname -a | grep -qi WSL2; then
  die "WSL was detected, but WSL2 was not confirmed."
fi

case "$ROOT_DIR" in
  /home/*) ;;
  *) die "Repository must live under the WSL Linux filesystem, not a Windows mount: $ROOT_DIR" ;;
esac

echo "==> Checking NVIDIA GPU passthrough"
if [ ! -e /dev/dxg ]; then
  die "/dev/dxg is missing; WSL GPU passthrough is not available."
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  die "nvidia-smi was not found inside WSL."
fi

nvidia-smi >/dev/null || die "nvidia-smi failed; NVIDIA GPU is not available inside WSL."
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

if command -v mamba >/dev/null 2>&1; then
  CONDA_FRONTEND="mamba"
elif command -v conda >/dev/null 2>&1; then
  CONDA_FRONTEND="conda"
else
  die "conda or mamba is required. Install Miniforge or make conda available on PATH."
fi

if command -v conda >/dev/null 2>&1; then
  CONDA_RUN="conda"
else
  CONDA_RUN="$CONDA_FRONTEND"
fi

echo "==> Using $($CONDA_FRONTEND --version | head -n 1)"

if [ -d "$ROOT_DIR/.venv" ]; then
  echo "==> Removing old project .venv created by the previous uv-only setup"
  rm -rf "$ROOT_DIR/.venv"
fi

if $CONDA_RUN env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "==> Updating conda environment: $ENV_NAME"
  $CONDA_FRONTEND env update -n "$ENV_NAME" -f environment.yml
else
  echo "==> Creating conda environment: $ENV_NAME"
  $CONDA_FRONTEND env create -f environment.yml
fi

ENV_PREFIX="$($CONDA_RUN run -n "$ENV_NAME" python -c 'import sys; print(sys.prefix)')"
ENV_PYTHON="$ENV_PREFIX/bin/python"

if [ ! -x "$ENV_PYTHON" ]; then
  die "Could not find conda env Python at $ENV_PYTHON"
fi

echo "==> Installing pinned pip wheels into $ENV_PREFIX"
$CONDA_RUN run -n "$ENV_NAME" uv pip sync \
  --python "$ENV_PYTHON" \
  --torch-backend cu130 \
  --strict \
  requirements.lock.txt

echo "==> Checking conda bootstrap versions"
$CONDA_RUN run -n "$ENV_NAME" python -c \
  "import sys; assert sys.version.split()[0] == '$EXPECTED_PYTHON', sys.version"
$CONDA_RUN run -n "$ENV_NAME" python -m pip --version | grep -q "pip $EXPECTED_PIP " \
  || die "expected pip $EXPECTED_PIP in the conda environment"
$CONDA_RUN run -n "$ENV_NAME" uv --version | grep -q "uv $EXPECTED_UV" \
  || die "expected uv $EXPECTED_UV in the conda environment"

echo "==> Checking installed dependency consistency"
$CONDA_RUN run -n "$ENV_NAME" uv pip check --python "$ENV_PYTHON"

echo "==> Verifying stack from conda environment"
$CONDA_RUN run -n "$ENV_NAME" python scripts/verify_stack.py

echo
echo "Setup complete."
echo "Activate with:"
echo "  conda activate $ENV_NAME"
echo "Verify manually with:"
echo "  conda run -n $ENV_NAME python scripts/verify_stack.py"
echo "Launch TensorBoard with:"
echo "  ./scripts/tensorboard.sh --logdir runs/<experiment>"
