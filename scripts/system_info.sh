#!/usr/bin/env bash
set -u

NVIDIA_OUT="$(mktemp "${TMPDIR:-/tmp}/g1-nvidia-smi.XXXXXX")"
NVIDIA_ERR="$(mktemp "${TMPDIR:-/tmp}/g1-nvidia-smi.XXXXXX")"
trap 'rm -f "$NVIDIA_OUT" "$NVIDIA_ERR"' EXIT

section() {
  printf '\n===== %s =====\n' "$1"
}

section "Working Directory"
pwd

section "Ubuntu"
if [ -f /etc/os-release ]; then
  cat /etc/os-release
else
  echo "/etc/os-release not found"
fi

section "Kernel"
uname -a
cat /proc/version 2>/dev/null || true

section "WSL Detection"
if grep -qi microsoft /proc/version /proc/sys/kernel/osrelease 2>/dev/null; then
  echo "WSL detected"
else
  echo "WSL not detected from kernel strings"
fi
if grep -qi wsl2 /proc/sys/kernel/osrelease 2>/dev/null || uname -a | grep -qi WSL2; then
  echo "WSL2 detected"
else
  echo "WSL2 not confirmed"
fi

section "WSL Version"
if command -v wsl.exe >/dev/null 2>&1; then
  wsl.exe --version 2>&1 | tr -d '\000' || true
else
  echo "wsl.exe not available from this shell"
fi

section "CPU"
lscpu || true

section "Memory"
free -h || true

section "Python"
command -v python3 || true
python3 --version 2>&1 || true
command -v python || true
python --version 2>&1 || true
find /usr/bin "$HOME/miniforge3" -path '*/bin/python*' -type f -executable 2>/dev/null \
  | grep -E '/python([0-9]+(\.[0-9]+)?)?$' \
  | sort \
  | while read -r p; do
  printf '%s: ' "$p"
  "$p" --version 2>&1 || true
done

section "Conda"
if command -v conda >/dev/null 2>&1; then
  conda --version 2>&1 || true
  printf 'CONDA_DEFAULT_ENV=%s\n' "${CONDA_DEFAULT_ENV:-}"
  printf 'CONDA_PREFIX=%s\n' "${CONDA_PREFIX:-}"
  conda env list 2>&1 || true
else
  echo "conda not found"
fi

section "Git"
git --version 2>&1 || true

section "NVIDIA WSL Devices"
ls -l /dev/dxg /usr/lib/wsl/lib/nvidia-smi /usr/lib/wsl/lib/libcuda.so.1 2>&1 || true

section "nvidia-smi"
if command -v nvidia-smi >/dev/null 2>&1; then
  command -v nvidia-smi
  nvidia-smi 2>&1 || true
else
  echo "nvidia-smi not found"
fi

section "GPU Summary"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >"$NVIDIA_OUT" 2>"$NVIDIA_ERR"; then
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader,nounits 2>/dev/null \
    | awk -F, '{gsub(/^ +| +$/, "", $1); gsub(/^ +| +$/, "", $2); print "GPU: " $1 "\nDriver: " $2}'
  sed -n 's/.*CUDA Version: *\([^ |]*\).*/Driver CUDA: \1/p' "$NVIDIA_OUT" | head -n 1
  echo "GPU passthrough appears to be working"
else
  cat "$NVIDIA_ERR" 2>/dev/null || true
  echo "GPU passthrough is not working or nvidia-smi failed"
fi
