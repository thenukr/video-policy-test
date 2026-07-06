#!/usr/bin/env bash
set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; this setup expects an NVIDIA GPU host." >&2
  exit 1
fi

DRIVER_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
NVIDIA_GL_PACKAGE="libnvidia-gl-${DRIVER_MAJOR}-server"

sudo apt-get update
sudo apt-get install -y \
  ffmpeg \
  libegl1 \
  libgl1 \
  libglfw3 \
  libglew2.2 \
  libosmesa6 \
  libvulkan1 \
  "$NVIDIA_GL_PACKAGE"

echo "Installed headless rendering libraries, including $NVIDIA_GL_PACKAGE."

