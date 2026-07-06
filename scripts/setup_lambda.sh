#!/usr/bin/env bash
set -euo pipefail

MIMIC_VIDEO_COMMIT="e3355dbc93132b576c02f920a59b4fc18a4f5906"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="${VIDEO_POLICY_CACHE:-.cache}"
MODELS=("object_full")

usage() {
  echo "Usage: $0 [--cache-dir PATH] [--models MODEL ...]"
  echo "Example: $0 --models object_full goal_half spatial_full"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cache-dir)
      CACHE_DIR="$2"
      shift 2
      ;;
    --models)
      MODELS=()
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        MODELS+=("$1")
        shift
      done
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "--models needs at least one model" >&2
  exit 2
fi

DOWNLOAD_MODELS=()
for model in "${MODELS[@]}"; do
  if [[ "$model" == libero_* || "$model" == *_cosmos_bridge ]]; then
    DOWNLOAD_MODELS+=("$model")
  else
    DOWNLOAD_MODELS+=("libero_$model")
  fi
done

mkdir -p "$CACHE_DIR"
CACHE_DIR="$(cd "$CACHE_DIR" && pwd)"
UPSTREAM="$CACHE_DIR/mimic-video"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if [[ ! -d "$UPSTREAM/.git" ]]; then
  git clone https://github.com/mimic-video/mimic-video.git "$UPSTREAM"
fi

CURRENT_REMOTE="$(git -C "$UPSTREAM" remote get-url origin)"
if [[ "$CURRENT_REMOTE" != "https://github.com/mimic-video/mimic-video.git" ]]; then
  echo "Refusing to modify unexpected repository at $UPSTREAM ($CURRENT_REMOTE)" >&2
  exit 1
fi
git -C "$UPSTREAM" fetch origin "$MIMIC_VIDEO_COMMIT"
git -C "$UPSTREAM" checkout --detach "$MIMIC_VIDEO_COMMIT"

# Upstream's multi-architecture source table can make recent uv versions select
# its CUDA 12.9 build dependencies while resolving the cu126 extra. Narrow it to
# the published x86_64 CUDA 12.6 wheels used by A100 hosts.
UV_PATCH="$REPO_ROOT/patches/mimic-video-a100-uv.patch"
if ! git -C "$UPSTREAM" apply --reverse --check "$UV_PATCH" 2>/dev/null; then
  git -C "$UPSTREAM" apply --check "$UV_PATCH"
  git -C "$UPSTREAM" apply "$UV_PATCH"
fi

echo "Installing the MimicVideo CUDA 12.6 environment (A100-compatible)..."
(
  cd "$UPSTREAM/model"
  uv sync --extra cu126
  uv pip install -r ../eval/libero/LIBERO/requirements.txt
  # robosuite 1.4 calls the MuJoCo 2.x mj_fullM signature, but its open-ended
  # dependency currently resolves to incompatible MuJoCo 3.x releases.
  uv pip install "mujoco==2.3.7"
  uv pip install -e ../eval/libero/LIBERO
  uv pip install -e "$REPO_ROOT"
)

echo "Downloading checkpoints: ${DOWNLOAD_MODELS[*]}"
(
  cd "$UPSTREAM/model"
  uv run python scripts/download_checkpoints.py \
    --checkpoint-dir "$CACHE_DIR/checkpoints" \
    --models "${DOWNLOAD_MODELS[@]}"
)

MODEL_CHECKPOINTS="$UPSTREAM/model/checkpoints"
if [[ -e "$MODEL_CHECKPOINTS" && ! -L "$MODEL_CHECKPOINTS" ]]; then
  echo "Refusing to replace existing checkpoint directory: $MODEL_CHECKPOINTS" >&2
  exit 1
fi
ln -sfn "$CACHE_DIR/checkpoints" "$MODEL_CHECKPOINTS"

echo
echo "Setup complete."
echo "Run:"
echo "  $UPSTREAM/model/.venv/bin/video-policy --cache-dir $CACHE_DIR doctor"
echo "  $UPSTREAM/model/.venv/bin/video-policy --cache-dir $CACHE_DIR rollout --model ${MODELS[0]} --tasks 0"
