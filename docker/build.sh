#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(realpath "$SCRIPT_DIR/..")"

IMAGE_NAME="${IMAGE_NAME:-m2t2:latest}"
# GPU arch(s) to compile pointnet2_ops for. Default matches this workstation's
# RTX A2000 (Ampere / sm_86). Override e.g. TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6" for
# other/multiple GPUs.
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
LOG_FILE="$SCRIPT_DIR/build-$(date +%Y%m%d-%H%M%S).log"

echo "Full build log: $LOG_FILE"

docker build \
  --file "$SCRIPT_DIR/Dockerfile" \
  --tag "$IMAGE_NAME" \
  --build-arg "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST" \
  "$REPO_DIR" \
  2>&1 | tee "$LOG_FILE"
