#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(realpath "$SCRIPT_DIR/..")"

IMAGE_NAME="${IMAGE_NAME:-m2t2:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-m2t2}"

# Code, weights and sample data are bind-mounted rather than baked into the image
# (weights/ alone is ~220MB and changes independently of the code).
# weights/ is additionally mounted at /checkpoints to match the image's default CMD
# (client-server/m2t2_server.py --checkpoint /checkpoints/m2t2.pth), a headless ZMQ
# server with no meshcat dependency.
docker run -e "TERM=xterm-256color" -t --rm -it \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --ipc=host \
  --shm-size="8g" \
  -p 5556:5556 \
  -v "$REPO_DIR:/workspace/m2t2" \
  -v "$REPO_DIR/weights:/checkpoints:ro" \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  "$IMAGE_NAME"

# To get an interactive shell instead of the default server (e.g. to run demo.py
# manually, which still needs meshcat-server & running on port 7000):
#   docker run ... "$IMAGE_NAME" /bin/bash
