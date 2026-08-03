## Native build attempt (abandoned)

Chain of failures, each masking the next:

uv's build isolation hides torch from setup.py → fix: --no-build-isolation
setuptools 83 in the venv dropped pkg_resources, which torch's cpp_extension.py still imports → fix: uv pip install "setuptools<81"
CUDA_HOME=/usr/local/cuda-12.8 points at a directory that doesn't exist → fix: CUDA_HOME=/usr (the real toolkit is the apt nvidia-cuda-toolkit 11.5, with nvcc at /usr/bin/nvcc)
Blocker: nvcc 11.5 only officially supports GCC ≤10, but Ubuntu 22.04 defaults to GCC 11 → compile error in <bits/std_function.h>
g++-10 via apt was declined. Untried alternative found at the time: a CUDA 11.8 toolkit cached locally at /home/nexus/.cache/packman/chk/cuda/11.8.0-3-linux-x86_64-release (from Isaac Sim/packman), which officially supports GCC 11.

## Resolution: containerized build

Rather than patch the host toolchain, the GCC/nvcc mismatch was sidestepped by building inside `docker/Dockerfile`, based on `nvidia/cuda:11.7.1-devel-ubuntu22.04` — CUDA 11.7 is the first release with official GCC 11 support, so the container's own nvcc matches Ubuntu 22.04's default compiler with no pinning needed. Image builds via `docker/build.sh` (`TORCH_CUDA_ARCH_LIST=8.6` for this workstation's RTX A2000), run via `docker/run.sh`.

**Verified functional (2026-08-03):**
- Image `m2t2:latest` builds and runs; `--gpus all` passthrough works — `torch.cuda.is_available()` is `True`, GPU detected as RTX A2000 12GB.
- `pointnet2_ops` CUDA extension and the `m2t2` package both import cleanly inside the container.
- End-to-end inference verified: `python3 demo.py eval.checkpoint=weights/m2t2.pth eval.data_dir=sample_data/real_world/00 eval.mask_thresh=0.4 eval.num_runs=1` ran successfully, producing `object_00 has 7 grasps`. Runtime is a few seconds once the visualizer is up.
- Gotcha: `demo.py` blocks on start with "Waiting for meshcat server..." until `meshcat-server` is running and reachable on port 7000 — start it first (`meshcat-server &` inside the container, per `run.sh`'s trailing comment), otherwise the run hangs indefinitely rather than failing.