# Dockerfile for M2T2 (Multi-Task Masked Transformer) pick-and-place inference.
#
# Base image is CUDA 11.7 devel on Ubuntu 22.04 rather than a pip/conda-only setup:
# pointnet2_ops compiles a CUDA extension at install time via nvcc, and nvcc needs a
# host GCC it officially supports. Ubuntu 22.04's system nvcc (11.5) only supports
# GCC <=10, but the OS defaults to GCC 11 -> compile failure in <bits/std_function.h>.
# CUDA 11.7 is the first release with official GCC 11 support, so building inside a
# real CUDA 11.7 devel image sidesteps the mismatch instead of patching toolchains.
ARG BASE_IMAGE=nvidia/cuda:11.7.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# python3.10 (Ubuntu 22.04 default) + build toolchain for the pointnet2_ops extension
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && \
    apt-get install --no-install-recommends -y \
        python3 \
        python3-dev \
        python3-pip \
        git \
        ninja-build \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# torch/torchvision pinned to the cu117 build matching the base image's CUDA runtime
# (see README.md install steps: torch==2.0.1 / torchvision==0.15.2)
RUN pip3 install torch==2.0.1+cu117 torchvision==0.15.2+cu117 \
        --index-url https://download.pytorch.org/whl/cu117

# numpy<2 — these torch/torchvision wheels are built against the NumPy 1.x C ABI;
# NumPy 2.x causes "A module compiled using NumPy 1.x cannot be run in NumPy 2.x"
RUN pip3 install "numpy<2"

WORKDIR /workspace/m2t2

# pointnet2_ops CUDA extension, built ahead of time against the target GPU's arch.
# Override at build time (--build-arg TORCH_CUDA_ARCH_LIST=...) for other GPUs;
# 8.6 covers this workstation's RTX A2000 (Ampere, GA106).
ARG TORCH_CUDA_ARCH_LIST="8.6"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV FORCE_CUDA=1

COPY pointnet2_ops ./pointnet2_ops
RUN pip3 install --no-build-isolation ./pointnet2_ops

# remaining python deps
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# m2t2 package itself. Not an editable install: its plain setup.py (no pyproject.toml)
# doesn't support PEP 660 editable hooks. m2t2/ also has no __init__.py, so pip installs
# it as an empty namespace-package stub; the actual code is only importable via
# PYTHONPATH below, which also makes bind-mounted local edits take effect regardless of
# which subdirectory the entry-point script lives in (e.g. client-server/m2t2_server.py).
COPY . .
RUN pip3 install --no-deps .
ENV PYTHONPATH=/workspace/m2t2

ENV PYTHONUNBUFFERED=1

# ZMQ port for the headless inference server (client-server/m2t2_server.py).
# No meshcat dependency: the server never calls create_visualizer/meshcat_utils.
EXPOSE 5556

CMD ["python3", "client-server/m2t2_server.py", "--checkpoint", "/checkpoints/m2t2.pth", "--port", "5556"]
