# Layers manifeel's training stack on top of the isaac-lab-base image built by
# IsaacLab/docker/container.py (``python3 docker/container.py build base`` from the
# IsaacLab repo root). That image already has Isaac Sim 6.1.0 + the full Isaac Lab
# SDK installed into a uv-managed venv at $VIRTUAL_ENV, with torch==2.12.0+cu130
# (verified sm_120/Blackwell-compatible on the RTX 5090).
#
# Build context is manifeel/ itself; diffusion_policy/ and maniflow/ (separate sibling
# checkouts, not part of this repo) are passed in as named additional build contexts so
# nothing outside manifeel/ needs its own .dockerignore:
#   cd /home/aist/hisham/contact_rich_sim/manifeel
#   docker build -f docker/Dockerfile \
#     --build-context diffusion_policy=../diffusion_policy \
#     --build-context maniflow=../maniflow \
#     -t manifeel-isaaclab .

ARG BASE_IMAGE=isaac-lab-base:latest
FROM ${BASE_IMAGE}

# isaacsim.pip.nv's pip_prebundle/ is supposed to vendor nvidia-cuda-nvrtc (used by the
# RTX renderer's shader JIT compiler) but the directory is simply absent -- confirmed
# missing even in the clean isaac-lab-base:latest layer, so this is an upstream Isaac
# Sim 6.1.0 packaging gap, not something manifeel broke. Every run logs "Could not load
# libnvrtc-builtins.so" as a result; suspected cause of the interactive viewport being
# unusably slow during continuous rendering (a single post-reset frame hid it, but
# sustained motion didn't) -- not yet confirmed fixed, just the most likely remaining
# culprit after ruling out GPU clock throttling and render mode.
RUN mkdir -p /isaac-sim/exts/isaacsim.pip.nv/pip_prebundle/nvidia/cuda_nvrtc/lib && \
    /opt/isaaclab-venv/bin/python3 -m pip install nvidia-cuda-nvrtc-cu12==12.8.61 \
        --no-deps --target=/tmp/nvrtc-fix && \
    cp /tmp/nvrtc-fix/nvidia/cuda_nvrtc/lib/*.so* \
        /isaac-sim/exts/isaacsim.pip.nv/pip_prebundle/nvidia/cuda_nvrtc/lib/ && \
    rm -rf /tmp/nvrtc-fix

# The venv is world-writable (the base image chmods it a+rwX), so install as the
# isaaclab runtime user directly instead of root -- new files land correctly owned
# from the start, which avoids a chown -R over the whole ~259-package venv
# afterward (that was observed taking multiple hours: re-stamping ownership on a
# tree that's already 95%+ correct, one lchown() syscall per file, on this disk).
# $HOME is inherited as /root (mode 750, not writable by isaaclab), so point uv's
# cache somewhere isaaclab can actually write instead.
USER isaaclab
ENV UV_CACHE_DIR=/tmp/uv-cache

# This whole list is a pinned set already verified to coexist with Isaac Lab's deps
# (including torch itself, which comes from the base image and isn't listed here) --
# --no-deps so uv doesn't re-resolve against individual packages' own metadata.
# Concretely: timm==1.0.11 declares an exact torch==2.12.0 dependency, which pulls in
# triton==3.7.0, conflicting with the triton==3.6.0 pinned below -- without --no-deps
# that contradiction fails the whole install, even though every real runtime dependency
# timm needs (pyyaml, huggingface-hub, safetensors, numpy) is already satisfied by
# either this file or the base image.
COPY --chown=isaaclab:isaaclab docker/requirements-extra.txt /tmp/requirements-extra.txt
RUN --mount=type=cache,target=/tmp/uv-cache,uid=1000,gid=1000 \
    uv pip install --no-deps -r /tmp/requirements-extra.txt && \
    rm /tmp/requirements-extra.txt

COPY --chown=isaaclab:isaaclab . /workspace/manifeel/
COPY --chown=isaaclab:isaaclab --from=diffusion_policy . /workspace/diffusion_policy/
COPY --chown=isaaclab:isaaclab --from=maniflow . /workspace/maniflow/

RUN --mount=type=cache,target=/tmp/uv-cache,uid=1000,gid=1000 \
    uv pip install --no-deps -e /workspace/manifeel -e /workspace/diffusion_policy -e /workspace/maniflow

ENV OMNI_KIT_ACCEPT_EULA=YES
ENV ACCEPT_EULA=Y

WORKDIR /workspace/manifeel
