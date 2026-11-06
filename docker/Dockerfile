# Layers manifeel's training stack on top of the isaac-lab-base image built by
# IsaacLab/docker/container.py (``python3 docker/container.py build base`` from the
# IsaacLab repo root). That image already has Isaac Sim 6.1.0 + the full Isaac Lab
# SDK installed into a uv-managed venv at $VIRTUAL_ENV, with torch==2.12.0+cu130
# (verified sm_120/Blackwell-compatible on the RTX 5090).
#
# Build context is manifeel/ itself; diffusion_policy/ (a separate sibling checkout,
# not part of this repo) is passed in as a named additional build context so nothing
# outside manifeel/ needs its own .dockerignore:
#   cd /home/aist/hisham/contact_rich_sim/manifeel
#   docker build -f docker/Dockerfile \
#     --build-context diffusion_policy=../diffusion_policy \
#     -t manifeel-isaaclab .

ARG BASE_IMAGE=isaac-lab-base:latest
FROM ${BASE_IMAGE}

# The venv is world-writable (the base image chmods it a+rwX), so install as the
# isaaclab runtime user directly instead of root -- new files land correctly owned
# from the start, which avoids a chown -R over the whole ~259-package venv
# afterward (that was observed taking multiple hours: re-stamping ownership on a
# tree that's already 95%+ correct, one lchown() syscall per file, on this disk).
# $HOME is inherited as /root (mode 750, not writable by isaaclab), so point uv's
# cache somewhere isaaclab can actually write instead.
USER isaaclab
ENV UV_CACHE_DIR=/tmp/uv-cache

# manifeel/diffusion_policy's own setup.py dependency lists are looser than the
# versions pinned below (resolved against the conda env that was proven to coexist
# with Isaac Lab's deps) -- --no-deps on both editable installs keeps pip from
# re-resolving and clobbering that pinned set.
COPY --chown=isaaclab:isaaclab docker/requirements-extra.txt /tmp/requirements-extra.txt
RUN --mount=type=cache,target=/tmp/uv-cache,uid=1000,gid=1000 \
    uv pip install -r /tmp/requirements-extra.txt && \
    rm /tmp/requirements-extra.txt

COPY --chown=isaaclab:isaaclab . /workspace/manifeel/
COPY --chown=isaaclab:isaaclab --from=diffusion_policy . /workspace/diffusion_policy/

RUN --mount=type=cache,target=/tmp/uv-cache,uid=1000,gid=1000 \
    uv pip install --no-deps -e /workspace/manifeel -e /workspace/diffusion_policy

ENV OMNI_KIT_ACCEPT_EULA=YES
ENV ACCEPT_EULA=Y

WORKDIR /workspace/manifeel
