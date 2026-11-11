FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_DRIVER_CAPABILITIES=all \
    NVIDIA_VISIBLE_DEVICES=all \
    LC_ALL=C.UTF-8 LANG=C.UTF-8 PYTHONIOENCODING=utf-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 python3.8-dev python3-pip python3.8-venv \
    build-essential git wget curl ca-certificates ninja-build \
    libx11-6 libxext6 libxrender1 libxrandr2 libxcursor1 libxi6 libxinerama1 \
    libglu1-mesa libglfw3 libegl1 libgl1 libgl1-mesa-dev freeglut3-dev \
    libvulkan1 vulkan-tools mesa-vulkan-drivers \
    libusb-1.0-0 libglib2.0-0 libisl-dev xvfb \
 && rm -rf /var/lib/apt/lists/* \
 && ln -sf /usr/bin/python3.8 /usr/local/bin/python \
 && ln -sf /usr/bin/pip3 /usr/local/bin/pip

RUN python -m pip install --upgrade "pip<25" setuptools wheel

# PyTorch matched to CUDA 11.7 (works on sm_86)
RUN pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# IsaacGym's own deps
RUN pip install pyyaml scipy pillow imageio ninja matplotlib gym==0.23.1 \
    pyvirtualdisplay "cython<3" tensorboard

# Same pins as install.sh
RUN pip install wandb==0.12.21 dill==0.3.9 tqdm==4.67.1 av==12.3.0 numpy==1.23.3 \
    opencv-python==4.10.0.84 zarr==2.16.1 einops==0.4.1 huggingface-hub==0.25.0 \
    diffusers==0.11.1 pandas==2.0.3 numba==0.56.4 rtree==1.3.0

RUN apt-get update && apt-get install -y --no-install-recommends cmake gfortran unzip python3-tk \
 && rm -rf /var/lib/apt/lists/*

# Remaining pins from manifeel/requirements.txt
# torchcfm pulls in `pot`, whose newest release has no py3.8 wheel and fails to compile
RUN pip install --only-binary=:all: pot==0.9.4

RUN pip install pytorch-lightning==1.1.1 escnn==1.0.11 torchcfm==1.0.5 transformers==4.42.3

# manifeel-isaacgymenvs deps (setup.py INSTALL_REQUIRES)
RUN pip install omegaconf termcolor jinja2 hydra-core==1.2.0 rl-games \
    urdfpy==0.0.22 pysdf==0.1.9 warp-lang==0.10.1 trimesh==3.23.5

# diffusion_policy deps (conda_environment.yaml)
RUN pip install numcodecs==0.10.2 scikit-video==1.1.11 scikit-image==0.19.3 \
    pymunk==6.2.1 threadpoolctl==3.1.0 shapely==1.8.4 tensorboardx==2.5.1 \
    accelerate==0.13.2 imagecodecs

# Later installs (rl-games, lightning-bolts, ...) upgrade torch; force the pair back (ManiFeel needs torchvision >= 0.15 for _functional_tensor).
RUN pip install --force-reinstall --no-deps \
    torch==2.0.1+cu117 torchvision==0.15.2+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117 \
 && pip check || true

# Bind-mounted repos are owned by the host user; let git (used by wandb) read them as root.
RUN git config --global --add safe.directory '*'

# Source trees are bind-mounted at /workspace; use PYTHONPATH instead of `pip install -e`
# so nothing is lost when the container exits.
ENV PYTHONPATH=/workspace/IsaacGym_Preview_TacSL_Package/isaacgym/python:/workspace/manifeel-isaacgymenvs:/workspace/diffusion_policy:/workspace/manifeel \
    LD_LIBRARY_PATH=/usr/local/lib/python3.8/dist-packages/torch/lib:${LD_LIBRARY_PATH}

WORKDIR /workspace/manifeel
CMD ["bash"]
