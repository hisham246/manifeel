#!/bin/bash
set -euo pipefail

# Trains ManiFlowUsbPolicy on usb_quan_Aug05_contact_manifeel with the SAME observation
# set as the manifeel diffusion_policy baseline (front + tactile_force_field_right +
# state -- see manifeel/config/task/visff_usb.yaml). Runs inside the manifeel-isaaclab
# Docker image (built from docker/Dockerfile) since IsaacGym can't be used on this
# machine's GPU; training itself is pure offline supervised learning over the saved
# dataset, so the simulator is irrelevant here regardless.
#
# Detached (`docker run -d`) instead of nohup: the container keeps running independently
# of this shell the moment it's launched, same guarantee nohup gives a bare process,
# with no extra flag needed inside the container.
#
# Mirrors scripts/run_local.sh's parameter style (SEED/NUM_DEMOS/NUM_EPOCH/EXP_NAME).

# -------------------------
# User-editable parameters
# -------------------------
SEED=${SEED:-44}
NUM_DEMOS=${NUM_DEMOS:-40}
NUM_EPOCH=${NUM_EPOCH:-1000}

DATASET_PATH=${DATASET_PATH:-data/usb_quan_Aug05_contact_manifeel}
TASK_NAME=${TASK_NAME:-visff_usb}

EXP_NAME=${EXP_NAME:-maniflow_usb_${NUM_DEMOS}demos}
IMAGE=${IMAGE:-manifeel-isaaclab:latest}
CONTAINER_NAME=${CONTAINER_NAME:-train_maniflow_usb}

# Optional: export WANDB_API_KEY in your host shell before running this script (get one
# from https://wandb.ai/authorize) to forward it into the container. Without it,
# wandb.init(mode='online') (the default -- see logging.mode in train_maniflow_usb.yaml)
# fails immediately in a non-interactive container with no prompt to fall back on --
# add `logging.mode=disabled` to EXTRA_HYDRA_ARGS below to skip wandb entirely instead.
WANDB_API_KEY=${WANDB_API_KEY:-}
EXTRA_HYDRA_ARGS=${EXTRA_HYDRA_ARGS:-}

# -------------------------
# Derived names
# -------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="data/outputs/${EXP_NAME}/${SEED}"

HYDRA_ARGS=(
  "--config-dir=."
  "--config-name=train_maniflow_usb.yaml"
  "task=${TASK_NAME}"
  "exp_name=${EXP_NAME}"
  "dataset_path=${DATASET_PATH}"
  "training.seed=${SEED}"
  "training.num_epochs=${NUM_EPOCH}"
  "task.dataset.max_train_episodes=${NUM_DEMOS}"
  "hydra.run.dir=${OUT_DIR}"
)

echo "Launching detached container '${CONTAINER_NAME}'..."
echo "Hydra run dir (host-visible, inside the repo): ${REPO_ROOT}/${OUT_DIR}"

# No --rm: if the container exits unexpectedly (e.g. crashes before Hydra even creates
# the run dir), `docker logs` still works afterward to see why -- `docker rm` it
# yourself once you're done inspecting. A bare filename passed to `-p` (e.g. "train.py")
# resolves against Isaac Lab's OWN scripts/ tree, not the current directory -- it picked
# up Isaac Lab's scripts/reinforcement_learning/train.py instead of ours the first time
# this was tried -- so this always uses the absolute path.
#
# --ipc=host: Docker's default /dev/shm is only 64MB, which the PyTorch DataLoader's
# worker processes (num_workers=3 for train, 1 for val -- see train_maniflow_usb.yaml)
# use to pass batch tensors back to the main process. That filled up and crashed training
# 97% through epoch 0 with "unable to allocate shared memory ... No space left on device"
# -- nothing to do with the dataset/model, just the container's IPC namespace being too
# small. --ipc=host shares the host's (effectively unlimited) /dev/shm instead.
docker run -d --gpus all \
  --name "${CONTAINER_NAME}" \
  --ipc=host \
  -v "${REPO_ROOT}:/workspace/manifeel" \
  -e WANDB_API_KEY="${WANDB_API_KEY}" \
  --entrypoint bash \
  "${IMAGE}" \
  -c "cd /workspace/manifeel && uv run isaaclab -p /workspace/manifeel/train.py ${HYDRA_ARGS[*]} ${EXTRA_HYDRA_ARGS}"

echo ""
echo "Started. Useful commands:"
echo "  docker logs -f ${CONTAINER_NAME}                 # follow training output"
echo "  tail -f ${OUT_DIR}/logs.json.txt                 # same, from the host (no docker needed)"
echo "  docker stop ${CONTAINER_NAME}                    # stop training early"
echo "  docker ps -a --filter name=${CONTAINER_NAME}      # check it's still running (or see why it exited)"
echo "  docker rm ${CONTAINER_NAME}                       # clean up after it's done/crashed and you've seen the logs"
