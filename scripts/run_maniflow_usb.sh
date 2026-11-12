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

docker run -d --rm --gpus all \
  --name "${CONTAINER_NAME}" \
  -v "${REPO_ROOT}:/workspace/manifeel" \
  --entrypoint bash \
  "${IMAGE}" \
  -c "cd /workspace/manifeel && uv run isaaclab -p train.py ${HYDRA_ARGS[*]}"

echo ""
echo "Started. Useful commands:"
echo "  docker logs -f ${CONTAINER_NAME}                 # follow training output"
echo "  tail -f ${OUT_DIR}/logs.json.txt                 # same, from the host (no docker needed)"
echo "  docker stop ${CONTAINER_NAME}                    # stop training early"
echo "  docker ps --filter name=${CONTAINER_NAME}         # check it's still running"
