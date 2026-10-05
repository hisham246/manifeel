#!/bin/bash
set -euo pipefail

# -------------------------
# User-editable parameters
# -------------------------
SEED=${SEED:-44}
NUM_DEMOS=${NUM_DEMOS:-50}
NUM_EPOCH=${NUM_EPOCH:-1000}

DATASET_PATH=${DATASET_PATH:-data/usb_quan_Aug05}

TASK_NAME=${TASK_NAME:-vision_wrist}  # vision_wrist | vistac_wrist | visff_wrist | vision_front | vistac_front | visff_front
INPUT_TYPE=${INPUT_TYPE:-vision}   # vision | vistac | tacff
ENV_TAG=${ENV_TAG:-usb_wrist_0805}
LOG_NAME=${LOG_NAME:-dp_usb_tacff}

# Optional Hydra overrides
IMAGENET_NORM=${IMAGENET_NORM:-false}   # set true for tacff runs if needed
ACTION_SHAPE=${ACTION_SHAPE:-}          # set to 7 for gripper tasks, leave empty otherwise

# Conda env with Python 3.12 + Blackwell-capable torch + Isaac Sim/Isaac Lab
CONDA_ENV=${CONDA_ENV:-manifeel-isaaclab}

# wandb logging mode: "offline" avoids requiring `wandb login`/an API key for local runs.
# Set to "online" if you've already run `wandb login` and want live dashboards.
LOGGING_MODE=${LOGGING_MODE:-offline}

# -------------------------
# Derived names
# -------------------------
EXP_NAME="${INPUT_TYPE}_${ENV_TAG}_${NUM_DEMOS}"
OUT_DIR="data/outputs/${EXP_NAME}/${SEED}"

# Repo root resolution (script works from any directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# -------------------------
# Build Hydra args
# -------------------------
HYDRA_ARGS=(
  "--config-name=train_diffusion_workspace.yaml"
  "task=${TASK_NAME}"
  "exp_name=${EXP_NAME}"
  "dataset_path=${DATASET_PATH}"
  "training.seed=${SEED}"
  "training.num_epochs=${NUM_EPOCH}"
  "task.dataset.max_train_episodes=${NUM_DEMOS}"
  "task.env_runner._target_=manifeel.env_runner.null_runner.NullEnvRunner"
  "hydra.run.dir=${OUT_DIR}"
  "logging.project=${LOG_NAME}"
  "logging.mode=${LOGGING_MODE}"
)

if [[ "${IMAGENET_NORM}" == "true" ]]; then
  HYDRA_ARGS+=("policy.obs_encoder.imagenet_norm=True")
fi

if [[ -n "${ACTION_SHAPE}" ]]; then
  HYDRA_ARGS+=("task.shape_meta.action.shape=[${ACTION_SHAPE}]")
fi

# -------------------------
# Run natively in a conda env (no apptainer/IsaacGym needed for this path)
# -------------------------
source /home/aist/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"
python train_no_sim.py "${HYDRA_ARGS[@]}"
