"""
Usage:
Training:
$ python train.py \
  --config-dir=. \
  --config-name=train_diffusion_vision_tactile_workspace.yaml \
  training.seed=42 \
  training.device=cuda:0 \
  hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

"""

# isaacgym must be imported before torch if it's going to be used at all (internal
# symbol-loading order requirement) -- but it can't even be imported on Blackwell/RTX
# 50-series GPUs (no compiled kernels for sm_120, since IsaacGym predates that
# architecture by ~3 years), so this can't be unconditional anymore. Workspaces that
# still route through the old IsaacGym env wrapper need to run on hardware where this
# import actually succeeds; ones that don't (e.g. ManiFlow training, pure offline from
# a saved dataset) work fine either way.
try:
    import isaacgym  # noqa: F401
except ImportError:
    pass


import os
os.environ["HYDRA_FULL_ERROR"] = "1"


import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        './manifeel','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()