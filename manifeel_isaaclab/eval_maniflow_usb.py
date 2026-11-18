"""Roll out a trained ManiFlowUsbPolicy checkpoint against the Isaac Lab USB insertion
task and record a video.

Launch pattern is ported from manifeel_isaaclab/view_usb_insertion.py (Isaac Lab
launcher args parsed first, launch_simulation(env_cfg, args_cli) context entered
before anything Isaac-Sim-specific is imported/constructed). Checkpoint loading is
ported from eval.py's main() (hydra.compose + workspace.load_payload), minus the
`import isaacgym` line, which doesn't apply here and can't even import on this
machine's GPU (see train.py's own try/except around it).

Usage (run inside the manifeel-isaaclab Docker container, same `uv run isaaclab -p`
pattern as scripts/run_maniflow_usb.sh -- bare filenames resolve against Isaac Lab's
own scripts/ tree, so use the absolute path):

    uv run isaaclab -p /workspace/manifeel/manifeel_isaaclab/eval_maniflow_usb.py \\
        --checkpoint data/outputs/maniflow_usb_40demos/44/checkpoints/latest_epoch1050.ckpt \\
        --output_dir data/outputs/maniflow_usb_40demos/44/eval_output \\
        --num_envs 1 \\
        --device cuda:0

Add --visualizer kit for an interactive Kit viewport instead of a purely headless run
(matches view_usb_insertion.py's usage) -- a video is recorded either way.
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True, help="Path to a .ckpt file, e.g. latest_epoch1050.ckpt")
parser.add_argument("--output_dir", required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--max_steps", type=int, default=200)
parser.add_argument("--test_seed", type=int, default=10000)
add_launcher_args(parser)
args_cli = parser.parse_args()

import sys  # noqa: E402

sys.path.insert(0, "/home/aist/hisham/contact_rich_sim/manifeel")

import json  # noqa: E402
import os  # noqa: E402
import pathlib  # noqa: E402

import dill  # noqa: E402
import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import wandb  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import manifeel_isaaclab.tasks.usb_insertion  # noqa: E402, F401 -- triggers gym.register
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK_ID = "Manifeel-USB-Insertion-Direct"

# Needed before entering launch_simulation, same as view_usb_insertion.py.
env_cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=args_cli.num_envs)

with launch_simulation(env_cfg, args_cli):
    # Deferred past launch_simulation: these pull in torch/CUDA extensions and
    # manifeel/maniflow code that assume Isaac Sim's own torch build is already active.
    from manifeel.env_runner.isaaclab_usb_maniflow_runner import IsaacLabUsbManiFlowRunner
    from manifeel.envs.isaaclab_usb_env_wrapper import IsaacLabUsbEnvWrapper
    from manifeel.gym_util.multistep_wrapper import MultiStepWrapper
    from manifeel.gym_util.video_recording_wrapper import VideoRecordingWrapper

    OmegaConf.register_new_resolver("eval", eval, replace=True)

    output_dir = args_cli.output_dir
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    hydra.initialize(config_path=str(pathlib.Path(__file__).parent.parent.joinpath(
        "manifeel", "config")), version_base=None)
    cfg = hydra.compose(config_name="train_maniflow_usb.yaml")

    payload = torch.load(open(args_cli.checkpoint, "rb"), pickle_module=dill)
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(args_cli.device)
    policy.to(device)
    policy.eval()

    shape_meta = OmegaConf.to_container(cfg.shape_meta, resolve=True)

    n_records = min(args_cli.num_envs, 6)
    fps = 10
    steps_per_render = max(10 // fps, 1)
    env = MultiStepWrapper(
        VideoRecordingWrapper(
            IsaacLabUsbEnvWrapper(shape_meta, num_envs=args_cli.num_envs, device=args_cli.device),
            output_dir=output_dir,
            n_records=n_records,
            fps=fps,
            crf=22,
            file_paths=None,
            steps_per_render=steps_per_render,
        ),
        n_obs_steps=cfg.n_obs_steps,
        n_action_steps=cfg.n_action_steps,
        max_episode_steps=args_cli.max_steps,
    )

    runner = IsaacLabUsbManiFlowRunner(
        output_dir=output_dir,
        env=env,
        max_steps=args_cli.max_steps,
        test_start_seed=args_cli.test_seed,
    )
    runner_log = runner.run(policy)

    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        elif isinstance(value, np.integer):
            json_log[key] = int(value)
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, "eval_log.json")
    json.dump(json_log, open(out_path, "w"), indent=2, sort_keys=True)
    print(f"Eval log written to {out_path}")
