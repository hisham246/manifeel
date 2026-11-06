"""Interactive 3D viewer for the USB insertion task.

Opens the Omniverse Kit GUI window and steps the environment with small random
actions in a loop until you close the window. Use the mouse to orbit/pan/zoom
the viewport like any 3D scene.

Usage:
    ./isaaclab.sh -p ../manifeel/manifeel_isaaclab/view_usb_insertion.py
(run from the IsaacLab repo root, with the manifeel-isaaclab conda env active)
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args_cli = parser.parse_args(["--visualizer", "kit"])

import sys  # noqa: E402

sys.path.insert(0, "/home/aist/hisham/contact_rich_sim/manifeel")

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import manifeel_isaaclab.tasks.usb_insertion  # noqa: E402, F401 -- triggers gym.register
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

env_cfg = parse_env_cfg("Manifeel-USB-Insertion-Direct", device="cuda:0", num_envs=1)

with launch_simulation(env_cfg, args_cli):
    env = gym.make("Manifeel-USB-Insertion-Direct", cfg=env_cfg)
    env.reset()

    print("Viewport open -- press Ctrl+C in this terminal to exit (closing the window may not cleanly stop the process).")
    try:
        while True:
            # Small random actions so there's visible motion to inspect, not a frozen frame.
            action = (torch.rand(env.action_space.shape, device=env.unwrapped.device) - 0.5) * 0.05
            env.step(action)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
