"""ManiFlowUsbPolicy rollout runner against the Isaac Lab USB insertion task.

Rollout loop is ported from manifeel/env_runner/vistac_pih_runner.py's ManifeelRunner
(same policy.predict_action()/env.step() contract -- ManiFlowUsbPolicy implements the
same predict_action(obs_dict) -> {'action': ...} interface the diffusion_policy
baseline does, see train_maniflow_usb_workspace.py's own predict_action call). The
tactile_camera_taxim resize block from that version is dropped: 'front' and
'tactile_force_field_right' already come out of IsaacLabUsbEnvWrapper at the exact
shapes training used (256x256 / 10x14), no resize needed.

Unlike ManifeelRunner, this class does NOT construct its own env from a cfg via
hydra -- IsaacLabUsbEnvWrapper's constructor must run inside an active
isaaclab.app.launch_simulation(...) context (Isaac Sim/Kit already running), which is
the calling script's responsibility (see manifeel_isaaclab/eval_maniflow_usb.py). This
runner just takes an already-built, already-wrapped env (MultiStepWrapper +
VideoRecordingWrapper around IsaacLabUsbEnvWrapper, same stacking ManifeelRunner uses)
and drives the rollout loop.
"""

import collections
import time

import numpy as np
import torch
import tqdm
import wandb

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class IsaacLabUsbManiFlowRunner(BaseImageRunner):
    def __init__(self,
            output_dir: str,
            env,
            max_steps=200,
            test_start_seed=10000,
            tqdm_interval_sec=5.0,
        ):
        super().__init__(output_dir)
        self.env = env
        self.max_steps = max_steps
        self.test_seed = test_start_seed
        self.tqdm_interval_sec = tqdm_interval_sec

    def run(self, policy: BaseImagePolicy):
        device = policy.device
        env = self.env
        n_envs = env.num_envs

        print("Number of envs runner: ", n_envs)

        env.seed(self.test_seed)
        obs = env.reset()
        time.sleep(2)
        policy.reset()

        pbar = tqdm.tqdm(total=self.max_steps, desc="Eval IsaacLabUsbManiFlowRunner",
            leave=False, mininterval=self.tqdm_interval_sec)
        done = False

        while not done:
            np_obs_dict = dict(obs)
            obs_dict = dict_apply(np_obs_dict,
                lambda x: torch.from_numpy(x).to(device=device))

            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)

            np_action_dict = dict_apply(action_dict,
                lambda x: x.detach().to('cpu').numpy())
            action = np_action_dict['action']

            obs, reward, done, info = env.step(action)
            done = np.all(done)

            pbar.update(action.shape[1])
        pbar.close()

        all_rewards = env.get_rewards()
        all_video_paths = env.render()
        _ = env.reset()

        max_rewards = collections.defaultdict(list)
        log_data = dict()
        for i in range(n_envs):
            prefix = 'test/'
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)

            video_path = all_video_paths[i]
            if video_path is not None:
                log_data[prefix + f'sim_video_{i}'] = wandb.Video(video_path)

        for prefix, value in max_rewards.items():
            log_data[prefix + 'mean_score'] = np.mean(value)

        return log_data
