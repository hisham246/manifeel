"""Isaac Lab env wrapper matching MultipleIsaacEnvWrapper's public contract.

Mirrors manifeel/manifeel/envs/vistac_isaacgym_multiple_env_wrapper.py's interface
(reset()/step()/render()/observation_space/action_space/num_envs) closely enough to
be a near drop-in replacement for MultiStepWrapper/VideoRecordingWrapper -- both of
those files are reused completely unmodified (see manifeel/manifeel/gym_util/).

Differences from the original wrapper, and why:

  - Constructor takes (shape_meta, num_envs, device) directly instead of a single
    isaacgym-style cfg object with rl_device/sim_device/graphics_device_id/headless/
    capture_video/force_render fields. Isaac Lab's launch_simulation() context
    manager owns the whole Kit/Isaac Sim process lifecycle and must wrap the entire
    training run (not just this wrapper's lifetime) -- the caller is expected to
    have already entered that context before constructing this wrapper.

  - reward is the real value UsbInsertionEnv computes (SDF + success term), not
    np.copy(reset) -- the original wrapper's reward was a confirmed no-op bug
    (verified by reading vistac_isaacgym_multiple_env_wrapper.py directly). Nothing
    in manifeel trains on this value (diffusion policy trains from the offline
    dataset only); it's used purely for eval-time logging, so using the real reward
    is a strict improvement with no behavior to preserve.

  - vision_obs_keys covers 'front' and 'tactile_force_field_right' (whichever the
    caller's shape_meta asks for) -- not the full multi-view set the original's
    render() concatenated side by side ('client', 'wrist_2', tactile RGB/depth, etc).
    tactile_force_field_right is treated as a vision-shaped key (moveaxis to
    channel-first) even though it's raw force data, not RGB -- matching how training
    treats it (manifeel/model/dual_resnet_obs_encoder.py's DualResNetObsEncoder runs
    it through its own ResNet-18 instance, same as 'front'). It skips the uint8/255
    normalization step real cameras need (see _transform_obs_data below).

  - Builds observation_space/action_space from the OLD `gym` package's spaces, not
    gymnasium's: MultiStepWrapper's repeated_space()/repeated_box() do
    isinstance(space, spaces.Box) checks against `gym.spaces.Box` specifically --
    a different class from gymnasium.spaces.Box even though structurally identical,
    so reusing Isaac Lab's gymnasium-native space objects directly would silently
    fail that isinstance check.
"""

import gym
import gymnasium
import numpy as np
import torch
from gym import spaces

import manifeel_isaaclab.tasks.usb_insertion  # noqa: F401 -- triggers gym.register


class IsaacLabUsbEnvWrapper:

    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    TASK_ID = "Manifeel-USB-Insertion-Direct"

    def __init__(self, shape_meta, num_envs=1, device="cuda:0"):
        """
        Precondition: must be constructed from within an active
        isaaclab.app.launch_simulation(...) context (Isaac Sim/Kit already running)
        -- see manifeel_isaaclab/view_usb_insertion.py for the pattern.
        """
        from isaaclab_tasks.utils import parse_env_cfg

        self.device = device
        self.render_cache = None
        self._num_envs = num_envs

        env_cfg = parse_env_cfg(self.TASK_ID, device=device, num_envs=num_envs)
        self.envs = gymnasium.make(self.TASK_ID, cfg=env_cfg)

        # obtain task observation keys from the shape_meta (same derivation as the
        # original wrapper), split into real cameras vs the force-field tactile key --
        # both are shape_meta type 'rgb' (matching how training's DualResNetObsEncoder
        # treats them), but only real cameras are uint8 images needing /255 (see
        # _transform_obs_data) or make sense to include in render()'s video preview.
        obs_shape_meta = shape_meta["obs"]
        camera_keys = []
        force_field_keys = []
        lowdim_keys = []
        for key, attr in obs_shape_meta.items():
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                if "force_field" in key:
                    force_field_keys.append(key)
                else:
                    camera_keys.append(key)
            elif obs_type == "low_dim":
                lowdim_keys.append(key)
        self.task_obs_keys = camera_keys + force_field_keys + lowdim_keys
        self.camera_keys = camera_keys
        self.force_field_keys = force_field_keys
        # render() only visualizes real camera feeds -- tactile_force_field_right isn't
        # photometric RGB (arbitrary-range force values), so including it there would
        # produce a meaningless clipped image.
        self.vision_obs_keys = camera_keys

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def num_acts(self) -> int:
        return self.envs.action_space.shape[-1]

    @property
    def num_obs(self) -> int:
        return int(sum(np.prod(v.shape) for v in self.observation_space.spaces.values()))

    @property
    def observation_space(self):
        """Per-env (unbatched) space, matching the original: MultiStepWrapper adds
        the n_obs_steps dimension itself, not num_envs."""
        spaces_dict = {}
        if "wrist" in self.task_obs_keys:
            spaces_dict["wrist"] = spaces.Box(low=0.0, high=1.0, shape=(3, 256, 256), dtype=np.float32)
        if "front" in self.task_obs_keys:
            spaces_dict["front"] = spaces.Box(low=0.0, high=1.0, shape=(3, 256, 256), dtype=np.float32)
        if "tactile_force_field_right" in self.task_obs_keys:
            # Raw normal/shear force values, not a [0,1]-bounded image -- unbounded box.
            spaces_dict["tactile_force_field_right"] = spaces.Box(
                low=np.float32(-np.inf), high=np.float32(np.inf), shape=(3, 10, 14), dtype=np.float32
            )
        if "state" in self.task_obs_keys:
            spaces_dict["state"] = spaces.Box(
                low=np.float32(-np.inf), high=np.float32(np.inf), shape=(7,), dtype=np.float32
            )
        return spaces.Dict(spaces_dict)

    @property
    def action_space(self):
        act_dim = self.envs.action_space.shape[-1]
        return spaces.Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(act_dim,), dtype=np.float32)

    def seed(self, seed=None):
        # Isaac Lab envs seed via the launch_simulation/parse_env_cfg path, not a
        # per-wrapper call; accepted here only for interface parity with the
        # original wrapper's seed() method.
        pass

    def reset(self):
        obs, info = self.envs.reset()
        obs = self._transform_obs_data(obs)
        self.render_cache = obs
        return self._apply_obs_by_keys(obs)

    def step(self, actions):
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        actions = actions.to(dtype=torch.float32, device=self.device)

        obs, reward, terminated, truncated, info = self.envs.step(actions)
        done = torch.logical_or(terminated, truncated)

        obs = self._transform_obs_data(obs)
        self.render_cache = obs
        obs = self._apply_obs_by_keys(obs)

        reward = reward.detach().cpu().numpy()
        done = done.detach().cpu().numpy()
        info = {k: v.detach().cpu().numpy() for k, v in info.items() if isinstance(v, torch.Tensor)}

        return obs, reward, done, info

    def render(self, mode="rgb_array"):
        if self.render_cache is None:
            raise RuntimeError("Must run reset or step before render.")
        img_list = []
        for obs_view in self.vision_obs_keys:
            imgs = self.render_cache[obs_view]  # (num_envs, 3, H, W) float32 [0, 1]
            imgs = np.moveaxis(imgs, 1, -1)  # (num_envs, H, W, 3)
            imgs = (imgs * 255).astype(np.uint8)
            img_list.append(imgs)
        return np.concatenate(img_list, axis=2)

    def _transform_obs_data(self, obs):
        tf_obs = {}
        for key in self.task_obs_keys:
            if key not in obs:
                continue
            value = obs[key].detach().cpu().numpy()
            if key in self.camera_keys:
                # (num_envs, H, W, 3) uint8 -> (num_envs, 3, H, W) float32 [0, 1].
                # The original pipeline normalizes at the task level
                # (compute_observations_dict_obs); UsbInsertionEnv returns raw
                # uint8 from the camera sensor, so this wrapper does it instead.
                value = value.astype(np.float32) / 255.0
                value = np.moveaxis(value, -1, 1)
            elif key in self.force_field_keys:
                # (num_envs, 10, 14, 3) raw normal/shear force values -- channel-last
                # to channel-first only, no /255 (not image intensities).
                value = value.astype(np.float32)
                value = np.moveaxis(value, -1, 1)
            tf_obs[key] = value
        return tf_obs

    def _apply_obs_by_keys(self, obs):
        return {key: obs[key] for key in self.task_obs_keys if key in obs}

    def close(self):
        self.envs.close()
