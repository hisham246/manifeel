"""Isaac Lab Direct-RL environment for the USB plug/socket insertion task.

Subclasses AssemblyEnv (action space, physics, SDF-reward machinery all unchanged --
confirmed earlier this session that AutoMate's 6D delta-pose task-space action
decomposition is an exact structural match for manifeel's existing policy output).

Two classes of override:

1. _get_observations(): adds a 'wrist' RGB camera image and a 'state' =
   concat(ee_pos, ee_quat) vector, matching the exact obs_dict shape
   manifeel/manifeel/envs/vistac_isaacgym_multiple_env_wrapper.py expects.

2. Reset convention: AutoMate's own randomize_initial_state() places the plug using a
   disassembly-curriculum convention driven by recorded demonstration data
   (plug_grasps.json, disassembly_dist.json, disassemble_traj.json) specific to AutoMate's
   own asset pipeline -- data we don't have for this asset. _load_assembly_info() and
   _load_disassembly_data() are overridden with physically-reasonable placeholders (not
   fabricated "real" data) so __init__ doesn't crash; randomize_initial_state() and
   _update_rew_buf() are overridden to use manifeel's actual reset convention (ported
   from manifeel-isaacgymenvs' TacSLTaskUSB.reset_idx/_randomize_gripper_pose/_reset_object)
   and to drop only the one reward term (imitation/soft-DTW) that depends on the missing
   demonstration trajectory -- the SDF-based reward term is kept as-is, using real mesh
   data (see UsbInsertion.assembly_dir in usb_assembly_tasks_cfg.py).
"""

import torch

from isaaclab.utils import index_fill_
from isaaclab.utils.math import combine_frame_transforms, quat_from_euler_xyz

from isaaclab_tasks.contrib.automate import industreal_algo_utils as industreal_algo
from isaaclab_tasks.contrib.automate.assembly_env import AssemblyEnv

from .usb_assembly_env_cfg import UsbInsertionEnvCfg


class UsbInsertionEnv(AssemblyEnv):
    cfg: UsbInsertionEnvCfg

    def _get_observations(self):
        obs = super()._get_observations()

        # ProxyArray wrapper -> actual torch.Tensor, shape (num_envs, H, W, 3)
        obs["wrist"] = self.scene["wrist_camera"].data.output["rgb"].torch

        # Matches vistac_isaacgym_multiple_env_wrapper.py's
        # state = np.concatenate([ee_pos, ee_quat], axis=1), shape (num_envs, 7)
        obs["state"] = torch.cat([self.fingertip_midpoint_pos, self.fingertip_midpoint_quat], dim=-1)

        # Per-finger net contact force against the held USB plug, shape (num_envs, 6) =
        # concat(left_finger_force_xyz, right_finger_force_xyz). num_sensors=1 per sensor
        # (single prim_path match), so .squeeze(1) drops that axis.
        left_force = self.scene["left_finger_contact"].data.net_normal_forces_w.torch.squeeze(1)
        right_force = self.scene["right_finger_contact"].data.net_normal_forces_w.torch.squeeze(1)
        obs["contact_force"] = torch.cat([left_force, right_force], dim=-1)

        return obs

    # ------------------------------------------------------------------
    # Placeholder data for AutoMate's own demonstration-driven machinery.
    # ------------------------------------------------------------------

    def _load_assembly_info(self):
        """No plug-grasp library or recorded disassembly distance for this asset.

        plug_grasps is unused -- our randomize_initial_state() override below doesn't
        call _move_gripper_to_grasp_pose (which is the only consumer). disassembly_dists
        IS still read by automate_algo.check_plug_inserted_in_socket() for success
        detection and by _get_curriculum_info()/the SDF reward's SBC scale factor, so it
        gets a physically-reasonable placeholder (the socket's depth) rather than zero.
        """
        plug_grasps = torch.zeros((self.num_envs, 7), device=self.device)
        plug_grasps[:, 6] = 1.0  # valid identity quat (x,y,z,w) in the unused columns 3:7

        disassembly_dists = torch.full(
            (self.num_envs,), self.cfg_task.fixed_asset_cfg.height, device=self.device
        )
        return plug_grasps, disassembly_dists

    def _load_disassembly_data(self):
        """No recorded disassembly trajectory for this asset -- imitation reward is
        dropped entirely in _update_rew_buf() below, so eef_pos_traj is never read."""
        self.eef_pos_traj = None

    def _update_rew_buf(self, curr_successes):
        """Same as AssemblyEnv._update_rew_buf, minus the imitation/soft-DTW term
        (which needs manifeel/eef_pos_traj data we don't have). SDF reward is kept
        as-is -- it uses real mesh data via UsbInsertion.assembly_dir/obj_path."""
        sdf_reward = industreal_algo.get_sdf_reward(
            self.plug_mesh,
            self.plug_sample_points,
            self.held_pos,
            self.held_quat,
            self.fixed_pos,
            self.fixed_quat,
            self.wp_device,
            self.device,
        )
        return self.cfg_task.sdf_rwd_scale * sdf_reward + curr_successes.clone().float()

    # ------------------------------------------------------------------
    # Reset convention, ported from manifeel-isaacgymenvs' TacSLTaskUSB
    # (reset_idx / _randomize_gripper_pose / _reset_object), replacing AutoMate's
    # disassembly-curriculum convention for the gripper+plug placement only.
    # Socket placement (randomize_fixed_initial_state) and the grasp-closing loop are
    # reused unmodified from AssemblyEnv -- see usb_assembly_tasks_cfg.py's
    # fixed_asset_init_pos_noise/_orn_range_deg and UsbInsertionEnvCfg.ctrl.reset_joints
    # for the numbers that make those generic methods reproduce manifeel's convention.
    # ------------------------------------------------------------------

    def randomize_initial_state(self, env_ids):
        # Disable gravity globally and temporarily (prevents the plug from falling before
        # it's grasped) -- matches TacSLTaskUSB.reset_idx's disable_gravity()/enable_gravity().
        self.sim.physics_manager.set_gravity((0.0, 0.0, 0.0))

        self.randomize_fixed_initial_state(env_ids)

        fixed_tip_pos_local = torch.zeros_like(self.fixed_pos)
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.height
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.base_height
        fixed_tip_pos, _ = combine_frame_transforms(
            self.fixed_pos, self.fixed_quat, fixed_tip_pos_local, self.identity_quat
        )
        self.fixed_pos_obs_frame[:] = fixed_tip_pos

        self._randomize_gripper_pose_manifeel(env_ids)
        self._place_plug_in_gripper_manifeel(env_ids)

        # Close hand (unchanged from AssemblyEnv.randomize_initial_state).
        reset_task_prop_gains = torch.tensor(self.cfg.ctrl.reset_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        reset_rot_deriv_scale = self.cfg.ctrl.reset_rot_deriv_scale
        self._set_gains(reset_task_prop_gains, reset_rot_deriv_scale)

        self.step_sim_no_action()

        grasp_time = 0.0
        while grasp_time < 1.0:
            index_fill_(self.ctrl_target_joint_pos[:, 7:], env_ids, 0.0)  # Close gripper.
            self.ctrl_target_gripper_dof_pos = 0.0
            self.move_gripper_in_place(ctrl_target_gripper_dof_pos=0.0)
            self.step_sim_no_action()
            grasp_time += self.sim.get_physics_dt()

        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        self.actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)
        self.fixed_pos_action_frame[:] = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise

        self.ee_angvel_fd[:, :] = 0.0
        self.ee_linvel_fd[:, :] = 0.0

        self._set_gains(self.default_gains)

        self.sim.physics_manager.set_gravity(tuple(self.cfg.sim.gravity))

    def _randomize_gripper_pose_manifeel(self, env_ids):
        """Move the gripper to an absolute target pose (TacSLTaskUSB.yaml
        randomize.fingertip_midpoint_pos_initial/_noise, _rot_initial/_noise), via
        AssemblyEnv's existing IK convergence loop (set_pos_inverse_kinematics)."""
        pos_initial = torch.tensor(self.cfg_task.fingertip_midpoint_pos_initial, device=self.device)
        pos_noise_scale = torch.tensor(self.cfg_task.fingertip_midpoint_pos_noise, device=self.device)
        pos_unit_noise = 2 * (torch.rand((self.num_envs, 3), device=self.device) - 0.5)
        self.ctrl_target_fingertip_midpoint_pos = (
            pos_initial.unsqueeze(0).repeat(self.num_envs, 1) + pos_unit_noise * pos_noise_scale
        )

        rot_initial = torch.tensor(self.cfg_task.fingertip_midpoint_rot_initial, device=self.device)
        rot_noise_scale = torch.tensor(self.cfg_task.fingertip_midpoint_rot_noise, device=self.device)
        rot_unit_noise = 2 * (torch.rand((self.num_envs, 3), device=self.device) - 0.5)
        target_euler = rot_initial.unsqueeze(0).repeat(self.num_envs, 1) + rot_unit_noise * rot_noise_scale
        self.ctrl_target_fingertip_midpoint_quat = quat_from_euler_xyz(
            target_euler[:, 0], target_euler[:, 1], target_euler[:, 2]
        )

        self.set_pos_inverse_kinematics(env_ids)
        self.step_sim_no_action()

    def _place_plug_in_gripper_manifeel(self, env_ids):
        """Place the plug relative to wherever the gripper actually ended up (not an
        absolute pose), ported from TacSLTaskUSB._reset_object."""
        socket_height = self.cfg_task.fixed_asset_cfg.height
        plug_length = self.cfg_task.held_asset_cfg.height
        peg_in_hand_buffer = self.cfg_task.peg_in_hand_buffer

        min_tip_dist = socket_height * 1.1 * peg_in_hand_buffer
        max_tip_dist = plug_length * 0.8
        tip_dist_range_center = (min_tip_dist + max_tip_dist) / 2.0
        tip_dist_range_mag = max_tip_dist - min_tip_dist

        noise = 2 * (torch.rand((self.num_envs, 3), device=self.device) - 0.5)
        z_sampled = (
            tip_dist_range_center
            + noise[:, 2] * tip_dist_range_mag / 2.0 * self.cfg_task.plug_pos_z_in_gripper_noise_multiplier
        )

        ee_to_plug_tip_pos_local = torch.zeros((self.num_envs, 3), device=self.device)
        ee_to_plug_tip_pos_local[:, 2] = z_sampled - plug_length
        xy_noise_scale = torch.tensor(self.cfg_task.plug_pos_in_gripper_noise_xy, device=self.device)
        ee_to_plug_tip_pos_local[:, :2] = noise[:, :2] * xy_noise_scale

        world_to_plug_tip_pos, world_to_plug_tip_quat = combine_frame_transforms(
            self.fingertip_midpoint_pos, self.fingertip_midpoint_quat, ee_to_plug_tip_pos_local, self.identity_quat
        )

        rot_noise = 2 * (torch.rand((self.num_envs, 3), device=self.device) - 0.5)
        rot_noise_scale = torch.tensor(self.cfg_task.plug_noise_rot_in_gripper, device=self.device)
        rot_noise = rot_noise * rot_noise_scale
        ee_to_plug_tip_rot_quat = quat_from_euler_xyz(rot_noise[:, 0], rot_noise[:, 1], rot_noise[:, 2])

        zero_translation = torch.zeros_like(world_to_plug_tip_pos)
        world_to_plug_tip_pos, world_to_plug_tip_quat = combine_frame_transforms(
            world_to_plug_tip_pos, world_to_plug_tip_quat, zero_translation, ee_to_plug_tip_rot_quat
        )

        plug_tip_to_base_local = torch.zeros((self.num_envs, 3), device=self.device)
        plug_tip_to_base_local[:, 2] = plug_length
        world_to_plug_base_pos, world_to_plug_base_quat = combine_frame_transforms(
            world_to_plug_tip_pos, world_to_plug_tip_quat, plug_tip_to_base_local, self.identity_quat
        )

        flip_z_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        world_to_plug_base_pos, world_to_plug_base_quat = combine_frame_transforms(
            world_to_plug_base_pos, world_to_plug_base_quat, zero_translation, flip_z_quat
        )

        held_state = torch.cat([world_to_plug_base_pos, world_to_plug_base_quat], dim=-1)
        self._held_asset.write_root_pose_to_sim_index(root_pose=held_state, env_ids=env_ids)
        self._held_asset.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((self.num_envs, 6), device=self.device), env_ids=env_ids
        )
        self._held_asset.reset()
        self.step_sim_no_action()
