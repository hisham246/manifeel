"""Replay a recorded ManiFeel USB-insertion dataset in Isaac Gym and re-record it with extra observations.

Each source episode i was recorded with env.seed(i) + env.reset() and teleoperated delta-pose actions, so
replaying the stored actions from the same seed reproduces the episode to ~1 mm (contact-rich steps can
drift slightly; per-episode drift is written to <dst>_report.json so diverged episodes can be dropped).

All original keys are re-recorded together (so modalities stay consistent) plus the contact-force keys enabled by
`task.rl.add_contact_force_plug_decomposed` (set in TacSLTaskUSB.yaml):
    plug_socket_force, plug_left_elastomer_force, plug_right_elastomer_force   [3] each, world frame, Newtons

Optional drift correction (--track-gain K > 0): at every step the recorded action is augmented with a term that pulls the
end effector back toward the recorded pose (the env's actions are *relative* to the current pose, so without it any
contact-induced error persists):
    action' = clip(action + K * [pos_err / pos_scale, rot_err / rot_scale], -1, 1)
The dataset still stores the ORIGINAL recorded action as the label.

Run inside the container, from the repo root (import isaacgym must precede torch):
    python scripts/regenerate_dataset.py --src data/usb_quan_Aug05 --dst data/usb_quan_Aug05_contact
"""
import argparse
import json
import os
import sys
import time

import isaacgym  # noqa: F401  (must be imported before torch)
import numpy as np
import torch
import zarr
from hydra import compose, initialize_config_dir

from diffusion_policy.common.replay_buffer import ReplayBuffer
from isaacgymenvs.tasks.tacsl.tacsl_task_USB import TacSLTaskUSB
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.utils import set_seed
from isaacgym import torch_utils

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OBS_KEYS = ['wrist', 'wrist_2', 'front', 'side', 'left_tactile_camera_taxim', 'right_tactile_camera_taxim',
            'tactile_force_field_right', 'tactile_depth_right']
CONTACT_KEYS = ['plug_socket_force', 'plug_left_elastomer_force', 'plug_right_elastomer_force']


def tracking_correction(env, ref_pos, ref_quat, pos_scale, rot_scale):
    """Action-space correction that moves the current end-effector pose toward (ref_pos, ref_quat)."""
    cur_pos, cur_quat = env.fingertip_midpoint_pos, env.fingertip_midpoint_quat
    pos_corr = (ref_pos - cur_pos) / pos_scale
    q_err = torch_utils.quat_mul(ref_quat, torch_utils.quat_conjugate(cur_quat))  # world-frame rotation cur -> ref
    q_err = torch.where(q_err[:, 3:4] < 0, -q_err, q_err)  # shortest rotation
    xyz, w = q_err[:, :3], q_err[:, 3]
    sin_half = xyz.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half.squeeze(-1), w).unsqueeze(-1)
    rotvec = xyz * torch.where(sin_half > 1e-8, angle / sin_half.clamp_min(1e-8), torch.full_like(sin_half, 2.0))
    rot_corr = rotvec / rot_scale
    return torch.cat([pos_corr, rot_corr], dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/usb_quan_Aug05')
    ap.add_argument('--dst', default='data/usb_quan_Aug05_contact')
    ap.add_argument('--config-name', default='isaacgym_config_usb')
    ap.add_argument('--track-gain', type=float, default=0.0,
                    help='K of the pull-back-to-recorded-pose correction; 0 = pure open-loop replay of the recorded actions')
    ap.add_argument('--gains-from-report', default=None,
                    help='use each episode\'s track_gain from an existing <dataset>_report.json (overrides --track-gain)')
    ap.add_argument('--dry-run', action='store_true',
                    help='replay and log drift/success only; do not write a dataset (writes the report to <dst>_report.json)')
    ap.add_argument('--episodes', type=int, nargs='*', default=None, help='source episode indices (default: all)')
    args = ap.parse_args()

    src = zarr.open(args.src, 'r')
    ends = src['meta/episode_ends'][:]
    starts = np.concatenate([[0], ends[:-1]])
    episodes = args.episodes if args.episodes else list(range(len(ends)))
    if os.path.exists(args.dst) and not args.dry_run:
        sys.exit(f'{args.dst} already exists; refusing to overwrite')
    gains = None
    if args.gains_from_report:
        with open(args.gains_from_report) as f:
            gains = {r['episode']: r.get('track_gain', 0.0) for r in json.load(f)}
    dtypes = {k: src['data'][k].dtype for k in OBS_KEYS + ['state', 'action']}
    out = None if args.dry_run else ReplayBuffer.create_from_path(args.dst, mode='w')

    cfg_dir = os.path.join(REPO_ROOT, 'manifeel', 'config')
    with initialize_config_dir(config_dir=cfg_dir, version_base='1.1'):
        cfg = compose(config_name=args.config_name, overrides=['num_envs=1', 'headless=True'])
        assert cfg.task.rl.add_contact_force_plug_decomposed, 'enable add_contact_force_plug_decomposed in the task config'
        env = TacSLTaskUSB(cfg=omegaconf_to_dict(cfg.task), rl_device=cfg.rl_device, sim_device=cfg.sim_device,
                           graphics_device_id=cfg.graphics_device_id, headless=cfg.headless,
                           virtual_screen_capture=False, force_render=cfg.force_render)

        pos_scale = torch.tensor(cfg.task.rl.pos_action_scale, device=env.device).view(1, 3)
        rot_scale = torch.tensor(cfg.task.rl.rot_action_scale, device=env.device).view(1, 3)
        report = []
        for ep in episodes:
            t0 = time.time()
            actions = src['data/action'][starts[ep]:ends[ep]]
            ref_state = src['data/state'][starts[ep]:ends[ep]]
            ref_ff = src['data/tactile_force_field_right'][starts[ep]:ends[ep]]
            T = len(actions)

            set_seed(ep, torch_deterministic=cfg.torch_deterministic)  # same seeding as demo_teleop.py
            env.reset_idx(torch.arange(env.num_envs))
            env.compute_observations()
            obs = env.reset()['obs']

            rec = {k: [] for k in OBS_KEYS + CONTACT_KEYS + ['state']}
            pos_err, ff_err, corr_mag, kp_dist = [], [], [], []
            gain = gains[ep] if gains is not None else args.track_gain
            for t in range(T):
                for k in OBS_KEYS + CONTACT_KEYS:
                    rec[k].append(obs[k][0].cpu().numpy().copy())
                ee_pos, ee_quat = obs['ee_pos'][0].cpu().numpy(), obs['ee_quat'][0].cpu().numpy()
                rec['state'].append(np.concatenate([ee_pos, ee_quat]))
                pos_err.append(np.linalg.norm(ee_pos - ref_state[t, :3]))
                kp_dist.append(torch.norm(env.keypoints_socket - env.keypoints_plug, dim=-1).mean(-1)[0].item())
                ff_err.append(np.abs(rec['tactile_force_field_right'][-1] - ref_ff[t]).mean() / (np.abs(ref_ff[t]).mean() + 1e-9))
                if t < T - 1:
                    act = torch.from_numpy(actions[t]).float().view(1, -1).to(env.device)
                    if gain > 0:
                        ref_pos = torch.from_numpy(ref_state[t, :3]).float().view(1, 3).to(env.device)
                        ref_quat = torch.from_numpy(ref_state[t, 3:7]).float().view(1, 4).to(env.device)
                        corr = tracking_correction(env, ref_pos, ref_quat, pos_scale, rot_scale)
                        corr_mag.append(corr.abs().max().item())
                        act = (act + gain * corr).clamp(-1.0, 1.0)
                    obs = env.step(act)[0]['obs']

            # The recorded episode ends right after its last action was applied (success terminates it), but that final
            # observation is not stored. Apply it once more here so the success flag sees the terminal outcome.
            act = torch.from_numpy(actions[T - 1]).float().view(1, -1).to(env.device)
            if gain > 0:
                ref_pos = torch.from_numpy(ref_state[T - 1, :3]).float().view(1, 3).to(env.device)
                ref_quat = torch.from_numpy(ref_state[T - 1, 3:7]).float().view(1, 4).to(env.device)
                act = (act + gain * tracking_correction(env, ref_pos, ref_quat, pos_scale, rot_scale)).clamp(-1.0, 1.0)
            env.step(act)
            kp_dist.append(torch.norm(env.keypoints_socket - env.keypoints_plug, dim=-1).mean(-1)[0].item())

            if out is not None:
                data = {k: np.stack(v).astype(dtypes.get(k, np.float32)) for k, v in rec.items()}
                data['action'] = actions.astype(dtypes['action'])
                out.add_episode(data, compressors='disk')
            else:
                data = {k: np.stack(v) for k, v in rec.items() if k in CONTACT_KEYS}
            kp = np.array(kp_dist)
            thresh = float(cfg.task.rl.close_error_thresh)
            succ_steps = np.nonzero(kp < thresh)[0]
            r = dict(episode=ep, length=T, ee_pos_err_end_mm=float(pos_err[-1] * 1e3), ee_pos_err_max_mm=float(max(pos_err) * 1e3),
                     tactile_ff_rel_err_mean=float(np.mean(ff_err)),
                     track_gain=gain, max_correction=float(max(corr_mag)) if corr_mag else 0.0,
                     success=bool(len(succ_steps) > 0), success_final=bool(kp[-1] < thresh),
                     first_success_step=int(succ_steps[0]) if len(succ_steps) else None,
                     keypoint_dist_final_mm=float(kp[-1] * 1e3), keypoint_dist_min_mm=float(kp.min() * 1e3),
                     plug_socket_force_max=float(np.abs(data['plug_socket_force']).max()),
                     seconds=round(time.time() - t0, 1))
            report.append(r)
            print('EPISODE', json.dumps(r), flush=True)

    with open(args.dst.rstrip('/') + '_report.json', 'w') as f:
        json.dump(report, f, indent=1)
    n_ok = sum(r['success'] for r in report)
    print(f'DONE {len(report)} episodes, success (at any step) {n_ok}/{len(report)}' +
          ('' if out is None else f', wrote {out.n_steps} steps to {args.dst}'))


if __name__ == '__main__':
    main()
