"""Re-record the USB-insertion dataset, retrying each episode until it succeeds.

Each source episode is replayed from its original seed with the recorded actions. If the replay does not end in a
successful insertion (mean plug-socket keypoint distance < task.rl.close_error_thresh), it is retried with different
drift-correction gains (pull back toward the recorded end-effector pose; 0 = plain open-loop replay), repeated attempts
(the simulator is not bit-deterministic), and, for later attempts, small action noise to break jams. Success is logged
DURING generation, so the flag in the report describes exactly the data that is written.

Per episode the lowest-drift successful attempt is kept; attempts stop early once one succeeds with drift <=
--accept-drift-mm or after --max-successes successful attempts. Episodes with no successful attempt are left out (or kept with --keep-failures) and listed in the report.

Stored keys: everything in the source dataset, the plug/socket/finger contact forces, and
    action           the original recorded teleop action (label)
    action_executed  the action actually sent to the sim (label + correction + noise); differs from `action` only when
                     correction or noise was used

Run inside the container from the repo root:
    python scripts/regenerate_best.py --dst data/usb_quan_Aug05_contact_v3
"""
import argparse
import json
import os
import sys
import time

from regenerate_dataset import CONTACT_KEYS, OBS_KEYS, REPO_ROOT, tracking_correction  # imports isaacgym first

import numpy as np
import torch
import zarr
from hydra import compose, initialize_config_dir

from diffusion_policy.common.replay_buffer import ReplayBuffer
from isaacgymenvs.tasks.tacsl.tacsl_task_USB import TacSLTaskUSB
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.utils import set_seed


def run_attempt(env, cfg, ep, actions, ref_state, gain, noise_std, rng, pos_scale, rot_scale, thresh):
    """Replay one episode once. Returns (recorded arrays, executed actions, metrics)."""
    T = len(actions)
    set_seed(ep, torch_deterministic=cfg.torch_deterministic)  # same seeding as demo_teleop.py
    env.reset_idx(torch.arange(env.num_envs))
    env.compute_observations()
    obs = env.reset()['obs']

    def keypoint_dist():
        return torch.norm(env.keypoints_socket - env.keypoints_plug, dim=-1).mean(-1)[0].item()

    rec = {k: [] for k in OBS_KEYS + CONTACT_KEYS + ['state']}
    executed = np.zeros_like(actions)
    pos_err, kp = [], [keypoint_dist()]
    for t in range(T):
        for k in OBS_KEYS + CONTACT_KEYS:
            rec[k].append(obs[k][0].cpu().numpy().copy())
        ee_pos, ee_quat = obs['ee_pos'][0].cpu().numpy(), obs['ee_quat'][0].cpu().numpy()
        rec['state'].append(np.concatenate([ee_pos, ee_quat]))
        pos_err.append(np.linalg.norm(ee_pos - ref_state[t, :3]))

        act = torch.from_numpy(actions[t]).float().view(1, -1).to(env.device)
        if gain > 0:
            ref_pos = torch.from_numpy(ref_state[t, :3]).float().view(1, 3).to(env.device)
            ref_quat = torch.from_numpy(ref_state[t, 3:7]).float().view(1, 4).to(env.device)
            act = act + gain * tracking_correction(env, ref_pos, ref_quat, pos_scale, rot_scale)
        if noise_std > 0:
            act = act + torch.from_numpy(rng.normal(0.0, noise_std, size=tuple(act.shape))).float().to(env.device)
        act = act.clamp(-1.0, 1.0)
        executed[t] = act[0].cpu().numpy()
        # the last action is applied too (it ends the recorded episode); its observation is not stored, as in the source
        step_obs = env.step(act)[0]['obs']
        kp.append(keypoint_dist())
        if t < T - 1:
            obs = step_obs

    kp = np.array(kp)
    metrics = dict(gain=gain, noise_std=noise_std, success=bool((kp[-1] < thresh)), success_any=bool((kp < thresh).any()),
                   keypoint_dist_final_mm=float(kp[-1] * 1e3), ee_pos_err_max_mm=float(max(pos_err) * 1e3),
                   ee_pos_err_end_mm=float(pos_err[-1] * 1e3), plug_socket_force_max=float(np.abs(rec['plug_socket_force']).max()))
    return rec, executed, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/usb_quan_Aug05')
    ap.add_argument('--dst', default='data/usb_quan_Aug05_contact_v3')
    ap.add_argument('--config-name', default='isaacgym_config_usb')
    ap.add_argument('--episodes', type=int, nargs='*', default=None)
    ap.add_argument('--gains', default='0,0.5,0,1.0,0.25,0.75,1.5,1.0,0.5,0',
                    help='comma-separated correction gain per attempt (also sets the attempt budget)')
    ap.add_argument('--noise-std', type=float, default=0.02, help='action noise (action units) for attempts >= --noise-from')
    ap.add_argument('--noise-from', type=int, default=5, help='first attempt index (0-based) that adds action noise')
    ap.add_argument('--accept-drift-mm', type=float, default=3.0, help='stop early on a success with max drift <= this')
    ap.add_argument('--max-successes', type=int, default=3,
                    help='stop after this many successful attempts (the lowest-drift one is kept)')
    ap.add_argument('--keep-failures', action='store_true', help='also write episodes with no successful attempt (best effort)')
    args = ap.parse_args()
    gains = [float(g) for g in args.gains.split(',')]

    if os.path.exists(args.dst):
        sys.exit(f'{args.dst} already exists; refusing to overwrite')
    src = zarr.open(args.src, 'r')
    ends = src['meta/episode_ends'][:]
    starts = np.concatenate([[0], ends[:-1]])
    episodes = args.episodes if args.episodes else list(range(len(ends)))
    dtypes = {k: src['data'][k].dtype for k in OBS_KEYS + ['state', 'action']}
    out = ReplayBuffer.create_from_path(args.dst, mode='w')
    report_path = args.dst.rstrip('/') + '_report.json'

    with initialize_config_dir(config_dir=os.path.join(REPO_ROOT, 'manifeel', 'config'), version_base='1.1'):
        cfg = compose(config_name=args.config_name, overrides=['num_envs=1', 'headless=True'])
        assert cfg.task.rl.add_contact_force_plug_decomposed, 'enable add_contact_force_plug_decomposed in the task config'
        env = TacSLTaskUSB(cfg=omegaconf_to_dict(cfg.task), rl_device=cfg.rl_device, sim_device=cfg.sim_device,
                           graphics_device_id=cfg.graphics_device_id, headless=cfg.headless,
                           virtual_screen_capture=False, force_render=cfg.force_render)
        pos_scale = torch.tensor(cfg.task.rl.pos_action_scale, device=env.device).view(1, 3)
        rot_scale = torch.tensor(cfg.task.rl.rot_action_scale, device=env.device).view(1, 3)
        thresh = float(cfg.task.rl.close_error_thresh)

        report = []
        for ep in episodes:
            t0 = time.time()
            actions = src['data/action'][starts[ep]:ends[ep]]
            ref_state = src['data/state'][starts[ep]:ends[ep]]
            rng = np.random.default_rng(1000 + ep)
            attempts, best = [], None  # best = (drift, rec, executed, index)
            fallback = None            # best failed attempt (lowest keypoint distance)
            n_success = 0
            for i, gain in enumerate(gains):
                noise = args.noise_std if i >= args.noise_from else 0.0
                rec, executed, m = run_attempt(env, cfg, ep, actions, ref_state, gain, noise, rng, pos_scale, rot_scale, thresh)
                attempts.append(m)
                if m['success']:
                    if best is None or m['ee_pos_err_max_mm'] < best[0]:
                        best = (m['ee_pos_err_max_mm'], rec, executed, i)
                    n_success += 1
                    if m['ee_pos_err_max_mm'] <= args.accept_drift_mm or n_success >= args.max_successes:
                        break
                elif fallback is None or m['keypoint_dist_final_mm'] < fallback[0]:
                    fallback = (m['keypoint_dist_final_mm'], rec, executed, i)

            chosen = best if best is not None else (fallback if args.keep_failures else None)
            if chosen is not None:
                _, rec, executed, idx = chosen
                data = {k: np.stack(v).astype(dtypes.get(k, np.float32)) for k, v in rec.items()}
                data['action'] = actions.astype(dtypes['action'])
                data['action_executed'] = executed.astype(dtypes['action'])
                out.add_episode(data, compressors='disk')
            entry = dict(episode=ep, length=len(actions), success=best is not None, written=chosen is not None,
                         chosen_attempt=None if chosen is None else chosen[3], n_attempts=len(attempts),
                         chosen=None if chosen is None else attempts[chosen[3]], attempts=attempts, seconds=round(time.time() - t0, 1))
            report.append(entry)
            with open(report_path, 'w') as f:  # saved after every episode so progress survives interruptions
                json.dump(report, f, indent=1)
            c = entry['chosen']
            print('EPISODE %2d %s attempts=%d %s' % (ep, 'OK  ' if entry['success'] else 'FAIL', len(attempts),
                  '' if c is None else 'gain=%.2f noise=%.2f drift=%.1fmm kp=%.1fmm' % (c['gain'], c['noise_std'], c['ee_pos_err_max_mm'], c['keypoint_dist_final_mm'])), flush=True)

    n_ok = sum(r['success'] for r in report)
    print(f'DONE {n_ok}/{len(report)} episodes succeeded; wrote {out.n_steps} steps to {args.dst}')


if __name__ == '__main__':
    main()
