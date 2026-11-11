"""Step through the demos of a regenerated ManiFeel USB dataset one by one.

Two modes (run inside the container, from the repo root; see the commands at the bottom):

  --mode sim     Replay each episode in the Isaac Gym viewer: same seed, same executed actions. This is a fresh
                 simulation, so it can differ slightly from the stored data (the sim is not bit-deterministic).
                 After each episode the terminal shows the outcome; press Enter for the next one.
  --mode images  Browse the STORED data: the 4 cameras, both tactile images and the contact forces, per step
                 (matplotlib window; no simulator needed).

Examples:
    python scripts/visualize_demos.py --mode sim
    python scripts/visualize_demos.py --mode images --episodes 0 9 38
"""
import argparse
import json
import os
import select
import sys
import time

import numpy as np
import zarr

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
IMAGE_KEYS = [('wrist', 'wrist'), ('wrist_2', 'wrist 2'), ('front', 'front'), ('side', 'side'),
              ('left_tactile_camera_taxim', 'left tactile'), ('right_tactile_camera_taxim', 'right tactile')]
FORCE_KEYS = [('plug_socket_force', 'plug <- socket'), ('plug_left_elastomer_force', 'plug <- left finger'),
              ('plug_right_elastomer_force', 'plug <- right finger')]


def load_dataset(path):
    ds = zarr.open(path, 'r')
    with open(path.rstrip('/') + '_report.json') as f:
        report = [r for r in json.load(f) if r.get('written', True)]
    ends = ds['meta/episode_ends'][:]
    assert len(ends) == len(report), f'{len(ends)} episodes in the dataset but {len(report)} in the report'
    starts = np.concatenate([[0], ends[:-1]])
    entries = [dict(report[i], idx=i, start=int(starts[i]), end=int(ends[i])) for i in range(len(report))]
    return ds, entries


def select_entries(entries, episodes):
    """`episodes` are SOURCE episode numbers (the ones in the report), not positions in the dataset."""
    if not episodes:
        return entries
    by_ep = {e['episode']: e for e in entries}
    missing = [ep for ep in episodes if ep not in by_ep]
    if missing:
        print(f'skipping episodes not in the dataset: {missing}')
    return [by_ep[ep] for ep in episodes if ep in by_ep]


def describe(e):
    c = e.get('chosen') or {}
    return (f"episode {e['episode']} ({e['idx'] + 1} in dataset, {e['length']} steps) | stored: success={e['success']} "
            f"gain={c.get('gain')} drift={c.get('ee_pos_err_max_mm', float('nan')):.1f}mm "
            f"final keypoint dist={c.get('keypoint_dist_final_mm', float('nan')):.1f}mm")


# --------------------------------------------------------------------------------------------- images mode
def run_images(args, ds, entries):
    import matplotlib
    matplotlib.use('Agg' if args.save_frame else 'TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(14, 9.5))
    gs = GridSpec(3, 3, height_ratios=[1, 1, 0.55], figure=fig)
    axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]
    ax_f = fig.add_subplot(gs[2, :])
    state = dict(i=0, t=0, playing=False, data=None, ims=None, cursor=None)

    def load(i):
        e = entries[i]
        s, en = e['start'], e['end']
        state['data'] = {k: ds['data'][k][s:en] for k, _ in IMAGE_KEYS}
        state['forces'] = {k: np.linalg.norm(ds['data'][k][s:en], axis=1) for k, _ in FORCE_KEYS}
        state['i'], state['t'] = i, 0
        ax_f.clear()
        for k, label in FORCE_KEYS:
            ax_f.plot(state['forces'][k], label=label)
        ax_f.set_xlabel('step'); ax_f.set_ylabel('|force| [N]'); ax_f.legend(loc='upper left', fontsize=8)
        state['cursor'] = ax_f.axvline(0, color='k', lw=1)
        state['ims'] = []
        for ax, (k, label) in zip(axes, IMAGE_KEYS):
            ax.clear(); ax.axis('off'); ax.set_title(label, fontsize=9)
            state['ims'].append(ax.imshow(np.clip(state['data'][k][0], 0, 1)))
        draw()

    def draw():
        e, t = entries[state['i']], state['t']
        for im, (k, _) in zip(state['ims'], IMAGE_KEYS):
            im.set_data(np.clip(state['data'][k][t], 0, 1))
        state['cursor'].set_xdata([t, t])
        fig.suptitle(f"{describe(e)}\nstep {t + 1}/{e['length']}   [right/left: step, space: play/pause, n/p: next/prev episode, q: quit]",
                     fontsize=9)
        fig.canvas.draw_idle()

    def on_key(event):
        T = entries[state['i']]['length']
        if event.key == 'right': state['t'] = min(state['t'] + 1, T - 1)
        elif event.key == 'left': state['t'] = max(state['t'] - 1, 0)
        elif event.key == ' ': state['playing'] = not state['playing']
        elif event.key == 'n' and state['i'] + 1 < len(entries): load(state['i'] + 1); return
        elif event.key == 'p' and state['i'] > 0: load(state['i'] - 1); return
        elif event.key == 'q': plt.close(fig); return
        draw()

    def tick():
        if state['playing']:
            T = entries[state['i']]['length']
            if state['t'] + 1 < T: state['t'] += 1; draw()
            else: state['playing'] = False

    load(0)
    if args.save_frame:
        state['t'] = entries[0]['length'] // 2
        draw()
        fig.savefig(args.save_frame, dpi=70)
        print('saved', args.save_frame)
        return
    fig.canvas.mpl_connect('key_press_event', on_key)
    timer = fig.canvas.new_timer(interval=int(1000 / args.fps))
    timer.add_callback(tick); timer.start()
    plt.show()


# ------------------------------------------------------------------------------------------------ sim mode
def run_sim(args, ds, entries):
    import isaacgym  # noqa: F401  (must come before torch)
    import torch
    from hydra import compose, initialize_config_dir
    from isaacgymenvs.tasks.tacsl.tacsl_task_USB import TacSLTaskUSB
    from isaacgymenvs.utils.reformat import omegaconf_to_dict
    from isaacgymenvs.utils.utils import set_seed

    with initialize_config_dir(config_dir=os.path.join(REPO_ROOT, 'manifeel', 'config'), version_base='1.1'):
        cfg = compose(config_name='isaacgym_config_usb',
                      overrides=['num_envs=1', f'headless={str(args.headless).lower()}'])
        env = TacSLTaskUSB(cfg=omegaconf_to_dict(cfg.task), rl_device=cfg.rl_device, sim_device=cfg.sim_device,
                           graphics_device_id=cfg.graphics_device_id, headless=cfg.headless,
                           virtual_screen_capture=False, force_render=cfg.force_render)
        thresh = float(cfg.task.rl.close_error_thresh)

        def keypoint_dist_mm():
            return torch.norm(env.keypoints_socket - env.keypoints_plug, dim=-1).mean(-1)[0].item() * 1e3

        def idle_until_enter(prompt):
            """Keep the viewer responsive while waiting for a line on stdin."""
            print(prompt, end='', flush=True)
            while True:
                if env.viewer is not None:
                    env.render()
                if select.select([sys.stdin], [], [], 0.03 if env.viewer is not None else None)[0]:
                    return sys.stdin.readline().strip().lower()

        def play(e):
            actions = ds['data/action_executed' if args.actions == 'executed' else 'data/action'][e['start']:e['end']]
            ref_state = ds['data/state'][e['start']:e['end']]
            set_seed(e['episode'], torch_deterministic=cfg.torch_deterministic)
            env.reset_idx(torch.arange(1)); env.compute_observations(); obs = env.reset()['obs']
            drift, best = [], keypoint_dist_mm()
            for t, a in enumerate(actions):
                drift.append(np.linalg.norm(obs['ee_pos'][0].cpu().numpy() - ref_state[t, :3]) * 1e3)
                t0 = time.time()
                obs = env.step(torch.from_numpy(a).float().view(1, -1).to(env.device))[0]['obs']
                best = min(best, keypoint_dist_mm())
                if args.min_step_time > 0:
                    time.sleep(max(0.0, args.min_step_time - (time.time() - t0)))
            final = keypoint_dist_mm()
            print(f"  replay: success={final < thresh * 1e3} final keypoint dist={final:.1f}mm (threshold {thresh * 1e3:.2f}mm) "
                  f"max drift vs stored path={max(drift):.1f}mm")

        i = 0
        while 0 <= i < len(entries):
            e = entries[i]
            print(f"\n[{i + 1}/{len(entries)}] {describe(e)}")
            play(e)
            if args.no_wait:
                i += 1
                continue
            cmd = idle_until_enter("  [Enter] next   [r] replay   [p] previous   [q] quit > ")
            if cmd == 'q': break
            elif cmd == 'r': continue
            elif cmd == 'p': i = max(0, i - 1)
            else: i += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=['sim', 'images'], default='sim')
    ap.add_argument('--dataset', default='data/usb_quan_Aug05_contact_v3')
    ap.add_argument('--episodes', type=int, nargs='*', default=None, help='source episode numbers to show (default: all in the dataset)')
    ap.add_argument('--actions', choices=['executed', 'original'], default='executed',
                    help='sim mode: action_executed (what generated the stored data) or the original teleop action')
    ap.add_argument('--headless', action='store_true', help='sim mode: no viewer (for testing)')
    ap.add_argument('--no-wait', action='store_true', help='sim mode: do not wait for Enter between episodes')
    ap.add_argument('--min-step-time', type=float, default=0.0, help='sim mode: slow playback to at least this many seconds per step')
    ap.add_argument('--fps', type=float, default=10.0, help='images mode: playback rate')
    ap.add_argument('--save-frame', default=None, help='images mode: save one frame of the first episode to this PNG and exit (for testing)')
    args = ap.parse_args()

    ds, entries = load_dataset(args.dataset)
    entries = select_entries(entries, args.episodes)
    if not entries:
        sys.exit('nothing to show')
    print(f'{len(entries)} episodes from {args.dataset}')
    (run_sim if args.mode == 'sim' else run_images)(args, ds, entries)


if __name__ == '__main__':
    main()
