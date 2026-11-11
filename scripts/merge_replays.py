"""Build a dataset by picking, per episode, the lowest-drift replay among several regenerated versions.

Each input was produced by scripts/regenerate_dataset.py and has a <dir>_report.json next to it. The base dataset
is used for an episode unless a candidate's replay has a max end-effector error lower by more than
--min-improvement-mm (so already-accurate episodes are never perturbed).

    python scripts/merge_replays.py --base data/usb_quan_Aug05_contact \\
        --candidates /path/tracked_a /path/tracked_b --dst data/usb_quan_Aug05_contact_v2
"""
import argparse
import json
import os
import sys

from diffusion_policy.common.replay_buffer import ReplayBuffer


def load(path):
    with open(path.rstrip('/') + '_report.json') as f:
        return ReplayBuffer.create_from_path(path, mode='r'), json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--candidates', nargs='+', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--min-improvement-mm', type=float, default=1.0)
    args = ap.parse_args()
    if os.path.exists(args.dst):
        sys.exit(f'{args.dst} already exists; refusing to overwrite')

    base_rb, base_report = load(args.base)
    options = {r['episode']: [(args.base, base_rb, i, r)] for i, r in enumerate(base_report)}
    for cand in args.candidates:
        rb, report = load(cand)
        for i, r in enumerate(report):
            options.setdefault(r['episode'], []).append((cand, rb, i, r))

    out = ReplayBuffer.create_from_path(args.dst, mode='w')
    merged = []
    for ep in sorted(options):
        base_opt = options[ep][0]
        best = base_opt
        for opt in options[ep][1:]:
            if opt[3]['ee_pos_err_max_mm'] < best[3]['ee_pos_err_max_mm'] - args.min_improvement_mm:
                best = opt
        path, rb, idx, r = best
        out.add_episode(rb.get_episode(idx, copy=True), compressors='disk')
        merged.append({**r, 'source': path, 'base_ee_pos_err_max_mm': base_opt[3]['ee_pos_err_max_mm']})
        print(f"EPISODE {ep:2d}: {'base' if best is base_opt else 'tracked'}  max err {base_opt[3]['ee_pos_err_max_mm']:.1f} -> {r['ee_pos_err_max_mm']:.1f} mm", flush=True)

    with open(args.dst.rstrip('/') + '_report.json', 'w') as f:
        json.dump(merged, f, indent=1)
    print(f'DONE wrote {len(merged)} episodes, {out.n_steps} steps to {args.dst}')


if __name__ == '__main__':
    main()
