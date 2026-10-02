"""Render every completed formal loss record up to an immutable live snapshot."""
import argparse
import json
import math
from pathlib import Path

from tools.action_video_foresight.summarize_live_campaign import read_steps


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--campaign-root', required=True)
    parser.add_argument('--output-prefix', required=True)
    parser.add_argument('--window', type=int, default=50)
    args = parser.parse_args()
    if args.window < 1:
        raise ValueError('Positive averaging window required')
    snapshot = json.loads(Path(args.snapshot).read_text())
    prefix = Path(args.output_prefix)
    outputs = [prefix.with_suffix('.json'), prefix.with_suffix('.svg')]
    if any(path.exists() for path in outputs):
        raise ValueError('Preserve previous learning-curve snapshots')
    keys = ('ego_fm', 'current_dino', 'future_clip', 'interaction')
    curves = {}
    for arm, spec in sorted(snapshot['runs'].items()):
        root = Path(args.campaign_root) / 'students' / spec['run_id']
        rows = [row for row in read_steps(root / 'steps.jsonl')
                if row['update'] <= spec['latest_recorded_update']]
        if [row['update'] for row in rows] != list(range(1, len(rows) + 1)):
            raise ValueError('Missing, repeated or unordered completed updates')
        if len(rows) != spec['latest_recorded_update']:
            raise ValueError('Snapshot and training journal disagree')
        if any(not math.isfinite(row['raw_losses'][key]) for row in rows for key in keys):
            raise ValueError('Nonfinite completed loss record; do not filter it out')
        bins = []
        for offset in range(0, len(rows), args.window):
            batch = rows[offset:offset + args.window]
            bins.append({'update_start': batch[0]['update'], 'update_end': batch[-1]['update'],
                'records': len(batch), 'scene_exposure': batch[-1]['exposure'],
                'raw_losses': {key: sum(row['raw_losses'][key] for row in batch) / len(batch)
                               for key in keys},
                'step_seconds': sum(row['seconds'] for row in batch) / len(batch)})
        curves[arm] = bins
    result = {'training_source': snapshot['training_source'],
        'snapshot_utc': snapshot['snapshot_utc'], 'formal_plan_identity': snapshot['formal_plan_identity'],
        'interpretation': 'Consecutive completed-update means, including the final partial window. '
            'No completed record is discarded. Different actual progress and target families '
            'are shown; this is not a common-endpoint ranking, convergence claim or PDMS result.',
        'curves': curves}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    for axis, key in zip(axes.flat, keys):
        for arm, bins in curves.items():
            axis.plot([row['update_end'] for row in bins],
                      [row['raw_losses'][key] for row in bins], label=arm)
        axis.set(xlabel='Optimizer update', ylabel='Raw loss', title=key)
        axis.grid(alpha=.25)
        axis.legend()
    figure.suptitle('Formal early learning: different live progress, no planning conclusion')
    figure.tight_layout()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(json.dumps(result, indent=2) + '\n')
    figure.savefig(outputs[1])
    plt.close(figure)
    outputs[1].write_text('\n'.join(line.rstrip() for line in outputs[1].read_text().splitlines()) + '\n')


if __name__ == '__main__':
    main()
