"""Same qualified NAVTEST pipeline, with the standard study's IL placement guard."""
import argparse
from pathlib import Path

from starVLA.rl.qwen_world_safety_exploration.contracts import ROOT
from .eval_owner import EvaluationOwner

STUDY = Path('/mnt/project/qwen-world-safety-exploration-20261031/standard_training_20261101')
PROTECTED = {'training-vlawm-tsinghua-zt', 'training-vla-zt'}


class StandardEvaluationOwner(EvaluationOwner):
    def samples(self):
        # Both CPU export/finalization and GPU shard placement consume these
        # samples. Preserve existing B/D learners and their guardians.
        return [(host, sample) for host, sample in super().samples() if host not in PROTECTED]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--registry', default=str(STUDY/'RUNS.json'))
    parser.add_argument('--max-shards', type=int, default=32)
    parser.add_argument('--max-workers', type=int, default=64)
    parser.add_argument('--minimum-free-mib', type=int, default=32768)
    parser.add_argument('--interval', type=float, default=30)
    args = parser.parse_args()
    if ROOT.resolve() != STUDY or Path(args.registry).resolve() != STUDY/'RUNS.json':
        raise ValueError('Only the explicitly configured standard study')
    if args.max_shards != 32 or args.max_workers != 64 or args.interval < 10:
        parser.error('Preserve 32 canonical shards, up to 64 workers, interval>=10')
    StandardEvaluationOwner(args.registry, max_shards=32, max_workers=64,
        minimum=args.minimum_free_mib).run(interval=args.interval)


if __name__ == '__main__':
    main()
