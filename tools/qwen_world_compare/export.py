"""Strict standalone four-framework IL/RL export (LoRA wrappers retained)."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    script = Path(__file__).resolve()
    # profile.py is a public CLI sibling, not Python's stdlib profile module.
    # Keep this directory off the bare-script import path (cProfile uses it).
    sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != script.parent]
    sys.path.insert(0, str(script.parents[2]))

from starVLA.rl.qwen_world_compare.deployment import export_policy


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', required=True, help='Complete new-schema IL native checkpoint')
    parser.add_argument('--output', required=True)
    parser.add_argument('--rl-checkpoint', help='Optional exact own-parent native RL checkpoint')
    args = parser.parse_args()
    meta = export_policy(args.checkpoint, args.output, args.rl_checkpoint)
    print(json.dumps(dict(status='COMPLETE', architecture_id=meta['architecture_id'],
                          model_sha256=meta['model_sha256'], LoRA_merge=False)))


if __name__ == '__main__':
    main()
