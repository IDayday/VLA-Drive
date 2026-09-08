"""Measure unique raw training GT. Input JSONL must be produced from an explicit training manifest."""
import argparse
import json
from pathlib import Path
from navsim.agents.EpisodeDrive.planreg_v2.normalizers import measured_statistics


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--records', required=True, help='JSONL: token, trajectory, valid')
    p.add_argument('--training-tokens', required=True, help='JSON array of authorized training tokens')
    p.add_argument('--split', choices=['train', 'trainval_final_fit'], required=True)
    p.add_argument('--data-version', required=True)
    p.add_argument('--std-floor', type=float, default=.001)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if Path(args.output).exists():
        raise FileExistsError('Refusing to overwrite a statistics version')
    allowed = set(json.loads(Path(args.training_tokens).read_text()))
    def records():
        with open(args.records) as stream:
            for line in stream:
                r = json.loads(line)
                if r['token'] not in allowed:
                    raise ValueError('Non-training token in statistics input: ' + r['token'])
                yield r['token'], r['trajectory'], r['valid']
    result = measured_statistics(records(), args.split, args.data_version, args.std_floor)
    if result.metadata['count'] != len(allowed):
        raise ValueError('Statistics do not cover the complete declared training token set')
    result.save(args.output)
    print(json.dumps(result.metadata, indent=2))


if __name__ == '__main__':
    main()
