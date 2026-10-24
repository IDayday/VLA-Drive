"""Build a sealed fixed TRAIN success archive from common scored candidates."""
import argparse
import json
from pathlib import Path
from starVLA.rl.qwen_flow_exploration.preservation.archive import ArchiveConfig, FrozenArchive


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--candidates',required=True)
    parser.add_argument('--population',required=True,help='Existing full log-isolated population manifest with train_tokens/token_logs')
    parser.add_argument('--reference-candidate-ids')
    parser.add_argument('--output',required=True)
    parser.add_argument('--minimum-reward',type=float,default=.9)
    parser.add_argument('--near-best-tolerance',type=float,default=.01)
    args=parser.parse_args()
    path=Path(args.candidates)
    records=[json.loads(row) for row in path.read_text().splitlines() if row.strip()] if path.suffix=='.jsonl' else json.loads(path.read_text())
    population=json.loads(Path(args.population).read_text())
    reference=json.loads(Path(args.reference_candidate_ids).read_text()) if args.reference_candidate_ids else None
    archive=FrozenArchive.build(records,train_tokens=population['train_tokens'],token_logs=population['token_logs'],
            config=ArchiveConfig(minimum_reward=args.minimum_reward,near_best_tolerance=args.near_best_tolerance),reference_candidate_ids=reference)
    output=Path(args.output)
    if output.exists():raise FileExistsError('Preserve immutable initial archive; use a new experiment ID')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(archive.state_dict(),indent=2)+'\n')
    print(json.dumps(archive.provenance,indent=2))


if __name__=='__main__':main()
