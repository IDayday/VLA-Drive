"""Compare any two matched single-output study policies using common statistics."""
import argparse
import csv
import json
from pathlib import Path
from starVLA.rl.qwen_flow_exploration.protocol import read_protocol,ROOT
from starVLA.rl.qwen_flow_exploration.common_analysis import analyze_pair,cpu_self_test
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_sourceflow.contracts import fingerprint,file_sha


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--protocol')
    parser.add_argument('--evaluation')
    parser.add_argument('--reference')
    parser.add_argument('--reference-strata',default=str(ROOT/'common_bank/reference_strata.json'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--paired-scenes-output',help='Private per-scene CSV; defaults beside aggregate output')
    parser.add_argument('--bootstrap-repeats',type=int,default=2000)
    parser.add_argument('--bootstrap-seed',type=int,default=42)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():
        raise FileExistsError('Keep old result/evidence; choose a new analysis output')
    if args.self_test:
        result=cpu_self_test();atomic_json(output,result);print(json.dumps(result));return
    if not args.evaluation or not args.reference:
        parser.error('--evaluation and --reference are required outside self-test')
    protocol=read_protocol(args.protocol)
    result,rows=analyze_pair(args.evaluation,args.reference,protocol,args.reference_strata,
                            repeats=args.bootstrap_repeats,bootstrap_seed=args.bootstrap_seed)
    csv_path=Path(args.paired_scenes_output) if args.paired_scenes_output else output.with_name(output.stem+'_paired_scenes.csv')
    if csv_path.exists():raise FileExistsError(csv_path)
    csv_path.parent.mkdir(parents=True,exist_ok=True)
    with csv_path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result.pop('sha256')
    result.update(paired_scene_results_path=str(csv_path.resolve()),paired_scene_results_sha256=file_sha(csv_path),
                  analysis_cli_sha256=file_sha(__file__))
    result['sha256']=fingerprint(result)
    atomic_json(output,result)
    print(json.dumps({key:result[key] for key in ['status','scenes','logs','current_PDMS','reference_PDMS','delta_PDMS_points']}))


if __name__=='__main__':main()
