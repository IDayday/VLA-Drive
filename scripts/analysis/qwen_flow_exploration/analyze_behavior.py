"""Read existing sealed deployment trajectories; never rerun inference/scoring."""
import argparse
import csv
from pathlib import Path
from starVLA.rl.qwen_flow_exploration.protocol import read_protocol
from starVLA.rl.qwen_flow_exploration.behavior_analysis import analyze_behavior
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--protocol',required=True);p.add_argument('--evaluation',required=True)
    p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();output=Path(a.output)
    if output.exists():raise FileExistsError('Preserve earlier derived summaries')
    result,rows=analyze_behavior(a.evaluation,a.reference,read_protocol(a.protocol))
    table=output.with_name(output.stem+'_paired_scenes.csv');table.parent.mkdir(parents=True,exist_ok=True)
    if table.exists():raise FileExistsError(table)
    with table.open('x',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    atomic_json(output,result)
    print({'status':'COMPLETE','scenes':len(rows),'output':str(output)})


if __name__=='__main__':main()
