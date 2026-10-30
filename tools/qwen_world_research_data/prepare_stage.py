"""Freeze the next adaptive census before admitting its training stage."""
import argparse
import json
from starVLA.rl.qwen_world_research_data.adaptive import prepare_stage

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--stream-manifest',required=True)
    p.add_argument('--stage-index',type=int,required=True)
    p.add_argument('--current-bank');p.add_argument('--baseline-bank')
    p.add_argument('--uniform-fraction',type=float,default=.35)
    p.add_argument('--low-score-fraction',type=float,default=.4)
    p.add_argument('--regression-fraction',type=float,default=.25)
    p.add_argument('--world-size',type=int,default=8)
    a=p.parse_args()
    result=prepare_stage(a.stream_manifest,a.stage_index,current_bank=a.current_bank,baseline_bank=a.baseline_bank,
                         uniform_fraction=a.uniform_fraction,low_score_fraction=a.low_score_fraction,
                         regression_fraction=a.regression_fraction,world_size=a.world_size)
    print(json.dumps({k:v for k,v in result.items() if k!='ordered_tokens'},indent=2))

if __name__=='__main__':main()
