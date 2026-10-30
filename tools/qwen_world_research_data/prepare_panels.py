"""Seal fixed 1024/4096/full TRAIN diagnostic censuses before policy scoring."""
import argparse
import json
from starVLA.rl.qwen_world_research_data.panels import prepare_panels

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=420301)
    a=p.parse_args();print(json.dumps(prepare_panels(a.data,a.output,seed=a.seed),indent=2))

if __name__=='__main__':main()
