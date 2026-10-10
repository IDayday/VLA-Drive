"""Bounded real cached-image qualification: expansion -> selector-only -> expansion."""
import argparse
import time
import torch
from iqe.config import load_config
from iqe.pipeline import Pipeline
from iqe.rounds import run_round
from iqe.io import atomic_json,read_json,file_hash
from iqe.resources import qualify


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    qualify('cpu');config=load_config(args.config);torch.set_num_threads(config['execution']['num_threads'])
    p=Pipeline(config,mode='smoke',max_samples=32,device='cpu');p.use_locked_base()
    records=[]
    for number,selector_only in [(1,False),(2,True),(3,False)]:
        start=time.perf_counter()
        result=run_round(p,number,resume=True,steps=2,selector_only=selector_only)
        records.append({'round':number,'selector_only':selector_only,'seconds':time.perf_counter()-start,'result':result})
        atomic_json(p.root/'REVIEW_PROGRESS.json',records)
    result={'status':'PASS','science':'UNTESTED','diagnostic_base':p.contract['query_checkpoint'],
            'base_mode':p.contract['base_mode'],'workflows':records,'experts':p.registry.candidate_pool(),
            'formal_ACTIVE_modified':False,'new_real_observations':0,'mode':'resampling_only'}
    atomic_json(p.root/'REVIEW_RESULT.json',result,immutable=True)


if __name__=='__main__':main()
