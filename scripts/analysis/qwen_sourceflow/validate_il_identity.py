"""Two-rank actual Gloo publication, delayed rank and strict resume rejection."""
import argparse
import json
import os
from pathlib import Path
import time
import torch.distributed as dist
from starVLA.rl.qwen_sourceflow.il_checkpoint import publish_identity
from starVLA.rl.simwam_recipe import checkpoint as common
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    dist.init_process_group('gloo');common.init_control_group();rank=dist.get_rank();root=Path(a.output)
    if rank==0:root.mkdir(parents=True,exist_ok=False)
    dist.barrier();identity=dict(schema='identity_publication_test',fixed=True)
    if rank==1:time.sleep(.3)
    publish_identity(root,identity,resume=False)
    checks=['late_rank_fresh_publication']
    try:publish_identity(root,identity,resume=False)
    except RuntimeError:checks.append('duplicate_fresh_rejected')
    else:raise AssertionError('Duplicate fresh run accepted')
    publish_identity(root,identity,resume=True);checks.append('same_identity_resume')
    try:publish_identity(root,dict(identity,fixed=rank==0),resume=True)
    except RuntimeError:checks.append('cross_rank_mismatch_rejected')
    else:raise AssertionError('Mismatched rank accepted')
    if json.loads((root/'identity.json').read_text())!=identity:raise AssertionError('Failed attempt changed identity')
    all_checks=common.gather(checks)
    if rank==0:atomic_json(root/'result.json',dict(status='PASS',ranks=all_checks,world=dist.get_world_size()))
    dist.destroy_process_group()


if __name__=='__main__':main()
