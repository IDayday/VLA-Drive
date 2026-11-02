"""Actual eight-rank D6 staged mid-inner resume equivalence; no GPU work."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from starVLA.rl.qwen_world_compare.contracts import file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json


JOURNAL_ADMIN={'invocation','at','update_seconds','collection_seconds_generation','collection_seconds_scoring','peak_GPU_bytes','model_work'}


def exact(a,b,path='root'):
    if type(a)is not type(b):raise ValueError('Type differs: '+path)
    if isinstance(a,torch.Tensor):
        if a.dtype!=b.dtype or a.shape!=b.shape or not torch.equal(a,b):raise ValueError('Tensor differs: '+path)
    elif isinstance(a,np.ndarray):
        if a.dtype!=b.dtype or not np.array_equal(a,b):raise ValueError('Array differs: '+path)
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise ValueError('Keys differ: '+path)
        for key in a:exact(a[key],b[key],path+'.'+str(key))
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b):raise ValueError('Length differs: '+path)
        for i,(x,y)in enumerate(zip(a,b)):exact(x,y,path+'.'+str(i))
    elif a!=b:raise ValueError('Value differs: '+path)


def validate(continuous,resumed,source):
    continuous,resumed,source=map(Path,(continuous,resumed,source))
    identity=json.loads((continuous/'identity.json').read_text());exact(identity,json.loads((resumed/'identity.json').read_text()))
    if identity['recipe']['arm']!='D6'or identity['world']!=8 or not identity['qualification']:raise ValueError('Requires real eight-rank staged qualification')
    fork=json.loads((resumed/'RECOVERY_FORK_QUALIFICATION_ONLY.json').read_text())
    if fork['restored_update']!=5 or fork['next_inner']!=1:raise ValueError('Expected actual stage-II inner1 fork')
    cp='checkpoints/update000016_inner0'
    for root in (continuous,resumed):
        if not(root/cp/'COMPLETE').exists():raise ValueError('Checkpoint not published')
        for rank in range(8):
            value=json.loads((root/f'result_rank{rank}.json').read_text())
            if value['updates']!=16 or value['status']!='QUALIFICATION_STOPPED'or value['frozen_before']!=value['frozen_after']:raise ValueError('Real staged run incomplete or frozen weights changed')
    for name in ['optimizer.pt']+[f'rank{rank}.pt'for rank in range(8)]:
        exact(torch.load(continuous/cp/name,map_location='cpu',weights_only=False),torch.load(resumed/cp/name,map_location='cpu',weights_only=False),name)
    for rank in range(8):
        load=lambda root:[{k:v for k,v in json.loads(line).items()if k not in JOURNAL_ADMIN}for line in(root/f'metrics_rank{rank}.jsonl').read_text().splitlines()]
        a,b=load(continuous),load(resumed)
        if [r['updates']for r in a]!=list(range(1,17)):raise ValueError('Unexpected continuous update history')
        exact(a,b,f'journal.rank{rank}')
    files=('tools/d44_native_rl/train.py','starVLA/rl/d44_native_rl/stages.py','starVLA/rl/d44_native_rl/history.py','starVLA/rl/d44_native_rl/recovery.py')
    return dict(status='PASS',continuous=str(continuous),resumed=str(resumed),restored_update=5,next_inner=1,final_update=16,actual_world=8,
        scope='All adapters/Adam/scheduler, all8rank complete RNG/state/history, losses/advantages/gradients/LR bitwise equal; compressed stages qualification only',
        excluded_journal_fields=sorted(JOURNAL_ADMIN),source_files={p:file_sha(source/p)for p in files},identity_sha256=identity['sha256'],
        formal_epochs_proven=False,qualification_weights_are_not_formal_initialization=True)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--continuous',required=True);p.add_argument('--resumed',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    value=validate(a.continuous,a.resumed,a.source);output=Path(a.output)
    if output.exists():
        if json.loads(output.read_text())!=value:raise FileExistsError('Keep earlier proof immutable')
    else:atomic_json(output,value)


if __name__=='__main__':main()
