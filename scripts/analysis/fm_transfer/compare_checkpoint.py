"""Finite execution patch gate: full numeric state, allowing declared source identity changes."""
import argparse
import json
from pathlib import Path
import torch
from scripts.analysis.fm_transfer.compare_resume import equal
from starVLA.rl.simwam_recipe.contracts import file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--left',required=True)
    p.add_argument('--right',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();paths=[Path(a.left),Path(a.right)];out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    ids=[]
    for path in paths:
        seal=json.loads((path/'COMPLETE').read_text());files=json.loads((path/'files.json').read_text())
        if file_sha(path/'files.json')!=seal['files_sha256']:raise ValueError('Unsealed state')
        for n,h in files.items():
            if file_sha(path/n)!=h:raise ValueError('Changed state '+n)
        ids.append(json.loads((path/'identity.json').read_text()))
    equal(ids[0]['config'],ids[1]['config'],'configuration')
    if not all(i['qualification'] for i in ids):raise ValueError('Execution patch gate must use isolated qualification runs')
    for name in ['model.pt','optimizer.pt',*[f'rank{i}.pt' for i in range(8)]]:
        left,right=[torch.load(path/name,map_location='cpu',weights_only=False,mmap=True) for path in paths]
        if name.startswith('rank'):
            equal(left['state'],right['state'],name+'/state');equal(left['rng'],right['rng'],name+'/rng')
        else:equal(left,right,name)
        del left,right
    result=dict(status='PASS',scope='initial-state publication patch; full model/Adam/scheduler/eight-rank RNG/cursor',
        left=str(paths[0]),right=str(paths[1]),old_identity=ids[0]['sha256'],new_identity=ids[1]['sha256'],
        declared_change='save initial update0; completion status label; no SFT forward/optimizer changes',
        full_numeric_state_exact=True,qualification_only=True)
    atomic_json(out,result);print(json.dumps(result))


if __name__=='__main__':main()
