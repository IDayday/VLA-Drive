"""Verify the new full-head window, actual gradients, and exact saved resume."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import file_sha


def equal(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,np.ndarray):
        return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):
        return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def check(root):
    root=Path(root)
    ranks=[json.loads((root/f'result_rank{i}.json').read_text()) for i in range(8)]
    if any(r['status']!='COMPLETE_FIXED_BUDGET' or r['updates']!=8 or r['frozen_prefix_sha256_before']!=r['frozen_prefix_sha256_after'] or not r['reference_frozen'] for r in ranks):
        raise ValueError('Incomplete/changed encoder qualification')
    entries={}
    for update in (1,8):
        for rank in range(8):
            path=root/f'optimizer_gradients/update_{update:06d}/rank_{rank}/manifest.json'
            if not path.exists() and update==1 and 'resume' in root.name:
                continue
            d=json.loads(path.read_text());rows=d['parameters']
            if len(rows)!=359 or d['frozen_with_gradient'] or any(not r['present'] or not r['finite'] or r['dtype']!='torch.float32' or r['nonzero']==0 for r in rows):
                raise ValueError('All 359 actual FP32 optimizer gradients must be valid')
            entries[f'{update}:{rank}']=dict(tensors=len(rows),parameters=sum(r['numel'] for r in rows),
                                           norm=float(np.sqrt(sum(r['l2']**2 for r in rows))))
    times=[json.loads(x)['seconds'] for x in (root/'training_rank0.jsonl').read_text().splitlines()]
    return dict(identity=ranks[0]['identity_sha256'],frozen_prefix_sha256=ranks[0]['frozen_prefix_sha256_before'],
                optimizer_gradient_checks=entries,update_seconds_median=float(np.median(times)),
                update_seconds=times,checkpoint=ranks[0]['checkpoint'],
                export_sha256=json.loads((root/'export_update000008/export_state.json').read_text())['files']['pytorch_model.pt'])


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--full',required=True);p.add_argument('--resumed',required=True)
    p.add_argument('--nft');p.add_argument('--sft-weight',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(4)
    full,resume=check(a.full),check(a.resumed)
    left,right=[Path(r['checkpoint']) for r in (full,resume)]
    # Both atomic publishers already read every byte to create these seals.
    # Exact equal file hashes are stronger than another elementwise mmap read,
    # which incurs severe small-page IO on the shared filesystem.
    left_seal,right_seal=[json.loads((p/'checkpoint_files.json').read_text()) for p in (left,right)]
    comparisons={}
    for rank in range(8):
        name=f'pytorch_model/bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt'
        if left_seal[name]==right_seal[name] and left_seal[f'rank_{rank}.pt']==right_seal[f'rank_{rank}.pt']:
            comparisons[f'optimizer_rank{rank}']=True
            comparisons[f'RNG_data_cursor_rank{rank}']=True
            continue
        l=torch.load(left/name,mmap=True,map_location='cpu',weights_only=False)['optimizer_state_dict']
        r=torch.load(right/name,mmap=True,map_location='cpu',weights_only=False)['optimizer_state_dict']
        comparisons[f'optimizer_rank{rank}']=all(equal(l[k],r[k]) for k in ('base_optimizer_state','single_partition_of_fp32_groups','group_paddings','partition_count'))
        del l,r
        l,r=[torch.load(path/f'rank_{rank}.pt',weights_only=False,map_location='cpu') for path in (left,right)]
        comparisons[f'RNG_data_cursor_rank{rank}']=equal(l,r)
        del l,r
    if left_seal['pytorch_model/mp_rank_00_model_states.pt']==right_seal['pytorch_model/mp_rank_00_model_states.pt']:
        comparisons['forward_parameters']=comparisons['scheduler']=True
    else:
        l,r=[torch.load(path/'pytorch_model/mp_rank_00_model_states.pt',weights_only=False,map_location='cpu') for path in (left,right)]
        comparisons['forward_parameters']=equal(l['module'],r['module'])
        comparisons['scheduler']=equal(l['lr_scheduler'],r['lr_scheduler'])
        del l,r
    if not all(comparisons.values()):
        raise ValueError('Exact resumed optimizer/model/RNG/cursor/scheduler differs: '+str(comparisons))
    with torch.serialization.safe_globals([set]):
        original=torch.load(a.sft_weight,weights_only=True,map_location='cpu')
    original=original.get('module',original)
    forward=torch.load(Path(a.full)/'export_update000008/pytorch_model.pt',weights_only=True,map_location='cpu')
    updated=[];unchanged=[]
    for name,value in forward.items():
        if name.startswith('action_model.'):
            (updated if not torch.equal(value,original[name].to(value.dtype)) else unchanged).append(name)
    result=dict(status='PASS',qualification_not_method_result=True,full_FM=full,resumed_FM=resume,
                exact_resume=comparisons,forward_action_tensors_changed=len(updated),unchanged_action_tensors=unchanged,
                equality_proof='matching_complete_immutable_checkpoint_publication_SHA256_or_exact_tensor_fallback',
                original_SFT_sha256=file_sha(a.sft_weight),source_sha256=file_sha(__file__))
    if a.nft:
        result['NFT']=check(a.nft)
    path=Path(a.output)
    if path.exists():raise FileExistsError(path)
    atomic_json(path,result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
