"""Bounded on-policy collection and state adapters for the existing ZeRO trainer."""
import copy
import math
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
import numpy as np
from starVLA.rl.flow_grpo import distributed as control
from starVLA.rl.flow_grpo.data import keys_for_positions
from starVLA.rl.flow_grpo.stability import lr_multiplier
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.policy_adapter import decode_request
from .contracts import fingerprint, seed_for, COSTS
from .component_advantages import component_advantages
from .effective_group_collector import EffectiveGroupCollector
from .safety_costs import costs_from_record


def cpu_tree(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu()
    if isinstance(value,dict): return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu_tree(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu_tree(v) for v in value)
    return value


def device_tree(value,device):
    if isinstance(value,torch.Tensor): return value.to(device)
    if isinstance(value,dict): return {k:device_tree(v,device) for k,v in value.items()}
    return value


class OrderedScenes:
    """Same original epoch permutations; bounded memoization only, no hidden cursor."""
    def __init__(self,tokens,seed):
        self.tokens=list(tokens);self.seed=seed;self.epoch=None;self.permutation=None
    def at(self,position):
        epoch,index=divmod(position,len(self.tokens))
        if epoch!=self.epoch:
            self.permutation=torch.randperm(len(self.tokens),generator=torch.Generator().manual_seed(self.seed+epoch)).tolist()
            self.epoch=epoch
        return self.tokens[self.permutation[index]]


def exposure_lr(raw_after_collection,total):
    # Same cosine implementation, with RAW scenes as its clock. The LR is
    # locked once per collection and remains identical in both inner epochs.
    spec=dict(type='warmup_cosine',total_updates=total,warmup_updates=max(1,min(total-1,math.ceil(total*.03))),min_lr_ratio=.1)
    return 1e-6*lr_multiplier(raw_after_collection-1,spec)


def gather_rows(local):
    values=[None]*dist.get_world_size()
    dist.all_gather_object(values,local,group=control._CONTROL_GROUP)
    result=sorted([entry for rows in values for entry in rows],key=lambda x:x[0])
    if len({i for i,_ in result})!=len(result): raise ValueError('Duplicate distributed scene groups')
    return result


@torch.no_grad()
def collect_batch(behavior,reference,store,service,stream,pop,collector,raw_start,count,calibration,rank,world):
    offset=len(collector.rows); local=[]
    start=time.monotonic()
    for i in range(offset,offset+count):
        if i%world!=rank: continue
        position=raw_start+i-offset;token=stream.at(position)
        raw=store.get(token)
        if raw is None: raise FileNotFoundError('Missing sealed TRAIN observation: '+token)
        device=next(behavior.parameters()).device
        t0=time.monotonic()
        chain=behavior.sample_chain(raw.to(device),[token],[position],version=collector.behavior_identity)
        torch.cuda.synchronize(); rollout_seconds=time.monotonic()-t0
        request=decode_request(chain['chain'][0,:,-1].float().cpu().numpy(),act_norm=1)
        t0=time.monotonic()
        records=service.score([(token,pop['token_logs'][token],request)])[0]
        scoring_seconds=time.monotonic()-t0
        rewards=torch.tensor([[r['metrics']['score'] for r in records]],dtype=torch.float64)
        costs=torch.from_numpy(np.stack([costs_from_record(r,calibration['thresholds']) for r in records]))[None]
        advantage,statistics=component_advantages(rewards,costs,collector.lambda_snapshot,calibration['component_floors'])
        t0=time.monotonic()
        ref=None
        if i<16:
            values=reference.saved_statistics(chain['raw'],chain['chain'])
            ref={k:values[k] for k in ('mean','std')}
        torch.cuda.synchronize(); reference_seconds=time.monotonic()-t0
        row=dict(token=token,position=position,scene_key=f'{collector.behavior_identity["collection"]}:{position}:{token}',
                 chain=chain,records=records,requests=request,costs=costs[0],rewards=rewards[0],advantage=advantage[0],
                 advantage_statistics=statistics,reference=ref,
                 rollout_seconds=rollout_seconds,scoring_seconds=scoring_seconds,reference_seconds=reference_seconds,
                 scoring_cpu_seconds=records[0]['scene_scoring_cpu_seconds'])
        local.append((i,cpu_tree(row)))
    return local


def update_dual(dual,collector,collection,device):
    # First16 only; explicit sum/count reduction counts each physical candidate
    # exactly once according to its deterministic owner, including zero-A groups.
    rank,world=dist.get_rank(),dist.get_world_size()
    first=collector.first_costs()
    selected=first[rank::world]
    sums=selected.double().sum((0,1)).to(device)
    count=torch.tensor(selected.shape[0]*16,dtype=torch.float64,device=device)
    dist.all_reduce(sums);dist.all_reduce(count)
    mean=sums/count
    if not torch.allclose(mean.cpu(),first.double().mean((0,1)),atol=1e-14,rtol=1e-14):
        raise ValueError('Distributed dual count mismatch')
    def update():
        before=dual.snapshot().tolist()
        measured=dual.update(first,collection=collection)
        report=dict(cost_means=measured.tolist(),before=before,after=dual.snapshot().tolist(),
                    budgets=list(dual.budgets),collection=collection)
        return report,dual.state_dict()
    report,state=control.rank0_result(update,device)
    if rank!=0: dual.load_state_dict(state)
    report['candidate_count']=int(count);report['scope']='unfiltered_first16_or_budget_remainder'
    return report


def micro_inputs(collector,slot,store,replay_labels,replay_stream,replay_position,device):
    selected=collector.accepted; first_count=min(16,len(collector.rows))
    pg_index=selected[slot] if slot<len(selected) else (selected[0] if selected else 0)
    ref_index=slot if slot<first_count else 0
    def view(index,include_reference=False):
        row=collector.rows[index]
        value={k:row['chain'][k] for k in ('raw','chain','old_elementwise_logp')}
        value.update(advantage=row['advantage'][None],scene_key=row['scene_key'])
        if include_reference:value['reference']=row['reference']
        return device_tree(value,device)
    token=replay_stream.at(replay_position)
    raw=store.get(token)
    if raw is None:raise FileNotFoundError('Missing uniform replay condition: '+token)
    return dict(pg=view(pg_index),retention=view(ref_index,True),
                replay={'raw':raw.to(device),'target':replay_labels.get(token)[None].to(device)},
                pg_factor=(16/max(1,len(selected))) if slot<len(selected) else 0.,
                retention_factor=16/first_count if slot<first_count else 0.,
                replay_seed=seed_for('cfn_GT_replay_FM',42,token,replay_position))


def archive_collection(root,collector,index):
    path=Path(root)/'behavior_collections'/f'{index:06d}.pt'
    if path.exists():raise FileExistsError('Cannot overwrite old behavior evidence')
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.incomplete')
    torch.save(collector.state_dict(),temp);os.replace(temp,path)
    from .contracts import file_sha
    atomic_json(path.with_suffix('.json'),dict(sha256=file_sha(path),behavior=collector.behavior_identity,
                raw_groups=len(collector.rows),accepted_groups=len(collector.accepted),candidates=len(collector.rows)*16))
    return str(path)
