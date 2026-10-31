"""Sparse component-gradient observation inside a loss, before DDP post-forward.

These are actual gradients of each source-reduced objective, manually averaged
over ranks. autograd.grad never populates .grad; the ordinary DDP backward and
Adam update still run exactly once. This observer must run *inside* forward,
before DDP prepares that forward's backward. No hooks are installed on tensors.
"""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from starVLA.rl.simwam_recipe import checkpoint

SCHEMA = 'world_sparse_global_component_gradient_probe_v1'


def parameter_scope(name):
    if name.startswith('scene_provider.'):
        return 'scene'
    if name.startswith('predictive_provider.'):
        return 'predictive'
    return 'action'


def optimizer_update(model):
    optimizer = model._loss.optimizer
    first = optimizer.param_groups[0]['params'][0]
    step = optimizer.state.get(first, {}).get('step', 0)
    return int(step.item()) if torch.is_tensor(step) else int(step)


def _cosine(left, right):
    a, b = float(left.norm()), float(right.norm())
    return float(torch.dot(left.double(), right.double())/(a*b)) if a*b else None


def measure_components(model, components):
    """Return JSON-serializable local and true equal-rank-mean measurements."""
    named = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    params = [p for _,p in named]
    if not params or any(p.grad is not None for p in params):
        raise ValueError('Probe requires trainables with empty .grad before DDP backward')
    labels = [parameter_scope(n) for n,_ in named]
    offsets, begin = [], 0
    for name,p in named:
        offsets.append((begin,begin+p.numel(),parameter_scope(name)))
        begin += p.numel()
    slices = {scope:[(lo,hi) for lo,hi,s in offsets if s in members]
              for scope,members in {'action':{'action'},'scene':{'scene'},'predictive':{'predictive'},
                                    'memory':{'scene','predictive'},'all':set(labels)}.items()
              if any(s in members for s in labels)}
    def selected(v, ranges):
        if len(ranges)==1:
            return v[ranges[0][0]:ranges[0][1]]
        if ranges[-1][1]-ranges[0][0] == sum(hi-lo for lo,hi in ranges):
            return v[ranges[0][0]:ranges[-1][1]]
        return torch.cat([v[lo:hi] for lo,hi in ranges])
    vectors = {}
    error = None
    try:
        for name,loss in sorted(components.items()):
            if loss.ndim or not loss.requires_grad:
                raise ValueError('Probe components must be live differentiable scalar objectives: '+name)
            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            vectors[name] = torch.cat([(torch.zeros_like(p) if g is None else g.detach()).reshape(-1).float()
                                       for p,g in zip(params,grads)])
            if not bool(torch.isfinite(vectors[name]).all()):
                raise FloatingPointError('Nonfinite component gradient: '+name)
        if any(p.grad is not None for p in params):
            raise ValueError('Probe unexpectedly populated parameter .grad')
    except Exception as exc:
        error = repr(exc)
    checkpoint.check_errors(error)
    local = {name:{scope:float(selected(v,ranges).norm()) for scope,ranges in slices.items()}
             for name,v in vectors.items()}
    local_by_rank = checkpoint.gather(local)
    world = dist.get_world_size() if dist.is_initialized() else 1
    for v in vectors.values():
        if dist.is_initialized():
            dist.all_reduce(v, op=dist.ReduceOp.SUM)
            v.div_(world)
    scopes = {}
    for scope,ranges in slices.items():
        parts = {name:selected(v,ranges) for name,v in vectors.items()}
        norms = {name:float(v.norm()) for name,v in parts.items()}
        mean_local = {name:sum(x[name][scope] for x in local_by_rank)/world for name in parts}
        scopes[scope] = dict(
            numel=sum(hi-lo for lo,hi in ranges), global_mean_gradient_norm=norms,
            mean_local_gradient_norm=mean_local,
            rank_alignment_ratio={name:norms[name]/mean_local[name] if mean_local[name] else None for name in parts},
            cosine={a+'__'+b:_cosine(parts[a],parts[b]) for a,b in itertools.combinations(parts,2)},
            summed_gradient_norm=float(torch.stack(list(parts.values())).sum(0).norm()))
    return dict(schema=SCHEMA,world=world,parameter_numel=begin,
                source_reduction='component source-local effective reduction then exact equal rank mean',
                local_gradient_norms_by_rank=local_by_rank,scopes=scopes,
                parameter_grad_untouched=True,optimizer_steps_added=0)


def maybe_probe_components(model, rollout, cfg, components):
    """Optionally write one sparse record and return scalar tensor metrics."""
    spec = cfg.get('research',{}).get('gradient_probe')
    if not spec:
        return {}
    if spec.get('schema') != SCHEMA:
        raise ValueError('Unknown sparse gradient probe protocol')
    behavior = rollout['behavior_version']
    update = int(behavior['update'])
    if update not in spec['collection_start_updates'] or optimizer_update(model) != update:
        return {}
    start = time.monotonic()
    record = measure_components(model,components)
    record.update(update=update,inner=0,behavior=behavior,at_unix=time.time(),seconds=time.monotonic()-start,
                  token_by_rank=checkpoint.gather(rollout['token']),
                  structured_parts_by_rank=checkpoint.gather({k:rollout['structured_advantage'][k].detach().cpu().tolist()
                      for k in ('feasible','A_safe','A_EP','A_raw')}))
    rank = dist.get_rank() if dist.is_initialized() else 0
    error = None
    if rank == 0:
        try:
            path=Path(spec['output'])/f'update{update:06}_inner0.json'
            path.parent.mkdir(parents=True,exist_ok=True)
            tmp=path.with_name(path.name+'.writing-'+str(os.getpid()))
            tmp.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
            os.replace(tmp,path)
        except Exception as exc:
            error=repr(exc)
    checkpoint.check_errors(error)
    return {'gradient_probe_seconds':torch.tensor(record['seconds'],device=model.device)}
