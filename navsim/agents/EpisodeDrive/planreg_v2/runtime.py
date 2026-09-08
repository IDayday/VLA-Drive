"""Deterministic padded exposure, complete RNG resume, explicit deployment/benchmark contracts."""
import math
import random
import numpy as np
import torch
from torch.utils.data import Sampler


def load_config(path):
    from pathlib import Path
    from omegaconf import OmegaConf
    visited=set()
    def unresolved(location):
        location=Path(location).resolve()
        if location in visited:raise ValueError('Cyclic V2 config inheritance')
        visited.add(location)
        child=OmegaConf.load(location)
        parent=child.pop('inherits',None)
        return OmegaConf.merge(unresolved(location.parent/parent),child) if parent else child
    return OmegaConf.to_container(unresolved(path),resolve=True)


class ExactExposureSampler(Sampler):
    def __init__(self,size,global_batch,world,rank,seed,epoch=0,start_step=0):
        if size <= 0 or global_batch % world: raise ValueError('Invalid exact exposure layout')
        self.steps = math.ceil(size/global_batch)
        generator = torch.Generator().manual_seed(seed+epoch)
        order = torch.randperm(size,generator=generator).tolist()
        padded = (order*math.ceil(self.steps*global_batch/size))[:self.steps*global_batch]
        self.indices = padded[start_step*global_batch:][rank::world]
        self.padding_count = self.steps*global_batch-size

    def __iter__(self): return iter(self.indices)
    def __len__(self): return len(self.indices)


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda']: torch.cuda.set_rng_state_all(state['cuda'])


def accumulated_batches(iterator,accumulate):
    """Count the whole optimizer batch before micro-forward; missing targets must not bias accumulation."""
    from itertools import islice
    import torch.distributed as dist
    while True:
        group=list(islice(iterator,accumulate))
        if not group:return
        if len(group)!=accumulate:raise ValueError('Sampler did not pad to a complete accumulated optimizer batch')
        counts=torch.zeros(8,dtype=torch.float32)
        for _,target in group:
            times,valid=target['motion_timestamps'],target['motion_valid']
            cover=[];left=0.
            for right in (.5,1.5,4.):
                interval=(times>(left+.02 if left else 0.))&(times<=right+.02)
                cover.append((((times-right).abs()<=.02)&valid).any(-1)&(~interval|valid).all(-1))
                left=right
            actions=torch.stack(cover,-1).long().cumprod(-1).bool()
            future=target['future_valid_mask']
            ro=future&actions
            tf=ro.clone();tf[:,1:]&=future[:,:-1].long().cumprod(-1).bool()
            counts[:3]+=tf.sum(0);counts[3:6]+=ro.sum(0)
            counts[6]+=target['trajectory_valid'].any(-1).sum()
            counts[7]+=target['trajectory_long_valid'].sum()
        if dist.is_available() and dist.is_initialized():
            if dist.get_backend()=='nccl':counts=counts.cuda()
            dist.all_reduce(counts)
        context=dict(tf=counts[:3],ro=counts[3:6],trajectory=counts[6],long=counts[7],accumulate=accumulate)
        for features,target in group:yield features,target,context


def validate_formal(config,manifest,layout):
    import json
    from pathlib import Path
    if config.get('world_model_enabled') is not True and config.get('variant') != 'no_wm':
        raise ValueError('Formal Base/VQA both require world model; no-WM must be an explicitly labeled control')
    if manifest['split'] != 'trainval_final_fit' or len(manifest['records']) != 103288 or manifest.get('smoke'):
        raise ValueError('Formal run requires the complete 103,288-scene final-fit manifest')
    if config['epochs'] != 27 or config['checkpoint_path'] is not None:
        raise ValueError('Formal endpoint is 27 epochs with no M0 initialization')
    if layout.get('architecture_version') != config['architecture_version'] or not layout.get('passed'):
        raise ValueError('V2 must be profiled; V1 throughput evidence cannot authorize this layout')
    if layout['global_batch'] != config['global_batch']:
        raise ValueError('Layout global batch mismatch')
    if not config.get('shared_init_path'): raise ValueError('Shared V2 trainable initialization required')
    stats=json.loads(Path(config['normalizer_path']).read_text())['metadata']
    if (stats['count'],stats['split'],stats['token_sha256']) != (
            103288,'trainval_final_fit',manifest['token_sha256']):
        raise ValueError('Formal normalizer must be measured on this exact unique final-fit token manifest, not smoke/held-out data')


def same_batch_gradient_audit(plan_loss,wm_loss,named_parameters,weight):
    selected = [(n,p) for n,p in named_parameters if p.requires_grad and
                ('.vision_model.' in n or '.planning_register_adapter.' in n)]
    parameters = [p for _,p in selected]
    plan = torch.autograd.grad(plan_loss,parameters,retain_graph=True,allow_unused=True)
    wm = torch.autograd.grad(wm_loss,parameters,retain_graph=True,allow_unused=True)
    results = {}
    for group,match in (('vision_lora',lambda n:'.vision_model.' in n),('registers',lambda n:'.planning_register_adapter.' in n)):
        aa=bb=ab=0.
        for (name,p),g,h in zip(selected,plan,wm):
            if not match(name): continue
            if g is not None: aa += float(g.detach().float().square().sum())
            if h is not None: bb += float(h.detach().float().square().sum())*weight*weight
            if g is not None and h is not None: ab += float((g.detach().float()*h.detach().float()).sum())*weight
        results[group] = dict(plan_norm=math.sqrt(aa),weighted_wm_norm=math.sqrt(bb),
                              cosine=ab/max(1e-30,math.sqrt(aa*bb)))
    return results
