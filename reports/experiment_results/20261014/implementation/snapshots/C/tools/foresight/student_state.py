"""Independent task RNG and optimizer-batch denominators, including tail batches."""
import random
import numpy as np
import torch
from torch import distributed as dist
from starVLA.model.modules.foresight.losses import select_horizons, request_future_horizons


def validate_rank_batches(size, global_batch, world, micro_batch):
    """Allow unequal final rank sizes only with matching backward call counts.

    E.g. NAVSIM101592, batch32, world16 has a24-scene tail: eight ranks
    receive2 scenes and eight receive1. With microbatch2 everyone executes one
    backward, normalized by the true global24. No padding/duplication/drop.
    """
    import math
    if min(size, global_batch, world, micro_batch) < 1 or global_batch % world:
        raise ValueError('Invalid distributed batch contract')
    for batch_size in {min(size, global_batch), size % global_batch or global_batch}:
        counts = [len(range(rank, batch_size, world)) for rank in range(world)]
        if min(counts) < 1:
            raise ValueError('Empty rank tail is unsupported; do not drop or duplicate scenes')
        if len({math.ceil(n / micro_batch) for n in counts}) != 1:
            raise ValueError('Unequal backward counts; increase common microbatch to cover tail')


def capture_rng(generators):
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state(),'generators':{k:g.get_state() for k,g in generators.items()}}


def restore_rng(state,generators):
    if set(state['generators'])!=set(generators):raise ValueError('Task RNG inventory changed')
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state(state['cuda'].cpu())
    for key,generator in generators.items():generator.set_state(state['generators'][key].cpu())


def optimizer_batch_counts(targets,horizon_generator,device):
    """Select LOSS horizons before microbatch slicing, never alter current inputs.

    All ranks SUM the entire optimizer batch denominator exactly once. The model
    contributes world_size * local_numerator / global_denominator, compensating
    the optimizer's gradient average. No further microbatch averaging is valid.
    """
    counts={'ego_scenes':len(targets['ego'])}
    for task in ('current_dino','future_dino'):
        if task not in targets: continue
        values, valid = targets[task], targets[task+'_valid']
        if valid.dtype!=torch.bool:raise ValueError('DINO spatial validity must be boolean')
        if task=='future_dino':
            if values.ndim!=6 or values.shape[1:3]!=(3,3):raise ValueError('DINO future layout')
            selected=request_future_horizons(len(values),horizon_generator)
            targets['dino_horizon']=selected
            values=values[torch.arange(len(values)),selected];valid=valid[torch.arange(len(valid)),selected]
        if values.ndim!=5 or valid.shape!=values.shape[:2]+values.shape[-2:]:raise ValueError('DINO grid/mask layout')
        counts[task]=int(valid.sum())*values.shape[2]
    if 'future_latent' in targets:
        values,valid=targets['future_latent'],targets['future_valid']
        if values.ndim!=6 or values.shape[:3]!=valid.shape:raise ValueError('Future batch layout')
        selected=select_horizons(valid,horizon_generator)
        targets['visual_horizon']=selected
        counts['visual']=int(valid[torch.arange(len(valid)),selected].sum())*int(np.prod(values.shape[3:]))
    if 'interaction_latent' in targets:
        values,valid=targets['interaction_latent'],targets['interaction_valid']
        if values.shape[1:]!=(8,512) or valid.shape!=(len(values),) or valid.dtype!=torch.bool:
            raise ValueError('Interaction batch layout')
        counts['interaction']=int(valid.sum())*8*512
    names=sorted(counts);packed=torch.tensor([counts[k] for k in names],device=device,dtype=torch.float64)
    if dist.is_initialized():dist.all_reduce(packed)
    return dict(zip(names,packed.tolist()))


def learning_rate(completed,base,minimum,warmup,horizon):
    import math
    if not 0<=completed<horizon or not 0<=warmup<horizon or not 0<minimum<=base:raise ValueError('Invalid fixed schedule')
    if completed<warmup:return base*(completed+1)/warmup
    return minimum+(base-minimum)*.5*(1+math.cos(math.pi*(completed-warmup)/(horizon-warmup)))
