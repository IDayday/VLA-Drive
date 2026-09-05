"""Three physical answers, not latent matching; globally normalized DDP masks."""
import torch
import torch.distributed as dist
import torch.nn.functional as F
from contextlib import contextmanager
from contextvars import ContextVar

_REDUCE_DDP = ContextVar('lite_reduce_ddp', default=True)


@contextmanager
def rank_local_physical_diagnostics():
    token = _REDUCE_DDP.set(False)
    try:
        yield
    finally:
        _REDUCE_DDP.reset(token)

HINDSIGHT_BINS = (0, 2, 5)
HORIZONS = (.5, 1.5, 3.)


def global_task_mean(element_losses, valid):
    """Exact global values and DDP-average gradients, ignoring unavailable tasks.

    A single packed reduction collects three counts and detached numerators.
    The differentiable local numerator is scaled by world size because DDP
    subsequently averages parameter gradients. Empty ranks still participate.
    """
    valid = valid.to(device=element_losses.device, dtype=torch.bool)
    values = torch.where(valid, element_losses, torch.zeros_like(element_losses))
    dims = tuple(range(values.ndim-1))
    numerator = values.sum(dims)
    counts = valid.sum(dims).to(numerator.dtype)
    packed = torch.stack((numerator.detach(),counts))
    world_size=1
    if _REDUCE_DDP.get() and dist.is_available() and dist.is_initialized():
        world_size=dist.get_world_size()
        dist.all_reduce(packed)
    active=packed[1]>0
    denominator=packed[1].clamp_min(1)
    local=numerator*world_size/denominator
    means=local+(packed[0]/denominator-local.detach())
    return (means*active).sum()/active.sum().clamp_min(1), means, packed[1]


def physical_element_losses(predictions, physical_values, gap_class):
    raw=torch.nan_to_num(physical_values.detach().float())
    road=raw[...,1].clamp(-2,2)/2
    progress=raw[...,2]/40
    logits=predictions['gap_logits'].float()
    gap=F.cross_entropy(logits.reshape(-1,5),gap_class.detach().long().reshape(-1),reduction='none').reshape(road.shape)
    return torch.stack((gap,
        F.smooth_l1_loss(predictions['road_margin'].float(),road,reduction='none'),
        F.smooth_l1_loss(predictions['route_progress'].float(),progress,reduction='none')), -1)


def distillation_element_losses(current, teacher):
    # Teacher probabilities and continuous outputs are targets, NEVER gradients.
    p=teacher['gap_logits'].detach().float().softmax(-1)
    gap=F.kl_div(current['gap_logits'].float().log_softmax(-1),p,reduction='none').sum(-1)
    return torch.stack((gap,
        F.smooth_l1_loss(current['road_margin'].float(),teacher['road_margin'].detach().float(),reduction='none'),
        F.smooth_l1_loss(current['route_progress'].float(),teacher['route_progress'].detach().float(),reduction='none')),-1)


def task_future_lite_loss(decoder, current_registers, trajectories, status,
                          target_current, target_future, future_valid, logged_future_poses,
                          physical_values, gap_class, physical_valid):
    current=decoder(current_registers,trajectories,status)
    current_loss, current_tasks, current_counts=global_task_mean(
        physical_element_losses(current,physical_values,gap_class),physical_valid)
    future_outputs=[]
    for h,time in enumerate(HORIZONS):
        # Poses of invalid frames are sanitized; their losses are excluded.
        pose=torch.where(future_valid[:,h,None],logged_future_poses[:,h],torch.zeros_like(logged_future_poses[:,h]))
        out=decoder.forward_hindsight(target_current,target_future[:,h],trajectories,status,pose,time)
        future_outputs.append({key:value[:,:,HINDSIGHT_BINS[h]] for key,value in out.items()})
    teacher={key:torch.stack([out[key] for out in future_outputs],2) for key in current}
    subset={key:value[:,:,HINDSIGHT_BINS] for key,value in current.items()}
    future_mask=physical_valid[:,:,HINDSIGHT_BINS] & future_valid[:,None,:,None]
    future_loss, future_tasks, future_counts=global_task_mean(
        physical_element_losses(teacher,physical_values[:,:,HINDSIGHT_BINS],gap_class[:,:,HINDSIGHT_BINS]),future_mask)
    kd_loss, kd_tasks, _=global_task_mean(distillation_element_losses(subset,teacher),future_mask)
    result=dict(wm_loss=current_loss+.5*future_loss+.25*kd_loss,
                physical_current_loss=current_loss, physical_future_loss=future_loss,
                task_distill_loss=kd_loss, legacy_future_register_loss=current_loss.new_zeros(()))
    for i,name in enumerate(('gap','road','progress')):
        result.update({f'physical_current_{name}':current_tasks[i],f'physical_future_{name}':future_tasks[i],
                       f'task_distill_{name}':kd_tasks[i],f'physical_current_{name}_count':current_counts[i],
                       f'physical_future_{name}_count':future_counts[i]})
    return result


def sample_training_candidates(gt, proposals, count=8):
    if proposals.shape[1:] != (64,8,3) or count != 8:
        raise ValueError('Lite training requires GT + 7 of the unchanged 64 proposals')
    indices=torch.stack([torch.randperm(64,device=proposals.device)[:7] for _ in range(len(proposals))])
    chosen=proposals.detach()[torch.arange(len(proposals),device=proposals.device)[:,None],indices]
    return torch.cat((gt.detach()[:,None],chosen),1), indices
