"""Per-scene SafetyEP; normalize raw advantages across the fresh collection."""
import torch
import torch.distributed as dist

FACTORS = ('no_at_fault_collisions', 'drivable_area_compliance', 'time_to_collision_within_bound',
           'ego_progress', 'comfort', 'driving_direction_compliance')


@torch.no_grad()
def raw_credit(factors, *, endpoint=None, stage2=False):
    values = factors.detach().double()
    if values.ndim != 3 or values.shape[-1] != 6 or not torch.isfinite(values).all():
        raise ValueError('Expected finite [scene,G,6] official factors')
    if not ((values>=0)&(values<=1)).all():
        raise ValueError('Native factors outside [0,1]')
    feasible = (values[..., :3] == 1).all(-1)
    count = feasible.sum(1, keepdim=True)
    has = count > 0
    safe = feasible.double()-feasible.double().mean(1, keepdim=True)
    cost = (1-values[..., :3]).mean(-1)
    cost_offsets=cost-cost[:,:1]
    safe = torch.where(has, safe, cost_offsets.mean(1, keepdim=True)-cost_offsets)
    quality = values[..., 3]
    if stage2:
        if endpoint is None or endpoint.shape != quality.shape:
            raise ValueError('StageII requires explicit TRAIN endpoint quality')
        quality = .70*quality+.15*values[..., 4]+.15*endpoint.detach().double()
    first=quality.gather(1,feasible.long().argmax(1,keepdim=True))
    offsets=quality-first
    average = (offsets*feasible).sum(1, keepdim=True)/count.clamp_min(1)
    # Never assign quality credit to an unsafe candidate.
    improvement = torch.where(feasible, offsets-average, torch.zeros_like(quality))
    return safe+improvement, feasible


@torch.no_grad()
def normalize(raw, *, all_raw=None):
    if all_raw is None:
        if dist.is_initialized():
            parts=[torch.empty_like(raw) for _ in range(dist.get_world_size())]
            dist.all_gather(parts, raw.double()); all_raw=torch.cat(parts)
        else: all_raw=raw
    std=all_raw.double().std(unbiased=False)
    result=(raw.double()/(std+1e-8)).clamp(-5,5).float()
    if not torch.isfinite(result).all(): raise FloatingPointError('Nonfinite SafetyEP')
    return result, float(std)
